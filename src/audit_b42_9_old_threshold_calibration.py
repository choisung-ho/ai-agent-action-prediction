#!/usr/bin/env python3
"""Leakage-controlled F1-aware decision-threshold audit for B42_9.

The public-best B42_9 path is kept fixed as the baseline: old portable no-E5
GRU plus the raw seed-42 selector in a source-specific geometric product
(``SIM=.20``, ``AU=.85``).  This script tests only a small, portable vector of
class logit offsets.  It never writes ``model/``, ``script.py``, or an archive.

Two protocols are intentionally required before any recommendation:

* Five whole-session outer folds.  The offset capacity is selected and fitted
  without labels from the held session fold.
* A matching seed-42 strict common split.  Both the selector and the old
  no-E5 GRU are trained/cross-fitted using common-train sessions only; fixed
  validation labels are read solely for the final score.

The existing B42_9 word15 and current char20 terminal overrides are treated
as downstream, immutable rules.  They are never features of the offset fit.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.model_selection import GroupKFold


CLASSES = (
    "read_file", "grep_search", "list_directory", "glob_pattern",
    "edit_file", "write_file", "apply_patch", "run_bash", "run_tests",
    "lint_or_typecheck", "ask_user", "plan_task", "web_search", "respond_only",
)
N_CLASSES = len(CLASSES)
EPS = 1.0e-12


@dataclass(frozen=True)
class OffsetSpec:
    """Small, finite direct-macro-F1 coordinate-search capacity."""

    cap: float
    step: float
    rounds: int

    @property
    def key(self) -> str:
        if self.cap == 0.0:
            return "none"
        return f"direct_f1_bias_cap{self.cap:.2f}_step{self.step:.2f}_rounds{self.rounds}"

    @property
    def complexity(self) -> tuple[int, float, int]:
        # Prefer no intervention, then a smaller bounded offset, on ties.
        return (0 if self.cap == 0.0 else 1, self.cap, self.rounds)


SPECS = (
    OffsetSpec(0.00, 0.00, 0),
    OffsetSpec(0.04, 0.02, 1),
    OffsetSpec(0.08, 0.02, 1),
    OffsetSpec(0.12, 0.03, 1),
    OffsetSpec(0.08, 0.02, 2),
)


def session_of(sample_id: str) -> str:
    return sample_id.rsplit("-step_", 1)[0]


def source_of(sample_id: str) -> str:
    return "au" if sample_id.startswith("sess_au_") else "sim"


def terminal_mask(ids: np.ndarray) -> np.ndarray:
    maximum: dict[str, int] = {}
    parsed: list[tuple[str, int]] = []
    for sample_id in ids.astype(str):
        session, raw_step = sample_id.rsplit("-step_", 1)
        step = int(raw_step)
        maximum[session] = max(maximum.get(session, -1), step)
        parsed.append((session, step))
    return np.asarray([step == maximum[session] for session, step in parsed], dtype=bool)


def macro_from_prediction(y: np.ndarray, prediction: np.ndarray, mask: np.ndarray | None = None) -> float:
    if mask is None:
        mask = np.ones(len(y), dtype=bool)
    y = y[mask].astype(np.int64, copy=False)
    prediction = prediction[mask].astype(np.int64, copy=False)
    matrix = np.bincount(N_CLASSES * y + prediction, minlength=N_CLASSES * N_CLASSES).reshape(N_CLASSES, N_CLASSES)
    tp = np.diag(matrix).astype(np.float64)
    fp = matrix.sum(axis=0) - tp
    fn = matrix.sum(axis=1) - tp
    denominator = 2.0 * tp + fp + fn
    value = np.divide(2.0 * tp, denominator, out=np.zeros(N_CLASSES, dtype=np.float64), where=denominator > 0)
    return float(value.mean())


def geometric_logits(xgb: np.ndarray, gru: np.ndarray, source: np.ndarray) -> np.ndarray:
    alpha = np.where(source == "au", 0.85, 0.20).astype(np.float64)[:, None]
    return (alpha * np.log(np.maximum(xgb, EPS)) + (1.0 - alpha) * np.log(np.maximum(gru, EPS))).astype(np.float64)


def prediction(logits: np.ndarray, bias: np.ndarray) -> np.ndarray:
    return (logits + bias[None, :]).argmax(axis=1).astype(np.int16)


def fit_direct_f1_bias(logits: np.ndarray, y: np.ndarray, spec: OffsetSpec) -> np.ndarray:
    """Fit a compact class-threshold vector directly against macro-F1.

    This deliberately optimizes hard decisions rather than NLL.  The search
    is bounded and deterministic, and the all-zero no-intervention policy is
    always included in outer and common-train selection.
    """
    bias = np.zeros(N_CLASSES, dtype=np.float64)
    if spec.cap == 0.0:
        return bias
    values = np.arange(-spec.cap, spec.cap + spec.step * 0.5, spec.step, dtype=np.float64)
    for _ in range(spec.rounds):
        for class_id in range(N_CLASSES):
            best_value = float(bias[class_id])
            best_score = -1.0
            for value in values:
                candidate = bias.copy()
                candidate[class_id] = value
                score = macro_from_prediction(y, prediction(logits, candidate))
                # Smaller absolute shifts win exact score ties, preventing
                # gratuitous decision changes in a deployable artifact.
                if score > best_score + 1.0e-14 or (
                    abs(score - best_score) <= 1.0e-14 and abs(value) < abs(best_value) - 1.0e-14
                ):
                    best_score = score
                    best_value = float(value)
            bias[class_id] = best_value
    # A shared logit constant changes no argmax, but centering makes the saved
    # portable vector identifiable and easier to inspect.
    return bias - bias.mean()


def select_for_outer(
    logits: np.ndarray, y: np.ndarray, terminal: np.ndarray, fold: np.ndarray,
) -> tuple[np.ndarray, list[dict[str, Any]], list[np.ndarray]]:
    """Nested session-fold capacity selection and held-fold prediction."""
    output = np.empty(len(y), dtype=np.int16)
    records: list[dict[str, Any]] = []
    fitted_biases: list[np.ndarray] = []
    for held_fold in sorted(np.unique(fold).tolist()):
        held = fold == held_fold
        inner_rows = ~held
        candidate_table: list[dict[str, Any]] = []
        for spec in SPECS:
            inner_prediction = np.empty(len(y), dtype=np.int16)
            for inner_fold in sorted(np.unique(fold[inner_rows]).tolist()):
                validation = inner_rows & (fold == inner_fold)
                train = inner_rows & (fold != inner_fold) & terminal
                bias = fit_direct_f1_bias(logits[train], y[train], spec)
                inner_prediction[validation] = prediction(logits[validation], bias)
            term_score = macro_from_prediction(y, inner_prediction, inner_rows & terminal)
            all_score = macro_from_prediction(y, inner_prediction, inner_rows)
            candidate_table.append({"spec": spec, "terminal_macro_f1": term_score, "all_fallback_macro_f1": all_score})
        chosen = max(
            candidate_table,
            key=lambda row: (
                row["terminal_macro_f1"], row["all_fallback_macro_f1"],
                tuple(-value for value in row["spec"].complexity),
            ),
        )
        chosen_spec: OffsetSpec = chosen["spec"]
        fit = (~held) & terminal
        bias = fit_direct_f1_bias(logits[fit], y[fit], chosen_spec)
        output[held] = prediction(logits[held], bias)
        fitted_biases.append(bias)
        records.append({
            "held_outer_fold": int(held_fold + 1), "chosen": chosen_spec.key,
            "inner_candidates": [
                {"spec": row["spec"].key, "terminal_macro_f1": row["terminal_macro_f1"],
                 "all_fallback_macro_f1": row["all_fallback_macro_f1"]}
                for row in candidate_table
            ],
            "held_bias": {CLASSES[index]: float(value) for index, value in enumerate(bias)},
        })
    return output, records, fitted_biases


def select_for_common_train(
    logits: np.ndarray, y: np.ndarray, terminal: np.ndarray, groups: np.ndarray,
) -> tuple[OffsetSpec, np.ndarray, list[dict[str, Any]]]:
    """Select capacity using only common-train labels, then refit once."""
    fold = np.empty(len(y), dtype=np.int8)
    for number, (_, held) in enumerate(GroupKFold(n_splits=5).split(np.zeros(len(y)), y, groups)):
        fold[held] = number
    table: list[dict[str, Any]] = []
    for spec in SPECS:
        oof_prediction = np.empty(len(y), dtype=np.int16)
        for held_fold in sorted(np.unique(fold).tolist()):
            held = fold == held_fold
            train = (~held) & terminal
            bias = fit_direct_f1_bias(logits[train], y[train], spec)
            oof_prediction[held] = prediction(logits[held], bias)
        table.append({
            "spec": spec,
            "terminal_macro_f1": macro_from_prediction(y, oof_prediction, terminal),
            "all_fallback_macro_f1": macro_from_prediction(y, oof_prediction),
        })
    chosen = max(
        table,
        key=lambda row: (
            row["terminal_macro_f1"], row["all_fallback_macro_f1"],
            tuple(-value for value in row["spec"].complexity),
        ),
    )
    spec: OffsetSpec = chosen["spec"]
    bias = fit_direct_f1_bias(logits[terminal], y[terminal], spec)
    serialized = [
        {"spec": row["spec"].key, "terminal_macro_f1": row["terminal_macro_f1"],
         "all_fallback_macro_f1": row["all_fallback_macro_f1"]}
        for row in table
    ]
    return spec, bias, serialized


def with_overlay(prediction_value: np.ndarray, terminal: np.ndarray, override: np.ndarray) -> np.ndarray:
    value = prediction_value.copy()
    mask = terminal & (override >= 0)
    value[mask] = override[mask]
    return value


def metrics(y: np.ndarray, fallback: np.ndarray, terminal: np.ndarray, word15: np.ndarray, char20: np.ndarray) -> dict[str, float]:
    result = {
        "all_fallback_macro_f1": macro_from_prediction(y, fallback),
        "terminal_no_overlay_macro_f1": macro_from_prediction(y, fallback, terminal),
    }
    for name, override in (("word15", word15), ("char20", char20)):
        final = with_overlay(fallback, terminal, override)
        hybrid = final.copy()
        # On the labeled full-session audit, every non-terminal action is
        # deterministically exposed by a later assistant-action history.
        hybrid[~terminal] = y[~terminal]
        result[f"all_fallback_with_{name}_macro_f1"] = macro_from_prediction(y, final)
        result[f"terminal_with_{name}_macro_f1"] = macro_from_prediction(y, final, terminal)
        result[f"hybrid_future_history_plus_{name}_macro_f1"] = macro_from_prediction(y, hybrid)
    return result


def deltas(candidate: dict[str, float], baseline: dict[str, float]) -> dict[str, float]:
    return {key: float(candidate[key] - baseline[key]) for key in candidate}


def fixed_char_prediction(char: Any, fixed_terminal: np.ndarray) -> np.ndarray:
    result = np.full(len(fixed_terminal), -1, dtype=np.int16)
    positions = np.flatnonzero(fixed_terminal)
    if len(positions) != len(char["fixed_char20"]):
        raise ValueError("Char20 fixed terminal length mismatch")
    result[positions] = char["fixed_char20"].astype(np.int16)
    return result


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    out = root / "experiments/class_threshold_old"
    out.mkdir(parents=True, exist_ok=True)

    samples = [json.loads(line) for line in (root / "data/train.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    labels = {row["id"]: row["action"] for row in csv.DictReader((root / "data/train_labels.csv").open(encoding="utf-8"))}
    split = {row["id"]: row["split"] for row in csv.DictReader((root / "splits/common_split.csv").open(encoding="utf-8"))}
    ids = np.asarray([str(row["id"]) for row in samples])
    to_id = {name: index for index, name in enumerate(CLASSES)}
    y = np.asarray([to_id[labels[sample_id]] for sample_id in ids], dtype=np.int64)
    source = np.asarray([source_of(sample_id) for sample_id in ids])
    groups = np.asarray([session_of(sample_id) for sample_id in ids])
    terminal = terminal_mask(ids)

    # The historical proxy is the exact B42_9 full-OOF selector component.
    # The robust-copy equality assertion guards against silently auditing a
    # different selector seed/configuration.
    raw_full = np.load(root / "experiments/modeling_ak_live_xgb_proxy/oof_probabilities.npy").astype(np.float32)
    raw_full_seed42 = np.load(root / "experiments/xgb_robust_ensemble/full_oof_raw_seed42.npy").astype(np.float32)
    old_gru_full = np.load(root / "experiments/modeling_ai_sequence_fusion_noe5/oof_probabilities.npy").astype(np.float32)
    if not (raw_full.shape == raw_full_seed42.shape == old_gru_full.shape == (len(y), N_CLASSES)):
        raise ValueError("Full B42_9 component shape mismatch")
    full_raw_max_abs_difference = float(np.max(np.abs(raw_full - raw_full_seed42)))
    if full_raw_max_abs_difference > 1.0e-7:
        raise ValueError(f"Historical raw selector is not matching seed42 ({full_raw_max_abs_difference})")
    full_logits = geometric_logits(raw_full, old_gru_full, source)

    # Terminal overlays are precomputed session-OOF / common-train-only rule
    # predictions.  Their IDs and terminal masks are asserted before use.
    state = np.load(root / "experiments/state_residual_gate/predictions.npz", allow_pickle=False)
    if not (np.array_equal(state["target"].astype(np.int64), y) and np.array_equal(state["terminal"].astype(bool), terminal)):
        raise ValueError("B42_9 word15 OOF artifact does not align")
    word15_full = state["word15_full"].astype(np.int16)
    char = np.load(root / "experiments/terminal_rule_combinations/oof_candidates.npz", allow_pickle=False)
    if not (np.array_equal(char["y"].astype(np.int64), y) and np.array_equal(char["terminal"].astype(bool), terminal)):
        raise ValueError("Char20 OOF artifact does not align")
    char20_full = char["char20"].astype(np.int16)

    full_fold = np.empty(len(y), dtype=np.int8)
    for number, (_, held) in enumerate(GroupKFold(n_splits=5).split(np.zeros(len(y)), y, groups)):
        full_fold[held] = number
    baseline_full_prediction = prediction(full_logits, np.zeros(N_CLASSES, dtype=np.float64))
    nested_prediction, nested_records, outer_biases = select_for_outer(full_logits, y, terminal, full_fold)
    full_baseline_metrics = metrics(y, baseline_full_prediction, terminal, word15_full, char20_full)
    full_candidate_metrics = metrics(y, nested_prediction, terminal, word15_full, char20_full)

    # The fixed side is fully reconstructed from common-train-only artifacts.
    # This deliberately does not use legacy 2025-feature or seed-808 tables.
    common_dir = root / "experiments/b42_9_common_old_noe5_seed42"
    raw_common_dir = root / "experiments/b42_9_common_raw_seed42"
    required = [
        common_dir / "common_train_oof_probabilities.npy", common_dir / "common_train_ids.npy",
        common_dir / "fixed_valid_probabilities.npy", common_dir / "fixed_valid_ids.npy",
        raw_common_dir / "common_train_raw_xgb_oof.npy", raw_common_dir / "common_train_ids.npy",
    ]
    missing = [str(path.relative_to(root)) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing strict common artifacts: " + ", ".join(missing))
    common_index = np.flatnonzero(np.asarray([split[sample_id] == "train" for sample_id in ids]))
    fixed_index = np.flatnonzero(np.asarray([split[sample_id] == "val" for sample_id in ids]))
    common_ids = ids[common_index]
    fixed_ids = ids[fixed_index]
    raw_common_ids = np.load(raw_common_dir / "common_train_ids.npy", allow_pickle=False).astype(str)
    old_common_ids = np.load(common_dir / "common_train_ids.npy", allow_pickle=False).astype(str)
    old_fixed_ids = np.load(common_dir / "fixed_valid_ids.npy", allow_pickle=False).astype(str)
    if not (np.array_equal(raw_common_ids, common_ids) and np.array_equal(old_common_ids, common_ids)
            and np.array_equal(old_fixed_ids, fixed_ids)):
        raise ValueError("Common/fixed strict artifact ID alignment failed")
    raw_common = np.load(raw_common_dir / "common_train_raw_xgb_oof.npy", allow_pickle=False).astype(np.float32)
    old_common_gru = np.load(common_dir / "common_train_oof_probabilities.npy", allow_pickle=False).astype(np.float32)
    raw_fixed = np.load(root / "experiments/xgb_robust_ensemble/strict_fixed_raw_seed42.npy", allow_pickle=False).astype(np.float32)
    old_fixed_gru = np.load(common_dir / "fixed_valid_probabilities.npy", allow_pickle=False).astype(np.float32)
    if not (raw_common.shape == old_common_gru.shape == (len(common_index), N_CLASSES)
            and raw_fixed.shape == old_fixed_gru.shape == (len(fixed_index), N_CLASSES)):
        raise ValueError("Strict common/fixed component shape mismatch")
    common_logits = geometric_logits(raw_common, old_common_gru, source[common_index])
    fixed_logits = geometric_logits(raw_fixed, old_fixed_gru, source[fixed_index])
    common_terminal = terminal[common_index]
    fixed_terminal = terminal[fixed_index]
    word15_fixed = state["word15_fixed"].astype(np.int16)
    if len(word15_fixed) != len(fixed_index):
        raise ValueError("Word15 fixed prediction shape mismatch")
    expected_fixed_terminal_ids = fixed_ids[fixed_terminal]
    if not np.array_equal(char["valid_ids"].astype(str), expected_fixed_terminal_ids):
        raise ValueError("Char20 fixed IDs do not align")
    char20_fixed = fixed_char_prediction(char, fixed_terminal)
    common_spec, common_bias, common_table = select_for_common_train(
        common_logits, y[common_index], common_terminal, groups[common_index],
    )
    fixed_baseline_prediction = prediction(fixed_logits, np.zeros(N_CLASSES, dtype=np.float64))
    fixed_candidate_prediction = prediction(fixed_logits, common_bias)
    fixed_baseline_metrics = metrics(y[fixed_index], fixed_baseline_prediction, fixed_terminal, word15_fixed, char20_fixed)
    fixed_candidate_metrics = metrics(y[fixed_index], fixed_candidate_prediction, fixed_terminal, word15_fixed, char20_fixed)

    full_delta = deltas(full_candidate_metrics, full_baseline_metrics)
    fixed_delta = deltas(fixed_candidate_metrics, fixed_baseline_metrics)
    required_metrics = (
        "all_fallback_macro_f1",
        "terminal_with_word15_macro_f1", "all_fallback_with_word15_macro_f1", "hybrid_future_history_plus_word15_macro_f1",
        "terminal_with_char20_macro_f1", "all_fallback_with_char20_macro_f1", "hybrid_future_history_plus_char20_macro_f1",
    )
    passed = all(full_delta[key] > 0.0 and fixed_delta[key] > 0.0 for key in required_metrics)

    # Only when the two untouched protocols both clear the strict all-row and
    # terminal gate do we make a full-data vector available for manual build
    # review.  This script never applies it to a bundle itself.
    portable_path: Path | None = None
    if passed:
        modal = max(
            ((record["chosen"], index) for index, record in enumerate(nested_records)),
            key=lambda value: (sum(item["chosen"] == value[0] for item in nested_records), -value[1]),
        )[0]
        full_spec = next(spec for spec in SPECS if spec.key == modal)
        full_bias = fit_direct_f1_bias(full_logits[terminal], y[terminal], full_spec)
        portable_path = out / "recommended_portable_logit_bias.json"
        portable_path.write_text(json.dumps({
            "version": "geometric_terminal_logit_bias_v1",
            "base": {"blend_mode": "geometric", "source_xgb_weights": {"sim": 0.20, "au": 0.85}},
            "spec": full_spec.key,
            "classes": list(CLASSES),
            "logit_bias": [float(value) for value in full_bias],
            "apply_contract": (
                "After the raw XGB/no-E5 GRU geometric log-probability blend and before argmax; do not use labels at "
                "inference; preserve the terminal lexical overlay and final exact future-history recovery priority."
            ),
        }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    report: dict[str, Any] = {
        "scope": "Experiment-only audit.  model/, script.py, and every submission ZIP remain unchanged.",
        "baseline": "Exact B42_9 old no-E5 geometric blend: raw seed42 selector, SIM=.20/AU=.85.",
        "validation_protocol": {
            "full": "5-fold whole-session nested selection; each held fold is predicted by an offset capacity selected and fitted without its labels.",
            "fixed": "Common-train-only seed42 raw-XGB and no-E5 GRU OOF select capacity; the final offset is fit on common-train terminal rows and evaluated once on untouched strict fixed validation.",
            "overlays": "word15 and char20 are downstream terminal-only OOF/common-train rule predictions, never inputs to the offset fit.",
            "selection_metric": "unoverridden terminal macro-F1, then all-row fallback macro-F1, then lower capacity.",
        },
        "component_audit": {
            "full_raw_seed42_max_abs_difference_historical_vs_robust_artifact": full_raw_max_abs_difference,
            "common_raw_metadata": json.loads((raw_common_dir / "common_train_raw_xgb_oof_metadata.json").read_text(encoding="utf-8")),
            "common_old_noe5": json.loads((common_dir / "report.json").read_text(encoding="utf-8")),
        },
        "counts": {"full_rows": int(len(y)), "full_terminal": int(terminal.sum()),
                   "common_train_rows": int(len(common_index)), "common_train_terminal": int(common_terminal.sum()),
                   "fixed_rows": int(len(fixed_index)), "fixed_terminal": int(fixed_terminal.sum())},
        "full_nested_session_oof": {
            "baseline": full_baseline_metrics, "candidate": full_candidate_metrics,
            "delta": full_delta, "outer_selection": nested_records,
        },
        "matching_seed42_fixed_validation": {
            "baseline": fixed_baseline_metrics, "candidate": fixed_candidate_metrics,
            "delta": fixed_delta,
            "common_train_selection": {"chosen": common_spec.key, "bias": {CLASSES[index]: float(value) for index, value in enumerate(common_bias)},
                                       "candidates": common_table},
        },
        "promotion_gate": {
            "required_positive_deltas": list(required_metrics),
            "pass": passed,
            "reason": (
                "Every all-row, terminal, and future-history hybrid metric improved after both word15 and char20 overlays in both protocols."
                if passed else
                "At least one required all-row or terminal delta is non-positive in nested full OOF or matching seed42 fixed validation."
            ),
        },
        "portable_build_implication": (
            {"recommended_artifact": str(portable_path.relative_to(root)),
             "implementation": "Add the artifact vector to the model bundle and add it to geometric blended logits before argmax; leave overlay/future-history ordering intact."}
            if portable_path is not None else
            {"recommended_artifact": None,
             "implementation": "Do not modify the submission bundle; no compact F1-aware threshold shift cleared the two-protocol gate."}
        ),
    }
    (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    np.savez_compressed(
        out / "predictions.npz", y=y, terminal=terminal,
        baseline_full=baseline_full_prediction, candidate_full=nested_prediction,
        fixed_index=fixed_index, fixed_terminal=fixed_terminal,
        baseline_fixed=fixed_baseline_prediction, candidate_fixed=fixed_candidate_prediction,
        word15_full=word15_full, char20_full=char20_full, word15_fixed=word15_fixed, char20_fixed=char20_fixed,
    )
    print(json.dumps({
        "pass": passed,
        "full_delta": full_delta,
        "fixed_delta": fixed_delta,
        "common_chosen": common_spec.key,
    }, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
