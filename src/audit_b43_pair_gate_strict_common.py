#!/usr/bin/env python3
"""Audit a compact terminal prediction-pair gate without touching submissions.

The public B43 fallback is retained exactly for the full 70,000-row outer-OOF
side of this audit.  The only candidate intervention is a terminal-only
lookup over the pair ``(B43 fallback prediction, dual-GRU fallback
prediction)``.  The lookup is fit from labels in training partitions, but at
runtime it consumes only those two predictions and the supplied-session
terminal marker.  It never consumes an ID, prompt/text feature, char20 rule,
or a label.

Two independent protocols are reported:

* Full data: five whole-session outer folds, with gate capacity selected in
  the other four folds and the held fold predicted without its labels.
* Fixed data: all gate-capacity selection and pair statistics use only the
  strict common-train cache.  The fixed common-validation labels are read
  only after the policy prediction has been formed.

This is an experiment-only audit.  It must not write model/, script.py,
build_zip.py, staging directories, or any ZIP file.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
N_CLASSES = 14
EPS = 1.0e-12


@dataclass(frozen=True)
class GateSpec:
    """A deliberately small family of pair-support policies.

    ``min_support`` and ``min_net`` are computed only for terminal rows whose
    base and dual predictions disagree.  ``positive_fold_count`` requires the
    pair to have a positive local dual-minus-base correctness net in at least
    that many independent session folds.  Lower ``complexity_rank`` wins
    exact validation-score ties, deliberately preferring a no-op or a more
    conservative rule.
    """

    key: str
    min_support: int
    min_net: int
    positive_fold_count: int
    complexity_rank: int


SPECS: tuple[GateSpec, ...] = (
    GateSpec("base", 1_000_000, 1_000_000, 0, 0),
    GateSpec("s5n1stable2", 5, 1, 2, 1),
    GateSpec("s4n1stable2", 4, 1, 2, 2),
    GateSpec("s3n1stable2", 3, 1, 2, 3),
    GateSpec("s2n2", 2, 2, 0, 4),
    GateSpec("s2n1", 2, 1, 0, 5),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--strict-dir", default="experiments/b43_pair_gate_strict_common")
    parser.add_argument("--out-dir", default="experiments/b43_pair_gate_strict_audit")
    parser.add_argument("--replicates", type=int, default=5000)
    parser.add_argument("--full-bootstrap-seed", type=int, default=20260715)
    parser.add_argument("--fixed-bootstrap-seed", type=int, default=20260716)
    return parser.parse_args()


def session_of(sample_id: str) -> str:
    return sample_id.rsplit("-step_", 1)[0]


def source_alpha(ids: np.ndarray) -> np.ndarray:
    return np.where(np.char.startswith(ids.astype(str), "sess_au_"), 0.85, 0.20).astype(np.float64)


def normalized_geometric(raw: np.ndarray, gru: np.ndarray, ids: np.ndarray) -> np.ndarray:
    alpha = source_alpha(ids)[:, None]
    logits = alpha * np.log(np.maximum(raw, EPS)) + (1.0 - alpha) * np.log(np.maximum(gru, EPS))
    logits -= logits.max(axis=1, keepdims=True)
    probability = np.exp(logits)
    return (probability / probability.sum(axis=1, keepdims=True)).astype(np.float32)


def macro_from_prediction(y: np.ndarray, prediction: np.ndarray, mask: np.ndarray | None = None) -> float:
    if mask is None:
        mask = np.ones(len(y), dtype=bool)
    actual = y[mask].astype(np.int64, copy=False)
    predicted = prediction[mask].astype(np.int64, copy=False)
    matrix = np.bincount(
        N_CLASSES * actual + predicted, minlength=N_CLASSES * N_CLASSES,
    ).reshape(N_CLASSES, N_CLASSES)
    true_positive = np.diag(matrix).astype(np.float64)
    false_positive = matrix.sum(axis=0) - true_positive
    false_negative = matrix.sum(axis=1) - true_positive
    denominator = 2.0 * true_positive + false_positive + false_negative
    per_class = np.divide(
        2.0 * true_positive, denominator, out=np.zeros(N_CLASSES, dtype=np.float64), where=denominator > 0,
    )
    return float(per_class.mean())


def with_char20(prediction: np.ndarray, terminal: np.ndarray, char20: np.ndarray) -> np.ndarray:
    output = prediction.astype(np.int16, copy=True)
    override = terminal & (char20 >= 0)
    output[override] = char20[override]
    return output


def metrics(y: np.ndarray, fallback: np.ndarray, terminal: np.ndarray, char20: np.ndarray) -> dict[str, float]:
    final = with_char20(fallback, terminal, char20)
    hybrid = final.copy()
    # This is only the established offline evaluation metric: at a later
    # assistant turn, the current non-terminal action is known from history.
    hybrid[~terminal] = y[~terminal]
    return {
        "all_fallback_macro_f1": macro_from_prediction(y, fallback),
        "terminal_fallback_macro_f1": macro_from_prediction(y, fallback, terminal),
        "terminal_with_char20_macro_f1": macro_from_prediction(y, final, terminal),
        "hybrid_future_history_plus_char20_macro_f1": macro_from_prediction(y, hybrid),
    }


def metric_delta(candidate: dict[str, float], baseline: dict[str, float]) -> dict[str, float]:
    return {key: float(candidate[key] - baseline[key]) for key in baseline}


def fit_gate(
    base: np.ndarray,
    dual: np.ndarray,
    y: np.ndarray,
    terminal: np.ndarray,
    fold: np.ndarray,
    train: np.ndarray,
    spec: GateSpec,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Fit an inference-safe pair lookup strictly on ``train`` rows."""
    table = np.zeros((N_CLASSES, N_CLASSES), dtype=bool)
    chosen: list[dict[str, Any]] = []
    if spec.key == "base":
        return table, chosen
    changed = train & terminal & (base != dual)
    for old_label in range(N_CLASSES):
        for new_label in range(N_CLASSES):
            if old_label == new_label:
                continue
            pair = changed & (base == old_label) & (dual == new_label)
            support = int(pair.sum())
            if support < spec.min_support:
                continue
            base_correct = int((y[pair] == old_label).sum())
            dual_correct = int((y[pair] == new_label).sum())
            net = dual_correct - base_correct
            if net < spec.min_net:
                continue
            local_positive = 0
            local_nets: dict[str, int] = {}
            if spec.positive_fold_count:
                for value in sorted(np.unique(fold[pair]).tolist()):
                    local = pair & (fold == value)
                    local_net = int((y[local] == new_label).sum()) - int((y[local] == old_label).sum())
                    local_nets[str(int(value))] = local_net
                    local_positive += int(local_net > 0)
                if local_positive < spec.positive_fold_count:
                    continue
            table[old_label, new_label] = True
            chosen.append({
                "base_label": old_label,
                "dual_label": new_label,
                "support": support,
                "base_correct": base_correct,
                "dual_correct": dual_correct,
                "net_dual_minus_base": net,
                "positive_fold_count": local_positive,
                "local_nets": local_nets,
            })
    return table, chosen


def apply_gate(base: np.ndarray, dual: np.ndarray, terminal: np.ndarray, table: np.ndarray) -> np.ndarray:
    """Runtime operation: only prediction pair and terminal marker are read."""
    output = base.astype(np.int16, copy=True)
    switch = terminal & (base != dual) & table[base, dual]
    output[switch] = dual[switch]
    return output


def crossfit_gate_predictions(
    base: np.ndarray,
    dual: np.ndarray,
    y: np.ndarray,
    terminal: np.ndarray,
    fold: np.ndarray,
    spec: GateSpec,
    active: np.ndarray,
) -> np.ndarray:
    """Second-level OOF predictions for one finite gate capacity."""
    output = base.astype(np.int16, copy=True)
    for held_value in sorted(np.unique(fold[active]).tolist()):
        held = active & (fold == held_value)
        train = active & (fold != held_value)
        table, _ = fit_gate(base, dual, y, terminal, fold, train, spec)
        output[held] = apply_gate(base[held], dual[held], terminal[held], table)
    return output


def select_key(row: dict[str, Any]) -> tuple[float, float, int]:
    return (
        float(row["terminal_fallback_macro_f1"]),
        float(row["all_fallback_macro_f1"]),
        -int(row["spec"].complexity_rank),
    )


def nested_full_prediction(
    base: np.ndarray,
    dual: np.ndarray,
    y: np.ndarray,
    terminal: np.ndarray,
    fold: np.ndarray,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Select capacity inside each held whole-session outer fold."""
    output = base.astype(np.int16, copy=True)
    records: list[dict[str, Any]] = []
    for held_value in sorted(np.unique(fold).tolist()):
        held = fold == held_value
        inner = ~held
        candidates: list[dict[str, Any]] = []
        for spec in SPECS:
            inner_prediction = crossfit_gate_predictions(base, dual, y, terminal, fold, spec, inner)
            candidates.append({
                "spec": spec,
                "terminal_fallback_macro_f1": macro_from_prediction(y, inner_prediction, inner & terminal),
                "all_fallback_macro_f1": macro_from_prediction(y, inner_prediction, inner),
            })
        selected = max(candidates, key=select_key)
        spec = selected["spec"]
        table, pair_stats = fit_gate(base, dual, y, terminal, fold, inner, spec)
        output[held] = apply_gate(base[held], dual[held], terminal[held], table)
        records.append({
            "held_outer_fold": int(held_value + 1),
            "selected_spec": spec.key,
            "inner_candidates": [
                {
                    "spec": value["spec"].key,
                    "terminal_fallback_macro_f1": value["terminal_fallback_macro_f1"],
                    "all_fallback_macro_f1": value["all_fallback_macro_f1"],
                }
                for value in candidates
            ],
            "selected_pair_count": int(table.sum()),
            "selected_pairs": pair_stats,
            "held_terminal_switches": int((terminal[held] & (output[held] != base[held])).sum()),
        })
    return output, records


def select_common_policy(
    base: np.ndarray,
    dual: np.ndarray,
    y: np.ndarray,
    terminal: np.ndarray,
    fold: np.ndarray,
) -> tuple[GateSpec, np.ndarray, list[dict[str, Any]], list[dict[str, Any]]]:
    """Select and refit a policy solely from strict common-train rows."""
    active = np.ones(len(y), dtype=bool)
    candidates: list[dict[str, Any]] = []
    for spec in SPECS:
        prediction = crossfit_gate_predictions(base, dual, y, terminal, fold, spec, active)
        candidates.append({
            "spec": spec,
            "terminal_fallback_macro_f1": macro_from_prediction(y, prediction, terminal),
            "all_fallback_macro_f1": macro_from_prediction(y, prediction),
        })
    selected = max(candidates, key=select_key)
    table, pair_stats = fit_gate(base, dual, y, terminal, fold, active, selected["spec"])
    serial = [
        {
            "spec": value["spec"].key,
            "terminal_fallback_macro_f1": value["terminal_fallback_macro_f1"],
            "all_fallback_macro_f1": value["all_fallback_macro_f1"],
        }
        for value in candidates
    ]
    return selected["spec"], table, serial, pair_stats


def terminal_session_bootstrap(
    y: np.ndarray,
    baseline: np.ndarray,
    candidate: np.ndarray,
    terminal: np.ndarray,
    groups: np.ndarray,
    replicates: int,
    seed: int,
) -> dict[str, float | int]:
    """Paired whole-session bootstrap of terminal char20-final predictions."""
    index = np.flatnonzero(terminal)
    sessions = groups[index]
    if len(np.unique(sessions)) != len(sessions):
        raise ValueError("Expected exactly one terminal row per session")
    y_terminal = y[index]
    base_terminal = baseline[index]
    candidate_terminal = candidate[index]
    exact = macro_from_prediction(y_terminal, candidate_terminal) - macro_from_prediction(y_terminal, base_terminal)
    rng = np.random.default_rng(seed)
    deltas = np.empty(replicates, dtype=np.float64)
    for number in range(replicates):
        sample = rng.integers(0, len(index), size=len(index))
        deltas[number] = (
            macro_from_prediction(y_terminal[sample], candidate_terminal[sample])
            - macro_from_prediction(y_terminal[sample], base_terminal[sample])
        )
    return {
        "replicates": int(replicates),
        "seed": int(seed),
        "sessions": int(len(index)),
        "exact_delta": float(exact),
        "mean_delta": float(deltas.mean()),
        "p_delta_gt_zero": float((deltas > 0.0).mean()),
        "p05_delta": float(np.quantile(deltas, 0.05)),
        "p95_delta": float(np.quantile(deltas, 0.95)),
    }


def load_data() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    classes = (
        "read_file", "grep_search", "list_directory", "glob_pattern",
        "edit_file", "write_file", "apply_patch", "run_bash", "run_tests",
        "lint_or_typecheck", "ask_user", "plan_task", "web_search", "respond_only",
    )
    samples = [json.loads(line) for line in (ROOT / "data/train.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    labels = {row["id"]: row["action"] for row in csv.DictReader((ROOT / "data/train_labels.csv").open(encoding="utf-8"))}
    split = {row["id"]: row["split"] for row in csv.DictReader((ROOT / "splits/common_split.csv").open(encoding="utf-8"))}
    ids = np.asarray([str(row["id"]) for row in samples])
    to_id = {name: number for number, name in enumerate(classes)}
    y = np.asarray([to_id[labels[value]] for value in ids], dtype=np.int64)
    groups = np.asarray([session_of(value) for value in ids])
    common = np.asarray([split[value] == "train" for value in ids], dtype=bool)
    fixed = np.asarray([split[value] == "val" for value in ids], dtype=bool)
    if not np.all(common | fixed) or np.any(common & fixed):
        raise ValueError("Common split is not a disjoint train/validation partition")
    terminal = np.zeros(len(ids), dtype=bool)
    for session in np.unique(groups):
        positions = np.flatnonzero(groups == session)
        steps = np.asarray([int(ids[position].rsplit("-step_", 1)[1]) for position in positions])
        terminal[positions[np.argmax(steps)]] = True
    return ids, y, groups, common, terminal


def assert_probability(name: str, value: np.ndarray, rows: int) -> None:
    if value.shape != (rows, N_CLASSES):
        raise ValueError(f"{name} has unexpected shape {value.shape}")
    if not np.isfinite(value).all() or np.any(value.sum(axis=1) <= 0.0):
        raise ValueError(f"{name} is not a valid probability table")


def main() -> None:
    args = parse_args()
    if args.replicates < 100:
        raise ValueError("At least 100 paired bootstrap replicates are required")
    strict_dir = ROOT / args.strict_dir
    out = ROOT / args.out_dir
    model_root = (ROOT / "model").resolve()
    if out.resolve() == model_root or model_root in out.resolve().parents:
        raise ValueError("Audit output must not be inside model/")
    out.mkdir(parents=True, exist_ok=True)

    ids, y, groups, common_mask, terminal = load_data()
    fixed_mask = ~common_mask
    common_index = np.flatnonzero(common_mask)
    fixed_index = np.flatnonzero(fixed_mask)
    common_ids = ids[common_index]
    fixed_ids = ids[fixed_index]
    common_y = y[common_index]
    fixed_y = y[fixed_index]
    common_terminal = terminal[common_index]
    fixed_terminal = terminal[fixed_index]

    # Exact public B43 controls are intentionally loaded from the prior exact
    # probability audit rather than reconstructed from a nearby experiment.
    exact = np.load(ROOT / "experiments/b43_old_strict_half_exact_audit/probability_audit.npz", allow_pickle=False)
    exact_y = exact["y"].astype(np.int64)
    exact_terminal = exact["terminal"].astype(bool)
    exact_base_full = exact["base_full_probability"].astype(np.float32).argmax(axis=1).astype(np.int16)
    exact_dual_full = exact["dual_full_probability"].astype(np.float32).argmax(axis=1).astype(np.int16)
    exact_base_fixed = exact["base_fixed_probability"].astype(np.float32).argmax(axis=1).astype(np.int16)
    exact_dual_fixed = exact["dual_fixed_probability"].astype(np.float32).argmax(axis=1).astype(np.int16)
    exact_fixed_index = exact["fixed_index"].astype(np.int64)
    exact_fixed_terminal = exact["fixed_terminal"].astype(bool)
    char_full = exact["char_full"].astype(np.int16)
    char_fixed = exact["char_fixed"].astype(np.int16)
    if not (
        np.array_equal(exact_y, y)
        and np.array_equal(exact_terminal, terminal)
        and np.array_equal(exact_fixed_index, fixed_index)
        and np.array_equal(exact_fixed_terminal, fixed_terminal)
        and char_full.shape == (len(y),)
        and char_fixed.shape == (len(fixed_y),)
    ):
        raise ValueError("Exact B43 audit alignment failed")

    full_fold = np.load(
        ROOT / "experiments/modeling_ai_sequence_nested_distill_e16_i4/outer_fold_assignment.npy", allow_pickle=False,
    ).astype(np.int8)
    if full_fold.shape != (len(y),) or np.any(full_fold < 0) or np.unique(full_fold).size != 5:
        raise ValueError("Exact B43 outer fold assignment is invalid")
    # The full B43 audit already recorded this, but recompute the no-session
    # overlap check directly from IDs to make this artifact self-contained.
    for held_value in np.unique(full_fold):
        if set(groups[full_fold == held_value]).intersection(set(groups[full_fold != held_value])):
            raise ValueError("B43 outer-fold assignment splits a session")

    full_baseline_metrics = metrics(y, exact_base_full, terminal, char_full)
    expected = {
        "all_fallback_macro_f1": 0.7603385811033483,
        "terminal_fallback_macro_f1": 0.7312748947080282,
        "terminal_with_char20_macro_f1": 0.7335225252513891,
        "hybrid_future_history_plus_char20_macro_f1": 0.98277826622222,
    }
    for name, target in expected.items():
        if abs(full_baseline_metrics[name] - target) > 1.0e-12:
            raise ValueError(f"Exact B43 full control mismatch for {name}: {full_baseline_metrics[name]} != {target}")

    # Full nested policy selection.  Char20 is not passed to any selector or
    # gate function: it is applied only after the terminal fallback is fixed.
    full_candidate, full_records = nested_full_prediction(
        exact_base_full, exact_dual_full, y, terminal, full_fold,
    )
    full_candidate_metrics = metrics(y, full_candidate, terminal, char_full)
    full_delta = metric_delta(full_candidate_metrics, full_baseline_metrics)
    full_final_base = with_char20(exact_base_full, terminal, char_full)
    full_final_candidate = with_char20(full_candidate, terminal, char_full)
    full_bootstrap = terminal_session_bootstrap(
        y, full_final_base, full_final_candidate, terminal, groups, args.replicates, args.full_bootstrap_seed,
    )
    for record in full_records:
        held = full_fold == (record["held_outer_fold"] - 1)
        record["held_metrics"] = {
            "baseline": metrics(y[held], exact_base_full[held], terminal[held], char_full[held]),
            "candidate": metrics(y[held], full_candidate[held], terminal[held], char_full[held]),
        }
        record["held_delta"] = metric_delta(record["held_metrics"]["candidate"], record["held_metrics"]["baseline"])

    # Strict common-only components.  These caches make the fixed score a
    # genuine untouched result: the fixed labels never enter model fitting,
    # pair selection, or pair-table fitting.
    old_dir = ROOT / "experiments/b42_9_common_old_noe5_seed42"
    raw_dir = ROOT / "experiments/b42_9_common_raw_seed42"
    required_paths = (
        strict_dir / "common_train_strict_oof_probabilities.npy",
        strict_dir / "common_train_ids.npy",
        strict_dir / "common_train_fold.npy",
        strict_dir / "fixed_strict_probabilities.npy",
        strict_dir / "fixed_ids.npy",
        old_dir / "common_train_oof_probabilities.npy",
        old_dir / "common_train_ids.npy",
        old_dir / "common_train_fold.npy",
        old_dir / "fixed_valid_probabilities.npy",
        old_dir / "fixed_valid_ids.npy",
        raw_dir / "common_train_raw_xgb_oof.npy",
        raw_dir / "common_train_ids.npy",
        raw_dir / "common_train_selector_fold.npy",
        ROOT / "experiments/xgb_robust_ensemble/strict_fixed_raw_seed42.npy",
    )
    missing = [str(path.relative_to(ROOT)) for path in required_paths if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing strict common component(s): " + ", ".join(missing))
    strict_common_ids = np.load(strict_dir / "common_train_ids.npy", allow_pickle=False).astype(str)
    strict_fixed_ids = np.load(strict_dir / "fixed_ids.npy", allow_pickle=False).astype(str)
    old_common_ids = np.load(old_dir / "common_train_ids.npy", allow_pickle=False).astype(str)
    old_fixed_ids = np.load(old_dir / "fixed_valid_ids.npy", allow_pickle=False).astype(str)
    raw_common_ids = np.load(raw_dir / "common_train_ids.npy", allow_pickle=False).astype(str)
    if not (
        np.array_equal(strict_common_ids, common_ids)
        and np.array_equal(old_common_ids, common_ids)
        and np.array_equal(raw_common_ids, common_ids)
        and np.array_equal(strict_fixed_ids, fixed_ids)
        and np.array_equal(old_fixed_ids, fixed_ids)
    ):
        raise ValueError("Strict common component ID alignment failed")
    raw_common = np.load(raw_dir / "common_train_raw_xgb_oof.npy", allow_pickle=False).astype(np.float32)
    old_common = np.load(old_dir / "common_train_oof_probabilities.npy", allow_pickle=False).astype(np.float32)
    strict_common = np.load(strict_dir / "common_train_strict_oof_probabilities.npy", allow_pickle=False).astype(np.float32)
    raw_fixed = np.load(ROOT / "experiments/xgb_robust_ensemble/strict_fixed_raw_seed42.npy", allow_pickle=False).astype(np.float32)
    old_fixed = np.load(old_dir / "fixed_valid_probabilities.npy", allow_pickle=False).astype(np.float32)
    strict_fixed = np.load(strict_dir / "fixed_strict_probabilities.npy", allow_pickle=False).astype(np.float32)
    for name, value, rows in (
        ("raw_common", raw_common, len(common_y)), ("old_common", old_common, len(common_y)),
        ("strict_common", strict_common, len(common_y)), ("raw_fixed", raw_fixed, len(fixed_y)),
        ("old_fixed", old_fixed, len(fixed_y)), ("strict_fixed", strict_fixed, len(fixed_y)),
    ):
        assert_probability(name, value, rows)
    common_fold = np.load(strict_dir / "common_train_fold.npy", allow_pickle=False).astype(np.int8)
    old_common_fold = np.load(old_dir / "common_train_fold.npy", allow_pickle=False).astype(np.int8)
    raw_common_fold = np.load(raw_dir / "common_train_selector_fold.npy", allow_pickle=False).astype(np.int8)
    if common_fold.shape != (len(common_y),) or np.any(common_fold < 0) or np.unique(common_fold).size != 5:
        raise ValueError("Strict common fold assignment is invalid")
    if old_common_fold.shape != common_fold.shape or raw_common_fold.shape != common_fold.shape:
        raise ValueError("Old/raw common fold assignment has an unexpected shape")
    # The component OOF tables must use the identical held session partition.
    # If a component was trained on a pair-gate held session, its prediction
    # could make the second-level gate look stronger than a deployable stack.
    # Therefore a label permutation or merely similar fold sizes is not
    # sufficient here: the row-wise assignments must match exactly.
    component_fold_exact_equal = bool(
        np.array_equal(common_fold, old_common_fold) and np.array_equal(common_fold, raw_common_fold)
    )
    if not component_fold_exact_equal:
        raise ValueError(
            "Strict/old/raw common OOF held-session assignments differ; refuse fixed pair-gate promotion"
        )
    common_groups = groups[common_index]
    for held_value in np.unique(common_fold):
        if set(common_groups[common_fold == held_value]).intersection(set(common_groups[common_fold != held_value])):
            raise ValueError("Strict common fold assignment splits a session")

    # The 50:50 old/strict GRU mixture is predeclared; there is no weight
    # tuning in this audit.  Only the small finite gate family is selected.
    common_base = normalized_geometric(raw_common, old_common, common_ids).argmax(axis=1).astype(np.int16)
    common_dual = normalized_geometric(raw_common, 0.5 * old_common + 0.5 * strict_common, common_ids).argmax(axis=1).astype(np.int16)
    matching_base = normalized_geometric(raw_fixed, old_fixed, fixed_ids).argmax(axis=1).astype(np.int16)
    matching_dual = normalized_geometric(raw_fixed, 0.5 * old_fixed + 0.5 * strict_fixed, fixed_ids).argmax(axis=1).astype(np.int16)
    selected_spec, common_table, common_candidates, common_pairs = select_common_policy(
        common_base, common_dual, common_y, common_terminal, common_fold,
    )
    matching_candidate = apply_gate(matching_base, matching_dual, fixed_terminal, common_table)
    matching_baseline_metrics = metrics(fixed_y, matching_base, fixed_terminal, char_fixed)
    matching_candidate_metrics = metrics(fixed_y, matching_candidate, fixed_terminal, char_fixed)
    matching_delta = metric_delta(matching_candidate_metrics, matching_baseline_metrics)
    matching_bootstrap = terminal_session_bootstrap(
        fixed_y,
        with_char20(matching_base, fixed_terminal, char_fixed),
        with_char20(matching_candidate, fixed_terminal, char_fixed),
        fixed_terminal, groups[fixed_index], args.replicates, args.fixed_bootstrap_seed,
    )

    # This is a diagnostic-only bridge to the exact B43 fixed probability
    # tables.  It does not contribute to promotion because the historical
    # fixed models chose their epoch on that split.  It does show whether the
    # common-only learned pair policy maps sensibly to the byte-exact B43
    # fallback/dual prediction pairs.
    bridge_candidate = apply_gate(exact_base_fixed, exact_dual_fixed, fixed_terminal, common_table)
    bridge_baseline_metrics = metrics(fixed_y, exact_base_fixed, fixed_terminal, char_fixed)
    bridge_candidate_metrics = metrics(fixed_y, bridge_candidate, fixed_terminal, char_fixed)
    bridge_delta = metric_delta(bridge_candidate_metrics, bridge_baseline_metrics)

    required_metrics = tuple(expected)
    full_outer_positive = {
        name: int(sum(record["held_delta"][name] > 0.0 for record in full_records))
        for name in required_metrics
    }
    point_positive = all(full_delta[name] > 0.0 and matching_delta[name] > 0.0 for name in required_metrics)
    bootstrap_positive = (
        full_bootstrap["p_delta_gt_zero"] >= 0.95
        and full_bootstrap["p05_delta"] > 0.0
        and matching_bootstrap["p_delta_gt_zero"] >= 0.95
        and matching_bootstrap["p05_delta"] > 0.0
    )
    fold_positive = (
        full_outer_positive["all_fallback_macro_f1"] >= 4
        and full_outer_positive["terminal_with_char20_macro_f1"] >= 4
    )
    promotion_pass = bool(point_positive and bootstrap_positive and fold_positive)

    report: dict[str, Any] = {
        "scope": "Experiment-only strict audit. model/, script.py, build_zip.py, staging directories, and all ZIP submissions are unchanged.",
        "candidate_contract": {
            "base": "Exact B43 fallback: raw seed42 + original no-E5 GRU using geometric SIM=.20/AU=.85 blend.",
            "dual": "The same geometric raw blend with a fixed arithmetic 50:50 old no-E5 / strict no-E5 GRU mixture.",
            "gate_runtime_inputs": ["terminal_marker", "base_fallback_class", "dual_fallback_class"],
            "gate_runtime_rule": "On terminal rows only, use the dual class only for an allowed ordered (base_class, dual_class) pair; otherwise retain the base class.",
            "explicitly_excluded_from_gate": ["IDs", "session IDs", "prompt/text", "future history", "char20", "labels", "probability confidences", "source type"],
            "postprocessing": "char20 is immutable and is applied only after fallback prediction; future-history recovery appears only in offline hybrid scoring.",
            "finite_specs": [asdict(spec) for spec in SPECS],
        },
        "validation_protocol": {
            "full": "Five whole-session outer folds. Each held fold selects the finite capacity family in the other four folds by unoverridden terminal macro-F1, then all-row fallback macro-F1, then conservative tie break.",
            "matching_fixed": "Gate capacity and final pair table use only strict common-train OOF labels. Fixed common-validation labels are read only after predictions are formed.",
            "strict_model_note": "The matching fixed protocol uses a fresh common-train-only no-E5 old/strict homolog. It is deliberately not claimed byte-identical to public B43 fixed models, which used fixed-split epoch selection.",
            "bridge_note": "The exact-B43 fixed bridge is diagnostic only and is excluded from the promotion decision.",
        },
        "alignment": {
            "full_rows": int(len(y)), "full_terminal": int(terminal.sum()),
            "common_train_rows": int(len(common_y)), "common_train_terminal": int(common_terminal.sum()),
            "fixed_rows": int(len(fixed_y)), "fixed_terminal": int(fixed_terminal.sum()),
            "full_session_fold_integrity": True,
            "common_session_fold_integrity": True,
            "strict_old_raw_common_fold_assignment_exact_equal": component_fold_exact_equal,
            "strict_old_raw_common_fold_values": {
                "strict": {str(value): int((common_fold == value).sum()) for value in np.unique(common_fold)},
                "old": {str(value): int((old_common_fold == value).sum()) for value in np.unique(old_common_fold)},
                "raw": {str(value): int((raw_common_fold == value).sum()) for value in np.unique(raw_common_fold)},
            },
        },
        "exact_b43_full_nested": {
            "baseline": full_baseline_metrics,
            "candidate": full_candidate_metrics,
            "delta": full_delta,
            "terminal_char20_session_bootstrap": full_bootstrap,
            "outer_selection": full_records,
        },
        "matching_strict_common_fixed": {
            "baseline": matching_baseline_metrics,
            "candidate": matching_candidate_metrics,
            "delta": matching_delta,
            "terminal_char20_session_bootstrap": matching_bootstrap,
            "common_train_selection": {
                "chosen_spec": selected_spec.key,
                "selected_pair_count": int(common_table.sum()),
                "selected_pairs": common_pairs,
                "second_level_oof_candidates": common_candidates,
            },
        },
        "diagnostic_exact_b43_fixed_bridge_not_for_promotion": {
            "baseline": bridge_baseline_metrics,
            "candidate": bridge_candidate_metrics,
            "delta": bridge_delta,
            "common_only_policy_spec": selected_spec.key,
            "common_only_policy_pair_count": int(common_table.sum()),
        },
        "promotion_gate": {
            "criteria": "All four full and matching-fixed deltas >0; both char20-final terminal session bootstraps P(delta>0)>=.95 and p05>0; at least 4/5 full outer folds positive for all fallback and terminal-char20.",
            "full_outer_positive_fold_count": full_outer_positive,
            "all_eight_point_deltas_positive": point_positive,
            "both_bootstraps_positive": bootstrap_positive,
            "full_outer_stability_positive": fold_positive,
            "pass": promotion_pass,
            "decision": (
                "PROMOTION ELIGIBLE FOR A SEPARATE BUILD REVIEW; no production file was modified by this audit."
                if promotion_pass else
                "DO NOT PROMOTE: the strict two-protocol stability gate did not clear. No submission artifact was changed."
            ),
        },
        "deployment_feasibility": (
            {
                "eligible": True,
                "artifact_requirement": "Create a train-only 14x14 boolean pair table after the approved fixed capacity is refit on labeled training data; store it as model metadata, not as ID-keyed data.",
                "runtime_requirement": "Compute the exact B43 fallback and the fixed 50:50 dual fallback; on supplied-session terminal rows only, look up table[base_class, dual_class]. No labels, IDs, prompts, char20 values, or future history may enter the lookup.",
                "order_requirement": "Apply the lookup after the raw/GRU fallback argmax and before the immutable char20 terminal overlay; retain the established future-history recovery order.",
                "integration_plan": "A separate build review must add the table and the already-required dual probability path to the model bundle, update script.py only under this contract, then run the existing build_zip.py unchanged and verify its produced archive. This audit itself neither stages nor packages anything.",
            }
            if promotion_pass else
            {
                "eligible": False,
                "integration_plan": None,
                "reason": "No integration plan is authorized unless every strict promotion criterion passes.",
            }
        ),
    }
    (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    np.savez_compressed(
        out / "predictions.npz",
        y=y,
        terminal=terminal,
        full_fold=full_fold,
        exact_b43_full_baseline=exact_base_full,
        exact_b43_full_candidate=full_candidate,
        fixed_index=fixed_index,
        fixed_terminal=fixed_terminal,
        matching_fixed_baseline=matching_base,
        matching_fixed_candidate=matching_candidate,
        bridge_exact_b43_fixed_baseline=exact_base_fixed,
        bridge_exact_b43_fixed_candidate=bridge_candidate,
        char_full=char_full,
        char_fixed=char_fixed,
    )
    print(json.dumps({
        "promotion_pass": promotion_pass,
        "full_delta": full_delta,
        "matching_fixed_delta": matching_delta,
        "bridge_exact_b43_fixed_delta": bridge_delta,
        "common_selected_spec": selected_spec.key,
        "full_bootstrap": full_bootstrap,
        "matching_bootstrap": matching_bootstrap,
    }, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
