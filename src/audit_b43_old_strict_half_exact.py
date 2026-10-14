#!/usr/bin/env python3
"""Exact probability-only audit of the fixed old/strict 50:50 B43 variant.

The only candidate here is the predeclared arithmetic mixture

    p_gru = .5 * p_old_noe5 + .5 * p_strict_noe5

followed by the public B43 raw seed-42 geometric fusion (SIM=.20/AU=.85),
then the immutable char20 terminal overlay.  It is deliberately not a model
search: it changes no alpha, rule, threshold, or selector and it writes no
live model or submission archive.  The report separately marks portable
candidate-runtime evidence as provisional until B42 archive parity is checked.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold


ROOT = Path(__file__).resolve().parents[1]
CLASSES = (
    "read_file", "grep_search", "list_directory", "glob_pattern",
    "edit_file", "write_file", "apply_patch", "run_bash", "run_tests",
    "lint_or_typecheck", "ask_user", "plan_task", "web_search", "respond_only",
)
TO_ID = {name: index for index, name in enumerate(CLASSES)}
N_CLASSES = len(CLASSES)
EPS = 1.0e-12
MIXTURE_OLD_WEIGHT = 0.50
MIXTURE_STRICT_WEIGHT = 0.50
BOOTSTRAP_REPLICATES = 5000
FULL_EXPECTED_B43 = {
    "all_fallback_macro_f1": 0.7603385811033483,
    "terminal_fallback_macro_f1": 0.7312748947080282,
    "terminal_with_char20_macro_f1": 0.7335225252513891,
    "hybrid_future_history_plus_char20_macro_f1": 0.98277826622222,
}
FIXED_EXPECTED_B43 = {
    "all_fallback_macro_f1": 0.7585157741730599,
    "terminal_fallback_macro_f1": 0.6969994677821684,
    "terminal_with_char20_macro_f1": 0.7000338475808981,
    "hybrid_future_history_plus_char20_macro_f1": 0.9853915549361344,
}
METRIC_KEYS = tuple(FULL_EXPECTED_B43)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out-dir",
        default="experiments/b43_old_strict_half_exact_audit",
        help="Isolated experiment output directory (never model/ or a submission ZIP).",
    )
    parser.add_argument("--bootstrap-replicates", type=int, default=BOOTSTRAP_REPLICATES)
    return parser.parse_args()


def session_of(sample_id: str) -> str:
    return sample_id.rsplit("-step_", 1)[0]


def terminal_mask(ids: np.ndarray) -> np.ndarray:
    maximum: dict[str, int] = {}
    parsed: list[tuple[str, int]] = []
    for sample_id in ids.astype(str):
        group, raw_step = sample_id.rsplit("-step_", 1)
        step = int(raw_step)
        maximum[group] = max(maximum.get(group, -1), step)
        parsed.append((group, step))
    return np.asarray([step == maximum[group] for group, step in parsed], dtype=bool)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_alpha(ids: np.ndarray) -> np.ndarray:
    source = np.asarray(["au" if value.startswith("sess_au_") else "sim" for value in ids.astype(str)])
    return np.where(source == "au", 0.85, 0.20).astype(np.float64)


def geometric(raw: np.ndarray, gru: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    value = alpha[:, None] * np.log(np.maximum(raw, EPS))
    value += (1.0 - alpha[:, None]) * np.log(np.maximum(gru, EPS))
    value -= value.max(axis=1, keepdims=True)
    value = np.exp(value)
    return (value / value.sum(axis=1, keepdims=True)).astype(np.float32)


def labels_with_char(
    probability: np.ndarray, terminal: np.ndarray, char20: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    fallback = probability.argmax(axis=1).astype(np.int16)
    final = fallback.copy()
    override = terminal & (char20 >= 0)
    final[override] = char20[override]
    return fallback, final


def macro(y: np.ndarray, prediction: np.ndarray) -> float:
    return float(f1_score(
        y, prediction, labels=np.arange(N_CLASSES), average="macro", zero_division=0,
    ))


def metric_bundle(
    y: np.ndarray, probability: np.ndarray, terminal: np.ndarray, char20: np.ndarray,
) -> dict[str, float]:
    fallback, final = labels_with_char(probability, terminal, char20)
    hybrid = final.copy()
    hybrid[~terminal] = y[~terminal]
    return {
        "all_fallback_macro_f1": macro(y, fallback),
        "terminal_fallback_macro_f1": macro(y[terminal], fallback[terminal]),
        "terminal_with_char20_macro_f1": macro(y[terminal], final[terminal]),
        "hybrid_future_history_plus_char20_macro_f1": macro(y, hybrid),
    }


def metric_delta(candidate: dict[str, float], baseline: dict[str, float]) -> dict[str, float]:
    return {key: float(candidate[key] - baseline[key]) for key in METRIC_KEYS}


def assert_control(
    actual: dict[str, float], expected: dict[str, float], protocol: str,
) -> None:
    mismatch = {
        key: (actual[key], expected[key])
        for key in METRIC_KEYS
        if abs(actual[key] - expected[key]) > 1.0e-12
    }
    if mismatch:
        raise ValueError(f"{protocol} exact old-B43 control mismatch: {mismatch}")


def weighted_macro(confusion: np.ndarray) -> float:
    true_positive = np.diag(confusion).astype(np.float64)
    denominator = confusion.sum(axis=0) + confusion.sum(axis=1)
    return float(np.mean(np.divide(
        2.0 * true_positive,
        denominator,
        out=np.zeros(N_CLASSES, dtype=np.float64),
        where=denominator > 0.0,
    )))


def session_bootstrap(
    y: np.ndarray,
    baseline: np.ndarray,
    candidate: np.ndarray,
    groups: np.ndarray,
    repeats: int,
    seed: int,
) -> dict[str, float | int]:
    """Whole-session bootstrap after char20; no row-wise pseudo-bootstrap."""
    unique, inverse = np.unique(groups, return_inverse=True)
    if len(unique) != len(groups):
        raise ValueError("Terminal session bootstrap requires one supplied terminal per session")
    baseline_cell = y.astype(np.int64) * N_CLASSES + baseline.astype(np.int64)
    candidate_cell = y.astype(np.int64) * N_CLASSES + candidate.astype(np.int64)
    random = np.random.default_rng(seed)
    deltas = np.empty(repeats, dtype=np.float64)
    for number in range(repeats):
        sampled = random.integers(0, len(unique), size=len(unique))
        session_count = np.bincount(sampled, minlength=len(unique))
        row_weight = session_count[inverse]
        base_confusion = np.bincount(
            baseline_cell, weights=row_weight, minlength=N_CLASSES * N_CLASSES,
        ).reshape(N_CLASSES, N_CLASSES)
        candidate_confusion = np.bincount(
            candidate_cell, weights=row_weight, minlength=N_CLASSES * N_CLASSES,
        ).reshape(N_CLASSES, N_CLASSES)
        deltas[number] = weighted_macro(candidate_confusion) - weighted_macro(base_confusion)
    exact = weighted_macro(np.bincount(
        candidate_cell, minlength=N_CLASSES * N_CLASSES,
    ).reshape(N_CLASSES, N_CLASSES)) - weighted_macro(np.bincount(
        baseline_cell, minlength=N_CLASSES * N_CLASSES,
    ).reshape(N_CLASSES, N_CLASSES))
    return {
        "replicates": int(repeats),
        "seed": int(seed),
        "sessions": int(len(unique)),
        "selected_rows": int(len(groups)),
        "exact_delta": float(exact),
        "mean_delta": float(deltas.mean()),
        "p_delta_gt_zero": float(np.mean(deltas > 0.0)),
        "p05_delta": float(np.quantile(deltas, 0.05)),
        "p95_delta": float(np.quantile(deltas, 0.95)),
    }


def fold_deltas(
    y: np.ndarray,
    baseline: np.ndarray,
    candidate: np.ndarray,
    terminal: np.ndarray,
    char20: np.ndarray,
    groups: np.ndarray,
    fold: np.ndarray,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for held in sorted(np.unique(fold).tolist()):
        mask = fold == held
        base = metric_bundle(y[mask], baseline[mask], terminal[mask], char20[mask])
        dual = metric_bundle(y[mask], candidate[mask], terminal[mask], char20[mask])
        rows.append({
            "held_session_fold": int(held) + 1,
            "rows": int(mask.sum()),
            "sessions": int(len(np.unique(groups[mask]))),
            "terminal_rows": int(terminal[mask].sum()),
            "old_b43": base,
            "old_strict_half": dual,
            "delta": metric_delta(dual, base),
        })
    return rows


def provisional_candidate_runtime(root: Path) -> dict[str, Any]:
    """Check only experimental portable assets; do not infer B42 archive parity."""
    from numpy_sequence_fusion import load_weights, probabilities

    old_dir = root / "experiments" / "modeling_ai_sequence_fusion_noe5_full"
    strict_dir = root / "experiments" / "modeling_ai_sequence_fusion_noe5_strictdistill035_full"
    old_arch_path = old_dir / "architecture.json"
    strict_arch_path = strict_dir / "architecture.json"
    old_weights_path = old_dir / "weights.npz"
    strict_weights_path = strict_dir / "weights.npz"
    old_probe_path = old_dir / "verification_probe.npz"
    strict_probe_path = strict_dir / "verification_probe.npz"
    old_arch = json.loads(old_arch_path.read_text(encoding="utf-8"))
    strict_arch = json.loads(strict_arch_path.read_text(encoding="utf-8"))
    schema_keys = ("format", "all_classes", "input", "architecture")
    schema_match = {key: old_arch.get(key) == strict_arch.get(key) for key in schema_keys}
    if not all(schema_match.values()):
        raise ValueError(f"Candidate old/strict portable schemas differ: {schema_match}")

    expected_probe = ("action", "status", "arg", "base", "meta", "torch_probabilities")
    old_probe = np.load(old_probe_path, allow_pickle=False)
    strict_probe = np.load(strict_probe_path, allow_pickle=False)
    if tuple(old_probe.files) != expected_probe or tuple(strict_probe.files) != expected_probe:
        raise ValueError("Unexpected portable inference verification-probe schema")
    old_weights = load_weights(old_weights_path)
    strict_weights = load_weights(strict_weights_path)

    def verify(probe: Any, weights: dict[str, np.ndarray]) -> dict[str, Any]:
        inputs = [probe[name] for name in expected_probe[:5]]
        reproduced = probabilities(*inputs, weights)
        target = probe["torch_probabilities"]
        return {
            "rows": int(len(reproduced)),
            "max_abs_probability_difference": float(np.max(np.abs(reproduced - target))),
            "argmax_agreement": float(np.mean(reproduced.argmax(axis=1) == target.argmax(axis=1))),
        }

    old_verify = verify(old_probe, old_weights)
    strict_verify = verify(strict_probe, strict_weights)
    timings: list[float] = []
    rows = 12000
    repeat = (rows + len(old_probe["action"]) - 1) // len(old_probe["action"])
    inputs = [
        np.concatenate([old_probe[name]] * repeat, axis=0)[:rows]
        for name in expected_probe[:5]
    ]
    probabilities(*inputs, old_weights)  # warm-up
    probabilities(*inputs, strict_weights)
    max_sum_error = 0.0
    for _ in range(3):
        started = time.perf_counter()
        old_probability = probabilities(*inputs, old_weights)
        strict_probability = probabilities(*inputs, strict_weights)
        mixture = (MIXTURE_OLD_WEIGHT * old_probability + MIXTURE_STRICT_WEIGHT * strict_probability).astype(np.float32)
        timings.append(time.perf_counter() - started)
        max_sum_error = max(max_sum_error, float(np.max(np.abs(mixture.sum(axis=1) - 1.0))))

    def file_info(path: Path) -> dict[str, Any]:
        return {
            "path": str(path.relative_to(root)).replace("\\", "/"),
            "bytes": int(path.stat().st_size),
            "sha256": sha256(path),
        }

    return {
        "scope": "Experimental full-refit portable assets only; B42.zip and workspace model/ were not read as deployment sources.",
        "archive_deployment_parity": "pending B42 archive weight/schema verification by the parent task; this result is not a deployability claim.",
        "schema_match": schema_match,
        "old_probe_verification": old_verify,
        "strict_probe_verification": strict_verify,
        "candidate_artifacts": {
            "old_weights": file_info(old_weights_path),
            "old_architecture": file_info(old_arch_path),
            "strict_weights": file_info(strict_weights_path),
            "strict_architecture": file_info(strict_arch_path),
            "numpy_forward": file_info(root / "src" / "numpy_sequence_fusion.py"),
        },
        "cpu_pair_forward_12000_rows": {
            "measurement": "Warm CPU-only NumPy/SciPy forward of both experimental GRUs plus arithmetic mixture; excludes parsing, features, raw selector, and terminal rules.",
            "three_seconds": [float(value) for value in timings],
            "median_seconds": float(np.median(timings)),
            "mixture_probability_sum_max_abs_error": max_sum_error,
        },
    }


def main() -> None:
    args = parse_args()
    if args.bootstrap_replicates <= 0:
        raise ValueError("--bootstrap-replicates must be positive")
    out = Path(args.out_dir)
    if not out.is_absolute():
        out = ROOT / out
    out = out.resolve()
    model_dir = (ROOT / "model").resolve()
    if out == model_dir or model_dir in out.parents:
        raise ValueError("Experiment output cannot be written inside model/")
    out.mkdir(parents=True, exist_ok=True)
    began = time.perf_counter()

    train_rows = [
        json.loads(line) for line in (ROOT / "data" / "train.jsonl").open(encoding="utf-8")
        if line.strip()
    ]
    labels = {
        row["id"]: row["action"]
        for row in csv.DictReader((ROOT / "data" / "train_labels.csv").open(encoding="utf-8"))
    }
    split = {
        row["id"]: row["split"]
        for row in csv.DictReader((ROOT / "splits" / "common_split.csv").open(encoding="utf-8"))
    }
    ids_global = np.asarray([str(row["id"]) for row in train_rows])
    y = np.asarray([TO_ID[labels[value]] for value in ids_global], dtype=np.int16)
    groups = np.asarray([session_of(value) for value in ids_global])
    terminal = terminal_mask(ids_global)
    alpha = source_alpha(ids_global)
    fixed_index = np.flatnonzero(np.asarray([split[value] == "val" for value in ids_global]))
    fixed_ids = ids_global[fixed_index]
    fixed_y = y[fixed_index]
    fixed_groups = groups[fixed_index]
    fixed_terminal = terminal[fixed_index]

    experiment = ROOT / "experiments"
    paths = {
        "full_raw_seed42": experiment / "xgb_robust_ensemble" / "full_oof_raw_seed42.npy",
        "full_old_noe5_gru": experiment / "modeling_ai_sequence_fusion_noe5" / "oof_probabilities.npy",
        "full_strict_noe5_gru": experiment / "modeling_ai_sequence_nested_distill_e16_i4" / "oof_probabilities.npy",
        "strict_outer_fold_assignment": experiment / "modeling_ai_sequence_nested_distill_e16_i4" / "outer_fold_assignment.npy",
        "fixed_raw_seed42": experiment / "xgb_robust_ensemble" / "strict_fixed_raw_seed42.npy",
        "fixed_old_noe5_gru": experiment / "modeling_ai_sequence_fusion_fixed_noe5" / "valid_probabilities.npy",
        "fixed_old_indices": experiment / "modeling_ai_sequence_fusion_fixed_noe5" / "valid_indices.npy",
        "fixed_old_terminal": experiment / "modeling_ai_sequence_fusion_fixed_noe5" / "terminal_mask.npy",
        "fixed_strict_noe5_gru": experiment / "modeling_ai_sequence_fusion_fixed_strictdistill_w35" / "valid_probabilities.npy",
        "fixed_strict_indices": experiment / "modeling_ai_sequence_fusion_fixed_strictdistill_w35" / "valid_indices.npy",
        "fixed_strict_terminal": experiment / "modeling_ai_sequence_fusion_fixed_strictdistill_w35" / "terminal_mask.npy",
        "char20": experiment / "terminal_rule_combinations" / "oof_candidates.npz",
    }
    raw = np.load(paths["full_raw_seed42"], allow_pickle=False).astype(np.float32)
    old = np.load(paths["full_old_noe5_gru"], allow_pickle=False).astype(np.float32)
    strict = np.load(paths["full_strict_noe5_gru"], allow_pickle=False).astype(np.float32)
    if not (raw.shape == old.shape == strict.shape == (len(y), N_CLASSES)):
        raise ValueError("Full raw/old/strict component shape mismatch")

    fold = np.empty(len(y), dtype=np.int8)
    for held_number, (_, held_index) in enumerate(GroupKFold(n_splits=5).split(np.zeros(len(y)), y, groups)):
        fold[held_index] = held_number
    if not np.array_equal(fold, np.load(paths["strict_outer_fold_assignment"], allow_pickle=False).astype(np.int8)):
        raise ValueError("Strict OOF is not aligned with the required five-fold whole-session GroupKFold")

    char = np.load(paths["char20"], allow_pickle=False)
    if not (
        np.array_equal(char["y"].astype(np.int16), y)
        and np.array_equal(char["terminal"].astype(bool), terminal)
        and np.array_equal(char["fold"].astype(np.int8)[terminal], fold[terminal])
    ):
        raise ValueError("Full char20 artifact does not align with the exact B43 protocol")
    char_full = char["char20"].astype(np.int16)

    raw_fixed = np.load(paths["fixed_raw_seed42"], allow_pickle=False).astype(np.float32)
    old_fixed = np.load(paths["fixed_old_noe5_gru"], allow_pickle=False).astype(np.float32)
    strict_fixed = np.load(paths["fixed_strict_noe5_gru"], allow_pickle=False).astype(np.float32)
    old_index = np.load(paths["fixed_old_indices"], allow_pickle=False).astype(np.int64)
    strict_index = np.load(paths["fixed_strict_indices"], allow_pickle=False).astype(np.int64)
    old_terminal = np.load(paths["fixed_old_terminal"], allow_pickle=False).astype(bool)
    strict_terminal = np.load(paths["fixed_strict_terminal"], allow_pickle=False).astype(bool)
    if not (
        raw_fixed.shape == old_fixed.shape == strict_fixed.shape == (len(fixed_index), N_CLASSES)
        and np.array_equal(old_index, fixed_index)
        and np.array_equal(strict_index, fixed_index)
        and np.array_equal(old_terminal, fixed_terminal)
        and np.array_equal(strict_terminal, fixed_terminal)
    ):
        raise ValueError("Matching seed-42 fixed raw/old/strict alignment failed")
    char_fixed = np.full(len(fixed_index), -1, dtype=np.int16)
    char_fixed[fixed_terminal] = char["fixed_char20"].astype(np.int16)
    if not (
        np.array_equal(char["fixed_y"].astype(np.int16), fixed_y[fixed_terminal])
        and np.array_equal(char["valid_ids"].astype(str), fixed_ids[fixed_terminal])
    ):
        raise ValueError("Matching seed-42 fixed char20 alignment failed")

    base_full_probability = geometric(raw, old, alpha)
    base_fixed_probability = geometric(raw_fixed, old_fixed, alpha[fixed_index])
    dual_full_gru = (MIXTURE_OLD_WEIGHT * old + MIXTURE_STRICT_WEIGHT * strict).astype(np.float32)
    dual_fixed_gru = (MIXTURE_OLD_WEIGHT * old_fixed + MIXTURE_STRICT_WEIGHT * strict_fixed).astype(np.float32)
    dual_full_probability = geometric(raw, dual_full_gru, alpha)
    dual_fixed_probability = geometric(raw_fixed, dual_fixed_gru, alpha[fixed_index])

    baseline_full = metric_bundle(y, base_full_probability, terminal, char_full)
    baseline_fixed = metric_bundle(fixed_y, base_fixed_probability, fixed_terminal, char_fixed)
    assert_control(baseline_full, FULL_EXPECTED_B43, "full")
    assert_control(baseline_fixed, FIXED_EXPECTED_B43, "matching_seed42_fixed")
    candidate_full = metric_bundle(y, dual_full_probability, terminal, char_full)
    candidate_fixed = metric_bundle(fixed_y, dual_fixed_probability, fixed_terminal, char_fixed)
    full_delta = metric_delta(candidate_full, baseline_full)
    fixed_delta = metric_delta(candidate_fixed, baseline_fixed)

    base_full_terminal = labels_with_char(base_full_probability, terminal, char_full)[1][terminal]
    dual_full_terminal = labels_with_char(dual_full_probability, terminal, char_full)[1][terminal]
    base_fixed_terminal = labels_with_char(base_fixed_probability, fixed_terminal, char_fixed)[1][fixed_terminal]
    dual_fixed_terminal = labels_with_char(dual_fixed_probability, fixed_terminal, char_fixed)[1][fixed_terminal]
    full_bootstrap = session_bootstrap(
        y[terminal], base_full_terminal, dual_full_terminal, groups[terminal],
        args.bootstrap_replicates, seed=20260713,
    )
    fixed_bootstrap = session_bootstrap(
        fixed_y[fixed_terminal], base_fixed_terminal, dual_fixed_terminal, fixed_groups[fixed_terminal],
        args.bootstrap_replicates, seed=20260714,
    )
    held_fold_deltas = fold_deltas(
        y, base_full_probability, dual_full_probability, terminal, char_full, groups, fold,
    )
    positive_fold_counts = {
        key: int(sum(row["delta"][key] > 0.0 for row in held_fold_deltas))
        for key in METRIC_KEYS
    }
    all_eight_positive = all(value > 0.0 for value in full_delta.values()) and all(
        value > 0.0 for value in fixed_delta.values()
    )
    stability_gate = {
        "all_eight_full_and_fixed_metrics_strictly_positive": bool(all_eight_positive),
        "full_terminal_bootstrap_p_delta_gt_zero_ge_0_95": bool(full_bootstrap["p_delta_gt_zero"] >= 0.95),
        "fixed_terminal_bootstrap_p_delta_gt_zero_ge_0_95": bool(fixed_bootstrap["p_delta_gt_zero"] >= 0.95),
        "full_terminal_bootstrap_p05_delta_gt_zero": bool(full_bootstrap["p05_delta"] > 0.0),
        "fixed_terminal_bootstrap_p05_delta_gt_zero": bool(fixed_bootstrap["p05_delta"] > 0.0),
        "full_all_fallback_positive_in_at_least_4_of_5_session_folds": bool(
            positive_fold_counts["all_fallback_macro_f1"] >= 4
        ),
        "full_terminal_char20_positive_in_at_least_4_of_5_session_folds": bool(
            positive_fold_counts["terminal_with_char20_macro_f1"] >= 4
        ),
    }
    runtime = provisional_candidate_runtime(ROOT)
    promotion_ready = bool(all(stability_gate.values()))

    report = {
        "scope": "Probability-only audit. model/, script.py, B42.zip, baseline_submit.zip, and all challenger ZIPs were not modified.",
        "candidate_contract": {
            "name": "fixed_arithmetic_old_noe5_0.50_plus_strict_noe5_0.50",
            "gru_mixture": {"old_weight": MIXTURE_OLD_WEIGHT, "strict_weight": MIXTURE_STRICT_WEIGHT},
            "raw_fusion": "Exact public B43 normalized geometric raw seed42/GRU blend: SIM raw alpha=.20, AU raw alpha=.85.",
            "postprocessing": "Exact B43 char20 terminal overlay is applied after argmax; future-history recovery appears only in the hybrid evaluation metric.",
            "excluded": "No alpha tuning, selection/model change, threshold, calibration, source gate, ID feature, target lookup, or future-label input.",
        },
        "component_provenance": {
            name: {
                "path": str(path.relative_to(ROOT)).replace("\\", "/"),
                "sha256": sha256(path),
                "bytes": int(path.stat().st_size),
            }
            for name, path in paths.items()
        },
        "alignment_checks": {
            "full_rows": int(len(y)),
            "fixed_rows": int(len(fixed_y)),
            "full_terminal_rows": int(terminal.sum()),
            "fixed_terminal_rows": int(fixed_terminal.sum()),
            "strict_full_outer_fold_matches_groupkfold": True,
            "fixed_old_and_strict_indices_match_common_val": True,
            "fixed_old_and_strict_terminal_masks_match": True,
            "char20_full_and_fixed_labels_ids_match": True,
        },
        "old_b43_control": {
            "full_outer_oof": baseline_full,
            "matching_seed42_fixed": baseline_fixed,
            "expected_controls": {"full_outer_oof": FULL_EXPECTED_B43, "matching_seed42_fixed": FIXED_EXPECTED_B43},
        },
        "fixed_half_candidate": {
            "full_outer_oof": candidate_full,
            "matching_seed42_fixed": candidate_fixed,
            "delta_vs_old_b43": {
                "full_outer_oof": full_delta,
                "matching_seed42_fixed": fixed_delta,
            },
            "terminal_char20_session_bootstrap": {
                "full_outer_oof": full_bootstrap,
                "matching_seed42_fixed": fixed_bootstrap,
            },
        },
        "whole_session_selection": {
            "applicable": False,
            "reason": "There is exactly one predeclared 50:50 arithmetic candidate, so no candidate/policy/weight selection is performed. The five held whole-session folds are reported only as stability diagnostics.",
            "held_fold_deltas": held_fold_deltas,
            "positive_fold_counts": positive_fold_counts,
        },
        "stability_gate": {
            "criteria": "All eight point-estimate metrics positive; both terminal-char20 session bootstraps P(delta>0)>=.95 and p05>0; at least 4/5 held session folds positive for full all-fallback and terminal-char20.",
            "checks": stability_gate,
            "pass": promotion_ready,
        },
        "provisional_runtime_feasibility": runtime,
        "submission_action": {
            "candidate_created": False,
            "zip_updated": False,
            "reason": "This bounded audit intentionally does not build a ZIP. Even a statistical pass remains pending B42 archive model-weight/schema parity verification.",
        },
        "elapsed_seconds": round(time.perf_counter() - began, 3),
    }
    np.savez_compressed(
        out / "probability_audit.npz",
        y=y,
        terminal=terminal,
        fixed_index=fixed_index,
        fixed_terminal=fixed_terminal,
        base_full_probability=base_full_probability,
        dual_full_probability=dual_full_probability,
        base_fixed_probability=base_fixed_probability,
        dual_fixed_probability=dual_fixed_probability,
        char_full=char_full,
        char_fixed=char_fixed,
    )
    (out / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    (out / "README.md").write_text(
        "# Exact old-B43 + strict 50:50 probability audit\n\n"
        "This directory is experiment-only. It contains no submission archive and does not modify `model/` or `script.py`. "
        "The report validates the exact historical B43 controls before evaluating the one fixed arithmetic 50:50 GRU mixture. "
        "Portable pair-runtime evidence is provisional until B42 archive parity is separately verified.\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "output": str(out),
        "full_delta": full_delta,
        "fixed_delta": fixed_delta,
        "full_bootstrap": full_bootstrap,
        "fixed_bootstrap": fixed_bootstrap,
        "stability_gate_pass": promotion_ready,
        "seconds": report["elapsed_seconds"],
    }, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
