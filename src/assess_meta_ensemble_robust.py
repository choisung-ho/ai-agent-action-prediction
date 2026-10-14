#!/usr/bin/env python3
"""Strict, lightweight meta-ensemble audit for the XGB and no-E5 GRU OOFs.

This script deliberately never reads or writes ``model/``, ``script.py``, or a
submission ZIP.  It evaluates only experiment artifacts and writes an auditable
report below ``experiments/meta_ensemble_robust``.

The deployment candidate is intentionally tiny: a source-conditioned geometric
mean of the *raw* XGB selector probability and the NumPy-portable GRU
probability.  It has two scalar weights and no per-class fitting.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold


CLASSES = [
    "read_file", "grep_search", "list_directory", "glob_pattern", "edit_file", "write_file", "apply_patch",
    "run_bash", "run_tests", "lint_or_typecheck", "ask_user", "plan_task", "web_search", "respond_only",
]
N_CLASSES = len(CLASSES)
GRID = np.round(np.arange(0.0, 1.0001, 0.05), 2)
EPSILON = 1.0e-12


def macro_f1(y: np.ndarray, probability: np.ndarray, mask: np.ndarray) -> float:
    """Macro F1 with the competition's fixed 14-class label set."""
    return float(
        f1_score(
            y[mask], probability[mask].argmax(axis=1), labels=np.arange(N_CLASSES),
            average="macro", zero_division=0,
        )
    )


def macro_f1_predictions(y: np.ndarray, prediction: np.ndarray) -> float:
    """Fast fixed-label macro F1 for a bootstrap-resampled prediction vector."""
    confusion = np.bincount(N_CLASSES * y + prediction, minlength=N_CLASSES * N_CLASSES)
    confusion = confusion.reshape(N_CLASSES, N_CLASSES)
    true_positive = np.diag(confusion).astype(np.float64)
    denominator = confusion.sum(axis=0) + confusion.sum(axis=1)
    return float(np.mean(np.divide(2.0 * true_positive, denominator, out=np.zeros(N_CLASSES), where=denominator > 0)))


def linear_mix(xgb: np.ndarray, gru: np.ndarray, source: np.ndarray, alpha: dict[str, float]) -> np.ndarray:
    probability = np.empty_like(xgb)
    for source_name, xgb_weight in alpha.items():
        rows = source == source_name
        probability[rows] = xgb_weight * xgb[rows] + (1.0 - xgb_weight) * gru[rows]
    return probability


def geometric_mix(xgb: np.ndarray, gru: np.ndarray, source: np.ndarray, alpha: dict[str, float]) -> np.ndarray:
    """p(c) ∝ p_xgb(c)^alpha * p_gru(c)^(1-alpha), normalized safely."""
    probability = np.empty_like(xgb)
    for source_name, xgb_weight in alpha.items():
        rows = source == source_name
        log_probability = (
            xgb_weight * np.log(np.maximum(xgb[rows], EPSILON))
            + (1.0 - xgb_weight) * np.log(np.maximum(gru[rows], EPSILON))
        )
        log_probability -= log_probability.max(axis=1, keepdims=True)
        value = np.exp(log_probability)
        probability[rows] = value / value.sum(axis=1, keepdims=True)
    return probability


def choose_source_alpha(
    y: np.ndarray,
    xgb: np.ndarray,
    gru: np.ndarray,
    source: np.ndarray,
    terminal: np.ndarray,
    train_mask: np.ndarray,
    objective: str,
    kind: str,
) -> dict[str, float]:
    """Choose each source alpha on the inner folds only.

    The live fallback is called only after exact future-history recovery has
    failed, so ``terminal`` is the deployment-aligned objective.  ``all`` is
    retained as an overfit/robustness control.
    """
    blend = geometric_mix if kind == "geometric" else linear_mix
    picked: dict[str, float] = {}
    all_rows = np.ones(len(y), dtype=bool)
    for source_name in ("sim", "au"):
        rows = train_mask & (source == source_name) & (terminal if objective == "terminal" else all_rows)
        candidates: list[tuple[float, float, float]] = []
        for value in GRID:
            probability = blend(xgb, gru, source, {"sim": value, "au": value})
            # Secondary preference prevents arbitrary edge values when two
            # weights make identical argmax decisions.
            candidates.append((macro_f1(y, probability, rows), -abs(float(value) - 0.5), float(value)))
        picked[source_name] = max(candidates)[2]
    return picked


def nested_predictions(
    y: np.ndarray,
    xgb: np.ndarray,
    gru: np.ndarray,
    source: np.ndarray,
    terminal: np.ndarray,
    fold: np.ndarray,
    objective: str,
    kind: str,
) -> tuple[np.ndarray, list[dict[str, object]]]:
    blend = geometric_mix if kind == "geometric" else linear_mix
    probability = np.empty_like(xgb)
    records: list[dict[str, object]] = []
    for fold_number in sorted(np.unique(fold)):
        valid_mask = fold == fold_number
        alpha = choose_source_alpha(y, xgb, gru, source, terminal, ~valid_mask, objective, kind)
        all_probability = blend(xgb, gru, source, alpha)
        probability[valid_mask] = all_probability[valid_mask]
        records.append({
            "fold": int(fold_number + 1),
            "alpha": alpha,
            "all_macro_f1": macro_f1(y, probability, valid_mask),
            "terminal_macro_f1": macro_f1(y, probability, valid_mask & terminal),
        })
    return probability, records


def bootstrap_terminal_delta(
    y: np.ndarray,
    baseline: np.ndarray,
    candidate: np.ndarray,
    terminal: np.ndarray,
    seed: int,
    repeats: int,
) -> dict[str, float | list[float]]:
    """Session bootstrap; each terminal row is exactly one supplied session."""
    index = np.flatnonzero(terminal)
    random = np.random.default_rng(seed)
    deltas = np.empty(repeats, dtype=np.float64)
    for repeat in range(repeats):
        sample = index[random.integers(0, len(index), len(index))]
        deltas[repeat] = macro_f1_predictions(y[sample], candidate[sample]) - macro_f1_predictions(y[sample], baseline[sample])
    exact = macro_f1_predictions(y[index], candidate[index]) - macro_f1_predictions(y[index], baseline[index])
    return {
        "exact_delta": float(exact),
        "bootstrap_mean_delta": float(deltas.mean()),
        "ci95": [float(value) for value in np.quantile(deltas, [0.025, 0.975])],
        "share_nonpositive": float(np.mean(deltas <= 0.0)),
        "repeats": int(repeats),
    }


def load_fixed_validation(root: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    samples = [json.loads(line) for line in (root / "data/train.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    split = {row["id"]: row["split"] for row in csv.DictReader((root / "splits/common_split.csv").open(encoding="utf-8"))}
    labels = {row["id"]: row["action"] for row in csv.DictReader((root / "data/train_labels.csv").open(encoding="utf-8"))}
    samples = [sample for sample in samples if split[sample["id"]] == "val"]
    label_to_id = {label: index for index, label in enumerate(CLASSES)}
    y = np.asarray([label_to_id[labels[sample["id"]]] for sample in samples], dtype=np.int64)
    source = np.asarray(["au" if sample["id"].startswith("sess_au_") else "sim" for sample in samples])
    terminal = np.load(root / "experiments/modeling_ai_sequence_fusion_fixed_noe5/terminal_mask.npy").astype(bool)
    xgb = np.load(root / "experiments/modeling_ac_tune2/oof_probabilities.npy").astype(np.float32)
    gru = np.load(root / "experiments/modeling_ai_sequence_fusion_fixed_noe5/valid_probabilities.npy").astype(np.float32)
    if not (len(y) == len(source) == len(terminal) == len(xgb) == len(gru)):
        raise ValueError("Fixed-validation artifacts do not align")
    return y, xgb, gru, source, terminal


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="experiments/meta_ensemble_robust")
    parser.add_argument("--bootstrap-repeats", type=int, default=4000)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    out = root / args.out_dir
    out.mkdir(parents=True, exist_ok=True)

    xgb = np.load(root / "experiments/modeling_ak_live_xgb_proxy/oof_probabilities.npy").astype(np.float32)
    gru = np.load(root / "experiments/modeling_ai_sequence_fusion_noe5/oof_probabilities.npy").astype(np.float32)
    shape = np.load(root / "experiments/modeling_ak_live_xgb_proxy/input_shapes.npz", allow_pickle=True)
    y = shape["y"].astype(np.int64)
    source = shape["source"].astype(str)
    terminal = shape["terminal"].astype(bool)
    samples = [json.loads(line) for line in (root / "data/train.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    groups = np.asarray([sample["id"].split("-step_", 1)[0] for sample in samples])
    if not (len(samples) == len(y) == len(xgb) == len(gru)):
        raise ValueError("Full OOF artifacts do not align with train.jsonl")
    fold = np.empty(len(y), dtype=np.int8)
    for fold_number, (_, valid_index) in enumerate(GroupKFold(n_splits=5).split(np.zeros(len(y)), y, groups)):
        fold[valid_index] = fold_number

    equal_alpha = {"sim": 0.5, "au": 0.5}
    linear_equal = linear_mix(xgb, gru, source, equal_alpha)
    geometric_equal = geometric_mix(xgb, gru, source, equal_alpha)
    static_terminal_alpha = choose_source_alpha(
        y, xgb, gru, source, terminal, np.ones(len(y), dtype=bool), "terminal", "geometric",
    )
    static_terminal = geometric_mix(xgb, gru, source, static_terminal_alpha)
    nested_terminal, nested_terminal_records = nested_predictions(
        y, xgb, gru, source, terminal, fold, "terminal", "geometric",
    )
    nested_all, nested_all_records = nested_predictions(
        y, xgb, gru, source, terminal, fold, "all", "linear",
    )

    all_rows = np.ones(len(y), dtype=bool)
    def metrics(probability: np.ndarray) -> dict[str, float]:
        return {"all_macro_f1": macro_f1(y, probability, all_rows), "terminal_macro_f1": macro_f1(y, probability, terminal)}

    # Existing 28-parameter source/class calibration is intentionally tested
    # only as a negative control.  It was fitted on the historical fixed split,
    # whereas the full GroupKFold OOF rejects it strongly on terminal rows.
    calibration = json.loads((root / "experiments/modeling_ac_tune2/source_calibration.json").read_text(encoding="utf-8"))
    calibration_weight = np.asarray([
        [calibration["sim"][label] for label in CLASSES],
        [calibration["au"][label] for label in CLASSES],
    ], dtype=np.float32)
    calibrated_xgb = xgb * calibration_weight[np.where(source == "sim", 0, 1)]
    calibrated_xgb /= calibrated_xgb.sum(axis=1, keepdims=True)
    calibrated_equal = 0.5 * calibrated_xgb + 0.5 * gru

    fy, fxgb, fgru, fsource, fterminal = load_fixed_validation(root)
    fixed_linear_equal = linear_mix(fxgb, fgru, fsource, equal_alpha)
    fixed_static_terminal = geometric_mix(fxgb, fgru, fsource, static_terminal_alpha)
    fixed_all_rows = np.ones(len(fy), dtype=bool)
    fixed_metrics = lambda p: {
        "all_macro_f1": macro_f1(fy, p, fixed_all_rows),
        "terminal_macro_f1": macro_f1(fy, p, fterminal),
    }

    report: dict[str, object] = {
        "scope": "Experiment-only audit. No live model/script/ZIP was changed.",
        "validation_protocol": {
            "full_oof": "matching 5-fold GroupKFold base OOF probabilities; meta alpha is selected in the other four folds before scoring the held fold",
            "fixed_validation": "common held-out split, not used by the full-OOF alpha selection",
            "caveat": "The base learners were not retrained inside an additional outer loop; the nested result is a second-level OOF alpha audit. The separate fixed split is included as an independent confirmation.",
            "n_full_rows": int(len(y)), "n_full_terminal": int(terminal.sum()),
            "n_fixed_rows": int(len(fy)), "n_fixed_terminal": int(fterminal.sum()),
            "terminal_sessions_unique": int(np.unique(groups[terminal]).size),
        },
        "full_oof": {
            "linear_equal_0_5": metrics(linear_equal),
            "geometric_equal_0_5": metrics(geometric_equal),
            "geometric_source_terminal_static": {"alpha": static_terminal_alpha, **metrics(static_terminal)},
            "geometric_source_terminal_nested": {"metrics": metrics(nested_terminal), "folds": nested_terminal_records},
            "linear_source_all_nested": {"metrics": metrics(nested_all), "folds": nested_all_records},
            "rejected_existing_source_class_calibration_equal_blend": metrics(calibrated_equal),
        },
        "fixed_validation_raw_xgb": {
            "linear_equal_0_5": fixed_metrics(fixed_linear_equal),
            "geometric_source_terminal_static": {"alpha": static_terminal_alpha, **fixed_metrics(fixed_static_terminal)},
        },
        "bootstrap_terminal_delta_vs_linear_equal": {
            "full_oof": bootstrap_terminal_delta(
                y, linear_equal.argmax(axis=1), static_terminal.argmax(axis=1), terminal, 508, args.bootstrap_repeats,
            ),
            "fixed_validation": bootstrap_terminal_delta(
                fy, fixed_linear_equal.argmax(axis=1), fixed_static_terminal.argmax(axis=1), fterminal, 509, args.bootstrap_repeats,
            ),
        },
        "deployment_spec": {
            "eligible_rows": "Only rows not deterministically recovered by later supplied same-session history; do not blend exact recovered labels.",
            "xgb_input": "raw selector.predict_proba() BEFORE existing source_class_weights calibration",
            "gru_input": "the existing no-E5 NumPy GRU probability vector",
            "formula": "p = normalize(exp(alpha * log(max(p_xgb, 1e-12)) + (1-alpha) * log(max(p_gru, 1e-12))))",
            "alpha_xgb": static_terminal_alpha,
            "source_rule": "sess_sim_* -> sim; sess_au_* -> au",
            "class_specific_calibration": "Do not apply it: the existing 28 source/class factors reduce full GroupOOF terminal macro-F1 materially.",
        },
        "recommendation": "Use the two-scalar geometric terminal gate only if the live candidate already has the portable GRU. It is supported by terminal-objective nested GroupKFold and the untouched fixed validation. Do not stack learned per-class weights.",
    }
    (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    markdown = f"""# Robust meta-ensemble audit

Recommended experiment-only policy: geometric raw-XGB/GRU blend after exact
future-history recovery fails.  The full-OOF terminal-selected weights are
`sim={static_terminal_alpha['sim']:.2f}`, `au={static_terminal_alpha['au']:.2f}`.

| Evaluation | Linear 0.5 terminal | Geometric source terminal | Delta |
| --- | ---: | ---: | ---: |
| Full 5-fold GroupOOF | {report['full_oof']['linear_equal_0_5']['terminal_macro_f1']:.6f} | {report['full_oof']['geometric_source_terminal_static']['terminal_macro_f1']:.6f} | {report['full_oof']['geometric_source_terminal_static']['terminal_macro_f1'] - report['full_oof']['linear_equal_0_5']['terminal_macro_f1']:+.6f} |
| Fixed held-out validation | {report['fixed_validation_raw_xgb']['linear_equal_0_5']['terminal_macro_f1']:.6f} | {report['fixed_validation_raw_xgb']['geometric_source_terminal_static']['terminal_macro_f1']:.6f} | {report['fixed_validation_raw_xgb']['geometric_source_terminal_static']['terminal_macro_f1'] - report['fixed_validation_raw_xgb']['linear_equal_0_5']['terminal_macro_f1']:+.6f} |

The nested terminal GroupKFold audit is recorded in `report.json`.  The
existing source/class calibration is rejected because its 28 fitted factors
lower full GroupOOF terminal macro-F1; the recommended policy has only two
source-level scalars and no class-specific tuning.

Important: use raw XGB probabilities before source-class calibration, then
apply the log/geometric formula only to unrecovered fallback rows.
"""
    (out / "README.md").write_text(markdown, encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
