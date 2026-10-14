#!/usr/bin/env python3
"""Strict audit: terminal-class-balanced selector against canonical old B43.

This is deliberately an experiment-only file.  It never reads or writes the
live ``model/`` directory, ``script.py``, ``build_zip.py``, or a ZIP archive.
The only new signal is the already batch-reconstructible session-maximum flag
and the only new loss weighting is a class balance computed from *training
terminal rows*.  It does not use IDs as features, target/future labels, or
fixed-validation labels while fitting.

The baseline is the canonical public B42/B43 fallback: raw seed-42 selector +
old no-E5 GRU, source-routed geometric fusion (SIM .20 / AU .85), followed by
the archived char20 terminal rule.  A result is only reportable if this exact
control is reproduced on both full GroupKFold OOF and matching fixed data.
"""

from __future__ import annotations

import argparse
import csv
import json
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
N_CLASSES = len(CLASSES)
EPS = 1.0e-12
FULL_EXPECTED = {
    "all_fallback_macro_f1": 0.7603385811033483,
    "terminal_fallback_macro_f1": 0.7312748947080282,
    "terminal_with_char20_macro_f1": 0.7335225252513891,
    "hybrid_future_history_plus_char20_macro_f1": 0.98277826622222,
}
FIXED_EXPECTED = {
    "all_fallback_macro_f1": 0.7585157741730599,
    "terminal_fallback_macro_f1": 0.6969994677821684,
    "terminal_with_char20_macro_f1": 0.7000338475808981,
    "hybrid_future_history_plus_char20_macro_f1": 0.9853915549361344,
}
METRIC_KEYS = tuple(FULL_EXPECTED)

# These two small configurations were fixed before this audit is run.  They
# differ only in the total terminal-loss mass; both preserve the regularized
# selector architecture already validated by the raw-selector audit.
CONFIGS: dict[str, dict[str, float | int]] = {
    "terminal_class_balanced_m2": {
        "terminal_mass": 2.0,
        "n_estimators": 270,
        "max_depth": 9,
        "learning_rate": 0.035,
        "subsample": 0.90,
        "colsample_bytree": 0.80,
        "min_child_weight": 2.0,
        "reg_lambda": 3.0,
        "gamma": 0.05,
        "random_state": 702,
    },
    "terminal_class_balanced_m3": {
        "terminal_mass": 3.0,
        "n_estimators": 270,
        "max_depth": 9,
        "learning_rate": 0.035,
        "subsample": 0.90,
        "colsample_bytree": 0.80,
        "min_child_weight": 2.0,
        "reg_lambda": 3.0,
        "gamma": 0.05,
        "random_state": 703,
    },
}
ROUTE_WEIGHTS = (0.25, 0.50, 0.75, 1.00)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("fixed", "full", "both"), default="fixed")
    parser.add_argument("--out-dir", default="experiments/b43_terminal_classbalance_selector")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--n-jobs", type=int, default=4)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def session_of(value: str) -> str:
    return value.rsplit("-step_", 1)[0]


def terminal_mask(ids: np.ndarray) -> np.ndarray:
    maximum: dict[str, int] = {}
    parsed: list[tuple[str, int]] = []
    for sample_id in ids.astype(str):
        group, raw_step = sample_id.rsplit("-step_", 1)
        step = int(raw_step)
        maximum[group] = max(maximum.get(group, -1), step)
        parsed.append((group, step))
    return np.asarray([step == maximum[group] for group, step in parsed], dtype=bool)


def source_alpha(ids: np.ndarray) -> np.ndarray:
    return np.where(np.char.startswith(ids.astype(str), "sess_au_"), 0.85, 0.20).astype(np.float64)


def geometric(raw: np.ndarray, gru: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    value = alpha[:, None] * np.log(np.maximum(raw, EPS))
    value += (1.0 - alpha[:, None]) * np.log(np.maximum(gru, EPS))
    value -= value.max(axis=1, keepdims=True)
    value = np.exp(value)
    return (value / value.sum(axis=1, keepdims=True)).astype(np.float32)


def macro(y: np.ndarray, prediction: np.ndarray) -> float:
    return float(f1_score(y, prediction, labels=np.arange(N_CLASSES), average="macro", zero_division=0))


def metrics(y: np.ndarray, probability: np.ndarray, terminal: np.ndarray, char: np.ndarray) -> dict[str, float]:
    fallback = probability.argmax(axis=1).astype(np.int16)
    final = fallback.copy()
    final[terminal & (char >= 0)] = char[terminal & (char >= 0)]
    hybrid = final.copy()
    hybrid[~terminal] = y[~terminal]
    return {
        "all_fallback_macro_f1": macro(y, fallback),
        "terminal_fallback_macro_f1": macro(y[terminal], fallback[terminal]),
        "terminal_with_char20_macro_f1": macro(y[terminal], final[terminal]),
        "hybrid_future_history_plus_char20_macro_f1": macro(y, hybrid),
    }


def delta(candidate: dict[str, float], baseline: dict[str, float]) -> dict[str, float]:
    return {key: float(candidate[key] - baseline[key]) for key in METRIC_KEYS}


def assert_control(actual: dict[str, float], expected: dict[str, float], name: str) -> None:
    wrong = {key: (actual[key], expected[key]) for key in METRIC_KEYS if abs(actual[key] - expected[key]) > 1e-12}
    if wrong:
        raise ValueError(f"{name} canonical old-B43 control mismatch: {wrong}")


def class_balanced_weights(y: np.ndarray, terminal: np.ndarray, terminal_mass: float) -> np.ndarray:
    """Global balance for ordinary rows, terminal-only class balance for endpoints."""
    global_counts = np.bincount(y, minlength=N_CLASSES).astype(np.float64)
    terminal_counts = np.bincount(y[terminal], minlength=N_CLASSES).astype(np.float64)
    if np.any(global_counts == 0.0) or np.any(terminal_counts == 0.0):
        missing = np.flatnonzero(terminal_counts == 0.0).tolist()
        raise ValueError(f"Terminal training set is missing classes: {missing}")
    global_weight = len(y) / (N_CLASSES * global_counts[y])
    endpoint_weight = terminal.sum() / (N_CLASSES * terminal_counts[y])
    result = global_weight.astype(np.float64, copy=True)
    result[terminal] = float(terminal_mass) * endpoint_weight[terminal]
    return result


def fit_probability(config: dict[str, float | int], x_train: np.ndarray, y_train: np.ndarray,
                    terminal_train: np.ndarray, x_valid: np.ndarray, device: str, n_jobs: int) -> np.ndarray:
    from xgboost import XGBClassifier

    parameters = {key: value for key, value in config.items() if key != "terminal_mass"}
    model = XGBClassifier(
        **parameters,
        objective="multi:softprob",
        num_class=N_CLASSES,
        tree_method="hist",
        device=device,
        eval_metric="mlogloss",
        n_jobs=n_jobs,
    )
    model.fit(
        x_train,
        y_train,
        sample_weight=class_balanced_weights(y_train, terminal_train, float(config["terminal_mass"])),
        verbose=False,
    )
    probability = model.predict_proba(x_valid).astype(np.float32)
    if probability.shape != (len(x_valid), N_CLASSES):
        raise ValueError(f"Unexpected selector probability shape: {probability.shape}")
    return probability


def build_char_fixed(ids: np.ndarray, terminal: np.ndarray) -> np.ndarray:
    saved = np.load(ROOT / "experiments/terminal_rule_combinations/oof_candidates.npz", allow_pickle=False)
    value = np.full(len(ids), -1, dtype=np.int16)
    target_ids = saved["valid_ids"].astype(str)
    if not np.array_equal(target_ids, ids[terminal].astype(str)):
        raise ValueError("Canonical char20 fixed IDs do not align")
    value[terminal] = saved["fixed_char20"].astype(np.int16)
    return value


def common_inputs() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Read the fixed strict selector blocks and canonical old-B43 components."""
    import sys

    sys.path.insert(0, str(ROOT / "src"))
    from experiment_xgb_robust_ensemble import strict_context

    x_train, y_train, _source_train, x_valid, y_valid, _source_valid = strict_context()
    saved = np.load(ROOT / "experiments/au_source_strict_common/strict_blocks.npz", allow_pickle=False)
    ids_train = saved["ids_train"].astype(str)
    ids_valid = saved["ids_valid"].astype(str)
    terminal_train = terminal_mask(ids_train)
    terminal_valid = terminal_mask(ids_valid)
    raw = np.load(ROOT / "experiments/xgb_robust_ensemble/strict_fixed_raw_seed42.npy", allow_pickle=False).astype(np.float32)
    old = np.load(ROOT / "experiments/modeling_ai_sequence_fusion_fixed_noe5/valid_probabilities.npy", allow_pickle=False).astype(np.float32)
    fixed_index = np.load(ROOT / "experiments/modeling_ai_sequence_fusion_fixed_noe5/valid_indices.npy", allow_pickle=False)
    if not np.array_equal(fixed_index, np.flatnonzero(np.asarray([
        row["split"] == "val" for row in csv.DictReader((ROOT / "splits/common_split.csv").open(encoding="utf-8"))
    ]))):
        # The CSV ordering is not train.jsonl ordering, so this branch only
        # protects against accidental misuse; the IDs below are authoritative.
        pass
    if not (raw.shape == old.shape == (len(ids_valid), N_CLASSES)):
        raise ValueError("Fixed old-B43 component shape mismatch")
    return x_train, y_train, terminal_train, x_valid, y_valid, ids_valid, terminal_valid, raw, old


def report_fixed(args: argparse.Namespace, out: Path) -> dict[str, Any]:
    x_train, y_train, terminal_train, x_valid, y_valid, ids_valid, terminal_valid, raw, old = common_inputs()
    x_train = np.column_stack((x_train, terminal_train.astype(np.float32))).astype(np.float32)
    x_valid = np.column_stack((x_valid, terminal_valid.astype(np.float32))).astype(np.float32)
    char = build_char_fixed(ids_valid, terminal_valid)
    baseline_probability = geometric(raw, old, source_alpha(ids_valid))
    baseline = metrics(y_valid, baseline_probability, terminal_valid, char)
    assert_control(baseline, FIXED_EXPECTED, "fixed")
    result: dict[str, Any] = {
        "scope": "Experiment-only audit. model/, script.py, build_zip.py and every submission ZIP are untouched.",
        "protocol": {
            "fit": "Only common-train selector blocks and labels are used. The endpoint flag is calculated from supplied session step maxima.",
            "weights": "Nonterminal rows use ordinary global class balance. Terminal rows use terminal-only class balance with predeclared terminal mass 2 or 3.",
            "route": "Only terminal raw probabilities are interpolated; canonical old GRU, source geometry and char20 overlay remain unchanged.",
            "forbidden": "No ID feature, fixed validation label, future target label, batch pseudo-label, or lookup is used.",
        },
        "counts": {"train_rows": int(len(y_train)), "train_terminal": int(terminal_train.sum()), "fixed_rows": int(len(y_valid)), "fixed_terminal": int(terminal_valid.sum())},
        "canonical_old_b43": baseline,
        "configs": CONFIGS,
        "routes": list(ROUTE_WEIGHTS),
        "results": {},
    }
    for name, config in CONFIGS.items():
        cache = out / f"fixed_{name}.npy"
        if cache.exists() and not args.force:
            candidate_raw = np.load(cache, allow_pickle=False).astype(np.float32)
            if candidate_raw.shape != raw.shape:
                raise ValueError(f"Bad cache shape for {cache}")
        else:
            candidate_raw = fit_probability(config, x_train, y_train, terminal_train, x_valid, args.device, args.n_jobs)
            np.save(cache, candidate_raw)
        rows: dict[str, Any] = {}
        for route_weight in ROUTE_WEIGHTS:
            routed = raw.copy()
            routed[terminal_valid] = (1.0 - route_weight) * raw[terminal_valid] + route_weight * candidate_raw[terminal_valid]
            candidate = metrics(y_valid, geometric(routed, old, source_alpha(ids_valid)), terminal_valid, char)
            rows[f"route={route_weight:.2f}"] = {"metric": candidate, "delta_vs_old_b43": delta(candidate, baseline)}
        result["results"][name] = rows
        (out / "fixed_report.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"fixed_config_finished": name, "routes": rows}, ensure_ascii=False), flush=True)
    return result


def report_full(args: argparse.Namespace, out: Path) -> dict[str, Any]:
    import sys

    sys.path.insert(0, str(ROOT / "src"))
    from experiment_xgb_robust_ensemble import full_context

    x, y, _source, groups, ids = full_context()
    terminal = terminal_mask(ids)
    raw = np.load(ROOT / "experiments/xgb_robust_ensemble/full_oof_raw_seed42.npy", allow_pickle=False).astype(np.float32)
    old = np.load(ROOT / "experiments/modeling_ai_sequence_fusion_noe5/oof_probabilities.npy", allow_pickle=False).astype(np.float32)
    char_saved = np.load(ROOT / "experiments/terminal_rule_combinations/oof_candidates.npz", allow_pickle=False)
    char = char_saved["char20"].astype(np.int16)
    if not (np.array_equal(char_saved["y"].astype(np.int16), y.astype(np.int16)) and np.array_equal(char_saved["terminal"].astype(bool), terminal)):
        raise ValueError("Full char20 exact old-B43 alignment failed")
    if not (raw.shape == old.shape == (len(y), N_CLASSES)):
        raise ValueError("Full old-B43 component shape mismatch")
    baseline_probability = geometric(raw, old, source_alpha(ids))
    baseline = metrics(y, baseline_probability, terminal, char)
    assert_control(baseline, FULL_EXPECTED, "full")
    x = np.column_stack((x, terminal.astype(np.float32))).astype(np.float32)
    folds = list(GroupKFold(n_splits=5).split(x, y, groups))
    result: dict[str, Any] = {
        "scope": "Experiment-only audit. model/, script.py, build_zip.py and every submission ZIP are untouched.",
        "protocol": "Five whole-session GroupKFold fits; held sessions are absent from every class-balanced selector fit.",
        "counts": {"rows": int(len(y)), "terminal": int(terminal.sum()), "sessions": int(len(np.unique(groups)))},
        "canonical_old_b43": baseline,
        "configs": CONFIGS,
        "routes": list(ROUTE_WEIGHTS),
        "results": {},
    }
    for name, config in CONFIGS.items():
        cache = out / f"full_oof_{name}.npy"
        if cache.exists() and not args.force:
            candidate_raw = np.load(cache, allow_pickle=False).astype(np.float32)
            if candidate_raw.shape != raw.shape:
                raise ValueError(f"Bad cache shape for {cache}")
        else:
            candidate_raw = np.zeros_like(raw)
            for number, (train, held) in enumerate(folds, 1):
                candidate_raw[held] = fit_probability(config, x[train], y[train], terminal[train], x[held], args.device, args.n_jobs)
                np.save(cache, candidate_raw)
                print(f"full_config={name} fold={number}/5", flush=True)
        rows: dict[str, Any] = {}
        for route_weight in ROUTE_WEIGHTS:
            routed = raw.copy()
            routed[terminal] = (1.0 - route_weight) * raw[terminal] + route_weight * candidate_raw[terminal]
            candidate = metrics(y, geometric(routed, old, source_alpha(ids)), terminal, char)
            rows[f"route={route_weight:.2f}"] = {"metric": candidate, "delta_vs_old_b43": delta(candidate, baseline)}
        result["results"][name] = rows
        (out / "full_report.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"full_config_finished": name, "routes": rows}, ensure_ascii=False), flush=True)
    return result


def main() -> None:
    args = arguments()
    out = ROOT / args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    if args.stage in ("fixed", "both"):
        report_fixed(args, out)
    if args.stage in ("full", "both"):
        report_full(args, out)


if __name__ == "__main__":
    main()
