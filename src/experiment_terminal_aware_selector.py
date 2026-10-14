#!/usr/bin/env python3
"""Audit a terminal-aware raw-selector route without mutating submission assets.

The live B43 fallback uses one XGBoost selector for every supplied row, then
blends its raw probabilities with the portable old no-E5 GRU.  In a complete
session batch, however, only the maximum supplied step needs a statistical
prediction: earlier actions are recovered exactly from a later raw history.

This isolated audit tests whether a *single additional boolean feature*
(``is_batch_terminal``) plus modest terminal-example weighting improves that
remaining decision.  The feature is reconstructed from the supplied batch at
inference; it contains no session identifier, label, or future action.

Nothing under ``model/``, ``script.py``, or any submission archive is opened
for writing.  A candidate is considered only after both full GroupKFold OOF
and the independent common fixed split are available.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold

from experiment_xgb_robust_ensemble import (
    N_CLASSES,
    ROOT,
    fixed_char20,
    full_context,
    strict_context,
)


EPS = 1.0e-12
CONFIGS = {
    "terminal_weight_2": {
        "terminal_weight": 2.0,
        "n_estimators": 210,
        "max_depth": 10,
        "learning_rate": 0.04,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "min_child_weight": 1.0,
        "reg_lambda": 1.0,
        "gamma": 0.0,
        "random_state": 4202,
    },
    "terminal_weight_4_regularized": {
        "terminal_weight": 4.0,
        "n_estimators": 270,
        "max_depth": 9,
        "learning_rate": 0.035,
        "subsample": 0.90,
        "colsample_bytree": 0.80,
        "min_child_weight": 2.0,
        "reg_lambda": 3.0,
        "gamma": 0.05,
        "random_state": 4204,
    },
}
TERM_MODEL_WEIGHTS = (0.25, 0.50, 0.75, 1.00)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("fixed", "full", "both"), default="fixed")
    parser.add_argument("--out-dir", default="experiments/terminal_aware_selector")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--n-jobs", type=int, default=4)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def terminal_mask(ids: np.ndarray) -> np.ndarray:
    maximum: dict[str, int] = {}
    parsed: list[tuple[str, int]] = []
    for sample_id in ids.astype(str):
        session, raw_step = sample_id.rsplit("-step_", 1)
        step = int(raw_step)
        maximum[session] = max(maximum.get(session, -1), step)
        parsed.append((session, step))
    return np.asarray([step == maximum[session] for session, step in parsed], dtype=bool)


def source_geometry(raw: np.ndarray, old_gru: np.ndarray, source: np.ndarray) -> np.ndarray:
    alpha = np.where(source.astype(str) == "au", 0.85, 0.20).astype(np.float64)[:, None]
    logits = alpha * np.log(np.maximum(raw, EPS)) + (1.0 - alpha) * np.log(np.maximum(old_gru, EPS))
    logits -= logits.max(axis=1, keepdims=True)
    probability = np.exp(logits)
    return (probability / probability.sum(axis=1, keepdims=True)).astype(np.float32)


def macro(y: np.ndarray, prediction: np.ndarray, mask: np.ndarray | None = None) -> float:
    if mask is None:
        mask = np.ones(len(y), dtype=bool)
    return float(f1_score(y[mask], prediction[mask], labels=np.arange(N_CLASSES), average="macro", zero_division=0))


def metrics(y: np.ndarray, probability: np.ndarray, terminal: np.ndarray, char: np.ndarray) -> dict[str, float]:
    fallback = probability.argmax(axis=1).astype(np.int16)
    final = fallback.copy()
    final[terminal & (char >= 0)] = char[terminal & (char >= 0)]
    hybrid = final.copy()
    hybrid[~terminal] = y[~terminal]
    return {
        "all_fallback_macro_f1": macro(y, fallback),
        "terminal_fallback_macro_f1": macro(y, fallback, terminal),
        "terminal_with_char20_macro_f1": macro(y, final, terminal),
        "hybrid_future_history_plus_char20_macro_f1": macro(y, hybrid),
    }


def augmented_features(x: np.ndarray, terminal: np.ndarray) -> np.ndarray:
    """Append only the batch-reconstructible terminal indicator."""
    return np.column_stack((x, terminal.astype(np.float32))).astype(np.float32, copy=False)


def fit_probability(
    config: dict[str, float | int], x_train: np.ndarray, y_train: np.ndarray,
    terminal_train: np.ndarray, x_predict: np.ndarray, device: str, n_jobs: int,
) -> np.ndarray:
    from xgboost import XGBClassifier

    counts = np.bincount(y_train, minlength=N_CLASSES)
    if np.any(counts == 0):
        raise ValueError(f"Missing labels in fit: {np.flatnonzero(counts == 0).tolist()}")
    weights = len(y_train) / (N_CLASSES * counts[y_train])
    weights = weights.astype(np.float64, copy=False)
    weights[terminal_train] *= float(config["terminal_weight"])
    params = {key: value for key, value in config.items() if key != "terminal_weight"}
    model = XGBClassifier(
        **params,
        objective="multi:softprob",
        num_class=N_CLASSES,
        tree_method="hist",
        device=device,
        eval_metric="mlogloss",
        n_jobs=n_jobs,
    )
    model.fit(x_train, y_train, sample_weight=weights, verbose=False)
    probability = model.predict_proba(x_predict).astype(np.float32)
    if probability.shape[1] != N_CLASSES:
        raise ValueError(f"Unexpected probability width {probability.shape}")
    return probability


def route_probability(global_raw: np.ndarray, terminal_raw: np.ndarray, terminal: np.ndarray, weight: float) -> np.ndarray:
    result = global_raw.copy()
    result[terminal] = (
        (1.0 - weight) * global_raw[terminal] + weight * terminal_raw[terminal]
    )
    return result


def candidate_rows(
    y: np.ndarray, source: np.ndarray, terminal: np.ndarray, char: np.ndarray,
    global_raw: np.ndarray, old_gru: np.ndarray, terminal_raws: dict[str, np.ndarray],
) -> dict[str, dict[str, object]]:
    baseline_probability = source_geometry(global_raw, old_gru, source)
    baseline = metrics(y, baseline_probability, terminal, char)
    rows: dict[str, dict[str, object]] = {"b43_control": baseline}
    for name, terminal_raw in terminal_raws.items():
        for weight in TERM_MODEL_WEIGHTS:
            route = route_probability(global_raw, terminal_raw, terminal, weight)
            value = metrics(y, source_geometry(route, old_gru, source), terminal, char)
            rows[f"{name}|terminal_raw_weight={weight:.2f}"] = {
                **value,
                "delta_vs_b43": {key: float(value[key] - baseline[key]) for key in baseline},
            }
    return rows


def full_char() -> np.ndarray:
    return np.load(ROOT / "experiments/terminal_rule_combinations/oof_candidates.npz", allow_pickle=False)["char20"].astype(np.int16)


def fixed_old_gru_and_ids() -> tuple[np.ndarray, np.ndarray]:
    path = ROOT / "experiments/b42_9_common_old_noe5_seed42"
    probability = np.load(path / "fixed_valid_probabilities.npy", allow_pickle=False).astype(np.float32)
    ids = np.load(path / "fixed_valid_ids.npy", allow_pickle=False).astype(str)
    return probability, ids


def run_fixed(args: argparse.Namespace, out: Path) -> None:
    x_train, y_train, _source_train, x_valid, y_valid, source_valid = strict_context()
    saved = np.load(ROOT / "experiments/au_source_strict_common/strict_blocks.npz", allow_pickle=False)
    ids_train = saved["ids_train"].astype(str)
    ids_valid = saved["ids_valid"].astype(str)
    terminal_train = terminal_mask(ids_train)
    terminal_valid = terminal_mask(ids_valid)
    old_gru, old_ids = fixed_old_gru_and_ids()
    if not np.array_equal(old_ids, ids_valid):
        raise ValueError("Fixed old-GRU IDs do not align with strict selector IDs")
    global_raw = np.load(ROOT / "experiments/xgb_robust_ensemble/strict_fixed_raw_seed42.npy", allow_pickle=False).astype(np.float32)
    if global_raw.shape != old_gru.shape or len(global_raw) != len(ids_valid):
        raise ValueError("Fixed B43 component shapes do not align")
    char = fixed_char20(ids_valid, terminal_valid)
    x_aug_train = augmented_features(x_train, terminal_train)
    x_aug_valid = augmented_features(x_valid, terminal_valid)
    terminal_raws: dict[str, np.ndarray] = {}
    for name, config in CONFIGS.items():
        cache = out / f"fixed_{name}.npy"
        if cache.exists() and not args.force:
            value = np.load(cache, allow_pickle=False).astype(np.float32)
            if value.shape != global_raw.shape:
                raise ValueError(f"Bad cached fixed shape for {cache}")
            terminal_raws[name] = value
            continue
        value = fit_probability(config, x_aug_train, y_train, terminal_train, x_aug_valid, args.device, args.n_jobs)
        np.save(cache, value)
        terminal_raws[name] = value
        print(f"fixed fitted {name}", flush=True)
    report = {
        "scope": "Experiment only; model/, script.py, B42.zip and baseline_submit.zip are not modified.",
        "protocol": {
            "train": "Strict common-train rows only; class-balanced selector loss with a fixed terminal-row multiplier.",
            "feature": "Established 2032 label-safe selector columns plus one is_batch_terminal feature derived from supplied session step maxima.",
            "route": "Only terminal rows interpolate the extra raw selector with B43 raw probability; old no-E5 GRU, geometric source weights and char20 final rule remain unchanged.",
            "fixed": "No common-validation label is used for fitting or route selection.",
        },
        "counts": {"train_rows": int(len(y_train)), "train_terminal": int(terminal_train.sum()), "valid_rows": int(len(y_valid)), "valid_terminal": int(terminal_valid.sum())},
        "configs": CONFIGS,
        "results": candidate_rows(y_valid, source_valid, terminal_valid, char, global_raw, old_gru, terminal_raws),
    }
    (out / "fixed_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["results"], ensure_ascii=False, indent=2), flush=True)


def run_full(args: argparse.Namespace, out: Path) -> None:
    x, y, source, groups, ids = full_context()
    terminal = terminal_mask(ids)
    global_raw = np.load(ROOT / "experiments/xgb_robust_ensemble/full_oof_raw_seed42.npy", allow_pickle=False).astype(np.float32)
    old_gru = np.load(ROOT / "experiments/modeling_ai_sequence_fusion_noe5/oof_probabilities.npy", allow_pickle=False).astype(np.float32)
    char = full_char()
    if not (global_raw.shape == old_gru.shape == (len(y), N_CLASSES)):
        raise ValueError("Full B43 component shapes do not align")
    x_aug = augmented_features(x, terminal)
    folds = list(GroupKFold(n_splits=5).split(x_aug, y, groups))
    terminal_raws: dict[str, np.ndarray] = {}
    fold_scores: dict[str, list[dict[str, float | int]]] = {}
    for name, config in CONFIGS.items():
        cache = out / f"full_oof_{name}.npy"
        if cache.exists() and not args.force:
            value = np.load(cache, allow_pickle=False).astype(np.float32)
            if value.shape != global_raw.shape:
                raise ValueError(f"Bad cached full shape for {cache}")
            terminal_raws[name] = value
            continue
        value = np.zeros_like(global_raw)
        records: list[dict[str, float | int]] = []
        for fold, (train, held) in enumerate(folds, 1):
            value[held] = fit_probability(config, x_aug[train], y[train], terminal[train], x_aug[held], args.device, args.n_jobs)
            records.append({"fold": fold, "train_rows": int(len(train)), "held_rows": int(len(held)), "held_terminal": int(terminal[held].sum())})
            print(f"full {name} fold={fold}/5", flush=True)
        np.save(cache, value)
        terminal_raws[name] = value
        fold_scores[name] = records
    report = {
        "scope": "Experiment only; model/, script.py, B42.zip and baseline_submit.zip are not modified.",
        "protocol": {
            "oof": "Five whole-session GroupKFold fits. The held session is excluded from every terminal-aware selector fit.",
            "feature": "Established 2032 label-safe selector columns plus one is_batch_terminal feature.",
            "route": "Only terminal rows interpolate an additional selector; B43 old-GRU geometry and char20 final rule remain unchanged.",
        },
        "counts": {"rows": int(len(y)), "terminal": int(terminal.sum()), "sessions": int(len(np.unique(groups)))},
        "configs": CONFIGS,
        "folds": fold_scores,
        "results": candidate_rows(y, source, terminal, char, global_raw, old_gru, terminal_raws),
    }
    (out / "full_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["results"], ensure_ascii=False, indent=2), flush=True)


def main() -> None:
    args = parse_args()
    out = ROOT / args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    if args.stage in {"fixed", "both"}:
        run_fixed(args, out)
    if args.stage in {"full", "both"}:
        run_full(args, out)


if __name__ == "__main__":
    main()
