#!/usr/bin/env python3
"""Strict AU-only selector audit against the live raw XGBoost fallback.

This is deliberately an experiment-only program.  It uses the exact global
five-fold session partition of ``modeling_ak_live_xgb_proxy`` so every AU
prediction is made without the held-out session in either the global fallback
or the AU-only candidate.  It never reads/writes ``model/``, ``script.py`` or
any ZIP artifact.

Why not simply split AU rows afresh?  A fresh AU GroupKFold would make a blend
comparison ambiguous: the raw fallback and the AU specialist could have seen
different held-out sessions.  Reusing the global fold IDs makes the direct
comparison and fold-by-fold sign check meaningful.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from script import ALL_CLASSES_LIST, build_state_dict, build_user_sequence_hash
from src.data import load_jsonl, load_labels


N_CLASSES = len(ALL_CLASSES_LIST)
LABEL_TO_ID = {label: position for position, label in enumerate(ALL_CLASSES_LIST)}


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="experiments/au_source_specialist")
    parser.add_argument(
        "--variants",
        default="same_raw",
        help="comma-separated candidate presets: same_raw,regularized,compact",
    )
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--n-jobs", type=int, default=4)
    return parser.parse_args()


def terminal_mask(ids: list[str]) -> np.ndarray:
    maximum: dict[str, int] = {}
    for sample_id in ids:
        session, step = sample_id.rsplit("-step_", 1)
        maximum[session] = max(maximum.get(session, -1), int(step))
    return np.asarray(
        [int(sample_id.rsplit("-step_", 1)[1]) == maximum[sample_id.rsplit("-step_", 1)[0]] for sample_id in ids],
        dtype=bool,
    )


def macro(target: np.ndarray, probability: np.ndarray, mask: np.ndarray | None = None) -> float:
    if mask is None:
        mask = np.ones(len(target), dtype=bool)
    return float(
        f1_score(
            target[mask], probability[mask].argmax(axis=1), labels=np.arange(N_CLASSES),
            average="macro", zero_division=0,
        )
    )


def geometric(left: np.ndarray, right: np.ndarray, alpha: float) -> np.ndarray:
    """Probability product with ``alpha`` weight on the raw global fallback."""
    value = alpha * np.log(np.maximum(left, 1.0e-12)) + (1.0 - alpha) * np.log(np.maximum(right, 1.0e-12))
    value -= value.max(axis=1, keepdims=True)
    value = np.exp(value)
    return value / value.sum(axis=1, keepdims=True)


def candidate_parameters(name: str) -> dict[str, object]:
    # The first preset exactly matches the live raw selector hyperparameters,
    # except that fitting is restricted to AU sessions.  The other two are
    # intentionally lower-capacity controls for this 5k-row source.
    presets: dict[str, dict[str, object]] = {
        "same_raw": dict(
            n_estimators=210, max_depth=10, learning_rate=0.04, subsample=0.85,
            colsample_bytree=0.85, min_child_weight=1.0, reg_lambda=1.0,
            weight_power=1.0,
        ),
        "regularized": dict(
            n_estimators=360, max_depth=7, learning_rate=0.035, subsample=0.90,
            colsample_bytree=0.80, min_child_weight=3.0, reg_lambda=5.0,
            gamma=0.05, weight_power=0.75,
        ),
        "compact": dict(
            n_estimators=520, max_depth=6, learning_rate=0.03, subsample=0.90,
            colsample_bytree=0.70, min_child_weight=4.0, reg_lambda=8.0,
            gamma=0.10, weight_power=0.60,
        ),
    }
    if name not in presets:
        raise ValueError(f"Unknown preset {name!r}; choose from {sorted(presets)}")
    return presets[name]


def build_au_features(samples: list[dict], au_index: np.ndarray) -> np.ndarray:
    """Reconstruct the deployed 2032-vector, retaining only AU rows."""
    base = np.load(ROOT / "experiments/modeling_n_refit/oof_stack_features.npy", mmap_mode="r")
    source = np.load(ROOT / "experiments/modeling_ak_live_xgb_proxy/source_oof.npy", mmap_mode="r")
    bundle = joblib.load(ROOT / "model_candidate_q/best_model.pkl")
    state = bundle["state_vectorizer"].transform([build_state_dict(sample) for sample in samples]).astype(np.float32)
    user = build_user_sequence_hash(samples).astype(np.float32)
    result = np.column_stack((base[au_index], state[au_index], source[au_index], user[au_index])).astype(np.float32)
    if result.shape[1] != 2032:
        raise ValueError(f"Expected deployed feature width 2032, got {result.shape[1]}")
    return result


def build_global_fold(target: np.ndarray, groups: np.ndarray) -> np.ndarray:
    result = np.empty(len(target), dtype=np.int8)
    for fold, (_, valid) in enumerate(GroupKFold(n_splits=5).split(np.zeros(len(target)), target, groups)):
        result[valid] = fold
    return result


def fit_variant(
    name: str,
    features: np.ndarray,
    target: np.ndarray,
    fold: np.ndarray,
    device: str,
    n_jobs: int,
) -> tuple[np.ndarray, list[dict[str, object]]]:
    setting = candidate_parameters(name)
    probability = np.zeros((len(target), N_CLASSES), dtype=np.float32)
    records: list[dict[str, object]] = []
    for held_out in range(5):
        train = fold != held_out
        valid = ~train
        counts = np.bincount(target[train], minlength=N_CLASSES)
        if np.any(counts == 0):
            absent = [ALL_CLASSES_LIST[index] for index in np.flatnonzero(counts == 0)]
            raise ValueError(f"AU fold {held_out + 1} training has absent labels: {absent}")
        weight = (int(train.sum()) / (N_CLASSES * counts[target[train]])) ** float(setting["weight_power"])
        xgb_args = dict(setting)
        xgb_args.pop("weight_power")
        model = XGBClassifier(
            **xgb_args,
            objective="multi:softprob", num_class=N_CLASSES, tree_method="hist", device=device,
            eval_metric="mlogloss", n_jobs=n_jobs, random_state=4242 + held_out,
        )
        model.fit(features[train], target[train], sample_weight=weight, verbose=False)
        probability[valid] = model.predict_proba(features[valid])
        records.append({
            "fold": held_out + 1,
            "n_train": int(train.sum()), "n_valid": int(valid.sum()),
            "macro_f1": macro(target, probability, valid),
        })
        print(f"variant={name} fold={held_out + 1}/5 macro={records[-1]['macro_f1']:.6f}", flush=True)
    return probability, records


def nested_blend(
    target: np.ndarray,
    raw: np.ndarray,
    candidate: np.ndarray,
    fold: np.ndarray,
    terminal: np.ndarray,
) -> tuple[np.ndarray, list[dict[str, object]]]:
    """Second-level OOF scalar selection; the report labels its limitation."""
    grid = np.round(np.arange(0.0, 1.0001, 0.05), 2)
    value = np.zeros_like(raw)
    record: list[dict[str, object]] = []
    for held_out in range(5):
        train = fold != held_out
        valid = ~train
        scores = [macro(target, geometric(raw, candidate, float(alpha)), train & terminal) for alpha in grid]
        best = int(np.argmax(scores))
        alpha = float(grid[best])
        current = geometric(raw, candidate, alpha)
        value[valid] = current[valid]
        record.append({
            "fold": held_out + 1, "alpha_raw_xgb": alpha,
            "selection_terminal_macro_f1": float(scores[best]),
            "heldout_terminal_macro_f1": macro(target, current, valid & terminal),
            "heldout_all_macro_f1": macro(target, current, valid),
        })
    return value, record


def main() -> None:
    args = arguments()
    out = ROOT / args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    samples = load_jsonl(str(ROOT / "data/train.jsonl"))
    labels = load_labels(str(ROOT / "data/train_labels.csv"))
    ids = [str(sample["id"]) for sample in samples]
    target_all = np.asarray([LABEL_TO_ID[labels[sample_id]] for sample_id in ids], dtype=np.int64)
    groups_all = np.asarray([sample_id.rsplit("-step_", 1)[0] for sample_id in ids])
    source = np.asarray([sample_id.startswith("sess_au_") for sample_id in ids])
    global_fold = build_global_fold(target_all, groups_all)
    au_index = np.flatnonzero(source)
    target = target_all[au_index]
    fold = global_fold[au_index]
    terminal = terminal_mask(ids)[au_index]
    raw_all = np.load(ROOT / "experiments/modeling_ak_live_xgb_proxy/oof_probabilities.npy")
    raw = raw_all[au_index].astype(np.float32)
    feature_path = out / "au_features_2032.npy"
    if feature_path.exists():
        features = np.load(feature_path, mmap_mode="r")
        if features.shape != (len(au_index), 2032):
            raise ValueError(f"Cached AU feature matrix has wrong shape: {features.shape}")
        print(f"loaded_cached_features={features.shape}", flush=True)
    else:
        features = build_au_features(samples, au_index)
        np.save(feature_path, features)
        print(f"saved_features={features.shape}", flush=True)
    np.savez_compressed(out / "alignment.npz", ids=np.asarray([ids[index] for index in au_index]), target=target, fold=fold, terminal=terminal)

    base = {
        "au_rows": int(len(target)), "au_sessions": int(len(np.unique(groups_all[au_index]))),
        "au_terminal_rows": int(terminal.sum()),
        "raw_xgb_all_macro_f1": macro(target, raw),
        "raw_xgb_terminal_macro_f1": macro(target, raw, terminal),
        "protocol": "AU-only training and validation use the exact global 5-fold GroupKFold session IDs of the raw XGB OOF.",
    }
    print(json.dumps(base, ensure_ascii=False), flush=True)
    variants = [item.strip() for item in args.variants.split(",") if item.strip()]
    report: dict[str, object] = {"scope": "Experiment only; model/script/ZIP unchanged.", "baseline": base, "variants": {}}
    for name in variants:
        probability, folds = fit_variant(name, features, target, fold, args.device, args.n_jobs)
        np.save(out / f"{name}_oof_probability.npy", probability)
        nested, nested_folds = nested_blend(target, raw, probability, fold, terminal)
        static_scores: dict[str, dict[str, float]] = {}
        for alpha in (0.0, 0.25, 0.50, 0.75, 1.0):
            current = geometric(raw, probability, alpha)
            static_scores[f"raw_alpha_{alpha:.2f}"] = {
                "all_macro_f1": macro(target, current),
                "terminal_macro_f1": macro(target, current, terminal),
            }
        report["variants"][name] = {
            "parameters": candidate_parameters(name), "folds": folds,
            "candidate_all_macro_f1": macro(target, probability),
            "candidate_terminal_macro_f1": macro(target, probability, terminal),
            "raw_vs_candidate_terminal_delta": macro(target, probability, terminal) - macro(target, raw, terminal),
            "static_geometric_blends": static_scores,
            "nested_scalar_blend": {
                "all_macro_f1": macro(target, nested), "terminal_macro_f1": macro(target, nested, terminal),
                "folds": nested_folds,
                "caveat": "Base OOF learners are not retrained inside a second outer loop; use only as a stability audit, not sole promotion evidence.",
            },
        }
        print(json.dumps({name: report["variants"][name]}, ensure_ascii=False), flush=True)
    (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
