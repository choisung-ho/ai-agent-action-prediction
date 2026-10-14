#!/usr/bin/env python3
"""Recreate a full group-OOF proxy for the live user/source XGBoost fallback.

The live selector is trained on 70k OOF E/D/B and source-E margins, plus
inference-safe state/user features.  The original build retained the full
selector but not its 5-fold selector probabilities.  This experiment restores
those probabilities so a GRU blend can be selected without relying solely on
the fixed validation split.  It never reads or writes ``model/``.
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

from data import load_jsonl, load_labels
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from script import ALL_CLASSES_LIST, build_state_dict, build_user_sequence_hash
from train_modeling_e import build_modeling_e_features
from train_modeling_r_source_stack import fit_source_models, source_e_features, source_of


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--base-oof", default="experiments/modeling_n_refit/oof_stack_features.npy")
    parser.add_argument("--q-bundle", default="model_candidate_q/best_model.pkl")
    parser.add_argument("--out-dir", default="experiments/modeling_ak_live_xgb_proxy")
    parser.add_argument("--folds", type=int, default=5)
    return parser.parse_args()


def terminal_mask(ids: list[str]) -> np.ndarray:
    maximum: dict[str, int] = {}
    for sample_id in ids:
        session, step = sample_id.rsplit("-step_", 1)
        maximum[session] = max(maximum.get(session, -1), int(step))
    return np.asarray([int(sample_id.rsplit("-step_", 1)[1]) == maximum[sample_id.rsplit("-step_", 1)[0]] for sample_id in ids])


def macro(y: np.ndarray, probability: np.ndarray, mask: np.ndarray | None = None) -> float:
    if mask is None:
        mask = np.ones(len(y), dtype=bool)
    return float(f1_score(y[mask], probability[mask].argmax(axis=1), labels=np.arange(len(ALL_CLASSES_LIST)), average="macro", zero_division=0))


def main() -> None:
    args = parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    samples = load_jsonl(str(Path(args.data_dir) / "train.jsonl"))
    labels = load_labels(str(Path(args.data_dir) / "train_labels.csv"))
    ids, e_texts = build_modeling_e_features(samples)
    to_id = {name: index for index, name in enumerate(ALL_CLASSES_LIST)}
    y = np.asarray([to_id[labels[sample_id]] for sample_id in ids], dtype=np.int64)
    groups = np.asarray([sample_id.split("-step_", 1)[0] for sample_id in ids])
    sources = np.asarray([source_of(sample_id) for sample_id in ids])
    folds = list(GroupKFold(n_splits=args.folds).split(np.zeros(len(ids)), y, groups))

    source_path = out / "source_oof.npy"
    if source_path.exists():
        source_oof = np.load(source_path).astype(np.float32)
        print(f"loaded source_oof={source_oof.shape}", flush=True)
    else:
        source_oof = None
        schema: list[str] | None = None
        for fold, (train_index, valid_index) in enumerate(folds, start=1):
            models = fit_source_models([e_texts[i] for i in train_index], np.asarray([labels[ids[i]] for i in train_index]), sources[train_index])
            values, current_schema, _ = source_e_features(models, [e_texts[i] for i in valid_index], sources[valid_index])
            if source_oof is None:
                source_oof = np.zeros((len(ids), values.shape[1]), dtype=np.float32)
                schema = current_schema
            if current_schema != schema:
                raise ValueError("Source feature schema changed across folds")
            source_oof[valid_index] = values
            print(f"source_fold={fold}/{args.folds} rows={len(valid_index)}", flush=True)
        assert source_oof is not None and schema is not None
        np.save(source_path, source_oof)
        (out / "source_schema.json").write_text(json.dumps(schema, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    base = np.load(args.base_oof).astype(np.float32)
    q_bundle = joblib.load(args.q_bundle)
    state = q_bundle["state_vectorizer"].transform([build_state_dict(sample) for sample in samples]).astype(np.float32)
    user = build_user_sequence_hash(samples)
    x = np.column_stack((base, state, source_oof, user)).astype(np.float32)
    if x.shape[1] != 2032:
        raise ValueError(f"Expected live selector width 2032, got {x.shape[1]}")
    np.savez_compressed(out / "input_shapes.npz", y=y, source=sources, terminal=terminal_mask(ids))
    probability = np.zeros((len(y), len(ALL_CLASSES_LIST)), dtype=np.float32)
    for fold, (train_index, valid_index) in enumerate(folds, start=1):
        counts = np.bincount(y[train_index], minlength=len(ALL_CLASSES_LIST))
        sample_weight = len(train_index) / (len(ALL_CLASSES_LIST) * counts[y[train_index]])
        model = XGBClassifier(
            n_estimators=210, max_depth=10, learning_rate=0.04, subsample=0.85, colsample_bytree=0.85,
            objective="multi:softprob", num_class=len(ALL_CLASSES_LIST), tree_method="hist", device="cuda",
            eval_metric="mlogloss", n_jobs=8, random_state=42,
        )
        model.fit(x[train_index], y[train_index], sample_weight=sample_weight, verbose=False)
        probability[valid_index] = model.predict_proba(x[valid_index])
        print(f"selector_fold={fold}/{args.folds} macro={macro(y[valid_index], probability[valid_index]):.6f}", flush=True)
    terminal = terminal_mask(ids)
    summary = {
        "raw_macro_f1": macro(y, probability),
        "raw_terminal_macro_f1": macro(y, probability, terminal),
        "n_rows": int(len(y)), "n_terminal": int(terminal.sum()),
        "shape": list(x.shape),
        "validation_method": "5-fold GroupKFold; source-E and E/D/B input blocks are out-of-fold; state/user are inference-safe",
        "note": "This is a reproducibility proxy for live XGB. Its source calibration is intentionally not fitted here.",
    }
    np.save(out / "oof_probabilities.npy", probability)
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
