#!/usr/bin/env python3
"""Build the full-data Q selector candidate without touching the live model.

The selector is trained on the leakage-safe 70k OOF margin matrix produced by
Modeling N, concatenated with Q's inference-safe state features.  The E/D/B
base models are read from the already full-refit candidate bundle.  Everything
new is written under ``model_candidate_q``.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction import DictVectorizer

from train_modeling_b import ALL_CLASSES
from train_modeling_q import state_dict


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--oof-path", default="experiments/modeling_n_refit/oof_stack_features.npy")
    parser.add_argument("--base-schema", default="experiments/modeling_n_refit/feature_schema.json")
    parser.add_argument("--base-bundle", default="model_candidate_full/best_model.pkl")
    parser.add_argument("--out-dir", default="model_candidate_q")
    parser.add_argument("--trees", type=int, default=300)
    parser.add_argument("--leaf", type=int, default=10)
    parser.add_argument("--jobs", type=int, default=8)
    return parser.parse_args()


def load_rows(data_dir: Path) -> tuple[list[dict], list[str], np.ndarray]:
    samples = [json.loads(line) for line in (data_dir / "train.jsonl").open(encoding="utf-8") if line.strip()]
    labels = {row["id"]: row["action"] for row in csv.DictReader((data_dir / "train_labels.csv").open(encoding="utf-8"))}
    ids = [str(sample["id"]) for sample in samples]
    missing = [sample_id for sample_id in ids if sample_id not in labels]
    if missing:
        raise ValueError(f"Missing labels; first ID: {missing[0]}")
    return samples, ids, np.asarray([labels[sample_id] for sample_id in ids])


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    samples, ids, labels = load_rows(data_dir)
    base = np.load(args.oof_path).astype(np.float32, copy=False)
    if len(base) != len(samples):
        raise ValueError(f"OOF row count {len(base)} != dataset row count {len(samples)}")
    base_schema = json.loads(Path(args.base_schema).read_text(encoding="utf-8"))
    if len(base_schema) != base.shape[1]:
        raise ValueError(f"Base schema width {len(base_schema)} != OOF width {base.shape[1]}")
    base_bundle = joblib.load(args.base_bundle)
    required = {"e_model", "d_model", "b_model"}
    missing = required - set(base_bundle)
    if missing:
        raise ValueError(f"Base bundle missing: {sorted(missing)}")

    print(f"loaded rows={len(samples)} base_oof={base.shape}", flush=True)
    state_vectorizer = DictVectorizer(sparse=False)
    state = state_vectorizer.fit_transform([state_dict(sample) for sample in samples]).astype(np.float32, copy=False)
    x = np.column_stack((base, state)).astype(np.float32, copy=False)
    print(f"state={state.shape} combined={x.shape}; fitting RF trees={args.trees} leaf={args.leaf}", flush=True)
    selector = RandomForestClassifier(
        n_estimators=args.trees,
        min_samples_leaf=args.leaf,
        max_features=0.7,
        class_weight="balanced_subsample",
        n_jobs=args.jobs,
        random_state=42,
    ).fit(x, labels)
    state_schema = state_vectorizer.feature_names_
    bundle = {
        "model_type": "stacked_selector_state_v1",
        "e_model": base_bundle["e_model"],
        "d_model": base_bundle["d_model"],
        "b_model": base_bundle["b_model"],
        "selector": selector,
        "feature_schema": base_schema,
        "state_vectorizer": state_vectorizer,
        "state_feature_schema": state_schema,
        "state_feature_version": "q_state_v1",
        "all_classes": ALL_CLASSES,
    }
    metadata = {
        "best_experiment": f"Q_full_refit_rf{args.trees}_leaf{args.leaf}",
        "best_macro_f1": 0.71441345843586,
        "validation_method": "Q2: fixed common validation, 5-fold GroupKFold selector OOF; full candidate selector fit on 70k Modeling N leakage-safe OOF margins plus all-row state features",
        "feature_builder": "stacked_state_v1",
        "feature_config": {},
        "model_type": "stacked_selector_state_v1",
        "state_feature_version": "q_state_v1",
        "state_feature_count": len(state_schema),
        "base_feature_count": len(base_schema),
        "selector_feature_count": int(x.shape[1]),
        "refit_full": True,
        "all_classes": ALL_CLASSES,
        "base_bundle_source": str(args.base_bundle),
        "oof_source": str(args.oof_path),
        "inference_contract": "Build 97 base stacked-selector margins with feature_schema, append state_vectorizer.transform([q_state_v1(sample)]), then selector.predict.",
    }
    joblib.dump(bundle, out_dir / "best_model.pkl", compress=3)
    (out_dir / "best_model_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (out_dir / "base_feature_schema.json").write_text(json.dumps(base_schema, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (out_dir / "state_feature_schema.json").write_text(json.dumps(state_schema, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (out_dir / "training_summary.json").write_text(json.dumps({
        "rows": len(samples), "base_shape": list(base.shape), "state_shape": list(state.shape),
        "combined_shape": list(x.shape), "selector": {"trees": args.trees, "min_samples_leaf": args.leaf, "n_jobs": args.jobs},
        "id_order_matches_train_jsonl": True,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"candidate written: {out_dir / 'best_model.pkl'}", flush=True)


if __name__ == "__main__":
    main()
