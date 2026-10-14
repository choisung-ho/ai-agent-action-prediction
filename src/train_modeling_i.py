#!/usr/bin/env python3
"""Modeling I: leakage-safe selector over E, D and B model decisions."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import GroupKFold

from data import load_jsonl, load_labels
from train_modeling_b import ALL_CLASSES, build_modeling_b_features
from train_modeling_d import predict_two_stage
from train_modeling_e import build_modeling_e_features, predict_tree


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="data")
    p.add_argument("--split-path", default="splits/common_split.csv")
    p.add_argument("--e-model", default="experiments/modeling_e/E0.pkl")
    p.add_argument("--d-model", default="experiments/modeling_d/D2.pkl")
    p.add_argument("--b-model", default="experiments/modeling_b/B3.pkl")
    p.add_argument("--out-dir", default="experiments/modeling_i")
    p.add_argument("--model-dir", default="model")
    p.add_argument("--promote", action="store_true")
    return p.parse_args()


def decision_scores(model, texts):
    values = np.asarray(model.decision_function(texts))
    if values.ndim == 1:
        values = np.column_stack((-values, values))
    return values.astype(np.float32, copy=False)


def stack_features(e_model, d_model, b_model, e_texts, b_texts):
    e_predictions = np.asarray(predict_tree(e_model, e_texts))
    d_predictions = np.asarray(predict_two_stage(d_model, b_texts))
    b_predictions = np.asarray(b_model.predict(b_texts)).astype(str)
    blocks = []
    feature_schema = []
    for node in sorted(e_model["models"]):
        component = e_model["models"][node]
        scores = decision_scores(component, e_texts)
        blocks.append(scores)
        feature_schema.extend(f"e:{node}:{label}" for label in component.classes_)
    component = d_model["coarse_model"]
    blocks.append(decision_scores(component, b_texts))
    feature_schema.extend(f"d:coarse:{label}" for label in component.classes_)
    for node in sorted(d_model["fine_models"]):
        component = d_model["fine_models"][node]
        blocks.append(decision_scores(component, b_texts))
        feature_schema.extend(f"d:{node}:{label}" for label in component.classes_)
    blocks.append(decision_scores(b_model, b_texts))
    feature_schema.extend(f"b:flat:{label}" for label in b_model.classes_)
    for model_name, predictions in (("e", e_predictions), ("d", d_predictions), ("b", b_predictions)):
        blocks.append(np.column_stack([predictions == label for label in ALL_CLASSES]).astype(np.float32))
        feature_schema.extend(f"{model_name}:prediction:{label}" for label in ALL_CLASSES)
    return np.column_stack(blocks), feature_schema, (e_predictions, d_predictions, b_predictions)


def candidate_selectors():
    return {
        "I2_rf_leaf5": RandomForestClassifier(
            n_estimators=300, min_samples_leaf=5, max_features=0.7,
            class_weight="balanced_subsample", n_jobs=-1, random_state=42,
        ),
        "I3_rf_leaf10": RandomForestClassifier(
            n_estimators=300, min_samples_leaf=10, max_features=0.7,
            class_weight="balanced_subsample", n_jobs=-1, random_state=42,
        ),
        "I4_extra": ExtraTreesClassifier(
            n_estimators=300, min_samples_leaf=6, max_features=0.7,
            class_weight="balanced", n_jobs=-1, random_state=42,
        ),
    }


def load_split(path):
    with open(path, newline="", encoding="utf-8") as f:
        return {row["id"]: row["split"] for row in csv.DictReader(f)}


def main():
    args = parse_args()
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    samples = load_jsonl(str(Path(args.data_dir) / "train.jsonl"))
    label_map = load_labels(str(Path(args.data_dir) / "train_labels.csv"))
    split = load_split(args.split_path)
    ids, e_texts = build_modeling_e_features(samples)
    b_ids, b_texts = build_modeling_b_features(samples, history_n=3, include_meta=True)
    if ids != b_ids:
        raise ValueError("E/B feature IDs are misaligned")
    val_idx = [i for i, sample_id in enumerate(ids) if split[sample_id] == "val"]
    val_e = [e_texts[i] for i in val_idx]
    val_b = [b_texts[i] for i in val_idx]
    y_val = np.asarray([label_map[ids[i]] for i in val_idx])
    groups = np.asarray([ids[i].split("-step_", 1)[0] for i in val_idx])
    e_model = joblib.load(args.e_model)
    d_model = joblib.load(args.d_model)
    b_model = joblib.load(args.b_model)
    start = time.time()
    features, schema, base_predictions = stack_features(e_model, d_model, b_model, val_e, val_b)
    print(f"selector_features={features.shape} build_seconds={time.time()-start:.1f}")
    for name, predictions in zip(("E0", "D2", "B3"), base_predictions):
        score = f1_score(y_val, predictions, labels=ALL_CLASSES, average="macro", zero_division=0)
        print(f"{name} macro_f1={score:.6f}")

    folds = list(GroupKFold(n_splits=5).split(features, y_val, groups))
    rows = []
    best = None
    for name, prototype in candidate_selectors().items():
        oof = np.empty(len(y_val), dtype=object)
        fold_scores = []
        for fold, (train_index, test_index) in enumerate(folds, start=1):
            model = prototype.__class__(**prototype.get_params())
            model.fit(features[train_index], y_val[train_index])
            oof[test_index] = model.predict(features[test_index])
            fold_score = f1_score(y_val[test_index], oof[test_index], labels=ALL_CLASSES,
                                  average="macro", zero_division=0)
            fold_scores.append(fold_score)
        score = f1_score(y_val, oof, labels=ALL_CLASSES, average="macro", zero_division=0)
        rows.append({"experiment": name, "oof_macro_f1": score,
                     "fold_scores": ";".join(f"{value:.6f}" for value in fold_scores)})
        print(f"{name} oof_macro_f1={score:.6f} folds={fold_scores}")
        if best is None or score > best["score"]:
            best = {"name": name, "score": score, "prototype": prototype, "oof": oof}
    if best is None:
        raise RuntimeError("No selector trained")
    selector = best["prototype"].__class__(**best["prototype"].get_params())
    selector.fit(features, y_val)
    bundle = {
        "model_type": "stacked_selector_v1",
        "e_model": e_model,
        "d_model": d_model,
        "b_model": b_model,
        "selector": selector,
        "feature_schema": schema,
        "all_classes": ALL_CLASSES,
    }
    artifact = out_dir / "I_best.pkl"
    joblib.dump(bundle, artifact, compress=3)
    (out_dir / "I_best_classification_report.txt").write_text(
        classification_report(y_val, best["oof"], labels=ALL_CLASSES, digits=4, zero_division=0), encoding="utf-8")
    with (out_dir / "modeling_i_results.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["experiment", "oof_macro_f1", "fold_scores"])
        writer.writeheader(); writer.writerows(rows)
    metadata = {
        "best_experiment": best["name"], "best_macro_f1": best["score"],
        "validation_method": "5-fold GroupKFold selector OOF on common validation; base models trained on common train",
        "feature_builder": "stacked_multi_v1", "feature_config": {},
        "model_type": "stacked_selector_v1", "split_path": args.split_path,
        "refit_full": False, "all_classes": ALL_CLASSES,
        "results_path": str(out_dir / "modeling_i_results.csv"),
    }
    (out_dir / "I_best_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.promote:
        model_dir = Path(args.model_dir); model_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(bundle, model_dir / "best_model.pkl", compress=3)
        (model_dir / "best_model_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Best={best['name']} oof_macro_f1={best['score']:.6f} artifact={artifact}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
