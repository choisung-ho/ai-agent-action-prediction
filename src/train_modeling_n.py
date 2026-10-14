#!/usr/bin/env python3
"""Modeling N: full-data out-of-fold stack, then full refit of all base models."""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import GroupKFold

from data import load_jsonl, load_labels
from train_modeling_b import ALL_CLASSES, build_modeling_b_features, make_classifier
from train_modeling_d import fit_two_stage
from train_modeling_e import build_modeling_e_features, fit_tree
from train_modeling_i import stack_features


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="data")
    p.add_argument("--out-dir", default="experiments/modeling_n")
    p.add_argument("--model-dir", default="model")
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--promote", action="store_true")
    return p.parse_args()


def selector_candidates():
    return {
        "N0_rf5": RandomForestClassifier(n_estimators=400, min_samples_leaf=5, max_features=0.7,
            class_weight="balanced_subsample", n_jobs=-1, random_state=42),
        "N1_rf10": RandomForestClassifier(n_estimators=400, min_samples_leaf=10, max_features=0.7,
            class_weight="balanced_subsample", n_jobs=-1, random_state=42),
        "N2_extra": ExtraTreesClassifier(n_estimators=400, min_samples_leaf=6, max_features=0.7,
            class_weight="balanced", n_jobs=-1, random_state=42),
    }


def clone(model):
    return model.__class__(**model.get_params())


def main():
    args = parse_args(); out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    samples = load_jsonl(str(Path(args.data_dir) / "train.jsonl"))
    label_map = load_labels(str(Path(args.data_dir) / "train_labels.csv"))
    ids, e_texts = build_modeling_e_features(samples)
    b_ids, b_texts = build_modeling_b_features(samples, history_n=3, include_meta=True)
    if ids != b_ids: raise ValueError("E/B ids are misaligned")
    labels = np.asarray([label_map[sample_id] for sample_id in ids])
    groups = np.asarray([sample_id.split("-step_", 1)[0] for sample_id in ids])
    folds = list(GroupKFold(n_splits=args.folds).split(np.zeros(len(ids)), labels, groups))
    oof_features = None; feature_schema = None
    for fold, (train_idx, val_idx) in enumerate(folds, start=1):
        start = time.time()
        e_train = [e_texts[i] for i in train_idx]; e_val = [e_texts[i] for i in val_idx]
        b_train = [b_texts[i] for i in train_idx]; b_val = [b_texts[i] for i in val_idx]
        y_train = labels[train_idx].tolist()
        e_model = fit_tree(e_train, y_train, c=0.5, ngram_max=3)
        d_model = fit_two_stage(b_train, y_train, {
            "coarse_c": 1.2, "fine_c": 0.8, "coarse_max_features": 100_000, "fine_max_features": 80_000,
        })
        b_model = make_classifier("B3"); b_model.fit(b_train, y_train)
        fold_features, schema, _ = stack_features(e_model, d_model, b_model, e_val, b_val)
        if oof_features is None:
            oof_features = np.zeros((len(ids), fold_features.shape[1]), dtype=np.float32)
            feature_schema = schema
        if schema != feature_schema: raise ValueError("Fold stack schema mismatch")
        oof_features[val_idx] = fold_features
        print(f"base_fold={fold}/{args.folds} rows={len(val_idx)} seconds={time.time()-start:.1f}", flush=True)
        del e_model, d_model, b_model
    np.save(out_dir / "oof_stack_features.npy", oof_features)
    (out_dir / "feature_schema.json").write_text(json.dumps(feature_schema, ensure_ascii=False, indent=2), encoding="utf-8")

    best = None; rows = []
    for name, prototype in selector_candidates().items():
        oof_predictions = np.empty(len(labels), dtype=object); fold_scores = []
        for train_idx, val_idx in folds:
            model = clone(prototype); model.fit(oof_features[train_idx], labels[train_idx])
            oof_predictions[val_idx] = model.predict(oof_features[val_idx])
            fold_scores.append(f1_score(labels[val_idx], oof_predictions[val_idx], labels=ALL_CLASSES,
                                        average="macro", zero_division=0))
        score = f1_score(labels, oof_predictions, labels=ALL_CLASSES, average="macro", zero_division=0)
        rows.append({"experiment": name, "oof_macro_f1": score,
                     "fold_scores": ";".join(f"{x:.6f}" for x in fold_scores)})
        print(f"{name} selector_oof_macro_f1={score:.6f} folds={fold_scores}", flush=True)
        if best is None or score > best["score"]:
            best = {"name": name, "score": score, "prototype": prototype, "predictions": oof_predictions}
    selector = clone(best["prototype"]); selector.fit(oof_features, labels)
    (out_dir / "N_best_classification_report.txt").write_text(
        classification_report(labels, best["predictions"], labels=ALL_CLASSES, digits=4, zero_division=0), encoding="utf-8")
    with (out_dir / "modeling_n_results.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["experiment", "oof_macro_f1", "fold_scores"])
        writer.writeheader(); writer.writerows(rows)

    print("Refitting E/D/B base models on all 70,000 rows...", flush=True)
    e_model = fit_tree(e_texts, labels.tolist(), c=0.5, ngram_max=3)
    d_model = fit_two_stage(b_texts, labels.tolist(), {
        "coarse_c": 1.2, "fine_c": 0.8, "coarse_max_features": 100_000, "fine_max_features": 80_000,
    })
    b_model = make_classifier("B3"); b_model.fit(b_texts, labels.tolist())
    bundle = {"model_type": "stacked_selector_v1", "e_model": e_model, "d_model": d_model,
              "b_model": b_model, "selector": selector, "feature_schema": feature_schema,
              "all_classes": ALL_CLASSES}
    artifact = out_dir / "N_best.pkl"; joblib.dump(bundle, artifact, compress=3)
    metadata = {"best_experiment": best["name"], "best_macro_f1": best["score"],
        "validation_method": "5-fold GroupKFold full-data OOF base features + 5-fold selector evaluation",
        "feature_builder": "stacked_multi_v1", "feature_config": {}, "model_type": "stacked_selector_v1",
        "refit_full": True, "all_classes": ALL_CLASSES, "results_path": str(out_dir / "modeling_n_results.csv")}
    (out_dir / "N_best_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.promote:
        model_dir = Path(args.model_dir); model_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(bundle, model_dir / "best_model.pkl", compress=3)
        (model_dir / "best_model_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Best={best['name']} score={best['score']:.6f} artifact={artifact}", flush=True)


if __name__ == "__main__": main()
