#!/usr/bin/env python3
"""Modeling D: two-stage intent group model.

D는 14개 action을 바로 맞히는 flat classifier 대신,
1) coarse intent group 예측
2) 해당 group 내부의 fine action 예측
으로 나눕니다.

제출 안정성을 위해 커스텀 estimator 클래스를 pickle하지 않고,
plain dict 안에 scikit-learn Pipeline들만 저장합니다.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from data import load_jsonl, load_labels
from train_modeling_b import ALL_CLASSES, build_modeling_b_features


GROUP_TO_LABELS = {
    "explore": ["read_file", "grep_search", "list_directory", "glob_pattern", "web_search"],
    "modify": ["edit_file", "write_file", "apply_patch"],
    "execute": ["run_bash", "run_tests", "lint_or_typecheck"],
    "coordinate": ["ask_user", "plan_task", "respond_only"],
}
LABEL_TO_GROUP = {label: group for group, labels in GROUP_TO_LABELS.items() for label in labels}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Modeling D two-stage experiments.")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--split-path", default="splits/common_split.csv")
    parser.add_argument("--out-dir", default="experiments/modeling_d")
    parser.add_argument("--model-dir", default="model")
    parser.add_argument("--best-name", default="best_model.pkl")
    parser.add_argument(
        "--experiments",
        default="D0,D1,D2",
        help="Comma-separated experiment ids: D0,D1,D2",
    )
    parser.add_argument("--refit-full", action="store_true")
    return parser.parse_args()


def experiment_config(exp_id: str) -> dict:
    configs = {
        "D0": {"coarse_c": 1.0, "fine_c": 1.0, "coarse_max_features": 100_000, "fine_max_features": 80_000},
        "D1": {"coarse_c": 0.7, "fine_c": 0.7, "coarse_max_features": 100_000, "fine_max_features": 80_000},
        "D2": {"coarse_c": 1.2, "fine_c": 0.8, "coarse_max_features": 100_000, "fine_max_features": 80_000},
    }
    if exp_id not in configs:
        raise ValueError(f"Unknown D experiment: {exp_id}")
    return configs[exp_id]


def make_svc(c_value: float, max_features: int) -> Pipeline:
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    analyzer="word",
                    ngram_range=(1, 2),
                    min_df=2,
                    max_features=max_features,
                    sublinear_tf=True,
                    lowercase=False,
                ),
            ),
            ("clf", LinearSVC(class_weight="balanced", C=c_value)),
        ]
    )


def load_split(path: Path) -> dict[str, str]:
    if not path.exists():
        raise FileNotFoundError(f"Missing common split: {path}")
    with path.open(newline="", encoding="utf-8") as f:
        return {row["id"]: row["split"] for row in csv.DictReader(f)}


def validate_dataset(samples: list[dict], labels: dict[str, str], split_map: dict[str, str]) -> None:
    sample_ids = {str(sample["id"]) for sample in samples}
    missing_labels = sample_ids - set(labels)
    missing_split = sample_ids - set(split_map)
    unknown_labels = sorted(set(labels.values()) - set(ALL_CLASSES))
    if missing_labels:
        raise ValueError(f"{len(missing_labels)} samples have no label. Example: {next(iter(missing_labels))}")
    if missing_split:
        raise ValueError(f"{len(missing_split)} samples are missing from split. Example: {next(iter(missing_split))}")
    if unknown_labels:
        raise ValueError(f"Unknown labels found: {unknown_labels}")


def majority_label(labels: list[str]) -> str:
    return Counter(labels).most_common(1)[0][0]


def fit_two_stage(texts: list[str], labels: list[str], config: dict) -> dict:
    coarse_y = [LABEL_TO_GROUP[label] for label in labels]
    coarse_model = make_svc(config["coarse_c"], config["coarse_max_features"])
    coarse_model.fit(texts, coarse_y)

    fine_models = {}
    default_label_by_group = {}
    for group, group_labels in GROUP_TO_LABELS.items():
        idx = [i for i, label in enumerate(labels) if label in group_labels]
        group_x = [texts[i] for i in idx]
        group_y = [labels[i] for i in idx]
        default_label_by_group[group] = majority_label(group_y)
        fine_model = make_svc(config["fine_c"], config["fine_max_features"])
        fine_model.fit(group_x, group_y)
        fine_models[group] = fine_model

    return {
        "model_type": "two_stage_group_v1",
        "coarse_model": coarse_model,
        "fine_models": fine_models,
        "group_to_labels": GROUP_TO_LABELS,
        "label_to_group": LABEL_TO_GROUP,
        "default_label_by_group": default_label_by_group,
        "config": config,
    }


def predict_two_stage(model_bundle: dict, texts: list[str]) -> list[str]:
    coarse_preds = [str(group) for group in model_bundle["coarse_model"].predict(texts)]
    grouped_indices = defaultdict(list)
    for idx, group in enumerate(coarse_preds):
        grouped_indices[group].append(idx)

    preds = [None] * len(texts)
    fine_models = model_bundle["fine_models"]
    defaults = model_bundle["default_label_by_group"]
    for group, indices in grouped_indices.items():
        if group not in fine_models:
            default = defaults.get(group, "respond_only")
            for idx in indices:
                preds[idx] = default
            continue
        group_texts = [texts[idx] for idx in indices]
        group_preds = [str(pred) for pred in fine_models[group].predict(group_texts)]
        allowed = set(GROUP_TO_LABELS.get(group, ALL_CLASSES))
        default = defaults.get(group, "respond_only")
        for idx, pred in zip(indices, group_preds):
            preds[idx] = pred if pred in allowed else default

    return [str(pred) for pred in preds]


def save_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    model_dir = Path(args.model_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    samples = load_jsonl(str(data_dir / "train.jsonl"))
    labels_map = load_labels(str(data_dir / "train_labels.csv"))
    split_map = load_split(Path(args.split_path))
    validate_dataset(samples, labels_map, split_map)

    ids, texts = build_modeling_b_features(samples, history_n=3, include_meta=True)
    labels = [labels_map[sample_id] for sample_id in ids]
    train_idx = [i for i, sample_id in enumerate(ids) if split_map[sample_id] == "train"]
    val_idx = [i for i, sample_id in enumerate(ids) if split_map[sample_id] == "val"]
    if not train_idx or not val_idx:
        raise ValueError("Common split must contain both train and val samples.")

    x_train = [texts[i] for i in train_idx]
    y_train = [labels[i] for i in train_idx]
    x_val = [texts[i] for i in val_idx]
    y_val = [labels[i] for i in val_idx]

    print(f"samples={len(samples)} train={len(x_train)} val={len(x_val)}")
    print(f"train_distribution={dict(Counter(y_train))}")
    print(f"val_distribution={dict(Counter(y_val))}")

    requested = [item.strip() for item in args.experiments.split(",") if item.strip()]
    rows = []
    best = None

    for exp_id in requested:
        config = experiment_config(exp_id)
        print(f"\n===== {exp_id} start config={config} =====")
        start = time.time()
        model_bundle = fit_two_stage(x_train, y_train, config)
        train_seconds = time.time() - start
        preds = predict_two_stage(model_bundle, x_val)
        invalid = sorted(set(preds) - set(ALL_CLASSES))
        if invalid:
            raise ValueError(f"{exp_id} produced invalid labels: {invalid}")

        macro_f1 = f1_score(y_val, preds, labels=ALL_CLASSES, average="macro", zero_division=0)
        report = classification_report(y_val, preds, labels=ALL_CLASSES, zero_division=0, digits=4)
        cm = confusion_matrix(y_val, preds, labels=ALL_CLASSES)
        artifact = out_dir / f"{exp_id}.pkl"
        joblib.dump(model_bundle, artifact, compress=3)
        save_text(out_dir / f"{exp_id}_classification_report.txt", report)
        save_text(out_dir / f"{exp_id}_confusion_matrix.json", json.dumps({"labels": ALL_CLASSES, "matrix": cm.tolist()}, ensure_ascii=False, indent=2))

        row = {
            "experiment": exp_id,
            "macro_f1": macro_f1,
            "train_seconds": round(train_seconds, 3),
            "n_train": len(x_train),
            "n_val": len(x_val),
            "coarse_c": config["coarse_c"],
            "fine_c": config["fine_c"],
            "artifact": str(artifact),
        }
        rows.append(row)
        print(f"{exp_id} macro_f1={macro_f1:.6f} train_seconds={train_seconds:.1f}")
        if best is None or macro_f1 > best["macro_f1"]:
            best = {**row, "model_bundle": model_bundle, "config": config}

    results_path = out_dir / "modeling_d_results.csv"
    with results_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["experiment", "macro_f1", "train_seconds", "n_train", "n_val", "coarse_c", "fine_c", "artifact"],
        )
        writer.writeheader()
        writer.writerows(rows)

    if best is None:
        raise RuntimeError("No D experiment was trained.")

    best_model = best["model_bundle"]
    if args.refit_full:
        print(f"\nRefitting {best['experiment']} on full training data...")
        best_model = fit_two_stage(texts, labels, best["config"])

    best_path = model_dir / args.best_name
    joblib.dump(best_model, best_path, compress=3)
    metadata = {
        "best_experiment": best["experiment"],
        "best_macro_f1": best["macro_f1"],
        "feature_builder": "modeling_b_v1",
        "feature_config": {"history_n": 3, "include_meta": True},
        "model_type": "two_stage_group_v1",
        "group_to_labels": GROUP_TO_LABELS,
        "split_path": args.split_path,
        "refit_full": bool(args.refit_full),
        "all_classes": ALL_CLASSES,
        "results_path": str(results_path),
    }
    save_text(model_dir / "best_model_metadata.json", json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")
    print(f"\nBest experiment: {best['experiment']} macro_f1={best['macro_f1']:.6f}")
    print(f"Saved best model: {best_path}")
    print(f"Saved metadata: {model_dir / 'best_model_metadata.json'}")
    print(f"Saved results: {results_path}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)
