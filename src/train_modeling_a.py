#!/usr/bin/env python3
"""Modeling A: baseline and traditional ML experiments.

Experiments:
- A0: current_prompt + word TF-IDF + LogisticRegression
- A1: current_prompt + word TF-IDF + LinearSVC
- A2: current_prompt + word+char TF-IDF + LogisticRegression
- A3: current_prompt + word+char TF-IDF + LinearSVC

The script uses the fixed common split:
    splits/common_split.csv

If the split does not exist, create it first:
    python src/make_common_split.py --data-dir data --out-dir splits --strategy group --val-size 0.2 --seed 42
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC

from data import load_jsonl, load_labels


ALL_CLASSES = [
    "read_file",
    "grep_search",
    "list_directory",
    "glob_pattern",
    "edit_file",
    "write_file",
    "apply_patch",
    "run_bash",
    "run_tests",
    "lint_or_typecheck",
    "ask_user",
    "plan_task",
    "web_search",
    "respond_only",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Modeling A experiments.")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--split-path", default="splits/common_split.csv")
    parser.add_argument("--out-dir", default="experiments/modeling_a")
    parser.add_argument("--model-dir", default="model")
    parser.add_argument("--best-name", default="best_model.pkl")
    parser.add_argument(
        "--experiments",
        default="A0,A1,A2,A3",
        help="Comma-separated experiment ids to run: A0,A1,A2,A3",
    )
    parser.add_argument("--refit-full", action="store_true", help="Refit the best model on all training data before saving best_model.pkl")
    return parser.parse_args()


def clean_text(text) -> str:
    if not isinstance(text, str):
        text = "" if text is None else str(text)
    return text.strip().lower()


def build_modeling_a_features(samples: list[dict]) -> tuple[list[str], list[str]]:
    """Prompt-only feature builder for Modeling A.

    This must match the final inference script while Modeling A's model is used.
    """
    ids: list[str] = []
    texts: list[str] = []
    for sample in samples:
        ids.append(str(sample.get("id", "")))
        texts.append(clean_text(sample.get("current_prompt", "")))
    return ids, texts


def load_split(path: Path) -> dict[str, str]:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing common split: {path}. Run "
            "python src/make_common_split.py --data-dir data --out-dir splits --strategy group --val-size 0.2 --seed 42"
        )
    split_map: dict[str, str] = {}
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or "id" not in reader.fieldnames or "split" not in reader.fieldnames:
            raise ValueError(f"{path} must contain id,split columns")
        for row in reader:
            split = row["split"]
            if split not in {"train", "val"}:
                raise ValueError(f"Invalid split value for id={row['id']}: {split}")
            split_map[row["id"]] = split
    return split_map


def validate_dataset(samples: list[dict], labels: dict[str, str], split_map: dict[str, str]) -> None:
    if not samples:
        raise ValueError("train.jsonl is empty. Put the official data into data/.")
    if not labels:
        raise ValueError("train_labels.csv is empty. Put the official labels into data/.")
    sample_ids = {str(sample["id"]) for sample in samples}
    label_ids = set(labels)
    split_ids = set(split_map)
    missing_labels = sample_ids - label_ids
    missing_split = sample_ids - split_ids
    if missing_labels:
        raise ValueError(f"{len(missing_labels)} samples have no label. Example: {next(iter(missing_labels))}")
    if missing_split:
        raise ValueError(f"{len(missing_split)} samples are missing from common_split.csv. Example: {next(iter(missing_split))}")
    unknown_labels = sorted(set(labels.values()) - set(ALL_CLASSES))
    if unknown_labels:
        raise ValueError(f"Unknown labels found: {unknown_labels}")


def make_experiments() -> dict[str, Pipeline]:
    word_tfidf = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        min_df=2,
        max_features=80_000,
        sublinear_tf=True,
        lowercase=False,
    )
    word_tfidf_svc = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        min_df=2,
        max_features=80_000,
        sublinear_tf=True,
        lowercase=False,
    )
    word_char_features_logreg = FeatureUnion(
        [
            (
                "word",
                TfidfVectorizer(
                    analyzer="word",
                    ngram_range=(1, 2),
                    min_df=2,
                    max_features=80_000,
                    sublinear_tf=True,
                    lowercase=False,
                ),
            ),
            (
                "char",
                TfidfVectorizer(
                    analyzer="char_wb",
                    ngram_range=(3, 5),
                    min_df=2,
                    max_features=120_000,
                    sublinear_tf=True,
                    lowercase=False,
                ),
            ),
        ]
    )
    word_char_features_svc = FeatureUnion(
        [
            (
                "word",
                TfidfVectorizer(
                    analyzer="word",
                    ngram_range=(1, 2),
                    min_df=2,
                    max_features=80_000,
                    sublinear_tf=True,
                    lowercase=False,
                ),
            ),
            (
                "char",
                TfidfVectorizer(
                    analyzer="char_wb",
                    ngram_range=(3, 5),
                    min_df=2,
                    max_features=120_000,
                    sublinear_tf=True,
                    lowercase=False,
                ),
            ),
        ]
    )
    return {
        "A0": Pipeline(
            [
                ("tfidf", word_tfidf),
                ("clf", LogisticRegression(max_iter=700, class_weight="balanced", C=2.0, n_jobs=-1)),
            ]
        ),
        "A1": Pipeline(
            [
                ("tfidf", word_tfidf_svc),
                ("clf", LinearSVC(class_weight="balanced", C=1.0)),
            ]
        ),
        "A2": Pipeline(
            [
                ("features", word_char_features_logreg),
                ("clf", LogisticRegression(max_iter=900, class_weight="balanced", C=2.0, n_jobs=-1)),
            ]
        ),
        "A3": Pipeline(
            [
                ("features", word_char_features_svc),
                ("clf", LinearSVC(class_weight="balanced", C=1.0)),
            ]
        ),
    }


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
    labels = load_labels(str(data_dir / "train_labels.csv"))
    split_map = load_split(Path(args.split_path))
    validate_dataset(samples, labels, split_map)

    ids, texts = build_modeling_a_features(samples)
    y = [labels[sample_id] for sample_id in ids]
    train_idx = [i for i, sample_id in enumerate(ids) if split_map[sample_id] == "train"]
    val_idx = [i for i, sample_id in enumerate(ids) if split_map[sample_id] == "val"]
    if not train_idx or not val_idx:
        raise ValueError("Common split must contain both train and val samples.")

    x_train = [texts[i] for i in train_idx]
    y_train = [y[i] for i in train_idx]
    x_val = [texts[i] for i in val_idx]
    y_val = [y[i] for i in val_idx]

    print(f"samples={len(samples)} train={len(x_train)} val={len(x_val)}")
    print(f"train_distribution={dict(Counter(y_train))}")
    print(f"val_distribution={dict(Counter(y_val))}")

    requested = [item.strip() for item in args.experiments.split(",") if item.strip()]
    experiments = make_experiments()
    unknown = [item for item in requested if item not in experiments]
    if unknown:
        raise ValueError(f"Unknown experiment ids: {unknown}")

    rows: list[dict[str, str | float]] = []
    best_score = -1.0
    best_exp = None
    best_model = None

    for exp_id in requested:
        model = experiments[exp_id]
        print(f"\n===== {exp_id} start =====")
        start = time.time()
        model.fit(x_train, y_train)
        train_seconds = time.time() - start
        preds = [str(p) for p in model.predict(x_val)]
        invalid = sorted(set(preds) - set(ALL_CLASSES))
        if invalid:
            raise ValueError(f"{exp_id} produced invalid labels: {invalid}")
        macro_f1 = f1_score(y_val, preds, labels=ALL_CLASSES, average="macro", zero_division=0)
        report = classification_report(y_val, preds, labels=ALL_CLASSES, zero_division=0)
        cm = confusion_matrix(y_val, preds, labels=ALL_CLASSES)
        print(f"{exp_id} macro_f1={macro_f1:.6f} train_seconds={train_seconds:.1f}")

        joblib.dump(model, out_dir / f"{exp_id}.pkl", compress=3)
        save_text(out_dir / f"{exp_id}_classification_report.txt", report)
        save_text(out_dir / f"{exp_id}_confusion_matrix.json", json.dumps({"labels": ALL_CLASSES, "matrix": cm.tolist()}, ensure_ascii=False, indent=2))

        rows.append(
            {
                "experiment": exp_id,
                "macro_f1": macro_f1,
                "train_seconds": round(train_seconds, 3),
                "n_train": len(x_train),
                "n_val": len(x_val),
                "artifact": str(out_dir / f"{exp_id}.pkl"),
            }
        )
        if macro_f1 > best_score:
            best_score = macro_f1
            best_exp = exp_id
            best_model = model

    results_path = out_dir / "modeling_a_results.csv"
    with results_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["experiment", "macro_f1", "train_seconds", "n_train", "n_val", "artifact"])
        writer.writeheader()
        writer.writerows(rows)

    if best_model is None or best_exp is None:
        raise RuntimeError("No experiment was trained.")

    if args.refit_full:
        print(f"\nRefitting best experiment {best_exp} on full training data...")
        full_model = make_experiments()[best_exp]
        full_model.fit(texts, y)
        best_model = full_model

    best_path = model_dir / args.best_name
    joblib.dump(best_model, best_path, compress=3)
    metadata = {
        "best_experiment": best_exp,
        "best_macro_f1": best_score,
        "feature_builder": "modeling_a_prompt_only",
        "split_path": args.split_path,
        "refit_full": args.refit_full,
        "all_classes": ALL_CLASSES,
        "results_path": str(results_path),
    }
    save_text(model_dir / "best_model_metadata.json", json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")
    print(f"\nBest experiment: {best_exp} macro_f1={best_score:.6f}")
    print(f"Saved best model: {best_path}")
    print(f"Saved results: {results_path}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)
