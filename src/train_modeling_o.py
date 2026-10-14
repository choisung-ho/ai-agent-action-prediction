#!/usr/bin/env python3
"""Modeling O: rich conversation-context linear models.

Unlike the earlier sparse baselines, this preserves the user messages,
assistant-action arguments, and result summaries around the current turn.
It is evaluated exclusively on the frozen common group split.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, f1_score
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC

from train_modeling_b import ALL_CLASSES


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--split-path", default="splits/common_split.csv")
    parser.add_argument("--out-dir", default="experiments/modeling_o")
    parser.add_argument("--experiments", default="O0,O1,O2")
    return parser.parse_args()


def flatten(value):
    if isinstance(value, dict):
        return " ".join(f"{key}={flatten(item)}" for key, item in sorted(value.items()))
    if isinstance(value, list):
        return " ".join(flatten(item) for item in value)
    return str(value or "").strip().lower()


def rich_text(sample, history_n, include_results, include_meta):
    parts = ["[CURRENT] " + flatten(sample.get("current_prompt"))]
    history = sample.get("history") if isinstance(sample.get("history"), list) else []
    for position, item in enumerate(history[-history_n:], start=max(1, len(history) - history_n + 1)):
        if not isinstance(item, dict):
            continue
        role = flatten(item.get("role", "unknown"))
        if role == "user":
            parts.append(f"[USER_{position}] {flatten(item.get('content'))}")
        elif role == "assistant_action":
            action = flatten(item.get("name"))
            args = flatten(item.get("args"))
            result = flatten(item.get("result_summary")) if include_results else ""
            parts.append(f"[ACTION_{position}] name={action} args={args} result={result}")
        else:
            parts.append(f"[{role}_{position}] {flatten(item)}")
    if include_meta:
        meta = sample.get("session_meta") if isinstance(sample.get("session_meta"), dict) else {}
        parts.append("[META] " + flatten(meta))
    return " ".join(parts)


def make_model(exp):
    if exp == "O0":
        return Pipeline([
            ("word", TfidfVectorizer(ngram_range=(1, 3), min_df=2, max_features=300_000,
                                     sublinear_tf=True, strip_accents="unicode")),
            ("clf", LinearSVC(C=1.0, class_weight="balanced")),
        ])
    if exp in {"O1", "O2"}:
        return Pipeline([
            ("features", FeatureUnion([
                ("word", TfidfVectorizer(ngram_range=(1, 3), min_df=2, max_features=260_000,
                                          sublinear_tf=True, strip_accents="unicode")),
                ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2,
                                          max_features=260_000, sublinear_tf=True)),
            ])),
            ("clf", LinearSVC(C=0.8 if exp == "O1" else 1.4, class_weight="balanced")),
        ])
    raise ValueError(f"Unknown experiment: {exp}")


def config(exp):
    return {
        "O0": dict(history_n=6, include_results=True, include_meta=True),
        "O1": dict(history_n=6, include_results=True, include_meta=True),
        "O2": dict(history_n=10, include_results=True, include_meta=True),
    }[exp]


def main():
    args = parse_args(); out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    samples = [json.loads(line) for line in open(Path(args.data_dir) / "train.jsonl", encoding="utf-8") if line.strip()]
    with open(Path(args.data_dir) / "train_labels.csv", newline="", encoding="utf-8") as f:
        labels = {row["id"]: row["action"] for row in csv.DictReader(f)}
    with open(args.split_path, newline="", encoding="utf-8") as f:
        split = {row["id"]: row["split"] for row in csv.DictReader(f)}
    train = [i for i, sample in enumerate(samples) if split[sample["id"]] == "train"]
    val = [i for i, sample in enumerate(samples) if split[sample["id"]] == "val"]
    requested = [name.strip() for name in args.experiments.split(",") if name.strip()]
    rows = []
    for exp in requested:
        cfg = config(exp)
        texts = [rich_text(sample, **cfg) for sample in samples]
        y = [labels[sample["id"]] for sample in samples]
        model = make_model(exp); start = time.time()
        model.fit([texts[i] for i in train], [y[i] for i in train])
        pred = model.predict([texts[i] for i in val])
        score = f1_score([y[i] for i in val], pred, labels=ALL_CLASSES, average="macro", zero_division=0)
        seconds = time.time() - start
        print(f"{exp} macro_f1={score:.6f} seconds={seconds:.1f}", flush=True)
        joblib.dump(model, out / f"{exp}.pkl", compress=3)
        (out / f"{exp}_classification_report.txt").write_text(
            classification_report([y[i] for i in val], pred, labels=ALL_CLASSES, digits=4, zero_division=0), encoding="utf-8")
        rows.append({"experiment": exp, "macro_f1": score, "seconds": round(seconds, 2), **cfg})
    with (out / "results.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


if __name__ == "__main__":
    main()
