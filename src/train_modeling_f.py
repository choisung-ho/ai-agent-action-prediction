#!/usr/bin/env python3
"""Modeling F: multilingual E5 embeddings + linear action classifier.

This is an experiment script.  It caches embeddings so classifier and stacking
experiments do not repeatedly run the transformer encoder.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import joblib
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.svm import LinearSVC
from transformers import AutoModel, AutoTokenizer

from train_modeling_b import ALL_CLASSES


MODEL_NAME = "intfloat/multilingual-e5-small"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="data")
    p.add_argument("--split-path", default="splits/common_split.csv")
    p.add_argument("--out-dir", default="experiments/modeling_f")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--max-length", type=int, default=128)
    p.add_argument("--force-embed", action="store_true")
    return p.parse_args()


def load_data(data_dir):
    samples = [json.loads(line) for line in open(Path(data_dir) / "train.jsonl", encoding="utf-8") if line.strip()]
    with open(Path(data_dir) / "train_labels.csv", newline="", encoding="utf-8") as f:
        labels = {row["id"]: row["action"] for row in csv.DictReader(f)}
    return samples, labels


def semantic_text(sample):
    prompt = str(sample.get("current_prompt", "") or "").strip()
    history = sample.get("history", [])
    if not isinstance(history, list):
        history = []
    actions = [item for item in history if isinstance(item, dict) and item.get("role") == "assistant_action"]
    if not actions:
        return "query: " + prompt
    latest = actions[-1]
    previous_action = str(latest.get("name", "none"))
    result = str(latest.get("result_summary", "") or "")[:160]
    return f"query: {prompt} [previous action: {previous_action}] [previous result: {result}]"


def average_pool(last_hidden_state, attention_mask):
    hidden = last_hidden_state.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]


def encode(texts, batch_size, max_length):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model.to(device).eval()
    rows = []
    start = time.time()
    with torch.inference_mode():
        for offset in range(0, len(texts), batch_size):
            batch = tokenizer(texts[offset:offset + batch_size], max_length=max_length, padding=True,
                              truncation=True, return_tensors="pt")
            batch = {key: value.to(device) for key, value in batch.items()}
            output = model(**batch)
            pooled = average_pool(output.last_hidden_state, batch["attention_mask"])
            pooled = F.normalize(pooled, p=2, dim=1)
            rows.append(pooled.cpu().numpy().astype("float32"))
            if offset == 0 or (offset // batch_size) % 50 == 0:
                elapsed = time.time() - start
                print(f"encoded={min(offset + batch_size, len(texts))}/{len(texts)} elapsed={elapsed:.1f}s", flush=True)
    return np.concatenate(rows)


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    samples, label_map = load_data(args.data_dir)
    ids = [str(sample["id"]) for sample in samples]
    labels = np.array([label_map[sample_id] for sample_id in ids])
    texts = [semantic_text(sample) for sample in samples]
    cache = out_dir / f"e5_embeddings_len{args.max_length}.npy"
    if cache.exists() and not args.force_embed:
        embeddings = np.load(cache)
    else:
        embeddings = encode(texts, args.batch_size, args.max_length)
        np.save(cache, embeddings)
    if len(embeddings) != len(samples):
        raise ValueError("Embedding cache row count mismatch")
    with open(args.split_path, newline="", encoding="utf-8") as f:
        split = {row["id"]: row["split"] for row in csv.DictReader(f)}
    train_idx = np.array([i for i, sample_id in enumerate(ids) if split[sample_id] == "train"])
    val_idx = np.array([i for i, sample_id in enumerate(ids) if split[sample_id] == "val"])
    rows = []
    candidates = [
        ("F0", LinearSVC(class_weight="balanced", C=0.1, dual=True)),
        ("F1", LinearSVC(class_weight="balanced", C=0.3, dual=True)),
        ("F2", LinearSVC(class_weight="balanced", C=1.0, dual=True)),
        ("F3", LogisticRegression(max_iter=1000, class_weight="balanced", C=1.0, n_jobs=-1)),
    ]
    for name, classifier in candidates:
        start = time.time()
        classifier.fit(embeddings[train_idx], labels[train_idx])
        pred = classifier.predict(embeddings[val_idx])
        score = f1_score(labels[val_idx], pred, labels=ALL_CLASSES, average="macro", zero_division=0)
        elapsed = time.time() - start
        joblib.dump(classifier, out_dir / f"{name}_classifier.pkl", compress=3)
        (out_dir / f"{name}_classification_report.txt").write_text(
            classification_report(labels[val_idx], pred, labels=ALL_CLASSES, digits=4, zero_division=0), encoding="utf-8")
        rows.append({"experiment": name, "macro_f1": score, "seconds": round(elapsed, 3)})
        print(f"{name} macro_f1={score:.6f} seconds={elapsed:.1f}", flush=True)
    with (out_dir / "modeling_f_results.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["experiment", "macro_f1", "seconds"])
        writer.writeheader(); writer.writerows(rows)


if __name__ == "__main__":
    main()
