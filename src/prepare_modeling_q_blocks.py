#!/usr/bin/env python3
"""Materialize one I3 base-margin block at a time for Modeling Q.

The existing ``stack_features`` helper calculates every margin in one call.
On this desktop that can exceed the per-command execution window, so this
utility writes independent, read-only cache blocks.  It does not train or
modify a submission model.
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import joblib
import numpy as np

from data import load_jsonl
from train_modeling_b import build_modeling_b_features
from train_modeling_e import build_modeling_e_features


def args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("component", help="e:<tree-node>, d:coarse, d:<fine-node>, or b")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--split-path", default="splits/common_split.csv")
    parser.add_argument("--e-model", default="experiments/modeling_e/E0.pkl")
    parser.add_argument("--d-model", default="experiments/modeling_d/D2.pkl")
    parser.add_argument("--b-model", default="experiments/modeling_b/B3.pkl")
    parser.add_argument("--out-dir", default="experiments/modeling_q/blocks")
    return parser.parse_args()


def decision_scores(model, texts: list[str]) -> np.ndarray:
    values = np.asarray(model.decision_function(texts))
    if values.ndim == 1:
        values = np.column_stack((-values, values))
    return values.astype(np.float32, copy=False)


def load_val_indices(ids: list[str], split_path: str) -> np.ndarray:
    with open(split_path, newline="", encoding="utf-8") as handle:
        split = {row["id"]: row["split"] for row in csv.DictReader(handle)}
    return np.asarray([index for index, sample_id in enumerate(ids) if split[sample_id] == "val"])


def main() -> None:
    a = args()
    out_dir = Path(a.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    samples = load_jsonl(str(Path(a.data_dir) / "train.jsonl"))
    if a.component.startswith("e:"):
        node = a.component.split(":", 1)[1]
        ids, texts = build_modeling_e_features(samples)
        indices = load_val_indices(ids, a.split_path)
        model = joblib.load(a.e_model)["models"].get(node)
        if model is None:
            raise ValueError(f"Unknown E tree node: {node}")
        scores = decision_scores(model, [texts[index] for index in indices])
        classes = [str(value) for value in model.classes_]
        stem = f"e__{node}"
    elif a.component.startswith("d:"):
        node = a.component.split(":", 1)[1]
        ids, texts = build_modeling_b_features(samples, history_n=3, include_meta=True)
        indices = load_val_indices(ids, a.split_path)
        bundle = joblib.load(a.d_model)
        if node == "coarse":
            model = bundle["coarse_model"]
        else:
            model = bundle["fine_models"].get(node)
        if model is None:
            raise ValueError(f"Unknown D node: {node}")
        scores = decision_scores(model, [texts[index] for index in indices])
        classes = [str(value) for value in model.classes_]
        stem = f"d__{node}"
    elif a.component == "b":
        ids, texts = build_modeling_b_features(samples, history_n=3, include_meta=True)
        indices = load_val_indices(ids, a.split_path)
        model = joblib.load(a.b_model)
        scores = decision_scores(model, [texts[index] for index in indices])
        classes = [str(value) for value in model.classes_]
        stem = "b"
    else:
        raise ValueError("component must be e:<node>, d:coarse, d:<node>, or b")
    np.save(out_dir / f"{stem}.npy", scores)
    metadata_path = out_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {"components": {}, "ids": []}
    metadata["components"][a.component] = {"file": f"{stem}.npy", "classes": classes}
    if not metadata["ids"]:
        metadata["ids"] = [ids[index] for index in indices]
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"component={a.component} shape={scores.shape} seconds={time.time()-started:.1f}", flush=True)


if __name__ == "__main__":
    main()
