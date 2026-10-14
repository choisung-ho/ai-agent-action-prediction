#!/usr/bin/env python3
"""Frozen Qwen2.5 semantic embeddings for the action-prediction task.

This is deliberately an experiment-only script.  It never reads or writes the
submission ``model/`` directory.  Embeddings and reports go below the supplied
``--out-dir`` so the candidate can be evaluated independently before any
promotion decision.

The prompt is designed as a classification instruction rather than a generic
sentence-embedding query.  That lets the final causal-LM state attend to the
action taxonomy while a regularized downstream classifier learns the data's
annotation conventions from the frozen training split.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import time
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.preprocessing import StandardScaler
from transformers import AutoModel, AutoTokenizer

from train_modeling_b import ALL_CLASSES


MODEL = "Qwen/Qwen2.5-1.5B-Instruct"

TAXONOMY = """Choose one next action:
read_file = inspect a known file; list_directory = list a folder; glob_pattern = find paths by wildcard;
grep_search = search code/text; edit_file = make a focused edit; write_file = create/replace a file;
apply_patch = make a patch spanning changes; run_bash = run a shell command/build; run_tests = run tests;
lint_or_typecheck = lint/typecheck/format; ask_user = ask a clarifying question; plan_task = outline a plan;
web_search = research online; respond_only = answer, recap, or stop without a tool."""


def compact(value: object, limit: int) -> str:
    if isinstance(value, dict):
        result = " ".join(f"{key}={compact(item, limit)}" for key, item in sorted(value.items()))
    elif isinstance(value, list):
        result = " ".join(compact(item, limit) for item in value)
    else:
        result = str(value or "")
    return " ".join(result.replace("\x00", " ").split())[:limit]


def render(sample: dict) -> str:
    """Put the current request last: it is the most predictive causal context."""
    parts = ["You classify the next coding-assistant action.", TAXONOMY, "Conversation context:"]
    history = sample.get("history") if isinstance(sample.get("history"), list) else []
    # A few recent turns are useful when a pronoun refers to a path/tool, but
    # raw result output is intentionally omitted because it is noisy.
    for item in history[-6:]:
        if not isinstance(item, dict):
            continue
        if item.get("role") == "user":
            parts.append("Earlier user: " + compact(item.get("content"), 180))
        elif item.get("role") == "assistant_action":
            args = item.get("args") if isinstance(item.get("args"), dict) else {}
            useful = {key: args[key] for key in ("path", "scope", "pattern", "target", "cmd", "query") if key in args}
            parts.append("Earlier action: " + compact(item.get("name"), 48) + " " + compact(useful, 120))
    meta = sample.get("session_meta") if isinstance(sample.get("session_meta"), dict) else {}
    workspace = meta.get("workspace") if isinstance(meta.get("workspace"), dict) else {}
    parts.append(
        "State: language=" + compact(meta.get("language_pref"), 16)
        + " turn=" + compact(meta.get("turn_index"), 8)
        + " ci=" + compact(workspace.get("last_ci_status"), 16)
    )
    parts.append("Current user request: " + compact(sample.get("current_prompt"), 700))
    parts.append("Action:")
    return "\n".join(parts)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--split-path", default="splits/common_split.csv")
    parser.add_argument("--out-dir", default="experiments/qwen15_embeddings")
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--max-length", type=int, default=384)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--mode", choices=("embed", "classify"), default="embed")
    parser.add_argument("--c-values", default="0.03,0.1,0.3,1.0,3.0")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_data(data_dir: Path, split_path: Path):
    samples = [json.loads(line) for line in (data_dir / "train.jsonl").open(encoding="utf-8") if line.strip()]
    labels = {row["id"]: row["action"] for row in csv.DictReader((data_dir / "train_labels.csv").open(encoding="utf-8"))}
    split = {row["id"]: row["split"] for row in csv.DictReader(split_path.open(encoding="utf-8"))}
    ids = [str(sample["id"]) for sample in samples]
    if set(ids) != set(labels) or set(ids) != set(split):
        raise ValueError("training data, labels, and frozen split must contain the same IDs")
    return samples, np.asarray([labels[sample_id] for sample_id in ids]), np.asarray([split[sample_id] for sample_id in ids]), ids


@torch.inference_mode()
def make_embeddings(samples: list[dict], tokenizer, model, max_length: int, batch_size: int, destination: Path) -> None:
    device = torch.device("cuda")
    dimension = int(model.config.hidden_size)
    matrix = np.lib.format.open_memmap(destination, mode="w+", dtype=np.float16, shape=(len(samples), dimension))
    started = time.time()
    for start in range(0, len(samples), batch_size):
        rows = samples[start:start + batch_size]
        batch = tokenizer(
            [render(sample) for sample in rows], padding=True, truncation=True,
            max_length=max_length, return_tensors="pt", pad_to_multiple_of=8,
        )
        batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            states = model(**batch, use_cache=False).last_hidden_state
            # The final non-padding state is the causal model's representation
            # after the explicit ``Action:`` cue.  It is generally more useful
            # for this decision task than a generic mean-pooled sentence vector.
            last = batch["attention_mask"].sum(dim=1) - 1
            pooled = states[torch.arange(states.shape[0], device=device), last]
            pooled = torch.nn.functional.normalize(pooled.float(), p=2, dim=1)
        matrix[start:start + len(rows)] = pooled.cpu().numpy().astype(np.float16)
        if start == 0 or (start // batch_size + 1) % 100 == 0 or start + len(rows) == len(samples):
            elapsed = time.time() - started
            print(f"embed {start + len(rows)}/{len(samples)} elapsed={elapsed:.1f}s", flush=True)
    del matrix


def classify(embeddings: np.ndarray, labels: np.ndarray, split: np.ndarray, out_dir: Path, c_values: list[float], seed: int) -> None:
    train = split == "train"; valid = split == "val"
    x_train = np.asarray(embeddings[train], dtype=np.float32)
    x_valid = np.asarray(embeddings[valid], dtype=np.float32)
    # Dense normalized embeddings need only a very light scale operation; it
    # makes the C grid stable while preserving cosine geometry.
    scaler = StandardScaler(copy=False)
    x_train = scaler.fit_transform(x_train)
    x_valid = scaler.transform(x_valid)
    rows = []
    best = None
    for c in c_values:
        model = LogisticRegression(
            C=c, max_iter=400, solver="saga", multi_class="multinomial",
            class_weight="balanced", n_jobs=-1, random_state=seed, verbose=0,
        )
        started = time.time(); model.fit(x_train, labels[train]); predicted = model.predict(x_valid)
        score = f1_score(labels[valid], predicted, labels=ALL_CLASSES, average="macro", zero_division=0)
        row = {"c": c, "macro_f1": score, "seconds": round(time.time() - started, 2)}
        rows.append(row); print(json.dumps(row), flush=True)
        if best is None or score > best["score"]:
            best = {"score": score, "c": c, "prediction": predicted}
    with (out_dir / "classification_results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["c", "macro_f1", "seconds"]); writer.writeheader(); writer.writerows(rows)
    (out_dir / "best_report.txt").write_text(
        classification_report(labels[valid], best["prediction"], labels=ALL_CLASSES, digits=4, zero_division=0), encoding="utf-8"
    )
    np.savez_compressed(out_dir / "best_predictions.npz", prediction=best["prediction"], target=labels[valid])
    (out_dir / "summary.json").write_text(json.dumps({"best_macro_f1": best["score"], "best_c": best["c"], "rows": rows}, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("This experiment requires CUDA")
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    samples, labels, split, ids = load_data(Path(args.data_dir), Path(args.split_path))
    (out_dir / "config.json").write_text(json.dumps({**vars(args), "rows": len(samples), "labels": ALL_CLASSES}, indent=2) + "\n", encoding="utf-8")
    embedding_path = out_dir / "qwen_embeddings.npy"
    if args.mode == "embed":
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        tokenizer.padding_side = "right"
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModel.from_pretrained(args.model, torch_dtype=torch.bfloat16, attn_implementation="sdpa").to("cuda").eval()
        make_embeddings(samples, tokenizer, model, args.max_length, args.batch_size, embedding_path)
        np.save(out_dir / "ids.npy", np.asarray(ids))
        np.save(out_dir / "labels.npy", labels)
        np.save(out_dir / "split.npy", split)
    else:
        if not embedding_path.exists():
            raise FileNotFoundError(f"Missing {embedding_path}; run --mode embed first")
        classify(np.load(embedding_path, mmap_mode="r"), labels, split, out_dir, [float(item) for item in args.c_values.split(",")], args.seed)


if __name__ == "__main__":
    main()
