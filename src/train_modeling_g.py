#!/usr/bin/env python3
"""Modeling G: fine-tune multilingual E5 for 14-way action prediction."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import classification_report, f1_score
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

from train_modeling_b import ALL_CLASSES


MODEL_NAME = "intfloat/multilingual-e5-small"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="data")
    p.add_argument("--split-path", default="splits/common_split.csv")
    p.add_argument("--out-dir", default="experiments/modeling_g")
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--eval-batch-size", type=int, default=128)
    p.add_argument("--max-length", type=int, default=160)
    p.add_argument("--learning-rate", type=float, default=2e-5)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup-ratio", type=float, default=0.06)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-train-samples", type=int, default=0)
    return p.parse_args()


def result_status(value):
    text = str(value or "").lower()
    if any(token in text for token in ("fail", "error", "exception", "traceback", "not found")):
        return "failed"
    if any(token in text for token in ("pass", "ok;", "exit=0", "modified", "patched", "found", "listed")):
        return "success"
    return "other"


def model_text(sample):
    prompt = str(sample.get("current_prompt", "") or "").strip()
    history = sample.get("history", [])
    history = history if isinstance(history, list) else []
    actions = [item for item in history if isinstance(item, dict) and item.get("role") == "assistant_action"]
    recent = list(reversed(actions[-4:]))
    action_context = []
    for position, item in enumerate(recent, start=1):
        action_context.append(
            f"previous_{position}={item.get('name', 'none')}:{result_status(item.get('result_summary'))}"
        )
    meta = sample.get("session_meta", {})
    meta = meta if isinstance(meta, dict) else {}
    workspace = meta.get("workspace", {})
    workspace = workspace if isinstance(workspace, dict) else {}
    language_mix = workspace.get("language_mix", {})
    if isinstance(language_mix, dict) and language_mix:
        main_language = max(language_mix, key=language_mix.get)
    else:
        main_language = "unknown"
    state = (
        f"turn={meta.get('turn_index', 'unknown')} tier={meta.get('user_tier', 'unknown')} "
        f"language={meta.get('language_pref', 'unknown')} workspace={main_language} "
        f"git_dirty={workspace.get('git_dirty', 'unknown')} ci={workspace.get('last_ci_status', 'unknown')}"
    )
    return "query: " + prompt + " [SEP] " + " ".join(action_context) + " [SEP] " + state


class TextDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        return self.texts[index], self.labels[index]


def make_collate(tokenizer, max_length):
    def collate(rows):
        texts, labels = zip(*rows)
        batch = tokenizer(list(texts), padding=True, truncation=True, max_length=max_length, return_tensors="pt")
        batch["labels"] = torch.tensor(labels, dtype=torch.long)
        return batch
    return collate


def evaluate(model, loader, device):
    model.eval()
    predictions, targets = [], []
    with torch.inference_mode():
        for batch in loader:
            labels = batch.pop("labels")
            batch = {key: value.to(device) for key, value in batch.items()}
            logits = model(**batch).logits
            predictions.extend(logits.argmax(dim=1).cpu().tolist())
            targets.extend(labels.tolist())
    pred_labels = [ALL_CLASSES[index] for index in predictions]
    target_labels = [ALL_CLASSES[index] for index in targets]
    score = f1_score(target_labels, pred_labels, labels=ALL_CLASSES, average="macro", zero_division=0)
    return score, target_labels, pred_labels


def main():
    args = parse_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    samples = [json.loads(line) for line in open(Path(args.data_dir) / "train.jsonl", encoding="utf-8") if line.strip()]
    with open(Path(args.data_dir) / "train_labels.csv", newline="", encoding="utf-8") as f:
        label_map = {row["id"]: row["action"] for row in csv.DictReader(f)}
    with open(args.split_path, newline="", encoding="utf-8") as f:
        split = {row["id"]: row["split"] for row in csv.DictReader(f)}
    label_to_id = {label: index for index, label in enumerate(ALL_CLASSES)}
    train_rows, val_rows = [], []
    for sample in samples:
        row = (model_text(sample), label_to_id[label_map[sample["id"]]])
        (train_rows if split[sample["id"]] == "train" else val_rows).append(row)
    if args.max_train_samples:
        random.shuffle(train_rows); train_rows = train_rows[:args.max_train_samples]
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=len(ALL_CLASSES), id2label={i: x for i, x in enumerate(ALL_CLASSES)},
        label2id=label_to_id, ignore_mismatched_sizes=True,
    )
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model.to(device)
    collate = make_collate(tokenizer, args.max_length)
    train_loader = DataLoader(TextDataset(*zip(*train_rows)), batch_size=args.batch_size, shuffle=True,
                              collate_fn=collate, num_workers=0)
    val_loader = DataLoader(TextDataset(*zip(*val_rows)), batch_size=args.eval_batch_size, shuffle=False,
                            collate_fn=collate, num_workers=0)
    counts = Counter(label for _, label in train_rows)
    # Square-root balancing is more stable than full inverse-frequency weighting.
    weights = torch.tensor(
        [math.sqrt(len(train_rows) / (len(ALL_CLASSES) * counts[index])) for index in range(len(ALL_CLASSES))],
        dtype=torch.float32, device=device,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, int(total_steps * args.warmup_ratio), total_steps
    )
    scaler = None
    best_score = -1.0
    log_rows = []
    print(f"device={device} train={len(train_rows)} val={len(val_rows)} steps={total_steps}", flush=True)
    for epoch in range(1, args.epochs + 1):
        model.train(); running_loss = 0.0; start = time.time()
        for step, batch in enumerate(train_loader, start=1):
            labels = batch.pop("labels").to(device)
            batch = {key: value.to(device) for key, value in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            logits = model(**batch).logits
            loss = torch.nn.functional.cross_entropy(logits, labels, weight=weights)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step(); scheduler.step()
            running_loss += loss.item()
            if step == 1 or step % 25 == 0:
                print(f"epoch={epoch} step={step}/{len(train_loader)} loss={running_loss/step:.4f} elapsed={time.time()-start:.1f}s", flush=True)
        score, targets, predictions = evaluate(model, val_loader, device)
        epoch_seconds = time.time() - start
        log_rows.append({"epoch": epoch, "macro_f1": score, "train_loss": running_loss / len(train_loader),
                         "seconds": round(epoch_seconds, 2)})
        print(f"epoch={epoch} macro_f1={score:.6f} seconds={epoch_seconds:.1f}", flush=True)
        (out_dir / f"epoch_{epoch}_classification_report.txt").write_text(
            classification_report(targets, predictions, labels=ALL_CLASSES, digits=4, zero_division=0), encoding="utf-8")
        if score > best_score:
            best_score = score
            save_dir = out_dir / "best_encoder"
            model.save_pretrained(save_dir, safe_serialization=True)
            tokenizer.save_pretrained(save_dir)
    with (out_dir / "modeling_g_results.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "macro_f1", "train_loss", "seconds"])
        writer.writeheader(); writer.writerows(log_rows)
    (out_dir / "metadata.json").write_text(json.dumps({"model_name": MODEL_NAME, "best_macro_f1": best_score,
        "max_length": args.max_length, "all_classes": ALL_CLASSES}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
