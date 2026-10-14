#!/usr/bin/env python3
"""Modeling J: correctly fine-tune multilingual E5 with mean pooling.

E5 is trained and documented with attention-mask mean pooling.  A standard
BertForSequenceClassification instead consumes the CLS/pooler output and also
gave the random classifier head the encoder's tiny learning rate.  J fixes both.
"""

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
import torch.nn.functional as F
from sklearn.metrics import classification_report, f1_score
from torch import nn
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

from train_modeling_b import ALL_CLASSES
from train_modeling_g import MODEL_NAME, TextDataset, make_collate, model_text


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="data")
    p.add_argument("--split-path", default="splits/common_split.csv")
    p.add_argument("--base-encoder", default=MODEL_NAME)
    p.add_argument("--out-dir", default="experiments/modeling_j")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--eval-batch-size", type=int, default=192)
    p.add_argument("--max-length", type=int, default=96)
    p.add_argument("--encoder-lr", type=float, default=2e-5)
    p.add_argument("--head-lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup-ratio", type=float, default=0.04)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


class MeanPoolClassifier(nn.Module):
    def __init__(self, model_name, num_labels):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = int(self.encoder.config.hidden_size)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        kwargs = {"input_ids": input_ids, "attention_mask": attention_mask}
        if token_type_ids is not None:
            kwargs["token_type_ids"] = token_type_ids
        output = self.encoder(**kwargs).last_hidden_state
        masked = output.masked_fill(~attention_mask[..., None].bool(), 0.0)
        pooled = masked.sum(dim=1) / attention_mask.sum(dim=1, keepdim=True)
        pooled = F.normalize(pooled, p=2, dim=1)
        return self.classifier(self.dropout(pooled))


def evaluate(model, loader, device):
    model.eval(); predictions, targets = [], []
    with torch.inference_mode():
        for batch in loader:
            labels = batch.pop("labels")
            batch = {key: value.to(device) for key, value in batch.items()}
            predictions.extend(model(**batch).argmax(dim=1).cpu().tolist())
            targets.extend(labels.tolist())
    pred_labels = [ALL_CLASSES[index] for index in predictions]
    target_labels = [ALL_CLASSES[index] for index in targets]
    score = f1_score(target_labels, pred_labels, labels=ALL_CLASSES, average="macro", zero_division=0)
    return score, target_labels, pred_labels


def main():
    args = parse_args()
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
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
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    collate = make_collate(tokenizer, args.max_length)
    train_loader = DataLoader(TextDataset(*zip(*train_rows)), batch_size=args.batch_size, shuffle=True,
                              collate_fn=collate, num_workers=0)
    val_loader = DataLoader(TextDataset(*zip(*val_rows)), batch_size=args.eval_batch_size, shuffle=False,
                            collate_fn=collate, num_workers=0)
    model = MeanPoolClassifier(args.base_encoder, len(ALL_CLASSES))
    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    model.to(device)
    counts = Counter(label for _, label in train_rows)
    weights = torch.tensor(
        [math.sqrt(len(train_rows) / (len(ALL_CLASSES) * counts[index])) for index in range(len(ALL_CLASSES))],
        dtype=torch.float32, device=device,
    )
    optimizer = torch.optim.AdamW([
        {"params": model.encoder.parameters(), "lr": args.encoder_lr, "weight_decay": args.weight_decay},
        {"params": model.classifier.parameters(), "lr": args.head_lr, "weight_decay": args.weight_decay},
    ])
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, int(total_steps * args.warmup_ratio), total_steps)
    best_score = -1.0; log_rows = []
    print(f"device={device} train={len(train_rows)} val={len(val_rows)} steps={total_steps}", flush=True)
    for epoch in range(1, args.epochs + 1):
        model.train(); running_loss = 0.0; start = time.time()
        for step, batch in enumerate(train_loader, start=1):
            labels = batch.pop("labels").to(device)
            batch = {key: value.to(device) for key, value in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            logits = model(**batch)
            loss = F.cross_entropy(logits, labels, weight=weights)
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step(); scheduler.step(); running_loss += loss.item()
            if step == 1 or step % 50 == 0:
                print(f"epoch={epoch} step={step}/{len(train_loader)} loss={running_loss/step:.4f} elapsed={time.time()-start:.1f}s", flush=True)
        score, targets, predictions = evaluate(model, val_loader, device)
        elapsed = time.time() - start
        log_rows.append({"epoch": epoch, "macro_f1": score, "loss": running_loss/len(train_loader), "seconds": elapsed})
        print(f"epoch={epoch} macro_f1={score:.6f} seconds={elapsed:.1f}", flush=True)
        (out_dir / f"epoch_{epoch}_classification_report.txt").write_text(
            classification_report(targets, predictions, labels=ALL_CLASSES, digits=4, zero_division=0), encoding="utf-8")
        if score > best_score:
            best_score = score
            save_dir = out_dir / "best_encoder"; save_dir.mkdir(parents=True, exist_ok=True)
            model.encoder.save_pretrained(save_dir, safe_serialization=True)
            tokenizer.save_pretrained(save_dir)
            torch.save(model.classifier.state_dict(), save_dir / "classifier.pt")
    with (out_dir / "modeling_j_results.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "macro_f1", "loss", "seconds"])
        writer.writeheader(); writer.writerows(log_rows)
    (out_dir / "metadata.json").write_text(json.dumps({"model_name": args.base_encoder, "best_macro_f1": best_score,
        "max_length": args.max_length, "all_classes": ALL_CLASSES}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
