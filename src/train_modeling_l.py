#!/usr/bin/env python3
"""Modeling L: shared multilingual encoder with jointly trained hierarchy heads."""

from __future__ import annotations

import argparse
import csv
import json
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
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

from train_modeling_b import ALL_CLASSES
from train_modeling_e import TREE
from train_modeling_g import make_collate, model_text


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="data")
    p.add_argument("--split-path", default="splits/common_split.csv")
    p.add_argument("--base-encoder", default="experiments/modeling_j/best_encoder")
    p.add_argument("--out-dir", default="experiments/modeling_l")
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--eval-batch-size", type=int, default=192)
    p.add_argument("--max-length", type=int, default=96)
    p.add_argument("--encoder-lr", type=float, default=1e-5)
    p.add_argument("--head-lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def child_for_label(node, label):
    for child, labels in TREE[node].items():
        if label in labels:
            return child
    return None


NODE_CLASSES = {node: list(children) for node, children in TREE.items()}
NODE_TARGETS = {
    node: {label: (NODE_CLASSES[node].index(child_for_label(node, label)) if child_for_label(node, label) else -100)
           for label in ALL_CLASSES}
    for node in TREE
}


class HierarchyDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts; self.labels = labels
    def __len__(self): return len(self.texts)
    def __getitem__(self, index): return self.texts[index], self.labels[index]


def make_hierarchy_collate(tokenizer, max_length):
    def collate(rows):
        texts, labels = zip(*rows)
        batch = tokenizer(list(texts), padding=True, truncation=True, max_length=max_length, return_tensors="pt")
        batch["labels"] = list(labels)
        return batch
    return collate


class HierarchyModel(nn.Module):
    def __init__(self, encoder_path):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(encoder_path)
        hidden = int(self.encoder.config.hidden_size)
        self.dropout = nn.Dropout(0.1)
        self.heads = nn.ModuleDict({node: nn.Linear(hidden, len(classes)) for node, classes in NODE_CLASSES.items()})

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        kwargs = {"input_ids": input_ids, "attention_mask": attention_mask}
        if token_type_ids is not None: kwargs["token_type_ids"] = token_type_ids
        hidden = self.encoder(**kwargs).last_hidden_state
        hidden = hidden.masked_fill(~attention_mask[..., None].bool(), 0.0)
        pooled = hidden.sum(dim=1) / attention_mask.sum(dim=1, keepdim=True)
        pooled = F.normalize(pooled, p=2, dim=1)
        pooled = self.dropout(pooled)
        return {node: head(pooled) for node, head in self.heads.items()}


def predict_labels(logit_map):
    n = next(iter(logit_map.values())).shape[0]
    child_predictions = {node: values.argmax(dim=1).cpu().tolist() for node, values in logit_map.items()}
    output = []
    for index in range(n):
        node = "root"
        while True:
            child = NODE_CLASSES[node][child_predictions[node][index]]
            labels = TREE[node][child]
            if len(labels) == 1:
                output.append(labels[0]); break
            node = child
    return output


def evaluate(model, loader, device):
    model.eval(); predictions, targets = [], []
    with torch.inference_mode():
        for batch in loader:
            labels = batch.pop("labels")
            batch = {key: value.to(device) for key, value in batch.items()}
            predictions.extend(predict_labels(model(**batch)))
            targets.extend(labels)
    return f1_score(targets, predictions, labels=ALL_CLASSES, average="macro", zero_division=0), targets, predictions


def main():
    args = parse_args(); os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    samples = [json.loads(line) for line in open(Path(args.data_dir) / "train.jsonl", encoding="utf-8") if line.strip()]
    with open(Path(args.data_dir) / "train_labels.csv", newline="", encoding="utf-8") as f:
        label_map = {row["id"]: row["action"] for row in csv.DictReader(f)}
    with open(args.split_path, newline="", encoding="utf-8") as f:
        split = {row["id"]: row["split"] for row in csv.DictReader(f)}
    train_rows, val_rows = [], []
    for sample in samples:
        row = (model_text(sample), label_map[sample["id"]])
        (train_rows if split[sample["id"]] == "train" else val_rows).append(row)
    tokenizer = AutoTokenizer.from_pretrained(args.base_encoder)
    collate = make_hierarchy_collate(tokenizer, args.max_length)
    train_loader = DataLoader(HierarchyDataset(*zip(*train_rows)), batch_size=args.batch_size, shuffle=True,
                              collate_fn=collate, num_workers=0)
    val_loader = DataLoader(HierarchyDataset(*zip(*val_rows)), batch_size=args.eval_batch_size, shuffle=False,
                            collate_fn=collate, num_workers=0)
    model = HierarchyModel(args.base_encoder)
    # CUDA is the primary training target; MPS is retained for Apple hosts.
    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")); model.to(device)
    node_weights = {}
    train_labels = [label for _, label in train_rows]
    for node, classes in NODE_CLASSES.items():
        counts = Counter(NODE_TARGETS[node][label] for label in train_labels if NODE_TARGETS[node][label] >= 0)
        total = sum(counts.values())
        node_weights[node] = torch.tensor(
            [np.sqrt(total / (len(classes) * counts[index])) for index in range(len(classes))],
            dtype=torch.float32, device=device,
        )
    optimizer = torch.optim.AdamW([
        {"params": model.encoder.parameters(), "lr": args.encoder_lr, "weight_decay": 0.01},
        {"params": model.heads.parameters(), "lr": args.head_lr, "weight_decay": 0.01},
    ])
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, int(total_steps * 0.03), total_steps)
    best_score = -1.0; result_rows = []
    print(f"device={device} train={len(train_rows)} val={len(val_rows)} nodes={len(TREE)} steps={total_steps}", flush=True)
    for epoch in range(1, args.epochs + 1):
        model.train(); total_loss = 0.0; start = time.time()
        for step, batch in enumerate(train_loader, start=1):
            labels = batch.pop("labels")
            batch = {key: value.to(device) for key, value in batch.items()}
            optimizer.zero_grad(set_to_none=True); logits = model(**batch); losses = []
            for node, node_logits in logits.items():
                targets = torch.tensor([NODE_TARGETS[node][label] for label in labels], dtype=torch.long, device=device)
                if (targets >= 0).any():
                    losses.append(F.cross_entropy(node_logits, targets, weight=node_weights[node], ignore_index=-100))
            loss = torch.stack(losses).mean(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step(); scheduler.step()
            total_loss += loss.item()
            if step == 1 or step % 50 == 0:
                print(f"epoch={epoch} step={step}/{len(train_loader)} loss={total_loss/step:.4f} elapsed={time.time()-start:.1f}s", flush=True)
        score, targets, predictions = evaluate(model, val_loader, device); elapsed = time.time()-start
        result_rows.append({"epoch": epoch, "macro_f1": score, "loss": total_loss/len(train_loader), "seconds": elapsed})
        print(f"epoch={epoch} macro_f1={score:.6f} seconds={elapsed:.1f}", flush=True)
        (out_dir / f"epoch_{epoch}_classification_report.txt").write_text(
            classification_report(targets, predictions, labels=ALL_CLASSES, digits=4, zero_division=0), encoding="utf-8")
        if score > best_score:
            best_score = score; save_dir = out_dir / "best_encoder"; save_dir.mkdir(parents=True, exist_ok=True)
            model.encoder.save_pretrained(save_dir, safe_serialization=True); tokenizer.save_pretrained(save_dir)
            torch.save(model.heads.state_dict(), save_dir / "hierarchy_heads.pt")
    with (out_dir / "modeling_l_results.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "macro_f1", "loss", "seconds"])
        writer.writeheader(); writer.writerows(result_rows)


if __name__ == "__main__": main()
