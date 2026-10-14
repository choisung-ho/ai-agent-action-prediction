#!/usr/bin/env python3
"""LoRA fine-tuning experiment for Qwen2.5 action classification.

All checkpoints, logits, and reports are confined to ``experiments/``.  This
trainer is intentionally not wired into the submission inference package.
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

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch
import torch.nn.functional as F
from peft import LoraConfig, TaskType, get_peft_model
from sklearn.metrics import classification_report, f1_score
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_cosine_schedule_with_warmup

from train_modeling_b import ALL_CLASSES
from experiment_qwen15_embeddings import MODEL, render


class TextDataset(Dataset):
    def __init__(self, texts: list[str], labels: list[int]):
        self.texts = texts
        self.labels = labels

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, index: int):
        return self.texts[index], self.labels[index]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--split-path", default="splits/common_split.csv")
    parser.add_argument("--out-dir", default="experiments/qwen15_lora")
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=320)
    parser.add_argument("--lr", type=float, default=1.5e-4)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_rows(data_dir: Path, split_path: Path):
    samples = [json.loads(line) for line in (data_dir / "train.jsonl").open(encoding="utf-8") if line.strip()]
    labels = {row["id"]: row["action"] for row in csv.DictReader((data_dir / "train_labels.csv").open(encoding="utf-8"))}
    split = {row["id"]: row["split"] for row in csv.DictReader(split_path.open(encoding="utf-8"))}
    label_to_id = {label: index for index, label in enumerate(ALL_CLASSES)}
    train = [(render(sample), label_to_id[labels[sample["id"]]]) for sample in samples if split[sample["id"]] == "train"]
    valid = [(render(sample), label_to_id[labels[sample["id"]]]) for sample in samples if split[sample["id"]] == "val"]
    return train, valid


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    train_rows, valid_rows = load_rows(Path(args.data_dir), Path(args.split_path))
    (out_dir / "run_config.json").write_text(json.dumps({**vars(args), "train_rows": len(train_rows), "valid_rows": len(valid_rows)}, indent=2) + "\n", encoding="utf-8")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def collate(rows):
        texts, labels = zip(*rows)
        batch = tokenizer(list(texts), padding=True, truncation=True, max_length=args.max_length, return_tensors="pt", pad_to_multiple_of=8)
        batch["labels"] = torch.tensor(labels, dtype=torch.long)
        return batch

    train_loader = DataLoader(TextDataset(*map(list, zip(*train_rows))), batch_size=args.batch_size, shuffle=True,
                              generator=torch.Generator().manual_seed(args.seed), collate_fn=collate, pin_memory=True)
    valid_loader = DataLoader(TextDataset(*map(list, zip(*valid_rows))), batch_size=args.eval_batch_size, shuffle=False,
                              collate_fn=collate, pin_memory=True)
    device = torch.device("cuda")
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model, num_labels=len(ALL_CLASSES), torch_dtype=torch.bfloat16, attn_implementation="sdpa",
        ignore_mismatched_sizes=True,
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    model = get_peft_model(model, LoraConfig(
        task_type=TaskType.SEQ_CLS, r=args.rank, lora_alpha=args.rank * 2, lora_dropout=args.dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        modules_to_save=["score"], bias="none",
    ))
    model.print_trainable_parameters()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model.to(device)
    train_labels = [label for _, label in train_rows]
    counts = np.bincount(train_labels, minlength=len(ALL_CLASSES))
    weights = torch.tensor(np.sqrt(len(train_labels) / (len(ALL_CLASSES) * counts)), dtype=torch.float32, device=device)
    optimizer = torch.optim.AdamW((parameter for parameter in model.parameters() if parameter.requires_grad), lr=args.lr, weight_decay=0.01)
    updates = math.ceil(len(train_loader) / args.grad_accum) * args.epochs
    scheduler = get_cosine_schedule_with_warmup(optimizer, int(updates * args.warmup_ratio), updates)
    # Training runs in bfloat16.  Unlike float16, bfloat16 does not need loss
    # scaling and recent PyTorch builds do not implement GradScaler.unscale_
    # for bfloat16 gradients on CUDA.  Keeping the update path unscaled also
    # makes this experiment work on the RTX 50-series runtime used here.
    best = -1.0; records = []
    for epoch in range(1, args.epochs + 1):
        model.train(); optimizer.zero_grad(set_to_none=True); start = time.time(); loss_sum = 0.0
        for step, batch in enumerate(train_loader, start=1):
            target = batch.pop("labels").to(device, non_blocking=True)
            batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model(**batch).logits
                loss = F.cross_entropy(logits.float(), target, weight=weights, label_smoothing=0.01) / args.grad_accum
            loss.backward(); loss_sum += loss.item() * args.grad_accum
            if step % args.grad_accum == 0 or step == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step(); optimizer.zero_grad(set_to_none=True); scheduler.step()
            if step == 1 or step % 250 == 0:
                print(f"epoch={epoch} step={step}/{len(train_loader)} loss={loss_sum / step:.4f} elapsed={time.time()-start:.1f}s", flush=True)
        model.eval(); logits_all=[]; target_all=[]
        with torch.inference_mode():
            for batch in valid_loader:
                target = batch.pop("labels")
                batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    logits_all.append(model(**batch).logits.float().cpu().numpy())
                target_all.append(target.numpy())
        logits = np.concatenate(logits_all); targets = np.concatenate(target_all); prediction = logits.argmax(axis=1)
        score = f1_score(np.asarray(ALL_CLASSES)[targets], np.asarray(ALL_CLASSES)[prediction], labels=ALL_CLASSES, average="macro", zero_division=0)
        record = {"epoch": epoch, "macro_f1": score, "loss": loss_sum / len(train_loader), "seconds": time.time() - start}
        records.append(record); print(json.dumps(record), flush=True)
        (out_dir / f"epoch_{epoch}_report.txt").write_text(classification_report(np.asarray(ALL_CLASSES)[targets], np.asarray(ALL_CLASSES)[prediction], labels=ALL_CLASSES, digits=4, zero_division=0), encoding="utf-8")
        if score > best:
            best = score
            np.savez_compressed(out_dir / "best_val_logits.npz", logits=logits, target=targets)
            model.save_pretrained(out_dir / "best_adapter")
            tokenizer.save_pretrained(out_dir / "tokenizer")
    (out_dir / "results.json").write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
