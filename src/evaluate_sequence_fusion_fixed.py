#!/usr/bin/env python3
"""Evaluate the action-history GRU fusion on the fixed group-held-out split.

This is an experiment-only companion to ``train_modeling_ai_sequence_fusion``.
It deliberately reads the existing leakage-safe base OOF margins and writes
only under ``experiments/``.  It is useful for measuring the exact terminal
(i.e. future-history-unrecoverable) subset that is the live fallback path.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import classification_report, f1_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from train_modeling_ai_sequence_fusion import ALL_CLASSES, SequenceFusion, macro, sequence_tensors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--split", default="splits/common_split.csv")
    parser.add_argument("--out-dir", default="experiments/modeling_ai_sequence_fusion_fixed")
    parser.add_argument("--base-oof", default="experiments/modeling_n_refit/oof_stack_features.npy")
    parser.add_argument("--e5-embeddings", default="experiments/modeling_f/e5_embeddings_len128.npy")
    parser.add_argument("--no-e5", action="store_true", help="Use a learned constant E5 branch for deployability ablation.")
    parser.add_argument("--epochs", type=int, default=28)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--fixed-epochs", type=int, default=None,
                        help="If set, fit exactly this many epochs without selecting on fixed-validation labels.")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--teacher-oof", default=None,
                        help="Optional group-OOF teacher probabilities used only as training targets.")
    parser.add_argument("--distill", type=float, default=0.0,
                        help="Teacher KL-loss coefficient in [0, 1); zero preserves hard-label evaluation.")
    return parser.parse_args()


@torch.inference_mode()
def predict(model: nn.Module, loader: DataLoader, device: torch.device) -> np.ndarray:
    model.eval()
    chunks: list[np.ndarray] = []
    for batch in loader:
        values = [value.to(device, non_blocking=True) for value in batch]
        chunks.append(torch.softmax(model(*values), dim=1).float().cpu().numpy())
    return np.concatenate(chunks)


def terminal_mask(ids: list[str]) -> np.ndarray:
    """A terminal session row is precisely unrecoverable by later history."""
    parsed: list[tuple[str, int]] = []
    maxima: dict[str, int] = {}
    pattern = re.compile(r"^(.*)-step_(\d+)$")
    for sample_id in ids:
        match = pattern.match(sample_id)
        if not match:
            parsed.append((sample_id, -1))
            continue
        session, step = match.group(1), int(match.group(2))
        parsed.append((session, step))
        maxima[session] = max(maxima.get(session, -1), step)
    return np.asarray([step >= 0 and maxima.get(session) == step for session, step in parsed])


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this experiment")
    if not 0.0 <= args.distill < 1.0:
        raise ValueError("--distill must be in [0, 1)")
    if args.distill > 0.0 and not args.teacher_oof:
        raise ValueError("--distill requires --teacher-oof")
    if args.fixed_epochs is not None and args.fixed_epochs < 1:
        raise ValueError("--fixed-epochs must be positive")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    samples = [json.loads(line) for line in open(Path(args.data_dir) / "train.jsonl", encoding="utf-8") if line.strip()]
    labels = {row["id"]: row["action"] for row in csv.DictReader(open(Path(args.data_dir) / "train_labels.csv", encoding="utf-8"))}
    partition = {row["id"]: row["split"] for row in csv.DictReader(open(args.split, encoding="utf-8"))}
    label_to_id = {name: index for index, name in enumerate(ALL_CLASSES)}
    y = np.asarray([label_to_id[labels[sample["id"]]] for sample in samples], dtype=np.int64)
    train_index = np.asarray([i for i, sample in enumerate(samples) if partition[sample["id"]] == "train"])
    valid_index = np.asarray([i for i, sample in enumerate(samples) if partition[sample["id"]] == "val"])

    base = np.load(args.base_oof).astype(np.float32)
    e5 = np.zeros((len(samples), 1), dtype=np.float32) if args.no_e5 else np.load(args.e5_embeddings).astype(np.float32)
    if len(base) != len(samples) or len(e5) != len(samples):
        raise ValueError("OOF/embedding rows do not align with training data")
    teacher: np.ndarray | None = None
    if args.teacher_oof:
        teacher = np.load(args.teacher_oof).astype(np.float32)
        if teacher.shape != (len(samples), len(ALL_CLASSES)):
            raise ValueError(f"Teacher OOF must be {(len(samples), len(ALL_CLASSES))}, got {teacher.shape}")
    action, status, arg, meta = sequence_tensors(samples)
    tensors = [torch.from_numpy(value) for value in (action, status, arg, base, e5, meta)]

    device = torch.device("cuda")
    model = SequenceFusion(base.shape[1], e5.shape[1], meta.shape[1]).to(device)
    counts = np.bincount(y[train_index], minlength=len(ALL_CLASSES))
    class_weight = torch.tensor(np.sqrt(len(train_index) / (len(ALL_CLASSES) * counts)), dtype=torch.float32, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.8e-3, weight_decay=1.0e-4)
    if teacher is None:
        train_data = TensorDataset(*(item[train_index] for item in tensors), torch.from_numpy(y[train_index]))
    else:
        train_data = TensorDataset(
            *(item[train_index] for item in tensors), torch.from_numpy(y[train_index]), torch.from_numpy(teacher[train_index]),
        )
    valid_data = TensorDataset(*(item[valid_index] for item in tensors))
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True,
                              generator=torch.Generator().manual_seed(args.seed), pin_memory=True)
    valid_loader = DataLoader(valid_data, batch_size=1024, shuffle=False, pin_memory=True)

    best_score = -1.0
    best_probability: np.ndarray | None = None
    stale = 0
    started = time.time()
    total_epochs = args.fixed_epochs if args.fixed_epochs is not None else args.epochs
    for epoch in range(1, total_epochs + 1):
        model.train()
        for batch in train_loader:
            values = [value.to(device, non_blocking=True) for value in batch]
            if teacher is None:
                *features, target = values
                soft_target = None
            else:
                *features, target, soft_target = values
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model(*features)
                hard_loss = nn.functional.cross_entropy(logits, target, weight=class_weight, label_smoothing=0.015)
                if soft_target is None:
                    loss = hard_loss
                else:
                    teacher_loss = nn.functional.kl_div(torch.log_softmax(logits, dim=1), soft_target, reduction="batchmean")
                    loss = (1.0 - args.distill) * hard_loss + args.distill * teacher_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
        if args.fixed_epochs is None:
            probability = predict(model, valid_loader, device)
            score = macro(y[valid_index], probability)
            print(f"epoch={epoch} macro_f1={score:.6f}", flush=True)
            if score > best_score:
                best_score = score
                best_probability = probability.copy()
                stale = 0
            else:
                stale += 1
                if stale >= args.patience:
                    break
        else:
            print(f"epoch={epoch} fixed_epoch_training", flush=True)
    if args.fixed_epochs is not None:
        best_probability = predict(model, valid_loader, device)
        best_score = macro(y[valid_index], best_probability)
        stale = 0
        print(f"fixed_epoch={args.fixed_epochs} macro_f1={best_score:.6f}", flush=True)
    assert best_probability is not None

    valid_ids = [samples[i]["id"] for i in valid_index]
    truth = y[valid_index]
    terminal = terminal_mask(valid_ids)
    prediction = best_probability.argmax(axis=1)
    result = {
        "macro_f1": float(macro(truth, best_probability)),
        "terminal_macro_f1": float(f1_score(truth[terminal], prediction[terminal], labels=np.arange(len(ALL_CLASSES)), average="macro", zero_division=0)),
        "terminal_accuracy": float(np.mean(truth[terminal] == prediction[terminal])),
        "n_validation": int(len(valid_index)),
        "n_terminal": int(terminal.sum()),
        "best_epoch": args.fixed_epochs if args.fixed_epochs is not None else epoch - stale,
        "epochs_ran": epoch,
        "fixed_epochs": args.fixed_epochs,
        "seconds": round(time.time() - started, 2),
        "feature_leakage": "base inputs are precomputed group-OOF margins; all other input features are inference-safe",
        "distillation": {
            "enabled": teacher is not None and args.distill > 0.0,
            "coefficient": args.distill,
            "teacher_oof": args.teacher_oof,
            "teacher_usage": "training-only group-OOF soft targets; teacher is absent at inference",
        },
    }
    np.save(out / "valid_probabilities.npy", best_probability)
    np.save(out / "valid_indices.npy", valid_index)
    np.save(out / "terminal_mask.npy", terminal)
    (out / "results.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (out / "classification_report.txt").write_text(classification_report(
        truth, prediction, labels=np.arange(len(ALL_CLASSES)), target_names=ALL_CLASSES, digits=4, zero_division=0,
    ), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
