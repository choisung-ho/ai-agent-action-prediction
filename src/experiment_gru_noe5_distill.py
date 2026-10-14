#!/usr/bin/env python3
"""Distill the rich E5 sequence teacher into the existing no-E5 GRU.

The student is deliberately architecture-compatible with the no-E5 sequence
fusion candidate: action/status/argument GRU, 97 OOF base margins and small
metadata.  It does not consume E5 embeddings or teacher probabilities during
inference.  Therefore its weights can follow the existing NumPy GRU export
path if (and only if) this cross-fitted experiment improves robustly.

All artifacts remain under ``experiments/``; no live submission files are
opened for writing.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from train_modeling_ai_sequence_fusion import SequenceFusion, sequence_tensors
from train_modeling_b import ALL_CLASSES


N_CLASSES = len(ALL_CLASSES)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--base-oof", default="experiments/modeling_n_refit/oof_stack_features.npy")
    parser.add_argument("--teacher-oof", default="experiments/modeling_ai_sequence_fusion_rerun/oof_probabilities.npy")
    parser.add_argument("--xgb-oof", default="experiments/modeling_ak_live_xgb_proxy/oof_probabilities.npy")
    parser.add_argument("--out-dir", default="experiments/gru_noe5_distill")
    parser.add_argument("--teacher-weight", type=float, default=0.15)
    parser.add_argument("--temperature", type=float, default=1.5)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=768)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def score(y: np.ndarray, p: np.ndarray, mask: np.ndarray | None = None) -> float:
    if mask is None:
        mask = np.ones(len(y), dtype=bool)
    return float(f1_score(y[mask], p[mask].argmax(1), labels=np.arange(N_CLASSES), average="macro", zero_division=0))


def terminal_mask(ids: list[str]) -> np.ndarray:
    maximum: dict[str, int] = {}
    for sample_id in ids:
        session, _, step = sample_id.rpartition("-step_")
        maximum[session] = max(maximum.get(session, -1), int(step))
    return np.asarray([int(sample_id.rpartition("-step_")[2]) == maximum[sample_id.rpartition("-step_")[0]] for sample_id in ids])


@torch.inference_mode()
def predict(model: nn.Module, loader: DataLoader, device: torch.device) -> np.ndarray:
    model.eval(); outputs: list[np.ndarray] = []
    for values in loader:
        values = [item.to(device, non_blocking=True) for item in values]
        outputs.append(torch.softmax(model(*values), dim=1).float().cpu().numpy())
    return np.concatenate(outputs)


def best_blend(y: np.ndarray, xgb: np.ndarray, teacher: np.ndarray, student: np.ndarray) -> dict[str, float]:
    """Small diagnostic grid; deployment weight needs an independent gate."""
    best: dict[str, float] = {"macro_f1": -1.0}
    for xw in np.arange(0.40, 0.651, 0.025):
        for sw in np.arange(0.0, 0.251, 0.025):
            tw = 1.0 - xw - sw
            if tw < 0:
                continue
            current = score(y, xw * xgb + tw * teacher + sw * student)
            if current > best["macro_f1"]:
                best = {"macro_f1": current, "xgb_weight": float(xw), "teacher_weight": float(tw), "student_weight": float(sw)}
    return best


def main() -> None:
    args = parse_args()
    if not 0 <= args.teacher_weight <= 1:
        raise ValueError("teacher-weight must be in [0, 1]")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the bounded student training run")
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    samples = [json.loads(line) for line in open(Path(args.data_dir) / "train.jsonl", encoding="utf-8") if line.strip()]
    labels = {row["id"]: row["action"] for row in csv.DictReader(open(Path(args.data_dir) / "train_labels.csv", encoding="utf-8"))}
    ids = [str(sample["id"]) for sample in samples]
    label_to_id = {name: index for index, name in enumerate(ALL_CLASSES)}
    y = np.asarray([label_to_id[labels[sample_id]] for sample_id in ids], dtype=np.int64)
    groups = np.asarray([sample_id.split("-step_", 1)[0] for sample_id in ids])
    base = np.load(args.base_oof).astype(np.float32, copy=False)
    teacher = np.load(args.teacher_oof).astype(np.float32, copy=False)
    xgb = np.load(args.xgb_oof).astype(np.float32, copy=False)
    if base.shape != (len(y), 97) or teacher.shape != (len(y), N_CLASSES) or xgb.shape != teacher.shape:
        raise ValueError("OOF artifact shape mismatch")
    action, status, arg, meta = sequence_tensors(samples)
    e5_zero = np.zeros((len(y), 1), dtype=np.float32)
    tensors = [torch.from_numpy(value) for value in (action, status, arg, base, e5_zero, meta)]
    teacher_tensor = torch.from_numpy(teacher)
    folds = list(GroupKFold(n_splits=5).split(base, y, groups))
    oof = np.zeros_like(teacher)
    records: list[dict[str, float | int]] = []
    device = torch.device("cuda")
    for fold, (train_index, valid_index) in enumerate(folds, start=1):
        started = time.time()
        model = SequenceFusion(base.shape[1], 1, meta.shape[1]).to(device)
        counts = np.bincount(y[train_index], minlength=N_CLASSES)
        weights = torch.tensor(np.sqrt(len(train_index) / (N_CLASSES * counts)), dtype=torch.float32, device=device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1.8e-3, weight_decay=1.0e-4)
        train = TensorDataset(*(part[train_index] for part in tensors), torch.from_numpy(y[train_index]), teacher_tensor[train_index])
        valid = TensorDataset(*(part[valid_index] for part in tensors))
        loader = DataLoader(train, batch_size=args.batch_size, shuffle=True,
                            generator=torch.Generator().manual_seed(args.seed + fold), pin_memory=True)
        valid_loader = DataLoader(valid, batch_size=1024, shuffle=False, pin_memory=True)
        best = -1.0; best_p: np.ndarray | None = None; stale = 0
        for epoch in range(1, args.epochs + 1):
            model.train()
            for batch in loader:
                *features, target, soft = [part.to(device, non_blocking=True) for part in batch]
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    logits = model(*features)
                    hard_loss = nn.functional.cross_entropy(logits, target, weight=weights, label_smoothing=0.015)
                    if args.teacher_weight:
                        normalized = torch.clamp(soft, min=1e-7)
                        normalized = normalized / normalized.sum(dim=1, keepdim=True)
                        soft_loss = nn.functional.kl_div(
                            nn.functional.log_softmax(logits / args.temperature, dim=1), normalized,
                            reduction="batchmean",
                        ) * (args.temperature ** 2)
                        loss = (1.0 - args.teacher_weight) * hard_loss + args.teacher_weight * soft_loss
                    else:
                        loss = hard_loss
                loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0); optimizer.step()
            current = predict(model, valid_loader, device)
            value = score(y[valid_index], current)
            print(f"fold={fold} epoch={epoch} macro={value:.6f}", flush=True)
            if value > best:
                best = value; best_p = current; stale = 0
            else:
                stale += 1
                if stale >= args.patience:
                    break
        assert best_p is not None
        oof[valid_index] = best_p
        record = {"fold": fold, "macro_f1": best, "epochs": epoch, "seconds": round(time.time() - started, 2)}
        records.append(record)
        print(f"fold={fold} BEST={best:.6f} sec={record['seconds']}", flush=True)
        del model; torch.cuda.empty_cache()

    terminal = terminal_mask(ids)
    summary = {
        "student_macro_f1": score(y, oof),
        "student_terminal_macro_f1": score(y, oof, terminal),
        "xgb_teacher_equal_macro_f1": score(y, (xgb + teacher) / 2),
        "diagnostic_three_way_best_in_sample_grid": best_blend(y, xgb, teacher, oof),
        "folds": records,
        "teacher_weight": args.teacher_weight,
        "architecture": "existing SequenceFusion(action/status/args GRU + 97 base + 15 metadata + zero E5 scalar)",
        "validation_method": "5-fold GroupKFold by session; soft targets are rich-GRU OOF values and are not inference inputs",
        "deployment_assessment": "No E5 embedding input. Existing NumPy GRU export is directly relevant; this file does not refit/export a candidate.",
    }
    np.save(out / "oof_probabilities.npy", oof)
    np.save(out / "oof_targets.npy", y)
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
