#!/usr/bin/env python3
"""Create common-train-only rich-GRU teacher targets for strict distillation.

This avoids the key stacking leak of using the global rich-GRU OOF signal as a
soft target: for every common-train row, its teacher was trained *only* on
other common-train sessions.  The common validation sessions are never read by
any teacher fit.  The resulting full-length array has zeros on validation rows
because only common-train targets are valid and needed by the student.

It is an experiment artifact only.  No live candidate, inference script or ZIP
is touched.
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

from train_modeling_ai_sequence_fusion import ALL_CLASSES, SequenceFusion, sequence_tensors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--split", default="splits/common_split.csv")
    parser.add_argument("--base-oof", default="experiments/modeling_n_refit/oof_stack_features.npy")
    parser.add_argument("--e5-embeddings", default="experiments/modeling_f/e5_embeddings_len128.npy")
    parser.add_argument("--out-dir", default="experiments/strict_common_train_teacher")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=26)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=142)
    return parser.parse_args()


def macro(y: np.ndarray, p: np.ndarray) -> float:
    return float(f1_score(y, p.argmax(1), labels=np.arange(len(ALL_CLASSES)), average="macro", zero_division=0))


@torch.inference_mode()
def predict(model: nn.Module, loader: DataLoader, device: torch.device) -> np.ndarray:
    model.eval(); chunks: list[np.ndarray] = []
    for batch in loader:
        batch = [value.to(device, non_blocking=True) for value in batch]
        chunks.append(torch.softmax(model(*batch), dim=1).float().cpu().numpy())
    return np.concatenate(chunks)


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the teacher construction")
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)

    samples = [json.loads(line) for line in open(Path(args.data_dir) / "train.jsonl", encoding="utf-8") if line.strip()]
    labels = {row["id"]: row["action"] for row in csv.DictReader(open(Path(args.data_dir) / "train_labels.csv", encoding="utf-8"))}
    split = {row["id"]: row["split"] for row in csv.DictReader(open(args.split, encoding="utf-8"))}
    label_to_id = {label: index for index, label in enumerate(ALL_CLASSES)}
    y_full = np.asarray([label_to_id[labels[str(sample["id"])]] for sample in samples], dtype=np.int64)
    train_index = np.asarray([i for i, sample in enumerate(samples) if split[str(sample["id"])] == "train"], dtype=np.int64)
    groups = np.asarray([str(samples[i]["id"]).split("-step_", 1)[0] for i in train_index])
    y = y_full[train_index]
    base = np.load(args.base_oof).astype(np.float32, copy=False)
    e5 = np.load(args.e5_embeddings).astype(np.float32, copy=False)
    if base.shape != (len(samples), 97) or e5.shape[0] != len(samples):
        raise ValueError("input cache shape mismatch")
    action, status, arg, meta = sequence_tensors(samples)
    tensors = [torch.from_numpy(value) for value in (action, status, arg, base, e5, meta)]

    folds = list(GroupKFold(n_splits=args.folds).split(np.zeros(len(train_index)), y, groups))
    local_oof = np.zeros((len(train_index), len(ALL_CLASSES)), dtype=np.float32)
    records: list[dict[str, float | int]] = []
    device = torch.device("cuda")
    for fold, (local_train, local_valid) in enumerate(folds, start=1):
        tr = train_index[local_train]; va = train_index[local_valid]
        started = time.time()
        model = SequenceFusion(base.shape[1], e5.shape[1], meta.shape[1]).to(device)
        counts = np.bincount(y_full[tr], minlength=len(ALL_CLASSES))
        weights = torch.tensor(np.sqrt(len(tr) / (len(ALL_CLASSES) * counts)), dtype=torch.float32, device=device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1.8e-3, weight_decay=1.0e-4)
        train_data = TensorDataset(*(value[tr] for value in tensors), torch.from_numpy(y_full[tr]))
        valid_data = TensorDataset(*(value[va] for value in tensors))
        loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True,
                            generator=torch.Generator().manual_seed(args.seed + fold), pin_memory=True)
        valid_loader = DataLoader(valid_data, batch_size=1024, shuffle=False, pin_memory=True)
        best = -1.0; best_p: np.ndarray | None = None; stale = 0
        for epoch in range(1, args.epochs + 1):
            model.train()
            for batch in loader:
                *features, target = [value.to(device, non_blocking=True) for value in batch]
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    loss = nn.functional.cross_entropy(model(*features), target, weight=weights, label_smoothing=0.015)
                loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0); optimizer.step()
            current = predict(model, valid_loader, device)
            value = macro(y_full[va], current)
            print(f"fold={fold} epoch={epoch} macro={value:.6f}", flush=True)
            if value > best:
                best = value; best_p = current; stale = 0
            else:
                stale += 1
                if stale >= args.patience:
                    break
        assert best_p is not None
        local_oof[local_valid] = best_p
        records.append({"fold": fold, "macro_f1": best, "epochs": epoch, "seconds": round(time.time() - started, 2)})
        print(f"fold={fold} BEST={best:.6f}", flush=True)
        del model; torch.cuda.empty_cache()

    full = np.zeros((len(samples), len(ALL_CLASSES)), dtype=np.float32)
    full[train_index] = local_oof
    np.save(out / "train_inner_oof_probabilities.npy", local_oof)
    np.save(out / "full_teacher_trainonly.npy", full)
    np.save(out / "train_indices.npy", train_index)
    summary = {
        "common_train_inner_oof_macro_f1": macro(y, local_oof),
        "rows": {"all": len(samples), "common_train": len(train_index), "common_val_excluded": len(samples) - len(train_index)},
        "folds": records,
        "teacher_is_strict_for_common_holdout": True,
        "teacher_target_contract": "For every common-train row, its probability comes from a rich E5-GRU trained solely on other common-train sessions. Validation rows are all zeros and are never used in student training.",
        "note": "The 97 base-margin block remains the pre-existing global OOF cache, so this audit specifically removes teacher-target leakage rather than rebuilding every base stack in a nested loop.",
    }
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
