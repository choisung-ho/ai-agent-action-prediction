#!/usr/bin/env python3
"""Strictly nested E5-teacher -> no-E5-student sequence distillation.

Why this exists
---------------
The convenient global teacher OOF table is *not* sufficient to audit a
student's outer GroupKFold score: a teacher that produced a soft target for an
outer-training row may itself have trained on labels from the outer validation
sessions.  This utility prevents that indirect path.

For every outer student fold it:

1. partitions only the outer-training sessions into inner group folds;
2. fits an E5 teacher on each inner-training partition and predicts the inner
   held-out partition, yielding cross-fitted soft targets for every student
   training row;
3. trains a no-E5 student on those targets and evaluates it on the untouched
   outer validation sessions.

The teacher never sees an outer-validation session, either directly or through
the soft targets it emits.  The script writes experiment artifacts only.
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

from train_modeling_ai_sequence_fusion import ALL_CLASSES, SequenceFusion, predict, sequence_tensors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--base-oof", default="experiments/modeling_n_refit/oof_stack_features.npy")
    parser.add_argument("--e5-embeddings", default="experiments/modeling_f/e5_embeddings_len128.npy")
    parser.add_argument("--out-dir", default="experiments/modeling_ai_sequence_nested_distill")
    parser.add_argument("--outer-folds", type=int, default=5)
    parser.add_argument("--inner-folds", type=int, default=4)
    parser.add_argument("--teacher-epochs", type=int, default=16,
                        help="Fixed epochs; no inner-valid early stopping can leak its labels into its soft targets.")
    parser.add_argument("--student-epochs", type=int, default=28)
    parser.add_argument("--student-patience", type=int, default=5)
    parser.add_argument("--student-fixed-epochs", type=int, default=None,
                        help="If set, train exactly this many student epochs and never select on outer-validation labels.")
    parser.add_argument("--student-hard-loss", choices=("balanced_ce", "balanced_focal"), default="balanced_ce",
                        help="Experiment-only hard-label loss. balanced_focal keeps the existing sqrt-balanced CE and adds a mild focal factor.")
    parser.add_argument("--focal-gamma", type=float, default=0.75,
                        help="Focal exponent used only with --student-hard-loss=balanced_focal.")
    parser.add_argument("--distill", type=float, default=0.35)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def macro(target: np.ndarray, probability: np.ndarray) -> float:
    return float(f1_score(target, probability.argmax(axis=1), labels=np.arange(len(ALL_CLASSES)), average="macro", zero_division=0))


def weight_for(target: np.ndarray, index: np.ndarray, device: torch.device) -> torch.Tensor:
    counts = np.bincount(target[index], minlength=len(ALL_CLASSES))
    return torch.tensor(np.sqrt(len(index) / (len(ALL_CLASSES) * counts)), dtype=torch.float32, device=device)


def hard_label_loss(
    logits: torch.Tensor,
    hard: torch.Tensor,
    class_weight: torch.Tensor,
    kind: str,
    focal_gamma: float,
) -> torch.Tensor:
    """Existing sqrt-balanced CE, optionally with a normalized focal factor.

    ``balanced_focal`` deliberately preserves the current class balancing and
    label smoothing.  Its per-row modulation is ``(1 - p_true)^gamma`` and
    its denominator is weighted by the same factor, so gamma=0 is exactly the
    existing weighted-mean CE.  This keeps the loss scale stable enough for a
    fair fixed-epoch comparison rather than silently changing the optimizer.
    """
    if kind == "balanced_ce":
        return nn.functional.cross_entropy(logits, hard, weight=class_weight, label_smoothing=0.015)
    if kind != "balanced_focal":
        raise ValueError(f"Unknown hard loss: {kind}")
    if focal_gamma < 0.0:
        raise ValueError("focal_gamma must be non-negative")
    per_row = nn.functional.cross_entropy(
        logits, hard, weight=class_weight, label_smoothing=0.015, reduction="none",
    ).float()
    true_probability = torch.softmax(logits.float(), dim=1).gather(1, hard[:, None]).squeeze(1)
    modulation = (1.0 - true_probability).clamp_min(1.0e-6).pow(focal_gamma)
    denominator = (class_weight[hard].float() * modulation).sum().clamp_min(1.0e-8)
    return (per_row * modulation).sum() / denominator


def fit_teacher(
    tensors: list[torch.Tensor],
    target: np.ndarray,
    train_index: np.ndarray,
    predict_index: np.ndarray,
    teacher_epochs: int,
    batch_size: int,
    seed: int,
    device: torch.device,
) -> np.ndarray:
    """Fixed-epoch E5 teacher, with no labels from ``predict_index`` used."""
    seed_everything(seed)
    model = SequenceFusion(tensors[3].shape[1], tensors[4].shape[1], tensors[5].shape[1]).to(device)
    train_data = TensorDataset(*(value[train_index] for value in tensors), torch.from_numpy(target[train_index]))
    loader = DataLoader(train_data, batch_size=batch_size, shuffle=True,
                        generator=torch.Generator().manual_seed(seed), pin_memory=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.8e-3, weight_decay=1.0e-4)
    class_weight = weight_for(target, train_index, device)
    for _ in range(teacher_epochs):
        model.train()
        for batch in loader:
            *features, hard = [value.to(device, non_blocking=True) for value in batch]
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                loss = nn.functional.cross_entropy(model(*features), hard, weight=class_weight, label_smoothing=0.015)
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0); optimizer.step()
    loader = DataLoader(TensorDataset(*(value[predict_index] for value in tensors)), batch_size=1024, shuffle=False, pin_memory=True)
    probability = predict(model, loader, device)
    del model; torch.cuda.empty_cache()
    return probability


def fit_student(
    tensors: list[torch.Tensor],
    target: np.ndarray,
    soft_target: np.ndarray,
    train_index: np.ndarray,
    valid_index: np.ndarray,
    epochs: int,
    patience: int,
    distill: float,
    batch_size: int,
    seed: int,
    device: torch.device,
    fixed_epochs: int | None = None,
    hard_loss: str = "balanced_ce",
    focal_gamma: float = 0.75,
) -> tuple[np.ndarray, int, float]:
    """Train the portable (E5=zero) student against strictly cross-fit soft labels.

    ``fixed_epochs`` is used for an independently seeded audit where the
    outer validation labels must not influence even epoch selection.  Metrics
    are still reported after fitting, but they are never consulted while
    constructing the model.
    """
    seed_everything(seed)
    model = SequenceFusion(tensors[3].shape[1], tensors[4].shape[1], tensors[5].shape[1]).to(device)
    train_data = TensorDataset(
        *(value[train_index] for value in tensors), torch.from_numpy(target[train_index]), torch.from_numpy(soft_target),
    )
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True,
                              generator=torch.Generator().manual_seed(seed), pin_memory=True)
    valid_loader = DataLoader(TensorDataset(*(value[valid_index] for value in tensors)), batch_size=1024, shuffle=False, pin_memory=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.8e-3, weight_decay=1.0e-4)
    class_weight = weight_for(target, train_index, device)
    if fixed_epochs is not None and fixed_epochs < 1:
        raise ValueError("fixed_epochs must be positive when supplied")
    total_epochs = fixed_epochs if fixed_epochs is not None else epochs
    best_score = -1.0; best_probability: np.ndarray | None = None; best_epoch = 0; stale = 0
    for epoch in range(1, total_epochs + 1):
        model.train()
        for batch in train_loader:
            *features, hard, soft = [value.to(device, non_blocking=True) for value in batch]
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model(*features)
                supervised = hard_label_loss(logits, hard, class_weight, hard_loss, focal_gamma)
                kd_loss = nn.functional.kl_div(torch.log_softmax(logits, dim=1), soft, reduction="batchmean")
                loss = (1.0 - distill) * supervised + distill * kd_loss
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0); optimizer.step()
        if fixed_epochs is None:
            probability = predict(model, valid_loader, device)
            score = macro(target[valid_index], probability)
            print(f"student epoch={epoch} macro_f1={score:.6f}", flush=True)
            if score > best_score:
                best_score = score; best_probability = probability; best_epoch = epoch; stale = 0
            else:
                stale += 1
                if stale >= patience:
                    break
        else:
            print(f"student epoch={epoch} fixed_epoch_training", flush=True)
    if fixed_epochs is not None:
        best_probability = predict(model, valid_loader, device)
        best_score = macro(target[valid_index], best_probability)
        best_epoch = fixed_epochs
        print(f"student fixed_epoch={fixed_epochs} macro_f1={best_score:.6f}", flush=True)
    assert best_probability is not None
    del model; torch.cuda.empty_cache()
    return best_probability, best_epoch, best_score


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for nested distillation")
    if not 0.0 < args.distill < 1.0:
        raise ValueError("--distill must be strictly within (0, 1)")
    if args.focal_gamma < 0.0:
        raise ValueError("--focal-gamma must be non-negative")
    if args.inner_folds < 2 or args.outer_folds < 2:
        raise ValueError("Both fold counts must be at least two")
    seed_everything(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    samples = [json.loads(line) for line in open(Path(args.data_dir) / "train.jsonl", encoding="utf-8") if line.strip()]
    labels = {row["id"]: row["action"] for row in csv.DictReader(open(Path(args.data_dir) / "train_labels.csv", encoding="utf-8"))}
    to_id = {label: index for index, label in enumerate(ALL_CLASSES)}
    target = np.asarray([to_id[labels[sample["id"]]] for sample in samples], dtype=np.int64)
    groups = np.asarray([sample["id"].split("-step_", 1)[0] for sample in samples])
    base = np.load(args.base_oof).astype(np.float32)
    e5 = np.load(args.e5_embeddings).astype(np.float32)
    if len(base) != len(samples) or len(e5) != len(samples):
        raise ValueError("Base/E5 arrays do not align with data")
    action, status, arg, meta = sequence_tensors(samples)
    teacher_tensors = [torch.from_numpy(value) for value in (action, status, arg, base, e5, meta)]
    student_e5 = np.zeros((len(samples), 1), dtype=np.float32)
    student_tensors = [torch.from_numpy(value) for value in (action, status, arg, base, student_e5, meta)]
    device = torch.device("cuda")
    outer = list(GroupKFold(n_splits=args.outer_folds).split(base, target, groups))
    oof = np.zeros((len(samples), len(ALL_CLASSES)), dtype=np.float32)
    nested_soft = np.zeros_like(oof)
    fold_assignment = np.empty(len(samples), dtype=np.int8)
    records: list[dict[str, object]] = []
    began = time.time()
    for outer_number, (outer_train, outer_valid) in enumerate(outer, start=1):
        print(f"OUTER {outer_number}/{args.outer_folds}: train={len(outer_train)} valid={len(outer_valid)}", flush=True)
        fold_assignment[outer_valid] = outer_number - 1
        soft = np.zeros((len(outer_train), len(ALL_CLASSES)), dtype=np.float32)
        seen = np.zeros(len(outer_train), dtype=bool)
        inner = list(GroupKFold(n_splits=args.inner_folds).split(outer_train, target[outer_train], groups[outer_train]))
        inner_records: list[dict[str, int]] = []
        outer_valid_groups = set(groups[outer_valid].tolist())
        for inner_number, (inner_train_local, inner_valid_local) in enumerate(inner, start=1):
            inner_train = outer_train[inner_train_local]
            inner_valid = outer_train[inner_valid_local]
            overlap = outer_valid_groups.intersection(groups[inner_train].tolist())
            if overlap:
                raise AssertionError("Outer validation groups entered teacher training")
            probability = fit_teacher(
                teacher_tensors, target, inner_train, inner_valid,
                args.teacher_epochs, args.batch_size, args.seed + outer_number * 100 + inner_number, device,
            )
            soft[inner_valid_local] = probability
            seen[inner_valid_local] = True
            inner_records.append({"inner_fold": inner_number, "teacher_train_rows": int(len(inner_train)),
                                  "teacher_predict_rows": int(len(inner_valid)), "outer_validation_group_overlap": int(len(overlap))})
            print(f"outer={outer_number} inner_teacher={inner_number}/{args.inner_folds} complete", flush=True)
        if not seen.all() or np.any(soft.sum(axis=1) <= 0.0):
            raise AssertionError("Each outer-training student row needs exactly one inner cross-fit teacher target")
        nested_soft[outer_train] = soft
        probability, best_epoch, fold_score = fit_student(
            student_tensors, target, soft, outer_train, outer_valid,
            args.student_epochs, args.student_patience, args.distill, args.batch_size,
            args.seed + outer_number, device, args.student_fixed_epochs,
            args.student_hard_loss, args.focal_gamma,
        )
        oof[outer_valid] = probability
        records.append({
            "outer_fold": outer_number, "student_train_rows": int(len(outer_train)), "student_valid_rows": int(len(outer_valid)),
            "student_best_epoch": int(best_epoch), "student_macro_f1": float(fold_score), "inner_teacher_folds": inner_records,
        })
        print(f"OUTER {outer_number} DONE macro_f1={fold_score:.6f} best_epoch={best_epoch}", flush=True)

    result = {
        "macro_f1": macro(target, oof),
        "outer_folds": args.outer_folds, "inner_folds": args.inner_folds,
        "teacher_epochs_fixed": args.teacher_epochs, "student_distill": args.distill,
        "student_fixed_epochs": args.student_fixed_epochs,
        "student_hard_loss": args.student_hard_loss, "focal_gamma": args.focal_gamma,
        "student_runtime_input": "action/status/args + base OOF + metadata + scalar-zero E5 branch; no E5 embeddings or teacher at inference",
        "strictness": "For every outer fold, each teacher model was fit only on a subset of outer-train sessions. Teacher soft targets for student training therefore exclude every outer-validation session label.",
        "folds": records, "seconds": round(time.time() - began, 2),
    }
    np.save(out / "oof_probabilities.npy", oof)
    np.save(out / "oof_targets.npy", target)
    np.save(out / "outer_fold_assignment.npy", fold_assignment)
    np.save(out / "nested_teacher_soft_targets.npy", nested_soft)
    (out / "results.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
