#!/usr/bin/env python3
"""Build a strict common-train-only no-E5 student OOF/cache for pair-gate audit.

This is deliberately an experiment-only builder.  It never reads or writes
``model/``, ``script.py``, any submission ZIP, or ``build_zip.py``.

For every common-train outer student fold, rich E5 teacher targets are made
inside that outer-training partition with a second GroupKFold.  Therefore the
student prediction for a held common-train session has no direct or indirect
teacher-label path from that held session.  The fixed common-validation rows
are excluded from every teacher and student fit.  A fixed-epoch common-train
refit then produces a disjoint fixed probability table.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from train_modeling_ai_sequence_fusion import ALL_CLASSES, SequenceFusion, predict, sequence_tensors
from train_modeling_ai_sequence_nested_distill import fit_student, fit_teacher, seed_everything


ROOT = Path(__file__).resolve().parents[1]
N_CLASSES = len(ALL_CLASSES)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--blocks", default="experiments/au_source_strict_common/strict_blocks.npz")
    parser.add_argument("--teacher-base", default="experiments/modeling_n_refit/oof_stack_features.npy")
    parser.add_argument("--teacher-e5", default="experiments/modeling_f/e5_embeddings_len128.npy")
    parser.add_argument("--common-teacher", default="experiments/strict_common_train_teacher/train_inner_oof_probabilities.npy")
    parser.add_argument("--out-dir", default="experiments/b43_pair_gate_strict_common")
    parser.add_argument("--outer-folds", type=int, default=5)
    parser.add_argument("--inner-folds", type=int, default=4)
    parser.add_argument("--teacher-epochs", type=int, default=16)
    parser.add_argument("--student-epochs", type=int, default=12)
    parser.add_argument("--distill", type=float, default=0.35)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def terminal_mask(ids: np.ndarray) -> np.ndarray:
    maximum: dict[str, int] = {}
    parsed: list[tuple[str, int]] = []
    for sample_id in ids.astype(str):
        session, raw_step = sample_id.rsplit("-step_", 1)
        step = int(raw_step)
        maximum[session] = max(maximum.get(session, -1), step)
        parsed.append((session, step))
    return np.asarray([step == maximum[session] for session, step in parsed], dtype=bool)


def macro(y: np.ndarray, probability: np.ndarray, mask: np.ndarray | None = None) -> float:
    if mask is None:
        mask = np.ones(len(y), dtype=bool)
    return float(f1_score(
        y[mask], probability[mask].argmax(axis=1), labels=np.arange(N_CLASSES),
        average="macro", zero_division=0,
    ))


def student_tensors(samples: list[dict], base: np.ndarray) -> list[torch.Tensor]:
    action, status, argument, meta = sequence_tensors(samples)
    zero_e5 = np.zeros((len(samples), 1), dtype=np.float32)
    return [torch.from_numpy(value) for value in (action, status, argument, base, zero_e5, meta)]


def refit_predict(
    train_tensors: list[torch.Tensor], valid_tensors: list[torch.Tensor], y_train: np.ndarray,
    soft: np.ndarray, epochs: int, distill: float, batch_size: int, seed: int, device: torch.device,
) -> np.ndarray:
    """Fit only on common-train rows, then predict fixed rows without reading labels."""
    seed_everything(seed)
    model = SequenceFusion(train_tensors[3].shape[1], train_tensors[4].shape[1], train_tensors[5].shape[1]).to(device)
    counts = np.bincount(y_train, minlength=N_CLASSES)
    if np.any(counts == 0):
        raise ValueError("Common train lacks a class")
    class_weight = torch.tensor(
        np.sqrt(len(y_train) / (N_CLASSES * counts)), dtype=torch.float32, device=device,
    )
    loader = DataLoader(
        TensorDataset(*train_tensors, torch.from_numpy(y_train), torch.from_numpy(soft)),
        batch_size=batch_size, shuffle=True, generator=torch.Generator().manual_seed(seed), pin_memory=True,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.8e-3, weight_decay=1.0e-4)
    for epoch in range(1, epochs + 1):
        model.train()
        for batch in loader:
            *features, hard, soft_target = [value.to(device, non_blocking=True) for value in batch]
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model(*features)
                hard_loss = nn.functional.cross_entropy(logits, hard, weight=class_weight, label_smoothing=0.015)
                teacher_loss = nn.functional.kl_div(torch.log_softmax(logits, dim=1), soft_target, reduction="batchmean")
                loss = (1.0 - distill) * hard_loss + distill * teacher_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
        print(json.dumps({"phase": "fixed_refit", "epoch": epoch}, ensure_ascii=False), flush=True)
    loader = DataLoader(TensorDataset(*valid_tensors), batch_size=1024, shuffle=False, pin_memory=True)
    probability = predict(model, loader, device).astype(np.float32)
    del model
    torch.cuda.empty_cache()
    return probability


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for strict nested distillation")
    if args.outer_folds < 2 or args.inner_folds < 2:
        raise ValueError("outer/inner folds must both be >= 2")
    if not 0.0 < args.distill < 1.0:
        raise ValueError("distill must be in (0, 1)")
    if args.teacher_epochs < 1 or args.student_epochs < 1:
        raise ValueError("epochs must be positive")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    out = (ROOT / args.out_dir).resolve()
    if (ROOT / "model").resolve() in out.parents or out == (ROOT / "model").resolve():
        raise ValueError("Experiment output must not be inside model/")
    out.mkdir(parents=True, exist_ok=True)

    blocks = np.load(ROOT / args.blocks, allow_pickle=False)
    required = {"ids_train", "ids_valid", "target_train", "target_valid", "base_train", "base_valid"}
    if missing := required - set(blocks.files):
        raise ValueError(f"Strict blocks missing {sorted(missing)}")
    ids_train = blocks["ids_train"].astype(str)
    ids_valid = blocks["ids_valid"].astype(str)
    y_train = blocks["target_train"].astype(np.int64)
    y_valid = blocks["target_valid"].astype(np.int64)
    base_train = blocks["base_train"].astype(np.float32)
    base_valid = blocks["base_valid"].astype(np.float32)
    if base_train.shape != (len(ids_train), 97) or base_valid.shape != (len(ids_valid), 97):
        raise ValueError("Expected strict 97-column B42-compatible blocks")

    samples = [json.loads(line) for line in (ROOT / "data/train.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    labels = {row["id"]: row["action"] for row in csv.DictReader((ROOT / "data/train_labels.csv").open(encoding="utf-8"))}
    split = {row["id"]: row["split"] for row in csv.DictReader((ROOT / "splits/common_split.csv").open(encoding="utf-8"))}
    full_ids = np.asarray([str(row["id"]) for row in samples])
    full_groups = np.asarray([value.rsplit("-step_", 1)[0] for value in full_ids])
    to_id = {name: number for number, name in enumerate(ALL_CLASSES)}
    y_full = np.asarray([to_id[labels[value]] for value in full_ids], dtype=np.int64)
    train_global = np.flatnonzero(np.asarray([split[value] == "train" for value in full_ids]))
    valid_global = np.flatnonzero(np.asarray([split[value] == "val" for value in full_ids]))
    if not (
        np.array_equal(full_ids[train_global], ids_train)
        and np.array_equal(full_ids[valid_global], ids_valid)
        and np.array_equal(y_full[train_global], y_train)
        and np.array_equal(y_full[valid_global], y_valid)
    ):
        raise ValueError("Strict blocks do not match the common split")

    # Teacher features follow the prior strict-common teacher contract.  Each
    # teacher fit below receives only common-train global indices; fixed groups
    # never enter a teacher or student fit.
    teacher_base = np.load(ROOT / args.teacher_base, allow_pickle=False).astype(np.float32)
    teacher_e5 = np.load(ROOT / args.teacher_e5, allow_pickle=False).astype(np.float32)
    if teacher_base.shape != (len(samples), 97) or teacher_e5.shape[0] != len(samples):
        raise ValueError("Teacher input cache shape mismatch")
    action, status, argument, meta = sequence_tensors(samples)
    teacher_tensors = [torch.from_numpy(value) for value in (action, status, argument, teacher_base, teacher_e5, meta)]
    train_samples = [samples[index] for index in train_global]
    valid_samples = [samples[index] for index in valid_global]
    train_tensors = student_tensors(train_samples, base_train)
    valid_tensors = student_tensors(valid_samples, base_valid)
    common_teacher = np.load(ROOT / args.common_teacher, allow_pickle=False).astype(np.float32)
    if common_teacher.shape != (len(ids_train), N_CLASSES) or np.any(common_teacher.sum(axis=1) <= 0.0):
        raise ValueError("Common-train cross-fitted teacher table is invalid")

    groups = np.asarray([value.rsplit("-step_", 1)[0] for value in ids_train])
    fixed_groups = np.asarray([value.rsplit("-step_", 1)[0] for value in ids_valid])
    if set(groups).intersection(fixed_groups):
        raise ValueError("Common train/fixed session overlap")
    terminal_train = terminal_mask(ids_train)
    terminal_valid = terminal_mask(ids_valid)
    device = torch.device("cuda")
    begun = time.time()
    outer = list(GroupKFold(n_splits=args.outer_folds).split(np.zeros(len(y_train)), y_train, groups))
    oof = np.zeros((len(ids_train), N_CLASSES), dtype=np.float32)
    assignment = np.empty(len(ids_train), dtype=np.int8)
    records: list[dict[str, object]] = []

    for outer_number, (outer_train_local, outer_valid_local) in enumerate(outer, start=1):
        assignment[outer_valid_local] = outer_number - 1
        outer_train_global = train_global[outer_train_local]
        outer_valid_global = train_global[outer_valid_local]
        blocked_groups = set(groups[outer_valid_local].tolist())
        print(json.dumps({
            "phase": "outer_start", "outer_fold": outer_number, "outer_folds": args.outer_folds,
            "student_train_rows": int(len(outer_train_local)), "student_held_rows": int(len(outer_valid_local)),
            "elapsed_seconds": round(time.time() - begun, 2),
        }, ensure_ascii=False), flush=True)
        soft = np.zeros((len(outer_train_local), N_CLASSES), dtype=np.float32)
        seen = np.zeros(len(outer_train_local), dtype=bool)
        inner_records: list[dict[str, int]] = []
        inner = list(GroupKFold(n_splits=args.inner_folds).split(
            np.zeros(len(outer_train_local)), y_train[outer_train_local], groups[outer_train_local],
        ))
        for inner_number, (inner_train_local, inner_held_local) in enumerate(inner, start=1):
            inner_train_global = outer_train_global[inner_train_local]
            inner_held_global = outer_train_global[inner_held_local]
            overlap = blocked_groups.intersection(set(full_groups[inner_train_global].tolist()))
            if overlap:
                raise AssertionError("Held outer sessions entered teacher fit")
            probability = fit_teacher(
                teacher_tensors, y_full, inner_train_global, inner_held_global,
                args.teacher_epochs, args.batch_size,
                args.seed + outer_number * 100 + inner_number, device,
            )
            soft[inner_held_local] = probability
            seen[inner_held_local] = True
            inner_record = {
                "inner_fold": int(inner_number), "teacher_train_rows": int(len(inner_train_global)),
                "teacher_predict_rows": int(len(inner_held_global)), "outer_held_group_overlap": int(len(overlap)),
            }
            inner_records.append(inner_record)
            print(json.dumps({"phase": "inner_teacher_complete", "outer_fold": outer_number, **inner_record,
                              "elapsed_seconds": round(time.time() - begun, 2)}, ensure_ascii=False), flush=True)
        if not seen.all() or np.any(soft.sum(axis=1) <= 0.0):
            raise AssertionError("Missing nested teacher targets")
        probability, epoch, score = fit_student(
            train_tensors, y_train, soft, outer_train_local, outer_valid_local,
            epochs=args.student_epochs, patience=1, distill=args.distill, batch_size=args.batch_size,
            seed=args.seed + outer_number, device=device, fixed_epochs=args.student_epochs,
        )
        oof[outer_valid_local] = probability
        record = {
            "outer_fold": int(outer_number), "student_train_rows": int(len(outer_train_local)),
            "student_held_rows": int(len(outer_valid_local)), "student_fixed_epoch": int(epoch),
            "student_all_macro_f1": float(score),
            "student_terminal_macro_f1": macro(y_train[outer_valid_local], probability, terminal_train[outer_valid_local]),
            "inner_teacher_folds": inner_records,
            "elapsed_seconds": round(time.time() - begun, 2),
        }
        records.append(record)
        np.save(out / "common_train_oof_checkpoint.npy", oof)
        (out / "progress.json").write_text(json.dumps({"completed_outer_folds": outer_number, "records": records}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"phase": "outer_complete", **record}, ensure_ascii=False), flush=True)

    fixed_started = time.time()
    fixed_probability = refit_predict(
        train_tensors, valid_tensors, y_train, common_teacher, args.student_epochs,
        args.distill, args.batch_size, args.seed + 10_000, device,
    )
    fixed_seconds = time.time() - fixed_started
    report = {
        "scope": "Experiment-only. model/, script.py, build_zip.py, B42.zip, baseline_submit.zip, and all challenger ZIPs were not modified.",
        "contract": {
            "outer_student_oof": "Five session GroupKFold folds inside common-train.",
            "nested_teacher": "For each outer student fold, every rich teacher is fit only on a subpartition of outer-train; held outer sessions are excluded from every teacher fit.",
            "fixed": "All fixed common-validation sessions are excluded from every teacher/student fit; student uses a fixed 12-epoch schedule and fixed labels are scored only after prediction.",
            "student": "Portable no-E5 GRU using strict 97-column common blocks; teacher/E5 values are training-only.",
            "teacher_base_note": "Teacher uses the pre-existing global 97-margin OOF cache, matching the prior strict-common teacher contract; all teacher label fits themselves are common-train-only.",
        },
        "config": {"outer_folds": args.outer_folds, "inner_folds": args.inner_folds,
                   "teacher_epochs": args.teacher_epochs, "student_fixed_epochs": args.student_epochs,
                   "distill": args.distill, "seed": args.seed},
        "counts": {"common_train": int(len(ids_train)), "common_terminal": int(terminal_train.sum()),
                   "fixed": int(len(ids_valid)), "fixed_terminal": int(terminal_valid.sum())},
        "common_nested_oof": {"all_macro_f1": macro(y_train, oof), "terminal_macro_f1": macro(y_train, oof, terminal_train), "folds": records},
        "fixed": {"all_macro_f1": macro(y_valid, fixed_probability),
                  "terminal_macro_f1": macro(y_valid, fixed_probability, terminal_valid),
                  "seconds": round(fixed_seconds, 2)},
        "elapsed_seconds": round(time.time() - begun, 2),
    }
    np.save(out / "common_train_strict_oof_probabilities.npy", oof)
    np.save(out / "common_train_ids.npy", ids_train)
    np.save(out / "common_train_fold.npy", assignment)
    np.save(out / "fixed_strict_probabilities.npy", fixed_probability)
    np.save(out / "fixed_ids.npy", ids_valid)
    np.save(out / "terminal_train.npy", terminal_train)
    np.save(out / "terminal_fixed.npy", terminal_valid)
    (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"phase": "complete", **report}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
