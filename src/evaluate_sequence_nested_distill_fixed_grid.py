#!/usr/bin/env python3
"""Fixed-split companion for the strict nested-distillation grid trainer.

Teachers are cross-fitted only inside common-train.  The fixed validation
sessions are excluded from every teacher and student update.  Each student
uses a predeclared fixed epoch; this program performs no early stopping or
hyperparameter search.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import GroupKFold

from train_modeling_ai_sequence_fusion import ALL_CLASSES, sequence_tensors
from train_modeling_ai_sequence_nested_distill import fit_student, fit_teacher, macro, seed_everything
from train_modeling_ai_sequence_nested_distill_grid import Candidate, parse_candidate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--split", default="splits/common_split.csv")
    parser.add_argument("--base-oof", default="experiments/modeling_n_refit/oof_stack_features.npy")
    parser.add_argument("--e5-embeddings", default="experiments/modeling_f/e5_embeddings_len128.npy")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--inner-folds", type=int, default=4)
    parser.add_argument("--teacher-epochs", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--teacher-seed", type=int, default=42)
    parser.add_argument("--candidate", type=parse_candidate, action="append", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for strict nested distillation")
    if args.inner_folds < 2 or args.teacher_epochs < 1:
        raise ValueError("Require at least two inner folds and positive teacher epochs")
    if len({candidate.name for candidate in args.candidate}) != len(args.candidate):
        raise ValueError("Candidate names must be unique")
    seed_everything(args.teacher_seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    samples = [
        json.loads(line)
        for line in (Path(args.data_dir) / "train.jsonl").open(encoding="utf-8")
        if line.strip()
    ]
    labels = {
        row["id"]: row["action"]
        for row in csv.DictReader((Path(args.data_dir) / "train_labels.csv").open(encoding="utf-8"))
    }
    split = {
        row["id"]: row["split"]
        for row in csv.DictReader(Path(args.split).open(encoding="utf-8"))
    }
    to_id = {label: index for index, label in enumerate(ALL_CLASSES)}
    target = np.asarray([to_id[labels[sample["id"]]] for sample in samples], dtype=np.int64)
    groups = np.asarray([sample["id"].split("-step_", 1)[0] for sample in samples])
    common_train = np.asarray([split[sample["id"]] == "train" for sample in samples])
    train_index = np.flatnonzero(common_train)
    valid_index = np.flatnonzero(~common_train)
    base = np.load(args.base_oof, allow_pickle=False).astype(np.float32)
    e5 = np.load(args.e5_embeddings, allow_pickle=False).astype(np.float32)
    if len(base) != len(samples) or len(e5) != len(samples):
        raise ValueError("Base/E5 input alignment failed")
    action, status, arg, meta = sequence_tensors(samples)
    teacher_tensors = [torch.from_numpy(value) for value in (action, status, arg, base, e5, meta)]
    zero_e5 = np.zeros((len(samples), 1), dtype=np.float32)
    student_tensors = [torch.from_numpy(value) for value in (action, status, arg, base, zero_e5, meta)]
    device = torch.device("cuda")
    soft = np.zeros((len(train_index), len(ALL_CLASSES)), dtype=np.float32)
    seen = np.zeros(len(train_index), dtype=bool)
    valid_groups = set(groups[valid_index].tolist())
    teacher_records = []
    began = time.time()
    inner = list(GroupKFold(n_splits=args.inner_folds).split(
        train_index, target[train_index], groups[train_index],
    ))
    for number, (inner_train_local, inner_valid_local) in enumerate(inner, start=1):
        inner_train = train_index[inner_train_local]
        inner_valid = train_index[inner_valid_local]
        overlap = valid_groups.intersection(groups[inner_train].tolist())
        if overlap:
            raise AssertionError("Fixed validation group entered teacher training")
        probability = fit_teacher(
            teacher_tensors, target, inner_train, inner_valid, args.teacher_epochs,
            args.batch_size, args.teacher_seed + 100 + number, device,
        )
        soft[inner_valid_local] = probability
        seen[inner_valid_local] = True
        teacher_records.append({
            "inner_fold": number,
            "teacher_train_rows": int(len(inner_train)),
            "teacher_predict_rows": int(len(inner_valid)),
            "fixed_validation_group_overlap": int(len(overlap)),
        })
        print(f"inner_teacher={number}/{args.inner_folds} complete", flush=True)
    if not seen.all() or np.any(soft.sum(axis=1) <= 0.0):
        raise AssertionError("Missing common-train teacher targets")
    np.save(out / "common_train_teacher_soft.npy", soft)
    terminal = np.load(
        "experiments/modeling_ai_sequence_fusion_fixed_noe5/terminal_mask.npy",
        allow_pickle=False,
    ).astype(bool)
    source = np.asarray([
        "au" if samples[index]["id"].startswith("sess_au_") else "sim"
        for index in valid_index
    ])
    summaries = {}
    for candidate in args.candidate:
        probability, epoch, score = fit_student(
            student_tensors, target, soft, train_index, valid_index,
            candidate.epochs, 0, candidate.distill, args.batch_size,
            candidate.seed + 999, device, fixed_epochs=candidate.epochs,
        )
        directory = out / "candidates" / candidate.name
        directory.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            directory / "valid.npz",
            ids=np.asarray([samples[index]["id"] for index in valid_index]),
            target=target[valid_index],
            terminal=terminal,
            source=source,
            student=probability,
        )
        summary = {
            "name": candidate.name,
            "distill": candidate.distill,
            "student_fixed_epochs": candidate.epochs,
            "student_seed": candidate.seed,
            "student_all_macro_f1": float(score),
            "student_terminal_macro_f1": macro(
                target[valid_index][terminal], probability[terminal],
            ),
        }
        (directory / "results.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
        )
        summaries[candidate.name] = summary
        print(
            f"candidate={candidate.name} macro_f1={score:.6f} "
            f"terminal={summary['student_terminal_macro_f1']:.6f}",
            flush=True,
        )
    report = {
        "strictness": (
            "Every teacher is fit only in common-train; every student uses fixed epochs. "
            "Fixed validation labels are used solely for final measurement."
        ),
        "teacher_inner_folds": teacher_records,
        "base_oof_caveat": (
            "The inherited global 97-column base OOF table is not fully nested within "
            "common-train."
        ),
        "seconds": round(time.time() - began, 2),
        "candidates": summaries,
    }
    (out / "grid_results.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
