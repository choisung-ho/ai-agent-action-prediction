#!/usr/bin/env python3
"""Strict inner-teacher nested distillation for a small predeclared grid.

The expensive inner E5 teachers are trained once per outer session fold. Each
candidate student is trained immediately from that fold's local soft targets.
This avoids reusing the lossy two-dimensional teacher-target artifact, which
cannot preserve targets for all five outer folds.

This writes experiment artifacts only and never changes a deployment model,
script, staging directory, or ZIP.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import GroupKFold

from train_modeling_ai_sequence_fusion import ALL_CLASSES, sequence_tensors
from train_modeling_ai_sequence_nested_distill import (
    fit_student,
    fit_teacher,
    macro,
    seed_everything,
)


@dataclass(frozen=True)
class Candidate:
    name: str
    distill: float
    epochs: int
    seed: int


def parse_candidate(value: str) -> Candidate:
    pieces = value.split(":")
    if len(pieces) != 4:
        raise argparse.ArgumentTypeError("candidate must be name:distill:epochs:seed")
    name, raw_distill, raw_epochs, raw_seed = pieces
    if not name.replace("_", "").replace("-", "").isalnum():
        raise argparse.ArgumentTypeError("candidate name may contain only letters, digits, _ and -")
    try:
        distill, epochs, seed = float(raw_distill), int(raw_epochs), int(raw_seed)
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error
    if not 0.0 < distill < 1.0 or epochs < 1:
        raise argparse.ArgumentTypeError("distill must be (0,1) and epochs must be positive")
    return Candidate(name, distill, epochs, seed)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--base-oof", default="experiments/modeling_n_refit/oof_stack_features.npy")
    parser.add_argument("--e5-embeddings", default="experiments/modeling_f/e5_embeddings_len128.npy")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--outer-folds", type=int, default=5)
    parser.add_argument("--inner-folds", type=int, default=4)
    parser.add_argument("--teacher-epochs", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--teacher-seed", type=int, default=42)
    parser.add_argument("--candidate", type=parse_candidate, action="append", required=True)
    return parser.parse_args()


def load_data(data_dir: Path) -> tuple[list[dict], np.ndarray, np.ndarray]:
    samples = [
        json.loads(line)
        for line in (data_dir / "train.jsonl").open(encoding="utf-8")
        if line.strip()
    ]
    labels = {
        row["id"]: row["action"]
        for row in csv.DictReader((data_dir / "train_labels.csv").open(encoding="utf-8"))
    }
    to_id = {label: index for index, label in enumerate(ALL_CLASSES)}
    target = np.asarray([to_id[labels[sample["id"]]] for sample in samples], dtype=np.int64)
    groups = np.asarray([sample["id"].split("-step_", 1)[0] for sample in samples])
    return samples, target, groups


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for strict nested distillation")
    if args.outer_folds != 5 or args.inner_folds < 2 or args.teacher_epochs < 1:
        raise ValueError("Require five outer folds, at least two inner folds, and positive teacher epochs")
    if len({candidate.name for candidate in args.candidate}) != len(args.candidate):
        raise ValueError("Candidate names must be unique")
    seed_everything(args.teacher_seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    samples, target, groups = load_data(Path(args.data_dir))
    base = np.load(args.base_oof, allow_pickle=False).astype(np.float32)
    e5 = np.load(args.e5_embeddings, allow_pickle=False).astype(np.float32)
    if len(base) != len(samples) or len(e5) != len(samples):
        raise ValueError("Base and E5 inputs do not align with train rows")
    action, status, arg, meta = sequence_tensors(samples)
    teacher_tensors = [torch.from_numpy(value) for value in (action, status, arg, base, e5, meta)]
    zero_e5 = np.zeros((len(samples), 1), dtype=np.float32)
    student_tensors = [torch.from_numpy(value) for value in (action, status, arg, base, zero_e5, meta)]
    outer = list(GroupKFold(n_splits=args.outer_folds).split(base, target, groups))
    device = torch.device("cuda")
    assignment = np.empty(len(samples), dtype=np.int8)
    teacher_soft = np.zeros((args.outer_folds, len(samples), len(ALL_CLASSES)), dtype=np.float32)
    oof = {
        candidate.name: np.zeros((len(samples), len(ALL_CLASSES)), dtype=np.float32)
        for candidate in args.candidate
    }
    records = {candidate.name: [] for candidate in args.candidate}
    outer_records = []
    began = time.time()
    for outer_number, (outer_train, outer_valid) in enumerate(outer, start=1):
        print(f"OUTER {outer_number}/{args.outer_folds}: train={len(outer_train)} valid={len(outer_valid)}", flush=True)
        assignment[outer_valid] = outer_number - 1
        soft = np.zeros((len(outer_train), len(ALL_CLASSES)), dtype=np.float32)
        seen = np.zeros(len(outer_train), dtype=bool)
        valid_groups = set(groups[outer_valid].tolist())
        teacher_records = []
        inner = list(GroupKFold(n_splits=args.inner_folds).split(
            outer_train, target[outer_train], groups[outer_train],
        ))
        for inner_number, (inner_train_local, inner_valid_local) in enumerate(inner, start=1):
            inner_train = outer_train[inner_train_local]
            inner_valid = outer_train[inner_valid_local]
            overlap = valid_groups.intersection(groups[inner_train].tolist())
            if overlap:
                raise AssertionError("Outer validation group entered teacher training")
            probability = fit_teacher(
                teacher_tensors, target, inner_train, inner_valid, args.teacher_epochs,
                args.batch_size, args.teacher_seed + outer_number * 100 + inner_number, device,
            )
            soft[inner_valid_local] = probability
            seen[inner_valid_local] = True
            teacher_records.append({
                "inner_fold": inner_number,
                "teacher_train_rows": int(len(inner_train)),
                "teacher_predict_rows": int(len(inner_valid)),
                "outer_validation_group_overlap": int(len(overlap)),
            })
            print(f"outer={outer_number} inner_teacher={inner_number}/{args.inner_folds} complete", flush=True)
        if not seen.all() or np.any(soft.sum(axis=1) <= 0.0):
            raise AssertionError("Missing inner cross-fit teacher targets")
        teacher_soft[outer_number - 1, outer_train] = soft
        if not (
            np.all(teacher_soft[outer_number - 1, outer_train].sum(axis=1) > 0.0)
            and np.all(teacher_soft[outer_number - 1, outer_valid] == 0.0)
        ):
            raise AssertionError("Per-outer teacher target storage invariant failed")
        for candidate in args.candidate:
            probability, epoch, score = fit_student(
                student_tensors, target, soft, outer_train, outer_valid,
                candidate.epochs, 0, candidate.distill, args.batch_size,
                candidate.seed + outer_number, device, fixed_epochs=candidate.epochs,
            )
            oof[candidate.name][outer_valid] = probability
            records[candidate.name].append({
                "outer_fold": outer_number,
                "student_fixed_epoch": int(epoch),
                "student_macro_f1": float(score),
            })
            print(
                f"outer={outer_number} candidate={candidate.name} "
                f"macro_f1={score:.6f} fixed_epoch={epoch}",
                flush=True,
            )
        outer_records.append({"outer_fold": outer_number, "teacher_inner_folds": teacher_records})

    np.save(out / "oof_targets.npy", target)
    np.save(out / "outer_fold_assignment.npy", assignment)
    np.save(out / "teacher_soft_by_outer.npy", teacher_soft)
    manifest = {
        "strictness": (
            "Each teacher is fit only inside its outer train partition. Students use fixed epochs "
            "and never select an epoch on outer-validation labels."
        ),
        "teacher_epochs_fixed": args.teacher_epochs,
        "outer_folds": args.outer_folds,
        "inner_folds": args.inner_folds,
        "teacher_seed": args.teacher_seed,
        "teacher_outer_records": outer_records,
        "base_oof_caveat": (
            "The inherited global 97-column base OOF cache is not fully outer-nested; "
            "these artifacts are teacher-strict, not fully base-strict."
        ),
        "seconds": round(time.time() - began, 2),
    }
    (out / "teacher_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    summaries = {}
    for candidate in args.candidate:
        directory = out / "candidates" / candidate.name
        directory.mkdir(parents=True, exist_ok=True)
        summary = {
            "name": candidate.name,
            "macro_f1": macro(target, oof[candidate.name]),
            "distill": candidate.distill,
            "student_fixed_epochs": candidate.epochs,
            "student_seed": candidate.seed,
            "folds": records[candidate.name],
        }
        np.save(directory / "oof_probabilities.npy", oof[candidate.name])
        (directory / "results.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
        )
        summaries[candidate.name] = summary
    (out / "grid_results.json").write_text(
        json.dumps({"teacher_manifest": manifest, "candidates": summaries}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"out_dir": str(out), "candidates": summaries}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
