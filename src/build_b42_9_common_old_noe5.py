#!/usr/bin/env python3
"""Build a fully common-train-only no-E5 GRU audit artifact for B42_9.

This is deliberately *not* a submission builder.  It recreates the compact
portable no-E5 sequence model used by the B42_9 family, but it uses only the
strict common-train 97-feature blocks for its five group-OOF training runs.
The final refit likewise sees no common-validation rows.  The resulting
probabilities let downstream decision-calibration tests evaluate a real,
seed-42, common-split holdout without leaking validation labels through either
the selector features or the GRU fit.
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
    parser.add_argument("--blocks", default="experiments/au_source_strict_common/strict_blocks.npz")
    parser.add_argument("--out-dir", default="experiments/b42_9_common_old_noe5_seed42")
    parser.add_argument("--epochs", type=int, default=16,
                        help="Fixed, predeclared epochs.  No validation-label early stopping is permitted.")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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
        y[mask], probability[mask].argmax(axis=1), labels=np.arange(len(ALL_CLASSES)),
        average="macro", zero_division=0,
    ))


def make_tensors(samples: list[dict], base: np.ndarray) -> list[torch.Tensor]:
    action, status, argument, meta = sequence_tensors(samples)
    # The B42_9 portable student replaces the E5 branch with its learned
    # constant-zero input; retain that exact shape/contract here.
    e5 = np.zeros((len(samples), 1), dtype=np.float32)
    return [torch.from_numpy(value) for value in (action, status, argument, base, e5, meta)]


def fit_probability(
    tensors: list[torch.Tensor], y: np.ndarray, train: np.ndarray, predict_index: np.ndarray,
    epochs: int, batch_size: int, seed: int, device: torch.device,
) -> np.ndarray:
    """Fixed-epoch B42_9 architecture fit with no target from ``predict_index``."""
    seed_everything(seed)
    model = SequenceFusion(tensors[3].shape[1], tensors[4].shape[1], tensors[5].shape[1]).to(device)
    counts = np.bincount(y[train], minlength=len(ALL_CLASSES))
    if np.any(counts == 0):
        raise ValueError(f"Training partition lacks classes {np.flatnonzero(counts == 0).tolist()}")
    class_weight = torch.tensor(
        np.sqrt(len(train) / (len(ALL_CLASSES) * counts)), dtype=torch.float32, device=device,
    )
    train_data = TensorDataset(*(value[train] for value in tensors), torch.from_numpy(y[train]))
    train_loader = DataLoader(
        train_data, batch_size=batch_size, shuffle=True,
        generator=torch.Generator().manual_seed(seed), pin_memory=True,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.8e-3, weight_decay=1.0e-4)
    for _ in range(epochs):
        model.train()
        for batch in train_loader:
            *features, target = [value.to(device, non_blocking=True) for value in batch]
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model(*features)
                loss = nn.functional.cross_entropy(logits, target, weight=class_weight, label_smoothing=0.015)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
    loader = DataLoader(
        TensorDataset(*(value[predict_index] for value in tensors)), batch_size=1024,
        shuffle=False, pin_memory=True,
    )
    probability = predict(model, loader, device).astype(np.float32)
    del model
    torch.cuda.empty_cache()
    return probability


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this audit artifact")
    if args.epochs < 1:
        raise ValueError("--epochs must be positive")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    root = Path(__file__).resolve().parents[1]
    blocks = np.load(root / args.blocks, allow_pickle=False)
    required = {"ids_train", "ids_valid", "target_train", "target_valid", "base_train", "base_valid"}
    if missing := required - set(blocks.files):
        raise ValueError(f"Strict blocks are missing {sorted(missing)}")
    ids_train = blocks["ids_train"].astype(str)
    ids_valid = blocks["ids_valid"].astype(str)
    y_train = blocks["target_train"].astype(np.int64)
    y_valid = blocks["target_valid"].astype(np.int64)
    base_train = blocks["base_train"].astype(np.float32)
    base_valid = blocks["base_valid"].astype(np.float32)
    if base_train.shape != (len(ids_train), 97) or base_valid.shape != (len(ids_valid), 97):
        raise ValueError("Expected strict 97-column B42_9 base blocks")

    samples = [json.loads(line) for line in (root / "data/train.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    label = {row["id"]: row["action"] for row in csv.DictReader((root / "data/train_labels.csv").open(encoding="utf-8"))}
    partition = {row["id"]: row["split"] for row in csv.DictReader((root / "splits/common_split.csv").open(encoding="utf-8"))}
    train_samples = [row for row in samples if partition[str(row["id"])] == "train"]
    valid_samples = [row for row in samples if partition[str(row["id"])] == "val"]
    if not (np.array_equal(np.asarray([str(row["id"]) for row in train_samples]), ids_train)
            and np.array_equal(np.asarray([str(row["id"]) for row in valid_samples]), ids_valid)):
        raise ValueError("Strict blocks do not align with common-split data order")
    to_id = {name: index for index, name in enumerate(ALL_CLASSES)}
    expected_train = np.asarray([to_id[label[sample_id]] for sample_id in ids_train], dtype=np.int64)
    expected_valid = np.asarray([to_id[label[sample_id]] for sample_id in ids_valid], dtype=np.int64)
    if not (np.array_equal(expected_train, y_train) and np.array_equal(expected_valid, y_valid)):
        raise ValueError("Strict block targets do not align")

    out = root / args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    train_tensors = make_tensors(train_samples, base_train)
    valid_tensors = make_tensors(valid_samples, base_valid)
    groups = np.asarray([sample_id.rsplit("-step_", 1)[0] for sample_id in ids_train])
    terminal_train = terminal_mask(ids_train)
    terminal_valid = terminal_mask(ids_valid)
    device = torch.device("cuda")

    oof = np.zeros((len(ids_train), len(ALL_CLASSES)), dtype=np.float32)
    folds = list(GroupKFold(n_splits=5).split(np.zeros(len(y_train)), y_train, groups))
    fold_assignment = np.empty(len(y_train), dtype=np.int8)
    records: list[dict[str, object]] = []
    for fold_number, (train, held) in enumerate(folds, start=1):
        fold_assignment[held] = fold_number - 1
        started = time.time()
        probability = fit_probability(
            train_tensors, y_train, train, held, args.epochs, args.batch_size,
            args.seed + fold_number, device,
        )
        oof[held] = probability
        record = {
            "fold": fold_number, "train_rows": int(len(train)), "held_rows": int(len(held)),
            "all_macro_f1": macro(y_train[held], probability),
            "terminal_macro_f1": macro(y_train[held], probability, terminal_train[held]),
            "seconds": round(time.time() - started, 2),
        }
        records.append(record)
        print(json.dumps(record, ensure_ascii=False), flush=True)

    started = time.time()
    full_train = np.arange(len(y_train), dtype=np.int64)
    # The final common-train-only refit predicts the separate strict fixed
    # tensor block.  It intentionally has no validation-data training path.
    seed_everything(args.seed + 100)
    model = SequenceFusion(train_tensors[3].shape[1], train_tensors[4].shape[1], train_tensors[5].shape[1]).to(device)
    counts = np.bincount(y_train, minlength=len(ALL_CLASSES))
    class_weight = torch.tensor(np.sqrt(len(y_train) / (len(ALL_CLASSES) * counts)), dtype=torch.float32, device=device)
    loader = DataLoader(
        TensorDataset(*(value[full_train] for value in train_tensors), torch.from_numpy(y_train)),
        batch_size=args.batch_size, shuffle=True, generator=torch.Generator().manual_seed(args.seed + 100), pin_memory=True,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.8e-3, weight_decay=1.0e-4)
    for _ in range(args.epochs):
        model.train()
        for batch in loader:
            *features, target = [value.to(device, non_blocking=True) for value in batch]
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                loss = nn.functional.cross_entropy(model(*features), target, weight=class_weight, label_smoothing=0.015)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
    prediction_loader = DataLoader(
        TensorDataset(*valid_tensors), batch_size=1024, shuffle=False, pin_memory=True,
    )
    fixed_probability = predict(model, prediction_loader, device).astype(np.float32)
    del model
    torch.cuda.empty_cache()
    fixed_seconds = round(time.time() - started, 2)

    np.save(out / "common_train_oof_probabilities.npy", oof)
    np.save(out / "common_train_ids.npy", ids_train)
    np.save(out / "common_train_fold.npy", fold_assignment)
    np.save(out / "fixed_valid_probabilities.npy", fixed_probability)
    np.save(out / "fixed_valid_ids.npy", ids_valid)
    report = {
        "scope": "Experiment-only.  No live model, script, or ZIP was opened for writing.",
        "contract": (
            "B42_9 no-E5 architecture; strict common-train 97-feature blocks; five session GroupKFold "
            "OOF models and one common-train-only refit.  Every fit uses predeclared 16 epochs and never reads "
            "common-validation labels."
        ),
        "seed": args.seed, "epochs": args.epochs,
        "counts": {"common_train": int(len(y_train)), "common_terminal": int(terminal_train.sum()),
                   "fixed_valid": int(len(y_valid)), "fixed_terminal": int(terminal_valid.sum())},
        "oof": {"all_macro_f1": macro(y_train, oof), "terminal_macro_f1": macro(y_train, oof, terminal_train), "folds": records},
        "fixed": {"all_macro_f1": macro(y_valid, fixed_probability),
                  "terminal_macro_f1": macro(y_valid, fixed_probability, terminal_valid), "seconds": fixed_seconds},
    }
    (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
