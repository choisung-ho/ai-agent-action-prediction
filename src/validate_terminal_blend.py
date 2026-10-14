#!/usr/bin/env python3
"""Outer-fold validation for a terminal-refiner blend.

This script deliberately uses the already strict common-train -> common-val
terminal refiner predictions.  It tests whether a global blend weight can be
chosen on other validation sessions and generalize to a held-out set of
sessions.  It prevents a one-off fixed-validation blend gain from being
mistaken for a promotion signal.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold


CLASSES = [
    "read_file", "grep_search", "list_directory", "glob_pattern",
    "edit_file", "write_file", "apply_patch", "run_bash",
    "run_tests", "lint_or_typecheck", "ask_user", "plan_task",
    "web_search", "respond_only",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/train.jsonl")
    parser.add_argument("--labels", default="data/train_labels.csv")
    parser.add_argument("--split", default="splits/common_split.csv")
    parser.add_argument("--predictions", default="experiments/terminal_fallback_nostitch_d8/fixed_val_terminal_predictions.npz")
    parser.add_argument("--out", default="experiments/terminal_fallback_nostitch_d8/nested_blend_summary.json")
    parser.add_argument("--folds", type=int, default=5)
    return parser.parse_args()


def terminal_ids(data: Path, split_path: Path) -> tuple[list[str], list[str]]:
    split = {row["id"]: row["split"] for row in csv.DictReader(split_path.open(encoding="utf-8"))}
    rows = [json.loads(line) for line in data.open(encoding="utf-8") if line.strip()]
    validation_ids = [str(row["id"]) for row in rows if split[str(row["id"])] == "val"]
    grouped: dict[str, list[tuple[int, str]]] = defaultdict(list)
    for sample_id in validation_ids:
        session, _, value = sample_id.rpartition("-step_")
        grouped[session].append((int(value), sample_id))
    terminals = [max(values)[1] for values in grouped.values()]
    return validation_ids, terminals


def macro(target: np.ndarray, probability: np.ndarray) -> float:
    return float(f1_score(target, probability.argmax(axis=1), labels=np.arange(len(CLASSES)), average="macro", zero_division=0))


def main() -> None:
    args = parse_args()
    labels = {row["id"]: row["action"] for row in csv.DictReader(Path(args.labels).open(encoding="utf-8"))}
    validation_ids, expected_ids = terminal_ids(Path(args.data), Path(args.split))
    saved = np.load(args.predictions)
    ids = [str(value) for value in saved["ids"]]
    if set(ids) != set(expected_ids) or len(ids) != len(expected_ids):
        raise ValueError("Terminal prediction IDs do not match the fixed validation terminal set")
    target = np.asarray([CLASSES.index(labels[sample_id]) for sample_id in ids], dtype=np.int64)
    direct = saved["terminal_xgb"]
    fallback = saved["current_fallback"]
    if fallback.shape != direct.shape:
        raise ValueError("Expected calibrated fallback probabilities alongside terminal refiner probabilities")

    alpha_grid = np.asarray((0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0))
    groups = np.asarray([sample_id.rsplit("-step_", 1)[0] for sample_id in ids])
    oof = np.zeros_like(fallback)
    records: list[dict[str, float | int]] = []
    for fold, (train_index, test_index) in enumerate(GroupKFold(args.folds).split(direct, target, groups), start=1):
        train_scores = [macro(target[train_index], (1.0 - alpha) * fallback[train_index] + alpha * direct[train_index]) for alpha in alpha_grid]
        best_index = int(np.argmax(train_scores))
        alpha = float(alpha_grid[best_index])
        oof[test_index] = (1.0 - alpha) * fallback[test_index] + alpha * direct[test_index]
        records.append({
            "fold": fold,
            "selected_direct_weight": alpha,
            "selection_terminal_macro_f1": float(train_scores[best_index]),
            "heldout_baseline_terminal_macro_f1": macro(target[test_index], fallback[test_index]),
            "heldout_blend_terminal_macro_f1": macro(target[test_index], oof[test_index]),
        })

    # Full hybrid treats every non-terminal fixed-validation row as correctly
    # recovered by raw future history.  This mirrors the live inference path.
    val_target = np.asarray([CLASSES.index(labels[sample_id]) for sample_id in validation_ids], dtype=np.int64)
    terminal_position = {sample_id: position for position, sample_id in enumerate(ids)}
    full_baseline = val_target.copy()
    full_oof = val_target.copy()
    for index, sample_id in enumerate(validation_ids):
        position = terminal_position.get(sample_id)
        if position is not None:
            full_baseline[index] = int(fallback[position].argmax())
            full_oof[index] = int(oof[position].argmax())
    result = {
        "method": "5-fold outer session split; blend alpha selected only on other terminal sessions",
        "terminal_rows": len(ids),
        "baseline_terminal_macro_f1": macro(target, fallback),
        "nested_blend_terminal_macro_f1": macro(target, oof),
        "baseline_full_recovery_hybrid_macro_f1": float(f1_score(val_target, full_baseline, labels=np.arange(len(CLASSES)), average="macro", zero_division=0)),
        "nested_blend_full_recovery_hybrid_macro_f1": float(f1_score(val_target, full_oof, labels=np.arange(len(CLASSES)), average="macro", zero_division=0)),
        "folds": records,
    }
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
