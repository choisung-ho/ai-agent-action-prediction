#!/usr/bin/env python3
"""Strict combinations of the two surviving sparse prompt rule families.

The individual rule audits showed a small but independent gain for (A) a
session-supported word n-gram table and (B) high-support word-internal char
fragments.  This file tests only how to combine them.  Rule dictionaries are
rebuilt inside every whole-session fold; a confidence threshold, when tested,
is selected only on the other outer folds.  It never edits live assets.
"""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold

from experiment_terminal_sparse_rules import (
    CLASSES, LABEL_TO_ID, geometric, labels_from, rows_from, session_of,
    source_of, split_from, terminal_mask,
)
from experiment_terminal_prompt_grid import (
    build_counts, decide, make_rule, token_char_features, word_features,
)


def score(y: np.ndarray, pred: np.ndarray) -> float:
    return float(f1_score(y, pred, labels=np.arange(len(CLASSES)), average="macro", zero_division=0))


def rule_oof(rows: list[dict], y: np.ndarray, terminal: np.ndarray, features: list[tuple[str, ...]],
             support: int, session_support: bool, folds: list[tuple[np.ndarray, np.ndarray]], groups: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    output = np.full(len(rows), -1, dtype=np.int16)
    fold_id = np.full(len(rows), -1, dtype=np.int8)
    for number, (train, valid) in enumerate(folds):
        counts = build_counts(features, y, groups, train, session_support)
        rule = make_rule(counts, support)
        valid_terminal = valid[terminal[valid]]
        output[valid_terminal] = np.asarray([decide(features[int(index)], rule) for index in valid_terminal], dtype=np.int16)
        fold_id[valid_terminal] = number
    return output, fold_id


def rule_fixed(rows: list[dict], y: np.ndarray, terminal: np.ndarray, split: dict[str, str], features: list[tuple[str, ...]],
               support: int, session_support: bool, groups: np.ndarray) -> np.ndarray:
    train = np.asarray([index for index, row in enumerate(rows) if split[str(row["id"])] == "train"], dtype=int)
    valid = np.asarray([index for index, row in enumerate(rows) if split[str(row["id"])] == "val"], dtype=int)
    valid_terminal = valid[terminal[valid]]
    rule = make_rule(build_counts(features, y, groups, train, session_support), support)
    return np.asarray([decide(features[int(index)], rule) for index in valid_terminal], dtype=np.int16)


def combine(values: dict[str, np.ndarray], policy: str) -> np.ndarray:
    a, b = values["word12"], values["char20"]
    out = np.full(len(a), -1, dtype=np.int16)
    if policy == "word_primary":
        out = a.copy(); out[out < 0] = b[out < 0]
    elif policy == "char_primary":
        out = b.copy(); out[out < 0] = a[out < 0]
    elif policy == "unanimous_or_single":
        only_a = (a >= 0) & (b < 0)
        only_b = (b >= 0) & (a < 0)
        agree = (a >= 0) & (b >= 0) & (a == b)
        out[only_a] = a[only_a]; out[only_b] = b[only_b]; out[agree] = a[agree]
    elif policy == "both_agree":
        agree = (a >= 0) & (b >= 0) & (a == b)
        out[agree] = a[agree]
    else:
        raise ValueError(policy)
    return out


def evaluate(y: np.ndarray, terminal: np.ndarray, base: np.ndarray, candidate: np.ndarray) -> dict:
    mask = terminal & (candidate >= 0)
    pred = base.copy(); pred[mask] = candidate[mask]
    return {
        "n_candidate": int(mask.sum()),
        "coverage_terminal": float(mask.sum() / terminal.sum()),
        "precision": float((candidate[mask] == y[mask]).mean()) if mask.any() else 0.0,
        "fixed": int(((base != y) & (candidate == y) & mask).sum()),
        "harmed": int(((base == y) & (candidate != y) & mask).sum()),
        "terminal_macro_f1": score(y[terminal], pred[terminal]),
        "hybrid_macro_f1": score(y, np.where(terminal, pred, y)),
    }


def evaluate_fixed(y: np.ndarray, base: np.ndarray, candidate: np.ndarray) -> dict:
    mask = candidate >= 0
    pred = base.copy(); pred[mask] = candidate[mask]
    return {
        "n_candidate": int(mask.sum()),
        "coverage_terminal": float(mask.mean()),
        "precision": float((candidate[mask] == y[mask]).mean()) if mask.any() else 0.0,
        "fixed": int(((base != y) & (candidate == y) & mask).sum()),
        "harmed": int(((base == y) & (candidate != y) & mask).sum()),
        "terminal_macro_f1": score(y, pred),
    }


def thresholded(candidate: np.ndarray, probability: np.ndarray, threshold: float) -> np.ndarray:
    value = candidate.copy()
    # A rule equal to the base prediction is a harmless no-op; threshold only
    # controls actual replacements.
    value[(probability.max(axis=1) > threshold) & (candidate >= 0)] = -1
    return value


def nested_threshold(y: np.ndarray, terminal: np.ndarray, base: np.ndarray, candidate: np.ndarray,
                     probability: np.ndarray, fold: np.ndarray, thresholds: tuple[float, ...]) -> tuple[np.ndarray, list[dict]]:
    out = np.full(len(candidate), -1, dtype=np.int16)
    details: list[dict] = []
    for number in sorted(np.unique(fold[terminal])):
        held = terminal & (fold == number)
        train = terminal & (fold != number)
        scored: list[tuple[float, float]] = []
        for threshold in thresholds:
            value = thresholded(candidate, probability, threshold)
            pred = base.copy(); mask = train & (value >= 0); pred[mask] = value[mask]
            scored.append((score(y[train], pred[train]), threshold))
        _, chosen = max(scored, key=lambda pair: (pair[0], pair[1]))
        value = thresholded(candidate, probability, chosen)
        out[held] = value[held]
        details.append({"fold": int(number + 1), "threshold": float(chosen)})
    return out, details


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    rows = rows_from(str(root / "data/train.jsonl"))
    labels = labels_from(str(root / "data/train_labels.csv"))
    split = split_from(str(root / "splits/common_split.csv"))
    y = np.asarray([LABEL_TO_ID[labels[str(row["id"])]] for row in rows], dtype=np.int64)
    terminal = terminal_mask(rows)
    groups = np.asarray([session_of(row) for row in rows])
    source = np.asarray([source_of(row) for row in rows])
    folds = list(GroupKFold(n_splits=5).split(np.zeros(len(rows)), y, groups))
    xgb = np.load(root / "experiments/modeling_ak_live_xgb_proxy/oof_probabilities.npy").astype(np.float32)
    gru = np.load(root / "experiments/modeling_ai_sequence_fusion_noe5/oof_probabilities.npy").astype(np.float32)
    probability = geometric(xgb, gru, source)
    base = probability.argmax(axis=1)
    valid = np.asarray([index for index, row in enumerate(rows) if split[str(row["id"])] == "val"], dtype=int)
    fixed_xgb = np.load(root / "experiments/modeling_ac_tune2/oof_probabilities.npy").astype(np.float32)
    fixed_gru = np.load(root / "experiments/modeling_ai_sequence_fusion_fixed_noe5/valid_probabilities.npy").astype(np.float32)
    fixed_probability_all = geometric(fixed_xgb, fixed_gru, source[valid])
    fixed_terminal_mask = terminal[valid]
    fixed_probability = fixed_probability_all[fixed_terminal_mask]
    fixed_base = fixed_probability.argmax(axis=1)
    fixed_y = y[valid][fixed_terminal_mask]

    print("features=word", flush=True)
    word = [word_features(row) for row in rows]
    print("oof=word12", flush=True)
    word_oof, fold_id = rule_oof(rows, y, terminal, word, 12, True, folds, groups)
    word_fixed = rule_fixed(rows, y, terminal, split, word, 12, True, groups)
    del word
    print("features=char", flush=True)
    char = [token_char_features(row) for row in rows]
    print("oof=char20", flush=True)
    char_oof, char_fold = rule_oof(rows, y, terminal, char, 20, False, folds, groups)
    char_fixed = rule_fixed(rows, y, terminal, split, char, 20, False, groups)
    if not np.array_equal(fold_id[terminal], char_fold[terminal]):
        raise ValueError("Outer fold mismatch")
    del char

    values = {"word12": word_oof, "char20": char_oof}
    fixed_values = {"word12": word_fixed, "char20": char_fixed}
    policies = ("word_primary", "char_primary", "unanimous_or_single", "both_agree")
    output: dict[str, object] = {
        "scope": "Experiment only; no model/script/ZIP modified.",
        "protocol": "Each rule family is rebuilt in the training sessions of every outer fold. Fixed split is independent.",
        "baseline": {
            "full_terminal_macro_f1": score(y[terminal], base[terminal]),
            "fixed_terminal_macro_f1": score(fixed_y, fixed_base),
        },
        "individual": {
            "word12": {"full": evaluate(y, terminal, base, word_oof), "fixed": evaluate_fixed(fixed_y, fixed_base, word_fixed)},
            "char20": {"full": evaluate(y, terminal, base, char_oof), "fixed": evaluate_fixed(fixed_y, fixed_base, char_fixed)},
        },
        "combinations": {},
        "threshold_gates": {},
    }
    for policy in policies:
        candidate = combine(values, policy)
        fixed_candidate = combine(fixed_values, policy)
        output["combinations"][policy] = {
            "full": evaluate(y, terminal, base, candidate),
            "fixed": evaluate_fixed(fixed_y, fixed_base, fixed_candidate),
        }

    # Threshold is deliberately selected on the other four outer folds before
    # applying to the fifth.  This is a diagnostic only; fixed split decides
    # whether any confidence gate is portable.
    thresholds = (0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.98, 0.99, 0.995, 0.999, 1.0)
    for policy in ("word_primary", "char_primary", "unanimous_or_single"):
        candidate = combine(values, policy)
        fixed_candidate = combine(fixed_values, policy)
        nested, details = nested_threshold(y, terminal, base, candidate, probability, fold_id, thresholds)
        nested_fixed_values: dict[str, dict] = {"details": details, "full": evaluate(y, terminal, base, nested), "fixed": {}}
        for threshold in thresholds:
            nested_fixed_values["fixed"][str(threshold)] = evaluate_fixed(
                fixed_y, fixed_base, thresholded(fixed_candidate, fixed_probability, threshold)
            )
        output["threshold_gates"][policy] = nested_fixed_values

    out = root / "experiments/terminal_rule_combinations"; out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out / "oof_candidates.npz", y=y, terminal=terminal, base=base, probability=probability,
        word12=word_oof, char20=char_oof, fold=fold_id, valid_ids=np.asarray([rows[index]["id"] for index in valid if terminal[index]]),
        fixed_y=fixed_y, fixed_base=fixed_base, fixed_probability=fixed_probability,
        fixed_word12=word_fixed, fixed_char20=char_fixed,
    )
    (out / "report.json").write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
