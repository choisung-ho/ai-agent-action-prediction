#!/usr/bin/env python3
"""Nested audit for high-support sparse prompt rule variants.

This follows the same deployment gate as the live future-history recovery:
rules are scored only for supplied-session terminal rows.  Tables are rebuilt
in every outer GroupKFold fold, and then checked again on the unrelated common
fixed split.  It is an experiment only and never writes model/script/ZIP.
"""

from __future__ import annotations

import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold

from experiment_terminal_sparse_rules import (
    CLASSES,
    LABEL_TO_ID,
    WORD,
    geometric,
    labels_from,
    rows_from,
    session_of,
    source_of,
    split_from,
    terminal_mask,
)


def word_features(sample: dict) -> tuple[str, ...]:
    tokens = WORD.findall(str(sample.get("current_prompt", "")).casefold())[:70]
    return tuple({
        "w|" + "_".join(tokens[offset:offset + width])
        for width in range(1, 5)
        for offset in range(max(0, len(tokens) - width + 1))
    })


def source_word_features(sample: dict) -> tuple[str, ...]:
    # Bigrams+ only: source conditioning makes one-word evidence too brittle.
    prefix = "sw=" + source_of(sample) + "|"
    tokens = WORD.findall(str(sample.get("current_prompt", "")).casefold())[:70]
    return tuple({
        prefix + "_".join(tokens[offset:offset + width])
        for width in range(2, 5)
        for offset in range(max(0, len(tokens) - width + 1))
    })


def normalized_word_features(sample: dict) -> tuple[str, ...]:
    tokens = [re.sub(r"\d+", "<n>", token) for token in WORD.findall(str(sample.get("current_prompt", "")).casefold())[:70]]
    return tuple({
        "nw|" + "_".join(tokens[offset:offset + width])
        for width in range(1, 5)
        for offset in range(max(0, len(tokens) - width + 1))
    })


def token_char_features(sample: dict) -> tuple[str, ...]:
    """Short word-internal character fragments; useful for Korean endings.

    Whole-prompt char n-grams tend to capture arbitrary repository names.  The
    bounded word-internal version is intentionally less expressive and easier
    to vet with a high support threshold.
    """
    values: set[str] = set()
    for raw in WORD.findall(str(sample.get("current_prompt", "")).casefold())[:70]:
        token = "^" + re.sub(r"\d+", "<n>", raw) + "$"
        for width in (3, 4, 5):
            values.update("c" + str(width) + "|" + token[offset:offset + width]
                          for offset in range(max(0, len(token) - width + 1)))
    return tuple(values)


def build_counts(features: list[tuple[str, ...]], y: np.ndarray, groups: np.ndarray, train: np.ndarray,
                 session_support: bool) -> dict[str, Counter[int]]:
    counts: dict[str, Counter[int]] = defaultdict(Counter)
    if not session_support:
        for index in train:
            label = int(y[index])
            for key in features[int(index)]:
                counts[key][label] += 1
        return counts
    by_group: dict[str, list[int]] = defaultdict(list)
    for index in train:
        by_group[str(groups[int(index)])].append(int(index))
    for index_list in by_group.values():
        seen: dict[str, set[int]] = defaultdict(set)
        for index in index_list:
            label = int(y[index])
            for key in features[index]:
                seen[key].add(label)
        for key, labels in seen.items():
            for label in labels:
                counts[key][label] += 1
    return counts


def make_rule(counts: dict[str, Counter[int]], support: int) -> dict[str, int]:
    return {
        key: next(iter(label_counts))
        for key, label_counts in counts.items()
        if sum(label_counts.values()) >= support and len(label_counts) == 1
    }


def decide(feature_values: tuple[str, ...], rule: dict[str, int]) -> int:
    values = {rule[key] for key in feature_values if key in rule}
    return next(iter(values)) if len(values) == 1 else -1


def score(y: np.ndarray, pred: np.ndarray) -> float:
    return float(f1_score(y, pred, labels=np.arange(len(CLASSES)), average="macro", zero_division=0))


def assessment(y: np.ndarray, terminal: np.ndarray, base: np.ndarray, candidate: np.ndarray) -> dict:
    mask = terminal & (candidate >= 0)
    overlay = base.copy(); overlay[mask] = candidate[mask]
    return {
        "n_candidate": int(mask.sum()),
        "coverage_terminal": float(mask.sum() / max(1, terminal.sum())),
        "precision": float((candidate[mask] == y[mask]).mean()) if mask.any() else 0.0,
        "fixed": int(((base != y) & (candidate == y) & mask).sum()),
        "harmed": int(((base == y) & (candidate != y) & mask).sum()),
        "terminal_macro_f1": score(y[terminal], overlay[terminal]),
        "full_recovery_hybrid_macro_f1": score(y, np.where(terminal, overlay, y)),
    }


def group_grid(rows: list[dict], y: np.ndarray, terminal: np.ndarray, base: np.ndarray,
               features: list[tuple[str, ...]], supports: tuple[int, ...], session_support: bool) -> dict[str, dict]:
    groups = np.asarray([session_of(row) for row in rows])
    all_predictions = {support: np.full(len(rows), -1, dtype=np.int16) for support in supports}
    folds: dict[int, list[dict]] = {support: [] for support in supports}
    for fold, (train, valid) in enumerate(GroupKFold(n_splits=5).split(np.zeros(len(rows)), y, groups), start=1):
        counts = build_counts(features, y, groups, train, session_support)
        valid_terminal = valid[terminal[valid]]
        for support in supports:
            rule = make_rule(counts, support)
            predicted = np.asarray([decide(features[int(index)], rule) for index in valid_terminal], dtype=np.int16)
            all_predictions[support][valid_terminal] = predicted
            fired = predicted >= 0
            folds[support].append({
                "fold": fold, "rule_count": len(rule), "n_fired": int(fired.sum()),
                "precision": float((predicted[fired] == y[valid_terminal][fired]).mean()) if fired.any() else None,
            })
    results: dict[str, dict] = {}
    for support in supports:
        result = assessment(y, terminal, base, all_predictions[support])
        result["folds"] = folds[support]
        results[str(support)] = result
    return results


def fixed_grid(rows: list[dict], y: np.ndarray, terminal: np.ndarray, split: dict[str, str],
               base: np.ndarray, features: list[tuple[str, ...]], supports: tuple[int, ...], session_support: bool) -> dict[str, dict]:
    train = np.asarray([index for index, row in enumerate(rows) if split[str(row["id"])] == "train"], dtype=int)
    valid = np.asarray([index for index, row in enumerate(rows) if split[str(row["id"])] == "val"], dtype=int)
    groups = np.asarray([session_of(row) for row in rows])
    valid_terminal = valid[terminal[valid]]
    counts = build_counts(features, y, groups, train, session_support)
    y_terminal = y[valid_terminal]
    result: dict[str, dict] = {}
    for support in supports:
        rule = make_rule(counts, support)
        candidate = np.asarray([decide(features[int(index)], rule) for index in valid_terminal], dtype=np.int16)
        mask = candidate >= 0
        overlay = base.copy(); overlay[mask] = candidate[mask]
        result[str(support)] = {
            "rule_count": len(rule), "n_candidate": int(mask.sum()),
            "coverage_terminal": float(mask.sum() / max(1, len(valid_terminal))),
            "precision": float((candidate[mask] == y_terminal[mask]).mean()) if mask.any() else 0.0,
            "fixed": int(((base != y_terminal) & (candidate == y_terminal) & mask).sum()),
            "harmed": int(((base == y_terminal) & (candidate != y_terminal) & mask).sum()),
            "terminal_macro_f1": score(y_terminal, overlay),
        }
    return result


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    rows = rows_from(str(root / "data/train.jsonl"))
    labels = labels_from(str(root / "data/train_labels.csv"))
    split = split_from(str(root / "splits/common_split.csv"))
    y = np.asarray([LABEL_TO_ID[labels[str(row["id"])]] for row in rows], dtype=np.int64)
    terminal = terminal_mask(rows)
    source = np.asarray([source_of(row) for row in rows])
    xgb = np.load(root / "experiments/modeling_ak_live_xgb_proxy/oof_probabilities.npy").astype(np.float32)
    gru = np.load(root / "experiments/modeling_ai_sequence_fusion_noe5/oof_probabilities.npy").astype(np.float32)
    base = geometric(xgb, gru, source).argmax(axis=1)
    valid = np.asarray([index for index, row in enumerate(rows) if split[str(row["id"])] == "val"], dtype=int)
    fixed_xgb = np.load(root / "experiments/modeling_ac_tune2/oof_probabilities.npy").astype(np.float32)
    fixed_gru = np.load(root / "experiments/modeling_ai_sequence_fusion_fixed_noe5/valid_probabilities.npy").astype(np.float32)
    fixed_base_all = geometric(fixed_xgb, fixed_gru, source[valid]).argmax(axis=1)
    fixed_base = fixed_base_all[terminal[valid]]

    variants = {
        "word_row": (word_features, (15, 18, 20, 25, 30, 40, 50), False),
        "word_session": (word_features, (5, 6, 8, 10, 12, 15, 20), True),
        "source_word_row": (source_word_features, (5, 6, 8, 10, 12, 15, 20), False),
        "normalized_word_row": (normalized_word_features, (15, 20, 25, 30, 40), False),
        "token_char_row": (token_char_features, (20, 30, 40, 60, 80), False),
    }
    output = {
        "scope": "Experiment only. No model/script/ZIP was modified.",
        "protocol": "Each rule dictionary is rebuilt only from outer-train whole sessions; common fixed split is a separate confirmation.",
        "baseline": {
            "full_terminal_macro_f1": score(y[terminal], base[terminal]),
            "fixed_terminal_macro_f1": score(y[valid][terminal[valid]], fixed_base),
        },
        "variants": {},
    }
    for name, (function, supports, use_session_support) in variants.items():
        print(f"building_features={name}", flush=True)
        features = [function(row) for row in rows]
        print(f"nested_grid={name}", flush=True)
        output["variants"][name] = {
            "support_unit": "session" if use_session_support else "row",
            "full_group_cv": group_grid(rows, y, terminal, base, features, supports, use_session_support),
            "fixed_split": fixed_grid(rows, y, terminal, split, fixed_base, features, supports, use_session_support),
        }
        del features
    out = root / "experiments/terminal_prompt_rule_grid"; out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
