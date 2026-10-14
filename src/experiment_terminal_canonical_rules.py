#!/usr/bin/env python3
"""Audit canonical prompt-template rules on terminal fallback rows only.

The public inference batch can deterministically recover non-terminal actions
from a later row's raw history.  This experiment therefore measures a
conservative rule overlay only on the maximum supplied step in each session.
Unlike an ID lookup, every candidate feature is computed from the current
prompt that is already visible to the inference program.  Rule dictionaries
are rebuilt inside whole-session folds and again from the frozen common-train
partition.  Nothing under ``model/`` or any ZIP archive is modified.

The purpose is to test whether normalization of templated file names, paths,
numbers, and quoted literals creates high-support label-pure phrases that are
not covered by the existing word-internal char20 rule.
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


ROOT = Path(__file__).resolve().parents[1]
CLASSES = (
    "read_file", "grep_search", "list_directory", "glob_pattern",
    "edit_file", "write_file", "apply_patch", "run_bash", "run_tests",
    "lint_or_typecheck", "ask_user", "plan_task", "web_search", "respond_only",
)
TO_ID = {label: index for index, label in enumerate(CLASSES)}
WORD = re.compile(r"(?u)\b\w+\b")
PATHISH = re.compile(r"(?:[A-Za-z]:)?[\\/]|\.(?:py|ts|tsx|js|jsx|json|ya?ml|md|txt|toml|cfg|ini|go|rs|java|css|html|sql|sh|bat|dockerfile)\b", re.I)
URLISH = re.compile(r"(?:https?://|www\.)", re.I)
QUOTED = re.compile(r"(?:`[^`]{1,160}`|'[^']{1,160}'|\"[^\"]{1,160}\")")


def session_of(sample: dict) -> str:
    return str(sample["id"]).split("-step_", 1)[0]


def step_of(sample: dict) -> int:
    return int(str(sample["id"]).rsplit("-step_", 1)[1])


def terminal_mask(rows: list[dict]) -> np.ndarray:
    maximum: dict[str, int] = {}
    for row in rows:
        group = session_of(row)
        maximum[group] = max(maximum.get(group, -1), step_of(row))
    return np.asarray([step_of(row) == maximum[session_of(row)] for row in rows], dtype=bool)


def source_of(row: dict) -> str:
    return "au" if str(row["id"]).startswith("sess_au_") else "sim"


def normalize_prompt(value: object, mode: str) -> list[str]:
    """Normalize ephemeral literals while retaining user intent wording."""
    text = str(value or "").casefold().replace("\x00", " ")
    # Quoted text is commonly a filename/query/config key.  Keeping an exact
    # random literal fragments a template, while a single marker preserves
    # the request's syntactic position.
    text = QUOTED.sub(" <quoted> ", text)
    if mode == "aggressive":
        text = URLISH.sub(" <url> ", text)
        text = re.sub(r"(?:[A-Za-z]:)?(?:[\\/][^\s`'\"]+)+", " <path> ", text)
        text = re.sub(r"\b[\w.-]+\.(?:py|ts|tsx|js|jsx|json|ya?ml|md|txt|toml|cfg|ini|go|rs|java|css|html|sql|sh|bat)\b", " <file> ", text, flags=re.I)
    tokens: list[str] = []
    for token in WORD.findall(text)[:90]:
        token = re.sub(r"\d+", "<n>", token)
        if mode == "shape":
            if PATHISH.search(token):
                token = "<path>"
            elif len(token) >= 18 and any(char.isalpha() for char in token):
                token = "<identifier>"
            elif "_" in token or "-" in token:
                token = "<identifier>"
        tokens.append(token)
    return tokens


def features_for(row: dict, family: str) -> tuple[str, ...]:
    if family.startswith("norm"):
        tokens = normalize_prompt(row.get("current_prompt"), "normal")
    elif family.startswith("aggressive"):
        tokens = normalize_prompt(row.get("current_prompt"), "aggressive")
    elif family.startswith("shape"):
        tokens = normalize_prompt(row.get("current_prompt"), "shape")
    else:
        raise ValueError(family)
    values: set[str] = set()
    # Long contiguous phrases describe the request template rather than a
    # single generic verb.  Width <=4 was already audited by the existing
    # prompt-rule grid, so keep this experiment intentionally narrow.
    prefix = family + "|"
    for width in (5, 6, 8):
        if len(tokens) < width:
            continue
        for offset in range(len(tokens) - width + 1):
            values.add(prefix + "_".join(tokens[offset:offset + width]))
    if 3 <= len(tokens) <= 34:
        values.add(prefix + "FULL|" + "_".join(tokens))
    # Prefix and suffix sketches are useful for natural-language templates
    # whose middle contains a random identifier.
    for width in (5, 6, 8):
        if len(tokens) >= width:
            values.add(prefix + "FIRST|" + "_".join(tokens[:width]))
            values.add(prefix + "LAST|" + "_".join(tokens[-width:]))
    return tuple(values)


def build_rule(features: list[tuple[str, ...]], target: np.ndarray, train: np.ndarray,
               support: int) -> dict[str, int]:
    counts: dict[str, Counter[int]] = defaultdict(Counter)
    for index in train:
        for value in features[int(index)]:
            counts[value][int(target[index])] += 1
    return {
        value: next(iter(labels))
        for value, labels in counts.items()
        if sum(labels.values()) >= support and len(labels) == 1
    }


def decide(values: tuple[str, ...], rule: dict[str, int]) -> int:
    labels = {rule[value] for value in values if value in rule}
    return next(iter(labels)) if len(labels) == 1 else -1


def geometric(left: np.ndarray, right: np.ndarray, source: np.ndarray) -> np.ndarray:
    weight = np.where(source == "au", 0.85, 0.25).astype(np.float32)
    value = (weight[:, None] * np.log(np.maximum(left, 1.0e-12))
             + (1.0 - weight[:, None]) * np.log(np.maximum(right, 1.0e-12)))
    value -= value.max(axis=1, keepdims=True)
    probability = np.exp(value)
    return (probability / probability.sum(axis=1, keepdims=True)).astype(np.float32)


def score(target: np.ndarray, pred: np.ndarray) -> float:
    return float(f1_score(target, pred, labels=np.arange(len(CLASSES)), average="macro", zero_division=0))


def overlay(base: np.ndarray, char: np.ndarray, candidate: np.ndarray, priority: str) -> np.ndarray:
    result = base.copy()
    if priority == "candidate_then_char":
        result[candidate >= 0] = candidate[candidate >= 0]
        result[char >= 0] = char[char >= 0]
    elif priority == "char_then_candidate":
        result[char >= 0] = char[char >= 0]
        result[candidate >= 0] = candidate[candidate >= 0]
    elif priority == "only_when_char_silent":
        result[char >= 0] = char[char >= 0]
        mask = (char < 0) & (candidate >= 0)
        result[mask] = candidate[mask]
    elif priority == "agree_or_single":
        result[char >= 0] = char[char >= 0]
        mask = (candidate >= 0) & ((char < 0) | (candidate == char))
        result[mask] = candidate[mask]
    else:
        raise ValueError(priority)
    return result


def evaluate(target: np.ndarray, base: np.ndarray, char: np.ndarray, candidate: np.ndarray,
             priority: str) -> dict[str, float | int]:
    pred = overlay(base, char, candidate, priority)
    candidate_mask = candidate >= 0
    changed = pred != overlay(base, char, np.full_like(candidate, -1), priority)
    return {
        "macro_f1": score(target, pred),
        "n_candidate": int(candidate_mask.sum()),
        "candidate_precision": float((candidate[candidate_mask] == target[candidate_mask]).mean()) if candidate_mask.any() else 0.0,
        "n_changed_after_char": int(changed.sum()),
        "fixed_after_char": int(((pred == target) & (overlay(base, char, np.full_like(candidate, -1), priority) != target)).sum()),
        "harmed_after_char": int(((pred != target) & (overlay(base, char, np.full_like(candidate, -1), priority) == target)).sum()),
    }


def main() -> None:
    rows = [json.loads(line) for line in (ROOT / "data/train.jsonl").open(encoding="utf-8") if line.strip()]
    labels = {row["id"]: row["action"] for row in csv.DictReader((ROOT / "data/train_labels.csv").open(encoding="utf-8"))}
    split = {row["id"]: row["split"] for row in csv.DictReader((ROOT / "splits/common_split.csv").open(encoding="utf-8"))}
    target = np.asarray([TO_ID[labels[str(row["id"])]] for row in rows], dtype=np.int16)
    terminal = terminal_mask(rows)
    groups = np.asarray([session_of(row) for row in rows])
    folds = list(GroupKFold(n_splits=5).split(np.zeros(len(rows)), target, groups))
    source = np.asarray([source_of(row) for row in rows])
    valid = np.asarray([index for index, row in enumerate(rows) if split[str(row["id"])] == "val"], dtype=int)

    full_xgb = np.load(ROOT / "experiments/xgb_robust_ensemble/full_oof_raw_seed42.npy").astype(np.float32)
    full_gru = np.load(ROOT / "experiments/modeling_ai_sequence_nested_distill_e16_i4/oof_probabilities.npy").astype(np.float32)
    full_probability = geometric(full_xgb, full_gru, source)
    full_base = full_probability.argmax(axis=1).astype(np.int16)
    fixed_xgb = np.load(ROOT / "experiments/xgb_robust_ensemble/strict_fixed_raw_seed42.npy").astype(np.float32)
    fixed_gru = np.load(ROOT / "experiments/modeling_ai_sequence_fusion_fixed_strictdistill_w35/valid_probabilities.npy").astype(np.float32)
    fixed_probability = geometric(fixed_xgb, fixed_gru, source[valid])
    fixed_terminal = terminal[valid]

    # These decisions are independent of the fallback probabilities.  They
    # were rebuilt session-safely in the existing char20 audit.
    char_artifact = np.load(ROOT / "experiments/terminal_rule_combinations/oof_candidates.npz")
    if not np.array_equal(char_artifact["y"], target) or not np.array_equal(char_artifact["terminal"], terminal):
        raise ValueError("Existing char20 OOF rows do not align")
    char_full = char_artifact["char20"].astype(np.int16)
    fixed_term_ids = np.asarray([str(rows[index]["id"]) for index in valid[fixed_terminal]])
    if not np.array_equal(char_artifact["valid_ids"], fixed_term_ids):
        raise ValueError("Existing char20 fixed IDs do not align")
    char_fixed = char_artifact["fixed_char20"].astype(np.int16)

    term = np.flatnonzero(terminal)
    full_base_term = full_base[term]
    full_char_term = char_full[term]
    fixed_target = target[valid][fixed_terminal]
    fixed_base_term = fixed_probability.argmax(axis=1).astype(np.int16)[fixed_terminal]
    report: dict[str, object] = {
        "scope": "Experiment only. No model/script/ZIP is changed.",
        "protocol": {
            "full": "5-fold whole-session GroupKFold; every normalized phrase rule is built from the other four folds.",
            "fixed": "Rules are fit only from common-train rows and evaluated on disjoint common-validation terminal sessions.",
            "fallback": "matching live-like raw seed42 XGB + strict nested-distilled GRU, geometric sim=.25/au=.85; char20 stays an independent final rule.",
            "features": "only current_prompt normalization; no IDs, target history, labels at inference, or future rows are used.",
        },
        "counts": {"rows": len(rows), "terminal": int(terminal.sum()), "fixed_terminal": int(fixed_terminal.sum())},
        "baseline": {
            "full_terminal_char20": score(target[term], overlay(full_base_term, full_char_term, np.full(len(term), -1, dtype=np.int16), "candidate_then_char")),
            "fixed_terminal_char20": score(fixed_target, overlay(fixed_base_term, char_fixed, np.full(len(fixed_target), -1, dtype=np.int16), "candidate_then_char")),
        },
        "families": {},
    }

    # Unlike char20, these are *terminal-distribution* template rules: each
    # session contributes at most one row, so supports count independent
    # sessions and the audit remains tractable without broad weak phrases.
    for family in ("norm10", "aggressive10", "shape10"):
        print(f"family={family} features", flush=True)
        features = [features_for(row, family) if terminal[index] else ()
                    for index, row in enumerate(rows)]
        family_result: dict[str, object] = {}
        # Build all outer-fold counts once; only the label-pure support cutoff
        # varies.  Rebuilding a dict for every cutoff is needlessly costly.
        fold_counts: list[tuple[np.ndarray, dict[str, Counter[int]]]] = []
        for fold, (train, holdout) in enumerate(folds, 1):
            train_terminal = train[terminal[train]]
            counts: dict[str, Counter[int]] = defaultdict(Counter)
            for index in train_terminal:
                for value in features[int(index)]:
                    counts[value][int(target[index])] += 1
            fold_counts.append((holdout, counts))
            print(f"family={family} count_fold={fold} terminal_train={len(train_terminal)} keys={len(counts)}", flush=True)
        fixed_train = np.asarray([index for index, row in enumerate(rows)
                                  if split[str(row["id"])] == "train" and terminal[index]], dtype=int)
        fixed_counts: dict[str, Counter[int]] = defaultdict(Counter)
        for index in fixed_train:
            for value in features[int(index)]:
                fixed_counts[value][int(target[index])] += 1
        for support in (3, 4, 5, 6, 8, 10, 12):
            candidate = np.full(len(rows), -1, dtype=np.int16)
            for fold, (holdout, counts) in enumerate(fold_counts, 1):
                rule = {
                    value: next(iter(labels))
                    for value, labels in counts.items()
                    if sum(labels.values()) >= support and len(labels) == 1
                }
                hold_term = holdout[terminal[holdout]]
                candidate[hold_term] = np.asarray([decide(features[int(index)], rule) for index in hold_term], dtype=np.int16)
                print(f"family={family} support={support} fold={fold} rules={len(rule)}", flush=True)
            fixed_rule = {
                value: next(iter(labels))
                for value, labels in fixed_counts.items()
                if sum(labels.values()) >= support and len(labels) == 1
            }
            fixed_candidate = np.asarray([decide(features[int(index)], fixed_rule) for index in valid[fixed_terminal]], dtype=np.int16)
            item: dict[str, object] = {"fixed_rule_count": len(fixed_rule), "priorities": {}}
            for priority in ("candidate_then_char", "only_when_char_silent", "agree_or_single"):
                item["priorities"][priority] = {
                    "full": evaluate(target[term], full_base_term, full_char_term, candidate[term], priority),
                    "fixed": evaluate(fixed_target, fixed_base_term, char_fixed, fixed_candidate, priority),
                }
            family_result[str(support)] = item
        report["families"][family] = family_result
    out = ROOT / "experiments/terminal_canonical_rules"
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
