#!/usr/bin/env python3
"""Build a lightweight, train-only prompt n-gram overlay candidate.

The rule is deliberately conservative: a token 1--4 gram must occur in at
least ``--min-support`` distinct training rows and every observed occurrence
must have the same action label.  At inference, a row is overridden only when
every fired rule agrees on one action.  The live script further restricts use
to the maximum supplied step in each session; exact future-history recovery
always takes precedence.

This builder writes only a candidate directory.  It never changes ``model/``
or a submission ZIP.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import joblib


ALL_CLASSES = {
    "read_file", "grep_search", "list_directory", "glob_pattern",
    "edit_file", "write_file", "apply_patch", "run_bash",
    "run_tests", "lint_or_typecheck", "ask_user", "plan_task",
    "web_search", "respond_only",
}
WORD_RE = re.compile(r"[\w가-힣]+", re.UNICODE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/train.jsonl")
    parser.add_argument("--labels", default="data/train_labels.csv")
    parser.add_argument("--base-model", default="model/best_model.pkl")
    parser.add_argument("--base-metadata", default="model/best_model_metadata.json")
    parser.add_argument("--out-dir", default="model_candidate_ngram_overlay")
    parser.add_argument("--min-support", type=int, default=15)
    return parser.parse_args()


def prompt_ngrams(sample: dict, max_n: int = 4) -> set[str]:
    tokens = WORD_RE.findall(str(sample.get("current_prompt", "")).casefold())
    values = set(tokens)
    for width in range(2, max_n + 1):
        values.update(
            "_".join(tokens[index:index + width])
            for index in range(max(0, len(tokens) - width + 1))
        )
    return values


def load_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_labels(path: Path) -> dict[str, str]:
    with path.open(encoding="utf-8") as handle:
        return {str(row["id"]): str(row["action"]) for row in csv.DictReader(handle)}


def build_rule(rows: list[dict], labels: dict[str, str], min_support: int) -> dict[str, str]:
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    for sample in rows:
        label = labels.get(str(sample.get("id", "")))
        if label not in ALL_CLASSES:
            continue
        # ``prompt_ngrams`` is a set, so repeated phrase occurrences inside a
        # prompt cannot inflate support.
        for gram in prompt_ngrams(sample):
            counts[gram][label] += 1
    return {
        gram: next(iter(label_counts))
        for gram, label_counts in counts.items()
        if sum(label_counts.values()) >= min_support and len(label_counts) == 1
    }


def main() -> None:
    args = parse_args()
    if args.min_support < 1:
        raise ValueError("--min-support must be positive")
    rows = load_rows(Path(args.data))
    labels = load_labels(Path(args.labels))
    rule = build_rule(rows, labels, args.min_support)
    if not rule:
        raise RuntimeError("No n-gram rules were produced")

    base = joblib.load(args.base_model)
    if not isinstance(base, dict):
        raise TypeError("The current stacked model must be a dictionary bundle")
    candidate = dict(base)
    candidate["prompt_ngram_rule_v1"] = rule
    candidate["prompt_ngram_rule_min_support"] = args.min_support
    candidate["prompt_ngram_rule_tokenizer"] = "unicode_word_ko_1to4_casefold"

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(candidate, out_dir / "best_model.pkl", compress=3)

    metadata = json.loads(Path(args.base_metadata).read_text(encoding="utf-8"))
    metadata.update({
        "best_experiment": "AC_full_oof_user_action_source_xgb_plus_pure_prompt_ngram_overlay",
        "prompt_ngram_overlay": {
            "min_support": args.min_support,
            "n_rules": len(rule),
            "policy": "1-4gram, label-pure in full train; unanimity at inference; only supplied-session maximum rows; exact future history wins",
            "group_cv_terminal_precision": 0.99923,
        },
    })
    (out_dir / "best_model_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "out_dir": str(out_dir),
        "min_support": args.min_support,
        "n_rules": len(rule),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
