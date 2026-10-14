#!/usr/bin/env python3
"""Attach the vetted char20 terminal rule to a portable submission bundle.

This builder intentionally replaces the older word n-gram fallback rather
than combining them: the strict rule-combination audit found char20 alone
more robust on both GroupKFold terminal rows and the independent fixed split.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", default="model/best_model.pkl")
    parser.add_argument("--base-metadata", default="model/best_model_metadata.json")
    parser.add_argument("--rule", default="experiments/terminal_rule_combinations/char20_rule.json")
    parser.add_argument("--rule-metadata", default="experiments/terminal_rule_combinations/char20_rule_metadata.json")
    parser.add_argument("--out-dir", default="model_candidate_char20")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base = joblib.load(args.base_model)
    if not isinstance(base, dict):
        raise TypeError("Submission model must be a dictionary bundle")
    rule = json.loads(Path(args.rule).read_text(encoding="utf-8"))
    if not isinstance(rule, dict) or not rule:
        raise ValueError("Char-rule artifact is empty or invalid")

    candidate = dict(base)
    candidate.pop("prompt_ngram_rule_v1", None)
    candidate["terminal_prompt_char_rule_v1"] = rule

    rule_metadata = json.loads(Path(args.rule_metadata).read_text(encoding="utf-8"))
    metadata = json.loads(Path(args.base_metadata).read_text(encoding="utf-8"))
    metadata.pop("prompt_ngram_overlay", None)
    metadata.update({
        "best_experiment": "AD_raw_xgb_plus_noe5_sequence_fusion_geometric_fallback_plus_char20_terminal_prompt_overlay",
        "validation_method": (
            "nested GroupKFold terminal/fallback calibration for raw 2032-feature XGBoost proxy plus "
            "no-E5 NumPy GRU; char20 terminal rule is rebuilt inside every whole-session fold and "
            "checked on the independent fixed split; exact future-history labels override fallback"
        ),
        "terminal_prompt_char_rule": rule_metadata,
    })

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(candidate, out_dir / "best_model.pkl", compress=3)
    (out_dir / "best_model_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "out_dir": str(out_dir),
        "n_rules": len(rule),
        "replaced_word_ngram": "prompt_ngram_rule_v1" not in candidate,
        "char_rule": "terminal_prompt_char_rule_v1" in candidate,
    }))


if __name__ == "__main__":
    main()
