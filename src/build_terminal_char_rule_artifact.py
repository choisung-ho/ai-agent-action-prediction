#!/usr/bin/env python3
"""Build the experiment-only full-data char-rule artifact for integration review."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from experiment_terminal_sparse_rules import CLASSES, LABEL_TO_ID, labels_from, rows_from, session_of
from experiment_terminal_prompt_grid import build_counts, decide, make_rule, token_char_features


def batch_terminal(rows: list[dict]) -> np.ndarray:
    maximum: dict[str, int] = {}
    for row in rows:
        sample_id = str(row.get("id", ""))
        session, _, value = sample_id.rpartition("-step_")
        try:
            maximum[session] = max(maximum.get(session, -1), int(value))
        except ValueError:
            continue
    values = []
    for row in rows:
        session, _, value = str(row.get("id", "")).rpartition("-step_")
        try:
            values.append(int(value) == maximum.get(session, -2))
        except ValueError:
            values.append(False)
    return np.asarray(values, dtype=bool)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    train = rows_from(str(root / "data/train.jsonl"))
    labels = labels_from(str(root / "data/train_labels.csv"))
    y = np.asarray([LABEL_TO_ID[labels[str(row["id"])]] for row in train], dtype=np.int64)
    groups = np.asarray([session_of(row) for row in train])
    features = [token_char_features(row) for row in train]
    rule_ids = make_rule(build_counts(features, y, groups, np.arange(len(train)), False), 20)
    rule = {key: CLASSES[int(value)] for key, value in rule_ids.items()}
    test = rows_from(str(root / "data/test.jsonl"))
    test_terminal = batch_terminal(test)
    test_fired = [decide(token_char_features(row), rule_ids) for row in test]
    metadata = {
        "version": "terminal_prompt_char_rule_v1",
        "training_scope": "all 70,000 train rows; one count per feature per row",
        "rule_policy": "casefold Unicode word tokens; replace digit runs with <n>; pad each token with ^/$; emit internal char 3/4/5-grams; retain support>=20 and exactly one observed label; inference requires every fired retained rule to agree",
        "support": 20,
        "n_rules": len(rule),
        "serialized_json_bytes": len(json.dumps(rule, ensure_ascii=False, separators=(",", ":")).encode("utf-8")),
        "batch_test": {
            "rows": len(test),
            "session_max_rows": int(test_terminal.sum()),
            "rule_fired_all": int(sum(value >= 0 for value in test_fired)),
            "rule_fired_session_max": int(sum(value >= 0 and test_terminal[index] for index, value in enumerate(test_fired))),
            "predicted_label_counts_session_max": {
                label: int(sum(value == index and test_terminal[row] for row, value in enumerate(test_fired)))
                for index, label in enumerate(CLASSES)
                if any(value == index and test_terminal[row] for row, value in enumerate(test_fired))
            },
        },
        "evaluation": {
            "full_group_oof_terminal_macro_f1": 0.733523,
            "full_group_oof_delta_vs_geometric": 0.002248,
            "full_group_oof_fixed_harmed": [9, 1],
            "fixed_terminal_macro_f1": 0.708050,
            "fixed_delta_vs_geometric": 0.003033,
            "fixed_fixed_harmed": [1, 0],
        },
        "deployment_gate": "Only if future-history recovery did not deterministically reveal an action AND the sample is the maximum supplied step of its session. Do not add a confidence gate or source/context rule.",
    }
    out = root / "experiments/terminal_rule_combinations"
    out.mkdir(parents=True, exist_ok=True)
    (out / "char20_rule.json").write_text(json.dumps(rule, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    (out / "char20_rule_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
