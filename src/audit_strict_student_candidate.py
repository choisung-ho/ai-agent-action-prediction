#!/usr/bin/env python3
"""Exact B43 audit for one predeclared strict-student candidate.

This writes an experiment report only.  It never changes the live model,
deployment script, or a ZIP archive.  The candidate is a fixed arithmetic
mixture of the archived B43 GRU and one supplied strict-student OOF table;
raw XGBoost fusion and the terminal char20 rule stay immutable.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.model_selection import GroupKFold

from audit_b43_old_strict_half_exact import (
    CLASSES,
    FIXED_EXPECTED_B43,
    FULL_EXPECTED_B43,
    ROOT,
    TO_ID,
    assert_control,
    geometric,
    labels_with_char,
    metric_bundle,
    metric_delta,
    session_bootstrap,
    source_alpha,
    terminal_mask,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--full-student", required=True)
    parser.add_argument("--full-folds", required=True)
    parser.add_argument(
        "--fixed-valid",
        required=True,
        help="valid.npz produced by evaluate_sequence_nested_distill_fixed.py",
    )
    parser.add_argument(
        "--candidate-weight",
        required=True,
        type=float,
        help="Predeclared candidate probability weight, from 0 through 1.",
    )
    parser.add_argument("--bootstrap-replicates", type=int, default=5000)
    parser.add_argument("--out-dir", required=True)
    return parser.parse_args()


def path(value: str) -> Path:
    candidate = Path(value)
    return candidate if candidate.is_absolute() else ROOT / candidate


def load_rows() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rows = [
        json.loads(line)
        for line in (ROOT / "data" / "train.jsonl").open(encoding="utf-8")
        if line.strip()
    ]
    labels = {
        row["id"]: row["action"]
        for row in csv.DictReader((ROOT / "data" / "train_labels.csv").open(encoding="utf-8"))
    }
    split = {
        row["id"]: row["split"]
        for row in csv.DictReader((ROOT / "splits" / "common_split.csv").open(encoding="utf-8"))
    }
    ids = np.asarray([str(row["id"]) for row in rows])
    y = np.asarray([TO_ID[labels[value]] for value in ids], dtype=np.int16)
    groups = np.asarray([value.rsplit("-step_", 1)[0] for value in ids])
    terminal = terminal_mask(ids)
    fixed_index = np.flatnonzero(np.asarray([split[value] == "val" for value in ids]))
    return ids, y, groups, terminal, fixed_index


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.candidate_weight <= 1.0:
        raise ValueError("--candidate-weight must be in [0, 1]")
    if args.bootstrap_replicates < 1:
        raise ValueError("--bootstrap-replicates must be positive")
    out = path(args.out_dir)
    if out == ROOT / "model" or ROOT / "model" in out.parents:
        raise ValueError("Report output cannot be placed in model/")
    out.mkdir(parents=True, exist_ok=True)

    ids, y, groups, terminal, fixed_index = load_rows()
    fixed_ids = ids[fixed_index]
    fixed_y = y[fixed_index]
    fixed_groups = groups[fixed_index]
    fixed_terminal = terminal[fixed_index]
    experiment = ROOT / "experiments"
    raw = np.load(experiment / "xgb_robust_ensemble" / "full_oof_raw_seed42.npy", allow_pickle=False)
    old = np.load(experiment / "modeling_ai_sequence_fusion_noe5" / "oof_probabilities.npy", allow_pickle=False)
    student = np.load(path(args.full_student), allow_pickle=False)
    if not (raw.shape == old.shape == student.shape == (len(y), len(CLASSES))):
        raise ValueError("Full raw/old/student probability tables must each be 70000 x 14")

    expected_fold = np.empty(len(y), dtype=np.int8)
    for number, (_, held) in enumerate(GroupKFold(n_splits=5).split(np.zeros(len(y)), y, groups)):
        expected_fold[held] = number
    if not np.array_equal(
        expected_fold, np.load(path(args.full_folds), allow_pickle=False).astype(np.int8),
    ):
        raise ValueError("Student OOF fold assignment does not equal exact B43 GroupKFold")

    rules = np.load(experiment / "terminal_rule_combinations" / "oof_candidates.npz", allow_pickle=False)
    try:
        if not (
            np.array_equal(rules["y"].astype(np.int16), y)
            and np.array_equal(rules["terminal"].astype(bool), terminal)
            and np.array_equal(rules["fold"].astype(np.int8)[terminal], expected_fold[terminal])
        ):
            raise ValueError("Full immutable char20 artifact does not align")
        char20 = rules["char20"].astype(np.int16)
        fixed_char20 = rules["fixed_char20"].astype(np.int16)
        if not (
            np.array_equal(rules["valid_ids"].astype(str), fixed_ids[fixed_terminal])
            and np.array_equal(rules["fixed_y"].astype(np.int16), fixed_y[fixed_terminal])
        ):
            raise ValueError("Fixed immutable char20 artifact does not align")
    finally:
        rules.close()

    raw_fixed = np.load(experiment / "xgb_robust_ensemble" / "strict_fixed_raw_seed42.npy", allow_pickle=False)
    old_fixed = np.load(
        experiment / "modeling_ai_sequence_fusion_fixed_noe5" / "valid_probabilities.npy",
        allow_pickle=False,
    )
    old_index = np.load(
        experiment / "modeling_ai_sequence_fusion_fixed_noe5" / "valid_indices.npy",
        allow_pickle=False,
    )
    old_terminal = np.load(
        experiment / "modeling_ai_sequence_fusion_fixed_noe5" / "terminal_mask.npy",
        allow_pickle=False,
    )
    if not (
        raw_fixed.shape == old_fixed.shape == (len(fixed_index), len(CLASSES))
        and np.array_equal(old_index, fixed_index)
        and np.array_equal(old_terminal.astype(bool), fixed_terminal)
    ):
        raise ValueError("Exact B43 fixed raw/old components do not align")

    valid = np.load(path(args.fixed_valid), allow_pickle=False)
    try:
        student_fixed = valid["student"].astype(np.float32)
        if not (
            student_fixed.shape == old_fixed.shape
            and np.array_equal(valid["ids"].astype(str), fixed_ids)
            and np.array_equal(valid["target"].astype(np.int16), fixed_y)
            and np.array_equal(valid["terminal"].astype(bool), fixed_terminal)
        ):
            raise ValueError("Candidate fixed student valid.npz does not align")
    finally:
        valid.close()

    fixed_char = np.full(len(fixed_index), -1, dtype=np.int16)
    fixed_char[fixed_terminal] = fixed_char20
    baseline_full_probability = geometric(raw, old, source_alpha(ids))
    baseline_fixed_probability = geometric(raw_fixed, old_fixed, source_alpha(fixed_ids))
    candidate_full_gru = (
        (1.0 - args.candidate_weight) * old + args.candidate_weight * student
    ).astype(np.float32)
    candidate_fixed_gru = (
        (1.0 - args.candidate_weight) * old_fixed + args.candidate_weight * student_fixed
    ).astype(np.float32)
    candidate_full_probability = geometric(raw, candidate_full_gru, source_alpha(ids))
    candidate_fixed_probability = geometric(raw_fixed, candidate_fixed_gru, source_alpha(fixed_ids))

    baseline_full = metric_bundle(y, baseline_full_probability, terminal, char20)
    baseline_fixed = metric_bundle(
        fixed_y, baseline_fixed_probability, fixed_terminal, fixed_char,
    )
    assert_control(baseline_full, FULL_EXPECTED_B43, "full")
    assert_control(baseline_fixed, FIXED_EXPECTED_B43, "fixed")
    candidate_full = metric_bundle(y, candidate_full_probability, terminal, char20)
    candidate_fixed = metric_bundle(
        fixed_y, candidate_fixed_probability, fixed_terminal, fixed_char,
    )
    _, baseline_full_final = labels_with_char(baseline_full_probability, terminal, char20)
    _, candidate_full_final = labels_with_char(candidate_full_probability, terminal, char20)
    _, baseline_fixed_final = labels_with_char(
        baseline_fixed_probability, fixed_terminal, fixed_char,
    )
    _, candidate_fixed_final = labels_with_char(
        candidate_fixed_probability, fixed_terminal, fixed_char,
    )
    full_bootstrap = session_bootstrap(
        y[terminal], baseline_full_final[terminal], candidate_full_final[terminal],
        groups[terminal], args.bootstrap_replicates, 4242,
    )
    fixed_bootstrap = session_bootstrap(
        fixed_y[fixed_terminal], baseline_fixed_final[fixed_terminal],
        candidate_fixed_final[fixed_terminal], fixed_groups[fixed_terminal],
        args.bootstrap_replicates, 4343,
    )
    fold_deltas = []
    for held in range(5):
        mask = expected_fold == held
        before = metric_bundle(
            y[mask], baseline_full_probability[mask], terminal[mask], char20[mask],
        )
        after = metric_bundle(
            y[mask], candidate_full_probability[mask], terminal[mask], char20[mask],
        )
        fold_deltas.append({
            "held_session_fold": held + 1,
            "rows": int(mask.sum()),
            "terminal_rows": int(terminal[mask].sum()),
            "delta": metric_delta(after, before),
        })

    full_delta = metric_delta(candidate_full, baseline_full)
    fixed_delta = metric_delta(candidate_fixed, baseline_fixed)
    all_positive = all(value > 0.0 for value in (*full_delta.values(), *fixed_delta.values()))
    positive_fold_count = sum(
        row["delta"]["terminal_with_char20_macro_f1"] > 0.0 for row in fold_deltas
    )
    stable = bool(
        all_positive
        and full_bootstrap["p_delta_gt_zero"] >= 0.95
        and full_bootstrap["p05_delta"] > 0.0
        and fixed_bootstrap["p_delta_gt_zero"] >= 0.95
        and fixed_bootstrap["p05_delta"] > 0.0
        and positive_fold_count >= 4
    )
    report = {
        "scope": "Experiment only; no model, script, staging directory, or ZIP changed.",
        "candidate_definition": {
            "name": args.name,
            "candidate_weight": args.candidate_weight,
            "definition": "old_gru*(1-w) + strict_student*w; B43 raw geometric fusion and char20 terminal-last unchanged",
            "base_oof_caveat": (
                "This checks strict inner teacher targets and fixed epoch selection only. "
                "The inherited global 97-column base OOF cache is not fully outer-nested."
            ),
        },
        "baseline_b43": {"full": baseline_full, "matching_seed42_fixed": baseline_fixed},
        "candidate": {
            "name": args.name,
            "candidate_weight": args.candidate_weight,
            "full": candidate_full,
            "matching_seed42_fixed": candidate_fixed,
        },
        "delta_vs_b43": {"full": full_delta, "matching_seed42_fixed": fixed_delta},
        "full_terminal_session_bootstrap": full_bootstrap,
        "fixed_terminal_session_bootstrap": fixed_bootstrap,
        "full_outer_fold_deltas": fold_deltas,
        "stability_gate": {
            "all_eight_point_metrics_positive": all_positive,
            "full_terminal_positive_folds": int(positive_fold_count),
            "pass": stable,
        },
        "export_eligible": False,
        "submission_action": "No ZIP created. A base-stack nesting audit is required even if this screen passes.",
    }
    (out / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    print(json.dumps({
        "name": args.name,
        "full_terminal_char20_delta": full_delta["terminal_with_char20_macro_f1"],
        "fixed_terminal_char20_delta": fixed_delta["terminal_with_char20_macro_f1"],
        "stability_gate_pass": stable,
        "report": str((out / "report.json").relative_to(ROOT)).replace("\\", "/"),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
