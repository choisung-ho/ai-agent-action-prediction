#!/usr/bin/env python3
"""Audit a label-free batch/session Viterbi decoder for the B42_9 fallback.

Every candidate only consumes base fallback probabilities for the current
batch and a transition table learned from *other* training sessions.  It does
not consume future target labels or a test-session ID lookup.  This matters
because the submission receives all supplied rows at once, so future-row
*features* are available even when future actions are not.

The experiment deliberately stays isolated under ``experiments/``.  It is
not a submission builder and never writes model/, script.py, or any ZIP.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold


ROOT = Path(__file__).resolve().parents[1]
CLASSES = [
    "read_file", "grep_search", "list_directory", "glob_pattern",
    "edit_file", "write_file", "apply_patch", "run_bash",
    "run_tests", "lint_or_typecheck", "ask_user", "plan_task",
    "web_search", "respond_only",
]
N_CLASSES = len(CLASSES)
TO_ID = {label: number for number, label in enumerate(CLASSES)}
EPS = 1.0e-12
# Small predeclared grid.  Lambda=0 is the exact B42_9 fallback baseline.
LAMBDAS = (0.0, 0.08, 0.16, 0.28, 0.45, 0.70, 1.00)
KINDS = ("global", "source")


def load_rows() -> tuple[list[dict], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rows = [json.loads(line) for line in (ROOT / "data/train.jsonl").open(encoding="utf-8") if line.strip()]
    labels = {row["id"]: row["action"] for row in csv.DictReader((ROOT / "data/train_labels.csv").open(encoding="utf-8"))}
    ids = np.asarray([str(row["id"]) for row in rows])
    y = np.asarray([TO_ID[labels[sample_id]] for sample_id in ids], dtype=np.int16)
    session = np.asarray([sample_id.rsplit("-step_", 1)[0] for sample_id in ids])
    step = np.asarray([int(sample_id.rsplit("-step_", 1)[1]) for sample_id in ids], dtype=np.int32)
    source = np.asarray(["au" if sample_id.startswith("sess_au_") else "sim" for sample_id in ids])
    return rows, y, session, step, source


def terminal_mask(session: np.ndarray, step: np.ndarray) -> np.ndarray:
    maximum: dict[str, int] = {}
    for group, value in zip(session.tolist(), step.tolist()):
        maximum[group] = max(maximum.get(group, -1), int(value))
    return np.asarray([value == maximum[group] for group, value in zip(session.tolist(), step.tolist())], dtype=bool)


def geometric(xgb: np.ndarray, gru: np.ndarray, source: np.ndarray) -> np.ndarray:
    alpha = np.where(source == "au", 0.85, 0.20).astype(np.float64)[:, None]
    value = alpha * np.log(np.maximum(xgb, EPS)) + (1.0 - alpha) * np.log(np.maximum(gru, EPS))
    value -= value.max(axis=1, keepdims=True)
    value = np.exp(value)
    return (value / value.sum(axis=1, keepdims=True)).astype(np.float32)


def macro(y: np.ndarray, prediction: np.ndarray, mask: np.ndarray) -> float:
    return float(f1_score(y[mask], prediction[mask], labels=np.arange(N_CLASSES), average="macro", zero_division=0))


def fit_transitions(y: np.ndarray, session: np.ndarray, step: np.ndarray, source: np.ndarray, train: np.ndarray, kind: str):
    """Return smoothed log start and transition probabilities.

    For the source mode, sparse AU counts back off to global counts.  The
    decoder's state itself contains only candidate labels, never true labels.
    """
    start_global = np.ones(N_CLASSES, dtype=np.float64)
    trans_global = np.ones((N_CLASSES, N_CLASSES), dtype=np.float64)
    start_source = {key: np.ones(N_CLASSES, dtype=np.float64) for key in ("sim", "au")}
    trans_source = {key: np.ones((N_CLASSES, N_CLASSES), dtype=np.float64) for key in ("sim", "au")}
    grouped: dict[str, list[int]] = defaultdict(list)
    for index in np.flatnonzero(train):
        grouped[str(session[index])].append(int(index))
    for indices in grouped.values():
        indices.sort(key=lambda index: int(step[index]))
        first = indices[0]
        group_source = str(source[first])
        start_global[int(y[first])] += 1.0
        start_source[group_source][int(y[first])] += 1.0
        for left, right in zip(indices, indices[1:]):
            trans_global[int(y[left]), int(y[right])] += 1.0
            trans_source[group_source][int(y[left]), int(y[right])] += 1.0
    global_payload = (
        np.log(start_global / start_global.sum()),
        np.log(trans_global / trans_global.sum(axis=1, keepdims=True)),
    )
    if kind == "global":
        return {"sim": global_payload, "au": global_payload}
    payload = {}
    for key in ("sim", "au"):
        # A convex global backoff protects the small AU source without
        # leaking the evaluation group or adding a selected fitted weight.
        start = start_source[key] + 0.20 * start_global
        trans = trans_source[key] + 0.20 * trans_global
        payload[key] = (np.log(start / start.sum()), np.log(trans / trans.sum(axis=1, keepdims=True)))
    return payload


def decode(probability: np.ndarray, session: np.ndarray, step: np.ndarray, source: np.ndarray, transition, strength: float) -> np.ndarray:
    output = probability.argmax(axis=1).astype(np.int16)
    if strength == 0.0:
        return output
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, group in enumerate(session.tolist()):
        grouped[group].append(index)
    log_emit = np.log(np.maximum(probability, EPS))
    for indices in grouped.values():
        indices.sort(key=lambda index: int(step[index]))
        group_source = str(source[indices[0]])
        start, trans = transition[group_source]
        state = np.empty((len(indices), N_CLASSES), dtype=np.float64)
        back = np.zeros((len(indices), N_CLASSES), dtype=np.int16)
        state[0] = log_emit[indices[0]] + strength * start
        for pos, index in enumerate(indices[1:], start=1):
            score = state[pos - 1][:, None] + strength * trans
            back[pos] = score.argmax(axis=0)
            state[pos] = log_emit[index] + score.max(axis=0)
        chosen = int(state[-1].argmax())
        for pos in range(len(indices) - 1, -1, -1):
            output[indices[pos]] = chosen
            chosen = int(back[pos, chosen])
    return output


def nested_full(y, session, step, source, probability, terminal, folds):
    """Select the small decoder config inside four folds, then score held fold."""
    prediction = np.empty(len(y), dtype=np.int16)
    records = []
    for held in sorted(np.unique(folds).tolist()):
        outer_train = folds != held
        candidates = []
        for kind in KINDS:
            for strength in LAMBDAS:
                inner = np.empty(len(y), dtype=np.int16)
                for valid in sorted(np.unique(folds).tolist()):
                    if valid == held:
                        continue
                    fit = outer_train & (folds != valid)
                    transition = fit_transitions(y, session, step, source, fit, kind)
                    valid_mask = folds == valid
                    inner[valid_mask] = decode(probability[valid_mask], session[valid_mask], step[valid_mask], source[valid_mask], transition, strength)
                mask = outer_train
                candidates.append((macro(y, inner, mask & terminal), macro(y, inner, mask), kind, strength))
        # Terminal objective first, all-row as an explicit tie-breaker, then
        # lower capacity.  This mirrors the terminal-heavy public setting.
        _, _, kind, strength = max(candidates, key=lambda item: (item[0], item[1], -item[3], item[2] == "global"))
        transition = fit_transitions(y, session, step, source, outer_train, kind)
        held_mask = folds == held
        prediction[held_mask] = decode(probability[held_mask], session[held_mask], step[held_mask], source[held_mask], transition, strength)
        records.append({"held_fold": int(held + 1), "kind": kind, "strength": strength})
    return prediction, records


def select_common(y, session, step, source, probability, terminal):
    folds = np.empty(len(y), dtype=np.int8)
    for number, (_, valid) in enumerate(GroupKFold(n_splits=5).split(np.zeros(len(y)), y, session)):
        folds[valid] = number
    table = []
    for kind in KINDS:
        for strength in LAMBDAS:
            prediction = np.empty(len(y), dtype=np.int16)
            for held in range(5):
                fit = folds != held
                transition = fit_transitions(y, session, step, source, fit, kind)
                valid = folds == held
                prediction[valid] = decode(probability[valid], session[valid], step[valid], source[valid], transition, strength)
            table.append({"kind": kind, "strength": strength, "terminal": macro(y, prediction, terminal), "all": macro(y, prediction, np.ones(len(y), dtype=bool))})
    chosen = max(table, key=lambda row: (row["terminal"], row["all"], -row["strength"], row["kind"] == "global"))
    return chosen, table


def load_fixed(rows, y, session, step, source):
    split = {row["id"]: row["split"] for row in csv.DictReader((ROOT / "splits/common_split.csv").open(encoding="utf-8"))}
    train = np.asarray([split[str(row["id"])] == "train" for row in rows], dtype=bool)
    valid = ~train
    old = np.load(ROOT / "experiments/b42_9_common_old_noe5_seed42/common_train_oof_probabilities.npy", allow_pickle=False).astype(np.float32)
    old_ids = np.load(ROOT / "experiments/b42_9_common_old_noe5_seed42/common_train_ids.npy", allow_pickle=False).astype(str)
    raw = np.load(ROOT / "experiments/b42_9_common_raw_seed42/common_train_raw_xgb_oof.npy", allow_pickle=False).astype(np.float32)
    raw_ids = np.load(ROOT / "experiments/b42_9_common_raw_seed42/common_train_ids.npy", allow_pickle=False).astype(str)
    train_ids = np.asarray([str(row["id"]) for row, value in zip(rows, train) if value])
    if not (np.array_equal(old_ids, train_ids) and np.array_equal(raw_ids, train_ids)):
        raise ValueError("Common-train probability IDs are not aligned")
    fixed_old = np.load(ROOT / "experiments/b42_9_common_old_noe5_seed42/fixed_valid_probabilities.npy", allow_pickle=False).astype(np.float32)
    fixed_ids = np.load(ROOT / "experiments/b42_9_common_old_noe5_seed42/fixed_valid_ids.npy", allow_pickle=False).astype(str)
    fixed_raw = np.load(ROOT / "experiments/xgb_robust_ensemble/strict_fixed_raw_seed42.npy", allow_pickle=False).astype(np.float32)
    valid_ids = np.asarray([str(row["id"]) for row, value in zip(rows, valid) if value])
    if not (np.array_equal(fixed_ids, valid_ids) and fixed_raw.shape == fixed_old.shape):
        raise ValueError("Fixed probability IDs are not aligned")
    return train, valid, geometric(raw, old, source[train]), geometric(fixed_raw, fixed_old, source[valid])


def main() -> None:
    out = ROOT / "experiments/viterbi_session_decoder"; out.mkdir(parents=True, exist_ok=True)
    rows, y, session, step, source = load_rows()
    terminal = terminal_mask(session, step)
    xgb = np.load(ROOT / "experiments/xgb_robust_ensemble/full_oof_raw_seed42.npy", allow_pickle=False).astype(np.float32)
    old = np.load(ROOT / "experiments/modeling_ai_sequence_fusion_noe5/oof_probabilities.npy", allow_pickle=False).astype(np.float32)
    probability = geometric(xgb, old, source)
    folds = np.empty(len(y), dtype=np.int8)
    for number, (_, valid) in enumerate(GroupKFold(n_splits=5).split(np.zeros(len(y)), y, session)):
        folds[valid] = number
    base = probability.argmax(axis=1)
    full_pred, fold_records = nested_full(y, session, step, source, probability, terminal, folds)
    train, valid, common_probability, fixed_probability = load_fixed(rows, y, session, step, source)
    chosen, common_table = select_common(y[train], session[train], step[train], source[train], common_probability, terminal[train])
    transition = fit_transitions(y, session, step, source, train, chosen["kind"])
    fixed_pred = decode(fixed_probability, session[valid], step[valid], source[valid], transition, float(chosen["strength"]))
    base_fixed = fixed_probability.argmax(axis=1)
    result = {
        "scope": "Experiment only; model/, script.py, B42.zip and baseline_submit.zip are unchanged.",
        "contract": "Viterbi uses only all supplied rows' model probabilities and transition tables fit from disjoint training sessions. No target future label, test ID lookup, or rule label is used.",
        "full_nested": {
            "base_all": macro(y, base, np.ones(len(y), dtype=bool)),
            "base_terminal": macro(y, base, terminal),
            "decoder_all": macro(y, full_pred, np.ones(len(y), dtype=bool)),
            "decoder_terminal": macro(y, full_pred, terminal),
            "fold_choices": fold_records,
        },
        "fixed": {
            "selected_common_train": chosen,
            "common_grid": common_table,
            "base_all": macro(y[valid], base_fixed, np.ones(valid.sum(), dtype=bool)),
            "base_terminal": macro(y[valid], base_fixed, terminal[valid]),
            "decoder_all": macro(y[valid], fixed_pred, np.ones(valid.sum(), dtype=bool)),
            "decoder_terminal": macro(y[valid], fixed_pred, terminal[valid]),
        },
    }
    full_good = result["full_nested"]["decoder_all"] > result["full_nested"]["base_all"] and result["full_nested"]["decoder_terminal"] > result["full_nested"]["base_terminal"]
    fixed_good = result["fixed"]["decoder_all"] > result["fixed"]["base_all"] and result["fixed"]["decoder_terminal"] > result["fixed"]["base_terminal"]
    result["decision"] = "PROMOTE_CANDIDATE" if full_good and fixed_good else "REJECT"
    (out / "report.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
