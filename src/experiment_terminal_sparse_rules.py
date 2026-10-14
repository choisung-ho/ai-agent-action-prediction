#!/usr/bin/env python3
"""Audit conservative terminal-only sparse lexical/context rules.

This is deliberately an experiment-only file.  It never changes ``model/``,
``script.py`` or a submission archive.  The intended deployment setting is
the batch-safe future-history path: every row not exactly exposed by a later
same-session record is the maximum supplied step in its session.  Such rows
are evaluated here with whole-session folds.

The main question is whether contextual, *label-pure* phrase rules can safely
cover residual terminal rows beyond the existing all-row prompt rule.  A rule
is emitted only where every supporting training terminal session has the same
label.  At prediction time every fired rule must agree; otherwise it abstains.
No model prediction or held-out label is included in a rule key.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold


CLASSES = (
    "read_file", "grep_search", "list_directory", "glob_pattern",
    "edit_file", "write_file", "apply_patch", "run_bash",
    "run_tests", "lint_or_typecheck", "ask_user", "plan_task",
    "web_search", "respond_only",
)
LABEL_TO_ID = {label: index for index, label in enumerate(CLASSES)}
WORD = re.compile(r"(?u)\b\w+\b")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/train.jsonl")
    parser.add_argument("--labels", default="data/train_labels.csv")
    parser.add_argument("--split", default="splits/common_split.csv")
    parser.add_argument("--xgb-oof", default="experiments/modeling_ak_live_xgb_proxy/oof_probabilities.npy")
    parser.add_argument("--gru-oof", default="experiments/modeling_ai_sequence_fusion_noe5/oof_probabilities.npy")
    parser.add_argument("--fixed-xgb", default="experiments/modeling_ac_tune2/oof_probabilities.npy")
    parser.add_argument("--fixed-gru", default="experiments/modeling_ai_sequence_fusion_fixed_noe5/valid_probabilities.npy")
    parser.add_argument("--out-dir", default="experiments/terminal_sparse_rules")
    return parser.parse_args()


def rows_from(path: str) -> list[dict]:
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def labels_from(path: str) -> dict[str, str]:
    with Path(path).open(encoding="utf-8") as handle:
        return {str(row["id"]): str(row["action"]) for row in csv.DictReader(handle)}


def split_from(path: str) -> dict[str, str]:
    with Path(path).open(encoding="utf-8") as handle:
        return {str(row["id"]): str(row["split"]) for row in csv.DictReader(handle)}


def session_of(sample: dict) -> str:
    return str(sample.get("id", "")).rsplit("-step_", 1)[0]


def step_of(sample: dict) -> int:
    try:
        return int(str(sample.get("id", "")).rsplit("-step_", 1)[1])
    except (IndexError, ValueError):
        return -1


def source_of(sample: dict) -> str:
    sample_id = str(sample.get("id", ""))
    if sample_id.startswith("sess_au_"):
        return "au"
    if sample_id.startswith("sess_sim_"):
        return "sim"
    return "other"


def result_bucket(value: object) -> str:
    text = str(value or "").casefold()
    if any(token in text for token in ("fail", "error", "exception", "traceback", "not found", "0 match", "0 file")):
        return "fail"
    if any(token in text for token in ("pass", "ok;", "exit=0", "modified", "patched", "matched", "found", "listed")):
        return "ok"
    return "other"


def prompt_grams(sample: dict, low: int, high: int) -> list[str]:
    tokens = WORD.findall(str(sample.get("current_prompt", "")).casefold())[:70]
    output: set[str] = set()
    for width in range(low, high + 1):
        output.update("_".join(tokens[pos:pos + width]) for pos in range(max(0, len(tokens) - width + 1)))
    return sorted(output)


def context_values(sample: dict) -> dict[str, str]:
    history = sample.get("history") if isinstance(sample.get("history"), list) else []
    actions = [
        item for item in history
        if isinstance(item, dict) and item.get("role") == "assistant_action"
    ]
    names = [str(item.get("name") or item.get("action") or "none").casefold() for item in actions]
    last = names[-1] if names else "none"
    second = names[-2] if len(names) > 1 else "none"
    last_result = result_bucket(actions[-1].get("result_summary")) if actions else "none"
    meta = sample.get("session_meta") if isinstance(sample.get("session_meta"), dict) else {}
    try:
        turn = int(meta.get("turn_index", step_of(sample)))
    except (TypeError, ValueError):
        turn = step_of(sample)
    if turn <= 2:
        turn_bucket = "early"
    elif turn <= 5:
        turn_bucket = "mid"
    elif turn <= 9:
        turn_bucket = "late"
    else:
        turn_bucket = "long"
    return {
        "source": source_of(sample),
        "last": last,
        "last2": second + ">" + last,
        "result": last_result,
        "turn": turn_bucket,
        "tier": str(meta.get("user_tier", "unknown")).casefold(),
        "language": str(meta.get("language_pref", "unknown")).casefold(),
    }


def feature_keys(sample: dict, family: str) -> Iterable[str]:
    context = context_values(sample)
    if family == "prompt":
        yield from ("p|" + gram for gram in prompt_grams(sample, 1, 4))
        return
    # Contextual keys deliberately start at bigrams: an isolated word combined
    # with a prior action is still common enough to look pure by chance.
    grams = prompt_grams(sample, 2, 4)
    if family == "source":
        prefix = "s=" + context["source"] + "|"
    elif family == "last":
        prefix = "a=" + context["last"] + "|"
    elif family == "last2":
        prefix = "a2=" + context["last2"] + "|"
    elif family == "result":
        prefix = "r=" + context["result"] + "|"
    elif family == "source_last":
        prefix = "sa=" + context["source"] + ":" + context["last"] + "|"
    elif family == "turn":
        prefix = "t=" + context["turn"] + "|"
    elif family == "meta":
        prefix = "m=" + context["tier"] + ":" + context["language"] + "|"
    else:
        raise ValueError(f"Unknown family {family!r}")
    yield from (prefix + gram for gram in grams)


def terminal_mask(rows: list[dict]) -> np.ndarray:
    maximum: dict[str, int] = {}
    for sample in rows:
        group = session_of(sample)
        maximum[group] = max(maximum.get(group, -1), step_of(sample))
    return np.asarray([step_of(sample) == maximum[session_of(sample)] for sample in rows], dtype=bool)


def fit_rules(rows: list[dict], labels: np.ndarray, train_index: np.ndarray, families: tuple[str, ...], min_support: int) -> dict[str, int]:
    """Fit label-pure rule table; terminal rows make support=session support."""
    counts: dict[str, Counter[int]] = defaultdict(Counter)
    for index in train_index:
        # feature_keys returns a set-derived list, so a terminal sample can
        # contribute at most once to any particular feature key.
        for key in feature_keys(rows[int(index)], family="prompt") if families == ("prompt",) else ():
            counts[key][int(labels[index])] += 1
        if families != ("prompt",):
            for family in families:
                for key in feature_keys(rows[int(index)], family):
                    counts[key][int(labels[index])] += 1
    return {
        key: next(iter(value))
        for key, value in counts.items()
        if sum(value.values()) >= min_support and len(value) == 1
    }


def predict_rules(rows: list[dict], index: np.ndarray, rule: dict[str, int], families: tuple[str, ...]) -> np.ndarray:
    prediction = np.full(len(index), -1, dtype=np.int16)
    for position, row_index in enumerate(index):
        fired: set[int] = set()
        for family in families:
            fired.update(rule[key] for key in feature_keys(rows[int(row_index)], family) if key in rule)
        if len(fired) == 1:
            prediction[position] = next(iter(fired))
    return prediction


def geometric(xgb: np.ndarray, gru: np.ndarray, source: np.ndarray) -> np.ndarray:
    alpha = np.where(source == "sim", 0.20, 0.85).astype(np.float32)
    log_probability = alpha[:, None] * np.log(np.maximum(xgb, 1.0e-12))
    log_probability += (1.0 - alpha[:, None]) * np.log(np.maximum(gru, 1.0e-12))
    log_probability -= log_probability.max(axis=1, keepdims=True)
    probability = np.exp(log_probability)
    return probability / probability.sum(axis=1, keepdims=True)


def macro(y: np.ndarray, prediction: np.ndarray) -> float:
    return float(f1_score(y, prediction, labels=np.arange(len(CLASSES)), average="macro", zero_division=0))


def assess_full_group_cv(rows: list[dict], y: np.ndarray, terminal: np.ndarray, base_prediction: np.ndarray) -> dict:
    """Fit terminal-only tables in each group fold and assess cascaded rules."""
    groups = np.asarray([session_of(sample) for sample in rows])
    configs = {
        "prompt_terminal_s4": (("prompt",), 4),
        "prompt_terminal_s6": (("prompt",), 6),
        "prompt_terminal_s8": (("prompt",), 8),
        "context_core_s3": (("source", "last", "result", "turn"), 3),
        "context_core_s4": (("source", "last", "result", "turn"), 4),
        "context_core_s5": (("source", "last", "result", "turn"), 5),
        "context_lastseq_s3": (("source", "last", "last2", "result", "source_last", "turn", "meta"), 3),
        "context_lastseq_s4": (("source", "last", "last2", "result", "source_last", "turn", "meta"), 4),
        "context_lastseq_s5": (("source", "last", "last2", "result", "source_last", "turn", "meta"), 5),
    }
    # Existing all-row prompt policy is included as the first safe cascade
    # stage.  Its support uses rows (matching the already-audited policy).
    baseline_name = "prompt_all_s15"
    all_results: dict[str, dict] = {}
    folds = list(GroupKFold(n_splits=5).split(np.zeros(len(rows)), y, groups))
    for name, (families, support) in configs.items():
        rule_prediction = np.full(len(rows), -1, dtype=np.int16)
        stage_prediction = np.full(len(rows), -1, dtype=np.int16)
        fold_records: list[dict] = []
        for fold, (train, valid) in enumerate(folds, start=1):
            train_terminal = train[terminal[train]]
            valid_terminal = valid[terminal[valid]]
            rule = fit_rules(rows, y, train_terminal, families, support)
            prediction = predict_rules(rows, valid_terminal, rule, families)
            rule_prediction[valid_terminal] = prediction
            fold_mask = prediction >= 0
            fold_records.append({
                "fold": fold,
                "rule_count": len(rule),
                "n_terminal": int(len(valid_terminal)),
                "n_fired": int(fold_mask.sum()),
                "precision": float((prediction[fold_mask] == y[valid_terminal][fold_mask]).mean()) if fold_mask.any() else None,
            })
        mask = terminal & (rule_prediction >= 0)
        overlay = base_prediction.copy()
        overlay[mask] = rule_prediction[mask]
        all_results[name] = {
            "families": list(families), "min_terminal_session_support": support,
            "n_candidate": int(mask.sum()),
            "coverage_terminal": float(mask.sum() / terminal.sum()),
            "precision": float((rule_prediction[mask] == y[mask]).mean()) if mask.any() else 0.0,
            "base_correct_candidate": int((base_prediction[mask] == y[mask]).sum()),
            "fixed": int(((base_prediction != y) & (rule_prediction == y) & mask).sum()),
            "harmed": int(((base_prediction == y) & (rule_prediction != y) & mask).sum()),
            "terminal_macro_f1": macro(y[terminal], overlay[terminal]),
            "all_hybrid_macro_f1": macro(y, np.where(terminal, overlay, y)),
            "folds": fold_records,
            "prediction": rule_prediction,
        }

    # First-stage global pure prompt rules, fitted on all non-held-out rows.
    primary = np.full(len(rows), -1, dtype=np.int16)
    primary_folds: list[dict] = []
    for fold, (train, valid) in enumerate(folds, start=1):
        valid_terminal = valid[terminal[valid]]
        rule = fit_rules(rows, y, train, ("prompt",), 15)
        prediction = predict_rules(rows, valid_terminal, rule, ("prompt",))
        primary[valid_terminal] = prediction
        fired = prediction >= 0
        primary_folds.append({"fold": fold, "rule_count": len(rule), "n_fired": int(fired.sum()),
                              "precision": float((prediction[fired] == y[valid_terminal][fired]).mean()) if fired.any() else None})
    primary_mask = terminal & (primary >= 0)
    primary_overlay = base_prediction.copy(); primary_overlay[primary_mask] = primary[primary_mask]
    all_results[baseline_name] = {
        "families": ["prompt"], "min_row_support": 15,
        "n_candidate": int(primary_mask.sum()), "coverage_terminal": float(primary_mask.sum() / terminal.sum()),
        "precision": float((primary[primary_mask] == y[primary_mask]).mean()),
        "base_correct_candidate": int((base_prediction[primary_mask] == y[primary_mask]).sum()),
        "fixed": int(((base_prediction != y) & (primary == y) & primary_mask).sum()),
        "harmed": int(((base_prediction == y) & (primary != y) & primary_mask).sum()),
        "terminal_macro_f1": macro(y[terminal], primary_overlay[terminal]),
        "all_hybrid_macro_f1": macro(y, np.where(terminal, primary_overlay, y)),
        "folds": primary_folds, "prediction": primary,
    }

    # Cascades never allow an auxiliary context rule to replace the established
    # all-row prompt decision.  They only act on primary abstentions.
    for name, record in list(all_results.items()):
        if name == baseline_name:
            continue
        candidate = record["prediction"]
        cascade = primary.copy()
        add = terminal & (cascade < 0) & (candidate >= 0)
        cascade[add] = candidate[add]
        mask = terminal & (cascade >= 0)
        overlay = base_prediction.copy(); overlay[mask] = cascade[mask]
        all_results["cascade_" + name] = {
            "primary": baseline_name, "secondary": name,
            "n_candidate": int(mask.sum()), "coverage_terminal": float(mask.sum() / terminal.sum()),
            "precision": float((cascade[mask] == y[mask]).mean()),
            "added_after_primary": int(add.sum()),
            "added_precision": float((candidate[add] == y[add]).mean()) if add.any() else None,
            "base_correct_candidate": int((base_prediction[mask] == y[mask]).sum()),
            "fixed": int(((base_prediction != y) & (cascade == y) & mask).sum()),
            "harmed": int(((base_prediction == y) & (cascade != y) & mask).sum()),
            "terminal_macro_f1": macro(y[terminal], overlay[terminal]),
            "all_hybrid_macro_f1": macro(y, np.where(terminal, overlay, y)),
            "folds": record["folds"], "prediction": cascade,
        }
    # arrays are only internal evaluation state and make JSON serializable.
    for record in all_results.values():
        record.pop("prediction", None)
    return all_results


def assess_fixed_split(rows: list[dict], y: np.ndarray, terminal: np.ndarray, split: dict[str, str], base_prediction: np.ndarray) -> dict:
    """Independent fixed split check for the strongest compact rule families."""
    train = np.asarray([idx for idx, sample in enumerate(rows) if split[str(sample["id"])] == "train"], dtype=int)
    valid = np.asarray([idx for idx, sample in enumerate(rows) if split[str(sample["id"])] == "val"], dtype=int)
    train_terminal = train[terminal[train]]
    valid_terminal = valid[terminal[valid]]
    specs = {
        "prompt_all_s15": (("prompt",), 15, train),
        "context_core_s3": (("source", "last", "result", "turn"), 3, train_terminal),
        "context_core_s4": (("source", "last", "result", "turn"), 4, train_terminal),
        "context_core_s5": (("source", "last", "result", "turn"), 5, train_terminal),
        "context_lastseq_s3": (("source", "last", "last2", "result", "source_last", "turn", "meta"), 3, train_terminal),
        "context_lastseq_s4": (("source", "last", "last2", "result", "source_last", "turn", "meta"), 4, train_terminal),
        "context_lastseq_s5": (("source", "last", "last2", "result", "source_last", "turn", "meta"), 5, train_terminal),
    }
    predictions: dict[str, np.ndarray] = {}
    result: dict[str, dict] = {}
    for name, (families, support, fit_index) in specs.items():
        rule = fit_rules(rows, y, fit_index, families, support)
        prediction = predict_rules(rows, valid_terminal, rule, families)
        predictions[name] = prediction
        mask = prediction >= 0
        overlay = base_prediction.copy(); overlay[mask] = prediction[mask]
        result[name] = {
            "rule_count": len(rule), "n_candidate": int(mask.sum()),
            "coverage_terminal": float(mask.sum() / len(valid_terminal)),
            "precision": float((prediction[mask] == y[valid_terminal][mask]).mean()) if mask.any() else 0.0,
            "fixed": int(((base_prediction != y[valid_terminal]) & (prediction == y[valid_terminal]) & mask).sum()),
            "harmed": int(((base_prediction == y[valid_terminal]) & (prediction != y[valid_terminal]) & mask).sum()),
            "terminal_macro_f1": macro(y[valid_terminal], overlay),
        }
    primary = predictions["prompt_all_s15"]
    for name, prediction in list(predictions.items()):
        if name == "prompt_all_s15":
            continue
        cascade = primary.copy()
        add = (cascade < 0) & (prediction >= 0)
        cascade[add] = prediction[add]
        mask = cascade >= 0
        overlay = base_prediction.copy(); overlay[mask] = cascade[mask]
        result["cascade_" + name] = {
            "n_candidate": int(mask.sum()), "coverage_terminal": float(mask.sum() / len(valid_terminal)),
            "precision": float((cascade[mask] == y[valid_terminal][mask]).mean()),
            "added_after_primary": int(add.sum()),
            "added_precision": float((prediction[add] == y[valid_terminal][add]).mean()) if add.any() else None,
            "fixed": int(((base_prediction != y[valid_terminal]) & (cascade == y[valid_terminal]) & mask).sum()),
            "harmed": int(((base_prediction == y[valid_terminal]) & (cascade != y[valid_terminal]) & mask).sum()),
            "terminal_macro_f1": macro(y[valid_terminal], overlay),
        }
    return result


def main() -> None:
    args = parse_args()
    rows = rows_from(args.data)
    labels = labels_from(args.labels)
    split = split_from(args.split)
    y = np.asarray([LABEL_TO_ID[labels[str(sample["id"])]] for sample in rows], dtype=np.int64)
    terminal = terminal_mask(rows)
    source = np.asarray([source_of(sample) for sample in rows])
    xgb = np.load(args.xgb_oof).astype(np.float32)
    gru = np.load(args.gru_oof).astype(np.float32)
    if xgb.shape != gru.shape or xgb.shape != (len(rows), len(CLASSES)):
        raise ValueError(f"Full OOF shape mismatch: xgb={xgb.shape}, gru={gru.shape}, rows={len(rows)}")
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    checkpoint = out / "full_group_cv_checkpoint.json"
    base = geometric(xgb, gru, source).argmax(axis=1)
    if checkpoint.exists():
        group = json.loads(checkpoint.read_text(encoding="utf-8"))
    else:
        group = assess_full_group_cv(rows, y, terminal, base)
        # The fold audit is the expensive portion; retain it before the
        # independent fixed-split calculation so an interrupted follow-up can
        # still be audited.
        checkpoint.write_text(json.dumps(group, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    valid_index = np.asarray([idx for idx, sample in enumerate(rows) if split[str(sample["id"])] == "val"], dtype=int)
    fixed_xgb = np.load(args.fixed_xgb).astype(np.float32)
    fixed_gru = np.load(args.fixed_gru).astype(np.float32)
    if fixed_xgb.shape != fixed_gru.shape or fixed_xgb.shape != (len(valid_index), len(CLASSES)):
        raise ValueError(f"Fixed OOF shape mismatch: xgb={fixed_xgb.shape}, gru={fixed_gru.shape}, val={len(valid_index)}")
    fixed_source = source[valid_index]
    fixed_base = geometric(fixed_xgb, fixed_gru, fixed_source).argmax(axis=1)
    fixed_terminal_base = fixed_base[terminal[valid_index]]
    fixed = assess_fixed_split(rows, y, terminal, split, fixed_terminal_base)

    result = {
        "scope": "Experiment only: no model/script/submission archive was modified.",
        "protocol": {
            "full": "5-fold GroupKFold, whole sessions held out; every terminal-only rule table is rebuilt using only the other sessions.",
            "fixed": "Common frozen train/validation split; rules fit only common-train sessions and evaluated on its terminal sessions.",
            "baseline": "raw XGB + no-E5 GRU geometric terminal fallback (sim alpha=0.20, au alpha=0.85).",
        },
        "counts": {"rows": len(rows), "terminal": int(terminal.sum()), "fixed_terminal": int(terminal[valid_index].sum())},
        "base": {
            "full_terminal_macro_f1": macro(y[terminal], base[terminal]),
            "fixed_terminal_macro_f1": macro(y[valid_index][terminal[valid_index]], fixed_terminal_base),
        },
        "full_group_cv": group,
        "fixed_split": fixed,
    }
    (out / "report.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
