#!/usr/bin/env python3
"""Session-state residual-rule audit against B42_9 and strict fallback paths.

This experiment tests a compact, causal residual representation that had not
been part of the previous probability-only calibration work:

``(source, raw selector argmax, latest observed action, latest result status)``.

For a supplied session-max row those fields are available before the target
action is chosen.  A terminal-only table may replace the learned fallback
only when the state has a label-pure, sufficiently supported training-side
outcome.  Existing B42_9 word15 and B43/current char20 prompt overrides still
run *after* the state table, preserving their intended priority.

All rule tables are fit inside held-session folds or on the common-train side
only.  The fixed validation evaluation deliberately keys common-train rows by
selector OOF outputs and the fixed rows by a selector fit without them.  This
is the deployment-relevant check for selector-state drift.  This file changes
only ``experiments/state_residual_gate``.
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
TO_ID = {value: index for index, value in enumerate(CLASSES)}
WORD = re.compile(r"(?u)\b\w+\b")


def session_of(row: dict) -> str:
    return str(row["id"]).split("-step_", 1)[0]


def step_of(row: dict) -> int:
    return int(str(row["id"]).rsplit("-step_", 1)[1])


def source_of(row: dict) -> str:
    return "au" if str(row["id"]).startswith("sess_au_") else "sim"


def terminal_mask(rows: list[dict]) -> np.ndarray:
    maximum: dict[str, int] = {}
    for row in rows:
        group = session_of(row)
        maximum[group] = max(maximum.get(group, -1), step_of(row))
    return np.asarray([step_of(row) == maximum[session_of(row)] for row in rows], dtype=bool)


def status(value: object) -> str:
    text = str(value or "").casefold()
    if any(token in text for token in ("fail", "error", "exception", "traceback", "not found", "0 match", "0 file")):
        return "fail"
    if any(token in text for token in ("pass", "ok;", "exit=0", "modified", "patched", "matched", "found", "listed")):
        return "ok"
    return "other"


def state_key(row: dict, raw_prediction: int) -> tuple[str, int, str, str]:
    """Stable causal state: no prompt text, target label, or later row."""
    history = row.get("history") if isinstance(row.get("history"), list) else []
    action = next((item for item in reversed(history)
                   if isinstance(item, dict) and item.get("role") == "assistant_action"), None)
    if action is None:
        name, result = "none", "other"
    else:
        name = str(action.get("name") or action.get("action") or "none").casefold()
        result = status(action.get("result_summary"))
    return source_of(row), int(raw_prediction), name, result


def build_state_rule(rows: list[dict], target: np.ndarray, train: np.ndarray,
                     raw_prediction: np.ndarray, support: int, purity: float) -> dict[tuple[str, int, str, str], int]:
    counts: dict[tuple[str, int, str, str], Counter[int]] = defaultdict(Counter)
    for index in train:
        value = int(index)
        counts[state_key(rows[value], int(raw_prediction[value]))][int(target[value])] += 1
    output: dict[tuple[str, int, str, str], int] = {}
    for key, labels in counts.items():
        total = sum(labels.values())
        label, count = labels.most_common(1)[0]
        if total >= support and count / total >= purity:
            output[key] = int(label)
    return output


def decide_state(rows: list[dict], query: np.ndarray, raw_prediction: np.ndarray,
                 rule: dict[tuple[str, int, str, str], int]) -> np.ndarray:
    output = np.full(len(rows), -1, dtype=np.int16)
    for index in query:
        value = int(index)
        output[value] = rule.get(state_key(rows[value], int(raw_prediction[value])), -1)
    return output


def prompt_features(row: dict) -> tuple[str, ...]:
    tokens = WORD.findall(str(row.get("current_prompt") or "").casefold())[:70]
    return tuple({
        "w|" + "_".join(tokens[offset:offset + width])
        for width in range(1, 5)
        for offset in range(max(0, len(tokens) - width + 1))
    })


def build_word15_rule(features: list[tuple[str, ...]], target: np.ndarray, train: np.ndarray) -> dict[str, int]:
    counts: dict[str, Counter[int]] = defaultdict(Counter)
    for index in train:
        for feature in features[int(index)]:
            counts[feature][int(target[index])] += 1
    return {
        key: next(iter(labels))
        for key, labels in counts.items()
        if sum(labels.values()) >= 15 and len(labels) == 1
    }


def decide_unanimous(features: tuple[str, ...], rule: dict[str, int]) -> int:
    labels = {rule[value] for value in features if value in rule}
    return next(iter(labels)) if len(labels) == 1 else -1


def geometric(left: np.ndarray, right: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    value = (alpha[:, None] * np.log(np.maximum(left, 1.0e-12))
             + (1.0 - alpha[:, None]) * np.log(np.maximum(right, 1.0e-12)))
    value -= value.max(axis=1, keepdims=True)
    probability = np.exp(value)
    return (probability / probability.sum(axis=1, keepdims=True)).astype(np.float32)


def macro(target: np.ndarray, prediction: np.ndarray) -> float:
    return float(f1_score(target, prediction, labels=np.arange(len(CLASSES)), average="macro", zero_division=0))


def combine(base_probability: np.ndarray, state_prediction: np.ndarray, prompt_prediction: np.ndarray) -> np.ndarray:
    """State correction first; approved lexical terminal rule retains priority."""
    prediction = base_probability.argmax(axis=1).astype(np.int16)
    state_mask = state_prediction >= 0
    prediction[state_mask] = state_prediction[state_mask]
    prompt_mask = prompt_prediction >= 0
    prediction[prompt_mask] = prompt_prediction[prompt_mask]
    return prediction


def record(target: np.ndarray, baseline: np.ndarray, candidate: np.ndarray) -> dict[str, float | int]:
    return {
        "terminal_macro_f1": macro(target, candidate),
        "changed": int(np.sum(candidate != baseline)),
        "fixed": int(np.sum((candidate == target) & (baseline != target))),
        "harmed": int(np.sum((candidate != target) & (baseline == target))),
    }


def oof_state_predictions(rows: list[dict], target: np.ndarray, terminal: np.ndarray,
                          groups: np.ndarray, raw_prediction: np.ndarray,
                          support: int, purity: float) -> tuple[np.ndarray, list[dict[str, int]]]:
    prediction = np.full(len(rows), -1, dtype=np.int16)
    audit: list[dict[str, int]] = []
    folds = GroupKFold(n_splits=5).split(np.zeros(len(rows)), target, groups)
    for number, (train, valid) in enumerate(folds, 1):
        train_terminal = train[terminal[train]]
        valid_terminal = valid[terminal[valid]]
        rule = build_state_rule(rows, target, train_terminal, raw_prediction, support, purity)
        prediction[valid_terminal] = decide_state(rows, valid_terminal, raw_prediction, rule)[valid_terminal]
        audit.append({"fold": number, "train_terminal": int(len(train_terminal)), "valid_terminal": int(len(valid_terminal)), "rule_count": int(len(rule))})
    return prediction, audit


def word15_oof(rows: list[dict], target: np.ndarray, terminal: np.ndarray, groups: np.ndarray,
               features: list[tuple[str, ...]]) -> tuple[np.ndarray, list[dict[str, int]]]:
    prediction = np.full(len(rows), -1, dtype=np.int16)
    audit: list[dict[str, int]] = []
    folds = GroupKFold(n_splits=5).split(np.zeros(len(rows)), target, groups)
    for number, (train, valid) in enumerate(folds, 1):
        rule = build_word15_rule(features, target, train)
        valid_terminal = valid[terminal[valid]]
        prediction[valid_terminal] = np.asarray(
            [decide_unanimous(features[int(index)], rule) for index in valid_terminal], dtype=np.int16
        )
        audit.append({"fold": number, "train_rows": int(len(train)), "valid_terminal": int(len(valid_terminal)), "rule_count": int(len(rule))})
    return prediction, audit


def nested_selection(target: np.ndarray, terminal: np.ndarray, groups: np.ndarray,
                     base_probability: np.ndarray, prompt: np.ndarray,
                     states: dict[str, np.ndarray]) -> tuple[np.ndarray, list[dict[str, object]]]:
    """Choose residual policy on the other four session folds only."""
    folds = list(GroupKFold(n_splits=5).split(np.zeros(len(target)), target, groups))
    baseline = combine(base_probability, np.full(len(target), -1, dtype=np.int16), prompt)
    output = baseline.copy()
    records: list[dict[str, object]] = []
    for number, (_, valid) in enumerate(folds, 1):
        fit = np.ones(len(target), dtype=bool); fit[valid] = False; fit &= terminal
        choices = [("none", np.full(len(target), -1, dtype=np.int16)), *states.items()]
        scored = []
        for name, state in choices:
            prediction = combine(base_probability, state, prompt)
            scored.append((macro(target[fit], prediction[fit]), name, prediction))
        # Prefer no intervention under an exact tie.
        best = max(scored, key=lambda value: (value[0], value[1] == "none"))
        output[valid] = best[2][valid]
        records.append({"fold": number, "chosen": best[1], "inner_terminal_macro_f1": best[0]})
    return output, records


def main() -> None:
    rows = [json.loads(line) for line in (ROOT / "data/train.jsonl").open(encoding="utf-8") if line.strip()]
    labels = {row["id"]: row["action"] for row in csv.DictReader((ROOT / "data/train_labels.csv").open(encoding="utf-8"))}
    split = {row["id"]: row["split"] for row in csv.DictReader((ROOT / "splits/common_split.csv").open(encoding="utf-8"))}
    target = np.asarray([TO_ID[labels[str(row["id"])]] for row in rows], dtype=np.int16)
    groups = np.asarray([session_of(row) for row in rows])
    sources = np.asarray([source_of(row) for row in rows])
    terminal = terminal_mask(rows)
    common_train = np.asarray([index for index, row in enumerate(rows) if split[str(row["id"])] == "train"], dtype=int)
    fixed = np.asarray([index for index, row in enumerate(rows) if split[str(row["id"])] == "val"], dtype=int)
    fixed_terminal = terminal[fixed]

    # B42_9 exact OOF components and strict-current components.
    raw_full_old = np.load(ROOT / "experiments/modeling_ak_live_xgb_proxy/oof_probabilities.npy").astype(np.float32)
    raw_full_strict = np.load(ROOT / "experiments/xgb_robust_ensemble/full_oof_raw_seed42.npy").astype(np.float32)
    old_gru_full = np.load(ROOT / "experiments/modeling_ai_sequence_fusion_noe5/oof_probabilities.npy").astype(np.float32)
    strict_gru_full = np.load(ROOT / "experiments/modeling_ai_sequence_nested_distill_e16_i4/oof_probabilities.npy").astype(np.float32)
    if raw_full_old.shape != raw_full_strict.shape or raw_full_old.shape != (len(rows), len(CLASSES)):
        raise ValueError("Full raw selector probability shapes do not align")
    old_full = geometric(raw_full_old, old_gru_full, np.where(sources == "au", .85, .20).astype(np.float32))
    strict_full = geometric(raw_full_strict, strict_gru_full, np.where(sources == "au", .85, .25).astype(np.float32))

    # B42_9's historical matching fixed side is retained for a public-best
    # audit.  Strict uses the matching 2032 feature/seed42 fixed selector.
    old_raw_fixed = np.load(ROOT / "experiments/modeling_ac_tune2/oof_probabilities.npy").astype(np.float32)
    old_gru_fixed = np.load(ROOT / "experiments/modeling_ai_sequence_fusion_fixed_noe5/valid_probabilities.npy").astype(np.float32)
    strict_raw_fixed = np.load(ROOT / "experiments/xgb_robust_ensemble/strict_fixed_raw_seed42.npy").astype(np.float32)
    strict_gru_fixed = np.load(ROOT / "experiments/modeling_ai_sequence_fusion_fixed_strictdistill_w35/valid_probabilities.npy").astype(np.float32)
    if any(value.shape != (len(fixed), len(CLASSES)) for value in (old_raw_fixed, old_gru_fixed, strict_raw_fixed, strict_gru_fixed)):
        raise ValueError("Fixed probability components do not align with common validation")
    old_fixed = geometric(old_raw_fixed, old_gru_fixed, np.where(sources[fixed] == "au", .85, .20).astype(np.float32))
    strict_fixed = geometric(strict_raw_fixed, strict_gru_fixed, np.where(sources[fixed] == "au", .85, .25).astype(np.float32))

    # Strict common-train raw selector OOF is necessary for a leakage-safe
    # table fit.  It provides an independent selector output for every
    # common-train session and never reads common-validation labels.
    common_raw = np.load(ROOT / "experiments/strict_blend_calibration/common_train_raw_xgb_oof.npy").astype(np.float32)
    common_ids = np.load(ROOT / "experiments/strict_blend_calibration/common_train_ids.npy").astype(str)
    if not np.array_equal(common_ids, np.asarray([str(rows[index]["id"]) for index in common_train])):
        raise ValueError("Common-train raw selector OOF IDs do not align")

    print("prompt_word15_features", flush=True)
    features = [prompt_features(row) for row in rows]
    print("prompt_word15_oof", flush=True)
    word15_full, word15_audit = word15_oof(rows, target, terminal, groups, features)
    word15_rule = build_word15_rule(features, target, common_train)
    word15_fixed = np.full(len(fixed), -1, dtype=np.int16)
    word15_fixed[fixed_terminal] = np.asarray(
        [decide_unanimous(features[int(index)], word15_rule) for index in fixed[fixed_terminal]], dtype=np.int16
    )

    # Existing char20 decisions were generated in the same session folds and
    # are independent of raw/GRU probability sources.
    char = np.load(ROOT / "experiments/terminal_rule_combinations/oof_candidates.npz")
    if not np.array_equal(char["y"], target) or not np.array_equal(char["terminal"], terminal):
        raise ValueError("char20 OOF alignment failed")
    char_full = char["char20"].astype(np.int16)
    fixed_term_ids = np.asarray([str(rows[index]["id"]) for index in fixed[fixed_terminal]])
    if not np.array_equal(char["valid_ids"].astype(str), fixed_term_ids):
        raise ValueError("char20 fixed ID alignment failed")
    char_fixed = np.full(len(fixed), -1, dtype=np.int16)
    char_fixed[fixed_terminal] = char["fixed_char20"].astype(np.int16)

    # Candidate family is deliberately small and state-only.  The purity=1
    # member is the high-confidence residual found by the screen; all other
    # members make nested selection prove whether it is stable.
    configs = ((5, 1.0), (10, 1.0), (15, 1.0), (20, 1.0), (10, .95), (20, .95))
    state_full: dict[str, np.ndarray] = {}
    state_audit: dict[str, object] = {}
    raw_full_prediction = raw_full_old.argmax(axis=1)
    for support, purity in configs:
        name = f"support{support}_purity{str(purity).replace('.', 'p')}"
        print(f"state_oof={name}", flush=True)
        value, audit = oof_state_predictions(rows, target, terminal, groups, raw_full_prediction, support, purity)
        state_full[name] = value
        state_audit[name] = audit

    # Common-train rule fit uses only terminal common-train sessions and their
    # inner-OOF raw selector class.  Query keys intentionally use each fixed
    # path's actual raw selector output, exposing key drift rather than hiding
    # it behind in-sample predictions.
    raw_common_prediction = common_raw.argmax(axis=1)
    terminal_common = common_train[terminal[common_train]]
    state_fixed_old: dict[str, np.ndarray] = {}
    state_fixed_strict: dict[str, np.ndarray] = {}
    full_rule_sizes: dict[str, int] = {}
    for support, purity in configs:
        name = f"support{support}_purity{str(purity).replace('.', 'p')}"
        # Reindex common raw predictions to the global rows expected by the
        # generic helper without ever assigning a value to fixed rows.
        global_common_prediction = np.full(len(rows), -1, dtype=np.int16)
        global_common_prediction[common_train] = raw_common_prediction.astype(np.int16)
        rule = build_state_rule(rows, target, terminal_common, global_common_prediction, support, purity)
        full_rule_sizes[name] = len(rule)
        old_query = np.full(len(rows), -1, dtype=np.int16); old_query[fixed] = old_raw_fixed.argmax(axis=1).astype(np.int16)
        strict_query = np.full(len(rows), -1, dtype=np.int16); strict_query[fixed] = strict_raw_fixed.argmax(axis=1).astype(np.int16)
        state_fixed_old[name] = decide_state(rows, fixed[fixed_terminal], old_query, rule)[fixed]
        state_fixed_strict[name] = decide_state(rows, fixed[fixed_terminal], strict_query, rule)[fixed]

    terminal_index = np.flatnonzero(terminal)
    old_base_full = combine(old_full, np.full(len(rows), -1, dtype=np.int16), word15_full)
    strict_base_full = combine(strict_full, np.full(len(rows), -1, dtype=np.int16), char_full)
    old_base_fixed = combine(old_fixed, np.full(len(fixed), -1, dtype=np.int16), word15_fixed)
    strict_base_fixed = combine(strict_fixed, np.full(len(fixed), -1, dtype=np.int16), char_fixed)
    report: dict[str, object] = {
        "scope": "Experiment only: model/, script.py, and every ZIP archive are unchanged.",
        "state_contract": "source + raw-selector argmax + latest visible assistant action + latest visible result status; terminal-only label-pure correction; current prompt is not a state-rule feature.",
        "fixed_contract": "common-train terminal state rules use five-fold common-train raw-selector OOF; fixed rows use selector predictions from a model trained without their labels.",
        "counts": {"rows": len(rows), "terminal": int(terminal.sum()), "common_train_terminal": int(terminal_common.size), "fixed_terminal": int(fixed_terminal.sum())},
        "word15_audit": word15_audit,
        "state_oof_audit": state_audit,
        "state_common_rule_sizes": full_rule_sizes,
        "baseline": {
            "b42_9_old_word15": {
                "full": record(target[terminal_index], old_base_full[terminal_index], old_base_full[terminal_index]),
                "fixed": record(target[fixed][fixed_terminal], old_base_fixed[fixed_terminal], old_base_fixed[fixed_terminal]),
            },
            "strict_char20": {
                "full": record(target[terminal_index], strict_base_full[terminal_index], strict_base_full[terminal_index]),
                "fixed": record(target[fixed][fixed_terminal], strict_base_fixed[fixed_terminal], strict_base_fixed[fixed_terminal]),
            },
        },
        "configs": {},
    }
    for name, state in state_full.items():
        old_candidate_full = combine(old_full, state, word15_full)
        strict_candidate_full = combine(strict_full, state, char_full)
        old_candidate_fixed = combine(old_fixed, state_fixed_old[name], word15_fixed)
        strict_candidate_fixed = combine(strict_fixed, state_fixed_strict[name], char_fixed)
        report["configs"][name] = {
            "b42_9_old_word15": {
                "full": record(target[terminal_index], old_base_full[terminal_index], old_candidate_full[terminal_index]),
                "fixed": record(target[fixed][fixed_terminal], old_base_fixed[fixed_terminal], old_candidate_fixed[fixed_terminal]),
            },
            "strict_char20": {
                "full": record(target[terminal_index], strict_base_full[terminal_index], strict_candidate_full[terminal_index]),
                "fixed": record(target[fixed][fixed_terminal], strict_base_fixed[fixed_terminal], strict_candidate_fixed[fixed_terminal]),
            },
        }

    old_nested, old_choices = nested_selection(target, terminal, groups, old_full, word15_full, state_full)
    strict_nested, strict_choices = nested_selection(target, terminal, groups, strict_full, char_full, state_full)
    report["nested_selection"] = {
        "b42_9_old_word15": {"full": record(target[terminal_index], old_base_full[terminal_index], old_nested[terminal_index]), "choices": old_choices},
        "strict_char20": {"full": record(target[terminal_index], strict_base_full[terminal_index], strict_nested[terminal_index]), "choices": strict_choices},
    }
    out = ROOT / "experiments/state_residual_gate"
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out / "predictions.npz", target=target, terminal=terminal, fixed_index=fixed,
        word15_full=word15_full, word15_fixed=word15_fixed, char_full=char_full, char_fixed=char_fixed,
        old_base_full=old_base_full, strict_base_full=strict_base_full,
        old_base_fixed=old_base_fixed, strict_base_fixed=strict_base_fixed,
        **{f"state_full_{name}": value for name, value in state_full.items()},
        **{f"state_fixed_old_{name}": value for name, value in state_fixed_old.items()},
        **{f"state_fixed_strict_{name}": value for name, value in state_fixed_strict.items()},
    )
    (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
