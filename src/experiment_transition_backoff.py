#!/usr/bin/env python3
"""Strict audit of a portable prior-action Dirichlet-backoff model.

This is deliberately an *experiment-only* utility.  It does not import or
write ``model/``, ``script.py`` or either submission ZIP.  The candidate sees
only fields which already exist before the action being predicted:

* prior ``assistant_action`` names (last one through last four),
* the final prior action's coarse result bucket,
* source, prior-action/history-length bucket, and last-CI-status bucket.

There is no prompt text, no sample ID lookup, no later row, and no
future-history recovery feature.  Counts are learned with an explicitly
hierarchical Dirichlet posterior.  The only fusion tests are predeclared,
small geometric perturbations of the strict raw-XGB + strict-GRU fallback and
one high-confidence correction gate.  Char20 remains the final terminal
override in every metric.
"""

from __future__ import annotations

import csv
import json
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold


CLASSES = [
    "read_file", "grep_search", "list_directory", "glob_pattern", "edit_file", "write_file", "apply_patch",
    "run_bash", "run_tests", "lint_or_typecheck", "ask_user", "plan_task", "web_search", "respond_only",
]
LABEL_TO_ID = {value: index for index, value in enumerate(CLASSES)}
N_CLASSES = len(CLASSES)
EPS = 1.0e-12


def session_of(sample_id: str) -> str:
    return sample_id.rsplit("-step_", 1)[0]


def source_of(sample_id: str) -> str:
    return "au" if sample_id.startswith("sess_au_") else "sim"


def terminal_mask(ids: Iterable[str]) -> np.ndarray:
    ids = list(ids)
    maxima: dict[str, int] = {}
    parsed: list[tuple[str, int]] = []
    for sample_id in ids:
        session, raw_step = sample_id.rsplit("-step_", 1)
        step = int(raw_step)
        maxima[session] = max(maxima.get(session, -1), step)
        parsed.append((session, step))
    return np.asarray([step == maxima[session] for session, step in parsed], dtype=bool)


def macro(y: np.ndarray, pred: np.ndarray, mask: np.ndarray | None = None) -> float:
    if mask is None:
        mask = np.ones(len(y), dtype=bool)
    return float(f1_score(y[mask], pred[mask], labels=np.arange(N_CLASSES), average="macro", zero_division=0))


def action_count_bucket(value: int) -> str:
    if value == 0:
        return "0"
    if value <= 3:
        return str(value)
    if value <= 5:
        return "4_5"
    if value <= 8:
        return "6_8"
    return "9_plus"


def result_bucket(value: object) -> str:
    """Stable, deliberately coarse status bucket for the latest prior result."""
    text = str(value or "").casefold()
    if re.search(r"fail|error|exception|denied|timeout|not found|no matches|0 matches", text):
        return "bad"
    if re.search(r"pass|ok|success|green|patched|applied|modified|read |listed|found |matched", text):
        return "ok"
    return "other"


def contexts(row: dict) -> tuple[tuple[str, ...], ...]:
    """Return parent-to-child contexts, all available at action time."""
    sample_id = str(row["id"])
    source = source_of(sample_id)
    actions = [
        str(item.get("name", "<unknown>"))
        for item in row.get("history", [])
        if isinstance(item, dict) and item.get("role") == "assistant_action"
    ]
    latest = actions[-1] if actions else "<bos>"
    history = row.get("history", [])
    last_action = next(
        (
            item for item in reversed(history)
            if isinstance(item, dict) and item.get("role") == "assistant_action"
        ),
        {},
    )
    latest_result = result_bucket(last_action.get("result_summary"))
    meta = row.get("session_meta") if isinstance(row.get("session_meta"), dict) else {}
    workspace = meta.get("workspace") if isinstance(meta.get("workspace"), dict) else {}
    ci_status = str(workspace.get("last_ci_status", "none")).casefold()
    turn = action_count_bucket(len(actions))
    tail2 = tuple(actions[-2:]) if len(actions) >= 2 else ("<bos>", latest)
    tail3 = tuple(actions[-3:]) if len(actions) >= 3 else ("<bos>",) * (3 - len(actions)) + tuple(actions)
    tail4 = tuple(actions[-4:]) if len(actions) >= 4 else ("<bos>",) * (4 - len(actions)) + tuple(actions)
    # Each level is a refinement of a preceding safe, lower-variance view.
    return (
        ("global",),
        ("source", source),
        ("source_turn_ci", source, turn, ci_status),
        ("last_action", source, latest),
        ("last_action_result", source, latest, latest_result),
        ("last_two", source, *tail2),
        ("last_three", source, *tail3),
        ("last_four", source, *tail4),
    )


class DirichletBackoff:
    """Small count model whose child posteriors shrink to parent posteriors."""

    # Fixed prior strength per child level.  Longer exact transitions are
    # strongly shrunk so a single session pattern cannot influence a row.
    SHRINK = (0.0, 220.0, 160.0, 110.0, 90.0, 70.0, 55.0, 45.0)
    MIN_SUPPORT = (1, 1, 8, 12, 12, 16, 22, 30)

    def __init__(self) -> None:
        self.tables: list[dict[tuple[str, ...], np.ndarray]] = []
        self.raw_contexts = 0

    def fit(self, rows: list[dict], y: np.ndarray, indices: np.ndarray) -> "DirichletBackoff":
        provisional: list[dict[tuple[str, ...], np.ndarray]] = [defaultdict(lambda: np.zeros(N_CLASSES, dtype=np.float64)) for _ in self.SHRINK]
        for index in indices:
            label = int(y[int(index)])
            for level, key in enumerate(contexts(rows[int(index)])):
                provisional[level][key][label] += 1.0
        self.raw_contexts = int(sum(len(table) for table in provisional))
        self.tables = []
        for level, table in enumerate(provisional):
            keep = {
                key: value for key, value in table.items()
                if value.sum() >= self.MIN_SUPPORT[level]
            }
            self.tables.append(keep)
        return self

    def predict(self, rows: list[dict], indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if not self.tables:
            raise RuntimeError("fit must be called before predict")
        result = np.zeros((len(indices), N_CLASSES), dtype=np.float32)
        deepest_support = np.zeros(len(indices), dtype=np.int32)
        global_counts = self.tables[0][("global",)]
        # A symmetric two-count prior prevents zero likelihoods, while all
        # subsequent contexts are genuine Dirichlet posteriors around parent.
        base = (global_counts + 2.0) / (global_counts.sum() + 2.0 * N_CLASSES)
        for row_number, index in enumerate(indices):
            posterior = base.copy()
            support = int(global_counts.sum())
            for level, key in enumerate(contexts(rows[int(index)])[1:], start=1):
                counts = self.tables[level].get(key)
                if counts is None:
                    continue
                total = counts.sum()
                posterior = (counts + self.SHRINK[level] * posterior) / (total + self.SHRINK[level])
                support = int(total)
            result[row_number] = posterior
            deepest_support[row_number] = support
        return result, deepest_support

    def compact_stats(self) -> dict[str, object]:
        return {
            "raw_contexts": self.raw_contexts,
            "retained_contexts_by_level": [len(table) for table in self.tables],
            "min_support_by_level": list(self.MIN_SUPPORT),
            "dirichlet_shrink_by_level": list(self.SHRINK),
        }


def last3_exact_context(row: dict) -> tuple[str, str, str, str]:
    """The pre-registered compact transition context used in the follow-up.

    It is intentionally *only* source plus exactly three previous action
    names.  Missing prefixes are left-padded with the literal ``"none"``.
    """
    actions = [
        str(item.get("name", "none"))
        for item in row.get("history", [])
        if isinstance(item, dict) and item.get("role") == "assistant_action"
    ]
    tail = (["none"] * max(0, 3 - len(actions)) + actions[-3:])[-3:]
    return (source_of(str(row["id"])), *tail)


class ExactLast3Transition:
    """Source-prior Dirichlet model specified for the compact reproduction.

    A context retained at support ``n>=10`` gets
    ``(counts + 2 * source_prior) / (n + 2)``.  ``source_prior`` itself is
    estimated from all fold-training rows with one count per class.  Nothing
    from later session rows is ever read.
    """

    def __init__(self) -> None:
        self.source_prior: dict[str, np.ndarray] = {}
        self.contexts: dict[tuple[str, str, str, str], np.ndarray] = {}
        self.raw_context_count = 0

    def fit(self, rows: list[dict], y: np.ndarray, indices: np.ndarray) -> "ExactLast3Transition":
        prior_counts = {"sim": np.ones(N_CLASSES, dtype=np.float64), "au": np.ones(N_CLASSES, dtype=np.float64)}
        provisional: dict[tuple[str, str, str, str], np.ndarray] = defaultdict(lambda: np.zeros(N_CLASSES, dtype=np.float64))
        for index in indices:
            row = rows[int(index)]
            source = source_of(str(row["id"]))
            label = int(y[int(index)])
            prior_counts[source][label] += 1.0
            provisional[last3_exact_context(row)][label] += 1.0
        self.source_prior = {source: counts / counts.sum() for source, counts in prior_counts.items()}
        self.raw_context_count = len(provisional)
        self.contexts = {key: value for key, value in provisional.items() if value.sum() >= 10.0}
        return self

    def predict(self, rows: list[dict], indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        probability = np.zeros((len(indices), N_CLASSES), dtype=np.float32)
        support = np.zeros(len(indices), dtype=np.int32)
        for position, index in enumerate(indices):
            key = last3_exact_context(rows[int(index)])
            prior = self.source_prior[key[0]]
            counts = self.contexts.get(key)
            if counts is None:
                probability[position] = prior
                continue
            total = counts.sum()
            probability[position] = (counts + 2.0 * prior) / (total + 2.0)
            support[position] = int(total)
        return probability, support

    def compact_stats(self) -> dict[str, int]:
        return {"raw_contexts": int(self.raw_context_count), "retained_contexts_support_ge_10": int(len(self.contexts))}


def geometric(left: np.ndarray, right: np.ndarray, left_weight: np.ndarray | float) -> np.ndarray:
    weight = np.asarray(left_weight, dtype=np.float64)
    if weight.ndim == 1:
        weight = weight[:, None]
    logits = weight * np.log(np.maximum(left, EPS)) + (1.0 - weight) * np.log(np.maximum(right, EPS))
    logits -= logits.max(axis=1, keepdims=True)
    probability = np.exp(logits)
    return (probability / probability.sum(axis=1, keepdims=True)).astype(np.float32)


def strict_base(raw_xgb: np.ndarray, strict_gru: np.ndarray, sources: np.ndarray) -> np.ndarray:
    # This task was specifically requested against the current sim=.25 /
    # au=.85 strict fallback candidate, not the earlier sim=.20 variant.
    alpha = np.where(sources == "sim", 0.25, 0.85).astype(np.float64)
    return geometric(raw_xgb, strict_gru, alpha)


def apply_char(probability: np.ndarray, terminal: np.ndarray, char_prediction: np.ndarray) -> np.ndarray:
    pred = probability.argmax(axis=1).astype(np.int16)
    use = terminal & (char_prediction >= 0)
    pred[use] = char_prediction[use]
    return pred


def metrics(y: np.ndarray, terminal: np.ndarray, probability: np.ndarray, char_prediction: np.ndarray) -> dict[str, float | int]:
    fallback = probability.argmax(axis=1).astype(np.int16)
    final = apply_char(probability, terminal, char_prediction)
    hybrid = np.where(terminal, final, y)
    return {
        "all_fallback_macro_f1": macro(y, fallback),
        "terminal_fallback_macro_f1": macro(y, fallback, terminal),
        "terminal_with_char20_macro_f1": macro(y, final, terminal),
        "hybrid_future_history_plus_char20_macro_f1": macro(y, hybrid),
        "char20_overrides": int(np.sum(terminal & (char_prediction >= 0))),
    }


def candidate_probability(name: str, base: np.ndarray, markov: np.ndarray, support: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The tiny, predeclared candidate family—intentionally no grid search."""
    if name == "mild_geometric_0p97":
        probability = geometric(base, markov, 0.97)
        return probability, np.ones(len(base), dtype=bool)
    if name == "mild_geometric_0p93":
        probability = geometric(base, markov, 0.93)
        return probability, np.ones(len(base), dtype=bool)
    if name == "high_confidence_transition_gate":
        # A 14-way transition distribution must be markedly concentrated, be
        # learned from substantial support, and only question a low-confidence
        # base decision.  It is a correction blend rather than a hard override.
        use = (markov.max(axis=1) >= 0.55) & (support >= 100) & (base.max(axis=1) <= 0.60)
        probability = base.copy()
        if np.any(use):
            probability[use] = geometric(base[use], markov[use], 0.80)
        return probability, use
    raise ValueError(name)


def exact_last3_candidate(base: np.ndarray, markov: np.ndarray, support: np.ndarray,
                          terminal: np.ndarray, base_weight: float) -> tuple[np.ndarray, np.ndarray]:
    """Apply the specified compact transition rule only on terminal rows."""
    gate = terminal & (support >= 10) & (markov.max(axis=1) >= 0.45)
    probability = base.copy()
    if np.any(gate):
        probability[gate] = geometric(base[gate], markov[gate], base_weight)
    return probability, gate


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    out = root / "experiments" / "transition_backoff_strict"
    out.mkdir(parents=True, exist_ok=True)
    started = time.time()

    rows = [json.loads(line) for line in (root / "data" / "train.jsonl").open(encoding="utf-8") if line.strip()]
    labels = {row["id"]: row["action"] for row in csv.DictReader((root / "data" / "train_labels.csv").open(encoding="utf-8"))}
    y = np.asarray([LABEL_TO_ID[labels[str(row["id"])]] for row in rows], dtype=np.int64)
    ids = np.asarray([str(row["id"]) for row in rows])
    groups = np.asarray([session_of(value) for value in ids])
    sources = np.asarray([source_of(value) for value in ids])
    terminal = terminal_mask(ids)
    split = {row["id"]: row["split"] for row in csv.DictReader((root / "splits" / "common_split.csv").open(encoding="utf-8"))}
    common_train = np.asarray([number for number, value in enumerate(ids) if split[value] == "train"], dtype=int)
    fixed_valid = np.asarray([number for number, value in enumerate(ids) if split[value] == "val"], dtype=int)

    # Full strict base OOF: raw 2032-XGB plus independently nested-distilled
    # portable GRU.  Assert fold identity so no stale probability table slips in.
    raw_full = np.load(root / "experiments" / "modeling_ak_live_xgb_proxy" / "oof_probabilities.npy").astype(np.float32)
    gru_full = np.load(root / "experiments" / "modeling_ai_sequence_nested_distill_e16_i4" / "oof_probabilities.npy").astype(np.float32)
    saved_outer = np.load(root / "experiments" / "modeling_ai_sequence_nested_distill_e16_i4" / "outer_fold_assignment.npy")
    folds = list(GroupKFold(n_splits=5).split(np.zeros(len(rows)), y, groups))
    fold_id = np.empty(len(rows), dtype=np.int8)
    for number, (_, valid) in enumerate(folds):
        fold_id[valid] = number
    if not np.array_equal(saved_outer.astype(np.int8), fold_id):
        raise ValueError("Strict GRU outer-fold assignment does not match the Markov session folds")
    if raw_full.shape != gru_full.shape or raw_full.shape != (len(rows), N_CLASSES):
        raise ValueError("Full strict base probability shape mismatch")
    base_full = strict_base(raw_full, gru_full, sources)

    # The char20 predictions are labels-only rule OOFs, independent of the
    # base probabilities.  The stored artifact saves runtime and lets us prove
    # its fold/ID alignment before using it as the final override.
    char_artifact = np.load(root / "experiments" / "terminal_rule_combinations" / "oof_candidates.npz", allow_pickle=False)
    if not np.array_equal(char_artifact["y"].astype(np.int64), y) or not np.array_equal(char_artifact["terminal"].astype(bool), terminal):
        raise ValueError("Char20 OOF target/terminal alignment failed")
    char_full = char_artifact["char20"].astype(np.int16)
    char_fold = char_artifact["fold"].astype(np.int8)
    if not np.array_equal(char_fold[terminal], fold_id[terminal]):
        raise ValueError("Char20 OOF fold assignment does not match strict GRU")
    fixed_terminal = terminal[fixed_valid]
    fixed_terminal_ids = ids[fixed_valid][fixed_terminal]
    if not np.array_equal(char_artifact["valid_ids"].astype(str), fixed_terminal_ids):
        raise ValueError("Char20 fixed IDs do not match common fixed terminal IDs")
    # The archived fixed vector stores terminal rows only; expand it to the
    # full 15,021 fixed rows so the final-only terminal contract is explicit.
    char_fixed = np.full(len(fixed_valid), -1, dtype=np.int16)
    char_fixed[fixed_terminal] = char_artifact["fixed_char20"].astype(np.int16)

    # Seed 42 is the deployment-matching strict fixed selector.  The older
    # AU audit's seed-808 probability remains useful only as a caveat and is
    # deliberately not used for the primary promotion decision.
    fixed_raw = np.load(root / "experiments" / "xgb_robust_ensemble" / "strict_fixed_raw_seed42.npy").astype(np.float32)
    fixed_gru = np.load(root / "experiments" / "modeling_ai_sequence_fusion_fixed_strictdistill_w35" / "valid_probabilities.npy").astype(np.float32)
    fixed_gru_indices = np.load(root / "experiments" / "modeling_ai_sequence_fusion_fixed_strictdistill_w35" / "valid_indices.npy")
    if not np.array_equal(fixed_gru_indices.astype(int), fixed_valid):
        raise ValueError("Strict GRU fixed validation indices do not match common split")
    if fixed_raw.shape != fixed_gru.shape or fixed_raw.shape != (len(fixed_valid), N_CLASSES):
        raise ValueError("Strict fixed 2032 raw-XGB / GRU probability mismatch")
    base_fixed = strict_base(fixed_raw, fixed_gru, sources[fixed_valid])

    markov_full = np.zeros((len(rows), N_CLASSES), dtype=np.float32)
    support_full = np.zeros(len(rows), dtype=np.int32)
    fold_stats: list[dict[str, object]] = []
    for number, (train, valid) in enumerate(folds, start=1):
        model = DirichletBackoff().fit(rows, y, train)
        probability, support = model.predict(rows, valid)
        markov_full[valid] = probability
        support_full[valid] = support
        fold_stats.append({"fold": number, "train_rows": int(len(train)), "valid_rows": int(len(valid)), **model.compact_stats()})
        print(json.dumps({"fold": number, "status": "done", **model.compact_stats()}, ensure_ascii=False), flush=True)

    fixed_model = DirichletBackoff().fit(rows, y, common_train)
    markov_fixed, support_fixed = fixed_model.predict(rows, fixed_valid)

    candidates = ("mild_geometric_0p97", "mild_geometric_0p93", "high_confidence_transition_gate")
    report: dict[str, object] = {
        "scope": "Experiment only: model/, script.py, baseline_submit.zip, and B42.zip were not modified.",
        "protocol": {
            "safe_inputs": "Prior assistant_action names (last 1-4), final prior result bucket, source, history-length bucket, and session last-CI-status only.",
            "forbidden_inputs": "No current prompt, target/ID lookup, later session row, future-history recovery, or test/train overlap lookup.",
            "base": "strict nested full GroupKFold raw-XGB plus strict no-E5 GRU geometric fallback; alpha XGB sim=.25 and au=.85.",
            "fixed": "matching deployed 2032-column strict common-train raw selector (seed 42) plus strict distill=.35 GRU validation probabilities; labels in common validation excluded from every Markov count. The legacy seed-808 AU audit is not used for promotion.",
            "char20": "Precomputed rule OOF is alignment-asserted and applied only after each Markov/base fallback on terminal rows.",
            "selection": "No score-tuned grid. Candidate weights/gate were predeclared in source: .97 / .93 all-row geometric perturbations and one support+confidence correction gate.",
        },
        "counts": {
            "rows": int(len(rows)), "terminal": int(terminal.sum()),
            "fixed_rows": int(len(fixed_valid)), "fixed_terminal": int(fixed_terminal.sum()),
        },
        "backoff": {
            "full_fold_stats": fold_stats,
            "full_markov_all_macro_f1": macro(y, markov_full.argmax(axis=1)),
            "full_markov_terminal_macro_f1": macro(y, markov_full.argmax(axis=1), terminal),
            "fixed_markov_all_macro_f1": macro(y[fixed_valid], markov_fixed.argmax(axis=1)),
            "fixed_markov_terminal_macro_f1": macro(y[fixed_valid], markov_fixed.argmax(axis=1), fixed_terminal),
            "fixed_full_refit_stats": fixed_model.compact_stats(),
            "full_markov_confidence_quantiles": [float(value) for value in np.quantile(markov_full.max(axis=1), (0.5, 0.9, 0.95, 0.99, 1.0))],
            "fixed_markov_confidence_quantiles": [float(value) for value in np.quantile(markov_fixed.max(axis=1), (0.5, 0.9, 0.95, 0.99, 1.0))],
        },
        "baseline": {
            "full": metrics(y, terminal, base_full, char_full),
            "fixed": metrics(y[fixed_valid], fixed_terminal, base_fixed, char_fixed),
        },
        "candidates": {},
    }
    arrays: dict[str, np.ndarray] = {
        "fold": fold_id, "terminal": terminal, "y": y, "base_full": base_full, "markov_full": markov_full,
        "support_full": support_full, "base_fixed": base_fixed, "markov_fixed": markov_fixed, "support_fixed": support_fixed,
        "char_full": char_full, "char_fixed": char_fixed,
    }
    baseline_full = report["baseline"]["full"]  # type: ignore[index]
    baseline_fixed = report["baseline"]["fixed"]  # type: ignore[index]
    full_candidate_probability: dict[str, np.ndarray] = {}
    fixed_candidate_probability: dict[str, np.ndarray] = {}
    for name in candidates:
        full_probability, full_gate = candidate_probability(name, base_full, markov_full, support_full)
        fixed_probability, fixed_gate = candidate_probability(name, base_fixed, markov_fixed, support_fixed)
        full_candidate_probability[name] = full_probability
        fixed_candidate_probability[name] = fixed_probability
        full_metric = metrics(y, terminal, full_probability, char_full)
        fixed_metric = metrics(y[fixed_valid], fixed_terminal, fixed_probability, char_fixed)
        report["candidates"][name] = {
            "full": full_metric,
            "fixed": fixed_metric,
            "full_transition_gate_rows": int(full_gate.sum()),
            "full_transition_gate_terminal_rows": int((full_gate & terminal).sum()),
            "fixed_transition_gate_rows": int(fixed_gate.sum()),
            "fixed_transition_gate_terminal_rows": int((fixed_gate & fixed_terminal).sum()),
            "delta_vs_baseline": {
                "full_all_fallback": float(full_metric["all_fallback_macro_f1"] - baseline_full["all_fallback_macro_f1"]),
                "full_terminal_with_char20": float(full_metric["terminal_with_char20_macro_f1"] - baseline_full["terminal_with_char20_macro_f1"]),
                "full_hybrid": float(full_metric["hybrid_future_history_plus_char20_macro_f1"] - baseline_full["hybrid_future_history_plus_char20_macro_f1"]),
                "fixed_all_fallback": float(fixed_metric["all_fallback_macro_f1"] - baseline_fixed["all_fallback_macro_f1"]),
                "fixed_terminal_with_char20": float(fixed_metric["terminal_with_char20_macro_f1"] - baseline_fixed["terminal_with_char20_macro_f1"]),
                "fixed_hybrid": float(fixed_metric["hybrid_future_history_plus_char20_macro_f1"] - baseline_fixed["hybrid_future_history_plus_char20_macro_f1"]),
            },
        }
        arrays[f"candidate_{name}_full"] = full_probability
        arrays[f"candidate_{name}_fixed"] = fixed_probability

    # Static OOF is only a screen.  The deployable choice is also audited
    # nestedly: for every held session fold, choose among the three already
    # predeclared conservative candidates using *only the other four folds'*
    # terminal fallback labels.  Char20 is deliberately outside selection and
    # remains the final held-fold override.  On a tie, retain the less invasive
    # configuration (0.97 before 0.93 before the correction gate).
    nested_probability = np.zeros_like(base_full)
    nested_choices: list[dict[str, object]] = []
    preference = {name: index for index, name in enumerate(candidates)}
    for held in range(5):
        select_mask = terminal & (fold_id != held)
        scored = [
            (macro(y, full_candidate_probability[name].argmax(axis=1), select_mask), name)
            for name in candidates
        ]
        score, selected = max(scored, key=lambda pair: (pair[0], -preference[pair[1]]))
        held_mask = fold_id == held
        nested_probability[held_mask] = full_candidate_probability[selected][held_mask]
        nested_choices.append({
            "held_fold": int(held + 1),
            "selected": selected,
            "outer_train_terminal_fallback_macro_f1": float(score),
            "outer_train_scores": {name: float(value) for value, name in scored},
        })
    selected_names = [str(item["selected"]) for item in nested_choices]
    # The fixed validation partition is never used for this vote.  The modal
    # outer selection is a deterministic, conservative deployment proxy; ties
    # go to the lower perturbation according to ``preference``.
    consensus = min(candidates, key=lambda name: (-selected_names.count(name), preference[name]))
    nested_metric = metrics(y, terminal, nested_probability, char_full)
    consensus_fixed_metric = metrics(y[fixed_valid], fixed_terminal, fixed_candidate_probability[consensus], char_fixed)
    report["nested_outer_selection"] = {
        "selection_objective": "terminal fallback macro-F1 before char20, using only the other four session folds",
        "held_fold_choices": nested_choices,
        "full": nested_metric,
        "fixed_consensus_config_without_fixed_tuning": consensus,
        "fixed_consensus": consensus_fixed_metric,
        "delta_vs_baseline": {
            "full_terminal_with_char20": float(nested_metric["terminal_with_char20_macro_f1"] - baseline_full["terminal_with_char20_macro_f1"]),
            "full_hybrid": float(nested_metric["hybrid_future_history_plus_char20_macro_f1"] - baseline_full["hybrid_future_history_plus_char20_macro_f1"]),
            "fixed_terminal_with_char20": float(consensus_fixed_metric["terminal_with_char20_macro_f1"] - baseline_fixed["terminal_with_char20_macro_f1"]),
            "fixed_hybrid": float(consensus_fixed_metric["hybrid_future_history_plus_char20_macro_f1"] - baseline_fixed["hybrid_future_history_plus_char20_macro_f1"]),
        },
    }
    arrays["nested_selected_full"] = nested_probability

    # ------------------------------------------------------------------
    # Exact compact last-3 reproduction requested after the broader
    # action/result backoff screen.  This uses no broad grid: base, 0.88, and
    # a deliberately weaker 0.96 geometric counterpart are the entire family.
    # ------------------------------------------------------------------
    exact_full = np.zeros((len(rows), N_CLASSES), dtype=np.float32)
    exact_support_full = np.zeros(len(rows), dtype=np.int32)
    exact_fold_stats: list[dict[str, object]] = []
    for number, (train, valid) in enumerate(folds, start=1):
        exact_model = ExactLast3Transition().fit(rows, y, train)
        exact_probability, exact_support = exact_model.predict(rows, valid)
        exact_full[valid] = exact_probability
        exact_support_full[valid] = exact_support
        exact_fold_stats.append({"fold": number, "train_rows": int(len(train)), "valid_rows": int(len(valid)), **exact_model.compact_stats()})
    exact_fixed_model = ExactLast3Transition().fit(rows, y, common_train)
    exact_fixed, exact_support_fixed = exact_fixed_model.predict(rows, fixed_valid)
    exact_full_088, exact_gate_full = exact_last3_candidate(
        base_full, exact_full, exact_support_full, terminal, 0.88
    )
    exact_fixed_088, exact_gate_fixed = exact_last3_candidate(
        base_fixed, exact_fixed, exact_support_fixed, fixed_terminal, 0.88
    )
    exact_full_096, exact_gate_full_096 = exact_last3_candidate(
        base_full, exact_full, exact_support_full, terminal, 0.96
    )
    exact_fixed_096, exact_gate_fixed_096 = exact_last3_candidate(
        base_fixed, exact_fixed, exact_support_fixed, fixed_terminal, 0.96
    )
    exact_options_full = {
        "base": base_full,
        "exact_last3_geo_0p88": exact_full_088,
        "exact_last3_geo_0p96": exact_full_096,
    }
    exact_options_fixed = {
        "base": base_fixed,
        "exact_last3_geo_0p88": exact_fixed_088,
        "exact_last3_geo_0p96": exact_fixed_096,
    }
    exact_preference = {"base": 0, "exact_last3_geo_0p96": 1, "exact_last3_geo_0p88": 2}
    exact_baseline_full = metrics(y, terminal, base_full, char_full)
    exact_baseline_fixed = metrics(y[fixed_valid], fixed_terminal, base_fixed, char_fixed)
    exact_static: dict[str, dict[str, object]] = {}
    for name in ("exact_last3_geo_0p88", "exact_last3_geo_0p96"):
        full_metric = metrics(y, terminal, exact_options_full[name], char_full)
        fixed_metric = metrics(y[fixed_valid], fixed_terminal, exact_options_fixed[name], char_fixed)
        exact_static[name] = {
            "full": full_metric,
            "fixed": fixed_metric,
            "full_gate_terminal_rows": int((exact_gate_full if name.endswith("0p88") else exact_gate_full_096).sum()),
            "fixed_gate_terminal_rows": int((exact_gate_fixed if name.endswith("0p88") else exact_gate_fixed_096).sum()),
            "delta_vs_base": {
                "full_all": float(full_metric["all_fallback_macro_f1"] - exact_baseline_full["all_fallback_macro_f1"]),
                "full_terminal_with_char20": float(full_metric["terminal_with_char20_macro_f1"] - exact_baseline_full["terminal_with_char20_macro_f1"]),
                "full_hybrid": float(full_metric["hybrid_future_history_plus_char20_macro_f1"] - exact_baseline_full["hybrid_future_history_plus_char20_macro_f1"]),
                "fixed_all": float(fixed_metric["all_fallback_macro_f1"] - exact_baseline_fixed["all_fallback_macro_f1"]),
                "fixed_terminal_with_char20": float(fixed_metric["terminal_with_char20_macro_f1"] - exact_baseline_fixed["terminal_with_char20_macro_f1"]),
                "fixed_hybrid": float(fixed_metric["hybrid_future_history_plus_char20_macro_f1"] - exact_baseline_fixed["hybrid_future_history_plus_char20_macro_f1"]),
            },
        }

    # Nestedly choose only from the requested base/.88/.96 set.  The char
    # override is applied *after* each candidate before scoring, and the held
    # fold's labels are never used in its choice.
    exact_nested = np.zeros_like(base_full)
    exact_nested_choices: list[dict[str, object]] = []
    for held in range(5):
        choose_mask = terminal & (fold_id != held)
        scored = [
            (macro(y, apply_char(probability, terminal, char_full), choose_mask), name)
            for name, probability in exact_options_full.items()
        ]
        score, selected = max(scored, key=lambda pair: (pair[0], -exact_preference[pair[1]]))
        exact_nested[fold_id == held] = exact_options_full[selected][fold_id == held]
        exact_nested_choices.append({
            "held_fold": int(held + 1),
            "selected": selected,
            "other_four_folds_terminal_with_char20_macro_f1": float(score),
            "other_four_scores": {name: float(value) for value, name in scored},
        })
    exact_selected = [str(value["selected"]) for value in exact_nested_choices]
    exact_consensus = min(exact_options_full, key=lambda name: (-exact_selected.count(name), exact_preference[name]))
    exact_nested_metric = metrics(y, terminal, exact_nested, char_full)
    exact_consensus_fixed = metrics(y[fixed_valid], fixed_terminal, exact_options_fixed[exact_consensus], char_fixed)
    exact_nested_delta = {
        "full_terminal_with_char20": float(exact_nested_metric["terminal_with_char20_macro_f1"] - exact_baseline_full["terminal_with_char20_macro_f1"]),
        "full_hybrid": float(exact_nested_metric["hybrid_future_history_plus_char20_macro_f1"] - exact_baseline_full["hybrid_future_history_plus_char20_macro_f1"]),
        "fixed_terminal_with_char20": float(exact_consensus_fixed["terminal_with_char20_macro_f1"] - exact_baseline_fixed["terminal_with_char20_macro_f1"]),
        "fixed_hybrid": float(exact_consensus_fixed["hybrid_future_history_plus_char20_macro_f1"] - exact_baseline_fixed["hybrid_future_history_plus_char20_macro_f1"]),
    }
    report["exact_last3_transition_reproduction"] = {
        "spec": {
            "context": "(source, last 3 assistant_action names), left-padded with literal 'none'",
            "training": "all fold-training rows; source-specific class prior uses all fold-training rows plus one per class",
            "retained_context": "exact support >= 10",
            "posterior": "(class_counts + 2 * source_prior) / (support + 2)",
            "gate": "terminal row AND exact support>=10 AND p_markov.max>=.45",
            "blend": "base^.88 * markov^.12; .96 counterpart is the only conservative control",
            "fixed_selector": "strict_fixed_raw_seed42.npy (matching deployed selector); legacy seed808 not used",
            "char20": "applied last after every candidate",
        },
        "fold_stats": exact_fold_stats,
        "fixed_refit_stats": exact_fixed_model.compact_stats(),
        "baseline": {"full": exact_baseline_full, "fixed": exact_baseline_fixed},
        "static": exact_static,
        "nested_other_four_fold_selection": {
            "choices": exact_nested_choices,
            "full": exact_nested_metric,
            "fixed_consensus_without_fixed_tuning": exact_consensus,
            "fixed_consensus": exact_consensus_fixed,
            "delta_vs_base": exact_nested_delta,
        },
    }
    arrays.update({
        "exact_last3_full": exact_full, "exact_last3_support_full": exact_support_full,
        "exact_last3_fixed": exact_fixed, "exact_last3_support_fixed": exact_support_fixed,
        "exact_last3_088_full": exact_full_088, "exact_last3_088_fixed": exact_fixed_088,
        "exact_last3_096_full": exact_full_096, "exact_last3_096_fixed": exact_fixed_096,
        "exact_last3_nested_full": exact_nested,
    })

    # Promotion rule intentionally requires a strictly positive terminal and
    # hybrid signal before char20 on both independent protocols.  This avoids
    # a context model winning merely through a lucky non-terminal proxy.
    safe: list[str] = []
    for name, item in report["candidates"].items():  # type: ignore[union-attr]
        delta = item["delta_vs_baseline"]  # type: ignore[index]
        if (
            delta["full_terminal_with_char20"] > 0.0 and delta["full_hybrid"] > 0.0
            and delta["fixed_terminal_with_char20"] > 0.0 and delta["fixed_hybrid"] > 0.0
        ):
            safe.append(name)
    nested_delta = report["nested_outer_selection"]["delta_vs_baseline"]  # type: ignore[index]
    nested_safe = (
        nested_delta["full_terminal_with_char20"] > 0.0 and nested_delta["full_hybrid"] > 0.0
        and nested_delta["fixed_terminal_with_char20"] > 0.0 and nested_delta["fixed_hybrid"] > 0.0
    )
    exact_static_safe: list[str] = []
    for name, item in exact_static.items():
        delta = item["delta_vs_base"]  # type: ignore[index]
        if (
            delta["full_terminal_with_char20"] > 0.0 and delta["full_hybrid"] > 0.0
            and delta["fixed_terminal_with_char20"] > 0.0 and delta["fixed_hybrid"] > 0.0
        ):
            exact_static_safe.append(name)
    exact_nested_safe = (
        exact_nested_delta["full_terminal_with_char20"] > 0.0 and exact_nested_delta["full_hybrid"] > 0.0
        and exact_nested_delta["fixed_terminal_with_char20"] > 0.0 and exact_nested_delta["fixed_hybrid"] > 0.0
    )
    report["decision"] = {
        "safe_two_split_candidates": safe,
        "nested_consensus_safe": bool(nested_safe),
        "exact_last3_static_two_split_candidates": exact_static_safe,
        "exact_last3_nested_consensus_safe": bool(exact_nested_safe),
        "promote": (bool(safe) and bool(nested_safe)) or (bool(exact_static_safe) and bool(exact_nested_safe)),
        "rule": "A candidate must pass the static two-split screen and the outer-held-fold nested choice must strictly improve terminal-with-char20 and future-history+char20 hybrid macro-F1 on the matching deployed-seed 2032 strict fixed proxy.",
        "reason": "No live asset is eligible unless the conservative two-split rule is satisfied.",
    }
    report["seconds"] = round(time.time() - started, 2)
    np.savez_compressed(out / "probabilities.npz", **arrays)
    (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
