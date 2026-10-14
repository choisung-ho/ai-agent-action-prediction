#!/usr/bin/env python3
"""Strict cross-session TF-IDF retrieval audit against exact public old-B43.

This is an experiment-only, terminal-only residual correction.  It uses a
predeclared word+character TF-IDF representation of the current prompt and
already supplied history, then votes among five nearest *other-session*
terminal examples.  The reference action labels are available only from the
training side of each split.  Session identifiers are used solely to enforce
exclusion, never as text or as a retrieval key.

The only candidate policy set is deliberately small and high-confidence.  A
five-fold outer GroupKFold obtains unbiased predictions; an inner GroupKFold
on each outer-train partition selects a policy (including no-op) without the
held-fold labels.  The matching seed-42 fixed split receives only the modal
outer policy.  Nothing in this program changes model/, script.py, or a ZIP.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize


ROOT = Path(__file__).resolve().parents[1]
CLASSES = (
    "read_file", "grep_search", "list_directory", "glob_pattern",
    "edit_file", "write_file", "apply_patch", "run_bash", "run_tests",
    "lint_or_typecheck", "ask_user", "plan_task", "web_search", "respond_only",
)
TO_ID = {name: number for number, name in enumerate(CLASSES)}
N_CLASSES = len(CLASSES)
EPS = 1.0e-12
BOOTSTRAP_REPLICATES = 5000
INNER_SPLITS = 4
NEIGHBORS = 5
FULL_EXPECTED_B43 = {
    "all_fallback_macro_f1": 0.7603385811033483,
    "terminal_fallback_macro_f1": 0.7312748947080282,
    "terminal_with_char20_macro_f1": 0.7335225252513891,
    "hybrid_future_history_plus_char20_macro_f1": 0.98277826622222,
}
FIXED_EXPECTED_B43 = {
    "all_fallback_macro_f1": 0.7585157741730599,
    "terminal_fallback_macro_f1": 0.6969994677821684,
    "terminal_with_char20_macro_f1": 0.7000338475808981,
    "hybrid_future_history_plus_char20_macro_f1": 0.9853915549361344,
}
METRIC_KEYS = tuple(FULL_EXPECTED_B43)
TOKEN_RE = re.compile(r"\w+", re.UNICODE)


@dataclass(frozen=True)
class Policy:
    name: str
    min_top_similarity: float | None
    min_purity: float | None


# Declared before any outcome scoring.  ``no_change`` is a real candidate in
# every nested choice, so retrieval cannot be forced onto a fold.
POLICIES = (
    Policy("no_change", None, None),
    Policy("knn5_sim065_purity080", 0.65, 0.80),
    Policy("knn5_sim075_purity080", 0.75, 0.80),
    Policy("knn5_sim080_purity100", 0.80, 1.00),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out-dir", default="experiments/b43_cross_session_tfidf_retrieval",
        help="Experiment-only output directory; model/ and submission ZIPs are rejected.",
    )
    parser.add_argument("--bootstrap-replicates", type=int, default=BOOTSTRAP_REPLICATES)
    return parser.parse_args()


def session_of(sample_id: str) -> str:
    return sample_id.rsplit("-step_", 1)[0]


def terminal_mask(ids: np.ndarray) -> np.ndarray:
    greatest: dict[str, int] = {}
    parsed: list[tuple[str, int]] = []
    for sample_id in ids.astype(str):
        group, raw_step = sample_id.rsplit("-step_", 1)
        step = int(raw_step)
        greatest[group] = max(greatest.get(group, -1), step)
        parsed.append((group, step))
    return np.asarray([step == greatest[group] for group, step in parsed], dtype=bool)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_alpha(ids: np.ndarray) -> np.ndarray:
    return np.asarray(
        [0.85 if value.startswith("sess_au_") else 0.20 for value in ids.astype(str)],
        dtype=np.float64,
    )


def geometric(raw: np.ndarray, gru: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    log_probability = alpha[:, None] * np.log(np.maximum(raw, EPS))
    log_probability += (1.0 - alpha[:, None]) * np.log(np.maximum(gru, EPS))
    log_probability -= log_probability.max(axis=1, keepdims=True)
    probability = np.exp(log_probability)
    return (probability / probability.sum(axis=1, keepdims=True)).astype(np.float32)


def labels_with_char(
    fallback: np.ndarray, terminal: np.ndarray, char20: np.ndarray,
) -> np.ndarray:
    final = fallback.astype(np.int16, copy=True)
    use = terminal & (char20 >= 0)
    final[use] = char20[use]
    return final


def macro(y: np.ndarray, prediction: np.ndarray) -> float:
    return float(f1_score(
        y, prediction, labels=np.arange(N_CLASSES), average="macro", zero_division=0,
    ))


def metric_bundle_from_labels(
    y: np.ndarray, fallback: np.ndarray, final: np.ndarray, terminal: np.ndarray,
) -> dict[str, float]:
    hybrid = final.copy()
    hybrid[~terminal] = y[~terminal]
    return {
        "all_fallback_macro_f1": macro(y, fallback),
        "terminal_fallback_macro_f1": macro(y[terminal], fallback[terminal]),
        "terminal_with_char20_macro_f1": macro(y[terminal], final[terminal]),
        "hybrid_future_history_plus_char20_macro_f1": macro(y, hybrid),
    }


def metric_delta(candidate: dict[str, float], baseline: dict[str, float]) -> dict[str, float]:
    return {key: float(candidate[key] - baseline[key]) for key in METRIC_KEYS}


def assert_control(actual: dict[str, float], expected: dict[str, float], name: str) -> None:
    mismatch = {
        key: (actual[key], expected[key])
        for key in METRIC_KEYS
        if abs(actual[key] - expected[key]) > 1.0e-12
    }
    if mismatch:
        raise ValueError(f"{name} exact old-B43 control mismatch: {mismatch}")


def clean(value: object, limit: int) -> str:
    return " ".join(str(value or "").replace("\x00", " ").casefold().split())[:limit]


def context_text(sample: dict[str, Any]) -> str:
    """Text-only representation; deliberately excludes IDs and session metadata."""
    parts = ["CURRENT " + clean(sample.get("current_prompt"), 800)]
    history = sample.get("history") if isinstance(sample.get("history"), list) else []
    # All of these fields were supplied before the current action at inference.
    for position, item in enumerate(reversed(history[-6:]), start=1):
        if not isinstance(item, dict):
            continue
        role = str(item.get("role") or "").casefold()
        if role == "user":
            parts.append(f"PREV_USER_{position} " + clean(item.get("content"), 320))
        elif role == "assistant_action":
            action = clean(item.get("name") or item.get("action"), 80)
            args = item.get("args") if isinstance(item.get("args"), dict) else {}
            argument = " ".join(
                f"{clean(key, 40)}_{clean(value, 120)}" for key, value in sorted(args.items())
            )
            result = clean(item.get("result_summary"), 180)
            parts.extend((
                f"PREV_ACTION_{position} {action}",
                f"PREV_ARGS_{position} {argument}",
                f"PREV_RESULT_{position} {result}",
            ))
        elif role:
            parts.append(f"PREV_{role}_{position} " + clean(item.get("content"), 220))
    # Do not append session_meta, filename lists, or the sample/session id.
    return " ".join(parts)


def vectorize_neighbors(
    reference_text: list[str], query_text: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    """Return top-k positions and cosine similarities from TF-IDF word+char."""
    word = TfidfVectorizer(
        analyzer="word", ngram_range=(1, 2), min_df=2, max_features=60_000,
        sublinear_tf=True, lowercase=False, dtype=np.float32,
    )
    char = TfidfVectorizer(
        analyzer="char_wb", ngram_range=(3, 5), min_df=3, max_features=80_000,
        sublinear_tf=True, lowercase=False, dtype=np.float32,
    )
    reference = hstack((
        word.fit_transform(reference_text), char.fit_transform(reference_text),
    ), format="csr", dtype=np.float32)
    query = hstack((
        word.transform(query_text), char.transform(query_text),
    ), format="csr", dtype=np.float32)
    normalize(reference, norm="l2", copy=False)
    normalize(query, norm="l2", copy=False)
    model = NearestNeighbors(n_neighbors=NEIGHBORS, metric="cosine", algorithm="brute", n_jobs=-1)
    model.fit(reference)
    distance, position = model.kneighbors(query, return_distance=True)
    return position.astype(np.int32), (1.0 - distance).astype(np.float32)


def retrieval_evidence(
    reference_index: np.ndarray,
    query_index: np.ndarray,
    texts: list[str],
    y: np.ndarray,
    groups: np.ndarray,
) -> dict[str, np.ndarray]:
    """Train a text retriever on terminal rows from other sessions only."""
    reference_groups = np.unique(groups[reference_index])
    query_groups = np.unique(groups[query_index])
    overlap = np.intersect1d(reference_groups, query_groups)
    if len(overlap):
        raise ValueError("Cross-session retrieval contract violated: a query session entered its reference pool")
    if len(reference_index) < NEIGHBORS:
        raise ValueError("Insufficient terminal reference rows for required neighbors")
    position, similarity = vectorize_neighbors(
        [texts[index] for index in reference_index], [texts[index] for index in query_index],
    )
    neighbor_labels = y[reference_index][position]
    vote = np.empty(len(query_index), dtype=np.int16)
    purity = np.empty(len(query_index), dtype=np.float32)
    for row, labels in enumerate(neighbor_labels):
        counts = np.bincount(labels.astype(np.int64), minlength=N_CLASSES)
        maximum = counts.max()
        tied = np.flatnonzero(counts == maximum)
        if len(tied) > 1:
            mass = np.asarray([similarity[row, labels == candidate].sum() for candidate in tied])
            winner = int(tied[np.flatnonzero(mass == mass.max())[0]])
        else:
            winner = int(tied[0])
        vote[row] = winner
        purity[row] = float(counts[winner] / NEIGHBORS)
    return {
        "vote": vote,
        "purity": purity,
        "top_similarity": similarity[:, 0],
        "mean_similarity": similarity.mean(axis=1),
        "neighbor_labels": neighbor_labels.astype(np.int16),
    }


def apply_policy_local(
    base_fallback: np.ndarray,
    char20: np.ndarray,
    query_position: np.ndarray,
    evidence: dict[str, np.ndarray],
    policy: Policy,
) -> tuple[np.ndarray, np.ndarray, dict[str, int | float]]:
    """Change only high-confidence terminal query positions, before char20-last."""
    fallback = base_fallback[query_position].astype(np.int16, copy=True)
    final = labels_with_char(
        fallback,
        np.ones(len(query_position), dtype=bool),
        char20[query_position],
    )
    if policy.min_top_similarity is None or policy.min_purity is None:
        return fallback, final, {
            "eligible": 0, "changed": 0, "top_similarity_mean": 0.0, "purity_mean": 0.0,
        }
    eligible = (
        (char20[query_position] < 0)
        & (evidence["top_similarity"] >= policy.min_top_similarity)
        & (evidence["purity"] >= policy.min_purity)
        & (evidence["vote"] != fallback)
    )
    fallback[eligible] = evidence["vote"][eligible]
    final = labels_with_char(
        fallback,
        np.ones(len(query_position), dtype=bool),
        char20[query_position],
    )
    return fallback, final, {
        "eligible": int(eligible.sum()),
        "changed": int(eligible.sum()),
        "top_similarity_mean": float(evidence["top_similarity"].mean()),
        "purity_mean": float(evidence["purity"].mean()),
    }


def weighted_macro(confusion: np.ndarray) -> float:
    true_positive = np.diag(confusion).astype(np.float64)
    denominator = confusion.sum(axis=0) + confusion.sum(axis=1)
    return float(np.mean(np.divide(
        2.0 * true_positive,
        denominator,
        out=np.zeros(N_CLASSES, dtype=np.float64),
        where=denominator > 0.0,
    )))


def session_bootstrap(
    y: np.ndarray,
    baseline: np.ndarray,
    candidate: np.ndarray,
    groups: np.ndarray,
    repeats: int,
    seed: int,
) -> dict[str, int | float]:
    unique, inverse = np.unique(groups, return_inverse=True)
    if len(unique) != len(groups):
        raise ValueError("Terminal bootstrap needs exactly one terminal row per session")
    base_cell = y.astype(np.int64) * N_CLASSES + baseline.astype(np.int64)
    candidate_cell = y.astype(np.int64) * N_CLASSES + candidate.astype(np.int64)
    random = np.random.default_rng(seed)
    delta = np.empty(repeats, dtype=np.float64)
    for number in range(repeats):
        sampled = random.integers(0, len(unique), size=len(unique))
        weight = np.bincount(sampled, minlength=len(unique))[inverse]
        base_confusion = np.bincount(
            base_cell, weights=weight, minlength=N_CLASSES * N_CLASSES,
        ).reshape(N_CLASSES, N_CLASSES)
        candidate_confusion = np.bincount(
            candidate_cell, weights=weight, minlength=N_CLASSES * N_CLASSES,
        ).reshape(N_CLASSES, N_CLASSES)
        delta[number] = weighted_macro(candidate_confusion) - weighted_macro(base_confusion)
    exact = weighted_macro(np.bincount(
        candidate_cell, minlength=N_CLASSES * N_CLASSES,
    ).reshape(N_CLASSES, N_CLASSES)) - weighted_macro(np.bincount(
        base_cell, minlength=N_CLASSES * N_CLASSES,
    ).reshape(N_CLASSES, N_CLASSES))
    return {
        "replicates": int(repeats),
        "seed": int(seed),
        "sessions": int(len(unique)),
        "exact_delta": float(exact),
        "mean_delta": float(delta.mean()),
        "p_delta_gt_zero": float(np.mean(delta > 0.0)),
        "p05_delta": float(np.quantile(delta, 0.05)),
        "p95_delta": float(np.quantile(delta, 0.95)),
    }


def rank_policy(
    metrics: dict[str, float], changes: int, policy: Policy,
) -> tuple[float, float, int, int]:
    return (
        metrics["terminal_with_char20_macro_f1"],
        metrics["terminal_fallback_macro_f1"],
        -changes,
        -next(index for index, item in enumerate(POLICIES) if item.name == policy.name),
    )


def full_final_from_fallback(
    fallback: np.ndarray, terminal: np.ndarray, char20: np.ndarray,
) -> np.ndarray:
    return labels_with_char(fallback, terminal, char20)


def policy_metric_for_subset(
    y: np.ndarray,
    fallback: np.ndarray,
    terminal: np.ndarray,
    char20: np.ndarray,
    subset: np.ndarray,
) -> dict[str, float]:
    final = full_final_from_fallback(fallback, terminal, char20)
    return metric_bundle_from_labels(y[subset], fallback[subset], final[subset], terminal[subset])


def main() -> None:
    args = parse_args()
    if args.bootstrap_replicates <= 0:
        raise ValueError("--bootstrap-replicates must be positive")
    out = Path(args.out_dir)
    if not out.is_absolute():
        out = ROOT / out
    out = out.resolve()
    model_dir = (ROOT / "model").resolve()
    if out == model_dir or model_dir in out.parents:
        raise ValueError("Experiment output cannot be inside model/")
    out.mkdir(parents=True, exist_ok=True)
    began = time.perf_counter()

    samples = [
        json.loads(line) for line in (ROOT / "data" / "train.jsonl").open(encoding="utf-8")
        if line.strip()
    ]
    ids = np.asarray([str(sample["id"]) for sample in samples])
    labels = {
        row["id"]: row["action"]
        for row in csv.DictReader((ROOT / "data" / "train_labels.csv").open(encoding="utf-8"))
    }
    split = {
        row["id"]: row["split"]
        for row in csv.DictReader((ROOT / "splits" / "common_split.csv").open(encoding="utf-8"))
    }
    y = np.asarray([TO_ID[labels[value]] for value in ids], dtype=np.int16)
    groups = np.asarray([session_of(value) for value in ids])
    terminal = terminal_mask(ids)
    texts = [context_text(sample) for sample in samples]
    fixed_index = np.flatnonzero(np.asarray([split[value] == "val" for value in ids]))
    common_index = np.flatnonzero(np.asarray([split[value] == "train" for value in ids]))
    fixed_y = y[fixed_index]
    fixed_groups = groups[fixed_index]
    fixed_terminal = terminal[fixed_index]
    if len(np.intersect1d(np.unique(groups[common_index]), np.unique(fixed_groups))):
        raise ValueError("common train/fixed split leaks sessions")

    experiment = ROOT / "experiments"
    paths = {
        "full_raw_seed42": experiment / "xgb_robust_ensemble" / "full_oof_raw_seed42.npy",
        "full_old_noe5_gru": experiment / "modeling_ai_sequence_fusion_noe5" / "oof_probabilities.npy",
        "fixed_raw_seed42": experiment / "xgb_robust_ensemble" / "strict_fixed_raw_seed42.npy",
        "fixed_old_noe5_gru": experiment / "modeling_ai_sequence_fusion_fixed_noe5" / "valid_probabilities.npy",
        "fixed_old_indices": experiment / "modeling_ai_sequence_fusion_fixed_noe5" / "valid_indices.npy",
        "fixed_old_terminal": experiment / "modeling_ai_sequence_fusion_fixed_noe5" / "terminal_mask.npy",
        "char20": experiment / "terminal_rule_combinations" / "oof_candidates.npz",
    }
    raw = np.load(paths["full_raw_seed42"], allow_pickle=False).astype(np.float32)
    old = np.load(paths["full_old_noe5_gru"], allow_pickle=False).astype(np.float32)
    raw_fixed = np.load(paths["fixed_raw_seed42"], allow_pickle=False).astype(np.float32)
    old_fixed = np.load(paths["fixed_old_noe5_gru"], allow_pickle=False).astype(np.float32)
    if not (raw.shape == old.shape == (len(y), N_CLASSES)):
        raise ValueError("full old-B43 raw/GRU shape mismatch")
    if not (raw_fixed.shape == old_fixed.shape == (len(fixed_index), N_CLASSES)):
        raise ValueError("fixed old-B43 raw/GRU shape mismatch")
    if not (
        np.array_equal(np.load(paths["fixed_old_indices"], allow_pickle=False).astype(np.int64), fixed_index)
        and np.array_equal(np.load(paths["fixed_old_terminal"], allow_pickle=False).astype(bool), fixed_terminal)
    ):
        raise ValueError("fixed old no-E5 artifact does not match exact seed-42 common split")

    outer_fold = np.empty(len(y), dtype=np.int8)
    outer_splits = list(GroupKFold(n_splits=5).split(np.zeros(len(y)), y, groups))
    for fold_number, (_, held) in enumerate(outer_splits):
        outer_fold[held] = fold_number
    char = np.load(paths["char20"], allow_pickle=False)
    if not (
        np.array_equal(char["y"].astype(np.int16), y)
        and np.array_equal(char["terminal"].astype(bool), terminal)
        and np.array_equal(char["fold"].astype(np.int8)[terminal], outer_fold[terminal])
    ):
        raise ValueError("full char20 artifact is not aligned with exact B43 folds")
    char_full = char["char20"].astype(np.int16)
    fixed_char = np.full(len(fixed_index), -1, dtype=np.int16)
    fixed_char[fixed_terminal] = char["fixed_char20"].astype(np.int16)
    if not (
        np.array_equal(char["fixed_y"].astype(np.int16), fixed_y[fixed_terminal])
        and np.array_equal(char["valid_ids"].astype(str), ids[fixed_index][fixed_terminal])
    ):
        raise ValueError("fixed char20 labels/IDs do not match exact B43 common split")

    full_probability = geometric(raw, old, source_alpha(ids))
    fixed_probability = geometric(raw_fixed, old_fixed, source_alpha(ids[fixed_index]))
    base_fallback = full_probability.argmax(axis=1).astype(np.int16)
    base_final = full_final_from_fallback(base_fallback, terminal, char_full)
    base_fixed_fallback = fixed_probability.argmax(axis=1).astype(np.int16)
    base_fixed_final = full_final_from_fallback(base_fixed_fallback, fixed_terminal, fixed_char)
    baseline_full = metric_bundle_from_labels(y, base_fallback, base_final, terminal)
    baseline_fixed = metric_bundle_from_labels(fixed_y, base_fixed_fallback, base_fixed_final, fixed_terminal)
    assert_control(baseline_full, FULL_EXPECTED_B43, "full")
    assert_control(baseline_fixed, FIXED_EXPECTED_B43, "matching_seed42_fixed")

    static_fallback = {policy.name: base_fallback.copy() for policy in POLICIES}
    nested_fallback = base_fallback.copy()
    outer_records: list[dict[str, Any]] = []
    retrieval_call_count = 0
    for held_fold, (outer_train, outer_held) in enumerate(outer_splits, start=1):
        outer_train_terminal = outer_train[terminal[outer_train]]
        outer_held_terminal = outer_held[terminal[outer_held]]
        inner_fallback = {policy.name: base_fallback.copy() for policy in POLICIES}
        inner_changed = {policy.name: 0 for policy in POLICIES}
        inner_splits = list(GroupKFold(n_splits=INNER_SPLITS).split(
            np.zeros(len(outer_train)), y[outer_train], groups[outer_train],
        ))
        for _, (inner_train_pos, inner_valid_pos) in enumerate(inner_splits, start=1):
            inner_train = outer_train[inner_train_pos]
            inner_valid = outer_train[inner_valid_pos]
            reference = inner_train[terminal[inner_train]]
            query = inner_valid[terminal[inner_valid]]
            evidence = retrieval_evidence(reference, query, texts, y, groups)
            retrieval_call_count += 1
            for policy in POLICIES:
                local_fallback, _local_final, diagnostic = apply_policy_local(
                    base_fallback, char_full, query, evidence, policy,
                )
                inner_fallback[policy.name][query] = local_fallback
                inner_changed[policy.name] += int(diagnostic["changed"])
        inner_metrics = {
            policy.name: policy_metric_for_subset(
                y, inner_fallback[policy.name], terminal, char_full, outer_train,
            )
            for policy in POLICIES
        }
        selected = max(
            POLICIES,
            key=lambda policy: rank_policy(inner_metrics[policy.name], inner_changed[policy.name], policy),
        )
        outer_evidence = retrieval_evidence(outer_train_terminal, outer_held_terminal, texts, y, groups)
        retrieval_call_count += 1
        held_diagnostic: dict[str, Any] = {}
        for policy in POLICIES:
            local_fallback, _local_final, diagnostic = apply_policy_local(
                base_fallback, char_full, outer_held_terminal, outer_evidence, policy,
            )
            static_fallback[policy.name][outer_held_terminal] = local_fallback
            held_diagnostic[policy.name] = diagnostic
            if policy.name == selected.name:
                nested_fallback[outer_held_terminal] = local_fallback
        outer_records.append({
            "held_outer_session_fold": held_fold,
            "outer_train_sessions": int(len(np.unique(groups[outer_train]))),
            "held_sessions": int(len(np.unique(groups[outer_held]))),
            "held_terminal_rows": int(len(outer_held_terminal)),
            "inner_selection": {
                "candidate_metrics_on_inner_cross_session_oof": inner_metrics,
                "candidate_changed_terminal_rows": inner_changed,
                "selected_without_held_fold_labels": selected.name,
            },
            "held_retrieval_diagnostic": held_diagnostic,
        })

    nested_final = full_final_from_fallback(nested_fallback, terminal, char_full)
    nested_metric = metric_bundle_from_labels(y, nested_fallback, nested_final, terminal)
    nested_delta = metric_delta(nested_metric, baseline_full)
    selected_names = [row["inner_selection"]["selected_without_held_fold_labels"] for row in outer_records]
    selected_count = Counter(selected_names)
    modal_policy = next(policy for policy in POLICIES if selected_count[policy.name] == max(selected_count.values()))
    static_full: dict[str, dict[str, Any]] = {}
    for policy in POLICIES:
        final = full_final_from_fallback(static_fallback[policy.name], terminal, char_full)
        metric = metric_bundle_from_labels(y, static_fallback[policy.name], final, terminal)
        static_full[policy.name] = {
            "metric": metric,
            "delta_vs_b43": metric_delta(metric, baseline_full),
            "changed_terminal_rows": int(np.sum(static_fallback[policy.name][terminal] != base_fallback[terminal])),
        }

    fixed_reference = common_index[terminal[common_index]]
    fixed_query_global = fixed_index[fixed_terminal]
    fixed_query_local = np.flatnonzero(fixed_terminal)
    fixed_evidence = retrieval_evidence(fixed_reference, fixed_query_global, texts, y, groups)
    retrieval_call_count += 1
    static_fixed: dict[str, dict[str, Any]] = {}
    fixed_fallback_by_policy: dict[str, np.ndarray] = {}
    for policy in POLICIES:
        candidate_fallback = base_fixed_fallback.copy()
        local_fallback, _local_final, diagnostic = apply_policy_local(
            base_fixed_fallback, fixed_char, fixed_query_local, fixed_evidence, policy,
        )
        candidate_fallback[fixed_query_local] = local_fallback
        candidate_final = full_final_from_fallback(candidate_fallback, fixed_terminal, fixed_char)
        metric = metric_bundle_from_labels(fixed_y, candidate_fallback, candidate_final, fixed_terminal)
        static_fixed[policy.name] = {
            "metric": metric,
            "delta_vs_b43": metric_delta(metric, baseline_fixed),
            "diagnostic": diagnostic,
        }
        fixed_fallback_by_policy[policy.name] = candidate_fallback
    fixed_nested_fallback = fixed_fallback_by_policy[modal_policy.name]
    fixed_nested_final = full_final_from_fallback(fixed_nested_fallback, fixed_terminal, fixed_char)
    fixed_nested_metric = metric_bundle_from_labels(
        fixed_y, fixed_nested_fallback, fixed_nested_final, fixed_terminal,
    )
    fixed_nested_delta = metric_delta(fixed_nested_metric, baseline_fixed)

    full_bootstrap = session_bootstrap(
        y[terminal], base_final[terminal], nested_final[terminal], groups[terminal],
        args.bootstrap_replicates, seed=20260715,
    )
    fixed_bootstrap = session_bootstrap(
        fixed_y[fixed_terminal], base_fixed_final[fixed_terminal], fixed_nested_final[fixed_terminal],
        fixed_groups[fixed_terminal], args.bootstrap_replicates, seed=20260716,
    )
    held_fold_delta = []
    for held_fold, (_, held) in enumerate(outer_splits, start=1):
        base = metric_bundle_from_labels(y[held], base_fallback[held], base_final[held], terminal[held])
        candidate = metric_bundle_from_labels(y[held], nested_fallback[held], nested_final[held], terminal[held])
        held_fold_delta.append({
            "held_outer_session_fold": held_fold,
            "delta": metric_delta(candidate, base),
        })
    positive_folds = {
        metric: int(sum(row["delta"][metric] > 0.0 for row in held_fold_delta))
        for metric in METRIC_KEYS
    }
    all_eight_positive = all(value > 0.0 for value in nested_delta.values()) and all(
        value > 0.0 for value in fixed_nested_delta.values()
    )
    promotion_gate = {
        "all_eight_full_fixed_metrics_strictly_positive": bool(all_eight_positive),
        "full_terminal_bootstrap_p_delta_gt_zero_ge_0_95": bool(full_bootstrap["p_delta_gt_zero"] >= 0.95),
        "fixed_terminal_bootstrap_p_delta_gt_zero_ge_0_95": bool(fixed_bootstrap["p_delta_gt_zero"] >= 0.95),
        "full_terminal_bootstrap_p05_delta_gt_zero": bool(full_bootstrap["p05_delta"] > 0.0),
        "fixed_terminal_bootstrap_p05_delta_gt_zero": bool(fixed_bootstrap["p05_delta"] > 0.0),
        "full_all_fallback_positive_in_at_least_4_of_5_folds": bool(
            positive_folds["all_fallback_macro_f1"] >= 4
        ),
        "full_terminal_char20_positive_in_at_least_4_of_5_folds": bool(
            positive_folds["terminal_with_char20_macro_f1"] >= 4
        ),
    }

    report = {
        "scope": "Experiment-only retrieval audit. No existing model artifact, deployment script.py, B42.zip, baseline_submit.zip, or challenger ZIP was modified.",
        "retrieval_contract": {
            "intervention_scope": "Terminal rows only; retrieval label replaces B43 fallback only when it differs and passes similarity/purity. Existing char20 override is always applied last.",
            "reference_population": "Only terminal rows of training-side sessions. All reference/query session intersections are asserted empty on every outer, inner, and fixed call.",
            "text_features": "Word 1-2 gram + char_wb 3-5 gram TF-IDF over current prompt and up to six supplied prior history messages/actions/args/results.",
            "excluded_from_text": ["sample ID", "session ID", "session_meta", "current target action", "future history", "fixed labels"],
            "label_use": "Training-side target actions supply only nearest-neighbor votes. They are never serialized as ID/key lookup rules and fixed/held labels are never used to fit a retriever or choose a fixed policy.",
            "predeclared_policies": [
                {
                    "name": policy.name,
                    "neighbors": NEIGHBORS,
                    "min_top_similarity": policy.min_top_similarity,
                    "min_vote_purity": policy.min_purity,
                }
                for policy in POLICIES
            ],
        },
        "component_provenance": {
            name: {
                "path": str(path.relative_to(ROOT)).replace("\\", "/"),
                "sha256": sha256(path),
                "bytes": int(path.stat().st_size),
            }
            for name, path in paths.items()
        },
        "alignment_checks": {
            "full_rows": int(len(y)),
            "fixed_rows": int(len(fixed_y)),
            "full_terminal_rows": int(terminal.sum()),
            "fixed_terminal_rows": int(fixed_terminal.sum()),
            "common_train_fixed_sessions_disjoint": True,
            "char20_full_and_fixed_alignment": True,
            "every_retrieval_reference_query_session_disjoint": True,
            "retrieval_calls": retrieval_call_count,
        },
        "old_b43_control": {
            "full_outer_oof": baseline_full,
            "matching_seed42_fixed": baseline_fixed,
        },
        "static_predeclared_policy_diagnostic": {
            "full_outer_oof": static_full,
            "matching_seed42_fixed": static_fixed,
            "note": "Static rows are descriptive only; the deployment-like result below is the nested outer OOF plus modal train-derived fixed policy.",
        },
        "nested_whole_session_result": {
            "selection_rule": "Per outer held session fold, choose policy on only inner GroupKFold cross-session OOF terminal-char20 F1; terminal fallback F1, fewer changes, then policy order break ties. No-op participates.",
            "outer_records": outer_records,
            "selected_policy_counts": {policy.name: int(selected_count[policy.name]) for policy in POLICIES},
            "matching_fixed_modal_policy": modal_policy.name,
            "full_outer_oof": nested_metric,
            "matching_seed42_fixed": fixed_nested_metric,
            "delta_vs_old_b43": {
                "full_outer_oof": nested_delta,
                "matching_seed42_fixed": fixed_nested_delta,
            },
            "terminal_char20_session_bootstrap": {
                "full_outer_oof": full_bootstrap,
                "matching_seed42_fixed": fixed_bootstrap,
            },
            "held_outer_fold_deltas": held_fold_delta,
            "positive_fold_counts": positive_folds,
        },
        "promotion_gate": {
            "criteria": "All eight deltas positive, both 5k terminal session bootstrap P(delta>0)>=.95 and p05>0, plus >=4/5 positive full all-fallback and terminal-char20 held folds.",
            "checks": promotion_gate,
            "pass": bool(all(promotion_gate.values())),
        },
        "submission_action": {
            "candidate_created": False,
            "zip_updated": False,
            "reason": "This is a bounded audit only. A ZIP is prohibited unless a separate parent decision follows a passing full/fixed gate and deployment feasibility validation.",
        },
        "elapsed_seconds": round(time.perf_counter() - began, 3),
    }
    np.savez_compressed(
        out / "nested_predictions.npz",
        y=y,
        terminal=terminal,
        base_fallback=base_fallback,
        base_final=base_final,
        nested_fallback=nested_fallback,
        nested_final=nested_final,
        fixed_index=fixed_index,
        fixed_terminal=fixed_terminal,
        base_fixed_fallback=base_fixed_fallback,
        base_fixed_final=base_fixed_final,
        nested_fixed_fallback=fixed_nested_fallback,
        nested_fixed_final=fixed_nested_final,
    )
    (out / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    (out / "README.md").write_text(
        "# Strict cross-session TF-IDF retrieval audit\n\n"
        "This experiment never modifies `model/`, deployment `script.py`, or a submission ZIP. "
        "It validates exact old-B43 first, then evaluates a terminal-only other-session nearest-neighbor correction with nested whole-session selection and matching seed-42 fixed validation.\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "output": str(out),
        "selected_policy_counts": report["nested_whole_session_result"]["selected_policy_counts"],
        "modal_policy": modal_policy.name,
        "full_delta": nested_delta,
        "fixed_delta": fixed_nested_delta,
        "promotion_gate_pass": report["promotion_gate"]["pass"],
        "seconds": report["elapsed_seconds"],
    }, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
