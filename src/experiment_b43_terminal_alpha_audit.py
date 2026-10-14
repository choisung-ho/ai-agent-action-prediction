#!/usr/bin/env python3
"""Audit terminal-only source alpha adjustments for the public old-B43 path.

This is an isolated probability audit.  It retains B43's exact non-terminal
probabilities, its char20 terminal overlay, and its later-history evaluation
ordering.  The sole intervention is a small predeclared geometric blend grid
applied to rows that are the maximum supplied step of a session.  It never
writes a model, ``script.py``, or any ZIP/submission asset.
"""

from __future__ import annotations

import csv
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments" / "b43_old_b43_terminal_alpha_audit"
CLASSES = (
    "read_file", "grep_search", "list_directory", "glob_pattern",
    "edit_file", "write_file", "apply_patch", "run_bash", "run_tests",
    "lint_or_typecheck", "ask_user", "plan_task", "web_search", "respond_only",
)
TO_ID = {name: number for number, name in enumerate(CLASSES)}
N_CLASSES = len(CLASSES)
EPS = 1.0e-12

# Declared before scoring.  The B43 global route is SIM=.20/AU=.85; only
# supplied-session terminal rows receive a candidate pair from this grid.
SIM_TERMINAL_ALPHAS = (0.10, 0.20, 0.30, 0.40, 0.50)
AU_TERMINAL_ALPHAS = (0.70, 0.85, 0.95)
BASE_POLICY = (0.20, 0.85)
REQUIRED = (
    "all_fallback_macro_f1",
    "terminal_with_char20_macro_f1",
    "hybrid_future_history_plus_char20_macro_f1",
)


def session_of(sample_id: str) -> str:
    return sample_id.rsplit("-step_", 1)[0]


def step_of(sample_id: str) -> int:
    return int(sample_id.rsplit("-step_", 1)[1])


def terminal_mask(ids: np.ndarray) -> np.ndarray:
    maximum: dict[str, int] = {}
    parsed: list[tuple[str, int]] = []
    for value in ids.astype(str):
        group = session_of(value)
        step = step_of(value)
        maximum[group] = max(maximum.get(group, -1), step)
        parsed.append((group, step))
    return np.asarray([step == maximum[group] for group, step in parsed], dtype=bool)


def source_of(ids: np.ndarray) -> np.ndarray:
    return np.asarray(["au" if value.startswith("sess_au_") else "sim" for value in ids.astype(str)])


def geometric(raw: np.ndarray, gru: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    value = alpha[:, None] * np.log(np.maximum(raw, EPS))
    value += (1.0 - alpha[:, None]) * np.log(np.maximum(gru, EPS))
    value -= value.max(axis=1, keepdims=True)
    value = np.exp(value)
    return (value / value.sum(axis=1, keepdims=True)).astype(np.float32)


def macro(y: np.ndarray, prediction: np.ndarray, mask: np.ndarray | None = None) -> float:
    if mask is None:
        mask = np.ones(len(y), dtype=bool)
    return float(f1_score(
        y[mask], prediction[mask], labels=np.arange(N_CLASSES), average="macro", zero_division=0,
    ))


def final_prediction(probability: np.ndarray, terminal: np.ndarray, char20: np.ndarray) -> np.ndarray:
    output = probability.argmax(axis=1).astype(np.int16)
    use = terminal & (char20 >= 0)
    output[use] = char20[use]
    return output


def metric_bundle(y: np.ndarray, probability: np.ndarray, terminal: np.ndarray, char20: np.ndarray) -> dict[str, float]:
    fallback = probability.argmax(axis=1).astype(np.int16)
    final = final_prediction(probability, terminal, char20)
    hybrid = final.copy()
    hybrid[~terminal] = y[~terminal]
    return {
        "all_fallback_macro_f1": macro(y, fallback),
        "terminal_fallback_macro_f1": macro(y, fallback, terminal),
        "terminal_with_char20_macro_f1": macro(y, final, terminal),
        "hybrid_future_history_plus_char20_macro_f1": macro(y, hybrid),
    }


def difference(candidate: dict[str, float], baseline: dict[str, float]) -> dict[str, float]:
    return {name: float(candidate[name] - baseline[name]) for name in baseline}


def policy_name(sim_alpha: float, au_alpha: float) -> str:
    return f"sim{sim_alpha:.2f}_au{au_alpha:.2f}"


def policy_distance(policy: tuple[float, float]) -> float:
    return abs(policy[0] - BASE_POLICY[0]) + abs(policy[1] - BASE_POLICY[1])


def terminal_adjusted_probability(
    raw: np.ndarray,
    gru: np.ndarray,
    source: np.ndarray,
    terminal: np.ndarray,
    policy: tuple[float, float],
) -> np.ndarray:
    """B43 globally, with a candidate alpha only for batch-terminal rows."""
    alpha = np.where(source == "au", BASE_POLICY[1], BASE_POLICY[0]).astype(np.float64)
    terminal_alpha = np.where(source == "au", policy[1], policy[0])
    alpha[terminal] = terminal_alpha[terminal]
    return geometric(raw, gru, alpha)


def fast_macro(y: np.ndarray, prediction: np.ndarray) -> float:
    confusion = np.bincount(N_CLASSES * y + prediction, minlength=N_CLASSES * N_CLASSES)
    confusion = confusion.reshape(N_CLASSES, N_CLASSES)
    true_positive = np.diag(confusion).astype(np.float64)
    denominator = confusion.sum(axis=0) + confusion.sum(axis=1)
    return float(np.mean(np.divide(2.0 * true_positive, denominator, out=np.zeros(N_CLASSES), where=denominator > 0)))


def session_bootstrap(
    y: np.ndarray, baseline: np.ndarray, candidate: np.ndarray, terminal: np.ndarray,
    repeats: int = 3000, seed: int = 20260713,
) -> dict[str, float | int]:
    # Every terminal row belongs to exactly one session.  Sampling terminal
    # positions is therefore a session bootstrap, not a row-leakage proxy.
    index = np.flatnonzero(terminal)
    random = np.random.default_rng(seed)
    delta = np.empty(repeats, dtype=np.float64)
    for number in range(repeats):
        sampled = index[random.integers(0, len(index), len(index))]
        delta[number] = fast_macro(y[sampled], candidate[sampled]) - fast_macro(y[sampled], baseline[sampled])
    exact = fast_macro(y[index], candidate[index]) - fast_macro(y[index], baseline[index])
    return {
        "repeats": int(repeats),
        "exact_delta": float(exact),
        "mean_delta": float(delta.mean()),
        "p_delta_gt_zero": float(np.mean(delta > 0.0)),
        "p05": float(np.quantile(delta, 0.05)),
        "p95": float(np.quantile(delta, 0.95)),
    }


def main() -> None:
    began = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(line) for line in (ROOT / "data" / "train.jsonl").open(encoding="utf-8") if line.strip()]
    labels = {row["id"]: row["action"] for row in csv.DictReader((ROOT / "data" / "train_labels.csv").open(encoding="utf-8"))}
    split = {row["id"]: row["split"] for row in csv.DictReader((ROOT / "splits" / "common_split.csv").open(encoding="utf-8"))}
    ids = np.asarray([str(row["id"]) for row in rows])
    y = np.asarray([TO_ID[labels[value]] for value in ids], dtype=np.int16)
    groups = np.asarray([session_of(value) for value in ids])
    source = source_of(ids)
    terminal = terminal_mask(ids)
    fixed_index = np.flatnonzero(np.asarray([split[value] == "val" for value in ids]))
    common_index = np.flatnonzero(np.asarray([split[value] == "train" for value in ids]))
    fixed_terminal = terminal[fixed_index]

    # These are the historical public-B43 components, not the later strict
    # distilled GRU path.  The full raw table happens to match the live proxy,
    # but this explicit seed-42 artifact is the provenance used by the
    # public-feedback reconstruction.
    raw_path = ROOT / "experiments" / "xgb_robust_ensemble" / "full_oof_raw_seed42.npy"
    gru_path = ROOT / "experiments" / "modeling_ai_sequence_fusion_noe5" / "oof_probabilities.npy"
    raw = np.load(raw_path, allow_pickle=False).astype(np.float32)
    gru = np.load(gru_path, allow_pickle=False).astype(np.float32)
    if raw.shape != gru.shape or raw.shape != (len(rows), N_CLASSES):
        raise ValueError("B43 full raw/GRU shape mismatch")

    char_data = np.load(ROOT / "experiments" / "terminal_rule_combinations" / "oof_candidates.npz", allow_pickle=False)
    if not (
        np.array_equal(char_data["y"].astype(np.int16), y)
        and np.array_equal(char_data["terminal"].astype(bool), terminal)
    ):
        raise ValueError("B43 full char20 artifact mismatch")
    char_full = char_data["char20"].astype(np.int16)
    if not np.array_equal(char_data["valid_ids"].astype(str), ids[fixed_index][fixed_terminal]):
        raise ValueError("B43 fixed char20 IDs do not align")
    char_fixed = np.full(len(fixed_index), -1, dtype=np.int16)
    char_fixed[fixed_terminal] = char_data["fixed_char20"].astype(np.int16)

    raw_fixed_path = ROOT / "experiments" / "xgb_robust_ensemble" / "strict_fixed_raw_seed42.npy"
    gru_fixed_path = ROOT / "experiments" / "modeling_ai_sequence_fusion_fixed_noe5" / "valid_probabilities.npy"
    raw_fixed = np.load(raw_fixed_path, allow_pickle=False).astype(np.float32)
    gru_fixed = np.load(gru_fixed_path, allow_pickle=False).astype(np.float32)
    if raw_fixed.shape != gru_fixed.shape or raw_fixed.shape != (len(fixed_index), N_CLASSES):
        raise ValueError("B43 fixed raw/GRU shape mismatch")

    policies = [(sim, au) for sim in SIM_TERMINAL_ALPHAS for au in AU_TERMINAL_ALPHAS]
    if BASE_POLICY not in policies:
        raise AssertionError("The exact B43 no-op policy must appear in the predeclared grid")
    name_of = {policy: policy_name(*policy) for policy in policies}

    base_full = terminal_adjusted_probability(raw, gru, source, terminal, BASE_POLICY)
    base_fixed = terminal_adjusted_probability(raw_fixed, gru_fixed, source[fixed_index], fixed_terminal, BASE_POLICY)
    baseline_full = metric_bundle(y, base_full, terminal, char_full)
    baseline_fixed = metric_bundle(y[fixed_index], base_fixed, fixed_terminal, char_fixed)
    # Exact historical public-B43 reconstruction from
    # public_feedback_ensemble_recheck/report.json (char20 route).
    expected = (0.7603385811033483, 0.7335225252513891, 0.7585157741730599, 0.7000338475808981)
    observed = (
        baseline_full["all_fallback_macro_f1"], baseline_full["terminal_with_char20_macro_f1"],
        baseline_fixed["all_fallback_macro_f1"], baseline_fixed["terminal_with_char20_macro_f1"],
    )
    if not np.allclose(observed, expected, rtol=0.0, atol=1.0e-12):
        raise ValueError(f"B43 exact-control mismatch: observed={observed}, expected={expected}")

    full_probability: dict[tuple[float, float], np.ndarray] = {}
    fixed_probability: dict[tuple[float, float], np.ndarray] = {}
    full_metric: dict[tuple[float, float], dict[str, float]] = {}
    fixed_metric: dict[tuple[float, float], dict[str, float]] = {}
    for policy in policies:
        full_probability[policy] = terminal_adjusted_probability(raw, gru, source, terminal, policy)
        fixed_probability[policy] = terminal_adjusted_probability(raw_fixed, gru_fixed, source[fixed_index], fixed_terminal, policy)
        full_metric[policy] = metric_bundle(y, full_probability[policy], terminal, char_full)
        fixed_metric[policy] = metric_bundle(y[fixed_index], fixed_probability[policy], fixed_terminal, char_fixed)

    static_positive = []
    for policy in policies:
        if policy == BASE_POLICY:
            continue
        full_delta = difference(full_metric[policy], baseline_full)
        fixed_delta = difference(fixed_metric[policy], baseline_fixed)
        if all(full_delta[key] > 0.0 and fixed_delta[key] > 0.0 for key in REQUIRED):
            static_positive.append(policy)

    folds = list(GroupKFold(n_splits=5).split(np.zeros(len(rows)), y, groups))
    fold_id = np.empty(len(rows), dtype=np.int8)
    for number, (_, held) in enumerate(folds):
        fold_id[held] = number
    nested = np.zeros_like(base_full)
    nested_choices = []
    for held_fold in range(5):
        selection = terminal & (fold_id != held_fold)
        def rank(policy: tuple[float, float]) -> tuple[float, float, float, float]:
            prediction = final_prediction(full_probability[policy], terminal, char_full)
            # Terminal post-char20 F1 is deployment-aligned; smaller terminal
            # movement is the conservative deterministic tiebreak.
            return (
                macro(y, prediction, selection),
                macro(y, full_probability[policy].argmax(axis=1), selection),
                -policy_distance(policy),
                -abs(policy[0] - BASE_POLICY[0]) - abs(policy[1] - BASE_POLICY[1]),
            )
        chosen = max(policies, key=rank)
        held = fold_id == held_fold
        nested[held] = full_probability[chosen][held]
        nested_choices.append({"held_fold": held_fold + 1, "policy": name_of[chosen]})
    counts = Counter(item["policy"] for item in nested_choices)
    consensus_name = max(
        counts,
        key=lambda value: (counts[value], -policy_distance(next(policy for policy, name in name_of.items() if name == value))),
    )
    consensus = next(policy for policy, name in name_of.items() if name == consensus_name)
    nested_full = metric_bundle(y, nested, terminal, char_full)
    consensus_fixed = fixed_metric[consensus]
    nested_delta_full = difference(nested_full, baseline_full)
    nested_delta_fixed = difference(consensus_fixed, baseline_fixed)
    nested_positive = all(nested_delta_full[key] > 0.0 and nested_delta_fixed[key] > 0.0 for key in REQUIRED)

    # Only candidates that are positive under both independent protocols are
    # included in the detailed result/bootstraps.  This keeps failed grid
    # points from being mistaken for submit-ready variants.
    base_full_prediction = final_prediction(base_full, terminal, char_full)
    base_fixed_prediction = final_prediction(base_fixed, fixed_terminal, char_fixed)
    eligible = []
    for policy in static_positive:
        candidate_full = final_prediction(full_probability[policy], terminal, char_full)
        candidate_fixed = final_prediction(fixed_probability[policy], fixed_terminal, char_fixed)
        eligible.append({
            "policy": name_of[policy],
            "full": full_metric[policy],
            "fixed": fixed_metric[policy],
            "delta_vs_b43": {
                "full": difference(full_metric[policy], baseline_full),
                "fixed": difference(fixed_metric[policy], baseline_fixed),
            },
            "session_bootstrap_terminal_char20": {
                "full": session_bootstrap(y, base_full_prediction, candidate_full, terminal),
                "fixed": session_bootstrap(y[fixed_index], base_fixed_prediction, candidate_fixed, fixed_terminal, seed=20260714),
            },
        })
    nested_record = None
    if nested_positive:
        nested_prediction = final_prediction(nested, terminal, char_full)
        consensus_prediction = final_prediction(fixed_probability[consensus], fixed_terminal, char_fixed)
        nested_record = {
            "consensus_policy": consensus_name,
            "choices": nested_choices,
            "full": nested_full,
            "fixed": consensus_fixed,
            "delta_vs_b43": {"full": nested_delta_full, "fixed": nested_delta_fixed},
            "session_bootstrap_terminal_char20": {
                "full": session_bootstrap(y, base_full_prediction, nested_prediction, terminal),
                "fixed": session_bootstrap(y[fixed_index], base_fixed_prediction, consensus_prediction, fixed_terminal, seed=20260714),
            },
        }

    report = {
        "scope": "Experiment only: terminal-aware selector, model/, script.py, B42.zip, baseline_submit.zip, and challenger ZIPs were not modified.",
        "contract": {
            "baseline": "Exact public old-B43 raw seed-42 selector plus old no-E5 GRU geometric SIM=.20/AU=.85.",
            "intervention": "Only supplied-session batch-terminal rows use a different predeclared source alpha; all non-terminal probabilities remain bitwise B43 geometry.",
            "downstream_order": "char20 remains terminal-last; later-history recovery appears only in the hybrid evaluation.",
            "forbidden": "No IDs as model features, target lookup, future-label input, terminal-aware selector, or fixed-label policy selection.",
        },
        "component_paths": {
            "full_raw_seed42": str(raw_path.relative_to(ROOT)),
            "full_old_noe5_gru": str(gru_path.relative_to(ROOT)),
            "fixed_raw_seed42": str(raw_fixed_path.relative_to(ROOT)),
            "fixed_old_noe5_gru": str(gru_fixed_path.relative_to(ROOT)),
            "char20": "experiments/terminal_rule_combinations/oof_candidates.npz",
        },
        "grid": {"sim_terminal_raw_alphas": list(SIM_TERMINAL_ALPHAS), "au_terminal_raw_alphas": list(AU_TERMINAL_ALPHAS), "policy_count": len(policies)},
        "counts": {"rows": int(len(rows)), "terminal": int(terminal.sum()), "common_train": int(len(common_index)), "fixed": int(len(fixed_index)), "fixed_terminal": int(fixed_terminal.sum())},
        "b43_control": {"full": baseline_full, "fixed": baseline_fixed},
        "positive_static_candidates": eligible,
        "nested_consensus_candidate": nested_record,
        "screen": {"static_positive_count": len(eligible), "nested_consensus_positive": bool(nested_positive), "nested_consensus_policy": consensus_name, "nested_choices": nested_choices},
        "decision": "Only entries in positive_static_candidates or nested_consensus_candidate passed strictly positive full and fixed metrics; an empty set means retain B43.",
        "seconds": round(time.time() - began, 2),
    }
    # Preserve compact auditable probability inputs for any later read-only
    # verification; no submission asset is written.
    np.savez_compressed(
        OUT / "predictions.npz", y=y, terminal=terminal, fixed_index=fixed_index,
        fixed_terminal=fixed_terminal, base_full=base_full, base_fixed=base_fixed,
        char_full=char_full, char_fixed=char_fixed,
    )
    (OUT / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "positive_static": [item["policy"] for item in eligible],
        "nested_consensus": consensus_name,
        "nested_positive": nested_positive,
        "seconds": report["seconds"],
    }, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
