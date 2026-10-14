#!/usr/bin/env python3
"""Exact-B43 source-routed arithmetic old/strict GRU mixture audit.

The only intervention is a fixed strict-distill GRU probability weight per
source: SIM and AU each choose from {0.00, 0.25, 0.50}, giving nine tiny
portable policies.  B43's seed-42 raw selector, geometric SIM=.20/AU=.85
fusion, immutable terminal char20 overlay, and future-history reporting order
are otherwise unchanged.

This script is experiment-only.  It writes only beneath the requested
``experiments/b43_dual_source_gate`` directory and never mutates model/,
script.py, or any submission ZIP.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import f1_score

from experiment_b43_adaptive_blend import (
    EXPECTED_B43,
    METRIC_KEYS,
    check_baseline,
    fold_deltas,
    geometric,
    labels_with_char,
    load_exact_b43,
    metric_bundle,
    metric_delta,
    session_bootstrap,
    source_alpha,
)


WEIGHTS = (0.00, 0.25, 0.50)
BOOTSTRAP_REPLICATES = 2000


@dataclass(frozen=True)
class SourcePolicy:
    sim_strict_weight: float
    au_strict_weight: float

    @property
    def key(self) -> str:
        return f"sim_strict={self.sim_strict_weight:.2f}|au_strict={self.au_strict_weight:.2f}"

    @property
    def complexity(self) -> tuple[float, float, float]:
        # Prefer exact B43 and then the lower total strict contribution when
        # selection scores tie exactly.
        return (
            self.sim_strict_weight + self.au_strict_weight,
            max(self.sim_strict_weight, self.au_strict_weight),
            self.au_strict_weight,
        )


POLICIES = tuple(SourcePolicy(sim, au) for sim in WEIGHTS for au in WEIGHTS)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="experiments/b43_dual_source_gate")
    parser.add_argument("--bootstrap-replicates", type=int, default=BOOTSTRAP_REPLICATES)
    return parser.parse_args()


def source_weight(source: np.ndarray, policy: SourcePolicy) -> np.ndarray:
    if not np.all(np.isin(source, ("sim", "au"))):
        raise ValueError(f"Unknown source: {np.unique(source).tolist()}")
    return np.where(source == "sim", policy.sim_strict_weight, policy.au_strict_weight).astype(np.float32)


def mixed_gru(old: np.ndarray, strict: np.ndarray, source: np.ndarray, policy: SourcePolicy) -> np.ndarray:
    strict_weight = source_weight(source, policy)[:, None]
    return ((1.0 - strict_weight) * old + strict_weight * strict).astype(np.float32)


def fallback_metrics(y: np.ndarray, probability: np.ndarray, terminal: np.ndarray) -> dict[str, float]:
    prediction = probability.argmax(axis=1)
    return {
        "terminal_fallback_macro_f1": float(f1_score(
            y[terminal], prediction[terminal], labels=np.arange(14), average="macro", zero_division=0,
        )),
        "all_fallback_macro_f1": float(f1_score(
            y, prediction, labels=np.arange(14), average="macro", zero_division=0,
        )),
    }


def choose_policy(
    y: np.ndarray,
    probability: dict[str, np.ndarray],
    terminal: np.ndarray,
    train: np.ndarray,
) -> tuple[SourcePolicy, list[dict[str, Any]]]:
    """Choose from nine static policies using no held-fold labels.

    Char20 deliberately remains out of this selection.  Its outer-fold rule
    artifact is immutable and evaluates only after the fallback mixture,
    avoiding any chance that a lexical overlay changes source-weight choice.
    """
    rows: list[dict[str, Any]] = []
    for policy in POLICIES:
        row = {"policy": policy, **fallback_metrics(y[train], probability[policy.key][train], terminal[train])}
        rows.append(row)
    chosen = max(
        rows,
        key=lambda row: (
            row["terminal_fallback_macro_f1"],
            row["all_fallback_macro_f1"],
            tuple(-value for value in row["policy"].complexity),
        ),
    )["policy"]
    return chosen, rows


def modal_policy(choices: list[SourcePolicy]) -> tuple[SourcePolicy, dict[str, int]]:
    counts = Counter(choice.key for choice in choices)
    high = max(counts.values())
    selected = next(policy for policy in POLICIES if counts.get(policy.key, 0) == high)
    return selected, {policy.key: int(counts.get(policy.key, 0)) for policy in POLICIES}


def nested_source_policy(
    y: np.ndarray,
    probability: dict[str, np.ndarray],
    terminal: np.ndarray,
    fold: np.ndarray,
) -> tuple[np.ndarray, list[dict[str, Any]], SourcePolicy, dict[str, int]]:
    selected_probability = np.zeros_like(next(iter(probability.values())))
    choices: list[SourcePolicy] = []
    records: list[dict[str, Any]] = []
    for held_fold in sorted(np.unique(fold).tolist()):
        held = fold == held_fold
        choice, train_rows = choose_policy(y, probability, terminal, ~held)
        selected_probability[held] = probability[choice.key][held]
        choices.append(choice)
        records.append({
            "held_outer_session_fold": int(held_fold + 1),
            "selected_without_held_labels": choice.key,
            "other_four_fold_fallback_metrics": [
                {
                    "policy": row["policy"].key,
                    "terminal_fallback_macro_f1": row["terminal_fallback_macro_f1"],
                    "all_fallback_macro_f1": row["all_fallback_macro_f1"],
                }
                for row in train_rows
            ],
        })
    consensus, counts = modal_policy(choices)
    return selected_probability, records, consensus, counts


def terminal_char_prediction(probability: np.ndarray, terminal: np.ndarray, char20: np.ndarray) -> np.ndarray:
    return labels_with_char(probability, terminal, char20)[1]


def fold_delta_summary(rows: list[dict[str, Any]], metric: str) -> dict[str, float | int]:
    values = np.asarray([float(row["delta"][metric]) for row in rows], dtype=np.float64)
    return {
        "mean": float(values.mean()),
        "std": float(values.std(ddof=0)),
        "minimum": float(values.min()),
        "maximum": float(values.max()),
        "positive_fold_count": int(np.sum(values > 0.0)),
    }


def main() -> None:
    args = parse_args()
    if args.bootstrap_replicates <= 0:
        raise ValueError("--bootstrap-replicates must be positive")
    root = Path(__file__).resolve().parents[1]
    out = Path(args.out_dir)
    if not out.is_absolute():
        out = root / out
    out = out.resolve()
    model_dir = (root / "model").resolve()
    if out == model_dir or model_dir in out.parents:
        raise ValueError("Experiment output must not be inside model/")
    out.mkdir(parents=True, exist_ok=True)

    data, provenance = load_exact_b43(root)
    b43_full_probability = geometric(data["raw"], data["old"], source_alpha(data["source"]))
    b43_fixed_probability = geometric(
        data["fixed_raw"], data["fixed_old"], source_alpha(data["fixed_source"]),
    )
    b43_full = metric_bundle(data["y"], b43_full_probability, data["terminal"], data["char20"])
    b43_fixed = metric_bundle(
        data["fixed_y"], b43_fixed_probability, data["fixed_terminal"], data["fixed_char20"],
    )
    check_baseline(b43_full, EXPECTED_B43["full"], "full")
    check_baseline(b43_fixed, EXPECTED_B43["fixed"], "matching_seed42_fixed")
    b43_full_terminal = terminal_char_prediction(
        b43_full_probability, data["terminal"], data["char20"],
    )[data["terminal"]]
    b43_fixed_terminal = terminal_char_prediction(
        b43_fixed_probability, data["fixed_terminal"], data["fixed_char20"],
    )[data["fixed_terminal"]]

    full_probability: dict[str, np.ndarray] = {}
    fixed_probability: dict[str, np.ndarray] = {}
    static_rows: dict[str, dict[str, Any]] = {}
    for policy in POLICIES:
        full_gru = mixed_gru(data["old"], data["strict"], data["source"], policy)
        fixed_gru = mixed_gru(data["fixed_old"], data["fixed_strict"], data["fixed_source"], policy)
        full_probability[policy.key] = geometric(data["raw"], full_gru, source_alpha(data["source"]))
        fixed_probability[policy.key] = geometric(
            data["fixed_raw"], fixed_gru, source_alpha(data["fixed_source"]),
        )
        full_metric = metric_bundle(data["y"], full_probability[policy.key], data["terminal"], data["char20"])
        fixed_metric = metric_bundle(
            data["fixed_y"], fixed_probability[policy.key], data["fixed_terminal"], data["fixed_char20"],
        )
        static_rows[policy.key] = {
            "sim_strict_weight": policy.sim_strict_weight,
            "au_strict_weight": policy.au_strict_weight,
            "full_outer_oof": full_metric,
            "matching_seed42_fixed": fixed_metric,
            "delta_vs_b43": {
                "full_outer_oof": metric_delta(full_metric, b43_full),
                "matching_seed42_fixed": metric_delta(fixed_metric, b43_fixed),
            },
        }

    # The original global 50:50 pair is retained only as the comparison arm:
    # source routing must improve its outer-fold/terminal stability, not just
    # find a favorable fixed point through a source split.
    uniform_50 = SourcePolicy(0.50, 0.50)
    uniform_50_full_probability = full_probability[uniform_50.key]
    uniform_50_fixed_probability = fixed_probability[uniform_50.key]
    uniform_50_full = metric_bundle(
        data["y"], uniform_50_full_probability, data["terminal"], data["char20"],
    )
    uniform_50_fixed = metric_bundle(
        data["fixed_y"], uniform_50_fixed_probability, data["fixed_terminal"], data["fixed_char20"],
    )
    uniform_50_full_terminal = terminal_char_prediction(
        uniform_50_full_probability, data["terminal"], data["char20"],
    )[data["terminal"]]
    uniform_50_fixed_terminal = terminal_char_prediction(
        uniform_50_fixed_probability, data["fixed_terminal"], data["fixed_char20"],
    )[data["fixed_terminal"]]
    uniform_50_folds = fold_deltas(
        data["y"], b43_full_probability, uniform_50_full_probability,
        data["terminal"], data["char20"], data["fold"],
    )
    uniform_50_bootstrap = {
        "full_outer_oof_terminal_char20": session_bootstrap(
            data["y"][data["terminal"]], b43_full_terminal, uniform_50_full_terminal,
            data["groups"][data["terminal"]], args.bootstrap_replicates,
        ),
        "matching_seed42_fixed_terminal_char20": session_bootstrap(
            data["fixed_y"][data["fixed_terminal"]], b43_fixed_terminal, uniform_50_fixed_terminal,
            data["fixed_groups"][data["fixed_terminal"]], args.bootstrap_replicates,
        ),
    }

    selected_full_probability, outer_choices, consensus, counts = nested_source_policy(
        data["y"], full_probability, data["terminal"], data["fold"],
    )
    # The fixed protocol receives the modal policy selected exclusively by
    # outer session folds. Fixed labels never participate in source-weight
    # selection; the fixed arrays themselves are the matching seed-42 path.
    selected_fixed_probability = fixed_probability[consensus.key]
    candidate_full = metric_bundle(
        data["y"], selected_full_probability, data["terminal"], data["char20"],
    )
    candidate_fixed = metric_bundle(
        data["fixed_y"], selected_fixed_probability, data["fixed_terminal"], data["fixed_char20"],
    )
    full_delta = metric_delta(candidate_full, b43_full)
    fixed_delta = metric_delta(candidate_fixed, b43_fixed)

    candidate_full_terminal = terminal_char_prediction(
        selected_full_probability, data["terminal"], data["char20"],
    )[data["terminal"]]
    candidate_fixed_terminal = terminal_char_prediction(
        selected_fixed_probability, data["fixed_terminal"], data["fixed_char20"],
    )[data["fixed_terminal"]]
    bootstrap = {
        "full_outer_selected_terminal_char20": session_bootstrap(
            data["y"][data["terminal"]], b43_full_terminal, candidate_full_terminal,
            data["groups"][data["terminal"]], args.bootstrap_replicates,
        ),
        "matching_seed42_fixed_terminal_char20": session_bootstrap(
            data["fixed_y"][data["fixed_terminal"]], b43_fixed_terminal, candidate_fixed_terminal,
            data["fixed_groups"][data["fixed_terminal"]], args.bootstrap_replicates,
        ),
    }
    fold_rows = fold_deltas(
        data["y"], b43_full_probability, selected_full_probability,
        data["terminal"], data["char20"], data["fold"],
    )
    all_positive = all(value > 0.0 for value in full_delta.values()) and all(
        value > 0.0 for value in fixed_delta.values()
    )
    source_vs_uniform = {
        "full_metric_delta": metric_delta(candidate_full, uniform_50_full),
        "fixed_metric_delta": metric_delta(candidate_fixed, uniform_50_fixed),
        "terminal_char20_outer_fold_summary_vs_b43": {
            "source_routed": fold_delta_summary(fold_rows, "terminal_with_char20_macro_f1"),
            "uniform_50_50": fold_delta_summary(uniform_50_folds, "terminal_with_char20_macro_f1"),
        },
        "terminal_char20_bootstrap_p_delta_gt_zero_vs_b43": {
            "source_routed_full": bootstrap["full_outer_selected_terminal_char20"]["p_delta_gt_zero"],
            "uniform_50_50_full": uniform_50_bootstrap["full_outer_oof_terminal_char20"]["p_delta_gt_zero"],
            "source_routed_fixed": bootstrap["matching_seed42_fixed_terminal_char20"]["p_delta_gt_zero"],
            "uniform_50_50_fixed": uniform_50_bootstrap["matching_seed42_fixed_terminal_char20"]["p_delta_gt_zero"],
        },
    }
    source_vs_uniform["improves_uniform_50_50"] = bool(
        source_vs_uniform["full_metric_delta"]["terminal_with_char20_macro_f1"] > 0.0
        and source_vs_uniform["terminal_char20_outer_fold_summary_vs_b43"]["source_routed"]["std"]
        < source_vs_uniform["terminal_char20_outer_fold_summary_vs_b43"]["uniform_50_50"]["std"]
        and source_vs_uniform["terminal_char20_bootstrap_p_delta_gt_zero_vs_b43"]["source_routed_full"]
        > source_vs_uniform["terminal_char20_bootstrap_p_delta_gt_zero_vs_b43"]["uniform_50_50_full"]
    )

    report: dict[str, Any] = {
        "scope": "Experiment-only. No model/, script.py, B42.zip, baseline_submit.zip, or other submission asset was changed.",
        "reference": "Exact B43: seed42 raw XGB + original no-E5 GRU geometric SIM=.20/AU=.85, then immutable char20 final terminal overlay.",
        "provenance": provenance,
        "policy_family": {
            "definition": "GRU = (1-w_source)*old_noE5 + w_source*strict_distill_noE5 before unchanged B43 geometric raw/GRU fusion.",
            "sim_strict_weight_options": list(WEIGHTS),
            "au_strict_weight_options": list(WEIGHTS),
            "policy_count": len(POLICIES),
            "char20": "Applied only after fallback argmax; it does not participate in policy selection.",
        },
        "b43_baseline": {"full_outer_oof": b43_full, "matching_seed42_fixed": b43_fixed},
        "static_nine_policy_table": static_rows,
        "uniform_50_50_comparison_arm": {
            "policy": uniform_50.key,
            "full_outer_oof": uniform_50_full,
            "matching_seed42_fixed": uniform_50_fixed,
            "delta_vs_b43": {
                "full_outer_oof": metric_delta(uniform_50_full, b43_full),
                "matching_seed42_fixed": metric_delta(uniform_50_fixed, b43_fixed),
            },
            "full_outer_session_fold_deltas": uniform_50_folds,
            "terminal_char20_session_bootstrap": uniform_50_bootstrap,
        },
        "outer_nested_source_policy": {
            "selection_rule": (
                "For each held whole-session GroupKFold fold, choose one of the nine source-only fixed policies "
                "using terminal fallback F1 then all fallback F1 only on the other four folds."
            ),
            "outer_choices": outer_choices,
            "selection_counts": counts,
            "fixed_modal_policy": consensus.key,
            "fixed_modal_policy_weights": {
                "sim_strict_weight": consensus.sim_strict_weight,
                "au_strict_weight": consensus.au_strict_weight,
            },
            "matching_fixed_contract": (
                "Fixed labels are not used to choose the source policy; the modal outer policy is evaluated once using "
                "matching seed42 raw, B42_9-common old GRU, strict-distill fixed GRU, and char20 fixed artifacts."
            ),
            "full_outer_selected": candidate_full,
            "matching_seed42_fixed_modal": candidate_fixed,
            "delta_vs_b43": {"full_outer_selected": full_delta, "matching_seed42_fixed_modal": fixed_delta},
            "full_outer_session_fold_deltas": fold_rows,
            "terminal_char20_session_bootstrap": bootstrap,
            "source_routing_vs_uniform_50_50": source_vs_uniform,
        },
        "promotion_gate": {
            "all_required_metrics": list(METRIC_KEYS),
            "full_all_four_strictly_positive": all(value > 0.0 for value in full_delta.values()),
            "fixed_all_four_strictly_positive": all(value > 0.0 for value in fixed_delta.values()),
            "pass": all_positive,
            "reason": (
                "All four B43 metrics improve on both outer-nested and matching fixed protocols."
                if all_positive else
                "At least one required full/fixed B43 metric is non-positive; do not create or modify a submission candidate."
            ),
        },
        "submission_action": {"candidate_created": False, "zip_updated": False},
    }
    (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    np.savez_compressed(
        out / "predictions.npz",
        y=data["y"], terminal=data["terminal"], groups=data["groups"], fold=data["fold"],
        b43_full_probability=b43_full_probability,
        selected_full_probability=selected_full_probability,
        fixed_y=data["fixed_y"], fixed_terminal=data["fixed_terminal"], fixed_groups=data["fixed_groups"],
        b43_fixed_probability=b43_fixed_probability,
        selected_fixed_probability=selected_fixed_probability,
        char20=data["char20"], fixed_char20=data["fixed_char20"],
    )
    print(json.dumps({
        "promotion_pass": all_positive,
        "fixed_modal_policy": consensus.key,
        "full_delta": full_delta,
        "fixed_delta": fixed_delta,
        "output": str(out),
    }, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
