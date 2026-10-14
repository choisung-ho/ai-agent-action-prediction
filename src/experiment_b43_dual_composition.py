#!/usr/bin/env python3
"""Small exact-B43 old/strict GRU composition audit.

Only four predeclared compositions are examined:

1. arithmetic old=.50 / strict=.50;
2. normalized geometric old=.50 / strict=.50;
3. normalized geometric old=.65 / strict=.35; and
4. strict-only.

The composition is followed by the unchanged B43 raw-XGB geometric fusion
(SIM=.20/AU=.85), then immutable char20 terminal override and the existing
future-history hybrid metric.  No temperature, logit bias, source gate, or
learned calibration is introduced.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from experiment_b43_adaptive_blend import (
    EPS,
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


BOOTSTRAP_REPLICATES = 2000


@dataclass(frozen=True)
class Composition:
    name: str
    kind: str
    strict_weight: float


COMPOSITIONS = (
    Composition("arithmetic_old0.50_strict0.50", "arithmetic", 0.50),
    Composition("normalized_geometric_old0.50_strict0.50", "geometric", 0.50),
    Composition("normalized_geometric_old0.65_strict0.35", "geometric", 0.35),
    Composition("strict_only", "arithmetic", 1.00),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="experiments/b43_dual_composition")
    parser.add_argument("--bootstrap-replicates", type=int, default=BOOTSTRAP_REPLICATES)
    return parser.parse_args()


def compose(old: np.ndarray, strict: np.ndarray, spec: Composition) -> np.ndarray:
    w = spec.strict_weight
    if spec.kind == "arithmetic":
        return ((1.0 - w) * old + w * strict).astype(np.float32)
    if spec.kind == "geometric":
        logits = (1.0 - w) * np.log(np.maximum(old, EPS)) + w * np.log(np.maximum(strict, EPS))
        logits -= logits.max(axis=1, keepdims=True)
        probability = np.exp(logits)
        return (probability / probability.sum(axis=1, keepdims=True)).astype(np.float32)
    raise ValueError(spec.kind)


def final_terminal(probability: np.ndarray, terminal: np.ndarray, char20: np.ndarray) -> np.ndarray:
    return labels_with_char(probability, terminal, char20)[1][terminal]


def all_eight_positive(record: dict[str, Any]) -> bool:
    return all(
        value > 0.0
        for side in record["delta_vs_b43"].values()
        for value in side.values()
    )


def strict_gate(record: dict[str, Any]) -> dict[str, Any]:
    full_boot = record["terminal_char20_session_bootstrap"]["full_outer_oof"]
    fixed_boot = record["terminal_char20_session_bootstrap"]["matching_seed42_fixed"]
    checks = {
        "all_eight_full_fixed_metrics_strictly_positive": all_eight_positive(record),
        "full_terminal_char20_bootstrap_p_ge_0_95": float(full_boot["p_delta_gt_zero"]) >= 0.95,
        "fixed_terminal_char20_bootstrap_p_ge_0_95": float(fixed_boot["p_delta_gt_zero"]) >= 0.95,
    }
    return {
        "checks": checks,
        "pass": bool(all(checks.values())),
        "reason": (
            "All eight B43 metrics improve and both terminal-char20 session bootstraps clear .95."
            if all(checks.values()) else
            "Reject: one or more full/fixed metric or session-bootstrap requirements is not met."
        ),
    }


def select_strongest(records: dict[str, dict[str, Any]]) -> tuple[str, dict[str, Any]]:
    """Select deterministically for a diagnostic fold table, never for deployment.

    A gate-passing row would win first.  When none passes, maximize the weakest
    of the eight deltas, then aggregate terminal-char delta, then bootstrap
    support.  This makes the one requested fold table auditable without
    relaxing the separate promotion gate.
    """
    def rank(item: tuple[str, dict[str, Any]]) -> tuple[float, float, float, float]:
        _, record = item
        values = [value for side in record["delta_vs_b43"].values() for value in side.values()]
        terminal_sum = (
            record["delta_vs_b43"]["full_outer_oof"]["terminal_with_char20_macro_f1"]
            + record["delta_vs_b43"]["matching_seed42_fixed"]["terminal_with_char20_macro_f1"]
        )
        bootstrap_min = min(
            float(record["terminal_char20_session_bootstrap"]["full_outer_oof"]["p_delta_gt_zero"]),
            float(record["terminal_char20_session_bootstrap"]["matching_seed42_fixed"]["p_delta_gt_zero"]),
        )
        return (float(record["strict_gate"]["pass"]), min(values), terminal_sum, bootstrap_min)

    return max(records.items(), key=rank)


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
    base_full_probability = geometric(data["raw"], data["old"], source_alpha(data["source"]))
    base_fixed_probability = geometric(
        data["fixed_raw"], data["fixed_old"], source_alpha(data["fixed_source"]),
    )
    baseline_full = metric_bundle(data["y"], base_full_probability, data["terminal"], data["char20"])
    baseline_fixed = metric_bundle(
        data["fixed_y"], base_fixed_probability, data["fixed_terminal"], data["fixed_char20"],
    )
    check_baseline(baseline_full, EXPECTED_B43["full"], "full")
    check_baseline(baseline_fixed, EXPECTED_B43["fixed"], "matching_seed42_fixed")
    base_full_terminal = final_terminal(base_full_probability, data["terminal"], data["char20"])
    base_fixed_terminal = final_terminal(base_fixed_probability, data["fixed_terminal"], data["fixed_char20"])

    records: dict[str, dict[str, Any]] = {}
    candidate_full_probability: dict[str, np.ndarray] = {}
    candidate_fixed_probability: dict[str, np.ndarray] = {}
    for spec in COMPOSITIONS:
        full_gru = compose(data["old"], data["strict"], spec)
        fixed_gru = compose(data["fixed_old"], data["fixed_strict"], spec)
        full_probability = geometric(data["raw"], full_gru, source_alpha(data["source"]))
        fixed_probability = geometric(data["fixed_raw"], fixed_gru, source_alpha(data["fixed_source"]))
        candidate_full_probability[spec.name] = full_probability
        candidate_fixed_probability[spec.name] = fixed_probability
        full_metric = metric_bundle(data["y"], full_probability, data["terminal"], data["char20"])
        fixed_metric = metric_bundle(
            data["fixed_y"], fixed_probability, data["fixed_terminal"], data["fixed_char20"],
        )
        candidate_full_terminal = final_terminal(full_probability, data["terminal"], data["char20"])
        candidate_fixed_terminal = final_terminal(
            fixed_probability, data["fixed_terminal"], data["fixed_char20"],
        )
        record: dict[str, Any] = {
            "composition": {"kind": spec.kind, "old_weight": 1.0 - spec.strict_weight, "strict_weight": spec.strict_weight},
            "full_outer_oof": full_metric,
            "matching_seed42_fixed": fixed_metric,
            "delta_vs_b43": {
                "full_outer_oof": metric_delta(full_metric, baseline_full),
                "matching_seed42_fixed": metric_delta(fixed_metric, baseline_fixed),
            },
            "terminal_char20_session_bootstrap": {
                "full_outer_oof": session_bootstrap(
                    data["y"][data["terminal"]], base_full_terminal, candidate_full_terminal,
                    data["groups"][data["terminal"]], args.bootstrap_replicates,
                ),
                "matching_seed42_fixed": session_bootstrap(
                    data["fixed_y"][data["fixed_terminal"]], base_fixed_terminal, candidate_fixed_terminal,
                    data["fixed_groups"][data["fixed_terminal"]], args.bootstrap_replicates,
                ),
            },
        }
        record["strict_gate"] = strict_gate(record)
        records[spec.name] = record

    strongest_name, strongest = select_strongest(records)
    strongest_full_probability = candidate_full_probability[strongest_name]
    strongest["full_outer_session_fold_deltas"] = fold_deltas(
        data["y"], base_full_probability, strongest_full_probability,
        data["terminal"], data["char20"], data["fold"],
    )
    any_pass = any(record["strict_gate"]["pass"] for record in records.values())
    report = {
        "scope": "Experiment-only. No model/, script.py, B42.zip, baseline_submit.zip, or submission artifact was changed.",
        "reference": "Exact B43 raw seed42 + original no-E5 GRU, geometric raw fusion SIM=.20/AU=.85, immutable char20 last.",
        "provenance": provenance,
        "composition_contract": {
            "allowed": [
                "arithmetic old/strict probability mixing",
                "normalized geometric old/strict probability mixing",
            ],
            "excluded": "Temperature scaling, extra logit calibration, source routing, thresholds, and learned selection are excluded.",
            "compositions": [spec.name for spec in COMPOSITIONS],
            "raw_fusion": "unchanged B43 geometric weights SIM=.20/AU=.85",
            "terminal_order": "fallback composition -> raw/GRU fusion -> argmax -> char20 terminal override -> future-history only in hybrid metric",
        },
        "b43_baseline": {"full_outer_oof": baseline_full, "matching_seed42_fixed": baseline_fixed},
        "candidates": records,
        "strongest_diagnostic_candidate": {
            "name": strongest_name,
            "selection_rule": "Gate-pass first; otherwise maximize worst of the eight metric deltas, then terminal-char total and weaker bootstrap support.",
            "record": strongest,
        },
        "promotion_gate": {
            "required_metric_keys": list(METRIC_KEYS),
            "requires": "All eight full/fixed metrics strictly positive and both terminal-char20 session-bootstrap p(delta>0) >= .95.",
            "pass": any_pass,
            "passing_candidates": [name for name, record in records.items() if record["strict_gate"]["pass"]],
            "reason": "No live/ZIP candidate is built unless a predeclared composition passes the strict gate.",
        },
        "submission_action": {"candidate_created": False, "zip_updated": False},
    }
    (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    np.savez_compressed(
        out / "predictions.npz",
        y=data["y"], terminal=data["terminal"], groups=data["groups"], fold=data["fold"],
        base_full_probability=base_full_probability,
        base_fixed_probability=base_fixed_probability,
        **{f"full_{name.replace('.', 'p')}": value for name, value in candidate_full_probability.items()},
        **{f"fixed_{name.replace('.', 'p')}": value for name, value in candidate_fixed_probability.items()},
    )
    print(json.dumps({
        "strict_gate_pass": any_pass,
        "strongest": strongest_name,
        "output": str(out),
        "summary": {
            name: {
                "full_terminal_char_delta": record["delta_vs_b43"]["full_outer_oof"]["terminal_with_char20_macro_f1"],
                "fixed_terminal_char_delta": record["delta_vs_b43"]["matching_seed42_fixed"]["terminal_with_char20_macro_f1"],
                "full_bootstrap_p": record["terminal_char20_session_bootstrap"]["full_outer_oof"]["p_delta_gt_zero"],
                "fixed_bootstrap_p": record["terminal_char20_session_bootstrap"]["matching_seed42_fixed"]["p_delta_gt_zero"],
            }
            for name, record in records.items()
        },
    }, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()

