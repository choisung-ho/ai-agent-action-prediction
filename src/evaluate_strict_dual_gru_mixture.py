#!/usr/bin/env python3
"""Audit a compact old/no-E5 + strict-distill/no-E5 GRU probability mixture.

This is a companion to the strict calibration audit.  It treats the symmetric
50/50 mixture as a predeclared deployable candidate, then reports (but does
not deploy from) a nested outer-fold weight-selection diagnostic.  Raw XGB is
always the strict 2032-feature selector on the common fixed split.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from experiment_strict_blend_calibration import (
    BASE_ALPHA,
    baseline_probability,
    load_data,
    macro,
    metric_row,
)


WEIGHTS = (0.0, 0.35, 0.50, 0.65, 0.70, 1.0)  # weight on strict-distilled student


def mixed(old: np.ndarray, strict: np.ndarray, weight: float) -> np.ndarray:
    return ((1.0 - weight) * old + weight * strict).astype(np.float32)


def delta(candidate: dict[str, float], baseline: dict[str, float]) -> dict[str, float]:
    return {
        "full_terminal": candidate["full_nested_group_oof"]["terminal_fallback_macro_f1"] - baseline["full_nested_group_oof"]["terminal_fallback_macro_f1"],
        "full_hybrid": candidate["full_nested_group_oof"]["hybrid_future_history_plus_char20_macro_f1"] - baseline["full_nested_group_oof"]["hybrid_future_history_plus_char20_macro_f1"],
        "fixed_terminal": candidate["common_fixed_validation"]["terminal_fallback_macro_f1"] - baseline["common_fixed_validation"]["terminal_fallback_macro_f1"],
        "fixed_hybrid": candidate["common_fixed_validation"]["hybrid_future_history_plus_char20_macro_f1"] - baseline["common_fixed_validation"]["hybrid_future_history_plus_char20_macro_f1"],
    }


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    out = root / "experiments/strict_blend_calibration"
    out.mkdir(parents=True, exist_ok=True)
    data = load_data(root)
    y = data["y"]; source = data["source"]; terminal = data["terminal"]
    xgb = data["xgb"]; strict = data["gru"]; old = data["old_gru"]; fold = data["fold"]
    char = data["char"]
    train = data["train_index"]
    strict_train_xgb = data["strict_train_xgb"]
    fixed_y = y[data["valid_index"]]; fixed_source = source[data["valid_index"]]
    fixed_xgb = data["fixed_xgb"]; fixed_strict = data["fixed_gru"]; fixed_old = data["fixed_old_gru"]
    fixed_terminal = data["fixed_terminal"]; fixed_char = data["fixed_char"]

    baseline = {
        "full_nested_group_oof": metric_row(y, baseline_probability(xgb, strict, source), terminal, char),
        "common_fixed_validation": metric_row(fixed_y, baseline_probability(fixed_xgb, fixed_strict, fixed_source), fixed_terminal, fixed_char),
    }

    static: dict[str, dict] = {}
    for weight in WEIGHTS:
        candidate = {
            "full_nested_group_oof": metric_row(y, baseline_probability(xgb, mixed(old, strict, weight), source), terminal, char),
            "common_fixed_validation": metric_row(
                fixed_y, baseline_probability(fixed_xgb, mixed(fixed_old, fixed_strict, weight), fixed_source),
                fixed_terminal, fixed_char,
            ),
        }
        static[f"strict_weight_{weight:.2f}"] = {**candidate, "delta_vs_strict_only": delta(candidate, baseline)}

    # This diagnostic selects w from other outer session folds only; the held
    # fold labels never enter the selection.  The fixed counterpart similarly
    # looks only at common-train terminal labels.  It is evidence for the
    # family, while the predeclared symmetric w=.50 is the portable candidate.
    nested_probability = np.zeros_like(xgb)
    outer_records: list[dict] = []
    for held_fold in sorted(np.unique(fold).tolist()):
        held = fold == held_fold
        train_mask = terminal & ~held
        candidates = []
        for weight in WEIGHTS:
            probability = baseline_probability(xgb, mixed(old, strict, weight), source)
            candidates.append((macro(y, probability.argmax(axis=1), train_mask), weight))
        score, picked = max(candidates, key=lambda pair: (pair[0], -abs(pair[1] - 0.5)))
        probability = baseline_probability(xgb, mixed(old, strict, picked), source)
        nested_probability[held] = probability[held]
        outer_records.append({"held_outer_fold": int(held_fold + 1), "weight": picked, "outer_train_terminal_macro_f1": score})
    nested = metric_row(y, nested_probability, terminal, char)

    train_terminal = terminal[train]
    common_grid = []
    for weight in WEIGHTS:
        probability = baseline_probability(strict_train_xgb, mixed(old[train], strict[train], weight), source[train])
        common_grid.append({"weight": weight, "common_train_terminal_macro_f1": macro(y[train], probability.argmax(axis=1), train_terminal)})
    common_choice = max(common_grid, key=lambda row: (row["common_train_terminal_macro_f1"], -abs(row["weight"] - 0.5)))
    fixed_weight = float(common_choice["weight"])
    fixed_selected = metric_row(
        fixed_y, baseline_probability(fixed_xgb, mixed(fixed_old, fixed_strict, fixed_weight), fixed_source),
        fixed_terminal, fixed_char,
    )

    symmetric = static["strict_weight_0.50"]
    symmetric_safe = all(value > 0.0 for value in symmetric["delta_vs_strict_only"].values())
    report = {
        "scope": "Experiment-only dual-GRU mixture audit; no live model, script, or ZIP is changed.",
        "protocol": {
            "full": "Static w=.50 is predeclared. Separately, each outer fold's diagnostic weight selection sees terminal labels only from the other four session folds.",
            "fixed": "Static w=.50 uses no fixed labels. The diagnostic weight selection scores only strict common-train terminal rows before fixed validation is read.",
            "strict_raw_xgb": "Common fixed side uses 2032-width strict common-train raw selector probabilities, random_state=808; current live selector uses random_state=42.",
            "char20": "Applied only after each base blend on terminal rows; never used to choose mixture weights.",
        },
        "baseline_strict_student_only": baseline,
        "static_weight_grid": static,
        "nested_outer_weight_diagnostic": {"metrics": nested, "folds": outer_records},
        "common_train_weight_diagnostic": {"grid": common_grid, "chosen": common_choice, "fixed_metrics": fixed_selected},
        "recommendation": (
            {
                "promote_for_integration_review": True,
                "portable_policy": {"old_gru_weight": 0.5, "strict_distill_gru_weight": 0.5},
                "reason": "The fixed symmetric mix improves terminal fallback and future-history+char20 hybrid on both full GroupOOF and strict fixed validation.",
            }
            if symmetric_safe else
            {"promote_for_integration_review": False, "reason": "The predeclared symmetric mix did not improve every required two-split metric."}
        ),
    }
    (out / "dual_gru_mixture_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "symmetric_delta": symmetric["delta_vs_strict_only"],
        "symmetric_safe": symmetric_safe,
        "nested_terminal": nested["terminal_fallback_macro_f1"],
        "fixed_selected_weight": fixed_weight,
        "fixed_selected_terminal": fixed_selected["terminal_fallback_macro_f1"],
    }, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
