#!/usr/bin/env python3
"""Audit a second strictly-trained no-E5 GRU seed without touching live files.

The alternate OOF matrix must come from an outer nested run whose teacher
targets were constructed inside each outer-train partition.  This script only
mixes already-held-out probabilities, and it makes the terminal char20 rule
the last operation just as the submission does.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from experiment_strict_blend_calibration import EPS, load_data, metric_row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--alternate-oof", required=True)
    parser.add_argument("--alternate-fold", required=True)
    parser.add_argument("--alternate-fixed", required=True)
    parser.add_argument("--out-dir", default="experiments/strict_seed_ensemble")
    parser.add_argument("--full-xgb", default="experiments/xgb_robust_ensemble/full_oof_raw_seed42.npy",
                        help="Seed-42 raw XGB strict full GroupOOF probabilities aligned to the GRU OOF.")
    parser.add_argument("--fixed-xgb", default="experiments/xgb_robust_ensemble/strict_fixed_raw_seed42.npy",
                        help="Matching seed-42 raw XGB probabilities fitted only on common-train.")
    parser.add_argument("--sim-raw-alpha", type=float, default=0.25)
    parser.add_argument("--au-raw-alpha", type=float, default=0.85)
    return parser.parse_args()


def normalized_geometric(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    logits = 0.5 * np.log(np.maximum(first, EPS)) + 0.5 * np.log(np.maximum(second, EPS))
    logits -= logits.max(axis=1, keepdims=True)
    value = np.exp(logits)
    return (value / value.sum(axis=1, keepdims=True)).astype(np.float32)


def source_raw_blend(xgb: np.ndarray, gru: np.ndarray, source: np.ndarray, sim_alpha: float, au_alpha: float) -> np.ndarray:
    alpha = np.where(source == "sim", sim_alpha, au_alpha).astype(np.float64)[:, None]
    logits = alpha * np.log(np.maximum(xgb, EPS)) + (1.0 - alpha) * np.log(np.maximum(gru, EPS))
    logits -= logits.max(axis=1, keepdims=True)
    value = np.exp(logits)
    return (value / value.sum(axis=1, keepdims=True)).astype(np.float32)


def difference(candidate: dict[str, float], baseline: dict[str, float]) -> dict[str, float]:
    return {key: float(candidate[key] - baseline[key]) for key in candidate}


def protocol_rows(
    y: np.ndarray, xgb: np.ndarray, current: np.ndarray, alternate: np.ndarray,
    source: np.ndarray, terminal: np.ndarray, char: np.ndarray, sim_alpha: float, au_alpha: float,
) -> dict[str, dict[str, float]]:
    students = {
        "current_strict_seed": current,
        "alternate_strict_seed": alternate,
        "arithmetic_equal_gru": ((current + alternate) * 0.5).astype(np.float32),
        "geometric_equal_gru": normalized_geometric(current, alternate),
    }
    result: dict[str, dict[str, float]] = {}
    for name, gru in students.items():
        result[name] = metric_row(y, source_raw_blend(xgb, gru, source, sim_alpha, au_alpha), terminal, char)
    return result


def gate(full: dict[str, float], fixed: dict[str, float], baseline_full: dict[str, float], baseline_fixed: dict[str, float]) -> dict[str, object]:
    # The requested acceptance criterion is deployment-aligned: all-row
    # fallback and the char20-final terminal path must each improve on both
    # independent session protocols.  Other movements remain visible.
    required = ("all_fallback_macro_f1", "terminal_with_char20_macro_f1")
    full_delta = difference(full, baseline_full)
    fixed_delta = difference(fixed, baseline_fixed)
    return {
        "full_delta": full_delta,
        "fixed_delta": fixed_delta,
        "passes_two_protocol_all_plus_terminal": all(full_delta[key] > 0.0 and fixed_delta[key] > 0.0 for key in required),
        "required_metrics": list(required),
    }


def main() -> None:
    args = parse_args()
    if not (0.0 < args.sim_raw_alpha < 1.0 and 0.0 < args.au_raw_alpha < 1.0):
        raise ValueError("Both source raw alphas must be in (0, 1)")
    root = Path(__file__).resolve().parents[1]
    data = load_data(root)
    alternate_oof = np.load(root / args.alternate_oof, allow_pickle=False).astype(np.float32)
    alternate_fold = np.load(root / args.alternate_fold, allow_pickle=False).astype(np.int8)
    alternate_fixed = np.load(root / args.alternate_fixed, allow_pickle=False).astype(np.float32)
    current = data["gru"]
    fixed_current = data["fixed_gru"]
    if alternate_oof.shape != current.shape or alternate_fixed.shape != fixed_current.shape:
        raise ValueError(f"Alternate shapes must be {current.shape} / {fixed_current.shape}, got {alternate_oof.shape} / {alternate_fixed.shape}")
    if not np.array_equal(alternate_fold, data["fold"]):
        raise ValueError("Alternate OOF GroupKFold assignment does not exactly match strict current OOF assignment")
    full_xgb = np.load(root / args.full_xgb, allow_pickle=False).astype(np.float32)
    fixed_xgb = np.load(root / args.fixed_xgb, allow_pickle=False).astype(np.float32)
    if full_xgb.shape != current.shape or fixed_xgb.shape != fixed_current.shape:
        raise ValueError("Matching seed-42 raw XGB artifact shapes do not align with strict GRU probabilities")
    full = protocol_rows(
        data["y"], full_xgb, current, alternate_oof, data["source"], data["terminal"], data["char"],
        args.sim_raw_alpha, args.au_raw_alpha,
    )
    fixed = protocol_rows(
        data["y"][data["valid_index"]], fixed_xgb, fixed_current, alternate_fixed,
        data["source"][data["valid_index"]], data["fixed_terminal"], data["fixed_char"], args.sim_raw_alpha, args.au_raw_alpha,
    )
    gate_result: dict[str, dict[str, object]] = {}
    baseline = full["current_strict_seed"]
    baseline_fixed = fixed["current_strict_seed"]
    for name, metrics in full.items():
        if name == "current_strict_seed":
            continue
        gate_result[name] = gate(metrics, fixed[name], baseline, baseline_fixed)
    safe = [name for name, value in gate_result.items() if value["passes_two_protocol_all_plus_terminal"]]
    report = {
        "scope": "Experiment-only second-seed GRU ensemble audit; no model/, script.py, or submission ZIP is modified.",
        "strictness": {
            "alternate_oof": "Accepted only after exact equality of its outer GroupKFold assignment with the current strictly nested student.",
            "teacher_contract": "Alternate outer-fold teacher targets must be fit inside that outer-train partition; no held-session labels may enter student soft targets.",
            "fixed_contract": "Alternate fixed student may use only common-train cross-fitted teacher targets; fixed validation rows are absent from its training target table.",
        },
        "protocol": {
            "raw_xgb": {"full": args.full_xgb, "fixed": args.fixed_xgb, "seed": 42},
            "source_alpha_xgb": {"sim": args.sim_raw_alpha, "au": args.au_raw_alpha},
            "gru_ensembles": "arithmetic 50/50 probabilities and normalized geometric mean",
            "raw_blend": "normalized source-wise geometric raw-XGB + GRU blend",
            "char20": "Applied only after every blend on terminal rows; future-history recovery is included by metric_row's hybrid metric.",
        },
        "full_nested_group_oof": full,
        "common_fixed_validation": fixed,
        "two_protocol_gate": gate_result,
        "recommendation": {
            "full_refit_allowed": bool(safe),
            "passing_candidates": safe,
            "rule": "Refit/export only if all-row fallback and char20-final terminal macro-F1 strictly improve in both protocols.",
        },
    }
    out = root / args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    compact = {name: {"pass": item["passes_two_protocol_all_plus_terminal"], "full": item["full_delta"], "fixed": item["fixed_delta"]} for name, item in gate_result.items()}
    print(json.dumps({"passing": safe, "deltas": compact}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
