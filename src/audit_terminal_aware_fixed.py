#!/usr/bin/env python3
"""Independent fixed-split session-bootstrap audit for a declared terminal route."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score

from experiment_terminal_aware_selector import (
    N_CLASSES,
    ROOT,
    route_probability,
    source_geometry,
)
from experiment_strict_blend_calibration import load_data


def macro(y: np.ndarray, pred: np.ndarray, mask: np.ndarray) -> float:
    return float(f1_score(y[mask], pred[mask], labels=np.arange(N_CLASSES), average="macro", zero_division=0))


def with_char(probability: np.ndarray, terminal: np.ndarray, char: np.ndarray) -> np.ndarray:
    result = probability.argmax(axis=1).astype(np.int16, copy=False)
    result = result.copy()
    result[terminal & (char >= 0)] = char[terminal & (char >= 0)]
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", default="terminal_weight_2")
    parser.add_argument("--raw-weight", type=float, default=0.5)
    parser.add_argument("--replicates", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=4273)
    parser.add_argument("--out", default="experiments/terminal_aware_selector/fixed_bootstrap_audit.json")
    args = parser.parse_args()
    if not 0.0 <= args.raw_weight <= 1.0:
        raise ValueError("--raw-weight must be in [0, 1]")

    # Use the exact old-B43 common-validation contract: seed-42 raw selector,
    # old no-E5 GRU and the aligned char20 fixed rule.  A different common
    # refit proxy is intentionally not used here.
    data = load_data(ROOT)
    valid_index = data["valid_index"]
    ids = np.asarray(data["ids"], dtype=str)[valid_index]
    y = data["y"][valid_index]
    source = data["source"][valid_index]
    terminal = data["fixed_terminal"]
    char = data["fixed_char"]
    old_gru = data["fixed_old_gru"]
    global_raw = np.load(ROOT / "experiments/xgb_robust_ensemble/strict_fixed_raw_seed42.npy", allow_pickle=False)
    terminal_raw = np.load(ROOT / "experiments/terminal_aware_selector" / f"fixed_{args.candidate}.npy", allow_pickle=False)
    if not (global_raw.shape == terminal_raw.shape == old_gru.shape == (len(y), N_CLASSES)):
        raise ValueError("Fixed components do not align")
    baseline = with_char(source_geometry(global_raw, old_gru, source), terminal, char)
    challenger = with_char(
        source_geometry(route_probability(global_raw, terminal_raw, terminal, args.raw_weight), old_gru, source),
        terminal,
        char,
    )
    term = terminal
    baseline_score = macro(y, baseline, term)
    challenger_score = macro(y, challenger, term)

    groups = np.asarray([sample_id.rsplit("-step_", 1)[0] for sample_id in ids])
    unique, inverse = np.unique(groups, return_inverse=True)
    rows_by_group = [np.flatnonzero(inverse == number) for number in range(len(unique))]
    generator = np.random.default_rng(args.seed)
    deltas = np.empty(args.replicates, dtype=np.float64)
    for replicate in range(args.replicates):
        selected = generator.integers(len(rows_by_group), size=len(rows_by_group))
        rows = np.concatenate([rows_by_group[number] for number in selected])
        terminal_rows = terminal[rows]
        deltas[replicate] = macro(y[rows], challenger[rows], terminal_rows) - macro(
            y[rows], baseline[rows], terminal_rows
        )
    report = {
        "scope": "Read-only exact old-B43 fixed-split audit; no submission artifact is modified.",
        "policy": {"candidate": args.candidate, "terminal_raw_weight": args.raw_weight},
        "counts": {"rows": int(len(y)), "sessions": int(len(unique)), "terminal_rows": int(term.sum()), "bootstrap_replicates": int(args.replicates)},
        "point_estimate": {"baseline_terminal_char20_macro_f1": baseline_score, "challenger_terminal_char20_macro_f1": challenger_score, "delta": challenger_score - baseline_score},
        "session_bootstrap": {
            "positive_probability": float(np.mean(deltas > 0.0)),
            "mean_delta": float(deltas.mean()),
            "median_delta": float(np.median(deltas)),
            "p05_delta": float(np.quantile(deltas, 0.05)),
            "p95_delta": float(np.quantile(deltas, 0.95)),
        },
    }
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
