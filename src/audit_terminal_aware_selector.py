#!/usr/bin/env python3
"""Session-bootstrap audit for the predeclared terminal-aware selector route.

This script only reads cached OOF predictions produced by
``experiment_terminal_aware_selector.py``.  It audits one named member of the
small predeclared route grid at a time.  Results are exploratory unless an
independent fixed-split and bootstrap gate also support the same policy.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score

from experiment_terminal_aware_selector import (
    N_CLASSES,
    ROOT,
    full_char,
    full_context,
    route_probability,
    source_geometry,
    terminal_mask,
)


def macro(y: np.ndarray, pred: np.ndarray, mask: np.ndarray) -> float:
    return float(
        f1_score(
            y[mask], pred[mask], labels=np.arange(N_CLASSES),
            average="macro", zero_division=0,
        )
    )


def final_predictions(probability: np.ndarray, terminal: np.ndarray, char: np.ndarray) -> np.ndarray:
    prediction = probability.argmax(axis=1).astype(np.int16, copy=False)
    final = prediction.copy()
    final[terminal & (char >= 0)] = char[terminal & (char >= 0)]
    return final


def score_delta(
    y: np.ndarray, baseline: np.ndarray, challenger: np.ndarray,
    terminal: np.ndarray, mask: np.ndarray,
) -> dict[str, float]:
    chosen = mask & terminal
    return {
        "baseline_terminal_char20_macro_f1": macro(y, baseline, chosen),
        "challenger_terminal_char20_macro_f1": macro(y, challenger, chosen),
        "delta": macro(y, challenger, chosen) - macro(y, baseline, chosen),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", default="terminal_weight_2")
    parser.add_argument("--raw-weight", type=float, default=1.0)
    parser.add_argument("--replicates", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=4271)
    parser.add_argument("--out", default="experiments/terminal_aware_selector/full_bootstrap_audit.json")
    args = parser.parse_args()
    if not 0.0 <= args.raw_weight <= 1.0:
        raise ValueError("--raw-weight must be in [0, 1]")

    x, y, source, groups, ids = full_context()
    del x
    terminal = terminal_mask(ids)
    char = full_char()
    global_raw = np.load(ROOT / "experiments/xgb_robust_ensemble/full_oof_raw_seed42.npy", allow_pickle=False)
    old_gru = np.load(ROOT / "experiments/modeling_ai_sequence_fusion_noe5/oof_probabilities.npy", allow_pickle=False)
    terminal_raw = np.load(
        ROOT / "experiments/terminal_aware_selector" / f"full_oof_{args.candidate}.npy",
        allow_pickle=False,
    )
    expected = (len(y), N_CLASSES)
    if not (global_raw.shape == old_gru.shape == terminal_raw.shape == expected):
        raise ValueError("OOF component shapes do not align")

    baseline = final_predictions(source_geometry(global_raw, old_gru, source), terminal, char)
    challenger = final_predictions(
        source_geometry(route_probability(global_raw, terminal_raw, terminal, args.raw_weight), old_gru, source),
        terminal,
        char,
    )
    full_mask = np.ones(len(y), dtype=bool)
    point = score_delta(y, baseline, challenger, terminal, full_mask)

    unique_groups, inverse = np.unique(groups, return_inverse=True)
    rows_by_group = [np.flatnonzero(inverse == index) for index in range(len(unique_groups))]
    generator = np.random.default_rng(args.seed)
    deltas = np.empty(args.replicates, dtype=np.float64)
    for replicate in range(args.replicates):
        selected = generator.integers(len(rows_by_group), size=len(rows_by_group))
        rows = np.concatenate([rows_by_group[index] for index in selected])
        selected_terminal = terminal[rows]
        if not np.any(selected_terminal):
            raise RuntimeError("A bootstrap replicate unexpectedly contains no terminal rows")
        deltas[replicate] = macro(y[rows], challenger[rows], selected_terminal) - macro(
            y[rows], baseline[rows], selected_terminal
        )
    report = {
        "scope": "Read-only OOF audit; no model, script, or submission archive is modified.",
        "policy": {
            "candidate": args.candidate,
            "terminal_raw_weight": args.raw_weight,
            "selection": "One named member of the predeclared route grid; this OOF audit alone is not a deployment gate.",
        },
        "counts": {
            "rows": int(len(y)),
            "sessions": int(len(unique_groups)),
            "terminal_rows": int(terminal.sum()),
            "bootstrap_replicates": int(args.replicates),
        },
        "point_estimate": point,
        "session_bootstrap": {
            "positive_probability": float(np.mean(deltas > 0.0)),
            "mean_delta": float(deltas.mean()),
            "median_delta": float(np.median(deltas)),
            "p05_delta": float(np.quantile(deltas, 0.05)),
            "p95_delta": float(np.quantile(deltas, 0.95)),
            "minimum_delta": float(deltas.min()),
            "maximum_delta": float(deltas.max()),
        },
    }
    destination = Path(args.out)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
