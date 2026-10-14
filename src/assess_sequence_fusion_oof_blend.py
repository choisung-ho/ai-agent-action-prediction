#!/usr/bin/env python3
"""Full group-OOF blend analysis for live-XGB proxy and no-E5 GRU."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score


CLASSES = [
    "read_file", "grep_search", "list_directory", "glob_pattern", "edit_file", "write_file", "apply_patch",
    "run_bash", "run_tests", "lint_or_typecheck", "ask_user", "plan_task", "web_search", "respond_only",
]


def score(y: np.ndarray, p: np.ndarray, mask: np.ndarray) -> float:
    return float(f1_score(y[mask], p[mask].argmax(axis=1), labels=np.arange(len(CLASSES)), average="macro", zero_division=0))


def rows_for(xgb: np.ndarray, gru: np.ndarray, y: np.ndarray, terminal: np.ndarray, source: np.ndarray) -> list[dict[str, float]]:
    all_rows: list[dict[str, float]] = []
    for alpha in np.arange(0.0, 1.0001, 0.025):
        p = alpha * xgb + (1.0 - alpha) * gru
        all_rows.append({
            "xgb_weight": float(alpha), "gru_weight": float(1.0 - alpha),
            "all_macro_f1": score(y, p, np.ones(len(y), dtype=bool)),
            "terminal_macro_f1": score(y, p, terminal),
            "sim_terminal_macro_f1": score(y, p, terminal & (source == "sim")),
            "au_terminal_macro_f1": score(y, p, terminal & (source == "au")),
        })
    return all_rows


def compact(table: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    return {
        "equal_0_5": min(table, key=lambda row: abs(row["xgb_weight"] - 0.5)),
        "best_terminal": max(table, key=lambda row: row["terminal_macro_f1"]),
        "best_all": max(table, key=lambda row: row["all_macro_f1"]),
    }


def main() -> None:
    out = Path("experiments/modeling_ai_sequence_fusion_noe5_full")
    xgb = np.load("experiments/modeling_ak_live_xgb_proxy/oof_probabilities.npy").astype(np.float32)
    gru = np.load("experiments/modeling_ai_sequence_fusion_noe5/oof_probabilities.npy").astype(np.float32)
    shape = np.load("experiments/modeling_ak_live_xgb_proxy/input_shapes.npz", allow_pickle=True)
    y = shape["y"].astype(np.int64)
    terminal = shape["terminal"].astype(bool)
    source = shape["source"].astype(str)
    if xgb.shape != gru.shape or len(y) != len(xgb):
        raise ValueError("OOF arrays are not aligned")
    raw_rows = rows_for(xgb, gru, y, terminal, source)
    calibration = json.loads(Path("experiments/modeling_ac_tune2/source_calibration.json").read_text(encoding="utf-8"))
    weights = np.asarray([[calibration["sim"][label] for label in CLASSES],
                          [calibration["au"][label] for label in CLASSES]], dtype=np.float32)
    calibrated = xgb * weights[np.where(source == "sim", 0, 1)]
    calibrated /= calibrated.sum(axis=1, keepdims=True)
    calibrated_rows = rows_for(calibrated, gru, y, terminal, source)
    summary = {
        "validation_method": "matching 5-fold full-data GroupKFold OOF probabilities for both models; source calibration itself was selected earlier on fixed validation and is shown separately",
        "n_rows": int(len(y)), "n_terminal": int(terminal.sum()),
        "pure_raw_xgb": {"all_macro_f1": score(y, xgb, np.ones(len(y), bool)), "terminal_macro_f1": score(y, xgb, terminal)},
        "pure_current_calibrated_xgb": {"all_macro_f1": score(y, calibrated, np.ones(len(y), bool)), "terminal_macro_f1": score(y, calibrated, terminal)},
        "pure_gru": {"all_macro_f1": score(y, gru, np.ones(len(y), bool)), "terminal_macro_f1": score(y, gru, terminal)},
        "raw_xgb_blend": compact(raw_rows),
        "current_calibrated_xgb_blend": compact(calibrated_rows),
        "recommendation_rule": "If equal blend is non-inferior on terminal for both raw and calibrated variants, use xgb=0.5/gru=0.5 rather than a split-specific alpha. Otherwise retain live XGB until another independent split confirms a gate.",
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "full_oof_blend_raw_grid.json").write_text(json.dumps(raw_rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (out / "full_oof_blend_calibrated_grid.json").write_text(json.dumps(calibrated_rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (out / "full_oof_blend_decision.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
