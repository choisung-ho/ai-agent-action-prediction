#!/usr/bin/env python3
"""Record fixed-validation blend/gating evidence for the no-E5 GRU candidate.

The artifact is intentionally descriptive, not a promotion command.  The
final alpha must also be checked against full 5-fold OOF proxy data before a
submission change is made.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score

sys.path.insert(0, "experiments/data_forensics_session_reveal")
from audit_future_history import ALL_CLASSES, recover_from_future_history  # noqa: E402


def f1(y: np.ndarray, prediction: np.ndarray, mask: np.ndarray | None = None) -> float:
    if mask is None:
        mask = np.ones(len(y), dtype=bool)
    return float(f1_score(y[mask], prediction[mask], labels=np.arange(len(ALL_CLASSES)), average="macro", zero_division=0))


def main() -> None:
    out = Path("experiments/modeling_ai_sequence_fusion_noe5_full")
    out.mkdir(parents=True, exist_ok=True)
    labels = {row["id"]: row["action"] for row in csv.DictReader(open("data/train_labels.csv", encoding="utf-8"))}
    split = {row["id"]: row["split"] for row in csv.DictReader(open("splits/common_split.csv", encoding="utf-8"))}
    rows = [json.loads(line) for line in open("data/train.jsonl", encoding="utf-8") if line.strip()]
    rows = [row for row in rows if split[row["id"]] == "val"]
    ids = [row["id"] for row in rows]
    label_to_id = {name: index for index, name in enumerate(ALL_CLASSES)}
    y = np.asarray([label_to_id[labels[sample_id]] for sample_id in ids])
    source = np.asarray([0 if sample_id.startswith("sess_sim_") else 1 for sample_id in ids])
    xgb = np.load("experiments/modeling_ac_tune2/oof_probabilities.npy").astype(np.float32)
    calibration = json.loads(Path("experiments/modeling_ac_tune2/source_calibration.json").read_text(encoding="utf-8"))
    factors = np.asarray([[calibration["sim"][label] for label in ALL_CLASSES],
                          [calibration["au"][label] for label in ALL_CLASSES]], dtype=np.float32)
    xgb = xgb * factors[source]
    xgb /= xgb.sum(axis=1, keepdims=True)
    gru = np.load("experiments/modeling_ai_sequence_fusion_fixed_noe5/valid_probabilities.npy").astype(np.float32)
    recovered, _ = recover_from_future_history(rows)
    terminal = np.asarray([sample_id not in recovered for sample_id in ids])
    if not terminal.any():
        raise RuntimeError("Expected unrecovered terminal rows")

    records: list[dict[str, float]] = []
    base_prediction = xgb.argmax(axis=1)
    for alpha in np.arange(0.0, 1.0001, 0.025):
        probability = alpha * xgb + (1.0 - alpha) * gru
        prediction = probability.argmax(axis=1)
        hybrid = prediction.copy()
        for index, sample_id in enumerate(ids):
            if sample_id in recovered:
                hybrid[index] = label_to_id[recovered[sample_id]]
        records.append({
            "xgb_weight": float(alpha), "gru_weight": float(1.0 - alpha),
            "all_macro_f1": f1(y, prediction), "terminal_macro_f1": f1(y, prediction, terminal),
            "terminal_accuracy": float(np.mean(y[terminal] == prediction[terminal])),
            "future_recovery_hybrid_macro_f1": f1(y, hybrid),
        })
    conservative = min(records, key=lambda row: abs(row["xgb_weight"] - 0.5))
    best_terminal = max(records, key=lambda row: row["terminal_macro_f1"])
    best_all = max(records, key=lambda row: row["all_macro_f1"])
    base_hybrid = base_prediction.copy()
    for index, sample_id in enumerate(ids):
        if sample_id in recovered:
            base_hybrid[index] = label_to_id[recovered[sample_id]]
    summary = {
        "purpose": "fixed-validation evidence only; use full-OOF proxy to make the final alpha decision",
        "n_rows": len(ids), "n_recovered": len(recovered), "n_terminal_unrecovered": int(terminal.sum()),
        "live_xgb_calibrated": {"all_macro_f1": f1(y, base_prediction), "terminal_macro_f1": f1(y, base_prediction, terminal),
                                  "future_recovery_hybrid_macro_f1": f1(y, base_hybrid)},
        "conservative_equal_blend": conservative,
        "fixed_best_terminal": best_terminal,
        "fixed_best_all": best_all,
        "deployment_gate": "Apply blend only if future-history recovery did not deterministically reveal the label. In the fixed audit these are exactly terminal session rows.",
    }
    (out / "fixed_blend_grid.json").write_text(json.dumps(records, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (out / "fixed_blend_decision.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
