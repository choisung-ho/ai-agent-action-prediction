#!/usr/bin/env python3
"""Create a non-live XGB+NumPy-GRU bundle for integration testing.

The output remains under ``model_candidate_sequence_gru_blend``.  It is never
copied into ``model/`` by this script.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np


def main() -> None:
    source = Path("model/best_model.pkl")
    weights_path = Path("experiments/modeling_ai_sequence_fusion_noe5_full/weights.npz")
    architecture_path = Path("experiments/modeling_ai_sequence_fusion_noe5_full/architecture.json")
    target = Path("model_candidate_sequence_gru_blend")
    target.mkdir(parents=True, exist_ok=True)
    bundle = joblib.load(source)
    with np.load(weights_path) as archive:
        # Ensure arrays own their memory after archive close and remain compact
        # inside the normal joblib artifact.
        weights = {name: archive[name].astype(np.float32, copy=True) for name in archive.files}
    bundle["sequence_fusion_noe5_weights"] = weights
    bundle["sequence_fusion_noe5_config"] = {
        "format": "sequence_fusion_noe5_numpy_v1",
        "xgb_weight": 0.5,
        "gru_weight": 0.5,
        "xgb_probability": "raw_before_source_class_weights",
        "gate": "only IDs without deterministic future-history revelation",
        "architecture": json.loads(architecture_path.read_text(encoding="utf-8"))["architecture"],
    }
    joblib.dump(bundle, target / "best_model.pkl", compress=3)
    metadata = json.loads(Path("model/best_model_metadata.json").read_text(encoding="utf-8"))
    metadata["sequence_fusion_fallback"] = {
        "artifact": "embedded sequence_fusion_noe5_weights",
        "blend": "0.5 raw_xgb + 0.5 no_e5_gru",
        "gate": "future-history unrecovered rows only",
        "full_oof_evidence": {"all_macro_f1": 0.7592050693979607, "terminal_macro_f1": 0.7244331927781053},
    }
    (target / "best_model_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {target / 'best_model.pkl'}", flush=True)


if __name__ == "__main__":
    main()
