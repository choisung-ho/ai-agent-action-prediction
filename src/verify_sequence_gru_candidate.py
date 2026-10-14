#!/usr/bin/env python3
"""Verify that the non-live candidate retains the NumPy GRU weights exactly."""

from __future__ import annotations

import json
import sys

import joblib
import numpy as np

sys.path.insert(0, "src")
from numpy_sequence_fusion import probabilities  # noqa: E402


def main() -> None:
    bundle = joblib.load("model_candidate_sequence_gru_blend/best_model.pkl")
    probe = np.load("experiments/modeling_ai_sequence_fusion_noe5_full/verification_probe.npz")
    values = probabilities(probe["action"], probe["status"], probe["arg"], probe["base"], probe["meta"],
                           bundle["sequence_fusion_noe5_weights"])
    result = {
        "has_weights": "sequence_fusion_noe5_weights" in bundle,
        "has_config": "sequence_fusion_noe5_config" in bundle,
        "n_weight_arrays": len(bundle["sequence_fusion_noe5_weights"]),
        "max_abs_probability_difference": float(np.max(np.abs(values - probe["torch_probabilities"]))),
        "argmax_agreement": float(np.mean(values.argmax(axis=1) == probe["torch_probabilities"].argmax(axis=1))),
    }
    with open("experiments/modeling_ai_sequence_fusion_noe5_full/candidate_verification.json", "w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
