#!/usr/bin/env python3
"""Attach the verified NumPy no-E5 GRU to the current submission bundle.

The resulting bundle contains every array needed by ``script.py``; no PyTorch,
encoder checkpoint, or extra ZIP member is required.  It is built as a
candidate only and must be smoke/pre-submit checked before promotion.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np


ALL_CLASSES = [
    "read_file", "grep_search", "list_directory", "glob_pattern",
    "edit_file", "write_file", "apply_patch", "run_bash",
    "run_tests", "lint_or_typecheck", "ask_user", "plan_task",
    "web_search", "respond_only",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", default="model/best_model.pkl")
    parser.add_argument("--base-metadata", default="model/best_model_metadata.json")
    parser.add_argument("--weights", default="experiments/modeling_ai_sequence_fusion_noe5_full/weights.npz")
    parser.add_argument("--architecture", default="experiments/modeling_ai_sequence_fusion_noe5_full/architecture.json")
    parser.add_argument("--out-dir", default="model_candidate_ngram_raw_xgb_gru")
    parser.add_argument("--xgb-weight", type=float, default=0.5)
    parser.add_argument("--sim-xgb-weight", type=float, default=None)
    parser.add_argument("--au-xgb-weight", type=float, default=None)
    parser.add_argument("--blend-mode", choices=("linear", "geometric"), default="linear")
    parser.add_argument("--evidence-json", default=None,
                        help="Optional experiment evidence to record verbatim instead of the legacy no-E5 metrics.")
    parser.add_argument("--experiment-name", default=None,
                        help="Optional metadata name; defaults to preserving the base bundle's name.")
    parser.add_argument("--best-macro-f1", type=float, default=None,
                        help="Optional validated fallback macro-F1 to record in metadata.")
    parser.add_argument("--validation-method", default=None,
                        help="Optional validation-method description to record in metadata.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    candidate_weights = [args.xgb_weight, args.sim_xgb_weight, args.au_xgb_weight]
    if any(value is not None and not 0.0 <= value <= 1.0 for value in candidate_weights):
        raise ValueError("--xgb-weight must be within [0, 1]")
    architecture = json.loads(Path(args.architecture).read_text(encoding="utf-8"))
    if architecture.get("format") != "sequence_fusion_noe5_numpy_v1":
        raise ValueError("Unexpected sequence-fusion export format")
    if architecture.get("all_classes") != ALL_CLASSES:
        raise ValueError("Sequence-fusion class order does not match submission class order")
    with np.load(args.weights) as archive:
        weights = {name: archive[name].astype(np.float32, copy=False) for name in archive.files}
    required = {
        "action_embedding.weight", "status_embedding.weight", "position_embedding.weight",
        "arg_projection.0.weight", "arg_projection.0.bias",
        "gru.weight_ih_l0", "gru.weight_hh_l0", "gru.bias_ih_l0", "gru.bias_hh_l0",
        "gru.weight_ih_l0_reverse", "gru.weight_hh_l0_reverse",
        "gru.bias_ih_l0_reverse", "gru.bias_hh_l0_reverse",
        "base.0.weight", "base.0.bias", "base.1.weight", "base.1.bias",
        "e5.0.weight", "e5.0.bias", "e5.1.weight", "e5.1.bias",
        "meta.0.weight", "meta.0.bias", "head.0.weight", "head.0.bias",
        "head.3.weight", "head.3.bias",
    }
    missing = required - set(weights)
    if missing:
        raise ValueError(f"Sequence-fusion export is missing arrays: {sorted(missing)}")

    base = joblib.load(args.base_model)
    if not isinstance(base, dict):
        raise TypeError("The stacked XGBoost submission model must be a dictionary bundle")
    candidate = dict(base)
    sequence_config = {
        "weights": weights,
        "xgb_weight": float(args.xgb_weight),
        "blend_mode": args.blend_mode,
        "base_probability": "raw_selector_predict_proba_before_source_calibration",
        "gate": "only rows not deterministically exposed by later supplied session history",
        "all_classes": ALL_CLASSES,
        "numpy_equivalence_max_abs_probability_difference": architecture["numpy_equivalence"]["max_abs_probability_difference"],
    }
    if args.sim_xgb_weight is not None or args.au_xgb_weight is not None:
        sequence_config["source_xgb_weights"] = {
            "sim": float(args.xgb_weight if args.sim_xgb_weight is None else args.sim_xgb_weight),
            "au": float(args.xgb_weight if args.au_xgb_weight is None else args.au_xgb_weight),
        }
    candidate["sequence_fusion_noe5_v1"] = sequence_config

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(candidate, out_dir / "best_model.pkl", compress=3)

    metadata = json.loads(Path(args.base_metadata).read_text(encoding="utf-8"))
    sequence_metadata = {
        "xgb_weight": float(args.xgb_weight),
        "source_xgb_weights": sequence_config.get("source_xgb_weights"),
        "blend_mode": args.blend_mode,
        "gru_weight": float(1.0 - args.xgb_weight),
        "xgb_probability": "raw pre-source-calibration selector probability",
        "gate": "only not exact future-history recovered rows",
        "numpy_equivalence_max_abs_probability_difference": architecture["numpy_equivalence"]["max_abs_probability_difference"],
    }
    if args.evidence_json:
        sequence_metadata["validation_evidence"] = json.loads(Path(args.evidence_json).read_text(encoding="utf-8"))
    else:
        # Backward-compatible historical evidence for the original candidate.
        sequence_metadata.update({
            "full_group_oof_raw_xgb_macro_f1": 0.750043170185729,
            "full_group_oof_gru_macro_f1": 0.7436452627035227,
            "full_group_oof_equal_blend_macro_f1": 0.7592050693979607,
            "full_group_oof_equal_blend_terminal_macro_f1": 0.7244331927781053,
            "full_group_oof_geometric_fallback_macro_f1": 0.7603385811033483,
            "full_group_oof_geometric_terminal_macro_f1": 0.731274589697604,
            "nested_group_oof_geometric_terminal_macro_f1": 0.730854,
            "fixed_validation_geometric_terminal_macro_f1": 0.705017,
        })
    metadata["sequence_fusion_noe5"] = sequence_metadata
    if args.experiment_name:
        metadata["best_experiment"] = args.experiment_name
    if args.best_macro_f1 is not None:
        metadata["best_macro_f1"] = float(args.best_macro_f1)
    if args.validation_method:
        metadata["validation_method"] = args.validation_method
    (out_dir / "best_model_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "out_dir": str(out_dir),
        "xgb_weight": args.xgb_weight,
        "source_xgb_weights": sequence_config.get("source_xgb_weights"),
        "n_weight_arrays": len(weights),
        "weight_bytes": int(sum(array.nbytes for array in weights.values())),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
