#!/usr/bin/env python3
"""Fit an isolated B43 terminal-aware selector bundle after its OOF gate passes.

The primary B43 bundle is only read.  The additional selector is trained on
the same established 2032-column out-of-fold selector contract as B43, with
one reconstructible batch-terminal indicator appended.  It is subsequently
called only for batch-terminal rows by the candidate-only inference script.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from xgboost import XGBClassifier

from experiment_terminal_aware_selector import (
    CONFIGS,
    N_CLASSES,
    augmented_features,
    full_context,
    terminal_mask,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", default="model/best_model.pkl")
    parser.add_argument("--base-metadata", default="model/best_model_metadata.json")
    parser.add_argument("--out-dir", default="experiments/terminal_aware_selector/model_candidate")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--n-jobs", type=int, default=4)
    parser.add_argument("--validation-report", default="experiments/terminal_aware_selector/full_report.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = CONFIGS["terminal_weight_2"]
    x, y, _source, _groups, ids = full_context()
    terminal = terminal_mask(ids)
    x_augmented = augmented_features(x, terminal)
    if x_augmented.shape != (len(y), 2033):
        raise ValueError(f"Unexpected augmented selector shape: {x_augmented.shape}")
    counts = np.bincount(y, minlength=N_CLASSES)
    if np.any(counts == 0):
        raise ValueError("A class is missing from full-refit labels")
    sample_weight = len(y) / (N_CLASSES * counts[y])
    sample_weight = sample_weight.astype(np.float64, copy=False)
    sample_weight[terminal] *= float(config["terminal_weight"])
    parameters = {key: value for key, value in config.items() if key != "terminal_weight"}
    selector = XGBClassifier(
        **parameters,
        objective="multi:softprob",
        num_class=N_CLASSES,
        tree_method="hist",
        device=args.device,
        eval_metric="mlogloss",
        n_jobs=args.n_jobs,
    )
    selector.fit(x_augmented, y, sample_weight=sample_weight, verbose=False)

    base_model = Path(args.base_model)
    base_metadata = Path(args.base_metadata)
    bundle = joblib.load(base_model)
    if int(bundle.get("selector_feature_count", -1)) != 2032:
        raise ValueError("Base bundle is not the expected B43 2032-feature selector contract")
    candidate = dict(bundle)
    candidate["terminal_aware_selector_v1"] = {
        "version": "terminal_aware_raw_selector_v1",
        "selector": selector,
        "feature_count": int(x_augmented.shape[1]),
        "terminal_raw_weight": 0.5,
        "terminal_indicator": "is_batch_terminal, computed from supplied session step maxima",
        "training": {
            "rows": int(len(y)),
            "terminal_rows": int(terminal.sum()),
            "class_balance": "inverse class frequency",
            "terminal_weight": float(config["terminal_weight"]),
            "config": parameters,
        },
    }
    metadata = json.loads(base_metadata.read_text(encoding="utf-8"))
    validation = json.loads(Path(args.validation_report).read_text(encoding="utf-8"))
    metadata["best_experiment"] = (
        "B45_candidate_B43_plus_terminal_aware_raw_selector_v1_before_existing_noE5_geometric_blend"
    )
    metadata["terminal_aware_selector"] = {
        "status": "candidate_only_not_baseline",
        "route": "Only supplied batch-terminal rows use the additional raw selector; all nonterminal rows retain B43 raw selector.",
        "raw_weight": 0.5,
        "indicator": "session-local maximum supplied numeric step; no labels, target lookup, or ID memorization",
        "training_contract": candidate["terminal_aware_selector_v1"]["training"],
        "validation_report": validation,
    }

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(candidate, out_dir / "best_model.pkl", compress=3)
    (out_dir / "best_model_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    manifest = {
        "scope": "Candidate-only model bundle; B43 baseline model and archives were read only.",
        "base_model": str(base_model),
        "out_dir": str(out_dir),
        "feature_count": int(x_augmented.shape[1]),
        "terminal_rows": int(terminal.sum()),
        "terminal_raw_weight": 0.5,
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
