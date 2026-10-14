#!/usr/bin/env python3
"""Create selector-level OOF probabilities inside the strict common train split.

The existing ``au_source_strict_common`` artifact already contains label-safe
2032-width train/validation feature blocks.  Its saved validation prediction
is a selector refit on all common-train rows.  This companion adds the missing
five-fold *selector* OOF table for common-train rows, so a probability
calibrator can be fitted without ever using common-validation labels.

It only writes a new experiment directory and never changes a submission
model, script, or ZIP.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold


N_CLASSES = 14


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--blocks", default="experiments/au_source_strict_common/strict_blocks.npz")
    parser.add_argument("--out-dir", default="experiments/strict_blend_calibration")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--n-jobs", type=int, default=4)
    parser.add_argument("--seed", type=int, default=808,
                        help="Matches the existing strict raw_global validation proxy.")
    return parser.parse_args()


def terminal_mask(ids: np.ndarray) -> np.ndarray:
    maxima: dict[str, int] = {}
    parsed: list[tuple[str, int]] = []
    for sample_id in ids.astype(str):
        session, raw_step = sample_id.rsplit("-step_", 1)
        step = int(raw_step)
        maxima[session] = max(maxima.get(session, -1), step)
        parsed.append((session, step))
    return np.asarray([step == maxima[session] for session, step in parsed], dtype=bool)


def macro(y: np.ndarray, probability: np.ndarray, mask: np.ndarray) -> float:
    return float(f1_score(y[mask], probability[mask].argmax(axis=1), labels=np.arange(N_CLASSES), average="macro", zero_division=0))


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    blocks = np.load(root / args.blocks, allow_pickle=False)
    out = root / args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    output = out / "common_train_raw_xgb_oof.npy"
    fold_output = out / "common_train_selector_fold.npy"
    ids_output = out / "common_train_ids.npy"
    metadata_output = out / "common_train_raw_xgb_oof_metadata.json"

    required = {"ids_train", "target_train", "base_train", "state_train", "source_e_train", "user_train"}
    if not required.issubset(blocks.files):
        raise ValueError(f"Strict blocks are missing {sorted(required - set(blocks.files))}")
    ids = blocks["ids_train"].astype(str)
    y = blocks["target_train"].astype(np.int64)
    x = np.column_stack((
        blocks["base_train"], blocks["state_train"], blocks["source_e_train"], blocks["user_train"],
    )).astype(np.float32)
    if x.shape != (len(y), 2032):
        raise ValueError(f"Expected (n, 2032) strict raw selector input, got {x.shape}")
    groups = np.asarray([value.rsplit("-step_", 1)[0] for value in ids])
    terminal = terminal_mask(ids)
    folds = list(GroupKFold(n_splits=5).split(np.zeros(len(y)), y, groups))
    fold = np.empty(len(y), dtype=np.int8)
    for number, (_, valid) in enumerate(folds):
        fold[valid] = number

    # The existing file is valid only when it is provably the same run
    # contract; otherwise overwrite it with the requested deterministic OOF.
    if output.exists() and fold_output.exists() and ids_output.exists() and metadata_output.exists():
        saved = np.load(output, allow_pickle=False)
        saved_fold = np.load(fold_output, allow_pickle=False)
        saved_ids = np.load(ids_output, allow_pickle=False).astype(str)
        metadata = json.loads(metadata_output.read_text(encoding="utf-8"))
        if (saved.shape == (len(y), N_CLASSES) and np.array_equal(saved_fold, fold) and np.array_equal(saved_ids, ids)
                and metadata.get("seed") == args.seed and metadata.get("device") == args.device):
            print(json.dumps({"status": "reused", "path": str(output), "shape": list(saved.shape)}, ensure_ascii=False))
            return

    # Import lazily so simple metadata reads never initialize a GPU context.
    from xgboost import XGBClassifier

    probability = np.zeros((len(y), N_CLASSES), dtype=np.float32)
    records: list[dict[str, float | int]] = []
    for number, (train, valid) in enumerate(folds, start=1):
        count = np.bincount(y[train], minlength=N_CLASSES)
        if np.any(count == 0):
            raise ValueError(f"Fold {number} lacks classes {np.flatnonzero(count == 0).tolist()}")
        weight = len(train) / (N_CLASSES * count[y[train]])
        model = XGBClassifier(
            n_estimators=210, max_depth=10, learning_rate=0.04, subsample=0.85, colsample_bytree=0.85,
            objective="multi:softprob", num_class=N_CLASSES, tree_method="hist", device=args.device,
            eval_metric="mlogloss", n_jobs=args.n_jobs, random_state=args.seed,
        )
        model.fit(x[train], y[train], sample_weight=weight, verbose=False)
        probability[valid] = model.predict_proba(x[valid]).astype(np.float32)
        record = {
            "fold": number, "train_rows": int(len(train)), "valid_rows": int(len(valid)),
            "all_macro_f1": macro(y, probability, np.isin(np.arange(len(y)), valid)),
            "terminal_macro_f1": macro(y, probability, terminal & np.isin(np.arange(len(y)), valid)),
        }
        records.append(record)
        print(json.dumps(record), flush=True)

    np.save(output, probability)
    np.save(fold_output, fold)
    np.save(ids_output, ids)
    metadata = {
        "scope": "Experiment-only. No submission asset is touched.",
        "contract": (
            "Five-fold session GroupKFold selector OOF within common-train. Inputs are the precomputed strict 2032-width "
            "common-train OOF base/source blocks plus label-free state/user features. Common validation labels are never read."
        ),
        "input_blocks": str(Path(args.blocks)),
        "shape": list(probability.shape), "seed": args.seed, "device": args.device,
        "xgboost": {"n_estimators": 210, "max_depth": 10, "learning_rate": 0.04, "subsample": 0.85,
                    "colsample_bytree": 0.85, "class_weight": "n/(14*class_count)", "random_state": args.seed},
        "aggregate": {"all_macro_f1": macro(y, probability, np.ones(len(y), dtype=bool)),
                      "terminal_macro_f1": macro(y, probability, terminal)},
        "folds": records,
        "caveat": "The strict validation proxy and this selector OOF use seed 808; the current live selector seed is 42.",
    }
    metadata_output.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "created", "path": str(output), **metadata["aggregate"]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
