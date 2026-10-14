#!/usr/bin/env python3
"""Experiment R: GPU CatBoost selector over frozen I3 margins plus Q state.

This intentionally reads only cached validation inputs and writes solely below
``experiments/modeling_r``.  It cannot alter the live submission model.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from catboost import CatBoostClassifier
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import GroupKFold

from train_modeling_b import ALL_CLASSES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--prepared", default="experiments/modeling_q/prepared_inputs.npz")
    parser.add_argument("--out-dir", default="experiments/modeling_r")
    parser.add_argument("--iterations", type=int, default=700)
    parser.add_argument("--depths", default="6,8")
    parser.add_argument("--learning-rate", type=float, default=0.08)
    parser.add_argument("--early-stopping", type=int, default=80)
    return parser.parse_args()


def load_validation(data_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    labels = {row["id"]: row["action"] for row in csv.DictReader((data_dir / "train_labels.csv").open(encoding="utf-8"))}
    split = {row["id"]: row["split"] for row in csv.DictReader((Path("splits") / "common_split.csv").open(encoding="utf-8"))}
    ids = []
    with (data_dir / "train.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                sample_id = str(json.loads(line)["id"])
                if split[sample_id] == "val":
                    ids.append(sample_id)
    y = np.asarray([labels[sample_id] for sample_id in ids])
    groups = np.asarray([sample_id.split("-step_", 1)[0] for sample_id in ids])
    return np.asarray(ids), y, groups


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache = np.load(args.prepared, allow_pickle=False)
    x = np.asarray(cache["x"], dtype=np.float32)
    ids, y, groups = load_validation(Path(args.data_dir))
    if len(x) != len(y):
        raise ValueError(f"Cached rows ({len(x)}) do not match validation labels ({len(y)})")
    folds = list(GroupKFold(n_splits=5).split(x, y, groups))
    rows = []
    all_predictions: dict[str, np.ndarray] = {}
    for depth in (int(value) for value in args.depths.split(",") if value.strip()):
        name = f"R_catboost_d{depth}"
        prediction = np.empty(len(y), dtype=object)
        best_iterations = []
        for fold, (train_index, valid_index) in enumerate(folds, start=1):
            model = CatBoostClassifier(
                iterations=args.iterations,
                depth=depth,
                learning_rate=args.learning_rate,
                loss_function="MultiClass",
                eval_metric="TotalF1:average=Macro",
                auto_class_weights="Balanced",
                l2_leaf_reg=6.0,
                random_seed=42 + fold,
                task_type="GPU",
                devices="0",
                allow_writing_files=False,
                verbose=False,
            )
            model.fit(
                x[train_index], y[train_index],
                eval_set=(x[valid_index], y[valid_index]),
                early_stopping_rounds=args.early_stopping,
                use_best_model=True,
            )
            prediction[valid_index] = model.predict(x[valid_index]).reshape(-1).astype(str)
            best_iterations.append(int(model.get_best_iteration()))
            fold_score = f1_score(y[valid_index], prediction[valid_index], labels=ALL_CLASSES, average="macro", zero_division=0)
            print(f"{name} fold={fold} best_iter={best_iterations[-1]} macro_f1={fold_score:.6f}", flush=True)
        score = f1_score(y, prediction, labels=ALL_CLASSES, average="macro", zero_division=0)
        print(f"{name} oof_macro_f1={score:.6f}", flush=True)
        rows.append({"experiment": name, "oof_macro_f1": score, "best_iterations": ";".join(map(str, best_iterations))})
        all_predictions[name] = prediction
        (out_dir / f"{name}_classification_report.txt").write_text(
            classification_report(y, prediction, labels=ALL_CLASSES, digits=4, zero_division=0), encoding="utf-8"
        )
    with (out_dir / "results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["experiment", "oof_macro_f1", "best_iterations"])
        writer.writeheader()
        writer.writerows(rows)
    np.savez_compressed(out_dir / "oof_predictions.npz", ids=ids, y=y, **all_predictions)


if __name__ == "__main__":
    main()
