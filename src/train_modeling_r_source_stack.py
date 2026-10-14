#!/usr/bin/env python3
"""Modeling R: add a source-routed E hierarchy to the proven I stack.

This is deliberately experiment-only.  It never writes ``model/`` or a ZIP.
The validation phase answers a narrow question: do separate ``sess_sim`` and
``sess_au`` E hierarchies add useful, non-leaky signal to the I3 selector?

The optional full-OOF phase is the promotion path.  It generates source-E
features with the exact 5-fold group split used by Modeling N, appends them to
the existing leakage-safe E/D/B OOF cache, and writes a candidate bundle
outside ``model/``.  It is intentionally separate from promotion so that a
candidate must still pass the normal submission checks before replacement.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import GroupKFold

from data import load_jsonl, load_labels
from train_modeling_b import ALL_CLASSES, build_modeling_b_features
from train_modeling_e import TREE, build_modeling_e_features, fit_tree, predict_tree
from train_modeling_i import stack_features


SOURCES = ("sim", "au")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("val", "full-oof"), default="val")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--split-path", default="splits/common_split.csv")
    parser.add_argument("--i-bundle", default="experiments/modeling_i/I_best.pkl")
    parser.add_argument("--n-oof-dir", default="experiments/modeling_n_refit")
    parser.add_argument("--out-dir", default="experiments/modeling_r_source_stack")
    parser.add_argument("--candidate-dir", default="model_candidate_source_stack")
    parser.add_argument("--trees", type=int, default=300)
    parser.add_argument("--leaf", type=int, default=10)
    parser.add_argument("--n-jobs", type=int, default=8, help="Selector CPU workers; keep room for concurrent experiments")
    return parser.parse_args()


def source_of(sample_id: str) -> str:
    return "au" if str(sample_id).startswith("sess_au_") else "sim"


def load_split(path: str) -> dict[str, str]:
    with open(path, newline="", encoding="utf-8") as handle:
        return {row["id"]: row["split"] for row in csv.DictReader(handle)}


def fit_source_models(texts: list[str], labels: np.ndarray, sources: np.ndarray) -> dict[str, dict]:
    models: dict[str, dict] = {}
    for source in SOURCES:
        indices = np.flatnonzero(sources == source)
        if not len(indices):
            raise ValueError(f"No {source} rows available to train source hierarchy")
        models[source] = fit_tree([texts[index] for index in indices], labels[indices].tolist(), c=0.5, ngram_max=3)
    return models


def canonical_node_classes(node: str) -> list[str]:
    return list(TREE[node])


def margins(model, texts: list[str]) -> np.ndarray:
    values = np.asarray(model.decision_function(texts))
    if values.ndim == 1:
        values = np.column_stack((-values, values))
    return values.astype(np.float32, copy=False)


def source_e_features(models: dict[str, dict], texts: list[str], sources: np.ndarray) -> tuple[np.ndarray, list[str], np.ndarray]:
    """Return fixed-width routed-E margins, hard predictions, and source bits.

    Each hierarchy is only queried for its own source.  Canonical class slots
    make OOF folds robust even if a rare node child were absent in one fold.
    """
    n_rows = len(texts)
    blocks: list[np.ndarray] = []
    schema: list[str] = []
    for node in sorted(TREE):
        labels = canonical_node_classes(node)
        block = np.zeros((n_rows, len(labels)), dtype=np.float32)
        for source in SOURCES:
            indices = np.flatnonzero(sources == source)
            if not len(indices):
                continue
            component = models[source]["models"][node]
            values = margins(component, [texts[index] for index in indices])
            position = {label: index for index, label in enumerate(component.classes_)}
            for target_col, label in enumerate(labels):
                source_col = position.get(label)
                if source_col is not None:
                    block[indices, target_col] = values[:, source_col]
        blocks.append(block)
        schema.extend(f"source_e:{node}:{label}" for label in labels)

    predictions = np.empty(n_rows, dtype=object)
    for source in SOURCES:
        indices = np.flatnonzero(sources == source)
        if len(indices):
            predictions[indices] = predict_tree(models[source], [texts[index] for index in indices])
    hard = np.column_stack([predictions == label for label in ALL_CLASSES]).astype(np.float32)
    blocks.append(hard)
    schema.extend(f"source_e:prediction:{label}" for label in ALL_CLASSES)
    source_bits = np.column_stack([sources == source for source in SOURCES]).astype(np.float32)
    blocks.append(source_bits)
    schema.extend(f"source={source}" for source in SOURCES)
    return np.column_stack(blocks), schema, predictions


def selector(trees: int, leaf: int, n_jobs: int) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=trees,
        min_samples_leaf=leaf,
        max_features=0.7,
        class_weight="balanced_subsample",
        n_jobs=n_jobs,
        random_state=42,
    )


def clone(model):
    return model.__class__(**model.get_params())


def evaluate_selector(features: np.ndarray, labels: np.ndarray, groups: np.ndarray, prototype) -> tuple[np.ndarray, float, list[float]]:
    predicted = np.empty(len(labels), dtype=object)
    fold_scores: list[float] = []
    for train_index, test_index in GroupKFold(n_splits=5).split(features, labels, groups):
        model = clone(prototype)
        model.fit(features[train_index], labels[train_index])
        predicted[test_index] = model.predict(features[test_index])
        fold_scores.append(f1_score(labels[test_index], predicted[test_index], labels=ALL_CLASSES, average="macro", zero_division=0))
    score = f1_score(labels, predicted, labels=ALL_CLASSES, average="macro", zero_division=0)
    return predicted, score, fold_scores


def val_phase(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    samples = load_jsonl(str(Path(args.data_dir) / "train.jsonl"))
    label_map = load_labels(str(Path(args.data_dir) / "train_labels.csv"))
    split = load_split(args.split_path)
    ids, e_texts = build_modeling_e_features(samples)
    b_ids, b_texts = build_modeling_b_features(samples, history_n=3, include_meta=True)
    if ids != b_ids:
        raise ValueError("E/B feature IDs are misaligned")
    val_index = np.asarray([index for index, sample_id in enumerate(ids) if split[sample_id] == "val"])
    train_index = np.asarray([index for index, sample_id in enumerate(ids) if split[sample_id] == "train"])
    y = np.asarray([label_map[ids[index]] for index in val_index])
    groups = np.asarray([ids[index].split("-step_", 1)[0] for index in val_index])
    all_sources = np.asarray([source_of(sample_id) for sample_id in ids])

    i_bundle = joblib.load(args.i_bundle)
    base, base_schema, _ = stack_features(
        i_bundle["e_model"], i_bundle["d_model"], i_bundle["b_model"],
        [e_texts[index] for index in val_index], [b_texts[index] for index in val_index],
    )
    source_models = fit_source_models(
        [e_texts[index] for index in train_index],
        np.asarray([label_map[ids[index]] for index in train_index]),
        all_sources[train_index],
    )
    extra, extra_schema, source_prediction = source_e_features(
        source_models, [e_texts[index] for index in val_index], all_sources[val_index]
    )
    combined = np.column_stack((base, extra)).astype(np.float32, copy=False)
    prototype = selector(args.trees, args.leaf, args.n_jobs)
    base_prediction, base_score, base_folds = evaluate_selector(base, y, groups, prototype)
    prediction, score, folds = evaluate_selector(combined, y, groups, prototype)
    source_score = f1_score(y, source_prediction, labels=ALL_CLASSES, average="macro", zero_division=0)

    rows = [
        {"experiment": "R0_i3_reproduction", "oof_macro_f1": base_score, "fold_scores": ";".join(f"{value:.6f}" for value in base_folds)},
        {"experiment": "R1_i3_plus_source_e", "oof_macro_f1": score, "fold_scores": ";".join(f"{value:.6f}" for value in folds)},
    ]
    with (out_dir / "results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    (out_dir / "R1_classification_report.txt").write_text(
        classification_report(y, prediction, labels=ALL_CLASSES, digits=4, zero_division=0), encoding="utf-8"
    )
    np.savez_compressed(
        out_dir / "val_oof_predictions.npz", ids=np.asarray([ids[index] for index in val_index]), y=y,
        i3=base_prediction, source_e=source_prediction, source_stack=prediction,
    )
    (out_dir / "feature_schema.json").write_text(json.dumps(base_schema + extra_schema, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "summary.json").write_text(json.dumps({
        "source_e_macro_f1": source_score,
        "i3_reproduction_macro_f1": base_score,
        "source_stack_macro_f1": score,
        "delta": score - base_score,
        "source_counts": {source: int(np.sum(all_sources[val_index] == source)) for source in SOURCES},
        "note": "Validation-only selector; use --phase full-oof for a candidate fitted on leakage-safe full OOF features.",
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"source_e={source_score:.6f} i3={base_score:.6f} source_stack={score:.6f} delta={score-base_score:+.6f}", flush=True)


def full_oof_phase(args: argparse.Namespace) -> None:
    """Build a full-data candidate without touching the live submission model.

    The original E/D/B OOF cache must have been generated by Modeling N using
    the same GroupKFold definition.  Source-E OOF features are regenerated
    here fold-by-fold, then the selector is fit only on OOF rows.
    """
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    samples = load_jsonl(str(Path(args.data_dir) / "train.jsonl"))
    label_map = load_labels(str(Path(args.data_dir) / "train_labels.csv"))
    ids, e_texts = build_modeling_e_features(samples)
    labels = np.asarray([label_map[sample_id] for sample_id in ids])
    sources = np.asarray([source_of(sample_id) for sample_id in ids])
    groups = np.asarray([sample_id.split("-step_", 1)[0] for sample_id in ids])
    folds = list(GroupKFold(n_splits=5).split(np.zeros(len(ids)), labels, groups))
    source_oof = None; source_schema = None
    for fold, (train_index, test_index) in enumerate(folds, start=1):
        models = fit_source_models([e_texts[index] for index in train_index], labels[train_index], sources[train_index])
        current, schema, _ = source_e_features(models, [e_texts[index] for index in test_index], sources[test_index])
        if source_oof is None:
            source_oof = np.zeros((len(ids), current.shape[1]), dtype=np.float32); source_schema = schema
        if schema != source_schema:
            raise ValueError("Source-E feature schema changed across folds")
        source_oof[test_index] = current
        print(f"source_e_fold={fold}/5 rows={len(test_index)}", flush=True)
    assert source_oof is not None and source_schema is not None
    np.save(out_dir / "oof_source_e_features.npy", source_oof)
    (out_dir / "source_e_schema.json").write_text(json.dumps(source_schema, ensure_ascii=False, indent=2), encoding="utf-8")

    n_dir = Path(args.n_oof_dir)
    base_oof = np.load(n_dir / "oof_stack_features.npy")
    if len(base_oof) != len(labels):
        raise ValueError("Modeling N OOF cache does not align with the training data")
    feature_oof = np.column_stack((base_oof, source_oof)).astype(np.float32, copy=False)
    prototype = selector(args.trees, args.leaf, args.n_jobs)
    prediction, score, fold_scores = evaluate_selector(feature_oof, labels, groups, prototype)
    (out_dir / "full_oof_report.txt").write_text(
        classification_report(labels, prediction, labels=ALL_CLASSES, digits=4, zero_division=0), encoding="utf-8"
    )
    print(f"full_oof_source_stack={score:.6f} folds={fold_scores}", flush=True)

    # Reuse full-data E/D/B base models from the live bundle read-only, then
    # add independently full-refit source hierarchies and a selector trained
    # strictly on OOF features.  The candidate location is intentionally not
    # ``model/``.
    live_bundle = joblib.load("model/best_model.pkl")
    full_source_models = fit_source_models(e_texts, labels, sources)
    full_selector = clone(prototype).fit(feature_oof, labels)
    candidate = dict(live_bundle)
    candidate.update({
        "model_type": "stacked_selector_source_e_v1",
        "source_e_models": full_source_models,
        "source_e_schema": source_schema,
        "source_selector": full_selector,
        "source_values": list(SOURCES),
    })
    candidate_dir = Path(args.candidate_dir); candidate_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(candidate, candidate_dir / "best_model.pkl", compress=3)
    (candidate_dir / "best_model_metadata.json").write_text(json.dumps({
        "best_experiment": "R1_full_oof_i3_plus_source_e",
        "best_macro_f1": score,
        "validation_method": "5-fold GroupKFold full-data OOF base features plus source-routed E OOF features",
        "feature_builder": "stacked_source_e_v1",
        "model_type": "stacked_selector_source_e_v1",
        "refit_full": True,
        "all_classes": ALL_CLASSES,
        "source_values": list(SOURCES),
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (out_dir / "full_oof_summary.json").write_text(json.dumps({
        "macro_f1": score, "fold_scores": fold_scores,
        "candidate": str(candidate_dir / "best_model.pkl"),
        "warning": "Candidate needs explicit script.py support and normal ZIP checks before promotion.",
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.phase == "val":
        val_phase(args)
    else:
        full_oof_phase(args)


if __name__ == "__main__":
    main()
