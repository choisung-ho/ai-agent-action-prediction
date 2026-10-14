#!/usr/bin/env python3
"""Independent common-split audit for an AU-only 2032-feature selector.

Unlike the fast AU audit, this program never lets common-validation session
labels enter a base/source feature fit.  It follows the real stacking contract:

* inner GroupKFold on common-train sessions creates E/D/B + source-E OOF
  blocks for selector training;
* E/D/B + source-E models then fit on *all* common-train sessions create the
  common-validation blocks;
* a global raw selector and an AU-only selector train only on those inner OOF
  blocks, then predict common validation.

All writes are under an experiment directory.  The live bundle, script, and
submission archive are intentionally untouched.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from script import ALL_CLASSES_LIST, build_state_dict, build_user_sequence_hash
from data import load_jsonl, load_labels
from train_modeling_b import build_modeling_b_features, make_classifier
from train_modeling_d import fit_two_stage
from train_modeling_e import build_modeling_e_features, fit_tree
from train_modeling_i import stack_features
from train_modeling_r_source_stack import fit_source_models, source_e_features, source_of


N_CLASSES = len(ALL_CLASSES_LIST)
LABEL_TO_ID = {label: index for index, label in enumerate(ALL_CLASSES_LIST)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="experiments/au_source_strict_common")
    parser.add_argument("--stage", choices=("features", "selectors", "all"), default="all")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--n-jobs", type=int, default=4)
    parser.add_argument("--folds", type=int, default=5)
    return parser.parse_args()


def macro(target: np.ndarray, probability: np.ndarray, mask: np.ndarray | None = None) -> float:
    if mask is None:
        mask = np.ones(len(target), dtype=bool)
    return float(
        f1_score(
            target[mask], probability[mask].argmax(axis=1), labels=np.arange(N_CLASSES),
            average="macro", zero_division=0,
        )
    )


def terminal_mask(ids: np.ndarray) -> np.ndarray:
    maximum: dict[str, int] = {}
    for sample_id in ids:
        session, step = str(sample_id).rsplit("-step_", 1)
        maximum[session] = max(maximum.get(session, -1), int(step))
    return np.asarray(
        [int(str(sample_id).rsplit("-step_", 1)[1]) == maximum[str(sample_id).rsplit("-step_", 1)[0]] for sample_id in ids],
        dtype=bool,
    )


def geometric(left: np.ndarray, right: np.ndarray, raw_alpha: float) -> np.ndarray:
    value = raw_alpha * np.log(np.maximum(left, 1e-12)) + (1.0 - raw_alpha) * np.log(np.maximum(right, 1e-12))
    value -= value.max(axis=1, keepdims=True)
    value = np.exp(value)
    return value / value.sum(axis=1, keepdims=True)


def load_context() -> tuple[list[dict], np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    samples = load_jsonl(str(ROOT / "data/train.jsonl"))
    labels = load_labels(str(ROOT / "data/train_labels.csv"))
    split = {row["id"]: row["split"] for row in csv.DictReader((ROOT / "splits/common_split.csv").open(encoding="utf-8"))}
    ids = np.asarray([str(sample["id"]) for sample in samples])
    target = np.asarray([LABEL_TO_ID[labels[sample_id]] for sample_id in ids], dtype=np.int64)
    train = np.asarray([split[sample_id] == "train" for sample_id in ids], dtype=bool)
    valid = ~train
    source = np.asarray([source_of(sample_id) for sample_id in ids])
    return samples, ids, target, train, valid, source


def _fit_base(
    e_texts: list[str], b_texts: list[str], labels: np.ndarray, train: np.ndarray, predict: np.ndarray,
) -> tuple[np.ndarray, list[str]]:
    """Fit exactly the deployed E/D/B components and score a disjoint split."""
    e_model = fit_tree([e_texts[index] for index in train], labels[train].tolist(), c=0.5, ngram_max=3)
    d_model = fit_two_stage(
        [b_texts[index] for index in train], labels[train].tolist(),
        {"coarse_c": 1.2, "fine_c": 0.8, "coarse_max_features": 100_000, "fine_max_features": 80_000},
    )
    b_model = make_classifier("B3")
    b_model.fit([b_texts[index] for index in train], labels[train].tolist())
    values, schema, _ = stack_features(
        e_model, d_model, b_model,
        [e_texts[index] for index in predict], [b_texts[index] for index in predict],
    )
    return values.astype(np.float32), schema


def _fit_source(
    e_texts: list[str], labels: np.ndarray, source: np.ndarray, train: np.ndarray, predict: np.ndarray,
) -> tuple[np.ndarray, list[str]]:
    models = fit_source_models([e_texts[index] for index in train], labels[train], source[train])
    values, schema, _ = source_e_features(models, [e_texts[index] for index in predict], source[predict])
    return values.astype(np.float32), schema


def prepare_features(out: Path, folds_count: int) -> None:
    cache = out / "strict_blocks.npz"
    if cache.exists():
        saved = np.load(cache, allow_pickle=False)
        required = {"train_index", "valid_index", "target_train", "target_valid", "source_train", "source_valid", "base_train", "base_valid", "source_e_train", "source_e_valid", "state_train", "state_valid", "user_train", "user_valid"}
        if required.issubset(saved.files):
            print(f"features cache exists: {cache}", flush=True)
            return

    samples, ids, target, is_train, is_valid, source = load_context()
    label_text = np.asarray([ALL_CLASSES_LIST[index] for index in target])
    e_ids, e_texts = build_modeling_e_features(samples)
    b_ids, b_texts = build_modeling_b_features(samples, history_n=3, include_meta=True)
    if e_ids != b_ids or list(ids) != e_ids:
        raise ValueError("Sample/E/B order mismatch")
    train_index = np.flatnonzero(is_train)
    valid_index = np.flatnonzero(is_valid)
    train_groups = np.asarray([ids[index].rsplit("-step_", 1)[0] for index in train_index])
    folds = list(GroupKFold(n_splits=folds_count).split(np.zeros(len(train_index)), target[train_index], train_groups))
    base_train: np.ndarray | None = None
    source_train: np.ndarray | None = None
    base_schema: list[str] | None = None
    source_schema: list[str] | None = None
    started = time.time()
    for fold, (local_train, local_valid) in enumerate(folds, start=1):
        inner_train = train_index[local_train]
        inner_valid = train_index[local_valid]
        current_base, current_base_schema = _fit_base(e_texts, b_texts, label_text, inner_train, inner_valid)
        current_source, current_source_schema = _fit_source(e_texts, label_text, source, inner_train, inner_valid)
        if base_train is None:
            base_train = np.zeros((len(train_index), current_base.shape[1]), dtype=np.float32)
            source_train = np.zeros((len(train_index), current_source.shape[1]), dtype=np.float32)
            base_schema, source_schema = current_base_schema, current_source_schema
        if current_base_schema != base_schema or current_source_schema != source_schema:
            raise ValueError("Feature schema changed across strict inner folds")
        base_train[local_valid] = current_base
        source_train[local_valid] = current_source
        print(f"strict_inner_fold={fold}/{folds_count} rows={len(inner_valid)} elapsed={time.time()-started:.1f}s", flush=True)
    assert base_train is not None and source_train is not None and base_schema is not None and source_schema is not None

    # The validation blocks are produced only after the inner OOF train blocks
    # are complete, using all common-train rows and no validation labels.
    base_valid, full_base_schema = _fit_base(e_texts, b_texts, label_text, train_index, valid_index)
    source_valid, full_source_schema = _fit_source(e_texts, label_text, source, train_index, valid_index)
    if full_base_schema != base_schema or full_source_schema != source_schema:
        raise ValueError("Full common-train feature schema differs from inner OOF schema")

    # Fitting a DictVectorizer on all *unlabelled* supplied records merely
    # freezes the inference vocabulary.  It consumes no action labels and
    # mirrors fitting it before a competition test batch is scored.
    vectorizer = DictVectorizer(sparse=False)
    state_all = vectorizer.fit_transform([build_state_dict(sample) for sample in samples]).astype(np.float32)
    user_all = build_user_sequence_hash(samples).astype(np.float32)
    expected = base_train.shape[1] + source_train.shape[1] + state_all.shape[1] + user_all.shape[1]
    if expected != 2032:
        raise ValueError(
            f"Strict feature width must match deployed 2032 (base={base_train.shape[1]} source={source_train.shape[1]} "
            f"state={state_all.shape[1]} user={user_all.shape[1]} => {expected})"
        )
    np.savez_compressed(
        cache,
        train_index=train_index, valid_index=valid_index,
        ids_train=ids[train_index], ids_valid=ids[valid_index],
        target_train=target[train_index], target_valid=target[valid_index],
        source_train=source[train_index], source_valid=source[valid_index],
        base_train=base_train, base_valid=base_valid,
        source_e_train=source_train, source_e_valid=source_valid,
        state_train=state_all[train_index], state_valid=state_all[valid_index],
        user_train=user_all[train_index], user_valid=user_all[valid_index],
    )
    (out / "feature_schema.json").write_text(json.dumps({
        "base": base_schema, "source_e": source_schema,
        "state": vectorizer.feature_names_, "width": int(expected),
        "contract": "inner OOF on common train; full common-train models on common validation; state vocab label-free",
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"strict feature cache saved; train={len(train_index)} valid={len(valid_index)} elapsed={time.time()-started:.1f}s", flush=True)


def xgb_parameters(name: str) -> tuple[dict[str, object], float]:
    if name == "raw_global":
        return dict(n_estimators=210, max_depth=10, learning_rate=0.04, subsample=0.85, colsample_bytree=0.85, min_child_weight=1.0, reg_lambda=1.0), 1.0
    if name == "au_same_raw":
        return dict(n_estimators=210, max_depth=10, learning_rate=0.04, subsample=0.85, colsample_bytree=0.85, min_child_weight=1.0, reg_lambda=1.0), 1.0
    if name == "au_regularized":
        return dict(n_estimators=360, max_depth=7, learning_rate=0.035, subsample=0.90, colsample_bytree=0.80, min_child_weight=3.0, reg_lambda=5.0, gamma=0.05), 0.75
    raise ValueError(name)


def train_selector(name: str, x_train: np.ndarray, y_train: np.ndarray, x_valid: np.ndarray, device: str, n_jobs: int) -> np.ndarray:
    # Import lazily so the CPU-only feature-materialization stage cannot
    # initialize a CUDA/XGBoost context while another GPU validation runs.
    from xgboost import XGBClassifier

    params, power = xgb_parameters(name)
    count = np.bincount(y_train, minlength=N_CLASSES)
    if np.any(count == 0):
        raise ValueError(f"{name} train labels missing: {[ALL_CLASSES_LIST[i] for i in np.flatnonzero(count == 0)]}")
    weights = (len(y_train) / (N_CLASSES * count[y_train])) ** power
    model = XGBClassifier(
        **params, objective="multi:softprob", num_class=N_CLASSES, tree_method="hist", device=device,
        eval_metric="mlogloss", n_jobs=n_jobs, random_state=808,
    )
    model.fit(x_train, y_train, sample_weight=weights, verbose=False)
    return model.predict_proba(x_valid).astype(np.float32)


def assess_selectors(out: Path, device: str, n_jobs: int) -> None:
    saved = np.load(out / "strict_blocks.npz", allow_pickle=False)
    x_train = np.column_stack((saved["base_train"], saved["state_train"], saved["source_e_train"], saved["user_train"])).astype(np.float32)
    x_valid = np.column_stack((saved["base_valid"], saved["state_valid"], saved["source_e_valid"], saved["user_valid"])).astype(np.float32)
    if x_train.shape[1] != 2032 or x_valid.shape[1] != 2032:
        raise ValueError(f"Unexpected strict selector width: train={x_train.shape} valid={x_valid.shape}")
    y_train = saved["target_train"].astype(np.int64)
    y_valid = saved["target_valid"].astype(np.int64)
    source_train = saved["source_train"].astype(str)
    source_valid = saved["source_valid"].astype(str)
    ids_valid = saved["ids_valid"].astype(str)
    terminal = terminal_mask(ids_valid)
    global_probability = train_selector("raw_global", x_train, y_train, x_valid, device, n_jobs)
    au_train = source_train == "au"
    au_valid = source_valid == "au"
    candidate: dict[str, np.ndarray] = {}
    for name in ("au_same_raw", "au_regularized"):
        candidate[name] = train_selector(name, x_train[au_train], y_train[au_train], x_valid[au_valid], device, n_jobs)
        np.save(out / f"{name}_valid_probability.npy", candidate[name])
    np.save(out / "raw_global_valid_probability.npy", global_probability)

    result: dict[str, object] = {
        "scope": "Strict experiment only; live model/script/ZIP are unchanged.",
        "protocol": {
            "outer": "common train/common validation sessions are disjoint",
            "selector_train_features": "5-fold GroupKFold OOF E/D/B + source-E blocks within common train",
            "validation_features": "E/D/B + source-E fit only on all common train",
            "state_user": "inference-safe label-free state vector and fixed FeatureHasher",
        },
        "counts": {
            "train_rows": int(len(y_train)), "valid_rows": int(len(y_valid)),
            "valid_au_rows": int(au_valid.sum()), "valid_au_terminal": int((au_valid & terminal).sum()),
        },
        "raw_global": {
            "all_macro_f1": macro(y_valid, global_probability),
            "au_all_macro_f1": macro(y_valid, global_probability, au_valid),
            "au_terminal_macro_f1": macro(y_valid, global_probability, au_valid & terminal),
        },
        "candidates": {},
    }
    for name, probability in candidate.items():
        raw_au = global_probability[au_valid]
        target_au = y_valid[au_valid]
        terminal_au = terminal[au_valid]
        static_blends = {}
        for alpha in (0.0, 0.25, 0.50, 0.75, 1.0):
            blend = geometric(raw_au, probability, alpha)
            static_blends[f"raw_alpha_{alpha:.2f}"] = {
                "au_all_macro_f1": macro(target_au, blend),
                "au_terminal_macro_f1": macro(target_au, blend, terminal_au),
            }
        # Simulate the public deterministic future-history path: nonterminal
        # rows become exact labels, while **both** SIM and AU terminal rows
        # keep a model prediction.  Starting from y_valid would accidentally
        # make SIM terminal rows perfect and inflate this diagnostic.
        hybrid = global_probability.argmax(axis=1)
        hybrid[~terminal] = y_valid[~terminal]
        hybrid[au_valid & terminal] = probability[terminal_au].argmax(axis=1)
        result["candidates"][name] = {
            "au_all_macro_f1": macro(target_au, probability),
            "au_terminal_macro_f1": macro(target_au, probability, terminal_au),
            "delta_terminal_vs_raw": macro(target_au, probability, terminal_au) - macro(y_valid, global_probability, au_valid & terminal),
            "static_geometric_blends": static_blends,
            "simulated_future_recovery_au_terminal_replaced_full_macro_f1": float(
                f1_score(y_valid, hybrid, labels=np.arange(N_CLASSES), average="macro", zero_division=0)
            ),
        }
    (out / "strict_report.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


def main() -> None:
    args = parse_args()
    out = ROOT / args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    if args.stage in {"features", "all"}:
        prepare_features(out, args.folds)
    if args.stage in {"selectors", "all"}:
        if not (out / "strict_blocks.npz").exists():
            raise FileNotFoundError("Run --stage features first")
        assess_selectors(out, args.device, args.n_jobs)


if __name__ == "__main__":
    main()
