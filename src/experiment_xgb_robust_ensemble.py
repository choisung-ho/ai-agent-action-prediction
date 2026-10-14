#!/usr/bin/env python3
"""Audit a small, pre-registered robust XGBoost selector ensemble.

This experiment is deliberately isolated from the submission artifact.  It
answers one narrow question: can a modest seed/regularization ensemble improve
the deployed 2032-feature raw selector *before* the terminal char20 override?

Validation has two independent parts:

* full-data 5-fold session GroupKFold OOF, using the existing leakage-safe
  E/D/B and source-E OOF blocks; and
* the untouched common fixed split, using ``strict_blocks.npz`` where every
  selector-training block is inner-OOF within common-train and validation
  blocks are fitted only on common-train labels.

No model/, script.py, or ZIP path is ever written here.  The four candidates
below are fixed before seeing this run; the report includes equal and
probability-geometric ensembles, but recommends nothing unless both gates beat
the matching raw selector + strict-distilled no-E5 GRU geometry.
"""

from __future__ import annotations

import argparse
import gc
import csv
import json
import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from data import load_jsonl, load_labels
from script import ALL_CLASSES_LIST, build_state_dict, build_user_sequence_hash
from train_modeling_r_source_stack import source_of


N_CLASSES = len(ALL_CLASSES_LIST)
LABEL_TO_ID = {label: number for number, label in enumerate(ALL_CLASSES_LIST)}

# This is intentionally a small, pre-registered set rather than a broad grid.
# ``raw_seed42`` exactly reproduces the full-OOF live-selector proxy.  The
# strict common counterpart was run with seed 808 by the independent AU audit,
# so both seed variants are retained to make the comparison transparent.
CANDIDATES: dict[str, dict[str, object]] = {
    "raw_seed42": {
        "n_estimators": 210, "max_depth": 10, "learning_rate": 0.04,
        "subsample": 0.85, "colsample_bytree": 0.85,
        "min_child_weight": 1.0, "reg_lambda": 1.0, "gamma": 0.0,
        "weight_power": 1.0, "random_state": 42,
    },
    "raw_seed808": {
        "n_estimators": 210, "max_depth": 10, "learning_rate": 0.04,
        "subsample": 0.85, "colsample_bytree": 0.85,
        "min_child_weight": 1.0, "reg_lambda": 1.0, "gamma": 0.0,
        "weight_power": 1.0, "random_state": 808,
    },
    "reg_d9": {
        "n_estimators": 270, "max_depth": 9, "learning_rate": 0.035,
        "subsample": 0.90, "colsample_bytree": 0.80,
        "min_child_weight": 2.0, "reg_lambda": 3.0, "gamma": 0.05,
        "weight_power": 1.0, "random_state": 142,
    },
    "reg_d8": {
        "n_estimators": 360, "max_depth": 8, "learning_rate": 0.035,
        "subsample": 0.90, "colsample_bytree": 0.80,
        "min_child_weight": 3.0, "reg_lambda": 5.0, "gamma": 0.05,
        "weight_power": 0.85, "random_state": 242,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("full-oof", "fixed", "alpha", "both"), default="both")
    parser.add_argument("--out-dir", default="experiments/xgb_robust_ensemble")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--n-jobs", type=int, default=4)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument(
        "--force", action="store_true",
        help="Recompute existing candidate probabilities.  Normally caches are reused.",
    )
    return parser.parse_args()


def terminal_mask(ids: np.ndarray) -> np.ndarray:
    maximum: dict[str, int] = {}
    for sample_id in ids.astype(str):
        session, step = sample_id.rsplit("-step_", 1)
        maximum[session] = max(maximum.get(session, -1), int(step))
    return np.asarray(
        [int(sample_id.rsplit("-step_", 1)[1]) == maximum[sample_id.rsplit("-step_", 1)[0]] for sample_id in ids.astype(str)],
        dtype=bool,
    )


def score(target: np.ndarray, probability: np.ndarray, mask: np.ndarray | None = None) -> float:
    if mask is None:
        mask = np.ones(len(target), dtype=bool)
    return float(
        f1_score(
            target[mask], probability[mask].argmax(axis=1), labels=np.arange(N_CLASSES),
            average="macro", zero_division=0,
        )
    )


def geometric(*probabilities: np.ndarray, weights: np.ndarray | None = None) -> np.ndarray:
    """A normalized product-of-experts, stable for either row or model weights."""
    if not probabilities:
        raise ValueError("At least one probability array is required")
    n_models = len(probabilities)
    if weights is None:
        weights = np.full(n_models, 1.0 / n_models, dtype=np.float32)
    weights = np.asarray(weights, dtype=np.float32)
    if weights.ndim != 1 or len(weights) != n_models:
        raise ValueError("Model geometric weights must have one value per probability array")
    value = np.zeros_like(probabilities[0], dtype=np.float32)
    for weight, probability in zip(weights, probabilities):
        value += float(weight) * np.log(np.maximum(probability, 1e-12))
    value -= value.max(axis=1, keepdims=True)
    value = np.exp(value)
    return value / value.sum(axis=1, keepdims=True)


def source_gru_geometry(
    xgb: np.ndarray, gru: np.ndarray, source: np.ndarray, sim_alpha: float = 0.20, au_alpha: float = 0.85,
) -> np.ndarray:
    """Source-routed raw-XGB/strict-GRU product-of-experts gate."""
    alpha = np.where(source.astype(str) == "au", au_alpha, sim_alpha).astype(np.float32)
    value = alpha[:, None] * np.log(np.maximum(xgb, 1e-12))
    value += (1.0 - alpha[:, None]) * np.log(np.maximum(gru, 1e-12))
    value -= value.max(axis=1, keepdims=True)
    value = np.exp(value)
    return value / value.sum(axis=1, keepdims=True)


def selector_probability(
    config: dict[str, object], x_train: np.ndarray, y_train: np.ndarray,
    x_predict: np.ndarray, device: str, n_jobs: int,
) -> np.ndarray:
    from xgboost import XGBClassifier

    counts = np.bincount(y_train, minlength=N_CLASSES)
    if np.any(counts == 0):
        missing = [ALL_CLASSES_LIST[index] for index in np.flatnonzero(counts == 0)]
        raise ValueError(f"Selector train partition is missing classes: {missing}")
    power = float(config["weight_power"])
    sample_weight = (len(y_train) / (N_CLASSES * counts[y_train])) ** power
    parameters = {key: value for key, value in config.items() if key not in {"weight_power"}}
    model = XGBClassifier(
        **parameters,
        objective="multi:softprob", num_class=N_CLASSES, tree_method="hist", device=device,
        eval_metric="mlogloss", n_jobs=n_jobs,
    )
    try:
        model.fit(x_train, y_train, sample_weight=sample_weight, verbose=False)
        return model.predict_proba(x_predict).astype(np.float32)
    finally:
        del model
        gc.collect()


def full_context() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return the established 2032 full-data OOF selector contract."""
    samples = load_jsonl(str(ROOT / "data/train.jsonl"))
    labels = load_labels(str(ROOT / "data/train_labels.csv"))
    ids = np.asarray([str(sample["id"]) for sample in samples])
    target = np.asarray([LABEL_TO_ID[labels[sample_id]] for sample_id in ids], dtype=np.int64)
    source = np.asarray([source_of(sample_id) for sample_id in ids])
    groups = np.asarray([sample_id.rsplit("-step_", 1)[0] for sample_id in ids])
    base = np.load(ROOT / "experiments/modeling_n_refit/oof_stack_features.npy", allow_pickle=False).astype(np.float32)
    source_e = np.load(ROOT / "experiments/modeling_ak_live_xgb_proxy/source_oof.npy", allow_pickle=False).astype(np.float32)
    # This vectorizer is the frozen label-free state feature contract used by
    # the existing live-selector proxy.  It is read-only; no submission model
    # is opened or modified.
    q_bundle = joblib.load(ROOT / "model_candidate_q/best_model.pkl")
    state = q_bundle["state_vectorizer"].transform([build_state_dict(sample) for sample in samples]).astype(np.float32)
    user = build_user_sequence_hash(samples).astype(np.float32)
    x = np.column_stack((base, state, source_e, user)).astype(np.float32)
    if x.shape != (len(ids), 2032):
        raise ValueError(f"Full OOF selector contract must be (70000,2032), got {x.shape}")
    return x, target, source, groups, ids


def strict_context() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    saved = np.load(ROOT / "experiments/au_source_strict_common/strict_blocks.npz", allow_pickle=False)
    x_train = np.column_stack((
        saved["base_train"], saved["state_train"], saved["source_e_train"], saved["user_train"],
    )).astype(np.float32)
    x_valid = np.column_stack((
        saved["base_valid"], saved["state_valid"], saved["source_e_valid"], saved["user_valid"],
    )).astype(np.float32)
    if x_train.shape[1] != 2032 or x_valid.shape[1] != 2032:
        raise ValueError(f"Strict selector widths must be 2032, got {x_train.shape} / {x_valid.shape}")
    return (
        x_train, saved["target_train"].astype(np.int64), saved["source_train"].astype(str),
        x_valid, saved["target_valid"].astype(np.int64), saved["source_valid"].astype(str),
    )


def full_gru() -> np.ndarray:
    value = np.load(
        ROOT / "experiments/modeling_ai_sequence_nested_distill_e16_i4/oof_probabilities.npy", allow_pickle=False,
    ).astype(np.float32)
    if value.shape != (70000, N_CLASSES):
        raise ValueError(f"Unexpected strict-distill OOF probability shape {value.shape}")
    return value


def fixed_gru_and_ids() -> tuple[np.ndarray, np.ndarray]:
    probability = np.load(
        ROOT / "experiments/modeling_ai_sequence_fusion_fixed_strictdistill_w35/valid_probabilities.npy",
        allow_pickle=False,
    ).astype(np.float32)
    index = np.load(
        ROOT / "experiments/modeling_ai_sequence_fusion_fixed_strictdistill_w35/valid_indices.npy",
        allow_pickle=False,
    ).astype(np.int64)
    samples = load_jsonl(str(ROOT / "data/train.jsonl"))
    ids = np.asarray([str(samples[number]["id"]) for number in index])
    if probability.shape != (len(ids), N_CLASSES):
        raise ValueError("Fixed strict-distill rows and probabilities do not align")
    return probability, ids


def full_char20() -> np.ndarray:
    value = np.load(ROOT / "experiments/terminal_rule_combinations/oof_candidates.npz", allow_pickle=False)["char20"].astype(np.int16)
    if value.shape != (70000,):
        raise ValueError(f"Unexpected full char20 shape {value.shape}")
    return value


def fixed_char20(ids: np.ndarray, terminal: np.ndarray) -> np.ndarray:
    saved = np.load(ROOT / "experiments/terminal_rule_combinations/oof_candidates.npz", allow_pickle=False)
    terminal_ids = saved["valid_ids"].astype(str)
    labels = saved["fixed_char20"].astype(np.int16)
    if len(terminal_ids) != len(labels):
        raise ValueError("Fixed char20 IDs and labels differ in length")
    mapping = dict(zip(terminal_ids, labels, strict=True))
    value = np.full(len(ids), -1, dtype=np.int16)
    for number, sample_id in enumerate(ids.astype(str)):
        if terminal[number] and sample_id in mapping:
            value[number] = mapping[sample_id]
    if set(terminal_ids) != set(ids[terminal].astype(str)):
        raise ValueError("Fixed char20 terminal IDs do not match strict fixed terminal IDs")
    return value


def apply_char20(probability: np.ndarray, char_label: np.ndarray) -> np.ndarray:
    result = probability.copy()
    apply = char_label >= 0
    if np.any(apply):
        result[apply] = 0.0
        result[np.flatnonzero(apply), char_label[apply]] = 1.0
    return result


def metrics(
    target: np.ndarray, probability: np.ndarray, terminal: np.ndarray, char_label: np.ndarray,
) -> dict[str, float]:
    after_char = apply_char20(probability, char_label)
    return {
        "all_macro_f1_before_char20": score(target, probability),
        "terminal_macro_f1_before_char20": score(target, probability, terminal),
        "all_macro_f1_with_char20": score(target, after_char),
        "terminal_macro_f1_with_char20": score(target, after_char, terminal),
    }


def ensemble_probability(probabilities: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    names = ("raw_seed42", "raw_seed808", "reg_d9", "reg_d8")
    if not all(name in probabilities for name in names):
        return {}
    seed_pair = [probabilities["raw_seed42"], probabilities["raw_seed808"]]
    robust = [probabilities[name] for name in names]
    return {
        "seed_pair_equal": np.mean(seed_pair, axis=0).astype(np.float32),
        "seed_pair_geometric": geometric(*seed_pair),
        "robust4_equal": np.mean(robust, axis=0).astype(np.float32),
        "robust4_geometric": geometric(*robust),
    }


def write_json(path: Path, value: dict[str, object]) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def run_full_oof(args: argparse.Namespace, out: Path) -> None:
    x, target, source, groups, ids = full_context()
    terminal = terminal_mask(ids)
    gru = full_gru()
    char = full_char20()
    if not np.array_equal(target, np.load(ROOT / "experiments/modeling_ai_sequence_nested_distill_e16_i4/oof_targets.npy", allow_pickle=False)):
        raise ValueError("Strict-distill full OOF target order does not match raw selector order")
    probabilities: dict[str, np.ndarray] = {
        "raw_seed42": np.load(ROOT / "experiments/modeling_ak_live_xgb_proxy/oof_probabilities.npy", allow_pickle=False).astype(np.float32),
    }
    folds = list(GroupKFold(n_splits=args.folds).split(x, target, groups))
    for name, config in CANDIDATES.items():
        cache = out / f"full_oof_{name}.npy"
        if name == "raw_seed42":
            np.save(cache, probabilities[name])
            continue
        if cache.exists() and not args.force:
            probability = np.load(cache, allow_pickle=False).astype(np.float32)
            if probability.shape != (len(target), N_CLASSES):
                raise ValueError(f"Bad cache shape for {cache}: {probability.shape}")
            probabilities[name] = probability
            print(f"full_oof {name}: reused {cache.name}", flush=True)
            continue
        probability = np.zeros((len(target), N_CLASSES), dtype=np.float32)
        for fold, (train_index, valid_index) in enumerate(folds, start=1):
            probability[valid_index] = selector_probability(
                config, x[train_index], target[train_index], x[valid_index], args.device, args.n_jobs,
            )
            print(f"full_oof {name} fold={fold}/{args.folds}", flush=True)
        np.save(cache, probability)
        probabilities[name] = probability

    candidates: dict[str, np.ndarray] = {**probabilities, **ensemble_probability(probabilities)}
    report: dict[str, object] = {
        "scope": "Experiment only; model/, script.py, and submission ZIP are unchanged.",
        "protocol": {
            "selector_features": "Existing 5-fold session-GroupKFold E/D/B and source-E OOF blocks; state/hash are label-free inference-safe features.",
            "selector_evaluation": "Each XGBoost candidate is fit on the four session folds and predicts the held session fold; no early stopping uses held labels.",
            "gru": "Strict nested-distilled no-E5 GRU OOF probabilities aligned by target order.",
            "char20": "Reported only as a downstream terminal override; selection fields are explicitly before_char20.",
        },
        "counts": {"rows": int(len(target)), "terminal": int(terminal.sum())},
        "candidate_config": CANDIDATES,
        "raw_selector": {},
        "with_strict_distill_geometric": {},
    }
    for name, probability in candidates.items():
        report["raw_selector"][name] = metrics(target, probability, terminal, char)
        blended = source_gru_geometry(probability, gru, source)
        report["with_strict_distill_geometric"][name] = metrics(target, blended, terminal, char)
    write_json(out / "full_oof_report.json", report)
    print(json.dumps(report["with_strict_distill_geometric"], ensure_ascii=False, indent=2), flush=True)


def run_fixed(args: argparse.Namespace, out: Path) -> None:
    x_train, y_train, _source_train, x_valid, target, source = strict_context()
    saved = np.load(ROOT / "experiments/au_source_strict_common/strict_blocks.npz", allow_pickle=False)
    ids = saved["ids_valid"].astype(str)
    terminal = terminal_mask(ids)
    gru, gru_ids = fixed_gru_and_ids()
    if not np.array_equal(ids, gru_ids):
        raise ValueError("Fixed strict-distill probability IDs do not match strict selector IDs")
    char = fixed_char20(ids, terminal)
    probabilities: dict[str, np.ndarray] = {}
    existing_seed808 = ROOT / "experiments/au_source_strict_common/raw_global_valid_probability.npy"
    for name, config in CANDIDATES.items():
        cache = out / f"strict_fixed_{name}.npy"
        if name == "raw_seed808" and existing_seed808.exists() and not args.force:
            probability = np.load(existing_seed808, allow_pickle=False).astype(np.float32)
            np.save(cache, probability)
            probabilities[name] = probability
            print(f"strict_fixed {name}: reused independent AU-audit raw_global probability", flush=True)
            continue
        if cache.exists() and not args.force:
            probability = np.load(cache, allow_pickle=False).astype(np.float32)
            if probability.shape != (len(target), N_CLASSES):
                raise ValueError(f"Bad cache shape for {cache}: {probability.shape}")
            probabilities[name] = probability
            print(f"strict_fixed {name}: reused {cache.name}", flush=True)
            continue
        probability = selector_probability(config, x_train, y_train, x_valid, args.device, args.n_jobs)
        np.save(cache, probability)
        probabilities[name] = probability
        print(f"strict_fixed {name}: trained", flush=True)

    candidates: dict[str, np.ndarray] = {**probabilities, **ensemble_probability(probabilities)}
    report: dict[str, object] = {
        "scope": "Experiment only; model/, script.py, and submission ZIP are unchanged.",
        "protocol": {
            "outer": "Common-train and common-validation sessions are disjoint.",
            "selector_training_features": "5-fold GroupKFold E/D/B + source-E OOF blocks within common-train.",
            "validation_features": "E/D/B + source-E models fit on common-train only.",
            "gru": "Strict common-train teacher/distilled no-E5 GRU probability artifact; IDs asserted identical.",
            "char20": "Reported only after the selection metric, and only on terminal rows.",
        },
        "counts": {"train_rows": int(len(y_train)), "valid_rows": int(len(target)), "terminal": int(terminal.sum())},
        "candidate_config": CANDIDATES,
        "raw_selector": {},
        "with_strict_distill_geometric": {},
    }
    for name, probability in candidates.items():
        report["raw_selector"][name] = metrics(target, probability, terminal, char)
        blended = source_gru_geometry(probability, gru, source)
        report["with_strict_distill_geometric"][name] = metrics(target, blended, terminal, char)
    write_json(out / "strict_fixed_report.json", report)
    print(json.dumps(report["with_strict_distill_geometric"], ensure_ascii=False, indent=2), flush=True)


def promotion_gate(out: Path) -> None:
    full_path, fixed_path = out / "full_oof_report.json", out / "strict_fixed_report.json"
    if not (full_path.exists() and fixed_path.exists()):
        return
    full = json.loads(full_path.read_text(encoding="utf-8"))["with_strict_distill_geometric"]
    fixed = json.loads(fixed_path.read_text(encoding="utf-8"))["with_strict_distill_geometric"]
    full_base = full["raw_seed42"]
    fixed_base = fixed["raw_seed808"]
    decisions: dict[str, object] = {
        "rule": "A candidate must strictly improve both all and terminal macro-F1 before char20 versus raw_seed42 on full GroupOOF and raw_seed808 on strict common fixed validation.",
        "full_baseline": full_base,
        "strict_fixed_baseline": fixed_base,
        "eligible": {},
        "recommendation": "No full-refit candidate is materialized by this experiment unless the two independent gates both pass.",
    }
    for name in full:
        if name in {"raw_seed42", "raw_seed808"}:
            continue
        full_value = full[name]
        fixed_value = fixed[name]
        passed = (
            full_value["all_macro_f1_before_char20"] > full_base["all_macro_f1_before_char20"]
            and full_value["terminal_macro_f1_before_char20"] > full_base["terminal_macro_f1_before_char20"]
            and fixed_value["all_macro_f1_before_char20"] > fixed_base["all_macro_f1_before_char20"]
            and fixed_value["terminal_macro_f1_before_char20"] > fixed_base["terminal_macro_f1_before_char20"]
        )
        decisions["eligible"][name] = {
            "pass": bool(passed),
            "full_delta_all": full_value["all_macro_f1_before_char20"] - full_base["all_macro_f1_before_char20"],
            "full_delta_terminal": full_value["terminal_macro_f1_before_char20"] - full_base["terminal_macro_f1_before_char20"],
            "fixed_delta_all": fixed_value["all_macro_f1_before_char20"] - fixed_base["all_macro_f1_before_char20"],
            "fixed_delta_terminal": fixed_value["terminal_macro_f1_before_char20"] - fixed_base["terminal_macro_f1_before_char20"],
        }
    if any(value["pass"] for value in decisions["eligible"].values()):
        decisions["recommendation"] = (
            "One or more pre-registered candidates passed both gates. A separate full-refit portable ensemble "
            "may now be built and independently smoke-tested; this script intentionally does not alter live assets."
        )
    write_json(out / "promotion_gate.json", decisions)


ALPHA_GRID = tuple((sim, au) for sim in (0.15, 0.20, 0.25) for au in (0.75, 0.80, 0.85, 0.90))


def alpha_key(sim: float, au: float) -> str:
    return f"sim_{sim:.2f}_au_{au:.2f}"


def alpha_selection_key(value: dict[str, float], sim: float, au: float) -> tuple[float, float, float, str]:
    """Pre-registered terminal-first selector with deterministic conservative ties."""
    return (
        value["terminal_macro_f1_before_char20"],
        value["all_macro_f1_before_char20"],
        -(abs(sim - 0.20) + abs(au - 0.85)),
        alpha_key(sim, au),
    )


def alpha_metrics(
    target: np.ndarray, xgb: np.ndarray, gru: np.ndarray, source: np.ndarray, terminal: np.ndarray,
    sim: float, au: float,
) -> dict[str, float]:
    probability = source_gru_geometry(xgb, gru, source, sim, au)
    return {
        "all_macro_f1_before_char20": score(target, probability),
        "terminal_macro_f1_before_char20": score(target, probability, terminal),
    }


def load_cached_probability(out: Path, prefix: str, name: str) -> np.ndarray:
    path = out / f"{prefix}_{name}.npy"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}; run the corresponding selector stage first")
    return np.load(path, allow_pickle=False).astype(np.float32)


def run_alpha_grid(out: Path) -> None:
    """Audit a 3x4, source-only alpha grid without retraining any base learner.

    The global alpha for a candidate is chosen from its full GroupOOF only.
    The fixed score is then a genuinely unseen check.  We also record a nested
    second-level OOF value (other four session folds choose alpha for the held
    fold) so a lucky static setting is visible rather than hidden.
    """
    _x, full_y, full_source, _groups, full_ids = full_context()
    full_terminal = terminal_mask(full_ids)
    full_student = full_gru()
    fold = np.load(
        ROOT / "experiments/modeling_ai_sequence_nested_distill_e16_i4/outer_fold_assignment.npy",
        allow_pickle=False,
    ).astype(np.int8)
    if fold.shape != full_y.shape or len(np.unique(fold)) != 5:
        raise ValueError("Strict nested student fold assignment is unavailable or invalid")

    _x_train, _y_train, _source_train, _x_valid, fixed_y, fixed_source = strict_context()
    strict_saved = np.load(ROOT / "experiments/au_source_strict_common/strict_blocks.npz", allow_pickle=False)
    fixed_ids = strict_saved["ids_valid"].astype(str)
    fixed_terminal = terminal_mask(fixed_ids)
    fixed_student, fixed_student_ids = fixed_gru_and_ids()
    if not np.array_equal(fixed_ids, fixed_student_ids):
        raise ValueError("Strict fixed GRU IDs do not align with selector IDs")

    names = tuple(CANDIDATES) + ("seed_pair_equal", "seed_pair_geometric", "robust4_equal", "robust4_geometric")
    full_raw = {name: load_cached_probability(out, "full_oof", name) for name in CANDIDATES}
    fixed_raw = {name: load_cached_probability(out, "strict_fixed", name) for name in CANDIDATES}
    full_probability = {**full_raw, **ensemble_probability(full_raw)}
    fixed_probability = {**fixed_raw, **ensemble_probability(fixed_raw)}
    if tuple(full_probability) != names or tuple(fixed_probability) != names:
        raise ValueError("Unexpected candidate probability cache names")

    result: dict[str, object] = {
        "scope": "Second-level alpha audit only; no live model/script/ZIP changed.",
        "grid": [{"sim_xgb_alpha": sim, "au_xgb_alpha": au} for sim, au in ALPHA_GRID],
        "selection": {
            "global": "Choose each candidate's alpha on full strict GroupOOF by terminal macro-F1, then all macro-F1, then nearest current alpha.",
            "nested_support": "For each strict student outer fold, choose alpha on the other four folds using the same rule before scoring held rows.",
            "fixed_gate": "Use the global full-OOF selected alpha unchanged on independent strict common validation.",
        },
        "baseline_current_alpha": {"sim_xgb_alpha": 0.20, "au_xgb_alpha": 0.85},
        "candidates": {},
    }

    for name in names:
        grid: dict[str, dict[str, float]] = {}
        for sim, au in ALPHA_GRID:
            grid[alpha_key(sim, au)] = alpha_metrics(
                full_y, full_probability[name], full_student, full_source, full_terminal, sim, au,
            )
        selected_key, selected = max(
            ((key, value) for key, value in grid.items()),
            key=lambda item: alpha_selection_key(
                item[1], float(item[0].split("_")[1]), float(item[0].split("_")[3]),
            ),
        )
        _prefix, sim_text, _middle, au_text = selected_key.split("_")
        selected_sim, selected_au = float(sim_text), float(au_text)
        fixed_value = alpha_metrics(
            fixed_y, fixed_probability[name], fixed_student, fixed_source, fixed_terminal, selected_sim, selected_au,
        )

        # Fully nested alpha selection is only an audit; a globally selected
        # deployment alpha is still evaluated once on the untouched fixed set.
        nested_probability = np.zeros_like(full_student)
        nested_rows: list[dict[str, object]] = []
        for held in sorted(np.unique(fold)):
            train_mask = fold != held
            held_mask = ~train_mask
            fold_grid: list[tuple[float, float, dict[str, float]]] = []
            for sim, au in ALPHA_GRID:
                value = alpha_metrics(
                    full_y[train_mask], full_probability[name][train_mask], full_student[train_mask],
                    full_source[train_mask], full_terminal[train_mask], sim, au,
                )
                fold_grid.append((sim, au, value))
            sim, au, _value = max(fold_grid, key=lambda item: alpha_selection_key(item[2], item[0], item[1]))
            nested_probability[held_mask] = source_gru_geometry(
                full_probability[name][held_mask], full_student[held_mask], full_source[held_mask], sim, au,
            )
            nested_rows.append({"held_fold": int(held), "sim_xgb_alpha": sim, "au_xgb_alpha": au})
        nested_value = {
            "all_macro_f1_before_char20": score(full_y, nested_probability),
            "terminal_macro_f1_before_char20": score(full_y, nested_probability, full_terminal),
        }
        result["candidates"][name] = {
            "full_grid": grid,
            "selected_from_full_oof": {
                "key": selected_key, "sim_xgb_alpha": selected_sim, "au_xgb_alpha": selected_au,
                **selected,
            },
            "fixed_at_full_oof_selected_alpha": fixed_value,
            "nested_full_oof_support": {**nested_value, "fold_selected_alpha": nested_rows},
        }

    baseline = result["candidates"]["raw_seed42"]
    fixed_baseline = baseline["fixed_at_full_oof_selected_alpha"]
    full_baseline = baseline["selected_from_full_oof"]
    eligibility: dict[str, object] = {}
    for name, value in result["candidates"].items():
        if name == "raw_seed42":
            continue
        current_full = value["selected_from_full_oof"]
        current_fixed = value["fixed_at_full_oof_selected_alpha"]
        passed = (
            current_full["all_macro_f1_before_char20"] > full_baseline["all_macro_f1_before_char20"]
            and current_full["terminal_macro_f1_before_char20"] > full_baseline["terminal_macro_f1_before_char20"]
            and current_fixed["all_macro_f1_before_char20"] > fixed_baseline["all_macro_f1_before_char20"]
            and current_fixed["terminal_macro_f1_before_char20"] > fixed_baseline["terminal_macro_f1_before_char20"]
        )
        eligibility[name] = {
            "pass": bool(passed),
            "full_delta_all": current_full["all_macro_f1_before_char20"] - full_baseline["all_macro_f1_before_char20"],
            "full_delta_terminal": current_full["terminal_macro_f1_before_char20"] - full_baseline["terminal_macro_f1_before_char20"],
            "fixed_delta_all": current_fixed["all_macro_f1_before_char20"] - fixed_baseline["all_macro_f1_before_char20"],
            "fixed_delta_terminal": current_fixed["terminal_macro_f1_before_char20"] - fixed_baseline["terminal_macro_f1_before_char20"],
        }
    result["promotion_gate"] = {
        "rule": "Non-baseline candidate must beat raw_seed42 with its independently full-OOF selected alpha on all and terminal macro-F1 in both full OOF and strict fixed validation, before char20.",
        "baseline": {"full": full_baseline, "fixed": fixed_baseline},
        "eligible": eligibility,
    }
    write_json(out / "alpha_grid_report.json", result)
    print(json.dumps(result["promotion_gate"], ensure_ascii=False, indent=2), flush=True)


def main() -> None:
    args = parse_args()
    out = ROOT / args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    if args.stage in {"full-oof", "both"}:
        run_full_oof(args, out)
    if args.stage in {"fixed", "both"}:
        run_fixed(args, out)
    if args.stage in {"alpha", "both"}:
        run_alpha_grid(out)
    promotion_gate(out)


if __name__ == "__main__":
    main()
