#!/usr/bin/env python3
"""Strict B43 audit for an AU-only raw-selector route.

The public-best B43 model keeps the old no-E5 GRU, source geometric blend,
and char20 overlay.  This experiment changes just one internal component:
for ``sess_au_*`` rows it mixes an AU-only XGBoost selector with the existing
global raw selector before the unchanged GRU geometry and char20 override.

All selection and fitting is split by whole session.  The script never reads
or writes a live model, inference script, or submission ZIP.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold

from experiment_xgb_robust_ensemble import (
    N_CLASSES, ROOT, apply_char20, fixed_char20, full_char20, full_context,
    geometric, selector_probability, strict_context, terminal_mask,
)


OUT = ROOT / "experiments/au_specialist_b43_v2"
ALL_MASK = None
GRID_WEIGHT_GLOBAL = (0.00, 0.25, 0.50, 0.75)
# These are deliberately the two previously observed AU-specialist families;
# no broad hyperparameter search is performed after looking at B43 feedback.
CONFIGS = {
    "regularized": {
        "n_estimators": 360, "max_depth": 7, "learning_rate": 0.035,
        "subsample": 0.90, "colsample_bytree": 0.80,
        "min_child_weight": 3.0, "reg_lambda": 5.0, "gamma": 0.05,
        "weight_power": 0.75, "random_state": 142,
    },
    "compact": {
        "n_estimators": 520, "max_depth": 6, "learning_rate": 0.03,
        "subsample": 0.90, "colsample_bytree": 0.70,
        "min_child_weight": 4.0, "reg_lambda": 8.0, "gamma": 0.10,
        "weight_power": 0.60, "random_state": 242,
    },
}

# The first complete outer-selection pass was intentionally stopped only
# after all five held-fold choices had been emitted, when the independent
# fixed block hit a shape bug.  These choices are replayed below without
# touching their held-fold labels, avoiding another 50-fit selection pass.
# Set this to ``None`` to recompute the full nested selection from scratch.
RESUME_OUTER_SELECTIONS: tuple[str, ...] | None = (
    "regularized|global_weight=0.00",
    "compact|global_weight=0.50",
    "compact|global_weight=0.50",
    "compact|global_weight=0.25",
    "regularized|global_weight=0.25",
)


def score(y: np.ndarray, probability: np.ndarray, mask: np.ndarray) -> float:
    return float(f1_score(
        y[mask], probability[mask].argmax(axis=1), labels=np.arange(N_CLASSES),
        average="macro", zero_division=0,
    ))


def make_folds(x: np.ndarray, y: np.ndarray, groups: np.ndarray) -> np.ndarray:
    fold = np.empty(len(y), dtype=np.int8)
    for number, (_, valid) in enumerate(GroupKFold(n_splits=5).split(x, y, groups)):
        fold[valid] = number
    return fold


def au_probability(
    config: dict[str, object], x: np.ndarray, y: np.ndarray, source: np.ndarray,
    train: np.ndarray, predict: np.ndarray,
) -> np.ndarray:
    """Fit only AU training sessions and return a 14-column probability block."""
    au_train = train & (source == "au")
    au_predict = predict & (source == "au")
    result = np.zeros((int(au_predict.sum()), N_CLASSES), dtype=np.float32)
    if not au_predict.any():
        return result
    result[:] = selector_probability(
        config, x[au_train], y[au_train], x[au_predict], device="cuda", n_jobs=4,
    )
    return result


def replace_au(global_probability: np.ndarray, specialist: np.ndarray, source: np.ndarray, global_weight: float) -> np.ndarray:
    output = global_probability.copy()
    mask = source == "au"
    output[mask] = float(global_weight) * global_probability[mask] + (1.0 - float(global_weight)) * specialist[mask]
    output /= output.sum(axis=1, keepdims=True)
    return output.astype(np.float32, copy=False)


def b43_probability(raw: np.ndarray, old_gru: np.ndarray, source: np.ndarray) -> np.ndarray:
    """Exact B43 geometry: SIM raw=.20, AU raw=.85, old no-E5 GRU."""
    alpha = np.where(source == "au", 0.85, 0.20).astype(np.float32)[:, None]
    value = alpha * np.log(np.maximum(raw, 1.0e-12)) + (1.0 - alpha) * np.log(np.maximum(old_gru, 1.0e-12))
    value -= value.max(axis=1, keepdims=True)
    value = np.exp(value)
    return (value / value.sum(axis=1, keepdims=True)).astype(np.float32)


def with_char(probability: np.ndarray, char: np.ndarray) -> np.ndarray:
    return apply_char20(probability, char)


def metric(y: np.ndarray, probability: np.ndarray, terminal: np.ndarray, char: np.ndarray) -> dict[str, float]:
    overridden = with_char(probability, char)
    return {
        "all_with_char20": score(y, overridden, np.ones(len(y), dtype=bool)),
        "terminal_with_char20": score(y, overridden, terminal),
        "all_before_char20": score(y, probability, np.ones(len(y), dtype=bool)),
        "terminal_before_char20": score(y, probability, terminal),
    }


def config_name(config_name: str, weight: float) -> str:
    return f"{config_name}|global_weight={weight:.2f}"


def full_nested(
    x: np.ndarray, y: np.ndarray, source: np.ndarray, groups: np.ndarray, fold: np.ndarray,
    raw: np.ndarray, old_gru: np.ndarray, terminal: np.ndarray, char: np.ndarray,
) -> tuple[np.ndarray, list[dict[str, object]], dict[str, np.ndarray]]:
    """Outer-session nested selection of the specialist config and mix weight."""
    candidate_outer: dict[str, np.ndarray] = {
        config_name(name, weight): np.zeros((len(y), N_CLASSES), dtype=np.float32)
        for name in CONFIGS for weight in GRID_WEIGHT_GLOBAL
    }
    selected = np.zeros((len(y), N_CLASSES), dtype=np.float32)
    records: list[dict[str, object]] = []
    for held in range(5):
        held_mask = fold == held
        outer_train = ~held_mask
        # Inner probability blocks are trained without either the outer held
        # session fold or the current inner held fold.
        inner_probs: dict[str, np.ndarray] = {
            config_name(name, weight): np.zeros((len(y), N_CLASSES), dtype=np.float32)
            for name in CONFIGS for weight in GRID_WEIGHT_GLOBAL
        }
        for inner in range(5):
            if inner == held:
                continue
            inner_mask = fold == inner
            train = outer_train & ~inner_mask
            for name, config in CONFIGS.items():
                block = np.zeros((len(y), N_CLASSES), dtype=np.float32)
                block[inner_mask & (source == "au")] = au_probability(config, x, y, source, train, inner_mask)
                # SIM uses global raw and AU rows not in this held inner fold
                # are irrelevant to the selection metric.
                for weight in GRID_WEIGHT_GLOBAL:
                    mixed_raw = raw.copy()
                    au_rows = inner_mask & (source == "au")
                    mixed_raw[au_rows] = weight * raw[au_rows] + (1.0 - weight) * block[au_rows]
                    inner_probs[config_name(name, weight)][inner_mask] = b43_probability(
                        mixed_raw[inner_mask], old_gru[inner_mask], source[inner_mask]
                    )
        choices = []
        inner_score_mask = outer_train
        for key, value in inner_probs.items():
            candidate = with_char(value, char)
            choices.append((score(y, candidate, inner_score_mask & terminal), score(y, candidate, inner_score_mask), key))
        # Lower specialist capacity is favored only when scores are exact ties.
        _, _, picked = max(choices, key=lambda row: (row[0], row[1], row[2].startswith("regularized")))
        name, raw_weight = picked.split("|global_weight=")
        raw_weight = float(raw_weight)
        specialist_block = np.zeros((len(y), N_CLASSES), dtype=np.float32)
        specialist_block[held_mask & (source == "au")] = au_probability(
            CONFIGS[name], x, y, source, outer_train, held_mask
        )
        for config_label, config in CONFIGS.items():
            # Store all held probabilities for an auditable static comparison;
            # only the nested-picked value is eligible for promotion.
            all_block = specialist_block if config_label == name else None
            if all_block is None:
                alternative = np.zeros((len(y), N_CLASSES), dtype=np.float32)
                alternative[held_mask & (source == "au")] = au_probability(config, x, y, source, outer_train, held_mask)
                all_block = alternative
            for weight in GRID_WEIGHT_GLOBAL:
                mixed_raw = raw[held_mask].copy()
                au_rows = source[held_mask] == "au"
                mixed_raw[au_rows] = weight * raw[held_mask][au_rows] + (1.0 - weight) * all_block[held_mask][au_rows]
                candidate_outer[config_name(config_label, weight)][held_mask] = b43_probability(
                    mixed_raw, old_gru[held_mask], source[held_mask]
                )
        selected[held_mask] = candidate_outer[picked][held_mask]
        records.append({"held_fold": held + 1, "selected": picked})
        print(json.dumps(records[-1]), flush=True)
    return selected, records, candidate_outer


def replay_outer_selection(
    x: np.ndarray, y: np.ndarray, source: np.ndarray, fold: np.ndarray,
    raw: np.ndarray, old_gru: np.ndarray,
) -> tuple[np.ndarray, list[dict[str, object]], dict[str, np.ndarray]]:
    """Refit held-fold predictions for logged, inner-only outer choices.

    This recovery mode is deliberately narrower than ``full_nested``: it
    cannot select a new policy, and it uses the five selections printed by a
    completed inner-only pass before an unrelated fixed-split shape error.
    Each held prediction is still fitted without that held session fold.
    """
    if RESUME_OUTER_SELECTIONS is None or len(RESUME_OUTER_SELECTIONS) != 5:
        raise ValueError("Replay selections must contain one policy per outer fold")
    selected = np.zeros((len(y), N_CLASSES), dtype=np.float32)
    records: list[dict[str, object]] = []
    for held, picked in enumerate(RESUME_OUTER_SELECTIONS):
        held_mask = fold == held
        outer_train = ~held_mask
        name, raw_weight = picked.split("|global_weight=")
        raw_weight = float(raw_weight)
        specialist = np.zeros((len(y), N_CLASSES), dtype=np.float32)
        specialist[held_mask & (source == "au")] = au_probability(
            CONFIGS[name], x, y, source, outer_train, held_mask
        )
        mixed_raw = raw[held_mask].copy()
        au_rows = source[held_mask] == "au"
        mixed_raw[au_rows] = (
            raw_weight * raw[held_mask][au_rows]
            + (1.0 - raw_weight) * specialist[held_mask][au_rows]
        )
        selected[held_mask] = b43_probability(mixed_raw, old_gru[held_mask], source[held_mask])
        records.append({
            "held_fold": held + 1,
            "selected": picked,
            "selection_origin": "replayed from prior completed inner-only selection pass",
        })
        print(json.dumps(records[-1]), flush=True)
    return selected, records, {}


def common_specialist_oof(
    x: np.ndarray, y: np.ndarray, source: np.ndarray, groups: np.ndarray,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    fold = make_folds(x, y, groups)
    output = {name: np.zeros((len(y), N_CLASSES), dtype=np.float32) for name in CONFIGS}
    for held in range(5):
        train = fold != held
        valid = fold == held
        for name, config in CONFIGS.items():
            output[name][valid & (source == "au")] = au_probability(config, x, y, source, train, valid)
    return output, fold


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    x, y, source, groups, ids = full_context()
    terminal = terminal_mask(ids)
    raw = np.load(ROOT / "experiments/xgb_robust_ensemble/full_oof_raw_seed42.npy", allow_pickle=False).astype(np.float32)
    old = np.load(ROOT / "experiments/modeling_ai_sequence_fusion_noe5/oof_probabilities.npy", allow_pickle=False).astype(np.float32)
    char = full_char20()
    if not (raw.shape == old.shape == (len(y), N_CLASSES)):
        raise ValueError("Full B43 component probabilities do not align")
    fold = make_folds(x, y, groups)
    baseline = b43_probability(raw, old, source)
    if RESUME_OUTER_SELECTIONS is None:
        nested, records, static = full_nested(x, y, source, groups, fold, raw, old, terminal, char)
    else:
        nested, records, static = replay_outer_selection(x, y, source, fold, raw, old)

    x_train, y_train, source_train, x_valid, y_valid, source_valid = strict_context()
    saved = np.load(ROOT / "experiments/au_source_strict_common/strict_blocks.npz", allow_pickle=False)
    ids_train = saved["ids_train"].astype(str)
    ids_valid = saved["ids_valid"].astype(str)
    groups_train = np.asarray([value.rsplit("-step_", 1)[0] for value in ids_train])
    terminal_train = terminal_mask(ids_train)
    terminal_valid = terminal_mask(ids_valid)
    raw_train = np.load(ROOT / "experiments/b42_9_common_raw_seed42/common_train_raw_xgb_oof.npy", allow_pickle=False).astype(np.float32)
    old_train = np.load(ROOT / "experiments/b42_9_common_old_noe5_seed42/common_train_oof_probabilities.npy", allow_pickle=False).astype(np.float32)
    raw_valid = np.load(ROOT / "experiments/xgb_robust_ensemble/strict_fixed_raw_seed42.npy", allow_pickle=False).astype(np.float32)
    old_valid = np.load(ROOT / "experiments/b42_9_common_old_noe5_seed42/fixed_valid_probabilities.npy", allow_pickle=False).astype(np.float32)
    if not (raw_train.shape == old_train.shape == (len(y_train), N_CLASSES)
            and raw_valid.shape == old_valid.shape == (len(y_valid), N_CLASSES)):
        raise ValueError("Common B43 component probabilities do not align")
    train_oof, _ = common_specialist_oof(x_train, y_train, source_train, groups_train)
    common_options = []
    for name, specialist in train_oof.items():
        for weight in GRID_WEIGHT_GLOBAL:
            mixed = replace_au(raw_train, specialist, source_train, weight)
            probability = b43_probability(mixed, old_train, source_train)
            common_options.append({
                "key": config_name(name, weight),
                "terminal_before_char20": score(y_train, probability, terminal_train),
                "all_before_char20": score(y_train, probability, np.ones(len(y_train), dtype=bool)),
            })
    chosen_common = max(common_options, key=lambda row: (row["terminal_before_char20"], row["all_before_char20"], row["key"].startswith("regularized")))
    chosen_name, chosen_weight = chosen_common["key"].split("|global_weight=")
    chosen_weight = float(chosen_weight)
    fixed_specialist = np.zeros((len(y_valid), N_CLASSES), dtype=np.float32)
    # ``au_probability`` expects both masks to refer to the same feature
    # matrix.  The fixed validation matrix is distinct, so fit its AU block
    # explicitly and write it back into the full validation-shaped array.
    if np.any(source_valid == "au"):
        fixed_specialist[source_valid == "au"] = selector_probability(
            CONFIGS[chosen_name], x_train[source_train == "au"], y_train[source_train == "au"],
            x_valid[source_valid == "au"], device="cuda", n_jobs=4,
        )
    fixed_raw_mixed = raw_valid.copy()
    if np.any(source_valid == "au"):
        fixed_raw_mixed[source_valid == "au"] = (
            chosen_weight * raw_valid[source_valid == "au"]
            + (1.0 - chosen_weight) * fixed_specialist[source_valid == "au"]
        )
    fixed_probability = b43_probability(fixed_raw_mixed, old_valid, source_valid)
    fixed_char = fixed_char20(ids_valid, terminal_valid)
    baseline_fixed = b43_probability(raw_valid, old_valid, source_valid)

    report = {
        "scope": "Experiment only; live model/script/B42.zip/baseline_submit.zip are unchanged.",
        "baseline": "B43: exact old no-E5 GRU + raw seed42 selector geometric SIM=.20/AU=.85 + immutable char20 overlay.",
        "configs": CONFIGS,
        "full_outer_nested": {
            "baseline": metric(y, baseline, terminal, char),
            "candidate": metric(y, nested, terminal, char),
            "fold_choices": records,
            "static_candidates": {key: metric(y, value, terminal, char) for key, value in static.items()},
        },
        "matching_seed42_fixed": {
            "baseline": metric(y_valid, baseline_fixed, terminal_valid, fixed_char),
            "candidate": metric(y_valid, fixed_probability, terminal_valid, fixed_char),
            "selected_from_common_train": chosen_common,
            "common_train_options": common_options,
        },
    }
    full_base = report["full_outer_nested"]["baseline"]
    full_candidate = report["full_outer_nested"]["candidate"]
    fixed_base = report["matching_seed42_fixed"]["baseline"]
    fixed_candidate = report["matching_seed42_fixed"]["candidate"]
    passed = all((
        full_candidate["all_with_char20"] > full_base["all_with_char20"],
        full_candidate["terminal_with_char20"] > full_base["terminal_with_char20"],
        fixed_candidate["all_with_char20"] > fixed_base["all_with_char20"],
        fixed_candidate["terminal_with_char20"] > fixed_base["terminal_with_char20"],
    ))
    report["promotion_gate"] = {
        "requires": "B43 all+terminal char20 macro-F1 improvement on both outer nested and matching fixed validation",
        "pass": passed,
        "decision": "PROMOTE_CANDIDATE" if passed else "REJECT",
    }
    (OUT / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["promotion_gate"], ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
