#!/usr/bin/env python3
"""Strict SIM-only raw-XGB selector audit.

This experiment is intentionally isolated from the submission bundle.  It
tests whether a selector trained only on ``sess_sim_*`` rows can replace or
blend with the globally trained 2032-feature raw selector.  Every SIM
specialist prediction is made for a held-out whole session, and the second
protocol trains only on common-train SIM sessions before scoring common-fixed
SIM sessions.  The matching deployed seed-42 global selector is the sole
baseline; legacy fixed selectors are never used.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from script import ALL_CLASSES_LIST, build_state_dict, build_user_sequence_hash
from data import load_jsonl, load_labels
from experiment_strict_blend_calibration import EPS, load_data


N_CLASSES = len(ALL_CLASSES_LIST)
LABEL_TO_ID = {value: index for index, value in enumerate(ALL_CLASSES_LIST)}

# Kept deliberately small: the exact deployed tree configuration and the
# previously audited depth-9 regularized configuration.  These are fixed
# before seeing SIM specialist results.
VARIANTS: dict[str, dict[str, object]] = {
    "same_raw_seed42": {
        "n_estimators": 210, "max_depth": 10, "learning_rate": 0.04,
        "subsample": 0.85, "colsample_bytree": 0.85,
        "min_child_weight": 1.0, "reg_lambda": 1.0, "gamma": 0.0,
        "weight_power": 1.0, "random_state": 42,
    },
    "regularized_d9": {
        "n_estimators": 270, "max_depth": 9, "learning_rate": 0.035,
        "subsample": 0.90, "colsample_bytree": 0.80,
        "min_child_weight": 2.0, "reg_lambda": 3.0, "gamma": 0.05,
        "weight_power": 1.0, "random_state": 142,
    },
}

# The existing public packages have used these two SIM raw-XGB exponents.
# They are reported separately and never chosen on the fixed validation set.
SOURCE_ALPHA = ((0.20, 0.85), (0.25, 0.85))
ROUTE_WEIGHTS = (0.25, 0.50, 0.75)  # weight on the global raw selector


@dataclass(frozen=True)
class Context:
    samples: list[dict]
    ids: np.ndarray
    target: np.ndarray
    source: np.ndarray
    groups: np.ndarray
    fold: np.ndarray
    terminal: np.ndarray
    sim_index: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="experiments/sim_source_selector_v1")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--n-jobs", type=int, default=4)
    parser.add_argument("--variants", default="same_raw_seed42,regularized_d9")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def macro(y: np.ndarray, probability: np.ndarray, mask: np.ndarray | None = None) -> float:
    if mask is None:
        mask = np.ones(len(y), dtype=bool)
    return float(
        f1_score(
            y[mask], probability[mask].argmax(axis=1), labels=np.arange(N_CLASSES),
            average="macro", zero_division=0,
        )
    )


def prediction_macro(y: np.ndarray, prediction: np.ndarray, mask: np.ndarray | None = None) -> float:
    if mask is None:
        mask = np.ones(len(y), dtype=bool)
    return float(
        f1_score(
            y[mask], prediction[mask], labels=np.arange(N_CLASSES), average="macro", zero_division=0,
        )
    )


def normalized_geometric(first: np.ndarray, second: np.ndarray, first_weight: float) -> np.ndarray:
    logits = first_weight * np.log(np.maximum(first, EPS)) + (1.0 - first_weight) * np.log(np.maximum(second, EPS))
    logits -= logits.max(axis=1, keepdims=True)
    value = np.exp(logits)
    return (value / value.sum(axis=1, keepdims=True)).astype(np.float32)


def raw_gru_geometry(xgb: np.ndarray, gru: np.ndarray, source: np.ndarray, sim_alpha: float, au_alpha: float) -> np.ndarray:
    alpha = np.where(source == "sim", sim_alpha, au_alpha).astype(np.float64)[:, None]
    logits = alpha * np.log(np.maximum(xgb, EPS)) + (1.0 - alpha) * np.log(np.maximum(gru, EPS))
    logits -= logits.max(axis=1, keepdims=True)
    value = np.exp(logits)
    return (value / value.sum(axis=1, keepdims=True)).astype(np.float32)


def char_override(prediction: np.ndarray, terminal: np.ndarray, char: np.ndarray) -> np.ndarray:
    result = prediction.copy()
    mask = terminal & (char >= 0)
    result[mask] = char[mask]
    return result


def metric_block(y: np.ndarray, probability: np.ndarray, terminal: np.ndarray, char: np.ndarray, source: np.ndarray) -> dict[str, float]:
    fallback = probability.argmax(axis=1)
    after_char = char_override(fallback, terminal, char)
    hybrid = after_char.copy()
    hybrid[~terminal] = y[~terminal]
    sim = source == "sim"
    return {
        "all_fallback_macro_f1": prediction_macro(y, fallback),
        "terminal_fallback_macro_f1": prediction_macro(y, fallback, terminal),
        "all_with_char20_macro_f1": prediction_macro(y, after_char),
        "terminal_with_char20_macro_f1": prediction_macro(y, after_char, terminal),
        "hybrid_future_history_plus_char20_macro_f1": prediction_macro(y, hybrid),
        "sim_all_fallback_macro_f1": prediction_macro(y, fallback, sim),
        "sim_terminal_fallback_macro_f1": prediction_macro(y, fallback, sim & terminal),
        "sim_terminal_with_char20_macro_f1": prediction_macro(y, after_char, sim & terminal),
    }


def build_context() -> Context:
    samples = load_jsonl(str(ROOT / "data/train.jsonl"))
    labels = load_labels(str(ROOT / "data/train_labels.csv"))
    ids = np.asarray([str(sample["id"]) for sample in samples])
    target = np.asarray([LABEL_TO_ID[labels[value]] for value in ids], dtype=np.int64)
    source = np.asarray(["au" if value.startswith("sess_au_") else "sim" for value in ids])
    groups = np.asarray([value.rsplit("-step_", 1)[0] for value in ids])
    fold = np.empty(len(ids), dtype=np.int8)
    for number, (_, valid) in enumerate(GroupKFold(n_splits=5).split(np.zeros(len(target)), target, groups)):
        fold[valid] = number
    data = load_data(ROOT)
    if not (
        np.array_equal(target, data["y"])
        and np.array_equal(source, data["source"])
        and np.array_equal(fold, data["fold"])
    ):
        raise ValueError("SIM source context is not aligned with strict nested artifacts")
    return Context(
        samples=samples, ids=ids, target=target, source=source, groups=groups,
        fold=fold, terminal=data["terminal"], sim_index=np.flatnonzero(source == "sim"),
    )


def full_sim_features(context: Context) -> np.ndarray:
    """Materialize the exact label-free 2032-vector only for SIM rows."""
    index = context.sim_index
    base = np.load(ROOT / "experiments/modeling_n_refit/oof_stack_features.npy", mmap_mode="r")
    source_e = np.load(ROOT / "experiments/modeling_ak_live_xgb_proxy/source_oof.npy", mmap_mode="r")
    bundle = joblib.load(ROOT / "model_candidate_q/best_model.pkl")
    state_all = bundle["state_vectorizer"].transform([build_state_dict(sample) for sample in context.samples]).astype(np.float32)
    user_all = build_user_sequence_hash(context.samples).astype(np.float32)
    result = np.column_stack((base[index], state_all[index], source_e[index], user_all[index])).astype(np.float32)
    del state_all, user_all, bundle
    gc.collect()
    if result.shape != (len(index), 2032):
        raise ValueError(f"SIM feature schema is {result.shape}; expected ({len(index)}, 2032)")
    return result


def fixed_sim_features() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    saved = np.load(ROOT / "experiments/au_source_strict_common/strict_blocks.npz", allow_pickle=False)
    source_train = saved["source_train"].astype(str)
    source_valid = saved["source_valid"].astype(str)
    train_sim = source_train == "sim"
    valid_sim = source_valid == "sim"
    x_train = np.column_stack((
        saved["base_train"][train_sim], saved["state_train"][train_sim],
        saved["source_e_train"][train_sim], saved["user_train"][train_sim],
    )).astype(np.float32)
    x_valid = np.column_stack((
        saved["base_valid"][valid_sim], saved["state_valid"][valid_sim],
        saved["source_e_valid"][valid_sim], saved["user_valid"][valid_sim],
    )).astype(np.float32)
    if x_train.shape[1] != 2032 or x_valid.shape[1] != 2032:
        raise ValueError("Strict fixed SIM selector schema is not 2032 columns")
    return (
        x_train, saved["target_train"][train_sim].astype(np.int64),
        x_valid, saved["target_valid"][valid_sim].astype(np.int64),
        saved["ids_valid"].astype(str), valid_sim,
    )


def fit_xgb(config: dict[str, object], x_train: np.ndarray, y_train: np.ndarray, x_valid: np.ndarray, device: str, n_jobs: int) -> np.ndarray:
    count = np.bincount(y_train, minlength=N_CLASSES)
    if np.any(count == 0):
        missing = [ALL_CLASSES_LIST[index] for index in np.flatnonzero(count == 0)]
        raise ValueError(f"SIM-only training lacks labels: {missing}")
    parameter = dict(config)
    power = float(parameter.pop("weight_power"))
    sample_weight = (len(y_train) / (N_CLASSES * count[y_train])) ** power
    model = XGBClassifier(
        **parameter, objective="multi:softprob", num_class=N_CLASSES,
        tree_method="hist", device=device, eval_metric="mlogloss", n_jobs=n_jobs,
    )
    try:
        model.fit(x_train, y_train, sample_weight=sample_weight, verbose=False)
        return model.predict_proba(x_valid).astype(np.float32)
    finally:
        del model
        gc.collect()


def fit_full_oof(
    out: Path, variant: str, context: Context, feature: np.ndarray, device: str, n_jobs: int, force: bool,
) -> np.ndarray:
    path = out / f"full_oof_{variant}.npy"
    if path.exists() and not force:
        value = np.load(path, allow_pickle=False).astype(np.float32)
        if value.shape == (len(context.sim_index), N_CLASSES):
            print(f"full {variant}: reused cache", flush=True)
            return value
    target = context.target[context.sim_index]
    fold = context.fold[context.sim_index]
    output = np.zeros((len(target), N_CLASSES), dtype=np.float32)
    for held in range(5):
        train = fold != held
        valid = ~train
        output[valid] = fit_xgb(VARIANTS[variant], feature[train], target[train], feature[valid], device, n_jobs)
        print(f"full {variant}: fold={held + 1}/5 macro={macro(target, output, valid):.6f}", flush=True)
    np.save(path, output)
    return output


def fit_fixed(
    out: Path, variant: str, device: str, n_jobs: int, force: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    path = out / f"fixed_{variant}.npy"
    x_train, y_train, x_valid, y_valid, ids_valid, valid_sim = fixed_sim_features()
    if path.exists() and not force:
        value = np.load(path, allow_pickle=False).astype(np.float32)
        if value.shape == (len(y_valid), N_CLASSES):
            print(f"fixed {variant}: reused cache", flush=True)
            return value, ids_valid, valid_sim
    value = fit_xgb(VARIANTS[variant], x_train, y_train, x_valid, device, n_jobs)
    np.save(path, value)
    print(f"fixed {variant}: sim_macro={macro(y_valid, value):.6f}", flush=True)
    return value, ids_valid, valid_sim


def route_probability(global_probability: np.ndarray, sim_index: np.ndarray, specialist: np.ndarray, mode: str, global_weight: float) -> np.ndarray:
    value = global_probability.copy()
    if mode == "direct":
        value[sim_index] = specialist
    elif mode == "linear":
        value[sim_index] = global_weight * global_probability[sim_index] + (1.0 - global_weight) * specialist
    elif mode == "geometric":
        value[sim_index] = normalized_geometric(global_probability[sim_index], specialist, global_weight)
    else:
        raise ValueError(mode)
    return value


def route_specs() -> list[tuple[str, float]]:
    result: list[tuple[str, float]] = [("direct", 0.0)]
    for mode in ("linear", "geometric"):
        result.extend((mode, weight) for weight in ROUTE_WEIGHTS)
    return result


def nested_route_support(
    y: np.ndarray, terminal: np.ndarray, fold: np.ndarray, source: np.ndarray,
    candidates: dict[str, np.ndarray],
) -> dict[str, Any]:
    """Second-level selection of a fixed route using only other session folds."""
    result = np.zeros((len(y), N_CLASSES), dtype=np.float32)
    choices: list[dict[str, Any]] = []
    for held in range(5):
        train = fold != held
        valid = ~train
        chosen_key, chosen = max(
            candidates.items(),
            key=lambda item: (
                macro(y, item[1], train & terminal & (source == "sim")),
                macro(y, item[1], train),
                item[0],
            ),
        )
        result[valid] = chosen[valid]
        choices.append({"held_fold": held + 1, "selected": chosen_key})
    count: dict[str, int] = {}
    for row in choices:
        count[row["selected"]] = count.get(row["selected"], 0) + 1
    return {
        "fold_choices": choices,
        "selection_count": count,
        "all_fallback_macro_f1": macro(y, result),
        "terminal_fallback_macro_f1": macro(y, result, terminal),
        "sim_terminal_fallback_macro_f1": macro(y, result, terminal & (source == "sim")),
    }


def delta(candidate: dict[str, float], baseline: dict[str, float]) -> dict[str, float]:
    return {key: float(candidate[key] - baseline[key]) for key in (
        "all_fallback_macro_f1", "terminal_fallback_macro_f1",
        "all_with_char20_macro_f1", "terminal_with_char20_macro_f1",
        "hybrid_future_history_plus_char20_macro_f1",
    )}


def main() -> None:
    args = parse_args()
    variants = [item.strip() for item in args.variants.split(",") if item.strip()]
    if not variants or any(item not in VARIANTS for item in variants):
        raise ValueError(f"--variants must be non-empty subset of {sorted(VARIANTS)}")
    out = ROOT / args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    context = build_context()
    data = load_data(ROOT)
    full_global = np.load(ROOT / "experiments/xgb_robust_ensemble/full_oof_raw_seed42.npy", allow_pickle=False).astype(np.float32)
    fixed_global = np.load(ROOT / "experiments/xgb_robust_ensemble/strict_fixed_raw_seed42.npy", allow_pickle=False).astype(np.float32)
    if full_global.shape != (len(context.ids), N_CLASSES) or fixed_global.shape != (len(data["valid_index"]), N_CLASSES):
        raise ValueError("Matching seed-42 global raw-selector artifacts have unexpected shapes")
    if not np.array_equal(context.target, data["y"]):
        raise ValueError("Target alignment failure")
    fixed_y = data["y"][data["valid_index"]]
    fixed_source = data["source"][data["valid_index"]]
    fixed_terminal = data["fixed_terminal"]
    fixed_char = data["fixed_char"]

    feature = full_sim_features(context)
    full_specialists: dict[str, np.ndarray] = {}
    fixed_specialists: dict[str, np.ndarray] = {}
    for variant in variants:
        full_specialists[variant] = fit_full_oof(out, variant, context, feature, args.device, args.n_jobs, args.force)
        fixed_value, fixed_ids, fixed_sim = fit_fixed(out, variant, args.device, args.n_jobs, args.force)
        if not np.array_equal(fixed_ids, context.ids[data["valid_index"]]) or not np.array_equal(fixed_sim, fixed_source == "sim"):
            raise ValueError("Fixed SIM specialist IDs/source route do not align")
        fixed_specialists[variant] = fixed_value
    del feature
    gc.collect()

    report: dict[str, Any] = {
        "scope": "Experiment only; model/, script.py, and submission ZIP files were not modified.",
        "protocol": {
            "full": "Five whole-session GroupKFold selector fits; each SIM specialist train partition excludes its held session. Existing base/source OOF blocks are retained exactly.",
            "fixed": "SIM specialist uses common-train SIM rows only, with matching 2032-feature seed-42 global selector on common validation.",
            "features": "Existing 97 base OOF + 360 label-free state + 39 source-E OOF + 1536 label-free user hash = 2032; source routing uses only sess_sim_/sess_au_ ID prefix.",
            "downstream": "Raw-selector routes are fed to existing strict-distilled portable GRU geometry; char20 remains a final terminal-only override and is never used for selection.",
        },
        "counts": {
            "full_rows": int(len(context.ids)), "full_sim_rows": int(len(context.sim_index)),
            "full_sim_terminal": int(np.sum(context.terminal & (context.source == "sim"))),
            "fixed_rows": int(len(fixed_y)), "fixed_sim_rows": int(np.sum(fixed_source == "sim")),
            "fixed_sim_terminal": int(np.sum(fixed_terminal & (fixed_source == "sim"))),
        },
        "variants": {name: VARIANTS[name] for name in variants},
        "results": {},
        "recommendation": None,
    }

    all_promotable: list[dict[str, Any]] = []
    for sim_alpha, au_alpha in SOURCE_ALPHA:
        alpha_key = f"sim_{sim_alpha:.2f}_au_{au_alpha:.2f}"
        baseline_full = raw_gru_geometry(full_global, data["gru"], context.source, sim_alpha, au_alpha)
        baseline_fixed = raw_gru_geometry(fixed_global, data["fixed_gru"], fixed_source, sim_alpha, au_alpha)
        alpha_result: dict[str, Any] = {
            "baseline_global": {
                "full": metric_block(context.target, baseline_full, context.terminal, data["char"], context.source),
                "fixed": metric_block(fixed_y, baseline_fixed, fixed_terminal, fixed_char, fixed_source),
            },
            "variants": {},
        }
        for variant in variants:
            # Specialist values are stored only in SIM-row order.  Expand them
            # into each protocol's global order through the permitted prefix route.
            full_routes: dict[str, np.ndarray] = {"global": full_global}
            fixed_routes: dict[str, np.ndarray] = {"global": fixed_global}
            fixed_sim_index = np.flatnonzero(fixed_source == "sim")
            for mode, weight in route_specs():
                name = f"{mode}_global_weight_{weight:.2f}"
                full_routes[name] = route_probability(full_global, context.sim_index, full_specialists[variant], mode, weight)
                fixed_routes[name] = route_probability(fixed_global, fixed_sim_index, fixed_specialists[variant], mode, weight)
            nested = nested_route_support(context.target, context.terminal, context.fold, context.source, {
                name: raw_gru_geometry(value, data["gru"], context.source, sim_alpha, au_alpha)
                for name, value in full_routes.items()
            })
            route_report: dict[str, Any] = {}
            baseline_metrics = alpha_result["baseline_global"]
            for name in full_routes:
                full_probability = raw_gru_geometry(full_routes[name], data["gru"], context.source, sim_alpha, au_alpha)
                fixed_probability = raw_gru_geometry(fixed_routes[name], data["fixed_gru"], fixed_source, sim_alpha, au_alpha)
                full_metrics = metric_block(context.target, full_probability, context.terminal, data["char"], context.source)
                fixed_metrics = metric_block(fixed_y, fixed_probability, fixed_terminal, fixed_char, fixed_source)
                route_report[name] = {
                    "full": full_metrics, "fixed": fixed_metrics,
                    "delta_vs_global": {"full": delta(full_metrics, baseline_metrics["full"]), "fixed": delta(fixed_metrics, baseline_metrics["fixed"])},
                }
                stable = nested["selection_count"].get(name, 0) >= 3
                passed = name != "global" and stable and all(
                    route_report[name]["delta_vs_global"][scope][metric] > 0.0
                    for scope in ("full", "fixed")
                    for metric in ("all_fallback_macro_f1", "terminal_fallback_macro_f1", "terminal_with_char20_macro_f1")
                )
                if passed:
                    all_promotable.append({"alpha": alpha_key, "variant": variant, "route": name})
            alpha_result["variants"][variant] = {
                "nested_route_support": nested,
                "routes": route_report,
            }
        report["results"][alpha_key] = alpha_result

    report["promotion_gate"] = {
        "rule": "A route must be selected in >=3/5 nested whole-session folds and strictly improve all-fallback, terminal-fallback, and terminal-char20 macro-F1 over the matching seed-42 global selector+strict-GRU baseline on both protocols.",
        "passing": all_promotable,
    }
    if all_promotable:
        report["recommendation"] = {
            "decision": "One or more SIM raw-selector routes passed both protocols; only a separate portable full-refit candidate may be built after review.",
            "portable_build": "Fit selected SIM-only XGB on all SIM training rows using the same 2032 inference-safe vector; store it in a candidate bundle and route only sess_sim_* rows before the existing raw-XGB/GRU geometric gate. Preserve final char20 and future-history order.",
        }
    else:
        report["recommendation"] = {
            "decision": "No SIM raw-selector route cleared the two-protocol stability gate; do not integrate a SIM specialist.",
        }
    (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary = {
        "counts": report["counts"], "passing": all_promotable,
        "recommendation": report["recommendation"],
    }
    (out / "README.md").write_text(
        "# SIM raw selector specialist audit\n\n"
        "No live submission assets were modified.  See `report.json` for both validation protocols.\n\n```json\n"
        + json.dumps(summary, ensure_ascii=False, indent=2) + "\n```\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
