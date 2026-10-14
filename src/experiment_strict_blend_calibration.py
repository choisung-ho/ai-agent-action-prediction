#!/usr/bin/env python3
"""Leakage-controlled calibration audit for raw-XGB + strict no-E5 GRU.

This is deliberately an experiment-only utility.  It never opens ``model/``,
``script.py`` or a submission ZIP.  The important distinction from a simple
OOF grid search is that, for every held-out *session* fold, the alpha or
stacker is fitted and its hyperparameters are selected without labels from
that held-out fold.  A second protocol repeats selection inside the common
training partition and evaluates only on the untouched common validation
partition.

The script compares the deployed two-scalar geometric gate to small portable
calibrators only:

* shared class-specific geometric alpha (14 deviations around source alpha),
* shared class alpha plus 14 conservative class-prior biases, and
* an L2 multinomial logistic stacker on source-interacted log probabilities.

Any eventual artifact contains only scalar / NumPy-compatible parameters.  It
is written *only* when a candidate improves the terminal fallback and the
future-history+char20 hybrid on both validation protocols.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logsumexp
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold


CLASSES = [
    "read_file", "grep_search", "list_directory", "glob_pattern", "edit_file", "write_file", "apply_patch",
    "run_bash", "run_tests", "lint_or_typecheck", "ask_user", "plan_task", "web_search", "respond_only",
]
N_CLASSES = len(CLASSES)
EPS = 1.0e-12
BASE_ALPHA = {"sim": 0.20, "au": 0.85}


@dataclass(frozen=True)
class Spec:
    """A deliberately small calibration family and its regularization value."""

    family: str
    strength: float
    weighting: str = "uniform"

    @property
    def key(self) -> str:
        return f"{self.family}|strength={self.strength:g}|weighting={self.weighting}"

    @property
    def complexity(self) -> tuple[int, float, int]:
        # Prefer a smaller family and heavier regularization on exact ties.
        family_rank = {"class_alpha": 0, "class_alpha_bias": 1, "logistic": 2}[self.family]
        # Alpha-family strength is L2 lambda (larger is safer); logistic
        # strength is C (smaller is safer).
        return (family_rank, -self.strength if self.family != "logistic" else self.strength,
                0 if self.weighting == "uniform" else 1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="experiments/strict_blend_calibration")
    parser.add_argument("--maxiter", type=int, default=90,
                        help="L-BFGS iterations for the small class-alpha calibrators.")
    parser.add_argument("--skip-logistic", action="store_true",
                        help="Run only compact geometric class-alpha variants.")
    parser.add_argument("--gru-input", choices=("strict_new", "dual_half"), default="strict_new",
                        help="Use only the strict student, or a predeclared 50/50 old+strict GRU probability mix.")
    parser.add_argument("--families", nargs="+", choices=("class_alpha", "class_alpha_bias", "logistic"), default=None,
                        help="Optional subset for a focused rerun; default evaluates every compact family.")
    return parser.parse_args()


def source_index(source: np.ndarray) -> np.ndarray:
    value = np.where(source == "sim", 0, np.where(source == "au", 1, -1)).astype(np.int8)
    if np.any(value < 0):
        raise ValueError(f"Unknown source labels: {np.unique(source[value < 0]).tolist()}")
    return value


def session_of(sample_id: str) -> str:
    return sample_id.split("-step_", 1)[0]


def terminal_for_ids(ids: list[str]) -> np.ndarray:
    maxima: dict[str, int] = {}
    parsed: list[tuple[str, int]] = []
    for sample_id in ids:
        session, sep, raw_step = sample_id.rpartition("-step_")
        if not sep:
            raise ValueError(f"Unexpected ID without a step suffix: {sample_id}")
        step = int(raw_step)
        parsed.append((session, step))
        maxima[session] = max(maxima.get(session, -1), step)
    return np.asarray([step == maxima[session] for session, step in parsed], dtype=bool)


def macro(y: np.ndarray, prediction: np.ndarray, mask: np.ndarray | None = None) -> float:
    if mask is None:
        mask = np.ones(len(y), dtype=bool)
    return float(f1_score(y[mask], prediction[mask], labels=np.arange(N_CLASSES), average="macro", zero_division=0))


def class_weight(y: np.ndarray, mode: str) -> np.ndarray:
    if mode == "uniform":
        return np.ones(len(y), dtype=np.float64)
    if mode == "sqrt_balanced":
        counts = np.bincount(y, minlength=N_CLASSES).astype(np.float64)
        # All of the selected terminal training folds contain every class, but
        # protect against a future smaller split nevertheless.
        value = np.sqrt(len(y) / np.maximum(N_CLASSES * counts[y], 1.0))
        return value / value.mean()
    raise ValueError(mode)


def baseline_probability(xgb: np.ndarray, gru: np.ndarray, source: np.ndarray) -> np.ndarray:
    alpha = np.where(source == "sim", BASE_ALPHA["sim"], BASE_ALPHA["au"]).astype(np.float64)[:, None]
    logits = alpha * np.log(np.maximum(xgb, EPS)) + (1.0 - alpha) * np.log(np.maximum(gru, EPS))
    logits -= logits.max(axis=1, keepdims=True)
    probability = np.exp(logits)
    return (probability / probability.sum(axis=1, keepdims=True)).astype(np.float32)


class CompactCalibrator:
    """A fitted portable calibrator with a common ``predict`` surface."""

    def __init__(self, spec: Spec, maxiter: int):
        self.spec = spec
        self.maxiter = maxiter
        self.payload: dict[str, Any] = {}

    @staticmethod
    def _features(xgb: np.ndarray, gru: np.ndarray, source: np.ndarray) -> np.ndarray:
        """Source-interacted centred log-probabilities for the LR stacker."""
        lx = np.log(np.maximum(xgb, EPS))
        lg = np.log(np.maximum(gru, EPS))
        # Removing a row-wise constant makes the feature scale stable while
        # preserving all class comparisons in a probability vector.
        lx = lx - lx.mean(axis=1, keepdims=True)
        lg = lg - lg.mean(axis=1, keepdims=True)
        sim = (source == "sim").astype(np.float64)[:, None]
        au = 1.0 - sim
        return np.column_stack((lx * sim, lg * sim, lx * au, lg * au)).astype(np.float64)

    def fit(self, xgb: np.ndarray, gru: np.ndarray, source: np.ndarray, y: np.ndarray) -> "CompactCalibrator":
        if len(y) == 0:
            raise ValueError("Cannot fit a calibrator without terminal rows")
        if len(np.unique(y)) != N_CLASSES:
            raise ValueError("A calibration train split is missing at least one class")
        if self.spec.family == "logistic":
            x = self._features(xgb, gru, source)
            mean = x.mean(axis=0)
            scale = x.std(axis=0)
            scale[scale < 1.0e-6] = 1.0
            x = (x - mean) / scale
            # ``strength`` is C for the standard L2 multinomial formulation.
            estimator = LogisticRegression(
                C=self.spec.strength, penalty="l2", solver="lbfgs", multi_class="multinomial",
                max_iter=280, tol=1.0e-5, random_state=619,
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ConvergenceWarning)
                estimator.fit(x, y, sample_weight=class_weight(y, self.spec.weighting))
            if not np.array_equal(estimator.classes_, np.arange(N_CLASSES)):
                raise ValueError("Logistic calibration lost a class")
            self.payload = {
                "mean": mean, "scale": scale,
                "coef": estimator.coef_.astype(np.float64), "intercept": estimator.intercept_.astype(np.float64),
            }
            return self

        if self.spec.family not in {"class_alpha", "class_alpha_bias"}:
            raise ValueError(self.spec.family)
        source_id = source_index(source)
        lx = np.log(np.maximum(xgb, EPS)).astype(np.float64)
        lg = np.log(np.maximum(gru, EPS)).astype(np.float64)
        base_logit = np.asarray([math.log(BASE_ALPHA["sim"] / (1.0 - BASE_ALPHA["sim"])),
                                 math.log(BASE_ALPHA["au"] / (1.0 - BASE_ALPHA["au"]))], dtype=np.float64)
        use_bias = self.spec.family == "class_alpha_bias"
        n_params = N_CLASSES * (2 if use_bias else 1)
        sample_weight = class_weight(y, self.spec.weighting)
        # ``strength`` is conventional L2 lambda; mean-normalized loss keeps
        # it comparable as the nested train set changes size.
        def objective(value: np.ndarray) -> tuple[float, np.ndarray]:
            theta = value[:N_CLASSES]
            alpha = expit(base_logit[source_id, None] + theta[None, :])
            logits = alpha * lx + (1.0 - alpha) * lg
            if use_bias:
                # Centre for softmax identifiability and to keep an explicit
                # portable 14-value vector.
                bias = value[N_CLASSES:] - value[N_CLASSES:].mean()
                logits = logits + bias[None, :]
            log_probability = logits - logsumexp(logits, axis=1, keepdims=True)
            nll = -np.sum(sample_weight * log_probability[np.arange(len(y)), y]) / sample_weight.sum()
            penalty = self.spec.strength * float(np.mean(value * value))
            # Analytic gradients are small enough to be worth keeping: this
            # makes every nested fit deterministic and fast on CPU.
            probability = np.exp(log_probability)
            residual = probability
            residual[np.arange(len(y)), y] -= 1.0
            residual *= (sample_weight / sample_weight.sum())[:, None]
            dalpha = alpha * (1.0 - alpha)
            gradient_theta = np.sum(residual * dalpha * (lx - lg), axis=0)
            if use_bias:
                gradient_bias = residual.sum(axis=0)
                # Gradient through centred bias; sum is numerically ~zero but
                # make the parameterization exact.
                gradient_bias -= gradient_bias.mean()
                gradient = np.concatenate((gradient_theta, gradient_bias))
            else:
                gradient = gradient_theta
            gradient += (2.0 * self.spec.strength / n_params) * value
            return float(nll + penalty), gradient

        result = minimize(
            fun=lambda value: objective(value), x0=np.zeros(n_params, dtype=np.float64), jac=True,
            method="L-BFGS-B", options={"maxiter": self.maxiter, "ftol": 1.0e-10, "gtol": 1.0e-7},
        )
        if not result.success and result.status not in {1}:
            raise RuntimeError(f"Calibration optimization failed: {result.message}")
        payload: dict[str, Any] = {"theta": result.x[:N_CLASSES].astype(np.float64)}
        if use_bias:
            payload["bias"] = (result.x[N_CLASSES:] - result.x[N_CLASSES:].mean()).astype(np.float64)
        self.payload = payload
        return self

    def predict(self, xgb: np.ndarray, gru: np.ndarray, source: np.ndarray) -> np.ndarray:
        if not self.payload:
            raise RuntimeError("Fit before prediction")
        if self.spec.family == "logistic":
            x = self._features(xgb, gru, source)
            value = (x - self.payload["mean"]) / self.payload["scale"]
            logits = value @ self.payload["coef"].T + self.payload["intercept"]
        else:
            source_id = source_index(source)
            base_logit = np.asarray([math.log(BASE_ALPHA["sim"] / (1.0 - BASE_ALPHA["sim"])),
                                     math.log(BASE_ALPHA["au"] / (1.0 - BASE_ALPHA["au"]))], dtype=np.float64)
            alpha = expit(base_logit[source_id, None] + self.payload["theta"][None, :])
            logits = alpha * np.log(np.maximum(xgb, EPS)) + (1.0 - alpha) * np.log(np.maximum(gru, EPS))
            if "bias" in self.payload:
                logits = logits + self.payload["bias"][None, :]
        logits = logits - logits.max(axis=1, keepdims=True)
        probability = np.exp(logits)
        return (probability / probability.sum(axis=1, keepdims=True)).astype(np.float32)

    def json_payload(self) -> dict[str, Any]:
        payload = {
            "format": "strict_blend_calibrator_v1",
            "family": self.spec.family,
            "strength": self.spec.strength,
            "weighting": self.spec.weighting,
            "input": "raw XGBoost selector probabilities and portable strict no-E5 GRU probabilities",
            "source_baseline_alpha": BASE_ALPHA,
        }
        payload.update({key: np.asarray(value).tolist() for key, value in self.payload.items()})
        return payload


def fold_select_and_predict(
    specs: list[Spec], xgb: np.ndarray, gru: np.ndarray, source: np.ndarray, y: np.ndarray,
    terminal: np.ndarray, folds: np.ndarray, maxiter: int,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Nested second-level OOF with no held-session-fold label in selection."""
    output = np.zeros((len(y), N_CLASSES), dtype=np.float32)
    records: list[dict[str, Any]] = []
    unique = sorted(np.unique(folds).tolist())
    for held_fold in unique:
        held = folds == held_fold
        inner_records: list[tuple[float, Spec]] = []
        for spec in specs:
            inner_probability = np.zeros((len(y), N_CLASSES), dtype=np.float32)
            for valid_fold in unique:
                if valid_fold == held_fold:
                    continue
                train = terminal & (folds != held_fold) & (folds != valid_fold)
                valid = terminal & (folds == valid_fold)
                fitted = CompactCalibrator(spec, maxiter).fit(xgb[train], gru[train], source[train], y[train])
                inner_probability[valid] = fitted.predict(xgb[valid], gru[valid], source[valid])
            inner_mask = terminal & ~held
            inner_records.append((macro(y, inner_probability.argmax(axis=1), inner_mask), spec))
        # The stable preference ordering removes accidental tie-driven
        # complexity.  Higher F1 wins; then favour the simpler/stronger spec.
        best_score, chosen = max(inner_records, key=lambda pair: (pair[0], tuple(-value for value in pair[1].complexity)))
        train = terminal & ~held
        fitted = CompactCalibrator(chosen, maxiter).fit(xgb[train], gru[train], source[train], y[train])
        output[held] = fitted.predict(xgb[held], gru[held], source[held])
        records.append({
            "held_outer_fold": int(held_fold + 1), "chosen": chosen.key, "inner_terminal_macro_f1": float(best_score),
            "inner_candidates": [{"spec": spec.key, "terminal_macro_f1": float(score)} for score, spec in inner_records],
        })
    return output, records


def select_inside_training(
    specs: list[Spec], xgb: np.ndarray, gru: np.ndarray, source: np.ndarray, y: np.ndarray,
    terminal: np.ndarray, groups: np.ndarray, maxiter: int,
) -> tuple[Spec, list[dict[str, Any]]]:
    """Select a hyperparameter with only common-train sessions and labels."""
    folds = np.empty(len(y), dtype=np.int8)
    # Use group folds across the whole common-training partition, but score
    # only its terminal samples.  This avoids choosing a config on any fixed
    # validation label.
    for number, (_, valid) in enumerate(GroupKFold(n_splits=5).split(np.zeros(len(y)), y, groups)):
        folds[valid] = number
    table: list[dict[str, Any]] = []
    for spec in specs:
        probability = np.zeros((len(y), N_CLASSES), dtype=np.float32)
        for held in range(5):
            train = terminal & (folds != held)
            valid = folds == held
            fitted = CompactCalibrator(spec, maxiter).fit(xgb[train], gru[train], source[train], y[train])
            probability[valid] = fitted.predict(xgb[valid], gru[valid], source[valid])
        table.append({"spec": spec.key, "terminal_macro_f1": macro(y, probability.argmax(axis=1), terminal)})
    def spec_for(record: dict[str, Any]) -> Spec:
        return next(value for value in specs if value.key == record["spec"])
    picked_record = max(table, key=lambda row: (row["terminal_macro_f1"], tuple(-value for value in spec_for(row).complexity)))
    return spec_for(picked_record), table


def char_override(prediction: np.ndarray, terminal: np.ndarray, char: np.ndarray) -> np.ndarray:
    result = prediction.copy()
    mask = terminal & (char >= 0)
    result[mask] = char[mask]
    return result


def metric_row(
    y: np.ndarray, probability: np.ndarray, terminal: np.ndarray, char: np.ndarray,
) -> dict[str, float]:
    fallback = probability.argmax(axis=1)
    with_char = char_override(fallback, terminal, char)
    hybrid = with_char.copy()
    # In a complete training session every non-terminal target is exactly
    # revealed by a later assistant_action.  It is the deployment-aligned
    # recovery path, not a learned prediction.
    hybrid[~terminal] = y[~terminal]
    return {
        "all_fallback_macro_f1": macro(y, fallback),
        "terminal_fallback_macro_f1": macro(y, fallback, terminal),
        "terminal_with_char20_macro_f1": macro(y, with_char, terminal),
        "hybrid_future_history_plus_char20_macro_f1": macro(y, hybrid),
    }


def load_data(root: Path) -> dict[str, Any]:
    samples = [json.loads(line) for line in (root / "data/train.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    labels = {row["id"]: row["action"] for row in csv.DictReader((root / "data/train_labels.csv").open(encoding="utf-8"))}
    split = {row["id"]: row["split"] for row in csv.DictReader((root / "splits/common_split.csv").open(encoding="utf-8"))}
    ids = [str(row["id"]) for row in samples]
    y = np.asarray([CLASSES.index(labels[sample_id]) for sample_id in ids], dtype=np.int64)
    source = np.asarray(["au" if sample_id.startswith("sess_au_") else "sim" for sample_id in ids])
    groups = np.asarray([session_of(sample_id) for sample_id in ids])
    terminal = terminal_for_ids(ids)
    xgb = np.load(root / "experiments/modeling_ak_live_xgb_proxy/oof_probabilities.npy").astype(np.float32)
    gru = np.load(root / "experiments/modeling_ai_sequence_nested_distill_e16_i4/oof_probabilities.npy").astype(np.float32)
    old_gru = np.load(root / "experiments/modeling_ai_sequence_fusion_noe5/oof_probabilities.npy").astype(np.float32)
    fold = np.load(root / "experiments/modeling_ai_sequence_nested_distill_e16_i4/outer_fold_assignment.npy").astype(np.int8)
    shape = np.load(root / "experiments/modeling_ak_live_xgb_proxy/input_shapes.npz", allow_pickle=True)
    if not (xgb.shape == gru.shape == old_gru.shape == (len(samples), N_CLASSES) and len(fold) == len(samples)):
        raise ValueError("Full OOF artifacts do not align")
    if not (np.array_equal(shape["y"].astype(np.int64), y) and np.array_equal(shape["source"].astype(str), source)
            and np.array_equal(shape["terminal"].astype(bool), terminal)):
        raise ValueError("Raw-XGB proxy metadata no longer aligns with training rows")
    expected_fold = np.empty(len(samples), dtype=np.int8)
    for number, (_, valid) in enumerate(GroupKFold(n_splits=5).split(np.zeros(len(y)), y, groups)):
        expected_fold[valid] = number
    if not np.array_equal(expected_fold, fold):
        raise ValueError("Strict student outer-fold assignment differs from current session GroupKFold")

    rule = np.load(root / "experiments/terminal_rule_combinations/oof_candidates.npz", allow_pickle=True)
    char = rule["char20"].astype(np.int16)
    # The rule artifact intentionally leaves its non-terminal fold IDs at -1
    # because it can only override terminal fallback rows.  Its terminal
    # outer-fold assignment must nevertheless exactly match the strict GRU.
    if not (np.array_equal(rule["y"].astype(np.int64), y) and np.array_equal(rule["terminal"].astype(bool), terminal)
            and np.array_equal(rule["fold"].astype(np.int8)[terminal], fold[terminal])):
        raise ValueError("Char20 OOF artifact does not align with strict student folds")

    valid_index = np.asarray([index for index, sample_id in enumerate(ids) if split[sample_id] == "val"], dtype=np.int64)
    train_index = np.asarray([index for index, sample_id in enumerate(ids) if split[sample_id] == "train"], dtype=np.int64)
    # This is intentionally *not* the old 2025-feature ``modeling_ac_tune2``
    # table.  It is the strict common-train-only 2032-feature raw selector.
    strict_blocks = np.load(root / "experiments/au_source_strict_common/strict_blocks.npz", allow_pickle=False)
    if not (np.array_equal(strict_blocks["ids_train"].astype(str), np.asarray(ids, dtype=str)[train_index])
            and np.array_equal(strict_blocks["ids_valid"].astype(str), np.asarray(ids, dtype=str)[valid_index])
            and np.array_equal(strict_blocks["target_valid"].astype(np.int64), y[valid_index])):
        raise ValueError("Strict 2032 selector blocks do not align with the common split")
    strict_train_xgb_path = root / "experiments/strict_blend_calibration/common_train_raw_xgb_oof.npy"
    strict_train_ids_path = root / "experiments/strict_blend_calibration/common_train_ids.npy"
    if not (strict_train_xgb_path.exists() and strict_train_ids_path.exists()):
        raise FileNotFoundError(
            "Strict common-train raw selector OOF is required. Run "
            "python src/build_strict_common_raw_selector_oof.py first."
        )
    strict_train_xgb = np.load(strict_train_xgb_path, allow_pickle=False).astype(np.float32)
    strict_train_ids = np.load(strict_train_ids_path, allow_pickle=False).astype(str)
    if not (strict_train_xgb.shape == (len(train_index), N_CLASSES)
            and np.array_equal(strict_train_ids, strict_blocks["ids_train"].astype(str))):
        raise ValueError("Strict common-train raw selector OOF does not align with strict blocks")
    fixed_xgb = np.load(root / "experiments/au_source_strict_common/raw_global_valid_probability.npy").astype(np.float32)
    fixed_gru = np.load(root / "experiments/modeling_ai_sequence_fusion_fixed_strictdistill_w35/valid_probabilities.npy").astype(np.float32)
    fixed_old_gru = np.load(root / "experiments/modeling_ai_sequence_fusion_fixed_noe5/valid_probabilities.npy").astype(np.float32)
    fixed_terminal = terminal[valid_index]
    saved_terminal = np.load(root / "experiments/modeling_ai_sequence_fusion_fixed_strictdistill_w35/terminal_mask.npy").astype(bool)
    if not (fixed_xgb.shape == fixed_gru.shape == fixed_old_gru.shape == (len(valid_index), N_CLASSES)
            and np.array_equal(saved_terminal, fixed_terminal)):
        raise ValueError("Fixed validation artifacts do not align")
    fixed_char = np.full(len(valid_index), -1, dtype=np.int16)
    fixed_terminal_positions = np.flatnonzero(fixed_terminal)
    if not (np.array_equal(rule["fixed_y"].astype(np.int64), y[valid_index][fixed_terminal])
            and len(rule["fixed_char20"]) == len(fixed_terminal_positions)):
        raise ValueError("Fixed char20 artifact does not align with validation labels")
    fixed_char[fixed_terminal_positions] = rule["fixed_char20"].astype(np.int16)
    return {
        "ids": ids, "y": y, "source": source, "groups": groups, "terminal": terminal,
        "xgb": xgb, "gru": gru, "old_gru": old_gru, "fold": fold, "char": char,
        "train_index": train_index, "valid_index": valid_index, "strict_train_xgb": strict_train_xgb,
        "fixed_xgb": fixed_xgb, "fixed_gru": fixed_gru, "fixed_old_gru": fixed_old_gru,
        "fixed_terminal": fixed_terminal, "fixed_char": fixed_char,
    }


def family_specs(skip_logistic: bool) -> dict[str, list[Spec]]:
    # Values are deliberately sparse.  The nested inner loop selects among
    # them without touching the outer held fold; no broad per-class grid is
    # searched.
    result = {
        "class_alpha": [
            Spec("class_alpha", strength, weighting)
            for strength in (0.10, 0.50, 2.00, 8.00)
            for weighting in ("uniform", "sqrt_balanced")
        ],
        "class_alpha_bias": [
            Spec("class_alpha_bias", strength, weighting)
            for strength in (0.50, 2.00, 8.00)
            for weighting in ("uniform", "sqrt_balanced")
        ],
    }
    if not skip_logistic:
        result["logistic"] = [
            Spec("logistic", strength, weighting)
            for strength in (0.001, 0.006)
            for weighting in ("uniform", "sqrt_balanced")
        ]
    return result


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    out = root / args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    data = load_data(root)
    y = data["y"]; source = data["source"]; terminal = data["terminal"]
    xgb = data["xgb"]; gru = data["gru"]; char = data["char"]; fold = data["fold"]
    train_index = data["train_index"]; valid_index = data["valid_index"]
    fixed_y = y[valid_index]; fixed_source = source[valid_index]
    strict_train_xgb = data["strict_train_xgb"]
    fixed_xgb = data["fixed_xgb"]; fixed_gru = data["fixed_gru"]
    if args.gru_input == "dual_half":
        # This is a fixed symmetric mixture, not a label-selected weight.  A
        # separate nested audit records whether any alternative weight is
        # supported before this compact inference option is considered.
        gru = (0.5 * data["old_gru"] + 0.5 * gru).astype(np.float32)
        fixed_gru = (0.5 * data["fixed_old_gru"] + 0.5 * fixed_gru).astype(np.float32)
    fixed_terminal = data["fixed_terminal"]; fixed_char = data["fixed_char"]

    baseline_full = baseline_probability(xgb, gru, source)
    baseline_fixed = baseline_probability(fixed_xgb, fixed_gru, fixed_source)
    baseline = {
        "full_nested_group_oof": metric_row(y, baseline_full, terminal, char),
        "common_fixed_validation": metric_row(fixed_y, baseline_fixed, fixed_terminal, fixed_char),
    }

    report: dict[str, Any] = {
        "scope": "Experiment-only strict calibration audit; model/, script.py and ZIP files are never modified.",
        "validation_protocol": {
            "full": "For each of five whole-session outer folds, every candidate's spec is selected by four-fold inner calibration and fitted without the held fold's labels.",
            "fixed": "Specs are selected by GroupKFold only within common-train labels, then fit on all common-train terminal rows and evaluated on untouched common validation labels.",
            "base_predictions": "raw 2032-feature XGB group-OOF + strictly nested-distilled no-E5 GRU group-OOF; fixed side uses strict common-train-only 2032 raw-XGB selector probabilities (seed 808) plus strict-distill GRU validation probabilities.",
            "char20": "Char20 is never used to fit a calibrator. It is evaluated only as the final terminal override after the base blend.",
            "gru_input": args.gru_input,
            "n_full_rows": int(len(y)), "n_full_terminal": int(terminal.sum()),
            "n_common_train": int(len(train_index)), "n_fixed_rows": int(len(valid_index)), "n_fixed_terminal": int(fixed_terminal.sum()),
        },
        "baseline_source_scalar_geometric": {
            "alpha_xgb": BASE_ALPHA,
            **baseline,
        },
        "temperature_exponent_check": {
            "result": "rejected_as_reparameterization",
            "reason": "With one scalar exponent per source/model, the argmax of p_xgb^a * p_gru^b depends only on a/(a+b), exactly the existing source alpha. A final temperature rescales all class logits and cannot change macro-F1 predictions.",
        },
        "families": {},
        "recommendation": None,
    }

    selected_families = family_specs(args.skip_logistic)
    if args.families is not None:
        selected_families = {name: specs for name, specs in selected_families.items() if name in args.families}
        if not selected_families:
            raise ValueError("No requested family is available (perhaps --skip-logistic was also set)")
    candidates: list[tuple[str, Spec, CompactCalibrator, dict[str, Any]]] = []
    for family, specs in selected_families.items():
        print(f"family={family} specs={len(specs)} strict_outer_nested=begin", flush=True)
        nested_probability, outer_records = fold_select_and_predict(
            specs, xgb, gru, source, y, terminal, fold, args.maxiter,
        )
        print(f"family={family} common_train_selection=begin", flush=True)
        common_spec, common_table = select_inside_training(
            specs, strict_train_xgb, gru[train_index], source[train_index], y[train_index],
            terminal[train_index], data["groups"][train_index], args.maxiter,
        )
        fixed_fitted = CompactCalibrator(common_spec, args.maxiter).fit(
            strict_train_xgb[terminal[train_index]], gru[train_index][terminal[train_index]],
            source[train_index][terminal[train_index]], y[train_index][terminal[train_index]],
        )
        fixed_probability = fixed_fitted.predict(fixed_xgb, fixed_gru, fixed_source)
        nested_metrics = metric_row(y, nested_probability, terminal, char)
        fixed_metrics = metric_row(fixed_y, fixed_probability, fixed_terminal, fixed_char)
        chosen_by_outer = [record["chosen"] for record in outer_records]
        family_result: dict[str, Any] = {
            "full_nested_group_oof": nested_metrics,
            "common_fixed_validation": fixed_metrics,
            "outer_fold_choices": outer_records,
            "common_train_selection": {"chosen": common_spec.key, "candidates": common_table},
            "delta_vs_source_scalar": {
                "full_terminal": nested_metrics["terminal_fallback_macro_f1"] - baseline["full_nested_group_oof"]["terminal_fallback_macro_f1"],
                "full_hybrid": nested_metrics["hybrid_future_history_plus_char20_macro_f1"] - baseline["full_nested_group_oof"]["hybrid_future_history_plus_char20_macro_f1"],
                "fixed_terminal": fixed_metrics["terminal_fallback_macro_f1"] - baseline["common_fixed_validation"]["terminal_fallback_macro_f1"],
                "fixed_hybrid": fixed_metrics["hybrid_future_history_plus_char20_macro_f1"] - baseline["common_fixed_validation"]["hybrid_future_history_plus_char20_macro_f1"],
            },
        }
        # Both operating regimes must improve on terminal fallback *and* the
        # deployment-aligned hybrid.  A non-terminal all-fallback movement is
        # reported but is intentionally not the decision objective, because
        # exact future-history recovery replaces those predictions.
        delta = family_result["delta_vs_source_scalar"]
        safe = all(delta[name] > 0.0 for name in ("full_terminal", "full_hybrid", "fixed_terminal", "fixed_hybrid"))
        family_result["safe_two_split_candidate"] = safe
        report["families"][family] = family_result
        if safe:
            # Full OOF refit gives a portable parameter set for review only.
            # It is not copied into the live bundle by this script.
            fitted_all = CompactCalibrator(common_spec, args.maxiter).fit(
                xgb[terminal], gru[terminal], source[terminal], y[terminal]
            )
            candidates.append((family, common_spec, fitted_all, family_result))
        print(json.dumps({"family": family, "safe": safe, "delta": delta}, ensure_ascii=False), flush=True)

    # Choose using only full nested terminal/hybrid evidence among candidates;
    # fixed values act as an independent gate, not a tiebreaker.
    if candidates:
        family, spec, fitted, result = max(
            candidates,
            key=lambda item: (
                item[3]["delta_vs_source_scalar"]["full_terminal"],
                item[3]["delta_vs_source_scalar"]["full_hybrid"],
                tuple(-value for value in item[1].complexity),
            ),
        )
        portable = fitted.json_payload()
        portable.update({
            "selection": {
                "family": family,
                "chosen_common_train_spec": spec.key,
                "gru_input": args.gru_input,
                "note": "Fit on all full-OOF terminal rows only after both held-out protocols passed. Review/integration remains manual; this script does not touch the submission bundle.",
            },
            "evidence": result["delta_vs_source_scalar"],
        })
        artifact = out / "recommended_portable_calibrator.json"
        artifact.write_text(json.dumps(portable, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        report["recommendation"] = {
            "family": family, "common_train_spec": spec.key,
            "artifact": str(artifact.relative_to(root)),
            "apply_contract": "Use only after deterministic future-history recovery fails; compute raw-XGB and strict no-E5-GRU probabilities, then run this JSON's portable logit transformation. Apply char20 only afterward on maximum supplied session rows.",
        }
    else:
        report["recommendation"] = {
            "decision": "No calibration family cleared the required two-split terminal + hybrid improvement gate; retain source scalar geometric alpha sim=0.20 / au=0.85.",
        }

    (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary = {
        "baseline_terminal_full": baseline["full_nested_group_oof"]["terminal_fallback_macro_f1"],
        "baseline_terminal_fixed": baseline["common_fixed_validation"]["terminal_fallback_macro_f1"],
        "recommendation": report["recommendation"],
        "families": {
            name: {"safe": value["safe_two_split_candidate"], "delta": value["delta_vs_source_scalar"]}
            for name, value in report["families"].items()
        },
    }
    (out / "README.md").write_text(
        "# Strict blend/calibration audit\n\n"
        "Run `python src/experiment_strict_blend_calibration.py`.  The machine-readable evidence is "
        "`report.json`; a portable JSON is emitted only after a candidate clears both validation protocols.\n\n"
        "```json\n" + json.dumps(summary, ensure_ascii=False, indent=2) + "\n```\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
