#!/usr/bin/env python3
"""Leakage-controlled source-by-class calibration audit for public-best B43.

The B43 baseline is deliberately fixed:

* raw 2032-feature seed-42 selector probability;
* the original portable no-E5 GRU probability;
* geometric raw/GRU blending with SIM=.20 and AU=.85; and
* the existing char20 rule as the immutable final terminal override.

This utility only tests a low-capacity alternative to a single alpha per
source.  For each source, classes are placed into 2--4 *training-only*
buckets according to their raw-vs-GRU true-label log-probability contrast.
One regularized alpha offset is fitted per bucket.  That is at most eight
source-by-class mixing parameters, rather than an unconstrained 28-value
class threshold table.  It never reads/writes model/, script.py, B42.zip, or
baseline_submit.zip.

Every candidate is selected without a held session fold's labels in a nested
five-fold GroupKFold audit.  A second, matching seed-42 common-train/fixed
protocol is required.  Char20 is never a calibration feature or a selection
signal; it is applied only after the fallback prediction is complete.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logsumexp
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold


CLASSES = (
    "read_file", "grep_search", "list_directory", "glob_pattern",
    "edit_file", "write_file", "apply_patch", "run_bash", "run_tests",
    "lint_or_typecheck", "ask_user", "plan_task", "web_search", "respond_only",
)
N_CLASSES = len(CLASSES)
EPS = 1.0e-12
BASE_ALPHA = {"sim": 0.20, "au": 0.85}
SOURCE_NAMES = ("sim", "au")


@dataclass(frozen=True)
class Spec:
    """A deliberately small, inference-portable source/class alpha family."""

    buckets: int
    l2: float
    weighting: str

    @property
    def baseline(self) -> bool:
        return self.buckets == 0

    @property
    def key(self) -> str:
        if self.baseline:
            return "b43_scalar_source_alpha"
        return f"source_bucket_alpha|buckets={self.buckets}|l2={self.l2:g}|weighting={self.weighting}"

    @property
    def complexity(self) -> tuple[int, int, float, int]:
        """Smaller is preferred when validation scores exactly tie."""
        return (
            0 if self.baseline else 1,
            self.buckets,
            -self.l2,
            0 if self.weighting == "uniform" else 1,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="experiments/b43_source_class_calibration")
    parser.add_argument("--maxiter", type=int, default=55)
    parser.add_argument(
        "--selection-mode",
        choices=("average", "stable_terminal", "stable_all"),
        default="average",
        help="Nested-spec selection policy. Stability modes require non-inferiority versus exact B43 in every inner fold.",
    )
    return parser.parse_args()


def session_of(sample_id: str) -> str:
    session, marker, _ = sample_id.rpartition("-step_")
    if not marker:
        raise ValueError(f"Unexpected ID without step suffix: {sample_id}")
    return session


def source_for_ids(ids: np.ndarray) -> np.ndarray:
    value = np.asarray(["au" if name.startswith("sess_au_") else "sim" for name in ids.astype(str)])
    if not np.isin(value, SOURCE_NAMES).all():
        raise ValueError("Unexpected source name")
    return value


def terminal_for_ids(ids: np.ndarray) -> np.ndarray:
    maximum: dict[str, int] = {}
    parsed: list[tuple[str, int]] = []
    for sample_id in ids.astype(str):
        session, marker, raw_step = sample_id.rpartition("-step_")
        if not marker:
            raise ValueError(f"Unexpected ID without step suffix: {sample_id}")
        step = int(raw_step)
        maximum[session] = max(maximum.get(session, -1), step)
        parsed.append((session, step))
    return np.asarray([step == maximum[session] for session, step in parsed], dtype=bool)


def source_index(source: np.ndarray) -> np.ndarray:
    result = np.where(source == "sim", 0, np.where(source == "au", 1, -1)).astype(np.int8)
    if np.any(result < 0):
        raise ValueError(f"Unknown source: {np.unique(source[result < 0]).tolist()}")
    return result


def macro(y: np.ndarray, prediction: np.ndarray, mask: np.ndarray | None = None) -> float:
    if mask is None:
        mask = np.ones(len(y), dtype=bool)
    return float(
        f1_score(
            y[mask], prediction[mask], labels=np.arange(N_CLASSES), average="macro", zero_division=0,
        )
    )


def scalar_geometric(raw: np.ndarray, gru: np.ndarray, source: np.ndarray) -> np.ndarray:
    """The exact source-scalar B43 fallback blend."""
    alpha = np.where(source == "sim", BASE_ALPHA["sim"], BASE_ALPHA["au"]).astype(np.float64)[:, None]
    logits = alpha * np.log(np.maximum(raw, EPS)) + (1.0 - alpha) * np.log(np.maximum(gru, EPS))
    logits -= logits.max(axis=1, keepdims=True)
    probability = np.exp(logits)
    return (probability / probability.sum(axis=1, keepdims=True)).astype(np.float32)


def with_char20(prediction: np.ndarray, terminal: np.ndarray, char20: np.ndarray) -> np.ndarray:
    output = prediction.copy()
    override = terminal & (char20 >= 0)
    output[override] = char20[override]
    return output


def metric_row(y: np.ndarray, probability: np.ndarray, terminal: np.ndarray, char20: np.ndarray) -> dict[str, float]:
    fallback = probability.argmax(axis=1).astype(np.int16)
    terminal_char = with_char20(fallback, terminal, char20)
    hybrid = terminal_char.copy()
    # On the fully labeled audit, future assistant history deterministically
    # exposes every non-terminal action.  This mirrors B43 inference order.
    hybrid[~terminal] = y[~terminal]
    return {
        "all_fallback_macro_f1": macro(y, fallback),
        "terminal_fallback_macro_f1": macro(y, fallback, terminal),
        "terminal_with_char20_macro_f1": macro(y, terminal_char, terminal),
        "hybrid_future_history_plus_char20_macro_f1": macro(y, hybrid),
    }


def deltas(candidate: dict[str, float], baseline: dict[str, float]) -> dict[str, float]:
    return {name: float(candidate[name] - baseline[name]) for name in baseline}


def class_weights(y: np.ndarray, mode: str) -> np.ndarray:
    if mode == "uniform":
        return np.ones(len(y), dtype=np.float64)
    if mode == "sqrt_balanced":
        counts = np.bincount(y, minlength=N_CLASSES).astype(np.float64)
        value = np.sqrt(len(y) / np.maximum(N_CLASSES * counts[y], 1.0))
        return value / value.mean()
    raise ValueError(mode)


class BucketAlphaCalibrator:
    """Source-specific alpha buckets fitted by regularized multiclass NLL.

    Bucket membership is derived afresh from *only the current fit rows*.
    A class's sort statistic is its average true-label raw-vs-GRU log
    probability contrast.  Consequently no held-fold label can affect a
    parameter or class bucket used to predict that held fold.
    """

    def __init__(self, spec: Spec, maxiter: int):
        self.spec = spec
        self.maxiter = maxiter
        self.membership: np.ndarray | None = None
        self.signals: np.ndarray | None = None
        self.theta: np.ndarray | None = None

    def _build_membership(
        self, lx: np.ndarray, lg: np.ndarray, source_id: np.ndarray, y: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        membership = np.zeros((len(SOURCE_NAMES), N_CLASSES), dtype=np.int8)
        signals = np.zeros((len(SOURCE_NAMES), N_CLASSES), dtype=np.float64)
        for source_number in range(len(SOURCE_NAMES)):
            source_rows = source_id == source_number
            if not np.any(source_rows):
                raise ValueError(f"No calibration rows for source {SOURCE_NAMES[source_number]}")
            for class_number in range(N_CLASSES):
                rows = source_rows & (y == class_number)
                if np.any(rows):
                    signals[source_number, class_number] = float(
                        np.mean(lx[rows, class_number] - lg[rows, class_number])
                    )
                # A class absent in a very small inner fold receives a neutral
                # statistic, then a class-id tie break below.  The full audit
                # has every class for both sources, but this remains safe.
            order = np.lexsort((np.arange(N_CLASSES), signals[source_number]))
            membership[source_number, order] = (
                np.arange(N_CLASSES, dtype=np.int8) * self.spec.buckets // N_CLASSES
            )
        return membership, signals

    def fit(self, raw: np.ndarray, gru: np.ndarray, source: np.ndarray, y: np.ndarray) -> "BucketAlphaCalibrator":
        if self.spec.baseline:
            return self
        if len(y) == 0:
            raise ValueError("Cannot calibrate without terminal training rows")
        source_id = source_index(source)
        lx = np.log(np.maximum(raw, EPS)).astype(np.float64)
        lg = np.log(np.maximum(gru, EPS)).astype(np.float64)
        membership, signals = self._build_membership(lx, lg, source_id, y)
        weights = class_weights(y, self.spec.weighting).astype(np.float64)
        weights /= weights.sum()
        base_logit = np.asarray(
            [
                math.log(BASE_ALPHA["sim"] / (1.0 - BASE_ALPHA["sim"])),
                math.log(BASE_ALPHA["au"] / (1.0 - BASE_ALPHA["au"])),
            ],
            dtype=np.float64,
        )
        n_params = len(SOURCE_NAMES) * self.spec.buckets
        class_index = np.arange(N_CLASSES)[None, :]
        row_membership = membership[source_id[:, None], class_index]

        def objective(flat_theta: np.ndarray) -> tuple[float, np.ndarray]:
            theta = flat_theta.reshape(len(SOURCE_NAMES), self.spec.buckets)
            theta_by_logit = theta[source_id[:, None], row_membership]
            alpha = expit(base_logit[source_id, None] + theta_by_logit)
            logits = alpha * lx + (1.0 - alpha) * lg
            log_probability = logits - logsumexp(logits, axis=1, keepdims=True)
            nll = -float(np.sum(weights * log_probability[np.arange(len(y)), y]))
            residual = np.exp(log_probability)
            residual[np.arange(len(y)), y] -= 1.0
            residual *= weights[:, None]
            alpha_gradient = alpha * (1.0 - alpha) * (lx - lg)
            gradient = np.zeros_like(theta)
            for source_number in range(len(SOURCE_NAMES)):
                rows = source_id == source_number
                for bucket in range(self.spec.buckets):
                    columns = membership[source_number] == bucket
                    gradient[source_number, bucket] = float(
                        np.sum(residual[rows][:, columns] * alpha_gradient[rows][:, columns])
                    )
            penalty = self.spec.l2 * float(np.mean(theta * theta))
            gradient += (2.0 * self.spec.l2 / n_params) * theta
            return nll + penalty, gradient.reshape(-1)

        result = minimize(
            fun=lambda value: objective(value),
            x0=np.zeros(n_params, dtype=np.float64),
            jac=True,
            method="L-BFGS-B",
            bounds=[(-2.0, 2.0)] * n_params,
            options={"maxiter": self.maxiter, "ftol": 1.0e-10, "gtol": 1.0e-7},
        )
        # Status 1 is the normal L-BFGS iteration limit and still yields a
        # deterministic bounded point.  Other failures should not silently
        # produce a candidate.
        if not result.success and result.status != 1:
            raise RuntimeError(f"Bucket calibration optimization failed: {result.message}")
        self.membership = membership
        self.signals = signals
        self.theta = result.x.reshape(len(SOURCE_NAMES), self.spec.buckets).astype(np.float64)
        return self

    def predict(self, raw: np.ndarray, gru: np.ndarray, source: np.ndarray) -> np.ndarray:
        if self.spec.baseline:
            return scalar_geometric(raw, gru, source)
        if self.membership is None or self.theta is None:
            raise RuntimeError("fit() must run before predict()")
        source_id = source_index(source)
        base_logit = np.asarray(
            [
                math.log(BASE_ALPHA["sim"] / (1.0 - BASE_ALPHA["sim"])),
                math.log(BASE_ALPHA["au"] / (1.0 - BASE_ALPHA["au"])),
            ],
            dtype=np.float64,
        )
        class_index = np.arange(N_CLASSES)[None, :]
        membership = self.membership[source_id[:, None], class_index]
        theta = self.theta[source_id[:, None], membership]
        alpha = expit(base_logit[source_id, None] + theta)
        logits = alpha * np.log(np.maximum(raw, EPS)) + (1.0 - alpha) * np.log(np.maximum(gru, EPS))
        logits -= logits.max(axis=1, keepdims=True)
        probability = np.exp(logits)
        return (probability / probability.sum(axis=1, keepdims=True)).astype(np.float32)

    def portable_payload(self) -> dict[str, Any]:
        if self.spec.baseline:
            return {
                "format": "b43_scalar_geometric_v1",
                "source_raw_weights": BASE_ALPHA,
                "note": "No source-by-class calibration selected.",
            }
        if self.membership is None or self.theta is None or self.signals is None:
            raise RuntimeError("No fitted payload")
        return {
            "format": "b43_source_bucket_alpha_v1",
            "base": {
                "raw_selector": "exact seed-42 2032-feature selector",
                "gru": "original B42_9 portable no-E5 GRU",
                "source_raw_weights": BASE_ALPHA,
                "blend": "per-class geometric alpha = sigmoid(logit(source_base_alpha) + theta[source][class_bucket])",
            },
            "spec": self.spec.key,
            "classes": list(CLASSES),
            "source_names": list(SOURCE_NAMES),
            "bucket_by_source_and_class": self.membership.astype(int).tolist(),
            "training_only_raw_vs_gru_true_label_signal": self.signals.tolist(),
            "theta": self.theta.tolist(),
            "apply_contract": (
                "Apply after raw selector and original no-E5 GRU probabilities, before fallback argmax. "
                "Preserve the immutable char20 terminal override and final exact future-history recovery ordering."
            ),
        }


def all_specs() -> list[Spec]:
    # 18 constrained alternatives plus the exact no-intervention B43 baseline.
    return [Spec(0, 0.0, "uniform")] + [
        Spec(buckets, l2, weighting)
        for buckets in (2, 3, 4)
        for l2 in (0.10, 0.50, 2.00)
        for weighting in ("uniform", "sqrt_balanced")
    ]


def choose_row(rows: list[dict[str, Any]], selection_mode: str) -> dict[str, Any]:
    """Choose a spec only from inner-fold evidence.

    ``stable_terminal`` refuses a source/class calibration whose inner held
    session fold lowers terminal fallback F1 against B43.  ``stable_all``
    applies the same safety requirement to all fallback F1.  The exact B43
    source-scalar baseline is always eligible, so these guards cannot force a
    risky calibration through an empty candidate set.
    """
    if selection_mode not in {"average", "stable_terminal", "stable_all"}:
        raise ValueError(selection_mode)
    eligible = rows
    if selection_mode != "average":
        eligible = [
            row for row in rows
            if row["min_inner_terminal_delta_vs_b43"] >= -1.0e-12
            and (
                selection_mode != "stable_all"
                or row["min_inner_all_delta_vs_b43"] >= -1.0e-12
            )
        ]
        if not eligible:
            raise RuntimeError("B43 baseline should always satisfy the stability gate")
    return max(
        eligible,
        key=lambda row: (
            row["terminal_fallback_macro_f1"],
            row["all_fallback_macro_f1"],
            tuple(-value for value in row["spec"].complexity),
        ),
    )


def nested_full_probability(
    specs: list[Spec], raw: np.ndarray, gru: np.ndarray, source: np.ndarray, y: np.ndarray,
    terminal: np.ndarray, folds: np.ndarray, maxiter: int, selection_mode: str,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Nested second-level GroupOOF: held sessions never select their spec."""
    output = np.zeros((len(y), N_CLASSES), dtype=np.float32)
    records: list[dict[str, Any]] = []
    fold_numbers = sorted(np.unique(folds).tolist())
    for held_fold in fold_numbers:
        held = folds == held_fold
        candidate_rows: list[dict[str, Any]] = []
        for spec in specs:
            inner_probability = np.zeros((len(y), N_CLASSES), dtype=np.float32)
            fold_scores: list[dict[str, float | int]] = []
            for inner_fold in fold_numbers:
                if inner_fold == held_fold:
                    continue
                fit_rows = terminal & (folds != held_fold) & (folds != inner_fold)
                valid_rows = folds == inner_fold
                fitted = BucketAlphaCalibrator(spec, maxiter).fit(
                    raw[fit_rows], gru[fit_rows], source[fit_rows], y[fit_rows]
                )
                inner_probability[valid_rows] = fitted.predict(raw[valid_rows], gru[valid_rows], source[valid_rows])
                inner_prediction = inner_probability[valid_rows].argmax(axis=1)
                fold_scores.append({
                    "fold": int(inner_fold + 1),
                    "terminal_fallback_macro_f1": macro(y[valid_rows], inner_prediction, terminal[valid_rows]),
                    "all_fallback_macro_f1": macro(y[valid_rows], inner_prediction),
                })
            prediction = inner_probability.argmax(axis=1)
            inner_rows = ~held
            candidate_rows.append({
                "spec": spec,
                "terminal_fallback_macro_f1": macro(y, prediction, inner_rows & terminal),
                "all_fallback_macro_f1": macro(y, prediction, inner_rows),
                "fold_scores": fold_scores,
            })
        baseline_row = next(row for row in candidate_rows if row["spec"].baseline)
        baseline_scores = {int(row["fold"]): row for row in baseline_row["fold_scores"]}
        for row in candidate_rows:
            deltas_for_row = [
                (
                    float(score["terminal_fallback_macro_f1"] - baseline_scores[int(score["fold"])]["terminal_fallback_macro_f1"]),
                    float(score["all_fallback_macro_f1"] - baseline_scores[int(score["fold"])]["all_fallback_macro_f1"]),
                )
                for score in row["fold_scores"]
            ]
            row["min_inner_terminal_delta_vs_b43"] = min(value[0] for value in deltas_for_row)
            row["min_inner_all_delta_vs_b43"] = min(value[1] for value in deltas_for_row)
        chosen = choose_row(candidate_rows, selection_mode)
        fit_rows = terminal & ~held
        fitted = BucketAlphaCalibrator(chosen["spec"], maxiter).fit(
            raw[fit_rows], gru[fit_rows], source[fit_rows], y[fit_rows]
        )
        output[held] = fitted.predict(raw[held], gru[held], source[held])
        records.append({
            "held_outer_fold": int(held_fold + 1),
            "chosen": chosen["spec"].key,
            "inner_terminal_fallback_macro_f1": chosen["terminal_fallback_macro_f1"],
            "inner_all_fallback_macro_f1": chosen["all_fallback_macro_f1"],
            "inner_candidates": [
                {
                    "spec": row["spec"].key,
                    "terminal_fallback_macro_f1": row["terminal_fallback_macro_f1"],
                    "all_fallback_macro_f1": row["all_fallback_macro_f1"],
                    "min_inner_terminal_delta_vs_b43": row["min_inner_terminal_delta_vs_b43"],
                    "min_inner_all_delta_vs_b43": row["min_inner_all_delta_vs_b43"],
                }
                for row in candidate_rows
            ],
        })
        print(
            json.dumps({"outer_fold": int(held_fold + 1), "chosen": chosen["spec"].key}, ensure_ascii=False),
            flush=True,
        )
    return output, records


def select_common_train(
    specs: list[Spec], raw: np.ndarray, gru: np.ndarray, source: np.ndarray, y: np.ndarray,
    terminal: np.ndarray, groups: np.ndarray, maxiter: int, selection_mode: str,
) -> tuple[Spec, list[dict[str, Any]]]:
    """Select only from common-train labels; fixed validation remains untouched."""
    folds = np.empty(len(y), dtype=np.int8)
    for number, (_, held) in enumerate(GroupKFold(n_splits=5).split(np.zeros(len(y)), y, groups)):
        folds[held] = number
    rows: list[dict[str, Any]] = []
    for spec in specs:
        probability = np.zeros((len(y), N_CLASSES), dtype=np.float32)
        fold_scores: list[dict[str, float | int]] = []
        for held_fold in range(5):
            fit_rows = terminal & (folds != held_fold)
            held = folds == held_fold
            fitted = BucketAlphaCalibrator(spec, maxiter).fit(
                raw[fit_rows], gru[fit_rows], source[fit_rows], y[fit_rows]
            )
            probability[held] = fitted.predict(raw[held], gru[held], source[held])
            prediction_held = probability[held].argmax(axis=1)
            fold_scores.append({
                "fold": int(held_fold + 1),
                "terminal_fallback_macro_f1": macro(y[held], prediction_held, terminal[held]),
                "all_fallback_macro_f1": macro(y[held], prediction_held),
            })
        prediction = probability.argmax(axis=1)
        rows.append({
            "spec": spec,
            "terminal_fallback_macro_f1": macro(y, prediction, terminal),
            "all_fallback_macro_f1": macro(y, prediction),
            "fold_scores": fold_scores,
        })
    baseline_row = next(row for row in rows if row["spec"].baseline)
    baseline_scores = {int(row["fold"]): row for row in baseline_row["fold_scores"]}
    for row in rows:
        deltas_for_row = [
            (
                float(score["terminal_fallback_macro_f1"] - baseline_scores[int(score["fold"])]["terminal_fallback_macro_f1"]),
                float(score["all_fallback_macro_f1"] - baseline_scores[int(score["fold"])]["all_fallback_macro_f1"]),
            )
            for score in row["fold_scores"]
        ]
        row["min_inner_terminal_delta_vs_b43"] = min(value[0] for value in deltas_for_row)
        row["min_inner_all_delta_vs_b43"] = min(value[1] for value in deltas_for_row)
    chosen = choose_row(rows, selection_mode)
    return chosen["spec"], rows


def bootstrap_terminal_delta(
    y: np.ndarray, baseline: np.ndarray, candidate: np.ndarray, groups: np.ndarray, replicates: int = 2500,
) -> dict[str, float | int]:
    """Whole-session bootstrap, intentionally executed only after a full gate pass."""
    unique, inverse = np.unique(groups, return_inverse=True)
    positions = [np.flatnonzero(inverse == number) for number in range(len(unique))]
    rng = np.random.default_rng(619)
    delta = np.empty(replicates, dtype=np.float64)
    for number in range(replicates):
        chosen = rng.integers(0, len(unique), size=len(unique))
        index = np.concatenate([positions[int(value)] for value in chosen])
        delta[number] = macro(y[index], candidate[index]) - macro(y[index], baseline[index])
    return {
        "replicates": replicates,
        "seed": 619,
        "mean_delta": float(delta.mean()),
        "p_delta_gt_zero": float(np.mean(delta > 0.0)),
        "p05_delta": float(np.quantile(delta, 0.05)),
        "p95_delta": float(np.quantile(delta, 0.95)),
    }


def load_b43_exact(root: Path) -> dict[str, Any]:
    samples = [
        json.loads(line) for line in (root / "data/train.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    labels = {
        row["id"]: row["action"]
        for row in csv.DictReader((root / "data/train_labels.csv").open(encoding="utf-8"))
    }
    split = {
        row["id"]: row["split"]
        for row in csv.DictReader((root / "splits/common_split.csv").open(encoding="utf-8"))
    }
    ids = np.asarray([str(row["id"]) for row in samples])
    y = np.asarray([CLASSES.index(labels[sample_id]) for sample_id in ids], dtype=np.int64)
    source = source_for_ids(ids)
    groups = np.asarray([session_of(sample_id) for sample_id in ids])
    terminal = terminal_for_ids(ids)
    folds = np.empty(len(y), dtype=np.int8)
    for number, (_, held) in enumerate(GroupKFold(n_splits=5).split(np.zeros(len(y)), y, groups)):
        folds[held] = number

    # Exact B43 full OOF components.
    raw = np.load(root / "experiments/xgb_robust_ensemble/full_oof_raw_seed42.npy", allow_pickle=False).astype(np.float32)
    gru = np.load(root / "experiments/modeling_ai_sequence_fusion_noe5/oof_probabilities.npy", allow_pickle=False).astype(np.float32)
    if raw.shape != gru.shape or raw.shape != (len(y), N_CLASSES):
        raise ValueError("Full B43 raw/old-GRU component shape mismatch")

    rule = np.load(root / "experiments/terminal_rule_combinations/oof_candidates.npz", allow_pickle=False)
    if not (
        np.array_equal(rule["y"].astype(np.int64), y)
        and np.array_equal(rule["terminal"].astype(bool), terminal)
        and np.array_equal(rule["fold"].astype(np.int8)[terminal], folds[terminal])
    ):
        raise ValueError("B43 char20 full OOF artifact does not align")
    char20 = rule["char20"].astype(np.int16)

    train_index = np.flatnonzero(np.asarray([split[sample_id] == "train" for sample_id in ids]))
    fixed_index = np.flatnonzero(np.asarray([split[sample_id] == "val" for sample_id in ids]))
    train_ids = ids[train_index]
    fixed_ids = ids[fixed_index]
    old_dir = root / "experiments/b42_9_common_old_noe5_seed42"
    raw_dir = root / "experiments/b42_9_common_raw_seed42"
    raw_common = np.load(raw_dir / "common_train_raw_xgb_oof.npy", allow_pickle=False).astype(np.float32)
    raw_common_ids = np.load(raw_dir / "common_train_ids.npy", allow_pickle=False).astype(str)
    old_common = np.load(old_dir / "common_train_oof_probabilities.npy", allow_pickle=False).astype(np.float32)
    old_common_ids = np.load(old_dir / "common_train_ids.npy", allow_pickle=False).astype(str)
    raw_fixed = np.load(root / "experiments/xgb_robust_ensemble/strict_fixed_raw_seed42.npy", allow_pickle=False).astype(np.float32)
    old_fixed = np.load(old_dir / "fixed_valid_probabilities.npy", allow_pickle=False).astype(np.float32)
    old_fixed_ids = np.load(old_dir / "fixed_valid_ids.npy", allow_pickle=False).astype(str)
    if not (
        np.array_equal(raw_common_ids, train_ids)
        and np.array_equal(old_common_ids, train_ids)
        and np.array_equal(old_fixed_ids, fixed_ids)
        and raw_common.shape == old_common.shape == (len(train_index), N_CLASSES)
        and raw_fixed.shape == old_fixed.shape == (len(fixed_index), N_CLASSES)
    ):
        raise ValueError("Matching B43 seed42 common/fixed artifacts do not align")
    # The dedicated B42_9 common artifact is authoritative here.  A later
    # fixed-noE5 compatibility artifact can differ because its fold/refit
    # procedure is not the exact B43 seed-42 common reconstruction.
    fixed_terminal = terminal[fixed_index]
    fixed_char20 = np.full(len(fixed_index), -1, dtype=np.int16)
    if not (
        np.array_equal(rule["valid_ids"].astype(str), fixed_ids[fixed_terminal])
        and np.array_equal(rule["fixed_y"].astype(np.int64), y[fixed_index][fixed_terminal])
        and len(rule["fixed_char20"]) == int(fixed_terminal.sum())
    ):
        raise ValueError("B43 char20 fixed artifact does not align")
    fixed_char20[fixed_terminal] = rule["fixed_char20"].astype(np.int16)

    return {
        "ids": ids,
        "y": y,
        "source": source,
        "groups": groups,
        "terminal": terminal,
        "folds": folds,
        "raw": raw,
        "gru": gru,
        "char20": char20,
        "train_index": train_index,
        "fixed_index": fixed_index,
        "raw_common": raw_common,
        "old_common": old_common,
        "raw_fixed": raw_fixed,
        "old_fixed": old_fixed,
        "fixed_terminal": fixed_terminal,
        "fixed_char20": fixed_char20,
    }


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    out = root / args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    data = load_b43_exact(root)
    y = data["y"]
    source = data["source"]
    terminal = data["terminal"]
    fixed_index = data["fixed_index"]
    fixed_y = y[fixed_index]
    fixed_source = source[fixed_index]

    baseline_full_probability = scalar_geometric(data["raw"], data["gru"], source)
    baseline_fixed_probability = scalar_geometric(data["raw_fixed"], data["old_fixed"], fixed_source)
    baseline_full = metric_row(y, baseline_full_probability, terminal, data["char20"])
    baseline_fixed = metric_row(
        fixed_y, baseline_fixed_probability, data["fixed_terminal"], data["fixed_char20"],
    )
    # These values are the B43-char20 provenance check.  Failing loudly is
    # preferable to silently calibrating a different historical archive.
    expected_full = {
        "all_fallback_macro_f1": 0.7603385811033483,
        "terminal_fallback_macro_f1": 0.7312748947080282,
        "terminal_with_char20_macro_f1": 0.7335225252513891,
        "hybrid_future_history_plus_char20_macro_f1": 0.98277826622222,
    }
    expected_fixed = {
        # Matching seed-42 common-train reconstruction.  This is distinct
        # from an older broad fixed proxy used by public-feedback diagnostics.
        "all_fallback_macro_f1": 0.7512384152823361,
        "terminal_fallback_macro_f1": 0.6982614924884087,
        "terminal_with_char20_macro_f1": 0.7012756083675601,
        "hybrid_future_history_plus_char20_macro_f1": 0.985722172114114,
    }
    for name, expected in expected_full.items():
        if abs(baseline_full[name] - expected) > 1.0e-10:
            raise ValueError(f"B43 full baseline mismatch for {name}: {baseline_full[name]} != {expected}")
    for name, expected in expected_fixed.items():
        if abs(baseline_fixed[name] - expected) > 1.0e-10:
            raise ValueError(f"B43 fixed baseline mismatch for {name}: {baseline_fixed[name]} != {expected}")

    specs = all_specs()
    print(json.dumps({"stage": "full_nested", "spec_count": len(specs)}, ensure_ascii=False), flush=True)
    full_probability, outer_records = nested_full_probability(
        specs, data["raw"], data["gru"], source, y, terminal, data["folds"], args.maxiter, args.selection_mode,
    )
    print(json.dumps({"stage": "fixed_common_selection"}, ensure_ascii=False), flush=True)
    common_spec, common_table = select_common_train(
        specs,
        data["raw_common"],
        data["old_common"],
        source[data["train_index"]],
        y[data["train_index"]],
        terminal[data["train_index"]],
        data["groups"][data["train_index"]],
        args.maxiter,
        args.selection_mode,
    )
    common_fit = terminal[data["train_index"]]
    fixed_calibrator = BucketAlphaCalibrator(common_spec, args.maxiter).fit(
        data["raw_common"][common_fit],
        data["old_common"][common_fit],
        source[data["train_index"]][common_fit],
        y[data["train_index"]][common_fit],
    )
    fixed_probability = fixed_calibrator.predict(data["raw_fixed"], data["old_fixed"], fixed_source)

    candidate_full = metric_row(y, full_probability, terminal, data["char20"])
    candidate_fixed = metric_row(fixed_y, fixed_probability, data["fixed_terminal"], data["fixed_char20"])
    full_delta = deltas(candidate_full, baseline_full)
    fixed_delta = deltas(candidate_fixed, baseline_fixed)
    required_metrics = tuple(baseline_full)
    passed = all(full_delta[name] > 0.0 and fixed_delta[name] > 0.0 for name in required_metrics)

    bootstrap: dict[str, Any] | None = None
    portable_path: Path | None = None
    if passed:
        # This is intentionally conditional: avoid treating a bootstrap as
        # evidence for a candidate that already failed the deterministic gate.
        full_base_terminal = with_char20(
            baseline_full_probability.argmax(axis=1).astype(np.int16), terminal, data["char20"],
        )[terminal]
        full_candidate_terminal = with_char20(
            full_probability.argmax(axis=1).astype(np.int16), terminal, data["char20"],
        )[terminal]
        fixed_base_terminal = with_char20(
            baseline_fixed_probability.argmax(axis=1).astype(np.int16), data["fixed_terminal"], data["fixed_char20"],
        )[data["fixed_terminal"]]
        fixed_candidate_terminal = with_char20(
            fixed_probability.argmax(axis=1).astype(np.int16), data["fixed_terminal"], data["fixed_char20"],
        )[data["fixed_terminal"]]
        bootstrap = {
            "full_terminal_char20": bootstrap_terminal_delta(
                y[terminal], full_base_terminal, full_candidate_terminal, data["groups"][terminal],
            ),
            "fixed_terminal_char20": bootstrap_terminal_delta(
                fixed_y[data["fixed_terminal"]], fixed_base_terminal, fixed_candidate_terminal,
                data["groups"][fixed_index][data["fixed_terminal"]],
            ),
        }
        full_fit = BucketAlphaCalibrator(common_spec, args.maxiter).fit(
            data["raw"][terminal], data["gru"][terminal], source[terminal], y[terminal],
        )
        portable_path = out / "recommended_portable_parameters.json"
        portable_path.write_text(
            json.dumps(full_fit.portable_payload(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
        )

    report: dict[str, Any] = {
        "scope": "Experiment-only B43 source-by-class alpha audit. model/, script.py, B42.zip, and baseline_submit.zip are never opened for modification.",
        "base_contract": {
            "raw": "experiments/xgb_robust_ensemble/full_oof_raw_seed42.npy and matching strict_fixed_raw_seed42.npy",
            "gru": "original no-E5 GRU, including b42_9_common_old_noe5_seed42 fixed/common artifacts",
            "blend": "geometric raw/GRU; B43 scalar baseline SIM=.20 / AU=.85",
            "char20": "immutable downstream terminal override; never a calibrator input or nested selection metric",
            "candidate": "Per-source classes sorted from training-only true-label raw-vs-GRU contrast; 2--4 equal-count buckets each receive one L2-anchored alpha shift.",
        },
        "validation_protocol": {
            "full": "5 outer whole-session GroupKFold folds; each held fold is predicted by a spec selected/fitted only from the other four folds.",
            "fixed": "Spec selection and fitting use only matching seed42 common-train OOF components; fixed validation labels are evaluated once after selection.",
            "selection": (
                "unoverridden terminal fallback macro-F1, then all fallback macro-F1, then lower capacity. "
                f"Mode={args.selection_mode}; Char20 is not used in selection."
            ),
            "gate": "All four B43 metrics must be strictly positive versus exact B43 on both full nested and matching fixed protocols.",
        },
        "counts": {
            "full_rows": int(len(y)),
            "full_terminal": int(terminal.sum()),
            "common_train_rows": int(len(data["train_index"])),
            "common_train_terminal": int(terminal[data["train_index"]].sum()),
            "fixed_rows": int(len(fixed_index)),
            "fixed_terminal": int(data["fixed_terminal"].sum()),
            "spec_count": len(specs),
        },
        "b43_baseline": {"full_nested_group_oof": baseline_full, "matching_seed42_fixed": baseline_fixed},
        "candidate": {
            "full_nested_group_oof": candidate_full,
            "matching_seed42_fixed": candidate_fixed,
            "full_delta": full_delta,
            "fixed_delta": fixed_delta,
            "outer_selection": outer_records,
            "common_train_selection": {
                "chosen": common_spec.key,
                "candidates": [
                    {
                        "spec": row["spec"].key,
                        "terminal_fallback_macro_f1": row["terminal_fallback_macro_f1"],
                        "all_fallback_macro_f1": row["all_fallback_macro_f1"],
                        "min_inner_terminal_delta_vs_b43": row["min_inner_terminal_delta_vs_b43"],
                        "min_inner_all_delta_vs_b43": row["min_inner_all_delta_vs_b43"],
                    }
                    for row in common_table
                ],
            },
        },
        "promotion_gate": {
            "required_metrics": list(required_metrics),
            "pass": passed,
            "reason": (
                "Every all/terminal/char20/hybrid metric improved in both independent B43 protocols."
                if passed else
                "At least one required B43 metric is non-positive; do not modify any submission bundle."
            ),
            "bootstrap": bootstrap if bootstrap is not None else "not run because the deterministic all-four-metric gate failed",
            "portable_parameters": str(portable_path.relative_to(root)) if portable_path is not None else None,
        },
    }
    (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    np.savez_compressed(
        out / "predictions.npz",
        y=y,
        terminal=terminal,
        groups=data["groups"],
        baseline_full_probability=baseline_full_probability,
        candidate_full_probability=full_probability,
        fixed_index=fixed_index,
        fixed_terminal=data["fixed_terminal"],
        baseline_fixed_probability=baseline_fixed_probability,
        candidate_fixed_probability=fixed_probability,
        char20=data["char20"],
        fixed_char20=data["fixed_char20"],
    )
    print(
        json.dumps(
            {
                "pass": passed,
                "common_spec": common_spec.key,
                "full_delta": full_delta,
                "fixed_delta": fixed_delta,
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
