#!/usr/bin/env python3
"""Exact B43 adaptive-gate and 50:50 dual-GRU experiment audit.

The program only writes to experiments/b43_adaptive_blend. It never writes
model/, script.py, or a submission archive. B43 means raw XGB seed42 plus
the old no-E5 GRU, geometric fusion at SIM=.20/AU=.85, then char20 last.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold


CLASSES = (
    "read_file", "grep_search", "list_directory", "glob_pattern",
    "edit_file", "write_file", "apply_patch", "run_bash", "run_tests",
    "lint_or_typecheck", "ask_user", "plan_task", "web_search",
    "respond_only",
)
N_CLASSES = len(CLASSES)
EPS = 1.0e-12
BOOTSTRAP_SEED = 619
BOOTSTRAP_REPLICATES = 2000
PROMOTION_BOOTSTRAP_P = 0.95
STATIC_STRICT_WEIGHTS = (0.25, 0.35, 0.50)
METRIC_KEYS = (
    "all_fallback_macro_f1",
    "terminal_fallback_macro_f1",
    "terminal_with_char20_macro_f1",
    "hybrid_future_history_plus_char20_macro_f1",
)

# A small, fixed policy set. Every non-base rule alters alpha only if the raw
# and old-GRU top classes disagree; there is no text, target, ID, or history
# feature and no fitted gating model.
POLICIES = (
    "base",
    "sim_raw_margin_up_005",
    "sim_balanced_margin_005",
    "sim_two_sided_margin_005",
    "au_two_sided_margin_005",
)
POLICY_DESCRIPTION = {
    "base": "B43 geometric SIM=.20 and AU=.85.",
    "sim_raw_margin_up_005": (
        "SIM disagreement only: raw top-margin exceeds old GRU margin by at "
        "least .05, so raw alpha becomes .30."
    ),
    "sim_balanced_margin_005": (
        "SIM raw-margin advantage >=.05 gives alpha .25; AU old-GRU-margin "
        "advantage >=.05 gives alpha .80."
    ),
    "sim_two_sided_margin_005": (
        "SIM raw-margin advantage >=.05 gives .30, old-GRU-margin advantage "
        ">=.05 gives .10."
    ),
    "au_two_sided_margin_005": (
        "AU raw-margin advantage >=.05 gives .90, old-GRU-margin advantage "
        ">=.05 gives .80."
    ),
}

EXPECTED_B43 = {
    "full": {
        "all_fallback_macro_f1": 0.7603385811033483,
        "terminal_fallback_macro_f1": 0.7312748947080282,
        "terminal_with_char20_macro_f1": 0.7335225252513891,
        "hybrid_future_history_plus_char20_macro_f1": 0.98277826622222,
    },
    "fixed": {
        "all_fallback_macro_f1": 0.7512384152823361,
        "terminal_fallback_macro_f1": 0.6982614924884087,
        "terminal_with_char20_macro_f1": 0.7012756083675601,
        "hybrid_future_history_plus_char20_macro_f1": 0.985722172114114,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="experiments/b43_adaptive_blend")
    parser.add_argument("--bootstrap-replicates", type=int, default=BOOTSTRAP_REPLICATES)
    return parser.parse_args()


def session_of(sample_id: str) -> str:
    return sample_id.rsplit("-step_", 1)[0]


def build_terminal(ids: list[str]) -> np.ndarray:
    maximum: dict[str, int] = {}
    parsed: list[tuple[str, int]] = []
    for sample_id in ids:
        session, raw_step = sample_id.rsplit("-step_", 1)
        step = int(raw_step)
        maximum[session] = max(maximum.get(session, -1), step)
        parsed.append((session, step))
    return np.asarray([step == maximum[session] for session, step in parsed], dtype=bool)


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def macro(y: np.ndarray, prediction: np.ndarray) -> float:
    return float(f1_score(
        y, prediction, labels=np.arange(N_CLASSES),
        average="macro", zero_division=0,
    ))


def source_alpha(source: np.ndarray) -> np.ndarray:
    if not np.all(np.isin(source, ("sim", "au"))):
        raise ValueError(f"Unknown source: {np.unique(source).tolist()}")
    return np.where(source == "sim", 0.20, 0.85).astype(np.float64)


def geometric(raw: np.ndarray, gru: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    logits = (
        alpha[:, None] * np.log(np.maximum(raw, EPS))
        + (1.0 - alpha[:, None]) * np.log(np.maximum(gru, EPS))
    )
    logits -= logits.max(axis=1, keepdims=True)
    result = np.exp(logits)
    return (result / result.sum(axis=1, keepdims=True)).astype(np.float32)


def margin(probability: np.ndarray) -> np.ndarray:
    ordered = np.sort(probability, axis=1)
    return ordered[:, -1] - ordered[:, -2]


def adaptive_alpha(raw: np.ndarray, old: np.ndarray, source: np.ndarray, policy: str) -> np.ndarray:
    if policy not in POLICIES:
        raise ValueError(policy)
    alpha = source_alpha(source)
    if policy == "base":
        return alpha
    disagreement = raw.argmax(axis=1) != old.argmax(axis=1)
    delta_margin = margin(raw) - margin(old)
    raw_advantage = disagreement & (delta_margin >= 0.05)
    old_advantage = disagreement & (delta_margin <= -0.05)
    sim = source == "sim"
    au = source == "au"
    if policy == "sim_raw_margin_up_005":
        alpha[sim & raw_advantage] = 0.30
    elif policy == "sim_balanced_margin_005":
        alpha[sim & raw_advantage] = 0.25
        alpha[au & old_advantage] = 0.80
    elif policy == "sim_two_sided_margin_005":
        alpha[sim & raw_advantage] = 0.30
        alpha[sim & old_advantage] = 0.10
    elif policy == "au_two_sided_margin_005":
        alpha[au & raw_advantage] = 0.90
        alpha[au & old_advantage] = 0.80
    return alpha


def labels_with_char(
    probability: np.ndarray, terminal: np.ndarray, char20: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    fallback = probability.argmax(axis=1).astype(np.int16)
    final = fallback.copy()
    override = terminal & (char20 >= 0)
    final[override] = char20[override]
    return fallback, final


def metric_bundle(
    y: np.ndarray, probability: np.ndarray, terminal: np.ndarray, char20: np.ndarray,
) -> dict[str, float]:
    fallback, final = labels_with_char(probability, terminal, char20)
    hybrid = final.copy()
    hybrid[~terminal] = y[~terminal]
    return {
        "all_fallback_macro_f1": macro(y, fallback),
        "terminal_fallback_macro_f1": macro(y[terminal], fallback[terminal]),
        "terminal_with_char20_macro_f1": macro(y[terminal], final[terminal]),
        "hybrid_future_history_plus_char20_macro_f1": macro(y, hybrid),
    }


def metric_delta(candidate: dict[str, float], baseline: dict[str, float]) -> dict[str, float]:
    return {key: float(candidate[key] - baseline[key]) for key in METRIC_KEYS}


def check_baseline(actual: dict[str, float], expected: dict[str, float], name: str) -> None:
    mismatch = {
        key: (actual[key], expected[key])
        for key in METRIC_KEYS
        if abs(actual[key] - expected[key]) > 1.0e-12
    }
    if mismatch:
        raise ValueError(f"{name} B43 baseline mismatch: {mismatch}")


def weighted_macro(confusion: np.ndarray) -> float:
    tp = np.diag(confusion)
    fp = confusion.sum(axis=0) - tp
    fn = confusion.sum(axis=1) - tp
    denominator = 2.0 * tp + fp + fn
    score = np.divide(
        2.0 * tp, denominator,
        out=np.zeros(N_CLASSES, dtype=np.float64), where=denominator > 0.0,
    )
    return float(score.mean())


def session_bootstrap(
    y: np.ndarray,
    baseline: np.ndarray,
    candidate: np.ndarray,
    groups: np.ndarray,
    replicates: int,
) -> dict[str, float | int]:
    """Bootstrap whole sessions after the terminal char20 overlay."""
    unique, inverse = np.unique(groups, return_inverse=True)
    baseline_cell = y.astype(np.int64) * N_CLASSES + baseline.astype(np.int64)
    candidate_cell = y.astype(np.int64) * N_CLASSES + candidate.astype(np.int64)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    delta = np.empty(replicates, dtype=np.float64)
    for number in range(replicates):
        sampled = rng.integers(0, len(unique), size=len(unique))
        count = np.bincount(sampled, minlength=len(unique))
        weight = count[inverse]
        base_confusion = np.bincount(
            baseline_cell, weights=weight, minlength=N_CLASSES * N_CLASSES,
        ).reshape(N_CLASSES, N_CLASSES)
        candidate_confusion = np.bincount(
            candidate_cell, weights=weight, minlength=N_CLASSES * N_CLASSES,
        ).reshape(N_CLASSES, N_CLASSES)
        delta[number] = weighted_macro(candidate_confusion) - weighted_macro(base_confusion)
    return {
        "replicates": int(replicates),
        "seed": BOOTSTRAP_SEED,
        "sessions": int(len(unique)),
        "mean_delta": float(delta.mean()),
        "p_delta_gt_zero": float(np.mean(delta > 0.0)),
        "p05_delta": float(np.quantile(delta, 0.05)),
        "p95_delta": float(np.quantile(delta, 0.95)),
    }


def fold_deltas(
    y: np.ndarray,
    baseline: np.ndarray,
    candidate: np.ndarray,
    terminal: np.ndarray,
    char20: np.ndarray,
    fold: np.ndarray,
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for held in sorted(np.unique(fold).tolist()):
        mask = fold == held
        base_metric = metric_bundle(y[mask], baseline[mask], terminal[mask], char20[mask])
        candidate_metric = metric_bundle(y[mask], candidate[mask], terminal[mask], char20[mask])
        result.append({
            "fold": int(held) + 1,
            "rows": int(mask.sum()),
            "terminal_rows": int(terminal[mask].sum()),
            "baseline": base_metric,
            "candidate": candidate_metric,
            "delta": metric_delta(candidate_metric, base_metric),
        })
    return result


def positive_fold_count(rows: list[dict[str, Any]], metric: str) -> int:
    return int(sum(row["delta"][metric] > 0.0 for row in rows))


def choose_policy(
    y: np.ndarray,
    probability: dict[str, np.ndarray],
    terminal: np.ndarray,
    char20: np.ndarray,
    train: np.ndarray,
) -> tuple[str, dict[str, dict[str, float]]]:
    metrics = {
        name: metric_bundle(y[train], value[train], terminal[train], char20[train])
        for name, value in probability.items()
    }
    selected = max(
        POLICIES,
        key=lambda name: (
            metrics[name]["terminal_with_char20_macro_f1"],
            metrics[name]["all_fallback_macro_f1"],
            -POLICIES.index(name),
        ),
    )
    return selected, metrics


def modal_policy(names: list[str]) -> tuple[str, dict[str, int]]:
    count = Counter(names)
    maximum = max(count.values())
    selected = next(name for name in POLICIES if count.get(name, 0) == maximum)
    return selected, {name: int(count.get(name, 0)) for name in POLICIES}


def nested_adaptive(
    y: np.ndarray,
    full_probability: dict[str, np.ndarray],
    fixed_probability: dict[str, np.ndarray],
    terminal: np.ndarray,
    char20: np.ndarray,
    fixed_y: np.ndarray,
    fixed_terminal: np.ndarray,
    fixed_char20: np.ndarray,
    fold: np.ndarray,
) -> dict[str, Any]:
    selected_probability = np.zeros_like(full_probability["base"])
    selected_names: list[str] = []
    choices: list[dict[str, Any]] = []
    for held in sorted(np.unique(fold).tolist()):
        train = fold != held
        selected, train_metrics = choose_policy(
            y, full_probability, terminal, char20, train,
        )
        held_mask = ~train
        selected_probability[held_mask] = full_probability[selected][held_mask]
        selected_names.append(selected)
        choices.append({
            "held_outer_session_fold": int(held) + 1,
            "selected_without_held_labels": selected,
            "other_four_fold_metrics": train_metrics,
            "held_metrics": metric_bundle(
                y[held_mask], full_probability[selected][held_mask],
                terminal[held_mask], char20[held_mask],
            ),
        })
    consensus, counts = modal_policy(selected_names)
    return {
        "selection_rule": (
            "Each held GroupKFold session fold selects from the fixed five-policy "
            "set on only the other four folds by terminal-char20 F1. "
            "All-fallback F1 and fixed policy order break ties. Fixed labels are unused."
        ),
        "outer_choices": choices,
        "selection_counts": counts,
        "fixed_consensus_policy": consensus,
        "full_outer_selected_probability": selected_probability,
        "full_outer_selected_metrics": metric_bundle(
            y, selected_probability, terminal, char20,
        ),
        "full_consensus_metrics_descriptive": metric_bundle(
            y, full_probability[consensus], terminal, char20,
        ),
        "fixed_consensus_probability": fixed_probability[consensus],
        "fixed_consensus_metrics": metric_bundle(
            fixed_y, fixed_probability[consensus], fixed_terminal, fixed_char20,
        ),
    }


def promotion_gate(
    full_delta: dict[str, float],
    fixed_delta: dict[str, float],
    full_bootstrap: dict[str, float | int],
    fixed_bootstrap: dict[str, float | int],
    folds: list[dict[str, Any]],
    consensus_nonbase: bool = True,
) -> dict[str, Any]:
    checks = {
        "full_all_four_metrics_positive": all(full_delta[key] > 0.0 for key in METRIC_KEYS),
        "fixed_all_four_metrics_positive": all(fixed_delta[key] > 0.0 for key in METRIC_KEYS),
        "full_terminal_bootstrap_p_ge_0_95": (
            float(full_bootstrap["p_delta_gt_zero"]) >= PROMOTION_BOOTSTRAP_P
        ),
        "fixed_terminal_bootstrap_p_ge_0_95": (
            float(fixed_bootstrap["p_delta_gt_zero"]) >= PROMOTION_BOOTSTRAP_P
        ),
        "full_all_fallback_positive_in_at_least_4_of_5_folds": (
            positive_fold_count(folds, "all_fallback_macro_f1") >= 4
        ),
        "full_terminal_char20_positive_in_at_least_4_of_5_folds": (
            positive_fold_count(folds, "terminal_with_char20_macro_f1") >= 4
        ),
        "nonbase_deployment_policy": consensus_nonbase,
    }
    return {
        "thresholds": {
            "bootstrap_p_delta_gt_zero": PROMOTION_BOOTSTRAP_P,
            "positive_outer_session_folds": 4,
            "all_metrics_must_improve": list(METRIC_KEYS),
        },
        "checks": checks,
        "promote": bool(all(checks.values())),
    }


def portable_dual_contract(root: Path) -> dict[str, Any]:
    """Verify the two portable GRU artifacts and time their CPU-only pair."""

    from numpy_sequence_fusion import load_weights, probabilities

    old_dir = root / "experiments/modeling_ai_sequence_fusion_noe5_full"
    strict_dir = root / "experiments/modeling_ai_sequence_fusion_noe5_strictdistill035_full"
    source_path = root / "src/numpy_sequence_fusion.py"
    old_architecture_path = old_dir / "architecture.json"
    strict_architecture_path = strict_dir / "architecture.json"
    old_weights_path = old_dir / "weights.npz"
    strict_weights_path = strict_dir / "weights.npz"
    old_probe_path = old_dir / "verification_probe.npz"
    strict_probe_path = strict_dir / "verification_probe.npz"
    old_architecture = json.loads(old_architecture_path.read_text(encoding="utf-8"))
    strict_architecture = json.loads(strict_architecture_path.read_text(encoding="utf-8"))
    compatible = (
        old_architecture["format"] == strict_architecture["format"]
        and old_architecture["all_classes"] == strict_architecture["all_classes"] == list(CLASSES)
        and old_architecture["input"] == strict_architecture["input"]
    )
    if not compatible:
        raise ValueError("Old and strict portable GRU deployment schemas differ")

    probe = np.load(old_probe_path, allow_pickle=False)
    expected_probe_keys = ("action", "status", "arg", "base", "meta", "torch_probabilities")
    if tuple(probe.files) != expected_probe_keys:
        raise ValueError(f"Unexpected old GRU verification probe fields: {probe.files}")
    strict_probe = np.load(strict_probe_path, allow_pickle=False)
    if tuple(strict_probe.files) != expected_probe_keys:
        raise ValueError(f"Unexpected strict GRU verification probe fields: {strict_probe.files}")
    old_weights = load_weights(old_weights_path)
    strict_weights = load_weights(strict_weights_path)

    def verify_probe(archive: Any, weights: dict[str, np.ndarray]) -> dict[str, Any]:
        inputs = [archive[name] for name in ("action", "status", "arg", "base", "meta")]
        reproduced = probabilities(*inputs, weights)
        reference = archive["torch_probabilities"]
        return {
            "rows": int(len(reproduced)),
            "max_abs_probability_difference": float(np.max(np.abs(reproduced - reference))),
            "argmax_agreement": float(np.mean(
                reproduced.argmax(axis=1) == reference.argmax(axis=1),
            )),
        }

    old_verification = verify_probe(probe, old_weights)
    strict_verification = verify_probe(strict_probe, strict_weights)

    timings: dict[str, Any] = {}
    for rows in (256, 12000):
        repeat = (rows + len(probe["action"]) - 1) // len(probe["action"])
        inputs = [
            np.concatenate([probe[name]] * repeat, axis=0)[:rows]
            for name in ("action", "status", "arg", "base", "meta")
        ]
        # Warm the vectorized NumPy/SciPy operations before timing the pair.
        probabilities(*inputs, old_weights)
        probabilities(*inputs, strict_weights)
        samples: list[float] = []
        probability_sum_error = 0.0
        for _ in range(3):
            started = time.perf_counter()
            old_probability = probabilities(*inputs, old_weights)
            strict_probability = probabilities(*inputs, strict_weights)
            mixture = (0.5 * old_probability + 0.5 * strict_probability).astype(np.float32)
            samples.append(time.perf_counter() - started)
            probability_sum_error = max(
                probability_sum_error,
                float(np.max(np.abs(mixture.sum(axis=1) - 1.0))),
            )
        timings[str(rows)] = {
            "three_pair_forward_seconds": [float(value) for value in samples],
            "median_pair_forward_seconds": float(np.median(samples)),
            "mixture_probability_sum_max_abs_error": probability_sum_error,
        }

    def relative(path: Path) -> str:
        return str(path.relative_to(root)).replace("\\", "/")

    artifacts = {
        "old_weights": old_weights_path,
        "old_architecture": old_architecture_path,
        "old_verification_probe": old_probe_path,
        "strict_weights": strict_weights_path,
        "strict_architecture": strict_architecture_path,
        "strict_verification_probe": strict_probe_path,
        "numpy_cpu_forward": source_path,
    }
    return {
        "inference_definition": (
            "Run the old portable no-E5 NumPy GRU and strict-distill portable "
            "no-E5 NumPy GRU on identical tensors, then calculate "
            "(1-strict_weight)*old_probability + strict_weight*strict_probability "
            "before the unchanged B43 geometric raw/GRU fusion."
        ),
        "full_refit_artifacts": {
            name: {
                "path": relative(path),
                "sha256": sha256(path),
                "bytes": int(path.stat().st_size),
            }
            for name, path in artifacts.items()
        },
        "schema_compatibility": {
            "format": old_architecture["format"],
            "class_order_matches": True,
            "input_schema_matches": True,
            "old_runtime_verification": old_verification,
            "strict_runtime_verification": strict_verification,
        },
        "cpu_feasibility": {
            "measurement": (
                "Warm CPU-only NumPy/SciPy pair-forward timing on this desktop; "
                "three repetitions per synthetic batch. This excludes JSON parsing, "
                "feature construction, raw XGB, and terminal-rule work."
            ),
            "pair_forward": timings,
        },
        "deployment_status": (
            "The pair is portable in principle, but no pair artifact is built or "
            "inserted into the live model until the statistical promotion gate passes."
        ),
    }


def recommend_static_weight(records: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Choose only among the requested fixed arithmetic weights.

    The ranking first requires both protocols to improve all four reported
    metrics. It then prefers the smaller of the two session-bootstrap
    probabilities, followed by aggregate terminal-char20 improvement. This
    does not relax the separate promotion gate.
    """

    eligible = [
        record for record in records.values()
        if all(
            value > 0.0
            for side in record["delta_vs_b43"].values()
            for value in side.values()
        )
    ]
    pool = eligible if eligible else list(records.values())
    chosen = max(
        pool,
        key=lambda record: (
            min(
                float(record["terminal_char20_session_bootstrap"]["full_outer_oof"]["p_delta_gt_zero"]),
                float(record["terminal_char20_session_bootstrap"]["matching_seed42_fixed"]["p_delta_gt_zero"]),
            ),
            record["delta_vs_b43"]["full_outer_oof"]["terminal_with_char20_macro_f1"]
            + record["delta_vs_b43"]["matching_seed42_fixed"]["terminal_with_char20_macro_f1"],
            record["strict_distill_weight"],
        ),
    )
    return {
        "recommended_strict_distill_weight": chosen["strict_distill_weight"],
        "recommended_old_noe5_weight": chosen["old_noe5_weight"],
        "eligible_weights_with_all_eight_full_fixed_deltas_positive": [
            record["strict_distill_weight"] for record in eligible
        ],
        "selection_rule": (
            "Requested fixed weights only; first require all four metrics to "
            "improve on full and matching fixed, then maximize the weaker "
            "terminal-char20 session-bootstrap probability."
        ),
        "promotion_gate_passed": chosen["promotion_gate"]["promote"],
        "reason": (
            "This is the strongest requested static point estimate under the "
            "defined two-protocol rule. It remains experiment-only unless its "
            "independent promotion gate also passes."
        ),
    }


def load_exact_b43(root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    ids = [
        str(json.loads(line)["id"])
        for line in (root / "data/train.jsonl").read_text(encoding="utf-8").splitlines()
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
    y = np.asarray([CLASSES.index(labels[item]) for item in ids], dtype=np.int64)
    source = np.asarray(["au" if item.startswith("sess_au_") else "sim" for item in ids])
    groups = np.asarray([session_of(item) for item in ids])
    terminal = build_terminal(ids)
    experiments = root / "experiments"

    raw_full_path = experiments / "xgb_robust_ensemble/full_oof_raw_seed42.npy"
    raw_proxy_path = experiments / "modeling_ak_live_xgb_proxy/oof_probabilities.npy"
    old_full_path = experiments / "modeling_ai_sequence_fusion_noe5/oof_probabilities.npy"
    strict_full_path = experiments / "modeling_ai_sequence_nested_distill_e16_i4/oof_probabilities.npy"
    strict_fold_path = experiments / "modeling_ai_sequence_nested_distill_e16_i4/outer_fold_assignment.npy"
    raw = np.load(raw_full_path, allow_pickle=False).astype(np.float32)
    raw_proxy = np.load(raw_proxy_path, allow_pickle=False).astype(np.float32)
    old = np.load(old_full_path, allow_pickle=False).astype(np.float32)
    strict = np.load(strict_full_path, allow_pickle=False).astype(np.float32)
    if not (raw.shape == raw_proxy.shape == old.shape == strict.shape == (len(y), N_CLASSES)):
        raise ValueError("B43 full component shape mismatch")
    raw_proxy_difference = float(np.max(np.abs(raw - raw_proxy)))
    if raw_proxy_difference > 1.0e-7:
        raise ValueError(f"raw proxy is not the B43 seed42 raw array: {raw_proxy_difference}")

    fold = np.empty(len(y), dtype=np.int8)
    for number, (_, held) in enumerate(GroupKFold(n_splits=5).split(np.zeros(len(y)), y, groups)):
        fold[held] = number
    if not np.array_equal(
        fold, np.load(strict_fold_path, allow_pickle=False).astype(np.int8),
    ):
        raise ValueError("strict GRU OOF fold is not the expected session GroupKFold")

    char_path = experiments / "terminal_rule_combinations/oof_candidates.npz"
    char = np.load(char_path, allow_pickle=False)
    char20 = char["char20"].astype(np.int16)
    if not (
        np.array_equal(char["y"].astype(np.int64), y)
        and np.array_equal(char["terminal"].astype(bool), terminal)
        and np.array_equal(char["fold"].astype(np.int8)[terminal], fold[terminal])
    ):
        raise ValueError("char20 OOF component does not align with B43")

    valid_index = np.asarray(
        [index for index, item in enumerate(ids) if split[item] == "val"],
        dtype=np.int64,
    )
    fixed_y = y[valid_index]
    fixed_source = source[valid_index]
    fixed_groups = groups[valid_index]
    fixed_terminal = terminal[valid_index]
    raw_fixed_path = experiments / "xgb_robust_ensemble/strict_fixed_raw_seed42.npy"
    old_fixed_path = experiments / "b42_9_common_old_noe5_seed42/fixed_valid_probabilities.npy"
    old_fixed_ids_path = experiments / "b42_9_common_old_noe5_seed42/fixed_valid_ids.npy"
    strict_fixed_path = experiments / "modeling_ai_sequence_fusion_fixed_strictdistill_w35/valid_probabilities.npy"
    strict_fixed_index_path = experiments / "modeling_ai_sequence_fusion_fixed_strictdistill_w35/valid_indices.npy"
    strict_fixed_terminal_path = experiments / "modeling_ai_sequence_fusion_fixed_strictdistill_w35/terminal_mask.npy"
    fixed_raw = np.load(raw_fixed_path, allow_pickle=False).astype(np.float32)
    fixed_old = np.load(old_fixed_path, allow_pickle=False).astype(np.float32)
    fixed_strict = np.load(strict_fixed_path, allow_pickle=False).astype(np.float32)
    fixed_old_ids = np.load(old_fixed_ids_path, allow_pickle=False).astype(str)
    fixed_strict_index = np.load(strict_fixed_index_path, allow_pickle=False).astype(np.int64)
    fixed_strict_terminal = np.load(strict_fixed_terminal_path, allow_pickle=False).astype(bool)
    if not (
        fixed_raw.shape == fixed_old.shape == fixed_strict.shape == (len(valid_index), N_CLASSES)
        and np.array_equal(fixed_old_ids, np.asarray(ids, dtype=str)[valid_index])
        and np.array_equal(fixed_strict_index, valid_index)
        and np.array_equal(fixed_strict_terminal, fixed_terminal)
    ):
        raise ValueError("matching B43 seed42 fixed component alignment failed")
    fixed_char20 = np.full(len(valid_index), -1, dtype=np.int16)
    fixed_char20[fixed_terminal] = char["fixed_char20"].astype(np.int16)
    if not (
        np.array_equal(char["fixed_y"].astype(np.int64), fixed_y[fixed_terminal])
        and np.array_equal(
            char["valid_ids"].astype(str),
            np.asarray(ids, dtype=str)[valid_index][fixed_terminal],
        )
    ):
        raise ValueError("char20 fixed component does not align with B43")

    input_paths = {
        "full_raw_seed42": raw_full_path,
        "full_raw_proxy": raw_proxy_path,
        "full_old_noe5": old_full_path,
        "full_strict_distill_noe5": strict_full_path,
        "fixed_raw_seed42": raw_fixed_path,
        "fixed_old_noe5_matching_seed42": old_fixed_path,
        "fixed_strict_distill_noe5": strict_fixed_path,
        "char20": char_path,
    }
    provenance = {
        "input_sha256": {name: sha256(path) for name, path in input_paths.items()},
        "full_raw_seed42_proxy_max_abs_difference": raw_proxy_difference,
        "full_rows": int(len(y)),
        "full_terminal_rows": int(terminal.sum()),
        "fixed_rows": int(len(fixed_y)),
        "fixed_terminal_rows": int(fixed_terminal.sum()),
        "session_outer_folds": int(len(np.unique(fold))),
    }
    return {
        "y": y, "source": source, "groups": groups, "terminal": terminal, "fold": fold,
        "raw": raw, "old": old, "strict": strict, "char20": char20,
        "fixed_y": fixed_y, "fixed_source": fixed_source, "fixed_groups": fixed_groups,
        "fixed_terminal": fixed_terminal, "fixed_raw": fixed_raw, "fixed_old": fixed_old,
        "fixed_strict": fixed_strict, "fixed_char20": fixed_char20,
    }, provenance


def write_readme(out: Path, report: dict[str, Any]) -> None:
    dual = report["dual_arithmetic_50_50"]["promotion_gate"]["promote"]
    adaptive = report["adaptive_margin_gate"]["promotion_gate"]["promote"]
    recommendation = report["dual_arithmetic_static_weight_table"]["recommendation"]
    text = "\n".join((
        "# B43 adaptive blend audit",
        "",
        "실험 전용 결과입니다. model/, script.py, B42.zip, baseline_submit.zip 및 제출 ZIP은 변경하지 않았습니다.",
        "기준은 raw seed42 + old no-E5 GRU 기하 결합(SIM=.20/AU=.85) 후 char20 terminal overlay입니다.",
        f"50:50 arithmetic dual promote: {dual}",
        f"adaptive margin gate promote: {adaptive}",
        (
            "Requested static-weight recommendation: strict="
            f"{recommendation['recommended_strict_distill_weight']:.2f}, old="
            f"{recommendation['recommended_old_noe5_weight']:.2f}; promotion="
            f"{recommendation['promotion_gate_passed']}"
        ),
        "두 경로 모두 세션 OOF, matching seed42 fixed, terminal-char20 session bootstrap 승인을 통과해야 제출 후보가 됩니다.",
    ))
    (out / "README.md").write_text(text + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.bootstrap_replicates <= 0:
        raise ValueError("bootstrap replicates must be positive")
    root = Path(__file__).resolve().parents[1]
    out = Path(args.out_dir)
    if not out.is_absolute():
        out = root / out
    out = out.resolve()
    model_dir = (root / "model").resolve()
    if out == model_dir or model_dir in out.parents:
        raise ValueError("output must not be inside model/")
    out.mkdir(parents=True, exist_ok=True)

    data, provenance = load_exact_b43(root)
    alpha = source_alpha(data["source"])
    fixed_alpha = source_alpha(data["fixed_source"])
    base_full_probability = geometric(data["raw"], data["old"], alpha)
    base_fixed_probability = geometric(data["fixed_raw"], data["fixed_old"], fixed_alpha)
    base_full = metric_bundle(data["y"], base_full_probability, data["terminal"], data["char20"])
    base_fixed = metric_bundle(
        data["fixed_y"], base_fixed_probability, data["fixed_terminal"], data["fixed_char20"],
    )
    check_baseline(base_full, EXPECTED_B43["full"], "full")
    check_baseline(base_fixed, EXPECTED_B43["fixed"], "fixed")

    base_full_terminal = labels_with_char(
        base_full_probability, data["terminal"], data["char20"],
    )[1][data["terminal"]]
    base_fixed_terminal = labels_with_char(
        base_fixed_probability, data["fixed_terminal"], data["fixed_char20"],
    )[1][data["fixed_terminal"]]
    static_dual_weights: dict[str, dict[str, Any]] = {}
    for strict_weight in STATIC_STRICT_WEIGHTS:
        old_weight = 1.0 - strict_weight
        full_gru = (old_weight * data["old"] + strict_weight * data["strict"]).astype(np.float32)
        fixed_gru = (
            old_weight * data["fixed_old"] + strict_weight * data["fixed_strict"]
        ).astype(np.float32)
        full_probability = geometric(data["raw"], full_gru, alpha)
        fixed_probability = geometric(data["fixed_raw"], fixed_gru, fixed_alpha)
        full_metric = metric_bundle(data["y"], full_probability, data["terminal"], data["char20"])
        fixed_metric = metric_bundle(
            data["fixed_y"], fixed_probability, data["fixed_terminal"], data["fixed_char20"],
        )
        folds = fold_deltas(
            data["y"], base_full_probability, full_probability,
            data["terminal"], data["char20"], data["fold"],
        )
        full_terminal = labels_with_char(
            full_probability, data["terminal"], data["char20"],
        )[1][data["terminal"]]
        fixed_terminal = labels_with_char(
            fixed_probability, data["fixed_terminal"], data["fixed_char20"],
        )[1][data["fixed_terminal"]]
        bootstrap_full = session_bootstrap(
            data["y"][data["terminal"]], base_full_terminal, full_terminal,
            data["groups"][data["terminal"]], args.bootstrap_replicates,
        )
        bootstrap_fixed = session_bootstrap(
            data["fixed_y"][data["fixed_terminal"]], base_fixed_terminal, fixed_terminal,
            data["fixed_groups"][data["fixed_terminal"]], args.bootstrap_replicates,
        )
        record = {
            "old_noe5_weight": old_weight,
            "strict_distill_weight": strict_weight,
            "full_outer_oof": full_metric,
            "matching_seed42_fixed": fixed_metric,
            "delta_vs_b43": {
                "full_outer_oof": metric_delta(full_metric, base_full),
                "matching_seed42_fixed": metric_delta(fixed_metric, base_fixed),
            },
            "full_outer_session_fold_deltas": folds,
            "terminal_char20_session_bootstrap": {
                "full_outer_oof": bootstrap_full,
                "matching_seed42_fixed": bootstrap_fixed,
            },
        }
        record["promotion_gate"] = promotion_gate(
            record["delta_vs_b43"]["full_outer_oof"],
            record["delta_vs_b43"]["matching_seed42_fixed"],
            bootstrap_full, bootstrap_fixed, folds,
        )
        static_dual_weights[f"strict_weight_{strict_weight:.2f}"] = record
    dual_50 = static_dual_weights["strict_weight_0.50"]
    static_weight_recommendation = recommend_static_weight(static_dual_weights)
    portable_contract = portable_dual_contract(root)

    policy_full: dict[str, np.ndarray] = {}
    policy_fixed: dict[str, np.ndarray] = {}
    policy_changes: dict[str, dict[str, int]] = {}
    for name in POLICIES:
        full_policy_alpha = adaptive_alpha(data["raw"], data["old"], data["source"], name)
        fixed_policy_alpha = adaptive_alpha(
            data["fixed_raw"], data["fixed_old"], data["fixed_source"], name,
        )
        policy_full[name] = geometric(data["raw"], data["old"], full_policy_alpha)
        policy_fixed[name] = geometric(data["fixed_raw"], data["fixed_old"], fixed_policy_alpha)
        policy_changes[name] = {
            "full_alpha_changed_rows": int(np.sum(full_policy_alpha != alpha)),
            "fixed_alpha_changed_rows": int(np.sum(fixed_policy_alpha != fixed_alpha)),
        }

    adaptive = nested_adaptive(
        data["y"], policy_full, policy_fixed, data["terminal"], data["char20"],
        data["fixed_y"], data["fixed_terminal"], data["fixed_char20"], data["fold"],
    )
    adaptive_nested_probability = adaptive.pop("full_outer_selected_probability")
    adaptive_fixed_probability = adaptive.pop("fixed_consensus_probability")
    adaptive_folds = fold_deltas(
        data["y"], base_full_probability, adaptive_nested_probability,
        data["terminal"], data["char20"], data["fold"],
    )
    adaptive_full_terminal = labels_with_char(
        adaptive_nested_probability, data["terminal"], data["char20"],
    )[1][data["terminal"]]
    adaptive_fixed_terminal = labels_with_char(
        adaptive_fixed_probability, data["fixed_terminal"], data["fixed_char20"],
    )[1][data["fixed_terminal"]]
    adaptive_bootstrap_full = session_bootstrap(
        data["y"][data["terminal"]], base_full_terminal, adaptive_full_terminal,
        data["groups"][data["terminal"]], args.bootstrap_replicates,
    )
    adaptive_bootstrap_fixed = session_bootstrap(
        data["fixed_y"][data["fixed_terminal"]], base_fixed_terminal, adaptive_fixed_terminal,
        data["fixed_groups"][data["fixed_terminal"]], args.bootstrap_replicates,
    )
    adaptive_gate = promotion_gate(
        metric_delta(adaptive["full_outer_selected_metrics"], base_full),
        metric_delta(adaptive["fixed_consensus_metrics"], base_fixed),
        adaptive_bootstrap_full, adaptive_bootstrap_fixed, adaptive_folds,
        consensus_nonbase=adaptive["fixed_consensus_policy"] != "base",
    )

    report: dict[str, Any] = {
        "scope": (
            "Experiment-only. No model/, script.py, B42.zip, baseline_submit.zip, "
            "B42_9 archive, B43 archive, or live ZIP was changed."
        ),
        "reference": (
            "Exact B43 raw seed42 plus old no-E5 GRU geometric SIM=.20/AU=.85 "
            "and immutable terminal char20 last."
        ),
        "provenance": provenance,
        "baseline_b43": {
            "full_outer_oof": base_full,
            "matching_seed42_fixed": base_fixed,
        },
        "portable_dual_gru_implementation": portable_contract,
        "dual_arithmetic_static_weight_table": {
            "definition": (
                "For each requested fixed strict weight, GRU probability equals "
                "(1-w)*old_noe5 + w*strict_distill_noe5. Raw/GRU fusion stays "
                "B43 geometric SIM=.20/AU=.85 and char20 remains terminal-last."
            ),
            "requested_strict_weights": list(STATIC_STRICT_WEIGHTS),
            "rows": static_dual_weights,
            "recommendation": static_weight_recommendation,
        },
        "dual_arithmetic_50_50": {
            "definition": (
                "The requested w=.50 row from the fixed static table, retained "
                "as the direct B43 50:50 comparison."
            ),
            **dual_50,
        },
        "adaptive_margin_gate": {
            "predeclared_policies": {
                name: {
                    "description": POLICY_DESCRIPTION[name],
                    "changed_rows": policy_changes[name],
                }
                for name in POLICIES
            },
            **adaptive,
            "delta_vs_b43": {
                "full_outer_selected": metric_delta(
                    adaptive["full_outer_selected_metrics"], base_full,
                ),
                "matching_seed42_fixed_consensus": metric_delta(
                    adaptive["fixed_consensus_metrics"], base_fixed,
                ),
            },
            "full_outer_session_fold_deltas": adaptive_folds,
            "terminal_char20_session_bootstrap": {
                "full_outer_selected": adaptive_bootstrap_full,
                "matching_seed42_fixed_consensus": adaptive_bootstrap_fixed,
            },
            "promotion_gate": adaptive_gate,
        },
        "submission_action": {
            "candidate_created": False,
            "zip_updated": False,
            "reason": (
                "A candidate is created only after all declared two-protocol, "
                "session-fold, and session-bootstrap gates pass."
            ),
        },
    }
    (out / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    write_readme(out, report)
    print(json.dumps({
        "dual_promote": dual_50["promotion_gate"]["promote"],
        "adaptive_promote": adaptive_gate["promote"],
        "output": str(out),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
