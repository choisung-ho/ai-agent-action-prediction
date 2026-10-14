#!/usr/bin/env python3
"""Targeted post-hoc audit of the sole static AU route under three GRU weights.

This is intentionally narrow: regularized AU OOF plus global raw=.25 is held
fixed, and only strict-distill weights .25/.35/.50 are compared.  It is not a
new policy search and cannot create a submission artifact.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from audit_b43_dual_au_static_deploy import (
    OUT,
    mix_au_raw,
    shared_session_bootstrap,
    strict_fixed_context,
    terminal_prediction,
    validate_full_au_oof,
)
from experiment_au_specialist_b43 import CONFIGS
from experiment_b43_adaptive_blend import (
    EXPECTED_B43,
    METRIC_KEYS,
    PROMOTION_BOOTSTRAP_P,
    check_baseline,
    geometric,
    load_exact_b43,
    metric_bundle,
    metric_delta,
    source_alpha,
)
from experiment_xgb_robust_ensemble import ROOT, selector_probability


STRICT_WEIGHTS = (0.25, 0.35, 0.50)
AU_GLOBAL_RAW_WEIGHT = 0.25
CACHE = OUT / "fixed_regularized_config142_common_train_probability.npz"


def digest_ids(ids: np.ndarray) -> str:
    value = hashlib.sha256()
    for sample_id in ids.astype(str):
        value.update(sample_id.encode("utf-8"))
        value.update(b"\n")
    return value.hexdigest()


def load_or_fit_fixed_regularized(
    root: Path,
    data: dict[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    """Use one exact config-142 common-train AU fit for all three GRU weights."""
    x_train, y_train, source_train, x_valid, y_valid, source_valid, context = strict_fixed_context(root, data)
    strict_blocks = np.load(
        root / "experiments/au_source_strict_common/strict_blocks.npz", allow_pickle=False,
    )
    valid_digest = digest_ids(strict_blocks["ids_valid"])
    train_digest = digest_ids(strict_blocks["ids_train"])
    expected_shape = (len(y_valid), data["fixed_raw"].shape[1])
    if CACHE.exists():
        cached = np.load(CACHE, allow_pickle=False)
        probability = cached["probability"].astype(np.float32)
        config_json = str(cached["config_json"].item())
        if not (
            probability.shape == expected_shape
            and str(cached["valid_ids_sha256"].item()) == valid_digest
            and str(cached["train_ids_sha256"].item()) == train_digest
            and config_json == json.dumps(CONFIGS["regularized"], sort_keys=True)
        ):
            raise ValueError("Unsafe fixed regularized cache: feature IDs or config differ")
        au_valid = source_valid == "au"
        if not (
            np.all(probability[~au_valid] == 0.0)
            and np.allclose(probability[au_valid].sum(axis=1), 1.0, rtol=0.0, atol=2.0e-6)
        ):
            raise ValueError("Unsafe fixed regularized cache: AU mask or normalization differs")
        return probability, {**context, "fixed_regularized_artifact": "cache_reused", "cache": str(CACHE.relative_to(root)).replace("\\", "/")}

    probability = np.zeros(expected_shape, dtype=np.float32)
    au_train = source_train == "au"
    au_valid = source_valid == "au"
    probability[au_valid] = selector_probability(
        CONFIGS["regularized"], x_train[au_train], y_train[au_train], x_valid[au_valid],
        device="cuda", n_jobs=4,
    )
    if not np.allclose(probability[au_valid].sum(axis=1), 1.0, rtol=0.0, atol=2.0e-6):
        raise ValueError("Fresh exact fixed regularized AU probabilities are not normalized")
    np.savez_compressed(
        CACHE,
        probability=probability,
        valid_ids_sha256=np.asarray(valid_digest),
        train_ids_sha256=np.asarray(train_digest),
        config_json=np.asarray(json.dumps(CONFIGS["regularized"], sort_keys=True)),
    )
    return probability, {**context, "fixed_regularized_artifact": "fresh_fit", "cache": str(CACHE.relative_to(root)).replace("\\", "/")}


def gru_mix(old: np.ndarray, strict: np.ndarray, strict_weight: float) -> np.ndarray:
    result = ((1.0 - strict_weight) * old + strict_weight * strict).astype(np.float32)
    if not np.allclose(result.sum(axis=1), 1.0, rtol=0.0, atol=2.0e-6):
        raise ValueError("GRU mixture is not normalized")
    return result


def all_positive(delta: dict[str, float]) -> bool:
    return bool(all(delta[name] > 0.0 for name in METRIC_KEYS))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--strict-weights", nargs="+", type=float, default=list(STRICT_WEIGHTS),
        help="Narrow, predeclared arithmetic strict-distill weights to audit.",
    )
    parser.add_argument("--report-name", default="weight_followup_report.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    weights = tuple(float(value) for value in args.strict_weights)
    if not weights or any(value < 0.0 or value > 1.0 for value in weights):
        raise ValueError("Strict weights must be nonempty and lie in [0, 1]")
    if Path(args.report_name).name != args.report_name:
        raise ValueError("--report-name must be a filename, not a path")
    OUT.mkdir(parents=True, exist_ok=True)
    root = ROOT
    data, provenance = load_exact_b43(root)
    base_full_probability = geometric(data["raw"], data["old"], source_alpha(data["source"]))
    base_fixed_probability = geometric(
        data["fixed_raw"], data["fixed_old"], source_alpha(data["fixed_source"]),
    )
    base_full = metric_bundle(data["y"], base_full_probability, data["terminal"], data["char20"])
    base_fixed = metric_bundle(
        data["fixed_y"], base_fixed_probability, data["fixed_terminal"], data["fixed_char20"],
    )
    check_baseline(base_full, EXPECTED_B43["full"], "full")
    check_baseline(base_fixed, EXPECTED_B43["fixed"], "matching_seed42_fixed")

    specialist_full, au_provenance = validate_full_au_oof(root, data)
    regularized_full_raw = mix_au_raw(
        data["raw"], data["source"], specialist_full["regularized"], AU_GLOBAL_RAW_WEIGHT,
    )
    fixed_regularized, fixed_context = load_or_fit_fixed_regularized(root, data)
    regularized_fixed_raw = mix_au_raw(
        data["fixed_raw"], data["fixed_source"], fixed_regularized, AU_GLOBAL_RAW_WEIGHT,
    )

    base_full_terminal = terminal_prediction(base_full_probability, data["terminal"], data["char20"])
    base_fixed_terminal = terminal_prediction(
        base_fixed_probability, data["fixed_terminal"], data["fixed_char20"],
    )
    full_probability: dict[str, np.ndarray] = {}
    fixed_probability: dict[str, np.ndarray] = {}
    full_terminal: dict[str, np.ndarray] = {}
    fixed_terminal: dict[str, np.ndarray] = {}
    full_metric: dict[str, dict[str, float]] = {}
    fixed_metric: dict[str, dict[str, float]] = {}
    for weight in weights:
        key = f"strict_weight_{weight:.2f}"
        full_probability[key] = geometric(
            regularized_full_raw, gru_mix(data["old"], data["strict"], weight), source_alpha(data["source"]),
        )
        fixed_probability[key] = geometric(
            regularized_fixed_raw, gru_mix(data["fixed_old"], data["fixed_strict"], weight), source_alpha(data["fixed_source"]),
        )
        full_metric[key] = metric_bundle(
            data["y"], full_probability[key], data["terminal"], data["char20"],
        )
        fixed_metric[key] = metric_bundle(
            data["fixed_y"], fixed_probability[key], data["fixed_terminal"], data["fixed_char20"],
        )
        full_terminal[key] = terminal_prediction(
            full_probability[key], data["terminal"], data["char20"],
        )
        fixed_terminal[key] = terminal_prediction(
            fixed_probability[key], data["fixed_terminal"], data["fixed_char20"],
        )

    full_bootstrap = shared_session_bootstrap(
        data["y"][data["terminal"]], base_full_terminal, full_terminal, data["groups"][data["terminal"]],
    )
    fixed_bootstrap = shared_session_bootstrap(
        data["fixed_y"][data["fixed_terminal"]], base_fixed_terminal, fixed_terminal,
        data["fixed_groups"][data["fixed_terminal"]],
    )
    records: dict[str, dict[str, Any]] = {}
    for weight in weights:
        key = f"strict_weight_{weight:.2f}"
        full_delta = metric_delta(full_metric[key], base_full)
        fixed_delta = metric_delta(fixed_metric[key], base_fixed)
        full_pass = all_positive(full_delta) and full_bootstrap[key]["p_delta_gt_zero"] >= PROMOTION_BOOTSTRAP_P
        fixed_pass = all_positive(fixed_delta) and fixed_bootstrap[key]["p_delta_gt_zero"] >= PROMOTION_BOOTSTRAP_P
        records[key] = {
            "old_noe5_weight": 1.0 - weight,
            "strict_distill_weight": weight,
            "full_outer_oof": full_metric[key],
            "matching_seed42_fixed": fixed_metric[key],
            "delta_vs_b43": {"full_outer_oof": full_delta, "matching_seed42_fixed": fixed_delta},
            "terminal_char20_session_bootstrap": {
                "full_outer_oof": full_bootstrap[key],
                "matching_seed42_fixed": fixed_bootstrap[key],
            },
            "screen": {
                "full_four_metrics_positive": all_positive(full_delta),
                "full_bootstrap_p_ge_0_95": bool(full_bootstrap[key]["p_delta_gt_zero"] >= PROMOTION_BOOTSTRAP_P),
                "fixed_four_metrics_positive": all_positive(fixed_delta),
                "fixed_bootstrap_p_ge_0_95": bool(fixed_bootstrap[key]["p_delta_gt_zero"] >= PROMOTION_BOOTSTRAP_P),
                "pass": bool(full_pass and fixed_pass),
            },
        }
    passes = [key for key, record in records.items() if record["screen"]["pass"]]
    report = {
        "scope": (
            "Targeted post-hoc/static follow-up. The full AU selector remains the original five-fold "
            "OOF probability block; only the exact matching-seed42 fixed regularized selector is freshly "
            "fit/reused. This is not nested deployment evidence and creates no ZIP/model/stage artifact."
        ),
        "fixed_au_route": {
            "config": "regularized",
            "global_raw_weight": AU_GLOBAL_RAW_WEIGHT,
            "full_probability_source": "original AU five-fold OOF",
            "fixed_probability_source": fixed_context,
        },
        "strict_weight_grid": list(weights),
        "provenance": provenance,
        "original_au_oof_alignment": au_provenance,
        "b43_baseline": {"full_outer_oof": base_full, "matching_seed42_fixed": base_fixed},
        "records": records,
        "gate": {
            "requires": "four positive full deltas + full 10k p>=.95 + four positive fixed deltas + fixed 10k p>=.95",
            "pass": bool(passes),
            "passing_weights": passes,
            "recommendation": (
                "No targeted weight clears both strict B43 stability gates; preserve B43."
                if not passes else
                "A static/post-hoc weight passes numerical gates; require an independent preregistered nested check before any deployment discussion."
            ),
        },
        "submission_action": {"candidate_created": False, "zip_updated": False, "live_files_changed": False},
    }
    path = OUT / args.report_name
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "report": str(path),
        "summary": {
            key: {
                "full_terminal_char_delta": value["delta_vs_b43"]["full_outer_oof"]["terminal_with_char20_macro_f1"],
                "fixed_terminal_char_delta": value["delta_vs_b43"]["matching_seed42_fixed"]["terminal_with_char20_macro_f1"],
                "full_bootstrap_p": value["terminal_char20_session_bootstrap"]["full_outer_oof"]["p_delta_gt_zero"],
                "fixed_bootstrap_p": value["terminal_char20_session_bootstrap"]["matching_seed42_fixed"]["p_delta_gt_zero"],
                "pass": value["screen"]["pass"],
            }
            for key, value in records.items()
        },
        "passes": passes,
    }, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
