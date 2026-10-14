#!/usr/bin/env python3
"""Strict audit: B43 dual-GRU 50:50 plus the preregistered AU specialist.

This is deliberately a *replay*, rather than a new model-selection pass.  Its
five full-data AU policies were selected previously using only the four outer
training folds and are recorded verbatim below.  For each held fold, this
program refits just that recorded AU specialist on the other four folds.  The
matching fixed split likewise uses the separately recorded common-train
choice, compact / global raw weight .50.  Neither held-fold nor fixed labels
are used to choose a configuration or mixing weight here.

The candidate changes only the AU raw-selector component.  It then applies
the fixed arithmetic dual GRU (old .50 + strict-distill .50), the unchanged
B43 source geometry (SIM raw=.20, AU raw=.85), immutable char20 terminal
overlay, and the existing future-history hybrid metric.  It is experiment
only and cannot write model/, script.py, or submission archives.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from experiment_au_specialist_b43 import CONFIGS
from experiment_b43_adaptive_blend import (
    EXPECTED_B43,
    METRIC_KEYS,
    N_CLASSES,
    PROMOTION_BOOTSTRAP_P,
    check_baseline,
    fold_deltas,
    geometric,
    labels_with_char,
    load_exact_b43,
    metric_bundle,
    metric_delta,
    session_bootstrap,
    source_alpha,
)
from experiment_xgb_robust_ensemble import full_context, selector_probability, strict_context


# These policies are the completed, inner-only choices logged by the AU audit.
# Index 0 corresponds to the held GroupKFold fold numbered 1 in reports.
OUTER_REPLAY_CHOICES: tuple[tuple[str, float], ...] = (
    ("regularized", 0.00),
    ("compact", 0.50),
    ("compact", 0.50),
    ("compact", 0.25),
    ("regularized", 0.25),
)
FIXED_COMMON_TRAIN_CHOICE: tuple[str, float] = ("compact", 0.50)
DEFAULT_BOOTSTRAP_REPLICATES = 2000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="experiments/b43_dual_au_composition")
    parser.add_argument("--bootstrap-replicates", type=int, default=DEFAULT_BOOTSTRAP_REPLICATES)
    parser.add_argument("--force", action="store_true", help="Refit any cached AU specialist block.")
    return parser.parse_args()


def dual_gru(old: np.ndarray, strict: np.ndarray) -> np.ndarray:
    """The exact predeclared arithmetic old=.50 / strict=.50 composition."""
    value = (0.50 * old + 0.50 * strict).astype(np.float32)
    if not np.allclose(value.sum(axis=1), 1.0, rtol=0.0, atol=2.0e-6):
        raise ValueError("Dual-GRU composition is not probability normalized")
    return value


def raw_with_au_specialist(
    raw: np.ndarray,
    source: np.ndarray,
    specialist: np.ndarray,
    global_raw_weight: float,
) -> np.ndarray:
    """Replace only AU rows with the fixed global/raw-specialist mixture."""
    if raw.shape != specialist.shape or raw.shape[0] != len(source):
        raise ValueError("Raw/specialist/source shapes do not align")
    output = raw.copy()
    au = source == "au"
    if np.any(au):
        output[au] = (
            float(global_raw_weight) * raw[au]
            + (1.0 - float(global_raw_weight)) * specialist[au]
        )
    output /= output.sum(axis=1, keepdims=True)
    return output.astype(np.float32, copy=False)


def specialist_full_block(
    out: Path,
    held: int,
    config_name: str,
    x: np.ndarray,
    y: np.ndarray,
    source: np.ndarray,
    fold: np.ndarray,
    force: bool,
) -> np.ndarray:
    """Refit exactly one logged outer-fold AU specialist, with safe caching."""
    target = out / f"full_held_fold_{held + 1}_{config_name}_au_specialist.npy"
    held_mask = fold == held
    au_held = held_mask & (source == "au")
    if target.exists() and not force:
        block = np.load(target, allow_pickle=False).astype(np.float32)
        if block.shape == (len(y), N_CLASSES):
            # The artifact must carry values only for the held AU rows.  This
            # guards against accidentally reusing an all-row or wrong-fold cache.
            if np.all(block[~au_held] == 0.0) and np.all(block[au_held].sum(axis=1) > 0.999):
                print(json.dumps({"held_fold": held + 1, "cache": "reused", "config": config_name}), flush=True)
                return block
        raise ValueError(f"Unsafe cached specialist block: {target}")
    block = np.zeros((len(y), N_CLASSES), dtype=np.float32)
    if np.any(au_held):
        block[au_held] = selector_probability(
            CONFIGS[config_name], x[(~held_mask) & (source == "au")], y[(~held_mask) & (source == "au")],
            x[au_held], device="cuda", n_jobs=4,
        )
    np.save(target, block, allow_pickle=False)
    print(json.dumps({"held_fold": held + 1, "cache": "written", "config": config_name, "au_rows": int(au_held.sum())}), flush=True)
    return block


def specialist_fixed_block(
    out: Path,
    config_name: str,
    x_train: np.ndarray,
    y_train: np.ndarray,
    source_train: np.ndarray,
    x_valid: np.ndarray,
    source_valid: np.ndarray,
    force: bool,
) -> np.ndarray:
    """Refit the one recorded common-train policy on AU data only."""
    target = out / f"fixed_common_train_{config_name}_au_specialist.npy"
    au_valid = source_valid == "au"
    if target.exists() and not force:
        block = np.load(target, allow_pickle=False).astype(np.float32)
        if block.shape == (len(source_valid), N_CLASSES):
            if np.all(block[~au_valid] == 0.0) and np.all(block[au_valid].sum(axis=1) > 0.999):
                print(json.dumps({"fixed": True, "cache": "reused", "config": config_name}), flush=True)
                return block
        raise ValueError(f"Unsafe cached fixed specialist block: {target}")
    block = np.zeros((len(source_valid), N_CLASSES), dtype=np.float32)
    if np.any(au_valid):
        au_train = source_train == "au"
        block[au_valid] = selector_probability(
            CONFIGS[config_name], x_train[au_train], y_train[au_train], x_valid[au_valid],
            device="cuda", n_jobs=4,
        )
    np.save(target, block, allow_pickle=False)
    print(json.dumps({"fixed": True, "cache": "written", "config": config_name, "au_rows": int(au_valid.sum())}), flush=True)
    return block


def final_terminal(probability: np.ndarray, terminal: np.ndarray, char20: np.ndarray) -> np.ndarray:
    return labels_with_char(probability, terminal, char20)[1][terminal]


def strict_gate(record: dict[str, Any]) -> dict[str, Any]:
    """Require an unambiguous B43 improvement before any deployment discussion."""
    versus_b43 = record["delta_vs_b43"]
    bootstrap = record["terminal_char20_session_bootstrap_vs_b43"]
    fold_rows = record["full_outer_session_fold_deltas_vs_b43"]
    checks = {
        "full_all_four_metrics_positive": all(
            float(versus_b43["full_outer_oof"][key]) > 0.0 for key in METRIC_KEYS
        ),
        "fixed_all_four_metrics_positive": all(
            float(versus_b43["matching_seed42_fixed"][key]) > 0.0 for key in METRIC_KEYS
        ),
        "full_terminal_char20_bootstrap_p_ge_0_95": (
            float(bootstrap["full_outer_oof"]["p_delta_gt_zero"]) >= PROMOTION_BOOTSTRAP_P
        ),
        "fixed_terminal_char20_bootstrap_p_ge_0_95": (
            float(bootstrap["matching_seed42_fixed"]["p_delta_gt_zero"]) >= PROMOTION_BOOTSTRAP_P
        ),
        "full_terminal_char20_positive_in_at_least_4_of_5_folds": (
            sum(
                row["delta"]["terminal_with_char20_macro_f1"] > 0.0
                for row in fold_rows
            ) >= 4
        ),
    }
    return {
        "thresholds": {
            "bootstrap_p_delta_gt_zero": PROMOTION_BOOTSTRAP_P,
            "all_metrics_must_improve": list(METRIC_KEYS),
            "positive_full_terminal_char20_outer_folds": 4,
        },
        "checks": checks,
        "pass": bool(all(checks.values())),
        "reason": (
            "Only a full/fixed, session-stable B43 improvement is eligible for a new submission."
            if all(checks.values()) else
            "Reject: the preregistered combined candidate does not clear every strict B43 gate."
        ),
    }


def main() -> None:
    args = parse_args()
    if args.bootstrap_replicates <= 0:
        raise ValueError("--bootstrap-replicates must be positive")
    root = Path(__file__).resolve().parents[1]
    out = Path(args.out_dir)
    if not out.is_absolute():
        out = root / out
    out = out.resolve()
    model_dir = (root / "model").resolve()
    if out == model_dir or model_dir in out.parents:
        raise ValueError("Experiment output must not be inside model/")
    out.mkdir(parents=True, exist_ok=True)

    data, provenance = load_exact_b43(root)
    b43_full_probability = geometric(data["raw"], data["old"], source_alpha(data["source"]))
    b43_fixed_probability = geometric(
        data["fixed_raw"], data["fixed_old"], source_alpha(data["fixed_source"]),
    )
    b43_full = metric_bundle(data["y"], b43_full_probability, data["terminal"], data["char20"])
    b43_fixed = metric_bundle(
        data["fixed_y"], b43_fixed_probability, data["fixed_terminal"], data["fixed_char20"],
    )
    check_baseline(b43_full, EXPECTED_B43["full"], "full")
    check_baseline(b43_fixed, EXPECTED_B43["fixed"], "matching_seed42_fixed")

    dual_full_gru = dual_gru(data["old"], data["strict"])
    dual_fixed_gru = dual_gru(data["fixed_old"], data["fixed_strict"])
    dual_full_probability = geometric(data["raw"], dual_full_gru, source_alpha(data["source"]))
    dual_fixed_probability = geometric(
        data["fixed_raw"], dual_fixed_gru, source_alpha(data["fixed_source"]),
    )
    dual_full = metric_bundle(data["y"], dual_full_probability, data["terminal"], data["char20"])
    dual_fixed = metric_bundle(
        data["fixed_y"], dual_fixed_probability, data["fixed_terminal"], data["fixed_char20"],
    )

    # The expensive feature construction is read-only and must exactly follow
    # the long-standing XGB selector contract.  Verify all labels/sources/folds
    # before any GPU fit so a prediction cannot silently be associated with a
    # different ordering than B43.
    x, x_y, x_source, x_groups, _ = full_context()
    if not (
        np.array_equal(x_y, data["y"])
        and np.array_equal(x_source.astype(str), data["source"].astype(str))
        and np.array_equal(x_groups.astype(str), data["groups"].astype(str))
    ):
        raise ValueError("Full selector context is not aligned with exact B43 rows")
    if len(OUTER_REPLAY_CHOICES) != len(np.unique(data["fold"])):
        raise ValueError("Need exactly one logged AU choice per outer session fold")

    full_specialist = np.zeros_like(data["raw"], dtype=np.float32)
    full_weights = np.zeros(len(data["y"]), dtype=np.float32)
    replay_records: list[dict[str, Any]] = []
    for held, (config_name, global_raw_weight) in enumerate(OUTER_REPLAY_CHOICES):
        if config_name not in CONFIGS:
            raise ValueError(f"Unknown replay config: {config_name}")
        held_mask = data["fold"] == held
        block = specialist_full_block(
            out, held, config_name, x, data["y"], data["source"], data["fold"], args.force,
        )
        full_specialist[held_mask] = block[held_mask]
        full_weights[held_mask] = global_raw_weight
        replay_records.append({
            "held_fold": held + 1,
            "config": config_name,
            "global_raw_weight": global_raw_weight,
            "selection_origin": "logged completed inner-only AU audit; replayed without held-fold labels",
            "held_rows": int(held_mask.sum()),
            "held_au_rows": int(np.sum(held_mask & (data["source"] == "au"))),
        })

    mixed_full_raw = data["raw"].copy()
    au_full = data["source"] == "au"
    mixed_full_raw[au_full] = (
        full_weights[au_full, None] * data["raw"][au_full]
        + (1.0 - full_weights[au_full, None]) * full_specialist[au_full]
    )
    mixed_full_raw /= mixed_full_raw.sum(axis=1, keepdims=True)
    candidate_full_probability = geometric(mixed_full_raw, dual_full_gru, source_alpha(data["source"]))

    x_train, y_train, source_train, x_valid, y_valid, source_valid = strict_context()
    strict_saved = np.load(root / "experiments/au_source_strict_common/strict_blocks.npz", allow_pickle=False)
    fixed_ids = np.load(
        root / "experiments/b42_9_common_old_noe5_seed42/fixed_valid_ids.npy", allow_pickle=False,
    ).astype(str)
    if not (
        len(fixed_ids) == len(data["fixed_y"])
        and np.array_equal(strict_saved["ids_valid"].astype(str), fixed_ids)
        and np.array_equal(y_valid, data["fixed_y"])
        and np.array_equal(source_valid.astype(str), data["fixed_source"].astype(str))
    ):
        raise ValueError("Fixed selector context is not aligned with exact B43 rows")
    fixed_name, fixed_global_raw_weight = FIXED_COMMON_TRAIN_CHOICE
    fixed_specialist = specialist_fixed_block(
        out, fixed_name, x_train, y_train, source_train, x_valid, source_valid, args.force,
    )
    mixed_fixed_raw = raw_with_au_specialist(
        data["fixed_raw"], data["fixed_source"], fixed_specialist, fixed_global_raw_weight,
    )
    candidate_fixed_probability = geometric(
        mixed_fixed_raw, dual_fixed_gru, source_alpha(data["fixed_source"]),
    )

    candidate_full = metric_bundle(
        data["y"], candidate_full_probability, data["terminal"], data["char20"],
    )
    candidate_fixed = metric_bundle(
        data["fixed_y"], candidate_fixed_probability, data["fixed_terminal"], data["fixed_char20"],
    )
    b43_full_terminal = final_terminal(b43_full_probability, data["terminal"], data["char20"])
    dual_full_terminal = final_terminal(dual_full_probability, data["terminal"], data["char20"])
    candidate_full_terminal = final_terminal(candidate_full_probability, data["terminal"], data["char20"])
    b43_fixed_terminal = final_terminal(b43_fixed_probability, data["fixed_terminal"], data["fixed_char20"])
    dual_fixed_terminal = final_terminal(dual_fixed_probability, data["fixed_terminal"], data["fixed_char20"])
    candidate_fixed_terminal = final_terminal(
        candidate_fixed_probability, data["fixed_terminal"], data["fixed_char20"],
    )

    record: dict[str, Any] = {
        "candidate_definition": {
            "gru": "arithmetic old no-E5=.50 + strict-distill no-E5=.50",
            "au_route": "Refit only logged AU-specialist policies and mix raw according to their recorded global raw weights.",
            "source_geometry": "unchanged B43 geometric raw/GRU: SIM raw=.20, AU raw=.85",
            "terminal_order": "raw mix -> dual GRU fusion -> argmax -> immutable char20 terminal override -> future-history only in hybrid metric",
        },
        "full_outer_replay_choices": replay_records,
        "fixed_common_train_choice": {
            "config": fixed_name,
            "global_raw_weight": fixed_global_raw_weight,
            "selection_origin": "recorded AU audit common-train choice; fixed labels were not used to select it",
        },
        "b43_baseline": {
            "full_outer_oof": b43_full,
            "matching_seed42_fixed": b43_fixed,
        },
        "dual_gru_50_50_baseline": {
            "full_outer_oof": dual_full,
            "matching_seed42_fixed": dual_fixed,
            "delta_vs_b43": {
                "full_outer_oof": metric_delta(dual_full, b43_full),
                "matching_seed42_fixed": metric_delta(dual_fixed, b43_fixed),
            },
        },
        "combined_candidate": {
            "full_outer_oof": candidate_full,
            "matching_seed42_fixed": candidate_fixed,
        },
        "delta_vs_b43": {
            "full_outer_oof": metric_delta(candidate_full, b43_full),
            "matching_seed42_fixed": metric_delta(candidate_fixed, b43_fixed),
        },
        "delta_vs_dual_gru_50_50": {
            "full_outer_oof": metric_delta(candidate_full, dual_full),
            "matching_seed42_fixed": metric_delta(candidate_fixed, dual_fixed),
        },
        "terminal_char20_session_bootstrap_vs_b43": {
            "full_outer_oof": session_bootstrap(
                data["y"][data["terminal"]], b43_full_terminal, candidate_full_terminal,
                data["groups"][data["terminal"]], args.bootstrap_replicates,
            ),
            "matching_seed42_fixed": session_bootstrap(
                data["fixed_y"][data["fixed_terminal"]], b43_fixed_terminal, candidate_fixed_terminal,
                data["fixed_groups"][data["fixed_terminal"]], args.bootstrap_replicates,
            ),
        },
        "terminal_char20_session_bootstrap_vs_dual_gru_50_50": {
            "full_outer_oof": session_bootstrap(
                data["y"][data["terminal"]], dual_full_terminal, candidate_full_terminal,
                data["groups"][data["terminal"]], args.bootstrap_replicates,
            ),
            "matching_seed42_fixed": session_bootstrap(
                data["fixed_y"][data["fixed_terminal"]], dual_fixed_terminal, candidate_fixed_terminal,
                data["fixed_groups"][data["fixed_terminal"]], args.bootstrap_replicates,
            ),
        },
    }
    record["full_outer_session_fold_deltas_vs_b43"] = fold_deltas(
        data["y"], b43_full_probability, candidate_full_probability,
        data["terminal"], data["char20"], data["fold"],
    )
    record["full_outer_session_fold_deltas_vs_dual_gru_50_50"] = fold_deltas(
        data["y"], dual_full_probability, candidate_full_probability,
        data["terminal"], data["char20"], data["fold"],
    )
    record["strict_gate"] = strict_gate(record)
    report = {
        "scope": "Experiment-only. model/, script.py, B42.zip, baseline_submit.zip, and every submission artifact are unchanged.",
        "provenance": provenance,
        "replay_contract": {
            "outer_choices": [
                {"fold": index + 1, "config": name, "global_raw_weight": weight}
                for index, (name, weight) in enumerate(OUTER_REPLAY_CHOICES)
            ],
            "fixed_common_train_choice": {"config": fixed_name, "global_raw_weight": fixed_global_raw_weight},
            "no_new_policy_selection": True,
            "held_or_fixed_labels_used_for_selection": False,
        },
        "result": record,
        "promotion_gate": {
            "pass": record["strict_gate"]["pass"],
            "reason": "No model or ZIP candidate is created unless this preregistered replay clears the strict B43 gate.",
        },
        "submission_action": {"candidate_created": False, "zip_updated": False},
    }
    (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    np.savez_compressed(
        out / "predictions.npz",
        y=data["y"], terminal=data["terminal"], groups=data["groups"], fold=data["fold"],
        b43_full_probability=b43_full_probability,
        dual_full_probability=dual_full_probability,
        candidate_full_probability=candidate_full_probability,
        b43_fixed_probability=b43_fixed_probability,
        dual_fixed_probability=dual_fixed_probability,
        candidate_fixed_probability=candidate_fixed_probability,
        full_au_specialist=full_specialist,
        full_au_global_raw_weight=full_weights,
        fixed_au_specialist=fixed_specialist,
    )
    print(json.dumps({
        "strict_gate_pass": record["strict_gate"]["pass"],
        "out": str(out),
        "full_terminal_char_delta_vs_b43": record["delta_vs_b43"]["full_outer_oof"]["terminal_with_char20_macro_f1"],
        "fixed_terminal_char_delta_vs_b43": record["delta_vs_b43"]["matching_seed42_fixed"]["terminal_with_char20_macro_f1"],
    }, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
