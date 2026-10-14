#!/usr/bin/env python3
"""Post-hoc static screen for a B43 dual-GRU + AU-selector deployment policy.

The screen deliberately does *not* select a policy inside nested folds.  It
reuses the already-recorded AU-only five-fold OOF probabilities for a fixed
eight-policy grid, then applies the exact B43 geometry and char20 overlay.
It is therefore evidence for rejecting weak policies, never sufficient on its
own to create a submission candidate.

Only ``experiments/b43_dual_au_static_deploy_audit`` is written.  In
particular, this program never writes model/, script.py, staging directories,
or ZIP files.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from experiment_au_specialist_b43 import CONFIGS
from experiment_b43_adaptive_blend import (
    BOOTSTRAP_SEED,
    EXPECTED_B43,
    METRIC_KEYS,
    N_CLASSES,
    PROMOTION_BOOTSTRAP_P,
    check_baseline,
    geometric,
    labels_with_char,
    load_exact_b43,
    metric_bundle,
    metric_delta,
    source_alpha,
    weighted_macro,
)
from experiment_xgb_robust_ensemble import ROOT, selector_probability, strict_context


OUT = ROOT / "experiments/b43_dual_au_static_deploy_audit"
RAW_WEIGHTS = (0.00, 0.25, 0.50, 0.75)
BOOTSTRAP_REPLICATES = 10_000


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def full_ids(root: Path) -> np.ndarray:
    return np.asarray([
        str(json.loads(line)["id"])
        for line in (root / "data/train.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ])


def dual_gru(old: np.ndarray, strict: np.ndarray) -> np.ndarray:
    value = (0.50 * old + 0.50 * strict).astype(np.float32)
    if not np.allclose(value.sum(axis=1), 1.0, rtol=0.0, atol=2.0e-6):
        raise ValueError("The exact dual50 GRU is not normalized")
    return value


def mix_au_raw(
    raw: np.ndarray,
    source: np.ndarray,
    specialist: np.ndarray,
    raw_weight: float,
) -> np.ndarray:
    if raw.shape != specialist.shape:
        raise ValueError("Raw and AU specialist shapes differ")
    output = raw.copy()
    au = source == "au"
    output[au] = (
        float(raw_weight) * raw[au]
        + (1.0 - float(raw_weight)) * specialist[au]
    )
    output /= output.sum(axis=1, keepdims=True)
    return output.astype(np.float32, copy=False)


def terminal_prediction(probability: np.ndarray, terminal: np.ndarray, char20: np.ndarray) -> np.ndarray:
    """Argmax first, immutable char20 terminal override last."""
    return labels_with_char(probability, terminal, char20)[1][terminal]


def shared_session_bootstrap(
    y: np.ndarray,
    baseline: np.ndarray,
    candidates: dict[str, np.ndarray],
    groups: np.ndarray,
    replicates: int = BOOTSTRAP_REPLICATES,
) -> dict[str, dict[str, float | int]]:
    """Exact seeded whole-session bootstrap, sharing draws across policies.

    This is algebraically identical to running the established per-candidate
    bootstrap with the same seed for each candidate, but avoids recomputing
    the baseline confusion matrix eight times per resample.
    """
    if not candidates:
        return {}
    unique, inverse = np.unique(groups.astype(str), return_inverse=True)
    base_cell = y.astype(np.int64) * N_CLASSES + baseline.astype(np.int64)
    candidate_cell = {
        key: y.astype(np.int64) * N_CLASSES + value.astype(np.int64)
        for key, value in candidates.items()
    }
    delta = {key: np.empty(replicates, dtype=np.float64) for key in candidates}
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    for number in range(replicates):
        sampled = rng.integers(0, len(unique), size=len(unique))
        count = np.bincount(sampled, minlength=len(unique))
        weight = count[inverse]
        base_confusion = np.bincount(
            base_cell, weights=weight, minlength=N_CLASSES * N_CLASSES,
        ).reshape(N_CLASSES, N_CLASSES)
        base_score = weighted_macro(base_confusion)
        for key, cell in candidate_cell.items():
            confusion = np.bincount(
                cell, weights=weight, minlength=N_CLASSES * N_CLASSES,
            ).reshape(N_CLASSES, N_CLASSES)
            delta[key][number] = weighted_macro(confusion) - base_score
    return {
        key: {
            "replicates": int(replicates),
            "seed": int(BOOTSTRAP_SEED),
            "sessions": int(len(unique)),
            "mean_delta": float(values.mean()),
            "p_delta_gt_zero": float(np.mean(values > 0.0)),
            "p05_delta": float(np.quantile(values, 0.05)),
            "p95_delta": float(np.quantile(values, 0.95)),
        }
        for key, values in delta.items()
    }


def validate_full_au_oof(
    root: Path,
    data: dict[str, Any],
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Map the original AU-only five-fold OOF blocks back to exact B43 rows."""
    au_dir = root / "experiments/au_source_specialist"
    align = np.load(au_dir / "alignment.npz", allow_pickle=False)
    ids = full_ids(root)
    au = data["source"] == "au"
    if not (
        np.array_equal(align["ids"].astype(str), ids[au])
        and np.array_equal(align["target"].astype(np.int64), data["y"][au])
        and np.array_equal(align["fold"].astype(np.int8), data["fold"][au])
        and np.array_equal(align["terminal"].astype(bool), data["terminal"][au])
    ):
        raise ValueError("Original AU 5-fold OOF alignment does not match exact B43 rows")

    output: dict[str, np.ndarray] = {}
    for name in ("regularized", "compact"):
        probability = np.load(
            au_dir / f"{name}_oof_probability.npy", allow_pickle=False,
        ).astype(np.float32)
        if probability.shape != (int(au.sum()), N_CLASSES):
            raise ValueError(f"Unexpected {name} AU OOF shape: {probability.shape}")
        if not np.allclose(probability.sum(axis=1), 1.0, rtol=0.0, atol=2.0e-6):
            raise ValueError(f"{name} AU OOF probabilities are not normalized")
        padded = np.zeros_like(data["raw"], dtype=np.float32)
        padded[au] = probability
        output[name] = padded

    original_global = np.load(
        au_dir / "same_raw_oof_probability.npy", allow_pickle=False,
    ).astype(np.float32)
    if original_global.shape != (int(au.sum()), N_CLASSES):
        raise ValueError("Original AU global raw OOF shape does not match")
    return output, {
        "au_rows": int(au.sum()),
        "au_terminal_rows": int(data["terminal"][au].sum()),
        "alignment": "ids, labels, GroupKFold assignment, and terminal mask exactly match B43 AU rows",
        "files": {
            name: {
                "path": str((au_dir / f"{name}_oof_probability.npy").relative_to(root)).replace("\\", "/"),
                "sha256": sha256(au_dir / f"{name}_oof_probability.npy"),
            }
            for name in ("regularized", "compact")
        },
        "original_global_raw_max_abs_difference_from_exact_b43_raw": float(
            np.max(np.abs(original_global - data["raw"][au]))
        ),
    }


def strict_fixed_context(
    root: Path,
    data: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Load the exact matching seed42 fixed feature contract and validate it."""
    x_train, y_train, source_train, x_valid, y_valid, source_valid = strict_context()
    blocks = np.load(root / "experiments/au_source_strict_common/strict_blocks.npz", allow_pickle=False)
    fixed_ids = np.load(
        root / "experiments/b42_9_common_old_noe5_seed42/fixed_valid_ids.npy",
        allow_pickle=False,
    ).astype(str)
    if not (
        np.array_equal(blocks["ids_valid"].astype(str), fixed_ids)
        and np.array_equal(y_valid.astype(np.int64), data["fixed_y"])
        and np.array_equal(source_valid.astype(str), data["fixed_source"].astype(str))
        and len(x_train) == len(y_train) == len(source_train)
        and len(x_valid) == len(y_valid) == len(source_valid)
    ):
        raise ValueError("Strict common-train/fixed context is not exact B43 matching seed42")
    return x_train, y_train, source_train, x_valid, y_valid, source_valid, {
        "common_train_rows": int(len(y_train)),
        "common_train_au_rows": int(np.sum(source_train == "au")),
        "fixed_rows": int(len(y_valid)),
        "fixed_au_rows": int(np.sum(source_valid == "au")),
        "fixed_ids_path": "experiments/b42_9_common_old_noe5_seed42/fixed_valid_ids.npy",
    }


def main() -> None:
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
    full_dual = dual_gru(data["old"], data["strict"])
    fixed_dual = dual_gru(data["fixed_old"], data["fixed_strict"])
    full_base_terminal = terminal_prediction(base_full_probability, data["terminal"], data["char20"])
    fixed_base_terminal = terminal_prediction(
        base_fixed_probability, data["fixed_terminal"], data["fixed_char20"],
    )

    full_probability: dict[str, np.ndarray] = {}
    full_metrics: dict[str, dict[str, float]] = {}
    full_terminal: dict[str, np.ndarray] = {}
    policy_spec: dict[str, tuple[str, float]] = {}
    for config_name, specialist in specialist_full.items():
        for raw_weight in RAW_WEIGHTS:
            key = f"{config_name}|global_raw_weight={raw_weight:.2f}"
            mixed_raw = mix_au_raw(data["raw"], data["source"], specialist, raw_weight)
            probability = geometric(mixed_raw, full_dual, source_alpha(data["source"]))
            full_probability[key] = probability
            full_metrics[key] = metric_bundle(
                data["y"], probability, data["terminal"], data["char20"],
            )
            full_terminal[key] = terminal_prediction(probability, data["terminal"], data["char20"])
            policy_spec[key] = (config_name, raw_weight)

    full_bootstrap = shared_session_bootstrap(
        data["y"][data["terminal"]], full_base_terminal, full_terminal,
        data["groups"][data["terminal"]],
    )
    full_records: dict[str, dict[str, Any]] = {}
    for key in policy_spec:
        delta = metric_delta(full_metrics[key], base_full)
        all_metric_positive = all(delta[name] > 0.0 for name in METRIC_KEYS)
        bootstrap_pass = full_bootstrap[key]["p_delta_gt_zero"] >= PROMOTION_BOOTSTRAP_P
        full_records[key] = {
            "config": policy_spec[key][0],
            "global_raw_weight": policy_spec[key][1],
            "full_outer_oof_metric": full_metrics[key],
            "delta_vs_b43": delta,
            "terminal_char20_session_bootstrap_vs_b43": full_bootstrap[key],
            "full_static_screen": {
                "all_four_metrics_positive": bool(all_metric_positive),
                "bootstrap_p_ge_0_95": bool(bootstrap_pass),
                "pass": bool(all_metric_positive and bootstrap_pass),
            },
        }
    full_survivors = [key for key, value in full_records.items() if value["full_static_screen"]["pass"]]
    print(json.dumps({
        "full_static_screen_complete": True,
        "policies": len(full_records),
        "full_survivors": full_survivors,
        "bootstrap_p": {key: value["terminal_char20_session_bootstrap_vs_b43"]["p_delta_gt_zero"] for key, value in full_records.items()},
    }, ensure_ascii=False), flush=True)

    fixed_records: dict[str, dict[str, Any]] = {}
    fixed_context_info: dict[str, Any] | None = None
    if full_survivors:
        x_train, y_train, source_train, x_valid, y_valid, source_valid, fixed_context_info = strict_fixed_context(root, data)
        fixed_specialist_by_config: dict[str, np.ndarray] = {}
        for key in full_survivors:
            config_name, _ = policy_spec[key]
            if config_name in fixed_specialist_by_config:
                continue
            specialist = np.zeros_like(data["fixed_raw"], dtype=np.float32)
            au_train = source_train == "au"
            au_valid = source_valid == "au"
            if np.any(au_valid):
                specialist[au_valid] = selector_probability(
                    CONFIGS[config_name], x_train[au_train], y_train[au_train], x_valid[au_valid],
                    device="cuda", n_jobs=4,
                )
            if not np.allclose(specialist[au_valid].sum(axis=1), 1.0, rtol=0.0, atol=2.0e-6):
                raise ValueError(f"Fixed {config_name} AU specialist is not normalized")
            fixed_specialist_by_config[config_name] = specialist

        fixed_terminal: dict[str, np.ndarray] = {}
        fixed_metric: dict[str, dict[str, float]] = {}
        fixed_probability: dict[str, np.ndarray] = {}
        for key in full_survivors:
            config_name, raw_weight = policy_spec[key]
            mixed_raw = mix_au_raw(
                data["fixed_raw"], data["fixed_source"], fixed_specialist_by_config[config_name], raw_weight,
            )
            probability = geometric(mixed_raw, fixed_dual, source_alpha(data["fixed_source"]))
            fixed_probability[key] = probability
            fixed_metric[key] = metric_bundle(
                data["fixed_y"], probability, data["fixed_terminal"], data["fixed_char20"],
            )
            fixed_terminal[key] = terminal_prediction(
                probability, data["fixed_terminal"], data["fixed_char20"],
            )
        fixed_bootstrap = shared_session_bootstrap(
            data["fixed_y"][data["fixed_terminal"]], fixed_base_terminal, fixed_terminal,
            data["fixed_groups"][data["fixed_terminal"]],
        )
        for key in full_survivors:
            delta = metric_delta(fixed_metric[key], base_fixed)
            metrics_pass = all(delta[name] > 0.0 for name in METRIC_KEYS)
            bootstrap_pass = fixed_bootstrap[key]["p_delta_gt_zero"] >= PROMOTION_BOOTSTRAP_P
            fixed_records[key] = {
                "matching_seed42_fixed_metric": fixed_metric[key],
                "delta_vs_b43": delta,
                "terminal_char20_session_bootstrap_vs_b43": fixed_bootstrap[key],
                "fixed_static_screen": {
                    "all_four_metrics_positive": bool(metrics_pass),
                    "bootstrap_p_ge_0_95": bool(bootstrap_pass),
                    "pass": bool(metrics_pass and bootstrap_pass),
                },
            }

    fully_eligible = [
        key for key in full_survivors
        if fixed_records.get(key, {}).get("fixed_static_screen", {}).get("pass", False)
    ]
    if fully_eligible:
        recommendation = (
            "STATIC_SCREEN_ONLY: one or more post-hoc policies pass both numerical screens, "
            "but no ZIP may be produced without a predeclared/nested confirmation."
        )
    else:
        recommendation = (
            "PRESERVE_B43: no static dual50+AU policy clears both the full and exact matching "
            "seed42 fixed four-metric and 10k session-bootstrap gates."
        )
    report = {
        "scope": (
            "Post-hoc, static/non-nested deployment-policy screen only. It writes no model/, "
            "script.py, staging directory, or ZIP artifact. Full AU predictions are the original "
            "five-fold OOF blocks, so this screen cannot by itself justify deployment."
        ),
        "candidate_definition": {
            "baseline": "B43: raw seed42 + old no-E5 GRU, B43 geometric SIM raw=.20/AU raw=.85, char20 last.",
            "challenger": "old no-E5=.50 + strict-distill no-E5=.50 arithmetic GRU, then one fixed AU-only selector policy before unchanged geometry and char20 last.",
            "configs": list(CONFIGS),
            "global_raw_weights": list(RAW_WEIGHTS),
            "policy_count": 8,
        },
        "provenance": provenance,
        "original_au_oof_alignment": au_provenance,
        "b43_baseline": {
            "full_outer_oof": base_full,
            "matching_seed42_fixed": base_fixed,
        },
        "full_static_policies": full_records,
        "full_survivors_requiring_fixed_refit": full_survivors,
        "matching_seed42_fixed_context": fixed_context_info,
        "fixed_screen_for_full_survivors": fixed_records,
        "fully_eligible_static_screen_policies": fully_eligible,
        "strict_screen_gate": {
            "requires": {
                "full": "all four deltas strictly positive and terminal-char20 session-bootstrap p(delta>0) >= .95 with 10,000 replicates",
                "fixed": "same gate after a fresh AU-only common-train fit on exact matching seed42 fixed validation",
            },
            "full_policy_count": 8,
            "full_survivor_count": len(full_survivors),
            "fully_eligible_count": len(fully_eligible),
            "recommendation": recommendation,
            "posthoc_not_deployment_approval": True,
        },
        "submission_action": {
            "candidate_created": False,
            "zip_updated": False,
            "live_files_changed": False,
        },
    }
    (OUT / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    readme = "\n".join((
        "# B43 dual50 + fixed AU static deployment audit",
        "",
        "이 결과는 원 AU 5-fold OOF를 사용하는 사후(static/non-nested) 스크린입니다.",
        "model/, script.py, staging directory 및 제출 ZIP은 변경하지 않았습니다.",
        f"Full survivors: {', '.join(full_survivors) if full_survivors else 'none'}.",
        f"Full + fixed static survivors: {', '.join(fully_eligible) if fully_eligible else 'none'}.",
        f"Recommendation: {recommendation}",
        "",
    ))
    (OUT / "README.md").write_text(readme, encoding="utf-8")
    print(json.dumps({
        "report": str(OUT / "report.json"),
        "full_survivors": full_survivors,
        "fully_eligible_static_screen_policies": fully_eligible,
        "recommendation": recommendation,
    }, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
