#!/usr/bin/env python3
"""B43 dual-GRU confidence-gate audit.

This is deliberately an experiment-only file.  It tests a *small fixed*
family of inference-portable policies that decide whether to replace B43's
old no-E5 GRU probability with an arithmetic 50:50 old/strict-distill GRU
probability.  Raw-XGB geometry (SIM=.20, AU=.85), char20-last, and the
future-history metric convention are unchanged.

No policy looks at text, labels, IDs, or future history at inference.  The
only gate inputs are the two already-computed GRU probability vectors:
top-label agreement and their top-1 margins.  This program never touches
``model/``, ``script.py``, or any ZIP file.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.model_selection import GroupKFold

import experiment_b43_adaptive_blend as b43


POLICIES = (
    "base",
    "dual_all",
    "dual_top1_agreement",
    "dual_top1_disagreement",
    "dual_both_margin_ge_005",
    "dual_both_margin_ge_010",
)

POLICY_DESCRIPTIONS = {
    "base": "Exact B43: retain the old no-E5 GRU for every row.",
    "dual_all": "Use arithmetic 50:50 old/strict-distill GRU probabilities for every row.",
    "dual_top1_agreement": "Use the dual only when old and strict GRUs have the same top-1 class.",
    "dual_top1_disagreement": "Use the dual only when old and strict GRUs have different top-1 classes.",
    "dual_both_margin_ge_005": "Use the dual only when both GRU top-1 margins are at least 0.05.",
    "dual_both_margin_ge_010": "Use the dual only when both GRU top-1 margins are at least 0.10.",
}

METRIC_KEYS = (
    "all_fallback_macro_f1",
    "terminal_fallback_macro_f1",
    "terminal_with_char20_macro_f1",
    "hybrid_future_history_plus_char20_macro_f1",
)
BOOTSTRAP_P_REQUIRED = 0.95


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="experiments/b43_dual_confidence_gate")
    parser.add_argument("--bootstrap-replicates", type=int, default=4000)
    return parser.parse_args()


def top_margin(probability: np.ndarray) -> np.ndarray:
    """Top-1 minus top-2 probability without sorting all 14 classes."""

    values = np.partition(probability, -2, axis=1)
    return values[:, -1] - values[:, -2]


def policy_mask(old: np.ndarray, strict: np.ndarray, policy: str) -> np.ndarray:
    if policy not in POLICIES:
        raise ValueError(f"Unknown policy: {policy}")
    if policy == "base":
        return np.zeros(len(old), dtype=bool)
    if policy == "dual_all":
        return np.ones(len(old), dtype=bool)
    agreement = old.argmax(axis=1) == strict.argmax(axis=1)
    if policy == "dual_top1_agreement":
        return agreement
    if policy == "dual_top1_disagreement":
        return ~agreement
    both_margin = (top_margin(old) >= 0.05) & (top_margin(strict) >= 0.05)
    if policy == "dual_both_margin_ge_005":
        return both_margin
    if policy == "dual_both_margin_ge_010":
        return (top_margin(old) >= 0.10) & (top_margin(strict) >= 0.10)
    raise AssertionError(policy)


def policy_probability(
    raw: np.ndarray,
    old: np.ndarray,
    strict: np.ndarray,
    source: np.ndarray,
    policy: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Produce an exact B43 blend after a predeclared dual-GRU gate."""

    mask = policy_mask(old, strict, policy)
    student = old.copy()
    student[mask] = 0.5 * old[mask] + 0.5 * strict[mask]
    probability = b43.geometric(raw, student, b43.source_alpha(source))
    return probability, mask


def metric_delta(candidate: dict[str, float], baseline: dict[str, float]) -> dict[str, float]:
    return {name: float(candidate[name] - baseline[name]) for name in METRIC_KEYS}


def score_policy(
    y: np.ndarray,
    terminal: np.ndarray,
    char20: np.ndarray,
    probabilities: dict[str, np.ndarray],
    mask: np.ndarray,
) -> dict[str, dict[str, float]]:
    return {
        name: b43.metric_bundle(y[mask], probability[mask], terminal[mask], char20[mask])
        for name, probability in probabilities.items()
    }


def pick_policy(rows: dict[str, dict[str, float]]) -> str:
    """Pick exclusively from training labels; favor B43 on an exact tie."""

    return max(
        POLICIES,
        key=lambda name: (
            rows[name]["terminal_with_char20_macro_f1"],
            rows[name]["all_fallback_macro_f1"],
            -POLICIES.index(name),
        ),
    )


def outer_nested_selection(
    y: np.ndarray,
    terminal: np.ndarray,
    char20: np.ndarray,
    fold: np.ndarray,
    probabilities: dict[str, np.ndarray],
) -> dict[str, Any]:
    """Nested held-session selection, with no held-fold label in a decision."""

    selected = np.zeros_like(probabilities["base"])
    choices: list[dict[str, Any]] = []
    for held in sorted(np.unique(fold).tolist()):
        train = fold != held
        held_mask = ~train
        train_rows = score_policy(y, terminal, char20, probabilities, train)
        choice = pick_policy(train_rows)
        selected[held_mask] = probabilities[choice][held_mask]
        choices.append({
            "held_outer_session_fold": int(held) + 1,
            "selected_using_other_four_folds_only": choice,
            "other_four_fold_metrics": train_rows,
            "held_fold_metrics": b43.metric_bundle(
                y[held_mask], probabilities[choice][held_mask], terminal[held_mask], char20[held_mask],
            ),
        })
    count = Counter(row["selected_using_other_four_folds_only"] for row in choices)
    return {
        "selection_rule": (
            "For each held whole-session GroupKFold fold, choose the policy using only "
            "the other four folds' terminal-char20 macro-F1; all-fallback macro-F1 and "
            "the fixed policy order break ties."
        ),
        "choices": choices,
        "selection_counts": {name: int(count.get(name, 0)) for name in POLICIES},
        "selected_probability": selected,
    }


def common_train_index(root: Path, n_rows: int) -> np.ndarray:
    ids = [
        str(json.loads(line)["id"])
        for line in (root / "data/train.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    split = {
        row["id"]: row["split"]
        for row in csv.DictReader((root / "splits/common_split.csv").open(encoding="utf-8"))
    }
    if len(ids) != n_rows:
        raise ValueError("Common-split IDs do not align with B43 rows")
    return np.asarray([number for number, sample_id in enumerate(ids) if split[sample_id] == "train"], dtype=np.int64)


def choose_fixed_policy_inside_common_train(
    y: np.ndarray,
    groups: np.ndarray,
    terminal: np.ndarray,
    char20: np.ndarray,
    probabilities: dict[str, np.ndarray],
) -> dict[str, Any]:
    """Choose the one inference policy for the untouched fixed validation.

    All candidate policies are fixed (they fit no learned gate), but selection
    still uses a five-fold GroupKFold score inside common-train only.  The
    matching fixed labels are never read here.
    """

    fold = np.empty(len(y), dtype=np.int8)
    splitter = GroupKFold(n_splits=5)
    for number, (_, valid) in enumerate(splitter.split(np.zeros(len(y)), y, groups)):
        fold[valid] = number
    oof = {name: np.zeros_like(value) for name, value in probabilities.items()}
    folds: list[dict[str, Any]] = []
    for held in range(5):
        held_mask = fold == held
        train = ~held_mask
        train_rows = score_policy(y, terminal, char20, probabilities, train)
        choice = pick_policy(train_rows)
        for name, probability in probabilities.items():
            oof[name][held_mask] = probability[held_mask]
        folds.append({
            "held_common_train_session_fold": held + 1,
            "selected_using_other_common_train_folds_only": choice,
            "other_four_fold_metrics": train_rows,
            "held_fold_metrics_for_selected": b43.metric_bundle(
                y[held_mask], probabilities[choice][held_mask], terminal[held_mask], char20[held_mask],
            ),
        })
    cv_metrics = {
        name: b43.metric_bundle(y, value, terminal, char20)
        for name, value in oof.items()
    }
    chosen = pick_policy(cv_metrics)
    count = Counter(row["selected_using_other_common_train_folds_only"] for row in folds)
    return {
        "selection_rule": (
            "Evaluate every predeclared policy in five whole-session folds of common-train "
            "only, then choose the highest aggregate held-fold terminal-char20 macro-F1. "
            "No matching-fixed label participates."
        ),
        "per_fold_selection": folds,
        "candidate_common_train_group_oof_metrics": cv_metrics,
        "selected_for_matching_fixed": chosen,
        "per_fold_selection_counts": {name: int(count.get(name, 0)) for name in POLICIES},
    }


def gate(
    full_delta: dict[str, float],
    fixed_delta: dict[str, float],
    full_bootstrap: dict[str, float | int],
    fixed_bootstrap: dict[str, float | int],
    fixed_policy: str,
) -> dict[str, Any]:
    checks = {
        "all_four_full_outer_nested_metrics_strictly_positive": all(
            full_delta[name] > 0.0 for name in METRIC_KEYS
        ),
        "all_four_matching_seed42_fixed_metrics_strictly_positive": all(
            fixed_delta[name] > 0.0 for name in METRIC_KEYS
        ),
        "full_terminal_char20_session_bootstrap_p_ge_0_95": (
            float(full_bootstrap["p_delta_gt_zero"]) >= BOOTSTRAP_P_REQUIRED
        ),
        "fixed_terminal_char20_session_bootstrap_p_ge_0_95": (
            float(fixed_bootstrap["p_delta_gt_zero"]) >= BOOTSTRAP_P_REQUIRED
        ),
        "fixed_deployment_policy_is_nonbase": fixed_policy != "base",
    }
    return {
        "strict_two_protocol_rule": (
            "Any candidate with even one non-positive one of the four full/fixed metrics is rejected."
        ),
        "bootstrap_rule": f"Both terminal-char20 whole-session bootstraps must have p(delta > 0) >= {BOOTSTRAP_P_REQUIRED:.2f}.",
        "checks": checks,
        "promote": bool(all(checks.values())),
    }


def main() -> None:
    args = parse_args()
    if args.bootstrap_replicates <= 0:
        raise ValueError("bootstrap replicates must be positive")
    root = Path(__file__).resolve().parents[1]
    out = Path(args.out_dir)
    if not out.is_absolute():
        out = root / out
    out = out.resolve()
    if out == (root / "model").resolve() or (root / "model").resolve() in out.parents:
        raise ValueError("Experiment output may not be placed inside model/")
    out.mkdir(parents=True, exist_ok=True)

    data, provenance = b43.load_exact_b43(root)
    base_full_probability, _ = policy_probability(
        data["raw"], data["old"], data["strict"], data["source"], "base",
    )
    base_fixed_probability, _ = policy_probability(
        data["fixed_raw"], data["fixed_old"], data["fixed_strict"], data["fixed_source"], "base",
    )
    base_full_metrics = b43.metric_bundle(
        data["y"], base_full_probability, data["terminal"], data["char20"],
    )
    base_fixed_metrics = b43.metric_bundle(
        data["fixed_y"], base_fixed_probability, data["fixed_terminal"], data["fixed_char20"],
    )
    b43.check_baseline(base_full_metrics, b43.EXPECTED_B43["full"], "full")
    b43.check_baseline(base_fixed_metrics, b43.EXPECTED_B43["fixed"], "fixed")

    full_probabilities: dict[str, np.ndarray] = {}
    fixed_probabilities: dict[str, np.ndarray] = {}
    policy_screen: dict[str, Any] = {}
    for name in POLICIES:
        full_probability, full_mask = policy_probability(
            data["raw"], data["old"], data["strict"], data["source"], name,
        )
        fixed_probability, fixed_mask = policy_probability(
            data["fixed_raw"], data["fixed_old"], data["fixed_strict"], data["fixed_source"], name,
        )
        full_probabilities[name] = full_probability
        fixed_probabilities[name] = fixed_probability
        full_metrics = b43.metric_bundle(data["y"], full_probability, data["terminal"], data["char20"])
        fixed_metrics = b43.metric_bundle(
            data["fixed_y"], fixed_probability, data["fixed_terminal"], data["fixed_char20"],
        )
        policy_screen[name] = {
            "description": POLICY_DESCRIPTIONS[name],
            "dual_applied_rows": {
                "full": int(full_mask.sum()),
                "full_terminal": int((full_mask & data["terminal"]).sum()),
                "fixed": int(fixed_mask.sum()),
                "fixed_terminal": int((fixed_mask & data["fixed_terminal"]).sum()),
                "full_sim": int((full_mask & (data["source"] == "sim")).sum()),
                "full_au": int((full_mask & (data["source"] == "au")).sum()),
            },
            "full_static_oof": full_metrics,
            "matching_seed42_fixed_static": fixed_metrics,
            "delta_vs_b43_static": {
                "full": metric_delta(full_metrics, base_full_metrics),
                "fixed": metric_delta(fixed_metrics, base_fixed_metrics),
            },
        }

    nested = outer_nested_selection(
        data["y"], data["terminal"], data["char20"], data["fold"], full_probabilities,
    )
    selected_full_probability = nested.pop("selected_probability")
    nested_full_metrics = b43.metric_bundle(
        data["y"], selected_full_probability, data["terminal"], data["char20"],
    )

    train_index = common_train_index(root, len(data["y"]))
    common_selection = choose_fixed_policy_inside_common_train(
        data["y"][train_index], data["groups"][train_index], data["terminal"][train_index],
        data["char20"][train_index], {name: probability[train_index] for name, probability in full_probabilities.items()},
    )
    fixed_policy = common_selection["selected_for_matching_fixed"]
    selected_fixed_probability = fixed_probabilities[fixed_policy]
    selected_fixed_metrics = b43.metric_bundle(
        data["fixed_y"], selected_fixed_probability, data["fixed_terminal"], data["fixed_char20"],
    )

    base_full_terminal = b43.labels_with_char(
        base_full_probability, data["terminal"], data["char20"],
    )[1][data["terminal"]]
    selected_full_terminal = b43.labels_with_char(
        selected_full_probability, data["terminal"], data["char20"],
    )[1][data["terminal"]]
    base_fixed_terminal = b43.labels_with_char(
        base_fixed_probability, data["fixed_terminal"], data["fixed_char20"],
    )[1][data["fixed_terminal"]]
    selected_fixed_terminal = b43.labels_with_char(
        selected_fixed_probability, data["fixed_terminal"], data["fixed_char20"],
    )[1][data["fixed_terminal"]]
    full_bootstrap = b43.session_bootstrap(
        data["y"][data["terminal"]], base_full_terminal, selected_full_terminal,
        data["groups"][data["terminal"]], args.bootstrap_replicates,
    )
    fixed_bootstrap = b43.session_bootstrap(
        data["fixed_y"][data["fixed_terminal"]], base_fixed_terminal, selected_fixed_terminal,
        data["fixed_groups"][data["fixed_terminal"]], args.bootstrap_replicates,
    )
    full_delta = metric_delta(nested_full_metrics, base_full_metrics)
    fixed_delta = metric_delta(selected_fixed_metrics, base_fixed_metrics)
    promotion = gate(full_delta, fixed_delta, full_bootstrap, fixed_bootstrap, fixed_policy)

    report = {
        "scope": (
            "Experiment-only: no live model, script.py, B42.zip, baseline_submit.zip, "
            "or other submission ZIP was modified."
        ),
        "reference": (
            "Exact B43 = raw seed42 + old no-E5 GRU geometric fusion "
            "(SIM=.20/AU=.85), then immutable char20 terminal overlay."
        ),
        "provenance": provenance,
        "inference_portability": {
            "portable": True,
            "formula": (
                "mask = fixed top-label-agreement or top-margin condition on the two portable "
                "GRU probability vectors; gru = where(mask, 0.5*old + 0.5*strict, old); "
                "p = normalize(raw_xgb**alpha_source * gru**(1-alpha_source)); char20 applies last."
            ),
            "requires": "Existing portable old and strict NumPy GRU forward passes plus scalar comparisons only.",
            "does_not_require": "torch, transformer, labels, IDs, prompt text, test lookup, or future-history input.",
        },
        "validation_contract": {
            "full": "Five whole-session GroupKFold outer selections; every held fold is selected only on its other four folds.",
            "fixed": "One policy is selected with five whole-session folds inside common-train only, then evaluated on untouched matching seed42 fixed labels.",
            "char20": "Immutable and terminal-last in every policy and metric.",
            "bootstrap": "Whole-session bootstrap of terminal labels after char20; base vs selected candidate.",
            "policy_family": "Six predeclared policies only; no learned gate or policy tuning outside these definitions.",
        },
        "baseline_b43": {
            "full_outer_oof": base_full_metrics,
            "matching_seed42_fixed": base_fixed_metrics,
        },
        "policy_static_screen": policy_screen,
        "full_outer_nested_selection": {
            **nested,
            "metrics": nested_full_metrics,
            "delta_vs_b43": full_delta,
        },
        "common_train_selection_for_fixed": common_selection,
        "matching_seed42_fixed": {
            "selected_policy": fixed_policy,
            "metrics": selected_fixed_metrics,
            "delta_vs_b43": fixed_delta,
        },
        "terminal_char20_whole_session_bootstrap": {
            "full_outer_nested": full_bootstrap,
            "matching_seed42_fixed": fixed_bootstrap,
        },
        "promotion_gate": promotion,
        "submission_action": {
            "candidate_created": False,
            "zip_updated": False,
            "reason": (
                "No artifact is promoted by this experiment unless every strict two-protocol "
                "metric and bootstrap requirement passes."
            ),
        },
    }
    (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    readme = "\n".join((
        "# B43 dual confidence-gate audit",
        "",
        "Experiment-only. The live bundle and submission ZIPs are unchanged.",
        "",
        f"Fixed selected policy: `{fixed_policy}`",
        f"Promotion gate passed: `{promotion['promote']}`",
        "",
        "The report records the predeclared policies, outer session-fold selection, common-train-only fixed selection, and terminal session bootstrap.",
        "",
    ))
    (out / "README.md").write_text(readme, encoding="utf-8")
    print(json.dumps({
        "fixed_policy": fixed_policy,
        "full_delta": full_delta,
        "fixed_delta": fixed_delta,
        "promote": promotion["promote"],
        "out": str(out),
    }, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
