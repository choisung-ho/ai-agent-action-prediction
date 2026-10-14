#!/usr/bin/env python3
"""Leakage-controlled ensemble recheck after public-score feedback.

This is deliberately an experiment-only audit.  It never reads or writes the
live ``model/`` directory, ``script.py`` or any submission ZIP.  The fixed
validation selector is *always* the matching 2032-feature seed-42 selector,
not the legacy 2025-feature table used by some early experiments.

The report compares:

* the portable original no-E5 GRU used by B42_9;
* the portable strictly nested-distilled no-E5 GRU used by B42_10;
* small, explicitly portable two-GRU mixtures; and
* source-conditioned arithmetic or geometric raw-XGB fusion.

Both the historical word15 overlay and char20 are rebuilt/evaluated only from
outer-train sessions.  A candidate is emitted only if it clears the declared
two-protocol gate against the reconstructed historical baseline.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import f1_score

from experiment_strict_blend_calibration import EPS, load_data
from experiment_terminal_prompt_grid import build_counts, decide, make_rule, word_features


N_CLASSES = 14
BOOTSTRAP_SEED = 619
BOOTSTRAP_REPLICATES = 1000


@dataclass(frozen=True)
class Student:
    """One inference-safe GRU probability source or fixed mixture."""

    name: str
    full: np.ndarray
    fixed: np.ndarray
    contract: dict[str, Any]


def macro(y: np.ndarray, prediction: np.ndarray, mask: np.ndarray | None = None) -> float:
    if mask is None:
        mask = np.ones(len(y), dtype=bool)
    return float(
        f1_score(
            y[mask], prediction[mask], labels=np.arange(N_CLASSES),
            average="macro", zero_division=0,
        )
    )


def normalized_geometric(first: np.ndarray, second: np.ndarray, first_weight: float) -> np.ndarray:
    logits = (
        first_weight * np.log(np.maximum(first, EPS))
        + (1.0 - first_weight) * np.log(np.maximum(second, EPS))
    )
    logits -= logits.max(axis=1, keepdims=True)
    result = np.exp(logits)
    return (result / result.sum(axis=1, keepdims=True)).astype(np.float32)


def source_blend(
    xgb: np.ndarray,
    gru: np.ndarray,
    source: np.ndarray,
    sim_alpha: float,
    au_alpha: float,
    mode: str,
) -> np.ndarray:
    alpha = np.where(source == "sim", sim_alpha, au_alpha).astype(np.float64)[:, None]
    if mode == "linear":
        return (alpha * xgb + (1.0 - alpha) * gru).astype(np.float32)
    if mode != "geometric":
        raise ValueError(mode)
    logits = alpha * np.log(np.maximum(xgb, EPS)) + (1.0 - alpha) * np.log(np.maximum(gru, EPS))
    logits -= logits.max(axis=1, keepdims=True)
    result = np.exp(logits)
    return (result / result.sum(axis=1, keepdims=True)).astype(np.float32)


def overlay(prediction: np.ndarray, terminal: np.ndarray, rule: np.ndarray) -> np.ndarray:
    result = prediction.copy()
    mask = terminal & (rule >= 0)
    result[mask] = rule[mask]
    return result


def metrics(y: np.ndarray, probability: np.ndarray, terminal: np.ndarray, rule: np.ndarray) -> dict[str, float]:
    fallback = probability.argmax(axis=1)
    ruled = overlay(fallback, terminal, rule)
    hybrid = ruled.copy()
    hybrid[~terminal] = y[~terminal]
    return {
        "all_fallback_macro_f1": macro(y, fallback),
        "terminal_fallback_macro_f1": macro(y, fallback, terminal),
        "terminal_with_rule_macro_f1": macro(y, ruled, terminal),
        "hybrid_future_history_plus_rule_macro_f1": macro(y, hybrid),
    }


def historical_word15(
    root: Path,
    data: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Rebuild B42_9's word15 table inside each allowed train partition.

    The B42_9 production table was fit on all rows.  For evaluation we rebuild
    it in the four outer-train session folds, and once in common-train, so no
    rule can see a held row's label.  This is the exact bounded unicode-word
    1--4 gram family used by the historical bundle.
    """

    import csv

    rows = [
        json.loads(line)
        for line in (root / "data/train.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    labels = {
        row["id"]: row["action"]
        for row in csv.DictReader((root / "data/train_labels.csv").open(encoding="utf-8"))
    }
    classes = [
        "read_file", "grep_search", "list_directory", "glob_pattern", "edit_file", "write_file", "apply_patch",
        "run_bash", "run_tests", "lint_or_typecheck", "ask_user", "plan_task", "web_search", "respond_only",
    ]
    y = np.asarray([classes.index(labels[str(row["id"])]) for row in rows], dtype=np.int64)
    if not np.array_equal(y, data["y"]):
        raise ValueError("Word15 labels are not aligned to strict ensemble artifacts")
    groups = np.asarray([str(row["id"]).rsplit("-step_", 1)[0] for row in rows])
    features = [word_features(row) for row in rows]
    terminal = data["terminal"]
    fold = data["fold"]
    output = np.full(len(rows), -1, dtype=np.int16)
    rule_sizes: list[int] = []
    for held in sorted(np.unique(fold).tolist()):
        train = np.flatnonzero(fold != held)
        valid_terminal = np.flatnonzero((fold == held) & terminal)
        rule = make_rule(build_counts(features, y, groups, train, session_support=False), 15)
        output[valid_terminal] = np.asarray(
            [decide(features[int(index)], rule) for index in valid_terminal], dtype=np.int16
        )
        rule_sizes.append(len(rule))

    train_index = data["train_index"]
    valid_index = data["valid_index"]
    fixed_terminal = data["fixed_terminal"]
    fixed_terminal_index = valid_index[fixed_terminal]
    fixed_rule = make_rule(build_counts(features, y, groups, train_index, session_support=False), 15)
    fixed_output = np.full(len(valid_index), -1, dtype=np.int16)
    fixed_output[fixed_terminal] = np.asarray(
        [decide(features[int(index)], fixed_rule) for index in fixed_terminal_index], dtype=np.int16
    )
    return output, fixed_output, {
        "family": "bounded_unicode_word_1to4",
        "support": 15,
        "session_support": False,
        "outer_rule_sizes": rule_sizes,
        "common_train_rule_size": len(fixed_rule),
        "full_terminal_fired": int(np.sum(output[terminal] >= 0)),
        "fixed_terminal_fired": int(np.sum(fixed_output[fixed_terminal] >= 0)),
    }


def build_students(data: dict[str, Any]) -> list[Student]:
    old = data["old_gru"]
    strict = data["gru"]
    old_fixed = data["fixed_old_gru"]
    strict_fixed = data["fixed_gru"]
    result = [
        Student(
            "old_noe5",
            old,
            old_fixed,
            {
                "kind": "single_numpy_gru",
                "weights": "experiments/modeling_ai_sequence_fusion_noe5_full/weights.npz",
            },
        ),
        Student(
            "strict_distill_noe5",
            strict,
            strict_fixed,
            {
                "kind": "single_numpy_gru",
                "weights": "experiments/modeling_ai_sequence_fusion_noe5_strictdistill035_full/weights.npz",
            },
        ),
    ]
    # These are fixed symmetric forward-pass mixtures.  They require two
    # existing portable NumPy GRU evaluations but no encoder, torch, or label
    # lookup at inference.  We deliberately keep the grid small.
    for strict_weight in (0.25, 0.50, 0.75):
        old_weight = 1.0 - strict_weight
        result.append(
            Student(
                f"dual_arithmetic_old{old_weight:g}_strict{strict_weight:g}",
                (old_weight * old + strict_weight * strict).astype(np.float32),
                (old_weight * old_fixed + strict_weight * strict_fixed).astype(np.float32),
                {
                    "kind": "dual_numpy_gru_arithmetic_probability",
                    "old_weight": old_weight,
                    "strict_weight": strict_weight,
                    "old_weights": "experiments/modeling_ai_sequence_fusion_noe5_full/weights.npz",
                    "strict_weights": "experiments/modeling_ai_sequence_fusion_noe5_strictdistill035_full/weights.npz",
                },
            )
        )
        result.append(
            Student(
                f"dual_geometric_old{old_weight:g}_strict{strict_weight:g}",
                normalized_geometric(old, strict, old_weight),
                normalized_geometric(old_fixed, strict_fixed, old_weight),
                {
                    "kind": "dual_numpy_gru_geometric_probability",
                    "old_weight": old_weight,
                    "strict_weight": strict_weight,
                    "old_weights": "experiments/modeling_ai_sequence_fusion_noe5_full/weights.npz",
                    "strict_weights": "experiments/modeling_ai_sequence_fusion_noe5_strictdistill035_full/weights.npz",
                },
            )
        )
    return result


def bootstrap_probability(
    y: np.ndarray,
    baseline: np.ndarray,
    candidate: np.ndarray,
    groups: np.ndarray,
) -> dict[str, float]:
    """Whole-session bootstrap for the terminal overlay delta."""

    unique, inverse = np.unique(groups, return_inverse=True)
    positions = [np.flatnonzero(inverse == index) for index in range(len(unique))]
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    deltas = np.empty(BOOTSTRAP_REPLICATES, dtype=np.float64)
    for number in range(BOOTSTRAP_REPLICATES):
        chosen = rng.integers(0, len(unique), size=len(unique))
        index = np.concatenate([positions[int(value)] for value in chosen])
        deltas[number] = macro(y[index], candidate[index]) - macro(y[index], baseline[index])
    return {
        "replicates": BOOTSTRAP_REPLICATES,
        "seed": BOOTSTRAP_SEED,
        "mean_delta": float(deltas.mean()),
        "p_delta_gt_zero": float(np.mean(deltas > 0.0)),
        "p05_delta": float(np.quantile(deltas, 0.05)),
        "p95_delta": float(np.quantile(deltas, 0.95)),
    }


def nested_support(
    y: np.ndarray,
    terminal: np.ndarray,
    fold: np.ndarray,
    rows: list[dict[str, Any]],
    rule: np.ndarray,
) -> dict[str, Any]:
    """Select a candidate only on the other four session folds."""

    choices: list[dict[str, Any]] = []
    selected_fallback = np.empty(len(y), dtype=np.int16)
    selected_terminal = np.empty(len(y), dtype=np.int16)
    for held in sorted(np.unique(fold).tolist()):
        train = fold != held
        held_mask = fold == held
        scored: list[tuple[float, float, str, dict[str, Any]]] = []
        for row in rows:
            fallback = row["full_fallback"]
            ruled = overlay(fallback, terminal, rule)
            scored.append((
                macro(y, ruled, terminal & train),
                macro(y, fallback, train),
                row["key"],
                row,
            ))
        _, _, key, chosen = max(scored, key=lambda item: (item[0], item[1], item[2]))
        selected_fallback[held_mask] = chosen["full_fallback"][held_mask]
        selected_terminal[held_mask] = overlay(chosen["full_fallback"], terminal, rule)[held_mask]
        choices.append({"held_fold": int(held + 1), "selected": key})
    hybrid = selected_terminal.copy()
    hybrid[~terminal] = y[~terminal]
    counts: dict[str, int] = {}
    for choice in choices:
        counts[choice["selected"]] = counts.get(choice["selected"], 0) + 1
    return {
        "fold_choices": choices,
        "selection_count": counts,
        "all_fallback_macro_f1": macro(y, selected_fallback),
        "terminal_with_word15_macro_f1": macro(y, selected_terminal, terminal),
        "hybrid_macro_f1": macro(y, hybrid),
    }


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    out = root / "experiments/public_feedback_ensemble_recheck"
    out.mkdir(parents=True, exist_ok=True)
    data = load_data(root)
    xgb = np.load(root / "experiments/xgb_robust_ensemble/full_oof_raw_seed42.npy", allow_pickle=False).astype(np.float32)
    fixed_xgb = np.load(root / "experiments/xgb_robust_ensemble/strict_fixed_raw_seed42.npy", allow_pickle=False).astype(np.float32)
    if xgb.shape != data["gru"].shape or fixed_xgb.shape != data["fixed_gru"].shape:
        raise ValueError("Matching seed-42 selector artifacts do not align with GRU artifacts")

    y = data["y"]
    source = data["source"]
    terminal = data["terminal"]
    valid_index = data["valid_index"]
    fixed_y = y[valid_index]
    fixed_source = source[valid_index]
    fixed_terminal = data["fixed_terminal"]
    word15, fixed_word15, word15_info = historical_word15(root, data)
    char20 = data["char"]
    fixed_char20 = data["fixed_char"]
    rules = {
        "historical_word15": (word15, fixed_word15),
        "char20": (char20, fixed_char20),
    }

    alpha_grid = ((0.20, 0.85), (0.25, 0.85), (0.30, 0.85), (0.25, 0.80), (0.25, 0.90))
    records: list[dict[str, Any]] = []
    for student in build_students(data):
        for blend_mode in ("geometric", "linear"):
            for sim_alpha, au_alpha in alpha_grid:
                full_probability = source_blend(xgb, student.full, source, sim_alpha, au_alpha, blend_mode)
                fixed_probability = source_blend(
                    fixed_xgb, student.fixed, fixed_source, sim_alpha, au_alpha, blend_mode
                )
                record: dict[str, Any] = {
                    "key": f"{student.name}|{blend_mode}|sim={sim_alpha:.2f}|au={au_alpha:.2f}",
                    "student": student.name,
                    "student_contract": student.contract,
                    "raw_xgb_blend": {
                        "mode": blend_mode,
                        "sim_alpha": sim_alpha,
                        "au_alpha": au_alpha,
                    },
                    # Stored only in-memory / omitted from JSON below; needed
                    # to calculate leakage-safe nested support.
                    "full_fallback": full_probability.argmax(axis=1).astype(np.int16),
                }
                for rule_name, (rule, fixed_rule) in rules.items():
                    record[rule_name] = {
                        "full": metrics(y, full_probability, terminal, rule),
                        "fixed": metrics(fixed_y, fixed_probability, fixed_terminal, fixed_rule),
                    }
                records.append(record)

    baseline = next(
        item for item in records
        if item["student"] == "old_noe5"
        and item["raw_xgb_blend"] == {"mode": "geometric", "sim_alpha": 0.20, "au_alpha": 0.85}
    )
    required = ("all_fallback_macro_f1", "terminal_with_rule_macro_f1")
    passing: list[dict[str, Any]] = []
    for record in records:
        delta = {
            scope: {
                metric: record["historical_word15"][scope][metric] - baseline["historical_word15"][scope][metric]
                for metric in required
            }
            for scope in ("full", "fixed")
        }
        record["delta_vs_historical_word15_baseline"] = delta
        if all(delta[scope][metric] > 0.0 for scope in ("full", "fixed") for metric in required):
            passing.append(record)

    nested = nested_support(y, terminal, data["fold"], records, word15)
    # The global choice uses only held-out session OOF evidence.  It must also
    # be selected by at least three of the five nested outer selections; the
    # matching seed-42 fixed validation remains an independent promotion gate.
    stable_keys = {
        key for key, count in nested["selection_count"].items()
        if count >= 3
    }
    stable_passing = [record for record in passing if record["key"] in stable_keys]
    chosen: dict[str, Any] | None = None
    if stable_passing:
        chosen = max(
            stable_passing,
            key=lambda item: (
                item["historical_word15"]["full"]["terminal_with_rule_macro_f1"],
                item["historical_word15"]["full"]["all_fallback_macro_f1"],
                min(item["delta_vs_historical_word15_baseline"]["fixed"].values()),
                item["key"],
            ),
        )
    manifest: dict[str, Any] | None = None
    if chosen is not None:
        baseline_full = overlay(baseline["full_fallback"], terminal, word15)
        candidate_full = overlay(chosen["full_fallback"], terminal, word15)
        baseline_fixed_probability = source_blend(
            fixed_xgb, data["fixed_old_gru"], fixed_source, 0.20, 0.85, "geometric"
        )
        candidate_fixed_probability = source_blend(
            fixed_xgb,
            next(student.fixed for student in build_students(data) if student.name == chosen["student"]),
            fixed_source,
            chosen["raw_xgb_blend"]["sim_alpha"],
            chosen["raw_xgb_blend"]["au_alpha"],
            chosen["raw_xgb_blend"]["mode"],
        )
        baseline_fixed = overlay(baseline_fixed_probability.argmax(axis=1), fixed_terminal, fixed_word15)
        candidate_fixed = overlay(candidate_fixed_probability.argmax(axis=1), fixed_terminal, fixed_word15)
        full_bootstrap = bootstrap_probability(y[terminal], baseline_full[terminal], candidate_full[terminal], data["groups"][terminal])
        fixed_bootstrap = bootstrap_probability(
            fixed_y[fixed_terminal], baseline_fixed[fixed_terminal], candidate_fixed[fixed_terminal],
            data["groups"][valid_index][fixed_terminal],
        )
        selected_folds = nested["selection_count"].get(chosen["key"], 0)
        robust = (
            full_bootstrap["p_delta_gt_zero"] >= 0.95
            and fixed_bootstrap["p_delta_gt_zero"] >= 0.95
            and selected_folds >= 3
        )
        chosen["bootstrap_terminal_word15"] = {"full": full_bootstrap, "fixed": fixed_bootstrap}
        chosen["outer_selection_support"] = {"selected_folds": selected_folds, "required_minimum": 3}
        chosen["robustly_dominant"] = robust
        if robust:
            contract = chosen["student_contract"]
            if contract["kind"] == "single_numpy_gru":
                formula = "p = normalize(raw_xgb ** alpha_source * portable_gru_probability ** (1-alpha_source))"
                runtime = "one existing NumPy GRU forward pass; no torch, transformer, training data, or label lookup at inference"
                script_change = "inject the selected single GRU weights into an experimental candidate bundle; retain B42_9 word15 and final future-history recovery"
            else:
                formula = (
                    f"gru_mix = {contract['old_weight']:.2f} * old_noe5_probability + "
                    f"{contract['strict_weight']:.2f} * strict_distill_noe5_probability; "
                    "p = normalize(raw_xgb ** alpha_source * gru_mix ** (1-alpha_source))"
                )
                runtime = "two existing NumPy GRU forward passes; no torch, transformer, training data, or label lookup at inference"
                script_change = "add an experimental dual-GRU branch; do not overwrite live script.py until smoke/pre-submit validation succeeds"
            manifest = {
                "format": "dual_portable_noe5_candidate_v1",
                "decision": "candidate_only_not_live",
                "base_bundle": "model_candidate_ngram_raw_xgb_gru_geometric/best_model.pkl",
                "historical_rule": "retain the embedded B42_9 word15 prompt_ngram_rule_v1; apply only at supplied session maximum; exact future-history recovery must remain final",
                "student": chosen["student_contract"],
                "raw_xgb_blend": chosen["raw_xgb_blend"],
                "integration_contract": {
                    "formula": formula,
                    "sources": {"sim_xgb_alpha": chosen["raw_xgb_blend"]["sim_alpha"], "au_xgb_alpha": chosen["raw_xgb_blend"]["au_alpha"]},
                    "runtime": runtime,
                    "required_script_change": script_change,
                },
                "evidence": {
                    "full": chosen["historical_word15"]["full"],
                    "fixed_seed42": chosen["historical_word15"]["fixed"],
                    "delta_vs_historical": chosen["delta_vs_historical_word15_baseline"],
                    "bootstrap_terminal_word15": chosen["bootstrap_terminal_word15"],
                },
            }

    def serializable(record: dict[str, Any]) -> dict[str, Any]:
        return {key: value for key, value in record.items() if key != "full_fallback"}

    report = {
        "scope": "Experiment-only recheck after public feedback; model/, script.py and submission ZIP files were not modified.",
        "public_feedback_context": {
            "previous_public_score": 0.7687709901,
            "current_public_score": 0.7666369512,
            "interpretation": "The historical and current archives differ in GRU, prompt rule, and SIM alpha, so public feedback alone cannot attribute the decrease to one component.",
        },
        "validation_contract": {
            "full": "Existing whole-session GroupKFold OOF probabilities; rules are rebuilt within outer-train sessions.",
            "fixed": "Matching seed-42 2032-feature raw selector fitted on common-train only; no legacy 2025 fixed selector used.",
            "selector_artifacts": {
                "full": "experiments/xgb_robust_ensemble/full_oof_raw_seed42.npy",
                "fixed": "experiments/xgb_robust_ensemble/strict_fixed_raw_seed42.npy",
            },
            "word15": word15_info,
        },
        "historical_baseline": serializable(baseline),
        "all_candidates": [serializable(record) for record in records],
        "two_protocol_passing_candidates": [serializable(record) for record in passing],
        "nested_full_oof_support": nested,
        "nested_stability_gate": {"required_outer_fold_selections": 3, "stable_keys": sorted(stable_keys)},
        "selected_candidate": serializable(chosen) if chosen is not None else None,
        "candidate_manifest_emitted": manifest is not None,
        "recommendation": (
            "A candidate manifest was emitted only after the two-protocol and bootstrap gates."
            if manifest is not None
            else "No candidate cleared every two-protocol/bootstrap/stability gate; retain the historical archive as the safe rollback."
        ),
    }
    (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if manifest is not None:
        (out / "candidate_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary = {
        "baseline_word15": baseline["historical_word15"],
        "n_candidates": len(records),
        "n_two_protocol_passing": len(passing),
        "selected": chosen["key"] if chosen else None,
        "manifest_emitted": manifest is not None,
    }
    (out / "README.md").write_text(
        "# Public-feedback ensemble recheck\n\n"
        "This audit leaves live submission assets untouched.  `report.json` contains all candidates; "
        "`candidate_manifest.json` exists only if the declared gate passed.\n\n```json\n"
        + json.dumps(summary, ensure_ascii=False, indent=2)
        + "\n```\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
