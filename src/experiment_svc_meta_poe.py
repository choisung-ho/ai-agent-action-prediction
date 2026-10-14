#!/usr/bin/env python3
"""Strict outer-fold audit of the existing cross-fitted rich-text SVC signal.

The SVC itself is already cross-fitted by whole session in
``portable_text_svc_oof/oof_scores.npz`` and has a common-train-only fixed
artifact for the matching validation rows.  This experiment does *not* train
on a held outer fold's labels: for every outer fold it chooses a compact
temperature/exponent product-of-experts (PoE) combiner on the other four
session folds, applies it to the held fold, and finally evaluates a modal
policy on the untouched common fixed validation.

The learned action model remains raw XGB + portable GRU.  SVC decision margins
are merely an extra independent probability block.  B42_9's word15 terminal
rule and the strict candidate's char20 terminal rule are both applied after
the PoE, exactly as they would be at inference.  No live model or ZIP is
modified.
"""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold


ROOT = Path(__file__).resolve().parents[1]
N_CLASSES = 14
TEMPERATURES = (.5, .75, 1.0, 1.5, 2.0, 3.0, 5.0)
SVC_WEIGHTS = (0.0, .01, .02, .03, .05, .08)


def source_of(sample_id: str) -> str:
    return "au" if str(sample_id).startswith("sess_au_") else "sim"


def session_of(sample_id: str) -> str:
    return str(sample_id).split("-step_", 1)[0]


def probability_from_margin(margin: np.ndarray, temperature: float) -> np.ndarray:
    logits = np.asarray(margin, dtype=np.float64) / float(temperature)
    logits -= logits.max(axis=1, keepdims=True)
    probability = np.exp(logits)
    return (probability / probability.sum(axis=1, keepdims=True)).astype(np.float32)


def geometric(base: np.ndarray, text: np.ndarray, text_weight: float) -> np.ndarray:
    weight = float(text_weight)
    logits = ((1.0 - weight) * np.log(np.maximum(base, 1.0e-12))
              + weight * np.log(np.maximum(text, 1.0e-12)))
    logits -= logits.max(axis=1, keepdims=True)
    probability = np.exp(logits)
    return (probability / probability.sum(axis=1, keepdims=True)).astype(np.float32)


def base_geometric(xgb: np.ndarray, gru: np.ndarray, source: np.ndarray, sim_xgb_weight: float) -> np.ndarray:
    alpha = np.where(source == "au", .85, sim_xgb_weight).astype(np.float32)
    logits = alpha[:, None] * np.log(np.maximum(xgb, 1.0e-12)) + (1.0 - alpha[:, None]) * np.log(np.maximum(gru, 1.0e-12))
    logits -= logits.max(axis=1, keepdims=True)
    probability = np.exp(logits)
    return (probability / probability.sum(axis=1, keepdims=True)).astype(np.float32)


def final_prediction(probability: np.ndarray, terminal_rule: np.ndarray) -> np.ndarray:
    prediction = probability.argmax(axis=1).astype(np.int16)
    mask = terminal_rule >= 0
    prediction[mask] = terminal_rule[mask]
    return prediction


def macro(target: np.ndarray, prediction: np.ndarray, mask: np.ndarray | None = None) -> float:
    if mask is None:
        mask = np.ones(len(target), dtype=bool)
    return float(f1_score(target[mask], prediction[mask], labels=np.arange(N_CLASSES), average="macro", zero_division=0))


def metrics(target: np.ndarray, terminal: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    hybrid = np.where(terminal, prediction, target)
    return {
        "terminal_macro_f1": macro(target, prediction, terminal),
        "hybrid_macro_f1": macro(target, hybrid),
        "all_fallback_macro_f1": macro(target, prediction),
    }


def config_key(temperature: float, weight: float) -> str:
    return f"T{temperature:g}_w{weight:g}"


def candidate_predictions(base: np.ndarray, margin: np.ndarray, rule: np.ndarray) -> dict[str, np.ndarray]:
    values: dict[str, np.ndarray] = {}
    for temperature in TEMPERATURES:
        text = probability_from_margin(margin, temperature)
        for weight in SVC_WEIGHTS:
            values[config_key(temperature, weight)] = final_prediction(geometric(base, text, weight), rule)
    return values


def nested_select(target: np.ndarray, terminal: np.ndarray, groups: np.ndarray,
                  candidates: dict[str, np.ndarray]) -> tuple[np.ndarray, list[dict[str, object]]]:
    """Nested selection: held fold labels never influence its configuration."""
    folds = list(GroupKFold(n_splits=5).split(np.zeros(len(target)), target, groups))
    output = np.empty(len(target), dtype=np.int16)
    choices: list[dict[str, object]] = []
    for fold, (_, valid) in enumerate(folds, 1):
        train_mask = np.ones(len(target), dtype=bool); train_mask[valid] = False; train_mask &= terminal
        ranked = [(macro(target, prediction, train_mask), name, prediction) for name, prediction in candidates.items()]
        # Conservative tie-break: lower SVC exponent, then temperature near 1.
        def ordering(value: tuple[float, str, np.ndarray]) -> tuple[float, float, float]:
            score, name, _ = value
            _, raw = name.split("_w", 1)
            weight = float(raw)
            temperature = float(name[1:].split("_", 1)[0])
            return score, -weight, -abs(np.log(temperature))
        best = max(ranked, key=ordering)
        output[valid] = best[2][valid]
        choices.append({"fold": fold, "config": best[1], "inner_terminal_macro_f1": best[0]})
    return output, choices


def modal_config(choices: list[dict[str, object]]) -> str:
    counts = Counter(str(item["config"]) for item in choices)
    # Same conservative tie break used by nested selection.
    def ordering(item: tuple[str, int]) -> tuple[int, float, float]:
        name, count = item
        _, raw = name.split("_w", 1)
        weight = float(raw)
        temperature = float(name[1:].split("_", 1)[0])
        return count, -weight, -abs(np.log(temperature))
    return max(counts.items(), key=ordering)[0]


def changed_summary(baseline: np.ndarray, candidate: np.ndarray, target: np.ndarray, terminal: np.ndarray) -> dict[str, int]:
    mask = terminal & (candidate != baseline)
    return {
        "terminal_changed": int(mask.sum()),
        "terminal_fixed": int(np.sum(mask & (candidate == target) & (baseline != target))),
        "terminal_harmed": int(np.sum(mask & (candidate != target) & (baseline == target))),
    }


def main() -> None:
    rows = [json.loads(line) for line in (ROOT / "data/train.jsonl").open(encoding="utf-8") if line.strip()]
    labels = {row["id"]: row["action"] for row in csv.DictReader((ROOT / "data/train_labels.csv").open(encoding="utf-8"))}
    classes = (
        "read_file", "grep_search", "list_directory", "glob_pattern", "edit_file", "write_file", "apply_patch",
        "run_bash", "run_tests", "lint_or_typecheck", "ask_user", "plan_task", "web_search", "respond_only",
    )
    to_id = {value: index for index, value in enumerate(classes)}
    target = np.asarray([to_id[labels[str(row["id"])]] for row in rows], dtype=np.int16)
    groups = np.asarray([session_of(str(row["id"])) for row in rows])
    sources = np.asarray([source_of(str(row["id"])) for row in rows])
    split = {row["id"]: row["split"] for row in csv.DictReader((ROOT / "splits/common_split.csv").open(encoding="utf-8"))}
    fixed_index = np.asarray([index for index, row in enumerate(rows) if split[str(row["id"])] == "val"], dtype=int)

    # Reuse already-audited terminal decisions.  word15 exactly mirrors B42_9;
    # char20 is the strict/B43 terminal overlay.  The file was created in a
    # previous experiment only and never touches the live bundle.
    stored = np.load(ROOT / "experiments/state_residual_gate/predictions.npz")
    if not np.array_equal(stored["target"], target) or not np.array_equal(stored["fixed_index"], fixed_index):
        raise ValueError("Stored terminal-rule artifacts do not align")
    terminal = stored["terminal"].astype(bool)
    word15_full = stored["word15_full"].astype(np.int16)
    word15_fixed = stored["word15_fixed"].astype(np.int16)
    char_full = stored["char_full"].astype(np.int16)
    char_fixed = stored["char_fixed"].astype(np.int16)
    fixed_terminal = terminal[fixed_index]

    svc = np.load(ROOT / "experiments/portable_text_svc_oof/oof_scores.npz")
    if not np.array_equal(svc["y"].astype(np.int16), target) or not np.array_equal(svc["terminal"].astype(bool), terminal):
        raise ValueError("SVC OOF target/terminal artifact does not align")
    if not np.array_equal(svc["fixed_ids"].astype(str), np.asarray([str(rows[index]["id"]) for index in fixed_index])):
        raise ValueError("SVC fixed IDs do not align")
    margin_full = svc["text_score"].astype(np.float32)
    margin_fixed = svc["fixed_score"].astype(np.float32)

    raw_full = np.load(ROOT / "experiments/modeling_ak_live_xgb_proxy/oof_probabilities.npy").astype(np.float32)
    strict_raw_full = np.load(ROOT / "experiments/xgb_robust_ensemble/full_oof_raw_seed42.npy").astype(np.float32)
    old_gru_full = np.load(ROOT / "experiments/modeling_ai_sequence_fusion_noe5/oof_probabilities.npy").astype(np.float32)
    strict_gru_full = np.load(ROOT / "experiments/modeling_ai_sequence_nested_distill_e16_i4/oof_probabilities.npy").astype(np.float32)
    old_raw_fixed = np.load(ROOT / "experiments/modeling_ac_tune2/oof_probabilities.npy").astype(np.float32)
    old_gru_fixed = np.load(ROOT / "experiments/modeling_ai_sequence_fusion_fixed_noe5/valid_probabilities.npy").astype(np.float32)
    strict_raw_fixed = np.load(ROOT / "experiments/xgb_robust_ensemble/strict_fixed_raw_seed42.npy").astype(np.float32)
    strict_gru_fixed = np.load(ROOT / "experiments/modeling_ai_sequence_fusion_fixed_strictdistill_w35/valid_probabilities.npy").astype(np.float32)
    if any(value.shape != (len(rows), N_CLASSES) for value in (raw_full, strict_raw_full, old_gru_full, strict_gru_full, margin_full)):
        raise ValueError("Full OOF input shape mismatch")
    if any(value.shape != (len(fixed_index), N_CLASSES) for value in (old_raw_fixed, old_gru_fixed, strict_raw_fixed, strict_gru_fixed, margin_fixed)):
        raise ValueError("Fixed input shape mismatch")

    old_base_full = base_geometric(raw_full, old_gru_full, sources, .20)
    strict_base_full = base_geometric(strict_raw_full, strict_gru_full, sources, .25)
    old_base_fixed = base_geometric(old_raw_fixed, old_gru_fixed, sources[fixed_index], .20)
    strict_base_fixed = base_geometric(strict_raw_fixed, strict_gru_fixed, sources[fixed_index], .25)
    old_baseline_full = final_prediction(old_base_full, word15_full)
    strict_baseline_full = final_prediction(strict_base_full, char_full)
    old_baseline_fixed = final_prediction(old_base_fixed, word15_fixed)
    strict_baseline_fixed = final_prediction(strict_base_fixed, char_fixed)
    # B43/control-oriented variants: retain the B42_9 raw+old-GRU fallback
    # while swapping terminal rules.  Char wins when both high-purity tables
    # fire, matching the newer rule's conservative unanimity contract.
    word_char_full = word15_full.copy(); word_char_full[char_full >= 0] = char_full[char_full >= 0]
    word_char_fixed = word15_fixed.copy(); word_char_fixed[char_fixed >= 0] = char_fixed[char_fixed >= 0]
    old_char_baseline_full = final_prediction(old_base_full, char_full)
    old_char_baseline_fixed = final_prediction(old_base_fixed, char_fixed)
    old_word_char_baseline_full = final_prediction(old_base_full, word_char_full)
    old_word_char_baseline_fixed = final_prediction(old_base_fixed, word_char_fixed)

    print("build_candidate_grid", flush=True)
    old_candidates_full = candidate_predictions(old_base_full, margin_full, word15_full)
    strict_candidates_full = candidate_predictions(strict_base_full, margin_full, char_full)
    old_candidates_fixed = candidate_predictions(old_base_fixed, margin_fixed, word15_fixed)
    strict_candidates_fixed = candidate_predictions(strict_base_fixed, margin_fixed, char_fixed)
    old_char_candidates_full = candidate_predictions(old_base_full, margin_full, char_full)
    old_char_candidates_fixed = candidate_predictions(old_base_fixed, margin_fixed, char_fixed)
    old_word_char_candidates_full = candidate_predictions(old_base_full, margin_full, word_char_full)
    old_word_char_candidates_fixed = candidate_predictions(old_base_fixed, margin_fixed, word_char_fixed)
    print("nested_old", flush=True)
    old_nested, old_choices = nested_select(target, terminal, groups, old_candidates_full)
    print("nested_strict", flush=True)
    strict_nested, strict_choices = nested_select(target, terminal, groups, strict_candidates_full)
    print("nested_old_char20", flush=True)
    old_char_nested, old_char_choices = nested_select(target, terminal, groups, old_char_candidates_full)
    print("nested_old_word15_char20", flush=True)
    old_word_char_nested, old_word_char_choices = nested_select(target, terminal, groups, old_word_char_candidates_full)
    old_modal = modal_config(old_choices)
    strict_modal = modal_config(strict_choices)
    old_char_modal = modal_config(old_char_choices)
    old_word_char_modal = modal_config(old_word_char_choices)

    # A shared policy is useful only if neither reference path gives up its
    # own terminal score.  Select it on the other four outer folds using the
    # lower of the two post-rule terminal scores, not the fixed labels.
    folds = list(GroupKFold(n_splits=5).split(np.zeros(len(target)), target, groups))
    joint = np.empty(len(target), dtype=np.int16); joint_choices: list[dict[str, object]] = []
    for fold, (_, valid) in enumerate(folds, 1):
        train_mask = np.ones(len(target), dtype=bool); train_mask[valid] = False; train_mask &= terminal
        values = []
        for name in old_candidates_full:
            old_score = macro(target, old_candidates_full[name], train_mask)
            strict_score = macro(target, strict_candidates_full[name], train_mask)
            values.append((min(old_score, strict_score), old_score + strict_score, name))
        choice = max(values, key=lambda value: (value[0], value[1], -float(value[2].split("_w", 1)[1])))
        joint[valid] = old_candidates_full[choice[2]][valid]
        joint_choices.append({"fold": fold, "config": choice[2], "inner_old_terminal": choice[1] - macro(target, strict_candidates_full[choice[2]], train_mask), "inner_strict_terminal": macro(target, strict_candidates_full[choice[2]], train_mask)})
    joint_modal = modal_config(joint_choices)

    out = ROOT / "experiments/svc_meta_poe"
    out.mkdir(parents=True, exist_ok=True)
    result = {
        "scope": "Experiment only; no model/, script.py, or ZIP was modified.",
        "protocol": {
            "text": "14 SVC margins are existing five-fold whole-session OOF values; fixed margins come from the saved SVC fit only on common-train.",
            "combiner": "Two-parameter calibrated PoE: P = P(raw-XGB+GRU)^(1-w) * softmax(SVC/T)^w. The grid includes w=0 no-op.",
            "outer_selection": "For each held session fold, T and w are selected exclusively on other folds' terminal labels. Common fixed labels are never selected on.",
            "rules": "B42_9 word15 / strict char20 terminal overrides are applied after the candidate probability exactly as in deployment.",
            "caveat": "This is nested second-level selection over pre-existing cross-fitted component margins; a promotion would still require an additional full-refit runtime check for the 81 MB SVC artifact.",
        },
        "grid": {"temperatures": list(TEMPERATURES), "svc_weights": list(SVC_WEIGHTS)},
        "baseline": {
            "b42_9_old_word15": {"full": metrics(target, terminal, old_baseline_full), "fixed": metrics(target[fixed_index], fixed_terminal, old_baseline_fixed)},
            "b42_9_old_char20": {"full": metrics(target, terminal, old_char_baseline_full), "fixed": metrics(target[fixed_index], fixed_terminal, old_char_baseline_fixed)},
            "b42_9_old_word15_char20": {"full": metrics(target, terminal, old_word_char_baseline_full), "fixed": metrics(target[fixed_index], fixed_terminal, old_word_char_baseline_fixed)},
            "strict_char20": {"full": metrics(target, terminal, strict_baseline_full), "fixed": metrics(target[fixed_index], fixed_terminal, strict_baseline_fixed)},
        },
        "nested": {
            "b42_9_old_word15": {"full": metrics(target, terminal, old_nested), "choices": old_choices, "modal_config": old_modal, "fixed_modal": metrics(target[fixed_index], fixed_terminal, old_candidates_fixed[old_modal]), "fixed_modal_changes": changed_summary(old_baseline_fixed, old_candidates_fixed[old_modal], target[fixed_index], fixed_terminal)},
            "b42_9_old_char20": {"full": metrics(target, terminal, old_char_nested), "choices": old_char_choices, "modal_config": old_char_modal, "fixed_modal": metrics(target[fixed_index], fixed_terminal, old_char_candidates_fixed[old_char_modal]), "fixed_modal_changes": changed_summary(old_char_baseline_fixed, old_char_candidates_fixed[old_char_modal], target[fixed_index], fixed_terminal)},
            "b42_9_old_word15_char20": {"full": metrics(target, terminal, old_word_char_nested), "choices": old_word_char_choices, "modal_config": old_word_char_modal, "fixed_modal": metrics(target[fixed_index], fixed_terminal, old_word_char_candidates_fixed[old_word_char_modal]), "fixed_modal_changes": changed_summary(old_word_char_baseline_fixed, old_word_char_candidates_fixed[old_word_char_modal], target[fixed_index], fixed_terminal)},
            "strict_char20": {"full": metrics(target, terminal, strict_nested), "choices": strict_choices, "modal_config": strict_modal, "fixed_modal": metrics(target[fixed_index], fixed_terminal, strict_candidates_fixed[strict_modal]), "fixed_modal_changes": changed_summary(strict_baseline_fixed, strict_candidates_fixed[strict_modal], target[fixed_index], fixed_terminal)},
            "shared_conservative": {"full_old": metrics(target, terminal, joint), "modal_config": joint_modal, "fixed_old_modal": metrics(target[fixed_index], fixed_terminal, old_candidates_fixed[joint_modal]), "fixed_strict_modal": metrics(target[fixed_index], fixed_terminal, strict_candidates_fixed[joint_modal])},
        },
        "static_diagnostic": {},
    }
    for name in old_candidates_full:
        result["static_diagnostic"][name] = {
            "old": {"full": metrics(target, terminal, old_candidates_full[name]), "fixed": metrics(target[fixed_index], fixed_terminal, old_candidates_fixed[name])},
            "strict": {"full": metrics(target, terminal, strict_candidates_full[name]), "fixed": metrics(target[fixed_index], fixed_terminal, strict_candidates_fixed[name])},
        }
    np.savez_compressed(
        out / "predictions.npz", target=target, terminal=terminal, fixed_index=fixed_index,
        old_baseline_full=old_baseline_full, strict_baseline_full=strict_baseline_full,
        old_baseline_fixed=old_baseline_fixed, strict_baseline_fixed=strict_baseline_fixed,
        old_char_baseline_full=old_char_baseline_full, old_char_baseline_fixed=old_char_baseline_fixed,
        old_word_char_baseline_full=old_word_char_baseline_full, old_word_char_baseline_fixed=old_word_char_baseline_fixed,
        old_nested=old_nested, strict_nested=strict_nested, old_char_nested=old_char_nested,
        old_word_char_nested=old_word_char_nested, joint=joint,
        **{f"old_full_{name}": value for name, value in old_candidates_full.items()},
        **{f"strict_full_{name}": value for name, value in strict_candidates_full.items()},
        **{f"old_fixed_{name}": value for name, value in old_candidates_fixed.items()},
        **{f"old_char_full_{name}": value for name, value in old_char_candidates_full.items()},
        **{f"old_char_fixed_{name}": value for name, value in old_char_candidates_fixed.items()},
        **{f"old_word_char_full_{name}": value for name, value in old_word_char_candidates_full.items()},
        **{f"old_word_char_fixed_{name}": value for name, value in old_word_char_candidates_fixed.items()},
        **{f"strict_fixed_{name}": value for name, value in strict_candidates_fixed.items()},
    )
    (out / "report.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"old_modal": old_modal, "old_char_modal": old_char_modal, "old_word_char_modal": old_word_char_modal, "strict_modal": strict_modal, "shared_modal": joint_modal, "nested": result["nested"]}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
