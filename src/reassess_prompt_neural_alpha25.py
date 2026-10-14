#!/usr/bin/env python3
"""Re-score portable prompt-CNN candidates against the latest raw alpha=.25 gate.

This is deliberately a read-only reassessment of already strict OOF/fixed
probabilities.  It writes one report under ``experiments/`` and never creates
portable weights, changes a live submission asset, or refits a neural model.
The reference is the matching raw-seed42 selector probability used by the
latest XGB audit, geometric sim=.25 / au=.85 strict-GRU mixing, followed by
the existing terminal char20 override.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[1]
N_CLASSES = 14
EPS = 1.0e-12


@dataclass(frozen=True)
class Policy:
    family: str
    base_weight: float = 1.0
    neural_confidence: float = 1.1
    base_confidence: float = 0.0

    def key(self) -> str:
        return (f"{self.family}|base={self.base_weight:.2f}|"
                f"neural_conf={self.neural_confidence:.2f}|base_conf={self.base_confidence:.2f}")

    def complexity(self) -> tuple[int, float, float, float]:
        rank = {"base": 0, "blend": 1, "gate": 2, "direct": 3}[self.family]
        return (-rank, self.base_weight, self.neural_confidence, -self.base_confidence)


def geometric(first: np.ndarray, second: np.ndarray, first_weight: np.ndarray | float) -> np.ndarray:
    weight = np.asarray(first_weight, dtype=np.float64)
    if weight.ndim == 1:
        weight = weight[:, None]
    logits = weight * np.log(np.maximum(first, EPS)) + (1.0 - weight) * np.log(np.maximum(second, EPS))
    logits -= logits.max(axis=1, keepdims=True)
    output = np.exp(logits)
    return (output / output.sum(axis=1, keepdims=True)).astype(np.float32)


def apply_char20(prediction: np.ndarray, terminal: np.ndarray, char20: np.ndarray) -> np.ndarray:
    output = prediction.copy()
    use = terminal & (char20 >= 0)
    output[use] = char20[use]
    return output


def prediction_for_policy(base: np.ndarray, neural: np.ndarray, policy: Policy) -> np.ndarray:
    if policy.family == "base":
        return base.argmax(axis=1).astype(np.int16)
    if policy.family == "direct":
        return neural.argmax(axis=1).astype(np.int16)
    if policy.family == "blend":
        return geometric(base, neural, policy.base_weight).argmax(axis=1).astype(np.int16)
    if policy.family == "gate":
        output = base.argmax(axis=1).astype(np.int16)
        use = (neural.max(axis=1) >= policy.neural_confidence) & (base.max(axis=1) <= policy.base_confidence)
        output[use] = neural.argmax(axis=1).astype(np.int16)[use]
        return output
    raise ValueError(policy.family)


def candidate_policies() -> list[Policy]:
    result = [Policy("base"), Policy("direct")]
    result.extend(Policy("blend", base_weight=value) for value in (0.45, 0.55, 0.65, 0.75, 0.82, 0.88, 0.93))
    result.extend(Policy("gate", neural_confidence=neural_conf, base_confidence=base_conf)
                  for neural_conf in (0.65, 0.72, 0.80, 0.86, 0.90, 0.94)
                  for base_conf in (0.55, 0.62, 0.70, 0.78, 0.86, 0.94))
    return result


def selection_terminal_score(y: np.ndarray, terminal: np.ndarray, base: np.ndarray, neural: np.ndarray,
                             char20: np.ndarray, policy: Policy) -> float:
    prediction = apply_char20(prediction_for_policy(base, neural, policy), terminal, char20)
    return macro(y, prediction, terminal)


def choose_nested(y: np.ndarray, terminal: np.ndarray, base: np.ndarray, neural: np.ndarray, char20: np.ndarray,
                  fold: np.ndarray, policies: list[Policy]) -> tuple[np.ndarray, list[dict], Policy]:
    output = prediction_for_policy(base, neural, Policy("base"))
    chosen: list[Policy] = []
    records: list[dict] = []
    for held in sorted(np.unique(fold).tolist()):
        train = fold != held
        candidates = [(selection_terminal_score(y[train], terminal[train], base[train], neural[train], char20[train], policy),
                       policy.complexity(), policy) for policy in policies]
        score, _, best = max(candidates, key=lambda value: (value[0], value[1]))
        hold = fold == held
        output[hold] = prediction_for_policy(base[hold], neural[hold], best)
        chosen.append(best)
        records.append({"held_outer_fold": int(held + 1), "selected": best.key(), "train_terminal_macro_f1": score})
    counts = Counter(item.key() for item in chosen)
    key = max(counts, key=lambda value: (counts[value], next(item for item in chosen if item.key() == value).complexity()))
    return output, records, next(item for item in chosen if item.key() == key)


def macro(y: np.ndarray, prediction: np.ndarray, mask: np.ndarray | None = None) -> float:
    if mask is None:
        mask = np.ones(len(y), dtype=bool)
    return float(f1_score(y[mask], prediction[mask], labels=np.arange(N_CLASSES), average="macro", zero_division=0))


def source_geometry(xgb: np.ndarray, gru: np.ndarray, source: np.ndarray) -> np.ndarray:
    alpha = np.where(source.astype(str) == "sim", 0.25, 0.85).astype(np.float32)
    return geometric(xgb, gru, alpha)


def metric(y: np.ndarray, terminal: np.ndarray, prediction: np.ndarray, char20: np.ndarray) -> dict[str, float]:
    with_char = apply_char20(prediction, terminal, char20)
    hybrid = with_char.copy(); hybrid[~terminal] = y[~terminal]
    return {
        "all_fallback_macro_f1_with_char20": macro(y, with_char),
        "terminal_fallback_macro_f1_with_char20": macro(y, with_char, terminal),
        "hybrid_future_history_plus_char20_macro_f1": macro(y, hybrid),
    }


def policy_metric(y: np.ndarray, terminal: np.ndarray, base: np.ndarray, neural: np.ndarray,
                  char20: np.ndarray, policy: Policy) -> dict[str, float | int]:
    prediction = prediction_for_policy(base, neural, policy)
    record: dict[str, float | int] = metric(y, terminal, prediction, char20)
    base_prediction = prediction_for_policy(base, base, Policy("base"))
    record["terminal_changed_before_char20"] = int(np.sum(terminal & (prediction != base_prediction)))
    record["terminal_changed_after_char20"] = int(np.sum(
        terminal & (apply_char20(prediction, terminal, char20) != apply_char20(base_prediction, terminal, char20))
    ))
    return record


def run_candidate(name: str, probability_path: Path, base: np.ndarray, fixed_base: np.ndarray,
                  y: np.ndarray, terminal: np.ndarray, char20: np.ndarray, fold: np.ndarray,
                  fixed_y: np.ndarray, fixed_terminal: np.ndarray, fixed_char20: np.ndarray) -> dict:
    saved = np.load(probability_path, allow_pickle=False)
    neural = saved["outer_probability"].astype(np.float32)
    fixed_neural = saved["fixed_probability"].astype(np.float32)
    if not (neural.shape == base.shape == (len(y), N_CLASSES) and fixed_neural.shape == fixed_base.shape == (len(fixed_y), N_CLASSES)):
        raise ValueError(f"{name}: probability shape mismatch")
    if not (np.array_equal(saved["y"].astype(np.int64), y) and np.array_equal(saved["terminal"].astype(bool), terminal)
            and np.array_equal(saved["fold_assignment"].astype(np.int8), fold)
            and np.array_equal(saved["fixed_y"].astype(np.int64), fixed_y)
            and np.array_equal(saved["fixed_terminal"].astype(bool), fixed_terminal)):
        raise ValueError(f"{name}: stored strict alignment contract failed")
    policies = candidate_policies()
    nested, selections, modal = choose_nested(y, terminal, base, neural, char20, fold, policies)
    baseline = policy_metric(y, terminal, base, neural, char20, Policy("base"))
    nested_metric = metric(y, terminal, nested, char20)
    fixed_baseline = policy_metric(fixed_y, fixed_terminal, fixed_base, fixed_neural, fixed_char20, Policy("base"))
    fixed_modal = policy_metric(fixed_y, fixed_terminal, fixed_base, fixed_neural, fixed_char20, modal)
    static_rows = [{"policy": policy.key(), **policy_metric(y, terminal, base, neural, char20, policy)} for policy in policies]
    static_best = max(static_rows, key=lambda row: (
        row["terminal_fallback_macro_f1_with_char20"], row["all_fallback_macro_f1_with_char20"], row["policy"]
    ))
    # A candidate needs strictly better fallback all *and* terminal F1 on both
    # the nested GroupKFold result and untouched matching strict fixed split.
    safe = (
        modal.family != "base"
        and nested_metric["all_fallback_macro_f1_with_char20"] > baseline["all_fallback_macro_f1_with_char20"]
        and nested_metric["terminal_fallback_macro_f1_with_char20"] > baseline["terminal_fallback_macro_f1_with_char20"]
        and fixed_modal["all_fallback_macro_f1_with_char20"] > fixed_baseline["all_fallback_macro_f1_with_char20"]
        and fixed_modal["terminal_fallback_macro_f1_with_char20"] > fixed_baseline["terminal_fallback_macro_f1_with_char20"]
    )
    return {
        "candidate": name,
        "direct_neural": {
            "full_all_macro_f1": macro(y, neural.argmax(axis=1)),
            "full_terminal_macro_f1": macro(y, neural.argmax(axis=1), terminal),
            "fixed_all_macro_f1": macro(fixed_y, fixed_neural.argmax(axis=1)),
            "fixed_terminal_macro_f1": macro(fixed_y, fixed_neural.argmax(axis=1), fixed_terminal),
        },
        "baseline": {"full": baseline, "fixed": fixed_baseline},
        "nested_selection": {"selected_by_held_fold": selections, "modal_policy": modal.key(), **nested_metric},
        "modal_policy_fixed": fixed_modal,
        "best_static_diagnostic_only": static_best,
        "two_split_all_and_terminal_safe": safe,
    }


def main() -> None:
    out = ROOT / "experiments/prompt_neural_alpha25_reassessment"
    out.mkdir(parents=True, exist_ok=True)
    base_artifact = np.load(ROOT / "experiments/terminal_rule_combinations/oof_candidates.npz", allow_pickle=False)
    y = base_artifact["y"].astype(np.int64)
    terminal = base_artifact["terminal"].astype(bool)
    char20 = base_artifact["char20"].astype(np.int16)
    # The char-rule artifact intentionally uses -1 for non-terminal rows
    # because the rule cannot fire there.  The student OOF uses the full
    # GroupKFold assignment, which is the strict nested-GRU assignment below.
    fold = np.load(ROOT / "experiments/modeling_ai_sequence_nested_distill_e16_i4/outer_fold_assignment.npy",
                   allow_pickle=False).astype(np.int8)
    if not np.array_equal(base_artifact["fold"].astype(np.int8)[terminal], fold[terminal]):
        raise ValueError("Char20 terminal fold assignment differs from strict prompt/GRU OOF")
    # raw_seed42 artifacts are the matching selector protocol, unlike the old
    # seed808 fixed file used by a pre-alpha review.
    xgb = np.load(ROOT / "experiments/xgb_robust_ensemble/full_oof_raw_seed42.npy", allow_pickle=False).astype(np.float32)
    gru = np.load(ROOT / "experiments/modeling_ai_sequence_nested_distill_e16_i4/oof_probabilities.npy", allow_pickle=False).astype(np.float32)
    rows = [json.loads(line) for line in (ROOT / "data/train.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    ids = np.asarray([str(row["id"]) for row in rows])
    source = np.asarray(["au" if value.startswith("sess_au_") else "sim" for value in ids])
    if not (xgb.shape == gru.shape == (len(y), N_CLASSES) and len(ids) == len(y)):
        raise ValueError("Full raw/GRU data do not align")
    base = source_geometry(xgb, gru, source)

    strict = np.load(ROOT / "experiments/au_source_strict_common/strict_blocks.npz", allow_pickle=False)
    fixed_y = strict["target_valid"].astype(np.int64)
    fixed_source = strict["source_valid"].astype(str)
    fixed_ids = strict["ids_valid"].astype(str)
    fixed_xgb = np.load(ROOT / "experiments/xgb_robust_ensemble/strict_fixed_raw_seed42.npy", allow_pickle=False).astype(np.float32)
    fixed_gru = np.load(ROOT / "experiments/modeling_ai_sequence_fusion_fixed_strictdistill_w35/valid_probabilities.npy", allow_pickle=False).astype(np.float32)
    if not (fixed_xgb.shape == fixed_gru.shape == (len(fixed_y), N_CLASSES)):
        raise ValueError("Fixed raw/GRU data do not align")
    fixed_base = source_geometry(fixed_xgb, fixed_gru, fixed_source)
    # Fixed char20 artifact stores only terminal rows and asserts its own order.
    maximum: dict[str, int] = {}
    parsed: list[tuple[str, int]] = []
    for value in fixed_ids:
        session, step = value.rsplit("-step_", 1); step_int = int(step)
        parsed.append((session, step_int)); maximum[session] = max(maximum.get(session, -1), step_int)
    fixed_terminal = np.asarray([step == maximum[session] for session, step in parsed], dtype=bool)
    fixed_char20 = np.full(len(fixed_y), -1, dtype=np.int16)
    if not np.array_equal(base_artifact["valid_ids"].astype(str), fixed_ids[fixed_terminal]):
        raise ValueError("Fixed char20 IDs do not align with matching strict selector")
    fixed_char20[fixed_terminal] = base_artifact["fixed_char20"].astype(np.int16)

    report = {
        "scope": "Read-only reassessment; no model/script/ZIP/portable weights are modified.",
        "reference": {
            "raw_selector": "xgb_robust_ensemble raw_seed42 matching full OOF and strict fixed artifacts",
            "geometric_source_alpha": {"sim": 0.25, "au": 0.85},
            "strict_gru": "strict nested-distilled no-E5; teacher/E5 absent at inference",
            "terminal_rule_order": "candidate prediction -> char20 final override -> future-history only for hybrid metric",
        },
        "candidates": [
            run_candidate("supervised_prompt_cnn", ROOT / "experiments/prompt_neural_portable/probabilities.npz",
                          base, fixed_base, y, terminal, char20, fold, fixed_y, fixed_terminal, fixed_char20),
            run_candidate("strict_teacher_distill20_prompt_cnn", ROOT / "experiments/prompt_neural_portable_distill20/probabilities.npz",
                          base, fixed_base, y, terminal, char20, fold, fixed_y, fixed_terminal, fixed_char20),
        ],
    }
    report["conclusion"] = (
        "No prompt-CNN candidate may be promoted unless two_split_all_and_terminal_safe is true. "
        "Both candidates were recalculated from fixed saved probabilities; no fixed labels selected a policy."
    )
    (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
