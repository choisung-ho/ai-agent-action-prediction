#!/usr/bin/env python3
"""Session-safe semantic reranking audit for B43.

This experiment examines a deliberately narrow direction that is absent from
the deployed selector: a separate lexical classifier is trained *only* for
the action families which dominate B43's remaining terminal errors.  It never
uses IDs, future labels, or a test-row lookup.  The classifier is used only to
rerank a B43 decision when B43's own top two labels already belong to that
same family.  Existing char20 and future-history rules remain downstream and
immutable.

All outputs stay under ``experiments/b43_group_rerank``.  In particular this
program does not write model/, script.py, B42.zip, or baseline_submit.zip.
"""

from __future__ import annotations

import csv
import gc
import json
import time
from pathlib import Path

import numpy as np
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments" / "b43_group_rerank"
CLASSES = (
    "read_file", "grep_search", "list_directory", "glob_pattern",
    "edit_file", "write_file", "apply_patch", "run_bash", "run_tests",
    "lint_or_typecheck", "ask_user", "plan_task", "web_search", "respond_only",
)
TO_ID = {label: index for index, label in enumerate(CLASSES)}
N_CLASSES = len(CLASSES)
EPS = 1.0e-12

# Chosen before model fitting from the B43 terminal confusion matrix.  These
# are intentionally small action families; modification/exploration are not
# included because their high B43 confidence makes an intervention risky.
FAMILIES = {
    "quality": np.asarray([TO_ID[value] for value in ("run_bash", "run_tests", "lint_or_typecheck")], dtype=np.int16),
    "coordinate": np.asarray([TO_ID[value] for value in ("ask_user", "plan_task", "web_search")], dtype=np.int16),
}

# A compact finite policy family.  The fixed split is never used to choose one
# of these weights; selection is nested across the full session OOF only.
POLICIES = {
    "base": {},
    "quality_w80": {"quality": 0.80},
    "quality_w90": {"quality": 0.90},
    "coordinate_w90": {"coordinate": 0.90},
    "coordinate_w95": {"coordinate": 0.95},
    "quality80_coordinate90": {"quality": 0.80, "coordinate": 0.90},
    "quality80_coordinate95": {"quality": 0.80, "coordinate": 0.95},
    "quality90_coordinate90": {"quality": 0.90, "coordinate": 0.90},
}


def session_of(sample_id: str) -> str:
    return sample_id.rsplit("-step_", 1)[0]


def step_of(sample_id: str) -> int:
    return int(sample_id.rsplit("-step_", 1)[1])


def terminal_mask(ids: np.ndarray) -> np.ndarray:
    maximum: dict[str, int] = {}
    parsed: list[tuple[str, int]] = []
    for sample_id in ids.astype(str):
        session = session_of(sample_id)
        step = step_of(sample_id)
        maximum[session] = max(maximum.get(session, -1), step)
        parsed.append((session, step))
    return np.asarray([step == maximum[session] for session, step in parsed], dtype=bool)


def source_of(ids: np.ndarray) -> np.ndarray:
    return np.asarray(["au" if value.startswith("sess_au_") else "sim" for value in ids.astype(str)])


def compact(value: object, limit: int) -> str:
    return " ".join(str(value or "").replace("\x00", " ").split())[:limit]


def rich_text(sample: dict) -> str:
    """Current request plus only earlier, inference-visible conversation state."""
    parts = ["CUR " + compact(sample.get("current_prompt"), 1000)]
    history = sample.get("history") if isinstance(sample.get("history"), list) else []
    users = [item for item in history if isinstance(item, dict) and item.get("role") == "user"]
    actions = [item for item in history if isinstance(item, dict) and item.get("role") == "assistant_action"]
    for number, item in enumerate(reversed(users[-3:]), 1):
        parts.append(f"USR{number} " + compact(item.get("content"), 260))
    for number, item in enumerate(reversed(actions[-4:]), 1):
        name = compact(item.get("name") or item.get("action"), 64)
        result = compact(item.get("result_summary"), 130)
        args = item.get("args") if isinstance(item.get("args"), dict) else {}
        args_text = " ".join(f"{key} {compact(value, 80)}" for key, value in sorted(args.items()))
        parts.extend((f"ACT{number} {name}", f"ARG{number} {args_text}", f"RES{number} {result}"))
    meta = sample.get("session_meta") if isinstance(sample.get("session_meta"), dict) else {}
    workspace = meta.get("workspace") if isinstance(meta.get("workspace"), dict) else {}
    parts.append(
        "META " + " ".join((
            compact(meta.get("user_tier"), 24), compact(meta.get("language_pref"), 24),
            compact(workspace.get("last_ci_status"), 24), str(workspace.get("git_dirty", "")),
            " ".join(compact(value, 80) for value in (workspace.get("open_files") or [])[-4:]),
        ))
    )
    return " ".join(parts)


def geometric(raw: np.ndarray, gru: np.ndarray, source: np.ndarray) -> np.ndarray:
    alpha = np.where(source == "au", 0.85, 0.20).astype(np.float64)[:, None]
    value = alpha * np.log(np.maximum(raw, EPS)) + (1.0 - alpha) * np.log(np.maximum(gru, EPS))
    value -= value.max(axis=1, keepdims=True)
    value = np.exp(value)
    return (value / value.sum(axis=1, keepdims=True)).astype(np.float32)


def macro(y: np.ndarray, prediction: np.ndarray, mask: np.ndarray | None = None) -> float:
    if mask is None:
        mask = np.ones(len(y), dtype=bool)
    return float(f1_score(y[mask], prediction[mask], labels=np.arange(N_CLASSES), average="macro", zero_division=0))


def metric_bundle(y: np.ndarray, probability: np.ndarray, terminal: np.ndarray, char: np.ndarray) -> dict[str, float | int]:
    fallback = probability.argmax(axis=1).astype(np.int16)
    final = fallback.copy()
    use = terminal & (char >= 0)
    final[use] = char[use]
    hybrid = final.copy()
    hybrid[~terminal] = y[~terminal]
    return {
        "all_fallback_macro_f1": macro(y, fallback),
        "terminal_fallback_macro_f1": macro(y, fallback, terminal),
        "terminal_with_char20_macro_f1": macro(y, final, terminal),
        "hybrid_future_history_plus_char20_macro_f1": macro(y, hybrid),
        "char20_rows": int(use.sum()),
    }


def build_matrices(train_text: list[str], valid_text: list[str]):
    """Fit lexical vocabulary only on the fold-training sessions."""
    word = TfidfVectorizer(
        ngram_range=(1, 2), min_df=2, max_features=110_000,
        sublinear_tf=True, dtype=np.float32,
    )
    char = TfidfVectorizer(
        analyzer="char_wb", ngram_range=(3, 5), min_df=2, max_features=90_000,
        sublinear_tf=True, dtype=np.float32,
    )
    x_train = hstack((word.fit_transform(train_text), char.fit_transform(train_text)), format="csr")
    x_valid = hstack((word.transform(valid_text), char.transform(valid_text)), format="csr")
    details = {"word_features": len(word.vocabulary_), "char_features": len(char.vocabulary_), "total_features": int(x_train.shape[1])}
    return x_train, x_valid, details


def family_probabilities(x_train, train_y: np.ndarray, x_valid) -> dict[str, np.ndarray]:
    """Fit a compact classifier per family and return globally aligned probabilities."""
    output: dict[str, np.ndarray] = {}
    for name, labels in FAMILIES.items():
        take = np.isin(train_y, labels)
        model = SGDClassifier(
            loss="log_loss", alpha=2.0e-5, class_weight="balanced", average=True,
            max_iter=120, tol=1.0e-3, random_state=1357,
        )
        model.fit(x_train[take], train_y[take])
        local = model.predict_proba(x_valid).astype(np.float32)
        full = np.full((x_valid.shape[0], N_CLASSES), EPS, dtype=np.float32)
        full[:, model.classes_.astype(int)] = local
        # The fitted model is deliberately only a conditional family model;
        # normalize within the family so that geometric reranking is coherent.
        full[:, labels] /= full[:, labels].sum(axis=1, keepdims=True)
        output[name] = full
    return output


def rerank(base: np.ndarray, family_probability: dict[str, np.ndarray], terminal: np.ndarray,
           char: np.ndarray, policy: dict[str, float]) -> tuple[np.ndarray, dict[str, int]]:
    """Conservatively rerank only B43's top-two-in-family terminal rows."""
    result = base.copy()
    top_two = np.argpartition(base, -2, axis=1)[:, -2:]
    details: dict[str, int] = {}
    for name, weight in policy.items():
        labels = FAMILIES[name]
        # A char20 decision is already a vetted downstream terminal rule and
        # must never be displaced by the semantic model.
        gate = terminal & (char < 0) & np.all(np.isin(top_two, labels), axis=1)
        if np.any(gate):
            positions = np.flatnonzero(gate)
            # Advanced indexing writes the label block cleanly and leaves all
            # non-family probabilities untouched.
            result[np.ix_(positions, labels)] = np.exp(
                weight * np.log(np.maximum(result[np.ix_(positions, labels)], EPS))
                + (1.0 - weight) * np.log(np.maximum(family_probability[name][np.ix_(positions, labels)], EPS))
            )
            result[positions] /= result[positions].sum(axis=1, keepdims=True)
        details[name] = int(gate.sum())
    return result, details


def bootstrap_delta(y: np.ndarray, base: np.ndarray, candidate: np.ndarray, terminal: np.ndarray,
                    repeats: int = 3000, seed: int = 101) -> dict[str, float | int]:
    index = np.flatnonzero(terminal)
    rng = np.random.default_rng(seed)
    values = np.empty(repeats, dtype=np.float64)
    for number in range(repeats):
        sample = index[rng.integers(0, len(index), len(index))]
        values[number] = macro(y[sample], candidate[sample]) - macro(y[sample], base[sample])
    return {
        "repeats": repeats,
        "exact_delta": float(macro(y[index], candidate[index]) - macro(y[index], base[index])),
        "mean_delta": float(values.mean()),
        "p_delta_gt_zero": float(np.mean(values > 0.0)),
        "p05": float(np.quantile(values, 0.05)),
        "p95": float(np.quantile(values, 0.95)),
    }


def main() -> None:
    started = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(line) for line in (ROOT / "data" / "train.jsonl").open(encoding="utf-8") if line.strip()]
    label_map = {row["id"]: row["action"] for row in csv.DictReader((ROOT / "data" / "train_labels.csv").open(encoding="utf-8"))}
    split = {row["id"]: row["split"] for row in csv.DictReader((ROOT / "splits" / "common_split.csv").open(encoding="utf-8"))}
    ids = np.asarray([str(row["id"]) for row in rows])
    y = np.asarray([TO_ID[label_map[value]] for value in ids], dtype=np.int16)
    groups = np.asarray([session_of(value) for value in ids])
    terminal = terminal_mask(ids)
    source = source_of(ids)
    texts = [rich_text(row) for row in rows]

    raw = np.load(ROOT / "experiments" / "modeling_ak_live_xgb_proxy" / "oof_probabilities.npy").astype(np.float32)
    gru = np.load(ROOT / "experiments" / "modeling_ai_sequence_nested_distill_e16_i4" / "oof_probabilities.npy").astype(np.float32)
    if raw.shape != gru.shape or raw.shape != (len(rows), N_CLASSES):
        raise ValueError("B43 full OOF component shape mismatch")
    base_full = geometric(raw, gru, source)

    char_data = np.load(ROOT / "experiments" / "terminal_rule_combinations" / "oof_candidates.npz", allow_pickle=False)
    if not (np.array_equal(char_data["y"].astype(np.int16), y) and np.array_equal(char_data["terminal"].astype(bool), terminal)):
        raise ValueError("char20 full OOF artifact mismatch")
    char_full = char_data["char20"].astype(np.int16)

    common = np.flatnonzero(np.asarray([split[value] == "train" for value in ids]))
    fixed = np.flatnonzero(np.asarray([split[value] == "val" for value in ids]))
    fixed_terminal = terminal[fixed]
    if len(char_data["fixed_char20"]) != int(fixed_terminal.sum()):
        raise ValueError("char20 fixed terminal count mismatch")
    fixed_terminal_ids = ids[fixed][fixed_terminal]
    if not np.array_equal(char_data["valid_ids"].astype(str), fixed_terminal_ids):
        raise ValueError("char20 fixed IDs do not align with common validation terminals")
    char_fixed = np.full(len(fixed), -1, dtype=np.int16)
    char_fixed[fixed_terminal] = char_data["fixed_char20"].astype(np.int16)

    raw_fixed = np.load(ROOT / "experiments" / "au_source_strict_common" / "raw_global_valid_probability.npy").astype(np.float32)
    gru_fixed = np.load(ROOT / "experiments" / "modeling_ai_sequence_fusion_fixed_strictdistill_w35" / "valid_probabilities.npy").astype(np.float32)
    if raw_fixed.shape != gru_fixed.shape or raw_fixed.shape != (len(fixed), N_CLASSES):
        raise ValueError("B43 strict fixed component shape mismatch")
    base_fixed = geometric(raw_fixed, gru_fixed, source[fixed])

    folds = list(GroupKFold(n_splits=5).split(np.zeros(len(rows)), y, groups))
    fold_id = np.empty(len(rows), dtype=np.int8)
    full_family = {name: np.full((len(rows), N_CLASSES), EPS, dtype=np.float32) for name in FAMILIES}
    fold_records = []
    for number, (fit, held) in enumerate(folds, 1):
        fold_id[held] = number - 1
        begun = time.time()
        x_fit, x_held, details = build_matrices([texts[int(index)] for index in fit], [texts[int(index)] for index in held])
        family = family_probabilities(x_fit, y[fit], x_held)
        for name in FAMILIES:
            full_family[name][held] = family[name]
        fold_records.append({"fold": number, "fit_rows": int(len(fit)), "held_rows": int(len(held)), **details, "seconds": round(time.time() - begun, 2)})
        print(json.dumps(fold_records[-1]), flush=True)
        del x_fit, x_held, family
        gc.collect()

    # One independent strict fixed validation fit, using common-train sessions
    # only.  It is intentionally performed after the OOF policy family has
    # already been declared above.
    x_common, x_fixed, fixed_feature_details = build_matrices([texts[int(index)] for index in common], [texts[int(index)] for index in fixed])
    fixed_family = family_probabilities(x_common, y[common], x_fixed)
    del x_common, x_fixed
    gc.collect()

    full_options: dict[str, np.ndarray] = {}
    fixed_options: dict[str, np.ndarray] = {}
    policy_report: dict[str, dict] = {}
    baseline_full = metric_bundle(y, base_full, terminal, char_full)
    baseline_fixed = metric_bundle(y[fixed], base_fixed, fixed_terminal, char_fixed)
    for name, policy in POLICIES.items():
        full_probability, full_gate = rerank(base_full, full_family, terminal, char_full, policy)
        fixed_probability, fixed_gate = rerank(base_fixed, fixed_family, fixed_terminal, char_fixed, policy)
        full_options[name] = full_probability
        fixed_options[name] = fixed_probability
        full_metric = metric_bundle(y, full_probability, terminal, char_full)
        fixed_metric = metric_bundle(y[fixed], fixed_probability, fixed_terminal, char_fixed)
        policy_report[name] = {
            "policy": policy,
            "full": full_metric,
            "fixed": fixed_metric,
            "full_gate_rows": full_gate,
            "fixed_gate_rows": fixed_gate,
            "delta_vs_b43": {
                "full": {key: float(full_metric[key] - baseline_full[key]) for key in baseline_full if key != "char20_rows"},
                "fixed": {key: float(fixed_metric[key] - baseline_fixed[key]) for key in baseline_fixed if key != "char20_rows"},
            },
        }

    # Nested selection sees only the other four outer folds.  Candidate model
    # probabilities are already fold-cross-fitted, so no held session label
    # participates in its own model fit or policy selection.
    preference = {name: number for number, name in enumerate(POLICIES)}
    nested = np.zeros_like(base_full)
    nested_choices = []
    for held in range(5):
        select = terminal & (fold_id != held)
        scores = {
            name: macro(y, np.where((terminal & (char_full >= 0)), char_full, probability.argmax(axis=1)), select)
            for name, probability in full_options.items()
        }
        selected = max(POLICIES, key=lambda name: (scores[name], -preference[name]))
        nested[fold_id == held] = full_options[selected][fold_id == held]
        nested_choices.append({"held_fold": held + 1, "selected": selected, "scores": scores})
    selections = [record["selected"] for record in nested_choices]
    consensus = min(POLICIES, key=lambda name: (-selections.count(name), preference[name]))
    nested_full = metric_bundle(y, nested, terminal, char_full)
    consensus_fixed = metric_bundle(y[fixed], fixed_options[consensus], fixed_terminal, char_fixed)
    required = ("all_fallback_macro_f1", "terminal_with_char20_macro_f1", "hybrid_future_history_plus_char20_macro_f1")
    static_safe = [
        name for name, record in policy_report.items() if name != "base" and all(
            record["delta_vs_b43"][side][key] > 0.0 for side in ("full", "fixed") for key in required
        )
    ]
    nested_delta = {
        "full": {key: float(nested_full[key] - baseline_full[key]) for key in required},
        "fixed": {key: float(consensus_fixed[key] - baseline_fixed[key]) for key in required},
    }
    nested_safe = all(nested_delta[side][key] > 0.0 for side in ("full", "fixed") for key in required)

    base_terminal_prediction = base_full.argmax(axis=1).astype(np.int16)
    base_terminal_prediction[terminal & (char_full >= 0)] = char_full[terminal & (char_full >= 0)]
    candidate_terminal_prediction = nested.argmax(axis=1).astype(np.int16)
    candidate_terminal_prediction[terminal & (char_full >= 0)] = char_full[terminal & (char_full >= 0)]
    report = {
        "scope": "Experiment only: model/, script.py, B42.zip, and baseline_submit.zip were not modified.",
        "direction": "Conditional semantic lexical reranking inside only B43's quality and coordinate top-two action families.",
        "safety_contract": {
            "features": "current_prompt, prior user prompts, prior assistant action names/results/arguments, and metadata already supplied at inference",
            "forbidden": "No ID features, no target label lookup, no future-row label, no later-history recovery feature, and no fixed labels in policy selection",
            "immutable_downstream": "char20 applies after every semantic rerank; future-history recovery is used only in hybrid evaluation",
        },
        "counts": {"rows": int(len(rows)), "terminal": int(terminal.sum()), "common_rows": int(len(common)), "fixed_rows": int(len(fixed)), "fixed_terminal": int(fixed_terminal.sum())},
        "base": {"formula": "B43 strict-distilled GRU and raw selector geometric p; sim=.20, au=.85", "full": baseline_full, "fixed": baseline_fixed},
        "lexical_model": {"word": "TF-IDF 1-2 gram, max 110000", "char": "char_wb TF-IDF 3-5 gram, max 90000", "classifier": "balanced averaged SGD logistic", "outer_folds": fold_records, "fixed_feature_fit": fixed_feature_details},
        "policies": policy_report,
        "nested_outer_selection": {"choices": nested_choices, "consensus_policy": consensus, "full": nested_full, "fixed_consensus": consensus_fixed, "delta_vs_b43": nested_delta},
        "bootstrap_terminal_char20": bootstrap_delta(y, base_terminal_prediction, candidate_terminal_prediction, terminal),
        "promotion_gate": {"required_positive_metrics": list(required), "static_two_split_candidates": static_safe, "nested_two_split_pass": bool(nested_safe), "pass": bool(static_safe) and bool(nested_safe), "decision": "Promote only if both static and nested gates pass; otherwise reject and retain B43."},
        "seconds": round(time.time() - started, 2),
    }
    np.savez_compressed(
        OUT / "predictions.npz", y=y, terminal=terminal, fold=fold_id, base_full=base_full,
        nested=nested, fixed_index=fixed, base_fixed=base_fixed, fixed_terminal=fixed_terminal,
        consensus_fixed=fixed_options[consensus], char_full=char_full, char_fixed=char_fixed,
    )
    (OUT / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"promotion_gate": report["promotion_gate"], "nested_delta": nested_delta, "consensus": consensus, "seconds": report["seconds"]}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
