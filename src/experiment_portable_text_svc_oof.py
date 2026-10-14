#!/usr/bin/env python3
"""Whole-session OOF audit for the only promising portable sparse-text gate.

This program intentionally writes only under ``experiments/``.  It recreates
the pre-existing R_text field-aware word/character LinearSVC using vectorizers
fit within each GroupKFold training partition.  A text prediction can replace
the current geometric fallback only for a terminal row with a high SVC margin
and a low fallback confidence; the existing char20 rule is always applied
afterward.  There is no ID feature, label lookup, or target-history feature.

The fixed test uses the existing fixed-train R_text artifact, which was fitted
before this audit and excludes the common-validation sessions.  Its raw XGB
reference is the strict 2032-column common-split selector (same deployed
hyperparameters, seed 808 rather than live seed 42).
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold
from sklearn.svm import LinearSVC

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from experiment_portable_text_residual import (  # noqa: E402
    CLASSES, LABEL_TO_ID, apply_char, current_geometric, source_of, terminal_mask,
)
from train_modeling_r_text import feature_fields  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="experiments/portable_text_svc_oof")
    parser.add_argument("--c", type=float, default=0.7)
    parser.add_argument("--word-features", type=int, default=160_000)
    parser.add_argument("--char-features", type=int, default=120_000)
    parser.add_argument("--history-features", type=int, default=80_000)
    return parser.parse_args()


def score(y: np.ndarray, prediction: np.ndarray) -> float:
    return float(f1_score(y, prediction, labels=np.arange(len(CLASSES)), average="macro", zero_division=0))


def make_matrices(train_fields: list[tuple[str, str, str]], valid_fields: list[tuple[str, str, str]],
                  word_features: int, char_features: int, history_features: int):
    word = TfidfVectorizer(
        ngram_range=(1, 3), min_df=2, max_features=word_features, sublinear_tf=True,
        lowercase=False, dtype=np.float32,
    )
    current_char = TfidfVectorizer(
        analyzer="char_wb", ngram_range=(3, 5), min_df=2, max_features=char_features,
        sublinear_tf=True, lowercase=False, dtype=np.float32,
    )
    history_word = TfidfVectorizer(
        ngram_range=(1, 2), min_df=2, max_features=history_features, sublinear_tf=True,
        lowercase=False, dtype=np.float32,
    )
    x_train = hstack((
        word.fit_transform([value[0] for value in train_fields]),
        current_char.fit_transform([value[1] for value in train_fields]),
        history_word.fit_transform([value[2] for value in train_fields]),
    ), format="csr")
    x_valid = hstack((
        word.transform([value[0] for value in valid_fields]),
        current_char.transform([value[1] for value in valid_fields]),
        history_word.transform([value[2] for value in valid_fields]),
    ), format="csr")
    return x_train, x_valid, (word, current_char, history_word)


def fit_scores(train_fields: list[tuple[str, str, str]], train_y: np.ndarray,
               valid_fields: list[tuple[str, str, str]], args: argparse.Namespace) -> tuple[np.ndarray, dict]:
    x_train, x_valid, vectorizers = make_matrices(
        train_fields, valid_fields, args.word_features, args.char_features, args.history_features
    )
    classifier = LinearSVC(C=args.c, class_weight="balanced", dual=True, max_iter=10_000)
    classifier.fit(x_train, train_y)
    raw_score = classifier.decision_function(x_valid).astype(np.float32)
    # Existing labels are strings in the historical R artifact; freshly fit
    # models use numeric labels.  Be robust to either representation.
    classes = np.asarray(classifier.classes_)
    if classes.dtype.kind in ("U", "S", "O"):
        ids = np.asarray([LABEL_TO_ID[str(value)] for value in classes], dtype=int)
    else:
        ids = classes.astype(int)
    aligned = np.full((len(valid_fields), len(CLASSES)), -1.0e9, dtype=np.float32)
    aligned[:, ids] = raw_score
    details = {
        "n_features": int(x_train.shape[1]),
        "word_vocabulary": int(len(vectorizers[0].vocabulary_)),
        "char_vocabulary": int(len(vectorizers[1].vocabulary_)),
        "history_vocabulary": int(len(vectorizers[2].vocabulary_)),
    }
    del x_train, x_valid, vectorizers, classifier
    gc.collect()
    return aligned, details


def override_prediction(base: np.ndarray, text_score: np.ndarray, terminal: np.ndarray, char: np.ndarray,
                        margin_threshold: float, base_confidence: float) -> np.ndarray:
    prediction = base.argmax(axis=1).astype(np.int16)
    ordering = np.sort(text_score, axis=1)
    margin = ordering[:, -1] - ordering[:, -2]
    text_prediction = text_score.argmax(axis=1).astype(np.int16)
    gate = terminal & (margin >= margin_threshold) & (base.max(axis=1) <= base_confidence)
    prediction[gate] = text_prediction[gate]
    # char20 is the last learned fallback rule, exactly as in the live path.
    prediction[terminal & (char >= 0)] = char[terminal & (char >= 0)]
    return prediction


def candidates() -> list[tuple[float, float]]:
    return [
        (margin, base_conf)
        for margin in (0.0, 0.25, 0.50, 0.75, 1.00, 1.25, 1.50, 2.00, 2.50, 3.00, 4.00)
        for base_conf in (0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00)
    ]


def terminal_metrics(y: np.ndarray, terminal: np.ndarray, prediction: np.ndarray) -> dict:
    return {
        "terminal_macro_f1": score(y[terminal], prediction[terminal]),
        "hybrid_macro_f1": score(y, np.where(terminal, prediction, y)),
    }


def nested_select(y: np.ndarray, terminal: np.ndarray, base: np.ndarray, text_score: np.ndarray,
                  char: np.ndarray, folds: list[tuple[np.ndarray, np.ndarray]]) -> tuple[dict, np.ndarray]:
    base_prediction = apply_char(base, terminal, char)
    output = base_prediction.copy()
    selections: list[dict] = []
    grid = candidates()
    for fold_number, (_, valid) in enumerate(folds, 1):
        train_mask = np.ones(len(y), dtype=bool); train_mask[valid] = False
        train_terminal = train_mask & terminal
        best: tuple[float, float, float, np.ndarray] | None = None
        best_key: tuple[float, float, float] | None = None
        for margin, base_conf in grid:
            prediction = override_prediction(base, text_score, terminal, char, margin, base_conf)
            value = score(y[train_terminal], prediction[train_terminal])
            # Prefer the stricter gate when scores tie, avoiding artificial
            # extra modifications for numerically equal score.
            key = (value, margin, -base_conf)
            if best_key is None or key > best_key:
                best = (value, margin, base_conf, prediction)
                best_key = key
        assert best is not None
        output[valid] = best[3][valid]
        selections.append({"fold": fold_number, "margin_threshold": best[1], "base_confidence": best[2], "train_terminal_macro_f1": best[0]})
    static = []
    for margin, base_conf in grid:
        prediction = override_prediction(base, text_score, terminal, char, margin, base_conf)
        static.append({"margin_threshold": margin, "base_confidence": base_conf, **terminal_metrics(y, terminal, prediction)})
    best_static = max(static, key=lambda value: (value["terminal_macro_f1"], value["margin_threshold"], -value["base_confidence"]))
    return {
        "baseline_with_char20": terminal_metrics(y, terminal, base_prediction),
        "nested_selection": {**terminal_metrics(y, terminal, output), "folds": selections},
        "best_static_diagnostic_only": best_static,
    }, output


def fixed_artifact_scores(fields: list[tuple[str, str, str]], artifact: Path) -> np.ndarray:
    bundle = joblib.load(artifact)
    matrix = hstack((
        bundle["word"].transform([value[0] for value in fields]),
        bundle["current_char"].transform([value[1] for value in fields]),
        bundle["history_word"].transform([value[2] for value in fields]),
    ), format="csr")
    raw = bundle["classifier"].decision_function(matrix).astype(np.float32)
    classes = np.asarray(bundle["classifier"].classes_)
    ids = np.asarray([LABEL_TO_ID[str(value)] for value in classes], dtype=int)
    output = np.full((len(fields), len(CLASSES)), -1.0e9, dtype=np.float32)
    output[:, ids] = raw
    return output


def main() -> None:
    args = parse_args()
    out = ROOT / args.out_dir; out.mkdir(parents=True, exist_ok=True)
    started = time.time()
    rows = [json.loads(line) for line in (ROOT / "data/train.jsonl").open(encoding="utf-8") if line.strip()]
    labels = {row["id"]: row["action"] for row in csv.DictReader((ROOT / "data/train_labels.csv").open(encoding="utf-8"))}
    split = {row["id"]: row["split"] for row in csv.DictReader((ROOT / "splits/common_split.csv").open(encoding="utf-8"))}
    y = np.asarray([LABEL_TO_ID[labels[str(row["id"])]] for row in rows], dtype=np.int64)
    terminal = terminal_mask(rows)
    sources = np.asarray([source_of(row) for row in rows])
    groups = np.asarray([str(row["id"]).split("-step_", 1)[0] for row in rows])
    folds = list(GroupKFold(n_splits=5).split(np.zeros(len(rows)), y, groups))
    valid = np.asarray([index for index, row in enumerate(rows) if split[str(row["id"])] == "val"], dtype=int)
    fields = [feature_fields(row) for row in rows]

    xgb = np.load(ROOT / "experiments/modeling_ak_live_xgb_proxy/oof_probabilities.npy").astype(np.float32)
    gru = np.load(ROOT / "experiments/modeling_ai_sequence_nested_distill_e16_i4/oof_probabilities.npy").astype(np.float32)
    base = current_geometric(xgb, gru, sources)
    # char20 is a label-pure table rebuilt in the same GroupKFold assignment;
    # it is independent of the fallback probability source, so reuse its
    # already serialized exact OOF decisions after verifying alignment.
    char_bundle = np.load(ROOT / "experiments/terminal_rule_combinations/oof_candidates.npz")
    if not np.array_equal(char_bundle["y"], y) or not np.array_equal(char_bundle["terminal"], terminal):
        raise ValueError("Stored char20 OOF rows do not align with current train order")
    char = char_bundle["char20"].astype(np.int16)

    score_outer = np.zeros((len(rows), len(CLASSES)), dtype=np.float32)
    fold_details: list[dict] = []
    for fold_number, (train, holdout) in enumerate(folds, 1):
        print(f"fold={fold_number}/5 vectorize_fit", flush=True)
        values, detail = fit_scores([fields[int(index)] for index in train], y[train], [fields[int(index)] for index in holdout], args)
        score_outer[holdout] = values
        fold_details.append({"fold": fold_number, "train_rows": int(len(train)), "valid_rows": int(len(holdout)), **detail})
        print(f"fold={fold_number}/5 n_features={detail['n_features']}", flush=True)
    outer_report, nested_prediction = nested_select(y, terminal, base, score_outer, char, folds)

    # The fixed model was fit in the historical R_text experiment exclusively
    # on common-train.  Reusing the saved model makes this verification fast
    # while preserving the strict split; its data/model contract is recorded
    # in the artifact's original script.
    artifact = ROOT / "experiments/modeling_r_text/R_text_wordchar_c07.pkl"
    fixed_score = fixed_artifact_scores([fields[int(index)] for index in valid], artifact)
    fixed_xgb = np.load(ROOT / "experiments/au_source_strict_common/raw_global_valid_probability.npy").astype(np.float32)
    fixed_gru = np.load(ROOT / "experiments/modeling_ai_sequence_fusion_fixed_strictdistill_w35/valid_probabilities.npy").astype(np.float32)
    fixed_base = current_geometric(fixed_xgb, fixed_gru, sources[valid])
    terminal_fixed = terminal[valid]
    valid_terminal_ids = np.asarray([str(rows[int(index)]["id"]) for index in valid[terminal_fixed]])
    if not np.array_equal(char_bundle["valid_ids"], valid_terminal_ids):
        raise ValueError("Stored fixed char20 IDs do not align with common validation")
    fixed_char = np.full(len(valid), -1, dtype=np.int16)
    fixed_char[terminal_fixed] = char_bundle["fixed_char20"].astype(np.int16)
    base_fixed_prediction = apply_char(fixed_base, terminal_fixed, fixed_char)

    fixed_grid: list[dict] = []
    for margin, base_conf in candidates():
        prediction = override_prediction(fixed_base, fixed_score, terminal_fixed, fixed_char, margin, base_conf)
        fixed_grid.append({
            "margin_threshold": margin, "base_confidence": base_conf,
            "terminal_macro_f1": score(y[valid][terminal_fixed], prediction[terminal_fixed]),
            "n_changed_vs_char20_base": int(np.sum(prediction[terminal_fixed] != base_fixed_prediction[terminal_fixed])),
        })
    best_fixed = max(fixed_grid, key=lambda value: (value["terminal_macro_f1"], value["margin_threshold"], -value["base_confidence"]))
    selected_configs = outer_report["nested_selection"]["folds"]
    fixed_nested_configs = []
    for value in selected_configs:
        prediction = override_prediction(
            fixed_base, fixed_score, terminal_fixed, fixed_char,
            value["margin_threshold"], value["base_confidence"],
        )
        fixed_nested_configs.append({
            "margin_threshold": value["margin_threshold"], "base_confidence": value["base_confidence"],
            "terminal_macro_f1": score(y[valid][terminal_fixed], prediction[terminal_fixed]),
            "n_changed_vs_char20_base": int(np.sum(prediction[terminal_fixed] != base_fixed_prediction[terminal_fixed])),
        })
    report = {
        "scope": "Experiment only: no live model/script/submission archive was changed.",
        "protocol": {
            "outer": "5-fold GroupKFold; TF-IDF vectorizers and LinearSVC fit only on each fold's train sessions.",
            "fixed": "R_text C=.7 artifact fit only on common-train; fixed base uses a 2032-feature strict common-split raw selector with seed 808 (live selector seed is 42) plus the strict distilled GRU.",
            "gate": "terminal only; high SVC top-two margin and low geometric-base confidence; char20 applied after the text decision.",
            "prohibited": "No ID, lookup table, target action, or future history field is a text feature.",
        },
        "counts": {"rows": int(len(rows)), "terminal": int(terminal.sum()), "fixed_rows": int(len(valid)), "fixed_terminal": int(terminal_fixed.sum())},
        "c": args.c,
        "outer": outer_report,
        "fixed": {
            "baseline_with_char20_terminal_macro_f1": score(y[valid][terminal_fixed], base_fixed_prediction[terminal_fixed]),
            "best_static_diagnostic_only": best_fixed,
            "scores_for_nested_selected_configs": fixed_nested_configs,
        },
        "folds": fold_details,
        "seconds": round(time.time() - started, 2),
    }
    np.savez_compressed(
        out / "oof_scores.npz", text_score=score_outer, nested_prediction=nested_prediction,
        y=y, terminal=terminal, base=base, char=char, fixed_score=fixed_score,
        fixed_ids=np.asarray([str(rows[int(index)]["id"]) for index in valid]),
    )
    (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
