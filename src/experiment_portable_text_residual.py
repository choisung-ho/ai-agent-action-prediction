#!/usr/bin/env python3
"""Strict audit of small, deployable sparse text residuals.

This is deliberately an experiment-only program.  It does not open or write
``model/``, ``script.py``, or either submission archive.  Every text
vectorizer and classifier is fitted inside the training side of a whole
session GroupKFold fold.  The unrelated common fixed split is then evaluated
with a separately fitted text model.

The baseline is the current portable raw-XGB + strict-distilled no-E5 GRU
geometric fallback.  The current char20 terminal rule is rebuilt in each fold
and applied *after* any text blend, which mirrors the proposed deployment
order.  No IDs, train-label lookup, or future target action is used as a
feature.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import re
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold


CLASSES = (
    "read_file", "grep_search", "list_directory", "glob_pattern",
    "edit_file", "write_file", "apply_patch", "run_bash", "run_tests",
    "lint_or_typecheck", "ask_user", "plan_task", "web_search", "respond_only",
)
LABEL_TO_ID = {label: number for number, label in enumerate(CLASSES)}
WORD = re.compile(r"(?u)\b\w+\b")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--split-path", default="splits/common_split.csv")
    parser.add_argument("--out-dir", default="experiments/portable_text_residual")
    parser.add_argument("--variants", default="prompt,rich,terminal_prompt",
                        help="comma-separated: prompt,rich,terminal_prompt")
    parser.add_argument("--alphas", default="1e-5,3e-5",
                        help="SGD L2 strengths to screen")
    parser.add_argument("--word-features", type=int, default=70000)
    parser.add_argument("--char-features", type=int, default=70000)
    return parser.parse_args()


def compact(value: object, limit: int) -> str:
    return " ".join(str(value or "").replace("\x00", " ").split())[:limit]


def rich_text(sample: dict) -> str:
    """Return only inference-safe prompt/history/session metadata text.

    Names and results belong to *preceding* assistant actions in the supplied
    history.  They are observable at inference and are not current labels.
    The fixed field prefixes stop the classifier from conflating a word in a
    current request with the same word in a previous result.
    """
    pieces = ["CUR " + compact(sample.get("current_prompt"), 1100)]
    history = sample.get("history") if isinstance(sample.get("history"), list) else []
    users = [entry for entry in history if isinstance(entry, dict) and entry.get("role") == "user"]
    actions = [entry for entry in history if isinstance(entry, dict) and entry.get("role") == "assistant_action"]
    for offset, item in enumerate(reversed(users[-3:]), 1):
        pieces.append(f"USR{offset} " + compact(item.get("content"), 320))
    for offset, item in enumerate(reversed(actions[-4:]), 1):
        action = compact(item.get("name") or item.get("action"), 80)
        result = compact(item.get("result_summary"), 220)
        args = item.get("args") if isinstance(item.get("args"), dict) else {}
        args_text = " ".join(f"{key} {compact(value, 100)}" for key, value in sorted(args.items()))
        pieces.extend((f"ACT{offset} {action}", f"ARG{offset} {args_text}", f"RES{offset} {result}"))
    meta = sample.get("session_meta") if isinstance(sample.get("session_meta"), dict) else {}
    workspace = meta.get("workspace") if isinstance(meta.get("workspace"), dict) else {}
    pieces.append("META " + " ".join((
        compact(meta.get("user_tier"), 30), compact(meta.get("language_pref"), 30),
        compact(workspace.get("last_ci_status"), 30), str(workspace.get("git_dirty", "")),
        " ".join(compact(value, 120) for value in (workspace.get("open_files") or [])[-4:]),
    )))
    return " ".join(pieces)


def prompt_text(sample: dict) -> str:
    return compact(sample.get("current_prompt"), 1200)


def session_of(sample: dict) -> str:
    return str(sample["id"]).split("-step_", 1)[0]


def step_of(sample: dict) -> int:
    return int(str(sample["id"]).rsplit("-step_", 1)[1])


def source_of(sample: dict) -> str:
    return "au" if session_of(sample).startswith("sess_au_") else "sim"


def terminal_mask(rows: list[dict]) -> np.ndarray:
    maximum: dict[str, int] = {}
    for row in rows:
        group = session_of(row)
        maximum[group] = max(maximum.get(group, -1), step_of(row))
    return np.asarray([step_of(row) == maximum[session_of(row)] for row in rows], dtype=bool)


def geometric(base: np.ndarray, text: np.ndarray, base_weight: float) -> np.ndarray:
    log_probability = (
        base_weight * np.log(np.maximum(base, 1.0e-12))
        + (1.0 - base_weight) * np.log(np.maximum(text, 1.0e-12))
    )
    log_probability -= log_probability.max(axis=1, keepdims=True)
    probability = np.exp(log_probability)
    return probability / probability.sum(axis=1, keepdims=True)


def current_geometric(xgb: np.ndarray, gru: np.ndarray, sources: np.ndarray) -> np.ndarray:
    weights = np.where(sources == "au", 0.85, 0.20).astype(np.float32)
    log_probability = (
        weights[:, None] * np.log(np.maximum(xgb, 1.0e-12))
        + (1.0 - weights[:, None]) * np.log(np.maximum(gru, 1.0e-12))
    )
    log_probability -= log_probability.max(axis=1, keepdims=True)
    probability = np.exp(log_probability)
    return (probability / probability.sum(axis=1, keepdims=True)).astype(np.float32)


def text_matrices(train_text: list[str], valid_text: list[str], word_features: int, char_features: int):
    word = TfidfVectorizer(
        token_pattern=r"(?u)\b\w+\b", ngram_range=(1, 2), min_df=2,
        max_features=word_features, sublinear_tf=True, strip_accents=None,
        dtype=np.float32,
    )
    char = TfidfVectorizer(
        analyzer="char_wb", ngram_range=(3, 5), min_df=2, max_features=char_features,
        sublinear_tf=True, dtype=np.float32,
    )
    x_train = hstack((word.fit_transform(train_text), char.fit_transform(train_text)), format="csr")
    x_valid = hstack((word.transform(valid_text), char.transform(valid_text)), format="csr")
    return x_train, x_valid, word, char


def fit_predict(train_text: list[str], train_y: np.ndarray, valid_text: list[str], alpha: float,
                word_features: int, char_features: int) -> tuple[np.ndarray, dict]:
    x_train, x_valid, word, char = text_matrices(train_text, valid_text, word_features, char_features)
    classifier = SGDClassifier(
        loss="log_loss", alpha=alpha, penalty="l2", class_weight="balanced",
        max_iter=80, tol=1.0e-4, average=True, random_state=42, n_jobs=-1,
    )
    classifier.fit(x_train, train_y)
    # Every training side contains all fourteen classes.  Still, map by class
    # ID rather than assuming the sparse model happens to preserve order.
    result = np.full((len(valid_text), len(CLASSES)), 1.0e-12, dtype=np.float32)
    result[:, classifier.classes_] = classifier.predict_proba(x_valid).astype(np.float32)
    result /= result.sum(axis=1, keepdims=True)
    details = {
        "n_features": int(x_train.shape[1]),
        "word_vocabulary": int(len(word.vocabulary_)),
        "char_vocabulary": int(len(char.vocabulary_)),
    }
    del x_train, x_valid, word, char, classifier
    gc.collect()
    return result, details


def char_features(sample: dict) -> tuple[str, ...]:
    values: set[str] = set()
    for raw in WORD.findall(str(sample.get("current_prompt", "")).casefold())[:70]:
        token = "^" + re.sub(r"\d+", "<n>", raw) + "$"
        for width in (3, 4, 5):
            values.update(
                "c" + str(width) + "|" + token[offset:offset + width]
                for offset in range(max(0, len(token) - width + 1))
            )
    return tuple(values)


def build_char_rule(features: list[tuple[str, ...]], y: np.ndarray, train_index: np.ndarray) -> dict[str, int]:
    counts: dict[str, Counter[int]] = defaultdict(Counter)
    for index in train_index:
        for key in features[int(index)]:
            counts[key][int(y[index])] += 1
    return {
        key: next(iter(value)) for key, value in counts.items()
        if sum(value.values()) >= 20 and len(value) == 1
    }


def char_decide(feature_values: tuple[str, ...], rule: dict[str, int]) -> int:
    values = {rule[key] for key in feature_values if key in rule}
    return next(iter(values)) if len(values) == 1 else -1


def char_oof(features: list[tuple[str, ...]], y: np.ndarray, terminal: np.ndarray,
             folds: list[tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
    output = np.full(len(y), -1, dtype=np.int16)
    for train_index, valid_index in folds:
        rule = build_char_rule(features, y, train_index)
        valid_terminal = valid_index[terminal[valid_index]]
        output[valid_terminal] = np.asarray(
            [char_decide(features[int(index)], rule) for index in valid_terminal], dtype=np.int16
        )
    return output


def macro(y: np.ndarray, prediction: np.ndarray) -> float:
    return float(f1_score(y, prediction, labels=np.arange(len(CLASSES)), average="macro", zero_division=0))


def apply_char(probability: np.ndarray, terminal: np.ndarray, char_prediction: np.ndarray) -> np.ndarray:
    prediction = probability.argmax(axis=1).astype(np.int16)
    mask = terminal & (char_prediction >= 0)
    prediction[mask] = char_prediction[mask]
    return prediction


def config_predictions(base: np.ndarray, text: np.ndarray, terminal: np.ndarray, char_prediction: np.ndarray,
                       base_weight: float, text_confidence: float, base_confidence: float) -> np.ndarray:
    candidate = geometric(base, text, base_weight)
    use_text = terminal & (text.max(axis=1) >= text_confidence) & (base.max(axis=1) <= base_confidence)
    probability = base.copy()
    probability[use_text] = candidate[use_text]
    return apply_char(probability, terminal, char_prediction)


def metric_record(y: np.ndarray, terminal: np.ndarray, prediction: np.ndarray) -> dict:
    # Non-terminal rows are exactly future-history recoverable for the offline
    # train construction, so the hybrid score is informative but terminal
    # macro-F1 remains the genuine fallback comparison.
    hybrid = np.where(terminal, prediction, y)
    return {
        "terminal_macro_f1": macro(y[terminal], prediction[terminal]),
        "hybrid_macro_f1": macro(y, hybrid),
        "terminal_changed": int(np.sum(prediction[terminal] != -1)),
    }


def assess_grid(y: np.ndarray, terminal: np.ndarray, base: np.ndarray, text: np.ndarray,
                char_prediction: np.ndarray, folds: list[tuple[np.ndarray, np.ndarray]]) -> tuple[dict, dict]:
    """Select configuration in the other four session folds, then score held fold."""
    base_prediction = apply_char(base, terminal, char_prediction)
    baseline = metric_record(y, terminal, base_prediction)
    configs = [
        (weight, text_conf, base_conf)
        for weight in (0.65, 0.75, 0.82, 0.88, 0.92, 0.96)
        for text_conf in (0.0, 0.35, 0.45, 0.55, 0.65, 0.75)
        for base_conf in (0.55, 0.65, 0.75, 0.85, 1.0)
    ]
    static: dict[str, dict] = {}
    for config in configs:
        pred = config_predictions(base, text, terminal, char_prediction, *config)
        static["/".join(map(str, config))] = metric_record(y, terminal, pred)
    output = base_prediction.copy()
    selected: list[dict] = []
    for fold_number, (_, valid_index) in enumerate(folds, 1):
        train_mask = np.ones(len(y), dtype=bool); train_mask[valid_index] = False
        train_terminal = train_mask & terminal
        best = None
        for config in configs:
            pred = config_predictions(base, text, terminal, char_prediction, *config)
            value = macro(y[train_terminal], pred[train_terminal])
            key = (value, config[0], config[1], -config[2])
            if best is None or key > best[0]:
                best = (key, config, pred)
        assert best is not None
        output[valid_index] = best[2][valid_index]
        selected.append({"fold": fold_number, "config": {
            "base_weight": best[1][0], "text_confidence": best[1][1], "base_confidence": best[1][2],
        }})
    nested = metric_record(y, terminal, output)
    nested["selected_configurations"] = selected
    best_static_key, best_static = max(
        static.items(), key=lambda item: (item[1]["terminal_macro_f1"], item[0])
    )
    return {
        "baseline_with_char20": baseline,
        "best_static_for_diagnostic_only": {"config": best_static_key, **best_static},
        "nested_config_selection": nested,
    }, {"nested_prediction": output, "base_prediction": base_prediction}


def main() -> None:
    args = parse_args()
    started = time.time()
    root = Path(__file__).resolve().parents[1]
    out = root / args.out_dir; out.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(line) for line in (root / args.data_dir / "train.jsonl").open(encoding="utf-8") if line.strip()]
    labels = {row["id"]: row["action"] for row in csv.DictReader((root / args.data_dir / "train_labels.csv").open(encoding="utf-8"))}
    split = {row["id"]: row["split"] for row in csv.DictReader((root / args.split_path).open(encoding="utf-8"))}
    y = np.asarray([LABEL_TO_ID[labels[str(row["id"])]] for row in rows], dtype=np.int64)
    groups = np.asarray([session_of(row) for row in rows])
    sources = np.asarray([source_of(row) for row in rows])
    terminal = terminal_mask(rows)
    folds = list(GroupKFold(n_splits=5).split(np.zeros(len(rows)), y, groups))
    fixed_train = np.asarray([number for number, row in enumerate(rows) if split[str(row["id"])] == "train"], dtype=int)
    fixed_valid = np.asarray([number for number, row in enumerate(rows) if split[str(row["id"])] == "val"], dtype=int)

    xgb = np.load(root / "experiments/modeling_ak_live_xgb_proxy/oof_probabilities.npy").astype(np.float32)
    gru = np.load(root / "experiments/modeling_ai_sequence_nested_distill_e16_i4/oof_probabilities.npy").astype(np.float32)
    base = current_geometric(xgb, gru, sources)
    # ``modeling_ac_tune2`` is an earlier 2025-column selector and must not
    # be used to judge an addition to the present 2032-column deployment.
    # This strict common-split artifact has inner-OOF selector training
    # blocks and full-common-train validation blocks under the 2032-feature
    # contract, with validation labels excluded throughout feature fitting.
    fixed_xgb = np.load(
        root / "experiments/au_source_strict_common/raw_global_valid_probability.npy"
    ).astype(np.float32)
    fixed_gru = np.load(root / "experiments/modeling_ai_sequence_fusion_fixed_strictdistill_w35/valid_probabilities.npy").astype(np.float32)
    if len(fixed_xgb) != len(fixed_valid) or len(fixed_gru) != len(fixed_valid):
        raise ValueError("Fixed fallback probabilities do not align with common validation IDs")
    fixed_base = current_geometric(fixed_xgb, fixed_gru, sources[fixed_valid])

    print("building_char20_features", flush=True)
    chars = [char_features(row) for row in rows]
    char_outer = char_oof(chars, y, terminal, folds)
    fixed_char_rule = build_char_rule(chars, y, fixed_train)
    char_fixed = np.asarray(
        [char_decide(chars[int(index)], fixed_char_rule) for index in fixed_valid], dtype=np.int16
    )
    # char20 is terminal-only; make the fixed vector use the validation-local
    # rows while leaving all nonterminals unchanged in scoring helpers.
    char_fixed[~terminal[fixed_valid]] = -1

    baseline_outer = apply_char(base, terminal, char_outer)
    baseline_fixed = apply_char(fixed_base, terminal[fixed_valid], char_fixed)
    report: dict[str, object] = {
        "scope": "Experiment only: no live model/script/submission archive was changed.",
        "protocol": {
            "text": "Each TF-IDF vocabulary and SGD logistic classifier is fit only on a fold's training sessions.",
            "base": "strict-distilled no-E5 GRU + raw XGB current geometric sim=0.20 / au=0.85.",
            "char20": "rebuild char20 rule from every outer train partition; apply it after text blend only on terminal rows.",
            "selection": "Text blend/gate grid is chosen using the other four full GroupKFold folds before each held session fold; common fixed validation is untouched.",
        },
        "counts": {"rows": int(len(rows)), "terminal": int(terminal.sum()), "fixed_rows": int(len(fixed_valid)), "fixed_terminal": int(terminal[fixed_valid].sum())},
        "baseline": {
            "full_char20_terminal_macro_f1": macro(y[terminal], baseline_outer[terminal]),
            "full_char20_hybrid_macro_f1": macro(y, np.where(terminal, baseline_outer, y)),
            "fixed_char20_terminal_macro_f1": macro(y[fixed_valid][terminal[fixed_valid]], baseline_fixed[terminal[fixed_valid]]),
        },
        "variants": {},
    }
    alpha_values = [float(value) for value in args.alphas.split(",") if value.strip()]
    variants = [value.strip() for value in args.variants.split(",") if value.strip()]
    text_getters = {"prompt": prompt_text, "rich": rich_text, "terminal_prompt": prompt_text}
    for variant in variants:
        if variant not in text_getters:
            raise ValueError(f"Unknown variant: {variant}")
        print(f"variant={variant} text_prepare", flush=True)
        texts = [text_getters[variant](row) for row in rows]
        variant_report: dict[str, object] = {"alphas": {}}
        for alpha in alpha_values:
            print(f"variant={variant} alpha={alpha} outer_oof", flush=True)
            text_outer = np.zeros((len(rows), len(CLASSES)), dtype=np.float32)
            fold_details = []
            for fold_number, (train_index, valid_index) in enumerate(folds, 1):
                fit_index = train_index[terminal[train_index]] if variant == "terminal_prompt" else train_index
                probability, details = fit_predict(
                    [texts[int(index)] for index in fit_index], y[fit_index],
                    [texts[int(index)] for index in valid_index], alpha,
                    args.word_features, args.char_features,
                )
                text_outer[valid_index] = probability
                fold_details.append({"fold": fold_number, "train_rows": int(len(fit_index)), **details})
                print(f"variant={variant} alpha={alpha} fold={fold_number}/5 features={details['n_features']}", flush=True)
            assessment, saved = assess_grid(y, terminal, base, text_outer, char_outer, folds)
            print(f"variant={variant} alpha={alpha} fixed", flush=True)
            fixed_fit = fixed_train[terminal[fixed_train]] if variant == "terminal_prompt" else fixed_train
            fixed_text, fixed_details = fit_predict(
                [texts[int(index)] for index in fixed_fit], y[fixed_fit],
                [texts[int(index)] for index in fixed_valid], alpha,
                args.word_features, args.char_features,
            )
            # Evaluate each nested-selected fold config directly on the fixed
            # set only as a diagnostic.  A single conservative static config
            # is also reported, but no config is promoted unless both views
            # improve over the char20 baseline.
            fixed_grid: dict[str, dict] = {}
            for base_weight in (0.65, 0.75, 0.82, 0.88, 0.92, 0.96):
                for text_conf in (0.0, 0.35, 0.45, 0.55, 0.65, 0.75):
                    for base_conf in (0.55, 0.65, 0.75, 0.85, 1.0):
                        prediction = config_predictions(
                            fixed_base, fixed_text, terminal[fixed_valid], char_fixed,
                            base_weight, text_conf, base_conf,
                        )
                        fixed_grid[f"{base_weight}/{text_conf}/{base_conf}"] = {
                            "terminal_macro_f1": macro(
                                y[fixed_valid][terminal[fixed_valid]], prediction[terminal[fixed_valid]]
                            ),
                            "n_text_changed_before_char": int(np.sum(
                                (terminal[fixed_valid]) & (prediction != baseline_fixed)
                            )),
                        }
            best_fixed_key, best_fixed_value = max(
                fixed_grid.items(), key=lambda item: (item[1]["terminal_macro_f1"], item[0])
            )
            # Apply the per-fold selected configurations cyclically to the
            # unrelated fixed partition.  This is not a deployment policy;
            # it shows whether all independently selected configurations are
            # directionally safe rather than cherry-picking one fixed score.
            selected_values = [item["config"] for item in assessment["nested_config_selection"]["selected_configurations"]]
            nested_fixed_scores = []
            for config in selected_values:
                prediction = config_predictions(
                    fixed_base, fixed_text, terminal[fixed_valid], char_fixed,
                    config["base_weight"], config["text_confidence"], config["base_confidence"],
                )
                nested_fixed_scores.append({
                    "config": config,
                    "terminal_macro_f1": macro(y[fixed_valid][terminal[fixed_valid]], prediction[terminal[fixed_valid]]),
                })
            assessment["fixed_best_static_diagnostic_only"] = {"config": best_fixed_key, **best_fixed_value}
            assessment["fixed_scores_for_nested_selected_configs"] = nested_fixed_scores
            variant_report["alphas"][str(alpha)] = {
                "outer": assessment,
                "fixed_vectorizer": fixed_details,
                "outer_folds": fold_details,
                "text_standalone_outer_terminal_macro_f1": macro(y[terminal], text_outer[terminal].argmax(axis=1)),
                "text_standalone_fixed_terminal_macro_f1": macro(y[fixed_valid][terminal[fixed_valid]], fixed_text[terminal[fixed_valid]].argmax(axis=1)),
            }
            np.savez_compressed(
                out / f"{variant}_a{str(alpha).replace('-', 'm').replace('.', 'p')}_probabilities.npz",
                text_outer=text_outer, text_fixed=fixed_text, nested_prediction=saved["nested_prediction"],
                base_prediction=saved["base_prediction"], fixed_ids=np.asarray([rows[int(index)]["id"] for index in fixed_valid]),
            )
            del text_outer, fixed_text
            gc.collect()
        report["variants"][variant] = variant_report
        del texts
        gc.collect()
    report["seconds"] = round(time.time() - started, 2)
    (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
