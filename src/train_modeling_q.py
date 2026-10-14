#!/usr/bin/env python3
"""Modeling Q: add explicit session-state features to the I3 stack selector.

The original selector sees only the three base classifiers' margins and hard
decisions.  This experiment deliberately leaves those models untouched and
tests whether state that is already available at inference (recent actions,
result status, and session metadata) lets a selector resolve their conflicts.

All artifacts are written below ``experiments/modeling_q``.  This script never
writes ``model/`` and is therefore safe to run alongside a submission model.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import GroupKFold

from data import load_jsonl, load_labels
from train_modeling_b import ALL_CLASSES, build_modeling_b_features
from train_modeling_e import build_modeling_e_features
from train_modeling_i import stack_features


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--split-path", default="splits/common_split.csv")
    parser.add_argument("--e-model", default="experiments/modeling_e/E0.pkl")
    parser.add_argument("--d-model", default="experiments/modeling_d/D2.pkl")
    parser.add_argument("--b-model", default="experiments/modeling_b/B3.pkl")
    parser.add_argument("--out-dir", default="experiments/modeling_q")
    parser.add_argument("--trees", type=int, default=350)
    parser.add_argument("--experiments", default="", help="Comma-separated Q ids; empty runs every candidate")
    parser.add_argument("--prepare-only", action="store_true", help="Materialize reusable selector inputs and exit")
    return parser.parse_args()


def load_split(path: str) -> dict[str, str]:
    with open(path, newline="", encoding="utf-8") as handle:
        return {row["id"]: row["split"] for row in csv.DictReader(handle)}


def result_bucket(value: object) -> str:
    text = str(value or "").lower()
    if any(token in text for token in ("fail", "error", "exception", "traceback", "not found", "0 match")):
        return "fail"
    if any(token in text for token in ("pass", "ok;", "exit=0", "modified", "patched", "matched", "found", "listed")):
        return "ok"
    return "other"


def action_name(item: object) -> str:
    if not isinstance(item, dict):
        return "none"
    return str(item.get("name") or item.get("action") or "none").lower()


def bucket(value: object, cuts: tuple[float, ...]) -> str:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return "unknown"
    for index, upper in enumerate(cuts):
        if value <= upper:
            return str(index)
    return str(len(cuts))


def state_dict(sample: dict) -> dict[str, float]:
    """Low-cardinality features only; no session id or prompt text leakage."""
    feature: dict[str, float] = {}
    history = sample.get("history")
    history = history if isinstance(history, list) else []
    assistant_actions = [item for item in history if isinstance(item, dict) and item.get("role") == "assistant_action"]
    recent = list(reversed(assistant_actions[-5:]))
    feature[f"kind={str(sample.get('id', '')).split('_')[1] if '_' in str(sample.get('id', '')) else 'other'}"] = 1.0
    feature[f"history_len={bucket(len(history), (0, 2, 4, 6, 9, 13))}"] = 1.0
    for position in range(1, 6):
        item = recent[position - 1] if len(recent) >= position else {}
        name = action_name(item)
        feature[f"pa{position}={name}"] = 1.0
        feature[f"pr{position}={result_bucket(item.get('result_summary') if isinstance(item, dict) else '')}"] = 1.0
        if position == 1 and isinstance(item, dict):
            args = item.get("args")
            args = args if isinstance(args, dict) else {}
            for key in ("path", "scope", "pattern", "target", "cmd"):
                if key in args:
                    feature[f"arg1_has_{key}"] = 1.0
    if len(recent) >= 2:
        feature[f"transition={action_name(recent[1])}>{action_name(recent[0])}"] = 1.0
    meta = sample.get("session_meta")
    meta = meta if isinstance(meta, dict) else {}
    workspace = meta.get("workspace")
    workspace = workspace if isinstance(workspace, dict) else {}
    for key, value in (
        ("tier", meta.get("user_tier")),
        ("language", meta.get("language_pref")),
        ("git", workspace.get("git_dirty")),
        ("ci", workspace.get("last_ci_status")),
        ("turn", bucket(meta.get("turn_index"), (1, 2, 3, 5, 8, 12, 16))),
        ("loc", bucket(workspace.get("loc"), (2_000, 8_000, 20_000, 50_000))),
        ("budget", bucket(meta.get("budget_tokens_remaining"), (20_000, 60_000, 120_000, 170_000))),
        ("elapsed", bucket(meta.get("elapsed_session_sec"), (200, 400, 700, 1_000))),
    ):
        feature[f"{key}={str(value).lower()}"] = 1.0
    open_files = workspace.get("open_files")
    feature[f"open_count={bucket(len(open_files) if isinstance(open_files, list) else -1, (0, 1, 2, 4, 8))}"] = 1.0
    language_mix = workspace.get("language_mix")
    if isinstance(language_mix, dict) and language_mix:
        dominant = max(language_mix, key=language_mix.get)
        feature[f"dominant_language={dominant}"] = 1.0
    prompt = str(sample.get("current_prompt") or "")
    feature[f"prompt_words={bucket(len(prompt.split()), (4, 8, 14, 22, 35, 55))}"] = 1.0
    feature[f"prompt_chars={bucket(len(prompt), (25, 50, 90, 150, 240, 400))}"] = 1.0
    # A compact set of stable, intent-oriented cues gives a tree selector a
    # useful escape hatch when base margins are close.  These are deliberately
    # broad phrases, not labels or row-specific identifiers.
    lower = prompt.lower()
    for cue in (
        "test", "spec", "lint", "typecheck", "format", "build", "run", "command",
        "search", "find", "grep", "where", "list", "directory", "folder", "file",
        "read", "open", "show", "write", "create", "add", "edit", "change", "patch",
        "apply", "web", "online", "google", "plan", "steps", "question", "clarify",
    ):
        if cue in lower:
            feature[f"cue={cue}"] = 1.0
    return feature


def clone(estimator):
    return estimator.__class__(**estimator.get_params())


def candidates(trees: int):
    return {
        "Q0_i3_reproduction": RandomForestClassifier(
            n_estimators=trees, min_samples_leaf=10, max_features=0.7,
            class_weight="balanced_subsample", n_jobs=-1, random_state=42,
        ),
        "Q1_rf_state_leaf6": RandomForestClassifier(
            n_estimators=trees, min_samples_leaf=6, max_features=0.55,
            class_weight="balanced_subsample", n_jobs=-1, random_state=42,
        ),
        "Q2_rf_state_leaf10": RandomForestClassifier(
            n_estimators=trees, min_samples_leaf=10, max_features=0.55,
            class_weight="balanced_subsample", n_jobs=-1, random_state=42,
        ),
        "Q3_extra_state_leaf5": ExtraTreesClassifier(
            n_estimators=trees, min_samples_leaf=5, max_features=0.7,
            class_weight="balanced", n_jobs=-1, random_state=42,
        ),
        "Q4_extra_state_leaf10": ExtraTreesClassifier(
            n_estimators=trees, min_samples_leaf=10, max_features=0.7,
            class_weight="balanced", n_jobs=-1, random_state=42,
        ),
    }


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_path = out_dir / "prepared_inputs.npz"
    samples = load_jsonl(str(Path(args.data_dir) / "train.jsonl"))
    labels = load_labels(str(Path(args.data_dir) / "train_labels.csv"))
    split = load_split(args.split_path)
    ids, e_texts = build_modeling_e_features(samples)
    b_ids, b_texts = build_modeling_b_features(samples, history_n=3, include_meta=True)
    if ids != b_ids:
        raise ValueError("Base text builders returned different ID orders")
    val_idx = np.asarray([index for index, sample_id in enumerate(ids) if split[sample_id] == "val"])
    val_ids = [ids[index] for index in val_idx]
    y = np.asarray([labels[sample_id] for sample_id in val_ids])
    groups = np.asarray([sample_id.split("-step_", 1)[0] for sample_id in val_ids])
    if cache_path.exists():
        cache = np.load(cache_path, allow_pickle=False)
        base, x = cache["base"], cache["x"]
        print(f"loaded_cache base={base.shape} combined={x.shape}", flush=True)
    else:
        e_model = joblib.load(args.e_model)
        d_model = joblib.load(args.d_model)
        b_model = joblib.load(args.b_model)
        start = time.time()
        base, schema, _ = stack_features(
            e_model, d_model, b_model,
            [e_texts[index] for index in val_idx],
            [b_texts[index] for index in val_idx],
        )
        vectorizer = DictVectorizer(sparse=False)
        state = vectorizer.fit_transform([state_dict(samples[index]) for index in val_idx]).astype(np.float32)
        x = np.column_stack((base, state)).astype(np.float32)
        print(f"base={base.shape} state={state.shape} combined={x.shape} seconds={time.time()-start:.1f}", flush=True)
        (out_dir / "state_feature_names.json").write_text(json.dumps(vectorizer.feature_names_, ensure_ascii=False, indent=2), encoding="utf-8")
        np.savez_compressed(cache_path, base=base, x=x)
        print(f"saved_cache={cache_path}", flush=True)
    if args.prepare_only:
        return
    folds = list(GroupKFold(n_splits=5).split(x, y, groups))
    rows = []
    best = None
    all_predictions = {}
    candidate_map = candidates(args.trees)
    requested = [item.strip() for item in args.experiments.split(",") if item.strip()]
    if requested:
        unknown = sorted(set(requested) - set(candidate_map))
        if unknown:
            raise ValueError(f"Unknown experiments: {unknown}")
        candidate_map = {name: candidate_map[name] for name in requested}
    for name, prototype in candidate_map.items():
        use_x = base if name == "Q0_i3_reproduction" else x
        prediction = np.empty(len(y), dtype=object)
        fold_scores = []
        started = time.time()
        for train_index, test_index in folds:
            model = clone(prototype)
            model.fit(use_x[train_index], y[train_index])
            prediction[test_index] = model.predict(use_x[test_index])
            fold_scores.append(f1_score(y[test_index], prediction[test_index], labels=ALL_CLASSES, average="macro", zero_division=0))
        score = f1_score(y, prediction, labels=ALL_CLASSES, average="macro", zero_division=0)
        by_kind = {}
        for kind in ("sim", "au"):
            mask = np.asarray([f"sess_{kind}_" in value for value in val_ids])
            if mask.any():
                by_kind[kind] = f1_score(y[mask], prediction[mask], labels=ALL_CLASSES, average="macro", zero_division=0)
        row = {
            "experiment": name, "oof_macro_f1": score,
            "fold_scores": ";".join(f"{value:.6f}" for value in fold_scores),
            "sim_macro_f1": by_kind.get("sim"), "au_macro_f1": by_kind.get("au"),
            "seconds": round(time.time() - started, 2),
        }
        rows.append(row)
        all_predictions[name] = prediction
        print(f"{name} macro_f1={score:.6f} folds={row['fold_scores']} kinds={by_kind}", flush=True)
        if best is None or score > best["score"]:
            best = {"name": name, "score": score, "prediction": prediction, "prototype": prototype}
    with (out_dir / "results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    np.savez_compressed(out_dir / "oof_predictions.npz", ids=np.asarray(val_ids), y=y, **all_predictions)
    (out_dir / "best_classification_report.txt").write_text(
        classification_report(y, best["prediction"], labels=ALL_CLASSES, digits=4, zero_division=0), encoding="utf-8"
    )
    print(f"BEST={best['name']} score={best['score']:.6f}", flush=True)


if __name__ == "__main__":
    main()
