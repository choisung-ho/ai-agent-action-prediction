#!/usr/bin/env python3
"""Modeling H: CatBoost over text, ordered history, categorical and numeric state."""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import classification_report, f1_score

from train_modeling_b import ALL_CLASSES


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="data")
    p.add_argument("--split-path", default="splits/common_split.csv")
    p.add_argument("--out-dir", default="experiments/modeling_h")
    p.add_argument("--iterations", type=int, default=1200)
    p.add_argument("--depth", type=int, default=8)
    p.add_argument("--learning-rate", type=float, default=0.08)
    p.add_argument("--early-stopping-rounds", type=int, default=150)
    p.add_argument("--include-last-user", action="store_true")
    return p.parse_args()


def result_status(value):
    text = str(value or "").lower()
    if any(token in text for token in ("fail", "error", "exception", "traceback", "not found", "0 match", "0 file")):
        return "fail"
    if any(token in text for token in ("pass", "ok;", "exit=0", "modified", "patched", "matched", "found", "listed")):
        return "ok"
    return "other"


def safe_number(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def sample_features(sample, include_last_user=False):
    history = sample.get("history", [])
    history = history if isinstance(history, list) else []
    actions = [item for item in history if isinstance(item, dict) and item.get("role") == "assistant_action"]
    users = [item for item in history if isinstance(item, dict) and item.get("role") == "user"]
    reverse_actions = list(reversed(actions[-4:]))
    meta = sample.get("session_meta", {})
    meta = meta if isinstance(meta, dict) else {}
    workspace = meta.get("workspace", {})
    workspace = workspace if isinstance(workspace, dict) else {}
    language_mix = workspace.get("language_mix", {})
    if isinstance(language_mix, dict) and language_mix:
        ordered_languages = sorted(language_mix, key=language_mix.get, reverse=True)
        primary_language = ordered_languages[0]
        language_profile = "_".join(ordered_languages[:2])
    else:
        primary_language = "unknown"
        language_profile = "unknown"
    open_files = workspace.get("open_files", [])
    open_files = open_files if isinstance(open_files, list) else []
    row = {
        "current_prompt": str(sample.get("current_prompt", "") or "").strip().lower(),
        "source": "au" if str(sample.get("id", "")).startswith("sess_au_") else "sim",
        "user_tier": str(meta.get("user_tier", "unknown")),
        "language_pref": str(meta.get("language_pref", "unknown")),
        "primary_language": primary_language,
        "language_profile": language_profile,
        "git_dirty": str(workspace.get("git_dirty", "unknown")),
        "last_ci_status": str(workspace.get("last_ci_status", "unknown")),
        "turn_index_cat": str(meta.get("turn_index", "unknown")),
        "history_length": str(len(history)),
        "open_count": str(len(open_files)),
        "budget_tokens": safe_number(meta.get("budget_tokens_remaining")),
        "elapsed_seconds": safe_number(meta.get("elapsed_session_sec")),
        "workspace_loc": safe_number(workspace.get("loc")),
    }
    for position in range(1, 5):
        if position <= len(reverse_actions):
            action = reverse_actions[position - 1]
            row[f"prev_action_{position}"] = str(action.get("name", "none"))
            row[f"prev_status_{position}"] = result_status(action.get("result_summary"))
        else:
            row[f"prev_action_{position}"] = "none"
            row[f"prev_status_{position}"] = "none"
    row["action_transition"] = row["prev_action_2"] + "__" + row["prev_action_1"]
    if include_last_user:
        row["last_user_prompt"] = str(users[-1].get("content", "") if users else "").strip().lower()
    return row


def main():
    args = parse_args()
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    samples = [json.loads(line) for line in open(Path(args.data_dir) / "train.jsonl", encoding="utf-8") if line.strip()]
    with open(Path(args.data_dir) / "train_labels.csv", newline="", encoding="utf-8") as f:
        labels = {row["id"]: row["action"] for row in csv.DictReader(f)}
    with open(args.split_path, newline="", encoding="utf-8") as f:
        split = {row["id"]: row["split"] for row in csv.DictReader(f)}
    frame = pd.DataFrame([sample_features(sample, args.include_last_user) for sample in samples])
    target = pd.Series([labels[sample["id"]] for sample in samples])
    train_idx = [i for i, sample in enumerate(samples) if split[sample["id"]] == "train"]
    val_idx = [i for i, sample in enumerate(samples) if split[sample["id"]] == "val"]
    text_features = ["current_prompt"] + (["last_user_prompt"] if args.include_last_user else [])
    numeric_features = {"budget_tokens", "elapsed_seconds", "workspace_loc"}
    categorical_features = [column for column in frame.columns if column not in numeric_features and column not in text_features]
    model = CatBoostClassifier(
        iterations=args.iterations,
        depth=args.depth,
        learning_rate=args.learning_rate,
        loss_function="MultiClass",
        eval_metric="TotalF1:average=Macro",
        auto_class_weights="Balanced",
        l2_leaf_reg=5.0,
        random_strength=0.5,
        random_seed=42,
        thread_count=-1,
        verbose=50,
        allow_writing_files=False,
    )
    start = time.time()
    model.fit(
        frame.iloc[train_idx], target.iloc[train_idx],
        cat_features=categorical_features, text_features=text_features,
        eval_set=(frame.iloc[val_idx], target.iloc[val_idx]),
        early_stopping_rounds=args.early_stopping_rounds,
        use_best_model=True,
    )
    predictions = [str(value) for value in model.predict(frame.iloc[val_idx]).reshape(-1)]
    score = f1_score(target.iloc[val_idx], predictions, labels=ALL_CLASSES, average="macro", zero_division=0)
    elapsed = time.time() - start
    suffix = "H1" if args.include_last_user else "H0"
    model.save_model(str(out_dir / f"{suffix}.cbm"))
    (out_dir / f"{suffix}_classification_report.txt").write_text(
        classification_report(target.iloc[val_idx], predictions, labels=ALL_CLASSES, digits=4, zero_division=0), encoding="utf-8")
    (out_dir / f"{suffix}_metadata.json").write_text(json.dumps({
        "experiment": suffix, "macro_f1": score, "seconds": elapsed,
        "best_iteration": model.get_best_iteration(), "text_features": text_features,
        "categorical_features": categorical_features, "feature_columns": list(frame.columns),
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"{suffix} macro_f1={score:.6f} best_iteration={model.get_best_iteration()} seconds={elapsed:.1f}")


if __name__ == "__main__":
    main()
