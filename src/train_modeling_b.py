#!/usr/bin/env python3
"""Modeling B: history and session_meta experiments.

Modeling B uses the EDA findings:
- previous action transition is a strong signal
- recent action sequence helps next-action prediction
- result_summary pass/fail/error keywords are useful
- selected session_meta fields such as git_dirty, last_ci_status, turn_index,
  and open_files count are useful

Experiments:
- B0: current_prompt + previous action only
- B1: current_prompt + recent 3 history actions/results
- B2: current_prompt + recent 5 history actions/results
- B3: B1 + selected session_meta
- B4: B3 + word+char TF-IDF

The script uses the fixed common split:
    splits/common_split.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import Counter
from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC

from data import load_jsonl, load_labels


ALL_CLASSES = [
    "read_file",
    "grep_search",
    "list_directory",
    "glob_pattern",
    "edit_file",
    "write_file",
    "apply_patch",
    "run_bash",
    "run_tests",
    "lint_or_typecheck",
    "ask_user",
    "plan_task",
    "web_search",
    "respond_only",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Modeling B experiments.")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--split-path", default="splits/common_split.csv")
    parser.add_argument("--out-dir", default="experiments/modeling_b")
    parser.add_argument("--model-dir", default="model")
    parser.add_argument("--best-name", default="best_model.pkl")
    parser.add_argument(
        "--experiments",
        default="B0,B1,B2,B3,B4",
        help="Comma-separated experiment ids to run: B0,B1,B2,B3,B4",
    )
    parser.add_argument("--refit-full", action="store_true", help="Refit the best model on all training data before saving best_model.pkl")
    return parser.parse_args()


def clean_text(text) -> str:
    if not isinstance(text, str):
        text = "" if text is None else str(text)
    return text.strip().lower()


def flatten_value(value) -> str:
    if isinstance(value, dict):
        parts = []
        for key in sorted(value):
            parts.append(f"{key}:{flatten_value(value[key])}")
        return " ".join(parts)
    if isinstance(value, list):
        return " ".join(flatten_value(item) for item in value)
    return clean_text(value)


def extract_action_from_history_item(item: dict) -> str:
    """Robustly extract previous action from multiple possible schemas."""
    if not isinstance(item, dict):
        return "none"
    if item.get("action"):
        return clean_text(item.get("action"))
    if item.get("name"):
        return clean_text(item.get("name"))
    assistant_action = item.get("assistant_action")
    if isinstance(assistant_action, dict):
        return clean_text(assistant_action.get("name") or assistant_action.get("action") or "none")
    if item.get("role") in {"assistant_action", "tool", "action"}:
        return clean_text(item.get("name") or item.get("type") or "none")
    return "none"


def extract_result_from_history_item(item: dict) -> str:
    """Robustly extract result summary from multiple possible schemas."""
    if not isinstance(item, dict):
        return "none"
    for key in ("result", "result_summary", "summary", "observation", "content"):
        if key in item and item[key] is not None:
            return clean_text(item[key])
    assistant_action = item.get("assistant_action")
    if isinstance(assistant_action, dict):
        for key in ("result_summary", "result", "summary"):
            if key in assistant_action and assistant_action[key] is not None:
                return clean_text(assistant_action[key])
    return "none"


def selected_meta_tokens(session_meta: dict) -> list[str]:
    if not isinstance(session_meta, dict):
        return []
    tokens: list[str] = []
    workspace = session_meta.get("workspace", {})
    if not isinstance(workspace, dict):
        workspace = {}

    selected = {
        "user_tier": session_meta.get("user_tier"),
        "language_pref": session_meta.get("language_pref"),
        "budget_tokens_remaining": session_meta.get("budget_tokens_remaining"),
        "turn_index": session_meta.get("turn_index"),
        "elapsed_session_sec": session_meta.get("elapsed_session_sec"),
        "workspace_language_mix": workspace.get("language_mix"),
        "workspace_loc": workspace.get("loc"),
        "git_dirty": workspace.get("git_dirty"),
        "last_ci_status": workspace.get("last_ci_status"),
    }
    open_files = workspace.get("open_files")
    if isinstance(open_files, list):
        selected["open_files_count"] = len(open_files)
        selected["open_files_ext"] = " ".join(Path(str(path)).suffix.lower() for path in open_files if Path(str(path)).suffix)
    elif open_files is not None:
        selected["open_files"] = open_files

    for key, value in selected.items():
        if value is not None:
            tokens.append(f"meta_{key}:{flatten_value(value)}")
    return tokens


def build_modeling_b_features(samples: list[dict], history_n: int = 3, include_meta: bool = True) -> tuple[list[str], list[str]]:
    ids: list[str] = []
    contexts: list[str] = []
    for sample in samples:
        ids.append(str(sample.get("id", "")))
        parts = [f"[CURRENT] {clean_text(sample.get('current_prompt', ''))}"]

        history = sample.get("history", [])
        if not isinstance(history, list):
            history = []
        recent = history[-history_n:] if history_n > 0 else []
        if recent:
            hist_tokens = []
            for item in recent:
                action = extract_action_from_history_item(item)
                result = extract_result_from_history_item(item)
                hist_tokens.append(f"[PREV_ACTION] {action} [PREV_RESULT] {result[:120]}")
            parts.append("[HISTORY] " + " ".join(hist_tokens))
        else:
            parts.append("[HISTORY_EMPTY]")

        if include_meta:
            meta = sample.get("session_meta", {})
            meta_tokens = selected_meta_tokens(meta)
            if meta_tokens:
                parts.append("[META] " + " ".join(meta_tokens))
            else:
                parts.append("[META_EMPTY]")

        contexts.append(" ".join(parts))
    return ids, contexts


def load_split(path: Path) -> dict[str, str]:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing common split: {path}. Run "
            "python src/make_common_split.py --data-dir data --out-dir splits --strategy group --val-size 0.2 --seed 42"
        )
    split_map: dict[str, str] = {}
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or "id" not in reader.fieldnames or "split" not in reader.fieldnames:
            raise ValueError(f"{path} must contain id,split columns")
        for row in reader:
            split = row["split"]
            if split not in {"train", "val"}:
                raise ValueError(f"Invalid split value for id={row['id']}: {split}")
            split_map[row["id"]] = split
    return split_map


def validate_dataset(samples: list[dict], labels: dict[str, str], split_map: dict[str, str]) -> None:
    if not samples:
        raise ValueError("train.jsonl is empty. Put the official data into data/.")
    if not labels:
        raise ValueError("train_labels.csv is empty. Put the official labels into data/.")
    sample_ids = {str(sample["id"]) for sample in samples}
    missing_labels = sample_ids - set(labels)
    missing_split = sample_ids - set(split_map)
    if missing_labels:
        raise ValueError(f"{len(missing_labels)} samples have no label. Example: {next(iter(missing_labels))}")
    if missing_split:
        raise ValueError(f"{len(missing_split)} samples are missing from common_split.csv. Example: {next(iter(missing_split))}")
    unknown_labels = sorted(set(labels.values()) - set(ALL_CLASSES))
    if unknown_labels:
        raise ValueError(f"Unknown labels found: {unknown_labels}")


def make_classifier(exp_id: str):
    if exp_id in {"B0", "B1", "B2", "B3"}:
        return Pipeline(
            [
                (
                    "tfidf",
                    TfidfVectorizer(
                        analyzer="word",
                        ngram_range=(1, 2),
                        min_df=2,
                        max_features=120_000,
                        sublinear_tf=True,
                        lowercase=False,
                    ),
                ),
                ("clf", LinearSVC(class_weight="balanced", C=1.0)),
            ]
        )
    if exp_id == "B4":
        return Pipeline(
            [
                (
                    "features",
                    FeatureUnion(
                        [
                            (
                                "word",
                                TfidfVectorizer(
                                    analyzer="word",
                                    ngram_range=(1, 2),
                                    min_df=2,
                                    max_features=120_000,
                                    sublinear_tf=True,
                                    lowercase=False,
                                ),
                            ),
                            (
                                "char",
                                TfidfVectorizer(
                                    analyzer="char_wb",
                                    ngram_range=(3, 5),
                                    min_df=2,
                                    max_features=120_000,
                                    sublinear_tf=True,
                                    lowercase=False,
                                ),
                            ),
                        ]
                    ),
                ),
                ("clf", LogisticRegression(max_iter=900, class_weight="balanced", C=2.0, n_jobs=-1)),
            ]
        )
    raise ValueError(f"Unknown experiment: {exp_id}")


def feature_config(exp_id: str) -> dict:
    return {
        "B0": {"history_n": 1, "include_meta": False},
        "B1": {"history_n": 3, "include_meta": False},
        "B2": {"history_n": 5, "include_meta": False},
        "B3": {"history_n": 3, "include_meta": True},
        "B4": {"history_n": 3, "include_meta": True},
    }[exp_id]


def save_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    model_dir = Path(args.model_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    samples = load_jsonl(str(data_dir / "train.jsonl"))
    labels = load_labels(str(data_dir / "train_labels.csv"))
    split_map = load_split(Path(args.split_path))
    validate_dataset(samples, labels, split_map)

    requested = [item.strip() for item in args.experiments.split(",") if item.strip()]
    unknown = [item for item in requested if item not in {"B0", "B1", "B2", "B3", "B4"}]
    if unknown:
        raise ValueError(f"Unknown experiment ids: {unknown}")

    sample_ids = [str(sample["id"]) for sample in samples]
    y_all = [labels[sample_id] for sample_id in sample_ids]
    train_idx = [i for i, sample_id in enumerate(sample_ids) if split_map[sample_id] == "train"]
    val_idx = [i for i, sample_id in enumerate(sample_ids) if split_map[sample_id] == "val"]
    if not train_idx or not val_idx:
        raise ValueError("Common split must contain both train and val samples.")

    print(f"samples={len(samples)} train={len(train_idx)} val={len(val_idx)}")
    print(f"train_distribution={dict(Counter(y_all[i] for i in train_idx))}")
    print(f"val_distribution={dict(Counter(y_all[i] for i in val_idx))}")

    rows: list[dict] = []
    best_score = -1.0
    best_exp = None
    best_model = None
    best_config = None

    for exp_id in requested:
        config = feature_config(exp_id)
        ids, texts = build_modeling_b_features(samples, **config)
        x_train = [texts[i] for i in train_idx]
        y_train = [y_all[i] for i in train_idx]
        x_val = [texts[i] for i in val_idx]
        y_val = [y_all[i] for i in val_idx]
        model = make_classifier(exp_id)

        print(f"\n===== {exp_id} start config={config} =====")
        start = time.time()
        model.fit(x_train, y_train)
        train_seconds = time.time() - start
        preds = [str(p) for p in model.predict(x_val)]
        invalid = sorted(set(preds) - set(ALL_CLASSES))
        if invalid:
            raise ValueError(f"{exp_id} produced invalid labels: {invalid}")
        macro_f1 = f1_score(y_val, preds, labels=ALL_CLASSES, average="macro", zero_division=0)
        report = classification_report(y_val, preds, labels=ALL_CLASSES, zero_division=0)
        cm = confusion_matrix(y_val, preds, labels=ALL_CLASSES)
        print(f"{exp_id} macro_f1={macro_f1:.6f} train_seconds={train_seconds:.1f}")

        joblib.dump(model, out_dir / f"{exp_id}.pkl", compress=3)
        save_text(out_dir / f"{exp_id}_classification_report.txt", report)
        save_text(out_dir / f"{exp_id}_confusion_matrix.json", json.dumps({"labels": ALL_CLASSES, "matrix": cm.tolist()}, ensure_ascii=False, indent=2))

        rows.append(
            {
                "experiment": exp_id,
                "macro_f1": macro_f1,
                "train_seconds": round(train_seconds, 3),
                "n_train": len(x_train),
                "n_val": len(x_val),
                "history_n": config["history_n"],
                "include_meta": config["include_meta"],
                "artifact": str(out_dir / f"{exp_id}.pkl"),
            }
        )
        if macro_f1 > best_score:
            best_score = macro_f1
            best_exp = exp_id
            best_model = model
            best_config = config

    results_path = out_dir / "modeling_b_results.csv"
    with results_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["experiment", "macro_f1", "train_seconds", "n_train", "n_val", "history_n", "include_meta", "artifact"],
        )
        writer.writeheader()
        writer.writerows(rows)

    if best_model is None or best_exp is None or best_config is None:
        raise RuntimeError("No experiment was trained.")

    if args.refit_full:
        print(f"\nRefitting best experiment {best_exp} on full training data...")
        _, full_texts = build_modeling_b_features(samples, **best_config)
        full_model = make_classifier(best_exp)
        full_model.fit(full_texts, y_all)
        best_model = full_model

    best_path = model_dir / args.best_name
    joblib.dump(best_model, best_path, compress=3)
    metadata = {
        "best_experiment": best_exp,
        "best_macro_f1": best_score,
        "feature_builder": "modeling_b_v1",
        "feature_config": best_config,
        "split_path": args.split_path,
        "refit_full": args.refit_full,
        "all_classes": ALL_CLASSES,
        "results_path": str(results_path),
    }
    save_text(model_dir / "best_model_metadata.json", json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")
    print(f"\nBest experiment: {best_exp} macro_f1={best_score:.6f} config={best_config}")
    print(f"Saved best model: {best_path}")
    print(f"Saved metadata: {model_dir / 'best_model_metadata.json'}")
    print(f"Saved results: {results_path}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)
