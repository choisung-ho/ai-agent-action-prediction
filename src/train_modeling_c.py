#!/usr/bin/env python3
"""Modeling C: richer structured-history text model.

Goal:
- Keep the submission stack simple and fast.
- Use information discarded by Modeling B:
  user messages in history, action args/paths, file extensions, full meta,
  step index, and explicit intent keyword tokens.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import time
from collections import Counter
from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC

from data import load_jsonl, load_labels


ALL_CLASSES = [
    "read_file", "grep_search", "list_directory", "glob_pattern",
    "edit_file", "write_file", "apply_patch", "run_bash",
    "run_tests", "lint_or_typecheck", "ask_user", "plan_task",
    "web_search", "respond_only",
]

PATH_RE = re.compile(r"[\w./@+-]+\.(?:py|js|ts|tsx|jsx|json|ya?ml|toml|md|txt|csv|html|css|scss|sql|go|rs|java|kt|swift|c|cc|cpp|h|hpp|sh|dockerfile|lock)", re.I)
BACKTICK_RE = re.compile(r"`([^`]{1,120})`")


def parse_args():
    parser = argparse.ArgumentParser(description="Run Modeling C rich-history experiments.")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--split-path", default="splits/common_split.csv")
    parser.add_argument("--out-dir", default="experiments/modeling_c")
    parser.add_argument("--model-dir", default="model")
    parser.add_argument("--best-name", default="best_model.pkl")
    parser.add_argument("--experiments", default="C0,C1,C2")
    parser.add_argument("--refit-full", action="store_true")
    return parser.parse_args()


def clean_text(value) -> str:
    if not isinstance(value, str):
        value = "" if value is None else str(value)
    return value.strip().lower()


def flatten_value(value, max_len: int = 500) -> str:
    if isinstance(value, dict):
        return " ".join(f"{clean_text(k)}:{flatten_value(v, max_len=max_len)}" for k, v in sorted(value.items()))
    if isinstance(value, list):
        return " ".join(flatten_value(v, max_len=max_len) for v in value[:20])
    text = clean_text(value)
    return text[:max_len]


def step_number(sample_id: str) -> str:
    if "-step_" not in sample_id:
        return "unknown"
    return sample_id.rsplit("-step_", 1)[-1]


def bucket_number(value, buckets):
    try:
        x = float(value)
    except Exception:
        return "unknown"
    for upper, name in buckets:
        if x <= upper:
            return name
    return "high"


def keyword_tokens(text: str) -> list[str]:
    text = clean_text(text)
    groups = {
        "kw_read": ["open", "read", "show", "cat", "look at", "inspect", "확인", "열어", "읽"],
        "kw_search": ["search", "grep", "find", "where", "locate", "검색", "찾"],
        "kw_list": ["list", "ls", "directory", "tree", "files", "폴더", "목록"],
        "kw_edit": ["edit", "change", "modify", "fix", "update", "refactor", "수정", "고쳐", "변경"],
        "kw_write": ["create", "write", "new file", "save", "generate", "작성", "생성"],
        "kw_patch": ["patch", "apply", "diff"],
        "kw_run": ["run", "execute", "bash", "command", "install", "실행"],
        "kw_test": ["test", "pytest", "unit", "spec", "검증", "테스트"],
        "kw_lint": ["lint", "typecheck", "mypy", "tsc", "ruff", "eslint", "타입"],
        "kw_plan": ["plan", "steps", "break down", "strategy", "계획", "단계"],
        "kw_ask": ["ask", "clarify", "question", "confirm", "물어", "확인해줘"],
        "kw_web": ["web", "internet", "latest", "docs", "browser", "검색해", "최신"],
    }
    tokens = []
    for token, words in groups.items():
        if any(word in text for word in words):
            tokens.append(token)
    for match in PATH_RE.findall(text):
        suffix = Path(match).suffix.lower().lstrip(".") or "file"
        tokens.append(f"file_ext_{suffix}")
    for match in BACKTICK_RE.findall(text):
        tokens.append("bt_" + clean_text(match).replace(" ", "_")[:80])
    return tokens


def history_item_tokens(item: dict, pos: int) -> list[str]:
    if not isinstance(item, dict):
        return [f"h{pos}_baditem"]
    role = clean_text(item.get("role", "none"))
    tokens = [f"h{pos}_role_{role}"]
    if role == "user":
        content = clean_text(item.get("content", ""))
        tokens.append(f"[H{pos}_USER] {content[:500]}")
        tokens.extend(f"h{pos}_{tok}" for tok in keyword_tokens(content))
    elif role == "assistant_action":
        name = clean_text(item.get("name", "none"))
        args = flatten_value(item.get("args", {}), max_len=350)
        result = clean_text(item.get("result_summary", ""))[:300]
        tokens.append(f"h{pos}_action_{name}")
        tokens.append(f"[H{pos}_ARGS] {args}")
        tokens.append(f"[H{pos}_RESULT] {result}")
        tokens.extend(f"h{pos}_{tok}" for tok in keyword_tokens(args + " " + result))
    else:
        tokens.append(f"[H{pos}_RAW] {flatten_value(item, max_len=600)}")
    return tokens


def meta_tokens(meta: dict) -> list[str]:
    if not isinstance(meta, dict):
        return ["meta_empty"]
    workspace = meta.get("workspace", {})
    if not isinstance(workspace, dict):
        workspace = {}
    tokens = [
        f"user_tier_{clean_text(meta.get('user_tier', 'none'))}",
        f"lang_pref_{clean_text(meta.get('language_pref', 'none'))}",
        f"budget_{bucket_number(meta.get('budget_tokens_remaining'), [(30_000, 'low'), (80_000, 'mid'), (140_000, 'normal')])}",
        f"turn_{bucket_number(meta.get('turn_index'), [(1, 'first'), (3, 'early'), (7, 'mid'), (12, 'late')])}",
        f"elapsed_{bucket_number(meta.get('elapsed_session_sec'), [(120, 'short'), (600, 'mid'), (1500, 'long')])}",
        f"git_dirty_{clean_text(workspace.get('git_dirty', 'none'))}",
        f"ci_{clean_text(workspace.get('last_ci_status', 'none'))}",
        f"loc_{bucket_number(workspace.get('loc'), [(1000, 'tiny'), (8000, 'small'), (30000, 'medium'), (100000, 'large')])}",
    ]
    language_mix = workspace.get("language_mix")
    if isinstance(language_mix, dict):
        for lang, ratio in sorted(language_mix.items()):
            tokens.append(f"mix_{clean_text(lang)}")
            try:
                if float(ratio) >= 0.3:
                    tokens.append(f"mainlang_{clean_text(lang)}")
            except Exception:
                pass
    open_files = workspace.get("open_files")
    if isinstance(open_files, list):
        tokens.append(f"open_count_{bucket_number(len(open_files), [(0, 'zero'), (2, 'few'), (6, 'some')])}")
        for path in open_files[:8]:
            suffix = Path(str(path)).suffix.lower().lstrip(".")
            if suffix:
                tokens.append(f"open_ext_{suffix}")
            tokens.append("open_path_" + clean_text(str(path)).replace("/", "_")[:120])
    return tokens


def build_modeling_c_features(samples: list[dict], history_items: int = 6, include_full_meta: bool = True) -> tuple[list[str], list[str]]:
    ids, texts = [], []
    for sample in samples:
        sample_id = str(sample.get("id", ""))
        prompt = clean_text(sample.get("current_prompt", ""))
        history = sample.get("history", [])
        if not isinstance(history, list):
            history = []

        parts = [
            f"id_step_{step_number(sample_id)}",
            f"hist_len_{bucket_number(len(history), [(0, 'zero'), (2, 'two'), (4, 'four'), (8, 'eight'), (12, 'twelve')])}",
            f"[CURRENT] {prompt}",
            " ".join(keyword_tokens(prompt)),
        ]

        recent = history[-history_items:] if history_items else []
        for pos, item in enumerate(recent, start=1):
            parts.extend(history_item_tokens(item, pos))
        if not recent:
            parts.append("history_empty")

        if include_full_meta:
            parts.extend(meta_tokens(sample.get("session_meta", {})))

        ids.append(sample_id)
        texts.append(" ".join(part for part in parts if part))
    return ids, texts


def load_split(path: Path) -> dict[str, str]:
    split_map = {}
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            split_map[row["id"]] = row["split"]
    return split_map


def validate_dataset(samples, labels, split_map):
    sample_ids = {str(s["id"]) for s in samples}
    missing_labels = sample_ids - set(labels)
    missing_split = sample_ids - set(split_map)
    if missing_labels:
        raise ValueError(f"Missing labels: {len(missing_labels)}")
    if missing_split:
        raise ValueError(f"Missing split: {len(missing_split)}")
    unknown = sorted(set(labels.values()) - set(ALL_CLASSES))
    if unknown:
        raise ValueError(f"Unknown labels: {unknown}")


def make_classifier(exp_id: str):
    if exp_id == "C0":
        return Pipeline([
            ("tfidf", TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2, max_features=180_000, sublinear_tf=True, lowercase=False)),
            ("clf", LinearSVC(class_weight="balanced", C=1.0)),
        ])
    if exp_id == "C1":
        return Pipeline([
            ("features", FeatureUnion([
                ("word", TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2, max_features=160_000, sublinear_tf=True, lowercase=False)),
                ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, max_features=80_000, sublinear_tf=True, lowercase=False)),
            ])),
            ("clf", LinearSVC(class_weight="balanced", C=0.8)),
        ])
    if exp_id == "C2":
        return Pipeline([
            ("tfidf", TfidfVectorizer(analyzer="word", ngram_range=(1, 3), min_df=2, max_features=220_000, sublinear_tf=True, lowercase=False)),
            ("clf", LinearSVC(class_weight="balanced", C=0.7)),
        ])
    raise ValueError(f"Unknown experiment: {exp_id}")


def save_text(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def main():
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

    ids, texts = build_modeling_c_features(samples)
    y = [labels[sample_id] for sample_id in ids]
    train_idx = [i for i, sample_id in enumerate(ids) if split_map[sample_id] == "train"]
    val_idx = [i for i, sample_id in enumerate(ids) if split_map[sample_id] == "val"]
    x_train, y_train = [texts[i] for i in train_idx], [y[i] for i in train_idx]
    x_val, y_val = [texts[i] for i in val_idx], [y[i] for i in val_idx]

    print(f"samples={len(samples)} train={len(x_train)} val={len(x_val)}")
    print(f"train_distribution={dict(Counter(y_train))}")
    print(f"val_distribution={dict(Counter(y_val))}")

    results = []
    requested = [x.strip() for x in args.experiments.split(",") if x.strip()]
    for exp_id in requested:
        print(f"\n===== {exp_id} start =====")
        model = make_classifier(exp_id)
        start = time.time()
        model.fit(x_train, y_train)
        train_seconds = time.time() - start
        pred = model.predict(x_val)
        macro = f1_score(y_val, pred, average="macro")
        artifact = out_dir / f"{exp_id}.pkl"
        joblib.dump(model, artifact)
        save_text(out_dir / f"{exp_id}_classification_report.txt", classification_report(y_val, pred, labels=ALL_CLASSES, digits=4, zero_division=0))
        cm = confusion_matrix(y_val, pred, labels=ALL_CLASSES)
        (out_dir / f"{exp_id}_confusion_matrix.json").write_text(json.dumps({"labels": ALL_CLASSES, "matrix": cm.tolist()}, ensure_ascii=False, indent=2), encoding="utf-8")
        results.append({"experiment": exp_id, "macro_f1": macro, "train_seconds": round(train_seconds, 3), "n_train": len(x_train), "n_val": len(x_val), "artifact": str(artifact)})
        print(f"{exp_id} macro_f1={macro:.6f} train_seconds={train_seconds:.1f}")

    results_path = out_dir / "modeling_c_results.csv"
    with results_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["experiment", "macro_f1", "train_seconds", "n_train", "n_val", "artifact"])
        writer.writeheader()
        writer.writerows(results)

    if results:
        best = max(results, key=lambda r: r["macro_f1"])
        best_model = joblib.load(best["artifact"])
        if args.refit_full:
            print(f"\nRefitting {best['experiment']} on full training data...")
            best_model.fit(texts, y)
        joblib.dump(best_model, model_dir / args.best_name)
        metadata = {
            "best_experiment": best["experiment"],
            "best_macro_f1": best["macro_f1"],
            "feature_builder": "modeling_c_rich_v1",
            "feature_config": {"history_items": 6, "include_full_meta": True},
            "split_path": args.split_path,
            "refit_full": bool(args.refit_full),
            "all_classes": ALL_CLASSES,
            "results_path": str(results_path),
        }
        (model_dir / "best_model_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nBest={best['experiment']} macro_f1={best['macro_f1']:.6f}")
        print(f"Saved best model to {model_dir / args.best_name}")


if __name__ == "__main__":
    main()
