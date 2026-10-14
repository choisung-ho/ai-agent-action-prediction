#!/usr/bin/env python3
"""Modeling E: structured context + hierarchical LinearSVC.

The previous models mixed ordered history and continuous metadata into ordinary
text.  E turns stable state into explicit categorical tokens and uses a deeper
intent tree so that easily confused actions are decided at their own node.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from data import load_jsonl, load_labels
from train_modeling_b import ALL_CLASSES


TREE = {
    "root": {
        "local_explore": ["read_file", "grep_search", "list_directory", "glob_pattern"],
        "modify": ["edit_file", "write_file", "apply_patch"],
        "execute": ["run_bash", "run_tests", "lint_or_typecheck"],
        "coordinate": ["ask_user", "plan_task", "respond_only"],
        "web_search": ["web_search"],
    },
    "local_explore": {
        "direct_access": ["read_file", "list_directory"],
        "pattern_search": ["grep_search", "glob_pattern"],
    },
    "modify": {
        "modify_existing": ["edit_file", "apply_patch"],
        "write_file": ["write_file"],
    },
    "execute": {
        "run_bash": ["run_bash"],
        "quality": ["run_tests", "lint_or_typecheck"],
    },
    "coordinate": {
        "clarify_or_plan": ["ask_user", "plan_task"],
        "respond_only": ["respond_only"],
    },
    "direct_access": {
        "read_file": ["read_file"],
        "list_directory": ["list_directory"],
    },
    "pattern_search": {
        "grep_search": ["grep_search"],
        "glob_pattern": ["glob_pattern"],
    },
    "modify_existing": {
        "edit_file": ["edit_file"],
        "apply_patch": ["apply_patch"],
    },
    "quality": {
        "run_tests": ["run_tests"],
        "lint_or_typecheck": ["lint_or_typecheck"],
    },
    "clarify_or_plan": {
        "ask_user": ["ask_user"],
        "plan_task": ["plan_task"],
    },
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="data")
    p.add_argument("--split-path", default="splits/common_split.csv")
    p.add_argument("--out-dir", default="experiments/modeling_e")
    p.add_argument("--model-dir", default="model")
    p.add_argument("--experiments", default="E0,E1,E2")
    p.add_argument("--refit-full", action="store_true")
    return p.parse_args()


def bucket(value, cuts):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return "unknown"
    for upper, name in cuts:
        if value <= upper:
            return name
    return "high"


def result_status(value):
    text = str(value or "").lower()
    bad = ("fail", "error", "exception", "traceback", "not found", "0 match", "0 file")
    good = ("pass", "ok;", "exit=0", "modified", "patched", "matched", "found", "listed")
    if any(token in text for token in bad):
        return "fail"
    if any(token in text for token in good):
        return "ok"
    return "other"


def build_modeling_e_features(samples):
    ids, texts = [], []
    for sample in samples:
        sample_id = str(sample.get("id", ""))
        prompt = str(sample.get("current_prompt", "") or "").strip().lower()
        history = sample.get("history", [])
        if not isinstance(history, list):
            history = []
        actions = [item for item in history if isinstance(item, dict) and item.get("role") == "assistant_action"]
        reverse_actions = list(reversed(actions[-4:]))
        parts = [f"[CURRENT] {prompt}"]
        for pos, action in enumerate(reverse_actions, start=1):
            name = str(action.get("name", "none")).lower()
            parts.extend((f"pa{pos}_{name}", f"pr{pos}_{result_status(action.get('result_summary'))}"))
            if pos == 1:
                args = action.get("args", {})
                if isinstance(args, dict):
                    for key, value in args.items():
                        if key in {"path", "target", "scope", "pattern", "cmd"}:
                            suffix = Path(str(value)).suffix.lower().lstrip(".") or "none"
                            parts.extend((f"parg_{key}", f"pext_{suffix}"))
        if len(reverse_actions) >= 2:
            older = str(reverse_actions[1].get("name", "none")).lower()
            latest = str(reverse_actions[0].get("name", "none")).lower()
            parts.append(f"trans_{older}_{latest}")

        meta = sample.get("session_meta", {})
        meta = meta if isinstance(meta, dict) else {}
        workspace = meta.get("workspace", {})
        workspace = workspace if isinstance(workspace, dict) else {}
        language_mix = workspace.get("language_mix", {})
        if isinstance(language_mix, dict) and language_mix:
            language_key = "_".join(sorted(language_mix, key=language_mix.get, reverse=True)[:2])
        else:
            language_key = "none"
        open_files = workspace.get("open_files", [])
        open_count = len(open_files) if isinstance(open_files, list) else "unknown"
        parts.extend(
            (
                f"step_{meta.get('turn_index', 'unknown')}",
                f"hlen_{len(history)}",
                f"tier_{meta.get('user_tier', 'unknown')}",
                f"lang_{meta.get('language_pref', 'unknown')}",
                f"langmix_{language_key}",
                f"git_{workspace.get('git_dirty', 'unknown')}",
                f"ci_{workspace.get('last_ci_status', 'unknown')}",
                f"open_{open_count}",
                f"loc_{bucket(workspace.get('loc'), [(2000, 'xs'), (8000, 's'), (20000, 'm'), (50000, 'l')])}",
                f"budget_{bucket(meta.get('budget_tokens_remaining'), [(20000, 'xs'), (60000, 's'), (120000, 'm'), (170000, 'l')])}",
                f"elapsed_{bucket(meta.get('elapsed_session_sec'), [(200, 'xs'), (400, 's'), (700, 'm'), (1000, 'l')])}",
            )
        )
        ids.append(sample_id)
        texts.append(" ".join(parts))
    return ids, texts


def make_svc(c_value, ngram_max=3, max_features=220_000):
    return Pipeline(
        [
            ("tfidf", TfidfVectorizer(ngram_range=(1, ngram_max), min_df=2, max_features=max_features,
                                      sublinear_tf=True, lowercase=False)),
            ("clf", LinearSVC(class_weight="balanced", C=c_value, dual=True)),
        ]
    )


def descendants(node):
    labels = []
    for child, child_labels in TREE[node].items():
        labels.extend(child_labels)
    return set(labels)


def child_for_label(node, label):
    for child, labels in TREE[node].items():
        if label in labels:
            return child
    raise ValueError(f"{label} is not under {node}")


def fit_tree(texts, labels, c, ngram_max):
    models = {}
    defaults = {}
    for node in TREE:
        allowed = descendants(node)
        indices = [i for i, label in enumerate(labels) if label in allowed]
        node_y = [child_for_label(node, labels[i]) for i in indices]
        defaults[node] = Counter(node_y).most_common(1)[0][0]
        model = make_svc(c, ngram_max=ngram_max)
        model.fit([texts[i] for i in indices], node_y)
        models[node] = model
    return {"model_type": "hierarchical_tree_v1", "tree": TREE, "models": models, "defaults": defaults}


def predict_tree(bundle, texts):
    predictions = [None] * len(texts)
    pending = {"root": list(range(len(texts)))}
    while pending:
        node, indices = pending.popitem()
        if not indices:
            continue
        children = bundle["models"][node].predict([texts[i] for i in indices])
        for index, child in zip(indices, children):
            child = str(child)
            labels = bundle["tree"][node].get(child)
            if not labels:
                child = bundle["defaults"][node]
                labels = bundle["tree"][node][child]
            if len(labels) == 1:
                predictions[index] = labels[0]
            else:
                pending.setdefault(child, []).append(index)
    return [str(value) for value in predictions]


def load_split(path):
    with open(path, newline="", encoding="utf-8") as f:
        return {row["id"]: row["split"] for row in csv.DictReader(f)}


def config(exp_id):
    return {
        "E0": {"c": 0.5, "ngram_max": 3},
        "E1": {"c": 0.7, "ngram_max": 3},
        "E2": {"c": 1.0, "ngram_max": 3},
    }[exp_id]


def main():
    args = parse_args()
    samples = load_jsonl(str(Path(args.data_dir) / "train.jsonl"))
    label_map = load_labels(str(Path(args.data_dir) / "train_labels.csv"))
    split = load_split(args.split_path)
    ids, texts = build_modeling_e_features(samples)
    labels = [label_map[sample_id] for sample_id in ids]
    train_idx = [i for i, sample_id in enumerate(ids) if split[sample_id] == "train"]
    val_idx = [i for i, sample_id in enumerate(ids) if split[sample_id] == "val"]
    x_train, y_train = [texts[i] for i in train_idx], [labels[i] for i in train_idx]
    x_val, y_val = [texts[i] for i in val_idx], [labels[i] for i in val_idx]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows, best = [], None
    for exp_id in [x.strip() for x in args.experiments.split(",") if x.strip()]:
        cfg = config(exp_id)
        start = time.time()
        bundle = fit_tree(x_train, y_train, **cfg)
        pred = predict_tree(bundle, x_val)
        score = f1_score(y_val, pred, labels=ALL_CLASSES, average="macro", zero_division=0)
        elapsed = time.time() - start
        joblib.dump(bundle, out_dir / f"{exp_id}.pkl", compress=3)
        (out_dir / f"{exp_id}_classification_report.txt").write_text(
            classification_report(y_val, pred, labels=ALL_CLASSES, digits=4, zero_division=0), encoding="utf-8")
        matrix = confusion_matrix(y_val, pred, labels=ALL_CLASSES)
        (out_dir / f"{exp_id}_confusion_matrix.json").write_text(
            json.dumps({"labels": ALL_CLASSES, "matrix": matrix.tolist()}, ensure_ascii=False, indent=2), encoding="utf-8")
        row = {"experiment": exp_id, "macro_f1": score, "train_seconds": round(elapsed, 3), **cfg}
        rows.append(row)
        print(f"{exp_id} macro_f1={score:.6f} seconds={elapsed:.1f}")
        if best is None or score > best["macro_f1"]:
            best = {**row, "bundle": bundle}
    with (out_dir / "modeling_e_results.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["experiment", "macro_f1", "train_seconds", "c", "ngram_max"])
        writer.writeheader(); writer.writerows(rows)
    if args.refit_full:
        best["bundle"] = fit_tree(texts, labels, c=best["c"], ngram_max=best["ngram_max"])
    model_dir = Path(args.model_dir); model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(best["bundle"], model_dir / "best_model.pkl", compress=3)
    metadata = {"best_experiment": best["experiment"], "best_macro_f1": best["macro_f1"],
                "feature_builder": "modeling_e_structured_v1", "feature_config": {},
                "model_type": "hierarchical_tree_v1", "tree": TREE, "split_path": args.split_path,
                "refit_full": bool(args.refit_full), "all_classes": ALL_CLASSES,
                "results_path": str(out_dir / "modeling_e_results.csv")}
    (model_dir / "best_model_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Best={best['experiment']} macro_f1={best['macro_f1']:.6f}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
