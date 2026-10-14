#!/usr/bin/env python3
"""Experiment R: field-aware lexical action classifiers on the frozen split.

This is deliberately experiment-only: it creates artifacts under the supplied
output directory and never opens or writes ``model/``.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import time
from pathlib import Path

import joblib
import numpy as np
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, f1_score
from sklearn.svm import LinearSVC

from train_modeling_b import ALL_CLASSES
from train_modeling_e import result_status


TOKEN = re.compile(r"[\w가-힣]+", re.UNICODE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--split-path", default="splits/common_split.csv")
    parser.add_argument("--out-dir", default="experiments/modeling_r_text")
    parser.add_argument("--c-values", default="0.35,0.7")
    return parser.parse_args()


def clean(value: object) -> str:
    return " ".join(str(value or "").lower().replace("\x00", " ").split())


def prefixed(text: str, prefix: str) -> str:
    return " ".join(prefix + token for token in TOKEN.findall(text.lower()))


def compact(value: object, limit: int = 160) -> str:
    return clean(value)[:limit]


def feature_fields(sample: dict) -> tuple[str, str, str]:
    """Return word-rich context, current-only text, and history-only text."""
    current = compact(sample.get("current_prompt"), 800)
    history = sample.get("history") if isinstance(sample.get("history"), list) else []
    action_items = [item for item in history if isinstance(item, dict) and item.get("role") == "assistant_action"]
    user_items = [item for item in history if isinstance(item, dict) and item.get("role") == "user"]
    parts = ["CURRENT_RAW " + current, "CURRENT_TOK " + prefixed(current, "cur_")]
    # Retain recent user requests as a distinct lexical field.  The E model
    # only retained action names; pronouns in the current turn often resolve
    # to this preceding request.
    for pos, item in enumerate(reversed(user_items[-3:]), start=1):
        text = compact(item.get("content"), 280)
        parts.extend((f"PREV_USER_{pos} {text}", prefixed(text, f"pu{pos}_")))
    for pos, item in enumerate(reversed(action_items[-5:]), start=1):
        name = clean(item.get("name") or item.get("action") or "none")
        args = item.get("args") if isinstance(item.get("args"), dict) else {}
        arg_text = " ".join(f"{key}_{compact(value, 100)}" for key, value in sorted(args.items()))
        result = compact(item.get("result_summary"), 180)
        parts.extend((
            f"PA_{pos}_{name}",
            f"PR_{pos}_{result_status(result)}",
            f"PA_ARGS_{pos} {arg_text}",
            prefixed(arg_text, f"pa{pos}_"),
        ))
        if pos <= 2:
            parts.extend((f"PA_RESULT_{pos} {result}", prefixed(result, f"pr{pos}_")))
    meta = sample.get("session_meta") if isinstance(sample.get("session_meta"), dict) else {}
    workspace = meta.get("workspace") if isinstance(meta.get("workspace"), dict) else {}
    open_files = workspace.get("open_files") if isinstance(workspace.get("open_files"), list) else []
    parts.extend((
        f"META_TIER_{clean(meta.get('user_tier'))}",
        f"META_LANG_{clean(meta.get('language_pref'))}",
        f"META_TURN_{meta.get('turn_index', 'unknown')}",
        f"META_CI_{clean(workspace.get('last_ci_status'))}",
        f"META_GIT_{workspace.get('git_dirty', 'unknown')}",
        "META_OPEN " + " ".join(str(path) for path in open_files[-4:]),
    ))
    history_text = " ".join(parts[2:])
    return " ".join(parts), current, history_text


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    labels = {row["id"]: row["action"] for row in csv.DictReader((Path(args.data_dir) / "train_labels.csv").open(encoding="utf-8"))}
    split = {row["id"]: row["split"] for row in csv.DictReader(Path(args.split_path).open(encoding="utf-8"))}
    samples = [json.loads(line) for line in (Path(args.data_dir) / "train.jsonl").open(encoding="utf-8") if line.strip()]
    ids = [str(sample["id"]) for sample in samples]
    fields = [feature_fields(sample) for sample in samples]
    train_index = [index for index, sample_id in enumerate(ids) if split[sample_id] == "train"]
    valid_index = [index for index, sample_id in enumerate(ids) if split[sample_id] == "val"]
    y_train = [labels[ids[index]] for index in train_index]
    y_valid = [labels[ids[index]] for index in valid_index]
    contexts = [value[0] for value in fields]
    currents = [value[1] for value in fields]
    histories = [value[2] for value in fields]
    start = time.time()
    # Keep this bounded: another full selector experiment may be resident in
    # memory on the shared workstation.  The fields are complementary, so a
    # moderate vocabulary for each has proved more useful than one huge block.
    word = TfidfVectorizer(ngram_range=(1, 3), min_df=2, max_features=160_000, sublinear_tf=True, lowercase=False, dtype=np.float32)
    current_char = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, max_features=120_000, sublinear_tf=True, lowercase=False, dtype=np.float32)
    history_word = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=80_000, sublinear_tf=True, lowercase=False, dtype=np.float32)
    x_train = hstack((
        word.fit_transform([contexts[index] for index in train_index]),
        current_char.fit_transform([currents[index] for index in train_index]),
        history_word.fit_transform([histories[index] for index in train_index]),
    ), format="csr")
    x_valid = hstack((
        word.transform([contexts[index] for index in valid_index]),
        current_char.transform([currents[index] for index in valid_index]),
        history_word.transform([histories[index] for index in valid_index]),
    ), format="csr")
    print(f"features train={x_train.shape} val={x_valid.shape} seconds={time.time()-start:.1f}", flush=True)
    rows = []
    best = None
    for c_value in (float(value) for value in args.c_values.split(",") if value.strip()):
        name = f"R_text_wordchar_c{str(c_value).replace('.', '')}"
        started = time.time()
        model = LinearSVC(C=c_value, class_weight="balanced", dual=True)
        model.fit(x_train, y_train)
        prediction = model.predict(x_valid).astype(str)
        score = f1_score(y_valid, prediction, labels=ALL_CLASSES, average="macro", zero_division=0)
        print(f"{name} macro_f1={score:.6f} seconds={time.time()-started:.1f}", flush=True)
        rows.append({"experiment": name, "macro_f1": score})
        artifact = out_dir / f"{name}.pkl"
        joblib.dump({"word": word, "current_char": current_char, "history_word": history_word, "classifier": model}, artifact, compress=3)
        (out_dir / f"{name}_classification_report.txt").write_text(
            classification_report(y_valid, prediction, labels=ALL_CLASSES, digits=4, zero_division=0), encoding="utf-8"
        )
        if best is None or score > best["score"]:
            best = {"name": name, "score": score, "prediction": prediction}
    with (out_dir / "results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["experiment", "macro_f1"]); writer.writeheader(); writer.writerows(rows)
    np.savez_compressed(out_dir / "val_predictions.npz", ids=np.asarray([ids[index] for index in valid_index]), y=np.asarray(y_valid), prediction=best["prediction"])
    print(f"BEST={best['name']} macro_f1={best['score']:.6f}", flush=True)


if __name__ == "__main__":
    main()
