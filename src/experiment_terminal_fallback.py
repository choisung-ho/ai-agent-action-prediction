#!/usr/bin/env python3
"""Validate a batch-safe last-observed-session fallback model.

The public inference input can contain several records from the same session.
Rows which are not exposed by a *future* row's history are normally the
maximum observed step for their session.  This experiment trains a separate
model for only those rows, without changing ``model/``, ``script.py`` or a
submission ZIP.

Besides the usual group-OOF base margins, it uses only information present in
the supplied JSONL batch:

* aligned assistant actions/results/argument keys from *all* raw histories in
  the same session, which reconstruct a longer preceding sequence than the
  six-action suffix in the current row; and
* current prompts from preceding rows in that same session.

No train label is used to build those reconstructed features.  The fixed
common split keeps sessions wholly in either train or validation, so the
reported terminal score is group-safe.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.feature_extraction import FeatureHasher
from sklearn.metrics import classification_report, f1_score
from xgboost import XGBClassifier

from train_modeling_b import ALL_CLASSES

# ``script.py`` is intentionally reused only for the public inference feature
# builders.  Make the AI_SW root importable when this file is run as
# ``python src/experiment_terminal_fallback.py``.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


TOKEN_RE = re.compile(r"[\w가-힣]+", re.UNICODE)
CLASS_TO_ID = {label: index for index, label in enumerate(ALL_CLASSES)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/train.jsonl")
    parser.add_argument("--labels", default="data/train_labels.csv")
    parser.add_argument("--split", default="splits/common_split.csv")
    parser.add_argument("--base-oof", default="experiments/modeling_n_refit/oof_stack_features.npy")
    parser.add_argument("--state-bundle", default="model_candidate_q/best_model.pkl")
    parser.add_argument("--fallback-oof", default="experiments/modeling_ac_tune2/oof_probabilities.npy")
    parser.add_argument("--source-calibration", default="experiments/modeling_ac_tune2/source_calibration.json")
    parser.add_argument("--out-dir", default="experiments/terminal_fallback")
    parser.add_argument("--trees", type=int, default=700)
    parser.add_argument("--depth", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=0.035)
    parser.add_argument("--weight-power", type=float, default=0.65)
    parser.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    parser.add_argument("--stitch-dim", type=int, default=2048)
    parser.add_argument("--no-stitch", action="store_true")
    parser.add_argument("--run-nonrespond", action="store_true")
    return parser.parse_args()


def read_rows(path: str) -> list[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def read_labels(path: str) -> dict[str, str]:
    with Path(path).open(encoding="utf-8") as handle:
        return {row["id"]: row["action"] for row in csv.DictReader(handle)}


def read_split(path: str) -> dict[str, str]:
    with Path(path).open(encoding="utf-8") as handle:
        return {row["id"]: row["split"] for row in csv.DictReader(handle)}


def parse_id(sample_id: str) -> tuple[str, int]:
    session, marker, step = str(sample_id).rpartition("-step_")
    if not marker or not step.isdigit():
        raise ValueError(f"Unparseable id: {sample_id!r}")
    return session, int(step)


def result_bucket(value: object) -> str:
    text = str(value or "").lower()
    if any(token in text for token in ("fail", "error", "exception", "traceback", "not found", "0 match", "0 file")):
        return "fail"
    if any(token in text for token in ("pass", "ok;", "exit=0", "modified", "patched", "matched", "found", "listed")):
        return "ok"
    return "other"


def assistant_actions(sample: dict[str, Any]) -> list[dict[str, Any]]:
    history = sample.get("history")
    if not isinstance(history, list):
        return []
    values: list[dict[str, Any]] = []
    for item in history:
        if not isinstance(item, dict) or item.get("role") != "assistant_action":
            continue
        name = str(item.get("name") or item.get("action") or "").strip().lower()
        if name:
            values.append(item)
    return values


def token_features(record: dict[str, float | int], prefix: str, text: object, limit: int = 80) -> None:
    tokens = TOKEN_RE.findall(str(text or "").lower())[:limit]
    for token in tokens:
        if len(token) > 1:
            record[f"{prefix}w:{token}"] = 1
    for left, right in zip(tokens, tokens[1:]):
        record[f"{prefix}b:{left}_{right}"] = 1


def bucket(value: int, cuts: tuple[int, ...]) -> str:
    for index, upper in enumerate(cuts):
        if value <= upper:
            return str(index)
    return str(len(cuts))


def stitched_batch_records(samples: list[dict[str, Any]]) -> tuple[np.ndarray, list[dict[str, float | int]], dict[str, Any]]:
    """Build inference-safe features aligned solely from raw batch fields.

    For a history in step ``t`` containing ``k`` assistant actions, the action
    suffix corresponds to steps ``t-k`` through ``t-1``.  Aggregating those
    suffixes across a session lets the last supplied row see a longer *past*
    sequence.  The target action itself is never inserted unless a later raw
    row exposes it, in which case the live deterministic-recovery path already
    handles that non-terminal row.
    """
    grouped: dict[str, list[tuple[int, int, dict[str, Any]]]] = defaultdict(list)
    for index, sample in enumerate(samples):
        session, step = parse_id(str(sample.get("id", "")))
        grouped[session].append((step, index, sample))

    terminal = np.zeros(len(samples), dtype=bool)
    records: list[dict[str, float | int]] = [{} for _ in samples]
    diagnostics: dict[str, Any] = {"sessions": len(grouped), "history_action_conflicts": 0, "terminal_rows": 0}

    for entries in grouped.values():
        entries.sort(key=lambda value: value[0])
        sample_steps = [value[0] for value in entries]
        action_candidates: dict[int, list[dict[str, Any]]] = defaultdict(list)
        prompts: dict[int, str] = {}
        for step, _, sample in entries:
            prompts[step] = str(sample.get("current_prompt") or "")
            actions = assistant_actions(sample)
            start = step - len(actions)
            for offset, action in enumerate(actions):
                action_candidates[start + offset].append(action)

        action_by_step: dict[int, dict[str, Any]] = {}
        for action_step, candidates in action_candidates.items():
            names = {str(item.get("name") or item.get("action") or "").lower() for item in candidates}
            if len(names) > 1:
                diagnostics["history_action_conflicts"] += 1
                # A contradictory raw batch must not be used for a speculative
                # feature.  The current row's native history remains in the
                # ordinary user/state features.
                continue
            action_by_step[action_step] = candidates[-1]

        max_step = max(sample_steps)
        min_step = min(sample_steps)
        for step, index, sample in entries:
            is_terminal = step == max_step
            terminal[index] = is_terminal
            if is_terminal:
                diagnostics["terminal_rows"] += 1

            feature: dict[str, float | int] = {}
            feature[f"batch_rows={bucket(len(entries), (1, 2, 4, 6, 9, 13, 18))}"] = 1
            feature[f"batch_span={bucket(max_step - min_step + 1, (1, 2, 4, 6, 9, 13, 18))}"] = 1
            feature[f"batch_step={bucket(step, (1, 2, 3, 5, 8, 12, 16))}"] = 1
            feature[f"batch_max_step={bucket(max_step, (1, 2, 3, 5, 8, 12, 16))}"] = 1
            feature[f"batch_gap_count={bucket((max_step - min_step + 1) - len(entries), (0, 1, 2, 4, 7))}"] = 1
            feature[f"batch_is_max={int(is_terminal)}"] = 1

            observed = [(action_step, action) for action_step, action in action_by_step.items() if action_step < step]
            observed.sort(key=lambda value: value[0])
            feature[f"stitched_actions={bucket(len(observed), (0, 1, 2, 4, 6, 9, 13, 18))}"] = 1
            if observed:
                suffix_start = observed[-1][0]
                for action_step, _ in reversed(observed[:-1]):
                    if action_step != suffix_start - 1:
                        break
                    suffix_start = action_step
                feature[f"stitched_suffix={bucket(observed[-1][0] - suffix_start + 1, (0, 1, 2, 4, 6, 9, 13, 18))}"] = 1

            names: list[str] = []
            for position, (_, action) in enumerate(reversed(observed[-20:]), start=1):
                name = str(action.get("name") or action.get("action") or "none").lower()
                names.append(name)
                feature[f"st_pa{position}={name}"] = 1
                feature[f"st_pr{position}={result_bucket(action.get('result_summary'))}"] = 1
                values = action.get("args") if isinstance(action.get("args"), dict) else {}
                for key in ("path", "scope", "pattern", "target", "cmd", "query", "n_files"):
                    if key in values:
                        feature[f"st_arg{position}_{key}"] = 1
                feature[f"st_count={name}"] = feature.get(f"st_count={name}", 0) + 1
            chronological_names = list(reversed(names))
            for width in (2, 3, 4, 5):
                for offset in range(len(chronological_names) - width + 1):
                    feature[f"st_ng{width}=" + ">".join(chronological_names[offset:offset + width])] = 1

            # Every preceding sample's current_prompt is a user request that
            # was already available in this batch before the terminal target.
            # Do not include prompt steps after the current target.
            prompt_steps = [prompt_step for prompt_step in sorted(prompts) if prompt_step <= step]
            for position, prompt_step in enumerate(reversed(prompt_steps[-18:]), start=1):
                token_features(feature, f"st_u{position}", prompts[prompt_step])
            records[index] = feature
    return terminal, records, diagnostics


def user_hash(samples: list[dict[str, Any]]) -> np.ndarray:
    """Reuse live inference-safe current/recent history hash for terminal rows."""
    # Importing ``script`` is safe: its execution entry point is protected by
    # ``if __name__ == '__main__'``.  Reusing this function avoids a feature
    # discrepancy between this experiment and the live fallback.
    from script import build_user_sequence_hash
    return build_user_sequence_hash(samples)


def source_calibrated_fallback(ids: list[str], path: str, calibration_path: str) -> np.ndarray | None:
    probability_path = Path(path)
    if not probability_path.exists():
        return None
    probability = np.load(probability_path)
    if probability.shape != (len(ids), len(ALL_CLASSES)):
        # This artifact is intentionally only the fixed validation rows.
        return None
    calibration = json.loads(Path(calibration_path).read_text(encoding="utf-8"))
    source_index = np.asarray([1 if sample_id.startswith("sess_au_") else 0 for sample_id in ids])
    weights = np.asarray([
        [calibration["sim"][label] for label in ALL_CLASSES],
        [calibration["au"][label] for label in ALL_CLASSES],
    ], dtype=np.float32)
    return probability * weights[source_index]


def metric(y: np.ndarray, probability: np.ndarray) -> float:
    return float(f1_score(y, probability.argmax(axis=1), labels=np.arange(len(ALL_CLASSES)), average="macro", zero_division=0))


def fit_terminal_model(x_train: np.ndarray, y_train: np.ndarray, x_valid: np.ndarray, args: argparse.Namespace,
                       random_state: int) -> tuple[XGBClassifier, np.ndarray]:
    counts = np.bincount(y_train, minlength=len(ALL_CLASSES))
    weights = (len(y_train) / (len(ALL_CLASSES) * counts[y_train])) ** args.weight_power
    model = XGBClassifier(
        n_estimators=args.trees,
        max_depth=args.depth,
        learning_rate=args.learning_rate,
        subsample=0.88,
        colsample_bytree=0.82,
        min_child_weight=1.5,
        objective="multi:softprob",
        num_class=len(ALL_CLASSES),
        tree_method="hist",
        device=args.device,
        eval_metric="mlogloss",
        n_jobs=8,
        random_state=random_state,
    )
    model.fit(x_train, y_train, sample_weight=weights, verbose=False)
    return model, model.predict_proba(x_valid)


def main() -> None:
    args = parse_args()
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    samples = read_rows(args.data)
    labels = read_labels(args.labels)
    split = read_split(args.split)
    ids = [str(sample["id"]) for sample in samples]
    y = np.asarray([CLASS_TO_ID[labels[sample_id]] for sample_id in ids], dtype=np.int64)
    terminal, stitch_records, diagnostics = stitched_batch_records(samples)
    train_index = np.asarray([index for index, sample_id in enumerate(ids) if terminal[index] and split[sample_id] == "train"])
    valid_index = np.asarray([index for index, sample_id in enumerate(ids) if terminal[index] and split[sample_id] == "val"])
    if not len(train_index) or not len(valid_index):
        raise ValueError("Expected both train and validation terminal rows")

    base = np.load(args.base_oof).astype(np.float32, copy=False)
    if len(base) != len(samples):
        raise ValueError("Base OOF artifact is not aligned with train.jsonl")
    state_bundle = joblib.load(args.state_bundle)
    state_vectorizer = state_bundle["state_vectorizer"]
    from script import build_state_dict
    terminal_samples = [samples[index] for index in np.flatnonzero(terminal)]
    state_values = state_vectorizer.transform([build_state_dict(sample) for sample in terminal_samples])
    state = (state_values.toarray() if hasattr(state_values, "toarray") else np.asarray(state_values)).astype(np.float32)
    current_user = user_hash(terminal_samples)
    terminal_all = np.flatnonzero(terminal)
    pieces = [base[terminal_all], state, current_user]
    if not args.no_stitch:
        stitched = FeatureHasher(n_features=args.stitch_dim, input_type="dict", alternate_sign=False).transform(
            [stitch_records[index] for index in terminal_all]
        ).toarray().astype(np.float32)
        pieces.append(stitched)
    x = np.column_stack(pieces).astype(np.float32, copy=False)
    terminal_position = {int(index): position for position, index in enumerate(terminal_all)}
    tr = np.asarray([terminal_position[int(index)] for index in train_index])
    va = np.asarray([terminal_position[int(index)] for index in valid_index])
    model, probability = fit_terminal_model(x[tr], y[train_index], x[va], args, random_state=842)

    valid_ids = [ids[index] for index in valid_index]
    fallback = source_calibrated_fallback(
        [sample_id for sample_id in ids if split[sample_id] == "val"], args.fallback_oof, args.source_calibration
    )
    fallback_terminal = None
    if fallback is not None:
        # ``modeling_ac_tune2`` is stored in original validation row order.
        val_indices = [index for index, sample_id in enumerate(ids) if split[sample_id] == "val"]
        fallback_by_id = {ids[index]: fallback[position] for position, index in enumerate(val_indices)}
        fallback_terminal = np.asarray([fallback_by_id[sample_id] for sample_id in valid_ids])

    scores: dict[str, Any] = {
        "terminal_xgb": metric(y[valid_index], probability),
        "terminal_xgb_accuracy": float((probability.argmax(axis=1) == y[valid_index]).mean()),
    }
    blends: list[dict[str, float]] = []
    if fallback_terminal is not None:
        scores["current_source_calibrated_fallback"] = metric(y[valid_index], fallback_terminal)
        scores["current_source_calibrated_accuracy"] = float((fallback_terminal.argmax(axis=1) == y[valid_index]).mean())
        for alpha in (0.0, 0.15, 0.25, 0.35, 0.5, 0.65, 0.75, 0.85, 1.0):
            value = metric(y[valid_index], (1.0 - alpha) * fallback_terminal + alpha * probability)
            blends.append({"terminal_xgb_weight": alpha, "terminal_macro_f1": value})
        scores["blends_fixed_validation"] = blends

    if args.run_nonrespond:
        nonrespond = CLASS_TO_ID["respond_only"]
        nonresponse_train = tr[y[train_index] != nonrespond]
        # Train a 13-way model and use it only where the established fallback
        # does not already emit its perfectly calibrated respond_only class.
        target_train = y[terminal_all[nonresponse_train]]
        compact = {old: new for new, old in enumerate(range(len(ALL_CLASSES) - 1))}
        compact_y = np.asarray([compact[int(value)] for value in target_train], dtype=np.int64)
        counts = np.bincount(compact_y, minlength=len(compact))
        weight = (len(compact_y) / (len(compact) * counts[compact_y])) ** args.weight_power
        non_model = XGBClassifier(
            n_estimators=args.trees, max_depth=args.depth, learning_rate=args.learning_rate,
            subsample=0.88, colsample_bytree=0.82, min_child_weight=1.5,
            objective="multi:softprob", num_class=len(compact), tree_method="hist", device=args.device,
            eval_metric="mlogloss", n_jobs=8, random_state=943,
        )
        non_model.fit(x[nonresponse_train], compact_y, sample_weight=weight, verbose=False)
        non_probability = non_model.predict_proba(x[va])
        expanded = np.zeros_like(probability)
        expanded[:, :len(compact)] = non_probability
        if fallback_terminal is not None:
            gated = fallback_terminal.copy()
            gate = fallback_terminal.argmax(axis=1) != nonrespond
            # Preserve fallback confidence for the already exact responder
            # class and replace only its non-responder branch.
            gated[gate] = expanded[gate]
            scores["nonrespond_xgb_gated_by_current_fallback"] = metric(y[valid_index], gated)
        joblib.dump(non_model, out / "nonrespond_train_split_model.pkl", compress=3)

    report = classification_report(
        y[valid_index], probability.argmax(axis=1), labels=np.arange(len(ALL_CLASSES)), target_names=ALL_CLASSES,
        digits=4, zero_division=0,
    )
    np.savez_compressed(
        out / "fixed_val_terminal_predictions.npz",
        ids=np.asarray(valid_ids), y=y[valid_index], terminal_xgb=probability,
        current_fallback=fallback_terminal if fallback_terminal is not None else np.empty((0, len(ALL_CLASSES))),
    )
    joblib.dump(model, out / "terminal_train_split_model.pkl", compress=3)
    (out / "classification_report.txt").write_text(report, encoding="utf-8")
    summary = {
        "fixed_validation": "common_split.csv; all examples are session-max rows; no session overlaps training",
        "rows": {"all": len(samples), "terminal": int(terminal.sum()), "train_terminal": len(train_index), "val_terminal": len(valid_index)},
        "feature_shape": list(x.shape),
        "uses_stitched_batch_features": not args.no_stitch,
        "stitch_dim": 0 if args.no_stitch else args.stitch_dim,
        "diagnostics": diagnostics,
        "scores": scores,
        "args": vars(args),
        "warning": "This is an experiment-only train-split artifact. It does not modify model/, script.py, or any ZIP.",
    }
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
