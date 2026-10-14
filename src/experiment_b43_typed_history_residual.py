#!/usr/bin/env python3
"""Finite typed tool-history residual audit for the exact B43 fallback.

This writes only to experiments/b43_typed_history_residual.  It never changes
the live model, inference script, requirements, packaging script, or any ZIP.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import time
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold


ROOT = Path(__file__).resolve().parents[1]
CLASSES = (
    "read_file", "grep_search", "list_directory", "glob_pattern",
    "edit_file", "write_file", "apply_patch", "run_bash", "run_tests",
    "lint_or_typecheck", "ask_user", "plan_task", "web_search", "respond_only",
)
N_CLASSES = len(CLASSES)
TO_ID = {value: number for number, value in enumerate(CLASSES)}
EPS = np.float32(1e-12)
POLICIES = ("base", "geo_995", "geo_990", "geo_980", "lowconf_disagree")
POLICY_RANK = {value: number for number, value in enumerate(POLICIES)}
ARG_KEYS = (
    "path", "target", "scope", "pattern", "cmd", "command", "file", "files",
    "query", "url", "content", "target_symbol", "goal", "question", "n_files",
)
EXTENSIONS = (
    "py", "js", "ts", "tsx", "jsx", "json", "yaml", "yml", "toml", "md",
    "txt", "sh", "bash", "sql", "html", "css", "java", "go", "rb", "rs",
    "other", "none",
)
COMMANDS = (
    "pytest", "js_test", "lint", "git", "search", "filesystem", "build",
    "container", "python", "package", "network", "compiled", "other", "none",
)
UNITS = ("match", "file", "test", "error", "warning", "changed")
FULL_B43_CONTROL = {
    "all_fallback_macro_f1": 0.7603385811033483,
    "terminal_fallback_macro_f1": 0.7312748947080282,
    "terminal_with_char20_macro_f1": 0.7335225252513891,
    "hybrid_future_history_plus_char20_macro_f1": 0.98277826622222,
}
FIXED_B43_CONTROL = {
    "all_fallback_macro_f1": 0.7585157741730599,
    "terminal_fallback_macro_f1": 0.6969994677821684,
    "terminal_with_char20_macro_f1": 0.7000338475808981,
    "hybrid_future_history_plus_char20_macro_f1": 0.9853915549361344,
}


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="experiments/b43_typed_history_residual")
    parser.add_argument("--trees", type=int, default=420)
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--bootstrap-replicates", type=int, default=5000)
    parser.add_argument("--preflight", action="store_true")
    return parser.parse_args()


def session_of(sample_id: str) -> str:
    return sample_id.rsplit("-step_", 1)[0]


def source_of(sample_id: str) -> str:
    return "au" if sample_id.startswith("sess_au_") else "sim"


def terminal_mask(ids: np.ndarray) -> np.ndarray:
    greatest: dict[str, int] = {}
    parsed: list[tuple[str, int]] = []
    for sample_id in ids.astype(str):
        sid, raw = sample_id.rsplit("-step_", 1)
        step = int(raw)
        greatest[sid] = max(greatest.get(sid, -1), step)
        parsed.append((sid, step))
    return np.asarray([step == greatest[sid] for sid, step in parsed], dtype=bool)


def macro(y: np.ndarray, pred: np.ndarray, mask: np.ndarray | None = None) -> float:
    if mask is None:
        mask = np.ones(len(y), dtype=bool)
    return float(f1_score(y[mask], pred[mask], labels=np.arange(N_CLASSES), average="macro", zero_division=0))


def geometry(left: np.ndarray, right: np.ndarray, left_weight: float) -> np.ndarray:
    value = left_weight * np.log(np.maximum(left, EPS)) + (1.0 - left_weight) * np.log(np.maximum(right, EPS))
    value -= value.max(axis=1, keepdims=True)
    value = np.exp(value)
    return (value / value.sum(axis=1, keepdims=True)).astype(np.float32)


def b43_probability(raw: np.ndarray, old: np.ndarray, source: np.ndarray) -> np.ndarray:
    weight = np.where(source == "au", 0.85, 0.20).astype(np.float64)[:, None]
    value = weight * np.log(np.maximum(raw, EPS)) + (1.0 - weight) * np.log(np.maximum(old, EPS))
    value -= value.max(axis=1, keepdims=True)
    value = np.exp(value)
    return (value / value.sum(axis=1, keepdims=True)).astype(np.float32)


def final_labels(probability: np.ndarray, terminal: np.ndarray, char20: np.ndarray) -> np.ndarray:
    value = probability.argmax(axis=1).astype(np.int16)
    use = terminal & (char20 >= 0)
    value[use] = char20[use]
    return value


def metrics(y: np.ndarray, probability: np.ndarray, terminal: np.ndarray, char20: np.ndarray) -> dict[str, float]:
    fallback = probability.argmax(axis=1).astype(np.int16)
    final = final_labels(probability, terminal, char20)
    hybrid = final.copy()
    hybrid[~terminal] = y[~terminal]
    return {
        "all_fallback_macro_f1": macro(y, fallback),
        "terminal_fallback_macro_f1": macro(y, fallback, terminal),
        "terminal_with_char20_macro_f1": macro(y, final, terminal),
        "hybrid_future_history_plus_char20_macro_f1": macro(y, hybrid),
    }


def delta(current: dict[str, float], baseline: dict[str, float]) -> dict[str, float]:
    return {key: float(current[key] - baseline[key]) for key in current}


def assert_control(value: dict[str, float], expected: dict[str, float], scope: str) -> None:
    mismatch = {
        key: {"expected": expected[key], "actual": value.get(key)}
        for key in expected
        if key not in value or not np.isclose(value[key], expected[key], rtol=0.0, atol=1e-9)
    }
    if mismatch:
        raise ValueError(f"exact B43 {scope} control mismatch: {mismatch}")


def assert_session_folds(groups: np.ndarray, fold: np.ndarray, scope: str) -> None:
    if fold.shape != groups.shape or np.any(fold < 0) or np.unique(fold).size != 5:
        raise ValueError(f"{scope} fold assignment is invalid")
    for value in np.unique(fold):
        held = set(groups[fold == value])
        train = set(groups[fold != value])
        if held.intersection(train):
            raise ValueError(f"{scope} fold assignment splits a session")


def text(value: object) -> str:
    return " ".join(str(value or "").casefold().replace("\x00", " ").split())


def hot(index: int, width: int) -> np.ndarray:
    value = np.zeros(width, dtype=np.float32)
    if 0 <= index < width:
        value[index] = 1.0
    return value


def number_bucket(value: int | None) -> int:
    if value is None:
        return 0
    if value == 0:
        return 1
    if value == 1:
        return 2
    if value <= 9:
        return 3
    if value <= 99:
        return 4
    return 5


def length_bucket(value: int) -> int:
    if value <= 0:
        return 0
    if value <= 8:
        return 1
    if value <= 32:
        return 2
    if value <= 120:
        return 3
    return 4


def status(value: str) -> int:
    if not value:
        return 0
    if any(key in value for key in ("fail", "error", "exception", "traceback", "denied", "timeout")):
        return 2
    if any(key in value for key in ("ok", "pass", "success", "found", "listed", "matched", "patched", "applied", "modified", "exit=0")):
        return 1
    return 3


def result_flags(value: str) -> np.ndarray:
    groups = (
        ("fail", "error", "exception", "traceback", "denied", "timeout"),
        ("not found", "no match", "0 match", "0 file", "missing"),
        ("ok", "success", "exit=0", "found", "listed", "matched"),
        ("test", "spec", "assert"),
        ("lint", "typecheck", "type check", "format"),
        ("patch", "applied", "modified", "wrote", "edit"),
        ("pass", "green", "succeeded"),
        ("warning", "warn"),
    )
    return np.asarray([float(any(part in value for part in group)) for group in groups], dtype=np.float32)


def count_bucket(value: str, unit: str) -> int:
    aliases = {
        "match": r"(?:match(?:es|ed)?|occurrence(?:s)?)",
        "file": r"(?:file(?:s)?|item(?:s)?|entr(?:y|ies)|dir(?:s|ector(?:y|ies))?)",
        "test": r"(?:test(?:s)?|spec(?:s)?|assertion(?:s)?)",
        "error": r"(?:error(?:s)?|failure(?:s)?)",
        "warning": r"(?:warning(?:s)?|warn(?:ing)?s?)",
        "changed": r"(?:change(?:s|d)?|edit(?:s)?|patch(?:es|ed)?)",
    }[unit]
    found = re.findall(r"(?<![\w.-])(\d+)\s+(?:" + aliases + r")\b", value)
    return number_bucket(max((int(item) for item in found), default=None))


def exit_bucket(value: str) -> int:
    found = re.search(r"\bexit\s*=?\s*(-?\d+)\b", value)
    if found is None:
        return 0
    return 1 if int(found.group(1)) == 0 else 2


def extension(value: object) -> int:
    found = re.search(r"\.([a-z0-9]{1,8})(?:$|[/?#])", text(value))
    if found is None:
        return EXTENSIONS.index("none")
    suffix = found.group(1)
    return EXTENSIONS.index(suffix if suffix in EXTENSIONS else "other")


def path_kind(value: object) -> int:
    value = text(value)
    if not value:
        return 0
    if any(mark in value for mark in ("*", "?", "[", "]", "{", "}")):
        return 3
    if re.match(r"^(?:[a-z]:[\\\\/]|/|~[\\\\/])", value):
        return 2
    if value.endswith(("/", "\\")) or "." not in value.rsplit("/", 1)[-1]:
        return 4
    return 1


def command(value: object) -> int:
    value = text(value)
    if not value:
        return COMMANDS.index("none")
    rules = (
        ("pytest", ("pytest", "unittest", "nose")),
        ("js_test", ("jest", "vitest", "mocha", "npm test", "yarn test", "pnpm test")),
        ("lint", ("eslint", "ruff", "mypy", "tsc", "prettier", "flake8")),
        ("git", ("git ",)),
        ("search", ("grep", "rg ", "ripgrep", "findstr")),
        ("filesystem", ("ls ", "dir ", "tree ", "find ")),
        ("build", ("build", "compile", "make ")),
        ("container", ("docker", "podman", "compose")),
        ("python", ("python", "pip ")),
        ("package", ("npm ", "yarn ", "pnpm ", "bundle")),
        ("network", ("curl", "wget", "http")),
        ("compiled", ("cargo", "go test", "gradle", "mvn ")),
    )
    for family, parts in rules:
        if any(part in value for part in parts):
            return COMMANDS.index(family)
    return COMMANDS.index("other")


def pattern(value: object) -> int:
    value = text(value)
    if not value:
        return 0
    if any(mark in value for mark in ("*", "?", "[", "]", "{", "}")):
        return 1
    if any(mark in value for mark in ("^", "$", "(", ")", "|", "\\\\", "+")):
        return 2
    return 4 if len(value) <= 2 else 3


def action_history(sample: dict) -> list[dict]:
    history = sample.get("history") if isinstance(sample.get("history"), list) else []
    return [item for item in history if isinstance(item, dict) and item.get("role") == "assistant_action"][-6:]


def event(item: dict | None) -> np.ndarray:
    if item is None:
        return np.concatenate((
            hot(N_CLASSES, N_CLASSES + 1), hot(0, 4), hot(0, 3), np.zeros(8, dtype=np.float32),
            np.zeros(len(UNITS) * 6, dtype=np.float32), np.zeros(len(ARG_KEYS), dtype=np.float32),
            hot(EXTENSIONS.index("none"), len(EXTENSIONS)), hot(0, 5), hot(COMMANDS.index("none"), len(COMMANDS)),
            hot(0, 5), hot(0, 5), hot(0, 6),
        ))
    name = text(item.get("name") or item.get("action"))
    summary = text(item.get("result_summary"))
    args = item.get("args") if isinstance(item.get("args"), dict) else {}
    args = {text(key): value for key, value in args.items()}
    path = args.get("path", args.get("file", ""))
    cmd = args.get("cmd", args.get("command", ""))
    pat = args.get("pattern", args.get("query", ""))
    max_len = max((len(text(value)) for value in args.values()), default=0)
    try:
        n_files = max(0, int(float(args.get("n_files")))) if "n_files" in args else None
    except (TypeError, ValueError):
        n_files = None
    return np.concatenate((
        hot(TO_ID.get(name, N_CLASSES), N_CLASSES + 1), hot(status(summary), 4), hot(exit_bucket(summary), 3),
        result_flags(summary), np.concatenate([hot(count_bucket(summary, unit), 6) for unit in UNITS]),
        np.asarray([float(key in args) for key in ARG_KEYS], dtype=np.float32),
        hot(extension(path), len(EXTENSIONS)), hot(path_kind(path), 5), hot(command(cmd), len(COMMANDS)),
        hot(pattern(pat), 5), hot(length_bucket(max_len), 5), hot(number_bucket(n_files), 6),
    )).astype(np.float32)


def typed_features(rows: list[dict]) -> tuple[np.ndarray, dict]:
    values: list[np.ndarray] = []
    lengths: list[int] = []
    coverage = Counter()
    for sample in rows:
        history = action_history(sample)
        lengths.append(len(history))
        blocks = [event(None)] * (6 - len(history)) + [event(item) for item in history]
        action_count = np.zeros(N_CLASSES, dtype=np.float32)
        status_count = np.zeros(4, dtype=np.float32)
        action_status_count = np.zeros((N_CLASSES, 4), dtype=np.float32)
        action_command_count = np.zeros((N_CLASSES, len(COMMANDS)), dtype=np.float32)
        for item in history:
            name = text(item.get("name") or item.get("action"))
            action_id = TO_ID.get(name)
            summary = text(item.get("result_summary"))
            args = item.get("args") if isinstance(item.get("args"), dict) else {}
            action_status = status(summary)
            action_command = command(args.get("cmd", args.get("command", "")))
            if action_id is not None:
                action_count[action_id] += 1.0
                action_status_count[action_id, action_status] += 1.0
                action_command_count[action_id, action_command] += 1.0
            status_count[action_status] += 1.0
            coverage["args"] += int(isinstance(item.get("args"), dict) and bool(item.get("args")))
            coverage["result"] += int(bool(summary))
        aggregate = np.concatenate((
            action_count / max(1, len(history)), status_count / max(1, len(history)),
            action_status_count.ravel() / max(1, len(history)),
            action_command_count.ravel() / max(1, len(history)),
            np.asarray([len(history) / 6.0, float(len(history) == 0), float(len(history) >= 3), float(len(history) == 6)], dtype=np.float32),
        ))
        values.append(np.concatenate((*blocks, aggregate)).astype(np.float32))
    matrix = np.stack(values)
    return matrix, {
        "rows": int(len(rows)), "width": int(matrix.shape[1]),
        "history_quantiles": [float(value) for value in np.quantile(lengths, (0, .5, .9, 1))],
        "events_with_args": int(coverage["args"]), "events_with_result_summary": int(coverage["result"]),
        "explicit_action_typed_crosses": ["action_x_status", "action_x_command_family"],
        "raw_prompt_or_value_tokens": False,
    }


def features(typed: np.ndarray, base: np.ndarray) -> np.ndarray:
    ordered = np.sort(base, axis=1)
    entropy = -(base * np.log(np.maximum(base, EPS))).sum(axis=1, keepdims=True)
    margin = (ordered[:, -1] - ordered[:, -2])[:, None]
    return np.column_stack((typed, base * 2, np.log(np.maximum(base, EPS)) * .25, entropy, margin)).astype(np.float32)


def fit_predict(x_fit: np.ndarray, y_fit: np.ndarray, x_pred: np.ndarray, trees: int, jobs: int, seed: int) -> np.ndarray:
    model = ExtraTreesClassifier(
        n_estimators=trees, criterion="log_loss", max_depth=12, min_samples_split=16, min_samples_leaf=8,
        max_features=.62, class_weight="balanced", n_jobs=jobs, random_state=seed,
    )
    model.fit(x_fit, y_fit)
    value = model.predict_proba(x_pred).astype(np.float32)
    result = np.full((len(x_pred), N_CLASSES), EPS, dtype=np.float32)
    result[:, model.classes_.astype(int)] = value
    return result / result.sum(axis=1, keepdims=True)


def policy(name: str, base: np.ndarray, residual: np.ndarray, terminal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    result = base.copy()
    gate = np.zeros(len(base), dtype=bool)
    pos = np.flatnonzero(terminal)
    if name == "base":
        return result, gate
    if name.startswith("geo_"):
        result[pos] = geometry(base[pos], residual, float(name.split("_")[1]) / 1000)
        gate[pos] = True
        return result, gate
    top = base[pos].max(1)
    rtop = residual.max(1)
    rsorted = np.sort(residual, axis=1)
    use = (top <= .57) & (rtop >= .40) & (base[pos].argmax(1) != residual.argmax(1)) & ((rsorted[:, -1] - rsorted[:, -2]) >= .12)
    gate[pos[use]] = True
    result[gate] = geometry(base[gate], residual[use], .88)
    return result, gate


def bootstrap(y: np.ndarray, base: np.ndarray, candidate: np.ndarray, groups: np.ndarray, replicates: int, seed: int) -> dict:
    unique, inverse = np.unique(groups, return_inverse=True)
    if len(unique) != len(groups):
        raise ValueError("one terminal row per session is required")
    base_cell = y.astype(int) * N_CLASSES + base.astype(int)
    cand_cell = y.astype(int) * N_CLASSES + candidate.astype(int)
    rng = np.random.default_rng(seed)
    delta_values = np.empty(replicates)
    def score(cell: np.ndarray, weights: np.ndarray) -> float:
        confusion = np.bincount(cell, weights=weights, minlength=N_CLASSES * N_CLASSES).reshape(N_CLASSES, N_CLASSES)
        denom = confusion.sum(0) + confusion.sum(1)
        return float(np.mean(np.divide(2 * np.diag(confusion), denom, out=np.zeros(N_CLASSES), where=denom > 0)))
    for number in range(replicates):
        picked = rng.integers(0, len(unique), len(unique))
        weights = np.bincount(picked, minlength=len(unique))[inverse]
        delta_values[number] = score(cand_cell, weights) - score(base_cell, weights)
    return {
        "replicates": int(replicates), "sessions": int(len(unique)),
        "exact_delta": float(macro(y, candidate) - macro(y, base)),
        "p_delta_gt_zero": float(np.mean(delta_values > 0)), "p05_delta": float(np.quantile(delta_values, .05)),
        "p95_delta": float(np.quantile(delta_values, .95)),
    }


def main() -> None:
    args = arguments()
    if args.trees < 1 or args.bootstrap_replicates < 100:
        raise ValueError("invalid trees or bootstrap count")
    out = (ROOT / args.out_dir).resolve()
    experiment_root = (ROOT / "experiments").resolve()
    try:
        out.relative_to(experiment_root)
    except ValueError as error:
        raise ValueError("--out-dir must stay under experiments/") from error
    out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    rows = [json.loads(line) for line in (ROOT / "data" / "train.jsonl").read_text(encoding="utf8").splitlines() if line.strip()]
    labels = {row["id"]: row["action"] for row in csv.DictReader((ROOT / "data" / "train_labels.csv").open(encoding="utf8"))}
    split = {row["id"]: row["split"] for row in csv.DictReader((ROOT / "splits" / "common_split.csv").open(encoding="utf8"))}
    ids = np.asarray([str(row["id"]) for row in rows])
    y = np.asarray([TO_ID[labels[item]] for item in ids], dtype=np.int16)
    groups = np.asarray([session_of(item) for item in ids])
    terminal = terminal_mask(ids)
    sources = np.asarray([source_of(item) for item in ids])
    typed, typed_report = typed_features(rows)
    raw = np.load(ROOT / "experiments" / "xgb_robust_ensemble" / "full_oof_raw_seed42.npy", allow_pickle=False).astype(np.float32)
    old = np.load(ROOT / "experiments" / "modeling_ai_sequence_fusion_noe5" / "oof_probabilities.npy", allow_pickle=False).astype(np.float32)
    if raw.shape != old.shape or raw.shape != (len(rows), N_CLASSES):
        raise ValueError("full B43 components do not align")
    base = b43_probability(raw, old, sources)
    stored_outer_fold = np.load(
        ROOT / "experiments" / "modeling_ai_sequence_nested_distill_e16_i4" / "outer_fold_assignment.npy",
        allow_pickle=False,
    ).astype(np.int8)
    recalculated_outer_fold = np.empty(len(y), dtype=np.int8)
    for number, (_, held) in enumerate(GroupKFold(n_splits=5).split(np.zeros(len(y)), y, groups)):
        recalculated_outer_fold[held] = number
    if not np.array_equal(stored_outer_fold, recalculated_outer_fold):
        raise ValueError("stored B43 outer fold does not match the full-row session GroupKFold")
    assert_session_folds(groups, stored_outer_fold, "full B43 outer")
    char = np.load(ROOT / "experiments" / "terminal_rule_combinations" / "oof_candidates.npz", allow_pickle=False)
    if not (
        np.array_equal(char["y"].astype(np.int16), y)
        and np.array_equal(char["terminal"].astype(bool), terminal)
        and np.array_equal(char["fold"].astype(np.int8)[terminal], stored_outer_fold[terminal])
    ):
        raise ValueError("char20 full alignment failure")
    char_full = char["char20"].astype(np.int16)
    baseline_full = metrics(y, base, terminal, char_full)
    common = np.flatnonzero(np.asarray([split[item] == "train" for item in ids]))
    fixed = np.flatnonzero(np.asarray([split[item] == "val" for item in ids]))
    raw_dir = ROOT / "experiments" / "b42_9_common_raw_seed42"
    old_dir = ROOT / "experiments" / "b42_9_common_old_noe5_seed42"
    fixed_old_dir = ROOT / "experiments" / "modeling_ai_sequence_fusion_fixed_noe5"
    raw_common = np.load(raw_dir / "common_train_raw_xgb_oof.npy", allow_pickle=False).astype(np.float32)
    old_common = np.load(old_dir / "common_train_oof_probabilities.npy", allow_pickle=False).astype(np.float32)
    raw_fixed = np.load(ROOT / "experiments" / "xgb_robust_ensemble" / "strict_fixed_raw_seed42.npy", allow_pickle=False).astype(np.float32)
    # This is the independently refit no-E5 component used by the shipped B43
    # selector.  The similarly named common-only artifact is deliberately not
    # used here: it produces a different fixed-validation baseline.
    old_fixed = np.load(fixed_old_dir / "valid_probabilities.npy", allow_pickle=False).astype(np.float32)
    if not (
        np.array_equal(np.load(raw_dir / "common_train_ids.npy", allow_pickle=False).astype(str), ids[common])
        and np.array_equal(np.load(old_dir / "common_train_ids.npy", allow_pickle=False).astype(str), ids[common])
        and np.array_equal(np.load(fixed_old_dir / "valid_indices.npy", allow_pickle=False).astype(np.int64), fixed)
        and np.array_equal(np.load(fixed_old_dir / "terminal_mask.npy", allow_pickle=False).astype(bool), terminal[fixed])
    ):
        raise ValueError("common/fixed B43 identifier mismatch")
    common_fold = np.load(raw_dir / "common_train_selector_fold.npy", allow_pickle=False).astype(np.int8)
    old_common_fold = np.load(old_dir / "common_train_fold.npy", allow_pickle=False).astype(np.int8)
    if not np.array_equal(common_fold, old_common_fold):
        raise ValueError("common raw/old components do not share an exact fold assignment")
    assert_session_folds(groups[common], common_fold, "common B43")
    base_common = b43_probability(raw_common, old_common, sources[common])
    base_fixed = b43_probability(raw_fixed, old_fixed, sources[fixed])
    fixed_terminal = terminal[fixed]
    char_fixed = np.full(len(fixed), -1, dtype=np.int16)
    if not (
        np.array_equal(char["valid_ids"].astype(str), ids[fixed][fixed_terminal])
        and np.array_equal(char["fixed_y"].astype(np.int16), y[fixed][fixed_terminal])
    ):
        raise ValueError("fixed char20 identifier mismatch")
    char_fixed[fixed_terminal] = char["fixed_char20"].astype(np.int16)
    baseline_fixed = metrics(y[fixed], base_fixed, fixed_terminal, char_fixed)
    assert_control(baseline_full, FULL_B43_CONTROL, "full")
    assert_control(baseline_fixed, FIXED_B43_CONTROL, "matching fixed")
    preflight = {
        "typed_features": typed_report, "rows": int(len(rows)), "terminal_rows": int(terminal.sum()),
        "common_rows": int(len(common)), "fixed_rows": int(len(fixed)), "fixed_terminal_rows": int(fixed_terminal.sum()),
        "baseline_full": baseline_full, "baseline_fixed": baseline_fixed,
        "exact_b43_controls": True,
        "full_outer_fold_matches_stored_b43": True,
        "common_component_fold_matches": True,
        "no_op_exact": bool(np.array_equal(base, policy("base", base, np.zeros((terminal.sum(), N_CLASSES), dtype=np.float32), terminal)[0])),
        "submission_files_modified": False,
    }
    (out / "preflight.json").write_text(json.dumps(preflight, ensure_ascii=False, indent=2) + "\n", encoding="utf8")
    if args.preflight:
        print(json.dumps(preflight, ensure_ascii=False, indent=2), flush=True)
        return
    term_ix = np.flatnonzero(terminal)
    term_position = np.full(len(y), -1, dtype=np.int32)
    term_position[term_ix] = np.arange(len(term_ix), dtype=np.int32)
    x = features(typed[term_ix], base[term_ix])
    yt = y[term_ix]
    residual = np.zeros((len(term_ix), N_CLASSES), dtype=np.float32)
    nested = base.copy()
    outer_choices = []
    selected_count = Counter()
    for held_fold in sorted(np.unique(stored_outer_fold).tolist()):
        outer_train = np.flatnonzero(stored_outer_fold != held_fold)
        outer_held = np.flatnonzero(stored_outer_fold == held_fold)
        outer_train_term = outer_train[terminal[outer_train]]
        outer_held_term = outer_held[terminal[outer_held]]
        inner_probability = {name: base[outer_train_term].copy() for name in POLICIES}
        inner_changed = {name: 0 for name in POLICIES}
        inner_records = []
        inner_splits = list(GroupKFold(n_splits=4).split(
            np.zeros(len(outer_train)), y[outer_train], groups[outer_train],
        ))
        for inner_number, (inner_train_pos, inner_valid_pos) in enumerate(inner_splits, 1):
            inner_train = outer_train[inner_train_pos]
            inner_valid = outer_train[inner_valid_pos]
            fit_global = inner_train[terminal[inner_train]]
            query_global = inner_valid[terminal[inner_valid]]
            query_residual = fit_predict(
                x[term_position[fit_global]], y[fit_global], x[term_position[query_global]],
                args.trees, args.jobs, 3100 + held_fold * 10 + inner_number,
            )
            query_local = np.searchsorted(outer_train_term, query_global)
            for name in POLICIES:
                candidate, _ = policy(
                    name, base[query_global], query_residual, np.ones(len(query_global), dtype=bool),
                )
                inner_probability[name][query_local] = candidate
                inner_changed[name] += int(np.sum(candidate.argmax(1) != base[query_global].argmax(1)))
            inner_records.append({
                "inner_fold": inner_number,
                "held_sessions": int(len(np.unique(groups[query_global]))),
                "typed_terminal_macro_f1": macro(y[query_global], query_residual.argmax(1)),
            })
        scores = {
            name: macro(y[outer_train_term], value.argmax(1))
            for name, value in inner_probability.items()
        }
        selected = max(
            POLICIES,
            key=lambda name: (scores[name], -inner_changed[name], -POLICY_RANK[name]),
        )
        held_residual = fit_predict(
            x[term_position[outer_train_term]], y[outer_train_term], x[term_position[outer_held_term]],
            args.trees, args.jobs, 3200 + held_fold,
        )
        residual[term_position[outer_held_term]] = held_residual
        held_probability, _ = policy(
            selected, base[outer_held_term], held_residual, np.ones(len(outer_held_term), dtype=bool),
        )
        nested[outer_held_term] = held_probability
        selected_count[selected] += 1
        outer_record = {
            "held_outer_fold": int(held_fold + 1),
            "held_sessions": int(len(np.unique(groups[outer_held]))),
            "held_terminal_rows": int(len(outer_held_term)),
            "inner_oof": inner_records,
            "selected_pre_char20": selected,
            "pre_char20_scores": scores,
            "candidate_changed_terminal_rows": inner_changed,
            "held_changed_terminal_rows": int(np.sum(held_probability.argmax(1) != base[outer_held_term].argmax(1))),
        }
        outer_choices.append(outer_record)
        print(json.dumps(outer_record, ensure_ascii=False), flush=True)
    if np.any(residual.sum(axis=1) == 0.0):
        raise ValueError("nested outer residual has unfilled terminal rows")
    nested_full = metrics(y, nested, terminal, char_full)

    common_term_global = common[terminal[common]]
    common_term_position = np.full(len(y), -1, dtype=np.int32)
    common_term_position[common_term_global] = np.arange(len(common_term_global), dtype=np.int32)
    x_common = features(typed[common_term_global], base_common[terminal[common]])
    y_common = y[common_term_global]
    common_residual = np.zeros((len(y_common), N_CLASSES), dtype=np.float32)
    common_records = []
    for held_fold in sorted(np.unique(common_fold).tolist()):
        fit_global = common[(common_fold != held_fold) & terminal[common]]
        held_global = common[(common_fold == held_fold) & terminal[common]]
        held_local = common_term_position[held_global]
        common_residual[held_local] = fit_predict(
            x_common[common_term_position[fit_global]], y[fit_global], x_common[held_local],
            args.trees, args.jobs, 4100 + held_fold,
        )
        common_records.append({
            "fold": int(held_fold + 1),
            "held_sessions": int(len(np.unique(groups[held_global]))),
            "typed_terminal_macro_f1": macro(y[held_global], common_residual[held_local].argmax(1)),
        })
    common_options = {}
    all_terminal = np.ones(len(y_common), dtype=bool)
    common_changed = {}
    for name in POLICIES:
        common_options[name], _ = policy(name, base_common[terminal[common]], common_residual, all_terminal)
        common_changed[name] = int(np.sum(common_options[name].argmax(1) != base_common[terminal[common]].argmax(1)))
    common_scores = {name: macro(y_common, value.argmax(1)) for name, value in common_options.items()}
    selected_fixed = max(
        POLICIES,
        key=lambda name: (common_scores[name], -common_changed[name], -POLICY_RANK[name]),
    )
    fixed_term_global = fixed[fixed_terminal]
    x_fixed = features(typed[fixed_term_global], base_fixed[fixed_terminal])
    fixed_residual = fit_predict(x_common, y_common, x_fixed, args.trees, args.jobs, 4999)
    fixed_probability, fixed_gate = policy(selected_fixed, base_fixed, fixed_residual, fixed_terminal)
    fixed_result = metrics(y[fixed], fixed_probability, fixed_terminal, char_fixed)

    base_full_labels = final_labels(base, terminal, char_full)[terminal]
    candidate_full_labels = final_labels(nested, terminal, char_full)[terminal]
    base_fixed_labels = final_labels(base_fixed, fixed_terminal, char_fixed)[fixed_terminal]
    candidate_fixed_labels = final_labels(fixed_probability, fixed_terminal, char_fixed)[fixed_terminal]
    boot_full = bootstrap(y[terminal], base_full_labels, candidate_full_labels, groups[terminal], args.bootstrap_replicates, 20260713)
    boot_fixed = bootstrap(y[fixed][fixed_terminal], base_fixed_labels, candidate_fixed_labels,
                           groups[fixed][fixed_terminal], args.bootstrap_replicates, 20260714)
    full_delta = delta(nested_full, baseline_full)
    fixed_delta = delta(fixed_result, baseline_fixed)
    held_fold_deltas = []
    for held_fold in sorted(np.unique(stored_outer_fold).tolist()):
        held = stored_outer_fold == held_fold
        held_baseline = metrics(y[held], base[held], terminal[held], char_full[held])
        held_candidate = metrics(y[held], nested[held], terminal[held], char_full[held])
        held_fold_deltas.append({
            "held_outer_fold": int(held_fold + 1),
            "delta": delta(held_candidate, held_baseline),
        })
    needed = tuple(FULL_B43_CONTROL)
    positive_outer_folds = {
        key: int(sum(record["delta"][key] > 0.0 for record in held_fold_deltas))
        for key in needed
    }
    outer_stability = (
        positive_outer_folds["all_fallback_macro_f1"] >= 4
        and positive_outer_folds["terminal_with_char20_macro_f1"] >= 4
    )
    passed = (
        all(full_delta[key] > 0 and fixed_delta[key] > 0 for key in needed)
        and boot_full["p_delta_gt_zero"] >= .95 and boot_fixed["p_delta_gt_zero"] >= .95
        and boot_full["p05_delta"] > 0 and boot_fixed["p05_delta"] > 0
        and outer_stability
    )
    report = {
        "scope": "Experiment only. No live model, script, requirements, build_zip or archive changed.",
        "contract": {
            "baseline": "B43 raw seed42 plus old noE5 geometry SIM=.20/AU=.85",
            "typed_input": "own current-row last six prior assistant events only; finite result and argument categories",
            "forbidden": ["raw prompt/value tokens", "id/session id", "target label", "sibling/later rows", "future recovery", "char20 input"],
            "order": "B43 geometry then residual then immutable char20 then external future-history override",
            "full_validation": "stored exact B43 five outer session folds; each outer policy is selected only from four-inner-fold residual OOF on its outer-train sessions",
            "fixed_validation": "common-train component-aligned residual OOF selects a pre-char20 policy; all common terminal rows then refit once for held fixed sessions",
            "all_row_typed_only_screen": "not run; a prior typed-only screen rejected this branch, so this audit is restricted to probability-conditioned terminal residuals",
            "frozen_base_diagnostic_caveat": "B43 component probabilities are frozen global OOF diagnostics and are not re-cross-fitted inside each residual outer/inner fold; this audit is a screen only and cannot authorize an export.",
        },
        "preflight": preflight,
        "model": {"kind": "ExtraTrees", "trees": args.trees, "max_depth": 12, "min_samples_leaf": 8},
        "full_oof": {
            "outer_choices": outer_choices,
            "selected_policy_counts": {name: int(selected_count[name]) for name in POLICIES},
            "metrics": nested_full, "delta": full_delta, "terminal_bootstrap": boot_full,
            "held_outer_fold_deltas": held_fold_deltas,
            "positive_outer_fold_counts": positive_outer_folds,
        },
        "common_to_fixed": {
            "folds": common_records, "pre_char20_scores": common_scores, "pre_char20_changed_rows": common_changed,
            "selected_policy": selected_fixed, "fixed_gate_rows": int(fixed_gate.sum()),
            "fixed_label_changed_rows": int(np.sum(fixed_probability.argmax(1) != base_fixed.argmax(1))),
            "metrics": fixed_result, "delta": fixed_delta, "terminal_bootstrap": boot_fixed,
        },
        "promotion_gate": {"required_positive_metrics": list(needed), "outer_stability": outer_stability,
                           "screen_pass": bool(passed), "export_eligible": False,
                           "decision": "screen result only; frozen-base caveat requires a separately rebuilt strict audit" if passed else "reject and preserve B43"},
        "seconds": round(time.perf_counter() - started, 2),
    }
    np.savez_compressed(out / "predictions.npz", ids=ids, y=y, terminal=terminal, outer_fold=stored_outer_fold, base=base,
                         residual=residual, nested=nested, fixed_index=fixed, fixed_terminal=fixed_terminal,
                        base_fixed=base_fixed, fixed_residual=fixed_residual, fixed_probability=fixed_probability)
    (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf8")
    print(json.dumps({"full_delta": full_delta, "fixed_delta": fixed_delta, "selected_fixed": selected_fixed,
                      "promotion_pass": passed, "seconds": report["seconds"]}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
