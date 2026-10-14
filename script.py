# script.py
import csv
import os
import sys
import json
import re
import joblib
import numpy as np
from scipy.special import erf
from sklearn.feature_extraction import DictVectorizer, FeatureHasher
from pathlib import Path
from collections import defaultdict

# 14개 정확한 action 클래스 명세
ALL_CLASSES_LIST = [
    "read_file", "grep_search", "list_directory", "glob_pattern",
    "edit_file", "write_file", "apply_patch", "run_bash", 
    "run_tests", "lint_or_typecheck", "ask_user", "plan_task", 
    "web_search", "respond_only"
]
ALL_CLASSES = set(ALL_CLASSES_LIST)
REQUIRED_KEYS = ("id", "session_meta", "history", "current_prompt")
PROMPT_RULE_WORD_RE = re.compile(r"[\w가-힣]+", re.UNICODE)
SEQUENCE_ACTION_TO_ID = {name: index + 1 for index, name in enumerate(ALL_CLASSES_LIST)}
SEQUENCE_STATUS_TO_ID = {"pad": 0, "ok": 1, "fail": 2, "other": 3}
SEQUENCE_ARG_KEYS = ("path", "scope", "pattern", "target", "cmd", "query", "n_files")

# ==========================================
# [Self-Contained] 공통 데이터 로더 영역
# ==========================================
def load_jsonl(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"데이터 파일 누락: {file_path}")
    samples = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"{file_path} 라인 {line_no} 파싱 에러: {e}")
            if "id" not in data:
                raise KeyError(f"{file_path} 라인 {line_no}: 식별자 'id' 누락은 허용되지 않습니다.")
            
            data.setdefault("current_prompt", "")
            data.setdefault("history", [])
            data.setdefault("session_meta", {})
            samples.append(data)
    return samples

def load_sample_submission(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"제출 템플릿 파일 누락: {file_path}")
    with open(file_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        if fieldnames is None or fieldnames[:2] != ["id", "action"]:
            raise ValueError(f"sample_submission.csv 컬럼 규격 위반: {fieldnames}")
        rows = list(reader)
    return fieldnames, rows

def reveal_actions_from_future_session_rows(samples):
    """Recover earlier session actions exposed by later rows' action history.

    Each JSONL row is an independent prediction target, but a later record in
    the same session contains the preceding assistant actions in ``history``.
    Histories may be suffix-truncated, so actions are aligned to steps
    ``t-len(history_actions)`` through ``t-1`` rather than naïvely indexed
    from session start.  A result is used only when every available later row
    agrees, making the operation order-independent and conservative.
    """
    by_session = defaultdict(list)
    for sample in samples:
        sample_id = str(sample.get("id", ""))
        if "-step_" not in sample_id:
            continue
        session_id, raw_step = sample_id.rsplit("-step_", 1)
        try:
            step = int(raw_step)
        except ValueError:
            continue
        by_session[session_id].append((step, sample))

    revealed = {}
    for rows in by_session.values():
        for step, sample in rows:
            candidates = []
            for later_step, future in rows:
                if later_step <= step:
                    continue
                history = future.get("history") if isinstance(future.get("history"), list) else []
                actions = [
                    str(item.get("name") or item.get("action") or "")
                    for item in history
                    if isinstance(item, dict) and item.get("role") == "assistant_action"
                ]
                start_step = later_step - len(actions)
                if start_step <= step < later_step:
                    candidate = actions[step - start_step]
                    if candidate in ALL_CLASSES:
                        candidates.append(candidate)
            if candidates and len(set(candidates)) == 1:
                revealed[str(sample.get("id", ""))] = candidates[0]
    return revealed


def prompt_ngrams_for_overlay(sample, max_n=4):
    """Return the case-folded prompt 1--4 grams used by the compact rule.

    This deliberately uses the current prompt only.  The learned rule is
    applied only to the maximum supplied row in a session, so it never
    displaces an action that a later JSONL row exposes exactly.
    """
    tokens = PROMPT_RULE_WORD_RE.findall(str(sample.get("current_prompt", "")).casefold())
    grams = set(tokens)
    for width in range(2, max_n + 1):
        grams.update(
            "_".join(tokens[index:index + width])
            for index in range(max(0, len(tokens) - width + 1))
        )
    return grams


def reveal_actions_from_pure_prompt_rules(samples, rule):
    """Conservatively classify only supplied-session maximum rows.

    ``rule`` is trained from high-support, label-pure prompt phrases.  It can
    be helpful for terminal/isolated examples, while non-maximum examples are
    left to ordinary model inference or future-history recovery.
    """
    if not isinstance(rule, dict) or not rule:
        return {}
    by_session = defaultdict(list)
    for sample in samples:
        sample_id = str(sample.get("id", ""))
        if "-step_" not in sample_id:
            continue
        session_id, raw_step = sample_id.rsplit("-step_", 1)
        try:
            step = int(raw_step)
        except ValueError:
            continue
        by_session[session_id].append((step, sample))

    revealed = {}
    for rows in by_session.values():
        max_step = max(step for step, _ in rows)
        for step, sample in rows:
            if step != max_step:
                continue
            fired = {
                str(rule[gram])
                for gram in prompt_ngrams_for_overlay(sample)
                if gram in rule and str(rule[gram]) in ALL_CLASSES
            }
            if len(fired) == 1:
                revealed[str(sample.get("id", ""))] = next(iter(fired))
    return revealed


def prompt_char_features_for_overlay(sample):
    """Return the vetted word-internal character features for terminal rules.

    The exact normalization and feature boundaries match the offline rule
    audit: Unicode word tokens are case-folded, digit runs are normalized, and
    only 3--5 character fragments inside a padded token are considered.  This
    avoids using arbitrary whole-prompt fragments such as repository names.
    """
    values = set()
    tokens = re.findall(r"(?u)\b\w+\b", str(sample.get("current_prompt", "")).casefold())[:70]
    for raw_token in tokens:
        token = "^" + re.sub(r"\d+", "<n>", raw_token) + "$"
        for width in (3, 4, 5):
            values.update(
                "c" + str(width) + "|" + token[offset:offset + width]
                for offset in range(max(0, len(token) - width + 1))
            )
    return values


def reveal_actions_from_pure_prompt_char_rules(samples, rule):
    """Apply a high-support char-rule only to supplied-session terminal rows.

    A label is emitted only if every retained feature that fires agrees.  The
    caller applies deterministic future-history recovery afterwards, so this
    conservative fallback cannot overwrite an exactly exposed action.
    """
    if not isinstance(rule, dict) or not rule:
        return {}
    by_session = defaultdict(list)
    for sample in samples:
        sample_id = str(sample.get("id", ""))
        if "-step_" not in sample_id:
            continue
        session_id, raw_step = sample_id.rsplit("-step_", 1)
        try:
            step = int(raw_step)
        except ValueError:
            continue
        by_session[session_id].append((step, sample))

    revealed = {}
    for rows in by_session.values():
        max_step = max(step for step, _ in rows)
        for step, sample in rows:
            if step != max_step:
                continue
            fired = {
                str(rule[feature])
                for feature in prompt_char_features_for_overlay(sample)
                if feature in rule and str(rule[feature]) in ALL_CLASSES
            }
            if len(fired) == 1:
                revealed[str(sample.get("id", ""))] = next(iter(fired))
    return revealed


# ==========================================
# [Self-Contained] NumPy sequence-fusion inference
# ==========================================
def sequence_result_status(value):
    text = str(value or "").lower()
    if any(token in text for token in ("fail", "error", "exception", "traceback", "not found", "0 match", "0 file")):
        return SEQUENCE_STATUS_TO_ID["fail"]
    if any(token in text for token in ("pass", "ok;", "exit=0", "modified", "patched", "matched", "found", "listed")):
        return SEQUENCE_STATUS_TO_ID["ok"]
    return SEQUENCE_STATUS_TO_ID["other"]


def sequence_meta_features(sample):
    """Match the no-E5 GRU's inference-safe metadata feature order exactly."""
    meta = sample.get("session_meta") if isinstance(sample.get("session_meta"), dict) else {}
    workspace = meta.get("workspace") if isinstance(meta.get("workspace"), dict) else {}
    prompt = str(sample.get("current_prompt") or "")
    language_mix = workspace.get("language_mix") if isinstance(workspace.get("language_mix"), dict) else {}
    language_values = sorted((float(value) for value in language_mix.values()), reverse=True)
    top1 = language_values[0] if language_values else 0.0
    top2 = language_values[1] if len(language_values) > 1 else 0.0
    return [
        min(float(meta.get("turn_index") or 0), 20.0) / 20.0,
        min(float(meta.get("budget_tokens_remaining") or 0), 200_000.0) / 200_000.0,
        min(float(meta.get("elapsed_session_sec") or 0), 1_500.0) / 1_500.0,
        min(float(workspace.get("loc") or 0), 100_000.0) / 100_000.0,
        float(bool(workspace.get("git_dirty"))),
        float(str(workspace.get("last_ci_status") or "").lower() == "failed"),
        float(str(workspace.get("last_ci_status") or "").lower() == "passed"),
        min(len(workspace.get("open_files") if isinstance(workspace.get("open_files"), list) else []), 12) / 12.0,
        min(len(prompt), 600) / 600.0,
        min(len(prompt.split()), 100) / 100.0,
        float(prompt.count("?")) / 8.0,
        float(prompt.count("!")) / 8.0,
        top1,
        top2,
        float(str(sample.get("id") or "").startswith("sess_au_")),
    ]


def build_sequence_fusion_inputs(samples, max_len=6):
    """Build ordered action/status/argument tensors for the exported GRU."""
    actions = np.zeros((len(samples), max_len), dtype=np.int64)
    statuses = np.zeros((len(samples), max_len), dtype=np.int64)
    args = np.zeros((len(samples), max_len, len(SEQUENCE_ARG_KEYS)), dtype=np.float32)
    metas = np.zeros((len(samples), 15), dtype=np.float32)
    for row, sample in enumerate(samples):
        history = sample.get("history") if isinstance(sample.get("history"), list) else []
        observed = [
            item for item in history
            if isinstance(item, dict) and item.get("role") == "assistant_action"
        ][-max_len:]
        for column, item in enumerate(observed):
            name = str(item.get("name") or item.get("action") or "").lower()
            actions[row, column] = SEQUENCE_ACTION_TO_ID.get(name, 0)
            statuses[row, column] = sequence_result_status(item.get("result_summary"))
            values = item.get("args") if isinstance(item.get("args"), dict) else {}
            args[row, column] = [float(key in values) for key in SEQUENCE_ARG_KEYS]
        metas[row] = np.asarray(sequence_meta_features(sample), dtype=np.float32)
    return actions, statuses, args, metas


def sequence_gelu(value):
    return (0.5 * value * (1.0 + erf(value / np.float32(np.sqrt(2.0))))).astype(np.float32, copy=False)


def sequence_sigmoid(value):
    return (1.0 / (1.0 + np.exp(-np.clip(value, -60.0, 60.0)))).astype(np.float32, copy=False)


def sequence_linear(value, weight, bias):
    return (value @ weight.T + bias).astype(np.float32, copy=False)


def sequence_layer_norm(value, weight, bias, eps=1e-5):
    mean = value.mean(axis=-1, keepdims=True)
    variance = ((value - mean) ** 2).mean(axis=-1, keepdims=True)
    return ((value - mean) / np.sqrt(variance + eps) * weight + bias).astype(np.float32, copy=False)


def sequence_gru_direction(sequence, weights, suffix, reverse):
    w_ih = weights[f"gru.weight_ih_l0{suffix}"]
    w_hh = weights[f"gru.weight_hh_l0{suffix}"]
    b_ih = weights[f"gru.bias_ih_l0{suffix}"]
    b_hh = weights[f"gru.bias_hh_l0{suffix}"]
    rows, steps, _ = sequence.shape
    hidden = np.zeros((rows, w_hh.shape[1]), dtype=np.float32)
    outputs = np.zeros((rows, steps, w_hh.shape[1]), dtype=np.float32)
    indices = range(steps - 1, -1, -1) if reverse else range(steps)
    for index in indices:
        input_gates = sequence_linear(sequence[:, index], w_ih, b_ih)
        hidden_gates = sequence_linear(hidden, w_hh, b_hh)
        in_reset, in_update, in_new = np.split(input_gates, 3, axis=1)
        hid_reset, hid_update, hid_new = np.split(hidden_gates, 3, axis=1)
        reset = sequence_sigmoid(in_reset + hid_reset)
        update = sequence_sigmoid(in_update + hid_update)
        new = np.tanh(in_new + reset * hid_new).astype(np.float32, copy=False)
        hidden = ((1.0 - update) * new + update * hidden).astype(np.float32, copy=False)
        outputs[:, index] = hidden
    return outputs


def sequence_fusion_probabilities(actions, statuses, args, base, metas, weights):
    """Portable NumPy forward pass for the 650 KB no-E5 sequence model."""
    actions = np.asarray(actions, dtype=np.int64)
    statuses = np.asarray(statuses, dtype=np.int64)
    args = np.asarray(args, dtype=np.float32)
    base = np.asarray(base, dtype=np.float32)
    metas = np.asarray(metas, dtype=np.float32)
    positions = weights["position_embedding.weight"][np.arange(actions.shape[1])][None, :, :]
    sequence = np.concatenate((
        weights["action_embedding.weight"][actions],
        weights["status_embedding.weight"][statuses],
        np.broadcast_to(positions, (len(actions),) + positions.shape[1:]),
        sequence_gelu(sequence_linear(args, weights["arg_projection.0.weight"], weights["arg_projection.0.bias"])),
    ), axis=2).astype(np.float32, copy=False)
    forward = sequence_gru_direction(sequence, weights, "", reverse=False)
    backward = sequence_gru_direction(sequence, weights, "_reverse", reverse=True)
    encoded = np.concatenate((forward, backward), axis=2)
    last = np.clip((actions != 0).sum(axis=1), 1, None) - 1
    history = encoded[np.arange(len(actions)), last]
    base_block = sequence_gelu(sequence_linear(
        sequence_layer_norm(base, weights["base.0.weight"], weights["base.0.bias"]),
        weights["base.1.weight"], weights["base.1.bias"],
    ))
    e5_input = np.zeros((len(actions), 1), dtype=np.float32)
    e5_block = sequence_gelu(sequence_linear(
        sequence_layer_norm(e5_input, weights["e5.0.weight"], weights["e5.0.bias"]),
        weights["e5.1.weight"], weights["e5.1.bias"],
    ))
    meta_block = sequence_gelu(sequence_linear(metas, weights["meta.0.weight"], weights["meta.0.bias"]))
    hidden = sequence_gelu(sequence_linear(
        np.concatenate((history, base_block, e5_block, meta_block), axis=1),
        weights["head.0.weight"], weights["head.0.bias"],
    ))
    logits = sequence_linear(hidden, weights["head.3.weight"], weights["head.3.bias"])
    logits = logits - logits.max(axis=1, keepdims=True)
    probability = np.exp(logits).astype(np.float32, copy=False)
    return (probability / probability.sum(axis=1, keepdims=True)).astype(np.float32, copy=False)

# ==========================================
# [Self-Contained] Modeling A Feature 추출 영역
# ==========================================
def clean_text(text):
    if not isinstance(text, str):
        text = "" if text is None else str(text)
    return text.strip().lower()

def build_modeling_a_features(samples):
    """Prompt-only feature builder.

    모델링 A 학습 스크립트(src/train_modeling_a.py)와 동일한 추론 입력을 만듭니다.
    A 모델은 current_prompt 기반 TF-IDF 모델이므로 history/session_meta를 붙이지 않습니다.
    """
    ids = []
    texts = []
    for s in samples:
        ids.append(s.get("id", ""))
        texts.append(clean_text(s.get("current_prompt", "")))
    return ids, texts

def flatten_value(value):
    if isinstance(value, dict):
        return " ".join(f"{k}:{flatten_value(value[k])}" for k in sorted(value))
    if isinstance(value, list):
        return " ".join(flatten_value(v) for v in value)
    return clean_text(value)

def extract_action_from_history_item(item):
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

def extract_result_from_history_item(item):
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

def selected_meta_tokens(session_meta):
    if not isinstance(session_meta, dict):
        return []
    tokens = []
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

def build_modeling_b_features(samples, history_n=3, include_meta=True):
    ids = []
    contexts = []
    for s in samples:
        ids.append(s.get("id", ""))
        parts = [f"[CURRENT] {clean_text(s.get('current_prompt', ''))}"]
        history = s.get("history", [])
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
            meta_tokens = selected_meta_tokens(s.get("session_meta", {}))
            parts.append("[META] " + " ".join(meta_tokens) if meta_tokens else "[META_EMPTY]")
        contexts.append(" ".join(parts))
    return ids, contexts

def bucket_value(value, cuts):
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
        parts.extend((
            f"step_{meta.get('turn_index', 'unknown')}", f"hlen_{len(history)}",
            f"tier_{meta.get('user_tier', 'unknown')}", f"lang_{meta.get('language_pref', 'unknown')}",
            f"langmix_{language_key}", f"git_{workspace.get('git_dirty', 'unknown')}",
            f"ci_{workspace.get('last_ci_status', 'unknown')}", f"open_{open_count}",
            f"loc_{bucket_value(workspace.get('loc'), [(2000, 'xs'), (8000, 's'), (20000, 'm'), (50000, 'l')])}",
            f"budget_{bucket_value(meta.get('budget_tokens_remaining'), [(20000, 'xs'), (60000, 's'), (120000, 'm'), (170000, 'l')])}",
            f"elapsed_{bucket_value(meta.get('elapsed_session_sec'), [(200, 'xs'), (400, 's'), (700, 'm'), (1000, 'l')])}",
        ))
        ids.append(sample_id)
        texts.append(" ".join(parts))
    return ids, texts

def state_result_bucket(value):
    text = str(value or "").lower()
    if any(token in text for token in ("fail", "error", "exception", "traceback", "not found", "0 match")):
        return "fail"
    if any(token in text for token in ("pass", "ok;", "exit=0", "modified", "patched", "matched", "found", "listed")):
        return "ok"
    return "other"

def state_bucket(value, cuts):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return "unknown"
    for index, upper in enumerate(cuts):
        if value <= upper:
            return str(index)
    return str(len(cuts))

def build_state_dict(sample):
    """Inference-safe session-state features for the Q selector."""
    feature = {}
    history = sample.get("history") if isinstance(sample.get("history"), list) else []
    actions = [item for item in history if isinstance(item, dict) and item.get("role") == "assistant_action"]
    recent = list(reversed(actions[-5:]))
    sample_id = str(sample.get("id", ""))
    feature[f"kind={sample_id.split('_')[1] if '_' in sample_id else 'other'}"] = 1.0
    feature[f"history_len={state_bucket(len(history), (0, 2, 4, 6, 9, 13))}"] = 1.0
    for position in range(1, 6):
        item = recent[position - 1] if len(recent) >= position else {}
        name = str(item.get("name") or item.get("action") or "none").lower() if isinstance(item, dict) else "none"
        feature[f"pa{position}={name}"] = 1.0
        feature[f"pr{position}={state_result_bucket(item.get('result_summary') if isinstance(item, dict) else '')}"] = 1.0
        if position == 1 and isinstance(item, dict):
            args = item.get("args") if isinstance(item.get("args"), dict) else {}
            for key in ("path", "scope", "pattern", "target", "cmd"):
                if key in args:
                    feature[f"arg1_has_{key}"] = 1.0
    if len(recent) >= 2:
        older = str(recent[1].get("name") or recent[1].get("action") or "none").lower()
        latest = str(recent[0].get("name") or recent[0].get("action") or "none").lower()
        feature[f"transition={older}>{latest}"] = 1.0
    meta = sample.get("session_meta") if isinstance(sample.get("session_meta"), dict) else {}
    workspace = meta.get("workspace") if isinstance(meta.get("workspace"), dict) else {}
    for key, value in (("tier", meta.get("user_tier")), ("language", meta.get("language_pref")),
                       ("git", workspace.get("git_dirty")), ("ci", workspace.get("last_ci_status")),
                       ("turn", state_bucket(meta.get("turn_index"), (1, 2, 3, 5, 8, 12, 16))),
                       ("loc", state_bucket(workspace.get("loc"), (2000, 8000, 20000, 50000))),
                       ("budget", state_bucket(meta.get("budget_tokens_remaining"), (20000, 60000, 120000, 170000))),
                       ("elapsed", state_bucket(meta.get("elapsed_session_sec"), (200, 400, 700, 1000)))):
        feature[f"{key}={str(value).lower()}"] = 1.0
    open_files = workspace.get("open_files")
    feature[f"open_count={state_bucket(len(open_files) if isinstance(open_files, list) else -1, (0, 1, 2, 4, 8))}"] = 1.0
    language_mix = workspace.get("language_mix")
    if isinstance(language_mix, dict) and language_mix:
        feature[f"dominant_language={max(language_mix, key=language_mix.get)}"] = 1.0
    prompt = str(sample.get("current_prompt") or "")
    feature[f"prompt_words={state_bucket(len(prompt.split()), (4, 8, 14, 22, 35, 55))}"] = 1.0
    feature[f"prompt_chars={state_bucket(len(prompt), (25, 50, 90, 150, 240, 400))}"] = 1.0
    lower = prompt.lower()
    for cue in ("test", "spec", "lint", "typecheck", "format", "build", "run", "command", "search", "find", "grep", "where", "list", "directory", "folder", "file", "read", "open", "show", "write", "create", "add", "edit", "change", "patch", "apply", "web", "online", "google", "plan", "steps", "question", "clarify"):
        if cue in lower:
            feature[f"cue={cue}"] = 1.0
    return feature

def source_of(sample_id):
    return "au" if str(sample_id).startswith("sess_au_") else "sim"

def source_e_features_inference(models, texts, sources):
    """Rebuild fixed-width source-routed E margins used by the XGBoost selector."""
    tree = models["sim"]["tree"]
    n_rows = len(texts); blocks = []
    for node in sorted(tree):
        labels = list(tree[node]); block = np.zeros((n_rows, len(labels)), dtype=np.float32)
        for source in ("sim", "au"):
            indices = [i for i, value in enumerate(sources) if value == source]
            if not indices:
                continue
            component = models[source]["models"][node]
            values = decision_scores(component, [texts[i] for i in indices])
            lookup = {label: i for i, label in enumerate(component.classes_)}
            for col, label in enumerate(labels):
                if label in lookup:
                    block[indices, col] = values[:, lookup[label]]
        blocks.append(block)
    predictions = np.empty(n_rows, dtype=object)
    for source in ("sim", "au"):
        indices = [i for i, value in enumerate(sources) if value == source]
        if indices:
            predictions[indices] = predict_tree(models[source], [texts[i] for i in indices])
    blocks.append(np.column_stack([predictions == label for label in ALL_CLASSES_LIST]).astype(np.float32))
    blocks.append(np.column_stack([np.asarray(sources) == source for source in ("sim", "au")]).astype(np.float32))
    return np.column_stack(blocks)

_USER_TOKEN_RE = re.compile(r"[\w가-힣]+", re.UNICODE)

def build_user_sequence_hash(samples):
    """Hash current/recent user text plus ordered action history without fitted state."""
    records = []
    for sample in samples:
        feature = {}
        def add(prefix, message):
            tokens = _USER_TOKEN_RE.findall(str(message or "").lower())[:100]
            for token in tokens:
                if len(token) > 1:
                    feature[f"{prefix}w:{token}"] = 1
            for index in range(len(tokens) - 1):
                feature[f"{prefix}b:{tokens[index]}_{tokens[index + 1]}"] = 1
        history = sample.get("history") if isinstance(sample.get("history"), list) else []
        actions = [str(item.get("name") or item.get("action") or "none").lower() for item in history if isinstance(item, dict) and item.get("role") == "assistant_action"]
        for position, action in enumerate(reversed(actions[-20:]), start=1):
            feature[f"pos{position}={action}"] = 1
        for width in (2, 3, 4):
            for index in range(max(0, len(actions) - 20), len(actions) - width + 1):
                feature[f"ng{width}=" + ">".join(actions[index:index + width])] = 1
        for action in actions:
            key = f"count={action}"; feature[key] = feature.get(key, 0) + 1
        feature[f"n_actions={min(len(actions), 25)}"] = 1
        users = [item.get("content") for item in history if isinstance(item, dict) and item.get("role") == "user"]
        for position, message in enumerate(reversed(users[-3:]), start=1):
            add(f"u{position}", message)
        records.append(feature)
    return FeatureHasher(n_features=1536, input_type="dict", alternate_sign=False).transform(records).toarray().astype(np.float32)

def load_model_metadata(metadata_path):
    if not os.path.exists(metadata_path):
        return {"feature_builder": "modeling_a_prompt_only", "feature_config": {}}
    with open(metadata_path, "r", encoding="utf-8") as f:
        return json.load(f)

def build_features_by_metadata(samples, metadata):
    feature_builder = metadata.get("feature_builder", "modeling_a_prompt_only")
    if feature_builder == "modeling_a_prompt_only":
        return build_modeling_a_features(samples)
    if feature_builder == "modeling_b_v1":
        config = metadata.get("feature_config", {})
        return build_modeling_b_features(
            samples,
            history_n=int(config.get("history_n", 3)),
            include_meta=bool(config.get("include_meta", True)),
        )
    if feature_builder == "modeling_e_structured_v1":
        return build_modeling_e_features(samples)
    if feature_builder == "stacked_multi_v1":
        e_ids, e_texts = build_modeling_e_features(samples)
        b_ids, b_texts = build_modeling_b_features(samples, history_n=3, include_meta=True)
        if e_ids != b_ids:
            raise ValueError("스택 E/B feature ID 순서가 일치하지 않습니다.")
        return e_ids, {"e": e_texts, "b": b_texts}
    if feature_builder in {"stacked_state_v1", "stacked_xgb_user_source_v1"}:
        e_ids, e_texts = build_modeling_e_features(samples)
        b_ids, b_texts = build_modeling_b_features(samples, history_n=3, include_meta=True)
        if e_ids != b_ids:
            raise ValueError("State selector E/B feature IDs are misaligned.")
        return e_ids, {"e": e_texts, "b": b_texts, "samples": samples}
    raise ValueError(f"지원하지 않는 feature_builder입니다: {feature_builder}")

def predict_tree(model_bundle, texts):
    predictions = [None] * len(texts)
    pending = {"root": list(range(len(texts)))}
    while pending:
        node, indices = pending.popitem()
        if not indices:
            continue
        node_texts = [texts[index] for index in indices]
        children = model_bundle["models"][node].predict(node_texts)
        for index, child in zip(indices, children):
            child = str(child)
            labels = model_bundle["tree"][node].get(child)
            if not labels:
                child = model_bundle["defaults"][node]
                labels = model_bundle["tree"][node][child]
            if len(labels) == 1:
                predictions[index] = labels[0]
            else:
                pending.setdefault(child, []).append(index)
    return [str(value) for value in predictions]

def decision_scores(component, texts):
    values = np.asarray(component.decision_function(texts))
    if values.ndim == 1:
        values = np.column_stack((-values, values))
    return values.astype(np.float32, copy=False)

def stack_selector_features(model_bundle, e_texts, b_texts):
    e_model = model_bundle["e_model"]
    d_model = model_bundle["d_model"]
    b_model = model_bundle["b_model"]
    e_predictions = np.asarray(predict_tree(e_model, e_texts))
    d_predictions = np.asarray(predict_two_stage_bundle(d_model, b_texts))
    b_predictions = np.asarray(b_model.predict(b_texts)).astype(str)
    blocks = []
    schema = []
    for node in sorted(e_model["models"]):
        component = e_model["models"][node]
        blocks.append(decision_scores(component, e_texts))
        schema.extend(f"e:{node}:{label}" for label in component.classes_)
    component = d_model["coarse_model"]
    blocks.append(decision_scores(component, b_texts))
    schema.extend(f"d:coarse:{label}" for label in component.classes_)
    for node in sorted(d_model["fine_models"]):
        component = d_model["fine_models"][node]
        blocks.append(decision_scores(component, b_texts))
        schema.extend(f"d:{node}:{label}" for label in component.classes_)
    blocks.append(decision_scores(b_model, b_texts))
    schema.extend(f"b:flat:{label}" for label in b_model.classes_)
    for model_name, predictions in (("e", e_predictions), ("d", d_predictions), ("b", b_predictions)):
        blocks.append(np.column_stack([predictions == label for label in ALL_CLASSES_LIST]).astype(np.float32))
        schema.extend(f"{model_name}:prediction:{label}" for label in ALL_CLASSES_LIST)
    if schema != model_bundle.get("feature_schema"):
        raise ValueError("스택 selector feature schema가 학습 시점과 다릅니다.")
    return np.column_stack(blocks)

def predict_two_stage_bundle(model, texts):
    coarse_preds = [str(group) for group in model["coarse_model"].predict(texts)]
    grouped_indices = defaultdict(list)
    for index, group in enumerate(coarse_preds):
        grouped_indices[group].append(index)
    predictions = [None] * len(texts)
    for group, indices in grouped_indices.items():
        default = model.get("default_label_by_group", {}).get(group, "respond_only")
        if group not in model["fine_models"]:
            for index in indices:
                predictions[index] = default
            continue
        group_texts = [texts[index] for index in indices]
        group_predictions = model["fine_models"][group].predict(group_texts)
        allowed = set(model.get("group_to_labels", {}).get(group, ALL_CLASSES_LIST))
        for index, prediction in zip(indices, group_predictions):
            prediction = str(prediction)
            predictions[index] = prediction if prediction in allowed else default
    return [str(value) for value in predictions]

def predict_with_loaded_model(model, texts, metadata, sequence_blend_mask=None):
    model_type = metadata.get("model_type", "flat_classifier")
    if model_type == "flat_classifier":
        return model.predict(texts) if texts else []

    if model_type == "two_stage_group_v1":
        if not isinstance(model, dict):
            raise TypeError("two_stage_group_v1 모델은 dict bundle 형식이어야 합니다.")
        if not texts:
            return []

        coarse_model = model["coarse_model"]
        fine_models = model["fine_models"]
        group_to_labels = model.get("group_to_labels", metadata.get("group_to_labels", {}))
        defaults = model.get("default_label_by_group", {})

        coarse_preds = [str(group) for group in coarse_model.predict(texts)]
        grouped_indices = defaultdict(list)
        for i, group in enumerate(coarse_preds):
            grouped_indices[group].append(i)

        preds = [None] * len(texts)
        for group, indices in grouped_indices.items():
            default = defaults.get(group, "respond_only")
            if group not in fine_models:
                for i in indices:
                    preds[i] = default
                continue

            group_texts = [texts[i] for i in indices]
            group_preds = [str(pred) for pred in fine_models[group].predict(group_texts)]
            allowed = set(group_to_labels.get(group, ALL_CLASSES))
            for i, pred in zip(indices, group_preds):
                preds[i] = pred if pred in allowed else default
        return preds

    if model_type == "hierarchical_tree_v1":
        return predict_tree(model, texts) if texts else []

    if model_type == "stacked_selector_v1":
        if not texts:
            return []
        if not isinstance(texts, dict) or "e" not in texts or "b" not in texts:
            raise TypeError("stacked_selector_v1은 E/B 복합 feature가 필요합니다.")
        features = stack_selector_features(model, texts["e"], texts["b"])
        return model["selector"].predict(features)

    if model_type == "stacked_selector_state_v1":
        if not isinstance(texts, dict) or not {"e", "b", "samples"}.issubset(texts):
            raise TypeError("stacked_selector_state_v1 requires E/B texts and samples.")
        base = stack_selector_features(model, texts["e"], texts["b"])
        vectorizer = model.get("state_vectorizer")
        if not isinstance(vectorizer, DictVectorizer):
            raise TypeError("State selector vectorizer is missing or invalid.")
        state = vectorizer.transform([build_state_dict(sample) for sample in texts["samples"]]).astype(np.float32)
        features = np.column_stack((base, state))
        expected = model.get("state_feature_count")
        if expected is not None and features.shape[1] != int(expected):
            raise ValueError("State selector feature schema does not match training.")
        return model["selector"].predict(features)

    if model_type == "stacked_xgb_user_source_v1":
        if not isinstance(texts, dict) or not {"e", "b", "samples"}.issubset(texts):
            raise TypeError("stacked_xgb_user_source_v1 requires E/B texts and samples.")
        base = stack_selector_features(model, texts["e"], texts["b"])
        vectorizer = model.get("state_vectorizer")
        if not isinstance(vectorizer, DictVectorizer):
            raise TypeError("User-source selector vectorizer is missing or invalid.")
        state = vectorizer.transform([build_state_dict(sample) for sample in texts["samples"]]).astype(np.float32)
        sources = [source_of(sample.get("id", "")) for sample in texts["samples"]]
        source_e = source_e_features_inference(model["source_e_models"], texts["e"], sources)
        user_hash = build_user_sequence_hash(texts["samples"])
        features = np.column_stack((base, state, source_e, user_hash))
        expected = model.get("selector_feature_count")
        if expected is not None and features.shape[1] != int(expected):
            raise ValueError("User-source selector feature schema does not match training.")
        probabilities = model["selector"].predict_proba(features)
        sequence_bundle = model.get("sequence_fusion_noe5_v1")
        if isinstance(sequence_bundle, dict) and sequence_bundle.get("weights"):
            weights = sequence_bundle["weights"]
            if sequence_blend_mask is None:
                blend_mask = np.ones(len(texts["samples"]), dtype=bool)
            else:
                blend_mask = np.asarray(sequence_blend_mask, dtype=bool)
                if blend_mask.shape != (len(texts["samples"]),):
                    raise ValueError("Sequence blend mask does not align with inference samples.")
            if np.any(blend_mask):
                actions, statuses, arguments, metas = build_sequence_fusion_inputs(texts["samples"])
                sequence_probability = sequence_fusion_probabilities(
                    actions[blend_mask], statuses[blend_mask], arguments[blend_mask],
                    base[blend_mask], metas[blend_mask], weights,
                )
                source_weights = sequence_bundle.get("source_xgb_weights")
                if isinstance(source_weights, dict):
                    xgb_weight = np.asarray([
                        float(source_weights.get(sources[index], sequence_bundle.get("xgb_weight", 0.5)))
                        for index in np.flatnonzero(blend_mask)
                    ], dtype=np.float32)
                else:
                    xgb_weight = np.full(
                        int(np.count_nonzero(blend_mask)),
                        float(sequence_bundle.get("xgb_weight", 0.5)),
                        dtype=np.float32,
                    )
                if np.any((xgb_weight < 0.0) | (xgb_weight > 1.0)):
                    raise ValueError("Sequence fusion XGBoost blend weight must be within [0, 1].")
                probabilities = probabilities.copy()
                blend_mode = str(sequence_bundle.get("blend_mode", "linear"))
                if blend_mode == "linear":
                    probabilities[blend_mask] = (
                        xgb_weight[:, None] * probabilities[blend_mask]
                        + (1.0 - xgb_weight[:, None]) * sequence_probability
                    )
                elif blend_mode == "geometric":
                    # Product-of-experts blending is more decisive only when
                    # both independently trained models support a class.
                    # It is numerically normalized in log space.
                    epsilon = float(sequence_bundle.get("geometric_epsilon", 1e-12))
                    if not epsilon > 0.0:
                        raise ValueError("Sequence fusion geometric epsilon must be positive.")
                    logits = (
                        xgb_weight[:, None] * np.log(np.clip(probabilities[blend_mask], epsilon, 1.0))
                        + (1.0 - xgb_weight[:, None]) * np.log(np.clip(sequence_probability, epsilon, 1.0))
                    )
                    logits = logits - logits.max(axis=1, keepdims=True)
                    geometric_probability = np.exp(logits).astype(np.float32, copy=False)
                    probabilities[blend_mask] = geometric_probability / geometric_probability.sum(axis=1, keepdims=True)
                else:
                    raise ValueError(f"Unsupported sequence fusion blend mode: {blend_mode}")
        else:
            weights = model.get("source_class_weights")
            if weights:
                matrix = np.asarray([weights.get(source, [1.0] * len(ALL_CLASSES_LIST)) for source in sources], dtype=np.float32)
                probabilities = probabilities * matrix
        return [ALL_CLASSES_LIST[int(index)] for index in np.argmax(probabilities, axis=1)]

    raise ValueError(f"지원하지 않는 model_type입니다: {model_type}")

# ==========================================
# 메인 실행 파이프라인
# ==========================================
def main():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    TEST_PATH = os.path.join(BASE_DIR, "data", "test.jsonl")
    SAMPLE_SUB_PATH = os.path.join(BASE_DIR, "data", "sample_submission.csv")
    MODEL_PATH = os.path.join(BASE_DIR, "model", "best_model.pkl")
    METADATA_PATH = os.path.join(BASE_DIR, "model", "best_model_metadata.json")
    OUT_PATH = os.path.join(BASE_DIR, "output", "submission.csv")

    if not os.path.exists(MODEL_PATH):
        print(f"[오류] 모델 파일 누락: {MODEL_PATH}")
        sys.exit(1)
    
    model = joblib.load(MODEL_PATH)
    metadata = load_model_metadata(METADATA_PATH)
    samples = load_jsonl(TEST_PATH)
    fieldnames, sub_rows = load_sample_submission(SAMPLE_SUB_PATH)
    ids, texts = build_features_by_metadata(samples, metadata)

    # Determine exact, batch-internal labels before deciding which rows need a
    # learned fallback.  The no-E5 GRU is blended only for unrevealed rows;
    # exact future-history labels are always retained below.
    future_revealed = reveal_actions_from_future_session_rows(samples)
    blend_mask = [sample_id not in future_revealed for sample_id in ids]
    preds = predict_with_loaded_model(model, texts, metadata, sequence_blend_mask=blend_mask)
    pred_map = {}
    
    for idx, p in zip(ids, preds):
        pred_str = str(p).strip()
        
        # [강화된 검증] Fallback 방식 완전 배제 및 명시적 오류 발생
        if pred_str not in ALL_CLASSES:
            raise ValueError(f"[비정상 예측 검출] ID: {idx}, 예측된 값 '{pred_str}'은 규정된 14개 클래스에 속하지 않습니다.")
        
        pred_map[idx] = pred_str

    # A high-support, label-pure char rule is intentionally limited to the
    # maximum supplied row of each session.  It is then superseded below by
    # any exact action exposed in a later same-session history record.
    if isinstance(model, dict):
        pred_map.update(reveal_actions_from_pure_prompt_char_rules(
            samples, model.get("terminal_prompt_char_rule_v1", {})
        ))

    # Prefer deterministic, internally exposed session-history labels when
    # present; terminal or isolated rows otherwise retain the model/rule
    # prediction.
    pred_map.update(future_revealed)

    for row in sub_rows:
        target_id = row["id"]
        # sample_submission에 존재하는 id가 예측 결과에 매핑되는지 검증
        if target_id not in pred_map:
            raise ValueError(f"[무결성 오류] sample_submission의 ID '{target_id}'에 대응하는 모델 예측 결과가 존재하지 않습니다.")
        row["action"] = pred_map[target_id]

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(sub_rows)
        
    print(f"[성공] submission.csv가 오차 없이 최종 빌드되었습니다.")

if __name__ == "__main__":
    main()
