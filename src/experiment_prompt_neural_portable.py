#!/usr/bin/env python3
"""Strict audit of a small portable prompt-sequence classifier.

This experiment is intentionally isolated under ``experiments/``.  It uses
only ``current_prompt`` (no ID, target lookup, future history, or E5 features)
and learns word order through a hash-token CNN plus attention pooling.  The
tokenizer and all tensors are small enough to be copied to a NumPy-only
submission implementation if --export is reached.

Validation protocol
-------------------
* five session-GroupKFold models, each trained for a predeclared fixed number
  of epochs (the held fold is never used for early stopping),
* one separate model fit only on the frozen common-train partition and tested
  on common validation, and
* blend/gate hyperparameters selected in nested outer folds only.  The common
  validation labels are evaluation-only.

The terminal char20 rule remains the final learned override and non-terminal
rows are scored as future-history-recovered, matching the live fallback path.
No live model, script, or ZIP is opened for writing.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import re
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

# Set before importing torch; this avoids excessive WDDM fragmentation.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


ROOT = Path(__file__).resolve().parents[1]
CLASSES = (
    "read_file", "grep_search", "list_directory", "glob_pattern",
    "edit_file", "write_file", "apply_patch", "run_bash", "run_tests",
    "lint_or_typecheck", "ask_user", "plan_task", "web_search", "respond_only",
)
N_CLASSES = len(CLASSES)
LABEL_TO_ID = {value: index for index, value in enumerate(CLASSES)}
WORD_RE = re.compile(r"(?u)\b\w+\b")
DIGIT_RE = re.compile(r"\d+")
EPS = 1.0e-12


@dataclass(frozen=True)
class NetConfig:
    max_len: int = 36
    max_subwords: int = 12
    word_buckets: int = 32768
    char_buckets: int = 16384
    dim: int = 48
    conv_dim: int = 56
    hidden_dim: int = 176
    dropout: float = 0.12
    kernels: tuple[int, ...] = (2, 3, 4)


@dataclass(frozen=True)
class Policy:
    """A small fixed terminal fallback policy.

    ``blend`` is geometric blending between current raw-XGB+strict-GRU base
    and this independent prompt network.  ``gate`` is a deliberately
    conservative direct-neural replacement only when its confidence is high
    and the base is uncertain.
    """

    family: str
    base_weight: float = 1.0
    neural_confidence: float = 1.1
    base_confidence: float = 0.0

    def key(self) -> str:
        return (
            f"{self.family}|base={self.base_weight:.2f}|"
            f"neural_conf={self.neural_confidence:.2f}|base_conf={self.base_confidence:.2f}"
        )

    def complexity(self) -> tuple[int, float, float, float]:
        # On ties prefer no alteration, then a higher base contribution and
        # stricter gate.  This makes nested selection deterministic.
        rank = {"base": 0, "blend": 1, "gate": 2, "direct": 3}[self.family]
        return (-rank, self.base_weight, self.neural_confidence, -self.base_confidence)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="experiments/prompt_neural_portable")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--epochs", type=int, default=11,
                        help="Predeclared training epochs for every OOF/fixed model; no held-fold early stopping.")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=2.2e-3)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--seed", type=int, default=1701)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--distill", type=float, default=0.0,
                        help="Optional strict teacher KL coefficient. The teacher is training-only and absent at inference.")
    parser.add_argument("--teacher-outer", default="experiments/modeling_ai_sequence_nested_distill_e16_i4/nested_teacher_soft_targets.npy",
                        help="Full strict nested teacher targets; used only for outer-fold training rows.")
    parser.add_argument("--teacher-fixed", default="experiments/strict_common_train_teacher/full_teacher_trainonly.npy",
                        help="Common-train cross-fit teacher targets; common-validation target rows must be zero.")
    parser.add_argument("--export", action="store_true",
                        help="Refit/export only if the preselected candidate improves both strict protocols.")
    return parser.parse_args()


def stable_hash(text: str, seed: int) -> int:
    """FNV-1a on UTF-8 bytes, specified so NumPy-only inference is exact."""
    value = (2166136261 ^ seed) & 0xFFFFFFFF
    for byte in text.encode("utf-8", errors="replace"):
        value ^= byte
        value = (value * 16777619) & 0xFFFFFFFF
    return value


def normalized_tokens(value: object, max_len: int) -> list[str]:
    text = str(value or "").casefold().replace("\x00", " ")
    return [DIGIT_RE.sub("<n>", token) for token in WORD_RE.findall(text)[:max_len]]


def char_pieces(token: str, cap: int) -> list[str]:
    """Ordered 2--4 character pieces for one word token.

    They are embedded *inside* a token before phrase CNN processing, so this
    is not a sparse character TF-IDF feature path.
    """
    padded = "^" + token + "$"
    values: list[str] = []
    for width in (2, 3, 4):
        values.extend(padded[start:start + width] for start in range(max(0, len(padded) - width + 1)))
    if len(values) <= cap:
        return values
    # Preserve prefix/suffix morphology and spread the remaining slots across
    # the token deterministically rather than retaining only a prefix.
    positions = np.linspace(0, len(values) - 1, num=cap, dtype=int).tolist()
    return [values[position] for position in positions]


def cache_path(config: NetConfig) -> Path:
    return ROOT / "experiments" / "prompt_neural_portable_cache" / (
        f"tokens_l{config.max_len}_s{config.max_subwords}_w{config.word_buckets}_c{config.char_buckets}.npz"
    )


def make_token_arrays(rows: list[dict], config: NetConfig, path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build/load only inference-safe prompt tokenizer arrays."""
    ids = np.asarray([str(row["id"]) for row in rows])
    if path.exists():
        loaded = np.load(path, allow_pickle=False)
        required = {"ids", "word", "char", "length"}
        if required.issubset(loaded.files) and np.array_equal(loaded["ids"].astype(str), ids):
            return (loaded["word"].astype(np.int64), loaded["char"].astype(np.int64),
                    loaded["length"].astype(np.int64), ids)

    path.parent.mkdir(parents=True, exist_ok=True)
    word = np.zeros((len(rows), config.max_len), dtype=np.uint16)
    char = np.zeros((len(rows), config.max_len, config.max_subwords), dtype=np.uint16)
    length = np.zeros(len(rows), dtype=np.uint8)
    for row_number, row in enumerate(rows):
        tokens = normalized_tokens(row.get("current_prompt"), config.max_len)
        length[row_number] = len(tokens)
        for position, token in enumerate(tokens):
            word[row_number, position] = 1 + stable_hash(token, 0x9E3779B9) % (config.word_buckets - 1)
            for piece_number, piece in enumerate(char_pieces(token, config.max_subwords)):
                char[row_number, position, piece_number] = (
                    1 + stable_hash(piece, 0x85EBCA6B) % (config.char_buckets - 1)
                )
        if row_number and row_number % 10000 == 0:
            print(f"tokenize_rows={row_number}/{len(rows)}", flush=True)
    np.savez_compressed(path, ids=ids, word=word, char=char, length=length)
    return word.astype(np.int64), char.astype(np.int64), length.astype(np.int64), ids


class HashPhraseCNN(nn.Module):
    """Hash-token phrase model with explicit word-order branches."""

    def __init__(self, config: NetConfig) -> None:
        super().__init__()
        self.config = config
        self.word_embedding = nn.Embedding(config.word_buckets, config.dim, padding_idx=0)
        self.char_embedding = nn.Embedding(config.char_buckets, config.dim, padding_idx=0)
        self.position_embedding = nn.Embedding(config.max_len, config.dim)
        self.input_norm = nn.LayerNorm(config.dim)
        self.convolutions = nn.ModuleList([
            nn.Conv1d(config.dim, config.conv_dim, kernel_size=kernel) for kernel in config.kernels
        ])
        self.attention = nn.Linear(config.dim, 1, bias=False)
        feature_dim = len(config.kernels) * config.conv_dim * 2 + config.dim
        self.head = nn.Sequential(
            nn.Linear(feature_dim, config.hidden_dim), nn.GELU(), nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, N_CLASSES),
        )

    def forward(self, word: torch.Tensor, char: torch.Tensor, length: torch.Tensor) -> torch.Tensor:
        char_value = self.char_embedding(char)
        char_mask = char.ne(0).unsqueeze(-1)
        char_sum = (char_value * char_mask).sum(dim=2)
        char_count = char_mask.sum(dim=2).clamp_min(1)
        token = self.word_embedding(word) + char_sum / char_count
        position = torch.arange(word.shape[1], device=word.device).unsqueeze(0)
        token = self.input_norm(token + self.position_embedding(position))
        token = F.gelu(token)
        parts: list[torch.Tensor] = []
        channel_first = token.transpose(1, 2)
        for kernel, convolution in zip(self.config.kernels, self.convolutions):
            phrase = F.gelu(convolution(channel_first)).transpose(1, 2)
            phrase_length = (length - kernel + 1).clamp_min(0)
            mask = torch.arange(phrase.shape[1], device=word.device).unsqueeze(0) < phrase_length.unsqueeze(1)
            maximum = phrase.masked_fill(~mask.unsqueeze(-1), -1.0e4).amax(dim=1)
            maximum = torch.where((phrase_length > 0).unsqueeze(1), maximum, torch.zeros_like(maximum))
            average = (phrase * mask.unsqueeze(-1)).sum(dim=1) / phrase_length.clamp_min(1).unsqueeze(1)
            parts.extend((maximum, average))
        token_mask = torch.arange(word.shape[1], device=word.device).unsqueeze(0) < length.unsqueeze(1)
        attention_logits = self.attention(token).squeeze(-1).masked_fill(~token_mask, -1.0e4)
        attention = torch.softmax(attention_logits, dim=1)
        attention = torch.where(token_mask.any(dim=1, keepdim=True), attention, torch.zeros_like(attention))
        attentive = (token * attention.unsqueeze(-1)).sum(dim=1)
        return self.head(torch.cat((*parts, attentive), dim=1))


def set_seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def macro(y: np.ndarray, prediction: np.ndarray, mask: np.ndarray | None = None) -> float:
    if mask is None:
        mask = np.ones(len(y), dtype=bool)
    return float(f1_score(y[mask], prediction[mask], labels=np.arange(N_CLASSES), average="macro", zero_division=0))


def probability(model: nn.Module, word: np.ndarray, char: np.ndarray, length: np.ndarray, indices: np.ndarray,
                batch_size: int, device: torch.device) -> np.ndarray:
    model.eval()
    dataset = TensorDataset(
        torch.from_numpy(word[indices]), torch.from_numpy(char[indices]), torch.from_numpy(length[indices]),
    )
    loader = DataLoader(dataset, batch_size=batch_size * 2, shuffle=False, pin_memory=device.type == "cuda")
    values: list[np.ndarray] = []
    with torch.inference_mode():
        for batch in loader:
            inputs = [item.to(device, non_blocking=True) for item in batch]
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                logits = model(*inputs)
            values.append(torch.softmax(logits, dim=1).float().cpu().numpy())
    return np.concatenate(values).astype(np.float32)


def fit_model(word: np.ndarray, char: np.ndarray, length: np.ndarray, y: np.ndarray, indices: np.ndarray,
              config: NetConfig, args: argparse.Namespace, seed: int, device: torch.device,
              teacher: np.ndarray | None = None) -> HashPhraseCNN:
    """Fit a fixed-epoch model; no evaluation labels enter training control."""
    set_seed(seed)
    model = HashPhraseCNN(config).to(device)
    count = np.bincount(y[indices], minlength=N_CLASSES).astype(np.float32)
    class_weight = torch.tensor(np.sqrt(len(indices) / (N_CLASSES * count)), device=device, dtype=torch.float32)
    tensors: list[torch.Tensor] = [
        torch.from_numpy(word[indices]), torch.from_numpy(char[indices]), torch.from_numpy(length[indices]),
        torch.from_numpy(y[indices]),
    ]
    if teacher is not None:
        targets = teacher[indices].astype(np.float32)
        if targets.shape != (len(indices), N_CLASSES) or not np.allclose(targets.sum(axis=1), 1.0, atol=2.0e-5):
            raise ValueError("Teacher targets must be normalized 14-class cross-fit probabilities for every training row")
        tensors.append(torch.from_numpy(targets))
    dataset = TensorDataset(*tensors)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                        generator=torch.Generator().manual_seed(seed), pin_memory=device.type == "cuda")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # A predetermined cosine schedule is independent of every held-fold label.
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1), eta_min=args.lr * 0.12)
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        for batch in loader:
            values = [item.to(device, non_blocking=True) for item in batch]
            if teacher is None:
                w, c, l, target = values
                soft_target = None
            else:
                w, c, l, target, soft_target = values
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                logits = model(w, c, l)
                hard_loss = F.cross_entropy(logits, target, weight=class_weight, label_smoothing=0.012)
                if soft_target is None:
                    loss = hard_loss
                else:
                    teacher_loss = F.kl_div(torch.log_softmax(logits, dim=1), soft_target, reduction="batchmean")
                    loss = (1.0 - args.distill) * hard_loss + args.distill * teacher_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            total += float(loss.detach().cpu()) * len(target)
        schedule.step()
        print(f"train_seed={seed} epoch={epoch}/{args.epochs} loss={total / len(indices):.5f}", flush=True)
    return model


def source_of_ids(ids: np.ndarray) -> np.ndarray:
    return np.asarray(["au" if value.startswith("sess_au_") else "sim" for value in ids.astype(str)])


def geometric(first: np.ndarray, second: np.ndarray, first_weight: np.ndarray | float) -> np.ndarray:
    weight = np.asarray(first_weight, dtype=np.float64)
    if weight.ndim == 1:
        weight = weight[:, None]
    logits = weight * np.log(np.maximum(first, EPS)) + (1.0 - weight) * np.log(np.maximum(second, EPS))
    logits -= logits.max(axis=1, keepdims=True)
    result = np.exp(logits)
    return (result / result.sum(axis=1, keepdims=True)).astype(np.float32)


def current_base(raw_xgb: np.ndarray, strict_gru: np.ndarray, source: np.ndarray) -> np.ndarray:
    weight = np.where(source == "sim", 0.20, 0.85).astype(np.float32)
    return geometric(raw_xgb, strict_gru, weight)


def candidate_policies() -> list[Policy]:
    values = [Policy("base")]
    values.extend(Policy("direct") for _ in [0])
    values.extend(Policy("blend", base_weight=value) for value in (0.45, 0.55, 0.65, 0.75, 0.82, 0.88, 0.93))
    values.extend(
        Policy("gate", neural_confidence=neural_conf, base_confidence=base_conf)
        for neural_conf in (0.65, 0.72, 0.80, 0.86, 0.90, 0.94)
        for base_conf in (0.55, 0.62, 0.70, 0.78, 0.86, 0.94)
    )
    return values


def prediction_for_policy(base: np.ndarray, neural: np.ndarray, policy: Policy) -> np.ndarray:
    if policy.family == "base":
        return base.argmax(axis=1).astype(np.int16)
    if policy.family == "direct":
        return neural.argmax(axis=1).astype(np.int16)
    if policy.family == "blend":
        return geometric(base, neural, policy.base_weight).argmax(axis=1).astype(np.int16)
    if policy.family == "gate":
        output = base.argmax(axis=1).astype(np.int16)
        neural_prediction = neural.argmax(axis=1).astype(np.int16)
        modify = (neural.max(axis=1) >= policy.neural_confidence) & (base.max(axis=1) <= policy.base_confidence)
        output[modify] = neural_prediction[modify]
        return output
    raise ValueError(policy.family)


def apply_char20(prediction: np.ndarray, terminal: np.ndarray, char20: np.ndarray) -> np.ndarray:
    output = prediction.copy()
    use = terminal & (char20 >= 0)
    output[use] = char20[use]
    return output


def metrics(y: np.ndarray, terminal: np.ndarray, prediction: np.ndarray, char20: np.ndarray) -> dict[str, float]:
    final = apply_char20(prediction, terminal, char20)
    hybrid = final.copy(); hybrid[~terminal] = y[~terminal]
    return {
        "terminal_fallback_macro_f1": macro(y, final, terminal),
        "hybrid_future_history_plus_char20_macro_f1": macro(y, hybrid),
        "changed_before_char20": float(np.sum(prediction[terminal] != prediction_for_policy_placeholder(prediction[terminal]))),
    }


def prediction_for_policy_placeholder(prediction: np.ndarray) -> np.ndarray:
    # Helper only keeps ``metrics``'s legacy-compatible changed counter out of
    # the model-selection decision.  Actual per-policy changes are recorded
    # by ``policy_metrics`` below.
    return prediction


def policy_metrics(y: np.ndarray, terminal: np.ndarray, base: np.ndarray, neural: np.ndarray,
                   char20: np.ndarray, policy: Policy) -> dict[str, float]:
    prediction = prediction_for_policy(base, neural, policy)
    final = apply_char20(prediction, terminal, char20)
    hybrid = final.copy(); hybrid[~terminal] = y[~terminal]
    base_prediction = prediction_for_policy(base, base, Policy("base"))
    return {
        "terminal_fallback_macro_f1": macro(y, final, terminal),
        "hybrid_future_history_plus_char20_macro_f1": macro(y, hybrid),
        "terminal_changed_before_char20": int(np.sum((prediction != base_prediction) & terminal)),
        "terminal_changed_after_char20": int(np.sum((final != apply_char20(base_prediction, terminal, char20)) & terminal)),
    }


def choose_nested(y: np.ndarray, terminal: np.ndarray, base: np.ndarray, neural: np.ndarray, char20: np.ndarray,
                  fold_assignment: np.ndarray, policies: list[Policy]) -> tuple[np.ndarray, list[dict], Policy]:
    """Nested selection: no held session-fold labels select its own gate."""
    output = prediction_for_policy(base, neural, Policy("base"))
    selected: list[dict] = []
    selected_policies: list[Policy] = []
    for held in sorted(np.unique(fold_assignment).tolist()):
        train = fold_assignment != held
        candidates: list[tuple[float, tuple[int, float, float, float], Policy]] = []
        for policy in policies:
            value = policy_metrics(y[train], terminal[train], base[train], neural[train], char20[train], policy)
            candidates.append((value["terminal_fallback_macro_f1"], policy.complexity(), policy))
        # First score, then deterministic conservative complexity preference.
        score, _, best = max(candidates, key=lambda item: (item[0], item[1]))
        hold = fold_assignment == held
        output[hold] = prediction_for_policy(base[hold], neural[hold], best)
        selected_policies.append(best)
        selected.append({"held_outer_fold": int(held + 1), "selected": best.key(), "train_terminal_macro_f1": score})
    # Deployment config uses only full-training OOF evidence, never common
    # validation labels.  A modal config is safer than an all-OOF best score.
    counts = Counter(policy.key() for policy in selected_policies)
    winner_key = max(counts, key=lambda key: (counts[key], next(p for p in selected_policies if p.key() == key).complexity()))
    winner = next(policy for policy in selected_policies if policy.key() == winner_key)
    return output, selected, winner


def load_inputs(config: NetConfig) -> dict[str, object]:
    rows = [json.loads(line) for line in (ROOT / "data/train.jsonl").open(encoding="utf-8") if line.strip()]
    labels = {row["id"]: row["action"] for row in csv.DictReader((ROOT / "data/train_labels.csv").open(encoding="utf-8"))}
    split = {row["id"]: row["split"] for row in csv.DictReader((ROOT / "splits/common_split.csv").open(encoding="utf-8"))}
    word, char, length, ids = make_token_arrays(rows, config, cache_path(config))
    y = np.asarray([LABEL_TO_ID[labels[row_id]] for row_id in ids], dtype=np.int64)
    groups = np.asarray([value.rsplit("-step_", 1)[0] for value in ids])
    terminal = np.zeros(len(ids), dtype=bool)
    maximum: dict[str, int] = {}
    parsed: list[tuple[str, int]] = []
    for value in ids:
        session, step = value.rsplit("-step_", 1); number = int(step)
        parsed.append((session, number)); maximum[session] = max(maximum.get(session, -1), number)
    terminal[:] = [step == maximum[session] for session, step in parsed]
    fixed_train = np.asarray([number for number, row_id in enumerate(ids) if split[row_id] == "train"], dtype=np.int64)
    fixed_valid = np.asarray([number for number, row_id in enumerate(ids) if split[row_id] == "val"], dtype=np.int64)
    folds = list(GroupKFold(n_splits=5).split(np.zeros(len(y)), y, groups))
    fold_assignment = np.empty(len(y), dtype=np.int8)
    for number, (_, valid) in enumerate(folds):
        fold_assignment[valid] = number

    raw_xgb = np.load(ROOT / "experiments/modeling_ak_live_xgb_proxy/oof_probabilities.npy").astype(np.float32)
    strict_gru = np.load(ROOT / "experiments/modeling_ai_sequence_nested_distill_e16_i4/oof_probabilities.npy").astype(np.float32)
    if raw_xgb.shape != strict_gru.shape or raw_xgb.shape != (len(y), N_CLASSES):
        raise ValueError("Full OOF fallback artifacts do not align")
    fixed_xgb = np.load(ROOT / "experiments/au_source_strict_common/raw_global_valid_probability.npy").astype(np.float32)
    fixed_gru = np.load(ROOT / "experiments/modeling_ai_sequence_fusion_fixed_strictdistill_w35/valid_probabilities.npy").astype(np.float32)
    if fixed_xgb.shape != fixed_gru.shape or fixed_xgb.shape != (len(fixed_valid), N_CLASSES):
        raise ValueError("Strict fixed fallback artifacts do not align")

    char_artifact = np.load(ROOT / "experiments/terminal_rule_combinations/oof_candidates.npz", allow_pickle=False)
    if not (np.array_equal(char_artifact["y"].astype(np.int64), y) and np.array_equal(char_artifact["terminal"].astype(bool), terminal)):
        raise ValueError("char20 outer artifact does not align")
    char20 = char_artifact["char20"].astype(np.int16)
    fixed_char20 = np.full(len(fixed_valid), -1, dtype=np.int16)
    fixed_terminal = terminal[fixed_valid]
    fixed_ids = ids[fixed_valid]
    if not np.array_equal(char_artifact["valid_ids"].astype(str), fixed_ids[fixed_terminal]):
        raise ValueError("char20 strict fixed IDs do not align")
    fixed_char20[fixed_terminal] = char_artifact["fixed_char20"].astype(np.int16)
    expected_strict_fold = np.load(ROOT / "experiments/modeling_ai_sequence_nested_distill_e16_i4/outer_fold_assignment.npy").astype(np.int8)
    if not np.array_equal(expected_strict_fold, fold_assignment):
        raise ValueError("Prompt OOF session folds differ from strict GRU folds")
    return {
        "rows": rows, "word": word, "char": char, "length": length, "ids": ids, "y": y, "groups": groups,
        "terminal": terminal, "folds": folds, "fold_assignment": fold_assignment, "fixed_train": fixed_train,
        "fixed_valid": fixed_valid, "raw_xgb": raw_xgb, "strict_gru": strict_gru, "fixed_xgb": fixed_xgb,
        "fixed_gru": fixed_gru, "char20": char20, "fixed_char20": fixed_char20,
    }


def export_portable(model: HashPhraseCNN, config: NetConfig, policy: Policy, out: Path) -> None:
    """Save only small tokenizer/config/NumPy arrays for potential integration."""
    export_dir = out / "portable_candidate"
    export_dir.mkdir(parents=True, exist_ok=True)
    arrays = {name.replace(".", "__"): tensor.detach().cpu().numpy().astype(np.float32)
              for name, tensor in model.state_dict().items()}
    np.savez_compressed(export_dir / "weights.npz", **arrays)
    payload = {
        "format": "prompt_hash_phrase_cnn_v1",
        "input": "current_prompt only; regex Unicode words, digit normalization, FNV-1a hashes, ordered 2/3/4-gram CNN + attention",
        "architecture": asdict(config),
        "policy": asdict(policy),
        "portable_runtime": "NumPy embedding lookup, GELU, valid convolution, max/mean/attention pooling, two dense layers",
        "forbidden_inputs": ["id", "labels", "future_history", "E5", "teacher"],
    }
    (export_dir / "config.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but unavailable")
    if not 0.0 <= args.distill < 1.0:
        raise ValueError("--distill must be in [0, 1)")
    config = NetConfig()
    out = ROOT / args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    started = time.time()
    data = load_inputs(config)
    preparation = {
        "scope": "Experiment only; no live model/script/submission archive is changed.",
        "architecture": asdict(config),
        "tokenizer": "current_prompt Unicode word hashes + in-token char piece hashes; no TF-IDF vocabulary or linear classifier",
        "rows": int(len(data["y"])), "terminal_rows": int(np.sum(data["terminal"])),
        "fixed_train_rows": int(len(data["fixed_train"])), "fixed_validation_rows": int(len(data["fixed_valid"])),
        "fixed_terminal_rows": int(np.sum(np.asarray(data["terminal"])[np.asarray(data["fixed_valid"])])),
        "epochs_predeclared": args.epochs,
        "distillation": {
            "enabled": args.distill > 0.0,
            "coefficient": args.distill,
            "contract": (
                "Only cross-fit teacher probabilities from other session groups are used for training rows; "
                "teacher/E5 are absent from prompt model inference."
            ),
        },
        "validation": "5-fold whole-session OOF with fixed epochs; independent strict common split; char20 final override after neural decision; nonterminal future-history recovery only in hybrid metric.",
    }
    (out / "preparation.json").write_text(json.dumps(preparation, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.prepare_only:
        print(json.dumps({"status": "prepared", **preparation}, ensure_ascii=False), flush=True)
        return

    device = torch.device(args.device)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True
    word = np.asarray(data["word"]); char = np.asarray(data["char"]); length = np.asarray(data["length"])
    y = np.asarray(data["y"]); terminal = np.asarray(data["terminal"]); folds = data["folds"]
    outer_teacher: np.ndarray | None = None
    fixed_teacher: np.ndarray | None = None
    if args.distill > 0.0:
        outer_teacher = np.load(ROOT / args.teacher_outer).astype(np.float32)
        fixed_teacher = np.load(ROOT / args.teacher_fixed).astype(np.float32)
        if outer_teacher.shape != (len(y), N_CLASSES) or fixed_teacher.shape != (len(y), N_CLASSES):
            raise ValueError("Strict teacher arrays do not align with current rows/classes")
        fixed_train_check = np.asarray(data["fixed_train"])
        fixed_valid_check = np.asarray(data["fixed_valid"])
        if not (np.allclose(outer_teacher.sum(axis=1), 1.0, atol=2.0e-5)
                and np.allclose(fixed_teacher[fixed_train_check].sum(axis=1), 1.0, atol=2.0e-5)
                and np.allclose(fixed_teacher[fixed_valid_check], 0.0, atol=2.0e-6)):
            raise ValueError("Teacher strictness contract failed: expected outer cross-fit and zero common-validation targets")
    outer_probability = np.zeros((len(y), N_CLASSES), dtype=np.float32)
    fold_records: list[dict] = []
    for number, (train, valid) in enumerate(folds, 1):
        begun = time.time()
        print(f"outer_fold={number}/5 fit_rows={len(train)} holdout_rows={len(valid)}", flush=True)
        model = fit_model(word, char, length, y, train.astype(np.int64), config, args, args.seed + number, device,
                          teacher=outer_teacher)
        outer_probability[valid] = probability(model, word, char, length, valid.astype(np.int64), args.batch_size, device)
        direct = outer_probability[valid].argmax(axis=1)
        fold_records.append({"fold": number, "train_rows": int(len(train)), "valid_rows": int(len(valid)),
                             "direct_macro_f1": macro(y[valid], direct), "seconds": round(time.time() - begun, 2)})
        print(json.dumps(fold_records[-1]), flush=True)
        del model
        if device.type == "cuda": torch.cuda.empty_cache()

    fixed_train = np.asarray(data["fixed_train"]); fixed_valid = np.asarray(data["fixed_valid"])
    print(f"fixed_fit_rows={len(fixed_train)} fixed_valid_rows={len(fixed_valid)}", flush=True)
    fixed_model = fit_model(word, char, length, y, fixed_train, config, args, args.seed + 99, device,
                            teacher=fixed_teacher)
    fixed_probability = probability(fixed_model, word, char, length, fixed_valid, args.batch_size, device)

    ids = np.asarray(data["ids"])
    source = source_of_ids(ids)
    fixed_source = source[fixed_valid]
    base = current_base(np.asarray(data["raw_xgb"]), np.asarray(data["strict_gru"]), source)
    fixed_base = current_base(np.asarray(data["fixed_xgb"]), np.asarray(data["fixed_gru"]), fixed_source)
    char20 = np.asarray(data["char20"]); fixed_char20 = np.asarray(data["fixed_char20"])
    fixed_y = y[fixed_valid]; fixed_terminal = terminal[fixed_valid]
    policies = candidate_policies()
    fold_assignment = np.asarray(data["fold_assignment"])
    nested_prediction, selected, modal_policy = choose_nested(y, terminal, base, outer_probability, char20, fold_assignment, policies)
    nested_final = apply_char20(nested_prediction, terminal, char20)
    nested_hybrid = nested_final.copy(); nested_hybrid[~terminal] = y[~terminal]
    baseline_outer = policy_metrics(y, terminal, base, outer_probability, char20, Policy("base"))
    nested_metrics = {
        "terminal_fallback_macro_f1": macro(y, nested_final, terminal),
        "hybrid_future_history_plus_char20_macro_f1": macro(y, nested_hybrid),
    }
    fixed_baseline = policy_metrics(fixed_y, fixed_terminal, fixed_base, fixed_probability, fixed_char20, Policy("base"))
    fixed_modal = policy_metrics(fixed_y, fixed_terminal, fixed_base, fixed_probability, fixed_char20, modal_policy)
    static = [
        {"policy": policy.key(), **policy_metrics(y, terminal, base, outer_probability, char20, policy)}
        for policy in policies
    ]
    static_best = max(static, key=lambda value: (value["terminal_fallback_macro_f1"], value["policy"]))
    # Require a non-base portable policy and strict gains in both independently
    # reported terminal and deployment-aligned hybrid metrics.  Fixed labels do
    # not select the policy; they only decide whether it is safe to export.
    safe = (
        modal_policy.family != "base"
        and nested_metrics["terminal_fallback_macro_f1"] > baseline_outer["terminal_fallback_macro_f1"]
        and nested_metrics["hybrid_future_history_plus_char20_macro_f1"] > baseline_outer["hybrid_future_history_plus_char20_macro_f1"]
        and fixed_modal["terminal_fallback_macro_f1"] > fixed_baseline["terminal_fallback_macro_f1"]
        and fixed_modal["hybrid_future_history_plus_char20_macro_f1"] > fixed_baseline["hybrid_future_history_plus_char20_macro_f1"]
    )
    report = {
        **preparation,
        "training": {"device": str(device), "lr": args.lr, "weight_decay": args.weight_decay, "batch_size": args.batch_size,
                     "seed": args.seed, "outer_folds": fold_records},
        "baseline": {"outer_char20": baseline_outer, "fixed_char20": fixed_baseline},
        "direct_neural": {
            "outer_terminal_macro_f1": macro(y, outer_probability.argmax(axis=1), terminal),
            "fixed_terminal_macro_f1": macro(fixed_y, fixed_probability.argmax(axis=1), fixed_terminal),
        },
        "nested_selection": {"selected_by_held_fold": selected, "modal_deployment_policy": modal_policy.key(), **nested_metrics},
        "fixed_modal_policy": fixed_modal,
        "best_static_diagnostic_only": static_best,
        "two_split_safe_for_export": safe,
        "seconds": round(time.time() - started, 2),
    }
    np.savez_compressed(out / "probabilities.npz", outer_probability=outer_probability, fixed_probability=fixed_probability,
                        y=y, terminal=terminal, base=base, fixed_base=fixed_base, fixed_y=fixed_y,
                        fixed_terminal=fixed_terminal, char20=char20, fixed_char20=fixed_char20,
                        fold_assignment=fold_assignment)
    (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)

    if args.export and safe:
        del fixed_model
        if device.type == "cuda": torch.cuda.empty_cache()
        print("two_split_safe=true; refitting full portable candidate", flush=True)
        full = fit_model(word, char, length, y, np.arange(len(y), dtype=np.int64), config, args, args.seed + 211, device,
                         teacher=outer_teacher)
        export_portable(full, config, modal_policy, out)
        print(f"portable_export={out / 'portable_candidate'}", flush=True)
    elif args.export:
        print("two_split_safe=false; no portable weights exported", flush=True)


if __name__ == "__main__":
    main()
