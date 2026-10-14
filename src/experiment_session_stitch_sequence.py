#!/usr/bin/env python3
"""Group-safe long-history fallback experiment.

Every JSONL row only retains the last few assistant actions.  When several
rows from one session arrive in the *same inference batch*, their histories
can be aligned by ``-step_N`` and stitched into a longer prefix.  A later row
also reveals the earlier target action exactly; that deterministic rule stays
outside this model.  This experiment therefore trains only the rows that are
*not* exactly revealed (normally the last visible step of each session).

The model deliberately sees three visibility variants during training:

* ``full``: every supplied row in the session is visible;
* ``partial``: a deterministic half of the other rows is visible;
* ``own``: only the target row's raw history is visible.

This makes the long-prefix signal useful for multi-row batches without
assuming that a public batch always contains a complete session.  All model
selection is GroupKFold by session, and this utility never writes ``model/``
or any submission file.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from train_modeling_ai_sequence_fusion import (
    ACTION_TO_ID,
    ALL_CLASSES,
    ARG_KEYS,
    STATUS_TO_ID,
    meta_features,
    result_status,
)


MAX_LEN = 17  # Train data has steps 1..18; all pre-target actions fit.
N_CLASSES = len(ALL_CLASSES)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--base-oof", default="experiments/modeling_n_refit/oof_stack_features.npy")
    parser.add_argument("--xgb-oof", default="experiments/modeling_ak_live_xgb_proxy/oof_probabilities.npy")
    parser.add_argument("--gru-oof", default="experiments/modeling_ai_sequence_fusion_noe5/oof_probabilities.npy")
    parser.add_argument("--teacher-oof", default=None,
                        help="Optional E5-GRU group-OOF probabilities for training-only distillation.")
    parser.add_argument("--distill", type=float, default=0.0,
                        help="Teacher loss coefficient; 0 disables distillation.")
    parser.add_argument("--out-dir", default="experiments/session_stitch_fusion_v1")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=28)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=384)
    parser.add_argument("--seed", type=int, default=31415)
    return parser.parse_args()


def _assistant_items(sample: dict) -> list[dict]:
    return [
        item for item in (sample.get("history") if isinstance(sample.get("history"), list) else [])
        if isinstance(item, dict) and item.get("role") == "assistant_action"
    ]


def _step(sample_id: str) -> int:
    return int(sample_id.rsplit("-step_", 1)[1])


def _session(sample_id: str) -> str:
    return sample_id.rsplit("-step_", 1)[0]


def _token(item: dict) -> tuple[int, int, np.ndarray]:
    name = str(item.get("name") or item.get("action") or "").lower()
    action = ACTION_TO_ID.get(name, 0)
    status = result_status(item.get("result_summary"))
    values = item.get("args") if isinstance(item.get("args"), dict) else {}
    args = np.asarray([float(key in values) for key in ARG_KEYS], dtype=np.float32)
    return action, status, args


@dataclass
class VariantTensors:
    action: np.ndarray
    status: np.ndarray
    args: np.ndarray
    position: np.ndarray
    stats: np.ndarray


def _make_variant(
    sessions: dict[str, list[tuple[int, int, dict]]],
    samples: list[dict],
    variant: str,
    seed: int,
) -> tuple[VariantTensors, np.ndarray, np.ndarray, np.ndarray]:
    """Build long prefix tensors from a visibility policy.

    ``revealed`` is deliberately calculated from all supplied raw histories,
    not from the subset used as model context.  It exactly mirrors the outer
    deterministic action-recovery gate: any row whose target is present in a
    later visible history needs no statistical fallback.
    """
    count = len(samples)
    action = np.zeros((count, MAX_LEN), dtype=np.int64)
    status = np.zeros((count, MAX_LEN), dtype=np.int64)
    args = np.zeros((count, MAX_LEN, len(ARG_KEYS)), dtype=np.float32)
    position = np.zeros((count, MAX_LEN), dtype=np.int64)
    # own-history length, stitched event count, added older events,
    # supplied session rows, and relative target step.
    stats = np.zeros((count, 5), dtype=np.float32)
    revealed = np.zeros(count, dtype=bool)
    terminal = np.zeros(count, dtype=bool)
    added = np.zeros(count, dtype=np.int16)

    for sid, rows in sessions.items():
        rows = sorted(rows)
        max_step = rows[-1][0]
        full_grid: dict[int, tuple[int, int, np.ndarray]] = {}
        # The histories are unambiguous in the supplied train data.  Retain
        # the first copy if a malformed external batch ever disagrees.
        for current_step, _, sample in rows:
            history = _assistant_items(sample)
            for offset, item in enumerate(history):
                absolute = current_step - len(history) + offset
                full_grid.setdefault(absolute, _token(item))

        # Deterministic partial visibility uses a stable session/row hash,
        # rather than random state, so the feature construction is reproducible
        # during every fold and can be recreated at inference if desired.
        selected: set[int]
        if variant == "full":
            selected = {index for _, index, _ in rows}
        elif variant == "own":
            selected = set()
        elif variant == "partial":
            selected = {
                index for current_step, index, _ in rows
                if ((sum(ord(ch) for ch in sid) + index * 1009 + seed) % 100) < 55
            }
        else:
            raise ValueError(f"Unknown variant: {variant}")

        visible_grid: dict[int, tuple[int, int, np.ndarray]] = {}
        for current_step, index, sample in rows:
            if variant == "own":
                # Filled separately per target below.
                continue
            if index not in selected:
                continue
            history = _assistant_items(sample)
            for offset, item in enumerate(history):
                absolute = current_step - len(history) + offset
                visible_grid.setdefault(absolute, _token(item))

        for current_step, index, sample in rows:
            own_items = _assistant_items(sample)
            own_positions = set(range(current_step - len(own_items), current_step))
            if variant == "own":
                current_grid = {
                    current_step - len(own_items) + offset: _token(item)
                    for offset, item in enumerate(own_items)
                }
            else:
                # The target row itself is always present in an inference
                # batch, even when the randomly thinned context excludes all
                # other session rows.
                current_grid = dict(visible_grid)
                for offset, item in enumerate(own_items):
                    current_grid.setdefault(current_step - len(own_items) + offset, _token(item))

            # Restrict to precedents.  In particular, never let a later row's
            # copy of the *current* target become a learned feature.
            events = [(absolute, value) for absolute, value in current_grid.items()
                      if absolute < current_step and absolute >= current_step - MAX_LEN]
            events.sort(key=lambda pair: pair[0])
            if len(events) > MAX_LEN:
                events = events[-MAX_LEN:]
            for col, (absolute, value) in enumerate(events):
                a, s, ar = value
                action[index, col] = a
                status[index, col] = s
                args[index, col] = ar
                # ``0`` is pad; positions are one-indexed relative to the
                # oldest retained window, preserving gaps in compact form.
                position[index, col] = max(1, absolute - (current_step - MAX_LEN) + 1)
            extra = sum(absolute not in own_positions for absolute, _ in events)
            own_len = len(own_items)
            stats[index] = np.asarray([
                min(own_len, MAX_LEN) / MAX_LEN,
                len(events) / MAX_LEN,
                min(extra, MAX_LEN) / MAX_LEN,
                min(len(rows), 20) / 20.0,
                min(current_step, 20) / 20.0,
            ], dtype=np.float32)
            added[index] = extra
            revealed[index] = current_step in full_grid
            terminal[index] = current_step == max_step
    return VariantTensors(action, status, args, position, stats), revealed, terminal, added


class StitchFusion(nn.Module):
    def __init__(self, base_dim: int, meta_dim: int) -> None:
        super().__init__()
        self.action_embedding = nn.Embedding(len(ACTION_TO_ID) + 1, 32, padding_idx=0)
        self.status_embedding = nn.Embedding(len(STATUS_TO_ID), 8, padding_idx=0)
        self.position_embedding = nn.Embedding(MAX_LEN + 1, 8, padding_idx=0)
        self.arg_projection = nn.Sequential(nn.Linear(len(ARG_KEYS), 16), nn.GELU())
        self.gru = nn.GRU(64, 72, batch_first=True, bidirectional=True)
        self.base = nn.Sequential(nn.LayerNorm(base_dim), nn.Linear(base_dim, 128), nn.GELU(), nn.Dropout(0.08))
        # Current XGB and no-E5 GRU are both raw group-OOF probability blocks.
        self.ensemble = nn.Sequential(nn.LayerNorm(2 * N_CLASSES), nn.Linear(2 * N_CLASSES, 64), nn.GELU(), nn.Dropout(0.05))
        self.meta = nn.Sequential(nn.Linear(meta_dim, 32), nn.GELU())
        self.stitch_stats = nn.Sequential(nn.Linear(5, 24), nn.GELU())
        self.head = nn.Sequential(
            nn.Linear(144 + 128 + 64 + 32 + 24, 256), nn.GELU(), nn.Dropout(0.14),
            nn.Linear(256, N_CLASSES),
        )

    def forward(
        self,
        action: torch.Tensor,
        status: torch.Tensor,
        arg: torch.Tensor,
        position: torch.Tensor,
        stitch_stats: torch.Tensor,
        base: torch.Tensor,
        ensemble: torch.Tensor,
        meta: torch.Tensor,
    ) -> torch.Tensor:
        sequence = torch.cat((
            self.action_embedding(action), self.status_embedding(status),
            self.position_embedding(position), self.arg_projection(arg),
        ), dim=-1)
        encoded, _ = self.gru(sequence)
        last = action.ne(0).sum(dim=1).clamp_min(1) - 1
        history = encoded[torch.arange(len(encoded), device=encoded.device), last]
        return self.head(torch.cat((
            history, self.base(base), self.ensemble(ensemble), self.meta(meta), self.stitch_stats(stitch_stats),
        ), dim=1))


def macro(target: np.ndarray, probability: np.ndarray, mask: np.ndarray | None = None) -> float:
    if mask is None:
        mask = np.ones(len(target), dtype=bool)
    return float(f1_score(target[mask], probability[mask].argmax(axis=1), labels=np.arange(N_CLASSES),
                          average="macro", zero_division=0))


@torch.inference_mode()
def predict(model: nn.Module, loader: DataLoader, device: torch.device) -> np.ndarray:
    model.eval()
    chunks: list[np.ndarray] = []
    for batch in loader:
        values = [value.to(device, non_blocking=True) for value in batch]
        chunks.append(torch.softmax(model(*values), dim=1).float().cpu().numpy())
    return np.concatenate(chunks)


def tensorize(
    variant: VariantTensors,
    base: np.ndarray,
    ensemble: np.ndarray,
    meta: np.ndarray,
) -> list[torch.Tensor]:
    return [torch.from_numpy(value) for value in (
        variant.action, variant.status, variant.args, variant.position, variant.stats, base, ensemble, meta,
    )]


def select_weight_nested(
    target: np.ndarray,
    groups: np.ndarray,
    base_probability: np.ndarray,
    stitch_probability: np.ndarray,
    folds: int,
) -> tuple[np.ndarray, list[dict[str, float | int]]]:
    """Select blend weight on other sessions, yielding conservative OOF blend."""
    grid = np.linspace(0.0, 1.0, 21, dtype=np.float32)
    result = np.zeros_like(base_probability)
    records: list[dict[str, float | int]] = []
    for fold, (train_index, valid_index) in enumerate(GroupKFold(n_splits=folds).split(base_probability, target, groups), start=1):
        scores = [macro(target[train_index], (1.0 - weight) * base_probability[train_index] + weight * stitch_probability[train_index])
                  for weight in grid]
        best = int(np.argmax(scores))
        weight = float(grid[best])
        result[valid_index] = (1.0 - weight) * base_probability[valid_index] + weight * stitch_probability[valid_index]
        records.append({"fold": fold, "stitch_weight": weight, "train_macro": float(scores[best]),
                        "valid_macro": macro(target[valid_index], result[valid_index])})
    return result, records


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this sequence experiment")
    if not 0.0 <= args.distill < 1.0:
        raise ValueError("--distill must be in [0, 1)")
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)

    samples = [json.loads(line) for line in open(Path(args.data_dir) / "train.jsonl", encoding="utf-8") if line.strip()]
    label_map = {row["id"]: row["action"] for row in csv.DictReader(open(Path(args.data_dir) / "train_labels.csv", encoding="utf-8"))}
    label_to_id = {label: index for index, label in enumerate(ALL_CLASSES)}
    target = np.asarray([label_to_id[label_map[sample["id"]]] for sample in samples], dtype=np.int64)
    groups = np.asarray([_session(str(sample["id"])) for sample in samples])
    sessions: dict[str, list[tuple[int, int, dict]]] = defaultdict(list)
    for index, sample in enumerate(samples):
        sessions[_session(str(sample["id"]))].append((_step(str(sample["id"])), index, sample))

    base = np.load(args.base_oof).astype(np.float32)
    xgb = np.load(args.xgb_oof).astype(np.float32)
    gru = np.load(args.gru_oof).astype(np.float32)
    if any(len(value) != len(samples) for value in (base, xgb, gru)):
        raise ValueError("OOF inputs must align to train.jsonl")
    if xgb.shape[1] != N_CLASSES or gru.shape[1] != N_CLASSES:
        raise ValueError("XGB/GRU OOF inputs must each have 14 class probabilities")
    teacher = None
    if args.teacher_oof:
        teacher = np.load(args.teacher_oof).astype(np.float32)
        if teacher.shape != xgb.shape:
            raise ValueError("Teacher OOF must be N x 14 and aligned to train.jsonl")
    meta = np.asarray([meta_features(sample) for sample in samples], dtype=np.float32)
    ensemble = np.column_stack((xgb, gru)).astype(np.float32)
    variants: dict[str, VariantTensors] = {}
    reports: dict[str, dict[str, float | int]] = {}
    revealed = terminal = added = None
    for name in ("full", "partial", "own"):
        values, current_revealed, current_terminal, current_added = _make_variant(sessions, samples, name, args.seed)
        variants[name] = values
        if revealed is None:
            revealed, terminal, added = current_revealed, current_terminal, current_added
        else:
            if not np.array_equal(revealed, current_revealed) or not np.array_equal(terminal, current_terminal):
                raise AssertionError("Visibility-invariant masks changed")
        reports[name] = {
            "mean_events": float((values.action != 0).sum(axis=1).mean()),
            "mean_added_actions": float(current_added.mean()),
        }
    assert revealed is not None and terminal is not None and added is not None
    fallback = ~revealed
    fallback_index = np.flatnonzero(fallback)
    fallback_target = target[fallback_index]
    fallback_groups = groups[fallback_index]
    if len(np.unique(fallback_groups)) < args.folds:
        raise ValueError("Not enough sessions for requested GroupKFold")
    print(json.dumps({
        "rows": len(samples), "fallback_rows": int(fallback.sum()), "terminal_rows": int(terminal.sum()),
        "nonterminal_unrevealed": int((fallback & ~terminal).sum()),
        "full_added_actions_terminal_mean": float(added[terminal].mean()), "variants": reports,
    }, ensure_ascii=False), flush=True)

    # Each training fallback row is observed in all three availability regimes;
    # duplicated rows retain their session group and never cross outer folds.
    tensors = {name: tensorize(values, base, ensemble, meta) for name, values in variants.items()}
    oof: dict[str, np.ndarray] = {name: np.zeros((len(fallback_index), N_CLASSES), dtype=np.float32)
                                  for name in variants}
    device = torch.device("cuda")
    fold_records: list[dict[str, float | int]] = []
    outer = list(GroupKFold(n_splits=args.folds).split(fallback_index, fallback_target, fallback_groups))
    for fold, (train_local, valid_local) in enumerate(outer, start=1):
        train_global = fallback_index[train_local]
        valid_global = fallback_index[valid_local]
        # Full, partial and singleton-like sequence variants teach the same
        # terminal action, while base/meta blocks remain unchanged.
        train_parts = []
        for name in ("full", "partial", "own"):
            train_parts.append([tensor[train_global] for tensor in tensors[name]])
        feature_train = [torch.cat([parts[column] for parts in train_parts], dim=0)
                         for column in range(len(train_parts[0]))]
        train_target = np.tile(target[train_global], 3)
        if teacher is not None:
            teacher_target = np.tile(teacher[train_global], (3, 1)).astype(np.float32)
            train_data = TensorDataset(*feature_train, torch.from_numpy(train_target), torch.from_numpy(teacher_target))
        else:
            train_data = TensorDataset(*feature_train, torch.from_numpy(train_target))
        train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True,
                                  generator=torch.Generator().manual_seed(args.seed + fold), pin_memory=True)
        valid_loaders = {
            name: DataLoader(TensorDataset(*[tensor[valid_global] for tensor in tensors[name]]),
                             batch_size=1024, shuffle=False, pin_memory=True)
            for name in variants
        }
        model = StitchFusion(base.shape[1], meta.shape[1]).to(device)
        counts = np.bincount(target[train_global], minlength=N_CLASSES)
        class_weight = torch.tensor(np.sqrt(len(train_global) / (N_CLASSES * counts)), dtype=torch.float32, device=device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1.6e-3, weight_decay=1.0e-4)
        best_score = -1.0; best_probability: dict[str, np.ndarray] | None = None; stale = 0
        for epoch in range(1, args.epochs + 1):
            model.train()
            for batch in train_loader:
                *features, hard = [value.to(device, non_blocking=True) for value in batch]
                soft = None
                if teacher is not None:
                    soft = hard
                    hard = features.pop()
                    # Layout is features..., hard-label, teacher-probability.
                    # The first pop above transfers the soft target here.
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    logits = model(*features)
                    hard_loss = nn.functional.cross_entropy(logits, hard, weight=class_weight, label_smoothing=0.015)
                    if teacher is not None:
                        assert soft is not None
                        # Teacher values are fixed group-OOF probabilities and
                        # are training-only targets, never model inputs.
                        kd_loss = nn.functional.kl_div(torch.log_softmax(logits, dim=1), soft, reduction="batchmean")
                        loss = (1.0 - args.distill) * hard_loss + args.distill * kd_loss
                    else:
                        loss = hard_loss
                loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0); optimizer.step()
            current = {name: predict(model, loader, device) for name, loader in valid_loaders.items()}
            # Full-context fallback is deployment's optimistic regime.  Its
            # early stop objective is balanced with own-only safety.
            score = 0.75 * macro(target[valid_global], current["full"]) + 0.25 * macro(target[valid_global], current["own"])
            print(f"fold={fold} epoch={epoch} full={macro(target[valid_global], current['full']):.6f} own={macro(target[valid_global], current['own']):.6f} select={score:.6f}", flush=True)
            if score > best_score:
                best_score = score; best_probability = current; stale = 0
            else:
                stale += 1
                if stale >= args.patience:
                    break
        assert best_probability is not None
        for name in oof:
            oof[name][valid_local] = best_probability[name]
        fold_records.append({"fold": fold, "best_select": float(best_score), "epochs": epoch,
                             "full_macro": macro(target[valid_global], best_probability["full"]),
                             "own_macro": macro(target[valid_global], best_probability["own"])})
        del model; torch.cuda.empty_cache()

    base_blend = 0.5 * xgb[fallback_index] + 0.5 * gru[fallback_index]
    result: dict[str, object] = {
        "method": "5-fold GroupKFold only on deterministic-fallback rows; three same-session visibility variants in every training fold",
        "fallback_rows": int(len(fallback_index)), "terminal_rows_within_fallback": int(terminal[fallback_index].sum()),
        "base_raw_xgb_gru_fallback_macro": macro(fallback_target, base_blend),
        "base_raw_xgb_gru_terminal_macro": macro(fallback_target, base_blend, terminal[fallback_index]),
        "folds": fold_records,
        "visibility": reports,
        "distillation": {"enabled": teacher is not None, "coefficient": args.distill,
                           "teacher": args.teacher_oof if teacher is not None else None},
        "variants": {},
    }
    np.savez_compressed(out / "oof.npz", ids=np.asarray([samples[index]["id"] for index in fallback_index]),
                        target=fallback_target, groups=fallback_groups, terminal=terminal[fallback_index],
                        added=added[fallback_index], base_blend=base_blend, **{f"stitch_{name}": value for name, value in oof.items()})
    for name, probability in oof.items():
        nested, nested_records = select_weight_nested(fallback_target, fallback_groups, base_blend, probability, args.folds)
        added_mask = added[fallback_index] > 0
        result["variants"][name] = {
            "stitch_only_fallback_macro": macro(fallback_target, probability),
            "stitch_only_terminal_macro": macro(fallback_target, probability, terminal[fallback_index]),
            "nested_blend_fallback_macro": macro(fallback_target, nested),
            "nested_blend_terminal_macro": macro(fallback_target, nested, terminal[fallback_index]),
            "base_fallback_added_macro": macro(fallback_target, base_blend, added_mask),
            "stitch_fallback_added_macro": macro(fallback_target, probability, added_mask),
            "nested_blend_added_macro": macro(fallback_target, nested, added_mask),
            "n_added_rows": int(added_mask.sum()),
            "nested_weights": nested_records,
        }
        np.save(out / f"nested_blend_{name}.npy", nested)
    (out / "results.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
