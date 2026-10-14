#!/usr/bin/env python3
"""Leakage-safe action-history GRU fusion experiment.

This deliberately differs from the existing hashed-history XGBoost selector:
it learns an ordered representation of the observed tool actions, result
statuses, and argument-key masks with a bidirectional GRU.  That representation
is fused with fixed out-of-fold E/D/B margins and frozen E5 context embeddings.

All output stays in ``experiments/modeling_ai_sequence_fusion``.  The script
never reads or writes the live ``model/`` directory, so its probabilities can
be audited and blended only after validation.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import GroupKFold
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from train_modeling_b import ALL_CLASSES


ACTION_TO_ID = {name: index + 1 for index, name in enumerate(ALL_CLASSES)}
STATUS_TO_ID = {"pad": 0, "ok": 1, "fail": 2, "other": 3}
ARG_KEYS = ("path", "scope", "pattern", "target", "cmd", "query", "n_files")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--out-dir", default="experiments/modeling_ai_sequence_fusion")
    parser.add_argument("--base-oof", default="experiments/modeling_n_refit/oof_stack_features.npy")
    parser.add_argument("--e5-embeddings", default="experiments/modeling_f/e5_embeddings_len128.npy")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=28)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-e5", action="store_true", help="Ablation: omit frozen E5 context embeddings.")
    parser.add_argument("--teacher-oof", default=None,
                        help="Optional N x 14 teacher probabilities. Used only as group-OOF training targets, never at inference.")
    parser.add_argument("--distill", type=float, default=0.0,
                        help="Teacher KL-loss coefficient in [0, 1); zero preserves the original hard-label run.")
    return parser.parse_args()


def result_status(value: object) -> int:
    text = str(value or "").lower()
    if any(token in text for token in ("fail", "error", "exception", "traceback", "not found", "0 match", "0 file")):
        return STATUS_TO_ID["fail"]
    if any(token in text for token in ("pass", "ok;", "exit=0", "modified", "patched", "matched", "found", "listed")):
        return STATUS_TO_ID["ok"]
    return STATUS_TO_ID["other"]


def meta_features(sample: dict) -> list[float]:
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


def sequence_tensors(samples: list[dict], max_len: int = 6) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    actions = np.zeros((len(samples), max_len), dtype=np.int64)
    statuses = np.zeros((len(samples), max_len), dtype=np.int64)
    args = np.zeros((len(samples), max_len, len(ARG_KEYS)), dtype=np.float32)
    metas = np.zeros((len(samples), len(meta_features(samples[0]))), dtype=np.float32)
    for row, sample in enumerate(samples):
        history = sample.get("history") if isinstance(sample.get("history"), list) else []
        observed = [item for item in history if isinstance(item, dict) and item.get("role") == "assistant_action"][-max_len:]
        # Right padding preserves chronological order and makes GRU's final
        # state correspond to the latest known action.
        for col, item in enumerate(observed):
            name = str(item.get("name") or item.get("action") or "").lower()
            actions[row, col] = ACTION_TO_ID.get(name, 0)
            statuses[row, col] = result_status(item.get("result_summary"))
            values = item.get("args") if isinstance(item.get("args"), dict) else {}
            args[row, col] = [float(key in values) for key in ARG_KEYS]
        metas[row] = np.asarray(meta_features(sample), dtype=np.float32)
    return actions, statuses, args, metas


class SequenceFusion(nn.Module):
    def __init__(self, base_dim: int, e5_dim: int, meta_dim: int) -> None:
        super().__init__()
        self.action_embedding = nn.Embedding(len(ACTION_TO_ID) + 1, 32, padding_idx=0)
        self.status_embedding = nn.Embedding(len(STATUS_TO_ID), 8, padding_idx=0)
        self.position_embedding = nn.Embedding(6, 8)
        self.arg_projection = nn.Sequential(nn.Linear(len(ARG_KEYS), 16), nn.GELU())
        self.gru = nn.GRU(64, 72, batch_first=True, bidirectional=True)
        self.base = nn.Sequential(nn.LayerNorm(base_dim), nn.Linear(base_dim, 128), nn.GELU(), nn.Dropout(0.08))
        self.e5 = nn.Sequential(nn.LayerNorm(e5_dim), nn.Linear(e5_dim, 96), nn.GELU(), nn.Dropout(0.10))
        self.meta = nn.Sequential(nn.Linear(meta_dim, 32), nn.GELU())
        self.head = nn.Sequential(
            nn.Linear(144 + 128 + 96 + 32, 240), nn.GELU(), nn.Dropout(0.14),
            nn.Linear(240, len(ALL_CLASSES)),
        )

    def forward(self, action: torch.Tensor, status: torch.Tensor, arg: torch.Tensor,
                base: torch.Tensor, e5: torch.Tensor, meta: torch.Tensor) -> torch.Tensor:
        positions = torch.arange(action.shape[1], device=action.device).unsqueeze(0)
        sequence = torch.cat((
            self.action_embedding(action), self.status_embedding(status),
            self.position_embedding(positions).expand(action.shape[0], -1, -1), self.arg_projection(arg),
        ), dim=-1)
        sequence, _ = self.gru(sequence)
        # Inputs use right padding: select the last observed action. Empty
        # histories use the zero-position GRU state, which is learned via its
        # accompanying base/meta features.
        lengths = action.ne(0).sum(dim=1).clamp_min(1) - 1
        history = sequence[torch.arange(len(sequence), device=action.device), lengths]
        return self.head(torch.cat((history, self.base(base), self.e5(e5), self.meta(meta)), dim=1))


def macro(y: np.ndarray, probability: np.ndarray) -> float:
    return f1_score(y, probability.argmax(axis=1), labels=np.arange(len(ALL_CLASSES)), average="macro", zero_division=0)


@torch.inference_mode()
def predict(model: nn.Module, loader: DataLoader, device: torch.device) -> np.ndarray:
    model.eval(); parts = []
    for batch in loader:
        batch = [value.to(device, non_blocking=True) for value in batch]
        parts.append(torch.softmax(model(*batch), dim=1).float().cpu().numpy())
    return np.concatenate(parts)


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("This GPU fusion experiment requires CUDA.")
    if not 0.0 <= args.distill < 1.0:
        raise ValueError("--distill must be in [0, 1)")
    if args.distill > 0.0 and not args.teacher_oof:
        raise ValueError("--distill requires --teacher-oof")
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    samples = [json.loads(line) for line in open(Path(args.data_dir) / "train.jsonl", encoding="utf-8") if line.strip()]
    label_map = {row["id"]: row["action"] for row in csv.DictReader(open(Path(args.data_dir) / "train_labels.csv", encoding="utf-8"))}
    label_to_id = {name: index for index, name in enumerate(ALL_CLASSES)}
    y = np.asarray([label_to_id[label_map[sample["id"]]] for sample in samples], dtype=np.int64)
    groups = np.asarray([sample["id"].split("-step_", 1)[0] for sample in samples])
    base = np.load(args.base_oof).astype(np.float32)
    if len(base) != len(samples):
        raise ValueError("Base OOF rows do not align with train.jsonl")
    if args.no_e5:
        e5 = np.zeros((len(samples), 1), dtype=np.float32)
    else:
        e5 = np.load(args.e5_embeddings).astype(np.float32)
        if len(e5) != len(samples):
            raise ValueError("E5 embedding rows do not align with train.jsonl")
    teacher: np.ndarray | None = None
    if args.teacher_oof:
        teacher = np.load(args.teacher_oof).astype(np.float32)
        if teacher.shape != (len(samples), len(ALL_CLASSES)):
            raise ValueError(f"Teacher OOF must be {(len(samples), len(ALL_CLASSES))}, got {teacher.shape}")
    action, status, arg, meta = sequence_tensors(samples)
    tensors = [torch.from_numpy(value) for value in (action, status, arg, base, e5, meta)]
    folds = list(GroupKFold(n_splits=args.folds).split(base, y, groups))
    oof = np.zeros((len(samples), len(ALL_CLASSES)), dtype=np.float32)
    records: list[dict] = []
    device = torch.device("cuda")
    for fold, (train_index, valid_index) in enumerate(folds, start=1):
        started = time.time()
        model = SequenceFusion(base.shape[1], e5.shape[1], meta.shape[1]).to(device)
        counts = np.bincount(y[train_index], minlength=len(ALL_CLASSES))
        weight = torch.tensor(np.sqrt(len(train_index) / (len(ALL_CLASSES) * counts)), device=device, dtype=torch.float32)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1.8e-3, weight_decay=1.0e-4)
        if teacher is None:
            train_data = TensorDataset(*(item[train_index] for item in tensors), torch.from_numpy(y[train_index]))
        else:
            train_data = TensorDataset(
                *(item[train_index] for item in tensors), torch.from_numpy(y[train_index]), torch.from_numpy(teacher[train_index]),
            )
        valid_data = TensorDataset(*(item[valid_index] for item in tensors))
        generator = torch.Generator().manual_seed(args.seed + fold)
        train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, generator=generator, pin_memory=True)
        valid_loader = DataLoader(valid_data, batch_size=1024, shuffle=False, pin_memory=True)
        best_score = -1.0; best_probability = None; stale = 0
        for epoch in range(1, args.epochs + 1):
            model.train()
            for batch in train_loader:
                values = [value.to(device, non_blocking=True) for value in batch]
                if teacher is None:
                    *features, target = values
                    soft_target = None
                else:
                    *features, target, soft_target = values
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    logits = model(*features)
                    hard_loss = nn.functional.cross_entropy(logits, target, weight=weight, label_smoothing=0.015)
                    if soft_target is None:
                        loss = hard_loss
                    else:
                        teacher_loss = nn.functional.kl_div(torch.log_softmax(logits, dim=1), soft_target, reduction="batchmean")
                        loss = (1.0 - args.distill) * hard_loss + args.distill * teacher_loss
                loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0); optimizer.step()
            probability = predict(model, valid_loader, device)
            score = macro(y[valid_index], probability)
            print(f"fold={fold} epoch={epoch} macro_f1={score:.6f}", flush=True)
            if score > best_score:
                best_score = score; best_probability = probability; stale = 0
            else:
                stale += 1
                if stale >= args.patience:
                    break
        assert best_probability is not None
        oof[valid_index] = best_probability
        records.append({"fold": fold, "macro_f1": best_score, "epochs": epoch, "seconds": round(time.time() - started, 2)})
        print(f"fold={fold} BEST={best_score:.6f} seconds={time.time()-started:.1f}", flush=True)
        del model; torch.cuda.empty_cache()
    score = macro(y, oof)
    np.save(out / "oof_probabilities.npy", oof)
    np.save(out / "oof_targets.npy", y)
    (out / "results.json").write_text(json.dumps({
        "macro_f1": score, "folds": records, "architecture": "biGRU(action,status,args)+OOF E/D/B margins+frozen E5+metadata",
        "base_shape": list(base.shape), "e5_shape": list(e5.shape), "feature_leakage": "base margins are existing group-OOF rows; all other inputs are inference-safe",
        "distillation": {
            "enabled": teacher is not None and args.distill > 0.0,
            "coefficient": args.distill,
            "teacher_oof": args.teacher_oof,
            "teacher_usage": "training-only group-OOF soft targets; no teacher values are provided to the student at inference",
        },
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (out / "classification_report.txt").write_text(classification_report(
        y, oof.argmax(axis=1), labels=np.arange(len(ALL_CLASSES)), target_names=ALL_CLASSES, digits=4, zero_division=0,
    ), encoding="utf-8")
    print(f"DONE sequence_fusion_macro_f1={score:.6f}", flush=True)


if __name__ == "__main__":
    main()
