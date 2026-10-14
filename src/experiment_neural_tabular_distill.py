#!/usr/bin/env python3
"""Cross-fitted lightweight tabular student for the E5-GRU signal.

This experiment intentionally never writes ``model/``, ``script.py`` or a
submission ZIP.  It uses exactly the 2,032 inference-safe columns of the live
XGBoost selector:

    97 E/D/B OOF margins + 360 state features + 39 routed-source margins
    + 1,536 hashed user/action features.

The E5 GRU's *out-of-fold* probabilities are optional soft targets during
training.  They are not an input to the student, so a selected full-refit
student can later be evaluated with only NumPy matrix operations and the
feature builders already present in ``script.py``.

The reported values are group-OOF by session.  The helper also reports
post-hoc blends with the independently generated live-XGB and E5-GRU OOF
streams; these are diagnostics only, not a promotion command.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import torch
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from data import load_jsonl, load_labels
from train_modeling_e import build_modeling_e_features
from train_modeling_r_source_stack import source_of
from train_modeling_ai_sequence_fusion import sequence_tensors

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from script import ALL_CLASSES_LIST, build_state_dict, build_user_sequence_hash  # noqa: E402


N_CLASSES = len(ALL_CLASSES_LIST)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--q-bundle", default="model_candidate_q/best_model.pkl")
    parser.add_argument("--base-oof", default="experiments/modeling_n_refit/oof_stack_features.npy")
    parser.add_argument("--source-oof", default="experiments/modeling_ak_live_xgb_proxy/source_oof.npy")
    parser.add_argument("--teacher-oof", default="experiments/modeling_ai_sequence_fusion_rerun/oof_probabilities.npy")
    parser.add_argument("--xgb-oof", default="experiments/modeling_ak_live_xgb_proxy/oof_probabilities.npy")
    parser.add_argument("--out-dir", default="experiments/neural_tabular_distill")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=38)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--batch-size", type=int, default=768)
    parser.add_argument("--teacher-weight", type=float, default=0.35,
                        help="KL weight in [0, 1]; 0 produces a purely supervised student.")
    parser.add_argument("--temperature", type=float, default=1.5)
    parser.add_argument("--with-sequence", action="store_true",
                        help="Add the current-row six-step action/status/args GRU. Still no E5 input.")
    parser.add_argument("--seed", type=int, default=420)
    return parser.parse_args()


def macro(y: np.ndarray, probability: np.ndarray, mask: np.ndarray | None = None) -> float:
    if mask is None:
        mask = np.ones(len(y), dtype=bool)
    return float(f1_score(y[mask], probability[mask].argmax(axis=1), labels=np.arange(N_CLASSES),
                          average="macro", zero_division=0))


def terminal_mask(ids: list[str]) -> np.ndarray:
    maximum: dict[str, int] = {}
    for sample_id in ids:
        session, step = sample_id.rsplit("-step_", 1)
        maximum[session] = max(maximum.get(session, -1), int(step))
    return np.asarray([
        int(sample_id.rsplit("-step_", 1)[1]) == maximum[sample_id.rsplit("-step_", 1)[0]]
        for sample_id in ids
    ], dtype=bool)


def build_live_x(samples: list[dict], args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Reconstruct the live selector input without any labels or fitted changes."""
    base = np.load(args.base_oof).astype(np.float32, copy=False)
    source = np.load(args.source_oof).astype(np.float32, copy=False)
    bundle = joblib.load(args.q_bundle)
    state = bundle["state_vectorizer"].transform([build_state_dict(sample) for sample in samples])
    state = (state.toarray() if hasattr(state, "toarray") else np.asarray(state)).astype(np.float32, copy=False)
    user = build_user_sequence_hash(samples)
    x = np.column_stack((base, state, source, user)).astype(np.float32, copy=False)
    if x.shape[1] != 2032:
        raise ValueError(f"Expected live selector width 2032, got {x.shape[1]}")
    return x, base, state, source


class TabularStudent(nn.Module):
    """Small dense model whose only operations are LayerNorm/Linear/GELU.

    Keeping this structure simple is deliberate: all learned parameters can
    be exported directly to an NPZ and evaluated without PyTorch at inference
    time.  The four blocks preserve the semantics of the live XGBoost input
    instead of flattening sparse user hashes into one unstructured projection.
    """

    def __init__(self) -> None:
        super().__init__()
        self.base = nn.Sequential(nn.LayerNorm(97), nn.Linear(97, 128), nn.GELU(), nn.Dropout(0.04))
        self.state = nn.Sequential(nn.LayerNorm(360), nn.Linear(360, 112), nn.GELU(), nn.Dropout(0.05))
        self.source = nn.Sequential(nn.LayerNorm(39), nn.Linear(39, 64), nn.GELU(), nn.Dropout(0.04))
        self.user = nn.Sequential(nn.LayerNorm(1536), nn.Linear(1536, 256), nn.GELU(), nn.Dropout(0.10))
        self.head = nn.Sequential(
            nn.Linear(128 + 112 + 64 + 256, 320), nn.GELU(), nn.Dropout(0.14),
            nn.Linear(320, N_CLASSES),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # FeatureHasher counts are nonnegative and occasionally exceed one;
        # log1p prevents a repeated history token from dominating a dense row.
        return self.head(torch.cat((
            self.base(x[:, :97]),
            self.state(x[:, 97:457]),
            self.source(x[:, 457:496]),
            self.user(torch.log1p(x[:, 496:])),
        ), dim=1))


class SequenceTabularStudent(nn.Module):
    """Tabular student plus an explicit no-E5 action/status/args encoder.

    This is deliberately a small extension of the exportable no-E5 GRU
    architecture already prototyped in ``numpy_sequence_fusion.py``.  It sees
    only the raw history available in the current JSONL record and the 2,032
    existing selector columns.
    """

    def __init__(self) -> None:
        super().__init__()
        self.action_embedding = nn.Embedding(N_CLASSES + 1, 32, padding_idx=0)
        self.status_embedding = nn.Embedding(4, 8, padding_idx=0)
        self.position_embedding = nn.Embedding(6, 8)
        self.arg_projection = nn.Sequential(nn.Linear(7, 16), nn.GELU())
        self.gru = nn.GRU(64, 72, batch_first=True, bidirectional=True)
        self.base = nn.Sequential(nn.LayerNorm(97), nn.Linear(97, 128), nn.GELU(), nn.Dropout(0.04))
        self.state = nn.Sequential(nn.LayerNorm(360), nn.Linear(360, 112), nn.GELU(), nn.Dropout(0.05))
        self.source = nn.Sequential(nn.LayerNorm(39), nn.Linear(39, 64), nn.GELU(), nn.Dropout(0.04))
        self.user = nn.Sequential(nn.LayerNorm(1536), nn.Linear(1536, 256), nn.GELU(), nn.Dropout(0.10))
        self.head = nn.Sequential(
            nn.Linear(144 + 128 + 112 + 64 + 256, 384), nn.GELU(), nn.Dropout(0.15),
            nn.Linear(384, N_CLASSES),
        )

    def forward(self, x: torch.Tensor, action: torch.Tensor, status: torch.Tensor, arg: torch.Tensor) -> torch.Tensor:
        positions = torch.arange(action.shape[1], device=action.device).unsqueeze(0)
        sequence = torch.cat((
            self.action_embedding(action), self.status_embedding(status),
            self.position_embedding(positions).expand(action.shape[0], -1, -1), self.arg_projection(arg),
        ), dim=-1)
        encoded, _ = self.gru(sequence)
        last = action.ne(0).sum(dim=1).clamp_min(1) - 1
        history = encoded[torch.arange(len(encoded), device=action.device), last]
        return self.head(torch.cat((
            history,
            self.base(x[:, :97]),
            self.state(x[:, 97:457]),
            self.source(x[:, 457:496]),
            self.user(torch.log1p(x[:, 496:])),
        ), dim=1))


@torch.inference_mode()
def predict(model: nn.Module, x: torch.Tensor, indices: np.ndarray, device: torch.device,
            action: torch.Tensor | None = None, status: torch.Tensor | None = None,
            arg: torch.Tensor | None = None) -> np.ndarray:
    model.eval()
    result: list[np.ndarray] = []
    for start in range(0, len(indices), 2048):
        part_index = indices[start:start + 2048]
        part = x[part_index].to(device, non_blocking=True)
        if action is None:
            logits = model(part)
        else:
            assert status is not None and arg is not None
            logits = model(
                part,
                action[part_index].to(device, non_blocking=True),
                status[part_index].to(device, non_blocking=True),
                arg[part_index].to(device, non_blocking=True),
            )
        result.append(torch.softmax(logits, dim=1).float().cpu().numpy())
    return np.concatenate(result, axis=0)


def train_fold(
    x: torch.Tensor,
    y: np.ndarray,
    teacher: torch.Tensor,
    train_index: np.ndarray,
    valid_index: np.ndarray,
    args: argparse.Namespace,
    seed: int,
    device: torch.device,
    action: torch.Tensor | None = None,
    status: torch.Tensor | None = None,
    arg: torch.Tensor | None = None,
) -> tuple[np.ndarray, dict[str, float | int]]:
    model: nn.Module = (SequenceTabularStudent() if action is not None else TabularStudent()).to(device)
    counts = np.bincount(y[train_index], minlength=N_CLASSES)
    # Sqrt weighting was more stable than fully balanced cross entropy for the
    # rare action labels in the sequence-fusion experiment.
    weights = torch.tensor(np.sqrt(len(train_index) / (N_CLASSES * counts)), dtype=torch.float32, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.6e-3, weight_decay=1.5e-4)
    if action is None:
        dataset = TensorDataset(x[train_index], torch.from_numpy(y[train_index]), teacher[train_index])
    else:
        assert status is not None and arg is not None
        dataset = TensorDataset(x[train_index], action[train_index], status[train_index], arg[train_index],
                                torch.from_numpy(y[train_index]), teacher[train_index])
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, generator=generator, pin_memory=True)
    target_temp = float(args.temperature)
    best_score = -1.0
    best_probability: np.ndarray | None = None
    stale = 0
    started = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        for batch in loader:
            if action is None:
                bx, by, bt = batch
                action_batch = status_batch = arg_batch = None
            else:
                bx, action_batch, status_batch, arg_batch, by, bt = batch
            bx = bx.to(device, non_blocking=True); by = by.to(device, non_blocking=True); bt = bt.to(device, non_blocking=True)
            if action_batch is not None:
                action_batch = action_batch.to(device, non_blocking=True)
                status_batch = status_batch.to(device, non_blocking=True)
                arg_batch = arg_batch.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model(bx) if action_batch is None else model(bx, action_batch, status_batch, arg_batch)
                supervised = nn.functional.cross_entropy(logits, by, weight=weights, label_smoothing=0.012)
                if args.teacher_weight:
                    # ``bt`` is an independently cross-fitted teacher signal.
                    # Clamp only protects against an accidental zero from a
                    # partial artifact; normal teacher probabilities are dense.
                    target = torch.clamp(bt, min=1e-7)
                    target = target / target.sum(dim=1, keepdim=True)
                    distill = nn.functional.kl_div(
                        nn.functional.log_softmax(logits / target_temp, dim=1),
                        target,
                        reduction="batchmean",
                    ) * (target_temp ** 2)
                    loss = (1.0 - args.teacher_weight) * supervised + args.teacher_weight * distill
                else:
                    loss = supervised
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
        probability = predict(model, x, valid_index, device, action, status, arg)
        score = macro(y[valid_index], probability)
        print(f"fold={seed} epoch={epoch} macro={score:.6f}", flush=True)
        if score > best_score:
            best_score = score; best_probability = probability; stale = 0
        else:
            stale += 1
            if stale >= args.patience:
                break
    assert best_probability is not None
    result = {"macro_f1": best_score, "epochs": epoch, "seconds": round(time.time() - started, 2)}
    del model
    torch.cuda.empty_cache()
    return best_probability, result


def best_blend(y: np.ndarray, xgb: np.ndarray, teacher: np.ndarray, student: np.ndarray, terminal: np.ndarray) -> dict[str, float]:
    """Use a coarse, explicitly disclosed diagnostic grid, not a fitted gate."""
    best: dict[str, float] = {"macro_f1": -1.0}
    # Keep the established XGB/teacher mixture near its independently observed
    # optimum and ask only whether a modest student contribution is additive.
    for xgb_weight in np.arange(0.40, 0.651, 0.025):
        for student_weight in np.arange(0.0, 0.301, 0.025):
            teacher_weight = 1.0 - xgb_weight - student_weight
            if teacher_weight < 0:
                continue
            p = xgb_weight * xgb + teacher_weight * teacher + student_weight * student
            value = macro(y, p)
            if value > best["macro_f1"]:
                best = {
                    "macro_f1": value,
                    "xgb_weight": float(xgb_weight),
                    "teacher_weight": float(teacher_weight),
                    "student_weight": float(student_weight),
                    "terminal_macro_f1": macro(y, p, terminal),
                }
    return best


def main() -> None:
    args = parse_args()
    if not (0.0 <= args.teacher_weight <= 1.0):
        raise ValueError("--teacher-weight must be in [0, 1]")
    if not torch.cuda.is_available():
        raise RuntimeError("GPU is required for this bounded experiment")
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    samples = load_jsonl(str(Path(args.data_dir) / "train.jsonl"))
    labels = load_labels(str(Path(args.data_dir) / "train_labels.csv"))
    ids, _ = build_modeling_e_features(samples)
    label_to_id = {name: index for index, name in enumerate(ALL_CLASSES_LIST)}
    y = np.asarray([label_to_id[labels[sample_id]] for sample_id in ids], dtype=np.int64)
    groups = np.asarray([sample_id.split("-step_", 1)[0] for sample_id in ids])
    source = np.asarray([source_of(sample_id) for sample_id in ids])
    terminal = terminal_mask(ids)
    x_np, _, _, _ = build_live_x(samples, args)
    teacher_np = np.load(args.teacher_oof).astype(np.float32, copy=False)
    xgb = np.load(args.xgb_oof).astype(np.float32, copy=False)
    if teacher_np.shape != (len(y), N_CLASSES) or xgb.shape != teacher_np.shape:
        raise ValueError("Teacher/XGB OOF artifact shape mismatch")

    x = torch.from_numpy(x_np)
    teacher = torch.from_numpy(teacher_np)
    if args.with_sequence:
        action_np, status_np, arg_np, _ = sequence_tensors(samples)
        action = torch.from_numpy(action_np)
        status = torch.from_numpy(status_np)
        arg = torch.from_numpy(arg_np)
    else:
        action = status = arg = None
    folds = list(GroupKFold(n_splits=args.folds).split(np.zeros(len(y)), y, groups))
    probability = np.zeros_like(teacher_np)
    records: list[dict[str, float | int]] = []
    device = torch.device("cuda")
    for fold, (train_index, valid_index) in enumerate(folds, start=1):
        # The fold seed changes optimization noise only; groups remain exactly
        # held out through GroupKFold.
        current, record = train_fold(
            x, y, teacher, train_index, valid_index, args, args.seed + fold, device,
            action=action, status=status, arg=arg,
        )
        probability[valid_index] = current
        record["fold"] = fold
        records.append(record)
        print(f"fold={fold} BEST={record['macro_f1']:.6f} sec={record['seconds']}", flush=True)

    blend = best_blend(y, xgb, teacher_np, probability, terminal)
    summary = {
        "student_macro_f1": macro(y, probability),
        "student_terminal_macro_f1": macro(y, probability, terminal),
        "student_sim_macro_f1": macro(y, probability, source == "sim"),
        "student_au_macro_f1": macro(y, probability, source == "au"),
        "xgb_teacher_equal_macro_f1": macro(y, (xgb + teacher_np) / 2),
        "diagnostic_three_way_best_in_sample_grid": blend,
        "folds": records,
        "shape": list(x_np.shape),
        "with_sequence": args.with_sequence,
        "teacher_weight": args.teacher_weight,
        "validation_method": (
            "5-fold GroupKFold by session. Student inputs are the current live selector's "
            "inference-safe stacked columns; teacher values are independently generated per-row OOF GRU probabilities."
        ),
        "deployment_assessment": (
            "The student does not consume E5 or teacher probabilities at inference. Its LayerNorm/Linear/GELU "
            "operations can be exported to NumPy after a separate full refit; no deployment change is made here."
        ),
    }
    np.save(out / "oof_probabilities.npy", probability)
    np.save(out / "oof_targets.npy", y)
    np.save(out / "terminal_mask.npy", terminal)
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
