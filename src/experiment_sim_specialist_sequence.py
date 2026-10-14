#!/usr/bin/env python3
"""Strict audit of a small SIM-only portable sequence specialist.

This experiment answers one deliberately narrow question: can a compact
hard-label-only sequence model add signal for ``sess_sim_*`` rows beyond the
current strict-distilled global no-E5 GRU?  It never modifies ``model/``,
``script.py``, or a ZIP file.

Protocol
--------
* The specialist sees exactly the portable action/status/argument/base/meta
  inputs used by the global no-E5 GRU.  It has no E5 input and never consumes
  teacher probabilities.
* Whole-session GroupKFold OOF: a specialist is fit only on SIM rows from the
  four outer-training folds and predicts SIM rows in the untouched fold.
* Common fixed split: it is fit only on common-train SIM sessions and predicts
  common-validation SIM sessions.  Epoch count is fixed before validation
  results are read; there is no early stopping against held labels.
* AU rows retain the existing strict global GRU exactly.  For SIM rows the
  pre-registered, variance-reducing route is a 50/50 geometric mixture of the
  global and specialist GRU.  Raw XGB then uses the current alpha=.25/.85
  source geometry and char20 remains the final terminal-only override.

The base-margin source is the same established GroupKFold OOF block used by
the current global GRU validation artifacts.  It is not fit or altered here.
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
from scipy.special import erf
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from train_modeling_ai_sequence_fusion import ALL_CLASSES, sequence_tensors


ROOT = Path(__file__).resolve().parents[1]
N_CLASSES = len(ALL_CLASSES)
LABEL_TO_ID = {label: number for number, label in enumerate(ALL_CLASSES)}
SIM_GLOBAL_MIX = 0.50  # pre-registered before this experiment is trained
SIM_XGB_ALPHA = 0.25
AU_XGB_ALPHA = 0.85


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="experiments/sim_specialist_sequence_v1")
    parser.add_argument("--stage", choices=("full", "fixed", "both"), default="both")
    parser.add_argument("--epochs", type=int, default=12,
                        help="Fixed before held-out labels are inspected; do not use early stopping.")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=314159)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def source_of(sample_id: str) -> str:
    return "au" if str(sample_id).startswith("sess_au_") else "sim"


def terminal_mask(ids: np.ndarray) -> np.ndarray:
    maximum: dict[str, int] = {}
    for sample_id in ids.astype(str):
        session, step = sample_id.rsplit("-step_", 1)
        maximum[session] = max(maximum.get(session, -1), int(step))
    return np.asarray([
        int(sample_id.rsplit("-step_", 1)[1]) == maximum[sample_id.rsplit("-step_", 1)[0]]
        for sample_id in ids.astype(str)
    ], dtype=bool)


def macro(target: np.ndarray, probability: np.ndarray, mask: np.ndarray | None = None) -> float:
    if mask is None:
        mask = np.ones(len(target), dtype=bool)
    return float(f1_score(
        target[mask], probability[mask].argmax(axis=1), labels=np.arange(N_CLASSES),
        average="macro", zero_division=0,
    ))


def geometric(first: np.ndarray, second: np.ndarray, first_weight: np.ndarray | float) -> np.ndarray:
    """Row-wise normalized product-of-experts."""
    weight = np.asarray(first_weight, dtype=np.float32)
    if weight.ndim == 0:
        weight = np.full(len(first), float(weight), dtype=np.float32)
    value = weight[:, None] * np.log(np.maximum(first, 1e-12))
    value += (1.0 - weight[:, None]) * np.log(np.maximum(second, 1e-12))
    value -= value.max(axis=1, keepdims=True)
    value = np.exp(value).astype(np.float32, copy=False)
    return (value / value.sum(axis=1, keepdims=True)).astype(np.float32, copy=False)


def routed_probability(raw_xgb: np.ndarray, global_gru: np.ndarray, specialist: np.ndarray,
                       source: np.ndarray) -> np.ndarray:
    """Keep AU global; blend SIM global/specialist before the current XGB gate."""
    routed = global_gru.copy()
    sim = source == "sim"
    routed[sim] = geometric(global_gru[sim], specialist[sim], SIM_GLOBAL_MIX)
    alpha = np.where(source == "au", AU_XGB_ALPHA, SIM_XGB_ALPHA).astype(np.float32)
    return geometric(raw_xgb, routed, alpha)


def apply_char20(probability: np.ndarray, char_label: np.ndarray) -> np.ndarray:
    value = probability.copy()
    use = char_label >= 0
    if np.any(use):
        value[use] = 0.0
        value[np.flatnonzero(use), char_label[use]] = 1.0
    return value


def metric_block(target: np.ndarray, probability: np.ndarray, terminal: np.ndarray,
                 char_label: np.ndarray) -> dict[str, float]:
    after = apply_char20(probability, char_label)
    return {
        "all_macro_f1_before_char20": macro(target, probability),
        "terminal_macro_f1_before_char20": macro(target, probability, terminal),
        "all_macro_f1_with_char20": macro(target, after),
        "terminal_macro_f1_with_char20": macro(target, after, terminal),
    }


class SimSequenceSpecialist(nn.Module):
    """Small no-E5 action-history model, intentionally simpler than global GRU."""

    def __init__(self, base_dim: int, meta_dim: int) -> None:
        super().__init__()
        self.action_embedding = nn.Embedding(N_CLASSES + 1, 24, padding_idx=0)
        self.status_embedding = nn.Embedding(4, 6, padding_idx=0)
        self.position_embedding = nn.Embedding(6, 6)
        self.arg_projection = nn.Sequential(nn.Linear(7, 12), nn.GELU())
        self.gru = nn.GRU(48, 48, batch_first=True, bidirectional=True)
        self.base_norm = nn.LayerNorm(base_dim)
        self.base_projection = nn.Sequential(nn.Linear(base_dim, 72), nn.GELU())
        self.meta_projection = nn.Sequential(nn.Linear(meta_dim, 24), nn.GELU())
        self.head = nn.Sequential(nn.Linear(96 + 72 + 24, 96), nn.GELU(), nn.Linear(96, N_CLASSES))

    def forward(self, action: torch.Tensor, status: torch.Tensor, arg: torch.Tensor,
                base: torch.Tensor, meta: torch.Tensor) -> torch.Tensor:
        positions = torch.arange(action.shape[1], device=action.device).unsqueeze(0)
        sequence = torch.cat((
            self.action_embedding(action), self.status_embedding(status),
            self.position_embedding(positions).expand(action.shape[0], -1, -1), self.arg_projection(arg),
        ), dim=-1)
        encoded, _ = self.gru(sequence)
        last = action.ne(0).sum(dim=1).clamp_min(1) - 1
        history = encoded[torch.arange(len(encoded), device=action.device), last]
        return self.head(torch.cat((
            history, self.base_projection(self.base_norm(base)), self.meta_projection(meta),
        ), dim=1))


def numpy_gelu(value: np.ndarray) -> np.ndarray:
    return (0.5 * value * (1.0 + erf(value / np.float32(np.sqrt(2.0))))).astype(np.float32, copy=False)


def numpy_linear(value: np.ndarray, weight: np.ndarray, bias: np.ndarray) -> np.ndarray:
    return (value @ weight.T + bias).astype(np.float32, copy=False)


def numpy_layer_norm(value: np.ndarray, weight: np.ndarray, bias: np.ndarray) -> np.ndarray:
    mean = value.mean(axis=1, keepdims=True)
    variance = ((value - mean) ** 2).mean(axis=1, keepdims=True)
    return ((value - mean) / np.sqrt(variance + 1e-5) * weight + bias).astype(np.float32, copy=False)


def numpy_sigmoid(value: np.ndarray) -> np.ndarray:
    return (1.0 / (1.0 + np.exp(-np.clip(value, -60.0, 60.0)))).astype(np.float32, copy=False)


def numpy_gru_direction(sequence: np.ndarray, weight: dict[str, np.ndarray], suffix: str,
                        reverse: bool) -> np.ndarray:
    w_ih = weight[f"gru.weight_ih_l0{suffix}"]
    w_hh = weight[f"gru.weight_hh_l0{suffix}"]
    b_ih = weight[f"gru.bias_ih_l0{suffix}"]
    b_hh = weight[f"gru.bias_hh_l0{suffix}"]
    rows, steps, _ = sequence.shape
    hidden = np.zeros((rows, w_hh.shape[1]), dtype=np.float32)
    output = np.zeros((rows, steps, w_hh.shape[1]), dtype=np.float32)
    for position in (range(steps - 1, -1, -1) if reverse else range(steps)):
        input_gate = numpy_linear(sequence[:, position], w_ih, b_ih)
        hidden_gate = numpy_linear(hidden, w_hh, b_hh)
        ir, iz, inn = np.split(input_gate, 3, axis=1)
        hr, hz, hnn = np.split(hidden_gate, 3, axis=1)
        reset = numpy_sigmoid(ir + hr)
        update = numpy_sigmoid(iz + hz)
        candidate = np.tanh(inn + reset * hnn).astype(np.float32, copy=False)
        hidden = ((1.0 - update) * candidate + update * hidden).astype(np.float32, copy=False)
        output[:, position] = hidden
    return output


def numpy_specialist_probability(action: np.ndarray, status: np.ndarray, arg: np.ndarray,
                                 base: np.ndarray, meta: np.ndarray,
                                 weight: dict[str, np.ndarray]) -> np.ndarray:
    """Portable NumPy contract used only to verify a promoted specialist export."""
    action = np.asarray(action, dtype=np.int64)
    status = np.asarray(status, dtype=np.int64)
    arg = np.asarray(arg, dtype=np.float32)
    base = np.asarray(base, dtype=np.float32)
    meta = np.asarray(meta, dtype=np.float32)
    positions = weight["position_embedding.weight"][np.arange(action.shape[1])][None, :, :]
    sequence = np.concatenate((
        weight["action_embedding.weight"][action], weight["status_embedding.weight"][status],
        np.broadcast_to(positions, (len(action),) + positions.shape[1:]),
        numpy_gelu(numpy_linear(arg, weight["arg_projection.0.weight"], weight["arg_projection.0.bias"])),
    ), axis=2).astype(np.float32, copy=False)
    forward = numpy_gru_direction(sequence, weight, "", reverse=False)
    backward = numpy_gru_direction(sequence, weight, "_reverse", reverse=True)
    encoded = np.concatenate((forward, backward), axis=2)
    last = np.clip((action != 0).sum(axis=1), 1, None) - 1
    history = encoded[np.arange(len(action)), last]
    base_block = numpy_gelu(numpy_linear(
        numpy_layer_norm(base, weight["base_norm.weight"], weight["base_norm.bias"]),
        weight["base_projection.0.weight"], weight["base_projection.0.bias"],
    ))
    meta_block = numpy_gelu(numpy_linear(
        meta, weight["meta_projection.0.weight"], weight["meta_projection.0.bias"],
    ))
    hidden = numpy_gelu(numpy_linear(
        np.concatenate((history, base_block, meta_block), axis=1),
        weight["head.0.weight"], weight["head.0.bias"],
    ))
    logits = numpy_linear(hidden, weight["head.2.weight"], weight["head.2.bias"])
    logits -= logits.max(axis=1, keepdims=True)
    value = np.exp(logits).astype(np.float32, copy=False)
    return (value / value.sum(axis=1, keepdims=True)).astype(np.float32, copy=False)


@torch.inference_mode()
def predict(model: nn.Module, tensors: list[torch.Tensor], index: np.ndarray,
            batch_size: int, device: torch.device) -> np.ndarray:
    loader = DataLoader(TensorDataset(*(item[index] for item in tensors)), batch_size=batch_size,
                        shuffle=False, pin_memory=True)
    pieces: list[np.ndarray] = []
    model.eval()
    for batch in loader:
        values = [value.to(device, non_blocking=True) for value in batch]
        pieces.append(torch.softmax(model(*values), dim=1).float().cpu().numpy())
    return np.concatenate(pieces).astype(np.float32, copy=False)


def fit_and_predict(tensors: list[torch.Tensor], target: np.ndarray, train_index: np.ndarray,
                    predict_index: np.ndarray, args: argparse.Namespace, fold_seed: int,
                    device: torch.device) -> tuple[np.ndarray, dict[str, object], SimSequenceSpecialist]:
    """Fit only supplied rows.  No held labels are used after this call begins."""
    if not len(train_index) or not len(predict_index):
        raise ValueError("SIM train/predict partitions must both be non-empty")
    counts = np.bincount(target[train_index], minlength=N_CLASSES)
    if np.any(counts == 0):
        missing = [ALL_CLASSES[number] for number in np.flatnonzero(counts == 0)]
        raise ValueError(f"SIM-only train split is missing classes: {missing}")
    seed_everything(fold_seed)
    model = SimSequenceSpecialist(tensors[3].shape[1], tensors[4].shape[1]).to(device)
    weight = torch.tensor(np.sqrt(len(train_index) / (N_CLASSES * counts)), dtype=torch.float32, device=device)
    dataset = TensorDataset(*(item[train_index] for item in tensors), torch.from_numpy(target[train_index]))
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                        generator=torch.Generator().manual_seed(fold_seed), pin_memory=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.8e-3, weight_decay=1.0e-4)
    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum = 0.0
        seen = 0
        for batch in loader:
            *features, hard = [value.to(device, non_blocking=True) for value in batch]
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model(*features)
                loss = nn.functional.cross_entropy(logits, hard, weight=weight, label_smoothing=0.015)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            loss_sum += float(loss.detach()) * len(hard)
            seen += len(hard)
        print(f"specialist seed={fold_seed} epoch={epoch}/{args.epochs} loss={loss_sum / seen:.6f}", flush=True)
    probability = predict(model, tensors, predict_index, args.batch_size, device)
    details: dict[str, object] = {
        "train_rows": int(len(train_index)), "predict_rows": int(len(predict_index)),
        "epochs": int(args.epochs), "final_train_loss": float(loss_sum / seen),
    }
    return probability, details, model


def load_rows() -> tuple[list[dict], np.ndarray, np.ndarray, np.ndarray]:
    rows = [json.loads(line) for line in open(ROOT / "data/train.jsonl", encoding="utf-8") if line.strip()]
    labels = {row["id"]: row["action"] for row in csv.DictReader(open(ROOT / "data/train_labels.csv", encoding="utf-8"))}
    ids = np.asarray([str(row["id"]) for row in rows])
    target = np.asarray([LABEL_TO_ID[labels[sample_id]] for sample_id in ids], dtype=np.int64)
    groups = np.asarray([sample_id.rsplit("-step_", 1)[0] for sample_id in ids])
    return rows, ids, target, groups


def full_char20() -> np.ndarray:
    value = np.load(ROOT / "experiments/terminal_rule_combinations/oof_candidates.npz", allow_pickle=False)["char20"]
    value = value.astype(np.int16, copy=False)
    if value.shape != (70000,):
        raise ValueError(f"Bad full char20 shape: {value.shape}")
    return value


def fixed_char20(ids: np.ndarray, terminal: np.ndarray) -> np.ndarray:
    saved = np.load(ROOT / "experiments/terminal_rule_combinations/oof_candidates.npz", allow_pickle=False)
    terminal_ids = saved["valid_ids"].astype(str)
    labels = saved["fixed_char20"].astype(np.int16)
    mapping = dict(zip(terminal_ids, labels, strict=True))
    value = np.full(len(ids), -1, dtype=np.int16)
    for row, sample_id in enumerate(ids.astype(str)):
        if terminal[row] and sample_id in mapping:
            value[row] = mapping[sample_id]
    if set(terminal_ids) != set(ids[terminal].astype(str)):
        raise ValueError("Stored fixed char20 terminal ID set does not match current fixed rows")
    return value


def run_full(args: argparse.Namespace, out: Path, rows: list[dict], ids: np.ndarray,
             target: np.ndarray, groups: np.ndarray, device: torch.device) -> dict[str, object]:
    base = np.load(ROOT / "experiments/modeling_n_refit/oof_stack_features.npy", allow_pickle=False).astype(np.float32)
    global_gru = np.load(ROOT / "experiments/modeling_ai_sequence_nested_distill_e16_i4/oof_probabilities.npy",
                         allow_pickle=False).astype(np.float32)
    raw_xgb = np.load(ROOT / "experiments/xgb_robust_ensemble/full_oof_raw_seed42.npy", allow_pickle=False).astype(np.float32)
    expected_fold = np.load(ROOT / "experiments/modeling_ai_sequence_nested_distill_e16_i4/outer_fold_assignment.npy",
                            allow_pickle=False).astype(np.int8)
    if base.shape != (len(rows), 97) or global_gru.shape != (len(rows), N_CLASSES) or raw_xgb.shape != global_gru.shape:
        raise ValueError("Full OOF inputs do not match the established strict baseline shapes")
    oof_target = np.load(ROOT / "experiments/modeling_ai_sequence_nested_distill_e16_i4/oof_targets.npy", allow_pickle=False)
    if not np.array_equal(target, oof_target):
        raise ValueError("Full target order does not match strict global GRU OOF")
    source = np.asarray([source_of(sample_id) for sample_id in ids])
    action, status, arg, meta = sequence_tensors(rows)
    tensors = [torch.from_numpy(value) for value in (action, status, arg, base, meta)]
    specialist = global_gru.copy()  # AU is intentionally unchanged in the routed output.
    assignment = np.empty(len(rows), dtype=np.int8)
    records: list[dict[str, object]] = []
    folds = list(GroupKFold(n_splits=5).split(base, target, groups))
    began = time.time()
    for fold, (train_index, valid_index) in enumerate(folds):
        assignment[valid_index] = fold
        held_groups = set(groups[valid_index].tolist())
        sim_train = train_index[source[train_index] == "sim"]
        sim_valid = valid_index[source[valid_index] == "sim"]
        if held_groups.intersection(groups[sim_train].tolist()):
            raise AssertionError("Held session entered SIM specialist training")
        probability, details, model = fit_and_predict(tensors, target, sim_train, sim_valid, args,
                                                       args.seed + fold, device)
        specialist[sim_valid] = probability
        details.update({"fold": fold + 1, "held_session_overlap": 0,
                        "sim_valid_macro_f1": macro(target[sim_valid], probability)})
        records.append(details)
        del model
        torch.cuda.empty_cache()
        print(f"full fold={fold + 1}/5 sim_valid_macro_f1={details['sim_valid_macro_f1']:.6f}", flush=True)
    if not np.array_equal(assignment, expected_fold):
        raise AssertionError("Specialist outer GroupKFold assignment differs from strict global GRU OOF")
    np.save(out / "specialist_full_oof.npy", specialist)
    baseline = routed_probability(raw_xgb, global_gru, global_gru, source)
    candidate = routed_probability(raw_xgb, global_gru, specialist, source)
    terminal = terminal_mask(ids)
    char = full_char20()
    report: dict[str, object] = {
        "protocol": "5-fold whole-session GroupKFold; SIM specialist train partitions exclude each held session; fixed 12 epochs, hard labels only, no E5 or teacher values.",
        "counts": {"rows": int(len(rows)), "sim_rows": int(np.sum(source == "sim")), "au_rows": int(np.sum(source == "au")),
                   "terminal": int(terminal.sum())},
        "folds": records,
        "baseline_raw_xgb_global_gru_char20": metric_block(target, baseline, terminal, char),
        "candidate_sim_specialist_mix_char20": metric_block(target, candidate, terminal, char),
        "seconds": round(time.time() - began, 2),
    }
    np.save(out / "candidate_full_probability.npy", candidate)
    return report


def run_fixed(args: argparse.Namespace, out: Path, rows: list[dict], ids: np.ndarray,
              target: np.ndarray, groups: np.ndarray, device: torch.device) -> dict[str, object]:
    split = {row["id"]: row["split"] for row in csv.DictReader(open(ROOT / "splits/common_split.csv", encoding="utf-8"))}
    train_index = np.asarray([number for number, sample_id in enumerate(ids) if split[sample_id] == "train"], dtype=np.int64)
    valid_index = np.asarray([number for number, sample_id in enumerate(ids) if split[sample_id] == "val"], dtype=np.int64)
    if set(groups[train_index].tolist()).intersection(groups[valid_index].tolist()):
        raise AssertionError("Common fixed train/validation sessions overlap")
    # Match the existing global GRU fixed artifact's portable base block exactly.
    base_all = np.load(ROOT / "experiments/modeling_n_refit/oof_stack_features.npy", allow_pickle=False).astype(np.float32)
    base = np.concatenate((base_all[train_index], base_all[valid_index]), axis=0)
    rows_local = [rows[number] for number in np.concatenate((train_index, valid_index))]
    ids_local = np.concatenate((ids[train_index], ids[valid_index]))
    target_local = np.concatenate((target[train_index], target[valid_index]))
    source = np.asarray([source_of(sample_id) for sample_id in ids_local])
    n_train = len(train_index)
    local_train = np.arange(n_train, dtype=np.int64)
    local_valid = np.arange(n_train, len(ids_local), dtype=np.int64)
    action, status, arg, meta = sequence_tensors(rows_local)
    tensors = [torch.from_numpy(value) for value in (action, status, arg, base, meta)]
    sim_train = local_train[source[local_train] == "sim"]
    sim_valid = local_valid[source[local_valid] == "sim"]
    probability, details, model = fit_and_predict(tensors, target_local, sim_train, sim_valid, args,
                                                   args.seed + 10_000, device)
    global_gru = np.load(ROOT / "experiments/modeling_ai_sequence_fusion_fixed_strictdistill_w35/valid_probabilities.npy",
                         allow_pickle=False).astype(np.float32)
    specialist = global_gru.copy()
    global_indices = np.load(ROOT / "experiments/modeling_ai_sequence_fusion_fixed_strictdistill_w35/valid_indices.npy",
                             allow_pickle=False).astype(np.int64)
    if not np.array_equal(global_indices, valid_index):
        raise ValueError("Fixed global GRU IDs do not match common validation order")
    specialist[source[local_valid] == "sim"] = probability
    raw_xgb = np.load(ROOT / "experiments/xgb_robust_ensemble/strict_fixed_raw_seed42.npy", allow_pickle=False).astype(np.float32)
    if raw_xgb.shape != specialist.shape:
        raise ValueError("Fixed XGB and GRU probabilities do not align")
    source_valid = source[local_valid]
    baseline = routed_probability(raw_xgb, global_gru, global_gru, source_valid)
    candidate = routed_probability(raw_xgb, global_gru, specialist, source_valid)
    terminal = terminal_mask(ids[valid_index])
    char = fixed_char20(ids[valid_index], terminal)
    details.update({"held_session_overlap": 0, "sim_valid_macro_f1": macro(target_local[sim_valid], probability)})
    report: dict[str, object] = {
        "protocol": "Independent common fixed split; specialist is fit only on common-train SIM sessions for the same fixed epoch count and receives no E5/teacher input.",
        "counts": {"train_rows": int(n_train), "valid_rows": int(len(valid_index)),
                   "sim_train_rows": int(len(sim_train)), "sim_valid_rows": int(len(sim_valid)),
                   "terminal": int(terminal.sum())},
        "fit": details,
        "baseline_raw_xgb_global_gru_char20": metric_block(target[valid_index], baseline, terminal, char),
        "candidate_sim_specialist_mix_char20": metric_block(target[valid_index], candidate, terminal, char),
    }
    np.save(out / "specialist_fixed_probability.npy", specialist)
    np.save(out / "candidate_fixed_probability.npy", candidate)
    del model
    torch.cuda.empty_cache()
    return report


def gate(full: dict[str, object], fixed: dict[str, object]) -> dict[str, object]:
    needed = ("all_macro_f1_with_char20", "terminal_macro_f1_with_char20")
    comparisons: dict[str, float] = {}
    passed = True
    for name, report in (("full", full), ("fixed", fixed)):
        baseline = report["baseline_raw_xgb_global_gru_char20"]  # type: ignore[index]
        candidate = report["candidate_sim_specialist_mix_char20"]  # type: ignore[index]
        for metric in needed:
            delta = float(candidate[metric]) - float(baseline[metric])  # type: ignore[index]
            comparisons[f"{name}_{metric}_delta"] = delta
            passed &= delta > 0.0
    return {
        "promote_for_portable_export": bool(passed),
        "required_metrics": list(needed), "deltas": comparisons,
        "rule": "Export only if the pre-registered SIM .50 global/specialist mixture strictly improves all and terminal macro-F1 after char20 on both session-OOF and independent common fixed validation.",
    }


def export_full_refit(args: argparse.Namespace, out: Path, rows: list[dict], ids: np.ndarray,
                      target: np.ndarray, device: torch.device) -> dict[str, object]:
    """Refit only after the two-split gate and verify the NumPy inference path."""
    export = out / "portable_full_refit"
    export.mkdir(parents=True, exist_ok=True)
    base = np.load(ROOT / "experiments/modeling_n_refit/oof_stack_features.npy", allow_pickle=False).astype(np.float32)
    action, status, arg, meta = sequence_tensors(rows)
    tensors = [torch.from_numpy(value) for value in (action, status, arg, base, meta)]
    source = np.asarray([source_of(sample_id) for sample_id in ids])
    sim_index = np.flatnonzero(source == "sim").astype(np.int64)
    probe = sim_index[:min(256, len(sim_index))]
    _unused, details, model = fit_and_predict(tensors, target, sim_index, probe, args,
                                               args.seed + 20_000, device)
    state = {name: value.detach().cpu().numpy().astype(np.float32) for name, value in model.state_dict().items()}
    model.eval()
    with torch.inference_mode():
        torch_probability = torch.softmax(model(
            tensors[0][probe].to(device), tensors[1][probe].to(device), tensors[2][probe].to(device),
            tensors[3][probe].to(device), tensors[4][probe].to(device),
        ), dim=1).float().cpu().numpy()
    numpy_probability = numpy_specialist_probability(action[probe], status[probe], arg[probe], base[probe], meta[probe], state)
    max_difference = float(np.max(np.abs(torch_probability - numpy_probability)))
    agreement = float(np.mean(torch_probability.argmax(axis=1) == numpy_probability.argmax(axis=1)))
    np.savez_compressed(export / "weights.npz", **state)
    np.savez_compressed(export / "verification_probe.npz", action=action[probe], status=status[probe], arg=arg[probe],
                        base=base[probe], meta=meta[probe], torch_probabilities=torch_probability)
    architecture = {
        "format": "sim_sequence_specialist_numpy_v1",
        "all_classes": ALL_CLASSES,
        "input": {"action_shape": [6], "status_shape": [6], "arg_shape": [6, 7], "base_dim": 97, "meta_dim": 15},
        "architecture": {"action_embedding": [N_CLASSES + 1, 24], "status_embedding": [4, 6],
                           "position_embedding": [6, 6], "arg_projection": [7, 12],
                           "gru": {"input_dim": 48, "hidden_dim": 48, "bidirectional": True, "layers": 1},
                           "base_projection": [97, 72], "meta_projection": [15, 24], "head": [192, 96, N_CLASSES]},
        "activation": "PyTorch GELU exact (erf)", "layer_norm_eps": 1e-5,
        "training": {"scope": "SIM rows only", "epochs": int(args.epochs), "seed": int(args.seed + 20_000),
                     "loss": "sqrt inverse-frequency weighted cross entropy with .015 label smoothing", "teacher": "none", "e5": "none"},
        "inference_route": {"sim": "geometric .50 global no-E5 GRU / .50 SIM specialist before raw-XGB alpha=.25",
                            "au": "global no-E5 GRU unchanged before raw-XGB alpha=.85"},
        "numpy_equivalence": {"probe_rows": int(len(probe)), "max_abs_probability_difference": max_difference,
                               "argmax_agreement": agreement},
    }
    (export / "architecture.json").write_text(json.dumps(architecture, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    del model
    torch.cuda.empty_cache()
    return {
        "path": str(export.relative_to(ROOT)).replace("\\", "/"), "fit": details,
        "numpy_equivalence": architecture["numpy_equivalence"],
        "integration_plan": "Add this isolated NumPy forward path to the submission bundle; invoke it only for SIM rows, geometric-mix its probability .50/.50 with the global portable GRU, retain AU global GRU and apply the existing source XGB geometry and char20 afterward.",
    }


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the compact SIM specialist fit")
    if args.epochs <= 0:
        raise ValueError("--epochs must be positive")
    seed_everything(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    out = ROOT / args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    rows, ids, target, groups = load_rows()
    device = torch.device("cuda")
    report: dict[str, object] = {
        "scope": "Experiment only; no live model, script, or ZIP file is touched.",
        "architecture": {
            "action_embedding": [N_CLASSES + 1, 24], "status_embedding": [4, 6], "position_embedding": [6, 6],
            "arg_projection": [7, 12], "bigru": {"input": 48, "hidden": 48, "bidirectional": True},
            "base_projection": [97, 72], "meta_projection": [15, 24], "head": [192, 96, N_CLASSES],
        },
        "pre_registered_route": {
            "sim_gru": "geometric(global_noE5_GRU=.50, sim_specialist=.50)",
            "au_gru": "global_noE5_GRU unchanged", "raw_xgb_alpha": {"sim": SIM_XGB_ALPHA, "au": AU_XGB_ALPHA},
            "char20": "final terminal-only cross-fitted overlay",
        },
        "safety": "Hard labels only; no teacher/E5 values. Whole-session partitions are asserted disjoint and fixed epochs prevent held-label early stopping.",
    }
    if args.stage in ("full", "both"):
        report["full_group_oof"] = run_full(args, out, rows, ids, target, groups, device)
    if args.stage in ("fixed", "both"):
        report["common_fixed_validation"] = run_fixed(args, out, rows, ids, target, groups, device)
    if args.stage == "both":
        report["promotion_gate"] = gate(report["full_group_oof"], report["common_fixed_validation"])  # type: ignore[arg-type]
        if report["promotion_gate"]["promote_for_portable_export"]:  # type: ignore[index]
            report["portable_export"] = export_full_refit(args, out, rows, ids, target, device)
    (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
