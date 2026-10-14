#!/usr/bin/env python3
"""Independent common-split check for the stitched-history fallback.

This consumes the same inference-safe inputs as the 5-fold experiment but
trains only on the fixed common-train sessions.  It exists to reject a gate
that happens to look attractive only when selected over the full OOF table.
No live model or submission artifact is touched.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from experiment_session_stitch_sequence import (
    ALL_CLASSES, N_CLASSES, StitchFusion, _make_variant, _session, _step,
    macro, predict, tensorize,
)
from train_modeling_ai_sequence_fusion import meta_features


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--split", default="splits/common_split.csv")
    parser.add_argument("--base-oof", default="experiments/modeling_n_refit/oof_stack_features.npy")
    parser.add_argument("--xgb-oof", default="experiments/modeling_ak_live_xgb_proxy/oof_probabilities.npy")
    parser.add_argument("--gru-oof", default="experiments/modeling_ai_sequence_fusion_noe5/oof_probabilities.npy")
    parser.add_argument("--teacher-oof", default="experiments/modeling_ai_sequence_fusion_rerun/oof_probabilities.npy")
    parser.add_argument("--distill", type=float, default=0.35)
    parser.add_argument("--out-dir", default="experiments/session_stitch_fusion_fixed_distill035")
    parser.add_argument("--epochs", type=int, default=24)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=31415)
    return parser.parse_args()


def f1(target: np.ndarray, probability: np.ndarray, mask: np.ndarray | None = None) -> float:
    if mask is None:
        mask = np.ones(len(target), dtype=bool)
    return macro(target, probability, mask)


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    samples = [json.loads(line) for line in open(Path(args.data_dir) / "train.jsonl", encoding="utf-8") if line.strip()]
    labels = {row["id"]: row["action"] for row in csv.DictReader(open(Path(args.data_dir) / "train_labels.csv", encoding="utf-8"))}
    split = {row["id"]: row["split"] for row in csv.DictReader(open(args.split, encoding="utf-8"))}
    to_id = {label: index for index, label in enumerate(ALL_CLASSES)}
    target = np.asarray([to_id[labels[sample["id"]]] for sample in samples], dtype=np.int64)
    train = np.asarray([split[sample["id"]] == "train" for sample in samples])
    valid = ~train
    sessions: dict[str, list[tuple[int, int, dict]]] = defaultdict(list)
    for index, sample in enumerate(samples):
        sessions[_session(str(sample["id"]))].append((_step(str(sample["id"])), index, sample))
    full, revealed, terminal, added = _make_variant(sessions, samples, "full", args.seed)
    own, _, _, _ = _make_variant(sessions, samples, "own", args.seed)
    partial, _, _, _ = _make_variant(sessions, samples, "partial", args.seed)
    fallback = ~revealed
    train_index = np.flatnonzero(train & fallback)
    valid_index = np.flatnonzero(valid & fallback)
    base = np.load(args.base_oof).astype(np.float32)
    xgb = np.load(args.xgb_oof).astype(np.float32)
    gru = np.load(args.gru_oof).astype(np.float32)
    teacher = np.load(args.teacher_oof).astype(np.float32)
    meta = np.asarray([meta_features(sample) for sample in samples], dtype=np.float32)
    ensemble = np.column_stack((xgb, gru)).astype(np.float32)
    tensors = {name: tensorize(value, base, ensemble, meta) for name, value in {
        "full": full, "partial": partial, "own": own,
    }.items()}
    train_parts = [[value[train_index] for value in tensors[name]] for name in ("full", "partial", "own")]
    features = [torch.cat([part[column] for part in train_parts], dim=0) for column in range(len(train_parts[0]))]
    train_target = np.tile(target[train_index], 3)
    train_teacher = np.tile(teacher[train_index], (3, 1)).astype(np.float32)
    train_loader = DataLoader(TensorDataset(*features, torch.from_numpy(train_target), torch.from_numpy(train_teacher)),
                              batch_size=args.batch_size, shuffle=True, generator=torch.Generator().manual_seed(args.seed), pin_memory=True)
    valid_loaders = {name: DataLoader(TensorDataset(*[value[valid_index] for value in tensors[name]]), batch_size=1024,
                                      shuffle=False, pin_memory=True) for name in tensors}
    device = torch.device("cuda")
    model = StitchFusion(base.shape[1], meta.shape[1]).to(device)
    counts = np.bincount(target[train_index], minlength=N_CLASSES)
    class_weight = torch.tensor(np.sqrt(len(train_index) / (N_CLASSES * counts)), dtype=torch.float32, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.6e-3, weight_decay=1e-4)
    best_score = -1.; best = None; stale = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        for batch in train_loader:
            *features_batch, hard, soft = [value.to(device, non_blocking=True) for value in batch]
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model(*features_batch)
                hard_loss = nn.functional.cross_entropy(logits, hard, weight=class_weight, label_smoothing=0.015)
                kd_loss = nn.functional.kl_div(torch.log_softmax(logits, dim=1), soft, reduction="batchmean")
                loss = (1. - args.distill) * hard_loss + args.distill * kd_loss
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0); optimizer.step()
        current = {name: predict(model, loader, device) for name, loader in valid_loaders.items()}
        score = .75 * f1(target[valid_index], current["full"]) + .25 * f1(target[valid_index], current["own"])
        print(f"epoch={epoch} full={f1(target[valid_index], current['full']):.6f} own={f1(target[valid_index], current['own']):.6f} select={score:.6f}", flush=True)
        if score > best_score:
            best_score = score; best = current; stale = 0
        else:
            stale += 1
            if stale >= args.patience:
                break
    assert best is not None
    base_blend = .5 * xgb[valid_index] + .5 * gru[valid_index]
    gate = added[valid_index] >= 2
    # 0.65 came from the independent 5-fold nested selection; it is not
    # re-tuned against this common validation partition.
    gated = base_blend.copy()
    gated[gate] = .35 * base_blend[gate] + .65 * best["full"][gate]
    truth = target[valid_index]
    terminal_valid = terminal[valid_index]
    recovered_valid = revealed[valid_index]
    # Construct full validation hybrid without ever learning the recovered
    # targets: the exact action is simply present in a later supplied history.
    hybrid_base = base_blend.argmax(1)
    hybrid_gate = gated.argmax(1)
    hybrid_base[recovered_valid] = truth[recovered_valid]
    hybrid_gate[recovered_valid] = truth[recovered_valid]
    from sklearn.metrics import f1_score
    result = {
        "n_train_fallback": int(len(train_index)), "n_valid_fallback": int(len(valid_index)),
        "n_valid_terminal": int(terminal_valid.sum()), "n_gate_added_ge2": int(gate.sum()),
        "best_epoch": epoch - stale,
        "base_fallback_macro": f1(truth, base_blend),
        "base_terminal_macro": f1(truth, base_blend, terminal_valid),
        "stitch_full_fallback_macro": f1(truth, best["full"]),
        "stitch_full_terminal_macro": f1(truth, best["full"], terminal_valid),
        "gate_fallback_macro": f1(truth, gated),
        "gate_terminal_macro": f1(truth, gated, terminal_valid),
        "base_added_ge2_macro": f1(truth, base_blend, gate),
        "stitch_added_ge2_macro": f1(truth, best["full"], gate),
        "gate_added_ge2_macro": f1(truth, gated, gate),
        "future_recovery_base_hybrid_macro": float(f1_score(truth, hybrid_base, labels=np.arange(N_CLASSES), average="macro", zero_division=0)),
        "future_recovery_gate_hybrid_macro": float(f1_score(truth, hybrid_gate, labels=np.arange(N_CLASSES), average="macro", zero_division=0)),
        "note": "0.65 stitch weight and added>=2 gate were chosen on full 5-fold nested OOF, before this split check.",
    }
    np.savez_compressed(out / "valid.npz", ids=np.asarray([samples[i]["id"] for i in valid_index]), target=truth,
                        terminal=terminal_valid, revealed=recovered_valid, added=added[valid_index], base=base_blend,
                        stitch_full=best["full"], stitch_own=best["own"], gated=gated)
    (out / "results.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
