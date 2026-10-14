#!/usr/bin/env python3
"""Full-data refit and PyTorch-free export for the no-E5 GRU fusion candidate.

Nothing in this utility promotes a model.  It creates an experiment artifact
whose weights can be evaluated by ``numpy_sequence_fusion.py`` without adding
PyTorch or a transformer to a competition submission.
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
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from numpy_sequence_fusion import probabilities as numpy_probabilities
from train_modeling_ai_sequence_fusion import ALL_CLASSES, SequenceFusion, sequence_tensors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--base-oof", default="experiments/modeling_n_refit/oof_stack_features.npy")
    parser.add_argument("--out-dir", default="experiments/modeling_ai_sequence_fusion_noe5_full")
    parser.add_argument("--epochs", type=int, default=10, help="OOF-best epoch average rounded to ten.")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--teacher-oof", default=None,
                        help="Optional cross-fitted rich-teacher probabilities used only while fitting the full student.")
    parser.add_argument("--distill", type=float, default=0.0,
                        help="Teacher KL-loss coefficient in [0, 1); zero retains the original hard-label refit.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to refit this experiment")
    if not 0.0 <= args.distill < 1.0:
        raise ValueError("--distill must be in [0, 1)")
    if args.distill > 0.0 and not args.teacher_oof:
        raise ValueError("--distill requires --teacher-oof")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    samples = [json.loads(line) for line in open(Path(args.data_dir) / "train.jsonl", encoding="utf-8") if line.strip()]
    label_map = {row["id"]: row["action"] for row in csv.DictReader(open(Path(args.data_dir) / "train_labels.csv", encoding="utf-8"))}
    to_id = {label: index for index, label in enumerate(ALL_CLASSES)}
    targets = np.asarray([to_id[label_map[sample["id"]]] for sample in samples], dtype=np.int64)
    base = np.load(args.base_oof).astype(np.float32)
    if len(base) != len(samples):
        raise ValueError("Base OOF rows do not align with train.jsonl")
    action, status, arg, meta = sequence_tensors(samples)
    e5 = np.zeros((len(samples), 1), dtype=np.float32)
    tensors = [torch.from_numpy(value) for value in (action, status, arg, base, e5, meta)]
    teacher: np.ndarray | None = None
    if args.teacher_oof:
        teacher = np.load(args.teacher_oof).astype(np.float32)
        if teacher.shape != (len(samples), len(ALL_CLASSES)):
            raise ValueError(f"Teacher OOF must be {(len(samples), len(ALL_CLASSES))}, got {teacher.shape}")
    if teacher is None:
        data = TensorDataset(*tensors, torch.from_numpy(targets))
    else:
        data = TensorDataset(*tensors, torch.from_numpy(targets), torch.from_numpy(teacher))
    loader = DataLoader(data, batch_size=args.batch_size, shuffle=True,
                        generator=torch.Generator().manual_seed(args.seed), pin_memory=True)

    device = torch.device("cuda")
    model = SequenceFusion(base.shape[1], 1, meta.shape[1]).to(device)
    counts = np.bincount(targets, minlength=len(ALL_CLASSES))
    weight = torch.tensor(np.sqrt(len(targets) / (len(ALL_CLASSES) * counts)), dtype=torch.float32, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.8e-3, weight_decay=1.0e-4)
    losses: list[float] = []
    started = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        weighted_loss = 0.0
        seen = 0
        for batch in loader:
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
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            weighted_loss += float(loss.detach()) * len(target)
            seen += len(target)
        losses.append(weighted_loss / seen)
        print(f"epoch={epoch} loss={losses[-1]:.6f}", flush=True)

    state = {name: value.detach().cpu().numpy().astype(np.float32) for name, value in model.state_dict().items()}
    np.savez_compressed(out / "weights.npz", **state)
    # A fixed probe establishes exact NumPy/PyTorch forward equivalence for
    # this artifact and can be rerun independently without retraining.
    probe = np.arange(min(256, len(samples)), dtype=np.int64)
    model.eval()
    with torch.inference_mode():
        torch_prob = torch.softmax(model(
            tensors[0][probe].to(device), tensors[1][probe].to(device), tensors[2][probe].to(device),
            tensors[3][probe].to(device), tensors[4][probe].to(device), tensors[5][probe].to(device),
        ), dim=1).float().cpu().numpy()
    numpy_prob = numpy_probabilities(action[probe], status[probe], arg[probe], base[probe], meta[probe], state)
    maximum_difference = float(np.max(np.abs(torch_prob - numpy_prob)))
    agreement = float(np.mean(torch_prob.argmax(axis=1) == numpy_prob.argmax(axis=1)))
    np.savez_compressed(out / "verification_probe.npz", action=action[probe], status=status[probe], arg=arg[probe],
                        base=base[probe], meta=meta[probe], torch_probabilities=torch_prob)
    architecture = {
        "format": "sequence_fusion_noe5_numpy_v1",
        "all_classes": ALL_CLASSES,
        "input": {"action_shape": [6], "status_shape": [6], "arg_shape": [6, 7],
                  "base_dim": int(base.shape[1]), "meta_dim": int(meta.shape[1]), "e5_input": "constant zeros [N,1]"},
        "architecture": {"action_embedding": [len(ALL_CLASSES) + 1, 32], "status_embedding": [4, 8],
                         "position_embedding": [6, 8], "arg_projection": [7, 16],
                         "gru": {"input_dim": 64, "hidden_dim": 72, "bidirectional": True, "layers": 1},
                         "base_projection": [int(base.shape[1]), 128], "e5_projection": [1, 96],
                         "meta_projection": [int(meta.shape[1]), 32], "head": [400, 240, len(ALL_CLASSES)]},
        "activation": "PyTorch GELU exact (erf)", "layer_norm_eps": 1e-5,
        "training": {
            "epochs": args.epochs, "seed": args.seed, "base_input": "70k GroupKFold OOF E/D/B margins", "losses": losses,
            "distillation": {
                "enabled": teacher is not None and args.distill > 0.0,
                "coefficient": args.distill, "teacher_oof": args.teacher_oof,
                "teacher_usage": "full-data training target only; the portable exported student receives no teacher or E5 input at inference",
            },
        },
        "numpy_equivalence": {"probe_rows": int(len(probe)), "max_abs_probability_difference": maximum_difference,
                               "argmax_agreement": agreement},
        "seconds": round(time.time() - started, 2),
    }
    (out / "architecture.json").write_text(json.dumps(architecture, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(architecture["numpy_equivalence"], ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
