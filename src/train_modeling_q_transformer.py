#!/usr/bin/env python3
"""Modeling Q: richer-context, hybrid leaf/hierarchy Transformer classifier.

This is deliberately an *experiment-only* trainer.  It reads the frozen
``common_split.csv`` and writes every checkpoint, report, and prediction under
``experiments/modeling_q_transformer``.  It never promotes a checkpoint into
``model/`` and therefore cannot alter the submission package by accident.

The previous Transformer experiments used multilingual-e5-small with a short,
action-only summary of context.  Q uses the larger multilingual-e5-base model,
keeps recent user requests as well as tool history, and trains a leaf classifier
jointly with the known action hierarchy.  At validation time it selects only
between pre-declared path-logit blending values and records every score.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import time
from collections import Counter
from pathlib import Path

# Avoid severe fragmentation on Windows WDDM GPUs.  This must be set before
# importing torch.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import classification_report, f1_score
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

from train_modeling_b import ALL_CLASSES
from train_modeling_e import TREE, bucket, child_for_label, result_status


DEFAULT_ENCODER = "intfloat/multilingual-e5-base"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--split-path", default="splits/common_split.csv")
    parser.add_argument("--out-dir", default="experiments/modeling_q_transformer")
    parser.add_argument("--base-encoder", default=DEFAULT_ENCODER)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--grad-accum", type=int, default=2)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--max-length", type=int, default=192)
    parser.add_argument("--encoder-lr", type=float, default=1.5e-5)
    parser.add_argument("--head-lr", type=float, default=6e-4)
    parser.add_argument("--hierarchy-loss", type=float, default=0.45)
    parser.add_argument("--label-smoothing", type=float, default=0.02)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-gradient-checkpointing", action="store_true")
    return parser.parse_args()


def compact(value: object, limit: int) -> str:
    """Flatten a field but cap each context segment before tokenization."""
    if isinstance(value, dict):
        text = " ".join(f"{key}={compact(item, limit)}" for key, item in sorted(value.items()))
    elif isinstance(value, list):
        text = " ".join(compact(item, limit) for item in value)
    else:
        text = str(value or "")
    text = " ".join(text.replace("\x00", " ").split())
    return text[:limit]


def rich_model_text(sample: dict) -> str:
    """Encode the current request first, then the most useful recent state.

    The order is intentional: ordinary tokenizer truncation keeps the current
    request even for unusually long conversations.  Prior *user* utterances
    are retained because they often establish an unresolved subject, something
    the action-only E5 input could not see.
    """
    prompt = compact(sample.get("current_prompt"), 720)
    parts = ["query: [CURRENT_REQUEST] " + prompt]
    history = sample.get("history") if isinstance(sample.get("history"), list) else []
    # Last six items cover roughly three user/action pairs without crowding the
    # current request out of a 192-token window.
    for relative, item in enumerate(reversed(history[-6:]), start=1):
        if not isinstance(item, dict):
            continue
        role = str(item.get("role") or "unknown").lower()
        if role == "user":
            parts.append(f"[PREVIOUS_USER_{relative}] {compact(item.get('content'), 260)}")
        elif role == "assistant_action":
            name = compact(item.get("name"), 48).lower() or "none"
            status = result_status(item.get("result_summary"))
            args = item.get("args") if isinstance(item.get("args"), dict) else {}
            useful_args = {key: args[key] for key in ("path", "target", "scope", "pattern", "cmd", "query") if key in args}
            # Result text is deliberately only a compact status; raw terminal
            # output made earlier sparse and Transformer models noisier.
            parts.append(
                f"[PREVIOUS_ACTION_{relative}] name={name} status={status} args={compact(useful_args, 120)}"
            )

    meta = sample.get("session_meta") if isinstance(sample.get("session_meta"), dict) else {}
    workspace = meta.get("workspace") if isinstance(meta.get("workspace"), dict) else {}
    language_mix = workspace.get("language_mix") if isinstance(workspace.get("language_mix"), dict) else {}
    language_key = "_".join(sorted(language_mix, key=language_mix.get, reverse=True)[:2]) if language_mix else "none"
    open_files = workspace.get("open_files") if isinstance(workspace.get("open_files"), list) else []
    parts.append(
        "[STATE] "
        f"turn={meta.get('turn_index', 'unknown')} tier={meta.get('user_tier', 'unknown')} "
        f"language={meta.get('language_pref', 'unknown')} langmix={language_key} "
        f"git_dirty={workspace.get('git_dirty', 'unknown')} ci={workspace.get('last_ci_status', 'unknown')} "
        f"open_files={len(open_files)} loc={bucket(workspace.get('loc'), [(2000, 'xs'), (8000, 's'), (20000, 'm'), (50000, 'l')])}"
    )
    return " ".join(parts)


NODE_CLASSES = {node: list(children) for node, children in TREE.items()}
LABEL_TO_ID = {label: index for index, label in enumerate(ALL_CLASSES)}
NODE_TARGETS = {
    node: {
        label: NODE_CLASSES[node].index(child_for_label(node, label)) if label in sum(TREE[node].values(), []) else -100
        for label in ALL_CLASSES
    }
    for node in TREE
}


class TextDataset(Dataset):
    def __init__(self, texts: list[str], labels: list[int]):
        self.texts = texts
        self.labels = labels

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, index: int) -> tuple[str, int]:
        return self.texts[index], self.labels[index]


def make_collate(tokenizer, max_length: int):
    def collate(rows: list[tuple[str, int]]) -> dict:
        texts, labels = zip(*rows)
        batch = tokenizer(
            list(texts), padding=True, truncation=True, max_length=max_length,
            return_tensors="pt", pad_to_multiple_of=8,
        )
        batch["labels"] = torch.tensor(labels, dtype=torch.long)
        return batch
    return collate


class HybridHierarchyClassifier(nn.Module):
    def __init__(self, encoder_name: str):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(encoder_name)
        hidden = int(self.encoder.config.hidden_size)
        self.dropout = nn.Dropout(0.12)
        self.leaf = nn.Linear(hidden, len(ALL_CLASSES))
        self.heads = nn.ModuleDict({node: nn.Linear(hidden, len(children)) for node, children in NODE_CLASSES.items()})

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, **kwargs) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        # Models such as XLM-R do not accept token_type_ids.
        kwargs = {key: value for key, value in kwargs.items() if key != "token_type_ids"}
        states = self.encoder(input_ids=input_ids, attention_mask=attention_mask, **kwargs).last_hidden_state
        states = states.masked_fill(~attention_mask[..., None].bool(), 0.0)
        pooled = states.sum(dim=1) / attention_mask.sum(dim=1, keepdim=True).clamp_min(1)
        pooled = F.normalize(pooled, p=2, dim=1)
        pooled = self.dropout(pooled)
        return self.leaf(pooled), {node: head(pooled) for node, head in self.heads.items()}


def make_weights(labels: list[str], device: torch.device) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    label_counts = Counter(labels)
    leaf_weights = torch.tensor(
        [math.sqrt(len(labels) / (len(ALL_CLASSES) * label_counts[label])) for label in ALL_CLASSES],
        dtype=torch.float32, device=device,
    )
    node_weights: dict[str, torch.Tensor] = {}
    for node, children in NODE_CLASSES.items():
        values = [NODE_TARGETS[node][label] for label in labels if NODE_TARGETS[node][label] >= 0]
        counts = Counter(values)
        total = len(values)
        node_weights[node] = torch.tensor(
            [math.sqrt(total / (len(children) * counts[index])) for index in range(len(children))],
            dtype=torch.float32, device=device,
        )
    return leaf_weights, node_weights


def path_log_probs(node_logits: dict[str, torch.Tensor]) -> torch.Tensor:
    """Return log P(label | hierarchy heads) in ALL_CLASSES order."""
    values: list[torch.Tensor] = []
    for label in ALL_CLASSES:
        node = "root"
        pieces: list[torch.Tensor] = []
        while True:
            child = child_for_label(node, label)
            child_index = NODE_CLASSES[node].index(child)
            pieces.append(F.log_softmax(node_logits[node], dim=1)[:, child_index])
            labels = TREE[node][child]
            if len(labels) == 1:
                break
            node = child
        values.append(torch.stack(pieces, dim=1).sum(dim=1))
    return torch.stack(values, dim=1)


@torch.inference_mode()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, alphas: list[float]) -> tuple[dict[float, float], np.ndarray, np.ndarray]:
    model.eval()
    all_leaf: list[np.ndarray] = []
    all_paths: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    for batch in loader:
        labels = batch.pop("labels")
        batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            leaf, nodes = model(**batch)
            paths = path_log_probs(nodes)
        all_leaf.append(leaf.float().cpu().numpy())
        all_paths.append(paths.float().cpu().numpy())
        all_labels.append(labels.numpy())
    leaf = np.concatenate(all_leaf); paths = np.concatenate(all_paths); targets = np.concatenate(all_labels)
    scores: dict[float, float] = {}
    for alpha in alphas:
        predictions = (F.log_softmax(torch.from_numpy(leaf), dim=1).numpy() + alpha * paths).argmax(axis=1)
        scores[alpha] = f1_score(
            [ALL_CLASSES[index] for index in targets], [ALL_CLASSES[index] for index in predictions],
            labels=ALL_CLASSES, average="macro", zero_division=0,
        )
    return scores, leaf, paths


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Modeling Q is intentionally GPU-only; CUDA was not detected.")
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    # Reproducible enough for a fixed split without turning off fast Tensor Core kernels.
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    samples = [json.loads(line) for line in open(Path(args.data_dir) / "train.jsonl", encoding="utf-8") if line.strip()]
    with open(Path(args.data_dir) / "train_labels.csv", newline="", encoding="utf-8") as handle:
        label_map = {row["id"]: row["action"] for row in csv.DictReader(handle)}
    with open(args.split_path, newline="", encoding="utf-8") as handle:
        split_map = {row["id"]: row["split"] for row in csv.DictReader(handle)}

    texts = [rich_model_text(sample) for sample in samples]
    y_labels = [label_map[sample["id"]] for sample in samples]
    y_ids = [LABEL_TO_ID[label] for label in y_labels]
    train_index = [index for index, sample in enumerate(samples) if split_map[sample["id"]] == "train"]
    val_index = [index for index, sample in enumerate(samples) if split_map[sample["id"]] == "val"]
    train_texts = [texts[index] for index in train_index]
    train_labels = [y_labels[index] for index in train_index]
    val_texts = [texts[index] for index in val_index]
    val_ids = [y_ids[index] for index in val_index]
    (out_dir / "run_config.json").write_text(json.dumps({
        **vars(args), "train_rows": len(train_index), "val_rows": len(val_index),
        "input_format": "current request + six recent history items + compact state",
        "submission_files_modified": False,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    tokenizer = AutoTokenizer.from_pretrained(args.base_encoder)
    collate = make_collate(tokenizer, args.max_length)
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        TextDataset(train_texts, [y_ids[index] for index in train_index]), batch_size=args.batch_size,
        shuffle=True, generator=generator, collate_fn=collate, num_workers=0, pin_memory=True,
    )
    val_loader = DataLoader(
        TextDataset(val_texts, val_ids), batch_size=args.eval_batch_size,
        shuffle=False, collate_fn=collate, num_workers=0, pin_memory=True,
    )
    device = torch.device("cuda")
    model = HybridHierarchyClassifier(args.base_encoder).to(device)
    if not args.no_gradient_checkpointing:
        model.encoder.gradient_checkpointing_enable()
        if hasattr(model.encoder.config, "use_cache"):
            model.encoder.config.use_cache = False
    leaf_weights, node_weights = make_weights(train_labels, device)
    optimizer = torch.optim.AdamW([
        {"params": model.encoder.parameters(), "lr": args.encoder_lr, "weight_decay": 0.01},
        {"params": list(model.leaf.parameters()) + list(model.heads.parameters()), "lr": args.head_lr, "weight_decay": 0.01},
    ], betas=(0.9, 0.999), eps=1e-8)
    updates_per_epoch = math.ceil(len(train_loader) / args.grad_accum)
    total_updates = updates_per_epoch * args.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, int(total_updates * args.warmup_ratio), total_updates)
    alphas = [0.0, 0.20, 0.35, 0.50, 0.70, 1.0]
    best_score = -1.0
    records: list[dict] = []
    print(
        f"device={torch.cuda.get_device_name(0)} train={len(train_index)} val={len(val_index)} "
        f"batch={args.batch_size} accum={args.grad_accum} updates={total_updates} encoder={args.base_encoder}", flush=True,
    )
    for epoch in range(1, args.epochs + 1):
        model.train(); running_loss = 0.0; running_leaf = 0.0; running_path = 0.0
        start = time.time(); optimizer.zero_grad(set_to_none=True)
        for step, batch in enumerate(train_loader, start=1):
            labels = batch.pop("labels").to(device, non_blocking=True)
            batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                leaf, nodes = model(**batch)
                leaf_loss = F.cross_entropy(leaf, labels, weight=leaf_weights, label_smoothing=args.label_smoothing)
                path_losses: list[torch.Tensor] = []
                label_names = [ALL_CLASSES[index] for index in labels.detach().cpu().tolist()]
                for node, logits in nodes.items():
                    targets = torch.tensor([NODE_TARGETS[node][label] for label in label_names], device=device, dtype=torch.long)
                    if (targets >= 0).any():
                        path_losses.append(F.cross_entropy(logits, targets, weight=node_weights[node], ignore_index=-100, label_smoothing=args.label_smoothing))
                hierarchy_loss = torch.stack(path_losses).mean()
                loss = ((1.0 - args.hierarchy_loss) * leaf_loss + args.hierarchy_loss * hierarchy_loss) / args.grad_accum
            loss.backward()
            running_loss += loss.item() * args.grad_accum
            running_leaf += leaf_loss.item(); running_path += hierarchy_loss.item()
            if step % args.grad_accum == 0 or step == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step(); scheduler.step(); optimizer.zero_grad(set_to_none=True)
            if step == 1 or step % 100 == 0:
                print(
                    f"epoch={epoch} step={step}/{len(train_loader)} loss={running_loss/step:.4f} "
                    f"leaf={running_leaf/step:.4f} hierarchy={running_path/step:.4f} elapsed={time.time()-start:.1f}s",
                    flush=True,
                )
        scores, leaf_logits, hierarchy_paths = evaluate(model, val_loader, device, alphas)
        alpha, score = max(scores.items(), key=lambda item: item[1])
        elapsed = time.time() - start
        record = {
            "epoch": epoch, "macro_f1": score, "selected_path_alpha": alpha,
            "all_alpha_scores": {str(key): value for key, value in scores.items()},
            "loss": running_loss / len(train_loader), "leaf_loss": running_leaf / len(train_loader),
            "hierarchy_loss": running_path / len(train_loader), "seconds": elapsed,
        }
        records.append(record)
        predictions = (F.log_softmax(torch.from_numpy(leaf_logits), dim=1).numpy() + alpha * hierarchy_paths).argmax(axis=1)
        target_names = [ALL_CLASSES[index] for index in val_ids]
        pred_names = [ALL_CLASSES[index] for index in predictions]
        (out_dir / f"epoch_{epoch}_classification_report.txt").write_text(
            classification_report(target_names, pred_names, labels=ALL_CLASSES, digits=4, zero_division=0), encoding="utf-8",
        )
        (out_dir / f"epoch_{epoch}_scores.json").write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"epoch={epoch} macro_f1={score:.6f} selected_alpha={alpha:.2f} alpha_scores={scores} seconds={elapsed:.1f}", flush=True)
        if score > best_score:
            best_score = score
            torch.save({
                "epoch": epoch, "state_dict": model.state_dict(), "best_macro_f1": score,
                "selected_path_alpha": alpha, "all_classes": ALL_CLASSES, "args": vars(args),
            }, out_dir / "best_checkpoint.pt")
            np.savez_compressed(out_dir / "best_val_logits.npz", leaf_logits=leaf_logits, hierarchy_paths=hierarchy_paths,
                                targets=np.asarray(val_ids), ids=np.asarray([samples[index]["id"] for index in val_index]))
            tokenizer.save_pretrained(out_dir / "tokenizer")
    (out_dir / "results.json").write_text(json.dumps({"best_macro_f1": best_score, "epochs": records}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"DONE best_macro_f1={best_score:.6f} out_dir={out_dir}", flush=True)


if __name__ == "__main__":
    main()
