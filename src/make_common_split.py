#!/usr/bin/env python3
"""Create the team's fixed common train/validation split.

Default split:
- group split
- validation size: 20%
- seed: 42
- no session_id overlap between train and validation

Outputs:
- splits/common_split.csv
- splits/train_ids.txt
- splits/val_ids.txt
- splits/split_summary.json
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path


ALL_CLASSES = [
    "read_file",
    "grep_search",
    "list_directory",
    "glob_pattern",
    "edit_file",
    "write_file",
    "apply_patch",
    "run_bash",
    "run_tests",
    "lint_or_typecheck",
    "ask_user",
    "plan_task",
    "web_search",
    "respond_only",
]

SESSION_PATTERNS = [
    re.compile(r"^(?P<session>.+?)-step[_-]?\d+$"),
    re.compile(r"^(?P<session>.+?)_step[_-]?\d+$"),
    re.compile(r"^(?P<session>.+?):step[_-]?\d+$"),
]


def extract_session_id(sample_id: str) -> str:
    for pattern in SESSION_PATTERNS:
        match = pattern.match(sample_id)
        if match:
            return match.group("session")
    return sample_id


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create common validation split files.")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--out-dir", default="splits")
    parser.add_argument("--strategy", choices=["group", "random"], default="group")
    parser.add_argument("--val-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_ids(path: Path) -> list[str]:
    ids: list[str] = []
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if "id" not in obj:
                raise ValueError(f"{path}:{line_no} missing id")
            ids.append(str(obj["id"]))
    if not ids:
        raise ValueError(f"{path} is empty. Put the official train.jsonl into data/.")
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate ids found in train.jsonl")
    return ids


def load_labels(path: Path) -> dict[str, str]:
    labels: dict[str, str] = {}
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or "id" not in reader.fieldnames or "action" not in reader.fieldnames:
            raise ValueError(f"{path} must contain id,action columns")
        for row in reader:
            action = row["action"]
            if action not in ALL_CLASSES:
                raise ValueError(f"Unknown action label: {action}")
            labels[row["id"]] = action
    if not labels:
        raise ValueError(f"{path} is empty. Put the official train_labels.csv into data/.")
    return labels


def validate_ids_and_labels(ids: list[str], labels: dict[str, str]) -> None:
    id_set = set(ids)
    label_set = set(labels)
    missing = id_set - label_set
    extra = label_set - id_set
    if missing:
        raise ValueError(f"{len(missing)} train samples have no labels. Example: {next(iter(missing))}")
    if extra:
        raise ValueError(f"{len(extra)} labels have no train sample. Example: {next(iter(extra))}")


def make_random_split(ids: list[str], labels: dict[str, str], val_size: float, seed: int) -> set[str]:
    rng = random.Random(seed)
    by_class: dict[str, list[str]] = defaultdict(list)
    for sample_id in ids:
        by_class[labels[sample_id]].append(sample_id)
    val_ids: set[str] = set()
    for class_ids in by_class.values():
        rng.shuffle(class_ids)
        n_val = max(1, round(len(class_ids) * val_size)) if len(class_ids) > 1 else 0
        val_ids.update(class_ids[:n_val])
    return val_ids


def make_group_split(ids: list[str], labels: dict[str, str], val_size: float, seed: int) -> set[str]:
    rng = random.Random(seed)
    group_to_ids: dict[str, list[str]] = defaultdict(list)
    for sample_id in ids:
        group_to_ids[extract_session_id(sample_id)].append(sample_id)

    total_counts = Counter(labels[sample_id] for sample_id in ids)
    target_counts = {klass: total_counts[klass] * val_size for klass in ALL_CLASSES}
    target_total = len(ids) * val_size

    groups = []
    for group_id, group_ids in group_to_ids.items():
        counts = Counter(labels[sample_id] for sample_id in group_ids)
        groups.append((group_id, group_ids, counts))
    rng.shuffle(groups)
    groups.sort(key=lambda item: (len(item[1]), max(item[2].values() or [0])), reverse=True)

    val_groups: set[str] = set()
    val_counts: Counter[str] = Counter()
    val_total = 0

    def score(counts: Counter[str], total: int) -> float:
        total_penalty = abs(total - target_total) / max(target_total, 1)
        class_penalty = 0.0
        for klass in ALL_CLASSES:
            target = max(target_counts[klass], 1e-9)
            class_penalty += abs(counts[klass] - target) / target
        return total_penalty + class_penalty / len(ALL_CLASSES)

    for group_id, group_ids, counts in groups:
        next_counts = val_counts + counts
        next_total = val_total + len(group_ids)
        if val_total < target_total or score(next_counts, next_total) < score(val_counts, val_total):
            val_groups.add(group_id)
            val_counts = next_counts
            val_total = next_total

    return {
        sample_id
        for group_id in val_groups
        for sample_id in group_to_ids[group_id]
    }


def write_outputs(out_dir: Path, ids: list[str], labels: dict[str, str], val_ids: set[str], strategy: str, seed: int, val_size: float) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    train_ids = [sample_id for sample_id in ids if sample_id not in val_ids]
    val_id_list = [sample_id for sample_id in ids if sample_id in val_ids]
    train_groups = {extract_session_id(sample_id) for sample_id in train_ids}
    val_groups = {extract_session_id(sample_id) for sample_id in val_id_list}
    total_counts = Counter(labels[sample_id] for sample_id in ids)
    train_counts = Counter(labels[sample_id] for sample_id in train_ids)
    val_counts = Counter(labels[sample_id] for sample_id in val_id_list)

    with (out_dir / "common_split.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "split", "action", "session_id"])
        writer.writeheader()
        for sample_id in ids:
            writer.writerow(
                {
                    "id": sample_id,
                    "split": "val" if sample_id in val_ids else "train",
                    "action": labels[sample_id],
                    "session_id": extract_session_id(sample_id),
                }
            )

    (out_dir / "train_ids.txt").write_text("\n".join(train_ids) + "\n", encoding="utf-8")
    (out_dir / "val_ids.txt").write_text("\n".join(val_id_list) + "\n", encoding="utf-8")
    summary = {
        "strategy": strategy,
        "seed": seed,
        "requested_val_size": val_size,
        "n_total": len(ids),
        "n_train": len(train_ids),
        "n_val": len(val_id_list),
        "actual_val_size": len(val_id_list) / len(ids),
        "n_groups_total": len({extract_session_id(sample_id) for sample_id in ids}),
        "n_groups_train": len(train_groups),
        "n_groups_val": len(val_groups),
        "n_overlap_groups": len(train_groups & val_groups),
        "class_distribution": {
            klass: {
                "total": total_counts[klass],
                "train": train_counts[klass],
                "val": val_counts[klass],
                "val_ratio": val_counts[klass] / total_counts[klass] if total_counts[klass] else 0,
            }
            for klass in ALL_CLASSES
        },
    }
    (out_dir / "split_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote split files to {out_dir.resolve()}")
    print(f"n_total={len(ids)} n_train={len(train_ids)} n_val={len(val_id_list)} overlap_groups={summary['n_overlap_groups']}")


def main() -> None:
    args = parse_args()
    if not 0 < args.val_size < 1:
        raise ValueError("--val-size must be between 0 and 1")
    data_dir = Path(args.data_dir)
    ids = load_ids(data_dir / "train.jsonl")
    labels = load_labels(data_dir / "train_labels.csv")
    validate_ids_and_labels(ids, labels)
    if args.strategy == "group":
        val_ids = make_group_split(ids, labels, args.val_size, args.seed)
    else:
        val_ids = make_random_split(ids, labels, args.val_size, args.seed)
    write_outputs(Path(args.out_dir), ids, labels, val_ids, args.strategy, args.seed, args.val_size)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)
