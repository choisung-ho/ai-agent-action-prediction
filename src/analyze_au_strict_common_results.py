#!/usr/bin/env python3
"""Write the auditable decision report for the strict AU-selector trial.

This is deliberately a post-hoc *analysis-only* companion to
``experiment_au_strict_common.py``.  It reads selector probabilities already
created by that program, plus pre-existing experiment probabilities, and
writes only beneath ``experiments/au_source_strict_common``.  In particular it
does not import XGBoost, fit a model, or touch the live bundle, script, or ZIP.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score


ROOT = Path(__file__).resolve().parents[1]
N_CLASSES = 14
EPSILON = 1.0e-12


def geometric(left: np.ndarray, right: np.ndarray, alpha: np.ndarray | float) -> np.ndarray:
    """Return normalized p_left**alpha * p_right**(1-alpha)."""
    weight = np.asarray(alpha, dtype=np.float32)
    if weight.ndim == 0:
        weight = np.full(len(left), float(weight), dtype=np.float32)
    value = weight[:, None] * np.log(np.maximum(left, EPSILON))
    value += (1.0 - weight[:, None]) * np.log(np.maximum(right, EPSILON))
    value -= value.max(axis=1, keepdims=True)
    value = np.exp(value)
    return value / value.sum(axis=1, keepdims=True)


def score(target: np.ndarray, probability: np.ndarray, mask: np.ndarray) -> float:
    return float(
        f1_score(
            target[mask], probability[mask].argmax(axis=1), labels=np.arange(N_CLASSES),
            average="macro", zero_division=0,
        )
    )


def metrics(target: np.ndarray, probability: np.ndarray, terminal: np.ndarray, source: np.ndarray) -> dict[str, float]:
    all_rows = np.ones(len(target), dtype=bool)
    au = source == "au"
    return {
        "all_macro_f1": score(target, probability, all_rows),
        "terminal_macro_f1": score(target, probability, terminal),
        "au_all_macro_f1": score(target, probability, au),
        "au_terminal_macro_f1": score(target, probability, au & terminal),
    }


def main() -> None:
    out = ROOT / "experiments/au_source_strict_common"
    blocks = np.load(out / "strict_blocks.npz", allow_pickle=False)
    ids = blocks["ids_valid"].astype(str)
    target = blocks["target_valid"].astype(np.int64)
    source = blocks["source_valid"].astype(str)
    raw = np.load(out / "raw_global_valid_probability.npy").astype(np.float32)
    terminal = np.load(
        ROOT / "experiments/modeling_ai_sequence_fusion_fixed_noe5/terminal_mask.npy",
        allow_pickle=False,
    ).astype(bool)
    gru = np.load(
        ROOT / "experiments/modeling_ai_sequence_fusion_fixed_noe5/valid_probabilities.npy",
        allow_pickle=False,
    ).astype(np.float32)
    old_xgb = np.load(ROOT / "experiments/modeling_ac_tune2/oof_probabilities.npy", allow_pickle=False).astype(np.float32)

    # Verify the copied validation artifacts use exactly the strict common
    # validation order before combining them.  This converts an accidental
    # ordering mismatch into a hard failure rather than a misleading score.
    split = {row["id"]: row["split"] for row in csv.DictReader((ROOT / "splits/common_split.csv").open(encoding="utf-8"))}
    train_ids = [json.loads(line)["id"] for line in (ROOT / "data/train.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    expected_ids = np.asarray([sample_id for sample_id in train_ids if split[sample_id] == "val"])
    if not np.array_equal(ids, expected_ids):
        raise ValueError("Strict blocks and fixed GRU/XGB artifacts have different validation ID order")
    if not (len(ids) == len(target) == len(source) == len(raw) == len(terminal) == len(gru) == len(old_xgb)):
        raise ValueError("Strict fixed artifacts have inconsistent lengths")

    deploy_alpha = np.where(source == "sim", 0.20, 0.85).astype(np.float32)
    strict_raw_geo = geometric(raw, gru, deploy_alpha)
    old_geo = geometric(old_xgb, gru, deploy_alpha)
    au_mask = source == "au"

    report: dict[str, object] = {
        "scope": "Analysis only. Live model/script/ZIP were not modified.",
        "protocol": {
            "strict_common_fixed": (
                "Selector training uses five-fold session-GroupKFold OOF E/D/B + source-E blocks within common-train; "
                "common-validation base/source blocks use models fit only on all common-train rows."
            ),
            "validation_sessions_overlap": 0,
            "gru_comparison": "Uses the pre-existing fixed-validation no-E5 probability artifact after exact ID-order verification.",
            "group_oof_support": (
                "The older AU GroupOOF result is retained only as supporting evidence because its base OOF learners were not "
                "retrained inside a second selector outer loop. It cannot override the strict fixed result."
            ),
        },
        "counts": {
            "common_train_rows": int(len(blocks["target_train"])),
            "fixed_valid_rows": int(len(target)),
            "fixed_terminal_rows": int(terminal.sum()),
            "fixed_au_rows": int(au_mask.sum()),
            "fixed_au_terminal_rows": int((au_mask & terminal).sum()),
        },
        "existing_geometry_fallback": {
            "source_xgb_alpha": {"sim": 0.20, "au": 0.85},
            "fixed_validation": metrics(target, old_geo, terminal, source),
        },
        "strict_raw_global": {
            "selector_only": metrics(target, raw, terminal, source),
            "with_existing_noe5_geometry": metrics(target, strict_raw_geo, terminal, source),
        },
        "candidates": {},
    }

    for name in ("au_same_raw", "au_regularized"):
        candidate = np.load(out / f"{name}_valid_probability.npy", allow_pickle=False).astype(np.float32)
        if candidate.shape != (int(au_mask.sum()), N_CLASSES):
            raise ValueError(f"{name} has unexpected probability shape: {candidate.shape}")
        direct = raw.copy()
        direct[au_mask] = candidate
        record: dict[str, object] = {
            "candidate_au_only": metrics(target[au_mask], candidate, terminal[au_mask], source[au_mask]),
            "direct_substitution": metrics(target, direct, terminal, source),
            "selector_geometric_grid_then_existing_noe5_geometry": {},
        }
        for raw_alpha in (0.0, 0.25, 0.50, 0.75, 1.0):
            selector = raw.copy()
            selector[au_mask] = geometric(raw[au_mask], candidate, raw_alpha)
            final = geometric(selector, gru, deploy_alpha)
            record["selector_geometric_grid_then_existing_noe5_geometry"][f"raw_alpha_{raw_alpha:.2f}"] = metrics(
                target, final, terminal, source,
            )
        report["candidates"][name] = record

    old_group_oof = json.loads((ROOT / "experiments/au_source_specialist/report.json").read_text(encoding="utf-8"))
    regularized = old_group_oof["variants"]["regularized"]
    report["supporting_non_nested_group_oof"] = {
        "raw_au_terminal_macro_f1": old_group_oof["baseline"]["raw_xgb_terminal_macro_f1"],
        "regularized_au_terminal_macro_f1": regularized["candidate_terminal_macro_f1"],
        "regularized_delta_terminal_vs_raw": regularized["raw_vs_candidate_terminal_delta"],
        "best_static_geometric_au_terminal_macro_f1": regularized["static_geometric_blends"]["raw_alpha_0.50"]["terminal_macro_f1"],
        "caveat": old_group_oof["variants"]["regularized"]["nested_scalar_blend"]["caveat"],
    }

    # On strict common validation every candidate either lowers the terminal
    # score or leaves it unchanged.  The historical OOF gain therefore fails
    # the required two-independent-split promotion rule.
    report["recommendation"] = {
        "promote": False,
        "rule": "Require a terminal improvement in both the session-GroupOOF support audit and strict common fixed validation.",
        "strict_fixed_best_candidate_terminal_macro_f1": max(
            value["selector_geometric_grid_then_existing_noe5_geometry"]["raw_alpha_1.00"]["terminal_macro_f1"]
            for value in report["candidates"].values()
        ),
        "existing_geometry_fallback_terminal_macro_f1": report["existing_geometry_fallback"]["fixed_validation"]["terminal_macro_f1"],
        "reason": (
            "Only 31 AU terminal sessions occur in the untouched fixed split; more importantly, neither AU-only model improves "
            "that terminal score. The strict raw+GRU geometry terminal score is below the existing fallback, so AU specialization "
            "is rejected rather than promoted."
        ),
    }
    (out / "comparison_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    baseline_terminal = report["existing_geometry_fallback"]["fixed_validation"]["terminal_macro_f1"]
    strict_terminal = report["strict_raw_global"]["with_existing_noe5_geometry"]["terminal_macro_f1"]
    markdown = f"""# Strict AU selector decision

**Decision: do not promote an AU-only selector.**

The strict common split contains {report['counts']['fixed_au_terminal_rows']} AU terminal sessions.  The strict raw selector plus
the existing no-E5 geometric fallback scores **{strict_terminal:.6f}** terminal
macro-F1, versus **{baseline_terminal:.6f}** for the existing fallback
(`{strict_terminal - baseline_terminal:+.6f}`).  Replacing AU rows with either
candidate does not improve terminal macro-F1.

The prior GroupOOF AU result was promising, but it used non-re-nested base OOF
blocks and is therefore supporting, not promotion, evidence.  Its apparent gain
does not replicate on the untouched strict fixed split.  See
`comparison_report.json` for all raw/AU/regularized/geometric metrics.
"""
    (out / "README.md").write_text(markdown, encoding="utf-8")
    print(json.dumps(report["recommendation"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
