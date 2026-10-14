#!/usr/bin/env python3
"""Assemble the experiment-only strict blend-calibration conclusion."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def compact(metrics: dict) -> dict:
    keys = (
        "all_fallback_macro_f1", "terminal_fallback_macro_f1",
        "terminal_with_char20_macro_f1", "hybrid_future_history_plus_char20_macro_f1",
    )
    return {key: metrics[key] for key in keys}


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    out = root / "experiments/strict_blend_calibration"
    class_report = load(out / "strict_new_classalpha/report.json")
    logistic_report = load(out / "strict_new_logistic/report.json")
    dual_report = load(out / "dual_gru_mixture_report.json")
    baseline = class_report["baseline_source_scalar_geometric"]

    summary: dict = {
        "scope": "Final experiment-only report. No model/, script.py, baseline_submit.zip, or B42.zip file was edited by this calibration audit.",
        "strict_proxy": {
            "fixed_xgb": "2032-width strict common-train raw selector. Its train OOF and fixed validation blocks were built without common-validation action labels.",
            "fixed_xgb_seed_caveat": "strict proxy uses random_state=808; the current live selector uses random_state=42.",
            "gru": "strict nested-distilled no-E5 GRU for full OOF and strict common-split distill=0.35 GRU for fixed validation.",
            "selection_rule": "Every outer session fold is predicted by a calibrator/spec selected and fitted without that held fold's labels. Fixed-side selection uses common-train labels only.",
            "char20": "Never a calibration feature; it is applied after the fallback blend only on terminal rows.",
        },
        "baseline_current_strict_new_source_geometric": {
            "alpha_xgb": baseline["alpha_xgb"],
            "full_nested_group_oof": compact(baseline["full_nested_group_oof"]),
            "strict_common_fixed_validation": compact(baseline["common_fixed_validation"]),
        },
        "candidate_results": {
            "source_temperature_or_global_exponent": {
                "decision": "rejected_as_reparameterization",
                "reason": class_report["temperature_exponent_check"]["reason"],
            },
            "shared_class_specific_geometric_alpha": class_report["families"]["class_alpha"],
            "class_specific_geometric_alpha_plus_bias": class_report["families"]["class_alpha_bias"],
            "regularized_multinomial_logistic_stacker": logistic_report["families"]["logistic"],
            "dual_old_plus_strict_gru_probability_mix": {
                "decision": "rejected",
                "reason": (
                    "No fixed portable weight improved both strict full terminal fallback and deployment-aligned hybrid. "
                    "The closest w(strict)=0.65 improved full terminal but reduced full hybrid."
                ),
                "static_weight_grid": dual_report["static_weight_grid"],
                "nested_outer_weight_diagnostic": dual_report["nested_outer_weight_diagnostic"],
                "common_train_weight_diagnostic": dual_report["common_train_weight_diagnostic"],
            },
        },
        "decision": {
            "promote_calibration": False,
            "retain": "source-conditioned scalar geometric raw-XGB/strict-noE5-GRU blend: alpha_xgb sim=0.20, au=0.85; exact future-history recovery first and char20 only as final terminal override.",
            "reason": "No candidate strictly improved terminal fallback and future-history+char20 hybrid on both the nested session-GroupKFold audit and the strict common fixed validation.",
            "portable_candidate_artifact": None,
        },
        "audit_outputs": {
            "strict_common_raw_selector_oof": "experiments/strict_blend_calibration/common_train_raw_xgb_oof.npy",
            "strict_common_raw_selector_metadata": "experiments/strict_blend_calibration/common_train_raw_xgb_oof_metadata.json",
            "class_alpha_report": "experiments/strict_blend_calibration/strict_new_classalpha/report.json",
            "logistic_report": "experiments/strict_blend_calibration/strict_new_logistic/report.json",
            "dual_gru_report": "experiments/strict_blend_calibration/dual_gru_mixture_report.json",
        },
        "live_integrity_snapshot": {
            str(path.relative_to(root)): {"sha256": digest(path), "bytes": path.stat().st_size}
            for path in (
                root / "script.py", root / "model/best_model.pkl", root / "model/best_model_metadata.json",
                root / "baseline_submit.zip", root / "B42.zip",
            ) if path.exists()
        },
    }
    (out / "final_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    base_full = summary["baseline_current_strict_new_source_geometric"]["full_nested_group_oof"]
    base_fixed = summary["baseline_current_strict_new_source_geometric"]["strict_common_fixed_validation"]
    rows = []
    for name, record in summary["candidate_results"].items():
        if "delta_vs_source_scalar" in record:
            delta = record["delta_vs_source_scalar"]
            rows.append(f"| {name} | {delta['full_terminal']:+.6f} | {delta['fixed_terminal']:+.6f} | {record['safe_two_split_candidate']} |")
    markdown = f"""# Strict blend/calibration conclusion

No probability calibration is recommended for integration.  The strict
2032-feature raw-XGB proxy uses a different seed (808) from the live selector
(42), so this conclusion is deliberately conservative.

| Baseline | Full terminal | Strict fixed terminal | Full hybrid | Strict fixed hybrid |
| --- | ---: | ---: | ---: | ---: |
| scalar geometric (sim=.20, au=.85) | {base_full['terminal_fallback_macro_f1']:.6f} | {base_fixed['terminal_fallback_macro_f1']:.6f} | {base_full['hybrid_future_history_plus_char20_macro_f1']:.6f} | {base_fixed['hybrid_future_history_plus_char20_macro_f1']:.6f} |

| Candidate | Full terminal delta | Fixed terminal delta | Two-split safe |
| --- | ---: | ---: | --- |
{chr(10).join(rows)}

Global source temperatures/exponents do not change hard predictions beyond
the existing alpha reparameterization.  The dual-GRU mix also failed the same
two-split terminal+hybrid gate.  No live model, inference script, or ZIP was
edited by this audit; hashes are recorded in `final_summary.json`.
"""
    (out / "FINAL_REPORT.md").write_text(markdown, encoding="utf-8")
    print(json.dumps({"decision": summary["decision"], "baseline": summary["baseline_current_strict_new_source_geometric"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
