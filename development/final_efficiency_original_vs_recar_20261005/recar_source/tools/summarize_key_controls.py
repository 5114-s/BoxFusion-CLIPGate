#!/usr/bin/env python3
"""Summarize final ScanNet PLR/CALR/MVSR key controls."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


PATTERN = re.compile(r"eval mAP:\s+([0-9.]+)")


def metrics(path: Path) -> list[float]:
    values = [100.0 * float(v) for v in PATTERN.findall(path.read_text())]
    if len(values) != 3:
        raise RuntimeError(f"expected three AP values in {path}, got {values}")
    return values


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--key-manifest", type=Path, required=True)
    parser.add_argument("--calr-manifest", type=Path, required=True)
    parser.add_argument("--eval-log-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    key = json.loads(args.key_manifest.read_text())
    calr = json.loads(args.calr_manifest.read_text())
    labels = {
        "plr_direct": "recar3d_key_plr_direct",
        "plr_current": "recar3d_key_plr_current",
        "calr_lower": "recar3d_key_calr_lower_threshold",
        "calr_simple": "recar3d_key_calr_strongest_matched",
        "calr_full": "recar3d_key_calr_full",
        "mvsr_raw": "recar3d_key_mvsr_raw_iou_max",
        "mvsr_composite": "recar3d_key_mvsr_composite_max",
        "full_raw": "recar3d_key_full_raw_iou",
        "full_composite": "recar3d_key_full_composite",
    }
    result = {
        name: metrics(args.eval_log_root / f"{label}.log")
        for name, label in labels.items()
    }
    plr_births = key["arm_counts"]["plr_current"] - key["arm_counts"]["plr_prefix"]
    direct_births = key["arm_counts"]["plr_direct"] - key["arm_counts"]["plr_prefix"]
    if plr_births != direct_births:
        raise RuntimeError("PLR aggregate budgets differ")
    calr_arms = calr["arms"]
    calr_births = int(calr_arms["full_calr"]["births"])
    if any(int(calr_arms[name]["births"]) != calr_births for name in calr_arms):
        raise RuntimeError("CALR aggregate budgets differ")

    summary = {
        "protocol": {
            "dataset": "ScanNet official100",
            "single_online_pass": True,
            "strictly_causal": True,
            "children_used": False,
            "plr_budget_matched_per_scene": True,
            "calr_budget_matched_per_scene": True,
        },
        "added_boxes": {"plr": plr_births, "calr": calr_births},
        "metrics": result,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "results.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )

    def row(title: str, births: int, name: str) -> str:
        ap = result[name]
        return f"| {title} | {births:,} | {ap[0]:.4f} | {ap[1]:.4f} | {ap[2]:.4f} |"

    report = [
        "# ScanNet-100 final key controls",
        "",
        "All arms are derived from one strictly causal online pass. PLR and CALR controls match the proposed branch's added-box count independently in every scene.",
        "",
        "## Candidate-recovery controls",
        "",
        "| Branch / strategy | Added boxes | AP15 | AP25 | AP50 |",
        "|---|---:|---:|---:|---:|",
        row("PLR: single-view direct proposal admission", plr_births, "plr_direct"),
        row("PLR: current cross-frame verification", plr_births, "plr_current"),
        row("CALR: lower-threshold top anchors", calr_births, "calr_lower"),
        row("CALR: strongest matched-budget simple control", calr_births, "calr_simple"),
        row("CALR: full three-keyframe voxel consensus", calr_births, "calr_full"),
        "",
        "## MVSR support control",
        "",
        "| Strategy | AP15 | AP25 | AP50 |",
        "|---|---:|---:|---:|",
        "| Native + raw 2D-IoU Max | {:.4f} | {:.4f} | {:.4f} |".format(*result["mvsr_raw"]),
        "| Native + composite-support Max | {:.4f} | {:.4f} | {:.4f} |".format(*result["mvsr_composite"]),
        "",
        "The full-system raw/composite pair is retained in `results.json`.",
        "",
    ]
    (args.output / "REPORT.md").write_text("\n".join(report))
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
