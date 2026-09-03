#!/usr/bin/env python3
"""Print a compact, decision-oriented summary of an R2b feature audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


EXPECTED_SCHEMA = "boxfusion.tr3d_r2b_feature_audit.v1"


def _percent(value: Any) -> str:
    return f"{100.0 * float(value):.2f}%"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    args = parser.parse_args()
    report = json.loads(args.report.read_text(encoding="utf-8"))
    if report.get("schema") != EXPECTED_SCHEMA:
        raise ValueError(f"unsupported R2b audit schema: {report.get('schema')!r}")

    anchor = report["anchor"]
    counts = report["counts"]
    print(f"anchor: {anchor['name']} {anchor['metrics_percent']}")
    print(
        "scenes/GT/anchor/TR3D/residual/multiview: "
        f"{counts['scenes']}/{counts['gt']}/{counts['anchor']}/"
        f"{counts['candidate']}/{counts['residual']}/"
        f"{counts['residual_with_multiview_feature']}"
    )
    print(
        "clear positive IoU>.50 / negative IoU<=.15: "
        f"{counts['clear_positive_iou50']}/{counts['clear_negative_iou15']}"
    )

    separation = report["feature_separation"]
    print(
        "feature cosine q50 positive/negative: "
        f"{separation['positive_residual_iou50']['q50']} / "
        f"{separation['negative_residual_iou15']['q50']}"
    )
    print("ranking AUC/AP:")
    for name, row in report["ranking_diagnostics"].items():
        print(
            f"  {name:<15} "
            f"{float(row['auc']):.6f} / {float(row['average_precision']):.6f}"
        )

    print("fixed budgets (score-only -> score+depth+feature):")
    print("budget  TP50       P50-upper       novelTP50  delta")
    budget_rows = report["fixed_budget_comparison"]
    for budget in sorted(budget_rows, key=int):
        row = budget_rows[budget]
        score = row["score_only"]
        joint = row["score_depth_feature"]
        score_novel = score["thresholds"]["0.50"]["novel_oracle_tp"]
        joint_novel = joint["thresholds"]["0.50"]["novel_oracle_tp"]
        increment = row["increment"]
        print(
            f"{int(budget):>6d}  "
            f"{score['independent_tp50']:>3d}->{joint['independent_tp50']:<3d} "
            f"{_percent(score['independent_precision50_upper_bound']):>8}"
            f"->{_percent(joint['independent_precision50_upper_bound']):<8} "
            f"{score_novel:>3d}->{joint_novel:<3d} "
            f"TP={increment['independent_tp50']:+d}, "
            f"P={float(increment['independent_precision50_pp']):+.2f}pp, "
            f"novel={increment['novel_oracle_tp50']:+d}"
        )

    decision = report["decision"]
    print(
        "decision: "
        + (
            "PASS incremental gate"
            if decision["passes_pre_registered_incremental_gate"]
            else "STOP active R2 route; retain R2b as weak/negative ablation"
        )
    )
    print(f"required: {decision['required']}")
    print(f"next: {decision['next_step']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
