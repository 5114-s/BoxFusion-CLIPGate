#!/usr/bin/env python3
"""Print the compact decision summary from an R2a depth audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    args = parser.parse_args()
    report = json.loads(args.report.read_text(encoding="utf-8"))
    print(
        "anchor: "
        f"{report['anchor']['name']} "
        f"{report['anchor']['metrics_percent']}"
    )
    counts = report["counts"]
    print(
        "scenes/GT/anchor/TR3D/residual: "
        f"{report['scene_count']}/{counts['gt']}/{counts['anchor']}/"
        f"{counts['candidate']}/{counts['residual']}"
    )
    print("fixed validation diagnostics (not deployment thresholds):")
    print("gate              candidates  P50-upper  deltaR50  novelTP50")
    for name, row in report["fixed_depth_gates"].items():
        metric = row["thresholds"]["0.50"]
        print(
            f"{name:<17} {row['candidate_count']:>10d} "
            f"{100*row['independent_precision50_upper_bound']:>9.2f}% "
            f"{100*metric['delta_recall']:>8.2f}% "
            f"{metric['novel_oracle_tp']:>10d}"
        )
    print("evidence q50: positive residual IoU>.50 / negative IoU<=.15")
    for name, rows in report["evidence_separation"].items():
        positive = rows["positive_residual_iou50"]["q50"]
        negative = rows["negative_residual_iou15"]["q50"]
        print(f"  {name}: {positive} / {negative}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
