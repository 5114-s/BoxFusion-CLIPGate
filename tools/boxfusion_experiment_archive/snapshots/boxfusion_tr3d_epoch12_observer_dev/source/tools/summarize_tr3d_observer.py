#!/usr/bin/env python3
"""Print the decision-relevant portion of a TR3D observer report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _percent(value: float) -> str:
    return f"{100.0 * float(value):.2f}%"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    args = parser.parse_args()
    payload = json.loads(args.report.read_text(encoding="utf-8"))
    print(f"anchor: {payload['anchor']['name']}")
    metrics = payload["anchor"]["metrics_percent"]
    print(
        "anchor AP15/AP25/AP50: "
        f"{metrics['AP15']:.4f}/{metrics['AP25']:.4f}/{metrics['AP50']:.4f}"
    )
    print(
        "scenes/GT/anchor/TR3D candidates: "
        f"{payload['scene_count']}/{payload['ground_truth_count']}/"
        f"{payload['b6_prediction_count']}/{payload['tr3d_candidate_count']}"
    )
    print(f"TR3D runtime: {payload['tr3d_runtime_s']:.3f}s total")
    for key in ("0.15", "0.25", "0.50"):
        row = payload["thresholds"][key]
        print(
            f"IoU {key}: anchor/union oracle recall="
            f"{_percent(row['b6_oracle_recall'])}/"
            f"{_percent(row['union_oracle_recall'])}; "
            f"delta={_percent(row['union_oracle_recall_gain'])}; "
            f"novelTP={row['novel_oracle_tp']}; "
            f"raw novel precision upper="
            f"{_percent(row['novel_precision_upper_bound'])}"
        )
    gate = payload["continuation_gate"]
    print(f"pre-registered raw-stream gate: {'PASS' if gate['pass'] else 'STOP'}")
    print("score frontier @ IoU0.50 (diagnostic only; no val threshold selection):")
    print("score  candidates  deltaR50  novelTP  novelP-upper")
    for row in payload["score_frontier"]["rows"].values():
        value = row["thresholds"]["0.50"]
        print(
            f"{row['candidate_score_threshold']:>5.2f}  "
            f"{row['candidate_count']:>10d}  "
            f"{_percent(value['union_oracle_recall_gain']):>8s}  "
            f"{value['novel_oracle_tp']:>7d}  "
            f"{_percent(value['novel_precision_upper_bound']):>13s}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
