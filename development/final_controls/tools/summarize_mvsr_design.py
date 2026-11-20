#!/usr/bin/env python3
"""Summarize official ScanNet MVSR factor and temporal ablations."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re


PATTERN = re.compile(r"eval mAP:\s+([0-9.]+)")
ARMS = (
    "native", "only_2d_iou", "full", "no_proposal_score", "no_visibility",
    "no_center_consistency", "no_scale_consistency", "no_overlap_consistency",
    "latest", "mean", "ema", "top3_mean", "maximum",
)


def metrics(path: Path) -> list[float]:
    values = [100.0 * float(value) for value in PATTERN.findall(path.read_text())]
    if len(values) != 3:
        raise RuntimeError(f"expected AP15/AP25/AP50 in {path}, got {values}")
    return values


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--eval-log-root", type=Path, required=True)
    parser.add_argument("--label-prefix", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    result = {
        arm: metrics(args.eval_log_root / f"{args.label_prefix}_{arm}.log")
        for arm in ARMS
    }
    payload = {"protocol": manifest, "metrics": result}
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "results.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )

    def table(title: str, arms: tuple[str, ...]) -> list[str]:
        lines = [f"## {title}", "", "| Arm | AP15 | AP25 | AP50 |", "|---|---:|---:|---:|"]
        for arm in arms:
            ap = result[arm]
            lines.append(f"| {arm} | {ap[0]:.4f} | {ap[1]:.4f} | {ap[2]:.4f} |")
        return lines + [""]

    report = [
        "# ScanNet MVSR design ablation", "",
        "All arms use the same strictly causal online evidence pass and the exact same native boxes, labels, and box count; only native confidence scores differ.", "",
    ]
    report += table("Composite-support factors", (
        "native", "only_2d_iou", "no_proposal_score", "no_visibility",
        "no_center_consistency", "no_scale_consistency",
        "no_overlap_consistency", "full",
    ))
    report += table("Temporal aggregation", (
        "latest", "mean", "ema", "top3_mean", "maximum",
    ))
    (args.output / "REPORT.md").write_text("\n".join(report), encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
