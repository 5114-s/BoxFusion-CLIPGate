#!/usr/bin/env python3
"""Summarize official ScanNet logs for a paired final factorial."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


ORDER = ("base", "p", "a", "m2", "p_a", "p_m2", "a_m2", "p_a_m2")
LABELS = {
    "base": "Base", "p": "PLR", "a": "CALR", "m2": "MVSR",
    "p_a": "PLR + CALR", "p_m2": "PLR + MVSR",
    "a_m2": "CALR + MVSR", "p_a_m2": "PLR + CALR + MVSR",
}


def metric(path: Path) -> list[float]:
    values = [100.0 * float(v) for v in re.findall(r"eval mAP:\s*([0-9.]+)", path.read_text())]
    if len(values) != 3:
        raise RuntimeError(f"expected three AP values in {path}")
    return values


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--log-root", type=Path, required=True)
    parser.add_argument("--log-prefix", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    results = {
        arm: metric(args.log_root / f"{args.log_prefix}_{arm}.log")
        for arm in ORDER
    }
    payload = {
        "schema": "boxfusion.recar3d.scannet_factorial.metrics.v2",
        "scene_count": manifest["scene_count"],
        "results": results,
        "arm_counts": manifest["arm_counts"],
        "full_minus_base": [
            a - b for a, b in zip(results["p_a_m2"], results["base"])
        ],
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "results.json").write_text(json.dumps(payload, indent=2) + "\n")
    lines = [
        "# ScanNet-100 final paired factorial", "",
        "All eight arms come from one strictly causal online pass.", "",
        "| Configuration | Boxes | AP15 | AP25 | AP50 |",
        "|---|---:|---:|---:|---:|",
    ]
    for arm in ORDER:
        ap = results[arm]
        lines.append(
            f"| {LABELS[arm]} | {manifest['arm_counts'][arm]:,} | "
            f"{ap[0]:.4f} | {ap[1]:.4f} | {ap[2]:.4f} |"
        )
    (args.output / "REPORT.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
