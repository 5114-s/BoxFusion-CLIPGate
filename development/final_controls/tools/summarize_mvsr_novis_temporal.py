#!/usr/bin/env python3
"""Summarize final three-factor MVSR temporal aggregation on ScanNet."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re


PATTERN = re.compile(r"eval mAP:\s+([0-9.]+)")
ARMS = ("native", "latest", "mean", "ema", "top3_mean", "maximum")


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
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    result = {
        arm: metrics(args.eval_log_root / f"{args.label_prefix}_{arm}.log")
        for arm in ARMS
    }
    payload = {"protocol": manifest, "metrics": result}
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "results.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# ScanNet final three-factor MVSR temporal aggregation", "",
        "All arms use proposal score, 2D overlap, and 3D geometry consistency;",
        "visibility is excluded. Native geometry, labels, and box count are fixed.",
        "", "| Aggregation | AP15 | AP25 | AP50 |",
        "|---|---:|---:|---:|",
    ]
    for arm in ARMS:
        ap = result[arm]
        lines.append(
            f"| {arm} | {ap[0]:.4f} | {ap[1]:.4f} | {ap[2]:.4f} |"
        )
    (args.output / "REPORT.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
