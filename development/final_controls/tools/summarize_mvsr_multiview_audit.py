#!/usr/bin/env python3
"""Summarize the ScanNet-100 MVSR distinct-view audit."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


ARMS = ("native", "single_view", "two_view", "three_view", "shuffled_negative")
PATTERN = re.compile(r"eval mAP:\s+([0-9.]+)")


def read_metrics(path: Path) -> list[float]:
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
    metrics = {
        arm: read_metrics(args.eval_log_root / f"{args.label_prefix}_{arm}.log")
        for arm in ARMS
    }
    payload = {"protocol": manifest, "metrics": metrics}
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "results.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    lines = [
        "# ScanNet-100 MVSR distinct-view audit", "",
        "All arms use the final no-visibility support. Native geometry and row count are fixed.",
        "The shuffled negative permutes the single-view logit corrections among native identities within each scene.",
        "", "| Arm | Activated native rows | AP15 | AP25 | AP50 |", "|---|---:|---:|---:|---:|",
    ]
    for arm in ARMS:
        ap = metrics[arm]
        active = 0 if arm == "native" else manifest["activated_rows"][arm]
        lines.append(f"| {arm} | {active} | {ap[0]:.4f} | {ap[1]:.4f} | {ap[2]:.4f} |")
    (args.output / "REPORT.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
