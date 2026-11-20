#!/usr/bin/env python3
"""Summarize original BoxFusion together with the controlled prefix table."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re


PATTERN = re.compile(r"eval mAP:\s+([0-9.]+)")
TAGS = ("p25", "p50", "p75", "p100")


def metrics(path: Path) -> list[float]:
    values = [100.0 * float(value) for value in PATTERN.findall(path.read_text())]
    if len(values) != 3:
        raise RuntimeError(f"expected AP15/AP25/AP50 in {path}, got {values}")
    return values


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log-root", type=Path, required=True)
    parser.add_argument("--label-prefix", required=True)
    parser.add_argument("--controlled-results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    original = {
        tag: metrics(args.log_root / f"{args.label_prefix}_{tag}.log")
        for tag in TAGS
    }
    controlled_payload = json.loads(args.controlled_results.read_text())
    controlled = controlled_payload["ap"]["base"]
    carr = controlled_payload["ap"]["recar"]
    payload = {
        "BoxFusion": original,
        "BoxFusion*": controlled,
        "CaRR-3D": carr,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "results.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )

    lines = [
        "# ScanNet-100 causal online-prefix detection", "",
        "All prefixes are evaluated against the complete scene annotation. "
        "No future observations are used at readout.", "",
        "| Method | 25% AP15 | AP25 | AP50 | 50% AP15 | AP25 | AP50 | 75% AP15 | AP25 | AP50 | 100% AP15 | AP25 | AP50 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for method in ("BoxFusion", "BoxFusion*", "CaRR-3D"):
        values = [value for tag in TAGS for value in payload[method][tag]]
        formatted = " | ".join(f"{value:.2f}" for value in values)
        lines.append(f"| {method} | {formatted} |")
    (args.output / "REPORT.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
