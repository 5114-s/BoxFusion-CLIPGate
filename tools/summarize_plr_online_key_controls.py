#!/usr/bin/env python3
"""Summarize official evaluator logs for final-online PLR controls."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re


PATTERN = re.compile(
    r"iou_thresh:\s*([0-9.]+).*?eval mAP:\s*([0-9.]+)", re.S
)


def metrics(path: Path) -> dict[str, float]:
    pairs = PATTERN.findall(path.read_text(encoding="utf-8"))
    result = {f"AP{int(round(float(iou) * 100))}": 100.0 * float(ap)
              for iou, ap in pairs}
    if set(result) != {"AP15", "AP25", "AP50"}:
        raise RuntimeError(f"incomplete metrics in {path}: {result}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--log-root", type=Path, required=True)
    parser.add_argument("--experiment-prefix", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    arms = ("direct_matched_budget", "without_native_dedup", "full_plr")
    results = {}
    for arm in arms:
        path = args.log_root / f"{args.experiment_prefix}_{arm}.log"
        results[arm] = {
            **manifest["arms"][arm],
            **metrics(path),
            "log": str(path.resolve()),
        }

    payload = {
        "schema": "boxfusion.plr.online_key_controls.report.v1",
        "protocol": "ScanNet official100 real-score evaluator",
        "results": results,
        "manifest": str(args.manifest.resolve()),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "results.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    labels = {
        "direct_matched_budget": "Direct append (matched budget)",
        "without_native_dedup": "PLR without native-map dedup",
        "full_plr": "Full online PLR",
    }
    lines = [
        "# Final-online proposal-only PLR key controls",
        "",
        "All arms use the same ScanNet official100 native map. Child evidence is disabled.",
        "",
        "| Strategy | Births | AP15 | AP25 | AP50 |",
        "|---|---:|---:|---:|---:|",
    ]
    for arm in arms:
        row = results[arm]
        lines.append(
            f"| {labels[arm]} | {row['births']} | {row['AP15']:.2f} | "
            f"{row['AP25']:.2f} | {row['AP50']:.2f} |"
        )
    lines += [
        "",
        "The direct-append control removes cross-frame confirmation, preserves "
        "native-map deduplication, recovery self-NMS, and size-based birth scores, "
        "and matches the full PLR output count separately for every scene. Its "
        "per-scene quota is copied from the reference PLR run; no ground truth is used.",
        "",
        "The no-dedup arm keeps the complete online cross-frame verification but "
        "disables every PLR/native overlap check. The production implementation is "
        "not modified; both controls are applied only inside their experiment process.",
    ]
    (args.output / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(args.output / "REPORT.md")


if __name__ == "__main__":
    main()
