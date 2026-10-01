#!/usr/bin/env python3
"""Build the six-row ScanNet candidate-recovery strategy table."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re


PATTERN = re.compile(
    r"iou_thresh:\s*([0-9.]+).*?eval mAP:\s*([0-9.]+)", re.S
)


def metrics(path: Path) -> dict[str, float]:
    result = {
        f"AP{int(round(float(iou) * 100))}": 100.0 * float(ap)
        for iou, ap in PATTERN.findall(path.read_text(encoding="utf-8"))
    }
    if set(result) != {"AP15", "AP25", "AP50"}:
        raise RuntimeError(f"incomplete official metrics in {path}: {result}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plr-results", type=Path, required=True)
    parser.add_argument("--calr-manifest", type=Path, required=True)
    parser.add_argument("--log-root", type=Path, required=True)
    parser.add_argument("--calr-prefix", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    plr = json.loads(args.plr_results.read_text(encoding="utf-8"))["results"]
    calr_manifest = json.loads(
        args.calr_manifest.read_text(encoding="utf-8")
    )
    calr = {}
    for arm in ("lower_threshold", "strongest_matched_budget", "full_calr"):
        calr[arm] = {
            **calr_manifest["arms"][arm],
            **metrics(args.log_root / f"{args.calr_prefix}_{arm}.log"),
        }

    rows = [
        ("PLR", "Direct append, matched budget", plr["direct_matched_budget"]),
        ("", "Without native-map dedup", plr["without_native_dedup"]),
        ("", "Full PLR", plr["full_plr"]),
        ("CALR", "Lower detection threshold", calr["lower_threshold"]),
        ("", "Strongest matched-budget control", calr["strongest_matched_budget"]),
        ("", "Full CALR", calr["full_calr"]),
    ]
    payload = {
        "schema": "boxfusion.candidate_recovery.table3.v1",
        "dataset": "ScanNet official100",
        "rows": [
            {"branch": branch, "strategy": strategy, **values}
            for branch, strategy, values in rows
        ],
        "plr_results": str(args.plr_results.resolve()),
        "calr_manifest": str(args.calr_manifest.resolve()),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "table3_results.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# ScanNet-100 candidate-recovery strategy ablation",
        "",
        "| Branch | Strategy | Births | AP15 | AP25 | AP50 |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for branch, strategy, values in rows:
        lines.append(
            f"| {branch} | {strategy} | {values['births']} | "
            f"{values['AP15']:.2f} | {values['AP25']:.2f} | "
            f"{values['AP50']:.2f} |"
        )
    lines += [
        "",
        "PLR arms share the current native map and disable child evidence. The "
        "direct control matches the full PLR output count in every scene. CALR "
        "arms share the current strict-online no-child PLR+MVSR prefix, and both "
        "controls match the full CALR output count in every scene. The strongest "
        "CALR control shares its pre-NMS anchor pool, voxel exemplar selection, "
        "and tail-score protocol with CALR, while removing only distinct-keyframe "
        "confirmation. Ground truth is used only by the official evaluator.",
    ]
    (args.output / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(args.output / "REPORT.md")


if __name__ == "__main__":
    main()
