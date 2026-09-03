#!/usr/bin/env python3
"""Compact terminal summary for a C2 GT audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ROUTE_ORDER = (
    "source_top3", "source_top5", "source_top10", "mask_any", "mask1",
    "mask2", "mask2_depth", "mask3_strict", "top5_mask1",
    "top5_mask2_depth",
)


def _print_routes(routes: dict, *, title: str | None = None) -> None:
    if title:
        print(title)
    print("route                    candidates   P(hit)@15/25/50   novelTP@15/25/50")
    for name in ROUTE_ORDER:
        route = routes[name]
        values = [route["thresholds"][key] for key in ("0.15", "0.25", "0.50")]
        precision = "/".join(
            f"{100 * item['independent_gt_hit_precision']:.1f}" for item in values
        )
        novel = "/".join(str(item["novel_oracle_tp"]) for item in values)
        print(f"{name:24s} {route['candidate_count']:10d}   {precision:>17s}   {novel}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    args = parser.parse_args()
    report = json.loads(args.report.read_text(encoding="utf-8"))
    partitions = report.get("partitions", {})
    heldout = partitions.get("heldout")
    if heldout is None:
        _print_routes(report["routes"])
    else:
        all_partition = partitions["all"]
        _print_routes(
            all_partition["routes"],
            title=f"[all: {all_partition['scene_count']} scenes]",
        )
        print()
        _print_routes(
            heldout["routes"],
            title=f"[heldout: {heldout['scene_count']} scenes; authoritative]",
        )
    decision = report["decision"]
    partition_name = report.get("decision_partition", "all")
    print(f"C2 primary ({partition_name}): {decision['primary_route']}")
    for name, value in decision["checks"].items():
        print(f"  {'PASS' if value else 'FAIL'} {name}")
    print(f"advance to separate C3 shadow: {'PASS' if decision['pass'] else 'FAIL'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
