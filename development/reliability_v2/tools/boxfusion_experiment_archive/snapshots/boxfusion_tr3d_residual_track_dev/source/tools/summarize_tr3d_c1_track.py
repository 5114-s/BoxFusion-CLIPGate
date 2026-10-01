#!/usr/bin/env python3
"""Print a compact C1 observer audit summary."""

from __future__ import annotations

import json
from pathlib import Path
import sys


def main() -> int:
    if len(sys.argv) != 2:
        raise SystemExit(f"usage: {Path(sys.argv[0]).name} AUDIT.json")
    payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    counts = payload["counts"]
    print(
        "C1 counts: "
        f"scenes={payload['scene_count']}, anchors={counts['active_anchor_predictions']}, "
        f"parent={counts['parent_proposals']}, unmatched_tracks={counts['unmatched_tracks']}"
    )
    print("route                    candidates  novel15  novel25  novel50  hitP25  dup25")
    for name, route in payload["routes"].items():
        p15 = route["thresholds"]["0.15"]
        p25 = route["thresholds"]["0.25"]
        p50 = route["thresholds"]["0.50"]
        print(
            f"{name:24s} {route['candidate_count']:10d} "
            f"{p15['novel_oracle_tp']:8d} {p25['novel_oracle_tp']:8d} "
            f"{p50['novel_oracle_tp']:8d} "
            f"{100*p25['independent_gt_hit_precision']:6.2f}% "
            f"{100*p25['self_duplicate_rate']:6.2f}%"
        )
    print("ranked budget            candidates  novel15  novel25  novel50  hitP25")
    for name, route in payload["ranked_budgets"].items():
        p15 = route["thresholds"]["0.15"]
        p25 = route["thresholds"]["0.25"]
        p50 = route["thresholds"]["0.50"]
        print(
            f"{name:24s} {route['candidate_count']:10d} "
            f"{p15['novel_oracle_tp']:8d} {p25['novel_oracle_tp']:8d} "
            f"{p50['novel_oracle_tp']:8d} "
            f"{100*p25['independent_gt_hit_precision']:6.2f}%"
        )
    decision = payload["pre_registered_advance_gate"]
    print(f"advance to separate C2 observer: {'PASS' if decision['pass'] else 'FAIL'}")
    print("standard AP: unchanged by construction (observer-only, applied_count=0)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
