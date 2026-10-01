#!/usr/bin/env python3
"""Print the authoritative decision summary from an R3 correction audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


EXPECTED_SCHEMA = "boxfusion.tr3d_r3_near_correction_audit.v1"
PRIMARY_RULE = "tr3d_score_gt_anchor_score"


def _partition(report: dict[str, Any]) -> tuple[str, dict[str, Any], bool]:
    counterfactual = report["counterfactual"]
    if counterfactual["mode"] == "development_fixed10_veto_only":
        return "development10 (veto-only)", counterfactual["development10"], False
    if counterfactual["mode"] == "full100_with_frozen_heldout90":
        return "heldout90 (authoritative)", counterfactual["heldout90"], True
    raise ValueError(f"unsupported R3 partition mode: {counterfactual['mode']!r}")


def _aps(rows: dict[str, Any]) -> str:
    return " / ".join(
        f"{100.0 * float(rows[key]['scored']['average_precision']):.4f}"
        for key in ("0.15", "0.25", "0.50")
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    args = parser.parse_args()
    report = json.loads(args.report.read_text(encoding="utf-8"))
    if report.get("schema") != EXPECTED_SCHEMA:
        raise ValueError(f"unsupported R3 audit schema: {report.get('schema')!r}")

    name, selected, authoritative = _partition(report)
    primary = selected["fixed_rules"][PRIMARY_RULE]
    replacement = {
        key: primary["thresholds"][key]["replacement"]
        for key in ("0.15", "0.25", "0.50")
    }
    print(f"partition: {name}")
    print(f"counts: {report['counts']}")
    print(f"G0 AP15/AP25/AP50: {_aps(selected['baseline'])}")
    print(f"R3 AP15/AP25/AP50: {_aps(replacement)}")
    print(
        "delta AP15/AP25/AP50: "
        + " / ".join(
            f"{100.0 * float(replacement[key]['delta_scored_ap']):+.4f}"
            for key in ("0.15", "0.25", "0.50")
        )
    )
    gate = selected["pre_registered_gate"]
    row = gate["rules"][PRIMARY_RULE]
    print(
        "AP50 crossing gain/loss/net, precision, positive scenes: "
        f"{row['cross50_gain']}/{row['cross50_loss']}/"
        f"{row['cross50_gain_minus_loss']}, "
        f"{100.0 * float(row['cross50_replacement_precision']):.2f}%, "
        f"{row['cross50_positive_scene_coverage']}"
    )
    if authoritative:
        print("decision: " + ("PASS heldout gate" if gate["primary_rule_pass"] else "STOP R3 route"))
    else:
        print("decision: development signal only; direct activation is forbidden")
    print(f"required: {gate['required']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
