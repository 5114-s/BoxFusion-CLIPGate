#!/usr/bin/env python3
"""Run the fixed-geometry paper100 oracle for sealed L3B-HBMedoid."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Sequence


ROOT = Path(__file__).resolve().parents[1]
if os.fspath(ROOT) not in sys.path:
    sys.path.insert(0, os.fspath(ROOT))

from tools.audit_scannet_l3_hbc_t1_selector_paper100_oracle import (  # noqa: E402
    L3OracleError,
    audit,
)
from tools.run_scannet_l3b_hbmedoid_t1_selector_paper100 import (  # noqa: E402
    PROTOCOL_ID,
    SCHEMA as SHADOW_SCHEMA,
)


SCHEMA = "boxfusion.scannet_l3b_hbmedoid_t1_selector_paper100_oracle.v1"
DEFAULT_SHADOW = ROOT / "logs/scannet_l3b_hbmedoid_t1_selector_paper100_score05/final/L3B_HBMEDOID_T1_SELECTOR_PAPER100.json"
DEFAULT_OUT = ROOT / "reports/l3b_hbmedoid_t1_selector_paper100_oracle/L3B_HBMEDOID_T1_SELECTOR_PAPER100_ORACLE.json"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-list", type=Path, default=ROOT / "evaluation/data_util/meta_data/scannetv2_val.txt")
    parser.add_argument("--shadow", type=Path, default=DEFAULT_SHADOW)
    parser.add_argument("--baseline-root", type=Path, default=ROOT / "results/scannet_t05_boxer_replay_active_score05")
    parser.add_argument("--gt-root", type=Path, default=ROOT / "evaluation/data_util/scannet_train_detection_data")
    parser.add_argument("--scan-root", type=Path, default=Path("/extra/ZhaoX/scannet_data/scans"))
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)
    if args.out.exists() or args.out.is_symlink():
        raise L3OracleError(f"refusing to overwrite L3B oracle: {args.out}")
    report = audit(
        scene_list=args.scene_list,
        shadow_path=args.shadow,
        baseline_root=args.baseline_root,
        gt_root=args.gt_root,
        scan_root=args.scan_root,
        expected_shadow_schema=SHADOW_SCHEMA,
        expected_protocol_id=PROTOCOL_ID,
        report_schema=SCHEMA,
    )
    # Replace inherited stage labels without changing the frozen gate result.
    gate = bool(report["decision"]["passes_ap50_plus10_and_144_match_gate"])
    report["decision"]["next_step"] = (
        "freeze_L4_gtfree_track_admission_policy"
        if gate
        else "discard_L3B_HBMedoid_geometry_selector_for_plus10_route"
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False)
        handle.write("\n")
    summary = {
        key: {
            "ap_points": row["gt_selected_track_suffix"]["official_evaluation"]["ap_points"],
            "delta_ap_points": row["gt_selected_track_suffix"]["delta_ap_points"],
            "additional_matches": row["additional_union_matching_over_native"],
        }
        for key, row in report["per_threshold"].items()
    }
    print(json.dumps({"out": os.fspath(args.out), "summary": summary, "decision": report["decision"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
