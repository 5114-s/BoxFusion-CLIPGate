#!/usr/bin/env python3
"""Run the paper100 oracle for sealed L1 minimum-two-view F3 identities."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Sequence


_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if os.fspath(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, os.fspath(_REPOSITORY_ROOT))

from tools.audit_scannet_l0_f3_f4_perview_paper100_oracle import (
    ROOT,
    L0OracleError,
    audit,
)
from tools.seal_scannet_l1_f3_2view_f4_perview_paper100 import (
    PROTOCOL_ID,
    SCHEMA as SEAL_SCHEMA,
)


SCHEMA = "boxfusion.scannet_l1_f3_2view_f4_perview_paper100_oracle.v1"
DEFAULT_SEAL = ROOT / "logs/scannet_l1_f3_2view_f4_perview_paper100_score05/final/L1_F3_2VIEW_F4_PERVIEW_PAPER100.json"
DEFAULT_OUT = ROOT / "reports/l1_f3_2view_f4_perview_paper100_oracle/L1_F3_2VIEW_F4_PERVIEW_PAPER100_ORACLE.json"


class L1OracleError(L0OracleError):
    pass


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-list", type=Path, default=ROOT / "evaluation/data_util/meta_data/scannetv2_val.txt")
    parser.add_argument("--seal", type=Path, default=DEFAULT_SEAL)
    parser.add_argument("--baseline-root", type=Path, default=ROOT / "results/scannet_t05_boxer_replay_active_score05")
    parser.add_argument("--gt-root", type=Path, default=ROOT / "evaluation/data_util/scannet_train_detection_data")
    parser.add_argument("--scan-root", type=Path, default=Path("/extra/ZhaoX/scannet_data/scans"))
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)
    if args.out.exists() or args.out.is_symlink():
        raise L1OracleError(f"refusing to overwrite L1 oracle report: {args.out}")
    report = audit(
        scene_list=args.scene_list,
        seal_path=args.seal,
        baseline_root=args.baseline_root,
        gt_root=args.gt_root,
        scan_root=args.scan_root,
        expected_seal_schema=SEAL_SCHEMA,
        expected_protocol_id=PROTOCOL_ID,
        report_schema=SCHEMA,
        seal_track_count_key="eligible_track_count",
        identity_description=(
            "one F3 track with at least two distinct retained views contributes "
            "at most one F4 per-view geometry"
        ),
    )
    counts = report["counts"]
    counts["minimum_two_view_track_identity_count"] = counts.pop(
        "confirmed_track_identity_count"
    )
    for scene_row in report["scenes"].values():
        scene_row["minimum_two_view_track_identity_count"] = scene_row.pop(
            "confirmed_track_identity_count"
        )
    ap50 = report["per_threshold"]["0.50"]
    gate = bool(
        ap50["gt_selected_candidate_suffix"]["delta_ap_points"] > 10.0
        and ap50["additional_union_matching_over_native"] >= 144
    )
    report["decision"] = {
        "passes_l1_ap50_plus10_and_144_match_gate": gate,
        "authorize_gt_free_two_view_best_view_selector_experiment": gate,
        "authorize_accuracy_claim": False,
        "next_step": (
            "freeze_gt_free_two_view_best_view_selector"
            if gate
            else "minimum_two_view_track_compression_below_target_use_source_preserving_supporter"
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False)
        handle.write("\n")
    print(
        json.dumps(
            {
                "out": os.fspath(args.out),
                "counts": report["counts"],
                "decision": report["decision"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
