#!/usr/bin/env python3
"""GT audit of PVQ-AR NMS-stage arbitration shadow decisions.

Labels every logged arbitration event (and in particular every
``refuse_contest`` would-refusal) against ScanNet GT:

* correct_refuse : the child carries a scored GT object that the parent
  does not own -> refusing the merge is the right call.
* wrong_refuse   : parent and child match the same GT object -> refusing
  breaks a correct dedup (the dangerous error class).
* unscored       : the child matches no GT -> harmless either way.
* recoverable    : subset of correct_refuse whose child GT object is also
  missed by the final predictions -> the headroom the module can convert
  into AP.

Usage:
  python tools/audit_scannet_pvq_nms_ar.py \
      --diagnostics-root diagnostics/pvq_nms_ar_shadow_score05 \
      --baseline-root results/scannet_t05_boxer_replay_active_score05 \
      --output logs/pvq_nms_ar_audit.json
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from audit_scannet_pvq_ar_choice_set import (  # noqa: E402
    box_iou,
    load_gt_world,
    match_gt_index,
)

GT_DIR_DEFAULT = (
    "/data/ZhaoX/BoxFusion/evaluation/data_util/scannet_train_detection_data"
)
ALIGN_DIR_DEFAULT = "/extra/ZhaoX/scannet_data/scannet_instance_data"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--diagnostics-root", required=True)
    parser.add_argument("--baseline-root", required=True)
    parser.add_argument("--gt-dir", default=GT_DIR_DEFAULT)
    parser.add_argument("--align-dir", default=ALIGN_DIR_DEFAULT)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--thresholds", type=float, nargs="+", default=[0.15, 0.25]
    )
    args = parser.parse_args()

    scene_files = sorted(
        glob.glob(os.path.join(args.diagnostics_root, "*_pvq_nms_ar.jsonl"))
    )
    if not scene_files:
        print("No NMS arbitration logs found", file=sys.stderr)
        return 1

    thresholds = args.thresholds
    totals = {
        thr: {
            "events": 0,
            "decisions": {},
            "refuse_labels": {},
            "refuse_recoverable": set(),
            "gt_total": 0,
            "gt_missed_final": set(),
        }
        for thr in thresholds
    }

    for scene_file in scene_files:
        scene = os.path.basename(scene_file).split("_pvq_nms_ar.jsonl")[0]
        gt_corners = load_gt_world(scene, args.gt_dir, args.align_dir, True)
        pred_path = os.path.join(args.baseline_root, f"{scene}_boxes.pkl")
        if not os.path.exists(pred_path):
            continue
        rows = pickle.load(open(pred_path, "rb"))[0]
        finals = [np.asarray(row[1]) for row in rows]

        for thr in thresholds:
            stats = totals[thr]
            stats["gt_total"] += len(gt_corners)
            matched = set()
            for corners in finals:
                index, _ = match_gt_index(corners, gt_corners, thr)
                if index >= 0:
                    matched.add(index)
            for index in range(len(gt_corners)):
                if index not in matched:
                    stats["gt_missed_final"].add(f"{scene}:{index}")

        for line in open(scene_file, encoding="utf-8"):
            record = json.loads(line)
            parent = np.asarray(record["parent_corners_world"])
            child = np.asarray(record["child_corners_world"])
            for thr in thresholds:
                stats = totals[thr]
                stats["events"] += 1
                decision = record["decision"]
                stats["decisions"][decision] = (
                    stats["decisions"].get(decision, 0) + 1
                )
                if decision != "refuse_contest":
                    continue
                parent_gt, _ = match_gt_index(parent, gt_corners, thr)
                child_gt, _ = match_gt_index(child, gt_corners, thr)
                if child_gt < 0:
                    label = "unscored"
                elif parent_gt == child_gt:
                    label = "wrong_refuse"
                else:
                    label = "correct_refuse"
                    if f"{scene}:{child_gt}" in stats["gt_missed_final"]:
                        stats["refuse_recoverable"].add(
                            f"{scene}:{child_gt}"
                        )
                stats["refuse_labels"][label] = (
                    stats["refuse_labels"].get(label, 0) + 1
                )

    result = {}
    for thr in thresholds:
        stats = totals[thr]
        refusals = sum(stats["refuse_labels"].values())
        correct = stats["refuse_labels"].get("correct_refuse", 0)
        wrong = stats["refuse_labels"].get("wrong_refuse", 0)
        result[f"iou_{thr:.2f}"] = {
            "events": stats["events"],
            "decisions": dict(
                sorted(stats["decisions"].items())
            ),
            "refuse_labels": dict(
                sorted(stats["refuse_labels"].items())
            ),
            "refuse_precision": round(
                correct / refusals, 4
            )
            if refusals
            else None,
            "refuse_wrong_rate": round(wrong / refusals, 4)
            if refusals
            else None,
            "refuse_recoverable_unique_gt": len(stats["refuse_recoverable"]),
            "gt_missed_final": len(stats["gt_missed_final"]),
            "recoverable_fraction_of_missed": round(
                len(stats["refuse_recoverable"])
                / max(len(stats["gt_missed_final"]), 1),
                4,
            ),
        }

    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
