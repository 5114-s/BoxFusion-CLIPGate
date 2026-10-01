#!/usr/bin/env python3
"""Absorbed-TP headroom audit for a potential NMS-stage PVQ-AR.

Reads the observer-only ``*_pvq_nms.jsonl`` merge logs and labels every
native NMS absorb decision (parent keeps, child merges in) against GT:

* correct-dedup : parent and child match the same GT object -> the merge
  is the desired behaviour.
* absorbed-tp   : the child matches a GT object the parent does not, and
  that GT object is missed by the final predictions.  Re-routing the
  child to its own track would recover a true positive: this is the only
  AP channel a rearrangement module can act on at the NMS hook point.
* clutter       : the child matches no GT object (unscored).
* already-covered: child's GT object is matched by some final prediction
  anyway, so the merge did not cost a detection.

The go/no-go gate: absorbed-tp recovery as a fraction of missed GT (and
of all GT) vs the ~+1 AP bar.

Usage:
  python tools/audit_scannet_pvq_nms_headroom.py \
      --diagnostics-root diagnostics/pvq_nms_observer_score05 \
      --baseline-root results/scannet_t05_boxer_replay_active_score05 \
      --output logs/pvq_nms_headroom_audit.json
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
        glob.glob(os.path.join(args.diagnostics_root, "*_pvq_nms.jsonl"))
    )
    if not scene_files:
        print("No NMS observer logs found", file=sys.stderr)
        return 1

    thresholds = args.thresholds
    totals = {
        thr: {
            "merges": 0,
            "correct_dedup": 0,
            "clutter": 0,
            "already_covered": 0,
            "absorbed_tp_events": 0,
            "absorbed_tp_unique_gt": set(),
            "gt_total": 0,
            "gt_missed_final": set(),
            "near_threshold_absorbed": 0,
        }
        for thr in thresholds
    }

    for scene_file in scene_files:
        scene = os.path.basename(scene_file).split("_pvq_nms.jsonl")[0]
        gt_corners = load_gt_world(scene, args.gt_dir, args.align_dir, True)
        pred_path = os.path.join(args.baseline_root, f"{scene}_boxes.pkl")
        if not os.path.exists(pred_path):
            continue
        rows = pickle.load(open(pred_path, "rb"))[0]
        finals = [np.asarray(row[1]) for row in rows]

        for thr in thresholds:
            totals[thr]["gt_total"] += len(gt_corners)
            matched_finals = set()
            for corners in finals:
                index, _ = match_gt_index(corners, gt_corners, thr)
                if index >= 0:
                    matched_finals.add(index)
            missed = set(range(len(gt_corners))) - matched_finals
            totals[thr]["gt_missed_final"].update(
                f"{scene}:{index}" for index in missed
            )

        for line in open(scene_file, encoding="utf-8"):
            record = json.loads(line)
            parent = np.asarray(record["parent_corners_world"])
            child = np.asarray(record["child_corners_world"])
            for thr in thresholds:
                stats = totals[thr]
                stats["merges"] += 1
                parent_gt, parent_iou = match_gt_index(
                    parent, gt_corners, thr
                )
                child_gt, child_iou = match_gt_index(
                    child, gt_corners, thr
                )
                if child_gt < 0:
                    stats["clutter"] += 1
                    continue
                if parent_gt == child_gt:
                    stats["correct_dedup"] += 1
                    continue
                # The child carried a scored object that the parent does
                # not own: check whether the final output still covers it.
                covered = any(
                    match_gt_index(corners, gt_corners, thr)[0] == child_gt
                    for corners in finals
                )
                if covered:
                    stats["already_covered"] += 1
                    continue
                stats["absorbed_tp_events"] += 1
                stats["absorbed_tp_unique_gt"].add(f"{scene}:{child_gt}")
                if record["iou"] <= 0.15:
                    stats["near_threshold_absorbed"] += 1

    result = {}
    for thr in thresholds:
        stats = totals[thr]
        missed_total = len(stats["gt_missed_final"])
        recovered = len(stats["absorbed_tp_unique_gt"])
        result[f"iou_{thr:.2f}"] = {
            "merges": stats["merges"],
            "correct_dedup": stats["correct_dedup"],
            "clutter": stats["clutter"],
            "already_covered": stats["already_covered"],
            "absorbed_tp_events": stats["absorbed_tp_events"],
            "absorbed_tp_unique_gt": recovered,
            "near_threshold_absorbed": stats["near_threshold_absorbed"],
            "gt_total": stats["gt_total"],
            "gt_missed_final": missed_total,
            "recovered_fraction_of_missed": round(
                recovered / max(missed_total, 1), 4
            ),
            "recovered_fraction_of_all_gt": round(
                recovered / max(stats["gt_total"], 1), 4
            ),
        }

    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
