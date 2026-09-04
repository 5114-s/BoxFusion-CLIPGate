#!/usr/bin/env python3
"""Offline choice-set oracle audit for PVQ-AR shadow diagnostics.

Reads the per-scene ``*_pvq_ar.jsonl`` logs written by a shadow run and
labels each ambiguous Top-1/Top-2 event against ScanNet ground truth:

* The proposal's world-frame corners are matched to the best GT object.
* Each candidate track's world-frame corners is matched to GT the same way.
* correct_choice \u2208 {top1, top2, both, neither} for the proposal's GT
  object, which measures the native error rate and the rearrangement
  headroom without running the active module.

The GT frame convention is calibrated once by brute force against the
baseline predictions, so the audit does not rely on hand-derived aligned /
unaligned assumptions.

Usage:
  python tools/audit_scannet_pvq_ar_choice_set.py \
      --diagnostics-root diagnostics/pvq_ar_shadow_score05 \
      --baseline-root results/scannet_t05_boxer_replay_active_score05 \
      --output logs/pvq_ar_choice_set_audit.json
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from collections import Counter

import numpy as np
from scipy.spatial import ConvexHull

FLIP = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]])
FLIP_INV = FLIP.T


def get_3d_box(box_size, heading_angle, center):
    """VoteNet-style OBB corners in the y-down camera convention."""
    l, h, w = box_size[0], box_size[1], box_size[2]
    x_corners = [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2]
    y_corners = [h / 2, h / 2, h / 2, h / 2, -h / 2, -h / 2, -h / 2, -h / 2]
    z_corners = [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2]
    corners = np.array([x_corners, y_corners, z_corners])
    c, s = np.cos(heading_angle), np.sin(heading_angle)
    rotation = np.array([[1, 0, 0], [0, c, s], [0, -s, c]])
    corners = rotation @ corners
    corners = corners + np.array(center).reshape(3, 1)
    return corners.T


def box_iou(corners1, corners2):
    """Monte-Carlo OBB IoU mirroring Instances3D.obb_iou."""
    def inside(points, hull):
        return np.all(
            points @ hull.equations[:, :3].T + hull.equations[:, 3] <= 1e-6,
            axis=1,
        )

    try:
        hull1 = ConvexHull(corners1)
        hull2 = ConvexHull(corners2)
    except Exception:
        return 0.0
    lower = np.minimum(corners1.min(0), corners2.min(0))
    upper = np.maximum(corners1.max(0), corners2.max(0))
    # 12^3 grid samples, same density class as the native sampler.
    axes = [np.linspace(lo, hi, 12) for lo, hi in zip(lower, upper)]
    grid = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
    mask1 = inside(grid, hull1)
    mask2 = inside(grid, hull2)
    count1, count2 = int(mask1.sum()), int(mask2.sum())
    common = int((mask1 & mask2).sum())
    if count1 == 0 or count2 == 0:
        return 0.0
    return common / (count1 + count2 - common + 1e-6)


def load_gt_world(scene, gt_dir, align_dir, use_axis_align):
    raw = np.load(os.path.join(gt_dir, f"{scene}_bbox.npy"))
    corners_list = []
    for row in raw:
        center_cam = FLIP @ row[:3]
        corners = get_3d_box(row[3:6], row[6], center_cam)
        corners = (FLIP_INV @ corners.T).T
        if use_axis_align:
            matrix_path = os.path.join(
                align_dir, f"{scene}_axis_align_matrix.npy"
            )
            if not os.path.exists(matrix_path):
                raise FileNotFoundError(matrix_path)
            matrix = np.load(matrix_path)
            inv = np.linalg.inv(matrix)
            corners = (inv[:3, :3] @ corners.T).T + inv[:3, 3]
        corners_list.append(corners)
    return corners_list


def best_gt_iou(corners, gt_corners):
    return max((box_iou(corners, gt) for gt in gt_corners), default=0.0)


def match_gt_index(corners, gt_corners, min_iou=0.25):
    best_index, best_value = -1, 0.0
    for index, gt in enumerate(gt_corners):
        value = box_iou(corners, gt)
        if value > best_value:
            best_index, best_value = index, value
    if best_value < min_iou:
        return -1, best_value
    return best_index, best_value


def calibrate(scene, gt_dir, align_dir, baseline_root):
    pred_path = os.path.join(baseline_root, f"{scene}_boxes.pkl")
    if not os.path.exists(pred_path):
        return None, None
    import pickle

    rows = pickle.load(open(pred_path, "rb"))[0]
    preds = np.array([row[1] for row in rows])
    scores = np.array([row[2] for row in rows])
    order = np.argsort(-scores)[:16]
    variants = {}
    for use_axis_align in (False, True):
        try:
            gt = load_gt_world(scene, gt_dir, align_dir, use_axis_align)
        except FileNotFoundError:
            continue
        mean_best = float(
            np.mean([best_gt_iou(preds[i], gt) for i in order])
        )
        variants[use_axis_align] = mean_best
    if not variants:
        return None, None
    best_variant = max(variants, key=variants.get)
    return best_variant, variants


def pvq_offline_decision(event, rearrange_margin, min_similarity):
    native = event["candidates"][0]["best_similarity"]
    alt = event["candidates"][1]["best_similarity"]
    if native is None or alt is None:
        return 0
    if alt < min_similarity:
        return 0
    if alt - native >= rearrange_margin:
        return 1
    return 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--diagnostics-root", required=True)
    parser.add_argument("--baseline-root", required=True)
    parser.add_argument(
        "--gt-dir",
        default="/data/ZhaoX/BoxFusion/evaluation/data_util/scannet_train_detection_data",
    )
    parser.add_argument(
        "--align-dir",
        default="/extra/ZhaoX/scannet_data/scannet_instance_data",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--rearrange-margin", type=float, default=0.05)
    parser.add_argument("--min-similarity", type=float, default=0.50)
    parser.add_argument("--match-iou", type=float, default=0.25)
    args = parser.parse_args()

    scene_files = sorted(
        glob.glob(os.path.join(args.diagnostics_root, "*_pvq_ar.jsonl"))
    )
    if not scene_files:
        print("No PVQ-AR diagnostics found", file=sys.stderr)
        return 1

    stats = {
        "scenes": 0,
        "keyframes": 0,
        "edges": 0,
        "edges_with_runnerup": 0,
        "ambiguous_events": 0,
        "weak_top1_edges_le_0.10": 0,
        "weak_top1_edges_le_0.15": 0,
    }
    gap_histogram = Counter()
    correct_counter = Counter()
    pvq_confusion = Counter()
    event_records = []

    use_axis_align = None
    for scene_file in scene_files:
        scene = os.path.basename(scene_file).split("_pvq_ar.jsonl")[0]
        records = [
            json.loads(line)
            for line in open(scene_file, encoding="utf-8")
        ]
        edges = [r for r in records if r.get("type") == "correspondence_edge"]
        events = [r for r in records if r.get("type") == "ambiguity_event"]
        summaries = glob.glob(
            os.path.join(args.diagnostics_root, f"{scene}_pvq_ar_summary.json")
        )
        stats["scenes"] += 1
        stats["edges"] += len(edges)
        for edge in edges:
            m2 = edge.get("top2_margin")
            if m2 is not None:
                stats["edges_with_runnerup"] += 1
            if edge["top1_margin"] <= 0.10:
                stats["weak_top1_edges_le_0.10"] += 1
            if edge["top1_margin"] <= 0.15:
                stats["weak_top1_edges_le_0.15"] += 1
        stats["ambiguous_events"] += len(events)
        for event in events:
            gap = event["candidates"][0]["margin"] - event["candidates"][1]["margin"]
            gap_histogram[round(float(gap), 2)] += 1

        if not events:
            continue
        if use_axis_align is None:
            use_axis_align, variants = calibrate(
                scene, args.gt_dir, args.align_dir, args.baseline_root
            )
            if use_axis_align is None:
                print(
                    "Calibration failed: no baseline predictions for "
                    f"{scene}; skipping GT labelling",
                    file=sys.stderr,
                )
                continue
            print(
                f"GT convention calibrated on {scene}: "
                f"use_axis_align={use_axis_align} "
                f"(variants={variants})"
            )
        gt_corners = load_gt_world(
            scene, args.gt_dir, args.align_dir, use_axis_align
        )
        for event in events:
            proposal = np.asarray(event["proposal_corners_world"])
            proposal_gt, proposal_iou = match_gt_index(
                proposal, gt_corners, args.match_iou
            )
            candidate_matches = [
                match_gt_index(
                    np.asarray(candidate["corners_world"]),
                    gt_corners,
                    args.match_iou,
                )
                for candidate in event["candidates"]
            ]
            if proposal_gt < 0:
                correct = "unlabeled"
            else:
                top1_hit = candidate_matches[0][0] == proposal_gt
                top2_hit = candidate_matches[1][0] == proposal_gt
                if top1_hit and top2_hit:
                    correct = "both"
                elif top1_hit:
                    correct = "top1"
                elif top2_hit:
                    correct = "top2"
                else:
                    correct = "neither"
            correct_counter[correct] += 1
            pvq_choice = pvq_offline_decision(
                event, args.rearrange_margin, args.min_similarity
            )
            if correct == "top1":
                pvq_confusion[
                    "native_correct_pvq_kept" if pvq_choice == 0
                    else "native_correct_pvq_flipped"
                ] += 1
            elif correct == "top2":
                pvq_confusion[
                    "native_wrong_pvq_fixed" if pvq_choice == 1
                    else "native_wrong_pvq_kept"
                ] += 1
            event_records.append(
                {
                    "scene": scene,
                    "frame_id": event["frame_id"],
                    "proposal_init_id": event["proposal_init_id"],
                    "proposal_gt_iou": round(float(proposal_iou), 4),
                    "correct": correct,
                    "pvq_offline_choice": pvq_choice,
                    "logged_reason": event["reason"],
                }
            )

    result = {
        **stats,
        "gap_histogram": dict(sorted(gap_histogram.items())),
        "correct_choice_distribution": dict(correct_counter),
        "pvq_offline_confusion": dict(pvq_confusion),
        "events": event_records,
    }
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)

    print(json.dumps({k: v for k, v in result.items() if k != "events"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
