#!/usr/bin/env python3
"""Offline GT-only recall audit for causal incremental TR3D tracks.

Inference diagnostics are required to attest that they were observer-only and
never accessed ground truth.  This auditor is intentionally separate from the
online process and never materializes predictions.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
if str(_ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(_ROOT / "tools"))

from audit_tr3d_residual_observer import (  # noqa: E402
    _alignment, _gt_boxes, _load_b6, _minmax, _transform,
    maximum_cardinality, pairwise_iou,
)


IOU_THRESHOLDS = (0.15, 0.25, 0.50)
SCORE_THRESHOLDS = (0.01, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50)
HIT_THRESHOLDS = (2, 3, 4)


def _scenes(path: Path) -> list[str]:
    rows = [line.split()[0] for line in path.read_text().splitlines()
            if line.strip() and not line.lstrip().startswith("#")]
    if not rows or len(rows) != len(set(rows)):
        raise ValueError("scene list is empty or contains duplicates")
    return rows


def _diagnostic(path: Path, scene: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    value = json.loads(path.read_text())
    if (value.get("schema") not in {
                "boxfusion.tr3d_incremental_online_observer.v2",
                "boxfusion.tr3d_incremental_online_observer.v3",
            }
            or value.get("observer_only") is not True
            or value.get("mutation_enabled") is not False
            or value.get("applied_count") != 0
            or value.get("ground_truth_access") is not False
            or value.get("scene_id") != scene
            or value.get("coordinate_frame") != "world_unaligned"):
        raise ValueError(f"{path}: invalid observer safety contract")
    rows = value.get("confirmed", [])
    corners = np.asarray([row["best_corners_world"] for row in rows], dtype=np.float64)
    if not len(rows):
        corners = np.empty((0, 8, 3), dtype=np.float64)
    if corners.shape != (len(rows), 8, 3) or not np.isfinite(corners).all():
        raise ValueError(f"{path}: malformed track corners")
    scores = np.asarray([row["best_score"] for row in rows], dtype=np.float64)
    hits = np.asarray([row["hit_count"] for row in rows], dtype=np.int64)
    return corners, scores, hits, value


def audit(args: argparse.Namespace) -> dict:
    scenes = _scenes(args.scene_list.resolve())
    total_gt = 0
    totals: dict[str, dict[str, float]] = {}
    per_scene = {}
    runtime = {"keyframes": 0, "calls": 0, "memory_s": 0.0, "provider_s": 0.0}
    for score in SCORE_THRESHOLDS:
        for hits in HIT_THRESHOLDS:
            totals[f"s{score:.2f}_h{hits}"] = {"candidates": 0, **{
                f"base_{iou:.2f}": 0 for iou in IOU_THRESHOLDS}, **{
                f"union_{iou:.2f}": 0 for iou in IOU_THRESHOLDS}, **{
                f"hits_{iou:.2f}": 0 for iou in IOU_THRESHOLDS}}
    for scene in scenes:
        path = args.diagnostics_root / f"{scene}_tr3d_incremental.json"
        corners, scores, hits, metadata = _diagnostic(path, scene)
        transform = _alignment(args.scans_root.resolve(), scene)
        candidates = _minmax(_transform(corners, transform))
        baseline_corners, _ = _load_b6(args.baseline_root / f"{scene}_boxes.pkl")
        baseline = _minmax(_transform(baseline_corners, transform))
        gt = _gt_boxes(args.ground_truth_root / f"{scene}_bbox.npy")
        base_iou = pairwise_iou(baseline, gt)
        candidate_iou = pairwise_iou(candidates, gt)
        total_gt += len(gt)
        runtime["keyframes"] += int(metadata["keyframes"])
        runtime["calls"] += int(metadata["provider_calls"])
        runtime["memory_s"] += float(metadata["memory_runtime_s"])
        runtime["provider_s"] += float(metadata["provider_runtime_s"])
        scene_rows = {}
        for score in SCORE_THRESHOLDS:
            for min_hits in HIT_THRESHOLDS:
                key = f"s{score:.2f}_h{min_hits}"
                selected = (scores >= score) & (hits >= min_hits)
                selected_iou = candidate_iou[selected]
                union = np.concatenate((base_iou, selected_iou), axis=0)
                row = totals[key]
                row["candidates"] += int(selected.sum())
                local = {"candidates": int(selected.sum())}
                for threshold in IOU_THRESHOLDS:
                    name = f"{threshold:.2f}"
                    base_tp = maximum_cardinality(base_iou, threshold)
                    union_tp = maximum_cardinality(union, threshold)
                    hit_count = int(np.sum(np.max(selected_iou, axis=1, initial=0.0) >= threshold))
                    row[f"base_{name}"] += base_tp
                    row[f"union_{name}"] += union_tp
                    row[f"hits_{name}"] += hit_count
                    local[name] = {"base_tp": base_tp, "union_tp": union_tp,
                                   "novel_tp": union_tp - base_tp,
                                   "candidate_gt_hits": hit_count}
                scene_rows[key] = local
        per_scene[scene] = {"ground_truth": len(gt), "tracks": len(corners),
                            "frontier": scene_rows}
    frontier = {}
    for key, row in totals.items():
        result = {"candidates": int(row["candidates"]), "thresholds": {}}
        for threshold in IOU_THRESHOLDS:
            name = f"{threshold:.2f}"
            base_tp, union_tp = int(row[f"base_{name}"]), int(row[f"union_{name}"])
            result["thresholds"][name] = {
                "baseline_tp": base_tp, "union_tp": union_tp,
                "novel_tp": union_tp - base_tp,
                "delta_recall": (union_tp - base_tp) / max(total_gt, 1),
                "candidate_gt_hits": int(row[f"hits_{name}"]),
                "novel_precision_upper_bound": (union_tp - base_tp) / max(int(row["candidates"]), 1),
            }
        frontier[key] = result
    report = {
        "schema": "boxfusion.tr3d_incremental_online_offline_audit.v1",
        "ground_truth_only_offline_audit": True,
        "inference_ground_truth_access": False,
        "scene_count": len(scenes), "ground_truth_count": total_gt,
        "runtime": {**runtime,
            "amortized_ms_per_keyframe": 1000.0 * (runtime["memory_s"] + runtime["provider_s"]) / max(runtime["keyframes"], 1)},
        "frontier": frontier, "per_scene": per_scene,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--diagnostics-root", type=Path, required=True)
    value.add_argument("--baseline-root", type=Path, required=True)
    value.add_argument("--scene-list", type=Path, required=True)
    value.add_argument("--ground-truth-root", type=Path, required=True)
    value.add_argument("--scans-root", type=Path, required=True)
    value.add_argument("--output", type=Path, required=True)
    return value


if __name__ == "__main__":
    result = audit(parser().parse_args())
    print(json.dumps({"scene_count": result["scene_count"],
                      "ground_truth_count": result["ground_truth_count"],
                      "runtime": result["runtime"],
                      "frontier": result["frontier"]}, indent=2))
