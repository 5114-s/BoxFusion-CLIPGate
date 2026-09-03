#!/usr/bin/env python3
"""Diagnose recall, duplicates, and score-ranking headroom.

This tool is deliberately separate from online inference.  It reads ScanNet
ground truth only to estimate upper bounds and must never be imported by the
runtime pipeline.
"""

from __future__ import annotations

import argparse
import json
import pickle
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy.optimize import linear_sum_assignment


DEFAULT_THRESHOLDS = (0.15, 0.25, 0.50)


@dataclass
class ThresholdReport:
    threshold: float
    predictions: int
    ground_truth: int
    matched: int
    recall: float
    final_precision: float
    average_precision: float
    oracle_rank_ap: float
    duplicate_or_false_positive: int
    ap_recall_efficiency: float
    oracle_gain: float


def read_scene_ids(path: Path) -> list[str]:
    scenes = [line.strip() for line in path.read_text().splitlines()]
    scenes = [scene for scene in scenes if scene]
    if not scenes:
        raise ValueError(f"No scenes found in {path}")
    if len(set(scenes)) != len(scenes):
        raise ValueError(f"Duplicate scene ids in {path}")
    return scenes


def load_axis_alignment(scan_root: Path, scene: str) -> np.ndarray:
    metadata = scan_root / scene / f"{scene}.txt"
    if not metadata.is_file():
        raise FileNotFoundError(metadata)
    for line in metadata.read_text().splitlines():
        if line.startswith("axisAlignment"):
            values = np.fromstring(line.split("=", 1)[1], sep=" ")
            if values.size != 16 or not np.isfinite(values).all():
                raise ValueError(f"Invalid axisAlignment in {metadata}")
            return values.reshape(4, 4)
    raise ValueError(f"axisAlignment missing from {metadata}")


def corners_to_minmax(corners: np.ndarray) -> np.ndarray:
    corners = np.asarray(corners, dtype=np.float64)
    if corners.ndim != 3 or corners.shape[1:] != (8, 3):
        raise ValueError(f"Expected corners [N,8,3], got {corners.shape}")
    if not np.isfinite(corners).all():
        raise ValueError("Prediction corners contain non-finite values")
    return np.concatenate((corners.min(axis=1), corners.max(axis=1)), axis=1)


def center_size_to_minmax(boxes: np.ndarray) -> np.ndarray:
    boxes = np.asarray(boxes, dtype=np.float64)
    if boxes.ndim != 2 or boxes.shape[1] < 6:
        raise ValueError(f"Expected GT boxes [N,>=6], got {boxes.shape}")
    boxes = boxes[:, :6]
    if not np.isfinite(boxes).all() or np.any(boxes[:, 3:6] <= 0):
        raise ValueError("GT boxes contain invalid values")
    half = boxes[:, 3:6] * 0.5
    return np.concatenate((boxes[:, :3] - half, boxes[:, :3] + half), axis=1)


def transform_corners(corners: np.ndarray, transform: np.ndarray) -> np.ndarray:
    rotation = transform[:3, :3]
    translation = transform[:3, 3]
    return corners @ rotation.T + translation


def pairwise_aabb_iou(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    pred = np.asarray(pred, dtype=np.float64)
    gt = np.asarray(gt, dtype=np.float64)
    if pred.size == 0 or gt.size == 0:
        return np.zeros((len(pred), len(gt)), dtype=np.float64)
    intersection_min = np.maximum(pred[:, None, :3], gt[None, :, :3])
    intersection_max = np.minimum(pred[:, None, 3:], gt[None, :, 3:])
    intersection_size = np.maximum(intersection_max - intersection_min, 0.0)
    intersection = np.prod(intersection_size, axis=2)
    pred_volume = np.prod(np.maximum(pred[:, 3:] - pred[:, :3], 0.0), axis=1)
    gt_volume = np.prod(np.maximum(gt[:, 3:] - gt[:, :3], 0.0), axis=1)
    union = pred_volume[:, None] + gt_volume[None, :] - intersection
    return np.divide(
        intersection,
        union,
        out=np.zeros_like(intersection),
        where=union > 0,
    )


def maximum_matches(iou: np.ndarray, threshold: float) -> tuple[np.ndarray, np.ndarray]:
    if iou.size == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    # A large invalid cost makes threshold-invalid assignments unattractive.
    valid = iou >= threshold
    cost = np.where(valid, 1.0 - iou, 1e6)
    pred_index, gt_index = linear_sum_assignment(cost)
    keep = valid[pred_index, gt_index]
    return pred_index[keep], gt_index[keep]


def voc_ap(recall: np.ndarray, precision: np.ndarray) -> float:
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))
    for index in range(mpre.size - 1, 0, -1):
        mpre[index - 1] = max(mpre[index - 1], mpre[index])
    changing = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[changing + 1] - mrec[changing]) * mpre[changing + 1]))


def ranked_metrics(
    records: Iterable[tuple[float, bool]],
    ground_truth_count: int,
) -> tuple[float, float, float]:
    ordered = sorted(records, key=lambda item: item[0], reverse=True)
    if not ordered or ground_truth_count == 0:
        return 0.0, 0.0, 0.0
    true_positive = np.asarray([match for _, match in ordered], dtype=np.float64)
    false_positive = 1.0 - true_positive
    cumulative_tp = np.cumsum(true_positive)
    cumulative_fp = np.cumsum(false_positive)
    recall = cumulative_tp / float(ground_truth_count)
    precision = cumulative_tp / np.maximum(cumulative_tp + cumulative_fp, 1e-12)
    return voc_ap(recall, precision), float(recall[-1]), float(precision[-1])


def score_scene(
    iou: np.ndarray,
    scores: np.ndarray,
    threshold: float,
) -> tuple[list[tuple[float, bool]], list[tuple[float, bool]], int]:
    # Real-score greedy matching follows standard detection evaluation.
    order = np.argsort(-scores, kind="stable")
    used_gt: set[int] = set()
    real_records: list[tuple[float, bool]] = []
    for pred_index in order:
        if iou.shape[1] == 0:
            real_records.append((float(scores[pred_index]), False))
            continue
        gt_index = int(np.argmax(iou[pred_index]))
        matched = bool(
            iou[pred_index, gt_index] >= threshold
            and gt_index not in used_gt
        )
        if matched:
            used_gt.add(gt_index)
        real_records.append((float(scores[pred_index]), matched))

    # Maximum-cardinality matching, followed by an intentionally ideal rank.
    matched_pred, _ = maximum_matches(iou, threshold)
    matched_set = set(matched_pred.tolist())
    oracle_records = [
        (
            2.0 + float(iou[index].max(initial=0.0))
            if index in matched_set
            else float(scores[index]) * 1e-3,
            index in matched_set,
        )
        for index in range(len(scores))
    ]
    return real_records, oracle_records, len(matched_set)


def load_scene_predictions(path: Path) -> tuple[np.ndarray, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    detections = payload[0] if payload else []
    if not detections:
        return (
            np.empty((0, 8, 3), dtype=np.float64),
            np.empty(0, dtype=np.float64),
        )
    corners = np.stack([np.asarray(item[1], dtype=np.float64) for item in detections])
    scores = np.asarray([float(item[2]) for item in detections], dtype=np.float64)
    if corners.shape[0] != scores.shape[0] or not np.isfinite(scores).all():
        raise ValueError(f"Invalid predictions in {path}")
    return corners, scores


def analyze(args: argparse.Namespace) -> dict:
    scenes = read_scene_ids(args.scene_list)
    threshold_records = {
        threshold: {"real": [], "oracle": [], "matched": 0}
        for threshold in args.thresholds
    }
    total_predictions = 0
    total_ground_truth = 0

    for scene in scenes:
        corners, scores = load_scene_predictions(
            args.pred_root / f"{scene}_boxes.pkl"
        )
        if args.constant_score:
            scores = np.ones_like(scores)
        transform = load_axis_alignment(args.scan_root, scene)
        pred_minmax = corners_to_minmax(transform_corners(corners, transform))
        gt_payload = np.load(args.gt_root / f"{scene}_bbox.npy")
        gt_minmax = center_size_to_minmax(gt_payload)
        iou = pairwise_aabb_iou(pred_minmax, gt_minmax)

        total_predictions += len(scores)
        total_ground_truth += len(gt_minmax)
        for threshold in args.thresholds:
            real, oracle, matched = score_scene(iou, scores, threshold)
            threshold_records[threshold]["real"].extend(real)
            threshold_records[threshold]["oracle"].extend(oracle)
            threshold_records[threshold]["matched"] += matched

    reports = []
    for threshold in args.thresholds:
        values = threshold_records[threshold]
        ap, recall, precision = ranked_metrics(
            values["real"], total_ground_truth
        )
        oracle_ap, _, _ = ranked_metrics(
            values["oracle"], total_ground_truth
        )
        matched = int(values["matched"])
        reports.append(
            ThresholdReport(
                threshold=threshold,
                predictions=total_predictions,
                ground_truth=total_ground_truth,
                matched=matched,
                recall=recall,
                final_precision=precision,
                average_precision=ap,
                oracle_rank_ap=oracle_ap,
                duplicate_or_false_positive=total_predictions - matched,
                ap_recall_efficiency=ap / recall if recall > 0 else 0.0,
                oracle_gain=oracle_ap - ap,
            )
        )

    return {
        "scene_count": len(scenes),
        "score_mode": "constant" if args.constant_score else "real",
        "reports": [asdict(report) for report in reports],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred-root", type=Path, required=True)
    parser.add_argument("--gt-root", type=Path, required=True)
    parser.add_argument("--scan-root", type=Path, required=True)
    parser.add_argument("--scene-list", type=Path, required=True)
    parser.add_argument(
        "--thresholds",
        type=lambda value: tuple(float(item) for item in value.split(",")),
        default=DEFAULT_THRESHOLDS,
    )
    parser.add_argument("--constant-score", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if any(threshold <= 0 or threshold > 1 for threshold in args.thresholds):
        parser.error("--thresholds must be in (0,1]")
    return args


def main() -> int:
    args = parse_args()
    report = analyze(args)
    rendered = json.dumps(report, indent=2, sort_keys=True)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
