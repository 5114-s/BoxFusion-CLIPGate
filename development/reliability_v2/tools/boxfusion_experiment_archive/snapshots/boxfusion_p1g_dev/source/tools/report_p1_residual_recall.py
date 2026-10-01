#!/usr/bin/env python3
"""Evaluate P1 residual proposals without changing BoxFusion predictions.

This report is intentionally class agnostic.  For each ScanNet scene it:

* aligns frozen B6 corners and P1 world-frame boxes with ``axisAlignment``;
* performs stable score-ordered, one-to-one matching;
* reports B6-only, P1-only and score-merged union recall at IoU
  0.15/0.25/0.50;
* counts P1 true positives that cover GT instances missed by B6;
* reports novel precision, duplicate rate, candidate volume, and observer
  runtime.

The standard P1 observer must leave the prediction pickle unchanged.  Thus
this is a proposal-recall diagnostic, not a claim of AP improvement.
Prediction pickle files are trusted local artifacts and must not come from an
untrusted source.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


REPORT_SCHEMA = "boxfusion.p1_residual_recall_report.v1"
DEFAULT_THRESHOLDS = (0.15, 0.25, 0.50)
_SCENE_PATTERN = re.compile(r"scene\d{4}_\d{2}")
_CORNER_SIGNS = np.asarray(
    [
        [-1.0, -1.0, -1.0],
        [-1.0, -1.0, 1.0],
        [-1.0, 1.0, -1.0],
        [-1.0, 1.0, 1.0],
        [1.0, -1.0, -1.0],
        [1.0, -1.0, 1.0],
        [1.0, 1.0, -1.0],
        [1.0, 1.0, 1.0],
    ],
    dtype=np.float64,
)


@dataclass(frozen=True)
class Predictions:
    corners_world: np.ndarray
    scores: np.ndarray


@dataclass(frozen=True)
class P1Candidates:
    corners_world: np.ndarray
    scores: np.ndarray
    candidate_ids: np.ndarray
    frame_ids: np.ndarray
    runtime_seconds: float
    mutation_enabled: bool | None
    applied_count: int | None


@dataclass(frozen=True)
class MatchResult:
    prediction_to_gt: np.ndarray
    matched_iou: np.ndarray

    @property
    def true_positive_count(self) -> int:
        return int(np.sum(self.prediction_to_gt >= 0))

    @property
    def matched_gt(self) -> np.ndarray:
        return np.unique(self.prediction_to_gt[self.prediction_to_gt >= 0])


def validate_thresholds(values: Iterable[float]) -> tuple[float, ...]:
    thresholds = tuple(float(value) for value in values)
    if not thresholds:
        raise ValueError("at least one IoU threshold is required")
    if (
        not np.isfinite(thresholds).all()
        or any(value <= 0.0 or value > 1.0 for value in thresholds)
    ):
        raise ValueError("IoU thresholds must be finite and in (0,1]")
    if len(set(thresholds)) != len(thresholds):
        raise ValueError("IoU thresholds must be unique")
    return thresholds


def read_scene_ids(path: str | os.PathLike[str]) -> tuple[str, ...]:
    scene_path = Path(path)
    if not scene_path.is_file():
        raise FileNotFoundError(scene_path)
    scenes = tuple(
        line.strip()
        for line in scene_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    if not scenes:
        raise ValueError(f"scene list is empty: {scene_path}")
    duplicates = sorted(
        scene for scene, count in Counter(scenes).items() if count > 1
    )
    if duplicates:
        raise ValueError(f"duplicate scene ids: {duplicates[:8]}")
    invalid = [scene for scene in scenes if _SCENE_PATTERN.fullmatch(scene) is None]
    if invalid:
        raise ValueError(f"invalid ScanNet scene id: {invalid[0]!r}")
    return scenes


def load_axis_alignment(
    scans_root: str | os.PathLike[str], scene_id: str
) -> np.ndarray:
    metadata = Path(scans_root) / scene_id / f"{scene_id}.txt"
    if not metadata.is_file():
        raise FileNotFoundError(metadata)
    values: np.ndarray | None = None
    for line in metadata.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped.startswith("axisAlignment"):
            if "=" not in stripped:
                raise ValueError(f"malformed axisAlignment in {metadata}")
            values = np.fromstring(stripped.split("=", 1)[1], sep=" ")
            break
    if values is None or values.size != 16 or not np.isfinite(values).all():
        raise ValueError(f"invalid or missing axisAlignment in {metadata}")
    transform = values.reshape(4, 4).astype(np.float64, copy=False)
    if not np.allclose(transform[3], [0.0, 0.0, 0.0, 1.0], atol=1e-6):
        raise ValueError(f"axisAlignment is not homogeneous in {metadata}")
    return transform


def center_size_to_corners(boxes: np.ndarray) -> np.ndarray:
    values = np.asarray(boxes, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 6:
        raise ValueError(f"center/size boxes must have shape [N,6], got {values.shape}")
    if not np.isfinite(values).all() or (
        len(values) and np.any(values[:, 3:] <= 0.0)
    ):
        raise ValueError("center/size boxes contain invalid extents")
    return values[:, None, :3] + _CORNER_SIGNS[None] * (
        0.5 * values[:, None, 3:]
    )


def transform_corners(corners: np.ndarray, transform: np.ndarray) -> np.ndarray:
    values = np.asarray(corners, dtype=np.float64)
    matrix = np.asarray(transform, dtype=np.float64)
    if values.ndim != 3 or values.shape[1:] != (8, 3):
        raise ValueError(f"corners must have shape [N,8,3], got {values.shape}")
    if matrix.shape != (4, 4):
        raise ValueError("transform must have shape [4,4]")
    if not np.isfinite(values).all() or not np.isfinite(matrix).all():
        raise ValueError("corners and transform must be finite")
    return values @ matrix[:3, :3].T + matrix[None, None, :3, 3]


def corners_to_minmax(corners: np.ndarray) -> np.ndarray:
    values = np.asarray(corners, dtype=np.float64)
    if values.ndim != 3 or values.shape[1:] != (8, 3):
        raise ValueError(f"corners must have shape [N,8,3], got {values.shape}")
    if not np.isfinite(values).all():
        raise ValueError("corners contain non-finite values")
    if not len(values):
        return np.empty((0, 6), dtype=np.float64)
    return np.concatenate((values.min(axis=1), values.max(axis=1)), axis=1)


def center_size_to_minmax(boxes: np.ndarray) -> np.ndarray:
    return corners_to_minmax(center_size_to_corners(boxes))


def pairwise_aabb_iou(predictions: np.ndarray, targets: np.ndarray) -> np.ndarray:
    pred = np.asarray(predictions, dtype=np.float64)
    gt = np.asarray(targets, dtype=np.float64)
    if pred.ndim != 2 or pred.shape[1] != 6:
        raise ValueError(f"predictions must have shape [N,6], got {pred.shape}")
    if gt.ndim != 2 or gt.shape[1] != 6:
        raise ValueError(f"targets must have shape [M,6], got {gt.shape}")
    if not np.isfinite(pred).all() or not np.isfinite(gt).all():
        raise ValueError("IoU boxes must be finite")
    if not len(pred) or not len(gt):
        return np.zeros((len(pred), len(gt)), dtype=np.float64)
    if np.any(pred[:, 3:] <= pred[:, :3]) or np.any(gt[:, 3:] <= gt[:, :3]):
        raise ValueError("IoU boxes must have positive extents")
    intersection_size = np.maximum(
        np.minimum(pred[:, None, 3:], gt[None, :, 3:])
        - np.maximum(pred[:, None, :3], gt[None, :, :3]),
        0.0,
    )
    intersection = np.prod(intersection_size, axis=2)
    pred_volume = np.prod(pred[:, 3:] - pred[:, :3], axis=1)
    gt_volume = np.prod(gt[:, 3:] - gt[:, :3], axis=1)
    union = pred_volume[:, None] + gt_volume[None, :] - intersection
    return np.divide(
        intersection,
        union,
        out=np.zeros_like(intersection),
        where=union > 0.0,
    )


def load_predictions(path: str | os.PathLike[str]) -> Predictions:
    prediction_path = Path(path)
    if not prediction_path.is_file():
        raise FileNotFoundError(prediction_path)
    with prediction_path.open("rb") as handle:
        payload = pickle.load(handle)  # noqa: S301 - trusted local artifact
    if (
        not isinstance(payload, (list, tuple))
        or len(payload) != 1
        or not isinstance(payload[0], (list, tuple))
    ):
        raise ValueError(f"{prediction_path}: invalid BoxFusion prediction batch")
    corners: list[np.ndarray] = []
    scores: list[float] = []
    for index, item in enumerate(payload[0]):
        if not isinstance(item, (list, tuple)) or len(item) != 3:
            raise ValueError(f"{prediction_path}: invalid detection {index}")
        value = np.asarray(item[1])
        if (
            value.shape != (8, 3)
            or not np.issubdtype(value.dtype, np.number)
            or not np.isfinite(value).all()
        ):
            raise ValueError(f"{prediction_path}: invalid corners at {index}")
        try:
            score = float(item[2])
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"{prediction_path}: invalid score at {index}"
            ) from error
        if not math.isfinite(score):
            raise ValueError(f"{prediction_path}: non-finite score at {index}")
        corners.append(np.asarray(value, dtype=np.float64))
        scores.append(score)
    return Predictions(
        corners_world=(
            np.stack(corners, axis=0)
            if corners
            else np.empty((0, 8, 3), dtype=np.float64)
        ),
        scores=np.asarray(scores, dtype=np.float64),
    )


def load_gt_boxes(path: str | os.PathLike[str]) -> np.ndarray:
    gt_path = Path(path)
    if not gt_path.is_file():
        raise FileNotFoundError(gt_path)
    payload = np.load(gt_path, allow_pickle=False)
    if payload.ndim != 2 or payload.shape[1] < 6:
        raise ValueError(f"{gt_path}: GT boxes must have shape [N,>=6]")
    boxes = np.asarray(payload[:, :6], dtype=np.float64)
    return center_size_to_minmax(boxes)


def _scalar(value: np.ndarray, name: str, path: Path) -> Any:
    array = np.asarray(value)
    if array.shape != () or array.dtype.hasobject:
        raise ValueError(f"{path}: {name} must be a non-object scalar")
    return array.item()


def _summary_runtime(archive: Mapping[str, np.ndarray], path: Path) -> float:
    direct_names = (
        "p1_runtime_s",
        "p1_runtime_seconds",
        "p1_observer_seconds",
        "p1_candidate_runtime_s",
    )
    for name in direct_names:
        if name in archive:
            values = np.asarray(archive[name], dtype=np.float64)
            if values.ndim > 1 or not np.isfinite(values).all() or np.any(values < 0):
                raise ValueError(f"{path}: invalid {name}")
            return float(np.sum(values))
    step_names = (
        "p1_step_voxelize_seconds",
        "p1_step_head_seconds",
        "p1_step_nms_seconds",
    )
    if any(name in archive for name in step_names):
        total = 0.0
        lengths: set[int] = set()
        for name in step_names:
            if name not in archive:
                continue
            values = np.asarray(archive[name], dtype=np.float64)
            if (
                values.ndim != 1
                or not np.isfinite(values).all()
                or np.any(values < 0.0)
            ):
                raise ValueError(f"{path}: invalid {name}")
            lengths.add(len(values))
            total += float(np.sum(values))
        if len(lengths) > 1:
            raise ValueError(f"{path}: P1 step runtime arrays disagree in length")
        return total
    if "summary_json" not in archive:
        return 0.0
    raw = _scalar(archive["summary_json"], "summary_json", path)
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    if not isinstance(raw, str):
        raise TypeError(f"{path}: summary_json must be a string")
    try:
        summary = json.loads(raw)
    except json.JSONDecodeError as error:
        raise ValueError(f"{path}: malformed summary_json") from error
    candidates: list[Any] = []
    for name in direct_names:
        if name in summary:
            candidates.append(summary[name])
    p1_summary = summary.get("p1")
    if isinstance(p1_summary, Mapping):
        for name in ("runtime_s", "runtime_seconds", "observer_seconds"):
            if name in p1_summary:
                candidates.append(p1_summary[name])
    if not candidates:
        return 0.0
    value = float(candidates[0])
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(f"{path}: invalid P1 runtime in summary_json")
    return value


def load_p1_candidates(
    path: str | os.PathLike[str], *, expected_scene_id: str | None = None
) -> P1Candidates:
    diagnostic_path = Path(path)
    if not diagnostic_path.is_file():
        raise FileNotFoundError(diagnostic_path)
    with np.load(diagnostic_path, allow_pickle=False) as archive_obj:
        archive = {
            key: np.array(archive_obj[key], copy=True)
            for key in archive_obj.files
        }
    if "scene_id" in archive and expected_scene_id is not None:
        value = _scalar(archive["scene_id"], "scene_id", diagnostic_path)
        if isinstance(value, bytes):
            value = value.decode("utf-8")
        if value != expected_scene_id:
            raise ValueError(
                f"{diagnostic_path}: scene_id {value!r} != {expected_scene_id!r}"
            )
    has_boxes = "p1_candidate_boxes" in archive
    has_corners = "p1_candidate_corners" in archive
    if not has_boxes and not has_corners:
        raise ValueError(
            f"{diagnostic_path}: missing p1_candidate_boxes/corners"
        )
    corners: np.ndarray
    if has_corners:
        corners = np.asarray(archive["p1_candidate_corners"])
        if (
            corners.ndim != 3
            or corners.shape[1:] != (8, 3)
            or not np.issubdtype(corners.dtype, np.floating)
            or not np.isfinite(corners).all()
        ):
            raise ValueError(
                f"{diagnostic_path}: p1_candidate_corners must be finite [C,8,3]"
            )
        corners = np.asarray(corners, dtype=np.float64)
    else:
        boxes = np.asarray(archive["p1_candidate_boxes"])
        if (
            boxes.ndim != 2
            or boxes.shape[1] != 6
            or not np.issubdtype(boxes.dtype, np.floating)
            or not np.isfinite(boxes).all()
        ):
            raise ValueError(
                f"{diagnostic_path}: p1_candidate_boxes must be finite [C,6]"
            )
        corners = center_size_to_corners(boxes)
    if has_boxes and has_corners:
        boxes = np.asarray(archive["p1_candidate_boxes"], dtype=np.float64)
        expected = corners_to_minmax(center_size_to_corners(boxes))
        observed = corners_to_minmax(corners)
        if boxes.shape != (len(corners), 6) or not np.allclose(
            expected, observed, rtol=1e-5, atol=1e-5
        ):
            raise ValueError(
                f"{diagnostic_path}: P1 box and corner aliases disagree"
            )
    count = len(corners)
    if "p1_candidate_scores" not in archive:
        raise ValueError(f"{diagnostic_path}: missing p1_candidate_scores")
    scores = np.asarray(archive["p1_candidate_scores"])
    if (
        scores.shape != (count,)
        or not np.issubdtype(scores.dtype, np.floating)
        or not np.isfinite(scores).all()
    ):
        raise ValueError(
            f"{diagnostic_path}: p1_candidate_scores must be finite [C]"
        )
    if np.any((scores < 0.0) | (scores > 1.0)):
        raise ValueError(f"{diagnostic_path}: candidate scores must be in [0,1]")
    if "p1_candidate_objectness" in archive:
        objectness = np.asarray(archive["p1_candidate_objectness"])
        if (
            objectness.shape != scores.shape
            or objectness.dtype.hasobject
            or not np.issubdtype(objectness.dtype, np.floating)
            or not np.isfinite(objectness).all()
            or not np.array_equal(objectness, scores)
        ):
            raise ValueError(
                f"{diagnostic_path}: score/objectness aliases disagree"
            )
    ids = np.asarray(
        archive.get("p1_candidate_ids", np.arange(count, dtype=np.int64))
    )
    frames = np.asarray(
        archive.get("p1_candidate_frame_ids", np.full(count, -1, dtype=np.int64))
    )
    if (
        ids.shape != (count,)
        or ids.dtype.hasobject
        or ids.dtype.kind not in {"i", "u", "U", "S"}
    ):
        raise ValueError(
            f"{diagnostic_path}: p1_candidate_ids must be non-object [C]"
        )
    if frames.shape != (count,) or not np.issubdtype(
        frames.dtype, np.integer
    ):
        raise ValueError(
            f"{diagnostic_path}: p1_candidate_frame_ids must be integer [C]"
        )
    if len(np.unique(ids)) != len(ids):
        raise ValueError(f"{diagnostic_path}: p1_candidate_ids are not unique")
    mutation_enabled = None
    if "p1_mutation_enabled" in archive:
        mutation_enabled = bool(
            _scalar(
                archive["p1_mutation_enabled"],
                "p1_mutation_enabled",
                diagnostic_path,
            )
        )
    applied_count = None
    if "p1_applied_count" in archive:
        applied_count = int(
            _scalar(
                archive["p1_applied_count"], "p1_applied_count", diagnostic_path
            )
        )
    return P1Candidates(
        corners_world=np.ascontiguousarray(corners),
        scores=np.asarray(scores, dtype=np.float64),
        candidate_ids=np.asarray(ids),
        frame_ids=np.asarray(frames, dtype=np.int64),
        runtime_seconds=_summary_runtime(archive, diagnostic_path),
        mutation_enabled=mutation_enabled,
        applied_count=applied_count,
    )


def score_ordered_match(
    iou: np.ndarray,
    scores: np.ndarray,
    threshold: float,
    *,
    allowed_gt: np.ndarray | None = None,
    tie_break_ids: np.ndarray | None = None,
) -> MatchResult:
    """Stable score-ordered greedy matching, identical in spirit to ScanNet AP."""

    overlaps = np.asarray(iou, dtype=np.float64)
    score_values = np.asarray(scores, dtype=np.float64)
    if overlaps.ndim != 2 or score_values.shape != (overlaps.shape[0],):
        raise ValueError("iou must be [P,G] and scores must be [P]")
    if not np.isfinite(overlaps).all() or not np.isfinite(score_values).all():
        raise ValueError("IoU and scores must be finite")
    if allowed_gt is None:
        allowed = np.ones(overlaps.shape[1], dtype=np.bool_)
    else:
        allowed = np.asarray(allowed_gt, dtype=np.bool_)
        if allowed.shape != (overlaps.shape[1],):
            raise ValueError("allowed_gt must have shape [G]")
    prediction_to_gt = np.full(overlaps.shape[0], -1, dtype=np.int64)
    matched_iou = np.zeros(overlaps.shape[0], dtype=np.float64)
    available = allowed.copy()
    if tie_break_ids is None:
        order = np.argsort(-score_values, kind="stable")
    else:
        ids = np.asarray(tie_break_ids)
        if (
            ids.shape != (overlaps.shape[0],)
            or ids.dtype.hasobject
            or ids.dtype.kind not in {"i", "u", "U", "S"}
        ):
            raise ValueError("tie_break_ids must be non-object scalar [P]")
        # lexsort uses its final key as primary: score descending first,
        # deterministic candidate id second.
        order = np.lexsort((ids, -score_values))
    for prediction_index in order:
        if not np.any(available):
            break
        candidates = np.flatnonzero(available)
        candidate_iou = overlaps[int(prediction_index), candidates]
        best_local = int(np.argmax(candidate_iou))
        gt_index = int(candidates[best_local])
        best_iou = float(candidate_iou[best_local])
        # Upstream ScanNet evaluator uses strict greater-than.
        if best_iou > float(threshold):
            prediction_to_gt[int(prediction_index)] = gt_index
            matched_iou[int(prediction_index)] = best_iou
            available[gt_index] = False
    return MatchResult(prediction_to_gt, matched_iou)


def _novel_candidate_metrics(
    baseline_iou: np.ndarray,
    baseline_scores: np.ndarray,
    candidate_iou: np.ndarray,
    candidate_scores: np.ndarray,
    threshold: float,
    candidate_ids: np.ndarray | None = None,
) -> dict[str, Any]:
    baseline = score_ordered_match(baseline_iou, baseline_scores, threshold)
    baseline_covered = np.zeros(baseline_iou.shape[1], dtype=np.bool_)
    baseline_covered[baseline.matched_gt] = True
    novel = score_ordered_match(
        candidate_iou,
        candidate_scores,
        threshold,
        allowed_gt=~baseline_covered,
        tie_break_ids=candidate_ids,
    )
    novel_tp = novel.true_positive_count
    baseline_duplicates = 0
    p1_duplicates = 0
    claimed_novel = set(int(index) for index in novel.matched_gt.tolist())
    for candidate_index in range(len(candidate_iou)):
        if novel.prediction_to_gt[candidate_index] >= 0:
            continue
        overlaps = candidate_iou[candidate_index]
        baseline_gt = np.flatnonzero(
            baseline_covered & (overlaps > float(threshold))
        )
        if len(baseline_gt):
            baseline_duplicates += 1
            continue
        if any(
            overlaps[gt_index] > float(threshold)
            for gt_index in claimed_novel
        ):
            p1_duplicates += 1
    count = len(candidate_iou)
    return {
        "novel_true_positives": int(novel_tp),
        "novel_precision": float(novel_tp / count) if count else 0.0,
        "baseline_duplicate_count": int(baseline_duplicates),
        "p1_duplicate_count": int(p1_duplicates),
        "duplicate_count": int(baseline_duplicates + p1_duplicates),
        "duplicate_rate": (
            float((baseline_duplicates + p1_duplicates) / count)
            if count
            else 0.0
        ),
        "baseline_matched_gt": baseline_covered,
    }


def evaluate_scene(
    *,
    baseline_boxes: np.ndarray,
    baseline_scores: np.ndarray,
    candidate_boxes: np.ndarray,
    candidate_scores: np.ndarray,
    gt_boxes: np.ndarray,
    thresholds: Sequence[float],
    candidate_ids: np.ndarray | None = None,
) -> dict[str, Any]:
    baseline_iou = pairwise_aabb_iou(baseline_boxes, gt_boxes)
    candidate_iou = pairwise_aabb_iou(candidate_boxes, gt_boxes)
    candidate_baseline_iou = pairwise_aabb_iou(
        candidate_boxes, baseline_boxes
    )
    union_iou = np.concatenate((baseline_iou, candidate_iou), axis=0)
    union_scores = np.concatenate((baseline_scores, candidate_scores), axis=0)
    by_threshold: dict[str, Any] = {}
    for threshold in thresholds:
        baseline = score_ordered_match(
            baseline_iou, baseline_scores, float(threshold)
        )
        candidate = score_ordered_match(
            candidate_iou,
            candidate_scores,
            float(threshold),
            tie_break_ids=candidate_ids,
        )
        union = score_ordered_match(union_iou, union_scores, float(threshold))
        novel = _novel_candidate_metrics(
            baseline_iou,
            baseline_scores,
            candidate_iou,
            candidate_scores,
            float(threshold),
            candidate_ids,
        )
        key = f"{float(threshold):.2f}"
        by_threshold[key] = {
            "ground_truth_count": int(len(gt_boxes)),
            "b6_true_positives": baseline.true_positive_count,
            "p1_true_positives": candidate.true_positive_count,
            "union_true_positives": union.true_positive_count,
            "novel_true_positives": novel["novel_true_positives"],
            "baseline_duplicate_count": novel["baseline_duplicate_count"],
            "p1_duplicate_count": novel["p1_duplicate_count"],
            "b6_geometric_duplicate_count": int(
                np.sum(
                    np.max(candidate_baseline_iou, axis=1, initial=0.0)
                    > float(threshold)
                )
            ),
        }
    return {
        "baseline_predictions": int(len(baseline_boxes)),
        "p1_candidates": int(len(candidate_boxes)),
        "ground_truth_count": int(len(gt_boxes)),
        "thresholds": by_threshold,
    }


def evaluate(
    *,
    scenes: Sequence[str],
    prediction_root: str | os.PathLike[str],
    diagnostics_root: str | os.PathLike[str],
    gt_root: str | os.PathLike[str],
    scans_root: str | os.PathLike[str],
    thresholds: Sequence[float] = DEFAULT_THRESHOLDS,
) -> dict[str, Any]:
    thresholds = validate_thresholds(thresholds)
    prediction_directory = Path(prediction_root)
    diagnostic_directory = Path(diagnostics_root)
    gt_directory = Path(gt_root)
    scans_directory = Path(scans_root)
    for role, root in (
        ("prediction", prediction_directory),
        ("diagnostics", diagnostic_directory),
        ("ground-truth", gt_directory),
        ("scans", scans_directory),
    ):
        if not root.is_dir():
            raise FileNotFoundError(f"{role} root not found: {root}")

    totals = {
        f"{value:.2f}": {
            "b6_true_positives": 0,
            "p1_true_positives": 0,
            "union_true_positives": 0,
            "novel_true_positives": 0,
            "baseline_duplicate_count": 0,
            "p1_duplicate_count": 0,
            "b6_geometric_duplicate_count": 0,
        }
        for value in thresholds
    }
    total_gt = 0
    total_baseline = 0
    total_candidates = 0
    total_runtime = 0.0
    all_candidate_scores: list[np.ndarray] = []
    unsafe_scenes: list[str] = []
    per_scene: dict[str, Any] = {}
    for scene_id in scenes:
        baseline = load_predictions(
            prediction_directory / f"{scene_id}_boxes.pkl"
        )
        candidates = load_p1_candidates(
            diagnostic_directory / f"{scene_id}_tracks.npz",
            expected_scene_id=scene_id,
        )
        alignment = load_axis_alignment(scans_directory, scene_id)
        baseline_boxes = corners_to_minmax(
            transform_corners(baseline.corners_world, alignment)
        )
        candidate_boxes = corners_to_minmax(
            transform_corners(candidates.corners_world, alignment)
        )
        gt_boxes = load_gt_boxes(gt_directory / f"{scene_id}_bbox.npy")
        scene_report = evaluate_scene(
            baseline_boxes=baseline_boxes,
            baseline_scores=baseline.scores,
            candidate_boxes=candidate_boxes,
            candidate_scores=candidates.scores,
            gt_boxes=gt_boxes,
            thresholds=thresholds,
            candidate_ids=candidates.candidate_ids,
        )
        scene_report["p1_runtime_seconds"] = candidates.runtime_seconds
        scene_report["p1_mutation_enabled"] = candidates.mutation_enabled
        scene_report["p1_applied_count"] = candidates.applied_count
        per_scene[scene_id] = scene_report
        if candidates.mutation_enabled is True or (
            candidates.applied_count is not None
            and candidates.applied_count != 0
        ):
            unsafe_scenes.append(scene_id)
        total_gt += len(gt_boxes)
        total_baseline += len(baseline_boxes)
        total_candidates += len(candidate_boxes)
        total_runtime += candidates.runtime_seconds
        all_candidate_scores.append(candidates.scores)
        for key, row in scene_report["thresholds"].items():
            for field in totals[key]:
                totals[key][field] += int(row[field])

    threshold_report: dict[str, Any] = {}
    for key, row in totals.items():
        candidate_count = total_candidates
        assignment_duplicate_count = (
            row["baseline_duplicate_count"] + row["p1_duplicate_count"]
        )
        duplicate_count = row["b6_geometric_duplicate_count"]
        threshold_report[key] = {
            **row,
            "ground_truth_count": int(total_gt),
            "b6_recall": float(row["b6_true_positives"] / max(total_gt, 1)),
            "p1_recall": float(row["p1_true_positives"] / max(total_gt, 1)),
            "union_recall": float(row["union_true_positives"] / max(total_gt, 1)),
            "union_recall_gain": float(
                (row["union_true_positives"] - row["b6_true_positives"])
                / max(total_gt, 1)
            ),
            "novel_recall_gain": float(
                row["novel_true_positives"] / max(total_gt, 1)
            ),
            "novel_precision": (
                float(row["novel_true_positives"] / candidate_count)
                if candidate_count
                else 0.0
            ),
            "duplicate_count": int(duplicate_count),
            "duplicate_rate": (
                float(duplicate_count / candidate_count)
                if candidate_count
                else 0.0
            ),
            "assignment_duplicate_count": int(assignment_duplicate_count),
            "assignment_duplicate_rate": (
                float(assignment_duplicate_count / candidate_count)
                if candidate_count
                else 0.0
            ),
        }
    gain_025 = threshold_report.get("0.25", {}).get(
        "novel_recall_gain", 0.0
    )
    gain_050 = threshold_report.get("0.50", {}).get(
        "novel_recall_gain", 0.0
    )
    score_values = (
        np.concatenate(all_candidate_scores)
        if all_candidate_scores
        else np.empty(0, dtype=np.float64)
    )
    score_quantiles = (
        {
            name: float(value)
            for name, value in zip(
                ("q10", "q25", "q50", "q75", "q90"),
                np.quantile(score_values, (0.10, 0.25, 0.50, 0.75, 0.90)),
            )
        }
        if len(score_values)
        else {name: None for name in ("q10", "q25", "q50", "q75", "q90")}
    )
    source_thresholds = {
        source: {
            key: {
                "true_positives": int(row[f"{field}_true_positives"]),
                "ground_truth_count": int(total_gt),
                "recall": float(row[f"{field}_recall"]),
            }
            for key, row in threshold_report.items()
        }
        for source, field in (("baseline", "b6"), ("b6", "b6"), ("p1", "p1"), ("union", "union"))
    }
    return {
        "schema": REPORT_SCHEMA,
        "matching_contract": (
            "class-agnostic, stable score-descending, strict IoU > threshold, "
            "one-to-one per scene"
        ),
        "observer_only": len(unsafe_scenes) == 0,
        "unsafe_scenes": unsafe_scenes,
        "scene_count": int(len(scenes)),
        "ground_truth_count": int(total_gt),
        "baseline_prediction_count": int(total_baseline),
        "p1_candidate_count": int(total_candidates),
        "p1_objectness_quantiles": score_quantiles,
        "candidates_per_scene": float(total_candidates / max(len(scenes), 1)),
        "p1_runtime_seconds": float(total_runtime),
        "p1_runtime_seconds_per_scene": float(
            total_runtime / max(len(scenes), 1)
        ),
        "thresholds": threshold_report,
        "baseline": {
            "prediction_count": int(total_baseline),
            "thresholds": source_thresholds["baseline"],
        },
        "b6": {
            "prediction_count": int(total_baseline),
            "thresholds": source_thresholds["b6"],
        },
        "p1": {
            "candidate_count": int(total_candidates),
            "thresholds": source_thresholds["p1"],
        },
        "union": {
            "prediction_count": int(total_baseline + total_candidates),
            "thresholds": source_thresholds["union"],
        },
        "development_stop_rule": {
            "required_novel_recall_gain_at_0p25": 0.03,
            "required_novel_recall_gain_at_0p50": 0.01,
            "passes": bool(gain_025 >= 0.03 and gain_050 >= 0.01),
            "note": (
                "Apply this gate on train-only development scenes, never tune "
                "it on the final validation report."
            ),
        },
        "per_scene": per_scene,
    }


def build_report(
    *,
    diagnostics_root: str | os.PathLike[str],
    gt_root: str | os.PathLike[str],
    scans_root: str | os.PathLike[str],
    scene_list: str | os.PathLike[str],
    prediction_root: str | os.PathLike[str] | None = None,
    pred_root: str | os.PathLike[str] | None = None,
    thresholds: Sequence[float] = DEFAULT_THRESHOLDS,
) -> dict[str, Any]:
    """Compatibility wrapper around :func:`evaluate` for test/CLI callers."""

    if prediction_root is None:
        prediction_root = pred_root
    elif pred_root is not None and Path(prediction_root) != Path(pred_root):
        raise ValueError("prediction_root and pred_root disagree")
    if prediction_root is None:
        raise ValueError("a frozen B6 prediction root is required")
    return evaluate(
        scenes=read_scene_ids(scene_list),
        prediction_root=prediction_root,
        diagnostics_root=diagnostics_root,
        gt_root=gt_root,
        scans_root=scans_root,
        thresholds=thresholds,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-list", required=True, type=Path)
    parser.add_argument("--prediction-root", required=True, type=Path)
    parser.add_argument("--diagnostics-root", required=True, type=Path)
    parser.add_argument("--gt-root", required=True, type=Path)
    parser.add_argument("--scans-root", required=True, type=Path)
    parser.add_argument(
        "--thresholds", nargs="+", type=float, default=DEFAULT_THRESHOLDS
    )
    parser.add_argument("--output", "--output-json", dest="output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = evaluate(
        scenes=read_scene_ids(args.scene_list),
        prediction_root=args.prediction_root,
        diagnostics_root=args.diagnostics_root,
        gt_root=args.gt_root,
        scans_root=args.scans_root,
        thresholds=args.thresholds,
    )
    rendered = json.dumps(report, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_name(args.output.name + ".tmp")
        temporary.write_text(rendered + "\n", encoding="utf-8")
        os.replace(temporary, args.output)
    print(rendered)
    return 0 if report["observer_only"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
