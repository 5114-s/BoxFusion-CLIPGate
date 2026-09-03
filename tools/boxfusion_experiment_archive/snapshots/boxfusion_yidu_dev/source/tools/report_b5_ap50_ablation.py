#!/usr/bin/env python3
"""Offline paired diagnostics for a score-preserving B5 box-refiner run.

The report pairs identity and candidate detections by their exported list
position, uses the identity box to choose a ScanNet GT target, and measures
the candidate against that same target.  It deliberately does not run an
inference model or import PyTorch.

Prediction pickle files are trusted local experiment artifacts with the
BoxFusion layout ``[[ (label, corners[8,3], score), ... ]]``.  Do not use
this tool on untrusted pickle files.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


SCENE_PATTERN = re.compile(r"scene\d{4}_\d{2}")
DEFAULT_THRESHOLDS = (0.15, 0.25, 0.50)
PREDICTION_SUFFIX = "_boxes.pkl"
DIAGNOSTIC_SUFFIX = "_tracks.npz"
REPORT_SCHEMA = "boxfusion.b5_ap50_paired_report"
REPORT_FORMAT_VERSION = 1


@dataclass(frozen=True)
class Detection:
    label: Any
    corners: np.ndarray
    score: float
    raw_label: Any
    raw_corners: np.ndarray
    raw_score: Any


@dataclass(frozen=True)
class ScenePredictions:
    detections: tuple[Detection, ...]


@dataclass(frozen=True)
class SceneDiagnostics:
    result_indices: np.ndarray
    refit_applied: np.ndarray
    summary: Mapping[str, Any]


def read_scene_ids(path: str | os.PathLike[str]) -> list[str]:
    """Read a non-empty, duplicate-free ScanNet scene list."""

    scene_path = Path(path)
    if not scene_path.is_file():
        raise FileNotFoundError(scene_path)
    scenes = [
        line.strip()
        for line in scene_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not scenes:
        raise ValueError(f"scene list is empty: {scene_path}")
    invalid = [scene for scene in scenes if SCENE_PATTERN.fullmatch(scene) is None]
    if invalid:
        raise ValueError(f"invalid ScanNet scene id: {invalid[0]!r}")
    if len(set(scenes)) != len(scenes):
        raise ValueError(f"scene list contains duplicates: {scene_path}")
    return scenes


def _python_scalar(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    return value


def load_predictions(path: str | os.PathLike[str]) -> ScenePredictions:
    """Load and validate one trusted BoxFusion prediction pickle."""

    prediction_path = Path(path)
    if not prediction_path.is_file():
        raise FileNotFoundError(prediction_path)
    with prediction_path.open("rb") as handle:
        payload = pickle.load(handle)
    if (
        not isinstance(payload, (list, tuple))
        or len(payload) != 1
        or not isinstance(payload[0], (list, tuple))
    ):
        raise ValueError(
            f"{prediction_path} must contain one BoxFusion detection batch"
        )

    detections: list[Detection] = []
    for index, item in enumerate(payload[0]):
        if not isinstance(item, (list, tuple)) or len(item) != 3:
            raise ValueError(
                f"{prediction_path}: detection {index} must be "
                "(label, corners, score)"
            )
        raw_label, raw_corners, raw_score = item
        corners = np.asarray(raw_corners)
        if (
            corners.shape != (8, 3)
            or not np.issubdtype(corners.dtype, np.number)
            or not np.isfinite(corners).all()
        ):
            raise ValueError(
                f"{prediction_path}: detection {index} has invalid corners"
            )
        try:
            score = float(raw_score)
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"{prediction_path}: detection {index} has invalid score"
            ) from error
        if not math.isfinite(score):
            raise ValueError(
                f"{prediction_path}: detection {index} has non-finite score"
            )
        detections.append(
            Detection(
                label=_python_scalar(raw_label),
                corners=np.asarray(corners, dtype=np.float64),
                score=score,
                raw_label=raw_label,
                raw_corners=np.asarray(raw_corners),
                raw_score=raw_score,
            )
        )
    return ScenePredictions(tuple(detections))


def _parse_scalar_text(value: np.ndarray, name: str) -> str:
    array = np.asarray(value)
    if array.dtype.hasobject or array.ndim != 0:
        raise ValueError(f"{name} must be a non-object scalar string")
    scalar = _python_scalar(array.item())
    if isinstance(scalar, bytes):
        scalar = scalar.decode("utf-8")
    if not isinstance(scalar, str):
        raise TypeError(f"{name} must be a string")
    return scalar


def load_diagnostics(
    path: str | os.PathLike[str],
    *,
    expected_scene_id: str,
) -> SceneDiagnostics:
    """Load final-application flags and cumulative runtime counters."""

    diagnostic_path = Path(path)
    if not diagnostic_path.is_file():
        raise FileNotFoundError(diagnostic_path)
    with np.load(diagnostic_path, allow_pickle=False) as archive:
        required = {"scene_id", "result_indices", "refit_applied", "summary_json"}
        missing = required - set(archive.files)
        if missing:
            raise ValueError(
                f"{diagnostic_path} is missing fields {sorted(missing)}"
            )
        scene_id = _parse_scalar_text(archive["scene_id"], "scene_id")
        result_indices = np.asarray(archive["result_indices"])
        refit_applied = np.asarray(archive["refit_applied"])
        summary_text = _parse_scalar_text(archive["summary_json"], "summary_json")

    if scene_id != expected_scene_id:
        raise ValueError(
            f"diagnostic scene {scene_id!r} does not match "
            f"{expected_scene_id!r}"
        )
    if result_indices.ndim != 1 or not np.issubdtype(
        result_indices.dtype, np.integer
    ):
        raise TypeError("result_indices must be a one-dimensional integer array")
    result_indices = np.asarray(result_indices, dtype=np.int64)
    if (result_indices < 0).any() or len(np.unique(result_indices)) != len(
        result_indices
    ):
        raise ValueError("result_indices must be unique and non-negative")
    if refit_applied.shape != result_indices.shape or refit_applied.dtype != np.bool_:
        raise TypeError("refit_applied must be Boolean and match result_indices")
    try:
        summary = json.loads(summary_text)
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid summary_json in {diagnostic_path}") from error
    if not isinstance(summary, Mapping):
        raise TypeError("summary_json must decode to an object")
    return SceneDiagnostics(
        result_indices=result_indices,
        refit_applied=np.asarray(refit_applied, dtype=np.bool_),
        summary=dict(summary),
    )


def load_axis_alignment(
    scan_root: str | os.PathLike[str], scene_id: str
) -> np.ndarray:
    """Load a rigid ScanNet ``axisAlignment`` transform."""

    metadata = Path(scan_root) / scene_id / f"{scene_id}.txt"
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
    if not np.allclose(
        transform[3], np.asarray([0.0, 0.0, 0.0, 1.0]), atol=1e-6
    ):
        raise ValueError(f"axisAlignment is not homogeneous in {metadata}")
    rotation = transform[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=2e-3):
        raise ValueError(f"axisAlignment is not rigid in {metadata}")
    if not np.isclose(abs(np.linalg.det(rotation)), 1.0, atol=2e-3):
        raise ValueError(f"axisAlignment rotation is singular in {metadata}")
    return transform


def load_gt_boxes(
    gt_root: str | os.PathLike[str], scene_id: str
) -> np.ndarray:
    """Load aligned ScanNet GT boxes as ``[cx,cy,cz,dx,dy,dz]``."""

    path = Path(gt_root) / f"{scene_id}_bbox.npy"
    if not path.is_file():
        raise FileNotFoundError(path)
    boxes = np.load(path, allow_pickle=False)
    if boxes.ndim != 2 or boxes.shape[1] < 6:
        raise ValueError(f"GT boxes in {path} must have shape [N, >=6]")
    boxes = np.asarray(boxes[:, :6], dtype=np.float64)
    if (
        not np.isfinite(boxes).all()
        or (len(boxes) and (boxes[:, 3:6] <= 0.0).any())
    ):
        raise ValueError(f"GT boxes in {path} are invalid")
    return boxes


def aligned_aabb(corners: np.ndarray, transform: np.ndarray) -> np.ndarray:
    """Transform corners to evaluator coordinates and return centre/size."""

    value = np.asarray(corners, dtype=np.float64)
    if value.shape != (8, 3) or not np.isfinite(value).all():
        raise ValueError("corners must have finite shape [8, 3]")
    homogeneous = np.concatenate((value, np.ones((8, 1))), axis=1)
    aligned = homogeneous @ np.asarray(transform, dtype=np.float64).T
    aligned = aligned[:, :3]
    minimum = aligned.min(axis=0)
    maximum = aligned.max(axis=0)
    dimensions = maximum - minimum
    if (dimensions <= 0.0).any():
        raise ValueError("corners define a degenerate aligned box")
    return np.concatenate(((minimum + maximum) * 0.5, dimensions))


def aabb_iou(box: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """Compute IoU between one centre/size box and ``N`` targets."""

    value = np.asarray(box, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)
    if value.shape != (6,) or targets.ndim != 2 or targets.shape[1] != 6:
        raise ValueError("IoU expects one [6] box and targets [N,6]")
    if not len(targets):
        return np.empty(0, dtype=np.float64)
    value_half = value[3:6] * 0.5
    target_half = targets[:, 3:6] * 0.5
    value_min, value_max = value[:3] - value_half, value[:3] + value_half
    target_min, target_max = (
        targets[:, :3] - target_half,
        targets[:, :3] + target_half,
    )
    intersection_size = np.maximum(
        np.minimum(value_max, target_max) - np.maximum(value_min, target_min),
        0.0,
    )
    intersection = np.prod(intersection_size, axis=1)
    value_volume = float(np.prod(value[3:6]))
    target_volume = np.prod(targets[:, 3:6], axis=1)
    union = value_volume + target_volume - intersection
    return np.divide(
        intersection,
        union,
        out=np.zeros_like(intersection),
        where=union > 0.0,
    )


def oriented_basis(corners: np.ndarray) -> np.ndarray:
    """Recover BoxFusion's ordered, right-handed local basis."""

    value = np.asarray(corners, dtype=np.float64)
    if value.shape != (8, 3) or not np.isfinite(value).all():
        raise ValueError("oriented corners must have finite shape [8, 3]")
    edges = np.stack(
        (
            value[1] - value[0],
            value[3] - value[0],
            value[4] - value[0],
        ),
        axis=1,
    )
    dimensions = np.linalg.norm(edges, axis=0)
    if (dimensions <= 1e-8).any():
        raise ValueError("oriented corners contain a degenerate edge")
    basis = edges / dimensions[None, :]
    if not np.allclose(basis.T @ basis, np.eye(3), atol=2e-3):
        raise ValueError("oriented box edges are not orthogonal")
    if np.linalg.det(basis) <= 0.0:
        raise ValueError("oriented box basis is not right handed")
    return basis


def _numeric_counter(summary: Mapping[str, Any], name: str) -> int:
    value = summary.get(name, 0)
    if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
        raise TypeError(f"summary counter {name!r} must be numeric")
    numeric = float(value)
    if not math.isfinite(numeric) or numeric < 0.0 or not numeric.is_integer():
        raise ValueError(f"summary counter {name!r} must be non-negative integer")
    return int(numeric)


def _quantiles(values: Iterable[float]) -> dict[str, float | None]:
    array = np.asarray(list(values), dtype=np.float64)
    if not len(array):
        return {
            "minimum": None,
            "q10": None,
            "q25": None,
            "q50": None,
            "q75": None,
            "q90": None,
            "maximum": None,
        }
    quantiles = np.quantile(array, (0.0, 0.10, 0.25, 0.50, 0.75, 0.90, 1.0))
    names = ("minimum", "q10", "q25", "q50", "q75", "q90", "maximum")
    return {name: float(value) for name, value in zip(names, quantiles)}


def _stable_score_order(detections: Sequence[Detection]) -> np.ndarray:
    scores = np.asarray([item.score for item in detections], dtype=np.float64)
    return np.argsort(-scores, kind="stable")


def _voc_ap(recall: np.ndarray, precision: np.ndarray) -> float:
    """Match the continuous VOC AP used by ``evaluation/utils/eval_det.py``."""

    recall = np.asarray(recall, dtype=np.float64)
    precision = np.asarray(precision, dtype=np.float64)
    extended_recall = np.concatenate(([0.0], recall, [1.0]))
    extended_precision = np.concatenate(([0.0], precision, [0.0]))
    for index in range(extended_precision.size - 1, 0, -1):
        extended_precision[index - 1] = max(
            extended_precision[index - 1],
            extended_precision[index],
        )
    changes = np.where(
        extended_recall[1:] != extended_recall[:-1]
    )[0]
    return float(
        np.sum(
            (extended_recall[changes + 1] - extended_recall[changes])
            * extended_precision[changes + 1]
        )
    )


def class_agnostic_ap(
    predictions: Mapping[str, Sequence[tuple[np.ndarray, float]]],
    ground_truth: Mapping[str, np.ndarray],
    threshold: float,
) -> dict[str, float | int]:
    """Compute the repository's class-agnostic ScanNet AP on aligned AABBs."""

    image_ids: list[str] = []
    boxes: list[np.ndarray] = []
    scores: list[float] = []
    for scene_id in predictions:
        for box, score in predictions[scene_id]:
            image_ids.append(scene_id)
            boxes.append(np.asarray(box, dtype=np.float64))
            scores.append(float(score))
    order = np.argsort(-np.asarray(scores, dtype=np.float64))
    image_ids = [image_ids[int(index)] for index in order]
    boxes = [boxes[int(index)] for index in order]
    matched = {
        scene_id: np.zeros(len(scene_boxes), dtype=np.bool_)
        for scene_id, scene_boxes in ground_truth.items()
    }
    true_positive = np.zeros(len(boxes), dtype=np.float64)
    false_positive = np.zeros(len(boxes), dtype=np.float64)
    for detection_index, (scene_id, box) in enumerate(zip(image_ids, boxes)):
        targets = np.asarray(
            ground_truth.get(scene_id, np.empty((0, 6))),
            dtype=np.float64,
        )
        overlaps = aabb_iou(box, targets)
        if len(overlaps):
            target_index = int(np.argmax(overlaps))
            # The upstream evaluator uses strict greater-than.
            if (
                float(overlaps[target_index]) > float(threshold)
                and not matched[scene_id][target_index]
            ):
                true_positive[detection_index] = 1.0
                matched[scene_id][target_index] = True
                continue
        false_positive[detection_index] = 1.0
    cumulative_tp = np.cumsum(true_positive)
    cumulative_fp = np.cumsum(false_positive)
    positive_count = int(sum(len(value) for value in ground_truth.values()))
    recall = cumulative_tp / float(positive_count + 1e-6)
    precision = cumulative_tp / np.maximum(
        cumulative_tp + cumulative_fp,
        np.finfo(np.float64).eps,
    )
    return {
        "ap": _voc_ap(recall, precision),
        "precision": float(precision[-1]) if len(precision) else 0.0,
        "recall": float(recall[-1]) if len(recall) else 0.0,
        "true_positives": int(cumulative_tp[-1]) if len(cumulative_tp) else 0,
        "false_positives": (
            int(cumulative_fp[-1]) if len(cumulative_fp) else 0
        ),
        "ground_truth_count": positive_count,
    }


def _all_threshold_metrics(
    predictions: Mapping[str, Sequence[tuple[np.ndarray, float]]],
    ground_truth: Mapping[str, np.ndarray],
    thresholds: Sequence[float],
) -> dict[str, dict[str, float | int]]:
    return {
        f"{threshold:.2f}": class_agnostic_ap(
            predictions, ground_truth, threshold
        )
        for threshold in thresholds
    }


def build_paired_report(
    *,
    identity_pred_root: str | os.PathLike[str],
    candidate_pred_root: str | os.PathLike[str],
    candidate_diagnostics_root: str | os.PathLike[str],
    scene_list: str | os.PathLike[str],
    scan_root: str | os.PathLike[str],
    gt_root: str | os.PathLike[str],
    thresholds: Sequence[float] = DEFAULT_THRESHOLDS,
    score_atol: float = 1e-7,
    iou_epsilon: float = 1e-4,
    yaw_cosine_threshold: float = 0.999,
) -> dict[str, Any]:
    """Build a JSON-safe paired B5 diagnostic report."""

    thresholds = tuple(float(item) for item in thresholds)
    if (
        not thresholds
        or any(not math.isfinite(item) or not 0.0 < item < 1.0 for item in thresholds)
        or len(set(thresholds)) != len(thresholds)
    ):
        raise ValueError("thresholds must be unique finite values in (0,1)")
    if not math.isfinite(score_atol) or score_atol < 0.0:
        raise ValueError("score_atol must be non-negative and finite")
    if not math.isfinite(iou_epsilon) or iou_epsilon < 0.0:
        raise ValueError("iou_epsilon must be non-negative and finite")
    if not math.isfinite(yaw_cosine_threshold) or not 0.0 <= yaw_cosine_threshold <= 1.0:
        raise ValueError("yaw_cosine_threshold must lie in [0,1]")

    scenes = read_scene_ids(scene_list)
    identity_root = Path(identity_pred_root)
    candidate_root = Path(candidate_pred_root)
    diagnostics_root = Path(candidate_diagnostics_root)
    for name, root in (
        ("identity prediction", identity_root),
        ("candidate prediction", candidate_root),
        ("candidate diagnostics", diagnostics_root),
        ("ScanNet scan", Path(scan_root)),
        ("ScanNet GT", Path(gt_root)),
    ):
        if not root.is_dir():
            raise FileNotFoundError(f"{name} root does not exist: {root}")

    identity_total = 0
    candidate_total = 0
    paired_total = 0
    count_mismatch_scenes: list[str] = []
    label_mismatch_scenes: list[str] = []
    score_order_mismatch_scenes: list[str] = []
    score_deltas: list[float] = []
    geometry_deltas: list[float] = []
    final_applied_total = 0
    final_applied_scenes: set[str] = set()
    final_applied_invalid_index = 0
    cumulative = {
        "attempted": 0,
        "accepted": 0,
        "quality_rejected": 0,
        "gate_rejected": 0,
        "invalid_orientation": 0,
    }
    accepted_deltas: list[float] = []
    accepted_original_ious: list[float] = []
    accepted_candidate_ious: list[float] = []
    accepted_improved = 0
    accepted_worsened = 0
    accepted_equal = 0
    accepted_unmatched_gt = 0
    accepted_yaw_signed: list[float] = []
    accepted_yaw_absolute: list[float] = []
    accepted_invalid_yaw = 0
    accepted_yaw_changed = 0
    crossing: dict[float, dict[str, Any]] = {
        threshold: {
            "up": 0,
            "down": 0,
            "up_scenes": set(),
            "down_scenes": set(),
        }
        for threshold in thresholds
    }
    identity_metric_predictions: dict[
        str, list[tuple[np.ndarray, float]]
    ] = {}
    candidate_metric_predictions: dict[
        str, list[tuple[np.ndarray, float]]
    ] = {}
    locked_metric_predictions: dict[
        str, list[tuple[np.ndarray, float]]
    ] = {}
    metric_ground_truth: dict[str, np.ndarray] = {}

    for scene_id in scenes:
        identity = load_predictions(
            identity_root / f"{scene_id}{PREDICTION_SUFFIX}"
        )
        candidate = load_predictions(
            candidate_root / f"{scene_id}{PREDICTION_SUFFIX}"
        )
        diagnostics = load_diagnostics(
            diagnostics_root / f"{scene_id}{DIAGNOSTIC_SUFFIX}",
            expected_scene_id=scene_id,
        )
        transform = load_axis_alignment(scan_root, scene_id)
        gt_boxes = load_gt_boxes(gt_root, scene_id)
        metric_ground_truth[scene_id] = gt_boxes
        identity_metric_predictions[scene_id] = [
            (aligned_aabb(item.corners, transform), item.score)
            for item in identity.detections
        ]
        candidate_metric_predictions[scene_id] = [
            (aligned_aabb(item.corners, transform), item.score)
            for item in candidate.detections
        ]

        identity_count = len(identity.detections)
        candidate_count = len(candidate.detections)
        identity_total += identity_count
        candidate_total += candidate_count
        if identity_count != candidate_count:
            count_mismatch_scenes.append(scene_id)
        pair_count = min(identity_count, candidate_count)
        paired_total += pair_count

        identity_labels = [
            item.label for item in identity.detections[:pair_count]
        ]
        candidate_labels = [
            item.label for item in candidate.detections[:pair_count]
        ]
        if identity_count != candidate_count or identity_labels != candidate_labels:
            label_mismatch_scenes.append(scene_id)
        if (
            identity_count != candidate_count
            or not np.array_equal(
                _stable_score_order(identity.detections),
                _stable_score_order(candidate.detections),
            )
        ):
            score_order_mismatch_scenes.append(scene_id)
        if (
            identity_count == candidate_count
            and identity_labels == candidate_labels
        ):
            locked_metric_predictions[scene_id] = [
                (
                    aligned_aabb(candidate_item.corners, transform),
                    identity_item.score,
                )
                for identity_item, candidate_item in zip(
                    identity.detections, candidate.detections
                )
            ]

        for identity_item, candidate_item in zip(
            identity.detections[:pair_count],
            candidate.detections[:pair_count],
        ):
            score_deltas.append(abs(identity_item.score - candidate_item.score))
            geometry_deltas.append(
                float(
                    np.max(
                        np.abs(
                            identity_item.corners - candidate_item.corners
                        )
                    )
                )
            )

        cumulative["attempted"] += _numeric_counter(
            diagnostics.summary, "neural_refits_attempted"
        )
        cumulative["accepted"] += _numeric_counter(
            diagnostics.summary, "neural_refits_accepted"
        )
        cumulative["quality_rejected"] += _numeric_counter(
            diagnostics.summary, "neural_refits_quality_rejected"
        )
        cumulative["gate_rejected"] += _numeric_counter(
            diagnostics.summary, "neural_refits_gate_rejected"
        )
        cumulative["invalid_orientation"] += _numeric_counter(
            diagnostics.summary, "neural_refits_invalid_orientation"
        )

        applied_indices = diagnostics.result_indices[
            diagnostics.refit_applied
        ]
        final_applied_total += int(len(applied_indices))
        if len(applied_indices):
            final_applied_scenes.add(scene_id)
        for result_index in applied_indices:
            index = int(result_index)
            if index >= identity_count or index >= candidate_count:
                final_applied_invalid_index += 1
                continue
            identity_item = identity.detections[index]
            candidate_item = candidate.detections[index]

            try:
                identity_basis = oriented_basis(identity_item.corners)
                candidate_basis = oriented_basis(candidate_item.corners)
                axis_cosines = np.diag(identity_basis.T @ candidate_basis)
                accepted_yaw_signed.extend(axis_cosines.tolist())
                accepted_yaw_absolute.extend(np.abs(axis_cosines).tolist())
                if float(np.min(np.abs(axis_cosines))) < yaw_cosine_threshold:
                    accepted_yaw_changed += 1
            except ValueError:
                accepted_invalid_yaw += 1

            if not len(gt_boxes):
                accepted_unmatched_gt += 1
                continue
            identity_box = aligned_aabb(identity_item.corners, transform)
            candidate_box = aligned_aabb(candidate_item.corners, transform)
            identity_ious = aabb_iou(identity_box, gt_boxes)
            gt_index = int(np.argmax(identity_ious))
            original_iou = float(identity_ious[gt_index])
            candidate_iou = float(
                aabb_iou(candidate_box, gt_boxes[gt_index : gt_index + 1])[0]
            )
            delta = candidate_iou - original_iou
            accepted_original_ious.append(original_iou)
            accepted_candidate_ious.append(candidate_iou)
            accepted_deltas.append(delta)
            if delta > iou_epsilon:
                accepted_improved += 1
            elif delta < -iou_epsilon:
                accepted_worsened += 1
            else:
                accepted_equal += 1
            for threshold in thresholds:
                if original_iou < threshold <= candidate_iou:
                    crossing[threshold]["up"] += 1
                    crossing[threshold]["up_scenes"].add(scene_id)
                elif candidate_iou < threshold <= original_iou:
                    crossing[threshold]["down"] += 1
                    crossing[threshold]["down_scenes"].add(scene_id)

    score_array = np.asarray(score_deltas, dtype=np.float64)
    geometry_array = np.asarray(geometry_deltas, dtype=np.float64)
    crossing_report: dict[str, Any] = {}
    for threshold in thresholds:
        values = crossing[threshold]
        up = int(values["up"])
        down = int(values["down"])
        crossing_report[f"{threshold:.2f}"] = {
            "up": up,
            "down": down,
            "net": up - down,
            "up_scene_count": len(values["up_scenes"]),
            "down_scene_count": len(values["down_scenes"]),
            "up_scenes": sorted(values["up_scenes"]),
            "down_scenes": sorted(values["down_scenes"]),
        }

    count_equal = not count_mismatch_scenes
    label_order_equal = not label_mismatch_scenes
    score_rank_order_equal = not score_order_mismatch_scenes
    identity_metrics = _all_threshold_metrics(
        identity_metric_predictions, metric_ground_truth, thresholds
    )
    candidate_metrics = _all_threshold_metrics(
        candidate_metric_predictions, metric_ground_truth, thresholds
    )
    locked_metrics = (
        _all_threshold_metrics(
            locked_metric_predictions, metric_ground_truth, thresholds
        )
        if count_equal and label_order_equal
        else None
    )
    locked_delta = (
        {
            key: {
                "ap": float(locked_metrics[key]["ap"])
                - float(identity_metrics[key]["ap"]),
                "precision": float(locked_metrics[key]["precision"])
                - float(identity_metrics[key]["precision"]),
                "recall": float(locked_metrics[key]["recall"])
                - float(identity_metrics[key]["recall"]),
            }
            for key in identity_metrics
        }
        if locked_metrics is not None
        else None
    )
    report: dict[str, Any] = {
        "schema": REPORT_SCHEMA,
        "format_version": REPORT_FORMAT_VERSION,
        "scene_count": len(scenes),
        "scenes": scenes,
        "inputs": {
            "identity_pred_root": str(identity_root.resolve()),
            "candidate_pred_root": str(candidate_root.resolve()),
            "candidate_diagnostics_root": str(diagnostics_root.resolve()),
            "scene_list": str(Path(scene_list).resolve()),
            "scan_root": str(Path(scan_root).resolve()),
            "gt_root": str(Path(gt_root).resolve()),
        },
        "protocol": {
            "thresholds": list(thresholds),
            "score_atol": float(score_atol),
            "iou_epsilon": float(iou_epsilon),
            "yaw_cosine_threshold": float(yaw_cosine_threshold),
            "gt_matching": "identity_best_iou_then_same_gt_for_candidate",
            "pairing": "scene_and_exported_list_position",
        },
        "pairing": {
            "identity_detection_count": identity_total,
            "candidate_detection_count": candidate_total,
            "paired_detection_count": paired_total,
            "count_equal": count_equal,
            "count_mismatch_scenes": count_mismatch_scenes,
            "label_order_equal": label_order_equal,
            "label_mismatch_scenes": label_mismatch_scenes,
            "score_rank_order_equal": score_rank_order_equal,
            "score_rank_order_mismatch_scenes": score_order_mismatch_scenes,
            "position_pairing_contract_valid": bool(
                count_equal and label_order_equal and score_rank_order_equal
            ),
        },
        "scores": {
            "exactly_equal": bool(
                len(score_array) == paired_total
                and np.count_nonzero(score_array) == 0
            ),
            "exact_changed_count": int(np.count_nonzero(score_array)),
            "above_atol_count": int(np.count_nonzero(score_array > score_atol)),
            "maximum_absolute_delta": (
                float(score_array.max()) if len(score_array) else 0.0
            ),
            "mean_absolute_delta": (
                float(score_array.mean()) if len(score_array) else 0.0
            ),
        },
        "metrics": {
            "note": (
                "CPU class-agnostic proxy matching the repository's "
                "axis-aligned ScanNet VOC-AP protocol; canonical headline "
                "remains evaluation/eval_scannet.py"
            ),
            "identity": identity_metrics,
            "candidate_native_scores": candidate_metrics,
            "candidate_identity_scores": locked_metrics,
            "candidate_identity_scores_delta": locked_delta,
        },
        "geometry": {
            "changed_count_above_1e-4": int(
                np.count_nonzero(geometry_array > 1e-4)
            ),
            "maximum_corner_absolute_delta": (
                float(geometry_array.max()) if len(geometry_array) else 0.0
            ),
        },
        "runtime": {
            "cumulative": cumulative,
            "final_applied": final_applied_total,
            "final_applied_scene_count": len(final_applied_scenes),
            "final_applied_scenes": sorted(final_applied_scenes),
            "final_applied_invalid_result_index": final_applied_invalid_index,
        },
        "accepted_iou": {
            "evaluated_count": len(accepted_deltas),
            "unmatched_gt_count": accepted_unmatched_gt,
            "improved": accepted_improved,
            "worsened": accepted_worsened,
            "equal": accepted_equal,
            "improvement_fraction_excluding_equal": (
                float(accepted_improved / (accepted_improved + accepted_worsened))
                if accepted_improved + accepted_worsened
                else None
            ),
            "original_iou_quantiles": _quantiles(accepted_original_ious),
            "candidate_iou_quantiles": _quantiles(accepted_candidate_ious),
            "delta_iou_quantiles": _quantiles(accepted_deltas),
        },
        "crossings": crossing_report,
        "yaw": {
            "evaluated_box_count": (
                final_applied_total
                - final_applied_invalid_index
                - accepted_invalid_yaw
            ),
            "invalid_box_count": accepted_invalid_yaw,
            "changed_box_count": accepted_yaw_changed,
            "signed_axis_cosine_quantiles": _quantiles(accepted_yaw_signed),
            "absolute_axis_cosine_quantiles": _quantiles(
                accepted_yaw_absolute
            ),
            "minimum_absolute_axis_cosine": (
                float(min(accepted_yaw_absolute))
                if accepted_yaw_absolute
                else None
            ),
        },
        "score_locked_pred_root": None,
    }
    return report


def _path_is_within(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def write_score_locked_predictions(
    *,
    identity_pred_root: str | os.PathLike[str],
    candidate_pred_root: str | os.PathLike[str],
    scene_list: str | os.PathLike[str],
    output_root: str | os.PathLike[str] | None,
) -> Path:
    """Write candidate geometry with identity scores to a new directory.

    ``output_root=None`` creates an explicitly named temporary directory.
    A specified path must not already exist and must not lie inside either
    input root.  Input files are never modified.
    """

    scenes = read_scene_ids(scene_list)
    identity_root = Path(identity_pred_root).resolve()
    candidate_root = Path(candidate_pred_root).resolve()
    for root in (identity_root, candidate_root):
        if not root.is_dir():
            raise FileNotFoundError(root)

    loaded: dict[str, tuple[ScenePredictions, ScenePredictions]] = {}
    for scene_id in scenes:
        identity = load_predictions(
            identity_root / f"{scene_id}{PREDICTION_SUFFIX}"
        )
        candidate = load_predictions(
            candidate_root / f"{scene_id}{PREDICTION_SUFFIX}"
        )
        if len(identity.detections) != len(candidate.detections):
            raise ValueError(
                f"cannot lock scores: detection count differs for {scene_id}"
            )
        if [item.label for item in identity.detections] != [
            item.label for item in candidate.detections
        ]:
            raise ValueError(
                f"cannot lock scores: label/order differs for {scene_id}"
            )
        loaded[scene_id] = (identity, candidate)

    temporary_output = output_root is None
    if temporary_output:
        destination = Path(
            tempfile.mkdtemp(prefix="boxfusion_b5_score_locked_")
        ).resolve()
    else:
        destination = Path(output_root).resolve()
        for input_root in (identity_root, candidate_root):
            if destination == input_root or _path_is_within(
                destination, input_root
            ):
                raise ValueError(
                    "score-locked output must be outside both input roots"
                )
        if destination.exists():
            raise FileExistsError(
                f"score-locked output already exists: {destination}"
            )
        destination.mkdir(parents=True, exist_ok=False)

    for scene_id, (identity, candidate) in loaded.items():
        locked = []
        for identity_item, candidate_item in zip(
            identity.detections, candidate.detections
        ):
            locked.append(
                (
                    candidate_item.raw_label,
                    np.asarray(candidate_item.raw_corners).copy(),
                    identity_item.raw_score,
                )
            )
        output = destination / f"{scene_id}{PREDICTION_SUFFIX}"
        temporary = output.with_name(output.name + ".tmp")
        with temporary.open("wb") as handle:
            pickle.dump([locked], handle, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(temporary, output)
    return destination


def parse_thresholds(value: str) -> tuple[float, ...]:
    try:
        thresholds = tuple(
            float(item.strip()) for item in value.split(",") if item.strip()
        )
    except ValueError as error:
        raise argparse.ArgumentTypeError("thresholds must be numeric") from error
    if (
        not thresholds
        or any(not math.isfinite(item) or not 0.0 < item < 1.0 for item in thresholds)
        or len(set(thresholds)) != len(thresholds)
    ):
        raise argparse.ArgumentTypeError(
            "thresholds must be unique values in (0,1)"
        )
    return thresholds


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--identity-pred-root", type=Path, required=True)
    parser.add_argument("--candidate-pred-root", type=Path, required=True)
    parser.add_argument("--candidate-diagnostics-root", type=Path, required=True)
    parser.add_argument("--scene-list", type=Path, required=True)
    parser.add_argument("--scan-root", type=Path, required=True)
    parser.add_argument("--gt-root", type=Path, required=True)
    parser.add_argument(
        "--thresholds",
        type=parse_thresholds,
        default=DEFAULT_THRESHOLDS,
        help="comma-separated IoU crossings (default: 0.15,0.25,0.5)",
    )
    parser.add_argument("--score-atol", type=float, default=1e-7)
    parser.add_argument("--iou-epsilon", type=float, default=1e-4)
    parser.add_argument("--yaw-cosine-threshold", type=float, default=0.999)
    parser.add_argument(
        "--lock-identity-scores",
        nargs="?",
        const="AUTO",
        metavar="OUTPUT_ROOT",
        help=(
            "write candidate geometry with identity scores; omit OUTPUT_ROOT "
            "to create a temporary directory"
        ),
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        help="optionally write the same JSON report atomically",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_argument_parser()
    arguments = parser.parse_args(argv)
    try:
        report = build_paired_report(
            identity_pred_root=arguments.identity_pred_root,
            candidate_pred_root=arguments.candidate_pred_root,
            candidate_diagnostics_root=arguments.candidate_diagnostics_root,
            scene_list=arguments.scene_list,
            scan_root=arguments.scan_root,
            gt_root=arguments.gt_root,
            thresholds=arguments.thresholds,
            score_atol=arguments.score_atol,
            iou_epsilon=arguments.iou_epsilon,
            yaw_cosine_threshold=arguments.yaw_cosine_threshold,
        )
        if arguments.lock_identity_scores is not None:
            requested = arguments.lock_identity_scores
            locked_root = write_score_locked_predictions(
                identity_pred_root=arguments.identity_pred_root,
                candidate_pred_root=arguments.candidate_pred_root,
                scene_list=arguments.scene_list,
                output_root=None if requested == "AUTO" else requested,
            )
            report["score_locked_pred_root"] = str(locked_root)
        text = json.dumps(report, indent=2, sort_keys=True)
        if arguments.output_json is not None:
            output = arguments.output_json
            output.parent.mkdir(parents=True, exist_ok=True)
            temporary = output.with_name(output.name + ".tmp")
            temporary.write_text(text + "\n", encoding="utf-8")
            os.replace(temporary, output)
    except (FileExistsError, FileNotFoundError, TypeError, ValueError) as error:
        parser.error(str(error))
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
