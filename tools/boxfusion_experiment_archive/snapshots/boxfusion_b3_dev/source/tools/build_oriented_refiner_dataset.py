#!/usr/bin/env python3
"""Build leakage-safe supervision for the orientation-aware B5-v2 refiner.

This is an offline-only tool.  Runtime diagnostics provide Top-K RGB-D points
and quality features, while the matching ``*_boxes.pkl`` file provides the
actual oriented BoxFusion corners.  ScanNet ground truth is used only here.

For every prediction the tool:

1. recovers the original OBB's orthonormal local frame from its corners;
2. transforms the observed points into that frame;
3. matches the evaluator's axis-aligned enclosure to the closest GT AABB;
4. projects the aligned GT centre exactly into the OBB local frame;
5. solves ``abs(aligned_basis) @ local_dims ~= gt_dims`` with a small exact
   non-negative least-squares solver;
6. clips that target to the neural refiner's reachable residual range; and
7. enables geometry supervision only when the reachable target improves
   aligned-AABB IoU.  All other samples remain quality-head negatives.

The output NPZ never contains Python objects and always carries ``scene_ids``.
The companion trainer consequently performs a scene-level split with no
train/validation leakage.
"""

from __future__ import annotations

import argparse
import pickle
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

from boxfusion.quality_score import QUALITY_FEATURE_NAMES


DATASET_SCHEMA = "boxfusion.oriented_box_refiner_dataset"
DATASET_FORMAT_VERSION = 1
QUALITY_FEATURE_DIM = len(QUALITY_FEATURE_NAMES)
SCENE_PATTERN = re.compile(r"scene\d{4}_\d{2}")
_REQUIRED_DIAGNOSTIC_FIELDS = frozenset(
    {
        "scene_id",
        "quality_features",
        "result_indices",
    }
)


@dataclass(frozen=True)
class BuildConfig:
    diagnostics_root: Path
    prediction_root: Path
    scan_root: Path
    gt_root: Path
    scene_list: Path
    output: Path
    min_match_iou: float = 0.15
    improvement_epsilon: float = 1e-4
    max_center_fraction: float = 0.15
    max_log_dimension_residual: float = float(np.log(1.25))

    def validated(self) -> "BuildConfig":
        for name in (
            "diagnostics_root",
            "prediction_root",
            "scan_root",
            "gt_root",
        ):
            path = Path(getattr(self, name))
            if not path.is_dir():
                raise FileNotFoundError(f"{name} is not a directory: {path}")
        if not Path(self.scene_list).is_file():
            raise FileNotFoundError(self.scene_list)
        for name in ("min_match_iou", "improvement_epsilon"):
            value = getattr(self, name)
            if not np.isscalar(value) or not np.isfinite(value):
                raise ValueError(f"{name} must be a finite scalar")
        if not 0.0 <= float(self.min_match_iou) <= 1.0:
            raise ValueError("min_match_iou must lie in [0, 1]")
        if float(self.improvement_epsilon) < 0.0:
            raise ValueError("improvement_epsilon must be non-negative")
        for name in (
            "max_center_fraction",
            "max_log_dimension_residual",
        ):
            value = getattr(self, name)
            if (
                not np.isscalar(value)
                or not np.isfinite(value)
                or float(value) <= 0.0
            ):
                raise ValueError(f"{name} must be positive and finite")
        return self


@dataclass(frozen=True)
class SceneDiagnostics:
    scene_id: str
    quality_features: np.ndarray
    points: np.ndarray
    point_mask: np.ndarray
    result_indices: np.ndarray
    track_ids: np.ndarray


@dataclass(frozen=True)
class BuildSummary:
    scenes: int
    samples: int
    geometry_positives: int
    quality_negatives: int
    invalid_oriented_boxes: int
    output: Path


def read_scene_ids(path: Path) -> list[str]:
    """Read a non-empty, duplicate-free ScanNet scene list."""

    scenes = [
        line.strip() for line in Path(path).read_text().splitlines()
        if line.strip()
    ]
    if not scenes:
        raise ValueError(f"No scenes found in {path}")
    invalid = [scene for scene in scenes if SCENE_PATTERN.fullmatch(scene) is None]
    if invalid:
        raise ValueError(f"Invalid ScanNet scene id: {invalid[0]!r}")
    if len(set(scenes)) != len(scenes):
        raise ValueError(f"Duplicate scene ids in {path}")
    return scenes


def resolve_diagnostic_path(root: Path, scene_id: str) -> Path:
    """Resolve exactly one diagnostics archive for ``scene_id``."""

    matches = sorted(Path(root).glob(f"{scene_id}*.npz"))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected exactly one diagnostic NPZ for {scene_id} in {root}; "
            f"found {len(matches)}"
        )
    return matches[0]


def _parse_scalar_string(value: np.ndarray, name: str) -> str:
    array = np.asarray(value)
    if array.dtype.hasobject:
        raise ValueError(f"{name} must not use object dtype")
    if array.ndim == 0:
        scalar = array.item()
    elif array.ndim == 1 and array.size == 1:
        scalar = array[0]
    else:
        raise ValueError(f"{name} must be a scalar string")
    if isinstance(scalar, bytes):
        scalar = scalar.decode("utf-8")
    if not isinstance(scalar, str):
        raise TypeError(f"{name} must be a string")
    return scalar


def load_scene_diagnostics(
    path: Path, expected_scene_id: str | None = None
) -> SceneDiagnostics:
    """Load and strictly validate one pickle-free runtime diagnostic."""

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    try:
        with np.load(path, allow_pickle=False) as payload:
            missing = _REQUIRED_DIAGNOSTIC_FIELDS - set(payload.files)
            if missing:
                raise ValueError(f"{path} is missing fields: {sorted(missing)}")
            scene_id = _parse_scalar_string(payload["scene_id"], "scene_id")
            quality = np.asarray(payload["quality_features"])
            if {
                "geometry_points",
                "geometry_point_mask",
            } <= set(payload.files):
                points = np.asarray(payload["geometry_points"])
                point_mask = np.asarray(payload["geometry_point_mask"])
            elif {"points", "point_mask"} <= set(payload.files):
                points = np.asarray(payload["points"])
                point_mask = np.asarray(payload["point_mask"])
            else:
                raise ValueError(
                    f"{path} requires geometry_points/geometry_point_mask "
                    "or points/point_mask"
                )
            result_indices = np.asarray(payload["result_indices"])
            track_ids = (
                np.asarray(payload["track_ids"])
                if "track_ids" in payload.files
                else np.arange(len(points), dtype=np.int64)
            )
            if "quality_feature_names" in payload.files:
                names = tuple(
                    str(item)
                    for item in np.asarray(
                        payload["quality_feature_names"]
                    ).tolist()
                )
                if names != QUALITY_FEATURE_NAMES:
                    raise ValueError(
                        "diagnostic quality feature schema/order mismatch"
                    )
    except ValueError as error:
        if "Object arrays cannot be loaded" in str(error):
            raise ValueError(f"{path} contains pickled/object arrays") from error
        raise

    if SCENE_PATTERN.fullmatch(scene_id) is None:
        raise ValueError(f"Invalid diagnostic scene_id: {scene_id!r}")
    if expected_scene_id is not None and scene_id != expected_scene_id:
        raise ValueError(
            f"Diagnostic scene_id {scene_id!r} does not match "
            f"requested scene {expected_scene_id!r}"
        )
    if points.ndim != 3 or points.shape[2] != 3 or points.shape[1] < 1:
        raise ValueError("points must have shape [N, P, 3] with P > 0")
    sample_count, point_count, _ = points.shape
    if not np.issubdtype(points.dtype, np.floating):
        raise TypeError("points must use floating-point dtype")
    points = np.asarray(points, dtype=np.float64)
    if not np.isfinite(points).all():
        raise ValueError("points must be finite")
    if point_mask.shape != (sample_count, point_count):
        raise ValueError("point_mask must have shape [N, P]")
    if point_mask.dtype != np.bool_:
        raise TypeError("point_mask must have Boolean dtype")
    if sample_count and not point_mask.any(axis=1).all():
        raise ValueError("every sample must contain at least one valid point")
    if quality.shape != (sample_count, QUALITY_FEATURE_DIM):
        raise ValueError(
            f"quality_features must have shape [N, {QUALITY_FEATURE_DIM}]"
        )
    if not np.issubdtype(quality.dtype, np.floating):
        raise TypeError("quality_features must use floating-point dtype")
    quality = np.asarray(quality, dtype=np.float64)
    if (
        not np.isfinite(quality).all()
        or (quality < 0.0).any()
        or (quality > 1.0).any()
    ):
        raise ValueError("quality_features must be finite and lie in [0, 1]")
    for name, value in (
        ("result_indices", result_indices),
        ("track_ids", track_ids),
    ):
        if value.shape != (sample_count,):
            raise ValueError(f"{name} must have shape [N]")
        if not np.issubdtype(value.dtype, np.integer):
            raise TypeError(f"{name} must use integer dtype")
    result_indices = np.asarray(result_indices, dtype=np.int64)
    track_ids = np.asarray(track_ids, dtype=np.int64)
    if (result_indices < 0).any():
        raise ValueError("result_indices must be non-negative")
    if len(np.unique(result_indices)) != sample_count:
        raise ValueError("result_indices must be unique within a scene")

    return SceneDiagnostics(
        scene_id=scene_id,
        quality_features=np.ascontiguousarray(quality),
        points=np.ascontiguousarray(points),
        point_mask=np.ascontiguousarray(point_mask),
        result_indices=np.ascontiguousarray(result_indices),
        track_ids=np.ascontiguousarray(track_ids),
    )


def load_prediction_corners(path: Path) -> np.ndarray:
    """Load trusted local BoxFusion pickle output and return ``[N,8,3]``.

    Pickle is intentionally confined to this offline data-preparation tool.
    Never use this loader on an untrusted prediction file.
    """

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    if not isinstance(payload, (list, tuple)):
        raise ValueError(f"Invalid BoxFusion payload in {path}")
    detections = payload[0] if payload else []
    if not isinstance(detections, (list, tuple)):
        raise ValueError(f"Invalid detection list in {path}")
    if not detections:
        return np.empty((0, 8, 3), dtype=np.float64)
    corners = []
    for index, item in enumerate(detections):
        if not isinstance(item, (list, tuple)) or len(item) < 3:
            raise ValueError(f"Invalid detection {index} in {path}")
        value = np.asarray(item[1])
        if value.shape != (8, 3) or not np.issubdtype(
            value.dtype, np.number
        ):
            raise ValueError(f"Detection {index} corners must have shape [8,3]")
        value = np.asarray(value, dtype=np.float64)
        if not np.isfinite(value).all():
            raise ValueError(f"Detection {index} corners must be finite")
        corners.append(value)
    return np.stack(corners, axis=0)


def load_axis_alignment(scan_root: Path, scene_id: str) -> np.ndarray:
    """Load a rigid ScanNet ``axisAlignment`` transform."""

    metadata = Path(scan_root) / scene_id / f"{scene_id}.txt"
    if not metadata.is_file():
        raise FileNotFoundError(metadata)
    values = None
    for line in metadata.read_text().splitlines():
        stripped = line.strip()
        if stripped.startswith("axisAlignment"):
            if "=" not in stripped:
                raise ValueError(f"Malformed axisAlignment in {metadata}")
            values = np.fromstring(stripped.split("=", 1)[1], sep=" ")
            break
    if values is None or values.size != 16 or not np.isfinite(values).all():
        raise ValueError(f"Invalid or missing axisAlignment in {metadata}")
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


def load_gt_boxes(gt_root: Path, scene_id: str) -> np.ndarray:
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
        or (boxes[:, 3:6] <= 0.0).any()
    ):
        raise ValueError(f"GT boxes in {path} are invalid")
    return boxes


def oriented_box_frame(
    corners: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Recover ``center, dimensions, basis`` from BoxFusion corner ordering.

    Basis vectors are columns and preserve the exact upstream yaw/sign
    convention.  The incident edges at corner zero are ``1-0``, ``3-0`` and
    ``4-0``.
    """

    value = np.asarray(corners, dtype=np.float64)
    if value.shape != (8, 3) or not np.isfinite(value).all():
        raise ValueError("corners must be a finite [8, 3] array")
    edges = np.stack(
        (value[1] - value[0], value[3] - value[0], value[4] - value[0]),
        axis=1,
    )
    dimensions = np.linalg.norm(edges, axis=0)
    if (dimensions <= 1e-6).any():
        raise ValueError("oriented box has a degenerate edge")
    basis = edges / dimensions[None, :]
    if not np.allclose(basis.T @ basis, np.eye(3), atol=2e-3):
        raise ValueError("oriented box edges are not orthogonal")
    if np.linalg.det(basis) <= 0.0:
        raise ValueError("oriented box basis must be right handed")
    center = value.mean(axis=0)
    local = (value - center) @ basis
    expected_half = dimensions * 0.5
    if not np.allclose(np.abs(local), expected_half[None, :], atol=2e-3):
        raise ValueError("corners do not form the recovered oriented box")
    signs = np.sign(local)
    if len({tuple(row.tolist()) for row in signs}) != 8:
        raise ValueError("oriented box corner signs are not unique")
    return center, dimensions, basis


def nonnegative_least_squares_3x3(
    matrix: np.ndarray, target: np.ndarray
) -> np.ndarray:
    """Solve a three-variable NNLS problem by enumerating active sets."""

    matrix = np.asarray(matrix, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if matrix.shape != (3, 3) or target.shape != (3,):
        raise ValueError("NNLS expects a [3,3] matrix and [3] target")
    if not np.isfinite(matrix).all() or not np.isfinite(target).all():
        raise ValueError("NNLS inputs must be finite")
    best = np.zeros(3, dtype=np.float64)
    best_error = float(np.dot(target, target))
    # The empty active set above is a valid non-negative candidate.
    for mask in range(1, 1 << 3):
        active = [index for index in range(3) if mask & (1 << index)]
        solution, _, _, _ = np.linalg.lstsq(
            matrix[:, active], target, rcond=None
        )
        if (solution < -1e-10).any():
            continue
        candidate = np.zeros(3, dtype=np.float64)
        candidate[active] = np.maximum(solution, 0.0)
        error = float(np.sum((matrix @ candidate - target) ** 2))
        if error < best_error - 1e-12:
            best = candidate
            best_error = error
    return best


def center_size_to_minmax(boxes: np.ndarray) -> np.ndarray:
    value = np.asarray(boxes, dtype=np.float64)
    if value.ndim != 2 or value.shape[1] != 6:
        raise ValueError("boxes must have shape [N, 6]")
    half = value[:, 3:6] * 0.5
    return np.concatenate((value[:, :3] - half, value[:, :3] + half), axis=1)


def pairwise_aabb_iou(predictions: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """Compute pairwise IoU between centre/size AABBs."""

    pred = center_size_to_minmax(predictions)
    target = center_size_to_minmax(targets)
    if len(pred) == 0 or len(target) == 0:
        return np.zeros((len(pred), len(target)), dtype=np.float64)
    intersection_min = np.maximum(pred[:, None, :3], target[None, :, :3])
    intersection_max = np.minimum(pred[:, None, 3:], target[None, :, 3:])
    intersection_size = np.maximum(intersection_max - intersection_min, 0.0)
    intersection = np.prod(intersection_size, axis=2)
    pred_volume = np.prod(pred[:, 3:] - pred[:, :3], axis=1)
    target_volume = np.prod(target[:, 3:] - target[:, :3], axis=1)
    union = pred_volume[:, None] + target_volume[None, :] - intersection
    return np.divide(
        intersection,
        union,
        out=np.zeros_like(intersection),
        where=union > 0.0,
    )


def _concatenate(parts: Iterable[np.ndarray], name: str) -> np.ndarray:
    values = list(parts)
    if not values:
        raise ValueError(f"No arrays collected for {name}")
    return np.concatenate(values, axis=0)


def _scene_training_arrays(
    diagnostics: SceneDiagnostics,
    prediction_corners: np.ndarray,
    transform: np.ndarray,
    gt_boxes: np.ndarray,
    config: BuildConfig,
) -> tuple[dict[str, np.ndarray], int]:
    count = len(diagnostics.points)
    point_count = diagnostics.points.shape[1]
    if count and diagnostics.result_indices.max() >= len(prediction_corners):
        raise ValueError(
            f"{diagnostics.scene_id}: result_indices exceed prediction count"
        )

    arrays = {
        "points_local": np.zeros((count, point_count, 3), dtype=np.float32),
        "point_mask": diagnostics.point_mask.astype(np.bool_, copy=True),
        "local_boxes": np.zeros((count, 6), dtype=np.float32),
        "quality_features": diagnostics.quality_features.astype(np.float32),
        "target_residual": np.zeros((count, 6), dtype=np.float32),
        "quality_target": np.zeros(count, dtype=np.float32),
        "geometry_mask": np.zeros(count, dtype=np.bool_),
        "original_iou": np.zeros(count, dtype=np.float32),
        "refined_iou": np.zeros(count, dtype=np.float32),
        "matched_gt_index": np.full(count, -1, dtype=np.int64),
        "target_center_local_unclipped": np.zeros((count, 3), dtype=np.float32),
        "target_dimensions_local_unclipped": np.zeros(
            (count, 3), dtype=np.float32
        ),
        "basis_world": np.zeros((count, 3, 3), dtype=np.float32),
        "result_indices": diagnostics.result_indices.astype(np.int64),
        "track_ids": diagnostics.track_ids.astype(np.int64),
    }
    rotation = transform[:3, :3]
    translation = transform[:3, 3]
    invalid_oriented_boxes = 0

    for row, result_index in enumerate(diagnostics.result_indices):
        corners = prediction_corners[int(result_index)]
        try:
            center_world, dimensions, basis_world = oriented_box_frame(corners)
            frame_valid = True
        except ValueError:
            # Invalid OBBs are retained as rejection examples.  A positive
            # local size keeps the archive trainable and the mask disables
            # their geometry loss.
            invalid_oriented_boxes += 1
            dimensions = np.maximum(
                corners.max(axis=0) - corners.min(axis=0), 1e-3
            )
            center_world = corners.mean(axis=0)
            basis_world = np.eye(3, dtype=np.float64)
            frame_valid = False

        arrays["local_boxes"][row, 3:6] = dimensions
        arrays["basis_world"][row] = basis_world
        valid = diagnostics.point_mask[row]
        arrays["points_local"][row, valid] = (
            diagnostics.points[row, valid] - center_world
        ) @ basis_world

        aligned_center = rotation @ center_world + translation
        aligned_basis = rotation @ basis_world
        original_aligned_dimensions = np.abs(aligned_basis) @ dimensions
        original_box = np.concatenate(
            (aligned_center, original_aligned_dimensions)
        )
        if not len(gt_boxes):
            continue
        pairwise = pairwise_aabb_iou(original_box[None, :], gt_boxes)[0]
        gt_index = int(np.argmax(pairwise))
        original_iou = float(pairwise[gt_index])
        target_gt = gt_boxes[gt_index]
        arrays["matched_gt_index"][row] = gt_index
        arrays["original_iou"][row] = original_iou

        # Exact orthogonal projection of the aligned GT centre into the
        # original OBB local coordinates.
        raw_center_local = aligned_basis.T @ (
            target_gt[:3] - aligned_center
        )
        # Find non-negative local dimensions whose aligned AABB best matches
        # the GT dimensions.  This is the evaluator-aware geometry target.
        raw_dimensions_local = nonnegative_least_squares_3x3(
            np.abs(aligned_basis), target_gt[3:6]
        )
        arrays["target_center_local_unclipped"][row] = raw_center_local
        arrays["target_dimensions_local_unclipped"][row] = (
            raw_dimensions_local
        )

        center_limit = float(config.max_center_fraction) * dimensions
        target_center_local = np.clip(
            raw_center_local, -center_limit, center_limit
        )
        dimension_ratio = np.divide(
            raw_dimensions_local,
            dimensions,
            out=np.ones(3, dtype=np.float64),
            where=dimensions > 0.0,
        )
        log_dimension_residual = np.log(
            np.maximum(dimension_ratio, 1e-8)
        )
        log_dimension_residual = np.clip(
            log_dimension_residual,
            -float(config.max_log_dimension_residual),
            float(config.max_log_dimension_residual),
        )
        target_dimensions_local = dimensions * np.exp(
            log_dimension_residual
        )
        center_fraction = target_center_local / dimensions
        arrays["target_residual"][row] = np.concatenate(
            (center_fraction, log_dimension_residual)
        )

        refined_center = aligned_center + aligned_basis @ target_center_local
        refined_dimensions = (
            np.abs(aligned_basis) @ target_dimensions_local
        )
        refined_box = np.concatenate((refined_center, refined_dimensions))
        refined_iou = float(
            pairwise_aabb_iou(refined_box[None, :], target_gt[None, :])[0, 0]
        )
        arrays["refined_iou"][row] = refined_iou
        geometry_valid = bool(
            frame_valid
            and original_iou >= float(config.min_match_iou)
            and refined_iou
            > original_iou + float(config.improvement_epsilon)
        )
        arrays["geometry_mask"][row] = geometry_valid
        arrays["quality_target"][row] = float(geometry_valid)

    return arrays, invalid_oriented_boxes


def build_oriented_refiner_dataset(config: BuildConfig) -> BuildSummary:
    """Build and atomically write a deterministic B5-v2 dataset."""

    config = config.validated()
    scenes = read_scene_ids(config.scene_list)
    collected: dict[str, list[np.ndarray]] = {}
    expected_point_count: int | None = None
    invalid_total = 0

    for scene_id in scenes:
        diagnostics = load_scene_diagnostics(
            resolve_diagnostic_path(config.diagnostics_root, scene_id),
            expected_scene_id=scene_id,
        )
        if expected_point_count is None:
            expected_point_count = diagnostics.points.shape[1]
        elif diagnostics.points.shape[1] != expected_point_count:
            raise ValueError(
                "All diagnostics must use the same point count; "
                f"{scene_id} has {diagnostics.points.shape[1]}, "
                f"expected {expected_point_count}"
            )
        prediction_corners = load_prediction_corners(
            Path(config.prediction_root) / f"{scene_id}_boxes.pkl"
        )
        transform = load_axis_alignment(config.scan_root, scene_id)
        gt_boxes = load_gt_boxes(config.gt_root, scene_id)
        arrays, invalid_count = _scene_training_arrays(
            diagnostics,
            prediction_corners,
            transform,
            gt_boxes,
            config,
        )
        invalid_total += invalid_count
        arrays["scene_ids"] = np.full(
            len(diagnostics.points), scene_id, dtype=f"<U{len(scene_id)}"
        )
        for name, value in arrays.items():
            collected.setdefault(name, []).append(value)

    output_arrays = {
        name: _concatenate(parts, name) for name, parts in collected.items()
    }
    sample_count = len(output_arrays["geometry_mask"])
    if sample_count < 2:
        raise ValueError("B5-v2 dataset requires at least two samples")
    if not output_arrays["geometry_mask"].any():
        raise ValueError(
            "No reachable target improves IoU; check coordinate roots and "
            "matching thresholds"
        )
    if output_arrays["geometry_mask"].all():
        raise ValueError(
            "B5-v2 rejection training requires at least one negative sample"
        )
    output_arrays.update(
        {
            "schema": np.asarray(DATASET_SCHEMA),
            "format_version": np.asarray(
                DATASET_FORMAT_VERSION, dtype=np.int64
            ),
            "coordinate_frame": np.asarray("box_local"),
            "quality_feature_names": np.asarray(
                QUALITY_FEATURE_NAMES, dtype=np.str_
            ),
            "max_center_fraction": np.asarray(
                config.max_center_fraction, dtype=np.float32
            ),
            "max_log_dimension_residual": np.asarray(
                config.max_log_dimension_residual, dtype=np.float32
            ),
        }
    )
    output = Path(config.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    try:
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, **output_arrays)
        temporary.replace(output)
    finally:
        if temporary.exists():
            temporary.unlink()

    positive_count = int(np.count_nonzero(output_arrays["geometry_mask"]))
    return BuildSummary(
        scenes=len(scenes),
        samples=sample_count,
        geometry_positives=positive_count,
        quality_negatives=sample_count - positive_count,
        invalid_oriented_boxes=invalid_total,
        output=output,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build box-local, evaluator-aware B5-v2 supervision."
    )
    parser.add_argument("--diagnostics-root", type=Path, required=True)
    parser.add_argument("--prediction-root", type=Path, required=True)
    parser.add_argument("--scan-root", type=Path, required=True)
    parser.add_argument("--gt-root", type=Path, required=True)
    parser.add_argument("--scene-list", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-match-iou", type=float, default=0.15)
    parser.add_argument("--improvement-epsilon", type=float, default=1e-4)
    parser.add_argument("--max-center-fraction", type=float, default=0.15)
    parser.add_argument(
        "--max-log-dimension-residual",
        type=float,
        default=float(np.log(1.25)),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = build_oriented_refiner_dataset(
        BuildConfig(
            diagnostics_root=args.diagnostics_root,
            prediction_root=args.prediction_root,
            scan_root=args.scan_root,
            gt_root=args.gt_root,
            scene_list=args.scene_list,
            output=args.output,
            min_match_iou=args.min_match_iou,
            improvement_epsilon=args.improvement_epsilon,
            max_center_fraction=args.max_center_fraction,
            max_log_dimension_residual=args.max_log_dimension_residual,
        )
    )
    print(
        "Built B5-v2 dataset: "
        f"scenes={summary.scenes}, samples={summary.samples}, "
        f"geometry_positives={summary.geometry_positives}, "
        f"quality_negatives={summary.quality_negatives}, "
        f"invalid_obb={summary.invalid_oriented_boxes}, "
        f"path={summary.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
