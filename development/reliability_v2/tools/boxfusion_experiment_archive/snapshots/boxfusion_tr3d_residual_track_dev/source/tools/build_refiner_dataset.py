#!/usr/bin/env python3
"""Build an offline BoxRefiner training set from runtime diagnostics.

The runtime diagnostics are deliberately stored as ``allow_pickle=False``
NPZ files.  Predicted boxes and object points are in the unaligned ScanNet
world frame, while ``*_bbox.npy`` ground truth is already axis aligned.  This
tool applies each scene's ``axisAlignment`` before matching predictions to
ground truth by maximum AABB IoU.

The resulting compressed NPZ contains the arrays consumed by the BoxRefiner
and quality-score trainers:

``points, point_mask, boxes, quality_features, target_boxes, target_iou,
scene_ids``.

Ground truth is used only here, never by online inference.
"""

from __future__ import annotations

import argparse
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


QUALITY_FEATURE_DIM = 12
REQUIRED_DIAGNOSTIC_FIELDS = frozenset(
    {"boxes", "scores", "quality_features", "points", "point_mask"}
)
SCENE_PATTERN = re.compile(r"(scene\d{4}_\d{2})")


@dataclass(frozen=True)
class BuildConfig:
    diagnostics_root: Path
    scan_root: Path
    gt_root: Path
    scene_list: Path
    output: Path
    min_iou: float = 0.15
    include_negatives: bool = False

    def validated(self) -> "BuildConfig":
        for name in (
            "diagnostics_root",
            "scan_root",
            "gt_root",
        ):
            path = Path(getattr(self, name))
            if not path.is_dir():
                raise FileNotFoundError(f"{name} is not a directory: {path}")
        if not Path(self.scene_list).is_file():
            raise FileNotFoundError(self.scene_list)
        if (
            not np.isscalar(self.min_iou)
            or not np.isfinite(self.min_iou)
            or not 0.0 <= float(self.min_iou) <= 1.0
        ):
            raise ValueError("min_iou must be a finite value in [0, 1]")
        if not isinstance(self.include_negatives, (bool, np.bool_)):
            raise TypeError("include_negatives must be Boolean")
        return self


@dataclass(frozen=True)
class SceneDiagnostics:
    scene_id: str
    boxes: np.ndarray
    scores: np.ndarray
    quality_features: np.ndarray
    points: np.ndarray
    point_mask: np.ndarray


@dataclass(frozen=True)
class BuildSummary:
    scenes: int
    input_predictions: int
    positives: int
    negatives: int
    output_samples: int
    output: Path


def read_scene_ids(path: Path) -> list[str]:
    """Read a non-empty, duplicate-free ScanNet scene list."""

    path = Path(path)
    scenes = [line.strip() for line in path.read_text().splitlines()]
    scenes = [scene for scene in scenes if scene]
    if not scenes:
        raise ValueError(f"No scenes found in {path}")
    invalid = [scene for scene in scenes if SCENE_PATTERN.fullmatch(scene) is None]
    if invalid:
        raise ValueError(f"Invalid ScanNet scene id: {invalid[0]!r}")
    if len(set(scenes)) != len(scenes):
        raise ValueError(f"Duplicate scene ids in {path}")
    return scenes


def load_axis_alignment(scan_root: Path, scene_id: str) -> np.ndarray:
    """Load and strictly validate a ScanNet axis-alignment transform."""

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
    if values is None:
        raise ValueError(f"axisAlignment missing from {metadata}")
    if values.size != 16 or not np.isfinite(values).all():
        raise ValueError(f"Invalid axisAlignment in {metadata}")
    transform = values.reshape(4, 4).astype(np.float64, copy=False)
    if not np.allclose(
        transform[3], np.asarray([0.0, 0.0, 0.0, 1.0]), atol=1e-6
    ):
        raise ValueError(f"axisAlignment is not homogeneous in {metadata}")
    rotation = transform[:3, :3]
    if not np.allclose(rotation @ rotation.T, np.eye(3), atol=2e-3):
        raise ValueError(f"axisAlignment is not rigid in {metadata}")
    if not np.isclose(abs(np.linalg.det(rotation)), 1.0, atol=2e-3):
        raise ValueError(f"axisAlignment rotation is singular in {metadata}")
    return transform


def _validate_boxes(boxes: np.ndarray, name: str) -> np.ndarray:
    value = np.asarray(boxes)
    if value.ndim != 2 or value.shape[1] != 6:
        raise ValueError(f"{name} must have shape [N, 6], got {value.shape}")
    if not np.issubdtype(value.dtype, np.number):
        raise TypeError(f"{name} must be numeric")
    value = np.asarray(value, dtype=np.float64)
    if not np.isfinite(value).all():
        raise ValueError(f"{name} must be finite")
    if np.any(value[:, 3:6] <= 0.0):
        raise ValueError(f"{name} dimensions must be positive")
    return value


def center_size_to_minmax(boxes: np.ndarray) -> np.ndarray:
    """Convert ``[cx,cy,cz,dx,dy,dz]`` AABBs to min/max form."""

    value = _validate_boxes(boxes, "boxes")
    half = value[:, 3:6] * 0.5
    return np.concatenate((value[:, :3] - half, value[:, :3] + half), axis=1)


def minmax_to_center_size(boxes: np.ndarray) -> np.ndarray:
    value = np.asarray(boxes, dtype=np.float64)
    if value.ndim != 2 or value.shape[1] != 6:
        raise ValueError(f"min/max boxes must have shape [N, 6], got {value.shape}")
    if not np.isfinite(value).all():
        raise ValueError("min/max boxes must be finite")
    dimensions = value[:, 3:6] - value[:, :3]
    if np.any(dimensions <= 0.0):
        raise ValueError("min/max box dimensions must be positive")
    centers = (value[:, :3] + value[:, 3:6]) * 0.5
    return np.concatenate((centers, dimensions), axis=1)


def transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    """Apply a homogeneous transform to points with arbitrary leading axes."""

    value = np.asarray(points, dtype=np.float64)
    transform = np.asarray(transform, dtype=np.float64)
    if value.ndim < 1 or value.shape[-1] != 3:
        raise ValueError("points must end in dimension 3")
    if transform.shape != (4, 4) or not np.isfinite(transform).all():
        raise ValueError("transform must be a finite [4, 4] matrix")
    if not np.isfinite(value).all():
        raise ValueError("points must be finite")
    return value @ transform[:3, :3].T + transform[:3, 3]


def align_center_size_boxes(
    boxes: np.ndarray, transform: np.ndarray
) -> np.ndarray:
    """Transform AABB corners and return their axis-aligned enclosure."""

    value = _validate_boxes(boxes, "boxes")
    offsets = np.asarray(
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
    corners = (
        value[:, None, :3]
        + offsets[None, :, :] * value[:, None, 3:6] * 0.5
    )
    transformed = transform_points(corners, transform)
    minmax = np.concatenate(
        (transformed.min(axis=1), transformed.max(axis=1)), axis=1
    )
    return minmax_to_center_size(minmax)


def pairwise_aabb_iou(predictions: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """Compute pairwise IoU between center/size AABBs."""

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


def _parse_scene_value(value: np.ndarray, path: Path) -> str:
    array = np.asarray(value)
    if array.dtype.hasobject:
        raise ValueError(f"scene_id in {path} must not use object dtype")
    if array.ndim == 0:
        scalar = array.item()
    elif array.ndim == 1 and array.size == 1:
        scalar = array[0]
    else:
        raise ValueError(f"scene_id in {path} must be a scalar string")
    if isinstance(scalar, bytes):
        scalar = scalar.decode("utf-8")
    if not isinstance(scalar, str) or SCENE_PATTERN.fullmatch(scalar) is None:
        raise ValueError(f"Invalid scene_id in {path}: {scalar!r}")
    return scalar


def infer_scene_from_filename(path: Path) -> str:
    matches = SCENE_PATTERN.findall(path.stem)
    if len(set(matches)) != 1:
        raise ValueError(
            f"Cannot infer one ScanNet scene id from diagnostic filename {path}"
        )
    return matches[0]


def load_scene_diagnostics(
    path: Path, expected_scene_id: str | None = None
) -> SceneDiagnostics:
    """Load one safe NPZ diagnostic file and validate every tensor."""

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    try:
        with np.load(path, allow_pickle=False) as payload:
            keys = set(payload.files)
            missing = REQUIRED_DIAGNOSTIC_FIELDS - keys
            if missing:
                raise ValueError(
                    f"{path} is missing fields: {sorted(missing)}"
                )
            scene_id = (
                _parse_scene_value(payload["scene_id"], path)
                if "scene_id" in keys
                else infer_scene_from_filename(path)
            )
            boxes = np.asarray(payload["boxes"])
            scores = np.asarray(payload["scores"])
            quality = np.asarray(payload["quality_features"])
            points = np.asarray(payload["points"])
            point_mask = np.asarray(payload["point_mask"])
    except ValueError as error:
        if "Object arrays cannot be loaded" in str(error):
            raise ValueError(f"{path} contains pickled/object arrays") from error
        raise

    if expected_scene_id is not None and scene_id != expected_scene_id:
        raise ValueError(
            f"Diagnostic scene_id {scene_id!r} does not match "
            f"requested scene {expected_scene_id!r}"
        )
    boxes = _validate_boxes(boxes, "boxes")
    sample_count = len(boxes)
    if scores.shape != (sample_count,):
        raise ValueError(f"scores must have shape [{sample_count}]")
    if not np.issubdtype(scores.dtype, np.number):
        raise TypeError("scores must be numeric")
    scores = np.asarray(scores, dtype=np.float64)
    if (
        not np.isfinite(scores).all()
        or np.any(scores < 0.0)
        or np.any(scores > 1.0)
    ):
        raise ValueError("scores must be finite and lie in [0, 1]")

    if quality.shape != (sample_count, QUALITY_FEATURE_DIM):
        raise ValueError(
            "quality_features must have shape "
            f"[{sample_count}, {QUALITY_FEATURE_DIM}]"
        )
    if not np.issubdtype(quality.dtype, np.number):
        raise TypeError("quality_features must be numeric")
    quality = np.asarray(quality, dtype=np.float64)
    if (
        not np.isfinite(quality).all()
        or np.any(quality < 0.0)
        or np.any(quality > 1.0)
    ):
        raise ValueError(
            "quality_features must be finite and lie in [0, 1]"
        )

    if points.ndim != 3 or points.shape[0] != sample_count or points.shape[2] != 3:
        raise ValueError("points must have shape [N, P, 3]")
    if points.shape[1] < 1:
        raise ValueError("points must contain at least one point slot")
    if not np.issubdtype(points.dtype, np.number):
        raise TypeError("points must be numeric")
    points = np.asarray(points, dtype=np.float64)
    if not np.isfinite(points).all():
        raise ValueError("points must be finite")
    if point_mask.shape != points.shape[:2]:
        raise ValueError("point_mask must have shape [N, P]")
    if point_mask.dtype != np.bool_:
        raise TypeError("point_mask must have Boolean dtype")
    if sample_count and np.any(~point_mask.any(axis=1)):
        raise ValueError("every diagnostic sample must contain a valid point")

    return SceneDiagnostics(
        scene_id=scene_id,
        boxes=boxes,
        scores=scores,
        quality_features=quality,
        points=points,
        point_mask=point_mask,
    )


def load_gt_boxes(gt_root: Path, scene_id: str) -> np.ndarray:
    path = Path(gt_root) / f"{scene_id}_bbox.npy"
    if not path.is_file():
        raise FileNotFoundError(path)
    values = np.load(path, allow_pickle=False)
    if values.ndim != 2 or values.shape[1] < 6:
        raise ValueError(f"GT boxes in {path} must have shape [N, >=6]")
    return _validate_boxes(values[:, :6], "target_boxes")


def resolve_diagnostic_path(root: Path, scene_id: str) -> Path:
    root = Path(root)
    preferred = (
        root / f"{scene_id}.npz",
        root / f"{scene_id}_diagnostics.npz",
        root / f"{scene_id}_online_refinement.npz",
    )
    existing = [path for path in preferred if path.is_file()]
    if len(existing) == 1:
        return existing[0]
    if len(existing) > 1:
        raise ValueError(
            f"Multiple diagnostic files found for {scene_id}: {existing}"
        )
    matches = sorted(root.glob(f"{scene_id}*.npz"))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected exactly one diagnostic NPZ for {scene_id} in {root}; "
            f"found {len(matches)}"
        )
    return matches[0]


def _concatenate(parts: Iterable[np.ndarray], name: str) -> np.ndarray:
    values = list(parts)
    if not values:
        raise ValueError(f"No arrays collected for {name}")
    return np.concatenate(values, axis=0)


def build_refiner_dataset(config: BuildConfig) -> BuildSummary:
    """Build and atomically write one deterministic merged training NPZ."""

    config = config.validated()
    scenes = read_scene_ids(config.scene_list)
    collected: dict[str, list[np.ndarray]] = {
        name: []
        for name in (
            "points",
            "point_mask",
            "boxes",
            "quality_features",
            "target_boxes",
            "target_iou",
            "scene_ids",
        )
    }
    point_count: int | None = None
    input_predictions = 0
    positive_count = 0
    negative_count = 0

    for scene_id in scenes:
        diagnostic_path = resolve_diagnostic_path(
            config.diagnostics_root, scene_id
        )
        diagnostics = load_scene_diagnostics(
            diagnostic_path, expected_scene_id=scene_id
        )
        if point_count is None:
            point_count = diagnostics.points.shape[1]
        elif diagnostics.points.shape[1] != point_count:
            raise ValueError(
                "All diagnostics must use the same point count; "
                f"{scene_id} has {diagnostics.points.shape[1]}, "
                f"expected {point_count}"
            )

        transform = load_axis_alignment(config.scan_root, scene_id)
        aligned_boxes = align_center_size_boxes(
            diagnostics.boxes, transform
        )
        aligned_points = np.zeros_like(diagnostics.points, dtype=np.float64)
        valid = diagnostics.point_mask
        aligned_points[valid] = transform_points(
            diagnostics.points[valid], transform
        )
        targets = load_gt_boxes(config.gt_root, scene_id)
        iou = pairwise_aabb_iou(aligned_boxes, targets)
        if len(targets):
            target_index = np.argmax(iou, axis=1)
            target_iou = iou[np.arange(len(aligned_boxes)), target_index]
            matched_targets = targets[target_index].copy()
        else:
            target_iou = np.zeros(len(aligned_boxes), dtype=np.float64)
            matched_targets = aligned_boxes.copy()

        positives = target_iou >= float(config.min_iou)
        matched_targets[~positives] = aligned_boxes[~positives]
        keep = (
            np.ones(len(aligned_boxes), dtype=bool)
            if config.include_negatives
            else positives
        )
        input_predictions += len(aligned_boxes)
        positive_count += int(np.count_nonzero(positives))
        if config.include_negatives:
            negative_count += int(np.count_nonzero(~positives))

        collected["points"].append(aligned_points[keep].astype(np.float32))
        collected["point_mask"].append(valid[keep].astype(bool, copy=True))
        collected["boxes"].append(aligned_boxes[keep].astype(np.float32))
        collected["quality_features"].append(
            diagnostics.quality_features[keep].astype(np.float32)
        )
        collected["target_boxes"].append(
            matched_targets[keep].astype(np.float32)
        )
        collected["target_iou"].append(
            target_iou[keep].astype(np.float32)
        )
        collected["scene_ids"].append(
            np.full(
                int(np.count_nonzero(keep)),
                scene_id,
                dtype=f"<U{len(scene_id)}",
            )
        )

    output_arrays = {
        name: _concatenate(parts, name) for name, parts in collected.items()
    }
    output_samples = len(output_arrays["boxes"])
    if output_samples == 0:
        raise ValueError(
            "No samples passed min_iou; lower --min-iou or use "
            "--include-negatives"
        )
    if any(len(value) != output_samples for value in output_arrays.values()):
        raise RuntimeError("Internal sample count mismatch")
    output_arrays["feature_names"] = np.asarray(
        QUALITY_FEATURE_NAMES, dtype=np.str_
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

    return BuildSummary(
        scenes=len(scenes),
        input_predictions=input_predictions,
        positives=positive_count,
        negatives=negative_count,
        output_samples=output_samples,
        output=output,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Axis-align runtime diagnostics and build a BoxRefiner NPZ."
        )
    )
    parser.add_argument("--diagnostics-root", type=Path, required=True)
    parser.add_argument("--scan-root", type=Path, required=True)
    parser.add_argument("--gt-root", type=Path, required=True)
    parser.add_argument("--scene-list", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-iou", type=float, default=0.15)
    parser.add_argument("--include-negatives", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = build_refiner_dataset(
        BuildConfig(
            diagnostics_root=args.diagnostics_root,
            scan_root=args.scan_root,
            gt_root=args.gt_root,
            scene_list=args.scene_list,
            output=args.output,
            min_iou=args.min_iou,
            include_negatives=args.include_negatives,
        )
    )
    print(
        "Built BoxRefiner dataset: "
        f"scenes={summary.scenes}, "
        f"input={summary.input_predictions}, "
        f"positives={summary.positives}, "
        f"negatives={summary.negatives}, "
        f"output={summary.output_samples}, "
        f"path={summary.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
