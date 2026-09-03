#!/usr/bin/env python3
"""Train the class-agnostic P1 residual voxel proposal head.

The online observer deliberately records inputs, not labels.  This utility
constructs labels offline from ScanNet *training* ground truth and frozen B6
predictions:

1. GT boxes already covered by B6 are removed.
2. The closest ``K`` residual voxels inside each remaining GT box are
   assigned to that box (the class-agnostic TR3D assignment principle).
3. The head predicts one objectness logit and six AABB parameters:
   ``center - voxel_center`` in metres and ``log(size_in_metres)``.

The output is a plain ``torch.save`` dictionary.  It contains no pickled model
object: only schema/config metadata, tensors from ``state_dict``, metrics, and
explicit train-scene provenance.  Runtime reconstructs
``ResidualVoxelProposalHead`` from ``model_config``.

Only trusted locally produced BoxFusion pickle files should be supplied.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import pickle
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


CHECKPOINT_SCHEMA = "boxfusion.p1_residual_head.v1"
TRAINING_SCHEMA = "boxfusion.p1_residual_training.v1"
DEFAULT_THRESHOLDS = (0.15, 0.25, 0.50)
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
class SceneVoxelInputs:
    """Validated observer inputs for one ScanNet scene."""

    scene_id: str
    features: np.ndarray
    centers_world: np.ndarray
    offsets: np.ndarray
    feature_names: tuple[str, ...]
    diagnostic_path: Path


@dataclass(frozen=True)
class ResidualTargets:
    """Dense voxel targets; ``assigned_gt == -1`` denotes a negative."""

    objectness: np.ndarray
    regression: np.ndarray
    assigned_gt: np.ndarray


@dataclass(frozen=True)
class ResidualTrainingData:
    features: np.ndarray
    objectness: np.ndarray
    regression: np.ndarray
    scene_ids: np.ndarray
    feature_names: tuple[str, ...]
    scene_summaries: tuple[Mapping[str, Any], ...]


class _FallbackResidualVoxelProposalHead(nn.Module):
    """Exact fallback for the runtime head's intentionally small MLP."""

    def __init__(self, input_dim: int, hidden_dim: int = 64) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(int(input_dim), int(hidden_dim)),
            nn.ReLU(inplace=False),
            nn.Linear(int(hidden_dim), int(hidden_dim)),
            nn.ReLU(inplace=False),
        )
        self.objectness = nn.Linear(int(hidden_dim), 1)
        self.regression = nn.Linear(int(hidden_dim), 6)

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.backbone(features)
        return self.objectness(hidden).squeeze(-1), self.regression(hidden)


def _head_class() -> type[nn.Module]:
    """Use the runtime implementation when available, otherwise its twin."""

    try:
        from boxfusion.residual_proposal import ResidualVoxelProposalHead
    except (ImportError, AttributeError):
        return _FallbackResidualVoxelProposalHead
    return ResidualVoxelProposalHead


def read_scene_ids(path: str | os.PathLike[str], *, role: str) -> tuple[str, ...]:
    scene_path = Path(path)
    if not scene_path.is_file():
        raise FileNotFoundError(f"{role} scene list not found: {scene_path}")
    scenes = tuple(
        line.strip()
        for line in scene_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    if not scenes:
        raise ValueError(f"{role} scene list is empty: {scene_path}")
    if len(set(scenes)) != len(scenes):
        raise ValueError(f"{role} scene list contains duplicates: {scene_path}")
    invalid = [
        scene
        for scene in scenes
        if not (
            len(scene) == 12
            and scene.startswith("scene")
            and scene[5:9].isdigit()
            and scene[9] == "_"
            and scene[10:].isdigit()
        )
    ]
    if invalid:
        raise ValueError(f"invalid ScanNet scene id in {role} list: {invalid[0]!r}")
    return scenes


def validate_train_split(
    train_scenes: Iterable[str], forbidden_scenes: Iterable[str]
) -> tuple[str, ...]:
    train = tuple(str(scene) for scene in train_scenes)
    forbidden = frozenset(str(scene) for scene in forbidden_scenes)
    overlap = sorted(set(train) & forbidden)
    if overlap:
        raise ValueError(
            "P1 training scenes overlap the forbidden validation split: "
            + ", ".join(overlap[:16])
        )
    return train


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _scalar_text(value: np.ndarray, name: str, path: Path) -> str:
    array = np.asarray(value)
    if array.shape != () or array.dtype.hasobject:
        raise ValueError(f"{path}: {name} must be a non-object scalar string")
    item = array.item()
    if isinstance(item, bytes):
        item = item.decode("utf-8")
    if not isinstance(item, str) or not item:
        raise ValueError(f"{path}: {name} must be a non-empty string")
    return item


def _feature_names(
    archive: Mapping[str, np.ndarray], input_dim: int, path: Path
) -> tuple[str, ...]:
    from boxfusion.residual_proposal import P1_FEATURE_NAMES

    if "p1_feature_names" not in archive:
        raise ValueError(f"{path}: missing fixed p1_feature_names")
    raw = np.asarray(archive["p1_feature_names"])
    if raw.dtype.hasobject:
        raise TypeError(f"{path}: p1_feature_names must not use object dtype")
    if raw.shape == ():
        text = _scalar_text(raw, "p1_feature_names", path)
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            parsed = [item.strip() for item in text.split(",") if item.strip()]
        names = tuple(str(item) for item in parsed)
    elif raw.ndim == 1:
        names = tuple(str(item) for item in raw.tolist())
    else:
        raise ValueError(f"{path}: p1_feature_names must be scalar JSON or [F]")
    if len(names) != input_dim or any(not name for name in names):
        raise ValueError(
            f"{path}: p1_feature_names has {len(names)} names for F={input_dim}"
        )
    if len(set(names)) != len(names):
        raise ValueError(f"{path}: p1_feature_names contains duplicates")
    if names != P1_FEATURE_NAMES:
        raise ValueError(f"{path}: P1 feature schema disagrees with runtime")
    return names


def _scalar_bool(
    archive: Mapping[str, np.ndarray], name: str, path: Path
) -> bool:
    array = np.asarray(archive[name])
    if array.shape != () or array.dtype != np.dtype(bool):
        raise ValueError(f"{path}: {name} must be a Boolean scalar")
    return bool(array.item())


def _scalar_integer(
    archive: Mapping[str, np.ndarray], name: str, path: Path
) -> int:
    array = np.asarray(archive[name])
    if (
        array.shape != ()
        or not np.issubdtype(array.dtype, np.integer)
        or np.issubdtype(array.dtype, np.bool_)
    ):
        raise ValueError(f"{path}: {name} must be an integer scalar")
    return int(array.item())


def _validate_observer_contract(
    archive: Mapping[str, np.ndarray], path: Path
) -> None:
    from boxfusion.residual_proposal import (
        P1_DIAGNOSTIC_SCHEMA,
        P1_FEATURE_DIM,
    )

    expected_text = {
        "p1_schema": P1_DIAGNOSTIC_SCHEMA,
        "p1_stage": "P1",
        "p1_profile": "p1_residual_proposal_observer",
    }
    expected_bool = {
        "p1_enabled": True,
        "p1_observer_only": True,
        "p1_uses_ground_truth": False,
        "p1_mutation_enabled": False,
        "p1_complete": True,
        "p1_class_agnostic": True,
    }
    required = {
        *expected_text,
        *expected_bool,
        "p1_applied_count",
        "p1_regression_dim",
        "p1_feature_names",
        "scene_id",
    }
    missing = sorted(required - set(archive))
    if missing:
        raise ValueError(f"{path}: missing P1 safety fields {missing}")
    for name, expected in expected_text.items():
        observed = _scalar_text(archive[name], name, path)
        if observed != expected:
            raise ValueError(
                f"{path}: {name}={observed!r}, expected {expected!r}"
            )
    for name, expected in expected_bool.items():
        observed = _scalar_bool(archive, name, path)
        if observed is not expected:
            raise ValueError(
                f"{path}: unsafe {name}={observed}, expected {expected}"
            )
    if _scalar_integer(archive, "p1_applied_count", path) != 0:
        raise ValueError(f"{path}: observer diagnostic applied output rows")
    if _scalar_integer(archive, "p1_regression_dim", path) != 6:
        raise ValueError(f"{path}: P1 regression must remain 6-D")
    features = np.asarray(archive["p1_voxel_features"])
    if features.ndim != 2 or features.shape[1] != P1_FEATURE_DIM:
        raise ValueError(
            f"{path}: P1 features must have fixed dimension {P1_FEATURE_DIM}"
        )


def load_scene_voxels(
    path: str | os.PathLike[str], *, expected_scene_id: str | None = None
) -> SceneVoxelInputs:
    """Load a strict, pickle-free P1 observer diagnostic."""

    diagnostic_path = Path(path)
    if not diagnostic_path.is_file():
        raise FileNotFoundError(diagnostic_path)
    with np.load(diagnostic_path, allow_pickle=False) as archive_obj:
        keys = set(archive_obj.files)
        required = {
            "p1_voxel_features",
            "p1_voxel_centers",
            "p1_voxel_offsets",
        }
        missing = sorted(required - keys)
        if missing:
            raise ValueError(f"{diagnostic_path}: missing fields {missing}")
        archive = {
            key: np.array(archive_obj[key], copy=True)
            for key in archive_obj.files
        }
    _validate_observer_contract(archive, diagnostic_path)

    scene_id = expected_scene_id
    if "scene_id" in archive:
        stored = _scalar_text(archive["scene_id"], "scene_id", diagnostic_path)
        if scene_id is not None and stored != scene_id:
            raise ValueError(
                f"{diagnostic_path}: scene_id {stored!r} != {scene_id!r}"
            )
        scene_id = stored
    if scene_id is None:
        scene_id = diagnostic_path.name.removesuffix("_tracks.npz")

    features_raw = np.asarray(archive["p1_voxel_features"])
    centers_raw = np.asarray(archive["p1_voxel_centers"])
    offsets_raw = np.asarray(archive["p1_voxel_offsets"])
    if (
        features_raw.ndim != 2
        or not np.issubdtype(features_raw.dtype, np.floating)
        or not np.isfinite(features_raw).all()
    ):
        raise ValueError(
            f"{diagnostic_path}: p1_voxel_features must be finite float [V,F]"
        )
    if (
        centers_raw.shape != (features_raw.shape[0], 3)
        or not np.issubdtype(centers_raw.dtype, np.floating)
        or not np.isfinite(centers_raw).all()
    ):
        raise ValueError(
            f"{diagnostic_path}: p1_voxel_centers must be finite float [V,3]"
        )
    if (
        offsets_raw.ndim != 1
        or len(offsets_raw) < 1
        or not np.issubdtype(offsets_raw.dtype, np.integer)
    ):
        raise ValueError(
            f"{diagnostic_path}: p1_voxel_offsets must be integer [S+1]"
        )
    offsets = np.asarray(offsets_raw, dtype=np.int64)
    if (
        int(offsets[0]) != 0
        or int(offsets[-1]) != len(features_raw)
        or np.any(offsets[1:] < offsets[:-1])
    ):
        raise ValueError(
            f"{diagnostic_path}: invalid p1_voxel_offsets ragged boundaries"
        )
    names = _feature_names(archive, int(features_raw.shape[1]), diagnostic_path)
    return SceneVoxelInputs(
        scene_id=scene_id,
        features=np.ascontiguousarray(features_raw, dtype=np.float32),
        centers_world=np.ascontiguousarray(centers_raw, dtype=np.float32),
        offsets=offsets,
        feature_names=names,
        diagnostic_path=diagnostic_path,
    )


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
        raise ValueError(f"boxes must have shape [N,6], got {values.shape}")
    if not np.isfinite(values).all() or (
        len(values) and np.any(values[:, 3:] <= 0.0)
    ):
        raise ValueError("boxes contain invalid center/size values")
    return values[:, None, :3] + _CORNER_SIGNS[None] * (
        0.5 * values[:, None, 3:]
    )


def transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    values = np.asarray(points, dtype=np.float64)
    matrix = np.asarray(transform, dtype=np.float64)
    if values.shape[-1] != 3 or matrix.shape != (4, 4):
        raise ValueError("points must end in 3 and transform must be [4,4]")
    if not np.isfinite(values).all() or not np.isfinite(matrix).all():
        raise ValueError("points and transform must be finite")
    return values @ matrix[:3, :3].T + matrix[:3, 3]


def corners_to_minmax(corners: np.ndarray) -> np.ndarray:
    values = np.asarray(corners, dtype=np.float64)
    if values.ndim != 3 or values.shape[1:] != (8, 3):
        raise ValueError(f"corners must have shape [N,8,3], got {values.shape}")
    if not np.isfinite(values).all():
        raise ValueError("corners must be finite")
    if not len(values):
        return np.empty((0, 6), dtype=np.float64)
    return np.concatenate((values.min(axis=1), values.max(axis=1)), axis=1)


def minmax_to_center_size(boxes: np.ndarray) -> np.ndarray:
    values = np.asarray(boxes, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 6:
        raise ValueError(f"minmax boxes must have shape [N,6], got {values.shape}")
    sizes = values[:, 3:] - values[:, :3]
    if not np.isfinite(values).all() or (len(values) and np.any(sizes <= 0.0)):
        raise ValueError("minmax boxes contain invalid extents")
    return np.concatenate(((values[:, :3] + values[:, 3:]) * 0.5, sizes), axis=1)


def center_size_to_minmax(boxes: np.ndarray) -> np.ndarray:
    values = np.asarray(boxes, dtype=np.float64)
    corners = center_size_to_corners(values)
    return corners_to_minmax(corners)


def pairwise_aabb_iou(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    left = np.asarray(first, dtype=np.float64)
    right = np.asarray(second, dtype=np.float64)
    if left.ndim != 2 or left.shape[1] != 6:
        raise ValueError("first boxes must have shape [N,6]")
    if right.ndim != 2 or right.shape[1] != 6:
        raise ValueError("second boxes must have shape [M,6]")
    if not len(left) or not len(right):
        return np.zeros((len(left), len(right)), dtype=np.float64)
    intersection_size = np.maximum(
        np.minimum(left[:, None, 3:], right[None, :, 3:])
        - np.maximum(left[:, None, :3], right[None, :, :3]),
        0.0,
    )
    intersection = np.prod(intersection_size, axis=2)
    left_volume = np.prod(left[:, 3:] - left[:, :3], axis=1)
    right_volume = np.prod(right[:, 3:] - right[:, :3], axis=1)
    union = left_volume[:, None] + right_volume[None, :] - intersection
    return np.divide(
        intersection,
        union,
        out=np.zeros_like(intersection),
        where=union > 0.0,
    )


def load_gt_boxes(path: str | os.PathLike[str]) -> np.ndarray:
    gt_path = Path(path)
    if not gt_path.is_file():
        raise FileNotFoundError(gt_path)
    payload = np.load(gt_path, allow_pickle=False)
    if payload.ndim != 2 or payload.shape[1] < 6:
        raise ValueError(f"{gt_path}: GT must have shape [N,>=6]")
    boxes = np.asarray(payload[:, :6], dtype=np.float64)
    center_size_to_corners(boxes)  # strict validation
    return boxes


def load_prediction_corners(path: str | os.PathLike[str]) -> np.ndarray:
    prediction_path = Path(path)
    if not prediction_path.is_file():
        raise FileNotFoundError(prediction_path)
    with prediction_path.open("rb") as handle:
        payload = pickle.load(handle)  # noqa: S301 - trusted local output
    if (
        not isinstance(payload, (list, tuple))
        or len(payload) != 1
        or not isinstance(payload[0], (list, tuple))
    ):
        raise ValueError(f"{prediction_path}: invalid BoxFusion prediction batch")
    corners: list[np.ndarray] = []
    for index, detection in enumerate(payload[0]):
        if not isinstance(detection, (list, tuple)) or len(detection) != 3:
            raise ValueError(f"{prediction_path}: invalid detection {index}")
        value = np.asarray(detection[1])
        if (
            value.shape != (8, 3)
            or not np.issubdtype(value.dtype, np.number)
            or not np.isfinite(value).all()
        ):
            raise ValueError(f"{prediction_path}: invalid corners at {index}")
        corners.append(np.asarray(value, dtype=np.float64))
    return (
        np.stack(corners, axis=0)
        if corners
        else np.empty((0, 8, 3), dtype=np.float64)
    )


def residual_gt_world_boxes(
    gt_aligned: np.ndarray,
    baseline_world_corners: np.ndarray,
    axis_alignment: np.ndarray,
    *,
    covered_iou: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return residual GT as world AABBs and its original GT indices."""

    if not 0.0 <= float(covered_iou) <= 1.0:
        raise ValueError("covered_iou must lie in [0,1]")
    gt_minmax = center_size_to_minmax(gt_aligned)
    aligned_prediction = corners_to_minmax(
        transform_points(baseline_world_corners, axis_alignment)
    )
    overlaps = pairwise_aabb_iou(aligned_prediction, gt_minmax)
    maximum = (
        overlaps.max(axis=0)
        if len(aligned_prediction)
        else np.zeros(len(gt_minmax), dtype=np.float64)
    )
    # ScanNet matching uses strict IoU > threshold, so equality is still
    # uncovered and must remain a residual target.
    residual_indices = np.flatnonzero(maximum <= float(covered_iou)).astype(
        np.int64
    )
    residual_aligned_corners = center_size_to_corners(gt_aligned[residual_indices])
    residual_world_corners = transform_points(
        residual_aligned_corners, np.linalg.inv(axis_alignment)
    )
    residual_world = minmax_to_center_size(
        corners_to_minmax(residual_world_corners)
    )
    return residual_world, residual_indices


def assign_residual_targets(
    voxel_centers: np.ndarray,
    gt_boxes: np.ndarray,
    *,
    topk: int = 9,
) -> ResidualTargets:
    """Assign at most ``topk`` closest inside voxels per residual GT.

    When one voxel is selected by multiple boxes it is assigned to the box
    with the smallest size-normalised centre distance.  The result exactly
    matches the runtime encoding contract.
    """

    centers = np.asarray(voxel_centers, dtype=np.float64)
    boxes = np.asarray(gt_boxes, dtype=np.float64)
    if centers.ndim != 2 or centers.shape[1] != 3:
        raise ValueError("voxel_centers must have shape [V,3]")
    if not np.isfinite(centers).all():
        raise ValueError("voxel_centers must be finite")
    center_size_to_corners(boxes)
    if isinstance(topk, bool) or int(topk) <= 0:
        raise ValueError("topk must be a positive integer")
    topk = int(topk)

    try:
        from boxfusion.residual_proposal import (
            assign_residual_targets as runtime_assign_residual_targets,
        )
    except (ImportError, AttributeError):
        runtime_assign_residual_targets = None
    if runtime_assign_residual_targets is not None:
        objectness, regression, assigned = runtime_assign_residual_targets(
            centers,
            boxes,
            topk=topk,
        )
        return ResidualTargets(
            objectness=np.asarray(objectness, dtype=np.float32),
            regression=np.asarray(regression, dtype=np.float32),
            assigned_gt=np.asarray(assigned, dtype=np.int64),
        )

    count = len(centers)
    assigned = np.full(count, -1, dtype=np.int64)
    assigned_distance = np.full(count, np.inf, dtype=np.float64)
    for gt_index, box in enumerate(boxes):
        half_size = 0.5 * box[3:]
        normalised = np.abs((centers - box[:3]) / np.maximum(half_size, 1e-6))
        inside = np.all(normalised <= 1.0 + 1e-7, axis=1)
        candidates = np.flatnonzero(inside)
        pool = candidates if len(candidates) else np.arange(len(centers))
        distance = np.linalg.norm(centers[pool] - box[:3], axis=1)
        order = np.lexsort((pool, distance))[:topk]
        for voxel_index, value in zip(pool[order], distance[order]):
            if float(value) < float(assigned_distance[voxel_index]):
                assigned[int(voxel_index)] = gt_index
                assigned_distance[int(voxel_index)] = float(value)

    objectness = (assigned >= 0).astype(np.float32)
    regression = np.zeros((count, 6), dtype=np.float32)
    positive = np.flatnonzero(assigned >= 0)
    if len(positive):
        targets = boxes[assigned[positive]]
        regression[positive, :3] = (
            targets[:, :3] - centers[positive]
        ).astype(np.float32)
        regression[positive, 3:] = np.log(
            np.maximum(targets[:, 3:], 1e-6)
        ).astype(np.float32)
    return ResidualTargets(
        objectness=objectness,
        regression=regression,
        assigned_gt=assigned,
    )


def _deterministic_subsample(
    scene_id: str,
    targets: ResidualTargets,
    *,
    maximum_voxels: int,
    negative_ratio: float,
    seed: int,
) -> np.ndarray:
    positive = np.flatnonzero(targets.objectness > 0.5)
    negative = np.flatnonzero(targets.objectness <= 0.5)
    if negative_ratio < 0.0 or not math.isfinite(negative_ratio):
        raise ValueError("negative_ratio must be finite and non-negative")
    desired_negative = min(
        len(negative),
        int(round(max(len(positive), 1) * negative_ratio)),
    )
    digest = hashlib.sha256(f"{seed}:{scene_id}".encode("utf-8")).digest()
    scene_seed = int.from_bytes(digest[:8], "little") % (2**32)
    rng = np.random.default_rng(scene_seed)
    chosen_negative = (
        np.sort(rng.choice(negative, size=desired_negative, replace=False))
        if desired_negative < len(negative)
        else negative
    )
    selected = np.sort(np.concatenate((positive, chosen_negative))).astype(np.int64)
    if maximum_voxels > 0 and len(selected) > maximum_voxels:
        if len(positive) >= maximum_voxels:
            selected = np.sort(
                rng.choice(positive, size=maximum_voxels, replace=False)
            ).astype(np.int64)
        else:
            remaining = maximum_voxels - len(positive)
            sampled_negative = (
                rng.choice(chosen_negative, size=remaining, replace=False)
                if remaining < len(chosen_negative)
                else chosen_negative
            )
            selected = np.sort(
                np.concatenate((positive, sampled_negative))
            ).astype(np.int64)
    return selected


def build_training_data(
    *,
    scenes: Sequence[str],
    diagnostics_root: str | os.PathLike[str],
    prediction_root: str | os.PathLike[str],
    gt_root: str | os.PathLike[str],
    scans_root: str | os.PathLike[str],
    covered_iou: float = 0.15,
    assignment_topk: int = 9,
    maximum_voxels_per_scene: int = 60000,
    negative_ratio: float = 8.0,
    seed: int = 1337,
) -> ResidualTrainingData:
    diagnostics = Path(diagnostics_root)
    predictions = Path(prediction_root)
    gt_directory = Path(gt_root)
    scans = Path(scans_root)
    for role, root in (
        ("diagnostics", diagnostics),
        ("prediction", predictions),
        ("ground-truth", gt_directory),
        ("scans", scans),
    ):
        if not root.is_dir():
            raise FileNotFoundError(f"{role} root not found: {root}")
    if isinstance(maximum_voxels_per_scene, bool) or int(
        maximum_voxels_per_scene
    ) < 0:
        raise ValueError("maximum_voxels_per_scene must be a non-negative integer")

    feature_parts: list[np.ndarray] = []
    objectness_parts: list[np.ndarray] = []
    regression_parts: list[np.ndarray] = []
    scene_id_parts: list[np.ndarray] = []
    summaries: list[Mapping[str, Any]] = []
    canonical_names: tuple[str, ...] | None = None
    for scene_id in scenes:
        inputs = load_scene_voxels(
            diagnostics / f"{scene_id}_tracks.npz",
            expected_scene_id=scene_id,
        )
        if canonical_names is None:
            canonical_names = inputs.feature_names
        elif inputs.feature_names != canonical_names:
            raise ValueError(
                f"{scene_id}: P1 feature schema differs across training scenes"
            )
        gt = load_gt_boxes(gt_directory / f"{scene_id}_bbox.npy")
        baseline = load_prediction_corners(
            predictions / f"{scene_id}_boxes.pkl"
        )
        alignment = load_axis_alignment(scans, scene_id)
        residual_boxes, residual_indices = residual_gt_world_boxes(
            gt,
            baseline,
            alignment,
            covered_iou=covered_iou,
        )
        targets = assign_residual_targets(
            inputs.centers_world,
            residual_boxes,
            topk=assignment_topk,
        )
        selected = _deterministic_subsample(
            scene_id,
            targets,
            maximum_voxels=maximum_voxels_per_scene,
            negative_ratio=negative_ratio,
            seed=seed,
        )
        feature_parts.append(inputs.features[selected])
        objectness_parts.append(targets.objectness[selected])
        regression_parts.append(targets.regression[selected])
        scene_id_parts.append(
            np.full(len(selected), scene_id, dtype=f"<U{max(12, len(scene_id))}")
        )
        summaries.append(
            {
                "scene_id": scene_id,
                "diagnostic_sha256": _file_sha256(
                    diagnostics / f"{scene_id}_tracks.npz"
                ),
                "prediction_sha256": _file_sha256(
                    predictions / f"{scene_id}_boxes.pkl"
                ),
                "ground_truth_sha256": _file_sha256(
                    gt_directory / f"{scene_id}_bbox.npy"
                ),
                "snapshots": int(len(inputs.offsets) - 1),
                "voxel_count": int(len(inputs.features)),
                "selected_voxels": int(len(selected)),
                "positive_voxels": int(np.sum(targets.objectness[selected] > 0.5)),
                "ground_truth_count": int(len(gt)),
                "residual_ground_truth_count": int(len(residual_boxes)),
                "residual_ground_truth_indices": residual_indices.tolist(),
            }
        )
    if canonical_names is None or not feature_parts:
        raise ValueError("no P1 training scenes were loaded")
    features = np.concatenate(feature_parts, axis=0)
    objectness = np.concatenate(objectness_parts, axis=0)
    regression = np.concatenate(regression_parts, axis=0)
    scene_ids = np.concatenate(scene_id_parts, axis=0)
    if not len(features):
        raise ValueError("P1 training data contains zero selected voxels")
    if not np.any(objectness > 0.5):
        raise ValueError(
            "P1 training data contains no positive residual voxels; "
            "inspect covered_iou and observer coverage"
        )
    return ResidualTrainingData(
        features=np.ascontiguousarray(features, dtype=np.float32),
        objectness=np.ascontiguousarray(objectness, dtype=np.float32),
        regression=np.ascontiguousarray(regression, dtype=np.float32),
        scene_ids=np.asarray(scene_ids, dtype=np.str_),
        feature_names=canonical_names,
        scene_summaries=tuple(summaries),
    )


def deterministic_scene_split(
    scene_ids: np.ndarray, *, validation_fraction: float, seed: int
) -> tuple[np.ndarray, np.ndarray, tuple[str, ...], tuple[str, ...]]:
    values = np.asarray(scene_ids)
    if values.ndim != 1 or values.dtype.hasobject:
        raise ValueError("scene_ids must be a non-object [N] array")
    unique = np.unique(values)
    if len(unique) < 2:
        raise ValueError("P1 training needs at least two distinct scenes")
    if not 0.0 < float(validation_fraction) < 1.0:
        raise ValueError("validation_fraction must lie strictly in (0,1)")
    rng = np.random.default_rng(int(seed))
    shuffled = unique[rng.permutation(len(unique))]
    validation_count = min(
        max(1, int(round(len(unique) * float(validation_fraction)))),
        len(unique) - 1,
    )
    validation_scenes = tuple(sorted(str(x) for x in shuffled[:validation_count]))
    training_scenes = tuple(sorted(str(x) for x in shuffled[validation_count:]))
    train_mask = np.isin(values, np.asarray(training_scenes))
    validation_mask = np.isin(values, np.asarray(validation_scenes))
    train_indices = np.flatnonzero(train_mask).astype(np.int64)
    validation_indices = np.flatnonzero(validation_mask).astype(np.int64)
    if len(train_indices) == 0 or len(validation_indices) == 0:
        raise RuntimeError("scene-level split produced an empty partition")
    return train_indices, validation_indices, training_scenes, validation_scenes


scene_disjoint_split = deterministic_scene_split


def _loss(
    logits: torch.Tensor,
    predicted_regression: torch.Tensor,
    objectness: torch.Tensor,
    target_regression: torch.Tensor,
    *,
    positive_weight: torch.Tensor,
    regression_weight: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if logits.ndim == 2 and logits.shape[1] == 1:
        logits = logits[:, 0]
    classification = F.binary_cross_entropy_with_logits(
        logits, objectness, pos_weight=positive_weight
    )
    positive = objectness > 0.5
    if bool(torch.any(positive)):
        regression = F.smooth_l1_loss(
            predicted_regression[positive],
            target_regression[positive],
            beta=0.10,
        )
    else:
        regression = predicted_regression.sum() * 0.0
    total = classification + float(regression_weight) * regression
    return total, classification, regression


def _metrics(
    logits: torch.Tensor,
    predicted_regression: torch.Tensor,
    objectness: torch.Tensor,
    target_regression: torch.Tensor,
) -> dict[str, float]:
    if logits.ndim == 2 and logits.shape[1] == 1:
        logits = logits[:, 0]
    probabilities = torch.sigmoid(logits)
    prediction = probabilities >= 0.5
    target = objectness > 0.5
    tp = int(torch.sum(prediction & target).item())
    fp = int(torch.sum(prediction & ~target).item())
    fn = int(torch.sum(~prediction & target).item())
    if bool(torch.any(target)):
        regression_mae = float(
            torch.mean(
                torch.abs(
                    predicted_regression[target] - target_regression[target]
                )
            ).item()
        )
    else:
        regression_mae = 0.0
    return {
        "precision_at_0p5": float(tp / max(tp + fp, 1)),
        "recall_at_0p5": float(tp / max(tp + fn, 1)),
        "regression_mae_positive": regression_mae,
        "positive_count": float(torch.sum(target).item()),
        "sample_count": float(len(target)),
    }


def train_residual_head(
    data: ResidualTrainingData,
    *,
    hidden_dim: int = 64,
    validation_fraction: float = 0.20,
    epochs: int = 120,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    regression_weight: float = 1.0,
    batch_size: int = 8192,
    seed: int = 1337,
    device: str = "cpu",
) -> tuple[nn.Module, dict[str, Any]]:
    """Train deterministically and return the best validation checkpoint."""

    if int(hidden_dim) <= 0 or int(epochs) <= 0 or int(batch_size) <= 0:
        raise ValueError("hidden_dim, epochs, and batch_size must be positive")
    if float(learning_rate) <= 0.0 or float(weight_decay) < 0.0:
        raise ValueError("learning_rate must be >0 and weight_decay >=0")
    if float(regression_weight) < 0.0:
        raise ValueError("regression_weight must be non-negative")
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    torch_device = torch.device(device)
    train_indices, validation_indices, training_scenes, validation_scenes = (
        deterministic_scene_split(
            data.scene_ids,
            validation_fraction=validation_fraction,
            seed=seed,
        )
    )
    model = _head_class()(
        input_dim=int(data.features.shape[1]),
        hidden_dim=int(hidden_dim),
    ).to(torch_device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(learning_rate),
        weight_decay=float(weight_decay),
    )
    train_positive = float(np.sum(data.objectness[train_indices] > 0.5))
    train_negative = float(len(train_indices) - train_positive)
    pos_weight_value = min(max(train_negative / max(train_positive, 1.0), 1.0), 50.0)
    positive_weight = torch.tensor(pos_weight_value, device=torch_device)

    features = torch.from_numpy(data.features)
    objectness = torch.from_numpy(data.objectness)
    regression = torch.from_numpy(data.regression)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    best_loss = math.inf
    best_epoch = -1
    best_state: dict[str, torch.Tensor] | None = None
    history: list[dict[str, float]] = []
    for epoch in range(int(epochs)):
        model.train()
        permutation = train_indices[
            torch.randperm(len(train_indices), generator=generator).numpy()
        ]
        total_loss = 0.0
        total_samples = 0
        for start in range(0, len(permutation), int(batch_size)):
            batch = permutation[start : start + int(batch_size)]
            batch_features = features[batch].to(torch_device)
            batch_objectness = objectness[batch].to(torch_device)
            batch_regression = regression[batch].to(torch_device)
            optimizer.zero_grad(set_to_none=True)
            logits, predicted = model(batch_features)
            loss, _, _ = _loss(
                logits,
                predicted,
                batch_objectness,
                batch_regression,
                positive_weight=positive_weight,
                regression_weight=regression_weight,
            )
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach().item()) * len(batch)
            total_samples += len(batch)

        model.eval()
        with torch.no_grad():
            validation_features = features[validation_indices].to(torch_device)
            validation_objectness = objectness[validation_indices].to(torch_device)
            validation_regression = regression[validation_indices].to(torch_device)
            validation_logits, validation_prediction = model(validation_features)
            validation_loss, validation_cls, validation_reg = _loss(
                validation_logits,
                validation_prediction,
                validation_objectness,
                validation_regression,
                positive_weight=positive_weight,
                regression_weight=regression_weight,
            )
        epoch_record = {
            "epoch": float(epoch),
            "train_loss": float(total_loss / max(total_samples, 1)),
            "validation_loss": float(validation_loss.item()),
            "validation_classification_loss": float(validation_cls.item()),
            "validation_regression_loss": float(validation_reg.item()),
        }
        history.append(epoch_record)
        if epoch_record["validation_loss"] < best_loss:
            best_loss = epoch_record["validation_loss"]
            best_epoch = epoch
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
    if best_state is None:
        raise RuntimeError("P1 training did not produce a checkpoint")
    model.load_state_dict(best_state, strict=True)
    model.eval()
    with torch.no_grad():
        validation_logits, validation_prediction = model(
            features[validation_indices].to(torch_device)
        )
        validation_metrics = _metrics(
            validation_logits,
            validation_prediction,
            objectness[validation_indices].to(torch_device),
            regression[validation_indices].to(torch_device),
        )
    summary: dict[str, Any] = {
        "best_epoch": int(best_epoch),
        "best_validation_loss": float(best_loss),
        "training_scenes": list(training_scenes),
        "validation_scenes": list(validation_scenes),
        "training_samples": int(len(train_indices)),
        "validation_samples": int(len(validation_indices)),
        "positive_weight": float(pos_weight_value),
        "validation_metrics": validation_metrics,
        "last_epoch": history[-1],
    }
    return model.cpu(), summary


train_head = train_residual_head


def save_checkpoint(
    output_path: str | os.PathLike[str],
    *,
    model: nn.Module,
    feature_names: Sequence[str],
    hidden_dim: int,
    training_config: Mapping[str, Any],
    metrics: Mapping[str, Any],
    provenance: Mapping[str, Any],
) -> Path:
    output = Path(output_path)
    if output.suffix.lower() not in {".pt", ".pth"}:
        raise ValueError("P1 checkpoint must end in .pt or .pth")
    model_config = {
        "input_dim": int(len(feature_names)),
        "hidden_dim": int(hidden_dim),
        "regression_dim": 6,
        "regression_encoding": "center_delta_m_log_size_m",
    }
    checkpoint = {
        "schema": CHECKPOINT_SCHEMA,
        "model_config": model_config,
        "config": {
            "model": dict(model_config),
            "training": dict(training_config),
        },
        "feature_names": list(str(name) for name in feature_names),
        "state_dict": {
            name: tensor.detach().cpu()
            for name, tensor in model.state_dict().items()
        },
        "training_config": dict(training_config),
        "metrics": dict(metrics),
        "provenance": dict(provenance),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    torch.save(checkpoint, temporary)
    os.replace(temporary, output)
    try:
        loaded = torch.load(output, map_location="cpu", weights_only=False)
    except TypeError:  # PyTorch < 2.0
        loaded = torch.load(output, map_location="cpu")
    if (
        not isinstance(loaded, dict)
        or loaded.get("schema") != CHECKPOINT_SCHEMA
        or tuple(loaded.get("feature_names", ())) != tuple(feature_names)
    ):
        raise RuntimeError("saved P1 checkpoint failed schema verification")
    return output


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diagnostics-root", required=True, type=Path)
    parser.add_argument("--prediction-root", required=True, type=Path)
    parser.add_argument("--gt-root", required=True, type=Path)
    parser.add_argument("--scans-root", required=True, type=Path)
    parser.add_argument("--train-scene-list", required=True, type=Path)
    parser.add_argument("--forbidden-scene-list", required=True, type=Path)
    parser.add_argument("--b6-checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--covered-iou", type=float, default=0.15)
    parser.add_argument("--assignment-topk", type=int, default=9)
    parser.add_argument("--max-voxels-per-scene", type=int, default=60000)
    parser.add_argument("--negative-ratio", type=float, default=8.0)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--validation-fraction", type=float, default=0.20)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--regression-weight", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--summary-json", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    train_scenes = read_scene_ids(args.train_scene_list, role="training")
    forbidden_scenes = read_scene_ids(
        args.forbidden_scene_list, role="forbidden validation"
    )
    train_scenes = validate_train_split(train_scenes, forbidden_scenes)
    data = build_training_data(
        scenes=train_scenes,
        diagnostics_root=args.diagnostics_root,
        prediction_root=args.prediction_root,
        gt_root=args.gt_root,
        scans_root=args.scans_root,
        covered_iou=args.covered_iou,
        assignment_topk=args.assignment_topk,
        maximum_voxels_per_scene=args.max_voxels_per_scene,
        negative_ratio=args.negative_ratio,
        seed=args.seed,
    )
    training_config = {
        "schema": TRAINING_SCHEMA,
        "covered_iou": float(args.covered_iou),
        "assignment_topk": int(args.assignment_topk),
        "maximum_voxels_per_scene": int(args.max_voxels_per_scene),
        "negative_ratio": float(args.negative_ratio),
        "hidden_dim": int(args.hidden_dim),
        "validation_fraction": float(args.validation_fraction),
        "epochs": int(args.epochs),
        "learning_rate": float(args.learning_rate),
        "weight_decay": float(args.weight_decay),
        "regression_weight": float(args.regression_weight),
        "batch_size": int(args.batch_size),
        "seed": int(args.seed),
        "device": str(args.device),
    }
    model, metrics = train_residual_head(
        data,
        hidden_dim=args.hidden_dim,
        validation_fraction=args.validation_fraction,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        regression_weight=args.regression_weight,
        batch_size=args.batch_size,
        seed=args.seed,
        device=args.device,
    )
    provenance = {
        "train_scene_list": str(args.train_scene_list.resolve()),
        "train_scene_list_sha256": _file_sha256(args.train_scene_list),
        "forbidden_scene_list": str(args.forbidden_scene_list.resolve()),
        "forbidden_scene_list_sha256": _file_sha256(
            args.forbidden_scene_list
        ),
        "train_scene_ids": list(train_scenes),
        "forbidden_scene_count": int(len(forbidden_scenes)),
        "forbidden_overlap": [],
        "b6_checkpoint": str(args.b6_checkpoint.resolve()),
        "b6_checkpoint_sha256": _file_sha256(args.b6_checkpoint),
        "diagnostics_root": str(args.diagnostics_root.resolve()),
        "prediction_root": str(args.prediction_root.resolve()),
        "gt_root": str(args.gt_root.resolve()),
        "scans_root": str(args.scans_root.resolve()),
        "scene_summaries": list(data.scene_summaries),
    }
    output = save_checkpoint(
        args.output,
        model=model,
        feature_names=data.feature_names,
        hidden_dim=args.hidden_dim,
        training_config=training_config,
        metrics=metrics,
        provenance=provenance,
    )
    summary = {
        "schema": CHECKPOINT_SCHEMA,
        "output": str(output.resolve()),
        "scene_count": int(len(train_scenes)),
        "sample_count": int(len(data.features)),
        "positive_count": int(np.sum(data.objectness > 0.5)),
        "feature_names": list(data.feature_names),
        "metrics": metrics,
    }
    rendered = json.dumps(summary, indent=2, sort_keys=True)
    if args.summary_json is not None:
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.summary_json.with_name(args.summary_json.name + ".tmp")
        temporary.write_text(rendered + "\n", encoding="utf-8")
        os.replace(temporary, args.summary_json)
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
