"""Class-agnostic residual 3D proposals for the BoxFusion P1 ablation.

This module is a clean-room, dependency-light implementation of the proposal
contract used by the P1 experiment.  It borrows the following *design* from
TR3D's Apache-2.0 licensed head:

* predict one objectness value and one box residual at each sparse voxel;
* encode a box as centre offsets plus logarithmic dimensions;
* keep a deterministic pre-NMS Top-K and apply class-agnostic 3D NMS.

It deliberately does not copy or import TR3D, MinkowskiEngine, MMDetection3D,
SGCDet, or SPGroup3D.  The current ScanNet experiment is axis aligned, so the
regression output has six values rather than TR3D's rotation-aware eight.
SGCDet-style occupancy selection and SPGroup3D-style grouping belong to the
later P2 and P3 ablations and are not silently folded into P1.

Ground truth is never accepted by the online observer.  Training targets are
constructed only by the offline :func:`assign_residual_targets` helper.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np

try:
    import torch
    from torch import nn
except Exception:  # pragma: no cover - exercised by dependency preflight
    torch = None
    nn = None


P1_DIAGNOSTIC_SCHEMA = "boxfusion.p1.residual_proposal_observer.v1"
P1_HEAD_SCHEMA = "boxfusion.p1_residual_head.v1"
P1_FEATURE_NAMES = (
    "log_point_count",
    "mean_red",
    "mean_green",
    "mean_blue",
    "camera_relative_x",
    "camera_relative_y",
    "camera_relative_z",
    "mean_range",
    "range_std",
    "occupancy_cube_r1",
    "occupancy_cube_r2",
    "occupancy_cube_r4",
    "vertical_neighbor_balance",
    "nearest_b6_distance",
)
P1_FEATURE_DIM = len(P1_FEATURE_NAMES)

_CORNERS_SIGNS = np.asarray(
    (
        (-1.0, -1.0, -1.0),
        (1.0, -1.0, -1.0),
        (1.0, 1.0, -1.0),
        (-1.0, 1.0, -1.0),
        (-1.0, -1.0, 1.0),
        (1.0, -1.0, 1.0),
        (1.0, 1.0, 1.0),
        (-1.0, 1.0, 1.0),
    ),
    dtype=np.float32,
)
def _finite_float(
    value: Any,
    name: str,
    *,
    lower: Optional[float] = None,
    upper: Optional[float] = None,
    strict_lower: bool = False,
) -> float:
    if isinstance(value, (bool, np.bool_)) or not np.isscalar(value):
        raise ValueError(f"{name} must be a finite scalar")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    if lower is not None:
        invalid = result <= lower if strict_lower else result < lower
        if invalid:
            relation = "greater than" if strict_lower else "at least"
            raise ValueError(f"{name} must be {relation} {lower}")
    if upper is not None and result > upper:
        raise ValueError(f"{name} must be at most {upper}")
    return result


def _positive_int(value: Any, name: str, minimum: int = 1) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise ValueError(f"{name} must be an integer")
    result = int(value)
    if result < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return result


@dataclass(frozen=True)
class ResidualProposalConfig:
    """Resolved P1 runtime and architecture configuration."""

    enabled: bool = False
    observer_only: bool = True
    mutate: bool = False
    collect_diagnostics: bool = False
    mode: str = "infer"
    checkpoint: Optional[str] = None
    device: Optional[str] = None
    depth_stride: int = 4
    min_depth: float = 0.15
    max_depth: float = 8.0
    voxel_size: float = 0.08
    explained_margin: float = 0.05
    min_voxel_points: int = 1
    max_voxels: int = 12000
    max_history_steps: int = 64
    collect_voxel_inputs: bool = False
    hidden_dim: int = 64
    assignment_topk: int = 6
    score_threshold: float = 0.05
    pre_nms_topk: int = 512
    max_candidates_per_step: int = 64
    max_scene_candidates: int = 256
    nms_iou: float = 0.25
    scene_nms_iou: float = 0.25
    min_box_extent: float = 0.08
    max_box_extent: float = 4.0
    max_center_offset: float = 1.0
    input_feature_names: Tuple[str, ...] = P1_FEATURE_NAMES

    def validated(self) -> "ResidualProposalConfig":
        if not isinstance(self.enabled, (bool, np.bool_)):
            raise ValueError("residual_proposal.enabled must be Boolean")
        if not isinstance(self.observer_only, (bool, np.bool_)):
            raise ValueError(
                "residual_proposal.observer_only must be Boolean"
            )
        if not bool(self.observer_only):
            raise ValueError("P1 must remain observer_only")
        if not isinstance(self.mutate, (bool, np.bool_)):
            raise ValueError("residual_proposal.mutate must be Boolean")
        if bool(self.mutate):
            raise ValueError("P1 is observer-only; mutate must be false")
        if not isinstance(self.collect_diagnostics, (bool, np.bool_)):
            raise ValueError(
                "residual_proposal.collect_diagnostics must be Boolean"
            )
        if not isinstance(self.collect_voxel_inputs, (bool, np.bool_)):
            raise ValueError(
                "residual_proposal.collect_voxel_inputs must be Boolean"
            )
        mode = str(self.mode).strip().lower()
        if mode not in {"infer", "collect"}:
            raise ValueError("residual_proposal.mode must be infer or collect")
        checkpoint = self.checkpoint
        if checkpoint is not None:
            if not isinstance(checkpoint, (str, Path)):
                raise TypeError(
                    "residual_proposal.checkpoint must be a path or null"
                )
            checkpoint = str(checkpoint).strip()
            if not checkpoint:
                raise ValueError(
                    "residual_proposal.checkpoint cannot be empty"
                )
        device = self.device
        if device is not None:
            if not isinstance(device, str) or not device.strip():
                raise ValueError(
                    "residual_proposal.device must be a non-empty string"
                )
            device = device.strip()
        names = tuple(str(name) for name in self.input_feature_names)
        if names != P1_FEATURE_NAMES:
            raise ValueError(
                "residual_proposal input feature schema does not match P1"
            )
        return ResidualProposalConfig(
            enabled=bool(self.enabled),
            observer_only=True,
            mutate=False,
            collect_diagnostics=bool(self.collect_diagnostics),
            mode=mode,
            checkpoint=checkpoint,
            device=device,
            depth_stride=_positive_int(
                self.depth_stride, "residual_proposal.depth_stride"
            ),
            min_depth=_finite_float(
                self.min_depth,
                "residual_proposal.min_depth",
                lower=0.0,
                strict_lower=True,
            ),
            max_depth=_finite_float(
                self.max_depth,
                "residual_proposal.max_depth",
                lower=0.0,
                strict_lower=True,
            ),
            voxel_size=_finite_float(
                self.voxel_size,
                "residual_proposal.voxel_size",
                lower=0.0,
                strict_lower=True,
            ),
            explained_margin=_finite_float(
                self.explained_margin,
                "residual_proposal.explained_margin",
                lower=0.0,
            ),
            min_voxel_points=_positive_int(
                self.min_voxel_points,
                "residual_proposal.min_voxel_points",
            ),
            max_voxels=_positive_int(
                self.max_voxels, "residual_proposal.max_voxels"
            ),
            max_history_steps=_positive_int(
                self.max_history_steps,
                "residual_proposal.max_history_steps",
            ),
            collect_voxel_inputs=bool(self.collect_voxel_inputs),
            hidden_dim=_positive_int(
                self.hidden_dim, "residual_proposal.hidden_dim"
            ),
            assignment_topk=_positive_int(
                self.assignment_topk,
                "residual_proposal.assignment_topk",
            ),
            score_threshold=_finite_float(
                self.score_threshold,
                "residual_proposal.score_threshold",
                lower=0.0,
                upper=1.0,
            ),
            pre_nms_topk=_positive_int(
                self.pre_nms_topk,
                "residual_proposal.pre_nms_topk",
            ),
            max_candidates_per_step=_positive_int(
                self.max_candidates_per_step,
                "residual_proposal.max_candidates_per_step",
            ),
            max_scene_candidates=_positive_int(
                self.max_scene_candidates,
                "residual_proposal.max_scene_candidates",
            ),
            nms_iou=_finite_float(
                self.nms_iou,
                "residual_proposal.nms_iou",
                lower=0.0,
                upper=1.0,
            ),
            scene_nms_iou=_finite_float(
                self.scene_nms_iou,
                "residual_proposal.scene_nms_iou",
                lower=0.0,
                upper=1.0,
            ),
            min_box_extent=_finite_float(
                self.min_box_extent,
                "residual_proposal.min_box_extent",
                lower=0.0,
                strict_lower=True,
            ),
            max_box_extent=_finite_float(
                self.max_box_extent,
                "residual_proposal.max_box_extent",
                lower=0.0,
                strict_lower=True,
            ),
            max_center_offset=_finite_float(
                self.max_center_offset,
                "residual_proposal.max_center_offset",
                lower=0.0,
                strict_lower=True,
            ),
            input_feature_names=names,
        )._validate_relations()

    def _validate_relations(self) -> "ResidualProposalConfig":
        if self.max_depth <= self.min_depth:
            raise ValueError(
                "residual_proposal.max_depth must exceed min_depth"
            )
        if self.max_box_extent < self.min_box_extent:
            raise ValueError(
                "residual_proposal.max_box_extent must exceed "
                "min_box_extent"
            )
        if self.max_candidates_per_step > self.pre_nms_topk:
            raise ValueError(
                "max_candidates_per_step cannot exceed pre_nms_topk"
            )
        if self.mode == "collect" and not self.collect_diagnostics:
            raise ValueError(
                "residual_proposal collect mode requires diagnostics"
            )
        if self.mode == "collect" and not self.collect_voxel_inputs:
            raise ValueError(
                "residual_proposal collect mode requires voxel inputs"
            )
        return self

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["input_feature_names"] = list(self.input_feature_names)
        return payload


def resolve_residual_proposal_config(
    config: Optional[Mapping[str, Any] | ResidualProposalConfig] = None,
) -> ResidualProposalConfig:
    """Resolve a strict P1 config without loading optional model state."""

    if config is None:
        return ResidualProposalConfig().validated()
    if isinstance(config, ResidualProposalConfig):
        return config.validated()
    if not isinstance(config, Mapping):
        raise TypeError("residual_proposal config must be a mapping")
    known = set(ResidualProposalConfig.__dataclass_fields__)
    unknown = sorted(set(config) - known)
    if unknown:
        raise ValueError(
            "Unknown residual_proposal key(s): " + ", ".join(unknown)
        )
    payload = dict(config)
    if "input_feature_names" in payload:
        payload["input_feature_names"] = tuple(payload["input_feature_names"])
    return ResidualProposalConfig(**payload).validated()


@dataclass(frozen=True)
class ResidualVoxelBatch:
    """One deterministic sparse voxel frame."""

    coordinates: np.ndarray
    centers: np.ndarray
    features: np.ndarray
    point_counts: np.ndarray
    input_point_count: int
    explained_point_count: int
    residual_point_count: int

    def __post_init__(self) -> None:
        coordinates = np.array(self.coordinates, dtype=np.int32, copy=True)
        centers = np.array(self.centers, dtype=np.float32, copy=True)
        features = np.array(self.features, dtype=np.float32, copy=True)
        counts = np.array(self.point_counts, dtype=np.int32, copy=True)
        size = len(coordinates)
        if coordinates.shape != (size, 3):
            raise ValueError("voxel coordinates must have shape [V,3]")
        if coordinates.dtype.kind not in {"i", "u"}:
            raise ValueError("voxel coordinates must be integer")
        if centers.shape != (size, 3):
            raise ValueError("voxel centers must have shape [V,3]")
        if features.shape != (size, P1_FEATURE_DIM):
            raise ValueError(
                f"voxel features must have shape [V,{P1_FEATURE_DIM}]"
            )
        if counts.shape != (size,) or counts.dtype.kind not in {"i", "u"}:
            raise ValueError("voxel point_counts must be integer [V]")
        if (
            not np.isfinite(centers).all()
            or not np.isfinite(features).all()
            or np.any(counts <= 0)
        ):
            raise ValueError("voxel batch contains invalid values")
        for array in (coordinates, centers, features, counts):
            array.setflags(write=False)
        object.__setattr__(self, "coordinates", coordinates)
        object.__setattr__(self, "centers", centers)
        object.__setattr__(self, "features", features)
        object.__setattr__(self, "point_counts", counts)

    @property
    def sparse_coordinates(self) -> np.ndarray:
        """Return TR3D-style ``[batch,x,y,z]`` integer coordinates."""

        batch = np.zeros((len(self.coordinates), 1), dtype=np.int32)
        return np.concatenate(
            (batch, np.asarray(self.coordinates, dtype=np.int32)), axis=1
        )

    @property
    def centers_world(self) -> np.ndarray:
        return self.centers

    @classmethod
    def empty(
        cls,
        *,
        input_point_count: int = 0,
        explained_point_count: int = 0,
        residual_point_count: int = 0,
    ) -> "ResidualVoxelBatch":
        return cls(
            coordinates=np.empty((0, 3), dtype=np.int32),
            centers=np.empty((0, 3), dtype=np.float32),
            features=np.empty((0, P1_FEATURE_DIM), dtype=np.float32),
            point_counts=np.empty((0,), dtype=np.int32),
            input_point_count=int(input_point_count),
            explained_point_count=int(explained_point_count),
            residual_point_count=int(residual_point_count),
        )


@dataclass(frozen=True)
class ResidualProposal:
    """One diagnostics-only P1 proposal in the world frame."""

    candidate_id: str
    frame_index: int
    provider_step: int
    box: np.ndarray
    corners: np.ndarray
    objectness: float
    residual_point_count: int
    nearest_b6_stable_id: int = -1
    nearest_b6_iou: float = 0.0
    source: str = "p1_tr3d_style_residual"

    def __post_init__(self) -> None:
        box = np.asarray(self.box)
        corners = np.asarray(self.corners)
        if box.shape != (6,) or corners.shape != (8, 3):
            raise ValueError("proposal box/corners have invalid shape")
        if (
            not np.isfinite(box).all()
            or not np.isfinite(corners).all()
            or np.any(box[3:6] <= 0.0)
        ):
            raise ValueError("proposal geometry must be finite and positive")
        if not np.isfinite(self.objectness) or not 0.0 <= self.objectness <= 1.0:
            raise ValueError("proposal objectness must lie in [0,1]")
        if not self.candidate_id:
            raise ValueError("proposal candidate_id cannot be empty")

    @property
    def score(self) -> float:
        return float(self.objectness)


@dataclass(frozen=True)
class ResidualObservation:
    """P1 output for one scheduled online call."""

    frame_index: int
    provider_step: int
    voxel_batch: ResidualVoxelBatch
    proposals: Tuple[ResidualProposal, ...]
    voxelize_seconds: float = 0.0
    head_seconds: float = 0.0
    nms_seconds: float = 0.0
    error: str = ""

    @property
    def total_seconds(self) -> float:
        return float(
            self.voxelize_seconds + self.head_seconds + self.nms_seconds
        )

    @property
    def observer_only(self) -> bool:
        return True

    @property
    def mutation_enabled(self) -> bool:
        return False

    @property
    def applied_count(self) -> int:
        return 0


def center_size_to_corners(boxes: Any) -> np.ndarray:
    values = np.asarray(boxes, dtype=np.float32)
    if values.size == 0:
        return np.empty((0, 8, 3), dtype=np.float32)
    if values.ndim == 1:
        values = values[None, :]
    if values.ndim != 2 or values.shape[1] != 6:
        raise ValueError("boxes must have shape [N,6]")
    if not np.isfinite(values).all() or np.any(values[:, 3:] <= 0.0):
        raise ValueError("boxes must be finite with positive dimensions")
    return (
        values[:, None, :3]
        + _CORNERS_SIGNS[None] * (0.5 * values[:, None, 3:6])
    ).astype(np.float32)


def corners_to_center_size(corners: Any) -> np.ndarray:
    values = np.asarray(corners, dtype=np.float32)
    if values.size == 0:
        return np.empty((0, 6), dtype=np.float32)
    if values.ndim != 3 or values.shape[1:] != (8, 3):
        raise ValueError("corners must have shape [N,8,3]")
    if not np.isfinite(values).all():
        raise ValueError("corners must be finite")
    lower = values.min(axis=1)
    upper = values.max(axis=1)
    dimensions = upper - lower
    if np.any(dimensions <= 0.0):
        raise ValueError("corners must enclose positive-volume boxes")
    return np.concatenate(((lower + upper) * 0.5, dimensions), axis=1)


def center_size_to_minmax(boxes: Any) -> np.ndarray:
    values = np.asarray(boxes, dtype=np.float64)
    if values.size == 0:
        return np.empty((0, 6), dtype=np.float64)
    if values.ndim == 1:
        values = values[None, :]
    if values.ndim != 2 or values.shape[1] != 6:
        raise ValueError("center-size boxes must have shape [N,6]")
    if not np.isfinite(values).all() or np.any(values[:, 3:] <= 0.0):
        raise ValueError("center-size boxes are invalid")
    half = 0.5 * values[:, 3:]
    return np.concatenate((values[:, :3] - half, values[:, :3] + half), axis=1)


def pairwise_aabb_iou(boxes_a: Any, boxes_b: Any) -> np.ndarray:
    """Pairwise IoU for centre-size, axis-aligned boxes."""

    first = center_size_to_minmax(boxes_a)
    second = center_size_to_minmax(boxes_b)
    if not len(first) or not len(second):
        return np.zeros((len(first), len(second)), dtype=np.float64)
    lower = np.maximum(first[:, None, :3], second[None, :, :3])
    upper = np.minimum(first[:, None, 3:], second[None, :, 3:])
    intersection = np.prod(np.maximum(upper - lower, 0.0), axis=2)
    volume_a = np.prod(first[:, 3:] - first[:, :3], axis=1)
    volume_b = np.prod(second[:, 3:] - second[:, :3], axis=1)
    union = volume_a[:, None] + volume_b[None] - intersection
    return np.divide(
        intersection,
        union,
        out=np.zeros_like(intersection),
        where=union > 0.0,
    )


@dataclass(frozen=True)
class ScoreOrderedMatch:
    true_positive: np.ndarray
    prediction_to_gt: np.ndarray
    matched_iou: np.ndarray

    def __iter__(self):
        # Backward-friendly unpacking for callers that only need the TP mask
        # and target assignment.
        yield self.true_positive
        yield self.prediction_to_gt

    @property
    def true_positive_count(self) -> int:
        return int(np.asarray(self.true_positive, dtype=bool).sum())


def score_ordered_match(
    proposal_boxes_or_iou: Any,
    proposal_scores: Any,
    target_boxes_or_threshold: Any,
    threshold: Optional[float] = None,
    *,
    proposal_ids: Optional[Sequence[str]] = None,
    eligible_targets: Optional[np.ndarray] = None,
) -> ScoreOrderedMatch:
    """Greedy one-to-one matching with stable score/ID ordering.

    A match follows the ScanNet AP convention used by this project:
    ``IoU > threshold`` (strictly greater, not greater-or-equal).
    """

    scores = np.asarray(proposal_scores, dtype=np.float64).reshape(-1)
    if threshold is None and np.isscalar(target_boxes_or_threshold):
        ious = np.asarray(proposal_boxes_or_iou, dtype=np.float64)
        if ious.ndim != 2 or len(scores) != ious.shape[0]:
            raise ValueError("IoU matrix and proposal scores must align")
        threshold_value = target_boxes_or_threshold
        proposal_count, target_count = ious.shape
    else:
        boxes = np.asarray(proposal_boxes_or_iou, dtype=np.float64)
        targets = np.asarray(target_boxes_or_threshold, dtype=np.float64)
        if boxes.size == 0:
            boxes = np.empty((0, 6), dtype=np.float64)
        if targets.size == 0:
            targets = np.empty((0, 6), dtype=np.float64)
        if boxes.shape != (len(boxes), 6) or len(scores) != len(boxes):
            raise ValueError("proposal boxes and scores must align")
        if targets.shape != (len(targets), 6):
            raise ValueError("target boxes must have shape [M,6]")
        ious = pairwise_aabb_iou(boxes, targets)
        threshold_value = threshold
        proposal_count, target_count = len(boxes), len(targets)
    threshold = _finite_float(
        threshold_value, "threshold", lower=0.0, upper=1.0
    )
    ids = (
        tuple(f"{index:012d}" for index in range(proposal_count))
        if proposal_ids is None
        else tuple(str(value) for value in proposal_ids)
    )
    if len(ids) != proposal_count or len(set(ids)) != len(ids):
        raise ValueError("proposal_ids must align and be unique")
    eligible = (
        np.ones(target_count, dtype=bool)
        if eligible_targets is None
        else np.asarray(eligible_targets, dtype=bool).reshape(-1)
    )
    if len(eligible) != target_count:
        raise ValueError("eligible_targets must align with targets")
    order = np.asarray(
        sorted(
            range(proposal_count),
            key=lambda index: (-scores[index], ids[index]),
        ),
        dtype=np.int64,
    )
    matched_target = np.full(proposal_count, -1, dtype=np.int64)
    matched_iou = np.zeros(proposal_count, dtype=np.float64)
    true_positive = np.zeros(proposal_count, dtype=bool)
    used: set[int] = set()
    for proposal_index in order.tolist():
        candidates = [
            target_index
            for target_index in range(target_count)
            if eligible[target_index]
            and target_index not in used
            and ious[proposal_index, target_index] > threshold
        ]
        if not candidates:
            continue
        target_index = min(
            candidates,
            key=lambda index: (-ious[proposal_index, index], index),
        )
        used.add(target_index)
        matched_target[proposal_index] = target_index
        matched_iou[proposal_index] = ious[proposal_index, target_index]
        true_positive[proposal_index] = True
    return ScoreOrderedMatch(
        true_positive=true_positive,
        prediction_to_gt=matched_target,
        matched_iou=matched_iou,
    )


def stable_nms_aabb(
    boxes: Any,
    scores: Any,
    threshold: float,
    *,
    tie_breakers: Optional[Sequence[str]] = None,
    max_output: Optional[int] = None,
) -> np.ndarray:
    boxes_array = np.asarray(boxes, dtype=np.float64)
    scores_array = np.asarray(scores, dtype=np.float64).reshape(-1)
    if boxes_array.size == 0:
        return np.empty((0,), dtype=np.int64)
    if (
        boxes_array.ndim != 2
        or boxes_array.shape[1] != 6
        or len(boxes_array) != len(scores_array)
    ):
        raise ValueError("NMS boxes and scores must align")
    ids = (
        tuple(f"{index:012d}" for index in range(len(boxes_array)))
        if tie_breakers is None
        else tuple(str(value) for value in tie_breakers)
    )
    if len(ids) != len(boxes_array):
        raise ValueError("NMS tie_breakers must align")
    threshold = _finite_float(
        threshold, "nms threshold", lower=0.0, upper=1.0
    )
    if max_output is not None:
        max_output = _positive_int(max_output, "max_output")
    order = sorted(
        range(len(boxes_array)),
        key=lambda index: (-scores_array[index], ids[index]),
    )
    ious = pairwise_aabb_iou(boxes_array, boxes_array)
    kept: list[int] = []
    for index in order:
        if any(ious[index, old] > threshold for old in kept):
            continue
        kept.append(index)
        if max_output is not None and len(kept) >= max_output:
            break
    return np.asarray(kept, dtype=np.int64)


def _oriented_box_frame(corners: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = np.asarray(corners, dtype=np.float64)
    if values.shape != (8, 3) or not np.isfinite(values).all():
        raise ValueError("box corners must have finite shape [8,3]")
    center = values.mean(axis=0)
    # BoxFusion and ``aabb_corners`` use edges (0->1, 0->3, 0->4).
    # This constant-time path covers online inference; the order-independent
    # search below is retained for cache adapters and tests.
    fast_edges = np.stack(
        (values[1] - values[0], values[3] - values[0], values[4] - values[0]),
        axis=1,
    )
    fast_lengths = np.linalg.norm(fast_edges, axis=0)
    if np.all(fast_lengths > 1e-8):
        fast_basis = fast_edges / fast_lengths[None]
        if np.allclose(fast_basis.T @ fast_basis, np.eye(3), atol=2e-3):
            if np.linalg.det(fast_basis) < 0.0:
                fast_basis[:, 2] *= -1.0
            fast_local = (values - center[None]) @ fast_basis
            fast_dimensions = (
                fast_local.max(axis=0) - fast_local.min(axis=0)
            )
            if np.all(fast_dimensions > 1e-8):
                return center, fast_dimensions, fast_basis
    scale = max(float(np.ptp(values, axis=0).max()), 1e-8)
    best: Optional[tuple[float, np.ndarray]] = None
    for origin_index in range(8):
        origin = values[origin_index]
        vectors = values - origin[None]
        candidate_indices = [
            index
            for index in range(8)
            if index != origin_index
            and np.linalg.norm(vectors[index]) > 1e-8
        ]
        for indices in itertools.combinations(candidate_indices, 3):
            edges = np.stack([vectors[index] for index in indices], axis=1)
            lengths = np.linalg.norm(edges, axis=0)
            unit = edges / lengths[None]
            orthogonality = float(
                np.max(np.abs(unit.T @ unit - np.eye(3)))
            )
            if orthogonality > 2e-3:
                continue
            reconstructed = np.asarray(
                [
                    origin
                    + bit0 * edges[:, 0]
                    + bit1 * edges[:, 1]
                    + bit2 * edges[:, 2]
                    for bit0 in (0.0, 1.0)
                    for bit1 in (0.0, 1.0)
                    for bit2 in (0.0, 1.0)
                ]
            )
            nearest = np.linalg.norm(
                reconstructed[:, None] - values[None], axis=2
            ).min(axis=1)
            reconstruction_error = float(nearest.max() / scale)
            error = orthogonality + reconstruction_error
            if best is None or error < best[0]:
                best = (error, unit)
    if best is None or best[0] > 5e-3:
        raise ValueError("box corners do not form a rectangular OBB")
    basis = best[1]
    if np.linalg.det(basis) < 0.0:
        basis[:, 2] *= -1.0
    local = (values - center[None]) @ basis
    dimensions = local.max(axis=0) - local.min(axis=0)
    if np.any(dimensions <= 1e-8):
        raise ValueError("box frame has non-positive dimensions")
    return center, dimensions, basis


def points_explained_by_boxes(
    points_world: Any,
    global_corners: Any,
    *,
    margin: float,
    chunk_size: int = 65536,
) -> np.ndarray:
    """Return whether each world point lies inside any expanded BoxFusion OBB."""

    points = np.asarray(points_world, dtype=np.float64)
    corners = np.asarray(global_corners, dtype=np.float64)
    if points.size == 0:
        return np.zeros((0,), dtype=bool)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points_world must have shape [N,3]")
    if corners.size == 0:
        return np.zeros((len(points),), dtype=bool)
    if corners.ndim != 3 or corners.shape[1:] != (8, 3):
        raise ValueError("global_corners must have shape [B,8,3]")
    if not np.isfinite(points).all() or not np.isfinite(corners).all():
        raise ValueError("points and corners must be finite")
    margin = _finite_float(margin, "explained margin", lower=0.0)
    chunk_size = _positive_int(chunk_size, "chunk_size")
    result = np.zeros(len(points), dtype=bool)
    frames = [_oriented_box_frame(box) for box in corners]
    for start in range(0, len(points), chunk_size):
        stop = min(start + chunk_size, len(points))
        block = points[start:stop]
        explained = np.zeros(len(block), dtype=bool)
        for center, dimensions, basis in frames:
            local = (block - center[None]) @ basis
            explained |= np.all(
                np.abs(local) <= (0.5 * dimensions + margin)[None],
                axis=1,
            )
            if explained.all():
                break
        result[start:stop] = explained
    return result


def _nearest_aabb_distance(
    points: np.ndarray,
    global_corners: np.ndarray,
    *,
    normalization: float,
) -> np.ndarray:
    if not len(points) or not len(global_corners):
        return np.ones(len(points), dtype=np.float32)
    boxes = corners_to_center_size(global_corners)
    minmax = center_size_to_minmax(boxes)
    distances = np.full(len(points), np.inf, dtype=np.float64)
    for bounds in minmax:
        delta = np.maximum(
            np.maximum(bounds[:3][None] - points, points - bounds[3:][None]),
            0.0,
        )
        distances = np.minimum(distances, np.linalg.norm(delta, axis=1))
    return np.clip(distances / max(float(normalization), 1e-6), 0.0, 1.0).astype(
        np.float32
    )


def backproject_residual_depth(
    *,
    image: Any,
    depth: Any,
    intrinsics: Any,
    camera_to_world: Any,
    global_corners: Any,
    config: ResidualProposalConfig | Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
    """Backproject a strided RGB-D frame and remove B6-explained OBB points."""

    cfg = resolve_residual_proposal_config(config)
    image_array = np.asarray(image)
    depth_array = np.asarray(depth, dtype=np.float32)
    intrinsics_array = np.asarray(intrinsics, dtype=np.float64)
    transform = np.asarray(camera_to_world, dtype=np.float64)
    corners = np.asarray(global_corners, dtype=np.float64)
    if image_array.ndim != 3 or image_array.shape[2] != 3:
        raise ValueError("image must have shape [H,W,3]")
    if depth_array.ndim == 3 and 1 in (
        depth_array.shape[0],
        depth_array.shape[-1],
    ):
        depth_array = np.squeeze(depth_array)
    if depth_array.ndim != 2 or depth_array.shape != image_array.shape[:2]:
        raise ValueError("depth must align with image and have shape [H,W]")
    if intrinsics_array.shape == (4, 4):
        intrinsics_array = intrinsics_array[:3, :3]
    if intrinsics_array.shape != (3, 3) or transform.shape != (4, 4):
        raise ValueError("intrinsics/pose have invalid shapes")
    if corners.size == 0:
        corners = np.empty((0, 8, 3), dtype=np.float64)
    if corners.shape != (len(corners), 8, 3):
        raise ValueError("global_corners must have shape [B,8,3]")
    stride = int(cfg.depth_stride)
    rows = np.arange(0, depth_array.shape[0], stride, dtype=np.int64)
    cols = np.arange(0, depth_array.shape[1], stride, dtype=np.int64)
    vv, uu = np.meshgrid(rows, cols, indexing="ij")
    sampled_depth = depth_array[vv, uu]
    valid = (
        np.isfinite(sampled_depth)
        & (sampled_depth >= cfg.min_depth)
        & (sampled_depth <= cfg.max_depth)
    )
    input_count = int(valid.sum())
    if not input_count:
        return (
            np.empty((0, 3), dtype=np.float32),
            np.empty((0, 3), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
            0,
            0,
        )
    z = sampled_depth[valid].astype(np.float64)
    u = uu[valid].astype(np.float64)
    v = vv[valid].astype(np.float64)
    fx, fy = intrinsics_array[0, 0], intrinsics_array[1, 1]
    cx, cy = intrinsics_array[0, 2], intrinsics_array[1, 2]
    if fx <= 0.0 or fy <= 0.0:
        raise ValueError("intrinsics focal lengths must be positive")
    camera_points = np.stack(
        ((u - cx) * z / fx, (v - cy) * z / fy, z), axis=1
    )
    world_points = (
        camera_points @ transform[:3, :3].T + transform[:3, 3][None]
    )
    colors = image_array[v.astype(np.int64), u.astype(np.int64)]
    colors = np.asarray(colors, dtype=np.float32)
    if colors.size and (colors.max() > 1.0 or colors.min() < 0.0):
        colors = np.clip(colors, 0.0, 255.0) / 255.0
    else:
        colors = np.clip(colors, 0.0, 1.0)
    explained = points_explained_by_boxes(
        world_points, corners, margin=cfg.explained_margin
    )
    residual = ~explained
    return (
        world_points[residual].astype(np.float32),
        colors[residual].astype(np.float32),
        z[residual].astype(np.float32),
        input_count,
        int(explained.sum()),
    )


def voxelize_residual_points(
    points_world: Any,
    *,
    colors: Optional[Any] = None,
    ranges: Optional[Any] = None,
    camera_position: Optional[Any] = None,
    global_corners: Optional[Any] = None,
    config: ResidualProposalConfig | Mapping[str, Any],
    input_point_count: Optional[int] = None,
    explained_point_count: int = 0,
) -> ResidualVoxelBatch:
    """Aggregate residual points into a deterministic sparse feature table."""

    cfg = resolve_residual_proposal_config(config)
    points = np.asarray(points_world, dtype=np.float64)
    if points.size == 0:
        return ResidualVoxelBatch.empty(
            input_point_count=(
                0 if input_point_count is None else int(input_point_count)
            ),
            explained_point_count=int(explained_point_count),
            residual_point_count=0,
        )
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points_world must have shape [N,3]")
    if not np.isfinite(points).all():
        raise ValueError("points_world must be finite")
    color_array = (
        np.zeros((len(points), 3), dtype=np.float64)
        if colors is None
        else np.asarray(colors, dtype=np.float64)
    )
    if color_array.shape != (len(points), 3) or not np.isfinite(
        color_array
    ).all():
        raise ValueError("colors must be finite [N,3]")
    if color_array.size and (
        color_array.max() > 1.0 or color_array.min() < 0.0
    ):
        color_array = np.clip(color_array, 0.0, 255.0) / 255.0
    color_array = np.clip(color_array, 0.0, 1.0)
    camera = (
        np.zeros(3, dtype=np.float64)
        if camera_position is None
        else np.asarray(camera_position, dtype=np.float64)
    )
    if camera.shape != (3,) or not np.isfinite(camera).all():
        raise ValueError("camera_position must be finite [3]")
    point_ranges = (
        np.linalg.norm(points - camera[None], axis=1)
        if ranges is None
        else np.asarray(ranges, dtype=np.float64).reshape(-1)
    )
    if len(point_ranges) != len(points) or not np.isfinite(
        point_ranges
    ).all():
        raise ValueError("ranges must be finite [N]")
    corners = (
        np.empty((0, 8, 3), dtype=np.float64)
        if global_corners is None
        else np.asarray(global_corners, dtype=np.float64)
    )
    if corners.size == 0:
        corners = np.empty((0, 8, 3), dtype=np.float64)
    if corners.shape != (len(corners), 8, 3):
        raise ValueError("global_corners must have shape [B,8,3]")

    coordinates = np.floor(points / cfg.voxel_size).astype(np.int64)
    unique, inverse, counts = np.unique(
        coordinates, axis=0, return_inverse=True, return_counts=True
    )
    valid_voxels = np.flatnonzero(counts >= cfg.min_voxel_points)
    if not len(valid_voxels):
        return ResidualVoxelBatch.empty(
            input_point_count=(
                len(points)
                if input_point_count is None
                else int(input_point_count)
            ),
            explained_point_count=int(explained_point_count),
            residual_point_count=len(points),
        )
    if len(valid_voxels) > cfg.max_voxels:
        ranked = sorted(
            valid_voxels.tolist(),
            key=lambda index: (
                -int(counts[index]),
                int(unique[index, 0]),
                int(unique[index, 1]),
                int(unique[index, 2]),
            ),
        )
        valid_voxels = np.asarray(
            ranked[: cfg.max_voxels], dtype=np.int64
        )
        valid_voxels = valid_voxels[
            np.lexsort(
                (
                    unique[valid_voxels, 2],
                    unique[valid_voxels, 1],
                    unique[valid_voxels, 0],
                )
            )
        ]
    else:
        valid_voxels = np.asarray(valid_voxels, dtype=np.int64)

    selected_coords = unique[valid_voxels]
    centers = (selected_coords.astype(np.float64) + 0.5) * cfg.voxel_size
    coordinate_to_output = {
        int(source): output
        for output, source in enumerate(valid_voxels.tolist())
    }
    point_output = np.asarray(
        [coordinate_to_output.get(int(source), -1) for source in inverse],
        dtype=np.int64,
    )
    output_counts = counts[valid_voxels].astype(np.int32)
    features = np.zeros((len(valid_voxels), P1_FEATURE_DIM), dtype=np.float64)
    nearest_distance = _nearest_aabb_distance(
        points,
        corners,
        normalization=max(cfg.max_center_offset, cfg.voxel_size),
    )
    retained_points = point_output >= 0
    output_index = point_output[retained_points]
    retained_count = output_counts.astype(np.float64)
    color_sum = np.zeros((len(valid_voxels), 3), dtype=np.float64)
    relative_sum = np.zeros((len(valid_voxels), 3), dtype=np.float64)
    range_sum = np.zeros(len(valid_voxels), dtype=np.float64)
    range_square_sum = np.zeros(len(valid_voxels), dtype=np.float64)
    distance_sum = np.zeros(len(valid_voxels), dtype=np.float64)
    np.add.at(color_sum, output_index, color_array[retained_points])
    np.add.at(
        relative_sum,
        output_index,
        points[retained_points] - camera[None],
    )
    np.add.at(range_sum, output_index, point_ranges[retained_points])
    np.add.at(
        range_square_sum,
        output_index,
        np.square(point_ranges[retained_points]),
    )
    np.add.at(
        distance_sum,
        output_index,
        nearest_distance[retained_points],
    )
    features[:, 0] = np.clip(
        np.log1p(retained_count) / math.log(33.0), 0.0, 1.0
    )
    features[:, 1:4] = color_sum / retained_count[:, None]
    features[:, 4:7] = np.clip(
        relative_sum / retained_count[:, None] / cfg.max_depth,
        -1.0,
        1.0,
    )
    range_mean = range_sum / retained_count
    range_variance = np.maximum(
        range_square_sum / retained_count - np.square(range_mean),
        0.0,
    )
    features[:, 7] = np.clip(
        range_mean / cfg.max_depth, 0.0, 1.0
    )
    features[:, 8] = np.clip(
        np.sqrt(range_variance) / cfg.max_depth, 0.0, 1.0
    )
    features[:, 13] = distance_sum / retained_count
    occupied = {tuple(row.tolist()) for row in selected_coords}
    try:
        from scipy.spatial import cKDTree

        tree = cKDTree(selected_coords.astype(np.float64))
        for feature_index, radius in zip((9, 10, 11), (1, 2, 4)):
            counts_here = tree.query_ball_point(
                selected_coords.astype(np.float64),
                r=float(radius) + 1e-7,
                p=np.inf,
                return_length=True,
            )
            features[:, feature_index] = np.asarray(
                counts_here, dtype=np.float64
            ) / float((2 * radius + 1) ** 3)
    except (ImportError, TypeError):
        # SciPy-free deterministic fallback.  It is slower but retains the
        # exact feature definition used by the train/runtime checkpoint.
        for output_index, coordinate in enumerate(selected_coords):
            for feature_index, radius in zip((9, 10, 11), (1, 2, 4)):
                count = 0
                for dx in range(-radius, radius + 1):
                    for dy in range(-radius, radius + 1):
                        for dz in range(-radius, radius + 1):
                            if (
                                int(coordinate[0] + dx),
                                int(coordinate[1] + dy),
                                int(coordinate[2] + dz),
                            ) in occupied:
                                count += 1
                features[output_index, feature_index] = (
                    count / float((2 * radius + 1) ** 3)
                )
    vertical_balance = np.zeros(len(selected_coords), dtype=np.float64)
    for output_index, coordinate in enumerate(selected_coords):
        above = sum(
            (
                int(coordinate[0]),
                int(coordinate[1]),
                int(coordinate[2] + offset),
            )
            in occupied
            for offset in range(1, 5)
        )
        below = sum(
            (
                int(coordinate[0]),
                int(coordinate[1]),
                int(coordinate[2] - offset),
            )
            in occupied
            for offset in range(1, 5)
        )
        vertical_balance[output_index] = (above - below) / 4.0
    features[:, 12] = vertical_balance
    return ResidualVoxelBatch(
        coordinates=selected_coords.astype(np.int32),
        centers=centers.astype(np.float32),
        features=features.astype(np.float32),
        point_counts=output_counts,
        input_point_count=(
            len(points) if input_point_count is None else int(input_point_count)
        ),
        explained_point_count=int(explained_point_count),
        residual_point_count=len(points),
    )


if nn is not None:

    class ResidualVoxelProposalHead(nn.Module):
        """Small per-voxel class-agnostic head with a TR3D-like contract."""

        def __init__(
            self,
            input_dim: int = P1_FEATURE_DIM,
            hidden_dim: int = 64,
            regression_dim: int = 6,
        ) -> None:
            super().__init__()
            if (
                int(input_dim) < 1
                or int(hidden_dim) < 1
                or int(regression_dim) != 6
            ):
                raise ValueError(
                    "P1 input/hidden dimensions must be positive and "
                    "regression_dim=6"
                )
            self.input_dim = int(input_dim)
            self.hidden_dim = int(hidden_dim)
            self.regression_dim = int(regression_dim)
            self.backbone = nn.Sequential(
                nn.Linear(self.input_dim, self.hidden_dim),
                nn.ReLU(inplace=False),
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.ReLU(inplace=False),
            )
            self.objectness = nn.Linear(self.hidden_dim, 1)
            self.regression = nn.Linear(
                self.hidden_dim, self.regression_dim
            )

        def forward(self, features: "torch.Tensor"):
            if features.ndim != 2 or features.shape[1] != self.input_dim:
                raise ValueError(
                    f"features must have shape [N,{self.input_dim}]"
                )
            encoded = self.backbone(features)
            return self.objectness(encoded), self.regression(encoded)

        def model_config(self) -> dict[str, int]:
            return {
                "input_dim": self.input_dim,
                "hidden_dim": self.hidden_dim,
                "regression_dim": self.regression_dim,
            }


    ResidualProposalHead = ResidualVoxelProposalHead
else:  # pragma: no cover

    class ResidualVoxelProposalHead:  # type: ignore[no-redef]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise ImportError("PyTorch is required for the P1 residual head")


    ResidualProposalHead = ResidualVoxelProposalHead


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_residual_proposal_head(
    checkpoint_path: str | Path,
    *,
    expected_config: ResidualProposalConfig,
    device: str,
    expected_b6_checkpoint_sha256: str,
) -> tuple[ResidualVoxelProposalHead, str, Mapping[str, Any]]:
    if torch is None:
        raise ImportError("PyTorch is required to load the P1 head")
    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(f"missing P1 checkpoint: {path}")
    try:
        payload = torch.load(
            path, map_location="cpu", weights_only=False
        )
    except TypeError:  # older PyTorch
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, Mapping):
        raise ValueError("P1 checkpoint must contain a mapping")
    if payload.get("schema") != P1_HEAD_SCHEMA:
        raise ValueError("P1 checkpoint schema mismatch")
    feature_names = tuple(payload.get("feature_names", ()))
    if feature_names != P1_FEATURE_NAMES:
        raise ValueError("P1 checkpoint feature schema mismatch")
    model_config = payload.get("model_config")
    state_dict = payload.get("state_dict")
    provenance = payload.get("provenance")
    if not isinstance(model_config, Mapping) or not isinstance(
        state_dict, Mapping
    ):
        raise ValueError("P1 checkpoint lacks model_config/state_dict")
    if not isinstance(provenance, Mapping):
        raise ValueError("P1 checkpoint lacks train-only provenance")
    train_scene_ids = provenance.get("train_scene_ids")
    forbidden_overlap = provenance.get("forbidden_overlap")
    b6_sha = provenance.get("b6_checkpoint_sha256")
    train_list_sha = provenance.get("train_scene_list_sha256")
    forbidden_list_sha = provenance.get("forbidden_scene_list_sha256")
    sha_pattern = re.compile(r"^[0-9a-f]{64}$")
    scene_pattern = re.compile(r"^scene[0-9]{4}_[0-9]{2}$")
    expected_b6_sha = str(expected_b6_checkpoint_sha256).lower()
    if (
        not isinstance(train_scene_ids, Sequence)
        or isinstance(train_scene_ids, (str, bytes))
        or not train_scene_ids
        or any(
            not isinstance(scene, str)
            or scene_pattern.fullmatch(scene) is None
            for scene in train_scene_ids
        )
        or len(set(train_scene_ids)) != len(train_scene_ids)
        or forbidden_overlap != []
        or not isinstance(b6_sha, str)
        or sha_pattern.fullmatch(b6_sha.lower()) is None
        or not isinstance(train_list_sha, str)
        or sha_pattern.fullmatch(train_list_sha.lower()) is None
        or not isinstance(forbidden_list_sha, str)
        or sha_pattern.fullmatch(forbidden_list_sha.lower()) is None
        or sha_pattern.fullmatch(expected_b6_sha) is None
    ):
        raise ValueError("P1 checkpoint train-only provenance is invalid")
    if b6_sha.lower() != expected_b6_sha:
        raise ValueError(
            "P1 checkpoint was trained against a different frozen B6 "
            "checkpoint"
        )
    if int(model_config.get("regression_dim", -1)) != 6:
        raise ValueError("P1 checkpoint is not class-agnostic 6-D geometry")
    if int(model_config.get("input_dim", -1)) != P1_FEATURE_DIM:
        raise ValueError("P1 checkpoint input_dim disagrees with feature schema")
    fork_devices: list[int] = []
    if str(device).startswith("cuda") and torch.cuda.is_available():
        parsed_device = torch.device(device)
        fork_devices = [
            torch.cuda.current_device()
            if parsed_device.index is None
            else int(parsed_device.index)
        ]
    # nn.Linear initializes parameters before the checkpoint overwrites them.
    # Preserve the caller's global CPU/CUDA RNG state so a diagnostics-only
    # observer cannot perturb later BoxFusion random-number consumers.
    with torch.random.fork_rng(devices=fork_devices, enabled=True):
        model = ResidualVoxelProposalHead(
            input_dim=int(model_config.get("input_dim", -1)),
            hidden_dim=int(model_config.get("hidden_dim", -1)),
            regression_dim=int(model_config.get("regression_dim", -1)),
        )
        model.load_state_dict(dict(state_dict), strict=True)
        model.to(device)
    model.eval()
    if model.hidden_dim != expected_config.hidden_dim:
        raise ValueError(
            "P1 checkpoint hidden_dim disagrees with runtime config"
        )
    return model, sha256_file(path), payload


def assign_residual_targets(
    voxel_centers: Any,
    target_boxes: Any,
    *,
    topk: int = 6,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Assign class-agnostic GT targets to sparse anchors for train-only use.

    The encoding is ``centre_gt - centre_voxel`` in metres followed by
    ``log(size_gt_metres)``.  Class labels are intentionally absent.
    """

    centers = np.asarray(voxel_centers, dtype=np.float64)
    targets = np.asarray(target_boxes, dtype=np.float64)
    if centers.size == 0:
        centers = np.empty((0, 3), dtype=np.float64)
    if targets.size == 0:
        targets = np.empty((0, 6), dtype=np.float64)
    if centers.ndim != 2 or centers.shape[1] != 3:
        raise ValueError("voxel_centers must have shape [V,3]")
    if targets.ndim != 2 or targets.shape[1] != 6:
        raise ValueError("target_boxes must have shape [G,6]")
    if (
        not np.isfinite(centers).all()
        or not np.isfinite(targets).all()
        or (len(targets) and np.any(targets[:, 3:] <= 0.0))
    ):
        raise ValueError("target assignment inputs are invalid")
    topk = _positive_int(topk, "topk")
    objectness = np.zeros(len(centers), dtype=np.float32)
    regression = np.zeros((len(centers), 6), dtype=np.float32)
    assigned = np.full(len(centers), -1, dtype=np.int64)
    if not len(centers) or not len(targets):
        return objectness, regression, assigned
    target_minmax = center_size_to_minmax(targets)
    assignments: list[tuple[float, int, int]] = []
    for target_index, target in enumerate(targets):
        bounds = target_minmax[target_index]
        inside = np.flatnonzero(
            np.all(
                (centers >= bounds[:3][None])
                & (centers <= bounds[3:][None]),
                axis=1,
            )
        )
        pool = inside if len(inside) else np.arange(len(centers))
        distances = np.linalg.norm(
            centers[pool] - target[:3][None], axis=1
        )
        order = np.lexsort((pool, distances))
        for local in order[: min(topk, len(order))]:
            voxel_index = int(pool[int(local)])
            assignments.append(
                (float(distances[int(local)]), target_index, voxel_index)
            )
    for distance, target_index, voxel_index in sorted(
        assignments, key=lambda row: (row[0], row[1], row[2])
    ):
        if assigned[voxel_index] >= 0:
            continue
        assigned[voxel_index] = target_index
        objectness[voxel_index] = 1.0
        regression[voxel_index, :3] = (
            targets[target_index, :3] - centers[voxel_index]
        ).astype(np.float32)
        regression[voxel_index, 3:] = np.log(
            targets[target_index, 3:]
        ).astype(np.float32)
    return objectness, regression, assigned


class P1ResidualProposalObserver:
    """Bounded, diagnostics-only online P1 observer."""

    def __init__(
        self,
        config: Mapping[str, Any] | ResidualProposalConfig,
        *,
        head: Optional[Any] = None,
        device: Optional[str] = None,
        expected_b6_checkpoint_sha256: Optional[str] = None,
    ) -> None:
        self.config = resolve_residual_proposal_config(config)
        self.device = str(
            device or self.config.device or ("cuda" if torch is not None and torch.cuda.is_available() else "cpu")
        )
        self.head = None
        self.checkpoint_sha256 = ""
        self.checkpoint_metadata: Mapping[str, Any] = {}
        if self.config.enabled and self.config.mode == "infer":
            if head is not None:
                self.head = head
                self.checkpoint_sha256 = "injected"
            elif self.config.checkpoint is None:
                raise ValueError(
                    "enabled P1 infer mode requires a checkpoint or injected head"
                )
            else:
                (
                    self.head,
                    self.checkpoint_sha256,
                    self.checkpoint_metadata,
                ) = load_residual_proposal_head(
                    self.config.checkpoint,
                    expected_config=self.config,
                    device=self.device,
                    expected_b6_checkpoint_sha256=(
                        expected_b6_checkpoint_sha256 or ""
                    ),
                )
            if hasattr(self.head, "eval"):
                self.head.eval()
        elif head is not None:
            raise ValueError("injected P1 head requires enabled infer mode")
        self.scene_id: Optional[str] = None
        self.observations: list[ResidualObservation] = []

    def reset(self, scene_id: str) -> None:
        if not isinstance(scene_id, str) or not scene_id.strip():
            raise ValueError("scene_id must be a non-empty string")
        self.scene_id = scene_id.strip()
        self.observations.clear()

    def build_voxel_batch(
        self,
        points_world: Any,
        *,
        colors: Optional[Any] = None,
        ranges: Optional[Any] = None,
        camera_position: Optional[Any] = None,
        global_corners: Optional[Any] = None,
        input_point_count: Optional[int] = None,
        explained_point_count: int = 0,
    ) -> ResidualVoxelBatch:
        points = np.asarray(points_world)
        corners = (
            np.empty((0, 8, 3), dtype=np.float32)
            if global_corners is None
            else np.asarray(global_corners)
        )
        color_values = None if colors is None else np.asarray(colors)
        range_values = None if ranges is None else np.asarray(ranges)
        # This public helper accepts either raw world points (tests/offline
        # collection) or points already filtered by backproject_residual_depth
        # (online path). Applying the same OBB test twice is idempotent.
        if len(points) and len(corners):
            explained = points_explained_by_boxes(
                points,
                corners,
                margin=self.config.explained_margin,
            )
            original_count = len(points)
            points = points[~explained]
            if color_values is not None:
                color_values = color_values[~explained]
            if range_values is not None:
                range_values = range_values[~explained]
            if input_point_count is None:
                input_point_count = original_count
            explained_point_count += int(explained.sum())
        return voxelize_residual_points(
            points,
            colors=color_values,
            ranges=range_values,
            camera_position=camera_position,
            global_corners=corners,
            config=self.config,
            input_point_count=input_point_count,
            explained_point_count=explained_point_count,
        )

    def _decode(
        self,
        batch: ResidualVoxelBatch,
        objectness_logits: Any,
        regression: Any,
        *,
        scene_id: str,
        frame_index: int,
        provider_step: int,
        global_corners: np.ndarray,
        global_stable_ids: np.ndarray,
    ) -> tuple[ResidualProposal, ...]:
        logits = np.asarray(objectness_logits, dtype=np.float64).reshape(-1)
        residuals = np.asarray(regression, dtype=np.float64)
        if len(logits) != len(batch.centers) or residuals.shape != (
            len(batch.centers),
            6,
        ):
            raise RuntimeError("P1 head returned an invalid output shape")
        if not np.isfinite(logits).all() or not np.isfinite(
            residuals
        ).all():
            raise RuntimeError("P1 head returned non-finite values")
        scores = 1.0 / (1.0 + np.exp(-np.clip(logits, -40.0, 40.0)))
        candidates = np.flatnonzero(scores >= self.config.score_threshold)
        candidates = np.asarray(
            sorted(
                candidates.tolist(),
                key=lambda index: (
                    -scores[index],
                    int(batch.coordinates[index, 0]),
                    int(batch.coordinates[index, 1]),
                    int(batch.coordinates[index, 2]),
                ),
            )[: self.config.pre_nms_topk],
            dtype=np.int64,
        )
        if not len(candidates):
            return ()
        delta = np.clip(
            residuals[candidates, :3],
            -self.config.max_center_offset,
            self.config.max_center_offset,
        )
        log_min = math.log(self.config.min_box_extent)
        log_max = math.log(self.config.max_box_extent)
        dimensions = np.exp(
            np.clip(residuals[candidates, 3:], log_min, log_max)
        )
        boxes = np.concatenate(
            (batch.centers[candidates] + delta, dimensions), axis=1
        ).astype(np.float32)
        tie_ids = [
            (
                f"{scene_id}:{provider_step:06d}:"
                f"{int(batch.coordinates[index, 0])}:"
                f"{int(batch.coordinates[index, 1])}:"
                f"{int(batch.coordinates[index, 2])}"
            )
            for index in candidates
        ]
        keep = stable_nms_aabb(
            boxes,
            scores[candidates],
            self.config.nms_iou,
            tie_breakers=tie_ids,
            max_output=self.config.max_candidates_per_step,
        )
        boxes = boxes[keep]
        selected = candidates[keep]
        selected_scores = scores[selected]
        selected_ids = [tie_ids[int(index)] for index in keep]
        global_boxes = corners_to_center_size(global_corners)
        overlaps = pairwise_aabb_iou(boxes, global_boxes)
        proposals: list[ResidualProposal] = []
        for rank, (box, score, voxel_index, candidate_id) in enumerate(
            zip(boxes, selected_scores, selected, selected_ids)
        ):
            del rank
            if overlaps.shape[1]:
                nearest_index = int(np.argmax(overlaps[len(proposals)]))
                nearest_iou = float(overlaps[len(proposals), nearest_index])
                nearest_id = int(global_stable_ids[nearest_index])
            else:
                nearest_iou = 0.0
                nearest_id = -1
            proposals.append(
                ResidualProposal(
                    candidate_id=candidate_id,
                    frame_index=int(frame_index),
                    provider_step=int(provider_step),
                    box=box.astype(np.float32),
                    corners=center_size_to_corners(box)[0],
                    objectness=float(score),
                    residual_point_count=int(
                        batch.point_counts[int(voxel_index)]
                    ),
                    nearest_b6_stable_id=nearest_id,
                    nearest_b6_iou=nearest_iou,
                )
            )
        return tuple(proposals)

    def decode(
        self,
        batch: ResidualVoxelBatch,
        objectness_logits: Any,
        regression: Any,
        *,
        scene_id: str,
        frame_index: int,
        provider_step: int,
        global_corners: Optional[Any] = None,
        global_stable_ids: Optional[Any] = None,
    ) -> tuple[ResidualProposal, ...]:
        """Public deterministic decoder used by tests and cache adapters."""

        corners = (
            np.empty((0, 8, 3), dtype=np.float32)
            if global_corners is None
            else np.asarray(global_corners, dtype=np.float32)
        )
        ids = (
            np.empty((0,), dtype=np.int64)
            if global_stable_ids is None
            else np.asarray(global_stable_ids, dtype=np.int64).reshape(-1)
        )
        if corners.size == 0:
            corners = np.empty((0, 8, 3), dtype=np.float32)
        if corners.shape != (len(corners), 8, 3) or len(ids) != len(corners):
            raise ValueError("decode global boxes and IDs must align")
        if torch is not None and isinstance(objectness_logits, torch.Tensor):
            objectness_logits = objectness_logits.detach().cpu().numpy()
        if torch is not None and isinstance(regression, torch.Tensor):
            regression = regression.detach().cpu().numpy()
        return self._decode(
            batch,
            objectness_logits,
            regression,
            scene_id=scene_id,
            frame_index=frame_index,
            provider_step=provider_step,
            global_corners=corners,
            global_stable_ids=ids,
        )

    def observe(
        self,
        *,
        image: Any,
        depth: Any,
        intrinsics: Any,
        camera_to_world: Any,
        global_corners: Any,
        global_stable_ids: Any,
        frame_index: int,
        provider_step: int,
        scene_id: str,
    ) -> ResidualObservation:
        if not self.config.enabled:
            raise RuntimeError("cannot observe with disabled P1")
        if not isinstance(scene_id, str) or not scene_id.strip():
            raise ValueError("scene_id must be a non-empty string")
        requested_scene = scene_id.strip()
        if self.scene_id != requested_scene:
            self.reset(requested_scene)
        corners = np.asarray(global_corners, dtype=np.float32)
        ids = np.asarray(global_stable_ids, dtype=np.int64).reshape(-1)
        if corners.size == 0:
            corners = np.empty((0, 8, 3), dtype=np.float32)
        if corners.shape != (len(corners), 8, 3) or len(ids) != len(corners):
            raise ValueError("global P1 boxes and stable IDs must align")
        if len(set(ids.tolist())) != len(ids) or np.any(ids < 0):
            raise ValueError("global stable IDs must be unique/non-negative")
        started = time.perf_counter()
        (
            residual_points,
            colors,
            ranges,
            input_count,
            explained_count,
        ) = backproject_residual_depth(
            image=image,
            depth=depth,
            intrinsics=intrinsics,
            camera_to_world=camera_to_world,
            global_corners=corners,
            config=self.config,
        )
        pose = np.asarray(camera_to_world, dtype=np.float32)
        batch = self.build_voxel_batch(
            residual_points,
            colors=colors,
            ranges=ranges,
            camera_position=pose[:3, 3],
            global_corners=corners,
            input_point_count=input_count,
            explained_point_count=explained_count,
        )
        voxelize_seconds = time.perf_counter() - started
        head_seconds = 0.0
        nms_seconds = 0.0
        proposals: tuple[ResidualProposal, ...] = ()
        if self.config.mode == "infer" and len(batch.centers):
            if self.head is None:
                raise RuntimeError("P1 infer mode has no loaded head")
            head_started = time.perf_counter()
            if torch is not None and isinstance(self.head, nn.Module):
                tensor = torch.tensor(
                    np.asarray(batch.features),
                    dtype=torch.float32,
                    device=self.device,
                )
                with torch.inference_mode():
                    logits_tensor, regression_tensor = self.head(tensor)
                logits = logits_tensor.detach().cpu().numpy()
                regression = regression_tensor.detach().cpu().numpy()
            else:
                logits, regression = self.head(batch.features)
            head_seconds = time.perf_counter() - head_started
            nms_started = time.perf_counter()
            proposals = self._decode(
                batch,
                logits,
                regression,
                scene_id=requested_scene,
                frame_index=int(frame_index),
                provider_step=int(provider_step),
                global_corners=corners,
                global_stable_ids=ids,
            )
            nms_seconds = time.perf_counter() - nms_started
        observation = ResidualObservation(
            frame_index=int(frame_index),
            provider_step=int(provider_step),
            voxel_batch=batch,
            proposals=proposals,
            voxelize_seconds=float(voxelize_seconds),
            head_seconds=float(head_seconds),
            nms_seconds=float(nms_seconds),
        )
        self.observations.append(observation)
        del self.observations[: -self.config.max_history_steps]
        return observation

    def scene_candidates(self) -> tuple[ResidualProposal, ...]:
        rows = [
            proposal
            for observation in self.observations
            for proposal in observation.proposals
        ]
        if not rows:
            return ()
        boxes = np.stack([row.box for row in rows])
        scores = np.asarray([row.objectness for row in rows])
        ids = [row.candidate_id for row in rows]
        keep = stable_nms_aabb(
            boxes,
            scores,
            self.config.scene_nms_iou,
            tie_breakers=ids,
            max_output=self.config.max_scene_candidates,
        )
        return tuple(rows[int(index)] for index in keep)

    def diagnostic_payload(self) -> dict[str, np.ndarray]:
        """Build a pickle-free, scene-level P1 diagnostics payload."""

        observations = tuple(self.observations)
        candidates = self.scene_candidates()
        raw_candidates = tuple(
            proposal
            for observation in observations
            for proposal in observation.proposals
        )
        voxel_offsets = [0]
        for observation in observations:
            voxel_offsets.append(
                voxel_offsets[-1] + len(observation.voxel_batch.centers)
            )
        if (
            observations
            and self.config.collect_voxel_inputs
            and voxel_offsets[-1] > 0
        ):
            voxel_coordinates = np.concatenate(
                [row.voxel_batch.coordinates for row in observations],
                axis=0,
            ).astype(np.int32)
            voxel_centers = np.concatenate(
                [row.voxel_batch.centers for row in observations], axis=0
            ).astype(np.float32)
            voxel_features = np.concatenate(
                [row.voxel_batch.features for row in observations], axis=0
            ).astype(np.float32)
            voxel_point_counts = np.concatenate(
                [row.voxel_batch.point_counts for row in observations],
                axis=0,
            ).astype(np.int32)
        else:
            voxel_offsets = [0] * (len(observations) + 1)
            voxel_coordinates = np.empty((0, 3), dtype=np.int32)
            voxel_centers = np.empty((0, 3), dtype=np.float32)
            voxel_features = np.empty(
                (0, P1_FEATURE_DIM), dtype=np.float32
            )
            voxel_point_counts = np.empty((0,), dtype=np.int32)

        def proposal_array(
            rows: Sequence[ResidualProposal],
            attribute: str,
            empty_shape: tuple[int, ...],
            dtype: Any,
        ) -> np.ndarray:
            if not rows:
                return np.empty(empty_shape, dtype=dtype)
            return np.asarray(
                [getattr(row, attribute) for row in rows], dtype=dtype
            )

        config_json = json.dumps(
            self.config.to_dict(),
            sort_keys=True,
            separators=(",", ":"),
        )
        return {
            "p1_schema": np.asarray(P1_DIAGNOSTIC_SCHEMA),
            "p1_stage": np.asarray("P1"),
            "p1_profile": np.asarray("p1_residual_proposal_observer"),
            "p1_enabled": np.asarray(self.config.enabled, dtype=bool),
            "p1_observer_only": np.asarray(True, dtype=bool),
            "p1_uses_ground_truth": np.asarray(False, dtype=bool),
            "p1_mutation_enabled": np.asarray(False, dtype=bool),
            "p1_applied_count": np.asarray(0, dtype=np.int64),
            "p1_complete": np.asarray(True, dtype=bool),
            "p1_class_agnostic": np.asarray(True, dtype=bool),
            "p1_regression_dim": np.asarray(6, dtype=np.int64),
            "p1_checkpoint_sha256": np.asarray(
                self.checkpoint_sha256
            ),
            "p1_config_json": np.asarray(config_json),
            "p1_feature_names": np.asarray(
                P1_FEATURE_NAMES, dtype=np.str_
            ),
            "p1_step_frame_ids": np.asarray(
                [row.frame_index for row in observations], dtype=np.int64
            ),
            "p1_step_provider_steps": np.asarray(
                [row.provider_step for row in observations], dtype=np.int64
            ),
            "p1_step_input_point_counts": np.asarray(
                [
                    row.voxel_batch.input_point_count
                    for row in observations
                ],
                dtype=np.int64,
            ),
            "p1_step_explained_point_counts": np.asarray(
                [
                    row.voxel_batch.explained_point_count
                    for row in observations
                ],
                dtype=np.int64,
            ),
            "p1_step_residual_point_counts": np.asarray(
                [
                    row.voxel_batch.residual_point_count
                    for row in observations
                ],
                dtype=np.int64,
            ),
            "p1_step_voxel_counts": np.asarray(
                [len(row.voxel_batch.centers) for row in observations],
                dtype=np.int64,
            ),
            "p1_step_candidate_counts": np.asarray(
                [len(row.proposals) for row in observations],
                dtype=np.int64,
            ),
            "p1_step_voxelize_seconds": np.asarray(
                [row.voxelize_seconds for row in observations],
                dtype=np.float64,
            ),
            "p1_step_head_seconds": np.asarray(
                [row.head_seconds for row in observations],
                dtype=np.float64,
            ),
            "p1_step_nms_seconds": np.asarray(
                [row.nms_seconds for row in observations],
                dtype=np.float64,
            ),
            "p1_voxel_offsets": np.asarray(
                voxel_offsets, dtype=np.int64
            ),
            "p1_voxel_coords": voxel_coordinates,
            "p1_voxel_centers": voxel_centers,
            "p1_voxel_features": voxel_features,
            "p1_voxel_point_counts": voxel_point_counts,
            "p1_candidate_ids": proposal_array(
                candidates, "candidate_id", (0,), np.str_
            ),
            "p1_candidate_frame_ids": proposal_array(
                candidates, "frame_index", (0,), np.int64
            ),
            "p1_candidate_provider_steps": proposal_array(
                candidates, "provider_step", (0,), np.int64
            ),
            "p1_candidate_boxes": proposal_array(
                candidates, "box", (0, 6), np.float32
            ),
            "p1_candidate_corners": proposal_array(
                candidates, "corners", (0, 8, 3), np.float32
            ),
            "p1_candidate_scores": proposal_array(
                candidates, "objectness", (0,), np.float32
            ),
            "p1_candidate_objectness": proposal_array(
                candidates, "objectness", (0,), np.float32
            ),
            "p1_candidate_residual_point_counts": proposal_array(
                candidates, "residual_point_count", (0,), np.int64
            ),
            "p1_candidate_nearest_b6_stable_ids": proposal_array(
                candidates, "nearest_b6_stable_id", (0,), np.int64
            ),
            "p1_candidate_nearest_b6_iou": proposal_array(
                candidates, "nearest_b6_iou", (0,), np.float32
            ),
            "p1_raw_candidate_ids": proposal_array(
                raw_candidates, "candidate_id", (0,), np.str_
            ),
            "p1_raw_candidate_boxes": proposal_array(
                raw_candidates, "box", (0, 6), np.float32
            ),
            "p1_raw_candidate_scores": proposal_array(
                raw_candidates, "objectness", (0,), np.float32
            ),
        }


__all__ = [
    "P1_DIAGNOSTIC_SCHEMA",
    "P1_FEATURE_DIM",
    "P1_FEATURE_NAMES",
    "P1_HEAD_SCHEMA",
    "P1ResidualProposalObserver",
    "ResidualObservation",
    "ResidualProposal",
    "ResidualProposalConfig",
    "ResidualProposalHead",
    "ResidualVoxelBatch",
    "ResidualVoxelProposalHead",
    "assign_residual_targets",
    "backproject_residual_depth",
    "center_size_to_corners",
    "corners_to_center_size",
    "load_residual_proposal_head",
    "pairwise_aabb_iou",
    "points_explained_by_boxes",
    "resolve_residual_proposal_config",
    "score_ordered_match",
    "sha256_file",
    "stable_nms_aabb",
    "voxelize_residual_points",
]
