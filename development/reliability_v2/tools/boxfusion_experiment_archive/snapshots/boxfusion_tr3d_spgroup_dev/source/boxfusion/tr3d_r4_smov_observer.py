"""Paired SMOV3D-inspired depth observer for terminal R3 replacements.

R2a scored every raw TR3D proposal independently.  R4 has a different task:
it observes only a terminal R3 replacement and the G0 anchor that replacement
would overwrite.  Both boxes are evaluated on the exact same causal views so
that support/free-space differences cannot be caused by different Top-K view
selection.

This module is pure compute.  It has no prediction writer, no ground-truth or
CLIP input, and no active veto path.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
import time
from typing import Any, Optional

import numpy as np

from .tr3d_r2_geometry import (
    classify_depth_rays,
    compose_depth_camera_to_world,
    project_yaw_obb_to_depth,
    stable_top_k_view_indices,
)
from .tr3d_r2_observer import (
    R2_DEPTH_CLASS_NAMES,
    TR3DR2FrameBundle,
    TR3DR2ObserverConfig,
    _decoded_depth,
    _default_depth_decoder,
    _default_pose_loader,
    _fractions_from_counts,
    _normalized_resource_map,
    _resource_matrix,
    _strict_manifest_frame_ids,
)


R4_PAIR_ROLES = ("anchor", "candidate")
_ROLE_COUNT = len(R4_PAIR_ROLES)
_DEPTH_CLASS_COUNT = len(R2_DEPTH_CLASS_NAMES)


def _readonly(value: object, dtype: np.dtype) -> np.ndarray:
    result = np.ascontiguousarray(value, dtype=dtype)
    result.setflags(write=False)
    return result


def _boxes(value: object, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.ndim != 2 or result.shape[1] != 7:
        raise ValueError(f"{name} must be finite [N,7] yaw boxes")
    if not np.isfinite(result).all() or np.any(result[:, 3:6] <= 0.0):
        raise ValueError(f"{name} must be finite with positive dimensions")
    return np.ascontiguousarray(result)


def _unique_nonnegative_ids(value: object, count: int, name: str) -> np.ndarray:
    raw = np.asarray(value)
    if raw.dtype.kind not in "iu" or raw.shape != (count,):
        raise ValueError(f"{name} must be an integer [N] array")
    result = raw.astype(np.int64, copy=False)
    if np.any(result < 0) or len(np.unique(result)) != count:
        raise ValueError(f"{name} must be unique and nonnegative")
    return np.ascontiguousarray(result)


def corners_to_yaw_boxes(corners_world: object) -> np.ndarray:
    """Convert gravity-aligned unordered cuboid corners to canonical yaw OBBs.

    The selected R3 candidate already carries its canonical TR3D yaw box, but
    the pre-R3 G0 prediction is stored as eight unordered corners.  This
    deterministic minimum-area XY rectangle conversion provides the paired
    anchor representation required by the depth ray caster.  Dimension order
    is canonicalized so ``dx >= dy``; swapping axes describes the same OBB.
    """

    corners = np.asarray(corners_world, dtype=np.float64)
    if corners.ndim != 3 or corners.shape[1:] != (8, 3):
        raise ValueError("corners_world must be finite [N,8,3]")
    if not np.isfinite(corners).all():
        raise ValueError("corners_world must be finite [N,8,3]")
    output = np.empty((len(corners), 7), dtype=np.float64)
    for row, points in enumerate(corners):
        z_min = float(points[:, 2].min())
        z_max = float(points[:, 2].max())
        if z_max - z_min <= 1e-8:
            raise ValueError(f"corner row {row} has zero height")
        xy = points[:, :2]
        differences = xy[:, None, :] - xy[None, :, :]
        lengths = np.linalg.norm(differences, axis=2)
        candidates: list[float] = []
        for vector in differences[lengths > 1e-8]:
            angle = float(np.arctan2(vector[1], vector[0]))
            # Rectangles repeat after pi/2.  A fixed half-open interval makes
            # ties invariant to corner order and input edge direction.
            angle = (angle + np.pi / 4.0) % (np.pi / 2.0) - np.pi / 4.0
            candidates.append(angle)
        if not candidates:
            raise ValueError(f"corner row {row} has zero XY extent")
        candidates = sorted(set(round(value, 12) for value in candidates))
        best: tuple[float, float, float, float, float, float] | None = None
        for angle in candidates:
            cosine, sine = float(np.cos(angle)), float(np.sin(angle))
            rotation = np.asarray(
                [[cosine, sine], [-sine, cosine]], dtype=np.float64
            )
            local = xy @ rotation.T
            minimum = local.min(axis=0)
            maximum = local.max(axis=0)
            dimensions = maximum - minimum
            area = float(dimensions[0] * dimensions[1])
            key = (
                round(area, 12),
                abs(float(angle)),
                float(angle),
                float(dimensions[0]),
                float(dimensions[1]),
                float((minimum[0] + maximum[0]) * 0.5),
            )
            if best is None or key < best:
                best = key
                best_angle = float(angle)
                best_rotation = rotation
                best_minimum = minimum
                best_maximum = maximum
        assert best is not None
        dimensions = best_maximum - best_minimum
        local_centre = (best_minimum + best_maximum) * 0.5
        centre_xy = local_centre @ best_rotation
        dx, dy = float(dimensions[0]), float(dimensions[1])
        yaw = best_angle
        if dy > dx:
            dx, dy = dy, dx
            yaw += np.pi / 2.0
        yaw = (yaw + np.pi / 2.0) % np.pi - np.pi / 2.0
        if min(dx, dy) <= 1e-8:
            raise ValueError(f"corner row {row} has zero XY dimension")
        output[row] = (
            float(centre_xy[0]),
            float(centre_xy[1]),
            (z_min + z_max) * 0.5,
            dx,
            dy,
            z_max - z_min,
            yaw,
        )
    return np.ascontiguousarray(output)


@dataclass(frozen=True)
class R4PairedDepthObservation:
    scene_id: str
    pose_source: str
    proposal_ids: np.ndarray
    anchor_indices: np.ndarray
    used_frame_ids: np.ndarray
    decoded_frame_ids: np.ndarray
    topk_frame_ids: np.ndarray
    topk_view_valid: np.ndarray
    topk_projected_area_pixels: np.ndarray
    topk_projected_area_fraction: np.ndarray
    per_view_depth_counts: np.ndarray
    per_view_depth_evidence: np.ndarray
    per_view_point_count: np.ndarray
    aggregate_depth_counts: np.ndarray
    aggregate_depth_evidence: np.ndarray
    aggregate_view_count: np.ndarray
    aggregate_point_count: np.ndarray
    candidate_minus_anchor_evidence: np.ndarray
    runtime_s: float

    @property
    def pair_count(self) -> int:
        return int(self.proposal_ids.shape[0])

    @property
    def topk(self) -> int:
        return int(self.topk_frame_ids.shape[1])


def observe_r3_replacement_pairs(
    *,
    anchor_boxes_world: object,
    candidate_boxes_world: object,
    proposal_ids: object,
    anchor_indices: object,
    prefix_manifest: Mapping[str, Any],
    frame_bundle: TR3DR2FrameBundle,
    config: TR3DR2ObserverConfig,
    decode_depth: Optional[Callable[[object], object]] = None,
    load_pose: Optional[Callable[[object], object]] = None,
) -> R4PairedDepthObservation:
    """Measure anchor/candidate evidence on one stable common Top-K."""

    started = time.perf_counter()
    if not isinstance(frame_bundle, TR3DR2FrameBundle):
        raise ValueError("frame_bundle must be TR3DR2FrameBundle")
    if not isinstance(config, TR3DR2ObserverConfig):
        raise ValueError("config must be TR3DR2ObserverConfig")
    anchors = _boxes(anchor_boxes_world, "anchor_boxes_world")
    candidates = _boxes(candidate_boxes_world, "candidate_boxes_world")
    if anchors.shape != candidates.shape:
        raise ValueError("anchor and candidate boxes must have identical shape")
    count = len(anchors)
    ids = _unique_nonnegative_ids(proposal_ids, count, "proposal_ids")
    anchor_ids = _unique_nonnegative_ids(
        anchor_indices, count, "anchor_indices"
    )
    depth_resources = _normalized_resource_map("depth", frame_bundle.depth)
    pose_resources = _normalized_resource_map("pose", frame_bundle.pose)
    frame_ids = _strict_manifest_frame_ids(
        prefix_manifest,
        bundle=frame_bundle,
        config=config,
        depth_resources=depth_resources,
        pose_resources=pose_resources,
    )
    intrinsic = _resource_matrix(
        frame_bundle.intrinsic_depth, "intrinsic_depth"
    )
    extrinsic = _resource_matrix(
        frame_bundle.extrinsic_depth, "extrinsic_depth"
    )
    pose_decoder = _default_pose_loader if load_pose is None else load_pose
    depth_decoder = _default_depth_decoder if decode_depth is None else decode_depth

    camera_to_world: dict[int, np.ndarray] = {}
    for frame_id in frame_ids.tolist():
        try:
            pose = np.asarray(
                pose_decoder(pose_resources[frame_id]), dtype=np.float64
            )
        except (OSError, TypeError, ValueError) as error:
            raise ValueError(f"frame {frame_id}: pose decode failed") from error
        camera_to_world[frame_id] = compose_depth_camera_to_world(
            pose, extrinsic
        )

    frame_count = len(frame_ids)
    projected_area = np.zeros(
        (count, frame_count, _ROLE_COUNT), dtype=np.float64
    )
    projected_fraction = np.zeros_like(projected_area)
    projected_valid = np.zeros(
        (count, frame_count, _ROLE_COUNT), dtype=np.bool_
    )
    paired_boxes = np.stack((anchors, candidates), axis=1)
    for frame_index, frame_id in enumerate(frame_ids.tolist()):
        transform = camera_to_world[frame_id]
        for pair_index in range(count):
            for role_index in range(_ROLE_COUNT):
                projection = project_yaw_obb_to_depth(
                    paired_boxes[pair_index, role_index],
                    intrinsic,
                    transform,
                    config.image_shape,
                    near_clip=config.near_clip,
                )
                if projection is None:
                    continue
                projected_area[pair_index, frame_index, role_index] = (
                    projection.area_pixels
                )
                projected_fraction[pair_index, frame_index, role_index] = (
                    projection.area_ratio
                )
                projected_valid[pair_index, frame_index, role_index] = True

    topk_frame_ids = np.full((count, config.top_k), -1, dtype=np.int64)
    topk_valid = np.zeros((count, config.top_k), dtype=np.bool_)
    topk_area = np.zeros(
        (count, config.top_k, _ROLE_COUNT), dtype=np.float32
    )
    topk_fraction = np.zeros_like(topk_area)
    for pair_index in range(count):
        common_valid = projected_valid[pair_index].all(axis=1)
        # The minimum area is the shared visible footprint and prevents one
        # oversized box from unilaterally selecting a view.
        reliability = projected_area[pair_index].min(axis=1)
        selected = stable_top_k_view_indices(
            reliability,
            config.top_k,
            frame_ids=frame_ids,
            valid_mask=common_valid,
        )
        valid_count = len(selected)
        if not valid_count:
            continue
        topk_frame_ids[pair_index, :valid_count] = frame_ids[selected]
        topk_valid[pair_index, :valid_count] = True
        topk_area[pair_index, :valid_count] = projected_area[
            pair_index, selected
        ]
        topk_fraction[pair_index, :valid_count] = projected_fraction[
            pair_index, selected
        ]

    per_view_counts = np.zeros(
        (count, config.top_k, _ROLE_COUNT, _DEPTH_CLASS_COUNT),
        dtype=np.int32,
    )
    decoded: dict[int, np.ndarray] = {}
    decoded_order: list[int] = []
    for pair_index in range(count):
        for slot in range(config.top_k):
            if not topk_valid[pair_index, slot]:
                continue
            frame_id = int(topk_frame_ids[pair_index, slot])
            if frame_id not in decoded:
                decoded[frame_id] = _decoded_depth(
                    frame_id,
                    depth_resources[frame_id],
                    depth_decoder,
                    config,
                )
                decoded_order.append(frame_id)
            for role_index in range(_ROLE_COUNT):
                classification = classify_depth_rays(
                    decoded[frame_id],
                    paired_boxes[pair_index, role_index],
                    intrinsic,
                    camera_to_world[frame_id],
                    pixel_stride=config.pixel_stride,
                    margin=config.margin,
                    min_depth=config.min_depth,
                    max_depth=config.max_depth,
                    near_clip=config.near_clip,
                )
                if classification is None or classification.sample_count < 1:
                    raise ValueError(
                        f"pair {pair_index}/frame {frame_id}/{R4_PAIR_ROLES[role_index]}: "
                        "common projection has no classifiable samples"
                    )
                values = np.asarray(
                    (
                        classification.support_count,
                        classification.occluded_count,
                        classification.free_space_count,
                        classification.invalid_count,
                    ),
                    dtype=np.int64,
                )
                if int(values.sum()) != classification.sample_count:
                    raise AssertionError("depth classes do not partition samples")
                if np.any(values > np.iinfo(np.int32).max):
                    raise OverflowError("per-view depth count exceeds int32")
                per_view_counts[pair_index, slot, role_index] = values

    per_view_points = per_view_counts.sum(axis=3, dtype=np.int32)
    per_view_evidence = _fractions_from_counts(
        per_view_counts, per_view_points
    )
    aggregate_counts = per_view_counts.sum(axis=1, dtype=np.int64)
    aggregate_points = aggregate_counts.sum(axis=2, dtype=np.int64)
    aggregate_evidence = _fractions_from_counts(
        aggregate_counts, aggregate_points
    )
    aggregate_views = topk_valid.sum(axis=1, dtype=np.int32)
    delta = aggregate_evidence[:, 1] - aggregate_evidence[:, 0]

    return R4PairedDepthObservation(
        scene_id=frame_bundle.scene_id,
        pose_source=frame_bundle.pose_source,
        proposal_ids=_readonly(ids, np.int64),
        anchor_indices=_readonly(anchor_ids, np.int64),
        used_frame_ids=_readonly(frame_ids, np.int64),
        decoded_frame_ids=_readonly(decoded_order, np.int64),
        topk_frame_ids=_readonly(topk_frame_ids, np.int64),
        topk_view_valid=_readonly(topk_valid, np.bool_),
        topk_projected_area_pixels=_readonly(topk_area, np.float32),
        topk_projected_area_fraction=_readonly(topk_fraction, np.float32),
        per_view_depth_counts=_readonly(per_view_counts, np.int32),
        per_view_depth_evidence=_readonly(per_view_evidence, np.float32),
        per_view_point_count=_readonly(per_view_points, np.int32),
        aggregate_depth_counts=_readonly(aggregate_counts, np.int64),
        aggregate_depth_evidence=_readonly(aggregate_evidence, np.float32),
        aggregate_view_count=_readonly(aggregate_views, np.int32),
        aggregate_point_count=_readonly(aggregate_points, np.int64),
        candidate_minus_anchor_evidence=_readonly(delta, np.float32),
        runtime_s=float(time.perf_counter() - started),
    )


__all__ = [
    "R4_PAIR_ROLES",
    "R4PairedDepthObservation",
    "corners_to_yaw_boxes",
    "observe_r3_replacement_pairs",
]
