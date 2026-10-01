"""Training-free RGB-D geometry for the N0 paper100 shadow route.

The module lifts already-frozen masks, constructs a 5 cm cross-view surface
consensus, rejects voxels contradicted by measured free space, and exposes two
identity-preserving geometries: a robust yaw-only OBB and the best sealed
per-view Boxer OBB.  It has no model, labels, predictions, evaluator, or GT.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping, Sequence

import cv2
import numpy as np

from boxfusion import fastsam_openbox_f3_shadow as f3
from boxfusion.target_masklift import robust_yaw_obb


SCHEMA = "boxfusion.sam2_tsdf_mv3dis_shadow.v1"
PROTOCOL_ID = "N0-RGBD-TSDF-MV3DIS-BOXER-DUAL-OBB-SHADOW-V1"
IMAGE_HEIGHT = 480
IMAGE_WIDTH = 640
VOXEL_SIZE_M = 0.05
MIN_DEPTH_M = 0.10
MAX_DEPTH_M = 6.0
DEPTH_EDGE_JUMP_M = 0.15
MASK_EROSION_PX = 1
MAX_POINTS_PER_VIEW = 2_048
MIN_VOXELS_PER_VIEW = 16
MIN_SURFACE_VIEWS = 2
SURFACE_TRUNCATION_M = 0.10
MAX_FREE_SPACE_VIOLATIONS = 0
MIN_CONSENSUS_VOXELS = 16
MAX_CONSENSUS_VOXELS = 8_192
ROBUST_QUANTILES = (0.02, 0.98)
MIN_OBB_EXTENT_M = 0.02
BOX_CONTAINMENT_MARGIN_M = 0.05
ROBUST_WIN_MARGIN = 0.03


class N0GeometryError(ValueError):
    """One input or output violated the frozen geometry contract."""


def _readonly(value: object, dtype: np.dtype, shape: tuple[int, ...] | None = None) -> np.ndarray:
    try:
        array = np.ascontiguousarray(np.asarray(value, dtype=dtype))
    except (TypeError, ValueError, OverflowError) as error:
        raise N0GeometryError("invalid numeric array") from error
    if shape is not None and array.shape != shape:
        raise N0GeometryError(f"array must have shape {shape}")
    if np.issubdtype(array.dtype, np.floating) and not np.isfinite(array).all():
        raise N0GeometryError("array must be finite")
    return np.frombuffer(array.tobytes(), dtype=dtype).reshape(array.shape)


@dataclass(frozen=True)
class LiftedMaskView:
    source_id: str
    frame_id: int
    mask: np.ndarray
    depth_m: np.ndarray
    intrinsic: np.ndarray
    camera_to_world: np.ndarray
    points_world: np.ndarray
    voxel_keys: np.ndarray
    support_pixel_count: int
    uncapped_voxel_count: int

    def __post_init__(self) -> None:
        if not isinstance(self.source_id, str) or not self.source_id:
            raise N0GeometryError("source_id must be non-empty")
        if isinstance(self.frame_id, bool) or not isinstance(self.frame_id, int):
            raise N0GeometryError("frame_id must be int")
        mask = _readonly(self.mask, np.bool_, (IMAGE_HEIGHT, IMAGE_WIDTH))
        depth = _readonly(self.depth_m, np.float64, (IMAGE_HEIGHT, IMAGE_WIDTH))
        intrinsic = _readonly(self.intrinsic, np.float64, (3, 3))
        pose = _readonly(self.camera_to_world, np.float64, (4, 4))
        points = _readonly(self.points_world, np.float64)
        keys = _readonly(self.voxel_keys, np.int64)
        if points.ndim != 2 or points.shape[1:] != (3,) or keys.shape != points.shape:
            raise N0GeometryError("points_world and voxel_keys must be aligned [N,3]")
        if not MIN_VOXELS_PER_VIEW <= len(points) <= MAX_POINTS_PER_VIEW:
            raise N0GeometryError("lifted view voxel count differs")
        if self.support_pixel_count < len(points) or self.uncapped_voxel_count < len(points):
            raise N0GeometryError("lifted view counts are inconsistent")
        object.__setattr__(self, "mask", mask)
        object.__setattr__(self, "depth_m", depth)
        object.__setattr__(self, "intrinsic", intrinsic)
        object.__setattr__(self, "camera_to_world", pose)
        object.__setattr__(self, "points_world", points)
        object.__setattr__(self, "voxel_keys", keys)


def _depth_edge_mask(depth: np.ndarray, valid: np.ndarray) -> np.ndarray:
    edge = np.zeros_like(valid)
    horizontal = valid[:, :-1] & valid[:, 1:] & (
        np.abs(depth[:, :-1] - depth[:, 1:]) > DEPTH_EDGE_JUMP_M
    )
    vertical = valid[:-1, :] & valid[1:, :] & (
        np.abs(depth[:-1, :] - depth[1:, :]) > DEPTH_EDGE_JUMP_M
    )
    edge[:, :-1] |= horizontal
    edge[:, 1:] |= horizontal
    edge[:-1, :] |= vertical
    edge[1:, :] |= vertical
    return edge


def lift_mask_view(
    *,
    source_id: str,
    frame_id: int,
    mask: object,
    depth_m: object,
    intrinsic: object,
    camera_to_world: object,
) -> LiftedMaskView | None:
    """Lift one current mask after the frozen boundary/depth-edge cleanup."""

    mask_array = np.asarray(mask)
    depth = np.asarray(depth_m, dtype=np.float64)
    matrix = np.asarray(intrinsic, dtype=np.float64)
    pose = np.asarray(camera_to_world, dtype=np.float64)
    if mask_array.shape != (IMAGE_HEIGHT, IMAGE_WIDTH):
        raise N0GeometryError("mask shape differs")
    if np.any((mask_array != 0) & (mask_array != 1)):
        raise N0GeometryError("mask must be binary")
    mask_array = mask_array.astype(np.bool_, copy=False)
    if depth.shape != (IMAGE_HEIGHT, IMAGE_WIDTH) or not np.isfinite(depth).all():
        raise N0GeometryError("depth must be finite [480,640]")
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all() or matrix[0, 0] <= 0 or matrix[1, 1] <= 0:
        raise N0GeometryError("intrinsic is invalid")
    if pose.shape != (4, 4) or not np.isfinite(pose).all():
        raise N0GeometryError("camera_to_world is invalid")
    try:
        inverse_intrinsic = np.linalg.inv(matrix)
    except np.linalg.LinAlgError as error:
        raise N0GeometryError("intrinsic is singular") from error

    valid = (depth >= MIN_DEPTH_M) & (depth <= MAX_DEPTH_M)
    edges = _depth_edge_mask(depth, valid)
    interior = cv2.erode(
        mask_array.astype(np.uint8),
        np.ones((3, 3), dtype=np.uint8),
        iterations=MASK_EROSION_PX,
        borderType=cv2.BORDER_CONSTANT,
        borderValue=0,
    ).astype(np.bool_)
    rows, columns = np.nonzero(interior & valid & ~edges)
    support_count = int(len(rows))
    if support_count < MIN_VOXELS_PER_VIEW:
        return None
    pixels = np.column_stack(
        (columns.astype(np.float64), rows.astype(np.float64), np.ones(support_count))
    )
    rays = pixels @ inverse_intrinsic.T
    rays /= rays[:, 2:3]
    camera_points = rays * depth[rows, columns, None]
    world_points = camera_points @ pose[:3, :3].T + pose[:3, 3]
    keys = np.floor(world_points / VOXEL_SIZE_M).astype(np.int64)
    unique_keys, first = np.unique(keys, axis=0, return_index=True)
    uncapped = int(len(unique_keys))
    if uncapped < MIN_VOXELS_PER_VIEW:
        return None
    representatives = world_points[first]
    if uncapped > MAX_POINTS_PER_VIEW:
        indices = (
            np.arange(MAX_POINTS_PER_VIEW, dtype=np.int64) * (uncapped - 1)
        ) // (MAX_POINTS_PER_VIEW - 1)
        unique_keys = unique_keys[indices]
        representatives = representatives[indices]
    return LiftedMaskView(
        source_id=source_id,
        frame_id=frame_id,
        mask=mask_array,
        depth_m=depth,
        intrinsic=matrix,
        camera_to_world=pose,
        points_world=representatives,
        voxel_keys=unique_keys,
        support_pixel_count=support_count,
        uncapped_voxel_count=uncapped,
    )


def _surface_free_space_evidence(
    centers_world: np.ndarray,
    view: LiftedMaskView,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    world_to_camera = np.linalg.inv(view.camera_to_world)
    camera = centers_world @ world_to_camera[:3, :3].T + world_to_camera[:3, 3]
    z = camera[:, 2]
    front = z > 1.0e-4
    safe_z = np.where(front, z, 1.0)
    u = view.intrinsic[0, 0] * camera[:, 0] / safe_z + view.intrinsic[0, 2]
    v = view.intrinsic[1, 1] * camera[:, 1] / safe_z + view.intrinsic[1, 2]
    columns = np.rint(u).astype(np.int64)
    rows = np.rint(v).astype(np.int64)
    inside = front & (columns >= 0) & (columns < IMAGE_WIDTH) & (rows >= 0) & (rows < IMAGE_HEIGHT)
    measured = np.zeros(len(centers_world), dtype=np.float64)
    mask_support = np.zeros(len(centers_world), dtype=np.bool_)
    if np.any(inside):
        selected = np.flatnonzero(inside)
        measured[selected] = view.depth_m[rows[selected], columns[selected]]
        mask_support[selected] = view.mask[rows[selected], columns[selected]]
    valid = inside & (measured >= MIN_DEPTH_M) & (measured <= MAX_DEPTH_M)
    residual = measured - z
    surface = valid & mask_support & (np.abs(residual) <= SURFACE_TRUNCATION_M)
    free_space = valid & (residual > SURFACE_TRUNCATION_M)
    occluded = valid & (residual < -SURFACE_TRUNCATION_M)
    return surface, free_space, occluded


def _pca_yaw(points: np.ndarray) -> float:
    centered = points[:, :2] - np.median(points[:, :2], axis=0, keepdims=True)
    covariance = centered.T @ centered / max(len(centered), 1)
    values, vectors = np.linalg.eigh(covariance)
    axis = vectors[:, int(np.argmax(values))]
    if axis[0] < 0.0 or (axis[0] == 0.0 and axis[1] < 0.0):
        axis = -axis
    return float(math.atan2(float(axis[1]), float(axis[0])))


def _rectangle_mask_iou(corners_world: np.ndarray, view: LiftedMaskView) -> float:
    world_to_camera = np.linalg.inv(view.camera_to_world)
    camera = corners_world @ world_to_camera[:3, :3].T + world_to_camera[:3, 3]
    if np.any(camera[:, 2] <= 1.0e-4):
        return 0.0
    uvw = camera @ view.intrinsic.T
    uv = uvw[:, :2] / uvw[:, 2:3]
    lower = np.floor(uv.min(axis=0)).astype(np.int64)
    upper = np.ceil(uv.max(axis=0)).astype(np.int64)
    x1 = int(np.clip(lower[0], 0, IMAGE_WIDTH - 1))
    y1 = int(np.clip(lower[1], 0, IMAGE_HEIGHT - 1))
    x2 = int(np.clip(upper[0], 0, IMAGE_WIDTH - 1))
    y2 = int(np.clip(upper[1], 0, IMAGE_HEIGHT - 1))
    if x2 < x1 or y2 < y1:
        return 0.0
    intersection = int(np.count_nonzero(view.mask[y1 : y2 + 1, x1 : x2 + 1]))
    rectangle = (x2 - x1 + 1) * (y2 - y1 + 1)
    mask_area = int(np.count_nonzero(view.mask))
    union = rectangle + mask_area - intersection
    return 0.0 if union <= 0 else float(intersection / union)


def _obb_containment(
    points: np.ndarray,
    *,
    center: np.ndarray,
    extent: np.ndarray,
    rotation: np.ndarray,
) -> float:
    local = (points - center[None, :]) @ rotation
    contained = np.all(
        np.abs(local) <= extent[None, :] * 0.5 + BOX_CONTAINMENT_MARGIN_M,
        axis=1,
    )
    return float(np.mean(contained)) if len(contained) else 0.0


def _hypothesis_metrics(
    views: Sequence[LiftedMaskView],
    points: np.ndarray,
    *,
    corners: np.ndarray,
    center: np.ndarray,
    extent: np.ndarray,
    rotation: np.ndarray,
) -> dict[str, Any]:
    projection = [_rectangle_mask_iou(corners, view) for view in views]
    containment = [
        _obb_containment(
            view.points_world,
            center=center,
            extent=extent,
            rotation=rotation,
        )
        for view in views
    ]
    projection_median = float(np.median(projection))
    containment_median = float(np.median(containment))
    return {
        "projection_mask_ious": projection,
        "projection_mask_iou_median": projection_median,
        "point_containments": containment,
        "point_containment_median": containment_median,
        "self_validation_score": 0.5 * projection_median + 0.5 * containment_median,
    }


def _boxer_hypothesis(
    views: Sequence[LiftedMaskView],
    consensus_points: np.ndarray,
    boxer_by_source: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    candidates: list[dict[str, Any]] = []
    for ordinal, view in enumerate(views):
        raw = boxer_by_source.get(view.source_id)
        if not isinstance(raw, Mapping) or raw.get("valid") is not True:
            continue
        try:
            center = np.asarray(raw["world_center"], dtype=np.float64)
            extent = np.asarray(raw["local_extent"], dtype=np.float64)
            rotation = np.asarray(raw["world_rotation"], dtype=np.float64)
            corners = np.asarray(raw["world_corners"], dtype=np.float64)
        except (KeyError, TypeError, ValueError) as error:
            raise N0GeometryError(f"invalid sealed Boxer hypothesis: {view.source_id}") from error
        if (
            center.shape != (3,)
            or extent.shape != (3,)
            or rotation.shape != (3, 3)
            or corners.shape != (8, 3)
            or not all(np.isfinite(item).all() for item in (center, extent, rotation, corners))
            or np.any(extent <= 0.0)
        ):
            raise N0GeometryError(f"malformed sealed Boxer geometry: {view.source_id}")
        metrics = _hypothesis_metrics(
            views,
            consensus_points,
            corners=corners,
            center=center,
            extent=extent,
            rotation=rotation,
        )
        candidates.append(
            {
                "source_id": view.source_id,
                "source_ordinal": ordinal,
                "world_center": center.tolist(),
                "local_extent": extent.tolist(),
                "world_rotation": rotation.tolist(),
                "world_corners": corners.tolist(),
                "world_aabb": [*corners.min(axis=0).tolist(), *corners.max(axis=0).tolist()],
                "metrics": metrics,
            }
        )
    if not candidates:
        return {"valid": False, "reason": "no_valid_boxer_view"}
    winner = sorted(
        candidates,
        key=lambda row: (
            -float(row["metrics"]["self_validation_score"]),
            int(row["source_ordinal"]),
            str(row["source_id"]),
        ),
    )[0]
    return {"valid": True, "reason": "best_past_only_self_validation", **winner}


def build_track_geometry(
    *,
    views: Sequence[LiftedMaskView],
    boxer_by_source: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Build the frozen TSDF/MV3DIS-lite consensus and dual OBB shadow."""

    if not 3 <= len(views) <= 5:
        raise N0GeometryError("track geometry requires three through five views")
    if len({view.frame_id for view in views}) != len(views):
        raise N0GeometryError("track geometry requires distinct ordered frames")
    if list(view.frame_id for view in views) != sorted(view.frame_id for view in views):
        raise N0GeometryError("track views must be chronological")
    union = np.unique(np.concatenate([view.voxel_keys for view in views], axis=0), axis=0)
    neighbourhood_support = np.zeros(len(union), dtype=np.int16)
    for view in views:
        neighbourhood_support += f3._view_neighbourhood_support_tree(
            union, view.voxel_keys
        ).astype(np.int16)
    multi_view = union[neighbourhood_support >= MIN_SURFACE_VIEWS]
    if not len(multi_view):
        return {
            "valid": False,
            "reason": "no_two_view_surface_consensus",
            "union_voxel_count": int(len(union)),
            "two_view_voxel_count": 0,
        }
    centers = (multi_view.astype(np.float64) + 0.5) * VOXEL_SIZE_M
    surface_count = np.zeros(len(centers), dtype=np.int16)
    free_count = np.zeros(len(centers), dtype=np.int16)
    occluded_count = np.zeros(len(centers), dtype=np.int16)
    for view in views:
        surface, free, occluded = _surface_free_space_evidence(centers, view)
        surface_count += surface.astype(np.int16)
        free_count += free.astype(np.int16)
        occluded_count += occluded.astype(np.int16)
    keep = (surface_count >= MIN_SURFACE_VIEWS) & (
        free_count <= MAX_FREE_SPACE_VIOLATIONS
    )
    retained_keys = multi_view[keep]
    before_cap = int(len(retained_keys))
    if before_cap < MIN_CONSENSUS_VOXELS:
        return {
            "valid": False,
            "reason": "too_few_mv3dis_consistent_voxels",
            "union_voxel_count": int(len(union)),
            "two_view_voxel_count": int(len(multi_view)),
            "mv3dis_voxel_count": before_cap,
            "free_space_rejected_voxel_count": int(np.count_nonzero(free_count > 0)),
        }
    if before_cap > MAX_CONSENSUS_VOXELS:
        indices = (
            np.arange(MAX_CONSENSUS_VOXELS, dtype=np.int64) * (before_cap - 1)
        ) // (MAX_CONSENSUS_VOXELS - 1)
        retained_keys = retained_keys[indices]
    retained_points = (retained_keys.astype(np.float64) + 0.5) * VOXEL_SIZE_M
    yaw = _pca_yaw(retained_points)
    robust = robust_yaw_obb(retained_points, yaw_rad=yaw)
    robust_rotation = np.asarray(
        [
            [math.cos(robust.yaw_rad), -math.sin(robust.yaw_rad), 0.0],
            [math.sin(robust.yaw_rad), math.cos(robust.yaw_rad), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    robust_valid = bool(np.all(robust.extent >= MIN_OBB_EXTENT_M - 1.0e-12))
    robust_metrics = _hypothesis_metrics(
        views,
        retained_points,
        corners=robust.corners,
        center=robust.center,
        extent=robust.extent,
        rotation=robust_rotation,
    )
    robust_row = {
        "valid": robust_valid,
        "reason": "valid" if robust_valid else "extent_below_0.02m",
        "world_center": robust.center.tolist(),
        "local_extent": robust.extent.tolist(),
        "yaw_rad": robust.yaw_rad,
        "world_rotation": robust_rotation.tolist(),
        "world_corners": robust.corners.tolist(),
        "world_aabb": [*robust.aabb_lower.tolist(), *robust.aabb_upper.tolist()],
        "metrics": robust_metrics,
    }
    boxer = _boxer_hypothesis(views, retained_points, boxer_by_source)
    robust_score = float(robust_metrics["self_validation_score"])
    boxer_score = (
        float(boxer["metrics"]["self_validation_score"])
        if boxer.get("valid") is True
        else -math.inf
    )
    if robust_valid and robust_score >= boxer_score + ROBUST_WIN_MARGIN:
        chosen, reason = "NROBUST", "robust_wins_by_at_least_0.03"
    elif boxer.get("valid") is True:
        chosen, reason = "HBEST", "boxer_valid_robust_not_better_by_0.03"
    elif robust_valid:
        chosen, reason = "NROBUST", "boxer_invalid_robust_valid"
    else:
        chosen, reason = None, "neither_dual_hypothesis_valid"
    return {
        "valid": chosen is not None,
        "reason": "valid" if chosen is not None else reason,
        "union_voxel_count": int(len(union)),
        "two_view_voxel_count": int(len(multi_view)),
        "mv3dis_voxel_count_before_cap": before_cap,
        "mv3dis_voxel_count": int(len(retained_keys)),
        "free_space_rejected_voxel_count": int(np.count_nonzero(free_count > 0)),
        "surface_support_histogram": {
            str(index): int(np.count_nonzero(surface_count == index))
            for index in range(len(views) + 1)
        },
        "hypotheses": {"NROBUST": robust_row, "HBEST": boxer},
        "selector": {
            "chosen": chosen,
            "reason": reason,
            "ground_truth": False,
            "score_margin": ROBUST_WIN_MARGIN,
        },
    }


def policy_receipt() -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "protocol_id": PROTOCOL_ID,
        "voxel_size_m": VOXEL_SIZE_M,
        "depth_range_m": [MIN_DEPTH_M, MAX_DEPTH_M],
        "depth_edge_jump_m": DEPTH_EDGE_JUMP_M,
        "mask_erosion_px": MASK_EROSION_PX,
        "max_points_per_view": MAX_POINTS_PER_VIEW,
        "minimum_surface_views": MIN_SURFACE_VIEWS,
        "surface_truncation_m": SURFACE_TRUNCATION_M,
        "maximum_free_space_violations": MAX_FREE_SPACE_VIOLATIONS,
        "minimum_consensus_voxels": MIN_CONSENSUS_VOXELS,
        "maximum_consensus_voxels": MAX_CONSENSUS_VOXELS,
        "robust_quantiles": list(ROBUST_QUANTILES),
        "robust_win_margin": ROBUST_WIN_MARGIN,
        "shadow_only": True,
        "birth": False,
        "native_output_mutation": False,
        "training": False,
        "online_learning": False,
        "ground_truth": False,
        "semantics": False,
    }
