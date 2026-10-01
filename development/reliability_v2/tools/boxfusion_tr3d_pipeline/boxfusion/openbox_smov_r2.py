"""Training-free OpenBox/SMOV-inspired geometry observer.

This module is intentionally a *shadow* path.  It extracts a depth-connected
component from the original CuTR 2D crop, keeps a bounded per-track view
memory after native BoxFusion association, and proposes counterfactual OBB
geometry at export time.  In addition to quantile fits, it contains a bounded
adaptation of OpenBox's visibility-based box extension: camera-to-face ray
dot products decide which observed face remains anchored, while a causal
median of the current track's CuTR proposal extents supplies the missing
extent.  It never mutates BoxFusion predictions, scores, or semantics and it
has no checkpoint, label, CLIP, or ground-truth input.

The terminal comparison is leave-one-view-out (LOO): every held-out view is
scored with a box fitted only from the remaining views.  The final all-view
box is emitted only as a diagnostic after the LOO rule succeeds.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, replace
from itertools import combinations
import json
from numbers import Integral, Real
import os
from pathlib import Path
import time
from typing import Iterable, Mapping, Optional, Sequence

import numpy as np

from .moon_qim_lite import CausalFusionIdRegistry
from .mv3dis_depth_lite import derive_committed_track_ids
from .object_memory import deterministic_bounded_sample, voxel_downsample
from .tr3d_r2_geometry import (
    intersect_rays_with_yaw_obb,
    project_yaw_obb_to_depth,
    yaw_obb_corners_world,
)
from .tr3d_r4_smov_observer import corners_to_yaw_boxes


SCHEMA = "boxfusion.openbox_smov_r2_shadow.v2"

DEFAULT_CONFIG = {
    "enabled": False,
    "observer_only": True,
    "pixel_stride": 4,
    "min_depth_m": 0.10,
    "max_depth_m": 8.0,
    "depth_edge_m": 0.15,
    "component_jump_m": 0.15,
    "min_component_pixels": 16,
    "voxel_size_m": 0.05,
    "max_points_per_view": 512,
    "max_points_per_track": 1024,
    "max_validation_rays_per_view": 1024,
    "max_views_per_track": 5,
    "max_tracks": 1024,
    "max_proposals_per_keyframe": 64,
    "min_views": 3,
    "min_points": 192,
    "translation_gap_m": 0.80,
    "rotation_gap_deg": 30.0,
    "lower_quantile": 0.02,
    "upper_quantile": 0.98,
    "face_front_dot": 0.25,
    "face_weak_dot": 0.05,
    "face_band_fraction": 0.10,
    "face_band_max_m": 0.20,
    "min_face_points": 8,
    "min_face_weak_points": 4,
    "face_extension_fraction": 0.25,
    "face_extension_min_m": 0.05,
    "face_extension_max_m": 0.30,
    "max_face_candidates_per_fit": 4,
    "minimum_extent_m": 0.05,
    "max_center_shift_diagonal": 0.60,
    "min_extent_ratio": 0.35,
    "max_extent_ratio": 2.50,
    "depth_margin_m": 0.05,
    "near_clip_m": 1e-3,
    "max_diagnostics": 1024,
    "timing_window": 4096,
    # Writer configuration is consumed only by demo.py.  Keeping it inside the
    # strict section makes misspelled experiment paths fail early.
    "diagnostics": {"root": None},
}


def _strict_bool(name: str, value: object) -> bool:
    if not isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a boolean")
    return bool(value)


def _strict_int(name: str, value: object, minimum: int = 0) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer")
    result = int(value)
    if result < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return result


def _finite(name: str, value: object, minimum: Optional[float] = None) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real number")
    result = float(value)
    if not np.isfinite(result) or (minimum is not None and result < minimum):
        raise ValueError(f"{name} is outside its finite domain")
    return result


def resolve_openbox_smov_r2_config(
    value: Optional[Mapping[str, object]] = None,
) -> dict[str, object]:
    if value is None:
        value = {}
    if not isinstance(value, Mapping):
        raise ValueError("openbox_smov_r2 must be a mapping")
    unknown = sorted(set(value) - set(DEFAULT_CONFIG))
    if unknown:
        raise ValueError("unknown openbox_smov_r2 key(s): " + ", ".join(unknown))
    cfg = dict(DEFAULT_CONFIG)
    cfg.update(value)
    cfg["enabled"] = _strict_bool("enabled", cfg["enabled"])
    cfg["observer_only"] = _strict_bool("observer_only", cfg["observer_only"])
    if cfg["enabled"] and not cfg["observer_only"]:
        raise ValueError("OpenBox-SMOV R2 must remain observer_only")
    diagnostics = cfg["diagnostics"]
    if not isinstance(diagnostics, Mapping):
        raise ValueError("diagnostics must be a mapping")
    unknown_diagnostics = sorted(set(diagnostics) - {"root"})
    if unknown_diagnostics:
        raise ValueError(
            "unknown openbox_smov_r2.diagnostics key(s): "
            + ", ".join(unknown_diagnostics)
        )
    diagnostic_root = diagnostics.get("root")
    if diagnostic_root is not None and (
        not isinstance(diagnostic_root, str) or not diagnostic_root.strip()
    ):
        raise ValueError("diagnostics.root must be null or a non-empty path")
    cfg["diagnostics"] = {"root": diagnostic_root}
    for key in (
        "pixel_stride",
        "min_component_pixels",
        "max_points_per_view",
        "max_points_per_track",
        "max_validation_rays_per_view",
        "max_views_per_track",
        "max_tracks",
        "max_proposals_per_keyframe",
        "min_views",
        "min_points",
        "min_face_points",
        "min_face_weak_points",
        "max_face_candidates_per_fit",
        "max_diagnostics",
        "timing_window",
    ):
        cfg[key] = _strict_int(key, cfg[key], 1)
    for key in (
        "min_depth_m",
        "max_depth_m",
        "depth_edge_m",
        "component_jump_m",
        "voxel_size_m",
        "translation_gap_m",
        "rotation_gap_deg",
        "lower_quantile",
        "upper_quantile",
        "face_front_dot",
        "face_weak_dot",
        "face_band_fraction",
        "face_band_max_m",
        "face_extension_fraction",
        "face_extension_min_m",
        "face_extension_max_m",
        "minimum_extent_m",
        "max_center_shift_diagonal",
        "min_extent_ratio",
        "max_extent_ratio",
        "depth_margin_m",
        "near_clip_m",
    ):
        cfg[key] = _finite(key, cfg[key], 0.0)
    if not cfg["max_depth_m"] > cfg["min_depth_m"]:
        raise ValueError("max_depth_m must exceed min_depth_m")
    if not 0.0 <= cfg["lower_quantile"] < cfg["upper_quantile"] <= 1.0:
        raise ValueError("quantiles must satisfy 0 <= lower < upper <= 1")
    if cfg["min_views"] > cfg["max_views_per_track"]:
        raise ValueError("min_views cannot exceed max_views_per_track")
    if cfg["min_points"] > cfg["max_points_per_track"]:
        raise ValueError("min_points cannot exceed max_points_per_track")
    if cfg["max_points_per_view"] > cfg["max_points_per_track"]:
        raise ValueError("per-view point cap cannot exceed per-track cap")
    if cfg["max_extent_ratio"] < cfg["min_extent_ratio"]:
        raise ValueError("max_extent_ratio must not be below min_extent_ratio")
    if not 0.0 <= cfg["face_weak_dot"] <= cfg["face_front_dot"] <= 1.0:
        raise ValueError("face dot thresholds must satisfy 0 <= weak <= front <= 1")
    if cfg["min_face_points"] < cfg["min_face_weak_points"]:
        raise ValueError("min_face_points must not be below min_face_weak_points")
    if not 0.0 < cfg["face_band_fraction"] < 0.25:
        raise ValueError("face_band_fraction must lie in (0, 0.25)")
    if cfg["face_band_max_m"] <= 0.0:
        raise ValueError("face_band_max_m must be positive")
    if cfg["face_extension_fraction"] <= 0.0:
        raise ValueError("face_extension_fraction must be positive")
    if not cfg["face_extension_max_m"] >= cfg["face_extension_min_m"] > 0.0:
        raise ValueError("face extension bounds must be positive and ordered")
    if cfg["near_clip_m"] <= 0.0 or cfg["voxel_size_m"] <= 0.0:
        raise ValueError("near_clip_m and voxel_size_m must be positive")
    if cfg["enabled"]:
        realtime_caps = (
            "max_points_per_view",
            "max_points_per_track",
            "max_validation_rays_per_view",
            "max_views_per_track",
            "max_tracks",
            "max_proposals_per_keyframe",
            "max_face_candidates_per_fit",
            "max_diagnostics",
            "timing_window",
        )
        exceeded = [
            key
            for key in realtime_caps
            if cfg[key] > DEFAULT_CONFIG[key]
        ]
        if exceeded:
            raise ValueError(
                "enabled R2 realtime caps must not exceed DEFAULT_CONFIG: "
                + ", ".join(exceeded)
            )
        frozen_decision_fields = (
            "pixel_stride",
            "min_depth_m",
            "max_depth_m",
            "depth_edge_m",
            "component_jump_m",
            "min_component_pixels",
            "voxel_size_m",
            "min_views",
            "min_points",
            "translation_gap_m",
            "rotation_gap_deg",
            "lower_quantile",
            "upper_quantile",
            "face_front_dot",
            "face_weak_dot",
            "face_band_fraction",
            "face_band_max_m",
            "min_face_points",
            "min_face_weak_points",
            "face_extension_fraction",
            "face_extension_min_m",
            "face_extension_max_m",
            "max_face_candidates_per_fit",
            "minimum_extent_m",
            "max_center_shift_diagonal",
            "min_extent_ratio",
            "max_extent_ratio",
            "depth_margin_m",
            "near_clip_m",
        )
        changed = [
            key
            for key in frozen_decision_fields
            if cfg[key] != DEFAULT_CONFIG[key]
        ]
        if changed:
            raise ValueError(
                "enabled R2 S0 uses frozen decision fields; changed: "
                + ", ".join(changed)
            )
    return cfg


def _readonly(value: object, dtype: np.dtype) -> np.ndarray:
    result = np.array(value, dtype=dtype, order="C", copy=True)
    result.setflags(write=False)
    return result


def _groups(value: Sequence[Iterable[int]]) -> tuple[tuple[int, ...], ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError("fusion groups must be a sequence")
    result = []
    for raw in value:
        group = tuple(sorted({_strict_int("source id", item, 0) for item in raw}))
        if not group:
            raise ValueError("fusion groups must not contain an empty group")
        result.append(group)
    return tuple(result)


def _ids(value: object, count: Optional[int] = None, name: str = "ids") -> np.ndarray:
    raw = np.asarray(value)
    if raw.ndim != 1 or raw.dtype.kind not in "iu":
        raise ValueError(f"{name} must be a one-dimensional integer array")
    result = raw.astype(np.int64, copy=False)
    if count is not None and len(result) != count:
        raise ValueError(f"{name} has the wrong length")
    if np.any(result < 0) or len(np.unique(result)) != len(result):
        raise ValueError(f"{name} must be unique and nonnegative")
    return np.ascontiguousarray(result)


def _rigid(value: object, name: str = "T_wc") -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (4, 4) or not np.isfinite(result).all():
        raise ValueError(f"{name} must be finite [4,4]")
    if np.max(np.abs(result[3] - [0.0, 0.0, 0.0, 1.0])) > 1e-7:
        raise ValueError(f"{name} must be homogeneous")
    rotation = result[:3, :3]
    if (
        np.max(np.abs(rotation.T @ rotation - np.eye(3))) > 1e-4
        or abs(float(np.linalg.det(rotation)) - 1.0) > 1e-4
    ):
        raise ValueError(f"{name} must contain a proper rotation")
    return np.ascontiguousarray(result)


def _intrinsics(value: object, shape: tuple[int, int]) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape == (4, 4):
        result = result[:3, :3]
    if result.shape != (3, 3) or not np.isfinite(result).all():
        raise ValueError("K must be finite [3,3] or [4,4]")
    if result[0, 0] <= 0.0 or result[1, 1] <= 0.0:
        raise ValueError("K focal lengths must be positive")
    if abs(float(np.linalg.det(result))) <= 1e-12:
        raise ValueError("K must be invertible")
    height, width = shape
    if not (0 <= result[0, 2] < width and 0 <= result[1, 2] < height):
        raise ValueError("K principal point must lie in the depth image")
    return np.ascontiguousarray(result)


def _bbox_iou(left: np.ndarray, right: np.ndarray) -> float:
    x1 = max(float(left[0]), float(right[0]))
    y1 = max(float(left[1]), float(right[1]))
    x2 = min(float(left[2]), float(right[2]))
    y2 = min(float(left[3]), float(right[3]))
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_left = max(0.0, float(left[2] - left[0])) * max(
        0.0, float(left[3] - left[1])
    )
    area_right = max(0.0, float(right[2] - right[0])) * max(
        0.0, float(right[3] - right[1])
    )
    union = area_left + area_right - intersection
    return float(intersection / union) if union > 0.0 else 0.0


@dataclass(frozen=True)
class R2ViewFragment:
    proposal_id: int
    frame_id: int
    score: float
    crop_xyxy_depth: np.ndarray
    image_shape: tuple[int, int]
    intrinsics: np.ndarray
    camera_to_world: np.ndarray
    points_world: np.ndarray
    ray_pixels: np.ndarray
    ray_directions_world: np.ndarray
    ray_depth_m: np.ndarray
    valid_depth_ratio: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "crop_xyxy_depth", _readonly(self.crop_xyxy_depth, np.float32))
        object.__setattr__(self, "intrinsics", _readonly(self.intrinsics, np.float64))
        object.__setattr__(self, "camera_to_world", _readonly(self.camera_to_world, np.float64))
        object.__setattr__(self, "points_world", _readonly(self.points_world, np.float32))
        object.__setattr__(self, "ray_pixels", _readonly(self.ray_pixels, np.int32))
        object.__setattr__(self, "ray_directions_world", _readonly(self.ray_directions_world, np.float32))
        object.__setattr__(self, "ray_depth_m", _readonly(self.ray_depth_m, np.float32))


@dataclass(frozen=True)
class R2KeyframeBatch:
    scene_id: str
    frame_id: int
    proposal_ids: tuple[int, ...]
    previous_fusion_groups: tuple[tuple[int, ...], ...]
    previous_stable_ids: tuple[int, ...]
    fragments: tuple[Optional[R2ViewFragment], ...]
    abstain_reason: Optional[str] = None


@dataclass(frozen=True)
class R2CommitResult:
    frame_id: int
    committed_track_ids: tuple[Optional[int], ...]
    accepted_track_ids: tuple[int, ...]
    current_stable_ids: tuple[int, ...]


@dataclass(frozen=True)
class R2TrackReceipt:
    native_index: int
    stable_id: int
    reason: str
    hypothesis: Optional[str]
    view_frame_ids: tuple[int, ...]
    native_corners: np.ndarray
    candidate_corners: Optional[np.ndarray]
    native_projection_iou: Optional[float]
    candidate_projection_iou: Optional[float]
    native_support: Optional[float]
    candidate_support: Optional[float]
    native_free_space: Optional[float]
    candidate_free_space: Optional[float]
    center_shift_m: Optional[float]
    volume_ratio: Optional[float]
    would_replace: bool
    face_extension_signs: Optional[tuple[int, int]] = None
    face_extension_delta_m: Optional[tuple[float, float]] = None
    face_strong_mask: Optional[tuple[bool, bool, bool, bool]] = None
    face_weak_mask: Optional[tuple[bool, bool, bool, bool]] = None

    def to_json_dict(self) -> dict[str, object]:
        def finite_or_none(value: Optional[float]) -> Optional[float]:
            return None if value is None else float(value)

        return {
            "native_index": int(self.native_index),
            "stable_id": int(self.stable_id),
            "reason": self.reason,
            "hypothesis": self.hypothesis,
            "view_frame_ids": [int(value) for value in self.view_frame_ids],
            "native_corners": np.asarray(self.native_corners).tolist(),
            "candidate_corners": (
                None
                if self.candidate_corners is None
                else np.asarray(self.candidate_corners).tolist()
            ),
            "native_projection_iou": finite_or_none(self.native_projection_iou),
            "candidate_projection_iou": finite_or_none(self.candidate_projection_iou),
            "native_support": finite_or_none(self.native_support),
            "candidate_support": finite_or_none(self.candidate_support),
            "native_free_space": finite_or_none(self.native_free_space),
            "candidate_free_space": finite_or_none(self.candidate_free_space),
            "center_shift_m": finite_or_none(self.center_shift_m),
            "volume_ratio": finite_or_none(self.volume_ratio),
            "would_replace": bool(self.would_replace),
            "face_extension_signs": (
                None
                if self.face_extension_signs is None
                else [int(value) for value in self.face_extension_signs]
            ),
            "face_extension_delta_m": (
                None
                if self.face_extension_delta_m is None
                else [float(value) for value in self.face_extension_delta_m]
            ),
            "face_strong_mask": (
                None
                if self.face_strong_mask is None
                else [bool(value) for value in self.face_strong_mask]
            ),
            "face_weak_mask": (
                None
                if self.face_weak_mask is None
                else [bool(value) for value in self.face_weak_mask]
            ),
        }


@dataclass(frozen=True)
class R2ShadowResult:
    native_corners: np.ndarray
    native_scores: np.ndarray
    stable_ids: np.ndarray
    counterfactual_corners: np.ndarray
    would_replace_mask: np.ndarray
    receipts: tuple[R2TrackReceipt, ...]

    @property
    def summary(self) -> "R2TerminalSummary":
        return R2TerminalSummary(
            native_count=int(len(self.native_corners)),
            counterfactual_count=int(np.count_nonzero(self.would_replace_mask)),
            would_replace_native_indices=tuple(
                int(value) for value in np.flatnonzero(self.would_replace_mask)
            ),
            would_replace_stable_ids=tuple(
                int(value)
                for value in self.stable_ids[self.would_replace_mask]
            ),
        )


@dataclass(frozen=True)
class R2TerminalSummary:
    native_count: int
    counterfactual_count: int
    would_replace_native_indices: tuple[int, ...]
    would_replace_stable_ids: tuple[int, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "native_count": int(self.native_count),
            "counterfactual_count": int(self.counterfactual_count),
            "would_replace_native_indices": [
                int(value) for value in self.would_replace_native_indices
            ],
            "would_replace_stable_ids": [
                int(value) for value in self.would_replace_stable_ids
            ],
            "native_export_mutated": False,
            "counterfactual_geometry_applied": False,
        }


@dataclass
class _TrackState:
    stable_id: int
    views: list[R2ViewFragment]


def _extract_fragment(
    *,
    proposal_id: int,
    frame_id: int,
    score: float,
    box_xyxy: object,
    proposal_image_shape: Sequence[int],
    depth_m: object,
    intrinsics: object,
    camera_to_world: object,
    cfg: Mapping[str, object],
) -> Optional[R2ViewFragment]:
    depth = np.asarray(depth_m)
    if depth.ndim != 2 or depth.dtype.kind not in "iuf" or min(depth.shape) < 1:
        raise ValueError("depth_m must be numeric [H,W]")
    depth = depth.astype(np.float32, copy=False)
    image_shape = tuple(int(value) for value in proposal_image_shape[:2])
    if len(image_shape) != 2 or min(image_shape) < 1:
        raise ValueError("proposal_image_shape must contain positive H,W")
    K = _intrinsics(intrinsics, depth.shape)
    T_wc = _rigid(camera_to_world)
    raw_box = np.asarray(box_xyxy, dtype=np.float64)
    if raw_box.shape != (4,) or not np.isfinite(raw_box).all():
        raise ValueError("box_xyxy must be finite [4]")
    height, width = depth.shape
    scale_x = width / float(image_shape[1])
    scale_y = height / float(image_shape[0])
    box = raw_box * np.asarray([scale_x, scale_y, scale_x, scale_y])
    box[[0, 2]] = np.clip(box[[0, 2]], 0.0, float(width - 1))
    box[[1, 3]] = np.clip(box[[1, 3]], 0.0, float(height - 1))
    if box[2] <= box[0] or box[3] <= box[1]:
        return None

    row_min, row_max = int(np.ceil(box[1])), int(np.floor(box[3]))
    col_min, col_max = int(np.ceil(box[0])), int(np.floor(box[2]))
    if row_min > row_max or col_min > col_max:
        return None
    stride = int(cfg["pixel_stride"])
    rows = np.arange(row_min, row_max + 1, stride, dtype=np.int64)
    cols = np.arange(col_min, col_max + 1, stride, dtype=np.int64)
    maximum_rays = int(cfg["max_validation_rays_per_view"])
    while len(rows) * len(cols) > maximum_rays:
        stride += 1
        rows = np.arange(row_min, row_max + 1, stride, dtype=np.int64)
        cols = np.arange(col_min, col_max + 1, stride, dtype=np.int64)
    if not len(rows) or not len(cols):
        return None
    grid_cols, grid_rows = np.meshgrid(cols, rows)
    sampled = depth[grid_rows, grid_cols]
    usable = (
        np.isfinite(sampled)
        & (sampled >= float(cfg["min_depth_m"]))
        & (sampled <= float(cfg["max_depth_m"]))
    )
    edge = np.zeros_like(usable)
    threshold = float(cfg["depth_edge_m"])
    horizontal = usable[:, :-1] & usable[:, 1:] & (
        np.abs(sampled[:, :-1] - sampled[:, 1:]) > threshold
    )
    vertical = usable[:-1, :] & usable[1:, :] & (
        np.abs(sampled[:-1, :] - sampled[1:, :]) > threshold
    )
    edge[:, :-1] |= horizontal
    edge[:, 1:] |= horizontal
    edge[:-1, :] |= vertical
    edge[1:, :] |= vertical
    component_valid = usable & ~edge
    target_row = int(np.argmin(np.abs(rows - (box[1] + box[3]) * 0.5)))
    target_col = int(np.argmin(np.abs(cols - (box[0] + box[2]) * 0.5)))
    if not component_valid[target_row, target_col]:
        return None

    visited = np.zeros_like(component_valid)
    queue: deque[tuple[int, int]] = deque([(target_row, target_col)])
    visited[target_row, target_col] = True
    component: list[tuple[int, int]] = []
    jump = float(cfg["component_jump_m"])
    while queue:
        row, col = queue.popleft()
        component.append((row, col))
        current_depth = float(sampled[row, col])
        for next_row, next_col in (
            (row - 1, col),
            (row + 1, col),
            (row, col - 1),
            (row, col + 1),
        ):
            if not (
                0 <= next_row < len(rows)
                and 0 <= next_col < len(cols)
                and not visited[next_row, next_col]
                and component_valid[next_row, next_col]
            ):
                continue
            if abs(float(sampled[next_row, next_col]) - current_depth) <= jump:
                # A pixel rejected from one neighbour may still be connected
                # through another neighbour with a closer depth.  Mark it only
                # after the actual edge predicate succeeds.
                visited[next_row, next_col] = True
                queue.append((next_row, next_col))
    if len(component) < int(cfg["min_component_pixels"]):
        return None

    component_indices = np.asarray(component, dtype=np.int64)
    component_rows = rows[component_indices[:, 0]]
    component_cols = cols[component_indices[:, 1]]
    component_depth = sampled[component_indices[:, 0], component_indices[:, 1]]
    pixels = np.column_stack(
        (component_cols, component_rows, np.ones(len(component_rows)))
    ).astype(np.float64)
    rays_camera = pixels @ np.linalg.inv(K).T
    rays_camera /= rays_camera[:, 2:3]
    points_camera = rays_camera * component_depth[:, None]
    points_world = points_camera @ T_wc[:3, :3].T + T_wc[:3, 3]
    points_world = voxel_downsample(points_world, float(cfg["voxel_size_m"]))
    points_world = deterministic_bounded_sample(
        points_world, int(cfg["max_points_per_view"])
    )
    if len(points_world) < int(cfg["min_component_pixels"]):
        return None

    all_pixels = np.column_stack(
        (grid_cols.reshape(-1), grid_rows.reshape(-1), np.ones(grid_rows.size))
    ).astype(np.float64)
    rays_camera_all = all_pixels @ np.linalg.inv(K).T
    rays_camera_all /= rays_camera_all[:, 2:3]
    rays_world = rays_camera_all @ T_wc[:3, :3].T
    valid_ratio = float(np.count_nonzero(usable) / usable.size)
    return R2ViewFragment(
        proposal_id=int(proposal_id),
        frame_id=int(frame_id),
        score=float(score),
        crop_xyxy_depth=box.astype(np.float32),
        image_shape=(height, width),
        intrinsics=K,
        camera_to_world=T_wc,
        points_world=points_world,
        ray_pixels=np.column_stack(
            (grid_cols.reshape(-1), grid_rows.reshape(-1))
        ),
        ray_directions_world=rays_world,
        ray_depth_m=sampled.reshape(-1),
        valid_depth_ratio=valid_ratio,
    )


def _rotation_distance_deg(left: np.ndarray, right: np.ndarray) -> float:
    cosine = (float(np.trace(left.T @ right)) - 1.0) * 0.5
    return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))


def _diverse_component(
    views: Sequence[R2ViewFragment], cfg: Mapping[str, object]
) -> tuple[int, ...]:
    count = len(views)
    adjacency = [set() for _ in range(count)]
    for left in range(count):
        for right in range(left + 1, count):
            translation = float(
                np.linalg.norm(
                    views[left].camera_to_world[:3, 3]
                    - views[right].camera_to_world[:3, 3]
                )
            )
            rotation = _rotation_distance_deg(
                views[left].camera_to_world[:3, :3],
                views[right].camera_to_world[:3, :3],
            )
            if (
                translation >= float(cfg["translation_gap_m"])
                or rotation >= float(cfg["rotation_gap_deg"])
            ):
                adjacency[left].add(right)
                adjacency[right].add(left)
    # V<=5, so exhaustive maximal-clique selection is tiny and prevents a
    # chain A--B--C from being misreported as three mutually diverse views.
    cliques = []
    for size in range(1, count + 1):
        for indices in combinations(range(count), size):
            if all(
                right in adjacency[left]
                for left, right in combinations(indices, 2)
            ):
                cliques.append(tuple(indices))
    cliques.sort(
        key=lambda indices: (
            -len(indices),
            tuple(views[index].frame_id for index in indices),
        )
    )
    return cliques[0] if cliques else ()


def _fit_yaw(
    points: np.ndarray, yaw: float, cfg: Mapping[str, object]
) -> np.ndarray:
    if len(points) < int(cfg["min_points"]):
        raise ValueError("insufficient points for OBB fit")
    cosine, sine = float(np.cos(yaw)), float(np.sin(yaw))
    world_to_local = np.asarray(
        [[cosine, sine, 0.0], [-sine, cosine, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    local = np.asarray(points, dtype=np.float64) @ world_to_local.T
    bounds = np.quantile(
        local,
        [float(cfg["lower_quantile"]), float(cfg["upper_quantile"])],
        axis=0,
    )
    dims = bounds[1] - bounds[0]
    if np.any(dims < float(cfg["minimum_extent_m"])):
        raise ValueError("fitted OBB has a sub-minimum extent")
    local_center = (bounds[0] + bounds[1]) * 0.5
    center = local_center @ world_to_local
    return np.asarray([*center, *dims, yaw], dtype=np.float64)


def _pca_yaw(points: np.ndarray, fallback: float) -> float:
    xy = np.asarray(points, dtype=np.float64)[:, :2]
    covariance = np.cov(xy - xy.mean(axis=0), rowvar=False)
    if covariance.shape != (2, 2) or not np.isfinite(covariance).all():
        return float(fallback)
    values, vectors = np.linalg.eigh(covariance)
    if values[1] <= 1e-10 or values[1] / max(values[0], 1e-10) < 1.05:
        return float(fallback)
    vector = vectors[:, 1]
    if vector[0] < 0.0 or (vector[0] == 0.0 and vector[1] < 0.0):
        vector = -vector
    yaw = float(np.arctan2(vector[1], vector[0]))
    return float((yaw + np.pi / 2.0) % np.pi - np.pi / 2.0)


_FACE_ORDER = ((0, -1), (0, 1), (1, -1), (1, 1))
_FACE_RECIPES = ("base", "face_x", "face_y", "face_xy")
_FACE_BAND_EXTENT_CAP_FRACTION = 0.24


@dataclass(frozen=True)
class _FaceEvidence:
    """Bounded OpenBox-style XY face visibility for one fitted box."""

    extension_signs: tuple[int, int]
    strong_mask: tuple[bool, bool, bool, bool]
    weak_mask: tuple[bool, bool, bool, bool]


@dataclass(frozen=True)
class _FoldCandidate:
    box: np.ndarray
    metric: tuple[float, float, float]
    signs: tuple[int, int]
    deltas: tuple[float, float]


@dataclass(frozen=True)
class _ScoredCandidate:
    improves: bool
    hypothesis: str
    recipe_rank: int
    yaw_rank: int
    box: np.ndarray
    native_metric: np.ndarray
    candidate_metric: np.ndarray
    center_shift_m: float
    volume_ratio: float
    signs: tuple[int, int]
    deltas: tuple[float, float]
    strong_mask: tuple[bool, bool, bool, bool]
    weak_mask: tuple[bool, bool, bool, bool]

    def ordering_key(self) -> tuple[object, ...]:
        return (
            not self.improves,
            -float(self.candidate_metric[0]),
            -float(self.candidate_metric[1]),
            float(self.candidate_metric[2]),
            int(self.recipe_rank),
            int(self.yaw_rank),
            self.hypothesis,
        )


def _face_visibility_xy(
    box: np.ndarray,
    views: Sequence[R2ViewFragment],
    cfg: Mapping[str, object],
) -> _FaceEvidence:
    """Estimate visible XY faces from only the supplied causal views.

    OpenBox evaluates the sign of a LiDAR ray/face-normal dot product.  The
    equivalent face-to-camera convention used here is positive for a front
    facing surface.  Sparse component points inside a narrow face band are a
    cheap surface-presence proxy, replacing OpenBox's offline SDF mesh.
    """

    fitted = np.asarray(box, dtype=np.float64)
    if (
        fitted.shape != (7,)
        or not np.isfinite(fitted).all()
        or np.any(fitted[3:6] <= 0.0)
    ):
        raise ValueError("face visibility requires a finite positive [7] box")
    cosine, sine = float(np.cos(fitted[6])), float(np.sin(fitted[6]))
    world_to_local = np.asarray(
        [[cosine, sine], [-sine, cosine]], dtype=np.float64
    )
    strong = np.zeros(4, dtype=bool)
    weak = np.zeros(4, dtype=bool)
    voxel = float(cfg["voxel_size_m"])
    front_dot = float(cfg["face_front_dot"])
    weak_dot = float(cfg["face_weak_dot"])
    strong_points = int(cfg["min_face_points"])
    weak_points = int(cfg["min_face_weak_points"])

    for view in views:
        points = np.asarray(view.points_world, dtype=np.float64)
        if points.ndim != 2 or points.shape[1:] != (3,) or not len(points):
            continue
        local = np.empty_like(points)
        local[:, :2] = (points[:, :2] - fitted[None, :2]) @ world_to_local.T
        local[:, 2] = points[:, 2] - fitted[2]
        camera = np.asarray(view.camera_to_world[:3, 3], dtype=np.float64)
        for face_index, (axis, side) in enumerate(_FACE_ORDER):
            extent = float(fitted[3 + axis])
            band = min(
                float(cfg["face_band_max_m"]),
                max(voxel, float(cfg["face_band_fraction"]) * extent),
                _FACE_BAND_EXTENT_CAP_FRACTION * extent,
            )
            other = 1 - axis
            near_face = (
                np.abs(local[:, axis] - side * 0.5 * extent) <= band
            )
            inside_other = (
                np.abs(local[:, other]) <= 0.5 * fitted[3 + other] + voxel
            )
            inside_height = np.abs(local[:, 2]) <= 0.5 * fitted[5] + voxel
            count = int(
                np.count_nonzero(near_face & inside_other & inside_height)
            )

            local_face = np.zeros(2, dtype=np.float64)
            local_face[axis] = side * 0.5 * extent
            face_center = np.asarray(
                [*(fitted[:2] + local_face @ world_to_local), fitted[2]],
                dtype=np.float64,
            )
            local_normal = np.zeros(2, dtype=np.float64)
            local_normal[axis] = float(side)
            normal_world_xy = local_normal @ world_to_local
            normal_world = np.asarray(
                [*normal_world_xy, 0.0], dtype=np.float64
            )
            face_to_camera = camera - face_center
            distance = float(np.linalg.norm(face_to_camera))
            if distance <= 1e-9:
                continue
            facing = float(np.dot(normal_world, face_to_camera / distance))
            if facing >= weak_dot and count >= weak_points:
                weak[face_index] = True
            if facing >= front_dot and count >= strong_points:
                strong[face_index] = True

    # Strict thresholds imply this already, but make the serialized invariant
    # explicit even if a future numeric implementation changes internally.
    weak |= strong
    extension_signs = [0, 0]
    for axis, (negative, positive) in enumerate(((0, 1), (2, 3))):
        if float(fitted[3 + axis]) < 4.0 * voxel:
            continue
        if strong[negative] and not weak[positive]:
            extension_signs[axis] = 1
        elif strong[positive] and not weak[negative]:
            extension_signs[axis] = -1
    return _FaceEvidence(
        extension_signs=tuple(int(value) for value in extension_signs),
        strong_mask=tuple(bool(value) for value in strong),
        weak_mask=tuple(bool(value) for value in weak),
    )


def _extend_face_candidate(
    base_box: np.ndarray,
    evidence: _FaceEvidence,
    recipe: str,
    cfg: Mapping[str, object],
) -> Optional[tuple[np.ndarray, tuple[int, int], tuple[float, float]]]:
    """Anchor observed faces and extend only requested unseen XY bounds."""

    if recipe not in _FACE_RECIPES:
        raise ValueError(f"unknown face-extension recipe: {recipe}")
    requested = (
        recipe in {"face_x", "face_xy"},
        recipe in {"face_y", "face_xy"},
    )
    if any(
        requested[axis] and evidence.extension_signs[axis] == 0
        for axis in range(2)
    ):
        return None
    candidate = np.asarray(base_box, dtype=np.float64).copy()
    if candidate.shape != (7,) or not np.isfinite(candidate).all():
        raise ValueError("face extension requires a finite [7] base box")
    signs = [0, 0]
    deltas = [0.0, 0.0]
    local_shift = np.zeros(2, dtype=np.float64)
    for axis in range(2):
        if not requested[axis]:
            continue
        sign = int(evidence.extension_signs[axis])
        delta = float(
            np.clip(
                float(cfg["face_extension_fraction"]) * candidate[3 + axis],
                float(cfg["face_extension_min_m"]),
                float(cfg["face_extension_max_m"]),
            )
        )
        signs[axis] = sign
        deltas[axis] = delta
        candidate[3 + axis] += delta
        local_shift[axis] = 0.5 * sign * delta
    cosine, sine = float(np.cos(candidate[6])), float(np.sin(candidate[6]))
    world_to_local = np.asarray(
        [[cosine, sine], [-sine, cosine]], dtype=np.float64
    )
    candidate[:2] += local_shift @ world_to_local
    return (
        candidate,
        tuple(int(value) for value in signs),
        tuple(float(value) for value in deltas),
    )


def _safety(
    candidate: np.ndarray, native: np.ndarray, cfg: Mapping[str, object]
) -> tuple[bool, float, float]:
    shift = float(np.linalg.norm(candidate[:3] - native[:3]))
    diagonal = float(np.linalg.norm(native[3:6]))
    native_dims = np.asarray([*sorted(native[3:5]), native[5]])
    candidate_dims = np.asarray([*sorted(candidate[3:5]), candidate[5]])
    ratios = candidate_dims / native_dims
    volume_ratio = float(np.prod(candidate[3:6]) / np.prod(native[3:6]))
    valid = (
        shift <= float(cfg["max_center_shift_diagonal"]) * diagonal
        and np.all(ratios >= float(cfg["min_extent_ratio"]))
        and np.all(ratios <= float(cfg["max_extent_ratio"]))
    )
    return bool(valid), shift, volume_ratio


def _evaluate(box: np.ndarray, view: R2ViewFragment, cfg: Mapping[str, object]):
    projection = project_yaw_obb_to_depth(
        box,
        view.intrinsics,
        view.camera_to_world,
        view.image_shape,
        near_clip=float(cfg["near_clip_m"]),
    )
    if projection is None:
        return None
    intersections = intersect_rays_with_yaw_obb(
        view.camera_to_world[:3, 3], view.ray_directions_world, box
    )
    observed = np.asarray(view.ray_depth_m, dtype=np.float64)
    valid_depth = (
        np.isfinite(observed)
        & (observed >= float(cfg["min_depth_m"]))
        & (observed <= float(cfg["max_depth_m"]))
    )
    usable = valid_depth & intersections.intersects
    margin = float(cfg["depth_margin_m"])
    support = usable & (observed >= intersections.t_near - margin) & (
        observed <= intersections.t_far + margin
    )
    free_space = usable & (observed > intersections.t_far + margin)
    classified = int(np.count_nonzero(usable))
    if classified < 1:
        return None
    return (
        _bbox_iou(projection.bbox_xyxy, view.crop_xyxy_depth),
        float(np.count_nonzero(support) / classified),
        float(np.count_nonzero(free_space) / classified),
    )


class OpenBoxSMOVR2Shadow:
    def __init__(self, config: Optional[Mapping[str, object]] = None):
        self.config = resolve_openbox_smov_r2_config(config)
        self.enabled = bool(self.config["enabled"])
        self.observer_only = True
        self._identity = CausalFusionIdRegistry()
        self._scene_id: Optional[str] = None
        self._pending: Optional[R2KeyframeBatch] = None
        self._last_frame_id: Optional[int] = None
        self._tracks: dict[int, _TrackState] = {}
        self._timings: deque[float] = deque(maxlen=int(self.config["timing_window"]))
        self._wrapper_timings: deque[float] = deque(maxlen=int(self.config["timing_window"]))
        self._closed = False
        self._receipts: tuple[R2TrackReceipt, ...] = ()
        self._stats = {
            "keyframes": 0,
            "proposals": 0,
            "proposal_cap_drops": 0,
            "valid_fragments": 0,
            "invalid_fragments": 0,
            "accepted_views": 0,
            "same_frame_duplicates": 0,
            "track_capacity_drops": 0,
            "retired_tracks": 0,
            "prepare_failures": 0,
            "would_replace": 0,
        }

    def _bind(self, scene_id: str) -> None:
        scene_id = str(scene_id)
        if not scene_id:
            raise ValueError("scene_id must not be empty")
        if self._scene_id is None:
            self._scene_id = scene_id
        elif self._scene_id != scene_id:
            raise ValueError("one R2 observer instance cannot span scenes")

    def prepare_keyframe(
        self,
        *,
        scene_id: str,
        frame_id: int,
        proposal_ids: object,
        boxes_xyxy: object,
        proposal_scores: object,
        proposal_image_shape: Sequence[int],
        depth_m: object,
        intrinsics: object,
        camera_to_world: object,
        previous_fusion_groups: Sequence[Iterable[int]],
    ) -> R2KeyframeBatch:
        started = time.perf_counter_ns()
        if not self.enabled:
            raise RuntimeError("OpenBox-SMOV R2 is disabled")
        if self._pending is not None or self._closed:
            raise RuntimeError("R2 prepare/commit transaction is not idle")
        self._bind(scene_id)
        frame_id = _strict_int("frame_id", frame_id, 0)
        if self._last_frame_id is not None and frame_id <= self._last_frame_id:
            raise ValueError("R2 frame_id must be strictly increasing")
        groups = _groups(previous_fusion_groups)
        previous_ids = self._identity.ids_for(groups)
        ids = _ids(proposal_ids, name="proposal_ids")
        boxes = np.asarray(boxes_xyxy, dtype=np.float64)
        scores = np.asarray(proposal_scores, dtype=np.float64)
        if boxes.shape != (len(ids), 4) or scores.shape != (len(ids),):
            raise ValueError("R2 proposal arrays are not row aligned")
        if not np.isfinite(boxes).all() or not np.isfinite(scores).all():
            raise ValueError("R2 proposal arrays must be finite")
        fragments: list[Optional[R2ViewFragment]] = [None] * len(ids)
        order = np.lexsort((ids, -scores))
        selected = order[: int(self.config["max_proposals_per_keyframe"])]
        self._stats["proposal_cap_drops"] += max(0, len(ids) - len(selected))
        for index in selected:
            try:
                fragments[index] = _extract_fragment(
                    proposal_id=int(ids[index]),
                    frame_id=frame_id,
                    score=float(scores[index]),
                    box_xyxy=boxes[index],
                    proposal_image_shape=proposal_image_shape,
                    depth_m=depth_m,
                    intrinsics=intrinsics,
                    camera_to_world=camera_to_world,
                    cfg=self.config,
                )
            except (ValueError, FloatingPointError, np.linalg.LinAlgError):
                fragments[index] = None
        valid = sum(fragment is not None for fragment in fragments)
        self._stats["keyframes"] += 1
        self._stats["proposals"] += len(ids)
        self._stats["valid_fragments"] += valid
        self._stats["invalid_fragments"] += len(ids) - valid
        batch = R2KeyframeBatch(
            scene_id=str(scene_id),
            frame_id=frame_id,
            proposal_ids=tuple(int(value) for value in ids),
            previous_fusion_groups=groups,
            previous_stable_ids=tuple(int(value) for value in previous_ids),
            fragments=tuple(fragments),
        )
        self._pending = batch
        self._last_frame_id = frame_id
        self._timings.append((time.perf_counter_ns() - started) / 1e6)
        return batch

    def prepare_abstain(
        self,
        *,
        scene_id: str,
        frame_id: int,
        proposal_ids: object,
        previous_fusion_groups: Sequence[Iterable[int]],
        reason: str,
    ) -> R2KeyframeBatch:
        if not self.enabled:
            raise RuntimeError("OpenBox-SMOV R2 is disabled")
        if self._pending is not None or self._closed:
            raise RuntimeError("R2 prepare/commit transaction is not idle")
        self._bind(scene_id)
        frame_id = _strict_int("frame_id", frame_id, 0)
        if self._last_frame_id is not None and frame_id <= self._last_frame_id:
            raise ValueError("R2 frame_id must be strictly increasing")
        ids = _ids(proposal_ids, name="proposal_ids")
        groups = _groups(previous_fusion_groups)
        previous = self._identity.ids_for(groups)
        batch = R2KeyframeBatch(
            scene_id=str(scene_id),
            frame_id=frame_id,
            proposal_ids=tuple(int(value) for value in ids),
            previous_fusion_groups=groups,
            previous_stable_ids=tuple(int(value) for value in previous),
            fragments=tuple(None for _ in ids),
            abstain_reason=str(reason),
        )
        self._pending = batch
        self._last_frame_id = frame_id
        self._stats["keyframes"] += 1
        self._stats["proposals"] += len(ids)
        self._stats["invalid_fragments"] += len(ids)
        self._stats["prepare_failures"] += 1
        return batch

    def _select_views(
        self, views: Sequence[R2ViewFragment], limit: int
    ) -> list[R2ViewFragment]:
        per_frame: dict[int, R2ViewFragment] = {}
        for view in views:
            current = per_frame.get(view.frame_id)
            key = (-len(view.points_world), -view.valid_depth_ratio, -view.score, view.proposal_id)
            if current is None:
                per_frame[view.frame_id] = view
            else:
                current_key = (-len(current.points_world), -current.valid_depth_ratio, -current.score, current.proposal_id)
                if key < current_key:
                    per_frame[view.frame_id] = view
        ordered = [per_frame[key] for key in sorted(per_frame)]
        if len(ordered) > limit:
            positions = np.linspace(0, len(ordered) - 1, limit, dtype=np.int64)
            ordered = [ordered[int(index)] for index in positions]
        # The persistent contract is aggregate, not merely a terminal-fit cap.
        # Split the fixed track budget deterministically across retained views.
        if ordered:
            total_budget = int(self.config["max_points_per_track"])
            base = total_budget // len(ordered)
            remainder = total_budget % len(ordered)
            bounded = []
            for index, view in enumerate(ordered):
                allowance = base + (1 if index < remainder else 0)
                points = deterministic_bounded_sample(
                    view.points_world, max(1, allowance)
                )
                bounded.append(replace(view, points_world=points))
            ordered = bounded
        return ordered

    def _merge_state(self, target: int, sources: Sequence[int]) -> None:
        combined = []
        if target in self._tracks:
            combined.extend(self._tracks[target].views)
        for source in sorted(set(int(value) for value in sources)):
            if source == target:
                continue
            state = self._tracks.pop(source, None)
            if state is not None:
                combined.extend(state.views)
        if combined:
            self._tracks[target] = _TrackState(
                target,
                self._select_views(combined, int(self.config["max_views_per_track"])),
            )

    def commit_keyframe(
        self,
        batch: R2KeyframeBatch,
        *,
        current_fusion_groups: Sequence[Iterable[int]],
        association_events: Sequence[Mapping[str, object]] = (),
    ) -> R2CommitResult:
        started = time.perf_counter_ns()
        if not self.enabled:
            raise RuntimeError("OpenBox-SMOV R2 is disabled")
        if batch is not self._pending:
            raise RuntimeError("R2 commit must consume the exact pending batch")
        current_groups = _groups(current_fusion_groups)
        current_ids = self._identity.update(current_groups)
        committed = list(derive_committed_track_ids(
            proposal_ids=np.asarray(batch.proposal_ids, dtype=np.int64),
            current_fusion_groups=current_groups,
            current_stable_ids=current_ids,
            association_events=association_events,
        ))
        # A rare collision can make two native rows share an old member.  The
        # generic connected-component resolver then (correctly) abstains for
        # that shared member, but a new proposal occurring in exactly one row
        # is still row-unambiguous and may commit its own current-frame view.
        for proposal_index, (proposal_id, track_id) in enumerate(
            zip(batch.proposal_ids, committed)
        ):
            if track_id is not None:
                continue
            matching_rows = [
                row_index
                for row_index, group in enumerate(current_groups)
                if int(proposal_id) in group
            ]
            if len(matching_rows) == 1:
                committed[proposal_index] = int(current_ids[matching_rows[0]])
        committed = tuple(committed)
        previous = list(
            zip(batch.previous_fusion_groups, batch.previous_stable_ids)
        )
        previous_member_ids: dict[int, set[int]] = {}
        for previous_group, previous_id in previous:
            for member in previous_group:
                previous_member_ids.setdefault(int(member), set()).add(
                    int(previous_id)
                )
        current_member_ids: dict[int, set[int]] = {}
        for current_group, current_id in zip(current_groups, current_ids):
            for member in current_group:
                current_member_ids.setdefault(int(member), set()).add(
                    int(current_id)
                )
        # A registry-inherited prior ID already owns its history.  It must
        # never be offered to another collision row merely because the two
        # rows share a source member.  Only retired prior IDs are absorbable,
        # and every such ID receives one deterministic target at most.
        retained_previous_ids = {
            int(previous_id)
            for _, previous_id in previous
            if int(previous_id) in {int(value) for value in current_ids}
        }
        lineage_claims: dict[int, list[tuple[object, ...]]] = {}

        def claim_lineage(
            source_id: int,
            target_id: int,
            *,
            event_evidence: bool,
            intersection: int = 0,
            union_size: int = 1,
            target_group: tuple[int, ...] = (),
            target_index: int = 0,
        ) -> None:
            source_id, target_id = int(source_id), int(target_id)
            if source_id in retained_previous_ids or source_id == target_id:
                return
            lineage_claims.setdefault(source_id, []).append(
                (
                    0 if event_evidence else 1,
                    -int(intersection),
                    -(float(intersection) / max(int(union_size), 1)),
                    target_id,
                    target_group,
                    int(target_index),
                )
            )

        previous_sets = [set(group) for group, _ in previous]
        for target_index, (group, target_id) in enumerate(
            zip(current_groups, current_ids)
        ):
            members = set(group)
            for previous_index, (previous_group, previous_id) in enumerate(
                previous
            ):
                intersection = len(members.intersection(previous_sets[previous_index]))
                if intersection:
                    claim_lineage(
                        int(previous_id),
                        int(target_id),
                        event_evidence=False,
                        intersection=intersection,
                        union_size=len(members.union(previous_group)),
                        target_group=group,
                        target_index=target_index,
                    )

        # Native groups stop retaining members at the fixed view cap.  Merge
        # events are therefore the only lineage evidence for an absorbed old
        # group whose members no longer appear in the winner group.
        proposal_to_committed = {
            int(proposal_id): int(track_id)
            for proposal_id, track_id in zip(batch.proposal_ids, committed)
            if track_id is not None
        }
        for event in association_events:
            members = tuple(
                int(member)
                for key in ("winner_members", "loser_members")
                for member in event.get(key, ())
            )
            targets: set[int] = set()
            sources: set[int] = set()
            for member in members:
                targets.update(current_member_ids.get(member, ()))
                sources.update(previous_member_ids.get(member, ()))
                if member in proposal_to_committed:
                    targets.add(proposal_to_committed[member])
            if len(targets) == 1:
                target = next(iter(targets))
                target_index = tuple(int(value) for value in current_ids).index(
                    int(target)
                )
                for source in sources:
                    claim_lineage(
                        source,
                        target,
                        event_evidence=True,
                        target_group=current_groups[target_index],
                        target_index=target_index,
                    )

        absorption_by_target: dict[int, list[int]] = {}
        for source_id, claims in lineage_claims.items():
            target_id = int(min(claims)[3])
            absorption_by_target.setdefault(target_id, []).append(source_id)
        for target_id in sorted(absorption_by_target):
            self._merge_state(
                target_id, tuple(sorted(absorption_by_target[target_id]))
            )

        by_track: dict[int, list[R2ViewFragment]] = {}
        for track_id, fragment in zip(committed, batch.fragments):
            if track_id is not None and fragment is not None:
                by_track.setdefault(int(track_id), []).append(fragment)
        accepted = []
        for track_id in sorted(by_track):
            if track_id not in self._tracks and len(self._tracks) >= int(self.config["max_tracks"]):
                self._stats["track_capacity_drops"] += len(by_track[track_id])
                continue
            choices = self._select_views(by_track[track_id], 1)
            self._stats["same_frame_duplicates"] += len(by_track[track_id]) - len(choices)
            existing = self._tracks.get(track_id, _TrackState(track_id, []))
            existing.views = self._select_views(
                [*existing.views, *choices], int(self.config["max_views_per_track"])
            )
            self._tracks[track_id] = existing
            accepted.append(track_id)
            self._stats["accepted_views"] += len(choices)

        active = {int(value) for value in current_ids}
        retired = [track_id for track_id in self._tracks if track_id not in active]
        for track_id in retired:
            self._tracks.pop(track_id, None)
        self._stats["retired_tracks"] += len(retired)
        self._pending = None
        self._timings.append((time.perf_counter_ns() - started) / 1e6)
        return R2CommitResult(
            frame_id=batch.frame_id,
            committed_track_ids=tuple(committed),
            accepted_track_ids=tuple(accepted),
            current_stable_ids=tuple(int(value) for value in current_ids),
        )

    def current_stable_ids(
        self, current_fusion_groups: Sequence[Iterable[int]]
    ) -> np.ndarray:
        return self._identity.ids_for(_groups(current_fusion_groups))

    def record_wrapper_timing(self, milliseconds: object) -> None:
        value = _finite("wrapper timing", milliseconds, 0.0)
        self._wrapper_timings.append(value)

    def _receipt(
        self, native_index: int, stable_id: int, corners: np.ndarray
    ) -> R2TrackReceipt:
        native_box = corners_to_yaw_boxes(corners[None, ...])[0]
        state = self._tracks.get(stable_id)
        empty = dict(
            native_index=native_index,
            stable_id=stable_id,
            hypothesis=None,
            view_frame_ids=(),
            native_corners=_readonly(corners, np.float32),
            candidate_corners=None,
            native_projection_iou=None,
            candidate_projection_iou=None,
            native_support=None,
            candidate_support=None,
            native_free_space=None,
            candidate_free_space=None,
            center_shift_m=None,
            volume_ratio=None,
            would_replace=False,
        )
        if state is None:
            return R2TrackReceipt(reason="no_track_memory", **empty)
        component = _diverse_component(state.views, self.config)
        if len(component) < int(self.config["min_views"]):
            return R2TrackReceipt(
                reason="insufficient_pose_diverse_views",
                **{**empty, "view_frame_ids": tuple(view.frame_id for view in state.views)},
            )
        views = [state.views[index] for index in component]
        if sum(len(view.points_world) for view in views) < int(self.config["min_points"]):
            return R2TrackReceipt(
                reason="insufficient_points",
                **{**empty, "view_frame_ids": tuple(view.frame_id for view in views)},
            )

        fold_inputs = []
        for held_out, view in enumerate(views):
            training_views = tuple(
                other
                for index, other in enumerate(views)
                if index != held_out
            )
            training_points = np.concatenate(
                [other.points_world for other in training_views], axis=0
            )
            training_points = deterministic_bounded_sample(
                voxel_downsample(
                    training_points, float(self.config["voxel_size_m"])
                ),
                int(self.config["max_points_per_track"]),
            )
            native_metric = _evaluate(native_box, view, self.config)
            if native_metric is None:
                fold_inputs = []
                break
            fold_inputs.append(
                (training_views, training_points, view, native_metric)
            )
        if not fold_inputs:
            return R2TrackReceipt(
                reason="loo_evidence_unavailable",
                **{**empty, "view_frame_ids": tuple(view.frame_id for view in views)},
            )

        all_points = np.concatenate(
            [view.points_world for view in views], axis=0
        )
        all_points = deterministic_bounded_sample(
            voxel_downsample(
                all_points, float(self.config["voxel_size_m"])
            ),
            int(self.config["max_points_per_track"]),
        )
        recipe_limit = int(self.config["max_face_candidates_per_fit"])
        allowed_recipes = _FACE_RECIPES[:recipe_limit]
        recipe_rank = {recipe: index for index, recipe in enumerate(_FACE_RECIPES)}
        candidates: list[_ScoredCandidate] = []
        for yaw_rank, yaw_hypothesis in enumerate(
            ("native_yaw_quantile", "pca_yaw_quantile")
        ):
            fold_candidates: list[dict[str, _FoldCandidate]] = []
            valid_hypothesis = True
            for training_views, training_points, held_out_view, _ in fold_inputs:
                try:
                    yaw = float(native_box[6])
                    if yaw_hypothesis == "pca_yaw_quantile":
                        yaw = _pca_yaw(training_points, yaw)
                    base_box = _fit_yaw(training_points, yaw, self.config)
                    evidence = _face_visibility_xy(
                        base_box, training_views, self.config
                    )
                except (ValueError, np.linalg.LinAlgError):
                    valid_hypothesis = False
                    break
                per_recipe: dict[str, _FoldCandidate] = {}
                for recipe in allowed_recipes:
                    extended = _extend_face_candidate(
                        base_box, evidence, recipe, self.config
                    )
                    if extended is None:
                        continue
                    fold_box, signs, deltas = extended
                    safe, _, _ = _safety(
                        fold_box, native_box, self.config
                    )
                    metric = _evaluate(
                        fold_box, held_out_view, self.config
                    )
                    if safe and metric is not None:
                        per_recipe[recipe] = _FoldCandidate(
                            box=fold_box,
                            metric=metric,
                            signs=signs,
                            deltas=deltas,
                        )
                fold_candidates.append(per_recipe)
            if not valid_hypothesis or len(fold_candidates) != len(views):
                continue

            common_recipes = set(allowed_recipes)
            for per_recipe in fold_candidates:
                common_recipes.intersection_update(per_recipe)
            try:
                final_yaw = float(native_box[6])
                if yaw_hypothesis == "pca_yaw_quantile":
                    final_yaw = _pca_yaw(all_points, final_yaw)
                final_base = _fit_yaw(all_points, final_yaw, self.config)
                final_evidence = _face_visibility_xy(
                    final_base, views, self.config
                )
            except (ValueError, np.linalg.LinAlgError):
                continue

            for recipe in allowed_recipes:
                if recipe not in common_recipes:
                    continue
                fold_signs = {
                    per_recipe[recipe].signs
                    for per_recipe in fold_candidates
                }
                if len(fold_signs) != 1:
                    continue
                extended = _extend_face_candidate(
                    final_base, final_evidence, recipe, self.config
                )
                if extended is None:
                    continue
                final_box, final_signs, final_deltas = extended
                if final_signs != next(iter(fold_signs)):
                    continue
                try:
                    safe, shift, volume_ratio = _safety(
                        final_box, native_box, self.config
                    )
                except (ValueError, np.linalg.LinAlgError):
                    continue
                if not safe:
                    continue
                native_agg = np.median(
                    np.asarray([row[3] for row in fold_inputs]), axis=0
                )
                candidate_agg = np.median(
                    np.asarray(
                        [
                            per_recipe[recipe].metric
                            for per_recipe in fold_candidates
                        ]
                    ),
                    axis=0,
                )
                improves = (
                    candidate_agg[0] >= native_agg[0]
                    and candidate_agg[1] >= native_agg[1]
                    and candidate_agg[2] <= native_agg[2]
                    and (
                        candidate_agg[0] > native_agg[0] + 1e-9
                        or candidate_agg[1] > native_agg[1] + 1e-9
                        or candidate_agg[2] < native_agg[2] - 1e-9
                    )
                )
                candidates.append(
                    _ScoredCandidate(
                        improves=bool(improves),
                        hypothesis=f"{yaw_hypothesis}+{recipe}",
                        recipe_rank=recipe_rank[recipe],
                        yaw_rank=yaw_rank,
                        box=final_box,
                        native_metric=native_agg,
                        candidate_metric=candidate_agg,
                        center_shift_m=float(shift),
                        volume_ratio=float(volume_ratio),
                        signs=final_signs,
                        deltas=final_deltas,
                        strong_mask=final_evidence.strong_mask,
                        weak_mask=final_evidence.weak_mask,
                    )
                )
        if not candidates:
            return R2TrackReceipt(
                reason="loo_evidence_unavailable",
                **{**empty, "view_frame_ids": tuple(view.frame_id for view in views)},
            )
        best = min(candidates, key=lambda row: row.ordering_key())
        candidate_corners = yaw_obb_corners_world(best.box).astype(np.float32)
        return R2TrackReceipt(
            native_index=native_index,
            stable_id=stable_id,
            reason="loo_improved" if best.improves else "loo_not_improved",
            hypothesis=best.hypothesis,
            view_frame_ids=tuple(view.frame_id for view in views),
            native_corners=_readonly(corners, np.float32),
            candidate_corners=_readonly(candidate_corners, np.float32),
            native_projection_iou=float(best.native_metric[0]),
            candidate_projection_iou=float(best.candidate_metric[0]),
            native_support=float(best.native_metric[1]),
            candidate_support=float(best.candidate_metric[1]),
            native_free_space=float(best.native_metric[2]),
            candidate_free_space=float(best.candidate_metric[2]),
            center_shift_m=float(best.center_shift_m),
            volume_ratio=float(best.volume_ratio),
            would_replace=bool(best.improves),
            face_extension_signs=best.signs,
            face_extension_delta_m=best.deltas,
            face_strong_mask=best.strong_mask,
            face_weak_mask=best.weak_mask,
        )

    def finalize_shadow(
        self,
        *,
        native_corners: object,
        native_scores: object,
        stable_ids: object,
        scene_id: Optional[str] = None,
    ) -> R2ShadowResult:
        started = time.perf_counter_ns()
        if not self.enabled or self._closed or self._pending is not None:
            raise RuntimeError("R2 finalize requires an enabled, idle, open observer")
        if scene_id is not None and str(scene_id) != self._scene_id:
            raise ValueError("R2 finalize scene_id does not match the live scene")
        corners_input = np.asarray(native_corners)
        scores_input = np.asarray(native_scores)
        if corners_input.ndim != 3 or corners_input.shape[1:] != (8, 3):
            raise ValueError("native_corners must have shape [N,8,3]")
        if scores_input.shape != (len(corners_input),):
            raise ValueError("native_scores must align with native_corners")
        ids = _ids(stable_ids, len(corners_input), "stable_ids")
        if not np.isfinite(corners_input).all() or not np.isfinite(scores_input).all():
            raise ValueError("native predictions must be finite")
        original_corners = np.array(corners_input, order="C", copy=True)
        original_scores = np.array(scores_input, order="C", copy=True)
        counterfactual = np.array(corners_input, dtype=np.float32, order="C", copy=True)
        receipts = []
        replace = np.zeros(len(corners_input), dtype=bool)
        for index, (stable_id, corners) in enumerate(zip(ids, corners_input)):
            try:
                receipt = self._receipt(index, int(stable_id), np.asarray(corners))
            except (ValueError, FloatingPointError, np.linalg.LinAlgError):
                receipt = R2TrackReceipt(
                    native_index=index,
                    stable_id=int(stable_id),
                    reason="geometry_exception_abstain",
                    hypothesis=None,
                    view_frame_ids=(),
                    native_corners=_readonly(corners, np.float32),
                    candidate_corners=None,
                    native_projection_iou=None,
                    candidate_projection_iou=None,
                    native_support=None,
                    candidate_support=None,
                    native_free_space=None,
                    candidate_free_space=None,
                    center_shift_m=None,
                    volume_ratio=None,
                    would_replace=False,
                )
            receipts.append(receipt)
            if receipt.would_replace and receipt.candidate_corners is not None:
                replace[index] = True
                counterfactual[index] = receipt.candidate_corners
        if not np.array_equal(corners_input, original_corners) or not np.array_equal(scores_input, original_scores):
            raise RuntimeError("R2 observer mutated native inputs")
        self._closed = True
        self._receipts = tuple(receipts[: int(self.config["max_diagnostics"])])
        self._stats["would_replace"] = int(np.count_nonzero(replace))
        self._timings.append((time.perf_counter_ns() - started) / 1e6)
        return R2ShadowResult(
            native_corners=_readonly(original_corners, corners_input.dtype),
            native_scores=_readonly(original_scores, scores_input.dtype),
            stable_ids=_readonly(ids, np.int64),
            counterfactual_corners=_readonly(counterfactual, np.float32),
            would_replace_mask=_readonly(replace, np.bool_),
            receipts=tuple(receipts),
        )

    @staticmethod
    def _timing(values: Sequence[float]) -> dict[str, float]:
        if not values:
            return {"mean_ms": 0.0, "p95_ms": 0.0, "max_ms": 0.0}
        array = np.asarray(values, dtype=np.float64)
        return {
            "mean_ms": float(array.mean()),
            "p95_ms": float(np.quantile(array, 0.95)),
            "max_ms": float(array.max()),
        }

    def summary(self) -> dict[str, object]:
        return {
            "schema": SCHEMA,
            "enabled": self.enabled,
            "observer_only": True,
            "active_authorized": False,
            "training_invoked": False,
            "online_learning": False,
            "ground_truth_access": False,
            "clip_access": False,
            "semantic_access": False,
            "checkpoint_access": False,
            "future_frame_access": False,
            "full_scene_reconstruction": False,
            "native_outputs_mutated": False,
            "counterfactual_geometry_applied": False,
            "closed": self._closed,
            "scene_id": self._scene_id,
            "effective_config": dict(self.config),
            **{key: int(value) for key, value in self._stats.items()},
            "active_tracks_at_close": len(self._tracks),
            "core_timing": self._timing(tuple(self._timings)),
            "wrapper_timing": self._timing(tuple(self._wrapper_timings)),
            "receipts": [receipt.to_json_dict() for receipt in self._receipts],
        }


def build_openbox_smov_r2(cfg: Mapping[str, object]) -> OpenBoxSMOVR2Shadow:
    if not isinstance(cfg, Mapping):
        raise ValueError("application config must be a mapping")
    section = cfg.get("openbox_smov_r2", {})
    return OpenBoxSMOVR2Shadow(section)


def save_r2_shadow_sidecar_create_only(
    result: R2ShadowResult, output_path: object
) -> Path:
    """Write one immutable counterfactual NPZ without touching predictions."""

    if not isinstance(result, R2ShadowResult):
        raise ValueError("result must be an R2ShadowResult")
    path = Path(output_path)
    if path.suffix != ".npz":
        raise ValueError("R2 sidecar path must end in .npz")
    path.parent.mkdir(parents=True, exist_ok=True)
    receipt_json = json.dumps(
        [receipt.to_json_dict() for receipt in result.receipts],
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    try:
        descriptor = os.open(
            path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o644,
        )
    except FileExistsError as error:
        raise FileExistsError(f"R2 sidecar already exists: {path}") from error
    try:
        with os.fdopen(descriptor, "wb") as handle:
            np.savez_compressed(
                handle,
                schema=np.asarray(SCHEMA),
                native_corners=np.asarray(result.native_corners),
                native_scores=np.asarray(result.native_scores),
                stable_ids=np.asarray(result.stable_ids),
                counterfactual_corners=np.asarray(
                    result.counterfactual_corners
                ),
                would_replace_mask=np.asarray(result.would_replace_mask),
                receipts_json=np.frombuffer(receipt_json, dtype=np.uint8),
            )
    except BaseException:
        # A failed exclusive write is never presented as a valid artifact.
        try:
            path.unlink()
        except FileNotFoundError:
            pass
        raise
    return path


__all__ = [
    "DEFAULT_CONFIG",
    "OpenBoxSMOVR2Shadow",
    "R2CommitResult",
    "R2KeyframeBatch",
    "R2ShadowResult",
    "R2TerminalSummary",
    "R2TrackReceipt",
    "R2ViewFragment",
    "SCHEMA",
    "build_openbox_smov_r2",
    "resolve_openbox_smov_r2_config",
    "save_r2_shadow_sidecar_create_only",
]
