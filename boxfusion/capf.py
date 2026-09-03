"""Counterfactual asymmetric parameter fusion for native 3D boxes.

CAPF is a training-free, geometry-only refinement.  A candidate changes one
directed face of the native fused box while keeping the other five faces and
the full SO(3) rotation fixed.  The candidate is generated from one selected
view and is accepted only when the remaining selected views improve under
RGB-D surface/free-space evidence.  Rejected refinements return the native
anchor exactly.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch


RAY_INVALID = 0
RAY_SURFACE = 1
RAY_OCCLUDED = 2
RAY_FREE_SPACE = 3


DEFAULT_CAPF_CONFIG = {
    "enabled": False,
    "min_views": 3,
    "max_ray_samples": 96,
    "min_valid_depth_samples": 12,
    "min_surface_rays": 4,
    "min_reference_rays": 8,
    "min_depth_m": 0.10,
    "max_depth_m": 8.0,
    "surface_band_m": 0.08,
    "occlusion_margin_m": 0.05,
    "free_space_margin_m": 0.05,
    "surface_residual_clip_m": 0.25,
    "surface_weight": 0.75,
    "free_space_weight": 0.25,
    "min_loss_improvement": 0.005,
    "max_heldout_regression": 0.0,
    "min_surface_retention": 0.90,
    "max_free_space_increase": 0.0,
    "min_face_visibility_cosine": 0.15,
    "min_candidate_shift_m": 0.01,
    "max_face_shift_m": 0.20,
    "max_face_shift_ratio": 0.25,
    "min_extent_m": 0.05,
    "max_accepted_faces": 3,
}


def resolve_capf_config(box_fusion_cfg: Mapping) -> dict:
    """Resolve and validate the sole ``box_fusion.capf`` configuration."""

    raw = box_fusion_cfg.get("capf", {})
    if raw is None:
        raw = {}
    if not hasattr(raw, "get"):
        raise ValueError("box_fusion.capf must be a mapping")
    raw = dict(raw)
    oracle_shadow = bool(raw.pop("oracle_shadow", False))
    oracle_diagnostics_dir = raw.pop("oracle_diagnostics_dir", None)
    if oracle_diagnostics_dir is not None:
        if not isinstance(oracle_diagnostics_dir, str) or not oracle_diagnostics_dir:
            raise ValueError("capf.oracle_diagnostics_dir must be a non-empty path")
        oracle_diagnostics_dir = os.path.abspath(oracle_diagnostics_dir)
    if oracle_shadow and oracle_diagnostics_dir is None:
        raise ValueError(
            "capf.oracle_shadow requires capf.oracle_diagnostics_dir"
        )
    if oracle_diagnostics_dir is not None and not oracle_shadow:
        raise ValueError(
            "capf.oracle_diagnostics_dir is diagnostic-only and requires oracle_shadow"
        )

    cfg = dict(DEFAULT_CAPF_CONFIG)
    cfg.update(raw)
    cfg["enabled"] = bool(cfg["enabled"])
    cfg["oracle_shadow"] = oracle_shadow
    cfg["oracle_diagnostics_dir"] = oracle_diagnostics_dir

    integer_keys = (
        "min_views",
        "max_ray_samples",
        "min_valid_depth_samples",
        "min_surface_rays",
        "min_reference_rays",
        "max_accepted_faces",
    )
    for key in integer_keys:
        cfg[key] = int(cfg[key])
        if cfg[key] < 1:
            raise ValueError(f"capf.{key} must be positive")
    if cfg["min_views"] < 3:
        raise ValueError("CAPF requires one fit view and at least two held-out views")
    if cfg["min_valid_depth_samples"] > cfg["max_ray_samples"]:
        raise ValueError(
            "capf.min_valid_depth_samples cannot exceed max_ray_samples"
        )
    if cfg["min_surface_rays"] > cfg["max_ray_samples"]:
        raise ValueError("capf.min_surface_rays cannot exceed max_ray_samples")
    if cfg["min_reference_rays"] > cfg["max_ray_samples"]:
        raise ValueError("capf.min_reference_rays cannot exceed max_ray_samples")

    float_keys = tuple(
        key
        for key in DEFAULT_CAPF_CONFIG
        if key not in {"enabled", *integer_keys}
    )
    for key in float_keys:
        cfg[key] = float(cfg[key])
        if cfg[key] < 0.0:
            raise ValueError(f"capf.{key} must be non-negative")
    positive_keys = (
        "min_depth_m",
        "max_depth_m",
        "surface_band_m",
        "occlusion_margin_m",
        "free_space_margin_m",
        "surface_residual_clip_m",
        "min_candidate_shift_m",
        "max_face_shift_m",
        "max_face_shift_ratio",
        "min_extent_m",
    )
    for key in positive_keys:
        if cfg[key] <= 0.0:
            raise ValueError(f"capf.{key} must be positive")
    if cfg["max_depth_m"] <= cfg["min_depth_m"]:
        raise ValueError("capf depth interval is empty")
    if not np.isclose(
        cfg["surface_weight"] + cfg["free_space_weight"],
        1.0,
        atol=1.0e-8,
    ):
        raise ValueError("CAPF evidence weights must sum to one")
    for key in (
        "min_surface_retention",
        "max_free_space_increase",
        "min_face_visibility_cosine",
    ):
        if not 0.0 <= cfg[key] <= 1.0:
            raise ValueError(f"capf.{key} must be in [0, 1]")
    return cfg


def _as_numpy(value, dtype=np.float64):
    if torch.is_tensor(value):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=dtype)


def box_to_local_faces(box_xyzlhw: np.ndarray) -> np.ndarray:
    """Return ``[[x-,x+],[y-,y+],[z-,z+]]`` in the box local frame."""

    box = np.asarray(box_xyzlhw)
    if box.shape != (6,) or not np.all(np.isfinite(box)):
        raise ValueError("box_xyzlhw must be a finite vector of shape [6]")
    if np.any(box[3:] <= 0.0):
        raise ValueError("box extents must be positive")
    half = np.asarray(box[3:], dtype=np.float64) * 0.5
    return np.stack((-half, half), axis=1)


def local_faces_to_box(
    reference_box_xyzlhw: np.ndarray,
    reference_rotation: np.ndarray,
    faces: np.ndarray,
) -> np.ndarray:
    """Convert directed local faces back to world center and ``[l,h,w]``."""

    reference = np.asarray(reference_box_xyzlhw, dtype=np.float64)
    rotation = np.asarray(reference_rotation, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.float64)
    if reference.shape != (6,) or rotation.shape != (3, 3):
        raise ValueError("invalid CAPF reference geometry")
    if faces.shape != (3, 2) or not np.all(np.isfinite(faces)):
        raise ValueError("CAPF faces must have shape [3,2]")
    extents = faces[:, 1] - faces[:, 0]
    if np.any(extents <= 0.0):
        raise ValueError("CAPF faces define a non-positive extent")
    local_center = 0.5 * (faces[:, 0] + faces[:, 1])
    world_center = reference[:3] + rotation @ local_center
    return np.concatenate((world_center, extents))


def _box_corners(box_xyzlhw: np.ndarray, rotation: np.ndarray) -> np.ndarray:
    signs = np.asarray(
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
    box = np.asarray(box_xyzlhw, dtype=np.float64)
    rotation = np.asarray(rotation, dtype=np.float64)
    return box[:3] + (signs * (box[3:] * 0.5)) @ rotation.T


def classify_rays(
    box_xyzlhw: np.ndarray,
    rotation: np.ndarray,
    camera_origin_world: np.ndarray,
    surface_points_world: np.ndarray,
    surface_valid: np.ndarray,
    *,
    surface_band_m: float,
    occlusion_margin_m: float,
    free_space_margin_m: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Classify fixed RGB-D rays against an oriented box.

    ``RAY_OCCLUDED`` and ``RAY_INVALID`` are unknown evidence.  Only a box
    entering the ray before the measured first surface is free-space-negative.
    The returned arrays are labels, ray-box entry ranges, and measured ranges.
    """

    box = np.asarray(box_xyzlhw, dtype=np.float64)
    rotation = np.asarray(rotation, dtype=np.float64)
    origin = np.asarray(camera_origin_world, dtype=np.float64)
    points = np.asarray(surface_points_world, dtype=np.float64)
    valid = np.asarray(surface_valid, dtype=bool).reshape(-1)
    if (
        box.shape != (6,)
        or rotation.shape != (3, 3)
        or origin.shape != (3,)
        or points.ndim != 2
        or points.shape[1] != 3
        or points.shape[0] != valid.shape[0]
    ):
        raise ValueError("invalid CAPF ray-classification input shapes")

    labels = np.full(points.shape[0], RAY_INVALID, dtype=np.int8)
    entries = np.full(points.shape[0], np.nan, dtype=np.float64)
    measured = np.linalg.norm(points - origin[None], axis=1)
    valid &= np.isfinite(points).all(axis=1)
    valid &= np.isfinite(measured) & (measured > 1.0e-8)
    valid &= np.isfinite(box).all() & np.isfinite(rotation).all()
    valid &= np.all(box[3:] > 0.0)
    if not np.any(valid):
        return labels, entries, measured

    directions = np.zeros_like(points)
    directions[valid] = (points[valid] - origin) / measured[valid, None]
    local_origin = (origin - box[:3]) @ rotation
    local_directions = directions @ rotation
    half = box[3:] * 0.5

    count = points.shape[0]
    near = np.full(count, -np.inf, dtype=np.float64)
    far = np.full(count, np.inf, dtype=np.float64)
    hit = valid.copy()
    for axis in range(3):
        direction = local_directions[:, axis]
        parallel = np.abs(direction) < 1.0e-10
        hit &= ~(parallel & (np.abs(local_origin[axis]) > half[axis]))
        nonparallel = ~parallel
        first = np.full(count, -np.inf, dtype=np.float64)
        second = np.full(count, np.inf, dtype=np.float64)
        first[nonparallel] = (
            -half[axis] - local_origin[axis]
        ) / direction[nonparallel]
        second[nonparallel] = (
            half[axis] - local_origin[axis]
        ) / direction[nonparallel]
        axis_near = np.minimum(first, second)
        axis_far = np.maximum(first, second)
        near = np.maximum(near, axis_near)
        far = np.minimum(far, axis_far)
    near = np.maximum(near, 0.0)
    hit &= far >= near
    hit &= far >= 0.0
    entries[hit] = near[hit]

    delta = measured - entries
    surface = hit & (np.abs(delta) <= float(surface_band_m))
    occluded = hit & ~surface & (delta < -float(occlusion_margin_m))
    free_space = hit & ~surface & (delta > float(free_space_margin_m))
    # The narrow undecided interval is surface-compatible, never negative.
    compatible = hit & ~(occluded | free_space)
    labels[compatible] = RAY_SURFACE
    labels[occluded] = RAY_OCCLUDED
    labels[free_space] = RAY_FREE_SPACE
    return labels, entries, measured


@dataclass(frozen=True)
class CAPFFaceUpdate:
    face_index: int
    source_view: int
    heldout_views: tuple[int, ...]
    face_value: float
    median_loss_improvement: float
    worst_loss_improvement: float


@dataclass(frozen=True)
class CAPFResult:
    accepted: bool
    reason: str
    box_xyzlhw: np.ndarray
    rotation: np.ndarray
    attempted_candidates: int
    updates: tuple[CAPFFaceUpdate, ...]


@dataclass(frozen=True)
class _ViewComparison:
    baseline_loss: float
    candidate_loss: float
    surface_retention: float
    baseline_free_ratio: float
    candidate_free_ratio: float


class CAPF:
    """Face-wise counterfactual refinement with strict native rollback."""

    EVIDENCE_FIELDS = (
        "capf_surface_points_world",
        "capf_surface_valid",
    )

    def __init__(self, box_fusion_cfg: Mapping):
        self.cfg = resolve_capf_config(box_fusion_cfg)
        self.enabled = self.cfg["enabled"]
        self.oracle_shadow = self.cfg["oracle_shadow"]
        self.oracle_diagnostics_dir = self.cfg["oracle_diagnostics_dir"]
        self._oracle_records: dict[tuple[int, ...], dict] = {}
        self._oracle_record_sequence = 0
        self.stats = {
            "observation_batches": 0,
            "observations": 0,
            "fusion_attempts": 0,
            "fusion_accepted": 0,
            "candidates": 0,
            "accepted_faces": 0,
            "reasons": {},
            "improvements": [],
        }

    @staticmethod
    def _track_key(values: Sequence[int]) -> tuple[int, ...]:
        key = tuple(int(value) for value in values)
        if not key or any(value < 0 for value in key):
            raise ValueError("CAPF oracle track keys must be non-empty and non-negative")
        return key

    def _record_oracle_snapshot(
        self,
        *,
        track_key: Sequence[int] | None,
        anchor_box: np.ndarray,
        rotation: np.ndarray,
        proposals: Sequence[tuple[int, int, float]],
        selected_box: np.ndarray,
        selected_updates: Sequence[CAPFFaceUpdate],
    ) -> None:
        """Keep a GT-free candidate bank for the terminal oracle audit."""

        if not self.oracle_shadow or track_key is None:
            return
        key = self._track_key(track_key)
        self._oracle_record_sequence += 1
        anchor = np.asarray(anchor_box, dtype=np.float64)
        rotation = np.asarray(rotation, dtype=np.float64)
        anchor_faces = box_to_local_faces(anchor)
        options: list[dict] = []
        seen: set[tuple[int, int]] = set()
        for face_index, source_view, proposed_value in proposals:
            bounded = self._bounded_candidate(
                anchor,
                rotation,
                anchor_faces,
                int(face_index),
                float(proposed_value),
            )
            if bounded is None:
                continue
            _, face_value = bounded
            # Micrometre quantisation only removes numerically identical view
            # proposals; it does not change candidate geometry.
            identity = (int(face_index), int(np.rint(face_value * 1.0e6)))
            if identity in seen:
                continue
            seen.add(identity)
            options.append(
                {
                    "face_index": int(face_index),
                    "source_view": int(source_view),
                    "proposed_value": float(proposed_value),
                    "face_value": float(face_value),
                }
            )
        options.sort(
            key=lambda row: (
                row["face_index"], row["face_value"], row["source_view"]
            )
        )
        self._oracle_records[key] = {
            "record_sequence": self._oracle_record_sequence,
            "track_key": list(key),
            "anchor_box_xyzlhw": anchor.tolist(),
            "anchor_rotation": rotation.tolist(),
            "anchor_faces": anchor_faces.tolist(),
            "face_options": options,
            "proxy_selected_box_xyzlhw": np.asarray(
                selected_box, dtype=np.float64
            ).tolist(),
            "proxy_selected_updates": [
                {
                    "face_index": int(update.face_index),
                    "source_view": int(update.source_view),
                    "heldout_views": [int(value) for value in update.heldout_views],
                    "face_value": float(update.face_value),
                    "median_loss_improvement": float(
                        update.median_loss_improvement
                    ),
                    "worst_loss_improvement": float(
                        update.worst_loss_improvement
                    ),
                }
                for update in selected_updates
            ],
        }

    def write_oracle_diagnostics(
        self,
        *,
        scene_id: str,
        final_track_keys: Sequence[Sequence[int]],
        final_corners_world,
        final_scores,
    ) -> Path | None:
        """Write one immutable, GT-free terminal candidate sidecar."""

        if not self.oracle_shadow:
            return None
        if not isinstance(scene_id, str) or not scene_id:
            raise ValueError("CAPF oracle scene_id must be non-empty")
        corners = np.asarray(final_corners_world, dtype=np.float64)
        scores = np.asarray(final_scores, dtype=np.float64).reshape(-1)
        if corners.shape != (len(final_track_keys), 8, 3):
            raise ValueError("CAPF oracle terminal corners do not match track keys")
        if scores.shape != (len(final_track_keys),):
            raise ValueError("CAPF oracle terminal scores do not match track keys")
        if not np.isfinite(corners).all() or not np.isfinite(scores).all():
            raise ValueError("CAPF oracle terminal rows must be finite")

        rows = []
        matched_records = 0
        for row_index, raw_key in enumerate(final_track_keys):
            key = self._track_key(raw_key)
            exact_record = self._oracle_records.get(key)
            candidate_records = []
            if exact_record is not None:
                candidate_records.append(exact_record)
            final_members = set(key)
            candidate_records.extend(
                record
                for record_key, record in self._oracle_records.items()
                if record is not exact_record
                and set(record_key).issubset(final_members)
            )
            candidate_records.sort(
                key=lambda item: (
                    len(item["track_key"]), item["record_sequence"]
                ),
                reverse=True,
            )
            record = None
            anchor_aabb_error = None
            for proposed_record in candidate_records:
                anchor_corners = _box_corners(
                    np.asarray(
                        proposed_record["anchor_box_xyzlhw"], dtype=np.float64
                    ),
                    np.asarray(
                        proposed_record["anchor_rotation"], dtype=np.float64
                    ),
                )
                anchor_aabb = np.concatenate(
                    (anchor_corners.min(axis=0), anchor_corners.max(axis=0))
                )
                terminal_aabb = np.concatenate(
                    (corners[row_index].min(axis=0), corners[row_index].max(axis=0))
                )
                proposed_error = float(
                    np.max(np.abs(anchor_aabb - terminal_aabb))
                )
                if proposed_error <= 1.0e-4:
                    record = proposed_record
                    anchor_aabb_error = proposed_error
                    matched_records += 1
                    break
                if anchor_aabb_error is None or proposed_error < anchor_aabb_error:
                    anchor_aabb_error = proposed_error
            rows.append(
                {
                    "row_index": int(row_index),
                    "track_key": list(key),
                    "native_corners_world": corners[row_index].tolist(),
                    "score": float(scores[row_index]),
                    "snapshot_anchor_aabb_max_error_m": anchor_aabb_error,
                    "candidate_snapshot": record,
                }
            )

        output_dir = Path(self.oracle_diagnostics_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output = output_dir / f"{scene_id}_capf_candidates.json"
        if output.exists() or output.is_symlink():
            raise FileExistsError(f"refusing to overwrite CAPF oracle sidecar: {output}")
        payload = {
            "schema": "boxfusion.capf_candidate_shadow.v1",
            "scene_id": scene_id,
            "gt_access": False,
            "oracle_shadow": True,
            "online_writeback": False,
            "candidate_generation": {
                "min_candidate_shift_m": self.cfg["min_candidate_shift_m"],
                "max_face_shift_m": self.cfg["max_face_shift_m"],
                "max_face_shift_ratio": self.cfg["max_face_shift_ratio"],
                "min_extent_m": self.cfg["min_extent_m"],
                "max_accepted_faces": self.cfg["max_accepted_faces"],
            },
            "final_row_count": len(rows),
            "candidate_snapshot_count": matched_records,
            "rows": rows,
        }
        temporary = output.with_name(f".{output.name}.tmp.{os.getpid()}")
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, sort_keys=True, separators=(",", ":"))
            handle.write("\n")
        os.replace(temporary, output)
        return output

    def attach_observations(
        self,
        instances,
        *,
        depth_m,
        intrinsics,
        image_height: int,
        image_width: int,
        camera_to_world,
    ) -> None:
        """Attach fixed-size world surface samples without changing native rows."""

        if not self.enabled or len(instances) == 0:
            return
        depth = np.squeeze(_as_numpy(depth_m, np.float64))
        intrinsic = _as_numpy(intrinsics, np.float64)[:3, :3]
        pose = _as_numpy(camera_to_world, np.float64)
        boxes = _as_numpy(instances.pred_boxes, np.float64)
        if depth.ndim != 2:
            raise ValueError(f"CAPF expects a 2D depth map, got {depth.shape}")
        if intrinsic.shape != (3, 3) or pose.shape != (4, 4):
            raise ValueError("CAPF expects [3,3] intrinsics and a [4,4] C2W pose")
        if image_height <= 0 or image_width <= 0:
            raise ValueError("CAPF image dimensions must be positive")

        sample_count = self.cfg["max_ray_samples"]
        points = np.zeros(
            (len(instances), sample_count, 3), dtype=np.float32
        )
        point_valid = np.zeros((len(instances), sample_count), dtype=bool)
        depth_height, depth_width = depth.shape
        scale_x = float(depth_width) / float(image_width)
        scale_y = float(depth_height) / float(image_height)

        for index, raw_box in enumerate(boxes):
            x_low, x_high = sorted((raw_box[0], raw_box[2]))
            y_low, y_high = sorted((raw_box[1], raw_box[3]))
            x_low = int(np.floor(np.clip(x_low * scale_x, 0, depth_width)))
            x_high = int(np.ceil(np.clip(x_high * scale_x, 0, depth_width)))
            y_low = int(np.floor(np.clip(y_low * scale_y, 0, depth_height)))
            y_high = int(np.ceil(np.clip(y_high * scale_y, 0, depth_height)))
            if x_high <= x_low or y_high <= y_low:
                continue

            aspect = max(float(x_high - x_low) / max(y_high - y_low, 1), 1e-3)
            columns = max(1, int(np.ceil(np.sqrt(sample_count * aspect))))
            rows = max(1, int(np.ceil(sample_count / columns)))
            x_values = np.linspace(x_low, x_high - 1, columns)
            y_values = np.linspace(y_low, y_high - 1, rows)
            grid_x, grid_y = np.meshgrid(x_values, y_values)
            grid_x = np.rint(grid_x).astype(np.int64).reshape(-1)
            grid_y = np.rint(grid_y).astype(np.int64).reshape(-1)
            if grid_x.size > sample_count:
                keep = np.linspace(
                    0, grid_x.size - 1, sample_count, dtype=np.int64
                )
                grid_x = grid_x[keep]
                grid_y = grid_y[keep]

            values = depth[grid_y, grid_x]
            valid = (
                np.isfinite(values)
                & (values >= self.cfg["min_depth_m"])
                & (values <= self.cfg["max_depth_m"])
            )
            if not np.any(valid):
                continue
            u = grid_x[valid].astype(np.float64)
            v = grid_y[valid].astype(np.float64)
            z = values[valid]
            camera_points = np.stack(
                (
                    (u - intrinsic[0, 2]) * z / intrinsic[0, 0],
                    (v - intrinsic[1, 2]) * z / intrinsic[1, 1],
                    z,
                ),
                axis=1,
            )
            world_points = camera_points @ pose[:3, :3].T + pose[:3, 3]
            finite = np.isfinite(world_points).all(axis=1)
            world_points = world_points[finite][:sample_count]
            count = world_points.shape[0]
            points[index, :count] = world_points.astype(np.float32)
            point_valid[index, :count] = True

        instances.capf_surface_points_world = torch.from_numpy(points)
        instances.capf_surface_valid = torch.from_numpy(point_valid)
        self.stats["observation_batches"] += 1
        self.stats["observations"] += len(instances)

    def _source_face_candidates(
        self,
        reference_box: np.ndarray,
        reference_rotation: np.ndarray,
        observation_box: np.ndarray,
        observation_rotation: np.ndarray,
        camera_position: np.ndarray,
        source_view: int,
    ) -> list[tuple[int, int, float]]:
        corners_world = _box_corners(observation_box, observation_rotation)
        corners_local = (
            corners_world - reference_box[:3][None]
        ) @ reference_rotation
        camera_local = (
            np.asarray(camera_position, dtype=np.float64) - reference_box[:3]
        ) @ reference_rotation
        norm = max(float(np.linalg.norm(camera_local)), 1.0e-12)
        direction = camera_local / norm
        candidates = []
        for axis in range(3):
            if abs(direction[axis]) < self.cfg["min_face_visibility_cosine"]:
                continue
            side = 1 if direction[axis] >= 0.0 else 0
            face_index = 2 * axis + side
            value = (
                float(np.max(corners_local[:, axis]))
                if side == 1
                else float(np.min(corners_local[:, axis]))
            )
            candidates.append((face_index, source_view, value))
        return candidates

    def _bounded_candidate(
        self,
        reference_box: np.ndarray,
        reference_rotation: np.ndarray,
        current_faces: np.ndarray,
        face_index: int,
        proposed_value: float,
    ) -> tuple[np.ndarray, float] | None:
        axis = face_index // 2
        side = face_index % 2
        old_value = float(current_faces[axis, side])
        maximum_shift = min(
            self.cfg["max_face_shift_m"],
            self.cfg["max_face_shift_ratio"] * reference_box[3 + axis],
        )
        value = float(
            np.clip(proposed_value, old_value - maximum_shift, old_value + maximum_shift)
        )
        if abs(value - old_value) < self.cfg["min_candidate_shift_m"]:
            return None
        opposite = float(current_faces[axis, 1 - side])
        if side == 0:
            value = min(value, opposite - self.cfg["min_extent_m"])
        else:
            value = max(value, opposite + self.cfg["min_extent_m"])
        if abs(value - old_value) < self.cfg["min_candidate_shift_m"]:
            return None
        faces = current_faces.copy()
        faces[axis, side] = value
        candidate = local_faces_to_box(reference_box, reference_rotation, faces)
        if not np.all(np.isfinite(candidate)) or np.any(
            candidate[3:] < self.cfg["min_extent_m"]
        ):
            return None
        return candidate, value

    def _compare_view(
        self,
        baseline_box: np.ndarray,
        candidate_box: np.ndarray,
        rotation: np.ndarray,
        camera_origin: np.ndarray,
        points: np.ndarray,
        valid: np.ndarray,
    ) -> _ViewComparison | None:
        if int(np.count_nonzero(valid)) < self.cfg["min_valid_depth_samples"]:
            return None
        kwargs = {
            "surface_band_m": self.cfg["surface_band_m"],
            "occlusion_margin_m": self.cfg["occlusion_margin_m"],
            "free_space_margin_m": self.cfg["free_space_margin_m"],
        }
        baseline_labels, baseline_entry, measured = classify_rays(
            baseline_box, rotation, camera_origin, points, valid, **kwargs
        )
        surface_reference = baseline_labels == RAY_SURFACE
        reference = surface_reference | (baseline_labels == RAY_FREE_SPACE)
        if (
            int(np.count_nonzero(surface_reference))
            < self.cfg["min_surface_rays"]
            or int(np.count_nonzero(reference))
            < self.cfg["min_reference_rays"]
        ):
            return None
        candidate_labels, candidate_entry, _ = classify_rays(
            candidate_box, rotation, camera_origin, points, valid, **kwargs
        )

        clip = self.cfg["surface_residual_clip_m"]
        baseline_residual = np.clip(
            np.abs(measured[surface_reference] - baseline_entry[surface_reference])
            / clip,
            0.0,
            1.0,
        )
        candidate_residual = np.ones(
            int(np.count_nonzero(surface_reference)), dtype=np.float64
        )
        candidate_comparable = (
            (candidate_labels[surface_reference] == RAY_SURFACE)
            | (candidate_labels[surface_reference] == RAY_FREE_SPACE)
        )
        candidate_residual[candidate_comparable] = np.clip(
            np.abs(
                measured[surface_reference][candidate_comparable]
                - candidate_entry[surface_reference][candidate_comparable]
            )
            / clip,
            0.0,
            1.0,
        )
        surface_retention = float(
            np.mean(candidate_labels[surface_reference] == RAY_SURFACE)
        )
        baseline_free = float(
            np.mean(baseline_labels[reference] == RAY_FREE_SPACE)
        )
        candidate_free = float(
            np.mean(candidate_labels[reference] == RAY_FREE_SPACE)
        )
        baseline_loss = (
            self.cfg["surface_weight"] * float(np.mean(baseline_residual))
            + self.cfg["free_space_weight"] * baseline_free
        )
        candidate_loss = (
            self.cfg["surface_weight"] * float(np.mean(candidate_residual))
            + self.cfg["free_space_weight"] * candidate_free
        )
        return _ViewComparison(
            baseline_loss=baseline_loss,
            candidate_loss=candidate_loss,
            surface_retention=surface_retention,
            baseline_free_ratio=baseline_free,
            candidate_free_ratio=candidate_free,
        )

    def _heldout_score(
        self,
        baseline_box: np.ndarray,
        candidate_box: np.ndarray,
        rotation: np.ndarray,
        camera_poses: np.ndarray,
        points: np.ndarray,
        valid: np.ndarray,
        source_view: int,
    ) -> tuple[float, float, tuple[int, ...]] | None:
        comparisons = []
        heldout_views = []
        for view_index in range(camera_poses.shape[0]):
            if view_index == source_view:
                continue
            comparison = self._compare_view(
                baseline_box,
                candidate_box,
                rotation,
                camera_poses[view_index, :3, 3],
                points[view_index],
                valid[view_index],
            )
            if comparison is None:
                continue
            comparisons.append(comparison)
            heldout_views.append(view_index)
        if len(comparisons) < self.cfg["min_views"] - 1:
            return None

        improvements = np.asarray(
            [item.baseline_loss - item.candidate_loss for item in comparisons],
            dtype=np.float64,
        )
        if np.any(
            [
                item.surface_retention
                < self.cfg["min_surface_retention"]
                for item in comparisons
            ]
        ):
            return None
        if np.any(
            [
                item.candidate_free_ratio
                > item.baseline_free_ratio
                + self.cfg["max_free_space_increase"]
                + 1.0e-12
                for item in comparisons
            ]
        ):
            return None
        median = float(np.median(improvements))
        worst = float(np.min(improvements))
        if median + 1.0e-12 < self.cfg["min_loss_improvement"]:
            return None
        if worst < -self.cfg["max_heldout_regression"] - 1.0e-12:
            return None
        return median, worst, tuple(heldout_views)

    def refine(
        self,
        *,
        anchor_box_xyzlhw,
        anchor_rotation,
        observation_boxes_xyzlhw,
        observation_rotations,
        camera_poses,
        surface_points_world,
        surface_valid,
        frame_ids: Sequence[int] | None = None,
        track_key: Sequence[int] | None = None,
    ) -> CAPFResult:
        """Refine an already-fused native anchor or return it exactly."""

        anchor_raw = np.asarray(anchor_box_xyzlhw)
        rotation_raw = np.asarray(anchor_rotation)
        fallback_box = anchor_raw.copy()
        fallback_rotation = rotation_raw.copy()
        if not self.enabled:
            return CAPFResult(
                accepted=False,
                reason="disabled",
                box_xyzlhw=fallback_box,
                rotation=fallback_rotation,
                attempted_candidates=0,
                updates=(),
            )
        self.stats["fusion_attempts"] += 1

        try:
            anchor = np.asarray(anchor_raw, dtype=np.float64)
            rotation = np.asarray(rotation_raw, dtype=np.float64)
            boxes = _as_numpy(observation_boxes_xyzlhw, np.float64)
            rotations = _as_numpy(observation_rotations, np.float64)
            poses = _as_numpy(camera_poses, np.float64)
            points = _as_numpy(surface_points_world, np.float64)
            valid = _as_numpy(surface_valid, bool)
            view_count = boxes.shape[0]
            if (
                anchor.shape != (6,)
                or rotation.shape != (3, 3)
                or boxes.shape != (view_count, 6)
                or rotations.shape != (view_count, 3, 3)
                or poses.shape != (view_count, 4, 4)
                or points.ndim != 3
                or points.shape[0] != view_count
                or points.shape[2] != 3
                or valid.shape != points.shape[:2]
                or view_count < self.cfg["min_views"]
                or not np.all(np.isfinite(anchor))
                or np.any(anchor[3:] <= 0.0)
                or not np.all(np.isfinite(rotation))
                or not np.all(np.isfinite(boxes))
                or np.any(boxes[:, 3:] <= 0.0)
                or not np.all(np.isfinite(rotations))
                or not np.all(np.isfinite(poses))
            ):
                return self._reject(
                    "invalid_or_insufficient_input",
                    fallback_box,
                    fallback_rotation,
                )
            if frame_ids is None:
                stable_order = np.arange(view_count, dtype=np.int64)
            else:
                frame_ids_array = _as_numpy(frame_ids, np.int64).reshape(-1)
                if frame_ids_array.shape != (view_count,):
                    return self._reject(
                        "invalid_frame_ids", fallback_box, fallback_rotation
                    )
                stable_order = np.argsort(frame_ids_array, kind="stable")

            proposals = []
            for source_view in stable_order.tolist():
                proposals.extend(
                    self._source_face_candidates(
                        anchor,
                        rotation,
                        boxes[source_view],
                        rotations[source_view],
                        poses[source_view, :3, 3],
                        source_view,
                    )
                )
            if not proposals:
                return self._reject(
                    "no_visible_face_candidates", fallback_box, fallback_rotation
                )

            current_box = anchor.copy()
            current_faces = box_to_local_faces(anchor)
            used_faces = set()
            updates = []
            attempted = 0
            while len(updates) < self.cfg["max_accepted_faces"]:
                accepted = []
                for face_index, source_view, proposed_value in proposals:
                    if face_index in used_faces:
                        continue
                    bounded = self._bounded_candidate(
                        anchor,
                        rotation,
                        current_faces,
                        face_index,
                        proposed_value,
                    )
                    if bounded is None:
                        continue
                    candidate_box, face_value = bounded
                    attempted += 1
                    scored = self._heldout_score(
                        current_box,
                        candidate_box,
                        rotation,
                        poses,
                        points,
                        valid,
                        source_view,
                    )
                    if scored is None:
                        continue
                    median, worst, heldout = scored
                    accepted.append(
                        (
                            -median,
                            face_index,
                            source_view,
                            candidate_box,
                            face_value,
                            median,
                            worst,
                            heldout,
                        )
                    )
                if not accepted:
                    break
                accepted.sort(key=lambda item: (item[0], item[1], item[2]))
                (
                    _,
                    face_index,
                    source_view,
                    current_box,
                    face_value,
                    median,
                    worst,
                    heldout,
                ) = accepted[0]
                axis, side = divmod(face_index, 2)
                current_faces[axis, side] = face_value
                used_faces.add(face_index)
                updates.append(
                    CAPFFaceUpdate(
                        face_index=face_index,
                        source_view=source_view,
                        heldout_views=heldout,
                        face_value=face_value,
                        median_loss_improvement=median,
                        worst_loss_improvement=worst,
                    )
                )

            self.stats["candidates"] += attempted
            if not updates:
                self._record_oracle_snapshot(
                    track_key=track_key,
                    anchor_box=anchor,
                    rotation=rotation,
                    proposals=proposals,
                    selected_box=anchor,
                    selected_updates=(),
                )
                return self._reject(
                    "no_heldout_improvement",
                    fallback_box,
                    fallback_rotation,
                    attempted=attempted,
                )
            output = np.asarray(current_box, dtype=anchor_raw.dtype)
            if not np.all(np.isfinite(output)) or np.any(
                output[3:] < self.cfg["min_extent_m"]
            ):
                return self._reject(
                    "invalid_final_candidate",
                    fallback_box,
                    fallback_rotation,
                    attempted=attempted,
                )
            self.stats["fusion_accepted"] += 1
            self.stats["accepted_faces"] += len(updates)
            self.stats["reasons"]["accepted"] = (
                self.stats["reasons"].get("accepted", 0) + 1
            )
            self.stats["improvements"].extend(
                item.median_loss_improvement for item in updates
            )
            self._record_oracle_snapshot(
                track_key=track_key,
                anchor_box=anchor,
                rotation=rotation,
                proposals=proposals,
                selected_box=output,
                selected_updates=updates,
            )
            return CAPFResult(
                accepted=True,
                reason="accepted",
                box_xyzlhw=output,
                rotation=fallback_rotation,
                attempted_candidates=attempted,
                updates=tuple(updates),
            )
        except (FloatingPointError, LinAlgError, ValueError) as error:
            return self._reject(
                f"exception:{type(error).__name__}",
                fallback_box,
                fallback_rotation,
            )

    def _reject(
        self,
        reason: str,
        fallback_box: np.ndarray,
        fallback_rotation: np.ndarray,
        attempted: int = 0,
    ) -> CAPFResult:
        self.stats["reasons"][reason] = self.stats["reasons"].get(reason, 0) + 1
        return CAPFResult(
            accepted=False,
            reason=reason,
            box_xyzlhw=fallback_box.copy(),
            rotation=fallback_rotation.copy(),
            attempted_candidates=int(attempted),
            updates=(),
        )

    def summary(self) -> str:
        attempts = int(self.stats["fusion_attempts"])
        accepted = int(self.stats["fusion_accepted"])
        improvements = np.asarray(self.stats["improvements"], dtype=np.float64)
        median_improvement = (
            float(np.median(improvements)) if improvements.size else float("nan")
        )
        return (
            "CAPF summary | "
            f"observations={self.stats['observations']}, "
            f"attempted={attempts}, accepted={accepted}, "
            f"accept_rate={accepted / max(attempts, 1):.4f}, "
            f"candidates={self.stats['candidates']}, "
            f"accepted_faces={self.stats['accepted_faces']}, "
            f"median_improvement={median_improvement:.6f}, "
            f"reasons={self.stats['reasons']}"
        )


# NumPy exposes this exception at module scope, but the alias keeps the
# defensive fail-open clause readable and testable across supported versions.
LinAlgError = np.linalg.LinAlgError
