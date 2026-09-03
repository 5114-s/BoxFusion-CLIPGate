"""Causal sparse RGB-D memory and periodic TR3D proposal tracking.

This module is prediction-agnostic and dependency-light.  The model provider
is injected, allowing the production path to use a persistent TR3D worker and
tests to use a deterministic fake.  Observer mode has no prediction writer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import time
from typing import Any, Protocol

import numpy as np


@dataclass(frozen=True)
class IncrementalTR3DConfig:
    voxel_size_m: float = 0.03
    pixel_stride: int = 4
    min_depth_m: float = 0.10
    max_depth_m: float = 6.0
    warmup_keyframes: int = 3
    inference_interval_keyframes: int = 5
    max_memory_voxels: int = 300_000
    max_snapshot_points: int = 200_000
    track_iou_threshold: float = 0.15
    track_center_threshold_m: float = 0.30
    min_track_hits: int = 2

    def __post_init__(self) -> None:
        if not 0.005 <= self.voxel_size_m <= 0.20:
            raise ValueError("voxel_size_m outside [0.005,0.20]")
        for name in (
            "pixel_stride", "warmup_keyframes", "inference_interval_keyframes",
            "max_memory_voxels", "max_snapshot_points", "min_track_hits",
        ):
            if int(getattr(self, name)) < 1:
                raise ValueError(f"{name} must be positive")
        if not 0.0 < self.min_depth_m < self.max_depth_m:
            raise ValueError("invalid depth range")


@dataclass(frozen=True)
class TR3DProviderResult:
    corners_world: np.ndarray
    scores: np.ndarray
    model_runtime_s: float
    point_counts: np.ndarray | None = None


class TR3DProvider(Protocol):
    def infer(
        self,
        *,
        scene_id: str,
        prefix_id: str,
        points_world_xyzrgb: np.ndarray,
        axis_align_matrix: np.ndarray,
    ) -> TR3DProviderResult:
        ...


def _readonly(value: Any, dtype: np.dtype) -> np.ndarray:
    result = np.ascontiguousarray(np.asarray(value, dtype=dtype))
    result.setflags(write=False)
    return result


def _validate_transform(value: Any, name: str) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64)
    if (
        matrix.shape != (4, 4) or not np.isfinite(matrix).all()
        or not np.allclose(matrix[3], [0.0, 0.0, 0.0, 1.0], atol=1e-8)
    ):
        raise ValueError(f"{name} must be homogeneous [4,4]")
    return matrix


def backproject_rgbd(
    depth_m: Any,
    image_rgb: Any,
    intrinsics: Any,
    camera_to_world: Any,
    *,
    pixel_stride: int,
    min_depth_m: float,
    max_depth_m: float,
) -> np.ndarray:
    depth = np.asarray(depth_m, dtype=np.float64)
    image = np.asarray(image_rgb)
    intrinsic = np.asarray(intrinsics, dtype=np.float64)
    pose = _validate_transform(camera_to_world, "camera_to_world")
    if depth.ndim != 2 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("depth/image must be [H,W]/[H,W,3]")
    if intrinsic.shape == (4, 4):
        intrinsic = intrinsic[:3, :3]
    if intrinsic.shape != (3, 3) or not np.isfinite(intrinsic).all():
        raise ValueError("intrinsics must be finite [3,3]/[4,4]")
    rows = np.arange(0, depth.shape[0], pixel_stride, dtype=np.int64)
    cols = np.arange(0, depth.shape[1], pixel_stride, dtype=np.int64)
    vv, uu = np.meshgrid(rows, cols, indexing="ij")
    z = depth[vv, uu]
    valid = np.isfinite(z) & (z >= min_depth_m) & (z <= max_depth_m)
    if not np.any(valid):
        return np.empty((0, 6), dtype=np.float32)
    u, v, z = uu[valid].astype(np.float64), vv[valid].astype(np.float64), z[valid]
    pixels = np.column_stack((u, v, np.ones(len(u))))
    rays = pixels @ np.linalg.inv(intrinsic).T
    camera = rays * (z / rays[:, 2])[:, None]
    world = camera @ pose[:3, :3].T + pose[:3, 3]
    # Nearest-neighbour color lookup permits different RGB/depth resolutions.
    color_v = np.minimum((v * image.shape[0] / depth.shape[0]).astype(np.int64), image.shape[0] - 1)
    color_u = np.minimum((u * image.shape[1] / depth.shape[1]).astype(np.int64), image.shape[1] - 1)
    color = image[color_v, color_u].astype(np.float64)
    if color.max(initial=0.0) <= 1.0:
        color *= 255.0
    return np.ascontiguousarray(np.column_stack((world, np.clip(color, 0.0, 255.0))), dtype=np.float32)


class CausalVoxelMemory:
    """Deterministic bounded voxel map with count-weighted XYZRGB means."""

    def __init__(self, config: IncrementalTR3DConfig) -> None:
        self.config = config
        self.keys = np.empty((0, 3), dtype=np.int32)
        self.means = np.empty((0, 6), dtype=np.float32)
        self.counts = np.empty((0,), dtype=np.int32)
        self.last_seen = np.empty((0,), dtype=np.int32)
        self.update_count = 0

    def update(self, points: Any, *, keyframe_index: int) -> None:
        values = np.asarray(points, dtype=np.float32)
        if values.ndim != 2 or values.shape[1] != 6 or not np.isfinite(values).all():
            raise ValueError("memory update points must be finite [N,6]")
        if not len(values):
            self.update_count += 1
            return
        keys = np.floor(values[:, :3] / self.config.voxel_size_m).astype(np.int32)
        frame_keys, inverse = np.unique(keys, axis=0, return_inverse=True)
        frame_sum = np.zeros((len(frame_keys), 6), dtype=np.float64)
        np.add.at(frame_sum, inverse, values.astype(np.float64))
        frame_count = np.bincount(inverse, minlength=len(frame_keys)).astype(np.int64)
        frame_mean = frame_sum / frame_count[:, None]

        all_keys = np.concatenate((self.keys, frame_keys), axis=0)
        all_mean = np.concatenate((self.means.astype(np.float64), frame_mean), axis=0)
        all_count = np.concatenate((self.counts.astype(np.int64), frame_count), axis=0)
        all_seen = np.concatenate((self.last_seen, np.full(len(frame_keys), keyframe_index, dtype=np.int32)))
        unique, merged_inverse = np.unique(all_keys, axis=0, return_inverse=True)
        weighted = np.zeros((len(unique), 6), dtype=np.float64)
        np.add.at(weighted, merged_inverse, all_mean * all_count[:, None])
        counts = np.bincount(merged_inverse, weights=all_count, minlength=len(unique)).astype(np.int64)
        seen = np.full(len(unique), -1, dtype=np.int32)
        np.maximum.at(seen, merged_inverse, all_seen)
        means = weighted / counts[:, None]
        if len(unique) > self.config.max_memory_voxels:
            # High support, recent, then lexicographic key: deterministic eviction.
            order = np.lexsort((unique[:, 2], unique[:, 1], unique[:, 0], -seen, -counts))
            keep = np.sort(order[: self.config.max_memory_voxels])
            unique, means, counts, seen = unique[keep], means[keep], counts[keep], seen[keep]
        self.keys = np.ascontiguousarray(unique, dtype=np.int32)
        self.means = np.ascontiguousarray(means, dtype=np.float32)
        self.counts = np.minimum(counts, np.iinfo(np.int32).max).astype(np.int32)
        self.last_seen = seen
        self.update_count += 1

    def snapshot(self) -> np.ndarray:
        if len(self.means) <= self.config.max_snapshot_points:
            return np.array(self.means, dtype=np.float32, order="C", copy=True)
        order = np.lexsort((self.keys[:, 2], self.keys[:, 1], self.keys[:, 0], -self.last_seen, -self.counts))
        selected = np.sort(order[: self.config.max_snapshot_points])
        return np.array(self.means[selected], dtype=np.float32, order="C", copy=True)


def _aabb_iou(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    if not len(left) or not len(right):
        return np.zeros((len(left), len(right)), dtype=np.float64)
    lmin, lmax = left.min(axis=1), left.max(axis=1)
    rmin, rmax = right.min(axis=1), right.max(axis=1)
    extent = np.maximum(np.minimum(lmax[:, None], rmax[None]) - np.maximum(lmin[:, None], rmin[None]), 0.0)
    intersection = np.prod(extent, axis=2)
    lv = np.prod(lmax - lmin, axis=1)
    rv = np.prod(rmax - rmin, axis=1)
    union = lv[:, None] + rv[None] - intersection
    return np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0.0)


@dataclass
class IncrementalProposalTrack:
    track_id: int
    best_corners: np.ndarray
    best_score: float
    first_call: int
    last_call: int
    hit_count: int = 1
    prefix_ids: list[str] = field(default_factory=list)
    score_sum: float = 0.0
    score_square_sum: float = 0.0
    center_sum: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    center_square_sum: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    extent_sum: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    extent_square_sum: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    match_iou_sum: float = 0.0
    best_point_count: int = 0


class IncrementalTR3DObserver:
    def __init__(self, config: IncrementalTR3DConfig, provider: TR3DProvider) -> None:
        self.config = config
        self.provider = provider
        self.memory = CausalVoxelMemory(config)
        self.scene_id: str | None = None
        self.axis_align_matrix: np.ndarray | None = None
        self.keyframe_count = 0
        self.provider_calls = 0
        self.provider_runtime_s = 0.0
        self.memory_runtime_s = 0.0
        self.tracks: list[IncrementalProposalTrack] = []
        self._next_track_id = 0

    def reset_scene(self, scene_id: str, axis_align_matrix: Any) -> None:
        self.__init__(self.config, self.provider)
        self.scene_id = str(scene_id)
        self.axis_align_matrix = _validate_transform(axis_align_matrix, "axis_align_matrix")

    def process_keyframe(
        self,
        *,
        scene_id: str,
        depth: Any,
        image: Any,
        intrinsics: Any,
        camera_to_world: Any,
        source_timestamp: int,
    ) -> None:
        if self.scene_id != scene_id or self.axis_align_matrix is None:
            raise ValueError("incremental observer scene is not initialized")
        started = time.perf_counter()
        points = backproject_rgbd(
            depth, image, intrinsics, camera_to_world,
            pixel_stride=self.config.pixel_stride,
            min_depth_m=self.config.min_depth_m, max_depth_m=self.config.max_depth_m,
        )
        self.memory.update(points, keyframe_index=self.keyframe_count)
        self.memory_runtime_s += time.perf_counter() - started
        self.keyframe_count += 1
        due = (
            self.keyframe_count >= self.config.warmup_keyframes
            and (self.keyframe_count - self.config.warmup_keyframes) % self.config.inference_interval_keyframes == 0
        )
        if not due:
            return
        prefix_id = f"k{self.keyframe_count:04d}_t{int(source_timestamp):06d}"
        result = self.provider.infer(
            scene_id=scene_id, prefix_id=prefix_id,
            points_world_xyzrgb=self.memory.snapshot(),
            axis_align_matrix=self.axis_align_matrix,
        )
        corners = np.asarray(result.corners_world, dtype=np.float32)
        scores = np.asarray(result.scores, dtype=np.float32)
        if corners.ndim != 3 or corners.shape[1:] != (8, 3) or scores.shape != (len(corners),):
            raise ValueError("TR3D provider returned malformed proposals")
        self.provider_runtime_s += float(result.model_runtime_s)
        self.provider_calls += 1
        point_counts = (
            np.zeros(len(corners), dtype=np.int64)
            if result.point_counts is None
            else np.asarray(result.point_counts, dtype=np.int64)
        )
        if point_counts.shape != (len(corners),) or np.any(point_counts < 0):
            raise ValueError("TR3D provider returned malformed point counts")
        self._associate(corners, scores, point_counts, prefix_id)

    def _associate(
        self, corners: np.ndarray, scores: np.ndarray,
        point_counts: np.ndarray, prefix_id: str,
    ) -> None:
        if not len(corners):
            return
        existing = np.asarray([track.best_corners for track in self.tracks], dtype=np.float32)
        overlaps = _aabb_iou(corners, existing) if len(existing) else np.empty((len(corners), 0))
        centres = corners.mean(axis=1)
        old_centres = existing.mean(axis=1) if len(existing) else np.empty((0, 3))
        used: set[int] = set()
        for row in np.argsort(-scores, kind="stable"):
            match = -1
            if len(existing):
                distance = np.linalg.norm(old_centres - centres[row], axis=1)
                valid = np.flatnonzero(
                    ((overlaps[row] >= self.config.track_iou_threshold) | (distance <= self.config.track_center_threshold_m))
                    & ~np.isin(np.arange(len(existing)), list(used))
                )
                if len(valid):
                    rank = np.lexsort((valid, distance[valid], -overlaps[row, valid]))
                    match = int(valid[rank[0]])
            if match < 0:
                center = corners[row].mean(axis=0).astype(np.float64)
                extent = np.ptp(corners[row], axis=0).astype(np.float64)
                score = float(scores[row])
                self.tracks.append(IncrementalProposalTrack(
                    self._next_track_id, np.array(corners[row], copy=True), score,
                    self.provider_calls - 1, self.provider_calls - 1, 1, [prefix_id],
                    score, score * score, center, center * center,
                    extent, extent * extent, 1.0, int(point_counts[row]),
                ))
                self._next_track_id += 1
            else:
                used.add(match)
                track = self.tracks[match]
                track.hit_count += 1; track.last_call = self.provider_calls - 1
                track.prefix_ids.append(prefix_id)
                score = float(scores[row])
                center = corners[row].mean(axis=0).astype(np.float64)
                extent = np.ptp(corners[row], axis=0).astype(np.float64)
                track.score_sum += score
                track.score_square_sum += score * score
                track.center_sum += center
                track.center_square_sum += center * center
                track.extent_sum += extent
                track.extent_square_sum += extent * extent
                track.match_iou_sum += float(overlaps[row, match])
                if score > track.best_score:
                    track.best_score = score
                    track.best_corners = np.array(corners[row], copy=True)
                    track.best_point_count = int(point_counts[row])

    def finalize(
        self,
        *,
        anchor_corners_world: Any | None = None,
        anchor_scores: Any | None = None,
    ) -> dict[str, Any]:
        confirmed = [track for track in self.tracks if track.hit_count >= self.config.min_track_hits]
        if anchor_corners_world is None:
            anchors = np.empty((0, 8, 3), dtype=np.float64)
            scores = np.empty((0,), dtype=np.float64)
        else:
            anchors = np.asarray(anchor_corners_world, dtype=np.float64)
            scores = np.asarray(anchor_scores, dtype=np.float64)
            if anchors.ndim != 3 or anchors.shape[1:] != (8, 3) or scores.shape != (len(anchors),):
                raise ValueError("incremental observer anchors must be [N,8,3]/[N]")
            if not np.isfinite(anchors).all() or not np.isfinite(scores).all():
                raise ValueError("incremental observer anchors must be finite")
        anchor_iou = _aabb_iou(
            np.asarray([row.best_corners for row in confirmed], dtype=np.float64),
            anchors,
        ) if confirmed else np.empty((0, len(anchors)), dtype=np.float64)
        anchor_centres = anchors.mean(axis=1) if len(anchors) else np.empty((0, 3))

        def serialize(index: int, row: IncrementalProposalTrack) -> dict[str, Any]:
            count = float(row.hit_count)
            center_mean = row.center_sum / count
            extent_mean = row.extent_sum / count
            center_variance = np.maximum(row.center_square_sum / count - center_mean ** 2, 0.0)
            extent_variance = np.maximum(row.extent_square_sum / count - extent_mean ** 2, 0.0)
            score_mean = row.score_sum / count
            score_std = float(np.sqrt(max(row.score_square_sum / count - score_mean ** 2, 0.0)))
            lifespan = row.last_call - row.first_call + 1
            best_center = row.best_corners.mean(axis=0)
            if len(anchors):
                distances = np.linalg.norm(anchor_centres - best_center, axis=1)
                maximum_iou = float(anchor_iou[index].max(initial=0.0))
                nearest = int(
                    np.argmax(anchor_iou[index])
                    if maximum_iou > 0.0 else np.argmin(distances)
                )
                nearest_distance = float(distances[nearest])
                matched_anchor_score = float(scores[nearest])
            else:
                maximum_iou = 0.0; nearest_distance = 1e3; matched_anchor_score = 0.0
            volume = float(np.prod(np.maximum(np.ptp(row.best_corners, axis=0), 1e-6)))
            return {
                "track_id": row.track_id, "hit_count": row.hit_count,
                "best_score": row.best_score, "score_mean": score_mean,
                "score_std": score_std, "first_call": row.first_call,
                "last_call": row.last_call, "lifespan_calls": lifespan,
                "hit_rate": row.hit_count / max(lifespan, 1),
                "center_jitter_m": float(np.sqrt(center_variance.sum())),
                "extent_jitter_m": float(np.sqrt(extent_variance.sum())),
                "extent_jitter_relative": float(
                    np.sqrt(extent_variance.sum()) / max(np.linalg.norm(extent_mean), 1e-6)
                ),
                "mean_match_iou": row.match_iou_sum / count,
                "point_support": row.best_point_count,
                "point_density": row.best_point_count / max(volume, 1e-6),
                "anchor_iou_max": maximum_iou,
                "anchor_center_distance_m": nearest_distance,
                "matched_anchor_score": matched_anchor_score,
                "prefix_ids": row.prefix_ids,
                "best_corners_world": np.asarray(row.best_corners, dtype=np.float64).tolist(),
            }
        return {
            "schema": "boxfusion.tr3d_incremental_online_observer.v3",
            "complete": True, "observer_only": True, "mutation_enabled": False,
            "ground_truth_access": False, "coordinate_frame": "world_unaligned",
            "applied_count": 0, "scene_id": self.scene_id,
            "keyframes": self.keyframe_count, "memory_voxels": len(self.memory.means),
            "provider_calls": self.provider_calls, "tracks": len(self.tracks),
            "confirmed_tracks": len(confirmed),
            "anchor_count": len(anchors),
            "anchor_corners_world": anchors.tolist(),
            "anchor_scores": scores.tolist(),
            "memory_runtime_s": self.memory_runtime_s,
            "provider_runtime_s": self.provider_runtime_s,
            "amortized_ms_per_keyframe": 1000.0 * (self.memory_runtime_s + self.provider_runtime_s) / max(self.keyframe_count, 1),
            "confirmed": [serialize(index, row) for index, row in enumerate(confirmed)],
        }


__all__ = [
    "CausalVoxelMemory", "IncrementalTR3DConfig", "IncrementalTR3DObserver",
    "TR3DProvider", "TR3DProviderResult", "backproject_rgbd",
]
