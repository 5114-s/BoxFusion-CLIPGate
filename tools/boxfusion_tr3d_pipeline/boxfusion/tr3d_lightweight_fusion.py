"""Isolated lightweight online evidence and asynchronous TR3D tracking.

This module is deliberately opt-in.  The registered B6/G0/terminal-R3 and
incremental novelty paths do not import or instantiate it unless the new
``--tr3d-lightweight-fusion`` flag is supplied.
"""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
import math
import time
from typing import Any

import numpy as np

from .tr3d_incremental_online import (
    CausalVoxelMemory,
    IncrementalProposalTrack,
    IncrementalTR3DConfig,
    IncrementalTR3DObserver,
    TR3DProvider,
    TR3DProviderResult,
    _aabb_iou,
    _validate_transform,
    backproject_rgbd,
)
from .tr3d_r2_geometry import classify_depth_rays, yaw_obb_corners_world
from .tr3d_r4_smov_observer import corners_to_yaw_boxes


@dataclass(frozen=True)
class LightweightFusionConfig:
    stage: int = 6
    top_k_views: int = 5
    diversity_weight: float = 0.30
    min_view_angle_deg: float = 12.0
    depth_pixel_stride: int = 6
    depth_margin_m: float = 0.05
    support_weight: float = 0.55
    occlusion_weight: float = 0.10
    free_space_weight: float = 0.75
    invalid_weight: float = 0.15
    fused_choice_margin: float = 0.02
    max_pending_snapshots: int = 1
    drain_on_finalize: bool = False

    def __post_init__(self) -> None:
        if self.stage not in range(1, 7):
            raise ValueError("stage must be one of L1..L6")
        if self.top_k_views < 1 or self.depth_pixel_stride < 1:
            raise ValueError("top_k_views/depth_pixel_stride must be positive")
        if not 0.0 <= self.diversity_weight <= 1.0:
            raise ValueError("diversity_weight must be in [0,1]")
        if not 0.0 <= self.min_view_angle_deg <= 180.0:
            raise ValueError("min_view_angle_deg must be in [0,180]")
        if self.depth_margin_m < 0.0 or self.fused_choice_margin < 0.0:
            raise ValueError("depth/fusion margins must be non-negative")
        if self.max_pending_snapshots != 1:
            raise ValueError("only latest-only max_pending_snapshots=1 is supported")


@dataclass(frozen=True)
class DepthViewEvidence:
    frame_id: int
    camera_center: np.ndarray
    view_direction: np.ndarray
    area_ratio: float
    support_ratio: float
    occluded_ratio: float
    free_space_ratio: float
    invalid_ratio: float
    quality: float


@dataclass
class LightweightTrackState:
    boxes: list[np.ndarray] = field(default_factory=list)
    scores: list[float] = field(default_factory=list)
    point_counts: list[int] = field(default_factory=list)
    evidences: list[DepthViewEvidence] = field(default_factory=list)
    evidence_boxes: list[np.ndarray] = field(default_factory=list)
    evidence_scores: list[float] = field(default_factory=list)


def _unit(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm > 1e-9 else np.zeros(3, dtype=np.float64)


def depth_view_evidence(
    corners_world: Any,
    *,
    depth: Any,
    intrinsics: Any,
    camera_to_world: Any,
    frame_id: int,
    config: LightweightFusionConfig,
) -> DepthViewEvidence | None:
    corners = np.asarray(corners_world, dtype=np.float64).reshape(1, 8, 3)
    box = corners_to_yaw_boxes(corners)[0]
    pose = _validate_transform(camera_to_world, "camera_to_world")
    evidence = classify_depth_rays(
        depth, box, intrinsics, pose,
        pixel_stride=config.depth_pixel_stride,
        margin=config.depth_margin_m,
        min_depth=0.10,
        max_depth=8.0,
    )
    centre = box[:3]
    camera_center = pose[:3, 3].astype(np.float64)
    direction = _unit(camera_center - centre)
    if evidence is None:
        return None
    area = float(evidence.projection.area_ratio)
    area_quality = float(np.clip(math.sqrt(max(area, 0.0) / 0.02), 0.0, 1.0))
    quality = (
        config.support_weight * evidence.support_ratio
        + config.occlusion_weight * evidence.occluded_ratio
        + 0.20 * area_quality
        - config.free_space_weight * evidence.free_space_ratio
        - config.invalid_weight * evidence.invalid_ratio
    )
    return DepthViewEvidence(
        int(frame_id), camera_center, direction, area,
        evidence.support_ratio, evidence.occluded_ratio,
        evidence.free_space_ratio, evidence.invalid_ratio,
        float(np.clip(quality, -1.0, 1.0)),
    )


def diverse_top_k_indices(
    evidence: list[DepthViewEvidence],
    top_k: int,
    *,
    diversity_weight: float,
    min_view_angle_deg: float,
) -> list[int]:
    """Greedy quality/diversity selection with deterministic tie breaking."""
    if not evidence:
        return []
    remaining = set(range(len(evidence)))
    selected: list[int] = []
    min_angle = math.radians(min_view_angle_deg)
    while remaining and len(selected) < top_k:
        ranked = []
        for index in remaining:
            row = evidence[index]
            if not selected:
                diversity = 1.0
                angle = math.pi
            else:
                angles = [
                    math.acos(float(np.clip(np.dot(row.view_direction, evidence[j].view_direction), -1.0, 1.0)))
                    for j in selected
                ]
                angle = min(angles)
                diversity = float(np.clip(angle / (math.pi / 2.0), 0.0, 1.0))
            score = (1.0 - diversity_weight) * row.quality + diversity_weight * diversity
            # Prefer genuinely different views, but never make the selector empty.
            penalty = 0.15 if selected and angle < min_angle else 0.0
            ranked.append((score - penalty, row.quality, -row.frame_id, -index, index))
        chosen = max(ranked)[-1]
        selected.append(chosen)
        remaining.remove(chosen)
    return selected


def fuse_yaw_boxes(
    boxes: np.ndarray, weights: np.ndarray,
) -> np.ndarray:
    values = np.asarray(boxes, dtype=np.float64)
    weight = np.asarray(weights, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 7 or weight.shape != (len(values),):
        raise ValueError("boxes/weights must be [N,7]/[N]")
    weight = np.maximum(weight, 1e-6)
    weight /= weight.sum()
    centre = np.sum(values[:, :3] * weight[:, None], axis=0)
    # Geometric mean is less sensitive to one inflated extent.
    dimensions = np.exp(np.sum(np.log(np.maximum(values[:, 3:6], 1e-5)) * weight[:, None], axis=0))
    double_yaw = values[:, 6] * 2.0
    yaw = 0.5 * math.atan2(float(np.sum(weight * np.sin(double_yaw))), float(np.sum(weight * np.cos(double_yaw))))
    return np.asarray([*centre, *dimensions, yaw], dtype=np.float64)


class LightweightAsyncTR3DObserver(IncrementalTR3DObserver):
    """Latest-only async provider plus depth-aware multi-view track memory."""

    def __init__(
        self,
        config: IncrementalTR3DConfig,
        provider: TR3DProvider,
        lightweight: LightweightFusionConfig,
    ) -> None:
        super().__init__(config, provider)
        self.lightweight = lightweight
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="tr3d-latest")
        self._future: Future[TR3DProviderResult] | None = None
        self._future_prefix: str | None = None
        self._pending: tuple[str, np.ndarray] | None = None
        self._latest_frame: tuple[np.ndarray, np.ndarray, np.ndarray, int] | None = None
        self.track_state: dict[int, LightweightTrackState] = {}
        self.async_submitted = 0
        self.async_completed = 0
        self.async_replaced = 0
        self.async_dropped_finalize = 0
        self.wall_provider_s = 0.0
        self._future_started = 0.0

    def reset_scene(self, scene_id: str, axis_align_matrix: Any) -> None:
        if self._future is not None:
            self._future.result()
        provider, config, lightweight = self.provider, self.config, self.lightweight
        self._executor.shutdown(wait=True)
        self.__init__(config, provider, lightweight)
        self.scene_id = str(scene_id)
        self.axis_align_matrix = _validate_transform(axis_align_matrix, "axis_align_matrix")

    def _submit(self, prefix_id: str, snapshot: np.ndarray) -> None:
        assert self.scene_id is not None and self.axis_align_matrix is not None
        self._future_prefix = prefix_id
        self._future_started = time.perf_counter()
        self._future = self._executor.submit(
            self.provider.infer,
            scene_id=self.scene_id,
            prefix_id=prefix_id,
            points_world_xyzrgb=snapshot,
            axis_align_matrix=self.axis_align_matrix,
        )
        self.async_submitted += 1

    def _consume(self, *, wait: bool = False) -> bool:
        if self._future is None or (not wait and not self._future.done()):
            return False
        result = self._future.result()
        prefix = str(self._future_prefix)
        self.wall_provider_s += time.perf_counter() - self._future_started
        self._future = None
        self._future_prefix = None
        corners = np.asarray(result.corners_world, dtype=np.float32)
        scores = np.asarray(result.scores, dtype=np.float32)
        counts = np.zeros(len(corners), np.int64) if result.point_counts is None else np.asarray(result.point_counts, np.int64)
        if corners.ndim != 3 or corners.shape[1:] != (8, 3) or scores.shape != (len(corners),) or counts.shape != (len(corners),):
            raise ValueError("TR3D provider returned malformed proposals")
        self.provider_runtime_s += float(result.model_runtime_s)
        self.provider_calls += 1
        self.async_completed += 1
        self._associate_lightweight(corners, scores, counts, prefix)
        if self._pending is not None:
            pending = self._pending
            self._pending = None
            self._submit(*pending)
        return True

    def process_keyframe(self, *, scene_id: str, depth: Any, image: Any, intrinsics: Any, camera_to_world: Any, source_timestamp: int) -> None:
        if self.scene_id != scene_id or self.axis_align_matrix is None:
            raise ValueError("incremental observer scene is not initialized")
        self._consume()
        started = time.perf_counter()
        points = backproject_rgbd(
            depth, image, intrinsics, camera_to_world,
            pixel_stride=self.config.pixel_stride,
            min_depth_m=self.config.min_depth_m,
            max_depth_m=self.config.max_depth_m,
        )
        self.memory.update(points, keyframe_index=self.keyframe_count)
        self.memory_runtime_s += time.perf_counter() - started
        self._latest_frame = (
            np.asarray(depth).copy(), np.asarray(intrinsics).copy(),
            np.asarray(camera_to_world).copy(), int(source_timestamp),
        )
        self.keyframe_count += 1
        due = self.keyframe_count >= self.config.warmup_keyframes and (self.keyframe_count - self.config.warmup_keyframes) % self.config.inference_interval_keyframes == 0
        if not due:
            return
        prefix = f"k{self.keyframe_count:04d}_t{int(source_timestamp):06d}"
        item = (prefix, self.memory.snapshot())
        if self.lightweight.stage < 3:
            started_provider = time.perf_counter()
            result = self.provider.infer(
                scene_id=scene_id, prefix_id=prefix,
                points_world_xyzrgb=item[1],
                axis_align_matrix=self.axis_align_matrix,
            )
            self.wall_provider_s += time.perf_counter() - started_provider
            corners = np.asarray(result.corners_world, dtype=np.float32)
            scores = np.asarray(result.scores, dtype=np.float32)
            counts = (
                np.zeros(len(corners), np.int64)
                if result.point_counts is None
                else np.asarray(result.point_counts, np.int64)
            )
            self.provider_runtime_s += float(result.model_runtime_s)
            self.provider_calls += 1
            self.async_completed += 1
            self._associate_lightweight(corners, scores, counts, prefix)
            return
        if self._future is None:
            self._submit(*item)
        else:
            if self._pending is not None:
                self.async_replaced += 1
            self._pending = item

    def _associate_lightweight(self, corners: np.ndarray, scores: np.ndarray, point_counts: np.ndarray, prefix_id: str) -> None:
        if not len(corners):
            return
        existing = np.asarray([track.best_corners for track in self.tracks], np.float32)
        overlaps = _aabb_iou(corners, existing) if len(existing) else np.empty((len(corners), 0))
        centres = corners.mean(axis=1)
        old_centres = existing.mean(axis=1) if len(existing) else np.empty((0, 3))
        used: set[int] = set()
        for row in np.argsort(-scores, kind="stable"):
            match = -1
            if len(existing):
                distance = np.linalg.norm(old_centres - centres[row], axis=1)
                valid = np.flatnonzero(((overlaps[row] >= self.config.track_iou_threshold) | (distance <= self.config.track_center_threshold_m)) & ~np.isin(np.arange(len(existing)), list(used)))
                if len(valid):
                    match = int(valid[np.lexsort((valid, distance[valid], -overlaps[row, valid]))[0]])
            score = float(scores[row])
            centre = centres[row].astype(np.float64)
            extent = np.ptp(corners[row], axis=0).astype(np.float64)
            if match < 0:
                track = IncrementalProposalTrack(
                    self._next_track_id, np.array(corners[row], copy=True), score,
                    self.provider_calls - 1, self.provider_calls - 1, 1, [prefix_id],
                    score, score * score, centre, centre * centre, extent,
                    extent * extent, 1.0, int(point_counts[row]),
                )
                self.tracks.append(track)
                self.track_state[track.track_id] = LightweightTrackState()
                self._next_track_id += 1
            else:
                used.add(match)
                track = self.tracks[match]
                track.hit_count += 1
                track.last_call = self.provider_calls - 1
                track.prefix_ids.append(prefix_id)
                track.score_sum += score
                track.score_square_sum += score * score
                track.center_sum += centre
                track.center_square_sum += centre * centre
                track.extent_sum += extent
                track.extent_square_sum += extent * extent
                track.match_iou_sum += float(overlaps[row, match])
                if score > track.best_score:
                    track.best_score = score
                    track.best_corners = np.array(corners[row], copy=True)
                    track.best_point_count = int(point_counts[row])
            state = self.track_state[track.track_id]
            state.boxes.append(corners_to_yaw_boxes(corners[row:row + 1])[0])
            state.scores.append(score)
            state.point_counts.append(int(point_counts[row]))
            if self.lightweight.stage >= 1 and self._latest_frame is not None:
                depth, intrinsics, pose, frame_id = self._latest_frame
                view = depth_view_evidence(corners[row], depth=depth, intrinsics=intrinsics, camera_to_world=pose, frame_id=frame_id, config=self.lightweight)
                if view is not None:
                    state.evidences.append(view)
                    state.evidence_boxes.append(
                        corners_to_yaw_boxes(corners[row:row + 1])[0]
                    )
                    state.evidence_scores.append(score)

    def _augment_summary(self, summary: dict[str, Any]) -> None:
        rows = {int(row["track_id"]): row for row in summary["confirmed"]}
        anchors = np.asarray(
            summary.get("anchor_corners_world", []), dtype=np.float64
        ).reshape(-1, 8, 3)
        anchor_centres = (
            anchors.mean(axis=1) if len(anchors) else np.empty((0, 3))
        )
        for track in self.tracks:
            row = rows.get(track.track_id)
            if row is None:
                continue
            state = self.track_state.get(track.track_id, LightweightTrackState())
            if self.lightweight.stage >= 2:
                indices = diverse_top_k_indices(
                    state.evidences, self.lightweight.top_k_views,
                    diversity_weight=self.lightweight.diversity_weight,
                    min_view_angle_deg=self.lightweight.min_view_angle_deg,
                )
            else:
                indices = sorted(
                    range(len(state.evidences)),
                    key=lambda index: (
                        -state.evidences[index].quality,
                        state.evidences[index].frame_id,
                        index,
                    ),
                )[: self.lightweight.top_k_views]
            views = [state.evidences[i] for i in indices]
            qualities = np.asarray([max(view.quality + 1.0, 1e-4) for view in views], np.float64)
            raw = np.asarray(row["best_corners_world"], np.float64)
            fused = raw
            if len(indices) >= 2:
                # Evidence is attached in observation order; use the matching
                # latest observations if projections were unavailable.
                box_indices = np.asarray(indices, np.int64)
                if int(box_indices.max(initial=-1)) < len(state.evidence_boxes):
                    fused_box = fuse_yaw_boxes(
                        np.asarray(state.evidence_boxes)[box_indices], qualities
                    )
                    fused = yaw_obb_corners_world(fused_box)
            if views:
                raw_local = max(
                    range(len(indices)),
                    key=lambda local: (
                        state.evidence_scores[indices[local]],
                        -state.evidences[indices[local]].frame_id,
                    ),
                )
                raw_quality = float(views[raw_local].quality)
                fused_quality = float(
                    np.average(
                        [view.quality for view in views], weights=qualities
                    )
                    + min(0.06, 0.02 * math.log2(len(views)))
                )
            else:
                raw_quality = -1.0
                fused_quality = -1.0
            use_fused = (
                self.lightweight.stage >= 5
                and len(views) >= 2
                and fused_quality
                >= raw_quality + self.lightweight.fused_choice_margin
            )
            selected = fused if use_fused else raw
            if len(anchors):
                selected_iou = _aabb_iou(selected[None], anchors)[0]
                selected_distances = np.linalg.norm(
                    anchor_centres - selected.mean(axis=0), axis=1
                )
                selected_anchor_iou = float(selected_iou.max(initial=0.0))
                selected_anchor_distance = float(selected_distances.min())
            else:
                selected_anchor_iou = 0.0
                selected_anchor_distance = 1e3
            row.update({
                "lightweight_schema": "boxfusion.tr3d_lightweight_track.v1",
                "visibility_view_count": len(state.evidences),
                "diverse_topk_count": len(views),
                "diverse_topk_frame_ids": [view.frame_id for view in views],
                "visibility_quality_mean": float(np.mean([view.quality for view in views])) if views else -1.0,
                "visibility_quality_max": float(np.max([view.quality for view in views])) if views else -1.0,
                "support_ratio_mean": float(np.mean([view.support_ratio for view in views])) if views else 0.0,
                "occluded_ratio_mean": float(np.mean([view.occluded_ratio for view in views])) if views else 0.0,
                "free_space_ratio_mean": float(np.mean([view.free_space_ratio for view in views])) if views else 1.0,
                "invalid_ratio_mean": float(np.mean([view.invalid_ratio for view in views])) if views else 1.0,
                "raw_corners_world": raw.tolist(),
                "fused_corners_world": np.asarray(fused).tolist(),
                "selected_geometry": "fused" if use_fused else "raw",
                "selected_corners_world": np.asarray(selected).tolist(),
                "selected_anchor_iou_max": selected_anchor_iou,
                "selected_anchor_center_distance_m": selected_anchor_distance,
                "raw_quality": float(raw_quality),
                "fused_quality": float(fused_quality),
            })

    def finalize(self, *, anchor_corners_world: Any | None = None, anchor_scores: Any | None = None) -> dict[str, Any]:
        if self.lightweight.drain_on_finalize:
            while self._future is not None:
                self._consume(wait=True)
        else:
            self._consume()
            if self._future is not None:
                self.async_dropped_finalize += 1
                self._future.result()  # orderly worker protocol shutdown
                self._future = None
            if self._pending is not None:
                self.async_dropped_finalize += 1
                self._pending = None
        summary = super().finalize(anchor_corners_world=anchor_corners_world, anchor_scores=anchor_scores)
        self._augment_summary(summary)
        summary.update({
            "schema": "boxfusion.tr3d_lightweight_online_observer.v1",
            "lightweight_fusion": True,
            "lightweight_stage": self.lightweight.stage,
            "async_latest_only": self.lightweight.stage >= 3,
            "async_submitted": self.async_submitted,
            "async_completed": self.async_completed,
            "async_replaced": self.async_replaced,
            "async_dropped_finalize": self.async_dropped_finalize,
            "provider_wall_s": self.wall_provider_s,
            "top_k_views": self.lightweight.top_k_views,
            "modules": [
                "ovscan_depth_visibility", "zoo3d_diverse_topk",
                "async_incremental_tr3d", "smov3d_free_space",
                "insfusion_raw_fused_choice", "source_aware_low_score_append",
            ][: self.lightweight.stage],
        })
        self._executor.shutdown(wait=True)
        return summary


__all__ = [
    "DepthViewEvidence", "LightweightAsyncTR3DObserver",
    "LightweightFusionConfig", "depth_view_evidence",
    "diverse_top_k_indices", "fuse_yaw_boxes",
]
