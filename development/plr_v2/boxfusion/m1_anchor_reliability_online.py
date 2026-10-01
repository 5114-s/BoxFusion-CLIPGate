"""Reliability-gated causal anchor recovery (isolated CALR-v2 prototype).

The frozen CALR baseline uses an exact center voxel as both association and
confirmation.  This prototype keeps the 0.3 m voxel only as a spatial hash,
associates compatible boxes across neighboring cells, and confirms tracks
with the shared bounded multi-view reliability state.  Outputs are a dynamic
top-K shadow view, so an early weak birth can be replaced without revisiting
past frames.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Iterable

import numpy as np

from boxfusion.causal_reliability import CausalReliabilityState
from boxfusion.m1_anchor_online import deterministic_tail_score
from boxfusion.online_candidate_map import _aabb_iou_matrix, _center


Voxel = tuple[int, int, int]


def _box_iou(left: np.ndarray, right: np.ndarray) -> float:
    return float(_aabb_iou_matrix(left[None], right[None])[0, 0])


def _scale_error(left: np.ndarray, right: np.ndarray) -> float:
    left_size = np.ptp(left, axis=0)
    right_size = np.ptp(right, axis=0)
    return float(
        np.mean(np.abs(np.log((right_size + 1e-6) / (left_size + 1e-6))))
    )


def _geometric_strength(left: np.ndarray, right: np.ndarray) -> float:
    reference = max(float(np.linalg.norm(np.ptp(left, axis=0))), 1e-6)
    center_score = math.exp(
        -float(np.linalg.norm(_center(left) - _center(right))) / reference
    )
    scale_score = math.exp(-_scale_error(left, right))
    return float(np.clip((center_score + scale_score + _box_iou(left, right)) / 3.0, 0.0, 1.0))


@dataclass(frozen=True)
class _AnchorObservation:
    frame_id: int
    anchor_id: int
    corners: np.ndarray
    score: float


@dataclass
class _AnchorTrack:
    track_id: int
    observations: list[_AnchorObservation] = field(default_factory=list)
    representative_index: int = 0
    last_ordinal: int = -1

    @property
    def representative(self) -> _AnchorObservation:
        return self.observations[self.representative_index]

    def add(self, observation: _AnchorObservation, ordinal: int, max_observations: int) -> None:
        self.observations.append(observation)
        self.last_ordinal = int(ordinal)
        if len(self.observations) > max_observations:
            # Keep a bounded temporal sample while retaining the strongest row.
            strongest = max(
                range(len(self.observations)),
                key=lambda index: (
                    self.observations[index].score,
                    -self.observations[index].frame_id,
                    -self.observations[index].anchor_id,
                ),
            )
            remove = 0 if strongest != 0 else 1
            del self.observations[remove]
        self._update_medoid()

    def _update_medoid(self) -> None:
        boxes = np.asarray([value.corners for value in self.observations])
        overlap = _aabb_iou_matrix(boxes, boxes)
        totals = overlap.sum(axis=1)
        self.representative_index = min(
            range(len(self.observations)),
            key=lambda index: (
                -float(totals[index]),
                -self.observations[index].score,
                self.observations[index].frame_id,
                self.observations[index].anchor_id,
            ),
        )


class OnlineReliableAnchorRecovery:
    """Bounded, causal anchor recovery with view-diverse confirmation."""

    def __init__(
        self,
        scene_id: str,
        *,
        voxel_m: float = 0.3,
        min_views: int = 3,
        min_reliability: float = 0.35,
        association_distance_m: float = 0.45,
        association_iou: float = 0.05,
        association_scale_log: float = 0.70,
        native_dedup_iou: float = 0.25,
        recovery_dedup_iou: float = 0.25,
        max_anchors_per_frame: int = 300,
        max_track_observations: int = 12,
        max_active_tracks: int = 4096,
        max_births: int = 640,
        track_ttl_keyframes: int = 12,
        max_view_slots: int = 8,
        min_angular_separation_deg: float = 20.0,
        lower_bound_kappa: float = 1.0,
    ) -> None:
        if voxel_m <= 0.0 or association_distance_m <= 0.0:
            raise ValueError("spatial bounds must be positive")
        if min_views < 1 or max_births < 1 or max_active_tracks < 1:
            raise ValueError("invalid CALR-v2 bounds")
        for name, value in (
            ("min_reliability", min_reliability),
            ("association_iou", association_iou),
            ("native_dedup_iou", native_dedup_iou),
            ("recovery_dedup_iou", recovery_dedup_iou),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must lie in [0,1]")
        self.scene_id = str(scene_id)
        self.voxel_m = float(voxel_m)
        self.min_views = int(min_views)
        self.min_reliability = float(min_reliability)
        self.association_distance_m = float(association_distance_m)
        self.association_iou = float(association_iou)
        self.association_scale_log = float(association_scale_log)
        self.native_dedup_iou = float(native_dedup_iou)
        self.recovery_dedup_iou = float(recovery_dedup_iou)
        self.max_anchors_per_frame = int(max_anchors_per_frame)
        self.max_track_observations = int(max_track_observations)
        self.max_active_tracks = int(max_active_tracks)
        self.max_births = int(max_births)
        self.track_ttl_keyframes = int(track_ttl_keyframes)
        self.tracks: dict[int, _AnchorTrack] = {}
        self.next_track_id = 0
        self.frames_seen = 0
        self.last_frame_id = -1
        self.reliability = CausalReliabilityState(
            max_view_slots=max_view_slots,
            min_angular_separation_deg=min_angular_separation_deg,
            state_ttl_keyframes=track_ttl_keyframes,
            # One keyframe can temporarily create up to max_anchors_per_frame
            # rows before the track cap is enforced.
            max_states=max_active_tracks + max_anchors_per_frame,
            lower_bound_kappa=lower_bound_kappa,
            negative_weight=0.0,
        )
        self.peak_tracks = 0
        self.capacity_drops = 0
        self.expired_tracks = 0
        self.associations = 0
        self.created_tracks = 0
        self.shadowed_native = 0
        self.shadowed_recovery = 0
        self.current_outputs = 0

    def _voxel(self, corners: np.ndarray) -> Voxel:
        return tuple(np.floor(_center(corners) / self.voxel_m).astype(np.int64).tolist())

    def _neighbors(self, key: Voxel) -> Iterable[Voxel]:
        radius = int(math.ceil(self.association_distance_m / self.voxel_m))
        for x in range(-radius, radius + 1):
            for y in range(-radius, radius + 1):
                for z in range(-radius, radius + 1):
                    yield key[0] + x, key[1] + y, key[2] + z

    def _drop_track(self, track_id: int) -> None:
        self.tracks.pop(track_id, None)
        self.reliability.discard(track_id)

    def _prune(self, ordinal: int) -> None:
        expired = [
            track_id
            for track_id, track in self.tracks.items()
            if ordinal - track.last_ordinal > self.track_ttl_keyframes
        ]
        for track_id in expired:
            self._drop_track(track_id)
        self.expired_tracks += len(expired)

    def _is_confirmed(self, track_id: int) -> bool:
        summary = self.reliability.summary(track_id)
        return (
            int(summary["positive_views"]) >= self.min_views
            and float(summary["lower"]) >= self.min_reliability
        )

    def _enforce_cap(self) -> None:
        overflow = len(self.tracks) - self.max_active_tracks
        if overflow <= 0:
            return
        ordered = sorted(
            self.tracks,
            key=lambda track_id: (
                self._is_confirmed(track_id),
                float(self.reliability.summary(track_id)["lower"]),
                self.tracks[track_id].last_ordinal,
                -track_id,
            ),
        )
        for track_id in ordered[:overflow]:
            self._drop_track(track_id)
        self.capacity_drops += overflow

    def _add_observation(
        self,
        track: _AnchorTrack,
        observation: _AnchorObservation,
        *,
        ordinal: int,
        camera_origin: np.ndarray,
        geometric_strength: float,
    ) -> None:
        direction = _center(observation.corners) - camera_origin
        if float(np.linalg.norm(direction)) <= 1e-9:
            direction = np.asarray([0.0, 0.0, 1.0])
        # Scores are auxiliary evidence, while repeatable geometry remains the
        # dominant signal for weak anchors.
        quality = 0.5 + 0.5 * float(observation.score)
        strength = math.sqrt(max(0.0, quality * geometric_strength))
        self.reliability.update(
            track.track_id,
            frame_id=observation.frame_id,
            ordinal=ordinal,
            view_direction=direction,
            strength=float(np.clip(strength, 0.0, 1.0)),
            visibility=1.0,
            matched=True,
        )
        track.add(observation, ordinal, self.max_track_observations)

    def update(
        self,
        ordinal: int,
        frame_id: int,
        anchor_ids: Iterable[int],
        corners: np.ndarray,
        scores: Iterable[float],
        camera_to_world: np.ndarray,
        *,
        native_corners: np.ndarray | None = None,
        proposal_recovery_corners: np.ndarray | None = None,
    ) -> list[dict[str, object]]:
        if ordinal != self.frames_seen:
            raise ValueError(f"expected ordinal {self.frames_seen}, got {ordinal}")
        if frame_id <= self.last_frame_id:
            raise ValueError("frame_id must increase strictly")
        ids = np.asarray(anchor_ids, dtype=np.int64).reshape(-1)
        boxes = np.asarray(corners, dtype=np.float64)
        if boxes.size == 0:
            boxes = np.empty((0, 8, 3), dtype=np.float64)
        boxes = boxes.reshape(-1, 8, 3)
        values = np.asarray(scores, dtype=np.float64).reshape(-1)
        pose = np.asarray(camera_to_world, dtype=np.float64)
        if len(ids) != len(boxes) or len(ids) != len(values):
            raise ValueError("anchor rows must align")
        if pose.shape != (4, 4) or not np.isfinite(pose).all():
            raise ValueError("camera_to_world must be a finite [4,4] matrix")
        if len(boxes) and (
            not np.isfinite(boxes).all()
            or np.any(np.ptp(boxes, axis=1) <= 0.0)
            or np.any((values < 0.0) | (values > 1.0))
        ):
            raise ValueError("invalid lifted anchor rows")

        self._prune(ordinal)
        order = sorted(
            range(len(ids)), key=lambda row: (-values[row], int(ids[row]), row)
        )[: self.max_anchors_per_frame]
        spatial: dict[Voxel, list[int]] = {}
        for track_id, track in self.tracks.items():
            spatial.setdefault(self._voxel(track.representative.corners), []).append(track_id)

        pairs: list[tuple[float, float, float, int, int]] = []
        for row in order:
            box = boxes[row]
            candidate_ids: set[int] = set()
            for key in self._neighbors(self._voxel(box)):
                candidate_ids.update(spatial.get(key, ()))
            for track_id in candidate_ids:
                reference = self.tracks[track_id].representative.corners
                distance = float(np.linalg.norm(_center(box) - _center(reference)))
                scale = _scale_error(reference, box)
                overlap = _box_iou(reference, box)
                if distance > self.association_distance_m or scale > self.association_scale_log:
                    continue
                if overlap < self.association_iou and distance > 0.5 * self.association_distance_m:
                    continue
                consistency = _geometric_strength(reference, box)
                pairs.append((consistency, -distance, -scale, track_id, row))
        pairs.sort(
            key=lambda value: (
                -value[0], -value[1], -value[2], value[3], int(ids[value[4]])
            )
        )
        used_tracks: set[int] = set()
        used_rows: set[int] = set()
        camera_origin = pose[:3, 3]
        for consistency, _, _, track_id, row in pairs:
            if track_id in used_tracks or row in used_rows:
                continue
            observation = _AnchorObservation(
                frame_id=int(frame_id),
                anchor_id=int(ids[row]),
                corners=np.array(boxes[row], copy=True),
                score=float(values[row]),
            )
            self._add_observation(
                self.tracks[track_id],
                observation,
                ordinal=ordinal,
                camera_origin=camera_origin,
                geometric_strength=consistency,
            )
            used_tracks.add(track_id)
            used_rows.add(row)
            self.associations += 1

        for row in order:
            if row in used_rows:
                continue
            track_id = self.next_track_id
            self.next_track_id += 1
            track = _AnchorTrack(track_id=track_id)
            self.tracks[track_id] = track
            observation = _AnchorObservation(
                frame_id=int(frame_id),
                anchor_id=int(ids[row]),
                corners=np.array(boxes[row], copy=True),
                score=float(values[row]),
            )
            self._add_observation(
                track,
                observation,
                ordinal=ordinal,
                camera_origin=camera_origin,
                geometric_strength=1.0,
            )
            self.created_tracks += 1

        self._enforce_cap()
        self.reliability.prune(ordinal, set(self.tracks))
        self.frames_seen += 1
        self.last_frame_id = int(frame_id)
        self.peak_tracks = max(self.peak_tracks, len(self.tracks))
        return self.rows(
            native_corners=native_corners,
            proposal_recovery_corners=proposal_recovery_corners,
        )

    @staticmethod
    def _corners_or_empty(value: np.ndarray | None) -> np.ndarray:
        if value is None:
            return np.empty((0, 8, 3), dtype=np.float64)
        array = np.asarray(value, dtype=np.float64)
        if array.size == 0:
            return np.empty((0, 8, 3), dtype=np.float64)
        return array.reshape(-1, 8, 3)

    def rows(
        self,
        *,
        native_corners: np.ndarray | None = None,
        proposal_recovery_corners: np.ndarray | None = None,
    ) -> list[dict[str, object]]:
        native = self._corners_or_empty(native_corners)
        recovered = self._corners_or_empty(proposal_recovery_corners)
        candidates = []
        shadowed_native = 0
        shadowed_recovery = 0
        for track_id, track in self.tracks.items():
            if not self._is_confirmed(track_id):
                continue
            representative = track.representative
            summary = self.reliability.summary(track_id)
            if len(native) and float(_aabb_iou_matrix(representative.corners[None], native).max()) >= self.native_dedup_iou:
                shadowed_native += 1
                continue
            if len(recovered) and float(_aabb_iou_matrix(representative.corners[None], recovered).max()) >= self.recovery_dedup_iou:
                shadowed_recovery += 1
                continue
            candidates.append((track_id, track, representative, summary))
        candidates.sort(
            key=lambda item: (
                -float(item[3]["lower"]),
                -int(item[3]["positive_views"]),
                -item[2].score,
                item[0],
            )
        )

        selected: list[tuple[int, _AnchorTrack, _AnchorObservation, dict]] = []
        for candidate in candidates:
            box = candidate[2].corners
            if selected:
                overlap = _aabb_iou_matrix(
                    box[None], np.asarray([item[2].corners for item in selected])
                )
                if float(overlap.max()) >= self.recovery_dedup_iou:
                    continue
            selected.append(candidate)
            if len(selected) >= self.max_births:
                break

        rows: list[dict[str, object]] = []
        for track_id, track, representative, summary in selected:
            combined = float(
                np.clip(0.8 * float(summary["lower"]) + 0.2 * representative.score, 0.0, 1.0)
            )
            rows.append(
                {
                    "track_id": track_id,
                    "box": np.array(representative.corners, copy=True),
                    "raw_score": representative.score,
                    "score": deterministic_tail_score(
                        combined,
                        self.scene_id,
                        representative.frame_id,
                        representative.anchor_id,
                    ),
                    "frame_id": representative.frame_id,
                    "anchor_id": representative.anchor_id,
                    "support_views": int(summary["positive_views"]),
                    "reliability": float(summary["lower"]),
                    "observations": len(track.observations),
                }
            )
        self.shadowed_native = shadowed_native
        self.shadowed_recovery = shadowed_recovery
        self.current_outputs = len(rows)
        return rows

    def diagnostics(self) -> dict[str, object]:
        confirmed = sum(self._is_confirmed(track_id) for track_id in self.tracks)
        return {
            "frames_seen": self.frames_seen,
            "tracks": len(self.tracks),
            "confirmed_tracks": confirmed,
            "current_outputs": self.current_outputs,
            "peak_tracks": self.peak_tracks,
            "max_active_tracks": self.max_active_tracks,
            "max_births": self.max_births,
            "capacity_drops": self.capacity_drops,
            "expired_tracks": self.expired_tracks,
            "associations": self.associations,
            "created_tracks": self.created_tracks,
            "shadowed_native_current": self.shadowed_native,
            "shadowed_recovery_current": self.shadowed_recovery,
            "reliability": self.reliability.diagnostics(),
        }
