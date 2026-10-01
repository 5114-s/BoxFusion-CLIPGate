"""Causal multi-view recovery for low-score Boxer-lifted proposals."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
import json
import math
import os
from pathlib import Path
import random
import tempfile
from typing import Any, Mapping

import numpy as np
import torch

from boxfusion.instances import Instances3D
from boxfusion.sealed_boxer_proposal_cache import (
    ProposalCache,
    ProposalCacheConfig,
)


SCHEMA = "boxfusion.boxer_mvpr.v1"
SCHEMA_V2 = "boxfusion.boxer_mvpr.v2"


def _aabb_geometry(corners: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    lower = corners.min(axis=0)
    upper = corners.max(axis=0)
    center = 0.5 * (lower + upper)
    volume = float(np.prod(upper - lower))
    return lower, upper, center, volume


def _aabb_iou(left: np.ndarray, right: np.ndarray) -> float:
    lo_a, hi_a, _, volume_a = _aabb_geometry(left)
    lo_b, hi_b, _, volume_b = _aabb_geometry(right)
    intersection = float(np.prod(np.maximum(np.minimum(hi_a, hi_b) - np.maximum(lo_a, lo_b), 0.0)))
    union = volume_a + volume_b - intersection
    return intersection / union if union > 0.0 else 0.0


def _box_iou_2d(left: np.ndarray, right: np.ndarray) -> float:
    intersection_wh = np.maximum(
        np.minimum(left[2:], right[2:]) - np.maximum(left[:2], right[:2]),
        0.0,
    )
    intersection = float(np.prod(intersection_wh))
    left_area = float(np.prod(np.maximum(left[2:] - left[:2], 0.0)))
    right_area = float(np.prod(np.maximum(right[2:] - right[:2], 0.0)))
    union = left_area + right_area - intersection
    return intersection / union if union > 0.0 else 0.0


@dataclass(frozen=True)
class BoxerMVPRConfig:
    enabled: bool = False
    low_score_min: float = 0.40
    native_score_min: float = 0.50
    max_candidates_per_frame: int = 8
    minimum_views: int = 3
    ttl_keyframes: int = 10
    match_aabb_iou: float = 0.10
    match_center_m: float = 0.50
    min_median_pairwise_aabb_iou: float = 0.25
    max_center_rms_m: float = 0.25
    min_medoid_aabb_extent_m: float = 0.20
    native_duplicate_2d_iou: float = 0.995
    rolling_consensus: bool = False
    evidence_window: int = 3
    promote_consensus_medoid: bool = False
    strong_two_view_confirmation: bool = False
    strong_two_view_min_pairwise_aabb_iou: float = 0.45
    strong_two_view_max_center_rms_m: float = 0.12
    strong_two_view_min_median_score: float = 0.46
    diagnostics_dir: str = ""

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> "BoxerMVPRConfig":
        values = {
            name: mapping.get(name, default)
            for name, default in (
                ("enabled", False),
                ("low_score_min", 0.40),
                ("native_score_min", 0.50),
                ("max_candidates_per_frame", 8),
                ("minimum_views", 3),
                ("ttl_keyframes", 10),
                ("match_aabb_iou", 0.10),
                ("match_center_m", 0.50),
                ("min_median_pairwise_aabb_iou", 0.25),
                ("max_center_rms_m", 0.25),
                ("min_medoid_aabb_extent_m", 0.20),
                ("native_duplicate_2d_iou", 0.995),
                ("rolling_consensus", False),
                ("evidence_window", 3),
                ("promote_consensus_medoid", False),
                ("strong_two_view_confirmation", False),
                ("strong_two_view_min_pairwise_aabb_iou", 0.45),
                ("strong_two_view_max_center_rms_m", 0.12),
                ("strong_two_view_min_median_score", 0.46),
                ("diagnostics_dir", ""),
            )
        }
        config = cls(**values)
        if not 0.0 <= config.low_score_min < config.native_score_min <= 1.0:
            raise ValueError("Boxer-MVPR score interval must satisfy 0 <= low < native <= 1")
        if config.minimum_views != 3:
            raise ValueError("Boxer-MVPR currently requires exactly three causal views")
        if not config.minimum_views <= config.evidence_window <= 16:
            raise ValueError(
                "Boxer-MVPR evidence_window must lie in [minimum_views,16]"
            )
        if config.promote_consensus_medoid and not config.rolling_consensus:
            raise ValueError(
                "Boxer-MVPR promote_consensus_medoid requires rolling_consensus"
            )
        if config.promote_consensus_medoid and config.strong_two_view_confirmation:
            raise ValueError(
                "Boxer-MVPR consensus medoid and strong two-view confirmation "
                "cannot be enabled together"
            )
        if not 1 <= config.max_candidates_per_frame <= 128:
            raise ValueError("Boxer-MVPR max_candidates_per_frame must lie in [1,128]")
        if not 1 <= config.ttl_keyframes <= 1000:
            raise ValueError("Boxer-MVPR ttl_keyframes must lie in [1,1000]")
        for name in (
            "match_aabb_iou",
            "min_median_pairwise_aabb_iou",
            "native_duplicate_2d_iou",
            "strong_two_view_min_pairwise_aabb_iou",
            "strong_two_view_min_median_score",
        ):
            value = float(getattr(config, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"Boxer-MVPR {name} must lie in [0,1]")
        for name in (
            "match_center_m",
            "max_center_rms_m",
            "min_medoid_aabb_extent_m",
            "strong_two_view_max_center_rms_m",
        ):
            if not math.isfinite(float(getattr(config, name))) or float(getattr(config, name)) <= 0.0:
                raise ValueError(f"Boxer-MVPR {name} must be positive and finite")
        if config.enabled and not config.diagnostics_dir:
            raise ValueError("Boxer-MVPR diagnostics_dir is required when enabled")
        return config


@dataclass(frozen=True)
class _Observation:
    frame_id: int
    local_index: int
    score: float
    corners: np.ndarray
    world_xyz_dims: np.ndarray
    world_rotation: np.ndarray

    @property
    def center(self) -> np.ndarray:
        return _aabb_geometry(self.corners)[2]


@dataclass
class _Track:
    track_id: int
    last_keyframe_step: int
    anchor: _Observation
    evidence: list[_Observation] = field(default_factory=list)
    confirmed: bool = False
    rejected: bool = False
    stability: dict[str, float] = field(default_factory=dict)


@contextmanager
def isolated_rng(device: torch.device | str):
    """Run an auxiliary branch without changing native-path RNG state."""

    device = torch.device(device)
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    cpu_state = torch.get_rng_state()
    cuda_state = None
    if device.type == "cuda":
        cuda_state = torch.cuda.get_rng_state(device)
    try:
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.set_rng_state(cpu_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state(cuda_state, device)


class BoxerMVPR:
    """Replay low-score proposals and promote only causally confirmed views."""

    def __init__(self, cfg: Mapping[str, Any], device: torch.device | str):
        section = cfg.get("lifting", {}).get("boxer_mvpr", {})
        self.config = BoxerMVPRConfig.from_mapping(section)
        if not self.config.enabled:
            raise ValueError("Do not construct BoxerMVPR when it is disabled")
        cache_mapping = section.get("proposal_cache", {})
        cache_config = ProposalCacheConfig.from_mapping(cache_mapping)
        if cache_config.mode != "replay":
            raise ValueError("Boxer-MVPR requires a replay-only low-score proposal cache")
        self.cache = ProposalCache(cache_config, device=torch.device(device))
        self.baseline_prediction_root = Path(
            str(cache_mapping.get("baseline_prediction_root", ""))
        )
        if not self.baseline_prediction_root.is_dir():
            raise ValueError("Boxer-MVPR baseline_prediction_root must be an existing directory")
        self.device = torch.device(device)
        self.scene_id: str | None = None
        self._tracks: dict[int, _Track] = {}
        self._next_track_id = 1
        self._keyframe_step = 0
        self._last_frame_id = -1
        self._finalized = False
        self._events: list[dict[str, Any]] = []
        self._stats = {
            "keyframes": 0,
            "low_rows_replayed": 0,
            "low_rows_in_interval": 0,
            "native_duplicates_removed": 0,
            "observations_selected": 0,
            "tracks_created": 0,
            "tracks_matched": 0,
            "tracks_retired": 0,
            "tracks_confirmed": 0,
            "tracks_rejected": 0,
            "tracks_reconfirmed": 0,
            "tracks_confirmed_two_view": 0,
            "promoted_observations": 0,
            "consensus_geometry_promotions": 0,
        }

    def bind_scene(self, scene_id: str, *, dataset_length: int, gap: int) -> None:
        if self.scene_id is not None and self.scene_id != scene_id:
            raise ValueError("One Boxer-MVPR instance cannot mix scenes")
        self.scene_id = str(scene_id)
        self.cache.bind_scene(self.scene_id, dataset_length=dataset_length, gap=gap)

    def replay_low_candidates(
        self,
        scene_id: str,
        frame_id: int,
        *,
        inputs: Mapping[str, Any],
        native_instances: Instances3D,
    ) -> tuple[Instances3D, str]:
        if self.scene_id != scene_id:
            raise ValueError("Bind Boxer-MVPR to the scene before replay")
        candidates, attempt_id = self.cache.replay(scene_id, frame_id, inputs=inputs)
        scores = candidates.scores.detach().float()
        interval = (scores >= self.config.low_score_min) & (scores < self.config.native_score_min)
        interval_indices = torch.nonzero(interval, as_tuple=False).flatten()
        self._stats["low_rows_replayed"] += len(candidates)
        self._stats["low_rows_in_interval"] += int(interval_indices.numel())
        low = candidates[interval_indices]
        if len(low) == 0 or len(native_instances) == 0:
            return low, attempt_id

        native_boxes = native_instances.pred_boxes.detach().float().cpu().numpy()
        low_boxes = low.pred_boxes.detach().float().cpu().numpy()
        keep = np.ones(len(low), dtype=bool)
        for low_index, low_box in enumerate(low_boxes):
            if any(
                _box_iou_2d(low_box, native_box) >= self.config.native_duplicate_2d_iou
                for native_box in native_boxes
            ):
                keep[low_index] = False
        removed = int((~keep).sum())
        self._stats["native_duplicates_removed"] += removed
        return low[torch.from_numpy(keep).to(low.scores.device)], attempt_id

    def _world_observations(
        self,
        frame_id: int,
        low_instances: Instances3D,
        camera_to_world: Any,
    ) -> list[_Observation]:
        if len(low_instances) == 0:
            return []
        scores = low_instances.scores.detach().float().cpu().numpy()
        selected = sorted(range(len(low_instances)), key=lambda index: (-float(scores[index]), index))
        selected = selected[: self.config.max_candidates_per_frame]
        boxes = low_instances.pred_boxes_3d.clone()
        pose = torch.as_tensor(camera_to_world).detach().float()
        if pose.ndim == 3:
            pose = pose[-1]
        poses = pose.unsqueeze(0).repeat(len(low_instances), 1, 1)
        boxes.transform2world(poses)
        corners = boxes.corners.detach().float().cpu().numpy()
        world_xyz_dims = boxes.tensor.detach().float().cpu().numpy()
        world_rotation = boxes.R.detach().float().cpu().numpy()
        return [
            _Observation(
                frame_id=int(frame_id),
                local_index=int(index),
                score=float(scores[index]),
                corners=np.array(corners[index], dtype=np.float64, copy=True),
                world_xyz_dims=np.array(
                    world_xyz_dims[index], dtype=np.float64, copy=True
                ),
                world_rotation=np.array(
                    world_rotation[index], dtype=np.float64, copy=True
                ),
            )
            for index in selected
        ]

    @staticmethod
    def _medoid(evidence: list[_Observation]) -> _Observation:
        if not evidence:
            raise ValueError("Boxer-MVPR cannot select a medoid without evidence")
        costs = [
            sum(1.0 - _aabb_iou(row.corners, other.corners) for other in evidence)
            for row in evidence
        ]
        index = min(
            range(len(evidence)),
            key=lambda candidate: (
                costs[candidate],
                evidence[candidate].frame_id,
                evidence[candidate].local_index,
            ),
        )
        return evidence[index]

    def _stability(self, evidence: list[_Observation]) -> tuple[bool, dict[str, float]]:
        pairwise_iou = []
        centers = np.stack([row.center for row in evidence], axis=0)
        for left in range(len(evidence)):
            for right in range(left + 1, len(evidence)):
                pairwise_iou.append(_aabb_iou(evidence[left].corners, evidence[right].corners))
        median_iou = float(np.median(pairwise_iou))
        centroid = centers.mean(axis=0)
        center_rms = float(np.sqrt(np.mean(np.sum((centers - centroid) ** 2, axis=1))))
        medoid = self._medoid(evidence)
        extent = float(np.min(np.ptp(medoid.corners, axis=0)))
        metrics = {
            "median_pairwise_aabb_iou": median_iou,
            "center_rms_m": center_rms,
            "medoid_min_aabb_extent_m": extent,
        }
        accepted = (
            median_iou >= self.config.min_median_pairwise_aabb_iou
            and center_rms <= self.config.max_center_rms_m
            and extent >= self.config.min_medoid_aabb_extent_m
        )
        return accepted, metrics

    def _strong_two_view_stability(
        self, evidence: list[_Observation]
    ) -> tuple[bool, dict[str, float]]:
        if len(evidence) != 2:
            raise ValueError("Strong two-view stability requires exactly two views")
        pairwise_iou = _aabb_iou(evidence[0].corners, evidence[1].corners)
        centers = np.stack([row.center for row in evidence], axis=0)
        centroid = centers.mean(axis=0)
        center_rms = float(
            np.sqrt(np.mean(np.sum((centers - centroid) ** 2, axis=1)))
        )
        median_score = float(np.median([row.score for row in evidence]))
        medoid = self._medoid(evidence)
        extent = float(np.min(np.ptp(medoid.corners, axis=0)))
        metrics = {
            "pairwise_aabb_iou": pairwise_iou,
            "center_rms_m": center_rms,
            "median_score": median_score,
            "medoid_min_aabb_extent_m": extent,
        }
        accepted = (
            pairwise_iou >= self.config.strong_two_view_min_pairwise_aabb_iou
            and center_rms <= self.config.strong_two_view_max_center_rms_m
            and median_score >= self.config.strong_two_view_min_median_score
            and extent >= self.config.min_medoid_aabb_extent_m
        )
        return accepted, metrics

    @staticmethod
    def _apply_world_consensus_geometry(
        instances: Instances3D,
        promoted_indices: list[int],
        consensus_by_local_index: Mapping[int, _Observation],
        camera_to_world: Any,
    ) -> int:
        if not consensus_by_local_index or len(instances) == 0:
            return 0
        pose = torch.as_tensor(
            camera_to_world,
            dtype=instances.pred_boxes_3d.tensor.dtype,
            device=instances.pred_boxes_3d.tensor.device,
        )
        if pose.ndim == 3:
            pose = pose[-1]
        world_to_camera_rotation = pose[:3, :3].transpose(0, 1)
        world_translation = pose[:3, 3]
        applied = 0
        for result_index, local_index in enumerate(promoted_indices):
            observation = consensus_by_local_index.get(local_index)
            if observation is None:
                continue
            world_xyz_dims = torch.as_tensor(
                observation.world_xyz_dims,
                dtype=instances.pred_boxes_3d.tensor.dtype,
                device=instances.pred_boxes_3d.tensor.device,
            )
            world_rotation = torch.as_tensor(
                observation.world_rotation,
                dtype=instances.pred_boxes_3d.R.dtype,
                device=instances.pred_boxes_3d.R.device,
            )
            instances.pred_boxes_3d.tensor[result_index, :3] = (
                world_to_camera_rotation
                @ (world_xyz_dims[:3] - world_translation)
            )
            instances.pred_boxes_3d.tensor[result_index, 3:6] = world_xyz_dims[3:6]
            instances.pred_boxes_3d.R[result_index] = (
                world_to_camera_rotation @ world_rotation
            )
            applied += 1
        return applied

    def recover(
        self,
        frame_id: int,
        low_instances: Instances3D,
        *,
        camera_to_world: Any,
    ) -> Instances3D:
        if self._finalized:
            raise RuntimeError("Boxer-MVPR is already finalized")
        if int(frame_id) <= self._last_frame_id:
            raise ValueError("Boxer-MVPR frame IDs must increase")
        self._last_frame_id = int(frame_id)
        step = self._keyframe_step
        self._keyframe_step += 1
        self._stats["keyframes"] += 1

        retired = [
            track_id
            for track_id, track in self._tracks.items()
            if step - track.last_keyframe_step > self.config.ttl_keyframes
        ]
        for track_id in retired:
            del self._tracks[track_id]
        self._stats["tracks_retired"] += len(retired)

        observations = self._world_observations(frame_id, low_instances, camera_to_world)
        self._stats["observations_selected"] += len(observations)
        used_tracks: set[int] = set()
        promoted: list[int] = []
        consensus_by_local_index: dict[int, _Observation] = {}
        frame_assignments: list[dict[str, Any]] = []
        for observation in observations:
            matches = []
            for track_id, track in self._tracks.items():
                if track_id in used_tracks:
                    continue
                reference = (
                    self._medoid(track.evidence)
                    if self.config.rolling_consensus
                    and track.confirmed
                    and track.evidence
                    else track.anchor
                )
                iou = _aabb_iou(observation.corners, reference.corners)
                center_distance = float(
                    np.linalg.norm(observation.center - reference.center)
                )
                if iou >= self.config.match_aabb_iou and center_distance <= self.config.match_center_m:
                    matches.append((-iou, center_distance, track_id))
            if matches:
                _, _, track_id = min(matches)
                track = self._tracks[track_id]
                track.last_keyframe_step = step
                track.anchor = observation
                if self.config.rolling_consensus:
                    track.evidence.append(observation)
                    track.evidence = track.evidence[-self.config.evidence_window :]
                elif len(track.evidence) < self.config.minimum_views:
                    track.evidence.append(observation)
                self._stats["tracks_matched"] += 1
                action = "matched"
            else:
                track_id = self._next_track_id
                self._next_track_id += 1
                track = _Track(
                    track_id=track_id,
                    last_keyframe_step=step,
                    anchor=observation,
                    evidence=[observation],
                )
                self._tracks[track_id] = track
                self._stats["tracks_created"] += 1
                action = "created"
            used_tracks.add(track_id)

            just_confirmed = False
            confirmation_views = 0
            stability_decision: tuple[bool, dict[str, float]] | None = None
            if (
                self.config.strong_two_view_confirmation
                and not track.confirmed
                and not track.rejected
                and len(track.evidence) == 2
            ):
                strong_accepted, strong_metrics = self._strong_two_view_stability(
                    track.evidence
                )
                if strong_accepted:
                    stability_decision = (True, strong_metrics)
                    confirmation_views = 2
            if (
                stability_decision is None
                and not track.confirmed
                and len(track.evidence) >= self.config.minimum_views
                and (self.config.rolling_consensus or not track.rejected)
            ):
                evidence = track.evidence[-self.config.minimum_views :]
                stability_decision = self._stability(evidence)
                confirmation_views = self.config.minimum_views
            if stability_decision is not None:
                was_rejected = track.rejected
                accepted, metrics = stability_decision
                track.stability = metrics
                track.confirmed = accepted
                track.rejected = not accepted
                if accepted:
                    just_confirmed = True
                    self._stats["tracks_confirmed"] += 1
                    if confirmation_views == 2:
                        self._stats["tracks_confirmed_two_view"] += 1
                    if was_rejected:
                        self._stats["tracks_reconfirmed"] += 1
                elif not was_rejected:
                    self._stats["tracks_rejected"] += 1
            if track.confirmed:
                promoted.append(observation.local_index)
                if just_confirmed and self.config.promote_consensus_medoid:
                    consensus_by_local_index[observation.local_index] = self._medoid(
                        track.evidence[-self.config.minimum_views :]
                    )
            frame_assignments.append(
                {
                    "local_index": observation.local_index,
                    "score": observation.score,
                    "track_id": track_id,
                    "action": action,
                    "confirmed": track.confirmed,
                    "rejected": track.rejected,
                    "consensus_geometry": (
                        observation.local_index in consensus_by_local_index
                    ),
                    "confirmation_views": confirmation_views if just_confirmed else 0,
                }
            )

        promoted = sorted(set(promoted))
        self._stats["promoted_observations"] += len(promoted)
        index = torch.as_tensor(
            promoted, dtype=torch.int64, device=low_instances.scores.device
        )
        recovered = low_instances[index]
        consensus_applied = self._apply_world_consensus_geometry(
            recovered,
            promoted,
            consensus_by_local_index,
            camera_to_world,
        )
        self._stats["consensus_geometry_promotions"] += consensus_applied
        self._events.append(
            {
                "frame_id": int(frame_id),
                "low_count": len(low_instances),
                "selected_count": len(observations),
                "promoted_local_indices": promoted,
                "consensus_geometry_local_indices": sorted(
                    consensus_by_local_index
                ),
                "assignments": frame_assignments,
            }
        )
        return recovered

    def finalize(self) -> dict[str, Any]:
        if self._finalized:
            return self.summary()
        if self.scene_id is None:
            raise RuntimeError("Boxer-MVPR was never bound to a scene")
        self.cache.verify_replay_complete(
            self.scene_id,
            baseline_prediction_path=self.baseline_prediction_root / f"{self.scene_id}_boxes.pkl",
        )
        self._finalized = True
        payload = self.summary()
        output_root = Path(self.config.diagnostics_dir)
        output_root.mkdir(parents=True, exist_ok=True)
        output_path = output_root / f"{self.scene_id}.json"
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{output_path.name}.", suffix=".tmp", dir=output_root
        )
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_name, output_path)
        finally:
            if os.path.exists(temporary_name):
                os.unlink(temporary_name)
        return payload

    def summary(self) -> dict[str, Any]:
        return {
            "schema": (
                SCHEMA_V2
                if (
                    self.config.rolling_consensus
                    or self.config.strong_two_view_confirmation
                )
                else SCHEMA
            ),
            "scene_id": self.scene_id,
            "enabled": True,
            "past_only": True,
            "query_against_committed_history": True,
            "minimum_distinct_views": (
                2
                if self.config.strong_two_view_confirmation
                else self.config.minimum_views
            ),
            "real_score_preserved": True,
            "native_rows_modified": False,
            "native_path_rng_isolated": True,
            "recovered_rows_enter_native_association_and_fusion": True,
            "config": asdict(self.config),
            "stats": dict(self._stats),
            "events": list(self._events),
        }


def build_boxer_mvpr(cfg: Mapping[str, Any], device: torch.device | str) -> BoxerMVPR | None:
    section = cfg.get("lifting", {}).get("boxer_mvpr", {})
    config = BoxerMVPRConfig.from_mapping(section)
    return BoxerMVPR(cfg, device) if config.enabled else None


__all__ = [
    "BoxerMVPR",
    "BoxerMVPRConfig",
    "SCHEMA",
    "SCHEMA_V2",
    "build_boxer_mvpr",
    "isolated_rng",
]
