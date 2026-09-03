"""Source-preserving grouped soft association for raw Boxer proposals.

The native score-0.5 Boxer path is left untouched.  This module replays the
bounded score-0.4 proposal cache, lifts only the low-score suffix with the same
frozen Boxer model, and exposes a current observation only after a causal
multi-view group has accumulated sufficient geometric support.  Geometry is
never averaged or replaced: grouping affects survival and score calibration,
not the frozen Boxer OBB.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping

import numpy as np
import torch

from boxfusion.instances import Instances3D
from boxfusion.sealed_boxer_proposal_cache import ProposalCache, ProposalCacheConfig


SCHEMA = "boxfusion.boxer_gsa.v1"


def _aabb(corners: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    lower = corners.min(axis=0)
    upper = corners.max(axis=0)
    extent = np.maximum(upper - lower, 1.0e-6)
    return lower, upper, 0.5 * (lower + upper), extent


def _aabb_iou(left: np.ndarray, right: np.ndarray) -> float:
    lo_a, hi_a, _, extent_a = _aabb(left)
    lo_b, hi_b, _, extent_b = _aabb(right)
    intersection = float(
        np.prod(np.maximum(np.minimum(hi_a, hi_b) - np.maximum(lo_a, lo_b), 0.0))
    )
    union = float(np.prod(extent_a) + np.prod(extent_b) - intersection)
    return intersection / union if union > 0.0 else 0.0


def _box_iou_2d(left: np.ndarray, right: np.ndarray) -> float:
    intersection_wh = np.maximum(
        np.minimum(left[2:], right[2:]) - np.maximum(left[:2], right[:2]),
        0.0,
    )
    intersection = float(np.prod(intersection_wh))
    area_left = float(np.prod(np.maximum(left[2:] - left[:2], 0.0)))
    area_right = float(np.prod(np.maximum(right[2:] - right[:2], 0.0)))
    union = area_left + area_right - intersection
    return intersection / union if union > 0.0 else 0.0


@dataclass(frozen=True)
class BoxerGSAConfig:
    enabled: bool = False
    low_score_min: float = 0.40
    native_score_min: float = 0.50
    max_candidates_per_frame: int = 8
    evidence_window: int = 5
    ttl_keyframes: int = 10
    native_duplicate_2d_iou: float = 0.995
    maximum_center_distance_m: float = 0.90
    minimum_pair_iou: float = 0.02
    minimum_center_similarity: float = 0.38
    minimum_size_similarity: float = 0.35
    minimum_affinity: float = 0.43
    two_view_quality: float = 0.70
    three_view_quality: float = 0.50
    maximum_group_center_rms_m: float = 0.45
    minimum_aabb_extent_m: float = 0.20
    score_support_bonus: float = 0.035
    score_quality_weight: float = 0.75
    diagnostics_dir: str = ""

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> "BoxerGSAConfig":
        values = {
            name: mapping.get(name, default)
            for name, default in (
                ("enabled", False),
                ("low_score_min", 0.40),
                ("native_score_min", 0.50),
                ("max_candidates_per_frame", 8),
                ("evidence_window", 5),
                ("ttl_keyframes", 10),
                ("native_duplicate_2d_iou", 0.995),
                ("maximum_center_distance_m", 0.90),
                ("minimum_pair_iou", 0.02),
                ("minimum_center_similarity", 0.38),
                ("minimum_size_similarity", 0.35),
                ("minimum_affinity", 0.43),
                ("two_view_quality", 0.70),
                ("three_view_quality", 0.50),
                ("maximum_group_center_rms_m", 0.45),
                ("minimum_aabb_extent_m", 0.20),
                ("score_support_bonus", 0.035),
                ("score_quality_weight", 0.75),
                ("diagnostics_dir", ""),
            )
        }
        config = cls(**values)
        if not 0.0 <= config.low_score_min < config.native_score_min <= 1.0:
            raise ValueError("Boxer-GSA score interval must satisfy 0 <= low < native <= 1")
        if not 1 <= config.max_candidates_per_frame <= 128:
            raise ValueError("Boxer-GSA max_candidates_per_frame must lie in [1,128]")
        if not 3 <= config.evidence_window <= 16:
            raise ValueError("Boxer-GSA evidence_window must lie in [3,16]")
        if not 1 <= config.ttl_keyframes <= 1000:
            raise ValueError("Boxer-GSA ttl_keyframes must lie in [1,1000]")
        for name in (
            "native_duplicate_2d_iou",
            "minimum_pair_iou",
            "minimum_center_similarity",
            "minimum_size_similarity",
            "minimum_affinity",
            "two_view_quality",
            "three_view_quality",
            "score_quality_weight",
        ):
            value = float(getattr(config, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"Boxer-GSA {name} must lie in [0,1]")
        for name in (
            "maximum_center_distance_m",
            "maximum_group_center_rms_m",
            "minimum_aabb_extent_m",
        ):
            value = float(getattr(config, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"Boxer-GSA {name} must be positive and finite")
        if not math.isfinite(float(config.score_support_bonus)) or config.score_support_bonus < 0.0:
            raise ValueError("Boxer-GSA score_support_bonus must be finite and nonnegative")
        if config.enabled and not config.diagnostics_dir:
            raise ValueError("Boxer-GSA diagnostics_dir is required when enabled")
        return config


@dataclass(frozen=True)
class _Observation:
    frame_id: int
    local_index: int
    score: float
    corners: np.ndarray

    @property
    def center(self) -> np.ndarray:
        return _aabb(self.corners)[2]

    @property
    def extent(self) -> np.ndarray:
        return _aabb(self.corners)[3]


@dataclass
class _Group:
    group_id: int
    last_keyframe_step: int
    evidence: list[_Observation] = field(default_factory=list)
    confirmed: bool = False
    quality: float = 0.0
    metrics: dict[str, float] = field(default_factory=dict)


class BoxerGSA:
    """Causal soft grouping and dynamic scoring for low-score Boxer OBBs."""

    def __init__(self, cfg: Mapping[str, Any], device: torch.device | str):
        section = cfg.get("lifting", {}).get("boxer_gsa", {})
        self.config = BoxerGSAConfig.from_mapping(section)
        if not self.config.enabled:
            raise ValueError("Do not construct BoxerGSA when it is disabled")
        cache_config = ProposalCacheConfig.from_mapping(section.get("proposal_cache", {}))
        if cache_config.mode != "replay":
            raise ValueError("Boxer-GSA requires a replay-only low-score proposal cache")
        self.cache = ProposalCache(cache_config, device=torch.device(device))
        self.baseline_prediction_root = Path(
            str(section.get("proposal_cache", {}).get("baseline_prediction_root", ""))
        )
        if not self.baseline_prediction_root.is_dir():
            raise ValueError("Boxer-GSA baseline_prediction_root must be an existing directory")
        self.device = torch.device(device)
        self.scene_id: str | None = None
        self._groups: dict[int, _Group] = {}
        self._next_group_id = 1
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
            "groups_created": 0,
            "groups_matched": 0,
            "groups_retired": 0,
            "groups_confirmed": 0,
            "groups_confirmed_two_view": 0,
            "groups_lost_gate": 0,
            "promoted_observations": 0,
            "scores_calibrated": 0,
        }

    def bind_scene(self, scene_id: str, *, dataset_length: int, gap: int) -> None:
        if self.scene_id is not None and self.scene_id != scene_id:
            raise ValueError("One Boxer-GSA instance cannot mix scenes")
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
            raise ValueError("Bind Boxer-GSA to the scene before replay")
        candidates, attempt_id = self.cache.replay(scene_id, frame_id, inputs=inputs)
        scores = candidates.scores.detach().float()
        interval = (scores >= self.config.low_score_min) & (
            scores < self.config.native_score_min
        )
        interval_indices = torch.nonzero(interval, as_tuple=False).flatten()
        self._stats["low_rows_replayed"] += len(candidates)
        self._stats["low_rows_in_interval"] += int(interval_indices.numel())
        low = candidates[interval_indices]
        if len(low) == 0 or len(native_instances) == 0:
            return low, attempt_id

        native_boxes = native_instances.pred_boxes.detach().float().cpu().numpy()
        low_boxes = low.pred_boxes.detach().float().cpu().numpy()
        keep = np.ones(len(low), dtype=bool)
        for index, low_box in enumerate(low_boxes):
            if any(
                _box_iou_2d(low_box, native_box) >= self.config.native_duplicate_2d_iou
                for native_box in native_boxes
            ):
                keep[index] = False
        self._stats["native_duplicates_removed"] += int((~keep).sum())
        index = torch.from_numpy(keep).to(low.scores.device)
        return low[index], attempt_id

    def _observations(
        self,
        frame_id: int,
        instances: Instances3D,
        camera_to_world: Any,
    ) -> list[_Observation]:
        if len(instances) == 0:
            return []
        scores = instances.scores.detach().float().cpu().numpy()
        selected = sorted(range(len(instances)), key=lambda index: (-float(scores[index]), index))
        selected = selected[: self.config.max_candidates_per_frame]
        boxes = instances.pred_boxes_3d.clone()
        pose = torch.as_tensor(camera_to_world).detach().float()
        if pose.ndim == 3:
            pose = pose[-1]
        boxes.transform2world(pose.unsqueeze(0).repeat(len(instances), 1, 1))
        corners = boxes.corners.detach().float().cpu().numpy()
        return [
            _Observation(
                frame_id=int(frame_id),
                local_index=int(index),
                score=float(scores[index]),
                corners=np.array(corners[index], dtype=np.float64, copy=True),
            )
            for index in selected
        ]

    @staticmethod
    def _reference(group: _Group) -> _Observation:
        costs = [
            sum(1.0 - _aabb_iou(row.corners, other.corners) for other in group.evidence)
            for row in group.evidence
        ]
        return group.evidence[min(range(len(costs)), key=lambda index: (costs[index], index))]

    def _affinity(self, observation: _Observation, reference: _Observation) -> tuple[bool, dict[str, float]]:
        iou = _aabb_iou(observation.corners, reference.corners)
        center_distance = float(np.linalg.norm(observation.center - reference.center))
        diagonal = max(
            0.5 * (float(np.linalg.norm(observation.extent)) + float(np.linalg.norm(reference.extent))),
            0.25,
        )
        center_similarity = float(math.exp(-center_distance / diagonal))
        size_delta = float(
            np.mean(np.abs(np.log(np.maximum(observation.extent, 1.0e-6) / np.maximum(reference.extent, 1.0e-6))))
        )
        size_similarity = float(math.exp(-size_delta))
        affinity = 0.45 * iou + 0.35 * center_similarity + 0.20 * size_similarity
        accepted = (
            center_distance <= self.config.maximum_center_distance_m
            and (iou >= self.config.minimum_pair_iou or center_similarity >= self.config.minimum_center_similarity)
            and size_similarity >= self.config.minimum_size_similarity
            and affinity >= self.config.minimum_affinity
        )
        return accepted, {
            "aabb_iou": iou,
            "center_distance_m": center_distance,
            "center_similarity": center_similarity,
            "size_similarity": size_similarity,
            "affinity": affinity,
        }

    def _group_quality(self, evidence: list[_Observation]) -> tuple[bool, float, dict[str, float]]:
        pairwise_iou: list[float] = []
        for left in range(len(evidence)):
            for right in range(left + 1, len(evidence)):
                pairwise_iou.append(_aabb_iou(evidence[left].corners, evidence[right].corners))
        centers = np.stack([row.center for row in evidence], axis=0)
        extents = np.stack([row.extent for row in evidence], axis=0)
        centroid = np.median(centers, axis=0)
        center_rms = float(np.sqrt(np.mean(np.sum((centers - centroid) ** 2, axis=1))))
        median_diagonal = max(float(np.median(np.linalg.norm(extents, axis=1))), 0.25)
        normalized_center_rms = center_rms / median_diagonal
        log_extents = np.log(np.maximum(extents, 1.0e-6))
        size_log_rms = float(
            np.sqrt(np.mean((log_extents - np.median(log_extents, axis=0)) ** 2))
        )
        median_iou = float(np.median(pairwise_iou))
        median_score = float(np.median([row.score for row in evidence]))
        score_reliability = float(
            np.clip(
                (median_score - self.config.low_score_min)
                / (self.config.native_score_min - self.config.low_score_min),
                0.0,
                1.0,
            )
        )
        quality = float(
            0.40 * median_iou
            + 0.30 * math.exp(-normalized_center_rms)
            + 0.20 * math.exp(-size_log_rms / 0.50)
            + 0.10 * score_reliability
        )
        support = len({row.frame_id for row in evidence})
        min_extent = float(np.min(np.median(extents, axis=0)))
        threshold = (
            self.config.two_view_quality if support == 2 else self.config.three_view_quality
        )
        accepted = (
            support >= 2
            and quality >= threshold
            and center_rms <= self.config.maximum_group_center_rms_m
            and min_extent >= self.config.minimum_aabb_extent_m
        )
        return accepted, quality, {
            "support_views": float(support),
            "median_pairwise_aabb_iou": median_iou,
            "center_rms_m": center_rms,
            "normalized_center_rms": normalized_center_rms,
            "size_log_rms": size_log_rms,
            "median_source_score": median_score,
            "minimum_median_extent_m": min_extent,
            "quality": quality,
            "quality_threshold": float(threshold),
        }

    def _calibrated_score(self, observation: _Observation, group: _Group) -> float:
        support = len({row.frame_id for row in group.evidence})
        evidence_bonus = self.config.score_support_bonus * (1.0 - math.exp(-(support - 1)))
        quality_factor = (1.0 - self.config.score_quality_weight) + self.config.score_quality_weight * group.quality
        score = observation.score * (0.90 + 0.10 * quality_factor) + evidence_bonus * group.quality
        return float(np.clip(score, self.config.low_score_min, self.config.native_score_min - 1.0e-4))

    def recover(
        self,
        frame_id: int,
        low_instances: Instances3D,
        *,
        camera_to_world: Any,
    ) -> Instances3D:
        if self._finalized:
            raise RuntimeError("Boxer-GSA is already finalized")
        if int(frame_id) <= self._last_frame_id:
            raise ValueError("Boxer-GSA frame IDs must increase")
        self._last_frame_id = int(frame_id)
        step = self._keyframe_step
        self._keyframe_step += 1
        self._stats["keyframes"] += 1

        retired = [
            group_id
            for group_id, group in self._groups.items()
            if step - group.last_keyframe_step > self.config.ttl_keyframes
        ]
        for group_id in retired:
            del self._groups[group_id]
        self._stats["groups_retired"] += len(retired)

        observations = self._observations(frame_id, low_instances, camera_to_world)
        self._stats["observations_selected"] += len(observations)
        used_groups: set[int] = set()
        promoted: list[int] = []
        scores_by_index: dict[int, float] = {}
        assignments: list[dict[str, Any]] = []

        for observation in observations:
            candidates: list[tuple[float, float, int, dict[str, float]]] = []
            for group_id, group in self._groups.items():
                if group_id in used_groups:
                    continue
                accepted, pair_metrics = self._affinity(observation, self._reference(group))
                if accepted:
                    candidates.append(
                        (-pair_metrics["affinity"], pair_metrics["center_distance_m"], group_id, pair_metrics)
                    )
            if candidates:
                _, _, group_id, pair_metrics = min(candidates)
                group = self._groups[group_id]
                group.last_keyframe_step = step
                group.evidence.append(observation)
                group.evidence = group.evidence[-self.config.evidence_window :]
                self._stats["groups_matched"] += 1
                action = "matched"
            else:
                group_id = self._next_group_id
                self._next_group_id += 1
                group = _Group(group_id=group_id, last_keyframe_step=step, evidence=[observation])
                self._groups[group_id] = group
                pair_metrics = {
                    "aabb_iou": 0.0,
                    "center_distance_m": 0.0,
                    "center_similarity": 1.0,
                    "size_similarity": 1.0,
                    "affinity": 1.0,
                }
                self._stats["groups_created"] += 1
                action = "created"
            used_groups.add(group_id)

            was_confirmed = group.confirmed
            if len(group.evidence) >= 2:
                accepted, quality, metrics = self._group_quality(group.evidence)
                group.confirmed = accepted
                group.quality = quality
                group.metrics = metrics
                if accepted and not was_confirmed:
                    self._stats["groups_confirmed"] += 1
                    if int(metrics["support_views"]) == 2:
                        self._stats["groups_confirmed_two_view"] += 1
                elif was_confirmed and not accepted:
                    self._stats["groups_lost_gate"] += 1
            if group.confirmed:
                promoted.append(observation.local_index)
                scores_by_index[observation.local_index] = self._calibrated_score(observation, group)

            assignments.append(
                {
                    "local_index": observation.local_index,
                    "source_score": observation.score,
                    "group_id": group_id,
                    "action": action,
                    "confirmed": group.confirmed,
                    "quality": group.quality,
                    "pair": pair_metrics,
                    "group_metrics": dict(group.metrics),
                }
            )

        promoted = sorted(set(promoted))
        index = torch.as_tensor(promoted, dtype=torch.int64, device=low_instances.scores.device)
        recovered = low_instances[index]
        if promoted:
            recovered.scores = torch.as_tensor(
                [scores_by_index[row] for row in promoted],
                dtype=low_instances.scores.dtype,
                device=low_instances.scores.device,
            )
        self._stats["promoted_observations"] += len(promoted)
        self._stats["scores_calibrated"] += len(promoted)
        self._events.append(
            {
                "frame_id": int(frame_id),
                "low_count": len(low_instances),
                "selected_count": len(observations),
                "promoted_local_indices": promoted,
                "promoted_scores": [scores_by_index[row] for row in promoted],
                "assignments": assignments,
            }
        )
        return recovered

    def finalize(self) -> dict[str, Any]:
        if self._finalized:
            return self.summary()
        if self.scene_id is None:
            raise RuntimeError("Boxer-GSA was never bound to a scene")
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
            "schema": SCHEMA,
            "scene_id": self.scene_id,
            "enabled": True,
            "past_only": True,
            "source_geometry_preserved": True,
            "native_rows_modified": False,
            "dynamic_group_score": True,
            "recovered_rows_enter_native_association_and_fusion": True,
            "config": asdict(self.config),
            "stats": dict(self._stats),
            "events": list(self._events),
        }


def build_boxer_gsa(cfg: Mapping[str, Any], device: torch.device | str) -> BoxerGSA | None:
    section = cfg.get("lifting", {}).get("boxer_gsa", {})
    config = BoxerGSAConfig.from_mapping(section)
    return BoxerGSA(cfg, device) if config.enabled else None


__all__ = ["BoxerGSA", "BoxerGSAConfig", "SCHEMA", "build_boxer_gsa"]
