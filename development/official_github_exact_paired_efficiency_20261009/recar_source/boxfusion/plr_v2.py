"""Causal proposal-recovery controls for the staged PLR-v2 experiment.

The variants in this module are cumulative and consume the same current-frame
post-NMS proposals as the frozen PLR implementation:

``assoc``
    One-to-one, medoid-referenced, scale-aware association.
``reliability``
    ``assoc`` plus causal geometry/view reliability for the bounded output
    ranking.
``score``
    ``reliability`` plus a fixed low-tail rank mapping for output scores.

No variant reads ground truth, future frames, NMS-suppressed children or a
complete-scene proposal cache.
"""
from __future__ import annotations

import math
from typing import Iterable

import numpy as np

from boxfusion.online_candidate_map import (
    OnlineProposalRecovery,
    ProposalBirth,
    _Observation,
    _ProposalTrack,
    _aabb_iou_matrix,
    _center,
    _corners,
    _vector,
)


class OnlineProposalRecoveryV2(OnlineProposalRecovery):
    """Staged PLR-v2 state used only by matched experimental controls."""

    STAGES = {"assoc", "reliability", "score"}

    def __init__(
        self,
        *,
        stage: str,
        match_center_ratio: float = 0.75,
        match_scale_log: float = 0.70,
        center_cost_weight: float = 0.50,
        scale_cost_weight: float = 0.25,
        min_view_separation_deg: float = 20.0,
        reliability_saturation: float = 2.0,
        score_min: float = 0.05,
        score_max: float = 0.50,
        **kwargs,
    ) -> None:
        if stage not in self.STAGES:
            raise ValueError(f"unknown PLR-v2 stage {stage!r}")
        if match_center_ratio <= 0.0 or match_scale_log < 0.0:
            raise ValueError("association gates must be nonnegative")
        if center_cost_weight < 0.0 or scale_cost_weight < 0.0:
            raise ValueError("association cost weights must be nonnegative")
        if not 0.0 <= min_view_separation_deg <= 180.0:
            raise ValueError("min_view_separation_deg must lie in [0,180]")
        if reliability_saturation <= 0.0:
            raise ValueError("reliability_saturation must be positive")
        if not 0.0 <= score_min < score_max < 1.0:
            raise ValueError("score interval must satisfy 0 <= min < max < 1")
        kwargs["use_children"] = False
        super().__init__(**kwargs)
        self.stage = stage
        self.match_center_ratio = float(match_center_ratio)
        self.match_scale_log = float(match_scale_log)
        self.center_cost_weight = float(center_cost_weight)
        self.scale_cost_weight = float(scale_cost_weight)
        self.min_view_separation_deg = float(min_view_separation_deg)
        self.reliability_saturation = float(reliability_saturation)
        self.score_min = float(score_min)
        self.score_max = float(score_max)
        self.view_directions: dict[int, dict[int, np.ndarray]] = {}
        self.track_reliability: dict[int, float] = {}
        self.association_matches = 0
        self.association_unmatched = 0
        self.association_edges = 0

    @staticmethod
    def _extent(corners: np.ndarray) -> np.ndarray:
        return np.maximum(np.ptp(corners, axis=0), 1.0e-6)

    def _assign_one_to_one(
        self, rows: list[_Observation]
    ) -> dict[int, _ProposalTrack]:
        """Return deterministic query-before-commit row-to-track matches."""
        tracks = sorted(self.tracks.values(), key=lambda value: value.track_id)
        if not rows or not tracks:
            return {}
        references = [self._medoid(track) for track in tracks]
        row_boxes = np.stack([row.corners for row in rows])
        ref_boxes = np.stack([row.corners for row in references])
        overlaps = _aabb_iou_matrix(row_boxes, ref_boxes)
        row_centers = np.stack([_center(row.corners) for row in rows])
        ref_centers = np.stack([_center(row.corners) for row in references])
        center_distance = np.linalg.norm(
            row_centers[:, None, :] - ref_centers[None, :, :], axis=2
        )
        ref_diagonal = np.maximum(
            np.linalg.norm(
                np.stack([self._extent(row.corners) for row in references]),
                axis=1,
            ),
            0.10,
        )
        normalized_center = center_distance / ref_diagonal[None, :]
        row_extent = np.stack([self._extent(row.corners) for row in rows])
        ref_extent = np.stack([self._extent(row.corners) for row in references])
        scale_error = np.mean(
            np.abs(np.log(row_extent[:, None, :] / ref_extent[None, :, :])),
            axis=2,
        )
        gate = (
            (overlaps >= self.match_iou)
            & (normalized_center <= self.match_center_ratio)
            & (scale_error <= self.match_scale_log)
        )
        row_indices, track_indices = np.where(gate)
        self.association_edges += len(row_indices)
        edges = []
        for row_index, track_index in zip(row_indices.tolist(), track_indices.tolist()):
            cost = (
                1.0
                - float(overlaps[row_index, track_index])
                + self.center_cost_weight
                * float(normalized_center[row_index, track_index])
                + self.scale_cost_weight * float(scale_error[row_index, track_index])
            )
            edges.append(
                (
                    cost,
                    -float(overlaps[row_index, track_index]),
                    float(normalized_center[row_index, track_index]),
                    float(scale_error[row_index, track_index]),
                    int(row_index),
                    int(tracks[track_index].track_id),
                    int(track_index),
                )
            )
        edges.sort()
        used_rows: set[int] = set()
        used_tracks: set[int] = set()
        assigned: dict[int, _ProposalTrack] = {}
        for *_, row_index, track_id, track_index in edges:
            if row_index in used_rows or track_id in used_tracks:
                continue
            used_rows.add(row_index)
            used_tracks.add(track_id)
            assigned[row_index] = tracks[track_index]
        self.association_matches += len(assigned)
        self.association_unmatched += len(rows) - len(assigned)
        return assigned

    def _effective_view_count(self, track_id: int) -> int:
        directions = self.view_directions.get(track_id, {})
        selected: list[np.ndarray] = []
        cosine = math.cos(math.radians(self.min_view_separation_deg))
        for frame_id in sorted(directions):
            direction = directions[frame_id]
            if not selected or all(float(np.dot(direction, row)) <= cosine for row in selected):
                selected.append(direction)
        return len(selected)

    def _reliability(self, track: _ProposalTrack) -> float:
        evidence = sorted(
            track.observations.values(),
            key=lambda row: (row.ordinal, row.source, row.source_id),
        )
        if not evidence:
            return 0.0
        boxes = np.stack([row.corners for row in evidence])
        medoid = self._medoid(track)
        medoid_overlap = _aabb_iou_matrix(
            boxes, np.asarray(medoid.corners).reshape(1, 8, 3)
        )[:, 0]
        reference_diagonal = max(
            float(np.linalg.norm(self._extent(medoid.corners))), 0.10
        )
        centers = np.stack([_center(row.corners) for row in evidence])
        center_rms = float(
            np.sqrt(np.mean(np.sum((centers - _center(medoid.corners)) ** 2, axis=1)))
        )
        center_stability = math.exp(-center_rms / reference_diagonal)
        extents = np.stack([self._extent(row.corners) for row in evidence])
        reference_extent = self._extent(medoid.corners)
        scale_error = float(
            np.mean(np.abs(np.log(extents / reference_extent[None, :])))
        )
        scale_stability = math.exp(-scale_error)
        geometry = float(
            np.clip(
                (
                    max(float(np.mean(medoid_overlap)), 1.0e-6)
                    * center_stability
                    * scale_stability
                )
                ** (1.0 / 3.0),
                0.0,
                1.0,
            )
        )
        proposal_quality = float(np.median([row.score for row in evidence]))
        effective_views = self._effective_view_count(track.track_id)
        support = effective_views / (effective_views + self.reliability_saturation)
        return float(np.clip(proposal_quality * geometry * support, 0.0, 1.0))

    def _record_view(
        self,
        track: _ProposalTrack,
        observation: _Observation,
        camera_to_world: np.ndarray,
    ) -> None:
        origin = np.asarray(camera_to_world, dtype=np.float64)[:3, 3]
        direction = _center(observation.corners) - origin
        norm = float(np.linalg.norm(direction))
        if norm <= 1.0e-8:
            return
        self.view_directions.setdefault(track.track_id, {})[
            observation.frame_id
        ] = direction / norm

    def _try_birth(
        self,
        track: _ProposalTrack,
        frame_id: int,
        native_corners: np.ndarray,
    ) -> ProposalBirth | None:
        birth = super()._try_birth(track, frame_id, native_corners)
        if track.track_id in self.births:
            self.track_reliability[track.track_id] = self._reliability(track)
        return birth

    def _birth_rank(self, birth: ProposalBirth) -> tuple:
        if self.stage == "assoc":
            return OnlineProposalRecovery._birth_rank(birth)
        return (
            -float(self.track_reliability.get(birth.track_id, 0.0)),
            -float(birth.raw_mean_score),
            int(birth.confirmation_frame_id),
            int(birth.track_id),
        )

    def _select_rows(
        self, *, native_corners: np.ndarray | None = None
    ) -> list[dict]:
        rows = super()._select_rows(native_corners=native_corners)
        for row in rows:
            row["reliability"] = float(
                self.track_reliability.get(int(row["track_id"]), 0.0)
            )
        if self.stage == "score" and rows:
            count = len(rows)
            for index, row in enumerate(rows):
                quantile = float(count - index) / float(count + 1)
                row["score"] = self.score_min + (
                    self.score_max - self.score_min
                ) * quantile
        return rows

    def update(
        self,
        ordinal: int,
        frame_id: int,
        *,
        proposal_ids: Iterable[int],
        proposal_corners: np.ndarray,
        proposal_scores: Iterable[float],
        native_corners: np.ndarray = np.empty((0, 8, 3)),
        camera_to_world: np.ndarray,
        **_ignored,
    ) -> list[ProposalBirth]:
        if ordinal != self.frames_seen:
            raise ValueError(f"expected ordinal {self.frames_seen}, got {ordinal}")
        if frame_id <= self.last_frame_id:
            raise ValueError("frame_id must increase strictly")
        native = _corners(native_corners, name="native_corners")
        boxes = _corners(proposal_corners, name="proposal_corners")
        ids = _vector(proposal_ids, len(boxes), name="proposal_ids", dtype=np.int64)
        scores = _vector(
            proposal_scores, len(boxes), name="proposal_scores", dtype=np.float64
        )
        if np.any((scores < 0.0) | (scores > 1.0)):
            raise ValueError("proposal_scores must lie in [0,1]")
        rows = [
            _Observation(
                frame_id=int(frame_id),
                ordinal=int(ordinal),
                source_id=int(source_id),
                source="proposal",
                score=float(score),
                corners=np.array(box, copy=True),
            )
            for source_id, box, score in zip(ids, boxes, scores)
        ]
        rows.sort(key=lambda row: (-row.score, row.source_id))
        if len(rows) > self.max_observations_per_frame:
            self.capacity_drops += len(rows) - self.max_observations_per_frame
            rows = rows[: self.max_observations_per_frame]

        if rows and len(native):
            self.native_drops += int(
                np.any(
                    _aabb_iou_matrix(
                        np.stack([row.corners for row in rows]), native
                    )
                    >= self.native_dedup_iou,
                    axis=1,
                ).sum()
            )

        self._expire(ordinal)
        self._retire_native_duplicates(native)
        assigned = self._assign_one_to_one(rows)
        born_now: list[ProposalBirth] = []
        for row_index, observation in enumerate(rows):
            track = assigned.get(row_index)
            if track is None:
                track = self._new_track()
            if track is None:
                continue
            track.add(observation, self.max_observations_per_track)
            self._record_view(track, observation, camera_to_world)
            birth = self._try_birth(track, int(frame_id), native)
            if birth is not None:
                born_now.append(birth)

        self._retire_native_duplicates(native)
        self.frames_seen += 1
        self.last_frame_id = int(frame_id)
        self.peak_tracks = max(self.peak_tracks, len(self.tracks))
        return born_now

    def diagnostics(self) -> dict:
        result = super().diagnostics()
        values = np.asarray(list(self.track_reliability.values()), dtype=np.float64)
        result.update(
            {
                "stage": self.stage,
                "association": "one_to_one_medoid_scale_aware",
                "association_matches": self.association_matches,
                "association_unmatched": self.association_unmatched,
                "association_edges": self.association_edges,
                "match_center_ratio": self.match_center_ratio,
                "match_scale_log": self.match_scale_log,
                "ranking": (
                    "raw_mean_score" if self.stage == "assoc" else "causal_reliability"
                ),
                "score_mode": (
                    "size_price" if self.stage != "score" else "reliability_rank_tail"
                ),
                "reliability_mean": float(values.mean()) if len(values) else 0.0,
                "reliability_max": float(values.max()) if len(values) else 0.0,
            }
        )
        return result

