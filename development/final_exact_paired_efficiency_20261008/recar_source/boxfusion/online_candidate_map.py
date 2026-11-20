"""Causal online candidate recovery and native-map reranking.

The state in this module consumes exactly one keyframe per :meth:`update`.
It never reads a terminal map, a future frame, ground truth, or a full-scene
proposal cache.  M1-P confirms proposal/child tracks as soon as enough
distinct keyframes support them, M1-A delegates to the existing bounded
voxel state, and M2 maintains a running maximum of current-frame proposal
support for each native row.

Recovered rows are deliberately kept outside BoxFusion's native association
state.  They are assembled with the current native map for output, so the
add-on cannot change the host mapper's future association decisions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import time
from typing import Iterable, Mapping, Sequence

import numpy as np

from boxfusion.m1_anchor_online import OnlineAnchorRecovery


SCHEMA = "boxfusion.online_candidate_map.v1"


def _corners(value: object, *, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.size == 0:
        return np.empty((0, 8, 3), dtype=np.float64)
    try:
        array = array.reshape(-1, 8, 3)
    except ValueError as error:
        raise ValueError(f"{name} must have shape [N,8,3]") from error
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite")
    if np.any(np.ptp(array, axis=1) <= 0.0):
        raise ValueError(f"{name} contains a degenerate box")
    return np.array(array, copy=True)


def _vector(value: object, length: int, *, name: str, dtype) -> np.ndarray:
    array = np.asarray(value, dtype=dtype).reshape(-1)
    if len(array) != length:
        raise ValueError(f"{name} must contain {length} values")
    if np.issubdtype(array.dtype, np.floating) and not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite")
    return np.array(array, copy=True)


def _aabb_iou(left: np.ndarray, right: np.ndarray) -> float:
    left_low, left_high = left.min(axis=0), left.max(axis=0)
    right_low, right_high = right.min(axis=0), right.max(axis=0)
    extent = np.maximum(
        np.minimum(left_high, right_high) - np.maximum(left_low, right_low),
        0.0,
    )
    intersection = float(np.prod(extent))
    union = (
        float(np.prod(left_high - left_low))
        + float(np.prod(right_high - right_low))
        - intersection
    )
    return 0.0 if union <= 0.0 else intersection / union


def _aabb_iou_matrix(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Vectorized AABB IoU for two corner-box collections."""
    left = np.asarray(left, dtype=np.float64).reshape(-1, 8, 3)
    right = np.asarray(right, dtype=np.float64).reshape(-1, 8, 3)
    if len(left) == 0 or len(right) == 0:
        return np.empty((len(left), len(right)), dtype=np.float64)
    left_low, left_high = left.min(axis=1), left.max(axis=1)
    right_low, right_high = right.min(axis=1), right.max(axis=1)
    extent = np.maximum(
        np.minimum(left_high[:, None], right_high[None])
        - np.maximum(left_low[:, None], right_low[None]),
        0.0,
    )
    intersection = np.prod(extent, axis=2)
    left_volume = np.prod(left_high - left_low, axis=1)
    right_volume = np.prod(right_high - right_low, axis=1)
    union = left_volume[:, None] + right_volume[None] - intersection
    return np.divide(
        intersection,
        union,
        out=np.zeros_like(intersection),
        where=union > 0.0,
    )


def _center(corners: np.ndarray) -> np.ndarray:
    return 0.5 * (corners.min(axis=0) + corners.max(axis=0))


def _size_price(corners: np.ndarray) -> float:
    """Frozen M1-P score bands used by the final scene-end implementation."""
    maximum_extent = float(np.ptp(corners, axis=0).max())
    for boundary, score in zip(
        (0.30, 0.50, 0.70, 1.00, math.inf),
        (0.05, 0.10, 0.25, 0.40, 0.50),
    ):
        if maximum_extent < boundary:
            return score
    raise AssertionError("unreachable")


def unique_native_ids(fusion_lineages) -> np.ndarray:
    """Choose one stable, unique source identity for each native-map row.

    BoxFusion lineages can overlap after a split or reassociation.  Processing
    the most constrained rows first preserves the usual minimum-source ID for
    disjoint lineages while allowing overlapping rows to use another source ID.
    A reserved negative identity is used only when a lineage is fully occupied.
    """
    candidates = []
    for row_index, lineage in enumerate(fusion_lineages):
        values = tuple(sorted({int(value) for value in lineage}))
        if not values:
            raise ValueError(f"native fusion lineage {row_index} is empty")
        candidates.append(values)

    order = sorted(
        range(len(candidates)),
        key=lambda index: (len(candidates[index]), candidates[index], index),
    )
    assigned = [None] * len(candidates)
    used = set()
    reserved = -(1 << 62)
    for index in order:
        identity = next(
            (value for value in candidates[index] if value not in used),
            None,
        )
        if identity is None:
            while reserved in used:
                reserved -= 1
            identity = reserved
            reserved -= 1
        assigned[index] = identity
        used.add(identity)

    return np.asarray(assigned, dtype=np.int64)


def _project_xyxy(
    corners_world: np.ndarray,
    camera_to_world: np.ndarray,
    intrinsic: np.ndarray,
    width: int,
    height: int,
) -> np.ndarray | None:
    world_to_camera = np.linalg.inv(camera_to_world)
    camera = (
        corners_world @ world_to_camera[:3, :3].T
        + world_to_camera[:3, 3]
    )
    if np.any(camera[:, 2] < 0.1):
        return None
    uvw = camera @ intrinsic[:3, :3].T
    uv = uvw[:, :2] / uvw[:, 2:3]
    low, high = uv.min(axis=0), uv.max(axis=0)
    if (
        high[0] <= 0
        or high[1] <= 0
        or low[0] >= width
        or low[1] >= height
    ):
        return None
    box = np.asarray(
        [
            max(0.0, low[0]),
            max(0.0, low[1]),
            min(float(width), high[0]),
            min(float(height), high[1]),
        ],
        dtype=np.float64,
    )
    if box[2] - box[0] < 2.0 or box[3] - box[1] < 2.0:
        return None
    return box


def _iou2d_matrix(boxes: np.ndarray, proposals: np.ndarray) -> np.ndarray:
    if len(boxes) == 0 or len(proposals) == 0:
        return np.empty((len(boxes), len(proposals)), dtype=np.float64)
    low = np.maximum(boxes[:, None, :2], proposals[None, :, :2])
    high = np.minimum(boxes[:, None, 2:], proposals[None, :, 2:])
    intersection = np.prod(np.maximum(high - low, 0.0), axis=2)
    box_area = np.prod(boxes[:, 2:] - boxes[:, :2], axis=1)
    proposal_area = np.prod(proposals[:, 2:] - proposals[:, :2], axis=1)
    union = box_area[:, None] + proposal_area[None, :] - intersection
    return np.divide(
        intersection,
        union,
        out=np.zeros_like(intersection),
        where=union > 0.0,
    )


@dataclass(frozen=True)
class EvidenceFrame:
    """Frozen evidence produced from one current RGB-D keyframe."""

    proposal_ids: np.ndarray
    proposal_boxes_2d: np.ndarray
    proposal_corners: np.ndarray
    proposal_scores: np.ndarray
    anchor_ids: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=np.int64)
    )
    anchor_corners: np.ndarray = field(
        default_factory=lambda: np.empty((0, 8, 3), dtype=np.float64)
    )
    anchor_scores: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=np.float64)
    )
    child_ids: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=np.int64)
    )
    child_corners: np.ndarray = field(
        default_factory=lambda: np.empty((0, 8, 3), dtype=np.float64)
    )
    child_scores: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=np.float64)
    )

    def __post_init__(self) -> None:
        proposal_corners = _corners(
            self.proposal_corners, name="proposal_corners"
        )
        proposal_ids = _vector(
            self.proposal_ids,
            len(proposal_corners),
            name="proposal_ids",
            dtype=np.int64,
        )
        proposal_scores = _vector(
            self.proposal_scores,
            len(proposal_corners),
            name="proposal_scores",
            dtype=np.float64,
        )
        boxes_2d = np.asarray(self.proposal_boxes_2d, dtype=np.float64)
        if boxes_2d.size == 0:
            boxes_2d = np.empty((0, 4), dtype=np.float64)
        try:
            boxes_2d = boxes_2d.reshape(-1, 4)
        except ValueError as error:
            raise ValueError("proposal_boxes_2d must have shape [N,4]") from error
        if len(boxes_2d) != len(proposal_corners):
            raise ValueError("proposal 2D and 3D rows must stay aligned")
        if not np.isfinite(boxes_2d).all() or np.any(
            boxes_2d[:, 2:] <= boxes_2d[:, :2]
        ):
            raise ValueError("proposal_boxes_2d contains an invalid rectangle")

        anchor_corners = _corners(self.anchor_corners, name="anchor_corners")
        anchor_ids = _vector(
            self.anchor_ids,
            len(anchor_corners),
            name="anchor_ids",
            dtype=np.int64,
        )
        anchor_scores = _vector(
            self.anchor_scores,
            len(anchor_corners),
            name="anchor_scores",
            dtype=np.float64,
        )
        child_corners = _corners(self.child_corners, name="child_corners")
        child_ids = _vector(
            self.child_ids,
            len(child_corners),
            name="child_ids",
            dtype=np.int64,
        )
        child_scores = _vector(
            self.child_scores,
            len(child_corners),
            name="child_scores",
            dtype=np.float64,
        )
        for name, scores in (
            ("proposal_scores", proposal_scores),
            ("anchor_scores", anchor_scores),
            ("child_scores", child_scores),
        ):
            if np.any((scores < 0.0) | (scores > 1.0)):
                raise ValueError(f"{name} must lie in [0,1]")
        for name, value in (
            ("proposal_ids", proposal_ids),
            ("proposal_boxes_2d", boxes_2d),
            ("proposal_corners", proposal_corners),
            ("proposal_scores", proposal_scores),
            ("anchor_ids", anchor_ids),
            ("anchor_corners", anchor_corners),
            ("anchor_scores", anchor_scores),
            ("child_ids", child_ids),
            ("child_corners", child_corners),
            ("child_scores", child_scores),
        ):
            object.__setattr__(self, name, np.array(value, copy=True))


@dataclass(frozen=True)
class NativeFrame:
    ids: np.ndarray
    corners: np.ndarray
    scores: np.ndarray
    camera_to_world: np.ndarray
    intrinsic: np.ndarray
    width: int
    height: int

    def __post_init__(self) -> None:
        corners = _corners(self.corners, name="native_corners")
        ids = _vector(self.ids, len(corners), name="native_ids", dtype=np.int64)
        scores = _vector(
            self.scores, len(corners), name="native_scores", dtype=np.float64
        )
        pose = np.asarray(self.camera_to_world, dtype=np.float64)
        intrinsic = np.asarray(self.intrinsic, dtype=np.float64)
        if pose.shape != (4, 4) or not np.isfinite(pose).all():
            raise ValueError("camera_to_world must be a finite [4,4] matrix")
        if intrinsic.shape not in ((3, 3), (4, 4)) or not np.isfinite(
            intrinsic
        ).all():
            raise ValueError("intrinsic must be a finite [3,3] or [4,4] matrix")
        if len(set(ids.tolist())) != len(ids):
            raise ValueError("native_ids must be unique in a keyframe")
        if np.any((scores < 0.0) | (scores > 1.0)):
            raise ValueError("native_scores must lie in [0,1]")
        if int(self.width) < 1 or int(self.height) < 1:
            raise ValueError("image size must be positive")
        object.__setattr__(self, "ids", ids)
        object.__setattr__(self, "corners", corners)
        object.__setattr__(self, "scores", scores)
        object.__setattr__(self, "camera_to_world", np.array(pose, copy=True))
        object.__setattr__(self, "intrinsic", np.array(intrinsic, copy=True))
        object.__setattr__(self, "width", int(self.width))
        object.__setattr__(self, "height", int(self.height))


@dataclass
class _Observation:
    frame_id: int
    ordinal: int
    source_id: int
    source: str
    score: float
    corners: np.ndarray


@dataclass
class _ProposalTrack:
    track_id: int
    observations: dict[int, _Observation] = field(default_factory=dict)
    last_ordinal: int = -1

    def add(self, observation: _Observation, max_observations: int) -> None:
        current = self.observations.get(observation.frame_id)
        candidate_rank = (-observation.score, observation.source, observation.source_id)
        current_rank = (
            (-current.score, current.source, current.source_id)
            if current is not None
            else None
        )
        if current_rank is None or candidate_rank < current_rank:
            self.observations[observation.frame_id] = observation
        self.last_ordinal = max(self.last_ordinal, observation.ordinal)
        if len(self.observations) > max_observations:
            oldest = min(
                self.observations.values(),
                key=lambda row: (row.ordinal, row.frame_id),
            )
            del self.observations[oldest.frame_id]

    @property
    def latest(self) -> _Observation:
        return max(
            self.observations.values(),
            key=lambda row: (row.ordinal, row.frame_id),
        )


@dataclass(frozen=True)
class ProposalBirth:
    track_id: int
    confirmation_frame_id: int
    corners: np.ndarray
    score: float
    evidence_frame_ids: tuple[int, ...]
    evidence_sources: tuple[str, ...]
    raw_mean_score: float


class OnlineProposalRecovery:
    """Bounded causal M1-P state with revisable ranked output.

    Confirmation remains causal, but it is not an irreversible admission into
    the output budget.  Confirmed tracks keep absorbing later observations and
    the currently strongest non-native candidates are selected on every
    keyframe.  This prevents early or subsequently covered births from
    permanently consuming the per-scene output budget.
    """

    def __init__(
        self,
        *,
        use_children: bool = True,
        proposal_min_views: int = 3,
        child_min_views: int = 2,
        ttl_keyframes: int = 10,
        match_iou: float = 0.10,
        match_center_m: float = 0.50,
        native_dedup_iou: float = 0.25,
        self_nms_iou: float = 0.50,
        max_tracks: int = 1024,
        max_births: int = 12,
        max_observations_per_track: int = 12,
        max_observations_per_frame: int = 192,
    ) -> None:
        if (
            proposal_min_views < 1
            or child_min_views < 1
            or ttl_keyframes < 0
            or max_tracks < 1
            or max_births < 1
            or max_observations_per_track < 1
            or max_observations_per_frame < 1
        ):
            raise ValueError("invalid M1-P bounds")
        for name, value in (
            ("match_iou", match_iou),
            ("native_dedup_iou", native_dedup_iou),
            ("self_nms_iou", self_nms_iou),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must lie in [0,1]")
        if match_center_m <= 0.0:
            raise ValueError("match_center_m must be positive")
        self.use_children = bool(use_children)
        self.proposal_min_views = int(proposal_min_views)
        self.child_min_views = int(child_min_views)
        self.ttl_keyframes = int(ttl_keyframes)
        self.match_iou = float(match_iou)
        self.match_center_m = float(match_center_m)
        self.native_dedup_iou = float(native_dedup_iou)
        self.self_nms_iou = float(self_nms_iou)
        self.max_tracks = int(max_tracks)
        self.max_births = int(max_births)
        self.max_observations_per_track = int(max_observations_per_track)
        self.max_observations_per_frame = int(max_observations_per_frame)
        # ``max_births`` is the current output budget.  Keep a larger but still
        # bounded pool of confirmed candidates so later, stronger evidence can
        # replace an early weak output without using future observations.
        self.max_confirmed_candidates = max(self.max_tracks, self.max_births)
        self.tracks: dict[int, _ProposalTrack] = {}
        self.births: dict[int, ProposalBirth] = {}
        self.retired_births: set[int] = set()
        self.next_track_id = 0
        self.frames_seen = 0
        self.last_frame_id = -1
        self.peak_tracks = 0
        self.capacity_drops = 0
        self.native_drops = 0
        self.self_nms_drops = 0
        self.retired_native_overlap = 0
        self.birth_revisions = 0

    def _required_views(self, track: _ProposalTrack) -> int:
        sources = [row.source for row in track.observations.values()]
        proposal_count = sum(source == "proposal" for source in sources)
        return (
            self.proposal_min_views
            if 2 * proposal_count > len(sources)
            else self.child_min_views
        )

    @staticmethod
    def _medoid(track: _ProposalTrack) -> _Observation:
        observations = sorted(
            track.observations.values(),
            key=lambda row: (row.ordinal, row.source, row.source_id),
        )
        overlaps = _aabb_iou_matrix(
            np.stack([row.corners for row in observations]),
            np.stack([row.corners for row in observations]),
        ).sum(axis=1)
        return max(
            enumerate(observations),
            key=lambda item: (
                float(overlaps[item[0]]),
                item[1].score,
                -item[1].ordinal,
                -item[1].source_id,
            ),
        )[1]

    def _overlaps_native(
        self, corners: np.ndarray, native_corners: np.ndarray
    ) -> bool:
        return bool(
            np.any(
                _aabb_iou_matrix(
                    np.asarray(corners).reshape(1, 8, 3), native_corners
                )
                >= self.native_dedup_iou
            )
        )

    def _expire(self, ordinal: int) -> None:
        expired = [
            track_id
            for track_id, track in self.tracks.items()
            if ordinal - track.last_ordinal > self.ttl_keyframes
        ]
        for track_id in expired:
            del self.tracks[track_id]

    def _retire_native_duplicates(self, native_corners: np.ndarray) -> None:
        track_ids = list(self.births)
        if track_ids and len(native_corners):
            boxes = np.stack([self.births[track_id].corners for track_id in track_ids])
            mask = np.any(
                _aabb_iou_matrix(boxes, native_corners)
                >= self.native_dedup_iou,
                axis=1,
            )
            covered = {
                track_id for track_id, value in zip(track_ids, mask) if value
            }
        else:
            covered = set()
        self.retired_native_overlap += len(covered - self.retired_births)
        # Coverage by the native map is a current state, not a permanent
        # decision.  A candidate becomes visible again if a transient native
        # row disappears at a later keyframe.
        self.retired_births = covered

    def _match(self, observation: _Observation) -> _ProposalTrack | None:
        tracks = list(self.tracks.values())
        if not tracks:
            return None
        center = _center(observation.corners)
        latest = [track.latest for track in tracks]
        boxes = np.stack([row.corners for row in latest])
        distances = np.linalg.norm(
            np.stack([_center(row.corners) for row in latest]) - center,
            axis=1,
        )
        overlaps = _aabb_iou_matrix(
            boxes, np.asarray(observation.corners).reshape(1, 8, 3)
        )[:, 0]
        choices = [
            (-float(overlap), float(distance), track.track_id, track)
            for track, distance, overlap in zip(tracks, distances, overlaps)
            if distance <= self.match_center_m and overlap >= self.match_iou
        ]
        return min(choices)[-1] if choices else None

    def _new_track(self) -> _ProposalTrack | None:
        if len(self.tracks) >= self.max_tracks:
            self.capacity_drops += 1
            return None
        track = _ProposalTrack(track_id=self.next_track_id)
        self.next_track_id += 1
        self.tracks[track.track_id] = track
        return track

    def _try_birth(
        self,
        track: _ProposalTrack,
        frame_id: int,
        native_corners: np.ndarray,
    ) -> ProposalBirth | None:
        if len(track.observations) < self._required_views(track):
            return None
        medoid = self._medoid(track)
        existing = self.births.get(track.track_id)
        if existing is None and len(self.births) >= self.max_confirmed_candidates:
            self.capacity_drops += 1
            del self.tracks[track.track_id]
            return None
        evidence = sorted(
            track.observations.values(),
            key=lambda row: (row.ordinal, row.source, row.source_id),
        )
        birth = ProposalBirth(
            track_id=track.track_id,
            confirmation_frame_id=(
                int(frame_id)
                if existing is None
                else existing.confirmation_frame_id
            ),
            corners=np.array(medoid.corners, copy=True),
            score=_size_price(medoid.corners),
            evidence_frame_ids=tuple(row.frame_id for row in evidence),
            evidence_sources=tuple(row.source for row in evidence),
            raw_mean_score=float(np.mean([row.score for row in evidence])),
        )
        self.births[track.track_id] = birth
        if existing is not None:
            self.birth_revisions += 1
            return None
        return birth

    @staticmethod
    def _birth_rank(birth: ProposalBirth) -> tuple[float, int, int]:
        return (
            -float(birth.raw_mean_score),
            int(birth.confirmation_frame_id),
            int(birth.track_id),
        )

    def _select_rows(
        self,
        *,
        native_corners: np.ndarray | None = None,
    ) -> list[dict]:
        if native_corners is None:
            candidates = [
                birth
                for track_id, birth in self.births.items()
                if track_id not in self.retired_births
            ]
        else:
            native = np.asarray(native_corners, dtype=np.float64).reshape(-1, 8, 3)
            candidates = [
                birth
                for birth in self.births.values()
                if not self._overlaps_native(birth.corners, native)
            ]
        candidates.sort(key=self._birth_rank)

        selected: list[ProposalBirth] = []
        selected_indices: list[int] = []
        hidden_by_self_nms = 0
        overlaps = (
            _aabb_iou_matrix(
                np.stack([birth.corners for birth in candidates]),
                np.stack([birth.corners for birth in candidates]),
            )
            if candidates
            else np.empty((0, 0), dtype=np.float64)
        )
        for index, birth in enumerate(candidates):
            if selected_indices and np.any(
                overlaps[index, selected_indices] >= self.self_nms_iou
            ):
                hidden_by_self_nms += 1
                continue
            selected.append(birth)
            selected_indices.append(index)
            if len(selected) >= self.max_births:
                break
        self.self_nms_drops = hidden_by_self_nms
        return [
            {
                "track_id": birth.track_id,
                "box": np.array(birth.corners, copy=True),
                "score": float(birth.score),
                "confirmation_frame_id": birth.confirmation_frame_id,
                "support_frames": len(birth.evidence_frame_ids),
                "evidence_frame_ids": birth.evidence_frame_ids,
                "evidence_sources": birth.evidence_sources,
                "raw_mean_score": birth.raw_mean_score,
            }
            for birth in selected
        ]

    def update(
        self,
        ordinal: int,
        frame_id: int,
        *,
        proposal_ids: Iterable[int],
        proposal_corners: np.ndarray,
        proposal_scores: Iterable[float],
        child_ids: Iterable[int] = (),
        child_corners: np.ndarray = np.empty((0, 8, 3)),
        child_scores: Iterable[float] = (),
        native_corners: np.ndarray = np.empty((0, 8, 3)),
    ) -> list[ProposalBirth]:
        if ordinal != self.frames_seen:
            raise ValueError(f"expected ordinal {self.frames_seen}, got {ordinal}")
        if frame_id <= self.last_frame_id:
            raise ValueError("frame_id must increase strictly")
        native = _corners(native_corners, name="native_corners")
        rows = []
        sources = [("proposal", proposal_ids, proposal_corners, proposal_scores)]
        if self.use_children:
            sources.append(("child", child_ids, child_corners, child_scores))
        for source, ids_value, corners_value, scores_value in sources:
            boxes = _corners(corners_value, name=f"{source}_corners")
            ids = _vector(ids_value, len(boxes), name=f"{source}_ids", dtype=np.int64)
            scores = _vector(
                scores_value, len(boxes), name=f"{source}_scores", dtype=np.float64
            )
            if np.any((scores < 0.0) | (scores > 1.0)):
                raise ValueError(f"{source}_scores must lie in [0,1]")
            rows.extend(
                _Observation(
                    frame_id=int(frame_id),
                    ordinal=int(ordinal),
                    source_id=int(source_id),
                    source=source,
                    score=float(score),
                    corners=np.array(box, copy=True),
                )
                for source_id, box, score in zip(ids, boxes, scores)
            )
        rows.sort(key=lambda row: (-row.score, row.source, row.source_id))
        if len(rows) > self.max_observations_per_frame:
            self.capacity_drops += len(rows) - self.max_observations_per_frame
            rows = rows[: self.max_observations_per_frame]

        if rows and len(native):
            row_boxes = np.stack([row.corners for row in rows])
            self.native_drops += int(
                np.any(
                    _aabb_iou_matrix(row_boxes, native)
                    >= self.native_dedup_iou,
                    axis=1,
                ).sum()
            )

        self._expire(ordinal)
        self._retire_native_duplicates(native)
        born_now = []
        for observation in rows:
            track = self._match(observation)
            if track is None:
                track = self._new_track()
            if track is None:
                continue
            track.add(observation, self.max_observations_per_track)
            birth = self._try_birth(track, int(frame_id), native)
            if birth is not None:
                born_now.append(birth)

        # Newly confirmed candidates may already be covered by the current
        # native map.  Keep them as bounded shadow state and exclude them only
        # from the current output selection.
        self._retire_native_duplicates(native)

        self.frames_seen += 1
        self.last_frame_id = int(frame_id)
        self.peak_tracks = max(self.peak_tracks, len(self.tracks))
        return born_now

    def rows(self) -> list[dict]:
        return self._select_rows()

    def terminal_rows(self, native_corners: np.ndarray) -> list[dict]:
        """Read confirmed births against the host mapper's final native map.

        A birth hidden earlier by a transient native row may become visible if
        that row is removed by the host mapper's normal terminal validity
        filter.  This is a readout only: it performs no proposal inference and
        does not alter the causal recovery state.
        """
        return self._select_rows(native_corners=native_corners)

    def diagnostics(self) -> dict:
        return {
            "use_children": self.use_children,
            "frames_seen": self.frames_seen,
            "active_tracks": len(self.tracks),
            "births_total": len(self.births),
            "births_active": len(self.rows()),
            "births_retired_native_overlap": self.retired_native_overlap,
            "peak_tracks": self.peak_tracks,
            "max_tracks": self.max_tracks,
            "max_births": self.max_births,
            "max_confirmed_candidates": self.max_confirmed_candidates,
            "capacity_drops": self.capacity_drops,
            "native_drops": self.native_drops,
            "self_nms_drops": self.self_nms_drops,
            "birth_revisions": self.birth_revisions,
        }


@dataclass
class _NativeScoreState:
    max_support: float = 0.0
    last_seen_ordinal: int = -1


class OnlineNativeReranker:
    """Causal M2 with running max support and native-only score updates."""

    def __init__(
        self,
        *,
        support_threshold: float = 0.50,
        logit_weight: float = 2.0,
        match_iou: float = 0.10,
        exclusive_matching: bool = True,
        state_ttl_keyframes: int = 10,
        max_states: int = 4096,
    ) -> None:
        if not 0.0 <= support_threshold <= 1.0:
            raise ValueError("support_threshold must lie in [0,1]")
        if logit_weight < 0.0 or not math.isfinite(logit_weight):
            raise ValueError("logit_weight must be finite and nonnegative")
        if not 0.0 <= match_iou <= 1.0:
            raise ValueError("match_iou must lie in [0,1]")
        if state_ttl_keyframes < 0 or max_states < 1:
            raise ValueError("invalid M2 state bounds")
        self.support_threshold = float(support_threshold)
        self.logit_weight = float(logit_weight)
        self.match_iou = float(match_iou)
        self.exclusive_matching = bool(exclusive_matching)
        self.state_ttl_keyframes = int(state_ttl_keyframes)
        self.max_states = int(max_states)
        self.states: dict[int, _NativeScoreState] = {}
        self.frames_seen = 0
        self.last_frame_id = -1
        self.peak_states = 0
        self.capacity_drops = 0
        self.last_ids = np.empty(0, dtype=np.int64)
        self.last_scores = np.empty(0, dtype=np.float64)
        self.last_support = np.empty(0, dtype=np.float64)

    def _prune(self, ordinal: int, current_ids: set[int]) -> None:
        expired = [
            identity
            for identity, state in self.states.items()
            if identity not in current_ids
            and ordinal - state.last_seen_ordinal > self.state_ttl_keyframes
        ]
        for identity in expired:
            del self.states[identity]

    def _enforce_cap(self, current_ids: set[int]) -> None:
        overflow = len(self.states) - self.max_states
        if overflow <= 0:
            return
        candidates = sorted(
            (
                (state.last_seen_ordinal, identity)
                for identity, state in self.states.items()
                if identity not in current_ids
            )
        )
        if len(candidates) < overflow:
            raise RuntimeError("max_states is smaller than the current native map")
        for _, identity in candidates[:overflow]:
            del self.states[identity]
        self.capacity_drops += overflow

    def update(
        self,
        ordinal: int,
        frame_id: int,
        native: NativeFrame,
        proposal_boxes_2d: np.ndarray,
    ) -> np.ndarray:
        if ordinal != self.frames_seen:
            raise ValueError(f"expected ordinal {self.frames_seen}, got {ordinal}")
        if frame_id <= self.last_frame_id:
            raise ValueError("frame_id must increase strictly")
        proposals = np.asarray(proposal_boxes_2d, dtype=np.float64)
        if proposals.size == 0:
            proposals = np.empty((0, 4), dtype=np.float64)
        try:
            proposals = proposals.reshape(-1, 4)
        except ValueError as error:
            raise ValueError("proposal_boxes_2d must have shape [N,4]") from error
        if not np.isfinite(proposals).all() or np.any(
            proposals[:, 2:] <= proposals[:, :2]
        ):
            raise ValueError("proposal_boxes_2d contains an invalid rectangle")

        current_ids = {int(value) for value in native.ids}
        self._prune(ordinal, current_ids)
        projected_rows = []
        projected_boxes = []
        for row_index, corners in enumerate(native.corners):
            projected = _project_xyxy(
                corners,
                native.camera_to_world,
                native.intrinsic,
                native.width,
                native.height,
            )
            if projected is not None:
                projected_rows.append(row_index)
                projected_boxes.append(projected)
        frame_support = np.zeros(len(native.ids), dtype=np.float64)
        if projected_boxes and len(proposals):
            overlap = _iou2d_matrix(np.asarray(projected_boxes), proposals)
            if self.exclusive_matching:
                pairs = [
                    (float(overlap[row, column]), row, column)
                    for row, column in zip(*np.where(overlap >= self.match_iou))
                ]
                pairs.sort(key=lambda item: (-item[0], item[1], item[2]))
                used_rows: set[int] = set()
                used_proposals: set[int] = set()
                for value, projected_row, proposal_row in pairs:
                    if (
                        projected_row in used_rows
                        or proposal_row in used_proposals
                    ):
                        continue
                    used_rows.add(projected_row)
                    used_proposals.add(proposal_row)
                    frame_support[projected_rows[projected_row]] = value
            else:
                frame_support[np.asarray(projected_rows)] = overlap.max(axis=1)

        reranked = np.array(native.scores, copy=True)
        cumulative = np.zeros(len(native.ids), dtype=np.float64)
        for row_index, (identity_value, native_score) in enumerate(
            zip(native.ids, native.scores)
        ):
            identity = int(identity_value)
            state = self.states.setdefault(identity, _NativeScoreState())
            state.max_support = max(state.max_support, float(frame_support[row_index]))
            state.last_seen_ordinal = int(ordinal)
            cumulative[row_index] = state.max_support
            clipped = min(max(float(native_score), 1e-4), 1.0 - 1e-4)
            logit = math.log(clipped / (1.0 - clipped))
            logit += self.logit_weight * max(
                0.0, state.max_support - self.support_threshold
            )
            reranked[row_index] = 1.0 / (1.0 + math.exp(-logit))

        self._enforce_cap(current_ids)
        self.frames_seen += 1
        self.last_frame_id = int(frame_id)
        self.peak_states = max(self.peak_states, len(self.states))
        self.last_ids = np.array(native.ids, copy=True)
        self.last_scores = reranked
        self.last_support = cumulative
        return np.array(reranked, copy=True)

    @staticmethod
    def _updated_score(native_score: float, support: float,
                       threshold: float, weight: float) -> float:
        clipped = min(max(float(native_score), 1e-4), 1.0 - 1e-4)
        logit = math.log(clipped / (1.0 - clipped))
        logit += weight * max(0.0, float(support) - threshold)
        return 1.0 / (1.0 + math.exp(-logit))

    def materialize(
        self, native_ids: np.ndarray, native_scores: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Apply accumulated causal support to a terminal native-map subset."""
        ids = np.asarray(native_ids, dtype=np.int64).reshape(-1)
        scores = np.asarray(native_scores, dtype=np.float64).reshape(-1)
        if len(ids) != len(scores):
            raise ValueError("terminal native IDs and scores must align")
        support = np.asarray([
            self.states.get(int(identity), _NativeScoreState()).max_support
            for identity in ids
        ], dtype=np.float64)
        reranked = np.asarray([
            self._updated_score(
                score, value, self.support_threshold, self.logit_weight
            )
            for score, value in zip(scores, support)
        ], dtype=np.float64)
        return reranked, support

    def diagnostics(self) -> dict:
        return {
            "frames_seen": self.frames_seen,
            "states": len(self.states),
            "peak_states": self.peak_states,
            "max_states": self.max_states,
            "capacity_drops": self.capacity_drops,
            "exclusive_matching": self.exclusive_matching,
            "support_threshold": self.support_threshold,
            "logit_weight": self.logit_weight,
        }


@dataclass(frozen=True)
class OnlineMapSnapshot:
    frame_id: int
    native_ids: np.ndarray
    native_corners: np.ndarray
    native_scores: np.ndarray
    native_support: np.ndarray
    boxes: np.ndarray
    scores: np.ndarray
    sources: tuple[str, ...]
    source_ids: tuple[int, ...]


class OnlineCandidateMap:
    """One-call-per-keyframe online M1-P, M1-A and M2 composition."""

    def __init__(
        self,
        scene_id: str,
        *,
        m1p: OnlineProposalRecovery | None = None,
        m1a: OnlineAnchorRecovery | None = None,
        m2: OnlineNativeReranker | None = None,
    ) -> None:
        self.scene_id = str(scene_id)
        self.m1p = m1p if m1p is not None else OnlineProposalRecovery()
        self.m1a = (
            m1a
            if m1a is not None
            else OnlineAnchorRecovery(
                self.scene_id,
                voxel_m=0.3,
                min_views=3,
                max_active=4096,
                max_births=640,
            )
        )
        self.m2 = m2 if m2 is not None else OnlineNativeReranker()
        self.frames_seen = 0
        self.last_snapshot: OnlineMapSnapshot | None = None
        self.update_seconds: list[float] = []
        self.terminal_readout = False
        self.terminal_counts: dict[str, int] | None = None

    def update(
        self,
        frame_id: int,
        native: NativeFrame,
        evidence: EvidenceFrame,
    ) -> OnlineMapSnapshot:
        started = time.perf_counter()
        ordinal = self.frames_seen
        self.m1p.update(
            ordinal,
            int(frame_id),
            proposal_ids=evidence.proposal_ids,
            proposal_corners=evidence.proposal_corners,
            proposal_scores=evidence.proposal_scores,
            child_ids=evidence.child_ids,
            child_corners=evidence.child_corners,
            child_scores=evidence.child_scores,
            native_corners=native.corners,
        )
        self.m1a.update(
            ordinal,
            int(frame_id),
            evidence.anchor_ids,
            evidence.anchor_corners,
            evidence.anchor_scores,
        )
        native_scores = self.m2.update(
            ordinal,
            int(frame_id),
            native,
            evidence.proposal_boxes_2d,
        )

        boxes = [np.array(box, copy=True) for box in native.corners]
        scores = native_scores.tolist()
        sources = ["native"] * len(boxes)
        source_ids = [int(value) for value in native.ids]
        for row in self.m1p.rows():
            boxes.append(np.array(row["box"], copy=True))
            scores.append(float(row["score"]))
            sources.append("m1p")
            source_ids.append(int(row["track_id"]))
        for row in self.m1a.rows():
            boxes.append(np.array(row["box"], copy=True))
            scores.append(float(row["score"]))
            sources.append("m1a")
            source_ids.append(int(row["anchor_id"]))
        box_array = (
            np.stack(boxes, axis=0)
            if boxes
            else np.empty((0, 8, 3), dtype=np.float64)
        )
        snapshot = OnlineMapSnapshot(
            frame_id=int(frame_id),
            native_ids=np.array(native.ids, copy=True),
            native_corners=np.array(native.corners, copy=True),
            native_scores=np.array(native_scores, copy=True),
            native_support=np.array(self.m2.last_support, copy=True),
            boxes=box_array,
            scores=np.asarray(scores, dtype=np.float64),
            sources=tuple(sources),
            source_ids=tuple(source_ids),
        )
        self.frames_seen += 1
        self.last_snapshot = snapshot
        self.update_seconds.append(time.perf_counter() - started)
        return snapshot

    def materialize_terminal(
        self,
        frame_id: int,
        native_ids: np.ndarray,
        native_corners: np.ndarray,
        native_scores: np.ndarray,
    ) -> OnlineMapSnapshot:
        """Assemble the final prefix on the host mapper's terminal native rows."""
        ids = np.asarray(native_ids, dtype=np.int64).reshape(-1)
        corners = np.asarray(native_corners, dtype=np.float64).reshape(-1, 8, 3)
        scores = np.asarray(native_scores, dtype=np.float64).reshape(-1)
        if not (len(ids) == len(corners) == len(scores)):
            raise ValueError("terminal native arrays must have equal length")
        reranked, support = self.m2.materialize(ids, scores)

        boxes = [np.array(box, copy=True) for box in corners]
        output_scores = reranked.tolist()
        sources = ["native"] * len(boxes)
        source_ids = [int(value) for value in ids]
        for row in self.m1p.terminal_rows(corners):
            boxes.append(np.array(row["box"], copy=True))
            output_scores.append(float(row["score"]))
            sources.append("m1p")
            source_ids.append(int(row["track_id"]))
        for row in self.m1a.rows():
            boxes.append(np.array(row["box"], copy=True))
            output_scores.append(float(row["score"]))
            sources.append("m1a")
            source_ids.append(int(row["anchor_id"]))
        box_array = (
            np.stack(boxes, axis=0)
            if boxes
            else np.empty((0, 8, 3), dtype=np.float64)
        )
        snapshot = OnlineMapSnapshot(
            frame_id=int(frame_id),
            native_ids=np.array(ids, copy=True),
            native_corners=np.array(corners, copy=True),
            native_scores=np.array(reranked, copy=True),
            native_support=np.array(support, copy=True),
            boxes=box_array,
            scores=np.asarray(output_scores, dtype=np.float64),
            sources=tuple(sources),
            source_ids=tuple(source_ids),
        )
        self.last_snapshot = snapshot
        self.terminal_readout = True
        self.terminal_counts = {
            "native": sources.count("native"),
            "m1p": sources.count("m1p"),
            "m1a": sources.count("m1a"),
            "total": len(sources),
        }
        return snapshot

    def diagnostics(self) -> dict:
        milliseconds = np.asarray(self.update_seconds, dtype=np.float64) * 1000.0
        timing = {
            "count": int(len(milliseconds)),
            "mean_ms": float(milliseconds.mean()) if len(milliseconds) else 0.0,
            "p50_ms": float(np.percentile(milliseconds, 50)) if len(milliseconds) else 0.0,
            "p95_ms": float(np.percentile(milliseconds, 95)) if len(milliseconds) else 0.0,
            "max_ms": float(milliseconds.max()) if len(milliseconds) else 0.0,
        }
        return {
            "schema": SCHEMA,
            "scene_id": self.scene_id,
            "strictly_causal": True,
            "online_incremental": True,
            "uses_future_frames": False,
            "uses_terminal_map": self.terminal_readout,
            "uses_terminal_map_for_inference": False,
            "uses_full_scene_cache": False,
            "births_feed_back_into_native_association": False,
            "frames_seen": self.frames_seen,
            "terminal_native_map_readout": self.terminal_readout,
            "terminal_counts": self.terminal_counts,
            "timing": timing,
            "m1p": self.m1p.diagnostics(),
            "m1a": self.m1a.diagnostics(),
            "m2": self.m2.diagnostics(),
        }


def build_online_candidate_map(
    scene_id: str, config: Mapping[str, object] | None = None
) -> OnlineCandidateMap:
    """Build the causal add-on from a small serializable configuration."""
    section = {} if config is None else dict(config)
    m1p = dict(section.get("m1p", {}) or {})
    m1a = dict(section.get("m1a", {}) or {})
    m2 = dict(section.get("m2", {}) or {})
    return OnlineCandidateMap(
        scene_id,
        m1p=OnlineProposalRecovery(**m1p),
        m1a=OnlineAnchorRecovery(
            scene_id,
            voxel_m=float(m1a.get("voxel_m", 0.3)),
            min_views=int(m1a.get("min_views", 3)),
            max_active=int(m1a.get("max_active", 4096)),
            max_births=int(m1a.get("max_births", 640)),
        ),
        m2=OnlineNativeReranker(**m2),
    )
