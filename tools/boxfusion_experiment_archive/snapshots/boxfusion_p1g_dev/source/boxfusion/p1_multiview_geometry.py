"""Detached multi-view geometry refinement for frozen P1S proposals.

P1G is deliberately an observer.  It groups the already decoded, frozen P1S
proposals around each final scene anchor, gathers bounded real-depth residual
points from the corresponding observations, selects quality-and-view-diverse
evidence, and asks the existing occupancy/MSR refiner for a six-face box
proposal.  Neither the P1S proposal stream nor BoxFusion's formal detections
are mutated.

The online caller is expected to expose two additional immutable attributes on
each ``ResidualObservation``:

``geometry_points_world``
    Finite real-depth residual points with shape ``[N, 3]``.

``camera_position``
    The camera centre in the same world frame, shape ``[3]``.

This module intentionally uses structural access instead of importing a
modified ``ResidualObservation`` definition.  That keeps the geometry
observer independently testable while the parent P1S contract remains frozen.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from numbers import Integral, Real
from typing import Any, Mapping, Optional, Sequence, Tuple

import numpy as np

from boxfusion.local_occupancy_msr_refiner import (
    LOCAL_OCCUPANCY_MSR_GATE_FEATURE_DIM,
    LocalOccupancyMSRProposal,
    propose_local_occupancy_msr,
    resolve_local_occupancy_msr_config,
)
from boxfusion.object_memory import (
    MemoryViewRecord,
    deterministic_bounded_sample,
    select_diverse_view_records,
)
from boxfusion.residual_proposal import (
    center_size_to_corners,
    corners_to_center_size,
    pairwise_aabb_iou,
)


P1G_DIAGNOSTIC_SCHEMA = "boxfusion.p1g.multiview_geometry_observer.v1"
P1G_PROFILE = "p1g_multiview_occupancy_msr_observer"
P1G_SOURCE = "p1g_multiview_occupancy_msr"


def _finite(
    name: str,
    value: Any,
    *,
    lower: Optional[float] = None,
    upper: Optional[float] = None,
    strict_lower: bool = False,
) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real number")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    if lower is not None:
        invalid = result <= lower if strict_lower else result < lower
        if invalid:
            relation = "greater than" if strict_lower else "at least"
            raise ValueError(f"{name} must be {relation} {lower}")
    if upper is not None and result > upper:
        raise ValueError(f"{name} must be at most {upper}")
    return result


def _integer(name: str, value: Any, minimum: int) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer")
    result = int(value)
    if result < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return result


def _readonly(
    value: Any,
    *,
    dtype: Any,
    shape: Optional[Tuple[int, ...]] = None,
) -> np.ndarray:
    array = np.asarray(value, dtype=dtype)
    if shape is not None and array.shape != shape:
        raise ValueError(f"array must have shape {shape}, got {array.shape}")
    if np.issubdtype(array.dtype, np.number) and not np.isfinite(array).all():
        raise ValueError("array must contain only finite values")
    result = np.array(array, dtype=dtype, copy=True)
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class P1MultiViewGeometryConfig:
    """Strict, bounded and permanently observer-only P1G configuration."""

    enabled: bool = False
    observer_only: bool = True
    mutate: bool = False
    collect_diagnostics: bool = False
    association_iou: float = 0.10
    crop_scale: float = 1.35
    top_k_views: int = 5
    view_diversity_weight: float = 0.25
    max_points_per_view: int = 768
    max_candidates: int = 256
    proposal: Mapping[str, Any] = field(default_factory=dict)

    def validated(self) -> "P1MultiViewGeometryConfig":
        for name in (
            "enabled",
            "observer_only",
            "mutate",
            "collect_diagnostics",
        ):
            if not isinstance(getattr(self, name), (bool, np.bool_)):
                raise ValueError(f"p1g.{name} must be Boolean")
        if not bool(self.observer_only):
            raise ValueError("P1G must remain observer_only")
        if bool(self.mutate):
            raise ValueError("P1G cannot mutate formal detections")
        if not isinstance(self.proposal, Mapping):
            raise TypeError("p1g.proposal must be a mapping")

        association_iou = _finite(
            "p1g.association_iou",
            self.association_iou,
            lower=0.0,
            upper=1.0,
        )
        crop_scale = _finite(
            "p1g.crop_scale",
            self.crop_scale,
            lower=1.0,
        )
        top_k_views = _integer("p1g.top_k_views", self.top_k_views, 1)
        diversity = _finite(
            "p1g.view_diversity_weight",
            self.view_diversity_weight,
            lower=0.0,
            upper=1.0,
        )
        max_points = _integer(
            "p1g.max_points_per_view", self.max_points_per_view, 1
        )
        max_candidates = _integer(
            "p1g.max_candidates", self.max_candidates, 1
        )

        proposal_updates = dict(self.proposal)
        bound_values = {
            "max_views": top_k_views,
            "max_points_per_view": max_points,
            "crop_scale": crop_scale,
        }
        for key, expected in bound_values.items():
            if key in proposal_updates:
                observed = proposal_updates[key]
                if isinstance(expected, int):
                    matches = (
                        not isinstance(observed, (bool, np.bool_))
                        and isinstance(observed, Integral)
                        and int(observed) == expected
                    )
                else:
                    matches = (
                        not isinstance(observed, (bool, np.bool_))
                        and isinstance(observed, Real)
                        and np.isfinite(float(observed))
                        and np.isclose(
                            float(observed),
                            float(expected),
                            rtol=0.0,
                            atol=1e-12,
                        )
                    )
                if not matches:
                    raise ValueError(
                        f"p1g.proposal.{key} must match p1g.{key}"
                    )
            proposal_updates[key] = expected
        resolved_proposal = resolve_local_occupancy_msr_config(
            proposal_updates
        )
        return P1MultiViewGeometryConfig(
            enabled=bool(self.enabled),
            observer_only=True,
            mutate=False,
            collect_diagnostics=bool(self.collect_diagnostics),
            association_iou=association_iou,
            crop_scale=crop_scale,
            top_k_views=top_k_views,
            view_diversity_weight=diversity,
            max_points_per_view=max_points,
            max_candidates=max_candidates,
            proposal=dict(resolved_proposal),
        )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["proposal"] = dict(self.proposal)
        return payload


def resolve_p1_multiview_geometry_config(
    config: Optional[
        Mapping[str, Any] | P1MultiViewGeometryConfig
    ] = None,
) -> P1MultiViewGeometryConfig:
    if config is None:
        return P1MultiViewGeometryConfig().validated()
    if isinstance(config, P1MultiViewGeometryConfig):
        return config.validated()
    if not isinstance(config, Mapping):
        raise TypeError("p1_multiview_geometry must be a mapping")
    known = set(P1MultiViewGeometryConfig.__dataclass_fields__)
    unknown = sorted(set(config) - known)
    if unknown:
        raise ValueError(
            "Unknown p1_multiview_geometry key(s): "
            + ", ".join(str(value) for value in unknown)
        )
    return P1MultiViewGeometryConfig(**dict(config)).validated()


@dataclass(frozen=True)
class P1GeometryCandidate:
    """One one-to-one, detached geometry proposal for a frozen P1S anchor."""

    parent_candidate_id: str
    refined_candidate_id: str
    parent_box: np.ndarray
    parent_corners: np.ndarray
    refined_box: np.ndarray
    refined_corners: np.ndarray
    score: float
    reason: str
    matched_view_count: int
    selected_view_count: int
    selected_frame_ids: Tuple[int, ...]
    cropped_point_count: int
    face_residuals: np.ndarray
    face_support: np.ndarray
    face_uncertainty: np.ndarray
    face_supported: np.ndarray
    feature_vector: np.ndarray
    total_seconds: float
    source: str = P1G_SOURCE

    def __post_init__(self) -> None:
        for name in ("parent_candidate_id", "refined_candidate_id", "reason"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"{name} must be a non-empty string")
        if self.source != P1G_SOURCE:
            raise ValueError("P1G candidate source is invalid")
        for name in ("parent_box", "refined_box"):
            value = _readonly(
                getattr(self, name), dtype=np.float32, shape=(6,)
            )
            if np.any(value[3:] <= 0.0):
                raise ValueError(f"{name} extents must be positive")
            object.__setattr__(self, name, value)
        for name in ("parent_corners", "refined_corners"):
            object.__setattr__(
                self,
                name,
                _readonly(
                    getattr(self, name), dtype=np.float32, shape=(8, 3)
                ),
            )
        object.__setattr__(
            self,
            "score",
            _finite("p1g candidate score", self.score, lower=0.0, upper=1.0),
        )
        object.__setattr__(
            self,
            "matched_view_count",
            _integer(
                "p1g matched_view_count", self.matched_view_count, 0
            ),
        )
        object.__setattr__(
            self,
            "selected_view_count",
            _integer(
                "p1g selected_view_count", self.selected_view_count, 0
            ),
        )
        if self.selected_view_count > self.matched_view_count:
            raise ValueError("selected views cannot exceed matched views")
        frame_ids = tuple(
            _integer("p1g selected frame id", value, 0)
            for value in self.selected_frame_ids
        )
        if len(frame_ids) != self.selected_view_count:
            raise ValueError("selected frame ids do not align with views")
        if len(set(frame_ids)) != len(frame_ids):
            raise ValueError("selected frame ids must be unique")
        object.__setattr__(self, "selected_frame_ids", frame_ids)
        object.__setattr__(
            self,
            "cropped_point_count",
            _integer(
                "p1g cropped_point_count", self.cropped_point_count, 0
            ),
        )
        for name, dtype in (
            ("face_residuals", np.float32),
            ("face_support", np.float32),
            ("face_uncertainty", np.float32),
            ("face_supported", np.bool_),
        ):
            object.__setattr__(
                self,
                name,
                _readonly(
                    getattr(self, name), dtype=dtype, shape=(3, 2)
                ),
            )
        object.__setattr__(
            self,
            "feature_vector",
            _readonly(
                self.feature_vector,
                dtype=np.float32,
                shape=(LOCAL_OCCUPANCY_MSR_GATE_FEATURE_DIM,),
            ),
        )
        object.__setattr__(
            self,
            "total_seconds",
            _finite("p1g total_seconds", self.total_seconds, lower=0.0),
        )

    @property
    def is_candidate(self) -> bool:
        return bool(
            self.reason == "candidate"
            and not np.array_equal(self.parent_corners, self.refined_corners)
        )

    @property
    def applied(self) -> bool:
        return False


@dataclass(frozen=True)
class _PreparedProposal:
    candidate_id: str
    box: np.ndarray
    score: float


@dataclass(frozen=True)
class _PreparedObservation:
    frame_index: int
    provider_step: int
    camera_position: np.ndarray
    points_world: np.ndarray
    proposals: Tuple[_PreparedProposal, ...]


def _proposal_value(proposal: Any) -> _PreparedProposal:
    candidate_id = str(getattr(proposal, "candidate_id"))
    if not candidate_id:
        raise ValueError("proposal candidate_id is empty")
    box = _readonly(getattr(proposal, "box"), dtype=np.float32, shape=(6,))
    if np.any(box[3:] <= 0.0):
        raise ValueError("proposal extents must be positive")
    score = _finite(
        "proposal objectness",
        getattr(proposal, "objectness"),
        lower=0.0,
        upper=1.0,
    )
    return _PreparedProposal(candidate_id, box, score)


def _prepare_observation(observation: Any) -> _PreparedObservation:
    frame_index = _integer(
        "observation frame_index",
        getattr(observation, "frame_index"),
        0,
    )
    provider_step = _integer(
        "observation provider_step",
        getattr(observation, "provider_step"),
        0,
    )
    camera_position = _readonly(
        getattr(observation, "camera_position"),
        dtype=np.float32,
        shape=(3,),
    )
    points = _readonly(
        getattr(observation, "geometry_points_world"),
        dtype=np.float32,
    )
    if points.ndim != 2 or points.shape[1:] != (3,):
        raise ValueError("geometry_points_world must have shape [N,3]")
    raw_proposals = getattr(observation, "proposals")
    if isinstance(raw_proposals, (str, bytes)) or not isinstance(
        raw_proposals, Sequence
    ):
        raise ValueError("observation proposals must be a sequence")
    proposals = []
    for proposal in raw_proposals:
        try:
            proposals.append(_proposal_value(proposal))
        except (AttributeError, TypeError, ValueError):
            # One malformed proposal must not poison the other view evidence.
            continue
    proposals.sort(key=lambda row: row.candidate_id)
    return _PreparedObservation(
        frame_index=frame_index,
        provider_step=provider_step,
        camera_position=camera_position,
        points_world=points,
        proposals=tuple(proposals),
    )


def _anchor_geometry(anchor: Any) -> tuple[str, np.ndarray, np.ndarray, float]:
    candidate_id = str(getattr(anchor, "candidate_id"))
    if not candidate_id:
        raise ValueError("anchor candidate_id is empty")
    box = _readonly(getattr(anchor, "box"), dtype=np.float32, shape=(6,))
    if np.any(box[3:] <= 0.0):
        raise ValueError("anchor extents must be positive")
    corners = _readonly(
        getattr(anchor, "corners"), dtype=np.float32, shape=(8, 3)
    )
    expected = center_size_to_corners(box)[0]
    if not np.allclose(
        corners.min(axis=0),
        expected.min(axis=0),
        atol=1e-5,
        rtol=0.0,
    ) or not np.allclose(
        corners.max(axis=0),
        expected.max(axis=0),
        atol=1e-5,
        rtol=0.0,
    ):
        raise ValueError("anchor box and corners disagree")
    score = _finite(
        "anchor objectness",
        getattr(anchor, "objectness"),
        lower=0.0,
        upper=1.0,
    )
    return candidate_id, box, corners, score


def _empty_face_diagnostics(box: np.ndarray) -> tuple[np.ndarray, ...]:
    uncertainty = np.repeat(
        np.asarray(box[3:], dtype=np.float32)[:, None], 2, axis=1
    )
    return (
        np.zeros((3, 2), dtype=np.float32),
        np.zeros((3, 2), dtype=np.float32),
        uncertainty,
        np.zeros((3, 2), dtype=np.bool_),
        np.zeros(
            (LOCAL_OCCUPANCY_MSR_GATE_FEATURE_DIM,), dtype=np.float32
        ),
    )


def _identity_candidate(
    anchor: Any,
    *,
    reason: str,
    elapsed: float,
    matched_view_count: int = 0,
    selected: Sequence[MemoryViewRecord] = (),
) -> P1GeometryCandidate:
    candidate_id, box, corners, score = _anchor_geometry(anchor)
    (
        face_residuals,
        face_support,
        face_uncertainty,
        face_supported,
        features,
    ) = _empty_face_diagnostics(box)
    return P1GeometryCandidate(
        parent_candidate_id=candidate_id,
        refined_candidate_id=f"{candidate_id}:p1g",
        parent_box=box,
        parent_corners=corners,
        refined_box=box,
        refined_corners=corners,
        score=score,
        reason=reason,
        matched_view_count=matched_view_count,
        selected_view_count=len(selected),
        selected_frame_ids=tuple(row.frame_id for row in selected),
        cropped_point_count=int(
            sum(len(row.points_world) for row in selected)
        ),
        face_residuals=face_residuals,
        face_support=face_support,
        face_uncertainty=face_uncertainty,
        face_supported=face_supported,
        feature_vector=features,
        total_seconds=max(float(elapsed), 0.0),
    )


class P1MultiViewGeometryObserver:
    """Scene-level, one-to-one and fail-open P1G geometry observer."""

    def __init__(
        self,
        config: Mapping[str, Any] | P1MultiViewGeometryConfig,
        *,
        parent_checkpoint_sha256: str,
    ) -> None:
        self.config = resolve_p1_multiview_geometry_config(config)
        if not isinstance(parent_checkpoint_sha256, str) or not (
            parent_checkpoint_sha256.strip()
        ):
            raise ValueError("P1G parent checkpoint identity is required")
        self.parent_checkpoint_sha256 = parent_checkpoint_sha256.strip()
        self.scene_id: Optional[str] = None
        self.candidates: Tuple[P1GeometryCandidate, ...] = ()
        self.failure_count = 0
        self.runtime_seconds = 0.0

    def reset(self, scene_id: str) -> None:
        if not isinstance(scene_id, str) or not scene_id.strip():
            raise ValueError("scene_id must be a non-empty string")
        self.scene_id = scene_id.strip()
        self.candidates = ()
        self.failure_count = 0
        self.runtime_seconds = 0.0

    def _view_records(
        self,
        box: np.ndarray,
        observations: Sequence[_PreparedObservation],
    ) -> tuple[Tuple[MemoryViewRecord, ...], int]:
        lower = box[:3] - 0.5 * self.config.crop_scale * box[3:]
        upper = box[:3] + 0.5 * self.config.crop_scale * box[3:]
        records: dict[int, MemoryViewRecord] = {}
        for observation in observations:
            if not observation.proposals or not len(observation.points_world):
                continue
            proposal_boxes = np.stack(
                [row.box for row in observation.proposals], axis=0
            )
            overlaps = pairwise_aabb_iou(box[None], proposal_boxes)[0]
            eligible = np.flatnonzero(
                overlaps + 1e-12 >= self.config.association_iou
            )
            if not len(eligible):
                continue
            best = min(
                eligible.tolist(),
                key=lambda index: (
                    -float(overlaps[index]),
                    -float(observation.proposals[index].score),
                    observation.proposals[index].candidate_id,
                ),
            )
            matched = observation.proposals[int(best)]
            points = observation.points_world
            inside = np.logical_and(
                points >= lower[None], points <= upper[None]
            ).all(axis=1)
            cropped = points[inside]
            if len(cropped) < int(
                self.config.proposal["min_points_per_view"]
            ):
                continue
            cropped = deterministic_bounded_sample(
                cropped, self.config.max_points_per_view
            )
            agreement = float(np.clip(overlaps[int(best)], 0.0, 1.0))
            quality = float(
                np.clip(matched.score * agreement, 0.0, 1.0)
            )
            record = MemoryViewRecord(
                frame_id=observation.frame_index,
                points_world=cropped,
                quality=quality,
                confidence=matched.score,
                valid_depth_ratio=1.0,
                projection_mask_iou=agreement,
                camera_position=observation.camera_position,
            )
            existing = records.get(record.frame_id)
            if existing is None or (
                record.quality,
                record.confidence,
                record.projection_mask_iou,
                len(record.points_world),
            ) > (
                existing.quality,
                existing.confidence,
                existing.projection_mask_iou,
                len(existing.points_world),
            ):
                records[record.frame_id] = record
        ordered = tuple(records[key] for key in sorted(records))
        return ordered, len(ordered)

    def _refine_one(
        self,
        anchor: Any,
        observations: Sequence[_PreparedObservation],
    ) -> tuple[P1GeometryCandidate, bool]:
        started = time.perf_counter()
        candidate_id, box, corners, score = _anchor_geometry(anchor)
        records, matched_view_count = self._view_records(box, observations)
        if records:
            selected = select_diverse_view_records(
                records,
                min(self.config.top_k_views, len(records)),
                self.config.view_diversity_weight,
            )
        else:
            selected = ()
        try:
            proposal: LocalOccupancyMSRProposal = (
                propose_local_occupancy_msr(
                    corners,
                    selected,
                    config=self.config.proposal,
                )
            )
            refined_corners = np.asarray(
                proposal.candidate_corners, dtype=np.float32
            )
            if (
                str(proposal.reason) != "candidate"
                or np.array_equal(refined_corners, corners)
            ):
                return (
                    P1GeometryCandidate(
                        parent_candidate_id=candidate_id,
                        refined_candidate_id=f"{candidate_id}:p1g",
                        parent_box=box,
                        parent_corners=corners,
                        refined_box=box,
                        refined_corners=corners,
                        score=score,
                        reason=str(proposal.reason),
                        matched_view_count=matched_view_count,
                        selected_view_count=len(selected),
                        selected_frame_ids=tuple(
                            int(row.frame_id) for row in selected
                        ),
                        cropped_point_count=int(
                            sum(len(row.points_world) for row in selected)
                        ),
                        face_residuals=proposal.face_residuals,
                        face_support=proposal.face_support,
                        face_uncertainty=proposal.face_uncertainty,
                        face_supported=proposal.face_supported,
                        feature_vector=proposal.feature_vector,
                        total_seconds=time.perf_counter() - started,
                    ),
                    False,
                )
            refined_box = corners_to_center_size(
                refined_corners[None]
            )[0]
            result = P1GeometryCandidate(
                parent_candidate_id=candidate_id,
                refined_candidate_id=f"{candidate_id}:p1g",
                parent_box=box,
                parent_corners=corners,
                refined_box=refined_box,
                refined_corners=refined_corners,
                score=score,
                reason=str(proposal.reason),
                matched_view_count=matched_view_count,
                selected_view_count=len(selected),
                selected_frame_ids=tuple(
                    int(row.frame_id) for row in selected
                ),
                cropped_point_count=int(
                    sum(len(row.points_world) for row in selected)
                ),
                face_residuals=proposal.face_residuals,
                face_support=proposal.face_support,
                face_uncertainty=proposal.face_uncertainty,
                face_supported=proposal.face_supported,
                feature_vector=proposal.feature_vector,
                total_seconds=time.perf_counter() - started,
            )
            return result, False
        except Exception as error:  # fail-open is the explicit P1G contract
            reason = f"identity_exception:{type(error).__name__}"
            return (
                _identity_candidate(
                    anchor,
                    reason=reason,
                    elapsed=time.perf_counter() - started,
                    matched_view_count=matched_view_count,
                    selected=selected,
                ),
                True,
            )

    def observe_scene(
        self,
        *,
        scene_id: str,
        anchors: Sequence[Any],
        observations: Sequence[Any],
    ) -> Tuple[P1GeometryCandidate, ...]:
        """Build detached refined candidates without a mutation path.

        Runtime data errors fail open.  A malformed observation is ignored;
        a per-anchor refiner error produces an identity candidate.  Invalid
        configuration and invalid parent anchors remain programming errors and
        fail closed.
        """

        if not isinstance(scene_id, str) or not scene_id.strip():
            raise ValueError("scene_id must be a non-empty string")
        if isinstance(anchors, (str, bytes)) or not isinstance(
            anchors, Sequence
        ):
            raise TypeError("anchors must be a sequence")
        if isinstance(observations, (str, bytes)) or not isinstance(
            observations, Sequence
        ):
            raise TypeError("observations must be a sequence")
        self.reset(scene_id.strip())
        if not self.config.enabled:
            return ()

        prepared = []
        malformed_observations = 0
        for observation in observations:
            try:
                prepared.append(_prepare_observation(observation))
            except (AttributeError, TypeError, ValueError):
                malformed_observations += 1
        prepared.sort(
            key=lambda row: (
                row.provider_step,
                row.frame_index,
            )
        )

        results = []
        failures = malformed_observations
        total_started = time.perf_counter()
        for anchor in tuple(anchors)[: self.config.max_candidates]:
            result, failed = self._refine_one(anchor, prepared)
            results.append(result)
            failures += int(failed)
        self.runtime_seconds = time.perf_counter() - total_started
        self.failure_count = failures
        self.candidates = tuple(results)
        return self.candidates

    def diagnostic_payload(self) -> dict[str, np.ndarray]:
        """Return a pickle-free, row-aligned P1G diagnostic payload."""

        rows = self.candidates
        count = len(rows)
        max_views = self.config.top_k_views

        def values(
            attribute: str,
            empty_shape: Tuple[int, ...],
            dtype: Any,
        ) -> np.ndarray:
            if not rows:
                return np.empty(empty_shape, dtype=dtype)
            return np.asarray(
                [getattr(row, attribute) for row in rows], dtype=dtype
            )

        selected_frames = np.full(
            (count, max_views), -1, dtype=np.int64
        )
        for index, row in enumerate(rows):
            selected_frames[
                index, : len(row.selected_frame_ids)
            ] = np.asarray(row.selected_frame_ids, dtype=np.int64)
        config_json = json.dumps(
            self.config.to_dict(),
            sort_keys=True,
            separators=(",", ":"),
        )
        return {
            "p1g_schema": np.asarray(P1G_DIAGNOSTIC_SCHEMA),
            "p1g_stage": np.asarray("P1G"),
            "p1g_profile": np.asarray(P1G_PROFILE),
            "p1g_parent_stage": np.asarray("P1S"),
            "p1g_enabled": np.asarray(self.config.enabled, dtype=bool),
            "p1g_observer_only": np.asarray(True, dtype=bool),
            "p1g_uses_ground_truth": np.asarray(False, dtype=bool),
            "p1g_reads_semantic_labels": np.asarray(False, dtype=bool),
            "p1g_mutation_enabled": np.asarray(False, dtype=bool),
            "p1g_applied_count": np.asarray(0, dtype=np.int64),
            "p1g_complete": np.asarray(True, dtype=bool),
            "p1g_class_agnostic": np.asarray(True, dtype=bool),
            "p1g_regression_dim": np.asarray(6, dtype=np.int64),
            "p1g_parent_checkpoint_sha256": np.asarray(
                self.parent_checkpoint_sha256
            ),
            "p1g_config_json": np.asarray(config_json),
            "p1g_runtime_seconds": np.asarray(
                self.runtime_seconds, dtype=np.float64
            ),
            "p1g_failure_count": np.asarray(
                self.failure_count, dtype=np.int64
            ),
            "p1g_parent_candidate_ids": values(
                "parent_candidate_id", (0,), np.str_
            ),
            "p1g_refined_candidate_ids": values(
                "refined_candidate_id", (0,), np.str_
            ),
            "p1g_parent_boxes": values(
                "parent_box", (0, 6), np.float32
            ),
            "p1g_parent_corners": values(
                "parent_corners", (0, 8, 3), np.float32
            ),
            "p1g_refined_boxes": values(
                "refined_box", (0, 6), np.float32
            ),
            "p1g_refined_corners": values(
                "refined_corners", (0, 8, 3), np.float32
            ),
            "p1g_candidate_scores": values(
                "score", (0,), np.float32
            ),
            "p1g_candidate_applied": np.zeros(count, dtype=bool),
            "p1g_is_candidate": np.asarray(
                [row.is_candidate for row in rows], dtype=bool
            ),
            "p1g_reasons": values("reason", (0,), np.str_),
            "p1g_sources": values("source", (0,), np.str_),
            "p1g_matched_view_counts": values(
                "matched_view_count", (0,), np.int64
            ),
            "p1g_selected_view_counts": values(
                "selected_view_count", (0,), np.int64
            ),
            "p1g_selected_frame_ids": selected_frames,
            "p1g_cropped_point_counts": values(
                "cropped_point_count", (0,), np.int64
            ),
            "p1g_face_residuals": values(
                "face_residuals", (0, 3, 2), np.float32
            ),
            "p1g_face_support": values(
                "face_support", (0, 3, 2), np.float32
            ),
            "p1g_face_uncertainty": values(
                "face_uncertainty", (0, 3, 2), np.float32
            ),
            "p1g_face_supported": values(
                "face_supported", (0, 3, 2), np.bool_
            ),
            "p1g_feature_vectors": values(
                "feature_vector",
                (0, LOCAL_OCCUPANCY_MSR_GATE_FEATURE_DIM),
                np.float32,
            ),
            "p1g_step_total_seconds": values(
                "total_seconds", (0,), np.float64
            ),
        }


__all__ = [
    "P1G_DIAGNOSTIC_SCHEMA",
    "P1G_PROFILE",
    "P1G_SOURCE",
    "P1GeometryCandidate",
    "P1MultiViewGeometryConfig",
    "P1MultiViewGeometryObserver",
    "resolve_p1_multiview_geometry_config",
]
