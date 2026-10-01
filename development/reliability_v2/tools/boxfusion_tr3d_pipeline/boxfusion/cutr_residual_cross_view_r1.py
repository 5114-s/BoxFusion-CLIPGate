"""Training-free causal cross-view confirmation for CuTR residual tracks.

R1 is deliberately isolated from native BoxFusion state.  It consumes the
row-aligned assignments emitted by :mod:`cutr_residual_birth_lite`, projects
only earlier committed depth guides into the current RGB-D keyframe, and
returns a counterfactual subset of the S0 terminal candidates.  It cannot add,
remove, rescore, or refine a native prediction.
"""

from __future__ import annotations

from collections import OrderedDict, deque
from dataclasses import dataclass
import math
from numbers import Integral, Real
import time
from typing import Callable, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

from .cutr_residual_birth_lite import (
    ResidualCandidate,
    ResidualCloseResult,
    ResidualKeyframeResult,
)
from .depth_guide_geometry import project_guide_metrics


SCHEMA = "boxfusion.cutr_residual_cross_view_r1_shadow.v1"

DESCRIPTOR_DIM = 256
DESCRIPTOR_COSINE = 0.80
TRANSLATION_GAP_M = 0.80
ROTATION_GAP_DEG = 30.0
DEPTH_ALPHA = 0.05
FRAME_VISIBILITY = 0.30
BOX_VISIBILITY = 0.90
MIN_COMPONENT_NODES = 3
MIN_COMPONENT_EDGES = 2
MAX_NODES_PER_TRACK = 5
PROJECTION_BUDGET_POINTS = 8192
MAX_RECEIPTS = 1024
MIN_GUIDE_POINTS = 16
MAX_GUIDE_POINTS = 64
_TIMING_WINDOW = 2048


def _strict_int(name: str, value: object, minimum: int = 0) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer")
    result = int(value)
    if result < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return result


def _finite_float(name: str, value: object) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite number")
    return result


def _immutable_array(
    value: object,
    *,
    name: str,
    shape: Optional[Tuple[int, ...]] = None,
    ndim: Optional[int] = None,
) -> np.ndarray:
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a finite numeric array") from error
    if shape is not None and array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}")
    if ndim is not None and array.ndim != ndim:
        raise ValueError(f"{name} must have {ndim} dimensions")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite")
    packed = np.array(array, dtype=np.float64, order="C", copy=True).tobytes()
    return np.frombuffer(packed, dtype=np.float64).reshape(array.shape)


def _proper_pose(value: object, name: str = "camera_to_world") -> np.ndarray:
    pose = _immutable_array(value, name=name, shape=(4, 4))
    if np.max(np.abs(pose[3] - [0.0, 0.0, 0.0, 1.0])) > 1e-8:
        raise ValueError(f"{name} must be homogeneous")
    rotation = pose[:3, :3]
    if (
        np.max(np.abs(rotation.T @ rotation - np.eye(3))) > 1e-5
        or abs(float(np.linalg.det(rotation)) - 1.0) > 1e-5
    ):
        raise ValueError(f"{name} must contain a proper rotation")
    return pose


def _box_xyxy(value: object) -> Tuple[float, float, float, float]:
    box = _immutable_array(value, name="raw_box_xyxy", shape=(4,))
    if box[2] <= box[0] or box[3] <= box[1]:
        raise ValueError("raw_box_xyxy must have positive area")
    return tuple(float(item) for item in box)


def _descriptor(value: object) -> np.ndarray:
    descriptor = _immutable_array(
        value, name="descriptor", shape=(DESCRIPTOR_DIM,)
    )
    norm = float(np.linalg.norm(descriptor))
    if not math.isfinite(norm) or norm <= 1e-6:
        raise ValueError("descriptor must have non-zero finite norm")
    normalized = np.asarray(descriptor / norm, dtype=np.float64)
    return _immutable_array(
        normalized, name="normalized_descriptor", shape=(DESCRIPTOR_DIM,)
    )


def _guide(value: object) -> np.ndarray:
    points = _immutable_array(value, name="guide_points_world", ndim=2)
    if points.shape[1:] != (3,) or not MIN_GUIDE_POINTS <= len(points) <= MAX_GUIDE_POINTS:
        raise ValueError("guide_points_world must contain 16..64 finite [x,y,z] rows")
    return points


@dataclass(frozen=True)
class ResidualCrossViewConfig:
    enabled: bool = False
    observer_only: bool = True
    descriptor_dim: int = DESCRIPTOR_DIM
    descriptor_cosine: float = DESCRIPTOR_COSINE
    translation_gap_m: float = TRANSLATION_GAP_M
    rotation_gap_deg: float = ROTATION_GAP_DEG
    depth_alpha: float = DEPTH_ALPHA
    frame_visibility: float = FRAME_VISIBILITY
    box_visibility: float = BOX_VISIBILITY
    min_component_nodes: int = MIN_COMPONENT_NODES
    min_component_edges: int = MIN_COMPONENT_EDGES
    max_nodes_per_track: int = MAX_NODES_PER_TRACK
    projection_budget_points: int = PROJECTION_BUDGET_POINTS
    max_receipts: int = MAX_RECEIPTS

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, (bool, np.bool_)):
            raise ValueError("enabled must be boolean")
        if not isinstance(self.observer_only, (bool, np.bool_)) or not self.observer_only:
            raise ValueError("observer_only must remain true")
        frozen = {
            "descriptor_dim": (self.descriptor_dim, DESCRIPTOR_DIM),
            "descriptor_cosine": (self.descriptor_cosine, DESCRIPTOR_COSINE),
            "translation_gap_m": (self.translation_gap_m, TRANSLATION_GAP_M),
            "rotation_gap_deg": (self.rotation_gap_deg, ROTATION_GAP_DEG),
            "depth_alpha": (self.depth_alpha, DEPTH_ALPHA),
            "frame_visibility": (self.frame_visibility, FRAME_VISIBILITY),
            "box_visibility": (self.box_visibility, BOX_VISIBILITY),
            "min_component_nodes": (self.min_component_nodes, MIN_COMPONENT_NODES),
            "min_component_edges": (self.min_component_edges, MIN_COMPONENT_EDGES),
            "max_nodes_per_track": (self.max_nodes_per_track, MAX_NODES_PER_TRACK),
            "projection_budget_points": (
                self.projection_budget_points,
                PROJECTION_BUDGET_POINTS,
            ),
            "max_receipts": (self.max_receipts, MAX_RECEIPTS),
        }
        for name, (actual, expected) in frozen.items():
            if isinstance(expected, int):
                actual_value = _strict_int(name, actual, 1)
            else:
                actual_value = _finite_float(name, actual)
            if actual_value != expected:
                raise ValueError(f"{name} is frozen at {expected}")
            object.__setattr__(self, name, actual_value)


@dataclass(frozen=True)
class ResidualCrossViewEvidence:
    """One accepted base-row's copied R1 evidence, or an explicit abstention."""

    frame_id: int
    raw_index: int
    descriptor: Optional[np.ndarray] = None
    camera_to_world: Optional[np.ndarray] = None
    raw_box_xyxy: Optional[Tuple[float, float, float, float]] = None
    guide_points_world: Optional[np.ndarray] = None
    reason: str = "valid"

    def __post_init__(self) -> None:
        object.__setattr__(self, "frame_id", _strict_int("frame_id", self.frame_id))
        object.__setattr__(self, "raw_index", _strict_int("raw_index", self.raw_index))
        supplied = (
            self.descriptor is not None,
            self.camera_to_world is not None,
            self.raw_box_xyxy is not None,
            self.guide_points_world is not None,
        )
        if all(supplied):
            if self.reason != "valid":
                raise ValueError("valid evidence must use reason='valid'")
            object.__setattr__(self, "descriptor", _descriptor(self.descriptor))
            object.__setattr__(
                self, "camera_to_world", _proper_pose(self.camera_to_world)
            )
            object.__setattr__(self, "raw_box_xyxy", _box_xyxy(self.raw_box_xyxy))
            object.__setattr__(
                self, "guide_points_world", _guide(self.guide_points_world)
            )
        elif not any(supplied):
            if not isinstance(self.reason, str) or not self.reason or self.reason == "valid":
                raise ValueError("invalid evidence requires a non-empty reason")
        else:
            raise ValueError("R1 evidence fields must be all present or all absent")

    @classmethod
    def abstain(cls, frame_id: int, raw_index: int, reason: str):
        return cls(frame_id=frame_id, raw_index=raw_index, reason=reason)

    @property
    def valid(self) -> bool:
        return self.descriptor is not None


@dataclass(frozen=True)
class ResidualCrossViewEdge:
    earlier_frame_id: int
    later_frame_id: int
    descriptor_cosine: float
    translation_m: float
    rotation_deg: float
    ray_angle_deg: float
    frame_visibility: float
    depth_consistency: float
    box_visibility: float
    box_depth_consistency: float
    affinity: float
    supporting: bool

    def to_json_dict(self) -> Dict[str, object]:
        return {
            "earlier_frame_id": self.earlier_frame_id,
            "later_frame_id": self.later_frame_id,
            "descriptor_cosine": self.descriptor_cosine,
            "translation_m": self.translation_m,
            "rotation_deg": self.rotation_deg,
            "ray_angle_deg": self.ray_angle_deg,
            "frame_visibility": self.frame_visibility,
            "depth_consistency": self.depth_consistency,
            "box_visibility": self.box_visibility,
            "box_depth_consistency": self.box_depth_consistency,
            "affinity": self.affinity,
            "supporting": self.supporting,
        }


@dataclass(frozen=True)
class ResidualCrossViewReceipt:
    track_id: int
    confirmation_frame_id: int
    component_frame_ids: Tuple[int, ...]
    supporting_edges: Tuple[ResidualCrossViewEdge, ...]

    def to_json_dict(self) -> Dict[str, object]:
        return {
            "track_id": self.track_id,
            "confirmation_frame_id": self.confirmation_frame_id,
            "component_frame_ids": list(self.component_frame_ids),
            "supporting_edge_count": len(self.supporting_edges),
            "supporting_edges": [edge.to_json_dict() for edge in self.supporting_edges],
        }


@dataclass(frozen=True)
class ResidualCrossViewKeyframeResult:
    frame_id: int
    assignment_count: int
    valid_node_count: int
    invalid_raw_indices: Tuple[int, ...]
    budget_abstained_raw_indices: Tuple[int, ...]
    projection_points_used: int
    newly_confirmed_track_ids: Tuple[int, ...]
    retired_state_track_ids: Tuple[int, ...]
    audit_complete: bool
    observe_ms: float


@dataclass(frozen=True)
class ResidualCrossViewCloseResult:
    candidates: Tuple[ResidualCandidate, ...]
    admitted_track_ids: Tuple[int, ...]
    rejected_track_ids: Tuple[int, ...]
    audit_complete: bool
    observer_only: bool = True
    active_authorized: bool = False
    native_mutation_applied: bool = False

    def to_json_dict(self) -> Dict[str, object]:
        return {
            "candidates": [candidate.to_json_dict() for candidate in self.candidates],
            "admitted_track_ids": list(self.admitted_track_ids),
            "rejected_track_ids": list(self.rejected_track_ids),
            "audit_complete": self.audit_complete,
            "observer_only": self.observer_only,
            "active_authorized": self.active_authorized,
            "native_mutation_applied": self.native_mutation_applied,
        }


@dataclass(frozen=True)
class _TrackState:
    track_id: int
    nodes: Tuple[ResidualCrossViewEvidence, ...]
    edges: Tuple[ResidualCrossViewEdge, ...]


def _rotation_degrees(left: np.ndarray, right: np.ndarray) -> float:
    relative = left[:3, :3].T @ right[:3, :3]
    cosine = float(np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def _ray_angle_degrees(
    left: ResidualCrossViewEvidence, right: ResidualCrossViewEvidence
) -> float:
    center = 0.5 * (
        np.mean(left.guide_points_world, axis=0)
        + np.mean(right.guide_points_world, axis=0)
    )
    left_ray = left.camera_to_world[:3, 3] - center
    right_ray = right.camera_to_world[:3, 3] - center
    denominator = float(np.linalg.norm(left_ray) * np.linalg.norm(right_ray))
    if denominator <= 1e-12:
        return 0.0
    cosine = float(np.clip(np.dot(left_ray, right_ray) / denominator, -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def _metric(result: object, name: str) -> float:
    value = result.get(name) if isinstance(result, Mapping) else getattr(result, name)
    number = _finite_float(f"projection.{name}", value)
    if not 0.0 <= number <= 1.0:
        raise ValueError(f"projection.{name} must lie in [0,1]")
    return number


def _component_receipt(
    track_id: int,
    confirmation_frame_id: int,
    nodes: Sequence[ResidualCrossViewEvidence],
    edges: Sequence[ResidualCrossViewEdge],
) -> Optional[ResidualCrossViewReceipt]:
    frames = tuple(sorted({node.frame_id for node in nodes}))
    supporting = tuple(edge for edge in edges if edge.supporting)
    adjacency: Dict[int, set[int]] = {frame: set() for frame in frames}
    for edge in supporting:
        if edge.earlier_frame_id in adjacency and edge.later_frame_id in adjacency:
            adjacency[edge.earlier_frame_id].add(edge.later_frame_id)
            adjacency[edge.later_frame_id].add(edge.earlier_frame_id)
    components = []
    unseen = set(frames)
    while unseen:
        start = min(unseen)
        stack = [start]
        component = set()
        while stack:
            current = stack.pop()
            if current in component:
                continue
            component.add(current)
            stack.extend(sorted(adjacency[current] - component, reverse=True))
        unseen -= component
        component_edges = tuple(
            edge
            for edge in supporting
            if edge.earlier_frame_id in component and edge.later_frame_id in component
        )
        components.append((tuple(sorted(component)), component_edges))
    qualified = [
        row
        for row in components
        if len(row[0]) >= MIN_COMPONENT_NODES and len(row[1]) >= MIN_COMPONENT_EDGES
    ]
    if not qualified:
        return None
    component_frames, component_edges = min(
        qualified,
        key=lambda row: (-len(row[0]), -len(row[1]), row[0]),
    )
    return ResidualCrossViewReceipt(
        track_id=track_id,
        confirmation_frame_id=confirmation_frame_id,
        component_frame_ids=component_frames,
        supporting_edges=component_edges,
    )


class CuTRResidualCrossViewR1:
    def __init__(
        self,
        config: ResidualCrossViewConfig,
        *,
        projection_adapter: Callable[..., object] = project_guide_metrics,
    ) -> None:
        if not isinstance(config, ResidualCrossViewConfig):
            raise ValueError("config must be ResidualCrossViewConfig")
        if not callable(projection_adapter):
            raise ValueError("projection_adapter must be callable")
        self.config = config
        self.enabled = bool(config.enabled)
        self.observer_only = True
        self._projection_adapter = projection_adapter
        self._tracks: Dict[int, _TrackState] = {}
        self._receipts: "OrderedDict[int, ResidualCrossViewReceipt]" = OrderedDict()
        self._last_frame_id: Optional[int] = None
        self._audit_complete = True
        self._closed = False
        self._observe_times_ms = deque(maxlen=_TIMING_WINDOW)
        self._wrapper_times_ms = deque(maxlen=_TIMING_WINDOW)
        self._close_result: Optional[ResidualCrossViewCloseResult] = None
        self._stats = {
            "keyframes": 0,
            "assignments": 0,
            "valid_nodes": 0,
            "invalid_nodes": 0,
            "projection_calls": 0,
            "projection_points": 0,
            "projection_failures": 0,
            "budget_abstained_rows": 0,
            "supporting_edges": 0,
            "receipts_created": 0,
            "receipt_archive_drops": 0,
            "retired_states_reclaimed": 0,
        }

    def _require_open(self) -> None:
        if not self.enabled:
            raise RuntimeError("cutr_residual_cross_view_r1 is disabled")
        if self._closed:
            raise RuntimeError("cutr_residual_cross_view_r1 is closed")

    def observe(
        self,
        *,
        frame_id: object,
        evidence_rows: Sequence[ResidualCrossViewEvidence],
        base_result: ResidualKeyframeResult,
        depth_m: object,
        K: object,
        T_wc: object,
    ) -> ResidualCrossViewKeyframeResult:
        self._require_open()
        frame = _strict_int("frame_id", frame_id)
        if self._last_frame_id is not None and frame <= self._last_frame_id:
            raise ValueError("R1 frame ids must be strictly increasing")
        if not isinstance(base_result, ResidualKeyframeResult) or base_result.frame_id != frame:
            raise ValueError("base_result must be the same-frame residual result")
        if isinstance(evidence_rows, (str, bytes)) or not isinstance(evidence_rows, Sequence):
            raise ValueError("evidence_rows must be a sequence")
        rows = tuple(evidence_rows)
        assignment_raw = tuple(row.raw_index for row in base_result.assignments)
        if assignment_raw != base_result.accepted_raw_indices:
            raise ValueError("base assignments are not aligned")
        if len(set(assignment_raw)) != len(assignment_raw):
            raise ValueError("base assignments contain duplicate raw ids")
        if tuple(row.raw_index for row in rows) != assignment_raw:
            raise ValueError("R1 evidence must align with base assignments")
        if any(not isinstance(row, ResidualCrossViewEvidence) or row.frame_id != frame for row in rows):
            raise ValueError("R1 evidence rows must belong to the current frame")
        current_pose = _proper_pose(T_wc, "T_wc")
        if any(
            row.valid
            and not np.allclose(
                row.camera_to_world, current_pose, rtol=0.0, atol=1e-8
            )
            for row in rows
        ):
            raise ValueError("R1 evidence pose must equal the current depth pose")

        started = time.perf_counter()
        tracks = dict(self._tracks)
        receipts = OrderedDict(self._receipts)
        remaining = PROJECTION_BUDGET_POINTS
        used = 0
        invalid_raw = []
        budget_raw = []
        newly_confirmed = []
        projection_calls = 0
        projection_failures = 0
        supporting_edges = 0

        assignment_by_raw = {row.raw_index: row for row in base_result.assignments}
        # Rows closest to the three-node confirmation are scheduled first;
        # ties are stable track/raw ids.  This affects only bounded shadow work.
        ordered_rows = sorted(
            rows,
            key=lambda row: (
                max(
                    0,
                    MIN_COMPONENT_NODES
                    - len(tracks.get(assignment_by_raw[row.raw_index].track_id, _TrackState(-1, (), ())).nodes),
                ),
                assignment_by_raw[row.raw_index].track_id,
                row.raw_index,
            ),
        )
        for row in ordered_rows:
            assignment = assignment_by_raw[row.raw_index]
            track_id = assignment.track_id
            if not row.valid:
                invalid_raw.append(row.raw_index)
                continue
            previous = tracks.get(track_id, _TrackState(track_id, (), ()))
            if any(node.frame_id == frame for node in previous.nodes):
                raise RuntimeError("a residual track received two nodes in one frame")
            new_edges = []
            row_projection_failed = False
            required = sum(len(node.guide_points_world) for node in previous.nodes)
            if track_id not in receipts and required > remaining:
                budget_raw.append(row.raw_index)
            elif track_id not in receipts:
                # Row-level preflight above guarantees all-or-none projection.
                for historical in previous.nodes:
                    # Charge the bounded work before invoking the adapter.
                    # A projection that computes and then fails validation is
                    # still real runtime work and must not reopen the budget.
                    points = len(historical.guide_points_world)
                    remaining -= points
                    used += points
                    projection_calls += 1
                    try:
                        metrics = self._projection_adapter(
                            historical.guide_points_world,
                            depth_m,
                            K,
                            current_pose,
                            proposal_box_xyxy=row.raw_box_xyxy,
                            alpha=DEPTH_ALPHA,
                        )
                        vf = _metric(metrics, "v_f")
                        df = _metric(metrics, "d_f")
                        vb = _metric(metrics, "v_b")
                        db = _metric(metrics, "d_b")
                        affinity = _metric(metrics, "affinity_a")
                    except (
                        AttributeError,
                        KeyError,
                        TypeError,
                        ValueError,
                        FloatingPointError,
                    ):
                        projection_failures += 1
                        row_projection_failed = True
                        break
                    cosine = float(np.clip(np.dot(historical.descriptor, row.descriptor), -1.0, 1.0))
                    translation = float(
                        np.linalg.norm(
                            historical.camera_to_world[:3, 3]
                            - row.camera_to_world[:3, 3]
                        )
                    )
                    rotation = _rotation_degrees(
                        historical.camera_to_world, row.camera_to_world
                    )
                    supporting = bool(
                        (translation > TRANSLATION_GAP_M or rotation > ROTATION_GAP_DEG)
                        and cosine >= DESCRIPTOR_COSINE
                        and vf > FRAME_VISIBILITY
                        and vb > BOX_VISIBILITY
                    )
                    edge = ResidualCrossViewEdge(
                        earlier_frame_id=historical.frame_id,
                        later_frame_id=frame,
                        descriptor_cosine=cosine,
                        translation_m=translation,
                        rotation_deg=rotation,
                        ray_angle_deg=_ray_angle_degrees(historical, row),
                        frame_visibility=vf,
                        depth_consistency=df,
                        box_visibility=vb,
                        box_depth_consistency=db,
                        affinity=affinity,
                        supporting=supporting,
                    )
                    new_edges.append(edge)
                    supporting_edges += int(supporting)

                if row_projection_failed:
                    # A row is an atomic cross-view query.  Never let a
                    # successful prefix of its history create a false edge
                    # when a later history projection is invalid.
                    supporting_edges -= sum(edge.supporting for edge in new_edges)
                    new_edges = []

            temporary_nodes = previous.nodes + (row,)
            temporary_edges = previous.edges + tuple(new_edges)
            # The frozen history cap applies before confirmation.  A sixth
            # node may replace the oldest node, but it may not temporarily
            # create a six-node evidence graph that can never be audited from
            # the retained state.
            kept_nodes = temporary_nodes[-MAX_NODES_PER_TRACK:]
            kept_frames = {node.frame_id for node in kept_nodes}
            kept_edges = tuple(
                edge
                for edge in temporary_edges
                if edge.earlier_frame_id in kept_frames
                and edge.later_frame_id in kept_frames
            )
            if track_id not in receipts:
                receipt = _component_receipt(
                    track_id, frame, kept_nodes, kept_edges
                )
                if receipt is not None:
                    receipts[track_id] = receipt
                    newly_confirmed.append(track_id)
            tracks[track_id] = _TrackState(track_id, kept_nodes, kept_edges)

        retired_states = []
        for track_id in base_result.newly_retired_track_ids:
            if track_id in tracks:
                del tracks[track_id]
                retired_states.append(track_id)

        archive_drops = []
        while len(receipts) > MAX_RECEIPTS:
            dropped_id, _ = receipts.popitem(last=False)
            archive_drops.append(dropped_id)
        audit_complete = bool(
            self._audit_complete
            and base_result.audit_complete
            and not budget_raw
            and not archive_drops
            and projection_failures == 0
        )
        elapsed = (time.perf_counter() - started) * 1000.0
        self._tracks = tracks
        self._receipts = receipts
        self._last_frame_id = frame
        self._audit_complete = audit_complete
        self._observe_times_ms.append(elapsed)
        self._stats["keyframes"] += 1
        self._stats["assignments"] += len(rows)
        self._stats["valid_nodes"] += sum(row.valid for row in rows)
        self._stats["invalid_nodes"] += len(invalid_raw)
        self._stats["projection_calls"] += projection_calls
        self._stats["projection_points"] += used
        self._stats["projection_failures"] += projection_failures
        self._stats["budget_abstained_rows"] += len(budget_raw)
        self._stats["supporting_edges"] += supporting_edges
        self._stats["receipts_created"] += len(newly_confirmed)
        self._stats["receipt_archive_drops"] += len(archive_drops)
        self._stats["retired_states_reclaimed"] += len(retired_states)
        return ResidualCrossViewKeyframeResult(
            frame_id=frame,
            assignment_count=len(rows),
            valid_node_count=sum(row.valid for row in rows),
            invalid_raw_indices=tuple(invalid_raw),
            budget_abstained_raw_indices=tuple(budget_raw),
            projection_points_used=used,
            newly_confirmed_track_ids=tuple(newly_confirmed),
            retired_state_track_ids=tuple(retired_states),
            audit_complete=audit_complete,
            observe_ms=elapsed,
        )

    def record_wrapper_timing(self, milliseconds: object) -> None:
        value = _finite_float("wrapper timing", milliseconds)
        if value < 0.0:
            raise ValueError("wrapper timing must be non-negative")
        self._wrapper_times_ms.append(value)

    def close(self, base_result: ResidualCloseResult) -> ResidualCrossViewCloseResult:
        self._require_open()
        if not isinstance(base_result, ResidualCloseResult):
            raise ValueError("base_result must be ResidualCloseResult")
        admitted = tuple(
            candidate
            for candidate in base_result.candidates
            if candidate.track_id in self._receipts
        )
        rejected = tuple(
            candidate.track_id
            for candidate in base_result.candidates
            if candidate.track_id not in self._receipts
        )
        result = ResidualCrossViewCloseResult(
            candidates=admitted,
            admitted_track_ids=tuple(candidate.track_id for candidate in admitted),
            rejected_track_ids=rejected,
            audit_complete=bool(self._audit_complete and base_result.audit_complete),
        )
        self._closed = True
        self._close_result = result
        return result

    @staticmethod
    def _timing(values: Sequence[float]) -> Tuple[Optional[float], Optional[float], Optional[float]]:
        if not values:
            return None, None, None
        array = np.asarray(values, dtype=np.float64)
        return (
            float(array.mean()),
            float(np.percentile(array, 95)),
            float(array.max()),
        )

    def summary(self) -> Dict[str, object]:
        observe_mean, observe_p95, observe_max = self._timing(self._observe_times_ms)
        wrapper_mean, wrapper_p95, wrapper_max = self._timing(self._wrapper_times_ms)
        return {
            "schema": SCHEMA,
            "enabled": self.enabled,
            "observer_only": True,
            "active_authorized": False,
            "native_mutation_applied": False,
            "native_export_appended": False,
            "training_free": True,
            "online_learning": False,
            "gt_access": False,
            "clip_access": False,
            "cutr_descriptor_access": True,
            "descriptor_is_clip": False,
            "history_depth_frames_retained": 0,
            "descriptor_dim": DESCRIPTOR_DIM,
            "descriptor_cosine": DESCRIPTOR_COSINE,
            "translation_gap_m": TRANSLATION_GAP_M,
            "rotation_gap_deg": ROTATION_GAP_DEG,
            "depth_alpha": DEPTH_ALPHA,
            "frame_visibility": FRAME_VISIBILITY,
            "box_visibility": BOX_VISIBILITY,
            "min_component_nodes": MIN_COMPONENT_NODES,
            "min_component_edges": MIN_COMPONENT_EDGES,
            "max_nodes_per_track": MAX_NODES_PER_TRACK,
            "projection_budget_points": PROJECTION_BUDGET_POINTS,
            "max_receipts": MAX_RECEIPTS,
            "audit_complete": self._audit_complete,
            "closed": self._closed,
            "active_track_histories": len(self._tracks),
            "receipt_count": len(self._receipts),
            "receipts": [row.to_json_dict() for row in self._receipts.values()],
            **self._stats,
            "observe_time_mean_ms": observe_mean,
            "observe_time_p95_ms": observe_p95,
            "observe_time_max_ms": observe_max,
            "wrapper_time_mean_ms": wrapper_mean,
            "wrapper_time_p95_ms": wrapper_p95,
            "wrapper_time_max_ms": wrapper_max,
            "close_result": (
                None if self._close_result is None else self._close_result.to_json_dict()
            ),
        }


def build_cutr_residual_cross_view_r1(
    root_config: Optional[Mapping[str, object]] = None,
    *,
    projection_adapter: Callable[..., object] = project_guide_metrics,
) -> CuTRResidualCrossViewR1:
    if root_config is None:
        root_config = {}
    if not isinstance(root_config, Mapping):
        raise ValueError("root config must be a mapping")
    section = root_config.get("cutr_residual_cross_view_r1", {})
    if not isinstance(section, Mapping):
        raise ValueError("cutr_residual_cross_view_r1 must be a mapping")
    allowed = set(ResidualCrossViewConfig.__dataclass_fields__)
    unknown = sorted(set(section) - allowed)
    if unknown:
        raise ValueError("Unknown R1 config key(s): " + ", ".join(unknown))
    config = ResidualCrossViewConfig(**dict(section))
    if config.enabled:
        residual = root_config.get("cutr_residual_birth_lite")
        if not isinstance(residual, Mapping) or residual.get("enabled") is not True:
            raise ValueError("enabled R1 requires enabled cutr_residual_birth_lite")
        lifting = root_config.get("lifting")
        if isinstance(lifting, Mapping) and lifting.get("backend", "cutr") != "cutr":
            raise ValueError("R1 is authorized only with CuTR-native lifting")
    return CuTRResidualCrossViewR1(config, projection_adapter=projection_adapter)


__all__ = [
    "CuTRResidualCrossViewR1",
    "ResidualCrossViewCloseResult",
    "ResidualCrossViewConfig",
    "ResidualCrossViewEdge",
    "ResidualCrossViewEvidence",
    "ResidualCrossViewKeyframeResult",
    "ResidualCrossViewReceipt",
    "build_cutr_residual_cross_view_r1",
]
