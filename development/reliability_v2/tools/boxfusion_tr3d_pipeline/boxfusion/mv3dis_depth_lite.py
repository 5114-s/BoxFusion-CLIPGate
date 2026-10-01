"""Causal, training-free MV3DIS-style depth observer for BoxFusion.

This module intentionally does *not* implement an association override.  It
records two counterfactual signals only:

* source-guide self-projection quality (not a fusion weight);
* a conservative ``would_veto_birth`` diagnostic supported by two committed
  historical views of one Moon-QIM-lite candidate.

The geometry is injected through :class:`DepthGuideGeometryAdapter`.  The
default adapter is loaded lazily from ``depth_guide_geometry`` when that helper
is present.  No PUF, detector score, semantic feature, ground truth, learned
parameter, or online parameter update is read here.

The transaction order is strict: ``query`` sees the current RGB-D frame as a
projection target but may retrieve guides only from earlier commits.  The
current per-proposal guide enters history only in ``commit``, after native
association has supplied an aligned stable track id.
"""

from __future__ import annotations

from collections import OrderedDict, deque
from dataclasses import dataclass
from numbers import Integral, Real
from time import perf_counter_ns
from typing import Dict, Iterable, Mapping, Optional, Protocol, Sequence, Tuple

import numpy as np

from .moon_qim_lite import QIMQueryBatch


DEFAULT_MV3DIS_DEPTH_LITE_CONFIG = {
    "enabled": False,
    "observer_only": True,
    "max_guides_per_track": 5,
    "max_depth_frames": 80,
    "max_proposals": 256,
    "max_qim_candidates": 3,
    "projection_budget_points": 8192,
    "points_per_projection": 64,
    "frame_visibility_threshold": 0.30,
    "box_visibility_threshold": 0.90,
    "candidate_dominance_threshold": 0.90,
    "min_history_views": 2,
    "alpha": 0.05,
    "max_diagnostic_examples": 64,
}


@dataclass(frozen=True)
class DepthGuideProjectionMetrics:
    """Normalized adapter result.

    ``visibility/depth_consistency/quality`` are source-view (branch A)
    metrics.  ``frame_visibility/box_visibility/box_depth_consistency/affinity``
    are historical matching (branch B) metrics.  Every value is in ``[0, 1]``.
    """

    visibility: float
    depth_consistency: float
    quality: float
    frame_visibility: float
    box_visibility: float
    box_depth_consistency: float
    affinity: float


class DepthGuideGeometryAdapter(Protocol):
    """Adapter implemented by ``depth_guide_geometry.project_guide_metrics``."""

    def __call__(
        self,
        points_world: np.ndarray,
        depth_m: np.ndarray,
        K: np.ndarray,
        T_wc: np.ndarray,
        proposal_box_xyxy: Optional[Tuple[float, float, float, float]] = None,
        alpha: float = 0.05,
    ) -> object:
        ...


@dataclass(frozen=True)
class GuideQualityObservation:
    proposal_id: int
    valid: bool
    visibility: Optional[float]
    depth_consistency: Optional[float]
    guide_quality: Optional[float]
    projected_points: int
    reason: str


@dataclass(frozen=True)
class HistoricalViewEvidence:
    frame_id: int
    valid: bool
    frame_visibility: Optional[float]
    box_visibility: Optional[float]
    box_depth_consistency: Optional[float]
    affinity: Optional[float]
    supporting: bool
    projected_points: int
    reason: str


@dataclass(frozen=True)
class HistoricalCandidateEvidence:
    track_id: int
    qim_rank: int
    history_views_available: int
    history_views_evaluated: int
    supporting_views: int
    support_score: float
    mean_frame_visibility: Optional[float]
    mean_box_visibility: Optional[float]
    mean_box_depth_consistency: Optional[float]
    mean_affinity: Optional[float]
    projected_points: int
    complete: bool
    views: Tuple[HistoricalViewEvidence, ...]


@dataclass(frozen=True)
class BirthVetoObservation:
    proposal_id: int
    candidates: Tuple[HistoricalCandidateEvidence, ...]
    would_veto_birth: bool
    recommended_track_id: Optional[int]
    candidate_dominance: Optional[float]
    action: str
    reason: str


@dataclass(frozen=True)
class MV3DISDepthLiteBatch:
    scene_id: str
    frame_id: int
    history_max_frame_id: Optional[int]
    proposal_ids: Tuple[int, ...]
    guide_quality_rows: Tuple[GuideQualityObservation, ...]
    birth_veto_rows: Tuple[BirthVetoObservation, ...]
    guide_quality_projection_points_used: int
    birth_veto_projection_points_used: int
    guide_quality_budget_exhausted: bool
    birth_veto_budget_exhausted: bool
    proposal_cap_exceeded: bool
    current_frame_valid: bool
    invalid_frame_reason: Optional[str]
    query_ms: float


@dataclass(frozen=True)
class MV3DISDepthLiteSnapshot:
    scene_id: Optional[str]
    history_max_frame_id: Optional[int]
    committed_frame_ids: Tuple[int, ...]
    track_guide_counts: Tuple[Tuple[int, int], ...]
    total_guides: int
    pending_frame_id: Optional[int]


@dataclass(frozen=True)
class _DepthFrameRecord:
    frame_id: int
    depth_m: np.ndarray
    K: np.ndarray
    T_wc: np.ndarray


@dataclass(frozen=True)
class _GuideRecord:
    track_id: int
    proposal_id: int
    frame_id: int
    points_world: np.ndarray
    proposal_box_xyxy: Optional[Tuple[float, float, float, float]]
    source_quality: Optional[float]


@dataclass(frozen=True)
class _PendingProposal:
    proposal_id: int
    points_world: np.ndarray
    proposal_box_xyxy: Optional[Tuple[float, float, float, float]]


@dataclass(frozen=True)
class _PendingPayload:
    batch: MV3DISDepthLiteBatch
    frame: Optional[_DepthFrameRecord]
    proposals: Tuple[_PendingProposal, ...]


def _strict_bool(name: str, value: object) -> bool:
    if not isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a boolean")
    return bool(value)


def _strict_int(name: str, value: object, minimum: int) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer")
    result = int(value)
    if result < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return result


def _finite_float(name: str, value: object, minimum: float) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real number")
    result = float(value)
    if not np.isfinite(result) or result < minimum:
        raise ValueError(f"{name} must be finite and at least {minimum}")
    return result


def _finite_real(name: str, value: object) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real number")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be a finite real number")
    return result


def resolve_mv3dis_depth_lite_config(
    config: Optional[Mapping[str, object]] = None,
) -> Dict[str, object]:
    """Validate the fail-closed S0 configuration."""

    if config is None:
        config = {}
    if not isinstance(config, Mapping):
        raise ValueError("mv3dis_depth_lite config must be a mapping")
    unknown = sorted(set(config) - set(DEFAULT_MV3DIS_DEPTH_LITE_CONFIG))
    if unknown:
        raise ValueError(
            "Unknown mv3dis_depth_lite config key(s): " + ", ".join(unknown)
        )
    resolved = dict(DEFAULT_MV3DIS_DEPTH_LITE_CONFIG)
    resolved.update(config)
    resolved["enabled"] = _strict_bool(
        "mv3dis_depth_lite.enabled", resolved["enabled"]
    )
    resolved["observer_only"] = _strict_bool(
        "mv3dis_depth_lite.observer_only", resolved["observer_only"]
    )
    if resolved["enabled"] and not resolved["observer_only"]:
        raise ValueError(
            "mv3dis_depth_lite active fusion/association is not authorized; "
            "observer_only must remain true"
        )
    integer_limits = {
        "max_guides_per_track": (1, 5),
        "max_depth_frames": (1, 80),
        "max_proposals": (1, 256),
        "max_qim_candidates": (1, 3),
        "projection_budget_points": (1, 8192),
        "points_per_projection": (1, 64),
        "min_history_views": (1, 5),
        "max_diagnostic_examples": (0, 1024),
    }
    for key, (minimum, maximum) in integer_limits.items():
        resolved[key] = _strict_int(
            f"mv3dis_depth_lite.{key}", resolved[key], minimum
        )
        if resolved[key] > maximum:
            raise ValueError(
                f"mv3dis_depth_lite.{key} must not exceed {maximum}"
            )
    if resolved["min_history_views"] > resolved["max_guides_per_track"]:
        raise ValueError(
            "mv3dis_depth_lite.min_history_views must not exceed "
            "max_guides_per_track"
        )
    for key in (
        "frame_visibility_threshold",
        "box_visibility_threshold",
        "candidate_dominance_threshold",
        "alpha",
    ):
        resolved[key] = _finite_float(
            f"mv3dis_depth_lite.{key}", resolved[key], 0.0
        )
        if resolved[key] > 1.0:
            raise ValueError(f"mv3dis_depth_lite.{key} must not exceed 1")
    if resolved["enabled"]:
        frozen = {
            "frame_visibility_threshold": 0.30,
            "box_visibility_threshold": 0.90,
            "candidate_dominance_threshold": 0.90,
            "min_history_views": 2,
            "alpha": 0.05,
            "points_per_projection": 64,
        }
        changed = [key for key, value in frozen.items() if resolved[key] != value]
        if changed:
            raise ValueError(
                "enabled mv3dis_depth_lite must keep frozen S0 thresholds: "
                + ", ".join(changed)
            )
    return resolved


def _normalized_groups(
    groups: Sequence[Iterable[int]], name: str
) -> Tuple[Tuple[int, ...], ...]:
    if isinstance(groups, (str, bytes)) or not isinstance(groups, Sequence):
        raise ValueError(f"{name} must be a sequence of integer sequences")
    result = []
    for index, group in enumerate(groups):
        if isinstance(group, (str, bytes)):
            raise ValueError(f"{name}[{index}] must be an integer sequence")
        try:
            raw = tuple(group)
        except TypeError as error:
            raise ValueError(
                f"{name}[{index}] must be an integer sequence"
            ) from error
        if not raw:
            raise ValueError(f"{name}[{index}] must not be empty")
        values = []
        for value in raw:
            if isinstance(value, (bool, np.bool_)) or not isinstance(
                value, Integral
            ):
                raise ValueError(f"{name}[{index}] must contain integers")
            if int(value) < 0:
                raise ValueError(f"{name}[{index}] must be non-negative")
            values.append(int(value))
        result.append(tuple(sorted(set(values))))
    return tuple(result)


def derive_committed_track_ids(
    *,
    proposal_ids: object,
    current_fusion_groups: Sequence[Iterable[int]],
    current_stable_ids: object,
    association_events: Sequence[Mapping[str, object]] = (),
) -> Tuple[Optional[int], ...]:
    """Resolve each proposal to one post-association stable track id.

    Current fusion groups alone lose proposal membership when BoxFusion's
    five-view cap is full.  This pure helper therefore unions both final group
    members and every recorded winner/loser merge event.  A proposal is
    commit-safe only when its connected component reaches exactly one current
    stable-id row; zero or multiple rows fail closed to ``None``.
    """

    raw_proposals = np.asarray(proposal_ids)
    if raw_proposals.ndim != 1 or not np.issubdtype(
        raw_proposals.dtype, np.integer
    ):
        raise ValueError("proposal_ids must be a one-dimensional integer array")
    proposals = raw_proposals.astype(np.int64, copy=False)
    if np.any(proposals < 0) or len(np.unique(proposals)) != len(proposals):
        raise ValueError("proposal_ids must be unique and non-negative")

    groups = _normalized_groups(current_fusion_groups, "current_fusion_groups")
    raw_stable = np.asarray(current_stable_ids)
    if raw_stable.ndim != 1 or len(raw_stable) != len(groups) or not np.issubdtype(
        raw_stable.dtype, np.integer
    ):
        raise ValueError(
            "current_stable_ids must be a row-aligned integer array"
        )
    stable_ids = raw_stable.astype(np.int64, copy=False)
    if np.any(stable_ids < 0) or len(np.unique(stable_ids)) != len(stable_ids):
        raise ValueError("current_stable_ids must be unique and non-negative")
    if isinstance(association_events, (str, bytes)) or not isinstance(
        association_events, Sequence
    ):
        raise ValueError("association_events must be a sequence of mappings")

    parent: Dict[int, int] = {}

    def find(value: int) -> int:
        parent.setdefault(value, value)
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root == right_root:
            return
        if left_root < right_root:
            parent[right_root] = left_root
        else:
            parent[left_root] = right_root

    evidenced = set()
    for group in groups:
        evidenced.update(group)
        find(group[0])
        for member in group[1:]:
            union(group[0], member)

    for index, event in enumerate(association_events):
        if not isinstance(event, Mapping):
            raise ValueError(f"association_events[{index}] must be a mapping")
        unknown = set(event) - {"stage", "winner_members", "loser_members"}
        if unknown:
            raise ValueError(
                f"association_events[{index}] has unknown keys: "
                + ", ".join(sorted(unknown))
            )
        members = []
        for key in ("winner_members", "loser_members"):
            normalized = _normalized_groups(
                [event.get(key, ())], f"association_events[{index}].{key}"
            )
            members.extend(normalized[0])
        evidenced.update(members)
        find(members[0])
        for member in members[1:]:
            union(members[0], member)

    component_stable_ids: Dict[int, set[int]] = {}
    for group, stable_id in zip(groups, stable_ids):
        root = find(group[0])
        component_stable_ids.setdefault(root, set()).add(int(stable_id))

    result = []
    for raw_proposal in proposals:
        proposal = int(raw_proposal)
        if proposal not in evidenced:
            result.append(None)
            continue
        matched = component_stable_ids.get(find(proposal), set())
        result.append(next(iter(matched)) if len(matched) == 1 else None)
    return tuple(result)


def _frame_id(value: object) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise ValueError("frame_id must be a non-negative integer")
    result = int(value)
    if result < 0:
        raise ValueError("frame_id must be a non-negative integer")
    return result


def _readonly_array(
    value: object,
    name: str,
    *,
    shape: Optional[Tuple[int, ...]] = None,
    ndim: Optional[int] = None,
    nonnegative: bool = False,
    finite: bool = True,
    dtype: np.dtype = np.float64,
) -> np.ndarray:
    candidate = value
    if hasattr(candidate, "detach"):
        candidate = candidate.detach()
    if hasattr(candidate, "cpu"):
        candidate = candidate.cpu()
    if hasattr(candidate, "numpy"):
        candidate = candidate.numpy()
    try:
        array = np.asarray(candidate, dtype=dtype)
    except Exception as error:
        raise ValueError(f"{name} cannot be converted to NumPy") from error
    if shape is not None and array.shape != shape:
        raise ValueError(f"{name} must have shape {list(shape)}")
    if ndim is not None and array.ndim != ndim:
        raise ValueError(f"{name} must have {ndim} dimensions")
    if finite and not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    if nonnegative and np.any(array[np.isfinite(array)] < 0.0):
        raise ValueError(f"{name} must be non-negative")
    result = np.array(array, dtype=dtype, order="C", copy=True)
    result.setflags(write=False)
    return result


def _validated_current_frame(
    *, frame_id: int, depth_m: object, K: object, T_wc: object
) -> Tuple[Optional[_DepthFrameRecord], Optional[str]]:
    """Validate an RGB-D target once; sensor failures abstain batch-wide."""

    try:
        depth = _readonly_array(
            depth_m,
            "depth_m",
            ndim=2,
            finite=False,
            dtype=np.float32,
        )
    except ValueError:
        return None, "invalid_depth_image"
    if depth.shape[0] < 1 or depth.shape[1] < 1:
        return None, "invalid_depth_image"
    valid_depth = (
        np.isfinite(depth) & (depth >= 0.10) & (depth <= 8.0)
    )
    if float(np.count_nonzero(valid_depth) / depth.size) < 0.01:
        return None, "insufficient_valid_depth"

    try:
        intrinsic = _readonly_array(K, "K", shape=(3, 3))
    except ValueError:
        return None, "invalid_intrinsics"
    height, width = depth.shape
    if (
        intrinsic[0, 0] <= 0.0
        or intrinsic[1, 1] <= 0.0
        or not np.allclose(
            intrinsic[2], (0.0, 0.0, 1.0), rtol=0.0, atol=1e-10
        )
        or not (0.0 <= intrinsic[0, 2] < width)
        or not (0.0 <= intrinsic[1, 2] < height)
        or abs(float(np.linalg.det(intrinsic))) <= 1e-12
    ):
        return None, "invalid_intrinsics"

    try:
        pose = _readonly_array(T_wc, "T_wc", shape=(4, 4))
    except ValueError:
        return None, "invalid_pose"
    rotation = pose[:3, :3]
    if (
        not np.allclose(
            pose[3], (0.0, 0.0, 0.0, 1.0), rtol=0.0, atol=1e-8
        )
        or not np.allclose(
            rotation.T @ rotation, np.eye(3), rtol=0.0, atol=1e-5
        )
        or not np.isclose(
            np.linalg.det(rotation), 1.0, rtol=0.0, atol=1e-5
        )
    ):
        return None, "invalid_pose"
    return (
        _DepthFrameRecord(
            frame_id=frame_id,
            depth_m=depth,
            K=intrinsic,
            T_wc=pose,
        ),
        None,
    )


def _proposal_points(
    values: object, count: int, max_points: int
) -> Tuple[np.ndarray, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise ValueError("proposal_points_world must be a sequence")
    if len(values) != count:
        raise ValueError("proposal_points_world must align with QIM proposals")
    result = []
    for index, value in enumerate(values):
        points = _readonly_array(
            value, f"proposal_points_world[{index}]", ndim=2
        )
        if points.shape[1:] != (3,):
            raise ValueError(
                f"proposal_points_world[{index}] must have shape [N, 3]"
            )
        if len(points) > max_points:
            indices = np.linspace(
                0, len(points) - 1, max_points, dtype=np.int64
            )
            points = np.array(
                points[indices], dtype=np.float64, order="C", copy=True
            )
            points.setflags(write=False)
        result.append(points)
    return tuple(result)


def _proposal_boxes(
    values: Optional[Sequence[Optional[Sequence[float]]]], count: int
) -> Tuple[Optional[Tuple[float, float, float, float]], ...]:
    if values is None:
        return (None,) * count
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise ValueError("proposal_boxes_xyxy must be a sequence")
    if len(values) != count:
        raise ValueError("proposal_boxes_xyxy must align with QIM proposals")
    result = []
    for index, value in enumerate(values):
        if value is None:
            result.append(None)
            continue
        try:
            raw = tuple(value)
        except TypeError as error:
            raise ValueError(
                f"proposal_boxes_xyxy[{index}] must contain four values"
            ) from error
        if len(raw) != 4:
            raise ValueError(
                f"proposal_boxes_xyxy[{index}] must contain four values"
            )
        box = tuple(
            _finite_real(f"proposal_boxes_xyxy[{index}]", item) for item in raw
        )
        if box[2] <= box[0] or box[3] <= box[1]:
            raise ValueError(
                f"proposal_boxes_xyxy[{index}] must have positive area"
            )
        result.append(box)
    return tuple(result)


def _metric_field(result: object, names: Sequence[str]) -> object:
    for name in names:
        if isinstance(result, Mapping) and name in result:
            return result[name]
        if hasattr(result, name):
            return getattr(result, name)
    raise ValueError("geometry result is missing " + "/".join(names))


def _normalize_metrics(
    result: object, *, require_backward: bool
) -> DepthGuideProjectionMetrics:
    aliases = {
        "visibility": ("visibility", "v_f"),
        "depth_consistency": ("depth_consistency", "d_f"),
        "quality": ("quality", "q_f"),
        "frame_visibility": ("frame_visibility", "v_f", "visibility"),
        "box_visibility": ("box_visibility", "v_b"),
        "box_depth_consistency": ("box_depth_consistency", "d_b"),
        "affinity": ("affinity", "affinity_a"),
    }
    values = {}
    for name, field_aliases in aliases.items():
        try:
            raw = _metric_field(result, field_aliases)
        except ValueError:
            if not require_backward and name in {
                "box_visibility",
                "box_depth_consistency",
                "affinity",
            }:
                raw = 0.0
            else:
                raise
        if raw is None and not require_backward and name in {
            "box_visibility",
            "box_depth_consistency",
            "affinity",
        }:
            raw = 0.0
        value = _finite_float(f"geometry.{name}", raw, 0.0)
        if value > 1.0:
            raise ValueError(f"geometry.{name} must not exceed 1")
        values[name] = value
    return DepthGuideProjectionMetrics(**values)


def _default_projection_adapter() -> Optional[DepthGuideGeometryAdapter]:
    try:
        from .depth_guide_geometry import project_guide_metrics
    except (ImportError, ModuleNotFoundError):
        return None
    return project_guide_metrics


class MV3DISDepthLiteObserver:
    """Bounded MV3DIS depth-quality and birth-veto shadow observer."""

    _LATENCY_WINDOW = 2048

    def __init__(
        self,
        config: Optional[Mapping[str, object]] = None,
        *,
        projection_adapter: Optional[DepthGuideGeometryAdapter] = None,
    ):
        self.config = resolve_mv3dis_depth_lite_config(config)
        self.enabled = bool(self.config["enabled"])
        self.observer_only = bool(self.config["observer_only"])
        self._projection_adapter = (
            projection_adapter
            if projection_adapter is not None
            else _default_projection_adapter()
        )
        if self._projection_adapter is not None and not callable(
            self._projection_adapter
        ):
            raise ValueError("projection_adapter must be callable")
        self._scene_id: Optional[str] = None
        self._last_query_frame_id: Optional[int] = None
        self._last_commit_frame_id: Optional[int] = None
        self._pending: Optional[_PendingPayload] = None
        self._last_committed_batch: Optional[MV3DISDepthLiteBatch] = None
        # Only frame ids are retained.  Historical depth/K/T arrays would cost
        # hundreds of MB and are unnecessary because old guide points are
        # projected into the current frame.
        self._committed_frames: "OrderedDict[int, None]" = OrderedDict()
        self._track_guides: Dict[int, Tuple[_GuideRecord, ...]] = {}
        self._query_samples = deque(maxlen=self._LATENCY_WINDOW)
        self._commit_samples = deque(maxlen=self._LATENCY_WINDOW)
        self._invalid_frame_reasons: Dict[str, int] = {}
        self._diagnostic_examples: list[Tuple[object, ...]] = []
        self._stats = self._new_stats()

    @staticmethod
    def _new_stats() -> Dict[str, object]:
        return {
            "queries": 0,
            "commits": 0,
            "proposals": 0,
            "proposal_cap_batches": 0,
            "invalid_frame_batches": 0,
            "guide_quality_rows_valid": 0,
            "guide_quality_rows_invalid": 0,
            "veto_recommendations": 0,
            "veto_evaluable": 0,
            "veto_correct": 0,
            "veto_wrong": 0,
            "veto_on_native_birth": 0,
            "native_history": 0,
            "native_birth": 0,
            "native_unresolved": 0,
            "native_diagnostics_skipped": 0,
            "geometry_calls": 0,
            "geometry_errors": 0,
            "projection_points": 0,
            "guide_quality_projection_points": 0,
            "birth_veto_projection_points": 0,
            "guide_quality_budget_exhaustions": 0,
            "birth_veto_budget_exhaustions": 0,
            "guides_committed": 0,
            "guides_replaced_same_frame": 0,
            "guides_evicted_track_cap": 0,
            "guides_evicted_frame_cap": 0,
            "committed_frames_evicted": 0,
            "max_committed_frames_observed": 0,
            "max_tracks_observed": 0,
            "max_guides_observed": 0,
            "query_ms_total": 0.0,
            "query_ms_max": 0.0,
            "commit_ms_total": 0.0,
            "commit_ms_max": 0.0,
            "pipeline_query_calls": 0,
            "pipeline_query_ms_total": 0.0,
            "pipeline_query_ms_max": 0.0,
            "pipeline_commit_calls": 0,
            "pipeline_commit_ms_total": 0.0,
            "pipeline_commit_ms_max": 0.0,
        }

    @property
    def scene_id(self) -> Optional[str]:
        return self._scene_id

    def reset_scene(self, scene_id: str) -> None:
        scene_id = str(scene_id)
        if not scene_id:
            raise ValueError("scene_id must not be empty")
        self._scene_id = scene_id
        self._last_query_frame_id = None
        self._last_commit_frame_id = None
        self._pending = None
        self._last_committed_batch = None
        self._committed_frames.clear()
        self._track_guides.clear()
        self._query_samples.clear()
        self._commit_samples.clear()
        self._invalid_frame_reasons.clear()
        self._diagnostic_examples.clear()
        self._stats = self._new_stats()

    def _bind_scene(self, scene_id: str) -> str:
        scene_id = str(scene_id)
        if not scene_id:
            raise ValueError("scene_id must not be empty")
        if self._scene_id is None:
            self.reset_scene(scene_id)
        elif self._scene_id != scene_id:
            raise ValueError(
                f"mv3dis_depth_lite is bound to {self._scene_id}, not {scene_id}"
            )
        return scene_id

    @staticmethod
    def _sample_points(points: np.ndarray, limit: int) -> np.ndarray:
        if len(points) <= limit:
            return points
        # Endpoint-preserving deterministic subsampling; no RNG or learned state.
        indices = np.linspace(0, len(points) - 1, limit, dtype=np.int64)
        sampled = np.array(points[indices], dtype=np.float64, order="C", copy=True)
        sampled.setflags(write=False)
        return sampled

    def _project(
        self,
        *,
        points_world: np.ndarray,
        frame: _DepthFrameRecord,
        proposal_box_xyxy: Optional[Tuple[float, float, float, float]],
        remaining_budget: int,
        require_backward: bool,
    ) -> Tuple[Optional[DepthGuideProjectionMetrics], int, Optional[str]]:
        if self._projection_adapter is None:
            return None, 0, "geometry_adapter_unavailable"
        if len(points_world) == 0:
            return None, 0, "empty_guide_points"
        if remaining_budget <= 0:
            return None, 0, "projection_budget_exhausted"
        count = min(
            len(points_world),
            int(self.config["points_per_projection"]),
            remaining_budget,
        )
        sampled = self._sample_points(points_world, count)
        try:
            raw = self._projection_adapter(
                sampled,
                frame.depth_m,
                frame.K,
                frame.T_wc,
                proposal_box_xyxy=proposal_box_xyxy,
                alpha=float(self.config["alpha"]),
            )
            metrics = _normalize_metrics(
                raw, require_backward=require_backward
            )
        except Exception as error:  # online observer must abstain on bad geometry
            self._stats["geometry_errors"] += 1
            return None, count, f"geometry_error:{type(error).__name__}"
        self._stats["geometry_calls"] += 1
        return metrics, count, None

    @staticmethod
    def _mean(values: Sequence[float]) -> Optional[float]:
        return float(sum(values) / len(values)) if values else None

    def _candidate_ids(self, raw_candidates: Sequence[object]) -> Tuple[int, ...]:
        result = []
        seen = set()
        for candidate in raw_candidates:
            if not bool(getattr(candidate, "active_at_last_commit", False)):
                continue
            track_id = getattr(candidate, "track_id", None)
            if (
                isinstance(track_id, (bool, np.bool_))
                or not isinstance(track_id, Integral)
                or int(track_id) < 0
            ):
                continue
            track_id = int(track_id)
            if track_id in seen:
                continue
            seen.add(track_id)
            result.append(track_id)
            if len(result) >= int(self.config["max_qim_candidates"]):
                break
        return tuple(result)

    def query(
        self,
        *,
        qim_batch: QIMQueryBatch,
        proposal_points_world: Sequence[object],
        depth_m: object,
        K: object,
        T_wc: object,
        proposal_boxes_xyxy: Optional[
            Sequence[Optional[Sequence[float]]]
        ] = None,
    ) -> MV3DISDepthLiteBatch:
        """Record depth evidence using only history from earlier commits."""

        start = perf_counter_ns()
        if not self.enabled:
            raise RuntimeError("mv3dis_depth_lite observer is disabled")
        if self._pending is not None:
            raise ValueError("previous depth query must be closed by commit")
        if not isinstance(qim_batch, QIMQueryBatch):
            raise ValueError("qim_batch must be a QIMQueryBatch")
        scene_id = self._bind_scene(qim_batch.scene_id)
        frame_id = _frame_id(qim_batch.frame_id)
        if self._last_query_frame_id is not None and frame_id <= self._last_query_frame_id:
            raise ValueError("depth query frame ids must be strictly increasing")
        if self._last_commit_frame_id is not None and frame_id <= self._last_commit_frame_id:
            raise ValueError("depth query must follow the previous commit")
        if qim_batch.history_max_frame_id is not None:
            qim_history = _frame_id(qim_batch.history_max_frame_id)
            if qim_history >= frame_id:
                raise ValueError("QIM history must precede the current depth query")
        proposal_ids_list = []
        for value in qim_batch.proposal_ids:
            if isinstance(value, (bool, np.bool_)) or not isinstance(
                value, Integral
            ):
                raise ValueError("QIM proposal ids must contain integers")
            if int(value) < 0:
                raise ValueError("QIM proposal ids must be non-negative")
            proposal_ids_list.append(int(value))
        proposal_ids = tuple(proposal_ids_list)
        if len(set(proposal_ids)) != len(proposal_ids):
            raise ValueError("QIM proposal ids must be unique")
        if len(qim_batch.candidates) != len(proposal_ids):
            raise ValueError("QIM candidate rows must align with proposal ids")

        points = _proposal_points(
            proposal_points_world,
            len(proposal_ids),
            int(self.config["points_per_projection"]),
        )
        boxes = _proposal_boxes(proposal_boxes_xyxy, len(proposal_ids))
        frame, invalid_frame_reason = _validated_current_frame(
            frame_id=frame_id,
            depth_m=depth_m,
            K=K,
            T_wc=T_wc,
        )
        proposals = tuple(
            _PendingProposal(proposal_id, proposal_points, box)
            for proposal_id, proposal_points, box in zip(proposal_ids, points, boxes)
        )
        cap_exceeded = len(proposal_ids) > int(self.config["max_proposals"])
        # A and B have independent fixed point budgets.  Guide self-quality
        # must never consume evidence capacity from the birth-veto branch.
        quality_remaining = int(self.config["projection_budget_points"])
        veto_remaining = int(self.config["projection_budget_points"])
        quality_used = 0
        veto_used = 0

        quality_by_id: Dict[int, GuideQualityObservation] = {}
        if cap_exceeded or frame is None:
            fail_reason = (
                "proposal_cap_defer_to_native"
                if cap_exceeded
                else f"{invalid_frame_reason}_defer_to_native"
            )
            for proposal in proposals:
                quality_by_id[proposal.proposal_id] = GuideQualityObservation(
                    proposal_id=proposal.proposal_id,
                    valid=False,
                    visibility=None,
                    depth_consistency=None,
                    guide_quality=None,
                    projected_points=0,
                    reason=fail_reason,
                )
        else:
            # This is source-guide self-projection quality, not a cross-view
            # BoxFusion weight.  It has its own independent budget.
            for proposal in sorted(proposals, key=lambda item: item.proposal_id):
                metrics, consumed, error = self._project(
                    points_world=proposal.points_world,
                    frame=frame,
                    proposal_box_xyxy=proposal.proposal_box_xyxy,
                    remaining_budget=quality_remaining,
                    require_backward=False,
                )
                quality_remaining -= consumed
                quality_used += consumed
                if metrics is None:
                    quality_by_id[proposal.proposal_id] = GuideQualityObservation(
                        proposal_id=proposal.proposal_id,
                        valid=False,
                        visibility=None,
                        depth_consistency=None,
                        guide_quality=None,
                        projected_points=consumed,
                        reason=str(error),
                    )
                else:
                    quality_by_id[proposal.proposal_id] = GuideQualityObservation(
                        proposal_id=proposal.proposal_id,
                        valid=True,
                        visibility=metrics.visibility,
                        depth_consistency=metrics.depth_consistency,
                        guide_quality=metrics.quality,
                        projected_points=consumed,
                        reason="record_only",
                    )

        veto_by_id: Dict[int, BirthVetoObservation] = {}
        if cap_exceeded or frame is None:
            fail_reason = (
                "proposal_cap_defer_to_native"
                if cap_exceeded
                else f"{invalid_frame_reason}_defer_to_native"
            )
            for proposal in proposals:
                veto_by_id[proposal.proposal_id] = BirthVetoObservation(
                    proposal_id=proposal.proposal_id,
                    candidates=(),
                    would_veto_birth=False,
                    recommended_track_id=None,
                    candidate_dominance=None,
                    action="defer_to_native",
                    reason=fail_reason,
                )
        else:
            row_by_id = {
                proposal_id: raw_candidates
                for proposal_id, raw_candidates in zip(
                    proposal_ids, qim_batch.candidates
                )
            }
            for proposal in sorted(proposals, key=lambda item: item.proposal_id):
                candidate_rows = []
                row_complete = True
                for qim_rank, track_id in enumerate(
                    self._candidate_ids(row_by_id[proposal.proposal_id])
                ):
                    guides = tuple(
                        sorted(
                            self._track_guides.get(track_id, ()),
                            key=lambda guide: guide.frame_id,
                            reverse=True,
                        )
                    )
                    frame_visibility = []
                    box_visibility = []
                    box_depth_consistency = []
                    affinities = []
                    supporting = 0
                    candidate_points = 0
                    evaluated = 0
                    complete = True
                    view_rows = []
                    for guide in guides:
                        if guide.frame_id >= frame_id:
                            raise RuntimeError(
                                "future guide reached causal depth query"
                            )
                        if guide.frame_id not in self._committed_frames:
                            complete = False
                            view_rows.append(
                                HistoricalViewEvidence(
                                    frame_id=guide.frame_id,
                                    valid=False,
                                    frame_visibility=None,
                                    box_visibility=None,
                                    box_depth_consistency=None,
                                    affinity=None,
                                    supporting=False,
                                    projected_points=0,
                                    reason="expired_frame_window",
                                )
                            )
                            continue
                        # MV3DIS direction: project each *committed historical
                        # guide* into the current RGB-D target and current raw
                        # proposal box.  The current points are never treated
                        # as historical evidence before commit.
                        metrics, consumed, error = self._project(
                            points_world=guide.points_world,
                            frame=frame,
                            proposal_box_xyxy=proposal.proposal_box_xyxy,
                            remaining_budget=veto_remaining,
                            require_backward=True,
                        )
                        veto_remaining -= consumed
                        veto_used += consumed
                        candidate_points += consumed
                        if metrics is None:
                            complete = False
                            view_rows.append(
                                HistoricalViewEvidence(
                                    frame_id=guide.frame_id,
                                    valid=False,
                                    frame_visibility=None,
                                    box_visibility=None,
                                    box_depth_consistency=None,
                                    affinity=None,
                                    supporting=False,
                                    projected_points=consumed,
                                    reason=str(error),
                                )
                            )
                            continue
                        evaluated += 1
                        frame_visibility.append(metrics.frame_visibility)
                        box_visibility.append(metrics.box_visibility)
                        box_depth_consistency.append(
                            metrics.box_depth_consistency
                        )
                        affinities.append(metrics.affinity)
                        supporting_view = (
                            metrics.frame_visibility
                            > float(self.config["frame_visibility_threshold"])
                            and metrics.box_visibility
                            > float(self.config["box_visibility_threshold"])
                        )
                        if supporting_view:
                            supporting += 1
                        view_rows.append(
                            HistoricalViewEvidence(
                                frame_id=guide.frame_id,
                                valid=True,
                                frame_visibility=metrics.frame_visibility,
                                box_visibility=metrics.box_visibility,
                                box_depth_consistency=(
                                    metrics.box_depth_consistency
                                ),
                                affinity=metrics.affinity,
                                supporting=bool(supporting_view),
                                projected_points=consumed,
                                reason="support" if supporting_view else "below_threshold",
                            )
                        )
                    complete = complete and evaluated == len(guides)
                    row_complete = row_complete and complete
                    candidate_rows.append(
                        HistoricalCandidateEvidence(
                            track_id=track_id,
                            qim_rank=qim_rank,
                            history_views_available=len(guides),
                            history_views_evaluated=evaluated,
                            supporting_views=supporting,
                            support_score=float(
                                sum(
                                    affinity
                                    for affinity, vf, vb in zip(
                                        affinities,
                                        frame_visibility,
                                        box_visibility,
                                    )
                                    if vf
                                    > float(
                                        self.config[
                                            "frame_visibility_threshold"
                                        ]
                                    )
                                    and vb
                                    > float(
                                        self.config[
                                            "box_visibility_threshold"
                                        ]
                                    )
                                )
                            ),
                            mean_frame_visibility=self._mean(frame_visibility),
                            mean_box_visibility=self._mean(box_visibility),
                            mean_box_depth_consistency=self._mean(
                                box_depth_consistency
                            ),
                            mean_affinity=self._mean(affinities),
                            projected_points=candidate_points,
                            complete=bool(complete),
                            views=tuple(view_rows),
                        )
                    )

                eligible = [
                    candidate
                    for candidate in candidate_rows
                    if candidate.complete
                    and candidate.supporting_views
                    >= int(self.config["min_history_views"])
                    and candidate.support_score > 0.0
                ]
                total_score = sum(
                    candidate.support_score for candidate in candidate_rows
                )
                best = (
                    sorted(
                        candidate_rows,
                        key=lambda candidate: (
                            -candidate.support_score,
                            -candidate.supporting_views,
                            candidate.qim_rank,
                            candidate.track_id,
                        ),
                    )[0]
                    if candidate_rows
                    else None
                )
                dominance = (
                    best.support_score / total_score
                    if best is not None and total_score > 0.0
                    else None
                )
                would_veto = bool(
                    row_complete
                    and len(eligible) == 1
                    and best is eligible[0]
                    and dominance is not None
                    and dominance
                    > float(self.config["candidate_dominance_threshold"])
                )
                if not candidate_rows:
                    reason = "no_qim_candidates_defer_to_native"
                elif not row_complete:
                    reason = "incomplete_projection_defer_to_native"
                elif not eligible:
                    reason = "insufficient_two_view_support_defer_to_native"
                elif len(eligible) != 1:
                    reason = "nonunique_candidate_defer_to_native"
                elif dominance is None or dominance <= float(
                    self.config["candidate_dominance_threshold"]
                ):
                    reason = "low_dominance_defer_to_native"
                else:
                    reason = "two_view_unique_candidate_shadow_veto"
                veto_by_id[proposal.proposal_id] = BirthVetoObservation(
                    proposal_id=proposal.proposal_id,
                    candidates=tuple(candidate_rows),
                    would_veto_birth=would_veto,
                    recommended_track_id=(best.track_id if would_veto else None),
                    candidate_dominance=(
                        float(dominance) if dominance is not None else None
                    ),
                    action="defer_to_native",
                    reason=reason,
                )

        quality_rows = tuple(quality_by_id[value] for value in proposal_ids)
        veto_rows = tuple(veto_by_id[value] for value in proposal_ids)
        elapsed_ms = (perf_counter_ns() - start) / 1e6
        quality_exhausted = quality_remaining <= 0
        veto_exhausted = veto_remaining <= 0
        batch = MV3DISDepthLiteBatch(
            scene_id=scene_id,
            frame_id=frame_id,
            history_max_frame_id=self._last_commit_frame_id,
            proposal_ids=proposal_ids,
            guide_quality_rows=quality_rows,
            birth_veto_rows=veto_rows,
            guide_quality_projection_points_used=quality_used,
            birth_veto_projection_points_used=veto_used,
            guide_quality_budget_exhausted=bool(quality_exhausted),
            birth_veto_budget_exhausted=bool(veto_exhausted),
            proposal_cap_exceeded=bool(cap_exceeded),
            current_frame_valid=frame is not None,
            invalid_frame_reason=invalid_frame_reason,
            query_ms=float(elapsed_ms),
        )
        pending_proposals = () if cap_exceeded or frame is None else proposals
        self._pending = _PendingPayload(
            batch=batch,
            frame=None if cap_exceeded else frame,
            proposals=pending_proposals,
        )
        self._last_query_frame_id = frame_id
        self._stats["queries"] += 1
        self._stats["proposals"] += len(proposal_ids)
        self._stats["proposal_cap_batches"] += int(cap_exceeded)
        self._stats["invalid_frame_batches"] += int(frame is None)
        if invalid_frame_reason is not None:
            self._invalid_frame_reasons[invalid_frame_reason] = (
                self._invalid_frame_reasons.get(invalid_frame_reason, 0) + 1
            )
        self._stats["guide_quality_rows_valid"] += sum(
            row.valid for row in quality_rows
        )
        self._stats["guide_quality_rows_invalid"] += sum(
            not row.valid for row in quality_rows
        )
        self._stats["veto_recommendations"] += sum(
            row.would_veto_birth for row in veto_rows
        )
        self._stats["guide_quality_projection_points"] += quality_used
        self._stats["birth_veto_projection_points"] += veto_used
        self._stats["projection_points"] += quality_used + veto_used
        self._stats["guide_quality_budget_exhaustions"] += int(
            quality_exhausted
        )
        self._stats["birth_veto_budget_exhaustions"] += int(veto_exhausted)
        self._stats["query_ms_total"] += elapsed_ms
        self._stats["query_ms_max"] = max(
            float(self._stats["query_ms_max"]), elapsed_ms
        )
        self._query_samples.append(float(elapsed_ms))
        return batch

    @staticmethod
    def _committed_ids(
        values: Sequence[Optional[int]], count: int
    ) -> Tuple[Optional[int], ...]:
        if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
            raise ValueError("committed_track_ids must be a sequence")
        if len(values) != count:
            raise ValueError("committed_track_ids must align with proposals")
        result = []
        for value in values:
            if value is None:
                result.append(None)
                continue
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
                raise ValueError("committed_track_ids must contain integers or None")
            if int(value) < 0:
                raise ValueError("committed_track_ids must be non-negative")
            result.append(int(value))
        return tuple(result)

    @staticmethod
    def _native_targets(
        values: Sequence[Optional[Sequence[int]]], count: int
    ) -> Tuple[Optional[Tuple[int, ...]], ...]:
        if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
            raise ValueError("native_target_track_ids must be a sequence")
        if len(values) != count:
            raise ValueError("native_target_track_ids must align with proposals")
        result = []
        for targets in values:
            if targets is None:
                result.append(None)
                continue
            normalized = set()
            for value in targets:
                if isinstance(value, (bool, np.bool_)) or not isinstance(
                    value, Integral
                ):
                    raise ValueError("native target ids must contain integers")
                if int(value) < 0:
                    raise ValueError("native target ids must be non-negative")
                normalized.add(int(value))
            result.append(tuple(sorted(normalized)))
        return tuple(result)

    def _record_native_diagnostics(
        self,
        batch: MV3DISDepthLiteBatch,
        targets: Optional[Tuple[Optional[Tuple[int, ...]], ...]],
    ) -> None:
        if targets is None:
            self._stats["native_diagnostics_skipped"] += len(batch.proposal_ids)
            return
        for row, target_ids in zip(batch.birth_veto_rows, targets):
            if len(self._diagnostic_examples) < int(
                self.config["max_diagnostic_examples"]
            ):
                candidate_evidence = tuple(
                    (
                        candidate.track_id,
                        candidate.qim_rank,
                        candidate.history_views_available,
                        candidate.history_views_evaluated,
                        candidate.supporting_views,
                        candidate.support_score,
                        candidate.complete,
                        tuple(
                            (
                                view.frame_id,
                                view.valid,
                                view.frame_visibility,
                                view.box_visibility,
                                view.box_depth_consistency,
                                view.affinity,
                                view.supporting,
                                view.projected_points,
                                view.reason,
                            )
                            for view in candidate.views
                        ),
                    )
                    for candidate in row.candidates
                )
                self._diagnostic_examples.append(
                    (
                        batch.scene_id,
                        batch.frame_id,
                        row.proposal_id,
                        target_ids,
                        row.would_veto_birth,
                        row.recommended_track_id,
                        row.candidate_dominance,
                        row.reason,
                        candidate_evidence,
                    )
                )
            if target_ids is None:
                self._stats["native_unresolved"] += 1
                continue
            if target_ids:
                self._stats["native_history"] += 1
            else:
                self._stats["native_birth"] += 1
            if not row.would_veto_birth:
                continue
            self._stats["veto_evaluable"] += 1
            correct = row.recommended_track_id in target_ids
            if correct:
                self._stats["veto_correct"] += 1
            else:
                self._stats["veto_wrong"] += 1
                self._stats["veto_on_native_birth"] += int(not target_ids)

    def _remove_evicted_frame(self, frame_id: int) -> None:
        removed = 0
        empty = []
        for track_id, guides in self._track_guides.items():
            retained = tuple(guide for guide in guides if guide.frame_id != frame_id)
            removed += len(guides) - len(retained)
            if retained:
                self._track_guides[track_id] = retained
            else:
                empty.append(track_id)
        for track_id in empty:
            del self._track_guides[track_id]
        self._stats["guides_evicted_frame_cap"] += removed

    def commit(
        self,
        batch: MV3DISDepthLiteBatch,
        *,
        committed_track_ids: Sequence[Optional[int]],
        native_target_track_ids: Optional[
            Sequence[Optional[Sequence[int]]]
        ] = None,
    ) -> None:
        """Commit current guides after native association and close the query."""

        start = perf_counter_ns()
        if not self.enabled:
            raise RuntimeError("mv3dis_depth_lite observer is disabled")
        if self._last_committed_batch is batch:
            raise ValueError("depth batch was already committed")
        if self._pending is None or batch is not self._pending.batch:
            raise ValueError("commit requires the pending depth batch")
        if batch.scene_id != self._scene_id:
            raise ValueError("depth batch belongs to a different scene")
        if batch.frame_id != self._last_query_frame_id:
            raise ValueError("depth commit must close the latest query")
        committed = self._committed_ids(
            committed_track_ids, len(batch.proposal_ids)
        )
        native = (
            self._native_targets(native_target_track_ids, len(batch.proposal_ids))
            if native_target_track_ids is not None
            else None
        )

        self._record_native_diagnostics(batch, native)
        pending = self._pending
        if pending.frame is not None:
            self._committed_frames[batch.frame_id] = None
            quality_rows = {
                row.proposal_id: row for row in batch.guide_quality_rows
            }
            chosen: Dict[int, _GuideRecord] = {}
            for proposal, track_id in zip(pending.proposals, committed):
                quality_row = quality_rows[proposal.proposal_id]
                if (
                    track_id is None
                    or len(proposal.points_world) == 0
                    or not quality_row.valid
                ):
                    continue
                quality = quality_row.guide_quality
                guide = _GuideRecord(
                    track_id=track_id,
                    proposal_id=proposal.proposal_id,
                    frame_id=batch.frame_id,
                    points_world=proposal.points_world,
                    proposal_box_xyxy=proposal.proposal_box_xyxy,
                    source_quality=quality,
                )
                previous = chosen.get(track_id)
                key = (
                    -(quality if quality is not None else -1.0),
                    proposal.proposal_id,
                )
                previous_key = (
                    -(
                        previous.source_quality
                        if previous is not None
                        and previous.source_quality is not None
                        else -1.0
                    ),
                    previous.proposal_id if previous is not None else 1 << 62,
                )
                if previous is None or key < previous_key:
                    chosen[track_id] = guide

            limit = int(self.config["max_guides_per_track"])
            for track_id, guide in sorted(chosen.items()):
                prior = tuple(
                    item
                    for item in self._track_guides.get(track_id, ())
                    if item.frame_id != batch.frame_id
                )
                self._stats["guides_replaced_same_frame"] += len(
                    self._track_guides.get(track_id, ())
                ) - len(prior)
                combined = tuple(sorted(prior + (guide,), key=lambda item: item.frame_id))
                overflow = max(0, len(combined) - limit)
                self._stats["guides_evicted_track_cap"] += overflow
                self._track_guides[track_id] = combined[-limit:]
                self._stats["guides_committed"] += 1

            frame_limit = int(self.config["max_depth_frames"])
            while len(self._committed_frames) > frame_limit:
                evicted_frame_id, _ = self._committed_frames.popitem(last=False)
                self._remove_evicted_frame(evicted_frame_id)
                self._stats["committed_frames_evicted"] += 1

        elapsed_ms = (perf_counter_ns() - start) / 1e6
        self._last_commit_frame_id = batch.frame_id
        self._last_committed_batch = batch
        self._pending = None
        self._stats["commits"] += 1
        self._stats["commit_ms_total"] += elapsed_ms
        self._stats["commit_ms_max"] = max(
            float(self._stats["commit_ms_max"]), elapsed_ms
        )
        self._commit_samples.append(float(elapsed_ms))
        total_guides = sum(len(guides) for guides in self._track_guides.values())
        self._stats["max_committed_frames_observed"] = max(
            int(self._stats["max_committed_frames_observed"]),
            len(self._committed_frames),
        )
        self._stats["max_tracks_observed"] = max(
            int(self._stats["max_tracks_observed"]), len(self._track_guides)
        )
        self._stats["max_guides_observed"] = max(
            int(self._stats["max_guides_observed"]), total_guides
        )

    def record_pipeline_timing(
        self,
        *,
        query_ms: Optional[float] = None,
        commit_ms: Optional[float] = None,
    ) -> None:
        if query_ms is None and commit_ms is None:
            raise ValueError("at least one pipeline timing value is required")
        for stage, value in (("query", query_ms), ("commit", commit_ms)):
            if value is None:
                continue
            timing = _finite_float(f"pipeline_{stage}_ms", value, 0.0)
            self._stats[f"pipeline_{stage}_calls"] += 1
            self._stats[f"pipeline_{stage}_ms_total"] += timing
            self._stats[f"pipeline_{stage}_ms_max"] = max(
                float(self._stats[f"pipeline_{stage}_ms_max"]), timing
            )

    def snapshot(self) -> MV3DISDepthLiteSnapshot:
        counts = tuple(
            (track_id, len(guides))
            for track_id, guides in sorted(self._track_guides.items())
        )
        return MV3DISDepthLiteSnapshot(
            scene_id=self._scene_id,
            history_max_frame_id=self._last_commit_frame_id,
            committed_frame_ids=tuple(self._committed_frames),
            track_guide_counts=counts,
            total_guides=sum(count for _, count in counts),
            pending_frame_id=(
                self._pending.batch.frame_id if self._pending is not None else None
            ),
        )

    @staticmethod
    def _rate(numerator: int, denominator: int) -> Optional[float]:
        return numerator / denominator if denominator else None

    @staticmethod
    def _percentile(values: Sequence[float], q: float) -> float:
        if not values:
            return 0.0
        return float(np.percentile(np.asarray(values, dtype=np.float64), q))

    def summary(self) -> Dict[str, object]:
        result = dict(self._stats)
        queries = int(result["queries"])
        commits = int(result["commits"])
        pipeline_queries = int(result["pipeline_query_calls"])
        pipeline_commits = int(result["pipeline_commit_calls"])
        result.update(
            {
                "schema": "boxfusion.mv3dis_depth_lite_s0_shadow.v1",
                "enabled": self.enabled,
                "observer_only": self.observer_only,
                "active_authorized": False,
                "training_free": True,
                "unsupervised": True,
                "causal": True,
                "bounded_history": True,
                "online_parameter_update": False,
                "ground_truth_access": False,
                "semantic_access": False,
                "semantic_mutation": False,
                "detector_score_access": False,
                "puf_access": False,
                "native_outputs_mutated": False,
                "guide_quality_computed": True,
                "fusion_weights_computed": False,
                "fusion_weights_applied": False,
                "birth_veto_applied": False,
                "hardcoded_scene_event_access": False,
                "scene_id": self._scene_id,
                "geometry_adapter_available": self._projection_adapter is not None,
                "effective_config": dict(self.config),
                "invalid_frame_reasons": tuple(
                    sorted(self._invalid_frame_reasons.items())
                ),
                "committed_frames_retained": len(self._committed_frames),
                "tracks_retained": len(self._track_guides),
                "guides_retained": sum(
                    len(guides) for guides in self._track_guides.values()
                ),
                "veto_precision": self._rate(
                    int(result["veto_correct"]), int(result["veto_evaluable"])
                ),
                "query_ms_mean": (
                    float(result["query_ms_total"]) / queries if queries else 0.0
                ),
                "query_ms_p95": self._percentile(self._query_samples, 95),
                "commit_ms_mean": (
                    float(result["commit_ms_total"]) / commits if commits else 0.0
                ),
                "commit_ms_p95": self._percentile(self._commit_samples, 95),
                "pipeline_query_ms_mean": (
                    float(result["pipeline_query_ms_total"]) / pipeline_queries
                    if pipeline_queries
                    else 0.0
                ),
                "pipeline_commit_ms_mean": (
                    float(result["pipeline_commit_ms_total"]) / pipeline_commits
                    if pipeline_commits
                    else 0.0
                ),
                "diagnostic_examples": tuple(self._diagnostic_examples),
            }
        )
        return result

    def summary_line(self) -> str:
        summary = self.summary()
        precision = summary["veto_precision"]
        precision_text = "nan" if precision is None else f"{float(precision):.4f}"
        return (
            "MV3DIS-Depth-lite S0 shadow summary | "
            f"queries/proposals/commits={summary['queries']}/"
            f"{summary['proposals']}/{summary['commits']}, "
            f"guide_quality_valid/invalid="
            f"{summary['guide_quality_rows_valid']}/"
            f"{summary['guide_quality_rows_invalid']}, "
            f"veto/evaluable/correct/wrong={summary['veto_recommendations']}/"
            f"{summary['veto_evaluable']}/{summary['veto_correct']}/"
            f"{summary['veto_wrong']}, veto_P={precision_text}, "
            f"query_mean/p95/max_ms={summary['query_ms_mean']:.3f}/"
            f"{summary['query_ms_p95']:.3f}/{summary['query_ms_max']:.3f}, "
            f"frames/tracks/guides={summary['committed_frames_retained']}/"
            f"{summary['tracks_retained']}/{summary['guides_retained']}"
        )


def build_mv3dis_depth_lite(
    config: Mapping[str, object],
    *,
    projection_adapter: Optional[DepthGuideGeometryAdapter] = None,
) -> MV3DISDepthLiteObserver:
    if not isinstance(config, Mapping):
        raise ValueError("application config must be a mapping")
    return MV3DISDepthLiteObserver(
        config.get("mv3dis_depth_lite", {}),
        projection_adapter=projection_adapter,
    )


__all__ = [
    "DEFAULT_MV3DIS_DEPTH_LITE_CONFIG",
    "BirthVetoObservation",
    "DepthGuideGeometryAdapter",
    "DepthGuideProjectionMetrics",
    "GuideQualityObservation",
    "HistoricalCandidateEvidence",
    "HistoricalViewEvidence",
    "MV3DISDepthLiteBatch",
    "MV3DISDepthLiteObserver",
    "MV3DISDepthLiteSnapshot",
    "build_mv3dis_depth_lite",
    "derive_committed_track_ids",
    "resolve_mv3dis_depth_lite_config",
]
