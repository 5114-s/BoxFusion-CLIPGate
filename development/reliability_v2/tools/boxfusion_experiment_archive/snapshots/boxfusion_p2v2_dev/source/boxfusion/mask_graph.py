"""Bounded incremental mask graph for supplemental-instance confirmation.

The graph in this module is deliberately observation-only by default.  It is
not imported by the released BoxFusion path and ``enabled`` defaults to
``False``.  This lets an online pipeline collect and inspect cross-view
identity evidence before allowing any supplemental track to enter the final
detection output.

The implementation is NumPy-only and accepts either the explicit data classes
defined here or duck-typed BoxFusion objects:

* a track may expose ``aabb``/``points`` directly or through ``memory``;
* a lifted proposal may expose ``box``, ``observation``, ``proposal`` and
  ``view`` in the same shape as :class:`online_refinement.LiftedProposal`.

Every cross-view edge combines metric 3D IoU, bidirectional point/volume
containment, projection of the *current track AABB* into the current mask,
optional appearance cosine, and optional label compatibility.  Geometry is a
hard prerequisite; appearance and labels are only bounded soft evidence.
"""

from __future__ import annotations

from collections import Counter, OrderedDict
from dataclasses import dataclass, replace
from numbers import Integral, Real
from typing import (
    Any,
    Callable,
    Dict,
    Hashable,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import numpy as np


NodeId = Union[int, str]
FrameId = Union[int, str]
LabelCompatibility = Union[
    float,
    Callable[[Optional[str], Optional[str]], float],
]


DEFAULT_MASK_GRAPH_CONFIG: Dict[str, Any] = {
    # Safe integration default: no state mutation and no changed output.
    "enabled": False,
    # Per-track memory bounds.
    "max_nodes": 32,
    "max_edges": 128,
    # Confirmation is based on distinct frames, never repeated proposals from
    # one image.
    "min_unique_frames": 2,
    # Geometry gate and score.
    "minimum_edge_score": 0.38,
    "minimum_iou_3d": 0.02,
    "minimum_mutual_inside": 0.10,
    "minimum_projection_iou": 0.05,
    "minimum_geometry_matches": 2,
    "require_projection": True,
    "iou_3d_weight": 0.35,
    "mutual_inside_weight": 0.30,
    "projection_iou_weight": 0.35,
    # Optional appearance is a soft penalty/bonus around the neutral cosine.
    "appearance_weight": 0.20,
    "appearance_neutral_cosine": 0.20,
    # Optional labels are also soft.  Exact or group-compatible labels score
    # one; an explicit mismatch remains non-zero and cannot veto geometry.
    "label_weight": 0.10,
    "label_unknown_score": 0.50,
    "label_mismatch_score": 0.25,
    "label_compatibility_groups": (),
    # Projection/rasterization controls.
    "mask_threshold": 0.50,
    "near_clip": 1e-3,
}


def _finite_float(name: str, value: object) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real number")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _strict_int(name: str, value: object, minimum: int) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer")
    result = int(value)
    if result < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return result


def _bounded_float(
    name: str,
    value: object,
    lower: float = 0.0,
    upper: float = 1.0,
) -> float:
    result = _finite_float(name, value)
    if result < lower or result > upper:
        raise ValueError(f"{name} must lie in [{lower}, {upper}]")
    return result


def _normalize_label(label: str) -> str:
    return " ".join(label.casefold().replace("_", " ").replace("-", " ").split())


def _resolve_label_groups(value: object) -> Tuple[Tuple[str, ...], ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(
            "mask_graph.label_compatibility_groups must be a sequence"
        )
    resolved: List[Tuple[str, ...]] = []
    occupied: Dict[str, int] = {}
    for group_index, group in enumerate(value):
        if isinstance(group, (str, bytes)) or not isinstance(group, Sequence):
            raise ValueError(
                "each mask_graph label compatibility group must be a sequence"
            )
        labels: List[str] = []
        for label_index, label in enumerate(group):
            if not isinstance(label, str) or not label.strip():
                raise ValueError(
                    "mask_graph label compatibility entries must be "
                    "non-empty strings"
                )
            normalized = _normalize_label(label)
            if normalized in labels:
                raise ValueError(
                    "mask_graph label compatibility groups cannot contain "
                    "duplicate labels"
                )
            if normalized in occupied:
                raise ValueError(
                    "a mask_graph label cannot occur in more than one "
                    "compatibility group"
                )
            occupied[normalized] = group_index
            labels.append(normalized)
        if len(labels) < 2:
            raise ValueError(
                "each mask_graph label compatibility group must contain "
                "at least two labels"
            )
        resolved.append(tuple(labels))
    return tuple(resolved)


def resolve_mask_graph_config(
    config: Optional[Mapping[str, object]] = None,
) -> Dict[str, Any]:
    """Return a strictly validated mask-graph configuration.

    ``config`` must be the ``mask_graph`` subsection rather than a complete
    application config.  Unknown keys are rejected so misspelled experimental
    knobs cannot silently fall back to defaults.
    """

    if config is None:
        config = {}
    if not isinstance(config, Mapping):
        raise ValueError("mask_graph config must be a mapping")

    unknown = sorted(set(config) - set(DEFAULT_MASK_GRAPH_CONFIG))
    if unknown:
        raise ValueError(
            "Unknown mask_graph config key(s): " + ", ".join(unknown)
        )

    resolved = dict(DEFAULT_MASK_GRAPH_CONFIG)
    resolved.update(config)

    for key in ("enabled", "require_projection"):
        if not isinstance(resolved[key], (bool, np.bool_)):
            raise ValueError(f"mask_graph.{key} must be a boolean")
        resolved[key] = bool(resolved[key])

    for key, minimum in (
        ("max_nodes", 1),
        ("max_edges", 1),
        ("min_unique_frames", 2),
        ("minimum_geometry_matches", 1),
    ):
        resolved[key] = _strict_int(
            f"mask_graph.{key}", resolved[key], minimum
        )

    if resolved["max_nodes"] < resolved["min_unique_frames"]:
        raise ValueError(
            "mask_graph.max_nodes must be at least min_unique_frames"
        )
    if resolved["minimum_geometry_matches"] > 3:
        raise ValueError(
            "mask_graph.minimum_geometry_matches cannot exceed 3"
        )

    for key in (
        "minimum_edge_score",
        "minimum_iou_3d",
        "minimum_mutual_inside",
        "minimum_projection_iou",
        "label_unknown_score",
        "label_mismatch_score",
        "mask_threshold",
    ):
        resolved[key] = _bounded_float(
            f"mask_graph.{key}", resolved[key]
        )

    for key in (
        "iou_3d_weight",
        "mutual_inside_weight",
        "projection_iou_weight",
        "appearance_weight",
        "label_weight",
    ):
        resolved[key] = _finite_float(
            f"mask_graph.{key}", resolved[key]
        )
        if resolved[key] < 0.0:
            raise ValueError(f"mask_graph.{key} must be non-negative")

    geometry_weight = sum(
        float(resolved[key])
        for key in (
            "iou_3d_weight",
            "mutual_inside_weight",
            "projection_iou_weight",
        )
    )
    if geometry_weight <= 0.0:
        raise ValueError(
            "mask_graph geometry weights must have a positive sum"
        )

    resolved["appearance_neutral_cosine"] = _bounded_float(
        "mask_graph.appearance_neutral_cosine",
        resolved["appearance_neutral_cosine"],
        -1.0,
        1.0,
    )
    resolved["near_clip"] = _finite_float(
        "mask_graph.near_clip", resolved["near_clip"]
    )
    if resolved["near_clip"] <= 0.0:
        raise ValueError("mask_graph.near_clip must be positive")

    resolved["label_compatibility_groups"] = _resolve_label_groups(
        resolved["label_compatibility_groups"]
    )
    return resolved


def _as_numpy(value: object, name: str) -> np.ndarray:
    candidate = value
    if hasattr(candidate, "detach"):
        candidate = candidate.detach()
    if hasattr(candidate, "cpu"):
        candidate = candidate.cpu()
    if hasattr(candidate, "numpy"):
        candidate = candidate.numpy()
    try:
        return np.asarray(candidate)
    except Exception as error:
        raise ValueError(f"{name} cannot be converted to a NumPy array") from error


def _readonly_array(
    value: object,
    name: str,
    shape: Optional[Tuple[int, ...]] = None,
    *,
    dtype: Any = np.float32,
) -> np.ndarray:
    array = _as_numpy(value, name)
    if not np.issubdtype(array.dtype, np.number):
        raise ValueError(f"{name} must be numeric")
    array = np.asarray(array, dtype=dtype)
    if shape is not None and array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    array = np.array(array, dtype=dtype, copy=True)
    array.setflags(write=False)
    return array


def _identifier(name: str, value: object) -> Union[int, str]:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be an integer or non-empty string")
    if isinstance(value, Integral):
        result = int(value)
        if result < 0:
            raise ValueError(f"{name} integer must be non-negative")
        return result
    if isinstance(value, str) and value.strip():
        return value.strip()
    raise ValueError(f"{name} must be an integer or non-empty string")


def _optional_points(value: object, name: str) -> Optional[np.ndarray]:
    if value is None:
        return None
    points = _as_numpy(value, name)
    if (
        points.ndim != 2
        or points.shape[1] != 3
        or not np.issubdtype(points.dtype, np.number)
        or not np.isfinite(points).all()
    ):
        raise ValueError(f"{name} must have finite shape [N, 3]")
    result = np.asarray(points, dtype=np.float32).copy()
    result.setflags(write=False)
    return result


def _optional_feature(value: object, name: str) -> Optional[np.ndarray]:
    if value is None:
        return None
    feature = _as_numpy(value, name)
    if (
        feature.ndim != 1
        or feature.size < 1
        or not np.issubdtype(feature.dtype, np.number)
        or not np.isfinite(feature).all()
    ):
        raise ValueError(f"{name} must be a non-empty finite vector")
    feature = np.asarray(feature, dtype=np.float32)
    norm = float(np.linalg.norm(feature))
    if norm <= 1e-8:
        raise ValueError(f"{name} must have non-zero norm")
    result = np.asarray(feature / norm, dtype=np.float32)
    result.setflags(write=False)
    return result


def _optional_label(value: object, name: str) -> Optional[str]:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string or None")
    return value.strip()


def _optional_mask(value: object, name: str) -> Optional[np.ndarray]:
    if value is None:
        return None
    mask = _as_numpy(value, name)
    if mask.ndim != 2 or min(mask.shape) < 1:
        raise ValueError(f"{name} must have shape [H, W]")
    if not (
        np.issubdtype(mask.dtype, np.bool_)
        or np.issubdtype(mask.dtype, np.number)
    ):
        raise ValueError(f"{name} must be boolean or numeric")
    if np.issubdtype(mask.dtype, np.number) and not np.isfinite(mask).all():
        raise ValueError(f"{name} must contain only finite values")
    result = np.array(mask, copy=True)
    result.setflags(write=False)
    return result


def _box_parts(box: object, name: str) -> Tuple[np.ndarray, np.ndarray]:
    if isinstance(box, (tuple, list)) and len(box) == 2:
        center_value, dims_value = box
    else:
        array = _as_numpy(box, name)
        if array.shape != (6,):
            raise ValueError(
                f"{name} must have shape [6] or be a (center, dims) pair"
            )
        center_value, dims_value = array[:3], array[3:6]
    center = _readonly_array(
        center_value, f"{name} center", (3,), dtype=np.float32
    )
    dims = _readonly_array(
        dims_value, f"{name} dims", (3,), dtype=np.float32
    )
    if np.any(dims <= 0.0):
        raise ValueError(f"{name} dims must be positive")
    return center, dims


@dataclass(frozen=True)
class MaskGraphNode:
    """One lifted mask observation retained by a per-track graph."""

    node_id: NodeId
    frame_id: FrameId
    center: np.ndarray
    dims: np.ndarray
    points_world: Optional[np.ndarray] = None
    mask: Optional[np.ndarray] = None
    intrinsics: Optional[np.ndarray] = None
    camera_to_world: Optional[np.ndarray] = None
    appearance_feature: Optional[np.ndarray] = None
    label: Optional[str] = None
    confidence: float = 1.0

    def __post_init__(self) -> None:
        node_id = _identifier("MaskGraphNode.node_id", self.node_id)
        frame_id = _identifier("MaskGraphNode.frame_id", self.frame_id)
        center, dims = _box_parts(
            (self.center, self.dims), "MaskGraphNode AABB"
        )
        confidence = _bounded_float(
            "MaskGraphNode.confidence", self.confidence
        )
        points = _optional_points(
            self.points_world, "MaskGraphNode.points_world"
        )
        feature = _optional_feature(
            self.appearance_feature,
            "MaskGraphNode.appearance_feature",
        )
        label = _optional_label(self.label, "MaskGraphNode.label")
        mask = _optional_mask(self.mask, "MaskGraphNode.mask")

        projection_fields = (
            mask,
            self.intrinsics,
            self.camera_to_world,
        )
        present_count = sum(value is not None for value in projection_fields)
        if present_count not in (0, 3):
            raise ValueError(
                "MaskGraphNode mask, intrinsics and camera_to_world must "
                "either all be present or all be absent"
            )
        intrinsics = None
        pose = None
        if present_count:
            intrinsics = _readonly_array(
                self.intrinsics,
                "MaskGraphNode.intrinsics",
                (3, 3),
            )
            pose = _readonly_array(
                self.camera_to_world,
                "MaskGraphNode.camera_to_world",
                (4, 4),
            )
            if abs(float(np.linalg.det(intrinsics))) <= 1e-12:
                raise ValueError("MaskGraphNode.intrinsics must be invertible")
            if abs(float(np.linalg.det(pose))) <= 1e-12:
                raise ValueError(
                    "MaskGraphNode.camera_to_world must be invertible"
                )

        object.__setattr__(self, "node_id", node_id)
        object.__setattr__(self, "frame_id", frame_id)
        object.__setattr__(self, "center", center)
        object.__setattr__(self, "dims", dims)
        object.__setattr__(self, "points_world", points)
        object.__setattr__(self, "mask", mask)
        object.__setattr__(self, "intrinsics", intrinsics)
        object.__setattr__(self, "camera_to_world", pose)
        object.__setattr__(self, "appearance_feature", feature)
        object.__setattr__(self, "label", label)
        object.__setattr__(self, "confidence", confidence)

    @property
    def box(self) -> np.ndarray:
        return np.concatenate((self.center, self.dims)).astype(np.float32)


@dataclass(frozen=True)
class MaskGraphProjectionContext:
    """Reusable projection data for one proposal observation.

    Constructing this context performs the proposal-wide work (mask
    thresholding, mask-area reduction, and pose inversion) exactly once.  The
    same immutable context can then be passed to :func:`evaluate_edge` for
    every candidate track compared with that proposal.
    """

    node_id: NodeId
    frame_id: FrameId
    binary_mask: np.ndarray
    mask_area: int
    world_to_camera: np.ndarray

    def __post_init__(self) -> None:
        node_id = _identifier(
            "MaskGraphProjectionContext.node_id", self.node_id
        )
        frame_id = _identifier(
            "MaskGraphProjectionContext.frame_id", self.frame_id
        )
        binary = _as_numpy(
            self.binary_mask, "MaskGraphProjectionContext.binary_mask"
        )
        if binary.ndim != 2 or min(binary.shape) < 1:
            raise ValueError(
                "MaskGraphProjectionContext.binary_mask must have shape "
                "[H, W]"
            )
        binary = np.asarray(binary, dtype=np.bool_).copy()
        binary.setflags(write=False)
        mask_area = _strict_int(
            "MaskGraphProjectionContext.mask_area", self.mask_area, 0
        )
        if mask_area > binary.size:
            raise ValueError(
                "MaskGraphProjectionContext.mask_area cannot exceed the "
                "binary mask size"
            )
        world_to_camera = _readonly_array(
            self.world_to_camera,
            "MaskGraphProjectionContext.world_to_camera",
            (4, 4),
            dtype=np.float64,
        )
        if abs(float(np.linalg.det(world_to_camera))) <= 1e-12:
            raise ValueError(
                "MaskGraphProjectionContext.world_to_camera must be "
                "invertible"
            )
        object.__setattr__(self, "node_id", node_id)
        object.__setattr__(self, "frame_id", frame_id)
        object.__setattr__(self, "binary_mask", binary)
        object.__setattr__(self, "mask_area", mask_area)
        object.__setattr__(self, "world_to_camera", world_to_camera)


@dataclass(frozen=True)
class MaskGraphTrackEvidence:
    """Immutable snapshot of the current aggregate track used for an edge."""

    center: np.ndarray
    dims: np.ndarray
    points_world: Optional[np.ndarray] = None
    appearance_feature: Optional[np.ndarray] = None
    label: Optional[str] = None

    def __post_init__(self) -> None:
        center, dims = _box_parts(
            (self.center, self.dims), "MaskGraphTrackEvidence AABB"
        )
        object.__setattr__(self, "center", center)
        object.__setattr__(self, "dims", dims)
        object.__setattr__(
            self,
            "points_world",
            _optional_points(
                self.points_world, "MaskGraphTrackEvidence.points_world"
            ),
        )
        object.__setattr__(
            self,
            "appearance_feature",
            _optional_feature(
                self.appearance_feature,
                "MaskGraphTrackEvidence.appearance_feature",
            ),
        )
        object.__setattr__(
            self,
            "label",
            _optional_label(self.label, "MaskGraphTrackEvidence.label"),
        )


@dataclass(frozen=True)
class MaskGraphEdge:
    """One evaluated temporal association, with mergeable running metrics."""

    source_id: NodeId
    target_id: NodeId
    source_frame_id: FrameId
    target_frame_id: FrameId
    accepted: bool
    reason: str
    score: float
    geometry_score: float
    iou_3d: float
    observation_inside_track: float
    track_inside_observation: float
    mutual_inside: float
    projection_iou: Optional[float]
    appearance_cosine: Optional[float]
    appearance_compatibility: Optional[float]
    label_compatibility: Optional[float]
    geometry_matches: int
    sample_count: int = 1
    unique_frame_ids: Tuple[FrameId, ...] = ()

    def __post_init__(self) -> None:
        source_id = _identifier("MaskGraphEdge.source_id", self.source_id)
        target_id = _identifier("MaskGraphEdge.target_id", self.target_id)
        source_frame = _identifier(
            "MaskGraphEdge.source_frame_id", self.source_frame_id
        )
        target_frame = _identifier(
            "MaskGraphEdge.target_frame_id", self.target_frame_id
        )
        if not isinstance(self.accepted, (bool, np.bool_)):
            raise ValueError("MaskGraphEdge.accepted must be a boolean")
        if not isinstance(self.reason, str) or not self.reason:
            raise ValueError("MaskGraphEdge.reason must be a non-empty string")
        for name in (
            "score",
            "geometry_score",
            "iou_3d",
            "observation_inside_track",
            "track_inside_observation",
            "mutual_inside",
        ):
            object.__setattr__(
                self,
                name,
                _bounded_float(f"MaskGraphEdge.{name}", getattr(self, name)),
            )
        for name in (
            "projection_iou",
            "appearance_compatibility",
            "label_compatibility",
        ):
            value = getattr(self, name)
            if value is not None:
                value = _bounded_float(f"MaskGraphEdge.{name}", value)
            object.__setattr__(self, name, value)
        cosine = self.appearance_cosine
        if cosine is not None:
            cosine = _bounded_float(
                "MaskGraphEdge.appearance_cosine", cosine, -1.0, 1.0
            )
        geometry_matches = _strict_int(
            "MaskGraphEdge.geometry_matches", self.geometry_matches, 0
        )
        sample_count = _strict_int(
            "MaskGraphEdge.sample_count", self.sample_count, 1
        )
        frame_ids = list(self.unique_frame_ids) or [
            source_frame,
            target_frame,
        ]
        unique_frames: List[FrameId] = []
        for value in frame_ids:
            normalized = _identifier(
                "MaskGraphEdge.unique_frame_ids entry", value
            )
            if normalized not in unique_frames:
                unique_frames.append(normalized)

        object.__setattr__(self, "source_id", source_id)
        object.__setattr__(self, "target_id", target_id)
        object.__setattr__(self, "source_frame_id", source_frame)
        object.__setattr__(self, "target_frame_id", target_frame)
        object.__setattr__(self, "accepted", bool(self.accepted))
        object.__setattr__(self, "appearance_cosine", cosine)
        object.__setattr__(self, "geometry_matches", geometry_matches)
        object.__setattr__(self, "sample_count", sample_count)
        object.__setattr__(self, "unique_frame_ids", tuple(unique_frames))

    def accumulate(self, other: "MaskGraphEdge") -> "MaskGraphEdge":
        """Return a sample-count-weighted accumulation of the same edge."""

        if not isinstance(other, MaskGraphEdge):
            raise ValueError("can only accumulate another MaskGraphEdge")
        if (
            self.source_id != other.source_id
            or self.target_id != other.target_id
        ):
            raise ValueError("MaskGraphEdge endpoints must match to accumulate")
        total = self.sample_count + other.sample_count

        def mean(name: str) -> float:
            return float(
                (
                    getattr(self, name) * self.sample_count
                    + getattr(other, name) * other.sample_count
                )
                / total
            )

        def optional_mean(name: str) -> Optional[float]:
            values = []
            weights = []
            for edge in (self, other):
                value = getattr(edge, name)
                if value is not None:
                    values.append(float(value))
                    weights.append(edge.sample_count)
            if not values:
                return None
            return float(np.average(values, weights=weights))

        frames = list(self.unique_frame_ids)
        for frame_id in other.unique_frame_ids:
            if frame_id not in frames:
                frames.append(frame_id)
        return MaskGraphEdge(
            source_id=self.source_id,
            target_id=self.target_id,
            source_frame_id=self.source_frame_id,
            target_frame_id=other.target_frame_id,
            accepted=self.accepted or other.accepted,
            reason=(
                "accepted"
                if self.accepted or other.accepted
                else other.reason
            ),
            score=mean("score"),
            geometry_score=mean("geometry_score"),
            iou_3d=mean("iou_3d"),
            observation_inside_track=mean("observation_inside_track"),
            track_inside_observation=mean("track_inside_observation"),
            mutual_inside=mean("mutual_inside"),
            projection_iou=optional_mean("projection_iou"),
            appearance_cosine=optional_mean("appearance_cosine"),
            appearance_compatibility=optional_mean(
                "appearance_compatibility"
            ),
            label_compatibility=optional_mean("label_compatibility"),
            geometry_matches=max(
                self.geometry_matches, other.geometry_matches
            ),
            sample_count=total,
            unique_frame_ids=tuple(frames),
        )


@dataclass(frozen=True)
class MaskGraphUpdate:
    """Outcome of one call to :func:`update_mask_graph`."""

    accepted: bool
    seeded: bool
    confirmed: bool
    became_confirmed: bool
    node_id: Optional[NodeId]
    edge: Optional[MaskGraphEdge]
    reason: str


class MaskGraphState:
    """A bounded, deterministic graph belonging to one candidate track."""

    def __init__(
        self,
        track_id: Optional[NodeId] = None,
        config: Optional[Mapping[str, object]] = None,
    ) -> None:
        self.config = resolve_mask_graph_config(config)
        self.track_id = (
            None
            if track_id is None
            else _identifier("MaskGraphState.track_id", track_id)
        )
        self.nodes: "OrderedDict[NodeId, MaskGraphNode]" = OrderedDict()
        self.edges: "OrderedDict[Tuple[NodeId, NodeId], MaskGraphEdge]" = (
            OrderedDict()
        )
        self._confirmed = False
        self.confirmation_frame_id: Optional[FrameId] = None
        self._next_node_index = 0

    @property
    def node_count(self) -> int:
        return len(self.nodes)

    @property
    def edge_count(self) -> int:
        return len(self.edges)

    @property
    def unique_frame_ids(self) -> Tuple[FrameId, ...]:
        frames: List[FrameId] = []
        for node in self.nodes.values():
            if node.frame_id not in frames:
                frames.append(node.frame_id)
        return tuple(frames)

    @property
    def unique_frame_count(self) -> int:
        return len(self.unique_frame_ids)

    @property
    def confirmed(self) -> bool:
        # Confirmation is deliberately latched.  Evicting old evidence to
        # enforce a memory bound must not demote a previously confirmed track.
        return self._confirmed

    def next_node_id(self, frame_id: FrameId) -> str:
        frame = _identifier("frame_id", frame_id)
        prefix = "track" if self.track_id is None else str(self.track_id)
        while True:
            value = f"{prefix}:{frame}:{self._next_node_index}"
            self._next_node_index += 1
            if value not in self.nodes:
                return value

    def add_node(self, node: MaskGraphNode) -> bool:
        """Add compact node metadata and evict oldest bounded evidence.

        Per-frame point clouds, masks, and camera matrices are only needed
        while evaluating the incoming proposal.  Retaining them in every
        confirmed track made memory grow with image resolution and point
        count, so graph persistence deliberately keeps only box, frame,
        appearance, label, and confidence metadata.
        """

        if not isinstance(node, MaskGraphNode):
            raise ValueError("MaskGraphState.add_node requires MaskGraphNode")
        if node.node_id in self.nodes:
            raise ValueError(f"duplicate mask-graph node_id: {node.node_id}")
        compact_node = replace(
            node,
            points_world=None,
            mask=None,
            intrinsics=None,
            camera_to_world=None,
        )
        self.nodes[node.node_id] = compact_node

        while len(self.nodes) > int(self.config["max_nodes"]):
            evicted_id, _ = self.nodes.popitem(last=False)
            for key in list(self.edges):
                if evicted_id in key:
                    del self.edges[key]

        became_confirmed = (
            not self._confirmed
            and self.unique_frame_count
            >= int(self.config["min_unique_frames"])
        )
        if became_confirmed:
            self._confirmed = True
            self.confirmation_frame_id = node.frame_id
        return became_confirmed

    def add_edge(self, edge: MaskGraphEdge) -> None:
        """Add or accumulate a directed temporal edge within memory bounds."""

        if not isinstance(edge, MaskGraphEdge):
            raise ValueError("MaskGraphState.add_edge requires MaskGraphEdge")
        if edge.source_id not in self.nodes or edge.target_id not in self.nodes:
            raise ValueError("mask-graph edge endpoints must exist as nodes")
        key = (edge.source_id, edge.target_id)
        if key in self.edges:
            self.edges[key] = self.edges[key].accumulate(edge)
        else:
            self.edges[key] = edge
        while len(self.edges) > int(self.config["max_edges"]):
            self.edges.popitem(last=False)

    def summary(self) -> Dict[str, Any]:
        return {
            "enabled": bool(self.config["enabled"]),
            "track_id": self.track_id,
            "nodes": self.node_count,
            "edges": self.edge_count,
            "unique_frames": self.unique_frame_count,
            "confirmed": self.confirmed,
            "confirmation_frame_id": self.confirmation_frame_id,
            "edge_samples": int(
                sum(edge.sample_count for edge in self.edges.values())
            ),
        }


def _get(value: object, name: str) -> Any:
    if isinstance(value, Mapping):
        return value.get(name)
    return getattr(value, name, None)


def _first_not_none(*values: object) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def _extract_box(value: object, name: str) -> Tuple[np.ndarray, np.ndarray]:
    center = _get(value, "center")
    dims = _get(value, "dims")
    if center is not None or dims is not None:
        if center is None or dims is None:
            raise ValueError(f"{name} must define both center and dims")
        return _box_parts((center, dims), name)
    box = _first_not_none(
        _get(value, "box"),
        _get(value, "aabb"),
        _get(value, "last_box"),
    )
    if box is None:
        memory = _get(value, "memory")
        box = _get(memory, "aabb") if memory is not None else None
    if box is None:
        raise ValueError(f"{name} does not expose an AABB")
    return _box_parts(box, name)


def _feature_from_stats(stats: object) -> Optional[np.ndarray]:
    if stats is None:
        return None
    feature = _first_not_none(
        _get(stats, "appearance_feature"),
        _get(stats, "feature"),
    )
    if feature is not None:
        return feature
    feature_sum = _get(stats, "feature_sum")
    feature_count = _get(stats, "feature_count")
    if feature_sum is None or feature_count is None:
        return None
    if isinstance(feature_count, (bool, np.bool_)) or not isinstance(
        feature_count, Integral
    ):
        raise ValueError("track stats.feature_count must be an integer")
    if int(feature_count) <= 0:
        return None
    return np.asarray(feature_sum, dtype=np.float32) / float(feature_count)


def _graph_feature(graph: MaskGraphState) -> Optional[np.ndarray]:
    features = [
        node.appearance_feature
        for node in graph.nodes.values()
        if node.appearance_feature is not None
    ]
    if not features:
        return None
    dimensions = {feature.shape for feature in features}
    if len(dimensions) != 1:
        return None
    mean = np.mean(np.stack(features), axis=0)
    if float(np.linalg.norm(mean)) <= 1e-8:
        return None
    return mean


def _graph_label(graph: MaskGraphState) -> Optional[str]:
    votes: Counter = Counter(
        node.label for node in graph.nodes.values() if node.label is not None
    )
    if not votes:
        return None
    return sorted(votes.items(), key=lambda item: (-item[1], item[0]))[0][0]


def coerce_track_evidence(
    track: object,
    graph: Optional[MaskGraphState] = None,
) -> MaskGraphTrackEvidence:
    """Adapt a CandidateTrack/GlobalEvidence/mapping to an immutable snapshot."""

    if isinstance(track, MaskGraphTrackEvidence):
        return track
    center, dims = _extract_box(track, "track")
    memory = _get(track, "memory")
    points = _first_not_none(
        _get(track, "points_world"),
        _get(track, "points"),
        _get(memory, "points") if memory is not None else None,
    )
    stats = _get(track, "stats")
    feature = _first_not_none(
        _get(track, "appearance_feature"),
        _get(track, "feature"),
        _feature_from_stats(stats),
        _graph_feature(graph) if graph is not None else None,
    )
    label = _first_not_none(
        _get(track, "label"),
        _get(stats, "label") if stats is not None else None,
        _graph_label(graph) if graph is not None else None,
    )
    return MaskGraphTrackEvidence(
        center=center,
        dims=dims,
        points_world=points,
        appearance_feature=feature,
        label=label,
    )


def coerce_mask_graph_node(
    observation: object,
    *,
    node_id: Optional[NodeId] = None,
    frame_id: Optional[FrameId] = None,
) -> MaskGraphNode:
    """Adapt an explicit node or lifted-like proposal to ``MaskGraphNode``."""

    if isinstance(observation, MaskGraphNode):
        if node_id is not None and node_id != observation.node_id:
            raise ValueError("node_id override disagrees with MaskGraphNode")
        if frame_id is not None and frame_id != observation.frame_id:
            raise ValueError("frame_id override disagrees with MaskGraphNode")
        return observation

    center, dims = _extract_box(observation, "observation")
    proposal = _get(observation, "proposal")
    view = _get(observation, "view")
    depth_observation = _get(observation, "observation")

    resolved_frame_id = _first_not_none(
        frame_id,
        _get(observation, "frame_id"),
        _get(observation, "frame_index"),
        _get(view, "frame_id") if view is not None else None,
        _get(view, "frame_index") if view is not None else None,
    )
    if resolved_frame_id is None:
        raise ValueError("observation does not expose a frame_id")
    resolved_node_id = _first_not_none(
        node_id,
        _get(observation, "node_id"),
    )
    if resolved_node_id is None:
        raise ValueError(
            "observation does not expose a node_id; pass node_id explicitly"
        )

    points = _first_not_none(
        _get(observation, "points_world"),
        _get(depth_observation, "points_world")
        if depth_observation is not None
        else None,
    )
    mask = _first_not_none(
        _get(observation, "mask"),
        _get(proposal, "mask") if proposal is not None else None,
    )
    intrinsics = _first_not_none(
        _get(observation, "intrinsics"),
        _get(view, "intrinsics") if view is not None else None,
    )
    pose = _first_not_none(
        _get(observation, "camera_to_world"),
        _get(view, "camera_to_world") if view is not None else None,
    )
    feature = _first_not_none(
        _get(observation, "appearance_feature"),
        _get(observation, "feature"),
        _get(proposal, "feature") if proposal is not None else None,
    )
    label = _first_not_none(
        _get(observation, "label"),
        _get(proposal, "label") if proposal is not None else None,
    )
    confidence = _first_not_none(
        _get(observation, "confidence"),
        _get(observation, "score"),
        _get(proposal, "score") if proposal is not None else None,
        1.0,
    )
    return MaskGraphNode(
        node_id=resolved_node_id,
        frame_id=resolved_frame_id,
        center=center,
        dims=dims,
        points_world=points,
        mask=mask,
        intrinsics=intrinsics,
        camera_to_world=pose,
        appearance_feature=feature,
        label=label,
        confidence=confidence,
    )


def build_projection_context(
    observation: object,
    config: Optional[Mapping[str, object]] = None,
    *,
    node_id: Optional[NodeId] = None,
    frame_id: Optional[FrameId] = None,
) -> Optional[MaskGraphProjectionContext]:
    """Build reusable proposal-wide projection data.

    ``observation`` may be a :class:`MaskGraphNode` or any lifted-like object
    accepted by :func:`coerce_mask_graph_node`.  ``None`` is returned when the
    observation has no complete projection tuple.  Callers comparing one
    proposal with many tracks should coerce the node and call this function
    once, then pass the returned context to each :func:`evaluate_edge` call.
    """

    node = coerce_mask_graph_node(
        observation, node_id=node_id, frame_id=frame_id
    )
    if (
        node.mask is None
        or node.intrinsics is None
        or node.camera_to_world is None
    ):
        return None
    resolved = resolve_mask_graph_config(config)
    binary = np.asarray(
        node.mask >= float(resolved["mask_threshold"]),
        dtype=np.bool_,
    )
    return MaskGraphProjectionContext(
        node_id=node.node_id,
        frame_id=node.frame_id,
        binary_mask=binary,
        mask_area=int(np.count_nonzero(binary)),
        world_to_camera=np.linalg.inv(
            node.camera_to_world.astype(np.float64)
        ),
    )


def _aabb_intersection(
    center_a: np.ndarray,
    dims_a: np.ndarray,
    center_b: np.ndarray,
    dims_b: np.ndarray,
) -> Tuple[float, float, float]:
    minimum_a = center_a.astype(np.float64) - dims_a * 0.5
    maximum_a = center_a.astype(np.float64) + dims_a * 0.5
    minimum_b = center_b.astype(np.float64) - dims_b * 0.5
    maximum_b = center_b.astype(np.float64) + dims_b * 0.5
    overlap_dims = np.maximum(
        np.minimum(maximum_a, maximum_b)
        - np.maximum(minimum_a, minimum_b),
        0.0,
    )
    intersection = float(np.prod(overlap_dims))
    volume_a = float(np.prod(dims_a))
    volume_b = float(np.prod(dims_b))
    return intersection, volume_a, volume_b


def _aabb_iou(track: MaskGraphTrackEvidence, node: MaskGraphNode) -> float:
    intersection, volume_track, volume_node = _aabb_intersection(
        track.center, track.dims, node.center, node.dims
    )
    union = volume_track + volume_node - intersection
    return float(np.clip(intersection / union, 0.0, 1.0))


def _point_inside_fraction(
    points: Optional[np.ndarray],
    center: np.ndarray,
    dims: np.ndarray,
) -> Optional[float]:
    if points is None or points.shape[0] == 0:
        return None
    half = dims.astype(np.float64) * 0.5
    inside = np.all(
        (points >= center[None, :] - half[None, :])
        & (points <= center[None, :] + half[None, :]),
        axis=1,
    )
    return float(np.mean(inside))


def _mutual_inside(
    track: MaskGraphTrackEvidence,
    node: MaskGraphNode,
) -> Tuple[float, float, float]:
    intersection, volume_track, volume_node = _aabb_intersection(
        track.center, track.dims, node.center, node.dims
    )
    node_inside_track = _point_inside_fraction(
        node.points_world, track.center, track.dims
    )
    if node_inside_track is None:
        node_inside_track = intersection / volume_node
    track_inside_node = _point_inside_fraction(
        track.points_world, node.center, node.dims
    )
    if track_inside_node is None:
        track_inside_node = intersection / volume_track
    node_inside_track = float(np.clip(node_inside_track, 0.0, 1.0))
    track_inside_node = float(np.clip(track_inside_node, 0.0, 1.0))
    return (
        node_inside_track,
        track_inside_node,
        min(node_inside_track, track_inside_node),
    )


def _aabb_corners(center: np.ndarray, dims: np.ndarray) -> np.ndarray:
    signs = np.asarray(
        [
            [-1.0, -1.0, -1.0],
            [1.0, -1.0, -1.0],
            [1.0, 1.0, -1.0],
            [-1.0, 1.0, -1.0],
            [-1.0, -1.0, 1.0],
            [1.0, -1.0, 1.0],
            [1.0, 1.0, 1.0],
            [-1.0, 1.0, 1.0],
        ],
        dtype=np.float64,
    )
    return center.astype(np.float64)[None, :] + signs * dims[None, :] * 0.5


def _project_track_mask_iou(
    track: MaskGraphTrackEvidence,
    node: MaskGraphNode,
    config: Mapping[str, Any],
    projection_context: Optional[MaskGraphProjectionContext] = None,
) -> Optional[float]:
    if (
        node.mask is None
        or node.intrinsics is None
        or node.camera_to_world is None
    ):
        return None
    if projection_context is None:
        projection_context = build_projection_context(node, config)
    elif not isinstance(
        projection_context, MaskGraphProjectionContext
    ):
        raise ValueError(
            "projection_context must be a MaskGraphProjectionContext or None"
        )
    if projection_context is None:
        return None
    if (
        projection_context.node_id != node.node_id
        or projection_context.frame_id != node.frame_id
    ):
        raise ValueError(
            "projection_context belongs to a different mask-graph node"
        )
    binary = projection_context.binary_mask
    if binary.shape != node.mask.shape:
        raise ValueError(
            "projection_context binary mask shape disagrees with node.mask"
        )
    height, width = binary.shape
    world_to_camera = projection_context.world_to_camera
    corners = _aabb_corners(track.center, track.dims)
    homogeneous = np.column_stack(
        (corners, np.ones(corners.shape[0], dtype=np.float64))
    )
    camera = (homogeneous @ world_to_camera.T)[:, :3]
    in_front = camera[:, 2] > float(config["near_clip"])
    if not np.all(in_front):
        return 0.0
    pixels_h = camera @ node.intrinsics.astype(np.float64).T
    pixels = pixels_h[:, :2] / pixels_h[:, 2:3]
    x = np.clip(pixels[:, 0], 0.0, float(width))
    y = np.clip(pixels[:, 1], 0.0, float(height))
    box = np.asarray([x.min(), y.min(), x.max(), y.max()])
    x_start = max(0, min(width, int(np.floor(box[0]))))
    y_start = max(0, min(height, int(np.floor(box[1]))))
    x_stop = max(0, min(width, int(np.ceil(box[2]))))
    y_stop = max(0, min(height, int(np.ceil(box[3]))))
    box_area = max(x_stop - x_start, 0) * max(y_stop - y_start, 0)
    mask_area = projection_context.mask_area
    if box_area == 0 or mask_area == 0:
        return 0.0
    intersection = int(
        np.count_nonzero(binary[y_start:y_stop, x_start:x_stop])
    )
    union = box_area + mask_area - intersection
    return float(intersection / union) if union > 0 else 0.0


def _appearance_compatibility(
    cosine: float,
    neutral_cosine: float,
) -> float:
    """Map cosine to [0, 1] with the configured neutral point mapped to 0.5."""

    cosine = float(np.clip(cosine, -1.0, 1.0))
    if cosine >= neutral_cosine:
        denominator = max(1.0 - neutral_cosine, 1e-12)
        return float(
            np.clip(
                0.5 + 0.5 * (cosine - neutral_cosine) / denominator,
                0.0,
                1.0,
            )
        )
    denominator = max(neutral_cosine + 1.0, 1e-12)
    return float(
        np.clip(0.5 * (cosine + 1.0) / denominator, 0.0, 1.0)
    )


def _default_label_compatibility(
    track_label: Optional[str],
    node_label: Optional[str],
    config: Mapping[str, Any],
) -> Optional[float]:
    if track_label is None or node_label is None:
        return None
    left = _normalize_label(track_label)
    right = _normalize_label(node_label)
    if left == right:
        return 1.0
    for group in config["label_compatibility_groups"]:
        if left in group and right in group:
            return 1.0
    return float(config["label_mismatch_score"])


def _label_score(
    track_label: Optional[str],
    node_label: Optional[str],
    config: Mapping[str, Any],
    override: Optional[LabelCompatibility],
) -> Optional[float]:
    if track_label is None or node_label is None:
        return None
    if override is None:
        return _default_label_compatibility(
            track_label, node_label, config
        )
    if callable(override):
        value = override(track_label, node_label)
    else:
        value = override
    return _bounded_float("label_compatibility override", value)


def _best_source_node(
    graph: MaskGraphState,
    node: MaskGraphNode,
) -> MaskGraphNode:
    if not graph.nodes:
        raise ValueError("cannot evaluate an edge for an empty MaskGraphState")
    # Highest AABB IoU first, then most recently inserted node.  The aggregate
    # track still supplies all edge metrics; this merely chooses a stable graph
    # endpoint for diagnostics and connectivity.
    values = list(graph.nodes.values())
    best_index = max(
        range(len(values)),
        key=lambda index: (
            _aabb_iou(
                MaskGraphTrackEvidence(
                    values[index].center,
                    values[index].dims,
                ),
                node,
            ),
            index,
        ),
    )
    return values[best_index]


def _geometry_rejection_edge(
    source: MaskGraphNode,
    node: MaskGraphNode,
    *,
    iou: float,
    observation_inside: float = 0.0,
    track_inside: float = 0.0,
    mutual_inside: float = 0.0,
    geometry_score: float = 0.0,
    geometry_matches: int = 0,
) -> MaskGraphEdge:
    """Return a cheap, explicit geometry rejection.

    Metrics that require skipped point scans or raster projection are left at
    zero/``None``.  This helper is only used when the hard geometry match
    count is provably unreachable, so those unevaluated metrics cannot change
    the association decision.
    """

    return MaskGraphEdge(
        source_id=source.node_id,
        target_id=node.node_id,
        source_frame_id=source.frame_id,
        target_frame_id=node.frame_id,
        accepted=False,
        reason="geometry",
        score=geometry_score,
        geometry_score=geometry_score,
        iou_3d=iou,
        observation_inside_track=observation_inside,
        track_inside_observation=track_inside,
        mutual_inside=mutual_inside,
        projection_iou=None,
        appearance_cosine=None,
        appearance_compatibility=None,
        label_compatibility=None,
        geometry_matches=geometry_matches,
    )


def evaluate_edge(
    track: object,
    graph: MaskGraphState,
    observation: object,
    *,
    node_id: Optional[NodeId] = None,
    frame_id: Optional[FrameId] = None,
    label_compatibility: Optional[LabelCompatibility] = None,
    projection_context: Optional[MaskGraphProjectionContext] = None,
) -> MaskGraphEdge:
    """Evaluate one new observation against the current aggregate track.

    This function never mutates ``graph``.  Use :func:`update_mask_graph` for
    seed insertion, accepted-node insertion, edge accumulation, and unique
    frame confirmation.
    """

    if not isinstance(graph, MaskGraphState):
        raise ValueError("graph must be a MaskGraphState")
    node = coerce_mask_graph_node(
        observation, node_id=node_id, frame_id=frame_id
    )
    if projection_context is not None:
        if not isinstance(
            projection_context, MaskGraphProjectionContext
        ):
            raise ValueError(
                "projection_context must be a "
                "MaskGraphProjectionContext or None"
            )
        if (
            projection_context.node_id != node.node_id
            or projection_context.frame_id != node.frame_id
        ):
            raise ValueError(
                "projection_context belongs to a different mask-graph node"
            )
    source = _best_source_node(graph, node)
    config = graph.config
    if not bool(config["enabled"]):
        return MaskGraphEdge(
            source_id=source.node_id,
            target_id=node.node_id,
            source_frame_id=source.frame_id,
            target_frame_id=node.frame_id,
            accepted=False,
            reason="disabled",
            score=0.0,
            geometry_score=0.0,
            iou_3d=0.0,
            observation_inside_track=0.0,
            track_inside_observation=0.0,
            mutual_inside=0.0,
            projection_iou=None,
            appearance_cosine=None,
            appearance_compatibility=None,
            label_compatibility=None,
            geometry_matches=0,
        )

    track_evidence = coerce_track_evidence(track, graph)
    intersection, track_volume, node_volume = _aabb_intersection(
        track_evidence.center,
        track_evidence.dims,
        node.center,
        node.dims,
    )
    union = track_volume + node_volume - intersection
    iou = float(np.clip(intersection / union, 0.0, 1.0))
    iou_match = int(iou >= float(config["minimum_iou_3d"]))
    projection_available = (
        node.mask is not None
        and node.intrinsics is not None
        and node.camera_to_world is not None
    )
    minimum_matches = int(config["minimum_geometry_matches"])

    # Disjoint AABBs cannot be the same local instance.  Reject before
    # scanning either point set or touching any image-sized mask.
    if intersection <= 0.0:
        return _geometry_rejection_edge(
            source,
            node,
            iou=iou,
            geometry_matches=iou_match,
        )

    # Mutual containment can contribute at most one match and projection can
    # contribute at most one.  If even this optimistic bound cannot meet the
    # hard gate, neither point scanning nor projection can affect the result.
    optimistic_matches = (
        iou_match + 1 + int(projection_available)
    )
    if optimistic_matches < minimum_matches:
        geometry_weights = [
            float(config["iou_3d_weight"]),
            float(config["mutual_inside_weight"]),
        ]
        base_weight = float(sum(geometry_weights))
        geometry_score = (
            float(
                np.average([iou, 0.0], weights=geometry_weights)
            )
            if base_weight > 0.0
            else 0.0
        )
        return _geometry_rejection_edge(
            source,
            node,
            iou=iou,
            geometry_score=geometry_score,
            geometry_matches=iou_match,
        )

    observation_inside, track_inside, mutual_inside = _mutual_inside(
        track_evidence, node
    )
    mutual_match = int(
        mutual_inside >= float(config["minimum_mutual_inside"])
    )
    base_matches = iou_match + mutual_match

    # Use the actual mutual-containment outcome to avoid camera projection
    # when its best possible contribution still cannot satisfy the gate.
    if base_matches + int(projection_available) < minimum_matches:
        geometry_weights = [
            float(config["iou_3d_weight"]),
            float(config["mutual_inside_weight"]),
        ]
        base_weight = float(sum(geometry_weights))
        geometry_score = (
            float(
                np.average(
                    [iou, mutual_inside],
                    weights=geometry_weights,
                )
            )
            if base_weight > 0.0
            else 0.0
        )
        return _geometry_rejection_edge(
            source,
            node,
            iou=iou,
            observation_inside=observation_inside,
            track_inside=track_inside,
            mutual_inside=mutual_inside,
            geometry_score=geometry_score,
            geometry_matches=base_matches,
        )

    projection_iou = _project_track_mask_iou(
        track_evidence,
        node,
        config,
        projection_context=projection_context,
    )

    geometry_values = [iou, mutual_inside]
    geometry_weights = [
        float(config["iou_3d_weight"]),
        float(config["mutual_inside_weight"]),
    ]
    geometry_matches = base_matches
    if projection_iou is not None:
        geometry_values.append(projection_iou)
        geometry_weights.append(float(config["projection_iou_weight"]))
        geometry_matches += int(
            projection_iou >= float(config["minimum_projection_iou"])
        )
    geometry_score = float(
        np.average(geometry_values, weights=geometry_weights)
    )

    appearance_cosine = None
    appearance_compatibility = None
    if (
        track_evidence.appearance_feature is not None
        and node.appearance_feature is not None
        and track_evidence.appearance_feature.shape
        == node.appearance_feature.shape
    ):
        appearance_cosine = float(
            np.clip(
                np.dot(
                    track_evidence.appearance_feature,
                    node.appearance_feature,
                ),
                -1.0,
                1.0,
            )
        )
        appearance_compatibility = _appearance_compatibility(
            appearance_cosine,
            float(config["appearance_neutral_cosine"]),
        )

    label_score = _label_score(
        track_evidence.label,
        node.label,
        config,
        label_compatibility,
    )

    score_sum = geometry_score
    score_weight = 1.0
    if appearance_compatibility is not None:
        weight = float(config["appearance_weight"])
        score_sum += weight * appearance_compatibility
        score_weight += weight
    if label_score is not None:
        weight = float(config["label_weight"])
        score_sum += weight * label_score
        score_weight += weight
    score = float(np.clip(score_sum / score_weight, 0.0, 1.0))

    projection_missing = (
        bool(config["require_projection"]) and projection_iou is None
    )
    projection_failed = (
        bool(config["require_projection"])
        and projection_iou is not None
        and projection_iou < float(config["minimum_projection_iou"])
    )
    enough_geometry = (
        geometry_matches >= int(config["minimum_geometry_matches"])
    )
    if projection_missing:
        accepted = False
        reason = "missing_projection"
    elif projection_failed:
        accepted = False
        reason = "projection"
    elif not enough_geometry:
        accepted = False
        reason = "geometry"
    elif score < float(config["minimum_edge_score"]):
        accepted = False
        reason = "score"
    else:
        accepted = True
        reason = "accepted"

    return MaskGraphEdge(
        source_id=source.node_id,
        target_id=node.node_id,
        source_frame_id=source.frame_id,
        target_frame_id=node.frame_id,
        accepted=accepted,
        reason=reason,
        score=score,
        geometry_score=geometry_score,
        iou_3d=iou,
        observation_inside_track=observation_inside,
        track_inside_observation=track_inside,
        mutual_inside=mutual_inside,
        projection_iou=projection_iou,
        appearance_cosine=appearance_cosine,
        appearance_compatibility=appearance_compatibility,
        label_compatibility=label_score,
        geometry_matches=geometry_matches,
    )


def update_mask_graph(
    track: object,
    graph: MaskGraphState,
    observation: object,
    *,
    node_id: Optional[NodeId] = None,
    frame_id: Optional[FrameId] = None,
    label_compatibility: Optional[LabelCompatibility] = None,
    projection_context: Optional[MaskGraphProjectionContext] = None,
) -> MaskGraphUpdate:
    """Seed or increment a graph and perform unique-frame confirmation."""

    if not isinstance(graph, MaskGraphState):
        raise ValueError("graph must be a MaskGraphState")
    if not bool(graph.config["enabled"]):
        return MaskGraphUpdate(
            accepted=False,
            seeded=False,
            confirmed=graph.confirmed,
            became_confirmed=False,
            node_id=None,
            edge=None,
            reason="disabled",
        )

    if isinstance(observation, MaskGraphNode):
        node = coerce_mask_graph_node(
            observation, node_id=node_id, frame_id=frame_id
        )
    else:
        resolved_frame = _first_not_none(
            frame_id,
            _get(observation, "frame_id"),
            _get(observation, "frame_index"),
            _get(_get(observation, "view"), "frame_id"),
            _get(_get(observation, "view"), "frame_index"),
        )
        if resolved_frame is None:
            raise ValueError("observation does not expose a frame_id")
        resolved_node = (
            node_id
            if node_id is not None
            else _get(observation, "node_id")
        )
        if resolved_node is None:
            resolved_node = graph.next_node_id(resolved_frame)
        node = coerce_mask_graph_node(
            observation,
            node_id=resolved_node,
            frame_id=resolved_frame,
        )

    if not graph.nodes:
        became_confirmed = graph.add_node(node)
        return MaskGraphUpdate(
            accepted=True,
            seeded=True,
            confirmed=graph.confirmed,
            became_confirmed=became_confirmed,
            node_id=node.node_id,
            edge=None,
            reason="seed",
        )

    edge = evaluate_edge(
        track,
        graph,
        node,
        label_compatibility=label_compatibility,
        projection_context=projection_context,
    )
    if not edge.accepted:
        return MaskGraphUpdate(
            accepted=False,
            seeded=False,
            confirmed=graph.confirmed,
            became_confirmed=False,
            node_id=node.node_id,
            edge=edge,
            reason=edge.reason,
        )
    became_confirmed = graph.add_node(node)
    graph.add_edge(edge)
    return MaskGraphUpdate(
        accepted=True,
        seeded=False,
        confirmed=graph.confirmed,
        became_confirmed=became_confirmed,
        node_id=node.node_id,
        edge=edge,
        reason=edge.reason,
    )


__all__ = [
    "DEFAULT_MASK_GRAPH_CONFIG",
    "LabelCompatibility",
    "MaskGraphEdge",
    "MaskGraphNode",
    "MaskGraphProjectionContext",
    "MaskGraphState",
    "MaskGraphTrackEvidence",
    "MaskGraphUpdate",
    "build_projection_context",
    "coerce_mask_graph_node",
    "coerce_track_evidence",
    "evaluate_edge",
    "resolve_mask_graph_config",
    "update_mask_graph",
]
