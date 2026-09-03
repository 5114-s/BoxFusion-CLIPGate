"""Deterministic observer for missing Mask-RGBD instances.

This module is an intentionally standalone, NumPy-only proposal-confirmation
path.  It does not import :mod:`online_refinement`, invoke a proposal provider,
write files, or mutate BoxFusion detections.  A caller supplies SAM3/YOLOE-like
mask and depth observations which have already failed the normal global B6
assignment.  The observer then:

* back-projects valid mask pixels and separates them with depth-aware 3D
  connected components;
* incrementally associates components with semantic and hard geometric gates;
* confirms a track only after evidence from distinct views;
* expires active tracks on a provider-call clock and optionally archives
  confirmed tracks;
* rejects global-box and confirmed-candidate duplicates; and
* emits immutable, gravity-aligned oriented-box records for diagnostics only.

The design borrows the central MaskClustering/Zoo3D idea that 2D masks are
graph nodes and a 3D instance is supported by compatible cross-view edges.
It is deliberately more conservative than a semantic or appearance tracker:
labels and optional appearance features can rank an already valid edge, but
they can never rescue a pair which failed metric geometry.

Every public array is defensively copied.  Unexpected observation failures
are isolated when ``fail_open`` is enabled, which is the default: an observer
failure can produce an empty diagnostic result, but it cannot alter a caller's
global boxes or primary output.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from numbers import Integral, Real
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import numpy as np


Identifier = Union[int, str]
MISSING_INSTANCE_GRAPH_SCHEMA = "boxfusion.missing_instance_graph.v1"

# Fixed gate/integration contract.  Every value is normalized to [0, 1] and
# emitted in exactly this order as ``OrientedMissingCandidate.feature_vector``.
MISSING_GRAPH_FEATURE_NAMES: Tuple[str, ...] = (
    "multi_view_confirmed",
    "unique_view_support",
    "node_support",
    "edge_support",
    "mean_detector_score",
    "mean_edge_score",
    "mean_iou_3d",
    "mean_containment",
    "mean_projection_support",
    "semantic_agreement",
    "mean_component_fraction",
    "lifecycle_active",
    "lifecycle_span",
    "lifecycle_freshness",
    "maximum_global_iou",
    "maximum_global_containment",
    "maximum_candidate_iou",
    "maximum_candidate_containment",
    "point_support",
    "orientation_anisotropy",
)


DEFAULT_MISSING_INSTANCE_GRAPH_CONFIG: Dict[str, Any] = {
    # This route is never a detection exporter.  Keeping it opt-in still
    # mirrors the safety convention used by the existing Mask Graph modules.
    "enabled": False,
    "fail_open": True,
    # Mask/depth lifting.
    "mask_threshold": 0.50,
    "min_depth": 0.10,
    "max_depth": 6.00,
    "depth_scale": 1.00,
    "component_connectivity": 8,
    "component_max_depth_jump": 0.20,
    "component_max_world_distance": 0.15,
    "minimum_component_pixels": 8,
    "minimum_component_points": 8,
    "maximum_components_per_proposal": 4,
    "max_points_per_component": 2048,
    "max_points_per_track": 8192,
    "aabb_lower_quantile": 0.02,
    "aabb_upper_quantile": 0.98,
    "minimum_dimension": 0.02,
    # Semantics are a compatibility gate, never a substitute for geometry.
    "semantic_compatibility_groups": (),
    "allow_unknown_semantics": True,
    "unknown_semantic_score": 0.50,
    "minimum_semantic_score": 0.50,
    # Incremental edge gates.  IoU, containment, and projection are three
    # independently thresholded geometry signals.
    "minimum_iou_3d": 0.02,
    "minimum_containment": 0.10,
    "minimum_projection_support": 0.05,
    "minimum_geometry_matches": 2,
    "maximum_center_distance": 0.75,
    # Projection uses the current component mask.  This tolerance is used by
    # the point-support diagnostic; AABB/mask projection remains available
    # even when no old point is depth-consistent in the current view.
    "projection_depth_tolerance": 0.15,
    # Cross-view confirmation and bounded graph state.
    "min_unique_frames": 2,
    "track_ttl_provider_calls": 10,
    "archive_confirmed": True,
    "max_nodes_per_track": 32,
    "max_edges_per_track": 128,
    # Same-view proposal NMS, frozen-global rejection, and final candidate NMS.
    "same_view_duplicate_iou": 0.85,
    "same_view_duplicate_containment": 0.95,
    "global_reject_iou": 0.30,
    "global_reject_containment": 0.70,
    "candidate_duplicate_iou": 0.35,
    "candidate_duplicate_containment": 0.70,
    # Near-isotropic XY point sets use yaw zero rather than an unstable PCA
    # direction.  The value is (lambda_max-lambda_min)/(lambda_max+lambda_min).
    "minimum_orientation_anisotropy": 0.05,
}


def _finite_float(name: str, value: object) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real number")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
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


def _strict_int(name: str, value: object, minimum: int = 0) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer")
    result = int(value)
    if result < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return result


def _strict_bool(name: str, value: object) -> bool:
    if not isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a boolean")
    return bool(value)


def _normalize_label(label: str) -> str:
    return " ".join(
        label.casefold().replace("_", " ").replace("-", " ").split()
    )


def _resolve_semantic_groups(
    value: object,
) -> Tuple[Tuple[str, ...], ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(
            "semantic_compatibility_groups must be a sequence of groups"
        )
    groups: List[Tuple[str, ...]] = []
    occupied: Dict[str, int] = {}
    for group_index, group in enumerate(value):
        if isinstance(group, (str, bytes)) or not isinstance(group, Sequence):
            raise ValueError(
                "each semantic compatibility group must be a sequence"
            )
        normalized: List[str] = []
        for label in group:
            if not isinstance(label, str) or not label.strip():
                raise ValueError(
                    "semantic compatibility labels must be non-empty strings"
                )
            item = _normalize_label(label)
            if item in normalized:
                raise ValueError(
                    "semantic compatibility groups cannot contain duplicates"
                )
            if item in occupied:
                raise ValueError(
                    "a semantic label cannot occur in more than one group"
                )
            occupied[item] = group_index
            normalized.append(item)
        if len(normalized) < 2:
            raise ValueError(
                "each semantic compatibility group needs at least two labels"
            )
        groups.append(tuple(normalized))
    return tuple(groups)


def resolve_missing_instance_graph_config(
    config: Optional[Mapping[str, object]] = None,
) -> Dict[str, Any]:
    """Return a detached, strictly validated observer configuration."""

    if config is None:
        config = {}
    if not isinstance(config, Mapping):
        raise ValueError("missing_instance_graph config must be a mapping")
    unknown = sorted(set(config) - set(DEFAULT_MISSING_INSTANCE_GRAPH_CONFIG))
    if unknown:
        raise ValueError(
            "Unknown missing_instance_graph config key(s): "
            + ", ".join(unknown)
        )
    resolved = dict(DEFAULT_MISSING_INSTANCE_GRAPH_CONFIG)
    resolved.update(config)

    for key in (
        "enabled",
        "fail_open",
        "allow_unknown_semantics",
        "archive_confirmed",
    ):
        resolved[key] = _strict_bool(
            f"missing_instance_graph.{key}", resolved[key]
        )

    for key, minimum in (
        ("minimum_component_pixels", 1),
        ("minimum_component_points", 1),
        ("maximum_components_per_proposal", 1),
        ("max_points_per_component", 1),
        ("max_points_per_track", 1),
        ("minimum_geometry_matches", 1),
        ("min_unique_frames", 2),
        ("track_ttl_provider_calls", 0),
        ("max_nodes_per_track", 2),
        ("max_edges_per_track", 1),
    ):
        resolved[key] = _strict_int(
            f"missing_instance_graph.{key}", resolved[key], minimum
        )
    connectivity = _strict_int(
        "missing_instance_graph.component_connectivity",
        resolved["component_connectivity"],
        4,
    )
    if connectivity not in (4, 8):
        raise ValueError(
            "missing_instance_graph.component_connectivity must be 4 or 8"
        )
    resolved["component_connectivity"] = connectivity
    if resolved["minimum_geometry_matches"] > 3:
        raise ValueError(
            "missing_instance_graph.minimum_geometry_matches cannot exceed 3"
        )
    if resolved["max_nodes_per_track"] < resolved["min_unique_frames"]:
        raise ValueError(
            "missing_instance_graph.max_nodes_per_track must be at least "
            "min_unique_frames"
        )

    for key in (
        "mask_threshold",
        "unknown_semantic_score",
        "minimum_semantic_score",
        "minimum_iou_3d",
        "minimum_containment",
        "minimum_projection_support",
        "same_view_duplicate_iou",
        "same_view_duplicate_containment",
        "global_reject_iou",
        "global_reject_containment",
        "candidate_duplicate_iou",
        "candidate_duplicate_containment",
        "minimum_orientation_anisotropy",
    ):
        resolved[key] = _bounded_float(
            f"missing_instance_graph.{key}", resolved[key]
        )
    for key in (
        "min_depth",
        "max_depth",
        "depth_scale",
        "component_max_depth_jump",
        "component_max_world_distance",
        "aabb_lower_quantile",
        "aabb_upper_quantile",
        "minimum_dimension",
        "maximum_center_distance",
        "projection_depth_tolerance",
    ):
        resolved[key] = _finite_float(
            f"missing_instance_graph.{key}", resolved[key]
        )
    for key in (
        "depth_scale",
        "component_max_depth_jump",
        "component_max_world_distance",
        "minimum_dimension",
        "maximum_center_distance",
        "projection_depth_tolerance",
    ):
        if resolved[key] <= 0.0:
            raise ValueError(
                f"missing_instance_graph.{key} must be positive"
            )
    if resolved["min_depth"] < 0.0:
        raise ValueError(
            "missing_instance_graph.min_depth must be non-negative"
        )
    if resolved["max_depth"] <= resolved["min_depth"]:
        raise ValueError(
            "missing_instance_graph.max_depth must exceed min_depth"
        )
    lower = resolved["aabb_lower_quantile"]
    upper = resolved["aabb_upper_quantile"]
    if not 0.0 <= lower < upper <= 1.0:
        raise ValueError(
            "missing_instance_graph AABB quantiles must satisfy "
            "0 <= lower < upper <= 1"
        )
    resolved["semantic_compatibility_groups"] = _resolve_semantic_groups(
        resolved["semantic_compatibility_groups"]
    )
    return resolved


def _as_array(value: object, name: str) -> np.ndarray:
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
        raise ValueError(f"{name} cannot be converted to an array") from error


def _readonly_array(
    value: object,
    name: str,
    *,
    dtype: Any = np.float32,
    shape: Optional[Tuple[int, ...]] = None,
    finite: bool = True,
) -> np.ndarray:
    array = _as_array(value, name)
    if not (
        np.issubdtype(array.dtype, np.number)
        or np.issubdtype(array.dtype, np.bool_)
    ):
        raise ValueError(f"{name} must be numeric")
    array = np.asarray(array, dtype=dtype)
    if shape is not None and array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}")
    if finite and not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    result = np.array(array, dtype=dtype, copy=True)
    result.setflags(write=False)
    return result


def _identifier(name: str, value: object) -> Identifier:
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


def _identifier_key(value: Identifier) -> Tuple[int, Union[int, str]]:
    if isinstance(value, int):
        return (0, value)
    return (1, value)


def _optional_label(value: object, name: str) -> Optional[str]:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string or None")
    return value.strip()


def _optional_feature(value: object, name: str) -> Optional[np.ndarray]:
    if value is None:
        return None
    feature = _readonly_array(value, name, dtype=np.float32)
    if feature.ndim != 1 or feature.size < 1:
        raise ValueError(f"{name} must be a non-empty vector")
    norm = float(np.linalg.norm(feature.astype(np.float64)))
    if norm <= 1e-12:
        raise ValueError(f"{name} must have non-zero norm")
    result = np.asarray(feature / norm, dtype=np.float32).copy()
    result.setflags(write=False)
    return result


def _readonly_points(value: object, name: str) -> np.ndarray:
    points = _readonly_array(value, name, dtype=np.float32)
    if points.ndim != 2 or points.shape[1:] != (3,):
        raise ValueError(f"{name} must have shape [N, 3]")
    return points


def _readonly_box(value: object, name: str, dimensions: int = 6) -> np.ndarray:
    box = _readonly_array(
        value, name, dtype=np.float32, shape=(dimensions,)
    )
    if np.any(box[3:6] <= 0.0):
        raise ValueError(f"{name} dimensions must be positive")
    return box


@dataclass(frozen=True)
class MaskDepthProposalObservation:
    """One unmatched SAM3/YOLOE mask with aligned metric depth.

    Depth may contain zeros, NaNs, or infinities; those pixels are treated as
    missing evidence by the lifting stage.  All other inputs are copied and
    made read-only during construction.
    """

    frame_id: Identifier
    proposal_id: Identifier
    mask: np.ndarray
    depth: np.ndarray
    intrinsics: np.ndarray
    camera_to_world: np.ndarray
    score: float
    label: Optional[str] = None
    feature: Optional[np.ndarray] = None
    provider: str = "unknown"

    def __post_init__(self) -> None:
        frame_id = _identifier(
            "MaskDepthProposalObservation.frame_id", self.frame_id
        )
        proposal_id = _identifier(
            "MaskDepthProposalObservation.proposal_id", self.proposal_id
        )
        mask = _as_array(
            self.mask, "MaskDepthProposalObservation.mask"
        )
        if mask.ndim != 2 or min(mask.shape, default=0) < 1:
            raise ValueError(
                "MaskDepthProposalObservation.mask must have shape [H, W]"
            )
        if not (
            np.issubdtype(mask.dtype, np.number)
            or np.issubdtype(mask.dtype, np.bool_)
        ):
            raise ValueError(
                "MaskDepthProposalObservation.mask must be numeric or boolean"
            )
        if not np.isfinite(mask).all():
            raise ValueError(
                "MaskDepthProposalObservation.mask must be finite"
            )
        mask = np.asarray(mask, dtype=np.float32).copy()
        mask.setflags(write=False)

        depth = _as_array(
            self.depth, "MaskDepthProposalObservation.depth"
        )
        if (
            depth.shape != mask.shape
            or not np.issubdtype(depth.dtype, np.number)
        ):
            raise ValueError(
                "MaskDepthProposalObservation.depth must have numeric shape "
                "matching mask"
            )
        # Non-finite depth is a normal missing-depth sentinel.
        depth = np.asarray(depth, dtype=np.float32).copy()
        depth.setflags(write=False)

        intrinsics = _readonly_array(
            self.intrinsics,
            "MaskDepthProposalObservation.intrinsics",
            dtype=np.float64,
            shape=(3, 3),
        )
        pose = _readonly_array(
            self.camera_to_world,
            "MaskDepthProposalObservation.camera_to_world",
            dtype=np.float64,
            shape=(4, 4),
        )
        if abs(float(np.linalg.det(intrinsics))) <= 1e-12:
            raise ValueError(
                "MaskDepthProposalObservation.intrinsics must be invertible"
            )
        if abs(float(np.linalg.det(pose))) <= 1e-12:
            raise ValueError(
                "MaskDepthProposalObservation.camera_to_world must be "
                "invertible"
            )
        score = _bounded_float(
            "MaskDepthProposalObservation.score", self.score
        )
        label = _optional_label(
            self.label, "MaskDepthProposalObservation.label"
        )
        feature = _optional_feature(
            self.feature, "MaskDepthProposalObservation.feature"
        )
        if not isinstance(self.provider, str) or not self.provider.strip():
            raise ValueError(
                "MaskDepthProposalObservation.provider must be a non-empty "
                "string"
            )
        provider = self.provider.strip().casefold()

        object.__setattr__(self, "frame_id", frame_id)
        object.__setattr__(self, "proposal_id", proposal_id)
        object.__setattr__(self, "mask", mask)
        object.__setattr__(self, "depth", depth)
        object.__setattr__(self, "intrinsics", intrinsics)
        object.__setattr__(self, "camera_to_world", pose)
        object.__setattr__(self, "score", score)
        object.__setattr__(self, "label", label)
        object.__setattr__(self, "feature", feature)
        object.__setattr__(self, "provider", provider)


# Short, discoverable aliases for callers which do not need the provider name
# in the type itself.
MissingMaskDepthObservation = MaskDepthProposalObservation
MaskDepthObservation = MaskDepthProposalObservation


@dataclass(frozen=True)
class LiftedMaskComponent:
    """One deterministic 3D connected component from a proposal mask."""

    component_id: str
    frame_id: Identifier
    proposal_id: Identifier
    component_index: int
    provider: str
    label: Optional[str]
    score: float
    feature: Optional[np.ndarray]
    points_world: np.ndarray
    aabb: np.ndarray
    component_mask: np.ndarray
    depth: np.ndarray
    intrinsics: np.ndarray
    camera_to_world: np.ndarray
    pixel_count: int
    source_mask_pixels: int
    source_valid_depth_pixels: int
    component_fraction: float

    def __post_init__(self) -> None:
        if not isinstance(self.component_id, str) or not self.component_id:
            raise ValueError(
                "LiftedMaskComponent.component_id must be a non-empty string"
            )
        frame_id = _identifier(
            "LiftedMaskComponent.frame_id", self.frame_id
        )
        proposal_id = _identifier(
            "LiftedMaskComponent.proposal_id", self.proposal_id
        )
        component_index = _strict_int(
            "LiftedMaskComponent.component_index",
            self.component_index,
            0,
        )
        if not isinstance(self.provider, str) or not self.provider:
            raise ValueError(
                "LiftedMaskComponent.provider must be a non-empty string"
            )
        label = _optional_label(self.label, "LiftedMaskComponent.label")
        score = _bounded_float("LiftedMaskComponent.score", self.score)
        feature = _optional_feature(
            self.feature, "LiftedMaskComponent.feature"
        )
        points = _readonly_points(
            self.points_world, "LiftedMaskComponent.points_world"
        )
        if len(points) < 1:
            raise ValueError(
                "LiftedMaskComponent.points_world must not be empty"
            )
        aabb = _readonly_box(self.aabb, "LiftedMaskComponent.aabb")
        component_mask = _readonly_array(
            self.component_mask,
            "LiftedMaskComponent.component_mask",
            dtype=np.bool_,
            finite=False,
        )
        if component_mask.ndim != 2 or min(component_mask.shape) < 1:
            raise ValueError(
                "LiftedMaskComponent.component_mask must have shape [H, W]"
            )
        depth = _as_array(self.depth, "LiftedMaskComponent.depth")
        if (
            depth.shape != component_mask.shape
            or not np.issubdtype(depth.dtype, np.number)
        ):
            raise ValueError(
                "LiftedMaskComponent.depth must match component_mask"
            )
        depth = np.asarray(depth, dtype=np.float32).copy()
        depth.setflags(write=False)
        intrinsics = _readonly_array(
            self.intrinsics,
            "LiftedMaskComponent.intrinsics",
            dtype=np.float64,
            shape=(3, 3),
        )
        pose = _readonly_array(
            self.camera_to_world,
            "LiftedMaskComponent.camera_to_world",
            dtype=np.float64,
            shape=(4, 4),
        )
        pixel_count = _strict_int(
            "LiftedMaskComponent.pixel_count", self.pixel_count, 1
        )
        source_mask_pixels = _strict_int(
            "LiftedMaskComponent.source_mask_pixels",
            self.source_mask_pixels,
            pixel_count,
        )
        source_valid_depth_pixels = _strict_int(
            "LiftedMaskComponent.source_valid_depth_pixels",
            self.source_valid_depth_pixels,
            0,
        )
        fraction = _bounded_float(
            "LiftedMaskComponent.component_fraction",
            self.component_fraction,
        )
        object.__setattr__(self, "frame_id", frame_id)
        object.__setattr__(self, "proposal_id", proposal_id)
        object.__setattr__(self, "component_index", component_index)
        object.__setattr__(self, "label", label)
        object.__setattr__(self, "score", score)
        object.__setattr__(self, "feature", feature)
        object.__setattr__(self, "points_world", points)
        object.__setattr__(self, "aabb", aabb)
        object.__setattr__(self, "component_mask", component_mask)
        object.__setattr__(self, "depth", depth)
        object.__setattr__(self, "intrinsics", intrinsics)
        object.__setattr__(self, "camera_to_world", pose)
        object.__setattr__(self, "pixel_count", pixel_count)
        object.__setattr__(self, "source_mask_pixels", source_mask_pixels)
        object.__setattr__(
            self,
            "source_valid_depth_pixels",
            source_valid_depth_pixels,
        )
        object.__setattr__(self, "component_fraction", fraction)


@dataclass(frozen=True)
class MissingObservationAudit:
    """Auditable fate of one proposal or lifted component."""

    provider_call_index: int
    frame_id: Optional[Identifier]
    proposal_id: Optional[Identifier]
    component_index: Optional[int]
    component_id: Optional[str]
    track_id: Optional[int]
    accepted: bool
    reason: str
    detail: str = ""
    pixel_count: int = 0
    point_count: int = 0
    component_fraction: float = 0.0
    maximum_global_iou: float = 0.0
    maximum_global_containment: float = 0.0
    duplicate_of: Optional[str] = None

    def __post_init__(self) -> None:
        call = _strict_int(
            "MissingObservationAudit.provider_call_index",
            self.provider_call_index,
            0,
        )
        frame = (
            None
            if self.frame_id is None
            else _identifier("MissingObservationAudit.frame_id", self.frame_id)
        )
        proposal = (
            None
            if self.proposal_id is None
            else _identifier(
                "MissingObservationAudit.proposal_id", self.proposal_id
            )
        )
        component_index = self.component_index
        if component_index is not None:
            component_index = _strict_int(
                "MissingObservationAudit.component_index",
                component_index,
                0,
            )
        track_id = self.track_id
        if track_id is not None:
            track_id = _strict_int(
                "MissingObservationAudit.track_id", track_id, 0
            )
        if not isinstance(self.accepted, (bool, np.bool_)):
            raise ValueError(
                "MissingObservationAudit.accepted must be a boolean"
            )
        if not isinstance(self.reason, str) or not self.reason:
            raise ValueError(
                "MissingObservationAudit.reason must be a non-empty string"
            )
        if not isinstance(self.detail, str):
            raise ValueError("MissingObservationAudit.detail must be a string")
        pixel_count = _strict_int(
            "MissingObservationAudit.pixel_count", self.pixel_count, 0
        )
        point_count = _strict_int(
            "MissingObservationAudit.point_count", self.point_count, 0
        )
        fraction = _bounded_float(
            "MissingObservationAudit.component_fraction",
            self.component_fraction,
        )
        global_iou = _bounded_float(
            "MissingObservationAudit.maximum_global_iou",
            self.maximum_global_iou,
        )
        global_containment = _bounded_float(
            "MissingObservationAudit.maximum_global_containment",
            self.maximum_global_containment,
        )
        if self.component_id is not None and (
            not isinstance(self.component_id, str) or not self.component_id
        ):
            raise ValueError(
                "MissingObservationAudit.component_id must be non-empty or "
                "None"
            )
        if self.duplicate_of is not None and (
            not isinstance(self.duplicate_of, str) or not self.duplicate_of
        ):
            raise ValueError(
                "MissingObservationAudit.duplicate_of must be non-empty or "
                "None"
            )
        object.__setattr__(self, "provider_call_index", call)
        object.__setattr__(self, "frame_id", frame)
        object.__setattr__(self, "proposal_id", proposal)
        object.__setattr__(self, "component_index", component_index)
        object.__setattr__(self, "track_id", track_id)
        object.__setattr__(self, "accepted", bool(self.accepted))
        object.__setattr__(self, "pixel_count", pixel_count)
        object.__setattr__(self, "point_count", point_count)
        object.__setattr__(self, "component_fraction", fraction)
        object.__setattr__(self, "maximum_global_iou", global_iou)
        object.__setattr__(
            self, "maximum_global_containment", global_containment
        )


@dataclass(frozen=True)
class MissingAssociationAudit:
    """One evaluated graph edge with every hard and soft metric."""

    track_id: int
    component_id: str
    accepted: bool
    reason: str
    score: float
    semantic_compatibility: float
    iou_3d: float
    containment: float
    mutual_inside: float
    projection_support: float
    point_projection_support: float
    center_distance: float
    geometry_matches: int
    appearance_cosine: Optional[float]

    def __post_init__(self) -> None:
        track_id = _strict_int(
            "MissingAssociationAudit.track_id", self.track_id, 0
        )
        if not isinstance(self.component_id, str) or not self.component_id:
            raise ValueError(
                "MissingAssociationAudit.component_id must be non-empty"
            )
        if not isinstance(self.accepted, (bool, np.bool_)):
            raise ValueError(
                "MissingAssociationAudit.accepted must be a boolean"
            )
        if not isinstance(self.reason, str) or not self.reason:
            raise ValueError(
                "MissingAssociationAudit.reason must be non-empty"
            )
        for name in (
            "score",
            "semantic_compatibility",
            "iou_3d",
            "containment",
            "mutual_inside",
            "projection_support",
            "point_projection_support",
        ):
            object.__setattr__(
                self,
                name,
                _bounded_float(
                    f"MissingAssociationAudit.{name}", getattr(self, name)
                ),
            )
        center_distance = _finite_float(
            "MissingAssociationAudit.center_distance",
            self.center_distance,
        )
        if center_distance < 0.0:
            raise ValueError(
                "MissingAssociationAudit.center_distance must be non-negative"
            )
        matches = _strict_int(
            "MissingAssociationAudit.geometry_matches",
            self.geometry_matches,
            0,
        )
        cosine = self.appearance_cosine
        if cosine is not None:
            cosine = _bounded_float(
                "MissingAssociationAudit.appearance_cosine",
                cosine,
                -1.0,
                1.0,
            )
        object.__setattr__(self, "track_id", track_id)
        object.__setattr__(self, "accepted", bool(self.accepted))
        object.__setattr__(self, "center_distance", center_distance)
        object.__setattr__(self, "geometry_matches", matches)
        object.__setattr__(self, "appearance_cosine", cosine)


@dataclass(frozen=True)
class MissingCandidateFeatures:
    """Named, serialization-friendly evidence for a candidate decision."""

    unique_views: int
    mean_detector_score: float
    maximum_detector_score: float
    mean_edge_score: float
    mean_iou_3d: float
    mean_containment: float
    mean_projection_support: float
    mean_center_distance: float
    semantic_agreement: float
    mean_component_fraction: float
    maximum_global_iou: float
    maximum_global_containment: float
    maximum_candidate_iou: float
    maximum_candidate_containment: float
    orientation_anisotropy: float

    def __post_init__(self) -> None:
        unique_views = _strict_int(
            "MissingCandidateFeatures.unique_views",
            self.unique_views,
            1,
        )
        for name in (
            "mean_detector_score",
            "maximum_detector_score",
            "mean_edge_score",
            "mean_iou_3d",
            "mean_containment",
            "mean_projection_support",
            "semantic_agreement",
            "mean_component_fraction",
            "maximum_global_iou",
            "maximum_global_containment",
            "maximum_candidate_iou",
            "maximum_candidate_containment",
            "orientation_anisotropy",
        ):
            object.__setattr__(
                self,
                name,
                _bounded_float(
                    f"MissingCandidateFeatures.{name}", getattr(self, name)
                ),
            )
        center_distance = _finite_float(
            "MissingCandidateFeatures.mean_center_distance",
            self.mean_center_distance,
        )
        if center_distance < 0.0:
            raise ValueError(
                "MissingCandidateFeatures.mean_center_distance must be "
                "non-negative"
            )
        object.__setattr__(self, "unique_views", unique_views)
        object.__setattr__(
            self, "mean_center_distance", center_distance
        )

    def as_dict(self) -> Dict[str, Union[int, float]]:
        return {
            name: getattr(self, name)
            for name in self.__dataclass_fields__
        }


@dataclass(frozen=True)
class OrientedMissingCandidate:
    """Observer-only, multi-view-confirmed missing-instance proposal."""

    track_id: int
    label: Optional[str]
    oriented_box: np.ndarray
    corners: np.ndarray
    aabb: np.ndarray
    score: float
    frame_ids: Tuple[Identifier, ...]
    provider_call_first: int
    provider_call_last: int
    node_count: int
    edge_count: int
    point_count: int
    lifecycle_state: str
    reason: str
    features: MissingCandidateFeatures
    feature_vector: np.ndarray
    appearance_feature: Optional[np.ndarray] = None
    audit_reasons: Tuple[str, ...] = (
        "hard_geometry",
        "multi_view_confirmed",
        "global_overlap_pass",
        "candidate_duplicate_pass",
        "observer_only",
    )
    observer_only: bool = True
    schema: str = MISSING_INSTANCE_GRAPH_SCHEMA

    def __post_init__(self) -> None:
        track_id = _strict_int(
            "OrientedMissingCandidate.track_id", self.track_id, 0
        )
        label = _optional_label(
            self.label, "OrientedMissingCandidate.label"
        )
        oriented_box = _readonly_box(
            self.oriented_box,
            "OrientedMissingCandidate.oriented_box",
            dimensions=7,
        )
        corners = _readonly_array(
            self.corners,
            "OrientedMissingCandidate.corners",
            dtype=np.float32,
            shape=(8, 3),
        )
        aabb = _readonly_box(
            self.aabb, "OrientedMissingCandidate.aabb"
        )
        score = _bounded_float(
            "OrientedMissingCandidate.score", self.score
        )
        frames: List[Identifier] = []
        for frame_id in self.frame_ids:
            normalized = _identifier(
                "OrientedMissingCandidate.frame_ids entry", frame_id
            )
            if normalized not in frames:
                frames.append(normalized)
        if len(frames) < 2:
            raise ValueError(
                "OrientedMissingCandidate requires at least two distinct "
                "frame_ids"
            )
        frames.sort(key=_identifier_key)
        first = _strict_int(
            "OrientedMissingCandidate.provider_call_first",
            self.provider_call_first,
            0,
        )
        last = _strict_int(
            "OrientedMissingCandidate.provider_call_last",
            self.provider_call_last,
            first,
        )
        node_count = _strict_int(
            "OrientedMissingCandidate.node_count", self.node_count, 2
        )
        edge_count = _strict_int(
            "OrientedMissingCandidate.edge_count", self.edge_count, 1
        )
        point_count = _strict_int(
            "OrientedMissingCandidate.point_count", self.point_count, 1
        )
        if self.lifecycle_state not in {"active", "archived"}:
            raise ValueError(
                "OrientedMissingCandidate.lifecycle_state must be active or "
                "archived"
            )
        if not isinstance(self.reason, str) or not self.reason:
            raise ValueError(
                "OrientedMissingCandidate.reason must be non-empty"
            )
        if not isinstance(self.features, MissingCandidateFeatures):
            raise ValueError(
                "OrientedMissingCandidate.features must be "
                "MissingCandidateFeatures"
            )
        feature_vector = _readonly_array(
            self.feature_vector,
            "OrientedMissingCandidate.feature_vector",
            dtype=np.float32,
            shape=(len(MISSING_GRAPH_FEATURE_NAMES),),
        )
        if np.any(feature_vector < 0.0) or np.any(feature_vector > 1.0):
            raise ValueError(
                "OrientedMissingCandidate.feature_vector must lie in [0, 1]"
            )
        feature = _optional_feature(
            self.appearance_feature,
            "OrientedMissingCandidate.appearance_feature",
        )
        reasons: List[str] = []
        for reason in self.audit_reasons:
            if not isinstance(reason, str) or not reason:
                raise ValueError(
                    "OrientedMissingCandidate.audit_reasons entries must be "
                    "non-empty strings"
                )
            if reason not in reasons:
                reasons.append(reason)
        if not isinstance(self.observer_only, (bool, np.bool_)) or not bool(
            self.observer_only
        ):
            raise ValueError(
                "OrientedMissingCandidate must remain observer_only"
            )
        if self.schema != MISSING_INSTANCE_GRAPH_SCHEMA:
            raise ValueError(
                "OrientedMissingCandidate.schema does not match the module "
                "schema"
            )
        object.__setattr__(self, "track_id", track_id)
        object.__setattr__(self, "label", label)
        object.__setattr__(self, "oriented_box", oriented_box)
        object.__setattr__(self, "corners", corners)
        object.__setattr__(self, "aabb", aabb)
        object.__setattr__(self, "score", score)
        object.__setattr__(self, "frame_ids", tuple(frames))
        object.__setattr__(self, "provider_call_first", first)
        object.__setattr__(self, "provider_call_last", last)
        object.__setattr__(self, "node_count", node_count)
        object.__setattr__(self, "edge_count", edge_count)
        object.__setattr__(self, "point_count", point_count)
        object.__setattr__(self, "feature_vector", feature_vector)
        object.__setattr__(self, "appearance_feature", feature)
        object.__setattr__(self, "audit_reasons", tuple(reasons))
        object.__setattr__(self, "observer_only", True)

    @property
    def candidate_id(self) -> int:
        """Stable scene-local identifier used by aggregate oracle exports."""

        return self.track_id

    @property
    def valid(self) -> bool:
        return True

    @property
    def verified(self) -> bool:
        return True

    @property
    def confirmed(self) -> bool:
        # Construction itself enforces at least two distinct frame ids.
        return True

    def as_dict(self) -> Dict[str, Any]:
        """Return a detached record suitable for JSON/NPZ adaptation."""

        return {
            "schema": self.schema,
            "candidate_id": self.candidate_id,
            "track_id": self.track_id,
            "label": self.label,
            "oriented_box": self.oriented_box.copy(),
            "corners": self.corners.copy(),
            "aabb": self.aabb.copy(),
            "score": self.score,
            "frame_ids": self.frame_ids,
            "provider_call_first": self.provider_call_first,
            "provider_call_last": self.provider_call_last,
            "node_count": self.node_count,
            "edge_count": self.edge_count,
            "point_count": self.point_count,
            "lifecycle_state": self.lifecycle_state,
            "reason": self.reason,
            "features": self.features.as_dict(),
            "feature_names": MISSING_GRAPH_FEATURE_NAMES,
            "feature_vector": self.feature_vector.copy(),
            "appearance_feature": (
                None
                if self.appearance_feature is None
                else self.appearance_feature.copy()
            ),
            "audit_reasons": self.audit_reasons,
            "observer_only": True,
            "valid": self.valid,
            "verified": self.verified,
            "confirmed": self.confirmed,
        }

    def as_supplemental_candidate(self) -> Dict[str, Any]:
        """Adapt to ``boxfusion.trifusion.supplemental_candidates.v1``.

        This is deliberately a zero-I/O adapter.  The aggregate artifact owns
        its scene id and schema; this record supplies exactly one canonical
        candidate row.  ``label`` remains diagnostic and never participates
        in geometric acceptance downstream.
        """

        return {
            "candidate_id": self.track_id,
            "source": "missing_graph",
            "corners": self.corners.copy(),
            "score": self.score,
            "label": self.label,
            "valid": True,
            "verified": True,
        }


# A shorter alias is convenient for consumers which already know the route.
MissingCandidateRecord = OrientedMissingCandidate


@dataclass(frozen=True)
class MissingCandidateDecision:
    """Why a track did or did not become an emitted observer record."""

    track_id: int
    lifecycle_state: str
    confirmed: bool
    unique_views: int
    accepted: bool
    reason: str
    duplicate_of_track_id: Optional[int]
    maximum_global_iou: float
    maximum_global_containment: float
    maximum_candidate_iou: float
    maximum_candidate_containment: float

    def __post_init__(self) -> None:
        track_id = _strict_int(
            "MissingCandidateDecision.track_id", self.track_id, 0
        )
        if self.lifecycle_state not in {"active", "archived"}:
            raise ValueError(
                "MissingCandidateDecision.lifecycle_state must be active or "
                "archived"
            )
        for name in ("confirmed", "accepted"):
            if not isinstance(getattr(self, name), (bool, np.bool_)):
                raise ValueError(
                    f"MissingCandidateDecision.{name} must be a boolean"
                )
        unique_views = _strict_int(
            "MissingCandidateDecision.unique_views",
            self.unique_views,
            1,
        )
        if not isinstance(self.reason, str) or not self.reason:
            raise ValueError(
                "MissingCandidateDecision.reason must be non-empty"
            )
        duplicate = self.duplicate_of_track_id
        if duplicate is not None:
            duplicate = _strict_int(
                "MissingCandidateDecision.duplicate_of_track_id",
                duplicate,
                0,
            )
        for name in (
            "maximum_global_iou",
            "maximum_global_containment",
            "maximum_candidate_iou",
            "maximum_candidate_containment",
        ):
            object.__setattr__(
                self,
                name,
                _bounded_float(
                    f"MissingCandidateDecision.{name}", getattr(self, name)
                ),
            )
        object.__setattr__(self, "track_id", track_id)
        object.__setattr__(self, "confirmed", bool(self.confirmed))
        object.__setattr__(self, "accepted", bool(self.accepted))
        object.__setattr__(self, "unique_views", unique_views)
        object.__setattr__(self, "duplicate_of_track_id", duplicate)


@dataclass(frozen=True)
class MissingInstanceGraphUpdate:
    """Complete observer result for one provider call."""

    provider_call_index: int
    candidates: Tuple[OrientedMissingCandidate, ...]
    decisions: Tuple[MissingCandidateDecision, ...]
    observations: Tuple[MissingObservationAudit, ...]
    associations: Tuple[MissingAssociationAudit, ...]
    expired_track_ids: Tuple[int, ...]
    archived_track_ids: Tuple[int, ...]
    discarded_track_ids: Tuple[int, ...]
    errors: Tuple[str, ...] = ()
    disabled: bool = False

    def __post_init__(self) -> None:
        call = _strict_int(
            "MissingInstanceGraphUpdate.provider_call_index",
            self.provider_call_index,
            0,
        )
        for name, expected in (
            ("candidates", OrientedMissingCandidate),
            ("decisions", MissingCandidateDecision),
            ("observations", MissingObservationAudit),
            ("associations", MissingAssociationAudit),
        ):
            values = tuple(getattr(self, name))
            if not all(isinstance(value, expected) for value in values):
                raise ValueError(
                    f"MissingInstanceGraphUpdate.{name} has an invalid entry"
                )
            object.__setattr__(self, name, values)
        for name in (
            "expired_track_ids",
            "archived_track_ids",
            "discarded_track_ids",
        ):
            values = tuple(
                _strict_int(
                    f"MissingInstanceGraphUpdate.{name} entry", value, 0
                )
                for value in getattr(self, name)
            )
            object.__setattr__(self, name, values)
        errors = tuple(self.errors)
        if not all(isinstance(error, str) and error for error in errors):
            raise ValueError(
                "MissingInstanceGraphUpdate.errors entries must be non-empty "
                "strings"
            )
        if not isinstance(self.disabled, (bool, np.bool_)):
            raise ValueError(
                "MissingInstanceGraphUpdate.disabled must be a boolean"
            )
        object.__setattr__(self, "provider_call_index", call)
        object.__setattr__(self, "errors", errors)
        object.__setattr__(self, "disabled", bool(self.disabled))

    @property
    def failed_open(self) -> bool:
        return bool(self.errors)

    def summary(self) -> Dict[str, Any]:
        rejection_counts: Dict[str, int] = {}
        for audit in self.observations:
            if not audit.accepted:
                rejection_counts[audit.reason] = (
                    rejection_counts.get(audit.reason, 0) + 1
                )
        for decision in self.decisions:
            if not decision.accepted:
                rejection_counts[decision.reason] = (
                    rejection_counts.get(decision.reason, 0) + 1
                )
        return {
            "schema": MISSING_INSTANCE_GRAPH_SCHEMA,
            "provider_call_index": self.provider_call_index,
            "observer_candidates": len(self.candidates),
            "candidate_decisions": len(self.decisions),
            "observation_audits": len(self.observations),
            "associations_evaluated": len(self.associations),
            "associations_accepted": sum(
                int(edge.accepted) for edge in self.associations
            ),
            "expired_tracks": len(self.expired_track_ids),
            "archived_tracks": len(self.archived_track_ids),
            "discarded_tracks": len(self.discarded_track_ids),
            "failed_open": self.failed_open,
            "disabled": self.disabled,
            "rejections": dict(sorted(rejection_counts.items())),
        }


def coerce_mask_depth_observation(
    value: object,
) -> MaskDepthProposalObservation:
    """Adapt a dataclass, mapping, or duck-typed proposal observation."""

    if isinstance(value, MaskDepthProposalObservation):
        return value

    def get(name: str, default: object = None) -> object:
        if isinstance(value, Mapping):
            return value.get(name, default)
        return getattr(value, name, default)

    return MaskDepthProposalObservation(
        frame_id=get("frame_id"),
        proposal_id=get("proposal_id"),
        mask=get("mask"),
        depth=get("depth"),
        intrinsics=get("intrinsics"),
        camera_to_world=get("camera_to_world"),
        score=get("score"),
        label=get("label"),
        feature=get("feature"),
        provider=get("provider", "unknown"),
    )


def coerce_lifted_mask_component(
    value: object,
    *,
    proposal_id: Optional[Identifier] = None,
    component_id: Optional[str] = None,
    component_index: int = 0,
    frame_id: Optional[Identifier] = None,
    config: Optional[Mapping[str, object]] = None,
) -> LiftedMaskComponent:
    """Adapt an already-lifted BoxFusion-like proposal without relifting it.

    The adapter understands the existing ``LiftedProposal`` layout:
    ``box``, ``observation.points_world``, ``proposal.mask/score/label/feature``
    and ``view.frame_index/intrinsics/camera_to_world``.  Raw aligned depth is
    optional on this path.  When absent, point-depth projection support is
    zero while projected AABB/mask IoU remains available as hard geometry.
    """

    if isinstance(value, LiftedMaskComponent):
        if proposal_id is not None and proposal_id != value.proposal_id:
            raise ValueError(
                "proposal_id override disagrees with LiftedMaskComponent"
            )
        if component_id is not None and component_id != value.component_id:
            raise ValueError(
                "component_id override disagrees with LiftedMaskComponent"
            )
        if frame_id is not None and frame_id != value.frame_id:
            raise ValueError(
                "frame_id override disagrees with LiftedMaskComponent"
            )
        return value
    resolved = resolve_missing_instance_graph_config(config)

    def get(source: object, name: str) -> object:
        if source is None:
            return None
        if isinstance(source, Mapping):
            return source.get(name)
        return getattr(source, name, None)

    proposal = get(value, "proposal")
    depth_observation = get(value, "observation")
    view = get(value, "view")
    resolved_frame = frame_id
    if resolved_frame is None:
        for candidate in (
            get(value, "frame_id"),
            get(value, "frame_index"),
            get(view, "frame_id"),
            get(view, "frame_index"),
        ):
            if candidate is not None:
                resolved_frame = candidate
                break
    if resolved_frame is None:
        raise ValueError("lifted proposal does not expose a frame_id")
    resolved_frame = _identifier("lifted frame_id", resolved_frame)
    resolved_proposal = proposal_id
    if resolved_proposal is None:
        for candidate in (
            get(value, "proposal_id"),
            get(value, "node_id"),
            get(proposal, "proposal_id"),
        ):
            if candidate is not None:
                resolved_proposal = candidate
                break
    if resolved_proposal is None:
        raise ValueError(
            "lifted proposal does not expose a proposal_id; pass one "
            "explicitly"
        )
    resolved_proposal = _identifier(
        "lifted proposal_id", resolved_proposal
    )
    resolved_component_index = _strict_int(
        "lifted component_index", component_index, 0
    )

    points = get(value, "points_world")
    if points is None:
        points = get(depth_observation, "points_world")
    if points is None:
        points = get(value, "points")
    points = _readonly_points(points, "lifted points_world")
    if len(points) < 1:
        raise ValueError("lifted points_world must not be empty")

    box = get(value, "aabb")
    if box is None:
        box = get(value, "box")
    if box is None:
        box = get(depth_observation, "aabb")
    if isinstance(box, (tuple, list)) and len(box) == 2:
        box = np.concatenate(
            (
                np.asarray(box[0], dtype=np.float64),
                np.asarray(box[1], dtype=np.float64),
            )
        )
    if box is None:
        box = _robust_aabb(points, resolved)
    aabb = _readonly_box(box, "lifted aabb")

    mask = get(value, "component_mask")
    if mask is None:
        mask = get(value, "mask")
    if mask is None:
        mask = get(proposal, "mask")
    mask_array = _as_array(mask, "lifted mask")
    if (
        mask_array.ndim != 2
        or min(mask_array.shape, default=0) < 1
        or not (
            np.issubdtype(mask_array.dtype, np.number)
            or np.issubdtype(mask_array.dtype, np.bool_)
        )
        or not np.isfinite(mask_array).all()
    ):
        raise ValueError("lifted mask must have finite numeric shape [H, W]")
    binary_mask = np.asarray(
        mask_array >= float(resolved["mask_threshold"]), dtype=np.bool_
    )
    if not np.any(binary_mask):
        raise ValueError("lifted mask must contain foreground pixels")

    intrinsics = get(value, "intrinsics")
    if intrinsics is None:
        intrinsics = get(view, "intrinsics")
    pose = get(value, "camera_to_world")
    if pose is None:
        pose = get(view, "camera_to_world")
    depth = get(value, "depth")
    if depth is None:
        depth = get(depth_observation, "depth")
    if depth is None:
        depth_array = np.full(binary_mask.shape, np.nan, dtype=np.float32)
    else:
        depth_array = _as_array(depth, "lifted depth")
        if (
            depth_array.shape != binary_mask.shape
            or not np.issubdtype(depth_array.dtype, np.number)
        ):
            raise ValueError("lifted depth must match mask shape")
        depth_array = np.asarray(depth_array, dtype=np.float32)

    score = get(value, "score")
    if score is None:
        score = get(proposal, "score")
    if score is None:
        score = 1.0
    label = get(value, "label")
    if label is None:
        label = get(proposal, "label")
    feature = get(value, "feature")
    if feature is None:
        feature = get(value, "appearance_feature")
    if feature is None:
        feature = get(proposal, "feature")
    provider = get(value, "provider")
    if provider is None:
        provider = get(proposal, "provider")
    if provider is None:
        provider = "unknown"
    if component_id is None:
        component_id = (
            f"{str(provider).strip().casefold()}:frame={resolved_frame}:"
            f"proposal={resolved_proposal}:"
            f"component={resolved_component_index}"
        )
    valid_depth = (
        binary_mask
        & np.isfinite(depth_array)
        & (
            depth_array.astype(np.float64)
            * float(resolved["depth_scale"])
            >= float(resolved["min_depth"])
        )
        & (
            depth_array.astype(np.float64)
            * float(resolved["depth_scale"])
            <= float(resolved["max_depth"])
        )
    )
    return LiftedMaskComponent(
        component_id=component_id,
        frame_id=resolved_frame,
        proposal_id=resolved_proposal,
        component_index=resolved_component_index,
        provider=str(provider),
        label=label,
        score=score,
        feature=feature,
        points_world=_bounded_points(
            points, int(resolved["max_points_per_component"])
        ),
        aabb=aabb,
        component_mask=binary_mask,
        depth=depth_array,
        intrinsics=intrinsics,
        camera_to_world=pose,
        pixel_count=int(np.count_nonzero(binary_mask)),
        source_mask_pixels=int(np.count_nonzero(binary_mask)),
        source_valid_depth_pixels=int(np.count_nonzero(valid_depth)),
        component_fraction=1.0,
    )


def _observation_sort_key(
    observation: MaskDepthProposalObservation,
) -> Tuple[Any, ...]:
    binary = observation.mask > 0.0
    nonzero = np.flatnonzero(binary)
    first_pixel = int(nonzero[0]) if len(nonzero) else observation.mask.size
    # The geometry terms make ordering independent of input list order even
    # when two providers happen to reuse the same proposal identifier.
    finite_depth = observation.depth[np.isfinite(observation.depth)]
    depth_sum = (
        float(np.sum(finite_depth, dtype=np.float64))
        if len(finite_depth)
        else 0.0
    )
    return (
        _identifier_key(observation.frame_id),
        observation.provider,
        _identifier_key(observation.proposal_id),
        "" if observation.label is None else _normalize_label(observation.label),
        -observation.score,
        int(np.count_nonzero(binary)),
        first_pixel,
        depth_sum,
    )


def _component_identifier(
    observation: MaskDepthProposalObservation,
    component_index: int,
) -> str:
    return (
        f"{observation.provider}:frame={observation.frame_id}:"
        f"proposal={observation.proposal_id}:component={component_index}"
    )


def _backproject_pixels(
    rows: np.ndarray,
    columns: np.ndarray,
    depths: np.ndarray,
    intrinsics: np.ndarray,
    camera_to_world: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    pixels = np.column_stack(
        (
            columns.astype(np.float64),
            rows.astype(np.float64),
            np.ones(len(rows), dtype=np.float64),
        )
    )
    rays = pixels @ np.linalg.inv(intrinsics).T
    ray_z = rays[:, 2]
    if np.any(np.abs(ray_z) <= 1e-12):
        raise ValueError("intrinsics produce a zero camera-z ray")
    camera_points = rays * (depths / ray_z)[:, None]
    homogeneous = np.column_stack(
        (camera_points, np.ones(len(camera_points), dtype=np.float64))
    )
    world = homogeneous @ camera_to_world.T
    world_points = world[:, :3]
    if not np.isfinite(world_points).all():
        raise ValueError("back-projection produced non-finite world points")
    return camera_points, world_points


def _union_find_components(
    rows: np.ndarray,
    columns: np.ndarray,
    camera_points: np.ndarray,
    world_points: np.ndarray,
    shape: Tuple[int, int],
    config: Mapping[str, Any],
) -> List[np.ndarray]:
    count = len(rows)
    if count == 0:
        return []
    pixel_to_index = np.full(shape, -1, dtype=np.int64)
    pixel_to_index[rows, columns] = np.arange(count, dtype=np.int64)
    parents = np.arange(count, dtype=np.int64)

    def find(index: int) -> int:
        root = index
        while parents[root] != root:
            root = int(parents[root])
        while parents[index] != index:
            parent = int(parents[index])
            parents[index] = root
            index = parent
        return root

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root == right_root:
            return
        # Smaller raster index is always the root.  Component identities are
        # therefore independent of hash/dictionary iteration.
        if left_root < right_root:
            parents[right_root] = left_root
        else:
            parents[left_root] = right_root

    offsets = [(-1, -1), (-1, 0), (-1, 1), (0, -1)]
    if int(config["component_connectivity"]) == 4:
        offsets = [(-1, 0), (0, -1)]
    max_depth_jump = float(config["component_max_depth_jump"])
    max_world_distance = float(config["component_max_world_distance"])
    height, width = shape
    for current in range(count):
        row = int(rows[current])
        column = int(columns[current])
        for row_offset, column_offset in offsets:
            neighbor_row = row + row_offset
            neighbor_column = column + column_offset
            if (
                neighbor_row < 0
                or neighbor_row >= height
                or neighbor_column < 0
                or neighbor_column >= width
            ):
                continue
            neighbor = int(
                pixel_to_index[neighbor_row, neighbor_column]
            )
            if neighbor < 0:
                continue
            if (
                abs(
                    float(camera_points[current, 2])
                    - float(camera_points[neighbor, 2])
                )
                > max_depth_jump
            ):
                continue
            distance = float(
                np.linalg.norm(
                    world_points[current] - world_points[neighbor]
                )
            )
            if distance <= max_world_distance:
                union(current, neighbor)

    groups: Dict[int, List[int]] = {}
    for index in range(count):
        groups.setdefault(find(index), []).append(index)
    components = [
        np.asarray(indices, dtype=np.int64)
        for _, indices in sorted(groups.items())
    ]
    components.sort(
        key=lambda indices: (
            -len(indices),
            int(rows[indices[0]]) * shape[1] + int(columns[indices[0]]),
        )
    )
    return components


def _bounded_points(points: np.ndarray, maximum: int) -> np.ndarray:
    values = np.asarray(points, dtype=np.float64)
    if len(values) == 0:
        result = np.empty((0, 3), dtype=np.float32)
        result.setflags(write=False)
        return result
    order = np.lexsort((values[:, 2], values[:, 1], values[:, 0]))
    values = values[order]
    if len(values) > maximum:
        # Evenly cover the deterministic lexicographic sequence instead of
        # relying on an RNG or on input pixel order.
        indices = np.linspace(
            0, len(values) - 1, maximum, dtype=np.int64
        )
        values = values[indices]
    result = np.asarray(values, dtype=np.float32).copy()
    result.setflags(write=False)
    return result


def _robust_aabb(
    points: np.ndarray,
    config: Mapping[str, Any],
) -> np.ndarray:
    values = np.asarray(points, dtype=np.float64)
    if values.ndim != 2 or values.shape[1:] != (3,) or len(values) == 0:
        raise ValueError("cannot form an AABB without finite [N, 3] points")
    if not np.isfinite(values).all():
        raise ValueError("AABB points must be finite")
    lower = np.quantile(
        values, float(config["aabb_lower_quantile"]), axis=0
    )
    upper = np.quantile(
        values, float(config["aabb_upper_quantile"]), axis=0
    )
    center = (lower + upper) * 0.5
    dims = upper - lower
    minimum = float(config["minimum_dimension"])
    dims = np.maximum(dims, minimum)
    result = np.concatenate((center, dims)).astype(np.float32)
    result.setflags(write=False)
    return result


def _lift_components_detailed(
    observation: MaskDepthProposalObservation,
    config: Mapping[str, Any],
) -> Tuple[Tuple[LiftedMaskComponent, ...], Tuple[Tuple[str, int, int], ...]]:
    binary = observation.mask >= float(config["mask_threshold"])
    mask_pixels = int(np.count_nonzero(binary))
    scaled_depth = (
        observation.depth.astype(np.float64)
        * float(config["depth_scale"])
    )
    valid = (
        binary
        & np.isfinite(scaled_depth)
        & (scaled_depth >= float(config["min_depth"]))
        & (scaled_depth <= float(config["max_depth"]))
    )
    rows, columns = np.nonzero(valid)
    valid_depth_pixels = len(rows)
    if valid_depth_pixels == 0:
        return (), (("no_valid_mask_depth", mask_pixels, 0),)
    depths = scaled_depth[rows, columns]
    camera_points, world_points = _backproject_pixels(
        rows,
        columns,
        depths,
        observation.intrinsics,
        observation.camera_to_world,
    )
    component_indices = _union_find_components(
        rows,
        columns,
        camera_points,
        world_points,
        observation.depth.shape,
        config,
    )
    components: List[LiftedMaskComponent] = []
    rejections: List[Tuple[str, int, int]] = []
    minimum_pixels = int(config["minimum_component_pixels"])
    minimum_points = int(config["minimum_component_points"])
    maximum_components = int(config["maximum_components_per_proposal"])
    accepted_rank = 0
    for raw_rank, indices in enumerate(component_indices):
        pixel_count = int(len(indices))
        point_count = pixel_count
        if pixel_count < minimum_pixels or point_count < minimum_points:
            rejections.append(
                ("component_too_small", pixel_count, point_count)
            )
            continue
        if accepted_rank >= maximum_components:
            rejections.append(
                ("component_limit", pixel_count, point_count)
            )
            continue
        selected_world = world_points[indices]
        aabb = _robust_aabb(selected_world, config)
        stored_points = _bounded_points(
            selected_world, int(config["max_points_per_component"])
        )
        component_mask = np.zeros(
            observation.depth.shape, dtype=np.bool_
        )
        component_mask[rows[indices], columns[indices]] = True
        component_mask.setflags(write=False)
        components.append(
            LiftedMaskComponent(
                component_id=_component_identifier(
                    observation, accepted_rank
                ),
                frame_id=observation.frame_id,
                proposal_id=observation.proposal_id,
                component_index=accepted_rank,
                provider=observation.provider,
                label=observation.label,
                score=observation.score,
                feature=observation.feature,
                points_world=stored_points,
                aabb=aabb,
                component_mask=component_mask,
                depth=observation.depth,
                intrinsics=observation.intrinsics,
                camera_to_world=observation.camera_to_world,
                pixel_count=pixel_count,
                source_mask_pixels=mask_pixels,
                source_valid_depth_pixels=valid_depth_pixels,
                component_fraction=(
                    float(point_count) / float(valid_depth_pixels)
                ),
            )
        )
        accepted_rank += 1
    if not components and not rejections:
        rejections.append(("no_eligible_component", mask_pixels, 0))
    return tuple(components), tuple(rejections)


def lift_mask_depth_components(
    observation: object,
    config: Optional[Mapping[str, object]] = None,
) -> Tuple[LiftedMaskComponent, ...]:
    """Lift a raw observation into deterministic 3D components.

    Small or over-limit components are omitted.  The observer's ``update``
    method additionally exposes those omissions as audit rows.
    """

    resolved = resolve_missing_instance_graph_config(config)
    normalized = coerce_mask_depth_observation(observation)
    components, _ = _lift_components_detailed(normalized, resolved)
    return components


def _box_intersection_metrics(
    box_a: np.ndarray,
    box_b: np.ndarray,
) -> Tuple[float, float, float]:
    minimum_a = box_a[:3].astype(np.float64) - box_a[3:6] * 0.5
    maximum_a = box_a[:3].astype(np.float64) + box_a[3:6] * 0.5
    minimum_b = box_b[:3].astype(np.float64) - box_b[3:6] * 0.5
    maximum_b = box_b[:3].astype(np.float64) + box_b[3:6] * 0.5
    overlap_dims = np.maximum(
        np.minimum(maximum_a, maximum_b)
        - np.maximum(minimum_a, minimum_b),
        0.0,
    )
    intersection = float(np.prod(overlap_dims))
    volume_a = float(np.prod(box_a[3:6]))
    volume_b = float(np.prod(box_b[3:6]))
    union = volume_a + volume_b - intersection
    iou = (
        float(np.clip(intersection / union, 0.0, 1.0))
        if union > 0.0
        else 0.0
    )
    smaller = min(volume_a, volume_b)
    containment = (
        float(np.clip(intersection / smaller, 0.0, 1.0))
        if smaller > 0.0
        else 0.0
    )
    return intersection, iou, containment


def _points_inside_fraction(points: np.ndarray, box: np.ndarray) -> float:
    if len(points) == 0:
        return 0.0
    half = box[3:6].astype(np.float64) * 0.5
    inside = np.all(
        (points >= box[None, :3] - half[None, :])
        & (points <= box[None, :3] + half[None, :]),
        axis=1,
    )
    return float(np.mean(inside))


def _maximum_box_overlap(
    candidate: np.ndarray,
    boxes: np.ndarray,
) -> Tuple[float, float]:
    maximum_iou = 0.0
    maximum_containment = 0.0
    for box in boxes:
        _, iou, containment = _box_intersection_metrics(candidate, box)
        maximum_iou = max(maximum_iou, iou)
        maximum_containment = max(maximum_containment, containment)
    return maximum_iou, maximum_containment


def _oriented_aabb_from_box(box: np.ndarray) -> np.ndarray:
    center = box[:3].astype(np.float64)
    dims = box[3:6].astype(np.float64)
    yaw = float(box[6])
    cosine = abs(float(np.cos(yaw)))
    sine = abs(float(np.sin(yaw)))
    envelope = np.asarray(
        [
            cosine * dims[0] + sine * dims[1],
            sine * dims[0] + cosine * dims[1],
            dims[2],
        ],
        dtype=np.float64,
    )
    return np.concatenate((center, envelope)).astype(np.float32)


def _coerce_global_boxes(value: object) -> np.ndarray:
    if value is None:
        return np.empty((0, 6), dtype=np.float32)
    array = _as_array(value, "global_boxes")
    if array.size == 0:
        return np.empty((0, 6), dtype=np.float32)
    if array.ndim == 1:
        array = array[None, :]
    elif array.ndim == 2 and array.shape == (8, 3):
        array = array[None, :, :]
    if array.ndim == 3 and array.shape[1:] == (8, 3):
        if not np.issubdtype(array.dtype, np.number):
            raise ValueError("global box corners must be numeric")
        corners = np.asarray(array, dtype=np.float64)
        if not np.isfinite(corners).all():
            raise ValueError("global box corners must be finite")
        minimum = corners.min(axis=1)
        maximum = corners.max(axis=1)
        array = np.column_stack(
            ((minimum + maximum) * 0.5, maximum - minimum)
        )
    elif array.ndim == 2 and array.shape[1] in (6, 7):
        if not np.issubdtype(array.dtype, np.number):
            raise ValueError("global_boxes must be numeric")
        array = np.asarray(array, dtype=np.float64)
        if not np.isfinite(array).all():
            raise ValueError("global_boxes must be finite")
        if array.shape[1] == 7:
            array = np.stack(
                [_oriented_aabb_from_box(row) for row in array], axis=0
            )
    else:
        raise ValueError(
            "global_boxes must have shape [N,6], [N,7], or [N,8,3]"
        )
    boxes = np.asarray(array, dtype=np.float32)
    if np.any(boxes[:, 3:6] <= 0.0):
        raise ValueError("global box dimensions must be positive")
    return boxes.copy()


def _aabb_corners(box: np.ndarray) -> np.ndarray:
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
    return (
        box[:3].astype(np.float64)[None, :]
        + signs * box[3:6].astype(np.float64)[None, :] * 0.5
    )


def _projected_aabb_mask_iou(
    box: np.ndarray,
    component: LiftedMaskComponent,
) -> float:
    corners = _aabb_corners(box)
    world_to_camera = np.linalg.inv(component.camera_to_world)
    homogeneous = np.column_stack(
        (corners, np.ones(8, dtype=np.float64))
    )
    camera = (homogeneous @ world_to_camera.T)[:, :3]
    if not np.all(camera[:, 2] > 1e-6):
        return 0.0
    pixels_h = camera @ component.intrinsics.T
    pixels = pixels_h[:, :2] / pixels_h[:, 2:3]
    height, width = component.component_mask.shape
    x = np.clip(pixels[:, 0], 0.0, float(width))
    y = np.clip(pixels[:, 1], 0.0, float(height))
    x_start = max(0, min(width, int(np.floor(x.min()))))
    x_stop = max(0, min(width, int(np.ceil(x.max()))))
    y_start = max(0, min(height, int(np.floor(y.min()))))
    y_stop = max(0, min(height, int(np.ceil(y.max()))))
    rectangle_area = (
        max(x_stop - x_start, 0) * max(y_stop - y_start, 0)
    )
    mask_area = int(np.count_nonzero(component.component_mask))
    if rectangle_area == 0 or mask_area == 0:
        return 0.0
    intersection = int(
        np.count_nonzero(
            component.component_mask[y_start:y_stop, x_start:x_stop]
        )
    )
    union = rectangle_area + mask_area - intersection
    return float(intersection / union) if union > 0 else 0.0


def _projected_point_support(
    points_world: np.ndarray,
    component: LiftedMaskComponent,
    config: Mapping[str, Any],
) -> float:
    if len(points_world) == 0:
        return 0.0
    world_to_camera = np.linalg.inv(component.camera_to_world)
    homogeneous = np.column_stack(
        (
            points_world.astype(np.float64),
            np.ones(len(points_world), dtype=np.float64),
        )
    )
    camera = (homogeneous @ world_to_camera.T)[:, :3]
    front = camera[:, 2] > 1e-6
    if not np.any(front):
        return 0.0
    camera = camera[front]
    pixels_h = camera @ component.intrinsics.T
    pixels = pixels_h[:, :2] / pixels_h[:, 2:3]
    columns = np.rint(pixels[:, 0]).astype(np.int64)
    rows = np.rint(pixels[:, 1]).astype(np.int64)
    height, width = component.component_mask.shape
    inside_image = (
        (rows >= 0)
        & (rows < height)
        & (columns >= 0)
        & (columns < width)
    )
    if not np.any(inside_image):
        return 0.0
    camera = camera[inside_image]
    rows = rows[inside_image]
    columns = columns[inside_image]
    observed_depth = (
        component.depth[rows, columns].astype(np.float64)
        * float(config["depth_scale"])
    )
    valid_depth = (
        np.isfinite(observed_depth)
        & (observed_depth >= float(config["min_depth"]))
        & (observed_depth <= float(config["max_depth"]))
    )
    if not np.any(valid_depth):
        return 0.0
    camera = camera[valid_depth]
    rows = rows[valid_depth]
    columns = columns[valid_depth]
    observed_depth = observed_depth[valid_depth]
    tolerance = float(config["projection_depth_tolerance"])
    observer = camera[:, 2] <= observed_depth + tolerance
    if not np.any(observer):
        return 0.0
    supported = (
        component.component_mask[rows, columns]
        & (np.abs(camera[:, 2] - observed_depth) <= tolerance)
    )
    return float(np.count_nonzero(supported & observer)) / float(
        np.count_nonzero(observer)
    )


def _semantic_compatibility(
    left: Optional[str],
    right: Optional[str],
    config: Mapping[str, Any],
) -> float:
    if left is None or right is None:
        if bool(config["allow_unknown_semantics"]):
            return float(config["unknown_semantic_score"])
        return 0.0
    normalized_left = _normalize_label(left)
    normalized_right = _normalize_label(right)
    if normalized_left == normalized_right:
        return 1.0
    for group in config["semantic_compatibility_groups"]:
        if normalized_left in group and normalized_right in group:
            return 1.0
    return 0.0


def _feature_cosine(
    left: Optional[np.ndarray],
    right: Optional[np.ndarray],
) -> Optional[float]:
    if left is None or right is None or left.shape != right.shape:
        return None
    return float(np.clip(np.dot(left, right), -1.0, 1.0))


def oriented_box_corners(oriented_box: object) -> np.ndarray:
    """Return immutable corners for ``[cx,cy,cz,dx,dy,dz,yaw]``."""

    box = _readonly_box(
        oriented_box, "oriented_box", dimensions=7
    ).astype(np.float64)
    center = box[:3]
    half = box[3:6] * 0.5
    local = np.asarray(
        [
            [-half[0], -half[1], -half[2]],
            [half[0], -half[1], -half[2]],
            [half[0], half[1], -half[2]],
            [-half[0], half[1], -half[2]],
            [-half[0], -half[1], half[2]],
            [half[0], -half[1], half[2]],
            [half[0], half[1], half[2]],
            [-half[0], half[1], half[2]],
        ],
        dtype=np.float64,
    )
    cosine = float(np.cos(box[6]))
    sine = float(np.sin(box[6]))
    basis = np.asarray(
        [[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    result = np.asarray(local @ basis.T + center[None, :], dtype=np.float32)
    result.setflags(write=False)
    return result


def oriented_box_from_points(
    points: object,
    config: Optional[Mapping[str, object]] = None,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Fit a deterministic gravity-aligned robust PCA box.

    Returns ``(oriented_box, corners, anisotropy)``.  Yaw has period pi and is
    canonicalized to ``[-pi/2, pi/2)``.  Near-isotropic XY support uses yaw
    zero to avoid eigensolver sign/noise instability.
    """

    resolved = resolve_missing_instance_graph_config(config)
    values = _as_array(points, "orientation points")
    if (
        values.ndim != 2
        or values.shape[1:] != (3,)
        or not np.issubdtype(values.dtype, np.number)
        or len(values) == 0
    ):
        raise ValueError("orientation points must have numeric shape [N, 3]")
    values = np.asarray(values, dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("orientation points must be finite")
    xy = values[:, :2]
    xy_center = np.mean(xy, axis=0)
    centered = xy - xy_center[None, :]
    if len(values) >= 2:
        covariance = centered.T @ centered / float(len(values))
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        eigenvalues = np.maximum(eigenvalues, 0.0)
        denominator = float(eigenvalues.sum())
        anisotropy = (
            float((eigenvalues[1] - eigenvalues[0]) / denominator)
            if denominator > 1e-12
            else 0.0
        )
        axis = eigenvectors[:, 1]
    else:
        anisotropy = 0.0
        axis = np.asarray([1.0, 0.0], dtype=np.float64)
    if anisotropy < float(resolved["minimum_orientation_anisotropy"]):
        axis = np.asarray([1.0, 0.0], dtype=np.float64)
    else:
        # PCA axes are sign-ambiguous.  Choose the sign whose first
        # significant coordinate is positive.
        if axis[0] < -1e-12 or (
            abs(float(axis[0])) <= 1e-12 and axis[1] < 0.0
        ):
            axis = -axis
    yaw = float(np.arctan2(axis[1], axis[0]))
    while yaw < -np.pi / 2:
        yaw += np.pi
    while yaw >= np.pi / 2:
        yaw -= np.pi
    cosine = float(np.cos(yaw))
    sine = float(np.sin(yaw))
    basis_xy = np.asarray(
        [[cosine, -sine], [sine, cosine]], dtype=np.float64
    )
    local_xy = xy @ basis_xy
    lower_quantile = float(resolved["aabb_lower_quantile"])
    upper_quantile = float(resolved["aabb_upper_quantile"])
    lower_xy = np.quantile(local_xy, lower_quantile, axis=0)
    upper_xy = np.quantile(local_xy, upper_quantile, axis=0)
    lower_z = float(np.quantile(values[:, 2], lower_quantile))
    upper_z = float(np.quantile(values[:, 2], upper_quantile))
    local_center_xy = (lower_xy + upper_xy) * 0.5
    center_xy = local_center_xy @ basis_xy.T
    center_z = (lower_z + upper_z) * 0.5
    minimum = float(resolved["minimum_dimension"])
    dims = np.maximum(
        np.asarray(
            [
                upper_xy[0] - lower_xy[0],
                upper_xy[1] - lower_xy[1],
                upper_z - lower_z,
            ],
            dtype=np.float64,
        ),
        minimum,
    )
    box = np.asarray(
        [
            center_xy[0],
            center_xy[1],
            center_z,
            dims[0],
            dims[1],
            dims[2],
            yaw,
        ],
        dtype=np.float32,
    )
    box.setflags(write=False)
    return box, oriented_box_corners(box), float(
        np.clip(anisotropy, 0.0, 1.0)
    )


@dataclass(frozen=True)
class _TrackNode:
    component_id: str
    frame_id: Identifier
    provider_call_index: int
    proposal_id: Identifier
    component_index: int
    aabb: np.ndarray
    score: float
    label: Optional[str]
    component_fraction: float
    point_count: int


@dataclass(frozen=True)
class _TrackEdge:
    source_component_id: str
    target_component_id: str
    score: float
    semantic_compatibility: float
    iou_3d: float
    containment: float
    mutual_inside: float
    projection_support: float
    point_projection_support: float
    center_distance: float
    appearance_cosine: Optional[float]


@dataclass(frozen=True)
class _MissingTrack:
    track_id: int
    nodes: Tuple[_TrackNode, ...]
    edges: Tuple[_TrackEdge, ...]
    points_world: np.ndarray
    frame_ids: Tuple[Identifier, ...]
    provider_call_first: int
    provider_call_last: int
    confirmed: bool
    confirmation_frame_id: Optional[Identifier]
    node_count_total: int
    edge_count_total: int
    score_sum: float
    maximum_score: float
    component_fraction_sum: float
    label_votes: Tuple[Tuple[str, int], ...]
    feature_sum: Optional[np.ndarray]
    feature_count: int


def _track_aabb(
    track: _MissingTrack,
    config: Mapping[str, Any],
) -> np.ndarray:
    return _robust_aabb(track.points_world, config)


def _track_label(track: _MissingTrack) -> Optional[str]:
    if not track.label_votes:
        return None
    return sorted(
        track.label_votes, key=lambda item: (-item[1], item[0])
    )[0][0]


def _track_feature(track: _MissingTrack) -> Optional[np.ndarray]:
    if track.feature_sum is None or track.feature_count <= 0:
        return None
    norm = float(np.linalg.norm(track.feature_sum.astype(np.float64)))
    if norm <= 1e-12:
        return None
    result = np.asarray(track.feature_sum / norm, dtype=np.float32)
    result.setflags(write=False)
    return result


def _label_votes_after(
    votes: Tuple[Tuple[str, int], ...],
    label: Optional[str],
) -> Tuple[Tuple[str, int], ...]:
    counts = dict(votes)
    if label is not None:
        normalized = _normalize_label(label)
        counts[normalized] = counts.get(normalized, 0) + 1
    return tuple(sorted(counts.items()))


def _node_from_component(
    component: LiftedMaskComponent,
    provider_call_index: int,
) -> _TrackNode:
    return _TrackNode(
        component_id=component.component_id,
        frame_id=component.frame_id,
        provider_call_index=provider_call_index,
        proposal_id=component.proposal_id,
        component_index=component.component_index,
        aabb=component.aabb,
        score=component.score,
        label=component.label,
        component_fraction=component.component_fraction,
        point_count=len(component.points_world),
    )


def _seed_track(
    track_id: int,
    component: LiftedMaskComponent,
    provider_call_index: int,
    config: Mapping[str, Any],
) -> _MissingTrack:
    node = _node_from_component(component, provider_call_index)
    feature_sum = (
        None
        if component.feature is None
        else np.asarray(component.feature, dtype=np.float64).copy()
    )
    if feature_sum is not None:
        feature_sum.setflags(write=False)
    confirmed = int(config["min_unique_frames"]) <= 1
    return _MissingTrack(
        track_id=track_id,
        nodes=(node,),
        edges=(),
        points_world=_bounded_points(
            component.points_world, int(config["max_points_per_track"])
        ),
        frame_ids=(component.frame_id,),
        provider_call_first=provider_call_index,
        provider_call_last=provider_call_index,
        confirmed=confirmed,
        confirmation_frame_id=(
            component.frame_id if confirmed else None
        ),
        node_count_total=1,
        edge_count_total=0,
        score_sum=component.score,
        maximum_score=component.score,
        component_fraction_sum=component.component_fraction,
        label_votes=_label_votes_after((), component.label),
        feature_sum=feature_sum,
        feature_count=int(component.feature is not None),
    )


def _append_track(
    track: _MissingTrack,
    component: LiftedMaskComponent,
    edge: MissingAssociationAudit,
    provider_call_index: int,
    config: Mapping[str, Any],
) -> _MissingTrack:
    node = _node_from_component(component, provider_call_index)
    nodes = (*track.nodes, node)
    if len(nodes) > int(config["max_nodes_per_track"]):
        nodes = nodes[-int(config["max_nodes_per_track"]):]
    source = max(
        track.nodes,
        key=lambda item: (
            _box_intersection_metrics(item.aabb, component.aabb)[1],
            item.provider_call_index,
            item.component_id,
        ),
    )
    stored_edge = _TrackEdge(
        source_component_id=source.component_id,
        target_component_id=component.component_id,
        score=edge.score,
        semantic_compatibility=edge.semantic_compatibility,
        iou_3d=edge.iou_3d,
        containment=edge.containment,
        mutual_inside=edge.mutual_inside,
        projection_support=edge.projection_support,
        point_projection_support=edge.point_projection_support,
        center_distance=edge.center_distance,
        appearance_cosine=edge.appearance_cosine,
    )
    edges = (*track.edges, stored_edge)
    if len(edges) > int(config["max_edges_per_track"]):
        edges = edges[-int(config["max_edges_per_track"]):]
    combined_points = np.concatenate(
        (track.points_world, component.points_world), axis=0
    )
    points = _bounded_points(
        combined_points, int(config["max_points_per_track"])
    )
    frame_ids = list(track.frame_ids)
    if component.frame_id not in frame_ids:
        frame_ids.append(component.frame_id)
    frame_ids.sort(key=_identifier_key)
    became_confirmed = (
        not track.confirmed
        and len(frame_ids) >= int(config["min_unique_frames"])
    )
    confirmed = track.confirmed or became_confirmed
    feature_sum = track.feature_sum
    feature_count = track.feature_count
    if component.feature is not None:
        addition = component.feature.astype(np.float64)
        feature_sum = (
            addition.copy()
            if feature_sum is None
            else feature_sum.astype(np.float64) + addition
        )
        feature_sum = np.asarray(feature_sum, dtype=np.float64)
        feature_sum.setflags(write=False)
        feature_count += 1
    return _MissingTrack(
        track_id=track.track_id,
        nodes=tuple(nodes),
        edges=tuple(edges),
        points_world=points,
        frame_ids=tuple(frame_ids),
        provider_call_first=track.provider_call_first,
        provider_call_last=provider_call_index,
        confirmed=confirmed,
        confirmation_frame_id=(
            component.frame_id
            if became_confirmed
            else track.confirmation_frame_id
        ),
        node_count_total=track.node_count_total + 1,
        edge_count_total=track.edge_count_total + 1,
        score_sum=track.score_sum + component.score,
        maximum_score=max(track.maximum_score, component.score),
        component_fraction_sum=(
            track.component_fraction_sum + component.component_fraction
        ),
        label_votes=_label_votes_after(track.label_votes, component.label),
        feature_sum=feature_sum,
        feature_count=feature_count,
    )


def _evaluate_association(
    track: _MissingTrack,
    component: LiftedMaskComponent,
    config: Mapping[str, Any],
) -> MissingAssociationAudit:
    track_box = _track_aabb(track, config)
    _, iou, containment = _box_intersection_metrics(
        track_box, component.aabb
    )
    observation_inside = _points_inside_fraction(
        component.points_world, track_box
    )
    track_inside = _points_inside_fraction(
        track.points_world, component.aabb
    )
    mutual_inside = min(observation_inside, track_inside)
    projection_iou = _projected_aabb_mask_iou(track_box, component)
    point_projection = _projected_point_support(
        track.points_world, component, config
    )
    projection_support = max(projection_iou, point_projection)
    center_distance = float(
        np.linalg.norm(track_box[:3] - component.aabb[:3])
    )
    semantic = _semantic_compatibility(
        _track_label(track), component.label, config
    )
    appearance_cosine = _feature_cosine(
        _track_feature(track), component.feature
    )
    geometry_matches = sum(
        (
            iou >= float(config["minimum_iou_3d"]),
            max(containment, mutual_inside)
            >= float(config["minimum_containment"]),
            projection_support
            >= float(config["minimum_projection_support"]),
        )
    )
    geometry_score = float(
        np.mean(
            [
                iou,
                max(containment, mutual_inside),
                projection_support,
            ]
        )
    )
    appearance_score = (
        0.5
        if appearance_cosine is None
        else 0.5 * (appearance_cosine + 1.0)
    )
    score = float(
        np.clip(
            (geometry_score + 0.05 * semantic + 0.025 * appearance_score)
            / 1.075,
            0.0,
            1.0,
        )
    )
    if semantic < float(config["minimum_semantic_score"]):
        accepted = False
        reason = "semantic"
    elif center_distance > float(config["maximum_center_distance"]):
        accepted = False
        reason = "center_distance"
    elif geometry_matches < int(config["minimum_geometry_matches"]):
        accepted = False
        reason = "geometry"
    else:
        accepted = True
        reason = "accepted"
    return MissingAssociationAudit(
        track_id=track.track_id,
        component_id=component.component_id,
        accepted=accepted,
        reason=reason,
        score=score,
        semantic_compatibility=semantic,
        iou_3d=iou,
        containment=containment,
        mutual_inside=mutual_inside,
        projection_support=projection_support,
        point_projection_support=point_projection,
        center_distance=center_distance,
        geometry_matches=geometry_matches,
        appearance_cosine=appearance_cosine,
    )


def _component_sort_key(
    component: LiftedMaskComponent,
) -> Tuple[Any, ...]:
    return (
        _identifier_key(component.frame_id),
        component.provider,
        _identifier_key(component.proposal_id),
        component.component_index,
        component.component_id,
    )


def _same_view_deduplicate(
    components: Sequence[LiftedMaskComponent],
    provider_call_index: int,
    config: Mapping[str, Any],
) -> Tuple[List[LiftedMaskComponent], List[MissingObservationAudit]]:
    ranked = sorted(
        components,
        key=lambda item: (
            -item.score,
            -len(item.points_world),
            _component_sort_key(item),
        ),
    )
    retained: List[LiftedMaskComponent] = []
    audits: List[MissingObservationAudit] = []
    for component in ranked:
        duplicate_of = None
        for previous in retained:
            if previous.frame_id != component.frame_id:
                continue
            semantic = _semantic_compatibility(
                previous.label, component.label, config
            )
            if semantic < float(config["minimum_semantic_score"]):
                continue
            _, iou, containment = _box_intersection_metrics(
                component.aabb, previous.aabb
            )
            if (
                iou >= float(config["same_view_duplicate_iou"])
                or containment
                >= float(config["same_view_duplicate_containment"])
            ):
                duplicate_of = previous.component_id
                break
        if duplicate_of is None:
            retained.append(component)
        else:
            audits.append(
                MissingObservationAudit(
                    provider_call_index=provider_call_index,
                    frame_id=component.frame_id,
                    proposal_id=component.proposal_id,
                    component_index=component.component_index,
                    component_id=component.component_id,
                    track_id=None,
                    accepted=False,
                    reason="same_view_duplicate",
                    pixel_count=component.pixel_count,
                    point_count=len(component.points_world),
                    component_fraction=component.component_fraction,
                    duplicate_of=duplicate_of,
                )
            )
    retained.sort(key=_component_sort_key)
    return retained, audits


def _mean_edge(track: _MissingTrack, name: str) -> float:
    if not track.edges:
        return 0.0
    return float(
        np.mean([float(getattr(edge, name)) for edge in track.edges])
    )


def _semantic_agreement(track: _MissingTrack) -> float:
    if not track.label_votes:
        return 0.0
    counts = [count for _, count in track.label_votes]
    return float(max(counts)) / float(sum(counts))


def _candidate_feature_vector(
    *,
    track: _MissingTrack,
    lifecycle_state: str,
    current_provider_call: int,
    features: MissingCandidateFeatures,
    config: Mapping[str, Any],
) -> np.ndarray:
    """Build the fixed, finite gate feature contract for one track."""

    view_denominator = max(int(config["min_unique_frames"]) + 2, 1)
    ttl_denominator = max(
        int(config["track_ttl_provider_calls"]) + 1, 1
    )
    age = max(current_provider_call - track.provider_call_last, 0)
    span = max(
        track.provider_call_last - track.provider_call_first, 0
    )
    values = np.asarray(
        [
            float(track.confirmed),
            min(len(track.frame_ids) / float(view_denominator), 1.0),
            min(
                track.node_count_total
                / float(int(config["max_nodes_per_track"])),
                1.0,
            ),
            min(
                track.edge_count_total
                / float(int(config["max_edges_per_track"])),
                1.0,
            ),
            features.mean_detector_score,
            features.mean_edge_score,
            features.mean_iou_3d,
            features.mean_containment,
            features.mean_projection_support,
            features.semantic_agreement,
            features.mean_component_fraction,
            float(lifecycle_state == "active"),
            min(span / float(ttl_denominator), 1.0),
            float(np.clip(1.0 - age / float(ttl_denominator), 0.0, 1.0)),
            features.maximum_global_iou,
            features.maximum_global_containment,
            features.maximum_candidate_iou,
            features.maximum_candidate_containment,
            min(
                len(track.points_world)
                / float(int(config["max_points_per_track"])),
                1.0,
            ),
            features.orientation_anisotropy,
        ],
        dtype=np.float32,
    )
    if values.shape != (len(MISSING_GRAPH_FEATURE_NAMES),):
        raise RuntimeError("missing-graph feature schema length drifted")
    if not np.isfinite(values).all():
        raise RuntimeError("missing-graph feature vector is not finite")
    values = np.clip(values, 0.0, 1.0)
    values.setflags(write=False)
    return values


def _candidate_base(
    track: _MissingTrack,
    lifecycle_state: str,
    maximum_global_iou: float,
    maximum_global_containment: float,
    current_provider_call: int,
    config: Mapping[str, Any],
) -> OrientedMissingCandidate:
    oriented_box, corners, anisotropy = oriented_box_from_points(
        track.points_world, config
    )
    aabb = _track_aabb(track, config)
    mean_detector = track.score_sum / float(track.node_count_total)
    mean_edge_score = _mean_edge(track, "score")
    score = float(
        np.clip(0.75 * mean_detector + 0.25 * mean_edge_score, 0.0, 1.0)
    )
    features = MissingCandidateFeatures(
        unique_views=len(track.frame_ids),
        mean_detector_score=mean_detector,
        maximum_detector_score=track.maximum_score,
        mean_edge_score=mean_edge_score,
        mean_iou_3d=_mean_edge(track, "iou_3d"),
        mean_containment=_mean_edge(track, "containment"),
        mean_projection_support=_mean_edge(
            track, "projection_support"
        ),
        mean_center_distance=_mean_edge(track, "center_distance"),
        semantic_agreement=_semantic_agreement(track),
        mean_component_fraction=(
            track.component_fraction_sum / float(track.node_count_total)
        ),
        maximum_global_iou=maximum_global_iou,
        maximum_global_containment=maximum_global_containment,
        maximum_candidate_iou=0.0,
        maximum_candidate_containment=0.0,
        orientation_anisotropy=anisotropy,
    )
    feature_vector = _candidate_feature_vector(
        track=track,
        lifecycle_state=lifecycle_state,
        current_provider_call=current_provider_call,
        features=features,
        config=config,
    )
    return OrientedMissingCandidate(
        track_id=track.track_id,
        label=_track_label(track),
        oriented_box=oriented_box,
        corners=corners,
        aabb=aabb,
        score=score,
        frame_ids=track.frame_ids,
        provider_call_first=track.provider_call_first,
        provider_call_last=track.provider_call_last,
        node_count=track.node_count_total,
        edge_count=track.edge_count_total,
        point_count=len(track.points_world),
        lifecycle_state=lifecycle_state,
        reason="confirmed_multiview",
        features=features,
        feature_vector=feature_vector,
        appearance_feature=_track_feature(track),
    )


class MissingInstanceGraphObserver:
    """Incremental, provider-call-clocked missing-instance observer."""

    def __init__(
        self,
        config: Optional[Mapping[str, object]] = None,
    ) -> None:
        self.config = resolve_missing_instance_graph_config(config)
        self._tracks: Dict[int, _MissingTrack] = {}
        self._archived_tracks: Dict[int, _MissingTrack] = {}
        self._next_track_id = 0
        self._last_provider_call_index: Optional[int] = None

    @property
    def last_provider_call_index(self) -> Optional[int]:
        return self._last_provider_call_index

    @property
    def active_track_ids(self) -> Tuple[int, ...]:
        return tuple(sorted(self._tracks))

    @property
    def archived_track_ids(self) -> Tuple[int, ...]:
        return tuple(sorted(self._archived_tracks))

    def reset(self) -> None:
        self._tracks.clear()
        self._archived_tracks.clear()
        self._next_track_id = 0
        self._last_provider_call_index = None

    def _provider_call_index(
        self, value: Optional[object]
    ) -> int:
        if value is None:
            return (
                0
                if self._last_provider_call_index is None
                else self._last_provider_call_index + 1
            )
        result = _strict_int("provider_call_index", value, 0)
        if (
            self._last_provider_call_index is not None
            and result <= self._last_provider_call_index
        ):
            raise ValueError(
                "provider_call_index must increase strictly between updates"
            )
        return result

    def _expire(
        self,
        tracks: Dict[int, _MissingTrack],
        archived_tracks: Dict[int, _MissingTrack],
        provider_call_index: int,
    ) -> Tuple[Tuple[int, ...], Tuple[int, ...], Tuple[int, ...]]:
        ttl = int(self.config["track_ttl_provider_calls"])
        expired = tuple(
            track_id
            for track_id, track in sorted(tracks.items())
            if provider_call_index - track.provider_call_last > ttl
        )
        archived: List[int] = []
        discarded: List[int] = []
        for track_id in expired:
            track = tracks.pop(track_id)
            if bool(self.config["archive_confirmed"]) and track.confirmed:
                archived_tracks[track_id] = track
                archived.append(track_id)
            else:
                discarded.append(track_id)
        return expired, tuple(archived), tuple(discarded)

    @staticmethod
    def _best_effort_identity(
        value: object, name: str
    ) -> Optional[Identifier]:
        try:
            candidate = (
                value.get(name)
                if isinstance(value, Mapping)
                else getattr(value, name, None)
            )
            return None if candidate is None else _identifier(name, candidate)
        except Exception:
            return None

    def _materialize(
        self,
        tracks: Mapping[int, _MissingTrack],
        archived_tracks: Mapping[int, _MissingTrack],
        global_boxes: np.ndarray,
    ) -> Tuple[
        Tuple[OrientedMissingCandidate, ...],
        Tuple[MissingCandidateDecision, ...],
    ]:
        rows: List[Tuple[_MissingTrack, str]] = [
            (track, "active")
            for _, track in sorted(tracks.items())
        ]
        rows.extend(
            (track, "archived")
            for _, track in sorted(archived_tracks.items())
        )
        decisions: List[MissingCandidateDecision] = []
        eligible: List[OrientedMissingCandidate] = []
        for track, state in rows:
            aabb = _track_aabb(track, self.config)
            global_iou, global_containment = _maximum_box_overlap(
                aabb, global_boxes
            )
            if not track.confirmed or (
                len(track.frame_ids) < int(self.config["min_unique_frames"])
            ):
                decisions.append(
                    MissingCandidateDecision(
                        track_id=track.track_id,
                        lifecycle_state=state,
                        confirmed=False,
                        unique_views=len(track.frame_ids),
                        accepted=False,
                        reason="insufficient_unique_views",
                        duplicate_of_track_id=None,
                        maximum_global_iou=global_iou,
                        maximum_global_containment=global_containment,
                        maximum_candidate_iou=0.0,
                        maximum_candidate_containment=0.0,
                    )
                )
                continue
            if (
                global_iou >= float(self.config["global_reject_iou"])
                or global_containment
                >= float(self.config["global_reject_containment"])
            ):
                decisions.append(
                    MissingCandidateDecision(
                        track_id=track.track_id,
                        lifecycle_state=state,
                        confirmed=True,
                        unique_views=len(track.frame_ids),
                        accepted=False,
                        reason="global_overlap",
                        duplicate_of_track_id=None,
                        maximum_global_iou=global_iou,
                        maximum_global_containment=global_containment,
                        maximum_candidate_iou=0.0,
                        maximum_candidate_containment=0.0,
                    )
                )
                continue
            eligible.append(
                _candidate_base(
                    track,
                    state,
                    global_iou,
                    global_containment,
                    (
                        track.provider_call_last
                        if self._last_provider_call_index is None
                        else self._last_provider_call_index
                    ),
                    self.config,
                )
            )

        eligible.sort(
            key=lambda item: (
                -item.features.unique_views,
                -item.score,
                -item.node_count,
                -item.point_count,
                item.track_id,
            )
        )
        accepted: List[OrientedMissingCandidate] = []
        for candidate in eligible:
            duplicate_of = None
            maximum_iou = 0.0
            maximum_containment = 0.0
            for previous in accepted:
                semantic = _semantic_compatibility(
                    candidate.label, previous.label, self.config
                )
                if semantic < float(
                    self.config["minimum_semantic_score"]
                ):
                    continue
                _, iou, containment = _box_intersection_metrics(
                    candidate.aabb, previous.aabb
                )
                maximum_iou = max(maximum_iou, iou)
                maximum_containment = max(
                    maximum_containment, containment
                )
                if (
                    iou
                    >= float(self.config["candidate_duplicate_iou"])
                    or containment
                    >= float(
                        self.config["candidate_duplicate_containment"]
                    )
                ):
                    duplicate_of = previous.track_id
                    break
            if duplicate_of is not None:
                decisions.append(
                    MissingCandidateDecision(
                        track_id=candidate.track_id,
                        lifecycle_state=candidate.lifecycle_state,
                        confirmed=True,
                        unique_views=candidate.features.unique_views,
                        accepted=False,
                        reason="duplicate_candidate",
                        duplicate_of_track_id=duplicate_of,
                        maximum_global_iou=(
                            candidate.features.maximum_global_iou
                        ),
                        maximum_global_containment=(
                            candidate.features.maximum_global_containment
                        ),
                        maximum_candidate_iou=maximum_iou,
                        maximum_candidate_containment=maximum_containment,
                    )
                )
                continue
            features = replace(
                candidate.features,
                maximum_candidate_iou=maximum_iou,
                maximum_candidate_containment=maximum_containment,
            )
            source_track = (
                tracks.get(candidate.track_id)
                or archived_tracks[candidate.track_id]
            )
            feature_vector = _candidate_feature_vector(
                track=source_track,
                lifecycle_state=candidate.lifecycle_state,
                current_provider_call=(
                    source_track.provider_call_last
                    if self._last_provider_call_index is None
                    else self._last_provider_call_index
                ),
                features=features,
                config=self.config,
            )
            candidate = replace(
                candidate,
                features=features,
                feature_vector=feature_vector,
            )
            accepted.append(candidate)
            decisions.append(
                MissingCandidateDecision(
                    track_id=candidate.track_id,
                    lifecycle_state=candidate.lifecycle_state,
                    confirmed=True,
                    unique_views=candidate.features.unique_views,
                    accepted=True,
                    reason="confirmed_multiview",
                    duplicate_of_track_id=None,
                    maximum_global_iou=(
                        candidate.features.maximum_global_iou
                    ),
                    maximum_global_containment=(
                        candidate.features.maximum_global_containment
                    ),
                    maximum_candidate_iou=maximum_iou,
                    maximum_candidate_containment=maximum_containment,
                )
            )
        # Candidate order is the evidence rank above.  Decision order is by
        # stable track id so auditing does not depend on rejection stage.
        decisions.sort(key=lambda item: item.track_id)
        return tuple(accepted), tuple(decisions)

    def candidates(
        self,
        global_boxes: object = None,
    ) -> Tuple[OrientedMissingCandidate, ...]:
        """Return current observer candidates without mutating lifecycle."""

        if not bool(self.config["enabled"]):
            return ()
        try:
            boxes = _coerce_global_boxes(global_boxes)
            candidates, _ = self._materialize(
                self._tracks, self._archived_tracks, boxes
            )
            return candidates
        except Exception:
            if bool(self.config["fail_open"]):
                return ()
            raise

    def _update_impl(
        self,
        observations: Iterable[object],
        global_boxes: object = None,
        *,
        provider_call_index: Optional[int] = None,
    ) -> MissingInstanceGraphUpdate:
        """Process one proposal-provider call transactionally.

        ``provider_call_index`` defaults to the next integer.  Gaps are legal
        and count toward TTL.  The supplied observations and global boxes are
        never mutated.
        """

        call = self._provider_call_index(provider_call_index)
        if not bool(self.config["enabled"]):
            return MissingInstanceGraphUpdate(
                provider_call_index=call,
                candidates=(),
                decisions=(),
                observations=(),
                associations=(),
                expired_track_ids=(),
                archived_track_ids=(),
                discarded_track_ids=(),
                disabled=True,
            )
        if isinstance(observations, (str, bytes, Mapping)):
            raise ValueError("observations must be an iterable of observations")
        try:
            observation_values = list(observations)
        except Exception as error:
            raise ValueError("observations must be iterable") from error

        # Track values are immutable, so shallow dictionary copies provide a
        # transactional staging area without copying bounded point arrays.
        tracks = dict(self._tracks)
        archived_tracks = dict(self._archived_tracks)
        next_track_id = self._next_track_id
        observation_audits: List[MissingObservationAudit] = []
        association_audits: List[MissingAssociationAudit] = []
        errors: List[str] = []
        try:
            boxes = _coerce_global_boxes(global_boxes)
        except Exception as error:
            if not bool(self.config["fail_open"]):
                raise
            # A malformed global guard must never be treated as "no globals".
            # Advance the provider clock and TTL, but emit nothing.
            expired, archived, discarded = self._expire(
                tracks, archived_tracks, call
            )
            self._tracks = tracks
            self._archived_tracks = archived_tracks
            self._last_provider_call_index = call
            message = f"{type(error).__name__}: {error}"
            return MissingInstanceGraphUpdate(
                provider_call_index=call,
                candidates=(),
                decisions=(),
                observations=(),
                associations=(),
                expired_track_ids=expired,
                archived_track_ids=archived,
                discarded_track_ids=discarded,
                errors=(message,),
            )

        expired, archived, discarded = self._expire(
            tracks, archived_tracks, call
        )
        normalized: List[MaskDepthProposalObservation] = []
        prelifted: List[LiftedMaskComponent] = []
        for value_index, value in enumerate(observation_values):
            try:
                if isinstance(value, LiftedMaskComponent):
                    prelifted.append(value)
                else:
                    normalized.append(coerce_mask_depth_observation(value))
            except Exception as error:
                if not bool(self.config["fail_open"]):
                    raise
                detail = f"{type(error).__name__}: {error}"
                errors.append(f"observation[{value_index}]: {detail}")
                observation_audits.append(
                    MissingObservationAudit(
                        provider_call_index=call,
                        frame_id=self._best_effort_identity(
                            value, "frame_id"
                        ),
                        proposal_id=self._best_effort_identity(
                            value, "proposal_id"
                        ),
                        component_index=None,
                        component_id=None,
                        track_id=None,
                        accepted=False,
                        reason="invalid_observation",
                        detail=detail,
                    )
                )
        normalized.sort(key=_observation_sort_key)

        components: List[LiftedMaskComponent] = sorted(
            prelifted, key=_component_sort_key
        )
        for observation in normalized:
            try:
                lifted, lifting_rejections = _lift_components_detailed(
                    observation, self.config
                )
            except Exception as error:
                if not bool(self.config["fail_open"]):
                    raise
                detail = f"{type(error).__name__}: {error}"
                errors.append(
                    f"proposal[{observation.proposal_id}]: {detail}"
                )
                observation_audits.append(
                    MissingObservationAudit(
                        provider_call_index=call,
                        frame_id=observation.frame_id,
                        proposal_id=observation.proposal_id,
                        component_index=None,
                        component_id=None,
                        track_id=None,
                        accepted=False,
                        reason="lifting_error",
                        detail=detail,
                    )
                )
                continue
            components.extend(lifted)
            for reason, pixel_count, point_count in lifting_rejections:
                observation_audits.append(
                    MissingObservationAudit(
                        provider_call_index=call,
                        frame_id=observation.frame_id,
                        proposal_id=observation.proposal_id,
                        component_index=None,
                        component_id=None,
                        track_id=None,
                        accepted=False,
                        reason=reason,
                        pixel_count=pixel_count,
                        point_count=point_count,
                    )
                )

        components, duplicate_audits = _same_view_deduplicate(
            components, call, self.config
        )
        observation_audits.extend(duplicate_audits)
        unique_components: List[LiftedMaskComponent] = []
        seen_component_ids = set()
        for component in components:
            if component.component_id not in seen_component_ids:
                seen_component_ids.add(component.component_id)
                unique_components.append(component)
                continue
            observation_audits.append(
                MissingObservationAudit(
                    provider_call_index=call,
                    frame_id=component.frame_id,
                    proposal_id=component.proposal_id,
                    component_index=component.component_index,
                    component_id=component.component_id,
                    track_id=None,
                    accepted=False,
                    reason="duplicate_component_id",
                    pixel_count=component.pixel_count,
                    point_count=len(component.points_world),
                    component_fraction=component.component_fraction,
                    duplicate_of=component.component_id,
                )
            )
        components = unique_components
        eligible_components: List[LiftedMaskComponent] = []
        for component in components:
            global_iou, global_containment = _maximum_box_overlap(
                component.aabb, boxes
            )
            if (
                global_iou >= float(self.config["global_reject_iou"])
                or global_containment
                >= float(self.config["global_reject_containment"])
            ):
                observation_audits.append(
                    MissingObservationAudit(
                        provider_call_index=call,
                        frame_id=component.frame_id,
                        proposal_id=component.proposal_id,
                        component_index=component.component_index,
                        component_id=component.component_id,
                        track_id=None,
                        accepted=False,
                        reason="global_overlap",
                        pixel_count=component.pixel_count,
                        point_count=len(component.points_world),
                        component_fraction=component.component_fraction,
                        maximum_global_iou=global_iou,
                        maximum_global_containment=global_containment,
                    )
                )
            else:
                eligible_components.append(component)

        # Evaluate every active cross-view pair.  Same-frame reuse is
        # forbidden, so repeated proposals from one view cannot confirm.
        accepted_edges: List[
            Tuple[MissingAssociationAudit, LiftedMaskComponent]
        ] = []
        for track_id, track in sorted(tracks.items()):
            for component in eligible_components:
                if component.frame_id in track.frame_ids:
                    continue
                edge = _evaluate_association(
                    track, component, self.config
                )
                association_audits.append(edge)
                if edge.accepted:
                    accepted_edges.append((edge, component))
        accepted_edges.sort(
            key=lambda item: (
                -item[0].score,
                -item[0].iou_3d,
                -item[0].projection_support,
                item[0].center_distance,
                item[0].track_id,
                _component_sort_key(item[1]),
            )
        )
        assignments: Dict[str, int] = {}
        used_tracks = set()
        selected_edges: Dict[str, MissingAssociationAudit] = {}
        component_by_id = {
            component.component_id: component
            for component in eligible_components
        }
        for edge, component in accepted_edges:
            if (
                edge.track_id in used_tracks
                or component.component_id in assignments
            ):
                continue
            assignments[component.component_id] = edge.track_id
            selected_edges[component.component_id] = edge
            used_tracks.add(edge.track_id)

        for component in sorted(
            eligible_components, key=_component_sort_key
        ):
            if component.component_id in assignments:
                track_id = assignments[component.component_id]
                tracks[track_id] = _append_track(
                    tracks[track_id],
                    component,
                    selected_edges[component.component_id],
                    call,
                    self.config,
                )
                reason = "associated"
            else:
                track_id = next_track_id
                next_track_id += 1
                tracks[track_id] = _seed_track(
                    track_id, component, call, self.config
                )
                reason = "seeded"
            observation_audits.append(
                MissingObservationAudit(
                    provider_call_index=call,
                    frame_id=component.frame_id,
                    proposal_id=component.proposal_id,
                    component_index=component.component_index,
                    component_id=component.component_id,
                    track_id=track_id,
                    accepted=True,
                    reason=reason,
                    pixel_count=component.pixel_count,
                    point_count=len(component.points_world),
                    component_fraction=component.component_fraction,
                )
            )

        # All staged state is internally owned and all track values are
        # immutable.  Commit before observer materialization: a diagnostic
        # formatting failure must not lose valid graph evidence.
        self._tracks = tracks
        self._archived_tracks = archived_tracks
        self._next_track_id = next_track_id
        self._last_provider_call_index = call
        try:
            candidates, decisions = self._materialize(
                tracks, archived_tracks, boxes
            )
        except Exception as error:
            if not bool(self.config["fail_open"]):
                raise
            errors.append(f"materialize: {type(error).__name__}: {error}")
            candidates = ()
            decisions = ()

        observation_audits.sort(
            key=lambda item: (
                (
                    (2, "")
                    if item.frame_id is None
                    else _identifier_key(item.frame_id)
                ),
                (
                    (2, "")
                    if item.proposal_id is None
                    else _identifier_key(item.proposal_id)
                ),
                (
                    -1
                    if item.component_index is None
                    else item.component_index
                ),
                item.reason,
                "" if item.component_id is None else item.component_id,
            )
        )
        association_audits.sort(
            key=lambda item: (item.track_id, item.component_id)
        )
        # Defensive invariant: selected IDs must refer to components from this
        # call.  Keeping this check near the transaction boundary makes future
        # association changes fail open rather than corrupting graph state.
        if not set(assignments).issubset(component_by_id):
            raise RuntimeError("association selected an unknown component")
        return MissingInstanceGraphUpdate(
            provider_call_index=call,
            candidates=candidates,
            decisions=decisions,
            observations=tuple(observation_audits),
            associations=tuple(association_audits),
            expired_track_ids=expired,
            archived_track_ids=archived,
            discarded_track_ids=discarded,
            errors=tuple(errors),
        )

    def update(
        self,
        observations: Iterable[object],
        global_boxes: object = None,
        *,
        provider_call_index: Optional[int] = None,
    ) -> MissingInstanceGraphUpdate:
        """Process one provider call with an observer-wide fail-open boundary."""

        # Lifecycle contract errors are caller bugs rather than observer
        # failures and remain fail-fast.
        call = self._provider_call_index(provider_call_index)
        if not bool(self.config["enabled"]):
            return self._update_impl(
                observations,
                global_boxes,
                provider_call_index=call,
            )
        previous_tracks = dict(self._tracks)
        previous_archived = dict(self._archived_tracks)
        previous_next_track_id = self._next_track_id
        previous_provider_call = self._last_provider_call_index
        try:
            return self._update_impl(
                observations,
                global_boxes,
                provider_call_index=call,
            )
        except Exception as error:
            self._tracks = previous_tracks
            self._archived_tracks = previous_archived
            self._next_track_id = previous_next_track_id
            self._last_provider_call_index = previous_provider_call
            if not bool(self.config["fail_open"]):
                raise
            # The provider call still happened, so its empty evidence advances
            # TTL.  No partially updated graph state is retained.
            staged_tracks = dict(previous_tracks)
            staged_archived = dict(previous_archived)
            expired, archived, discarded = self._expire(
                staged_tracks, staged_archived, call
            )
            self._tracks = staged_tracks
            self._archived_tracks = staged_archived
            self._last_provider_call_index = call
            return MissingInstanceGraphUpdate(
                provider_call_index=call,
                candidates=(),
                decisions=(),
                observations=(),
                associations=(),
                expired_track_ids=expired,
                archived_track_ids=archived,
                discarded_track_ids=discarded,
                errors=(f"{type(error).__name__}: {error}",),
            )

    def update_lifted(
        self,
        components: Iterable[LiftedMaskComponent],
        global_boxes: object = None,
        *,
        provider_call_index: Optional[int] = None,
    ) -> MissingInstanceGraphUpdate:
        """Feed already-lifted unmatched components through the same graph.

        Use :func:`coerce_lifted_mask_component` to adapt the existing online
        ``LiftedProposal`` layout.  Keeping adaptation separate makes the
        provider-call transaction explicit and avoids silently assigning
        unstable proposal identifiers.
        """

        try:
            values = tuple(components)
        except Exception as error:
            raise ValueError("components must be iterable") from error
        if not all(
            isinstance(component, LiftedMaskComponent)
            for component in values
        ):
            raise ValueError(
                "update_lifted requires LiftedMaskComponent entries; use "
                "coerce_lifted_mask_component first"
            )
        return self.update(
            values,
            global_boxes,
            provider_call_index=provider_call_index,
        )

    def advance_provider_call(
        self,
        global_boxes: object = None,
        *,
        provider_call_index: Optional[int] = None,
    ) -> MissingInstanceGraphUpdate:
        """Advance TTL for an empty/failed proposal-provider call."""

        return self.update(
            (),
            global_boxes,
            provider_call_index=provider_call_index,
        )

    # Explicit names make call sites read naturally without creating another
    # lifecycle implementation.
    process_provider_call = update
    process_view = update

    def summary(self) -> Dict[str, Any]:
        active_confirmed = sum(
            int(track.confirmed) for track in self._tracks.values()
        )
        return {
            "schema": MISSING_INSTANCE_GRAPH_SCHEMA,
            "enabled": bool(self.config["enabled"]),
            "observer_only": True,
            "provider_call_index": self._last_provider_call_index,
            "active_tracks": len(self._tracks),
            "active_confirmed_tracks": active_confirmed,
            "archived_tracks": len(self._archived_tracks),
            "next_track_id": self._next_track_id,
        }


# The route name from the design is also the concise constructor name.
MissingInstanceGraph = MissingInstanceGraphObserver
MissingInstanceGraphState = MissingInstanceGraphObserver


__all__ = [
    "DEFAULT_MISSING_INSTANCE_GRAPH_CONFIG",
    "MISSING_GRAPH_FEATURE_NAMES",
    "MISSING_INSTANCE_GRAPH_SCHEMA",
    "LiftedMaskComponent",
    "MaskDepthObservation",
    "MaskDepthProposalObservation",
    "MissingAssociationAudit",
    "MissingCandidateDecision",
    "MissingCandidateFeatures",
    "MissingCandidateRecord",
    "MissingInstanceGraph",
    "MissingInstanceGraphObserver",
    "MissingInstanceGraphState",
    "MissingInstanceGraphUpdate",
    "MissingMaskDepthObservation",
    "MissingObservationAudit",
    "OrientedMissingCandidate",
    "coerce_lifted_mask_component",
    "coerce_mask_depth_observation",
    "lift_mask_depth_components",
    "oriented_box_corners",
    "oriented_box_from_points",
    "resolve_missing_instance_graph_config",
]
