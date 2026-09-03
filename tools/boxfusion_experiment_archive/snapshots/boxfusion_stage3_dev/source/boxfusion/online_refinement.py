"""Online orchestration for supplemental masks, RGB-D memory, and box refinement.

The controller in this module is intentionally external to BoxFusion's own
association state.  It observes the fused objects at each keyframe and changes
only the final exported detections.  This makes the feature opt-in and keeps a
disabled run on the exact legacy path.

Runtime inputs never include ground truth.  The only metric geometry source is
the sensor depth supplied by the caller.
"""

from __future__ import annotations

import json
import os
import tempfile
import time
from collections import Counter
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from boxfusion.box_refiner import (
    BoxRefinerConfig,
    apply_box_residual_numpy,
    build_box_refiner,
)
from boxfusion.object_memory import (
    CandidateTrackManager,
    ObjectGeometryMemory,
    ObjectObservation,
    aabb_corners,
    aabb_iou,
    deterministic_bounded_sample,
    extract_masked_world_points,
    points_inside_aabb,
    points_inside_aabb_fraction,
    project_aabb_to_image,
    projected_aabb_mask_iou,
    resolve_object_memory_config,
    robust_quantile_aabb,
)
from boxfusion.quality_score import (
    QUALITY_FEATURE_DIM,
    QUALITY_FEATURE_NAMES,
    load_quality_scorer,
    make_quality_scorer,
    quality_feature_vector,
    soft_nms_aabb_3d,
)
from boxfusion.supplemental_proposals import (
    ProposalProvider,
    SupplementalProposal,
    build_provider,
)


DEFAULT_ONLINE_REFINEMENT_CONFIG: Dict[str, Any] = {
    "enabled": False,
    "scannet_axis_aligned_only": True,
    "inference_every_keyframes": 1,
    "candidate_lifecycle": {
        # Legacy-compatible default.  Stage-3 configs explicitly switch this
        # to ``provider_call`` so TTL counts missed proposal-provider updates.
        "ttl_clock": "keyframe",
        "archive_confirmed": False,
    },
    "supplemental_proposals": {},
    "appearance_memory": {
        "enabled": True,
        "masked_crop": True,
    },
    "object_memory": {},
    "matching": {
        "global_match_iou": 0.05,
        "global_match_2d_iou": 0.20,
        "max_center_distance": 0.75,
        "crop_to_global_expansion": 1.50,
        "rekey_iou": 0.50,
        "absorb_supplemental_iou": 0.35,
    },
    "refit": {
        "enabled": True,
        "min_views": 2,
        "min_points": 192,
        "blend": 0.70,
        "extent_padding": 0.02,
        "max_center_shift_ratio": 0.60,
        "min_extent_ratio": 0.35,
        "max_extent_ratio": 2.50,
        "min_original_point_support": 0.20,
        "min_reprojection_iou": 0.15,
        "min_reprojection_improvement": -0.02,
    },
    "box_refiner": {
        "enabled": False,
        "checkpoint": None,
        "device": None,
        "point_count": 512,
        "min_quality": 0.20,
        "architecture": {},
    },
    "quality": {
        "enabled": True,
        "mode": "heuristic",
        "checkpoint": None,
        "blend_with_detector": 0.60,
        "preserve_original_floor": False,
        "apply_to_unobserved": False,
        "support_reference_points": 512,
        "target_views": 3,
        "max_view_records": 5,
        "soft_nms": {
            "enabled": True,
            "method": "gaussian",
            "iou_threshold": 0.30,
            "sigma": 0.50,
            "score_threshold": 0.05,
            "max_detections": None,
        },
    },
    "supplemental_output": {
        "enabled": True,
        "min_confirmations": 2,
        "min_score": 0.15,
        # Backward-compatible default.  Conservative B1 experiments require
        # the final 3D AABB to agree with its stored multi-view 2D boxes.
        "min_projection_iou": 0.0,
        "drop_if_global_iou": 0.70,
        "drop_if_supplemental_iou": 0.70,
    },
    "output_filter": {
        "minimum_extent": 0.0,
    },
    "diagnostics": {
        "enabled": False,
        "dump_track_memory": False,
        "root": None,
        "point_count": 512,
    },
}


def _deep_copy(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _deep_copy(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_deep_copy(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_deep_copy(item) for item in value)
    return value


def _merge_known(
    defaults: Mapping[str, Any],
    updates: Optional[Mapping[str, Any]],
    *,
    prefix: str,
) -> Dict[str, Any]:
    if updates is None:
        updates = {}
    if not isinstance(updates, Mapping):
        raise TypeError(f"{prefix} must be a mapping")
    # Empty mappings are deliberate extension points whose keys are validated
    # by the owning component (proposal provider, object memory, or model).
    if not defaults:
        return _deep_copy(updates)
    unknown = sorted(set(updates) - set(defaults))
    if unknown:
        raise ValueError(
            f"Unknown {prefix} key(s): " + ", ".join(str(key) for key in unknown)
        )
    output = _deep_copy(defaults)
    for key, value in updates.items():
        if isinstance(defaults[key], Mapping):
            output[key] = _merge_known(
                defaults[key],
                value,
                prefix=f"{prefix}.{key}",
            )
        else:
            output[key] = value
    return output


def _finite_float(
    config: Mapping[str, Any],
    key: str,
    *,
    lower: Optional[float] = None,
    upper: Optional[float] = None,
    strict_lower: bool = False,
) -> float:
    value = config[key]
    if isinstance(value, (bool, np.bool_)) or not np.isscalar(value):
        raise ValueError(f"{key} must be a finite scalar")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{key} must be finite")
    if lower is not None:
        invalid = result <= lower if strict_lower else result < lower
        if invalid:
            relation = "greater than" if strict_lower else "at least"
            raise ValueError(f"{key} must be {relation} {lower}")
    if upper is not None and result > upper:
        raise ValueError(f"{key} must be at most {upper}")
    return result


def _positive_int(config: Mapping[str, Any], key: str, minimum: int = 1) -> int:
    value = config[key]
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise ValueError(f"{key} must be an integer")
    value = int(value)
    if value < minimum:
        raise ValueError(f"{key} must be at least {minimum}")
    return value


def resolve_online_refinement_config(
    cfg: Optional[Mapping[str, Any]],
) -> Dict[str, Any]:
    """Resolve and validate the opt-in controller configuration.

    ``cfg`` may be the full BoxFusion configuration or the
    ``online_refinement`` subsection.  A disabled subsection is returned
    without validating or touching any optional provider/model settings.
    """

    if cfg is None:
        raw: Mapping[str, Any] = {}
    elif not isinstance(cfg, Mapping):
        raise TypeError("online refinement config must be a mapping")
    elif "online_refinement" in cfg:
        nested = cfg.get("online_refinement")
        if nested is None:
            raw = {}
        elif not isinstance(nested, Mapping):
            raise TypeError("online_refinement must be a mapping")
        else:
            raw = nested
    else:
        raw = cfg

    enabled = raw.get("enabled", False)
    if not isinstance(enabled, (bool, np.bool_)):
        raise ValueError("online_refinement.enabled must be Boolean")
    if not bool(enabled):
        disabled = _deep_copy(DEFAULT_ONLINE_REFINEMENT_CONFIG)
        disabled["enabled"] = False
        return disabled

    resolved = _merge_known(
        DEFAULT_ONLINE_REFINEMENT_CONFIG,
        raw,
        prefix="online_refinement",
    )
    resolved["enabled"] = True
    if not isinstance(
        resolved["scannet_axis_aligned_only"], (bool, np.bool_)
    ):
        raise ValueError("scannet_axis_aligned_only must be Boolean")
    resolved["scannet_axis_aligned_only"] = bool(
        resolved["scannet_axis_aligned_only"]
    )
    resolved["inference_every_keyframes"] = _positive_int(
        resolved, "inference_every_keyframes"
    )
    lifecycle = resolved["candidate_lifecycle"]
    lifecycle["ttl_clock"] = str(lifecycle["ttl_clock"]).strip().lower()
    if lifecycle["ttl_clock"] not in {"keyframe", "provider_call"}:
        raise ValueError(
            "candidate_lifecycle.ttl_clock must be keyframe or provider_call"
        )
    if not isinstance(lifecycle["archive_confirmed"], (bool, np.bool_)):
        raise ValueError(
            "candidate_lifecycle.archive_confirmed must be Boolean"
        )
    lifecycle["archive_confirmed"] = bool(
        lifecycle["archive_confirmed"]
    )
    appearance = resolved["appearance_memory"]
    for key in ("enabled", "masked_crop"):
        if not isinstance(appearance[key], (bool, np.bool_)):
            raise ValueError(f"appearance_memory.{key} must be Boolean")
        appearance[key] = bool(appearance[key])

    matching = resolved["matching"]
    for key in (
        "global_match_iou",
        "global_match_2d_iou",
        "rekey_iou",
        "absorb_supplemental_iou",
    ):
        matching[key] = _finite_float(matching, key, lower=0.0, upper=1.0)
    matching["max_center_distance"] = _finite_float(
        matching, "max_center_distance", lower=0.0, strict_lower=True
    )
    matching["crop_to_global_expansion"] = _finite_float(
        matching, "crop_to_global_expansion", lower=1.0
    )

    refit = resolved["refit"]
    if not isinstance(refit["enabled"], (bool, np.bool_)):
        raise ValueError("refit.enabled must be Boolean")
    refit["enabled"] = bool(refit["enabled"])
    refit["min_views"] = _positive_int(refit, "min_views")
    refit["min_points"] = _positive_int(refit, "min_points")
    for key in ("blend", "min_original_point_support", "min_reprojection_iou"):
        refit[key] = _finite_float(refit, key, lower=0.0, upper=1.0)
    refit["extent_padding"] = _finite_float(
        refit, "extent_padding", lower=0.0
    )
    refit["max_center_shift_ratio"] = _finite_float(
        refit, "max_center_shift_ratio", lower=0.0, strict_lower=True
    )
    refit["min_extent_ratio"] = _finite_float(
        refit, "min_extent_ratio", lower=0.0, strict_lower=True
    )
    refit["max_extent_ratio"] = _finite_float(
        refit, "max_extent_ratio", lower=0.0, strict_lower=True
    )
    if refit["max_extent_ratio"] < refit["min_extent_ratio"]:
        raise ValueError("refit max_extent_ratio must exceed min_extent_ratio")
    refit["min_reprojection_improvement"] = _finite_float(
        refit, "min_reprojection_improvement"
    )

    box_refiner = resolved["box_refiner"]
    if not isinstance(box_refiner["enabled"], (bool, np.bool_)):
        raise ValueError("box_refiner.enabled must be Boolean")
    box_refiner["enabled"] = bool(box_refiner["enabled"])
    box_refiner["point_count"] = _positive_int(
        box_refiner, "point_count"
    )
    box_refiner["min_quality"] = _finite_float(
        box_refiner, "min_quality", lower=0.0, upper=1.0
    )
    if not isinstance(box_refiner["architecture"], Mapping):
        raise TypeError("box_refiner.architecture must be a mapping")

    quality = resolved["quality"]
    for key in (
        "enabled",
        "preserve_original_floor",
        "apply_to_unobserved",
    ):
        if not isinstance(quality[key], (bool, np.bool_)):
            raise ValueError(f"quality.{key} must be Boolean")
        quality[key] = bool(quality[key])
    quality["mode"] = str(quality["mode"]).strip().lower()
    if quality["mode"] not in {"heuristic", "linear", "mlp"}:
        raise ValueError("quality.mode must be heuristic, linear, or mlp")
    quality["blend_with_detector"] = _finite_float(
        quality, "blend_with_detector", lower=0.0, upper=1.0
    )
    quality["support_reference_points"] = _positive_int(
        quality, "support_reference_points"
    )
    quality["target_views"] = _positive_int(quality, "target_views")
    quality["max_view_records"] = _positive_int(
        quality, "max_view_records"
    )
    soft_nms = quality["soft_nms"]
    if not isinstance(soft_nms["enabled"], (bool, np.bool_)):
        raise ValueError("quality.soft_nms.enabled must be Boolean")
    soft_nms["enabled"] = bool(soft_nms["enabled"])
    soft_nms["method"] = str(soft_nms["method"]).strip().lower()
    if soft_nms["method"] not in {"linear", "gaussian", "hard"}:
        raise ValueError("quality.soft_nms.method is invalid")
    for key in ("iou_threshold", "score_threshold"):
        soft_nms[key] = _finite_float(
            soft_nms, key, lower=0.0, upper=1.0
        )
    soft_nms["sigma"] = _finite_float(
        soft_nms, "sigma", lower=0.0, strict_lower=True
    )
    if soft_nms["max_detections"] is not None:
        soft_nms["max_detections"] = _positive_int(
            soft_nms, "max_detections"
        )

    supplemental = resolved["supplemental_output"]
    if not isinstance(supplemental["enabled"], (bool, np.bool_)):
        raise ValueError("supplemental_output.enabled must be Boolean")
    supplemental["enabled"] = bool(supplemental["enabled"])
    supplemental["min_confirmations"] = _positive_int(
        supplemental, "min_confirmations", minimum=2
    )
    for key in (
        "min_score",
        "min_projection_iou",
        "drop_if_global_iou",
        "drop_if_supplemental_iou",
    ):
        supplemental[key] = _finite_float(
            supplemental, key, lower=0.0, upper=1.0
        )

    output_filter = resolved["output_filter"]
    output_filter["minimum_extent"] = _finite_float(
        output_filter, "minimum_extent", lower=0.0
    )

    diagnostics = resolved["diagnostics"]
    for key in ("enabled", "dump_track_memory"):
        if not isinstance(diagnostics[key], (bool, np.bool_)):
            raise ValueError(f"diagnostics.{key} must be Boolean")
        diagnostics[key] = bool(diagnostics[key])
    diagnostics["point_count"] = _positive_int(
        diagnostics, "point_count"
    )
    if diagnostics["enabled"] and diagnostics["root"] is None:
        raise ValueError("diagnostics.root is required when diagnostics are enabled")
    return resolved


def corners_to_center_size(corners: Any) -> np.ndarray:
    """Convert arbitrary 8-corner boxes to enclosing world AABBs."""

    values = np.asarray(corners, dtype=np.float32)
    if values.size == 0:
        return np.empty((0, 6), dtype=np.float32)
    if values.ndim != 3 or values.shape[1:] != (8, 3):
        raise ValueError("corners must have shape [N, 8, 3]")
    if not np.isfinite(values).all():
        raise ValueError("corners must be finite")
    minimum = values.min(axis=1)
    maximum = values.max(axis=1)
    dims = maximum - minimum
    if np.any(dims <= 0.0):
        raise ValueError("corners must define positive-volume boxes")
    return np.concatenate(((minimum + maximum) * 0.5, dims), axis=1)


def center_size_to_corners(boxes: Any) -> np.ndarray:
    values = np.asarray(boxes, dtype=np.float32)
    if values.size == 0:
        return np.empty((0, 8, 3), dtype=np.float32)
    if values.ndim == 1:
        values = values[None, :]
    if values.ndim != 2 or values.shape[1] != 6:
        raise ValueError("boxes must have shape [N, 6]")
    return np.stack([aabb_corners(box[:3], box[3:6]) for box in values])


def bbox_iou_2d(box_a: Any, box_b: Any) -> float:
    a = np.asarray(box_a, dtype=np.float64).reshape(-1)
    b = np.asarray(box_b, dtype=np.float64).reshape(-1)
    if a.shape != (4,) or b.shape != (4,):
        raise ValueError("2D boxes must each have shape [4]")
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("2D boxes must be finite")
    intersection_size = np.maximum(
        np.minimum(a[2:], b[2:]) - np.maximum(a[:2], b[:2]),
        0.0,
    )
    intersection = float(np.prod(intersection_size))
    area_a = float(np.prod(np.maximum(a[2:] - a[:2], 0.0)))
    area_b = float(np.prod(np.maximum(b[2:] - b[:2], 0.0)))
    union = area_a + area_b - intersection
    return 0.0 if union <= 0.0 else float(intersection / union)


@dataclass(frozen=True)
class ViewEvidence:
    frame_index: int
    score: float
    bbox: np.ndarray
    intrinsics: np.ndarray
    camera_to_world: np.ndarray
    image_shape: Tuple[int, int]
    area_ratio: float

    def __post_init__(self) -> None:
        bbox = np.asarray(self.bbox, dtype=np.float32)
        intrinsics = np.asarray(self.intrinsics, dtype=np.float32)
        pose = np.asarray(self.camera_to_world, dtype=np.float32)
        if bbox.shape != (4,) or intrinsics.shape != (3, 3) or pose.shape != (4, 4):
            raise ValueError("invalid view-evidence array shape")
        for value in (bbox, intrinsics, pose):
            if not np.isfinite(value).all():
                raise ValueError("view evidence must be finite")
        object.__setattr__(self, "bbox", bbox.copy())
        object.__setattr__(self, "intrinsics", intrinsics.copy())
        object.__setattr__(self, "camera_to_world", pose.copy())


@dataclass
class EvidenceStats:
    scores: List[float] = field(default_factory=list)
    view_records: List[ViewEvidence] = field(default_factory=list)
    label_votes: Counter = field(default_factory=Counter)
    feature_sum: Optional[np.ndarray] = None
    feature_count: int = 0
    absorbed_views: int = 0

    def record(
        self,
        proposal: SupplementalProposal,
        view: ViewEvidence,
        *,
        max_views: int,
    ) -> None:
        self.scores.append(float(proposal.score))
        self.scores = self.scores[-64:]
        self.view_records.append(view)
        self.view_records.sort(key=lambda item: (-item.score, item.frame_index))
        del self.view_records[max_views:]
        if proposal.label is not None:
            self.label_votes[proposal.label] += float(proposal.score)
        if proposal.feature is not None:
            feature = np.asarray(proposal.feature, dtype=np.float32)
            norm = float(np.linalg.norm(feature))
            if norm > 1e-8:
                feature = feature / norm
                if self.feature_sum is None:
                    self.feature_sum = np.zeros_like(feature)
                if self.feature_sum.shape == feature.shape:
                    self.feature_sum += feature
                    self.feature_count += 1

    def merge_from(
        self,
        other: "EvidenceStats",
        *,
        max_views: int,
    ) -> None:
        """Merge frozen candidate evidence without dropping appearance state."""

        self.scores = (list(other.scores) + self.scores)[-64:]
        self.view_records.extend(other.view_records)
        self.view_records.sort(
            key=lambda item: (-item.score, item.frame_index)
        )
        del self.view_records[max_views:]
        self.label_votes.update(other.label_votes)
        self.absorbed_views += int(other.absorbed_views)
        if other.feature_sum is None or other.feature_count <= 0:
            return
        if self.feature_sum is None:
            self.feature_sum = other.feature_sum.copy()
            self.feature_count = int(other.feature_count)
        elif self.feature_sum.shape == other.feature_sum.shape:
            self.feature_sum += other.feature_sum
            self.feature_count += int(other.feature_count)

    @property
    def mean_score(self) -> float:
        return float(np.mean(self.scores)) if self.scores else 0.0

    @property
    def label(self) -> Optional[str]:
        if not self.label_votes:
            return None
        return sorted(
            self.label_votes.items(), key=lambda item: (-item[1], item[0])
        )[0][0]

    @property
    def appearance_consistency(self) -> float:
        if self.feature_sum is None or self.feature_count == 0:
            return 0.5
        if self.feature_count == 1:
            return 0.5
        return float(
            np.clip(
                np.linalg.norm(self.feature_sum) / float(self.feature_count),
                0.0,
                1.0,
            )
        )


@dataclass
class GlobalEvidence:
    stable_id: int
    memory: ObjectGeometryMemory
    stats: EvidenceStats
    detector_score: float
    last_box: np.ndarray


@dataclass
class SupplementalEvidence:
    track_id: int
    stats: EvidenceStats = field(default_factory=EvidenceStats)


@dataclass(frozen=True)
class LiftedProposal:
    proposal: SupplementalProposal
    observation: ObjectObservation
    box: np.ndarray
    depth_ratio: float
    view: ViewEvidence


@dataclass(frozen=True)
class FinalRefinementResult:
    corners: np.ndarray
    boxes: np.ndarray
    scores: np.ndarray
    source_indices: np.ndarray
    stable_ids: np.ndarray
    labels: Tuple[Optional[str], ...]
    quality_features: np.ndarray
    summary: Mapping[str, Any]


def _empty_runtime_stats() -> Dict[str, Any]:
    return {
        "keyframes": 0,
        "provider_calls": 0,
        "provider_seconds": 0.0,
        "appearance_seconds": 0.0,
        "geometry_seconds": 0.0,
        "proposals": 0,
        "lifted": 0,
        "matched_global": 0,
        "candidate_updates": 0,
        "candidate_archived": 0,
        "candidate_discarded": 0,
        "supplemental_considered": 0,
        "supplemental_rejected_extent": 0,
        "supplemental_rejected_score": 0,
        "supplemental_rejected_projection": 0,
        "supplemental_rejected_global": 0,
        "supplemental_output": 0,
        "supplemental_deduplicated": 0,
        "refits_attempted": 0,
        "refits_accepted": 0,
        "neural_refits_accepted": 0,
        "rejected": Counter(),
    }


def _validate_runtime_arrays(
    image: Any,
    depth: Any,
    intrinsics: Any,
    camera_to_world: Any,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    image = np.asarray(image)
    depth = np.asarray(depth)
    intrinsics = np.asarray(intrinsics, dtype=np.float32)
    camera_to_world = np.asarray(camera_to_world, dtype=np.float32)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("image must have shape [H, W, 3]")
    if depth.ndim == 3 and 1 in (depth.shape[0], depth.shape[-1]):
        depth = np.squeeze(depth)
    if depth.ndim != 2:
        raise ValueError("depth must have shape [H, W]")
    if intrinsics.shape == (4, 4):
        intrinsics = intrinsics[:3, :3]
    if intrinsics.shape != (3, 3):
        raise ValueError("intrinsics must have shape [3,3] or [4,4]")
    if camera_to_world.shape != (4, 4):
        raise ValueError("camera_to_world must have shape [4,4]")
    if not np.isfinite(intrinsics).all() or not np.isfinite(camera_to_world).all():
        raise ValueError("camera matrices must be finite")
    return image, depth, intrinsics, camera_to_world


class OnlineRefinementController:
    """Opt-in final-output refinement controller."""

    def __init__(
        self,
        config: Mapping[str, Any],
        *,
        device: str = "cpu",
        provider: Optional[ProposalProvider] = None,
        appearance_encoder: Any = None,
        box_refiner: Any = None,
        quality_scorer: Any = None,
    ) -> None:
        self.config = resolve_online_refinement_config(config)
        self.enabled = bool(self.config["enabled"])
        self.device = str(device)
        if (
            self.enabled
            and self.config["scannet_axis_aligned_only"]
            and "dataset" in config
            and str(config["dataset"]).lower() != "scannet"
        ):
            raise ValueError(
                "this refinement configuration is restricted to ScanNet "
                "axis-aligned evaluation"
            )
        self.provider: Optional[ProposalProvider] = None
        self.appearance_encoder = None
        self.box_refiner = None
        self.quality_scorer = None
        self.object_config: Dict[str, Any] = {}
        self.track_manager: Optional[CandidateTrackManager] = None
        self.global_tracks: Dict[int, GlobalEvidence] = {}
        self.supplemental_metadata: Dict[int, SupplementalEvidence] = {}
        self.keyframe_count = 0
        self.scene_id: Optional[str] = None
        self.stats: Dict[str, Any] = _empty_runtime_stats()
        if not self.enabled:
            return

        proposals_cfg = self.config["supplemental_proposals"]
        provider_device = proposals_cfg.get("device", self.device)
        self.provider = (
            provider
            if provider is not None
            else build_provider(proposals_cfg, str(provider_device))
        )
        if self.config["appearance_memory"]["enabled"]:
            self.appearance_encoder = appearance_encoder
        self.object_config = resolve_object_memory_config(
            self.config["object_memory"]
        )
        self.track_manager = CandidateTrackManager(
            self.object_config,
            archive_confirmed=self.config["candidate_lifecycle"][
                "archive_confirmed"
            ],
        )

        refiner_cfg = self.config["box_refiner"]
        architecture = BoxRefinerConfig(
            **dict(refiner_cfg["architecture"])
        ).validated()
        if architecture.quality_feature_dim != QUALITY_FEATURE_DIM:
            raise ValueError(
                "BoxRefiner quality_feature_dim must match the fixed "
                f"quality schema ({QUALITY_FEATURE_DIM})"
            )
        refiner_device = refiner_cfg["device"] or self.device
        self.box_refiner = (
            box_refiner
            if box_refiner is not None
            else build_box_refiner(
                enabled=refiner_cfg["enabled"],
                checkpoint_path=refiner_cfg["checkpoint"],
                config=architecture,
                device=refiner_device,
            )
        )

        quality_cfg = self.config["quality"]
        if quality_scorer is not None:
            self.quality_scorer = quality_scorer
        elif quality_cfg["enabled"]:
            if quality_cfg["mode"] == "heuristic":
                self.quality_scorer = make_quality_scorer("heuristic")
            else:
                if quality_cfg["checkpoint"] is None:
                    raise ValueError(
                        "learned quality scoring requires quality.checkpoint"
                    )
                self.quality_scorer = load_quality_scorer(
                    quality_cfg["checkpoint"],
                    method=quality_cfg["mode"],
                )

    def reset_scene(self, scene_id: str) -> None:
        """Clear all geometry/track state while retaining loaded models."""

        if not isinstance(scene_id, str) or not scene_id.strip():
            raise ValueError("scene_id must be a non-empty string")
        self.scene_id = scene_id.strip()
        self.keyframe_count = 0
        self.global_tracks.clear()
        self.supplemental_metadata.clear()
        if self.enabled:
            self.track_manager = CandidateTrackManager(
                self.object_config,
                archive_confirmed=self.config["candidate_lifecycle"][
                    "archive_confirmed"
                ],
            )
        self.stats = _empty_runtime_stats()

    @classmethod
    def from_config(
        cls,
        cfg: Mapping[str, Any],
        *,
        device: str = "cpu",
    ) -> "OnlineRefinementController":
        return cls(cfg, device=device)

    def _global_inputs(
        self,
        global_corners: Any,
        global_scores: Any,
        stable_ids: Any,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        corners = np.asarray(global_corners, dtype=np.float32)
        if corners.size == 0:
            corners = np.empty((0, 8, 3), dtype=np.float32)
        boxes = corners_to_center_size(corners)
        scores = np.asarray(global_scores, dtype=np.float32).reshape(-1)
        ids = np.asarray(stable_ids, dtype=np.int64).reshape(-1)
        if len(boxes) != len(scores) or len(boxes) != len(ids):
            raise ValueError("global boxes, scores, and stable_ids must align")
        if not np.isfinite(scores).all():
            raise ValueError("global scores must be finite")
        if ((scores < 0.0) | (scores > 1.0)).any():
            raise ValueError("global scores must lie in [0,1]")
        if len(set(ids.tolist())) != len(ids):
            raise ValueError("stable_ids must be unique")
        if np.any(ids < 0):
            raise ValueError("global stable_ids must be non-negative")
        return boxes, scores, ids

    def _sync_global_tracks(
        self,
        boxes: np.ndarray,
        scores: np.ndarray,
        stable_ids: np.ndarray,
    ) -> None:
        current_ids = set(stable_ids.tolist())
        matching_cfg = self.config["matching"]
        for stable_id, box, score in zip(stable_ids, boxes, scores):
            key = int(stable_id)
            if key not in self.global_tracks:
                candidates = []
                for old_key, evidence in self.global_tracks.items():
                    if old_key in current_ids:
                        continue
                    overlap = aabb_iou(
                        box[:3], box[3:6],
                        evidence.last_box[:3], evidence.last_box[3:6],
                    )
                    if overlap >= matching_cfg["rekey_iou"]:
                        candidates.append((-overlap, old_key))
                if candidates:
                    _, old_key = min(candidates)
                    evidence = self.global_tracks.pop(old_key)
                    evidence.stable_id = key
                    self.global_tracks[key] = evidence
            if key in self.global_tracks:
                evidence = self.global_tracks[key]
                evidence.last_box = box.copy()
                evidence.detector_score = float(score)

    def _new_global_evidence(
        self,
        stable_id: int,
        box: np.ndarray,
        score: float,
    ) -> GlobalEvidence:
        evidence = GlobalEvidence(
            stable_id=int(stable_id),
            memory=ObjectGeometryMemory(
                track_id=int(stable_id), config=self.object_config
            ),
            stats=EvidenceStats(),
            detector_score=float(score),
            last_box=np.asarray(box, dtype=np.float32).copy(),
        )
        self.global_tracks[int(stable_id)] = evidence
        return evidence

    def _lift_proposals(
        self,
        proposals: Sequence[SupplementalProposal],
        *,
        depth: np.ndarray,
        intrinsics: np.ndarray,
        camera_to_world: np.ndarray,
        frame_index: int,
        image_shape: Tuple[int, int],
    ) -> List[LiftedProposal]:
        lifted: List[LiftedProposal] = []
        started = time.perf_counter()
        minimum_points = int(self.object_config["min_points_for_aabb"])
        for proposal in proposals:
            depth_observation = extract_masked_world_points(
                depth,
                proposal.mask,
                intrinsics,
                camera_to_world,
                self.object_config,
            )
            if depth_observation.retained_point_count < minimum_points:
                continue
            center, dims = robust_quantile_aabb(
                depth_observation.points_world,
                lower_quantile=float(
                    self.object_config["aabb_lower_quantile"]
                ),
                upper_quantile=float(
                    self.object_config["aabb_upper_quantile"]
                ),
                min_points=minimum_points,
                minimum_dimension=float(
                    self.object_config["minimum_aabb_dimension"]
                ),
            )
            box = np.concatenate((center, dims)).astype(np.float32)
            height, width = image_shape
            area = max(
                0.0,
                float(
                    (proposal.bbox[2] - proposal.bbox[0])
                    * (proposal.bbox[3] - proposal.bbox[1])
                ),
            )
            view = ViewEvidence(
                frame_index=frame_index,
                score=float(proposal.score),
                bbox=proposal.bbox,
                intrinsics=intrinsics,
                camera_to_world=camera_to_world,
                image_shape=image_shape,
                area_ratio=float(
                    np.clip(area / max(height * width, 1), 0.0, 1.0)
                ),
            )
            observation = ObjectObservation.from_depth_observation(
                depth_observation,
                confidence=float(proposal.score),
                projection_mask_iou=1.0,
            )
            lifted.append(
                LiftedProposal(
                    proposal=proposal,
                    observation=observation,
                    box=box,
                    depth_ratio=depth_observation.valid_depth_ratio,
                    view=view,
                )
            )
        self.stats["geometry_seconds"] += time.perf_counter() - started
        self.stats["lifted"] += len(lifted)
        return lifted

    def _match_to_globals(
        self,
        lifted: Sequence[LiftedProposal],
        boxes: np.ndarray,
        intrinsics: np.ndarray,
        camera_to_world: np.ndarray,
    ) -> Dict[int, int]:
        cfg = self.config["matching"]
        candidates: List[Tuple[float, float, float, int, int]] = []
        for proposal_index, item in enumerate(lifted):
            for global_index, box in enumerate(boxes):
                overlap = aabb_iou(
                    item.box[:3],
                    item.box[3:6],
                    box[:3],
                    box[3:6],
                )
                center_distance = float(
                    np.linalg.norm(item.box[:3] - box[:3])
                )
                projection_iou = projected_aabb_mask_iou(
                    box[:3],
                    box[3:6],
                    intrinsics,
                    camera_to_world,
                    item.proposal.mask,
                    threshold=float(self.object_config["mask_threshold"]),
                )
                valid = overlap >= cfg["global_match_iou"] or (
                    projection_iou >= cfg["global_match_2d_iou"]
                    and center_distance <= cfg["max_center_distance"]
                )
                if not valid:
                    continue
                center_quality = max(
                    0.0, 1.0 - center_distance / cfg["max_center_distance"]
                )
                point_support = points_inside_aabb_fraction(
                    item.observation.points_world,
                    box[:3],
                    box[3:6] * cfg["crop_to_global_expansion"],
                )
                score = (
                    2.0 * overlap
                    + projection_iou
                    + 0.50 * point_support
                    + 0.25 * center_quality
                )
                candidates.append(
                    (
                        -float(score),
                        -float(overlap),
                        -float(projection_iou),
                        proposal_index,
                        global_index,
                    )
                )
        candidates.sort()
        assignments: Dict[int, int] = {}
        used_globals = set()
        for _, _, _, proposal_index, global_index in candidates:
            if proposal_index in assignments or global_index in used_globals:
                continue
            assignments[proposal_index] = global_index
            used_globals.add(global_index)
        return assignments

    def _projection_iou_for_view(
        self, box: np.ndarray, view: ViewEvidence
    ) -> float:
        projected = project_aabb_to_image(
            box[:3],
            box[3:6],
            view.intrinsics,
            view.camera_to_world,
            view.image_shape,
            require_all_in_front=False,
        )
        if projected is None:
            return 0.0
        return bbox_iou_2d(projected, view.bbox)

    def _add_global_observation(
        self,
        evidence: GlobalEvidence,
        item: LiftedProposal,
        *,
        box: np.ndarray,
        frame_index: int,
    ) -> None:
        expansion = float(
            self.config["matching"]["crop_to_global_expansion"]
        )
        points = item.observation.points_world
        inside = points_inside_aabb(
            points, box[:3], box[3:6] * expansion
        )
        minimum_points = int(self.object_config["min_points_for_aabb"])
        retained = points[inside]
        if retained.shape[0] < minimum_points:
            retained = points
        projection_iou = self._projection_iou_for_view(box, item.view)
        observation = ObjectObservation(
            points_world=retained,
            confidence=float(item.proposal.score),
            mask_pixels=item.observation.mask_pixels,
            valid_depth_pixels=item.observation.valid_depth_pixels,
            projection_mask_iou=projection_iou,
        )
        evidence.memory.add_observation(observation, frame_index)
        evidence.stats.record(
            item.proposal,
            item.view,
            max_views=int(self.config["quality"]["max_view_records"]),
        )

    def _absorb_candidate_track(
        self,
        evidence: GlobalEvidence,
        item: LiftedProposal,
        *,
        frame_index: int,
    ) -> None:
        if self.track_manager is None:
            return
        threshold = float(
            self.config["matching"]["absorb_supplemental_iou"]
        )
        candidates = []
        for archived, tracks in (
            (False, self.track_manager.tracks),
            (True, self.track_manager.archived_tracks),
        ):
            for track_id, track in tracks.items():
                track_box = track.memory.aabb
                if track_box is None:
                    continue
                overlap = aabb_iou(
                    item.box[:3],
                    item.box[3:6],
                    track_box[0],
                    track_box[1],
                )
                if overlap >= threshold:
                    # Prefer the highest overlap, then an active track, then
                    # the stable track id.
                    candidates.append((-overlap, archived, track_id))
        if not candidates:
            return
        _, archived, track_id = min(candidates)
        source = (
            self.track_manager.archived_tracks
            if archived
            else self.track_manager.tracks
        )
        track = source.pop(track_id)
        metadata = self.supplemental_metadata.pop(track_id, None)
        if track.memory.num_points:
            summary = track.memory.quality_summary()
            evidence.memory.add_observation(
                ObjectObservation(
                    points_world=track.memory.points,
                    confidence=float(
                        summary.get("mean_confidence") or 0.5
                    ),
                    projection_mask_iou=float(
                        summary.get("mean_projection_mask_iou") or 0.5
                    ),
                ),
                frame_index,
            )
        evidence.stats.absorbed_views += int(track.view_count)
        if metadata is not None:
            evidence.stats.merge_from(
                metadata.stats,
                max_views=int(
                    self.config["quality"]["max_view_records"]
                ),
            )

    def _record_candidate_retirement(self, result: Any) -> None:
        """Keep archived metadata and discard only unconfirmed-track metadata."""

        archived = set(result.archived_track_ids)
        for discarded_id in result.discarded_track_ids:
            self.supplemental_metadata.pop(discarded_id, None)
        # Defensive compatibility: an implementation returning only
        # ``expired_track_ids`` must not leak metadata for discarded tracks.
        for expired_id in result.expired_track_ids:
            if expired_id not in archived:
                self.supplemental_metadata.pop(expired_id, None)
        self.stats["candidate_archived"] += len(result.archived_track_ids)
        self.stats["candidate_discarded"] += len(
            result.discarded_track_ids
        )

    def _advance_candidate_lifecycle(
        self,
        *,
        frame_index: int,
        lifecycle_step: int,
    ) -> None:
        if self.track_manager is None:
            return
        result = self.track_manager.update(
            [],
            frame_index,
            lifecycle_step=lifecycle_step,
        )
        self._record_candidate_retirement(result)

    def _update_candidates(
        self,
        unmatched: Sequence[LiftedProposal],
        *,
        frame_index: int,
        lifecycle_step: int,
    ) -> None:
        if self.track_manager is None:
            return
        result = self.track_manager.update(
            [item.observation for item in unmatched],
            frame_index,
            lifecycle_step=lifecycle_step,
        )
        self._record_candidate_retirement(result)
        for local_index, track_id in result.assignments.items():
            item = unmatched[local_index]
            metadata = self.supplemental_metadata.setdefault(
                track_id, SupplementalEvidence(track_id)
            )
            metadata.stats.record(
                item.proposal,
                item.view,
                max_views=int(self.config["quality"]["max_view_records"]),
            )
        self.stats["candidate_updates"] += len(result.assignments)

    def process_keyframe(
        self,
        *,
        image: Any,
        depth: Any,
        intrinsics: Any,
        camera_to_world: Any,
        frame_id: int,
        scene_id: str,
        cache_frame_id: Optional[str] = None,
        global_corners: Any,
        global_scores: Any,
        stable_ids: Any,
    ) -> None:
        """Run one scheduled proposal/memory update.

        The method never mutates the supplied global arrays.
        """

        if not self.enabled:
            return
        if not isinstance(frame_id, (int, np.integer)) or int(frame_id) < 0:
            raise ValueError("frame_id must be a non-negative integer")
        if not isinstance(scene_id, str) or not scene_id.strip():
            raise ValueError("scene_id must be a non-empty string")
        if cache_frame_id is not None and (
            not isinstance(cache_frame_id, str)
            or not cache_frame_id.strip()
        ):
            raise ValueError("cache_frame_id must be a non-empty string")
        image, depth, intrinsics, camera_to_world = _validate_runtime_arrays(
            image, depth, intrinsics, camera_to_world
        )
        boxes, scores, ids = self._global_inputs(
            global_corners, global_scores, stable_ids
        )
        requested_scene = scene_id.strip()
        if self.scene_id is None:
            self.scene_id = requested_scene
        elif self.scene_id != requested_scene:
            self.reset_scene(requested_scene)
        frame_index = self.keyframe_count
        self.keyframe_count += 1
        self.stats["keyframes"] += 1
        self._sync_global_tracks(boxes, scores, ids)

        interval = int(self.config["inference_every_keyframes"])
        if frame_index % interval != 0:
            if (
                self.config["candidate_lifecycle"]["ttl_clock"]
                == "keyframe"
            ):
                self._advance_candidate_lifecycle(
                    frame_index=frame_index,
                    lifecycle_step=frame_index,
                )
            return

        if self.provider is None:
            raise RuntimeError("enabled controller has no proposal provider")
        provider_step = int(self.stats["provider_calls"])
        started = time.perf_counter()
        batches = self.provider.predict(
            [image],
            frame_ids=[
                f"{self.scene_id}:"
                + (
                    cache_frame_id.strip()
                    if cache_frame_id is not None
                    else f"{int(frame_id):06d}"
                )
            ],
        )
        self.stats["provider_seconds"] += time.perf_counter() - started
        self.stats["provider_calls"] += 1
        if len(batches) != 1:
            raise RuntimeError("proposal provider returned the wrong batch size")
        lifecycle_step = (
            provider_step
            if self.config["candidate_lifecycle"]["ttl_clock"]
            == "provider_call"
            else frame_index
        )
        self._advance_candidate_lifecycle(
            frame_index=frame_index,
            lifecycle_step=lifecycle_step,
        )
        proposals = batches[0]
        if self.appearance_encoder is not None and proposals:
            appearance_started = time.perf_counter()
            encoded = self.appearance_encoder(image, proposals)
            encoded = list(encoded)
            self.stats["appearance_seconds"] += (
                time.perf_counter() - appearance_started
            )
            if len(encoded) != len(proposals):
                raise RuntimeError(
                    "appearance encoder returned the wrong number of features"
                )
            proposals = [
                replace(
                    proposal,
                    feature=(
                        proposal.feature
                        if proposal.feature is not None
                        else np.asarray(feature, dtype=np.float32)
                    ),
                )
                for proposal, feature in zip(proposals, encoded)
            ]
        self.stats["proposals"] += len(proposals)
        lifted = self._lift_proposals(
            proposals,
            depth=depth,
            intrinsics=intrinsics,
            camera_to_world=camera_to_world,
            frame_index=frame_index,
            image_shape=image.shape[:2],
        )
        assignments = self._match_to_globals(
            lifted, boxes, intrinsics, camera_to_world
        )
        for proposal_index, global_index in assignments.items():
            stable_id = int(ids[global_index])
            evidence = self.global_tracks.get(stable_id)
            if evidence is None:
                evidence = self._new_global_evidence(
                    stable_id, boxes[global_index], scores[global_index]
                )
            item = lifted[proposal_index]
            self._add_global_observation(
                evidence,
                item,
                box=boxes[global_index],
                frame_index=frame_index,
            )
            self._absorb_candidate_track(
                evidence, item, frame_index=frame_index
            )
        unmatched = [
            item
            for index, item in enumerate(lifted)
            if index not in assignments
        ]
        self._update_candidates(
            unmatched,
            frame_index=frame_index,
            lifecycle_step=lifecycle_step,
        )
        self.stats["matched_global"] += len(assignments)

    def _mean_projection_iou(
        self, box: np.ndarray, stats: EvidenceStats
    ) -> float:
        if not stats.view_records:
            return 0.0
        values = np.asarray(
            [
                self._projection_iou_for_view(box, view)
                for view in stats.view_records
            ],
            dtype=np.float32,
        )
        weights = np.asarray(
            [max(view.score, 1e-4) for view in stats.view_records],
            dtype=np.float32,
        )
        return float(np.average(values, weights=weights))

    @staticmethod
    def _box_stability(original: np.ndarray, final: np.ndarray) -> float:
        diagonal = max(float(np.linalg.norm(original[3:6])), 1e-6)
        center_shift = float(np.linalg.norm(final[:3] - original[:3]))
        log_scale = float(
            np.mean(
                np.abs(
                    np.log(
                        np.maximum(final[3:6], 1e-6)
                        / np.maximum(original[3:6], 1e-6)
                    )
                )
            )
        )
        return float(np.exp(-(center_shift / diagonal + log_scale)))

    def _quality_mapping(
        self,
        *,
        original_box: np.ndarray,
        final_box: np.ndarray,
        detector_score: float,
        memory: Optional[ObjectGeometryMemory],
        stats: Optional[EvidenceStats],
        supplemental: bool,
        refiner_quality: float,
    ) -> Dict[str, float]:
        quality_cfg = self.config["quality"]
        if memory is None or stats is None:
            return {
                "detector_score": float(np.clip(detector_score, 0.0, 1.0)),
                "mask_confidence": 0.0,
                "valid_depth_ratio": 0.0,
                "depth_support": 0.0,
                "projection_iou": 0.0,
                "geometry_consistency": 1.0,
                "appearance_consistency": 0.5,
                "view_count_quality": 0.0,
                "box_stability": 1.0,
                "source_agreement": 0.0,
                "area_quality": 0.0,
                "refiner_quality": float(np.clip(refiner_quality, 0.0, 1.0)),
            }
        summary = memory.quality_summary()
        view_count = memory.unique_view_count + stats.absorbed_views
        point_support = points_inside_aabb_fraction(
            memory.points, final_box[:3], final_box[3:6]
        )
        geometry_consistency = 0.5 * (
            aabb_iou(
                original_box[:3],
                original_box[3:6],
                final_box[:3],
                final_box[3:6],
            )
            + point_support
        )
        area_quality = (
            float(np.mean([view.area_ratio for view in stats.view_records]))
            if stats.view_records
            else 0.0
        )
        source_agreement = (
            min(view_count / 2.0, 1.0)
            if supplemental
            else min(0.5 + view_count / 4.0, 1.0)
        )
        return {
            "detector_score": float(np.clip(detector_score, 0.0, 1.0)),
            "mask_confidence": float(np.clip(stats.mean_score, 0.0, 1.0)),
            "valid_depth_ratio": float(
                np.clip(summary.get("mean_valid_depth_ratio") or 0.0, 0.0, 1.0)
            ),
            "depth_support": float(
                np.clip(
                    memory.num_points
                    / float(quality_cfg["support_reference_points"]),
                    0.0,
                    1.0,
                )
            ),
            "projection_iou": float(
                np.clip(self._mean_projection_iou(final_box, stats), 0.0, 1.0)
            ),
            "geometry_consistency": float(
                np.clip(geometry_consistency, 0.0, 1.0)
            ),
            "appearance_consistency": stats.appearance_consistency,
            "view_count_quality": float(
                np.clip(
                    view_count / float(quality_cfg["target_views"]),
                    0.0,
                    1.0,
                )
            ),
            "box_stability": self._box_stability(original_box, final_box),
            "source_agreement": float(np.clip(source_agreement, 0.0, 1.0)),
            "area_quality": float(np.clip(area_quality / 0.10, 0.0, 1.0)),
            "refiner_quality": float(np.clip(refiner_quality, 0.0, 1.0)),
        }

    def _refit_gate(
        self,
        original: np.ndarray,
        candidate: np.ndarray,
        evidence: GlobalEvidence,
    ) -> Tuple[bool, str]:
        cfg = self.config["refit"]
        if evidence.memory.unique_view_count + evidence.stats.absorbed_views < cfg[
            "min_views"
        ]:
            return False, "views"
        if evidence.memory.num_points < cfg["min_points"]:
            return False, "points"
        if not np.isfinite(candidate).all() or np.any(candidate[3:6] <= 0.0):
            return False, "invalid"
        diagonal = max(float(np.linalg.norm(original[3:6])), 1e-6)
        shift_ratio = float(
            np.linalg.norm(candidate[:3] - original[:3]) / diagonal
        )
        if shift_ratio > cfg["max_center_shift_ratio"]:
            return False, "center_shift"
        extent_ratio = candidate[3:6] / np.maximum(original[3:6], 1e-6)
        if np.any(extent_ratio < cfg["min_extent_ratio"]) or np.any(
            extent_ratio > cfg["max_extent_ratio"]
        ):
            return False, "extent"
        support = points_inside_aabb_fraction(
            evidence.memory.points, original[:3], original[3:6]
        )
        if support < cfg["min_original_point_support"]:
            return False, "support"
        original_projection = self._mean_projection_iou(
            original, evidence.stats
        )
        candidate_projection = self._mean_projection_iou(
            candidate, evidence.stats
        )
        if candidate_projection < cfg["min_reprojection_iou"]:
            return False, "reprojection"
        if (
            candidate_projection - original_projection
            < cfg["min_reprojection_improvement"]
        ):
            return False, "reprojection_delta"
        return True, "accepted"

    def _robust_refit(
        self,
        original: np.ndarray,
        evidence: GlobalEvidence,
    ) -> Tuple[np.ndarray, bool, str]:
        cfg = self.config["refit"]
        if not cfg["enabled"]:
            return original.copy(), False, "disabled"
        memory_box = evidence.memory.aabb
        if memory_box is None:
            return original.copy(), False, "points"
        memory_center, memory_dims = memory_box
        memory_dims = memory_dims + 2.0 * float(cfg["extent_padding"])
        blend = float(cfg["blend"])
        candidate = np.concatenate(
            (
                (1.0 - blend) * original[:3] + blend * memory_center,
                (1.0 - blend) * original[3:6] + blend * memory_dims,
            )
        ).astype(np.float32)
        accepted, reason = self._refit_gate(original, candidate, evidence)
        return (
            candidate if accepted else original.copy(),
            bool(accepted),
            reason,
        )

    def _run_neural_refiner(
        self,
        box: np.ndarray,
        evidence: GlobalEvidence,
        feature_mapping: Mapping[str, float],
    ) -> Tuple[np.ndarray, float, bool]:
        if self.box_refiner is None:
            return box, 0.5, False
        import torch

        point_count = int(self.config["box_refiner"]["point_count"])
        sampled = deterministic_bounded_sample(
            evidence.memory.points, point_count
        )
        valid_count = sampled.shape[0]
        if valid_count == 0:
            return box, 0.0, False
        points = np.zeros((point_count, 3), dtype=np.float32)
        mask = np.zeros(point_count, dtype=bool)
        points[:valid_count] = sampled
        mask[:valid_count] = True
        parameter = next(self.box_refiner.parameters())
        with torch.no_grad():
            output = self.box_refiner(
                torch.from_numpy(points[None]).to(parameter.device),
                torch.from_numpy(box[None].astype(np.float32)).to(
                    parameter.device
                ),
                torch.from_numpy(
                    quality_feature_vector(feature_mapping)[None]
                ).to(parameter.device),
                torch.from_numpy(mask[None]).to(parameter.device),
            )
        center_residual = (
            output["center_residual"].detach().float().cpu().numpy()
        )
        dimension_residual = (
            output["log_dimension_residual"].detach().float().cpu().numpy()
        )
        quality = float(
            output["quality"].detach().float().cpu().numpy().reshape(-1)[0]
        )
        if quality < float(self.config["box_refiner"]["min_quality"]):
            return box, quality, False
        refined = apply_box_residual_numpy(
            box,
            center_residual[0],
            dimension_residual[0],
            max_center_fraction=self.box_refiner.config.max_center_fraction,
            max_abs_log_dimension_residual=(
                self.box_refiner.config.max_log_dimension_residual
            ),
            minimum_dimension=self.box_refiner.config.minimum_dimension,
        )
        accepted, _ = self._refit_gate(box, refined, evidence)
        return (refined if accepted else box), quality, bool(accepted)

    def _score(
        self,
        detector_score: float,
        mapping: Mapping[str, float],
        *,
        observed: bool,
    ) -> float:
        cfg = self.config["quality"]
        if (
            not cfg["enabled"]
            or self.quality_scorer is None
            or (not observed and not cfg["apply_to_unobserved"])
        ):
            return float(detector_score)
        quality_score = float(self.quality_scorer(mapping))
        blend = float(cfg["blend_with_detector"])
        score = blend * float(detector_score) + (1.0 - blend) * quality_score
        if cfg["preserve_original_floor"]:
            score = max(score, float(detector_score))
        return float(np.clip(score, 0.0, 1.0))

    def _supplemental_outputs(
        self,
        global_boxes: np.ndarray,
    ) -> List[
        Tuple[
            np.ndarray,
            float,
            int,
            Optional[str],
            np.ndarray,
            ObjectGeometryMemory,
        ]
    ]:
        output = []
        deduplicated = 0
        cfg = self.config["supplemental_output"]
        for key in (
            "supplemental_considered",
            "supplemental_rejected_extent",
            "supplemental_rejected_score",
            "supplemental_rejected_projection",
            "supplemental_rejected_global",
            "supplemental_output",
        ):
            self.stats[key] = 0
        if not cfg["enabled"] or self.track_manager is None:
            self.stats["supplemental_deduplicated"] = 0
            return output
        candidates = []
        for track in self.track_manager.confirmed_tracks(
            include_archived=True
        ):
            metadata = self.supplemental_metadata.get(track.track_id)
            if metadata is None or track.view_count < cfg["min_confirmations"]:
                continue
            track_box = track.memory.aabb
            if track_box is None:
                continue
            self.stats["supplemental_considered"] += 1
            box = np.concatenate(track_box).astype(np.float32)
            minimum_extent = float(
                self.config["output_filter"]["minimum_extent"]
            )
            if minimum_extent > 0.0 and np.any(
                box[3:6] < minimum_extent
            ):
                self.stats["supplemental_rejected_extent"] += 1
                continue
            detector_score = metadata.stats.mean_score
            if detector_score < cfg["min_score"]:
                self.stats["supplemental_rejected_score"] += 1
                continue
            mapping = self._quality_mapping(
                original_box=box,
                final_box=box,
                detector_score=detector_score,
                memory=track.memory,
                stats=metadata.stats,
                supplemental=True,
                refiner_quality=0.5,
            )
            if mapping["projection_iou"] < cfg["min_projection_iou"]:
                self.stats["supplemental_rejected_projection"] += 1
                continue
            if len(global_boxes):
                maximum_overlap = max(
                    aabb_iou(
                        box[:3],
                        box[3:6],
                        global_box[:3],
                        global_box[3:6],
                    )
                    for global_box in global_boxes
                )
                if maximum_overlap >= cfg["drop_if_global_iou"]:
                    self.stats["supplemental_rejected_global"] += 1
                    continue
            score = self._score(detector_score, mapping, observed=True)
            if score < cfg["min_score"]:
                self.stats["supplemental_rejected_score"] += 1
                continue
            candidates.append(
                (
                    -score,
                    -track.view_count,
                    track.track_id,
                    box,
                    score,
                    -(track.track_id + 1),
                    metadata.stats.label,
                    quality_feature_vector(mapping),
                    track.memory,
                )
            )

        # A long-lived archive can contain a new track for an object that
        # re-entered the camera after its active TTL elapsed.  Keep the
        # strongest representative for high-overlap duplicates while leaving
        # neighbouring/nested objects untouched at the conservative threshold.
        accepted_boxes = []
        for (
            _,
            _,
            _,
            box,
            score,
            supplemental_id,
            label,
            features,
            memory,
        ) in sorted(candidates):
            if accepted_boxes and max(
                aabb_iou(
                    box[:3],
                    box[3:6],
                    accepted[:3],
                    accepted[3:6],
                )
                for accepted in accepted_boxes
            ) >= cfg["drop_if_supplemental_iou"]:
                deduplicated += 1
                continue
            accepted_boxes.append(box)
            output.append(
                (
                    box,
                    score,
                    supplemental_id,
                    label,
                    features,
                    memory,
                )
            )
        self.stats["supplemental_deduplicated"] = deduplicated
        self.stats["supplemental_output"] = len(output)
        return output

    def finalize(
        self,
        *,
        global_corners: Any,
        global_scores: Any,
        stable_ids: Any,
        scene_id: Optional[str] = None,
    ) -> FinalRefinementResult:
        """Return refined and supplemental detections without mutating inputs."""

        corners_input = np.asarray(global_corners, dtype=np.float32)
        if corners_input.size == 0:
            corners_input = np.empty((0, 8, 3), dtype=np.float32)
        boxes, scores, ids = self._global_inputs(
            corners_input, global_scores, stable_ids
        )
        if not self.enabled:
            empty_features = np.zeros(
                (len(boxes), QUALITY_FEATURE_DIM), dtype=np.float32
            )
            return FinalRefinementResult(
                corners=corners_input.copy(),
                boxes=boxes,
                scores=scores.copy(),
                source_indices=np.arange(len(boxes), dtype=np.int64),
                stable_ids=ids.copy(),
                labels=tuple(None for _ in boxes),
                quality_features=empty_features,
                summary={"enabled": False},
            )

        self._sync_global_tracks(boxes, scores, ids)
        final_corners: List[np.ndarray] = []
        final_boxes: List[np.ndarray] = []
        final_scores: List[float] = []
        source_indices: List[int] = []
        result_ids: List[int] = []
        labels: List[Optional[str]] = []
        feature_rows: List[np.ndarray] = []
        memories: List[Optional[ObjectGeometryMemory]] = []

        for index, (original, detector_score, stable_id) in enumerate(
            zip(boxes, scores, ids)
        ):
            evidence = self.global_tracks.get(int(stable_id))
            refined = original.copy()
            robust_accepted = False
            refiner_quality = 0.5
            if evidence is not None:
                self.stats["refits_attempted"] += 1
                refined, robust_accepted, reason = self._robust_refit(
                    original, evidence
                )
                if robust_accepted:
                    self.stats["refits_accepted"] += 1
                elif reason != "disabled":
                    self.stats["rejected"][reason] += 1
                preliminary = self._quality_mapping(
                    original_box=original,
                    final_box=refined,
                    detector_score=float(detector_score),
                    memory=evidence.memory,
                    stats=evidence.stats,
                    supplemental=False,
                    refiner_quality=0.5,
                )
                neural, refiner_quality, neural_accepted = (
                    self._run_neural_refiner(refined, evidence, preliminary)
                )
                if neural_accepted:
                    refined = neural
                    self.stats["neural_refits_accepted"] += 1
            mapping = self._quality_mapping(
                original_box=original,
                final_box=refined,
                detector_score=float(detector_score),
                memory=evidence.memory if evidence is not None else None,
                stats=evidence.stats if evidence is not None else None,
                supplemental=False,
                refiner_quality=refiner_quality,
            )
            score = self._score(
                float(detector_score),
                mapping,
                observed=evidence is not None
                and evidence.memory.observation_count > 0,
            )
            if robust_accepted or not np.array_equal(refined, original):
                corners = aabb_corners(refined[:3], refined[3:6])
            else:
                corners = corners_input[index].copy()
            final_corners.append(corners)
            final_boxes.append(refined)
            final_scores.append(score)
            source_indices.append(index)
            result_ids.append(int(stable_id))
            labels.append(evidence.stats.label if evidence is not None else None)
            feature_rows.append(quality_feature_vector(mapping))
            memories.append(evidence.memory if evidence is not None else None)

        minimum_extent = float(
            self.config["output_filter"]["minimum_extent"]
        )
        eligible_global_boxes = np.asarray(
            final_boxes, dtype=np.float32
        )
        if len(eligible_global_boxes) and minimum_extent > 0.0:
            eligible_global_boxes = eligible_global_boxes[
                np.all(
                    eligible_global_boxes[:, 3:6] >= minimum_extent,
                    axis=1,
                )
            ]
        for (
            box,
            score,
            supplemental_id,
            label,
            features,
            memory,
        ) in self._supplemental_outputs(eligible_global_boxes):
            final_corners.append(aabb_corners(box[:3], box[3:6]))
            final_boxes.append(box)
            final_scores.append(score)
            source_indices.append(-1)
            result_ids.append(supplemental_id)
            labels.append(label)
            feature_rows.append(features)
            memories.append(memory)

        if final_boxes:
            boxes_array = np.asarray(final_boxes, dtype=np.float32)
            corners_array = np.asarray(final_corners, dtype=np.float32)
            scores_array = np.asarray(final_scores, dtype=np.float32)
            source_array = np.asarray(source_indices, dtype=np.int64)
            ids_array = np.asarray(result_ids, dtype=np.int64)
            features_array = np.asarray(feature_rows, dtype=np.float32)
        else:
            boxes_array = np.empty((0, 6), dtype=np.float32)
            corners_array = np.empty((0, 8, 3), dtype=np.float32)
            scores_array = np.empty(0, dtype=np.float32)
            source_array = np.empty(0, dtype=np.int64)
            ids_array = np.empty(0, dtype=np.int64)
            features_array = np.empty(
                (0, QUALITY_FEATURE_DIM), dtype=np.float32
            )

        if len(boxes_array) and minimum_extent > 0.0:
            valid_output = np.all(
                boxes_array[:, 3:6] >= minimum_extent, axis=1
            )
            boxes_array = boxes_array[valid_output]
            corners_array = corners_array[valid_output]
            scores_array = scores_array[valid_output]
            source_array = source_array[valid_output]
            ids_array = ids_array[valid_output]
            features_array = features_array[valid_output]
            labels = [
                label
                for label, keep in zip(labels, valid_output)
                if bool(keep)
            ]
            memories = [
                memory
                for memory, keep in zip(memories, valid_output)
                if bool(keep)
            ]

        soft_cfg = self.config["quality"]["soft_nms"]
        if (
            self.config["quality"]["enabled"]
            and soft_cfg["enabled"]
            and len(boxes_array)
        ):
            keep, decayed = soft_nms_aabb_3d(
                boxes_array,
                scores_array,
                method=soft_cfg["method"],
                iou_threshold=soft_cfg["iou_threshold"],
                sigma=soft_cfg["sigma"],
                score_threshold=soft_cfg["score_threshold"],
                max_detections=soft_cfg["max_detections"],
            )
            boxes_array = boxes_array[keep]
            corners_array = corners_array[keep]
            source_array = source_array[keep]
            ids_array = ids_array[keep]
            features_array = features_array[keep]
            scores_array = decayed
            labels = [labels[int(index)] for index in keep]
            memories = [memories[int(index)] for index in keep]

        summary = self.summary()
        result = FinalRefinementResult(
            corners=corners_array,
            boxes=boxes_array,
            scores=scores_array,
            source_indices=source_array,
            stable_ids=ids_array,
            labels=tuple(labels),
            quality_features=features_array,
            summary=summary,
        )
        selected_scene = scene_id or self.scene_id
        if selected_scene is not None:
            self._dump_diagnostics(result, memories, selected_scene)
        return result

    def _dump_diagnostics(
        self,
        result: FinalRefinementResult,
        memories: Sequence[Optional[ObjectGeometryMemory]],
        scene_id: str,
    ) -> None:
        cfg = self.config["diagnostics"]
        if not cfg["enabled"] or not cfg["dump_track_memory"]:
            return
        root = Path(cfg["root"])
        root.mkdir(parents=True, exist_ok=True)
        point_count = int(cfg["point_count"])
        observed_indices = np.asarray(
            [
                index
                for index, memory in enumerate(memories)
                if memory is not None and memory.num_points > 0
            ],
            dtype=np.int64,
        )
        points = np.zeros(
            (len(observed_indices), point_count, 3), dtype=np.float32
        )
        point_mask = np.zeros(
            (len(observed_indices), point_count), dtype=bool
        )
        for output_index, result_index in enumerate(observed_indices):
            memory = memories[int(result_index)]
            assert memory is not None
            sampled = deterministic_bounded_sample(
                memory.points, point_count
            )
            points[output_index, : len(sampled)] = sampled
            point_mask[output_index, : len(sampled)] = True
        labels = np.asarray(
            [
                result.labels[int(index)]
                if result.labels[int(index)] is not None
                else ""
                for index in observed_indices
            ],
            dtype=np.str_,
        )
        summary_json = json.dumps(
            dict(result.summary), sort_keys=True, default=str
        )
        destination = root / f"{scene_id}_tracks.npz"
        with tempfile.NamedTemporaryFile(
            dir=root, suffix=".npz", delete=False
        ) as handle:
            temporary = Path(handle.name)
        try:
            np.savez_compressed(
                temporary,
                scene_id=np.asarray(scene_id),
                boxes=result.boxes[observed_indices],
                scores=result.scores[observed_indices],
                quality_features=result.quality_features[observed_indices],
                points=points,
                point_mask=point_mask,
                source_indices=result.source_indices[observed_indices],
                track_ids=result.stable_ids[observed_indices],
                result_indices=observed_indices,
                labels=labels,
                quality_feature_names=np.asarray(
                    QUALITY_FEATURE_NAMES, dtype=np.str_
                ),
                summary_json=np.asarray(summary_json),
            )
            os.replace(temporary, destination)
        finally:
            if temporary.exists():
                temporary.unlink()

    def summary(self) -> Dict[str, Any]:
        rejected = dict(sorted(self.stats["rejected"].items()))
        return {
            "enabled": self.enabled,
            "keyframes": int(self.stats["keyframes"]),
            "provider_calls": int(self.stats["provider_calls"]),
            "provider_seconds": float(self.stats["provider_seconds"]),
            "appearance_seconds": float(self.stats["appearance_seconds"]),
            "geometry_seconds": float(self.stats["geometry_seconds"]),
            "proposals": int(self.stats["proposals"]),
            "lifted": int(self.stats["lifted"]),
            "matched_global": int(self.stats["matched_global"]),
            "candidate_updates": int(self.stats["candidate_updates"]),
            "candidate_ttl_clock": (
                self.config["candidate_lifecycle"]["ttl_clock"]
                if self.enabled
                else "disabled"
            ),
            "candidate_archived_total": int(
                self.stats["candidate_archived"]
            ),
            "candidate_discarded_total": int(
                self.stats["candidate_discarded"]
            ),
            "global_memories": len(self.global_tracks),
            "active_supplemental_tracks": (
                len(self.track_manager.tracks)
                if self.track_manager is not None
                else 0
            ),
            "archived_supplemental_tracks": (
                len(self.track_manager.archived_tracks)
                if self.track_manager is not None
                else 0
            ),
            "confirmed_supplemental_tracks": (
                len(
                    self.track_manager.confirmed_tracks(
                        include_archived=True
                    )
                )
                if self.track_manager is not None
                else 0
            ),
            "supplemental_considered": int(
                self.stats["supplemental_considered"]
            ),
            "supplemental_rejected_extent": int(
                self.stats["supplemental_rejected_extent"]
            ),
            "supplemental_rejected_score": int(
                self.stats["supplemental_rejected_score"]
            ),
            "supplemental_rejected_projection": int(
                self.stats["supplemental_rejected_projection"]
            ),
            "supplemental_rejected_global": int(
                self.stats["supplemental_rejected_global"]
            ),
            "supplemental_output": int(
                self.stats["supplemental_output"]
            ),
            "supplemental_deduplicated": int(
                self.stats["supplemental_deduplicated"]
            ),
            "refits_attempted": int(self.stats["refits_attempted"]),
            "refits_accepted": int(self.stats["refits_accepted"]),
            "neural_refits_accepted": int(
                self.stats["neural_refits_accepted"]
            ),
            "refit_rejections": rejected,
        }

    def summary_text(self) -> str:
        summary = self.summary()
        return (
            "Online refinement summary | "
            f"keyframes={summary['keyframes']}, "
            f"provider_calls={summary['provider_calls']}, "
            f"proposals={summary['proposals']}, "
            f"lifted={summary['lifted']}, "
            f"matched_global={summary['matched_global']}, "
            f"candidate_updates={summary['candidate_updates']}, "
            f"candidate_clock={summary['candidate_ttl_clock']}, "
            f"active/archived="
            f"{summary['active_supplemental_tracks']}/"
            f"{summary['archived_supplemental_tracks']}, "
            "supp_filter="
            f"{summary['supplemental_considered']}->"
            f"{summary['supplemental_output']} "
            "(extent/score/proj/global/dedup="
            f"{summary['supplemental_rejected_extent']}/"
            f"{summary['supplemental_rejected_score']}/"
            f"{summary['supplemental_rejected_projection']}/"
            f"{summary['supplemental_rejected_global']}/"
            f"{summary['supplemental_deduplicated']}), "
            f"refits={summary['refits_accepted']}/"
            f"{summary['refits_attempted']}, "
            f"neural_refits={summary['neural_refits_accepted']}, "
            f"provider_s={summary['provider_seconds']:.3f}, "
            f"appearance_s={summary['appearance_seconds']:.3f}, "
            f"geometry_s={summary['geometry_seconds']:.3f}"
        )


def build_online_refinement_controller(
    cfg: Mapping[str, Any],
    *,
    device: str = "cpu",
    appearance_encoder: Any = None,
) -> OnlineRefinementController:
    """Public factory used by ``demo.py``."""

    return OnlineRefinementController(
        cfg,
        device=device,
        appearance_encoder=appearance_encoder,
    )


__all__ = [
    "DEFAULT_ONLINE_REFINEMENT_CONFIG",
    "FinalRefinementResult",
    "OnlineRefinementController",
    "bbox_iou_2d",
    "build_online_refinement_controller",
    "center_size_to_corners",
    "corners_to_center_size",
    "resolve_online_refinement_config",
]
