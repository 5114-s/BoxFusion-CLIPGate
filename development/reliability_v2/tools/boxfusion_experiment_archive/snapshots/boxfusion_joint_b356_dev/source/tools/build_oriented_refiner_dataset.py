#!/usr/bin/env python3
"""Build leakage-safe supervision for the orientation-aware B5-v2 refiner.

This is an offline-only tool.  Runtime diagnostics provide Top-K RGB-D points
and quality features, while the matching ``*_boxes.pkl`` file provides the
actual oriented BoxFusion corners.  ScanNet ground truth is used only here.

For every prediction the tool:

1. recovers the original OBB's orthonormal local frame from its corners;
2. transforms the observed points into that frame;
3. matches the evaluator's axis-aligned enclosure to the closest GT AABB;
4. projects the aligned GT centre exactly into the OBB local frame;
5. solves ``abs(aligned_basis) @ local_dims ~= gt_dims`` with a small exact
   non-negative least-squares solver;
6. clips that target to the neural refiner's reachable residual range; and
7. enables geometry supervision only when the reachable target improves
   aligned-AABB IoU.  All other samples remain quality-head negatives.

The output NPZ never contains Python objects and always carries ``scene_ids``.
The companion trainer consequently performs a scene-level split with no
train/validation leakage.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

from boxfusion.quality_score import QUALITY_FEATURE_NAMES


DATASET_SCHEMA = "boxfusion.oriented_box_refiner_dataset"
DATASET_FORMAT_VERSION = 1
AP50_DATASET_FORMAT_VERSION = 2
TRAINING_OBJECTIVES = ("improvement", "ap50")
TARGET_LINE_SEARCH_ALPHAS = (0.25, 0.50, 0.75, 1.00)
QUALITY_FEATURE_DIM = len(QUALITY_FEATURE_NAMES)
SCENE_PATTERN = re.compile(r"scene\d{4}_\d{2}")
_REQUIRED_DIAGNOSTIC_FIELDS = frozenset(
    {
        "scene_id",
        "quality_features",
        "result_indices",
    }
)
_STRICT_K5_DIAGNOSTIC_FIELDS = frozenset(
    {
        "box_refiner_points_local",
        "box_refiner_point_mask",
        "box_refiner_local_boxes",
        "box_refiner_frame_valid",
        "box_refiner_gate_points_local",
        "box_refiner_gate_point_mask",
        "box_refiner_view_valid",
        "box_refiner_view_frame_ids",
        "box_refiner_view_scores",
        "box_refiner_view_bboxes",
        "box_refiner_view_intrinsics",
        "box_refiner_view_camera_to_world",
        "box_refiner_view_image_shapes",
        "selected_view_counts",
        "selected_view_frame_ids",
        "top_k_view_valid",
        "box_refiner_frame_centers",
        "box_refiner_frame_basis",
        "runtime_diagnostics_schema",
        "box_refiner_input_schema",
        "online_ablation_profile",
        "candidate_ttl_clock",
        "candidate_track_ttl",
        "archive_confirmed_tracks",
        "top_k_views",
        "mutation_refit_enabled",
        "mutation_box_refiner_enabled",
        "mutation_quality_enabled",
        "mutation_supplemental_output_enabled",
        "mutation_soft_nms_enabled",
        "output_minimum_extent",
        "box_refiner_point_count",
        "box_refiner_gate_point_count",
        "box_refiner_max_view_records",
        "box_refiner_coordinate_frame",
        "refit_gate_min_views",
        "refit_gate_min_points",
        "refit_gate_max_center_shift_ratio",
        "refit_gate_min_extent_ratio",
        "refit_gate_max_extent_ratio",
        "refit_gate_min_original_point_support",
        "refit_gate_min_candidate_point_support",
        "refit_gate_max_candidate_support_drop",
        "refit_gate_min_reprojection_iou",
        "refit_gate_min_reprojection_improvement",
        "summary_json",
    }
)

_STRICT_PROVENANCE_EXPECTED = {
    "runtime_diagnostics_schema": "box_refiner_k5_runtime_v1",
    "box_refiner_input_schema": "oriented_local_refiner_input_v1",
    "online_ablation_profile": "b5v2_memory_observer",
    "candidate_ttl_clock": "provider_call",
    "candidate_track_ttl": 3,
    "archive_confirmed_tracks": False,
    "top_k_views": 5,
    "mutation_refit_enabled": False,
    "mutation_box_refiner_enabled": False,
    "mutation_quality_enabled": False,
    "mutation_supplemental_output_enabled": False,
    "mutation_soft_nms_enabled": False,
    "output_minimum_extent": 0.40,
    "box_refiner_point_count": 512,
    "box_refiner_gate_point_count": 8192,
    "box_refiner_max_view_records": 5,
    "box_refiner_coordinate_frame": "box_local",
    "refit_gate_min_views": 2,
    "refit_gate_min_points": 128,
    "refit_gate_max_center_shift_ratio": 0.16,
    "refit_gate_min_extent_ratio": 0.80,
    "refit_gate_max_extent_ratio": 1.25,
    "refit_gate_min_original_point_support": 0.55,
    "refit_gate_min_candidate_point_support": 0.55,
    "refit_gate_max_candidate_support_drop": 0.08,
    "refit_gate_min_reprojection_iou": 0.20,
    "refit_gate_min_reprojection_improvement": 0.0,
}
# Public read-only-by-convention view used by the trainer's schema validator.
STRICT_PROVENANCE_EXPECTED = dict(_STRICT_PROVENANCE_EXPECTED)

BASE_SAMPLE_KEYS = frozenset(
    {
        "points_local",
        "point_mask",
        "local_boxes",
        "quality_features",
        "target_residual",
        "quality_target",
        "geometry_mask",
        "scene_ids",
        "original_iou",
        "refined_iou",
        "matched_gt_index",
        "target_center_local_unclipped",
        "target_dimensions_local_unclipped",
        "basis_world",
        "result_indices",
        "track_ids",
    }
)
BASE_METADATA_KEYS = frozenset(
    {
        "schema",
        "format_version",
        "coordinate_frame",
        "quality_feature_names",
        "max_center_fraction",
        "max_log_dimension_residual",
    }
)
V2_DYNAMIC_SAMPLE_KEYS = frozenset(
    {
        "aligned_basis",
        "original_aligned_center",
        "matched_gt_box",
        "iou_gain",
        "cross_iou50",
        "near_iou50",
        "ap50_weight",
        "runtime_eligible",
        "selected_view_counts",
        "identity_tp50",
        "candidate_oracle_tp50",
    }
)
V2_SAMPLE_KEYS = BASE_SAMPLE_KEYS | V2_DYNAMIC_SAMPLE_KEYS
V2_METADATA_KEYS = BASE_METADATA_KEYS | frozenset(
    {
        "objective",
        "strict_k5_diagnostics",
        "expected_top_k_views",
        "min_runtime_views",
        "min_runtime_points",
        "runtime_minimum_extent",
        "near_iou50_band",
        "gain_cap",
        "gain_sample_weight",
        "cross_iou50_sample_weight",
        "near_iou50_sample_weight",
        "min_match_iou",
        "improvement_epsilon",
        "target_line_search_alphas",
        "forbidden_scene_count",
        "forbidden_scene_sha256",
        "training_scene_count",
        "training_scene_sha256",
    }
) | frozenset(_STRICT_PROVENANCE_EXPECTED)


@dataclass(frozen=True)
class BuildConfig:
    diagnostics_root: Path
    prediction_root: Path
    scan_root: Path
    gt_root: Path
    scene_list: Path
    output: Path
    min_match_iou: float = 0.15
    improvement_epsilon: float = 1e-4
    max_center_fraction: float = 0.15
    max_log_dimension_residual: float = float(np.log(1.25))
    objective: str = "improvement"
    strict_k5_diagnostics: bool = False
    forbidden_scene_list: Path | None = None
    expected_top_k_views: int = 5
    min_runtime_views: int = 2
    min_runtime_points: int = 128
    runtime_minimum_extent: float = 0.40
    near_iou50_band: float = 0.15
    gain_cap: float = 0.25
    gain_sample_weight: float = 2.0
    cross_iou50_sample_weight: float = 4.0
    near_iou50_sample_weight: float = 2.0
    max_center_shift_ratio: float = 0.16
    min_extent_ratio: float = 0.80
    max_extent_ratio: float = 1.25
    min_original_point_support: float = 0.55
    min_candidate_point_support: float = 0.55
    max_candidate_support_drop: float = 0.08
    min_reprojection_iou: float = 0.20
    min_reprojection_improvement: float = 0.0

    def validated(self) -> "BuildConfig":
        for name in (
            "diagnostics_root",
            "prediction_root",
            "scan_root",
            "gt_root",
        ):
            path = Path(getattr(self, name))
            if not path.is_dir():
                raise FileNotFoundError(f"{name} is not a directory: {path}")
        if not Path(self.scene_list).is_file():
            raise FileNotFoundError(self.scene_list)
        if self.objective not in TRAINING_OBJECTIVES:
            raise ValueError(
                f"objective must be one of {TRAINING_OBJECTIVES}"
            )
        strict_schema = bool(
            self.strict_k5_diagnostics or self.objective == "ap50"
        )
        if strict_schema:
            if self.forbidden_scene_list is None:
                raise ValueError(
                    "strict K5 supervision requires forbidden_scene_list"
                )
            if not Path(self.forbidden_scene_list).is_file():
                raise FileNotFoundError(self.forbidden_scene_list)
        if not isinstance(self.strict_k5_diagnostics, (bool, np.bool_)):
            raise TypeError("strict_k5_diagnostics must be Boolean")
        for name in ("min_match_iou", "improvement_epsilon"):
            value = getattr(self, name)
            if not np.isscalar(value) or not np.isfinite(value):
                raise ValueError(f"{name} must be a finite scalar")
        if not 0.0 <= float(self.min_match_iou) <= 1.0:
            raise ValueError("min_match_iou must lie in [0, 1]")
        if float(self.improvement_epsilon) < 0.0:
            raise ValueError("improvement_epsilon must be non-negative")
        for name in (
            "max_center_fraction",
            "max_log_dimension_residual",
            "runtime_minimum_extent",
            "near_iou50_band",
            "gain_cap",
            "max_center_shift_ratio",
            "min_extent_ratio",
            "max_extent_ratio",
            "min_original_point_support",
            "min_candidate_point_support",
            "min_reprojection_iou",
        ):
            value = getattr(self, name)
            if (
                not np.isscalar(value)
                or not np.isfinite(value)
                or float(value) <= 0.0
            ):
                raise ValueError(f"{name} must be positive and finite")
        for name in (
            "max_candidate_support_drop",
            "min_reprojection_improvement",
        ):
            value = getattr(self, name)
            if not np.isscalar(value) or not np.isfinite(value):
                raise ValueError(f"{name} must be finite")
        if not 0.0 <= float(self.max_candidate_support_drop) <= 1.0:
            raise ValueError(
                "max_candidate_support_drop must lie in [0, 1]"
            )
        if float(self.min_reprojection_improvement) < -1.0:
            raise ValueError(
                "min_reprojection_improvement must be at least -1"
            )
        for name in (
            "min_original_point_support",
            "min_candidate_point_support",
            "min_reprojection_iou",
        ):
            if float(getattr(self, name)) > 1.0:
                raise ValueError(f"{name} must not exceed one")
        if float(self.max_extent_ratio) < float(self.min_extent_ratio):
            raise ValueError(
                "max_extent_ratio cannot be below min_extent_ratio"
            )
        for name in (
            "expected_top_k_views",
            "min_runtime_views",
            "min_runtime_points",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(
                value, (int, np.integer)
            ):
                raise TypeError(f"{name} must be an integer")
            if int(value) <= 0:
                raise ValueError(f"{name} must be positive")
        if int(self.min_runtime_views) > int(self.expected_top_k_views):
            raise ValueError(
                "min_runtime_views cannot exceed expected_top_k_views"
            )
        if strict_schema:
            strict_config = {
                "top_k_views": int(self.expected_top_k_views),
                "refit_gate_min_views": int(self.min_runtime_views),
                "refit_gate_min_points": int(self.min_runtime_points),
                "output_minimum_extent": float(
                    self.runtime_minimum_extent
                ),
                "refit_gate_max_center_shift_ratio": float(
                    self.max_center_shift_ratio
                ),
                "refit_gate_min_extent_ratio": float(
                    self.min_extent_ratio
                ),
                "refit_gate_max_extent_ratio": float(
                    self.max_extent_ratio
                ),
                "refit_gate_min_original_point_support": float(
                    self.min_original_point_support
                ),
                "refit_gate_min_candidate_point_support": float(
                    self.min_candidate_point_support
                ),
                "refit_gate_max_candidate_support_drop": float(
                    self.max_candidate_support_drop
                ),
                "refit_gate_min_reprojection_iou": float(
                    self.min_reprojection_iou
                ),
                "refit_gate_min_reprojection_improvement": float(
                    self.min_reprojection_improvement
                ),
            }
            for name, actual in strict_config.items():
                expected = _STRICT_PROVENANCE_EXPECTED[name]
                if not _provenance_matches(actual, expected):
                    raise ValueError(
                        f"strict K5 builder {name}={actual!r}, "
                        f"expected runtime value {expected!r}"
                    )
        for name in (
            "gain_sample_weight",
            "cross_iou50_sample_weight",
            "near_iou50_sample_weight",
        ):
            value = getattr(self, name)
            if (
                not np.isscalar(value)
                or not np.isfinite(value)
                or float(value) < 0.0
            ):
                raise ValueError(f"{name} must be non-negative and finite")
        return self


@dataclass(frozen=True)
class SceneDiagnostics:
    scene_id: str
    quality_features: np.ndarray
    points: np.ndarray
    point_mask: np.ndarray
    result_indices: np.ndarray
    track_ids: np.ndarray
    top_k_views: int = 0
    selected_view_counts: np.ndarray | None = None
    selected_view_frame_ids: np.ndarray | None = None
    top_k_view_valid: np.ndarray | None = None
    local_boxes: np.ndarray | None = None
    frame_valid: np.ndarray | None = None
    gate_points_local: np.ndarray | None = None
    gate_point_mask: np.ndarray | None = None
    frame_centers: np.ndarray | None = None
    frame_basis: np.ndarray | None = None
    view_valid: np.ndarray | None = None
    view_frame_ids: np.ndarray | None = None
    view_scores: np.ndarray | None = None
    view_bboxes: np.ndarray | None = None
    view_intrinsics: np.ndarray | None = None
    view_camera_to_world: np.ndarray | None = None
    view_image_shapes: np.ndarray | None = None
    provenance: dict[str, object] | None = None


@dataclass(frozen=True)
class BuildSummary:
    scenes: int
    samples: int
    geometry_positives: int
    quality_negatives: int
    invalid_oriented_boxes: int
    output: Path
    cross_iou50_positives: int = 0


def read_scene_ids(path: Path) -> list[str]:
    """Read a non-empty, duplicate-free ScanNet scene list."""

    scenes = [
        line.strip() for line in Path(path).read_text().splitlines()
        if line.strip()
    ]
    if not scenes:
        raise ValueError(f"No scenes found in {path}")
    invalid = [scene for scene in scenes if SCENE_PATTERN.fullmatch(scene) is None]
    if invalid:
        raise ValueError(f"Invalid ScanNet scene id: {invalid[0]!r}")
    if len(set(scenes)) != len(scenes):
        raise ValueError(f"Duplicate scene ids in {path}")
    return scenes


def resolve_diagnostic_path(root: Path, scene_id: str) -> Path:
    """Resolve exactly one diagnostics archive for ``scene_id``."""

    matches = sorted(Path(root).glob(f"{scene_id}*.npz"))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected exactly one diagnostic NPZ for {scene_id} in {root}; "
            f"found {len(matches)}"
        )
    return matches[0]


def _parse_scalar_string(value: np.ndarray, name: str) -> str:
    array = np.asarray(value)
    if array.dtype.hasobject:
        raise ValueError(f"{name} must not use object dtype")
    if array.ndim == 0:
        scalar = array.item()
    elif array.ndim == 1 and array.size == 1:
        scalar = array[0]
    else:
        raise ValueError(f"{name} must be a scalar string")
    if isinstance(scalar, bytes):
        scalar = scalar.decode("utf-8")
    if not isinstance(scalar, str):
        raise TypeError(f"{name} must be a string")
    return scalar


def _parse_provenance_scalar(
    value: np.ndarray, name: str, expected: object
) -> object:
    array = np.asarray(value)
    if array.ndim != 0:
        raise ValueError(f"{name} must be scalar")
    scalar = array.item()
    if isinstance(expected, bool):
        if not isinstance(scalar, (bool, np.bool_)):
            raise TypeError(f"{name} must be Boolean")
        return bool(scalar)
    if isinstance(expected, int):
        if isinstance(scalar, (bool, np.bool_)) or not isinstance(
            scalar, (int, np.integer)
        ):
            raise TypeError(f"{name} must be an integer")
        return int(scalar)
    if isinstance(expected, float):
        if isinstance(scalar, (bool, np.bool_)) or not isinstance(
            scalar, (int, float, np.integer, np.floating)
        ):
            raise TypeError(f"{name} must be numeric")
        result = float(scalar)
        if not np.isfinite(result):
            raise ValueError(f"{name} must be finite")
        return result
    return _parse_scalar_string(array, name)


def _provenance_matches(actual: object, expected: object) -> bool:
    if isinstance(expected, float):
        return (
            not isinstance(actual, (bool, np.bool_))
            and isinstance(
                actual, (int, float, np.integer, np.floating)
            )
            and bool(
                np.isclose(float(actual), expected, atol=1e-8, rtol=0.0)
            )
        )
    if isinstance(expected, bool):
        return isinstance(actual, (bool, np.bool_)) and bool(actual) == expected
    if isinstance(expected, int):
        return (
            not isinstance(actual, (bool, np.bool_))
            and isinstance(actual, (int, np.integer))
            and int(actual) == expected
        )
    return actual == expected


def load_scene_diagnostics(
    path: Path,
    expected_scene_id: str | None = None,
    *,
    objective: str = "improvement",
    expected_top_k_views: int = 5,
    strict_k5_diagnostics: bool = False,
) -> SceneDiagnostics:
    """Load and strictly validate one pickle-free runtime diagnostic."""

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    if objective not in TRAINING_OBJECTIVES:
        raise ValueError(f"objective must be one of {TRAINING_OBJECTIVES}")
    strict_k5 = bool(strict_k5_diagnostics or objective == "ap50")
    if strict_k5 and int(expected_top_k_views) != 5:
        raise ValueError("strict K5 diagnostics require K exactly equal to 5")
    try:
        with np.load(path, allow_pickle=False) as payload:
            required = set(_REQUIRED_DIAGNOSTIC_FIELDS)
            if strict_k5:
                required.update(_STRICT_K5_DIAGNOSTIC_FIELDS)
            missing = required - set(payload.files)
            if missing:
                raise ValueError(f"{path} is missing fields: {sorted(missing)}")
            scene_id = _parse_scalar_string(payload["scene_id"], "scene_id")
            quality = np.asarray(payload["quality_features"])
            if strict_k5:
                points = np.asarray(payload["box_refiner_points_local"])
                point_mask = np.asarray(payload["box_refiner_point_mask"])
                local_boxes = np.asarray(
                    payload["box_refiner_local_boxes"]
                )
                frame_valid = np.asarray(
                    payload["box_refiner_frame_valid"]
                )
                gate_points = np.asarray(
                    payload["box_refiner_gate_points_local"]
                )
                gate_mask = np.asarray(
                    payload["box_refiner_gate_point_mask"]
                )
                frame_centers = np.asarray(
                    payload["box_refiner_frame_centers"]
                )
                frame_basis = np.asarray(
                    payload["box_refiner_frame_basis"]
                )
                view_valid = np.asarray(
                    payload["box_refiner_view_valid"]
                )
                view_frame_ids = np.asarray(
                    payload["box_refiner_view_frame_ids"]
                )
                view_scores = np.asarray(
                    payload["box_refiner_view_scores"]
                )
                view_bboxes = np.asarray(
                    payload["box_refiner_view_bboxes"]
                )
                view_intrinsics = np.asarray(
                    payload["box_refiner_view_intrinsics"]
                )
                view_camera_to_world = np.asarray(
                    payload["box_refiner_view_camera_to_world"]
                )
                view_image_shapes = np.asarray(
                    payload["box_refiner_view_image_shapes"]
                )
                selected_view_counts = np.asarray(
                    payload["selected_view_counts"]
                )
                selected_view_frame_ids = np.asarray(
                    payload["selected_view_frame_ids"]
                )
                top_k_view_valid = np.asarray(
                    payload["top_k_view_valid"]
                )
                provenance = {
                    name: _parse_provenance_scalar(
                        payload[name], name, expected
                    )
                    for name, expected in _STRICT_PROVENANCE_EXPECTED.items()
                }
                for name, expected in _STRICT_PROVENANCE_EXPECTED.items():
                    if not _provenance_matches(provenance[name], expected):
                        raise ValueError(
                            f"strict runtime provenance {name}="
                            f"{provenance[name]!r}, expected {expected!r}"
                        )
                summary_text = _parse_scalar_string(
                    payload["summary_json"], "summary_json"
                )
                try:
                    summary = json.loads(summary_text)
                except json.JSONDecodeError as error:
                    raise ValueError("summary_json is invalid JSON") from error
                if not isinstance(summary, dict):
                    raise TypeError("summary_json must encode a mapping")
                for name, expected in _STRICT_PROVENANCE_EXPECTED.items():
                    if name not in summary:
                        raise ValueError(
                            f"summary_json is missing provenance {name}"
                        )
                    if not _provenance_matches(summary[name], expected):
                        raise ValueError(
                            f"summary_json provenance {name} is invalid"
                        )
            elif {
                "geometry_points",
                "geometry_point_mask",
            } <= set(payload.files):
                points = np.asarray(payload["geometry_points"])
                point_mask = np.asarray(payload["geometry_point_mask"])
            elif {"points", "point_mask"} <= set(payload.files):
                points = np.asarray(payload["points"])
                point_mask = np.asarray(payload["point_mask"])
            else:
                raise ValueError(
                    f"{path} requires geometry_points/geometry_point_mask "
                    "or points/point_mask"
                )
            result_indices = np.asarray(payload["result_indices"])
            track_ids = (
                np.asarray(payload["track_ids"])
                if "track_ids" in payload.files
                else np.arange(len(points), dtype=np.int64)
            )
            top_k_views = 5 if strict_k5 else 0
            if not strict_k5:
                selected_view_counts = None
            if "quality_feature_names" in payload.files:
                names = tuple(
                    str(item)
                    for item in np.asarray(
                        payload["quality_feature_names"]
                    ).tolist()
                )
                if names != QUALITY_FEATURE_NAMES:
                    raise ValueError(
                        "diagnostic quality feature schema/order mismatch"
                    )
    except ValueError as error:
        if "Object arrays cannot be loaded" in str(error):
            raise ValueError(f"{path} contains pickled/object arrays") from error
        raise

    if SCENE_PATTERN.fullmatch(scene_id) is None:
        raise ValueError(f"Invalid diagnostic scene_id: {scene_id!r}")
    if expected_scene_id is not None and scene_id != expected_scene_id:
        raise ValueError(
            f"Diagnostic scene_id {scene_id!r} does not match "
            f"requested scene {expected_scene_id!r}"
        )
    if points.ndim != 3 or points.shape[2] != 3 or points.shape[1] < 1:
        raise ValueError("points must have shape [N, P, 3] with P > 0")
    sample_count, point_count, _ = points.shape
    if not np.issubdtype(points.dtype, np.floating):
        raise TypeError("points must use floating-point dtype")
    if strict_k5 and points.dtype != np.float32:
        raise TypeError("box_refiner_points_local must use float32")
    if not strict_k5 and not np.isfinite(points).all():
        raise ValueError("points must be finite")
    points = np.asarray(points, dtype=np.float64)
    if point_mask.shape != (sample_count, point_count):
        raise ValueError("point_mask must have shape [N, P]")
    if point_mask.dtype != np.bool_:
        raise TypeError("point_mask must have Boolean dtype")
    if not strict_k5 and sample_count and not point_mask.any(axis=1).all():
        raise ValueError("every sample must contain at least one valid point")
    if strict_k5:
        if point_count != 512:
            raise ValueError("box_refiner_points_local must use 512 points")
        expected_gate_shape = (sample_count, 8192, 3)
        if (
            gate_points.shape != expected_gate_shape
            or gate_points.dtype != np.float32
        ):
            raise TypeError(
                "box_refiner_gate_points_local must be float32 with shape "
                f"{expected_gate_shape}"
            )
        if (
            gate_mask.shape != expected_gate_shape[:-1]
            or gate_mask.dtype != np.bool_
        ):
            raise TypeError(
                "box_refiner_gate_point_mask must be Boolean with shape "
                f"{expected_gate_shape[:-1]}"
            )
        if local_boxes.shape != (sample_count, 6):
            raise ValueError(
                "box_refiner_local_boxes must have shape [N, 6]"
            )
        if local_boxes.dtype != np.float32:
            raise TypeError("box_refiner_local_boxes must use float32")
        if (
            frame_valid.shape != (sample_count,)
            or frame_valid.dtype != np.bool_
        ):
            raise TypeError(
                "box_refiner_frame_valid must be Boolean with shape [N]"
            )
        if frame_centers.shape != (sample_count, 3):
            raise ValueError(
                "box_refiner_frame_centers must have shape [N, 3]"
            )
        if frame_centers.dtype != np.float64:
            raise TypeError("box_refiner_frame_centers must use float64")
        if frame_basis.shape != (sample_count, 3, 3):
            raise ValueError(
                "box_refiner_frame_basis must have shape [N, 3, 3]"
            )
        if frame_basis.dtype != np.float64:
            raise TypeError("box_refiner_frame_basis must use float64")
        for row in range(sample_count):
            valid_frame = bool(frame_valid[row])
            if valid_frame:
                if not point_mask[row].any():
                    raise ValueError(
                        "valid box-refiner frame has no model input points"
                    )
                if (
                    not np.isfinite(points[row, point_mask[row]]).all()
                    or not np.all(points[row, ~point_mask[row]] == 0.0)
                    or not np.isfinite(
                        gate_points[row, gate_mask[row]]
                    ).all()
                    or not np.all(
                        gate_points[row, ~gate_mask[row]] == 0.0
                    )
                ):
                    raise ValueError(
                        "box-refiner point padding/value contract is invalid"
                    )
                if (
                    not np.isfinite(local_boxes[row]).all()
                    or not np.allclose(
                        local_boxes[row, :3], 0.0, atol=1e-7
                    )
                    or np.any(local_boxes[row, 3:6] <= 0.0)
                    or not np.isfinite(frame_centers[row]).all()
                    or not np.isfinite(frame_basis[row]).all()
                    or not np.allclose(
                        frame_basis[row].T @ frame_basis[row],
                        np.eye(3),
                        atol=2e-3,
                    )
                    or np.linalg.det(frame_basis[row]) <= 0.0
                ):
                    raise ValueError(
                        "valid box-refiner local frame contract is invalid"
                    )
            else:
                if point_mask[row].any() or gate_mask[row].any():
                    raise ValueError(
                        "invalid box-refiner frame must have empty masks"
                    )
                if not (
                    np.isnan(local_boxes[row]).all()
                    and np.isnan(frame_centers[row]).all()
                    and np.isnan(frame_basis[row]).all()
                ):
                    raise ValueError(
                        "invalid box-refiner frame must use NaN frame values"
                    )
    if quality.shape != (sample_count, QUALITY_FEATURE_DIM):
        raise ValueError(
            f"quality_features must have shape [N, {QUALITY_FEATURE_DIM}]"
        )
    if not np.issubdtype(quality.dtype, np.floating):
        raise TypeError("quality_features must use floating-point dtype")
    quality = np.asarray(quality, dtype=np.float64)
    if (
        not np.isfinite(quality).all()
        or (quality < 0.0).any()
        or (quality > 1.0).any()
    ):
        raise ValueError("quality_features must be finite and lie in [0, 1]")
    for name, value in (
        ("result_indices", result_indices),
        ("track_ids", track_ids),
    ):
        if value.shape != (sample_count,):
            raise ValueError(f"{name} must have shape [N]")
        if not np.issubdtype(value.dtype, np.integer):
            raise TypeError(f"{name} must use integer dtype")
    result_indices = np.asarray(result_indices, dtype=np.int64)
    track_ids = np.asarray(track_ids, dtype=np.int64)
    if (result_indices < 0).any():
        raise ValueError("result_indices must be non-negative")
    if len(np.unique(result_indices)) != sample_count:
        raise ValueError("result_indices must be unique within a scene")
    if strict_k5:
        expected_view_shape = (sample_count, 5)
        expected_shapes = {
            "box_refiner_view_valid": (view_valid, expected_view_shape),
            "box_refiner_view_frame_ids": (
                view_frame_ids,
                expected_view_shape,
            ),
            "box_refiner_view_scores": (view_scores, expected_view_shape),
            "box_refiner_view_bboxes": (
                view_bboxes,
                expected_view_shape + (4,),
            ),
            "box_refiner_view_intrinsics": (
                view_intrinsics,
                expected_view_shape + (3, 3),
            ),
            "box_refiner_view_camera_to_world": (
                view_camera_to_world,
                expected_view_shape + (4, 4),
            ),
            "box_refiner_view_image_shapes": (
                view_image_shapes,
                expected_view_shape + (2,),
            ),
        }
        for name, (value, shape) in expected_shapes.items():
            if value.shape != shape:
                raise ValueError(f"{name} must have shape {shape}")
        if view_valid.dtype != np.bool_:
            raise TypeError("box_refiner_view_valid must be Boolean")
        if not np.issubdtype(view_frame_ids.dtype, np.integer):
            raise TypeError("box_refiner_view_frame_ids must be integer")
        if not np.issubdtype(view_image_shapes.dtype, np.integer):
            raise TypeError("box_refiner_view_image_shapes must be integer")
        if (
            selected_view_counts.shape != (sample_count,)
            or not np.issubdtype(
                selected_view_counts.dtype, np.integer
            )
        ):
            raise TypeError(
                "selected_view_counts must be integer with shape [N]"
            )
        if (
            selected_view_frame_ids.shape != expected_view_shape
            or not np.issubdtype(
                selected_view_frame_ids.dtype, np.integer
            )
        ):
            raise TypeError(
                "selected_view_frame_ids must be integer with shape [N,5]"
            )
        if (
            top_k_view_valid.shape != expected_view_shape
            or top_k_view_valid.dtype != np.bool_
        ):
            raise TypeError(
                "top_k_view_valid must be Boolean with shape [N,5]"
            )
        if not np.array_equal(
            selected_view_counts,
            top_k_view_valid.sum(axis=1),
        ):
            raise ValueError(
                "selected_view_counts disagree with top_k_view_valid"
            )
        if (
            (selected_view_counts < 0).any()
            or (selected_view_counts > 5).any()
        ):
            raise ValueError("selected_view_counts must lie in [0,5]")
        for name, value in (
            ("box_refiner_view_scores", view_scores),
            ("box_refiner_view_bboxes", view_bboxes),
            ("box_refiner_view_intrinsics", view_intrinsics),
            (
                "box_refiner_view_camera_to_world",
                view_camera_to_world,
            ),
        ):
            if value.dtype != np.float32:
                raise TypeError(f"{name} must use float32")
        for row in range(sample_count):
            selected_valid = top_k_view_valid[row]
            selected_invalid = ~selected_valid
            selected_ids = selected_view_frame_ids[row, selected_valid]
            if (
                (selected_ids < 0).any()
                or len(np.unique(selected_ids)) != len(selected_ids)
                or not np.all(
                    selected_view_frame_ids[row, selected_invalid] == -1
                )
            ):
                raise ValueError(
                    "selected Top-K frame ids violate validity contract"
                )
            valid_slots = view_valid[row]
            invalid_slots = ~valid_slots
            if valid_slots.any():
                if (
                    (view_frame_ids[row, valid_slots] < 0).any()
                    or not np.isfinite(view_scores[row, valid_slots]).all()
                    or (
                        (view_scores[row, valid_slots] < 0.0)
                        | (view_scores[row, valid_slots] > 1.0)
                    ).any()
                    or not np.isfinite(view_bboxes[row, valid_slots]).all()
                    or not np.isfinite(
                        view_intrinsics[row, valid_slots]
                    ).all()
                    or not np.isfinite(
                        view_camera_to_world[row, valid_slots]
                    ).all()
                    or (view_image_shapes[row, valid_slots] <= 0).any()
                ):
                    raise ValueError("valid camera/view evidence is invalid")
                for slot in np.flatnonzero(valid_slots):
                    height, width = view_image_shapes[row, slot]
                    bbox = view_bboxes[row, slot]
                    if not (
                        0.0 <= bbox[0] < bbox[2] <= float(width)
                        and 0.0 <= bbox[1] < bbox[3] <= float(height)
                    ):
                        raise ValueError("view bbox is outside image bounds")
                    intrinsics = view_intrinsics[row, slot]
                    if intrinsics[0, 0] <= 0.0 or intrinsics[1, 1] <= 0.0:
                        raise ValueError("view intrinsics are invalid")
                    pose = view_camera_to_world[row, slot]
                    if not np.allclose(
                        pose[3], [0.0, 0.0, 0.0, 1.0], atol=1e-6
                    ):
                        raise ValueError(
                            "camera_to_world must be homogeneous"
                        )
                    if abs(float(np.linalg.det(pose[:3, :3]))) < 1e-6:
                        raise ValueError("camera_to_world is singular")
            if invalid_slots.any():
                if (
                    not np.all(view_frame_ids[row, invalid_slots] == -1)
                    or not np.all(view_image_shapes[row, invalid_slots] == -1)
                    or not np.isnan(view_scores[row, invalid_slots]).all()
                    or not np.isnan(view_bboxes[row, invalid_slots]).all()
                    or not np.isnan(
                        view_intrinsics[row, invalid_slots]
                    ).all()
                    or not np.isnan(
                        view_camera_to_world[row, invalid_slots]
                    ).all()
                ):
                    raise ValueError(
                        "invalid camera/view slots violate sentinel contract"
                    )

    return SceneDiagnostics(
        scene_id=scene_id,
        quality_features=np.ascontiguousarray(quality),
        points=np.ascontiguousarray(points),
        point_mask=np.ascontiguousarray(point_mask),
        result_indices=np.ascontiguousarray(result_indices),
        track_ids=np.ascontiguousarray(track_ids),
        top_k_views=top_k_views,
        selected_view_counts=(
            None
            if selected_view_counts is None
            else np.ascontiguousarray(
                selected_view_counts, dtype=np.int64
            )
        ),
        selected_view_frame_ids=(
            np.ascontiguousarray(
                selected_view_frame_ids, dtype=np.int64
            )
            if strict_k5
            else None
        ),
        top_k_view_valid=(
            np.ascontiguousarray(top_k_view_valid)
            if strict_k5
            else None
        ),
        local_boxes=(
            np.ascontiguousarray(local_boxes, dtype=np.float64)
            if strict_k5
            else None
        ),
        frame_valid=(
            np.ascontiguousarray(frame_valid) if strict_k5 else None
        ),
        gate_points_local=(
            np.ascontiguousarray(gate_points, dtype=np.float64)
            if strict_k5
            else None
        ),
        gate_point_mask=(
            np.ascontiguousarray(gate_mask) if strict_k5 else None
        ),
        frame_centers=(
            np.ascontiguousarray(frame_centers, dtype=np.float64)
            if strict_k5
            else None
        ),
        frame_basis=(
            np.ascontiguousarray(frame_basis, dtype=np.float64)
            if strict_k5
            else None
        ),
        view_valid=(
            np.ascontiguousarray(view_valid) if strict_k5 else None
        ),
        view_frame_ids=(
            np.ascontiguousarray(view_frame_ids, dtype=np.int64)
            if strict_k5
            else None
        ),
        view_scores=(
            np.ascontiguousarray(view_scores, dtype=np.float64)
            if strict_k5
            else None
        ),
        view_bboxes=(
            np.ascontiguousarray(view_bboxes, dtype=np.float64)
            if strict_k5
            else None
        ),
        view_intrinsics=(
            np.ascontiguousarray(view_intrinsics, dtype=np.float64)
            if strict_k5
            else None
        ),
        view_camera_to_world=(
            np.ascontiguousarray(
                view_camera_to_world, dtype=np.float64
            )
            if strict_k5
            else None
        ),
        view_image_shapes=(
            np.ascontiguousarray(view_image_shapes, dtype=np.int64)
            if strict_k5
            else None
        ),
        provenance=provenance if strict_k5 else None,
    )


def load_prediction_detections(
    path: Path,
) -> tuple[np.ndarray, np.ndarray]:
    """Load trusted BoxFusion corners and real detector scores.

    Pickle is intentionally confined to this offline data-preparation tool.
    Never use this loader on an untrusted prediction file.
    """

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    if not isinstance(payload, (list, tuple)):
        raise ValueError(f"Invalid BoxFusion payload in {path}")
    detections = payload[0] if payload else []
    if not isinstance(detections, (list, tuple)):
        raise ValueError(f"Invalid detection list in {path}")
    if not detections:
        return (
            np.empty((0, 8, 3), dtype=np.float64),
            np.empty(0, dtype=np.float64),
        )
    corners = []
    scores = []
    for index, item in enumerate(detections):
        if not isinstance(item, (list, tuple)) or len(item) < 3:
            raise ValueError(f"Invalid detection {index} in {path}")
        value = np.asarray(item[1])
        if value.shape != (8, 3) or not np.issubdtype(
            value.dtype, np.number
        ):
            raise ValueError(f"Detection {index} corners must have shape [8,3]")
        value = np.asarray(value, dtype=np.float64)
        if not np.isfinite(value).all():
            raise ValueError(f"Detection {index} corners must be finite")
        corners.append(value)
        score_array = np.asarray(item[2])
        if (
            score_array.ndim != 0
            or not np.issubdtype(score_array.dtype, np.number)
        ):
            raise ValueError(
                f"Detection {index} score must be a numeric scalar"
            )
        score = float(score_array)
        if not np.isfinite(score) or not 0.0 <= score <= 1.0:
            raise ValueError(
                f"Detection {index} score must be finite in [0, 1]"
            )
        scores.append(score)
    return (
        np.stack(corners, axis=0),
        np.asarray(scores, dtype=np.float64),
    )


def load_prediction_corners(path: Path) -> np.ndarray:
    """Backward-compatible corners-only wrapper."""

    return load_prediction_detections(path)[0]


def load_axis_alignment(scan_root: Path, scene_id: str) -> np.ndarray:
    """Load a rigid ScanNet ``axisAlignment`` transform."""

    metadata = Path(scan_root) / scene_id / f"{scene_id}.txt"
    if not metadata.is_file():
        raise FileNotFoundError(metadata)
    values = None
    for line in metadata.read_text().splitlines():
        stripped = line.strip()
        if stripped.startswith("axisAlignment"):
            if "=" not in stripped:
                raise ValueError(f"Malformed axisAlignment in {metadata}")
            values = np.fromstring(stripped.split("=", 1)[1], sep=" ")
            break
    if values is None or values.size != 16 or not np.isfinite(values).all():
        raise ValueError(f"Invalid or missing axisAlignment in {metadata}")
    transform = values.reshape(4, 4).astype(np.float64, copy=False)
    if not np.allclose(
        transform[3], np.asarray([0.0, 0.0, 0.0, 1.0]), atol=1e-6
    ):
        raise ValueError(f"axisAlignment is not homogeneous in {metadata}")
    rotation = transform[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=2e-3):
        raise ValueError(f"axisAlignment is not rigid in {metadata}")
    if not np.isclose(abs(np.linalg.det(rotation)), 1.0, atol=2e-3):
        raise ValueError(f"axisAlignment rotation is singular in {metadata}")
    return transform


def load_gt_boxes(gt_root: Path, scene_id: str) -> np.ndarray:
    """Load aligned ScanNet GT boxes as ``[cx,cy,cz,dx,dy,dz]``."""

    path = Path(gt_root) / f"{scene_id}_bbox.npy"
    if not path.is_file():
        raise FileNotFoundError(path)
    boxes = np.load(path, allow_pickle=False)
    if boxes.ndim != 2 or boxes.shape[1] < 6:
        raise ValueError(f"GT boxes in {path} must have shape [N, >=6]")
    boxes = np.asarray(boxes[:, :6], dtype=np.float64)
    if (
        not np.isfinite(boxes).all()
        or (boxes[:, 3:6] <= 0.0).any()
    ):
        raise ValueError(f"GT boxes in {path} are invalid")
    return boxes


def oriented_box_frame(
    corners: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Recover ``center, dimensions, basis`` from BoxFusion corner ordering.

    Basis vectors are columns and preserve the exact upstream yaw/sign
    convention.  The incident edges at corner zero are ``1-0``, ``3-0`` and
    ``4-0``.
    """

    value = np.asarray(corners, dtype=np.float64)
    if value.shape != (8, 3) or not np.isfinite(value).all():
        raise ValueError("corners must be a finite [8, 3] array")
    edges = np.stack(
        (value[1] - value[0], value[3] - value[0], value[4] - value[0]),
        axis=1,
    )
    dimensions = np.linalg.norm(edges, axis=0)
    if (dimensions <= 1e-6).any():
        raise ValueError("oriented box has a degenerate edge")
    basis = edges / dimensions[None, :]
    if not np.allclose(basis.T @ basis, np.eye(3), atol=2e-3):
        raise ValueError("oriented box edges are not orthogonal")
    if np.linalg.det(basis) <= 0.0:
        raise ValueError("oriented box basis must be right handed")
    center = value.mean(axis=0)
    local = (value - center) @ basis
    expected_half = dimensions * 0.5
    if not np.allclose(np.abs(local), expected_half[None, :], atol=2e-3):
        raise ValueError("corners do not form the recovered oriented box")
    signs = np.sign(local)
    if len({tuple(row.tolist()) for row in signs}) != 8:
        raise ValueError("oriented box corner signs are not unique")
    return center, dimensions, basis


def nonnegative_least_squares_3x3(
    matrix: np.ndarray, target: np.ndarray
) -> np.ndarray:
    """Solve a three-variable NNLS problem by enumerating active sets."""

    matrix = np.asarray(matrix, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if matrix.shape != (3, 3) or target.shape != (3,):
        raise ValueError("NNLS expects a [3,3] matrix and [3] target")
    if not np.isfinite(matrix).all() or not np.isfinite(target).all():
        raise ValueError("NNLS inputs must be finite")
    best = np.zeros(3, dtype=np.float64)
    best_error = float(np.dot(target, target))
    # The empty active set above is a valid non-negative candidate.
    for mask in range(1, 1 << 3):
        active = [index for index in range(3) if mask & (1 << index)]
        solution, _, _, _ = np.linalg.lstsq(
            matrix[:, active], target, rcond=None
        )
        if (solution < -1e-10).any():
            continue
        candidate = np.zeros(3, dtype=np.float64)
        candidate[active] = np.maximum(solution, 0.0)
        error = float(np.sum((matrix @ candidate - target) ** 2))
        if error < best_error - 1e-12:
            best = candidate
            best_error = error
    return best


def center_size_to_minmax(boxes: np.ndarray) -> np.ndarray:
    value = np.asarray(boxes, dtype=np.float64)
    if value.ndim != 2 or value.shape[1] != 6:
        raise ValueError("boxes must have shape [N, 6]")
    half = value[:, 3:6] * 0.5
    return np.concatenate((value[:, :3] - half, value[:, :3] + half), axis=1)


def pairwise_aabb_iou(predictions: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """Compute pairwise IoU between centre/size AABBs."""

    pred = center_size_to_minmax(predictions)
    target = center_size_to_minmax(targets)
    if len(pred) == 0 or len(target) == 0:
        return np.zeros((len(pred), len(target)), dtype=np.float64)
    intersection_min = np.maximum(pred[:, None, :3], target[None, :, :3])
    intersection_max = np.minimum(pred[:, None, 3:], target[None, :, 3:])
    intersection_size = np.maximum(intersection_max - intersection_min, 0.0)
    intersection = np.prod(intersection_size, axis=2)
    pred_volume = np.prod(pred[:, 3:] - pred[:, :3], axis=1)
    target_volume = np.prod(target[:, 3:] - target[:, :3], axis=1)
    union = pred_volume[:, None] + target_volume[None, :] - intersection
    return np.divide(
        intersection,
        union,
        out=np.zeros_like(intersection),
        where=union > 0.0,
    )


def _local_box_to_world_corners(
    local_box: np.ndarray,
    frame_center: np.ndarray,
    frame_basis: np.ndarray,
) -> np.ndarray:
    box = np.asarray(local_box, dtype=np.float64)
    center = np.asarray(frame_center, dtype=np.float64)
    basis = np.asarray(frame_basis, dtype=np.float64)
    if (
        box.shape != (6,)
        or center.shape != (3,)
        or basis.shape != (3, 3)
        or not np.isfinite(box).all()
        or not np.isfinite(center).all()
        or not np.isfinite(basis).all()
        or np.any(box[3:6] <= 0.0)
    ):
        raise ValueError("invalid local oriented box frame")
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
    local_corners = (
        box[:3][None, :] + signs * box[3:6][None, :] * 0.5
    ).astype(np.float32)
    # Runtime ``aabb_corners`` rounds the local corners to float32, performs
    # the world transform in float64, then stores exported corners as
    # float32. Preserve that path so zero-margin reprojection gates agree.
    return (
        center[None, :]
        + local_corners.astype(np.float64) @ basis.T
    ).astype(np.float32)


def _runtime_corners_to_center_size(corners: np.ndarray) -> np.ndarray:
    """Mirror ``online_refinement.corners_to_center_size`` for gate replay."""

    value = np.asarray(corners, dtype=np.float32)
    if value.shape != (8, 3) or not np.isfinite(value).all():
        raise ValueError("runtime corners must have finite shape [8, 3]")
    minimum = value.min(axis=0)
    maximum = value.max(axis=0)
    dimensions = maximum - minimum
    if np.any(dimensions <= 0.0):
        raise ValueError("runtime corners must have positive world extent")
    return np.concatenate((0.5 * (minimum + maximum), dimensions))


def _corners_to_center_size(corners: np.ndarray) -> np.ndarray:
    value = np.asarray(corners, dtype=np.float64)
    if value.shape != (8, 3) or not np.isfinite(value).all():
        raise ValueError("corners must have finite shape [8, 3]")
    minimum = value.min(axis=0)
    maximum = value.max(axis=0)
    dimensions = maximum - minimum
    if np.any(dimensions <= 0.0):
        raise ValueError("corners must have positive world AABB extent")
    return np.concatenate((0.5 * (minimum + maximum), dimensions))


def _transform_corners(corners: np.ndarray, transform: np.ndarray) -> np.ndarray:
    value = np.asarray(corners, dtype=np.float64)
    matrix = np.asarray(transform, dtype=np.float64)
    if value.shape != (8, 3) or matrix.shape != (4, 4):
        raise ValueError("corner transform shape mismatch")
    return value @ matrix[:3, :3].T + matrix[:3, 3][None, :]


def _points_inside_aabb_fraction(
    points: np.ndarray,
    center: np.ndarray,
    dimensions: np.ndarray,
) -> float:
    value = np.asarray(points, dtype=np.float64)
    if value.ndim != 2 or value.shape[1] != 3:
        raise ValueError("support points must have shape [N, 3]")
    if not len(value):
        return 0.0
    half = np.asarray(dimensions, dtype=np.float64) * 0.5
    center = np.asarray(center, dtype=np.float64)
    inside = np.all(
        (value >= center[None, :] - half[None, :])
        & (value <= center[None, :] + half[None, :]),
        axis=1,
    )
    return float(np.mean(inside))


def _bbox_iou_2d(box_a: np.ndarray, box_b: np.ndarray) -> float:
    first = np.asarray(box_a, dtype=np.float64)
    second = np.asarray(box_b, dtype=np.float64)
    intersection_min = np.maximum(first[:2], second[:2])
    intersection_max = np.minimum(first[2:], second[2:])
    intersection_dims = np.maximum(intersection_max - intersection_min, 0.0)
    intersection = float(np.prod(intersection_dims))
    area_a = float(np.prod(np.maximum(first[2:] - first[:2], 0.0)))
    area_b = float(np.prod(np.maximum(second[2:] - second[:2], 0.0)))
    union = area_a + area_b - intersection
    return float(intersection / union) if union > 0.0 else 0.0


def _projection_iou_for_corners(
    corners: np.ndarray,
    bbox: np.ndarray,
    intrinsics: np.ndarray,
    camera_to_world: np.ndarray,
    image_shape: np.ndarray,
) -> float:
    values = np.asarray(corners, dtype=np.float64)
    # ViewEvidence stores pose/K/bbox as float32. In particular, NumPy keeps
    # the inverse of a float32 pose in float32; casting before the inverse
    # creates small sign flips at the runtime's zero-delta reprojection gate.
    pose = np.asarray(camera_to_world, dtype=np.float32)
    camera_matrix = np.asarray(intrinsics, dtype=np.float32)
    world_to_camera = np.linalg.inv(pose)
    homogeneous = np.column_stack(
        (values, np.ones(8, dtype=np.float64))
    )
    camera = (homogeneous @ world_to_camera.T)[:, :3]
    in_front = camera[:, 2] > 1e-3
    if not np.any(in_front):
        return 0.0
    camera = camera[in_front]
    projected = camera @ camera_matrix.T
    pixels = projected[:, :2] / projected[:, 2:3]
    height, width = np.asarray(image_shape, dtype=np.int64)
    x = np.clip(pixels[:, 0], 0.0, float(width))
    y = np.clip(pixels[:, 1], 0.0, float(height))
    projected_box = np.asarray(
        [x.min(), y.min(), x.max(), y.max()], dtype=np.float32
    )
    if (
        projected_box[2] <= projected_box[0]
        or projected_box[3] <= projected_box[1]
    ):
        return 0.0
    return _bbox_iou_2d(projected_box, bbox)


def _mean_reprojection_iou(
    corners: np.ndarray,
    view_valid: np.ndarray,
    view_scores: np.ndarray,
    view_bboxes: np.ndarray,
    view_intrinsics: np.ndarray,
    view_camera_to_world: np.ndarray,
    view_image_shapes: np.ndarray,
) -> float:
    valid = np.asarray(view_valid, dtype=np.bool_)
    if not valid.any():
        return 0.0
    indices = np.flatnonzero(valid)
    values = np.asarray(
        [
            _projection_iou_for_corners(
                corners,
                view_bboxes[index],
                view_intrinsics[index],
                view_camera_to_world[index],
                view_image_shapes[index],
            )
            for index in indices
        ],
        dtype=np.float32,
    )
    weights = np.asarray(
        [max(float(view_scores[index]), 1e-4) for index in indices],
        dtype=np.float32,
    )
    return float(np.average(values, weights=weights))


def runtime_refit_gate(
    original_local_box: np.ndarray,
    candidate_local_box: np.ndarray,
    *,
    gate_points_local: np.ndarray,
    gate_point_mask: np.ndarray,
    frame_center: np.ndarray,
    frame_basis: np.ndarray,
    selected_view_frame_ids: np.ndarray,
    top_k_view_valid: np.ndarray,
    view_valid: np.ndarray,
    view_frame_ids: np.ndarray,
    view_scores: np.ndarray,
    view_bboxes: np.ndarray,
    view_intrinsics: np.ndarray,
    view_camera_to_world: np.ndarray,
    view_image_shapes: np.ndarray,
    config: BuildConfig,
) -> tuple[bool, str, dict[str, float]]:
    """Reproduce the runtime oriented B5-v2 ``_refit_gate`` exactly."""

    original = np.asarray(original_local_box, dtype=np.float32)
    candidate = np.asarray(candidate_local_box, dtype=np.float32)
    metrics: dict[str, float] = {}
    selected_valid = np.asarray(top_k_view_valid, dtype=np.bool_)
    if int(np.count_nonzero(selected_valid)) < int(
        config.min_runtime_views
    ):
        return False, "views", metrics
    selected_ids = np.asarray(
        selected_view_frame_ids, dtype=np.int64
    )[selected_valid]
    record_valid = np.asarray(view_valid, dtype=np.bool_) & np.isin(
        np.asarray(view_frame_ids, dtype=np.int64), selected_ids
    )
    point_mask = np.asarray(gate_point_mask, dtype=np.bool_)
    if int(np.count_nonzero(point_mask)) < int(config.min_runtime_points):
        return False, "points", metrics
    if (
        candidate.shape != (6,)
        or not np.isfinite(candidate).all()
        or np.any(candidate[3:6] <= 0.0)
    ):
        return False, "invalid", metrics
    original_corners = _local_box_to_world_corners(
        original, frame_center, frame_basis
    )
    candidate_corners = _local_box_to_world_corners(
        candidate, frame_center, frame_basis
    )
    original_world_box = _runtime_corners_to_center_size(original_corners)
    candidate_world_box = _runtime_corners_to_center_size(candidate_corners)
    minimum_extent = float(config.runtime_minimum_extent)
    if minimum_extent > 0.0:
        original_survives = bool(
            np.all(original_world_box[3:6] >= minimum_extent)
        )
        candidate_survives = bool(
            np.all(candidate_world_box[3:6] >= minimum_extent)
        )
        metrics["original_survives"] = float(original_survives)
        metrics["candidate_survives"] = float(candidate_survives)
        if original_survives != candidate_survives:
            return False, "extent_filter", metrics
    diagonal = max(float(np.linalg.norm(original[3:6])), 1e-6)
    shift_ratio = float(
        np.linalg.norm(candidate[:3] - original[:3]) / diagonal
    )
    metrics["center_shift_ratio"] = shift_ratio
    if shift_ratio > float(config.max_center_shift_ratio):
        return False, "center_shift", metrics
    extent_ratio = candidate[3:6] / np.maximum(original[3:6], 1e-6)
    metrics["min_extent_ratio"] = float(extent_ratio.min())
    metrics["max_extent_ratio"] = float(extent_ratio.max())
    if np.any(extent_ratio < float(config.min_extent_ratio)) or np.any(
        extent_ratio > float(config.max_extent_ratio)
    ):
        return False, "extent", metrics
    support_points = np.asarray(
        gate_points_local, dtype=np.float64
    )[point_mask]
    original_support = _points_inside_aabb_fraction(
        support_points, original[:3], original[3:6]
    )
    metrics["original_support"] = original_support
    if original_support < float(config.min_original_point_support):
        return False, "support", metrics
    candidate_support = _points_inside_aabb_fraction(
        support_points, candidate[:3], candidate[3:6]
    )
    metrics["candidate_support"] = candidate_support
    if candidate_support < float(config.min_candidate_point_support):
        return False, "candidate_support", metrics
    support_drop = original_support - candidate_support
    metrics["candidate_support_drop"] = support_drop
    if support_drop > float(config.max_candidate_support_drop):
        return False, "candidate_support_drop", metrics
    original_projection = _mean_reprojection_iou(
        original_corners,
        record_valid,
        view_scores,
        view_bboxes,
        view_intrinsics,
        view_camera_to_world,
        view_image_shapes,
    )
    candidate_projection = _mean_reprojection_iou(
        candidate_corners,
        record_valid,
        view_scores,
        view_bboxes,
        view_intrinsics,
        view_camera_to_world,
        view_image_shapes,
    )
    metrics["original_reprojection_iou"] = original_projection
    metrics["candidate_reprojection_iou"] = candidate_projection
    if candidate_projection < float(config.min_reprojection_iou):
        return False, "reprojection", metrics
    reprojection_delta = candidate_projection - original_projection
    metrics["reprojection_improvement"] = reprojection_delta
    if reprojection_delta < float(config.min_reprojection_improvement):
        return False, "reprojection_delta", metrics
    return True, "accepted", metrics


def greedy_scene_tp50_flags(
    boxes: np.ndarray,
    scores: np.ndarray,
    gt_boxes: np.ndarray,
) -> np.ndarray:
    """Return evaluator-style per-prediction TP50 flags for one scene.

    A duplicate prediction remains a false positive even if it overlaps an
    unoccupied second-best target: the evaluator first chooses the maximum-IoU
    target and then checks whether that target is already occupied.
    """

    predictions = np.asarray(boxes, dtype=np.float64)
    confidence = np.asarray(scores, dtype=np.float64)
    targets = np.asarray(gt_boxes, dtype=np.float64)
    if predictions.ndim != 2 or predictions.shape[1] != 6:
        raise ValueError("boxes must have shape [N, 6]")
    if confidence.shape != (len(predictions),):
        raise ValueError("scores must have shape [N]")
    flags = np.zeros(len(predictions), dtype=np.bool_)
    if not len(predictions) or not len(targets):
        return flags
    overlaps = pairwise_aabb_iou(predictions, targets)
    occupied = np.zeros(len(targets), dtype=np.bool_)
    order = np.argsort(-confidence, kind="stable")
    for prediction_index in order:
        gt_index = int(np.argmax(overlaps[prediction_index]))
        if (
            float(overlaps[prediction_index, gt_index]) > 0.50
            and not occupied[gt_index]
        ):
            flags[prediction_index] = True
            occupied[gt_index] = True
    return flags


def _scene_id_sha256(scene_ids: Iterable[str]) -> str:
    canonical = "\n".join(sorted(str(scene) for scene in scene_ids)) + "\n"
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _concatenate(parts: Iterable[np.ndarray], name: str) -> np.ndarray:
    values = list(parts)
    if not values:
        raise ValueError(f"No arrays collected for {name}")
    return np.concatenate(values, axis=0)


def _scene_training_arrays_legacy(
    diagnostics: SceneDiagnostics,
    prediction_corners: np.ndarray,
    transform: np.ndarray,
    gt_boxes: np.ndarray,
    config: BuildConfig,
) -> tuple[dict[str, np.ndarray], int]:
    count = len(diagnostics.points)
    point_count = diagnostics.points.shape[1]
    if count and diagnostics.result_indices.max() >= len(prediction_corners):
        raise ValueError(
            f"{diagnostics.scene_id}: result_indices exceed prediction count"
        )

    arrays = {
        "points_local": np.zeros((count, point_count, 3), dtype=np.float32),
        "point_mask": diagnostics.point_mask.astype(np.bool_, copy=True),
        "local_boxes": np.zeros((count, 6), dtype=np.float32),
        "quality_features": diagnostics.quality_features.astype(np.float32),
        "target_residual": np.zeros((count, 6), dtype=np.float32),
        "quality_target": np.zeros(count, dtype=np.float32),
        "geometry_mask": np.zeros(count, dtype=np.bool_),
        "original_iou": np.zeros(count, dtype=np.float32),
        "refined_iou": np.zeros(count, dtype=np.float32),
        "matched_gt_index": np.full(count, -1, dtype=np.int64),
        "target_center_local_unclipped": np.zeros((count, 3), dtype=np.float32),
        "target_dimensions_local_unclipped": np.zeros(
            (count, 3), dtype=np.float32
        ),
        "basis_world": np.zeros((count, 3, 3), dtype=np.float32),
        "result_indices": diagnostics.result_indices.astype(np.int64),
        "track_ids": diagnostics.track_ids.astype(np.int64),
    }
    if config.objective == "ap50":
        arrays.update(
            {
                "aligned_basis": np.zeros(
                    (count, 3, 3), dtype=np.float32
                ),
                "original_aligned_center": np.zeros(
                    (count, 3), dtype=np.float32
                ),
                "matched_gt_box": np.zeros(
                    (count, 6), dtype=np.float32
                ),
                "iou_gain": np.zeros(count, dtype=np.float32),
                "cross_iou50": np.zeros(count, dtype=np.bool_),
                "near_iou50": np.zeros(count, dtype=np.float32),
                "ap50_weight": np.ones(count, dtype=np.float32),
                "runtime_eligible": np.zeros(count, dtype=np.bool_),
                "selected_view_counts": np.asarray(
                    diagnostics.selected_view_counts, dtype=np.int64
                ).copy(),
            }
        )
    rotation = transform[:3, :3]
    translation = transform[:3, 3]
    invalid_oriented_boxes = 0

    for row, result_index in enumerate(diagnostics.result_indices):
        corners = prediction_corners[int(result_index)]
        try:
            center_world, dimensions, basis_world = oriented_box_frame(corners)
            frame_valid = True
        except ValueError:
            # Invalid OBBs are retained as rejection examples.  A positive
            # local size keeps the archive trainable and the mask disables
            # their geometry loss.
            invalid_oriented_boxes += 1
            dimensions = np.maximum(
                corners.max(axis=0) - corners.min(axis=0), 1e-3
            )
            center_world = corners.mean(axis=0)
            basis_world = np.eye(3, dtype=np.float64)
            frame_valid = False

        arrays["local_boxes"][row, 3:6] = dimensions
        arrays["basis_world"][row] = basis_world
        valid = diagnostics.point_mask[row]
        arrays["points_local"][row, valid] = (
            diagnostics.points[row, valid] - center_world
        ) @ basis_world

        aligned_center = rotation @ center_world + translation
        aligned_basis = rotation @ basis_world
        if config.objective == "ap50":
            arrays["aligned_basis"][row] = aligned_basis
            arrays["original_aligned_center"][row] = aligned_center
        original_aligned_dimensions = np.abs(aligned_basis) @ dimensions
        original_box = np.concatenate(
            (aligned_center, original_aligned_dimensions)
        )
        if not len(gt_boxes):
            continue
        pairwise = pairwise_aabb_iou(original_box[None, :], gt_boxes)[0]
        gt_index = int(np.argmax(pairwise))
        original_iou = float(pairwise[gt_index])
        target_gt = gt_boxes[gt_index]
        arrays["matched_gt_index"][row] = gt_index
        arrays["original_iou"][row] = original_iou
        if config.objective == "ap50":
            arrays["matched_gt_box"][row] = target_gt

        # Exact orthogonal projection of the aligned GT centre into the
        # original OBB local coordinates.
        raw_center_local = aligned_basis.T @ (
            target_gt[:3] - aligned_center
        )
        # Find non-negative local dimensions whose aligned AABB best matches
        # the GT dimensions.  This is the evaluator-aware geometry target.
        raw_dimensions_local = nonnegative_least_squares_3x3(
            np.abs(aligned_basis), target_gt[3:6]
        )
        arrays["target_center_local_unclipped"][row] = raw_center_local
        arrays["target_dimensions_local_unclipped"][row] = (
            raw_dimensions_local
        )

        center_limit = float(config.max_center_fraction) * dimensions
        target_center_local = np.clip(
            raw_center_local, -center_limit, center_limit
        )
        dimension_ratio = np.divide(
            raw_dimensions_local,
            dimensions,
            out=np.ones(3, dtype=np.float64),
            where=dimensions > 0.0,
        )
        log_dimension_residual = np.log(
            np.maximum(dimension_ratio, 1e-8)
        )
        log_dimension_residual = np.clip(
            log_dimension_residual,
            -float(config.max_log_dimension_residual),
            float(config.max_log_dimension_residual),
        )
        target_dimensions_local = dimensions * np.exp(
            log_dimension_residual
        )
        center_fraction = target_center_local / dimensions
        arrays["target_residual"][row] = np.concatenate(
            (center_fraction, log_dimension_residual)
        )

        refined_center = aligned_center + aligned_basis @ target_center_local
        refined_dimensions = (
            np.abs(aligned_basis) @ target_dimensions_local
        )
        refined_box = np.concatenate((refined_center, refined_dimensions))
        refined_iou = float(
            pairwise_aabb_iou(refined_box[None, :], target_gt[None, :])[0, 0]
        )
        arrays["refined_iou"][row] = refined_iou
        runtime_eligible = True
        strict_runtime = bool(
            config.strict_k5_diagnostics or config.objective == "ap50"
        )
        if strict_runtime:
            selected_views = int(
                np.asarray(diagnostics.selected_view_counts)[row]
            )
            valid_points = int(np.count_nonzero(valid))
            runtime_eligible = bool(
                frame_valid
                and selected_views >= int(config.min_runtime_views)
                and valid_points >= int(config.min_runtime_points)
                and np.all(
                    original_aligned_dimensions
                    >= float(config.runtime_minimum_extent)
                )
                and np.all(
                    refined_dimensions
                    >= float(config.runtime_minimum_extent)
                )
            )
            if config.objective == "ap50":
                arrays["runtime_eligible"][row] = runtime_eligible
        geometry_valid = bool(
            frame_valid
            and runtime_eligible
            and original_iou >= float(config.min_match_iou)
            and refined_iou
            > original_iou + float(config.improvement_epsilon)
        )
        arrays["geometry_mask"][row] = geometry_valid
        if config.objective == "ap50":
            gain = max(refined_iou - original_iou, 0.0)
            gain_normalized = min(
                gain / float(config.gain_cap), 1.0
            )
            cross_iou50 = bool(
                geometry_valid
                and original_iou < 0.50
                and refined_iou >= 0.50
            )
            near_iou50 = (
                max(
                    1.0
                    - abs(original_iou - 0.50)
                    / float(config.near_iou50_band),
                    0.0,
                )
                if runtime_eligible
                and original_iou >= float(config.min_match_iou)
                else 0.0
            )
            arrays["iou_gain"][row] = gain
            arrays["cross_iou50"][row] = cross_iou50
            arrays["near_iou50"][row] = near_iou50
            arrays["ap50_weight"][row] = (
                1.0
                + float(config.gain_sample_weight) * gain_normalized
                + float(config.cross_iou50_sample_weight)
                * float(cross_iou50)
                + float(config.near_iou50_sample_weight) * near_iou50
            )
            if geometry_valid:
                quality_target = 0.55 + 0.35 * gain_normalized
                if cross_iou50:
                    quality_target = max(quality_target, 0.95)
                arrays["quality_target"][row] = min(
                    quality_target, 1.0
                )
        else:
            arrays["quality_target"][row] = float(geometry_valid)

    return arrays, invalid_oriented_boxes


def _strict_scene_training_arrays(
    diagnostics: SceneDiagnostics,
    prediction_corners: np.ndarray,
    prediction_scores: np.ndarray,
    transform: np.ndarray,
    gt_boxes: np.ndarray,
    config: BuildConfig,
) -> tuple[dict[str, np.ndarray], int]:
    """Build schema-v2 rows from exact runtime-local K=5 tensors."""

    required_values = (
        diagnostics.local_boxes,
        diagnostics.frame_valid,
        diagnostics.gate_points_local,
        diagnostics.gate_point_mask,
        diagnostics.frame_centers,
        diagnostics.frame_basis,
        diagnostics.view_valid,
        diagnostics.view_scores,
        diagnostics.view_bboxes,
        diagnostics.view_intrinsics,
        diagnostics.view_camera_to_world,
        diagnostics.view_image_shapes,
        diagnostics.selected_view_counts,
        diagnostics.selected_view_frame_ids,
        diagnostics.top_k_view_valid,
    )
    if any(value is None for value in required_values):
        raise ValueError(
            "strict schema-v2 construction requires exact runtime tensors"
        )
    prediction_corners = np.asarray(prediction_corners, dtype=np.float64)
    prediction_scores = np.asarray(prediction_scores, dtype=np.float64)
    if prediction_scores.shape != (len(prediction_corners),):
        raise ValueError("prediction_scores must align with predictions")
    if len(diagnostics.result_indices) and (
        diagnostics.result_indices.max() >= len(prediction_corners)
    ):
        raise ValueError(
            f"{diagnostics.scene_id}: result_indices exceed prediction count"
        )

    frame_valid = np.asarray(diagnostics.frame_valid, dtype=np.bool_)
    model_nonempty = np.asarray(
        diagnostics.point_mask.any(axis=1), dtype=np.bool_
    )
    keep = frame_valid & model_nonempty
    retained_rows = np.flatnonzero(keep)
    invalid_oriented_boxes = int(len(frame_valid) - len(retained_rows))
    count = len(retained_rows)
    point_count = diagnostics.points.shape[1]
    arrays = {
        "points_local": np.asarray(
            diagnostics.points[retained_rows], dtype=np.float32
        ).copy(),
        "point_mask": np.asarray(
            diagnostics.point_mask[retained_rows], dtype=np.bool_
        ).copy(),
        "local_boxes": np.asarray(
            diagnostics.local_boxes[retained_rows], dtype=np.float32
        ).copy(),
        "quality_features": np.asarray(
            diagnostics.quality_features[retained_rows], dtype=np.float32
        ).copy(),
        "target_residual": np.zeros((count, 6), dtype=np.float32),
        "quality_target": np.zeros(count, dtype=np.float32),
        "geometry_mask": np.zeros(count, dtype=np.bool_),
        "original_iou": np.zeros(count, dtype=np.float32),
        "refined_iou": np.zeros(count, dtype=np.float32),
        "matched_gt_index": np.full(count, -1, dtype=np.int64),
        "target_center_local_unclipped": np.zeros(
            (count, 3), dtype=np.float32
        ),
        "target_dimensions_local_unclipped": np.zeros(
            (count, 3), dtype=np.float32
        ),
        "basis_world": np.asarray(
            diagnostics.frame_basis[retained_rows], dtype=np.float32
        ).copy(),
        "result_indices": np.asarray(
            diagnostics.result_indices[retained_rows], dtype=np.int64
        ).copy(),
        "track_ids": np.asarray(
            diagnostics.track_ids[retained_rows], dtype=np.int64
        ).copy(),
        "aligned_basis": np.zeros((count, 3, 3), dtype=np.float32),
        "original_aligned_center": np.zeros(
            (count, 3), dtype=np.float32
        ),
        "matched_gt_box": np.zeros((count, 6), dtype=np.float32),
        "iou_gain": np.zeros(count, dtype=np.float32),
        "cross_iou50": np.zeros(count, dtype=np.bool_),
        "near_iou50": np.zeros(count, dtype=np.float32),
        "ap50_weight": np.ones(count, dtype=np.float32),
        "runtime_eligible": np.zeros(count, dtype=np.bool_),
        "selected_view_counts": np.asarray(
            diagnostics.selected_view_counts[retained_rows],
            dtype=np.int64,
        ).copy(),
        "identity_tp50": np.zeros(count, dtype=np.bool_),
        "candidate_oracle_tp50": np.zeros(count, dtype=np.bool_),
    }
    if arrays["points_local"].shape != (count, point_count, 3):
        raise RuntimeError("strict model-input row selection failed")

    full_aligned_boxes = np.stack(
        [
            _corners_to_center_size(
                _transform_corners(corners, transform)
            )
            for corners in prediction_corners
        ],
        axis=0,
    ) if len(prediction_corners) else np.empty((0, 6), dtype=np.float64)
    oracle_aligned_boxes = full_aligned_boxes.copy()
    rotation = np.asarray(transform, dtype=np.float64)[:3, :3]
    translation = np.asarray(transform, dtype=np.float64)[:3, 3]
    gate_points = np.asarray(
        diagnostics.gate_points_local, dtype=np.float64
    )
    gate_masks = np.asarray(
        diagnostics.gate_point_mask, dtype=np.bool_
    )
    frame_centers = np.asarray(
        diagnostics.frame_centers, dtype=np.float64
    )
    frame_bases = np.asarray(
        diagnostics.frame_basis, dtype=np.float64
    )
    local_boxes = np.asarray(
        diagnostics.local_boxes, dtype=np.float64
    )

    for output_row, diagnostic_row in enumerate(retained_rows):
        result_index = int(diagnostics.result_indices[diagnostic_row])
        prediction = prediction_corners[result_index]
        try:
            recovered_center, recovered_dimensions, recovered_basis = (
                oriented_box_frame(prediction)
            )
        except ValueError as error:
            raise ValueError(
                f"{diagnostics.scene_id}: valid runtime frame for invalid "
                f"prediction {result_index}"
            ) from error
        frame_center = frame_centers[diagnostic_row]
        frame_basis = frame_bases[diagnostic_row]
        local_original = local_boxes[diagnostic_row]
        if (
            not np.allclose(
                recovered_center, frame_center, atol=2e-4, rtol=0.0
            )
            or not np.allclose(
                recovered_dimensions,
                local_original[3:6],
                atol=2e-3,
                rtol=0.0,
            )
            or not np.allclose(
                recovered_basis, frame_basis, atol=2e-3, rtol=0.0
            )
        ):
            raise ValueError(
                f"{diagnostics.scene_id}: runtime local frame does not "
                f"reconstruct prediction {result_index}"
            )
        reconstructed = _local_box_to_world_corners(
            local_original, frame_center, frame_basis
        )
        if not np.allclose(
            reconstructed, prediction, atol=3e-3, rtol=0.0
        ):
            raise ValueError(
                f"{diagnostics.scene_id}: runtime local corners disagree "
                f"with prediction {result_index}"
            )

        aligned_center = rotation @ frame_center + translation
        aligned_basis = rotation @ frame_basis
        arrays["aligned_basis"][output_row] = aligned_basis
        arrays["original_aligned_center"][output_row] = aligned_center
        original_box = full_aligned_boxes[result_index]
        if not len(gt_boxes):
            oracle_aligned_boxes[result_index] = original_box
            continue
        overlaps = pairwise_aabb_iou(
            original_box[None, :], gt_boxes
        )[0]
        gt_index = int(np.argmax(overlaps))
        original_iou = float(overlaps[gt_index])
        target_gt = np.asarray(gt_boxes[gt_index], dtype=np.float64)
        arrays["matched_gt_index"][output_row] = gt_index
        arrays["matched_gt_box"][output_row] = target_gt
        arrays["original_iou"][output_row] = original_iou

        raw_center_local = aligned_basis.T @ (
            target_gt[:3] - aligned_center
        )
        raw_dimensions_local = nonnegative_least_squares_3x3(
            np.abs(aligned_basis), target_gt[3:6]
        )
        arrays["target_center_local_unclipped"][
            output_row
        ] = raw_center_local
        arrays["target_dimensions_local_unclipped"][
            output_row
        ] = raw_dimensions_local
        center_limit = (
            float(config.max_center_fraction) * local_original[3:6]
        )
        clipped_center = np.clip(
            raw_center_local, -center_limit, center_limit
        )
        center_fraction = clipped_center / local_original[3:6]
        dimension_ratio = np.divide(
            raw_dimensions_local,
            local_original[3:6],
            out=np.ones(3, dtype=np.float64),
            where=local_original[3:6] > 0.0,
        )
        clipped_log_dimension = np.clip(
            np.log(np.maximum(dimension_ratio, 1e-8)),
            -float(config.max_log_dimension_residual),
            float(config.max_log_dimension_residual),
        )

        best_iou = -np.inf
        best_alpha: float | None = None
        best_local_candidate = local_original.copy()
        best_candidate_corners = prediction.copy()
        best_aligned_box = original_box.copy()
        for alpha in TARGET_LINE_SEARCH_ALPHAS:
            candidate = np.concatenate(
                (
                    center_fraction
                    * float(alpha)
                    * local_original[3:6],
                    local_original[3:6]
                    * np.exp(clipped_log_dimension * float(alpha)),
                )
            ).astype(np.float32)
            accepted, _, _ = runtime_refit_gate(
                local_original,
                candidate,
                gate_points_local=gate_points[diagnostic_row],
                gate_point_mask=gate_masks[diagnostic_row],
                frame_center=frame_center,
                frame_basis=frame_basis,
                selected_view_frame_ids=(
                    diagnostics.selected_view_frame_ids[diagnostic_row]
                ),
                top_k_view_valid=(
                    diagnostics.top_k_view_valid[diagnostic_row]
                ),
                view_valid=diagnostics.view_valid[diagnostic_row],
                view_frame_ids=(
                    diagnostics.view_frame_ids[diagnostic_row]
                ),
                view_scores=diagnostics.view_scores[diagnostic_row],
                view_bboxes=diagnostics.view_bboxes[diagnostic_row],
                view_intrinsics=(
                    diagnostics.view_intrinsics[diagnostic_row]
                ),
                view_camera_to_world=(
                    diagnostics.view_camera_to_world[diagnostic_row]
                ),
                view_image_shapes=(
                    diagnostics.view_image_shapes[diagnostic_row]
                ),
                config=config,
            )
            if not accepted:
                continue
            candidate_corners = _local_box_to_world_corners(
                candidate, frame_center, frame_basis
            )
            aligned_candidate = _corners_to_center_size(
                _transform_corners(candidate_corners, transform)
            )
            candidate_iou = float(
                pairwise_aabb_iou(
                    aligned_candidate[None, :], target_gt[None, :]
                )[0, 0]
            )
            # Alphas are ascending, so a numerical tie deterministically
            # retains the more conservative candidate.
            if candidate_iou > best_iou + 1e-12:
                best_iou = candidate_iou
                best_alpha = float(alpha)
                best_local_candidate = candidate
                best_candidate_corners = candidate_corners
                best_aligned_box = aligned_candidate

        if best_alpha is None:
            arrays["refined_iou"][output_row] = original_iou
            oracle_aligned_boxes[result_index] = original_box
            continue
        arrays["runtime_eligible"][output_row] = True
        arrays["target_residual"][output_row] = np.concatenate(
            (
                center_fraction * best_alpha,
                clipped_log_dimension * best_alpha,
            )
        )
        arrays["refined_iou"][output_row] = best_iou
        oracle_aligned_boxes[result_index] = best_aligned_box
        # Keep this assertion local to the builder: it catches any future
        # disagreement between the selected target and its stored residual.
        residual_candidate = np.concatenate(
            (
                arrays["target_residual"][output_row, :3]
                * local_original[3:6],
                local_original[3:6]
                * np.exp(
                    arrays["target_residual"][output_row, 3:6]
                ),
            )
        )
        if not np.allclose(
            residual_candidate, best_local_candidate, atol=2e-6
        ) or not np.allclose(
            best_candidate_corners,
            _local_box_to_world_corners(
                residual_candidate, frame_center, frame_basis
            ),
            atol=2e-6,
        ):
            raise RuntimeError("line-search target reconstruction failed")

    identity_flags = greedy_scene_tp50_flags(
        full_aligned_boxes, prediction_scores, gt_boxes
    )
    oracle_flags = greedy_scene_tp50_flags(
        oracle_aligned_boxes, prediction_scores, gt_boxes
    )
    if count:
        result_indices = arrays["result_indices"]
        arrays["identity_tp50"] = identity_flags[result_indices].copy()
        arrays["candidate_oracle_tp50"] = (
            oracle_flags[result_indices].copy()
        )
    arrays["cross_iou50"] = (
        arrays["runtime_eligible"]
        & arrays["candidate_oracle_tp50"]
        & ~arrays["identity_tp50"]
    )

    improvement = (
        arrays["runtime_eligible"]
        & (arrays["original_iou"] >= float(config.min_match_iou))
        & (
            arrays["refined_iou"]
            > arrays["original_iou"] + float(config.improvement_epsilon)
        )
    )
    arrays["geometry_mask"] = (
        improvement | arrays["cross_iou50"]
        if config.objective == "ap50"
        else improvement
    )
    arrays["iou_gain"] = np.maximum(
        arrays["refined_iou"] - arrays["original_iou"], 0.0
    ).astype(np.float32)
    eligible_matched = (
        arrays["runtime_eligible"]
        & (arrays["matched_gt_index"] >= 0)
        & (arrays["original_iou"] >= float(config.min_match_iou))
    )
    arrays["near_iou50"] = np.where(
        eligible_matched,
        np.maximum(
            1.0
            - np.abs(arrays["original_iou"] - 0.50)
            / float(config.near_iou50_band),
            0.0,
        ),
        0.0,
    ).astype(np.float32)
    gain_normalized = np.minimum(
        arrays["iou_gain"] / float(config.gain_cap), 1.0
    )
    arrays["ap50_weight"] = (
        1.0
        + float(config.gain_sample_weight) * gain_normalized
        + float(config.cross_iou50_sample_weight)
        * arrays["cross_iou50"].astype(np.float32)
        + float(config.near_iou50_sample_weight)
        * arrays["near_iou50"]
    ).astype(np.float32)
    if config.objective == "ap50":
        positive = arrays["geometry_mask"]
        quality = 0.55 + 0.35 * gain_normalized
        quality = np.where(
            arrays["cross_iou50"], np.maximum(quality, 0.95), quality
        )
        arrays["quality_target"][positive] = np.minimum(
            quality[positive], 1.0
        )
    else:
        arrays["quality_target"] = arrays["geometry_mask"].astype(
            np.float32
        )
    return arrays, invalid_oriented_boxes


def _scene_training_arrays(
    diagnostics: SceneDiagnostics,
    prediction_corners: np.ndarray,
    transform: np.ndarray,
    gt_boxes: np.ndarray,
    config: BuildConfig,
    prediction_scores: np.ndarray | None = None,
) -> tuple[dict[str, np.ndarray], int]:
    strict_schema = bool(
        config.strict_k5_diagnostics or config.objective == "ap50"
    )
    if not strict_schema:
        return _scene_training_arrays_legacy(
            diagnostics, prediction_corners, transform, gt_boxes, config
        )
    if prediction_scores is None:
        raise ValueError(
            "schema-v2 construction requires real prediction scores"
        )
    return _strict_scene_training_arrays(
        diagnostics,
        prediction_corners,
        prediction_scores,
        transform,
        gt_boxes,
        config,
    )


def build_oriented_refiner_dataset(config: BuildConfig) -> BuildSummary:
    """Build and atomically write a deterministic B5-v2 dataset."""

    config = config.validated()
    scenes = read_scene_ids(config.scene_list)
    strict_schema = bool(
        config.strict_k5_diagnostics or config.objective == "ap50"
    )
    forbidden_scenes: set[str] = set()
    if strict_schema:
        assert config.forbidden_scene_list is not None
        forbidden_scenes = set(read_scene_ids(config.forbidden_scene_list))
        leaked = sorted(set(scenes) & forbidden_scenes)
        if leaked:
            raise ValueError(
                "strict training scene list overlaps forbidden validation "
                f"scenes: {leaked[:5]}"
            )
    collected: dict[str, list[np.ndarray]] = {}
    expected_point_count: int | None = None
    invalid_total = 0

    for scene_id in scenes:
        diagnostics = load_scene_diagnostics(
            resolve_diagnostic_path(config.diagnostics_root, scene_id),
            expected_scene_id=scene_id,
            objective=config.objective,
            expected_top_k_views=config.expected_top_k_views,
            strict_k5_diagnostics=config.strict_k5_diagnostics,
        )
        if expected_point_count is None:
            expected_point_count = diagnostics.points.shape[1]
        elif diagnostics.points.shape[1] != expected_point_count:
            raise ValueError(
                "All diagnostics must use the same point count; "
                f"{scene_id} has {diagnostics.points.shape[1]}, "
                f"expected {expected_point_count}"
            )
        prediction_corners, prediction_scores = load_prediction_detections(
            Path(config.prediction_root) / f"{scene_id}_boxes.pkl"
        )
        transform = load_axis_alignment(config.scan_root, scene_id)
        gt_boxes = load_gt_boxes(config.gt_root, scene_id)
        arrays, invalid_count = _scene_training_arrays(
            diagnostics,
            prediction_corners,
            transform,
            gt_boxes,
            config,
            prediction_scores=prediction_scores,
        )
        invalid_total += invalid_count
        arrays["scene_ids"] = np.full(
            len(arrays["geometry_mask"]),
            scene_id,
            dtype=f"<U{len(scene_id)}",
        )
        for name, value in arrays.items():
            collected.setdefault(name, []).append(value)

    output_arrays = {
        name: _concatenate(parts, name) for name, parts in collected.items()
    }
    sample_count = len(output_arrays["geometry_mask"])
    if sample_count < 2:
        raise ValueError("B5-v2 dataset requires at least two samples")
    if not output_arrays["geometry_mask"].any():
        raise ValueError(
            "No reachable target improves IoU; check coordinate roots and "
            "matching thresholds"
        )
    if output_arrays["geometry_mask"].all():
        raise ValueError(
            "B5-v2 rejection training requires at least one negative sample"
        )
    metadata = {
        "schema": np.asarray(DATASET_SCHEMA),
        "format_version": np.asarray(
            (
                AP50_DATASET_FORMAT_VERSION
                if strict_schema
                else DATASET_FORMAT_VERSION
            ),
            dtype=np.int64,
        ),
        "coordinate_frame": np.asarray("box_local"),
        "quality_feature_names": np.asarray(
            QUALITY_FEATURE_NAMES, dtype=np.str_
        ),
        "max_center_fraction": np.asarray(
            config.max_center_fraction, dtype=np.float32
        ),
        "max_log_dimension_residual": np.asarray(
            config.max_log_dimension_residual, dtype=np.float32
        ),
    }
    if strict_schema:
        provenance_metadata = {
            name: np.asarray(value)
            for name, value in _STRICT_PROVENANCE_EXPECTED.items()
        }
        metadata.update(
            {
                "objective": np.asarray(config.objective),
                "strict_k5_diagnostics": np.asarray(True, dtype=np.bool_),
                "expected_top_k_views": np.asarray(
                    config.expected_top_k_views, dtype=np.int64
                ),
                "min_runtime_views": np.asarray(
                    config.min_runtime_views, dtype=np.int64
                ),
                "min_runtime_points": np.asarray(
                    config.min_runtime_points, dtype=np.int64
                ),
                "runtime_minimum_extent": np.asarray(
                    config.runtime_minimum_extent, dtype=np.float32
                ),
                "near_iou50_band": np.asarray(
                    config.near_iou50_band, dtype=np.float32
                ),
                "gain_cap": np.asarray(
                    config.gain_cap, dtype=np.float32
                ),
                "gain_sample_weight": np.asarray(
                    config.gain_sample_weight, dtype=np.float32
                ),
                "cross_iou50_sample_weight": np.asarray(
                    config.cross_iou50_sample_weight, dtype=np.float32
                ),
                "near_iou50_sample_weight": np.asarray(
                    config.near_iou50_sample_weight, dtype=np.float32
                ),
                "min_match_iou": np.asarray(
                    config.min_match_iou, dtype=np.float32
                ),
                "improvement_epsilon": np.asarray(
                    config.improvement_epsilon, dtype=np.float32
                ),
                "target_line_search_alphas": np.asarray(
                    TARGET_LINE_SEARCH_ALPHAS, dtype=np.float32
                ),
                "forbidden_scene_count": np.asarray(
                    len(forbidden_scenes), dtype=np.int64
                ),
                "forbidden_scene_sha256": np.asarray(
                    _scene_id_sha256(forbidden_scenes)
                ),
                "training_scene_count": np.asarray(
                    len(scenes), dtype=np.int64
                ),
                "training_scene_sha256": np.asarray(
                    _scene_id_sha256(scenes)
                ),
            }
        )
        metadata.update(provenance_metadata)
    output_arrays.update(metadata)
    expected_keys = (
        V2_SAMPLE_KEYS | V2_METADATA_KEYS
        if strict_schema
        else BASE_SAMPLE_KEYS | BASE_METADATA_KEYS
    )
    if set(output_arrays) != set(expected_keys):
        raise RuntimeError(
            "internal dataset schema mismatch: "
            f"missing={sorted(set(expected_keys) - set(output_arrays))}, "
            f"unexpected={sorted(set(output_arrays) - set(expected_keys))}"
        )
    output = Path(config.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    try:
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, **output_arrays)
        temporary.replace(output)
    finally:
        if temporary.exists():
            temporary.unlink()

    positive_count = int(np.count_nonzero(output_arrays["geometry_mask"]))
    cross_count = int(
        np.count_nonzero(output_arrays.get("cross_iou50", []))
    )
    return BuildSummary(
        scenes=len(scenes),
        samples=sample_count,
        geometry_positives=positive_count,
        quality_negatives=sample_count - positive_count,
        invalid_oriented_boxes=invalid_total,
        output=output,
        cross_iou50_positives=cross_count,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build box-local, evaluator-aware B5-v2 supervision."
    )
    parser.add_argument("--diagnostics-root", type=Path, required=True)
    parser.add_argument("--prediction-root", type=Path, required=True)
    parser.add_argument("--scan-root", type=Path, required=True)
    parser.add_argument("--gt-root", type=Path, required=True)
    parser.add_argument("--scene-list", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-match-iou", type=float, default=0.15)
    parser.add_argument("--improvement-epsilon", type=float, default=1e-4)
    parser.add_argument("--max-center-fraction", type=float, default=0.15)
    parser.add_argument(
        "--max-log-dimension-residual",
        type=float,
        default=float(np.log(1.25)),
    )
    parser.add_argument(
        "--objective",
        choices=TRAINING_OBJECTIVES,
        default="improvement",
    )
    parser.add_argument(
        "--strict-k5-diagnostics",
        action="store_true",
        help=(
            "require geometry_points and exact K-view diagnostic provenance "
            "even for the legacy improvement objective"
        ),
    )
    parser.add_argument("--forbidden-scene-list", type=Path)
    parser.add_argument("--expected-top-k-views", type=int, default=5)
    parser.add_argument("--min-runtime-views", type=int, default=2)
    parser.add_argument("--min-runtime-points", type=int, default=128)
    parser.add_argument(
        "--runtime-minimum-extent", type=float, default=0.40
    )
    parser.add_argument("--near-iou50-band", type=float, default=0.15)
    parser.add_argument("--gain-cap", type=float, default=0.25)
    parser.add_argument("--gain-sample-weight", type=float, default=2.0)
    parser.add_argument(
        "--cross-iou50-sample-weight", type=float, default=4.0
    )
    parser.add_argument(
        "--near-iou50-sample-weight", type=float, default=2.0
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = build_oriented_refiner_dataset(
        BuildConfig(
            diagnostics_root=args.diagnostics_root,
            prediction_root=args.prediction_root,
            scan_root=args.scan_root,
            gt_root=args.gt_root,
            scene_list=args.scene_list,
            output=args.output,
            min_match_iou=args.min_match_iou,
            improvement_epsilon=args.improvement_epsilon,
            max_center_fraction=args.max_center_fraction,
            max_log_dimension_residual=args.max_log_dimension_residual,
            objective=args.objective,
            strict_k5_diagnostics=args.strict_k5_diagnostics,
            forbidden_scene_list=args.forbidden_scene_list,
            expected_top_k_views=args.expected_top_k_views,
            min_runtime_views=args.min_runtime_views,
            min_runtime_points=args.min_runtime_points,
            runtime_minimum_extent=args.runtime_minimum_extent,
            near_iou50_band=args.near_iou50_band,
            gain_cap=args.gain_cap,
            gain_sample_weight=args.gain_sample_weight,
            cross_iou50_sample_weight=(
                args.cross_iou50_sample_weight
            ),
            near_iou50_sample_weight=args.near_iou50_sample_weight,
        )
    )
    print(
        "Built B5-v2 dataset: "
        f"scenes={summary.scenes}, samples={summary.samples}, "
        f"geometry_positives={summary.geometry_positives}, "
        f"quality_negatives={summary.quality_negatives}, "
        f"cross_iou50={summary.cross_iou50_positives}, "
        f"invalid_obb={summary.invalid_oriented_boxes}, "
        f"path={summary.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
