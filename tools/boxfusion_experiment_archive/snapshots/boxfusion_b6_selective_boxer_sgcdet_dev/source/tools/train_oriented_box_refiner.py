#!/usr/bin/env python3
"""Train the orientation-aware B5-v2 local box refiner on CPU.

Unlike the legacy trainer, this tool requires ``scene_ids`` and splits whole
scenes, never individual samples.  Training batches are deterministically
balanced between reachable geometry improvements and rejection examples.
Geometry losses are evaluated only on the former; negative samples supervise
only the binary quality/acceptance head.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence

import numpy as np

try:
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Dataset
except ImportError as error:  # pragma: no cover
    raise ImportError("training B5-v2 requires PyTorch") from error

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

from boxfusion.oriented_box_refiner import (
    OrientedBoxRefinerConfig,
    PointNetOrientedBoxRefiner,
    make_oriented_box_refiner_checkpoint,
)
from boxfusion.quality_score import QUALITY_FEATURE_NAMES
from tools.build_oriented_refiner_dataset import (
    AP50_DATASET_FORMAT_VERSION,
    BASE_METADATA_KEYS,
    BASE_SAMPLE_KEYS,
    DATASET_FORMAT_VERSION,
    DATASET_SCHEMA,
    TARGET_LINE_SEARCH_ALPHAS,
    TRAINING_OBJECTIVES,
    V2_METADATA_KEYS,
    V2_SAMPLE_KEYS,
    strict_provenance_for_profile,
)


COORDINATE_FRAME = "box_local"
QUALITY_FEATURE_DIM = len(QUALITY_FEATURE_NAMES)
SCENE_PATTERN = re.compile(r"scene\d{4}_\d{2}")
AP50_EVALUATOR_THRESHOLD = 0.50
AP50_LOSS_MARGIN = 1e-4
AP50_LOSS_TARGET = AP50_EVALUATOR_THRESHOLD + AP50_LOSS_MARGIN


@dataclass(frozen=True)
class OrientedRefinerTrainingData:
    points_local: np.ndarray
    point_mask: np.ndarray
    local_boxes: np.ndarray
    quality_features: np.ndarray
    target_residual: np.ndarray
    quality_target: np.ndarray
    geometry_mask: np.ndarray
    scene_ids: np.ndarray
    matched_gt_index: np.ndarray
    original_iou: np.ndarray
    refined_iou: np.ndarray
    aligned_basis: np.ndarray
    original_aligned_center: np.ndarray
    matched_gt_box: np.ndarray
    iou_gain: np.ndarray
    cross_iou50: np.ndarray
    near_iou50: np.ndarray
    ap50_weight: np.ndarray
    runtime_eligible: np.ndarray
    identity_tp50: np.ndarray
    candidate_oracle_tp50: np.ndarray
    objective: str
    max_center_fraction: float
    max_log_dimension_residual: float

    @property
    def sample_count(self) -> int:
        return int(self.points_local.shape[0])


def _scalar_string(value: np.ndarray, name: str) -> str:
    array = np.asarray(value)
    if array.dtype.hasobject:
        raise ValueError(f"{name} must not use object dtype")
    if array.ndim != 0:
        raise ValueError(f"{name} must be scalar")
    scalar = array.item()
    if isinstance(scalar, bytes):
        scalar = scalar.decode("utf-8")
    if not isinstance(scalar, str):
        raise TypeError(f"{name} must be a string")
    return scalar


def _scalar_float(value: np.ndarray, name: str) -> float:
    array = np.asarray(value)
    if array.ndim != 0 or not np.issubdtype(array.dtype, np.number):
        raise ValueError(f"{name} must be a numeric scalar")
    scalar = float(array)
    if not np.isfinite(scalar):
        raise ValueError(f"{name} must be finite")
    return scalar


def _scalar_integer(value: np.ndarray, name: str) -> int:
    array = np.asarray(value)
    if (
        array.ndim != 0
        or array.dtype == np.bool_
        or not np.issubdtype(array.dtype, np.integer)
    ):
        raise TypeError(f"{name} must be an integer scalar")
    return int(array)


def _scalar_boolean(value: np.ndarray, name: str) -> bool:
    array = np.asarray(value)
    if array.ndim != 0 or array.dtype != np.bool_:
        raise TypeError(f"{name} must be a Boolean scalar")
    return bool(array)


def _validate_v2_metadata(
    archive: np.lib.npyio.NpzFile,
) -> tuple[str, int, str]:
    """Validate the exact strict-K5 provenance stored in schema v2."""

    objective = _scalar_string(archive["objective"], "objective")
    if objective not in TRAINING_OBJECTIVES:
        raise ValueError(f"objective must be one of {TRAINING_OBJECTIVES}")
    if not _scalar_boolean(
        archive["strict_k5_diagnostics"], "strict_k5_diagnostics"
    ):
        raise ValueError("format v2 requires strict_k5_diagnostics=true")

    integers = {
        name: _scalar_integer(archive[name], name)
        for name in (
            "expected_top_k_views",
            "min_runtime_views",
            "min_runtime_points",
            "forbidden_scene_count",
            "training_scene_count",
        )
    }
    if any(value <= 0 for value in integers.values()):
        raise ValueError("strict-K5 integer metadata must be positive")
    if integers["expected_top_k_views"] != 5:
        raise ValueError("format v2 dataset must use exactly K=5")
    if integers["min_runtime_views"] > 5:
        raise ValueError("min_runtime_views exceeds K=5")

    positive_float_names = (
        "runtime_minimum_extent",
        "near_iou50_band",
        "gain_cap",
    )
    positive_floats = {
        name: _scalar_float(archive[name], name)
        for name in positive_float_names
    }
    if any(value <= 0.0 for value in positive_floats.values()):
        raise ValueError("strict-K5 positive metadata must be positive")
    nonnegative_float_names = (
        "gain_sample_weight",
        "cross_iou50_sample_weight",
        "near_iou50_sample_weight",
        "improvement_epsilon",
    )
    if any(
        _scalar_float(archive[name], name) < 0.0
        for name in nonnegative_float_names
    ):
        raise ValueError("schema-v2 weights/epsilon must be non-negative")
    min_match_iou = _scalar_float(
        archive["min_match_iou"], "min_match_iou"
    )
    if not 0.0 <= min_match_iou <= 1.0:
        raise ValueError("min_match_iou must lie in [0, 1]")

    alphas = np.asarray(archive["target_line_search_alphas"])
    if (
        alphas.ndim != 1
        or not np.issubdtype(alphas.dtype, np.floating)
        or not np.isfinite(alphas).all()
        or alphas.shape != (len(TARGET_LINE_SEARCH_ALPHAS),)
        or not np.allclose(
            alphas,
            np.asarray(TARGET_LINE_SEARCH_ALPHAS),
            atol=1e-8,
            rtol=0.0,
        )
    ):
        raise ValueError(
            "target_line_search_alphas disagree with the strict builder"
        )

    digests = {}
    for name in ("forbidden_scene_sha256", "training_scene_sha256"):
        digest = _scalar_string(archive[name], name)
        if re.fullmatch(r"[0-9a-f]{64}", digest) is None:
            raise ValueError(f"{name} must be a lowercase SHA-256 digest")
        digests[name] = digest
    if digests["forbidden_scene_sha256"] == digests["training_scene_sha256"]:
        raise ValueError("training and forbidden scene digests must differ")

    provenance_profile = _scalar_string(
        archive["online_ablation_profile"], "online_ablation_profile"
    )
    expected_provenance = strict_provenance_for_profile(
        provenance_profile
    )
    provenance: dict[str, object] = {}
    for name, expected in expected_provenance.items():
        if isinstance(expected, bool):
            actual: object = _scalar_boolean(archive[name], name)
        elif isinstance(expected, int):
            actual = _scalar_integer(archive[name], name)
        elif isinstance(expected, float):
            actual = _scalar_float(archive[name], name)
        else:
            actual = _scalar_string(archive[name], name)
        if isinstance(expected, float):
            matches = np.isclose(
                float(actual), expected, atol=1e-8, rtol=0.0
            )
        else:
            matches = actual == expected
        if not bool(matches):
            raise ValueError(
                f"schema-v2 provenance {name}={actual!r}, "
                f"expected {expected!r}"
            )
        provenance[name] = actual

    cross_checks = {
        "top_k_views": integers["expected_top_k_views"],
        "refit_gate_min_views": integers["min_runtime_views"],
        "refit_gate_min_points": integers["min_runtime_points"],
        "output_minimum_extent": positive_floats[
            "runtime_minimum_extent"
        ],
    }
    for name, expected in cross_checks.items():
        actual = provenance[name]
        if isinstance(expected, float):
            matches = np.isclose(
                float(actual), expected, atol=1e-8, rtol=0.0
            )
        else:
            matches = actual == expected
        if not bool(matches):
            raise ValueError(
                f"schema-v2 metadata {name} disagrees with runtime provenance"
            )
    return (
        objective,
        integers["training_scene_count"],
        digests["training_scene_sha256"],
    )


def load_oriented_refiner_dataset(
    path: str | os.PathLike,
) -> OrientedRefinerTrainingData:
    """Load and strictly validate a pickle-free B5-v2 archive."""

    dataset_path = Path(path)
    if not dataset_path.is_file():
        raise FileNotFoundError(dataset_path)
    if dataset_path.suffix.lower() != ".npz":
        raise ValueError("B5-v2 dataset must end in .npz")
    with np.load(dataset_path, allow_pickle=False) as archive:
        keys = set(archive.files)
        if "format_version" not in keys:
            raise ValueError(
                "B5-v2 dataset keys are invalid: "
                "missing=['format_version']"
            )
        version = np.asarray(archive["format_version"])
        if version.ndim != 0 or not np.issubdtype(
            version.dtype, np.integer
        ):
            raise ValueError("B5-v2 dataset format_version mismatch")
        format_version = int(version)
        stored_training_scene_count: int | None = None
        stored_training_scene_sha256: str | None = None
        if format_version == DATASET_FORMAT_VERSION:
            objective = "improvement"
            sample_keys = BASE_SAMPLE_KEYS
            metadata_keys = BASE_METADATA_KEYS
        elif format_version == AP50_DATASET_FORMAT_VERSION:
            sample_keys = V2_SAMPLE_KEYS
            metadata_keys = V2_METADATA_KEYS
        else:
            raise ValueError("B5-v2 dataset format_version mismatch")
        expected = sample_keys | metadata_keys
        if keys != expected:
            missing = sorted(expected - keys)
            extra = sorted(keys - expected)
            raise ValueError(
                f"B5-v2 dataset keys are invalid: missing={missing}, "
                f"unexpected={extra}"
            )
        if _scalar_string(archive["schema"], "schema") != DATASET_SCHEMA:
            raise ValueError("B5-v2 dataset schema mismatch")
        if format_version == AP50_DATASET_FORMAT_VERSION:
            (
                objective,
                stored_training_scene_count,
                stored_training_scene_sha256,
            ) = _validate_v2_metadata(archive)
        if (
            _scalar_string(archive["coordinate_frame"], "coordinate_frame")
            != COORDINATE_FRAME
        ):
            raise ValueError("B5-v2 dataset must use box_local coordinates")
        feature_names = tuple(
            str(item)
            for item in np.asarray(archive["quality_feature_names"]).tolist()
        )
        if feature_names != QUALITY_FEATURE_NAMES:
            raise ValueError("quality feature schema/order mismatch")
        arrays = {
            name: np.asarray(archive[name]).copy() for name in sample_keys
        }
        max_center_fraction = _scalar_float(
            archive["max_center_fraction"], "max_center_fraction"
        )
        max_log_dimension_residual = _scalar_float(
            archive["max_log_dimension_residual"],
            "max_log_dimension_residual",
        )

    if max_center_fraction <= 0.0 or max_log_dimension_residual <= 0.0:
        raise ValueError("stored model residual bounds must be positive")
    points = arrays["points_local"]
    if (
        points.ndim != 3
        or points.shape[2] != 3
        or points.shape[0] < 2
        or points.shape[1] < 1
    ):
        raise ValueError("points_local must have shape [N>=2, P>=1, 3]")
    if not np.issubdtype(points.dtype, np.floating):
        raise TypeError("points_local must use floating-point dtype")
    points = np.asarray(points, dtype=np.float32)
    if not np.isfinite(points).all():
        raise ValueError("points_local must be finite")
    sample_count, point_count, _ = points.shape

    point_mask = arrays["point_mask"]
    if point_mask.shape != (sample_count, point_count):
        raise ValueError("point_mask must have shape [N, P]")
    if point_mask.dtype != np.bool_:
        raise TypeError("point_mask must have Boolean dtype")
    if not point_mask.any(axis=1).all():
        raise ValueError("every sample must contain at least one valid point")
    # Padding must be canonical so it can never encode scene-specific garbage.
    if not np.all(points[~point_mask] == 0.0):
        raise ValueError("masked points_local padding must be exactly zero")

    local_boxes = arrays["local_boxes"]
    quality_features = arrays["quality_features"]
    target_residual = arrays["target_residual"]
    expected_shapes = {
        "local_boxes": (sample_count, 6),
        "quality_features": (sample_count, QUALITY_FEATURE_DIM),
        "target_residual": (sample_count, 6),
    }
    for name, expected_shape in expected_shapes.items():
        value = locals()[name]
        if value.shape != expected_shape:
            raise ValueError(f"{name} must have shape {expected_shape}")
        if not np.issubdtype(value.dtype, np.floating):
            raise TypeError(f"{name} must use floating-point dtype")
        if not np.isfinite(value).all():
            raise ValueError(f"{name} must be finite")
    local_boxes = np.asarray(local_boxes, dtype=np.float32)
    quality_features = np.asarray(quality_features, dtype=np.float32)
    target_residual = np.asarray(target_residual, dtype=np.float32)
    if not np.allclose(local_boxes[:, :3], 0.0, atol=1e-7):
        raise ValueError("local_boxes centres must be the local origin")
    if (local_boxes[:, 3:6] <= 0.0).any():
        raise ValueError("local_boxes dimensions must be positive")
    if (
        (quality_features < 0.0).any()
        or (quality_features > 1.0).any()
    ):
        raise ValueError("quality_features must lie in [0, 1]")
    if (
        np.abs(target_residual[:, :3]) > max_center_fraction + 1e-5
    ).any():
        raise ValueError("target centre residual exceeds stored model bound")
    if (
        np.abs(target_residual[:, 3:])
        > max_log_dimension_residual + 1e-5
    ).any():
        raise ValueError("target dimension residual exceeds stored model bound")

    geometry_mask = arrays["geometry_mask"]
    if geometry_mask.shape != (sample_count,) or geometry_mask.dtype != np.bool_:
        raise TypeError("geometry_mask must be Boolean with shape [N]")
    quality_target = arrays["quality_target"]
    if quality_target.shape != (sample_count,) or not np.issubdtype(
        quality_target.dtype, np.floating
    ):
        raise ValueError("quality_target must be floating with shape [N]")
    quality_target = np.asarray(quality_target, dtype=np.float32)
    if (
        not np.isfinite(quality_target).all()
        or (quality_target < 0.0).any()
        or (quality_target > 1.0).any()
    ):
        raise ValueError("quality_target must lie in [0, 1]")
    if objective == "improvement":
        if not np.isin(quality_target, (0.0, 1.0)).all():
            raise ValueError(
                "improvement quality_target must contain only 0 and 1"
            )
        if not np.array_equal(quality_target.astype(bool), geometry_mask):
            raise ValueError(
                "quality_target must exactly encode geometry_mask"
            )
    elif not np.array_equal(quality_target > 0.5, geometry_mask):
        raise ValueError(
            "AP50 quality_target threshold must exactly encode geometry_mask"
        )
    if not geometry_mask.any() or geometry_mask.all():
        raise ValueError("dataset must contain geometry positives and negatives")

    scene_ids = arrays["scene_ids"]
    if (
        scene_ids.shape != (sample_count,)
        or scene_ids.dtype.hasobject
        or scene_ids.dtype.kind not in {"U", "S"}
    ):
        raise TypeError("scene_ids must be a non-object string array [N]")
    scene_ids = scene_ids.astype(np.str_)
    if any(SCENE_PATTERN.fullmatch(scene) is None for scene in scene_ids):
        raise ValueError("scene_ids contains an invalid ScanNet scene id")
    if len(np.unique(scene_ids)) < 2:
        raise ValueError("scene-level splitting requires at least two scenes")
    if format_version == AP50_DATASET_FORMAT_VERSION:
        unique_scenes = sorted(np.unique(scene_ids).tolist())
        if stored_training_scene_count != len(unique_scenes):
            raise ValueError(
                "training_scene_count disagrees with scene_ids"
            )
        canonical = "\n".join(unique_scenes) + "\n"
        actual_training_digest = hashlib.sha256(
            canonical.encode("utf-8")
        ).hexdigest()
        if stored_training_scene_sha256 != actual_training_digest:
            raise ValueError(
                "training_scene_sha256 disagrees with scene_ids"
            )

    original_iou = np.asarray(arrays["original_iou"], dtype=np.float32)
    refined_iou = np.asarray(arrays["refined_iou"], dtype=np.float32)
    matched_gt_index = np.asarray(arrays["matched_gt_index"])
    if (
        matched_gt_index.shape != (sample_count,)
        or not np.issubdtype(matched_gt_index.dtype, np.integer)
        or (matched_gt_index < -1).any()
    ):
        raise ValueError(
            "matched_gt_index must be integer in [-1,...) with shape [N]"
        )
    matched_gt_index = np.asarray(matched_gt_index, dtype=np.int64)
    for name, value in (
        ("original_iou", original_iou),
        ("refined_iou", refined_iou),
    ):
        if (
            value.shape != (sample_count,)
            or not np.isfinite(value).all()
            or (value < 0.0).any()
            or (value > 1.0).any()
        ):
            raise ValueError(f"{name} must be finite in [0, 1] with shape [N]")

    if format_version == AP50_DATASET_FORMAT_VERSION:
        aligned_basis = np.asarray(
            arrays["aligned_basis"], dtype=np.float32
        )
        original_aligned_center = np.asarray(
            arrays["original_aligned_center"], dtype=np.float32
        )
        matched_gt_box = np.asarray(
            arrays["matched_gt_box"], dtype=np.float32
        )
        iou_gain = np.asarray(arrays["iou_gain"], dtype=np.float32)
        cross_iou50 = arrays["cross_iou50"]
        near_iou50 = np.asarray(arrays["near_iou50"], dtype=np.float32)
        ap50_weight = np.asarray(arrays["ap50_weight"], dtype=np.float32)
        runtime_eligible = arrays["runtime_eligible"]
        selected_view_counts = np.asarray(arrays["selected_view_counts"])
        identity_tp50 = arrays["identity_tp50"]
        candidate_oracle_tp50 = arrays["candidate_oracle_tp50"]
        if aligned_basis.shape != (sample_count, 3, 3):
            raise ValueError("aligned_basis must have shape [N, 3, 3]")
        if original_aligned_center.shape != (sample_count, 3):
            raise ValueError(
                "original_aligned_center must have shape [N, 3]"
            )
        if matched_gt_box.shape != (sample_count, 6):
            raise ValueError("matched_gt_box must have shape [N, 6]")
        for name, value in (
            ("aligned_basis", aligned_basis),
            ("original_aligned_center", original_aligned_center),
            ("matched_gt_box", matched_gt_box),
        ):
            if not np.isfinite(value).all():
                raise ValueError(f"{name} must be finite")
        if (
            iou_gain.shape != (sample_count,)
            or not np.isfinite(iou_gain).all()
            or (iou_gain < 0.0).any()
            or (iou_gain > 1.0).any()
        ):
            raise ValueError("iou_gain must be finite in [0, 1]")
        expected_gain = np.maximum(refined_iou - original_iou, 0.0)
        if not np.allclose(iou_gain, expected_gain, atol=2e-5):
            raise ValueError("iou_gain disagrees with stored IoUs")
        if (
            cross_iou50.shape != (sample_count,)
            or cross_iou50.dtype != np.bool_
        ):
            raise TypeError("cross_iou50 must be Boolean with shape [N]")
        for name, value in (
            ("near_iou50", near_iou50),
            ("ap50_weight", ap50_weight),
        ):
            if value.shape != (sample_count,) or not np.isfinite(value).all():
                raise ValueError(f"{name} must be finite with shape [N]")
        if (near_iou50 < 0.0).any() or (near_iou50 > 1.0).any():
            raise ValueError("near_iou50 must lie in [0, 1]")
        if (ap50_weight < 1.0).any():
            raise ValueError("ap50_weight must be at least one")
        if (
            runtime_eligible.shape != (sample_count,)
            or runtime_eligible.dtype != np.bool_
        ):
            raise TypeError(
                "runtime_eligible must be Boolean with shape [N]"
            )
        if np.any(geometry_mask & ~runtime_eligible):
            raise ValueError(
                "geometry positives must be runtime eligible"
            )
        if (
            selected_view_counts.shape != (sample_count,)
            or selected_view_counts.dtype == np.bool_
            or not np.issubdtype(selected_view_counts.dtype, np.integer)
            or np.any(selected_view_counts < 0)
            or np.any(selected_view_counts > 5)
        ):
            raise ValueError(
                "selected_view_counts must be integer in [0, 5] with shape [N]"
            )
        for name, value in (
            ("identity_tp50", identity_tp50),
            ("candidate_oracle_tp50", candidate_oracle_tp50),
        ):
            if value.shape != (sample_count,) or value.dtype != np.bool_:
                raise TypeError(f"{name} must be Boolean with shape [N]")
            if np.any(value & (matched_gt_index < 0)):
                raise ValueError(f"{name} may only mark matched samples")
        expected_cross = (
            runtime_eligible
            & ~identity_tp50
            & candidate_oracle_tp50
        )
        if not np.array_equal(cross_iou50, expected_cross):
            raise ValueError(
                "cross_iou50 must encode eligible scene-aware TP50 events"
            )
    else:
        aligned_basis = np.tile(
            np.eye(3, dtype=np.float32), (sample_count, 1, 1)
        )
        original_aligned_center = np.zeros(
            (sample_count, 3), dtype=np.float32
        )
        matched_gt_box = np.zeros(
            (sample_count, 6), dtype=np.float32
        )
        iou_gain = np.maximum(
            refined_iou - original_iou, 0.0
        ).astype(np.float32)
        cross_iou50 = (
            geometry_mask
            & (original_iou <= AP50_EVALUATOR_THRESHOLD)
            & (refined_iou > AP50_EVALUATOR_THRESHOLD)
        )
        near_iou50 = np.zeros(sample_count, dtype=np.float32)
        ap50_weight = np.ones(sample_count, dtype=np.float32)
        runtime_eligible = np.ones(sample_count, dtype=np.bool_)
        identity_tp50 = (
            (matched_gt_index >= 0)
            & (original_iou > AP50_EVALUATOR_THRESHOLD)
        )
        candidate_oracle_tp50 = (
            (matched_gt_index >= 0)
            & (refined_iou > AP50_EVALUATOR_THRESHOLD)
        )

    return OrientedRefinerTrainingData(
        points_local=np.ascontiguousarray(points),
        point_mask=np.ascontiguousarray(point_mask),
        local_boxes=np.ascontiguousarray(local_boxes),
        quality_features=np.ascontiguousarray(quality_features),
        target_residual=np.ascontiguousarray(target_residual),
        quality_target=np.ascontiguousarray(quality_target),
        geometry_mask=np.ascontiguousarray(geometry_mask),
        scene_ids=np.ascontiguousarray(scene_ids),
        matched_gt_index=np.ascontiguousarray(matched_gt_index),
        original_iou=np.ascontiguousarray(original_iou),
        refined_iou=np.ascontiguousarray(refined_iou),
        aligned_basis=np.ascontiguousarray(aligned_basis),
        original_aligned_center=np.ascontiguousarray(
            original_aligned_center
        ),
        matched_gt_box=np.ascontiguousarray(matched_gt_box),
        iou_gain=np.ascontiguousarray(iou_gain),
        cross_iou50=np.ascontiguousarray(cross_iou50),
        near_iou50=np.ascontiguousarray(near_iou50),
        ap50_weight=np.ascontiguousarray(ap50_weight),
        runtime_eligible=np.ascontiguousarray(runtime_eligible),
        identity_tp50=np.ascontiguousarray(identity_tp50),
        candidate_oracle_tp50=np.ascontiguousarray(
            candidate_oracle_tp50
        ),
        objective=objective,
        max_center_fraction=max_center_fraction,
        max_log_dimension_residual=max_log_dimension_residual,
    )


def deterministic_scene_split(
    scene_ids: np.ndarray,
    validation_fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Split complete scenes deterministically and assert zero leakage."""

    scene_ids = np.asarray(scene_ids)
    if scene_ids.ndim != 1 or scene_ids.dtype.hasobject:
        raise TypeError("scene_ids must be a one-dimensional safe string array")
    unique_scenes = np.unique(scene_ids)
    if len(unique_scenes) < 2:
        raise ValueError("scene split requires at least two unique scenes")
    if (
        not np.isscalar(validation_fraction)
        or not np.isfinite(validation_fraction)
        or not 0.0 < float(validation_fraction) < 1.0
    ):
        raise ValueError("validation_fraction must lie strictly in (0, 1)")
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
        raise TypeError("seed must be an integer")
    validation_scene_count = int(
        round(len(unique_scenes) * float(validation_fraction))
    )
    validation_scene_count = min(
        max(validation_scene_count, 1), len(unique_scenes) - 1
    )
    permutation = np.random.default_rng(int(seed)).permutation(unique_scenes)
    validation_scenes = set(permutation[:validation_scene_count].tolist())
    validation = np.flatnonzero(
        np.isin(scene_ids, list(validation_scenes))
    ).astype(np.int64)
    training = np.flatnonzero(
        ~np.isin(scene_ids, list(validation_scenes))
    ).astype(np.int64)
    if set(scene_ids[training].tolist()) & set(scene_ids[validation].tolist()):
        raise RuntimeError("scene-level split leaked scenes")
    return training, validation


def balanced_epoch_indices(
    training_indices: np.ndarray,
    geometry_mask: np.ndarray,
    seed: int,
) -> np.ndarray:
    """Return a deterministic, uniformly sampled 50/50 epoch.

    AP50 importance is applied exactly once inside the loss.  Keeping this
    sampler uniform within each class prevents accidental squared weighting
    from weighted sampling followed by weighted loss.
    """

    indices = np.asarray(training_indices, dtype=np.int64)
    mask = np.asarray(geometry_mask, dtype=np.bool_)
    positives = indices[mask[indices]]
    negatives = indices[~mask[indices]]
    if not len(positives) or not len(negatives):
        raise ValueError(
            "training scenes must contain geometry positives and negatives"
        )
    per_class = max(len(positives), len(negatives))
    rng = np.random.default_rng(int(seed))
    positive_sample = rng.choice(
        positives,
        per_class,
        replace=len(positives) < per_class,
    )
    negative_sample = rng.choice(
        negatives,
        per_class,
        replace=len(negatives) < per_class,
    )
    combined = np.concatenate((positive_sample, negative_sample))
    rng.shuffle(combined)
    return combined.astype(np.int64, copy=False)


class _ArrayDataset(Dataset):
    def __init__(
        self, data: OrientedRefinerTrainingData, indices: np.ndarray
    ) -> None:
        self.points = torch.from_numpy(data.points_local[indices])
        self.point_mask = torch.from_numpy(data.point_mask[indices])
        self.boxes = torch.from_numpy(data.local_boxes[indices])
        self.quality_features = torch.from_numpy(
            data.quality_features[indices]
        )
        self.target_residual = torch.from_numpy(
            data.target_residual[indices]
        )
        self.quality_target = torch.from_numpy(data.quality_target[indices])
        self.geometry_mask = torch.from_numpy(data.geometry_mask[indices])
        self.original_iou = torch.from_numpy(data.original_iou[indices])
        self.aligned_basis = torch.from_numpy(data.aligned_basis[indices])
        self.original_aligned_center = torch.from_numpy(
            data.original_aligned_center[indices]
        )
        self.matched_gt_box = torch.from_numpy(
            data.matched_gt_box[indices]
        )
        self.iou_gain = torch.from_numpy(data.iou_gain[indices])
        self.cross_iou50 = torch.from_numpy(data.cross_iou50[indices])
        self.ap50_weight = torch.from_numpy(data.ap50_weight[indices])
        self.runtime_eligible = torch.from_numpy(
            data.runtime_eligible[indices]
        )
        self.identity_tp50 = torch.from_numpy(
            data.identity_tp50[indices]
        )
        self.candidate_oracle_tp50 = torch.from_numpy(
            data.candidate_oracle_tp50[indices]
        )

    def __len__(self) -> int:
        return int(len(self.points))

    def __getitem__(self, index: int) -> tuple[torch.Tensor, ...]:
        return (
            self.points[index],
            self.point_mask[index],
            self.boxes[index],
            self.quality_features[index],
            self.target_residual[index],
            self.quality_target[index],
            self.geometry_mask[index],
            self.original_iou[index],
            self.aligned_basis[index],
            self.original_aligned_center[index],
            self.matched_gt_box[index],
            self.iou_gain[index],
            self.cross_iou50[index],
            self.ap50_weight[index],
            self.runtime_eligible[index],
            self.identity_tp50[index],
            self.candidate_oracle_tp50[index],
        )


def differentiable_aligned_aabb_iou(
    output: Mapping[str, torch.Tensor],
    local_boxes: torch.Tensor,
    aligned_basis: torch.Tensor,
    original_aligned_center: torch.Tensor,
    matched_gt_box: torch.Tensor,
) -> torch.Tensor:
    """Reconstruct refined evaluator AABBs and return differentiable IoU."""

    center_fraction = output["center_residual_fraction"]
    log_dimensions = output["log_dimension_residual"]
    batch_size = local_boxes.shape[0]
    if local_boxes.shape != (batch_size, 6):
        raise ValueError("local_boxes must have shape [B, 6]")
    if aligned_basis.shape != (batch_size, 3, 3):
        raise ValueError("aligned_basis must have shape [B, 3, 3]")
    if original_aligned_center.shape != (batch_size, 3):
        raise ValueError(
            "original_aligned_center must have shape [B, 3]"
        )
    if matched_gt_box.shape != (batch_size, 6):
        raise ValueError("matched_gt_box must have shape [B, 6]")
    if center_fraction.shape != (batch_size, 3):
        raise ValueError(
            "center_residual_fraction must have shape [B, 3]"
        )
    if log_dimensions.shape != (batch_size, 3):
        raise ValueError(
            "log_dimension_residual must have shape [B, 3]"
        )

    local_dimensions = local_boxes[:, 3:6]
    center_shift_local = center_fraction * local_dimensions
    predicted_center = original_aligned_center + torch.bmm(
        aligned_basis, center_shift_local.unsqueeze(-1)
    ).squeeze(-1)
    predicted_local_dimensions = local_dimensions * torch.exp(
        log_dimensions
    )
    predicted_dimensions = torch.bmm(
        aligned_basis.abs(), predicted_local_dimensions.unsqueeze(-1)
    ).squeeze(-1)

    predicted_min = predicted_center - 0.5 * predicted_dimensions
    predicted_max = predicted_center + 0.5 * predicted_dimensions
    target_min = matched_gt_box[:, :3] - 0.5 * matched_gt_box[:, 3:6]
    target_max = matched_gt_box[:, :3] + 0.5 * matched_gt_box[:, 3:6]
    intersection_dimensions = (
        torch.minimum(predicted_max, target_max)
        - torch.maximum(predicted_min, target_min)
    ).clamp_min(0.0)
    intersection = intersection_dimensions.prod(dim=1)
    predicted_volume = predicted_dimensions.clamp_min(0.0).prod(dim=1)
    target_volume = matched_gt_box[:, 3:6].clamp_min(0.0).prod(dim=1)
    union = predicted_volume + target_volume - intersection
    return torch.where(
        union > 0.0,
        intersection / union.clamp_min(torch.finfo(union.dtype).eps),
        torch.zeros_like(union),
    )


def _weighted_masked_mean(
    values: torch.Tensor,
    mask: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    selected = values[mask]
    if selected.numel() == 0:
        return values.sum() * 0.0
    selected_weights = weights[mask]
    selected_weights = selected_weights / selected_weights.mean().clamp_min(
        torch.finfo(selected_weights.dtype).eps
    )
    return (selected * selected_weights).mean()


def local_net_tp50_proxy(
    cross_success_count: float,
    drop50_count: float,
    eligible_matched_count: float,
) -> float:
    """Return a local net-TP50 proxy with one shared denominator.

    This is deliberately *not* AP50.  It is a checkpoint-selection proxy
    over runtime-eligible, GT-matched validation examples.  True AP50 is
    computed later by the fixed-scene paired evaluator.
    """

    values = {
        "cross_success_count": cross_success_count,
        "drop50_count": drop50_count,
        "eligible_matched_count": eligible_matched_count,
    }
    converted: dict[str, float] = {}
    for name, value in values.items():
        if (
            not np.isscalar(value)
            or not np.isfinite(value)
            or float(value) < 0.0
        ):
            raise ValueError(f"{name} must be non-negative and finite")
        converted[name] = float(value)
    denominator = converted["eligible_matched_count"]
    if denominator == 0.0:
        return 0.0
    if (
        converted["cross_success_count"] > denominator
        or converted["drop50_count"] > denominator
        or (
            converted["cross_success_count"]
            + converted["drop50_count"]
            > denominator
        )
    ):
        raise ValueError("proxy event counts cannot exceed its denominator")
    return (
        converted["cross_success_count"] - converted["drop50_count"]
    ) / denominator


def oriented_refiner_loss(
    output: Mapping[str, torch.Tensor],
    target_residual: torch.Tensor,
    quality_target: torch.Tensor,
    geometry_mask: torch.Tensor,
    *,
    objective: str = "improvement",
    local_boxes: torch.Tensor | None = None,
    original_iou: torch.Tensor | None = None,
    aligned_basis: torch.Tensor | None = None,
    original_aligned_center: torch.Tensor | None = None,
    matched_gt_box: torch.Tensor | None = None,
    iou_gain_target: torch.Tensor | None = None,
    cross_iou50: torch.Tensor | None = None,
    ap50_weight: torch.Tensor | None = None,
    runtime_eligible: torch.Tensor | None = None,
    identity_tp50: torch.Tensor | None = None,
    candidate_oracle_tp50: torch.Tensor | None = None,
    center_weight: float = 1.0,
    dimension_weight: float = 1.0,
    quality_weight: float = 1.0,
    iou_gain_weight: float = 2.0,
    cross_iou50_weight: float = 4.0,
) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Train either legacy improvement or evaluator-aligned AP50 objective."""

    required = {
        "center_residual_fraction",
        "log_dimension_residual",
        "quality",
    }
    missing = required - set(output)
    if missing:
        raise ValueError(f"model output is missing {sorted(missing)}")
    if target_residual.ndim != 2 or target_residual.shape[1] != 6:
        raise ValueError("target_residual must have shape [B, 6]")
    batch_size = target_residual.shape[0]
    if quality_target.shape != (batch_size,):
        raise ValueError("quality_target must have shape [B]")
    if geometry_mask.shape != (batch_size,) or geometry_mask.dtype is not torch.bool:
        raise TypeError("geometry_mask must be Boolean with shape [B]")
    if not torch.isfinite(target_residual).all():
        raise ValueError("target_residual must be finite")
    if objective not in TRAINING_OBJECTIVES:
        raise ValueError(f"objective must be one of {TRAINING_OBJECTIVES}")
    if not torch.isfinite(quality_target).all() or torch.any(
        (quality_target < 0.0) | (quality_target > 1.0)
    ):
        raise ValueError("quality_target must lie in [0, 1]")

    unit_weights = target_residual.new_ones(batch_size)
    loss_weights = unit_weights
    if objective == "ap50":
        # Direct-call compatibility for the first schema-v2 prototype.
        # Dataset-backed training always supplies the builder's scene-aware
        # event flags below; this fallback is intentionally local-only.
        if runtime_eligible is None:
            runtime_eligible = torch.ones_like(geometry_mask)
        if identity_tp50 is None and original_iou is not None:
            identity_tp50 = original_iou > AP50_EVALUATOR_THRESHOLD
        if (
            candidate_oracle_tp50 is None
            and identity_tp50 is not None
            and cross_iou50 is not None
        ):
            candidate_oracle_tp50 = identity_tp50 | cross_iou50
        required_ap50 = (
            local_boxes,
            original_iou,
            aligned_basis,
            original_aligned_center,
            matched_gt_box,
            iou_gain_target,
            cross_iou50,
            ap50_weight,
            runtime_eligible,
            identity_tp50,
            candidate_oracle_tp50,
        )
        if any(value is None for value in required_ap50):
            raise ValueError("AP50 objective requires evaluator geometry")
        assert local_boxes is not None
        assert original_iou is not None
        assert aligned_basis is not None
        assert original_aligned_center is not None
        assert matched_gt_box is not None
        assert iou_gain_target is not None
        assert cross_iou50 is not None
        assert ap50_weight is not None
        assert runtime_eligible is not None
        assert identity_tp50 is not None
        assert candidate_oracle_tp50 is not None
        if original_iou.shape != (batch_size,):
            raise ValueError("original_iou must have shape [B]")
        if iou_gain_target.shape != (batch_size,):
            raise ValueError("iou_gain_target must have shape [B]")
        if (
            cross_iou50.shape != (batch_size,)
            or cross_iou50.dtype is not torch.bool
        ):
            raise TypeError("cross_iou50 must be Boolean with shape [B]")
        if (
            ap50_weight.shape != (batch_size,)
            or not torch.isfinite(ap50_weight).all()
            or torch.any(ap50_weight <= 0.0)
        ):
            raise ValueError(
                "ap50_weight must be positive finite with shape [B]"
            )
        for name, value in (
            ("runtime_eligible", runtime_eligible),
            ("identity_tp50", identity_tp50),
            ("candidate_oracle_tp50", candidate_oracle_tp50),
        ):
            if value.shape != (batch_size,) or value.dtype is not torch.bool:
                raise TypeError(f"{name} must be Boolean with shape [B]")
        if not torch.equal(
            cross_iou50,
            runtime_eligible & ~identity_tp50 & candidate_oracle_tp50,
        ):
            raise ValueError(
                "cross_iou50 must use scene-aware identity/candidate events"
            )
        if not torch.equal(quality_target > 0.5, geometry_mask):
            raise ValueError(
                "AP50 quality threshold must encode geometry_mask"
            )
        loss_weights = ap50_weight
    elif not torch.all(
        (quality_target == 0.0) | (quality_target == 1.0)
    ):
        raise ValueError("improvement quality_target must be binary")

    center_per_sample = F.smooth_l1_loss(
        output["center_residual_fraction"],
        target_residual[:, :3],
        reduction="none",
    ).mean(dim=1)
    dimension_per_sample = F.smooth_l1_loss(
        output["log_dimension_residual"],
        target_residual[:, 3:],
        reduction="none",
    ).mean(dim=1)
    center_loss = _weighted_masked_mean(
        center_per_sample, geometry_mask, loss_weights
    )
    dimension_loss = _weighted_masked_mean(
        dimension_per_sample, geometry_mask, loss_weights
    )
    quality = output["quality"]
    if quality.shape == (batch_size, 1):
        quality = quality[:, 0]
    if quality.shape != (batch_size,):
        raise ValueError("model quality output must have shape [B]")
    quality_per_sample = F.binary_cross_entropy(
        quality.clamp(1e-6, 1.0 - 1e-6),
        quality_target,
        reduction="none",
    )
    quality_loss = _weighted_masked_mean(
        quality_per_sample,
        torch.ones_like(geometry_mask),
        loss_weights,
    )
    zero = output["center_residual_fraction"].sum() * 0.0
    iou_gain_loss = zero
    cross_iou50_loss = zero
    predicted_iou = original_iou if original_iou is not None else quality * 0.0
    if objective == "ap50":
        assert local_boxes is not None
        assert aligned_basis is not None
        assert original_aligned_center is not None
        assert matched_gt_box is not None
        assert original_iou is not None
        assert iou_gain_target is not None
        assert cross_iou50 is not None
        predicted_iou = differentiable_aligned_aabb_iou(
            output,
            local_boxes,
            aligned_basis,
            original_aligned_center,
            matched_gt_box,
        )
        predicted_gain = predicted_iou - original_iou
        gain_per_sample = F.smooth_l1_loss(
            predicted_gain,
            iou_gain_target,
            reduction="none",
        )
        iou_gain_loss = _weighted_masked_mean(
            gain_per_sample, geometry_mask, loss_weights
        )
        # The official ScanNet evaluator requires IoU strictly greater than
        # 0.50. A small training margin avoids treating the equality boundary
        # as a completed crossing while retaining the official event test.
        cross_per_sample = torch.relu(
            AP50_LOSS_TARGET - predicted_iou
        ).square()
        cross_iou50_loss = _weighted_masked_mean(
            cross_per_sample, cross_iou50, loss_weights
        )
    total = (
        float(center_weight) * center_loss
        + float(dimension_weight) * dimension_loss
        + float(quality_weight) * quality_loss
        + float(iou_gain_weight) * iou_gain_loss
        + float(cross_iou50_weight) * cross_iou50_loss
    )
    prediction = quality >= 0.5
    accuracy = (
        prediction == quality_target.to(dtype=torch.bool)
    ).to(dtype=quality.dtype).mean()
    metrics = {
        "loss": total.detach(),
        "center_loss": center_loss.detach(),
        "dimension_loss": dimension_loss.detach(),
        "quality_loss": quality_loss.detach(),
        "quality_accuracy": accuracy.detach(),
        "iou_gain_loss": iou_gain_loss.detach(),
        "cross_iou50_loss": cross_iou50_loss.detach(),
        "geometry_positive_fraction": geometry_mask.to(
            dtype=quality.dtype
        ).mean().detach(),
    }
    if objective == "ap50":
        assert original_iou is not None
        assert cross_iou50 is not None
        assert runtime_eligible is not None
        assert identity_tp50 is not None
        assert candidate_oracle_tp50 is not None
        accepted = quality >= 0.5
        matched = matched_gt_box[:, 3:6].prod(dim=1) > 0.0
        eligible_matched = runtime_eligible & matched
        cross_event = (
            eligible_matched
            & ~identity_tp50
            & candidate_oracle_tp50
        )
        metrics.update(
            {
                "cross50_success_count": (
                    cross_event
                    & accepted
                    & (predicted_iou > AP50_EVALUATOR_THRESHOLD)
                ).sum().detach(),
                "cross50_event_count": cross_event.sum().detach(),
                "drop50_count": (
                    eligible_matched
                    & identity_tp50
                    & accepted
                    & (predicted_iou <= AP50_EVALUATOR_THRESHOLD)
                ).sum().detach(),
                "identity_tp50_count": (
                    eligible_matched & identity_tp50
                ).sum().detach(),
                "predicted_gain_sum": (
                    (predicted_iou - original_iou)
                    * eligible_matched.to(dtype=predicted_iou.dtype)
                ).sum().detach(),
                "eligible_matched_count": eligible_matched.sum().detach(),
            }
        )
    return total, metrics


def _set_determinism(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)


def _run_epoch(
    model: PointNetOrientedBoxRefiner,
    loader: DataLoader,
    *,
    optimizer: Optional[torch.optim.Optimizer],
    objective: str,
    loss_weights: Mapping[str, float],
) -> Dict[str, float]:
    training = optimizer is not None
    model.train(training)
    totals: Dict[str, float] = {}
    total_samples = 0
    for (
        points,
        point_mask,
        boxes,
        quality_features,
        target_residual,
        quality_target,
        geometry_mask,
        original_iou,
        aligned_basis,
        original_aligned_center,
        matched_gt_box,
        iou_gain,
        cross_iou50,
        ap50_weight,
        runtime_eligible,
        identity_tp50,
        candidate_oracle_tp50,
    ) in loader:
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            output = model(
                points, boxes, quality_features, point_mask=point_mask
            )
            loss, metrics = oriented_refiner_loss(
                output,
                target_residual,
                quality_target,
                geometry_mask,
                objective=objective,
                local_boxes=boxes,
                original_iou=original_iou,
                aligned_basis=aligned_basis,
                original_aligned_center=original_aligned_center,
                matched_gt_box=matched_gt_box,
                iou_gain_target=iou_gain,
                cross_iou50=cross_iou50,
                ap50_weight=ap50_weight,
                runtime_eligible=runtime_eligible,
                identity_tp50=identity_tp50,
                candidate_oracle_tp50=candidate_oracle_tp50,
                **loss_weights,
            )
            if optimizer is not None:
                loss.backward()
                optimizer.step()
        batch_size = int(points.shape[0])
        total_samples += batch_size
        for name, value in metrics.items():
            if name.endswith("_count") or name.endswith("_sum"):
                totals[name] = totals.get(name, 0.0) + float(value)
            else:
                totals[name] = (
                    totals.get(name, 0.0) + float(value) * batch_size
                )
    if total_samples == 0:
        raise ValueError("data loader produced no samples")
    result = {
        name: (
            value
            if name.endswith("_count") or name.endswith("_sum")
            else value / total_samples
        )
        for name, value in totals.items()
    }
    if objective == "ap50":
        denominator = result.get("eligible_matched_count", 0.0)
        cross_success = result.get("cross50_success_count", 0.0)
        drop50 = result.get("drop50_count", 0.0)
        result["local_cross50_success_rate"] = (
            cross_success / denominator if denominator > 0.0 else 0.0
        )
        result["local_drop50_rate"] = (
            drop50 / denominator if denominator > 0.0 else 0.0
        )
        result["mean_predicted_gain"] = (
            result.get("predicted_gain_sum", 0.0) / denominator
            if denominator > 0.0
            else 0.0
        )
        proxy = local_net_tp50_proxy(
            cross_success, drop50, denominator
        )
        result["local_net_tp50_proxy"] = proxy
        # Compatibility alias for existing experiment readers.  The explicit
        # name above documents that this is not evaluator AP50.
        result["ap50_proxy"] = proxy
    return result


def _positive_finite(name: str, value: float) -> float:
    if (
        not np.isscalar(value)
        or not np.isfinite(value)
        or float(value) <= 0.0
    ):
        raise ValueError(f"{name} must be positive and finite")
    return float(value)


def train_oriented_box_refiner(
    dataset_path: str | os.PathLike,
    output_path: str | os.PathLike,
    *,
    objective: str = "improvement",
    config: Optional[OrientedBoxRefinerConfig] = None,
    epochs: int = 30,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-5,
    validation_fraction: float = 0.2,
    seed: int = 1337,
    center_weight: float = 1.0,
    dimension_weight: float = 1.0,
    quality_weight: float = 1.0,
    iou_gain_weight: float = 2.0,
    cross_iou50_weight: float = 4.0,
) -> Dict[str, object]:
    """Train B5-v2 and write a strict runtime-compatible checkpoint."""

    for name, value in (("epochs", epochs), ("batch_size", batch_size)):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise TypeError(f"{name} must be an integer")
        if int(value) < 1:
            raise ValueError(f"{name} must be positive")
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
        raise TypeError("seed must be an integer")
    if objective not in TRAINING_OBJECTIVES:
        raise ValueError(f"objective must be one of {TRAINING_OBJECTIVES}")
    learning_rate = _positive_finite("learning_rate", learning_rate)
    if (
        not np.isscalar(weight_decay)
        or not np.isfinite(weight_decay)
        or float(weight_decay) < 0.0
    ):
        raise ValueError("weight_decay must be non-negative and finite")
    loss_weights = {
        name: _positive_finite(name, value)
        for name, value in (
            ("center_weight", center_weight),
            ("dimension_weight", dimension_weight),
            ("quality_weight", quality_weight),
            ("iou_gain_weight", iou_gain_weight),
            ("cross_iou50_weight", cross_iou50_weight),
        )
    }
    data = load_oriented_refiner_dataset(dataset_path)
    if data.objective != objective:
        raise ValueError(
            f"dataset objective {data.objective!r} does not match "
            f"requested objective {objective!r}"
        )
    model_config = (config or OrientedBoxRefinerConfig()).validated()
    if model_config.point_feature_dim != 3:
        raise ValueError("B5-v2 training data contains xyz only")
    if model_config.quality_feature_dim != QUALITY_FEATURE_DIM:
        raise ValueError("quality_feature_dim does not match runtime schema")
    if not np.isclose(
        model_config.max_center_fraction,
        data.max_center_fraction,
        atol=1e-7,
    ):
        raise ValueError("model and dataset max_center_fraction differ")
    if not np.isclose(
        model_config.max_log_dimension_residual,
        data.max_log_dimension_residual,
        atol=1e-7,
    ):
        raise ValueError(
            "model and dataset max_log_dimension_residual differ"
        )
    training_indices, validation_indices = deterministic_scene_split(
        data.scene_ids, validation_fraction, seed
    )
    training_scenes = sorted(set(data.scene_ids[training_indices].tolist()))
    validation_scenes = sorted(
        set(data.scene_ids[validation_indices].tolist())
    )
    if set(training_scenes) & set(validation_scenes):
        raise RuntimeError("train/validation scene leakage")
    # Fail before constructing a model: balancing cannot invent a missing
    # class, and silently falling back would violate the B5-v2 protocol.
    balanced_epoch_indices(
        training_indices,
        data.geometry_mask,
        seed,
    )
    _set_determinism(int(seed))

    validation_loader = DataLoader(
        _ArrayDataset(data, validation_indices),
        batch_size=min(int(batch_size), len(validation_indices)),
        shuffle=False,
        num_workers=0,
    )
    model = PointNetOrientedBoxRefiner(model_config).cpu()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=float(weight_decay),
    )
    train_metrics: Dict[str, float] = {}
    validation_metrics: Dict[str, float] = {}
    best_validation_loss = float("inf")
    best_validation_proxy = -float("inf")
    best_state: Dict[str, torch.Tensor] | None = None
    best_epoch: int | None = None
    best_train_metrics: Dict[str, float] | None = None
    best_validation_metrics: Dict[str, float] | None = None
    for epoch in range(int(epochs)):
        epoch_indices = balanced_epoch_indices(
            training_indices,
            data.geometry_mask,
            int(seed) + epoch,
        )
        training_loader = DataLoader(
            _ArrayDataset(data, epoch_indices),
            batch_size=min(int(batch_size), len(epoch_indices)),
            shuffle=False,
            num_workers=0,
        )
        train_metrics = _run_epoch(
            model,
            training_loader,
            optimizer=optimizer,
            objective=objective,
            loss_weights=loss_weights,
        )
        validation_metrics = _run_epoch(
            model,
            validation_loader,
            optimizer=None,
            objective=objective,
            loss_weights=loss_weights,
        )
        validation_proxy = validation_metrics.get(
            "local_net_tp50_proxy", -float("inf")
        )
        # The local TP50 proxy cannot replay the model prediction through the
        # full 8192-point/reprojection gate without bloating the training
        # archive. Select checkpoints by the complete held-out objective loss
        # and use the proxy only as a deterministic tie-break/diagnostic.
        better = (
            validation_metrics["loss"] < best_validation_loss - 1e-12
            or (
                math.isclose(
                    validation_metrics["loss"],
                    best_validation_loss,
                    abs_tol=1e-12,
                )
                and validation_proxy > best_validation_proxy
            )
        )
        if better:
            best_validation_loss = validation_metrics["loss"]
            best_validation_proxy = validation_proxy
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
            best_epoch = int(epoch)
            best_train_metrics = dict(train_metrics)
            best_validation_metrics = dict(validation_metrics)
    if (
        best_state is None
        or best_epoch is None
        or best_train_metrics is None
        or best_validation_metrics is None
    ):
        raise RuntimeError("training did not produce a checkpoint state")
    model.load_state_dict(best_state, strict=True)

    output = Path(output_path)
    if output.suffix.lower() not in {".pt", ".pth"}:
        raise ValueError("B5-v2 checkpoint must end in .pt or .pth")
    output.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = make_oriented_box_refiner_checkpoint(model)
    temporary = output.with_name(output.name + ".tmp")
    try:
        torch.save(checkpoint, temporary)
        os.replace(temporary, output)
    finally:
        if temporary.exists():
            temporary.unlink()
    return {
        "output": str(output),
        "samples": data.sample_count,
        "train_samples": int(len(training_indices)),
        "validation_samples": int(len(validation_indices)),
        "train_scenes": training_scenes,
        "validation_scenes": validation_scenes,
        "scene_leakage": False,
        "objective": objective,
        "epochs": int(epochs),
        "best_epoch": best_epoch,
        "seed": int(seed),
        "balanced_epoch_samples": int(
            len(
                balanced_epoch_indices(
                    training_indices,
                    data.geometry_mask,
                    seed,
                )
            )
        ),
        "best_validation_loss": float(best_validation_loss),
        "best_validation_ap50_proxy": (
            float(best_validation_proxy)
            if objective == "ap50"
            else None
        ),
        "best_validation_local_net_tp50_proxy": (
            float(best_validation_proxy)
            if objective == "ap50"
            else None
        ),
        "checkpoint_selection_note": (
            "minimum scene-held-out AP50-aware validation loss; local "
            "net-TP50 proxy is diagnostic only because it does not replay "
            "the full runtime gate; fixed-scene paired report is true AP"
            if objective == "ap50"
            else "minimum scene-held-out validation loss"
        ),
        "train": best_train_metrics,
        "validation": best_validation_metrics,
    }


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="B5-v2 training NPZ")
    parser.add_argument("--output", required=True, help="output .pt checkpoint")
    parser.add_argument(
        "--objective",
        choices=TRAINING_OBJECTIVES,
        default="improvement",
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--point-hidden-dim", type=int, default=64)
    parser.add_argument("--point-embedding-dim", type=int, default=128)
    parser.add_argument("--head-hidden-dim", type=int, default=128)
    parser.add_argument("--max-center-fraction", type=float, default=0.15)
    parser.add_argument(
        "--max-log-dimension-residual",
        type=float,
        default=float(math.log(1.25)),
    )
    parser.add_argument("--center-weight", type=float, default=1.0)
    parser.add_argument("--dimension-weight", type=float, default=1.0)
    parser.add_argument("--quality-weight", type=float, default=1.0)
    parser.add_argument("--iou-gain-weight", type=float, default=2.0)
    parser.add_argument("--cross-iou50-weight", type=float, default=4.0)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_argument_parser()
    arguments = parser.parse_args(argv)
    try:
        config = OrientedBoxRefinerConfig(
            point_feature_dim=3,
            quality_feature_dim=QUALITY_FEATURE_DIM,
            point_hidden_dim=arguments.point_hidden_dim,
            point_embedding_dim=arguments.point_embedding_dim,
            head_hidden_dim=arguments.head_hidden_dim,
            max_center_fraction=arguments.max_center_fraction,
            max_log_dimension_residual=(
                arguments.max_log_dimension_residual
            ),
        ).validated()
        result = train_oriented_box_refiner(
            arguments.input,
            arguments.output,
            objective=arguments.objective,
            config=config,
            epochs=arguments.epochs,
            batch_size=arguments.batch_size,
            learning_rate=arguments.learning_rate,
            weight_decay=arguments.weight_decay,
            validation_fraction=arguments.validation_fraction,
            seed=arguments.seed,
            center_weight=arguments.center_weight,
            dimension_weight=arguments.dimension_weight,
            quality_weight=arguments.quality_weight,
            iou_gain_weight=arguments.iou_gain_weight,
            cross_iou50_weight=arguments.cross_iou50_weight,
        )
    except (FileNotFoundError, TypeError, ValueError) as error:
        parser.error(str(error))
    json.dump(result, sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
