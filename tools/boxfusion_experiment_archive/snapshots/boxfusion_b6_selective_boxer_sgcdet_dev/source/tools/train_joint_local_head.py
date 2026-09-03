#!/usr/bin/env python3
"""Train the joint K=5 local geometry/quality head deterministically on CPU.

The input is the pickle-free archive produced by
``tools/build_joint_local_dataset.py``.  Complete ScanNet scenes are held out,
geometry positives/rejections are sampled uniformly by class, and AP50
importance weights are applied exactly once inside the losses.

The candidate quality branch is supervised against the IoU *realized by the
model's current residual*, detached from the geometry graph.  It is not
trained against the oracle B5 target candidate, which would violate the
runtime contract whenever the predicted residual differs from that oracle.
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
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np

try:
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Dataset
except ImportError as error:  # pragma: no cover
    raise ImportError("joint local-head training requires PyTorch") from error

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

from boxfusion.joint_local_head import (
    JOINT_LOCAL_HEAD_COORDINATE_FRAME,
    JOINT_LOCAL_HEAD_INPUT_SCHEMA,
    JOINT_QUALITY_BRANCH_NAMES,
    JOINT_QUALITY_COMPONENT_NAMES,
    JOINT_VIEW_FEATURE_DIM,
    JOINT_VIEW_FEATURE_NAMES,
    JointLocalHeadConfig,
    MultiViewJointLocalHead,
    load_joint_local_head_checkpoint,
    make_joint_local_head_checkpoint,
)
from boxfusion.quality_score import (
    IOU_AWARE_THRESHOLDS,
    QUALITY_FEATURE_DIM,
    QUALITY_FEATURE_NAMES,
)
from tools.build_joint_local_dataset import (
    EXPECTED_TOP_K_VIEWS,
    LEGACY_REFINER_QUALITY_NORMALIZATION,
    JOINT_LOCAL_DATASET_FORMAT_VERSION,
    JOINT_LOCAL_DATASET_SCHEMA,
    JOINT_METADATA_KEYS,
    JOINT_SAMPLE_KEYS,
    RUNTIME_QUALITY_FEATURE_SOURCE,
    validate_runtime_quality_feature_relation,
)
from tools.build_oriented_refiner_dataset import (
    AP50_DATASET_FORMAT_VERSION,
    DATASET_SCHEMA as B5_DATASET_SCHEMA,
    strict_provenance_for_profile,
)
from tools.train_oriented_box_refiner import (
    balanced_epoch_indices,
    deterministic_scene_split,
    differentiable_aligned_aabb_iou,
)


SCENE_PATTERN = re.compile(r"scene\d{4}_\d{2}")
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
AP50_EVALUATOR_THRESHOLD = 0.50
AP50_LOSS_TARGET = 0.5001


@dataclass(frozen=True)
class JointLocalTrainingData:
    points_local: np.ndarray
    point_mask: np.ndarray
    view_features: np.ndarray
    view_mask: np.ndarray
    local_boxes: np.ndarray
    quality_features: np.ndarray
    target_residual: np.ndarray
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
    ap50_weight: np.ndarray
    runtime_eligible: np.ndarray
    identity_tp50: np.ndarray
    candidate_oracle_tp50: np.ndarray
    max_center_fraction: float
    max_log_dimension_residual: float
    source_dataset_sha256: str
    forbidden_scene_count: int
    forbidden_scene_sha256: str
    training_scene_sha256: str
    points_per_view: int

    @property
    def sample_count(self) -> int:
        return int(self.points_local.shape[0])


def _scalar_string(value: Any, name: str) -> str:
    array = np.asarray(value)
    if array.ndim != 0 or array.dtype.hasobject:
        raise TypeError(f"{name} must be a safe scalar string")
    scalar = array.item()
    if isinstance(scalar, bytes):
        scalar = scalar.decode("utf-8")
    if not isinstance(scalar, str):
        raise TypeError(f"{name} must be a string")
    return scalar


def _scalar_integer(value: Any, name: str) -> int:
    array = np.asarray(value)
    if (
        array.ndim != 0
        or array.dtype == np.bool_
        or not np.issubdtype(array.dtype, np.integer)
    ):
        raise TypeError(f"{name} must be an integer scalar")
    return int(array)


def _scalar_float(value: Any, name: str) -> float:
    array = np.asarray(value)
    if (
        array.ndim != 0
        or array.dtype == np.bool_
        or not np.issubdtype(array.dtype, np.number)
    ):
        raise TypeError(f"{name} must be a numeric scalar")
    result = float(array)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _scalar_boolean(value: Any, name: str) -> bool:
    array = np.asarray(value)
    if array.ndim != 0 or array.dtype != np.bool_:
        raise TypeError(f"{name} must be a Boolean scalar")
    return bool(array)


def _sequence_strings(value: Any, name: str) -> tuple[str, ...]:
    array = np.asarray(value)
    if array.ndim != 1 or array.dtype.hasobject or array.dtype.kind not in {
        "U",
        "S",
    }:
        raise TypeError(f"{name} must be a safe one-dimensional string array")
    return tuple(str(item) for item in array.tolist())


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _scene_sha256(scene_ids: Sequence[str]) -> str:
    canonical = "\n".join(sorted(set(scene_ids))) + "\n"
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _provenance_matches(actual: Any, expected: Any) -> bool:
    if isinstance(expected, bool):
        return isinstance(actual, (bool, np.bool_)) and bool(actual) == expected
    if isinstance(expected, int):
        return (
            not isinstance(actual, (bool, np.bool_))
            and isinstance(actual, (int, np.integer))
            and int(actual) == expected
        )
    if isinstance(expected, float):
        return (
            not isinstance(actual, (bool, np.bool_))
            and np.isscalar(actual)
            and np.isfinite(actual)
            and bool(
                np.isclose(float(actual), expected, atol=1e-8, rtol=0.0)
            )
        )
    return actual == expected


def _validate_metadata(
    arrays: Mapping[str, np.ndarray],
) -> tuple[float, float, str, int, str, str, int]:
    if _scalar_string(arrays["schema"], "schema") != JOINT_LOCAL_DATASET_SCHEMA:
        raise ValueError("joint dataset schema mismatch")
    if (
        _scalar_integer(arrays["format_version"], "format_version")
        != JOINT_LOCAL_DATASET_FORMAT_VERSION
    ):
        raise ValueError("joint dataset format version mismatch")
    if (
        _scalar_string(
            arrays["source_dataset_schema"], "source_dataset_schema"
        )
        != B5_DATASET_SCHEMA
        or _scalar_integer(
            arrays["source_dataset_format_version"],
            "source_dataset_format_version",
        )
        != AP50_DATASET_FORMAT_VERSION
    ):
        raise ValueError("joint dataset source is not strict B5-v2")
    if _scalar_string(arrays["objective"], "objective") != "ap50":
        raise ValueError("joint dataset requires objective='ap50'")
    if not _scalar_boolean(
        arrays["strict_k5_diagnostics"], "strict_k5_diagnostics"
    ):
        raise ValueError("joint dataset requires strict K5 diagnostics")
    for name in ("expected_top_k_views", "joint_top_k_views"):
        if _scalar_integer(arrays[name], name) != EXPECTED_TOP_K_VIEWS:
            raise ValueError(f"{name} must equal 5")
    points_per_view = _scalar_integer(
        arrays["joint_points_per_view"], "joint_points_per_view"
    )
    if points_per_view != 128:
        raise ValueError("joint runtime/training contract requires P=128")
    if (
        _scalar_string(arrays["coordinate_frame"], "coordinate_frame")
        != JOINT_LOCAL_HEAD_COORDINATE_FRAME
        or _scalar_string(arrays["input_schema"], "input_schema")
        != JOINT_LOCAL_HEAD_INPUT_SCHEMA
    ):
        raise ValueError("joint coordinate/input schema mismatch")
    if (
        _sequence_strings(
            arrays["view_feature_names"], "view_feature_names"
        )
        != JOINT_VIEW_FEATURE_NAMES
        or _sequence_strings(
            arrays["quality_feature_names"], "quality_feature_names"
        )
        != QUALITY_FEATURE_NAMES
        or _sequence_strings(
            arrays["quality_branch_names"], "quality_branch_names"
        )
        != JOINT_QUALITY_BRANCH_NAMES
        or _sequence_strings(
            arrays["quality_component_names"], "quality_component_names"
        )
        != JOINT_QUALITY_COMPONENT_NAMES
    ):
        raise ValueError("joint feature/output schema order mismatch")
    thresholds = np.asarray(arrays["iou_thresholds"])
    if (
        thresholds.shape != (len(IOU_AWARE_THRESHOLDS),)
        or thresholds.dtype != np.float32
        or not np.array_equal(
            thresholds, np.asarray(IOU_AWARE_THRESHOLDS, dtype=np.float32)
        )
    ):
        raise ValueError("joint IoU thresholds mismatch")
    if not np.isclose(
        _scalar_float(
            arrays["appearance_consistency_default"],
            "appearance_consistency_default",
        ),
        0.5,
        atol=1e-8,
        rtol=0.0,
    ):
        raise ValueError("appearance_consistency default must equal 0.5")
    if (
        _scalar_string(
            arrays["quality_feature_source"], "quality_feature_source"
        )
        != RUNTIME_QUALITY_FEATURE_SOURCE
    ):
        raise ValueError(
            "joint model inputs must use runtime-exact quality features"
        )
    if (
        _scalar_string(
            arrays["legacy_refiner_quality_normalization"],
            "legacy_refiner_quality_normalization",
        )
        != LEGACY_REFINER_QUALITY_NORMALIZATION
    ):
        raise ValueError("joint refiner-quality normalization mismatch")
    normalized_rows = _scalar_integer(
        arrays["legacy_refiner_quality_normalized_rows"],
        "legacy_refiner_quality_normalized_rows",
    )
    joint_sample_count = _scalar_integer(
        arrays["joint_sample_count"], "joint_sample_count"
    )
    if normalized_rows < 0 or normalized_rows > joint_sample_count:
        raise ValueError(
            "legacy refiner-quality normalized-row count is invalid"
        )
    provenance_profile = _scalar_string(
        arrays["online_ablation_profile"], "online_ablation_profile"
    )
    expected_provenance = strict_provenance_for_profile(
        provenance_profile
    )
    for name, expected in expected_provenance.items():
        array = np.asarray(arrays[name])
        if array.ndim != 0:
            raise TypeError(f"{name} must be scalar")
        actual = array.item()
        if isinstance(actual, bytes):
            actual = actual.decode("utf-8")
        if not _provenance_matches(actual, expected):
            raise ValueError(f"strict B5 provenance {name} mismatch")

    source_digest = _scalar_string(
        arrays["source_dataset_sha256"], "source_dataset_sha256"
    )
    forbidden_digest = _scalar_string(
        arrays["forbidden_scene_sha256"], "forbidden_scene_sha256"
    )
    training_digest = _scalar_string(
        arrays["training_scene_sha256"], "training_scene_sha256"
    )
    diagnostic_digest = _scalar_string(
        arrays["diagnostic_scene_sha256"], "diagnostic_scene_sha256"
    )
    source_training_digest = _scalar_string(
        arrays["source_training_scene_sha256"],
        "source_training_scene_sha256",
    )
    for name, digest in (
        ("source_dataset_sha256", source_digest),
        ("forbidden_scene_sha256", forbidden_digest),
        ("training_scene_sha256", training_digest),
        ("diagnostic_scene_sha256", diagnostic_digest),
        ("source_training_scene_sha256", source_training_digest),
    ):
        if SHA256_PATTERN.fullmatch(digest) is None:
            raise ValueError(f"{name} is not a lowercase SHA-256 digest")
    if training_digest != diagnostic_digest:
        raise ValueError("training and diagnostic scene hashes differ")
    forbidden_count = _scalar_integer(
        arrays["forbidden_scene_count"], "forbidden_scene_count"
    )
    if forbidden_count <= 0:
        raise ValueError("forbidden_scene_count must be positive")
    source_training_count = _scalar_integer(
        arrays["source_training_scene_count"], "source_training_scene_count"
    )
    training_count = _scalar_integer(
        arrays["training_scene_count"], "training_scene_count"
    )
    if source_training_count < training_count or training_count <= 0:
        raise ValueError("source/retained training scene counts are invalid")
    max_center = _scalar_float(
        arrays["max_center_fraction"], "max_center_fraction"
    )
    max_log_dimension = _scalar_float(
        arrays["max_log_dimension_residual"],
        "max_log_dimension_residual",
    )
    if max_center <= 0.0 or max_log_dimension <= 0.0:
        raise ValueError("joint geometry residual bounds must be positive")
    return (
        max_center,
        max_log_dimension,
        source_digest,
        forbidden_count,
        forbidden_digest,
        training_digest,
        points_per_view,
    )


def load_joint_local_dataset(
    path: str | os.PathLike[str],
) -> JointLocalTrainingData:
    """Load and fail-closed validate a joint training archive."""

    dataset_path = Path(path)
    if not dataset_path.is_file():
        raise FileNotFoundError(dataset_path)
    if dataset_path.suffix.lower() != ".npz":
        raise ValueError("joint training dataset must end in .npz")
    try:
        with np.load(dataset_path, allow_pickle=False) as archive:
            expected = JOINT_SAMPLE_KEYS | JOINT_METADATA_KEYS
            keys = set(archive.files)
            if keys != expected:
                raise ValueError(
                    "joint dataset keys are invalid: "
                    f"missing={sorted(expected - keys)}, "
                    f"unexpected={sorted(keys - expected)}"
                )
            arrays = {
                name: np.asarray(archive[name]).copy()
                for name in archive.files
            }
    except ValueError as error:
        if "Object arrays cannot be loaded" in str(error):
            raise ValueError(
                f"{dataset_path} contains forbidden object arrays"
            ) from error
        raise
    if any(value.dtype.hasobject for value in arrays.values()):
        raise ValueError("joint dataset must not contain object arrays")
    (
        max_center,
        max_log_dimension,
        source_digest,
        forbidden_count,
        forbidden_digest,
        training_digest,
        points_per_view,
    ) = _validate_metadata(arrays)
    normalized_rows = _scalar_integer(
        arrays["legacy_refiner_quality_normalized_rows"],
        "legacy_refiner_quality_normalized_rows",
    )
    provenance_profile = _scalar_string(
        arrays["online_ablation_profile"], "online_ablation_profile"
    )

    # Runtime-exact joint arrays.
    points = arrays["joint_points_local"]
    point_mask = arrays["joint_point_mask"]
    view_features = arrays["joint_view_features"]
    view_mask = arrays["joint_view_mask"]
    if (
        points.ndim != 4
        or points.shape[0] < 2
        or points.shape[1:] != (
            EXPECTED_TOP_K_VIEWS,
            points_per_view,
            3,
        )
        or points.dtype != np.float32
    ):
        raise TypeError("joint_points_local must be float32 [N,5,128,3]")
    sample_count = int(points.shape[0])
    if (
        point_mask.shape != points.shape[:-1]
        or point_mask.dtype != np.bool_
    ):
        raise TypeError("joint_point_mask must be Boolean [N,5,128]")
    if (
        view_features.shape
        != (sample_count, EXPECTED_TOP_K_VIEWS, JOINT_VIEW_FEATURE_DIM)
        or view_features.dtype != np.float32
    ):
        raise TypeError("joint_view_features must be float32 [N,5,9]")
    if (
        view_mask.shape != (sample_count, EXPECTED_TOP_K_VIEWS)
        or view_mask.dtype != np.bool_
    ):
        raise TypeError("joint_view_mask must be Boolean [N,5]")
    if not np.array_equal(view_mask, point_mask.any(axis=2)):
        raise ValueError("joint view/point masks are misaligned")
    if not view_mask.any(axis=1).all():
        raise ValueError("every joint sample must contain a valid view")
    if (
        not np.isfinite(points).all()
        or not np.all(points[~point_mask] == 0.0)
    ):
        raise ValueError("joint local points/padding are invalid")
    if (
        not np.isfinite(view_features).all()
        or (view_features < 0.0).any()
        or (view_features > 1.0).any()
        or not np.all(view_features[~view_mask] == 0.0)
    ):
        raise ValueError("joint view features/padding are invalid")

    # Use the exact tensors serialized by the active joint runtime.  The
    # strict B5 fields remain in the archive solely as immutable target and
    # provenance inputs.
    source_local_boxes = arrays["local_boxes"]
    local_boxes = arrays["joint_local_boxes"]
    source_quality_features = arrays["quality_features"]
    quality_features = arrays["joint_quality_features"]
    target_residual = arrays["target_residual"]
    for name, value, shape in (
        ("local_boxes", local_boxes, (sample_count, 6)),
        (
            "quality_features",
            quality_features,
            (sample_count, QUALITY_FEATURE_DIM),
        ),
        ("target_residual", target_residual, (sample_count, 6)),
    ):
        if value.shape != shape or value.dtype != np.float32:
            raise TypeError(f"{name} must be float32 with shape {shape}")
        if not np.isfinite(value).all():
            raise ValueError(f"{name} must be finite")
    if (
        not np.allclose(local_boxes[:, :3], 0.0, atol=1e-7, rtol=0.0)
        or (local_boxes[:, 3:6] <= 0.0).any()
    ):
        raise ValueError("joint local boxes are invalid")
    if not np.array_equal(source_local_boxes, local_boxes):
        raise ValueError(
            "strict B5 and runtime-exact joint local boxes disagree"
        )
    actual_normalized_rows = validate_runtime_quality_feature_relation(
        source_quality_features, quality_features
    )
    if actual_normalized_rows != normalized_rows:
        raise ValueError(
            "legacy refiner-quality normalized-row metadata is inconsistent"
        )
    if (
        (quality_features < 0.0).any()
        or (quality_features > 1.0).any()
    ):
        raise ValueError("quality_features must lie in [0,1]")
    appearance_index = QUALITY_FEATURE_NAMES.index(
        "appearance_consistency"
    )
    if (
        provenance_profile == "b5v2_memory_observer"
        and not np.allclose(
            quality_features[:, appearance_index],
            0.5,
            atol=1e-7,
            rtol=0.0,
        )
    ):
        raise ValueError("appearance_consistency must be fixed to 0.5")
    if (
        np.abs(target_residual[:, :3]) > max_center + 1e-5
    ).any() or (
        np.abs(target_residual[:, 3:]) > max_log_dimension + 1e-5
    ).any():
        raise ValueError("target residual exceeds joint architecture bounds")

    bool_names = (
        "geometry_mask",
        "cross_iou50",
        "runtime_eligible",
        "identity_tp50",
        "candidate_oracle_tp50",
    )
    for name in bool_names:
        value = arrays[name]
        if value.shape != (sample_count,) or value.dtype != np.bool_:
            raise TypeError(f"{name} must be Boolean [N]")
    geometry_mask = arrays["geometry_mask"]
    if not geometry_mask.any() or geometry_mask.all():
        raise ValueError(
            "joint dataset must contain geometry positives and rejections"
        )
    if np.any(geometry_mask & ~arrays["runtime_eligible"]):
        raise ValueError("geometry positives must be runtime eligible")

    float_vector_names = (
        "original_iou",
        "refined_iou",
        "iou_gain",
        "near_iou50",
        "ap50_weight",
    )
    for name in float_vector_names:
        value = arrays[name]
        if (
            value.shape != (sample_count,)
            or value.dtype != np.float32
            or not np.isfinite(value).all()
        ):
            raise TypeError(f"{name} must be finite float32 [N]")
    original_iou = arrays["original_iou"]
    refined_iou = arrays["refined_iou"]
    if (
        (original_iou < 0.0).any()
        or (original_iou > 1.0).any()
        or (refined_iou < 0.0).any()
        or (refined_iou > 1.0).any()
    ):
        raise ValueError("stored IoUs must lie in [0,1]")
    expected_gain = np.maximum(refined_iou - original_iou, 0.0)
    if not np.allclose(
        arrays["iou_gain"], expected_gain, atol=2e-5, rtol=0.0
    ):
        raise ValueError("iou_gain disagrees with stored source IoUs")
    if (arrays["ap50_weight"] < 1.0).any():
        raise ValueError("ap50_weight must be at least one")
    if (
        (arrays["near_iou50"] < 0.0).any()
        or (arrays["near_iou50"] > 1.0).any()
    ):
        raise ValueError("near_iou50 must lie in [0,1]")

    scene_ids = arrays["scene_ids"]
    if (
        scene_ids.shape != (sample_count,)
        or scene_ids.dtype.hasobject
        or scene_ids.dtype.kind not in {"U", "S"}
    ):
        raise TypeError("scene_ids must be a safe string array [N]")
    scene_ids = scene_ids.astype(np.str_)
    unique_scenes = sorted(np.unique(scene_ids).tolist())
    if (
        len(unique_scenes) < 2
        or any(SCENE_PATTERN.fullmatch(scene) is None for scene in unique_scenes)
    ):
        raise ValueError("joint scene ids are invalid")
    if (
        _scalar_integer(arrays["training_scene_count"], "training_scene_count")
        != len(unique_scenes)
        or _scalar_integer(
            arrays["diagnostic_scene_count"], "diagnostic_scene_count"
        )
        != len(unique_scenes)
        or _scene_sha256(unique_scenes) != training_digest
    ):
        raise ValueError("joint scene provenance is inconsistent")
    if (
        _scalar_integer(arrays["joint_sample_count"], "joint_sample_count")
        != sample_count
    ):
        raise ValueError("joint_sample_count disagrees with arrays")

    matched = arrays["matched_gt_index"]
    if (
        matched.shape != (sample_count,)
        or matched.dtype != np.int64
        or (matched < -1).any()
    ):
        raise TypeError("matched_gt_index must be int64 [N] in [-1,...)")
    for name in ("result_indices", "track_ids"):
        value = arrays[name]
        if value.shape != (sample_count,) or value.dtype != np.int64:
            raise TypeError(f"{name} must be int64 [N]")
    for scene in unique_scenes:
        rows = scene_ids == scene
        if len(np.unique(arrays["result_indices"][rows])) != int(rows.sum()):
            raise ValueError(f"{scene}: result_indices are not unique")

    aligned_basis = arrays["aligned_basis"]
    aligned_center = arrays["original_aligned_center"]
    matched_gt_box = arrays["matched_gt_box"]
    for name, value, shape in (
        ("aligned_basis", aligned_basis, (sample_count, 3, 3)),
        (
            "original_aligned_center",
            aligned_center,
            (sample_count, 3),
        ),
        ("matched_gt_box", matched_gt_box, (sample_count, 6)),
    ):
        if value.shape != shape or value.dtype != np.float32:
            raise TypeError(f"{name} must be float32 with shape {shape}")
        if not np.isfinite(value).all():
            raise ValueError(f"{name} must be finite")
    if np.any(
        arrays["identity_tp50"] & (matched < 0)
    ) or np.any(arrays["candidate_oracle_tp50"] & (matched < 0)):
        raise ValueError("TP50 flags may only mark GT-matched rows")
    expected_cross = (
        arrays["runtime_eligible"]
        & ~arrays["identity_tp50"]
        & arrays["candidate_oracle_tp50"]
    )
    if not np.array_equal(arrays["cross_iou50"], expected_cross):
        raise ValueError("cross_iou50 does not match strict source events")
    selected_counts = arrays["selected_view_counts"]
    if (
        selected_counts.shape != (sample_count,)
        or selected_counts.dtype != np.int64
        or not np.array_equal(selected_counts, view_mask.sum(axis=1))
    ):
        raise ValueError(
            "selected_view_counts disagree with exact joint view masks"
        )

    return JointLocalTrainingData(
        points_local=np.ascontiguousarray(points),
        point_mask=np.ascontiguousarray(point_mask),
        view_features=np.ascontiguousarray(view_features),
        view_mask=np.ascontiguousarray(view_mask),
        local_boxes=np.ascontiguousarray(local_boxes),
        quality_features=np.ascontiguousarray(quality_features),
        target_residual=np.ascontiguousarray(target_residual),
        geometry_mask=np.ascontiguousarray(geometry_mask),
        scene_ids=np.ascontiguousarray(scene_ids),
        matched_gt_index=np.ascontiguousarray(matched),
        original_iou=np.ascontiguousarray(original_iou),
        refined_iou=np.ascontiguousarray(refined_iou),
        aligned_basis=np.ascontiguousarray(aligned_basis),
        original_aligned_center=np.ascontiguousarray(aligned_center),
        matched_gt_box=np.ascontiguousarray(matched_gt_box),
        iou_gain=np.ascontiguousarray(arrays["iou_gain"]),
        cross_iou50=np.ascontiguousarray(arrays["cross_iou50"]),
        ap50_weight=np.ascontiguousarray(arrays["ap50_weight"]),
        runtime_eligible=np.ascontiguousarray(
            arrays["runtime_eligible"]
        ),
        identity_tp50=np.ascontiguousarray(arrays["identity_tp50"]),
        candidate_oracle_tp50=np.ascontiguousarray(
            arrays["candidate_oracle_tp50"]
        ),
        max_center_fraction=max_center,
        max_log_dimension_residual=max_log_dimension,
        source_dataset_sha256=source_digest,
        forbidden_scene_count=forbidden_count,
        forbidden_scene_sha256=forbidden_digest,
        training_scene_sha256=training_digest,
        points_per_view=points_per_view,
    )


class _JointArrayDataset(Dataset):
    def __init__(
        self, data: JointLocalTrainingData, indices: np.ndarray
    ) -> None:
        indices = np.asarray(indices, dtype=np.int64)
        self.values = {
            "points_local": torch.from_numpy(data.points_local[indices]),
            "point_mask": torch.from_numpy(data.point_mask[indices]),
            "view_features": torch.from_numpy(
                data.view_features[indices]
            ),
            "view_mask": torch.from_numpy(data.view_mask[indices]),
            "local_boxes": torch.from_numpy(data.local_boxes[indices]),
            "quality_features": torch.from_numpy(
                data.quality_features[indices]
            ),
            "target_residual": torch.from_numpy(
                data.target_residual[indices]
            ),
            "geometry_mask": torch.from_numpy(
                data.geometry_mask[indices]
            ),
            "matched_gt_index": torch.from_numpy(
                data.matched_gt_index[indices]
            ),
            "original_iou": torch.from_numpy(data.original_iou[indices]),
            "refined_iou": torch.from_numpy(data.refined_iou[indices]),
            "aligned_basis": torch.from_numpy(
                data.aligned_basis[indices]
            ),
            "original_aligned_center": torch.from_numpy(
                data.original_aligned_center[indices]
            ),
            "matched_gt_box": torch.from_numpy(
                data.matched_gt_box[indices]
            ),
            "iou_gain": torch.from_numpy(data.iou_gain[indices]),
            "cross_iou50": torch.from_numpy(data.cross_iou50[indices]),
            "ap50_weight": torch.from_numpy(data.ap50_weight[indices]),
            "runtime_eligible": torch.from_numpy(
                data.runtime_eligible[indices]
            ),
            "identity_tp50": torch.from_numpy(
                data.identity_tp50[indices]
            ),
        }

    def __len__(self) -> int:
        return int(len(self.values["points_local"]))

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        return {name: value[index] for name, value in self.values.items()}


def _weighted_mean(
    values: torch.Tensor,
    weights: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if mask is not None:
        values = values[mask]
        weights = weights[mask]
    if values.numel() == 0:
        return values.sum() * 0.0
    normalized = weights / weights.mean().clamp_min(
        torch.finfo(weights.dtype).eps
    )
    return (values * normalized).mean()


def joint_local_loss(
    output: Mapping[str, torch.Tensor],
    *,
    target_residual: torch.Tensor,
    geometry_mask: torch.Tensor,
    local_boxes: torch.Tensor,
    matched_gt_index: torch.Tensor,
    original_iou: torch.Tensor,
    refined_iou: torch.Tensor,
    aligned_basis: torch.Tensor,
    original_aligned_center: torch.Tensor,
    matched_gt_box: torch.Tensor,
    iou_gain_target: torch.Tensor,
    cross_iou50: torch.Tensor,
    ap50_weight: torch.Tensor,
    runtime_eligible: torch.Tensor,
    identity_tp50: torch.Tensor,
    center_weight: float = 1.0,
    dimension_weight: float = 1.0,
    identity_weight: float = 0.25,
    improvement_weight: float = 1.0,
    iou_gain_weight: float = 2.0,
    cross_iou50_weight: float = 4.0,
    preserve_iou50_weight: float = 2.0,
    dual_iou_weight: float = 1.0,
    ordinal_weight: float = 1.0,
    uncertainty_weight: float = 0.10,
) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Compute AP50-aware geometry plus dual realized-quality supervision."""

    batch = int(local_boxes.shape[0])
    expected_output_shapes = {
        "center_residual_fraction": (batch, 3),
        "log_dimension_residual": (batch, 3),
        "improvement_probability": (batch,),
        "quality_components": (batch, 2, 4),
        "quality_log_variance": (batch, 2),
    }
    for name, shape in expected_output_shapes.items():
        if name not in output or output[name].shape != shape:
            raise ValueError(f"output {name} must have shape {shape}")
    if geometry_mask.dtype != torch.bool:
        raise TypeError("geometry_mask must be Boolean")
    if ap50_weight.shape != (batch,) or not torch.all(
        ap50_weight >= 1.0
    ):
        raise ValueError("ap50_weight must be [B] and at least one")

    center_values = F.smooth_l1_loss(
        output["center_residual_fraction"],
        target_residual[:, :3],
        beta=0.05,
        reduction="none",
    ).mean(dim=1)
    dimension_values = F.smooth_l1_loss(
        output["log_dimension_residual"],
        target_residual[:, 3:],
        beta=0.05,
        reduction="none",
    ).mean(dim=1)
    center_loss = _weighted_mean(
        center_values, ap50_weight, geometry_mask
    )
    dimension_loss = _weighted_mean(
        dimension_values, ap50_weight, geometry_mask
    )
    identity_values = F.smooth_l1_loss(
        torch.cat(
            (
                output["center_residual_fraction"],
                output["log_dimension_residual"],
            ),
            dim=1,
        ),
        torch.zeros_like(target_residual),
        beta=0.05,
        reduction="none",
    ).mean(dim=1)
    identity_loss = _weighted_mean(
        identity_values, ap50_weight, ~geometry_mask
    )

    improve_target = geometry_mask.to(
        output["improvement_probability"].dtype
    )
    improvement_values = F.binary_cross_entropy(
        output["improvement_probability"].clamp(1e-6, 1.0 - 1e-6),
        improve_target,
        reduction="none",
    )
    improvement_loss = _weighted_mean(
        improvement_values, ap50_weight
    )

    realized_candidate_iou = differentiable_aligned_aabb_iou(
        output,
        local_boxes,
        aligned_basis,
        original_aligned_center,
        matched_gt_box,
    )
    matched = matched_gt_index >= 0
    realized_candidate_iou = torch.where(
        matched, realized_candidate_iou, torch.zeros_like(realized_candidate_iou)
    )
    realized_gain = realized_candidate_iou - original_iou
    eligible_matched = runtime_eligible & matched
    gain_shortfall = torch.relu(iou_gain_target - realized_gain)
    iou_gain_loss = _weighted_mean(
        gain_shortfall, ap50_weight, eligible_matched
    )
    cross_values = torch.relu(
        torch.as_tensor(
            AP50_LOSS_TARGET,
            dtype=realized_candidate_iou.dtype,
            device=realized_candidate_iou.device,
        )
        - realized_candidate_iou
    )
    cross_loss = _weighted_mean(
        cross_values, ap50_weight, cross_iou50
    )
    preserve_loss = _weighted_mean(
        cross_values, ap50_weight, identity_tp50 & eligible_matched
    )

    # Crucial contract: current realized candidate IoU, not oracle refined_iou.
    dual_targets = torch.stack(
        (original_iou, realized_candidate_iou.detach()), dim=1
    )
    predicted_iou = output["quality_components"][..., 0]
    dual_iou_values = F.smooth_l1_loss(
        predicted_iou, dual_targets, beta=0.10, reduction="none"
    ).mean(dim=1)
    dual_iou_loss = _weighted_mean(
        dual_iou_values, ap50_weight
    )
    thresholds = torch.as_tensor(
        IOU_AWARE_THRESHOLDS,
        dtype=dual_targets.dtype,
        device=dual_targets.device,
    )
    ordinal_targets = (
        dual_targets[:, :, None] >= thresholds[None, None, :]
    ).to(dual_targets.dtype)
    ordinal_probabilities = output["quality_components"][
        ..., 1:
    ].clamp(1e-6, 1.0 - 1e-6)
    ordinal_values = F.binary_cross_entropy(
        ordinal_probabilities, ordinal_targets, reduction="none"
    ).mean(dim=(1, 2))
    ordinal_loss = _weighted_mean(ordinal_values, ap50_weight)

    log_variance = output["quality_log_variance"]
    squared_error = (predicted_iou - dual_targets).square()
    uncertainty_values = 0.5 * (
        squared_error * torch.exp(-log_variance) + log_variance
    )
    uncertainty_loss = _weighted_mean(
        uncertainty_values.mean(dim=1), ap50_weight
    )

    weights = {
        "center": float(center_weight),
        "dimension": float(dimension_weight),
        "identity": float(identity_weight),
        "improvement": float(improvement_weight),
        "iou_gain": float(iou_gain_weight),
        "cross_iou50": float(cross_iou50_weight),
        "preserve_iou50": float(preserve_iou50_weight),
        "dual_iou": float(dual_iou_weight),
        "ordinal": float(ordinal_weight),
        "uncertainty": float(uncertainty_weight),
    }
    losses = {
        "center": center_loss,
        "dimension": dimension_loss,
        "identity": identity_loss,
        "improvement": improvement_loss,
        "iou_gain": iou_gain_loss,
        "cross_iou50": cross_loss,
        "preserve_iou50": preserve_loss,
        "dual_iou": dual_iou_loss,
        "ordinal": ordinal_loss,
        "uncertainty": uncertainty_loss,
    }
    total = sum(weights[name] * losses[name] for name in losses)
    if not torch.isfinite(total):
        raise RuntimeError("joint local-head loss became non-finite")

    candidate_tp50 = realized_candidate_iou >= AP50_EVALUATOR_THRESHOLD
    cross_success = cross_iou50 & candidate_tp50
    drop50 = identity_tp50 & eligible_matched & ~candidate_tp50
    metrics: Dict[str, torch.Tensor] = {
        "loss": total.detach(),
        **{
            f"{name}_loss": value.detach()
            for name, value in losses.items()
        },
        "realized_candidate_iou": realized_candidate_iou.mean().detach(),
        "original_iou_mae": (
            predicted_iou[:, 0] - original_iou
        ).abs().mean().detach(),
        "candidate_iou_mae": (
            predicted_iou[:, 1] - realized_candidate_iou.detach()
        ).abs().mean().detach(),
        "mean_uncertainty": output["quality_uncertainty"].mean().detach(),
        "cross50_success_count": cross_success.sum().detach(),
        "drop50_count": drop50.sum().detach(),
        "eligible_matched_count": eligible_matched.sum().detach(),
        "geometry_positive_count": geometry_mask.sum().detach(),
        "candidate_iou_sum": realized_candidate_iou.detach().sum(),
    }
    # ``refined_iou`` is intentionally diagnostic only.  Keeping this metric
    # makes the oracle/realized gap visible without using oracle labels for
    # candidate quality.
    metrics["oracle_realized_iou_gap"] = (
        refined_iou - realized_candidate_iou.detach()
    ).abs().mean()
    return total, metrics


def _set_determinism(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)


def _run_epoch(
    model: MultiViewJointLocalHead,
    loader: DataLoader,
    *,
    optimizer: Optional[torch.optim.Optimizer],
    loss_weights: Mapping[str, float],
) -> Dict[str, float]:
    training = optimizer is not None
    model.train(training)
    totals: Dict[str, float] = {}
    total_samples = 0
    for batch in loader:
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            output = model(
                batch["points_local"],
                batch["point_mask"],
                batch["view_features"],
                batch["view_mask"],
                batch["local_boxes"],
                batch["quality_features"],
            )
            loss, metrics = joint_local_loss(
                output,
                target_residual=batch["target_residual"],
                geometry_mask=batch["geometry_mask"],
                local_boxes=batch["local_boxes"],
                matched_gt_index=batch["matched_gt_index"],
                original_iou=batch["original_iou"],
                refined_iou=batch["refined_iou"],
                aligned_basis=batch["aligned_basis"],
                original_aligned_center=(
                    batch["original_aligned_center"]
                ),
                matched_gt_box=batch["matched_gt_box"],
                iou_gain_target=batch["iou_gain"],
                cross_iou50=batch["cross_iou50"],
                ap50_weight=batch["ap50_weight"],
                runtime_eligible=batch["runtime_eligible"],
                identity_tp50=batch["identity_tp50"],
                **loss_weights,
            )
            if optimizer is not None:
                loss.backward()
                optimizer.step()
        batch_size = int(batch["points_local"].shape[0])
        total_samples += batch_size
        for name, value in metrics.items():
            numeric = float(value)
            if name.endswith("_count") or name.endswith("_sum"):
                totals[name] = totals.get(name, 0.0) + numeric
            else:
                totals[name] = (
                    totals.get(name, 0.0) + numeric * batch_size
                )
    if total_samples == 0:
        raise ValueError("joint data loader produced no samples")
    result = {
        name: (
            value
            if name.endswith("_count") or name.endswith("_sum")
            else value / total_samples
        )
        for name, value in totals.items()
    }
    denominator = result.get("eligible_matched_count", 0.0)
    cross = result.get("cross50_success_count", 0.0)
    drops = result.get("drop50_count", 0.0)
    result["local_net_tp50_proxy"] = (
        (cross - drops) / denominator if denominator > 0.0 else 0.0
    )
    result["cross50_success_rate"] = (
        cross / denominator if denominator > 0.0 else 0.0
    )
    result["drop50_rate"] = (
        drops / denominator if denominator > 0.0 else 0.0
    )
    return result


def _nonnegative_finite(name: str, value: float) -> float:
    if (
        isinstance(value, (bool, np.bool_))
        or not np.isscalar(value)
        or not np.isfinite(value)
        or float(value) < 0.0
    ):
        raise ValueError(f"{name} must be non-negative and finite")
    return float(value)


def _positive_finite(name: str, value: float) -> float:
    result = _nonnegative_finite(name, value)
    if result == 0.0:
        raise ValueError(f"{name} must be positive")
    return result


def train_joint_local_head(
    dataset_path: str | os.PathLike[str],
    output_path: str | os.PathLike[str],
    *,
    config: Optional[JointLocalHeadConfig] = None,
    epochs: int = 40,
    batch_size: int = 16,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-5,
    validation_fraction: float = 0.2,
    seed: int = 1337,
    center_weight: float = 1.0,
    dimension_weight: float = 1.0,
    identity_weight: float = 0.25,
    improvement_weight: float = 1.0,
    iou_gain_weight: float = 2.0,
    cross_iou50_weight: float = 4.0,
    preserve_iou50_weight: float = 2.0,
    dual_iou_weight: float = 1.0,
    ordinal_weight: float = 1.0,
    uncertainty_weight: float = 0.10,
) -> Dict[str, Any]:
    """Train on CPU and atomically write a strict joint checkpoint."""

    for name, value in (("epochs", epochs), ("batch_size", batch_size)):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise TypeError(f"{name} must be an integer")
        if int(value) < 1:
            raise ValueError(f"{name} must be positive")
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
        raise TypeError("seed must be an integer")
    learning_rate = _positive_finite("learning_rate", learning_rate)
    weight_decay = _nonnegative_finite("weight_decay", weight_decay)
    if (
        not np.isscalar(validation_fraction)
        or not np.isfinite(validation_fraction)
        or not 0.0 < float(validation_fraction) < 1.0
    ):
        raise ValueError("validation_fraction must lie strictly in (0,1)")
    loss_weights = {
        name: _nonnegative_finite(name, value)
        for name, value in (
            ("center_weight", center_weight),
            ("dimension_weight", dimension_weight),
            ("identity_weight", identity_weight),
            ("improvement_weight", improvement_weight),
            ("iou_gain_weight", iou_gain_weight),
            ("cross_iou50_weight", cross_iou50_weight),
            ("preserve_iou50_weight", preserve_iou50_weight),
            ("dual_iou_weight", dual_iou_weight),
            ("ordinal_weight", ordinal_weight),
            ("uncertainty_weight", uncertainty_weight),
        )
    }
    if sum(loss_weights.values()) <= 0.0:
        raise ValueError("at least one joint loss weight must be positive")
    dataset = Path(dataset_path)
    dataset_sha256 = _sha256_file(dataset)
    data = load_joint_local_dataset(dataset)
    model_config = (config or JointLocalHeadConfig()).validated()
    if (
        not np.isclose(
            model_config.max_center_fraction,
            data.max_center_fraction,
            atol=1e-7,
            rtol=0.0,
        )
        or not np.isclose(
            model_config.max_log_dimension_residual,
            data.max_log_dimension_residual,
            atol=1e-7,
            rtol=0.0,
        )
    ):
        raise ValueError("joint model and dataset residual bounds differ")

    training_indices, validation_indices = deterministic_scene_split(
        data.scene_ids, float(validation_fraction), int(seed)
    )
    training_scenes = sorted(
        set(data.scene_ids[training_indices].tolist())
    )
    validation_scenes = sorted(
        set(data.scene_ids[validation_indices].tolist())
    )
    if set(training_scenes) & set(validation_scenes):
        raise RuntimeError("joint train/validation scenes leaked")
    # Fail before model construction if scene-held-out training cannot balance.
    balanced_epoch_indices(
        training_indices, data.geometry_mask, int(seed)
    )
    _set_determinism(int(seed))

    model = MultiViewJointLocalHead(model_config).cpu()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    validation_loader = DataLoader(
        _JointArrayDataset(data, validation_indices),
        batch_size=min(int(batch_size), len(validation_indices)),
        shuffle=False,
        num_workers=0,
    )
    best_state: Optional[Dict[str, torch.Tensor]] = None
    best_epoch: Optional[int] = None
    best_validation_loss = float("inf")
    best_validation_proxy = -float("inf")
    best_train_metrics: Optional[Dict[str, float]] = None
    best_validation_metrics: Optional[Dict[str, float]] = None
    for epoch in range(int(epochs)):
        epoch_indices = balanced_epoch_indices(
            training_indices,
            data.geometry_mask,
            int(seed) + epoch,
        )
        training_loader = DataLoader(
            _JointArrayDataset(data, epoch_indices),
            batch_size=min(int(batch_size), len(epoch_indices)),
            shuffle=False,
            num_workers=0,
        )
        train_metrics = _run_epoch(
            model,
            training_loader,
            optimizer=optimizer,
            loss_weights=loss_weights,
        )
        validation_metrics = _run_epoch(
            model,
            validation_loader,
            optimizer=None,
            loss_weights=loss_weights,
        )
        validation_loss = float(validation_metrics["loss"])
        validation_proxy = float(
            validation_metrics["local_net_tp50_proxy"]
        )
        better = (
            validation_loss < best_validation_loss - 1e-12
            or (
                math.isclose(
                    validation_loss,
                    best_validation_loss,
                    abs_tol=1e-12,
                )
                and validation_proxy > best_validation_proxy
            )
        )
        if better:
            best_validation_loss = validation_loss
            best_validation_proxy = validation_proxy
            best_epoch = int(epoch)
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
            best_train_metrics = dict(train_metrics)
            best_validation_metrics = dict(validation_metrics)
    if (
        best_state is None
        or best_epoch is None
        or best_train_metrics is None
        or best_validation_metrics is None
    ):
        raise RuntimeError("joint training did not produce a checkpoint")
    model.load_state_dict(best_state, strict=True)
    if dataset_sha256 != _sha256_file(dataset):
        raise RuntimeError("joint dataset changed during training")

    metadata: Dict[str, Any] = {
        "training_dataset_schema": JOINT_LOCAL_DATASET_SCHEMA,
        "training_dataset_format_version": (
            JOINT_LOCAL_DATASET_FORMAT_VERSION
        ),
        "training_dataset_sha256": dataset_sha256,
        "source_dataset_sha256": data.source_dataset_sha256,
        "objective": "joint_ap50_realized_candidate_quality",
        "top_k_views": EXPECTED_TOP_K_VIEWS,
        "points_per_view": data.points_per_view,
        "appearance_consistency_default": 0.5,
        "samples": data.sample_count,
        "training_samples": int(len(training_indices)),
        "validation_samples": int(len(validation_indices)),
        "training_scenes": training_scenes,
        "validation_scenes": validation_scenes,
        "training_scene_sha256": _scene_sha256(training_scenes),
        "validation_scene_sha256": _scene_sha256(validation_scenes),
        "forbidden_scene_count": data.forbidden_scene_count,
        "forbidden_scene_sha256": data.forbidden_scene_sha256,
        "scene_leakage": False,
        "seed": int(seed),
        "epochs": int(epochs),
        "best_epoch": best_epoch,
        "best_validation_loss": best_validation_loss,
        "best_validation_local_net_tp50_proxy": best_validation_proxy,
        "loss_weights": dict(loss_weights),
        "iou_thresholds": [float(value) for value in IOU_AWARE_THRESHOLDS],
        "candidate_quality_target": "realized_candidate_iou_detached",
        "checkpoint_selection": (
            "minimum scene-held-out joint validation loss; local net-TP50 "
            "proxy is a deterministic tie-break diagnostic only"
        ),
        "device": "cpu",
        "train_metrics": best_train_metrics,
        "validation_metrics": best_validation_metrics,
    }
    checkpoint = make_joint_local_head_checkpoint(
        model, metadata=metadata
    )
    output = Path(output_path)
    if output.suffix.lower() not in {".pt", ".pth"}:
        raise ValueError("joint checkpoint must end in .pt or .pth")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    try:
        torch.save(checkpoint, temporary)
        os.replace(temporary, output)
    finally:
        if temporary.exists():
            temporary.unlink()
    # Exercise the same strict loader used by runtime before reporting success.
    verification_model = MultiViewJointLocalHead(model_config).cpu()
    loaded_metadata = load_joint_local_head_checkpoint(
        verification_model, output, map_location="cpu"
    )
    if loaded_metadata != metadata:
        raise RuntimeError("saved joint checkpoint metadata changed on reload")
    return {
        "output": str(output),
        "samples": data.sample_count,
        "train_samples": int(len(training_indices)),
        "validation_samples": int(len(validation_indices)),
        "train_scenes": training_scenes,
        "validation_scenes": validation_scenes,
        "scene_leakage": False,
        "epochs": int(epochs),
        "best_epoch": best_epoch,
        "seed": int(seed),
        "best_validation_loss": best_validation_loss,
        "best_validation_local_net_tp50_proxy": best_validation_proxy,
        "train": best_train_metrics,
        "validation": best_validation_metrics,
    }


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="joint training NPZ")
    parser.add_argument("--output", required=True, help="joint .pt checkpoint")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--point-hidden-dim", type=int, default=48)
    parser.add_argument("--point-embedding-dim", type=int, default=96)
    parser.add_argument("--view-embedding-dim", type=int, default=96)
    parser.add_argument("--head-hidden-dim", type=int, default=128)
    parser.add_argument("--max-center-fraction", type=float, default=0.15)
    parser.add_argument(
        "--max-log-dimension-residual",
        type=float,
        default=float(math.log(1.25)),
    )
    parser.add_argument("--center-weight", type=float, default=1.0)
    parser.add_argument("--dimension-weight", type=float, default=1.0)
    parser.add_argument("--identity-weight", type=float, default=0.25)
    parser.add_argument("--improvement-weight", type=float, default=1.0)
    parser.add_argument("--iou-gain-weight", type=float, default=2.0)
    parser.add_argument("--cross-iou50-weight", type=float, default=4.0)
    parser.add_argument("--preserve-iou50-weight", type=float, default=2.0)
    parser.add_argument("--dual-iou-weight", type=float, default=1.0)
    parser.add_argument("--ordinal-weight", type=float, default=1.0)
    parser.add_argument("--uncertainty-weight", type=float, default=0.10)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    arguments = build_argument_parser().parse_args(argv)
    config = JointLocalHeadConfig(
        point_hidden_dim=arguments.point_hidden_dim,
        point_embedding_dim=arguments.point_embedding_dim,
        view_embedding_dim=arguments.view_embedding_dim,
        head_hidden_dim=arguments.head_hidden_dim,
        max_center_fraction=arguments.max_center_fraction,
        max_log_dimension_residual=(
            arguments.max_log_dimension_residual
        ),
    ).validated()
    result = train_joint_local_head(
        arguments.input,
        arguments.output,
        config=config,
        epochs=arguments.epochs,
        batch_size=arguments.batch_size,
        learning_rate=arguments.learning_rate,
        weight_decay=arguments.weight_decay,
        validation_fraction=arguments.validation_fraction,
        seed=arguments.seed,
        center_weight=arguments.center_weight,
        dimension_weight=arguments.dimension_weight,
        identity_weight=arguments.identity_weight,
        improvement_weight=arguments.improvement_weight,
        iou_gain_weight=arguments.iou_gain_weight,
        cross_iou50_weight=arguments.cross_iou50_weight,
        preserve_iou50_weight=arguments.preserve_iou50_weight,
        dual_iou_weight=arguments.dual_iou_weight,
        ordinal_weight=arguments.ordinal_weight,
        uncertainty_weight=arguments.uncertainty_weight,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
