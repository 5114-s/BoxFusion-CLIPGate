#!/usr/bin/env python3
"""Enrich a strict B5-v2 archive with exact K=5 local-view tensors.

The strict B5-v2 builder remains the single owner of GT matching, reachable
geometry targets, runtime-gate eligibility and AP50 event labels.  This tool
does not recompute any of them.  It joins the already validated B5 rows back
to the *same* observer diagnostics by ``(scene_id, result_index)`` and adds
the per-view Mask-RGBD inputs consumed by :mod:`boxfusion.joint_local_head`.

All joins are exact and fail closed.  In particular:

* only schema-v2, AP50, strict-K5 source archives are accepted;
* the supplied forbidden validation list must match the source archive hash;
* every B5 row must map to exactly one runtime diagnostic row;
* aggregate B5 tensors are checked against that row before view injection;
* invalid/padded model inputs are canonical zero; and
* the output is pickle-free and atomically replaced.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np

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
)
from boxfusion.quality_score import (
    IOU_AWARE_THRESHOLDS,
    QUALITY_FEATURE_NAMES,
)
from tools.build_oriented_refiner_dataset import (
    AP50_DATASET_FORMAT_VERSION,
    DATASET_SCHEMA as B5_DATASET_SCHEMA,
    V2_METADATA_KEYS,
    V2_SAMPLE_KEYS,
    load_scene_diagnostics,
    read_scene_ids,
    resolve_diagnostic_path,
    strict_provenance_for_profile,
)
from tools.train_oriented_box_refiner import (
    load_oriented_refiner_dataset,
)


JOINT_LOCAL_DATASET_SCHEMA = "boxfusion.joint_local_dataset"
JOINT_LOCAL_DATASET_FORMAT_VERSION = 2
EXPECTED_TOP_K_VIEWS = 5
SCENE_PATTERN = re.compile(r"scene\d{4}_\d{2}")
RUNTIME_QUALITY_FEATURE_SOURCE = "joint_quality_features"
LEGACY_REFINER_QUALITY_NORMALIZATION = (
    "b5v2_observer_disabled_refiner_0_to_neutral_0p5_v1"
)
REFINER_QUALITY_INDEX = QUALITY_FEATURE_NAMES.index("refiner_quality")

JOINT_VIEW_SAMPLE_KEYS = frozenset(
    {
        "joint_points_local",
        "joint_point_mask",
        "joint_view_features",
        "joint_view_mask",
        "joint_local_boxes",
        "joint_quality_features",
        "joint_frame_center",
        "joint_frame_centers",
        "joint_frame_basis",
        "joint_input_valid",
    }
)
JOINT_SAMPLE_KEYS = V2_SAMPLE_KEYS | JOINT_VIEW_SAMPLE_KEYS
JOINT_METADATA_KEYS = (
    V2_METADATA_KEYS - {"schema", "format_version"}
) | frozenset(
    {
        "schema",
        "format_version",
        "source_dataset_schema",
        "source_dataset_format_version",
        "source_dataset_sha256",
        "input_schema",
        "view_feature_names",
        "quality_branch_names",
        "quality_component_names",
        "iou_thresholds",
        "joint_top_k_views",
        "joint_points_per_view",
        "joint_sample_count",
        "diagnostic_scene_count",
        "diagnostic_scene_sha256",
        "appearance_consistency_default",
        "quality_feature_source",
        "legacy_refiner_quality_normalization",
        "legacy_refiner_quality_normalized_rows",
        "source_training_scene_count",
        "source_training_scene_sha256",
    }
)

_TOP_K_DIAGNOSTIC_FIELDS = frozenset(
    {
        "scene_id",
        "result_indices",
        "track_ids",
        "quality_features",
        "selected_view_counts",
        "selected_view_frame_ids",
        "top_k_view_valid",
        "box_refiner_points_local",
        "box_refiner_point_mask",
        "box_refiner_local_boxes",
        "box_refiner_frame_valid",
        "box_refiner_frame_centers",
        "box_refiner_frame_basis",
        "joint_points_local",
        "joint_point_mask",
        "joint_view_features",
        "joint_view_mask",
        "joint_local_boxes",
        "joint_quality_features",
        "joint_frame_center",
        "joint_frame_centers",
        "joint_frame_basis",
        "joint_input_valid",
    }
)


@dataclass(frozen=True)
class JointDatasetBuildConfig:
    b5_dataset: Path
    diagnostics_root: Path
    forbidden_scene_list: Path
    output: Path

    def validated(self) -> "JointDatasetBuildConfig":
        b5_dataset = Path(self.b5_dataset)
        diagnostics_root = Path(self.diagnostics_root)
        forbidden_scene_list = Path(self.forbidden_scene_list)
        output = Path(self.output)
        if not b5_dataset.is_file():
            raise FileNotFoundError(b5_dataset)
        if b5_dataset.suffix.lower() != ".npz":
            raise ValueError("b5_dataset must end in .npz")
        if not diagnostics_root.is_dir():
            raise FileNotFoundError(
                f"diagnostics_root is not a directory: {diagnostics_root}"
            )
        if not forbidden_scene_list.is_file():
            raise FileNotFoundError(forbidden_scene_list)
        if output.suffix.lower() != ".npz":
            raise ValueError("joint dataset output must end in .npz")
        try:
            if b5_dataset.resolve() == output.resolve():
                raise ValueError("output must not overwrite the B5 source")
        except OSError:
            pass
        return self


@dataclass(frozen=True)
class JointDatasetBuildSummary:
    scenes: int
    samples: int
    valid_views: int
    points_per_view: int
    legacy_refiner_quality_normalized_rows: int
    source_dataset_sha256: str
    output: Path


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


def _scalar_string(value: Any, name: str) -> str:
    array = np.asarray(value)
    if array.dtype.hasobject or array.ndim != 0:
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


def _safe_archive_copy(path: Path) -> dict[str, np.ndarray]:
    """Load an exact strict-v2 source without permitting object arrays."""

    try:
        with np.load(path, allow_pickle=False) as archive:
            expected = V2_SAMPLE_KEYS | V2_METADATA_KEYS
            keys = set(archive.files)
            if keys != expected:
                raise ValueError(
                    "strict B5-v2 keys are invalid: "
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
                f"{path} contains forbidden pickled/object arrays"
            ) from error
        raise
    for name, value in arrays.items():
        if value.dtype.hasobject:
            raise ValueError(f"{name} must not use object dtype")
    return arrays


def validate_runtime_quality_feature_relation(
    source_quality: np.ndarray,
    joint_quality: np.ndarray,
) -> int:
    """Validate source-vs-runtime features and return normalized row count.

    A historical ``b5v2_memory_observer`` run emitted 0.0 for the disabled
    legacy refiner after finalization, while its exact joint input used the
    neutral 0.5 sentinel.  Only that transition is accepted.  This helper is
    also used by the training loader so a hand-crafted archive cannot bypass
    the builder's fail-closed migration contract.
    """

    source = np.asarray(source_quality)
    joint = np.asarray(joint_quality)
    expected_tail = (len(QUALITY_FEATURE_NAMES),)
    if (
        source.shape != joint.shape
        or source.ndim not in (1, 2)
        or source.shape[-1:] != expected_tail
        or source.dtype != np.float32
        or joint.dtype != np.float32
    ):
        raise TypeError(
            "source and joint quality features must be matching float32 "
            f"[..., {len(QUALITY_FEATURE_NAMES)}] arrays"
        )
    if source.ndim == 1:
        source = source[None, :]
        joint = joint[None, :]
    if (
        not np.isfinite(source).all()
        or not np.isfinite(joint).all()
        or (source < 0.0).any()
        or (source > 1.0).any()
        or (joint < 0.0).any()
        or (joint > 1.0).any()
    ):
        raise ValueError("source/joint quality features must lie in [0,1]")

    non_refiner = np.ones(len(QUALITY_FEATURE_NAMES), dtype=np.bool_)
    non_refiner[REFINER_QUALITY_INDEX] = False
    if not np.array_equal(source[:, non_refiner], joint[:, non_refiner]):
        raise ValueError(
            "source row disagrees with exact joint quality_features"
        )
    joint_refiner = joint[:, REFINER_QUALITY_INDEX]
    if not np.all(joint_refiner == np.float32(0.5)):
        raise ValueError(
            "exact joint refiner_quality must use neutral 0.5"
        )
    source_refiner = source[:, REFINER_QUALITY_INDEX]
    normalized = source_refiner == np.float32(0.0)
    unchanged = source_refiner == joint_refiner
    if not np.all(normalized | unchanged):
        raise ValueError(
            "legacy source refiner_quality must be 0.0 or exact joint 0.5"
        )
    return int(np.count_nonzero(normalized))


def _load_strict_source(
    path: Path,
    forbidden_scene_list: Path,
) -> tuple[dict[str, np.ndarray], str, str]:
    """Validate the source with the canonical B5 loader, then copy it."""

    validated = load_oriented_refiner_dataset(path)
    if validated.objective != "ap50":
        raise ValueError("joint training requires objective='ap50'")
    arrays = _safe_archive_copy(path)
    if _scalar_string(arrays["schema"], "schema") != B5_DATASET_SCHEMA:
        raise ValueError("source dataset schema is not strict B5-v2")
    if (
        _scalar_integer(arrays["format_version"], "format_version")
        != AP50_DATASET_FORMAT_VERSION
    ):
        raise ValueError("source dataset must use strict schema version 2")
    strict = np.asarray(arrays["strict_k5_diagnostics"])
    if strict.ndim != 0 or strict.dtype != np.bool_ or not bool(strict):
        raise ValueError("source dataset must use strict K=5 diagnostics")
    if (
        _scalar_integer(arrays["expected_top_k_views"], "expected_top_k_views")
        != EXPECTED_TOP_K_VIEWS
    ):
        raise ValueError("source dataset must use exactly K=5")
    appearance_index = QUALITY_FEATURE_NAMES.index(
        "appearance_consistency"
    )
    appearance = np.asarray(
        arrays["quality_features"][:, appearance_index],
        dtype=np.float32,
    )
    provenance_profile = _scalar_string(
        arrays["online_ablation_profile"], "online_ablation_profile"
    )
    strict_provenance_for_profile(provenance_profile)
    if (
        provenance_profile == "b5v2_memory_observer"
        and not np.allclose(appearance, 0.5, atol=1e-7, rtol=0.0)
    ):
        raise ValueError(
            "joint train/runtime distribution requires "
            "appearance_consistency fixed to 0.5"
        )
    if (
        not np.isfinite(appearance).all()
        or (appearance < 0.0).any()
        or (appearance > 1.0).any()
    ):
        raise ValueError("appearance_consistency must lie in [0,1]")

    forbidden = read_scene_ids(forbidden_scene_list)
    forbidden_digest = _scene_sha256(forbidden)
    stored_count = _scalar_integer(
        arrays["forbidden_scene_count"], "forbidden_scene_count"
    )
    stored_digest = _scalar_string(
        arrays["forbidden_scene_sha256"], "forbidden_scene_sha256"
    )
    if stored_count != len(forbidden) or stored_digest != forbidden_digest:
        raise ValueError(
            "forbidden validation list does not match the strict B5 source"
        )
    training_scenes = sorted(np.unique(validated.scene_ids).tolist())
    leaked = sorted(set(training_scenes) & set(forbidden))
    if leaked:
        raise ValueError(
            "training data overlaps forbidden validation scenes: "
            f"{leaked[:5]}"
        )
    if (
        _scalar_integer(
            arrays["training_scene_count"], "training_scene_count"
        )
        != len(training_scenes)
        or _scalar_string(
            arrays["training_scene_sha256"], "training_scene_sha256"
        )
        != _scene_sha256(training_scenes)
    ):
        raise ValueError("source training-scene provenance is inconsistent")
    return arrays, _sha256_file(path), provenance_profile


def _load_top_k_payload(
    path: Path,
    expected_scene_id: str,
    strict_provenance_profile: str,
) -> tuple[Any, dict[str, np.ndarray]]:
    """Validate base strict provenance and load additional exact view arrays."""

    diagnostics = load_scene_diagnostics(
        path,
        expected_scene_id=expected_scene_id,
        objective="ap50",
        expected_top_k_views=EXPECTED_TOP_K_VIEWS,
        strict_k5_diagnostics=True,
        strict_provenance_profile=strict_provenance_profile,
    )
    try:
        with np.load(path, allow_pickle=False) as archive:
            missing = _TOP_K_DIAGNOSTIC_FIELDS - set(archive.files)
            if missing:
                raise ValueError(
                    f"{path} is missing joint Top-K fields: {sorted(missing)}"
                )
            raw = {
                name: np.asarray(archive[name]).copy()
                for name in _TOP_K_DIAGNOSTIC_FIELDS
            }
    except ValueError as error:
        if "Object arrays cannot be loaded" in str(error):
            raise ValueError(
                f"{path} contains forbidden pickled/object arrays"
            ) from error
        raise
    for name, value in raw.items():
        if value.dtype.hasobject:
            raise ValueError(f"{name} must not use object dtype")
    return diagnostics, raw


def _validate_joint_payload(
    raw: Mapping[str, np.ndarray],
    *,
    sample_count: int,
    allow_dynamic_appearance: bool = False,
) -> int:
    """Validate the exact tensors emitted by the runtime joint helper."""

    points = raw["joint_points_local"]
    point_mask = raw["joint_point_mask"]
    view_features = raw["joint_view_features"]
    view_mask = raw["joint_view_mask"]
    input_valid = raw["joint_input_valid"]
    if (
        points.ndim != 4
        or points.shape[:2] != (sample_count, EXPECTED_TOP_K_VIEWS)
        or points.shape[3] != 3
        or points.shape[2] != 128
        or points.dtype != np.float32
    ):
        raise TypeError(
            "joint_points_local must be exact float32 [N,5,128,3]"
        )
    point_count = int(points.shape[2])
    if (
        point_mask.shape != points.shape[:-1]
        or point_mask.dtype != np.bool_
    ):
        raise TypeError(
            "joint_point_mask must be exact Boolean [N,5,128]"
        )
    if (
        view_features.shape
        != (sample_count, EXPECTED_TOP_K_VIEWS, JOINT_VIEW_FEATURE_DIM)
        or view_features.dtype != np.float32
    ):
        raise TypeError("joint_view_features must be exact float32 [N,5,9]")
    if (
        view_mask.shape != (sample_count, EXPECTED_TOP_K_VIEWS)
        or view_mask.dtype != np.bool_
    ):
        raise TypeError("joint_view_mask must be exact Boolean [N,5]")
    if input_valid.shape != (sample_count,) or input_valid.dtype != np.bool_:
        raise TypeError("joint_input_valid must be exact Boolean [N]")
    if not np.array_equal(view_mask, point_mask.any(axis=2)):
        raise ValueError(
            "joint_view_mask must equal joint_point_mask.any(axis=2)"
        )
    if not np.array_equal(input_valid, view_mask.any(axis=1)):
        raise ValueError(
            "joint_input_valid must equal joint_view_mask.any(axis=1)"
        )
    if not np.isfinite(points).all():
        raise ValueError("joint local view points must be finite")
    if not np.all(points[~point_mask] == 0.0):
        raise ValueError("masked joint point padding must be zero")
    if (
        not np.isfinite(view_features).all()
        or (view_features < 0.0).any()
        or (view_features > 1.0).any()
        or not np.all(view_features[~view_mask] == 0.0)
    ):
        raise ValueError("joint view features/padding are invalid")

    local_boxes = raw["joint_local_boxes"]
    quality_features = raw["joint_quality_features"]
    centers = raw["joint_frame_center"]
    centers_alias = raw["joint_frame_centers"]
    basis = raw["joint_frame_basis"]
    expected_values = (
        ("joint_local_boxes", local_boxes, (sample_count, 6), np.float32),
        (
            "joint_quality_features",
            quality_features,
            (sample_count, len(QUALITY_FEATURE_NAMES)),
            np.float32,
        ),
        ("joint_frame_center", centers, (sample_count, 3), np.float64),
        (
            "joint_frame_centers",
            centers_alias,
            (sample_count, 3),
            np.float64,
        ),
        (
            "joint_frame_basis",
            basis,
            (sample_count, 3, 3),
            np.float64,
        ),
    )
    for name, value, shape, dtype in expected_values:
        if value.shape != shape or value.dtype != dtype:
            raise TypeError(f"{name} must use {dtype} with shape {shape}")
    if not np.array_equal(centers, centers_alias, equal_nan=True):
        raise ValueError("joint frame-center aliases disagree")
    appearance_index = QUALITY_FEATURE_NAMES.index(
        "appearance_consistency"
    )
    for row in range(sample_count):
        valid = bool(input_valid[row])
        if valid:
            if (
                quality_features[row, REFINER_QUALITY_INDEX]
                != np.float32(0.5)
            ):
                raise ValueError(
                    "exact joint refiner_quality must use neutral 0.5"
                )
            if (
                not np.isfinite(local_boxes[row]).all()
                or not np.allclose(
                    local_boxes[row, :3], 0.0, atol=1e-7, rtol=0.0
                )
                or np.any(local_boxes[row, 3:6] <= 0.0)
                or not np.isfinite(quality_features[row]).all()
                or np.any(quality_features[row] < 0.0)
                or np.any(quality_features[row] > 1.0)
                or (
                    not allow_dynamic_appearance
                    and not np.isclose(
                        quality_features[row, appearance_index],
                        0.5,
                        atol=1e-7,
                        rtol=0.0,
                    )
                )
                or not np.isfinite(centers[row]).all()
                or not np.isfinite(basis[row]).all()
                or not np.allclose(
                    basis[row].T @ basis[row],
                    np.eye(3),
                    atol=2e-3,
                    rtol=0.0,
                )
                or float(np.linalg.det(basis[row])) <= 0.0
            ):
                raise ValueError("valid exact joint input row is invalid")
        else:
            if (
                point_mask[row].any()
                or view_mask[row].any()
                or not np.all(points[row] == 0.0)
                or not np.all(view_features[row] == 0.0)
                or not np.all(quality_features[row] == 0.0)
                or not np.isnan(local_boxes[row]).all()
                or not np.isnan(centers[row]).all()
                or not np.isnan(basis[row]).all()
            ):
                raise ValueError(
                    "invalid exact joint row violates runtime sentinels"
                )
    return point_count


def _assert_source_row_matches_diagnostic(
    source: Mapping[str, np.ndarray],
    source_row: int,
    raw: Mapping[str, np.ndarray],
    diagnostic_row: int,
) -> bool:
    """Prove the join points to the row used by the strict B5 builder."""

    exact_pairs = (
        ("points_local", "box_refiner_points_local"),
        ("point_mask", "box_refiner_point_mask"),
        ("local_boxes", "box_refiner_local_boxes"),
        ("quality_features", "quality_features"),
        ("selected_view_counts", "selected_view_counts"),
        ("track_ids", "track_ids"),
    )
    for source_name, diagnostic_name in exact_pairs:
        if not np.array_equal(
            source[source_name][source_row],
            raw[diagnostic_name][diagnostic_row],
        ):
            raise ValueError(
                f"source row disagrees with diagnostics: {source_name}"
            )
    expected_basis = raw["box_refiner_frame_basis"][
        diagnostic_row
    ].astype(np.float32)
    if not np.array_equal(source["basis_world"][source_row], expected_basis):
        raise ValueError("source row basis_world disagrees with diagnostics")
    if not bool(raw["box_refiner_frame_valid"][diagnostic_row]):
        raise ValueError("strict B5 row maps to an invalid local frame")
    if not bool(raw["joint_input_valid"][diagnostic_row]):
        return False
    if not np.array_equal(
        source["local_boxes"][source_row],
        raw["joint_local_boxes"][diagnostic_row],
    ):
        raise ValueError("source row disagrees with exact joint local_boxes")

    # ``b5v2_memory_observer`` historically serialized the disabled legacy
    # refiner's post-output sentinel as 0.0, while the exact joint runtime
    # input correctly used the neutral value 0.5.  Permit only that one known
    # legacy transition.  Every other feature remains bit-exact, so this does
    # not weaken protection against stale or misjoined diagnostics.
    source_quality = source["quality_features"][source_row]
    joint_quality = raw["joint_quality_features"][diagnostic_row]
    normalized_legacy_refiner_quality = bool(
        validate_runtime_quality_feature_relation(
            source_quality, joint_quality
        )
    )
    if not np.array_equal(
        source["basis_world"][source_row],
        raw["joint_frame_basis"][diagnostic_row].astype(np.float32),
    ):
        raise ValueError("source basis disagrees with exact joint frame")
    if not np.array_equal(
        raw["box_refiner_frame_centers"][diagnostic_row],
        raw["joint_frame_center"][diagnostic_row],
    ) or not np.array_equal(
        raw["box_refiner_frame_basis"][diagnostic_row],
        raw["joint_frame_basis"][diagnostic_row],
    ):
        raise ValueError("B5 and joint exact local frames disagree")
    return normalized_legacy_refiner_quality


def build_joint_local_dataset(
    config: JointDatasetBuildConfig,
) -> JointDatasetBuildSummary:
    """Build and atomically write the exact-view joint training archive."""

    config = config.validated()
    source, source_sha256, provenance_profile = _load_strict_source(
        Path(config.b5_dataset), Path(config.forbidden_scene_list)
    )
    scene_ids = np.asarray(source["scene_ids"]).astype(np.str_)
    scenes = sorted(np.unique(scene_ids).tolist())
    if any(SCENE_PATTERN.fullmatch(scene) is None for scene in scenes):
        raise ValueError("source contains an invalid ScanNet scene id")
    sample_count = int(len(scene_ids))
    joined = np.zeros(sample_count, dtype=np.bool_)
    legacy_refiner_quality_normalized_rows = 0
    point_count: Optional[int] = None
    joint_outputs: Optional[dict[str, np.ndarray]] = None

    for scene_id in scenes:
        path = resolve_diagnostic_path(
            Path(config.diagnostics_root), scene_id
        )
        diagnostics, raw = _load_top_k_payload(
            path, scene_id, provenance_profile
        )
        diagnostic_count = int(len(diagnostics.result_indices))
        current_point_count = _validate_joint_payload(
            raw,
            sample_count=diagnostic_count,
            allow_dynamic_appearance=(
                provenance_profile == "sgcdet_sparse_observer"
            ),
        )
        if point_count is None:
            point_count = current_point_count
            joint_outputs = {
                "joint_points_local": np.zeros(
                    (
                        sample_count,
                        EXPECTED_TOP_K_VIEWS,
                        point_count,
                        3,
                    ),
                    dtype=np.float32,
                ),
                "joint_point_mask": np.zeros(
                    (
                        sample_count,
                        EXPECTED_TOP_K_VIEWS,
                        point_count,
                    ),
                    dtype=np.bool_,
                ),
                "joint_view_features": np.zeros(
                    (
                        sample_count,
                        EXPECTED_TOP_K_VIEWS,
                        JOINT_VIEW_FEATURE_DIM,
                    ),
                    dtype=np.float32,
                ),
                "joint_view_mask": np.zeros(
                    (sample_count, EXPECTED_TOP_K_VIEWS),
                    dtype=np.bool_,
                ),
                "joint_local_boxes": np.full(
                    (sample_count, 6), np.nan, dtype=np.float32
                ),
                "joint_quality_features": np.zeros(
                    (sample_count, len(QUALITY_FEATURE_NAMES)),
                    dtype=np.float32,
                ),
                "joint_frame_center": np.full(
                    (sample_count, 3), np.nan, dtype=np.float64
                ),
                "joint_frame_centers": np.full(
                    (sample_count, 3), np.nan, dtype=np.float64
                ),
                "joint_frame_basis": np.full(
                    (sample_count, 3, 3), np.nan, dtype=np.float64
                ),
                "joint_input_valid": np.zeros(
                    sample_count, dtype=np.bool_
                ),
            }
        elif current_point_count != point_count:
            raise ValueError("all diagnostics must use one points/view count")

        diagnostic_lookup = {
            int(result_index): row
            for row, result_index in enumerate(diagnostics.result_indices)
        }
        if len(diagnostic_lookup) != diagnostic_count:
            raise ValueError(f"{scene_id}: duplicate diagnostic result index")
        source_rows = np.flatnonzero(scene_ids == scene_id)
        for source_row in source_rows:
            result_index = int(source["result_indices"][source_row])
            if result_index not in diagnostic_lookup:
                raise ValueError(
                    f"{scene_id}: result_index {result_index} is missing "
                    "from diagnostics"
                )
            diagnostic_row = diagnostic_lookup[result_index]
            normalized = _assert_source_row_matches_diagnostic(
                source, int(source_row), raw, diagnostic_row
            )
            legacy_refiner_quality_normalized_rows += int(normalized)
            assert joint_outputs is not None
            for name in JOINT_VIEW_SAMPLE_KEYS:
                joint_outputs[name][source_row] = raw[name][diagnostic_row]
            joined[source_row] = True

    if not joined.all():
        missing = np.flatnonzero(~joined)
        raise RuntimeError(f"joint diagnostics join missed rows: {missing[:10]}")
    assert point_count is not None
    assert joint_outputs is not None
    keep = joint_outputs["joint_input_valid"].copy()
    if int(np.count_nonzero(keep)) < 2:
        raise ValueError(
            "joint dataset requires at least two joint_input_valid rows"
        )
    retained_scene_ids = scene_ids[keep]
    retained_scenes = sorted(np.unique(retained_scene_ids).tolist())
    if len(retained_scenes) < 2:
        raise ValueError(
            "joint_input_valid filtering must retain at least two scenes"
        )
    retained_sample_count = int(np.count_nonzero(keep))
    retained_geometry = np.asarray(source["geometry_mask"])[keep]
    if not retained_geometry.any() or retained_geometry.all():
        raise ValueError(
            "joint_input_valid rows must retain geometry positives and "
            "rejections"
        )
    for name in joint_outputs:
        joint_outputs[name] = np.ascontiguousarray(
            joint_outputs[name][keep]
        )
    if not joint_outputs["joint_input_valid"].all():
        raise RuntimeError("invalid joint rows survived strict filtering")
    if not np.array_equal(
        joint_outputs["joint_view_mask"],
        joint_outputs["joint_point_mask"].any(axis=2),
    ):
        raise RuntimeError("joint view masks became misaligned")

    output_arrays: dict[str, np.ndarray] = {}
    for name, value in source.items():
        if name in V2_SAMPLE_KEYS:
            output_arrays[name] = np.ascontiguousarray(
                value[keep].copy()
            )
        elif name in (V2_METADATA_KEYS - {"schema", "format_version"}):
            # ``np.ascontiguousarray`` promotes 0-D metadata to shape [1].
            # Preserve strict scalar shape while still copying it.
            output_arrays[name] = value.copy()
    source_training_scene_count = _scalar_integer(
        source["training_scene_count"], "training_scene_count"
    )
    source_training_scene_sha256 = _scalar_string(
        source["training_scene_sha256"], "training_scene_sha256"
    )
    output_arrays["training_scene_count"] = np.asarray(
        len(retained_scenes), dtype=np.int64
    )
    output_arrays["training_scene_sha256"] = np.asarray(
        _scene_sha256(retained_scenes)
    )
    output_arrays.update(joint_outputs)
    output_arrays.update(
        {
            "schema": np.asarray(JOINT_LOCAL_DATASET_SCHEMA),
            "format_version": np.asarray(
                JOINT_LOCAL_DATASET_FORMAT_VERSION, dtype=np.int64
            ),
            "source_dataset_schema": np.asarray(B5_DATASET_SCHEMA),
            "source_dataset_format_version": np.asarray(
                AP50_DATASET_FORMAT_VERSION, dtype=np.int64
            ),
            "source_dataset_sha256": np.asarray(source_sha256),
            "input_schema": np.asarray(JOINT_LOCAL_HEAD_INPUT_SCHEMA),
            "view_feature_names": np.asarray(
                JOINT_VIEW_FEATURE_NAMES, dtype=np.str_
            ),
            "quality_branch_names": np.asarray(
                JOINT_QUALITY_BRANCH_NAMES, dtype=np.str_
            ),
            "quality_component_names": np.asarray(
                JOINT_QUALITY_COMPONENT_NAMES, dtype=np.str_
            ),
            "iou_thresholds": np.asarray(
                IOU_AWARE_THRESHOLDS, dtype=np.float32
            ),
            "joint_top_k_views": np.asarray(
                EXPECTED_TOP_K_VIEWS, dtype=np.int64
            ),
            "joint_points_per_view": np.asarray(
                point_count, dtype=np.int64
            ),
            "joint_sample_count": np.asarray(
                retained_sample_count, dtype=np.int64
            ),
            "diagnostic_scene_count": np.asarray(
                len(retained_scenes), dtype=np.int64
            ),
            "diagnostic_scene_sha256": np.asarray(
                _scene_sha256(retained_scenes)
            ),
            "appearance_consistency_default": np.asarray(
                0.5, dtype=np.float32
            ),
            "quality_feature_source": np.asarray(
                RUNTIME_QUALITY_FEATURE_SOURCE
            ),
            "legacy_refiner_quality_normalization": np.asarray(
                LEGACY_REFINER_QUALITY_NORMALIZATION
            ),
            "legacy_refiner_quality_normalized_rows": np.asarray(
                legacy_refiner_quality_normalized_rows,
                dtype=np.int64,
            ),
            "source_training_scene_count": np.asarray(
                source_training_scene_count, dtype=np.int64
            ),
            "source_training_scene_sha256": np.asarray(
                source_training_scene_sha256
            ),
        }
    )
    if tuple(
        str(value)
        for value in np.asarray(output_arrays["quality_feature_names"])
    ) != QUALITY_FEATURE_NAMES:
        raise RuntimeError("source quality feature order changed")
    expected_keys = JOINT_SAMPLE_KEYS | JOINT_METADATA_KEYS
    if set(output_arrays) != expected_keys:
        raise RuntimeError(
            "internal joint dataset schema mismatch: "
            f"missing={sorted(expected_keys - set(output_arrays))}, "
            f"unexpected={sorted(set(output_arrays) - expected_keys)}"
        )
    for name, value in output_arrays.items():
        if not isinstance(value, np.ndarray) or value.dtype.hasobject:
            raise ValueError(
                f"joint output {name} must be a non-object NumPy array"
            )

    output = Path(config.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    try:
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, **output_arrays)
        os.replace(temporary, output)
    finally:
        if temporary.exists():
            temporary.unlink()
    if source_sha256 != _sha256_file(Path(config.b5_dataset)):
        raise RuntimeError("B5 source changed while building joint dataset")
    return JointDatasetBuildSummary(
        scenes=len(retained_scenes),
        samples=retained_sample_count,
        valid_views=int(
            np.count_nonzero(joint_outputs["joint_view_mask"])
        ),
        points_per_view=point_count,
        legacy_refiner_quality_normalized_rows=(
            legacy_refiner_quality_normalized_rows
        ),
        source_dataset_sha256=source_sha256,
        output=output,
    )


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--b5-dataset", required=True, type=Path, help="strict B5-v2 NPZ"
    )
    parser.add_argument(
        "--diagnostics-root",
        required=True,
        type=Path,
        help="matching b5v2_memory_observer diagnostics",
    )
    parser.add_argument(
        "--forbidden-scene-list",
        required=True,
        type=Path,
        help="exact validation list already hashed into the B5 source",
    )
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    arguments = build_argument_parser().parse_args(argv)
    summary = build_joint_local_dataset(
        JointDatasetBuildConfig(
            b5_dataset=arguments.b5_dataset,
            diagnostics_root=arguments.diagnostics_root,
            forbidden_scene_list=arguments.forbidden_scene_list,
            output=arguments.output,
        )
    )
    print(
        "Built exact joint-local dataset: "
        f"{summary.samples} samples / {summary.scenes} scenes / "
        f"{summary.valid_views} valid views / "
        f"{summary.legacy_refiner_quality_normalized_rows} legacy "
        f"refiner-quality rows normalized -> {summary.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
