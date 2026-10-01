#!/usr/bin/env python3
"""Read-only post-hoc geometry oracle for the sealed S3a MobileSAM shadow.

The oracle compares three geometries over one identical, frozen Boxer Top-4
membership: raw Boxer OBB, MobileSAM-depth q02/q98 AABB, and the prespecified
q00/q100 diagnostic AABB.  It computes geometry-only maximum matching and
never constructs or evaluates a prediction suffix.

Only the fixed dev3 scenes are accepted.  The tool cannot authorize H10,
full100, active birth, AP claims, threshold tuning, or candidate selection.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if os.fspath(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, os.fspath(_REPOSITORY_ROOT))

from tools.audit_scannet_boxer_per_view_topk_ceiling import (  # noqa: E402
    INPUT_SCHEMA as RAW_BOXER_SCHEMA,
    _aligned_candidate_minmax,
    _load_sealed_sidecar as _load_raw_boxer_sidecar,
    _select_per_frame_topk,
    _selection_sha256,
)
from tools.audit_scannet_boxer_unexplained_oracle import (  # noqa: E402
    aligned_iou_matrix,
    load_axis_alignment,
    load_baseline_boxes,
    load_gt_minmax,
    strict_maximum_matching,
)


SCHEMA = "boxfusion.scannet_s3a_mobilesam_masklift_oracle.v1"
SHADOW_SCHEMA = "boxfusion.boxer_mobilesam_masklift_shadow.v1"
TOPK_RECEIPT_SCHEMA = "boxfusion.scannet_boxer_per_view_topk_ceiling.v1"
PREREGISTRATION_SHA256 = (
    "ee742d4b0b9d3e26208ed8b59e587ed6de046ed850a22b80314fd8f939cad191"
)
RAW_BOXER_JSON_SHA256 = (
    "84eb4f2c62d1573d9e9f1ec4c3df5a6cac16ad10c8cece0989d37dd97b734e9e"
)
RAW_BOXER_NPZ_SHA256 = (
    "c1a921d70de447bf528711a71deb34cf93a9bf671d3514baafa42b7b1b8b4a6c"
)
TOPK_RECEIPT_SHA256 = (
    "d4ba67b37d362842333ac525abe32f6807c4fba90af83b699bbfc1494aa5ea1f"
)
TOP4_SELECTION_SHA256 = (
    "68049b78dba86441a6b691d1687b9fd2c90fc22f9f6e4c7c78548cc64384b306"
)
DEV3_SCENES = ("scene0568_00", "scene0606_01", "scene0377_02")
THRESHOLDS = (0.15, 0.25, 0.50)
CONTINUATION_MIN_MATCHES = 3
PACKED_MASK_BYTES = 480 * 640 // 8
SCENE_PATTERN = re.compile(r"^scene[0-9]{4}_[0-9]{2}$")

_REQUIRED_ARRAYS = {
    "scene_ids",
    "scene_index",
    "schedule_ordinal",
    "frame_id",
    "sealed_npz_row",
    "boxer_source_row",
    "boxer_csv_line_number",
    "source_instance_id",
    "owl_csv_source_row",
    "owl_csv_line_number",
    "source_score",
    "owl_box_xyxy_960",
    "prompt_box_xyxy_640x480",
    "raw_boxer_center_world",
    "raw_boxer_quaternion_wxyz",
    "raw_boxer_extent_xyz",
    "selected_hypothesis_index",
    "predicted_iou",
    "sam_mask_packed",
    "cleaned_depth_mask_packed",
    "sam_mask_pixel_count",
    "valid_depth_pixel_count",
    "raw_point_count",
    "unique_voxel_count",
    "retained_point_count",
    "median_depth_m",
    "point_offsets",
    "points_world",
    "accepted",
    "abstention_code",
    "reported_q02_q98_center_world",
    "reported_q02_q98_extent_xyz",
    "diagnostic_q00_q100_center_world",
    "diagnostic_q00_q100_extent_xyz",
    "diagnostic_box_valid",
    "encoder_ms",
    "decoder_ms",
    "frame_provider_ms",
    "lifting_ms",
}
_EXPECTED_ARRAYS = _REQUIRED_ARRAYS | {
    "manifest_schedule_ordinal",
    "sam_mask_sha256",
    "cleaned_depth_mask_sha256",
    "median_depth_valid",
    "points_sha256",
    "decode_ms",
}
_FORBIDDEN_ARRAY_TOKENS = ("label", "semantic", "class", "clip", "track", "terminal")
_AABB_SIGNS = np.asarray(
    [[x, y, z] for x in (-1.0, 1.0) for y in (-1.0, 1.0) for z in (-1.0, 1.0)],
    dtype=np.float64,
)


class S3aMaskliftOracleError(ValueError):
    """Raised when an input violates the frozen read-only oracle contract."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _regular_file(path: Path, label: str) -> Path:
    path = path.resolve()
    if path.is_symlink() or not path.is_file():
        raise S3aMaskliftOracleError(
            f"{label} must be a regular non-symlink file: {path}"
        )
    return path


def _read_json(path: Path, label: str) -> dict[str, Any]:
    path = _regular_file(path, label)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise S3aMaskliftOracleError(f"invalid {label}: {path}") from error
    if not isinstance(value, dict):
        raise S3aMaskliftOracleError(f"{label} must contain an object: {path}")
    return value


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise S3aMaskliftOracleError(f"{label} must be an object")
    return value


def _array_content_sha256(arrays: Mapping[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for name in sorted(arrays):
        array = np.ascontiguousarray(arrays[name])
        digest.update(name.encode("ascii"))
        digest.update(b"\0")
        digest.update(array.dtype.str.encode("ascii"))
        digest.update(b"\0")
        digest.update(
            json.dumps(list(array.shape), separators=(",", ":")).encode("ascii")
        )
        digest.update(b"\0")
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _hash_array(array: np.ndarray) -> str:
    value = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(value.dtype.str.encode("ascii"))
    digest.update(b"\0")
    digest.update(json.dumps(list(value.shape), separators=(",", ":")).encode("ascii"))
    digest.update(b"\0")
    digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def _threshold_key(threshold: float) -> str:
    return f"{threshold:.2f}"


def _expect_equal(mapping: Mapping[str, Any], key: str, expected: Any, label: str) -> None:
    if mapping.get(key) != expected:
        raise S3aMaskliftOracleError(
            f"{label} mismatch for {key}: {mapping.get(key)!r} != {expected!r}"
        )


def _expect_float(
    mapping: Mapping[str, Any], key: str, expected: float, label: str
) -> None:
    value = mapping.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isclose(
        float(value), expected, rel_tol=0.0, abs_tol=1e-12
    ):
        raise S3aMaskliftOracleError(
            f"{label} mismatch for {key}: {value!r} != {expected!r}"
        )


def _validate_embedded_file_ledgers(value: Any, label: str = "manifest") -> None:
    """Rehash every embedded conventional {path, sha256, bytes} ledger."""

    if isinstance(value, Mapping):
        if "path" in value and "sha256" in value:
            path = _regular_file(Path(str(value["path"])), f"{label} file")
            digest = value.get("sha256")
            if not isinstance(digest, str) or _sha256(path) != digest:
                raise S3aMaskliftOracleError(f"embedded file hash mismatch in {label}")
            if "bytes" in value and value.get("bytes") != path.stat().st_size:
                raise S3aMaskliftOracleError(f"embedded file size mismatch in {label}")
        for key, child in value.items():
            _validate_embedded_file_ledgers(child, f"{label}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _validate_embedded_file_ledgers(child, f"{label}[{index}]")


def _load_topk_receipt(
    path: Path,
    *,
    scenes: Sequence[str],
    expected_selection_sha256: str,
    enforce_production_hashes: bool,
) -> tuple[dict[str, Any], str]:
    path = _regular_file(path, "frozen Top-K ceiling receipt")
    digest = _sha256(path)
    if enforce_production_hashes and digest != TOPK_RECEIPT_SHA256:
        raise S3aMaskliftOracleError("Top-K ceiling receipt SHA-256 mismatch")
    receipt = _read_json(path, "frozen Top-K ceiling receipt")
    required = {
        "schema": TOPK_RECEIPT_SCHEMA,
        "posthoc_dev_diagnostic": True,
        "not_deployable": True,
        "before_mobilesam": True,
        "MobileSAM_used": False,
        "H10_not_authorized": True,
        "full100_not_authorized": True,
        "active_birth_authorized": False,
        "selection_used_gt": False,
        "selection_used_only_frozen_source_score": True,
        "selection_completed_before_gt_access": True,
    }
    for key, expected in required.items():
        _expect_equal(receipt, key, expected, "Top-K receipt")
    if receipt.get("scene_order") != list(scenes):
        raise S3aMaskliftOracleError("Top-K receipt scene order mismatch")
    budget = _mapping(_mapping(receipt.get("budgets"), "Top-K budgets").get("4"), "Top-4")
    _expect_equal(budget, "top_k_per_frame", 4, "Top-4 receipt")
    _expect_equal(
        budget,
        "selection_sha256",
        expected_selection_sha256,
        "Top-4 receipt",
    )
    return receipt, digest


def _frozen_topk_native_hashes(
    receipt: Mapping[str, Any], scenes: Sequence[str]
) -> dict[str, str]:
    """Return the native T05 hashes frozen by the pre-MobileSAM receipt.

    The S3a manifest is not authoritative for the native prefix: otherwise a
    producer can consistently bind both its before/after ledgers to the wrong
    result root.  The independently sealed Top-K ceiling receipt is the trust
    anchor, so every later manifest and on-disk baseline must agree with it.
    """

    before = _mapping(
        receipt.get("input_sha256_before"), "Top-K input-before ledger"
    )
    after = _mapping(receipt.get("input_sha256_after"), "Top-K input-after ledger")
    if before != after:
        raise S3aMaskliftOracleError("Top-K input hash identity mismatch")
    scene_ledger = _mapping(before.get("scenes"), "Top-K scene hash ledger")
    if set(scene_ledger) != set(scenes):
        raise S3aMaskliftOracleError("Top-K scene hash ledger keys mismatch")
    hashes: dict[str, str] = {}
    for scene in scenes:
        digest = _mapping(scene_ledger[scene], f"Top-K hashes for {scene}").get(
            "baseline"
        )
        if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
            raise S3aMaskliftOracleError(
                f"Top-K frozen native hash is invalid for {scene}"
            )
        hashes[scene] = digest
    return hashes


def _validate_profile(profile: Mapping[str, Any]) -> None:
    exact = {
        "image_height": 480,
        "image_width": 640,
        "owl_box_mapping": "x*2/3,y*1/2_then_clip",
        "prompt": "box_only",
        "shared_image_embedding_once_per_frame": True,
        "multimask_output": True,
        "mask_choice": "maximum_frozen_predicted_iou_lowest_index_tie",
        "mask_edge_margin_pixels": 1,
        "max_points_per_observation": 2048,
        "minimum_unique_clean_voxels": 16,
        "voxel_indexing": "signed_floor",
        "primary_aabb_quantiles": [0.02, 0.98],
        "diagnostic_aabb_quantiles": [0.0, 1.0],
        "coordinate_frame": "scannet_world",
    }
    for key, expected in exact.items():
        _expect_equal(profile, key, expected, "S3a geometry profile")
    for key, expected in {
        "mask_probability_threshold_equivalent": 0.5,
        "min_depth_m": 0.10,
        "max_depth_m": 6.00,
        "depth_discontinuity_m": 0.15,
        "voxel_size_m": 0.02,
        "primary_minimum_dimension_m": 0.02,
    }.items():
        _expect_float(profile, key, expected, "S3a geometry profile")


def _validate_shapes_and_values(
    arrays: Mapping[str, np.ndarray], count: int
) -> tuple[np.ndarray, Mapping[int, str]]:
    shapes = {
        "scene_index": (count,),
        "schedule_ordinal": (count,),
        "manifest_schedule_ordinal": (count,),
        "frame_id": (count,),
        "sealed_npz_row": (count,),
        "boxer_source_row": (count,),
        "boxer_csv_line_number": (count,),
        "source_instance_id": (count,),
        "owl_csv_source_row": (count,),
        "owl_csv_line_number": (count,),
        "source_score": (count,),
        "owl_box_xyxy_960": (count, 4),
        "prompt_box_xyxy_640x480": (count, 4),
        "raw_boxer_center_world": (count, 3),
        "raw_boxer_quaternion_wxyz": (count, 4),
        "raw_boxer_extent_xyz": (count, 3),
        "selected_hypothesis_index": (count,),
        "predicted_iou": (count,),
        "sam_mask_packed": (count, PACKED_MASK_BYTES),
        "cleaned_depth_mask_packed": (count, PACKED_MASK_BYTES),
        "sam_mask_pixel_count": (count,),
        "valid_depth_pixel_count": (count,),
        "raw_point_count": (count,),
        "unique_voxel_count": (count,),
        "retained_point_count": (count,),
        "median_depth_m": (count,),
        "median_depth_valid": (count,),
        "point_offsets": (count + 1,),
        "accepted": (count,),
        "abstention_code": (count,),
        "reported_q02_q98_center_world": (count, 3),
        "reported_q02_q98_extent_xyz": (count, 3),
        "diagnostic_q00_q100_center_world": (count, 3),
        "diagnostic_q00_q100_extent_xyz": (count, 3),
        "diagnostic_box_valid": (count,),
        "encoder_ms": (count,),
        "decoder_ms": (count,),
        "frame_provider_ms": (count,),
        "lifting_ms": (count,),
        "decode_ms": (count,),
        "sam_mask_sha256": (count,),
        "cleaned_depth_mask_sha256": (count,),
        "points_sha256": (count,),
    }
    for name, shape in shapes.items():
        if arrays[name].shape != shape:
            raise S3aMaskliftOracleError(
                f"S3a array {name} shape mismatch: {arrays[name].shape} != {shape}"
            )
    if arrays["scene_ids"].ndim != 1 or arrays["scene_ids"].dtype.kind != "U":
        raise S3aMaskliftOracleError("S3a scene_ids schema is invalid")
    if arrays["points_world"].ndim != 2 or arrays["points_world"].shape[1:] != (3,):
        raise S3aMaskliftOracleError("S3a points_world must have shape Mx3")
    integer_names = {
        "scene_index",
        "schedule_ordinal",
        "manifest_schedule_ordinal",
        "frame_id",
        "sealed_npz_row",
        "boxer_source_row",
        "boxer_csv_line_number",
        "source_instance_id",
        "owl_csv_source_row",
        "owl_csv_line_number",
        "selected_hypothesis_index",
        "sam_mask_pixel_count",
        "valid_depth_pixel_count",
        "raw_point_count",
        "unique_voxel_count",
        "retained_point_count",
        "point_offsets",
        "abstention_code",
    }
    for name in integer_names:
        if arrays[name].dtype.kind not in "iu":
            raise S3aMaskliftOracleError(f"S3a array {name} is not integer")
    if any(
        arrays[name].dtype != np.dtype(bool)
        for name in ("accepted", "diagnostic_box_valid", "median_depth_valid")
    ):
        raise S3aMaskliftOracleError("S3a validity arrays must be Boolean")
    if arrays["sam_mask_packed"].dtype != np.dtype(np.uint8) or arrays[
        "cleaned_depth_mask_packed"
    ].dtype != np.dtype(np.uint8):
        raise S3aMaskliftOracleError("S3a packed masks must be uint8")

    accepted = arrays["accepted"]
    if not np.array_equal(accepted, arrays["abstention_code"] == 0):
        raise S3aMaskliftOracleError("accepted rows differ from abstention code zero")
    expected_diagnostic_valid = arrays["retained_point_count"] > 0
    if not np.array_equal(expected_diagnostic_valid, arrays["diagnostic_box_valid"]):
        raise S3aMaskliftOracleError("q00 diagnostic validity differs from retained points")
    if not np.array_equal(expected_diagnostic_valid, arrays["median_depth_valid"]):
        raise S3aMaskliftOracleError("median-depth validity differs from retained points")
    for name in (
        "sam_mask_pixel_count",
        "valid_depth_pixel_count",
        "raw_point_count",
        "unique_voxel_count",
        "retained_point_count",
    ):
        if np.any(arrays[name] < 0):
            raise S3aMaskliftOracleError(f"negative S3a count in {name}")
    if np.any(arrays["retained_point_count"] > 2048):
        raise S3aMaskliftOracleError("S3a retained point cap exceeds 2048")
    if np.any(arrays["unique_voxel_count"][accepted] < 16):
        raise S3aMaskliftOracleError("accepted S3a row has fewer than 16 voxels")
    if np.any(
        arrays["retained_point_count"][accepted]
        > arrays["unique_voxel_count"][accepted]
    ):
        raise S3aMaskliftOracleError("retained points exceed unique voxels")
    expected_retained = np.minimum(arrays["unique_voxel_count"], 2048)
    if not np.array_equal(arrays["retained_point_count"], expected_retained):
        raise S3aMaskliftOracleError("S3a retained point count differs from fixed cap")
    sam_bit_counts = np.unpackbits(arrays["sam_mask_packed"], axis=1).sum(axis=1)
    cleaned_bit_counts = np.unpackbits(
        arrays["cleaned_depth_mask_packed"], axis=1
    ).sum(axis=1)
    if not np.array_equal(sam_bit_counts, arrays["sam_mask_pixel_count"]):
        raise S3aMaskliftOracleError("S3a packed SAM mask pixel count mismatch")
    if not np.array_equal(cleaned_bit_counts, arrays["raw_point_count"]):
        raise S3aMaskliftOracleError("S3a cleaned-depth mask/point count mismatch")
    if np.any(arrays["raw_point_count"] > arrays["valid_depth_pixel_count"]):
        raise S3aMaskliftOracleError("cleaned points exceed valid-depth pixels")
    if np.any(arrays["valid_depth_pixel_count"] > arrays["sam_mask_pixel_count"]):
        raise S3aMaskliftOracleError("valid-depth pixels exceed SAM mask pixels")
    if np.any(arrays["unique_voxel_count"] > arrays["raw_point_count"]):
        raise S3aMaskliftOracleError("unique voxels exceed cleaned raw points")
    offsets = arrays["point_offsets"].astype(np.int64, copy=False)
    if (
        len(offsets) != count + 1
        or offsets[0] != 0
        or np.any(np.diff(offsets) < 0)
        or offsets[-1] != len(arrays["points_world"])
        or not np.array_equal(np.diff(offsets), arrays["retained_point_count"])
    ):
        raise S3aMaskliftOracleError("S3a point offsets/counts are inconsistent")
    if not np.isfinite(arrays["points_world"]).all():
        raise S3aMaskliftOracleError("S3a retained points are non-finite")

    finite_always = (
        "source_score",
        "owl_box_xyxy_960",
        "prompt_box_xyxy_640x480",
        "raw_boxer_center_world",
        "raw_boxer_quaternion_wxyz",
        "raw_boxer_extent_xyz",
        "predicted_iou",
        "median_depth_m",
        "encoder_ms",
        "decoder_ms",
        "frame_provider_ms",
        "lifting_ms",
        "decode_ms",
    )
    for name in finite_always:
        if not np.isfinite(arrays[name]).all():
            raise S3aMaskliftOracleError(f"S3a array {name} is non-finite")
    if np.any((arrays["source_score"] < 0.0) | (arrays["source_score"] >= 1.0)):
        raise S3aMaskliftOracleError("S3a source score is outside [0,1)")
    # SAM's ``iou_predictions`` head is an unbounded quality regressor, not a
    # calibrated probability.  MobileSAM can therefore emit values slightly
    # above one (the sealed dev3 artifact does); finiteness is the applicable
    # integrity constraint.  The value is used only by the frozen argmax.
    hypotheses = arrays["selected_hypothesis_index"]
    if np.any((hypotheses < -1) | (hypotheses > 2)) or np.any(
        (hypotheses[accepted] < 0) | (hypotheses[accepted] > 2)
    ):
        raise S3aMaskliftOracleError("S3a mask hypothesis index is invalid")
    if np.any(arrays["raw_boxer_extent_xyz"] <= 0.0):
        raise S3aMaskliftOracleError("S3a raw Boxer extent is non-positive")
    raw_norm = np.linalg.norm(arrays["raw_boxer_quaternion_wxyz"], axis=1)
    if np.any(raw_norm <= 1e-6):
        raise S3aMaskliftOracleError("S3a raw Boxer quaternion is degenerate")
    for name in ("sam_mask_sha256", "cleaned_depth_mask_sha256", "points_sha256"):
        if arrays[name].dtype.kind != "U" or arrays[name].dtype.itemsize != 64 * 4:
            raise S3aMaskliftOracleError(f"S3a {name} is not a U64 digest array")
    for row in range(count):
        expected_hashes = {
            "sam_mask_sha256": _hash_array(arrays["sam_mask_packed"][row]),
            "cleaned_depth_mask_sha256": _hash_array(
                arrays["cleaned_depth_mask_packed"][row]
            ),
            "points_sha256": _hash_array(
                arrays["points_world"][offsets[row] : offsets[row + 1]]
            ),
        }
        for name, expected_hash in expected_hashes.items():
            if str(arrays[name][row]) != expected_hash:
                raise S3aMaskliftOracleError(
                    f"S3a per-row hash mismatch for {name} at row {row}"
                )

    primary_center = arrays["reported_q02_q98_center_world"]
    primary_extent = arrays["reported_q02_q98_extent_xyz"]
    diag_center = arrays["diagnostic_q00_q100_center_world"]
    diag_extent = arrays["diagnostic_q00_q100_extent_xyz"]
    diagnostic_valid = arrays["diagnostic_box_valid"]
    for name, values in (
        ("q02 center", primary_center[accepted]),
        ("q02 extent", primary_extent[accepted]),
        ("q00 center", diag_center[diagnostic_valid]),
        ("q00 extent", diag_extent[diagnostic_valid]),
    ):
        if not np.isfinite(values).all():
            raise S3aMaskliftOracleError(f"accepted S3a {name} is non-finite")
    if np.any(primary_extent[accepted] <= 0.0) or np.any(
        diag_extent[diagnostic_valid] < 0.0
    ):
        raise S3aMaskliftOracleError("valid S3a box has invalid extent")
    for row in np.flatnonzero(diagnostic_valid):
        points = arrays["points_world"][offsets[row] : offsets[row + 1]].astype(
            np.float64, copy=False
        )
        q02_lower = np.quantile(points, 0.02, axis=0)
        q02_upper = np.quantile(points, 0.98, axis=0)
        q00_lower = points.min(axis=0)
        q00_upper = points.max(axis=0)
        expected = {
            "q00 center": diag_center[row],
            "q00 extent": diag_extent[row],
        }
        actual = {
            "q00 center": (q00_lower + q00_upper) / 2.0,
            "q00 extent": q00_upper - q00_lower,
        }
        if accepted[row]:
            expected.update(
                {
                    "q02 center": primary_center[row],
                    "q02 extent": primary_extent[row],
                }
            )
            actual.update(
                {
                    "q02 center": (q02_lower + q02_upper) / 2.0,
                    "q02 extent": np.maximum(q02_upper - q02_lower, 0.02),
                }
            )
        for name in actual:
            if not np.allclose(expected[name], actual[name], rtol=0.0, atol=2e-5):
                raise S3aMaskliftOracleError(
                    f"S3a reported {name} differs from retained points at row {row}"
                )
    rejected = ~accepted
    for name, values in (
        ("q02 center", primary_center[rejected]),
        ("q02 extent", primary_extent[rejected]),
        ("q00 center", diag_center[~diagnostic_valid]),
        ("q00 extent", diag_extent[~diagnostic_valid]),
    ):
        if np.any(values != 0.0):
            raise S3aMaskliftOracleError(f"abstained S3a {name} is not zero-filled")
    return accepted, {}


def _load_s3a_sidecar(
    *,
    json_path: Path,
    npz_path: Path,
    raw_json_path: Path,
    raw_npz_path: Path,
    topk_receipt_path: Path,
    baseline_root: Path,
    preregistration_path: Path,
    raw_arrays: Mapping[str, np.ndarray],
    scenes: Sequence[str],
    selected_by_scene: Sequence[np.ndarray],
    selection_sha256: str,
    enforce_production_hashes: bool,
) -> tuple[dict[str, Any], dict[str, np.ndarray], dict[str, str], Mapping[int, str]]:
    json_path = _regular_file(json_path, "sealed S3a JSON")
    npz_path = _regular_file(npz_path, "sealed S3a NPZ")
    before = {"json": _sha256(json_path), "npz": _sha256(npz_path)}
    manifest = _read_json(json_path, "sealed S3a JSON")
    required = {
        "schema": SHADOW_SCHEMA,
        "mode": "shadow",
        "output_inert": True,
        "per_view_only": True,
        "birth": False,
        "active_authorized": False,
        "gt_access": False,
        "oracle_access": False,
        "labels_loaded": False,
        "labels_exported": False,
        "semantic_columns_decoded": False,
        "semantic_columns_consumed": False,
        "clip_access": False,
        "tracking": False,
        "suppression": False,
        "terminal_fusion": False,
        "native_overlap_rejection": False,
        "unexplained_depth_gate": False,
        "future_frame_access": False,
        "native_mutation_applied": False,
        "training": False,
        "optimizer": False,
        "threshold_tuning_performed": False,
        "posthoc_selection_performed": False,
        "H10_not_authorized": True,
        "full100_not_authorized": True,
        "not_deployable": True,
        "input_hash_identity": True,
    }
    for key, expected in required.items():
        _expect_equal(manifest, key, expected, "sealed S3a manifest")
    if manifest.get("scene_order") != list(scenes) or manifest.get("scene_count") != len(
        scenes
    ):
        raise S3aMaskliftOracleError("sealed S3a scene order/count mismatch")
    if enforce_production_hashes:
        _expect_equal(manifest, "engineering_smoke", False, "sealed S3a manifest")
        _expect_equal(manifest, "dev3_complete", True, "sealed S3a manifest")
        _expect_equal(
            manifest,
            "complete_exact_top4_membership_for_dev3",
            True,
            "sealed S3a manifest",
        )
    if manifest.get("npz_file") != npz_path.name:
        raise S3aMaskliftOracleError("sealed S3a NPZ filename mismatch")
    if manifest.get("npz_sha256") != before["npz"]:
        raise S3aMaskliftOracleError("sealed S3a NPZ SHA-256 mismatch")
    if manifest.get("preregistration_sha256") != PREREGISTRATION_SHA256:
        raise S3aMaskliftOracleError("sealed S3a preregistration hash mismatch")
    if _sha256(preregistration_path) != PREREGISTRATION_SHA256:
        raise S3aMaskliftOracleError("actual S3a preregistration hash mismatch")
    if manifest.get("input_sha256_before") != manifest.get("input_sha256_after"):
        raise S3aMaskliftOracleError("sealed S3a input before/after ledger differs")
    _validate_embedded_file_ledgers(
        manifest.get("input_sha256_before"), "sealed S3a input ledger"
    )

    selection = _mapping(manifest.get("selection"), "sealed S3a selection")
    _expect_equal(selection, "top_k_per_frame", 4, "sealed S3a selection")
    _expect_equal(
        selection,
        "selection_sha256",
        selection_sha256,
        "sealed S3a selection",
    )
    _expect_equal(
        selection,
        "selection_rule",
        "descending_source_score_then_ascending_source_row",
        "sealed S3a selection",
    )
    inputs = _mapping(manifest.get("input"), "sealed S3a input")
    if Path(str(inputs.get("baseline_root"))).resolve() != baseline_root:
        raise S3aMaskliftOracleError("sealed S3a baseline root mismatch")
    expected_input_hashes = {
        "sealed_boxer_json_sha256": _sha256(raw_json_path),
        "sealed_boxer_npz_sha256": _sha256(raw_npz_path),
        "topk_receipt_sha256": _sha256(topk_receipt_path),
    }
    for key, expected in expected_input_hashes.items():
        _expect_equal(inputs, key, expected, "sealed S3a input")
    _validate_profile(_mapping(manifest.get("geometry_profile"), "S3a profile"))

    try:
        with np.load(npz_path, allow_pickle=False) as source:
            actual_arrays = set(source.files)
            missing = _EXPECTED_ARRAYS - actual_arrays
            extra = actual_arrays - _EXPECTED_ARRAYS
            if missing or extra:
                raise S3aMaskliftOracleError(
                    "sealed S3a NPZ schema mismatch; missing="
                    + ",".join(sorted(missing))
                    + " extra="
                    + ",".join(sorted(extra))
                )
            forbidden = sorted(
                name
                for name in source.files
                if any(token in name.lower() for token in _FORBIDDEN_ARRAY_TOKENS)
            )
            if forbidden:
                raise S3aMaskliftOracleError(
                    "sealed S3a NPZ contains forbidden semantic/track arrays: "
                    + ", ".join(forbidden)
                )
            arrays = {name: np.array(source[name], copy=True) for name in source.files}
    except (OSError, ValueError) as error:
        if isinstance(error, S3aMaskliftOracleError):
            raise
        raise S3aMaskliftOracleError(f"invalid sealed S3a NPZ: {npz_path}") from error
    if _sha256(npz_path) != before["npz"]:
        raise S3aMaskliftOracleError("sealed S3a NPZ changed while loading")
    if manifest.get("candidate_content_sha256") != _array_content_sha256(arrays):
        raise S3aMaskliftOracleError("sealed S3a candidate content hash mismatch")

    expected_rows = np.concatenate(selected_by_scene).astype(np.int64, copy=False)
    count = len(expected_rows)
    if manifest.get("row_count") != count:
        raise S3aMaskliftOracleError("sealed S3a row count differs from Top-4 membership")
    accepted, _ = _validate_shapes_and_values(arrays, count)
    if tuple(str(value) for value in arrays["scene_ids"].tolist()) != tuple(scenes):
        raise S3aMaskliftOracleError("sealed S3a NPZ scene order mismatch")
    if not np.array_equal(
        arrays["sealed_npz_row"].astype(np.int64, copy=False), expected_rows
    ):
        raise S3aMaskliftOracleError("S3a membership/order differs from frozen Top-4")

    expected_scene_index = raw_arrays["per_view_scene_index"][expected_rows]
    exact_integer_pairs = {
        "scene_index": expected_scene_index,
        "frame_id": raw_arrays["per_view_frame_id"][expected_rows],
        "boxer_source_row": raw_arrays["per_view_source_row"][expected_rows],
        "source_instance_id": raw_arrays["per_view_source_instance_id"][expected_rows],
    }
    for name, expected in exact_integer_pairs.items():
        if not np.array_equal(arrays[name].astype(np.int64, copy=False), expected):
            raise S3aMaskliftOracleError(f"S3a provenance mismatch for {name}")
    exact_float_pairs = {
        "source_score": raw_arrays["per_view_source_score"][expected_rows],
        "raw_boxer_center_world": raw_arrays["per_view_center_world"][expected_rows],
        "raw_boxer_quaternion_wxyz": raw_arrays["per_view_quaternion_wxyz"][
            expected_rows
        ],
        "raw_boxer_extent_xyz": raw_arrays["per_view_extent_xyz"][expected_rows],
    }
    for name, expected in exact_float_pairs.items():
        if not np.array_equal(arrays[name], np.asarray(expected, dtype=arrays[name].dtype)):
            raise S3aMaskliftOracleError(f"S3a raw provenance mismatch for {name}")
    scaled_boxes = arrays["owl_box_xyxy_960"].astype(np.float64) * np.asarray(
        [2.0 / 3.0, 1.0 / 2.0, 2.0 / 3.0, 1.0 / 2.0], dtype=np.float64
    )
    scaled_boxes[:, (0, 2)] = np.clip(scaled_boxes[:, (0, 2)], 0.0, 640.0)
    scaled_boxes[:, (1, 3)] = np.clip(scaled_boxes[:, (1, 3)], 0.0, 480.0)
    if not np.allclose(
        arrays["prompt_box_xyxy_640x480"], scaled_boxes, rtol=0.0, atol=1e-4
    ):
        raise S3aMaskliftOracleError("S3a OWL-to-depth box transform mismatch")
    for scene_index in range(len(scenes)):
        positions = np.flatnonzero(arrays["scene_index"] == scene_index)
        ordinals = arrays["schedule_ordinal"][positions]
        frames = arrays["frame_id"][positions]
        if np.any(ordinals < 0) or np.any(np.diff(ordinals) < 0) or np.any(
            np.diff(frames) < 0
        ):
            raise S3aMaskliftOracleError("S3a schedule order is not chronological")

    codebook_raw = _mapping(manifest.get("abstention_codes"), "abstention codes")
    try:
        codebook = {int(key): str(value) for key, value in codebook_raw.items()}
    except (TypeError, ValueError) as error:
        raise S3aMaskliftOracleError("invalid abstention codebook") from error
    if codebook.get(0) != "emitted_q02_q98":
        raise S3aMaskliftOracleError(
            "abstention code zero must mean emitted_q02_q98"
        )
    observed_codes = set(int(value) for value in np.unique(arrays["abstention_code"]))
    if not observed_codes.issubset(codebook):
        raise S3aMaskliftOracleError("S3a contains an undefined abstention code")
    observed_code_counts = {
        str(code): int(np.count_nonzero(arrays["abstention_code"] == code))
        for code in sorted(codebook)
    }
    if manifest.get("abstention_count_by_code") != observed_code_counts:
        raise S3aMaskliftOracleError("S3a abstention count ledger mismatch")
    accepted_count = int(np.count_nonzero(accepted))
    if (
        manifest.get("accepted_row_count") != accepted_count
        or manifest.get("abstained_row_count") != count - accepted_count
    ):
        raise S3aMaskliftOracleError("S3a accepted/abstained count mismatch")
    scene_ledger = _mapping(manifest.get("scenes"), "sealed S3a scenes")
    if set(scene_ledger) != set(scenes):
        raise S3aMaskliftOracleError("S3a scene ledger keys mismatch")
    for scene_index, scene in enumerate(scenes):
        ledger = _mapping(scene_ledger[scene], f"S3a scene {scene}")
        positions = np.flatnonzero(arrays["scene_index"] == scene_index)
        if (
            ledger.get("row_count") != len(positions)
            or ledger.get("accepted_row_count")
            != int(np.count_nonzero(accepted[positions]))
        ):
            raise S3aMaskliftOracleError(f"S3a scene count ledger mismatch for {scene}")
        source = _mapping(ledger.get("source"), f"S3a source {scene}")
        for prefix in ("owl", "boxer"):
            source_path = _regular_file(
                Path(str(source.get(f"{prefix}_csv_path"))),
                f"S3a {prefix} CSV for {scene}",
            )
            if _sha256(source_path) != source.get(f"{prefix}_csv_sha256"):
                raise S3aMaskliftOracleError(
                    f"S3a {prefix} CSV hash mismatch for {scene}"
                )
        schedule = _mapping(ledger.get("schedule"), f"S3a schedule {scene}")
        schedule_path = _regular_file(
            Path(str(schedule.get("manifest_path"))), f"S3a schedule for {scene}"
        )
        if _sha256(schedule_path) != schedule.get("manifest_sha256"):
            raise S3aMaskliftOracleError(f"S3a schedule hash mismatch for {scene}")
    return manifest, arrays, before, codebook


def _aligned_aabb_minmax(
    center: np.ndarray, extent: np.ndarray, alignment: np.ndarray
) -> np.ndarray:
    center = np.asarray(center, dtype=np.float64)
    extent = np.asarray(extent, dtype=np.float64)
    if center.shape != extent.shape or center.ndim != 2 or center.shape[1:] != (3,):
        raise S3aMaskliftOracleError("S3a AABB center/extent schema is invalid")
    if len(center) == 0:
        return np.empty((0, 6), dtype=np.float64)
    corners = center[:, None, :] + _AABB_SIGNS[None, :, :] * extent[:, None, :] / 2.0
    aligned = corners @ alignment[:3, :3].T + alignment[:3, 3]
    return np.concatenate((aligned.min(axis=1), aligned.max(axis=1)), axis=1)


def _best_values(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if matrix.shape[1] == 0:
        return np.zeros(len(matrix), dtype=np.float64), np.full(len(matrix), -1, np.int64)
    indices = np.argmax(matrix, axis=1).astype(np.int64)
    return matrix[np.arange(len(matrix)), indices], indices


def _best_distribution(matrices: Sequence[np.ndarray]) -> dict[str, Any]:
    values = np.concatenate([_best_values(matrix)[0] for matrix in matrices])
    if len(values) == 0:
        return {
            "count": 0,
            "minimum": 0.0,
            "mean": 0.0,
            "median": 0.0,
            "maximum": 0.0,
            "strictly_above": {_threshold_key(t): 0 for t in THRESHOLDS},
        }
    return {
        "count": int(len(values)),
        "minimum": float(values.min()),
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "maximum": float(values.max()),
        "strictly_above": {
            _threshold_key(t): int(np.count_nonzero(values > t)) for t in THRESHOLDS
        },
    }


def _geometry_report(
    *,
    scenes: Sequence[str],
    candidate_iou: Sequence[np.ndarray],
    candidate_global_rows: Sequence[np.ndarray],
    baseline_iou: Sequence[np.ndarray],
    total_gt: int,
) -> dict[str, Any]:
    reports: dict[str, Any] = {}
    for threshold in THRESHOLDS:
        candidate_total = native_total = union_total = 0
        per_scene: dict[str, Any] = {}
        all_pairs: list[dict[str, Any]] = []
        for scene, candidates, global_rows, native in zip(
            scenes, candidate_iou, candidate_global_rows, baseline_iou
        ):
            pairs = strict_maximum_matching(candidates, threshold)
            native_pairs = strict_maximum_matching(native, threshold)
            union_pairs = strict_maximum_matching(
                np.concatenate((native, candidates), axis=0), threshold
            )
            candidate_total += len(pairs)
            native_total += len(native_pairs)
            union_total += len(union_pairs)
            for local_row, gt_index in pairs:
                all_pairs.append(
                    {
                        "scene_id": scene,
                        "sidecar_row": int(global_rows[local_row]),
                        "gt_index": int(gt_index),
                        "iou": float(candidates[local_row, gt_index]),
                    }
                )
            per_scene[scene] = {
                "candidate_count": int(len(candidates)),
                "candidate_maximum_matching_count": len(pairs),
                "native_maximum_matching_count": len(native_pairs),
                "native_union_maximum_matching_count": len(union_pairs),
                "additional_union_matching_over_native": len(union_pairs)
                - len(native_pairs),
            }
        additional = union_total - native_total
        reports[_threshold_key(threshold)] = {
            "iou_threshold": threshold,
            "strict_iou_comparison": ">",
            "candidate_maximum_matching_count": candidate_total,
            "candidate_maximum_matching_pairs": all_pairs,
            "native_maximum_matching_count": native_total,
            "native_union_maximum_matching_count": union_total,
            "additional_union_matching_over_native": additional,
            "incremental_recall_headroom_points": (
                100.0 * additional / total_gt if total_gt else 0.0
            ),
            "continuation_required_additional_matches": CONTINUATION_MIN_MATCHES,
            "passes_plus3_continuation_gate": additional
            >= CONTINUATION_MIN_MATCHES,
            "per_scene": per_scene,
        }
    return {
        "candidate_count": int(sum(len(matrix) for matrix in candidate_iou)),
        "best_iou_distribution": _best_distribution(candidate_iou),
        "per_threshold": reports,
    }


def _per_row_changes(
    *,
    scenes: Sequence[str],
    arrays: Mapping[str, np.ndarray],
    raw_iou: Sequence[np.ndarray],
    primary_iou: Sequence[np.ndarray],
    diagnostic_iou: Sequence[np.ndarray],
    primary_rows: Sequence[np.ndarray],
    diagnostic_rows: Sequence[np.ndarray],
) -> dict[str, Any]:
    count = len(arrays["sealed_npz_row"])
    raw_best = np.zeros(count, dtype=np.float64)
    raw_gt = np.full(count, -1, dtype=np.int64)
    primary_best = np.full(count, np.nan, dtype=np.float64)
    primary_gt = np.full(count, -1, dtype=np.int64)
    diagnostic_best = np.full(count, np.nan, dtype=np.float64)
    diagnostic_gt = np.full(count, -1, dtype=np.int64)
    for scene_index, _scene in enumerate(scenes):
        scene_rows = np.flatnonzero(arrays["scene_index"] == scene_index)
        raw_values, raw_indices = _best_values(raw_iou[scene_index])
        if len(scene_rows) != len(raw_values):
            raise S3aMaskliftOracleError("raw per-row IoU order mismatch")
        raw_best[scene_rows] = raw_values
        raw_gt[scene_rows] = raw_indices
        rows = primary_rows[scene_index]
        values, indices = _best_values(primary_iou[scene_index])
        diag_rows = diagnostic_rows[scene_index]
        diag_values, diag_indices = _best_values(diagnostic_iou[scene_index])
        if len(rows) != len(values) or len(diag_rows) != len(diag_values):
            raise S3aMaskliftOracleError("MobileSAM per-row IoU order mismatch")
        primary_best[rows] = values
        primary_gt[rows] = indices
        diagnostic_best[diag_rows] = diag_values
        diagnostic_gt[diag_rows] = diag_indices

    accepted = arrays["accepted"]
    rows: list[dict[str, Any]] = []
    for row in range(count):
        item: dict[str, Any] = {
            "sidecar_row": row,
            "scene_id": scenes[int(arrays["scene_index"][row])],
            "frame_id": int(arrays["frame_id"][row]),
            "sealed_npz_row": int(arrays["sealed_npz_row"][row]),
            "boxer_source_row": int(arrays["boxer_source_row"][row]),
            "accepted": bool(accepted[row]),
            "abstention_code": int(arrays["abstention_code"][row]),
            "raw_best_iou": float(raw_best[row]),
            "raw_best_gt_index": int(raw_gt[row]),
            "primary_best_iou": None,
            "primary_best_gt_index": None,
            "primary_minus_raw_best_iou": None,
            "diagnostic_best_iou": None,
            "diagnostic_best_gt_index": None,
            "diagnostic_minus_raw_best_iou": None,
        }
        if accepted[row]:
            item.update(
                {
                    "primary_best_iou": float(primary_best[row]),
                    "primary_best_gt_index": int(primary_gt[row]),
                    "primary_minus_raw_best_iou": float(
                        primary_best[row] - raw_best[row]
                    ),
                }
            )
        if arrays["diagnostic_box_valid"][row]:
            item.update(
                {
                    "diagnostic_best_iou": float(diagnostic_best[row]),
                    "diagnostic_best_gt_index": int(diagnostic_gt[row]),
                    "diagnostic_minus_raw_best_iou": float(
                        diagnostic_best[row] - raw_best[row]
                    ),
                }
            )
        rows.append(item)

    crossing: dict[str, Any] = {}
    for threshold in THRESHOLDS:
        raw_above = raw_best > threshold
        primary_above = primary_best > threshold
        diagnostic_above = diagnostic_best > threshold

        def counts(other: np.ndarray, valid: np.ndarray) -> dict[str, int]:
            return {
                "gain_crossing": int(np.count_nonzero(valid & ~raw_above & other)),
                "loss_crossing": int(np.count_nonzero(valid & raw_above & ~other)),
                "both_above": int(np.count_nonzero(valid & raw_above & other)),
                "both_below_or_equal": int(
                    np.count_nonzero(valid & ~raw_above & ~other)
                ),
                "invalid": int(np.count_nonzero(~valid)),
            }

        crossing[_threshold_key(threshold)] = {
            "iou_threshold": threshold,
            "strict_iou_comparison": ">",
            "primary_q02_q98": counts(primary_above, accepted),
            "diagnostic_q00_q100": counts(
                diagnostic_above, arrays["diagnostic_box_valid"]
            ),
        }
    delta = primary_best[accepted] - raw_best[accepted]
    return {
        "row_count": count,
        "accepted_row_count": int(np.count_nonzero(accepted)),
        "primary_minus_raw_distribution": {
            "count": int(len(delta)),
            "minimum": float(delta.min()) if len(delta) else 0.0,
            "mean": float(delta.mean()) if len(delta) else 0.0,
            "median": float(np.median(delta)) if len(delta) else 0.0,
            "maximum": float(delta.max()) if len(delta) else 0.0,
            "improved": int(np.count_nonzero(delta > 0.0)),
            "unchanged": int(np.count_nonzero(delta == 0.0)),
            "worsened": int(np.count_nonzero(delta < 0.0)),
        },
        "strict_threshold_crossings": crossing,
        "rows": rows,
    }


def _abstention_report(
    arrays: Mapping[str, np.ndarray], scenes: Sequence[str], codebook: Mapping[int, str]
) -> dict[str, Any]:
    accepted = arrays["accepted"]
    codes = arrays["abstention_code"]

    def summarize(positions: np.ndarray) -> dict[str, Any]:
        total = len(positions)
        accepted_count = int(np.count_nonzero(accepted[positions]))
        counts = {
            str(code): int(np.count_nonzero(codes[positions] == code))
            for code in sorted(codebook)
        }
        return {
            "row_count": total,
            "accepted_count": accepted_count,
            "abstained_count": total - accepted_count,
            "abstention_rate": (float((total - accepted_count) / total) if total else 0.0),
            "code_counts": counts,
        }

    return {
        "codebook": {str(key): value for key, value in sorted(codebook.items())},
        "overall": summarize(np.arange(len(accepted), dtype=np.int64)),
        "per_scene": {
            scene: summarize(np.flatnonzero(arrays["scene_index"] == scene_index))
            for scene_index, scene in enumerate(scenes)
        },
    }


def audit_scannet_s3a_mobilesam_masklift_oracle(
    *,
    s3a_json: Path,
    s3a_npz: Path,
    raw_boxer_json: Path,
    raw_boxer_npz: Path,
    topk_receipt: Path,
    preregistration: Path,
    baseline_root: Path,
    gt_root: Path,
    scan_root: Path,
    enforce_production_hashes: bool = True,
) -> dict[str, Any]:
    """Audit one sealed S3a per-view sidecar without creating predictions."""

    paths = {
        "s3a_json": s3a_json.resolve(),
        "s3a_npz": s3a_npz.resolve(),
        "raw_boxer_json": raw_boxer_json.resolve(),
        "raw_boxer_npz": raw_boxer_npz.resolve(),
        "topk_receipt": topk_receipt.resolve(),
        "preregistration": preregistration.resolve(),
    }
    baseline_root = baseline_root.resolve()
    gt_root = gt_root.resolve()
    scan_root = scan_root.resolve()
    if not baseline_root.is_dir():
        raise S3aMaskliftOracleError("baseline root must be a directory")

    # Finish every no-GT validation, including exact Top-4 membership and
    # native hashes, before resolving the first GT or axis-alignment file.
    try:
        raw_manifest, raw_arrays, scenes, raw_hashes = _load_raw_boxer_sidecar(
            paths["raw_boxer_json"], paths["raw_boxer_npz"]
        )
    except Exception as error:
        raise S3aMaskliftOracleError(f"invalid sealed raw Boxer sidecar: {error}") from error
    if raw_manifest.get("schema") != RAW_BOXER_SCHEMA:
        raise S3aMaskliftOracleError("raw Boxer schema mismatch")
    if enforce_production_hashes:
        if tuple(scenes) != DEV3_SCENES:
            raise S3aMaskliftOracleError("oracle accepts only the fixed dev3 scene order")
        if raw_hashes != {
            "json": RAW_BOXER_JSON_SHA256,
            "npz": RAW_BOXER_NPZ_SHA256,
        }:
            raise S3aMaskliftOracleError("raw Boxer production hashes mismatch")
    elif any(SCENE_PATTERN.fullmatch(scene) is None for scene in scenes):
        raise S3aMaskliftOracleError("invalid synthetic scene name")

    selections = _select_per_frame_topk(raw_arrays, scenes)[4]
    selection_sha = _selection_sha256(selections)
    if enforce_production_hashes and selection_sha != TOP4_SELECTION_SHA256:
        raise S3aMaskliftOracleError("recomputed Top-4 selection hash mismatch")
    topk, topk_hash = _load_topk_receipt(
        paths["topk_receipt"],
        scenes=scenes,
        expected_selection_sha256=selection_sha,
        enforce_production_hashes=enforce_production_hashes,
    )
    frozen_t05_hashes = _frozen_topk_native_hashes(topk, scenes)
    for scene in scenes:
        frozen_path = _regular_file(
            baseline_root / f"{scene}_boxes.pkl",
            f"frozen T05 prediction for {scene}",
        )
        if _sha256(frozen_path) != frozen_t05_hashes[scene]:
            raise S3aMaskliftOracleError(
                f"baseline does not match frozen T05 receipt for {scene}"
            )
    manifest, arrays, s3a_hashes, codebook = _load_s3a_sidecar(
        json_path=paths["s3a_json"],
        npz_path=paths["s3a_npz"],
        raw_json_path=paths["raw_boxer_json"],
        raw_npz_path=paths["raw_boxer_npz"],
        topk_receipt_path=paths["topk_receipt"],
        baseline_root=baseline_root,
        preregistration_path=paths["preregistration"],
        raw_arrays=raw_arrays,
        scenes=scenes,
        selected_by_scene=selections,
        selection_sha256=selection_sha,
        enforce_production_hashes=enforce_production_hashes,
    )

    native_before_manifest = _mapping(
        manifest.get("native_prediction_sha256_before"), "S3a native-before hashes"
    )
    native_after_manifest = _mapping(
        manifest.get("native_prediction_sha256_after"), "S3a native-after hashes"
    )
    if native_before_manifest != native_after_manifest or set(native_before_manifest) != set(
        scenes
    ):
        raise S3aMaskliftOracleError("S3a native before/after identity mismatch")
    baseline_paths: dict[str, Path] = {}
    baseline_hashes: dict[str, str] = {}
    for scene in scenes:
        path = _regular_file(
            baseline_root / f"{scene}_boxes.pkl", f"frozen T05 prediction for {scene}"
        )
        digest = _sha256(path)
        if native_before_manifest.get(scene) != digest:
            raise S3aMaskliftOracleError(f"S3a native prediction hash mismatch for {scene}")
        if digest != frozen_t05_hashes[scene]:
            raise S3aMaskliftOracleError(
                f"S3a native ledger does not match frozen T05 receipt for {scene}"
            )
        baseline_paths[scene] = path
        baseline_hashes[scene] = digest

    no_gt_before = {
        **{name: _sha256(path) for name, path in paths.items()},
        "baseline": dict(baseline_hashes),
    }

    gt_counts: list[int] = []
    baseline_iou: list[np.ndarray] = []
    raw_iou: list[np.ndarray] = []
    primary_iou: list[np.ndarray] = []
    diagnostic_iou: list[np.ndarray] = []
    raw_rows_by_scene: list[np.ndarray] = []
    valid_rows_by_scene: list[np.ndarray] = []
    diagnostic_rows_by_scene: list[np.ndarray] = []
    oracle_hashes_before: dict[str, dict[str, str]] = {}
    for scene_index, scene in enumerate(scenes):
        gt_path = _regular_file(gt_root / f"{scene}_bbox.npy", f"ScanNet GT for {scene}")
        axis_path = _regular_file(
            scan_root / scene / f"{scene}.txt", f"axis alignment for {scene}"
        )
        alignment = load_axis_alignment(axis_path)
        gt = load_gt_minmax(gt_path)
        _, native_aligned = load_baseline_boxes(baseline_paths[scene], alignment)
        baseline_iou.append(aligned_iou_matrix(native_aligned, gt))
        gt_counts.append(len(gt))

        sidecar_rows = np.flatnonzero(arrays["scene_index"] == scene_index)
        expected_raw_rows = selections[scene_index]
        if not np.array_equal(arrays["sealed_npz_row"][sidecar_rows], expected_raw_rows):
            raise S3aMaskliftOracleError(f"scene membership order mismatch for {scene}")
        raw_aligned = _aligned_candidate_minmax(
            raw_arrays, expected_raw_rows, alignment
        )
        raw_iou.append(aligned_iou_matrix(raw_aligned, gt))
        raw_rows_by_scene.append(sidecar_rows)

        valid_rows = sidecar_rows[arrays["accepted"][sidecar_rows]]
        valid_rows_by_scene.append(valid_rows)
        diagnostic_rows = sidecar_rows[
            arrays["diagnostic_box_valid"][sidecar_rows]
        ]
        diagnostic_rows_by_scene.append(diagnostic_rows)
        primary_aligned = _aligned_aabb_minmax(
            arrays["reported_q02_q98_center_world"][valid_rows],
            arrays["reported_q02_q98_extent_xyz"][valid_rows],
            alignment,
        )
        diagnostic_aligned = _aligned_aabb_minmax(
            arrays["diagnostic_q00_q100_center_world"][diagnostic_rows],
            arrays["diagnostic_q00_q100_extent_xyz"][diagnostic_rows],
            alignment,
        )
        primary_iou.append(aligned_iou_matrix(primary_aligned, gt))
        diagnostic_iou.append(aligned_iou_matrix(diagnostic_aligned, gt))
        oracle_hashes_before[scene] = {
            "gt": _sha256(gt_path),
            "axis_alignment": _sha256(axis_path),
        }

    total_gt = int(sum(gt_counts))
    geometries = {
        "raw_boxer_obb": {
            "source": "sealed_frozen_boxer_oriented_box",
            "membership": "exact_frozen_top4_all_rows",
            **_geometry_report(
                scenes=scenes,
                candidate_iou=raw_iou,
                candidate_global_rows=raw_rows_by_scene,
                baseline_iou=baseline_iou,
                total_gt=total_gt,
            ),
        },
        "primary_q02_q98": {
            "source": "mobilesam_cleaned_depth_points_world_q02_q98",
            "membership": "exact_frozen_top4_with_abstentions_recorded",
            **_geometry_report(
                scenes=scenes,
                candidate_iou=primary_iou,
                candidate_global_rows=valid_rows_by_scene,
                baseline_iou=baseline_iou,
                total_gt=total_gt,
            ),
        },
        "diagnostic_q00_q100": {
            "source": "same_mobilesam_cleaned_depth_points_world_q00_q100",
            "membership": "identical_to_primary_valid_rows_no_gt_selection",
            **_geometry_report(
                scenes=scenes,
                candidate_iou=diagnostic_iou,
                candidate_global_rows=diagnostic_rows_by_scene,
                baseline_iou=baseline_iou,
                total_gt=total_gt,
            ),
        },
    }
    primary_thresholds = geometries["primary_q02_q98"]["per_threshold"]
    gate_by_threshold = {
        key: {
            "additional_union_matching_over_native": int(
                primary_thresholds[key]["additional_union_matching_over_native"]
            ),
            "required": CONTINUATION_MIN_MATCHES,
            "passes": bool(primary_thresholds[key]["passes_plus3_continuation_gate"]),
        }
        for key in (_threshold_key(value) for value in THRESHOLDS)
    }
    continuation_passes = all(row["passes"] for row in gate_by_threshold.values())
    row_changes = _per_row_changes(
        scenes=scenes,
        arrays=arrays,
        raw_iou=raw_iou,
        primary_iou=primary_iou,
        diagnostic_iou=diagnostic_iou,
        primary_rows=valid_rows_by_scene,
        diagnostic_rows=diagnostic_rows_by_scene,
    )

    no_gt_after = {
        **{name: _sha256(path) for name, path in paths.items()},
        "baseline": {scene: _sha256(path) for scene, path in baseline_paths.items()},
    }
    oracle_hashes_after = {
        scene: {
            "gt": _sha256(gt_root / f"{scene}_bbox.npy"),
            "axis_alignment": _sha256(scan_root / scene / f"{scene}.txt"),
        }
        for scene in scenes
    }
    if no_gt_after != no_gt_before or oracle_hashes_after != oracle_hashes_before:
        raise S3aMaskliftOracleError("one or more oracle inputs changed during execution")

    return {
        "schema": SCHEMA,
        "posthoc_dev_diagnostic": True,
        "deployable": False,
        "not_deployable": True,
        "H10_authorized": False,
        "H10_not_authorized": True,
        "h10_gt_accessed": False,
        "full100_authorized": False,
        "full100_not_authorized": True,
        "full100_accessed": False,
        "active_birth_authorized": False,
        "birth": False,
        "prediction_suffix_created": False,
        "active_suffix_ap_computed": False,
        "ap_computed": False,
        "threshold_tuning_performed": False,
        "candidate_selection_applied": False,
        "candidate_suppression_applied": False,
        "candidate_ranking_applied": False,
        "labels_read": False,
        "clip_read": False,
        "strict_iou_comparison": ">",
        "scene_order": list(scenes),
        "thresholds": list(THRESHOLDS),
        "gt_count": total_gt,
        "frozen_membership_count": int(len(arrays["sealed_npz_row"])),
        "abstentions": _abstention_report(arrays, scenes, codebook),
        "geometries": geometries,
        "per_row_best_iou_changes": row_changes,
        "continuation_gate": {
            "geometry": "primary_q02_q98",
            "required_additional_union_matches_at_every_threshold": CONTINUATION_MIN_MATCHES,
            "per_threshold": gate_by_threshold,
            "passes_all_thresholds": continuation_passes,
            "only_authorizes_new_preregistered_s3b_shadow": continuation_passes,
            "does_not_authorize_H10_full100_birth_or_AP": True,
        },
        "input_sha256_before": {
            "no_gt": no_gt_before,
            "oracle_only": oracle_hashes_before,
        },
        "input_sha256_after": {
            "no_gt": no_gt_after,
            "oracle_only": oracle_hashes_after,
        },
        "input_hash_identity": True,
        "native_prediction_hash_identity": True,
        "sealed_s3a": {
            "json_path": os.fspath(paths["s3a_json"]),
            "json_sha256": s3a_hashes["json"],
            "npz_path": os.fspath(paths["s3a_npz"]),
            "npz_sha256": s3a_hashes["npz"],
            "candidate_content_sha256": manifest["candidate_content_sha256"],
            "schema": manifest["schema"],
        },
        "frozen_raw_boxer": {
            "json_sha256": raw_hashes["json"],
            "npz_sha256": raw_hashes["npz"],
            "schema": raw_manifest["schema"],
            "top4_selection_sha256": selection_sha,
            "topk_receipt_sha256": topk_hash,
        },
        "conclusion_guardrail": (
            "Post-hoc per-view geometry diagnosis only. A passing +3 gate can "
            "justify a separately preregistered no-GT S3b shadow, but cannot "
            "authorize H10, full100, a prediction suffix, AP, or active birth."
        ),
    }


def _write_json_create_only(path: Path, payload: Mapping[str, Any]) -> None:
    path = path.resolve()
    if path.exists() or path.is_symlink():
        raise S3aMaskliftOracleError(f"refusing to overwrite output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError as error:
        raise S3aMaskliftOracleError(f"refusing to overwrite output: {path}") from error


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--s3a-json", required=True, type=Path)
    parser.add_argument("--s3a-npz", required=True, type=Path)
    parser.add_argument("--raw-boxer-json", required=True, type=Path)
    parser.add_argument("--raw-boxer-npz", required=True, type=Path)
    parser.add_argument("--topk-receipt", required=True, type=Path)
    parser.add_argument("--preregistration", required=True, type=Path)
    parser.add_argument("--baseline-root", required=True, type=Path)
    parser.add_argument("--gt-root", required=True, type=Path)
    parser.add_argument("--scan-root", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    output = args.out.resolve()
    protected_roots = (
        args.baseline_root.resolve(),
        args.gt_root.resolve(),
        args.scan_root.resolve(),
    )
    protected_files = {
        args.s3a_json.resolve(),
        args.s3a_npz.resolve(),
        args.raw_boxer_json.resolve(),
        args.raw_boxer_npz.resolve(),
        args.topk_receipt.resolve(),
        args.preregistration.resolve(),
    }
    if any(_is_relative_to(output, root) for root in protected_roots) or output in protected_files:
        raise S3aMaskliftOracleError("output must be outside every protected input")
    report = audit_scannet_s3a_mobilesam_masklift_oracle(
        s3a_json=args.s3a_json,
        s3a_npz=args.s3a_npz,
        raw_boxer_json=args.raw_boxer_json,
        raw_boxer_npz=args.raw_boxer_npz,
        topk_receipt=args.topk_receipt,
        preregistration=args.preregistration,
        baseline_root=args.baseline_root,
        gt_root=args.gt_root,
        scan_root=args.scan_root,
    )
    _write_json_create_only(output, report)
    print(
        json.dumps(
            {
                "schema": SCHEMA,
                "candidate_count": report["frozen_membership_count"],
                "continuation_gate_passes": report["continuation_gate"][
                    "passes_all_thresholds"
                ],
                "H10_authorized": False,
                "full100_authorized": False,
                "out": os.fspath(output),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
