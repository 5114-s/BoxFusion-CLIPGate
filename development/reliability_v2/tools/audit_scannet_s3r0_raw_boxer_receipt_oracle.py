#!/usr/bin/env python3
"""Frozen dev3 post-hoc geometry oracle for the sealed S3R0 receipt shadow.

This tool is diagnostic only.  It validates every no-GT trust anchor before
opening the first ScanNet GT or axis-alignment file, computes matching-only
headroom for raw K8, receipt-medoid, and track-any-evidence geometry, and can
write one create-only JSON receipt.  It never computes AP, selects a receipt,
exports an evidence argmax, creates a suffix, or authorizes birth/H10/full100.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from math import fsum
import hashlib
import importlib
import json
import os
from pathlib import Path
import stat
import sys
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if os.fspath(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, os.fspath(REPOSITORY_ROOT))


SCHEMA = "boxfusion.scannet_s3r0_raw_boxer_receipt_oracle.v1"
SHADOW_SCHEMA = "boxfusion.s3r_raw_boxer_past_only_shadow.v1"
TOPK_SCHEMA = "boxfusion.scannet_boxer_per_view_topk_ceiling.v1"
DEV3_SCENES = ("scene0568_00", "scene0606_01", "scene0377_02")
THRESHOLDS = (0.15, 0.25, 0.50)
CONTINUATION_MIN_MATCHES = 3

SHADOW_ROOT = (
    REPOSITORY_ROOT / "logs" / "scannet_s3r_raw_boxer_k8_receipt_shadow_dev3_score05"
)
SHADOW_JSON = SHADOW_ROOT / "s3r_raw_boxer_k8_receipt_shadow.json"
SHADOW_NPZ = SHADOW_ROOT / "s3r_raw_boxer_k8_receipt_shadow.npz"
RAW_ROOT = (
    REPOSITORY_ROOT
    / "logs"
    / "scannet_boxer_unexplained_shadow_clean_in2_v5_score05"
    / "sealed"
)
RAW_JSON = RAW_ROOT / "boxer_shadow_candidates.json"
RAW_NPZ = RAW_ROOT / "boxer_shadow_candidates.npz"
TOPK_RECEIPT = (
    REPOSITORY_ROOT
    / "logs"
    / "scannet_boxer_per_view_topk_raw_ceiling_score05_dev3_v5.json"
)
PREREGISTRATION = (
    REPOSITORY_ROOT / "docs" / "S3R_RAW_BOXER_PAST_ONLY_PREREGISTRATION.md"
)
TOPK_TOOL = REPOSITORY_ROOT / "tools" / "audit_scannet_boxer_per_view_topk_ceiling.py"
GEOMETRY_HELPERS_SOURCE = (
    REPOSITORY_ROOT / "tools" / "audit_scannet_boxer_unexplained_oracle.py"
)
TRACKER_SOURCE = REPOSITORY_ROOT / "boxfusion" / "s3r_receipt_tracker.py"
TRACKER_TEST = REPOSITORY_ROOT / "tests" / "test_s3r_receipt_tracker.py"
RUNNER_SOURCE = (
    REPOSITORY_ROOT / "tools" / "run_scannet_s3r_raw_boxer_k8_receipt_shadow.py"
)
RUNNER_TEST = (
    REPOSITORY_ROOT / "tests" / "test_run_scannet_s3r_raw_boxer_k8_receipt_shadow.py"
)
ORACLE_SOURCE = Path(__file__).resolve()
ORACLE_TEST = (
    REPOSITORY_ROOT / "tests" / "test_audit_scannet_s3r0_raw_boxer_receipt_oracle.py"
)
SCHEDULE_ROOT = (
    REPOSITORY_ROOT
    / "cache"
    / "cutr_postfilter_v3"
    / "scannet-graw-e2-score05-preflight3-v3-r1"
)
BASELINE_ROOT = REPOSITORY_ROOT / "results" / "scannet_topk_fusion_score05"
GT_ROOT = REPOSITORY_ROOT / "evaluation" / "data_util" / "scannet_train_detection_data"
SCAN_ROOT = Path("/extra/ZhaoX/scannet_data/scans")

EXPECTED_SHADOW_JSON_SHA256 = (
    "58e6f16aeb3731ca4090a9efbcc2dcf5b93db316065cfe84c1206819bc64fa17"
)
EXPECTED_SHADOW_NPZ_SHA256 = (
    "ee95be66995dace2b467805b873ad54decd7333bd5d09a3524f48af8cb280fa6"
)
EXPECTED_SHADOW_CONTENT_SHA256 = (
    "a29622260ffef166aaae3b820a06ebf56edbe9b674f6f4f6fe14ccd2fb28cefd"
)
EXPECTED_RAW_JSON_SHA256 = (
    "84eb4f2c62d1573d9e9f1ec4c3df5a6cac16ad10c8cece0989d37dd97b734e9e"
)
EXPECTED_RAW_NPZ_SHA256 = (
    "c1a921d70de447bf528711a71deb34cf93a9bf671d3514baafa42b7b1b8b4a6c"
)
EXPECTED_RAW_CONTENT_SHA256 = (
    "8b2362cc11517a58f2a05b371698cf3a45db6805b27c4c1dd10a3c9b899ab529"
)
EXPECTED_TOPK_RECEIPT_SHA256 = (
    "d4ba67b37d362842333ac525abe32f6807c4fba90af83b699bbfc1494aa5ea1f"
)
EXPECTED_PREREGISTRATION_SHA256 = (
    "14f29a50dd65ee791be2df519e0000cf22bfc94a0209880f3539159acf4f7df3"
)
EXPECTED_K8_SELECTION_SHA256 = (
    "34ee638d51b3bc137253b3e361a60d84e110e114d2b46c487651550e708aa638"
)
EXPECTED_ALLOWED_SOURCE_CONTENT_SHA256 = (
    "be772e9c6a58a1d7e3b3fe132b5ec7125b08ef9ba0f2a5f5244107739f330ac1"
)
EXPECTED_TOPK_TOOL_SHA256 = (
    "9a756f474e40e7b991453b09cb006b1147432aab124a55d33e4613d2adad1b44"
)
EXPECTED_GEOMETRY_HELPERS_SHA256 = (
    "fcd29c9cd33199544e7f5221f78d916b8c1ae7f8e83007a28e9c6fa834f65a50"
)
EXPECTED_TRACKER_SOURCE_SHA256 = (
    "277316c36b7a7fcb8005a24e907e0f232e41f6b5874411293eb26b0744df9628"
)
EXPECTED_TRACKER_TEST_SHA256 = (
    "f08fd59ee2888c936e5b783de668fd789ba6b676bc4864e001b000ea287b1e3c"
)
EXPECTED_RUNNER_SOURCE_SHA256 = (
    "0747204029db234d17d2a55f3aa0131bafc5e1b37bd1ebec650f9a2aedc6b3db"
)
EXPECTED_RUNNER_TEST_SHA256 = (
    "9b73cc24ef8430e47c635a73d26247129dab27320a1177d97ede379c4b551b35"
)
EXPECTED_SCHEDULE_SHA256 = {
    "scene0568_00": "1ee049e9ad8263e8d7c19838a1038445129a1ae7265434f042ea0c438f3ab19a",
    "scene0606_01": "aedfe2f230c252fb9aaad10b678e3264b8855cfe1150f8b36b291d48e5032753",
    "scene0377_02": "9a8c127b09c36140494a8288425d6b23087b5865d3789b295ed55744d6edf80e",
}
EXPECTED_NATIVE_SHA256 = {
    "scene0568_00": "b55ce48fb6eb4dad9ee5bfe7007c3dbc9898b3f72ddbc5ad428b8be6414bcd2d",
    "scene0606_01": "d4e8d6dc85c917ac1634b81a45adb3866279d3e02f470c43b23bd71f5bb3ef1c",
    "scene0377_02": "ed7f849a33d45eebe846559a90aeb7de1a97f2eb169c3a7c0cb5de61d3dab35b",
}
EXPECTED_GT_SHA256 = {
    "scene0568_00": "2bc805498d02d2052e0a79c613b50b9308a9318778da5f4ad16e3087bd0abd40",
    "scene0606_01": "62ffe090a5b9278172b932147a2ceac9fdb11df19d99ad3c4a053eaebb47f82c",
    "scene0377_02": "108958cc4bcab574ca73143c39c7bd9f62c5131e215beb37227e5ddde90c6ac1",
}
EXPECTED_AXIS_SHA256 = {
    "scene0568_00": "5607de5e14f92129490eaf29ebc6f72aa471770704b0a33771737c2f33cb8a0c",
    "scene0606_01": "bc09aa77d542c80a9470b6943099ed5bd5efc4c14f1bb086660337db6b1c2e3c",
    "scene0377_02": "7df6c8217d604f244c8121e41eb914a0570f1351f0a78fcbee2c8b320be77d3b",
}
EXPECTED_SELECTED_COUNTS = {
    "scene0568_00": 501,
    "scene0606_01": 854,
    "scene0377_02": 216,
}
EXPECTED_RECEIPT_COUNTS = {
    "scene0568_00": 66,
    "scene0606_01": 104,
    "scene0377_02": 28,
}
EXPECTED_VALID_FRAME_COUNTS = {
    "scene0568_00": 66,
    "scene0606_01": 112,
    "scene0377_02": 30,
}
EXPECTED_CANDIDATE_FRAME_COUNTS = {
    "scene0568_00": 66,
    "scene0606_01": 110,
    "scene0377_02": 30,
}
EXPECTED_SELECTED_COUNT = 1571
EXPECTED_RECEIPT_COUNT = 198
EXPECTED_EVIDENCE_COUNT = 594
EXPECTED_VALID_FRAME_COUNT = 208
EXPECTED_GT_COUNT = 28

_EXPECTED_ARRAY_DTYPES = {
    "assignment_aabb_iou": "<f8",
    "assignment_action": "|i1",
    "assignment_center_distance_m": "<f8",
    "assignment_track_id": "<i8",
    "evidence_corners_world": "<f8",
    "evidence_frame_id": "<i8",
    "evidence_offsets": "<i8",
    "evidence_sealed_npz_row": "<i8",
    "evidence_selected_index": "<i8",
    "evidence_source_instance_id": "<i8",
    "evidence_source_row": "<i8",
    "evidence_source_score": "<f8",
    "frame_active_track_count": "<i4",
    "frame_adapter_cpu_ns": "<i8",
    "frame_adapter_wall_ns": "<i8",
    "frame_audit_complete": "|b1",
    "frame_cap_event_count": "<i2",
    "frame_created_count": "<i2",
    "frame_eligibility_check_count": "<i4",
    "frame_matched_count": "<i2",
    "frame_new_receipt_count": "<i2",
    "frame_prior_track_count": "<i4",
    "frame_retired_offsets": "<i8",
    "frame_selected_offsets": "<i8",
    "frame_tracker_cpu_ns": "<i8",
    "frame_tracker_wall_ns": "<i8",
    "receipt_center_rms_m": "<f8",
    "receipt_confirmation_frame_id": "<i8",
    "receipt_corners_world": "<f8",
    "receipt_median_pairwise_aabb_iou": "<f8",
    "receipt_medoid_evidence_index": "|i1",
    "receipt_min_medoid_aabb_extent_m": "<f8",
    "receipt_pairwise_aabb_iou": "<f8",
    "receipt_pairwise_center_distance_m": "<f8",
    "receipt_raw_mean_score": "<f8",
    "receipt_scene_index": "<i2",
    "receipt_track_id": "<i8",
    "retired_track_id": "<i8",
    "scene_ids": "<U12",
    "scene_receipt_offsets": "<i8",
    "scene_schedule_offsets": "<i8",
    "scene_selected_offsets": "<i8",
    "schedule_frame_id": "<i8",
    "schedule_scene_index": "<i2",
    "selected_center_world": "<f8",
    "selected_corners_world": "<f8",
    "selected_extent_xyz": "<f8",
    "selected_frame_id": "<i8",
    "selected_quaternion_wxyz": "<f8",
    "selected_rank_in_frame": "|i1",
    "selected_scene_index": "<i2",
    "selected_schedule_index": "<i8",
    "selected_sealed_npz_row": "<i8",
    "selected_source_instance_id": "<i8",
    "selected_source_row": "<i8",
    "selected_source_score": "<f8",
}

_SIGNS = np.asarray(
    [[x, y, z] for x in (-1.0, 1.0) for y in (-1.0, 1.0) for z in (-1.0, 1.0)],
    dtype=np.float64,
)


class S3R0OracleError(ValueError):
    """Raised when a frozen input or diagnostic invariant is invalid."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _regular_file(path: Path, label: str) -> Path:
    raw = Path(os.path.abspath(os.fspath(path)))
    current = Path(raw.anchor)
    final_stat: os.stat_result | None = None
    for part in raw.parts[1:]:
        current /= part
        try:
            final_stat = os.lstat(current)
        except OSError as error:
            raise S3R0OracleError(f"{label} must be a regular file: {raw}") from error
        if stat.S_ISLNK(final_stat.st_mode):
            raise S3R0OracleError(
                f"{label} path must not be a symlink or contain one: {current}"
            )
    if final_stat is None or not stat.S_ISREG(final_stat.st_mode):
        raise S3R0OracleError(f"{label} must be a regular file: {raw}")
    return raw


def _read_json(path: Path, label: str) -> dict[str, Any]:
    source = _regular_file(path, label)
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise S3R0OracleError(f"invalid {label}: {source}") from error
    if not isinstance(value, dict):
        raise S3R0OracleError(f"{label} must contain a JSON object: {source}")
    return value


def _hash_array(array: np.ndarray) -> str:
    value = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(value.dtype.str.encode("ascii"))
    digest.update(b"\0")
    digest.update(json.dumps(list(value.shape), separators=(",", ":")).encode("ascii"))
    digest.update(b"\0")
    digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def _array_content_sha256(arrays: Mapping[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for name in sorted(arrays):
        value = np.ascontiguousarray(arrays[name])
        digest.update(name.encode("ascii"))
        digest.update(b"\0")
        digest.update(value.dtype.str.encode("ascii"))
        digest.update(b"\0")
        digest.update(
            json.dumps(list(value.shape), separators=(",", ":")).encode("ascii")
        )
        digest.update(b"\0")
        digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def _selection_sha256(indices_by_scene: Sequence[np.ndarray]) -> str:
    digest = hashlib.sha256()
    for scene_index, values in enumerate(indices_by_scene):
        array = np.ascontiguousarray(values, dtype=np.int64)
        digest.update(np.asarray([scene_index, len(array)], dtype=np.int64).tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()


def _select_per_frame_topk(
    arrays: Mapping[str, np.ndarray], scenes: Sequence[str]
) -> dict[int, list[np.ndarray]]:
    scene_index_array = arrays["per_view_scene_index"]
    frame_ids = arrays["per_view_frame_id"]
    scores = arrays["per_view_source_score"]
    source_rows = arrays["per_view_source_row"]
    selections: list[np.ndarray] = []
    for scene_index, _scene in enumerate(scenes):
        scene_positions = np.flatnonzero(scene_index_array == scene_index)
        selected: list[int] = []
        for frame_id in sorted(np.unique(frame_ids[scene_positions]).tolist()):
            positions = scene_positions[frame_ids[scene_positions] == frame_id]
            order = sorted(
                positions.tolist(),
                key=lambda row: (
                    -float(scores[row]),
                    int(source_rows[row]),
                    int(row),
                ),
            )
            selected.extend(order[:8])
        selections.append(np.asarray(selected, dtype=np.int64))
    return {8: selections}


@dataclass(frozen=True)
class _DependencyBundle:
    load_sealed_sidecar: Any
    aligned_iou_matrix: Any
    load_axis_alignment: Any
    load_baseline_boxes: Any
    load_gt_minmax: Any
    strict_maximum_matching: Any


_DEPENDENCIES: _DependencyBundle | None = None


def _load_dependency_bundle() -> _DependencyBundle:
    """Hash the complete local dependency closure before importing any of it."""

    global _DEPENDENCIES
    if _DEPENDENCIES is not None:
        return _DEPENDENCIES
    frozen = (
        (TOPK_TOOL, EXPECTED_TOPK_TOOL_SHA256, "Top-K helper source"),
        (
            GEOMETRY_HELPERS_SOURCE,
            EXPECTED_GEOMETRY_HELPERS_SHA256,
            "geometry helper source",
        ),
    )
    before: dict[Path, str] = {}
    for path, expected, label in frozen:
        source = _regular_file(path, label)
        digest = _sha256(source)
        _require_equal(digest, expected, f"{label} bootstrap SHA-256")
        before[source] = digest
    module_names = (
        "tools.audit_scannet_boxer_per_view_topk_ceiling",
        "tools.audit_scannet_boxer_unexplained_oracle",
    )
    preloaded = [name for name in module_names if name in sys.modules]
    if preloaded:
        raise S3R0OracleError(
            "frozen oracle dependencies must first load in a fresh process: "
            + ", ".join(preloaded)
        )
    try:
        topk = importlib.import_module(module_names[0])
        geometry = importlib.import_module(module_names[1])
    except Exception as error:
        raise S3R0OracleError("could not import frozen oracle dependencies") from error
    after = {path: _sha256(path) for path in before}
    if after != before:
        raise S3R0OracleError("oracle dependency changed while importing")
    module_sources = (
        (topk, TOPK_TOOL, module_names[0]),
        (geometry, GEOMETRY_HELPERS_SOURCE, module_names[1]),
    )
    for module, expected_source, expected_name in module_sources:
        actual_name = getattr(module, "__name__", None)
        actual_file = getattr(module, "__file__", None)
        module_spec = getattr(module, "__spec__", None)
        spec_origin = getattr(module_spec, "origin", None)
        _require_equal(actual_name, expected_name, f"{expected_name} module name")
        if not isinstance(actual_file, str) or not isinstance(spec_origin, str):
            raise S3R0OracleError(f"{expected_name} lacks import provenance")
        actual_source = _regular_file(Path(actual_file), f"{expected_name} __file__")
        actual_origin = _regular_file(Path(spec_origin), f"{expected_name} spec origin")
        _require_equal(
            actual_source,
            expected_source,
            f"{expected_name} imported source",
        )
        _require_equal(
            actual_origin,
            expected_source,
            f"{expected_name} import spec origin",
        )
    names = {
        "load_sealed_sidecar": getattr(topk, "_load_sealed_sidecar", None),
        "aligned_iou_matrix": getattr(geometry, "aligned_iou_matrix", None),
        "load_axis_alignment": getattr(geometry, "load_axis_alignment", None),
        "load_baseline_boxes": getattr(geometry, "load_baseline_boxes", None),
        "load_gt_minmax": getattr(geometry, "load_gt_minmax", None),
        "strict_maximum_matching": getattr(geometry, "strict_maximum_matching", None),
    }
    if not all(callable(value) for value in names.values()):
        raise S3R0OracleError("frozen oracle dependency API mismatch")
    expected_callable_modules = {
        "load_sealed_sidecar": module_names[0],
        "aligned_iou_matrix": module_names[1],
        "load_axis_alignment": module_names[1],
        "load_baseline_boxes": module_names[1],
        "load_gt_minmax": module_names[1],
        "strict_maximum_matching": module_names[1],
    }
    for name, value in names.items():
        _require_equal(
            getattr(value, "__module__", None),
            expected_callable_modules[name],
            f"{name} callable module",
        )
    _DEPENDENCIES = _DependencyBundle(**names)
    return _DEPENDENCIES


def _load_sealed_sidecar(*args: Any, **kwargs: Any) -> Any:
    return _load_dependency_bundle().load_sealed_sidecar(*args, **kwargs)


def aligned_iou_matrix(*args: Any, **kwargs: Any) -> Any:
    return _load_dependency_bundle().aligned_iou_matrix(*args, **kwargs)


def load_axis_alignment(*args: Any, **kwargs: Any) -> Any:
    return _load_dependency_bundle().load_axis_alignment(*args, **kwargs)


def load_baseline_boxes(*args: Any, **kwargs: Any) -> Any:
    return _load_dependency_bundle().load_baseline_boxes(*args, **kwargs)


def load_gt_minmax(*args: Any, **kwargs: Any) -> Any:
    return _load_dependency_bundle().load_gt_minmax(*args, **kwargs)


def strict_maximum_matching(*args: Any, **kwargs: Any) -> Any:
    return _load_dependency_bundle().strict_maximum_matching(*args, **kwargs)


def _require_equal(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        raise S3R0OracleError(f"{label} mismatch: {actual!r} != {expected!r}")


def _require_array_equal(actual: np.ndarray, expected: np.ndarray, label: str) -> None:
    if not np.array_equal(actual, expected):
        raise S3R0OracleError(f"{label} mismatch")


def _require_allclose(actual: np.ndarray, expected: np.ndarray, label: str) -> None:
    if not np.allclose(actual, expected, rtol=0.0, atol=1e-12):
        raise S3R0OracleError(f"{label} mismatch")


def _ledger_entry(
    ledger: Mapping[str, Any], key: str, path: Path, expected_sha256: str
) -> str:
    value = ledger.get(key)
    if not isinstance(value, Mapping):
        raise S3R0OracleError(f"missing frozen ledger entry: {key}")
    source = _regular_file(path, f"frozen {key}")
    digest = _sha256(source)
    _require_equal(value.get("path"), os.fspath(source), f"{key} path")
    _require_equal(value.get("sha256"), expected_sha256, f"{key} ledger SHA-256")
    _require_equal(digest, expected_sha256, f"{key} actual SHA-256")
    _require_equal(value.get("bytes"), source.stat().st_size, f"{key} byte count")
    return digest


def _quaternion_rotation(quaternion_wxyz: np.ndarray) -> np.ndarray:
    q = np.asarray(quaternion_wxyz, dtype=np.float64)
    if q.shape != (4,) or not np.isfinite(q).all():
        raise S3R0OracleError("invalid raw Boxer quaternion")
    norm_squared = float(q @ q)
    if norm_squared <= 1e-12:
        raise S3R0OracleError("degenerate raw Boxer quaternion")
    w, x, y, z = q
    scale = 2.0 / norm_squared
    return np.asarray(
        [
            [
                1.0 - scale * (y * y + z * z),
                scale * (x * y - z * w),
                scale * (x * z + y * w),
            ],
            [
                scale * (x * y + z * w),
                1.0 - scale * (x * x + z * z),
                scale * (y * z - x * w),
            ],
            [
                scale * (x * z - y * w),
                scale * (y * z + x * w),
                1.0 - scale * (x * x + y * y),
            ],
        ],
        dtype=np.float64,
    )


def _raw_obb_corners(
    centers: np.ndarray, extents: np.ndarray, quaternions: np.ndarray
) -> np.ndarray:
    center = np.asarray(centers, dtype=np.float64)
    extent = np.asarray(extents, dtype=np.float64)
    quaternion = np.asarray(quaternions, dtype=np.float64)
    if center.ndim != 2 or center.shape[1:] != (3,) or extent.shape != center.shape:
        raise S3R0OracleError("raw Boxer center/extent schema is invalid")
    if quaternion.shape != (len(center), 4):
        raise S3R0OracleError("raw Boxer quaternion schema is invalid")
    if not np.isfinite(center).all() or not np.isfinite(extent).all():
        raise S3R0OracleError("raw Boxer geometry is non-finite")
    if np.any(extent <= 0.0):
        raise S3R0OracleError("raw Boxer extent is non-positive")
    corners = []
    for row in range(len(center)):
        local = _SIGNS * (extent[row] / 2.0)
        corners.append(local @ _quaternion_rotation(quaternion[row]).T + center[row])
    if not corners:
        return np.empty((0, 8, 3), dtype=np.float64)
    return np.stack(corners, axis=0)


def _evidence_metrics(
    evidence_corners: np.ndarray,
    evidence_frames: np.ndarray,
    evidence_source_rows: np.ndarray,
    evidence_npz_rows: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float, float, int, float]:
    corners = np.asarray(evidence_corners, dtype=np.float64)
    if corners.shape != (3, 8, 3) or not np.isfinite(corners).all():
        raise S3R0OracleError("receipt evidence corners must have shape 3x8x3")
    lower = corners.min(axis=1)
    upper = corners.max(axis=1)
    centers = 0.5 * (lower + upper)
    volumes = np.prod(upper - lower, axis=1)
    intersection_extent = np.maximum(
        np.minimum(upper[:, None, :], upper[None, :, :])
        - np.maximum(lower[:, None, :], lower[None, :, :]),
        0.0,
    )
    intersection = np.prod(intersection_extent, axis=2)
    union = volumes[:, None] + volumes[None, :] - intersection
    pairwise_iou = np.divide(
        intersection,
        union,
        out=np.zeros_like(intersection),
        where=union > 0.0,
    )
    pairwise_center = np.linalg.norm(centers[:, None, :] - centers[None, :, :], axis=2)
    median_iou = float(np.median(pairwise_iou[np.triu_indices(3, 1)]))
    centroid = centers.mean(axis=0)
    center_rms = float(np.sqrt(np.mean(np.sum((centers - centroid) ** 2, axis=1))))
    costs = np.sum(1.0 - pairwise_iou, axis=1)
    medoid = min(
        range(3),
        key=lambda index: (
            float(costs[index]),
            int(evidence_frames[index]),
            int(evidence_source_rows[index]),
            int(evidence_npz_rows[index]),
        ),
    )
    min_extent = float(np.min(upper[medoid] - lower[medoid]))
    return pairwise_iou, pairwise_center, median_iou, center_rms, medoid, min_extent


def _threshold_key(threshold: float) -> str:
    return f"{threshold:.2f}"


def _aligned_enclosing_aabb(
    corners_world: np.ndarray, alignment: np.ndarray
) -> np.ndarray:
    corners = np.asarray(corners_world, dtype=np.float64)
    transform = np.asarray(alignment, dtype=np.float64)
    if corners.ndim != 3 or corners.shape[1:] != (8, 3):
        raise S3R0OracleError(f"corners must have shape Nx8x3, got {corners.shape}")
    if transform.shape != (4, 4) or not np.isfinite(transform).all():
        raise S3R0OracleError("axis alignment must be a finite 4x4 matrix")
    if not np.isfinite(corners).all():
        raise S3R0OracleError("corners must be finite")
    if len(corners) == 0:
        return np.empty((0, 6), dtype=np.float64)
    aligned = corners @ transform[:3, :3].T + transform[:3, 3]
    return np.concatenate((aligned.min(axis=1), aligned.max(axis=1)), axis=1)


def _track_any_evidence_iou(
    evidence_iou: np.ndarray, evidence_offsets: np.ndarray
) -> np.ndarray:
    matrix = np.asarray(evidence_iou, dtype=np.float64)
    offsets = np.asarray(evidence_offsets, dtype=np.int64)
    if matrix.ndim != 2 or not np.isfinite(matrix).all():
        raise S3R0OracleError("evidence IoU must be a finite matrix")
    if offsets.ndim != 1 or len(offsets) == 0 or offsets[0] != 0:
        raise S3R0OracleError("invalid evidence offsets")
    if offsets[-1] != len(matrix) or np.any(np.diff(offsets) != 3):
        raise S3R0OracleError("every receipt must have exactly three evidence rows")
    if len(offsets) == 1:
        return np.empty((0, matrix.shape[1]), dtype=np.float64)
    return np.max(matrix.reshape(len(offsets) - 1, 3, matrix.shape[1]), axis=1)


def _expected_array_shapes(
    arrays: Mapping[str, np.ndarray]
) -> dict[str, tuple[int, ...]]:
    selected = EXPECTED_SELECTED_COUNT
    receipts = EXPECTED_RECEIPT_COUNT
    evidence = EXPECTED_EVIDENCE_COUNT
    frames = EXPECTED_VALID_FRAME_COUNT
    retired = len(arrays["retired_track_id"])
    return {
        "assignment_aabb_iou": (selected,),
        "assignment_action": (selected,),
        "assignment_center_distance_m": (selected,),
        "assignment_track_id": (selected,),
        "evidence_corners_world": (evidence, 8, 3),
        "evidence_frame_id": (evidence,),
        "evidence_offsets": (receipts + 1,),
        "evidence_sealed_npz_row": (evidence,),
        "evidence_selected_index": (evidence,),
        "evidence_source_instance_id": (evidence,),
        "evidence_source_row": (evidence,),
        "evidence_source_score": (evidence,),
        "frame_active_track_count": (frames,),
        "frame_adapter_cpu_ns": (frames,),
        "frame_adapter_wall_ns": (frames,),
        "frame_audit_complete": (frames,),
        "frame_cap_event_count": (frames,),
        "frame_created_count": (frames,),
        "frame_eligibility_check_count": (frames,),
        "frame_matched_count": (frames,),
        "frame_new_receipt_count": (frames,),
        "frame_prior_track_count": (frames,),
        "frame_retired_offsets": (frames + 1,),
        "frame_selected_offsets": (frames + 1,),
        "frame_tracker_cpu_ns": (frames,),
        "frame_tracker_wall_ns": (frames,),
        "receipt_center_rms_m": (receipts,),
        "receipt_confirmation_frame_id": (receipts,),
        "receipt_corners_world": (receipts, 8, 3),
        "receipt_median_pairwise_aabb_iou": (receipts,),
        "receipt_medoid_evidence_index": (receipts,),
        "receipt_min_medoid_aabb_extent_m": (receipts,),
        "receipt_pairwise_aabb_iou": (receipts, 3, 3),
        "receipt_pairwise_center_distance_m": (receipts, 3, 3),
        "receipt_raw_mean_score": (receipts,),
        "receipt_scene_index": (receipts,),
        "receipt_track_id": (receipts,),
        "retired_track_id": (retired,),
        "scene_ids": (len(DEV3_SCENES),),
        "scene_receipt_offsets": (len(DEV3_SCENES) + 1,),
        "scene_schedule_offsets": (len(DEV3_SCENES) + 1,),
        "scene_selected_offsets": (len(DEV3_SCENES) + 1,),
        "schedule_frame_id": (frames,),
        "schedule_scene_index": (frames,),
        "selected_center_world": (selected, 3),
        "selected_corners_world": (selected, 8, 3),
        "selected_extent_xyz": (selected, 3),
        "selected_frame_id": (selected,),
        "selected_quaternion_wxyz": (selected, 4),
        "selected_rank_in_frame": (selected,),
        "selected_scene_index": (selected,),
        "selected_schedule_index": (selected,),
        "selected_sealed_npz_row": (selected,),
        "selected_source_instance_id": (selected,),
        "selected_source_row": (selected,),
        "selected_source_score": (selected,),
    }


def _validate_s3r_arrays(
    *,
    manifest: Mapping[str, Any],
    arrays: Mapping[str, np.ndarray],
    raw_arrays: Mapping[str, np.ndarray],
    selections: Sequence[np.ndarray],
) -> None:
    if set(arrays) != set(_EXPECTED_ARRAY_DTYPES):
        missing = sorted(set(_EXPECTED_ARRAY_DTYPES) - set(arrays))
        extra = sorted(set(arrays) - set(_EXPECTED_ARRAY_DTYPES))
        raise S3R0OracleError(
            f"unexpected S3R NPZ array set; missing={missing}, extra={extra}"
        )
    expected_shapes = _expected_array_shapes(arrays)
    for name, expected_dtype in _EXPECTED_ARRAY_DTYPES.items():
        array = arrays[name]
        if array.dtype.str != expected_dtype:
            raise S3R0OracleError(
                f"S3R {name} dtype mismatch: {array.dtype.str} != {expected_dtype}"
            )
        if array.shape != expected_shapes[name]:
            raise S3R0OracleError(
                f"S3R {name} shape mismatch: {array.shape} != {expected_shapes[name]}"
            )
        if array.dtype.kind == "f" and not np.isfinite(array).all():
            raise S3R0OracleError(f"S3R {name} contains non-finite values")

    _require_array_equal(
        arrays["scene_ids"], np.asarray(DEV3_SCENES, dtype="<U12"), "scene IDs"
    )
    selected_offsets = np.asarray(
        [
            0,
            EXPECTED_SELECTED_COUNTS[DEV3_SCENES[0]],
            EXPECTED_SELECTED_COUNTS[DEV3_SCENES[0]]
            + EXPECTED_SELECTED_COUNTS[DEV3_SCENES[1]],
            EXPECTED_SELECTED_COUNT,
        ],
        dtype=np.int64,
    )
    receipt_offsets = np.asarray(
        [
            0,
            EXPECTED_RECEIPT_COUNTS[DEV3_SCENES[0]],
            EXPECTED_RECEIPT_COUNTS[DEV3_SCENES[0]]
            + EXPECTED_RECEIPT_COUNTS[DEV3_SCENES[1]],
            EXPECTED_RECEIPT_COUNT,
        ],
        dtype=np.int64,
    )
    schedule_offsets = np.asarray(
        [
            0,
            EXPECTED_VALID_FRAME_COUNTS[DEV3_SCENES[0]],
            EXPECTED_VALID_FRAME_COUNTS[DEV3_SCENES[0]]
            + EXPECTED_VALID_FRAME_COUNTS[DEV3_SCENES[1]],
            EXPECTED_VALID_FRAME_COUNT,
        ],
        dtype=np.int64,
    )
    _require_array_equal(
        arrays["scene_selected_offsets"], selected_offsets, "scene selected offsets"
    )
    _require_array_equal(
        arrays["scene_receipt_offsets"], receipt_offsets, "scene receipt offsets"
    )
    _require_array_equal(
        arrays["scene_schedule_offsets"], schedule_offsets, "scene schedule offsets"
    )
    for name, expected_end in (
        ("frame_selected_offsets", EXPECTED_SELECTED_COUNT),
        ("frame_retired_offsets", len(arrays["retired_track_id"])),
    ):
        offsets = arrays[name]
        if (
            offsets[0] != 0
            or offsets[-1] != expected_end
            or np.any(np.diff(offsets) < 0)
        ):
            raise S3R0OracleError(f"{name} is not a complete monotonic offset array")
    expected_selected_scene = np.repeat(
        np.arange(len(DEV3_SCENES), dtype=np.int16), np.diff(selected_offsets)
    )
    expected_receipt_scene = np.repeat(
        np.arange(len(DEV3_SCENES), dtype=np.int16), np.diff(receipt_offsets)
    )
    expected_schedule_scene = np.repeat(
        np.arange(len(DEV3_SCENES), dtype=np.int16), np.diff(schedule_offsets)
    )
    _require_array_equal(
        arrays["selected_scene_index"], expected_selected_scene, "selected scene index"
    )
    _require_array_equal(
        arrays["receipt_scene_index"], expected_receipt_scene, "receipt scene index"
    )
    _require_array_equal(
        arrays["schedule_scene_index"], expected_schedule_scene, "schedule scene index"
    )

    flat_selection = np.concatenate(selections).astype(np.int64, copy=False)
    if flat_selection.shape != (EXPECTED_SELECTED_COUNT,):
        raise S3R0OracleError("recomputed K8 selection count mismatch")
    _require_array_equal(
        arrays["selected_sealed_npz_row"], flat_selection, "exact K8 membership/order"
    )
    raw_mapping = {
        "selected_scene_index": "per_view_scene_index",
        "selected_frame_id": "per_view_frame_id",
        "selected_source_row": "per_view_source_row",
        "selected_source_instance_id": "per_view_source_instance_id",
        "selected_source_score": "per_view_source_score",
        "selected_center_world": "per_view_center_world",
        "selected_extent_xyz": "per_view_extent_xyz",
        "selected_quaternion_wxyz": "per_view_quaternion_wxyz",
    }
    for selected_name, raw_name in raw_mapping.items():
        expected = raw_arrays[raw_name][flat_selection].astype(
            arrays[selected_name].dtype, copy=False
        )
        _require_array_equal(
            arrays[selected_name], expected, f"raw binding {selected_name}"
        )
    expected_corners = _raw_obb_corners(
        arrays["selected_center_world"],
        arrays["selected_extent_xyz"],
        arrays["selected_quaternion_wxyz"],
    )
    _require_allclose(
        arrays["selected_corners_world"], expected_corners, "selected OBB corners"
    )
    if np.any(arrays["selected_extent_xyz"] <= 0.0):
        raise S3R0OracleError("selected extent is non-positive")
    if _hash_array(arrays["selected_sealed_npz_row"]) != manifest.get(
        "selection", {}
    ).get("selected_npz_rows_sha256"):
        raise S3R0OracleError("selected NPZ-row content hash mismatch")
    schedule_index = arrays["selected_schedule_index"]
    if np.any((schedule_index < 0) | (schedule_index >= EXPECTED_VALID_FRAME_COUNT)):
        raise S3R0OracleError("selected schedule index is out of range")
    _require_array_equal(
        arrays["selected_frame_id"],
        arrays["schedule_frame_id"][schedule_index],
        "selected frame/schedule binding",
    )
    _require_array_equal(
        arrays["selected_scene_index"],
        arrays["schedule_scene_index"][schedule_index],
        "selected scene/schedule binding",
    )
    for frame_position in range(EXPECTED_VALID_FRAME_COUNT):
        start = int(arrays["frame_selected_offsets"][frame_position])
        end = int(arrays["frame_selected_offsets"][frame_position + 1])
        if start < 0 or end < start or end - start > 8:
            raise S3R0OracleError("invalid per-frame K8 offsets")
        _require_array_equal(
            arrays["selected_rank_in_frame"][start:end],
            np.arange(end - start, dtype=np.int8),
            "selected rank in frame",
        )
    if int(arrays["frame_selected_offsets"][-1]) != EXPECTED_SELECTED_COUNT:
        raise S3R0OracleError("frame-selected offsets do not cover exact K8 membership")
    if not arrays["frame_audit_complete"].all() or np.any(
        arrays["frame_cap_event_count"] != 0
    ):
        raise S3R0OracleError("incomplete or capped S3R frame trace")

    action = arrays["assignment_action"]
    if np.any((action != 0) & (action != 1)) or np.any(
        arrays["assignment_track_id"] < 0
    ):
        raise S3R0OracleError("invalid assignment action/track identity")
    created = action == 0
    matched = action == 1
    if np.any(arrays["assignment_aabb_iou"][created] != -1.0) or np.any(
        arrays["assignment_center_distance_m"][created] != -1.0
    ):
        raise S3R0OracleError("created assignment metrics are invalid")
    if np.any(arrays["assignment_aabb_iou"][matched] < 0.10) or np.any(
        arrays["assignment_center_distance_m"][matched] > 0.50
    ):
        raise S3R0OracleError("matched assignment violates the frozen association gate")

    expected_evidence_offsets = np.arange(
        0, EXPECTED_EVIDENCE_COUNT + 1, 3, dtype=np.int64
    )
    _require_array_equal(
        arrays["evidence_offsets"], expected_evidence_offsets, "evidence offsets"
    )
    selected_index = arrays["evidence_selected_index"]
    if np.any((selected_index < 0) | (selected_index >= EXPECTED_SELECTED_COUNT)):
        raise S3R0OracleError("evidence selected index is out of range")
    if len(np.unique(selected_index)) != len(selected_index):
        raise S3R0OracleError("one selected observation appears in multiple receipts")
    evidence_mapping = {
        "evidence_frame_id": "selected_frame_id",
        "evidence_sealed_npz_row": "selected_sealed_npz_row",
        "evidence_source_row": "selected_source_row",
        "evidence_source_instance_id": "selected_source_instance_id",
        "evidence_source_score": "selected_source_score",
        "evidence_corners_world": "selected_corners_world",
    }
    for evidence_name, selected_name in evidence_mapping.items():
        _require_array_equal(
            arrays[evidence_name],
            arrays[selected_name][selected_index],
            f"evidence binding {evidence_name}",
        )

    for scene_index in range(len(DEV3_SCENES)):
        start = int(receipt_offsets[scene_index])
        end = int(receipt_offsets[scene_index + 1])
        track_ids = arrays["receipt_track_id"][start:end]
        if len(np.unique(track_ids)) != len(track_ids):
            raise S3R0OracleError("duplicate receipt track ID within one scene")
    for receipt_index in range(EXPECTED_RECEIPT_COUNT):
        start = int(arrays["evidence_offsets"][receipt_index])
        end = int(arrays["evidence_offsets"][receipt_index + 1])
        links = selected_index[start:end]
        scene_index = int(arrays["receipt_scene_index"][receipt_index])
        if not np.all(arrays["selected_scene_index"][links] == scene_index):
            raise S3R0OracleError("receipt evidence crosses scene boundary")
        if not np.all(
            arrays["assignment_track_id"][links]
            == arrays["receipt_track_id"][receipt_index]
        ):
            raise S3R0OracleError("receipt evidence/assignment track mismatch")
        frames = arrays["evidence_frame_id"][start:end]
        if not np.all(np.diff(frames) > 0):
            raise S3R0OracleError("receipt evidence frames are not strictly causal")
        if arrays["receipt_confirmation_frame_id"][receipt_index] != frames[-1]:
            raise S3R0OracleError(
                "receipt confirmation frame is not third evidence frame"
            )
        metrics = _evidence_metrics(
            arrays["evidence_corners_world"][start:end],
            frames,
            arrays["evidence_source_row"][start:end],
            arrays["evidence_sealed_npz_row"][start:end],
        )
        medoid = metrics[4]
        if int(arrays["receipt_medoid_evidence_index"][receipt_index]) != medoid:
            raise S3R0OracleError("receipt medoid index mismatch")
        _require_array_equal(
            arrays["receipt_corners_world"][receipt_index],
            arrays["evidence_corners_world"][start + medoid],
            "receipt medoid corners",
        )
        _require_allclose(
            arrays["receipt_pairwise_aabb_iou"][receipt_index],
            metrics[0],
            "receipt pairwise AABB IoU",
        )
        _require_allclose(
            arrays["receipt_pairwise_center_distance_m"][receipt_index],
            metrics[1],
            "receipt pairwise center distance",
        )
        expected_mean = (
            fsum(float(value) for value in arrays["evidence_source_score"][start:end])
            / 3.0
        )
        if abs(arrays["receipt_raw_mean_score"][receipt_index] - expected_mean) > 1e-12:
            raise S3R0OracleError("receipt raw mean score mismatch")
        scalar_checks = (
            ("receipt_median_pairwise_aabb_iou", metrics[2]),
            ("receipt_center_rms_m", metrics[3]),
            ("receipt_min_medoid_aabb_extent_m", metrics[5]),
        )
        for name, expected in scalar_checks:
            if abs(float(arrays[name][receipt_index]) - expected) > 1e-12:
                raise S3R0OracleError(f"{name} mismatch")


def _matching_report(
    *,
    scenes: Sequence[str],
    candidate_iou: Sequence[np.ndarray],
    native_iou: Sequence[np.ndarray],
) -> dict[str, Any]:
    if len(scenes) != len(candidate_iou) or len(scenes) != len(native_iou):
        raise S3R0OracleError("matching scene counts differ")
    report: dict[str, Any] = {}
    for threshold in THRESHOLDS:
        candidate_total = native_total = union_total = 0
        per_scene: dict[str, Any] = {}
        for scene, candidate, native in zip(scenes, candidate_iou, native_iou):
            candidate = np.asarray(candidate, dtype=np.float64)
            native = np.asarray(native, dtype=np.float64)
            if (
                candidate.ndim != 2
                or native.ndim != 2
                or candidate.shape[1] != native.shape[1]
                or not np.isfinite(candidate).all()
                or not np.isfinite(native).all()
            ):
                raise S3R0OracleError(
                    f"candidate/native IoU matrices are incompatible for {scene}"
                )
            candidate_pairs = strict_maximum_matching(candidate, threshold)
            native_pairs = strict_maximum_matching(native, threshold)
            union_pairs = strict_maximum_matching(
                np.concatenate((native, candidate), axis=0), threshold
            )
            candidate_total += len(candidate_pairs)
            native_total += len(native_pairs)
            union_total += len(union_pairs)
            per_scene[scene] = {
                "candidate_count": int(len(candidate)),
                "candidate_maximum_matching_count": len(candidate_pairs),
                "native_maximum_matching_count": len(native_pairs),
                "native_union_maximum_matching_count": len(union_pairs),
                "additional_union_matching_over_native": len(union_pairs)
                - len(native_pairs),
            }
        additional = union_total - native_total
        report[_threshold_key(threshold)] = {
            "iou_threshold": threshold,
            "strict_iou_comparison": ">",
            "candidate_maximum_matching_count": candidate_total,
            "native_maximum_matching_count": native_total,
            "native_union_maximum_matching_count": union_total,
            "additional_union_matching_over_native": additional,
            "passes_plus3_continuation_floor": additional >= CONTINUATION_MIN_MATCHES,
            "per_scene": per_scene,
        }
    return report


def _continuation_gate(
    medoid_thresholds: Mapping[str, Mapping[str, Any]],
    any_thresholds: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    expected_keys = {_threshold_key(value) for value in THRESHOLDS}
    if set(medoid_thresholds) != expected_keys or set(any_thresholds) != expected_keys:
        raise S3R0OracleError("continuation reports do not cover the fixed thresholds")
    by_threshold: dict[str, Any] = {}
    for threshold in THRESHOLDS:
        key = _threshold_key(threshold)
        medoid_additional = int(
            medoid_thresholds[key]["additional_union_matching_over_native"]
        )
        any_additional = int(
            any_thresholds[key]["additional_union_matching_over_native"]
        )
        medoid_passes = medoid_additional >= CONTINUATION_MIN_MATCHES
        any_passes = any_additional >= CONTINUATION_MIN_MATCHES
        by_threshold[key] = {
            "required_additional_union_matches": CONTINUATION_MIN_MATCHES,
            "receipt_medoid_additional_union_matches": medoid_additional,
            "track_any_evidence_additional_union_matches": any_additional,
            "receipt_medoid_passes": medoid_passes,
            "track_any_evidence_passes": any_passes,
            "both_confirmed_geometries_pass": medoid_passes and any_passes,
        }
    passes = all(
        value["both_confirmed_geometries_pass"] for value in by_threshold.values()
    )
    return {
        "confirmed_geometries": ["receipt_medoid", "track_any_evidence"],
        "raw_k8_is_context_only": True,
        "required_additional_union_matches_at_every_threshold": CONTINUATION_MIN_MATCHES,
        "per_threshold": by_threshold,
        "passes_all_thresholds": passes,
        "only_permits_already_preregistered_one_shot_h10_shadow": passes,
        "does_not_authorize_birth_suffix_ap_c87_or_full100": True,
    }


def _load_topk_receipt() -> tuple[dict[str, Any], str]:
    path = _regular_file(TOPK_RECEIPT, "frozen Top-K ceiling receipt")
    digest = _sha256(path)
    _require_equal(digest, EXPECTED_TOPK_RECEIPT_SHA256, "Top-K receipt SHA-256")
    receipt = _read_json(path, "frozen Top-K ceiling receipt")
    required = {
        "schema": TOPK_SCHEMA,
        "posthoc_dev_diagnostic": True,
        "not_deployable": True,
        "H10_not_authorized": True,
        "full100_not_authorized": True,
        "threshold_tuning_performed": False,
        "selection_used_gt": False,
        "selection_used_semantics": False,
        "selection_used_only_frozen_source_score": True,
        "selection_completed_before_gt_access": True,
        "selection_tie_break": "ascending_source_row_then_sealed_npz_row",
        "scene_order": list(DEV3_SCENES),
        "thresholds": list(THRESHOLDS),
        "gt_count": EXPECTED_GT_COUNT,
    }
    for key, expected in required.items():
        _require_equal(receipt.get(key), expected, f"Top-K receipt {key}")
    _require_equal(
        receipt.get("input_sha256_before"),
        receipt.get("input_sha256_after"),
        "Top-K input hash identity",
    )
    _require_equal(
        receipt.get("input_hash_identity"), True, "Top-K input identity flag"
    )
    sealed = receipt.get("sealed_sidecar")
    if not isinstance(sealed, Mapping):
        raise S3R0OracleError("Top-K sealed-sidecar ledger is missing")
    _require_equal(
        sealed.get("json_sha256"), EXPECTED_RAW_JSON_SHA256, "Top-K raw JSON"
    )
    _require_equal(sealed.get("npz_sha256"), EXPECTED_RAW_NPZ_SHA256, "Top-K raw NPZ")
    _require_equal(
        sealed.get("candidate_content_sha256"),
        EXPECTED_RAW_CONTENT_SHA256,
        "Top-K raw content",
    )
    budget = receipt.get("budgets", {}).get("8")
    if not isinstance(budget, Mapping):
        raise S3R0OracleError("Top-K K8 budget receipt is missing")
    _require_equal(budget.get("top_k_per_frame"), 8, "Top-K K8 budget")
    _require_equal(
        budget.get("selection_sha256"),
        EXPECTED_K8_SELECTION_SHA256,
        "Top-K K8 selection",
    )
    _require_equal(
        budget.get("candidate_count"), EXPECTED_SELECTED_COUNT, "Top-K K8 count"
    )
    _require_equal(
        budget.get("candidate_count_by_scene"),
        EXPECTED_SELECTED_COUNTS,
        "Top-K K8 scene counts",
    )
    ledger = receipt.get("input_sha256_before")
    if not isinstance(ledger, Mapping):
        raise S3R0OracleError("Top-K input ledger is missing")
    _require_equal(
        ledger.get("shadow_json"), EXPECTED_RAW_JSON_SHA256, "Top-K raw JSON input"
    )
    _require_equal(
        ledger.get("shadow_npz"), EXPECTED_RAW_NPZ_SHA256, "Top-K raw NPZ input"
    )
    scene_ledger = ledger.get("scenes")
    if not isinstance(scene_ledger, Mapping) or set(scene_ledger) != set(DEV3_SCENES):
        raise S3R0OracleError("Top-K scene input ledger is invalid")
    for scene in DEV3_SCENES:
        row = scene_ledger[scene]
        if not isinstance(row, Mapping):
            raise S3R0OracleError(f"Top-K scene ledger is invalid for {scene}")
        _require_equal(
            row.get("baseline"), EXPECTED_NATIVE_SHA256[scene], f"Top-K native {scene}"
        )
        _require_equal(row.get("gt"), EXPECTED_GT_SHA256[scene], f"Top-K GT {scene}")
        _require_equal(
            row.get("axis_alignment"),
            EXPECTED_AXIS_SHA256[scene],
            f"Top-K axis {scene}",
        )
    return receipt, digest


def _validate_shadow_input_ledger(manifest: Mapping[str, Any]) -> dict[str, str]:
    before = manifest.get("input_sha256_before")
    after = manifest.get("input_sha256_after")
    if not isinstance(before, Mapping) or not isinstance(after, Mapping):
        raise S3R0OracleError("S3R frozen input ledgers are missing")
    _require_equal(before, after, "S3R input before/after ledger")
    _require_equal(manifest.get("input_hash_identity"), True, "S3R input identity flag")
    _require_equal(
        before.get("allowed_numeric_source_content_sha256"),
        EXPECTED_ALLOWED_SOURCE_CONTENT_SHA256,
        "S3R allowed source content",
    )
    fixed = before.get("fixed_assets")
    if not isinstance(fixed, Mapping):
        raise S3R0OracleError("S3R fixed-assets ledger is missing")
    expected_fixed = {
        "preregistration": (PREREGISTRATION, EXPECTED_PREREGISTRATION_SHA256),
        "sealed_json": (RAW_JSON, EXPECTED_RAW_JSON_SHA256),
        "sealed_npz": (RAW_NPZ, EXPECTED_RAW_NPZ_SHA256),
        "topk_receipt": (TOPK_RECEIPT, EXPECTED_TOPK_RECEIPT_SHA256),
        "topk_tool": (TOPK_TOOL, EXPECTED_TOPK_TOOL_SHA256),
        "tracker_source": (TRACKER_SOURCE, EXPECTED_TRACKER_SOURCE_SHA256),
        "tracker_test": (TRACKER_TEST, EXPECTED_TRACKER_TEST_SHA256),
    }
    if set(fixed) != set(expected_fixed):
        raise S3R0OracleError("S3R fixed-assets ledger key set mismatch")
    hashes = {
        key: _ledger_entry(fixed, key, path, digest)
        for key, (path, digest) in expected_fixed.items()
    }
    hashes["runner_source"] = _ledger_entry(
        before, "runner_source", RUNNER_SOURCE, EXPECTED_RUNNER_SOURCE_SHA256
    )
    hashes["runner_test"] = _ledger_entry(
        before, "runner_test", RUNNER_TEST, EXPECTED_RUNNER_TEST_SHA256
    )
    schedules = before.get("schedules")
    if not isinstance(schedules, Mapping) or set(schedules) != set(DEV3_SCENES):
        raise S3R0OracleError("S3R schedule ledger is invalid")
    for scene in DEV3_SCENES:
        entry = schedules[scene]
        if not isinstance(entry, Mapping):
            raise S3R0OracleError(f"S3R schedule ledger is invalid for {scene}")
        path = _regular_file(
            SCHEDULE_ROOT / scene / "manifest.json", f"schedule {scene}"
        )
        _require_equal(entry.get("path"), os.fspath(path), f"schedule path {scene}")
        digest = _sha256(path)
        _require_equal(
            entry.get("sha256"),
            EXPECTED_SCHEDULE_SHA256[scene],
            f"schedule ledger {scene}",
        )
        _require_equal(
            digest, EXPECTED_SCHEDULE_SHA256[scene], f"schedule actual {scene}"
        )
        hashes[f"schedule:{scene}"] = digest
    native = before.get("native_t05")
    if not isinstance(native, Mapping) or set(native) != set(DEV3_SCENES):
        raise S3R0OracleError("S3R native ledger is invalid")
    for scene in DEV3_SCENES:
        hashes[f"native:{scene}"] = _ledger_entry(
            native,
            scene,
            BASELINE_ROOT / f"{scene}_boxes.pkl",
            EXPECTED_NATIVE_SHA256[scene],
        )
    return hashes


def _load_s3r_sidecar(
    *, raw_arrays: Mapping[str, np.ndarray], selections: Sequence[np.ndarray]
) -> tuple[dict[str, Any], dict[str, np.ndarray], dict[str, str]]:
    json_path = _regular_file(SHADOW_JSON, "sealed S3R JSON")
    npz_path = _regular_file(SHADOW_NPZ, "sealed S3R NPZ")
    hashes = {"json": _sha256(json_path), "npz": _sha256(npz_path)}
    _require_equal(hashes["json"], EXPECTED_SHADOW_JSON_SHA256, "S3R JSON SHA-256")
    _require_equal(hashes["npz"], EXPECTED_SHADOW_NPZ_SHA256, "S3R NPZ SHA-256")
    manifest = _read_json(json_path, "sealed S3R JSON")
    required = {
        "schema": SHADOW_SCHEMA,
        "mode": "shadow",
        "output_inert": True,
        "receipt_only": True,
        "birth": False,
        "active_authorized": False,
        "native_mutation_applied": False,
        "ap_evaluation": False,
        "gt_access": False,
        "oracle_access": False,
        "semantics_access": False,
        "labels_loaded": False,
        "labels_exported": False,
        "clip_access": False,
        "depth_access": False,
        "rgb_access": False,
        "native_prediction_deserialized": False,
        "native_hash_only_access": True,
        "training": False,
        "online_learning": False,
        "optimizer_access": False,
        "past_only": True,
        "query_before_commit": True,
        "same_frame_confirmation": False,
        "within_frame_deduplication": False,
        "H10_not_authorized": True,
        "C87_not_authorized": True,
        "full100_not_authorized": True,
        "not_deployable": True,
        "audit_complete": True,
        "dev3_complete": True,
        "scene_order": list(DEV3_SCENES),
        "scene_count": len(DEV3_SCENES),
        "valid_frame_count": EXPECTED_VALID_FRAME_COUNT,
        "selected_row_count": EXPECTED_SELECTED_COUNT,
        "assignment_count": EXPECTED_SELECTED_COUNT,
        "receipt_count": EXPECTED_RECEIPT_COUNT,
        "evidence_count": EXPECTED_EVIDENCE_COUNT,
        "cap_event_count": 0,
        "npz_file": SHADOW_NPZ.name,
        "npz_sha256": EXPECTED_SHADOW_NPZ_SHA256,
        "candidate_content_sha256": EXPECTED_SHADOW_CONTENT_SHA256,
        "formal_t05_root": os.fspath(BASELINE_ROOT.resolve()),
        "formal_t05_expected_sha256": EXPECTED_NATIVE_SHA256,
        "native_prediction_sha256_before": EXPECTED_NATIVE_SHA256,
        "native_prediction_sha256_after": EXPECTED_NATIVE_SHA256,
        "native_prediction_hash_identity": True,
    }
    for key, expected in required.items():
        _require_equal(manifest.get(key), expected, f"S3R manifest {key}")
    selection = manifest.get("selection")
    if not isinstance(selection, Mapping):
        raise S3R0OracleError("S3R selection ledger is missing")
    selection_required = {
        "top_k_per_valid_candidate_frame": 8,
        "rule": "descending_source_score_then_ascending_source_row_then_ascending_sealed_npz_row",
        "selection_sha256": EXPECTED_K8_SELECTION_SHA256,
        "expected_selection_sha256": EXPECTED_K8_SELECTION_SHA256,
        "selected_count_by_scene": EXPECTED_SELECTED_COUNTS,
        "candidate_frame_count_by_scene": EXPECTED_CANDIDATE_FRAME_COUNTS,
        "complete_exact_k8_membership": True,
        "selection_used_gt": False,
        "selection_used_semantics": False,
        "selection_used_only_frozen_source_score": True,
    }
    for key, expected in selection_required.items():
        _require_equal(selection.get(key), expected, f"S3R selection {key}")
    contracts = manifest.get("contracts")
    if not isinstance(contracts, Mapping):
        raise S3R0OracleError("S3R contract ledger is missing")
    contract_required = {
        "preregistration_sha256": EXPECTED_PREREGISTRATION_SHA256,
        "tracker_source_sha256": EXPECTED_TRACKER_SOURCE_SHA256,
        "tracker_test_sha256": EXPECTED_TRACKER_TEST_SHA256,
        "sealed_boxer_json_sha256": EXPECTED_RAW_JSON_SHA256,
        "sealed_boxer_npz_sha256": EXPECTED_RAW_NPZ_SHA256,
        "sealed_boxer_candidate_content_sha256": EXPECTED_RAW_CONTENT_SHA256,
        "topk_receipt_sha256": EXPECTED_TOPK_RECEIPT_SHA256,
        "topk_tool_sha256": EXPECTED_TOPK_TOOL_SHA256,
        "schedule_sha256_by_scene": EXPECTED_SCHEDULE_SHA256,
    }
    for key, expected in contract_required.items():
        _require_equal(contracts.get(key), expected, f"S3R contract {key}")
    runtime = manifest.get("runtime")
    if not isinstance(runtime, Mapping):
        raise S3R0OracleError("S3R runtime ledger is missing")
    for key in (
        "tracker_cpu_budget_pass",
        "tracker_memory_upper_bound_pass",
        "resource_budget_pass",
    ):
        _require_equal(runtime.get(key), True, f"S3R runtime {key}")
    ledger_hashes = _validate_shadow_input_ledger(manifest)
    try:
        with np.load(npz_path, allow_pickle=False) as source:
            arrays = {name: np.array(source[name], copy=True) for name in source.files}
    except (OSError, ValueError) as error:
        raise S3R0OracleError(f"invalid sealed S3R NPZ: {npz_path}") from error
    if _sha256(npz_path) != hashes["npz"]:
        raise S3R0OracleError("S3R NPZ changed while loading")
    _require_equal(
        _array_content_sha256(arrays),
        EXPECTED_SHADOW_CONTENT_SHA256,
        "S3R candidate content SHA-256",
    )
    _validate_s3r_arrays(
        manifest=manifest, arrays=arrays, raw_arrays=raw_arrays, selections=selections
    )
    for array in arrays.values():
        array.setflags(write=False)
    return manifest, arrays, {**hashes, **ledger_hashes}


@dataclass(frozen=True)
class _FrozenContext:
    shadow_manifest: Mapping[str, Any]
    shadow_arrays: Mapping[str, np.ndarray]
    raw_manifest: Mapping[str, Any]
    raw_arrays: Mapping[str, np.ndarray]
    selections: tuple[np.ndarray, ...]
    topk_receipt: Mapping[str, Any]
    baseline_paths: Mapping[str, Path]
    no_gt_hashes_before: Mapping[str, Any]


def _snapshot_no_gt_hashes(
    manifest: Mapping[str, Any], baseline_paths: Mapping[str, Path]
) -> dict[str, Any]:
    oracle_source = _regular_file(ORACLE_SOURCE, "S3R0 oracle source")
    oracle_test = _regular_file(ORACLE_TEST, "S3R0 oracle tests")
    geometry_helpers = _regular_file(GEOMETRY_HELPERS_SOURCE, "geometry helper source")
    geometry_digest = _sha256(geometry_helpers)
    _require_equal(
        geometry_digest,
        EXPECTED_GEOMETRY_HELPERS_SHA256,
        "geometry helper source SHA-256",
    )
    return {
        "shadow_json": _sha256(_regular_file(SHADOW_JSON, "sealed S3R JSON")),
        "shadow_npz": _sha256(_regular_file(SHADOW_NPZ, "sealed S3R NPZ")),
        "raw_json": _sha256(_regular_file(RAW_JSON, "sealed raw Boxer JSON")),
        "raw_npz": _sha256(_regular_file(RAW_NPZ, "sealed raw Boxer NPZ")),
        "topk_receipt": _sha256(
            _regular_file(TOPK_RECEIPT, "frozen Top-K ceiling receipt")
        ),
        "preregistration": _sha256(
            _regular_file(PREREGISTRATION, "S3R preregistration")
        ),
        "oracle_source": {
            "path": os.fspath(oracle_source),
            "sha256": _sha256(oracle_source),
            "bytes": oracle_source.stat().st_size,
        },
        "oracle_test": {
            "path": os.fspath(oracle_test),
            "sha256": _sha256(oracle_test),
            "bytes": oracle_test.stat().st_size,
        },
        "geometry_helpers": {
            "path": os.fspath(geometry_helpers),
            "sha256": geometry_digest,
            "bytes": geometry_helpers.stat().st_size,
        },
        "baseline": {
            scene: _sha256(_regular_file(path, f"formal T05 prediction {scene}"))
            for scene, path in baseline_paths.items()
        },
        "shadow_bound_assets": {
            "json": _sha256(_regular_file(SHADOW_JSON, "sealed S3R JSON")),
            "npz": _sha256(_regular_file(SHADOW_NPZ, "sealed S3R NPZ")),
            **_validate_shadow_input_ledger(manifest),
        },
    }


def _validate_no_gt_frozen() -> _FrozenContext:
    raw_json = _regular_file(RAW_JSON, "sealed raw Boxer JSON")
    raw_npz = _regular_file(RAW_NPZ, "sealed raw Boxer NPZ")
    _require_equal(
        _sha256(raw_json), EXPECTED_RAW_JSON_SHA256, "raw Boxer JSON SHA-256"
    )
    _require_equal(_sha256(raw_npz), EXPECTED_RAW_NPZ_SHA256, "raw Boxer NPZ SHA-256")
    try:
        raw_manifest, raw_arrays, scenes, raw_hashes = _load_sealed_sidecar(
            raw_json, raw_npz
        )
    except Exception as error:
        raise S3R0OracleError(f"invalid sealed raw Boxer sidecar: {error}") from error
    _require_equal(tuple(scenes), DEV3_SCENES, "raw Boxer scene order")
    _require_equal(raw_hashes.get("json"), EXPECTED_RAW_JSON_SHA256, "raw loader JSON")
    _require_equal(raw_hashes.get("npz"), EXPECTED_RAW_NPZ_SHA256, "raw loader NPZ")
    _require_equal(
        raw_manifest.get("candidate_content_sha256"),
        EXPECTED_RAW_CONTENT_SHA256,
        "raw Boxer content SHA-256",
    )
    selections_by_budget = _select_per_frame_topk(raw_arrays, scenes)
    selections = tuple(selections_by_budget[8])
    _require_equal(
        _selection_sha256(selections),
        EXPECTED_K8_SELECTION_SHA256,
        "recomputed exact K8 selection",
    )
    _require_equal(
        {scene: len(rows) for scene, rows in zip(DEV3_SCENES, selections)},
        EXPECTED_SELECTED_COUNTS,
        "recomputed K8 scene counts",
    )
    for array in raw_arrays.values():
        array.setflags(write=False)
    topk_receipt, topk_hash = _load_topk_receipt()
    shadow_manifest, shadow_arrays, shadow_hashes = _load_s3r_sidecar(
        raw_arrays=raw_arrays, selections=selections
    )
    baseline_paths: dict[str, Path] = {}
    baseline_hashes: dict[str, str] = {}
    for scene in DEV3_SCENES:
        path = _regular_file(
            BASELINE_ROOT / f"{scene}_boxes.pkl", f"formal T05 prediction {scene}"
        )
        digest = _sha256(path)
        _require_equal(digest, EXPECTED_NATIVE_SHA256[scene], f"formal T05 {scene}")
        baseline_paths[scene] = path
        baseline_hashes[scene] = digest
    no_gt_hashes = _snapshot_no_gt_hashes(shadow_manifest, baseline_paths)
    _require_equal(no_gt_hashes["topk_receipt"], topk_hash, "Top-K receipt snapshot")
    _require_equal(no_gt_hashes["baseline"], baseline_hashes, "formal T05 snapshot")
    _require_equal(
        no_gt_hashes["shadow_bound_assets"], shadow_hashes, "S3R bound-asset snapshot"
    )
    return _FrozenContext(
        shadow_manifest=shadow_manifest,
        shadow_arrays=shadow_arrays,
        raw_manifest=raw_manifest,
        raw_arrays=raw_arrays,
        selections=selections,
        topk_receipt=topk_receipt,
        baseline_paths=baseline_paths,
        no_gt_hashes_before=no_gt_hashes,
    )


def audit_scannet_s3r0_raw_boxer_receipt_oracle() -> dict[str, Any]:
    """Validate the frozen artifact, then compute the fixed dev3 diagnostic."""

    # Hard barrier: every source, membership, receipt, cap, runtime, and native
    # identity check completes before the first GT/axis path is resolved.
    context = _validate_no_gt_frozen()
    arrays = context.shadow_arrays

    raw_iou: list[np.ndarray] = []
    medoid_iou: list[np.ndarray] = []
    any_evidence_iou: list[np.ndarray] = []
    native_iou: list[np.ndarray] = []
    gt_counts: dict[str, int] = {}
    matrix_shapes: dict[str, dict[str, list[int]]] = {
        "raw_k8": {},
        "receipt_medoid": {},
        "track_any_evidence": {},
    }
    oracle_hashes_before: dict[str, dict[str, str]] = {}

    for scene_index, scene in enumerate(DEV3_SCENES):
        # These two local regular-file checks are the first GT/axis accesses.
        gt_path = _regular_file(GT_ROOT / f"{scene}_bbox.npy", f"ScanNet GT {scene}")
        axis_path = _regular_file(
            SCAN_ROOT / scene / f"{scene}.txt", f"ScanNet axis alignment {scene}"
        )
        gt_digest = _sha256(gt_path)
        axis_digest = _sha256(axis_path)
        _require_equal(gt_digest, EXPECTED_GT_SHA256[scene], f"ScanNet GT hash {scene}")
        _require_equal(
            axis_digest, EXPECTED_AXIS_SHA256[scene], f"ScanNet axis hash {scene}"
        )
        oracle_hashes_before[scene] = {
            "gt": gt_digest,
            "axis_alignment": axis_digest,
        }
        alignment = load_axis_alignment(axis_path)
        gt = load_gt_minmax(gt_path)
        _, native_aligned = load_baseline_boxes(
            context.baseline_paths[scene], alignment
        )
        native_matrix = aligned_iou_matrix(native_aligned, gt)
        native_iou.append(native_matrix)
        gt_counts[scene] = int(len(gt))

        selected_start = int(arrays["scene_selected_offsets"][scene_index])
        selected_end = int(arrays["scene_selected_offsets"][scene_index + 1])
        raw_aligned = _aligned_enclosing_aabb(
            arrays["selected_corners_world"][selected_start:selected_end], alignment
        )
        raw_matrix = aligned_iou_matrix(raw_aligned, gt)
        raw_iou.append(raw_matrix)

        receipt_start = int(arrays["scene_receipt_offsets"][scene_index])
        receipt_end = int(arrays["scene_receipt_offsets"][scene_index + 1])
        medoid_aligned = _aligned_enclosing_aabb(
            arrays["receipt_corners_world"][receipt_start:receipt_end], alignment
        )
        medoid_matrix = aligned_iou_matrix(medoid_aligned, gt)
        medoid_iou.append(medoid_matrix)

        evidence_start = int(arrays["evidence_offsets"][receipt_start])
        evidence_end = int(arrays["evidence_offsets"][receipt_end])
        evidence_aligned = _aligned_enclosing_aabb(
            arrays["evidence_corners_world"][evidence_start:evidence_end], alignment
        )
        evidence_matrix = aligned_iou_matrix(evidence_aligned, gt)
        local_offsets = (
            arrays["evidence_offsets"][receipt_start : receipt_end + 1] - evidence_start
        )
        any_matrix = _track_any_evidence_iou(evidence_matrix, local_offsets)
        any_evidence_iou.append(any_matrix)

        matrix_shapes["raw_k8"][scene] = list(raw_matrix.shape)
        matrix_shapes["receipt_medoid"][scene] = list(medoid_matrix.shape)
        matrix_shapes["track_any_evidence"][scene] = list(any_matrix.shape)

    geometries = {
        "raw_k8": {
            "source": "exact_frozen_score_only_k8_raw_boxer_obbs",
            "candidate_identity": "selected_k8_row",
            "oracle_only": True,
            "matrix_shape_by_scene": matrix_shapes["raw_k8"],
            "per_threshold": _matching_report(
                scenes=DEV3_SCENES,
                candidate_iou=raw_iou,
                native_iou=native_iou,
            ),
        },
        "receipt_medoid": {
            "source": "immutable_first_three_aabb_iou_medoid_raw_obb",
            "candidate_identity": "scene_qualified_receipt",
            "single_frozen_geometry_per_receipt": True,
            "oracle_only": True,
            "deployed": False,
            "matrix_shape_by_scene": matrix_shapes["receipt_medoid"],
            "per_threshold": _matching_report(
                scenes=DEV3_SCENES,
                candidate_iou=medoid_iou,
                native_iou=native_iou,
            ),
        },
        "track_any_evidence": {
            "source": "elementwise_max_iou_over_exactly_three_frozen_evidence_obbs",
            "candidate_identity": "scene_qualified_receipt",
            "oracle_only": True,
            "deployable_geometry": False,
            "evidence_argmax_computed": False,
            "evidence_argmax_exported": False,
            "matrix_shape_by_scene": matrix_shapes["track_any_evidence"],
            "per_threshold": _matching_report(
                scenes=DEV3_SCENES,
                candidate_iou=any_evidence_iou,
                native_iou=native_iou,
            ),
        },
    }
    continuation_gate = _continuation_gate(
        geometries["receipt_medoid"]["per_threshold"],
        geometries["track_any_evidence"]["per_threshold"],
    )

    no_gt_hashes_after = _snapshot_no_gt_hashes(
        context.shadow_manifest, context.baseline_paths
    )
    oracle_hashes_after = {
        scene: {
            "gt": _sha256(
                _regular_file(GT_ROOT / f"{scene}_bbox.npy", f"ScanNet GT {scene}")
            ),
            "axis_alignment": _sha256(
                _regular_file(
                    SCAN_ROOT / scene / f"{scene}.txt",
                    f"ScanNet axis alignment {scene}",
                )
            ),
        }
        for scene in DEV3_SCENES
    }
    if no_gt_hashes_after != context.no_gt_hashes_before:
        raise S3R0OracleError("one or more no-GT frozen inputs changed during oracle")
    if oracle_hashes_after != oracle_hashes_before:
        raise S3R0OracleError("one or more GT/axis inputs changed during oracle")
    _require_equal(sum(gt_counts.values()), EXPECTED_GT_COUNT, "formal dev3 GT count")

    return {
        "schema": SCHEMA,
        "posthoc_dev_diagnostic": True,
        "development_informed": True,
        "validation_claimed": False,
        "deployable": False,
        "not_deployable": True,
        "output_inert": True,
        "receipt_membership_mutated": False,
        "native_mutation_applied": False,
        "birth": False,
        "active_birth_authorized": False,
        "prediction_suffix_created": False,
        "ap_computed": False,
        "active_suffix_ap_computed": False,
        "threshold_tuning_performed": False,
        "posthoc_gt_informed_candidate_selection_applied": False,
        "posthoc_gt_informed_candidate_ranking_applied": False,
        "frozen_pre_gt_k8_membership_revalidated": True,
        "candidate_suppression_applied": False,
        "gt_geometry_accessed": True,
        "gt_semantic_columns_consumed": False,
        "labels_or_clip_consumed": False,
        "H10_authorized": False,
        "H10_not_authorized": True,
        "h10_gt_accessed": False,
        "C87_authorized": False,
        "C87_not_authorized": True,
        "full100_authorized": False,
        "full100_not_authorized": True,
        "full100_accessed": False,
        "strict_iou_comparison": ">",
        "thresholds": list(THRESHOLDS),
        "scene_order": list(DEV3_SCENES),
        "gt_count": int(sum(gt_counts.values())),
        "gt_count_by_scene": gt_counts,
        "raw_k8_count": EXPECTED_SELECTED_COUNT,
        "receipt_count": EXPECTED_RECEIPT_COUNT,
        "evidence_count": EXPECTED_EVIDENCE_COUNT,
        "geometries": geometries,
        "continuation_gate": continuation_gate,
        "input_sha256_before": {
            "no_gt": context.no_gt_hashes_before,
            "oracle_only": oracle_hashes_before,
        },
        "input_sha256_after": {
            "no_gt": no_gt_hashes_after,
            "oracle_only": oracle_hashes_after,
        },
        "input_hash_identity": True,
        "native_prediction_hash_identity": True,
        "native_prediction_deserialized_for_posthoc_geometry": True,
        "sealed_s3r_shadow": {
            "json_path": os.fspath(SHADOW_JSON.resolve()),
            "json_sha256": EXPECTED_SHADOW_JSON_SHA256,
            "npz_path": os.fspath(SHADOW_NPZ.resolve()),
            "npz_sha256": EXPECTED_SHADOW_NPZ_SHA256,
            "candidate_content_sha256": EXPECTED_SHADOW_CONTENT_SHA256,
            "schema": SHADOW_SCHEMA,
            "selection_sha256": EXPECTED_K8_SELECTION_SHA256,
        },
        "conclusion_guardrail": (
            "Matching-only dev3 engineering diagnostic. A passing gate can only "
            "permit the already-preregistered one-shot H10 shadow sequence; it "
            "cannot authorize birth, a prediction suffix, AP, C87, or full100."
        ),
    }


def _preflight_output_lexical(path: Path) -> Path:
    """Reject obvious output conflicts without resolving any GT/axis path."""

    output = Path(os.path.abspath(os.fspath(path)))
    current = Path(output.anchor)
    for part in output.parent.parts[1:]:
        current /= part
        try:
            component_stat = os.lstat(current)
        except FileNotFoundError:
            break
        except OSError as error:
            raise S3R0OracleError(
                f"could not inspect output parent component: {current}"
            ) from error
        if stat.S_ISLNK(component_stat.st_mode):
            raise S3R0OracleError(
                f"output parent must not be a symlink or contain one: {current}"
            )
    protected_roots = tuple(
        Path(os.path.abspath(os.fspath(root)))
        for root in (
            SHADOW_ROOT,
            RAW_ROOT,
            SCHEDULE_ROOT,
            BASELINE_ROOT,
            GT_ROOT,
            SCAN_ROOT,
        )
    )
    if any(output == root or root in output.parents for root in protected_roots):
        raise S3R0OracleError("output must be outside every protected input root")
    protected_files = {
        Path(os.path.abspath(os.fspath(source)))
        for source in (
            SHADOW_JSON,
            SHADOW_NPZ,
            RAW_JSON,
            RAW_NPZ,
            TOPK_RECEIPT,
            PREREGISTRATION,
            TOPK_TOOL,
            GEOMETRY_HELPERS_SOURCE,
            TRACKER_SOURCE,
            TRACKER_TEST,
            RUNNER_SOURCE,
            RUNNER_TEST,
            ORACLE_SOURCE,
            ORACLE_TEST,
        )
    }
    if output in protected_files:
        raise S3R0OracleError("output must differ from every protected input file")
    try:
        os.lstat(output)
    except FileNotFoundError:
        pass
    except OSError as error:
        raise S3R0OracleError(f"could not inspect output: {output}") from error
    else:
        raise S3R0OracleError(f"refusing to overwrite output: {output}")
    return output


def _preflight_output(path: Path) -> Path:
    output = _preflight_output_lexical(path)
    resolved_parent = output.parent.resolve()
    resolved_output = resolved_parent / output.name
    return resolved_output


def _write_json_create_only(path: Path, payload: Mapping[str, Any]) -> None:
    resolved_output = _preflight_output(path)
    resolved_parent = resolved_output.parent
    resolved_parent.mkdir(parents=True, exist_ok=True)
    try:
        encoded = (
            json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError, UnicodeEncodeError) as error:
        raise S3R0OracleError("could not serialize oracle report") from error

    temporary_path: Path | None = None
    try:
        descriptor, raw_temporary_path = tempfile.mkstemp(
            dir=resolved_parent,
            prefix=f".{resolved_output.name}.",
            suffix=".tmp",
        )
        temporary_path = Path(raw_temporary_path)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary_path, resolved_output, follow_symlinks=False)
    except FileExistsError as error:
        raise S3R0OracleError(
            f"refusing to overwrite output: {resolved_output}"
        ) from error
    except OSError as error:
        raise S3R0OracleError(
            f"could not publish oracle report: {resolved_output}"
        ) from error
    finally:
        if temporary_path is not None:
            try:
                temporary_path.unlink()
            except FileNotFoundError:
                pass

    directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    try:
        directory_descriptor = os.open(resolved_parent, directory_flags)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    except OSError as error:
        raise S3R0OracleError(
            f"could not seal oracle report directory: {resolved_parent}"
        ) from error


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    _preflight_output_lexical(args.output)
    report = audit_scannet_s3r0_raw_boxer_receipt_oracle()
    _write_json_create_only(args.output, report)
    print(
        json.dumps(
            {
                "schema": SCHEMA,
                "H10_authorized": False,
                "full100_authorized": False,
                "output": os.fspath(args.output),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
