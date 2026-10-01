#!/usr/bin/env python3
"""Materialize the frozen target-first MobileSAM mask-lift birth policy.

This program is deliberately a terminal, no-GT materializer.  It consumes a
sealed shadow sidecar containing only causal three-view receipts and fused
geometry, applies one fixed admission/novelty/NMS policy, and appends accepted
boxes to an unchanged native Cbest prefix.  It has no annotation or evaluator
argument and refuses to overwrite an output root.

``--plan-only`` performs the same validation and selection but creates no
files.  This makes it possible to freeze and inspect the active population
before any AP evaluation is run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import sys
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if os.fspath(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, os.fspath(REPOSITORY_ROOT))

from tools.materialize_scannet_raw_boxer_past3_birth_full100 import (  # noqa: E402
    BirthMaterializationError,
    NativePrediction,
    _aabb_overlap_matrices,
    _assert_native_prefix,
    _load_native_prediction,
    _regular_file,
    _scene_list,
    _sha256,
    _write_json,
    _write_pickle,
)


SCHEMA = "boxfusion.scannet_target_first_mobilesam_birth_full100.v1"
MANIFEST_NAME = "TARGET_FIRST_MOBILESAM_BIRTH_FULL100.json"
PREDICTION_SUFFIX = "_boxes.pkl"

# Frozen R15 active policy.  Every threshold is fixed before AP access.
MIN_RAW_MEAN_SCORE = 0.50
MIN_EVIDENCE_SCORE = 0.40
MIN_MEDIAN_PAIRWISE_MASK_AABB_IOU = 0.15
MAX_PAIRWISE_MASK_CENTER_DISTANCE_M = 0.50
MIN_FIRST_LAST_FRAME_SPAN = 50
MIN_CAMERA_BASELINE_M = 0.10
MIN_VIEW_RAY_SPAN_DEG = 5.0
MIN_FUSED_SUPPORTED_VOXELS = 24
MIN_SUPPORTED_VOXELS_PER_VIEW = 8
MIN_FUSED_OBB_EXTENT_M = 0.05
MAX_FUSED_CENTER_TO_RAW_MEDOID_M = 0.75
NATIVE_NOVELTY_AABB_IOU = 0.10
NATIVE_MAX_BIDIRECTIONAL_CONTAINMENT = 0.50
SELF_NMS_AABB_IOU = 0.15
SELF_NMS_BIDIRECTIONAL_CONTAINMENT = 0.25
MAX_BIRTHS_PER_SCENE = 4
APPENDED_CLASS_ID = 0
APPENDED_SCORE = 1.0


@dataclass(frozen=True)
class MaskLiftReceipt:
    scene: str
    track_id: int
    confirmation_frame_id: int
    evidence_frame_ids: tuple[int, int, int]
    evidence_source_rows: tuple[int, int, int]
    evidence_scores: tuple[float, float, float]
    min_evidence_score: float
    raw_mean_score: float
    median_pairwise_mask_aabb_iou: float
    max_pairwise_mask_center_distance_m: float
    first_last_frame_span: int
    max_camera_baseline_m: float
    max_view_ray_span_deg: float
    supported_voxel_count: int
    view_supported_voxel_counts: tuple[int, int, int]
    fused_obb_extent_xyz: tuple[float, float, float]
    fused_center_to_raw_medoid_m: float
    corners: np.ndarray
    target_group: str | None


@dataclass(frozen=True)
class MaskLiftSidecar:
    path: Path
    sha256: str
    schema: str
    receipts: dict[str, tuple[MaskLiftReceipt, ...]]
    receipt_count: int
    npz_path: Path | None
    npz_sha256: str | None
    npz_array_sha256: dict[str, str]


def _strict_int(value: object, label: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise BirthMaterializationError(f"{label} must be an integer")
    result = int(value)
    if result < 0:
        raise BirthMaterializationError(f"{label} must be nonnegative")
    return result


def _strict_float(value: object, label: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise BirthMaterializationError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise BirthMaterializationError(f"{label} must be finite")
    return result


def _one_of(source: Mapping[str, Any], names: Sequence[str], label: str) -> Any:
    present = [name for name in names if name in source]
    if len(present) != 1:
        raise BirthMaterializationError(
            f"{label} requires exactly one of {list(names)}, found {present}"
        )
    return source[present[0]]


def _nested_geometry(record: Mapping[str, Any]) -> Mapping[str, Any]:
    present = [
        key
        for key in (
            "fused_same_yaw_robust_obb",
            "fused_obb",
            "fused_geometry",
        )
        if key in record
    ]
    if len(present) > 1:
        raise BirthMaterializationError(
            f"receipt has ambiguous fused geometry objects: {present}"
        )
    if not present:
        return record
    value = record[present[0]]
    if not isinstance(value, dict):
        raise BirthMaterializationError(f"{present[0]} must be an object")
    return value


def _triple_ints(value: object, label: str) -> tuple[int, int, int]:
    if not isinstance(value, list) or len(value) != 3:
        raise BirthMaterializationError(f"{label} must be a three-element list")
    return tuple(_strict_int(row, f"{label}[{index}]") for index, row in enumerate(value))  # type: ignore[return-value]


def _extent_xyz(record: Mapping[str, Any], geometry: Mapping[str, Any]) -> tuple[float, float, float]:
    source: object | None = None
    owners = (geometry,) if geometry is record else (geometry, record)
    for owner in owners:
        for key in ("extent_xyz", "fused_obb_extent_xyz", "extent"):
            if key in owner:
                if source is not None:
                    raise BirthMaterializationError("ambiguous fused OBB extent fields")
                source = owner[key]
    if source is None and "fused_min_obb_extent_m" in record:
        scalar = _strict_float(
            record["fused_min_obb_extent_m"], "fused_min_obb_extent_m"
        )
        source = [scalar, scalar, scalar]
    if not isinstance(source, list) or len(source) != 3:
        raise BirthMaterializationError("receipt requires fused OBB extent_xyz[3]")
    result = tuple(_strict_float(value, f"fused_obb_extent_xyz[{index}]") for index, value in enumerate(source))
    if min(result) < 0.0:
        raise BirthMaterializationError("fused OBB extents must be nonnegative")
    if "fused_min_obb_extent_m" in record:
        declared_min = _strict_float(
            record["fused_min_obb_extent_m"], "fused_min_obb_extent_m"
        )
        if not math.isclose(declared_min, min(result), rel_tol=1e-6, abs_tol=1e-7):
            raise BirthMaterializationError("fused minimum extent disagrees with extent_xyz")
    return result  # type: ignore[return-value]


def _corners(record: Mapping[str, Any], geometry: Mapping[str, Any]) -> np.ndarray:
    fields: list[object] = []
    owners = (geometry,) if geometry is record else (geometry, record)
    for owner in owners:
        for key in ("corners_world", "fused_corners_world"):
            if key in owner:
                fields.append(owner[key])
    if len(fields) == 2:
        try:
            left = np.asarray(fields[0], dtype=np.float64)
            right = np.asarray(fields[1], dtype=np.float64)
        except (TypeError, ValueError) as error:
            raise BirthMaterializationError("invalid duplicate fused corners") from error
        if left.shape != (8, 3) or right.shape != (8, 3) or not np.array_equal(left, right):
            raise BirthMaterializationError("duplicate fused corner fields disagree")
        fields = [fields[0]]
    if len(fields) != 1:
        raise BirthMaterializationError(
            "receipt requires exactly one fused corners_world/fused_corners_world field"
        )
    try:
        corners = np.asarray(fields[0], dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise BirthMaterializationError("invalid fused corners") from error
    if corners.shape != (8, 3) or not np.isfinite(corners).all():
        raise BirthMaterializationError("fused corners must be finite [8,3]")
    # Rejected shadow receipts may carry the runner's finite all-zero sentinel
    # when cross-view voxel consensus is empty.  Such a row is retained for a
    # complete audit and is rejected by the voxel/extent active gates before it
    # could ever become a suffix.
    return np.ascontiguousarray(corners)


def _parse_receipt(scene: str, record: object, index: int) -> MaskLiftReceipt:
    if not isinstance(record, dict):
        raise BirthMaterializationError(f"{scene} receipt {index} must be an object")
    prefix = f"{scene} receipt {index}"
    if "scene" in record and record["scene"] != scene:
        raise BirthMaterializationError(f"{prefix} carries a different scene")
    frames = _triple_ints(record.get("evidence_frame_ids"), f"{prefix} evidence_frame_ids")
    rows = _triple_ints(record.get("evidence_source_rows"), f"{prefix} evidence_source_rows")
    if not frames[0] < frames[1] < frames[2]:
        raise BirthMaterializationError(f"{prefix} evidence frames must be strictly increasing")
    if len(set(rows)) != 3:
        raise BirthMaterializationError(f"{prefix} source rows must be distinct")
    confirmation = _strict_int(record.get("confirmation_frame_id"), f"{prefix} confirmation_frame_id")
    if confirmation != frames[-1]:
        raise BirthMaterializationError(f"{prefix} confirmation must be the third causal view")
    frame_span = _strict_int(record.get("first_last_frame_span"), f"{prefix} first_last_frame_span")
    if frame_span != frames[-1] - frames[0]:
        raise BirthMaterializationError(f"{prefix} frame span disagrees with evidence")
    geometry = _nested_geometry(record)
    extents = _extent_xyz(record, geometry)
    corners = _corners(record, geometry)
    score = _strict_float(record.get("raw_mean_score"), f"{prefix} raw_mean_score")
    evidence_scores_value = record.get("evidence_scores")
    if not isinstance(evidence_scores_value, list) or len(evidence_scores_value) != 3:
        raise BirthMaterializationError(
            f"{prefix} evidence_scores must be a three-element list"
        )
    evidence_scores = tuple(
        _strict_float(value, f"{prefix} evidence_scores[{position}]")
        for position, value in enumerate(evidence_scores_value)
    )
    if any(value < 0.0 or value > 1.0 for value in evidence_scores):
        raise BirthMaterializationError(f"{prefix} evidence score is outside [0,1]")
    if not math.isclose(
        score, float(np.mean(evidence_scores)), rel_tol=1e-6, abs_tol=1e-7
    ):
        raise BirthMaterializationError(
            f"{prefix} raw_mean_score disagrees with evidence_scores"
        )
    iou = _strict_float(
        _one_of(
            record,
            ("median_pairwise_mask_aabb_iou", "median_pairwise_lifted_aabb_iou"),
            f"{prefix} R15 metric",
        ),
        f"{prefix} median pairwise mask IoU",
    )
    if not 0.0 <= score <= 1.0 or not 0.0 <= iou <= 1.0:
        raise BirthMaterializationError(f"{prefix} score/IoU is outside [0,1]")
    view_voxels = _triple_ints(
        _one_of(
            record,
            ("view_supported_voxel_counts", "per_view_supported_voxel_counts"),
            f"{prefix} per-view voxel counts",
        ),
        f"{prefix} view_supported_voxel_counts",
    )
    target_group = record.get("target_group")
    if target_group is not None and (not isinstance(target_group, str) or not target_group):
        raise BirthMaterializationError(f"{prefix} target_group must be a nonempty string")
    return MaskLiftReceipt(
        scene=scene,
        track_id=_strict_int(record.get("track_id"), f"{prefix} track_id"),
        confirmation_frame_id=confirmation,
        evidence_frame_ids=frames,
        evidence_source_rows=rows,
        evidence_scores=evidence_scores,  # type: ignore[arg-type]
        min_evidence_score=min(evidence_scores),
        raw_mean_score=score,
        median_pairwise_mask_aabb_iou=iou,
        max_pairwise_mask_center_distance_m=_strict_float(
            _one_of(
                record,
                (
                    "max_pairwise_mask_center_distance_m",
                    "max_pairwise_lifted_center_distance_m",
                ),
                f"{prefix} pair center metric",
            ),
            f"{prefix} max pairwise mask center distance",
        ),
        first_last_frame_span=frame_span,
        max_camera_baseline_m=_strict_float(
            record.get("max_camera_baseline_m"), f"{prefix} max_camera_baseline_m"
        ),
        max_view_ray_span_deg=_strict_float(
            record.get("max_view_ray_span_deg"), f"{prefix} max_view_ray_span_deg"
        ),
        supported_voxel_count=_strict_int(
            _one_of(
                record,
                ("supported_voxel_count", "fused_supported_voxel_count"),
                f"{prefix} fused voxel count",
            ),
            f"{prefix} supported_voxel_count",
        ),
        view_supported_voxel_counts=view_voxels,
        fused_obb_extent_xyz=extents,
        fused_center_to_raw_medoid_m=_strict_float(
            record.get("fused_center_to_raw_medoid_m"),
            f"{prefix} fused_center_to_raw_medoid_m",
        ),
        corners=corners,
        target_group=target_group,
    )


def _hash_array(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(b"\0")
    digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode("ascii"))
    digest.update(b"\0")
    digest.update(array.tobytes())
    return digest.hexdigest()


def _resolve_sidecar_json(
    path: Path,
    *,
    directory_manifest_name: str = "TARGET_FIRST_MOBILESAM_MASKLIFT_FULL100.json",
) -> Path:
    if path.is_symlink():
        raise BirthMaterializationError(f"sidecar path must not be a symlink: {path}")
    if path.is_dir():
        path = path / directory_manifest_name
    return _regular_file(path, "target-first mask-lift sidecar")


def load_masklift_sidecar(
    path: Path,
    *,
    exact_schema: str | None = None,
    directory_manifest_name: str = "TARGET_FIRST_MOBILESAM_MASKLIFT_FULL100.json",
) -> MaskLiftSidecar:
    """Load and strictly validate a no-GT mask-lift sidecar."""

    path = _resolve_sidecar_json(
        path, directory_manifest_name=directory_manifest_name
    )
    digest = _sha256(path)

    def _reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant {value}")

    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"), parse_constant=_reject_constant
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise BirthMaterializationError(f"invalid mask-lift sidecar: {path}") from error
    if not isinstance(payload, dict):
        raise BirthMaterializationError("mask-lift sidecar must contain an object")
    schema = payload.get("schema")
    schema_valid = (
        schema == exact_schema
        if exact_schema is not None
        else isinstance(schema, str)
        and schema.startswith(
            "boxfusion.scannet_target_first_mobilesam_masklift_full100."
        )
    )
    if not schema_valid:
        raise BirthMaterializationError(f"unsupported mask-lift sidecar schema: {schema!r}")
    contracts = payload.get("contracts", {})
    if not isinstance(contracts, dict):
        raise BirthMaterializationError("mask-lift sidecar contracts must be an object")
    for key in ("gt_access", "evaluator_access"):
        declarations = [owner[key] for owner in (payload, contracts) if key in owner]
        if not declarations or any(value is not False for value in declarations):
            raise BirthMaterializationError(f"sidecar must declare {key}=false")
    for key in ("annotation_path_argument", "target_dataset_training", "online_learning"):
        declared = [payload.get(key), contracts.get(key)]
        if any(value not in (None, False) for value in declared):
            raise BirthMaterializationError(f"sidecar declares forbidden {key}")
    if not (
        payload.get("past_only_confirmation") is True
        or payload.get("past_only_tracking") is True
    ):
        raise BirthMaterializationError(
            "sidecar must declare past_only_confirmation/tracking=true"
        )
    scenes_node = payload.get("scenes")
    if not isinstance(scenes_node, (dict, list)):
        raise BirthMaterializationError("sidecar scenes must be an object or list")
    parsed: dict[str, tuple[MaskLiftReceipt, ...]] = {}
    identities: set[tuple[str, int, tuple[int, int, int]]] = set()
    shadow_acceptance: dict[tuple[str, int, tuple[int, int, int]], bool] = {}
    if isinstance(scenes_node, dict):
        scene_items = list(scenes_node.items())
    else:
        scene_items = []
        for index, scene_node in enumerate(scenes_node):
            if not isinstance(scene_node, dict):
                raise BirthMaterializationError(
                    f"sidecar scenes[{index}] must be an object"
                )
            scene = _one_of(
                scene_node, ("scene", "scene_id"), f"sidecar scenes[{index}] identity"
            )
            scene_items.append((scene, scene_node))
    for scene, scene_node in scene_items:
        if not isinstance(scene, str):
            raise BirthMaterializationError("sidecar scene keys must be strings")
        if isinstance(scene_node, list):
            records = scene_node
        elif isinstance(scene_node, dict):
            record_keys = [key for key in ("receipts", "tracks", "records") if key in scene_node]
            if len(record_keys) != 1 or not isinstance(scene_node[record_keys[0]], list):
                raise BirthMaterializationError(
                    f"{scene} requires exactly one receipt/track/record list"
                )
            records = scene_node[record_keys[0]]
        else:
            raise BirthMaterializationError(f"invalid scene sidecar node: {scene}")
        receipts = tuple(
            _parse_receipt(scene, record, index) for index, record in enumerate(records)
        )
        for receipt in receipts:
            identity = (scene, receipt.track_id, receipt.evidence_source_rows)
            if identity in identities:
                raise BirthMaterializationError(f"duplicate receipt identity: {identity}")
            identities.add(identity)
        for record, receipt in zip(records, receipts):
            assert isinstance(record, dict)
            if "accepted" in record or "decision" in record:
                accepted = record.get("accepted")
                decision = record.get("decision")
                if not isinstance(accepted, bool) or not isinstance(decision, str):
                    raise BirthMaterializationError(
                        f"{scene} receipt accepted/decision fields are malformed"
                    )
                if accepted != (decision == "accepted_shadow"):
                    raise BirthMaterializationError(
                        f"{scene} receipt accepted and decision disagree"
                    )
                shadow_acceptance[(scene, receipt.track_id, receipt.evidence_source_rows)] = accepted
        parsed[scene] = receipts
    count = sum(map(len, parsed.values()))
    declared_count = payload.get("receipt_count")
    if declared_count is not None and _strict_int(declared_count, "receipt_count") != count:
        raise BirthMaterializationError("sidecar receipt_count does not match scenes")
    npz_path: Path | None = None
    npz_digest: str | None = None
    array_hashes: dict[str, str] = {}
    npz_file = payload.get("npz_file")
    npz_declarations = payload.get("npz_arrays")
    if (npz_file is None) != (npz_declarations is None):
        raise BirthMaterializationError("sidecar must declare both npz_file and npz_arrays")
    if npz_file is not None:
        if (
            not isinstance(npz_file, str)
            or not npz_file
            or Path(npz_file).name != npz_file
        ):
            raise BirthMaterializationError("npz_file must be a safe basename")
        if not isinstance(npz_declarations, dict):
            raise BirthMaterializationError("npz_arrays must be an object")
        npz_path = _regular_file(path.parent / npz_file, "mask-lift NPZ").resolve()
        npz_digest = _sha256(npz_path)
        try:
            with np.load(npz_path, allow_pickle=False) as archive:
                if set(archive.files) != set(npz_declarations):
                    raise BirthMaterializationError("NPZ array names disagree with manifest")
                loaded_arrays: dict[str, np.ndarray] = {}
                for name in archive.files:
                    declaration = npz_declarations[name]
                    if not isinstance(declaration, dict):
                        raise BirthMaterializationError(f"invalid NPZ declaration: {name}")
                    array = np.asarray(archive[name])
                    if array.dtype.hasobject:
                        raise BirthMaterializationError(f"object NPZ array is forbidden: {name}")
                    digest_value = _hash_array(array)
                    if (
                        declaration.get("dtype") != array.dtype.str
                        or declaration.get("shape") != list(array.shape)
                        or declaration.get("sha256") != digest_value
                    ):
                        raise BirthMaterializationError(f"NPZ metadata/hash mismatch: {name}")
                    loaded_arrays[name] = array
                    array_hashes[name] = digest_value
        except BirthMaterializationError:
            raise
        except (OSError, ValueError, zipfile.BadZipFile) as error:
            raise BirthMaterializationError(f"invalid mask-lift NPZ: {npz_path}") from error

        required_arrays = {
            "track_scene_index",
            "track_group_track_id",
            "track_fused_obb_corners",
            "track_accepted_shadow",
        }
        if not required_arrays.issubset(loaded_arrays):
            raise BirthMaterializationError("NPZ lacks required track arrays")
        declared_order = payload.get("scene_order")
        if not isinstance(declared_order, list) or any(
            not isinstance(scene, str) for scene in declared_order
        ):
            raise BirthMaterializationError("NPZ sidecar requires scene_order")
        if len(set(declared_order)) != len(declared_order) or set(declared_order) != set(parsed):
            raise BirthMaterializationError("scene_order disagrees with canonical scenes")
        flattened = [receipt for scene in declared_order for receipt in parsed[scene]]
        track_count = len(flattened)
        for name in required_arrays:
            if loaded_arrays[name].shape[0] != track_count:
                raise BirthMaterializationError(f"NPZ track length mismatch: {name}")
        expected_scene_indices = np.concatenate(
            [np.full(len(parsed[scene]), index) for index, scene in enumerate(declared_order)]
        ).astype(loaded_arrays["track_scene_index"].dtype, copy=False)
        if not np.array_equal(loaded_arrays["track_scene_index"], expected_scene_indices):
            raise BirthMaterializationError("NPZ track_scene_index disagrees with JSON")
        expected_track_ids = np.asarray(
            [receipt.track_id for receipt in flattened],
            dtype=loaded_arrays["track_group_track_id"].dtype,
        )
        if not np.array_equal(loaded_arrays["track_group_track_id"], expected_track_ids):
            raise BirthMaterializationError("NPZ track IDs disagree with JSON")
        expected_corners = np.asarray(
            [receipt.corners for receipt in flattened], dtype=np.float32
        ).reshape(-1, 8, 3)
        if not np.array_equal(loaded_arrays["track_fused_obb_corners"], expected_corners):
            raise BirthMaterializationError("NPZ fused corners disagree with JSON")
        if len(shadow_acceptance) == track_count:
            expected_acceptance = np.asarray(
                [
                    shadow_acceptance[
                        (receipt.scene, receipt.track_id, receipt.evidence_source_rows)
                    ]
                    for receipt in flattened
                ],
                dtype=bool,
            )
            if not np.array_equal(
                loaded_arrays["track_accepted_shadow"], expected_acceptance
            ):
                raise BirthMaterializationError("NPZ shadow decisions disagree with JSON")

    if _sha256(path) != digest:
        raise BirthMaterializationError("mask-lift sidecar changed while it was read")
    if npz_path is not None and _sha256(npz_path) != npz_digest:
        raise BirthMaterializationError("mask-lift NPZ changed while it was read")
    return MaskLiftSidecar(
        path=path.resolve(),
        sha256=digest,
        schema=schema,
        receipts=parsed,
        receipt_count=count,
        npz_path=npz_path,
        npz_sha256=npz_digest,
        npz_array_sha256=array_hashes,
    )


def _rank_key(receipt: MaskLiftReceipt) -> tuple[float, float, float, int, float, int]:
    return (
        -receipt.median_pairwise_mask_aabb_iou,
        -receipt.min_evidence_score,
        -receipt.raw_mean_score,
        -receipt.supported_voxel_count,
        receipt.max_pairwise_mask_center_distance_m,
        receipt.track_id,
    )


def select_births(
    receipts: Sequence[MaskLiftReceipt], native_corners: np.ndarray
) -> tuple[tuple[MaskLiftReceipt, ...], list[dict[str, Any]]]:
    """Apply the frozen R15, native-novelty, NMS, and cap-four policy."""

    ranked = tuple(sorted(receipts, key=_rank_key))
    corners = (
        np.stack([receipt.corners for receipt in ranked])
        if ranked
        else np.empty((0, 8, 3), dtype=np.float64)
    )
    native_iou, candidate_in_native, native_in_candidate = _aabb_overlap_matrices(
        corners, native_corners
    )
    self_iou, self_left, self_right = _aabb_overlap_matrices(corners, corners)
    kept_indices: list[int] = []
    selected: list[MaskLiftReceipt] = []
    decisions: list[dict[str, Any]] = []
    for index, receipt in enumerate(ranked):
        max_native_iou = float(native_iou[index].max()) if native_iou.shape[1] else 0.0
        max_candidate_in_native = (
            float(candidate_in_native[index].max()) if candidate_in_native.shape[1] else 0.0
        )
        max_native_in_candidate = (
            float(native_in_candidate[index].max()) if native_in_candidate.shape[1] else 0.0
        )
        decision = "accepted"
        if receipt.min_evidence_score < MIN_EVIDENCE_SCORE:
            decision = "min_score"
        elif receipt.raw_mean_score < MIN_RAW_MEAN_SCORE:
            decision = "score"
        elif receipt.median_pairwise_mask_aabb_iou < MIN_MEDIAN_PAIRWISE_MASK_AABB_IOU:
            decision = "r15"
        elif receipt.max_pairwise_mask_center_distance_m > MAX_PAIRWISE_MASK_CENTER_DISTANCE_M:
            decision = "center_distance"
        elif receipt.first_last_frame_span < MIN_FIRST_LAST_FRAME_SPAN:
            decision = "frame_span"
        elif (
            receipt.max_camera_baseline_m < MIN_CAMERA_BASELINE_M
            or receipt.max_view_ray_span_deg < MIN_VIEW_RAY_SPAN_DEG
        ):
            decision = "view_diversity"
        elif (
            receipt.supported_voxel_count < MIN_FUSED_SUPPORTED_VOXELS
            or min(receipt.view_supported_voxel_counts) < MIN_SUPPORTED_VOXELS_PER_VIEW
        ):
            decision = "voxel_support"
        elif min(receipt.fused_obb_extent_xyz) < MIN_FUSED_OBB_EXTENT_M:
            decision = "too_small"
        elif receipt.fused_center_to_raw_medoid_m > MAX_FUSED_CENTER_TO_RAW_MEDOID_M:
            decision = "raw_medoid_distance"
        elif max_native_iou >= NATIVE_NOVELTY_AABB_IOU:
            decision = "native_overlap"
        elif (
            max_candidate_in_native >= NATIVE_MAX_BIDIRECTIONAL_CONTAINMENT
            or max_native_in_candidate >= NATIVE_MAX_BIDIRECTIONAL_CONTAINMENT
        ):
            decision = "native_containment"
        else:
            for kept_index in kept_indices:
                if (
                    self_iou[index, kept_index] >= SELF_NMS_AABB_IOU
                    or self_left[index, kept_index] >= SELF_NMS_BIDIRECTIONAL_CONTAINMENT
                    or self_right[index, kept_index] >= SELF_NMS_BIDIRECTIONAL_CONTAINMENT
                ):
                    decision = "self_nms"
                    break
        if decision == "accepted" and len(selected) >= MAX_BIRTHS_PER_SCENE:
            decision = "scene_cap"
        if decision == "accepted":
            kept_indices.append(index)
            selected.append(receipt)
        decisions.append(
            {
                "track_id": receipt.track_id,
                "decision": decision,
                "confirmation_frame_id": receipt.confirmation_frame_id,
                "evidence_frame_ids": list(receipt.evidence_frame_ids),
                "evidence_source_rows": list(receipt.evidence_source_rows),
                "target_group": receipt.target_group,
                "evidence_scores": list(receipt.evidence_scores),
                "min_evidence_score": receipt.min_evidence_score,
                "raw_mean_score": receipt.raw_mean_score,
                "median_pairwise_mask_aabb_iou": receipt.median_pairwise_mask_aabb_iou,
                "max_pairwise_mask_center_distance_m": receipt.max_pairwise_mask_center_distance_m,
                "first_last_frame_span": receipt.first_last_frame_span,
                "max_camera_baseline_m": receipt.max_camera_baseline_m,
                "max_view_ray_span_deg": receipt.max_view_ray_span_deg,
                "supported_voxel_count": receipt.supported_voxel_count,
                "view_supported_voxel_counts": list(receipt.view_supported_voxel_counts),
                "fused_obb_extent_xyz": list(receipt.fused_obb_extent_xyz),
                "fused_center_to_raw_medoid_m": receipt.fused_center_to_raw_medoid_m,
                "max_native_aabb_iou": max_native_iou,
                "max_candidate_in_native_containment": max_candidate_in_native,
                "max_native_in_candidate_containment": max_native_in_candidate,
            }
        )
    return tuple(selected), decisions


def _augmented_payload(
    native: NativePrediction, selected: Sequence[MaskLiftReceipt]
) -> list[Any] | tuple[Any, ...]:
    suffix = [
        (
            APPENDED_CLASS_ID,
            np.ascontiguousarray(receipt.corners, dtype=np.float32),
            APPENDED_SCORE,
        )
        for receipt in selected
    ]
    rows: list[Any] | tuple[Any, ...]
    if isinstance(native.rows, tuple):
        rows = tuple(native.rows) + tuple(suffix)
    else:
        rows = list(native.rows) + suffix
    output: list[Any] | tuple[Any, ...]
    output = (rows,) if isinstance(native.payload, tuple) else [rows]
    _assert_native_prefix(native.rows, output[0], "in-memory target-first output")
    return output


def _policy_manifest() -> dict[str, Any]:
    return {
        "receipt_admission_name": "R15",
        "min_evidence_score_gte": MIN_EVIDENCE_SCORE,
        "min_raw_mean_score_gte": MIN_RAW_MEAN_SCORE,
        "median_pairwise_mask_aabb_iou_gte": MIN_MEDIAN_PAIRWISE_MASK_AABB_IOU,
        "max_pairwise_mask_center_distance_m_lte": MAX_PAIRWISE_MASK_CENTER_DISTANCE_M,
        "min_first_last_frame_span_gte": MIN_FIRST_LAST_FRAME_SPAN,
        "min_camera_baseline_m_gte": MIN_CAMERA_BASELINE_M,
        "min_view_ray_span_deg_gte": MIN_VIEW_RAY_SPAN_DEG,
        "min_fused_supported_voxels_gte": MIN_FUSED_SUPPORTED_VOXELS,
        "min_supported_voxels_each_view_gte": MIN_SUPPORTED_VOXELS_PER_VIEW,
        "min_fused_obb_extent_each_axis_m_gte": MIN_FUSED_OBB_EXTENT_M,
        "max_fused_center_to_raw_medoid_m_lte": MAX_FUSED_CENTER_TO_RAW_MEDOID_M,
        "fused_geometry": "same_raw_medoid_yaw_local_q02_q98_robust_obb",
        "native_novelty_aabb_iou_gte_reject": NATIVE_NOVELTY_AABB_IOU,
        "native_bidirectional_containment_gte_reject": NATIVE_MAX_BIDIRECTIONAL_CONTAINMENT,
        "self_nms_aabb_iou_gte_reject": SELF_NMS_AABB_IOU,
        "self_nms_bidirectional_containment_gte_reject": SELF_NMS_BIDIRECTIONAL_CONTAINMENT,
        "max_births_per_scene": MAX_BIRTHS_PER_SCENE,
        "ranking": [
            "median_pairwise_mask_aabb_iou_desc",
            "min_evidence_score_desc",
            "raw_mean_score_desc",
            "supported_voxel_count_desc",
            "max_pairwise_mask_center_distance_m_asc",
            "track_id_asc",
        ],
        "appended_class_id": APPENDED_CLASS_ID,
        "appended_score": APPENDED_SCORE,
    }


def materialize_scannet_target_first_mobilesam_birth_full100(
    *,
    scene_list: Path,
    baseline_root: Path,
    masklift_sidecar: Path,
    output_root: Path,
    expected_scene_count: int = 100,
    plan_only: bool = False,
    exact_sidecar_schema: str | None = None,
    sidecar_directory_manifest_name: str = "TARGET_FIRST_MOBILESAM_MASKLIFT_FULL100.json",
    output_schema: str = SCHEMA,
    output_manifest_name: str = MANIFEST_NAME,
    materializer_adapter_source: Path | None = None,
) -> dict[str, Any]:
    """Validate, select, and optionally create the full prediction root."""

    if baseline_root.is_symlink() or not baseline_root.is_dir():
        raise BirthMaterializationError(
            f"baseline root must be a non-symlink directory: {baseline_root}"
        )
    if expected_scene_count <= 0:
        raise BirthMaterializationError("expected scene count must be positive")
    output_root = output_root.resolve()
    if output_root.exists() or output_root.is_symlink():
        raise BirthMaterializationError(f"refusing to overwrite output root: {output_root}")
    scenes = _scene_list(scene_list, expected_scene_count)
    sidecar = load_masklift_sidecar(
        masklift_sidecar,
        exact_schema=exact_sidecar_schema,
        directory_manifest_name=sidecar_directory_manifest_name,
    )
    if set(sidecar.receipts) != set(scenes):
        missing = sorted(set(scenes) - set(sidecar.receipts))
        extra = sorted(set(sidecar.receipts) - set(scenes))
        raise BirthMaterializationError(
            f"sidecar/protocol scene mismatch; missing={missing[:3]}, extra={extra[:3]}"
        )

    stage: Path | None = None
    if not plan_only:
        output_root.parent.mkdir(parents=True, exist_ok=True)
        stage = Path(
            tempfile.mkdtemp(prefix=f".{output_root.name}.stage-", dir=output_root.parent)
        )
    baseline_hashes: dict[str, str] = {}
    output_hashes: dict[str, str] = {}
    reports: dict[str, Any] = {}
    total_native = 0
    total_births = 0
    try:
        for scene in scenes:
            native_path = _regular_file(
                baseline_root / f"{scene}{PREDICTION_SUFFIX}", "native Cbest prediction"
            )
            native_digest = _sha256(native_path)
            baseline_hashes[scene] = native_digest
            native = _load_native_prediction(native_path)
            selected, decisions = select_births(sidecar.receipts[scene], native.corners)
            if _sha256(native_path) != native_digest:
                raise BirthMaterializationError(f"native prediction changed: {scene}")

            if not plan_only:
                assert stage is not None
                output_path = stage / f"{scene}{PREDICTION_SUFFIX}"
                _write_pickle(output_path, _augmented_payload(native, selected))
                reloaded = _load_native_prediction(output_path)
                _assert_native_prefix(native.rows, reloaded.rows, scene)
                if len(reloaded.rows) != len(native.rows) + len(selected):
                    raise BirthMaterializationError(f"suffix count changed: {scene}")
                output_hashes[scene] = _sha256(output_path)

            reasons = (
                "accepted",
                "min_score",
                "score",
                "r15",
                "center_distance",
                "frame_span",
                "view_diversity",
                "voxel_support",
                "too_small",
                "raw_medoid_distance",
                "native_overlap",
                "native_containment",
                "self_nms",
                "scene_cap",
            )
            reports[scene] = {
                "native_count": len(native.rows),
                "masklift_receipt_count": len(sidecar.receipts[scene]),
                "birth_count": len(selected),
                "decision_counts": {
                    reason: sum(row["decision"] == reason for row in decisions)
                    for reason in reasons
                },
                "native_prefix_row_identity_verified": not plan_only,
                "suffix": [
                    {
                        "suffix_index": index,
                        "track_id": receipt.track_id,
                        "class_id": APPENDED_CLASS_ID,
                        "score": APPENDED_SCORE,
                        "corners_world": receipt.corners.tolist(),
                        "confirmation_frame_id": receipt.confirmation_frame_id,
                        "evidence_frame_ids": list(receipt.evidence_frame_ids),
                        "evidence_source_rows": list(receipt.evidence_source_rows),
                        "target_group": receipt.target_group,
                    }
                    for index, receipt in enumerate(selected)
                ],
                "receipt_decisions": decisions,
            }
            total_native += len(native.rows)
            total_births += len(selected)

        if _sha256(sidecar.path) != sidecar.sha256:
            raise BirthMaterializationError("mask-lift sidecar changed during materialization")
        if sidecar.npz_path is not None and _sha256(sidecar.npz_path) != sidecar.npz_sha256:
            raise BirthMaterializationError("mask-lift NPZ changed during materialization")
        adapter_source = (
            None
            if materializer_adapter_source is None
            else _regular_file(
                materializer_adapter_source, "materializer adapter source"
            ).resolve()
        )
        manifest: dict[str, Any] = {
            "schema": output_schema,
            "mode": "plan_only" if plan_only else "active_birth_r15",
            "plan_only": plan_only,
            "training_free": True,
            "target_dataset_training": False,
            "external_pretraining_frozen": True,
            "online_learning": False,
            "past_only_confirmation": True,
            "minimum_distinct_views": 3,
            "gt_access": False,
            "evaluator_access": False,
            "annotation_path_argument": False,
            "depth_access": "sealed_masklift_sidecar_only",
            "rgb_access": "sealed_masklift_sidecar_only",
            "native_rows_are_unchanged_prefix": True,
            "native_clip_unchanged": True,
            "coordinate_frame": "scannet_world",
            "score_mode": "constant_1.0",
            "class_mode": "inert_0_scannet_class_agnostic_evaluator",
            "scene_count": len(scenes),
            "native_count": total_native,
            "masklift_receipt_count": sidecar.receipt_count,
            "birth_count": total_births,
            "frozen_policy": _policy_manifest(),
            "inputs": {
                "scene_list": os.fspath(scene_list.resolve()),
                "scene_list_sha256": _sha256(scene_list),
                "baseline_root": os.fspath(baseline_root.resolve()),
                "masklift_sidecar": os.fspath(sidecar.path),
                "masklift_sidecar_sha256": sidecar.sha256,
                "masklift_sidecar_schema": sidecar.schema,
                "masklift_npz": (
                    None if sidecar.npz_path is None else os.fspath(sidecar.npz_path)
                ),
                "masklift_npz_sha256": sidecar.npz_sha256,
                "masklift_npz_array_sha256": sidecar.npz_array_sha256,
                "materializer_source": os.fspath(Path(__file__).resolve()),
                "materializer_source_sha256": _sha256(Path(__file__).resolve()),
                "materializer_adapter_source": (
                    None if adapter_source is None else os.fspath(adapter_source)
                ),
                "materializer_adapter_source_sha256": (
                    None if adapter_source is None else _sha256(adapter_source)
                ),
            },
            "native_prediction_sha256": baseline_hashes,
            "output_prediction_sha256": output_hashes,
            "scenes": reports,
        }
        if plan_only:
            return manifest
        assert stage is not None
        _write_json(stage / output_manifest_name, manifest)
        directory_fd = os.open(stage, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        if output_root.exists() or output_root.is_symlink():
            raise BirthMaterializationError(f"refusing to overwrite output root: {output_root}")
        os.rename(stage, output_root)
        stage = None
        return manifest
    except Exception:
        if stage is not None and stage.exists():
            shutil.rmtree(stage)
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Materialize frozen target-first MobileSAM R15 births"
    )
    parser.add_argument(
        "--scene-list",
        type=Path,
        default=REPOSITORY_ROOT / "evaluation/data_util/meta_data/scannetv2_val.txt",
    )
    parser.add_argument(
        "--baseline-root",
        type=Path,
        default=REPOSITORY_ROOT / "results/scannet_t05_boxer_replay_active_score05",
    )
    parser.add_argument("--masklift-sidecar", type=Path, required=True)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPOSITORY_ROOT / "results/scannet_target_first_mobilesam_birth_r15_score05",
    )
    parser.add_argument("--expected-scene-count", type=int, default=100)
    parser.add_argument(
        "--plan-only",
        action="store_true",
        help="validate and select all scenes without creating an output root",
    )
    return parser


def main() -> None:
    args = _parser().parse_args()
    manifest = materialize_scannet_target_first_mobilesam_birth_full100(
        scene_list=args.scene_list,
        baseline_root=args.baseline_root,
        masklift_sidecar=args.masklift_sidecar,
        output_root=args.output_root,
        expected_scene_count=args.expected_scene_count,
        plan_only=args.plan_only,
    )
    print(
        json.dumps(
            {
                "schema": manifest["schema"],
                "mode": manifest["mode"],
                "scene_count": manifest["scene_count"],
                "native_count": manifest["native_count"],
                "masklift_receipt_count": manifest["masklift_receipt_count"],
                "birth_count": manifest["birth_count"],
                "output_root": None if args.plan_only else os.fspath(args.output_root.resolve()),
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
