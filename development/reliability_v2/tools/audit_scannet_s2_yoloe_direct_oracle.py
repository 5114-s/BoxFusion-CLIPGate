#!/usr/bin/env python3
"""Post-hoc ScanNet oracle for a sealed S2 YOLOE-direct shadow suffix.

The no-GT materializer fixes candidate membership and order.  This program
first validates that complete seal, including the native T05 prefix, and only
then opens ScanNet ground truth.  Ground truth is used exclusively to report
the fixed counterfactual; it cannot select, suppress, rank, or modify a row.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import pickle
import re
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if os.fspath(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, os.fspath(_REPOSITORY_ROOT))

from tools.audit_scannet_boxer_unexplained_oracle import (  # noqa: E402
    aligned_iou_matrix,
    load_axis_alignment,
    load_baseline_boxes,
    load_gt_minmax,
    official_constant_evaluate,
    strict_maximum_matching,
)


SCHEMA = "boxfusion.scannet_s2_yoloe_direct_oracle.v1"
SHADOW_SCHEMA = "boxfusion.s2_yoloe_direct_shadow.v1"
THRESHOLDS = (0.15, 0.25, 0.50)
SCENE_PATTERN = re.compile(r"^scene[0-9]{4}_[0-9]{2}$")

NATIVE_NOVELTY_IOU = 0.10
SELF_NMS_IOU = 0.25
MAX_OUTPUTS_PER_SCENE = 6
FORMAL_SCORE = 1.0

_EXPECTED_ARRAY_DTYPES: Mapping[str, np.dtype[Any]] = {
    "scene_ids": np.dtype("<U12"),
    "candidate_scene_index": np.dtype(np.int16),
    "candidate_terminal_rank": np.dtype(np.int16),
    "candidate_diagnostic_row": np.dtype(np.int32),
    "candidate_result_index": np.dtype(np.int32),
    "candidate_track_id": np.dtype(np.int64),
    "candidate_box_center_extent": np.dtype(np.float32),
    "candidate_corners_world": np.dtype(np.float32),
    "candidate_raw_score_provenance": np.dtype(np.float32),
    "candidate_stored_appended_score_diagnostic_only": np.dtype(np.float32),
    "candidate_formal_evaluation_score": np.dtype(np.float32),
    "candidate_max_native_aabb_iou": np.dtype(np.float32),
    "candidate_valid_point_count": np.dtype(np.int16),
    "counterfactual_scene_offsets": np.dtype(np.int32),
    "counterfactual_corners_world": np.dtype(np.float32),
    "counterfactual_stored_score_provenance": np.dtype(np.float32),
    "counterfactual_formal_evaluation_score": np.dtype(np.float32),
    "counterfactual_is_native_prefix": np.dtype(bool),
}


class S2OracleError(ValueError):
    """Raised when the sealed S2 artifact or evaluation contract is invalid."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _regular_file(path: Path, label: str) -> Path:
    path = path.resolve()
    if path.is_symlink() or not path.is_file():
        raise S2OracleError(f"{label} must be a regular non-symlink file: {path}")
    return path


def _read_json(path: Path, label: str) -> dict[str, Any]:
    _regular_file(path, label)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise S2OracleError(f"invalid {label}: {path}") from error
    if not isinstance(value, dict):
        raise S2OracleError(f"{label} must contain a JSON object: {path}")
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


def _row_payload_sha256(rows: Sequence[Any]) -> str:
    return hashlib.sha256(
        pickle.dumps(list(rows), protocol=pickle.HIGHEST_PROTOCOL)
    ).hexdigest()


def _load_prediction_rows(
    path: Path, label: str
) -> tuple[list[Any], np.ndarray, np.ndarray]:
    path = _regular_file(path, label)
    try:
        with path.open("rb") as handle:
            payload = pickle.load(handle)
    except Exception as error:
        raise S2OracleError(f"could not load {label}: {path}") from error
    if not isinstance(payload, (list, tuple)) or len(payload) != 1:
        raise S2OracleError(f"invalid {label} outer schema: {path}")
    rows = payload[0]
    if not isinstance(rows, (list, tuple)):
        raise S2OracleError(f"invalid {label} row container: {path}")
    corners: list[np.ndarray] = []
    scores: list[float] = []
    for row_index, row in enumerate(rows):
        if not isinstance(row, (list, tuple)) or len(row) != 3:
            raise S2OracleError(f"invalid {label} row {row_index}: {path}")
        box = np.asarray(row[1])
        try:
            score = float(row[2])
        except (TypeError, ValueError) as error:
            raise S2OracleError(f"invalid {label} score at row {row_index}") from error
        if (
            box.shape != (8, 3)
            or not np.issubdtype(box.dtype, np.number)
            or not np.isfinite(box).all()
            or not math.isfinite(score)
        ):
            raise S2OracleError(f"invalid {label} geometry/score at row {row_index}")
        corners.append(np.asarray(box, dtype=np.float32))
        scores.append(score)
    corner_array = (
        np.stack(corners).astype(np.float32, copy=False)
        if corners
        else np.empty((0, 8, 3), dtype=np.float32)
    )
    return list(rows), corner_array, np.asarray(scores, dtype=np.float32)


def _aabb_iou_matrix(left_corners: np.ndarray, right_corners: np.ndarray) -> np.ndarray:
    left = np.asarray(left_corners, dtype=np.float64)
    right = np.asarray(right_corners, dtype=np.float64)
    if left.shape[1:] != (8, 3) or right.shape[1:] != (8, 3):
        raise S2OracleError("corner arrays must have shape Nx8x3")
    left_boxes = np.concatenate((left.min(1), left.max(1)), axis=1)
    right_boxes = np.concatenate((right.min(1), right.max(1)), axis=1)
    return aligned_iou_matrix(left_boxes, right_boxes)


def _require_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise S2OracleError(f"{label} must be an object")
    return value


def _validate_manifest_contract(manifest: Mapping[str, Any]) -> None:
    required = {
        "schema": SHADOW_SCHEMA,
        "mode": "shadow",
        "output_inert": True,
        "birth": False,
        "active_authorized": False,
        "native_mutation_applied": False,
        "gt_access": False,
        "oracle_access": False,
        "training_free": True,
        "online_learning": False,
        "past_current_only": True,
        "future_frames_used": False,
        "detector_semantics_used_for_gate": False,
        "native_clip_access": False,
        "native_clip_unchanged": True,
        "coordinate_frame": "scannet_world",
        "score_mode_for_formal_evaluation": "constant_1.0",
        "stored_scores_are_diagnostic_only": True,
    }
    for key, expected in required.items():
        if manifest.get(key) != expected:
            raise S2OracleError(
                f"shadow contract mismatch for {key}: "
                f"expected={expected!r}, actual={manifest.get(key)!r}"
            )
    policy = _require_mapping(manifest.get("frozen_policy"), "frozen_policy")
    expected_policy = {
        "candidate_source_index": -1,
        "diagnostic_order_preserved": True,
        "native_novelty_aabb_iou_strict_less_than": NATIVE_NOVELTY_IOU,
        "candidate_self_nms_aabb_iou_strict_less_than": SELF_NMS_IOU,
        "maximum_appended_candidates_per_scene": MAX_OUTPUTS_PER_SCENE,
        "terminal_candidate_labels_ignored": True,
        "terminal_clip_access": False,
        "native_prefix_rows_exact": True,
        "formal_evaluation_score": FORMAL_SCORE,
    }
    for key, expected in expected_policy.items():
        if policy.get(key) != expected:
            raise S2OracleError(
                f"frozen policy mismatch for {key}: "
                f"expected={expected!r}, actual={policy.get(key)!r}"
            )
    identities = _require_mapping(
        manifest.get("input_hash_identity"), "input_hash_identity"
    )
    expected_identities = {
        "candidate_diagnostics_before_after",
        "native_predictions_before_after",
        "frozen_sources_before_after",
        "preregistration_before_after",
        "materializer_before_after",
    }
    if set(identities) != expected_identities or any(
        identities[key] is not True for key in expected_identities
    ):
        raise S2OracleError("one or more materializer inputs lack before/after identity")
    inputs = _require_mapping(manifest.get("input"), "input")
    for key in (
        "baseline_root",
        "preregistration",
        "preregistration_expected_sha256",
        "preregistration_sha256",
        "materializer_source",
        "materializer_source_sha256",
        "frozen_inputs",
    ):
        if key not in inputs:
            raise S2OracleError(f"shadow input ledger is missing {key}")
    expected_prereg = inputs["preregistration_expected_sha256"]
    actual_prereg = inputs["preregistration_sha256"]
    if (
        not isinstance(expected_prereg, str)
        or not re.fullmatch(r"[0-9a-f]{64}", expected_prereg)
        or actual_prereg != expected_prereg
    ):
        raise S2OracleError("preregistration SHA-256 seal is invalid")


def _validate_array_shapes_and_values(
    arrays: Mapping[str, np.ndarray], scenes: tuple[str, ...], count: int
) -> None:
    shapes = {
        "scene_ids": (len(scenes),),
        "candidate_scene_index": (count,),
        "candidate_terminal_rank": (count,),
        "candidate_diagnostic_row": (count,),
        "candidate_result_index": (count,),
        "candidate_track_id": (count,),
        "candidate_box_center_extent": (count, 6),
        "candidate_corners_world": (count, 8, 3),
        "candidate_raw_score_provenance": (count,),
        "candidate_stored_appended_score_diagnostic_only": (count,),
        "candidate_formal_evaluation_score": (count,),
        "candidate_max_native_aabb_iou": (count,),
        "candidate_valid_point_count": (count,),
        "counterfactual_scene_offsets": (len(scenes) + 1,),
    }
    for name, expected in shapes.items():
        if arrays[name].shape != expected:
            raise S2OracleError(
                f"shadow array {name} shape mismatch: "
                f"expected={expected}, actual={arrays[name].shape}"
            )
    offsets = arrays["counterfactual_scene_offsets"]
    total_rows = int(offsets[-1]) if len(offsets) else -1
    for name in (
        "counterfactual_corners_world",
        "counterfactual_stored_score_provenance",
        "counterfactual_formal_evaluation_score",
        "counterfactual_is_native_prefix",
    ):
        tail = (8, 3) if name == "counterfactual_corners_world" else ()
        if arrays[name].shape != (total_rows, *tail):
            raise S2OracleError(f"shadow array {name} counterfactual shape mismatch")
    if int(offsets[0]) != 0 or np.any(np.diff(offsets) < 0):
        raise S2OracleError("counterfactual scene offsets are invalid")

    finite_names = {
        name for name, value in arrays.items() if value.dtype.kind in "fc"
    }
    if any(not np.isfinite(arrays[name]).all() for name in finite_names):
        raise S2OracleError("shadow arrays contain a non-finite value")
    scene_index = arrays["candidate_scene_index"].astype(np.int64)
    if np.any((scene_index < 0) | (scene_index >= len(scenes))):
        raise S2OracleError("candidate scene index is out of range")
    if count and np.any(np.diff(scene_index) < 0):
        raise S2OracleError("candidate rows do not preserve sealed scene order")
    boxes = arrays["candidate_box_center_extent"]
    if np.any(boxes[:, 3:] <= 0.0):
        raise S2OracleError("candidate box has a non-positive extent")
    corners = arrays["candidate_corners_world"]
    if count:
        expected_min = boxes[:, :3] - boxes[:, 3:] / 2.0
        expected_max = boxes[:, :3] + boxes[:, 3:] / 2.0
        if not (
            np.allclose(corners.min(1), expected_min, rtol=0.0, atol=2e-6)
            and np.allclose(corners.max(1), expected_max, rtol=0.0, atol=2e-6)
        ):
            raise S2OracleError("candidate center/extent and corners disagree")
    raw = arrays["candidate_raw_score_provenance"]
    stored = arrays["candidate_stored_appended_score_diagnostic_only"]
    if np.any((raw <= 0.0) | (raw > 1.0)) or np.any(
        (stored <= 0.0) | (stored > 1.0)
    ):
        raise S2OracleError("candidate diagnostic score is outside (0,1]")
    if not np.array_equal(
        arrays["candidate_formal_evaluation_score"],
        np.ones(count, dtype=np.float32),
    ):
        raise S2OracleError("candidate formal scores are not all exactly 1.0")
    if not np.array_equal(
        arrays["counterfactual_formal_evaluation_score"],
        np.ones(total_rows, dtype=np.float32),
    ):
        raise S2OracleError("counterfactual formal scores are not all exactly 1.0")
    overlap = arrays["candidate_max_native_aabb_iou"]
    if np.any((overlap < 0.0) | (overlap >= NATIVE_NOVELTY_IOU)):
        raise S2OracleError("candidate violates terminal native novelty gate")
    if np.any(arrays["candidate_track_id"] >= 0):
        raise S2OracleError("S2 supplemental track IDs must be negative")
    if np.any(arrays["candidate_valid_point_count"] <= 0):
        raise S2OracleError("candidate has no valid depth points")
    if np.any(arrays["candidate_diagnostic_row"] < 0) or np.any(
        arrays["candidate_result_index"] < 0
    ):
        raise S2OracleError("candidate provenance index is negative")

    for scene_index_value in range(len(scenes)):
        positions = np.flatnonzero(scene_index == scene_index_value)
        if len(positions) > MAX_OUTPUTS_PER_SCENE:
            raise S2OracleError("scene exceeds the frozen candidate cap")
        if not np.array_equal(
            arrays["candidate_terminal_rank"][positions],
            np.arange(len(positions), dtype=np.int16),
        ):
            raise S2OracleError("candidate terminal rank/order is invalid")
        track_ids = arrays["candidate_track_id"][positions]
        if len(np.unique(track_ids)) != len(track_ids):
            raise S2OracleError("candidate track ID is duplicated within a scene")
        if len(positions) > 1:
            self_iou = _aabb_iou_matrix(corners[positions], corners[positions])
            np.fill_diagonal(self_iou, 0.0)
            if np.any(self_iou >= SELF_NMS_IOU):
                raise S2OracleError("candidate pair violates frozen self-NMS gate")


def _validate_manifest_candidate_mirror(
    manifest: Mapping[str, Any], arrays: Mapping[str, np.ndarray], scenes: tuple[str, ...]
) -> None:
    ledger = _require_mapping(manifest.get("scenes"), "scenes")
    if set(ledger) != set(scenes):
        raise S2OracleError("manifest and NPZ scene ledgers differ")
    field_map = {
        "scene_index": "candidate_scene_index",
        "terminal_rank": "candidate_terminal_rank",
        "diagnostic_row": "candidate_diagnostic_row",
        "result_index": "candidate_result_index",
        "track_id": "candidate_track_id",
        "box_center_extent": "candidate_box_center_extent",
        "corners_world": "candidate_corners_world",
        "raw_score_provenance": "candidate_raw_score_provenance",
        "stored_appended_score_diagnostic_only": (
            "candidate_stored_appended_score_diagnostic_only"
        ),
        "formal_evaluation_score": "candidate_formal_evaluation_score",
        "max_native_aabb_iou": "candidate_max_native_aabb_iou",
        "valid_point_count": "candidate_valid_point_count",
    }
    for scene_index, scene in enumerate(scenes):
        row = _require_mapping(ledger[scene], f"scenes[{scene}]")
        if row.get("scene_index") != scene_index:
            raise S2OracleError(f"manifest scene index mismatch: {scene}")
        positions = np.flatnonzero(arrays["candidate_scene_index"] == scene_index)
        public = row.get("accepted_candidates")
        if not isinstance(public, list) or len(public) != len(positions):
            raise S2OracleError(f"accepted-candidate ledger mismatch: {scene}")
        if row.get("accepted_candidate_count") != len(positions):
            raise S2OracleError(f"accepted candidate count mismatch: {scene}")
        for position, candidate in zip(positions, public):
            candidate = _require_mapping(candidate, "accepted candidate")
            if candidate.get("scene_id") != scene:
                raise S2OracleError("accepted candidate scene ID mismatch")
            for json_name, array_name in field_map.items():
                expected = arrays[array_name][position]
                actual = candidate.get(json_name)
                if expected.ndim == 0:
                    if expected.dtype.kind in "f":
                        equal = np.float32(actual) == expected
                    else:
                        equal = actual == expected.item()
                else:
                    try:
                        actual_array = np.asarray(actual, dtype=expected.dtype)
                    except (TypeError, ValueError) as error:
                        raise S2OracleError(
                            f"invalid accepted candidate field: {json_name}"
                        ) from error
                    equal = actual_array.shape == expected.shape and np.array_equal(
                        actual_array, expected
                    )
                if not bool(equal):
                    raise S2OracleError(
                        f"manifest/NPZ accepted candidate mismatch for {json_name}"
                    )


def _load_sealed_shadow(
    json_path: Path, npz_path: Path
) -> tuple[dict[str, Any], dict[str, np.ndarray], tuple[str, ...], dict[str, str]]:
    """Fully validate the no-GT sidecar before any GT path is touched."""

    json_path = _regular_file(json_path, "S2 shadow manifest")
    npz_path = _regular_file(npz_path, "S2 shadow NPZ")
    initial_hashes = {"json": _sha256(json_path), "npz": _sha256(npz_path)}
    manifest = _read_json(json_path, "S2 shadow manifest")
    _validate_manifest_contract(manifest)
    if manifest.get("npz_file") != npz_path.name:
        raise S2OracleError("shadow NPZ filename mismatch")
    if manifest.get("npz_sha256") != initial_hashes["npz"]:
        raise S2OracleError("shadow NPZ SHA-256 mismatch")
    try:
        with np.load(npz_path, allow_pickle=False) as source:
            if set(source.files) != set(_EXPECTED_ARRAY_DTYPES):
                raise S2OracleError("unexpected S2 shadow NPZ schema")
            arrays = {name: np.array(source[name], copy=True) for name in source.files}
    except (OSError, ValueError) as error:
        if isinstance(error, S2OracleError):
            raise
        raise S2OracleError(f"invalid S2 shadow NPZ: {npz_path}") from error
    if _sha256(npz_path) != initial_hashes["npz"]:
        raise S2OracleError("shadow NPZ changed while it was loaded")
    for name, dtype in _EXPECTED_ARRAY_DTYPES.items():
        if arrays[name].dtype != dtype:
            raise S2OracleError(
                f"shadow array {name} dtype mismatch: "
                f"expected={dtype}, actual={arrays[name].dtype}"
            )
    scenes = tuple(str(value) for value in arrays["scene_ids"].tolist())
    if (
        not scenes
        or len(set(scenes)) != len(scenes)
        or any(SCENE_PATTERN.fullmatch(scene) is None for scene in scenes)
        or manifest.get("scene_count") != len(scenes)
        or manifest.get("scene_order") != list(scenes)
    ):
        raise S2OracleError("invalid sealed S2 scene order")
    count = manifest.get("candidate_count")
    if isinstance(count, bool) or not isinstance(count, int) or count < 0:
        raise S2OracleError("invalid sealed S2 candidate count")
    _validate_array_shapes_and_values(arrays, scenes, count)
    if manifest.get("candidate_content_sha256") != _array_content_sha256(arrays):
        raise S2OracleError("S2 candidate content SHA-256 mismatch")
    _validate_manifest_candidate_mirror(manifest, arrays, scenes)
    return manifest, arrays, scenes, initial_hashes


def _preflight_native_seal(
    *,
    manifest: Mapping[str, Any],
    arrays: Mapping[str, np.ndarray],
    scenes: tuple[str, ...],
    baseline_root: Path,
) -> tuple[dict[str, str], dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Validate native/counterfactual files before ground truth is opened."""

    inputs = _require_mapping(manifest["input"], "input")
    if Path(str(inputs["baseline_root"])).resolve() != baseline_root:
        raise S2OracleError("CLI baseline root differs from the sealed baseline root")
    ledger = _require_mapping(manifest["scenes"], "scenes")
    output_hashes = _require_mapping(
        manifest.get("counterfactual_prediction_sha256"),
        "counterfactual_prediction_sha256",
    )
    output_root = Path(str(manifest.get("counterfactual_prediction_root"))).resolve()
    if set(output_hashes) != set(scenes):
        raise S2OracleError("counterfactual output hash ledger differs from scene order")
    offsets = arrays["counterfactual_scene_offsets"]
    native_hashes: dict[str, str] = {}
    native_corners_by_scene: dict[str, np.ndarray] = {}
    native_scores_by_scene: dict[str, np.ndarray] = {}
    for scene_index, scene in enumerate(scenes):
        scene_ledger = _require_mapping(ledger[scene], f"scenes[{scene}]")
        native_path = _regular_file(
            baseline_root / f"{scene}_boxes.pkl", "sealed native T05 prediction"
        )
        native_hash = _sha256(native_path)
        native_hashes[scene] = native_hash
        for key in (
            "native_prediction_sha256_before",
            "native_prediction_sha256_after",
        ):
            if scene_ledger.get(key) != native_hash:
                raise S2OracleError(f"native T05 prediction differs from seal: {scene}")
        if scene_ledger.get("native_input_unchanged") is not True or scene_ledger.get(
            "native_prefix_exact"
        ) is not True:
            raise S2OracleError(f"native T05 identity was not proven: {scene}")
        native_rows, native_corners, native_scores = _load_prediction_rows(
            native_path, "sealed native T05 prediction"
        )
        native_corners_by_scene[scene] = native_corners
        native_scores_by_scene[scene] = native_scores
        native_count = len(native_rows)
        if scene_ledger.get("native_prefix_row_count") != native_count:
            raise S2OracleError(f"native T05 row count differs from seal: {scene}")
        prefix_digest = _row_payload_sha256(native_rows)
        if (
            scene_ledger.get("native_prefix_payload_sha256_input") != prefix_digest
            or scene_ledger.get("native_prefix_payload_sha256_output") != prefix_digest
        ):
            raise S2OracleError(f"native T05 row payload differs from seal: {scene}")

        positions = np.flatnonzero(arrays["candidate_scene_index"] == scene_index)
        lower, upper = int(offsets[scene_index]), int(offsets[scene_index + 1])
        expected_count = native_count + len(positions)
        if upper - lower != expected_count or scene_ledger.get(
            "counterfactual_row_count"
        ) != expected_count:
            raise S2OracleError(f"counterfactual row count mismatch: {scene}")
        flags = arrays["counterfactual_is_native_prefix"][lower:upper]
        expected_flags = np.concatenate(
            (np.ones(native_count, dtype=bool), np.zeros(len(positions), dtype=bool))
        )
        if not np.array_equal(flags, expected_flags):
            raise S2OracleError(f"native-prefix flags are invalid: {scene}")
        sealed_corners = arrays["counterfactual_corners_world"][lower:upper]
        expected_corners = np.concatenate(
            (native_corners, arrays["candidate_corners_world"][positions]), axis=0
        )
        if not np.array_equal(sealed_corners, expected_corners):
            raise S2OracleError(f"counterfactual corner order/content mismatch: {scene}")
        sealed_scores = arrays["counterfactual_stored_score_provenance"][lower:upper]
        expected_scores = np.concatenate(
            (
                native_scores,
                arrays["candidate_stored_appended_score_diagnostic_only"][positions],
            )
        ).astype(np.float32, copy=False)
        if not np.array_equal(sealed_scores, expected_scores):
            raise S2OracleError(f"counterfactual diagnostic scores mismatch: {scene}")
        if len(positions):
            recomputed = _aabb_iou_matrix(
                arrays["candidate_corners_world"][positions], native_corners
            )
            maxima = recomputed.max(axis=1) if native_count else np.zeros(len(positions))
            if not np.allclose(
                maxima,
                arrays["candidate_max_native_aabb_iou"][positions],
                rtol=0.0,
                atol=1e-6,
            ):
                raise S2OracleError(f"candidate/native IoU provenance mismatch: {scene}")
            if np.any(maxima >= NATIVE_NOVELTY_IOU):
                raise S2OracleError(f"candidate violates native novelty gate: {scene}")

        output_path = _regular_file(
            output_root / f"{scene}_boxes.pkl", "sealed counterfactual prediction"
        )
        output_hash = _sha256(output_path)
        if (
            output_hashes.get(scene) != output_hash
            or scene_ledger.get("counterfactual_prediction_sha256") != output_hash
        ):
            raise S2OracleError(f"counterfactual prediction hash mismatch: {scene}")
        output_rows, output_corners, output_scores = _load_prediction_rows(
            output_path, "sealed counterfactual prediction"
        )
        if len(output_rows) != expected_count:
            raise S2OracleError(f"counterfactual prediction row count mismatch: {scene}")
        if _row_payload_sha256(output_rows[:native_count]) != prefix_digest:
            raise S2OracleError(f"counterfactual native prefix is not exact: {scene}")
        if not np.array_equal(output_corners, expected_corners) or not np.array_equal(
            output_scores, expected_scores
        ):
            raise S2OracleError(f"counterfactual prediction differs from sidecar: {scene}")
    return native_hashes, native_corners_by_scene, native_scores_by_scene


def _threshold_key(threshold: float) -> str:
    return f"{threshold:.2f}"


def _json_evaluation(
    evaluation: Mapping[str, object], scenes: Sequence[str]
) -> dict[str, Any]:
    masks = evaluation["matched_gt_masks"]
    assert isinstance(masks, list)
    result = {
        key: value
        for key, value in evaluation.items()
        if key not in {"matched_gt_masks", "evaluation_order"}
    }
    result["per_scene"] = {
        scene: {
            "greedy_tp": int(np.count_nonzero(mask)),
            "matched_gt_indices": np.flatnonzero(mask).tolist(),
            "unmatched_gt_indices": np.flatnonzero(~mask).tolist(),
        }
        for scene, mask in zip(scenes, masks)
    }
    return result


def audit_scannet_s2_yoloe_direct_oracle(
    *,
    shadow_json: Path,
    shadow_npz: Path,
    baseline_root: Path,
    gt_root: Path,
    scan_root: Path,
) -> dict[str, Any]:
    """Evaluate the sealed fixed suffix without GT-dependent selection."""

    shadow_json = shadow_json.resolve()
    shadow_npz = shadow_npz.resolve()
    baseline_root = baseline_root.resolve()
    gt_root = gt_root.resolve()
    scan_root = scan_root.resolve()

    # This entire preflight intentionally precedes the first GT/axis path access.
    manifest, arrays, scenes, shadow_before = _load_sealed_shadow(
        shadow_json, shadow_npz
    )
    native_before, _, _ = _preflight_native_seal(
        manifest=manifest,
        arrays=arrays,
        scenes=scenes,
        baseline_root=baseline_root,
    )

    gt_counts: list[int] = []
    baseline_iou: list[np.ndarray] = []
    candidate_iou: list[np.ndarray] = []
    input_hashes: dict[str, Any] = {"scenes": {}}
    scene_reports: dict[str, Any] = {}
    for scene_index, scene in enumerate(scenes):
        gt_path = _regular_file(gt_root / f"{scene}_bbox.npy", "ScanNet GT")
        metadata_path = _regular_file(
            scan_root / scene / f"{scene}.txt", "ScanNet axis alignment"
        )
        alignment = load_axis_alignment(metadata_path)
        gt = load_gt_minmax(gt_path)
        _, baseline_aligned = load_baseline_boxes(
            baseline_root / f"{scene}_boxes.pkl", alignment
        )
        positions = np.flatnonzero(arrays["candidate_scene_index"] == scene_index)
        corners = arrays["candidate_corners_world"][positions].astype(np.float64)
        if len(corners):
            aligned_corners = corners @ alignment[:3, :3].T + alignment[:3, 3]
            candidate_aligned = np.concatenate(
                (aligned_corners.min(axis=1), aligned_corners.max(axis=1)), axis=1
            )
        else:
            candidate_aligned = np.empty((0, 6), dtype=np.float64)
        baseline_matrix = aligned_iou_matrix(baseline_aligned, gt)
        candidate_matrix = aligned_iou_matrix(candidate_aligned, gt)
        gt_counts.append(len(gt))
        baseline_iou.append(baseline_matrix)
        candidate_iou.append(candidate_matrix)
        scene_reports[scene] = {
            "gt_count": int(len(gt)),
            "native_prediction_count": int(len(baseline_aligned)),
            "fixed_candidate_count": int(len(candidate_aligned)),
            "candidate_global_indices": positions.astype(int).tolist(),
            "candidate_terminal_ranks": arrays["candidate_terminal_rank"][
                positions
            ].astype(int).tolist(),
            "candidate_track_ids": arrays["candidate_track_id"][positions]
            .astype(int)
            .tolist(),
        }
        input_hashes["scenes"][scene] = {
            "native_prediction": native_before[scene],
            "gt": _sha256(gt_path),
            "axis_alignment": _sha256(metadata_path),
        }

    baseline_eval = {
        threshold: official_constant_evaluate(baseline_iou, gt_counts, threshold)
        for threshold in THRESHOLDS
    }
    combined_iou = [
        np.concatenate((native, suffix), axis=0)
        for native, suffix in zip(baseline_iou, candidate_iou)
    ]
    combined_eval = {
        threshold: official_constant_evaluate(combined_iou, gt_counts, threshold)
        for threshold in THRESHOLDS
    }

    total_candidates = int(sum(len(matrix) for matrix in candidate_iou))
    total_gt = int(sum(gt_counts))
    threshold_reports: dict[str, Any] = {}
    strict_ap_pass: dict[str, bool] = {}
    union_pass: dict[str, bool] = {}
    for threshold in THRESHOLDS:
        key = _threshold_key(threshold)
        candidate_matching = 0
        native_matching = 0
        union_matching = 0
        recovered_official_unmatched = 0
        per_scene: dict[str, Any] = {}
        baseline_masks = baseline_eval[threshold]["matched_gt_masks"]
        assert isinstance(baseline_masks, list)
        for scene, native, suffix, official_matched in zip(
            scenes, baseline_iou, candidate_iou, baseline_masks
        ):
            candidate_pairs = strict_maximum_matching(suffix, threshold)
            native_pairs = strict_maximum_matching(native, threshold)
            union_pairs = strict_maximum_matching(
                np.concatenate((native, suffix), axis=0), threshold
            )
            recovered_pairs = strict_maximum_matching(
                suffix, threshold, ~np.asarray(official_matched, dtype=bool)
            )
            candidate_matching += len(candidate_pairs)
            native_matching += len(native_pairs)
            union_matching += len(union_pairs)
            recovered_official_unmatched += len(recovered_pairs)
            per_scene[scene] = {
                "candidate_maximum_matching_count": len(candidate_pairs),
                "candidate_maximum_matching_pairs": [list(pair) for pair in candidate_pairs],
                "native_maximum_matching_count": len(native_pairs),
                "native_union_maximum_matching_count": len(union_pairs),
                "additional_union_matching_over_native": len(union_pairs)
                - len(native_pairs),
                "official_baseline_unmatched_recovered_count": len(recovered_pairs),
                "official_baseline_unmatched_recovery_pairs": [
                    list(pair) for pair in recovered_pairs
                ],
            }
        baseline_ap = float(baseline_eval[threshold]["ap"])
        combined_ap = float(combined_eval[threshold]["ap"])
        delta_ap = 100.0 * (combined_ap - baseline_ap)
        additional_union = union_matching - native_matching
        strict_ap_pass[key] = delta_ap > 0.0
        union_pass[key] = additional_union >= 1
        threshold_reports[key] = {
            "iou_threshold": threshold,
            "candidate_maximum_matching_count": candidate_matching,
            "candidate_tp_precision_maximum_matching": (
                candidate_matching / total_candidates if total_candidates else 0.0
            ),
            "candidate_maximum_matching_recall": (
                candidate_matching / total_gt if total_gt else 0.0
            ),
            "native_maximum_matching_count": native_matching,
            "native_union_maximum_matching_count": union_matching,
            "additional_union_matching_over_native": additional_union,
            "incremental_recall_headroom_points": (
                100.0 * additional_union / total_gt if total_gt else 0.0
            ),
            "official_baseline_unmatched_recovered_count": recovered_official_unmatched,
            "baseline_t05_constant_score": _json_evaluation(
                baseline_eval[threshold], scenes
            ),
            "native_plus_fixed_suffix_constant_score": _json_evaluation(
                combined_eval[threshold], scenes
            ),
            "fixed_suffix_delta_ap_points": delta_ap,
            "fixed_suffix_delta_greedy_tp": int(combined_eval[threshold]["greedy_tp"])
            - int(baseline_eval[threshold]["greedy_tp"]),
            "fixed_suffix_delta_false_positive": int(
                combined_eval[threshold]["false_positive"]
            )
            - int(baseline_eval[threshold]["false_positive"]),
            "per_scene": per_scene,
        }

    passes_ap = all(strict_ap_pass.values())
    passes_union = all(union_pass.values())
    passes = passes_ap and passes_union
    promotion = {
        "preregistered": True,
        "active_birth_enabled": False,
        "requires_strictly_positive_constant_score_delta_ap_all_thresholds": True,
        "strictly_positive_delta_ap": strict_ap_pass,
        "passes_strictly_positive_delta_ap_all_thresholds": passes_ap,
        "requires_at_least_one_additional_union_match_all_thresholds": True,
        "additional_union_match": union_pass,
        "passes_additional_union_match_all_thresholds": passes_union,
        "passes_sealed_dev3_promotion_gate": passes,
        "decision": "promote_to_h10_shadow" if passes else "reject_s2_active_birth",
        "h10_gt_authorized_by_this_report": passes,
        "full100_authorized": False,
    }

    native_after = {
        scene: _sha256(baseline_root / f"{scene}_boxes.pkl") for scene in scenes
    }
    shadow_after = {"json": _sha256(shadow_json), "npz": _sha256(shadow_npz)}
    if native_after != native_before:
        raise S2OracleError("native T05 predictions changed during oracle")
    if shadow_after != shadow_before:
        raise S2OracleError("sealed S2 sidecar changed during oracle")
    return {
        "schema": SCHEMA,
        "oracle_only": True,
        "deployable_candidate_selection": True,
        "candidate_selection_used_gt": False,
        "candidate_suppression_used_gt": False,
        "candidate_ranking_used_gt": False,
        "evaluation_used_gt": True,
        "birth_enabled": False,
        "native_predictions_modified": False,
        "score_mode": "constant_1.0",
        "stored_scores_used_for_evaluation": False,
        "class_mode": "class_agnostic",
        "strict_iou_comparison": ">",
        "candidate_order": "sealed_scene_order_then_terminal_rank",
        "native_rows_are_on_disk_prefix": True,
        "official_tie_order": "numpy.argsort_default_all_scores_1.0",
        "scene_order": list(scenes),
        "thresholds": list(THRESHOLDS),
        "totals": {
            "scene_count": len(scenes),
            "gt_count": total_gt,
            "native_prediction_count": int(sum(len(matrix) for matrix in baseline_iou)),
            "fixed_candidate_count": total_candidates,
        },
        "per_threshold": threshold_reports,
        "promotion": promotion,
        "scenes": scene_reports,
        "shadow": {
            "json_path": os.fspath(shadow_json),
            "json_sha256": shadow_before["json"],
            "npz_path": os.fspath(shadow_npz),
            "npz_sha256": shadow_before["npz"],
            "schema": manifest["schema"],
            "candidate_content_sha256": manifest["candidate_content_sha256"],
            "preregistration": manifest["input"]["preregistration"],
            "preregistration_sha256": manifest["input"][
                "preregistration_sha256"
            ],
        },
        "input_sha256": input_hashes,
        "native_prediction_sha256_before": native_before,
        "native_prediction_sha256_after": native_after,
        "shadow_sha256_before": shadow_before,
        "shadow_sha256_after": shadow_after,
    }


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate one sealed S2 YOLOE-direct fixed candidate suffix"
    )
    parser.add_argument("--shadow-json", required=True, type=Path)
    parser.add_argument("--shadow-npz", required=True, type=Path)
    parser.add_argument("--baseline-root", required=True, type=Path)
    parser.add_argument("--gt-root", required=True, type=Path)
    parser.add_argument("--scan-root", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    out = args.out.resolve()
    if out.suffix.lower() != ".json":
        raise S2OracleError("oracle output must have a .json suffix")
    if out.exists() or out.is_symlink():
        raise S2OracleError(f"refusing to overwrite oracle output: {out}")
    protected_roots = (
        args.baseline_root.resolve(),
        args.gt_root.resolve(),
        args.scan_root.resolve(),
        args.shadow_json.resolve().parent,
    )
    if any(_is_relative_to(out, root) for root in protected_roots):
        raise S2OracleError("oracle output must be outside every protected input root")
    report = audit_scannet_s2_yoloe_direct_oracle(
        shadow_json=args.shadow_json,
        shadow_npz=args.shadow_npz,
        baseline_root=args.baseline_root,
        gt_root=args.gt_root,
        scan_root=args.scan_root,
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    print(
        json.dumps(
            {
                "schema": SCHEMA,
                "out": os.fspath(out),
                "totals": report["totals"],
                "promotion": report["promotion"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
