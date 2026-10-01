#!/usr/bin/env python3
"""Seal a causal three-view Boxer confirmation sidecar without changing T05.

The tool consumes only the already sealed per-view OWLv2+Boxer candidates,
camera poses, and terminal native T05 predictions.  Per-view candidates are
processed in chronological order by the transferred training-free geometry
tracker.  Native predictions are consulted only at terminal close for duplicate
suppression; they are never changed.  No ground-truth path is accepted.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import pickle
import re
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if os.fspath(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, os.fspath(_REPOSITORY_ROOT))

from tools.boxfusion_tr3d_pipeline.boxfusion.cutr_residual_birth_lite import (
    CuTRResidualBirthLite,
    ResidualBirthLiteConfig,
    ResidualObservation,
)


SCHEMA = "boxfusion.boxer_past3_shadow.v1"
INPUT_SCHEMA = "boxfusion.owl_boxer_shadow_candidates.v1"
TRACKER_SCHEMA = "boxfusion.cutr_residual_birth_lite_shadow.v1"
SCENE_PATTERN = re.compile(r"^scene[0-9]{4}_[0-9]{2}$")

MIN_CAMERA_BASELINE_M = 0.15
MIN_VIEW_RAY_SPAN_DEG = 10.0
SCORE_CEILING = 1.0
MAX_TRACKS = 1024
MAX_OBSERVATIONS_PER_FRAME = 64

_SIGNS = np.asarray(
    [
        [-1.0, -1.0, -1.0],
        [-1.0, -1.0, 1.0],
        [-1.0, 1.0, -1.0],
        [-1.0, 1.0, 1.0],
        [1.0, -1.0, -1.0],
        [1.0, -1.0, 1.0],
        [1.0, 1.0, -1.0],
        [1.0, 1.0, 1.0],
    ],
    dtype=np.float64,
)

_EXPECTED_ARRAYS = {
    "per_view_center_world",
    "per_view_extent_xyz",
    "per_view_frame_id",
    "per_view_quaternion_wxyz",
    "per_view_scene_index",
    "per_view_source_instance_id",
    "per_view_source_row",
    "per_view_source_score",
    "scene_ids",
    "tracked_center_world",
    "tracked_extent_xyz",
    "tracked_instance_id",
    "tracked_quaternion_wxyz",
    "tracked_scene_index",
    "tracked_source_row",
    "tracked_source_score",
}


class ShadowError(ValueError):
    """Raised when the frozen shadow contract is violated."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _regular_file(path: Path, label: str) -> Path:
    path = path.resolve()
    if path.is_symlink() or not path.is_file():
        raise ShadowError(f"{label} must be a regular non-symlink file: {path}")
    return path


def _load_json(path: Path, label: str) -> dict[str, Any]:
    _regular_file(path, label)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ShadowError(f"invalid {label}: {path}") from error
    if not isinstance(value, dict):
        raise ShadowError(f"{label} must contain a JSON object: {path}")
    return value


def _validate_input_manifest(
    manifest_path: Path, npz_path: Path
) -> tuple[dict[str, Any], tuple[str, ...]]:
    manifest = _load_json(manifest_path, "Boxer candidate seal")
    required = {
        "schema": INPUT_SCHEMA,
        "profile": "clean_in2",
        "mode": "shadow",
        "output_inert": True,
        "birth": False,
        "gt_access": False,
        "gt_access_guard_verified": True,
        "native_before_after_identity": True,
        "native_clip_unchanged": True,
        "semantic_source_exported": False,
        "coordinate_frame": "scannet_world",
    }
    for key, expected in required.items():
        if manifest.get(key) != expected:
            raise ShadowError(
                f"Boxer candidate seal mismatch for {key}: "
                f"expected={expected!r}, actual={manifest.get(key)!r}"
            )
    if manifest.get("npz_file") != npz_path.name:
        raise ShadowError("Boxer candidate NPZ filename does not match its seal")
    if manifest.get("npz_sha256") != _sha256(_regular_file(npz_path, "Boxer NPZ")):
        raise ShadowError("Boxer candidate NPZ SHA-256 mismatch")
    rows = manifest.get("scenes")
    if not isinstance(rows, list) or not rows:
        raise ShadowError("Boxer candidate scene ledger is empty")
    scenes: list[str] = []
    for scene_index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ShadowError("Boxer candidate scene ledger contains a non-object")
        scene = row.get("scene_id")
        if not isinstance(scene, str) or SCENE_PATTERN.fullmatch(scene) is None:
            raise ShadowError(f"invalid sealed scene ID: {scene!r}")
        if row.get("scene_index") != scene_index:
            raise ShadowError("Boxer candidate scene order/index mismatch")
        if row.get("gt_access_guard_verified") is not True:
            raise ShadowError(f"no-GT guard is unverified for {scene}")
        if row.get("per_view_extra_schedule_rows_excluded") != 0:
            raise ShadowError(f"off-schedule Boxer rows remain for {scene}")
        scenes.append(scene)
    if len(set(scenes)) != len(scenes):
        raise ShadowError("Boxer candidate scene IDs are not unique")
    if manifest.get("scene_count") != len(scenes):
        raise ShadowError("Boxer candidate scene count mismatch")
    return manifest, tuple(scenes)


def _load_sealed_scene_schedule(
    schedule_root: Path,
    scene: str,
    scene_ledger: Mapping[str, Any],
) -> tuple[tuple[int, ...], str]:
    path = schedule_root / scene / "manifest.json"
    manifest = _load_json(path, "sealed T05 schedule")
    if manifest.get("schema") != "boxfusion.cutr_postfilter_cache.v3":
        raise ShadowError(f"unexpected T05 schedule schema for {scene}")
    if manifest.get("scene_id") != scene:
        raise ShadowError(f"T05 schedule scene mismatch for {scene}")
    if scene_ledger.get("sealed_schedule_manifest_sha256") != _sha256(path):
        raise ShadowError(f"T05 schedule hash differs from Boxer seal for {scene}")
    raw = manifest.get("recorded_frame_ids")
    count = manifest.get("record_count")
    if (
        not isinstance(raw, list)
        or not raw
        or any(type(value) is not int or value < 0 for value in raw)
        or raw != sorted(raw)
        or len(set(raw)) != len(raw)
        or count != len(raw)
    ):
        raise ShadowError(f"invalid recorded T05 schedule for {scene}")
    mode = scene_ledger.get("sealed_schedule_mode")
    invalid = scene_ledger.get("sealed_schedule_invalid_pose_frame_ids_excluded")
    if not isinstance(invalid, list) or any(type(value) is not int for value in invalid):
        raise ShadowError(f"invalid pose-abstention ledger for {scene}")
    if mode == "valid_recorded_frames":
        invalid_set = set(invalid)
        schedule = tuple(value for value in raw if value not in invalid_set)
    elif mode == "legacy_record_count":
        if invalid:
            raise ShadowError(f"legacy schedule unexpectedly excludes poses for {scene}")
        schedule = tuple(raw)
    else:
        raise ShadowError(f"unknown Boxer sealed schedule mode for {scene}: {mode!r}")
    if len(schedule) != scene_ledger.get("sealed_schedule_frame_count"):
        raise ShadowError(f"sealed schedule count mismatch for {scene}")
    namespace = manifest.get("namespace")
    if not isinstance(namespace, str) or not namespace:
        raise ShadowError(f"invalid sealed schedule namespace for {scene}")
    return schedule, namespace


def _validate_arrays(
    npz_path: Path, scenes: Sequence[str], expected_count: int
) -> dict[str, np.ndarray]:
    try:
        with np.load(npz_path, allow_pickle=False) as source:
            if set(source.files) != _EXPECTED_ARRAYS:
                raise ShadowError("unexpected Boxer candidate NPZ schema")
            arrays = {name: np.array(source[name], copy=True) for name in source.files}
    except (OSError, ValueError) as error:
        if isinstance(error, ShadowError):
            raise
        raise ShadowError(f"invalid Boxer candidate NPZ: {npz_path}") from error
    if arrays["scene_ids"].tolist() != list(scenes):
        raise ShadowError("Boxer candidate NPZ scene order mismatch")
    count = len(arrays["per_view_scene_index"])
    if count != expected_count:
        raise ShadowError("Boxer candidate count differs from its JSON seal")
    shapes = {
        "per_view_center_world": (count, 3),
        "per_view_extent_xyz": (count, 3),
        "per_view_frame_id": (count,),
        "per_view_quaternion_wxyz": (count, 4),
        "per_view_scene_index": (count,),
        "per_view_source_instance_id": (count,),
        "per_view_source_row": (count,),
        "per_view_source_score": (count,),
    }
    for name, shape in shapes.items():
        if arrays[name].shape != shape:
            raise ShadowError(f"unexpected {name} shape: {arrays[name].shape}")
    numeric = np.concatenate(
        (
            arrays["per_view_center_world"].reshape(-1),
            arrays["per_view_extent_xyz"].reshape(-1),
            arrays["per_view_quaternion_wxyz"].reshape(-1),
            arrays["per_view_source_score"].reshape(-1),
        )
    )
    if not np.isfinite(numeric).all():
        raise ShadowError("Boxer per-view arrays contain non-finite values")
    if np.any(arrays["per_view_extent_xyz"] <= 0.0):
        raise ShadowError("Boxer per-view extents must be positive")
    if np.any(
        (arrays["per_view_source_score"] < 0.0)
        | (arrays["per_view_source_score"] >= SCORE_CEILING)
    ):
        raise ShadowError("Boxer source scores must be in [0,1)")
    if np.any(
        (arrays["per_view_scene_index"] < 0)
        | (arrays["per_view_scene_index"] >= len(scenes))
    ):
        raise ShadowError("Boxer candidate scene index is out of range")
    for scene_index in range(len(scenes)):
        positions = np.flatnonzero(arrays["per_view_scene_index"] == scene_index)
        frames = arrays["per_view_frame_id"][positions]
        rows = arrays["per_view_source_row"][positions]
        if len(np.unique(rows)) != len(rows):
            raise ShadowError(f"duplicate source rows in scene {scenes[scene_index]}")
        for frame_id in np.unique(frames):
            frame_rows = rows[frames == frame_id]
            if len(frame_rows) > MAX_OBSERVATIONS_PER_FRAME:
                raise ShadowError(
                    f"sealed per-frame candidate cap exceeded in {scenes[scene_index]}"
                )
    for value in arrays.values():
        value.setflags(write=False)
    return arrays


def _quaternion_rotation(quaternion: np.ndarray) -> np.ndarray:
    q = np.asarray(quaternion, dtype=np.float64)
    norm_squared = float(q @ q)
    if q.shape != (4,) or not math.isfinite(norm_squared) or norm_squared <= 1e-12:
        raise ShadowError("invalid Boxer quaternion")
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


def _obb_corners(center: np.ndarray, extent: np.ndarray, quaternion: np.ndarray) -> np.ndarray:
    rotation = _quaternion_rotation(quaternion)
    return _SIGNS * (np.asarray(extent, dtype=np.float64) / 2.0) @ rotation.T + center


def _load_prediction(path: Path) -> tuple[np.ndarray, np.ndarray]:
    _regular_file(path, "native T05 prediction")
    try:
        with path.open("rb") as handle:
            payload = pickle.load(handle)
    except Exception as error:
        raise ShadowError(f"could not load native prediction: {path}") from error
    if not isinstance(payload, (list, tuple)) or len(payload) != 1:
        raise ShadowError(f"invalid native prediction outer schema: {path}")
    rows = payload[0]
    if not isinstance(rows, (list, tuple)):
        raise ShadowError(f"invalid native prediction rows: {path}")
    corners: list[np.ndarray] = []
    scores: list[float] = []
    for row_index, row in enumerate(rows):
        if not isinstance(row, (list, tuple)) or len(row) != 3:
            raise ShadowError(f"invalid native row {row_index}: {path}")
        box = np.asarray(row[1], dtype=np.float64)
        score = float(row[2])
        if box.shape != (8, 3) or not np.isfinite(box).all():
            raise ShadowError(f"invalid native corners at row {row_index}: {path}")
        if not math.isfinite(score) or not 0.0 < score <= 1.0:
            raise ShadowError(f"invalid native score at row {row_index}: {path}")
        corners.append(box)
        scores.append(score)
    boxes = (
        np.stack(corners, axis=0)
        if corners
        else np.empty((0, 8, 3), dtype=np.float64)
    )
    return boxes, np.asarray(scores, dtype=np.float64)


def _load_camera_center(scene_root: Path, scene: str, frame_id: int) -> tuple[np.ndarray, str]:
    path = scene_root / scene / "frames" / "pose" / f"{frame_id}.txt"
    _regular_file(path, "sealed-frame camera pose")
    try:
        pose = np.loadtxt(path, dtype=np.float64)
    except (OSError, ValueError) as error:
        raise ShadowError(f"invalid camera pose: {path}") from error
    if pose.shape != (4, 4) or not np.isfinite(pose).all():
        raise ShadowError(f"invalid camera pose matrix: {path}")
    if np.max(np.abs(pose[3] - np.asarray([0.0, 0.0, 0.0, 1.0]))) > 1e-5:
        raise ShadowError(f"invalid homogeneous camera pose row: {path}")
    return np.ascontiguousarray(pose[:3, 3]), _sha256(path)


def _view_diversity(
    centers: np.ndarray, candidate_center: np.ndarray
) -> tuple[float, float]:
    centers = np.asarray(centers, dtype=np.float64)
    if centers.ndim != 2 or centers.shape[1:] != (3,) or len(centers) < 3:
        raise ShadowError("a confirmed candidate must have at least three camera centers")
    pairwise_baseline = np.linalg.norm(
        centers[:, None, :] - centers[None, :, :], axis=2
    )
    rays = np.asarray(candidate_center, dtype=np.float64)[None, :] - centers
    norms = np.linalg.norm(rays, axis=1)
    if np.any(norms <= 1e-6):
        raise ShadowError("candidate center coincides with an evidence camera")
    rays /= norms[:, None]
    angles = np.degrees(np.arccos(np.clip(rays @ rays.T, -1.0, 1.0)))
    return float(pairwise_baseline.max()), float(angles.max())


def _array_content_sha256(arrays: Mapping[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for name in sorted(arrays):
        array = np.ascontiguousarray(arrays[name])
        digest.update(name.encode("ascii"))
        digest.update(b"\0")
        digest.update(array.dtype.str.encode("ascii"))
        digest.update(b"\0")
        digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode("ascii"))
        digest.update(b"\0")
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _write_deterministic_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    with path.open("xb") as raw:
        with zipfile.ZipFile(
            raw, mode="w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
        ) as archive:
            for name in sorted(arrays):
                payload = io.BytesIO()
                np.lib.format.write_array(
                    payload, np.ascontiguousarray(arrays[name]), allow_pickle=False
                )
                info = zipfile.ZipInfo(
                    f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0)
                )
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = 0o600 << 16
                archive.writestr(
                    info,
                    payload.getvalue(),
                    compress_type=zipfile.ZIP_DEFLATED,
                    compresslevel=9,
                )
        raw.flush()
        os.fsync(raw.fileno())


def _write_json_exclusive(path: Path, value: Mapping[str, Any]) -> None:
    payload = (
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    )
    with path.open("x", encoding="utf-8") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def materialize_boxer_past3_shadow(
    *,
    input_json: Path,
    input_npz: Path,
    baseline_root: Path,
    schedule_root: Path,
    scene_rgbd_root: Path,
    preregistration: Path,
    output_json: Path,
    output_npz: Path,
) -> dict[str, Any]:
    """Materialize one deterministic, output-inert Boxer-Past3 sidecar."""

    input_json = input_json.resolve()
    input_npz = input_npz.resolve()
    baseline_root = baseline_root.resolve()
    schedule_root = schedule_root.resolve()
    scene_rgbd_root = scene_rgbd_root.resolve()
    preregistration = preregistration.resolve()
    output_json = output_json.resolve()
    output_npz = output_npz.resolve()
    if output_json == output_npz or output_json.parent != output_npz.parent:
        raise ShadowError("output JSON and NPZ must be distinct files in one directory")
    if output_json.exists() or output_json.is_symlink():
        raise ShadowError(f"refusing to overwrite shadow JSON: {output_json}")
    if output_npz.exists() or output_npz.is_symlink():
        raise ShadowError(f"refusing to overwrite shadow NPZ: {output_npz}")
    if (
        not baseline_root.is_dir()
        or not schedule_root.is_dir()
        or not scene_rgbd_root.is_dir()
    ):
        raise ShadowError(
            "baseline, sealed schedule and ScanNet RGB-D roots must be directories"
        )
    _regular_file(preregistration, "Boxer-Past3 preregistration")

    input_manifest, scenes = _validate_input_manifest(input_json, input_npz)
    arrays = _validate_arrays(
        input_npz, scenes, int(input_manifest["per_view_candidate_count"])
    )
    tracker_source = Path(__file__).resolve().parent.parent / (
        "tools/boxfusion_tr3d_pipeline/boxfusion/cutr_residual_birth_lite.py"
    )
    _regular_file(tracker_source, "transferred causal tracker source")

    native_before: dict[str, str] = {}
    native_after: dict[str, str] = {}
    scene_reports: dict[str, Any] = {}
    accepted_rows: list[dict[str, Any]] = []
    pose_hashes: dict[str, dict[str, str]] = {}
    schedule_hashes: dict[str, str] = {}
    schedule_namespace: str | None = None

    for scene_index, scene in enumerate(scenes):
        scene_ledger = input_manifest["scenes"][scene_index]
        schedule, namespace = _load_sealed_scene_schedule(
            schedule_root, scene, scene_ledger
        )
        schedule_hashes[scene] = _sha256(schedule_root / scene / "manifest.json")
        if schedule_namespace is None:
            schedule_namespace = namespace
        elif schedule_namespace != namespace:
            raise ShadowError("sealed T05 schedule namespace changes across scenes")
        prediction_path = baseline_root / f"{scene}_boxes.pkl"
        native_before[scene] = _sha256(_regular_file(prediction_path, "T05 prediction"))
        native_corners, native_scores = _load_prediction(prediction_path)
        positions = np.flatnonzero(arrays["per_view_scene_index"] == scene_index)
        tracker = CuTRResidualBirthLite(
            ResidualBirthLiteConfig(
                enabled=True,
                observer_only=True,
                score_ceiling=SCORE_CEILING,
                max_tracks=MAX_TRACKS,
                max_observations_per_frame=MAX_OBSERVATIONS_PER_FRAME,
            )
        )
        confirmation_frame: dict[int, int] = {}
        observed_frame_ids = set(
            int(value) for value in arrays["per_view_frame_id"][positions]
        )
        if not observed_frame_ids.issubset(schedule):
            raise ShadowError(f"off-schedule Boxer candidate reached tracker for {scene}")
        for frame_id in schedule:
            frame_positions = positions[
                arrays["per_view_frame_id"][positions] == frame_id
            ]
            observations = [
                ResidualObservation(
                    frame_id=frame_id,
                    raw_index=int(arrays["per_view_source_row"][row]),
                    score=float(arrays["per_view_source_score"][row]),
                    corners=_obb_corners(
                        arrays["per_view_center_world"][row],
                        arrays["per_view_extent_xyz"][row],
                        arrays["per_view_quaternion_wxyz"][row],
                    ),
                )
                for row in frame_positions
            ]
            batch = tracker.observe(frame_id, observations)
            for track_id in batch.newly_confirmed_track_ids:
                if track_id in confirmation_frame:
                    raise ShadowError("a Boxer track was confirmed more than once")
                confirmation_frame[track_id] = frame_id

        close_result = tracker.close(native_corners, native_scores)
        summary = tracker.summary()
        if summary.get("schema") != TRACKER_SCHEMA or summary.get("gt_access") is not False:
            raise ShadowError("transferred tracker violated its frozen no-GT contract")
        if not close_result.audit_complete:
            raise ShadowError(f"incomplete bounded tracker audit for {scene}")

        view_rejections: list[dict[str, Any]] = []
        accepted_scene: list[dict[str, Any]] = []
        scene_pose_hashes: dict[str, str] = {}
        for candidate in close_result.candidates:
            if candidate.track_id not in confirmation_frame:
                raise ShadowError("terminal Boxer candidate lacks a causal confirmation frame")
            evidence_frames = tuple(int(value) for value in candidate.evidence_frame_ids)
            if len(set(evidence_frames)) < 3 or tuple(sorted(evidence_frames)) != evidence_frames:
                raise ShadowError("terminal Boxer candidate lacks three ordered evidence frames")
            camera_centers = []
            for frame_id in evidence_frames:
                center, digest = _load_camera_center(
                    scene_rgbd_root, scene, frame_id
                )
                camera_centers.append(center)
                scene_pose_hashes[str(frame_id)] = digest
            baseline_m, ray_span_deg = _view_diversity(
                np.stack(camera_centers), candidate.corners.mean(axis=0)
            )
            reasons = []
            if baseline_m < MIN_CAMERA_BASELINE_M:
                reasons.append("camera_baseline_below_0.15m")
            if ray_span_deg < MIN_VIEW_RAY_SPAN_DEG:
                reasons.append("view_ray_span_below_10deg")
            row = {
                "scene_id": scene,
                "scene_index": scene_index,
                "track_id": int(candidate.track_id),
                "confirmation_frame_id": int(confirmation_frame[candidate.track_id]),
                "evidence_frame_ids": list(evidence_frames),
                "evidence_source_rows": [
                    int(value) for value in candidate.evidence_raw_indices
                ],
                "raw_mean_score": float(candidate.raw_mean_score),
                "appended_score_diagnostic_only": float(candidate.appended_score),
                "median_pairwise_aabb_iou": float(candidate.median_pairwise_iou),
                "center_rms_m": float(candidate.center_rms_m),
                "max_terminal_native_aabb_iou": float(candidate.max_native_iou),
                "max_camera_baseline_m": baseline_m,
                "max_view_ray_span_deg": ray_span_deg,
                "corners_world": candidate.corners.tolist(),
                "view_gate_pass": not reasons,
                "view_gate_reasons": reasons,
            }
            if reasons:
                view_rejections.append(row)
            else:
                accepted_scene.append(row)
                accepted_rows.append(row)
        pose_hashes[scene] = dict(sorted(scene_pose_hashes.items(), key=lambda item: int(item[0])))
        scene_reports[scene] = {
            "raw_per_view_candidates": int(len(positions)),
            "processed_keyframes": int(len(schedule)),
            "nonempty_candidate_keyframes": int(len(observed_frame_ids)),
            "zero_candidate_keyframes": int(len(schedule) - len(observed_frame_ids)),
            "sealed_schedule_namespace": namespace,
            "native_terminal_predictions": int(len(native_corners)),
            "causally_confirmed_tracks": int(summary["confirmed_tracks"]),
            "pre_view_gate_terminal_candidates": int(len(close_result.candidates)),
            "view_gate_accepted_candidates": int(len(accepted_scene)),
            "view_gate_rejected_candidates": int(len(view_rejections)),
            "accepted_candidates": accepted_scene,
            "view_gate_rejections": view_rejections,
            "tracker_summary": summary,
        }

    native_after = {
        scene: _sha256(baseline_root / f"{scene}_boxes.pkl") for scene in scenes
    }
    if native_after != native_before:
        raise ShadowError("native T05 predictions changed during shadow materialization")

    evidence_offsets = [0]
    evidence_frames: list[int] = []
    evidence_rows: list[int] = []
    for row in accepted_rows:
        evidence_frames.extend(row["evidence_frame_ids"])
        evidence_rows.extend(row["evidence_source_rows"])
        evidence_offsets.append(len(evidence_frames))
    candidate_arrays: dict[str, np.ndarray] = {
        "scene_ids": np.asarray(scenes, dtype="<U12"),
        "candidate_scene_index": np.asarray(
            [row["scene_index"] for row in accepted_rows], dtype=np.int16
        ),
        "candidate_track_id": np.asarray(
            [row["track_id"] for row in accepted_rows], dtype=np.int32
        ),
        "candidate_confirmation_frame_id": np.asarray(
            [row["confirmation_frame_id"] for row in accepted_rows], dtype=np.int64
        ),
        "candidate_corners_world": (
            np.asarray(
                [row["corners_world"] for row in accepted_rows], dtype=np.float32
            ).reshape((-1, 8, 3))
        ),
        "candidate_raw_mean_score": np.asarray(
            [row["raw_mean_score"] for row in accepted_rows], dtype=np.float32
        ),
        "candidate_appended_score_diagnostic_only": np.asarray(
            [row["appended_score_diagnostic_only"] for row in accepted_rows],
            dtype=np.float32,
        ),
        "candidate_median_pairwise_aabb_iou": np.asarray(
            [row["median_pairwise_aabb_iou"] for row in accepted_rows],
            dtype=np.float32,
        ),
        "candidate_center_rms_m": np.asarray(
            [row["center_rms_m"] for row in accepted_rows], dtype=np.float32
        ),
        "candidate_max_terminal_native_aabb_iou": np.asarray(
            [row["max_terminal_native_aabb_iou"] for row in accepted_rows],
            dtype=np.float32,
        ),
        "candidate_max_camera_baseline_m": np.asarray(
            [row["max_camera_baseline_m"] for row in accepted_rows], dtype=np.float32
        ),
        "candidate_max_view_ray_span_deg": np.asarray(
            [row["max_view_ray_span_deg"] for row in accepted_rows], dtype=np.float32
        ),
        "candidate_evidence_offsets": np.asarray(evidence_offsets, dtype=np.int32),
        "evidence_frame_id": np.asarray(evidence_frames, dtype=np.int64),
        "evidence_source_row": np.asarray(evidence_rows, dtype=np.int32),
    }
    for value in candidate_arrays.values():
        value.setflags(write=False)

    output_json.parent.mkdir(parents=True, exist_ok=True)
    _write_deterministic_npz(output_npz, candidate_arrays)
    manifest: dict[str, Any] = {
        "schema": SCHEMA,
        "mode": "shadow",
        "output_inert": True,
        "birth": False,
        "active_authorized": False,
        "native_mutation_applied": False,
        "gt_access": False,
        "training_free": True,
        "online_learning": False,
        "past_only_association": True,
        "terminal_native_novelty_only": True,
        "future_frames_used": False,
        "tracked_boxer_pool_used": False,
        "detector_semantics_used": False,
        "detector_score_used_for_ranking_only": True,
        "native_clip_access": False,
        "native_clip_unchanged": True,
        "score_mode_for_formal_evaluation": "constant_1.0",
        "coordinate_frame": "scannet_world",
        "scene_count": len(scenes),
        "candidate_count": len(accepted_rows),
        "npz_file": output_npz.name,
        "npz_sha256": _sha256(output_npz),
        "candidate_content_sha256": _array_content_sha256(candidate_arrays),
        "input": {
            "candidate_json": os.fspath(input_json),
            "candidate_json_sha256": _sha256(input_json),
            "candidate_npz": os.fspath(input_npz),
            "candidate_npz_sha256": _sha256(input_npz),
            "candidate_schema": INPUT_SCHEMA,
            "candidate_content_sha256": input_manifest.get(
                "candidate_content_sha256"
            ),
            "preregistration": os.fspath(preregistration),
            "preregistration_sha256": _sha256(preregistration),
            "tracker_source": os.fspath(tracker_source),
            "tracker_source_sha256": _sha256(tracker_source),
            "baseline_root": os.fspath(baseline_root),
            "schedule_root": os.fspath(schedule_root),
            "schedule_sha256": schedule_hashes,
            "schedule_namespace": schedule_namespace,
            "scene_rgbd_root": os.fspath(scene_rgbd_root),
            "pose_sha256": pose_hashes,
        },
        "frozen_policy": {
            "transferred_tracker_schema": TRACKER_SCHEMA,
            "score_ceiling": SCORE_CEILING,
            "max_tracks": MAX_TRACKS,
            "max_observations_per_frame": MAX_OBSERVATIONS_PER_FRAME,
            "minimum_distinct_evidence_frames": 3,
            "minimum_camera_baseline_m": MIN_CAMERA_BASELINE_M,
            "minimum_view_ray_span_deg": MIN_VIEW_RAY_SPAN_DEG,
            "terminal_native_novelty_is_post_confirmation": True,
        },
        "native_prediction_sha256_before": native_before,
        "native_prediction_sha256_after": native_after,
        "native_before_after_identity": native_before == native_after,
        "scenes": scene_reports,
    }
    try:
        _write_json_exclusive(output_json, manifest)
    except Exception:
        # The NPZ was newly created by this invocation and no manifest can seal
        # it.  Remove only that exact output so a retry cannot mistake it for a
        # valid sidecar.
        output_npz.unlink(missing_ok=True)
        raise
    return manifest


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Seal a no-GT, output-inert Boxer past-three-view sidecar"
    )
    parser.add_argument("--input-json", required=True, type=Path)
    parser.add_argument("--input-npz", required=True, type=Path)
    parser.add_argument("--baseline-root", required=True, type=Path)
    parser.add_argument("--schedule-root", required=True, type=Path)
    parser.add_argument("--scene-rgbd-root", required=True, type=Path)
    parser.add_argument("--preregistration", required=True, type=Path)
    parser.add_argument("--out-json", required=True, type=Path)
    parser.add_argument("--out-npz", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    manifest = materialize_boxer_past3_shadow(
        input_json=args.input_json,
        input_npz=args.input_npz,
        baseline_root=args.baseline_root,
        schedule_root=args.schedule_root,
        scene_rgbd_root=args.scene_rgbd_root,
        preregistration=args.preregistration,
        output_json=args.out_json,
        output_npz=args.out_npz,
    )
    print(
        json.dumps(
            {
                "schema": SCHEMA,
                "scene_count": manifest["scene_count"],
                "candidate_count": manifest["candidate_count"],
                "out_json": os.fspath(args.out_json),
                "out_npz": os.fspath(args.out_npz),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
