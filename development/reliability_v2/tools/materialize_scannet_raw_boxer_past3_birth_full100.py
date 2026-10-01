#!/usr/bin/env python3
"""Materialize a fixed, no-GT Raw-Boxer Past3 birth suffix for ScanNet.

The input Raw Boxer workers publish a completion-ledger row only after a scene
CSV has been closed and validated.  This tool waits for every requested ledger
row before reading any candidate CSV, replays the exact valid T05 keyframe
schedule (including empty keyframes), and freezes the IoU medoid of the exact
first three associated views using :mod:`boxfusion.s3r_receipt_tracker`.

At terminal close, receipts are compared only with the frozen native T05 boxes.
Native-overlapping receipts are rejected, remaining receipts undergo fixed
class-agnostic self-NMS and a per-policy birth cap, and accepted rows are
appended to the byte-identical native row prefix.  The ScanNet evaluator is
class agnostic, so suffix labels are the inert integer ``0``; the formal
constant-score protocol sets every suffix score to ``1.0``.  No annotation,
evaluator, training, RGB image, or depth image is read by this program.  The
optional v3 policy consumes only a sealed, no-GT frozen-CLIP decision sidecar
joined by scene, track, and exact causal source rows.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import pickle
import re
import shutil
import struct
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if os.fspath(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, os.fspath(REPOSITORY_ROOT))

from boxfusion.s3r_receipt_tracker import (  # noqa: E402
    S3RObservation,
    S3RReceipt,
    S3RReceiptTracker,
)


SCHEMA = "boxfusion.scannet_raw_boxer_past3_birth_full100.v1"
V2_SCHEMA = "boxfusion.scannet_raw_boxer_past3_birth_full100.v2_m50"
V3_SCHEMA = "boxfusion.scannet_raw_boxer_past3_birth_full100.v3_clip_vocab"
CLIP_GATE_SIDECAR_SCHEMA = (
    "boxfusion.scannet_raw_boxer_clip_vocab_shadow_full100.v1"
)
SCENE_PATTERN = re.compile(r"^scene[0-9]{4}_[0-9]{2}$")
PREDICTION_SUFFIX = "_boxes.pkl"
CSV_RELATIVE = Path("boxer_raw")
CSV_NAME = "boxer_3dbbs.csv"

# Frozen, training-free terminal selection policy.
NATIVE_NOVELTY_AABB_IOU = 0.10
SELF_NMS_AABB_IOU = 0.25
MAX_BIRTHS_PER_SCENE = 6
APPENDED_CLASS_ID = 0
APPENDED_SCORE = 1.0
TOP_K_PER_FRAME = 8
MIN_STABLE_MEDIAN_PAIRWISE_AABB_IOU = 0.25
MAX_STABLE_CENTER_RMS_M = 0.25
MIN_MEDOID_AABB_EXTENT_M = 0.30
MIN_CAMERA_BASELINE_M = 0.15
MIN_VIEW_RAY_SPAN_DEG = 10.0

# Frozen birth-v2-M50 admission policy.  These gates consume only the three
# causal receipt rows, the corresponding camera poses, and the terminal native
# prediction snapshot.  In particular, they have no annotation/evaluator API.
V2_MIN_EVIDENCE_SCORE = 0.45
V2_MIN_MEAN_SCORE = 0.55
V2_MIN_PAIRWISE_AABB_IOU = 0.35
V2_MAX_PAIRWISE_CENTER_DISTANCE_M = 0.25
V2_MIN_FIRST_LAST_FRAME_SPAN = 50
V2_MIN_CAMERA_BASELINE_M = 0.15
V2_MIN_VIEW_RAY_SPAN_DEG = 10.0
V2_MIN_MEDOID_AABB_EXTENT_M = 0.20
V2_NATIVE_NOVELTY_AABB_IOU = 0.10
V2_NATIVE_MAX_BIDIRECTIONAL_CONTAINMENT = 0.50
V2_SELF_NMS_AABB_IOU = 0.15
V2_SELF_NMS_BIDIRECTIONAL_CONTAINMENT = 0.25
V2_MAX_BIRTHS_PER_SCENE = 2

RAW_COLUMNS = (
    "time_ns",
    "tx_world_object",
    "ty_world_object",
    "tz_world_object",
    "qw_world_object",
    "qx_world_object",
    "qy_world_object",
    "qz_world_object",
    "scale_x",
    "scale_y",
    "scale_z",
    "name",
    "instance",
    "sem_id",
    "prob",
)

SIGNS = np.asarray(
    [
        (-1.0, -1.0, -1.0),
        (-1.0, -1.0, +1.0),
        (-1.0, +1.0, -1.0),
        (-1.0, +1.0, +1.0),
        (+1.0, -1.0, -1.0),
        (+1.0, -1.0, +1.0),
        (+1.0, +1.0, -1.0),
        (+1.0, +1.0, +1.0),
    ],
    dtype=np.float64,
)


class BirthMaterializationError(ValueError):
    """Raised when an input or frozen-policy invariant is violated."""


@dataclass(frozen=True)
class CompletionRow:
    scene: str
    manifest_keyframes: int
    valid_keyframes: int
    invalid_pose_keyframes: int
    raw_candidate_frames: int
    raw_candidates: int
    ledger_path: Path


@dataclass(frozen=True)
class NativePrediction:
    payload: list[Any] | tuple[Any, ...]
    rows: list[Any] | tuple[Any, ...]
    corners: np.ndarray


@dataclass(frozen=True)
class ClipGateSidecar:
    """Validated no-GT CLIP decisions indexed by causal receipt identity."""

    path: Path
    sha256: str
    schema: str | None
    records: dict[str, dict[tuple[int, tuple[int, int, int]], dict[str, Any]]]
    record_count: int


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _regular_file(path: Path, label: str) -> Path:
    if path.is_symlink() or not path.is_file():
        raise BirthMaterializationError(
            f"{label} must be a regular non-symlink file: {path}"
        )
    return path


def _read_json(path: Path, label: str) -> dict[str, Any]:
    _regular_file(path, label)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise BirthMaterializationError(f"invalid {label}: {path}") from error
    if not isinstance(value, dict):
        raise BirthMaterializationError(f"{label} must contain a JSON object: {path}")
    return value


def _load_clip_gate_sidecar(path: Path) -> ClipGateSidecar:
    """Load a strict no-GT sidecar without depending on one writer layout.

    Supported layouts are ``{"scenes": {scene: {"tracks": [...]}}}``,
    ``{"scenes": {scene: [...]}}``, and a top-level list of records carrying a
    ``scene`` field.  A ``tracks`` object keyed by track id is also accepted.
    The actual join is always the full causal identity
    ``(scene, track_id, evidence_source_rows)``; a track id by itself is never
    sufficient.
    """

    _regular_file(path, "CLIP gate sidecar")

    def _reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant {value}")

    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"), parse_constant=_reject_constant
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise BirthMaterializationError(f"invalid CLIP gate sidecar: {path}") from error
    if not isinstance(payload, (dict, list)):
        raise BirthMaterializationError(
            "CLIP gate sidecar must contain a JSON object or list"
        )
    metadata = payload if isinstance(payload, dict) else {}
    schema = metadata.get("schema")
    if schema is not None and schema != CLIP_GATE_SIDECAR_SCHEMA:
        raise BirthMaterializationError(
            f"unrecognized CLIP gate sidecar schema: {schema!r}"
        )
    contracts = metadata.get("contracts", {})
    if not isinstance(contracts, dict):
        raise BirthMaterializationError("CLIP gate contracts must be an object")
    if schema == CLIP_GATE_SIDECAR_SCHEMA:
        for key in ("gt_access", "evaluator_access"):
            if contracts.get(key) is not False:
                raise BirthMaterializationError(
                    f"recognized CLIP runner sidecar must declare contracts.{key}=false"
                )
    for key in ("gt_access", "evaluator_access"):
        declarations = []
        if key in metadata:
            declarations.append(metadata[key])
        if key in contracts:
            declarations.append(contracts[key])
        if any(value is not False for value in declarations):
            raise BirthMaterializationError(
                f"CLIP gate sidecar must not declare {key} other than false"
            )
    for key in ("annotation_path_argument", "target_dataset_training"):
        if metadata.get(key) is True:
            raise BirthMaterializationError(
                f"CLIP gate sidecar declares forbidden {key}=true"
            )

    collected: list[tuple[str | None, object | None, Mapping[str, Any]]] = []

    def _collect_scene(scene_hint: str | None, node: object) -> None:
        if isinstance(node, list):
            for record in node:
                if not isinstance(record, dict):
                    raise BirthMaterializationError(
                        "CLIP gate track lists may contain only JSON objects"
                    )
                if "tracks" in record:
                    nested_scene = record.get("scene", scene_hint)
                    if nested_scene is not None and not isinstance(nested_scene, str):
                        raise BirthMaterializationError("invalid sidecar scene field")
                    _collect_scene(nested_scene, record["tracks"])
                else:
                    collected.append((scene_hint, None, record))
            return
        if not isinstance(node, dict):
            raise BirthMaterializationError(
                "CLIP gate scene payload must be a track list or object"
            )
        if "tracks" in node:
            _collect_scene(scene_hint, node["tracks"])
            return
        # A dict carrying the record fields is one record; otherwise interpret
        # it as an object keyed by track id.
        if "gate_pass" in node or "evidence_source_rows" in node:
            collected.append((scene_hint, None, node))
            return
        for track_hint, record in node.items():
            if not isinstance(record, dict):
                raise BirthMaterializationError(
                    "CLIP gate track mapping values must be JSON objects"
                )
            collected.append((scene_hint, track_hint, record))

    if isinstance(payload, list):
        _collect_scene(None, payload)
    elif "scenes" in payload:
        scene_payload = payload["scenes"]
        if isinstance(scene_payload, dict):
            for scene, node in scene_payload.items():
                if not isinstance(scene, str):
                    raise BirthMaterializationError("invalid sidecar scene key")
                _collect_scene(scene, node)
        elif isinstance(scene_payload, list):
            _collect_scene(None, scene_payload)
        else:
            raise BirthMaterializationError(
                "CLIP gate scenes must be an object or list"
            )
    elif "tracks" in payload:
        _collect_scene(None, payload["tracks"])
    else:
        raise BirthMaterializationError(
            "CLIP gate sidecar has no scenes or tracks payload"
        )

    records: dict[
        str, dict[tuple[int, tuple[int, int, int]], dict[str, Any]]
    ] = {}
    for scene_hint, track_hint, record in collected:
        scene = record.get("scene", scene_hint)
        if not isinstance(scene, str) or SCENE_PATTERN.fullmatch(scene) is None:
            raise BirthMaterializationError("CLIP gate record has invalid scene")
        raw_track_id = record.get("track_id", track_hint)
        track_id = _nonnegative_int(raw_track_id, "CLIP gate track_id")
        raw_rows = record.get("evidence_source_rows")
        if not isinstance(raw_rows, list) or len(raw_rows) != 3:
            raise BirthMaterializationError(
                "CLIP gate evidence_source_rows must be a length-three list"
            )
        evidence_rows = tuple(
            _nonnegative_int(value, "CLIP gate evidence_source_row")
            for value in raw_rows
        )
        direct_gate_pass = record.get("gate_pass")
        clip_summary = record.get("clip_summary")
        if clip_summary is not None and not isinstance(clip_summary, dict):
            raise BirthMaterializationError(
                "CLIP gate clip_summary must be an object"
            )
        nested_gate_pass = (
            clip_summary.get("gate_pass")
            if isinstance(clip_summary, dict)
            else None
        )
        if direct_gate_pass is not None and type(direct_gate_pass) is not bool:
            raise BirthMaterializationError(
                "CLIP gate gate_pass must be a JSON boolean"
            )
        if nested_gate_pass is not None and type(nested_gate_pass) is not bool:
            raise BirthMaterializationError(
                "CLIP gate clip_summary.gate_pass must be a JSON boolean"
            )
        if direct_gate_pass is None and nested_gate_pass is None:
            raise BirthMaterializationError(
                "CLIP gate record has no gate_pass decision"
            )
        if (
            direct_gate_pass is not None
            and nested_gate_pass is not None
            and direct_gate_pass is not nested_gate_pass
        ):
            raise BirthMaterializationError(
                "CLIP gate top-level and clip_summary gate_pass disagree"
            )
        gate_pass = (
            direct_gate_pass
            if direct_gate_pass is not None
            else nested_gate_pass
        )
        key = (track_id, evidence_rows)
        scene_records = records.setdefault(scene, {})
        if key in scene_records:
            raise BirthMaterializationError(
                f"duplicate CLIP gate receipt identity: {scene}/{track_id}/{evidence_rows}"
            )
        # Round-trip through JSON to detach the provenance object and retain
        # every writer-supplied field in the final audit manifest.
        normalized_record = json.loads(
            json.dumps(record, sort_keys=True, allow_nan=False)
        )
        normalized_record["gate_pass"] = gate_pass
        scene_records[key] = normalized_record
    return ClipGateSidecar(
        path=path.resolve(),
        sha256=_sha256(path),
        schema=schema if isinstance(schema, str) else None,
        records=records,
        record_count=sum(len(value) for value in records.values()),
    )


def _scene_list(path: Path, expected_scene_count: int) -> tuple[str, ...]:
    _regular_file(path, "scene list")
    scenes = tuple(
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    )
    if len(scenes) != expected_scene_count:
        raise BirthMaterializationError(
            f"scene list must contain {expected_scene_count} scenes; found {len(scenes)}"
        )
    if len(set(scenes)) != len(scenes):
        raise BirthMaterializationError("scene list contains duplicate IDs")
    invalid = [scene for scene in scenes if SCENE_PATTERN.fullmatch(scene) is None]
    if invalid:
        raise BirthMaterializationError(f"invalid scene IDs: {invalid[:3]}")
    return scenes


def _nonnegative_int(value: object, label: str) -> int:
    try:
        normalized = int(str(value))
    except (TypeError, ValueError) as error:
        raise BirthMaterializationError(f"{label} must be an integer") from error
    if normalized < 0 or str(value).strip() != str(normalized):
        raise BirthMaterializationError(f"{label} must be a canonical nonnegative integer")
    return normalized


def _completion_rows(log_root: Path) -> dict[str, CompletionRow]:
    ledgers = sorted(log_root.glob("schedule_audit_worker*_of_*.tsv"))
    rows: dict[str, CompletionRow] = {}
    for ledger in ledgers:
        _regular_file(ledger, "Raw Boxer completion ledger")
        try:
            with ledger.open("r", encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle, delimiter="\t")
                expected = {
                    "scene",
                    "manifest_keyframes",
                    "valid_keyframes",
                    "invalid_pose_keyframes",
                    "raw_candidate_frames",
                    "raw_candidates",
                }
                if reader.fieldnames is None or set(reader.fieldnames) != expected:
                    raise BirthMaterializationError(
                        f"unexpected completion-ledger columns: {ledger}"
                    )
                for source in reader:
                    scene = source.get("scene", "")
                    if SCENE_PATTERN.fullmatch(scene) is None:
                        # A concurrently appended, incomplete final line is not a
                        # completion event.  The wait loop will retry it.
                        continue
                    row = CompletionRow(
                        scene=scene,
                        manifest_keyframes=_nonnegative_int(
                            source["manifest_keyframes"], "manifest_keyframes"
                        ),
                        valid_keyframes=_nonnegative_int(
                            source["valid_keyframes"], "valid_keyframes"
                        ),
                        invalid_pose_keyframes=_nonnegative_int(
                            source["invalid_pose_keyframes"],
                            "invalid_pose_keyframes",
                        ),
                        raw_candidate_frames=_nonnegative_int(
                            source["raw_candidate_frames"], "raw_candidate_frames"
                        ),
                        raw_candidates=_nonnegative_int(
                            source["raw_candidates"], "raw_candidates"
                        ),
                        ledger_path=ledger,
                    )
                    if scene in rows:
                        raise BirthMaterializationError(
                            f"duplicate completion-ledger scene: {scene}"
                        )
                    rows[scene] = row
        except UnicodeDecodeError as error:
            raise BirthMaterializationError(f"invalid completion ledger: {ledger}") from error
    return rows


def wait_for_raw_completion(
    *,
    log_root: Path,
    scenes: Sequence[str],
    timeout_seconds: float,
    poll_seconds: float,
) -> dict[str, CompletionRow]:
    """Wait for all post-close ledger rows before any candidate CSV is read."""

    if timeout_seconds < 0.0 or not math.isfinite(timeout_seconds):
        raise BirthMaterializationError("wait timeout must be finite and nonnegative")
    if poll_seconds <= 0.0 or not math.isfinite(poll_seconds):
        raise BirthMaterializationError("poll interval must be finite and positive")
    requested = set(scenes)
    deadline = time.monotonic() + timeout_seconds
    while True:
        rows = _completion_rows(log_root)
        unknown = sorted(set(rows) - requested)
        if unknown:
            raise BirthMaterializationError(
                f"completion ledger contains off-protocol scenes: {unknown[:3]}"
            )
        completed = requested.intersection(rows)
        files_ready = {
            scene
            for scene in completed
            if (log_root / CSV_RELATIVE / scene / CSV_NAME).is_file()
        }
        if files_ready == requested:
            return {scene: rows[scene] for scene in scenes}
        if time.monotonic() >= deadline:
            missing = sorted(requested - files_ready)
            raise BirthMaterializationError(
                "Raw Boxer input is incomplete; refusing to read partial CSVs: "
                f"{len(files_ready)}/{len(scenes)} complete, missing={missing[:8]}"
            )
        time.sleep(min(poll_seconds, max(0.0, deadline - time.monotonic())))


def _valid_schedule(
    *,
    schedule_root: Path,
    scene_rgbd_root: Path,
    scene: str,
    completion: CompletionRow,
) -> tuple[tuple[int, ...], str, np.ndarray, dict[int, np.ndarray]]:
    manifest_path = schedule_root / scene / "manifest.json"
    manifest = _read_json(manifest_path, "T05 schedule manifest")
    if manifest.get("record_count") != completion.manifest_keyframes:
        raise BirthMaterializationError(f"schedule/ledger record count differs for {scene}")
    raw = manifest.get("recorded_frame_ids")
    if (
        not isinstance(raw, list)
        or any(type(value) is not int or value < 0 for value in raw)
        or raw != sorted(raw)
        or len(set(raw)) != len(raw)
        or len(raw) != completion.manifest_keyframes
    ):
        raise BirthMaterializationError(f"invalid T05 schedule for {scene}")
    valid = []
    camera_centers: dict[int, np.ndarray] = {}
    for frame_id in raw:
        pose_path = scene_rgbd_root / scene / "frames" / "pose" / f"{frame_id}.txt"
        if not pose_path.is_file() or pose_path.is_symlink():
            continue
        try:
            pose = np.loadtxt(pose_path, dtype=np.float64)
        except (OSError, ValueError):
            continue
        if pose.shape == (4, 4) and np.isfinite(pose).all():
            valid.append(frame_id)
            camera_centers[frame_id] = np.ascontiguousarray(pose[:3, 3])
    if len(valid) != completion.valid_keyframes:
        raise BirthMaterializationError(f"valid-pose schedule/ledger count differs for {scene}")
    if len(raw) - len(valid) != completion.invalid_pose_keyframes:
        raise BirthMaterializationError(f"invalid-pose schedule/ledger count differs for {scene}")
    namespace = manifest.get("namespace")
    if not isinstance(namespace, str) or not namespace:
        raise BirthMaterializationError(f"invalid T05 schedule namespace for {scene}")
    if not valid:
        raise BirthMaterializationError(f"scene has no valid exact-schedule pose: {scene}")
    # Released Boxer subtracts the first valid provider-pose translation before
    # exporting CSV centers.  Restore it before constructing world OBB corners.
    world_offset = np.array(camera_centers[valid[0]], dtype=np.float64, copy=True)
    return tuple(valid), _sha256(manifest_path), world_offset, camera_centers


def _finite_float(source: Mapping[str, str], key: str, row_number: int) -> float:
    try:
        value = float(source[key])
    except (KeyError, TypeError, ValueError) as error:
        raise BirthMaterializationError(
            f"Raw Boxer row {row_number} has invalid {key}"
        ) from error
    if not math.isfinite(value):
        raise BirthMaterializationError(f"Raw Boxer row {row_number} has nonfinite {key}")
    return value


def _quaternion_rotation(quaternion: np.ndarray) -> np.ndarray:
    q = np.asarray(quaternion, dtype=np.float64)
    norm_squared = float(q @ q)
    if q.shape != (4,) or not math.isfinite(norm_squared) or norm_squared <= 1e-12:
        raise BirthMaterializationError("invalid Raw Boxer quaternion")
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
    return SIGNS * (extent / 2.0) @ rotation.T + center


def _load_raw_observations(
    *,
    path: Path,
    schedule: Sequence[int],
    completion: CompletionRow,
    world_offset: np.ndarray,
) -> tuple[dict[int, tuple[S3RObservation, ...]], str]:
    _regular_file(path, "completed Raw Boxer CSV")
    digest_before = _sha256(path)
    offset = np.asarray(world_offset, dtype=np.float64)
    if offset.shape != (3,) or not np.isfinite(offset).all():
        raise BirthMaterializationError("Raw Boxer world offset must be finite [3]")
    by_frame: dict[int, list[S3RObservation]] = {}
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames != list(RAW_COLUMNS):
                raise BirthMaterializationError(f"unexpected Raw Boxer CSV columns: {path}")
            for source_row, source in enumerate(reader):
                frame_id = _nonnegative_int(source["time_ns"], "Raw Boxer time_ns")
                center_recentered = np.asarray(
                    [
                        _finite_float(source, "tx_world_object", source_row),
                        _finite_float(source, "ty_world_object", source_row),
                        _finite_float(source, "tz_world_object", source_row),
                    ],
                    dtype=np.float64,
                )
                center_absolute = center_recentered + offset
                quaternion = np.asarray(
                    [
                        _finite_float(source, "qw_world_object", source_row),
                        _finite_float(source, "qx_world_object", source_row),
                        _finite_float(source, "qy_world_object", source_row),
                        _finite_float(source, "qz_world_object", source_row),
                    ],
                    dtype=np.float64,
                )
                extent = np.asarray(
                    [
                        _finite_float(source, "scale_x", source_row),
                        _finite_float(source, "scale_y", source_row),
                        _finite_float(source, "scale_z", source_row),
                    ],
                    dtype=np.float64,
                )
                if np.any(extent <= 0.0):
                    raise BirthMaterializationError(
                        f"Raw Boxer row {source_row} has nonpositive extent"
                    )
                score = _finite_float(source, "prob", source_row)
                if score < 0.0 or score > 1.0:
                    raise BirthMaterializationError(
                        f"Raw Boxer row {source_row} score is outside [0,1]"
                    )
                source_instance_id = _nonnegative_int(
                    source["instance"], "Raw Boxer instance"
                )
                observation = S3RObservation(
                    frame_id=frame_id,
                    source_row=source_row,
                    sealed_npz_row=source_row,
                    source_instance_id=source_instance_id,
                    score=score,
                    corners=_obb_corners(center_absolute, extent, quaternion),
                )
                by_frame.setdefault(frame_id, []).append(observation)
    except UnicodeDecodeError as error:
        raise BirthMaterializationError(f"invalid Raw Boxer CSV: {path}") from error
    observed = set(by_frame)
    scheduled = set(schedule)
    if not observed.issubset(scheduled):
        raise BirthMaterializationError(
            f"Raw Boxer CSV contains off-schedule frames: {sorted(observed-scheduled)[:8]}"
        )
    row_count = sum(len(rows) for rows in by_frame.values())
    if row_count != completion.raw_candidates:
        raise BirthMaterializationError("Raw Boxer CSV/ledger row count differs")
    if len(observed) != completion.raw_candidate_frames:
        raise BirthMaterializationError("Raw Boxer CSV/ledger frame count differs")
    if _sha256(path) != digest_before:
        raise BirthMaterializationError("Raw Boxer CSV changed while it was read")
    return {key: tuple(value) for key, value in by_frame.items()}, digest_before


def _load_raw_semantic_ids(path: Path) -> dict[int, int]:
    """Load only the immutable integer semantic identity for v2 admission.

    The row number is the same provenance identity stored in every S3R
    receipt.  Free-form names are intentionally not decoded or mapped to a
    ScanNet class list.
    """

    _regular_file(path, "completed Raw Boxer CSV")
    result: dict[int, int] = {}
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames != list(RAW_COLUMNS):
                raise BirthMaterializationError(
                    f"unexpected Raw Boxer CSV columns: {path}"
                )
            for source_row, source in enumerate(reader):
                result[source_row] = _nonnegative_int(
                    source.get("sem_id"), "Raw Boxer sem_id"
                )
    except UnicodeDecodeError as error:
        raise BirthMaterializationError(f"invalid Raw Boxer CSV: {path}") from error
    return result


def _load_native_prediction(path: Path) -> NativePrediction:
    _regular_file(path, "native T05 prediction")
    try:
        with path.open("rb") as handle:
            payload = pickle.load(handle)
    except Exception as error:
        raise BirthMaterializationError(f"could not deserialize native prediction: {path}") from error
    if not isinstance(payload, (list, tuple)) or len(payload) != 1:
        raise BirthMaterializationError(f"invalid native prediction outer schema: {path}")
    rows = payload[0]
    if not isinstance(rows, (list, tuple)):
        raise BirthMaterializationError(f"invalid native prediction row container: {path}")
    corners = []
    for row_index, row in enumerate(rows):
        if not isinstance(row, (list, tuple)) or len(row) != 3:
            raise BirthMaterializationError(f"invalid native row {row_index}: {path}")
        label, raw_corners, raw_score = row
        if isinstance(label, (bool, np.bool_)) or not isinstance(label, (int, np.integer)):
            raise BirthMaterializationError(f"invalid native class at row {row_index}: {path}")
        box = np.asarray(raw_corners)
        if box.shape != (8, 3) or not np.issubdtype(box.dtype, np.number):
            raise BirthMaterializationError(f"invalid native corners at row {row_index}: {path}")
        if not np.isfinite(box).all() or np.any(np.ptp(box.astype(np.float64), axis=0) <= 0.0):
            raise BirthMaterializationError(f"invalid native geometry at row {row_index}: {path}")
        try:
            score = float(raw_score)
        except (TypeError, ValueError) as error:
            raise BirthMaterializationError(
                f"invalid native score at row {row_index}: {path}"
            ) from error
        if not math.isfinite(score) or score <= 0.0 or score > 1.0:
            raise BirthMaterializationError(f"invalid native score at row {row_index}: {path}")
        corners.append(np.asarray(box, dtype=np.float64))
    stacked = np.stack(corners) if corners else np.empty((0, 8, 3), dtype=np.float64)
    return NativePrediction(payload=payload, rows=rows, corners=stacked)


def _bounds_and_volume(corners: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    boxes = np.asarray(corners, dtype=np.float64)
    if boxes.size == 0:
        return (
            np.empty((0, 3), dtype=np.float64),
            np.empty((0, 3), dtype=np.float64),
            np.empty((0,), dtype=np.float64),
        )
    lower = boxes.min(axis=1)
    upper = boxes.max(axis=1)
    return lower, upper, np.prod(upper - lower, axis=1)


def _aabb_iou_matrix(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left_lower, left_upper, left_volume = _bounds_and_volume(left)
    right_lower, right_upper, right_volume = _bounds_and_volume(right)
    if not len(left) or not len(right):
        return np.zeros((len(left), len(right)), dtype=np.float64)
    intersection_extent = np.maximum(
        np.minimum(left_upper[:, None], right_upper[None])
        - np.maximum(left_lower[:, None], right_lower[None]),
        0.0,
    )
    intersection = np.prod(intersection_extent, axis=2)
    union = left_volume[:, None] + right_volume[None] - intersection
    return np.divide(
        intersection, union, out=np.zeros_like(intersection), where=union > 0.0
    )


def _aabb_overlap_matrices(
    left: np.ndarray, right: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return IoU, intersection/left-volume, and intersection/right-volume."""

    left_lower, left_upper, left_volume = _bounds_and_volume(left)
    right_lower, right_upper, right_volume = _bounds_and_volume(right)
    if not len(left) or not len(right):
        shape = (len(left), len(right))
        empty = np.zeros(shape, dtype=np.float64)
        return empty, empty.copy(), empty.copy()
    intersection_extent = np.maximum(
        np.minimum(left_upper[:, None], right_upper[None])
        - np.maximum(left_lower[:, None], right_lower[None]),
        0.0,
    )
    intersection = np.prod(intersection_extent, axis=2)
    union = left_volume[:, None] + right_volume[None] - intersection
    iou = np.divide(
        intersection, union, out=np.zeros_like(intersection), where=union > 0.0
    )
    left_containment = np.divide(
        intersection,
        left_volume[:, None],
        out=np.zeros_like(intersection),
        where=left_volume[:, None] > 0.0,
    )
    right_containment = np.divide(
        intersection,
        right_volume[None],
        out=np.zeros_like(intersection),
        where=right_volume[None] > 0.0,
    )
    return iou, left_containment, right_containment


def _rank_key(receipt: S3RReceipt) -> tuple[float, float, float, int]:
    return (
        -receipt.raw_mean_score,
        -receipt.median_pairwise_aabb_iou,
        receipt.center_rms_m,
        receipt.track_id,
    )


def _view_diversity(
    receipt: S3RReceipt, camera_centers: Mapping[int, np.ndarray]
) -> tuple[float, float]:
    try:
        centers = np.stack(
            [camera_centers[frame_id] for frame_id in receipt.evidence_frame_ids]
        ).astype(np.float64, copy=False)
    except KeyError as error:
        raise BirthMaterializationError(
            f"receipt references a frame outside the valid-pose schedule: {error.args[0]}"
        ) from error
    if centers.shape != (3, 3) or not np.isfinite(centers).all():
        raise BirthMaterializationError("Past3 evidence camera centers must be finite [3,3]")
    baselines = np.linalg.norm(centers[:, None] - centers[None], axis=2)
    receipt_center = np.asarray(receipt.corners, dtype=np.float64).mean(axis=0)
    rays = receipt_center[None] - centers
    norms = np.linalg.norm(rays, axis=1)
    if np.any(norms <= 1e-9):
        return float(baselines.max()), 0.0
    unit = rays / norms[:, None]
    ray_angles = np.degrees(np.arccos(np.clip(unit @ unit.T, -1.0, 1.0)))
    return float(baselines.max()), float(ray_angles.max())


def _select_births(
    receipts: Sequence[S3RReceipt],
    native_corners: np.ndarray,
    camera_centers: Mapping[int, np.ndarray],
) -> tuple[tuple[S3RReceipt, ...], list[dict[str, Any]]]:
    """Apply the frozen native-novelty, self-NMS, and per-scene cap policy."""

    ranked = tuple(sorted(receipts, key=_rank_key))
    receipt_corners = (
        np.stack([row.corners for row in ranked])
        if ranked
        else np.empty((0, 8, 3), dtype=np.float64)
    )
    native_iou = _aabb_iou_matrix(receipt_corners, native_corners)
    self_iou = _aabb_iou_matrix(receipt_corners, receipt_corners)
    decisions: list[dict[str, Any]] = []
    kept_indices: list[int] = []
    selected: list[S3RReceipt] = []
    for index, receipt in enumerate(ranked):
        camera_baseline_m, view_ray_span_deg = _view_diversity(
            receipt, camera_centers
        )
        max_native_iou = float(native_iou[index].max()) if native_iou.shape[1] else 0.0
        decision = "accepted"
        if (
            receipt.median_pairwise_aabb_iou
            < MIN_STABLE_MEDIAN_PAIRWISE_AABB_IOU
            or receipt.center_rms_m > MAX_STABLE_CENTER_RMS_M
        ):
            decision = "unstable"
        elif receipt.min_medoid_aabb_extent_m < MIN_MEDOID_AABB_EXTENT_M:
            decision = "too_small"
        elif (
            camera_baseline_m < MIN_CAMERA_BASELINE_M
            or view_ray_span_deg < MIN_VIEW_RAY_SPAN_DEG
        ):
            decision = "view_diversity"
        elif max_native_iou >= NATIVE_NOVELTY_AABB_IOU:
            decision = "native_overlap"
        else:
            for kept_index in kept_indices:
                if self_iou[index, kept_index] >= SELF_NMS_AABB_IOU:
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
                "raw_mean_score": receipt.raw_mean_score,
                "median_pairwise_aabb_iou": receipt.median_pairwise_aabb_iou,
                "center_rms_m": receipt.center_rms_m,
                "min_medoid_aabb_extent_m": receipt.min_medoid_aabb_extent_m,
                "max_camera_baseline_m": camera_baseline_m,
                "max_view_ray_span_deg": view_ray_span_deg,
                "max_native_aabb_iou": max_native_iou,
            }
        )
    return tuple(selected), decisions


def _v2_rank_key(receipt: S3RReceipt) -> tuple[float, float, float, float, int]:
    upper = receipt.pairwise_aabb_iou[np.triu_indices(3, 1)]
    return (
        -float(np.min(upper)),
        -float(min(receipt.evidence_scores)),
        -receipt.raw_mean_score,
        receipt.center_rms_m,
        receipt.track_id,
    )


def _select_births_v2_m50(
    receipts: Sequence[S3RReceipt],
    native_corners: np.ndarray,
    camera_centers: Mapping[int, np.ndarray],
    semantic_ids: Mapping[int, int],
    clip_gate_records: Mapping[
        tuple[int, tuple[int, int, int]], Mapping[str, Any]
    ]
    | None = None,
) -> tuple[tuple[S3RReceipt, ...], list[dict[str, Any]]]:
    """Apply birth-v2-M50, optionally inserting CLIP before NMS and cap.

    The optional semantic gate is deliberately placed after every v2 scalar
    and native-novelty check, but before self-NMS and the per-scene cap.  Thus
    a CLIP-rejected candidate cannot suppress a later passing candidate.
    """

    ranked = tuple(sorted(receipts, key=_v2_rank_key))
    receipt_corners = (
        np.stack([row.corners for row in ranked])
        if ranked
        else np.empty((0, 8, 3), dtype=np.float64)
    )
    native_iou, candidate_in_native, native_in_candidate = _aabb_overlap_matrices(
        receipt_corners, native_corners
    )
    self_iou, self_left_containment, self_right_containment = (
        _aabb_overlap_matrices(receipt_corners, receipt_corners)
    )
    decisions: list[dict[str, Any]] = []
    kept_indices: list[int] = []
    selected: list[S3RReceipt] = []
    triangle = np.triu_indices(3, 1)
    for index, receipt in enumerate(ranked):
        clip_gate_record: Mapping[str, Any] | None = None
        try:
            evidence_semantic_ids = tuple(
                semantic_ids[source_row]
                for source_row in receipt.evidence_source_rows
            )
        except KeyError as error:
            raise BirthMaterializationError(
                f"v2 receipt references missing Raw Boxer semantic row: {error.args[0]}"
            ) from error
        pairwise_iou = receipt.pairwise_aabb_iou[triangle]
        pairwise_center = receipt.pairwise_center_distance_m[triangle]
        min_pairwise_iou = float(np.min(pairwise_iou))
        max_pairwise_center_m = float(np.max(pairwise_center))
        min_evidence_score = float(min(receipt.evidence_scores))
        first_last_frame_span = (
            receipt.evidence_frame_ids[-1] - receipt.evidence_frame_ids[0]
        )
        camera_baseline_m, view_ray_span_deg = _view_diversity(
            receipt, camera_centers
        )
        max_native_iou = (
            float(native_iou[index].max()) if native_iou.shape[1] else 0.0
        )
        max_candidate_in_native = (
            float(candidate_in_native[index].max())
            if candidate_in_native.shape[1]
            else 0.0
        )
        max_native_in_candidate = (
            float(native_in_candidate[index].max())
            if native_in_candidate.shape[1]
            else 0.0
        )

        decision = "accepted"
        if len(set(evidence_semantic_ids)) != 1:
            decision = "semantic_inconsistent"
        elif (
            min_evidence_score < V2_MIN_EVIDENCE_SCORE
            or receipt.raw_mean_score < V2_MIN_MEAN_SCORE
        ):
            decision = "score"
        elif min_pairwise_iou < V2_MIN_PAIRWISE_AABB_IOU:
            decision = "pairwise_iou"
        elif max_pairwise_center_m > V2_MAX_PAIRWISE_CENTER_DISTANCE_M:
            decision = "center_distance"
        elif first_last_frame_span < V2_MIN_FIRST_LAST_FRAME_SPAN:
            decision = "frame_span"
        elif (
            camera_baseline_m < V2_MIN_CAMERA_BASELINE_M
            or view_ray_span_deg < V2_MIN_VIEW_RAY_SPAN_DEG
        ):
            decision = "view_diversity"
        elif receipt.min_medoid_aabb_extent_m < V2_MIN_MEDOID_AABB_EXTENT_M:
            decision = "too_small"
        elif max_native_iou >= V2_NATIVE_NOVELTY_AABB_IOU:
            decision = "native_overlap"
        elif (
            max_candidate_in_native >= V2_NATIVE_MAX_BIDIRECTIONAL_CONTAINMENT
            or max_native_in_candidate >= V2_NATIVE_MAX_BIDIRECTIONAL_CONTAINMENT
        ):
            decision = "native_containment"
        else:
            if clip_gate_records is not None:
                receipt_key = (
                    receipt.track_id,
                    tuple(receipt.evidence_source_rows),
                )
                clip_gate_record = clip_gate_records.get(receipt_key)
                if clip_gate_record is None:
                    raise BirthMaterializationError(
                        "CLIP gate sidecar is missing a v2-admissible receipt: "
                        f"track={receipt.track_id}, rows={receipt.evidence_source_rows}"
                    )
                if clip_gate_record.get("gate_pass") is not True:
                    decision = "clip_gate"
            if decision == "accepted":
                for kept_index in kept_indices:
                    if (
                        self_iou[index, kept_index] >= V2_SELF_NMS_AABB_IOU
                        or self_left_containment[index, kept_index]
                        >= V2_SELF_NMS_BIDIRECTIONAL_CONTAINMENT
                        or self_right_containment[index, kept_index]
                        >= V2_SELF_NMS_BIDIRECTIONAL_CONTAINMENT
                    ):
                        decision = "self_nms"
                        break
        if decision == "accepted" and len(selected) >= V2_MAX_BIRTHS_PER_SCENE:
            decision = "scene_cap"
        if decision == "accepted":
            kept_indices.append(index)
            selected.append(receipt)
        decision_record = {
                "track_id": receipt.track_id,
                "decision": decision,
                "confirmation_frame_id": receipt.confirmation_frame_id,
                "evidence_frame_ids": list(receipt.evidence_frame_ids),
                "evidence_source_rows": list(receipt.evidence_source_rows),
                "evidence_semantic_ids": list(evidence_semantic_ids),
                "semantic_consistent": len(set(evidence_semantic_ids)) == 1,
                "min_evidence_score": min_evidence_score,
                "raw_mean_score": receipt.raw_mean_score,
                "min_pairwise_aabb_iou": min_pairwise_iou,
                "median_pairwise_aabb_iou": receipt.median_pairwise_aabb_iou,
                "max_pairwise_center_distance_m": max_pairwise_center_m,
                "center_rms_m": receipt.center_rms_m,
                "first_last_frame_span": first_last_frame_span,
                "min_medoid_aabb_extent_m": receipt.min_medoid_aabb_extent_m,
                "max_camera_baseline_m": camera_baseline_m,
                "max_view_ray_span_deg": view_ray_span_deg,
                "max_native_aabb_iou": max_native_iou,
                "max_candidate_in_native_containment": max_candidate_in_native,
                "max_native_in_candidate_containment": max_native_in_candidate,
            }
        if clip_gate_records is not None:
            decision_record.update(
                {
                    "clip_gate_evaluated": clip_gate_record is not None,
                    "clip_gate_pass": (
                        None
                        if clip_gate_record is None
                        else clip_gate_record.get("gate_pass")
                    ),
                    "clip_gate_sidecar_record": (
                        None
                        if clip_gate_record is None
                        else dict(clip_gate_record)
                    ),
                }
            )
        decisions.append(decision_record)
    return tuple(selected), decisions


def _same_scalar_bits(left: object, right: object) -> bool:
    try:
        return struct.pack("!d", float(left)) == struct.pack("!d", float(right))
    except (TypeError, ValueError, OverflowError):
        return False


def _assert_native_prefix(
    native_rows: Sequence[Any], output_rows: Sequence[Any], label: str
) -> None:
    if len(output_rows) < len(native_rows):
        raise BirthMaterializationError(f"native prefix was truncated for {label}")
    for index, (before, after) in enumerate(zip(native_rows, output_rows)):
        if not isinstance(after, (list, tuple)) or len(after) != 3:
            raise BirthMaterializationError(f"output prefix schema changed at {label}:{index}")
        if type(after) is not type(before):
            raise BirthMaterializationError(f"output prefix row type changed at {label}:{index}")
        if type(after[0]) is not type(before[0]) or after[0] != before[0]:
            raise BirthMaterializationError(f"output prefix class changed at {label}:{index}")
        left = np.asarray(before[1])
        right = np.asarray(after[1])
        if left.dtype != right.dtype or left.shape != right.shape or left.tobytes() != right.tobytes():
            raise BirthMaterializationError(f"output prefix corners changed at {label}:{index}")
        if type(after[2]) is not type(before[2]) or not _same_scalar_bits(after[2], before[2]):
            raise BirthMaterializationError(f"output prefix score changed at {label}:{index}")


def _augmented_payload(
    native: NativePrediction, selected: Sequence[S3RReceipt]
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
    if isinstance(native.payload, tuple):
        output: list[Any] | tuple[Any, ...] = (rows,)
    else:
        output = [rows]
    _assert_native_prefix(native.rows, output[0], "in-memory output")
    return output


def _write_pickle(path: Path, payload: object) -> None:
    with path.open("xb") as handle:
        pickle.dump(payload, handle, protocol=4)
        handle.flush()
        os.fsync(handle.fileno())


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    encoded = (
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False)
        + "\n"
    ).encode("ascii")
    with path.open("xb") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())


def materialize_scannet_raw_boxer_past3_birth_full100(
    *,
    scene_list: Path,
    raw_log_root: Path,
    baseline_root: Path,
    schedule_root: Path,
    scene_rgbd_root: Path,
    output_root: Path,
    expected_scene_count: int = 100,
    wait_timeout_seconds: float = 0.0,
    poll_seconds: float = 30.0,
    selection_policy: str = "v1",
    clip_gate_sidecar: Path | None = None,
) -> dict[str, Any]:
    """Create a complete prediction root and return its audit manifest."""

    for path, label in (
        (raw_log_root, "Raw Boxer log root"),
        (baseline_root, "native T05 root"),
        (schedule_root, "T05 schedule root"),
        (scene_rgbd_root, "ScanNet RGB-D root"),
    ):
        if path.is_symlink() or not path.is_dir():
            raise BirthMaterializationError(f"{label} must be a non-symlink directory: {path}")
    if expected_scene_count <= 0:
        raise BirthMaterializationError("expected scene count must be positive")
    if selection_policy not in ("v1", "v2_m50", "v3_clip_vocab"):
        raise BirthMaterializationError(
            "selection_policy must be 'v1', 'v2_m50', or 'v3_clip_vocab'"
        )
    if selection_policy == "v3_clip_vocab" and clip_gate_sidecar is None:
        raise BirthMaterializationError(
            "v3_clip_vocab requires --clip-gate-sidecar"
        )
    if selection_policy != "v3_clip_vocab" and clip_gate_sidecar is not None:
        raise BirthMaterializationError(
            "--clip-gate-sidecar is valid only for v3_clip_vocab"
        )
    output_root = output_root.resolve()
    if output_root.exists() or output_root.is_symlink():
        raise BirthMaterializationError(f"refusing to overwrite output root: {output_root}")
    scenes = _scene_list(scene_list, expected_scene_count)
    clip_sidecar = (
        _load_clip_gate_sidecar(clip_gate_sidecar)
        if clip_gate_sidecar is not None
        else None
    )
    if clip_sidecar is not None:
        unknown_clip_scenes = sorted(set(clip_sidecar.records) - set(scenes))
        if unknown_clip_scenes:
            raise BirthMaterializationError(
                "CLIP gate sidecar contains off-protocol scenes: "
                f"{unknown_clip_scenes[:3]}"
            )
    completion_rows = wait_for_raw_completion(
        log_root=raw_log_root,
        scenes=scenes,
        timeout_seconds=wait_timeout_seconds,
        poll_seconds=poll_seconds,
    )

    output_root.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{output_root.name}.stage-", dir=output_root.parent))
    scene_reports: dict[str, Any] = {}
    baseline_hashes: dict[str, str] = {}
    raw_hashes: dict[str, str] = {}
    output_hashes: dict[str, str] = {}
    schedule_hashes: dict[str, str] = {}
    total_receipts = 0
    total_births = 0
    total_native = 0
    total_clip_gate_evaluated = 0
    try:
        for scene in scenes:
            completion = completion_rows[scene]
            schedule, schedule_digest, world_offset, camera_centers = _valid_schedule(
                schedule_root=schedule_root,
                scene_rgbd_root=scene_rgbd_root,
                scene=scene,
                completion=completion,
            )
            schedule_hashes[scene] = schedule_digest
            csv_path = raw_log_root / CSV_RELATIVE / scene / CSV_NAME
            by_frame, raw_digest = _load_raw_observations(
                path=csv_path,
                schedule=schedule,
                completion=completion,
                world_offset=world_offset,
            )
            semantic_ids = (
                _load_raw_semantic_ids(csv_path)
                if selection_policy in ("v2_m50", "v3_clip_vocab")
                else {}
            )
            raw_hashes[scene] = raw_digest
            native_path = baseline_root / f"{scene}{PREDICTION_SUFFIX}"
            baseline_hashes[scene] = _sha256(_regular_file(native_path, "native T05 prediction"))
            native = _load_native_prediction(native_path)

            tracker = S3RReceiptTracker()
            selected_observation_count = 0
            for frame_id in schedule:
                selected_frame = tuple(
                    sorted(
                        by_frame.get(frame_id, ()),
                        key=lambda row: (
                            -row.score,
                            row.source_row,
                            row.sealed_npz_row,
                        ),
                    )[:TOP_K_PER_FRAME]
                )
                selected_observation_count += len(selected_frame)
                query = tracker.query(frame_id, selected_frame)
                commit = tracker.commit(query)
                if (
                    query.selected_source_rows
                    != tuple(row.source_row for row in selected_frame)
                    or query.observation_capacity_dropped_source_rows
                    or not query.audit_complete
                    or not commit.audit_complete
                ):
                    raise BirthMaterializationError(
                        f"S3R tracker changed or truncated frozen K8 for {scene}/{frame_id}"
                    )
            summary = tracker.summary()
            if not summary.get("audit_complete"):
                raise BirthMaterializationError(f"bounded Past3 audit is incomplete for {scene}")
            receipts = tracker.receipts()
            if selection_policy == "v3_clip_vocab":
                assert clip_sidecar is not None
                selected, decisions = _select_births_v2_m50(
                    receipts,
                    native.corners,
                    camera_centers,
                    semantic_ids,
                    clip_gate_records=clip_sidecar.records.get(scene, {}),
                )
            elif selection_policy == "v2_m50":
                selected, decisions = _select_births_v2_m50(
                    receipts, native.corners, camera_centers, semantic_ids
                )
            else:
                selected, decisions = _select_births(
                    receipts, native.corners, camera_centers
                )
            payload = _augmented_payload(native, selected)
            output_path = stage / f"{scene}{PREDICTION_SUFFIX}"
            _write_pickle(output_path, payload)
            reloaded = _load_native_prediction(output_path)
            _assert_native_prefix(native.rows, reloaded.rows, scene)
            if len(reloaded.rows) != len(native.rows) + len(selected):
                raise BirthMaterializationError(f"suffix count changed during write for {scene}")
            if _sha256(native_path) != baseline_hashes[scene]:
                raise BirthMaterializationError(f"native T05 prediction changed for {scene}")
            if _sha256(csv_path) != raw_hashes[scene]:
                raise BirthMaterializationError(f"Raw Boxer CSV changed after processing {scene}")
            output_hashes[scene] = _sha256(output_path)

            if selection_policy in ("v2_m50", "v3_clip_vocab"):
                decision_reasons = (
                    "accepted",
                    "semantic_inconsistent",
                    "score",
                    "pairwise_iou",
                    "center_distance",
                    "frame_span",
                    "view_diversity",
                    "too_small",
                    "native_overlap",
                    "native_containment",
                    *(("clip_gate",) if selection_policy == "v3_clip_vocab" else ()),
                    "self_nms",
                    "scene_cap",
                )
            else:
                decision_reasons = (
                    "accepted",
                    "unstable",
                    "too_small",
                    "view_diversity",
                    "native_overlap",
                    "self_nms",
                    "scene_cap",
                )
            counts = {
                reason: sum(row["decision"] == reason for row in decisions)
                for reason in decision_reasons
            }
            scene_reports[scene] = {
                "native_count": len(native.rows),
                "raw_candidate_count": completion.raw_candidates,
                "raw_candidate_frame_count": completion.raw_candidate_frames,
                "k8_selected_observation_count": selected_observation_count,
                "valid_keyframe_count": len(schedule),
                "world_offset_frame_id": schedule[0],
                "world_offset_xyz": world_offset.tolist(),
                "past3_receipt_count": len(receipts),
                "birth_count": len(selected),
                "clip_gate_evaluated_count": sum(
                    row.get("clip_gate_evaluated") is True for row in decisions
                ),
                "decision_counts": counts,
                "native_prefix_row_identity_verified": True,
                "suffix": [
                    {
                        "suffix_index": index,
                        "track_id": receipt.track_id,
                        "class_id": APPENDED_CLASS_ID,
                        "score": APPENDED_SCORE,
                        "corners_world": np.asarray(receipt.corners).tolist(),
                        "confirmation_frame_id": receipt.confirmation_frame_id,
                        "evidence_frame_ids": list(receipt.evidence_frame_ids),
                        "evidence_source_rows": list(receipt.evidence_source_rows),
                        "raw_mean_score": receipt.raw_mean_score,
                    }
                    for index, receipt in enumerate(selected)
                ],
                "receipt_decisions": decisions,
                "tracker_summary": summary,
            }
            total_receipts += len(receipts)
            total_births += len(selected)
            total_native += len(native.rows)
            total_clip_gate_evaluated += sum(
                row.get("clip_gate_evaluated") is True for row in decisions
            )

        manifest: dict[str, Any] = {
            "schema": (
                V3_SCHEMA
                if selection_policy == "v3_clip_vocab"
                else V2_SCHEMA if selection_policy == "v2_m50" else SCHEMA
            ),
            "mode": (
                "active_birth_v3_clip_vocab"
                if selection_policy == "v3_clip_vocab"
                else "active_birth_v2_m50"
                if selection_policy == "v2_m50"
                else "active_birth"
            ),
            "selection_policy": selection_policy,
            "training_free": True,
            "target_dataset_training": False,
            "external_pretraining_frozen": True,
            "online_learning": False,
            "past_only_confirmation": True,
            "minimum_distinct_views": 3,
            "gt_access": False,
            "evaluator_access": False,
            "annotation_path_argument": False,
            "detector_semantics_used": selection_policy in (
                "v2_m50",
                "v3_clip_vocab",
            ),
            "detector_semantics_usage": (
                "exact_three_view_sem_id_identity_only"
                if selection_policy in ("v2_m50", "v3_clip_vocab")
                else "none"
            ),
            "clip_access": selection_policy == "v3_clip_vocab",
            "depth_access": False,
            "rgb_access": selection_policy == "v3_clip_vocab",
            "pose_access": "world-offset restoration and fixed view-diversity gate",
            "native_rows_are_unchanged_prefix": True,
            "native_clip_unchanged": True,
            "coordinate_frame": "scannet_world",
            "score_mode": "constant_1.0",
            "class_mode": "inert_0_scannet_class_agnostic_evaluator",
            "scene_count": len(scenes),
            "native_count": total_native,
            "past3_receipt_count": total_receipts,
            "birth_count": total_births,
            "clip_gate_evaluated_count": total_clip_gate_evaluated,
            "frozen_policy": {
                "tracker": "boxfusion.s3r_receipt_tracker.v1",
                "top_k_per_frame": TOP_K_PER_FRAME,
                "top_k_selection": (
                    "source_score_desc_then_source_row_asc_then_raw_csv_row_asc"
                ),
                "receipt_geometry": "aabb_iou_medoid_of_exact_first_three_rows",
                "match_rule": "aabb_iou_gte_0.10_AND_center_distance_lte_0.50m",
                "ttl_valid_keyframes": 10,
                "recenter_to_world_rule": (
                    "center_world=center_boxer_recentered+translation_of_first_"
                    "valid_exact_schedule_pose"
                ),
                "stable_median_pairwise_aabb_iou_gte": (
                    MIN_STABLE_MEDIAN_PAIRWISE_AABB_IOU
                ),
                "stable_center_rms_m_lte": MAX_STABLE_CENTER_RMS_M,
                "min_medoid_aabb_extent_m_gte": MIN_MEDOID_AABB_EXTENT_M,
                "min_camera_baseline_m_gte": MIN_CAMERA_BASELINE_M,
                "min_view_ray_span_deg_gte": MIN_VIEW_RAY_SPAN_DEG,
                "ranking": [
                    "raw_mean_score_desc",
                    "median_pairwise_aabb_iou_desc",
                    "center_rms_m_asc",
                    "track_id_asc",
                ],
                "native_novelty_aabb_iou_gte_reject": NATIVE_NOVELTY_AABB_IOU,
                "self_nms_aabb_iou_gte_reject": SELF_NMS_AABB_IOU,
                "max_births_per_scene": MAX_BIRTHS_PER_SCENE,
                "appended_class_id": APPENDED_CLASS_ID,
                "appended_score": APPENDED_SCORE,
            },
            "inputs": {
                "scene_list": os.fspath(scene_list.resolve()),
                "scene_list_sha256": _sha256(scene_list),
                "raw_log_root": os.fspath(raw_log_root.resolve()),
                "baseline_root": os.fspath(baseline_root.resolve()),
                "schedule_root": os.fspath(schedule_root.resolve()),
                "scene_rgbd_root": os.fspath(scene_rgbd_root.resolve()),
                "materializer_source": os.fspath(Path(__file__).resolve()),
                "materializer_source_sha256": _sha256(Path(__file__).resolve()),
                "tracker_source": os.fspath(
                    (REPOSITORY_ROOT / "boxfusion" / "s3r_receipt_tracker.py").resolve()
                ),
                "tracker_source_sha256": _sha256(
                    REPOSITORY_ROOT / "boxfusion" / "s3r_receipt_tracker.py"
                ),
                "completion_ledgers": {
                    os.fspath(path.resolve()): _sha256(path)
                    for path in sorted({row.ledger_path for row in completion_rows.values()})
                },
                "clip_gate_sidecar": (
                    None if clip_sidecar is None else os.fspath(clip_sidecar.path)
                ),
                "clip_gate_sidecar_sha256": (
                    None if clip_sidecar is None else clip_sidecar.sha256
                ),
                "clip_gate_sidecar_schema": (
                    None if clip_sidecar is None else clip_sidecar.schema
                ),
                "clip_gate_sidecar_record_count": (
                    0 if clip_sidecar is None else clip_sidecar.record_count
                ),
            },
            "native_prediction_sha256": baseline_hashes,
            "raw_boxer_csv_sha256": raw_hashes,
            "schedule_manifest_sha256": schedule_hashes,
            "output_prediction_sha256": output_hashes,
            "scenes": scene_reports,
        }
        if selection_policy in ("v2_m50", "v3_clip_vocab"):
            for legacy_key in (
                "stable_median_pairwise_aabb_iou_gte",
                "stable_center_rms_m_lte",
            ):
                manifest["frozen_policy"].pop(legacy_key, None)
            manifest["frozen_policy"].update(
                {
                    "semantic_confirmation": "exact_sem_id_equal_across_first_three",
                    "min_evidence_score_gte": V2_MIN_EVIDENCE_SCORE,
                    "min_mean_score_gte": V2_MIN_MEAN_SCORE,
                    "min_all_pair_aabb_iou_gte": V2_MIN_PAIRWISE_AABB_IOU,
                    "max_all_pair_center_distance_m_lte": (
                        V2_MAX_PAIRWISE_CENTER_DISTANCE_M
                    ),
                    "min_first_last_frame_span_gte": (
                        V2_MIN_FIRST_LAST_FRAME_SPAN
                    ),
                    "min_camera_baseline_m_gte": V2_MIN_CAMERA_BASELINE_M,
                    "min_view_ray_span_deg_gte": V2_MIN_VIEW_RAY_SPAN_DEG,
                    "min_medoid_aabb_extent_m_gte": (
                        V2_MIN_MEDOID_AABB_EXTENT_M
                    ),
                    "native_novelty_aabb_iou_gte_reject": (
                        V2_NATIVE_NOVELTY_AABB_IOU
                    ),
                    "native_bidirectional_containment_gte_reject": (
                        V2_NATIVE_MAX_BIDIRECTIONAL_CONTAINMENT
                    ),
                    "self_nms_aabb_iou_gte_reject": V2_SELF_NMS_AABB_IOU,
                    "self_nms_bidirectional_containment_gte_reject": (
                        V2_SELF_NMS_BIDIRECTIONAL_CONTAINMENT
                    ),
                    "max_births_per_scene": V2_MAX_BIRTHS_PER_SCENE,
                    "ranking": [
                        "min_pair_aabb_iou_desc",
                        "min_evidence_score_desc",
                        "raw_mean_score_desc",
                        "center_rms_m_asc",
                        "track_id_asc",
                    ],
                }
            )
        if selection_policy == "v3_clip_vocab":
            assert clip_sidecar is not None
            if _sha256(clip_sidecar.path) != clip_sidecar.sha256:
                raise BirthMaterializationError(
                    "CLIP gate sidecar changed while materialization was running"
                )
            manifest["frozen_policy"].update(
                {
                    "clip_gate_join_key": [
                        "scene",
                        "track_id",
                        "evidence_source_rows",
                    ],
                    "clip_gate_position": (
                        "after_v2_scalar_and_native_novelty_gates_before_"
                        "self_nms_and_scene_cap"
                    ),
                    "clip_gate_required_value": "gate_pass=true",
                    "clip_gate_missing_admissible_receipt": "fatal",
                }
            )
        _write_json(stage / "RAW_BOXER_PAST3_BIRTH_FULL100.json", manifest)
        directory_fd = os.open(stage, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        if output_root.exists() or output_root.is_symlink():
            raise BirthMaterializationError(f"refusing to overwrite output root: {output_root}")
        os.rename(stage, output_root)
        return manifest
    except Exception:
        if stage.exists():
            shutil.rmtree(stage)
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Materialize a fixed no-GT Raw-Boxer Past3 birth full100 root"
    )
    parser.add_argument(
        "--scene-list",
        type=Path,
        default=REPOSITORY_ROOT / "evaluation/data_util/meta_data/scannetv2_val.txt",
    )
    parser.add_argument(
        "--raw-log-root",
        type=Path,
        default=REPOSITORY_ROOT / "logs/scannet_raw_boxer_full100_score05_v1",
    )
    parser.add_argument(
        "--baseline-root",
        type=Path,
        default=REPOSITORY_ROOT / "results/scannet_topk_fusion_score05",
    )
    parser.add_argument(
        "--schedule-root",
        type=Path,
        default=Path(
            "/data/ZhaoX/OVM3D-Dett/boxfusion_boxer_dev/cache/cutr_proposals/"
            "scannet-score05-gap25-postfilter-v2"
        ),
    )
    parser.add_argument(
        "--scene-rgbd-root",
        type=Path,
        default=REPOSITORY_ROOT / "upstream_clean/scannet_readme_frames",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPOSITORY_ROOT / "results/scannet_t05_raw_boxer_past3_birth_score05",
    )
    parser.add_argument("--expected-scene-count", type=int, default=100)
    parser.add_argument("--wait-timeout-seconds", type=float, default=86400.0)
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.add_argument(
        "--selection-policy",
        choices=("v1", "v2_m50", "v3_clip_vocab"),
        default="v1",
        help=(
            "Frozen terminal selector; v3_clip_vocab inserts a sidecar CLIP gate "
            "after v2 scalar/native gates and before NMS/cap."
        ),
    )
    parser.add_argument(
        "--clip-gate-sidecar",
        type=Path,
        default=None,
        help="No-GT CLIP decision JSON required by v3_clip_vocab.",
    )
    return parser


def main() -> None:
    args = _parser().parse_args()
    manifest = materialize_scannet_raw_boxer_past3_birth_full100(
        scene_list=args.scene_list,
        raw_log_root=args.raw_log_root,
        baseline_root=args.baseline_root,
        schedule_root=args.schedule_root,
        scene_rgbd_root=args.scene_rgbd_root,
        output_root=args.output_root,
        expected_scene_count=args.expected_scene_count,
        wait_timeout_seconds=args.wait_timeout_seconds,
        poll_seconds=args.poll_seconds,
        selection_policy=args.selection_policy,
        clip_gate_sidecar=args.clip_gate_sidecar,
    )
    print(
        json.dumps(
            {
                "schema": manifest["schema"],
                "scene_count": manifest["scene_count"],
                "native_count": manifest["native_count"],
                "past3_receipt_count": manifest["past3_receipt_count"],
                "birth_count": manifest["birth_count"],
                "output_root": os.fspath(args.output_root.resolve()),
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
