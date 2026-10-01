#!/usr/bin/env python3
"""Seal frozen OWLv2+Boxer CSVs into an output-inert candidate sidecar.

This tool deliberately has no dataset-annotation or evaluation input.  It
validates the preregistered inference metadata, restores the translation that
Boxer's ScanNet loader subtracts from every camera pose, and writes geometry
only.  OWLv2 labels are parsed for CSV integrity but are not exported.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import io
import json
import math
import os
import re
import tempfile
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


SCHEMA = "boxfusion.owl_boxer_shadow_candidates.v1"
SCENE_RE = re.compile(r"scene[0-9]{4}_[0-9]{2}\Z")

# Frozen clean-in2 model profile shared by the sealed v3/v4 shadow runs.
PROFILE = "clean_in2"
EXPECTED_OWL_CHECKPOINT = "owlv2-base-patch16-ensemble.pt"
EXPECTED_OWL_SHA256 = (
    "14aa78ffe7b13e5b3ebf55845bc9a07e339a095cfd88f4c4e8f726b38ce1ebbf"
)
EXPECTED_OWL_TEXT_CACHE = (
    "owlv2-base-patch16-ensemble_textemb_878186d327b0.pt"
)
EXPECTED_OWL_TEXT_CACHE_SHA256 = (
    "59193fc014d381b2200edf1c1e6dc86324edb55a067189d3e84226a184185283"
)
EXPECTED_BOXER_CHECKPOINT = "boxernet_hw960in2x6d768-c88128f8.ckpt"
EXPECTED_BOXER_SHA256 = (
    "d5a30b348a8f5b0e5990ff3aa0e8f473ce77d860da22586322e7f47abc83ca6f"
)
EXPECTED_RUN_BOXER_SHA256 = (
    "8ff93e62881db5bd4d0fb20cbddfb5767ec2c4a941e873672c3acf603ecdad1b"
)
EXPECTED_OWL_WRAPPER_SHA256 = (
    "7cf26a25bba1e67d8d8230ef47eb8288a48a728eda27d846e4f57bc6d4b6c628"
)
EXPECTED_BOXERNET_SOURCE_SHA256 = (
    "a8009c1c0932aaab98bb074a2a4c50e55a3fbdfc3c6cb1afc9e1aef0e5324130"
)
EXPECTED_DINOV3_CHECKPOINT = (
    "dinov3_vits16plus_pretrain_lvd1689m-4057cbaa.pth"
)
EXPECTED_DINOV3_SHA256 = (
    "4057cbaaad8c16657adb09d6815f28d4164eeba30532fde23f0d17313124caea"
)
EXPECTED_TAXONOMY = "lvisplus"
EXPECTED_TAXONOMY_COUNT = 1220
EXPECTED_TAXONOMY_SHA256 = (
    "3d6fd6fedb15ec5ea2f8ae80d2a5da310e64bece64aa38bb14f16cb7ac05cb3e"
)
EXPECTED_THRESH_2D = 0.25
EXPECTED_THRESH_3D = 0.5
EXPECTED_NMS_IOU = 0.5
EXPECTED_DETECTOR_HW = 960
EXPECTED_START_N = 1
EXPECTED_SKIP_N = 25

# Hard resource caps.  Overflow is either deterministically top-score capped
# (candidate rows) or rejected before unbounded input can be accumulated.
MAX_SCENES = 128
MAX_FRAMES_PER_SCENE = 512
MAX_PER_FRAME_CANDIDATES = 64
MAX_TRACKED_CANDIDATES_PER_SCENE = 256
MAX_INPUT_ROWS_PER_SCENE = 32768
MAX_TOTAL_PER_VIEW_CANDIDATES = 1_048_576
MAX_TOTAL_TRACKED_CANDIDATES = 32768
MAX_RAW_CSV_BYTES = 64 * 1024 * 1024
MAX_TRACKED_CSV_BYTES = 8 * 1024 * 1024
MAX_LOG_BYTES = 16 * 1024 * 1024
MAX_SCHEDULE_MANIFEST_BYTES = 16 * 1024 * 1024
MAX_LABEL_BYTES = 128

CSV_COLUMNS = (
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
GT_ACCESS_GUARD = "BOXFUSION_SHADOW_GT_ACCESS=forbidden annotation_path=None"


class SealError(RuntimeError):
    """Raised when a shadow artifact violates the sealed protocol."""


def _sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(chunk_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _require_file(path: Path, *, max_bytes: int | None = None) -> None:
    if not path.is_file():
        raise SealError(f"required file is absent: {path}")
    size = path.stat().st_size
    if max_bytes is not None and size > max_bytes:
        raise SealError(f"file exceeds hard cap ({size}>{max_bytes}): {path}")


def _read_scene_ids(path: Path) -> tuple[str, ...]:
    _require_file(path, max_bytes=1024 * 1024)
    scenes: list[str] = []
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if not SCENE_RE.fullmatch(line):
            raise SealError(f"invalid scene id at {path}:{line_number}: {line!r}")
        scenes.append(line)
    if not scenes:
        raise SealError("scene list is empty")
    if len(scenes) > MAX_SCENES:
        raise SealError(f"scene count exceeds hard cap ({len(scenes)}>{MAX_SCENES})")
    if len(set(scenes)) != len(scenes):
        raise SealError("scene list contains duplicates")
    return tuple(scenes)


def _parse_sealed_hashes(path: Path) -> dict[Path, str]:
    _require_file(path, max_bytes=1024 * 1024)
    result: dict[Path, str] = {}
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line:
            continue
        parts = line.split(maxsplit=1)
        if len(parts) != 2 or not re.fullmatch(r"[0-9a-f]{64}", parts[0]):
            raise SealError(f"invalid SHA-256 row at {path}:{line_number}")
        item = Path(parts[1].lstrip("* ")).resolve()
        if item in result:
            raise SealError(f"duplicate sealed-hash path: {item}")
        result[item] = parts[0]
    return result


def _validate_assets(boxer_root: Path, run_root: Path) -> dict[str, Any]:
    boxer_root = boxer_root.resolve()
    run_boxer = boxer_root / "run_boxer.py"
    boxer_checkpoint = boxer_root / "ckpts" / EXPECTED_BOXER_CHECKPOINT
    taxonomy = boxer_root / "owl" / f"{EXPECTED_TAXONOMY}_classes.csv"
    owl_wrapper = boxer_root / "owl" / "owl_wrapper.py"
    boxernet_source = boxer_root / "boxernet" / "boxernet.py"
    for path in (
        run_boxer,
        boxer_checkpoint,
        taxonomy,
        owl_wrapper,
        boxernet_source,
    ):
        _require_file(path)

    sealed_path = run_root / "frozen_inputs_sha256.txt"
    sealed = _parse_sealed_hashes(sealed_path)
    owl_candidates = [
        path
        for path, digest in sealed.items()
        if path.name == EXPECTED_OWL_CHECKPOINT and digest == EXPECTED_OWL_SHA256
    ]
    if len(owl_candidates) != 1:
        raise SealError(
            "frozen-input ledger must identify exactly one pinned OWLv2 checkpoint"
        )
    owl_checkpoint = owl_candidates[0]
    _require_file(owl_checkpoint)
    text_cache_candidates = [
        path
        for path, digest in sealed.items()
        if path.name == EXPECTED_OWL_TEXT_CACHE
        and digest == EXPECTED_OWL_TEXT_CACHE_SHA256
    ]
    if len(text_cache_candidates) != 1:
        raise SealError(
            "frozen-input ledger must identify exactly one pinned OWLv2 text cache"
        )
    owl_text_cache = text_cache_candidates[0]
    _require_file(owl_text_cache)
    dino_candidates = [
        path
        for path, digest in sealed.items()
        if path.name == EXPECTED_DINOV3_CHECKPOINT
        and digest == EXPECTED_DINOV3_SHA256
    ]
    if len(dino_candidates) != 1:
        raise SealError(
            "frozen-input ledger must identify exactly one pinned DINOv3 checkpoint"
        )
    dinov3_checkpoint = dino_candidates[0]
    _require_file(dinov3_checkpoint)

    actual = {
        "run_boxer": _sha256_file(run_boxer),
        "owl_checkpoint": _sha256_file(owl_checkpoint),
        "owl_text_cache": _sha256_file(owl_text_cache),
        "boxer_checkpoint": _sha256_file(boxer_checkpoint),
        "dinov3_checkpoint": _sha256_file(dinov3_checkpoint),
        "taxonomy": _sha256_file(taxonomy),
        "owl_wrapper": _sha256_file(owl_wrapper),
        "boxernet_source": _sha256_file(boxernet_source),
    }
    expected = {
        "run_boxer": EXPECTED_RUN_BOXER_SHA256,
        "owl_checkpoint": EXPECTED_OWL_SHA256,
        "owl_text_cache": EXPECTED_OWL_TEXT_CACHE_SHA256,
        "boxer_checkpoint": EXPECTED_BOXER_SHA256,
        "dinov3_checkpoint": EXPECTED_DINOV3_SHA256,
        "taxonomy": EXPECTED_TAXONOMY_SHA256,
        "owl_wrapper": EXPECTED_OWL_WRAPPER_SHA256,
        "boxernet_source": EXPECTED_BOXERNET_SOURCE_SHA256,
    }
    for key, wanted in expected.items():
        if actual[key] != wanted:
            raise SealError(
                f"frozen {key} SHA-256 mismatch: expected={wanted}, actual={actual[key]}"
            )

    labels = [line.strip() for line in taxonomy.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(labels) != EXPECTED_TAXONOMY_COUNT:
        raise SealError(
            "frozen taxonomy row-count mismatch: "
            f"expected={EXPECTED_TAXONOMY_COUNT}, actual={len(labels)}"
        )
    if len(set(labels)) != len(labels):
        raise SealError("frozen taxonomy contains duplicate prompts")

    nms_default = _owl_nms_default(owl_wrapper)
    if not math.isclose(nms_default, EXPECTED_NMS_IOU, rel_tol=0.0, abs_tol=1e-12):
        raise SealError(
            f"OWLv2 NMS default mismatch: expected={EXPECTED_NMS_IOU}, actual={nms_default}"
        )

    for path, key in (
        (run_boxer, "run_boxer"),
        (owl_wrapper, "owl_wrapper"),
        (boxernet_source, "boxernet_source"),
        (owl_checkpoint, "owl_checkpoint"),
        (owl_text_cache, "owl_text_cache"),
        (boxer_checkpoint, "boxer_checkpoint"),
        (dinov3_checkpoint, "dinov3_checkpoint"),
    ):
        resolved = path.resolve()
        if resolved not in sealed:
            raise SealError(f"asset missing from frozen-input ledger: {resolved}")
        if sealed[resolved] != actual[key]:
            raise SealError(
                f"asset changed after runner ledger was written: {resolved}"
            )

    return {
        "root": str(boxer_root),
        "profile": PROFILE,
        "detector": "owl",
        "owl_checkpoint": EXPECTED_OWL_CHECKPOINT,
        "owl_checkpoint_path": str(owl_checkpoint),
        "owl_checkpoint_sha256": actual["owl_checkpoint"],
        "owl_text_cache": EXPECTED_OWL_TEXT_CACHE,
        "owl_text_cache_path": str(owl_text_cache),
        "owl_text_cache_sha256": actual["owl_text_cache"],
        "boxer_checkpoint": EXPECTED_BOXER_CHECKPOINT,
        "boxer_checkpoint_sha256": actual["boxer_checkpoint"],
        "dinov3_checkpoint": EXPECTED_DINOV3_CHECKPOINT,
        "dinov3_checkpoint_path": str(dinov3_checkpoint),
        "dinov3_checkpoint_sha256": actual["dinov3_checkpoint"],
        "taxonomy": EXPECTED_TAXONOMY,
        "taxonomy_count": len(labels),
        "taxonomy_sha256": actual["taxonomy"],
        "run_boxer_sha256": actual["run_boxer"],
        "owl_wrapper_sha256": actual["owl_wrapper"],
        "boxernet_source_sha256": actual["boxernet_source"],
        "threshold_2d": EXPECTED_THRESH_2D,
        "threshold_3d": EXPECTED_THRESH_3D,
        "nms_iou_2d": EXPECTED_NMS_IOU,
        "detector_hw": EXPECTED_DETECTOR_HW,
        "start_n": EXPECTED_START_N,
        "skip_n": EXPECTED_SKIP_N,
    }


def _owl_nms_default(path: Path) -> float:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "OwlWrapper":
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) and item.name == "__init__":
                    args = item.args.args
                    defaults = item.args.defaults
                    mapping = dict(zip((arg.arg for arg in args[-len(defaults) :]), defaults))
                    if "nms_iou_threshold" not in mapping:
                        break
                    value = ast.literal_eval(mapping["nms_iou_threshold"])
                    return float(value)
    raise SealError(f"could not verify OwlWrapper NMS default in {path}")


def _parse_namespace_log(path: Path) -> tuple[dict[str, Any], str, bool]:
    _require_file(path, max_bytes=MAX_LOG_BYTES)
    text = path.read_text(encoding="utf-8", errors="strict")
    lines = text.splitlines()
    guard_verified = bool(lines and lines[0].strip() == GT_ACCESS_GUARD)
    namespace_line = (
        lines[1]
        if guard_verified and len(lines) > 1
        else (lines[0] if lines else "")
    )
    try:
        expression = ast.parse(namespace_line, mode="eval").body
    except SyntaxError as error:
        raise SealError(f"invalid runner metadata line in {path}: {error}") from error
    if not (
        isinstance(expression, ast.Call)
        and isinstance(expression.func, ast.Name)
        and expression.func.id == "Namespace"
        and not expression.args
    ):
        raise SealError(f"runner log does not begin with argparse Namespace: {path}")
    values: dict[str, Any] = {}
    for keyword in expression.keywords:
        if keyword.arg is None or keyword.arg in values:
            raise SealError(f"invalid duplicate/expanded runner metadata in {path}")
        try:
            values[keyword.arg] = ast.literal_eval(keyword.value)
        except (ValueError, SyntaxError) as error:
            raise SealError(f"non-literal runner metadata in {path}") from error
    return values, text, guard_verified


def _validate_run_metadata(
    metadata: Mapping[str, Any],
    log_text: str,
    *,
    scene_id: str,
    scene_root: Path,
    run_root: Path,
    boxer_root: Path,
) -> None:
    expected: dict[str, Any] = {
        "detector": "owl",
        "thresh2d": EXPECTED_THRESH_2D,
        "thresh3d": EXPECTED_THRESH_3D,
        "labels": [EXPECTED_TAXONOMY],
        "detector_hw": EXPECTED_DETECTOR_HW,
        "start_n": EXPECTED_START_N,
        "skip_n": EXPECTED_SKIP_N,
        "track": True,
        "fuse": False,
        "gt2d": False,
        "cache2d": False,
        "cache3d": False,
        "no_sdp": False,
        "no_csv": False,
        "force_cpu": False,
        "force_precision": "bfloat16",
        "skip_viz": True,
        "write_name": "boxer",
    }
    for key, wanted in expected.items():
        if key not in metadata:
            raise SealError(f"runner metadata lacks {key!r} for {scene_id}")
        actual = metadata[key]
        if isinstance(wanted, float):
            matches = isinstance(actual, (int, float)) and math.isclose(
                float(actual), wanted, rel_tol=0.0, abs_tol=1e-12
            )
        else:
            matches = actual == wanted
        if not matches:
            raise SealError(
                f"runner metadata mismatch for {scene_id}.{key}: "
                f"expected={wanted!r}, actual={actual!r}"
            )

    expected_input = (scene_root / scene_id).resolve()
    expected_output = (run_root / "boxer_raw").resolve()
    expected_ckpt = (boxer_root / "ckpts" / EXPECTED_BOXER_CHECKPOINT).resolve()
    path_fields = {
        "input": expected_input,
        "output_dir": expected_output,
        "ckpt": expected_ckpt,
    }
    for key, wanted in path_fields.items():
        if key not in metadata or Path(str(metadata[key])).resolve() != wanted:
            raise SealError(
                f"runner path mismatch for {scene_id}.{key}: expected={wanted}, "
                f"actual={metadata.get(key)!r}"
            )

    completion_markers = (
        f"ScanNetLoader: {scene_id}, {metadata.get('max_n')} frames, 0 3D boxes",
        "Loaded OWLv2 on cuda with "
        f"{EXPECTED_TAXONOMY_COUNT} text prompts",
        f'Loading checkpoint from "{expected_ckpt}"',
        "Saved 3D BBs to",
        "Saved 2D BBs to",
    )
    for marker in completion_markers:
        if marker not in log_text:
            raise SealError(f"runner log is incomplete for {scene_id}: missing {marker!r}")
    tracked_written_count = log_text.count("tracked OBBs to")
    tracked_empty_count = sum(
        line.strip() == "==> 0 active tracks from inline tracker"
        for line in log_text.splitlines()
    )
    if (tracked_written_count, tracked_empty_count) not in ((1, 0), (0, 1)):
        raise SealError(
            f"runner log has ambiguous terminal-track status for {scene_id}: "
            f"written_count={tracked_written_count}, "
            f"empty_count={tracked_empty_count}"
        )


def _valid_frame_ids_and_offset(scene_dir: Path) -> tuple[set[int], int, np.ndarray]:
    color_dir = scene_dir / "frames" / "color"
    pose_dir = scene_dir / "frames" / "pose"
    if not color_dir.is_dir() or not pose_dir.is_dir():
        raise SealError(f"ScanNet RGB/pose directories are absent: {scene_dir}")
    frame_ids: set[int] = set()
    for path in color_dir.iterdir():
        if path.suffix.lower() not in (".png", ".jpg", ".jpeg"):
            continue
        try:
            frame_ids.add(int(path.stem))
        except ValueError as error:
            raise SealError(f"non-numeric ScanNet frame name: {path.name}") from error
    selected = sorted(frame_ids)[EXPECTED_START_N - 1 :: EXPECTED_SKIP_N]
    valid: list[int] = []
    first_pose: np.ndarray | None = None
    first_frame = -1
    for frame_id in selected:
        pose_path = pose_dir / f"{frame_id}.txt"
        if not pose_path.is_file():
            continue
        try:
            pose = np.loadtxt(pose_path, dtype=np.float64)
        except (OSError, ValueError) as error:
            raise SealError(f"could not read ScanNet pose {pose_path}: {error}") from error
        if pose.shape != (4, 4) or not np.all(np.isfinite(pose)):
            continue
        valid.append(frame_id)
        if first_pose is None:
            first_pose = pose
            first_frame = frame_id
    if first_pose is None:
        raise SealError(f"scene has no valid sampled ScanNet pose: {scene_dir}")
    return set(valid), first_frame, np.ascontiguousarray(first_pose[:3, 3])


def _validate_native_identity(run_root: Path, scenes: Sequence[str]) -> str:
    before = run_root / "native_before_sha256.txt"
    after = run_root / "native_after_sha256.txt"
    _require_file(before, max_bytes=1024 * 1024)
    _require_file(after, max_bytes=1024 * 1024)
    before_bytes = before.read_bytes()
    after_bytes = after.read_bytes()
    if before_bytes != after_bytes:
        raise SealError("native prediction hashes changed during the shadow run")
    rows = _parse_sealed_hashes(before)
    if len(rows) != len(scenes):
        raise SealError(
            "native identity ledger count does not match the sealed scene list"
        )
    names = {path.name for path in rows}
    expected_names = {f"{scene}_boxes.pkl" for scene in scenes}
    if names != expected_names:
        raise SealError("native identity ledger scenes do not match the sealed scene list")
    return hashlib.sha256(before_bytes).hexdigest()


def _load_sealed_schedules(
    run_root: Path, scenes: Sequence[str]
) -> dict[str, dict[str, Any]]:
    ledger = _parse_sealed_hashes(run_root / "frozen_inputs_sha256.txt")
    schedules: dict[str, dict[str, Any]] = {}
    for scene_id in scenes:
        candidates = [
            (path, digest)
            for path, digest in ledger.items()
            if path.name == "manifest.json" and path.parent.name == scene_id
        ]
        if len(candidates) != 1:
            raise SealError(
                f"frozen-input ledger must identify one schedule manifest for {scene_id}"
            )
        path, sealed_digest = candidates[0]
        _require_file(path, max_bytes=MAX_SCHEDULE_MANIFEST_BYTES)
        actual_digest = _sha256_file(path)
        if actual_digest != sealed_digest:
            raise SealError(f"sealed schedule manifest changed after inference: {path}")
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise SealError(f"invalid sealed schedule manifest {path}: {error}") from error
        if not isinstance(value, dict):
            raise SealError(f"sealed schedule manifest is not an object: {path}")
        count = value.get("record_count")
        frame_ids = value.get("recorded_frame_ids")
        if (
            not isinstance(count, int)
            or count < 1
            or count > MAX_FRAMES_PER_SCENE
            or not isinstance(frame_ids, list)
            or len(frame_ids) != count
            or any(not isinstance(frame_id, int) or frame_id < 0 for frame_id in frame_ids)
            or len(set(frame_ids)) != len(frame_ids)
            or frame_ids != sorted(frame_ids)
        ):
            raise SealError(f"invalid recorded schedule in {path}")
        schedules[scene_id] = {
            "path": path,
            "sha256": actual_digest,
            "record_count": count,
            "frame_ids": tuple(frame_ids),
        }
    return schedules


def _parse_int(value: str, *, field: str, path: Path, row_number: int) -> int:
    try:
        return int(value)
    except ValueError as error:
        raise SealError(f"invalid integer {field} at {path}:{row_number}") from error


def _parse_float(value: str, *, field: str, path: Path, row_number: int) -> float:
    try:
        result = float(value)
    except ValueError as error:
        raise SealError(f"invalid float {field} at {path}:{row_number}") from error
    if not math.isfinite(result):
        raise SealError(f"non-finite {field} at {path}:{row_number}")
    return result


def _read_obb_csv(path: Path, *, max_bytes: int) -> list[dict[str, Any]]:
    _require_file(path, max_bytes=max_bytes)
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != CSV_COLUMNS:
            raise SealError(
                f"unexpected Boxer CSV schema in {path}: {reader.fieldnames!r}"
            )
        for source_row, row in enumerate(reader):
            row_number = source_row + 2
            if source_row >= MAX_INPUT_ROWS_PER_SCENE:
                raise SealError(
                    f"CSV row count exceeds hard cap ({MAX_INPUT_ROWS_PER_SCENE}): {path}"
                )
            if None in row or any(row[name] is None for name in CSV_COLUMNS):
                raise SealError(f"malformed CSV row at {path}:{row_number}")
            name = row["name"]
            if len(name.encode("utf-8")) > MAX_LABEL_BYTES:
                raise SealError(f"source label exceeds hard cap at {path}:{row_number}")

            center = np.asarray(
                [
                    _parse_float(row[name], field=name, path=path, row_number=row_number)
                    for name in (
                        "tx_world_object",
                        "ty_world_object",
                        "tz_world_object",
                    )
                ],
                dtype=np.float64,
            )
            quaternion = np.asarray(
                [
                    _parse_float(row[name], field=name, path=path, row_number=row_number)
                    for name in (
                        "qw_world_object",
                        "qx_world_object",
                        "qy_world_object",
                        "qz_world_object",
                    )
                ],
                dtype=np.float64,
            )
            extent = np.asarray(
                [
                    _parse_float(row[name], field=name, path=path, row_number=row_number)
                    for name in ("scale_x", "scale_y", "scale_z")
                ],
                dtype=np.float64,
            )
            probability = _parse_float(
                row["prob"], field="prob", path=path, row_number=row_number
            )
            if np.max(np.abs(center)) > 10_000.0:
                raise SealError(f"implausible OBB center at {path}:{row_number}")
            if np.any(extent <= 0.0) or np.max(extent) > 100.0:
                raise SealError(f"invalid OBB extent at {path}:{row_number}")
            quaternion_norm = float(np.linalg.norm(quaternion))
            if not math.isclose(quaternion_norm, 1.0, rel_tol=0.0, abs_tol=5e-3):
                raise SealError(f"invalid OBB quaternion at {path}:{row_number}")
            if not (0.0 <= probability <= 1.0):
                raise SealError(f"invalid OBB probability at {path}:{row_number}")
            rows.append(
                {
                    "frame_id": _parse_int(
                        row["time_ns"], field="time_ns", path=path, row_number=row_number
                    ),
                    "center_recentered": center,
                    "quaternion_wxyz": quaternion / quaternion_norm,
                    "extent_xyz": extent,
                    "instance_id": _parse_int(
                        row["instance"], field="instance", path=path, row_number=row_number
                    ),
                    # Parsed only to reject malformed CSV.  Neither field is exported.
                    "semantic_id": _parse_int(
                        row["sem_id"], field="sem_id", path=path, row_number=row_number
                    ),
                    "probability": probability,
                    "source_row": source_row,
                }
            )
    return rows


def _stack(rows: Sequence[dict[str, Any]], key: str, width: int) -> np.ndarray:
    if not rows:
        return np.empty((0, width), dtype=np.float32)
    return np.ascontiguousarray(np.stack([row[key] for row in rows]).astype(np.float32))


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
    with path.open("wb") as raw:
        with zipfile.ZipFile(
            raw,
            mode="w",
            compression=zipfile.ZIP_DEFLATED,
            compresslevel=9,
        ) as archive:
            for name in sorted(arrays):
                payload = io.BytesIO()
                np.lib.format.write_array(
                    payload,
                    np.ascontiguousarray(arrays[name]),
                    allow_pickle=False,
                )
                info = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
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


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    payload = (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def seal_candidates(
    *,
    run_root: Path,
    scene_root: Path,
    scene_list: Path,
    boxer_root: Path,
    output_json: Path,
    output_npz: Path,
) -> dict[str, Any]:
    """Validate and seal one fixed OWLv2+Boxer shadow run."""

    run_root = run_root.resolve()
    scene_root = scene_root.resolve()
    boxer_root = boxer_root.resolve()
    output_json = output_json.resolve()
    output_npz = output_npz.resolve()
    if output_json == output_npz:
        raise SealError("JSON and NPZ outputs must be different paths")
    if output_json.parent != output_npz.parent:
        raise SealError("JSON and NPZ outputs must share one directory")
    if output_json.exists() or output_npz.exists():
        raise SealError("sealed output already exists; refusing to overwrite")
    output_json.parent.mkdir(parents=True, exist_ok=True)

    scenes = _read_scene_ids(scene_list.resolve())
    assets = _validate_assets(boxer_root, run_root)
    native_identity_sha256 = _validate_native_identity(run_root, scenes)
    schedules = _load_sealed_schedules(run_root, scenes)

    per_view_rows: list[dict[str, Any]] = []
    tracked_rows: list[dict[str, Any]] = []
    scene_summaries: list[dict[str, Any]] = []

    for scene_index, scene_id in enumerate(scenes):
        scene_dir = scene_root / scene_id
        raw_dir = run_root / "boxer_raw" / scene_id
        raw_csv = raw_dir / "boxer_3dbbs.csv"
        tracked_csv = raw_dir / "boxer_3dbbs_tracked.csv"
        owl_csv = raw_dir / "owl_2dbbs.csv"
        log_path = run_root / "scenes" / f"{scene_id}.log"
        _require_file(owl_csv, max_bytes=MAX_RAW_CSV_BYTES)

        run_metadata, log_text, gt_access_guard_verified = _parse_namespace_log(
            log_path
        )
        if not gt_access_guard_verified:
            raise SealError(f"strict no-GT access guard is absent for {scene_id}")
        _validate_run_metadata(
            run_metadata,
            log_text,
            scene_id=scene_id,
            scene_root=scene_root,
            run_root=run_root,
            boxer_root=boxer_root,
        )
        valid_frame_ids, offset_frame_id, world_offset = _valid_frame_ids_and_offset(
            scene_dir
        )
        schedule = schedules[scene_id]
        recorded_schedule_frame_ids = set(schedule["frame_ids"])
        valid_schedule_frame_ids = recorded_schedule_frame_ids.intersection(
            valid_frame_ids
        )
        invalid_schedule_frame_ids = recorded_schedule_frame_ids.difference(
            valid_frame_ids
        )
        runner_max_n = run_metadata.get("max_n")
        if runner_max_n == len(valid_schedule_frame_ids):
            schedule_mode = "valid_recorded_frames"
            schedule_frame_ids = valid_schedule_frame_ids
        elif runner_max_n == schedule["record_count"]:
            # Legacy v3 ran the released loader by count.  Keep support for
            # sealing that immutable artifact, including its explicit tail
            # exclusion diagnostics.
            schedule_mode = "legacy_record_count"
            schedule_frame_ids = recorded_schedule_frame_ids
        else:
            raise SealError(
                f"runner keyframe count mismatch for {scene_id}: "
                f"recorded={schedule['record_count']}, "
                f"valid_recorded={len(valid_schedule_frame_ids)}, "
                f"actual={runner_max_n!r}"
            )
        boxer_frame_ids = set(sorted(valid_frame_ids)[: int(runner_max_n)])
        if len(boxer_frame_ids) != int(runner_max_n):
            raise SealError(
                f"Boxer input has fewer valid sampled frames than the sealed schedule "
                f"for {scene_id}"
            )
        progress_marker = f"{runner_max_n}/{runner_max_n}"
        if progress_marker not in log_text:
            raise SealError(
                f"runner log lacks completed keyframe marker for {scene_id}: "
                f"{progress_marker}"
            )
        raw = _read_obb_csv(raw_csv, max_bytes=MAX_RAW_CSV_BYTES)
        tracked_empty = any(
            line.strip() == "==> 0 active tracks from inline tracker"
            for line in log_text.splitlines()
        )
        if tracked_csv.is_file():
            tracked = _read_obb_csv(tracked_csv, max_bytes=MAX_TRACKED_CSV_BYTES)
            if tracked_empty and tracked:
                raise SealError(
                    f"runner reported zero active tracks but wrote terminal rows: "
                    f"{tracked_csv}"
                )
            tracked_csv_sha256: str | None = _sha256_file(tracked_csv)
        else:
            if not tracked_empty:
                raise SealError(f"required file is absent: {tracked_csv}")
            tracked = []
            tracked_csv_sha256 = None

        grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
        extra_schedule_rows = 0
        extra_schedule_frame_ids: set[int] = set()
        for row in raw:
            frame_id = row["frame_id"]
            if frame_id not in schedule_frame_ids:
                extra_schedule_rows += 1
                extra_schedule_frame_ids.add(frame_id)
                continue
            grouped[frame_id].append(row)
        if len(grouped) > MAX_FRAMES_PER_SCENE:
            raise SealError(
                f"sampled frame count exceeds hard cap for {scene_id}: {len(grouped)}"
            )

        kept_raw: list[dict[str, Any]] = []
        dropped_raw = 0
        for frame_id in sorted(grouped):
            ordered = sorted(
                grouped[frame_id],
                key=lambda row: (-row["probability"], row["source_row"]),
            )
            kept_raw.extend(ordered[:MAX_PER_FRAME_CANDIDATES])
            dropped_raw += max(0, len(ordered) - MAX_PER_FRAME_CANDIDATES)

        ordered_tracks = sorted(
            tracked,
            key=lambda row: (-row["probability"], row["source_row"]),
        )
        # Terminal tracks have no per-observation provenance in the released
        # CSV.  If the loader processed even one frame outside the sealed T05
        # schedule, no terminal row can be proven independent of that frame.
        # Conservatively exclude the entire terminal pool for that scene.
        tracked_schedule_clean = not extra_schedule_frame_ids
        if tracked_schedule_clean:
            kept_tracks = ordered_tracks[:MAX_TRACKED_CANDIDATES_PER_SCENE]
            dropped_tracks_contaminated = 0
        else:
            kept_tracks = []
            dropped_tracks_contaminated = len(ordered_tracks)
        dropped_tracks = max(
            0,
            len(ordered_tracks) - len(kept_tracks) - dropped_tracks_contaminated,
        )
        if any(row["frame_id"] != 0 for row in kept_tracks):
            raise SealError(f"terminal tracked CSV must use time_ns=0: {tracked_csv}")

        missing_candidate_frame_ids = schedule_frame_ids.difference(grouped)
        missing_invalid_pose_frame_ids = missing_candidate_frame_ids.difference(
            valid_frame_ids
        )
        missing_zero_candidate_frame_ids = missing_candidate_frame_ids.intersection(
            valid_frame_ids
        )

        for rank, row in enumerate(kept_raw):
            item = dict(row)
            item["scene_index"] = scene_index
            item["candidate_rank"] = rank
            item["center_world"] = row["center_recentered"] + world_offset
            per_view_rows.append(item)
        for rank, row in enumerate(kept_tracks):
            item = dict(row)
            item["scene_index"] = scene_index
            item["candidate_rank"] = rank
            item["center_world"] = row["center_recentered"] + world_offset
            tracked_rows.append(item)

        if len(per_view_rows) > MAX_TOTAL_PER_VIEW_CANDIDATES:
            raise SealError("global per-view candidate cap exceeded")
        if len(tracked_rows) > MAX_TOTAL_TRACKED_CANDIDATES:
            raise SealError("global tracked candidate cap exceeded")

        scene_summaries.append(
            {
                "scene_id": scene_id,
                "scene_index": scene_index,
                "gt_access_guard_verified": gt_access_guard_verified,
                "world_offset_frame_id": offset_frame_id,
                "world_offset_xyz": world_offset.tolist(),
                "sampled_valid_frame_count": len(valid_frame_ids),
                "sealed_manifest_record_count": schedule["record_count"],
                "sealed_schedule_frame_count": len(schedule_frame_ids),
                "sealed_schedule_mode": schedule_mode,
                "sealed_schedule_invalid_pose_frame_ids_excluded": sorted(
                    invalid_schedule_frame_ids
                    if schedule_mode == "valid_recorded_frames"
                    else ()
                ),
                "sealed_schedule_manifest_sha256": schedule["sha256"],
                "schedule_vs_boxer_frame_id_symmetric_difference": len(
                    schedule_frame_ids.symmetric_difference(boxer_frame_ids)
                ),
                "runner_max_n": run_metadata["max_n"],
                "per_view_input_rows": len(raw),
                "per_view_schedule_rows": len(raw) - extra_schedule_rows,
                "per_view_extra_schedule_rows_excluded": extra_schedule_rows,
                "per_view_extra_schedule_frame_ids_excluded": sorted(
                    extra_schedule_frame_ids
                ),
                "per_view_kept_rows": len(kept_raw),
                "per_view_cap_dropped_rows": dropped_raw,
                "schedule_frames_without_candidates": sorted(
                    missing_candidate_frame_ids
                ),
                "schedule_frames_without_candidates_invalid_pose": sorted(
                    missing_invalid_pose_frame_ids
                ),
                "schedule_frames_without_candidates_valid_pose": sorted(
                    missing_zero_candidate_frame_ids
                ),
                "tracked_input_rows": len(tracked),
                "tracked_kept_rows": len(kept_tracks),
                "tracked_cap_dropped_rows": dropped_tracks,
                "tracked_csv_present": tracked_csv.is_file(),
                "tracked_zero_active_verified": tracked_empty,
                "tracked_schedule_clean": tracked_schedule_clean,
                "tracked_schedule_contaminated_rows_excluded": (
                    dropped_tracks_contaminated
                ),
                "inputs": {
                    "boxer_3dbbs_csv_sha256": _sha256_file(raw_csv),
                    "boxer_3dbbs_tracked_csv_sha256": tracked_csv_sha256,
                    "owl_2dbbs_csv_sha256": _sha256_file(owl_csv),
                    "runner_log_sha256": _sha256_file(log_path),
                },
            }
        )

    arrays: dict[str, np.ndarray] = {
        "scene_ids": np.asarray(scenes, dtype="<U12"),
        "per_view_scene_index": np.asarray(
            [row["scene_index"] for row in per_view_rows], dtype=np.int16
        ),
        "per_view_frame_id": np.asarray(
            [row["frame_id"] for row in per_view_rows], dtype=np.int64
        ),
        "per_view_source_row": np.asarray(
            [row["source_row"] for row in per_view_rows], dtype=np.int32
        ),
        "per_view_source_instance_id": np.asarray(
            [row["instance_id"] for row in per_view_rows], dtype=np.int32
        ),
        "per_view_center_world": _stack(per_view_rows, "center_world", 3),
        "per_view_quaternion_wxyz": _stack(per_view_rows, "quaternion_wxyz", 4),
        "per_view_extent_xyz": _stack(per_view_rows, "extent_xyz", 3),
        "per_view_source_score": np.asarray(
            [row["probability"] for row in per_view_rows], dtype=np.float32
        ),
        "tracked_scene_index": np.asarray(
            [row["scene_index"] for row in tracked_rows], dtype=np.int16
        ),
        "tracked_source_row": np.asarray(
            [row["source_row"] for row in tracked_rows], dtype=np.int32
        ),
        "tracked_instance_id": np.asarray(
            [row["instance_id"] for row in tracked_rows], dtype=np.int32
        ),
        "tracked_center_world": _stack(tracked_rows, "center_world", 3),
        "tracked_quaternion_wxyz": _stack(tracked_rows, "quaternion_wxyz", 4),
        "tracked_extent_xyz": _stack(tracked_rows, "extent_xyz", 3),
        "tracked_source_score": np.asarray(
            [row["probability"] for row in tracked_rows], dtype=np.float32
        ),
    }
    for value in arrays.values():
        value.setflags(write=False)
    content_sha256 = _array_content_sha256(arrays)

    descriptor, temporary_npz_name = tempfile.mkstemp(
        prefix=f".{output_npz.name}.", dir=output_npz.parent
    )
    os.close(descriptor)
    temporary_npz = Path(temporary_npz_name)
    try:
        _write_deterministic_npz(temporary_npz, arrays)
        npz_sha256 = _sha256_file(temporary_npz)
        manifest: dict[str, Any] = {
            "schema": SCHEMA,
            "profile": PROFILE,
            "mode": "shadow",
            "output_inert": True,
            "birth": False,
            "gt_access": False,
            "gt_access_guard": GT_ACCESS_GUARD,
            "gt_access_guard_verified": all(
                scene["gt_access_guard_verified"] for scene in scene_summaries
            ),
            "semantic_source_exported": False,
            "native_clip_unchanged": True,
            "native_before_after_identity": True,
            "native_identity_ledger_sha256": native_identity_sha256,
            "coordinate_frame": "scannet_world",
            "world_offset_rule": (
                "center_world=center_boxer_recentered+translation_of_first_valid_"
                "sampled_camera_pose"
            ),
            "assets_and_protocol": assets,
            "caps": {
                "max_scenes": MAX_SCENES,
                "max_frames_per_scene": MAX_FRAMES_PER_SCENE,
                "max_per_frame_candidates": MAX_PER_FRAME_CANDIDATES,
                "max_tracked_candidates_per_scene": MAX_TRACKED_CANDIDATES_PER_SCENE,
                "max_input_rows_per_scene": MAX_INPUT_ROWS_PER_SCENE,
                "max_total_per_view_candidates": MAX_TOTAL_PER_VIEW_CANDIDATES,
                "max_total_tracked_candidates": MAX_TOTAL_TRACKED_CANDIDATES,
            },
            "scene_count": len(scenes),
            "per_view_candidate_count": len(per_view_rows),
            "tracked_candidate_count": len(tracked_rows),
            "candidate_content_sha256": content_sha256,
            "npz_file": output_npz.name,
            "npz_sha256": npz_sha256,
            "scenes": scene_summaries,
        }
        os.replace(temporary_npz, output_npz)
        _atomic_json(output_json, manifest)
    finally:
        if temporary_npz.exists():
            temporary_npz.unlink()
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--scene-root", type=Path, required=True)
    parser.add_argument("--scene-list", type=Path, required=True)
    parser.add_argument("--boxer-root", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-npz", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        manifest = seal_candidates(
            run_root=args.run_root,
            scene_root=args.scene_root,
            scene_list=args.scene_list,
            boxer_root=args.boxer_root,
            output_json=args.output_json,
            output_npz=args.output_npz,
        )
    except SealError as error:
        raise SystemExit(f"seal failed: {error}") from error
    print(
        json.dumps(
            {
                "schema": manifest["schema"],
                "scene_count": manifest["scene_count"],
                "per_view_candidate_count": manifest["per_view_candidate_count"],
                "tracked_candidate_count": manifest["tracked_candidate_count"],
                "npz_sha256": manifest["npz_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
