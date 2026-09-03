#!/usr/bin/env python3
"""Seal and verify the exact no-GT ScanNet inputs of a proposal-cache stream.

The proposal-cache manifest stores hashes of the *processed* arrays presented
to ``ProposalCache.record``.  Hashing the source JPEG/PNG/text files alone is
not equivalent: JPEG decoding and resizing are runtime dependent.  This tool
therefore reconstructs the five producer inputs and executes the cache
producer's own ``_input_signature`` function, extracted from a pinned source
file.  A supplemental raw-file ledger is also recorded to catch changes that
may happen to decode to the same arrays.

No ground-truth path is accepted or discovered.  The only scene files opened
are scheduled RGB, depth and pose files plus the four ScanNet calibration
files.  ``seal`` is create-only; ``verify`` is read-only and fail-closed.
"""

from __future__ import annotations

import argparse
import ast
import cv2
import hashlib
import json
import math
import os
import platform
import re
from pathlib import Path
import sys
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch


SCHEMA = "boxfusion.scannet_stream_input_seal.v1"
ALLOWED_SCHEDULE_SCHEMAS = frozenset(
    {
        "boxfusion.cutr_postfilter_cache.v2",
        "boxfusion.cutr_postfilter_cache.v3",
    }
)
INPUT_FIELDS = (
    "image",
    "depth",
    "image_K",
    "depth_K",
    "camera_to_world",
)
INTRINSIC_FILES = (
    "intrinsic_color.txt",
    "intrinsic_depth.txt",
    "extrinsic_color.txt",
    "extrinsic_depth.txt",
)
SCENE_RE = re.compile(r"^scene[0-9]{4}_[0-9]{2}$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
FORBIDDEN_PATH_MARKERS = (
    "full_annotations.json",
    "_bbox.npy",
    "/oracle/",
    "/scannet_train_detection_data/",
)
SIGNATURE_FUNCTIONS = frozenset(
    {
        "_canonical_tensor",
        "_tensor_sha256",
        "_array_sha256",
        "_shape_of",
        "_input_signature",
    }
)
SAFE_SIGNATURE_DIRECT_CALLS = frozenset(
    {
        "ProposalCacheError",
        "_array_sha256",
        "_canonical_tensor",
        "_shape_of",
        "_tensor_sha256",
        "int",
        "isinstance",
        "len",
        "repr",
        "set",
        "str",
        "tuple",
        "type",
    }
)
SAFE_SIGNATURE_METHOD_CALLS = frozenset(
    {
        "asarray",
        "ascontiguousarray",
        "clone",
        "contiguous",
        "cpu",
        "detach",
        "encode",
        "hexdigest",
        "keys",
        "numpy",
        "sha256",
        "tobytes",
        "update",
        "view",
    }
)


class StreamInputError(RuntimeError):
    """Raised when the no-GT stream-input contract cannot be proven."""


def _safe_resolve(path: Path, label: str, *, directory: bool = False) -> Path:
    try:
        resolved = path.expanduser().resolve(strict=True)
    except OSError as error:
        raise StreamInputError(f"missing {label}: {path}") from error
    lowered = os.fspath(resolved).lower().replace("\\", "/")
    if any(marker in lowered for marker in FORBIDDEN_PATH_MARKERS):
        raise StreamInputError(f"forbidden GT/oracle path for {label}: {resolved}")
    if directory:
        if not resolved.is_dir():
            raise StreamInputError(f"{label} must be a directory: {resolved}")
    elif not resolved.is_file():
        raise StreamInputError(f"{label} must be a file: {resolved}")
    return resolved


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("ascii")


def _load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise StreamInputError(f"invalid {label}: {path}") from error
    if not isinstance(value, dict):
        raise StreamInputError(f"{label} must be a JSON object: {path}")
    return value


def _strict_positive_int(name: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise StreamInputError(f"{name} must be a positive integer")
    return int(value)


def _strict_digest(name: str, value: Any) -> str:
    if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
        raise StreamInputError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _read_scene_list(path: Path) -> tuple[str, ...]:
    rows: list[str] = []
    for line_number, raw in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        value = raw.strip()
        if not value or value.startswith("#"):
            continue
        if SCENE_RE.fullmatch(value) is None:
            raise StreamInputError(
                f"invalid scene ID at {path}:{line_number}: {value!r}"
            )
        rows.append(value)
    if not rows:
        raise StreamInputError("scene list is empty")
    if len(rows) != len(set(rows)):
        raise StreamInputError("scene list contains duplicate IDs")
    return tuple(rows)


def _producer_signature_function(
    source_path: Path,
) -> Callable[[Mapping[str, Any]], dict[str, str]]:
    """Compile the producer's hashing functions without importing its model code."""

    try:
        tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
    except (OSError, UnicodeDecodeError, SyntaxError) as error:
        raise StreamInputError(f"invalid proposal-cache producer source: {source_path}") from error
    definitions = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name in SIGNATURE_FUNCTIONS
    ]
    names = {node.name for node in definitions}
    required = {"_canonical_tensor", "_tensor_sha256", "_array_sha256", "_input_signature"}
    if not required.issubset(names):
        raise StreamInputError(
            "proposal-cache source lacks the reusable input-signature functions: "
            + ", ".join(sorted(required - names))
        )
    # ``exec`` is limited to the producer's small pure hashing kernel.  Fail
    # closed if a future/mutated source tries to import, decorate, perform I/O,
    # or call anything outside the operations used by the v2/v3 producers.
    for definition in definitions:
        if definition.decorator_list or definition.args.defaults or any(
            value is not None for value in definition.args.kw_defaults
        ):
            raise StreamInputError(
                f"unsafe producer signature definition: {definition.name}"
            )
    for node in ast.walk(ast.Module(body=definitions, type_ignores=[])):
        if isinstance(
            node,
            (
                ast.AsyncFunctionDef,
                ast.Await,
                ast.Global,
                ast.Import,
                ast.ImportFrom,
                ast.Lambda,
                ast.Nonlocal,
                ast.Yield,
                ast.YieldFrom,
            ),
        ):
            raise StreamInputError("unsafe construct in producer signature functions")
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Name):
            if node.func.id not in SAFE_SIGNATURE_DIRECT_CALLS:
                raise StreamInputError(
                    f"unsafe call in producer signature functions: {node.func.id}"
                )
        elif isinstance(node.func, ast.Attribute):
            if node.func.attr not in SAFE_SIGNATURE_METHOD_CALLS:
                raise StreamInputError(
                    f"unsafe method call in producer signature functions: {node.func.attr}"
                )
        else:
            raise StreamInputError("unsafe dynamic call in producer signature functions")
    module = ast.fix_missing_locations(ast.Module(body=definitions, type_ignores=[]))
    namespace: dict[str, Any] = {
        "Any": Any,
        "Callable": Callable,
        "Dict": dict,
        "Mapping": Mapping,
        "Tuple": tuple,
        "ProposalCacheError": StreamInputError,
        "_INPUT_FIELDS": INPUT_FIELDS,
        "hashlib": hashlib,
        "np": np,
        "torch": torch,
    }
    try:
        exec(compile(module, str(source_path), "exec"), namespace)
    except Exception as error:
        raise StreamInputError(
            f"could not compile producer input-signature functions: {source_path}"
        ) from error
    function = namespace.get("_input_signature")
    if not callable(function):
        raise StreamInputError("producer _input_signature is not callable")

    def wrapped(inputs: Mapping[str, Any]) -> dict[str, str]:
        try:
            result = function(inputs)
        except Exception as error:
            raise StreamInputError("producer _input_signature rejected reconstructed input") from error
        if not isinstance(result, Mapping) or set(result) != set(INPUT_FIELDS):
            raise StreamInputError("producer _input_signature returned an unexpected schema")
        return {
            field: _strict_digest(f"recomputed input_signature.{field}", result[field])
            for field in INPUT_FIELDS
        }

    return wrapped


def _numeric_files(directory: Path, suffix: str, label: str) -> tuple[Path, ...]:
    rows: list[tuple[int, Path]] = []
    for path in directory.iterdir():
        if not path.is_file() or path.suffix.lower() != suffix:
            continue
        try:
            identifier = int(path.stem)
        except ValueError as error:
            raise StreamInputError(f"non-numeric {label} filename: {path}") from error
        if identifier < 0:
            raise StreamInputError(f"negative {label} frame ID: {path}")
        rows.append((identifier, _safe_resolve(path, f"{label} frame {identifier}")))
    rows.sort(key=lambda row: row[0])
    if not rows:
        raise StreamInputError(f"no {label} files in {directory}")
    identifiers = [row[0] for row in rows]
    if len(identifiers) != len(set(identifiers)):
        raise StreamInputError(f"duplicate {label} frame IDs in {directory}")
    return tuple(row[1] for row in rows)


def _color_files(directory: Path) -> tuple[Path, ...]:
    # This intentionally matches ScannetDataset's producer contract: JPG only.
    return _numeric_files(directory, ".jpg", "color")


def _load_pose(path: Path) -> np.ndarray:
    try:
        rows = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle.readlines():
                rows.append(list(map(float, line.split(" "))))
        pose = np.asarray(rows, dtype=np.float64).reshape(4, 4)
    except (OSError, ValueError) as error:
        raise StreamInputError(f"invalid ScanNet pose: {path}") from error
    if np.isnan(pose).any():
        # The pinned producer substitutes infinities but silently retains NaNs;
        # accepting that ambiguous state would undermine a validity ledger.
        raise StreamInputError(f"NaN ScanNet pose is unsupported: {path}")
    return pose


def _orientation_code(pose_f32: np.ndarray) -> int:
    pose = torch.from_numpy(np.ascontiguousarray(pose_f32, dtype=np.float32))
    z_vec = pose[..., 2, :3]
    candidates = torch.tensor(
        [
            [0.0, -1.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
        ],
        dtype=pose.dtype,
    )
    return int((candidates @ z_vec).argmax(dim=-1).item())


def _rotate_image_intrinsics(K: torch.Tensor, orientation: int, size: tuple[int, int]) -> torch.Tensor:
    if orientation == 0:
        return K.clone()
    if orientation in (1, 3):
        return torch.stack(
            [
                torch.stack([K[1, 1], K[0, 1], K[1, 2]]),
                torch.stack([K[1, 0], K[0, 0], K[0, 2]]),
                torch.stack([K[2, 0], K[2, 1], K[2, 2]]),
            ]
        ).to(K)
    if orientation == 2:
        return torch.stack(
            [
                torch.stack([K[0, 0], K[0, 1], K.new_tensor(float(size[0])) - K[0, 2]]),
                torch.stack([K[1, 0], K[1, 1], K.new_tensor(float(size[1])) - K[1, 2]]),
                torch.stack([K[2, 0], K[2, 1], K[2, 2]]),
            ]
        ).to(K)
    raise StreamInputError(f"invalid producer orientation code: {orientation}")


def _processed_inputs(
    *,
    color_path: Path,
    depth_path: Path,
    effective_pose: np.ndarray,
    K_f32: np.ndarray,
    width: int,
    height: int,
    depth_scale: float,
) -> Mapping[str, Any]:
    color_bgr = cv2.imread(os.fspath(color_path), cv2.IMREAD_COLOR)
    depth_raw = cv2.imread(os.fspath(depth_path), cv2.IMREAD_UNCHANGED)
    if color_bgr is None or color_bgr.ndim != 3 or color_bgr.shape[2] != 3:
        raise StreamInputError(f"could not decode color image: {color_path}")
    if depth_raw is None or depth_raw.ndim != 2:
        raise StreamInputError(f"could not decode depth image: {depth_path}")
    if depth_raw.shape != (height, width):
        raise StreamInputError(
            f"raw depth shape differs from producer H/W at {depth_path}: "
            f"{depth_raw.shape} != {(height, width)}"
        )
    color_rgb = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)
    depth_f32 = depth_raw.astype(np.float32) / depth_scale
    color_rgb = cv2.resize(color_rgb, (depth_raw.shape[1], depth_raw.shape[0]))
    image_hwc = np.asarray(color_rgb).reshape((height, width, 3))
    depth_f32 = cv2.resize(depth_f32, (width, height))
    image_nchw = torch.tensor(np.moveaxis(image_hwc, -1, 0))[None]
    depth_nhw = torch.tensor(
        depth_f32.view(dtype=np.float32).reshape((height, width))
    )[None].float()
    pose_f32 = np.asarray(effective_pose, dtype=np.float32).reshape(4, 4)
    orientation = _orientation_code(pose_f32)
    rotation_k = {0: 0, 1: -1, 2: 2, 3: 1}[orientation]
    image_nchw = torch.rot90(image_nchw, rotation_k, dims=(-2, -1))
    depth_nhw = torch.rot90(depth_nhw, rotation_k, dims=(-2, -1))
    K_tensor = torch.tensor(np.asarray(K_f32, dtype=np.float32))
    image_K = _rotate_image_intrinsics(K_tensor, orientation, (width, height))
    image = np.moveaxis(image_nchw[-1].numpy(), 0, -1)
    return {
        "image": image,
        "depth": depth_nhw[-1],
        "image_K": image_K,
        "depth_K": K_tensor,
        "camera_to_world": torch.from_numpy(pose_f32),
    }


def _validate_schedule(path: Path, scene: str) -> dict[str, Any]:
    manifest = _load_json(path, "proposal-cache schedule manifest")
    schema = manifest.get("schema")
    if schema not in ALLOWED_SCHEDULE_SCHEMAS:
        raise StreamInputError(f"unsupported schedule schema for {scene}: {schema!r}")
    if manifest.get("scene_id") != scene:
        raise StreamInputError(f"schedule scene ID mismatch for {scene}")
    records = manifest.get("records")
    recorded = manifest.get("recorded_frame_ids")
    if not isinstance(records, list) or not records:
        raise StreamInputError(f"empty schedule records for {scene}")
    if not isinstance(recorded, list) or len(recorded) != len(records):
        raise StreamInputError(f"invalid recorded frame ledger for {scene}")
    frame_ids: list[int] = []
    for index, row in enumerate(records):
        if not isinstance(row, dict) or type(row.get("frame_id")) is not int:
            raise StreamInputError(f"invalid schedule record {index} for {scene}")
        frame_id = int(row["frame_id"])
        if frame_id < 0:
            raise StreamInputError(f"negative schedule frame ID for {scene}")
        signature = row.get("input_signature")
        if not isinstance(signature, Mapping) or set(signature) != set(INPUT_FIELDS):
            raise StreamInputError(f"invalid input signature at {scene}/{frame_id}")
        for field in INPUT_FIELDS:
            _strict_digest(f"manifest {scene}/{frame_id}.{field}", signature[field])
        frame_ids.append(frame_id)
    if frame_ids != recorded or frame_ids != sorted(set(frame_ids)):
        raise StreamInputError(f"unordered/duplicate schedule records for {scene}")
    if manifest.get("record_count") != len(records):
        raise StreamInputError(f"schedule record_count mismatch for {scene}")
    schedule = manifest.get("schedule")
    if not isinstance(schedule, Mapping):
        raise StreamInputError(f"missing schedule contract for {scene}")
    dataset_length = _strict_positive_int(
        f"{scene}.schedule.dataset_length", schedule.get("dataset_length")
    )
    gap = _strict_positive_int(f"{scene}.schedule.gap", schedule.get("gap"))
    if schedule.get("terminal_policy") != "upstream_boxfusion_early_exit_v1":
        raise StreamInputError(f"unexpected terminal policy for {scene}")
    final_native_frame = max(0, dataset_length - gap - 1)
    expected = list(range(0, final_native_frame + 1, gap))
    if frame_ids != expected:
        raise StreamInputError(f"schedule frame sequence differs from producer policy for {scene}")
    return manifest


def _scene_record(
    *,
    scene: str,
    scene_index: int,
    scene_root: Path,
    schedule_root: Path,
    signature_function: Callable[[Mapping[str, Any]], dict[str, str]],
    width: int,
    height: int,
    depth_scale: float,
    start: int,
) -> dict[str, Any]:
    frames_root = _safe_resolve(scene_root / scene / "frames", f"{scene} frames", directory=True)
    color_dir = _safe_resolve(frames_root / "color", f"{scene} color", directory=True)
    depth_dir = _safe_resolve(frames_root / "depth", f"{scene} depth", directory=True)
    pose_dir = _safe_resolve(frames_root / "pose", f"{scene} pose", directory=True)
    intrinsic_dir = _safe_resolve(
        frames_root / "intrinsic", f"{scene} intrinsic", directory=True
    )
    colors = _color_files(color_dir)
    depths = _numeric_files(depth_dir, ".png", "depth")
    poses = _numeric_files(pose_dir, ".txt", "pose")
    color_ids = [int(path.stem) for path in colors]
    depth_ids = [int(path.stem) for path in depths]
    pose_ids = [int(path.stem) for path in poses]
    if color_ids != depth_ids or color_ids != pose_ids:
        raise StreamInputError(f"RGB/depth/pose frame IDs disagree for {scene}")
    if start < 0 or start >= len(colors):
        raise StreamInputError(f"start index is outside {scene}: {start}")
    # ScannetDataset loads/substitutes every pose first and only then applies
    # ``start``.  In particular, an invalid first retained pose may inherit a
    # valid pose before ``start``.  Preserve that exact order here.
    all_source_ids = color_ids

    intrinsic_paths = {
        name: _safe_resolve(intrinsic_dir / name, f"{scene} {name}")
        for name in INTRINSIC_FILES
    }
    try:
        K = np.loadtxt(intrinsic_paths["intrinsic_depth.txt"], dtype=np.float64)
    except (OSError, ValueError) as error:
        raise StreamInputError(f"invalid depth intrinsics for {scene}") from error
    if K.shape != (4, 4) or not np.isfinite(K).all():
        raise StreamInputError(f"depth intrinsics must be finite 4x4 for {scene}")
    K_f32 = np.ascontiguousarray(K[:3, :3].astype(np.float32))

    all_effective_poses: list[np.ndarray] = []
    all_pose_status: list[str] = []
    all_substituted_from: list[int | None] = []
    last_valid: np.ndarray | None = None
    last_valid_source: int | None = None
    all_invalid_source_ids: list[int] = []
    all_valid_source_ids: list[int] = []
    for source_id, pose_path in zip(all_source_ids, poses):
        pose = _load_pose(pose_path)
        if np.isinf(pose).any():
            all_invalid_source_ids.append(source_id)
            if last_valid is None or last_valid_source is None:
                raise StreamInputError(
                    f"first source pose is invalid and cannot be substituted: {pose_path}"
                )
            all_effective_poses.append(np.array(last_valid, copy=True))
            all_pose_status.append("substituted_previous_valid")
            all_substituted_from.append(last_valid_source)
        else:
            all_valid_source_ids.append(source_id)
            last_valid = pose
            last_valid_source = source_id
            all_effective_poses.append(np.array(pose, copy=True))
            all_pose_status.append("valid")
            all_substituted_from.append(None)

    colors, depths, poses = colors[start:], depths[start:], poses[start:]
    source_ids = all_source_ids[start:]
    effective_poses = all_effective_poses[start:]
    pose_status = all_pose_status[start:]
    substituted_from = all_substituted_from[start:]

    schedule_path = _safe_resolve(
        schedule_root / scene / "manifest.json", f"{scene} schedule manifest"
    )
    schedule = _validate_schedule(schedule_path, scene)
    if schedule["schedule"]["dataset_length"] != len(colors):
        raise StreamInputError(
            f"current dataset length differs from schedule for {scene}: "
            f"{len(colors)} != {schedule['schedule']['dataset_length']}"
        )
    records: list[dict[str, Any]] = []
    scheduled_valid: list[int] = []
    scheduled_invalid: list[int] = []
    for manifest_row in schedule["records"]:
        frame_id = int(manifest_row["frame_id"])
        if frame_id >= len(colors):
            raise StreamInputError(f"scheduled frame is outside current stream: {scene}/{frame_id}")
        current_inputs = _processed_inputs(
            color_path=colors[frame_id],
            depth_path=depths[frame_id],
            effective_pose=effective_poses[frame_id],
            K_f32=K_f32,
            width=width,
            height=height,
            depth_scale=depth_scale,
        )
        recomputed = signature_function(current_inputs)
        expected = {field: manifest_row["input_signature"][field] for field in INPUT_FIELDS}
        if recomputed != expected:
            changed = [field for field in INPUT_FIELDS if recomputed[field] != expected[field]]
            raise StreamInputError(
                f"producer input signature mismatch at {scene}/{frame_id}: "
                + ", ".join(changed)
            )
        source_id = source_ids[frame_id]
        if pose_status[frame_id] == "valid":
            scheduled_valid.append(source_id)
        else:
            scheduled_invalid.append(source_id)
        records.append(
            {
                "frame_id": frame_id,
                "source_frame_id": source_id,
                "pose_status": pose_status[frame_id],
                "pose_substituted_from_source_frame_id": substituted_from[frame_id],
                "input_signature": recomputed,
                "raw_file_sha256": {
                    "color": _sha256(colors[frame_id]),
                    "depth": _sha256(depths[frame_id]),
                    "pose": _sha256(poses[frame_id]),
                },
            }
        )
    return {
        "scene_id": scene,
        "scene_index": scene_index,
        "frames_root": os.fspath(frames_root),
        "dataset_length": len(colors),
        "source_frame_id_first": source_ids[0],
        "source_frame_id_last": source_ids[-1],
        "schedule": {
            "path": os.fspath(schedule_path),
            "sha256": _sha256(schedule_path),
            "schema": schedule["schema"],
            "namespace": schedule.get("namespace"),
            "record_count": schedule["record_count"],
            "recorded_frame_ids": schedule["recorded_frame_ids"],
            "producer_fingerprint": schedule.get("producer_fingerprint"),
        },
        "pose_ledger": {
            "all_valid_source_frame_ids": all_valid_source_ids,
            "all_invalid_source_frame_ids": all_invalid_source_ids,
            "retained_valid_source_frame_ids": [
                source_id
                for source_id, status in zip(source_ids, pose_status)
                if status == "valid"
            ],
            "retained_invalid_source_frame_ids": [
                source_id
                for source_id, status in zip(source_ids, pose_status)
                if status != "valid"
            ],
            "scheduled_valid_source_frame_ids": scheduled_valid,
            "scheduled_invalid_source_frame_ids": scheduled_invalid,
            "producer_invalid_pose_policy": "replace_inf_pose_with_previous_valid_pose",
        },
        "intrinsic_file_sha256": {
            name: _sha256(path) for name, path in sorted(intrinsic_paths.items())
        },
        "records": records,
    }


def build_seal(
    *,
    schedule_root: Path,
    scene_root: Path,
    scene_list: Path,
    producer_source: Path,
    capture_source: Path,
    width: int = 640,
    height: int = 480,
    depth_scale: float = 1000.0,
    start: int = 0,
) -> dict[str, Any]:
    schedule_root = _safe_resolve(schedule_root, "schedule root", directory=True)
    scene_root = _safe_resolve(scene_root, "ScanNet scene root", directory=True)
    scene_list = _safe_resolve(scene_list, "scene list")
    producer_source = _safe_resolve(producer_source, "proposal-cache producer source")
    capture_source = _safe_resolve(capture_source, "ScanNet capture source")
    tool_path = _safe_resolve(Path(__file__), "stream-input seal tool")
    if isinstance(width, bool) or not isinstance(width, int) or width < 1:
        raise StreamInputError("width must be a positive integer")
    if isinstance(height, bool) or not isinstance(height, int) or height < 1:
        raise StreamInputError("height must be a positive integer")
    if isinstance(start, bool) or not isinstance(start, int) or start < 0:
        raise StreamInputError("start must be a non-negative integer")
    if not isinstance(depth_scale, (int, float)) or isinstance(depth_scale, bool):
        raise StreamInputError("depth_scale must be a finite positive number")
    depth_scale = float(depth_scale)
    if not math.isfinite(depth_scale) or depth_scale <= 0.0:
        raise StreamInputError("depth_scale must be a finite positive number")
    scenes = _read_scene_list(scene_list)
    signature_function = _producer_signature_function(producer_source)
    scene_rows = [
        _scene_record(
            scene=scene,
            scene_index=index,
            scene_root=scene_root,
            schedule_root=schedule_root,
            signature_function=signature_function,
            width=width,
            height=height,
            depth_scale=depth_scale,
            start=start,
        )
        for index, scene in enumerate(scenes)
    ]
    schedule_hash_rows = [
        {
            "scene_id": row["scene_id"],
            "manifest_sha256": row["schedule"]["sha256"],
        }
        for row in scene_rows
    ]
    schedule_ledger_sha256 = hashlib.sha256(
        json.dumps(
            schedule_hash_rows,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    ).hexdigest()
    return {
        "schema": SCHEMA,
        "mode": "no_gt_stream_input_seal",
        "gt_access": False,
        "oracle_access": False,
        "training": False,
        "output_mutation": False,
        "signature_mode": "exact_producer_array_signature",
        "signature_fields": list(INPUT_FIELDS),
        "supplemental_raw_file_ledger": True,
        "raw_file_ledger_equivalent_to_producer_signature": False,
        "scene_count": len(scenes),
        "record_count": sum(row["schedule"]["record_count"] for row in scene_rows),
        "schedule_manifest_ledger_sha256": schedule_ledger_sha256,
        "paths": {
            "schedule_root": os.fspath(schedule_root),
            "scene_root": os.fspath(scene_root),
            "scene_list": os.fspath(scene_list),
            "scene_list_sha256": _sha256(scene_list),
        },
        "producer_signature": {
            "function": "_input_signature",
            "source": os.fspath(producer_source),
            "source_sha256": _sha256(producer_source),
            "reuse_method": "AST-extracted producer function bodies",
        },
        "capture_contract": {
            "source": os.fspath(capture_source),
            "source_sha256": _sha256(capture_source),
            "dataset": "ScannetDataset",
            "start": start,
            "width": width,
            "height": height,
            "depth_scale": depth_scale,
            "color_decode": "cv2.imread_color+bgr_to_rgb+linear_resize_to_depth_hw",
            "depth_decode": "cv2.imread_unchanged+float32_scale+linear_resize",
            "orientation": "producer_torch_rot90_to_upright",
        },
        "runtime": {
            "python": platform.python_version(),
            "numpy": str(np.__version__),
            "torch": str(torch.__version__),
            "opencv": str(cv2.__version__),
        },
        "tool": {
            "path": os.fspath(tool_path),
            "sha256": _sha256(tool_path),
        },
        "scenes": scene_rows,
    }


def write_seal_exclusive(path: Path, value: Mapping[str, Any]) -> None:
    path = path.expanduser().resolve()
    lowered = os.fspath(path).lower().replace("\\", "/")
    if any(marker in lowered for marker in FORBIDDEN_PATH_MARKERS):
        raise StreamInputError(f"forbidden output path: {path}")
    if path.exists() or path.is_symlink():
        raise StreamInputError(f"refusing to overwrite stream-input seal: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = _canonical_json_bytes(value)
    try:
        with path.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError as error:
        raise StreamInputError(f"refusing to overwrite stream-input seal: {path}") from error


def verify_seal(
    *,
    seal_path: Path,
    schedule_root: Path,
    scene_root: Path,
    scene_list: Path,
    producer_source: Path,
    capture_source: Path,
    width: int = 640,
    height: int = 480,
    depth_scale: float = 1000.0,
    start: int = 0,
) -> dict[str, Any]:
    seal_path = _safe_resolve(seal_path, "stream-input seal")
    expected = _load_json(seal_path, "stream-input seal")
    if expected.get("schema") != SCHEMA or expected.get("gt_access") is not False:
        raise StreamInputError("unexpected or unsafe stream-input seal schema")
    current = build_seal(
        schedule_root=schedule_root,
        scene_root=scene_root,
        scene_list=scene_list,
        producer_source=producer_source,
        capture_source=capture_source,
        width=width,
        height=height,
        depth_scale=depth_scale,
        start=start,
    )
    if current != expected:
        raise StreamInputError(
            "current ScanNet stream/schedule/runtime differs from the sealed ledger"
        )
    return {
        "schema": SCHEMA,
        "verified": True,
        "gt_access": False,
        "scene_count": current["scene_count"],
        "record_count": current["record_count"],
        "seal_sha256": _sha256(seal_path),
    }


def _common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--schedule-root", required=True, type=Path)
    parser.add_argument("--scene-root", required=True, type=Path)
    parser.add_argument("--scene-list", required=True, type=Path)
    parser.add_argument("--producer-source", required=True, type=Path)
    parser.add_argument("--capture-source", required=True, type=Path)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--depth-scale", type=float, default=1000.0)
    parser.add_argument("--start", type=int, default=0)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Seal or verify exact no-GT ScanNet proposal-cache inputs"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    seal = commands.add_parser("seal", help="create a new immutable input seal")
    _common_arguments(seal)
    seal.add_argument("--out-json", required=True, type=Path)
    verify = commands.add_parser("verify", help="verify current inputs against a seal")
    _common_arguments(verify)
    verify.add_argument("--seal-json", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _build_parser().parse_args(argv)
    common = {
        "schedule_root": arguments.schedule_root,
        "scene_root": arguments.scene_root,
        "scene_list": arguments.scene_list,
        "producer_source": arguments.producer_source,
        "capture_source": arguments.capture_source,
        "width": arguments.width,
        "height": arguments.height,
        "depth_scale": arguments.depth_scale,
        "start": arguments.start,
    }
    if arguments.command == "seal":
        value = build_seal(**common)
        write_seal_exclusive(arguments.out_json, value)
        summary = {
            "schema": SCHEMA,
            "sealed": True,
            "gt_access": False,
            "scene_count": value["scene_count"],
            "record_count": value["record_count"],
            "out_json": os.fspath(arguments.out_json.resolve()),
        }
    else:
        summary = verify_seal(seal_path=arguments.seal_json, **common)
    print(json.dumps(summary, sort_keys=True, ensure_ascii=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except StreamInputError as error:
        print(f"stream-input seal error: {error}", file=sys.stderr)
        raise SystemExit(1)
