#!/usr/bin/env python3
"""Export the frozen dev3 raw-Boxer K8 past-only receipt shadow.

This runner has no annotation, evaluator, RGB, depth, CLIP, semantic, native
prediction deserialization, or output-mutation surface.  It reads only the
numeric per-view geometry in the sealed Boxer sidecar, replays every frame in
the three exact frozen gap-25 schedules (including empty valid frames), and
serializes the complete K8 assignment and immutable receipt trace.

The final directory is published with Linux ``renameat2(RENAME_NOREPLACE)``.
If the kernel cannot provide true atomic no-replace semantics, the run fails
closed instead of falling back to an overwrite-capable rename.
"""

from __future__ import annotations

import argparse
import ctypes
import errno
import hashlib
import io
import json
import math
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time
from typing import Any, Mapping, Sequence
import zipfile

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if os.fspath(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, os.fspath(REPOSITORY_ROOT))

from boxfusion.s3r_receipt_tracker import (  # noqa: E402
    S3RObservation,
    S3RReceipt,
    S3RReceiptTracker,
)


SCHEMA = "boxfusion.s3r_raw_boxer_past_only_shadow.v1"
DEV3_SCENES = ("scene0568_00", "scene0606_01", "scene0377_02")
TOP_K = 8

SEALED_ROOT = (
    REPOSITORY_ROOT
    / "logs"
    / "scannet_boxer_unexplained_shadow_clean_in2_v5_score05"
)
SEALED_JSON = SEALED_ROOT / "sealed" / "boxer_shadow_candidates.json"
SEALED_NPZ = SEALED_ROOT / "sealed" / "boxer_shadow_candidates.npz"
TOPK_RECEIPT = (
    REPOSITORY_ROOT
    / "logs"
    / "scannet_boxer_per_view_topk_raw_ceiling_score05_dev3_v5.json"
)
TOPK_TOOL = REPOSITORY_ROOT / "tools" / "audit_scannet_boxer_per_view_topk_ceiling.py"
PREREGISTRATION = REPOSITORY_ROOT / "docs" / "S3R_RAW_BOXER_PAST_ONLY_PREREGISTRATION.md"
TRACKER_SOURCE = REPOSITORY_ROOT / "boxfusion" / "s3r_receipt_tracker.py"
TRACKER_TEST = REPOSITORY_ROOT / "tests" / "test_s3r_receipt_tracker.py"
RUNNER_TEST = (
    REPOSITORY_ROOT
    / "tests"
    / "test_run_scannet_s3r_raw_boxer_k8_receipt_shadow.py"
)
SCHEDULE_ROOT = (
    REPOSITORY_ROOT
    / "cache"
    / "cutr_postfilter_v3"
    / "scannet-graw-e2-score05-preflight3-v3-r1"
)
FORMAL_T05_ROOT = REPOSITORY_ROOT / "results" / "scannet_topk_fusion_score05"

OUTPUT_JSON_NAME = "s3r_raw_boxer_k8_receipt_shadow.json"
OUTPUT_NPZ_NAME = "s3r_raw_boxer_k8_receipt_shadow.npz"

EXPECTED_INPUT_SCHEMA = "boxfusion.owl_boxer_shadow_candidates.v1"
EXPECTED_SOURCE_CANDIDATE_COUNT = 3085
EXPECTED_SOURCE_CONTENT_SHA256 = (
    "8b2362cc11517a58f2a05b371698cf3a45db6805b27c4c1dd10a3c9b899ab529"
)
EXPECTED_SELECTION_SHA256 = (
    "34ee638d51b3bc137253b3e361a60d84e110e114d2b46c487651550e708aa638"
)
EXPECTED_SELECTION_COUNTS: Mapping[str, int] = {
    "scene0568_00": 501,
    "scene0606_01": 854,
    "scene0377_02": 216,
}
EXPECTED_CANDIDATE_FRAME_COUNTS: Mapping[str, int] = {
    "scene0568_00": 66,
    "scene0606_01": 110,
    "scene0377_02": 30,
}
EXPECTED_FORMAL_T05_SHA256: Mapping[str, str] = {
    "scene0568_00": "b55ce48fb6eb4dad9ee5bfe7007c3dbc9898b3f72ddbc5ad428b8be6414bcd2d",
    "scene0606_01": "d4e8d6dc85c917ac1634b81a45adb3866279d3e02f470c43b23bd71f5bb3ef1c",
    "scene0377_02": "ed7f849a33d45eebe846559a90aeb7de1a97f2eb169c3a7c0cb5de61d3dab35b",
}
EXPECTED_SCHEDULES: Mapping[str, Mapping[str, object]] = {
    "scene0568_00": {
        "sha256": "1ee049e9ad8263e8d7c19838a1038445129a1ae7265434f042ea0c438f3ab19a",
        "recorded_count": 66,
        "valid_count": 66,
        "invalid_pose_frame_ids": (),
        "empty_valid_frame_ids": (),
    },
    "scene0606_01": {
        "sha256": "aedfe2f230c252fb9aaad10b678e3264b8855cfe1150f8b36b291d48e5032753",
        "recorded_count": 113,
        "valid_count": 112,
        "invalid_pose_frame_ids": (1325,),
        "empty_valid_frame_ids": (1300, 1350),
    },
    "scene0377_02": {
        "sha256": "9a8c127b09c36140494a8288425d6b23087b5865d3789b295ed55744d6edf80e",
        "recorded_count": 30,
        "valid_count": 30,
        "invalid_pose_frame_ids": (),
        "empty_valid_frame_ids": (),
    },
}

EXPECTED_FIXED_HASHES: Mapping[str, tuple[Path, str]] = {
    "preregistration": (
        PREREGISTRATION,
        "14f29a50dd65ee791be2df519e0000cf22bfc94a0209880f3539159acf4f7df3",
    ),
    "tracker_source": (
        TRACKER_SOURCE,
        "277316c36b7a7fcb8005a24e907e0f232e41f6b5874411293eb26b0744df9628",
    ),
    "tracker_test": (
        TRACKER_TEST,
        "f08fd59ee2888c936e5b783de668fd789ba6b676bc4864e001b000ea287b1e3c",
    ),
    "sealed_json": (
        SEALED_JSON,
        "84eb4f2c62d1573d9e9f1ec4c3df5a6cac16ad10c8cece0989d37dd97b734e9e",
    ),
    "sealed_npz": (
        SEALED_NPZ,
        "c1a921d70de447bf528711a71deb34cf93a9bf671d3514baafa42b7b1b8b4a6c",
    ),
    "topk_receipt": (
        TOPK_RECEIPT,
        "d4ba67b37d362842333ac525abe32f6807c4fba90af83b699bbfc1494aa5ea1f",
    ),
    "topk_tool": (
        TOPK_TOOL,
        "9a756f474e40e7b991453b09cb006b1147432aab124a55d33e4613d2adad1b44",
    ),
}

EXPECTED_NPZ_INPUT_ARRAYS = {
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
ALLOWED_SOURCE_ARRAYS = (
    "scene_ids",
    "per_view_scene_index",
    "per_view_frame_id",
    "per_view_source_row",
    "per_view_source_instance_id",
    "per_view_source_score",
    "per_view_center_world",
    "per_view_extent_xyz",
    "per_view_quaternion_wxyz",
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

MAX_VALID_FRAMES_PER_SCENE = 4096
MAX_SELECTED_ROWS_PER_SCENE = 32768
MAX_ELIGIBILITY_CHECKS_PER_FRAME = 8192
MAX_TRACE_UNCOMPRESSED_BYTES = 32 * 1024 * 1024
MAX_TRACKER_INCREMENTAL_MEMORY_BYTES = 64 * 1024 * 1024
TRACKER_CPU_P95_LIMIT_NS = 2_000_000
TRACKER_CPU_MAX_LIMIT_NS = 10_000_000

REQUIRED_NUMERIC_THREAD_ENVIRONMENT: Mapping[str, str] = {
    "OPENBLAS_NUM_THREADS": "1",
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
}

_AT_FDCWD = -100
_RENAME_NOREPLACE = 1


class S3RShadowError(ValueError):
    """Raised when a frozen contract or output-inert invariant fails."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _regular_file(path: Path, label: str) -> Path:
    raw = Path(path)
    if raw.is_symlink():
        raise S3RShadowError(f"{label} must not be a symlink: {raw}")
    try:
        resolved = raw.resolve(strict=True)
    except (OSError, RuntimeError) as error:
        raise S3RShadowError(f"missing {label}: {raw}") from error
    if not resolved.is_file():
        raise S3RShadowError(f"{label} must be a regular file: {raw}")
    return resolved


def _read_json(path: Path, label: str) -> dict[str, Any]:
    path = _regular_file(path, label)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise S3RShadowError(f"invalid {label}: {path}") from error
    if not isinstance(value, dict):
        raise S3RShadowError(f"{label} must contain a JSON object: {path}")
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


def _deterministic_npz_bytes(arrays: Mapping[str, np.ndarray]) -> bytes:
    output = io.BytesIO()
    with zipfile.ZipFile(
        output, mode="w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
    ) as archive:
        for name in sorted(arrays):
            payload = io.BytesIO()
            np.lib.format.write_array(
                payload, np.ascontiguousarray(arrays[name]), allow_pickle=False
            )
            info = zipfile.ZipInfo(f"{name}.npy", (1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o600 << 16
            archive.writestr(info, payload.getvalue(), compresslevel=9)
    return output.getvalue()


def _json_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False)
        + "\n"
    ).encode("ascii")


def _write_exclusive_fsync(path: Path, payload: bytes) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        os.close(descriptor)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _rename_noreplace(source: Path, destination: Path) -> None:
    """Atomically publish a directory, failing if destination already exists."""

    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        raise S3RShadowError("renameat2 is unavailable; refusing non-atomic fallback")
    renameat2.argtypes = (
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    )
    renameat2.restype = ctypes.c_int
    result = renameat2(
        _AT_FDCWD,
        os.fsencode(source),
        _AT_FDCWD,
        os.fsencode(destination),
        _RENAME_NOREPLACE,
    )
    if result == 0:
        return
    error_number = ctypes.get_errno()
    if error_number in (errno.EEXIST, errno.ENOTEMPTY):
        raise S3RShadowError(f"refusing to overwrite output root: {destination}")
    if error_number in (errno.ENOSYS, errno.EINVAL, errno.EOPNOTSUPP):
        raise S3RShadowError(
            "atomic RENAME_NOREPLACE is unsupported; refusing unsafe fallback"
        )
    raise OSError(error_number, os.strerror(error_number), os.fspath(destination))


def _validate_fixed_assets() -> dict[str, dict[str, object]]:
    output: dict[str, dict[str, object]] = {}
    for name, (raw_path, expected_hash) in EXPECTED_FIXED_HASHES.items():
        path = _regular_file(raw_path, f"frozen {name}")
        actual_hash = _sha256(path)
        if actual_hash != expected_hash:
            raise S3RShadowError(
                f"frozen {name} SHA-256 mismatch: "
                f"expected={expected_hash}, actual={actual_hash}"
            )
        output[name] = {
            "path": os.fspath(path),
            "sha256": actual_hash,
            "bytes": path.stat().st_size,
        }
    return output


def _load_sealed_candidates() -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    manifest = _read_json(SEALED_JSON, "sealed Boxer JSON")
    required = {
        "schema": EXPECTED_INPUT_SCHEMA,
        "profile": "clean_in2",
        "mode": "shadow",
        "output_inert": True,
        "birth": False,
        "gt_access": False,
        "gt_access_guard_verified": True,
        "semantic_source_exported": False,
        "native_clip_unchanged": True,
        "coordinate_frame": "scannet_world",
        "per_view_candidate_count": EXPECTED_SOURCE_CANDIDATE_COUNT,
        "scene_count": len(DEV3_SCENES),
    }
    for key, expected in required.items():
        if manifest.get(key) != expected:
            raise S3RShadowError(
                f"sealed Boxer contract mismatch for {key}: {manifest.get(key)!r}"
            )
    if (
        manifest.get("npz_file") != SEALED_NPZ.name
        or manifest.get("npz_sha256") != EXPECTED_FIXED_HASHES["sealed_npz"][1]
        or manifest.get("candidate_content_sha256")
        != EXPECTED_SOURCE_CONTENT_SHA256
    ):
        raise S3RShadowError("sealed Boxer JSON no longer binds the frozen NPZ")
    assets = manifest.get("assets_and_protocol")
    expected_assets = {
        "profile": "clean_in2",
        "detector": "owl",
        "detector_hw": 960,
        "threshold_2d": 0.25,
        "threshold_3d": 0.5,
        "nms_iou_2d": 0.5,
        "start_n": 1,
        "skip_n": 25,
    }
    if not isinstance(assets, Mapping):
        raise S3RShadowError("sealed Boxer asset ledger is absent")
    for key, expected in expected_assets.items():
        if assets.get(key) != expected:
            raise S3RShadowError(f"sealed Boxer asset mismatch for {key}")

    # Deliberately load only the permitted per-view numeric fields.  The
    # tracked arrays are schema-checked by filename and byte-sealed by the NPZ
    # hash, but are never decoded by S3R0.
    try:
        with np.load(SEALED_NPZ, allow_pickle=False) as source:
            if set(source.files) != EXPECTED_NPZ_INPUT_ARRAYS:
                raise S3RShadowError("unexpected sealed Boxer NPZ array schema")
            arrays = {
                name: np.array(source[name], copy=True) for name in ALLOWED_SOURCE_ARRAYS
            }
    except (OSError, ValueError) as error:
        if isinstance(error, S3RShadowError):
            raise
        raise S3RShadowError("could not load sealed Boxer numeric arrays") from error

    if tuple(str(value) for value in arrays["scene_ids"].tolist()) != DEV3_SCENES:
        raise S3RShadowError("sealed Boxer scene order changed")
    count = EXPECTED_SOURCE_CANDIDATE_COUNT
    expected_schema = {
        "scene_ids": ((len(DEV3_SCENES),), "U"),
        "per_view_scene_index": ((count,), "i"),
        "per_view_frame_id": ((count,), "i"),
        "per_view_source_row": ((count,), "i"),
        "per_view_source_instance_id": ((count,), "i"),
        "per_view_source_score": ((count,), "f"),
        "per_view_center_world": ((count, 3), "f"),
        "per_view_extent_xyz": ((count, 3), "f"),
        "per_view_quaternion_wxyz": ((count, 4), "f"),
    }
    for name, (shape, kind) in expected_schema.items():
        if arrays[name].shape != shape or arrays[name].dtype.kind != kind:
            raise S3RShadowError(f"sealed numeric schema mismatch for {name}")
    numeric = np.concatenate(
        [
            arrays["per_view_source_score"].reshape(-1),
            arrays["per_view_center_world"].reshape(-1),
            arrays["per_view_extent_xyz"].reshape(-1),
            arrays["per_view_quaternion_wxyz"].reshape(-1),
        ]
    )
    if not np.isfinite(numeric).all():
        raise S3RShadowError("sealed Boxer numeric source contains non-finite values")
    if (
        np.any(arrays["per_view_extent_xyz"] <= 0.0)
        or np.any(arrays["per_view_source_score"] < 0.0)
        or np.any(arrays["per_view_source_score"] > 1.0)
    ):
        raise S3RShadowError("sealed Boxer extent or score is out of range")
    quaternion_norms = np.sum(
        np.asarray(arrays["per_view_quaternion_wxyz"], dtype=np.float64) ** 2,
        axis=1,
    )
    if np.any(quaternion_norms <= 1e-12):
        raise S3RShadowError("sealed Boxer quaternion is degenerate")
    scene_indices = arrays["per_view_scene_index"]
    if np.any((scene_indices < 0) | (scene_indices >= len(DEV3_SCENES))):
        raise S3RShadowError("sealed Boxer scene index is out of range")

    scene_ledgers = manifest.get("scenes")
    if not isinstance(scene_ledgers, list) or len(scene_ledgers) != len(DEV3_SCENES):
        raise S3RShadowError("sealed Boxer scene ledger is invalid")
    for scene_index, (scene, ledger) in enumerate(zip(DEV3_SCENES, scene_ledgers)):
        positions = np.flatnonzero(scene_indices == scene_index)
        expected_schedule = EXPECTED_SCHEDULES[scene]
        if (
            not isinstance(ledger, Mapping)
            or ledger.get("scene_id") != scene
            or ledger.get("scene_index") != scene_index
            or ledger.get("gt_access_guard_verified") is not True
            or ledger.get("per_view_kept_rows") != len(positions)
            or ledger.get("per_view_extra_schedule_rows_excluded") != 0
            or ledger.get("sealed_schedule_manifest_sha256")
            != expected_schedule["sha256"]
            or ledger.get("sealed_schedule_frame_count")
            != expected_schedule["valid_count"]
            or tuple(ledger.get("sealed_schedule_invalid_pose_frame_ids_excluded", ()))
            != expected_schedule["invalid_pose_frame_ids"]
        ):
            raise S3RShadowError(f"sealed Boxer scene ledger mismatch for {scene}")
        if len(np.unique(arrays["per_view_source_row"][positions])) != len(positions):
            raise S3RShadowError(f"duplicate source_row identity in {scene}")
    for array in arrays.values():
        array.setflags(write=False)
    return manifest, arrays


def _selection_sha256(indices_by_scene: Sequence[np.ndarray]) -> str:
    digest = hashlib.sha256()
    for scene_index, values in enumerate(indices_by_scene):
        array = np.ascontiguousarray(values, dtype=np.int64)
        digest.update(np.asarray([scene_index, len(array)], dtype=np.int64).tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()


def _select_k8(
    arrays: Mapping[str, np.ndarray],
) -> tuple[tuple[np.ndarray, ...], str]:
    selections: list[np.ndarray] = []
    scenes = arrays["per_view_scene_index"]
    frames = arrays["per_view_frame_id"]
    scores = arrays["per_view_source_score"]
    source_rows = arrays["per_view_source_row"]
    for scene_index, scene in enumerate(DEV3_SCENES):
        scene_positions = np.flatnonzero(scenes == scene_index)
        selected: list[int] = []
        for frame_id in sorted(np.unique(frames[scene_positions]).tolist()):
            positions = scene_positions[frames[scene_positions] == frame_id]
            order = sorted(
                positions.tolist(),
                key=lambda row: (
                    -float(scores[row]),
                    int(source_rows[row]),
                    int(row),
                ),
            )
            selected.extend(order[:TOP_K])
        value = np.asarray(selected, dtype=np.int64)
        if len(value) != EXPECTED_SELECTION_COUNTS[scene]:
            raise S3RShadowError(f"unexpected K8 row count for {scene}")
        selections.append(value)
    result = tuple(selections)
    digest = _selection_sha256(result)
    if digest != EXPECTED_SELECTION_SHA256:
        raise S3RShadowError(
            f"K8 selection mismatch: expected={EXPECTED_SELECTION_SHA256}, actual={digest}"
        )
    if sum(len(value) for value in result) != 1571:
        raise S3RShadowError("unexpected complete K8 row count")
    return result, digest


def _load_schedules(
    source_manifest: Mapping[str, Any], arrays: Mapping[str, np.ndarray]
) -> dict[str, dict[str, Any]]:
    source_ledgers = source_manifest.get("scenes")
    if not isinstance(source_ledgers, list):
        raise S3RShadowError("sealed Boxer scene ledgers are missing")
    by_scene = {row.get("scene_id"): row for row in source_ledgers if isinstance(row, Mapping)}
    output: dict[str, dict[str, Any]] = {}
    for scene_index, scene in enumerate(DEV3_SCENES):
        expected = EXPECTED_SCHEDULES[scene]
        path = _regular_file(
            SCHEDULE_ROOT / scene / "manifest.json", f"frozen schedule for {scene}"
        )
        actual_hash = _sha256(path)
        if actual_hash != expected["sha256"]:
            raise S3RShadowError(f"frozen schedule SHA-256 mismatch for {scene}")
        schedule = _read_json(path, f"frozen schedule for {scene}")
        recorded = schedule.get("recorded_frame_ids")
        if (
            not isinstance(recorded, list)
            or any(isinstance(value, bool) or not isinstance(value, int) for value in recorded)
            or recorded != sorted(set(recorded))
            or schedule.get("record_count") != len(recorded)
            or len(recorded) != expected["recorded_count"]
        ):
            raise S3RShadowError(f"invalid frozen schedule for {scene}")
        invalid = tuple(int(value) for value in expected["invalid_pose_frame_ids"])
        if not set(invalid).issubset(recorded):
            raise S3RShadowError(f"invalid-pose ledger is off-schedule for {scene}")
        valid = tuple(frame for frame in recorded if frame not in set(invalid))
        if len(valid) != expected["valid_count"]:
            raise S3RShadowError(f"valid schedule count mismatch for {scene}")
        source_positions = np.flatnonzero(arrays["per_view_scene_index"] == scene_index)
        candidate_frames = tuple(
            int(value)
            for value in np.unique(arrays["per_view_frame_id"][source_positions]).tolist()
        )
        if (
            len(candidate_frames) != EXPECTED_CANDIDATE_FRAME_COUNTS[scene]
            or not set(candidate_frames).issubset(valid)
            or tuple(frame for frame in valid if frame not in set(candidate_frames))
            != expected["empty_valid_frame_ids"]
        ):
            raise S3RShadowError(f"candidate/empty schedule mismatch for {scene}")
        source_ledger = by_scene.get(scene)
        if (
            not isinstance(source_ledger, Mapping)
            or source_ledger.get("sealed_schedule_manifest_sha256") != actual_hash
        ):
            raise S3RShadowError(f"source does not bind frozen schedule for {scene}")
        output[scene] = {
            "path": os.fspath(path),
            "sha256": actual_hash,
            "recorded_frame_ids": tuple(recorded),
            "valid_frame_ids": valid,
            "invalid_pose_frame_ids": invalid,
            "candidate_frame_ids": candidate_frames,
            "empty_valid_frame_ids": tuple(expected["empty_valid_frame_ids"]),
        }
    return output


def _hash_formal_t05_predictions() -> dict[str, dict[str, object]]:
    raw_root = FORMAL_T05_ROOT
    if raw_root.is_symlink():
        raise S3RShadowError("formal T05 root must not be a symlink")
    try:
        root = raw_root.resolve(strict=True)
    except (OSError, RuntimeError) as error:
        raise S3RShadowError("formal T05 root is missing") from error
    expected_root = (
        REPOSITORY_ROOT / "results" / "scannet_topk_fusion_score05"
    ).resolve(strict=True)
    if not root.is_dir() or root != expected_root:
        raise S3RShadowError(
            f"formal T05 root mismatch: expected={expected_root}, actual={root}"
        )
    output: dict[str, dict[str, object]] = {}
    for scene in DEV3_SCENES:
        path = _regular_file(
            root / f"{scene}_boxes.pkl", f"formal T05 prediction for {scene}"
        )
        actual = _sha256(path)
        expected = EXPECTED_FORMAL_T05_SHA256[scene]
        if actual != expected:
            raise S3RShadowError(
                f"formal T05 SHA-256 mismatch for {scene}: "
                f"expected={expected}, actual={actual}"
            )
        output[scene] = {
            "path": os.fspath(path),
            "sha256": actual,
            "bytes": path.stat().st_size,
        }
    return output


def _quaternion_rotation(quaternion_wxyz: np.ndarray) -> np.ndarray:
    quaternion = np.asarray(quaternion_wxyz, dtype=np.float64)
    if quaternion.shape != (4,) or not np.isfinite(quaternion).all():
        raise S3RShadowError("quaternion_wxyz must be finite with shape (4,)")
    norm_squared = float(quaternion @ quaternion)
    if not math.isfinite(norm_squared) or norm_squared <= 1e-12:
        raise S3RShadowError("quaternion_wxyz has invalid squared norm")
    w, x, y, z = quaternion
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


def _obb_corners(
    center_world: np.ndarray,
    extent_xyz: np.ndarray,
    quaternion_wxyz: np.ndarray,
) -> np.ndarray:
    center = np.asarray(center_world, dtype=np.float64)
    extent = np.asarray(extent_xyz, dtype=np.float64)
    if (
        center.shape != (3,)
        or extent.shape != (3,)
        or not np.isfinite(center).all()
        or not np.isfinite(extent).all()
        or np.any(extent <= 0.0)
    ):
        raise S3RShadowError("raw OBB center/extent is invalid")
    rotation = _quaternion_rotation(quaternion_wxyz)
    return np.ascontiguousarray(SIGNS * (extent / 2.0) @ rotation.T + center)


def _aabb_pair_metrics(left: np.ndarray, right: np.ndarray) -> tuple[float, float]:
    left_lower = left.min(axis=0)
    left_upper = left.max(axis=0)
    right_lower = right.min(axis=0)
    right_upper = right.max(axis=0)
    intersection = np.prod(
        np.maximum(np.minimum(left_upper, right_upper) - np.maximum(left_lower, right_lower), 0.0)
    )
    left_volume = np.prod(left_upper - left_lower)
    right_volume = np.prod(right_upper - right_lower)
    union = left_volume + right_volume - intersection
    iou = 0.0 if union <= 0.0 else float(intersection / union)
    center_distance = float(
        np.linalg.norm(0.5 * (left_lower + left_upper) - 0.5 * (right_lower + right_upper))
    )
    return iou, center_distance


def _rss_bytes() -> int:
    try:
        fields = Path("/proc/self/statm").read_text(encoding="ascii").split()
        return int(fields[1]) * int(os.sysconf("SC_PAGE_SIZE"))
    except (OSError, ValueError, IndexError):
        return -1


def _validate_numeric_thread_environment() -> dict[str, str]:
    actual = {
        name: os.environ.get(name, "")
        for name in REQUIRED_NUMERIC_THREAD_ENVIRONMENT
    }
    if actual != dict(REQUIRED_NUMERIC_THREAD_ENVIRONMENT):
        raise S3RShadowError(
            "numeric thread environment must be pinned exactly to one thread: "
            f"expected={dict(REQUIRED_NUMERIC_THREAD_ENVIRONMENT)}, actual={actual}"
        )
    return actual


def _percentile_ns(values: Sequence[int], percentile: float) -> int:
    if not values:
        return 0
    return int(math.ceil(float(np.percentile(np.asarray(values, dtype=np.int64), percentile))))


def _runtime_stats(values: Sequence[int]) -> dict[str, int]:
    return {
        "count": len(values),
        "p50_ns": _percentile_ns(values, 50.0),
        "p95_ns": _percentile_ns(values, 95.0),
        "max_ns": max(values, default=0),
    }


def _empty_array(values: Sequence[Any], dtype: object, tail: tuple[int, ...] = ()) -> np.ndarray:
    return np.asarray(values, dtype=dtype).reshape((len(values),) + tail)


def _run_tracking(
    *,
    source_arrays: Mapping[str, np.ndarray],
    selections: Sequence[np.ndarray],
    schedules: Mapping[str, Mapping[str, Any]],
    tracker_factory: Any = S3RReceiptTracker,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    numeric_thread_environment = _validate_numeric_thread_environment()
    selected_scene_index: list[int] = []
    selected_schedule_index: list[int] = []
    selected_frame_id: list[int] = []
    selected_rank: list[int] = []
    selected_npz_row: list[int] = []
    selected_source_row: list[int] = []
    selected_instance_id: list[int] = []
    selected_score: list[float] = []
    selected_center: list[np.ndarray] = []
    selected_extent: list[np.ndarray] = []
    selected_quaternion: list[np.ndarray] = []
    selected_corners: list[np.ndarray] = []
    assignment_track_id: list[int] = []
    assignment_action: list[int] = []
    assignment_iou: list[float] = []
    assignment_center_m: list[float] = []

    schedule_scene_index: list[int] = []
    schedule_frame_id: list[int] = []
    frame_selected_offsets = [0]
    frame_retired_offsets = [0]
    retired_track_ids: list[int] = []
    frame_prior_track_count: list[int] = []
    frame_active_track_count: list[int] = []
    frame_created_count: list[int] = []
    frame_matched_count: list[int] = []
    frame_new_receipt_count: list[int] = []
    frame_eligibility_checks: list[int] = []
    frame_tracker_cpu_ns: list[int] = []
    frame_tracker_wall_ns: list[int] = []
    frame_adapter_cpu_ns: list[int] = []
    frame_adapter_wall_ns: list[int] = []
    frame_audit_complete: list[bool] = []
    frame_cap_event_count: list[int] = []

    receipts: list[tuple[int, S3RReceipt]] = []
    scene_schedule_offsets = [0]
    scene_selected_offsets = [0]
    scene_receipt_offsets = [0]
    scene_summaries: dict[str, Any] = {}
    tracker_summaries: dict[str, Any] = {}
    rss_start = _rss_bytes()
    rss_peak = rss_start
    cold_init_wall_ns = 0
    cold_init_cpu_ns = 0

    frame_array = source_arrays["per_view_frame_id"]
    for scene_index, scene in enumerate(DEV3_SCENES):
        selection = np.asarray(selections[scene_index], dtype=np.int64)
        if len(selection) > MAX_SELECTED_ROWS_PER_SCENE:
            raise S3RShadowError(f"selected-row safety cap exceeded for {scene}")
        valid_frames = tuple(int(value) for value in schedules[scene]["valid_frame_ids"])
        if len(valid_frames) > MAX_VALID_FRAMES_PER_SCENE:
            raise S3RShadowError(f"valid-frame safety cap exceeded for {scene}")
        by_frame: dict[int, list[int]] = {frame: [] for frame in valid_frames}
        for npz_row in selection.tolist():
            frame_id = int(frame_array[npz_row])
            if frame_id not in by_frame:
                raise S3RShadowError(f"selected row is off frozen schedule for {scene}")
            by_frame[frame_id].append(int(npz_row))
        if any(len(rows) > TOP_K for rows in by_frame.values()):
            raise S3RShadowError(f"more than K8 selected rows in one frame for {scene}")

        start_wall = time.perf_counter_ns()
        start_cpu = time.process_time_ns()
        tracker = tracker_factory()
        cold_init_cpu_ns += time.process_time_ns() - start_cpu
        cold_init_wall_ns += time.perf_counter_ns() - start_wall
        anchors: dict[int, np.ndarray] = {}
        scene_selected_start = len(selected_npz_row)
        scene_receipt_start = len(receipts)
        scene_created = 0
        scene_matched = 0
        scene_retired = 0

        for frame_id in valid_frames:
            schedule_index = len(schedule_frame_id)
            rows = by_frame[frame_id]
            adapter_wall_start = time.perf_counter_ns()
            adapter_cpu_start = time.process_time_ns()
            observations: list[S3RObservation] = []
            row_corners: list[np.ndarray] = []
            for rank, npz_row in enumerate(rows):
                corners = _obb_corners(
                    source_arrays["per_view_center_world"][npz_row],
                    source_arrays["per_view_extent_xyz"][npz_row],
                    source_arrays["per_view_quaternion_wxyz"][npz_row],
                )
                row_corners.append(corners)
                observations.append(
                    S3RObservation(
                        frame_id=frame_id,
                        source_row=int(source_arrays["per_view_source_row"][npz_row]),
                        sealed_npz_row=npz_row,
                        source_instance_id=int(
                            source_arrays["per_view_source_instance_id"][npz_row]
                        ),
                        score=float(source_arrays["per_view_source_score"][npz_row]),
                        corners=corners,
                    )
                )

            tracker_wall_start = time.perf_counter_ns()
            tracker_cpu_start = time.process_time_ns()
            query = tracker.query(frame_id, observations)
            commit = tracker.commit(query)
            tracker_cpu = time.process_time_ns() - tracker_cpu_start
            tracker_wall = time.perf_counter_ns() - tracker_wall_start
            adapter_cpu = time.process_time_ns() - adapter_cpu_start
            adapter_wall = time.perf_counter_ns() - adapter_wall_start

            expected_rows = tuple(
                int(source_arrays["per_view_source_row"][row]) for row in rows
            )
            expected_npz_rows = tuple(rows)
            if (
                query.selected_source_rows != expected_rows
                or query.selected_sealed_npz_rows != expected_npz_rows
                or query.accepted_source_rows != expected_rows
                or query.accepted_sealed_npz_rows != expected_npz_rows
                or len(query.assignments) != len(rows)
                or commit.assignments != query.assignments
            ):
                raise S3RShadowError(f"tracker changed exact K8 order for {scene}/{frame_id}")
            cap_count = (
                len(query.observation_capacity_dropped_source_rows)
                + len(query.track_capacity_dropped_source_rows)
                + len(query.receipt_capacity_dropped_track_ids)
            )
            if cap_count or not query.audit_complete or not commit.audit_complete:
                raise S3RShadowError(
                    f"capacity event invalidated S3R audit for {scene}/{frame_id}"
                )
            checks = len(rows) * len(query.prior_track_ids)
            if checks > MAX_ELIGIBILITY_CHECKS_PER_FRAME:
                raise S3RShadowError(f"eligibility-check cap exceeded for {scene}/{frame_id}")

            for track_id in query.newly_retired_track_ids:
                if track_id not in anchors:
                    raise S3RShadowError("retired track is absent from committed anchor ledger")
                del anchors[track_id]
            for rank, (npz_row, corners, assignment) in enumerate(
                zip(rows, row_corners, query.assignments)
            ):
                if (
                    assignment.source_row
                    != int(source_arrays["per_view_source_row"][npz_row])
                    or assignment.sealed_npz_row != npz_row
                    or assignment.source_instance_id
                    != int(source_arrays["per_view_source_instance_id"][npz_row])
                ):
                    raise S3RShadowError("assignment provenance differs from selected row")
                if assignment.action == "matched":
                    if assignment.track_id not in anchors:
                        raise S3RShadowError("matched track has no prior committed anchor")
                    iou, center_m = _aabb_pair_metrics(corners, anchors[assignment.track_id])
                    if iou < 0.10 or center_m > 0.50:
                        raise S3RShadowError("exported matched metrics violate tracker gates")
                    action_code = 1
                elif assignment.action == "created":
                    if assignment.track_id in anchors:
                        raise S3RShadowError("created track already exists in prior ledger")
                    iou, center_m = -1.0, -1.0
                    action_code = 0
                else:
                    raise S3RShadowError("unknown tracker assignment action")

                selected_scene_index.append(scene_index)
                selected_schedule_index.append(schedule_index)
                selected_frame_id.append(frame_id)
                selected_rank.append(rank)
                selected_npz_row.append(npz_row)
                selected_source_row.append(assignment.source_row)
                selected_instance_id.append(assignment.source_instance_id)
                selected_score.append(float(source_arrays["per_view_source_score"][npz_row]))
                selected_center.append(
                    np.asarray(source_arrays["per_view_center_world"][npz_row], dtype=np.float64)
                )
                selected_extent.append(
                    np.asarray(source_arrays["per_view_extent_xyz"][npz_row], dtype=np.float64)
                )
                selected_quaternion.append(
                    np.asarray(
                        source_arrays["per_view_quaternion_wxyz"][npz_row],
                        dtype=np.float64,
                    )
                )
                selected_corners.append(corners)
                assignment_track_id.append(assignment.track_id)
                assignment_action.append(action_code)
                assignment_iou.append(iou)
                assignment_center_m.append(center_m)
                anchors[assignment.track_id] = corners

            if set(anchors) != set(commit.active_track_ids):
                raise S3RShadowError("committed anchor ledger differs from tracker snapshot")
            receipts.extend((scene_index, receipt) for receipt in commit.newly_frozen_receipts)

            schedule_scene_index.append(scene_index)
            schedule_frame_id.append(frame_id)
            frame_selected_offsets.append(len(selected_npz_row))
            retired_track_ids.extend(query.newly_retired_track_ids)
            frame_retired_offsets.append(len(retired_track_ids))
            frame_prior_track_count.append(len(query.prior_track_ids))
            frame_active_track_count.append(len(commit.active_track_ids))
            frame_created_count.append(len(commit.created_track_ids))
            frame_matched_count.append(len(commit.matched_track_ids))
            frame_new_receipt_count.append(len(commit.newly_frozen_receipts))
            frame_eligibility_checks.append(checks)
            frame_tracker_cpu_ns.append(tracker_cpu)
            frame_tracker_wall_ns.append(tracker_wall)
            frame_adapter_cpu_ns.append(adapter_cpu)
            frame_adapter_wall_ns.append(adapter_wall)
            frame_audit_complete.append(True)
            frame_cap_event_count.append(cap_count)
            scene_created += len(commit.created_track_ids)
            scene_matched += len(commit.matched_track_ids)
            scene_retired += len(commit.newly_retired_track_ids)
            current_rss = _rss_bytes()
            if current_rss >= 0:
                rss_peak = max(rss_peak, current_rss)

        tracker_summary = tracker.summary()
        if (
            tracker_summary.get("audit_complete") is not True
            or tracker_summary.get("keyframes") != len(valid_frames)
            or tracker_summary.get("observations_received") != len(selection)
            or tracker_summary.get("observations_accepted") != len(selection)
            or tracker_summary.get("observation_capacity_drops") != 0
            or tracker_summary.get("track_capacity_drops") != 0
            or tracker_summary.get("receipt_capacity_drops") != 0
        ):
            raise S3RShadowError(f"incomplete tracker summary for {scene}")
        final_receipts = tracker.receipts()
        scene_receipt_rows = [receipt for index, receipt in receipts if index == scene_index]
        incremental_by_track_id = {
            receipt.track_id: receipt for receipt in scene_receipt_rows
        }
        final_by_track_id = {receipt.track_id: receipt for receipt in final_receipts}
        if (
            len(incremental_by_track_id) != len(scene_receipt_rows)
            or len(final_by_track_id) != len(final_receipts)
            or incremental_by_track_id != final_by_track_id
        ):
            raise S3RShadowError(f"incremental/final receipt identity mismatch for {scene}")
        # Per-frame counters preserve causal confirmation chronology.  The
        # exported receipt table itself has a separate deterministic contract:
        # scene order, then ascending track_id, independent of confirmation
        # order.
        del receipts[scene_receipt_start:]
        receipts.extend((scene_index, receipt) for receipt in final_receipts)
        tracker_summaries[scene] = tracker_summary
        scene_summaries[scene] = {
            "valid_frame_count": len(valid_frames),
            "candidate_frame_count": len(schedules[scene]["candidate_frame_ids"]),
            "empty_valid_frame_ids": list(schedules[scene]["empty_valid_frame_ids"]),
            "selected_row_count": len(selection),
            "created_assignment_count": scene_created,
            "matched_assignment_count": scene_matched,
            "retired_track_count": scene_retired,
            "receipt_count": len(final_receipts),
        }
        if len(selected_npz_row) - scene_selected_start != len(selection):
            raise S3RShadowError(f"selected-row export is incomplete for {scene}")
        if len(receipts) - scene_receipt_start != len(final_receipts):
            raise S3RShadowError(f"receipt export is incomplete for {scene}")
        scene_schedule_offsets.append(len(schedule_frame_id))
        scene_selected_offsets.append(len(selected_npz_row))
        scene_receipt_offsets.append(len(receipts))

    expected_flat_selection = np.concatenate(selections).astype(np.int64, copy=False)
    if not np.array_equal(np.asarray(selected_npz_row, dtype=np.int64), expected_flat_selection):
        raise S3RShadowError("complete exported K8 membership differs from frozen selection")

    selected_index_by_scene_row = {
        (scene_index, npz_row): index
        for index, (scene_index, npz_row) in enumerate(
            zip(selected_scene_index, selected_npz_row)
        )
    }
    receipt_scene_index: list[int] = []
    receipt_track_id: list[int] = []
    receipt_confirmation_frame_id: list[int] = []
    receipt_corners: list[np.ndarray] = []
    receipt_medoid_index: list[int] = []
    receipt_pairwise_iou: list[np.ndarray] = []
    receipt_pairwise_center: list[np.ndarray] = []
    receipt_mean_score: list[float] = []
    receipt_median_iou: list[float] = []
    receipt_center_rms: list[float] = []
    receipt_min_extent: list[float] = []
    evidence_offsets = [0]
    evidence_selected_index: list[int] = []
    evidence_frame_id: list[int] = []
    evidence_source_row: list[int] = []
    evidence_npz_row: list[int] = []
    evidence_instance_id: list[int] = []
    evidence_score: list[float] = []
    evidence_corners: list[np.ndarray] = []
    for scene_index, receipt in receipts:
        if len(receipt.evidence_frame_ids) != 3:
            raise S3RShadowError("receipt does not contain exactly three evidence rows")
        receipt_scene_index.append(scene_index)
        receipt_track_id.append(receipt.track_id)
        receipt_confirmation_frame_id.append(receipt.confirmation_frame_id)
        receipt_corners.append(receipt.corners)
        receipt_medoid_index.append(receipt.medoid_evidence_index)
        receipt_pairwise_iou.append(receipt.pairwise_aabb_iou)
        receipt_pairwise_center.append(receipt.pairwise_center_distance_m)
        receipt_mean_score.append(receipt.raw_mean_score)
        receipt_median_iou.append(receipt.median_pairwise_aabb_iou)
        receipt_center_rms.append(receipt.center_rms_m)
        receipt_min_extent.append(receipt.min_medoid_aabb_extent_m)
        for index in range(3):
            npz_row = receipt.evidence_sealed_npz_rows[index]
            key = (scene_index, npz_row)
            if key not in selected_index_by_scene_row:
                raise S3RShadowError("receipt evidence is absent from selected K8 rows")
            evidence_selected_index.append(selected_index_by_scene_row[key])
            evidence_frame_id.append(receipt.evidence_frame_ids[index])
            evidence_source_row.append(receipt.evidence_source_rows[index])
            evidence_npz_row.append(npz_row)
            evidence_instance_id.append(receipt.evidence_source_instance_ids[index])
            evidence_score.append(receipt.evidence_scores[index])
            evidence_corners.append(receipt.evidence_corners[index])
        evidence_offsets.append(len(evidence_npz_row))

    arrays = {
        "scene_ids": np.asarray(DEV3_SCENES, dtype="<U12"),
        "scene_schedule_offsets": np.asarray(scene_schedule_offsets, dtype=np.int64),
        "scene_selected_offsets": np.asarray(scene_selected_offsets, dtype=np.int64),
        "scene_receipt_offsets": np.asarray(scene_receipt_offsets, dtype=np.int64),
        "schedule_scene_index": np.asarray(schedule_scene_index, dtype=np.int16),
        "schedule_frame_id": np.asarray(schedule_frame_id, dtype=np.int64),
        "frame_selected_offsets": np.asarray(frame_selected_offsets, dtype=np.int64),
        "frame_retired_offsets": np.asarray(frame_retired_offsets, dtype=np.int64),
        "retired_track_id": np.asarray(retired_track_ids, dtype=np.int64),
        "frame_prior_track_count": np.asarray(frame_prior_track_count, dtype=np.int32),
        "frame_active_track_count": np.asarray(frame_active_track_count, dtype=np.int32),
        "frame_created_count": np.asarray(frame_created_count, dtype=np.int16),
        "frame_matched_count": np.asarray(frame_matched_count, dtype=np.int16),
        "frame_new_receipt_count": np.asarray(frame_new_receipt_count, dtype=np.int16),
        "frame_eligibility_check_count": np.asarray(frame_eligibility_checks, dtype=np.int32),
        "frame_tracker_cpu_ns": np.asarray(frame_tracker_cpu_ns, dtype=np.int64),
        "frame_tracker_wall_ns": np.asarray(frame_tracker_wall_ns, dtype=np.int64),
        "frame_adapter_cpu_ns": np.asarray(frame_adapter_cpu_ns, dtype=np.int64),
        "frame_adapter_wall_ns": np.asarray(frame_adapter_wall_ns, dtype=np.int64),
        "frame_audit_complete": np.asarray(frame_audit_complete, dtype=np.bool_),
        "frame_cap_event_count": np.asarray(frame_cap_event_count, dtype=np.int16),
        "selected_scene_index": np.asarray(selected_scene_index, dtype=np.int16),
        "selected_schedule_index": np.asarray(selected_schedule_index, dtype=np.int64),
        "selected_frame_id": np.asarray(selected_frame_id, dtype=np.int64),
        "selected_rank_in_frame": np.asarray(selected_rank, dtype=np.int8),
        "selected_sealed_npz_row": np.asarray(selected_npz_row, dtype=np.int64),
        "selected_source_row": np.asarray(selected_source_row, dtype=np.int64),
        "selected_source_instance_id": np.asarray(selected_instance_id, dtype=np.int64),
        "selected_source_score": np.asarray(selected_score, dtype=np.float64),
        "selected_center_world": _empty_array(selected_center, np.float64, (3,)),
        "selected_extent_xyz": _empty_array(selected_extent, np.float64, (3,)),
        "selected_quaternion_wxyz": _empty_array(selected_quaternion, np.float64, (4,)),
        "selected_corners_world": _empty_array(selected_corners, np.float64, (8, 3)),
        "assignment_track_id": np.asarray(assignment_track_id, dtype=np.int64),
        "assignment_action": np.asarray(assignment_action, dtype=np.int8),
        "assignment_aabb_iou": np.asarray(assignment_iou, dtype=np.float64),
        "assignment_center_distance_m": np.asarray(assignment_center_m, dtype=np.float64),
        "receipt_scene_index": np.asarray(receipt_scene_index, dtype=np.int16),
        "receipt_track_id": np.asarray(receipt_track_id, dtype=np.int64),
        "receipt_confirmation_frame_id": np.asarray(
            receipt_confirmation_frame_id, dtype=np.int64
        ),
        "receipt_corners_world": _empty_array(receipt_corners, np.float64, (8, 3)),
        "receipt_medoid_evidence_index": np.asarray(receipt_medoid_index, dtype=np.int8),
        "receipt_pairwise_aabb_iou": _empty_array(
            receipt_pairwise_iou, np.float64, (3, 3)
        ),
        "receipt_pairwise_center_distance_m": _empty_array(
            receipt_pairwise_center, np.float64, (3, 3)
        ),
        "receipt_raw_mean_score": np.asarray(receipt_mean_score, dtype=np.float64),
        "receipt_median_pairwise_aabb_iou": np.asarray(
            receipt_median_iou, dtype=np.float64
        ),
        "receipt_center_rms_m": np.asarray(receipt_center_rms, dtype=np.float64),
        "receipt_min_medoid_aabb_extent_m": np.asarray(
            receipt_min_extent, dtype=np.float64
        ),
        "evidence_offsets": np.asarray(evidence_offsets, dtype=np.int64),
        "evidence_selected_index": np.asarray(evidence_selected_index, dtype=np.int64),
        "evidence_frame_id": np.asarray(evidence_frame_id, dtype=np.int64),
        "evidence_source_row": np.asarray(evidence_source_row, dtype=np.int64),
        "evidence_sealed_npz_row": np.asarray(evidence_npz_row, dtype=np.int64),
        "evidence_source_instance_id": np.asarray(evidence_instance_id, dtype=np.int64),
        "evidence_source_score": np.asarray(evidence_score, dtype=np.float64),
        "evidence_corners_world": _empty_array(evidence_corners, np.float64, (8, 3)),
    }
    trace_bytes = int(sum(array.nbytes for array in arrays.values()))
    if trace_bytes > MAX_TRACE_UNCOMPRESSED_BYTES:
        raise S3RShadowError("uncompressed diagnostic trace cap exceeded")
    rss_end = _rss_bytes()
    rss_measurement_complete = rss_start >= 0 and rss_peak >= 0 and rss_end >= 0
    rss_increment = (
        max(0, int(rss_peak - rss_start)) if rss_measurement_complete else -1
    )
    tracker_cpu_stats = _runtime_stats(frame_tracker_cpu_ns)
    tracker_cpu_budget_pass = (
        tracker_cpu_stats["p95_ns"] <= TRACKER_CPU_P95_LIMIT_NS
        and tracker_cpu_stats["max_ns"] <= TRACKER_CPU_MAX_LIMIT_NS
    )
    tracker_memory_upper_bound_pass = (
        rss_measurement_complete
        and rss_increment <= MAX_TRACKER_INCREMENTAL_MEMORY_BYTES
    )
    resource_budget_pass = (
        tracker_cpu_budget_pass and tracker_memory_upper_bound_pass
    )
    runtime = {
        "cold_tracker_initialization_cpu_ns": cold_init_cpu_ns,
        "cold_tracker_initialization_wall_ns": cold_init_wall_ns,
        "tracker_cpu": tracker_cpu_stats,
        "tracker_wall": _runtime_stats(frame_tracker_wall_ns),
        "adapter_cpu": _runtime_stats(frame_adapter_cpu_ns),
        "adapter_wall": _runtime_stats(frame_adapter_wall_ns),
        "rss_start_bytes": rss_start,
        "rss_peak_sampled_bytes": rss_peak,
        "rss_end_bytes": rss_end,
        "rss_measurement_complete": rss_measurement_complete,
        "runner_incremental_rss_upper_bound_bytes": rss_increment,
        "tracker_memory_limit_bytes": MAX_TRACKER_INCREMENTAL_MEMORY_BYTES,
        "tracker_cpu_p95_limit_ns": TRACKER_CPU_P95_LIMIT_NS,
        "tracker_cpu_max_limit_ns": TRACKER_CPU_MAX_LIMIT_NS,
        "tracker_cpu_budget_pass": tracker_cpu_budget_pass,
        "tracker_memory_upper_bound_pass": tracker_memory_upper_bound_pass,
        "resource_budget_pass": resource_budget_pass,
        "numeric_thread_environment": numeric_thread_environment,
        "numeric_thread_environment_pinned": True,
        "precomputed_sidecar_replay": True,
        "integrated_provider_runtime_qualified": False,
        "native_online_fps_claimed": False,
    }
    summary = {
        "audit_complete": resource_budget_pass,
        "cap_event_count": int(sum(frame_cap_event_count)),
        "trace_uncompressed_bytes": trace_bytes,
        "selected_row_count": len(selected_npz_row),
        "assignment_count": len(assignment_track_id),
        "receipt_count": len(receipt_track_id),
        "evidence_count": len(evidence_npz_row),
        "valid_frame_count": len(schedule_frame_id),
        "scenes": scene_summaries,
        "tracker_summaries": tracker_summaries,
        "runtime": runtime,
    }
    return arrays, summary


def _snapshot_inputs(
    *,
    fixed_assets: Mapping[str, Mapping[str, object]],
    source_arrays: Mapping[str, np.ndarray],
    schedules: Mapping[str, Mapping[str, Any]],
    native: Mapping[str, Mapping[str, object]],
) -> dict[str, Any]:
    runner = _regular_file(Path(__file__), "S3R runner source")
    runner_test = _regular_file(RUNNER_TEST, "S3R runner focused tests")
    return {
        "fixed_assets": fixed_assets,
        "runner_source": {
            "path": os.fspath(runner),
            "sha256": _sha256(runner),
            "bytes": runner.stat().st_size,
        },
        "runner_test": {
            "path": os.fspath(runner_test),
            "sha256": _sha256(runner_test),
            "bytes": runner_test.stat().st_size,
        },
        "allowed_numeric_source_content_sha256": _array_content_sha256(source_arrays),
        "schedules": {
            scene: {
                "path": schedules[scene]["path"],
                "sha256": _sha256(Path(schedules[scene]["path"])),
            }
            for scene in DEV3_SCENES
        },
        "native_t05": native,
    }


def _native_hashes(native: Mapping[str, Mapping[str, object]]) -> dict[str, str]:
    return {scene: str(native[scene]["sha256"]) for scene in DEV3_SCENES}


def _build_manifest(
    *,
    arrays: Mapping[str, np.ndarray],
    summary: Mapping[str, Any],
    selection_sha256: str,
    schedules: Mapping[str, Mapping[str, Any]],
    input_before: Mapping[str, Any],
    input_after: Mapping[str, Any],
    native_before: Mapping[str, Mapping[str, object]],
    native_after: Mapping[str, Mapping[str, object]],
    npz_sha256: str,
) -> dict[str, Any]:
    if input_before != input_after:
        raise S3RShadowError("one or more frozen inputs changed during S3R replay")
    before_native = _native_hashes(native_before)
    after_native = _native_hashes(native_after)
    if before_native != after_native:
        raise S3RShadowError("formal T05 bytes changed during S3R replay")
    if summary.get("audit_complete") is not True or summary.get("cap_event_count") != 0:
        raise S3RShadowError("incomplete/capped trace cannot receive the valid schema")
    runtime = summary.get("runtime")
    if (
        not isinstance(runtime, Mapping)
        or runtime.get("tracker_cpu_budget_pass") is not True
        or runtime.get("tracker_memory_upper_bound_pass") is not True
        or runtime.get("resource_budget_pass") is not True
        or runtime.get("numeric_thread_environment_pinned") is not True
        or runtime.get("numeric_thread_environment")
        != dict(REQUIRED_NUMERIC_THREAD_ENVIRONMENT)
    ):
        raise S3RShadowError("resource-budget failure cannot receive the valid schema")
    return {
        "schema": SCHEMA,
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
        "valid_frame_count": int(summary["valid_frame_count"]),
        "selected_row_count": int(summary["selected_row_count"]),
        "assignment_count": int(summary["assignment_count"]),
        "receipt_count": int(summary["receipt_count"]),
        "evidence_count": int(summary["evidence_count"]),
        "cap_event_count": 0,
        "trace_uncompressed_bytes": int(summary["trace_uncompressed_bytes"]),
        "trace_uncompressed_cap_bytes": MAX_TRACE_UNCOMPRESSED_BYTES,
        "selection": {
            "top_k_per_valid_candidate_frame": TOP_K,
            "rule": (
                "descending_source_score_then_ascending_source_row_then_"
                "ascending_sealed_npz_row"
            ),
            "selection_sha256": selection_sha256,
            "expected_selection_sha256": EXPECTED_SELECTION_SHA256,
            "selected_count_by_scene": dict(EXPECTED_SELECTION_COUNTS),
            "candidate_frame_count_by_scene": dict(EXPECTED_CANDIDATE_FRAME_COUNTS),
            "complete_exact_k8_membership": True,
            "selection_used_gt": False,
            "selection_used_semantics": False,
            "selection_used_only_frozen_source_score": True,
            "selected_npz_rows_sha256": _hash_array(arrays["selected_sealed_npz_row"]),
        },
        "contracts": {
            "preregistration_sha256": EXPECTED_FIXED_HASHES["preregistration"][1],
            "tracker_source_sha256": EXPECTED_FIXED_HASHES["tracker_source"][1],
            "tracker_test_sha256": EXPECTED_FIXED_HASHES["tracker_test"][1],
            "sealed_boxer_json_sha256": EXPECTED_FIXED_HASHES["sealed_json"][1],
            "sealed_boxer_npz_sha256": EXPECTED_FIXED_HASHES["sealed_npz"][1],
            "sealed_boxer_candidate_content_sha256": EXPECTED_SOURCE_CONTENT_SHA256,
            "topk_receipt_sha256": EXPECTED_FIXED_HASHES["topk_receipt"][1],
            "topk_tool_sha256": EXPECTED_FIXED_HASHES["topk_tool"][1],
            "schedule_sha256_by_scene": {
                scene: schedules[scene]["sha256"] for scene in DEV3_SCENES
            },
        },
        "formal_t05_root": os.fspath(FORMAL_T05_ROOT.resolve()),
        "formal_t05_expected_sha256": dict(EXPECTED_FORMAL_T05_SHA256),
        "native_prediction_sha256_before": before_native,
        "native_prediction_sha256_after": after_native,
        "native_prediction_hash_identity": True,
        "input_sha256_before": input_before,
        "input_sha256_after": input_after,
        "input_hash_identity": True,
        "geometry": {
            "coordinate_frame": "scannet_world",
            "quaternion_convention": "Hamilton_wxyz",
            "quaternion_scale": "2_over_squared_norm",
            "corner_sign_order": SIGNS.astype(np.int8).tolist(),
            "corner_transform": "local_at_R_transpose_plus_center",
            "association_geometry": "world_axis_enclosing_AABB",
            "match_aabb_iou_gte": 0.10,
            "match_center_distance_m_lte": 0.50,
            "minimum_distinct_provider_frames": 3,
            "active_track_ttl_valid_keyframes": 10,
            "receipt_geometry": "first_three_AABB_IoU_medoid",
        },
        "caps": {
            "max_observations_per_frame": TOP_K,
            "max_live_tracks": 1024,
            "max_receipts": 1024,
            "max_valid_frames_per_scene": MAX_VALID_FRAMES_PER_SCENE,
            "max_selected_rows_per_scene": MAX_SELECTED_ROWS_PER_SCENE,
            "max_eligibility_checks_per_frame": MAX_ELIGIBILITY_CHECKS_PER_FRAME,
            "max_trace_uncompressed_bytes": MAX_TRACE_UNCOMPRESSED_BYTES,
            "cap_failure_publishes_valid_seal": False,
        },
        "runtime": summary["runtime"],
        "scenes": summary["scenes"],
        "tracker_summaries": summary["tracker_summaries"],
        "npz_file": OUTPUT_NPZ_NAME,
        "npz_sha256": npz_sha256,
        "candidate_content_sha256": _array_content_sha256(arrays),
        "conclusion_guardrail": (
            "Receipt-only matching headroom is not AP, precision, birth, or realtime "
            "integration evidence. No GT or evaluator was opened by this shadow."
        ),
    }


def _check_output_location(output_root: Path) -> Path:
    raw = Path(output_root)
    if os.path.lexists(raw):
        raise S3RShadowError(f"refusing to overwrite output root: {raw}")
    output = raw.resolve(strict=False)
    protected = (SEALED_ROOT.resolve(), SCHEDULE_ROOT.resolve(), FORMAL_T05_ROOT.resolve())
    for root in protected:
        try:
            output.relative_to(root)
        except ValueError:
            continue
        raise S3RShadowError(f"output root is inside protected input: {root}")
    return output


def _verify_staged_pair(
    *,
    directory: Path,
    arrays: Mapping[str, np.ndarray],
    manifest: Mapping[str, Any],
    npz_payload: bytes,
) -> None:
    npz_path = _regular_file(directory / OUTPUT_NPZ_NAME, "staged S3R NPZ")
    json_path = _regular_file(directory / OUTPUT_JSON_NAME, "staged S3R JSON")
    expected_npz_hash = hashlib.sha256(npz_payload).hexdigest()
    if _sha256(npz_path) != expected_npz_hash or manifest.get("npz_sha256") != expected_npz_hash:
        raise S3RShadowError("staged S3R NPZ hash mismatch")
    loaded_manifest = _read_json(json_path, "staged S3R JSON")
    if loaded_manifest != dict(manifest):
        raise S3RShadowError("staged S3R JSON differs from in-memory seal")
    try:
        with np.load(npz_path, allow_pickle=False) as source:
            if set(source.files) != set(arrays):
                raise S3RShadowError("staged S3R NPZ array set mismatch")
            loaded = {name: np.array(source[name], copy=True) for name in source.files}
    except (OSError, ValueError) as error:
        if isinstance(error, S3RShadowError):
            raise
        raise S3RShadowError("could not verify staged S3R NPZ") from error
    if _array_content_sha256(loaded) != manifest.get("candidate_content_sha256"):
        raise S3RShadowError("staged S3R content SHA-256 mismatch")


def _publish_create_only(
    *,
    output_root: Path,
    arrays: Mapping[str, np.ndarray],
    manifest: Mapping[str, Any],
    npz_payload: bytes | None = None,
) -> None:
    if manifest.get("audit_complete") is not True or manifest.get("cap_event_count") != 0:
        raise S3RShadowError("refusing to publish an incomplete or capped trace")
    runtime = manifest.get("runtime")
    if (
        not isinstance(runtime, Mapping)
        or runtime.get("tracker_cpu_budget_pass") is not True
        or runtime.get("tracker_memory_upper_bound_pass") is not True
        or runtime.get("resource_budget_pass") is not True
        or runtime.get("numeric_thread_environment_pinned") is not True
        or runtime.get("numeric_thread_environment")
        != dict(REQUIRED_NUMERIC_THREAD_ENVIRONMENT)
    ):
        raise S3RShadowError("refusing to publish a resource-budget failure")
    output = _check_output_location(output_root)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.parent.is_symlink():
        raise S3RShadowError("output parent must not be a symlink")
    payload = _deterministic_npz_bytes(arrays) if npz_payload is None else npz_payload
    if hashlib.sha256(payload).hexdigest() != manifest.get("npz_sha256"):
        raise S3RShadowError("manifest does not bind deterministic NPZ bytes")
    if _array_content_sha256(arrays) != manifest.get("candidate_content_sha256"):
        raise S3RShadowError("manifest does not bind deterministic array content")
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.stage.", dir=output.parent))
    published = False
    try:
        _write_exclusive_fsync(staging / OUTPUT_NPZ_NAME, payload)
        _write_exclusive_fsync(staging / OUTPUT_JSON_NAME, _json_bytes(manifest))
        _fsync_directory(staging)
        _verify_staged_pair(
            directory=staging, arrays=arrays, manifest=manifest, npz_payload=payload
        )
        _rename_noreplace(staging, output)
        published = True
        _fsync_directory(output.parent)
        if _sha256(output / OUTPUT_NPZ_NAME) != manifest["npz_sha256"]:
            raise S3RShadowError("published S3R NPZ differs from staged bytes")
    finally:
        if not published and staging.exists():
            shutil.rmtree(staging, ignore_errors=True)


def run_s3r_shadow(*, output_root: Path) -> dict[str, Any]:
    output = _check_output_location(output_root)
    fixed_before = _validate_fixed_assets()
    source_manifest, source_arrays = _load_sealed_candidates()
    source_memory_hash_before = _array_content_sha256(source_arrays)
    selections, selection_hash = _select_k8(source_arrays)
    schedules_before = _load_schedules(source_manifest, source_arrays)
    native_before = _hash_formal_t05_predictions()
    input_before = _snapshot_inputs(
        fixed_assets=fixed_before,
        source_arrays=source_arrays,
        schedules=schedules_before,
        native=native_before,
    )

    arrays, summary = _run_tracking(
        source_arrays=source_arrays,
        selections=selections,
        schedules=schedules_before,
    )

    fixed_after = _validate_fixed_assets()
    schedules_after = _load_schedules(source_manifest, source_arrays)
    native_after = _hash_formal_t05_predictions()
    input_after = _snapshot_inputs(
        fixed_assets=fixed_after,
        source_arrays=source_arrays,
        schedules=schedules_after,
        native=native_after,
    )
    if _array_content_sha256(source_arrays) != source_memory_hash_before:
        raise S3RShadowError("loaded source arrays changed during S3R replay")

    npz_payload = _deterministic_npz_bytes(arrays)
    npz_sha256 = hashlib.sha256(npz_payload).hexdigest()
    manifest = _build_manifest(
        arrays=arrays,
        summary=summary,
        selection_sha256=selection_hash,
        schedules=schedules_before,
        input_before=input_before,
        input_after=input_after,
        native_before=native_before,
        native_after=native_after,
        npz_sha256=npz_sha256,
    )
    _publish_create_only(
        output_root=output,
        arrays=arrays,
        manifest=manifest,
        npz_payload=npz_payload,
    )
    return manifest


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    manifest = run_s3r_shadow(output_root=args.output_root)
    print(
        json.dumps(
            {
                "schema": manifest["schema"],
                "output_root": os.fspath(args.output_root.resolve()),
                "scene_order": manifest["scene_order"],
                "selected_row_count": manifest["selected_row_count"],
                "receipt_count": manifest["receipt_count"],
                "gt_access": False,
                "birth": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
