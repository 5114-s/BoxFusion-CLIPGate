#!/usr/bin/env python3
"""Replay the sealed H10 raw-Boxer K8 source through the past-only tracker.

The runner accepts exactly one numeric source JSON/NPZ pair produced by the
approved H10 source sealer.  K8 membership is consumed from that seal and is
independently verified; it is never re-selected for the experiment.  Every
entry in the sealed 769-frame ledger, including empty frames, is replayed in
order with ``query`` before ``commit``.  The resulting assignment/receipt
trace is shadow-only and cannot create or mutate native BoxFusion predictions.

No annotation, label, evaluator, AP code, RGB/depth input, native prediction,
or Python-pickle deserialization surface exists in this module.  Publication
is create-only and uses Linux ``renameat2(RENAME_NOREPLACE)``.
"""

from __future__ import annotations

import argparse
import ast
import ctypes
import errno
import hashlib
import io
import json
import math
import os
from pathlib import Path
import secrets
import stat
import sys
import time
from typing import Any, Callable, Mapping, Sequence
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


SCHEMA = "boxfusion.s3r_h10_k8_receipt_shadow.v1"
EXPECTED_SOURCE_SCHEMA = "boxfusion.s3r_h10_raw_boxer_source.v1"
OUTPUT_JSON_NAME = "S3R_H10_K8_RECEIPT_SHADOW.json"
OUTPUT_NPZ_NAME = "S3R_H10_K8_RECEIPT_SHADOW.npz"
SOURCE_JSON_NAME = "S3R_H10_RAW_BOXER_SOURCE.json"
SOURCE_NPZ_NAME = "S3R_H10_RAW_BOXER_SOURCE.npz"

SOURCE_SEALER = REPOSITORY_ROOT / "tools" / "seal_scannet_s3r_h10_raw_boxer_source.py"
SOURCE_SEALER_TEST = REPOSITORY_ROOT / "tests" / "test_seal_scannet_s3r_h10_raw_boxer_source.py"
TRACKER_SOURCE = REPOSITORY_ROOT / "boxfusion" / "s3r_receipt_tracker.py"
TRACKER_TEST = REPOSITORY_ROOT / "tests" / "test_s3r_receipt_tracker.py"
RUNNER_TEST = REPOSITORY_ROOT / "tests" / "test_run_scannet_s3r_h10_k8_receipt_shadow.py"

EXPECTED_SOURCE_SEALER_SHA256 = "46642862d78ebc10f88e23b869607e4d0fbd3f61f9644fe0df7122983dc7fea7"
EXPECTED_SOURCE_SEALER_TEST_SHA256 = "e395cf820d6ffa3a9dd607d23c5286d1f9939b3ae98b63c2fa9a4f52acc1ffaf"
EXPECTED_TRACKER_SHA256 = "277316c36b7a7fcb8005a24e907e0f232e41f6b5874411293eb26b0744df9628"
EXPECTED_TRACKER_TEST_SHA256 = "f08fd59ee2888c936e5b783de668fd789ba6b676bc4864e001b000ea287b1e3c"
EXPECTED_SCHEDULE_SHA256 = "1ce565a65510b80d69a0402fe7a40ea89920625f6a81147d42f9232f7a7761e9"
EXPECTED_SOURCE_JSON_SHA256 = "ca65214f3e6327cea66ec8cb700ab3501572be9325af4366beaffa2b7cc2859e"
EXPECTED_SOURCE_NPZ_SHA256 = "fdb688cc1372985f2ffaf3d257ed470cd4de28ff42f7a2d04a5f72311a1225f2"
EXPECTED_SOURCE_ARRAY_CONTENT_SHA256 = "a5efdb8d0d2c7b95f63368a3249229659a1052c400539321ce461da32732b862"
EXPECTED_K8_MEMBERSHIP_SHA256 = "a2a94b11461e8c1bdd15d6a4ad99d058f42db6fd73690c69269ff1b89deb6391"
EXPECTED_K8_MEMBERSHIP_COUNT = 4557
DEFAULT_SOURCE_ROOT = REPOSITORY_ROOT / "logs" / "scannet_s3r_h10_raw_boxer_source_score05_v1"
EXPECTED_SCENE_COUNT = 10
EXPECTED_EXACT_FRAME_COUNT = 769
EXPECTED_RAW_FRAME_COUNT = 770

TOP_K = 8
SOURCE_INSTANCE_STRIDE = 2048
MAX_RAW_ROWS = EXPECTED_EXACT_FRAME_COUNT * SOURCE_INSTANCE_STRIDE
MAX_SOURCE_JSON_BYTES = 32 * 1024 * 1024
MAX_SOURCE_NPZ_BYTES = 512 * 1024 * 1024
MAX_CONTRACT_BYTES = 4 * 1024 * 1024
MAX_CODE_BYTES = 8 * 1024 * 1024
MAX_SCENES = 32
MAX_VALID_FRAMES_PER_SCENE = 4096
MAX_SELECTED_ROWS_PER_SCENE = TOP_K * MAX_VALID_FRAMES_PER_SCENE
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

SOURCE_ARRAYS = frozenset(
    {
        "scene_ids",
        "per_view_scene_index",
        "per_view_frame_id",
        "per_view_source_row",
        "per_view_source_instance_id",
        "per_view_source_score",
        "per_view_center_world",
        "per_view_extent_xyz",
        "per_view_quaternion_wxyz",
    }
)
SOURCE_MANIFEST_KEYS = frozenset(
    {
        "schema",
        "mode",
        "create_only",
        "association_applied",
        "tracking_enabled",
        "tracked_artifact_present",
        "coordinate_frame",
        "coordinate_contract_sha256",
        "scene_ids",
        "scene_count",
        "exact_frame_count",
        "raw_frame_count",
        "raw_row_count",
        "empty_frame_count",
        "empty_frame_identities",
        "frame_row_ledger",
        "scene_row_counts",
        "source_instance_id_rule",
        "provider_bindings",
        "input_identity",
        "k8",
        "array_names",
        "array_content_sha256",
        "npz_file",
        "npz_sha256",
    }
)
K8_KEYS = frozenset(
    {
        "top_k",
        "sort_key",
        "identity_columns",
        "membership_identities",
        "membership_count",
        "membership_per_scene",
        "membership_sha256",
    }
)
K8_COLUMNS = (
    "scene_index",
    "frame_id",
    "source_row",
    "source_instance_id",
    "sealed_npz_row",
)
K8_SORT_KEY = ("descending_source_score", "source_row", "sealed_npz_row")
PROVIDER_BINDING_KEYS = frozenset(
    {
        "schedule_sha256",
        "run_provenance_sha256",
        "final_seal_sha256",
        "journal_sha256",
        "provider_contract_sha256",
        "frozen_assets_sha256",
        "exact_input_ledger_sha256",
        "code_hashes",
        "model_hashes",
        "protocol_hashes",
    }
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

_HEX = frozenset("0123456789abcdef")
_RENAME_NOREPLACE = 1


class H10ReceiptShadowError(RuntimeError):
    """A sealed-input, causal-trace, or publication invariant failed."""


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in _HEX for character in value)
    )


def _require_sha256(value: object, label: str) -> str:
    if not _is_sha256(value):
        raise H10ReceiptShadowError(f"{label} must be lowercase SHA-256 hex")
    return str(value)


def _duplicate_guard(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _parse_json_bytes(payload: bytes, label: str) -> dict[str, Any]:
    try:
        value = json.loads(payload.decode("utf-8"), object_pairs_hook=_duplicate_guard)
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise H10ReceiptShadowError(f"{label} is not strict UTF-8 JSON") from error
    if not isinstance(value, dict):
        raise H10ReceiptShadowError(f"{label} root must be an object")
    return value


def _hash_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _assert_no_symlink_ancestors(path: Path, label: str) -> None:
    absolute = Path(os.path.abspath(os.fspath(path)))
    parts = absolute.parts
    current = Path(parts[0])
    for part in parts[1:]:
        current = current / part
        try:
            metadata = os.lstat(current)
        except FileNotFoundError:
            break
        except OSError as error:
            raise H10ReceiptShadowError(f"cannot inspect {label}: {current}") from error
        if stat.S_ISLNK(metadata.st_mode):
            raise H10ReceiptShadowError(f"{label} contains symlink component: {current}")


def _read_regular_bytes(
    path: Path, *, max_bytes: int, label: str
) -> tuple[bytes, dict[str, int | str]]:
    absolute = Path(os.path.abspath(os.fspath(path)))
    _assert_no_symlink_ancestors(absolute, label)
    try:
        before = os.lstat(absolute)
    except OSError as error:
        raise H10ReceiptShadowError(f"cannot stat {label}: {absolute}") from error
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
        raise H10ReceiptShadowError(f"{label} must be a non-symlink regular file")
    if before.st_size > max_bytes:
        raise H10ReceiptShadowError(f"{label} exceeds byte cap")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(absolute, flags)
    try:
        opened = os.fstat(descriptor)
        if (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino):
            raise H10ReceiptShadowError(f"{label} identity changed while opening")
        chunks: list[bytes] = []
        remaining = max_bytes + 1
        while remaining:
            chunk = os.read(descriptor, min(1024 * 1024, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        payload = b"".join(chunks)
        after = os.fstat(descriptor)
        if (
            len(payload) > max_bytes
            or len(payload) != opened.st_size
            or (after.st_size, after.st_mtime_ns) != (opened.st_size, opened.st_mtime_ns)
        ):
            raise H10ReceiptShadowError(f"{label} changed while reading")
    finally:
        os.close(descriptor)
    return payload, {
        "path": os.fspath(absolute),
        "sha256": _hash_bytes(payload),
        "bytes": len(payload),
        "device": int(opened.st_dev),
        "inode": int(opened.st_ino),
    }


def _rehash_identity(path: Path, expected: Mapping[str, object], *, max_bytes: int, label: str) -> dict[str, int | str]:
    _, current = _read_regular_bytes(path, max_bytes=max_bytes, label=label)
    if current != dict(expected):
        raise H10ReceiptShadowError(f"{label} changed during receipt replay")
    return current


def _array_content_sha256(arrays: Mapping[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for name in sorted(arrays):
        value = np.ascontiguousarray(arrays[name])
        digest.update(name.encode("ascii"))
        digest.update(b"\0")
        digest.update(value.dtype.str.encode("ascii"))
        digest.update(b"\0")
        digest.update(json.dumps(list(value.shape), separators=(",", ":")).encode("ascii"))
        digest.update(b"\0")
        digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def _numeric_matrix_sha256(name: str, value: np.ndarray) -> str:
    return _array_content_sha256({name: np.ascontiguousarray(value)})


def _deterministic_npz_bytes(arrays: Mapping[str, np.ndarray]) -> bytes:
    output = io.BytesIO()
    with zipfile.ZipFile(output, mode="w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name in sorted(arrays):
            payload = io.BytesIO()
            np.lib.format.write_array(payload, np.ascontiguousarray(arrays[name]), allow_pickle=False)
            info = zipfile.ZipInfo(f"{name}.npy", (1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o600 << 16
            archive.writestr(info, payload.getvalue(), compresslevel=9)
    return output.getvalue()


def _canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(value, sort_keys=True, indent=2, ensure_ascii=True, allow_nan=False)
        + "\n"
    ).encode("ascii")


def _strict_int(value: object, label: str, *, minimum: int = 0, maximum: int = (1 << 63) - 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise H10ReceiptShadowError(f"{label} must be an integer")
    result = int(value)
    if result < minimum or result > maximum:
        raise H10ReceiptShadowError(f"{label} is out of range")
    return result


def _strict_int_matrix(value: object, label: str, columns: int) -> np.ndarray:
    if not isinstance(value, list):
        raise H10ReceiptShadowError(f"{label} must be a list")
    rows: list[list[int]] = []
    for row_index, row in enumerate(value):
        if not isinstance(row, list) or len(row) != columns:
            raise H10ReceiptShadowError(f"{label}[{row_index}] has invalid width")
        rows.append([_strict_int(cell, f"{label}[{row_index}]") for cell in row])
    return np.asarray(rows, dtype=np.int64).reshape((-1, columns))


def _load_npz_bytes(payload: bytes) -> dict[str, np.ndarray]:
    try:
        with zipfile.ZipFile(io.BytesIO(payload), "r") as archive:
            infos = archive.infolist()
            expected_members = {f"{name}.npy" for name in SOURCE_ARRAYS}
            if len(infos) != len(expected_members) or {info.filename for info in infos} != expected_members:
                raise H10ReceiptShadowError("source NPZ member set differs from numeric schema")
            if any(info.is_dir() or info.file_size > MAX_SOURCE_NPZ_BYTES for info in infos):
                raise H10ReceiptShadowError("source NPZ contains an invalid member")
            if sum(info.file_size for info in infos) > MAX_SOURCE_NPZ_BYTES:
                raise H10ReceiptShadowError("source NPZ uncompressed payload exceeds cap")
        with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
            if set(archive.files) != SOURCE_ARRAYS:
                raise H10ReceiptShadowError("source NPZ array set differs")
            arrays = {name: np.array(archive[name], copy=True) for name in archive.files}
    except (OSError, ValueError, zipfile.BadZipFile) as error:
        if isinstance(error, H10ReceiptShadowError):
            raise
        raise H10ReceiptShadowError("source NPZ cannot be decoded as numeric arrays") from error
    return arrays


def _validate_numeric_arrays(
    manifest: Mapping[str, Any], arrays: Mapping[str, np.ndarray]
) -> tuple[tuple[str, ...], int]:
    scene_values = arrays["scene_ids"]
    if scene_values.ndim != 1 or scene_values.dtype.kind != "U":
        raise H10ReceiptShadowError("scene_ids must be a one-dimensional Unicode array")
    scenes = tuple(str(value) for value in scene_values.tolist())
    if (
        not scenes
        or len(scenes) > MAX_SCENES
        or len(set(scenes)) != len(scenes)
        or any(not scene or "/" in scene or "\\" in scene for scene in scenes)
    ):
        raise H10ReceiptShadowError("source scene_ids are invalid")
    if list(scenes) != manifest.get("scene_ids") or len(scenes) != manifest.get("scene_count"):
        raise H10ReceiptShadowError("source JSON/NPZ scene order differs")
    if len(scenes) != EXPECTED_SCENE_COUNT:
        raise H10ReceiptShadowError("source scene count differs from frozen H10")

    count = _strict_int(manifest.get("raw_row_count"), "raw_row_count", maximum=MAX_RAW_ROWS)
    shapes = {
        "per_view_scene_index": ((count,), "iu"),
        "per_view_frame_id": ((count,), "iu"),
        "per_view_source_row": ((count,), "iu"),
        "per_view_source_instance_id": ((count,), "iu"),
        "per_view_source_score": ((count,), "f"),
        "per_view_center_world": ((count, 3), "f"),
        "per_view_extent_xyz": ((count, 3), "f"),
        "per_view_quaternion_wxyz": ((count, 4), "f"),
    }
    for name, (shape, kinds) in shapes.items():
        value = arrays[name]
        if value.shape != shape or value.dtype.kind not in kinds or value.dtype.hasobject:
            raise H10ReceiptShadowError(f"numeric source schema mismatch for {name}")
    numeric = np.concatenate(
        [
            arrays["per_view_source_score"].reshape(-1),
            arrays["per_view_center_world"].reshape(-1),
            arrays["per_view_extent_xyz"].reshape(-1),
            arrays["per_view_quaternion_wxyz"].reshape(-1),
        ]
    )
    if not np.isfinite(numeric).all():
        raise H10ReceiptShadowError("numeric source contains non-finite values")
    if (
        np.any(arrays["per_view_source_score"] < 0.0)
        or np.any(arrays["per_view_source_score"] > 1.0)
        or np.any(arrays["per_view_extent_xyz"] <= 0.0)
    ):
        raise H10ReceiptShadowError("source score or extent is out of range")
    q2 = np.sum(np.asarray(arrays["per_view_quaternion_wxyz"], dtype=np.float64) ** 2, axis=1)
    if np.any(q2 <= 1e-12):
        raise H10ReceiptShadowError("source contains a degenerate quaternion")
    scene_index = arrays["per_view_scene_index"]
    if np.any((scene_index < 0) | (scene_index >= len(scenes))):
        raise H10ReceiptShadowError("source scene index is out of range")
    return scenes, count


def _validate_frame_ledger(
    manifest: Mapping[str, Any], arrays: Mapping[str, np.ndarray], scenes: Sequence[str]
) -> tuple[tuple[tuple[int, int, int], ...], tuple[tuple[int, ...], ...]]:
    ledger_matrix = _strict_int_matrix(manifest.get("frame_row_ledger"), "frame_row_ledger", 3)
    if len(ledger_matrix) != EXPECTED_EXACT_FRAME_COUNT or manifest.get("exact_frame_count") != len(ledger_matrix):
        raise H10ReceiptShadowError("frame ledger count differs from frozen H10")
    if manifest.get("raw_frame_count") != EXPECTED_RAW_FRAME_COUNT:
        raise H10ReceiptShadowError("raw frame count differs from frozen H10")
    frames_by_scene: list[list[int]] = [[] for _ in scenes]
    row_counts = [0] * len(scenes)
    empty: list[list[int]] = []
    cursor = 0
    last_scene = -1
    last_frame = -1
    for schedule_index, (scene_index64, frame_id64, count64) in enumerate(ledger_matrix):
        scene_index, frame_id, count = int(scene_index64), int(frame_id64), int(count64)
        if scene_index >= len(scenes) or count > SOURCE_INSTANCE_STRIDE:
            raise H10ReceiptShadowError("frame ledger scene/count is out of range")
        if scene_index < last_scene or scene_index > last_scene + 1:
            raise H10ReceiptShadowError("frame ledger scene groups are not contiguous")
        if scene_index != last_scene:
            last_scene, last_frame = scene_index, -1
        if frame_id <= last_frame:
            raise H10ReceiptShadowError("frame ledger IDs are not strictly increasing per scene")
        last_frame = frame_id
        frames_by_scene[scene_index].append(frame_id)
        stop = cursor + count
        if stop > len(arrays["per_view_scene_index"]):
            raise H10ReceiptShadowError("frame ledger exceeds source row count")
        positions = slice(cursor, stop)
        if count:
            if (
                not np.all(arrays["per_view_scene_index"][positions] == scene_index)
                or not np.all(arrays["per_view_frame_id"][positions] == frame_id)
                or not np.array_equal(
                    np.asarray(arrays["per_view_source_row"][positions], dtype=np.int64),
                    np.arange(count, dtype=np.int64),
                )
            ):
                raise H10ReceiptShadowError("frame ledger does not index exact numeric rows")
            expected_instance = schedule_index * SOURCE_INSTANCE_STRIDE + np.arange(count, dtype=np.int64)
            if not np.array_equal(
                np.asarray(arrays["per_view_source_instance_id"][positions], dtype=np.int64),
                expected_instance,
            ):
                raise H10ReceiptShadowError("source_instance_id rule differs from seal")
        else:
            empty.append([scene_index, frame_id])
        row_counts[scene_index] += count
        cursor = stop
    if cursor != len(arrays["per_view_scene_index"]):
        raise H10ReceiptShadowError("frame ledger does not consume all source rows")
    if any(not values or len(values) > MAX_VALID_FRAMES_PER_SCENE for values in frames_by_scene):
        raise H10ReceiptShadowError("one or more scene frame ledgers are empty/oversized")
    if manifest.get("scene_row_counts") != row_counts:
        raise H10ReceiptShadowError("scene row-count ledger differs")
    if manifest.get("empty_frame_identities") != empty or manifest.get("empty_frame_count") != len(empty):
        raise H10ReceiptShadowError("empty-frame ledger differs")
    return (
        tuple((int(a), int(b), int(c)) for a, b, c in ledger_matrix.tolist()),
        tuple(tuple(values) for values in frames_by_scene),
    )


def _verify_frozen_membership(
    manifest: Mapping[str, Any], arrays: Mapping[str, np.ndarray], ledger: Sequence[tuple[int, int, int]], scene_count: int
) -> tuple[np.ndarray, tuple[np.ndarray, ...], str]:
    k8 = manifest.get("k8")
    if not isinstance(k8, Mapping) or set(k8) != K8_KEYS:
        raise H10ReceiptShadowError("K8 seal schema differs")
    if (
        k8.get("top_k") != TOP_K
        or tuple(k8.get("sort_key", ())) != K8_SORT_KEY
        or tuple(k8.get("identity_columns", ())) != K8_COLUMNS
    ):
        raise H10ReceiptShadowError("K8 contract differs")
    supplied = _strict_int_matrix(k8.get("membership_identities"), "k8.membership_identities", 5)
    expected_rows: list[list[int]] = []
    per_scene = [0] * scene_count
    cursor = 0
    for scene_index, frame_id, count in ledger:
        ranked = sorted(
            range(cursor, cursor + count),
            key=lambda row: (
                -float(arrays["per_view_source_score"][row]),
                int(arrays["per_view_source_row"][row]),
                row,
            ),
        )[:TOP_K]
        for row in ranked:
            expected_rows.append(
                [
                    scene_index,
                    frame_id,
                    int(arrays["per_view_source_row"][row]),
                    int(arrays["per_view_source_instance_id"][row]),
                    row,
                ]
            )
        per_scene[scene_index] += len(ranked)
        cursor += count
    expected = np.asarray(expected_rows, dtype=np.int64).reshape((-1, 5))
    if not np.array_equal(supplied, expected):
        raise H10ReceiptShadowError("fixed K8 membership differs from independent verification")
    digest = _numeric_matrix_sha256("k8_membership_identity", supplied)
    if (
        k8.get("membership_count") != len(supplied)
        or k8.get("membership_per_scene") != per_scene
        or k8.get("membership_sha256") != digest
    ):
        raise H10ReceiptShadowError("K8 membership count/hash ledger differs")
    selections = tuple(
        np.asarray(supplied[supplied[:, 0] == scene_index, 4], dtype=np.int64)
        for scene_index in range(scene_count)
    )
    if any(len(value) > MAX_SELECTED_ROWS_PER_SCENE for value in selections):
        raise H10ReceiptShadowError("K8 per-scene row cap exceeded")
    supplied.setflags(write=False)
    for value in selections:
        value.setflags(write=False)
    return supplied, selections, digest


def _validate_source_manifest(
    manifest: Mapping[str, Any], arrays: Mapping[str, np.ndarray], *, npz_sha256: str
) -> tuple[tuple[str, ...], tuple[tuple[int, ...], ...], np.ndarray, tuple[np.ndarray, ...], str]:
    if set(manifest) != SOURCE_MANIFEST_KEYS:
        raise H10ReceiptShadowError("source manifest top-level schema differs")
    required = {
        "schema": EXPECTED_SOURCE_SCHEMA,
        "mode": "sealed_raw_observer_source",
        "create_only": True,
        "association_applied": False,
        "tracking_enabled": False,
        "tracked_artifact_present": False,
        "coordinate_frame": "scannet_world",
        "source_instance_id_rule": "global_exact_schedule_index*2048+per_frame_source_row",
        "npz_file": SOURCE_NPZ_NAME,
    }
    for key, expected in required.items():
        if manifest.get(key) != expected:
            raise H10ReceiptShadowError(f"source contract mismatch for {key}")
    _require_sha256(manifest.get("coordinate_contract_sha256"), "coordinate contract")
    bindings = manifest.get("provider_bindings")
    if (
        not isinstance(bindings, Mapping)
        or set(bindings) != PROVIDER_BINDING_KEYS
        or bindings.get("schedule_sha256") != EXPECTED_SCHEDULE_SHA256
    ):
        raise H10ReceiptShadowError("source does not bind frozen H10 schedule")
    for name in (
        "run_provenance_sha256",
        "final_seal_sha256",
        "journal_sha256",
        "provider_contract_sha256",
        "frozen_assets_sha256",
        "exact_input_ledger_sha256",
    ):
        _require_sha256(bindings.get(name), f"provider_bindings.{name}")
    for name in ("code_hashes", "model_hashes", "protocol_hashes"):
        ledger = bindings.get(name)
        if not isinstance(ledger, Mapping) or not ledger:
            raise H10ReceiptShadowError(f"provider_bindings.{name} is invalid")
        for key, digest in ledger.items():
            if not isinstance(key, str) or not key:
                raise H10ReceiptShadowError(f"provider_bindings.{name} key is invalid")
            _require_sha256(digest, f"provider_bindings.{name}.{key}")
    identity = manifest.get("input_identity")
    if (
        not isinstance(identity, Mapping)
        or identity.get("byte_identical") is not True
        or identity.get("snapshot_sha256_before") != identity.get("snapshot_sha256_after")
        or not _is_sha256(identity.get("snapshot_sha256_before"))
    ):
        raise H10ReceiptShadowError("source input identity is incomplete")
    if manifest.get("npz_sha256") != npz_sha256:
        raise H10ReceiptShadowError("source manifest does not bind supplied NPZ bytes")
    if manifest.get("array_names") != sorted(SOURCE_ARRAYS):
        raise H10ReceiptShadowError("source manifest array-name ledger differs")
    content_hash = _array_content_sha256(arrays)
    if (
        manifest.get("array_content_sha256") != content_hash
        or content_hash != EXPECTED_SOURCE_ARRAY_CONTENT_SHA256
    ):
        raise H10ReceiptShadowError("source manifest numeric content hash differs")
    scenes, _ = _validate_numeric_arrays(manifest, arrays)
    ledger, frames_by_scene = _validate_frame_ledger(manifest, arrays, scenes)
    membership, selections, membership_hash = _verify_frozen_membership(
        manifest, arrays, ledger, len(scenes)
    )
    if (
        len(membership) != EXPECTED_K8_MEMBERSHIP_COUNT
        or membership_hash != EXPECTED_K8_MEMBERSHIP_SHA256
    ):
        raise H10ReceiptShadowError("K8 membership differs from frozen H10 receipt universe")
    return scenes, frames_by_scene, membership, selections, membership_hash


def _quaternion_rotation(quaternion_wxyz: np.ndarray) -> np.ndarray:
    quaternion = np.asarray(quaternion_wxyz, dtype=np.float64)
    if quaternion.shape != (4,) or not np.isfinite(quaternion).all():
        raise H10ReceiptShadowError("quaternion_wxyz must be finite with shape (4,)")
    norm_squared = float(quaternion @ quaternion)
    if not math.isfinite(norm_squared) or norm_squared <= 1e-12:
        raise H10ReceiptShadowError("quaternion_wxyz has invalid squared norm")
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
        raise H10ReceiptShadowError("raw OBB center/extent is invalid")
    rotation = _quaternion_rotation(quaternion_wxyz)
    return np.ascontiguousarray(SIGNS * (extent / 2.0) @ rotation.T + center)


def _aabb_pair_metrics(left: np.ndarray, right: np.ndarray) -> tuple[float, float]:
    left_lower = left.min(axis=0)
    left_upper = left.max(axis=0)
    right_lower = right.min(axis=0)
    right_upper = right.max(axis=0)
    intersection = np.prod(
        np.maximum(
            np.minimum(left_upper, right_upper)
            - np.maximum(left_lower, right_lower),
            0.0,
        )
    )
    left_volume = np.prod(left_upper - left_lower)
    right_volume = np.prod(right_upper - right_lower)
    union = left_volume + right_volume - intersection
    iou = 0.0 if union <= 0.0 else float(intersection / union)
    center_distance = float(
        np.linalg.norm(
            0.5 * (left_lower + left_upper)
            - 0.5 * (right_lower + right_upper)
        )
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
        raise H10ReceiptShadowError(
            "numeric thread environment must be pinned exactly to one thread: "
            f"expected={dict(REQUIRED_NUMERIC_THREAD_ENVIRONMENT)}, actual={actual}"
        )
    return actual


def _validate_cpu_only_implementation() -> dict[str, Any]:
    """Prove the tracker/adapter import graph has no GPU runtime surface.

    This is a static implementation audit, not a fabricated device-memory
    measurement.  The tracker and this adapter use NumPy on CPU and never
    import or call a CUDA-capable framework.
    """

    forbidden_roots = frozenset(
        {"torch", "cupy", "jax", "tensorflow", "numba", "pycuda"}
    )
    audited: dict[str, str] = {}
    for label, path in (
        ("runner", Path(__file__).resolve()),
        ("tracker", TRACKER_SOURCE),
    ):
        payload, identity = _read_regular_bytes(
            path, max_bytes=MAX_CODE_BYTES, label=f"CPU-only {label} source"
        )
        try:
            tree = ast.parse(payload.decode("utf-8"), filename=os.fspath(path))
        except (UnicodeDecodeError, SyntaxError) as error:
            raise H10ReceiptShadowError(
                f"CPU-only {label} source cannot be statically audited"
            ) from error
        imported_roots: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported_roots.update(alias.name.split(".", 1)[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported_roots.add(node.module.split(".", 1)[0])
        forbidden = sorted(imported_roots & forbidden_roots)
        if forbidden:
            raise H10ReceiptShadowError(
                f"CPU-only {label} imports forbidden GPU surface: {forbidden}"
            )
        audited[label] = str(identity["sha256"])
    return {
        "audit_method": "static_AST_import_audit",
        "forbidden_import_roots": sorted(forbidden_roots),
        "audited_source_sha256": audited,
        "tracker_execution_device": "cpu",
        "tracker_gpu_execution": False,
        "tracker_cuda_api_access": False,
        "tracker_gpu_allocation_bytes_by_construction": 0,
        "gpu_memory_measurement_claimed": False,
    }


def _percentile_ns(values: Sequence[int], percentile: float) -> int:
    if not values:
        return 0
    return int(
        math.ceil(
            float(np.percentile(np.asarray(values, dtype=np.int64), percentile))
        )
    )


def _runtime_stats(values: Sequence[int]) -> dict[str, int]:
    return {
        "count": len(values),
        "p50_ns": _percentile_ns(values, 50.0),
        "p95_ns": _percentile_ns(values, 95.0),
        "max_ns": max(values, default=0),
    }


def _empty_array(
    values: Sequence[Any], dtype: object, tail: tuple[int, ...] = ()
) -> np.ndarray:
    return np.asarray(values, dtype=dtype).reshape((len(values),) + tail)


def _run_tracking(
    *,
    scene_ids: Sequence[str],
    frames_by_scene: Sequence[Sequence[int]],
    source_arrays: Mapping[str, np.ndarray],
    selections: Sequence[np.ndarray],
    tracker_factory: Callable[[], S3RReceiptTracker] = S3RReceiptTracker,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Replay exact membership, querying committed history before each commit."""

    numeric_thread_environment = _validate_numeric_thread_environment()
    cpu_only_audit = _validate_cpu_only_implementation()
    if len(scene_ids) != len(frames_by_scene) or len(scene_ids) != len(selections):
        raise H10ReceiptShadowError("scene/frame/selection ledgers differ in length")

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

    source_frames = source_arrays["per_view_frame_id"]
    source_scenes = source_arrays["per_view_scene_index"]
    for scene_index, scene in enumerate(scene_ids):
        selection = np.asarray(selections[scene_index], dtype=np.int64)
        valid_frames = tuple(int(value) for value in frames_by_scene[scene_index])
        if len(selection) > MAX_SELECTED_ROWS_PER_SCENE:
            raise H10ReceiptShadowError(f"selected-row safety cap exceeded for {scene}")
        if not valid_frames or len(valid_frames) > MAX_VALID_FRAMES_PER_SCENE:
            raise H10ReceiptShadowError(f"valid-frame safety cap exceeded for {scene}")
        if tuple(sorted(set(valid_frames))) != valid_frames:
            raise H10ReceiptShadowError(f"frame order is invalid for {scene}")
        by_frame: dict[int, list[int]] = {frame: [] for frame in valid_frames}
        for npz_row in selection.tolist():
            if npz_row < 0 or npz_row >= len(source_frames):
                raise H10ReceiptShadowError("selected sealed row is out of range")
            frame_id = int(source_frames[npz_row])
            if int(source_scenes[npz_row]) != scene_index or frame_id not in by_frame:
                raise H10ReceiptShadowError(f"selected row is off scene/schedule for {scene}")
            by_frame[frame_id].append(int(npz_row))
        if any(len(rows) > TOP_K for rows in by_frame.values()):
            raise H10ReceiptShadowError(f"more than K8 rows occur in one frame for {scene}")

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
            for npz_row in rows:
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
                raise H10ReceiptShadowError(
                    f"tracker changed exact K8 order for {scene}/{frame_id}"
                )
            cap_count = (
                len(query.observation_capacity_dropped_source_rows)
                + len(query.track_capacity_dropped_source_rows)
                + len(query.receipt_capacity_dropped_track_ids)
            )
            if cap_count or not query.audit_complete or not commit.audit_complete:
                raise H10ReceiptShadowError(
                    f"capacity event invalidated audit for {scene}/{frame_id}"
                )
            checks = len(rows) * len(query.prior_track_ids)
            if checks > MAX_ELIGIBILITY_CHECKS_PER_FRAME:
                raise H10ReceiptShadowError(
                    f"eligibility-check cap exceeded for {scene}/{frame_id}"
                )

            for track_id in query.newly_retired_track_ids:
                if track_id not in anchors:
                    raise H10ReceiptShadowError("retired track is absent from prior anchor ledger")
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
                    raise H10ReceiptShadowError("assignment provenance differs from K8 row")
                if assignment.action == "matched":
                    if assignment.track_id not in anchors:
                        raise H10ReceiptShadowError("matched track has no prior committed anchor")
                    iou, center_m = _aabb_pair_metrics(corners, anchors[assignment.track_id])
                    if iou < 0.10 or center_m > 0.50:
                        raise H10ReceiptShadowError("matched metrics violate tracker gates")
                    action_code = 1
                elif assignment.action == "created":
                    if assignment.track_id in anchors:
                        raise H10ReceiptShadowError("created track already exists in prior ledger")
                    iou, center_m = -1.0, -1.0
                    action_code = 0
                else:
                    raise H10ReceiptShadowError("unknown tracker assignment action")

                selected_scene_index.append(scene_index)
                selected_schedule_index.append(schedule_index)
                selected_frame_id.append(frame_id)
                selected_rank.append(rank)
                selected_npz_row.append(npz_row)
                selected_source_row.append(assignment.source_row)
                selected_instance_id.append(assignment.source_instance_id)
                selected_score.append(float(source_arrays["per_view_source_score"][npz_row]))
                selected_center.append(np.asarray(source_arrays["per_view_center_world"][npz_row], dtype=np.float64))
                selected_extent.append(np.asarray(source_arrays["per_view_extent_xyz"][npz_row], dtype=np.float64))
                selected_quaternion.append(np.asarray(source_arrays["per_view_quaternion_wxyz"][npz_row], dtype=np.float64))
                selected_corners.append(corners)
                assignment_track_id.append(assignment.track_id)
                assignment_action.append(action_code)
                assignment_iou.append(iou)
                assignment_center_m.append(center_m)
                anchors[assignment.track_id] = corners

            if set(anchors) != set(commit.active_track_ids):
                raise H10ReceiptShadowError("committed anchor ledger differs from tracker")
            for receipt in commit.newly_frozen_receipts:
                evidence_frames = tuple(int(value) for value in receipt.evidence_frame_ids)
                if (
                    len(evidence_frames) != 3
                    or len(set(evidence_frames)) != 3
                    or evidence_frames != tuple(sorted(evidence_frames))
                    or evidence_frames[-1] != frame_id
                    or receipt.confirmation_frame_id != frame_id
                ):
                    raise H10ReceiptShadowError("receipt is not supported by three distinct causal frames")
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
            raise H10ReceiptShadowError(f"incomplete tracker summary for {scene}")
        final_receipts = tracker.receipts()
        chronological = [receipt for index, receipt in receipts if index == scene_index]
        if (
            len({receipt.track_id for receipt in chronological}) != len(chronological)
            or {receipt.track_id: receipt for receipt in chronological}
            != {receipt.track_id: receipt for receipt in final_receipts}
        ):
            raise H10ReceiptShadowError(f"incremental/final receipt mismatch for {scene}")
        del receipts[scene_receipt_start:]
        receipts.extend((scene_index, receipt) for receipt in final_receipts)
        tracker_summaries[scene] = tracker_summary
        empty_frames = [frame for frame in valid_frames if not by_frame[frame]]
        scene_summaries[scene] = {
            "valid_frame_count": len(valid_frames),
            "candidate_frame_count": len(valid_frames) - len(empty_frames),
            "empty_valid_frame_ids": empty_frames,
            "selected_row_count": len(selection),
            "created_assignment_count": scene_created,
            "matched_assignment_count": scene_matched,
            "retired_track_count": scene_retired,
            "receipt_count": len(final_receipts),
        }
        if len(selected_npz_row) - scene_selected_start != len(selection):
            raise H10ReceiptShadowError(f"selected-row export is incomplete for {scene}")
        scene_schedule_offsets.append(len(schedule_frame_id))
        scene_selected_offsets.append(len(selected_npz_row))
        scene_receipt_offsets.append(len(receipts))

    expected_flat_selection = (
        np.concatenate(selections).astype(np.int64, copy=False)
        if selections
        else np.empty((0,), dtype=np.int64)
    )
    if not np.array_equal(np.asarray(selected_npz_row, dtype=np.int64), expected_flat_selection):
        raise H10ReceiptShadowError("exported membership differs from frozen K8 order")

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
        frames = tuple(int(value) for value in receipt.evidence_frame_ids)
        if (
            len(frames) != 3
            or len(set(frames)) != 3
            or frames != tuple(sorted(frames))
            or frames[-1] != receipt.confirmation_frame_id
        ):
            raise H10ReceiptShadowError("exported receipt violates three-frame causality")
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
                raise H10ReceiptShadowError("receipt evidence is absent from K8 membership")
            evidence_selected_index.append(selected_index_by_scene_row[key])
            evidence_frame_id.append(receipt.evidence_frame_ids[index])
            evidence_source_row.append(receipt.evidence_source_rows[index])
            evidence_npz_row.append(npz_row)
            evidence_instance_id.append(receipt.evidence_source_instance_ids[index])
            evidence_score.append(receipt.evidence_scores[index])
            evidence_corners.append(receipt.evidence_corners[index])
        evidence_offsets.append(len(evidence_npz_row))

    unicode_width = max(1, max(len(scene) for scene in scene_ids))
    arrays = {
        "scene_ids": np.asarray(scene_ids, dtype=f"<U{unicode_width}"),
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
        "receipt_confirmation_frame_id": np.asarray(receipt_confirmation_frame_id, dtype=np.int64),
        "receipt_corners_world": _empty_array(receipt_corners, np.float64, (8, 3)),
        "receipt_medoid_evidence_index": np.asarray(receipt_medoid_index, dtype=np.int8),
        "receipt_pairwise_aabb_iou": _empty_array(receipt_pairwise_iou, np.float64, (3, 3)),
        "receipt_pairwise_center_distance_m": _empty_array(receipt_pairwise_center, np.float64, (3, 3)),
        "receipt_raw_mean_score": np.asarray(receipt_mean_score, dtype=np.float64),
        "receipt_median_pairwise_aabb_iou": np.asarray(receipt_median_iou, dtype=np.float64),
        "receipt_center_rms_m": np.asarray(receipt_center_rms, dtype=np.float64),
        "receipt_min_medoid_aabb_extent_m": np.asarray(receipt_min_extent, dtype=np.float64),
        "evidence_offsets": np.asarray(evidence_offsets, dtype=np.int64),
        "evidence_selected_index": np.asarray(evidence_selected_index, dtype=np.int64),
        "evidence_frame_id": np.asarray(evidence_frame_id, dtype=np.int64),
        "evidence_source_row": np.asarray(evidence_source_row, dtype=np.int64),
        "evidence_sealed_npz_row": np.asarray(evidence_npz_row, dtype=np.int64),
        "evidence_source_instance_id": np.asarray(evidence_instance_id, dtype=np.int64),
        "evidence_source_score": np.asarray(evidence_score, dtype=np.float64),
        "evidence_corners_world": _empty_array(evidence_corners, np.float64, (8, 3)),
    }
    trace_bytes = int(sum(value.nbytes for value in arrays.values()))
    if trace_bytes > MAX_TRACE_UNCOMPRESSED_BYTES:
        raise H10ReceiptShadowError("uncompressed diagnostic trace cap exceeded")
    rss_end = _rss_bytes()
    rss_measurement_complete = rss_start >= 0 and rss_peak >= 0 and rss_end >= 0
    rss_increment = max(0, int(rss_peak - rss_start)) if rss_measurement_complete else -1
    tracker_cpu_stats = _runtime_stats(frame_tracker_cpu_ns)
    tracker_cpu_budget_pass = (
        tracker_cpu_stats["p95_ns"] <= TRACKER_CPU_P95_LIMIT_NS
        and tracker_cpu_stats["max_ns"] <= TRACKER_CPU_MAX_LIMIT_NS
    )
    tracker_memory_upper_bound_pass = (
        rss_measurement_complete
        and rss_increment <= MAX_TRACKER_INCREMENTAL_MEMORY_BYTES
    )
    resource_budget_pass = tracker_cpu_budget_pass and tracker_memory_upper_bound_pass
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
        "cpu_only_implementation": cpu_only_audit,
        "tracker_execution_device": "cpu",
        "tracker_gpu_execution": False,
        "tracker_cuda_api_access": False,
        "tracker_gpu_allocation_bytes": 0,
        "tracker_gpu_allocation_semantics": (
            "by_construction_from_static_no_GPU_runtime_import_or_API_audit"
        ),
        "gpu_memory_measurement_claimed": False,
        "sealed_numeric_replay": True,
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


def _validate_fixed_assets() -> dict[str, dict[str, int | str]]:
    expected = {
        "source_sealer": (SOURCE_SEALER, EXPECTED_SOURCE_SEALER_SHA256),
        "source_sealer_test": (
            SOURCE_SEALER_TEST,
            EXPECTED_SOURCE_SEALER_TEST_SHA256,
        ),
        "tracker_source": (TRACKER_SOURCE, EXPECTED_TRACKER_SHA256),
        "tracker_test": (TRACKER_TEST, EXPECTED_TRACKER_TEST_SHA256),
    }
    result: dict[str, dict[str, int | str]] = {}
    for name, (path, expected_hash) in expected.items():
        expected_digest = _require_sha256(expected_hash, f"frozen {name} hash")
        _, identity = _read_regular_bytes(
            path, max_bytes=MAX_CODE_BYTES, label=f"frozen {name}"
        )
        if identity["sha256"] != expected_digest:
            raise H10ReceiptShadowError(
                f"frozen {name} SHA-256 mismatch: "
                f"expected={expected_digest}, actual={identity['sha256']}"
            )
        result[name] = identity
    return result


def _source_root_identity(source_root: Path) -> tuple[Path, dict[str, int | str]]:
    root = Path(os.path.abspath(os.fspath(source_root)))
    _assert_no_symlink_ancestors(root, "sealed source root")
    try:
        metadata = os.lstat(root)
    except OSError as error:
        raise H10ReceiptShadowError(f"sealed source root is unavailable: {root}") from error
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        raise H10ReceiptShadowError("sealed source root must be a non-symlink directory")
    return root, {
        "path": os.fspath(root),
        "device": int(metadata.st_dev),
        "inode": int(metadata.st_ino),
    }


def _verify_source_root_identity(
    source_root: Path, expected: Mapping[str, object]
) -> None:
    root, actual = _source_root_identity(source_root)
    if actual != dict(expected):
        raise H10ReceiptShadowError(f"sealed source root identity changed: {root}")


def _load_sealed_source(
    source_root: Path,
) -> tuple[
    dict[str, Any],
    dict[str, np.ndarray],
    tuple[str, ...],
    tuple[tuple[int, ...], ...],
    np.ndarray,
    tuple[np.ndarray, ...],
    str,
    dict[str, Any],
]:
    root, root_identity = _source_root_identity(source_root)
    json_payload, json_identity = _read_regular_bytes(
        root / SOURCE_JSON_NAME,
        max_bytes=MAX_SOURCE_JSON_BYTES,
        label="sealed H10 source JSON",
    )
    npz_payload, npz_identity = _read_regular_bytes(
        root / SOURCE_NPZ_NAME,
        max_bytes=MAX_SOURCE_NPZ_BYTES,
        label="sealed H10 source NPZ",
    )
    if json_identity["sha256"] != EXPECTED_SOURCE_JSON_SHA256:
        raise H10ReceiptShadowError("sealed H10 source JSON differs from frozen bytes")
    if npz_identity["sha256"] != EXPECTED_SOURCE_NPZ_SHA256:
        raise H10ReceiptShadowError("sealed H10 source NPZ differs from frozen bytes")
    manifest = _parse_json_bytes(json_payload, "sealed H10 source JSON")
    arrays = _load_npz_bytes(npz_payload)
    scenes, frames_by_scene, membership, selections, membership_hash = (
        _validate_source_manifest(
            manifest, arrays, npz_sha256=str(npz_identity["sha256"])
        )
    )
    for value in arrays.values():
        value.setflags(write=False)
    snapshot = {
        "root": root_identity,
        "json": json_identity,
        "npz": npz_identity,
    }
    return (
        manifest,
        arrays,
        scenes,
        frames_by_scene,
        membership,
        selections,
        membership_hash,
        snapshot,
    )


def _load_contract(
    path: Path, expected_sha256: str
) -> tuple[Path, dict[str, int | str]]:
    expected = _require_sha256(expected_sha256, "expected receipt contract hash")
    absolute = Path(os.path.abspath(os.fspath(path)))
    _, identity = _read_regular_bytes(
        absolute, max_bytes=MAX_CONTRACT_BYTES, label="receipt contract"
    )
    if identity["sha256"] != expected:
        raise H10ReceiptShadowError("receipt contract SHA-256 mismatch")
    # The contract is deliberately opaque here.  Parsing it would add a
    # second policy interpretation surface and could create a circular bind.
    return absolute, identity


def _validate_runner_bindings(
    expected_runner_sha256: str, expected_runner_test_sha256: str
) -> dict[str, dict[str, int | str]]:
    runner_expected = _require_sha256(
        expected_runner_sha256, "expected H10 receipt runner hash"
    )
    runner_test_expected = _require_sha256(
        expected_runner_test_sha256, "expected H10 receipt runner test hash"
    )
    _, runner = _read_regular_bytes(
        Path(__file__).resolve(),
        max_bytes=MAX_CODE_BYTES,
        label="H10 receipt runner",
    )
    _, runner_test = _read_regular_bytes(
        RUNNER_TEST, max_bytes=MAX_CODE_BYTES, label="H10 receipt runner tests"
    )
    if runner["sha256"] != runner_expected:
        raise H10ReceiptShadowError("H10 receipt runner SHA-256 mismatch")
    if runner_test["sha256"] != runner_test_expected:
        raise H10ReceiptShadowError("H10 receipt runner test SHA-256 mismatch")
    return {"runner_source": runner, "runner_test": runner_test}


def _snapshot_inputs(
    *,
    source_root: Path,
    source_snapshot: Mapping[str, Any],
    contract_path: Path,
    contract_snapshot: Mapping[str, object],
    fixed_assets: Mapping[str, Mapping[str, object]],
    source_arrays: Mapping[str, np.ndarray],
    expected_runner_sha256: str,
    expected_runner_test_sha256: str,
) -> dict[str, Any]:
    _verify_source_root_identity(source_root, source_snapshot["root"])
    source_json = _rehash_identity(
        source_root / SOURCE_JSON_NAME,
        source_snapshot["json"],
        max_bytes=MAX_SOURCE_JSON_BYTES,
        label="sealed H10 source JSON",
    )
    source_npz = _rehash_identity(
        source_root / SOURCE_NPZ_NAME,
        source_snapshot["npz"],
        max_bytes=MAX_SOURCE_NPZ_BYTES,
        label="sealed H10 source NPZ",
    )
    contract = _rehash_identity(
        contract_path,
        contract_snapshot,
        max_bytes=MAX_CONTRACT_BYTES,
        label="receipt contract",
    )
    fixed_after = _validate_fixed_assets()
    if fixed_after != {name: dict(value) for name, value in fixed_assets.items()}:
        raise H10ReceiptShadowError("fixed implementation assets changed during replay")
    runner_bindings = _validate_runner_bindings(
        expected_runner_sha256, expected_runner_test_sha256
    )
    return {
        "source_root": dict(source_snapshot["root"]),
        "source_json": source_json,
        "source_npz": source_npz,
        "receipt_contract": contract,
        "fixed_assets": fixed_after,
        **runner_bindings,
        "loaded_numeric_array_content_sha256": _array_content_sha256(source_arrays),
    }


def _build_manifest(
    *,
    source_manifest: Mapping[str, Any],
    source_arrays: Mapping[str, np.ndarray],
    trace_arrays: Mapping[str, np.ndarray],
    summary: Mapping[str, Any],
    scenes: Sequence[str],
    membership: np.ndarray,
    membership_sha256: str,
    receipt_contract_sha256: str,
    expected_runner_sha256: str,
    expected_runner_test_sha256: str,
    input_before: Mapping[str, Any],
    input_after: Mapping[str, Any],
    npz_sha256: str,
) -> dict[str, Any]:
    if input_before != input_after:
        raise H10ReceiptShadowError("one or more frozen inputs changed during replay")
    if summary.get("audit_complete") is not True or summary.get("cap_event_count") != 0:
        raise H10ReceiptShadowError("incomplete/capped trace cannot receive a valid seal")
    runtime = summary.get("runtime")
    cpu_only = runtime.get("cpu_only_implementation") if isinstance(runtime, Mapping) else None
    if (
        not isinstance(runtime, Mapping)
        or runtime.get("tracker_cpu_budget_pass") is not True
        or runtime.get("tracker_memory_upper_bound_pass") is not True
        or runtime.get("resource_budget_pass") is not True
        or runtime.get("numeric_thread_environment_pinned") is not True
        or runtime.get("numeric_thread_environment")
        != dict(REQUIRED_NUMERIC_THREAD_ENVIRONMENT)
        or runtime.get("tracker_execution_device") != "cpu"
        or runtime.get("tracker_gpu_execution") is not False
        or runtime.get("tracker_cuda_api_access") is not False
        or runtime.get("tracker_gpu_allocation_bytes") != 0
        or runtime.get("gpu_memory_measurement_claimed") is not False
        or not isinstance(cpu_only, Mapping)
        or cpu_only.get("audit_method") != "static_AST_import_audit"
        or cpu_only.get("tracker_gpu_execution") is not False
        or cpu_only.get("tracker_cuda_api_access") is not False
        or cpu_only.get("tracker_gpu_allocation_bytes_by_construction") != 0
        or cpu_only.get("audited_source_sha256")
        != {
            "runner": expected_runner_sha256,
            "tracker": EXPECTED_TRACKER_SHA256,
        }
    ):
        raise H10ReceiptShadowError("resource-budget failure cannot receive a valid seal")
    source_bindings = source_manifest["provider_bindings"]
    return {
        "schema": SCHEMA,
        "mode": "shadow",
        "create_only": True,
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
        "rgb_access": False,
        "depth_access": False,
        "native_prediction_access": False,
        "native_prediction_deserialized": False,
        "native_prediction_hash_access": False,
        "pickle_deserialization": False,
        "training": False,
        "online_learning": False,
        "optimizer_access": False,
        "past_only": True,
        "query_before_commit": True,
        "same_frame_confirmation": False,
        "within_frame_deduplication": False,
        "three_distinct_frame_receipts": True,
        "birth_not_authorized": True,
        "H10_shadow_complete": True,
        "H10_oracle_not_run": True,
        "H10_oracle_authorized": False,
        "gt_access_authorized": False,
        "H10_gate_not_evaluated": True,
        "C87_not_authorized": True,
        "full100_not_authorized": True,
        "not_deployable": True,
        "audit_complete": True,
        "scene_order": list(scenes),
        "scene_count": len(scenes),
        "valid_frame_count": int(summary["valid_frame_count"]),
        "selected_row_count": int(summary["selected_row_count"]),
        "assignment_count": int(summary["assignment_count"]),
        "receipt_count": int(summary["receipt_count"]),
        "evidence_count": int(summary["evidence_count"]),
        "cap_event_count": 0,
        "trace_uncompressed_bytes": int(summary["trace_uncompressed_bytes"]),
        "trace_uncompressed_cap_bytes": MAX_TRACE_UNCOMPRESSED_BYTES,
        "source": {
            "schema": EXPECTED_SOURCE_SCHEMA,
            "source_json_sha256": EXPECTED_SOURCE_JSON_SHA256,
            "source_npz_sha256": EXPECTED_SOURCE_NPZ_SHA256,
            "source_array_content_sha256": EXPECTED_SOURCE_ARRAY_CONTENT_SHA256,
            "schedule_sha256": source_bindings["schedule_sha256"],
            "coordinate_contract_sha256": source_manifest[
                "coordinate_contract_sha256"
            ],
            "provider_bindings": source_bindings,
            "raw_row_count": int(source_manifest["raw_row_count"]),
            "exact_frame_count": int(source_manifest["exact_frame_count"]),
            "empty_frame_count": int(source_manifest["empty_frame_count"]),
            "only_sealed_numeric_source_consumed": True,
        },
        "selection": {
            "top_k": TOP_K,
            "membership_count": len(membership),
            "membership_sha256": membership_sha256,
            "expected_membership_count": EXPECTED_K8_MEMBERSHIP_COUNT,
            "expected_membership_sha256": EXPECTED_K8_MEMBERSHIP_SHA256,
            "membership_consumed_not_reselected": True,
            "independent_membership_verification": True,
            "identity_columns": list(K8_COLUMNS),
            "sort_key": list(K8_SORT_KEY),
            "selected_npz_rows_sha256": _numeric_matrix_sha256(
                "selected_sealed_npz_row",
                np.asarray(trace_arrays["selected_sealed_npz_row"], dtype=np.int64),
            ),
        },
        "contracts": {
            "receipt_contract_sha256": receipt_contract_sha256,
            "runner_source_sha256": expected_runner_sha256,
            "runner_test_sha256": expected_runner_test_sha256,
            "source_sealer_sha256": EXPECTED_SOURCE_SEALER_SHA256,
            "source_sealer_test_sha256": EXPECTED_SOURCE_SEALER_TEST_SHA256,
            "tracker_source_sha256": EXPECTED_TRACKER_SHA256,
            "tracker_test_sha256": EXPECTED_TRACKER_TEST_SHA256,
        },
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
            "max_live_tracks_per_scene": 1024,
            "max_receipts_per_scene": 1024,
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
        "trace_array_content_sha256": _array_content_sha256(trace_arrays),
        "conclusion_guardrail": (
            "Receipt matching is a no-GT shadow trace, not AP, precision, birth, "
            "or integrated realtime evidence."
        ),
    }


def _write_exclusive_fsync_at(
    directory_fd: int, name: str, payload: bytes
) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    flags |= getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(name, flags, 0o600, dir_fd=directory_fd)
    try:
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("short create-only write")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _read_regular_bytes_at(
    directory_fd: int, name: str, *, max_bytes: int, label: str
) -> bytes:
    try:
        before = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
    except OSError as error:
        raise H10ReceiptShadowError(f"cannot stat {label}") from error
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
        raise H10ReceiptShadowError(f"{label} must be a non-symlink regular file")
    if before.st_size > max_bytes:
        raise H10ReceiptShadowError(f"{label} exceeds byte cap")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(name, flags, dir_fd=directory_fd)
    try:
        opened = os.fstat(descriptor)
        if (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino):
            raise H10ReceiptShadowError(f"{label} identity changed")
        chunks: list[bytes] = []
        remaining = max_bytes + 1
        while remaining:
            chunk = os.read(descriptor, min(1024 * 1024, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        payload = b"".join(chunks)
        after = os.fstat(descriptor)
        if (
            len(payload) > max_bytes
            or len(payload) != opened.st_size
            or (after.st_size, after.st_mtime_ns)
            != (opened.st_size, opened.st_mtime_ns)
        ):
            raise H10ReceiptShadowError(f"{label} changed while reading")
    finally:
        os.close(descriptor)
    return payload


def _verify_path_identity(path: Path, expected: os.stat_result, label: str) -> None:
    try:
        observed = os.stat(path, follow_symlinks=False)
    except OSError as error:
        raise H10ReceiptShadowError(f"{label} identity became unavailable") from error
    if (
        stat.S_ISLNK(observed.st_mode)
        or not stat.S_ISDIR(observed.st_mode)
        or (observed.st_dev, observed.st_ino)
        != (expected.st_dev, expected.st_ino)
    ):
        raise H10ReceiptShadowError(f"{label} identity changed")


def _require_exact_directory_entries(
    directory_fd: int, expected: frozenset[str], label: str
) -> None:
    try:
        observed = frozenset(os.listdir(directory_fd))
    except OSError as error:
        raise H10ReceiptShadowError(f"cannot enumerate {label}") from error
    if observed != expected:
        raise H10ReceiptShadowError(
            f"{label} entry set differs: missing={sorted(expected-observed)}, "
            f"extra={sorted(observed-expected)}"
        )


def _rename_noreplace(
    source_directory_fd: int,
    source_name: str,
    destination_directory_fd: int,
    destination_name: str,
) -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        raise H10ReceiptShadowError("renameat2 unavailable; refusing unsafe fallback")
    renameat2.argtypes = (
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    )
    renameat2.restype = ctypes.c_int
    result = renameat2(
        source_directory_fd,
        os.fsencode(source_name),
        destination_directory_fd,
        os.fsencode(destination_name),
        _RENAME_NOREPLACE,
    )
    if result == 0:
        return
    number = ctypes.get_errno()
    if number in (errno.EEXIST, errno.ENOTEMPTY):
        raise H10ReceiptShadowError(
            f"refusing to overwrite output root entry: {destination_name}"
        )
    if number in (errno.ENOSYS, errno.EINVAL, errno.EOPNOTSUPP):
        raise H10ReceiptShadowError(
            "atomic RENAME_NOREPLACE unsupported; refusing unsafe fallback"
        )
    raise OSError(number, os.strerror(number), destination_name)


def _check_output_location(output_root: Path, source_root: Path) -> Path:
    output = Path(os.path.abspath(os.fspath(output_root)))
    source = Path(os.path.abspath(os.fspath(source_root)))
    if output.name in ("", ".", ".."):
        raise H10ReceiptShadowError("output root must name one fresh directory")
    if os.path.lexists(output):
        raise H10ReceiptShadowError(f"refusing to overwrite output root: {output}")
    if output == source or source in output.parents or output in source.parents:
        raise H10ReceiptShadowError("output and sealed source roots must not overlap")
    return output


def _publish_create_only(
    *,
    output_root: Path,
    source_root: Path,
    arrays: Mapping[str, np.ndarray],
    manifest: Mapping[str, Any],
    npz_payload: bytes,
) -> None:
    if manifest.get("audit_complete") is not True or manifest.get("cap_event_count") != 0:
        raise H10ReceiptShadowError("refusing to publish incomplete/capped receipt trace")
    output = _check_output_location(output_root, source_root)
    output.parent.mkdir(parents=True, exist_ok=True)
    _assert_no_symlink_ancestors(output.parent, "output parent")
    parent_stat = os.lstat(output.parent)
    if stat.S_ISLNK(parent_stat.st_mode) or not stat.S_ISDIR(parent_stat.st_mode):
        raise H10ReceiptShadowError("output parent must be a non-symlink directory")
    if _hash_bytes(npz_payload) != manifest.get("npz_sha256"):
        raise H10ReceiptShadowError("manifest does not bind output NPZ bytes")
    if _array_content_sha256(arrays) != manifest.get("trace_array_content_sha256"):
        raise H10ReceiptShadowError("manifest does not bind output numeric content")
    manifest_payload = _canonical_json_bytes(manifest)
    directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    directory_flags |= getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    parent_fd = os.open(output.parent, directory_flags)
    staging_fd = -1
    output_fd = -1
    staging_name = f".{output.name}.stage.{secrets.token_hex(16)}"
    published = False
    try:
        opened_parent = os.fstat(parent_fd)
        if (opened_parent.st_dev, opened_parent.st_ino) != (
            parent_stat.st_dev,
            parent_stat.st_ino,
        ):
            raise H10ReceiptShadowError("output parent identity changed while opening")
        os.mkdir(staging_name, mode=0o700, dir_fd=parent_fd)
        created_staging = os.stat(
            staging_name, dir_fd=parent_fd, follow_symlinks=False
        )
        if not stat.S_ISDIR(created_staging.st_mode):
            raise H10ReceiptShadowError("created output staging entry is not a directory")
        staging_fd = os.open(staging_name, directory_flags, dir_fd=parent_fd)
        staging_identity = os.fstat(staging_fd)
        if (staging_identity.st_dev, staging_identity.st_ino) != (
            created_staging.st_dev,
            created_staging.st_ino,
        ):
            raise H10ReceiptShadowError("output staging identity changed while opening")
        _write_exclusive_fsync_at(staging_fd, OUTPUT_NPZ_NAME, npz_payload)
        _write_exclusive_fsync_at(staging_fd, OUTPUT_JSON_NAME, manifest_payload)
        os.fsync(staging_fd)
        expected_entries = frozenset({OUTPUT_NPZ_NAME, OUTPUT_JSON_NAME})
        _require_exact_directory_entries(
            staging_fd, expected_entries, "output staging directory"
        )
        if _hash_bytes(
            _read_regular_bytes_at(
                staging_fd,
                OUTPUT_NPZ_NAME,
                max_bytes=len(npz_payload),
                label="staged receipt NPZ",
            )
        ) != manifest["npz_sha256"]:
            raise H10ReceiptShadowError("staged receipt NPZ differs")
        if _hash_bytes(
            _read_regular_bytes_at(
                staging_fd,
                OUTPUT_JSON_NAME,
                max_bytes=len(manifest_payload),
                label="staged receipt JSON",
            )
        ) != _hash_bytes(manifest_payload):
            raise H10ReceiptShadowError("staged receipt JSON differs")
        _verify_path_identity(output.parent, opened_parent, "output parent")
        named_staging = os.stat(
            staging_name, dir_fd=parent_fd, follow_symlinks=False
        )
        if not stat.S_ISDIR(named_staging.st_mode) or (
            named_staging.st_dev,
            named_staging.st_ino,
        ) != (staging_identity.st_dev, staging_identity.st_ino):
            raise H10ReceiptShadowError("output staging identity changed")
        _rename_noreplace(parent_fd, staging_name, parent_fd, output.name)
        published = True
        os.fsync(parent_fd)
        _verify_path_identity(output.parent, opened_parent, "output parent")
        output_fd = os.open(output.name, directory_flags, dir_fd=parent_fd)
        published_identity = os.fstat(output_fd)
        if (published_identity.st_dev, published_identity.st_ino) != (
            staging_identity.st_dev,
            staging_identity.st_ino,
        ):
            raise H10ReceiptShadowError("published output directory identity differs")
        _require_exact_directory_entries(
            output_fd, expected_entries, "published output directory"
        )
        if _hash_bytes(
            _read_regular_bytes_at(
                output_fd,
                OUTPUT_NPZ_NAME,
                max_bytes=len(npz_payload),
                label="published receipt NPZ",
            )
        ) != manifest["npz_sha256"]:
            raise H10ReceiptShadowError("published receipt NPZ differs")
        if _hash_bytes(
            _read_regular_bytes_at(
                output_fd,
                OUTPUT_JSON_NAME,
                max_bytes=len(manifest_payload),
                label="published receipt JSON",
            )
        ) != _hash_bytes(manifest_payload):
            raise H10ReceiptShadowError("published receipt JSON differs")
    finally:
        if output_fd >= 0:
            os.close(output_fd)
        if not published and staging_fd >= 0:
            try:
                named = os.stat(
                    staging_name, dir_fd=parent_fd, follow_symlinks=False
                )
                held = os.fstat(staging_fd)
                if (named.st_dev, named.st_ino) == (held.st_dev, held.st_ino):
                    for name in (OUTPUT_NPZ_NAME, OUTPUT_JSON_NAME):
                        try:
                            os.unlink(name, dir_fd=staging_fd)
                        except FileNotFoundError:
                            pass
                    try:
                        os.rmdir(staging_name, dir_fd=parent_fd)
                    except OSError:
                        pass
            except OSError:
                pass
        if staging_fd >= 0:
            os.close(staging_fd)
        os.close(parent_fd)


def run_h10_receipt_shadow(
    *,
    source_root: Path,
    receipt_contract: Path,
    expected_receipt_contract_sha256: str,
    expected_runner_sha256: str,
    expected_runner_test_sha256: str,
    output_root: Path,
) -> dict[str, Any]:
    """Validate, replay, and create-only publish one H10 receipt shadow."""

    output = _check_output_location(output_root, source_root)
    _validate_runner_bindings(
        expected_runner_sha256, expected_runner_test_sha256
    )
    fixed_before = _validate_fixed_assets()
    contract_path, contract_snapshot = _load_contract(
        receipt_contract, expected_receipt_contract_sha256
    )
    (
        source_manifest,
        source_arrays,
        scenes,
        frames_by_scene,
        membership,
        selections,
        membership_hash,
        source_snapshot,
    ) = _load_sealed_source(source_root)
    source_memory_hash_before = _array_content_sha256(source_arrays)
    input_before = _snapshot_inputs(
        source_root=Path(source_snapshot["root"]["path"]),
        source_snapshot=source_snapshot,
        contract_path=contract_path,
        contract_snapshot=contract_snapshot,
        fixed_assets=fixed_before,
        source_arrays=source_arrays,
        expected_runner_sha256=expected_runner_sha256,
        expected_runner_test_sha256=expected_runner_test_sha256,
    )

    trace_arrays, summary = _run_tracking(
        scene_ids=scenes,
        frames_by_scene=frames_by_scene,
        source_arrays=source_arrays,
        selections=selections,
        tracker_factory=S3RReceiptTracker,
    )
    if _array_content_sha256(source_arrays) != source_memory_hash_before:
        raise H10ReceiptShadowError("loaded numeric source changed in memory")
    input_after = _snapshot_inputs(
        source_root=Path(source_snapshot["root"]["path"]),
        source_snapshot=source_snapshot,
        contract_path=contract_path,
        contract_snapshot=contract_snapshot,
        fixed_assets=fixed_before,
        source_arrays=source_arrays,
        expected_runner_sha256=expected_runner_sha256,
        expected_runner_test_sha256=expected_runner_test_sha256,
    )
    npz_payload = _deterministic_npz_bytes(trace_arrays)
    npz_sha256 = _hash_bytes(npz_payload)
    manifest = _build_manifest(
        source_manifest=source_manifest,
        source_arrays=source_arrays,
        trace_arrays=trace_arrays,
        summary=summary,
        scenes=scenes,
        membership=membership,
        membership_sha256=membership_hash,
        receipt_contract_sha256=str(contract_snapshot["sha256"]),
        expected_runner_sha256=expected_runner_sha256,
        expected_runner_test_sha256=expected_runner_test_sha256,
        input_before=input_before,
        input_after=input_after,
        npz_sha256=npz_sha256,
    )
    _publish_create_only(
        output_root=output,
        source_root=Path(source_snapshot["root"]["path"]),
        arrays=trace_arrays,
        manifest=manifest,
        npz_payload=npz_payload,
    )
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--receipt-contract", type=Path, required=True)
    parser.add_argument("--expected-receipt-contract-sha256", required=True)
    parser.add_argument("--expected-runner-sha256", required=True)
    parser.add_argument("--expected-runner-test-sha256", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        manifest = run_h10_receipt_shadow(
            source_root=args.source_root,
            receipt_contract=args.receipt_contract,
            expected_receipt_contract_sha256=args.expected_receipt_contract_sha256,
            expected_runner_sha256=args.expected_runner_sha256,
            expected_runner_test_sha256=args.expected_runner_test_sha256,
            output_root=args.output_root,
        )
    except H10ReceiptShadowError as error:
        raise SystemExit(f"H10 receipt shadow failed: {error}") from error
    print(
        json.dumps(
            {
                "schema": manifest["schema"],
                "output_root": os.fspath(Path(args.output_root).resolve()),
                "scene_count": manifest["scene_count"],
                "valid_frame_count": manifest["valid_frame_count"],
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
