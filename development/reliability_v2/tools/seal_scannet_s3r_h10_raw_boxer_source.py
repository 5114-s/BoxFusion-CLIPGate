#!/usr/bin/env python3
"""Seal fresh H10 per-frame Boxer receipts into one inert raw source.

The sealer accepts only the exact 769-frame provider namespace and the frozen
H10 schedule.  It verifies every journal row and numeric frame artifact before
flattening geometry.  It performs no association, semantic lookup, proposal
activation, or evaluation.  The score-only K8 table is a frozen membership
receipt; it does not alter the complete raw source.
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
import secrets
import stat
import sys
from typing import Any, Mapping, Sequence
import zipfile

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if os.fspath(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, os.fspath(REPOSITORY_ROOT))

from boxfusion.s3r_h10_provider_core import (  # noqa: E402
    EXPECTED_VALID_FRAME_COUNT,
    JOURNAL_SCHEMA,
    MAX_RAW_ROWS_PER_FRAME,
    PRECOMMIT_RUNTIME_SEMANTICS,
    SCHEDULE_SCHEMA,
    SEAL_SCHEMA,
    ExactScheduleBundle,
    SceneSchedule,
    ScheduledFrame,
    parse_exact_schedule_bundle,
)

SCHEMA = "boxfusion.s3r_h10_raw_boxer_source.v1"
PROVIDER_RUN_SCHEMA = "boxfusion.s3r_h10_fresh_boxer_provider_run.v1"
EXPECTED_SCHEDULE_SHA256 = (
    "1ce565a65510b80d69a0402fe7a40ea89920625f6a81147d42f9232f7a7761e9"
)
EXPECTED_HOLDOUT_LIST_SHA256 = (
    "8965d0534ed3028f85d8b0ea7227d348a6faa1387b858ddf42c3183bd9ebdf90"
)
EXPECTED_PROVIDER_CONTRACT_SHA256 = (
    "11cc5ab398809ccfab9fafdcc9645e796321eb2db527e78ef2515e99946883d0"
)
EXPECTED_PROVIDER_CORE_SHA256 = (
    "c70e114dabe1ef1081967027e4b5a15955ac16bab745652984dfe981100f21dd"
)
EXPECTED_PROVIDER_RUNNER_SHA256 = (
    "72e42f3a3865ee9f52687d2a5a5a40ecabe189864c4d7d2cce18daf6be056403"
)
DEFAULT_SCHEDULE = REPOSITORY_ROOT / "docs" / "data" / "S3R_H10_EXACT_SCHEDULE_V2.json"
OUTPUT_JSON_NAME = "S3R_H10_RAW_BOXER_SOURCE.json"
OUTPUT_NPZ_NAME = "S3R_H10_RAW_BOXER_SOURCE.npz"
PROVENANCE_NAME = "RUN_PROVENANCE.json"
PROVIDER_SEAL_NAME = "FINAL_SEAL.json"
JOURNAL_NAME = "frames.journal.jsonl"
FRAMES_DIRECTORY_NAME = "frames"
TOP_K = 8
SOURCE_INSTANCE_STRIDE = MAX_RAW_ROWS_PER_FRAME
MAX_TOTAL_RAW_ROWS = EXPECTED_VALID_FRAME_COUNT * MAX_RAW_ROWS_PER_FRAME
MAX_FRAME_NPZ_BYTES = 1024 * 1024
MAX_FRAME_UNCOMPRESSED_BYTES = 512 * 1024
MAX_PROVENANCE_BYTES = 32 * 1024 * 1024
MAX_JOURNAL_BYTES = 32 * 1024 * 1024
MAX_SEAL_BYTES = 128 * 1024

_RENAME_NOREPLACE = 1
_HEX = frozenset("0123456789abcdef")

_FRAME_ARRAYS = frozenset(
    {
        "center",
        "extent",
        "quaternion",
        "score",
        "source_row",
        "input_sha256",
        "runtime_seconds",
    }
)
_SOURCE_ARRAYS = frozenset(
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
_JOURNAL_HEADER_KEYS = frozenset({"schema", "schedule_sha256", "expected_frame_count"})
_JOURNAL_ROW_KEYS = frozenset(
    {
        "scene_id",
        "frame_id",
        "relative_path",
        "row_count",
        "file_sha256",
        "input_sha256",
        "runtime_seconds",
        "runtime_seconds_semantics",
    }
)
_INPUT_HASH_KEYS = frozenset({"intrinsic_color", "color", "depth", "pose"})
_PROVIDER_SEAL_KEYS = frozenset(
    {
        "schema",
        "schedule_sha256",
        "run_provenance_sha256",
        "completed_frame_count",
        "journal_sha256",
        "frame_record_sha256",
        "total_runtime_seconds",
        "runtime_seconds_semantics",
    }
)
_RUNTIME_FRAME_KEYS = frozenset(
    {
        "scene_id",
        "frame_id",
        "row_count",
        "precommit_compute_seconds",
        "end_to_end_seconds",
    }
)
_PROVENANCE_KEYS = frozenset(
    {
        "schema",
        "audit_complete",
        "shadow_only",
        "birth_enabled",
        "ap_evaluated",
        "gt_used",
        "target_dataset_training_used",
        "schedule",
        "provider_contract",
        "model_runtime",
        "environment",
        "frozen_assets",
        "formal_t05",
        "frame_inputs",
        "runtime",
        "output",
    }
)
_RUNTIME_KEYS = frozenset(
    {
        "cold_start_model_load_and_warmup_seconds",
        "cold_first_frame",
        "cold_first_frame_end_to_end_seconds",
        "cold_start_total_seconds",
        "precommit_compute_definition",
        "end_to_end_definition",
        "precommit_compute_summary",
        "all_frame_end_to_end_summary",
        "warm_frame_end_to_end_summary",
        "warm_frame_count",
        "deadline_uses",
        "process_peak_rss_bytes",
        "integrated_realtime_qualified",
        "frames",
    }
)
_OUTPUT_KEYS = frozenset(
    {
        "committed_frame_count",
        "raw_row_count",
        "empty_frame_count",
        "native_prediction_mutation",
        "tracked_csv_created",
    }
)
_ASSET_RECORD_KEYS = frozenset(
    {"path", "sha256_before", "expected_sha256", "sha256_after"}
)

_EXPECTED_MODEL_HASHES = {
    "owl_checkpoint": "14aa78ffe7b13e5b3ebf55845bc9a07e339a095cfd88f4c4e8f726b38ce1ebbf",
    "owl_text_cache": "59193fc014d381b2200edf1c1e6dc86324edb55a067189d3e84226a184185283",
    "boxer_checkpoint": "d5a30b348a8f5b0e5990ff3aa0e8f473ce77d860da22586322e7f47abc83ca6f",
    "dino_checkpoint": "4057cbaaad8c16657adb09d6815f28d4164eeba30532fde23f0d17313124caea",
}
_EXPECTED_EXTERNAL_CODE_HASHES = {
    "run_boxer.py": "8ff93e62881db5bd4d0fb20cbddfb5767ec2c4a941e873672c3acf603ecdad1b",
    "boxernet/boxernet.py": "a8009c1c0932aaab98bb074a2a4c50e55a3fbdfc3c6cb1afc9e1aef0e5324130",
    "owl/owl_wrapper.py": "7cf26a25bba1e67d8d8230ef47eb8288a48a728eda27d846e4f57bc6d4b6c628",
    "owl/clip_tokenizer.py": "39ac9e78731d91d0e50be80ac5ab1a2045ab28ab41e07ce35017b0eaa677dfe3",
    "owl/lvisplus_classes.csv": "3d6fd6fedb15ec5ea2f8ae80d2a5da310e64bece64aa38bb14f16cb7ac05cb3e",
    "utils/taxonomy.py": "42f26d270d6305c6cf3dbddc1635c4e7473837b6bd7bcc1654d8b43bf2018ec7",
    "utils/tw/camera.py": "dd31d0df949b2e937e81e76d994e40680e8c6412b7e424b22d8a5b43207521cf",
    "utils/tw/pose.py": "61091d10b5ecbb2720bf86ee78da21d8b0059ee45c03a5b127a4816606004703",
    "loaders/base_loader.py": "93e3e1fb600960b3f8dfcd9091a745787ccd6be258ef9ae54d08bebb3107839d",
    "loaders/scannet_loader.py": "93a451d70cd57ba01290e152ef5d7b95d4de7f0a835a010f3437b55242b9d4bf",
}


class RawSourceSealError(RuntimeError):
    """A provider receipt or create-only publication invariant failed."""


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in _HEX for character in value)
    )


def _require_sha256(value: object, label: str) -> str:
    if not _is_sha256(value):
        raise RawSourceSealError(f"{label} must be lowercase SHA-256 hex")
    return value


def _strict_int(value: object, label: str, *, minimum: int = 0) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, int):
        raise RawSourceSealError(f"{label} must be an integer")
    if value < minimum:
        raise RawSourceSealError(f"{label} must be >= {minimum}")
    return value


def _finite_nonnegative(value: object, label: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float)):
        raise RawSourceSealError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise RawSourceSealError(f"{label} must be finite and nonnegative")
    return result


def _exact_keys(value: Mapping[str, Any], expected: frozenset[str], label: str) -> None:
    actual = frozenset(value)
    if actual != expected:
        raise RawSourceSealError(
            f"{label} keys differ: missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}"
        )


def _duplicate_guard(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise RawSourceSealError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _canonical_json_bytes(
    value: Mapping[str, Any], *, indent: int | None = None
) -> bytes:
    separators = (",", ":") if indent is None else None
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=separators,
            indent=indent,
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("ascii")


def _canonical_json_line(value: Mapping[str, Any]) -> bytes:
    return _canonical_json_bytes(value).rstrip(b"\n") + b"\n"


def _read_regular_bytes(path: Path, *, max_bytes: int, label: str) -> bytes:
    absolute = Path(os.path.abspath(os.fspath(path)))
    try:
        before = os.lstat(absolute)
    except OSError as error:
        raise RawSourceSealError(f"cannot stat {label}: {absolute}: {error}") from error
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
        raise RawSourceSealError(f"{label} must be a non-symlink regular file")
    if before.st_size > max_bytes:
        raise RawSourceSealError(f"{label} exceeds byte cap")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(absolute, flags)
    try:
        opened = os.fstat(descriptor)
        if (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino):
            raise RawSourceSealError(f"{label} identity changed")
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
            raise RawSourceSealError(f"{label} changed while reading")
    finally:
        os.close(descriptor)
    return payload


def _parse_json_bytes(payload: bytes, label: str) -> dict[str, Any]:
    try:
        value = json.loads(payload.decode("utf-8"), object_pairs_hook=_duplicate_guard)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RawSourceSealError(f"{label} is not strict UTF-8 JSON") from error
    if not isinstance(value, dict):
        raise RawSourceSealError(f"{label} root must be an object")
    return value


def _hash_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _hash_file(path: Path, *, max_bytes: int, label: str) -> str:
    return _hash_bytes(_read_regular_bytes(path, max_bytes=max_bytes, label=label))


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


def _numeric_matrix_sha256(name: str, value: np.ndarray) -> str:
    return _array_content_sha256({name: np.ascontiguousarray(value)})


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


def _assert_no_symlink_ancestors(path: Path, label: str) -> None:
    absolute = Path(os.path.abspath(os.fspath(path)))
    current = Path(absolute.anchor)
    for component in absolute.parts[1:]:
        current /= component
        try:
            metadata = os.lstat(current)
        except FileNotFoundError:
            raise RawSourceSealError(f"{label} path component is absent: {current}")
        if stat.S_ISLNK(metadata.st_mode):
            raise RawSourceSealError(f"{label} has a symlink path component: {current}")


def _assert_provider_directory(provider_root: Path) -> tuple[Path, Path]:
    root = Path(os.path.abspath(os.fspath(provider_root)))
    _assert_no_symlink_ancestors(root, "provider root")
    try:
        root_stat = os.lstat(root)
    except OSError as error:
        raise RawSourceSealError(f"cannot stat provider root: {error}") from error
    if stat.S_ISLNK(root_stat.st_mode) or not stat.S_ISDIR(root_stat.st_mode):
        raise RawSourceSealError("provider root must be a non-symlink directory")
    expected_root_entries = {
        FRAMES_DIRECTORY_NAME,
        JOURNAL_NAME,
        PROVENANCE_NAME,
        PROVIDER_SEAL_NAME,
    }
    observed_root_entries = {entry.name for entry in os.scandir(root)}
    if observed_root_entries != expected_root_entries:
        raise RawSourceSealError(
            "provider root entry set differs; tracked, temporary, or extra output present"
        )
    frames = root / FRAMES_DIRECTORY_NAME
    frames_stat = os.lstat(frames)
    if stat.S_ISLNK(frames_stat.st_mode) or not stat.S_ISDIR(frames_stat.st_mode):
        raise RawSourceSealError(
            "provider frames entry must be a non-symlink directory"
        )
    return root, frames


def _expected_frame_name(scene: SceneSchedule, frame: ScheduledFrame) -> str:
    return f"{scene.scene_id}.{frame.frame_id:06d}.npz"


def _validate_frame_directory(
    frames_directory: Path, bundle: ExactScheduleBundle
) -> None:
    expected = {
        _expected_frame_name(scene, frame) for scene, frame in bundle.ordered_frames
    }
    observed: set[str] = set()
    for entry in os.scandir(frames_directory):
        if entry.is_symlink() or not entry.is_file(follow_symlinks=False):
            raise RawSourceSealError("provider frames contain a non-regular entry")
        observed.add(entry.name)
    if observed != expected:
        raise RawSourceSealError(
            f"provider frame file set differs: missing={len(expected-observed)}, "
            f"extra={len(observed-expected)}"
        )


def _validate_provider_seal(
    value: Mapping[str, Any], *, schedule_sha256: str, provenance_sha256: str
) -> dict[str, Any]:
    _exact_keys(value, _PROVIDER_SEAL_KEYS, "provider final seal")
    if value["schema"] != SEAL_SCHEMA:
        raise RawSourceSealError("provider final seal schema differs")
    if value["schedule_sha256"] != schedule_sha256:
        raise RawSourceSealError("provider final seal schedule hash differs")
    if value["run_provenance_sha256"] != provenance_sha256:
        raise RawSourceSealError("provider final seal provenance hash differs")
    if value["completed_frame_count"] != EXPECTED_VALID_FRAME_COUNT:
        raise RawSourceSealError("provider final seal frame count differs")
    _require_sha256(value["journal_sha256"], "provider journal hash")
    _require_sha256(value["frame_record_sha256"], "provider frame-record hash")
    _finite_nonnegative(value["total_runtime_seconds"], "provider total runtime")
    if value["runtime_seconds_semantics"] != PRECOMMIT_RUNTIME_SEMANTICS:
        raise RawSourceSealError("provider runtime semantics differ")
    return dict(value)


def _validate_provider_contract(value: object) -> tuple[dict[str, Any], str]:
    if not isinstance(value, Mapping):
        raise RawSourceSealError("provider contract must be an object")
    required = {
        "model_process_count": 1,
        "owl_instance_count": 1,
        "boxernet_instance_count": 1,
        "taxonomy": "lvisplus",
        "prompt_count": 1220,
        "threshold_2d": 0.25,
        "nms_iou_2d": 0.5,
        "threshold_3d": 0.5,
        "score_rule": "mean(owl_2d_score,boxer_3d_score)_after_3d_threshold",
        "image_hw": [960, 960],
        "precision": "bfloat16",
        "seed": 0,
        "temporal_state": False,
        "prefetch": False,
        "frame_directory_enumeration": False,
        "coordinate_convention": (
            "absolute_scannet_world=center_boxer_recentered+"
            "translation_of_first_valid_exact_schedule_pose;"
            "extent_unchanged;Hamilton_wxyz_quaternion_l2_normalized"
        ),
    }
    _exact_keys(value, frozenset(required), "provider contract")
    for key, expected in required.items():
        if value.get(key) != expected:
            raise RawSourceSealError(f"provider contract differs for {key}")
    frozen = dict(value)
    return frozen, _hash_bytes(_canonical_json_bytes(frozen))


def _validate_assets(
    value: object,
) -> tuple[dict[str, str], dict[str, str], dict[str, str], str]:
    if not isinstance(value, Mapping):
        raise RawSourceSealError("provider frozen-assets ledger must be an object")
    expected_names = {
        "schedule",
        "holdout_list",
        "provider_contract",
        "owl_checkpoint",
        "owl_text_cache",
        "boxer_checkpoint",
        "dino_checkpoint",
        "runner_source",
        "provider_core_source",
        *(f"boxer_code:{name}" for name in _EXPECTED_EXTERNAL_CODE_HASHES),
    }
    if set(value) != expected_names:
        raise RawSourceSealError("provider frozen-assets entry set differs")
    code_hashes: dict[str, str] = {}
    model_hashes: dict[str, str] = {}
    protocol_hashes: dict[str, str] = {}
    canonical: dict[str, dict[str, Any]] = {}
    for name in sorted(expected_names):
        record = value[name]
        if not isinstance(record, Mapping):
            raise RawSourceSealError(f"provider asset {name} must be an object")
        _exact_keys(record, _ASSET_RECORD_KEYS, f"provider asset {name}")
        if not isinstance(record["path"], str) or not record["path"]:
            raise RawSourceSealError(f"provider asset {name} path is invalid")
        before = _require_sha256(record["sha256_before"], f"provider asset {name}")
        after = _require_sha256(record["sha256_after"], f"provider asset {name}")
        if before != after:
            raise RawSourceSealError(f"provider asset {name} changed during inference")
        approved_hash: str
        ledger_expected_hash: str | None
        if name == "schedule":
            approved_hash = EXPECTED_SCHEDULE_SHA256
            ledger_expected_hash = approved_hash
            protocol_hashes[name] = before
        elif name == "holdout_list":
            approved_hash = EXPECTED_HOLDOUT_LIST_SHA256
            ledger_expected_hash = approved_hash
            protocol_hashes[name] = before
        elif name == "provider_contract":
            approved_hash = EXPECTED_PROVIDER_CONTRACT_SHA256
            ledger_expected_hash = approved_hash
            protocol_hashes[name] = before
        elif name in _EXPECTED_MODEL_HASHES:
            approved_hash = _EXPECTED_MODEL_HASHES[name]
            ledger_expected_hash = approved_hash
            model_hashes[name] = before
        elif name.startswith("boxer_code:"):
            relative = name.split(":", 1)[1]
            approved_hash = _EXPECTED_EXTERNAL_CODE_HASHES[relative]
            ledger_expected_hash = approved_hash
            code_hashes[name] = before
        elif name == "runner_source":
            approved_hash = EXPECTED_PROVIDER_RUNNER_SHA256
            ledger_expected_hash = None
            code_hashes[name] = before
        elif name == "provider_core_source":
            approved_hash = EXPECTED_PROVIDER_CORE_SHA256
            ledger_expected_hash = None
            code_hashes[name] = before
        else:  # pragma: no cover - exact asset-name set makes this unreachable
            raise RawSourceSealError(f"unclassified provider asset: {name}")
        if record["expected_sha256"] != ledger_expected_hash or before != approved_hash:
            raise RawSourceSealError(f"provider asset {name} frozen hash differs")
        canonical[name] = dict(record)
    return (
        code_hashes,
        model_hashes,
        protocol_hashes,
        _hash_bytes(_canonical_json_bytes(canonical)),
    )


def _percentile_summary(
    values: Sequence[float], *, expected_count: int
) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.shape != (expected_count,) or not np.isfinite(array).all():
        raise RawSourceSealError("provider runtime vector differs")
    return {
        "p50_seconds": float(np.percentile(array, 50)),
        "p95_seconds": float(np.percentile(array, 95)),
        "max_seconds": float(np.max(array)),
    }


def _same_runtime_summary(observed: object, expected: Mapping[str, float]) -> bool:
    if not isinstance(observed, Mapping) or set(observed) != set(expected):
        return False
    return all(
        isinstance(observed[key], (int, float))
        and not isinstance(observed[key], bool)
        and math.isfinite(float(observed[key]))
        and math.isclose(float(observed[key]), value, rel_tol=0.0, abs_tol=1e-12)
        for key, value in expected.items()
    )


def _exact_input_ledger_sha256(bundle: ExactScheduleBundle) -> tuple[int, str]:
    digest = hashlib.sha256()
    count = 0
    for scene in bundle.scenes:
        entries: list[tuple[str, str, str]] = [
            (
                "intrinsic",
                scene.intrinsic_color_relpath,
                scene.intrinsic_color_sha256,
            )
        ]
        for frame in scene.frames:
            entries.extend(
                (
                    (
                        f"{frame.frame_id}:color",
                        frame.color_relpath,
                        frame.color_sha256,
                    ),
                    (
                        f"{frame.frame_id}:depth",
                        frame.depth_relpath,
                        frame.depth_sha256,
                    ),
                    (f"{frame.frame_id}:pose", frame.pose_relpath, frame.pose_sha256),
                )
            )
        for role, relative, expected_hash in entries:
            digest.update(scene.scene_id.encode("ascii"))
            digest.update(b"\0")
            digest.update(role.encode("ascii"))
            digest.update(b"\0")
            digest.update(relative.encode("ascii"))
            digest.update(b"\0")
            digest.update(expected_hash.encode("ascii"))
            digest.update(b"\n")
            count += 1
    return count, digest.hexdigest()


def _validate_provenance(
    value: Mapping[str, Any], bundle: ExactScheduleBundle
) -> dict[str, Any]:
    _exact_keys(value, _PROVENANCE_KEYS, "provider run provenance")
    if value.get("schema") != PROVIDER_RUN_SCHEMA:
        raise RawSourceSealError("provider run schema differs")
    if value.get("audit_complete") is not True or value.get("shadow_only") is not True:
        raise RawSourceSealError("provider run is not a complete shadow receipt")
    if (
        value.get("birth_enabled") is not False
        or value.get("ap_evaluated") is not False
        or value.get("gt_used") is not False
        or value.get("target_dataset_training_used") is not False
    ):
        raise RawSourceSealError("provider run violates frozen shadow-only controls")
    schedule = value.get("schedule")
    if not isinstance(schedule, Mapping):
        raise RawSourceSealError("provider schedule receipt is absent")
    expected_schedule = {
        "schema": SCHEDULE_SCHEMA,
        "sha256": bundle.sha256,
        "scene_order": list(bundle.scene_order),
        "valid_frame_count": bundle.valid_frame_count,
        "raw_frame_count": bundle.raw_frame_count,
        "excluded_frame_count": bundle.raw_frame_count - bundle.valid_frame_count,
    }
    if dict(schedule) != expected_schedule:
        raise RawSourceSealError("provider schedule receipt differs")
    input_count, input_ledger_hash = _exact_input_ledger_sha256(bundle)
    frame_inputs = value.get("frame_inputs")
    expected_frame_inputs = {
        "before_each_frame_read_verified": True,
        "after_complete_stream_verified": True,
        "frame_inputs_before_read_and_after_stream_verified": True,
        "verified_file_count": input_count,
        "expected_file_count": input_count,
        "exact_input_ledger_sha256": input_ledger_hash,
    }
    if (
        not isinstance(frame_inputs, Mapping)
        or dict(frame_inputs) != expected_frame_inputs
    ):
        raise RawSourceSealError("provider exact-input receipt differs")
    contract, contract_hash = _validate_provider_contract(
        value.get("provider_contract")
    )
    code_hashes, model_hashes, protocol_hashes, assets_hash = _validate_assets(
        value.get("frozen_assets")
    )
    model_runtime = value.get("model_runtime")
    if (
        not isinstance(model_runtime, Mapping)
        or model_runtime.get("owl_instance_count") != 1
        or model_runtime.get("boxernet_instance_count") != 1
        or model_runtime.get("prompt_count") != 1220
        or model_runtime.get("boxer_image_hw") != 960
        or model_runtime.get("owl_use_bfloat16") is not True
    ):
        raise RawSourceSealError("provider model-runtime receipt differs")
    output = value.get("output")
    if not isinstance(output, Mapping):
        raise RawSourceSealError("provider output receipt is absent")
    _exact_keys(output, _OUTPUT_KEYS, "provider output receipt")
    if output.get("committed_frame_count") != bundle.valid_frame_count:
        raise RawSourceSealError("provider committed-frame count differs")
    if (
        output.get("tracked_csv_created") is not False
        or output.get("native_prediction_mutation") is not False
    ):
        raise RawSourceSealError("provider reports a forbidden output mutation")
    raw_row_count = _strict_int(output.get("raw_row_count"), "provider raw-row count")
    empty_frame_count = _strict_int(
        output.get("empty_frame_count"), "provider empty-frame count"
    )

    runtime = value.get("runtime")
    if not isinstance(runtime, Mapping):
        raise RawSourceSealError("provider runtime receipt is absent")
    _exact_keys(runtime, _RUNTIME_KEYS, "provider runtime receipt")
    if runtime.get("precommit_compute_definition") != (
        "current-frame verified reads + synchronous datum construction + "
        "OWL + Boxer + CUDA synchronize; excludes persistence"
    ) or runtime.get("end_to_end_definition") != (
        "precommit compute + frame NPZ fsync + frame-directory fsync + journal fsync"
    ):
        raise RawSourceSealError("provider runtime definitions differ")
    if runtime.get("integrated_realtime_qualified") is not False:
        raise RawSourceSealError(
            "provider overclaims integrated realtime qualification"
        )
    _finite_nonnegative(
        runtime.get("cold_start_model_load_and_warmup_seconds"),
        "provider cold-start runtime",
    )
    if runtime.get("deadline_uses") != (
        "warm_frame_end_to_end_summary_after_global_first_committed_frame"
    ):
        raise RawSourceSealError("provider deadline accounting differs")
    rows = runtime.get("frames")
    if not isinstance(rows, list) or len(rows) != bundle.valid_frame_count:
        raise RawSourceSealError("provider runtime frame ledger differs")
    normalized_rows: list[dict[str, Any]] = []
    precommit_values: list[float] = []
    end_to_end_values: list[float] = []
    for schedule_index, ((scene, frame), row) in enumerate(
        zip(bundle.ordered_frames, rows)
    ):
        if not isinstance(row, Mapping):
            raise RawSourceSealError("provider runtime frame row is not an object")
        _exact_keys(row, _RUNTIME_FRAME_KEYS, f"runtime frame {schedule_index}")
        if row["scene_id"] != scene.scene_id or row["frame_id"] != frame.frame_id:
            raise RawSourceSealError("provider runtime frame order differs")
        count = _strict_int(row["row_count"], "provider runtime row count")
        if count > MAX_RAW_ROWS_PER_FRAME:
            raise RawSourceSealError("provider runtime raw-row cap exceeded")
        precommit = _finite_nonnegative(
            row["precommit_compute_seconds"], "provider precommit runtime"
        )
        end_to_end = _finite_nonnegative(
            row["end_to_end_seconds"], "provider end-to-end runtime"
        )
        if end_to_end < precommit:
            raise RawSourceSealError("provider end-to-end runtime precedes precommit")
        normalized_rows.append(dict(row))
        precommit_values.append(precommit)
        end_to_end_values.append(end_to_end)
    if runtime.get("cold_first_frame") != normalized_rows[0]:
        raise RawSourceSealError("provider cold-frame receipt differs")
    cold_frame_runtime = _finite_nonnegative(
        runtime.get("cold_first_frame_end_to_end_seconds"),
        "provider cold-frame runtime",
    )
    if cold_frame_runtime != end_to_end_values[0]:
        raise RawSourceSealError("provider cold-frame runtime differs")
    cold_start = _finite_nonnegative(
        runtime.get("cold_start_model_load_and_warmup_seconds"),
        "provider cold-start runtime",
    )
    cold_start_total = _finite_nonnegative(
        runtime.get("cold_start_total_seconds"),
        "provider cold-start total",
    )
    if cold_start_total != cold_start + end_to_end_values[0]:
        raise RawSourceSealError("provider cold-start total differs")
    if runtime.get("warm_frame_count") != bundle.valid_frame_count - 1:
        raise RawSourceSealError("provider warm-frame count differs")
    _strict_int(runtime.get("process_peak_rss_bytes"), "provider peak RSS")
    if (
        not _same_runtime_summary(
            runtime.get("precommit_compute_summary"),
            _percentile_summary(
                precommit_values, expected_count=bundle.valid_frame_count
            ),
        )
        or not _same_runtime_summary(
            runtime.get("all_frame_end_to_end_summary"),
            _percentile_summary(
                end_to_end_values, expected_count=bundle.valid_frame_count
            ),
        )
        or not _same_runtime_summary(
            runtime.get("warm_frame_end_to_end_summary"),
            _percentile_summary(
                end_to_end_values[1:], expected_count=bundle.valid_frame_count - 1
            ),
        )
    ):
        raise RawSourceSealError("provider runtime summary differs from frame ledger")
    if sum(int(row["row_count"]) for row in normalized_rows) != raw_row_count:
        raise RawSourceSealError("provider raw-row total differs from runtime ledger")
    if sum(int(row["row_count"]) == 0 for row in normalized_rows) != empty_frame_count:
        raise RawSourceSealError(
            "provider empty-frame total differs from runtime ledger"
        )
    return {
        "contract": contract,
        "contract_sha256": contract_hash,
        "code_hashes": code_hashes,
        "model_hashes": model_hashes,
        "protocol_hashes": protocol_hashes,
        "assets_sha256": assets_hash,
        "runtime_rows": normalized_rows,
        "raw_row_count": raw_row_count,
        "empty_frame_count": empty_frame_count,
        "exact_input_ledger_sha256": input_ledger_hash,
    }


def _parse_journal(
    payload: bytes,
    *,
    bundle: ExactScheduleBundle,
    final_seal: Mapping[str, Any],
) -> list[dict[str, Any]]:
    if not payload.endswith(b"\n"):
        raise RawSourceSealError("provider journal must end with newline")
    lines = payload.splitlines(keepends=True)
    if len(lines) != bundle.valid_frame_count + 1 or any(
        line == b"\n" for line in lines
    ):
        raise RawSourceSealError("provider journal line count differs")
    header = _parse_json_bytes(lines[0], "provider journal header")
    _exact_keys(header, _JOURNAL_HEADER_KEYS, "provider journal header")
    if header != {
        "schema": JOURNAL_SCHEMA,
        "schedule_sha256": bundle.sha256,
        "expected_frame_count": bundle.valid_frame_count,
    }:
        raise RawSourceSealError("provider journal header differs")
    if _canonical_json_line(header) != lines[0]:
        raise RawSourceSealError("provider journal header is not canonical")
    record_digest = hashlib.sha256()
    records: list[dict[str, Any]] = []
    total_runtime = 0.0
    for schedule_index, ((scene, frame), line) in enumerate(
        zip(bundle.ordered_frames, lines[1:])
    ):
        row = _parse_json_bytes(line, f"provider journal row {schedule_index}")
        _exact_keys(row, _JOURNAL_ROW_KEYS, f"provider journal row {schedule_index}")
        if _canonical_json_line(row) != line:
            raise RawSourceSealError("provider journal row is not canonical")
        expected_relative = f"frames/{_expected_frame_name(scene, frame)}"
        if (
            row["scene_id"] != scene.scene_id
            or row["frame_id"] != frame.frame_id
            or row["relative_path"] != expected_relative
        ):
            raise RawSourceSealError("provider journal frame order differs")
        count = _strict_int(row["row_count"], "provider journal row count")
        if count > MAX_RAW_ROWS_PER_FRAME:
            raise RawSourceSealError("provider journal raw-row cap exceeded")
        _require_sha256(row["file_sha256"], "provider frame artifact hash")
        input_hashes = row["input_sha256"]
        if not isinstance(input_hashes, Mapping):
            raise RawSourceSealError("provider journal input hashes must be an object")
        _exact_keys(input_hashes, _INPUT_HASH_KEYS, "provider journal input hashes")
        expected_hashes = {
            "intrinsic_color": scene.intrinsic_color_sha256,
            "color": frame.color_sha256,
            "depth": frame.depth_sha256,
            "pose": frame.pose_sha256,
        }
        if dict(input_hashes) != expected_hashes:
            raise RawSourceSealError(
                "provider journal input hashes differ from schedule"
            )
        runtime = _finite_nonnegative(
            row["runtime_seconds"], "provider journal runtime"
        )
        if row["runtime_seconds_semantics"] != PRECOMMIT_RUNTIME_SEMANTICS:
            raise RawSourceSealError("provider journal runtime semantics differ")
        total_runtime += runtime
        record_digest.update(line)
        records.append(dict(row))
    if _hash_bytes(payload) != final_seal["journal_sha256"]:
        raise RawSourceSealError("provider journal hash differs from final seal")
    if record_digest.hexdigest() != final_seal["frame_record_sha256"]:
        raise RawSourceSealError("provider frame-record hash differs from final seal")
    if total_runtime != float(final_seal["total_runtime_seconds"]):
        raise RawSourceSealError("provider total runtime differs from journal")
    return records


def _validate_frame_zip(payload: bytes) -> None:
    try:
        with zipfile.ZipFile(io.BytesIO(payload), mode="r") as archive:
            infos = archive.infolist()
            expected_names = {f"{name}.npy" for name in _FRAME_ARRAYS}
            names = [info.filename for info in infos]
            if len(names) != len(set(names)) or set(names) != expected_names:
                raise RawSourceSealError("provider frame NPZ member set differs")
            if any(
                info.is_dir()
                or info.flag_bits & 0x1
                or info.file_size > MAX_FRAME_UNCOMPRESSED_BYTES
                for info in infos
            ):
                raise RawSourceSealError(
                    "provider frame NPZ member violates safety cap"
                )
            if sum(info.file_size for info in infos) > MAX_FRAME_UNCOMPRESSED_BYTES:
                raise RawSourceSealError("provider frame NPZ expands beyond safety cap")
    except (OSError, zipfile.BadZipFile) as error:
        raise RawSourceSealError(
            "provider frame artifact is not a valid NPZ"
        ) from error


def _load_frame_arrays(
    payload: bytes,
    *,
    scene: SceneSchedule,
    frame: ScheduledFrame,
    journal: Mapping[str, Any],
) -> dict[str, np.ndarray]:
    _validate_frame_zip(payload)
    try:
        with np.load(io.BytesIO(payload), allow_pickle=False) as source:
            if set(source.files) != _FRAME_ARRAYS:
                raise RawSourceSealError("provider frame array set differs")
            arrays = {name: np.array(source[name], copy=True) for name in source.files}
    except (OSError, ValueError) as error:
        if isinstance(error, RawSourceSealError):
            raise
        raise RawSourceSealError("provider frame arrays cannot be loaded") from error
    count = int(journal["row_count"])
    expected_shapes = {
        "center": (count, 3),
        "extent": (count, 3),
        "quaternion": (count, 4),
        "score": (count,),
        "source_row": (count,),
        "input_sha256": (4, 32),
        "runtime_seconds": (1,),
    }
    expected_dtypes = {
        "center": np.dtype(np.float64),
        "extent": np.dtype(np.float64),
        "quaternion": np.dtype(np.float64),
        "score": np.dtype(np.float64),
        "source_row": np.dtype(np.int64),
        "input_sha256": np.dtype(np.uint8),
        "runtime_seconds": np.dtype(np.float64),
    }
    for name in _FRAME_ARRAYS:
        if (
            arrays[name].shape != expected_shapes[name]
            or arrays[name].dtype != expected_dtypes[name]
        ):
            raise RawSourceSealError(f"provider frame {name} shape or dtype differs")
    if not all(
        np.isfinite(arrays[name]).all()
        for name in ("center", "extent", "quaternion", "score", "runtime_seconds")
    ):
        raise RawSourceSealError("provider frame contains non-finite numeric values")
    if np.any(arrays["extent"] <= 0.0):
        raise RawSourceSealError("provider frame extent must be positive")
    if count and np.max(np.abs(arrays["center"])) > 10_000.0:
        raise RawSourceSealError("provider frame center exceeds frozen bound")
    if count and np.max(arrays["extent"]) > 100.0:
        raise RawSourceSealError("provider frame extent exceeds frozen bound")
    if np.any((arrays["score"] < 0.0) | (arrays["score"] > 1.0)):
        raise RawSourceSealError("provider frame score is outside [0,1]")
    if not np.array_equal(arrays["source_row"], np.arange(count, dtype=np.int64)):
        raise RawSourceSealError("provider frame source rows differ from 0..N-1")
    if count:
        squared_norm = np.einsum("ij,ij->i", arrays["quaternion"], arrays["quaternion"])
        if np.any(squared_norm <= 1e-12) or not np.allclose(
            squared_norm, np.ones(count), rtol=0.0, atol=1e-12
        ):
            raise RawSourceSealError("provider frame quaternion norm is invalid")
    expected_input = np.stack(
        [
            np.frombuffer(bytes.fromhex(value), dtype=np.uint8)
            for value in (
                scene.intrinsic_color_sha256,
                frame.color_sha256,
                frame.depth_sha256,
                frame.pose_sha256,
            )
        ]
    )
    if not np.array_equal(arrays["input_sha256"], expected_input):
        raise RawSourceSealError("provider frame input hashes differ from schedule")
    if arrays["runtime_seconds"][0] != float(journal["runtime_seconds"]):
        raise RawSourceSealError("provider frame runtime differs from journal")
    return arrays


def _snapshot_sha256(snapshot: Mapping[str, str]) -> str:
    digest = hashlib.sha256()
    for name in sorted(snapshot):
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(snapshot[name].encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _rehash_snapshot(
    provider_root: Path,
    bundle: ExactScheduleBundle,
    schedule_path: Path,
) -> dict[str, str]:
    snapshot = {
        "schedule": _hash_file(
            schedule_path, max_bytes=32 * 1024 * 1024, label="exact schedule"
        ),
        f"provider/{PROVENANCE_NAME}": _hash_file(
            provider_root / PROVENANCE_NAME,
            max_bytes=MAX_PROVENANCE_BYTES,
            label="provider provenance",
        ),
        f"provider/{PROVIDER_SEAL_NAME}": _hash_file(
            provider_root / PROVIDER_SEAL_NAME,
            max_bytes=MAX_SEAL_BYTES,
            label="provider final seal",
        ),
        f"provider/{JOURNAL_NAME}": _hash_file(
            provider_root / JOURNAL_NAME,
            max_bytes=MAX_JOURNAL_BYTES,
            label="provider journal",
        ),
    }
    for scene, frame in bundle.ordered_frames:
        name = _expected_frame_name(scene, frame)
        relative = f"provider/{FRAMES_DIRECTORY_NAME}/{name}"
        snapshot[relative] = _hash_file(
            provider_root / FRAMES_DIRECTORY_NAME / name,
            max_bytes=MAX_FRAME_NPZ_BYTES,
            label=f"provider frame {scene.scene_id}/{frame.frame_id}",
        )
    return snapshot


def _build_source_arrays(
    *,
    provider_root: Path,
    bundle: ExactScheduleBundle,
    journal_records: Sequence[Mapping[str, Any]],
    runtime_rows: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, np.ndarray], list[list[int]], list[list[int]], list[int]]:
    counts = np.asarray(
        [int(row["row_count"]) for row in journal_records], dtype=np.int64
    )
    total_count = int(np.sum(counts, dtype=np.int64))
    if total_count > MAX_TOTAL_RAW_ROWS:
        raise RawSourceSealError("provider total raw-row cap exceeded")
    arrays: dict[str, np.ndarray] = {
        "scene_ids": np.asarray(bundle.scene_order, dtype="<U12"),
        "per_view_scene_index": np.empty(total_count, dtype=np.int16),
        "per_view_frame_id": np.empty(total_count, dtype=np.int64),
        "per_view_source_row": np.empty(total_count, dtype=np.int64),
        "per_view_source_instance_id": np.empty(total_count, dtype=np.int64),
        "per_view_source_score": np.empty(total_count, dtype=np.float64),
        "per_view_center_world": np.empty((total_count, 3), dtype=np.float64),
        "per_view_extent_xyz": np.empty((total_count, 3), dtype=np.float64),
        "per_view_quaternion_wxyz": np.empty((total_count, 4), dtype=np.float64),
    }
    scene_lookup = {scene: index for index, scene in enumerate(bundle.scene_order)}
    frame_ledger: list[list[int]] = []
    empty_identities: list[list[int]] = []
    scene_row_counts = [0] * len(bundle.scene_order)
    cursor = 0
    for schedule_index, ((scene, frame), journal, runtime) in enumerate(
        zip(bundle.ordered_frames, journal_records, runtime_rows)
    ):
        if int(runtime["row_count"]) != int(journal["row_count"]):
            raise RawSourceSealError("provider runtime and journal row counts differ")
        if float(runtime["precommit_compute_seconds"]) != float(
            journal["runtime_seconds"]
        ):
            raise RawSourceSealError(
                "provider provenance and journal precommit runtimes differ"
            )
        name = _expected_frame_name(scene, frame)
        payload = _read_regular_bytes(
            provider_root / FRAMES_DIRECTORY_NAME / name,
            max_bytes=MAX_FRAME_NPZ_BYTES,
            label=f"provider frame {scene.scene_id}/{frame.frame_id}",
        )
        if _hash_bytes(payload) != journal["file_sha256"]:
            raise RawSourceSealError(
                "provider frame artifact hash differs from journal"
            )
        frame_arrays = _load_frame_arrays(
            payload, scene=scene, frame=frame, journal=journal
        )
        count = len(frame_arrays["source_row"])
        scene_index = scene_lookup[scene.scene_id]
        frame_ledger.append([scene_index, frame.frame_id, count])
        if count == 0:
            empty_identities.append([scene_index, frame.frame_id])
            continue
        positions = slice(cursor, cursor + count)
        arrays["per_view_scene_index"][positions] = scene_index
        arrays["per_view_frame_id"][positions] = frame.frame_id
        arrays["per_view_source_row"][positions] = frame_arrays["source_row"]
        arrays["per_view_source_instance_id"][positions] = (
            schedule_index * SOURCE_INSTANCE_STRIDE + frame_arrays["source_row"]
        )
        arrays["per_view_source_score"][positions] = frame_arrays["score"]
        arrays["per_view_center_world"][positions] = frame_arrays["center"]
        arrays["per_view_extent_xyz"][positions] = frame_arrays["extent"]
        arrays["per_view_quaternion_wxyz"][positions] = frame_arrays["quaternion"]
        cursor += count
        scene_row_counts[scene_index] += count
    if cursor != total_count:
        raise RawSourceSealError("flattened provider row count differs")
    for value in arrays.values():
        value.setflags(write=False)
    return arrays, frame_ledger, empty_identities, scene_row_counts


def _freeze_k8(
    arrays: Mapping[str, np.ndarray], frame_ledger: Sequence[Sequence[int]]
) -> tuple[np.ndarray, list[int]]:
    membership: list[list[int]] = []
    per_scene = [0] * len(arrays["scene_ids"])
    cursor = 0
    for scene_index, frame_id, count in frame_ledger:
        start = cursor
        stop = start + count
        ranked = sorted(
            range(start, stop),
            key=lambda sealed_row: (
                -float(arrays["per_view_source_score"][sealed_row]),
                int(arrays["per_view_source_row"][sealed_row]),
                sealed_row,
            ),
        )[:TOP_K]
        for sealed_row in ranked:
            membership.append(
                [
                    int(scene_index),
                    int(frame_id),
                    int(arrays["per_view_source_row"][sealed_row]),
                    int(arrays["per_view_source_instance_id"][sealed_row]),
                    int(sealed_row),
                ]
            )
        per_scene[int(scene_index)] += len(ranked)
        cursor = stop
    if cursor != len(arrays["per_view_scene_index"]):
        raise RawSourceSealError("K8 traversal did not consume the complete raw source")
    if membership:
        matrix = np.asarray(membership, dtype=np.int64)
    else:
        matrix = np.empty((0, 5), dtype=np.int64)
    matrix.setflags(write=False)
    return matrix, per_scene


def _validate_staged_source(
    *,
    npz_payload: bytes,
    arrays: Mapping[str, np.ndarray],
    manifest: Mapping[str, Any],
) -> None:
    try:
        with np.load(io.BytesIO(npz_payload), allow_pickle=False) as source:
            if set(source.files) != _SOURCE_ARRAYS:
                raise RawSourceSealError("staged source NPZ array set differs")
            loaded = {name: np.array(source[name], copy=True) for name in source.files}
    except (OSError, ValueError) as error:
        if isinstance(error, RawSourceSealError):
            raise
        raise RawSourceSealError("staged source NPZ cannot be loaded") from error
    if _array_content_sha256(loaded) != manifest["array_content_sha256"]:
        raise RawSourceSealError("staged source content hash differs")
    if _array_content_sha256(arrays) != manifest["array_content_sha256"]:
        raise RawSourceSealError("source arrays changed before publication")
    if _hash_bytes(npz_payload) != manifest["npz_sha256"]:
        raise RawSourceSealError("staged source NPZ byte hash differs")


def _write_exclusive_fsync_at(directory_fd: int, name: str, payload: bytes) -> None:
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
        raise RawSourceSealError(f"cannot stat {label}: {error}") from error
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
        raise RawSourceSealError(f"{label} must be a non-symlink regular file")
    if before.st_size > max_bytes:
        raise RawSourceSealError(f"{label} exceeds byte cap")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(name, flags, dir_fd=directory_fd)
    try:
        opened = os.fstat(descriptor)
        if (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino):
            raise RawSourceSealError(f"{label} identity changed")
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
            raise RawSourceSealError(f"{label} changed while reading")
    finally:
        os.close(descriptor)
    return payload


def _verify_path_identity(path: Path, expected: os.stat_result, label: str) -> None:
    try:
        observed = os.stat(path, follow_symlinks=False)
    except OSError as error:
        raise RawSourceSealError(f"{label} identity became unavailable") from error
    if (
        stat.S_ISLNK(observed.st_mode)
        or not stat.S_ISDIR(observed.st_mode)
        or (observed.st_dev, observed.st_ino) != (expected.st_dev, expected.st_ino)
    ):
        raise RawSourceSealError(f"{label} identity changed")


def _require_exact_directory_entries(
    directory_fd: int, expected: frozenset[str], label: str
) -> None:
    try:
        observed = frozenset(os.listdir(directory_fd))
    except OSError as error:
        raise RawSourceSealError(f"cannot enumerate {label}") from error
    if observed != expected:
        raise RawSourceSealError(
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
        raise RawSourceSealError("renameat2 unavailable; refusing non-atomic fallback")
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
        raise RawSourceSealError(
            f"refusing to overwrite output root entry: {destination_name}"
        )
    if number in (errno.ENOSYS, errno.EINVAL, errno.EOPNOTSUPP):
        raise RawSourceSealError(
            "atomic RENAME_NOREPLACE unsupported; refusing unsafe fallback"
        )
    raise OSError(number, os.strerror(number), destination_name)


def _publish_create_only(
    *,
    output_root: Path,
    arrays: Mapping[str, np.ndarray],
    manifest: Mapping[str, Any],
) -> None:
    output = Path(os.path.abspath(os.fspath(output_root)))
    if output.name in ("", ".", ".."):
        raise RawSourceSealError("output root must name one fresh directory")
    parent = output.parent
    _assert_no_symlink_ancestors(parent, "output parent")
    parent_stat = os.lstat(parent)
    if stat.S_ISLNK(parent_stat.st_mode) or not stat.S_ISDIR(parent_stat.st_mode):
        raise RawSourceSealError("output parent must be a non-symlink directory")
    npz_payload = _deterministic_npz_bytes(arrays)
    if _hash_bytes(npz_payload) != manifest["npz_sha256"]:
        raise RawSourceSealError("manifest does not bind source NPZ bytes")
    _validate_staged_source(npz_payload=npz_payload, arrays=arrays, manifest=manifest)
    manifest_payload = _canonical_json_bytes(manifest, indent=2)
    directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    directory_flags |= getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    parent_fd = os.open(parent, directory_flags)
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
            raise RawSourceSealError("output parent identity changed while opening")
        os.mkdir(staging_name, mode=0o700, dir_fd=parent_fd)
        created_staging = os.stat(staging_name, dir_fd=parent_fd, follow_symlinks=False)
        if not stat.S_ISDIR(created_staging.st_mode):
            raise RawSourceSealError("created output staging entry is not a directory")
        staging_fd = os.open(staging_name, directory_flags, dir_fd=parent_fd)
        staging_identity = os.fstat(staging_fd)
        if (staging_identity.st_dev, staging_identity.st_ino) != (
            created_staging.st_dev,
            created_staging.st_ino,
        ):
            raise RawSourceSealError("output staging identity changed while opening")
        _write_exclusive_fsync_at(staging_fd, OUTPUT_NPZ_NAME, npz_payload)
        _write_exclusive_fsync_at(staging_fd, OUTPUT_JSON_NAME, manifest_payload)
        os.fsync(staging_fd)
        expected_entries = frozenset({OUTPUT_NPZ_NAME, OUTPUT_JSON_NAME})
        _require_exact_directory_entries(
            staging_fd, expected_entries, "output staging directory"
        )
        _verify_path_identity(parent, opened_parent, "output parent")
        named_staging = os.stat(staging_name, dir_fd=parent_fd, follow_symlinks=False)
        if not stat.S_ISDIR(named_staging.st_mode) or (
            named_staging.st_dev,
            named_staging.st_ino,
        ) != (staging_identity.st_dev, staging_identity.st_ino):
            raise RawSourceSealError("output staging identity changed")
        _rename_noreplace(parent_fd, staging_name, parent_fd, output.name)
        published = True
        os.fsync(parent_fd)
        _verify_path_identity(parent, opened_parent, "output parent")
        output_fd = os.open(output.name, directory_flags, dir_fd=parent_fd)
        published_identity = os.fstat(output_fd)
        if (published_identity.st_dev, published_identity.st_ino) != (
            staging_identity.st_dev,
            staging_identity.st_ino,
        ):
            raise RawSourceSealError("published output directory identity differs")
        _require_exact_directory_entries(
            output_fd, expected_entries, "published output directory"
        )
        if (
            _hash_bytes(
                _read_regular_bytes_at(
                    output_fd,
                    OUTPUT_NPZ_NAME,
                    max_bytes=len(npz_payload),
                    label="published source NPZ",
                )
            )
            != manifest["npz_sha256"]
        ):
            raise RawSourceSealError("published source NPZ differs")
        if _hash_bytes(
            _read_regular_bytes_at(
                output_fd,
                OUTPUT_JSON_NAME,
                max_bytes=len(manifest_payload),
                label="published source JSON",
            )
        ) != _hash_bytes(manifest_payload):
            raise RawSourceSealError("published source JSON differs")
    finally:
        if output_fd >= 0:
            os.close(output_fd)
        if not published and staging_fd >= 0:
            try:
                named = os.stat(staging_name, dir_fd=parent_fd, follow_symlinks=False)
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


def seal_raw_source(
    *, provider_root: Path, schedule_path: Path, output_root: Path
) -> dict[str, Any]:
    """Validate a completed fresh provider and publish one raw source pair."""

    _assert_no_symlink_ancestors(Path(schedule_path), "exact schedule")
    bundle = parse_exact_schedule_bundle(schedule_path)
    if bundle.sha256 != EXPECTED_SCHEDULE_SHA256:
        raise RawSourceSealError("schedule bytes differ from frozen H10 V2")
    provider, frames = _assert_provider_directory(provider_root)
    _validate_frame_directory(frames, bundle)
    provider_identity = os.stat(provider, follow_symlinks=False)
    frames_identity = os.stat(frames, follow_symlinks=False)
    output = Path(os.path.abspath(os.fspath(output_root)))
    if output == provider or provider in output.parents or output in provider.parents:
        raise RawSourceSealError("output and provider roots must not overlap")

    provenance_payload = _read_regular_bytes(
        provider / PROVENANCE_NAME,
        max_bytes=MAX_PROVENANCE_BYTES,
        label="provider provenance",
    )
    provenance_hash = _hash_bytes(provenance_payload)
    provenance = _parse_json_bytes(provenance_payload, "provider provenance")
    provenance_receipt = _validate_provenance(provenance, bundle)

    final_seal_payload = _read_regular_bytes(
        provider / PROVIDER_SEAL_NAME,
        max_bytes=MAX_SEAL_BYTES,
        label="provider final seal",
    )
    final_seal_hash = _hash_bytes(final_seal_payload)
    final_seal = _validate_provider_seal(
        _parse_json_bytes(final_seal_payload, "provider final seal"),
        schedule_sha256=bundle.sha256,
        provenance_sha256=provenance_hash,
    )
    journal_payload = _read_regular_bytes(
        provider / JOURNAL_NAME,
        max_bytes=MAX_JOURNAL_BYTES,
        label="provider journal",
    )
    journal_records = _parse_journal(
        journal_payload, bundle=bundle, final_seal=final_seal
    )
    if [int(row["row_count"]) for row in journal_records] != [
        int(row["row_count"]) for row in provenance_receipt["runtime_rows"]
    ]:
        raise RawSourceSealError("provider journal/runtime row-count ledgers differ")

    input_before = _rehash_snapshot(provider, bundle, schedule_path)
    expected_parsed_hashes = {
        "schedule": bundle.sha256,
        f"provider/{PROVENANCE_NAME}": provenance_hash,
        f"provider/{PROVIDER_SEAL_NAME}": final_seal_hash,
        f"provider/{JOURNAL_NAME}": _hash_bytes(journal_payload),
    }
    if any(
        input_before[name] != digest for name, digest in expected_parsed_hashes.items()
    ):
        raise RawSourceSealError("parsed provider inputs changed before flattening")
    arrays, frame_ledger, empty_identities, scene_row_counts = _build_source_arrays(
        provider_root=provider,
        bundle=bundle,
        journal_records=journal_records,
        runtime_rows=provenance_receipt["runtime_rows"],
    )
    if len(arrays["per_view_scene_index"]) != provenance_receipt["raw_row_count"]:
        raise RawSourceSealError("flattened raw-row count differs from provenance")
    if len(empty_identities) != provenance_receipt["empty_frame_count"]:
        raise RawSourceSealError("flattened empty-frame count differs from provenance")
    membership, membership_per_scene = _freeze_k8(arrays, frame_ledger)
    provider_after, frames_after = _assert_provider_directory(provider)
    _validate_frame_directory(frames_after, bundle)
    provider_after_identity = os.stat(provider_after, follow_symlinks=False)
    frames_after_identity = os.stat(frames_after, follow_symlinks=False)
    if (provider_identity.st_dev, provider_identity.st_ino) != (
        provider_after_identity.st_dev,
        provider_after_identity.st_ino,
    ) or (frames_identity.st_dev, frames_identity.st_ino) != (
        frames_after_identity.st_dev,
        frames_after_identity.st_ino,
    ):
        raise RawSourceSealError("provider directory identity changed while sealing")
    input_after = _rehash_snapshot(provider, bundle, schedule_path)
    if input_before != input_after:
        raise RawSourceSealError("provider or schedule input changed while sealing")

    content_hash = _array_content_sha256(arrays)
    npz_payload = _deterministic_npz_bytes(arrays)
    membership_hash = _numeric_matrix_sha256("k8_membership_identity", membership)
    manifest: dict[str, Any] = {
        "schema": SCHEMA,
        "mode": "sealed_raw_observer_source",
        "create_only": True,
        "association_applied": False,
        "tracking_enabled": False,
        "tracked_artifact_present": False,
        "coordinate_frame": "scannet_world",
        "coordinate_contract_sha256": provenance_receipt["contract_sha256"],
        "scene_ids": list(bundle.scene_order),
        "scene_count": len(bundle.scene_order),
        "exact_frame_count": bundle.valid_frame_count,
        "raw_frame_count": bundle.raw_frame_count,
        "raw_row_count": len(arrays["per_view_scene_index"]),
        "empty_frame_count": len(empty_identities),
        "empty_frame_identities": empty_identities,
        "frame_row_ledger": frame_ledger,
        "scene_row_counts": scene_row_counts,
        "source_instance_id_rule": (
            "global_exact_schedule_index*2048+per_frame_source_row"
        ),
        "provider_bindings": {
            "schedule_sha256": bundle.sha256,
            "run_provenance_sha256": provenance_hash,
            "final_seal_sha256": final_seal_hash,
            "journal_sha256": final_seal["journal_sha256"],
            "provider_contract_sha256": provenance_receipt["contract_sha256"],
            "frozen_assets_sha256": provenance_receipt["assets_sha256"],
            "exact_input_ledger_sha256": provenance_receipt[
                "exact_input_ledger_sha256"
            ],
            "code_hashes": provenance_receipt["code_hashes"],
            "model_hashes": provenance_receipt["model_hashes"],
            "protocol_hashes": provenance_receipt["protocol_hashes"],
        },
        "input_identity": {
            "snapshot_entry_count": len(input_before),
            "snapshot_sha256_before": _snapshot_sha256(input_before),
            "snapshot_sha256_after": _snapshot_sha256(input_after),
            "byte_identical": True,
        },
        "k8": {
            "top_k": TOP_K,
            "sort_key": ["descending_source_score", "source_row", "sealed_npz_row"],
            "identity_columns": [
                "scene_index",
                "frame_id",
                "source_row",
                "source_instance_id",
                "sealed_npz_row",
            ],
            "membership_identities": membership.tolist(),
            "membership_count": len(membership),
            "membership_per_scene": membership_per_scene,
            "membership_sha256": membership_hash,
        },
        "array_names": sorted(arrays),
        "array_content_sha256": content_hash,
        "npz_file": OUTPUT_NPZ_NAME,
        "npz_sha256": _hash_bytes(npz_payload),
    }
    _publish_create_only(output_root=output, arrays=arrays, manifest=manifest)
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider-root", type=Path, required=True)
    parser.add_argument("--schedule", type=Path, default=DEFAULT_SCHEDULE)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        manifest = seal_raw_source(
            provider_root=args.provider_root,
            schedule_path=args.schedule,
            output_root=args.output_root,
        )
    except RawSourceSealError as error:
        raise SystemExit(f"raw-source seal failed: {error}") from error
    print(
        json.dumps(
            {
                "output_root": os.fspath(args.output_root),
                "raw_row_count": manifest["raw_row_count"],
                "empty_frame_count": manifest["empty_frame_count"],
                "k8_membership_count": manifest["k8"]["membership_count"],
                "array_content_sha256": manifest["array_content_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
