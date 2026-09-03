#!/usr/bin/env python3
"""Build or validate the no-GT full-native H10 input manifest.

This preparation-only tool enumerates exactly the ``color``, ``depth`` and
``pose`` directories of the ten frozen H10 scenes.  It hashes inert input
bytes, validates the already frozen 769-frame provider schedule as an exact
subset, and resolves an infinite pose only to the most recent *past* finite
native pose while rejecting NaNs.  It never opens annotations, predictions,
or evaluator inputs, and it does not construct a model or GPU harness.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import stat
import sys
from typing import Any, Mapping, Sequence

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if os.fspath(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, os.fspath(REPOSITORY_ROOT))

from boxfusion.s3r_h10_provider_core import (  # noqa: E402
    EXPECTED_RAW_FRAME_COUNT,
    EXPECTED_SCENE_ORDER,
    EXPECTED_VALID_FRAME_COUNT,
    ExactScheduleBundle,
    SceneSchedule,
    ScheduledFrame,
    parse_exact_schedule_bundle,
)

SCHEMA = "boxfusion.s3r_h10_native_full_stream.v1"
EXPECTED_PROVIDER_SCHEDULE_SHA256 = (
    "1ce565a65510b80d69a0402fe7a40ea89920625f6a81147d42f9232f7a7761e9"
)
DEFAULT_PROVIDER_SCHEDULE = (
    REPOSITORY_ROOT / "docs" / "data" / "S3R_H10_EXACT_SCHEDULE_V2.json"
)
DEFAULT_SCENE_ROOT = REPOSITORY_ROOT / "upstream_clean" / "scannet_readme_frames"
SUGGESTED_OUTPUT = (
    REPOSITORY_ROOT / "docs" / "data" / "S3R_H10_NATIVE_FULL_STREAM_V1.json"
)

MAX_FRAMES_PER_SCENE = 100_000
MAX_TOTAL_FRAMES = 1_000_000
MAX_IMAGE_BYTES = 128 * 1024 * 1024
MAX_MATRIX_BYTES = 1024 * 1024
MAX_MANIFEST_BYTES = 512 * 1024 * 1024
_O_TMPFILE = getattr(os, "O_TMPFILE", 0o20200000)

_HEX = frozenset("0123456789abcdef")
_COLOR_SUFFIXES = frozenset({".jpg"})
_MOUNT_ROLES = ("color", "depth", "pose", "intrinsic")
_POLICY = {
    "enumerated_directories": ["frames/color", "frames/depth", "frames/pose"],
    "fixed_intrinsic_files": [
        "frames/intrinsic/intrinsic_color.txt",
        "frames/intrinsic/intrinsic_depth.txt",
    ],
    "native_color_producer_glob": "frames/color/*.jpg",
    "frame_set_rule": "equal_numeric_id_sets_across_color_depth_pose",
    "frame_order": "strictly_increasing_numeric_frame_id",
    "native_runtime_start": 0,
    "frame_id_mapping": "filename_numeric_id_equals_zero_based_runtime_ordinal",
    "nonfinite_pose": "past_most_recent_valid_native_pose",
    "pose_fallback_trigger": "positive_or_negative_infinity_only",
    "nan_pose": "fail_closed",
    "future_pose_substitution": False,
    "provider_subset": "exact_relative_path_and_sha256_identity",
    "provider_excluded_frame_behavior": (
        "provider_abstains_native_stream_uses_causal_pose"
    ),
    "prediction_deserialization": False,
}

_TOP_KEYS = frozenset(
    {
        "schema",
        "scene_order",
        "scene_count",
        "native_frame_count",
        "native_finite_pose_frame_count",
        "native_nonfinite_pose_frame_count",
        "per_scene_native_frame_count",
        "provider_schedule",
        "policy",
        "native_input_identity_sha256",
        "provider_subset_identity_sha256",
        "role_mount_identity_sha256",
        "scenes",
    }
)
_PROVIDER_KEYS = frozenset(
    {
        "schema",
        "sha256",
        "raw_frame_count",
        "valid_frame_count",
        "excluded_frame_count",
    }
)
_SCENE_KEYS = frozenset(
    {
        "scene_id",
        "native_frame_count",
        "finite_pose_frame_count",
        "nonfinite_pose_frame_count",
        "provider_member_frame_count",
        "provider_abstention_frame_count",
        "intrinsic_color_relpath",
        "intrinsic_color_sha256",
        "intrinsic_depth_relpath",
        "intrinsic_depth_sha256",
        "role_mounts",
        "frame_ids",
        "nonfinite_pose_frame_ids",
        "frames",
    }
)
_MOUNT_KEYS = frozenset(
    {
        "role",
        "relpath",
        "entry_type",
        "link_target_sha256",
        "link_device",
        "link_inode",
        "link_mtime_ns",
        "target_device",
        "target_inode",
        "target_mtime_ns",
        "identity_sha256",
    }
)
_FRAME_KEYS = frozenset(
    {
        "frame_id",
        "color_relpath",
        "color_sha256",
        "depth_relpath",
        "depth_sha256",
        "pose_relpath",
        "pose_sha256",
        "raw_pose_finite",
        "effective_pose_frame_id",
        "effective_pose_relpath",
        "effective_pose_sha256",
        "pose_resolution",
        "intrinsic_color_relpath",
        "intrinsic_color_sha256",
        "provider_status",
    }
)


class NativeManifestError(ValueError):
    """A full-native input or manifest invariant failed closed."""


def _exact_keys(value: Mapping[str, Any], expected: frozenset[str], label: str) -> None:
    actual = frozenset(value)
    if actual != expected:
        raise NativeManifestError(
            f"{label} keys differ: missing={sorted(expected-actual)}, "
            f"extra={sorted(actual-expected)}"
        )


def _duplicate_guard(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise NativeManifestError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _strict_int(value: object, label: str, *, minimum: int = 0) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, int):
        raise NativeManifestError(f"{label} must be an integer")
    if value < minimum:
        raise NativeManifestError(f"{label} must be >= {minimum}")
    return value


def _sha_text(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in _HEX for character in value)
    ):
        raise NativeManifestError(f"{label} must be lowercase SHA-256 hex")
    return value


def _relative_path(value: object, label: str) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        raise NativeManifestError(f"{label} must be a nonempty POSIX relative path")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in ("", ".", "..") for part in path.parts):
        raise NativeManifestError(f"{label} must be a normalized relative path")
    if path.as_posix() != value:
        raise NativeManifestError(f"{label} must be a normalized relative path")
    return value


def _canonical_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("ascii")


def _hash_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _read_regular_bytes(path: Path, *, maximum: int, label: str) -> bytes:
    absolute = Path(os.path.abspath(os.fspath(path)))
    try:
        before = os.lstat(absolute)
    except OSError as error:
        raise NativeManifestError(f"cannot stat {label}: {absolute}") from error
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
        raise NativeManifestError(f"{label} must be a non-symlink regular file")
    if before.st_size > maximum:
        raise NativeManifestError(f"{label} exceeds byte cap")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(absolute, flags)
    try:
        opened = os.fstat(descriptor)
        if (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino):
            raise NativeManifestError(f"{label} identity changed while opening")
        chunks: list[bytes] = []
        remaining = maximum + 1
        while remaining:
            chunk = os.read(descriptor, min(1024 * 1024, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        payload = b"".join(chunks)
        after = os.fstat(descriptor)
        if (
            len(payload) > maximum
            or len(payload) != opened.st_size
            or (after.st_size, after.st_mtime_ns)
            != (opened.st_size, opened.st_mtime_ns)
        ):
            raise NativeManifestError(f"{label} changed while reading")
    finally:
        os.close(descriptor)
    return payload


def _read_regular_bytes_at(
    directory_fd: int, name: str, *, maximum: int, label: str
) -> bytes:
    try:
        before = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
    except OSError as error:
        raise NativeManifestError(f"cannot stat {label}") from error
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
        raise NativeManifestError(f"{label} must be a non-symlink regular file")
    if before.st_size > maximum:
        raise NativeManifestError(f"{label} exceeds byte cap")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(name, flags, dir_fd=directory_fd)
    try:
        opened = os.fstat(descriptor)
        if (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino):
            raise NativeManifestError(f"{label} identity changed while opening")
        chunks: list[bytes] = []
        remaining = maximum + 1
        while remaining:
            chunk = os.read(descriptor, min(1024 * 1024, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        payload = b"".join(chunks)
        after = os.fstat(descriptor)
        if (
            len(payload) > maximum
            or len(payload) != opened.st_size
            or (after.st_size, after.st_mtime_ns)
            != (opened.st_size, opened.st_mtime_ns)
        ):
            raise NativeManifestError(f"{label} changed while reading")
    finally:
        os.close(descriptor)
    return payload


def _pose_is_finite(payload: bytes, label: str) -> bool:
    try:
        rows = [
            list(map(float, line.split(" ")))
            for line in payload.decode("utf-8").splitlines(keepends=True)
        ]
        matrix = np.asarray(rows, dtype=np.float64)
    except (UnicodeDecodeError, ValueError) as error:
        raise NativeManifestError(f"{label} is not a numeric pose matrix") from error
    if matrix.shape != (4, 4):
        raise NativeManifestError(f"{label} must have shape 4x4")
    if np.isnan(matrix).any():
        raise NativeManifestError(f"{label} contains NaN")
    # Match ScannetDataset exactly: only +/-Inf activates its causal fallback.
    return not bool(np.isinf(matrix).any())


def _intrinsic_is_finite(payload: bytes, label: str) -> bool:
    try:
        matrix = np.loadtxt(io.BytesIO(payload), dtype=np.float64)
    except (OSError, ValueError) as error:
        raise NativeManifestError(f"{label} is not a numeric matrix") from error
    if matrix.shape != (4, 4):
        raise NativeManifestError(f"{label} must have shape 4x4")
    return bool(np.isfinite(matrix).all())


def _mount_identity(path: Path, *, role: str, label: str) -> dict[str, Any]:
    """Bind a logical role entry and its directory target without exporting it."""

    logical = Path(os.path.abspath(os.fspath(path)))
    try:
        before = os.lstat(logical)
        target = os.stat(logical, follow_symlinks=True)
    except OSError as error:
        raise NativeManifestError(f"missing {label}: {logical}") from error
    if stat.S_ISLNK(before.st_mode):
        try:
            link_payload = os.fsencode(os.readlink(logical))
        except OSError as error:
            raise NativeManifestError(f"cannot read {label} directory mount") from error
        entry_type = "symlink_directory_mount"
        link_target_sha256: str | None = _hash_bytes(link_payload)
    elif stat.S_ISDIR(before.st_mode):
        entry_type = "directory"
        link_target_sha256 = None
    else:
        raise NativeManifestError(f"{label} must be a directory or directory mount")
    if not stat.S_ISDIR(target.st_mode):
        raise NativeManifestError(f"{label} mount target must be a directory")
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_CLOEXEC", 0)
    descriptor = os.open(logical, flags)
    try:
        opened = os.fstat(descriptor)
        if (opened.st_dev, opened.st_ino) != (target.st_dev, target.st_ino):
            raise NativeManifestError(f"{label} target identity changed while opening")
        after = os.lstat(logical)
        if (
            (after.st_dev, after.st_ino, after.st_mtime_ns)
            != (before.st_dev, before.st_ino, before.st_mtime_ns)
            or (opened.st_dev, opened.st_ino, opened.st_mtime_ns)
            != (target.st_dev, target.st_ino, target.st_mtime_ns)
        ):
            raise NativeManifestError(f"{label} mount identity changed")
    finally:
        os.close(descriptor)
    base: dict[str, Any] = {
        "role": role,
        "relpath": f"frames/{role}",
        "entry_type": entry_type,
        "link_target_sha256": link_target_sha256,
        "link_device": int(before.st_dev),
        "link_inode": int(before.st_ino),
        "link_mtime_ns": int(before.st_mtime_ns),
        "target_device": int(target.st_dev),
        "target_inode": int(target.st_ino),
        "target_mtime_ns": int(target.st_mtime_ns),
    }
    return {**base, "identity_sha256": _hash_bytes(_canonical_bytes(base))}


def _scene_mounts(frames_directory: Path, *, scene_id: str) -> dict[str, Any]:
    return {
        role: _mount_identity(
            frames_directory / role,
            role=role,
            label=f"{scene_id} {role}",
        )
        for role in _MOUNT_ROLES
    }


def _open_role_directory(
    path: Path, *, expected_mount: Mapping[str, Any], scene_id: str, role: str
) -> int:
    before = _mount_identity(
        path, role=role, label=f"{scene_id} {role} directory"
    )
    if before != dict(expected_mount):
        raise NativeManifestError(f"{scene_id} {role} mount identity differs")
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_CLOEXEC", 0)
    descriptor = os.open(path, flags)
    try:
        opened = os.fstat(descriptor)
        if (opened.st_dev, opened.st_ino) != (
            expected_mount["target_device"],
            expected_mount["target_inode"],
        ):
            raise NativeManifestError(
                f"{scene_id} {role} held target identity differs"
            )
        after = _mount_identity(
            path, role=role, label=f"{scene_id} {role} directory"
        )
        if after != before:
            raise NativeManifestError(
                f"{scene_id} {role} mount changed while opening"
            )
    except BaseException:
        os.close(descriptor)
        raise
    return descriptor


def _enumerate_numeric_files(
    directory: Path,
    *,
    role: str,
    scene_id: str,
    expected_mount: Mapping[str, Any] | None = None,
    directory_fd: int | None = None,
) -> dict[int, str]:
    logical = Path(os.path.abspath(os.fspath(directory)))
    before = _mount_identity(
        logical, role=role, label=f"{scene_id} {role} directory"
    )
    if expected_mount is not None and dict(expected_mount) != before:
        raise NativeManifestError(f"{scene_id} {role} mount identity differs")
    opened_here = directory_fd is None
    if directory_fd is None:
        directory_fd = _open_role_directory(
            logical,
            expected_mount=before,
            scene_id=scene_id,
            role=role,
        )
    opened = os.fstat(directory_fd)
    if (opened.st_dev, opened.st_ino) != (
        before["target_device"],
        before["target_inode"],
    ):
        if opened_here:
            os.close(directory_fd)
        raise NativeManifestError(f"{scene_id} {role} held target identity differs")
    result: dict[int, str] = {}
    try:
        try:
            entries = list(os.scandir(directory_fd))
        except OSError as error:
            raise NativeManifestError(
                f"cannot enumerate {scene_id} {role}"
            ) from error
        if not entries or len(entries) > MAX_FRAMES_PER_SCENE:
            raise NativeManifestError(f"{scene_id} {role} entry count is invalid")
        for entry in entries:
            if entry.is_symlink() or not entry.is_file(follow_symlinks=False):
                raise NativeManifestError(
                    f"{scene_id} {role} contains a symlink or non-regular entry"
                )
            suffix = Path(entry.name).suffix
            allowed = (
                _COLOR_SUFFIXES
                if role == "color"
                else {".png" if role == "depth" else ".txt"}
            )
            stem = entry.name[: -len(suffix)] if suffix else ""
            if suffix not in allowed or not stem.isdecimal() or str(int(stem)) != stem:
                raise NativeManifestError(
                    f"{scene_id} {role} contains a noncanonical frame filename"
                )
            frame_id = int(stem)
            if frame_id in result:
                raise NativeManifestError(f"{scene_id} {role} has duplicate frame ID")
            result[frame_id] = entry.name
        after = _mount_identity(
            logical, role=role, label=f"{scene_id} {role} directory"
        )
        if after != before:
            raise NativeManifestError(
                f"{scene_id} {role} mount changed during enumeration"
            )
    finally:
        if opened_here:
            os.close(directory_fd)
    return result


def _identity_digest(rows: Sequence[Mapping[str, Any]]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update(_canonical_bytes(row))
    return digest.hexdigest()


def _provider_identity(scene_id: str, frame: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "scene_id": scene_id,
        "frame_id": frame["frame_id"],
        "intrinsic_color_relpath": frame["intrinsic_color_relpath"],
        "intrinsic_color_sha256": frame["intrinsic_color_sha256"],
        "color_relpath": frame["color_relpath"],
        "color_sha256": frame["color_sha256"],
        "depth_relpath": frame["depth_relpath"],
        "depth_sha256": frame["depth_sha256"],
        "pose_relpath": frame["pose_relpath"],
        "pose_sha256": frame["pose_sha256"],
    }


def _native_identity(scene_id: str, frame: Mapping[str, Any]) -> dict[str, Any]:
    return {"scene_id": scene_id, **dict(frame)}


def _native_depth_intrinsic_identity(
    scene_id: str, *, relative: str, sha256: str
) -> dict[str, Any]:
    return {
        "scene_id": scene_id,
        "input_role": "intrinsic_depth",
        "relpath": relative,
        "sha256": sha256,
    }


def _load_frozen_provider_bundle(path: Path) -> ExactScheduleBundle:
    try:
        bundle = parse_exact_schedule_bundle(path)
    except Exception as error:
        raise NativeManifestError("provider V2 schedule is invalid") from error
    if (
        bundle.sha256 != EXPECTED_PROVIDER_SCHEDULE_SHA256
        or bundle.scene_order != EXPECTED_SCENE_ORDER
        or bundle.raw_frame_count != EXPECTED_RAW_FRAME_COUNT
        or bundle.valid_frame_count != EXPECTED_VALID_FRAME_COUNT
    ):
        raise NativeManifestError("provider V2 schedule differs from frozen identity")
    return bundle


def _require_exact_frozen_provider_bundle(
    bundle: ExactScheduleBundle, path: Path = DEFAULT_PROVIDER_SCHEDULE
) -> None:
    expected = _load_frozen_provider_bundle(path)
    if bundle != expected:
        raise NativeManifestError(
            "provider bundle object differs from parsed frozen H10 V2"
        )


def _provider_maps(
    scene: SceneSchedule,
) -> tuple[dict[int, ScheduledFrame], dict[int, Any]]:
    valid = {frame.frame_id: frame for frame in scene.frames}
    excluded = {frame.frame_id: frame for frame in scene.excluded_frames}
    if set(valid) & set(excluded) or set(valid) | set(excluded) != set(
        scene.raw_frame_ids
    ):
        raise NativeManifestError(f"{scene.scene_id} provider frame partition differs")
    return valid, excluded


def _build_scene_record(
    root: Path, provider_scene: SceneSchedule
) -> tuple[
    dict[str, Any],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    int,
    int,
]:
    """Read one scene through held role descriptors and bind its mount identity."""

    scene_id = provider_scene.scene_id
    scene_directory = root / scene_id
    frames_directory = scene_directory / "frames"
    role_mounts = _scene_mounts(frames_directory, scene_id=scene_id)
    role_fds: dict[str, int] = {}
    try:
        for role in _MOUNT_ROLES:
            role_fds[role] = _open_role_directory(
                frames_directory / role,
                expected_mount=role_mounts[role],
                scene_id=scene_id,
                role=role,
            )

        intrinsic_relative = "frames/intrinsic/intrinsic_color.txt"
        intrinsic_payload = _read_regular_bytes_at(
            role_fds["intrinsic"],
            "intrinsic_color.txt",
            maximum=MAX_MATRIX_BYTES,
            label=f"{scene_id} intrinsic",
        )
        intrinsic_hash = _hash_bytes(intrinsic_payload)
        if (
            intrinsic_relative != provider_scene.intrinsic_color_relpath
            or intrinsic_hash != provider_scene.intrinsic_color_sha256
        ):
            raise NativeManifestError(
                f"{scene_id} intrinsic differs from provider V2 identity"
            )
        if not _intrinsic_is_finite(
            intrinsic_payload, f"{scene_id} color intrinsic"
        ):
            raise NativeManifestError(f"{scene_id} color intrinsic must be finite")
        intrinsic_depth_relative = "frames/intrinsic/intrinsic_depth.txt"
        intrinsic_depth_payload = _read_regular_bytes_at(
            role_fds["intrinsic"],
            "intrinsic_depth.txt",
            maximum=MAX_MATRIX_BYTES,
            label=f"{scene_id} depth intrinsic",
        )
        if not _intrinsic_is_finite(
            intrinsic_depth_payload, f"{scene_id} depth intrinsic"
        ):
            raise NativeManifestError(f"{scene_id} depth intrinsic must be finite")
        intrinsic_depth_hash = _hash_bytes(intrinsic_depth_payload)

        enumerated = {
            role: _enumerate_numeric_files(
                frames_directory / role,
                role=role,
                scene_id=scene_id,
                expected_mount=role_mounts[role],
                directory_fd=role_fds[role],
            )
            for role in ("color", "depth", "pose")
        }
        color_files = enumerated["color"]
        depth_files = enumerated["depth"]
        pose_files = enumerated["pose"]
        if set(color_files) != set(depth_files) or set(color_files) != set(
            pose_files
        ):
            raise NativeManifestError(
                f"{scene_id} color/depth/pose numeric frame ID sets differ"
            )
        frame_ids = sorted(color_files)
        if len(frame_ids) > MAX_FRAMES_PER_SCENE or any(
            left >= right for left, right in zip(frame_ids, frame_ids[1:])
        ):
            raise NativeManifestError(f"{scene_id} native frame order is invalid")
        if frame_ids != list(range(len(frame_ids))):
            raise NativeManifestError(
                f"{scene_id} native filename IDs differ from runtime ordinals"
            )

        provider_valid, provider_excluded = _provider_maps(provider_scene)
        scene_frames: list[dict[str, Any]] = []
        scene_native_identities: list[dict[str, Any]] = [
            _native_depth_intrinsic_identity(
                scene_id,
                relative=intrinsic_depth_relative,
                sha256=intrinsic_depth_hash,
            )
        ]
        scene_provider_identities: list[dict[str, Any]] = []
        nonfinite_ids: list[int] = []
        latest_finite: tuple[int, str, str] | None = None
        provider_member_count = 0
        provider_abstention_count = 0
        for frame_id in frame_ids:
            color_name = color_files[frame_id]
            depth_name = depth_files[frame_id]
            pose_name = pose_files[frame_id]
            color_relative = f"frames/color/{color_name}"
            depth_relative = f"frames/depth/{depth_name}"
            pose_relative = f"frames/pose/{pose_name}"
            color_hash = _hash_bytes(
                _read_regular_bytes_at(
                    role_fds["color"],
                    color_name,
                    maximum=MAX_IMAGE_BYTES,
                    label=f"{scene_id}/{frame_id} color",
                )
            )
            depth_hash = _hash_bytes(
                _read_regular_bytes_at(
                    role_fds["depth"],
                    depth_name,
                    maximum=MAX_IMAGE_BYTES,
                    label=f"{scene_id}/{frame_id} depth",
                )
            )
            pose_payload = _read_regular_bytes_at(
                role_fds["pose"],
                pose_name,
                maximum=MAX_MATRIX_BYTES,
                label=f"{scene_id}/{frame_id} pose",
            )
            pose_hash = _hash_bytes(pose_payload)
            pose_finite = _pose_is_finite(pose_payload, f"{scene_id}/{frame_id} pose")
            if pose_finite:
                latest_finite = (frame_id, pose_relative, pose_hash)
                effective_id, effective_relative, effective_hash = latest_finite
                pose_resolution = "current_finite"
            else:
                nonfinite_ids.append(frame_id)
                if latest_finite is None:
                    raise NativeManifestError(
                        f"{scene_id}/{frame_id} has no past finite pose"
                    )
                effective_id, effective_relative, effective_hash = latest_finite
                pose_resolution = "past_most_recent_valid"

            provider_status = "outside_provider_gap25"
            expected = provider_valid.get(frame_id)
            if expected is not None:
                if not pose_finite:
                    raise NativeManifestError(
                        f"{scene_id}/{frame_id} provider member pose is non-finite"
                    )
                observed_identity = (
                    color_relative,
                    color_hash,
                    depth_relative,
                    depth_hash,
                    pose_relative,
                    pose_hash,
                )
                expected_identity = (
                    expected.color_relpath,
                    expected.color_sha256,
                    expected.depth_relpath,
                    expected.depth_sha256,
                    expected.pose_relpath,
                    expected.pose_sha256,
                )
                if observed_identity != expected_identity:
                    raise NativeManifestError(
                        f"{scene_id}/{frame_id} provider subset identity differs"
                    )
                provider_status = "provider_member"
                provider_member_count += 1
            excluded = provider_excluded.get(frame_id)
            if excluded is not None:
                if pose_finite or (pose_relative, pose_hash) != (
                    excluded.pose_relpath,
                    excluded.pose_sha256,
                ):
                    raise NativeManifestError(
                        f"{scene_id}/{frame_id} provider abstention identity differs"
                    )
                provider_status = "provider_abstain_nonfinite_pose"
                provider_abstention_count += 1

            record = {
                "frame_id": frame_id,
                "color_relpath": color_relative,
                "color_sha256": color_hash,
                "depth_relpath": depth_relative,
                "depth_sha256": depth_hash,
                "pose_relpath": pose_relative,
                "pose_sha256": pose_hash,
                "raw_pose_finite": pose_finite,
                "effective_pose_frame_id": effective_id,
                "effective_pose_relpath": effective_relative,
                "effective_pose_sha256": effective_hash,
                "pose_resolution": pose_resolution,
                "intrinsic_color_relpath": intrinsic_relative,
                "intrinsic_color_sha256": intrinsic_hash,
                "provider_status": provider_status,
            }
            scene_frames.append(record)
            scene_native_identities.append(_native_identity(scene_id, record))
            if provider_status == "provider_member":
                scene_provider_identities.append(_provider_identity(scene_id, record))

        if set(provider_valid) - set(frame_ids) or set(provider_excluded) - set(
            frame_ids
        ):
            raise NativeManifestError(f"{scene_id} provider IDs are absent natively")
        if provider_member_count != len(provider_valid):
            raise NativeManifestError(f"{scene_id} provider subset count differs")
        if provider_abstention_count != len(provider_excluded):
            raise NativeManifestError(f"{scene_id} provider abstention count differs")
        if _scene_mounts(frames_directory, scene_id=scene_id) != role_mounts:
            raise NativeManifestError(f"{scene_id} role mount changed during build")
    finally:
        for descriptor in reversed(tuple(role_fds.values())):
            os.close(descriptor)

    finite_count = len(frame_ids) - len(nonfinite_ids)
    scene_record = {
        "scene_id": scene_id,
        "native_frame_count": len(frame_ids),
        "finite_pose_frame_count": finite_count,
        "nonfinite_pose_frame_count": len(nonfinite_ids),
        "provider_member_frame_count": provider_member_count,
        "provider_abstention_frame_count": provider_abstention_count,
        "intrinsic_color_relpath": intrinsic_relative,
        "intrinsic_color_sha256": intrinsic_hash,
        "intrinsic_depth_relpath": intrinsic_depth_relative,
        "intrinsic_depth_sha256": intrinsic_depth_hash,
        "role_mounts": role_mounts,
        "frame_ids": frame_ids,
        "nonfinite_pose_frame_ids": nonfinite_ids,
        "frames": scene_frames,
    }
    scene_mount_identities = [
        {"scene_id": scene_id, **role_mounts[role]} for role in _MOUNT_ROLES
    ]
    return (
        scene_record,
        scene_native_identities,
        scene_provider_identities,
        scene_mount_identities,
        finite_count,
        len(nonfinite_ids),
    )


def build_native_manifest(
    *,
    scene_root: Path = DEFAULT_SCENE_ROOT,
    provider_bundle: ExactScheduleBundle | None = None,
    provider_schedule_path: Path = DEFAULT_PROVIDER_SCHEDULE,
    require_frozen_provider: bool = True,
    verify_after_build: bool = True,
) -> dict[str, Any]:
    """Enumerate only native sensor roles and construct a strict manifest."""

    if provider_bundle is None:
        provider_bundle = _load_frozen_provider_bundle(provider_schedule_path)
    elif require_frozen_provider:
        _require_exact_frozen_provider_bundle(
            provider_bundle, path=provider_schedule_path
        )

    root = Path(os.path.abspath(os.fspath(scene_root)))
    scene_records: list[dict[str, Any]] = []
    native_identities: list[dict[str, Any]] = []
    provider_identities: list[dict[str, Any]] = []
    mount_identities: list[dict[str, Any]] = []
    per_scene_count: dict[str, int] = {}
    finite_total = 0
    nonfinite_total = 0

    for provider_scene in provider_bundle.scenes:
        (
            scene_record,
            scene_native_identities,
            scene_provider_identities,
            scene_mount_identities,
            finite_count,
            nonfinite_count,
        ) = _build_scene_record(root, provider_scene)
        scene_id = provider_scene.scene_id
        scene_records.append(scene_record)
        native_identities.extend(scene_native_identities)
        provider_identities.extend(scene_provider_identities)
        mount_identities.extend(scene_mount_identities)
        finite_total += finite_count
        nonfinite_total += nonfinite_count
        per_scene_count[scene_id] = scene_record["native_frame_count"]

    native_total = sum(per_scene_count.values())
    if native_total > MAX_TOTAL_FRAMES:
        raise NativeManifestError("full-native H10 frame cap exceeded")
    manifest: dict[str, Any] = {
        "schema": SCHEMA,
        "scene_order": list(provider_bundle.scene_order),
        "scene_count": len(provider_bundle.scenes),
        "native_frame_count": native_total,
        "native_finite_pose_frame_count": finite_total,
        "native_nonfinite_pose_frame_count": nonfinite_total,
        "per_scene_native_frame_count": per_scene_count,
        "provider_schedule": {
            "schema": provider_bundle.schema,
            "sha256": provider_bundle.sha256,
            "raw_frame_count": provider_bundle.raw_frame_count,
            "valid_frame_count": provider_bundle.valid_frame_count,
            "excluded_frame_count": (
                provider_bundle.raw_frame_count - provider_bundle.valid_frame_count
            ),
        },
        "policy": dict(_POLICY),
        "native_input_identity_sha256": _identity_digest(native_identities),
        "provider_subset_identity_sha256": _identity_digest(provider_identities),
        "role_mount_identity_sha256": _identity_digest(mount_identities),
        "scenes": scene_records,
    }
    validated = validate_native_manifest(
        manifest,
        provider_bundle=provider_bundle,
        require_frozen_provider=require_frozen_provider,
    )
    if verify_after_build:
        verify_manifest_files(validated, scene_root=root)
    return validated


def _validate_provider_receipt(
    value: object,
    bundle: ExactScheduleBundle,
    *,
    require_frozen_provider: bool,
) -> None:
    if not isinstance(value, Mapping):
        raise NativeManifestError("provider schedule receipt must be an object")
    _exact_keys(value, _PROVIDER_KEYS, "provider schedule receipt")
    observed = {
        "schema": value["schema"],
        "sha256": _sha_text(value["sha256"], "provider schedule hash"),
        "raw_frame_count": _strict_int(
            value["raw_frame_count"], "provider raw frame count"
        ),
        "valid_frame_count": _strict_int(
            value["valid_frame_count"], "provider valid frame count"
        ),
        "excluded_frame_count": _strict_int(
            value["excluded_frame_count"], "provider excluded frame count"
        ),
    }
    if observed != {
        "schema": bundle.schema,
        "sha256": bundle.sha256,
        "raw_frame_count": bundle.raw_frame_count,
        "valid_frame_count": bundle.valid_frame_count,
        "excluded_frame_count": bundle.raw_frame_count - bundle.valid_frame_count,
    }:
        raise NativeManifestError("provider schedule receipt differs")
    if require_frozen_provider and (
        bundle.sha256 != EXPECTED_PROVIDER_SCHEDULE_SHA256
        or bundle.scene_order != EXPECTED_SCENE_ORDER
        or bundle.raw_frame_count != EXPECTED_RAW_FRAME_COUNT
        or bundle.valid_frame_count != EXPECTED_VALID_FRAME_COUNT
    ):
        raise NativeManifestError("provider schedule is not frozen H10 V2")


def _validate_frame_path(
    relative: str, *, role: str, frame_id: int, label: str
) -> None:
    value = _relative_path(relative, label)
    path = PurePosixPath(value)
    expected_parent = PurePosixPath("frames") / role
    if path.parent != expected_parent or not path.stem.isdecimal():
        raise NativeManifestError(f"{label} does not identify its native role")
    if int(path.stem) != frame_id or str(frame_id) != path.stem:
        raise NativeManifestError(f"{label} frame ID differs")
    suffixes = (
        _COLOR_SUFFIXES if role == "color" else {".png" if role == "depth" else ".txt"}
    )
    if path.suffix not in suffixes:
        raise NativeManifestError(f"{label} suffix differs")


def _validate_mount_record(
    value: object, *, scene_id: str, role: str
) -> dict[str, Any]:
    label = f"{scene_id} {role} mount"
    if not isinstance(value, Mapping):
        raise NativeManifestError(f"{label} must be an object")
    _exact_keys(value, _MOUNT_KEYS, label)
    if value["role"] != role or value["relpath"] != f"frames/{role}":
        raise NativeManifestError(f"{label} role identity differs")
    entry_type = value["entry_type"]
    if entry_type not in ("directory", "symlink_directory_mount"):
        raise NativeManifestError(f"{label} entry type differs")
    link_target = value["link_target_sha256"]
    if entry_type == "directory":
        if link_target is not None:
            raise NativeManifestError(f"{label} direct directory has link hash")
    else:
        _sha_text(link_target, f"{label} link target hash")
    base: dict[str, Any] = {
        "role": role,
        "relpath": f"frames/{role}",
        "entry_type": entry_type,
        "link_target_sha256": link_target,
    }
    for key in (
        "link_device",
        "link_inode",
        "link_mtime_ns",
        "target_device",
        "target_inode",
        "target_mtime_ns",
    ):
        base[key] = _strict_int(value[key], f"{label} {key}")
    if entry_type == "directory" and (
        base["link_device"],
        base["link_inode"],
        base["link_mtime_ns"],
    ) != (
        base["target_device"],
        base["target_inode"],
        base["target_mtime_ns"],
    ):
        raise NativeManifestError(f"{label} direct directory identity differs")
    identity = _sha_text(value["identity_sha256"], f"{label} identity hash")
    if identity != _hash_bytes(_canonical_bytes(base)):
        raise NativeManifestError(f"{label} identity hash differs")
    return {**base, "identity_sha256": identity}


def validate_native_manifest(
    value: Mapping[str, Any],
    *,
    provider_bundle: ExactScheduleBundle,
    require_frozen_provider: bool = True,
) -> dict[str, Any]:
    """Strictly validate structure, causality, counts, and provider identity."""

    if require_frozen_provider:
        _require_exact_frozen_provider_bundle(provider_bundle)
    if not isinstance(value, Mapping):
        raise NativeManifestError("native manifest root must be an object")
    _exact_keys(value, _TOP_KEYS, "native manifest")
    if value["schema"] != SCHEMA:
        raise NativeManifestError("native manifest schema differs")
    if value["scene_order"] != list(provider_bundle.scene_order):
        raise NativeManifestError("native manifest scene order differs")
    if _strict_int(value["scene_count"], "native manifest scene count") != len(
        provider_bundle.scene_order
    ):
        raise NativeManifestError("native manifest scene count differs")
    policy = value["policy"]
    if not isinstance(policy, Mapping):
        raise NativeManifestError("native manifest policy must be an object")
    try:
        exact_policy = _canonical_bytes(dict(policy)) == _canonical_bytes(_POLICY)
    except (TypeError, ValueError) as error:
        raise NativeManifestError("native manifest policy is not strict JSON") from error
    if not exact_policy:
        raise NativeManifestError("native manifest policy differs")
    _validate_provider_receipt(
        value["provider_schedule"],
        provider_bundle,
        require_frozen_provider=require_frozen_provider,
    )

    scenes = value["scenes"]
    if not isinstance(scenes, list) or len(scenes) != len(provider_bundle.scenes):
        raise NativeManifestError("native manifest scene records differ")
    per_scene_value = value["per_scene_native_frame_count"]
    if not isinstance(per_scene_value, Mapping) or set(per_scene_value) != set(
        provider_bundle.scene_order
    ):
        raise NativeManifestError("per-scene native count ledger keys differ")
    per_scene = {
        scene_id: _strict_int(
            per_scene_value[scene_id], f"{scene_id} per-scene native count"
        )
        for scene_id in provider_bundle.scene_order
    }

    native_identities: list[dict[str, Any]] = []
    provider_identities: list[dict[str, Any]] = []
    mount_identities: list[dict[str, Any]] = []
    native_total = 0
    finite_total = 0
    nonfinite_total = 0
    for scene_index, (scene_value, provider_scene) in enumerate(
        zip(scenes, provider_bundle.scenes)
    ):
        if not isinstance(scene_value, Mapping):
            raise NativeManifestError(f"scene {scene_index} must be an object")
        _exact_keys(scene_value, _SCENE_KEYS, f"scene {scene_index}")
        scene_id = provider_scene.scene_id
        if scene_value["scene_id"] != scene_id:
            raise NativeManifestError("native scene order differs")
        intrinsic_relative = _relative_path(
            scene_value["intrinsic_color_relpath"], f"{scene_id} intrinsic path"
        )
        intrinsic_hash = _sha_text(
            scene_value["intrinsic_color_sha256"], f"{scene_id} intrinsic hash"
        )
        if (
            intrinsic_relative != provider_scene.intrinsic_color_relpath
            or intrinsic_hash != provider_scene.intrinsic_color_sha256
        ):
            raise NativeManifestError(
                f"{scene_id} intrinsic differs from provider schedule"
            )
        intrinsic_depth_relative = _relative_path(
            scene_value["intrinsic_depth_relpath"],
            f"{scene_id} depth intrinsic path",
        )
        if intrinsic_depth_relative != "frames/intrinsic/intrinsic_depth.txt":
            raise NativeManifestError(f"{scene_id} depth intrinsic path differs")
        intrinsic_depth_hash = _sha_text(
            scene_value["intrinsic_depth_sha256"],
            f"{scene_id} depth intrinsic hash",
        )
        role_mounts = scene_value["role_mounts"]
        if not isinstance(role_mounts, Mapping) or set(role_mounts) != set(
            _MOUNT_ROLES
        ):
            raise NativeManifestError(f"{scene_id} role mount ledger differs")
        for role in _MOUNT_ROLES:
            mount_record = _validate_mount_record(
                role_mounts[role], scene_id=scene_id, role=role
            )
            mount_identities.append({"scene_id": scene_id, **mount_record})
        native_identities.append(
            _native_depth_intrinsic_identity(
                scene_id,
                relative=intrinsic_depth_relative,
                sha256=intrinsic_depth_hash,
            )
        )
        frame_ids = scene_value["frame_ids"]
        frames = scene_value["frames"]
        if not isinstance(frame_ids, list) or not isinstance(frames, list):
            raise NativeManifestError(f"{scene_id} frame ledgers must be arrays")
        if len(frame_ids) != len(frames) or not frame_ids:
            raise NativeManifestError(f"{scene_id} frame ledger length differs")
        parsed_ids = [
            _strict_int(frame_id, f"{scene_id} frame ID") for frame_id in frame_ids
        ]
        if any(left >= right for left, right in zip(parsed_ids, parsed_ids[1:])):
            raise NativeManifestError(
                f"{scene_id} frame IDs are not strictly increasing"
            )
        if parsed_ids != list(range(len(parsed_ids))):
            raise NativeManifestError(
                f"{scene_id} frame IDs differ from runtime ordinals"
            )
        if len(parsed_ids) > MAX_FRAMES_PER_SCENE:
            raise NativeManifestError(f"{scene_id} frame cap exceeded")

        provider_valid, provider_excluded = _provider_maps(provider_scene)
        latest_finite: tuple[int, str, str] | None = None
        observed_nonfinite: list[int] = []
        member_count = 0
        abstention_count = 0
        for row_index, (frame_id, frame) in enumerate(zip(parsed_ids, frames)):
            if not isinstance(frame, Mapping):
                raise NativeManifestError(
                    f"{scene_id} frame {row_index} is not an object"
                )
            _exact_keys(frame, _FRAME_KEYS, f"{scene_id} frame {row_index}")
            observed_frame_id = _strict_int(
                frame["frame_id"], f"{scene_id} frame row ID"
            )
            if observed_frame_id != frame_id:
                raise NativeManifestError(f"{scene_id} frame ledger ID differs")
            for role in ("color", "depth", "pose"):
                _validate_frame_path(
                    frame[f"{role}_relpath"],
                    role=role,
                    frame_id=frame_id,
                    label=f"{scene_id}/{frame_id} {role} path",
                )
                _sha_text(frame[f"{role}_sha256"], f"{scene_id}/{frame_id} {role} hash")
            if (
                frame["intrinsic_color_relpath"] != intrinsic_relative
                or frame["intrinsic_color_sha256"] != intrinsic_hash
            ):
                raise NativeManifestError(
                    f"{scene_id}/{frame_id} intrinsic identity differs"
                )
            if not isinstance(frame["raw_pose_finite"], bool):
                raise NativeManifestError(
                    f"{scene_id}/{frame_id} raw_pose_finite must be boolean"
                )
            effective_id = _strict_int(
                frame["effective_pose_frame_id"],
                f"{scene_id}/{frame_id} effective pose ID",
            )
            effective_relative = _relative_path(
                frame["effective_pose_relpath"],
                f"{scene_id}/{frame_id} effective pose path",
            )
            effective_hash = _sha_text(
                frame["effective_pose_sha256"],
                f"{scene_id}/{frame_id} effective pose hash",
            )
            if frame["raw_pose_finite"]:
                expected_effective = (
                    frame_id,
                    frame["pose_relpath"],
                    frame["pose_sha256"],
                )
                if (
                    effective_id,
                    effective_relative,
                    effective_hash,
                ) != expected_effective or frame["pose_resolution"] != "current_finite":
                    raise NativeManifestError(
                        f"{scene_id}/{frame_id} finite pose resolution differs"
                    )
                latest_finite = expected_effective
            else:
                observed_nonfinite.append(frame_id)
                if (
                    latest_finite is None
                    or (
                        effective_id,
                        effective_relative,
                        effective_hash,
                    )
                    != latest_finite
                ):
                    raise NativeManifestError(
                        f"{scene_id}/{frame_id} is not past-most-recent-valid"
                    )
                if (
                    effective_id >= frame_id
                    or frame["pose_resolution"] != "past_most_recent_valid"
                ):
                    raise NativeManifestError(
                        f"{scene_id}/{frame_id} uses a future pose substitution"
                    )

            expected_status = "outside_provider_gap25"
            provider_frame = provider_valid.get(frame_id)
            if provider_frame is not None:
                if not frame["raw_pose_finite"]:
                    raise NativeManifestError(
                        f"{scene_id}/{frame_id} provider member is non-finite"
                    )
                expected_identity = (
                    provider_frame.color_relpath,
                    provider_frame.color_sha256,
                    provider_frame.depth_relpath,
                    provider_frame.depth_sha256,
                    provider_frame.pose_relpath,
                    provider_frame.pose_sha256,
                )
                observed_identity = (
                    frame["color_relpath"],
                    frame["color_sha256"],
                    frame["depth_relpath"],
                    frame["depth_sha256"],
                    frame["pose_relpath"],
                    frame["pose_sha256"],
                )
                if observed_identity != expected_identity:
                    raise NativeManifestError(
                        f"{scene_id}/{frame_id} provider subset identity differs"
                    )
                expected_status = "provider_member"
                member_count += 1
                provider_identities.append(_provider_identity(scene_id, frame))
            excluded = provider_excluded.get(frame_id)
            if excluded is not None:
                if frame["raw_pose_finite"] or (
                    frame["pose_relpath"],
                    frame["pose_sha256"],
                ) != (excluded.pose_relpath, excluded.pose_sha256):
                    raise NativeManifestError(
                        f"{scene_id}/{frame_id} provider abstention identity differs"
                    )
                expected_status = "provider_abstain_nonfinite_pose"
                abstention_count += 1
            if frame["provider_status"] != expected_status:
                raise NativeManifestError(
                    f"{scene_id}/{frame_id} provider status differs"
                )
            native_identities.append(_native_identity(scene_id, frame))

        if set(provider_valid) - set(parsed_ids) or set(provider_excluded) - set(
            parsed_ids
        ):
            raise NativeManifestError(f"{scene_id} provider IDs are absent natively")
        nonfinite_value = scene_value["nonfinite_pose_frame_ids"]
        if not isinstance(nonfinite_value, list):
            raise NativeManifestError(
                f"{scene_id} non-finite pose ledger must be an array"
            )
        expected_nonfinite = [
            _strict_int(frame_id, f"{scene_id} non-finite pose frame ID")
            for frame_id in nonfinite_value
        ]
        if expected_nonfinite != observed_nonfinite:
            raise NativeManifestError(f"{scene_id} non-finite pose ledger differs")
        finite_count = len(frames) - len(observed_nonfinite)
        expected_counts = {
            "native_frame_count": len(frames),
            "finite_pose_frame_count": finite_count,
            "nonfinite_pose_frame_count": len(observed_nonfinite),
            "provider_member_frame_count": member_count,
            "provider_abstention_frame_count": abstention_count,
        }
        for key, count in expected_counts.items():
            if _strict_int(scene_value[key], f"{scene_id} {key}") != count:
                raise NativeManifestError(f"{scene_id} scene counts differ")
        if per_scene.get(scene_id) != len(frames):
            raise NativeManifestError(f"{scene_id} per-scene count differs")
        native_total += len(frames)
        finite_total += finite_count
        nonfinite_total += len(observed_nonfinite)

    if native_total > MAX_TOTAL_FRAMES:
        raise NativeManifestError("full-native frame cap exceeded")
    expected_totals = {
        "native_frame_count": native_total,
        "native_finite_pose_frame_count": finite_total,
        "native_nonfinite_pose_frame_count": nonfinite_total,
    }
    for key, count in expected_totals.items():
        if _strict_int(value[key], f"native manifest {key}") != count:
            raise NativeManifestError("full-native manifest totals differ")
    if _sha_text(
        value["native_input_identity_sha256"], "native input identity hash"
    ) != _identity_digest(native_identities):
        raise NativeManifestError("native input identity hash differs")
    if _sha_text(
        value["provider_subset_identity_sha256"], "provider subset identity hash"
    ) != _identity_digest(provider_identities):
        raise NativeManifestError("provider subset identity hash differs")
    if _sha_text(
        value["role_mount_identity_sha256"], "role mount identity hash"
    ) != _identity_digest(mount_identities):
        raise NativeManifestError("role mount identity hash differs")
    if len(provider_identities) != provider_bundle.valid_frame_count:
        raise NativeManifestError("provider subset total differs")
    return dict(value)


def verify_manifest_files(value: Mapping[str, Any], *, scene_root: Path) -> None:
    """Re-enumerate only sensor roles and verify every manifest byte hash."""

    root = Path(os.path.abspath(os.fspath(scene_root)))
    for scene in value["scenes"]:
        scene_id = scene["scene_id"]
        scene_directory = root / scene_id
        frames_directory = scene_directory / "frames"
        role_mounts = scene["role_mounts"]
        if _scene_mounts(frames_directory, scene_id=scene_id) != role_mounts:
            raise NativeManifestError(f"{scene_id} role mount identity changed")
        role_fds: dict[str, int] = {}
        try:
            for role in _MOUNT_ROLES:
                role_fds[role] = _open_role_directory(
                    frames_directory / role,
                    expected_mount=role_mounts[role],
                    scene_id=scene_id,
                    role=role,
                )
            color_intrinsic = _read_regular_bytes_at(
                role_fds["intrinsic"],
                "intrinsic_color.txt",
                maximum=MAX_MATRIX_BYTES,
                label=f"{scene_id} color intrinsic",
            )
            if _hash_bytes(color_intrinsic) != scene["intrinsic_color_sha256"]:
                raise NativeManifestError(
                    f"{scene_id} color intrinsic hash changed"
                )
            if not _intrinsic_is_finite(
                color_intrinsic, f"{scene_id} color intrinsic"
            ):
                raise NativeManifestError(
                    f"{scene_id} color intrinsic must be finite"
                )
            depth_intrinsic = _read_regular_bytes_at(
                role_fds["intrinsic"],
                "intrinsic_depth.txt",
                maximum=MAX_MATRIX_BYTES,
                label=f"{scene_id} depth intrinsic",
            )
            if _hash_bytes(depth_intrinsic) != scene["intrinsic_depth_sha256"]:
                raise NativeManifestError(
                    f"{scene_id} depth intrinsic hash changed"
                )
            if not _intrinsic_is_finite(
                depth_intrinsic, f"{scene_id} depth intrinsic"
            ):
                raise NativeManifestError(
                    f"{scene_id} depth intrinsic must be finite"
                )

            enumerated = {
                role: _enumerate_numeric_files(
                    frames_directory / role,
                    role=role,
                    scene_id=scene_id,
                    expected_mount=role_mounts[role],
                    directory_fd=role_fds[role],
                )
                for role in ("color", "depth", "pose")
            }
            expected_ids = scene["frame_ids"]
            if any(sorted(mapping) != expected_ids for mapping in enumerated.values()):
                raise NativeManifestError(f"{scene_id} native frame set changed")
            for frame in scene["frames"]:
                frame_id = frame["frame_id"]
                for role in ("color", "depth", "pose"):
                    name = enumerated[role][frame_id]
                    if f"frames/{role}/{name}" != frame[f"{role}_relpath"]:
                        raise NativeManifestError(
                            f"{scene_id}/{frame_id} {role} path changed"
                        )
                    maximum = MAX_MATRIX_BYTES if role == "pose" else MAX_IMAGE_BYTES
                    payload = _read_regular_bytes_at(
                        role_fds[role],
                        name,
                        maximum=maximum,
                        label=f"{scene_id}/{frame_id} {role}",
                    )
                    if _hash_bytes(payload) != frame[f"{role}_sha256"]:
                        raise NativeManifestError(
                            f"{scene_id}/{frame_id} {role} hash changed"
                        )
                    if (
                        role == "pose"
                        and _pose_is_finite(payload, f"{scene_id}/{frame_id} pose")
                        is not frame["raw_pose_finite"]
                    ):
                        raise NativeManifestError(
                            f"{scene_id}/{frame_id} pose finiteness changed"
                        )
            if _scene_mounts(frames_directory, scene_id=scene_id) != role_mounts:
                raise NativeManifestError(
                    f"{scene_id} role mount changed during verification"
                )
        finally:
            for descriptor in reversed(tuple(role_fds.values())):
                os.close(descriptor)


def load_and_validate_manifest(
    path: Path,
    *,
    provider_bundle: ExactScheduleBundle,
    scene_root: Path | None = None,
    require_frozen_provider: bool = True,
) -> dict[str, Any]:
    payload = _read_regular_bytes(
        path, maximum=MAX_MANIFEST_BYTES, label="native full-stream manifest"
    )
    try:
        value = json.loads(payload.decode("utf-8"), object_pairs_hook=_duplicate_guard)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise NativeManifestError("native manifest is not strict UTF-8 JSON") from error
    if not isinstance(value, dict):
        raise NativeManifestError("native manifest root must be an object")
    validated = validate_native_manifest(
        value,
        provider_bundle=provider_bundle,
        require_frozen_provider=require_frozen_provider,
    )
    if scene_root is not None:
        verify_manifest_files(validated, scene_root=scene_root)
    return validated


def _open_bound_directory_chain(
    path: Path,
) -> tuple[list[int], list[tuple[str, int, int]]]:
    """Open every path component beneath / so public names remain inode-bound."""

    absolute = Path(os.path.abspath(os.fspath(path)))
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    flags |= getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptors = [os.open("/", flags)]
    bindings: list[tuple[str, int, int]] = []
    try:
        for component in absolute.parts[1:]:
            try:
                before = os.stat(
                    component,
                    dir_fd=descriptors[-1],
                    follow_symlinks=False,
                )
            except OSError as error:
                raise NativeManifestError(
                    "output parent must already exist"
                ) from error
            if stat.S_ISLNK(before.st_mode) or not stat.S_ISDIR(before.st_mode):
                raise NativeManifestError(
                    "output parent path must contain only non-symlink directories"
                )
            child = os.open(component, flags, dir_fd=descriptors[-1])
            opened = os.fstat(child)
            after = os.stat(
                component,
                dir_fd=descriptors[-1],
                follow_symlinks=False,
            )
            identity = (int(before.st_dev), int(before.st_ino))
            if (
                (opened.st_dev, opened.st_ino) != identity
                or (after.st_dev, after.st_ino) != identity
            ):
                os.close(child)
                raise NativeManifestError("output parent identity changed while opening")
            descriptors.append(child)
            bindings.append((component, *identity))
    except BaseException:
        for descriptor in reversed(descriptors):
            os.close(descriptor)
        raise
    return descriptors, bindings


def _verify_directory_chain(
    descriptors: Sequence[int], bindings: Sequence[tuple[str, int, int]]
) -> None:
    if len(descriptors) != len(bindings) + 1:
        raise NativeManifestError("output parent binding ledger differs")
    for index, (component, device, inode) in enumerate(bindings):
        try:
            named = os.stat(
                component,
                dir_fd=descriptors[index],
                follow_symlinks=False,
            )
            opened = os.fstat(descriptors[index + 1])
        except OSError as error:
            raise NativeManifestError("output parent public path changed") from error
        if (
            stat.S_ISLNK(named.st_mode)
            or not stat.S_ISDIR(named.st_mode)
            or (named.st_dev, named.st_ino) != (device, inode)
            or (opened.st_dev, opened.st_ino) != (device, inode)
        ):
            raise NativeManifestError("output parent public path identity changed")


def _read_descriptor(descriptor: int, maximum: int) -> bytes:
    os.lseek(descriptor, 0, os.SEEK_SET)
    chunks: list[bytes] = []
    remaining = maximum + 1
    while remaining:
        chunk = os.read(descriptor, min(1024 * 1024, remaining))
        if not chunk:
            break
        chunks.append(chunk)
        remaining -= len(chunk)
    payload = b"".join(chunks)
    if len(payload) > maximum:
        raise NativeManifestError("published manifest exceeds byte cap")
    return payload


def _publish_create_only(path: Path, payload: bytes) -> None:
    if not isinstance(payload, bytes):
        raise NativeManifestError("manifest payload must be bytes")
    if len(payload) > MAX_MANIFEST_BYTES:
        raise NativeManifestError("manifest payload exceeds byte cap")
    output = Path(os.path.abspath(os.fspath(path)))
    parent = output.parent
    if output.name in ("", ".", ".."):
        raise NativeManifestError("output must name one manifest file")

    directory_fds, directory_bindings = _open_bound_directory_chain(parent)
    parent_fd = directory_fds[-1]
    descriptor = -1
    held_identity: tuple[int, int] | None = None
    published = False
    complete = False
    try:
        _verify_directory_chain(directory_fds, directory_bindings)
        file_flags = os.O_RDWR | _O_TMPFILE | getattr(os, "O_CLOEXEC", 0)
        try:
            descriptor = os.open(".", file_flags, 0o600, dir_fd=parent_fd)
        except OSError as error:
            raise NativeManifestError(
                "output filesystem does not support anonymous create-only publication"
            ) from error
        created = os.fstat(descriptor)
        if not stat.S_ISREG(created.st_mode):
            raise NativeManifestError("anonymous manifest must be a regular file")
        held_identity = (int(created.st_dev), int(created.st_ino))
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("short manifest write")
            view = view[written:]
        os.fsync(descriptor)
        held = os.fstat(descriptor)
        if (held.st_dev, held.st_ino) != held_identity:
            raise NativeManifestError("anonymous manifest held identity changed")
        if held.st_size != len(payload):
            raise NativeManifestError("anonymous manifest size differs")
        _verify_directory_chain(directory_fds, directory_bindings)
        try:
            os.link(
                f"/proc/self/fd/{descriptor}",
                output.name,
                dst_dir_fd=parent_fd,
                follow_symlinks=True,
            )
        except FileExistsError as error:
            raise NativeManifestError(
                "output exists or lost create-only race"
            ) from error
        published = True
        # Cleanup authority is limited to the inode we created, never whichever
        # inode an attacker may later substitute at the public name.
        named_output = os.stat(output.name, dir_fd=parent_fd, follow_symlinks=False)
        if (
            not stat.S_ISREG(named_output.st_mode)
            or (named_output.st_dev, named_output.st_ino) != held_identity
            or named_output.st_size != len(payload)
        ):
            raise NativeManifestError("published manifest identity differs")
        _verify_directory_chain(directory_fds, directory_bindings)

        held_payload = _read_descriptor(descriptor, MAX_MANIFEST_BYTES)
        published_payload = _read_regular_bytes_at(
            parent_fd,
            output.name,
            maximum=MAX_MANIFEST_BYTES,
            label="published native full-stream manifest",
        )
        expected_hash = _hash_bytes(payload)
        if (
            held_payload != payload
            or published_payload != payload
            or _hash_bytes(held_payload) != expected_hash
            or _hash_bytes(published_payload) != expected_hash
        ):
            raise NativeManifestError("published manifest readback differs")
        after_read = os.fstat(descriptor)
        if (
            (after_read.st_dev, after_read.st_ino) != held_identity
            or after_read.st_size != len(payload)
            or after_read.st_mtime_ns != held.st_mtime_ns
        ):
            raise NativeManifestError("published manifest changed during readback")
        os.fsync(parent_fd)
        _verify_directory_chain(directory_fds, directory_bindings)
        final_output = os.stat(
            output.name, dir_fd=parent_fd, follow_symlinks=False
        )
        if (final_output.st_dev, final_output.st_ino) != held_identity:
            raise NativeManifestError("published manifest final identity differs")
        _verify_directory_chain(directory_fds, directory_bindings)
        complete = True
    finally:
        if not complete and published and held_identity is not None:
            try:
                current_output = os.stat(
                    output.name, dir_fd=parent_fd, follow_symlinks=False
                )
                if (current_output.st_dev, current_output.st_ino) == held_identity:
                    os.unlink(output.name, dir_fd=parent_fd)
                    os.fsync(parent_fd)
            except FileNotFoundError:
                pass
        if descriptor >= 0:
            os.close(descriptor)
        for directory_fd in reversed(directory_fds):
            os.close(directory_fd)


def _check_output_location(output: Path, scene_root: Path) -> None:
    """Reject every lexical or resolved overlap with the immutable input tree."""

    output_absolute = Path(os.path.abspath(os.fspath(output)))
    root_absolute = Path(os.path.abspath(os.fspath(scene_root)))
    output_resolved = Path(os.path.realpath(os.fspath(output_absolute)))
    root_resolved = Path(os.path.realpath(os.fspath(root_absolute)))
    for output_path, root_path in (
        (output_absolute, root_absolute),
        (output_resolved, root_resolved),
    ):
        if (
            output_path == root_path
            or root_path in output_path.parents
            or output_path in root_path.parents
        ):
            raise NativeManifestError(
                "output and scene root must be disjoint path trees"
            )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    operation = parser.add_mutually_exclusive_group(required=True)
    operation.add_argument("--output", type=Path)
    operation.add_argument("--validate", type=Path)
    parser.add_argument("--scene-root", type=Path, default=DEFAULT_SCENE_ROOT)
    parser.add_argument(
        "--provider-schedule", type=Path, default=DEFAULT_PROVIDER_SCHEDULE
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.output is not None:
        _check_output_location(args.output, args.scene_root)
    provider_bundle = _load_frozen_provider_bundle(args.provider_schedule)
    if args.validate is not None:
        value = load_and_validate_manifest(
            args.validate,
            provider_bundle=provider_bundle,
            scene_root=args.scene_root,
        )
        payload = _read_regular_bytes(
            args.validate,
            maximum=MAX_MANIFEST_BYTES,
            label="native full-stream manifest",
        )
        print(
            json.dumps(
                {
                    "manifest_sha256": _hash_bytes(payload),
                    "native_frame_count": value["native_frame_count"],
                    "native_nonfinite_pose_frame_count": value[
                        "native_nonfinite_pose_frame_count"
                    ],
                    "provider_subset_frame_count": provider_bundle.valid_frame_count,
                },
                sort_keys=True,
            )
        )
        return 0

    value = build_native_manifest(
        scene_root=args.scene_root,
        provider_bundle=provider_bundle,
    )
    payload = _canonical_bytes(value)
    _publish_create_only(args.output, payload)
    print(
        json.dumps(
            {
                "output": os.fspath(args.output),
                "manifest_sha256": _hash_bytes(payload),
                "native_frame_count": value["native_frame_count"],
                "native_nonfinite_pose_frame_count": value[
                    "native_nonfinite_pose_frame_count"
                ],
                "provider_subset_frame_count": provider_bundle.valid_frame_count,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
