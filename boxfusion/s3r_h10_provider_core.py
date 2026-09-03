"""Fail-closed control plane for the fresh S3R H10 Boxer provider.

The module intentionally has no model, image, annotation, semantic, scoring,
or evaluation dependency.  It validates the frozen exact-frame schedule and
persists each provider frame before the caller is allowed to advance.  Frame
files are numeric, pickle-free NPZ containers published with a hard-link
create-if-absent operation.

There is deliberately no resume mode.  A crash, an existing output, a race,
or a durability error leaves the run unusable and requires a fresh namespace.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from io import BytesIO
import json
from math import isfinite
from numbers import Integral, Real
import os
from pathlib import Path, PurePosixPath
import secrets
import stat
from typing import Any, Mapping, Sequence

import numpy as np

SCHEDULE_SCHEMA = "boxfusion.s3r_h10_exact_schedule.v1"
JOURNAL_SCHEMA = "boxfusion.s3r_h10_frame_journal.v1"
SEAL_SCHEMA = "boxfusion.s3r_h10_provider_seal.v1"

HOLDOUT_LIST_SHA256 = "8965d0534ed3028f85d8b0ea7227d348a6faa1387b858ddf42c3183bd9ebdf90"
EXPECTED_SCENE_ORDER = (
    "scene0304_00",
    "scene0412_00",
    "scene0019_00",
    "scene0575_00",
    "scene0426_00",
    "scene0426_03",
    "scene0578_00",
    "scene0665_00",
    "scene0050_01",
    "scene0025_00",
)
EXPECTED_RAW_COUNTS = (70, 94, 23, 117, 92, 49, 59, 41, 148, 77)
EXPECTED_SOURCE_MANIFEST_SHA256 = {
    "scene0304_00": "eb4f86b14376a32bc34977185f9845caafb3fe7e888de7775902428e9db4a9d7",
    "scene0412_00": "c5cd532f63c14708fee04a5a8ac21e3d61b24f342fe2eed19642302ff92ad900",
    "scene0019_00": "b10a9eabbe56e532ddfd316286a06e248aa3896c09593378c3a9a5d56fd3b49a",
    "scene0575_00": "a9aa923fd30b02b63c9e5c48909f7abb9e4ae4628511c067d5fbe187ad38939a",
    "scene0426_00": "fdfa46d8c4398b3a8138cd8b8578c357da2d354681c165262db4f92bc18a3bf9",
    "scene0426_03": "b4c3a6b35690ced5cb659e307e59441a3e8acfceafa144abaa0897621573371d",
    "scene0578_00": "c302e6aed65ceb5ee84217052553be1a608c364c2acfac6629f446abbb970a6c",
    "scene0665_00": "c2b1bcb36b2cdd61feb22bd5a088ee08410a62845f4403468ba253309dc9c5ae",
    "scene0050_01": "0fae3f64e16e8b5ac5816d839149c4c45c14f6f397962d01bb1b3056e4733d81",
    "scene0025_00": "9c11a6b2d30c244c7d16907e17b24b8676e05cda9962e9fdd507c42566a871ad",
}
EXPECTED_FORMAL_T05_SHA256 = {
    "scene0304_00": "8f118eee95237037a4f7f9e27ce1acd713054fd25c05812bcdabb7affd581b48",
    "scene0412_00": "504af4e0dcbc7663af72dc85d7d8aafe2031174ee58729dbe6a5cbf63c5fa0cb",
    "scene0019_00": "9a595dc17968aa45f83b283e6c2e48b7907acebfd20aa212f2ce8e44ce30faa8",
    "scene0575_00": "f0949f9e83829f97aebff64a7df3776c3dc8bd7b24e201e336da024ed642043b",
    "scene0426_00": "96890e541d1d5515f966ec4cfdb0040ff21b90f60a5733769b4703b106430d04",
    "scene0426_03": "029f7222bb658ecdd1cc53f414a6bbfb535b2c325f138346f70b9f5ebf1f287d",
    "scene0578_00": "ceca961e81edc33336df7774a9dac9f3801357ce94e7d1f23f7dbec1377720d0",
    "scene0665_00": "377fa1b9da5ecc9f2097eb781fcccbd953ae5c443adf2e28df339f64ccaa4d5d",
    "scene0050_01": "09ec901dfd47618e0357ab2aa07fdff0848518b6864ae5c8dcf3e8f3431592ef",
    "scene0025_00": "ea34fedf8abe64f17366b22e4a1ab188a7d2cf99d8680954c4a29af4e4f7eeb3",
}
EXPECTED_RAW_FRAME_COUNT = 770
EXPECTED_VALID_FRAME_COUNT = 769
EXCLUDED_SCENE_ID = "scene0412_00"
EXCLUDED_FRAME_ID = 2325
EXCLUDED_REASON = "nonfinite_pose"
EXCLUDED_POSE_SHA256 = (
    "8981acaa5e7d946d6031737ac0d55d4fe29ceda0a7fb10241ad4bae2e84bf467"
)
MAX_RAW_ROWS_PER_FRAME = 2048
MAX_RUN_PROVENANCE_BYTES = 8 * 1024 * 1024
PRECOMMIT_RUNTIME_SEMANTICS = "provider_compute_before_frame_transaction_commit"

_TOP_LEVEL_KEYS = frozenset(
    {
        "schema",
        "scene_order",
        "raw_frame_count",
        "valid_frame_count",
        "holdout_list_sha256",
        "provider",
        "scenes",
    }
)
_PROVIDER_KEYS = frozenset(
    {
        "annotation_path",
        "track",
        "directory_enumeration",
        "prefetch",
        "persist_before_advance",
    }
)
_SCENE_KEYS = frozenset(
    {
        "scene_id",
        "source_schedule_manifest_relpath",
        "source_schedule_manifest_sha256",
        "formal_t05_relpath",
        "formal_t05_sha256",
        "intrinsic_color_relpath",
        "intrinsic_color_sha256",
        "raw_frame_ids",
        "valid_frame_ids",
        "excluded_frames",
        "frames",
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
    }
)
_EXCLUDED_FRAME_KEYS = frozenset({"frame_id", "reason", "pose_relpath", "pose_sha256"})
_NPZ_KEYS = frozenset(
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
_HEX_DIGITS = frozenset("0123456789abcdef")
_MAX_SCHEDULE_BYTES = 32 * 1024 * 1024
_MAX_FRAME_ID = (1 << 31) - 1


class ScheduleValidationError(ValueError):
    """The exact schedule is malformed or does not describe frozen H10."""


class FrameTransactionError(RuntimeError):
    """A frame transaction violated ordering, exclusivity, or durability."""


def _exact_keys(value: Mapping[str, Any], expected: frozenset[str], where: str) -> None:
    actual = frozenset(value)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ScheduleValidationError(
            f"{where} keys differ (missing={missing}, extra={extra})"
        )


def _strict_int(value: object, where: str, *, maximum: int = _MAX_FRAME_ID) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise ScheduleValidationError(f"{where} must be an integer")
    result = int(value)
    if result < 0 or result > maximum:
        raise ScheduleValidationError(f"{where} is outside [0,{maximum}]")
    return result


def _sha256_text(value: object, where: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in _HEX_DIGITS for character in value)
    ):
        raise ScheduleValidationError(f"{where} must be lowercase SHA-256 hex")
    return value


def _relative_path(value: object, where: str) -> str:
    if not isinstance(value, str) or not value or "\x00" in value or "\\" in value:
        raise ScheduleValidationError(f"{where} must be a safe POSIX relative path")
    path = PurePosixPath(value)
    if path.is_absolute() or value != path.as_posix():
        raise ScheduleValidationError(f"{where} must be a normalized relative path")
    if any(part in ("", ".", "..") for part in path.parts):
        raise ScheduleValidationError(f"{where} must not contain dot components")
    return value


def _strict_id_list(value: object, where: str) -> tuple[int, ...]:
    if not isinstance(value, list):
        raise ScheduleValidationError(f"{where} must be a JSON array")
    result = tuple(
        _strict_int(item, f"{where}[{index}]") for index, item in enumerate(value)
    )
    if any(right <= left for left, right in zip(result, result[1:])):
        raise ScheduleValidationError(f"{where} must be strictly increasing and unique")
    return result


@dataclass(frozen=True)
class ScheduledFrame:
    frame_id: int
    color_relpath: str
    color_sha256: str
    depth_relpath: str
    depth_sha256: str
    pose_relpath: str
    pose_sha256: str

    @property
    def input_sha256(self) -> tuple[str, str, str]:
        return (self.color_sha256, self.depth_sha256, self.pose_sha256)


@dataclass(frozen=True)
class ExcludedFrame:
    frame_id: int
    reason: str
    pose_relpath: str
    pose_sha256: str


@dataclass(frozen=True)
class SceneSchedule:
    scene_id: str
    source_schedule_manifest_relpath: str
    source_schedule_manifest_sha256: str
    formal_t05_relpath: str
    formal_t05_sha256: str
    intrinsic_color_relpath: str
    intrinsic_color_sha256: str
    raw_frame_ids: tuple[int, ...]
    valid_frame_ids: tuple[int, ...]
    excluded_frames: tuple[ExcludedFrame, ...]
    frames: tuple[ScheduledFrame, ...]


@dataclass(frozen=True)
class ExactScheduleBundle:
    schema: str
    scene_order: tuple[str, ...]
    raw_frame_count: int
    valid_frame_count: int
    holdout_list_sha256: str
    scenes: tuple[SceneSchedule, ...]
    sha256: str

    @property
    def ordered_frames(self) -> tuple[tuple[SceneSchedule, ScheduledFrame], ...]:
        return tuple((scene, frame) for scene in self.scenes for frame in scene.frames)


def _reject_duplicate_json_keys(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ScheduleValidationError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _read_schedule_file(path: Path) -> bytes:
    absolute = Path(os.path.abspath(os.fspath(path)))
    try:
        metadata = os.lstat(absolute)
    except OSError as error:
        raise ScheduleValidationError(
            f"cannot stat schedule bundle: {error}"
        ) from error
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise ScheduleValidationError(
            "schedule bundle must be a non-symlink regular file"
        )
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(absolute, flags)
    try:
        opened = os.fstat(descriptor)
        if (
            not stat.S_ISREG(opened.st_mode)
            or (opened.st_dev, opened.st_ino) != (metadata.st_dev, metadata.st_ino)
            or opened.st_size > _MAX_SCHEDULE_BYTES
        ):
            raise ScheduleValidationError("schedule bundle identity or size changed")
        chunks: list[bytes] = []
        remaining = _MAX_SCHEDULE_BYTES + 1
        while remaining:
            chunk = os.read(descriptor, min(1024 * 1024, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        payload = b"".join(chunks)
        if len(payload) > _MAX_SCHEDULE_BYTES:
            raise ScheduleValidationError("schedule bundle is too large")
        if len(payload) != opened.st_size:
            raise ScheduleValidationError("schedule bundle changed while reading")
        return payload
    finally:
        os.close(descriptor)


def _parse_frame(value: object, where: str) -> ScheduledFrame:
    if not isinstance(value, Mapping):
        raise ScheduleValidationError(f"{where} must be an object")
    _exact_keys(value, _FRAME_KEYS, where)
    return ScheduledFrame(
        frame_id=_strict_int(value["frame_id"], f"{where}.frame_id"),
        color_relpath=_relative_path(value["color_relpath"], f"{where}.color_relpath"),
        color_sha256=_sha256_text(value["color_sha256"], f"{where}.color_sha256"),
        depth_relpath=_relative_path(value["depth_relpath"], f"{where}.depth_relpath"),
        depth_sha256=_sha256_text(value["depth_sha256"], f"{where}.depth_sha256"),
        pose_relpath=_relative_path(value["pose_relpath"], f"{where}.pose_relpath"),
        pose_sha256=_sha256_text(value["pose_sha256"], f"{where}.pose_sha256"),
    )


def _parse_excluded_frame(value: object, where: str) -> ExcludedFrame:
    if not isinstance(value, Mapping):
        raise ScheduleValidationError(f"{where} must be an object")
    _exact_keys(value, _EXCLUDED_FRAME_KEYS, where)
    reason = value["reason"]
    if reason != EXCLUDED_REASON:
        raise ScheduleValidationError(f"{where}.reason must be {EXCLUDED_REASON!r}")
    return ExcludedFrame(
        frame_id=_strict_int(value["frame_id"], f"{where}.frame_id"),
        reason=reason,
        pose_relpath=_relative_path(value["pose_relpath"], f"{where}.pose_relpath"),
        pose_sha256=_sha256_text(value["pose_sha256"], f"{where}.pose_sha256"),
    )


def _parse_scene(
    value: object, expected_scene: str, expected_raw_count: int, index: int
) -> SceneSchedule:
    where = f"scenes[{index}]"
    if not isinstance(value, Mapping):
        raise ScheduleValidationError(f"{where} must be an object")
    _exact_keys(value, _SCENE_KEYS, where)
    if value["scene_id"] != expected_scene:
        raise ScheduleValidationError(f"{where}.scene_id is out of frozen order")
    raw_ids = _strict_id_list(value["raw_frame_ids"], f"{where}.raw_frame_ids")
    valid_ids = _strict_id_list(value["valid_frame_ids"], f"{where}.valid_frame_ids")
    if len(raw_ids) != expected_raw_count:
        raise ScheduleValidationError(f"{where} raw frame count differs")
    if raw_ids != tuple(range(0, expected_raw_count * 25, 25)):
        raise ScheduleValidationError(
            f"{where} raw IDs differ from the sealed gap-25 schedule"
        )
    if not isinstance(value["excluded_frames"], list):
        raise ScheduleValidationError(f"{where}.excluded_frames must be a JSON array")
    excluded = tuple(
        _parse_excluded_frame(item, f"{where}.excluded_frames[{item_index}]")
        for item_index, item in enumerate(value["excluded_frames"])
    )
    expected_excluded = (
        (EXCLUDED_FRAME_ID,) if expected_scene == EXCLUDED_SCENE_ID else ()
    )
    if tuple(item.frame_id for item in excluded) != expected_excluded:
        raise ScheduleValidationError(f"{where} excluded frame set differs")
    if valid_ids != tuple(
        frame_id for frame_id in raw_ids if frame_id not in expected_excluded
    ):
        raise ScheduleValidationError(
            f"{where} valid IDs are not raw IDs minus exclusions"
        )
    if not isinstance(value["frames"], list):
        raise ScheduleValidationError(f"{where}.frames must be a JSON array")
    frames = tuple(
        _parse_frame(item, f"{where}.frames[{frame_index}]")
        for frame_index, item in enumerate(value["frames"])
    )
    if tuple(frame.frame_id for frame in frames) != valid_ids:
        raise ScheduleValidationError(
            f"{where}.frames do not exactly match valid_frame_ids"
        )
    source_relpath = _relative_path(
        value["source_schedule_manifest_relpath"],
        f"{where}.source_schedule_manifest_relpath",
    )
    source_hash = _sha256_text(
        value["source_schedule_manifest_sha256"],
        f"{where}.source_schedule_manifest_sha256",
    )
    formal_relpath = _relative_path(
        value["formal_t05_relpath"], f"{where}.formal_t05_relpath"
    )
    formal_hash = _sha256_text(value["formal_t05_sha256"], f"{where}.formal_t05_sha256")
    intrinsic_relpath = _relative_path(
        value["intrinsic_color_relpath"], f"{where}.intrinsic_color_relpath"
    )
    if source_relpath != f"{expected_scene}/manifest.json":
        raise ScheduleValidationError(f"{where} source manifest path differs")
    if source_hash != EXPECTED_SOURCE_MANIFEST_SHA256[expected_scene]:
        raise ScheduleValidationError(f"{where} source manifest hash differs")
    if formal_relpath != (
        f"results/scannet_topk_fusion_score05/{expected_scene}_boxes.pkl"
    ):
        raise ScheduleValidationError(f"{where} formal T05 path differs")
    if formal_hash != EXPECTED_FORMAL_T05_SHA256[expected_scene]:
        raise ScheduleValidationError(f"{where} formal T05 hash differs")
    if intrinsic_relpath != "frames/intrinsic/intrinsic_color.txt":
        raise ScheduleValidationError(f"{where} intrinsic path differs")
    for frame in frames:
        expected_paths = (
            f"frames/color/{frame.frame_id}.jpg",
            f"frames/depth/{frame.frame_id}.png",
            f"frames/pose/{frame.frame_id}.txt",
        )
        actual_paths = (frame.color_relpath, frame.depth_relpath, frame.pose_relpath)
        if actual_paths != expected_paths:
            raise ScheduleValidationError(
                f"{where} frame {frame.frame_id} input paths differ"
            )
    if excluded:
        item = excluded[0]
        if (
            item.pose_relpath != f"frames/pose/{EXCLUDED_FRAME_ID}.txt"
            or item.pose_sha256 != EXCLUDED_POSE_SHA256
        ):
            raise ScheduleValidationError(f"{where} excluded pose binding differs")
    return SceneSchedule(
        scene_id=expected_scene,
        source_schedule_manifest_relpath=source_relpath,
        source_schedule_manifest_sha256=source_hash,
        formal_t05_relpath=formal_relpath,
        formal_t05_sha256=formal_hash,
        intrinsic_color_relpath=intrinsic_relpath,
        intrinsic_color_sha256=_sha256_text(
            value["intrinsic_color_sha256"], f"{where}.intrinsic_color_sha256"
        ),
        raw_frame_ids=raw_ids,
        valid_frame_ids=valid_ids,
        excluded_frames=excluded,
        frames=frames,
    )


def parse_exact_schedule_bundle(
    source: os.PathLike[str] | str | Mapping[str, Any],
) -> ExactScheduleBundle:
    """Parse and fully validate the one allowed H10 exact-frame bundle."""

    if isinstance(source, Mapping):
        value = source
        try:
            payload = json.dumps(
                source, sort_keys=True, separators=(",", ":"), ensure_ascii=True
            ).encode("ascii")
        except (TypeError, ValueError) as error:
            raise ScheduleValidationError(
                "schedule mapping is not JSON-safe"
            ) from error
    else:
        payload = _read_schedule_file(Path(source))
        try:
            value = json.loads(
                payload.decode("utf-8"), object_pairs_hook=_reject_duplicate_json_keys
            )
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ScheduleValidationError(
                "schedule bundle is not strict UTF-8 JSON"
            ) from error
    if not isinstance(value, Mapping):
        raise ScheduleValidationError("schedule bundle root must be an object")
    _exact_keys(value, _TOP_LEVEL_KEYS, "bundle")
    if value["schema"] != SCHEDULE_SCHEMA:
        raise ScheduleValidationError("schedule schema differs")
    scene_order_value = value["scene_order"]
    if (
        not isinstance(scene_order_value, list)
        or tuple(scene_order_value) != EXPECTED_SCENE_ORDER
    ):
        raise ScheduleValidationError("scene_order differs from frozen H10 order")
    raw_count = _strict_int(value["raw_frame_count"], "raw_frame_count", maximum=10_000)
    valid_count = _strict_int(
        value["valid_frame_count"], "valid_frame_count", maximum=10_000
    )
    if (
        raw_count != EXPECTED_RAW_FRAME_COUNT
        or valid_count != EXPECTED_VALID_FRAME_COUNT
    ):
        raise ScheduleValidationError(
            "bundle frame counts differ from 770 raw / 769 valid"
        )
    holdout_hash = _sha256_text(value["holdout_list_sha256"], "holdout_list_sha256")
    if holdout_hash != HOLDOUT_LIST_SHA256:
        raise ScheduleValidationError("holdout list hash differs")
    provider = value["provider"]
    if not isinstance(provider, Mapping):
        raise ScheduleValidationError("provider must be an object")
    _exact_keys(provider, _PROVIDER_KEYS, "provider")
    expected_provider = {
        "annotation_path": None,
        "track": False,
        "directory_enumeration": False,
        "prefetch": False,
        "persist_before_advance": True,
    }
    if dict(provider) != expected_provider:
        raise ScheduleValidationError("provider safety contract differs")
    scenes_value = value["scenes"]
    if not isinstance(scenes_value, list) or len(scenes_value) != len(
        EXPECTED_SCENE_ORDER
    ):
        raise ScheduleValidationError("scenes must contain exactly ten entries")
    scenes = tuple(
        _parse_scene(item, scene_id, raw_scene_count, index)
        for index, (item, scene_id, raw_scene_count) in enumerate(
            zip(scenes_value, EXPECTED_SCENE_ORDER, EXPECTED_RAW_COUNTS)
        )
    )
    if sum(len(scene.raw_frame_ids) for scene in scenes) != raw_count:
        raise ScheduleValidationError("scene raw counts do not equal bundle count")
    if sum(len(scene.valid_frame_ids) for scene in scenes) != valid_count:
        raise ScheduleValidationError("scene valid counts do not equal bundle count")
    return ExactScheduleBundle(
        schema=SCHEDULE_SCHEMA,
        scene_order=EXPECTED_SCENE_ORDER,
        raw_frame_count=raw_count,
        valid_frame_count=valid_count,
        holdout_list_sha256=holdout_hash,
        scenes=scenes,
        sha256=sha256(payload).hexdigest(),
    )


@dataclass(frozen=True)
class FrameToken:
    """Opaque, single-use authorization for the exact next scheduled frame."""

    scene_id: str
    frame_id: int
    _nonce: object


@dataclass(frozen=True)
class FrameCommit:
    scene_id: str
    frame_id: int
    relative_path: str
    row_count: int
    file_sha256: str
    runtime_seconds: float


def _canonical_json_line(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        + "\n"
    ).encode("ascii")


def _write_all(descriptor: int, payload: bytes) -> None:
    view = memoryview(payload)
    while view:
        written = os.write(descriptor, view)
        if written <= 0:
            raise OSError("short write")
        view = view[written:]


def _fsync(descriptor: int, role: str) -> None:
    """Named durability hook; ``role`` exists so tests can audit ordering."""

    del role
    os.fsync(descriptor)


def _assert_no_symlink_ancestors(path: Path) -> None:
    absolute = Path(os.path.abspath(os.fspath(path)))
    current = Path(absolute.anchor)
    for component in absolute.parts[1:]:
        current /= component
        try:
            metadata = os.lstat(current)
        except FileNotFoundError:
            return
        if stat.S_ISLNK(metadata.st_mode):
            raise FrameTransactionError(f"symlink path component rejected: {current}")


def _open_directory_nofollow(path: Path) -> int:
    _assert_no_symlink_ancestors(path)
    metadata = os.lstat(path)
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        raise FrameTransactionError(f"not a non-symlink directory: {path}")
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    opened = os.fstat(descriptor)
    if (opened.st_dev, opened.st_ino) != (metadata.st_dev, metadata.st_ino):
        os.close(descriptor)
        raise FrameTransactionError(f"directory identity changed: {path}")
    return descriptor


def _open_created_directory_at(parent_fd: int, name: str, label: str) -> int:
    """Open a just-created child and prove its directory entry was not swapped."""

    try:
        created = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    except OSError as error:
        raise FrameTransactionError(f"cannot stat created {label}: {error}") from error
    if not stat.S_ISDIR(created.st_mode) or stat.S_ISLNK(created.st_mode):
        raise FrameTransactionError(f"created {label} is not a real directory")
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    flags |= getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(name, flags, dir_fd=parent_fd)
    except OSError as error:
        raise FrameTransactionError(f"cannot open created {label}: {error}") from error
    opened = os.fstat(descriptor)
    if (
        not stat.S_ISDIR(opened.st_mode)
        or (opened.st_dev, opened.st_ino) != (created.st_dev, created.st_ino)
    ):
        os.close(descriptor)
        raise FrameTransactionError(f"created {label} identity changed before open")
    return descriptor


def _validated_matrix(
    value: object, shape_tail: tuple[int, ...], name: str
) -> np.ndarray:
    try:
        array = np.asarray(value)
    except (TypeError, ValueError) as error:
        raise FrameTransactionError(f"{name} is not a numeric array") from error
    if array.ndim != 1 + len(shape_tail) or array.shape[1:] != shape_tail:
        raise FrameTransactionError(
            f"{name} must have shape (N,{','.join(map(str, shape_tail))})"
        )
    if array.dtype.kind not in "fiu" or array.dtype.kind == "b":
        raise FrameTransactionError(f"{name} must be numeric")
    result = np.array(array, dtype=np.float64, order="C", copy=True)
    if not np.isfinite(result).all():
        raise FrameTransactionError(f"{name} contains non-finite values")
    return result


def _validated_vector(value: object, name: str) -> np.ndarray:
    try:
        array = np.asarray(value)
    except (TypeError, ValueError) as error:
        raise FrameTransactionError(f"{name} is not a numeric array") from error
    if array.ndim != 1 or array.dtype.kind not in "fiu" or array.dtype.kind == "b":
        raise FrameTransactionError(f"{name} must have shape (N,) and be numeric")
    result = np.array(array, dtype=np.float64, order="C", copy=True)
    if not np.isfinite(result).all():
        raise FrameTransactionError(f"{name} contains non-finite values")
    return result


def _validated_source_rows(value: object, count: int) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != 1 or array.dtype.kind not in "iu" or array.dtype.kind == "b":
        raise FrameTransactionError(
            "source_row must be an integer array with shape (N,)"
        )
    if len(array) != count:
        raise FrameTransactionError("source_row length differs")
    result = np.array(array, dtype=np.int64, order="C", copy=True)
    if not np.array_equal(result, np.arange(count, dtype=np.int64)):
        raise FrameTransactionError("source_row must equal 0..N-1 without duplicates")
    return result


def _npz_payload(
    *,
    center: object,
    extent: object,
    quaternion: object,
    score: object,
    source_row: object,
    input_hashes: Sequence[str],
    runtime_seconds: float,
) -> tuple[bytes, int]:
    centers = _validated_matrix(center, (3,), "center")
    extents = _validated_matrix(extent, (3,), "extent")
    quaternions = _validated_matrix(quaternion, (4,), "quaternion")
    scores = _validated_vector(score, "score")
    count = len(centers)
    if count > MAX_RAW_ROWS_PER_FRAME:
        raise FrameTransactionError(
            f"raw row cap exceeded ({count}>{MAX_RAW_ROWS_PER_FRAME})"
        )
    if len(extents) != count or len(quaternions) != count or len(scores) != count:
        raise FrameTransactionError("provider arrays have different row counts")
    rows = _validated_source_rows(source_row, count)
    if np.any(extents <= 0.0):
        raise FrameTransactionError("extent must be strictly positive")
    if np.any(scores < 0.0) or np.any(scores > 1.0):
        raise FrameTransactionError("score must be in [0,1]")
    if count and np.any(np.einsum("ij,ij->i", quaternions, quaternions) <= 1e-12):
        raise FrameTransactionError(
            "quaternion squared norm must be greater than 1e-12"
        )
    packed_hashes = np.stack(
        [np.frombuffer(bytes.fromhex(item), dtype=np.uint8) for item in input_hashes],
        axis=0,
    )
    output = BytesIO()
    np.savez(
        output,
        center=centers,
        extent=extents,
        quaternion=quaternions,
        score=scores,
        source_row=rows,
        input_sha256=packed_hashes,
        runtime_seconds=np.asarray([runtime_seconds], dtype=np.float64),
    )
    return output.getvalue(), count


class FrameTransaction:
    """Create-only, ordered per-frame persistence for one fresh H10 run."""

    def __init__(
        self, output_root: os.PathLike[str] | str, bundle: ExactScheduleBundle
    ):
        if not isinstance(bundle, ExactScheduleBundle):
            raise FrameTransactionError("bundle must be a parsed ExactScheduleBundle")
        self.output_root = Path(os.path.abspath(os.fspath(output_root)))
        if self.output_root.name in ("", ".", ".."):
            raise FrameTransactionError("output root must name one fresh directory")
        self.bundle = bundle
        self._ordered = bundle.ordered_frames
        self._next_index = 0
        self._pending: FrameToken | None = None
        self._poisoned = False
        self._sealed = False
        self._closed = False
        self._journal_hash = sha256()
        self._frame_hash = sha256()
        self._total_runtime_seconds = 0.0
        self._grandparent_fd = -1
        self._parent_fd = -1
        self._root_fd = -1
        self._frames_fd = -1
        self._journal_fd = -1
        self._grandparent_identity: tuple[int, int] | None = None
        self._parent_identity: tuple[int, int] | None = None
        self._root_identity: tuple[int, int] | None = None
        self._frames_identity: tuple[int, int] | None = None
        self._journal_identity: tuple[int, int] | None = None
        self._provenance_identity: tuple[int, int] | None = None
        self._run_provenance_sha256: str | None = None
        self._create_namespace()

    def _create_namespace(self) -> None:
        parent = self.output_root.parent
        if parent.name in ("", ".", ".."):
            raise FrameTransactionError(
                "output parent must be a named directory below a grandparent"
            )
        grandparent = parent.parent
        self._grandparent_fd = _open_directory_nofollow(grandparent)
        try:
            parent_entry = os.stat(
                parent.name,
                dir_fd=self._grandparent_fd,
                follow_symlinks=False,
            )
            self._parent_fd = _open_directory_nofollow(parent)
            parent_opened = os.fstat(self._parent_fd)
            grandparent_opened = os.fstat(self._grandparent_fd)
        except BaseException:
            self.close()
            raise
        if (
            not stat.S_ISDIR(parent_entry.st_mode)
            or (parent_entry.st_dev, parent_entry.st_ino)
            != (parent_opened.st_dev, parent_opened.st_ino)
        ):
            self.close()
            raise FrameTransactionError(
                "output parent identity changed before transaction creation"
            )
        self._grandparent_identity = (
            grandparent_opened.st_dev,
            grandparent_opened.st_ino,
        )
        self._parent_identity = (parent_opened.st_dev, parent_opened.st_ino)
        try:
            os.mkdir(self.output_root.name, mode=0o700, dir_fd=self._parent_fd)
        except FileExistsError as error:
            self.close()
            raise FrameTransactionError("output root must be create-only") from error
        try:
            _fsync(self._parent_fd, "output-parent-directory")
            self._root_fd = _open_created_directory_at(
                self._parent_fd, self.output_root.name, "output root"
            )
            root_stat = os.fstat(self._root_fd)
            self._root_identity = (root_stat.st_dev, root_stat.st_ino)
            os.mkdir("frames", mode=0o700, dir_fd=self._root_fd)
            self._frames_fd = _open_created_directory_at(
                self._root_fd, "frames", "frames directory"
            )
            frames_stat = os.fstat(self._frames_fd)
            self._frames_identity = (frames_stat.st_dev, frames_stat.st_ino)
            journal_flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_APPEND
            journal_flags |= getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
            self._journal_fd = os.open(
                "frames.journal.jsonl", journal_flags, 0o600, dir_fd=self._root_fd
            )
            journal_stat = os.fstat(self._journal_fd)
            self._journal_identity = (journal_stat.st_dev, journal_stat.st_ino)
            header = _canonical_json_line(
                {
                    "schema": JOURNAL_SCHEMA,
                    "schedule_sha256": self.bundle.sha256,
                    "expected_frame_count": self.bundle.valid_frame_count,
                }
            )
            _write_all(self._journal_fd, header)
            _fsync(self._journal_fd, "journal")
            self._journal_hash.update(header)
            _fsync(self._root_fd, "output-root-directory")
        except BaseException:
            self._poisoned = True
            self.close()
            raise

    def _verify_namespace_identity(self) -> None:
        """Prove held descriptors still name the published output namespace."""

        try:
            parent_entry = os.stat(
                self.output_root.parent.name,
                dir_fd=self._grandparent_fd,
                follow_symlinks=False,
            )
            root_entry = os.stat(
                self.output_root.name,
                dir_fd=self._parent_fd,
                follow_symlinks=False,
            )
            frames_entry = os.stat(
                "frames", dir_fd=self._root_fd, follow_symlinks=False
            )
            journal_entry = os.stat(
                "frames.journal.jsonl",
                dir_fd=self._root_fd,
                follow_symlinks=False,
            )
            grandparent_opened = os.fstat(self._grandparent_fd)
            parent_opened = os.fstat(self._parent_fd)
            root_opened = os.fstat(self._root_fd)
            frames_opened = os.fstat(self._frames_fd)
            journal_opened = os.fstat(self._journal_fd)
            provenance_entry = (
                os.stat(
                    "RUN_PROVENANCE.json",
                    dir_fd=self._root_fd,
                    follow_symlinks=False,
                )
                if self._provenance_identity is not None
                else None
            )
        except OSError as error:
            self._poisoned = True
            raise FrameTransactionError(
                f"output namespace cannot be verified: {error}"
            ) from error
        grandparent_identity = (
            grandparent_opened.st_dev,
            grandparent_opened.st_ino,
        )
        parent_identity = (parent_opened.st_dev, parent_opened.st_ino)
        root_identity = (root_opened.st_dev, root_opened.st_ino)
        frames_identity = (frames_opened.st_dev, frames_opened.st_ino)
        journal_identity = (journal_opened.st_dev, journal_opened.st_ino)
        if (
            not stat.S_ISDIR(parent_entry.st_mode)
            or not stat.S_ISDIR(root_entry.st_mode)
            or not stat.S_ISDIR(frames_entry.st_mode)
            or grandparent_identity != self._grandparent_identity
            or parent_identity != self._parent_identity
            or root_identity != self._root_identity
            or frames_identity != self._frames_identity
            or journal_identity != self._journal_identity
            or (parent_entry.st_dev, parent_entry.st_ino) != parent_identity
            or (root_entry.st_dev, root_entry.st_ino) != root_identity
            or (frames_entry.st_dev, frames_entry.st_ino) != frames_identity
            or (journal_entry.st_dev, journal_entry.st_ino) != journal_identity
            or (
                provenance_entry is not None
                and (provenance_entry.st_dev, provenance_entry.st_ino)
                != self._provenance_identity
            )
        ):
            self._poisoned = True
            raise FrameTransactionError("output namespace identity changed")

    @property
    def completed_frame_count(self) -> int:
        return self._next_index

    @property
    def poisoned(self) -> bool:
        return self._poisoned

    @property
    def sealed(self) -> bool:
        return self._sealed

    def _require_live(self) -> None:
        if self._closed:
            raise FrameTransactionError("transaction is closed")
        if self._poisoned:
            raise FrameTransactionError("transaction is poisoned")
        if self._sealed:
            raise FrameTransactionError("transaction is already sealed")

    def begin(self, scene_id: str, frame_id: int) -> FrameToken:
        """Authorize exactly the next frame; a pending frame blocks advancement."""

        self._require_live()
        self._verify_namespace_identity()
        if self._pending is not None:
            self._poisoned = True
            raise FrameTransactionError("previous frame is not durably committed")
        if self._next_index >= len(self._ordered):
            self._poisoned = True
            raise FrameTransactionError("all scheduled frames are already committed")
        expected_scene, expected_frame = self._ordered[self._next_index]
        if (
            scene_id != expected_scene.scene_id
            or isinstance(frame_id, (bool, np.bool_))
            or not isinstance(frame_id, Integral)
            or int(frame_id) != expected_frame.frame_id
        ):
            self._poisoned = True
            raise FrameTransactionError(
                f"off-order frame; expected {expected_scene.scene_id}/{expected_frame.frame_id}"
            )
        token = FrameToken(expected_scene.scene_id, expected_frame.frame_id, object())
        self._pending = token
        return token

    def _publish_frame(self, final_name: str, payload: bytes) -> None:
        self._verify_namespace_identity()
        temporary_name = f".{final_name}.{secrets.token_hex(16)}.tmp"
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        flags |= getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
        descriptor = -1
        linked = False
        try:
            try:
                os.stat(final_name, dir_fd=self._frames_fd, follow_symlinks=False)
            except FileNotFoundError:
                pass
            else:
                raise FrameTransactionError(
                    f"frame output already exists: {final_name}"
                )
            descriptor = os.open(temporary_name, flags, 0o600, dir_fd=self._frames_fd)
            _write_all(descriptor, payload)
            _fsync(descriptor, "frame-file")
            os.close(descriptor)
            descriptor = -1
            os.link(
                temporary_name,
                final_name,
                src_dir_fd=self._frames_fd,
                dst_dir_fd=self._frames_fd,
                follow_symlinks=False,
            )
            linked = True
            os.unlink(temporary_name, dir_fd=self._frames_fd)
            _fsync(self._frames_fd, "frame-directory")
        except BaseException:
            if descriptor >= 0:
                os.close(descriptor)
            try:
                os.unlink(temporary_name, dir_fd=self._frames_fd)
            except FileNotFoundError:
                pass
            # A linked final is intentionally retained: deleting it could erase
            # a competing writer's file after a race.  This run is poisoned.
            self._poisoned = True
            raise
        if not linked:  # pragma: no cover - defensive control-flow assertion
            self._poisoned = True
            raise FrameTransactionError("frame publication did not link output")
        self._verify_namespace_identity()

    def _append_journal(self, record: Mapping[str, Any]) -> bytes:
        line = _canonical_json_line(record)
        _write_all(self._journal_fd, line)
        _fsync(self._journal_fd, "journal")
        self._journal_hash.update(line)
        return line

    def commit(
        self,
        token: FrameToken,
        *,
        center: object,
        extent: object,
        quaternion: object,
        score: object,
        source_row: object,
        runtime_seconds: object,
    ) -> FrameCommit:
        """Persist one frame, including an empty frame.

        ``runtime_seconds`` is only provider computation elapsed before this
        method is entered.  A runner must measure begin-to-commit-return
        latency separately when accounting for NPZ, directory, and journal
        durability overhead, then bind that ledger through the run-provenance
        hash required by :meth:`seal`.
        """

        self._require_live()
        if self._pending is None or token is not self._pending:
            self._poisoned = True
            raise FrameTransactionError("commit requires the exact pending frame token")
        if isinstance(runtime_seconds, (bool, np.bool_)) or not isinstance(
            runtime_seconds, Real
        ):
            self._poisoned = True
            raise FrameTransactionError(
                "runtime_seconds must be a finite nonnegative number"
            )
        runtime = float(runtime_seconds)
        if not isfinite(runtime) or runtime < 0.0:
            self._poisoned = True
            raise FrameTransactionError(
                "runtime_seconds must be a finite nonnegative number"
            )
        scene, frame = self._ordered[self._next_index]
        try:
            hashes = (scene.intrinsic_color_sha256, *frame.input_sha256)
            payload, row_count = _npz_payload(
                center=center,
                extent=extent,
                quaternion=quaternion,
                score=score,
                source_row=source_row,
                input_hashes=hashes,
                runtime_seconds=runtime,
            )
            file_hash = sha256(payload).hexdigest()
            final_name = f"{scene.scene_id}.{frame.frame_id:06d}.npz"
            self._publish_frame(final_name, payload)
            relative_path = f"frames/{final_name}"
            record = {
                "scene_id": scene.scene_id,
                "frame_id": frame.frame_id,
                "relative_path": relative_path,
                "row_count": row_count,
                "file_sha256": file_hash,
                "input_sha256": {
                    "intrinsic_color": hashes[0],
                    "color": hashes[1],
                    "depth": hashes[2],
                    "pose": hashes[3],
                },
                "runtime_seconds": runtime,
                "runtime_seconds_semantics": PRECOMMIT_RUNTIME_SEMANTICS,
            }
            journal_line = self._append_journal(record)
            self._verify_namespace_identity()
        except BaseException:
            self._poisoned = True
            self._pending = None
            raise
        self._frame_hash.update(journal_line)
        self._total_runtime_seconds += runtime
        self._next_index += 1
        self._pending = None
        return FrameCommit(
            scene_id=scene.scene_id,
            frame_id=frame.frame_id,
            relative_path=relative_path,
            row_count=row_count,
            file_sha256=file_hash,
            runtime_seconds=runtime,
        )

    def publish_run_provenance(self, payload: bytes) -> str:
        """Publish the one provenance file through the held output-root fd."""

        self._require_live()
        self._verify_namespace_identity()
        if self._pending is not None or self._next_index != len(self._ordered):
            raise FrameTransactionError(
                "run provenance requires every frame to be durably committed"
            )
        if self._run_provenance_sha256 is not None:
            self._poisoned = True
            raise FrameTransactionError("run provenance is already published")
        if not isinstance(payload, bytes) or not payload:
            self._poisoned = True
            raise FrameTransactionError("run provenance payload must be nonempty bytes")
        if len(payload) > MAX_RUN_PROVENANCE_BYTES:
            self._poisoned = True
            raise FrameTransactionError("run provenance payload exceeds byte cap")

        final_name = "RUN_PROVENANCE.json"
        temporary_name = f".{final_name}.{secrets.token_hex(16)}.tmp"
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        flags |= getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
        descriptor = -1
        linked = False
        try:
            try:
                os.stat(final_name, dir_fd=self._root_fd, follow_symlinks=False)
            except FileNotFoundError:
                pass
            else:
                raise FrameTransactionError("run provenance already exists")
            descriptor = os.open(
                temporary_name, flags, 0o600, dir_fd=self._root_fd
            )
            _write_all(descriptor, payload)
            _fsync(descriptor, "run-provenance-file")
            os.close(descriptor)
            descriptor = -1
            os.link(
                temporary_name,
                final_name,
                src_dir_fd=self._root_fd,
                dst_dir_fd=self._root_fd,
                follow_symlinks=False,
            )
            linked = True
            os.unlink(temporary_name, dir_fd=self._root_fd)
            _fsync(self._root_fd, "output-root-directory")
            provenance_stat = os.stat(
                final_name, dir_fd=self._root_fd, follow_symlinks=False
            )
            self._provenance_identity = (
                provenance_stat.st_dev,
                provenance_stat.st_ino,
            )
            self._verify_namespace_identity()
        except BaseException:
            if descriptor >= 0:
                os.close(descriptor)
            try:
                os.unlink(temporary_name, dir_fd=self._root_fd)
            except FileNotFoundError:
                pass
            self._poisoned = True
            raise
        if not linked:  # pragma: no cover - defensive control-flow assertion
            self._poisoned = True
            raise FrameTransactionError("run provenance was not published")
        digest = sha256(payload).hexdigest()
        self._run_provenance_sha256 = digest
        return digest

    def seal(self, *, run_provenance_sha256: object) -> dict[str, Any]:
        """Publish the final seal, but only after every exact frame committed."""

        self._require_live()
        if (
            not isinstance(run_provenance_sha256, str)
            or len(run_provenance_sha256) != 64
            or any(character not in _HEX_DIGITS for character in run_provenance_sha256)
        ):
            self._poisoned = True
            raise FrameTransactionError(
                "run_provenance_sha256 must be lowercase SHA-256 hex"
            )
        if self._pending is not None:
            self._poisoned = True
            raise FrameTransactionError("cannot seal with a pending frame")
        if self._next_index != len(self._ordered):
            raise FrameTransactionError(
                f"cannot seal {self._next_index}/{len(self._ordered)} frames"
            )
        if (
            self._run_provenance_sha256 is None
            or run_provenance_sha256 != self._run_provenance_sha256
        ):
            self._poisoned = True
            raise FrameTransactionError(
                "seal hash must match provenance published through this transaction"
            )
        self._verify_namespace_identity()
        record = {
            "schema": SEAL_SCHEMA,
            "schedule_sha256": self.bundle.sha256,
            "run_provenance_sha256": run_provenance_sha256,
            "completed_frame_count": self._next_index,
            "journal_sha256": self._journal_hash.hexdigest(),
            "frame_record_sha256": self._frame_hash.hexdigest(),
            "total_runtime_seconds": self._total_runtime_seconds,
            "runtime_seconds_semantics": PRECOMMIT_RUNTIME_SEMANTICS,
        }
        payload = (
            json.dumps(
                record, sort_keys=True, separators=(",", ":"), ensure_ascii=True
            ).encode("ascii")
            + b"\n"
        )
        temporary_name = f".FINAL_SEAL.json.{secrets.token_hex(16)}.tmp"
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        flags |= getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
        descriptor = -1
        try:
            try:
                os.stat("FINAL_SEAL.json", dir_fd=self._root_fd, follow_symlinks=False)
            except FileNotFoundError:
                pass
            else:
                raise FrameTransactionError("final seal already exists")
            descriptor = os.open(temporary_name, flags, 0o600, dir_fd=self._root_fd)
            _write_all(descriptor, payload)
            _fsync(descriptor, "seal-file")
            os.close(descriptor)
            descriptor = -1
            os.link(
                temporary_name,
                "FINAL_SEAL.json",
                src_dir_fd=self._root_fd,
                dst_dir_fd=self._root_fd,
                follow_symlinks=False,
            )
            os.unlink(temporary_name, dir_fd=self._root_fd)
            _fsync(self._root_fd, "output-root-directory")
            self._verify_namespace_identity()
        except BaseException:
            if descriptor >= 0:
                os.close(descriptor)
            try:
                os.unlink(temporary_name, dir_fd=self._root_fd)
            except FileNotFoundError:
                pass
            self._poisoned = True
            raise
        self._sealed = True
        return record

    def close(self) -> None:
        if self._closed:
            return
        for name in (
            "_journal_fd",
            "_frames_fd",
            "_root_fd",
            "_parent_fd",
            "_grandparent_fd",
        ):
            descriptor = getattr(self, name, -1)
            if descriptor >= 0:
                try:
                    os.close(descriptor)
                finally:
                    setattr(self, name, -1)
        self._closed = True

    def __enter__(self) -> "FrameTransaction":
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()


__all__ = [
    "EXPECTED_RAW_FRAME_COUNT",
    "EXPECTED_SCENE_ORDER",
    "EXPECTED_VALID_FRAME_COUNT",
    "ExactScheduleBundle",
    "ExcludedFrame",
    "FrameCommit",
    "FrameToken",
    "FrameTransaction",
    "FrameTransactionError",
    "HOLDOUT_LIST_SHA256",
    "JOURNAL_SCHEMA",
    "MAX_RAW_ROWS_PER_FRAME",
    "PRECOMMIT_RUNTIME_SEMANTICS",
    "SCHEDULE_SCHEMA",
    "SEAL_SCHEMA",
    "SceneSchedule",
    "ScheduleValidationError",
    "ScheduledFrame",
    "parse_exact_schedule_bundle",
]
