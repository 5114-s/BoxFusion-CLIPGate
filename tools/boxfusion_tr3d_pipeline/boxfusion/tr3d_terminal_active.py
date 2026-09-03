"""Fail-closed terminal-p100 TR3D R3 geometry overlay.

This module is deliberately dependency-light.  It consumes an immutable
TR3D parent cache produced from the exact BoxFusion keyframes that have
already been observed at the terminal timestamp, associates those proposals
with the *current* post-processed B6 + Selective-Boxer rows, and changes only
the eight box corners.  Labels, detector scores, row order, and row count are
outside this API and therefore cannot be changed.

The cache-replay provider is an engineering bridge used to validate the
same-run insertion point.  It is not a live-TR3D latency claim; diagnostics
report the cached model runtime separately from the actual replay overhead.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
from io import BytesIO
import json
import math
import os
from pathlib import Path
import pickle
import re
import tempfile
import time
from typing import Any, Mapping, Sequence

import numpy as np


TR3D_RESIDUAL_CACHE_SCHEMA = "boxfusion.tr3d_residual_cache.v1"
TERMINAL_ACTIVE_SCHEMA = "boxfusion.tr3d_r3_terminal_active.v1"
TERMINAL_DIAGNOSTIC_SCHEMA = "boxfusion.tr3d_r3_terminal_active_scene.v1"
PREFIX_SCHEMA = "boxfusion.tr3d.trajectory_prefix.v1"
FROZEN_NEAR_IOU = 0.15
FROZEN_CLOCK_POLICY = "g0_post_frame_tail_guard_v1"
FROZEN_POSE_POLICY = "previous_valid_inf_only_v1"
FROZEN_TIMESTAMP_SEMANTICS = "zero_based_scannet_dataset_index"
FROZEN_CHECKPOINT_SHA256 = (
    "a484fd79093aa3004f4d2984e7ad8763c5d5cbec7edd04172d775643a8436448"
)
FROZEN_CONFIG_SHA256 = (
    "709b66d9e244ef4385dfa9bbc89895ad06c78534f9d14bb7149b687fd58da785"
)

_SCENE_RE = re.compile(r"^scene[0-9]{4}_[0-9]{2}$")
_PREFIX_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def prediction_payload(
    corners_world: Any,
    scores: Any,
) -> list[list[tuple[int, np.ndarray, float]]]:
    """Build the canonical BoxFusion payload, including zero predictions.

    The same-run baseline and post-R3 result both use this path so dtype or
    memory-layout differences cannot be mistaken for geometry mutations.
    """

    corners = np.asarray(corners_world)
    confidence = np.asarray(scores)
    if corners.size == 0 and corners.shape in {(0,), (0, 8, 3)}:
        corners = np.empty((0, 8, 3), dtype=np.float32)
    if confidence.size == 0 and confidence.shape in {(0,), (0, 1)}:
        confidence = np.empty((0,), dtype=np.float32)
    if corners.ndim != 3 or corners.shape[1:] != (8, 3):
        raise ValueError("prediction corners must have shape [N,8,3]")
    if confidence.ndim != 1 or confidence.shape != (len(corners),):
        raise ValueError("prediction scores must have shape [N]")
    if not np.issubdtype(corners.dtype, np.number):
        raise ValueError("prediction corners must be numeric")
    if not np.issubdtype(confidence.dtype, np.number):
        raise ValueError("prediction scores must be numeric")
    if not np.isfinite(corners).all() or not np.isfinite(confidence).all():
        raise ValueError("prediction arrays must contain only finite values")

    canonical_corners = np.ascontiguousarray(corners, dtype=np.float32)
    canonical_scores = np.ascontiguousarray(confidence, dtype=np.float32)
    return [[
        (
            0,
            np.array(
                canonical_corners[index],
                dtype=np.float32,
                order="C",
                copy=True,
            ),
            float(canonical_scores[index]),
        )
        for index in range(len(canonical_corners))
    ]]


def save_prediction_create_only(
    corners_world: Any,
    scores: Any,
    filename: str | os.PathLike[str],
) -> Path:
    """Atomically create one canonical prediction without overwriting it."""

    payload = prediction_payload(corners_world, scores)
    target = Path(filename).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=target.parent,
        prefix=f".{target.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, target)
        except FileExistsError as error:
            raise FileExistsError(
                f"refusing to overwrite prediction: {target}"
            ) from error
        target.chmod(0o444)
        directory_descriptor = os.open(target.parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        if temporary.exists():
            temporary.unlink()
    return target


def link_prediction_create_only(
    source: str | os.PathLike[str],
    filename: str | os.PathLike[str],
) -> Path:
    """Create an immutable byte-identical link to one finalized prediction.

    Re-serialization could be semantically equal without proving byte
    identity.  A create-only hard link proves that both protocol paths refer
    to the exact bytes emitted by one finalizer invocation.  Cross-filesystem
    requests and existing targets fail closed.
    """

    raw_source = Path(source)
    if raw_source.is_symlink():
        raise ValueError(
            f"source prediction must not be a symlink: {raw_source}"
        )
    source_path = raw_source.resolve(strict=True)
    if not source_path.is_file():
        raise ValueError(
            f"source prediction must be a regular non-symlink file: {source_path}"
        )
    target = Path(filename).resolve()
    if target == source_path:
        raise ValueError("identity-link target must differ from the source")
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source_path, target, follow_symlinks=False)
    except FileExistsError as error:
        raise FileExistsError(
            f"refusing to overwrite identity prediction: {target}"
        ) from error
    except OSError as error:
        raise OSError(
            "same-run identity predictions must be on one filesystem: "
            f"{source_path} -> {target}"
        ) from error
    target.chmod(0o444)
    directory_descriptor = os.open(target.parent, os.O_RDONLY)
    try:
        os.fsync(directory_descriptor)
    finally:
        os.close(directory_descriptor)
    if not os.path.samefile(source_path, target):
        raise RuntimeError("same-run prediction identity link is not exact")
    return target


_PARENT_FIELDS = frozenset(
    {
        "schema", "scene_id", "sample_idx", "prefix_id",
        "prefix_fraction", "complete", "observer_only",
        "mutation_enabled", "applied_count", "class_agnostic",
        "coordinate_frame", "box_mode", "corner_semantics",
        "boxes_world", "corners_world", "aligned_to_unaligned",
        "axis_alignment_sha256", "scores_3d", "labels_3d",
        "proposal_ids", "point_count", "voxel_size", "runtime_s",
        "num_input_points", "checkpoint_sha256", "config_sha256",
        "source_scene_sha256",
    }
)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_array(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _readonly(value: Any, dtype: np.dtype) -> np.ndarray:
    result = np.ascontiguousarray(value, dtype=dtype)
    result.setflags(write=False)
    return result


def _scalar(values: Mapping[str, np.ndarray], name: str, dtype: np.dtype) -> Any:
    if name not in values:
        raise ValueError(f"TR3D cache is missing {name}")
    value = np.asarray(values[name])
    if value.shape != () or value.dtype != np.dtype(dtype):
        raise ValueError(f"TR3D cache {name} must be a {np.dtype(dtype)} scalar")
    return value.item()


def _text(values: Mapping[str, np.ndarray], name: str) -> str:
    if name not in values:
        raise ValueError(f"TR3D cache is missing {name}")
    value = np.asarray(values[name])
    if value.shape != () or value.dtype.hasobject:
        raise ValueError(f"TR3D cache {name} must be a non-object string scalar")
    result = value.item()
    if isinstance(result, bytes):
        result = result.decode("utf-8")
    if not isinstance(result, str):
        raise ValueError(f"TR3D cache {name} must be text")
    return result


def _homogeneous(value: Any, name: str) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64)
    if (
        matrix.shape != (4, 4)
        or not np.isfinite(matrix).all()
        or not np.allclose(matrix[3], [0.0, 0.0, 0.0, 1.0], atol=1e-8)
    ):
        raise ValueError(f"{name} must be a finite homogeneous [4,4] matrix")
    return np.ascontiguousarray(matrix)


@dataclass(frozen=True)
class PrefixRecord:
    scene_id: str
    prefix_id: str
    prefix_fraction: float
    last_source_timestamp: int
    used_source_timestamps: tuple[int, ...]
    point_count: int
    point_path: Path
    axis_alignment: np.ndarray
    manifest_row_sha256: str


@dataclass(frozen=True)
class ParentCache:
    scene_id: str
    prefix_id: str
    prefix_fraction: float
    proposal_ids: np.ndarray
    corners_world: np.ndarray
    scores_3d: np.ndarray
    point_count: np.ndarray
    aligned_to_unaligned: np.ndarray
    runtime_s: float
    num_input_points: int
    checkpoint_sha256: str
    config_sha256: str
    source_scene_sha256: str
    cache_sha256: str


@dataclass(frozen=True)
class OverlaySelection:
    anchor_index: int
    proposal_row: int
    proposal_id: int
    tr3d_score: float
    anchor_score: float
    anchor_iou: float
    geometry_changed: bool


@dataclass(frozen=True)
class TerminalActiveSummary:
    scene_id: str
    prefix_id: str
    current_source_timestamp: int
    observed_source_timestamps: tuple[int, ...]
    prefix_last_source_timestamp: int
    prediction_count: int
    parent_proposal_count: int
    near_candidate_count: int
    represented_anchor_count: int
    selected_count: int
    changed_count: int
    cache_model_runtime_s: float
    cache_load_s: float
    association_s: float
    geometry_apply_s: float
    replay_total_s: float
    input_geometry_sha256: str
    input_row_geometry_sha256: tuple[str, ...]
    input_scores_sha256: str
    output_geometry_sha256: str
    cache_sha256: str
    source_point_sha256: str
    checkpoint_sha256: str
    config_sha256: str
    manifest_row_sha256: str
    selections: tuple[OverlaySelection, ...]

    def as_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["schema"] = TERMINAL_ACTIVE_SCHEMA
        result["provider_mode"] = "immutable_parent_cache_replay"
        result["ground_truth_access"] = False
        result["clip_access"] = False
        result["labels_scores_order_count_unchanged_by_construction"] = True
        result["terminal_only"] = True
        result["live_tr3d_latency_authoritative"] = False
        return result


def load_prefix_manifest(path: str | os.PathLike[str]) -> dict[str, PrefixRecord]:
    source = Path(path).resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    records: dict[str, PrefixRecord] = {}
    for line_number, raw in enumerate(
        source.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not raw.strip():
            continue
        try:
            row = json.loads(raw)
        except json.JSONDecodeError as error:
            raise ValueError(f"{source}:{line_number}: malformed JSON") from error
        if not isinstance(row, Mapping):
            raise ValueError(f"{source}:{line_number}: row must be an object")
        if row.get("schema") != PREFIX_SCHEMA:
            raise ValueError(f"{source}:{line_number}: unexpected prefix schema")
        if row.get("clock_policy") != FROZEN_CLOCK_POLICY:
            raise ValueError(f"{source}:{line_number}: clock policy mismatch")
        if (
            row.get("status") != "exported"
            or row.get("pose_policy") != FROZEN_POSE_POLICY
            or row.get("source_timestamp_semantics")
            != FROZEN_TIMESTAMP_SEMANTICS
            or row.get("coordinate_frame") != "world_unaligned"
            or int(row.get("frame_stride", -1)) != 25
            or int(row.get("tail_guard_frames", -1)) != 25
        ):
            raise ValueError(f"{source}:{line_number}: frozen prefix contract mismatch")
        scene_id = str(row.get("scene_id", ""))
        prefix_id = str(row.get("prefix_id") or row.get("tag") or "")
        if _SCENE_RE.fullmatch(scene_id) is None:
            raise ValueError(f"{source}:{line_number}: invalid scene_id")
        if _PREFIX_RE.fullmatch(prefix_id) is None:
            raise ValueError(f"{source}:{line_number}: invalid prefix_id")
        if scene_id in records:
            raise ValueError(
                f"{source}: terminal manifest has multiple rows for {scene_id}"
            )
        fraction = float(row.get("prefix_fraction", row.get("fraction", -1)))
        last_timestamp = int(row.get("last_source_timestamp", -1))
        used = tuple(int(value) for value in row.get("used_source_timestamps", ()))
        source_timestamps = tuple(
            int(value) for value in row.get("source_timestamps", ())
        )
        source_frame_count = int(row.get("source_frame_count", -1))
        processed_frame_count = int(row.get("processed_frame_count", -1))
        sampled_frame_count = int(row.get("sampled_frame_count", -1))
        expected_processed = max(1, source_frame_count - 25)
        expected_schedule = tuple(range(0, expected_processed, 25))
        if (
            not math.isclose(fraction, 1.0, rel_tol=0.0, abs_tol=0.0)
            or last_timestamp < 0
            or not used
            or tuple(sorted(set(used))) != used
            or used[-1] != last_timestamp
            or any(value < 0 or value > last_timestamp for value in used)
            or source_timestamps != used
            or sampled_frame_count != len(used)
            or processed_frame_count != expected_processed
            or used != expected_schedule
        ):
            raise ValueError(
                f"{source}:{line_number}: row is not a strict terminal prefix"
            )
        point_count = int(row.get("point_count", -1))
        if point_count < 1:
            raise ValueError(f"{source}:{line_number}: invalid point_count")
        raw_point_path = row.get("point_path")
        if not isinstance(raw_point_path, str) or not raw_point_path:
            raise ValueError(f"{source}:{line_number}: missing point_path")
        point_candidate = Path(raw_point_path)
        if not point_candidate.is_absolute():
            point_candidate = source.parent / point_candidate
        if point_candidate.is_symlink() or not point_candidate.is_file():
            raise FileNotFoundError(point_candidate)
        point_path = point_candidate.resolve()
        expected_bytes = point_count * 6 * np.dtype(np.float32).itemsize
        if point_path.stat().st_size != expected_bytes:
            raise ValueError(
                f"{source}:{line_number}: point file size/count mismatch"
            )
        axis = _homogeneous(row.get("axis_align_matrix"), "axis_align_matrix")
        provenance = row.get("pose_provenance")
        if not isinstance(provenance, list) or len(provenance) != len(used):
            raise ValueError(f"{source}:{line_number}: pose provenance mismatch")
        for expected_timestamp, pose_row in zip(used, provenance):
            if (
                not isinstance(pose_row, Mapping)
                or int(pose_row.get("source_timestamp", -1))
                != expected_timestamp
                or int(pose_row.get("resolved_pose_source_timestamp", -1))
                > expected_timestamp
                or _SHA256_RE.fullmatch(
                    str(pose_row.get("resolved_pose_sha256", ""))
                )
                is None
            ):
                raise ValueError(
                    f"{source}:{line_number}: invalid pose provenance"
                )
        canonical = json.dumps(
            row, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
        records[scene_id] = PrefixRecord(
            scene_id=scene_id,
            prefix_id=prefix_id,
            prefix_fraction=fraction,
            last_source_timestamp=last_timestamp,
            used_source_timestamps=used,
            point_count=point_count,
            point_path=point_path,
            axis_alignment=axis,
            manifest_row_sha256=_sha256_bytes(canonical),
        )
    if not records:
        raise ValueError(f"{source}: prefix manifest is empty")
    return records


def load_parent_cache(
    path: str | os.PathLike[str],
    *,
    prefix: PrefixRecord,
    expected_checkpoint_sha256: str = FROZEN_CHECKPOINT_SHA256,
    expected_config_sha256: str = FROZEN_CONFIG_SHA256,
) -> ParentCache:
    candidate = Path(path)
    if candidate.is_symlink() or not candidate.is_file():
        raise FileNotFoundError(candidate)
    source = candidate.resolve()
    cache_bytes = source.read_bytes()
    if not cache_bytes:
        raise FileNotFoundError(source)
    with np.load(BytesIO(cache_bytes), allow_pickle=False) as archive:
        values = {name: np.array(archive[name], copy=True) for name in archive.files}
    if frozenset(values) != _PARENT_FIELDS:
        raise ValueError(f"{source}: TR3D cache fields disagree")
    if any(np.asarray(value).dtype.hasobject for value in values.values()):
        raise ValueError(f"{source}: TR3D cache contains object arrays")
    if _text(values, "schema") != TR3D_RESIDUAL_CACHE_SCHEMA:
        raise ValueError(f"{source}: unexpected TR3D cache schema")
    scene_id = _text(values, "scene_id")
    prefix_id = _text(values, "prefix_id")
    if _text(values, "sample_idx") != f"{scene_id}:{prefix_id}":
        raise ValueError(f"{source}: sample_idx mismatch")
    fraction = float(_scalar(values, "prefix_fraction", np.float64))
    if (
        scene_id != prefix.scene_id
        or prefix_id != prefix.prefix_id
        or not math.isclose(
            fraction, prefix.prefix_fraction, rel_tol=0.0, abs_tol=0.0
        )
    ):
        raise ValueError(f"{source}: cache/manifest identity mismatch")
    checkpoint_sha = _text(values, "checkpoint_sha256")
    config_sha = _text(values, "config_sha256")
    source_sha = _text(values, "source_scene_sha256")
    for name, value in (
        ("checkpoint_sha256", checkpoint_sha),
        ("config_sha256", config_sha),
        ("source_scene_sha256", source_sha),
    ):
        if _SHA256_RE.fullmatch(value) is None:
            raise ValueError(f"{source}: invalid {name}")
    if checkpoint_sha != expected_checkpoint_sha256:
        raise ValueError(f"{source}: checkpoint SHA256 mismatch")
    if config_sha != expected_config_sha256:
        raise ValueError(f"{source}: config SHA256 mismatch")
    point_sha = _sha256_file(prefix.point_path)
    if source_sha != point_sha:
        raise ValueError(
            f"{source}: source point SHA256 disagrees with terminal prefix"
        )
    for flag_name, expected in (
        ("complete", True),
        ("observer_only", True),
        ("mutation_enabled", False),
        ("class_agnostic", True),
    ):
        raw = np.asarray(values[flag_name])
        if raw.shape != () or raw.dtype != np.dtype(np.bool_) or bool(raw) != expected:
            raise ValueError(f"{source}: invalid {flag_name}")
    if int(_scalar(values, "applied_count", np.int64)) != 0:
        raise ValueError(f"{source}: observer cache applied_count is nonzero")
    if (
        _text(values, "coordinate_frame") != "scannet_unaligned_world"
        or _text(values, "box_mode") != "depth_center_size_yaw_z"
        or _text(values, "corner_semantics")
        != "unordered_8_corners_minmax_only"
    ):
        raise ValueError(f"{source}: canonical geometry contract mismatch")
    proposal_ids = np.asarray(values.get("proposal_ids"))
    boxes = np.asarray(values.get("boxes_world"))
    corners = np.asarray(values.get("corners_world"))
    scores = np.asarray(values.get("scores_3d"))
    labels = np.asarray(values.get("labels_3d"))
    point_count = np.asarray(values.get("point_count"))
    count = len(proposal_ids) if proposal_ids.ndim == 1 else -1
    if (
        count < 0
        or proposal_ids.dtype != np.dtype(np.int64)
        or np.any(proposal_ids < 0)
        or boxes.dtype != np.dtype(np.float32)
        or boxes.shape != (count, 7)
        or corners.dtype != np.dtype(np.float32)
        or corners.shape != (count, 8, 3)
        or scores.dtype != np.dtype(np.float32)
        or scores.shape != (count,)
        or labels.dtype != np.dtype(np.int64)
        or labels.shape != (count,)
        or np.any(labels != 0)
        or point_count.dtype != np.dtype(np.int32)
        or point_count.shape != (count,)
        or len(np.unique(proposal_ids)) != count
        or not np.isfinite(boxes).all()
        or (count and np.any(boxes[:, 3:6] <= 0.0))
        or not np.isfinite(corners).all()
        or not np.isfinite(scores).all()
        or np.any(scores < 0.0)
        or np.any(scores > 1.0)
        or np.any(point_count < 0)
    ):
        raise ValueError(f"{source}: malformed proposal arrays")
    if count and (
        np.any(np.ptp(corners, axis=1) <= 0.0)
        or not np.allclose(
            corners.mean(axis=1), boxes[:, :3], rtol=1e-4, atol=1e-4
        )
    ):
        raise ValueError(f"{source}: boxes/corners disagree")
    transform = _homogeneous(
        np.asarray(values.get("aligned_to_unaligned")),
        "aligned_to_unaligned",
    )
    if not np.allclose(
        transform, np.linalg.inv(prefix.axis_alignment), rtol=0.0, atol=1e-7
    ):
        raise ValueError(f"{source}: axis alignment mismatch")
    axis_hash = _text(values, "axis_alignment_sha256")
    canonical_transform = np.asarray(transform, dtype="<f8")
    if axis_hash != hashlib.sha256(canonical_transform.tobytes()).hexdigest():
        raise ValueError(f"{source}: axis-alignment SHA256 mismatch")
    runtime_s = float(_scalar(values, "runtime_s", np.float64))
    voxel_size = float(_scalar(values, "voxel_size", np.float64))
    num_input_points = int(_scalar(values, "num_input_points", np.int64))
    if (
        not math.isfinite(runtime_s)
        or runtime_s < 0.0
        or not math.isclose(voxel_size, 0.01, rel_tol=0.0, abs_tol=0.0)
        or num_input_points != prefix.point_count
    ):
        raise ValueError(f"{source}: runtime/point-count mismatch")
    return ParentCache(
        scene_id=scene_id,
        prefix_id=prefix_id,
        prefix_fraction=fraction,
        proposal_ids=_readonly(proposal_ids, np.int64),
        corners_world=_readonly(corners, np.float32),
        scores_3d=_readonly(scores, np.float32),
        point_count=_readonly(point_count, np.int32),
        aligned_to_unaligned=_readonly(transform, np.float64),
        runtime_s=runtime_s,
        num_input_points=num_input_points,
        checkpoint_sha256=checkpoint_sha,
        config_sha256=config_sha,
        source_scene_sha256=source_sha,
        cache_sha256=_sha256_bytes(cache_bytes),
    )


def axis_aligned_minmax(corners_world: Any, axis_alignment: Any) -> np.ndarray:
    corners = np.asarray(corners_world, dtype=np.float64)
    matrix = _homogeneous(axis_alignment, "axis_alignment")
    if corners.ndim != 3 or corners.shape[1:] != (8, 3):
        raise ValueError("corners_world must have shape [N,8,3]")
    if not np.isfinite(corners).all():
        raise ValueError("corners_world must be finite")
    aligned = corners @ matrix[:3, :3].T + matrix[None, None, :3, 3]
    if not len(aligned):
        return np.empty((0, 6), dtype=np.float64)
    result = np.concatenate((aligned.min(axis=1), aligned.max(axis=1)), axis=1)
    if np.any(result[:, 3:] <= result[:, :3]):
        raise ValueError("corners must define positive-volume AABBs")
    return result


def pairwise_aabb_iou(left: Any, right: Any) -> np.ndarray:
    lhs = np.asarray(left, dtype=np.float64)
    rhs = np.asarray(right, dtype=np.float64)
    if lhs.ndim != 2 or lhs.shape[1:] != (6,):
        raise ValueError("left boxes must have shape [N,6]")
    if rhs.ndim != 2 or rhs.shape[1:] != (6,):
        raise ValueError("right boxes must have shape [M,6]")
    if not len(lhs) or not len(rhs):
        return np.zeros((len(lhs), len(rhs)), dtype=np.float64)
    overlap = np.maximum(
        np.minimum(lhs[:, None, 3:], rhs[None, :, 3:])
        - np.maximum(lhs[:, None, :3], rhs[None, :, :3]),
        0.0,
    )
    intersection = np.prod(overlap, axis=2)
    left_volume = np.prod(lhs[:, 3:] - lhs[:, :3], axis=1)
    right_volume = np.prod(rhs[:, 3:] - rhs[:, :3], axis=1)
    union = left_volume[:, None] + right_volume[None, :] - intersection
    return np.divide(
        intersection,
        union,
        out=np.zeros_like(intersection),
        where=union > 0.0,
    )


def apply_r3_geometry_overlay(
    *,
    scene_id: str,
    current_source_timestamp: int,
    observed_source_timestamps: Sequence[int],
    anchor_corners_world: Any,
    anchor_scores: Any,
    parent: ParentCache,
    prefix: PrefixRecord,
    cache_load_s: float = 0.0,
) -> tuple[np.ndarray, TerminalActiveSummary]:
    """Apply the frozen R3 rule to one current terminal snapshot."""

    started = time.perf_counter()
    if scene_id != parent.scene_id or scene_id != prefix.scene_id:
        raise ValueError("scene identity mismatch")
    if int(current_source_timestamp) != prefix.last_source_timestamp:
        raise ValueError(
            "terminal R3 requires the exact final observed source timestamp"
        )
    if any(value > int(current_source_timestamp) for value in prefix.used_source_timestamps):
        raise ValueError("prefix contains a future source timestamp")
    observed = tuple(int(value) for value in observed_source_timestamps)
    if observed != prefix.used_source_timestamps:
        raise ValueError("same-run observed keyframes disagree with terminal prefix")
    corners_input = np.asarray(anchor_corners_world)
    scores = np.asarray(anchor_scores, dtype=np.float64)
    if (
        corners_input.dtype != np.dtype(np.float32)
        or corners_input.ndim != 3
        or corners_input.shape[1:] != (8, 3)
        or scores.shape != (len(corners_input),)
        or not np.isfinite(corners_input).all()
        or not np.isfinite(scores).all()
    ):
        raise ValueError("terminal anchors must be float32 [N,8,3] with scores [N]")

    association_started = time.perf_counter()
    candidate_boxes = axis_aligned_minmax(
        parent.corners_world, prefix.axis_alignment
    )
    anchor_boxes = axis_aligned_minmax(corners_input, prefix.axis_alignment)
    if not len(candidate_boxes) or not len(anchor_boxes):
        near_rows = np.empty(0, dtype=np.int64)
        associated = np.empty(0, dtype=np.int64)
        best_iou = np.empty(0, dtype=np.float64)
    else:
        iou = pairwise_aabb_iou(candidate_boxes, anchor_boxes)
        candidate_centres = (candidate_boxes[:, :3] + candidate_boxes[:, 3:]) * 0.5
        anchor_centres = (anchor_boxes[:, :3] + anchor_boxes[:, 3:]) * 0.5
        distances = np.linalg.norm(
            candidate_centres[:, None] - anchor_centres[None, :, :], axis=2
        )
        best_iou = iou.max(axis=1)
        associated = np.empty(len(candidate_boxes), dtype=np.int64)
        for row in range(len(candidate_boxes)):
            tied = np.flatnonzero(iou[row] == best_iou[row])
            local = distances[row, tied]
            associated[row] = int(tied[int(np.argmin(local))])
        near_rows = np.flatnonzero(best_iou > FROZEN_NEAR_IOU)

    selected_parent_rows: list[int] = []
    near_anchor = associated[near_rows]
    for anchor in np.unique(near_anchor):
        local_rows = np.flatnonzero(near_anchor == anchor)
        parent_rows = near_rows[local_rows]
        order = np.lexsort(
            (parent.proposal_ids[parent_rows], -parent.scores_3d[parent_rows])
        )
        parent_row = int(parent_rows[int(order[0])])
        if float(parent.scores_3d[parent_row]) > float(scores[int(anchor)]):
            selected_parent_rows.append(parent_row)
    association_s = time.perf_counter() - association_started

    apply_started = time.perf_counter()
    output = np.array(corners_input, dtype=np.float32, order="C", copy=True)
    selections: list[OverlaySelection] = []
    for parent_row in selected_parent_rows:
        anchor = int(associated[parent_row])
        candidate = np.array(
            parent.corners_world[parent_row],
            dtype=np.float32,
            order="C",
            copy=True,
        )
        changed = output[anchor].tobytes(order="C") != candidate.tobytes(order="C")
        output[anchor] = candidate
        selections.append(
            OverlaySelection(
                anchor_index=anchor,
                proposal_row=parent_row,
                proposal_id=int(parent.proposal_ids[parent_row]),
                tr3d_score=float(parent.scores_3d[parent_row]),
                anchor_score=float(scores[anchor]),
                anchor_iou=float(best_iou[parent_row]),
                geometry_changed=changed,
            )
        )
    geometry_apply_s = time.perf_counter() - apply_started
    replay_total_s = time.perf_counter() - started + float(cache_load_s)
    summary = TerminalActiveSummary(
        scene_id=scene_id,
        prefix_id=prefix.prefix_id,
        current_source_timestamp=int(current_source_timestamp),
        observed_source_timestamps=observed,
        prefix_last_source_timestamp=prefix.last_source_timestamp,
        prediction_count=len(corners_input),
        parent_proposal_count=len(parent.proposal_ids),
        near_candidate_count=len(near_rows),
        represented_anchor_count=len(np.unique(near_anchor)),
        selected_count=len(selections),
        changed_count=sum(item.geometry_changed for item in selections),
        cache_model_runtime_s=parent.runtime_s,
        cache_load_s=float(cache_load_s),
        association_s=association_s,
        geometry_apply_s=geometry_apply_s,
        replay_total_s=replay_total_s,
        input_geometry_sha256=_sha256_array(corners_input),
        input_row_geometry_sha256=tuple(
            _sha256_array(corners_input[index])
            for index in range(len(corners_input))
        ),
        input_scores_sha256=_sha256_array(scores),
        output_geometry_sha256=_sha256_array(output),
        cache_sha256=parent.cache_sha256,
        source_point_sha256=parent.source_scene_sha256,
        checkpoint_sha256=parent.checkpoint_sha256,
        config_sha256=parent.config_sha256,
        manifest_row_sha256=prefix.manifest_row_sha256,
        selections=tuple(selections),
    )
    return output, summary


class TerminalR3CacheReplay:
    """Same-run terminal hook backed by immutable, causal p100 caches."""

    def __init__(
        self,
        *,
        manifest_path: str | os.PathLike[str],
        parent_cache_root: str | os.PathLike[str],
        diagnostics_root: str | os.PathLike[str] | None = None,
        expected_checkpoint_sha256: str = FROZEN_CHECKPOINT_SHA256,
        expected_config_sha256: str = FROZEN_CONFIG_SHA256,
    ) -> None:
        self.manifest_path = Path(manifest_path).resolve()
        self.parent_cache_root = Path(parent_cache_root).resolve()
        self.diagnostics_root = (
            None if diagnostics_root is None else Path(diagnostics_root).resolve()
        )
        if not self.parent_cache_root.is_dir():
            raise FileNotFoundError(self.parent_cache_root)
        if _SHA256_RE.fullmatch(expected_checkpoint_sha256) is None:
            raise ValueError("invalid expected checkpoint SHA256")
        if _SHA256_RE.fullmatch(expected_config_sha256) is None:
            raise ValueError("invalid expected config SHA256")
        self.expected_checkpoint_sha256 = expected_checkpoint_sha256
        self.expected_config_sha256 = expected_config_sha256
        self.prefixes = load_prefix_manifest(self.manifest_path)

    def apply(
        self,
        *,
        scene_id: str,
        current_source_timestamp: int,
        observed_source_timestamps: Sequence[int],
        anchor_corners_world: Any,
        anchor_scores: Any,
    ) -> tuple[np.ndarray, TerminalActiveSummary]:
        if scene_id not in self.prefixes:
            raise ValueError(f"terminal prefix manifest has no row for {scene_id}")
        prefix = self.prefixes[scene_id]
        cache_path = (
            self.parent_cache_root / scene_id / f"{prefix.prefix_id}.npz"
        )
        load_started = time.perf_counter()
        parent = load_parent_cache(
            cache_path,
            prefix=prefix,
            expected_checkpoint_sha256=self.expected_checkpoint_sha256,
            expected_config_sha256=self.expected_config_sha256,
        )
        cache_load_s = time.perf_counter() - load_started
        output, summary = apply_r3_geometry_overlay(
            scene_id=scene_id,
            current_source_timestamp=current_source_timestamp,
            observed_source_timestamps=observed_source_timestamps,
            anchor_corners_world=anchor_corners_world,
            anchor_scores=anchor_scores,
            parent=parent,
            prefix=prefix,
            cache_load_s=cache_load_s,
        )
        if self.diagnostics_root is not None:
            self._write_diagnostic(summary)
        return output, summary

    def _write_diagnostic(self, summary: TerminalActiveSummary) -> None:
        assert self.diagnostics_root is not None
        self.diagnostics_root.mkdir(parents=True, exist_ok=True)
        target = self.diagnostics_root / f"{summary.scene_id}_tr3d_terminal.json"
        payload = summary.as_dict()
        payload["schema"] = TERMINAL_DIAGNOSTIC_SCHEMA
        payload["manifest_path"] = str(self.manifest_path)
        payload["parent_cache_root"] = str(self.parent_cache_root)
        encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode(
            "utf-8"
        )
        descriptor, temporary = tempfile.mkstemp(
            dir=self.diagnostics_root,
            prefix=f".{summary.scene_id}.",
            suffix=".json",
        )
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(encoded)
                handle.flush()
                os.fsync(handle.fileno())
            try:
                os.link(temporary, target)
            except FileExistsError as error:
                raise FileExistsError(
                    f"terminal R3 diagnostic already exists: {target}"
                ) from error
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

    @staticmethod
    def summary_text(summary: TerminalActiveSummary) -> str:
        return (
            "TR3D terminal active summary | "
            f"prefix={summary.prefix_id}, "
            f"timestamp={summary.current_source_timestamp}, "
            f"proposals/near/selected/changed="
            f"{summary.parent_proposal_count}/{summary.near_candidate_count}/"
            f"{summary.selected_count}/{summary.changed_count}, "
            f"cached_model_ms={summary.cache_model_runtime_s * 1000.0:.3f}, "
            f"replay_ms={summary.replay_total_s * 1000.0:.3f}"
        )


__all__ = [
    "FROZEN_CHECKPOINT_SHA256",
    "FROZEN_CONFIG_SHA256",
    "FROZEN_NEAR_IOU",
    "OverlaySelection",
    "ParentCache",
    "PrefixRecord",
    "TerminalActiveSummary",
    "TerminalR3CacheReplay",
    "apply_r3_geometry_overlay",
    "axis_aligned_minmax",
    "load_parent_cache",
    "load_prefix_manifest",
    "link_prediction_create_only",
    "pairwise_aabb_iou",
    "prediction_payload",
    "save_prediction_create_only",
]
