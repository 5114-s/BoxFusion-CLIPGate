"""Immutable, class-agnostic TR3D residual-proposal cache.

The cache is deliberately independent from the historical P1/P2 diagnostic
schemas.  One file represents one ScanNet trajectory prefix and contains only
the official TR3D inference outputs plus the provenance required to replay
them.  Loading is fail-closed: object arrays, unknown fields, malformed boxes,
non-zero labels, and provenance mismatches are rejected.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from io import BytesIO
import os
from pathlib import Path
import re
import tempfile
from typing import Mapping, Sequence

import numpy as np


TR3D_RESIDUAL_CACHE_SCHEMA = "boxfusion.tr3d_residual_cache.v1"
TR3D_COORDINATE_FRAME = "scannet_unaligned_world"
TR3D_BOX_MODE = "depth_center_size_yaw_z"
_SCENE_RE = re.compile(r"^scene[0-9]{4}_[0-9]{2}$")
_PREFIX_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

_FIELDS = frozenset(
    {
        "schema",
        "scene_id",
        "sample_idx",
        "prefix_id",
        "prefix_fraction",
        "complete",
        "observer_only",
        "mutation_enabled",
        "applied_count",
        "class_agnostic",
        "coordinate_frame",
        "box_mode",
        "corner_semantics",
        "boxes_world",
        "corners_world",
        "aligned_to_unaligned",
        "axis_alignment_sha256",
        "scores_3d",
        "labels_3d",
        "proposal_ids",
        "point_count",
        "voxel_size",
        "runtime_s",
        "num_input_points",
        "checkpoint_sha256",
        "config_sha256",
        "source_scene_sha256",
    }
)


@dataclass(frozen=True)
class TR3DResidualCache:
    scene_id: str
    sample_idx: str
    prefix_id: str
    prefix_fraction: float
    boxes_world: np.ndarray
    corners_world: np.ndarray
    aligned_to_unaligned: np.ndarray
    axis_alignment_sha256: str
    scores_3d: np.ndarray
    labels_3d: np.ndarray
    proposal_ids: np.ndarray
    point_count: np.ndarray
    voxel_size: float
    runtime_s: float
    num_input_points: int
    checkpoint_sha256: str
    config_sha256: str
    source_scene_sha256: str

    @property
    def proposal_count(self) -> int:
        return int(self.boxes_world.shape[0])

    def as_npz_payload(self) -> dict[str, np.ndarray]:
        """Return the canonical serialization payload."""

        return {
            "schema": np.asarray(TR3D_RESIDUAL_CACHE_SCHEMA),
            "scene_id": np.asarray(self.scene_id),
            "sample_idx": np.asarray(self.sample_idx),
            "prefix_id": np.asarray(self.prefix_id),
            "prefix_fraction": np.asarray(
                self.prefix_fraction, dtype=np.float64
            ),
            "complete": np.asarray(True, dtype=np.bool_),
            "observer_only": np.asarray(True, dtype=np.bool_),
            "mutation_enabled": np.asarray(False, dtype=np.bool_),
            "applied_count": np.asarray(0, dtype=np.int64),
            "class_agnostic": np.asarray(True, dtype=np.bool_),
            "coordinate_frame": np.asarray(TR3D_COORDINATE_FRAME),
            "box_mode": np.asarray(TR3D_BOX_MODE),
            "corner_semantics": np.asarray(
                "unordered_8_corners_minmax_only"
            ),
            "boxes_world": np.asarray(self.boxes_world, dtype=np.float32),
            "corners_world": np.asarray(
                self.corners_world, dtype=np.float32
            ),
            "aligned_to_unaligned": np.asarray(
                self.aligned_to_unaligned, dtype=np.float64
            ),
            "axis_alignment_sha256": np.asarray(
                self.axis_alignment_sha256
            ),
            "scores_3d": np.asarray(self.scores_3d, dtype=np.float32),
            "labels_3d": np.asarray(self.labels_3d, dtype=np.int64),
            "proposal_ids": np.asarray(self.proposal_ids, dtype=np.int64),
            "point_count": np.asarray(self.point_count, dtype=np.int32),
            "voxel_size": np.asarray(self.voxel_size, dtype=np.float64),
            "runtime_s": np.asarray(self.runtime_s, dtype=np.float64),
            "num_input_points": np.asarray(
                self.num_input_points, dtype=np.int64
            ),
            "checkpoint_sha256": np.asarray(self.checkpoint_sha256),
            "config_sha256": np.asarray(self.config_sha256),
            "source_scene_sha256": np.asarray(self.source_scene_sha256),
        }


def _depth_box_corners(boxes: np.ndarray) -> np.ndarray:
    values = np.asarray(boxes, dtype=np.float64)
    local = _CORNER_SIGNS[None] * (0.5 * values[:, None, 3:6])
    cosine = np.cos(values[:, 6])
    sine = np.sin(values[:, 6])
    result = np.empty_like(local)
    result[:, :, 0] = (
        local[:, :, 0] * cosine[:, None]
        - local[:, :, 1] * sine[:, None]
    )
    result[:, :, 1] = (
        local[:, :, 0] * sine[:, None]
        + local[:, :, 1] * cosine[:, None]
    )
    result[:, :, 2] = local[:, :, 2]
    return result + values[:, None, :3]


_CORNER_SIGNS = np.asarray(
    [
        [-1, -1, -1],
        [-1, -1, 1],
        [-1, 1, -1],
        [-1, 1, 1],
        [1, -1, -1],
        [1, -1, 1],
        [1, 1, -1],
        [1, 1, 1],
    ],
    dtype=np.float64,
)


def make_tr3d_residual_cache_from_aligned(
    *,
    scene_id: str,
    boxes_aligned: np.ndarray,
    scores_3d: np.ndarray,
    unaligned_to_aligned: np.ndarray,
    checkpoint_sha256: str,
    config_sha256: str,
    source_scene_sha256: str,
    prefix_id: str = "full",
    prefix_fraction: float = 1.0,
    labels_3d: np.ndarray | None = None,
    proposal_ids: np.ndarray | None = None,
    point_count: np.ndarray | None = None,
    voxel_size: float = 0.02,
    runtime_s: float = 0.0,
    num_input_points: int = 0,
) -> TR3DResidualCache:
    """Convert official aligned TR3D outputs into the immutable cache.

    Official ScanNet TR3D predictions are emitted after ``GlobalAlignment``.
    This factory applies the exact inverse matrix to the eight corners and
    makes those corners authoritative.  Six-dimensional boxes are accepted
    and receive yaw zero.
    """

    aligned = np.asarray(boxes_aligned, dtype=np.float64)
    if aligned.ndim != 2 or aligned.shape[1] not in {6, 7}:
        raise ValueError("boxes_aligned must be [N,6] or [N,7]")
    if aligned.shape[1] == 6:
        aligned = np.concatenate(
            (aligned, np.zeros((len(aligned), 1), dtype=np.float64)),
            axis=1,
        )
    transform = np.asarray(unaligned_to_aligned, dtype=np.float64)
    if (
        transform.shape != (4, 4)
        or not np.isfinite(transform).all()
        or not np.allclose(transform[3], [0, 0, 0, 1], atol=1e-8)
    ):
        raise ValueError("unaligned_to_aligned must be homogeneous [4,4]")
    inverse = np.linalg.inv(transform)
    corners_aligned = _depth_box_corners(aligned)
    corners_world = (
        corners_aligned @ inverse[:3, :3].T
        + inverse[None, None, :3, 3]
    )
    centers_world = corners_world.mean(axis=1)
    heading_aligned = np.stack(
        (
            np.cos(aligned[:, 6]),
            np.sin(aligned[:, 6]),
            np.zeros(len(aligned)),
        ),
        axis=1,
    )
    heading_world = heading_aligned @ inverse[:3, :3].T
    yaw_world = np.arctan2(heading_world[:, 1], heading_world[:, 0])
    boxes_world = np.concatenate(
        (centers_world, aligned[:, 3:6], yaw_world[:, None]), axis=1
    )
    count = len(aligned)
    cache = TR3DResidualCache(
        scene_id=scene_id,
        sample_idx=f"{scene_id}:{prefix_id}",
        prefix_id=prefix_id,
        prefix_fraction=float(prefix_fraction),
        boxes_world=np.asarray(boxes_world, dtype=np.float32),
        corners_world=np.asarray(corners_world, dtype=np.float32),
        aligned_to_unaligned=np.asarray(inverse, dtype=np.float64),
        axis_alignment_sha256=transform_sha256(inverse),
        scores_3d=np.asarray(scores_3d, dtype=np.float32),
        labels_3d=np.asarray(
            (
                np.zeros(count, dtype=np.int64)
                if labels_3d is None
                else labels_3d
            ),
            dtype=np.int64,
        ),
        proposal_ids=np.asarray(
            (
                np.arange(count, dtype=np.int64)
                if proposal_ids is None
                else proposal_ids
            ),
            dtype=np.int64,
        ),
        point_count=np.asarray(
            (
                np.zeros(count, dtype=np.int32)
                if point_count is None
                else point_count
            ),
            dtype=np.int32,
        ),
        voxel_size=float(voxel_size),
        runtime_s=float(runtime_s),
        num_input_points=int(num_input_points),
        checkpoint_sha256=checkpoint_sha256,
        config_sha256=config_sha256,
        source_scene_sha256=source_scene_sha256,
    )
    return validate_tr3d_residual_payload(cache.as_npz_payload())


def _readonly(value: np.ndarray) -> np.ndarray:
    result = np.ascontiguousarray(value)
    result.setflags(write=False)
    return result


def _scalar(
    values: Mapping[str, np.ndarray],
    name: str,
    *,
    kind: str | None = None,
):
    value = np.asarray(values[name])
    if value.shape != () or value.dtype.hasobject:
        raise ValueError(f"{name} must be a non-object scalar")
    if kind is not None and value.dtype.kind != kind:
        raise ValueError(f"{name} must have dtype kind {kind!r}")
    return value.item()


def _text(values: Mapping[str, np.ndarray], name: str) -> str:
    raw = _scalar(values, name)
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    if not isinstance(raw, str):
        raise ValueError(f"{name} must be a string scalar")
    return raw


def _boolean(values: Mapping[str, np.ndarray], name: str) -> bool:
    raw = np.asarray(values[name])
    if raw.shape != () or raw.dtype != np.dtype(np.bool_):
        raise ValueError(f"{name} must be a bool scalar")
    return bool(raw.item())


def _typed_scalar(
    values: Mapping[str, np.ndarray], name: str, dtype: np.dtype
):
    raw = np.asarray(values[name])
    if raw.shape != () or raw.dtype != np.dtype(dtype):
        raise ValueError(f"{name} must be a {np.dtype(dtype)} scalar")
    return raw.item()


def _hash(value: str, name: str) -> str:
    normalized = value.lower()
    if _SHA256_RE.fullmatch(normalized) is None:
        raise ValueError(f"{name} must be a lowercase SHA256")
    return normalized


def transform_sha256(transform: np.ndarray) -> str:
    """Hash a transform in canonical little-endian float64 C order."""

    value = np.asarray(transform, dtype="<f8")
    if value.shape != (4, 4) or not np.isfinite(value).all():
        raise ValueError("transform must be finite [4,4]")
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def _exact_array(
    values: Mapping[str, np.ndarray],
    name: str,
    *,
    dtype: np.dtype,
    shape: tuple[int | None, ...],
) -> np.ndarray:
    value = np.asarray(values[name])
    if value.dtype != np.dtype(dtype) or value.ndim != len(shape):
        raise ValueError(
            f"{name} must have dtype {np.dtype(dtype)} and shape {shape}"
        )
    for actual, expected in zip(value.shape, shape):
        if expected is not None and actual != expected:
            raise ValueError(f"{name} must have shape {shape}")
    return value


def validate_tr3d_residual_payload(
    values: Mapping[str, np.ndarray],
    *,
    expected_scene_id: str | None = None,
    expected_prefix_id: str | None = None,
    expected_checkpoint_sha256: str | None = None,
    expected_config_sha256: str | None = None,
    expected_source_scene_sha256: str | None = None,
) -> TR3DResidualCache:
    """Validate a decoded NPZ mapping and return immutable arrays."""

    fields = frozenset(values)
    if fields != _FIELDS:
        missing = sorted(_FIELDS - fields)
        unknown = sorted(fields - _FIELDS)
        raise ValueError(
            f"TR3D cache fields disagree; missing={missing}, unknown={unknown}"
        )
    if any(np.asarray(value).dtype.hasobject for value in values.values()):
        raise ValueError("TR3D cache must not contain object arrays")
    if _text(values, "schema") != TR3D_RESIDUAL_CACHE_SCHEMA:
        raise ValueError("unsupported TR3D residual cache schema")
    scene_id = _text(values, "scene_id")
    if _SCENE_RE.fullmatch(scene_id) is None:
        raise ValueError(f"invalid ScanNet scene id: {scene_id!r}")
    if expected_scene_id is not None and scene_id != expected_scene_id:
        raise ValueError(
            f"TR3D cache scene {scene_id!r} != {expected_scene_id!r}"
        )
    sample_idx = _text(values, "sample_idx")
    prefix_id = _text(values, "prefix_id")
    if _PREFIX_RE.fullmatch(prefix_id) is None:
        raise ValueError(f"invalid prefix_id: {prefix_id!r}")
    if expected_prefix_id is not None and prefix_id != expected_prefix_id:
        raise ValueError(
            f"TR3D cache prefix {prefix_id!r} != {expected_prefix_id!r}"
        )
    if not sample_idx or sample_idx != f"{scene_id}:{prefix_id}":
        raise ValueError("sample_idx must equal '<scene_id>:<prefix_id>'")

    prefix_fraction = float(
        _typed_scalar(values, "prefix_fraction", np.float64)
    )
    voxel_size = float(_typed_scalar(values, "voxel_size", np.float64))
    runtime_s = float(_typed_scalar(values, "runtime_s", np.float64))
    num_input_points = int(
        _typed_scalar(values, "num_input_points", np.int64)
    )
    applied_count = int(
        _typed_scalar(values, "applied_count", np.int64)
    )
    if (
        not np.isfinite(prefix_fraction)
        or prefix_fraction <= 0.0
        or prefix_fraction > 1.0
    ):
        raise ValueError("prefix_fraction must be finite and in (0,1]")
    if not np.isfinite(voxel_size) or voxel_size <= 0.0:
        raise ValueError("voxel_size must be finite and positive")
    if not np.isfinite(runtime_s) or runtime_s < 0.0:
        raise ValueError("runtime_s must be finite and nonnegative")
    if num_input_points < 0:
        raise ValueError("num_input_points must be nonnegative")
    if applied_count != 0:
        raise ValueError("observer cache applied_count must be zero")
    if not _boolean(values, "complete"):
        raise ValueError("TR3D cache is incomplete")
    if not _boolean(values, "observer_only"):
        raise ValueError("TR3D cache is not observer-only")
    if _boolean(values, "mutation_enabled"):
        raise ValueError("TR3D cache enables mutation")
    if not _boolean(values, "class_agnostic"):
        raise ValueError("TR3D residual cache must be class-agnostic")
    if _text(values, "coordinate_frame") != TR3D_COORDINATE_FRAME:
        raise ValueError("TR3D cache coordinate frame is not canonical")
    if _text(values, "box_mode") != TR3D_BOX_MODE:
        raise ValueError("TR3D cache box mode is not canonical")
    if (
        _text(values, "corner_semantics")
        != "unordered_8_corners_minmax_only"
    ):
        raise ValueError("TR3D cache corner semantics are not canonical")

    boxes = _exact_array(
        values, "boxes_world", dtype=np.float32, shape=(None, 7)
    )
    count = int(boxes.shape[0])
    corners = _exact_array(
        values, "corners_world", dtype=np.float32, shape=(count, 8, 3)
    )
    aligned_to_unaligned = _exact_array(
        values,
        "aligned_to_unaligned",
        dtype=np.float64,
        shape=(4, 4),
    )
    if (
        not np.isfinite(aligned_to_unaligned).all()
        or not np.allclose(
            aligned_to_unaligned[3], [0.0, 0.0, 0.0, 1.0], atol=1e-8
        )
        or abs(float(np.linalg.det(aligned_to_unaligned[:3, :3]))) < 1e-8
    ):
        raise ValueError("aligned_to_unaligned must be invertible homogeneous")
    scores = _exact_array(
        values, "scores_3d", dtype=np.float32, shape=(count,)
    )
    labels = _exact_array(
        values, "labels_3d", dtype=np.int64, shape=(count,)
    )
    proposal_ids = _exact_array(
        values, "proposal_ids", dtype=np.int64, shape=(count,)
    )
    point_count = _exact_array(
        values, "point_count", dtype=np.int32, shape=(count,)
    )
    if not np.isfinite(boxes).all() or (
        count and np.any(boxes[:, 3:6] <= 0.0)
    ):
        raise ValueError(
            "boxes_world contains non-finite or non-positive boxes"
        )
    corner_extent = np.ptp(corners, axis=1) if count else np.empty((0, 3))
    if (
        not np.isfinite(corners).all()
        or (count and np.any(corner_extent <= 0.0))
        or (
            count
            and not np.allclose(
                corners.mean(axis=1),
                boxes[:, :3],
                rtol=1e-4,
                atol=1e-4,
            )
        )
    ):
        raise ValueError(
            "corners_world must be finite, positive, and centered on "
            "boxes_world"
        )
    if (
        not np.isfinite(scores).all()
        or np.any(scores < 0.0)
        or np.any(scores > 1.0)
    ):
        raise ValueError("scores_3d must be finite and in [0,1]")
    if np.any(labels != 0):
        raise ValueError("class-agnostic labels_3d must all be zero")
    if (
        np.any(proposal_ids < 0)
        or len(np.unique(proposal_ids)) != count
    ):
        raise ValueError("proposal_ids must be unique and nonnegative")
    if np.any(point_count < 0):
        raise ValueError("point_count must be nonnegative")

    checkpoint_sha256 = _hash(
        _text(values, "checkpoint_sha256"), "checkpoint_sha256"
    )
    config_sha256 = _hash(
        _text(values, "config_sha256"), "config_sha256"
    )
    source_scene_sha256 = _hash(
        _text(values, "source_scene_sha256"), "source_scene_sha256"
    )
    axis_alignment_sha256 = _hash(
        _text(values, "axis_alignment_sha256"), "axis_alignment_sha256"
    )
    if axis_alignment_sha256 != transform_sha256(aligned_to_unaligned):
        raise ValueError(
            "axis_alignment_sha256 must hash aligned_to_unaligned"
        )
    if (
        expected_checkpoint_sha256 is not None
        and checkpoint_sha256 != _hash(
            expected_checkpoint_sha256, "expected_checkpoint_sha256"
        )
    ):
        raise ValueError("TR3D cache checkpoint SHA256 mismatch")
    if (
        expected_config_sha256 is not None
        and config_sha256
        != _hash(expected_config_sha256, "expected_config_sha256")
    ):
        raise ValueError("TR3D cache config SHA256 mismatch")
    if (
        expected_source_scene_sha256 is not None
        and source_scene_sha256
        != _hash(
            expected_source_scene_sha256,
            "expected_source_scene_sha256",
        )
    ):
        raise ValueError("TR3D cache source-scene SHA256 mismatch")

    return TR3DResidualCache(
        scene_id=scene_id,
        sample_idx=sample_idx,
        prefix_id=prefix_id,
        prefix_fraction=prefix_fraction,
        boxes_world=_readonly(boxes),
        corners_world=_readonly(corners),
        aligned_to_unaligned=_readonly(aligned_to_unaligned),
        axis_alignment_sha256=axis_alignment_sha256,
        scores_3d=_readonly(scores),
        labels_3d=_readonly(labels),
        proposal_ids=_readonly(proposal_ids),
        point_count=_readonly(point_count),
        voxel_size=voxel_size,
        runtime_s=runtime_s,
        num_input_points=num_input_points,
        checkpoint_sha256=checkpoint_sha256,
        config_sha256=config_sha256,
        source_scene_sha256=source_scene_sha256,
    )


def tr3d_residual_cache_path(
    root: str | os.PathLike[str], scene_id: str, prefix_id: str = "full"
) -> Path:
    if _SCENE_RE.fullmatch(scene_id) is None:
        raise ValueError(f"invalid ScanNet scene id: {scene_id!r}")
    if _PREFIX_RE.fullmatch(prefix_id) is None:
        raise ValueError(f"invalid prefix_id: {prefix_id!r}")
    return Path(root) / scene_id / f"{prefix_id}.npz"


def load_tr3d_residual_cache(
    path: str | os.PathLike[str],
    **expected,
) -> TR3DResidualCache:
    cache_path = Path(path)
    if not cache_path.is_file():
        raise FileNotFoundError(cache_path)
    with np.load(cache_path, allow_pickle=False) as archive:
        values = {
            name: np.array(archive[name], copy=True)
            for name in archive.files
        }
    return validate_tr3d_residual_payload(values, **expected)


def write_tr3d_residual_cache(
    path: str | os.PathLike[str], cache: TR3DResidualCache
) -> Path:
    """Create a cache exactly once; existing paths are never overwritten."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    canonical = validate_tr3d_residual_payload(cache.as_npz_payload())
    buffer = BytesIO()
    np.savez_compressed(buffer, **canonical.as_npz_payload())
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{target.name}.",
            suffix=".tmp",
            dir=target.parent,
            delete=False,
        ) as handle:
            temporary_name = handle.name
            handle.write(buffer.getvalue())
            handle.flush()
            os.fsync(handle.fileno())
        # A hard link gives us atomic create-if-absent semantics: unlike
        # os.replace it can never overwrite an immutable cache.
        os.link(temporary_name, target)
        target.chmod(0o444)
    except FileExistsError as error:
        raise FileExistsError(f"immutable TR3D cache exists: {target}") from error
    finally:
        if temporary_name is not None:
            try:
                Path(temporary_name).unlink()
            except FileNotFoundError:
                pass
    return target


def validate_tr3d_residual_cache_set(
    root: str | os.PathLike[str],
    scene_ids: Sequence[str],
    *,
    prefix_id: str = "full",
    expected_checkpoint_sha256: str | None = None,
    expected_config_sha256: str | None = None,
) -> dict[str, TR3DResidualCache]:
    if not scene_ids or len(scene_ids) != len(set(scene_ids)):
        raise ValueError("scene_ids must be non-empty and unique")
    result: dict[str, TR3DResidualCache] = {}
    for scene_id in scene_ids:
        result[scene_id] = load_tr3d_residual_cache(
            tr3d_residual_cache_path(root, scene_id, prefix_id),
            expected_scene_id=scene_id,
            expected_prefix_id=prefix_id,
            expected_checkpoint_sha256=expected_checkpoint_sha256,
            expected_config_sha256=expected_config_sha256,
        )
    return result
