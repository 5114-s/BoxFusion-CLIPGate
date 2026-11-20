"""Raw, bounded 5 cm depth fragments for the Graw ablation.

This module is the deliberately *uncleaned* arm of the Group3D-lite
experiment.  A selected proposal contributes every sampled depth pixel that
is finite and lies in the frozen metric depth interval.  It does not perform
depth-edge suppression, jump tests, seed selection, or connected-component
filtering.

Coordinate and quantization contract
------------------------------------
``box_xyxy`` uses continuous, zero-based ``x=column, y=row`` coordinates in a
proposal image already registered to the depth image.  The caller supplies a
positive-scale, axis-aligned homogeneous 3x3
``proposal_to_depth_affine``.  Pixel centers from ``ceil(min)`` through
``floor(max)`` are sampled inclusively with a frozen initial stride of four;
the stride is increased deterministically until the ray cap is met.

World points are converted directly to signed ``int64`` voxel keys with
``floor(world_coordinate / 0.05)``.  Thus voxel ``k`` denotes the half-open
interval ``[0.05*k, 0.05*(k+1))`` on every axis, including negative axes:
an infinitesimal negative coordinate belongs to voxel ``-1``, not voxel 0.
Only these integer keys are retained.  No floating-point voxel centroid is
stored or re-quantized later.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral, Real
import time
from types import MappingProxyType
from typing import Mapping, Optional, Sequence

import numpy as np


SCHEMA = "boxfusion.graw_raw_fragments.v1"
VOXEL_SIZE_METERS = 0.05

MAX_INPUT_DEPTH_PIXELS = 4_194_304
MAX_INPUT_PROPOSALS = 4_096

# Private backstops keep executable policy independent of rebinding the public
# audit/compatibility aliases above.
_F_VOXEL_SIZE_METERS = 0.05
_F_MAX_INPUT_DEPTH_PIXELS = 4_194_304
_F_MAX_INPUT_PROPOSALS = 4_096

# Names and values intentionally match the extraction-relevant subset of
# smov_fragments.DEFAULT_CONFIG so raw and clean arms can use the same proposal
# selection and resource envelope.
_DEFAULT_CONFIG: Mapping[str, object] = MappingProxyType(
    {
        "pixel_stride": 4,
        "max_rays_per_proposal": 1024,
        "min_depth_m": 0.10,
        "max_depth_m": 8.0,
        "min_fragment_points": 16,
        "voxel_size_m": _F_VOXEL_SIZE_METERS,
        "max_points_per_view": 512,
        "max_proposals_per_keyframe": 64,
    }
)
DEFAULT_CONFIG = _DEFAULT_CONFIG

_FROZEN_GEOMETRY = (
    "pixel_stride",
    "min_depth_m",
    "max_depth_m",
    "min_fragment_points",
    "voxel_size_m",
)

_HARD_CAPS = (
    "max_rays_per_proposal",
    "max_points_per_view",
    "max_proposals_per_keyframe",
)


def _strict_int(name: str, value: object, minimum: int = 1) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer")
    result = int(value)
    if result < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return result


def _strict_real(name: str, value: object) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real number")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be a finite real number")
    return result


def resolve_config(value: Optional[Mapping[str, object]] = None) -> Mapping[str, object]:
    """Return a validated immutable config for the frozen raw arm.

    Geometry fields cannot be changed.  Resource caps can only be reduced,
    which keeps local smoke tests bounded without permitting a more permissive
    experiment than the preregistered arm.
    """

    if value is None:
        value = {}
    if not isinstance(value, Mapping):
        raise ValueError("Graw raw-fragment config must be a mapping")
    unknown = sorted(set(value) - set(_DEFAULT_CONFIG))
    if unknown:
        raise ValueError("unknown Graw config key(s): " + ", ".join(unknown))
    cfg = dict(_DEFAULT_CONFIG)
    cfg.update(value)

    for name in (
        "pixel_stride",
        "max_rays_per_proposal",
        "min_fragment_points",
        "max_points_per_view",
        "max_proposals_per_keyframe",
    ):
        cfg[name] = _strict_int(name, cfg[name])
    for name in ("min_depth_m", "max_depth_m", "voxel_size_m"):
        cfg[name] = _strict_real(name, cfg[name])

    changed = [name for name in _FROZEN_GEOMETRY if cfg[name] != _DEFAULT_CONFIG[name]]
    if changed:
        raise ValueError("Graw geometry fields are frozen; changed: " + ", ".join(changed))
    exceeded = [name for name in _HARD_CAPS if cfg[name] > _DEFAULT_CONFIG[name]]
    if exceeded:
        raise ValueError("Graw resource caps exceed hard limits: " + ", ".join(exceeded))
    minimum = int(cfg["min_fragment_points"])
    if int(cfg["max_rays_per_proposal"]) < minimum:
        raise ValueError("max_rays_per_proposal cannot be below min_fragment_points")
    if int(cfg["max_points_per_view"]) < minimum:
        raise ValueError("max_points_per_view cannot be below min_fragment_points")
    return MappingProxyType(cfg)


def _readonly(value: object, dtype: np.dtype) -> np.ndarray:
    result = np.array(value, dtype=dtype, order="C", copy=True)
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class RawFragmentCoverage:
    effective_stride: int = 0
    sampled_rays: int = 0
    usable_rays: int = 0
    unique_voxels: int = 0
    output_voxels: int = 0
    valid_depth_ratio: float = 0.0


@dataclass(frozen=True)
class RawViewFragment:
    proposal_id: int
    frame_id: int
    score: float
    crop_xyxy_depth: np.ndarray
    depth_shape: tuple[int, int]
    proposal_to_depth_affine: np.ndarray
    intrinsics: np.ndarray
    camera_to_world: np.ndarray
    voxel_keys: np.ndarray
    coverage: RawFragmentCoverage

    def __post_init__(self) -> None:
        object.__setattr__(self, "crop_xyxy_depth", _readonly(self.crop_xyxy_depth, np.float32))
        object.__setattr__(
            self,
            "proposal_to_depth_affine",
            _readonly(self.proposal_to_depth_affine, np.float64),
        )
        object.__setattr__(self, "intrinsics", _readonly(self.intrinsics, np.float64))
        object.__setattr__(self, "camera_to_world", _readonly(self.camera_to_world, np.float64))
        object.__setattr__(self, "voxel_keys", _readonly(self.voxel_keys, np.int64))

    @property
    def voxels(self) -> np.ndarray:
        """Read-only matcher-facing alias for ``voxel_keys``."""

        return self.voxel_keys


@dataclass(frozen=True)
class RawProposalDiagnostic:
    proposal_id: int
    selected: bool
    reason: Optional[str]
    coverage: RawFragmentCoverage
    elapsed_ms: float
    fragment: Optional[RawViewFragment]

    @property
    def accepted(self) -> bool:
        return self.fragment is not None


@dataclass(frozen=True)
class PreparedRawKeyframe:
    scene_id: str
    frame_id: int
    proposal_ids: tuple[int, ...]
    selected_proposal_ids: tuple[int, ...]
    diagnostics: tuple[RawProposalDiagnostic, ...]
    elapsed_ms: float


class _Abstain(ValueError):
    def __init__(self, reason: str, coverage: Optional[RawFragmentCoverage] = None):
        super().__init__(reason)
        self.reason = reason
        self.coverage = coverage or RawFragmentCoverage()


def _validate_depth(value: object) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise _Abstain("depth_m_must_be_numpy")
    if value.ndim != 2 or min(value.shape, default=0) < 1:
        raise _Abstain("invalid_depth_m")
    if int(value.shape[0]) * int(value.shape[1]) > _F_MAX_INPUT_DEPTH_PIXELS:
        raise _Abstain("depth_pixel_cap")
    try:
        raw = np.asarray(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise _Abstain("invalid_depth_m") from error
    if raw.dtype.kind not in "iuf":
        raise _Abstain("invalid_depth_m")
    try:
        return np.array(raw, dtype=np.float32, order="C", copy=True)
    except (TypeError, ValueError, OverflowError) as error:
        raise _Abstain("invalid_depth_m") from error


def _validate_image_shape(value: Sequence[int]) -> tuple[int, int]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence) or len(value) < 2:
        raise ValueError("proposal_image_shape must contain positive integer H,W")
    return (
        _strict_int("proposal_image_shape[0]", value[0]),
        _strict_int("proposal_image_shape[1]", value[1]),
    )


def aligned_resize_affine(
    proposal_image_shape: Sequence[int], depth_shape: Sequence[int]
) -> np.ndarray:
    """Construct the explicit resize-only proposal-to-depth registration."""

    source_height, source_width = _validate_image_shape(proposal_image_shape)
    depth_height, depth_width = _validate_image_shape(depth_shape)
    result = np.array(
        [
            [depth_width / source_width, 0.0, 0.0],
            [0.0, depth_height / source_height, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    result.setflags(write=False)
    return result


def _validate_registration_affine(
    value: object,
    proposal_image_shape: tuple[int, int],
    depth_shape: tuple[int, int],
) -> np.ndarray:
    try:
        affine = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as error:
        raise _Abstain("invalid_proposal_to_depth_affine") from error
    if affine.shape != (3, 3) or not np.isfinite(affine).all():
        raise _Abstain("invalid_proposal_to_depth_affine")
    tolerance = 1e-9
    if np.max(np.abs(affine[2] - [0.0, 0.0, 1.0])) > tolerance:
        raise _Abstain("invalid_proposal_to_depth_affine")
    if abs(float(affine[0, 1])) > tolerance or abs(float(affine[1, 0])) > tolerance:
        raise _Abstain("proposal_to_depth_affine_must_be_axis_aligned")
    if affine[0, 0] <= 0.0 or affine[1, 1] <= 0.0:
        raise _Abstain("proposal_to_depth_affine_must_have_positive_scale")

    source_height, source_width = proposal_image_shape
    depth_height, depth_width = depth_shape
    mapped_x = np.asarray(
        [affine[0, 2], affine[0, 0] * source_width + affine[0, 2]]
    )
    mapped_y = np.asarray(
        [affine[1, 2], affine[1, 1] * source_height + affine[1, 2]]
    )
    if (
        mapped_x[1] <= 0.0
        or mapped_x[0] >= depth_width
        or mapped_y[1] <= 0.0
        or mapped_y[0] >= depth_height
    ):
        raise _Abstain("proposal_to_depth_affine_has_no_overlap")
    return np.array(affine, dtype=np.float64, order="C", copy=True)


def _validate_intrinsics(value: object, depth_shape: tuple[int, int]) -> np.ndarray:
    try:
        matrix = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as error:
        raise _Abstain("invalid_intrinsics") from error
    if matrix.shape == (4, 4):
        matrix = matrix[:3, :3]
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        raise _Abstain("invalid_intrinsics")
    if matrix[0, 0] <= 0.0 or matrix[1, 1] <= 0.0:
        raise _Abstain("invalid_intrinsics")
    if abs(float(np.linalg.det(matrix))) <= 1e-12:
        raise _Abstain("invalid_intrinsics")
    height, width = depth_shape
    if not (0.0 <= matrix[0, 2] < width and 0.0 <= matrix[1, 2] < height):
        raise _Abstain("invalid_intrinsics")
    return np.array(matrix, dtype=np.float64, order="C", copy=True)


def _validate_pose(value: object) -> np.ndarray:
    try:
        pose = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as error:
        raise _Abstain("invalid_camera_to_world") from error
    if pose.shape != (4, 4) or not np.isfinite(pose).all():
        raise _Abstain("invalid_camera_to_world")
    if np.max(np.abs(pose[3] - np.asarray([0.0, 0.0, 0.0, 1.0]))) > 1e-7:
        raise _Abstain("invalid_camera_to_world")
    rotation = pose[:3, :3]
    if (
        np.max(np.abs(rotation.T @ rotation - np.eye(3))) > 1e-4
        or abs(float(np.linalg.det(rotation)) - 1.0) > 1e-4
    ):
        raise _Abstain("invalid_camera_to_world")
    return np.array(pose, dtype=np.float64, order="C", copy=True)


def _map_and_clip_box(
    box_xyxy: object,
    registration: np.ndarray,
    depth_shape: tuple[int, int],
) -> np.ndarray:
    try:
        raw_box = np.asarray(box_xyxy, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as error:
        raise _Abstain("invalid_box_xyxy") from error
    if raw_box.shape != (4,) or not np.isfinite(raw_box).all():
        raise _Abstain("invalid_box_xyxy")
    corners = np.asarray(
        [
            [raw_box[0], raw_box[1], 1.0],
            [raw_box[2], raw_box[1], 1.0],
            [raw_box[0], raw_box[3], 1.0],
            [raw_box[2], raw_box[3], 1.0],
        ],
        dtype=np.float64,
    )
    mapped = corners @ registration.T
    box = np.asarray(
        [
            np.min(mapped[:, 0]),
            np.min(mapped[:, 1]),
            np.max(mapped[:, 0]),
            np.max(mapped[:, 1]),
        ],
        dtype=np.float64,
    )
    depth_height, depth_width = depth_shape
    box[[0, 2]] = np.clip(box[[0, 2]], 0.0, float(depth_width - 1))
    box[[1, 3]] = np.clip(box[[1, 3]], 0.0, float(depth_height - 1))
    if box[2] <= box[0] or box[3] <= box[1]:
        raise _Abstain("empty_mapped_crop")
    return box


def _direct_voxel_keys(points_world: np.ndarray, voxel_size: float) -> np.ndarray:
    """Return the lexicographic set under the frozen signed-floor rule."""

    scaled = points_world.astype(np.float64, copy=False) / voxel_size
    limit = np.iinfo(np.int64).max / 4
    if not np.isfinite(scaled).all() or np.max(np.abs(scaled), initial=0.0) > limit:
        raise _Abstain("point_range_overflow")
    keys = np.floor(scaled).astype(np.int64)
    return np.unique(keys, axis=0)


def _bounded_voxel_keys(keys: np.ndarray, maximum: int) -> np.ndarray:
    if len(keys) <= maximum:
        return np.array(keys, dtype=np.int64, order="C", copy=True)
    # _direct_voxel_keys is already lexicographically sorted.  Sample its
    # positions directly; never form centroids and never re-floor floats.
    positions = np.linspace(0, len(keys) - 1, maximum, dtype=np.int64)
    return np.array(keys[positions], dtype=np.int64, order="C", copy=True)


def _extract_validated(
    *,
    proposal_id: int,
    frame_id: int,
    score: float,
    box_xyxy: object,
    proposal_to_depth_affine: np.ndarray,
    depth: np.ndarray,
    intrinsics: np.ndarray,
    camera_to_world: np.ndarray,
    cfg: Mapping[str, object],
) -> RawViewFragment:
    box = _map_and_clip_box(box_xyxy, proposal_to_depth_affine, depth.shape)
    row_min, row_max = int(np.ceil(box[1])), int(np.floor(box[3]))
    col_min, col_max = int(np.ceil(box[0])), int(np.floor(box[2]))
    if row_min > row_max or col_min > col_max:
        raise _Abstain("empty_mapped_crop")

    stride = int(cfg["pixel_stride"])
    maximum_rays = int(cfg["max_rays_per_proposal"])
    rows = np.arange(row_min, row_max + 1, stride, dtype=np.int64)
    cols = np.arange(col_min, col_max + 1, stride, dtype=np.int64)
    while len(rows) * len(cols) > maximum_rays:
        stride += 1
        rows = np.arange(row_min, row_max + 1, stride, dtype=np.int64)
        cols = np.arange(col_min, col_max + 1, stride, dtype=np.int64)
    if not len(rows) or not len(cols):
        raise _Abstain("no_sampled_rays")

    grid_cols, grid_rows = np.meshgrid(cols, rows)
    sampled = depth[grid_rows, grid_cols]
    usable = (
        np.isfinite(sampled)
        & (sampled >= float(cfg["min_depth_m"]))
        & (sampled <= float(cfg["max_depth_m"]))
    )
    sampled_count = int(sampled.size)
    usable_count = int(np.count_nonzero(usable))

    def coverage(unique_voxels: int = 0, output_voxels: int = 0) -> RawFragmentCoverage:
        return RawFragmentCoverage(
            effective_stride=stride,
            sampled_rays=sampled_count,
            usable_rays=usable_count,
            unique_voxels=unique_voxels,
            output_voxels=output_voxels,
            valid_depth_ratio=usable_count / sampled_count,
        )

    if usable_count < int(cfg["min_fragment_points"]):
        raise _Abstain("insufficient_valid_depth_pixels", coverage())

    valid_rows = grid_rows[usable].astype(np.float64)
    valid_cols = grid_cols[usable].astype(np.float64)
    valid_depth = sampled[usable].astype(np.float64)
    pixels = np.column_stack(
        (valid_cols, valid_rows, np.ones(usable_count, dtype=np.float64))
    )
    rays_camera = pixels @ np.linalg.inv(intrinsics).T
    rays_camera /= rays_camera[:, 2:3]
    points_camera = rays_camera * valid_depth[:, None]
    points_world = points_camera @ camera_to_world[:3, :3].T + camera_to_world[:3, 3]

    keys = _direct_voxel_keys(points_world, float(cfg["voxel_size_m"]))
    unique_count = len(keys)
    keys = _bounded_voxel_keys(keys, int(cfg["max_points_per_view"]))
    if len(keys) < int(cfg["min_fragment_points"]):
        raise _Abstain(
            "insufficient_voxels_after_quantization",
            coverage(unique_count, len(keys)),
        )
    final_coverage = coverage(unique_count, len(keys))
    return RawViewFragment(
        proposal_id=proposal_id,
        frame_id=frame_id,
        score=score,
        crop_xyxy_depth=box,
        depth_shape=tuple(int(value) for value in depth.shape),
        proposal_to_depth_affine=proposal_to_depth_affine,
        intrinsics=intrinsics,
        camera_to_world=camera_to_world,
        voxel_keys=keys,
        coverage=final_coverage,
    )


def extract_fragment(
    *,
    proposal_id: int,
    frame_id: int,
    score: float,
    box_xyxy: object,
    proposal_image_shape: Sequence[int],
    proposal_to_depth_affine: object,
    depth_m: object,
    intrinsics: object,
    camera_to_world: object,
    config: Optional[Mapping[str, object]] = None,
) -> RawProposalDiagnostic:
    """Extract one uncleaned fragment; data failures return an abstention."""

    cfg = resolve_config(config)
    proposal_id = _strict_int("proposal_id", proposal_id, 0)
    frame_id = _strict_int("frame_id", frame_id, 0)
    score = _strict_real("score", score)
    image_shape = _validate_image_shape(proposal_image_shape)
    started = time.perf_counter_ns()
    try:
        depth = _validate_depth(depth_m)
        matrix = _validate_intrinsics(intrinsics, depth.shape)
        pose = _validate_pose(camera_to_world)
        registration = _validate_registration_affine(
            proposal_to_depth_affine, image_shape, depth.shape
        )
        fragment = _extract_validated(
            proposal_id=proposal_id,
            frame_id=frame_id,
            score=score,
            box_xyxy=box_xyxy,
            proposal_to_depth_affine=registration,
            depth=depth,
            intrinsics=matrix,
            camera_to_world=pose,
            cfg=cfg,
        )
        reason, coverage = None, fragment.coverage
    except (_Abstain, FloatingPointError, np.linalg.LinAlgError) as error:
        fragment = None
        if isinstance(error, _Abstain):
            reason, coverage = error.reason, error.coverage
        else:
            reason, coverage = "numeric_failure", RawFragmentCoverage()
    return RawProposalDiagnostic(
        proposal_id=proposal_id,
        selected=True,
        reason=reason,
        coverage=coverage,
        elapsed_ms=(time.perf_counter_ns() - started) / 1e6,
        fragment=fragment,
    )


class RawFragmentExtractor:
    """Stateless keyframe adapter with SMOV-comparable proposal membership."""

    def __setattr__(self, name: str, value: object) -> None:
        if name == "_config":
            try:
                object.__getattribute__(self, "_config")
            except AttributeError:
                pass
            else:
                raise AttributeError("_config is write-once")
        object.__setattr__(self, name, value)

    def __delattr__(self, name: str) -> None:
        if name == "_config":
            raise AttributeError("_config is write-once and cannot be deleted")
        object.__delattr__(self, name)

    def __init__(self, config: Optional[Mapping[str, object]] = None):
        self._config = resolve_config(config)

    @property
    def config(self) -> Mapping[str, object]:
        return self._config

    def prepare_keyframe(
        self,
        *,
        scene_id: str,
        frame_id: int,
        proposal_ids: object,
        boxes_xyxy: object,
        proposal_scores: object,
        proposal_image_shape: Sequence[int],
        proposal_to_depth_affine: object,
        depth_m: object,
        intrinsics: object,
        camera_to_world: object,
    ) -> PreparedRawKeyframe:
        started = time.perf_counter_ns()
        if not isinstance(scene_id, str) or not scene_id:
            raise ValueError("scene_id must be a non-empty string")
        frame_id = _strict_int("frame_id", frame_id, 0)
        for name, value in (
            ("proposal_ids", proposal_ids),
            ("boxes_xyxy", boxes_xyxy),
            ("proposal_scores", proposal_scores),
        ):
            try:
                input_count = len(value)  # type: ignore[arg-type]
            except TypeError as error:
                raise ValueError(f"{name} must be a sized row sequence") from error
            if input_count > _F_MAX_INPUT_PROPOSALS:
                raise ValueError(
                    f"{name} exceeds the hard input proposal cap of "
                    f"{_F_MAX_INPUT_PROPOSALS}"
                )

        ids_raw = np.asarray(proposal_ids)
        if ids_raw.ndim != 1 or ids_raw.dtype.kind not in "iu":
            raise ValueError("proposal_ids must be a one-dimensional integer array")
        ids = ids_raw.astype(np.int64, copy=True)
        if np.any(ids < 0) or len(np.unique(ids)) != len(ids):
            raise ValueError("proposal_ids must be unique and nonnegative")
        try:
            boxes = np.asarray(boxes_xyxy)
            scores = np.asarray(proposal_scores)
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError("proposal arrays must be rectangular and row aligned") from error
        if boxes.shape != (len(ids), 4) or scores.shape != (len(ids),):
            raise ValueError("proposal arrays must be row aligned")
        if scores.dtype.kind not in "iuf" or not np.isfinite(scores).all():
            raise ValueError("proposal_scores must be finite numeric values")
        boxes = np.array(boxes, order="C", copy=True)
        scores = np.array(scores, dtype=np.float64, order="C", copy=True)
        image_shape = _validate_image_shape(proposal_image_shape)

        cap = int(self._config["max_proposals_per_keyframe"])
        order = np.lexsort((ids, -scores))
        selected_order = tuple(int(index) for index in order[:cap])
        selected_indices = set(selected_order)
        diagnostics: list[Optional[RawProposalDiagnostic]] = [None] * len(ids)
        for index, proposal_id in enumerate(ids):
            if index not in selected_indices:
                diagnostics[index] = RawProposalDiagnostic(
                    proposal_id=int(proposal_id),
                    selected=False,
                    reason="proposal_cap",
                    coverage=RawFragmentCoverage(),
                    elapsed_ms=0.0,
                    fragment=None,
                )

        try:
            depth = _validate_depth(depth_m)
            matrix = _validate_intrinsics(intrinsics, depth.shape)
            pose = _validate_pose(camera_to_world)
            registration = _validate_registration_affine(
                proposal_to_depth_affine, image_shape, depth.shape
            )
            global_failure: Optional[_Abstain] = None
        except _Abstain as error:
            depth = np.empty((0, 0), dtype=np.float32)
            matrix = np.eye(3, dtype=np.float64)
            pose = np.eye(4, dtype=np.float64)
            registration = np.eye(3, dtype=np.float64)
            global_failure = error

        for index in sorted(selected_indices):
            proposal_started = time.perf_counter_ns()
            fragment: Optional[RawViewFragment] = None
            try:
                if global_failure is not None:
                    raise global_failure
                fragment = _extract_validated(
                    proposal_id=int(ids[index]),
                    frame_id=frame_id,
                    score=float(scores[index]),
                    box_xyxy=boxes[index],
                    proposal_to_depth_affine=registration,
                    depth=depth,
                    intrinsics=matrix,
                    camera_to_world=pose,
                    cfg=self._config,
                )
                reason, coverage = None, fragment.coverage
            except (_Abstain, FloatingPointError, np.linalg.LinAlgError) as error:
                if isinstance(error, _Abstain):
                    reason, coverage = error.reason, error.coverage
                else:
                    reason, coverage = "numeric_failure", RawFragmentCoverage()
            diagnostics[index] = RawProposalDiagnostic(
                proposal_id=int(ids[index]),
                selected=True,
                reason=reason,
                coverage=coverage,
                elapsed_ms=(time.perf_counter_ns() - proposal_started) / 1e6,
                fragment=fragment,
            )

        completed = tuple(item for item in diagnostics if item is not None)
        selected_ids = tuple(int(ids[index]) for index in selected_order)
        return PreparedRawKeyframe(
            scene_id=scene_id,
            frame_id=frame_id,
            proposal_ids=tuple(int(value) for value in ids),
            selected_proposal_ids=selected_ids,
            diagnostics=completed,
            elapsed_ms=(time.perf_counter_ns() - started) / 1e6,
        )


__all__ = [
    "DEFAULT_CONFIG",
    "MAX_INPUT_DEPTH_PIXELS",
    "MAX_INPUT_PROPOSALS",
    "PreparedRawKeyframe",
    "RawFragmentCoverage",
    "RawFragmentExtractor",
    "RawProposalDiagnostic",
    "RawViewFragment",
    "SCHEMA",
    "VOXEL_SIZE_METERS",
    "aligned_resize_affine",
    "extract_fragment",
    "resolve_config",
]
