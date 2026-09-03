"""Deterministic DFU-LGF geometry hypotheses for frozen FastSAM candidates.

This module implements the output-inert F2 shadow experiment.  It consumes
only the bounded metric geometry already produced by F0: at most 2,048
representative points from unique 2 cm voxels and the original q02/q98 AABB.
It has no model, image, history, semantic, annotation, or ground-truth input
and cannot emit a birth or mutate a native prediction.

Three *grouped* hypotheses are returned for each source candidate:

``H0``
    The exact original F0 q02/q98 box.
``HL``
    A q02/q98 box after retaining points with at least three *other* points
    within a 6 cm Euclidean radius.  Neighbour counts use an exact, single-
    threaded SciPy cKDTree query; no quadratic all-pairs array is constructed.
``HLG``
    ``HL`` followed by a global radial robust filter.  From the coordinate-
    wise median, ``rho`` is thresholded at
    ``median(rho) + 3.5 * max(1.4826 * MAD(rho), 0.02)``.

If a stage leaves fewer than sixteen points, or cannot form a valid AABB, its
hypothesis fails open to the preceding effective hypothesis.  Every fitted
box uses q02/q98 with a 2 cm minimum extent.  Constants are deliberately not
configurable so an F2 replay cannot silently tune on ScanNet validation data.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import time
from types import MappingProxyType
from typing import Mapping, Optional

import numpy as np
from scipy.spatial import cKDTree


SCHEMA = "boxfusion.fastsam_dfu_lgf_shadow.f2.v1"
MODE = "shadow"

# Public audit constants.  Execution uses the private literals below so that
# rebinding a public module attribute cannot alter a sealed experiment.
VOXEL_SIZE_M = 0.02
MAX_POINTS = 2_048
MIN_POINTS = 16
LOCAL_RADIUS_M = 0.06
LOCAL_MIN_OTHER_NEIGHBORS = 3
GLOBAL_MAD_FACTOR = 1.4826
GLOBAL_SIGMA_MULTIPLIER = 3.5
GLOBAL_SCALE_FLOOR_M = 0.02
WORLD_QUANTILES = (0.02, 0.98)
MIN_AABB_EXTENT_M = 0.02

_F_VOXEL_SIZE_M = 0.02
_F_MAX_POINTS = 2_048
_F_MIN_POINTS = 16
_F_LOCAL_RADIUS_M = 0.06
_F_LOCAL_RADIUS_SQ = 0.06 * 0.06
_F_LOCAL_MIN_OTHER_NEIGHBORS = 3
_F_GLOBAL_MAD_FACTOR = 1.4826
_F_GLOBAL_SIGMA_MULTIPLIER = 3.5
_F_GLOBAL_SCALE_FLOOR_M = 0.02
_F_Q_LOW = 0.02
_F_Q_HIGH = 0.98
_F_MIN_AABB_EXTENT_M = 0.02

POLICY: Mapping[str, object] = MappingProxyType(
    {
        "input": "f0_selected_2cm_voxel_representatives",
        "point_count": (_F_MIN_POINTS, _F_MAX_POINTS),
        "voxel_size_m": _F_VOXEL_SIZE_M,
        "local_radius_m": _F_LOCAL_RADIUS_M,
        "local_min_other_neighbors": _F_LOCAL_MIN_OTHER_NEIGHBORS,
        "local_index": "scipy_ckdtree_query_pairs_then_exact_squared_predicate",
        "global_center": "coordinate_median",
        "global_distance": "euclidean_rho",
        "global_scale": "max(1.4826*MAD,0.02m)",
        "global_threshold": "median_rho+3.5*scale",
        "world_quantiles": (_F_Q_LOW, _F_Q_HIGH),
        "min_aabb_extent_m": _F_MIN_AABB_EXTENT_M,
        "stage_fail_open_min_points": _F_MIN_POINTS,
        "training": False,
        "ground_truth": False,
        "history": False,
        "birth": False,
    }
)


def _readonly(value: object, dtype: np.dtype, shape: Optional[tuple[int, ...]] = None) -> np.ndarray:
    """Return a detached NumPy array backed by immutable bytes."""

    array = np.asarray(value, dtype=dtype)
    if shape is not None and array.shape != shape:
        raise ValueError(f"array must have shape {shape}, got {array.shape}")
    packed = np.ascontiguousarray(array).tobytes()
    return np.frombuffer(packed, dtype=dtype).reshape(array.shape)


@dataclass(frozen=True)
class AABBHypothesis:
    """One immutable grouped geometry hypothesis for a source candidate."""

    name: str
    world_q02: np.ndarray
    world_q98: np.ndarray
    world_center: np.ndarray
    world_extent: np.ndarray
    retained_indices: np.ndarray
    source_point_count: int
    fallback_from: Optional[str]
    reason: str

    def __post_init__(self) -> None:
        if self.name not in {"h0", "hl", "hlg"}:
            raise ValueError("unknown geometry hypothesis name")
        q02 = _readonly(self.world_q02, np.float64, (3,))
        q98 = _readonly(self.world_q98, np.float64, (3,))
        center = _readonly(self.world_center, np.float64, (3,))
        extent = _readonly(self.world_extent, np.float64, (3,))
        indices = _readonly(self.retained_indices, np.int64)
        if indices.ndim != 1 or len(indices) < _F_MIN_POINTS:
            raise ValueError("retained_indices must contain at least sixteen rows")
        if np.any(indices < 0) or (len(indices) > 1 and np.any(indices[1:] <= indices[:-1])):
            raise ValueError("retained_indices must be sorted unique nonnegative indices")
        if self.source_point_count < len(indices) or self.source_point_count > _F_MAX_POINTS:
            raise ValueError("source_point_count is inconsistent with retained_indices")
        if len(indices) and int(indices[-1]) >= self.source_point_count:
            raise ValueError("retained_indices exceed source_point_count")
        if not (
            np.isfinite(q02).all()
            and np.isfinite(q98).all()
            and np.isfinite(center).all()
            and np.isfinite(extent).all()
        ):
            raise ValueError("hypothesis geometry must be finite")
        if np.any(q98 < q02) or np.any(extent < _F_MIN_AABB_EXTENT_M - 1e-12):
            raise ValueError("hypothesis violates the 2 cm minimum extent")
        if not np.allclose(center, (q02 + q98) * 0.5, rtol=0.0, atol=1e-12):
            raise ValueError("hypothesis center is inconsistent with its bounds")
        if not np.allclose(extent, q98 - q02, rtol=0.0, atol=1e-12):
            raise ValueError("hypothesis extent is inconsistent with its bounds")
        if self.fallback_from not in {None, "h0", "hl"}:
            raise ValueError("invalid fallback source")
        object.__setattr__(self, "world_q02", q02)
        object.__setattr__(self, "world_q98", q98)
        object.__setattr__(self, "world_center", center)
        object.__setattr__(self, "world_extent", extent)
        object.__setattr__(self, "retained_indices", indices)

    @property
    def point_count(self) -> int:
        return int(len(self.retained_indices))

    @property
    def failed_open(self) -> bool:
        return self.fallback_from is not None


@dataclass(frozen=True)
class DFULGFDiagnostics:
    """Per-candidate accounting and non-output-affecting timing."""

    input_point_count: int
    spatial_hash_bucket_count: int
    spatial_hash_bucket_probes: int
    local_distance_pair_tests: int
    local_retained_before_fallback: int
    local_effective_count: int
    local_reason: str
    global_source_count: int
    global_retained_before_fallback: int
    global_effective_count: int
    global_reason: str
    rho_median_m: Optional[float]
    rho_mad_m: Optional[float]
    rho_scale_m: Optional[float]
    rho_threshold_m: Optional[float]
    validation_elapsed_ms: float
    local_elapsed_ms: float
    global_elapsed_ms: float
    total_elapsed_ms: float

    def __post_init__(self) -> None:
        count_fields = (
            self.input_point_count,
            self.spatial_hash_bucket_count,
            self.spatial_hash_bucket_probes,
            self.local_distance_pair_tests,
            self.local_retained_before_fallback,
            self.local_effective_count,
            self.global_source_count,
            self.global_retained_before_fallback,
            self.global_effective_count,
        )
        if any(value < 0 for value in count_fields):
            raise ValueError("diagnostic counts cannot be negative")
        optional_reals = (
            self.rho_median_m,
            self.rho_mad_m,
            self.rho_scale_m,
            self.rho_threshold_m,
        )
        if any(value is not None and not np.isfinite(value) for value in optional_reals):
            raise ValueError("global diagnostic measurements must be finite or None")
        timings = (
            self.validation_elapsed_ms,
            self.local_elapsed_ms,
            self.global_elapsed_ms,
            self.total_elapsed_ms,
        )
        if any(not np.isfinite(value) or value < 0.0 for value in timings):
            raise ValueError("timings must be finite and nonnegative")


@dataclass(frozen=True)
class DFULGFShadowResult:
    """Immutable F2 shadow result; hypotheses share one source identity."""

    h0: AABBHypothesis
    hl: AABBHypothesis
    hlg: AABBHypothesis
    diagnostics: DFULGFDiagnostics
    input_sha256: str
    result_sha256: str
    schema: str = SCHEMA
    mode: str = MODE

    def __post_init__(self) -> None:
        if self.schema != SCHEMA or self.mode != MODE:
            raise ValueError("F2 schema or mode mismatch")
        if (self.h0.name, self.hl.name, self.hlg.name) != ("h0", "hl", "hlg"):
            raise ValueError("F2 hypotheses must be ordered h0, hl, hlg")
        for digest in (self.input_sha256, self.result_sha256):
            if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
                raise ValueError("F2 digests must be lowercase SHA256 strings")


def _validate_inputs(
    points_world: object,
    world_q02: object,
    world_q98: object,
    voxel_keys: object,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if not isinstance(points_world, np.ndarray):
        raise ValueError("points_world must be a numpy array with shape [N,3]")
    if points_world.ndim != 2 or points_world.shape[1:] != (3,) or points_world.dtype.kind not in "iuf":
        raise ValueError("points_world must be a numeric numpy array with shape [N,3]")
    if not (_F_MIN_POINTS <= len(points_world) <= _F_MAX_POINTS):
        raise ValueError("points_world count must be between 16 and 2048")
    points = np.array(points_world, dtype=np.float64, order="C", copy=True)
    if not np.isfinite(points).all():
        raise ValueError("points_world must contain only finite values")
    scaled = points / _F_VOXEL_SIZE_M
    if np.max(np.abs(scaled), initial=0.0) > np.iinfo(np.int64).max / 4:
        raise ValueError("points_world exceed the safe 2 cm voxel coordinate range")
    derived_keys = np.floor(scaled).astype(np.int64)
    if len(np.unique(derived_keys, axis=0)) != len(points):
        raise ValueError("points_world must be unique 2 cm voxel representatives")

    if voxel_keys is None:
        keys = derived_keys
    else:
        if not isinstance(voxel_keys, np.ndarray):
            raise ValueError("voxel_keys must be a signed integer numpy array")
        if voxel_keys.shape != points.shape or not np.issubdtype(voxel_keys.dtype, np.signedinteger):
            raise ValueError("voxel_keys must be signed integer [N,3] aligned with points_world")
        keys = np.array(voxel_keys, dtype=np.int64, order="C", copy=True)
        if not np.array_equal(keys, derived_keys):
            raise ValueError("voxel_keys must equal floor(points_world/0.02)")
        if len(np.unique(keys, axis=0)) != len(keys):
            raise ValueError("voxel_keys must be unique")

    try:
        q02 = np.asarray(world_q02, dtype=np.float64)
        q98 = np.asarray(world_q98, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("world_q02 and world_q98 must be finite [3] arrays") from error
    if q02.shape != (3,) or q98.shape != (3,) or not np.isfinite(q02).all() or not np.isfinite(q98).all():
        raise ValueError("world_q02 and world_q98 must be finite [3] arrays")
    q02 = np.array(q02, dtype=np.float64, order="C", copy=True)
    q98 = np.array(q98, dtype=np.float64, order="C", copy=True)
    if np.any(q98 < q02) or np.any(q98 - q02 < _F_MIN_AABB_EXTENT_M - 1e-12):
        raise ValueError("H0 bounds must satisfy the 2 cm minimum extent")
    return points, keys, q02, q98


def _hypothesis(
    name: str,
    q02: np.ndarray,
    q98: np.ndarray,
    retained_indices: np.ndarray,
    source_point_count: int,
    fallback_from: Optional[str],
    reason: str,
) -> AABBHypothesis:
    return AABBHypothesis(
        name=name,
        world_q02=q02,
        world_q98=q98,
        world_center=(q02 + q98) * 0.5,
        world_extent=q98 - q02,
        retained_indices=retained_indices,
        source_point_count=source_point_count,
        fallback_from=fallback_from,
        reason=reason,
    )


def _fit_q02_q98(points: np.ndarray) -> Optional[tuple[np.ndarray, np.ndarray]]:
    """Fit the frozen robust AABB, returning ``None`` for invalid geometry."""

    try:
        raw = np.quantile(points, [_F_Q_LOW, _F_Q_HIGH], axis=0, method="linear")
    except (FloatingPointError, TypeError, ValueError, OverflowError):
        return None
    raw_q02 = np.asarray(raw[0], dtype=np.float64)
    raw_q98 = np.asarray(raw[1], dtype=np.float64)
    if raw_q02.shape != (3,) or raw_q98.shape != (3,) or not np.isfinite(raw).all():
        return None
    center = (raw_q02 + raw_q98) * 0.5
    extent = np.maximum(raw_q98 - raw_q02, _F_MIN_AABB_EXTENT_M)
    q02 = center - extent * 0.5
    q98 = center + extent * 0.5
    if not np.isfinite(q02).all() or not np.isfinite(q98).all():
        return None
    return q02, q98


# Only one member of each symmetric neighbouring-cell pair is visited.
_HALF_NEIGHBOR_OFFSETS = tuple(
    (dx, dy, dz)
    for dx in (-1, 0, 1)
    for dy in (-1, 0, 1)
    for dz in (-1, 0, 1)
    if (dx, dy, dz) >= (0, 0, 0)
)


def _local_neighbor_counts_fixed_hash_reference(
    points: np.ndarray,
) -> tuple[np.ndarray, int, int, int]:
    """Slow audit reference for the exact 6 cm neighbour rule."""

    # A cell side equal to the query radius means only the current and 26
    # adjacent cells can contain a neighbour.  Unique 2 cm input voxels bound
    # the occupancy of each 6 cm cell by a fixed constant.
    hash_keys = np.floor(points / _F_LOCAL_RADIUS_M).astype(np.int64)
    mutable: dict[tuple[int, int, int], list[int]] = {}
    for index, key_row in enumerate(hash_keys):
        key = (int(key_row[0]), int(key_row[1]), int(key_row[2]))
        mutable.setdefault(key, []).append(index)
    buckets = {
        key: np.asarray(indices, dtype=np.int64)
        for key, indices in mutable.items()
    }
    counts = np.zeros(len(points), dtype=np.int32)
    bucket_probes = 0
    pair_tests = 0
    radius_sq_inclusive = np.nextafter(_F_LOCAL_RADIUS_SQ, np.inf)

    for key in sorted(buckets):
        left_indices = buckets[key]
        left_points = points[left_indices]
        for offset in _HALF_NEIGHBOR_OFFSETS:
            bucket_probes += 1
            other_key = (
                key[0] + offset[0],
                key[1] + offset[1],
                key[2] + offset[2],
            )
            right_indices = buckets.get(other_key)
            if right_indices is None:
                continue
            if offset == (0, 0, 0):
                size = len(left_indices)
                if size < 2:
                    continue
                differences = left_points[:, None, :] - left_points[None, :, :]
                distance_sq = np.einsum("ijk,ijk->ij", differences, differences)
                within = distance_sq <= radius_sq_inclusive
                np.fill_diagonal(within, False)
                counts[left_indices] += np.sum(within, axis=1, dtype=np.int32)
                pair_tests += size * (size - 1) // 2
            else:
                right_points = points[right_indices]
                differences = left_points[:, None, :] - right_points[None, :, :]
                distance_sq = np.einsum("ijk,ijk->ij", differences, differences)
                within = distance_sq <= radius_sq_inclusive
                counts[left_indices] += np.sum(within, axis=1, dtype=np.int32)
                counts[right_indices] += np.sum(within, axis=0, dtype=np.int32)
                pair_tests += len(left_indices) * len(right_indices)
    return counts, len(buckets), bucket_probes, pair_tests


def _local_neighbor_counts(points: np.ndarray) -> tuple[np.ndarray, int, int, int]:
    """Count exact 6 cm neighbours with a deterministic compiled query.

    cKDTree first enumerates a conservative pair superset.  The original
    squared-distance predicate is then applied explicitly, including its
    one-ULP boundary allowance.  Thus the compiled index changes only how
    candidate pairs are found, not which pairs pass.  ``eps=0`` prohibits
    approximate search.  The three accounting values retain the original
    diagnostic shape: occupied 6 cm audit cells, the fixed-hash reference
    probe envelope, and the number of accepted unordered neighbour pairs.
    They do not affect geometry.
    """

    tree = cKDTree(
        points,
        leafsize=16,
        compact_nodes=True,
        copy_data=True,
        balanced_tree=True,
    )
    radius_sq_inclusive = np.nextafter(_F_LOCAL_RADIUS_SQ, np.inf)
    conservative_outer_radius = np.nextafter(
        np.sqrt(radius_sq_inclusive), np.inf
    )
    candidate_pairs = np.asarray(
        tree.query_pairs(
            r=conservative_outer_radius,
            p=2.0,
            eps=0.0,
            output_type="ndarray",
        ),
        dtype=np.int64,
    )
    if candidate_pairs.size:
        candidate_pairs = candidate_pairs.reshape(-1, 2)
        differences = (
            points[candidate_pairs[:, 0]] - points[candidate_pairs[:, 1]]
        )
        distance_sq = np.einsum("ij,ij->i", differences, differences)
        accepted_pairs = candidate_pairs[
            distance_sq <= radius_sq_inclusive
        ]
        counts = np.bincount(
            accepted_pairs.reshape(-1), minlength=len(points)
        ).astype(np.int32, copy=False)
    else:
        accepted_pairs = np.empty((0, 2), dtype=np.int64)
        counts = np.zeros(len(points), dtype=np.int32)
    if counts.shape != (len(points),):
        raise ValueError("exact local-neighbour query returned invalid counts")
    audit_keys = np.floor(points / _F_LOCAL_RADIUS_M).astype(np.int64)
    audit_bucket_count = int(len(np.unique(audit_keys, axis=0)))
    audit_bucket_probes = audit_bucket_count * len(_HALF_NEIGHBOR_OFFSETS)
    returned_unordered_pairs = int(len(accepted_pairs))
    return counts, audit_bucket_count, audit_bucket_probes, returned_unordered_pairs


def _input_digest(points: np.ndarray, keys: np.ndarray, q02: np.ndarray, q98: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(np.asarray(points, dtype="<f8").tobytes())
    digest.update(np.asarray(keys, dtype="<i8").tobytes())
    digest.update(np.asarray(q02, dtype="<f8").tobytes())
    digest.update(np.asarray(q98, dtype="<f8").tobytes())
    return digest.hexdigest()


def _result_digest(input_sha256: str, hypotheses: tuple[AABBHypothesis, ...]) -> str:
    digest = hashlib.sha256(bytes.fromhex(input_sha256))
    for item in hypotheses:
        digest.update(item.name.encode("ascii"))
        digest.update(np.asarray(item.world_q02, dtype="<f8").tobytes())
        digest.update(np.asarray(item.world_q98, dtype="<f8").tobytes())
        digest.update(np.asarray(item.retained_indices, dtype="<i8").tobytes())
        digest.update((item.fallback_from or "").encode("ascii"))
        digest.update(item.reason.encode("ascii"))
    return digest.hexdigest()


def refine_fastsam_candidate(
    *,
    points_world: np.ndarray,
    world_q02: np.ndarray,
    world_q98: np.ndarray,
    voxel_keys: Optional[np.ndarray] = None,
) -> DFULGFShadowResult:
    """Construct H0/HL/HLG for one frozen F0 selected candidate.

    The function is stateless and current-candidate only.  Structural contract
    violations raise ``ValueError``; data-dependent stage failures retain the
    preceding hypothesis and are recorded in diagnostics.
    """

    total_started = time.perf_counter_ns()
    validation_started = total_started
    points, keys, q02, q98 = _validate_inputs(
        points_world, world_q02, world_q98, voxel_keys
    )
    all_indices = np.arange(len(points), dtype=np.int64)
    h0 = _hypothesis(
        "h0", q02, q98, all_indices, len(points), None, "native_f0"
    )
    input_sha256 = _input_digest(points, keys, q02, q98)
    validation_elapsed_ms = (time.perf_counter_ns() - validation_started) / 1e6

    local_started = time.perf_counter_ns()
    neighbor_counts, bucket_count, bucket_probes, pair_tests = _local_neighbor_counts(points)
    local_selected = np.flatnonzero(
        neighbor_counts >= _F_LOCAL_MIN_OTHER_NEIGHBORS
    ).astype(np.int64, copy=False)
    local_before_fallback = int(len(local_selected))
    local_fit = _fit_q02_q98(points[local_selected]) if len(local_selected) >= _F_MIN_POINTS else None
    if len(local_selected) < _F_MIN_POINTS:
        local_effective = all_indices
        h1 = _hypothesis(
            "hl", q02, q98, local_effective, len(points), "h0", "too_few_points"
        )
        local_reason = "too_few_points"
    elif local_fit is None:
        local_effective = all_indices
        h1 = _hypothesis(
            "hl", q02, q98, local_effective, len(points), "h0", "invalid_aabb"
        )
        local_reason = "invalid_aabb"
    else:
        local_effective = local_selected
        h1 = _hypothesis(
            "hl",
            local_fit[0],
            local_fit[1],
            local_effective,
            len(points),
            None,
            "accepted",
        )
        local_reason = "accepted"
    local_elapsed_ms = (time.perf_counter_ns() - local_started) / 1e6

    global_started = time.perf_counter_ns()
    global_source = points[local_effective]
    rho_median: Optional[float] = None
    rho_mad: Optional[float] = None
    rho_scale: Optional[float] = None
    rho_threshold: Optional[float] = None
    global_selected = np.empty(0, dtype=np.int64)
    global_fit: Optional[tuple[np.ndarray, np.ndarray]] = None
    try:
        coordinate_median = np.median(global_source, axis=0)
        rho = np.linalg.norm(global_source - coordinate_median, axis=1)
        rho_median = float(np.median(rho))
        rho_mad = float(np.median(np.abs(rho - rho_median)))
        rho_scale = float(max(_F_GLOBAL_MAD_FACTOR * rho_mad, _F_GLOBAL_SCALE_FLOOR_M))
        rho_threshold = float(
            rho_median + _F_GLOBAL_SIGMA_MULTIPLIER * rho_scale
        )
        if not np.isfinite(coordinate_median).all() or not all(
            np.isfinite(value)
            for value in (rho_median, rho_mad, rho_scale, rho_threshold)
        ):
            raise FloatingPointError("invalid robust radial statistics")
        keep = rho <= np.nextafter(rho_threshold, np.inf)
        global_selected = local_effective[keep]
        if len(global_selected) >= _F_MIN_POINTS:
            global_fit = _fit_q02_q98(points[global_selected])
    except (FloatingPointError, TypeError, ValueError, OverflowError):
        global_selected = np.empty(0, dtype=np.int64)
        global_fit = None

    global_before_fallback = int(len(global_selected))
    if len(global_selected) < _F_MIN_POINTS:
        h2 = _hypothesis(
            "hlg",
            h1.world_q02,
            h1.world_q98,
            h1.retained_indices,
            len(points),
            "hl",
            "too_few_points" if global_selected.size else "invalid_statistics_or_too_few_points",
        )
        global_reason = h2.reason
    elif global_fit is None:
        h2 = _hypothesis(
            "hlg",
            h1.world_q02,
            h1.world_q98,
            h1.retained_indices,
            len(points),
            "hl",
            "invalid_aabb",
        )
        global_reason = "invalid_aabb"
    else:
        h2 = _hypothesis(
            "hlg",
            global_fit[0],
            global_fit[1],
            global_selected,
            len(points),
            None,
            "accepted",
        )
        global_reason = "accepted"
    global_elapsed_ms = (time.perf_counter_ns() - global_started) / 1e6

    result_sha256 = _result_digest(input_sha256, (h0, h1, h2))
    total_elapsed_ms = (time.perf_counter_ns() - total_started) / 1e6
    diagnostics = DFULGFDiagnostics(
        input_point_count=len(points),
        spatial_hash_bucket_count=bucket_count,
        spatial_hash_bucket_probes=bucket_probes,
        local_distance_pair_tests=pair_tests,
        local_retained_before_fallback=local_before_fallback,
        local_effective_count=h1.point_count,
        local_reason=local_reason,
        global_source_count=len(local_effective),
        global_retained_before_fallback=global_before_fallback,
        global_effective_count=h2.point_count,
        global_reason=global_reason,
        rho_median_m=rho_median,
        rho_mad_m=rho_mad,
        rho_scale_m=rho_scale,
        rho_threshold_m=rho_threshold,
        validation_elapsed_ms=validation_elapsed_ms,
        local_elapsed_ms=local_elapsed_ms,
        global_elapsed_ms=global_elapsed_ms,
        total_elapsed_ms=total_elapsed_ms,
    )
    return DFULGFShadowResult(
        h0=h0,
        hl=h1,
        hlg=h2,
        diagnostics=diagnostics,
        input_sha256=input_sha256,
        result_sha256=result_sha256,
    )


def _hypothesis_to_dict(value: AABBHypothesis) -> dict[str, object]:
    return {
        "name": value.name,
        "world_q02": value.world_q02.tolist(),
        "world_q98": value.world_q98.tolist(),
        "world_center": value.world_center.tolist(),
        "world_extent": value.world_extent.tolist(),
        "retained_indices": value.retained_indices.tolist(),
        "point_count": value.point_count,
        "source_point_count": value.source_point_count,
        "fallback_from": value.fallback_from,
        "reason": value.reason,
    }


def dfu_lgf_result_to_dict(value: DFULGFShadowResult) -> dict[str, object]:
    """Convert one result to a deterministic JSON-compatible record."""

    diagnostics = value.diagnostics
    return {
        "schema": value.schema,
        "mode": value.mode,
        "policy": dict(POLICY),
        "input_sha256": value.input_sha256,
        "result_sha256": value.result_sha256,
        "hypotheses": {
            "h0": _hypothesis_to_dict(value.h0),
            "hl": _hypothesis_to_dict(value.hl),
            "hlg": _hypothesis_to_dict(value.hlg),
        },
        "diagnostics": {
            "input_point_count": diagnostics.input_point_count,
            "spatial_hash_bucket_count": diagnostics.spatial_hash_bucket_count,
            "spatial_hash_bucket_probes": diagnostics.spatial_hash_bucket_probes,
            "local_distance_pair_tests": diagnostics.local_distance_pair_tests,
            "local_retained_before_fallback": diagnostics.local_retained_before_fallback,
            "local_effective_count": diagnostics.local_effective_count,
            "local_reason": diagnostics.local_reason,
            "global_source_count": diagnostics.global_source_count,
            "global_retained_before_fallback": diagnostics.global_retained_before_fallback,
            "global_effective_count": diagnostics.global_effective_count,
            "global_reason": diagnostics.global_reason,
            "rho_median_m": diagnostics.rho_median_m,
            "rho_mad_m": diagnostics.rho_mad_m,
            "rho_scale_m": diagnostics.rho_scale_m,
            "rho_threshold_m": diagnostics.rho_threshold_m,
            "validation_elapsed_ms": diagnostics.validation_elapsed_ms,
            "local_elapsed_ms": diagnostics.local_elapsed_ms,
            "global_elapsed_ms": diagnostics.global_elapsed_ms,
            "total_elapsed_ms": diagnostics.total_elapsed_ms,
        },
    }
