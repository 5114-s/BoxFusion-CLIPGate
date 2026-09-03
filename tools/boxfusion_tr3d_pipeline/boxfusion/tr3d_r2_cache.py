"""Immutable R2a depth-evidence sidecars for TR3D proposal caches.

An R2a file never owns proposal geometry and never mutates its parent TR3D
cache.  It stores only verifier evidence and a fail-closed provenance chain
back to the exact parent NPZ bytes, causal prefix artifacts, and R2 code and
configuration.  Loading always requires the authoritative parent cache and
the four external R2 provenance hashes.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from io import BytesIO
import math
import os
from pathlib import Path
import re
import tempfile
from typing import Mapping, Sequence

import numpy as np

from .tr3d_residual_cache import (
    TR3DResidualCache,
    load_tr3d_residual_cache,
)


TR3D_R2_CACHE_SCHEMA = "boxfusion.tr3d_r2a_depth_evidence.v1"
DEPTH_EVIDENCE_NAMES = (
    "support_fraction",
    "occluded_fraction",
    "free_space_fraction",
    "invalid_fraction",
)
_DEPTH_EVIDENCE_DIM = len(DEPTH_EVIDENCE_NAMES)
_SCENE_RE = re.compile(r"^scene[0-9]{4}_[0-9]{2}$")
_PREFIX_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

_FIELDS = frozenset(
    {
        "schema",
        "complete",
        "observer_only",
        "mutation_enabled",
        "applied_count",
        "scene_id",
        "sample_idx",
        "prefix_id",
        "prefix_fraction",
        "parent_cache_sha256",
        "parent_checkpoint_sha256",
        "parent_config_sha256",
        "parent_source_scene_sha256",
        "parent_axis_alignment_sha256",
        "prefix_manifest_row_sha256",
        "frame_artifact_tree_sha256",
        "r2_config_sha256",
        "r2_code_sha256",
        "proposal_ids",
        "lineage_ids",
        "topk_frame_ids",
        "topk_view_valid",
        "topk_projected_area_pixels",
        "topk_projected_area_fraction",
        "per_view_depth_evidence",
        "per_view_depth_counts",
        "per_view_point_count",
        "aggregate_depth_evidence",
        "aggregate_depth_counts",
        "aggregate_view_count",
        "aggregate_point_count",
        "runtime_s",
    }
)


@dataclass(frozen=True)
class TR3DR2Cache:
    """Canonical R2a verifier evidence for a subset of parent proposals.

    ``proposal_ids`` identify immutable rows in the parent TR3D cache.
    ``lineage_ids`` are stable R2 lineage identifiers; R2a itself keeps a
    one-to-one mapping and therefore requires both arrays to be unique.

    Top-K arrays have shape ``[P, K]``.  Valid views must occupy the leading
    slots of each row.  Invalid slots use frame id ``-1`` and zero evidence.
    Depth evidence uses :data:`DEPTH_EVIDENCE_NAMES` in its last dimension.
    The corresponding count arrays store the exact support, occluded,
    free-space, and invalid sampled-pixel counts.  Fractions and aggregate
    counts are deliberately redundant: the loader recomputes every one from
    ``per_view_depth_counts`` and rejects even finite in-range disagreement.
    """

    scene_id: str
    prefix_id: str
    prefix_fraction: float
    parent_cache_sha256: str
    parent_checkpoint_sha256: str
    parent_config_sha256: str
    parent_source_scene_sha256: str
    parent_axis_alignment_sha256: str
    prefix_manifest_row_sha256: str
    frame_artifact_tree_sha256: str
    r2_config_sha256: str
    r2_code_sha256: str
    proposal_ids: np.ndarray
    lineage_ids: np.ndarray
    topk_frame_ids: np.ndarray
    topk_view_valid: np.ndarray
    topk_projected_area_pixels: np.ndarray
    topk_projected_area_fraction: np.ndarray
    per_view_depth_evidence: np.ndarray
    per_view_depth_counts: np.ndarray
    per_view_point_count: np.ndarray
    aggregate_depth_evidence: np.ndarray
    aggregate_depth_counts: np.ndarray
    aggregate_view_count: np.ndarray
    aggregate_point_count: np.ndarray
    runtime_s: float = 0.0

    @property
    def sample_idx(self) -> str:
        return f"{self.scene_id}:{self.prefix_id}"

    @property
    def proposal_count(self) -> int:
        return int(np.asarray(self.proposal_ids).shape[0])

    @property
    def topk(self) -> int:
        value = np.asarray(self.topk_frame_ids)
        return int(value.shape[1]) if value.ndim == 2 else 0

    def as_npz_payload(self) -> dict[str, np.ndarray]:
        return {
            "schema": np.asarray(TR3D_R2_CACHE_SCHEMA),
            "complete": np.asarray(True, dtype=np.bool_),
            "observer_only": np.asarray(True, dtype=np.bool_),
            "mutation_enabled": np.asarray(False, dtype=np.bool_),
            "applied_count": np.asarray(0, dtype=np.int64),
            "scene_id": np.asarray(self.scene_id),
            "sample_idx": np.asarray(self.sample_idx),
            "prefix_id": np.asarray(self.prefix_id),
            "prefix_fraction": np.asarray(
                self.prefix_fraction, dtype=np.float64
            ),
            "parent_cache_sha256": np.asarray(self.parent_cache_sha256),
            "parent_checkpoint_sha256": np.asarray(
                self.parent_checkpoint_sha256
            ),
            "parent_config_sha256": np.asarray(self.parent_config_sha256),
            "parent_source_scene_sha256": np.asarray(
                self.parent_source_scene_sha256
            ),
            "parent_axis_alignment_sha256": np.asarray(
                self.parent_axis_alignment_sha256
            ),
            "prefix_manifest_row_sha256": np.asarray(
                self.prefix_manifest_row_sha256
            ),
            "frame_artifact_tree_sha256": np.asarray(
                self.frame_artifact_tree_sha256
            ),
            "r2_config_sha256": np.asarray(self.r2_config_sha256),
            "r2_code_sha256": np.asarray(self.r2_code_sha256),
            "proposal_ids": np.asarray(self.proposal_ids, dtype=np.int64),
            "lineage_ids": np.asarray(self.lineage_ids, dtype=np.int64),
            "topk_frame_ids": np.asarray(
                self.topk_frame_ids, dtype=np.int64
            ),
            "topk_view_valid": np.asarray(
                self.topk_view_valid, dtype=np.bool_
            ),
            "topk_projected_area_pixels": np.asarray(
                self.topk_projected_area_pixels, dtype=np.float32
            ),
            "topk_projected_area_fraction": np.asarray(
                self.topk_projected_area_fraction, dtype=np.float32
            ),
            "per_view_depth_evidence": np.asarray(
                self.per_view_depth_evidence, dtype=np.float32
            ),
            "per_view_depth_counts": np.asarray(
                self.per_view_depth_counts, dtype=np.int32
            ),
            "per_view_point_count": np.asarray(
                self.per_view_point_count, dtype=np.int32
            ),
            "aggregate_depth_evidence": np.asarray(
                self.aggregate_depth_evidence, dtype=np.float32
            ),
            "aggregate_depth_counts": np.asarray(
                self.aggregate_depth_counts, dtype=np.int64
            ),
            "aggregate_view_count": np.asarray(
                self.aggregate_view_count, dtype=np.int32
            ),
            "aggregate_point_count": np.asarray(
                self.aggregate_point_count, dtype=np.int64
            ),
            "runtime_s": np.asarray(self.runtime_s, dtype=np.float64),
        }


def sha256_file(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tr3d_r2_cache_path(
    root: str | os.PathLike[str],
    scene_id: str,
    prefix_id: str = "full",
) -> Path:
    if _SCENE_RE.fullmatch(scene_id) is None:
        raise ValueError(f"invalid ScanNet scene id: {scene_id!r}")
    if _PREFIX_RE.fullmatch(prefix_id) is None:
        raise ValueError(f"invalid prefix_id: {prefix_id!r}")
    return Path(root) / scene_id / f"{prefix_id}.r2a.npz"


def _scalar(values: Mapping[str, np.ndarray], name: str) -> np.ndarray:
    value = np.asarray(values[name])
    if value.shape != () or value.dtype.hasobject:
        raise ValueError(f"{name} must be a non-object scalar")
    return value


def _text(values: Mapping[str, np.ndarray], name: str) -> str:
    raw = _scalar(values, name).item()
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    if not isinstance(raw, str):
        raise ValueError(f"{name} must be a string scalar")
    return raw


def _typed_scalar(
    values: Mapping[str, np.ndarray], name: str, dtype: np.dtype
):
    value = _scalar(values, name)
    if value.dtype != np.dtype(dtype):
        raise ValueError(f"{name} must be a {np.dtype(dtype)} scalar")
    return value.item()


def _boolean(values: Mapping[str, np.ndarray], name: str) -> bool:
    return bool(_typed_scalar(values, name, np.bool_))


def _sha256(value: str, name: str) -> str:
    if value != value.lower() or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA256")
    return value


def _expected_sha256(value: str, name: str) -> str:
    return _sha256(str(value), f"expected_{name}")


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
    if any(
        expected is not None and actual != expected
        for actual, expected in zip(value.shape, shape)
    ):
        raise ValueError(f"{name} must have shape {shape}")
    return value


def _readonly(value: np.ndarray) -> np.ndarray:
    result = np.ascontiguousarray(value)
    result.setflags(write=False)
    return result


def depth_evidence_fractions(depth_counts: np.ndarray) -> np.ndarray:
    """Derive canonical float32 evidence fractions from category counts.

    The last axis is ordered as support, occluded, free-space, invalid.
    Rows with zero total count deterministically map to four zeros.  This
    helper is also the sole fraction implementation used by the loader, so a
    writer can construct arrays that satisfy the exact recomputation rule.
    """

    raw = np.asarray(depth_counts)
    if raw.ndim < 1 or raw.shape[-1] != _DEPTH_EVIDENCE_DIM:
        raise ValueError("depth_counts must end in four evidence categories")
    if not np.issubdtype(raw.dtype, np.integer) or np.any(raw < 0):
        raise ValueError("depth_counts must be nonnegative integers")
    counts = raw.astype(np.int64, copy=False)
    totals = counts.sum(axis=-1, dtype=np.int64)
    fractions64 = np.zeros(counts.shape, dtype=np.float64)
    np.divide(
        counts,
        totals[..., None],
        out=fractions64,
        where=totals[..., None] != 0,
    )
    return _readonly(fractions64.astype(np.float32))


def _verified_parent(
    path: str | os.PathLike[str],
) -> tuple[TR3DResidualCache, str]:
    parent_path = Path(path)
    if not parent_path.is_file():
        raise FileNotFoundError(parent_path)
    before = sha256_file(parent_path)
    parent = load_tr3d_residual_cache(parent_path)
    after = sha256_file(parent_path)
    if before != after:
        raise RuntimeError("parent TR3D cache changed while being verified")
    return parent, before


def _validate_parent_binding(
    cache: TR3DR2Cache,
    parent: TR3DResidualCache,
    parent_file_sha256: str,
) -> None:
    comparisons = {
        "parent cache": (cache.parent_cache_sha256, parent_file_sha256),
        "scene": (cache.scene_id, parent.scene_id),
        "prefix": (cache.prefix_id, parent.prefix_id),
        "parent checkpoint": (
            cache.parent_checkpoint_sha256,
            parent.checkpoint_sha256,
        ),
        "parent config": (
            cache.parent_config_sha256,
            parent.config_sha256,
        ),
        "parent source scene": (
            cache.parent_source_scene_sha256,
            parent.source_scene_sha256,
        ),
        "parent axis alignment": (
            cache.parent_axis_alignment_sha256,
            parent.axis_alignment_sha256,
        ),
    }
    for label, (observed, expected) in comparisons.items():
        if observed != expected:
            raise ValueError(f"R2a {label} provenance mismatch")
    if not math.isclose(
        cache.prefix_fraction,
        parent.prefix_fraction,
        rel_tol=0.0,
        abs_tol=0.0,
    ):
        raise ValueError("R2a prefix fraction provenance mismatch")

    parent_ids = np.asarray(parent.proposal_ids)
    index = {int(value): row for row, value in enumerate(parent_ids)}
    missing = [
        int(value) for value in cache.proposal_ids if int(value) not in index
    ]
    if missing:
        raise ValueError(
            "R2a proposal_ids are absent from parent TR3D cache: "
            f"{missing[:8]}"
        )


def validate_tr3d_r2_payload(
    values: Mapping[str, np.ndarray],
    *,
    parent_cache_path: str | os.PathLike[str],
    expected_prefix_manifest_row_sha256: str,
    expected_frame_artifact_tree_sha256: str,
    expected_r2_config_sha256: str,
    expected_r2_code_sha256: str,
    expected_scene_id: str | None = None,
    expected_prefix_id: str | None = None,
    expected_prefix_fraction: float | None = None,
    expected_allowed_frame_ids: Sequence[int] | np.ndarray | None = None,
) -> TR3DR2Cache:
    """Validate decoded sidecar arrays and every external provenance link."""

    fields = frozenset(values)
    if fields != _FIELDS:
        raise ValueError(
            "R2a cache fields disagree; "
            f"missing={sorted(_FIELDS-fields)}, "
            f"unknown={sorted(fields-_FIELDS)}"
        )
    if any(np.asarray(value).dtype.hasobject for value in values.values()):
        raise ValueError("R2a cache must not contain object arrays")
    if _text(values, "schema") != TR3D_R2_CACHE_SCHEMA:
        raise ValueError("unsupported R2a cache schema")
    if not _boolean(values, "complete"):
        raise ValueError("R2a cache is incomplete")
    if not _boolean(values, "observer_only"):
        raise ValueError("R2a cache is not observer-only")
    if _boolean(values, "mutation_enabled"):
        raise ValueError("R2a cache enables mutation")
    if int(_typed_scalar(values, "applied_count", np.int64)) != 0:
        raise ValueError("R2a observer applied_count must be zero")

    scene_id = _text(values, "scene_id")
    prefix_id = _text(values, "prefix_id")
    sample_idx = _text(values, "sample_idx")
    if _SCENE_RE.fullmatch(scene_id) is None:
        raise ValueError(f"invalid ScanNet scene id: {scene_id!r}")
    if _PREFIX_RE.fullmatch(prefix_id) is None:
        raise ValueError(f"invalid prefix_id: {prefix_id!r}")
    if sample_idx != f"{scene_id}:{prefix_id}":
        raise ValueError("sample_idx must equal '<scene_id>:<prefix_id>'")
    if expected_scene_id is not None and scene_id != expected_scene_id:
        raise ValueError("R2a scene id mismatch")
    if expected_prefix_id is not None and prefix_id != expected_prefix_id:
        raise ValueError("R2a prefix id mismatch")

    prefix_fraction = float(
        _typed_scalar(values, "prefix_fraction", np.float64)
    )
    runtime_s = float(_typed_scalar(values, "runtime_s", np.float64))
    if not math.isfinite(prefix_fraction) or not 0.0 < prefix_fraction <= 1.0:
        raise ValueError("prefix_fraction must be finite and in (0,1]")
    if expected_prefix_fraction is not None and not math.isclose(
        prefix_fraction,
        float(expected_prefix_fraction),
        rel_tol=0.0,
        abs_tol=0.0,
    ):
        raise ValueError("R2a expected prefix fraction mismatch")
    if not math.isfinite(runtime_s) or runtime_s < 0.0:
        raise ValueError("runtime_s must be finite and nonnegative")

    hash_names = (
        "parent_cache_sha256",
        "parent_checkpoint_sha256",
        "parent_config_sha256",
        "parent_source_scene_sha256",
        "parent_axis_alignment_sha256",
        "prefix_manifest_row_sha256",
        "frame_artifact_tree_sha256",
        "r2_config_sha256",
        "r2_code_sha256",
    )
    hashes = {
        name: _sha256(_text(values, name), name) for name in hash_names
    }
    expected_hashes = {
        "prefix_manifest_row_sha256": _expected_sha256(
            expected_prefix_manifest_row_sha256,
            "prefix_manifest_row_sha256",
        ),
        "frame_artifact_tree_sha256": _expected_sha256(
            expected_frame_artifact_tree_sha256,
            "frame_artifact_tree_sha256",
        ),
        "r2_config_sha256": _expected_sha256(
            expected_r2_config_sha256, "r2_config_sha256"
        ),
        "r2_code_sha256": _expected_sha256(
            expected_r2_code_sha256, "r2_code_sha256"
        ),
    }
    for name, expected in expected_hashes.items():
        if hashes[name] != expected:
            raise ValueError(f"R2a {name} provenance mismatch")

    proposal_ids = _exact_array(
        values, "proposal_ids", dtype=np.int64, shape=(None,)
    )
    count = int(len(proposal_ids))
    lineage_ids = _exact_array(
        values, "lineage_ids", dtype=np.int64, shape=(count,)
    )
    if (
        np.any(proposal_ids < 0)
        or len(np.unique(proposal_ids)) != count
    ):
        raise ValueError("proposal_ids must be unique and nonnegative")
    if np.any(lineage_ids < 0) or len(np.unique(lineage_ids)) != count:
        raise ValueError("lineage_ids must be unique and nonnegative in R2a")

    topk_frame_ids = _exact_array(
        values, "topk_frame_ids", dtype=np.int64, shape=(count, None)
    )
    topk = int(topk_frame_ids.shape[1])
    if topk < 1:
        raise ValueError("R2a Top-K dimension must be positive")
    topk_view_valid = _exact_array(
        values, "topk_view_valid", dtype=np.bool_, shape=(count, topk)
    )
    topk_projected_area_pixels = _exact_array(
        values,
        "topk_projected_area_pixels",
        dtype=np.float32,
        shape=(count, topk),
    )
    topk_projected_area_fraction = _exact_array(
        values,
        "topk_projected_area_fraction",
        dtype=np.float32,
        shape=(count, topk),
    )
    per_view_depth_evidence = _exact_array(
        values,
        "per_view_depth_evidence",
        dtype=np.float32,
        shape=(count, topk, _DEPTH_EVIDENCE_DIM),
    )
    per_view_depth_counts = _exact_array(
        values,
        "per_view_depth_counts",
        dtype=np.int32,
        shape=(count, topk, _DEPTH_EVIDENCE_DIM),
    )
    per_view_point_count = _exact_array(
        values,
        "per_view_point_count",
        dtype=np.int32,
        shape=(count, topk),
    )
    aggregate_depth_evidence = _exact_array(
        values,
        "aggregate_depth_evidence",
        dtype=np.float32,
        shape=(count, _DEPTH_EVIDENCE_DIM),
    )
    aggregate_depth_counts = _exact_array(
        values,
        "aggregate_depth_counts",
        dtype=np.int64,
        shape=(count, _DEPTH_EVIDENCE_DIM),
    )
    aggregate_view_count = _exact_array(
        values,
        "aggregate_view_count",
        dtype=np.int32,
        shape=(count,),
    )
    aggregate_point_count = _exact_array(
        values,
        "aggregate_point_count",
        dtype=np.int64,
        shape=(count,),
    )

    if (
        not np.isfinite(topk_projected_area_pixels).all()
        or np.any(topk_projected_area_pixels < 0.0)
    ):
        raise ValueError(
            "Top-K projected pixel area must be finite and nonnegative"
        )
    if (
        not np.isfinite(topk_projected_area_fraction).all()
        or np.any(topk_projected_area_fraction < 0.0)
        or np.any(topk_projected_area_fraction > 1.0)
    ):
        raise ValueError(
            "Top-K projected area fraction must be finite in [0,1]"
        )
    if (
        not np.isfinite(per_view_depth_evidence).all()
        or np.any(per_view_depth_evidence < 0.0)
        or np.any(per_view_depth_evidence > 1.0)
    ):
        raise ValueError("per-view depth evidence must be finite in [0,1]")
    if (
        not np.isfinite(aggregate_depth_evidence).all()
        or np.any(aggregate_depth_evidence < 0.0)
        or np.any(aggregate_depth_evidence > 1.0)
    ):
        raise ValueError("aggregate depth evidence must be finite in [0,1]")
    if (
        np.any(per_view_depth_counts < 0)
        or np.any(per_view_point_count < 0)
        or np.any(aggregate_depth_counts < 0)
        or np.any(aggregate_point_count < 0)
    ):
        raise ValueError("depth pixel counts must be nonnegative")
    if np.any(aggregate_view_count < 0) or np.any(
        aggregate_view_count > topk
    ):
        raise ValueError("aggregate_view_count must lie in [0,K]")
    if not np.array_equal(
        aggregate_view_count.astype(np.int64),
        topk_view_valid.sum(axis=1, dtype=np.int64),
    ):
        raise ValueError("aggregate_view_count disagrees with valid views")

    if topk > 1 and np.any(
        (~topk_view_valid[:, :-1]) & topk_view_valid[:, 1:]
    ):
        raise ValueError("valid Top-K views must occupy leading slots")
    if np.any(topk_frame_ids[topk_view_valid] < 0):
        raise ValueError("valid Top-K frame ids must be nonnegative")
    if expected_allowed_frame_ids is not None:
        allowed_array = np.asarray(expected_allowed_frame_ids)
        if allowed_array.ndim != 1 or allowed_array.dtype.kind not in "iu":
            raise ValueError(
                "expected_allowed_frame_ids must be a one-dimensional "
                "integer sequence"
            )
        allowed_array = allowed_array.astype(np.int64, copy=False)
        if np.any(allowed_array < 0) or len(np.unique(allowed_array)) != len(
            allowed_array
        ):
            raise ValueError(
                "expected_allowed_frame_ids must be unique and nonnegative"
            )
        valid_frame_ids = topk_frame_ids[topk_view_valid]
        unexpected = np.setdiff1d(
            np.unique(valid_frame_ids), allowed_array, assume_unique=False
        )
        if len(unexpected):
            raise ValueError(
                "R2a Top-K frame ids are absent from the expected causal "
                f"frame set: {unexpected[:8].tolist()}"
            )
    if np.any(topk_frame_ids[~topk_view_valid] != -1):
        raise ValueError("invalid Top-K slots must use frame id -1")
    if np.any(topk_projected_area_pixels[~topk_view_valid] != 0.0) or np.any(
        topk_projected_area_fraction[~topk_view_valid] != 0.0
    ):
        raise ValueError("invalid Top-K slots must have zero projected area")
    if np.any(topk_projected_area_pixels[topk_view_valid] <= 0.0) or np.any(
        topk_projected_area_fraction[topk_view_valid] <= 0.0
    ):
        raise ValueError("valid Top-K slots must have positive projected area")
    if np.any(per_view_depth_counts[~topk_view_valid] != 0):
        raise ValueError("invalid Top-K slots must have zero depth counts")
    if np.any(per_view_point_count[~topk_view_valid] != 0):
        raise ValueError("invalid Top-K slots must have zero point count")
    if np.any(per_view_depth_evidence[~topk_view_valid] != 0.0):
        raise ValueError("invalid Top-K slots must have zero evidence")
    for row in range(count):
        selected = topk_frame_ids[row, topk_view_valid[row]]
        if len(np.unique(selected)) != len(selected):
            raise ValueError("Top-K frame ids must be unique per proposal")
    per_view_totals = per_view_depth_counts.sum(axis=2, dtype=np.int64)
    if np.any(per_view_totals > np.iinfo(np.int32).max):
        raise ValueError("per-view depth pixel total exceeds int32")
    if not np.array_equal(
        per_view_point_count, per_view_totals.astype(np.int32)
    ):
        raise ValueError(
            "per_view_point_count disagrees with per-view depth counts"
        )
    if np.any(topk_view_valid & (per_view_point_count == 0)):
        raise ValueError("valid Top-K views must contain sampled depth pixels")

    expected_per_view_evidence = depth_evidence_fractions(
        per_view_depth_counts
    )
    if not np.array_equal(
        per_view_depth_evidence, expected_per_view_evidence
    ):
        raise ValueError(
            "per-view depth evidence disagrees with depth counts"
        )

    expected_aggregate_counts = np.where(
        topk_view_valid[..., None],
        per_view_depth_counts.astype(np.int64),
        0,
    ).sum(axis=1, dtype=np.int64)
    if not np.array_equal(
        aggregate_depth_counts, expected_aggregate_counts
    ):
        raise ValueError(
            "aggregate depth counts disagree with per-view depth counts"
        )
    expected_aggregate_points = expected_aggregate_counts.sum(
        axis=1, dtype=np.int64
    )
    if not np.array_equal(
        aggregate_point_count, expected_aggregate_points
    ):
        raise ValueError(
            "aggregate_point_count disagrees with aggregate depth counts"
        )
    expected_aggregate_evidence = depth_evidence_fractions(
        expected_aggregate_counts
    )
    if not np.array_equal(
        aggregate_depth_evidence, expected_aggregate_evidence
    ):
        raise ValueError(
            "aggregate depth evidence disagrees with aggregate counts"
        )

    cache = TR3DR2Cache(
        scene_id=scene_id,
        prefix_id=prefix_id,
        prefix_fraction=prefix_fraction,
        parent_cache_sha256=hashes["parent_cache_sha256"],
        parent_checkpoint_sha256=hashes["parent_checkpoint_sha256"],
        parent_config_sha256=hashes["parent_config_sha256"],
        parent_source_scene_sha256=hashes[
            "parent_source_scene_sha256"
        ],
        parent_axis_alignment_sha256=hashes[
            "parent_axis_alignment_sha256"
        ],
        prefix_manifest_row_sha256=hashes[
            "prefix_manifest_row_sha256"
        ],
        frame_artifact_tree_sha256=hashes[
            "frame_artifact_tree_sha256"
        ],
        r2_config_sha256=hashes["r2_config_sha256"],
        r2_code_sha256=hashes["r2_code_sha256"],
        proposal_ids=_readonly(proposal_ids),
        lineage_ids=_readonly(lineage_ids),
        topk_frame_ids=_readonly(topk_frame_ids),
        topk_view_valid=_readonly(topk_view_valid),
        topk_projected_area_pixels=_readonly(
            topk_projected_area_pixels
        ),
        topk_projected_area_fraction=_readonly(
            topk_projected_area_fraction
        ),
        per_view_depth_evidence=_readonly(per_view_depth_evidence),
        per_view_depth_counts=_readonly(per_view_depth_counts),
        per_view_point_count=_readonly(per_view_point_count),
        aggregate_depth_evidence=_readonly(aggregate_depth_evidence),
        aggregate_depth_counts=_readonly(aggregate_depth_counts),
        aggregate_view_count=_readonly(aggregate_view_count),
        aggregate_point_count=_readonly(aggregate_point_count),
        runtime_s=runtime_s,
    )
    parent, parent_file_sha256 = _verified_parent(parent_cache_path)
    _validate_parent_binding(cache, parent, parent_file_sha256)
    return cache


def make_tr3d_r2_cache(
    *,
    parent_cache_path: str | os.PathLike[str],
    prefix_manifest_row_sha256: str,
    frame_artifact_tree_sha256: str,
    r2_config_sha256: str,
    r2_code_sha256: str,
    proposal_ids: np.ndarray,
    lineage_ids: np.ndarray,
    topk_frame_ids: np.ndarray,
    topk_view_valid: np.ndarray,
    topk_projected_area_pixels: np.ndarray,
    topk_projected_area_fraction: np.ndarray,
    per_view_depth_evidence: np.ndarray,
    per_view_depth_counts: np.ndarray,
    per_view_point_count: np.ndarray,
    aggregate_depth_evidence: np.ndarray,
    aggregate_depth_counts: np.ndarray,
    aggregate_view_count: np.ndarray,
    aggregate_point_count: np.ndarray,
    runtime_s: float = 0.0,
    expected_allowed_frame_ids: Sequence[int] | np.ndarray | None = None,
) -> TR3DR2Cache:
    """Build and fully validate an R2a sidecar from an immutable parent."""

    parent, parent_file_sha256 = _verified_parent(parent_cache_path)
    cache = TR3DR2Cache(
        scene_id=parent.scene_id,
        prefix_id=parent.prefix_id,
        prefix_fraction=parent.prefix_fraction,
        parent_cache_sha256=parent_file_sha256,
        parent_checkpoint_sha256=parent.checkpoint_sha256,
        parent_config_sha256=parent.config_sha256,
        parent_source_scene_sha256=parent.source_scene_sha256,
        parent_axis_alignment_sha256=parent.axis_alignment_sha256,
        prefix_manifest_row_sha256=_sha256(
            prefix_manifest_row_sha256, "prefix_manifest_row_sha256"
        ),
        frame_artifact_tree_sha256=_sha256(
            frame_artifact_tree_sha256, "frame_artifact_tree_sha256"
        ),
        r2_config_sha256=_sha256(r2_config_sha256, "r2_config_sha256"),
        r2_code_sha256=_sha256(r2_code_sha256, "r2_code_sha256"),
        proposal_ids=np.asarray(proposal_ids, dtype=np.int64),
        lineage_ids=np.asarray(lineage_ids, dtype=np.int64),
        topk_frame_ids=np.asarray(topk_frame_ids, dtype=np.int64),
        topk_view_valid=np.asarray(topk_view_valid, dtype=np.bool_),
        topk_projected_area_pixels=np.asarray(
            topk_projected_area_pixels, dtype=np.float32
        ),
        topk_projected_area_fraction=np.asarray(
            topk_projected_area_fraction, dtype=np.float32
        ),
        per_view_depth_evidence=np.asarray(
            per_view_depth_evidence, dtype=np.float32
        ),
        per_view_depth_counts=np.asarray(
            per_view_depth_counts, dtype=np.int32
        ),
        per_view_point_count=np.asarray(
            per_view_point_count, dtype=np.int32
        ),
        aggregate_depth_evidence=np.asarray(
            aggregate_depth_evidence, dtype=np.float32
        ),
        aggregate_depth_counts=np.asarray(
            aggregate_depth_counts, dtype=np.int64
        ),
        aggregate_view_count=np.asarray(
            aggregate_view_count, dtype=np.int32
        ),
        aggregate_point_count=np.asarray(
            aggregate_point_count, dtype=np.int64
        ),
        runtime_s=float(runtime_s),
    )
    return validate_tr3d_r2_payload(
        cache.as_npz_payload(),
        parent_cache_path=parent_cache_path,
        expected_prefix_manifest_row_sha256=prefix_manifest_row_sha256,
        expected_frame_artifact_tree_sha256=frame_artifact_tree_sha256,
        expected_r2_config_sha256=r2_config_sha256,
        expected_r2_code_sha256=r2_code_sha256,
        expected_allowed_frame_ids=expected_allowed_frame_ids,
    )


def load_tr3d_r2_cache(
    path: str | os.PathLike[str],
    *,
    parent_cache_path: str | os.PathLike[str],
    expected_prefix_manifest_row_sha256: str,
    expected_frame_artifact_tree_sha256: str,
    expected_r2_config_sha256: str,
    expected_r2_code_sha256: str,
    expected_scene_id: str | None = None,
    expected_prefix_id: str | None = None,
    expected_prefix_fraction: float | None = None,
    expected_allowed_frame_ids: Sequence[int] | np.ndarray | None = None,
) -> TR3DR2Cache:
    cache_path = Path(path)
    if not cache_path.is_file():
        raise FileNotFoundError(cache_path)
    with np.load(cache_path, allow_pickle=False) as archive:
        values = {
            name: np.array(archive[name], copy=True) for name in archive.files
        }
    return validate_tr3d_r2_payload(
        values,
        parent_cache_path=parent_cache_path,
        expected_prefix_manifest_row_sha256=(
            expected_prefix_manifest_row_sha256
        ),
        expected_frame_artifact_tree_sha256=(
            expected_frame_artifact_tree_sha256
        ),
        expected_r2_config_sha256=expected_r2_config_sha256,
        expected_r2_code_sha256=expected_r2_code_sha256,
        expected_scene_id=expected_scene_id,
        expected_prefix_id=expected_prefix_id,
        expected_prefix_fraction=expected_prefix_fraction,
        expected_allowed_frame_ids=expected_allowed_frame_ids,
    )


def write_tr3d_r2_cache(
    path: str | os.PathLike[str],
    cache: TR3DR2Cache,
    *,
    parent_cache_path: str | os.PathLike[str],
    expected_allowed_frame_ids: Sequence[int] | np.ndarray | None = None,
) -> Path:
    """Atomically create one immutable sidecar; never overwrite a path."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    canonical = validate_tr3d_r2_payload(
        cache.as_npz_payload(),
        parent_cache_path=parent_cache_path,
        expected_prefix_manifest_row_sha256=(
            cache.prefix_manifest_row_sha256
        ),
        expected_frame_artifact_tree_sha256=(
            cache.frame_artifact_tree_sha256
        ),
        expected_r2_config_sha256=cache.r2_config_sha256,
        expected_r2_code_sha256=cache.r2_code_sha256,
        expected_allowed_frame_ids=expected_allowed_frame_ids,
    )
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
        os.link(temporary_name, target)
        target.chmod(0o444)
    except FileExistsError as error:
        raise FileExistsError(
            f"immutable R2a cache exists: {target}"
        ) from error
    finally:
        if temporary_name is not None:
            try:
                Path(temporary_name).unlink()
            except FileNotFoundError:
                pass
    return target
