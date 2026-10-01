"""Immutable R2b multi-view feature-evidence sidecars.

R2b is an observer-only child of one exact R2a NPZ.  It stores no proposal
geometry and cannot change detections.  Every load verifies the complete R2a
provenance chain, the exact parent file bytes, the proposal/lineage/Top-K row
identity, and the feature extractor checkpoint, configuration, and code
hashes.

Feature vectors may be float16 or float32.  Missing feature slots use a strict
zero sentinel.  Aggregate vectors and pairwise cosine statistics are
redundant by design: the loader derives them again from the stored per-view
vectors and counts and rejects any disagreement.
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
from typing import Mapping

import numpy as np

from .tr3d_r2_cache import TR3DR2Cache, load_tr3d_r2_cache


TR3D_R2B_CACHE_SCHEMA = "boxfusion.tr3d_r2b_feature_evidence.v1"
PAIRWISE_COSINE_STATISTIC_NAMES = (
    "mean",
    "median",
    "min",
    "max",
    "std",
)

_SCENE_RE = re.compile(r"^scene[0-9]{4}_[0-9]{2}$")
_PREFIX_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_FEATURE_DTYPES = (np.dtype(np.float16), np.dtype(np.float32))

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
        "parent_r2a_cache_sha256",
        "parent_prefix_manifest_row_sha256",
        "parent_frame_artifact_tree_sha256",
        "parent_r2_config_sha256",
        "parent_r2_code_sha256",
        "feature_checkpoint_sha256",
        "feature_config_sha256",
        "feature_code_sha256",
        "proposal_ids",
        "lineage_ids",
        "topk_frame_ids",
        "topk_view_valid",
        "per_view_feature_valid",
        "per_view_feature_count",
        "per_view_feature_vector",
        "aggregate_feature_vector",
        "aggregate_view_count",
        "aggregate_feature_count",
        "pairwise_cosine_count",
        "pairwise_cosine_mean",
        "pairwise_cosine_median",
        "pairwise_cosine_min",
        "pairwise_cosine_max",
        "pairwise_cosine_std",
        "runtime_s",
    }
)


@dataclass(frozen=True)
class TR3DR2BFeatureCache:
    """Canonical feature-consistency evidence for every R2a proposal row."""

    scene_id: str
    prefix_id: str
    prefix_fraction: float
    parent_r2a_cache_sha256: str
    parent_prefix_manifest_row_sha256: str
    parent_frame_artifact_tree_sha256: str
    parent_r2_config_sha256: str
    parent_r2_code_sha256: str
    feature_checkpoint_sha256: str
    feature_config_sha256: str
    feature_code_sha256: str
    proposal_ids: np.ndarray
    lineage_ids: np.ndarray
    topk_frame_ids: np.ndarray
    topk_view_valid: np.ndarray
    per_view_feature_valid: np.ndarray
    per_view_feature_count: np.ndarray
    per_view_feature_vector: np.ndarray
    aggregate_feature_vector: np.ndarray
    aggregate_view_count: np.ndarray
    aggregate_feature_count: np.ndarray
    pairwise_cosine_count: np.ndarray
    pairwise_cosine_mean: np.ndarray
    pairwise_cosine_median: np.ndarray
    pairwise_cosine_min: np.ndarray
    pairwise_cosine_max: np.ndarray
    pairwise_cosine_std: np.ndarray
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

    @property
    def feature_dim(self) -> int:
        value = np.asarray(self.per_view_feature_vector)
        return int(value.shape[2]) if value.ndim == 3 else 0

    def as_npz_payload(self) -> dict[str, np.ndarray]:
        feature_vector = np.asarray(self.per_view_feature_vector)
        if feature_vector.dtype not in _FEATURE_DTYPES:
            # Validation provides the user-facing error; avoid silently
            # changing numerical evidence while serialising.
            feature_vector = np.asarray(feature_vector)
        return {
            "schema": np.asarray(TR3D_R2B_CACHE_SCHEMA),
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
            "parent_r2a_cache_sha256": np.asarray(
                self.parent_r2a_cache_sha256
            ),
            "parent_prefix_manifest_row_sha256": np.asarray(
                self.parent_prefix_manifest_row_sha256
            ),
            "parent_frame_artifact_tree_sha256": np.asarray(
                self.parent_frame_artifact_tree_sha256
            ),
            "parent_r2_config_sha256": np.asarray(
                self.parent_r2_config_sha256
            ),
            "parent_r2_code_sha256": np.asarray(
                self.parent_r2_code_sha256
            ),
            "feature_checkpoint_sha256": np.asarray(
                self.feature_checkpoint_sha256
            ),
            "feature_config_sha256": np.asarray(
                self.feature_config_sha256
            ),
            "feature_code_sha256": np.asarray(self.feature_code_sha256),
            "proposal_ids": np.asarray(self.proposal_ids, dtype=np.int64),
            "lineage_ids": np.asarray(self.lineage_ids, dtype=np.int64),
            "topk_frame_ids": np.asarray(
                self.topk_frame_ids, dtype=np.int64
            ),
            "topk_view_valid": np.asarray(
                self.topk_view_valid, dtype=np.bool_
            ),
            "per_view_feature_valid": np.asarray(
                self.per_view_feature_valid, dtype=np.bool_
            ),
            "per_view_feature_count": np.asarray(
                self.per_view_feature_count, dtype=np.int32
            ),
            "per_view_feature_vector": feature_vector,
            "aggregate_feature_vector": np.asarray(
                self.aggregate_feature_vector, dtype=np.float32
            ),
            "aggregate_view_count": np.asarray(
                self.aggregate_view_count, dtype=np.int32
            ),
            "aggregate_feature_count": np.asarray(
                self.aggregate_feature_count, dtype=np.int64
            ),
            "pairwise_cosine_count": np.asarray(
                self.pairwise_cosine_count, dtype=np.int32
            ),
            "pairwise_cosine_mean": np.asarray(
                self.pairwise_cosine_mean, dtype=np.float32
            ),
            "pairwise_cosine_median": np.asarray(
                self.pairwise_cosine_median, dtype=np.float32
            ),
            "pairwise_cosine_min": np.asarray(
                self.pairwise_cosine_min, dtype=np.float32
            ),
            "pairwise_cosine_max": np.asarray(
                self.pairwise_cosine_max, dtype=np.float32
            ),
            "pairwise_cosine_std": np.asarray(
                self.pairwise_cosine_std, dtype=np.float32
            ),
            "runtime_s": np.asarray(self.runtime_s, dtype=np.float64),
        }


def sha256_file(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tr3d_r2b_cache_path(
    root: str | os.PathLike[str],
    scene_id: str,
    prefix_id: str = "full",
) -> Path:
    if _SCENE_RE.fullmatch(scene_id) is None:
        raise ValueError(f"invalid ScanNet scene id: {scene_id!r}")
    if _PREFIX_RE.fullmatch(prefix_id) is None:
        raise ValueError(f"invalid prefix_id: {prefix_id!r}")
    return Path(root) / scene_id / f"{prefix_id}.r2b.npz"


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


def _feature_array(
    values: Mapping[str, np.ndarray],
    name: str,
    *,
    shape: tuple[int | None, ...],
) -> np.ndarray:
    value = np.asarray(values[name])
    if value.dtype not in _FEATURE_DTYPES or value.ndim != len(shape):
        raise ValueError(
            f"{name} must have dtype float16 or float32 and shape {shape}"
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


def derive_feature_aggregates(
    per_view_feature_vector: np.ndarray,
    per_view_feature_valid: np.ndarray,
    per_view_feature_count: np.ndarray,
) -> dict[str, np.ndarray]:
    """Derive canonical aggregates and pairwise cosine statistics.

    Each valid view vector is L2-normalised in float64.  The aggregate is the
    feature-count-weighted mean of unit view vectors, L2-normalised and stored
    as float32.  Pairwise statistics use each unordered pair once.  Rows with
    fewer than two views use a deterministic zero statistic sentinel.
    """

    vectors = np.asarray(per_view_feature_vector)
    valid = np.asarray(per_view_feature_valid)
    counts = np.asarray(per_view_feature_count)
    if vectors.dtype not in _FEATURE_DTYPES or vectors.ndim != 3:
        raise ValueError("feature vectors must be float16/float32 [P,K,D]")
    proposal_count, topk, feature_dim = vectors.shape
    if feature_dim < 1:
        raise ValueError("feature dimension must be positive")
    if valid.dtype != np.bool_ or valid.shape != (proposal_count, topk):
        raise ValueError("feature validity must be bool [P,K]")
    if counts.dtype != np.int32 or counts.shape != (proposal_count, topk):
        raise ValueError("feature counts must be int32 [P,K]")
    if np.any(counts < 0):
        raise ValueError("feature counts must be nonnegative")
    if not np.array_equal(valid, counts > 0):
        raise ValueError("feature validity must equal feature_count > 0")
    if not np.isfinite(vectors).all():
        raise ValueError("feature vectors must be finite")
    if np.any(vectors[~valid] != 0):
        raise ValueError("invalid feature slots must use zero vectors")

    aggregate = np.zeros((proposal_count, feature_dim), dtype=np.float32)
    view_count = valid.sum(axis=1, dtype=np.int64).astype(np.int32)
    feature_count = counts.astype(np.int64).sum(axis=1, dtype=np.int64)
    pair_count = np.zeros(proposal_count, dtype=np.int32)
    statistics = {
        name: np.zeros(proposal_count, dtype=np.float32)
        for name in PAIRWISE_COSINE_STATISTIC_NAMES
    }

    for row in range(proposal_count):
        selected = vectors[row, valid[row]].astype(np.float64, copy=False)
        if selected.shape[0] == 0:
            continue
        norms = np.linalg.norm(selected, axis=1)
        if not np.isfinite(norms).all() or np.any(norms <= 0.0):
            raise ValueError("valid feature vectors must have positive norm")
        unit = selected / norms[:, None]
        weights = counts[row, valid[row]].astype(np.float64, copy=False)
        pooled = (unit * weights[:, None]).sum(axis=0)
        pooled_norm = float(np.linalg.norm(pooled))
        if not math.isfinite(pooled_norm) or pooled_norm <= 0.0:
            raise ValueError("valid feature vectors have zero aggregate norm")
        aggregate[row] = (pooled / pooled_norm).astype(np.float32)

        nviews = int(selected.shape[0])
        npairs = nviews * (nviews - 1) // 2
        pair_count[row] = npairs
        if npairs == 0:
            continue
        matrix = unit @ unit.T
        pair_values = matrix[np.triu_indices(nviews, k=1)]
        # Numerical products can differ from the mathematical cosine by a
        # few ulps.  Clipping defines a stable physical range and is part of
        # the canonical statistic algorithm.
        pair_values = np.clip(pair_values, -1.0, 1.0)
        statistics["mean"][row] = np.float32(np.mean(pair_values))
        statistics["median"][row] = np.float32(np.median(pair_values))
        statistics["min"][row] = np.float32(np.min(pair_values))
        statistics["max"][row] = np.float32(np.max(pair_values))
        statistics["std"][row] = np.float32(np.std(pair_values))

    result = {
        "aggregate_feature_vector": _readonly(aggregate),
        "aggregate_view_count": _readonly(view_count),
        "aggregate_feature_count": _readonly(feature_count),
        "pairwise_cosine_count": _readonly(pair_count),
    }
    result.update(
        {
            f"pairwise_cosine_{name}": _readonly(value)
            for name, value in statistics.items()
        }
    )
    return result


def _verified_r2a_parent(
    r2a_path: str | os.PathLike[str],
    *,
    parent_tr3d_cache_path: str | os.PathLike[str],
    expected_prefix_manifest_row_sha256: str,
    expected_frame_artifact_tree_sha256: str,
    expected_r2_config_sha256: str,
    expected_r2_code_sha256: str,
) -> tuple[TR3DR2Cache, str]:
    path = Path(r2a_path)
    if not path.is_file():
        raise FileNotFoundError(path)
    before = sha256_file(path)
    parent = load_tr3d_r2_cache(
        path,
        parent_cache_path=parent_tr3d_cache_path,
        expected_prefix_manifest_row_sha256=(
            expected_prefix_manifest_row_sha256
        ),
        expected_frame_artifact_tree_sha256=(
            expected_frame_artifact_tree_sha256
        ),
        expected_r2_config_sha256=expected_r2_config_sha256,
        expected_r2_code_sha256=expected_r2_code_sha256,
    )
    after = sha256_file(path)
    if before != after:
        raise RuntimeError("parent R2a cache changed while being verified")
    return parent, before


def _validate_parent_binding(
    cache: TR3DR2BFeatureCache,
    parent: TR3DR2Cache,
    parent_file_sha256: str,
) -> None:
    scalar_comparisons = {
        "cache": (cache.parent_r2a_cache_sha256, parent_file_sha256),
        "scene": (cache.scene_id, parent.scene_id),
        "prefix": (cache.prefix_id, parent.prefix_id),
        "prefix manifest row": (
            cache.parent_prefix_manifest_row_sha256,
            parent.prefix_manifest_row_sha256,
        ),
        "frame artifact tree": (
            cache.parent_frame_artifact_tree_sha256,
            parent.frame_artifact_tree_sha256,
        ),
        "R2 config": (
            cache.parent_r2_config_sha256,
            parent.r2_config_sha256,
        ),
        "R2 code": (cache.parent_r2_code_sha256, parent.r2_code_sha256),
    }
    for label, (observed, expected) in scalar_comparisons.items():
        if observed != expected:
            raise ValueError(f"R2b parent R2a {label} provenance mismatch")
    if not math.isclose(
        cache.prefix_fraction,
        parent.prefix_fraction,
        rel_tol=0.0,
        abs_tol=0.0,
    ):
        raise ValueError("R2b parent R2a prefix fraction mismatch")
    array_comparisons = {
        "proposal_ids": (cache.proposal_ids, parent.proposal_ids),
        "lineage_ids": (cache.lineage_ids, parent.lineage_ids),
        "topk_frame_ids": (cache.topk_frame_ids, parent.topk_frame_ids),
        "topk_view_valid": (
            cache.topk_view_valid,
            parent.topk_view_valid,
        ),
    }
    for label, (observed, expected) in array_comparisons.items():
        if not np.array_equal(observed, expected):
            raise ValueError(f"R2b {label} disagrees with exact R2a parent")


def validate_tr3d_r2b_payload(
    values: Mapping[str, np.ndarray],
    *,
    parent_r2a_cache_path: str | os.PathLike[str],
    parent_tr3d_cache_path: str | os.PathLike[str],
    expected_parent_prefix_manifest_row_sha256: str,
    expected_parent_frame_artifact_tree_sha256: str,
    expected_parent_r2_config_sha256: str,
    expected_parent_r2_code_sha256: str,
    expected_feature_checkpoint_sha256: str,
    expected_feature_config_sha256: str,
    expected_feature_code_sha256: str,
    expected_scene_id: str | None = None,
    expected_prefix_id: str | None = None,
    expected_prefix_fraction: float | None = None,
) -> TR3DR2BFeatureCache:
    """Decode and fail-closed validate an R2b feature sidecar."""

    fields = frozenset(values)
    if fields != _FIELDS:
        raise ValueError(
            "R2b cache fields disagree; "
            f"missing={sorted(_FIELDS-fields)}, "
            f"unknown={sorted(fields-_FIELDS)}"
        )
    if any(np.asarray(value).dtype.hasobject for value in values.values()):
        raise ValueError("R2b cache must not contain object arrays")
    if _text(values, "schema") != TR3D_R2B_CACHE_SCHEMA:
        raise ValueError("unsupported R2b cache schema")
    if not _boolean(values, "complete"):
        raise ValueError("R2b cache is incomplete")
    if not _boolean(values, "observer_only"):
        raise ValueError("R2b cache is not observer-only")
    if _boolean(values, "mutation_enabled"):
        raise ValueError("R2b cache enables mutation")
    if int(_typed_scalar(values, "applied_count", np.int64)) != 0:
        raise ValueError("R2b observer applied_count must be zero")

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
        raise ValueError("R2b scene id mismatch")
    if expected_prefix_id is not None and prefix_id != expected_prefix_id:
        raise ValueError("R2b prefix id mismatch")

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
        raise ValueError("R2b expected prefix fraction mismatch")
    if not math.isfinite(runtime_s) or runtime_s < 0.0:
        raise ValueError("runtime_s must be finite and nonnegative")

    hash_names = (
        "parent_r2a_cache_sha256",
        "parent_prefix_manifest_row_sha256",
        "parent_frame_artifact_tree_sha256",
        "parent_r2_config_sha256",
        "parent_r2_code_sha256",
        "feature_checkpoint_sha256",
        "feature_config_sha256",
        "feature_code_sha256",
    )
    hashes = {
        name: _sha256(_text(values, name), name) for name in hash_names
    }
    expected_hashes = {
        "parent_prefix_manifest_row_sha256": _expected_sha256(
            expected_parent_prefix_manifest_row_sha256,
            "parent_prefix_manifest_row_sha256",
        ),
        "parent_frame_artifact_tree_sha256": _expected_sha256(
            expected_parent_frame_artifact_tree_sha256,
            "parent_frame_artifact_tree_sha256",
        ),
        "parent_r2_config_sha256": _expected_sha256(
            expected_parent_r2_config_sha256,
            "parent_r2_config_sha256",
        ),
        "parent_r2_code_sha256": _expected_sha256(
            expected_parent_r2_code_sha256,
            "parent_r2_code_sha256",
        ),
        "feature_checkpoint_sha256": _expected_sha256(
            expected_feature_checkpoint_sha256,
            "feature_checkpoint_sha256",
        ),
        "feature_config_sha256": _expected_sha256(
            expected_feature_config_sha256,
            "feature_config_sha256",
        ),
        "feature_code_sha256": _expected_sha256(
            expected_feature_code_sha256,
            "feature_code_sha256",
        ),
    }
    for name, expected in expected_hashes.items():
        if hashes[name] != expected:
            raise ValueError(f"R2b {name} provenance mismatch")

    proposal_ids = _exact_array(
        values, "proposal_ids", dtype=np.int64, shape=(None,)
    )
    proposal_count = int(len(proposal_ids))
    lineage_ids = _exact_array(
        values, "lineage_ids", dtype=np.int64, shape=(proposal_count,)
    )
    topk_frame_ids = _exact_array(
        values,
        "topk_frame_ids",
        dtype=np.int64,
        shape=(proposal_count, None),
    )
    topk = int(topk_frame_ids.shape[1])
    if topk < 1:
        raise ValueError("R2b Top-K dimension must be positive")
    topk_view_valid = _exact_array(
        values,
        "topk_view_valid",
        dtype=np.bool_,
        shape=(proposal_count, topk),
    )
    feature_valid = _exact_array(
        values,
        "per_view_feature_valid",
        dtype=np.bool_,
        shape=(proposal_count, topk),
    )
    feature_count = _exact_array(
        values,
        "per_view_feature_count",
        dtype=np.int32,
        shape=(proposal_count, topk),
    )
    feature_vector = _feature_array(
        values,
        "per_view_feature_vector",
        shape=(proposal_count, topk, None),
    )
    feature_dim = int(feature_vector.shape[2])
    if feature_dim < 1:
        raise ValueError("R2b feature dimension must be positive")
    aggregate_feature_vector = _exact_array(
        values,
        "aggregate_feature_vector",
        dtype=np.float32,
        shape=(proposal_count, feature_dim),
    )
    aggregate_view_count = _exact_array(
        values,
        "aggregate_view_count",
        dtype=np.int32,
        shape=(proposal_count,),
    )
    aggregate_feature_count = _exact_array(
        values,
        "aggregate_feature_count",
        dtype=np.int64,
        shape=(proposal_count,),
    )
    pairwise_cosine_count = _exact_array(
        values,
        "pairwise_cosine_count",
        dtype=np.int32,
        shape=(proposal_count,),
    )
    statistics = {
        name: _exact_array(
            values,
            f"pairwise_cosine_{name}",
            dtype=np.float32,
            shape=(proposal_count,),
        )
        for name in PAIRWISE_COSINE_STATISTIC_NAMES
    }

    if not np.array_equal(feature_valid, feature_count > 0):
        raise ValueError("feature validity must equal feature_count > 0")
    if np.any(feature_count < 0):
        raise ValueError("feature counts must be nonnegative")
    if np.any(feature_valid & ~topk_view_valid):
        raise ValueError("feature-valid slots must be valid R2a Top-K views")
    if np.any(feature_count[~feature_valid] != 0):
        raise ValueError("invalid feature slots must use zero counts")
    if not np.isfinite(feature_vector).all():
        raise ValueError("feature vectors must be finite")
    if np.any(feature_vector[~feature_valid] != 0):
        raise ValueError("invalid feature slots must use zero vectors")
    if not np.isfinite(aggregate_feature_vector).all():
        raise ValueError("aggregate feature vectors must be finite")
    for name, value in statistics.items():
        if not np.isfinite(value).all():
            raise ValueError(f"pairwise cosine {name} must be finite")
        if name != "std" and (
            np.any(value < -1.0) or np.any(value > 1.0)
        ):
            raise ValueError(f"pairwise cosine {name} must lie in [-1,1]")
        if name == "std" and (
            np.any(value < 0.0) or np.any(value > 1.0)
        ):
            raise ValueError("pairwise cosine std must lie in [0,1]")

    derived = derive_feature_aggregates(
        feature_vector, feature_valid, feature_count
    )
    stored_derived = {
        "aggregate_feature_vector": aggregate_feature_vector,
        "aggregate_view_count": aggregate_view_count,
        "aggregate_feature_count": aggregate_feature_count,
        "pairwise_cosine_count": pairwise_cosine_count,
        **{
            f"pairwise_cosine_{name}": value
            for name, value in statistics.items()
        },
    }
    for name, expected in derived.items():
        if not np.array_equal(stored_derived[name], expected):
            raise ValueError(f"{name} disagrees with per-view features")

    cache = TR3DR2BFeatureCache(
        scene_id=scene_id,
        prefix_id=prefix_id,
        prefix_fraction=prefix_fraction,
        parent_r2a_cache_sha256=hashes["parent_r2a_cache_sha256"],
        parent_prefix_manifest_row_sha256=hashes[
            "parent_prefix_manifest_row_sha256"
        ],
        parent_frame_artifact_tree_sha256=hashes[
            "parent_frame_artifact_tree_sha256"
        ],
        parent_r2_config_sha256=hashes["parent_r2_config_sha256"],
        parent_r2_code_sha256=hashes["parent_r2_code_sha256"],
        feature_checkpoint_sha256=hashes["feature_checkpoint_sha256"],
        feature_config_sha256=hashes["feature_config_sha256"],
        feature_code_sha256=hashes["feature_code_sha256"],
        proposal_ids=_readonly(proposal_ids),
        lineage_ids=_readonly(lineage_ids),
        topk_frame_ids=_readonly(topk_frame_ids),
        topk_view_valid=_readonly(topk_view_valid),
        per_view_feature_valid=_readonly(feature_valid),
        per_view_feature_count=_readonly(feature_count),
        per_view_feature_vector=_readonly(feature_vector),
        aggregate_feature_vector=_readonly(aggregate_feature_vector),
        aggregate_view_count=_readonly(aggregate_view_count),
        aggregate_feature_count=_readonly(aggregate_feature_count),
        pairwise_cosine_count=_readonly(pairwise_cosine_count),
        pairwise_cosine_mean=_readonly(statistics["mean"]),
        pairwise_cosine_median=_readonly(statistics["median"]),
        pairwise_cosine_min=_readonly(statistics["min"]),
        pairwise_cosine_max=_readonly(statistics["max"]),
        pairwise_cosine_std=_readonly(statistics["std"]),
        runtime_s=runtime_s,
    )
    parent, parent_sha256 = _verified_r2a_parent(
        parent_r2a_cache_path,
        parent_tr3d_cache_path=parent_tr3d_cache_path,
        expected_prefix_manifest_row_sha256=(
            expected_parent_prefix_manifest_row_sha256
        ),
        expected_frame_artifact_tree_sha256=(
            expected_parent_frame_artifact_tree_sha256
        ),
        expected_r2_config_sha256=expected_parent_r2_config_sha256,
        expected_r2_code_sha256=expected_parent_r2_code_sha256,
    )
    _validate_parent_binding(cache, parent, parent_sha256)
    return cache


def make_tr3d_r2b_cache(
    *,
    parent_r2a_cache_path: str | os.PathLike[str],
    parent_tr3d_cache_path: str | os.PathLike[str],
    parent_prefix_manifest_row_sha256: str,
    parent_frame_artifact_tree_sha256: str,
    parent_r2_config_sha256: str,
    parent_r2_code_sha256: str,
    feature_checkpoint_sha256: str,
    feature_config_sha256: str,
    feature_code_sha256: str,
    per_view_feature_valid: np.ndarray,
    per_view_feature_count: np.ndarray,
    per_view_feature_vector: np.ndarray,
    runtime_s: float = 0.0,
) -> TR3DR2BFeatureCache:
    """Build a fully derived and validated R2b sidecar."""

    parent, parent_sha256 = _verified_r2a_parent(
        parent_r2a_cache_path,
        parent_tr3d_cache_path=parent_tr3d_cache_path,
        expected_prefix_manifest_row_sha256=(
            parent_prefix_manifest_row_sha256
        ),
        expected_frame_artifact_tree_sha256=(
            parent_frame_artifact_tree_sha256
        ),
        expected_r2_config_sha256=parent_r2_config_sha256,
        expected_r2_code_sha256=parent_r2_code_sha256,
    )
    vectors = np.asarray(per_view_feature_vector)
    if vectors.dtype not in _FEATURE_DTYPES:
        raise ValueError("per_view_feature_vector must be float16 or float32")
    valid = np.asarray(per_view_feature_valid, dtype=np.bool_)
    counts = np.asarray(per_view_feature_count, dtype=np.int32)
    derived = derive_feature_aggregates(vectors, valid, counts)
    cache = TR3DR2BFeatureCache(
        scene_id=parent.scene_id,
        prefix_id=parent.prefix_id,
        prefix_fraction=parent.prefix_fraction,
        parent_r2a_cache_sha256=parent_sha256,
        parent_prefix_manifest_row_sha256=_sha256(
            parent_prefix_manifest_row_sha256,
            "parent_prefix_manifest_row_sha256",
        ),
        parent_frame_artifact_tree_sha256=_sha256(
            parent_frame_artifact_tree_sha256,
            "parent_frame_artifact_tree_sha256",
        ),
        parent_r2_config_sha256=_sha256(
            parent_r2_config_sha256, "parent_r2_config_sha256"
        ),
        parent_r2_code_sha256=_sha256(
            parent_r2_code_sha256, "parent_r2_code_sha256"
        ),
        feature_checkpoint_sha256=_sha256(
            feature_checkpoint_sha256, "feature_checkpoint_sha256"
        ),
        feature_config_sha256=_sha256(
            feature_config_sha256, "feature_config_sha256"
        ),
        feature_code_sha256=_sha256(
            feature_code_sha256, "feature_code_sha256"
        ),
        proposal_ids=np.asarray(parent.proposal_ids, dtype=np.int64),
        lineage_ids=np.asarray(parent.lineage_ids, dtype=np.int64),
        topk_frame_ids=np.asarray(parent.topk_frame_ids, dtype=np.int64),
        topk_view_valid=np.asarray(parent.topk_view_valid, dtype=np.bool_),
        per_view_feature_valid=valid,
        per_view_feature_count=counts,
        per_view_feature_vector=vectors,
        aggregate_feature_vector=derived["aggregate_feature_vector"],
        aggregate_view_count=derived["aggregate_view_count"],
        aggregate_feature_count=derived["aggregate_feature_count"],
        pairwise_cosine_count=derived["pairwise_cosine_count"],
        pairwise_cosine_mean=derived["pairwise_cosine_mean"],
        pairwise_cosine_median=derived["pairwise_cosine_median"],
        pairwise_cosine_min=derived["pairwise_cosine_min"],
        pairwise_cosine_max=derived["pairwise_cosine_max"],
        pairwise_cosine_std=derived["pairwise_cosine_std"],
        runtime_s=float(runtime_s),
    )
    return validate_tr3d_r2b_payload(
        cache.as_npz_payload(),
        parent_r2a_cache_path=parent_r2a_cache_path,
        parent_tr3d_cache_path=parent_tr3d_cache_path,
        expected_parent_prefix_manifest_row_sha256=(
            parent_prefix_manifest_row_sha256
        ),
        expected_parent_frame_artifact_tree_sha256=(
            parent_frame_artifact_tree_sha256
        ),
        expected_parent_r2_config_sha256=parent_r2_config_sha256,
        expected_parent_r2_code_sha256=parent_r2_code_sha256,
        expected_feature_checkpoint_sha256=feature_checkpoint_sha256,
        expected_feature_config_sha256=feature_config_sha256,
        expected_feature_code_sha256=feature_code_sha256,
    )


def load_tr3d_r2b_cache(
    path: str | os.PathLike[str],
    *,
    parent_r2a_cache_path: str | os.PathLike[str],
    parent_tr3d_cache_path: str | os.PathLike[str],
    expected_parent_prefix_manifest_row_sha256: str,
    expected_parent_frame_artifact_tree_sha256: str,
    expected_parent_r2_config_sha256: str,
    expected_parent_r2_code_sha256: str,
    expected_feature_checkpoint_sha256: str,
    expected_feature_config_sha256: str,
    expected_feature_code_sha256: str,
    expected_scene_id: str | None = None,
    expected_prefix_id: str | None = None,
    expected_prefix_fraction: float | None = None,
) -> TR3DR2BFeatureCache:
    cache_path = Path(path)
    if not cache_path.is_file():
        raise FileNotFoundError(cache_path)
    with np.load(cache_path, allow_pickle=False) as archive:
        values = {
            name: np.array(archive[name], copy=True) for name in archive.files
        }
    return validate_tr3d_r2b_payload(
        values,
        parent_r2a_cache_path=parent_r2a_cache_path,
        parent_tr3d_cache_path=parent_tr3d_cache_path,
        expected_parent_prefix_manifest_row_sha256=(
            expected_parent_prefix_manifest_row_sha256
        ),
        expected_parent_frame_artifact_tree_sha256=(
            expected_parent_frame_artifact_tree_sha256
        ),
        expected_parent_r2_config_sha256=expected_parent_r2_config_sha256,
        expected_parent_r2_code_sha256=expected_parent_r2_code_sha256,
        expected_feature_checkpoint_sha256=expected_feature_checkpoint_sha256,
        expected_feature_config_sha256=expected_feature_config_sha256,
        expected_feature_code_sha256=expected_feature_code_sha256,
        expected_scene_id=expected_scene_id,
        expected_prefix_id=expected_prefix_id,
        expected_prefix_fraction=expected_prefix_fraction,
    )


def write_tr3d_r2b_cache(
    path: str | os.PathLike[str],
    cache: TR3DR2BFeatureCache,
    *,
    parent_r2a_cache_path: str | os.PathLike[str],
    parent_tr3d_cache_path: str | os.PathLike[str],
) -> Path:
    """Atomically create one immutable R2b sidecar; never overwrite."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    canonical = validate_tr3d_r2b_payload(
        cache.as_npz_payload(),
        parent_r2a_cache_path=parent_r2a_cache_path,
        parent_tr3d_cache_path=parent_tr3d_cache_path,
        expected_parent_prefix_manifest_row_sha256=(
            cache.parent_prefix_manifest_row_sha256
        ),
        expected_parent_frame_artifact_tree_sha256=(
            cache.parent_frame_artifact_tree_sha256
        ),
        expected_parent_r2_config_sha256=cache.parent_r2_config_sha256,
        expected_parent_r2_code_sha256=cache.parent_r2_code_sha256,
        expected_feature_checkpoint_sha256=(
            cache.feature_checkpoint_sha256
        ),
        expected_feature_config_sha256=cache.feature_config_sha256,
        expected_feature_code_sha256=cache.feature_code_sha256,
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
            f"immutable R2b cache exists: {target}"
        ) from error
    finally:
        if temporary_name is not None:
            try:
                Path(temporary_name).unlink()
            except FileNotFoundError:
                pass
    return target
