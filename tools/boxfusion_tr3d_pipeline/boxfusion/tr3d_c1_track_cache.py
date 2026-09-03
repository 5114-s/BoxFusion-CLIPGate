"""Immutable C1 unmatched-TR3D multi-view evidence-track sidecars.

C1 does not add, remove, reorder, rescore, or refine a BoxFusion detection.
It only records evidence for TR3D proposals whose AABB IoU with every frozen
R3-active prediction is at most 0.15.  Because the available parent cache is
the terminal ``p100`` prefix, a C1 track is explicitly a cross-*view*
evidence track for one immutable 3D proposal, not a cross-prefix trajectory.
"""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
import hashlib
import math
import os
from pathlib import Path
import re
import tempfile
from typing import Mapping

import numpy as np


SCHEMA = "boxfusion.tr3d_c1_multiview_residual_track.v1"
TRACK_SCOPE = "cross_view_fixed_terminal_prefix"
RESIDUAL_ANCHOR_IOU_MAX = 0.15
MIN_VIEW_SAMPLES = 16
SUPPORTIVE_VIEW_SUPPORT_MIN = 0.10
SUPPORTIVE_VIEW_FREE_MAX = 0.50
CONTRADICTORY_VIEW_FREE_MIN = 0.50
FEATURE_COSINE_MIN = 0.50
GATE_NAMES = (
    "visible2",
    "depth2",
    "depth3_strict",
    "depth_feature2",
)

_SCENE_RE = re.compile(r"^scene[0-9]{4}_[0-9]{2}$")
_PREFIX_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
_SHA_RE = re.compile(r"^[0-9a-f]{64}$")


def sha256_file(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sidecar_path(root: str | os.PathLike[str], scene_id: str, prefix_id: str) -> Path:
    if _SCENE_RE.fullmatch(scene_id) is None:
        raise ValueError(f"invalid scene id: {scene_id!r}")
    if _PREFIX_RE.fullmatch(prefix_id) is None:
        raise ValueError(f"invalid prefix id: {prefix_id!r}")
    return Path(root) / scene_id / f"{prefix_id}.c1-track.npz"


def _readonly(value: object, dtype: np.dtype) -> np.ndarray:
    array = np.ascontiguousarray(np.asarray(value, dtype=dtype))
    array.setflags(write=False)
    return array


def _safe_ratio(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    return np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator, dtype=np.float64),
        where=denominator > 0,
    )


def derive_track_features(
    *,
    tr3d_score: object,
    topk_frame_ids: object,
    topk_view_valid: object,
    per_view_depth_counts: object,
    aggregate_depth_evidence: object,
    per_view_feature_valid: object,
    pairwise_cosine_count: object,
    pairwise_cosine_mean: object,
) -> dict[str, np.ndarray]:
    """Derive every C1 statistic from immutable R2a/R2b evidence."""

    score = np.asarray(tr3d_score, dtype=np.float64)
    frame_ids = np.asarray(topk_frame_ids, dtype=np.int64)
    view_valid = np.asarray(topk_view_valid, dtype=np.bool_)
    counts = np.asarray(per_view_depth_counts, dtype=np.int64)
    aggregate = np.asarray(aggregate_depth_evidence, dtype=np.float64)
    feature_valid = np.asarray(per_view_feature_valid, dtype=np.bool_)
    pair_count = np.asarray(pairwise_cosine_count, dtype=np.int64)
    pair_mean = np.asarray(pairwise_cosine_mean, dtype=np.float64)
    if score.ndim != 1 or frame_ids.ndim != 2:
        raise ValueError("score must be [P] and frame ids must be [P,K]")
    p, k = frame_ids.shape
    if view_valid.shape != (p, k) or counts.shape != (p, k, 4):
        raise ValueError("R2a evidence shape mismatch")
    if aggregate.shape != (p, 4) or feature_valid.shape != (p, k):
        raise ValueError("aggregate/R2b evidence shape mismatch")
    if pair_count.shape != (p,) or pair_mean.shape != (p,) or score.shape != (p,):
        raise ValueError("per-proposal evidence shape mismatch")

    samples = counts.sum(axis=2, dtype=np.int64)
    support = _safe_ratio(counts[:, :, 0], samples)
    free = _safe_ratio(counts[:, :, 2], samples)
    invalid = _safe_ratio(counts[:, :, 3], samples)
    sampled = view_valid & (samples >= MIN_VIEW_SAMPLES)
    supportive = (
        sampled
        & (support >= SUPPORTIVE_VIEW_SUPPORT_MIN)
        & (free <= SUPPORTIVE_VIEW_FREE_MAX)
    )
    contradictory = (
        sampled
        & (free > CONTRADICTORY_VIEW_FREE_MIN)
        & (free > support)
    )
    valid_views = view_valid.sum(axis=1, dtype=np.int32)
    supportive_views = supportive.sum(axis=1, dtype=np.int32)
    contradictory_views = contradictory.sum(axis=1, dtype=np.int32)
    feature_views = feature_valid.sum(axis=1, dtype=np.int32)

    temporal_span = np.zeros(p, dtype=np.int32)
    largest_gap = np.zeros(p, dtype=np.int32)
    for row in range(p):
        selected = np.sort(frame_ids[row, view_valid[row]])
        if len(selected) >= 2:
            temporal_span[row] = int(selected[-1] - selected[0])
            largest_gap[row] = int(np.diff(selected).max(initial=0))

    aggregate_support = aggregate[:, 0]
    aggregate_free = aggregate[:, 2]
    aggregate_invalid = aggregate[:, 3]
    consensus = (supportive_views.astype(np.float64) + 1.0) / (
        valid_views.astype(np.float64) + 2.0
    )
    contradiction = _safe_ratio(
        contradictory_views.astype(np.float64), valid_views.astype(np.float64)
    )
    depth_quality = _safe_ratio(
        aggregate_support, aggregate_support + aggregate_free
    ) * (1.0 - aggregate_invalid)
    feature_quality = np.where(
        pair_count > 0,
        np.clip((pair_mean + 1.0) * 0.5, 0.0, 1.0),
        0.5,
    )
    depth_score = (
        score
        * (0.5 + 0.5 * consensus)
        * (1.0 - 0.5 * contradiction)
        * (0.5 + 0.5 * depth_quality)
    )
    depth_feature_score = depth_score * (0.75 + 0.25 * feature_quality)

    visible2 = valid_views >= 2
    depth2 = (
        (supportive_views >= 2)
        & (contradictory_views <= 1)
        & (aggregate_support >= 0.10)
        & (aggregate_free <= 0.50)
    )
    depth3 = (
        (supportive_views >= 3)
        & (contradictory_views == 0)
        & (aggregate_support >= 0.20)
        & (aggregate_free <= 0.25)
    )
    depth_feature2 = (
        depth2
        & (feature_views >= 2)
        & (pair_count >= 1)
        & (pair_mean >= FEATURE_COSINE_MIN)
    )
    gates = np.stack((visible2, depth2, depth3, depth_feature2), axis=1)
    return {
        "per_view_support_fraction": support.astype(np.float32),
        "per_view_free_space_fraction": free.astype(np.float32),
        "per_view_invalid_fraction": invalid.astype(np.float32),
        "view_supportive": supportive,
        "view_contradictory": contradictory,
        "valid_view_count": valid_views,
        "supportive_view_count": supportive_views,
        "contradictory_view_count": contradictory_views,
        "feature_view_count": feature_views,
        "temporal_span_frames": temporal_span,
        "largest_frame_gap": largest_gap,
        "depth_track_score": depth_score.astype(np.float32),
        "depth_feature_track_score": depth_feature_score.astype(np.float32),
        "gate_mask": gates.astype(np.bool_),
    }


@dataclass(frozen=True)
class TR3DC1TrackCache:
    scene_id: str
    prefix_id: str
    parent_cache_sha256: str
    r2a_cache_sha256: str
    r2b_cache_sha256: str
    anchor_prediction_sha256: str
    config_sha256: str
    code_sha256: str
    proposal_ids: np.ndarray
    parent_rows: np.ndarray
    max_anchor_iou: np.ndarray
    tr3d_score: np.ndarray
    topk_frame_ids: np.ndarray
    topk_view_valid: np.ndarray
    per_view_support_fraction: np.ndarray
    per_view_free_space_fraction: np.ndarray
    per_view_invalid_fraction: np.ndarray
    per_view_sample_count: np.ndarray
    view_supportive: np.ndarray
    view_contradictory: np.ndarray
    valid_view_count: np.ndarray
    supportive_view_count: np.ndarray
    contradictory_view_count: np.ndarray
    feature_view_count: np.ndarray
    temporal_span_frames: np.ndarray
    largest_frame_gap: np.ndarray
    aggregate_depth_evidence: np.ndarray
    aggregate_point_count: np.ndarray
    feature_pair_count: np.ndarray
    feature_pair_cosine_mean: np.ndarray
    depth_track_score: np.ndarray
    depth_feature_track_score: np.ndarray
    gate_mask: np.ndarray
    runtime_s: float = 0.0

    @property
    def track_count(self) -> int:
        return int(np.asarray(self.proposal_ids).shape[0])

    def as_payload(self) -> dict[str, np.ndarray]:
        result = {
            "schema": np.asarray(SCHEMA),
            "complete": np.asarray(True, dtype=np.bool_),
            "observer_only": np.asarray(True, dtype=np.bool_),
            "mutation_enabled": np.asarray(False, dtype=np.bool_),
            "applied_count": np.asarray(0, dtype=np.int64),
            "track_scope": np.asarray(TRACK_SCOPE),
            "cross_prefix_tracking": np.asarray(False, dtype=np.bool_),
            "scene_id": np.asarray(self.scene_id),
            "prefix_id": np.asarray(self.prefix_id),
            "parent_cache_sha256": np.asarray(self.parent_cache_sha256),
            "r2a_cache_sha256": np.asarray(self.r2a_cache_sha256),
            "r2b_cache_sha256": np.asarray(self.r2b_cache_sha256),
            "anchor_prediction_sha256": np.asarray(self.anchor_prediction_sha256),
            "config_sha256": np.asarray(self.config_sha256),
            "code_sha256": np.asarray(self.code_sha256),
            "gate_names": np.asarray(GATE_NAMES),
            "residual_anchor_iou_max": np.asarray(RESIDUAL_ANCHOR_IOU_MAX, dtype=np.float64),
            "proposal_ids": np.asarray(self.proposal_ids, dtype=np.int64),
            "parent_rows": np.asarray(self.parent_rows, dtype=np.int64),
            "max_anchor_iou": np.asarray(self.max_anchor_iou, dtype=np.float32),
            "tr3d_score": np.asarray(self.tr3d_score, dtype=np.float32),
            "topk_frame_ids": np.asarray(self.topk_frame_ids, dtype=np.int64),
            "topk_view_valid": np.asarray(self.topk_view_valid, dtype=np.bool_),
            "per_view_support_fraction": np.asarray(self.per_view_support_fraction, dtype=np.float32),
            "per_view_free_space_fraction": np.asarray(self.per_view_free_space_fraction, dtype=np.float32),
            "per_view_invalid_fraction": np.asarray(self.per_view_invalid_fraction, dtype=np.float32),
            "per_view_sample_count": np.asarray(self.per_view_sample_count, dtype=np.int32),
            "view_supportive": np.asarray(self.view_supportive, dtype=np.bool_),
            "view_contradictory": np.asarray(self.view_contradictory, dtype=np.bool_),
            "valid_view_count": np.asarray(self.valid_view_count, dtype=np.int32),
            "supportive_view_count": np.asarray(self.supportive_view_count, dtype=np.int32),
            "contradictory_view_count": np.asarray(self.contradictory_view_count, dtype=np.int32),
            "feature_view_count": np.asarray(self.feature_view_count, dtype=np.int32),
            "temporal_span_frames": np.asarray(self.temporal_span_frames, dtype=np.int32),
            "largest_frame_gap": np.asarray(self.largest_frame_gap, dtype=np.int32),
            "aggregate_depth_evidence": np.asarray(self.aggregate_depth_evidence, dtype=np.float32),
            "aggregate_point_count": np.asarray(self.aggregate_point_count, dtype=np.int64),
            "feature_pair_count": np.asarray(self.feature_pair_count, dtype=np.int32),
            "feature_pair_cosine_mean": np.asarray(self.feature_pair_cosine_mean, dtype=np.float32),
            "depth_track_score": np.asarray(self.depth_track_score, dtype=np.float32),
            "depth_feature_track_score": np.asarray(self.depth_feature_track_score, dtype=np.float32),
            "gate_mask": np.asarray(self.gate_mask, dtype=np.bool_),
            "runtime_s": np.asarray(self.runtime_s, dtype=np.float64),
        }
        return result


def _scalar(values: Mapping[str, np.ndarray], name: str):
    value = np.asarray(values[name])
    if value.shape != () or value.dtype.hasobject:
        raise ValueError(f"{name} must be a non-object scalar")
    return value.item()


def validate_payload(values: Mapping[str, np.ndarray]) -> TR3DC1TrackCache:
    if str(_scalar(values, "schema")) != SCHEMA:
        raise ValueError("unsupported C1 sidecar schema")
    if not bool(_scalar(values, "complete")):
        raise ValueError("incomplete C1 sidecar")
    if not bool(_scalar(values, "observer_only")) or bool(_scalar(values, "mutation_enabled")):
        raise ValueError("C1 observer contract violation")
    if int(_scalar(values, "applied_count")) != 0:
        raise ValueError("C1 applied_count must be zero")
    if bool(_scalar(values, "cross_prefix_tracking")):
        raise ValueError("terminal-prefix C1 cannot claim cross-prefix tracking")
    if str(_scalar(values, "track_scope")) != TRACK_SCOPE:
        raise ValueError("C1 track scope mismatch")
    scene_id = str(_scalar(values, "scene_id"))
    prefix_id = str(_scalar(values, "prefix_id"))
    if _SCENE_RE.fullmatch(scene_id) is None or _PREFIX_RE.fullmatch(prefix_id) is None:
        raise ValueError("invalid C1 scene/prefix")
    hashes = {}
    for name in (
        "parent_cache_sha256", "r2a_cache_sha256", "r2b_cache_sha256",
        "anchor_prediction_sha256", "config_sha256", "code_sha256",
    ):
        value = str(_scalar(values, name))
        if _SHA_RE.fullmatch(value) is None:
            raise ValueError(f"invalid {name}")
        hashes[name] = value
    if not np.array_equal(np.asarray(values["gate_names"]), np.asarray(GATE_NAMES)):
        raise ValueError("C1 gate names mismatch")
    if not math.isclose(float(_scalar(values, "residual_anchor_iou_max")), RESIDUAL_ANCHOR_IOU_MAX, rel_tol=0, abs_tol=0):
        raise ValueError("C1 residual threshold mismatch")

    proposal_ids = np.asarray(values["proposal_ids"])
    if proposal_ids.ndim != 1 or proposal_ids.dtype != np.int64:
        raise ValueError("proposal_ids must be int64 [P]")
    p = len(proposal_ids)
    if len(np.unique(proposal_ids)) != p:
        raise ValueError("proposal_ids must be unique")
    frame_ids = np.asarray(values["topk_frame_ids"])
    if frame_ids.ndim != 2 or frame_ids.dtype != np.int64 or frame_ids.shape[0] != p:
        raise ValueError("topk_frame_ids must be int64 [P,K]")
    k = frame_ids.shape[1]

    def exact(name: str, dtype: np.dtype, shape: tuple[int, ...]) -> np.ndarray:
        array = np.asarray(values[name])
        if array.dtype != np.dtype(dtype) or array.shape != shape:
            raise ValueError(f"{name} must have dtype {np.dtype(dtype)} and shape {shape}")
        return array

    parent_rows = exact("parent_rows", np.int64, (p,))
    max_iou = exact("max_anchor_iou", np.float32, (p,))
    score = exact("tr3d_score", np.float32, (p,))
    valid = exact("topk_view_valid", np.bool_, (p, k))
    support = exact("per_view_support_fraction", np.float32, (p, k))
    free = exact("per_view_free_space_fraction", np.float32, (p, k))
    invalid = exact("per_view_invalid_fraction", np.float32, (p, k))
    sample_count = exact("per_view_sample_count", np.int32, (p, k))
    view_supportive = exact("view_supportive", np.bool_, (p, k))
    view_contradictory = exact("view_contradictory", np.bool_, (p, k))
    valid_count = exact("valid_view_count", np.int32, (p,))
    supportive_count = exact("supportive_view_count", np.int32, (p,))
    contradictory_count = exact("contradictory_view_count", np.int32, (p,))
    feature_views = exact("feature_view_count", np.int32, (p,))
    temporal_span = exact("temporal_span_frames", np.int32, (p,))
    largest_gap = exact("largest_frame_gap", np.int32, (p,))
    aggregate = exact("aggregate_depth_evidence", np.float32, (p, 4))
    point_count = exact("aggregate_point_count", np.int64, (p,))
    pair_count = exact("feature_pair_count", np.int32, (p,))
    pair_mean = exact("feature_pair_cosine_mean", np.float32, (p,))
    depth_score = exact("depth_track_score", np.float32, (p,))
    feature_score = exact("depth_feature_track_score", np.float32, (p,))
    gates = exact("gate_mask", np.bool_, (p, len(GATE_NAMES)))
    if np.any(parent_rows < 0) or len(np.unique(parent_rows)) != p:
        raise ValueError("parent_rows must be unique and nonnegative")
    if not np.isfinite(max_iou).all() or np.any(max_iou < 0) or np.any(max_iou > RESIDUAL_ANCHOR_IOU_MAX + 1e-6):
        raise ValueError("C1 contains a non-residual proposal")
    for name, array in (("score", score), ("support", support), ("free", free), ("invalid", invalid), ("aggregate", aggregate), ("pair_mean", pair_mean), ("depth_score", depth_score), ("feature_score", feature_score)):
        if not np.isfinite(array).all():
            raise ValueError(f"{name} must be finite")
    if np.any(frame_ids[~valid] != -1):
        raise ValueError("invalid view slots must use frame -1")
    if np.any(view_supportive & ~valid) or np.any(view_contradictory & ~valid):
        raise ValueError("view decisions must be subsets of valid views")
    if not np.array_equal(valid_count, valid.sum(1, dtype=np.int32)):
        raise ValueError("valid view count mismatch")
    if not np.array_equal(supportive_count, view_supportive.sum(1, dtype=np.int32)):
        raise ValueError("supportive view count mismatch")
    if not np.array_equal(contradictory_count, view_contradictory.sum(1, dtype=np.int32)):
        raise ValueError("contradictory view count mismatch")
    if np.any(sample_count < 0) or np.any(point_count < 0) or np.any(pair_count < 0) or np.any(feature_views < 0):
        raise ValueError("negative C1 evidence count")
    if np.any(temporal_span < 0) or np.any(largest_gap < 0):
        raise ValueError("negative temporal statistic")
    if np.any(feature_views > valid_count):
        raise ValueError("feature views cannot exceed valid views")
    if np.any((support < 0) | (support > 1)) or np.any((free < 0) | (free > 1)) or np.any((invalid < 0) | (invalid > 1)):
        raise ValueError("per-view fractions must lie in [0,1]")
    expected_supportive = (
        valid & (sample_count >= MIN_VIEW_SAMPLES)
        & (support >= SUPPORTIVE_VIEW_SUPPORT_MIN)
        & (free <= SUPPORTIVE_VIEW_FREE_MAX)
    )
    expected_contradictory = (
        valid & (sample_count >= MIN_VIEW_SAMPLES)
        & (free > CONTRADICTORY_VIEW_FREE_MIN) & (free > support)
    )
    if not np.array_equal(view_supportive, expected_supportive):
        raise ValueError("supportive-view decision mismatch")
    if not np.array_equal(view_contradictory, expected_contradictory):
        raise ValueError("contradictory-view decision mismatch")
    expected_span = np.zeros(p, dtype=np.int32)
    expected_gap = np.zeros(p, dtype=np.int32)
    for row in range(p):
        selected = np.sort(frame_ids[row, valid[row]])
        if len(selected) >= 2:
            expected_span[row] = int(selected[-1] - selected[0])
            expected_gap[row] = int(np.diff(selected).max(initial=0))
    if not np.array_equal(temporal_span, expected_span) or not np.array_equal(largest_gap, expected_gap):
        raise ValueError("temporal track statistics mismatch")
    expected_gates = np.stack(
        (
            valid_count >= 2,
            (supportive_count >= 2) & (contradictory_count <= 1)
            & (aggregate[:, 0] >= 0.10) & (aggregate[:, 2] <= 0.50),
            (supportive_count >= 3) & (contradictory_count == 0)
            & (aggregate[:, 0] >= 0.20) & (aggregate[:, 2] <= 0.25),
            (supportive_count >= 2) & (contradictory_count <= 1)
            & (aggregate[:, 0] >= 0.10) & (aggregate[:, 2] <= 0.50)
            & (feature_views >= 2) & (pair_count >= 1)
            & (pair_mean >= FEATURE_COSINE_MIN),
        ),
        axis=1,
    )
    if not np.array_equal(gates, expected_gates):
        raise ValueError("fixed C1 gate decision mismatch")
    consensus = (supportive_count.astype(np.float64) + 1.0) / (
        valid_count.astype(np.float64) + 2.0
    )
    contradiction = _safe_ratio(
        contradictory_count.astype(np.float64), valid_count.astype(np.float64)
    )
    depth_quality = _safe_ratio(
        aggregate[:, 0].astype(np.float64),
        (aggregate[:, 0] + aggregate[:, 2]).astype(np.float64),
    ) * (1.0 - aggregate[:, 3].astype(np.float64))
    expected_depth_score64 = (
        score.astype(np.float64) * (0.5 + 0.5 * consensus)
        * (1.0 - 0.5 * contradiction) * (0.5 + 0.5 * depth_quality)
    )
    expected_depth_score = expected_depth_score64.astype(np.float32)
    feature_quality = np.where(
        pair_count > 0, np.clip((pair_mean.astype(np.float64) + 1.0) * 0.5, 0.0, 1.0), 0.5
    )
    expected_feature_score = (
        expected_depth_score64 * (0.75 + 0.25 * feature_quality)
    ).astype(np.float32)
    if not np.allclose(depth_score, expected_depth_score, rtol=0.0, atol=1e-7):
        raise ValueError("depth track score mismatch")
    if not np.allclose(feature_score, expected_feature_score, rtol=0.0, atol=1e-7):
        raise ValueError("depth-feature track score mismatch")

    runtime = float(_scalar(values, "runtime_s"))
    if not math.isfinite(runtime) or runtime < 0:
        raise ValueError("runtime must be finite and nonnegative")
    return TR3DC1TrackCache(
        scene_id=scene_id, prefix_id=prefix_id, **hashes,
        proposal_ids=_readonly(proposal_ids, np.int64),
        parent_rows=_readonly(parent_rows, np.int64),
        max_anchor_iou=_readonly(max_iou, np.float32),
        tr3d_score=_readonly(score, np.float32),
        topk_frame_ids=_readonly(frame_ids, np.int64),
        topk_view_valid=_readonly(valid, np.bool_),
        per_view_support_fraction=_readonly(support, np.float32),
        per_view_free_space_fraction=_readonly(free, np.float32),
        per_view_invalid_fraction=_readonly(invalid, np.float32),
        per_view_sample_count=_readonly(sample_count, np.int32),
        view_supportive=_readonly(view_supportive, np.bool_),
        view_contradictory=_readonly(view_contradictory, np.bool_),
        valid_view_count=_readonly(valid_count, np.int32),
        supportive_view_count=_readonly(supportive_count, np.int32),
        contradictory_view_count=_readonly(contradictory_count, np.int32),
        feature_view_count=_readonly(feature_views, np.int32),
        temporal_span_frames=_readonly(temporal_span, np.int32),
        largest_frame_gap=_readonly(largest_gap, np.int32),
        aggregate_depth_evidence=_readonly(aggregate, np.float32),
        aggregate_point_count=_readonly(point_count, np.int64),
        feature_pair_count=_readonly(pair_count, np.int32),
        feature_pair_cosine_mean=_readonly(pair_mean, np.float32),
        depth_track_score=_readonly(depth_score, np.float32),
        depth_feature_track_score=_readonly(feature_score, np.float32),
        gate_mask=_readonly(gates, np.bool_), runtime_s=runtime,
    )


def write_sidecar(path: str | os.PathLike[str], cache: TR3DC1TrackCache) -> str:
    target = Path(path)
    canonical = validate_payload(cache.as_payload())
    buffer = BytesIO()
    np.savez_compressed(buffer, **canonical.as_payload())
    encoded = buffer.getvalue()
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent, delete=False) as handle:
            temporary_name = handle.name
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary_name, target)
        target.chmod(0o444)
    except FileExistsError as error:
        raise FileExistsError(f"immutable C1 sidecar exists: {target}") from error
    finally:
        if temporary_name is not None:
            try:
                os.unlink(temporary_name)
            except FileNotFoundError:
                pass
    return sha256_file(target)


def load_sidecar(path: str | os.PathLike[str]) -> TR3DC1TrackCache:
    with np.load(Path(path), allow_pickle=False) as values:
        payload = {name: values[name] for name in values.files}
    return validate_payload(payload)


__all__ = [
    "FEATURE_COSINE_MIN", "GATE_NAMES", "RESIDUAL_ANCHOR_IOU_MAX",
    "SCHEMA", "TRACK_SCOPE", "TR3DC1TrackCache", "derive_track_features",
    "load_sidecar", "sha256_file", "sidecar_path", "write_sidecar",
]
