"""Paired multi-view DINO evidence for R4 terminal-R3 harm verification."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from .tr3d_r2_observer import TR3DR2ObserverConfig
from .tr3d_r2b_observer import (
    FEATURE_STAT_NAMES,
    TR3DR2BFrameBundle,
    observe_tr3d_r2b_scene,
)


def _readonly(value: object, dtype: np.dtype) -> np.ndarray:
    result = np.ascontiguousarray(value, dtype=dtype)
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class R4PairedFeatureObservation:
    scene_id: str
    proposal_ids: np.ndarray
    anchor_indices: np.ndarray
    topk_frame_ids: np.ndarray
    topk_view_valid: np.ndarray
    feature_view_valid: np.ndarray
    per_view_feature_count: np.ndarray
    per_view_support_point_count: np.ndarray
    per_view_features: np.ndarray
    aggregate_feature_statistics: np.ndarray
    aggregate_feature_view_count: np.ndarray
    aggregate_feature_pair_count: np.ndarray
    candidate_minus_anchor_statistics: np.ndarray
    decoded_rgb_frame_ids: np.ndarray
    decoded_depth_frame_ids: np.ndarray
    encoded_frame_ids: np.ndarray
    feature_runtime_s: float
    geometry_runtime_s: float
    total_runtime_s: float

    @property
    def pair_count(self) -> int:
        return int(self.proposal_ids.shape[0])


def observe_r3_replacement_pair_features(
    *,
    anchor_boxes_world: object,
    candidate_boxes_world: object,
    proposal_ids: object,
    anchor_indices: object,
    topk_frame_ids: object,
    topk_view_valid: object,
    frame_bundle: TR3DR2BFrameBundle,
    depth_config: TR3DR2ObserverConfig,
    encode_rgb: Callable[[np.ndarray], object],
    min_support_points: int = 2,
    min_feature_cells: int = 1,
    decode_rgb: Optional[Callable[[object], object]] = None,
    decode_depth: Optional[Callable[[object], object]] = None,
) -> R4PairedFeatureObservation:
    """Pool anchor/candidate DINO features over the shared R4-D Top-K."""

    anchors = np.asarray(anchor_boxes_world, dtype=np.float64)
    candidates = np.asarray(candidate_boxes_world, dtype=np.float64)
    if (
        anchors.ndim != 2
        or anchors.shape[1] != 7
        or candidates.shape != anchors.shape
        or not np.isfinite(anchors).all()
        or not np.isfinite(candidates).all()
        or np.any(anchors[:, 3:6] <= 0.0)
        or np.any(candidates[:, 3:6] <= 0.0)
    ):
        raise ValueError("anchor/candidate boxes must be finite positive [N,7]")
    count = len(anchors)
    ids = np.asarray(proposal_ids)
    anchor_ids = np.asarray(anchor_indices)
    if (
        ids.dtype.kind not in "iu"
        or anchor_ids.dtype.kind not in "iu"
        or ids.shape != (count,)
        or anchor_ids.shape != (count,)
        or np.any(ids < 0)
        or np.any(anchor_ids < 0)
        or len(np.unique(ids)) != count
        or len(np.unique(anchor_ids)) != count
    ):
        raise ValueError("proposal_ids/anchor_indices must be unique nonnegative [N]")
    frames = np.asarray(topk_frame_ids)
    valid = np.asarray(topk_view_valid)
    if (
        frames.dtype.kind not in "iu"
        or frames.ndim != 2
        or frames.shape[0] != count
        or valid.shape != frames.shape
        or valid.dtype != np.bool_
    ):
        raise ValueError("Top-K arrays must be integer/boolean [N,K]")

    paired_boxes = np.stack((anchors, candidates), axis=1)
    flattened_boxes = paired_boxes.reshape(count * 2, 7)
    flattened_frames = np.repeat(frames, 2, axis=0)
    flattened_valid = np.repeat(valid, 2, axis=0)
    # Synthetic row ids are internal to the reused R2b pure-compute kernel;
    # the returned paired object restores authoritative proposal/anchor ids.
    synthetic_ids = np.arange(count * 2, dtype=np.int64)
    observed = observe_tr3d_r2b_scene(
        boxes_world=flattened_boxes,
        proposal_ids=synthetic_ids,
        topk_frame_ids=flattened_frames,
        topk_view_valid=flattened_valid,
        frame_bundle=frame_bundle,
        depth_config=depth_config,
        encode_rgb=encode_rgb,
        min_support_points=min_support_points,
        min_feature_cells=min_feature_cells,
        decode_rgb=decode_rgb,
        decode_depth=decode_depth,
    )
    topk = frames.shape[1]
    feature_dim = observed.per_view_features.shape[2]

    def paired(value: np.ndarray, *tail: int) -> np.ndarray:
        reshaped = np.asarray(value).reshape(count, 2, topk, *tail)
        axes = (0, 2, 1) + tuple(range(3, reshaped.ndim))
        return np.transpose(reshaped, axes)

    feature_valid = paired(observed.feature_view_valid)
    feature_counts = paired(observed.per_view_feature_count)
    support_counts = paired(observed.per_view_support_point_count)
    per_view_features = paired(observed.per_view_features, feature_dim)
    statistics = observed.aggregate_feature_statistics.reshape(
        count, 2, len(FEATURE_STAT_NAMES)
    )
    view_counts = observed.aggregate_feature_view_count.reshape(count, 2)
    pair_counts = observed.aggregate_feature_pair_count.reshape(count, 2)
    delta = statistics[:, 1] - statistics[:, 0]
    return R4PairedFeatureObservation(
        scene_id=frame_bundle.scene_id,
        proposal_ids=_readonly(ids, np.int64),
        anchor_indices=_readonly(anchor_ids, np.int64),
        topk_frame_ids=_readonly(frames, np.int64),
        topk_view_valid=_readonly(valid, np.bool_),
        feature_view_valid=_readonly(feature_valid, np.bool_),
        per_view_feature_count=_readonly(feature_counts, np.int32),
        per_view_support_point_count=_readonly(support_counts, np.int32),
        per_view_features=_readonly(per_view_features, np.float32),
        aggregate_feature_statistics=_readonly(statistics, np.float32),
        aggregate_feature_view_count=_readonly(view_counts, np.int32),
        aggregate_feature_pair_count=_readonly(pair_counts, np.int32),
        candidate_minus_anchor_statistics=_readonly(delta, np.float32),
        decoded_rgb_frame_ids=_readonly(observed.decoded_rgb_frame_ids, np.int64),
        decoded_depth_frame_ids=_readonly(observed.decoded_depth_frame_ids, np.int64),
        encoded_frame_ids=_readonly(observed.encoded_frame_ids, np.int64),
        feature_runtime_s=float(observed.feature_runtime_s),
        geometry_runtime_s=float(observed.geometry_runtime_s),
        total_runtime_s=float(observed.total_runtime_s),
    )


__all__ = [
    "FEATURE_STAT_NAMES",
    "R4PairedFeatureObservation",
    "observe_r3_replacement_pair_features",
]
