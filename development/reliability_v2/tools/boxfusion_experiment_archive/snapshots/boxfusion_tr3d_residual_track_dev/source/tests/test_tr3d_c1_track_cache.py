from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from boxfusion.tr3d_c1_track_cache import (
    GATE_NAMES,
    TR3DC1TrackCache,
    derive_track_features,
    load_sidecar,
    validate_payload,
    write_sidecar,
)


HASH = "a" * 64


def _cache() -> TR3DC1TrackCache:
    frame_ids = np.asarray([[10, 20, 30], [15, -1, -1]], dtype=np.int64)
    valid = frame_ids >= 0
    counts = np.asarray(
        [
            [[30, 5, 5, 10], [20, 5, 5, 10], [20, 0, 0, 10]],
            [[2, 1, 25, 4], [0, 0, 0, 0], [0, 0, 0, 0]],
        ],
        dtype=np.int32,
    )
    aggregate_counts = counts.sum(axis=1, dtype=np.int64)
    aggregate_points = aggregate_counts.sum(axis=1, dtype=np.int64)
    aggregate = np.divide(
        aggregate_counts,
        aggregate_points[:, None],
        out=np.zeros_like(aggregate_counts, dtype=np.float64),
        where=aggregate_points[:, None] > 0,
    ).astype(np.float32)
    feature_valid = np.asarray([[True, True, True], [True, False, False]])
    pair_count = np.asarray([3, 0], dtype=np.int32)
    pair_mean = np.asarray([0.8, 0.0], dtype=np.float32)
    score = np.asarray([0.8, 0.4], dtype=np.float32)
    derived = derive_track_features(
        tr3d_score=score,
        topk_frame_ids=frame_ids,
        topk_view_valid=valid,
        per_view_depth_counts=counts,
        aggregate_depth_evidence=aggregate,
        per_view_feature_valid=feature_valid,
        pairwise_cosine_count=pair_count,
        pairwise_cosine_mean=pair_mean,
    )
    return TR3DC1TrackCache(
        scene_id="scene0000_00",
        prefix_id="p100",
        parent_cache_sha256=HASH,
        r2a_cache_sha256=HASH,
        r2b_cache_sha256=HASH,
        anchor_prediction_sha256=HASH,
        config_sha256=HASH,
        code_sha256=HASH,
        proposal_ids=np.asarray([3, 7], dtype=np.int64),
        parent_rows=np.asarray([1, 4], dtype=np.int64),
        max_anchor_iou=np.asarray([0.0, 0.15], dtype=np.float32),
        tr3d_score=score,
        topk_frame_ids=frame_ids,
        topk_view_valid=valid,
        per_view_sample_count=counts.sum(axis=2, dtype=np.int32),
        aggregate_depth_evidence=aggregate,
        aggregate_point_count=aggregate_points,
        feature_pair_count=pair_count,
        feature_pair_cosine_mean=pair_mean,
        runtime_s=0.01,
        **derived,
    )


def test_c1_roundtrip_and_fixed_gates(tmp_path):
    cache = _cache()
    path = tmp_path / "scene0000_00" / "p100.c1-track.npz"
    write_sidecar(path, cache)
    loaded = load_sidecar(path)
    assert loaded.track_count == 2
    assert loaded.gate_mask.shape == (2, len(GATE_NAMES))
    assert loaded.gate_mask[0].tolist() == [True, True, True, True]
    assert loaded.gate_mask[1].tolist() == [False, False, False, False]
    assert loaded.temporal_span_frames.tolist() == [20, 0]
    with pytest.raises(FileExistsError):
        write_sidecar(path, cache)


def test_c1_rejects_gate_tampering():
    payload = _cache().as_payload()
    payload["gate_mask"] = payload["gate_mask"].copy()
    payload["gate_mask"][0, 1] = False
    with pytest.raises(ValueError, match="gate decision mismatch"):
        validate_payload(payload)


def test_c1_rejects_matched_candidate():
    cache = replace(
        _cache(), max_anchor_iou=np.asarray([0.151, 0.0], dtype=np.float32)
    )
    with pytest.raises(ValueError, match="non-residual"):
        validate_payload(cache.as_payload())
