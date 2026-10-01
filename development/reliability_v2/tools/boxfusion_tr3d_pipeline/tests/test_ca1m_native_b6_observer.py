from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from boxfusion.ca1m_native_b6_observer import (
    FEATURE_NAMES,
    SCHEMA,
    build_ca1m_native_b6_observer,
)
from boxfusion.tr3d_r2_geometry import yaw_obb_corners_world


def _cfg(root: Path, *, top_k: int = 2) -> dict:
    return {
        "dataset": "CA1M",
        "online_refinement": {"enabled": False},
        "ca1m_native_b6_observer": {
            "enabled": True,
            "observer_only": True,
            "top_k_views": top_k,
            "pixel_stride": 1,
            "depth_margin_m": 0.0,
            "min_depth_m": 0.1,
            "max_depth_m": 8.0,
            "near_clip_m": 1e-3,
            "max_cached_keyframes": 8,
            "diagnostics": {"enabled": True, "root": str(root)},
        },
    }


def _k(shape=(17, 17), focal=12.0) -> np.ndarray:
    height, width = shape
    return np.asarray(
        [[focal, 0, (width - 1) / 2], [0, focal, (height - 1) / 2], [0, 0, 1]],
        dtype=np.float64,
    )


def _record(observer, frame_id: int, depth: np.ndarray) -> None:
    observer.record_keyframe(
        scene_id="42898811",
        frame_id=frame_id,
        source_frame_id=str(frame_id),
        depth_meters=depth,
        intrinsics=_k(depth.shape),
        camera_to_world=np.eye(4, dtype=np.float64),
    )


def test_final_box_observer_maps_every_row_and_is_create_only(tmp_path: Path) -> None:
    observer = build_ca1m_native_b6_observer(_cfg(tmp_path))
    _record(observer, 0, np.full((17, 17), 5.0, dtype=np.float32))
    _record(observer, 20, np.full((17, 17), 5.0, dtype=np.float32))
    corners = np.stack(
        (
            yaw_obb_corners_world([0, 0, 5, 2, 2, 2, 0.4]),
            yaw_obb_corners_world([100, 0, 5, 1, 1, 1, 0.0]),
        )
    ).astype(np.float32)
    scores = np.asarray([0.8, 0.4], dtype=np.float32)
    stable_ids = np.asarray([3, 9], dtype=np.int64)
    corners_before, scores_before, ids_before = (
        corners.copy(), scores.copy(), stable_ids.copy()
    )

    summary = observer.finalize(
        scene_id="42898811",
        corners=corners,
        scores=scores,
        stable_ids=stable_ids,
    )
    assert summary.prediction_rows == 2
    assert summary.mapping_rows == 2
    assert summary.projectable_rows == 1
    assert summary.valid_evidence_rows == 1
    assert np.array_equal(corners, corners_before)
    assert np.array_equal(scores, scores_before)
    assert np.array_equal(stable_ids, ids_before)

    path = tmp_path / "42898811_ca1m_native_b6.npz"
    assert path.is_file()
    with np.load(path, allow_pickle=False) as payload:
        assert payload["schema"].item() == SCHEMA
        np.testing.assert_array_equal(payload["result_indices"], [0, 1])
        np.testing.assert_array_equal(payload["stable_ids"], stable_ids)
        np.testing.assert_array_equal(payload["corners"], corners)
        np.testing.assert_array_equal(payload["scores"], scores)
        assert tuple(payload["feature_names"].tolist()) == FEATURE_NAMES
        assert payload["features"].shape == (2, len(FEATURE_NAMES))
        assert np.isfinite(payload["features"]).all()
        np.testing.assert_array_equal(payload["valid_evidence"], [True, False])
        summary_json = json.loads(payload["summary_json"].item())
        assert summary_json["mapping_rows"] == 2
        assert not summary_json["ground_truth_access"]
        assert not summary_json["clip_access"]

    with pytest.raises(FileExistsError, match="immutable"):
        observer.finalize(
            scene_id="42898811",
            corners=corners,
            scores=scores,
            stable_ids=stable_ids,
        )


def test_depth_moves_occluded_to_support_to_free_space(tmp_path: Path) -> None:
    counts = []
    for name, depth_value in (("near", 3.0), ("inside", 5.0), ("far", 7.0)):
        root = tmp_path / name
        observer = build_ca1m_native_b6_observer(_cfg(root, top_k=1))
        _record(observer, 0, np.full((17, 17), depth_value, dtype=np.float32))
        corners = yaw_obb_corners_world([0, 0, 5, 2, 2, 2, 0.5])[None].astype(
            np.float32
        )
        observer.finalize(
            scene_id="42898811",
            corners=corners,
            scores=np.asarray([0.7], dtype=np.float32),
            stable_ids=np.asarray([1], dtype=np.int64),
        )
        with np.load(root / "42898811_ca1m_native_b6.npz", allow_pickle=False) as p:
            counts.append(p["aggregate_depth_counts"][0].copy())
    assert counts[0][1] > 0  # foreground occlusion
    assert counts[1][0] > 0  # surface support
    assert counts[2][2] > 0  # free-space conflict


def test_zero_predictions_still_produces_complete_sidecar(tmp_path: Path) -> None:
    observer = build_ca1m_native_b6_observer(_cfg(tmp_path))
    _record(observer, 0, np.ones((9, 11), dtype=np.float32))
    summary = observer.finalize(
        scene_id="42898811",
        corners=np.empty((0, 8, 3), dtype=np.float32),
        scores=np.empty((0,), dtype=np.float32),
        stable_ids=np.empty((0,), dtype=np.int64),
    )
    assert summary.mapping_rows == 0
    with np.load(tmp_path / "42898811_ca1m_native_b6.npz", allow_pickle=False) as p:
        assert p["features"].shape == (0, len(FEATURE_NAMES))
        assert p["result_indices"].shape == (0,)


def test_record_rejects_bad_principal_point_and_duplicate_frame(tmp_path: Path) -> None:
    observer = build_ca1m_native_b6_observer(_cfg(tmp_path))
    depth = np.ones((9, 11), dtype=np.float32)
    bad = _k(depth.shape)
    bad[0, 2] = 999
    with pytest.raises(ValueError, match="principal point"):
        observer.record_keyframe(
            scene_id="42898811", frame_id=0, source_frame_id="0",
            depth_meters=depth, intrinsics=bad,
            camera_to_world=np.eye(4),
        )
    _record(observer, 0, depth)
    with pytest.raises(ValueError, match="unique"):
        _record(observer, 0, depth)


def test_disabled_builder_is_noop() -> None:
    observer = build_ca1m_native_b6_observer({})
    assert not observer.enabled
    observer.record_keyframe(
        scene_id="ignored", frame_id=0, source_frame_id="0",
        depth_meters=np.zeros((1, 1)), intrinsics=np.eye(3),
        camera_to_world=np.eye(4),
    )
