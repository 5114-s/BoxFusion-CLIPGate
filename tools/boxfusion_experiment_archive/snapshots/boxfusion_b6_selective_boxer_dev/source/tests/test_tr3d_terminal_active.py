from __future__ import annotations

import json
import hashlib
from pathlib import Path
import pickle

import numpy as np
import pytest

from boxfusion.tr3d_terminal_active import (
    FROZEN_CHECKPOINT_SHA256,
    FROZEN_CONFIG_SHA256,
    TerminalR3CacheReplay,
    prediction_payload,
    save_prediction_create_only,
)


_SIGNS = np.asarray(
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
    dtype=np.float32,
)


def _corners(center, extent):
    return np.asarray(center, dtype=np.float32) + _SIGNS * (
        np.asarray(extent, dtype=np.float32) * 0.5
    )


def test_prediction_payload_canonicalizes_types_and_preserves_empty_shape():
    assert prediction_payload(
        np.empty((0, 8, 3), dtype=np.float64),
        np.empty((0,), dtype=np.float64),
    ) == [[]]

    corners = np.stack((_corners([0, 0, 0], [1, 1, 1]),)).astype(
        np.float64
    )
    payload = prediction_payload(corners, np.asarray([0.75], dtype=np.float64))
    label, saved_corners, score = payload[0][0]
    assert label == 0
    assert saved_corners.dtype == np.dtype(np.float32)
    assert saved_corners.flags.c_contiguous
    assert score == pytest.approx(0.75)


def test_prediction_create_only_writes_empty_payload_and_refuses_overwrite(
    tmp_path,
):
    target = tmp_path / "scene0000_00_boxes.pkl"
    saved = save_prediction_create_only(
        np.empty((0, 8, 3), dtype=np.float32),
        np.empty((0,), dtype=np.float32),
        target,
    )
    original = target.read_bytes()
    assert saved == target.resolve()
    with target.open("rb") as handle:
        assert pickle.load(handle) == [[]]

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        save_prediction_create_only(
            np.empty((0, 8, 3), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
            target,
        )
    assert target.read_bytes() == original
    assert not list(tmp_path.glob(f".{target.name}.*.tmp"))


def _fixture(tmp_path: Path, *, diagnostics: bool = False):
    manifest = tmp_path / "prefix.jsonl"
    point_path = tmp_path / "p100.bin"
    points = np.zeros((10, 6), dtype=np.float32)
    point_path.write_bytes(points.tobytes())
    point_sha256 = hashlib.sha256(point_path.read_bytes()).hexdigest()
    row = {
        "schema": "boxfusion.tr3d.trajectory_prefix.v1",
        "clock_policy": "g0_post_frame_tail_guard_v1",
        "status": "exported",
        "pose_policy": "previous_valid_inf_only_v1",
        "source_timestamp_semantics": "zero_based_scannet_dataset_index",
        "coordinate_frame": "world_unaligned",
        "frame_stride": 25,
        "tail_guard_frames": 25,
        "source_frame_count": 80,
        "processed_frame_count": 55,
        "sampled_frame_count": 3,
        "scene_id": "scene0000_00",
        "tag": "p100",
        "fraction": 1.0,
        "last_source_timestamp": 50,
        "source_timestamps": [0, 25, 50],
        "used_source_timestamps": [0, 25, 50],
        "pose_provenance": [
            {
                "source_timestamp": value,
                "resolved_pose_source_timestamp": value,
                "resolved_pose_sha256": "1" * 64,
            }
            for value in (0, 25, 50)
        ],
        "point_count": 10,
        "point_path": str(point_path),
        "axis_align_matrix": np.eye(4).tolist(),
    }
    manifest.write_text(json.dumps(row) + "\n", encoding="utf-8")
    cache_root = tmp_path / "cache"
    cache_path = cache_root / "scene0000_00" / "p100.npz"
    cache_path.parent.mkdir(parents=True)
    proposal_corners = np.stack(
        (
            _corners([0.05, 0.0, 0.0], [0.9, 1.0, 1.0]),
            _corners([0.0, 0.0, 0.0], [0.8, 1.0, 1.0]),
        )
    ).astype(np.float32)
    boxes_world = np.asarray(
        [
            [0.05, 0.0, 0.0, 0.9, 1.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.8, 1.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )
    axis_hash = hashlib.sha256(
        np.asarray(np.eye(4), dtype="<f8").tobytes()
    ).hexdigest()
    np.savez_compressed(
        cache_path,
        schema=np.asarray("boxfusion.tr3d_residual_cache.v1"),
        scene_id=np.asarray("scene0000_00"),
        sample_idx=np.asarray("scene0000_00:p100"),
        prefix_id=np.asarray("p100"),
        prefix_fraction=np.asarray(1.0, dtype=np.float64),
        complete=np.asarray(True, dtype=np.bool_),
        observer_only=np.asarray(True, dtype=np.bool_),
        mutation_enabled=np.asarray(False, dtype=np.bool_),
        applied_count=np.asarray(0, dtype=np.int64),
        class_agnostic=np.asarray(True, dtype=np.bool_),
        coordinate_frame=np.asarray("scannet_unaligned_world"),
        box_mode=np.asarray("depth_center_size_yaw_z"),
        corner_semantics=np.asarray("unordered_8_corners_minmax_only"),
        boxes_world=boxes_world,
        checkpoint_sha256=np.asarray(FROZEN_CHECKPOINT_SHA256),
        config_sha256=np.asarray(FROZEN_CONFIG_SHA256),
        source_scene_sha256=np.asarray(point_sha256),
        proposal_ids=np.asarray([10, 5], dtype=np.int64),
        corners_world=proposal_corners,
        axis_alignment_sha256=np.asarray(axis_hash),
        scores_3d=np.asarray([0.8, 0.8], dtype=np.float32),
        labels_3d=np.asarray([0, 0], dtype=np.int64),
        point_count=np.asarray([8, 7], dtype=np.int32),
        voxel_size=np.asarray(0.01, dtype=np.float64),
        aligned_to_unaligned=np.eye(4, dtype=np.float64),
        runtime_s=np.asarray(0.025, dtype=np.float64),
        num_input_points=np.asarray(10, dtype=np.int64),
    )
    diagnostic_root = tmp_path / "diagnostics" if diagnostics else None
    controller = TerminalR3CacheReplay(
        manifest_path=manifest,
        parent_cache_root=cache_root,
        diagnostics_root=diagnostic_root,
    )
    return controller, proposal_corners, diagnostic_root


def test_terminal_overlay_uses_score_then_proposal_id_tie_break(tmp_path):
    controller, proposals, _ = _fixture(tmp_path)
    anchors = np.stack((_corners([0, 0, 0], [1, 1, 1]),)).astype(np.float32)
    original = anchors.copy()
    scores = np.asarray([0.5], dtype=np.float64)

    output, summary = controller.apply(
        scene_id="scene0000_00",
        current_source_timestamp=50,
        observed_source_timestamps=(0, 25, 50),
        anchor_corners_world=anchors,
        anchor_scores=scores,
    )

    assert np.array_equal(anchors, original)
    assert np.array_equal(output[0], proposals[1])
    assert summary.selected_count == 1
    assert summary.changed_count == 1
    assert summary.selections[0].proposal_id == 5
    assert summary.cache_model_runtime_s == pytest.approx(0.025)


def test_terminal_overlay_rejects_future_or_nonterminal_use(tmp_path):
    controller, _, _ = _fixture(tmp_path)
    anchors = np.stack((_corners([0, 0, 0], [1, 1, 1]),)).astype(np.float32)
    with pytest.raises(ValueError, match="exact final observed"):
        controller.apply(
            scene_id="scene0000_00",
            current_source_timestamp=25,
            observed_source_timestamps=(0, 25),
            anchor_corners_world=anchors,
            anchor_scores=np.asarray([0.5]),
        )


def test_terminal_overlay_rejects_point_lineage_tampering(tmp_path):
    controller, _, _ = _fixture(tmp_path)
    controller.prefixes["scene0000_00"].point_path.write_bytes(
        np.ones((10, 6), dtype=np.float32).tobytes()
    )
    anchors = np.stack((_corners([0, 0, 0], [1, 1, 1]),)).astype(np.float32)
    with pytest.raises(ValueError, match="source point SHA256"):
        controller.apply(
            scene_id="scene0000_00",
            current_source_timestamp=50,
            observed_source_timestamps=(0, 25, 50),
            anchor_corners_world=anchors,
            anchor_scores=np.asarray([0.5]),
        )


def test_terminal_overlay_preserves_identity_when_score_does_not_pass(tmp_path):
    controller, _, _ = _fixture(tmp_path)
    anchors = np.stack((_corners([0, 0, 0], [1, 1, 1]),)).astype(np.float32)
    output, summary = controller.apply(
        scene_id="scene0000_00",
        current_source_timestamp=50,
        observed_source_timestamps=(0, 25, 50),
        anchor_corners_world=anchors,
        anchor_scores=np.asarray([0.9]),
    )
    assert output.dtype == anchors.dtype
    assert output.tobytes() == anchors.tobytes()
    assert summary.selected_count == 0
    assert summary.changed_count == 0


def test_diagnostic_is_create_only_and_declares_replay_latency(tmp_path):
    controller, _, diagnostic_root = _fixture(tmp_path, diagnostics=True)
    anchors = np.stack((_corners([0, 0, 0], [1, 1, 1]),)).astype(np.float32)
    kwargs = dict(
        scene_id="scene0000_00",
        current_source_timestamp=50,
        observed_source_timestamps=(0, 25, 50),
        anchor_corners_world=anchors,
        anchor_scores=np.asarray([0.5]),
    )
    controller.apply(**kwargs)
    diagnostic = json.loads(
        (diagnostic_root / "scene0000_00_tr3d_terminal.json").read_text()
    )
    assert diagnostic["ground_truth_access"] is False
    assert diagnostic["clip_access"] is False
    assert diagnostic["provider_mode"] == "immutable_parent_cache_replay"
    assert diagnostic["live_tr3d_latency_authoritative"] is False
    with pytest.raises(FileExistsError, match="already exists"):
        controller.apply(**kwargs)
