from __future__ import annotations

import json
import hashlib

import numpy as np
import pytest

from boxfusion.ca1m_tr3d_terminal import (
    SCHEMA,
    TerminalObserverSummary,
    aligned_boxes_to_world_corners,
    associate_terminal_candidates,
    observation_payload,
    terminal_world_to_local,
    voxel_downsample_first,
    write_npz_create_only,
)


SIGNS = np.asarray(
    [
        [-1, -1, -1], [-1, -1, 1], [-1, 1, -1], [-1, 1, 1],
        [1, -1, -1], [1, -1, 1], [1, 1, -1], [1, 1, 1],
    ],
    dtype=np.float32,
)


def corners(center, extent):
    return np.asarray(center, dtype=np.float32) + SIGNS * (
        np.asarray(extent, dtype=np.float32) * 0.5
    )


def summary(*, anchor_count=1, candidate_count=1):
    code_manifest = json.dumps(
        {"files": {"test": "9" * 64}, "schema": "test.v1"}, sort_keys=True
    )
    return TerminalObserverSummary(
        scene_id="48018894",
        anchor_count=anchor_count,
        candidate_count=candidate_count,
        near_candidate_count=candidate_count,
        represented_anchor_count=min(anchor_count, candidate_count),
        legacy_rule_selected_count=min(anchor_count, candidate_count),
        used_frame_count=2,
        point_count=10,
        model_runtime_s=0.01,
        source_anchor_prediction_sha256="1" * 64,
        active_anchor_scores_sha256="2" * 64,
        native_b6_diagnostic_sha256="3" * 64,
        native_b6_checkpoint_sha256="4" * 64,
        native_b6_manifest_sha256="5" * 64,
        source_points_sha256="6" * 64,
        checkpoint_sha256="7" * 64,
        config_sha256="8" * 64,
        code_manifest_sha256=hashlib.sha256(code_manifest.encode()).hexdigest(),
        adapter_mode="genuine",
        prefix_id="p100_gap20",
        device="cuda:0",
        pixel_stride=4,
        voxel_size_m=0.01,
        min_depth_m=0.10,
        max_depth_m=6.0,
        near_iou=0.15,
        score_threshold=0.01,
        max_proposals=256,
        materialized_active_verified=False,
    )


def code_manifest():
    return json.dumps(
        {"files": {"test": "9" * 64}, "schema": "test.v1"}, sort_keys=True
    )


def test_translation_normalization_roundtrips_boxes_to_ca_world():
    pose = np.eye(4, dtype=np.float64)
    pose[:3, 3] = [10.0, -3.0, 120.0]
    world_to_local = terminal_world_to_local(pose)
    boxes_local = np.asarray([[1.0, 2.0, 0.5, 2.0, 4.0, 1.0, 0.0]])
    result = aligned_boxes_to_world_corners(boxes_local, world_to_local)
    assert result.dtype == np.float32
    assert result.flags.c_contiguous
    assert np.allclose(result.mean(axis=1)[0], [11.0, -1.0, 120.5])
    assert np.allclose(np.ptp(result, axis=1)[0], [2.0, 4.0, 1.0])


def test_voxel_downsample_retains_first_acquisition_point():
    points = np.asarray(
        [
            [0.001, 0.001, 0.001, 1, 2, 3],
            [0.009, 0.009, 0.009, 4, 5, 6],
            [0.011, 0.001, 0.001, 7, 8, 9],
        ],
        dtype=np.float32,
    )
    output = voxel_downsample_first(points, 0.01)
    assert np.array_equal(output, points[[0, 2]])


def test_terminal_association_records_but_does_not_apply_legacy_rule():
    anchors = np.stack((corners([0, 0, 0], [1, 1, 1]),)).astype(np.float32)
    candidates = np.stack(
        (
            corners([0.05, 0, 0], [1, 1, 1]),
            corners([0.10, 0, 0], [1, 1, 1]),
            corners([5.0, 0, 0], [1, 1, 1]),
        )
    ).astype(np.float32)
    original = anchors.copy()
    result = associate_terminal_candidates(
        anchor_corners=anchors,
        anchor_scores=np.asarray([0.5], dtype=np.float32),
        candidate_corners=candidates,
        candidate_scores=np.asarray([0.7, 0.8, 0.95], dtype=np.float32),
    )
    assert np.array_equal(anchors, original)
    assert result.near_mask.tolist() == [True, True, False]
    assert result.represented_anchor_indices.tolist() == [0]
    assert result.legacy_rule_selected_candidate_rows.tolist() == [1]
    assert result.legacy_rule_selected_anchor_indices.tolist() == [0]


def test_observer_payload_and_writer_are_create_only(tmp_path):
    anchors = np.stack((corners([0, 0, 0], [1, 1, 1]),)).astype(np.float32)
    candidates = np.stack((corners([0.05, 0, 0], [1, 1, 1]),)).astype(np.float32)
    anchor_scores = np.asarray([0.5], dtype=np.float32)
    candidate_scores = np.asarray([0.8], dtype=np.float32)
    association = associate_terminal_candidates(
        anchor_corners=anchors,
        anchor_scores=anchor_scores,
        candidate_corners=candidates,
        candidate_scores=candidate_scores,
    )
    payload = observation_payload(
        summary=summary(),
        used_frame_ids=np.asarray([0, 20], dtype=np.int64),
        world_to_local=np.eye(4),
        anchor_corners=anchors,
        anchor_scores=anchor_scores,
        candidate_corners=candidates,
        candidate_scores=candidate_scores,
        candidate_point_count=np.asarray([12], dtype=np.int64),
        candidate_boxes_local=np.asarray(
            [[0.05, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0]], dtype=np.float32
        ),
        candidate_labels=np.asarray([0], dtype=np.int64),
        association=association,
        code_manifest_json=code_manifest(),
    )
    target = tmp_path / "48018894_ca1m_tr3d_terminal.npz"
    write_npz_create_only(target, payload)
    original = target.read_bytes()
    with np.load(target, allow_pickle=False) as archive:
        assert archive["schema"].item() == SCHEMA
        report = json.loads(archive["summary_json"].item())
        assert report["observer_only"] is True
        assert report["mutation_enabled"] is False
        assert report["ground_truth_access"] is False
        assert report["validation_policy_selection_authorized"] is False
        assert archive["active_anchor_scores_sha256"].item() == "2" * 64
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        write_npz_create_only(target, payload)
    assert target.read_bytes() == original


def test_numeric_scene_and_hash_contract_fail_closed():
    bad = summary()
    object.__setattr__(bad, "scene_id", "scene0000_00")
    anchors = np.stack((corners([0, 0, 0], [1, 1, 1]),)).astype(np.float32)
    association = associate_terminal_candidates(
        anchor_corners=anchors,
        anchor_scores=np.asarray([0.5], dtype=np.float32),
        candidate_corners=anchors,
        candidate_scores=np.asarray([0.8], dtype=np.float32),
    )
    with pytest.raises(ValueError, match="invalid CA-1M scene"):
        observation_payload(
            summary=bad,
            used_frame_ids=np.asarray([0, 20], dtype=np.int64),
            world_to_local=np.eye(4),
            anchor_corners=anchors,
            anchor_scores=np.asarray([0.5], dtype=np.float32),
            candidate_corners=anchors,
            candidate_scores=np.asarray([0.8], dtype=np.float32),
            candidate_point_count=np.asarray([1], dtype=np.int64),
            candidate_boxes_local=np.asarray(
                [[0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0]], dtype=np.float32
            ),
            candidate_labels=np.asarray([0], dtype=np.int64),
            association=association,
            code_manifest_json=code_manifest(),
        )
