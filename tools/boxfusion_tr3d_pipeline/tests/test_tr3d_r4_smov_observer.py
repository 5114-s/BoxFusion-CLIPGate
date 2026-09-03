from __future__ import annotations

import numpy as np
import pytest

from boxfusion.tr3d_r2_geometry import yaw_obb_corners_world
from boxfusion.tr3d_r2_observer import TR3DR2FrameBundle, TR3DR2ObserverConfig
from boxfusion.tr3d_r4_smov_cache import (
    load_r4_depth_sidecar,
    make_r4_depth_sidecar,
    write_r4_depth_sidecar,
)
from boxfusion.tr3d_r4_smov_observer import (
    corners_to_yaw_boxes,
    observe_r3_replacement_pairs,
)
from boxfusion.tr3d_r2b_observer import TR3DR2BFrameBundle
from boxfusion.tr3d_r4_smov_feature import (
    observe_r3_replacement_pair_features,
)


def _intrinsics(height: int, width: int, focal: float) -> np.ndarray:
    matrix = np.eye(4, dtype=np.float64)
    matrix[0, 0] = focal
    matrix[1, 1] = focal
    matrix[0, 2] = (width - 1) / 2.0
    matrix[1, 2] = (height - 1) / 2.0
    return matrix


def _bundle(frame_ids=(10, 20), shape=(9, 9)) -> TR3DR2FrameBundle:
    return TR3DR2FrameBundle(
        scene_id="scene0001_00",
        pose_source="scannet_g0_resolved_pose_v1",
        depth={int(frame): f"depth-{frame}" for frame in frame_ids},
        pose={int(frame): np.eye(4, dtype=np.float64) for frame in frame_ids},
        intrinsic_depth=_intrinsics(*shape, 8.0),
        extrinsic_depth=np.eye(4, dtype=np.float64),
    )


def _config(top_k=1, shape=(9, 9)) -> TR3DR2ObserverConfig:
    return TR3DR2ObserverConfig(
        image_shape=shape,
        pose_source="scannet_g0_resolved_pose_v1",
        top_k=top_k,
        pixel_stride=1,
        depth_scale=1.0,
        margin=0.0,
        min_depth=0.1,
        max_depth=8.0,
    )


def _manifest(frame_ids=(10, 20)):
    return {
        "scene_id": "scene0001_00",
        "used_frame_ids": list(frame_ids),
        "pose_source": "scannet_g0_resolved_pose_v1",
    }


def test_pair_uses_one_common_view_and_exposes_candidate_minus_anchor():
    calls: list[str] = []

    def decode(resource):
        calls.append(resource)
        return np.full((9, 9), 5.0, dtype=np.float32)

    result = observe_r3_replacement_pairs(
        anchor_boxes_world=np.asarray([[0, 0, 5, 20, 20, 2, 0]]),
        candidate_boxes_world=np.asarray([[0, 0, 4, 20, 20, 1, 0]]),
        proposal_ids=np.asarray([7], dtype=np.int64),
        anchor_indices=np.asarray([3], dtype=np.int64),
        prefix_manifest=_manifest(),
        frame_bundle=_bundle(),
        config=_config(),
        decode_depth=decode,
    )

    # Identical views tie; the stable smaller frame id is decoded once for
    # both roles.  Anchor supports z=5 while the shallower candidate predicts
    # empty/free space in front of that observed surface.
    np.testing.assert_array_equal(result.topk_frame_ids, [[10]])
    assert calls == ["depth-10"]
    assert result.aggregate_depth_counts[0, 0, 0] > 0
    assert result.aggregate_depth_counts[0, 1, 2] > 0
    assert result.candidate_minus_anchor_evidence[0, 0] < 0
    assert result.candidate_minus_anchor_evidence[0, 2] > 0
    assert not result.per_view_depth_counts.flags.writeable


def test_common_visibility_prevents_independent_topk_bias():
    result = observe_r3_replacement_pairs(
        anchor_boxes_world=np.asarray([[0, 0, 5, 2, 2, 2, 0]]),
        candidate_boxes_world=np.asarray([[0, 0, 5, 1, 1, 1, 0]]),
        proposal_ids=np.asarray([1], dtype=np.int64),
        anchor_indices=np.asarray([0], dtype=np.int64),
        prefix_manifest=_manifest(),
        frame_bundle=_bundle(),
        config=_config(top_k=2),
        decode_depth=lambda _: np.full((9, 9), 5.0, dtype=np.float32),
    )
    np.testing.assert_array_equal(result.topk_frame_ids, [[10, 20]])
    assert np.all(result.topk_view_valid)
    # Both role footprints are recorded on exactly those same frame slots.
    assert result.topk_projected_area_pixels.shape == (1, 2, 2)
    assert np.all(result.topk_projected_area_pixels > 0)


def test_invisible_pair_uses_zero_sentinels_and_does_not_decode():
    calls: list[object] = []
    result = observe_r3_replacement_pairs(
        anchor_boxes_world=np.asarray([[100, 0, 5, 1, 1, 1, 0]]),
        candidate_boxes_world=np.asarray([[100, 0, 5, 1, 1, 1, 0]]),
        proposal_ids=np.asarray([1], dtype=np.int64),
        anchor_indices=np.asarray([0], dtype=np.int64),
        prefix_manifest=_manifest(),
        frame_bundle=_bundle(),
        config=_config(top_k=2),
        decode_depth=lambda value: calls.append(value),
    )
    assert calls == []
    np.testing.assert_array_equal(result.topk_frame_ids, [[-1, -1]])
    assert not np.any(result.topk_view_valid)
    assert not np.any(result.aggregate_depth_counts)
    assert not np.any(result.candidate_minus_anchor_evidence)


@pytest.mark.parametrize("yaw", [0.0, 0.3, -1.2, np.pi / 2])
def test_unordered_corners_round_trip_to_equivalent_yaw_box(yaw):
    source = np.asarray([1.0, -2.0, 4.0, 3.0, 1.5, 2.0, yaw])
    corners = yaw_obb_corners_world(source)[[7, 1, 5, 0, 3, 6, 2, 4]]
    recovered = corners_to_yaw_boxes(corners[None])[0]
    recovered_corners = yaw_obb_corners_world(recovered)

    def rows(value):
        return np.asarray(sorted(map(tuple, np.round(value, 6))))

    np.testing.assert_allclose(rows(recovered_corners), rows(corners), atol=1e-5)


def test_sidecar_is_create_only_and_recomputes_redundant_evidence(tmp_path):
    observation = observe_r3_replacement_pairs(
        anchor_boxes_world=np.asarray([[0, 0, 5, 20, 20, 2, 0]]),
        candidate_boxes_world=np.asarray([[0, 0, 4, 20, 20, 1, 0]]),
        proposal_ids=np.asarray([7], dtype=np.int64),
        anchor_indices=np.asarray([3], dtype=np.int64),
        prefix_manifest=_manifest(frame_ids=(10,)),
        frame_bundle=_bundle(frame_ids=(10,)),
        config=_config(),
        decode_depth=lambda _: np.full((9, 9), 5.0, dtype=np.float32),
    )
    hashes = {
        "parent_cache_sha256": "1" * 64,
        "prefix_manifest_row_sha256": "2" * 64,
        "frame_artifact_tree_sha256": "3" * 64,
        "r3_diagnostic_sha256": "4" * 64,
        "input_geometry_sha256": "5" * 64,
        "input_scores_sha256": "6" * 64,
        "r4_config_sha256": "7" * 64,
        "r4_code_sha256": "8" * 64,
    }
    sidecar = make_r4_depth_sidecar(
        observation=observation,
        scene_id="scene0001_00",
        prefix_id="p100",
        final_source_timestamp=10,
        tr3d_scores=np.asarray([0.9]),
        anchor_scores=np.asarray([0.7]),
        anchor_iou=np.asarray([0.4]),
        anchor_boxes_world=np.asarray([[0, 0, 5, 20, 20, 2, 0]]),
        candidate_boxes_world=np.asarray([[0, 0, 4, 20, 20, 1, 0]]),
        **hashes,
    )
    path = tmp_path / "scene0001_00.r4d.npz"
    write_r4_depth_sidecar(path, sidecar)
    loaded = load_r4_depth_sidecar(path)
    np.testing.assert_array_equal(
        loaded.candidate_minus_anchor_evidence,
        sidecar.candidate_minus_anchor_evidence,
    )
    with pytest.raises(FileExistsError, match="immutable"):
        write_r4_depth_sidecar(path, sidecar)

    # A separately decoded, tampered file cannot exploit redundant fractions.
    with np.load(path, allow_pickle=False) as archive:
        values = {name: np.array(archive[name], copy=True) for name in archive.files}
    values["candidate_minus_anchor_evidence"][0, 0] += 0.1
    tampered = tmp_path / "tampered.npz"
    np.savez(tampered, **values)
    with pytest.raises(ValueError, match="delta disagrees"):
        load_r4_depth_sidecar(tampered)


def test_paired_feature_observer_encodes_each_shared_frame_once():
    calls: list[int] = []
    frames = (10, 20)
    frame_bundle = TR3DR2BFrameBundle(
        scene_id="scene0001_00",
        pose_source="scannet_g0_resolved_pose_v1",
        color={frame: np.full((9, 9, 3), frame, dtype=np.uint8) for frame in frames},
        depth={frame: np.full((9, 9), 5.0, dtype=np.float32) for frame in frames},
        pose={frame: np.eye(4, dtype=np.float64) for frame in frames},
        intrinsic_depth=_intrinsics(9, 9, 8.0),
        intrinsic_color=_intrinsics(9, 9, 8.0),
        extrinsic_depth=np.eye(4, dtype=np.float64),
        extrinsic_color=np.eye(4, dtype=np.float64),
    )

    def encode(rgb):
        calls.append(int(rgb[0, 0, 0]))
        # Spatially varying cells ensure that support-mask pooling is active.
        return np.asarray(
            [[[1.0, 0.0], [0.0, 1.0]], [[0.0, 1.0], [1.0, 0.0]]],
            dtype=np.float32,
        )

    observed = observe_r3_replacement_pair_features(
        anchor_boxes_world=np.asarray([[0, 0, 5, 2, 2, 2, 0]]),
        candidate_boxes_world=np.asarray([[0, 0, 5, 2, 2, 2, 0]]),
        proposal_ids=np.asarray([7], dtype=np.int64),
        anchor_indices=np.asarray([3], dtype=np.int64),
        topk_frame_ids=np.asarray([[10, 20]], dtype=np.int64),
        topk_view_valid=np.asarray([[True, True]], dtype=np.bool_),
        frame_bundle=frame_bundle,
        depth_config=_config(top_k=2),
        encode_rgb=encode,
        min_support_points=1,
    )
    assert calls == [10, 20]
    assert observed.feature_view_valid.shape == (1, 2, 2)
    assert observed.per_view_features.shape[:3] == (1, 2, 2)
    np.testing.assert_allclose(observed.candidate_minus_anchor_statistics, 0.0)
    assert not observed.per_view_features.flags.writeable
