from __future__ import annotations

import numpy as np
import pytest

from boxfusion.tr3d_r2_observer import (
    R2_DEPTH_CLASS_NAMES,
    TR3DR2FrameBundle,
    TR3DR2ObserverConfig,
    observe_tr3d_r2_scene,
)


def _intrinsics(height: int, width: int, focal: float) -> np.ndarray:
    matrix = np.eye(4, dtype=np.float64)
    matrix[0, 0] = focal
    matrix[1, 1] = focal
    matrix[0, 2] = (width - 1) / 2.0
    matrix[1, 2] = (height - 1) / 2.0
    return matrix


def _pose(camera_world_z: float = 0.0) -> np.ndarray:
    result = np.eye(4, dtype=np.float64)
    result[2, 3] = camera_world_z
    return result


def _bundle(
    *,
    depth: dict[int, object],
    pose: dict[int, object],
    image_shape: tuple[int, int] = (9, 9),
    focal: float = 8.0,
) -> TR3DR2FrameBundle:
    return TR3DR2FrameBundle(
        scene_id="scene0001_00",
        pose_source="scannet_sens_pose",
        depth=depth,
        pose=pose,
        intrinsic_depth=_intrinsics(*image_shape, focal),
        extrinsic_depth=np.eye(4, dtype=np.float64),
    )


def _config(
    *, image_shape: tuple[int, int] = (9, 9), top_k: int = 2
) -> TR3DR2ObserverConfig:
    return TR3DR2ObserverConfig(
        image_shape=image_shape,
        pose_source="scannet_sens_pose",
        top_k=top_k,
        pixel_stride=1,
        depth_scale=1.0,
        margin=0.0,
        min_depth=0.1,
        max_depth=8.0,
    )


def _manifest(frame_ids: list[int]) -> dict[str, object]:
    return {
        "scene_id": "scene0001_00",
        "used_frame_ids": frame_ids,
        "pose_source": "scannet_sens_pose",
    }


def test_projection_first_stable_topk_decodes_only_union_of_selected_views():
    resources = {10: "depth-10", 20: "depth-20", 30: "depth-30"}
    poses = {10: _pose(), 20: _pose(), 30: _pose(2.0)}
    calls: list[str] = []

    def decode(resource: object) -> np.ndarray:
        assert isinstance(resource, str)
        calls.append(resource)
        value = 3.0 if resource == "depth-30" else 5.0
        return np.full((9, 9), value, dtype=np.float32)

    boxes = np.asarray(
        [[0, 0, 5, 2, 2, 2, 0], [0, 0, 5, 2, 2, 2, 0]],
        dtype=np.float32,
    )
    boxes_before = boxes.copy()
    proposal_ids = np.asarray([7, 9], dtype=np.int64)
    ids_before = proposal_ids.copy()
    result = observe_tr3d_r2_scene(
        boxes_world=boxes,
        proposal_ids=proposal_ids,
        prefix_manifest=_manifest([10, 20, 30]),
        frame_bundle=_bundle(depth=resources, pose=poses),
        config=_config(),
        decode_depth=decode,
    )

    # Frame 30 has the largest projection. Frames 10 and 20 tie, so the
    # smaller manifest frame id wins deterministically.
    np.testing.assert_array_equal(result.topk_frame_ids, [[30, 10], [30, 10]])
    assert calls == ["depth-30", "depth-10"]
    np.testing.assert_array_equal(result.decoded_frame_ids, [30, 10])
    assert "depth-20" not in calls
    assert np.all(result.topk_view_valid)
    assert np.all(result.topk_projected_area_pixels[:, 0] > result.topk_projected_area_pixels[:, 1])
    np.testing.assert_array_equal(boxes, boxes_before)
    np.testing.assert_array_equal(proposal_ids, ids_before)
    assert not result.per_view_depth_counts.flags.writeable
    assert not result.proposal_ids.flags.writeable


def test_explicit_pixel_counts_and_fractions_align_with_cache_contract():
    depth = np.asarray([[5.0, 3.0], [7.0, np.nan]], dtype=np.float32)
    result = observe_tr3d_r2_scene(
        boxes_world=np.asarray([[0, 0, 5, 20, 20, 2, 0]], dtype=np.float32),
        proposal_ids=np.asarray([11], dtype=np.int64),
        prefix_manifest=_manifest([10]),
        frame_bundle=_bundle(
            depth={10: "depth"},
            pose={10: _pose()},
            image_shape=(2, 2),
            focal=1.0,
        ),
        config=_config(image_shape=(2, 2), top_k=1),
        decode_depth=lambda _: depth,
    )

    assert R2_DEPTH_CLASS_NAMES == (
        "support",
        "occluded",
        "free_space",
        "invalid",
    )
    np.testing.assert_array_equal(result.per_view_depth_counts, [[[1, 1, 1, 1]]])
    np.testing.assert_array_equal(result.per_view_point_count, [[4]])
    np.testing.assert_allclose(result.per_view_depth_evidence, [[[0.25] * 4]])
    np.testing.assert_array_equal(result.aggregate_depth_counts, [[1, 1, 1, 1]])
    np.testing.assert_array_equal(result.aggregate_point_count, [4])
    np.testing.assert_allclose(result.aggregate_depth_evidence, [[0.25] * 4])
    np.testing.assert_array_equal(result.aggregate_view_count, [1])


def test_no_visible_projection_decodes_nothing_and_uses_cache_sentinels():
    calls: list[object] = []
    result = observe_tr3d_r2_scene(
        boxes_world=np.asarray([[100, 0, 5, 1, 1, 1, 0]], dtype=np.float32),
        proposal_ids=np.asarray([3], dtype=np.int64),
        prefix_manifest=_manifest([10, 20]),
        frame_bundle=_bundle(
            depth={10: "a", 20: "b"},
            pose={10: _pose(), 20: _pose()},
        ),
        config=_config(top_k=3),
        decode_depth=lambda value: calls.append(value),
    )
    assert calls == []
    np.testing.assert_array_equal(result.topk_frame_ids, [[-1, -1, -1]])
    assert not np.any(result.topk_view_valid)
    assert not np.any(result.per_view_depth_counts)
    assert not np.any(result.per_view_depth_evidence)
    np.testing.assert_array_equal(result.aggregate_view_count, [0])
    np.testing.assert_array_equal(result.aggregate_point_count, [0])


@pytest.mark.parametrize(
    ("manifest", "message"),
    [
        ({"pose_source": "scannet_sens_pose"}, "requires used_frame_ids"),
        (
            {"used_frame_ids": [10], "pose_source": "other"},
            "pose_source provenance mismatch",
        ),
        (
            {"used_frame_ids": [20, 10], "pose_source": "scannet_sens_pose"},
            "strictly increasing",
        ),
        (
            {"used_frame_ids": [10, 10], "pose_source": "scannet_sens_pose"},
            "strictly increasing",
        ),
        (
            {"used_frame_ids": [99], "pose_source": "scannet_sens_pose"},
            "absent from frame bundle",
        ),
    ],
)
def test_strict_manifest_and_frame_bounds_fail_closed(manifest, message):
    with pytest.raises(ValueError, match=message):
        observe_tr3d_r2_scene(
            boxes_world=np.asarray([[0, 0, 5, 1, 1, 1, 0]]),
            proposal_ids=np.asarray([0], dtype=np.int64),
            prefix_manifest=manifest,
            frame_bundle=_bundle(
                depth={10: "depth", 20: "depth"},
                pose={10: _pose(), 20: _pose()},
            ),
            config=_config(),
            decode_depth=lambda _: np.ones((9, 9)),
        )


def test_missing_pose_nonfinite_pose_and_nonfinite_parent_fail_closed():
    common = dict(
        boxes_world=np.asarray([[0, 0, 5, 1, 1, 1, 0]]),
        proposal_ids=np.asarray([0], dtype=np.int64),
        prefix_manifest=_manifest([10]),
        config=_config(),
        decode_depth=lambda _: np.ones((9, 9)),
    )
    with pytest.raises(ValueError, match="missing_pose"):
        observe_tr3d_r2_scene(
            **common,
            frame_bundle=_bundle(depth={10: "depth"}, pose={}),
        )

    nonfinite_pose = _pose()
    nonfinite_pose[0, 3] = np.nan
    with pytest.raises(ValueError, match="pose decode failed"):
        observe_tr3d_r2_scene(
            **common,
            frame_bundle=_bundle(
                depth={10: "depth"}, pose={10: nonfinite_pose}
            ),
        )

    bad_boxes = np.asarray([[0, 0, np.nan, 1, 1, 1, 0]])
    with pytest.raises(ValueError, match="finite"):
        observe_tr3d_r2_scene(
            **{**common, "boxes_world": bad_boxes},
            frame_bundle=_bundle(
                depth={10: "depth"}, pose={10: _pose()}
            ),
        )


def test_bad_calibration_depth_shape_and_proposal_ids_fail_closed():
    base_bundle = _bundle(depth={10: "depth"}, pose={10: _pose()})
    bad_intrinsic = np.asarray(base_bundle.intrinsic_depth).copy()
    bad_intrinsic[0, 0] = np.nan
    with pytest.raises(ValueError, match="intrinsic_depth"):
        observe_tr3d_r2_scene(
            boxes_world=np.asarray([[0, 0, 5, 1, 1, 1, 0]]),
            proposal_ids=np.asarray([0], dtype=np.int64),
            prefix_manifest=_manifest([10]),
            frame_bundle=TR3DR2FrameBundle(
                scene_id=base_bundle.scene_id,
                pose_source=base_bundle.pose_source,
                depth=base_bundle.depth,
                pose=base_bundle.pose,
                intrinsic_depth=bad_intrinsic,
                extrinsic_depth=base_bundle.extrinsic_depth,
            ),
            config=_config(),
            decode_depth=lambda _: np.ones((9, 9)),
        )

    with pytest.raises(ValueError, match="depth shape"):
        observe_tr3d_r2_scene(
            boxes_world=np.asarray([[0, 0, 5, 1, 1, 1, 0]]),
            proposal_ids=np.asarray([0], dtype=np.int64),
            prefix_manifest=_manifest([10]),
            frame_bundle=base_bundle,
            config=_config(),
            decode_depth=lambda _: np.ones((8, 9)),
        )

    with pytest.raises(ValueError, match="unique"):
        observe_tr3d_r2_scene(
            boxes_world=np.asarray(
                [[0, 0, 5, 1, 1, 1, 0], [0, 0, 6, 1, 1, 1, 0]]
            ),
            proposal_ids=np.asarray([0, 0], dtype=np.int64),
            prefix_manifest=_manifest([10]),
            frame_bundle=base_bundle,
            config=_config(),
            decode_depth=lambda _: np.ones((9, 9)),
        )
