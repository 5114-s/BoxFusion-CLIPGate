import numpy as np
import pytest

from boxfusion.stream3dv3_trigger import (
    DepthResidualEventGate,
    DepthTriggerConfig,
    preselect_fastsam_masks,
)


def _camera():
    intrinsics = np.asarray(
        [[574.0, 0.0, 320.0], [0.0, 577.0, 240.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    return intrinsics, np.eye(4, dtype=np.float64)


def test_depth_trigger_is_query_before_commit_and_promotes_persistence():
    config = DepthTriggerConfig(
        sample_stride=16,
        confirmations=2,
        min_persistent_voxels=1,
        min_persistent_fraction=0.0,
        cooldown_keyframes=1,
        burst_keyframes=0,
        max_confirmed_voxels=5_000,
        max_tentative_voxels=5_000,
    )
    gate = DepthResidualEventGate(config)
    depth = np.full((480, 640), 2.0, dtype=np.float32)
    intrinsics, pose = _camera()

    first = gate.query(
        frame_id=0,
        frame_ordinal=0,
        depth_m=depth,
        intrinsics=intrinsics,
        camera_to_world=pose,
        native_boxes_xyxy=np.empty((0, 4)),
    )
    assert first.run_discovery and first.reason == "bootstrap"
    with pytest.raises(RuntimeError, match="not committed"):
        gate.query(
            frame_id=1,
            frame_ordinal=1,
            depth_m=depth,
            intrinsics=intrinsics,
            camera_to_world=pose,
            native_boxes_xyxy=np.empty((0, 4)),
        )
    with pytest.raises(ValueError, match="token"):
        gate.commit(first, token="0" * 64)
    gate.commit(first)

    second = gate.query(
        frame_id=1,
        frame_ordinal=1,
        depth_m=depth,
        intrinsics=intrinsics,
        camera_to_world=pose,
        native_boxes_xyxy=np.empty((0, 4)),
    )
    assert second.run_discovery
    assert second.reason == "persistent_novel_depth"
    assert second.persistent_voxels > 0
    gate.commit(second)

    third = gate.query(
        frame_id=2,
        frame_ordinal=2,
        depth_m=depth,
        intrinsics=intrinsics,
        camera_to_world=pose,
        native_boxes_xyxy=np.empty((0, 4)),
    )
    assert not third.run_discovery
    assert third.unknown_voxels == 0
    gate.commit(third)
    assert gate.summary()["confirmed_voxels"] <= 5_000


def test_mask_preselection_preserves_original_fastsam_indices():
    masks = np.zeros((4, 480, 640), dtype=np.bool_)
    masks[0, 100:180, 100:180] = True
    masks[1, 200:300, 200:300] = True
    masks[2, 300:380, 400:500] = True
    masks[3, 20:30, 20:30] = True  # below the frozen area/side limits
    boxes = np.asarray(
        [[100, 100, 179, 179], [200, 200, 299, 299], [400, 300, 499, 379], [20, 20, 29, 29]],
        dtype=np.float32,
    )
    confidences = np.asarray([0.60, 0.95, 0.80, 0.99], dtype=np.float32)
    depth = np.full((480, 640), 2.0, dtype=np.float32)
    native = np.asarray([[190, 190, 310, 310]], dtype=np.float32)

    result = preselect_fastsam_masks(
        masks=masks,
        confidences=confidences,
        boxes_xyxy=boxes,
        depth_m=depth,
        native_boxes_xyxy=native,
        box_shortlist=3,
        mask_cap=2,
    )
    assert result.input_count == 4
    assert result.original_indices.tolist() == [2, 0]
    assert 1 not in result.original_indices  # explained by the native box
    assert 3 not in result.original_indices
