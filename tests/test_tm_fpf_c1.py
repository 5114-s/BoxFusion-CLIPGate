import inspect

import numpy as np
import pytest

from boxfusion.tm_fpf_c1 import (
    PROTOCOL_ID,
    TMFPFC1,
    TMFPFC1ContractError,
    make_target_mask_view,
    match_fastsam_target_masks,
    resolve_tm_fpf_c1_config,
)


def _mapping(**overrides):
    values = {
        "enabled": True,
        "minimum_mask_pixels": 9,
        "minimum_face_observations": 2,
        "minimum_normalized_face_uncertainty": 0.01,
        "maximum_views": 3,
        "mask_erosion_pixels": 0,
        "min_views": 3,
        "max_ray_samples": 25,
        "min_valid_depth_samples": 9,
        "min_surface_rays": 4,
        "min_reference_rays": 4,
        "min_loss_improvement": 0.001,
        "min_face_visibility_cosine": 0.50,
    }
    values.update(overrides)
    return {"tm_fpf_c1": values}


def _view(config, *, source_id, frame_id, box, target_depth):
    mask = np.zeros((7, 7), dtype=bool)
    mask[1:6, 1:6] = True
    depth = np.zeros((7, 7), dtype=np.float32)
    depth[mask] = target_depth
    intrinsic = np.asarray(
        [[100.0, 0.0, 3.0], [0.0, 100.0, 3.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    snapshots = tuple(row.copy() for row in (mask, depth, intrinsic))
    result = make_target_mask_view(
        source_id=source_id,
        frame_id=frame_id,
        observation_box_xyzlhw=box,
        observation_rotation=np.eye(3),
        target_mask=mask,
        depth_m=depth,
        intrinsics=intrinsic,
        camera_to_world=np.eye(4),
        config=config,
    )
    for actual, expected in zip((mask, depth, intrinsic), snapshots):
        np.testing.assert_array_equal(actual, expected)
    return result


def _case(heldout_depths):
    route = TMFPFC1(_mapping())
    config = route.config
    anchor = np.asarray([[0.0, 0.0, 2.0, 2.0, 2.0, 2.0]], dtype=np.float32)
    rotations = np.eye(3, dtype=np.float32)[None]
    source_box = anchor[0].astype(np.float64).copy()
    # The source proposes z- = 0.93 instead of the native z- = 1.00.
    source_box[2] = 1.965
    source_box[5] = 2.07
    views = [
        _view(
            config,
            source_id="source-0",
            frame_id=0,
            box=source_box,
            target_depth=1.50,
        ),
        _view(
            config,
            source_id="source-1",
            frame_id=25,
            box=anchor[0],
            target_depth=heldout_depths[0],
        ),
        _view(
            config,
            source_id="source-2",
            frame_id=50,
            box=anchor[0],
            target_depth=heldout_depths[1],
        ),
    ]
    return route, anchor, rotations, views


def test_config_freezes_c1_and_has_no_oracle_surface():
    disabled = resolve_tm_fpf_c1_config({})
    assert disabled.enabled is False
    assert disabled.capf["max_accepted_faces"] == 1

    with pytest.raises(TMFPFC1ContractError, match="max_accepted_faces"):
        resolve_tm_fpf_c1_config(_mapping(max_accepted_faces=2))
    with pytest.raises(TMFPFC1ContractError, match="unknown"):
        resolve_tm_fpf_c1_config(_mapping(oracle_shadow=True))
    with pytest.raises(TMFPFC1ContractError, match="at least two held-out"):
        resolve_tm_fpf_c1_config(_mapping(min_views=2))
    with pytest.raises(TMFPFC1ContractError, match=r"in \[0,1\]"):
        resolve_tm_fpf_c1_config(_mapping(mask_match_min_containment=1.1))

    signature = inspect.signature(TMFPFC1.refine_terminal)
    assert tuple(signature.parameters) == (
        "self",
        "boxes_xyzlhw",
        "rotations",
        "scores",
        "track_views",
    )
    assert PROTOCOL_ID.endswith("HELDOUT-V1")


def test_fastsam_target_mask_match_is_one_to_one_deterministic_and_rejects_background():
    config = resolve_tm_fpf_c1_config(_mapping())
    native = np.asarray(
        [[1.0, 1.0, 5.0, 5.0], [6.0, 1.0, 10.0, 5.0]],
        dtype=np.float32,
    )
    masks = np.zeros((4, 12, 12), dtype=bool)
    masks[0, 1:5, 1:5] = True
    masks[1, 1:5, 6:10] = True
    masks[2] = True  # Obvious whole-image/background mask.
    masks[3, 7:11, 1:5] = True  # Residual region outside every native box.
    boxes = np.asarray(
        [
            [1.0, 1.0, 5.0, 5.0],
            [6.0, 1.0, 10.0, 5.0],
            [0.0, 0.0, 12.0, 12.0],
            [1.0, 7.0, 5.0, 11.0],
        ],
        dtype=np.float32,
    )
    confidences = np.asarray([0.8, 0.7, 0.99, 0.95], dtype=np.float32)
    snapshots = tuple(row.copy() for row in (native, masks, boxes, confidences))
    first = match_fastsam_target_masks(
        native_boxes_xyxy=native,
        automatic_masks=masks,
        automatic_boxes_xyxy=boxes,
        automatic_confidences=confidences,
        config=config,
    )
    second = match_fastsam_target_masks(
        native_boxes_xyxy=native,
        automatic_masks=masks,
        automatic_boxes_xyxy=boxes,
        automatic_confidences=confidences,
        config=config,
    )
    assert first == second == (0, 1)
    for actual, expected in zip((native, masks, boxes, confidences), snapshots):
        np.testing.assert_array_equal(actual, expected)

    # A single target mask cannot be reused by two native rows.
    duplicated_native = np.repeat(native[:1], 2, axis=0)
    one_to_one = match_fastsam_target_masks(
        native_boxes_xyxy=duplicated_native,
        automatic_masks=masks[:1],
        automatic_boxes_xyxy=boxes[:1],
        automatic_confidences=confidences[:1],
        config=config,
    )
    assert one_to_one == (0, None)

    no_target = match_fastsam_target_masks(
        native_boxes_xyxy=native,
        automatic_masks=masks[2:],
        automatic_boxes_xyxy=boxes[2:],
        automatic_confidences=confidences[2:],
        config=config,
    )
    assert no_target == (None, None)


def test_target_mask_builder_is_target_only_bounded_and_immutable():
    config = resolve_tm_fpf_c1_config(_mapping())
    box = np.asarray([0.0, 0.0, 2.0, 2.0, 2.0, 2.0])
    view = _view(
        config,
        source_id="masked",
        frame_id=3,
        box=box,
        target_depth=0.93,
    )
    assert view.evidence_kind == "target_mask_rgbd"
    assert view.target_mask_pixel_count == 25
    assert view.target_valid_depth_pixel_count == 25
    assert view.target_surface_valid.sum() == 25
    assert np.allclose(
        view.target_surface_points_world[view.target_surface_valid, 2], 0.93
    )
    assert not view.target_surface_points_world.flags.writeable
    assert not view.target_surface_valid.flags.writeable

    empty = np.zeros((7, 7), dtype=bool)
    with pytest.raises(TMFPFC1ContractError, match="too few pixels"):
        make_target_mask_view(
            source_id="empty",
            frame_id=0,
            observation_box_xyzlhw=box,
            observation_rotation=np.eye(3),
            target_mask=empty,
            depth_m=np.ones((7, 7)),
            intrinsics=np.eye(3),
            camera_to_world=np.eye(4),
            config=config,
        )


def test_good_heldout_target_masks_update_only_highest_uncertainty_face_once():
    route, anchor, rotations, views = _case((0.93, 0.93))
    scores = np.asarray([0.73125], dtype=np.float32)
    snapshots = tuple(row.copy() for row in (anchor, rotations, scores))
    result = route.refine_terminal(
        boxes_xyzlhw=anchor,
        rotations=rotations,
        scores=scores,
        track_views=[views],
    )
    for actual, expected in zip((anchor, rotations, scores), snapshots):
        np.testing.assert_array_equal(actual, expected)
    assert result.online_writeback is False
    assert result.accepted_count == 1
    decision = result.decisions[0]
    assert decision.accepted
    assert decision.face_index == 4
    assert decision.update is not None
    assert decision.update.heldout_views == (1, 2)
    assert decision.normalized_face_uncertainty == pytest.approx(0.0175)
    np.testing.assert_allclose(
        result.boxes_xyzlhw[0],
        [0.0, 0.0, 1.965, 2.0, 2.0, 2.07],
        atol=1e-6,
    )
    np.testing.assert_array_equal(result.rotations, rotations)
    np.testing.assert_array_equal(result.scores, scores)
    assert result.boxes_xyzlhw.dtype == anchor.dtype
    assert result.scores.dtype == scores.dtype


def test_bad_heldout_masks_and_missing_masks_roll_back_bit_exact():
    route, anchor, rotations, views = _case((1.00, 1.00))
    scores = np.asarray([0.8], dtype=np.float64)
    rejected = route.refine_terminal(
        boxes_xyzlhw=anchor,
        rotations=rotations,
        scores=scores,
        track_views=[views],
    )
    assert not rejected.decisions[0].accepted
    assert rejected.decisions[0].reason == "no_heldout_improvement"
    np.testing.assert_array_equal(rejected.boxes_xyzlhw, anchor)
    np.testing.assert_array_equal(rejected.rotations, rotations)
    np.testing.assert_array_equal(rejected.scores, scores)

    missing = route.refine_terminal(
        boxes_xyzlhw=anchor,
        rotations=rotations,
        scores=scores,
        track_views=[()],
    )
    assert missing.decisions[0].reason == "no_target_mask_evidence"
    np.testing.assert_array_equal(missing.boxes_xyzlhw, anchor)
    np.testing.assert_array_equal(missing.scores, scores)


def test_scores_never_change_geometry_decisions_order_or_count():
    route, anchor, rotations, views = _case((0.93, 0.93))
    low_scores = np.asarray([0.01], dtype=np.float32)
    high_scores = np.asarray([0.99], dtype=np.float32)
    low = route.refine_terminal(
        boxes_xyzlhw=anchor,
        rotations=rotations,
        scores=low_scores,
        track_views=[views],
    )
    high = route.refine_terminal(
        boxes_xyzlhw=anchor,
        rotations=rotations,
        scores=high_scores,
        track_views=[views],
    )
    np.testing.assert_array_equal(low.boxes_xyzlhw, high.boxes_xyzlhw)
    assert low.decisions == high.decisions
    assert low.boxes_xyzlhw.shape == anchor.shape
    assert high.boxes_xyzlhw.shape == anchor.shape
    np.testing.assert_array_equal(low.scores, low_scores)
    np.testing.assert_array_equal(high.scores, high_scores)
