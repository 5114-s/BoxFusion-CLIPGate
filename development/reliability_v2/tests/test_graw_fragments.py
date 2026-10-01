import dataclasses
from types import MappingProxyType

import numpy as np
import pytest

from boxfusion import graw_fragments as graw
from boxfusion import smov_fragments as smov


def scene_inputs(size=100, depth_value=2.0):
    depth = np.full((size, size), depth_value, dtype=np.float32)
    intrinsics = np.array(
        [[80.0, 0.0, size / 2], [0.0, 80.0, size / 2], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    pose = np.eye(4, dtype=np.float64)
    return depth, intrinsics, pose


def extract(**overrides):
    depth, intrinsics, pose = scene_inputs()
    values = dict(
        proposal_id=7,
        frame_id=3,
        score=0.9,
        box_xyxy=np.array([8.0, 8.0, 92.0, 92.0]),
        proposal_image_shape=(100, 100),
        proposal_to_depth_affine=np.eye(3),
        depth_m=depth,
        intrinsics=intrinsics,
        camera_to_world=pose,
    )
    values.update(overrides)
    return graw.extract_fragment(**values)


def prepare_raw(extractor, proposal_ids, scores, **overrides):
    depth, intrinsics, pose = scene_inputs()
    proposal_ids = np.asarray(proposal_ids, dtype=np.int64)
    values = dict(
        scene_id="scene",
        frame_id=0,
        proposal_ids=proposal_ids,
        boxes_xyxy=np.tile([8.0, 8.0, 92.0, 92.0], (len(proposal_ids), 1)),
        proposal_scores=np.asarray(scores, dtype=np.float64),
        proposal_image_shape=(100, 100),
        proposal_to_depth_affine=np.eye(3),
        depth_m=depth,
        intrinsics=intrinsics,
        camera_to_world=pose,
    )
    values.update(overrides)
    return extractor.prepare_keyframe(**values)


def prepare_clean(shadow, proposal_ids, scores, **overrides):
    depth, intrinsics, pose = scene_inputs()
    proposal_ids = np.asarray(proposal_ids, dtype=np.int64)
    values = dict(
        scene_id="scene",
        frame_id=0,
        proposal_ids=proposal_ids,
        boxes_xyxy=np.tile([8.0, 8.0, 92.0, 92.0], (len(proposal_ids), 1)),
        proposal_scores=np.asarray(scores, dtype=np.float64),
        proposal_image_shape=(100, 100),
        proposal_to_depth_affine=np.eye(3),
        depth_m=depth,
        intrinsics=intrinsics,
        camera_to_world=pose,
    )
    values.update(overrides)
    return shadow.prepare_keyframe(**values)


def test_raw_and_clean_proposal_membership_and_resource_caps_are_comparable():
    ids = np.arange(70, dtype=np.int64)
    # Include tied scores so the shared proposal-ID tie break is exercised.
    scores = np.repeat(np.arange(35, dtype=np.float64), 2)
    raw = prepare_raw(graw.RawFragmentExtractor(), ids, scores)
    clean_shadow = smov.FragmentShadow()
    clean = prepare_clean(clean_shadow, ids, scores)

    raw_membership = tuple(item.proposal_id for item in raw.diagnostics if item.selected)
    clean_membership = tuple(item.proposal_id for item in clean.diagnostics if item.selected)
    assert raw_membership == clean_membership
    assert len(raw_membership) == len(clean_membership) == 64
    assert set(raw.selected_proposal_ids) == set(raw_membership)
    expected_order = tuple(
        int(index) for index in np.lexsort((ids, -scores))[:64]
    )
    assert raw.selected_proposal_ids == expected_order
    assert raw.selected_proposal_ids[:4] == (68, 69, 66, 67)

    for name in (
        "pixel_stride",
        "max_rays_per_proposal",
        "min_depth_m",
        "max_depth_m",
        "min_fragment_points",
        "voxel_size_m",
        "max_points_per_view",
        "max_proposals_per_keyframe",
    ):
        assert graw.DEFAULT_CONFIG[name] == smov.DEFAULT_CONFIG[name]

    assert all(
        item.coverage.sampled_rays <= graw.DEFAULT_CONFIG["max_rays_per_proposal"]
        and item.coverage.output_voxels <= graw.DEFAULT_CONFIG["max_points_per_view"]
        for item in raw.diagnostics
        if item.accepted
    )
    assert all(
        item.coverage.sampled_rays <= smov.DEFAULT_CONFIG["max_rays_per_proposal"]
        and item.coverage.output_points <= smov.DEFAULT_CONFIG["max_points_per_view"]
        for item in clean.diagnostics
        if item.accepted
    )


def test_signed_floor_quantization_freezes_negative_boundary_semantics():
    tiny = np.nextafter(0.0, 1.0)
    points = np.asarray(
        [
            [-tiny, 0.0, 0.049999999],
            [0.0, -tiny, 0.05],
            [-0.05, -0.050000001, -0.0],
            [0.049999999, 0.050000001, -0.000000001],
        ],
        dtype=np.float64,
    )
    keys = graw._direct_voxel_keys(points, graw.VOXEL_SIZE_METERS)
    expected = np.unique(
        np.asarray(
            [
                [-1, 0, 0],
                [0, -1, 1],
                [-1, -2, 0],
                [0, 1, -1],
            ],
            dtype=np.int64,
        ),
        axis=0,
    )
    np.testing.assert_array_equal(keys, expected)
    assert keys.dtype == np.int64
    assert np.issubdtype(keys.dtype, np.signedinteger)


def test_fragment_stores_direct_readonly_integer_keys_without_float_points():
    result = extract()
    assert result.accepted
    fragment = result.fragment
    assert fragment.voxel_keys.dtype == np.int64
    assert fragment.voxels is fragment.voxel_keys
    assert not fragment.voxel_keys.flags.writeable
    assert "points_world" not in {field.name for field in dataclasses.fields(fragment)}
    with pytest.raises(ValueError):
        fragment.voxel_keys[0, 0] = 0


def test_raw_arm_keeps_all_valid_samples_without_center_or_jump_cleaning():
    depth, intrinsics, pose = scene_inputs(depth_value=4.0)
    depth[:, :50] = 2.0
    depth[48:53, 48:53] = np.nan
    raw = extract(depth_m=depth, intrinsics=intrinsics, camera_to_world=pose)
    clean = smov.extract_fragment(
        proposal_id=7,
        frame_id=3,
        score=0.9,
        box_xyxy=np.array([8.0, 8.0, 92.0, 92.0]),
        proposal_image_shape=(100, 100),
        proposal_to_depth_affine=np.eye(3),
        depth_m=depth,
        intrinsics=intrinsics,
        camera_to_world=pose,
    )

    assert raw.accepted
    assert {40, 80}.issubset(set(int(value) for value in raw.fragment.voxel_keys[:, 2]))
    assert not clean.accepted
    assert clean.reason == "center_seed_unusable"


def test_empty_and_invalid_inputs_abstain_or_validate_without_partial_output():
    assert extract(depth_m=np.empty((0, 5), dtype=np.float32)).reason == "invalid_depth_m"
    assert extract(depth_m=[[2.0]]).reason == "depth_m_must_be_numpy"
    assert extract(depth_m=np.full((100, 100), np.nan, dtype=np.float32)).reason == (
        "insufficient_valid_depth_pixels"
    )
    assert extract(box_xyxy=[20.0, 20.0, 20.0, 30.0]).reason == "empty_mapped_crop"
    assert extract(box_xyxy=["bad", 0, 10, 10]).reason == "invalid_box_xyxy"

    extractor = graw.RawFragmentExtractor()
    empty = prepare_raw(extractor, np.empty(0, dtype=np.int64), np.empty(0))
    assert empty.proposal_ids == ()
    assert empty.selected_proposal_ids == ()
    assert empty.diagnostics == ()

    with pytest.raises(ValueError, match="row aligned"):
        prepare_raw(
            extractor,
            [1, 2],
            [0.5, 0.6],
            boxes_xyxy=np.zeros((1, 4), dtype=np.float32),
        )
    with pytest.raises(ValueError, match="unique and nonnegative"):
        prepare_raw(extractor, [1, 1], [0.5, 0.6])

    failed = prepare_raw(
        extractor,
        [1, 2],
        [0.5, 0.6],
        depth_m=np.asarray([1.0, 2.0]),
    )
    assert all(item.selected and not item.accepted for item in failed.diagnostics)
    assert {item.reason for item in failed.diagnostics} == {"invalid_depth_m"}


def test_extraction_is_deterministic_and_does_not_modify_any_input():
    depth, intrinsics, pose = scene_inputs()
    depth[:, ::7] = 2.4
    pose[:3, 3] = [-0.051, 0.101, -0.001]
    affine = graw.aligned_resize_affine((200, 200), depth.shape)
    box = np.asarray([16.0, 16.0, 184.0, 184.0], dtype=np.float64)
    originals = [value.copy() for value in (depth, intrinsics, pose, affine, box)]
    kwargs = dict(
        box_xyxy=box,
        proposal_image_shape=(200, 200),
        proposal_to_depth_affine=affine,
        depth_m=depth,
        intrinsics=intrinsics,
        camera_to_world=pose,
    )

    first = extract(**kwargs)
    second = extract(**kwargs)
    assert first.accepted and second.accepted
    np.testing.assert_array_equal(first.fragment.voxel_keys, second.fragment.voxel_keys)
    assert first.fragment.coverage == second.fragment.coverage
    np.testing.assert_array_equal(first.fragment.crop_xyxy_depth, second.fragment.crop_xyxy_depth)
    for current, original in zip((depth, intrinsics, pose, affine, box), originals):
        np.testing.assert_array_equal(current, original)


def test_stride_and_voxel_caps_are_enforced_before_output():
    depth, intrinsics, pose = scene_inputs(size=400)
    intrinsics[:2, :2] = np.diag([400.0, 400.0])
    result = extract(
        box_xyxy=[0.0, 0.0, 399.0, 399.0],
        proposal_image_shape=(400, 400),
        depth_m=depth,
        intrinsics=intrinsics,
        camera_to_world=pose,
    )
    assert result.accepted
    assert result.coverage.effective_stride >= 4
    assert result.coverage.sampled_rays <= 1024
    assert result.coverage.output_voxels <= 512
    assert len(result.fragment.voxel_keys) == result.coverage.output_voxels


def test_config_is_frozen_bounded_and_immutable():
    with pytest.raises(ValueError, match="unknown"):
        graw.resolve_config({"mystery": 1})
    with pytest.raises(ValueError, match="frozen"):
        graw.resolve_config({"voxel_size_m": 0.10})
    with pytest.raises(ValueError, match="hard limits"):
        graw.resolve_config({"max_points_per_view": 513})
    with pytest.raises(ValueError, match="below min_fragment_points"):
        graw.resolve_config({"max_rays_per_proposal": 15})

    supplied = {"max_proposals_per_keyframe": 8}
    extractor = graw.RawFragmentExtractor(supplied)
    supplied["max_proposals_per_keyframe"] = 1
    assert extractor.config["max_proposals_per_keyframe"] == 8
    with pytest.raises(TypeError):
        extractor.config["max_proposals_per_keyframe"] = 1
    with pytest.raises(AttributeError, match="write-once"):
        extractor._config = {}
    with pytest.raises(AttributeError, match="cannot be deleted"):
        del extractor._config


def test_public_policy_alias_rebinding_cannot_change_executable_voxel_rule(monkeypatch):
    weakened = dict(graw.DEFAULT_CONFIG)
    weakened["voxel_size_m"] = 1.0
    weakened["max_proposals_per_keyframe"] = 1
    monkeypatch.setattr(graw, "VOXEL_SIZE_METERS", 1.0)
    monkeypatch.setattr(graw, "DEFAULT_CONFIG", MappingProxyType(weakened))

    effective = graw.resolve_config()
    assert effective["voxel_size_m"] == 0.05
    assert effective["max_proposals_per_keyframe"] == 64
    result = extract()
    assert result.accepted
    assert set(int(value) for value in result.fragment.voxel_keys[:, 2]) == {40}
