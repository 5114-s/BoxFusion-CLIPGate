import inspect
import json
from collections.abc import Sequence
from types import MappingProxyType

import numpy as np
import pytest

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
    return smov.extract_fragment(**values)


def prepare(shadow, frame_id, *, proposal_ids=(1,), boxes=None, scores=None, **overrides):
    depth, intrinsics, pose = scene_inputs()
    proposal_ids = np.asarray(proposal_ids, dtype=np.int64)
    if boxes is None:
        boxes = np.tile([8.0, 8.0, 92.0, 92.0], (len(proposal_ids), 1))
    if scores is None:
        scores = np.linspace(0.5, 0.9, len(proposal_ids))
    values = dict(
        scene_id="scene",
        frame_id=frame_id,
        proposal_ids=proposal_ids,
        boxes_xyxy=boxes,
        proposal_scores=scores,
        proposal_image_shape=(100, 100),
        proposal_to_depth_affine=np.eye(3),
        depth_m=depth,
        intrinsics=intrinsics,
        camera_to_world=pose,
    )
    values.update(overrides)
    return shadow.prepare_keyframe(**values)


def test_rgb_box_mapping_backprojection_pose_and_no_input_mutation():
    depth, intrinsics, pose = scene_inputs()
    pose[:3, 3] = [1.0, 2.0, 3.0]
    box = np.array([20.0, 20.0, 180.0, 180.0])
    copies = [value.copy() for value in (depth, intrinsics, pose, box)]

    result = extract(
        box_xyxy=box,
        proposal_image_shape=(200, 200),
        proposal_to_depth_affine=smov.aligned_resize_affine(
            (200, 200), depth.shape
        ),
        depth_m=depth,
        intrinsics=intrinsics,
        camera_to_world=pose,
    )

    assert result.accepted
    np.testing.assert_allclose(result.fragment.crop_xyxy_depth, [10, 10, 90, 90])
    assert np.any(np.all(np.isclose(result.fragment.points_world, [1, 2, 5]), axis=1))
    for original, copied in zip((depth, intrinsics, pose, box), copies):
        np.testing.assert_array_equal(original, copied)
    for array in (
        result.fragment.crop_xyxy_depth,
        result.fragment.proposal_to_depth_affine,
        result.fragment.intrinsics,
        result.fragment.camera_to_world,
        result.fragment.points_world,
    ):
        assert not array.flags.writeable


def test_stride_starts_at_four_and_adapts_to_ray_budget():
    depth, intrinsics, pose = scene_inputs(size=400)
    intrinsics[:2, :2] = np.diag([400.0, 400.0])
    result = extract(
        box_xyxy=[0, 0, 399, 399],
        proposal_image_shape=(400, 400),
        depth_m=depth,
        intrinsics=intrinsics,
        camera_to_world=pose,
    )
    assert result.accepted
    assert result.coverage.effective_stride >= 4
    assert result.coverage.sampled_rays <= 1024
    assert result.coverage.output_points <= 512


def test_clean_fragment_exposes_direct_readonly_signed_voxel_keys():
    result = extract()
    assert result.accepted
    fragment = result.fragment
    assert fragment.voxel_keys.dtype == np.int64
    assert np.issubdtype(fragment.voxel_keys.dtype, np.signedinteger)
    assert fragment.voxels is fragment.voxel_keys
    assert not fragment.voxel_keys.flags.writeable
    assert len(fragment.voxel_keys) == len(fragment.points_world)
    assert result.coverage.unique_voxels >= result.coverage.output_voxels
    assert result.coverage.output_voxels == result.coverage.output_points
    assert result.coverage.output_voxels == len(fragment.voxel_keys)
    with pytest.raises(ValueError):
        fragment.voxel_keys[0, 0] = 0


def test_signed_floor_quantization_freezes_negative_half_open_boundaries():
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
    keys = smov._direct_voxel_keys(points, smov.VOXEL_SIZE_METERS)
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


def test_four_neighbour_edges_and_center_component_reject_background():
    depth, intrinsics, pose = scene_inputs(depth_value=4.0)
    depth[24:76, 24:76] = 2.0
    result = extract(depth_m=depth, intrinsics=intrinsics, camera_to_world=pose)
    assert result.accepted
    assert result.coverage.edge_pixels > 0
    assert result.coverage.component_pixels >= 16
    np.testing.assert_allclose(result.fragment.points_world[:, 2], 2.0)


@pytest.mark.parametrize(
    ("depth_value", "reason"),
    [(0.09, "center_seed_unusable"), (8.01, "center_seed_unusable"), (np.nan, "center_seed_unusable")],
)
def test_depth_domain_abstains(depth_value, reason):
    result = extract(depth_m=np.full((100, 100), depth_value, dtype=np.float32))
    assert not result.accepted
    assert result.reason == reason
    assert result.coverage.valid_depth_ratio == 0.0


def test_center_seed_is_required_even_when_crop_has_valid_depth():
    depth, _, _ = scene_inputs()
    depth[46:55, 46:55] = 0.0
    result = extract(depth_m=depth)
    assert not result.accepted
    assert result.reason == "center_seed_unusable"


def test_post_voxel_minimum_causes_abstention():
    depth, intrinsics, pose = scene_inputs()
    intrinsics[0, 0] = intrinsics[1, 1] = 100000.0
    result = extract(depth_m=depth, intrinsics=intrinsics, camera_to_world=pose)
    assert not result.accepted
    assert result.reason == "insufficient_points_after_voxel"
    assert result.coverage.output_points < 16


def test_prepare_is_fail_open_per_proposal_and_globally():
    shadow = smov.FragmentShadow()
    boxes = np.array([[8, 8, 92, 92], [50, 50, 50, 60]], dtype=np.float64)
    batch = prepare(shadow, 0, proposal_ids=(10, 11), boxes=boxes)
    assert batch.diagnostics[0].accepted
    assert batch.diagnostics[1].reason == "empty_mapped_crop"
    shadow.commit_keyframe(batch, track_ids=(100, 101))

    invalid = prepare(shadow, 1, depth_m=np.array([1.0, 2.0]))
    assert invalid.diagnostics[0].reason == "invalid_depth_m"
    shadow.commit_keyframe(invalid, track_ids=(100,))
    failures = shadow.diagnostics()["failure_reasons"]
    assert failures["empty_mapped_crop"] == 1
    assert failures["invalid_depth_m"] == 1


def test_bad_numeric_conversion_is_also_fail_open():
    result = extract(box_xyxy=["bad", 0, 10, 10])
    assert result.reason == "invalid_box_xyxy"
    invalid_pose = extract(camera_to_world=[["bad"]])
    assert invalid_pose.reason == "invalid_camera_to_world"


def test_top_score_proposal_cap_is_64_and_all_rows_are_diagnosed():
    shadow = smov.FragmentShadow()
    ids = np.arange(70, dtype=np.int64)
    batch = prepare(shadow, 0, proposal_ids=ids, scores=np.arange(70, dtype=np.float64))
    assert len(batch.diagnostics) == 70
    assert sum(item.selected for item in batch.diagnostics) == 64
    assert [item.proposal_id for item in batch.diagnostics if not item.selected] == list(range(6))
    assert all(item.reason == "proposal_cap" for item in batch.diagnostics[:6])


def test_strict_configuration_rejects_unknown_types_geometry_and_relaxed_caps():
    with pytest.raises(ValueError, match="unknown"):
        smov.resolve_config({"mystery": 1})
    with pytest.raises(ValueError, match="integer"):
        smov.resolve_config({"max_tracks": True})
    with pytest.raises(ValueError, match="frozen"):
        smov.resolve_config({"pixel_stride": 5})
    with pytest.raises(ValueError, match="hard limits"):
        smov.resolve_config({"max_tracks": 1025})
    with pytest.raises(ValueError, match="max_rays_per_proposal"):
        smov.resolve_config({"max_rays_per_proposal": 15})
    reduced = smov.resolve_config({"max_tracks": 1, "max_views_per_track": 2})
    assert reduced["max_tracks"] == 1


def test_public_default_resolved_and_instance_configs_are_immutable():
    supplied = {"max_tracks": 2}
    resolved = smov.resolve_config(supplied)
    shadow = smov.FragmentShadow(supplied)
    supplied["max_tracks"] = 1

    assert resolved["max_tracks"] == 2
    assert shadow.config["max_tracks"] == 2
    for config in (smov.DEFAULT_CONFIG, resolved, shadow.config):
        with pytest.raises(TypeError):
            config["max_tracks"] = 9999
    with pytest.raises(AttributeError):
        shadow.config = {"max_tracks": 9999}
    with pytest.raises(AttributeError, match="write-once"):
        shadow._config = {"max_tracks": 9999}
    with pytest.raises(AttributeError, match="cannot be deleted"):
        del shadow._config
    assert shadow.config["max_tracks"] == 2


def test_batch_extractor_has_no_pending_track_or_scene_state():
    extractor = smov.SMOVFragmentExtractor()
    first = prepare(extractor, 0)
    repeated = prepare(extractor, 0)
    other_scene = prepare(extractor, 0, scene_id="another_scene")

    assert first.scene_id == repeated.scene_id == "scene"
    assert other_scene.scene_id == "another_scene"
    assert set(vars(extractor)) == {"_config"}
    assert not ({"_pending", "_tracks", "_scene_id", "_last_prepared_frame"} & set(vars(extractor)))
    for left, right in zip(first.diagnostics, repeated.diagnostics):
        assert left.proposal_id == right.proposal_id
        assert left.selected == right.selected
        assert left.reason == right.reason
        assert left.coverage == right.coverage
        np.testing.assert_array_equal(
            left.fragment.voxel_keys, right.fragment.voxel_keys
        )

    with pytest.raises(TypeError):
        extractor.config["max_proposals_per_keyframe"] = 1
    with pytest.raises(AttributeError, match="write-once"):
        extractor._config = {}
    with pytest.raises(AttributeError, match="cannot be deleted"):
        del extractor._config


def test_batch_extractor_and_track_shadow_share_exact_fragment_logic():
    extractor = smov.SMOVFragmentExtractor()
    shadow = smov.FragmentShadow()
    clean_batch = prepare(extractor, 7, proposal_ids=(5, 6), scores=(0.4, 0.8))
    shadow_batch = prepare(shadow, 7, proposal_ids=(5, 6), scores=(0.4, 0.8))

    assert clean_batch.proposal_ids == shadow_batch.proposal_ids
    for direct, legacy in zip(clean_batch.diagnostics, shadow_batch.diagnostics):
        assert direct.proposal_id == legacy.proposal_id
        assert direct.selected == legacy.selected
        assert direct.reason == legacy.reason
        assert direct.coverage == legacy.coverage
        np.testing.assert_array_equal(
            direct.fragment.voxel_keys, legacy.fragment.voxel_keys
        )
        np.testing.assert_array_equal(
            direct.fragment.points_world, legacy.fragment.points_world
        )


def test_batch_extractor_fail_opens_global_numeric_failure(monkeypatch):
    extractor = smov.SMOVFragmentExtractor()

    def fail_edge_mask(*args, **kwargs):
        raise FloatingPointError("injected")

    monkeypatch.setattr(smov, "_full_resolution_edge_mask", fail_edge_mask)
    batch = prepare(extractor, 0, proposal_ids=(1, 2))
    assert len(batch.diagnostics) == 2
    assert all(item.selected and not item.accepted for item in batch.diagnostics)
    assert {item.reason for item in batch.diagnostics} == {"numeric_failure"}


def test_smov_batch_record_and_scene_writer_are_json_ready_and_atomic(tmp_path):
    batch = prepare(
        smov.SMOVFragmentExtractor(),
        12,
        proposal_ids=(3, 4),
        boxes=np.asarray([[8, 8, 92, 92], [50, 50, 50, 60]]),
    )
    record = smov.smov_batch_to_dict(batch)
    assert record == smov.smov_batch_to_dict(batch)
    assert record["frame_id"] == 12
    assert record["selected_count"] == 2
    assert record["accepted_count"] == 1
    assert record["abstained_count"] == 1
    assert record["failure_reasons"] == {"empty_mapped_crop": 1}
    assert record["voxel_statistics"]["output"]["total"] > 0
    assert len(record["proposals"]) == 2

    destination = tmp_path / "scene.smov_shadow.json"
    returned = smov.write_smov_shadow_diagnostics(
        destination,
        "scene",
        [record],
        {"accepted": np.int64(1)},
        True,
    )
    assert returned == str(destination.resolve())
    decoded = json.loads(destination.read_text(encoding="utf-8"))
    assert decoded["schema"] == smov.SCHEMA
    assert decoded["scene_id"] == "scene"
    assert decoded["trace_valid"] is True
    assert decoded["frame_count"] == 1
    assert decoded["frames"] == [record]
    assert decoded["summary"] == {"accepted": 1}
    assert not list(tmp_path.glob("*.tmp"))


def test_smov_scene_writer_preserves_old_file_on_replace_failure(tmp_path, monkeypatch):
    batch = prepare(smov.SMOVFragmentExtractor(), 0)
    record = smov.smov_batch_to_dict(batch)
    destination = tmp_path / "scene.smov_shadow.json"
    destination.write_text("old", encoding="utf-8")

    def fail_replace(*args, **kwargs):
        raise OSError("injected replace failure")

    monkeypatch.setattr(smov.os, "replace", fail_replace)
    with pytest.raises(OSError, match="injected"):
        smov.write_smov_shadow_diagnostics(
            destination, "scene", [record], {}, True
        )
    assert destination.read_text(encoding="utf-8") == "old"
    assert not list(tmp_path.glob(".*.tmp"))


def test_smov_scene_writer_enforces_finite_json_and_32_mib_cap(tmp_path, monkeypatch):
    destination = tmp_path / "scene.smov_shadow.json"
    with pytest.raises(ValueError, match="JSON compliant|range"):
        smov.write_smov_shadow_diagnostics(
            destination, "scene", [], {"bad": np.nan}, True
        )
    assert not destination.exists()

    monkeypatch.setattr(smov, "_F_MAX_DIAGNOSTIC_BYTES", 64)
    with pytest.raises(ValueError, match="32 MiB cap"):
        smov.write_smov_shadow_diagnostics(
            destination, "scene", [], {"payload": "x" * 128}, True
        )
    assert not destination.exists()


def test_public_policy_alias_rebinding_cannot_weaken_executable_rules(monkeypatch):
    weakened = dict(smov.DEFAULT_CONFIG)
    weakened["voxel_size_m"] = 1.0
    weakened["max_proposals_per_keyframe"] = 1
    monkeypatch.setattr(smov, "VOXEL_SIZE_METERS", 1.0)
    monkeypatch.setattr(smov, "MAX_INPUT_PROPOSALS", 100_000)
    monkeypatch.setattr(smov, "MAX_INPUT_DEPTH_PIXELS", 100_000_000)
    monkeypatch.setattr(smov, "DEFAULT_CONFIG", MappingProxyType(weakened))

    effective = smov.resolve_config()
    assert effective["voxel_size_m"] == 0.05
    assert effective["max_proposals_per_keyframe"] == 64
    result = extract()
    assert result.accepted
    assert set(int(value) for value in result.fragment.voxel_keys[:, 2]) == {40}


def test_input_proposal_cap_precedes_numpy_conversion_and_sort(monkeypatch):
    count = smov.MAX_INPUT_PROPOSALS + 1
    ids = np.arange(count, dtype=np.int64)
    boxes = np.zeros((count, 4), dtype=np.float32)
    scores = np.zeros(count, dtype=np.float32)
    depth, intrinsics, pose = scene_inputs()
    affine = np.eye(3)
    shadow = smov.FragmentShadow()

    def forbidden(*args, **kwargs):
        raise AssertionError("unbounded input reached NumPy conversion")

    monkeypatch.setattr(smov.np, "asarray", forbidden)
    monkeypatch.setattr(smov.np, "lexsort", forbidden)
    with pytest.raises(ValueError, match="hard input proposal cap"):
        shadow.prepare_keyframe(
            scene_id="scene",
            frame_id=0,
            proposal_ids=ids,
            boxes_xyxy=boxes,
            proposal_scores=scores,
            proposal_image_shape=(100, 100),
            proposal_to_depth_affine=affine,
            depth_m=depth,
            intrinsics=intrinsics,
            camera_to_world=pose,
        )


def test_depth_pixel_cap_precedes_full_copy(monkeypatch):
    scalar = np.asarray([2.0], dtype=np.float32)
    oversized = np.lib.stride_tricks.as_strided(
        scalar,
        shape=(2049, 2048),
        strides=(0, 0),
        writeable=False,
    )
    original_array = smov.np.array

    def guarded_array(value, *args, **kwargs):
        if value is oversized:
            raise AssertionError("oversized depth was copied")
        return original_array(value, *args, **kwargs)

    monkeypatch.setattr(smov.np, "array", guarded_array)
    result = extract(depth_m=oversized)
    assert result.reason == "depth_pixel_cap"


def test_nonuniform_explicit_affine_mapping_and_constraints():
    depth, intrinsics, pose = scene_inputs()
    affine = smov.aligned_resize_affine((200, 400), depth.shape)
    result = extract(
        box_xyxy=[40, 20, 360, 180],
        proposal_image_shape=(200, 400),
        proposal_to_depth_affine=affine,
        depth_m=depth,
        intrinsics=intrinsics,
        camera_to_world=pose,
    )
    assert result.accepted
    np.testing.assert_allclose(result.fragment.crop_xyxy_depth, [10, 10, 90, 90])

    shear = np.array([[0.25, 0.01, 0], [0, 0.5, 0], [0, 0, 1]])
    invalid = extract(
        proposal_image_shape=(200, 400),
        proposal_to_depth_affine=shear,
    )
    assert invalid.reason == "proposal_to_depth_affine_must_be_axis_aligned"
    reflected = extract(
        proposal_to_depth_affine=np.diag([-1.0, 1.0, 1.0])
    )
    assert reflected.reason == "proposal_to_depth_affine_must_have_positive_scale"


def test_ninety_degree_camera_pose_backprojection():
    depth, intrinsics, pose = scene_inputs()
    pose[:3, :3] = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    pose[:3, 3] = [1, 2, 3]
    result = extract(
        box_xyxy=[10, 10, 90, 90],
        depth_m=depth,
        intrinsics=intrinsics,
        camera_to_world=pose,
    )
    assert result.accepted
    assert np.any(np.all(np.isclose(result.fragment.points_world, [1, 2, 5]), axis=1))
    assert np.any(np.all(np.isclose(result.fragment.points_world, [1, 3, 5]), axis=1))
    assert np.any(np.all(np.isclose(result.fragment.points_world, [2, 2, 5]), axis=1))
    recovered_camera = (result.fragment.points_world - pose[:3, 3]) @ pose[:3, :3]
    np.testing.assert_allclose(recovered_camera[:, 2], 2.0)


def test_coordinate_contract_documents_registered_axis_and_pixel_conventions():
    documentation = inspect.getdoc(smov)
    assert "already registered to the depth image" in documentation
    assert "x=column, y=row" in documentation
    assert "axis-aligned affine" in documentation
    assert "pixel-center coordinates" in documentation


def test_smooth_full_resolution_slope_is_not_a_stride_edge():
    depth, intrinsics, pose = scene_inputs()
    depth[:] = 1.0 + 0.05 * np.arange(100, dtype=np.float32)[None, :]
    result = extract(depth_m=depth, intrinsics=intrinsics, camera_to_world=pose)
    assert result.accepted
    assert result.coverage.effective_stride == 4
    assert result.coverage.edge_pixels == 0
    assert result.coverage.component_pixels == result.coverage.sampled_rays


def test_unsampled_one_pixel_depth_edge_blocks_stride_connectivity():
    depth, intrinsics, pose = scene_inputs()
    depth[:, 51:] = 2.4
    result = extract(depth_m=depth, intrinsics=intrinsics, camera_to_world=pose)
    assert result.accepted
    assert result.coverage.component_pixels < result.coverage.usable_rays
    np.testing.assert_allclose(result.fragment.points_world[:, 2], 2.0)


def test_state_is_causal_bounded_and_terminal_rejects_stale_frame():
    shadow = smov.FragmentShadow()
    with pytest.raises(ValueError, match="most recently committed"):
        shadow.terminal_snapshot(frame_id=0)

    for frame_id in range(6):
        batch = prepare(shadow, frame_id)
        shadow.commit_keyframe(batch, track_ids=(42,), active_track_ids=(42,))

    snapshot = shadow.terminal_snapshot(frame_id=5)
    assert len(snapshot.tracks) == 1
    track = snapshot.tracks[0]
    assert [view.frame_id for view in track.views] == [1, 2, 3, 4, 5]
    assert len(track.views) <= 5
    assert len(track.points_world) <= 1024
    assert all(len(view.points_world) <= 512 for view in track.views)
    with pytest.raises(ValueError, match="most recently committed"):
        shadow.terminal_snapshot(frame_id=4)
    with pytest.raises(ValueError, match="strictly increasing"):
        prepare(shadow, 5)


def test_terminal_cannot_skip_pending_and_close_is_final():
    shadow = smov.FragmentShadow()
    batch = prepare(shadow, 2)
    with pytest.raises(RuntimeError, match="pending"):
        shadow.terminal_snapshot(frame_id=2)
    shadow.commit_keyframe(batch, track_ids=(9,))
    terminal = shadow.terminal_snapshot(frame_id=2, close=True)
    assert terminal.frame_id == 2
    with pytest.raises(RuntimeError, match="not idle"):
        prepare(shadow, 3)


def test_commit_validation_failure_rolls_back_and_releases_pending():
    shadow = smov.FragmentShadow()
    batch = prepare(shadow, 0)
    with pytest.raises(ValueError, match="align"):
        shadow.commit_keyframe(batch, track_ids=())
    diagnostics = shadow.diagnostics()
    assert diagnostics["pending"] is False
    assert diagnostics["tracks"] == 0
    assert diagnostics["stats"]["failed_commits"] == 1

    next_batch = prepare(shadow, 1)
    with pytest.raises(ValueError, match="AbortReason"):
        shadow.abort_keyframe(next_batch, reason="arbitrary:user:value")
    assert shadow.diagnostics()["pending"] is True
    shadow.abort_keyframe(next_batch, reason=smov.AbortReason.WRAPPER_FAILURE)
    diagnostics = shadow.diagnostics()
    assert diagnostics["pending"] is False
    assert diagnostics["stats"]["aborted_keyframes"] == 1
    assert diagnostics["failure_reasons"]["abort:wrapper_failure"] == 1


def test_active_track_cap_is_checked_before_iteration():
    shadow = smov.FragmentShadow({"max_tracks": 1})
    batch = prepare(shadow, 0)

    class OversizedActiveIds(Sequence):
        def __len__(self):
            return 2

        def __getitem__(self, index):
            raise AssertionError("oversized active IDs were traversed")

    with pytest.raises(ValueError, match="configured track cap"):
        shadow.commit_keyframe(
            batch,
            track_ids=(10,),
            active_track_ids=OversizedActiveIds(),
        )
    assert shadow.diagnostics()["pending"] is False
    assert shadow.diagnostics()["tracks"] == 0


def test_commit_internal_failure_is_atomic(monkeypatch):
    shadow = smov.FragmentShadow()
    initial = prepare(shadow, 0)
    shadow.commit_keyframe(initial, track_ids=(10,), active_track_ids=(10,))
    before = shadow.terminal_snapshot(frame_id=0)
    before_points = before.tracks[0].points_world.copy()
    before_frames = tuple(view.frame_id for view in before.tracks[0].views)

    batch = prepare(shadow, 1, proposal_ids=(2, 3))
    original = shadow._bounded_views
    calls = 0

    def fail_second(views):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("injected commit failure")
        return original(views)

    monkeypatch.setattr(shadow, "_bounded_views", fail_second)
    with pytest.raises(RuntimeError, match="injected"):
        shadow.commit_keyframe(
            batch,
            track_ids=(10, 11),
            active_track_ids=(10, 11),
        )

    assert shadow.diagnostics()["pending"] is False
    after = shadow.terminal_snapshot(frame_id=0)
    assert tuple(view.frame_id for view in after.tracks[0].views) == before_frames
    np.testing.assert_array_equal(after.tracks[0].points_world, before_points)
    assert [track.track_id for track in after.tracks] == [10]


def test_same_frame_duplicate_capacity_and_active_retirement():
    shadow = smov.FragmentShadow({"max_tracks": 1})
    first = prepare(shadow, 0, proposal_ids=(1, 2), scores=np.array([0.8, 0.9]))
    result = shadow.commit_keyframe(first, track_ids=(10, 10), active_track_ids=(10,))
    assert sum(item.accepted for item in result.decisions) == 1
    assert {item.reason for item in result.decisions} == {"accepted", "same_frame_duplicate"}

    second = prepare(shadow, 1, proposal_ids=(3,))
    capacity = shadow.commit_keyframe(second, track_ids=(11,))
    assert capacity.decisions[0].reason == "track_capacity"

    third = prepare(shadow, 2, proposal_ids=(4,))
    inactive = shadow.commit_keyframe(third, track_ids=(10,), active_track_ids=())
    assert inactive.accepted_track_ids == ()
    assert inactive.decisions[0].reason == "inactive_track_id"
    assert shadow.terminal_snapshot(frame_id=2).tracks == ()


def test_active_retirement_happens_before_capacity_admission():
    shadow = smov.FragmentShadow({"max_tracks": 1})
    first = prepare(shadow, 0, proposal_ids=(1,))
    shadow.commit_keyframe(first, track_ids=(10,), active_track_ids=(10,))

    replacement = prepare(shadow, 1, proposal_ids=(2,))
    result = shadow.commit_keyframe(
        replacement, track_ids=(11,), active_track_ids=(11,)
    )
    assert result.accepted_track_ids == (11,)
    assert result.decisions[0].reason == "accepted"
    assert [track.track_id for track in shadow.terminal_snapshot(frame_id=1).tracks] == [11]


def test_track_alias_atomically_merges_losing_history_before_retirement(monkeypatch):
    shadow = smov.FragmentShadow()
    first = prepare(shadow, 0, proposal_ids=(1,))
    shadow.commit_keyframe(first, track_ids=(10,), active_track_ids=(10,))
    second = prepare(shadow, 1, proposal_ids=(2,))
    shadow.commit_keyframe(
        second, track_ids=(20,), active_track_ids=(10, 20)
    )

    before = shadow.terminal_snapshot(frame_id=1)
    assert [track.track_id for track in before.tracks] == [10, 20]
    original = shadow._bounded_views

    def injected_failure(views):
        raise RuntimeError("alias merge failure")

    third = prepare(shadow, 2, proposal_ids=(3,))
    monkeypatch.setattr(shadow, "_bounded_views", injected_failure)
    with pytest.raises(RuntimeError, match="alias merge failure"):
        shadow.commit_keyframe(
            third,
            track_ids=(10,),
            active_track_ids=(10,),
            track_aliases={20: 10},
        )
    after_failure = shadow.terminal_snapshot(frame_id=1)
    assert [track.track_id for track in after_failure.tracks] == [10, 20]

    monkeypatch.setattr(shadow, "_bounded_views", original)
    fourth = prepare(shadow, 3, proposal_ids=(4,))
    result = shadow.commit_keyframe(
        fourth,
        track_ids=(10,),
        active_track_ids=(10,),
        track_aliases={20: 10},
    )
    assert result.accepted_track_ids == (10,)
    terminal = shadow.terminal_snapshot(frame_id=3)
    assert [track.track_id for track in terminal.tracks] == [10]
    assert [view.frame_id for view in terminal.tracks[0].views] == [0, 1, 3]
    assert len(terminal.tracks[0].points_world) <= 1024
    assert shadow.diagnostics()["stats"]["absorbed_track_aliases"] == 1


def test_track_alias_validation_is_bounded_acyclic_and_targets_active():
    shadow = smov.FragmentShadow()
    first = prepare(shadow, 0)
    shadow.commit_keyframe(first, track_ids=(10,), active_track_ids=(10,))

    cycle = prepare(shadow, 1)
    with pytest.raises(ValueError, match="acyclic"):
        shadow.commit_keyframe(
            cycle,
            track_ids=(10,),
            active_track_ids=(10, 20),
            track_aliases={10: 20, 20: 10},
        )

    inactive_target = prepare(shadow, 2)
    with pytest.raises(ValueError, match="targets must be active"):
        shadow.commit_keyframe(
            inactive_target,
            track_ids=(11,),
            active_track_ids=(11,),
            track_aliases={20: 10},
        )

    class OversizedAliases(dict):
        def __len__(self):
            return smov.DEFAULT_CONFIG["max_tracks"] + 1

        def items(self):
            raise AssertionError("oversized aliases were traversed")

    oversized = prepare(shadow, 3)
    with pytest.raises(ValueError, match="configured track cap"):
        shadow.commit_keyframe(
            oversized,
            track_ids=(10,),
            active_track_ids=(10,),
            track_aliases=OversizedAliases(),
        )
    assert shadow.diagnostics()["pending"] is False


def test_diagnostics_include_coverage_and_timing_without_repo_dependencies():
    shadow = smov.FragmentShadow()
    batch = prepare(shadow, 0)
    shadow.commit_keyframe(batch, track_ids=(2,))
    shadow.terminal_snapshot(frame_id=0)
    diagnostics = shadow.diagnostics()
    assert diagnostics["coverage"]["samples"] == 1
    assert diagnostics["coverage"]["mean_valid_depth_ratio"] > 0
    assert len(diagnostics["timing"]["prepare"]["samples_ms"]) == 1
    assert len(diagnostics["timing"]["commit"]["samples_ms"]) == 1
    assert len(diagnostics["timing"]["terminal"]["samples_ms"]) == 1

    source = inspect.getsource(smov)
    assert "import torch" not in source
    assert "boxfusion." not in source.replace(smov.SCHEMA, "")
    parameters = inspect.signature(smov.extract_fragment).parameters
    assert not ({"rgb", "clip", "ground_truth", "checkpoint"} & set(parameters))
