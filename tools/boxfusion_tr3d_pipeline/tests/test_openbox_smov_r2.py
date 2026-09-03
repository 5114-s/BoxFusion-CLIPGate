from __future__ import annotations

from dataclasses import fields

import numpy as np
import pytest

import boxfusion.openbox_smov_r2 as r2
from boxfusion.tr3d_r2_geometry import yaw_obb_corners_world


def _enabled(**overrides):
    config = {"enabled": True, "observer_only": True}
    config.update(overrides)
    return config


def _intrinsics(shape=(65, 65), focal=32.0):
    height, width = shape
    matrix = np.eye(3, dtype=np.float64)
    matrix[0, 0] = focal
    matrix[1, 1] = focal
    matrix[0, 2] = (width - 1) / 2.0
    matrix[1, 2] = (height - 1) / 2.0
    return matrix


def _prepare(
    observer,
    *,
    frame_id,
    proposal_ids,
    previous_groups,
    pose_x=0.0,
    boxes=None,
):
    proposal_ids = np.asarray(proposal_ids, dtype=np.int64)
    if boxes is None:
        boxes = np.tile(
            np.asarray([[0.0, 0.0, 64.0, 64.0]], dtype=np.float64),
            (len(proposal_ids), 1),
        )
    pose = np.eye(4, dtype=np.float64)
    pose[0, 3] = pose_x
    return observer.prepare_keyframe(
        scene_id="scene",
        frame_id=frame_id,
        proposal_ids=proposal_ids,
        boxes_xyxy=boxes,
        proposal_scores=np.linspace(0.9, 0.8, len(proposal_ids)),
        proposal_image_shape=(65, 65),
        depth_m=np.full((65, 65), 2.0, dtype=np.float32),
        intrinsics=_intrinsics(),
        camera_to_world=pose,
        previous_fusion_groups=previous_groups,
    )


def _view(frame_id, marker):
    grid_x, grid_y = np.meshgrid(
        np.arange(10, dtype=np.float32) * 0.1,
        np.arange(10, dtype=np.float32) * 0.1,
    )
    points = np.column_stack(
        (
            grid_x.reshape(-1),
            grid_y.reshape(-1),
            np.full(100, marker, dtype=np.float32),
        )
    )
    pose = np.eye(4, dtype=np.float64)
    pose[0, 3] = float(frame_id)
    return r2.R2ViewFragment(
        proposal_id=frame_id,
        frame_id=frame_id,
        score=0.5,
        crop_xyxy_depth=np.asarray([2, 2, 8, 8], dtype=np.float32),
        image_shape=(11, 11),
        intrinsics=_intrinsics((11, 11), 8.0),
        camera_to_world=pose,
        points_world=points,
        ray_pixels=np.asarray([[5, 5]], dtype=np.int32),
        ray_directions_world=np.asarray([[0, 0, 1]], dtype=np.float32),
        ray_depth_m=np.asarray([3], dtype=np.float32),
        valid_depth_ratio=1.0,
    )


def _face_view(box, axis, side, *, frame_id=0, count=8):
    box = np.asarray(box, dtype=np.float64)
    local = np.zeros((count, 3), dtype=np.float64)
    local[:, axis] = side * box[3 + axis] * 0.5
    local[:, 1 - axis] = np.linspace(-0.3, 0.3, count)
    local[:, 2] = np.linspace(-0.25, 0.25, count)
    cosine, sine = np.cos(box[6]), np.sin(box[6])
    rotation = np.asarray([[cosine, -sine], [sine, cosine]])
    points = local.copy()
    points[:, :2] = local[:, :2] @ rotation.T + box[None, :2]
    camera_local = np.zeros(2, dtype=np.float64)
    camera_local[axis] = side * (box[3 + axis] * 0.5 + 3.0)
    camera = np.asarray(
        [*(camera_local @ rotation.T + box[:2]), box[2]], dtype=np.float64
    )
    pose = np.eye(4, dtype=np.float64)
    pose[:3, 3] = camera
    return r2.R2ViewFragment(
        proposal_id=frame_id,
        frame_id=frame_id,
        score=0.5,
        crop_xyxy_depth=np.asarray([2, 2, 8, 8], dtype=np.float32),
        image_shape=(11, 11),
        intrinsics=_intrinsics((11, 11), 8.0),
        camera_to_world=pose,
        points_world=points,
        ray_pixels=np.asarray([[5, 5]], dtype=np.int32),
        ray_directions_world=np.asarray([[0, 0, 1]], dtype=np.float32),
        ray_depth_m=np.asarray([3], dtype=np.float32),
        valid_depth_ratio=1.0,
    )


def test_config_is_default_off_strict_shadow_and_json_safe():
    config = r2.resolve_openbox_smov_r2_config()
    assert config["enabled"] is False
    assert config["observer_only"] is True
    assert config["voxel_size_m"] == 0.05
    assert config["max_views_per_track"] == 5
    assert config["max_points_per_track"] == 1024
    assert config["max_face_candidates_per_fit"] == 4
    assert config["face_extension_max_m"] == 0.30
    assert r2.SCHEMA == "boxfusion.openbox_smov_r2_shadow.v2"
    assert r2.build_openbox_smov_r2({}).enabled is False

    with pytest.raises(ValueError, match="unknown openbox_smov_r2"):
        r2.resolve_openbox_smov_r2_config({"typo": 1})
    with pytest.raises(ValueError, match="observer_only"):
        r2.resolve_openbox_smov_r2_config(
            {"enabled": True, "observer_only": False}
        )


@pytest.mark.parametrize(
    "key,value",
    [
        ("component_jump_m", 0.25),
        ("lower_quantile", 0.10),
        ("min_views", 2),
        ("face_front_dot", 0.30),
        ("face_extension_fraction", 0.20),
        ("max_center_shift_diagonal", 0.9),
    ],
)
def test_enabled_s0_rejects_changed_frozen_geometry_thresholds(key, value):
    with pytest.raises(ValueError, match="frozen"):
        r2.resolve_openbox_smov_r2_config(_enabled(**{key: value}))


@pytest.mark.parametrize(
    "key,value",
    [
        ("max_points_per_view", 513),
        ("max_points_per_track", 1025),
        ("max_validation_rays_per_view", 1025),
        ("max_views_per_track", 6),
        ("max_tracks", 1025),
        ("max_proposals_per_keyframe", 65),
        ("max_face_candidates_per_fit", 5),
        ("max_diagnostics", 1025),
        ("timing_window", 4097),
    ],
)
def test_enabled_runtime_budgets_cannot_exceed_frozen_realtime_caps(key, value):
    with pytest.raises(ValueError, match="must not exceed|bounded"):
        r2.resolve_openbox_smov_r2_config(_enabled(**{key: value}))


@pytest.mark.parametrize(
    "overrides,match",
    [
        ({"face_front_dot": 0.1, "face_weak_dot": 0.2}, "threshold"),
        ({"min_face_points": 3, "min_face_weak_points": 4}, "min_face"),
        ({"face_band_fraction": 0.25}, "face_band_fraction"),
        (
            {"face_extension_min_m": 0.4, "face_extension_max_m": 0.3},
            "extension bounds",
        ),
    ],
)
def test_face_config_relations_fail_closed_while_disabled(overrides, match):
    with pytest.raises(ValueError, match=match):
        r2.resolve_openbox_smov_r2_config(overrides)


@pytest.mark.parametrize("yaw", [0.0, np.deg2rad(37.0)])
@pytest.mark.parametrize(
    "axis,side,face_index", [(0, -1, 0), (0, 1, 1), (1, -1, 2), (1, 1, 3)]
)
def test_face_visibility_uses_local_yaw_face_center_ray(
    yaw, axis, side, face_index
):
    box = np.asarray([1.0, -2.0, 0.5, 2.0, 2.0, 2.0, yaw])
    evidence = r2._face_visibility_xy(
        box, [_face_view(box, axis, side)], r2.DEFAULT_CONFIG
    )
    expected_mask = [False] * 4
    expected_mask[face_index] = True
    assert evidence.strong_mask == tuple(expected_mask)
    assert evidence.weak_mask == tuple(expected_mask)
    expected_signs = [0, 0]
    expected_signs[axis] = -side
    assert evidence.extension_signs == tuple(expected_signs)

    inside = _face_view(box, axis, side)
    inside_pose = np.array(inside.camera_to_world, copy=True)
    inside_pose[:3, 3] = box[:3]
    inside = r2.replace(inside, camera_to_world=inside_pose)
    inside_evidence = r2._face_visibility_xy(
        box, [inside], r2.DEFAULT_CONFIG
    )
    assert not any(inside_evidence.strong_mask)
    assert not any(inside_evidence.weak_mask)


def test_face_visibility_requires_per_view_support_and_opposite_weak_veto():
    box = np.asarray([0.0, 0.0, 0.0, 2.0, 2.0, 2.0, 0.0])
    strong = r2._face_visibility_xy(
        box, [_face_view(box, 0, -1, count=8)], r2.DEFAULT_CONFIG
    )
    assert strong.strong_mask == (True, False, False, False)
    assert strong.weak_mask == (True, False, False, False)
    assert strong.extension_signs == (1, 0)

    weak = r2._face_visibility_xy(
        box, [_face_view(box, 0, -1, count=4)], r2.DEFAULT_CONFIG
    )
    assert weak.strong_mask == (False, False, False, False)
    assert weak.weak_mask == (True, False, False, False)
    assert weak.extension_signs == (0, 0)

    split = r2._face_visibility_xy(
        box,
        [
            _face_view(box, 0, -1, frame_id=1, count=4),
            _face_view(box, 0, -1, frame_id=2, count=4),
        ],
        r2.DEFAULT_CONFIG,
    )
    assert not any(split.strong_mask)

    vetoed = r2._face_visibility_xy(
        box,
        [
            _face_view(box, 0, -1, frame_id=1, count=8),
            _face_view(box, 0, 1, frame_id=2, count=4),
        ],
        r2.DEFAULT_CONFIG,
    )
    assert vetoed.strong_mask[0]
    assert vetoed.weak_mask[1]
    assert vetoed.extension_signs == (0, 0)


def test_face_extension_keeps_visible_bounds_and_uses_frozen_delta_clip():
    yaw = np.deg2rad(31.0)
    base = np.asarray([1.0, -2.0, 0.5, 2.0, 1.2, 1.5, yaw])
    evidence = r2._FaceEvidence(
        extension_signs=(1, -1),
        strong_mask=(True, False, False, True),
        weak_mask=(True, False, False, True),
    )
    candidate, signs, deltas = r2._extend_face_candidate(
        base, evidence, "face_xy", r2.DEFAULT_CONFIG
    )
    assert signs == (1, -1)
    assert deltas == pytest.approx((0.30, 0.30))
    assert candidate[3:5] == pytest.approx(base[3:5] + [0.30, 0.30])
    assert candidate[2] == base[2]
    assert candidate[5] == base[5]
    assert candidate[6] == base[6]

    def local_bounds(box):
        corners = yaw_obb_corners_world(box)
        cosine, sine = np.cos(yaw), np.sin(yaw)
        world_to_local = np.asarray([[cosine, sine], [-sine, cosine]])
        local = (corners[:, :2] - base[None, :2]) @ world_to_local.T
        return local.min(axis=0), local.max(axis=0)

    base_min, base_max = local_bounds(base)
    candidate_min, candidate_max = local_bounds(candidate)
    assert candidate_min[0] == pytest.approx(base_min[0])
    assert candidate_max[0] == pytest.approx(base_max[0] + 0.30)
    assert candidate_max[1] == pytest.approx(base_max[1])
    assert candidate_min[1] == pytest.approx(base_min[1] - 0.30)

    small = base.copy()
    small[3:5] = 0.10
    small_candidate, _, small_delta = r2._extend_face_candidate(
        small, evidence, "face_xy", r2.DEFAULT_CONFIG
    )
    assert small_delta == pytest.approx((0.05, 0.05))
    assert small_candidate[3:5] == pytest.approx([0.15, 0.15])
    base_candidate, base_signs, base_delta = r2._extend_face_candidate(
        base, evidence, "base", r2.DEFAULT_CONFIG
    )
    np.testing.assert_array_equal(base_candidate, base)
    assert base_signs == (0, 0)
    assert base_delta == (0.0, 0.0)


def test_prepare_extracts_bounded_sparse_crop_geometry_not_full_depth():
    observer = r2.OpenBoxSMOVR2Shadow(_enabled())
    batch = _prepare(
        observer,
        frame_id=0,
        proposal_ids=(10,),
        previous_groups=(),
    )
    fragment = batch.fragments[0]
    assert fragment is not None
    assert len(fragment.points_world) <= 512
    assert len(fragment.ray_depth_m) <= 1024
    assert fragment.ray_pixels.shape == (len(fragment.ray_depth_m), 2)
    assert fragment.ray_directions_world.shape == (
        len(fragment.ray_depth_m),
        3,
    )
    assert not fragment.points_world.flags.writeable
    assert not fragment.ray_depth_m.flags.writeable
    assert "depth_m" not in {field.name for field in fields(r2.R2ViewFragment)}
    assert fragment.ray_depth_m.ndim == 1


def test_component_search_does_not_poison_an_alternate_valid_depth_path():
    # X (1.28 m) is first inspected from A (1.00 m), where the jump is too
    # large, but is connected through B (1.14 m).  Rejected neighbours must
    # not be marked visited before that second valid path is considered.
    depth = np.zeros((3, 3), dtype=np.float32)
    depth[1, 1] = 1.00  # seed
    depth[0, 1] = 1.00  # A, dequeued before B
    depth[1, 2] = 1.14  # B
    depth[0, 2] = 1.28  # X
    config = dict(r2.DEFAULT_CONFIG)
    config.update(
        pixel_stride=1,
        min_component_pixels=4,
        depth_edge_m=10.0,
        component_jump_m=0.15,
        voxel_size_m=0.001,
    )
    fragment = r2._extract_fragment(
        proposal_id=1,
        frame_id=0,
        score=1.0,
        box_xyxy=np.asarray([0.0, 0.0, 2.0, 2.0]),
        proposal_image_shape=(3, 3),
        depth_m=depth,
        intrinsics=_intrinsics((3, 3), 2.0),
        camera_to_world=np.eye(4),
        cfg=config,
    )
    assert fragment is not None
    assert len(fragment.points_world) == 4


def test_prepare_commit_is_causal_exactly_once_and_ids_are_row_aligned():
    observer = r2.OpenBoxSMOVR2Shadow(_enabled())
    batch = _prepare(
        observer,
        frame_id=0,
        proposal_ids=(10,),
        previous_groups=(),
    )
    with pytest.raises(RuntimeError, match="not idle"):
        _prepare(
            observer,
            frame_id=1,
            proposal_ids=(11,),
            previous_groups=(),
        )

    committed = observer.commit_keyframe(
        batch, current_fusion_groups=((10,),)
    )
    assert committed.committed_track_ids == (10,)
    assert committed.current_stable_ids == (10,)
    np.testing.assert_array_equal(observer.current_stable_ids(((10,),)), [10])
    with pytest.raises(RuntimeError, match="exact pending batch"):
        observer.commit_keyframe(batch, current_fusion_groups=((10,),))
    with pytest.raises(ValueError, match="without a causal registry update"):
        observer.current_stable_ids(((10, 11),))


def test_commit_event_recovers_proposal_omitted_by_native_five_view_cap():
    observer = r2.OpenBoxSMOVR2Shadow(_enabled())
    first = _prepare(
        observer,
        frame_id=0,
        proposal_ids=(10,),
        previous_groups=(),
    )
    observer.commit_keyframe(first, current_fusion_groups=((10,),))
    second = _prepare(
        observer,
        frame_id=1,
        proposal_ids=(11,),
        previous_groups=((10,),),
        pose_x=1.0,
    )
    receipt = observer.commit_keyframe(
        second,
        current_fusion_groups=((10,),),
        association_events=(
            {
                "stage": "boxfusion",
                "winner_members": (10,),
                "loser_members": (11,),
            },
        ),
    )
    assert receipt.committed_track_ids == (10,)
    assert [view.frame_id for view in observer._tracks[10].views] == [0, 1]


def test_collision_groups_supported_by_independent_causal_registry():
    # The shared source member is the rare native collision case that the
    # independent CausalFusionIdRegistry is specifically designed to repair.
    assert r2._groups(((1, 2), (2, 3))) == ((1, 2), (2, 3))


def test_collision_group_cannot_steal_history_from_registry_inheritor():
    observer = r2.OpenBoxSMOVR2Shadow(_enabled())
    np.testing.assert_array_equal(observer._identity.update(((2,),)), [2])
    observer._tracks[2] = r2._TrackState(
        stable_id=2,
        views=[_view(0, 1.0)],
    )

    batch = _prepare(
        observer,
        frame_id=1,
        proposal_ids=(3,),
        previous_groups=((2,),),
        pose_x=1.0,
    )
    committed = observer.commit_keyframe(
        batch,
        current_fusion_groups=((1, 2), (2, 3)),
    )

    inherited, collision_repair = committed.current_stable_ids
    assert inherited == 2
    assert collision_repair != inherited
    assert [view.frame_id for view in observer._tracks[inherited].views] == [0]
    assert [
        view.frame_id for view in observer._tracks[collision_repair].views
    ] == [1]


def test_retired_prior_history_is_consumed_by_only_one_collision_target():
    observer = r2.OpenBoxSMOVR2Shadow(_enabled())
    np.testing.assert_array_equal(
        observer._identity.update(((1,), (2,), (3,))), [1, 2, 3]
    )
    observer._tracks = {
        stable_id: r2._TrackState(
            stable_id=stable_id,
            views=[_view(stable_id, float(stable_id))],
        )
        for stable_id in (1, 2, 3)
    }
    batch = observer.prepare_abstain(
        scene_id="scene",
        frame_id=10,
        proposal_ids=np.empty((0,), dtype=np.int64),
        previous_fusion_groups=((1,), (2,), (3,)),
        reason="no_current_proposals",
    )
    committed = observer.commit_keyframe(
        batch,
        current_fusion_groups=((1, 3), (2, 3)),
    )
    assert committed.current_stable_ids == (1, 2)
    frames_by_track = {
        track_id: {view.frame_id for view in state.views}
        for track_id, state in observer._tracks.items()
    }
    assert frames_by_track == {1: {1, 3}, 2: {2}}


def test_memory_keeps_one_view_per_frame_five_views_and_1024_total_points():
    observer = r2.OpenBoxSMOVR2Shadow(_enabled())
    groups = ()
    members = []
    for frame_id in range(7):
        proposal_id = 100 + frame_id
        members.append(proposal_id)
        batch = _prepare(
            observer,
            frame_id=frame_id,
            proposal_ids=(proposal_id,),
            previous_groups=groups,
            pose_x=float(frame_id),
        )
        groups = (tuple(members),)
        observer.commit_keyframe(batch, current_fusion_groups=groups)

    state = next(iter(observer._tracks.values()))
    assert len(state.views) == 5
    assert len({view.frame_id for view in state.views}) == 5
    assert sum(len(view.points_world) for view in state.views) <= 1024


def test_same_frame_commits_at_most_one_fragment_to_a_track():
    observer = r2.OpenBoxSMOVR2Shadow(_enabled())
    batch = _prepare(
        observer,
        frame_id=0,
        proposal_ids=(10, 11),
        previous_groups=(),
    )
    receipt = observer.commit_keyframe(
        batch, current_fusion_groups=((10, 11),)
    )
    assert receipt.accepted_track_ids == (10,)
    assert len(observer._tracks[10].views) == 1
    assert observer.summary()["same_frame_duplicates"] == 1


def test_terminal_uses_true_loo_and_never_mutates_native(monkeypatch):
    observer = r2.OpenBoxSMOVR2Shadow(_enabled())
    views = [_view(0, 1.0), _view(1, 2.0), _view(2, 3.0)]
    observer._tracks[7] = r2._TrackState(stable_id=7, views=views)
    fit_markers = []

    def fake_fit(points, yaw, cfg):
        fit_markers.append(tuple(sorted(set(np.asarray(points)[:, 2].tolist()))))
        return np.asarray([0.1, 0.0, 3.0, 2.0, 2.0, 2.0, yaw])

    def fake_evaluate(box, view, cfg):
        if np.isclose(box[0], 0.1):
            return (0.8, 0.8, 0.1)
        return (0.7, 0.7, 0.2)

    monkeypatch.setattr(r2, "_fit_yaw", fake_fit)
    monkeypatch.setattr(r2, "_evaluate", fake_evaluate)
    monkeypatch.setattr(r2, "_safety", lambda *args: (True, 0.1, 1.0))
    monkeypatch.setattr(r2, "_pca_yaw", lambda points, fallback: 0.2)

    native = yaw_obb_corners_world(
        np.asarray([0.0, 0.0, 3.0, 2.0, 2.0, 2.0, 0.0])
    )[None].astype(np.float64)
    scores = np.asarray([0.9], dtype=np.float64)
    before_native = native.copy()
    before_scores = scores.copy()
    result = observer.finalize_shadow(
        native_corners=native,
        native_scores=scores,
        stable_ids=np.asarray([7], dtype=np.int64),
    )

    # First hypothesis: three fits each exclude exactly its held-out view,
    # followed by one diagnostic all-view fit.  The PCA hypothesis repeats it.
    expected = [(2.0, 3.0), (1.0, 3.0), (1.0, 2.0), (1.0, 2.0, 3.0)]
    assert fit_markers[:4] == expected
    assert fit_markers[4:] == expected
    np.testing.assert_array_equal(native, before_native)
    np.testing.assert_array_equal(scores, before_scores)
    np.testing.assert_array_equal(result.native_corners, before_native)
    np.testing.assert_array_equal(result.native_scores, before_scores)
    assert result.would_replace_mask.tolist() == [True]
    assert not result.native_corners.flags.writeable
    assert not result.counterfactual_corners.flags.writeable
    assert result.summary.as_dict()["counterfactual_geometry_applied"] is False


def test_openbox_face_candidate_is_strict_loo_and_can_win_in_shadow(monkeypatch):
    observer = r2.OpenBoxSMOVR2Shadow(_enabled())
    views = [_view(0, 1.0), _view(1, 2.0), _view(2, 3.0)]
    observer._tracks[7] = r2._TrackState(stable_id=7, views=views)
    visibility_calls = []

    def fake_fit(points, yaw, cfg):
        return np.asarray([0.1, 0.0, 3.0, 2.0, 2.0, 2.0, yaw])

    def fake_visibility(box, supplied_views, cfg):
        visibility_calls.append(tuple(view.frame_id for view in supplied_views))
        return r2._FaceEvidence(
            extension_signs=(1, 0),
            strong_mask=(True, False, False, False),
            weak_mask=(True, False, False, False),
        )

    def fake_evaluate(box, view, cfg):
        if box[3] > 2.0 + 1e-9:
            return (0.9, 0.9, 0.05)
        if np.isclose(box[0], 0.1):
            return (0.8, 0.8, 0.10)
        return (0.7, 0.7, 0.20)

    monkeypatch.setattr(r2, "_fit_yaw", fake_fit)
    monkeypatch.setattr(r2, "_face_visibility_xy", fake_visibility)
    monkeypatch.setattr(r2, "_evaluate", fake_evaluate)
    monkeypatch.setattr(r2, "_pca_yaw", lambda points, fallback: fallback)

    native = yaw_obb_corners_world(
        np.asarray([0.0, 0.0, 3.0, 2.0, 2.0, 2.0, 0.0])
    )[None].astype(np.float32)
    before = native.copy()
    result = observer.finalize_shadow(
        native_corners=native,
        native_scores=np.asarray([0.9], dtype=np.float32),
        stable_ids=np.asarray([7], dtype=np.int64),
    )
    expected_calls = [(1, 2), (0, 2), (0, 1), (0, 1, 2)]
    assert visibility_calls[:4] == expected_calls
    assert visibility_calls[4:] == expected_calls
    receipt = result.receipts[0]
    assert receipt.would_replace
    assert receipt.hypothesis == "native_yaw_quantile+face_x"
    assert receipt.face_extension_signs == (1, 0)
    assert receipt.face_extension_delta_m == pytest.approx((0.30, 0.0))
    assert receipt.face_strong_mask == (True, False, False, False)
    assert receipt.face_weak_mask == (True, False, False, False)
    np.testing.assert_array_equal(native, before)
    np.testing.assert_array_equal(result.native_corners, before)


def test_face_recipe_abstains_on_any_fold_direction_flip(monkeypatch):
    observer = r2.OpenBoxSMOVR2Shadow(_enabled())
    views = [_view(0, 1.0), _view(1, 2.0), _view(2, 3.0)]
    observer._tracks[7] = r2._TrackState(stable_id=7, views=views)

    monkeypatch.setattr(
        r2,
        "_fit_yaw",
        lambda points, yaw, cfg: np.asarray(
            [0.1, 0.0, 3.0, 2.0, 2.0, 2.0, yaw]
        ),
    )

    def inconsistent_visibility(box, supplied_views, cfg):
        frames = tuple(view.frame_id for view in supplied_views)
        sign = -1 if frames == (0, 2) else 1
        anchor = 1 if sign < 0 else 0
        strong = [False] * 4
        strong[anchor] = True
        return r2._FaceEvidence(
            extension_signs=(sign, 0),
            strong_mask=tuple(strong),
            weak_mask=tuple(strong),
        )

    monkeypatch.setattr(r2, "_face_visibility_xy", inconsistent_visibility)
    monkeypatch.setattr(
        r2,
        "_evaluate",
        lambda box, view, cfg: (
            (0.9, 0.9, 0.05)
            if box[3] > 2.0 + 1e-9
            else ((0.8, 0.8, 0.10) if np.isclose(box[0], 0.1) else (0.7, 0.7, 0.2))
        ),
    )
    monkeypatch.setattr(r2, "_pca_yaw", lambda points, fallback: fallback)
    native = yaw_obb_corners_world(
        np.asarray([0.0, 0.0, 3.0, 2.0, 2.0, 2.0, 0.0])
    )[None]
    result = observer.finalize_shadow(
        native_corners=native,
        native_scores=np.asarray([0.9]),
        stable_ids=np.asarray([7]),
    )
    assert result.receipts[0].hypothesis.endswith("+base")
    assert result.receipts[0].face_extension_signs == (0, 0)


def test_finalize_abstains_when_pose_diversity_is_insufficient():
    observer = r2.OpenBoxSMOVR2Shadow(_enabled())
    views = [_view(0, 1.0), _view(0, 2.0), _view(0, 3.0)]
    observer._tracks[7] = r2._TrackState(stable_id=7, views=views)
    native = yaw_obb_corners_world(
        np.asarray([0.0, 0.0, 3.0, 2.0, 2.0, 2.0, 0.0])
    )[None]
    result = observer.finalize_shadow(
        native_corners=native,
        native_scores=np.asarray([0.9]),
        stable_ids=np.asarray([7]),
    )
    assert not result.would_replace_mask[0]
    assert result.receipts[0].reason == "insufficient_pose_diverse_views"


def test_summary_attests_training_free_geometry_only_shadow_contract():
    observer = r2.OpenBoxSMOVR2Shadow(_enabled())
    summary = observer.summary()
    for key in (
        "training_invoked",
        "online_learning",
        "ground_truth_access",
        "clip_access",
        "semantic_access",
        "checkpoint_access",
        "future_frame_access",
        "full_scene_reconstruction",
        "native_outputs_mutated",
        "counterfactual_geometry_applied",
        "active_authorized",
    ):
        assert summary[key] is False
    assert summary["observer_only"] is True


def test_shadow_sidecar_is_create_only_and_contains_no_active_predictions(tmp_path):
    corners = yaw_obb_corners_world(
        np.asarray([0.0, 0.0, 3.0, 2.0, 2.0, 2.0, 0.0])
    )[None].astype(np.float32)
    result = r2.R2ShadowResult(
        native_corners=corners,
        native_scores=np.asarray([0.9], dtype=np.float32),
        stable_ids=np.asarray([7], dtype=np.int64),
        counterfactual_corners=corners.copy(),
        would_replace_mask=np.asarray([False]),
        receipts=(),
    )
    path = tmp_path / "r2-shadow.npz"
    returned = r2.save_r2_shadow_sidecar_create_only(result, path)
    assert returned == path
    with np.load(path, allow_pickle=False) as archive:
        assert set(archive.files) == {
            "schema",
            "native_corners",
            "native_scores",
            "stable_ids",
            "counterfactual_corners",
            "would_replace_mask",
            "receipts_json",
        }
        assert archive["schema"].item() == r2.SCHEMA
        np.testing.assert_array_equal(archive["native_corners"], corners)
    with pytest.raises(FileExistsError, match="already exists"):
        r2.save_r2_shadow_sidecar_create_only(result, path)


def test_disabled_observer_cannot_accidentally_enter_runtime_path():
    observer = r2.OpenBoxSMOVR2Shadow()
    with pytest.raises(RuntimeError, match="disabled"):
        _prepare(
            observer,
            frame_id=0,
            proposal_ids=(1,),
            previous_groups=(),
        )
    with pytest.raises(RuntimeError, match="disabled"):
        observer.prepare_abstain(
            scene_id="scene",
            frame_id=0,
            proposal_ids=np.empty((0,), dtype=np.int64),
            previous_fusion_groups=(),
            reason="sensor_unavailable",
        )
    synthetic = r2.R2KeyframeBatch(
        scene_id="scene",
        frame_id=0,
        proposal_ids=(),
        previous_fusion_groups=(),
        previous_stable_ids=(),
        fragments=(),
    )
    observer._pending = synthetic
    with pytest.raises(RuntimeError, match="disabled"):
        observer.commit_keyframe(synthetic, current_fusion_groups=())
