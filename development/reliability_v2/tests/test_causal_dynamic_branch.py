from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from boxfusion.causal_dynamic_branch import (
    DynamicBranchContractError,
    build_causal_dynamic_branch,
    dynamic_event_ledger_complete,
    resolve_causal_dynamic_branch_config,
)


class _Geometry:
    def __init__(self, boxes, rotations=None):
        self.tensor = np.asarray(boxes, dtype=np.float32)
        count = len(self.tensor)
        self.R = np.asarray(
            rotations
            if rotations is not None
            else np.repeat(np.eye(3, dtype=np.float32)[None], count, axis=0),
            dtype=np.float32,
        )


class _Instances:
    def __init__(self, init_ids, scores, boxes, appearances=None):
        self.init_id = np.asarray(init_ids, dtype=np.int64)
        self.scores = np.asarray(scores, dtype=np.float32)
        self.pred_boxes_3d = _Geometry(boxes)
        self._appearances = (
            None if appearances is None else np.asarray(appearances, dtype=np.float32)
        )

    def __len__(self):
        return len(self.init_id)

    def has(self, name):
        if name == "appearance_features":
            return self._appearances is not None
        return hasattr(self, name)

    @property
    def appearance_features(self):
        if self._appearances is None:
            raise AttributeError
        return self._appearances


def _box(x, z=2.0):
    return [x, 0.0, z, 0.8, 0.7, 1.0]


def _cfg(tmp_path: Path, *, mode="active", **state):
    return {
        "causal_dynamic_branch": {
            "mode": mode,
            "events_root": str(tmp_path / "events") if mode != "disabled" else None,
            "current_output_root": (
                str(tmp_path / "current") if mode == "active" else None
            ),
        },
        "dynamic_objects": {
            "enabled": mode != "disabled",
            "min_confirmed_hits": 1,
            "max_center_distance_m": 0.8,
            "max_relocation_distance_m": 5.0,
            "null_track_cost": 0.40,
            "null_observation_cost": 0.40,
            **state,
        },
    }


def _process(
    branch,
    source_frame,
    instances,
    fusion_list,
    depth=None,
    appearance_by_init_id=None,
):
    return branch.process_keyframe(
        source_frame_id=source_frame,
        current_instances=instances,
        fusion_list=fusion_list,
        depth_m=depth,
        intrinsics=np.asarray([[100.0, 0.0, 2.0], [0.0, 100.0, 2.0], [0, 0, 1]]),
        camera_to_world=np.eye(4),
        appearance_by_init_id=appearance_by_init_id,
    )


def test_config_is_disabled_by_default_and_active_requires_output_paths(tmp_path):
    assert resolve_causal_dynamic_branch_config({}).mode.value == "disabled"
    with pytest.raises(DynamicBranchContractError, match="events_root"):
        resolve_causal_dynamic_branch_config(
            {"causal_dynamic_branch": {"mode": "shadow"}}
        )
    with pytest.raises(DynamicBranchContractError, match="current_output_root"):
        resolve_causal_dynamic_branch_config(
            {
                "causal_dynamic_branch": {
                    "mode": "active",
                    "events_root": str(tmp_path),
                }
            }
        )


def test_disabled_branch_is_a_strict_native_noop(tmp_path):
    branch = build_causal_dynamic_branch({}, scene_id="scene_noop")
    result = branch.process_keyframe(
        source_frame_id=0,
        current_instances=None,
        fusion_list=(),
    )
    assert result.commit is None
    assert branch.is_dynamic_fusion_ids((1, 2, 3)) is False
    native = _Instances([1], [0.75], [_box(0.0)])
    materialized = branch.materialize_current(native, [[1]])
    assert materialized.modified_rows == ()
    assert materialized.dropped_rows == ()
    assert materialized.scores.tolist() == pytest.approx([0.75])
    assert materialized.mask.tolist() == [True]


def test_native_row_deduplication_and_keyframe_ordinal_are_causal(tmp_path):
    branch = build_causal_dynamic_branch(_cfg(tmp_path), scene_id="scene0000_00")
    first = _Instances([10, 11], [0.7, 0.9], [_box(0.0), _box(0.02)])
    result = _process(branch, 0, first, [[10, 11]])
    assert result.frame_ordinal == 0
    assert len(result.commit.birth_observation_ids) == 1
    assert len(result.observation_to_track) == 1
    assert result.commit.query_before_commit is True

    second = _Instances([20], [0.9], [_box(0.12)])
    result = _process(branch, 25, second, [[10, 11, 20]])
    assert result.frame_ordinal == 1
    assert result.commit.birth_observation_ids == ()
    assert result.commit.memory_version_after == 2
    branch.close()


def test_motion_uses_keyframe_time_and_current_overlay_does_not_mutate_native(tmp_path):
    cfg = _cfg(
        tmp_path,
        dynamic_speed_low_m_per_frame=0.02,
        dynamic_speed_high_m_per_frame=0.08,
        dynamic_probability_update_rate=0.8,
        velocity_update_rate=1.0,
    )
    branch = build_causal_dynamic_branch(cfg, scene_id="scene0001_00")
    fusion = [[]]
    for ordinal, source_frame in enumerate((0, 25, 50)):
        init_id = 10 + ordinal
        fusion[0].append(init_id)
        _process(
            branch,
            source_frame,
            _Instances([init_id], [0.8], [_box(0.15 * ordinal)]),
            fusion,
        )
    assert branch.dynamic_fusion_ids == frozenset({10})
    assert branch.is_dynamic_fusion_ids(fusion[0]) is True

    native = _Instances([10], [0.8], [_box(0.0)])
    before_box = native.pred_boxes_3d.tensor.copy()
    before_score = native.scores.copy()
    current = branch.materialize_current(native, fusion)
    assert current.modified_rows == (0,)
    assert current.mask.tolist() == [True]
    assert current.corners[0, :, 0].mean() > 0.15
    assert np.array_equal(native.pred_boxes_3d.tensor, before_box)
    assert np.array_equal(native.scores, before_score)
    assert current.corners.flags.writeable is False
    branch.close()


def test_dynamic_identity_stays_latched_after_object_stops(tmp_path):
    cfg = _cfg(
        tmp_path,
        dynamic_speed_low_m_per_frame=0.01,
        dynamic_speed_high_m_per_frame=0.05,
        dynamic_probability_update_rate=0.9,
        velocity_update_rate=1.0,
    )
    branch = build_causal_dynamic_branch(cfg, scene_id="scene_latched")
    fusion = [[]]
    positions = (0.0, 0.2, 0.4, 0.4, 0.4, 0.4)
    for ordinal, position in enumerate(positions):
        init_id = 100 + ordinal
        fusion[0].append(init_id)
        _process(
            branch,
            ordinal * 25,
            _Instances([init_id], [0.8], [_box(position)]),
            fusion,
        )

    track_id = branch.state.current_snapshots()[0].track_id
    assert branch.state.track_snapshot(track_id).motion_state.value != "dynamic"
    assert branch.is_dynamic_fusion_ids(fusion[0]) is True
    terminal = _Instances([100], [0.8], [_box(0.4)])
    persistent = branch.materialize_persistent(terminal, fusion)
    assert persistent.corners[0, :, 0].mean() == pytest.approx(0.0)
    branch.close()


def test_shadow_computes_motion_but_never_bypasses_or_changes_output(tmp_path):
    cfg = _cfg(
        tmp_path,
        mode="shadow",
        dynamic_speed_low_m_per_frame=0.01,
        dynamic_speed_high_m_per_frame=0.05,
        dynamic_probability_update_rate=0.9,
    )
    branch = build_causal_dynamic_branch(cfg, scene_id="scene0002_00")
    fusion = [[]]
    for frame in range(3):
        init_id = 30 + frame
        fusion[0].append(init_id)
        _process(
            branch,
            frame * 25,
            _Instances([init_id], [0.7], [_box(frame * 0.2)]),
            fusion,
        )
    assert branch.dynamic_fusion_ids
    assert branch.is_dynamic_fusion_ids(fusion[0]) is False
    native = _Instances([30], [0.7], [_box(0.0)])
    current = branch.materialize_current(native, fusion)
    assert current.modified_rows == ()
    assert current.dropped_rows == ()
    assert np.array_equal(
        current.corners,
        branch.materialize_current(native, fusion).corners,
    )
    branch.close()


def test_occlusion_is_not_negative_but_free_space_retires_current_object(tmp_path):
    cfg = _cfg(
        tmp_path,
        dormant_after_visible_misses=1,
        retire_after_visible_misses=2,
        visible_miss_score_decay=0.5,
    )
    branch = build_causal_dynamic_branch(cfg, scene_id="scene0003_00")
    first = _Instances([1], [0.8], [_box(0.0)])
    _process(branch, 0, first, [[1]], depth=np.full((5, 5), 2.0))
    track_id = branch.state.current_snapshots()[0].track_id
    score = branch.state.track_snapshot(track_id).current_score

    _process(branch, 25, _Instances([], [], np.empty((0, 6))), [[1]], depth=np.full((5, 5), 0.5))
    assert branch.state.track_snapshot(track_id).current_score == score

    _process(branch, 50, _Instances([], [], np.empty((0, 6))), [[1]], depth=np.full((5, 5), 5.0))
    assert branch.state.track_snapshot(track_id).lifecycle.value == "dormant"
    _process(branch, 75, _Instances([], [], np.empty((0, 6))), [[1]], depth=np.full((5, 5), 5.0))
    assert branch.state.track_snapshot(track_id).lifecycle.value == "retired"
    terminal = _Instances([1], [0.8], [_box(0.0)])
    current = branch.materialize_current(terminal, [[1]])
    assert current.mask.tolist() == [False]
    assert current.dropped_rows == (0,)
    branch.close()


def test_static_object_out_of_view_stays_in_global_current_map(tmp_path):
    cfg = _cfg(
        tmp_path,
        max_unobserved_coast_frames=2,
        max_reactivation_age=3,
    )
    branch = build_causal_dynamic_branch(cfg, scene_id="scene_out_of_view")
    _process(branch, 0, _Instances([1], [0.8], [_box(0.0)]), [[1]])
    empty = _Instances([], [], np.empty((0, 6)))
    for source_frame in (25, 50, 75, 100):
        _process(branch, source_frame, empty, [[1]])
    track_id = branch.state.persistent_snapshots()[0].track_id
    assert branch.state.track_snapshot(track_id).lifecycle.value == "retired"
    assert branch.state.track_snapshot(track_id).consecutive_visible_misses == 0
    terminal = _Instances([1], [0.8], [_box(0.0)])
    current = branch.materialize_current(terminal, [[1]])
    assert current.mask.tolist() == [True]
    assert current.dropped_rows == ()
    branch.close()


def test_relocation_reuses_id_and_current_map_suppresses_old_native_alias(tmp_path):
    cfg = _cfg(
        tmp_path,
        dormant_after_visible_misses=1,
        retire_after_visible_misses=2,
        min_reactivation_appearance=0.7,
        max_center_distance_m=0.4,
    )
    branch = build_causal_dynamic_branch(cfg, scene_id="scene0004_00")
    feature = [1.0, 0.0, 0.0]
    _process(
        branch,
        0,
        _Instances([1], [0.8], [_box(0.0)]),
        [[1]],
        appearance_by_init_id={1: feature},
    )
    old_id = branch.state.current_snapshots()[0].track_id
    empty = _Instances([], [], np.empty((0, 6)))
    _process(branch, 25, empty, [[1]], depth=np.full((5, 5), 5.0))
    _process(branch, 50, empty, [[1]], depth=np.full((5, 5), 5.0))
    result = _process(
        branch,
        75,
        _Instances([9], [0.9], [_box(2.0)]),
        [[1], [9]],
        appearance_by_init_id={9: feature},
    )
    assert result.reactivated_track_ids == (old_id,)
    assert result.relocated_track_ids == (old_id,)

    terminal = _Instances([1, 9], [0.8, 0.9], [_box(0.0), _box(2.0)])
    persistent = branch.materialize_persistent(terminal, [[1], [9]])
    current = branch.materialize_current(terminal, [[1], [9]])
    assert persistent.mask.tolist() == [True, False]
    assert persistent.dropped_rows == (1,)
    assert persistent.modified_rows == (0,)
    assert persistent.corners[0, :, 0].mean() == pytest.approx(0.0)
    assert current.mask.tolist() == [False, True]
    assert current.dropped_rows == (0,)
    assert current.modified_rows == (1,)
    branch.close()


def test_event_log_is_streamed_and_summary_is_bounded(tmp_path):
    branch = build_causal_dynamic_branch(_cfg(tmp_path), scene_id="scene0005_00")
    _process(branch, 0, _Instances([1], [0.8], [_box(0.0)]), [[1]])
    event_path = tmp_path / "events" / "scene0005_00.jsonl"
    assert not dynamic_event_ledger_complete(event_path, scene_id="scene0005_00")
    summary = branch.close()
    assert summary["keyframes"] == 1
    assert summary["event_records"] == 1
    rows = event_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(rows) == 2
    assert '"query_before_commit": true' in rows[0]
    assert '"type": "summary"' in rows[1]
    assert dynamic_event_ledger_complete(event_path, scene_id="scene0005_00")
