import ast
import inspect
from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from boxfusion.moon_qim_lite import QIMCandidate, QIMQueryBatch
from boxfusion.mv3dis_depth_lite import (
    DepthGuideProjectionMetrics,
    MV3DISDepthLiteObserver,
    build_mv3dis_depth_lite,
    derive_committed_track_ids,
    resolve_mv3dis_depth_lite_config,
)


def enabled_config(**overrides):
    values = {"enabled": True, "observer_only": True}
    values.update(overrides)
    return values


def candidate(track_id, *, active=True):
    return QIMCandidate(
        track_id=track_id,
        shared_key_count=3,
        shared_key_fraction=1.0,
        center_distance_m=0.0,
        aabb_iou=1.0,
        age_keyframes=0,
        active_at_last_commit=active,
    )


def qim_batch(
    frame_id,
    proposal_ids=(10,),
    rows=((),),
    *,
    scene_id="scene",
    history_marker="auto",
):
    history = (
        (None if frame_id == 0 else frame_id - 1)
        if history_marker == "auto"
        else history_marker
    )
    return QIMQueryBatch(
        scene_id=scene_id,
        frame_id=frame_id,
        proposal_ids=tuple(proposal_ids),
        candidates=tuple(tuple(row) for row in rows),
        history_max_frame_id=history,
        query_ms=0.0,
    )


def points(offset=0.0, count=4):
    result = np.zeros((count, 3), dtype=np.float64)
    result[:, 0] = np.linspace(offset, offset + 0.1, count)
    result[:, 2] = 1.0
    return result


def transform(frame_id):
    result = np.eye(4, dtype=np.float64)
    result[0, 3] = frame_id
    return result


def query(
    observer,
    frame_id,
    *,
    proposal_ids=(10,),
    rows=((),),
    scene_id="scene",
    boxes=None,
    point_rows=None,
    history_marker="auto",
):
    if point_rows is None:
        point_rows = tuple(points(index) for index in range(len(proposal_ids)))
    if boxes is None:
        boxes = tuple((0.0, 0.0, 2.0, 2.0) for _ in proposal_ids)
    return observer.query(
        qim_batch=qim_batch(
            frame_id,
            proposal_ids,
            rows,
            scene_id=scene_id,
            history_marker=history_marker,
        ),
        proposal_points_world=point_rows,
        depth_m=np.ones((4, 4), dtype=np.float64),
        K=np.eye(3, dtype=np.float64),
        T_wc=transform(frame_id),
        proposal_boxes_xyxy=boxes,
    )


class FakeGeometry:
    def __init__(self, metric_for=None):
        self.calls = []
        self.metric_for = metric_for

    def __call__(
        self,
        points_world,
        depth_m,
        K,
        T_wc,
        proposal_box_xyxy=None,
        alpha=0.05,
    ):
        frame_id = int(round(float(T_wc[0, 3])))
        self.calls.append(
            (
                frame_id,
                len(points_world),
                proposal_box_xyxy,
                alpha,
            )
        )
        if self.metric_for is not None:
            custom = self.metric_for(
                frame_id, proposal_box_xyxy, points_world
            )
            if custom is not None:
                return custom
        return DepthGuideProjectionMetrics(
            visibility=0.8,
            depth_consistency=0.9,
            quality=0.85,
            frame_visibility=0.8,
            box_visibility=0.95,
            box_depth_consistency=0.9,
            affinity=1.0,
        )


def commit(observer, batch, track_ids=(7,), native=None):
    observer.commit(
        batch,
        committed_track_ids=track_ids,
        native_target_track_ids=native,
    )


def seed_two_views(observer, track_id=7, *, scene_id="scene", boxes=None):
    first = query(observer, 0, scene_id=scene_id, boxes=boxes)
    commit(observer, first, (track_id,))
    second = query(
        observer,
        1,
        rows=((candidate(track_id),),),
        scene_id=scene_id,
        boxes=boxes,
    )
    commit(observer, second, (track_id,))


def test_config_default_off_strict_shadow_caps_and_frozen_thresholds():
    resolved = resolve_mv3dis_depth_lite_config()
    assert resolved["enabled"] is False
    assert resolved["observer_only"] is True
    assert resolved["max_guides_per_track"] == 5
    assert resolved["max_depth_frames"] == 80
    assert resolved["max_proposals"] == 256
    assert resolved["projection_budget_points"] == 8192
    assert resolved["points_per_projection"] == 64
    assert build_mv3dis_depth_lite({}).enabled is False
    with pytest.raises(ValueError, match="Unknown mv3dis_depth_lite"):
        resolve_mv3dis_depth_lite_config({"typo": 1})
    with pytest.raises(ValueError, match="not authorized"):
        resolve_mv3dis_depth_lite_config(
            {"enabled": True, "observer_only": False}
        )
    with pytest.raises(ValueError, match="frozen S0 thresholds"):
        resolve_mv3dis_depth_lite_config(
            {"enabled": True, "box_visibility_threshold": 0.89}
        )
    with pytest.raises(ValueError, match="must not exceed 5"):
        resolve_mv3dis_depth_lite_config({"max_guides_per_track": 6})
    with pytest.raises(ValueError, match="must not exceed 80"):
        resolve_mv3dis_depth_lite_config({"max_depth_frames": 81})
    with pytest.raises(ValueError, match="must not exceed 256"):
        resolve_mv3dis_depth_lite_config({"max_proposals": 257})
    with pytest.raises(ValueError, match="must not exceed 8192"):
        resolve_mv3dis_depth_lite_config({"projection_budget_points": 8193})
    with pytest.raises(ValueError, match="frozen S0 thresholds"):
        resolve_mv3dis_depth_lite_config(
            {"enabled": True, "points_per_projection": 32}
        )


def test_source_guide_quality_is_recorded_only_and_outputs_are_frozen_scalars():
    geometry = FakeGeometry()
    observer = MV3DISDepthLiteObserver(
        enabled_config(), projection_adapter=geometry
    )
    batch = query(observer, 0)
    quality = batch.guide_quality_rows[0]
    veto = batch.birth_veto_rows[0]
    assert quality.valid is True
    assert quality.visibility == pytest.approx(0.8)
    assert quality.depth_consistency == pytest.approx(0.9)
    assert quality.guide_quality == pytest.approx(0.85)
    assert quality.reason == "record_only"
    assert veto.would_veto_birth is False
    assert veto.action == "defer_to_native"
    assert batch.history_max_frame_id is None
    with pytest.raises(FrozenInstanceError):
        quality.guide_quality = 1.0
    with pytest.raises(FrozenInstanceError):
        batch.frame_id = 9
    assert not any(isinstance(value, np.ndarray) for value in vars(batch).values())
    summary = observer.summary()
    assert summary["guide_quality_computed"] is True
    assert summary["fusion_weights_computed"] is False
    assert summary["fusion_weights_applied"] is False
    assert summary["birth_veto_applied"] is False
    assert summary["native_outputs_mutated"] is False


def test_default_geometry_field_names_are_explicitly_adapted():
    class HelperStyleMetrics:
        v_f = 0.7
        d_f = 0.8
        q_f = 0.56
        v_b = 0.95
        d_b = 0.9
        affinity_a = 0.855

    class HelperStyleAdapter:
        def __call__(self, *args, **kwargs):
            return HelperStyleMetrics()

    observer = MV3DISDepthLiteObserver(
        enabled_config(), projection_adapter=HelperStyleAdapter()
    )
    batch = query(observer, 0)
    row = batch.guide_quality_rows[0]
    assert row.visibility == pytest.approx(0.7)
    assert row.depth_consistency == pytest.approx(0.8)
    assert row.guide_quality == pytest.approx(0.56)


def test_current_guide_is_not_history_until_commit_and_two_prior_views_veto():
    observer = MV3DISDepthLiteObserver(
        enabled_config(), projection_adapter=FakeGeometry()
    )
    first = query(observer, 0)
    assert first.birth_veto_rows[0].candidates == ()
    assert observer.snapshot().total_guides == 0
    commit(observer, first, (7,))
    assert observer.snapshot().total_guides == 1

    second = query(observer, 1, rows=((candidate(7),),))
    one_view = second.birth_veto_rows[0]
    assert one_view.candidates[0].history_views_available == 1
    assert one_view.candidates[0].supporting_views == 1
    assert one_view.would_veto_birth is False
    commit(observer, second, (7,))

    third = query(observer, 2, rows=((candidate(7),),))
    two_view = third.birth_veto_rows[0]
    assert two_view.candidates[0].history_views_available == 2
    assert two_view.candidates[0].supporting_views == 2
    assert two_view.candidate_dominance == pytest.approx(1.0)
    assert two_view.would_veto_birth is True
    assert two_view.recommended_track_id == 7
    assert two_view.action == "defer_to_native"


def test_empty_current_guide_can_query_history_but_is_not_committed():
    observer = MV3DISDepthLiteObserver(
        enabled_config(), projection_adapter=FakeGeometry()
    )
    seed_two_views(observer, track_id=7)
    before = observer.snapshot().total_guides
    batch = query(
        observer,
        2,
        rows=((candidate(7),),),
        point_rows=(np.empty((0, 3), dtype=np.float64),),
    )
    assert batch.guide_quality_rows[0].valid is False
    assert batch.guide_quality_rows[0].reason == "empty_guide_points"
    assert batch.birth_veto_rows[0].would_veto_birth is True
    commit(observer, batch, (7,))
    assert observer.snapshot().total_guides == before


@pytest.mark.parametrize(
    "frame_visibility,box_visibility",
    [(0.30, 0.95), (0.80, 0.90)],
)
def test_visibility_thresholds_are_strict(frame_visibility, box_visibility):
    def metrics(_, __, ___):
        return DepthGuideProjectionMetrics(
            visibility=0.8,
            depth_consistency=0.9,
            quality=0.85,
            frame_visibility=frame_visibility,
            box_visibility=box_visibility,
            box_depth_consistency=0.9,
            affinity=1.0,
        )

    observer = MV3DISDepthLiteObserver(
        enabled_config(), projection_adapter=FakeGeometry(metrics)
    )
    seed_two_views(observer)
    batch = query(observer, 2, rows=((candidate(7),),))
    row = batch.birth_veto_rows[0]
    assert row.candidates[0].supporting_views == 0
    assert row.would_veto_birth is False


def test_unique_candidate_dominance_must_be_strictly_above_point_nine():
    # Historical guide points encode the candidate.  Track 7 has two views;
    # track 8 has one, and contributes exactly 2/9 of track 7's score.
    def metric_for(_, box, guide_points):
        affinity = (
            1.0
            if float(np.mean(guide_points[:, 0])) < 0.5
            else 2.0 / 9.0
        )
        return DepthGuideProjectionMetrics(
            visibility=0.8,
            depth_consistency=0.9,
            quality=0.85,
            frame_visibility=0.8,
            box_visibility=0.95,
            box_depth_consistency=0.9,
            affinity=affinity,
        )

    observer = MV3DISDepthLiteObserver(
        enabled_config(), projection_adapter=FakeGeometry(metric_for)
    )
    first = query(
        observer,
        0,
        proposal_ids=(10, 11),
        rows=((), ()),
        boxes=((0.0, 0.0, 2.0, 2.0), (10.0, 0.0, 12.0, 2.0)),
    )
    commit(observer, first, (7, 8))
    second = query(
        observer,
        1,
        rows=((candidate(7),),),
        boxes=((0.0, 0.0, 2.0, 2.0),),
    )
    commit(observer, second, (7,))

    boundary = query(
        observer,
        2,
        rows=((candidate(7), candidate(8)),),
    )
    row = boundary.birth_veto_rows[0]
    assert row.candidate_dominance == pytest.approx(0.9)
    assert row.would_veto_birth is False
    assert row.reason == "low_dominance_defer_to_native"


def test_multiple_two_view_candidates_abstain_even_with_dominant_winner():
    def metric_for(_, box, guide_points):
        affinity = (
            1.0 if float(np.mean(guide_points[:, 0])) < 0.5 else 0.01
        )
        return DepthGuideProjectionMetrics(
            visibility=0.8,
            depth_consistency=0.9,
            quality=0.85,
            frame_visibility=0.8,
            box_visibility=0.95,
            box_depth_consistency=0.9,
            affinity=affinity,
        )

    observer = MV3DISDepthLiteObserver(
        enabled_config(), projection_adapter=FakeGeometry(metric_for)
    )
    for frame_id in (0, 1):
        batch = query(
            observer,
            frame_id,
            proposal_ids=(10, 11),
            rows=((), ()),
            boxes=((0.0, 0.0, 2.0, 2.0), (10.0, 0.0, 12.0, 2.0)),
        )
        commit(observer, batch, (7, 8))
    batch = query(
        observer, 2, rows=((candidate(7), candidate(8)),)
    )
    row = batch.birth_veto_rows[0]
    assert row.candidate_dominance > 0.9
    assert [item.supporting_views for item in row.candidates] == [2, 2]
    assert row.would_veto_birth is False
    assert row.reason == "nonunique_candidate_defer_to_native"


def test_qim_candidate_set_is_deduplicated_and_capped_at_three():
    observer = MV3DISDepthLiteObserver(
        enabled_config(), projection_adapter=FakeGeometry()
    )
    batch = query(
        observer,
        0,
        rows=((candidate(5), candidate(5), candidate(6), candidate(7), candidate(8)),),
    )
    assert [row.track_id for row in batch.birth_veto_rows[0].candidates] == [5, 6, 7]


def test_stale_qim_candidates_are_never_projected_or_vetoed():
    observer = MV3DISDepthLiteObserver(
        enabled_config(), projection_adapter=FakeGeometry()
    )
    seed_two_views(observer, track_id=7)
    batch = query(observer, 2, rows=((candidate(7, active=False),),))
    row = batch.birth_veto_rows[0]
    assert row.candidates == ()
    assert row.would_veto_birth is False
    assert row.reason == "no_qim_candidates_defer_to_native"


def test_proposal_cap_is_zero_work_fail_closed_and_does_not_enter_history():
    geometry = FakeGeometry()
    observer = MV3DISDepthLiteObserver(
        enabled_config(max_proposals=1), projection_adapter=geometry
    )
    batch = query(
        observer,
        0,
        proposal_ids=(10, 11),
        rows=((), ()),
    )
    assert batch.proposal_cap_exceeded is True
    assert geometry.calls == []
    assert all(not row.valid for row in batch.guide_quality_rows)
    assert all(not row.would_veto_birth for row in batch.birth_veto_rows)
    commit(observer, batch, (7, 8))
    assert observer.snapshot().total_guides == 0
    assert observer.summary()["proposal_cap_batches"] == 1


@pytest.mark.parametrize(
    "depth,K_value,T_value,reason",
    [
        (
            np.zeros((10, 10), dtype=np.float32),
            np.eye(3),
            np.eye(4),
            "insufficient_valid_depth",
        ),
        (
            np.ones((10, 10), dtype=np.float32),
            np.asarray([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
            np.eye(4),
            "invalid_intrinsics",
        ),
        (
            np.ones((10, 10), dtype=np.float32),
            np.eye(3),
            np.diag([2.0, 1.0, 1.0, 1.0]),
            "invalid_pose",
        ),
    ],
)
def test_invalid_current_frame_abstains_batchwide_and_commits_no_guide(
    depth, K_value, T_value, reason
):
    geometry = FakeGeometry()
    observer = MV3DISDepthLiteObserver(
        enabled_config(), projection_adapter=geometry
    )
    batch = observer.query(
        qim_batch=qim_batch(0),
        proposal_points_world=(points(),),
        depth_m=depth,
        K=K_value,
        T_wc=T_value,
        proposal_boxes_xyxy=((0.0, 0.0, 2.0, 2.0),),
    )
    assert batch.current_frame_valid is False
    assert batch.invalid_frame_reason == reason
    assert geometry.calls == []
    assert batch.guide_quality_rows[0].reason == f"{reason}_defer_to_native"
    assert batch.birth_veto_rows[0].would_veto_birth is False
    commit(observer, batch, (7,))
    assert observer.snapshot().total_guides == 0
    assert observer.snapshot().committed_frame_ids == ()
    summary = observer.summary()
    assert summary["invalid_frame_batches"] == 1
    assert summary["invalid_frame_reasons"] == ((reason, 1),)


def test_exactly_one_percent_valid_depth_is_accepted_and_pending_depth_is_float32():
    class DtypeGeometry(FakeGeometry):
        def __call__(self, points_world, depth_m, *args, **kwargs):
            assert depth_m.dtype == np.float32
            return super().__call__(points_world, depth_m, *args, **kwargs)

    depth = np.zeros((10, 10), dtype=np.float64)
    depth[0, 0] = 1.0
    observer = MV3DISDepthLiteObserver(
        enabled_config(), projection_adapter=DtypeGeometry()
    )
    batch = observer.query(
        qim_batch=qim_batch(0),
        proposal_points_world=(points(),),
        depth_m=depth,
        K=np.eye(3),
        T_wc=np.eye(4),
        proposal_boxes_xyxy=((0.0, 0.0, 2.0, 2.0),),
    )
    assert batch.current_frame_valid is True


def test_global_projection_budget_is_never_exceeded_and_partial_row_abstains():
    observer = MV3DISDepthLiteObserver(
        enabled_config(
            projection_budget_points=2,
        ),
        projection_adapter=FakeGeometry(),
    )
    seed_two_views(observer)
    batch = query(observer, 2, rows=((candidate(7),),))
    row = batch.birth_veto_rows[0]
    assert batch.guide_quality_projection_points_used == 2
    assert batch.birth_veto_projection_points_used == 2
    assert batch.guide_quality_budget_exhausted is True
    assert batch.birth_veto_budget_exhausted is True
    assert row.candidates[0].history_views_available == 2
    assert row.candidates[0].history_views_evaluated == 1
    assert row.candidates[0].complete is False
    assert row.would_veto_birth is False
    assert row.reason == "incomplete_projection_defer_to_native"


def test_per_track_and_depth_frame_histories_are_bounded_and_causal():
    observer = MV3DISDepthLiteObserver(
        enabled_config(max_guides_per_track=2, max_depth_frames=2),
        projection_adapter=FakeGeometry(),
    )
    for frame_id in range(4):
        batch = query(
            observer,
            frame_id,
            rows=((candidate(7),),) if frame_id else ((),),
        )
        commit(observer, batch, (7,))
    snapshot = observer.snapshot()
    assert snapshot.committed_frame_ids == (2, 3)
    assert snapshot.track_guide_counts == ((7, 2),)
    assert snapshot.total_guides == 2
    summary = observer.summary()
    assert summary["max_committed_frames_observed"] <= 2
    assert summary["guides_evicted_track_cap"] >= 1
    assert summary["committed_frames_evicted"] == 2


def test_duplicate_same_frame_track_mapping_keeps_one_best_guide():
    def metric_for(_, box, guide_points):
        quality = 0.9 if box[0] == 0.0 else 0.4
        return DepthGuideProjectionMetrics(
            visibility=0.8,
            depth_consistency=0.9,
            quality=quality,
            frame_visibility=0.8,
            box_visibility=0.95,
            box_depth_consistency=0.9,
            affinity=1.0,
        )

    observer = MV3DISDepthLiteObserver(
        enabled_config(), projection_adapter=FakeGeometry(metric_for)
    )
    batch = query(
        observer,
        0,
        proposal_ids=(10, 11),
        rows=((), ()),
        boxes=((0.0, 0.0, 2.0, 2.0), (10.0, 0.0, 12.0, 2.0)),
    )
    commit(observer, batch, (7, 7))
    assert observer.snapshot().track_guide_counts == ((7, 1),)


def test_query_commit_transaction_cross_scene_and_future_history_fail_closed():
    observer = MV3DISDepthLiteObserver(
        enabled_config(), projection_adapter=FakeGeometry()
    )
    batch = query(observer, 0)
    with pytest.raises(ValueError, match="closed by commit"):
        query(observer, 1)
    with pytest.raises(ValueError, match="pending depth batch"):
        observer.commit(
            type(batch)(**{**vars(batch), "query_ms": 123.0}),
            committed_track_ids=(7,),
        )
    with pytest.raises(ValueError, match="align with proposals"):
        commit(observer, batch, ())
    assert observer.snapshot().pending_frame_id == 0
    commit(observer, batch, (7,))
    with pytest.raises(ValueError, match="already committed"):
        commit(observer, batch, (7,))
    with pytest.raises(ValueError, match="strictly increasing"):
        query(observer, 0)
    with pytest.raises(ValueError, match="bound to scene"):
        query(observer, 1, scene_id="other")

    fresh = MV3DISDepthLiteObserver(
        enabled_config(), projection_adapter=FakeGeometry()
    )
    with pytest.raises(ValueError, match="QIM history must precede"):
        query(fresh, 4, history_marker=4)


def test_native_diagnostics_do_not_feed_back_into_later_queries():
    left = MV3DISDepthLiteObserver(
        enabled_config(), projection_adapter=FakeGeometry()
    )
    right = MV3DISDepthLiteObserver(
        enabled_config(), projection_adapter=FakeGeometry()
    )
    for observer in (left, right):
        seed_two_views(observer)
    left_batch = query(left, 2, rows=((candidate(7),),))
    right_batch = query(right, 2, rows=((candidate(7),),))
    assert left_batch.birth_veto_rows == right_batch.birth_veto_rows
    commit(left, left_batch, (7,), native=((7,),))
    commit(right, right_batch, (7,), native=((),))

    next_left = query(left, 3, rows=((candidate(7),),))
    next_right = query(right, 3, rows=((candidate(7),),))
    assert next_left.guide_quality_rows == next_right.guide_quality_rows
    assert next_left.birth_veto_rows == next_right.birth_veto_rows
    assert left.summary()["veto_correct"] == 1
    assert right.summary()["veto_wrong"] == 1
    assert right.summary()["veto_on_native_birth"] == 1
    diagnostic = left.summary()["diagnostic_examples"][-1]
    assert diagnostic[:3] == ("scene", 2, 10)
    assert diagnostic[3] == (7,)
    assert diagnostic[4] is True
    assert diagnostic[5] == 7
    candidate_row = diagnostic[8][0]
    assert candidate_row[0] == 7
    assert len(candidate_row[7]) == 2
    assert all(view[1] is True for view in candidate_row[7])
    assert all(view[2:6] == pytest.approx((0.8, 0.95, 0.9, 1.0)) for view in candidate_row[7])


def test_known_event_can_be_joined_offline_by_key_without_runtime_label():
    observer = MV3DISDepthLiteObserver(
        enabled_config(), projection_adapter=FakeGeometry()
    )
    seed_two_views(observer, track_id=15, scene_id="scene0598_01")
    # Strictly increasing frames are sufficient; history need not be dense.
    batch = query(
        observer,
        175,
        proposal_ids=(18,),
        rows=((candidate(15),),),
        scene_id="scene0598_01",
        history_marker=174,
    )
    assert (batch.scene_id, batch.frame_id, batch.proposal_ids[0]) == (
        "scene0598_01",
        175,
        18,
    )
    row = batch.birth_veto_rows[0]
    assert row.recommended_track_id == 15
    assert row.candidates[0].supporting_views == 2
    assert row.would_veto_birth is True
    summary = observer.summary()
    assert summary["hardcoded_scene_event_access"] is False
    source = inspect.getsource(inspect.getmodule(MV3DISDepthLiteObserver))
    assert "scene0598_01" not in source


def test_bad_or_missing_geometry_adapter_abstains_without_crashing_pipeline():
    class BrokenGeometry:
        def __call__(self, *args, **kwargs):
            raise RuntimeError("synthetic failure")

    observer = MV3DISDepthLiteObserver(
        enabled_config(), projection_adapter=BrokenGeometry()
    )
    first = query(observer, 0)
    assert first.guide_quality_rows[0].valid is False
    assert first.guide_quality_rows[0].reason == "geometry_error:RuntimeError"
    commit(observer, first, (7,))
    second = query(observer, 1, rows=((candidate(7),),))
    assert second.birth_veto_rows[0].would_veto_birth is False
    assert second.birth_veto_rows[0].candidates[0].history_views_available == 0
    assert observer.snapshot().total_guides == 0
    assert observer.summary()["geometry_errors"] == 2


def test_summary_timing_safety_contract_and_no_puf_import():
    observer = MV3DISDepthLiteObserver(
        enabled_config(), projection_adapter=FakeGeometry()
    )
    batch = query(observer, 0)
    commit(observer, batch, (7,), native=(None,))
    observer.record_pipeline_timing(query_ms=0.2, commit_ms=0.1)
    summary = observer.summary()
    assert summary["schema"] == "boxfusion.mv3dis_depth_lite_s0_shadow.v1"
    assert summary["active_authorized"] is False
    assert summary["training_free"] is True
    assert summary["unsupervised"] is True
    assert summary["causal"] is True
    assert summary["ground_truth_access"] is False
    assert summary["semantic_access"] is False
    assert summary["semantic_mutation"] is False
    assert summary["detector_score_access"] is False
    assert summary["puf_access"] is False
    assert summary["online_parameter_update"] is False
    assert summary["native_unresolved"] == 1
    assert summary["pipeline_query_ms_mean"] == pytest.approx(0.2)
    assert summary["pipeline_commit_ms_mean"] == pytest.approx(0.1)
    assert "MV3DIS-Depth-lite S0 shadow summary" in observer.summary_line()

    tree = ast.parse(inspect.getsource(inspect.getmodule(MV3DISDepthLiteObserver)))
    imported_modules = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    } | {
        node.module or ""
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
    }
    assert not any("puf" in name.lower() for name in imported_modules)


def test_derive_committed_ids_uses_final_row_and_five_view_merge_event():
    result = derive_committed_track_ids(
        proposal_ids=np.asarray([100, 101, 102]),
        current_fusion_groups=((1, 2, 3, 4, 5), (101,)),
        current_stable_ids=np.asarray([77, 88]),
        association_events=(
            {
                "stage": "spatial",
                "winner_members": (1, 2, 3, 4, 5),
                "loser_members": (100,),
            },
        ),
    )
    assert result == (77, 88, None)


def test_derive_committed_ids_handles_chained_new_new_then_old_merge():
    result = derive_committed_track_ids(
        proposal_ids=np.asarray([100, 101]),
        current_fusion_groups=((1, 2, 3, 4, 5),),
        current_stable_ids=np.asarray([77]),
        association_events=(
            {
                "stage": "spatial",
                "winner_members": (100,),
                "loser_members": (101,),
            },
            {
                "stage": "correspondence",
                "winner_members": (1, 2, 3, 4, 5),
                "loser_members": (100,),
            },
        ),
    )
    assert result == (77, 77)


def test_derive_committed_ids_rejects_ambiguous_multirow_component():
    result = derive_committed_track_ids(
        proposal_ids=np.asarray([100]),
        current_fusion_groups=((1,), (2,)),
        current_stable_ids=np.asarray([77, 88]),
        association_events=(
            {
                "stage": "spatial",
                "winner_members": (1,),
                "loser_members": (100,),
            },
            {
                "stage": "correspondence",
                "winner_members": (2,),
                "loser_members": (100,),
            },
        ),
    )
    assert result == (None,)


def test_derive_committed_ids_fails_closed_without_evidence_and_validates():
    assert derive_committed_track_ids(
        proposal_ids=np.asarray([100]),
        current_fusion_groups=((1,),),
        current_stable_ids=np.asarray([77]),
    ) == (None,)
    with pytest.raises(ValueError, match="unique and non-negative"):
        derive_committed_track_ids(
            proposal_ids=np.asarray([100, 100]),
            current_fusion_groups=((1,),),
            current_stable_ids=np.asarray([77]),
        )
    with pytest.raises(ValueError, match="unknown keys"):
        derive_committed_track_ids(
            proposal_ids=np.asarray([100]),
            current_fusion_groups=((1,),),
            current_stable_ids=np.asarray([77]),
            association_events=(
                {
                    "winner_members": (1,),
                    "loser_members": (100,),
                    "target": 77,
                },
            ),
        )
