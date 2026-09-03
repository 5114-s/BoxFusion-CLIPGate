from types import SimpleNamespace

import numpy as np
import pytest

from boxfusion.cutr_residual_birth_lite import (
    ResidualAssignment,
    ResidualCandidate,
    ResidualCloseResult,
    ResidualKeyframeResult,
)
from boxfusion.cutr_residual_cross_view_r1 import (
    CuTRResidualCrossViewR1,
    ResidualCrossViewConfig,
    ResidualCrossViewEvidence,
    build_cutr_residual_cross_view_r1,
)


def cube(center=(0.0, 0.0, 0.0), extent=(1.0, 1.0, 1.0)):
    center = np.asarray(center, dtype=np.float64)
    half = 0.5 * np.asarray(extent, dtype=np.float64)
    signs = np.asarray(
        [
            [-1, -1, -1], [-1, -1, 1], [-1, 1, -1], [-1, 1, 1],
            [1, -1, -1], [1, -1, 1], [1, 1, -1], [1, 1, 1],
        ],
        dtype=np.float64,
    )
    return center + signs * half


def pose(x=0.0, degrees=0.0):
    angle = np.radians(degrees)
    c, s = np.cos(angle), np.sin(angle)
    value = np.eye(4, dtype=np.float64)
    value[:3, :3] = [[c, -s, 0], [s, c, 0], [0, 0, 1]]
    value[0, 3] = x
    return value


def descriptor(axis=0, cosine=None):
    value = np.zeros(256, dtype=np.float64)
    if cosine is None:
        value[axis] = 1.0
    else:
        value[0] = cosine
        value[1] = np.sqrt(1.0 - cosine**2)
    return value


def guide(frame=0, points=64):
    rows = np.zeros((points, 3), dtype=np.float64)
    rows[:, 0] = float(frame)
    rows[:, 1] = np.linspace(-0.1, 0.1, points)
    rows[:, 2] = 2.0
    return rows


def evidence(frame, raw=0, track_pose=None, desc=None, points=64):
    return ResidualCrossViewEvidence(
        frame_id=frame,
        raw_index=raw,
        descriptor=descriptor() if desc is None else desc,
        camera_to_world=pose(frame) if track_pose is None else track_pose,
        raw_box_xyxy=(10.0, 10.0, 100.0, 100.0),
        guide_points_world=guide(frame, points),
    )


def base_result(frame, rows, *, retired=(), audit=True):
    assignments = tuple(
        ResidualAssignment(raw_index=raw, track_id=track, action=action)
        for raw, track, action in rows
    )
    return ResidualKeyframeResult(
        frame_id=frame,
        accepted_raw_indices=tuple(row.raw_index for row in assignments),
        assignments=assignments,
        matched_track_ids=tuple(
            row.track_id for row in assignments if row.action == "matched"
        ),
        created_track_ids=tuple(
            row.track_id for row in assignments if row.action == "created"
        ),
        newly_confirmed_track_ids=(),
        newly_retired_track_ids=tuple(retired),
        duplicate_dropped_raw_indices=(),
        capacity_dropped_raw_indices=(),
        track_capacity_dropped_raw_indices=(),
        active_track_ids=tuple(row.track_id for row in assignments),
        audit_complete=audit,
    )


def projection(vf=1.0, vb=1.0, df=1.0, db=1.0):
    return SimpleNamespace(
        v_f=vf,
        d_f=df,
        v_b=vb,
        d_b=db,
        affinity_a=vb * db,
    )


def observer(adapter=lambda *args, **kwargs: projection()):
    return CuTRResidualCrossViewR1(
        ResidualCrossViewConfig(enabled=True), projection_adapter=adapter
    )


def observe(target, frame, rows, assignments, **kwargs):
    p = kwargs.pop("current_pose", pose(frame))
    return target.observe(
        frame_id=frame,
        evidence_rows=rows,
        base_result=base_result(frame, assignments, **kwargs),
        depth_m=np.full((2, 2), float(frame + 1)),
        K=np.eye(3),
        T_wc=p,
    )


def candidate(track_id):
    return ResidualCandidate(
        track_id=track_id,
        corners=cube(),
        raw_mean_score=0.2,
        appended_score=0.1,
        evidence_frame_ids=(0, 1, 2),
        evidence_raw_indices=(0, 0, 0),
        median_pairwise_iou=1.0,
        center_rms_m=0.0,
        max_native_iou=0.0,
    )


def base_close(*track_ids, audit=True):
    rows = tuple(candidate(track) for track in track_ids)
    return ResidualCloseResult(
        candidates=rows,
        eligible_track_ids=tuple(track_ids),
        unstable_track_ids=(),
        too_small_track_ids=(),
        native_overlap_rejected_track_ids=(),
        self_nms_rejected_track_ids=(),
        output_cap_rejected_track_ids=(),
        audit_complete=audit,
    )


def test_config_is_default_off_strict_and_requires_base_observer():
    assert build_cutr_residual_cross_view_r1({}).enabled is False
    root = {
        "cutr_residual_birth_lite": {"enabled": True},
        "cutr_residual_cross_view_r1": {"enabled": True},
    }
    assert build_cutr_residual_cross_view_r1(root).enabled is True
    with pytest.raises(ValueError, match="requires enabled"):
        build_cutr_residual_cross_view_r1(
            {"cutr_residual_cross_view_r1": {"enabled": True}}
        )
    with pytest.raises(ValueError, match="frozen"):
        ResidualCrossViewConfig(enabled=True, descriptor_cosine=0.79)
    with pytest.raises(ValueError, match="CuTR-native"):
        build_cutr_residual_cross_view_r1(
            {**root, "lifting": {"backend": "boxer"}}
        )


def test_evidence_normalizes_descriptor_and_is_strongly_immutable():
    source = descriptor() * 17.0
    row = evidence(0, desc=source)
    source[:] = 0.0
    assert np.linalg.norm(row.descriptor) == pytest.approx(1.0)
    for value in (row.descriptor, row.camera_to_world, row.guide_points_world):
        assert value.flags.writeable is False
        with pytest.raises(ValueError):
            value.setflags(write=True)
    invalid = ResidualCrossViewEvidence.abstain(0, 1, "no_depth_guide")
    assert invalid.valid is False
    with pytest.raises(ValueError, match="all present or all absent"):
        ResidualCrossViewEvidence(0, 0, descriptor=descriptor(), reason="bad")


def test_mixed_create_match_assignments_remain_row_aligned():
    target = observer()
    rows = [evidence(0, 7, pose(0)), evidence(0, 3, pose(0))]
    result = observe(
        target,
        0,
        rows,
        [(7, 10, "created"), (3, 2, "matched")],
        current_pose=pose(0),
    )
    assert result.assignment_count == 2
    with pytest.raises(ValueError, match="align"):
        observer().observe(
            frame_id=0,
            evidence_rows=rows[::-1],
            base_result=base_result(
                0, [(7, 10, "created"), (3, 2, "matched")]
            ),
            depth_m=np.ones((2, 2)), K=np.eye(3), T_wc=pose(0),
        )


def test_history_to_current_only_confirms_three_node_component():
    calls = []

    def adapter(points, depth, K, T, **kwargs):
        calls.append((float(points[0, 0]), float(depth[0, 0])))
        return projection()

    target = observer(adapter)
    observe(target, 0, [evidence(0, track_pose=pose(0))], [(0, 7, "created")], current_pose=pose(0))
    second = observe(target, 1, [evidence(1, track_pose=pose(1))], [(0, 7, "matched")], current_pose=pose(1))
    third = observe(target, 2, [evidence(2, track_pose=pose(2))], [(0, 7, "matched")], current_pose=pose(2))
    assert second.newly_confirmed_track_ids == ()
    assert third.newly_confirmed_track_ids == (7,)
    # No source-frame self projection: 0->1, then 0->2 and 1->2.
    assert calls == [(0.0, 2.0), (0.0, 3.0), (1.0, 3.0)]
    assert target.summary()["receipt_count"] == 1


def test_pose_descriptor_and_visibility_thresholds_are_exact():
    cases = [
        (pose(0.8), descriptor(), projection(), False),
        (pose(0.0, 30.0), descriptor(), projection(), False),
        (pose(0.81), descriptor(cosine=0.799), projection(), False),
        (pose(0.81), descriptor(cosine=0.8), projection(vf=0.30), False),
        (pose(0.81), descriptor(cosine=0.8), projection(vb=0.90), False),
        (pose(0.81), descriptor(cosine=0.8), projection(), True),
    ]
    for current_pose, current_desc, metrics, should_support in cases:
        target = observer(lambda *args, value=metrics, **kwargs: value)
        observe(target, 0, [evidence(0, track_pose=pose(0))], [(0, 1, "created")], current_pose=pose(0))
        observe(target, 1, [evidence(1, track_pose=current_pose, desc=current_desc)], [(0, 1, "matched")], current_pose=current_pose)
        receipts = target.summary()["supporting_edges"]
        assert bool(receipts) is should_support


def test_disconnected_supported_edges_do_not_confirm():
    def adapter(points, depth, K, T, **kwargs):
        pair = (int(points[0, 0]), int(depth[0, 0] - 1))
        return projection() if pair in {(0, 1), (2, 3)} else projection(vf=0.0)

    target = observer(adapter)
    for frame in range(4):
        observe(
            target,
            frame,
            [evidence(frame, track_pose=pose(frame))],
            [(0, 9, "created" if frame == 0 else "matched")],
            current_pose=pose(frame),
        )
    summary = target.summary()
    assert summary["supporting_edges"] == 2
    assert summary["receipt_count"] == 0


def test_budget_preflight_is_row_atomic_and_current_node_still_commits():
    calls = []

    def adapter(points, *args, **kwargs):
        calls.append(int(points[0, 0]))
        return projection(vf=0.0)

    target = observer(adapter)
    track_count = 33
    for frame in range(5):
        rows = [
            evidence(frame, raw=track, track_pose=pose(frame))
            for track in range(track_count)
        ]
        assignments = [
            (track, track, "created" if frame == 0 else "matched")
            for track in range(track_count)
        ]
        result = observe(target, frame, rows, assignments, current_pose=pose(frame))
    # At frame four each row needs 4*64 points. Exactly 32 rows fit 8192;
    # the 33rd makes zero calls rather than a partial history projection.
    assert result.projection_points_used == 8192
    assert result.budget_abstained_raw_indices == (32,)
    assert len(calls[-128:]) == 128
    assert result.audit_complete is False
    assert target.summary()["active_track_histories"] == track_count


def test_projection_failure_discards_support_and_does_not_crash_native_path():
    target = observer(lambda *args, **kwargs: (_ for _ in ()).throw(ValueError("sensor")))
    observe(target, 0, [evidence(0, track_pose=pose(0))], [(0, 1, "created")], current_pose=pose(0))
    result = observe(target, 1, [evidence(1, track_pose=pose(1))], [(0, 1, "matched")], current_pose=pose(1))
    assert result.newly_confirmed_track_ids == ()
    assert result.projection_points_used == 64
    assert result.audit_complete is False
    assert target.summary()["projection_failures"] == 1
    assert target.summary()["projection_calls"] == 1


def test_multi_history_projection_failure_discards_successful_prefix_edges():
    calls = 0

    def adapter(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 3:  # frame2: first history succeeds, second fails.
            raise ValueError("second history invalid")
        return projection()

    target = observer(adapter)
    observe(target, 0, [evidence(0, track_pose=pose(0))], [(0, 1, "created")], current_pose=pose(0))
    observe(target, 1, [evidence(1, track_pose=pose(1))], [(0, 1, "matched")], current_pose=pose(1))
    result = observe(target, 2, [evidence(2, track_pose=pose(2))], [(0, 1, "matched")], current_pose=pose(2))
    assert result.newly_confirmed_track_ids == ()
    assert result.projection_points_used == 128
    assert result.audit_complete is False
    # Only the complete frame0->frame1 edge survives.
    assert target.summary()["supporting_edges"] == 1
    assert target.summary()["receipt_count"] == 0


def test_mapping_projection_missing_metric_is_a_charged_fail_closed_row():
    target = observer(lambda *args, **kwargs: {"v_f": 1.0})
    observe(
        target,
        0,
        [evidence(0, track_pose=pose(0))],
        [(0, 3, "created")],
        current_pose=pose(0),
    )
    result = observe(
        target,
        1,
        [evidence(1, track_pose=pose(1))],
        [(0, 3, "matched")],
        current_pose=pose(1),
    )
    assert result.projection_points_used == 64
    assert result.audit_complete is False
    assert target.summary()["projection_failures"] == 1


def test_receipt_is_sticky_across_bad_evidence_history_roll_and_retirement():
    target = observer()
    for frame in range(3):
        observe(
            target, frame, [evidence(frame, track_pose=pose(frame))],
            [(0, 4, "created" if frame == 0 else "matched")],
            current_pose=pose(frame),
        )
    frozen = target.summary()["receipts"]
    observe(
        target, 3,
        [ResidualCrossViewEvidence.abstain(3, 0, "missing_guide")],
        [(0, 4, "matched")], current_pose=pose(3),
    )
    observe(target, 4, [], [], retired=(4,), current_pose=pose(4))
    assert target.summary()["receipts"] == frozen
    assert target.summary()["active_track_histories"] == 0
    closed = target.close(base_close(4, 8))
    assert closed.admitted_track_ids == (4,)
    assert closed.rejected_track_ids == (8,)
    assert closed.candidates[0] is base_close(4).candidates[0] or closed.candidates[0].track_id == 4


def test_close_is_ordered_subset_and_never_creates_or_rescores_candidate():
    target = observer()
    for frame in range(3):
        observe(
            target, frame, [evidence(frame, track_pose=pose(frame))],
            [(0, 2, "created" if frame == 0 else "matched")],
            current_pose=pose(frame),
        )
    base = base_close(1, 2, 3)
    result = target.close(base)
    assert result.candidates == (base.candidates[1],)
    assert result.candidates[0].appended_score == base.candidates[1].appended_score
    assert result.active_authorized is False
    assert result.native_mutation_applied is False


def test_base_incomplete_propagates_and_wrapper_timing_is_reported():
    target = observer()
    observe(target, 0, [], [], audit=False, current_pose=pose(0))
    target.record_wrapper_timing(1.25)
    result = target.close(base_close(audit=False))
    summary = target.summary()
    assert result.audit_complete is False
    assert summary["audit_complete"] is False
    assert summary["wrapper_time_p95_ms"] == pytest.approx(1.25)
    assert summary["gt_access"] is False and summary["clip_access"] is False
    assert summary["training_free"] is True
