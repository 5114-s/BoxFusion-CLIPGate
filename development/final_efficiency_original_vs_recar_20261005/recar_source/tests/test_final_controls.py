import numpy as np

from boxfusion.native_reliability_reranker import _project_with_visibility
from boxfusion.online_candidate_map import EvidenceFrame, NativeFrame
from boxfusion.reliability_candidate_map import build_reliability_candidate_map


SIGNS = np.asarray(
    [[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)],
    dtype=np.float64,
)
K = np.asarray([[100.0, 0.0, 128.0], [0.0, 100.0, 128.0], [0.0, 0.0, 1.0]])


def box(x):
    return SIGNS * 0.5 + np.asarray([x, 0.0, 5.0])


def pose(x):
    value = np.eye(4)
    value[0, 3] = x
    return value


def test_paired_raw_mvsr_and_direct_plr_controls_share_one_causal_pass():
    state = build_reliability_candidate_map(
        "scene",
        {
            "audit_controls": True,
            "audit_direct_plr": True,
            "audit_raw_iou_max": True,
            "m1p": {"use_children": False, "max_births": 2},
            "m1a": {"min_angular_separation_deg": 15.0},
            "m2": {"support_mode": "max"},
        },
    )
    proposal, native_box = box(0.0), box(-2.0)
    snapshot = None
    for frame, camera_x in enumerate((-3.0, 0.0, 3.0)):
        camera = pose(camera_x)
        projection = _project_with_visibility(proposal, camera, K, 256, 256)
        assert projection is not None
        native = NativeFrame(
            ids=np.asarray([7]), corners=np.asarray([native_box]),
            scores=np.asarray([0.2]), camera_to_world=camera,
            intrinsic=K, width=256, height=256,
        )
        evidence = EvidenceFrame(
            proposal_ids=np.asarray([frame]),
            proposal_boxes_2d=np.asarray([projection[0]]),
            proposal_corners=np.asarray([proposal]),
            proposal_scores=np.asarray([0.9]),
        )
        snapshot = state.update(frame, native, evidence)
    snapshot = state.materialize_terminal(
        2, snapshot.native_ids, snapshot.native_corners, np.asarray([0.2])
    )
    components = state.experiment_components(
        snapshot.native_ids, snapshot.native_corners, np.asarray([0.2])
    )
    assert "native_raw_iou_max" in components
    assert "native_max" in components
    assert "births_plr_direct" in components
    assert len(components["births_plr_direct"][0]) == len(components["births_plr_v1"][0])
    diagnostics = state.diagnostics()
    assert diagnostics["raw_m2_control"]["frames_seen"] == 3
    assert diagnostics["direct_plr_control"]["temporal_confirmation_disabled"]


def test_selected_runtime_uses_calr_v1_without_updating_calr_v2():
    state = build_reliability_candidate_map(
        "scene",
        {
            "selected_calr_v1": True,
            "export_qualitative_trace": True,
            "m1p": {"use_children": False},
            "m1a": {"min_views": 3},
            "m2": {"support_mode": "max"},
        },
    )
    snapshot = None
    for frame in range(3):
        native = NativeFrame(
            ids=np.asarray([7]), corners=np.asarray([box(-2.0)]),
            scores=np.asarray([0.2]), camera_to_world=pose(float(frame)),
            intrinsic=K, width=256, height=256,
        )
        evidence = EvidenceFrame(
            proposal_ids=np.empty(0, dtype=np.int64),
            proposal_boxes_2d=np.empty((0, 4), dtype=np.float64),
            proposal_corners=np.empty((0, 8, 3), dtype=np.float64),
            proposal_scores=np.empty(0, dtype=np.float64),
            anchor_ids=np.asarray([frame]),
            anchor_corners=np.asarray([box(0.0)]),
            anchor_scores=np.asarray([0.7]),
        )
        snapshot = state.update(frame, native, evidence)
    assert snapshot.sources.count("m1a") == 1
    snapshot = state.materialize_terminal(
        2, snapshot.native_ids, snapshot.native_corners, np.asarray([0.2])
    )
    diagnostic = state.diagnostics()
    assert diagnostic["selected_calr_v1"] is True
    assert diagnostic["m1a_control"]["frames_seen"] == 3
    assert diagnostic["m1a"]["frames_seen"] == 0
    assert len(diagnostic["qualitative_trace"]["calr"]) == 1
    assert diagnostic["qualitative_trace"]["calr"][0]["support_frame_ids"] == [0, 1, 2]
