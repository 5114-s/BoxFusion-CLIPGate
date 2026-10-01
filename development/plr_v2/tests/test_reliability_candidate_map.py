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


def pose(camera_x):
    value = np.eye(4)
    value[0, 3] = camera_x
    return value


def test_full_v2_composition_is_online_bounded_and_keeps_branches_separate():
    state = build_reliability_candidate_map(
        "scene",
        {
            "variant": "reliability_v2",
            "audit_controls": True,
            "audit_plr_controls": True,
            "cross_branch_dedup": False,
            "m1p": {"use_children": False},
            "m1a": {"min_angular_separation_deg": 15.0},
            "m2": {"support_mode": "reliability"},
        },
    )
    proposal = box(0.0)
    anchor = box(2.0)
    native_box = box(-2.0)
    snapshot = None
    for frame, camera_x in enumerate((-3.0, 0.0, 3.0)):
        camera = pose(camera_x)
        projected = _project_with_visibility(proposal, camera, K, 256, 256)
        assert projected is not None
        current = NativeFrame(
            ids=np.asarray([7]),
            corners=np.asarray([native_box]),
            scores=np.asarray([0.2]),
            camera_to_world=camera,
            intrinsic=K,
            width=256,
            height=256,
        )
        evidence = EvidenceFrame(
            proposal_ids=np.asarray([frame]),
            proposal_boxes_2d=np.asarray([projected[0]]),
            proposal_corners=np.asarray([proposal]),
            proposal_scores=np.asarray([0.9]),
            anchor_ids=np.asarray([100 + frame]),
            anchor_corners=np.asarray([anchor]),
            anchor_scores=np.asarray([0.2]),
        )
        snapshot = state.update(frame, current, evidence)
    assert snapshot is not None
    assert snapshot.sources.count("native") == 1
    assert snapshot.sources.count("m1p") == 1
    assert snapshot.sources.count("m1a") == 1
    assert len(snapshot.native_corners) == 1
    diagnostics = state.diagnostics()
    assert diagnostics["strictly_causal"]
    assert diagnostics["uses_full_scene_cache"] is False
    assert diagnostics["m2"]["support_mode"] == "reliability"
    components = state.experiment_components(
        snapshot.native_ids, snapshot.native_corners, np.asarray([0.2])
    )
    assert {
        "native_native",
        "native_first",
        "native_mean",
        "native_max",
        "native_ema",
        "native_diverse_max",
        "native_reliability",
        "births_plr_v1",
        "births_plr_assoc",
        "births_plr_reliability",
        "births_plr_score",
        "births_calr_v1",
        "births_calr_v2",
    } == set(components)
    assert len(components["births_plr_v1"][0]) == 1
    assert len(components["births_plr_assoc"][0]) == 1
    assert len(components["births_plr_reliability"][0]) == 1
    assert len(components["births_plr_score"][0]) == 1
    assert len(components["births_calr_v1"][0]) == 1
    assert len(components["births_calr_v2"][0]) == 1
