import numpy as np

from boxfusion.native_reliability_reranker import (
    OnlineNativeReliabilityReranker,
    _project_with_visibility,
)
from boxfusion.online_candidate_map import EvidenceFrame, NativeFrame


SIGNS = np.asarray(
    [[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)],
    dtype=np.float64,
)
K = np.asarray([[100.0, 0.0, 128.0], [0.0, 100.0, 128.0], [0.0, 0.0, 1.0]])


def box(x=0.0, z=4.0, extent=1.0):
    return SIGNS * (extent / 2.0) + np.asarray([x, 0.0, z])


def pose(camera_x=0.0):
    value = np.eye(4)
    value[0, 3] = camera_x
    return value


def native(corners, camera_x=0.0, identity=7, score=0.2):
    return NativeFrame(
        ids=np.asarray([identity]),
        corners=np.asarray([corners]),
        scores=np.asarray([score]),
        camera_to_world=pose(camera_x),
        intrinsic=K,
        width=256,
        height=256,
    )


def evidence(corners, camera_x=0.0, score=1.0):
    projected = _project_with_visibility(corners, pose(camera_x), K, 256, 256)
    assert projected is not None
    return EvidenceFrame(
        proposal_ids=np.asarray([11]),
        proposal_boxes_2d=np.asarray([projected[0]]),
        proposal_corners=np.asarray([corners]),
        proposal_scores=np.asarray([score]),
    )


def test_reliability_requires_diverse_views_and_changes_scores_only():
    state = OnlineNativeReliabilityReranker(
        support_mode="reliability", min_angular_separation_deg=20.0
    )
    original = box()
    scores = []
    for ordinal, camera_x in enumerate((-3.0, 0.0, 3.0)):
        current = native(original, camera_x)
        scores.append(state.update(ordinal, ordinal, current, evidence(original, camera_x))[0])
        np.testing.assert_array_equal(current.corners[0], original)
        assert len(state.last_scores) == 1
    assert state.reliability.summary(7)["positive_views"] == 3
    assert scores[-1] > 0.2
    assert state.last_frame_strength[0] > 0.99


def test_geometry_change_resets_old_evidence_for_stable_identity():
    state = OnlineNativeReliabilityReranker(
        support_mode="reliability", min_angular_separation_deg=20.0
    )
    first = box(0.0)
    for ordinal, camera_x in enumerate((-3.0, 0.0, 3.0)):
        state.update(ordinal, ordinal, native(first, camera_x), evidence(first, camera_x))
    assert state.reliability.summary(7)["positive_views"] == 3

    moved = box(2.0)
    state.update(3, 3, native(moved, 2.0), evidence(moved, 2.0))
    assert state.diagnostics()["geometry_resets"] == 1
    assert state.reliability.summary(7)["positive_views"] == 1
    assert state.last_scores[0] == 0.2


def test_native_ablation_and_exclusive_matching_preserve_count():
    state = OnlineNativeReliabilityReranker(support_mode="native")
    current_box = box()
    current = NativeFrame(
        ids=np.asarray([1, 2]),
        corners=np.asarray([current_box, current_box]),
        scores=np.asarray([0.2, 0.3]),
        camera_to_world=pose(),
        intrinsic=K,
        width=256,
        height=256,
    )
    output = state.update(0, 0, current, evidence(current_box))
    np.testing.assert_array_equal(output, current.scores)
    assert np.count_nonzero(state.last_frame_strength) == 1
    assert len(output) == 2

    arms = state.materialize_all(current.ids, current.scores)
    assert set(arms) == state.MODES
    np.testing.assert_array_equal(arms["native"][0], current.scores)
    assert all(len(scores) == 2 and len(support) == 2 for scores, support in arms.values())
