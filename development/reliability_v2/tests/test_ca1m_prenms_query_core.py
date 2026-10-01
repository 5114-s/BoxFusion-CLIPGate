import numpy as np
import pytest

from tools.ca1m_prenms_query_core import (
    PendingMemory, box_iou2d, box_iou3d, normalize_features, project_box, select_local,
)


def cube(center=(0.0, 0.0, 2.0), extent=1.0):
    return np.array([[x, y, z] for x in (-0.5, 0.5) for y in (-0.5, 0.5)
                     for z in (-0.5, 0.5)]) * extent + np.asarray(center)


def commit_one(memory, ordinal, frame=None, box=None, feature=None, score=0.8):
    return memory.commit(ordinal, ordinal * 25 if frame is None else frame,
                         np.asarray([cube() if box is None else box]),
                         np.asarray([[1.0, 0.0] if feature is None else feature]), [score])


def test_iou2d_empty_invalid_and_exact_values():
    boxes = np.array([[0, 0, 2, 2], [1, 0, 3, 2], [0, 0, 0, 1], [np.nan, 0, 1, 1]])
    result = box_iou2d(boxes, boxes[:1])
    np.testing.assert_allclose(result[:, 0], [1, 1 / 3, 0, 0])
    assert box_iou2d([], boxes).shape == (0, 4)
    assert box_iou2d(boxes, []).shape == (4, 0)


def test_iou3d_empty_invalid_and_corner_order_invariance():
    boxes = np.asarray([cube(), cube((0.5, 0, 2)), cube() * np.nan])
    np.testing.assert_allclose(box_iou3d(boxes, boxes[:1])[:, 0], [1, 1 / 3, 0])
    assert box_iou3d([], boxes).shape == (0, 3)
    assert box_iou3d(boxes, []).shape == (3, 0)
    assert box_iou3d(boxes[:1, ::-1], boxes[:1])[0, 0] == 1


def test_project_box_intrinsics_pose_and_no_mutation():
    corners = cube()
    original = corners.copy()
    K = np.array([[100, 0, 320], [0, 120, 240], [0, 0, 1]])
    result = project_box(corners, np.eye(4), K, 640, 480)
    np.testing.assert_allclose(result, [320 - 100 / 3, 200, 320 + 100 / 3, 280])
    pose = np.eye(4)
    pose[0, 3] = 1.0
    shifted = project_box(corners + [1, 0, 0], pose, K, 640, 480)
    np.testing.assert_allclose(result, shifted)
    np.testing.assert_array_equal(original, corners)


@pytest.mark.parametrize("box", [cube((0, 0, 0)), cube((0, 0, -2)), cube((100, 0, 2))])
def test_project_rejects_crossing_behind_and_offscreen(box):
    K = np.array([[100, 0, 320], [0, 100, 240], [0, 0, 1]])
    assert project_box(box, np.eye(4), K, 640, 480) is None


def test_project_rejects_singular_pose_and_invalid_depth():
    assert project_box(cube(), np.zeros((4, 4)), np.eye(3), 640, 480) is None
    assert project_box(cube() * np.nan, np.eye(4), np.eye(3), 640, 480) is None


def test_normalize_zero_invalid_and_dtype():
    x = np.array([[3, 4], [0, 0], [np.nan, 1], [np.inf, 1]], dtype=float)
    original = x.copy()
    result = normalize_features(x)
    assert result.dtype == np.float32
    np.testing.assert_allclose(result, [[0.6, 0.8], [0, 0], [0, 0], [0, 0]])
    np.testing.assert_array_equal(x, original)


def test_select_same_pool_query_and_control_different_rankings():
    boxes = np.array([[0, 0, 4, 10], [6, 0, 10, 10], [30, 0, 40, 10]], dtype=float)
    result = select_local(boxes, [.1, .9, 1], [[1, 0], [0, 1], [1, 0]],
                          [0, 0, 10, 10], [[1, 0]], [])
    assert result["pool_ids"].tolist() == [0, 1]
    assert result["query_ids"].tolist() == [0]
    assert result["control_ids"].tolist() == [1]
    np.testing.assert_allclose(result["query_cosines"], [1])


def test_select_stable_ties_identical_nms_and_exclusions():
    boxes = np.array([[0, 0, 4, 10], [0, 0, 4, 10], [6, 0, 10, 10]], dtype=float)
    args = (boxes, [.5, .5, .5], [[1, 0]] * 3, [0, 0, 10, 10], [[1, 0]])
    result = select_local(*args, [], limit=3)
    assert result["query_ids"].tolist() == result["control_ids"].tolist() == [0, 2]
    excluded = select_local(*args, [0], limit=3)
    assert excluded["pool_ids"].tolist() == [1, 2]
    assert excluded["query_ids"].tolist() == [1, 2]


def test_select_geometry_boundaries_invalid_and_degenerate():
    boxes = np.array([[-5, 0, 0, 10], [-5, 0, 1, 10], [-6, 0, 0, 10],
                      [0, 0, 0, 2], [0, 0, 4, 10], [6, 0, 10, 10]])
    scores = [.5] * 5 + [np.nan]
    feature = [[1, 0]] * 4 + [[np.inf, 0], [1, 0]]
    result = select_local(boxes, scores, feature, [0, 0, 10, 10], [[1, 0]], [], 6)
    assert result["pool_ids"].tolist() == [1]


def test_select_empty_no_prototype_and_limit_zero():
    assert select_local(np.empty((0, 4)), [], np.empty((0, 2)), [0, 0, 10, 10],
                        [], [])['pool_ids'].size == 0
    args = ([[0, 0, 5, 5]], [.5], [[1, 0]], [0, 0, 10, 10], [], [])
    result = select_local(*args)
    assert result["query_ids"].size == 0
    assert result["control_ids"].tolist() == [0]
    assert select_local(*args, limit=0)["control_ids"].size == 0


def test_memory_query_before_commit_distinct_frames_and_confirmation():
    memory = PendingMemory()
    assert memory.eligible(0) == []
    commit_one(memory, 0)
    assert memory.eligible(0) == []
    first = memory.eligible(1)[0]
    assert first["frames"] == [0]
    assert first["frame_count"] == 1
    commit_one(memory, 1)
    assert memory.eligible(2)[0]["frames"] == [0, 25]
    commit_one(memory, 2)
    assert memory.eligible(3) == []
    assert memory._tracks[0]["confirmed"]


def test_memory_snapshot_isolation_does_not_add_query_evidence():
    memory = PendingMemory()
    commit_one(memory, 0)
    snapshot = memory.eligible(1)[0]
    snapshot["obs"][0][:] = 999
    snapshot["features"][0][:] = 999
    snapshot["frames"].append(25)
    fresh = memory.eligible(1)[0]
    np.testing.assert_array_equal(fresh["obs"][0], cube())
    assert fresh["frames"] == [0]
    assert fresh["frame_count"] == 1


def test_memory_duplicate_same_frame_uses_highest_score_only():
    memory = PendingMemory()
    stats = memory.commit(0, 0, [cube(), cube()], [[0, 1], [1, 0]], [.2, .9])
    assert stats["added"] == 1 and stats["skipped"] == 1
    state = memory.eligible(1)[0]
    assert state["score"] == .9 and state["frame_count"] == 1
    np.testing.assert_array_equal(state["features"], [[1, 0]])


def test_memory_duplicate_commit_is_idempotent_and_old_frames_rejected():
    memory = PendingMemory()
    commit_one(memory, 0)
    assert commit_one(memory, 0)["duplicate_commit"]
    assert memory.eligible(1)[0]["frame_count"] == 1
    with pytest.raises(ValueError, match="strictly increasing"):
        commit_one(memory, 1, frame=0)
    with pytest.raises(ValueError, match="backwards"):
        memory.eligible(0)


def test_memory_ttl_boundary_and_no_reused_id():
    memory = PendingMemory(ttl=2)
    commit_one(memory, 0)
    assert memory.eligible(2)[0]["id"] == 0
    assert memory.eligible(3) == []
    commit_one(memory, 3)
    assert memory.eligible(4)[0]["id"] == 1


def test_memory_capacity_keeps_highest_current_score_and_bounds_history():
    memory = PendingMemory(max_tracks=2, max_obs=3)
    boxes = np.asarray([cube((3 * i, 0, 2)) for i in range(3)])
    memory.commit(0, 0, boxes, [[1, 0]] * 3, [.2, .9, .7])
    assert sorted(t["score"] for t in memory.eligible(1)) == [.7, .9]
    for ordinal in range(1, 8):
        commit_one(memory, ordinal, box=boxes[1], feature=[1, ordinal])
    assert len(memory._tracks) <= 2
    track = next(t for t in memory._tracks.values() if t["confirmed"])
    assert len(track["obs"]) == len(track["frames"]) == 3
    assert len(track["features"]) <= 2
    assert track["frame_count"] == 8


def test_memory_center_gate_and_best_iou_association():
    memory = PendingMemory()
    memory.commit(0, 0, [cube(), cube((0.7, 0, 2))], [[1, 0], [0, 1]], [.8, .7])
    commit_one(memory, 1, box=cube((0.45, 0, 2)))
    states = {t["id"]: t for t in memory.eligible(2)}
    assert states[0]["frame_count"] == 1
    assert states[1]["frame_count"] == 2


def test_memory_invalid_observations_and_feature_dimension():
    memory = PendingMemory()
    stats = memory.commit(0, 0, [cube(), cube() * np.nan], [[0, 0], [1, 0]], [.9, .9])
    assert stats["skipped"] == 2 and memory.eligible(1) == []
    with pytest.raises(ValueError, match="dimension"):
        memory.commit(1, 25, [cube()], [[1, 0, 0]], [.5])


def test_memory_prioritizes_nearly_confirmed_before_score():
    memory = PendingMemory()
    memory.commit(0, 0, [cube(), cube((3, 0, 2))], [[1, 0], [0, 1]], [.1, .9])
    commit_one(memory, 1, score=.1)
    assert memory.eligible(2, limit=1)[0]["id"] == 1
