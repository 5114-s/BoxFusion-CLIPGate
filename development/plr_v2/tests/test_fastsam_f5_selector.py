from __future__ import annotations

import copy

import numpy as np
import pytest

from boxfusion.fastsam_f5_selector import (
    F5ContractError,
    F5SelectorState,
    F5SourceEvidence,
    POLICY,
    canonical_result_sha256,
)


def _aabb_row(lower=(-0.5, -0.5, 1.5), upper=(0.5, 0.5, 2.5), *, count=27):
    lower_array = np.asarray(lower, dtype=np.float64)
    upper_array = np.asarray(upper, dtype=np.float64)
    return {
        "valid": True,
        "q02": lower_array.tolist(),
        "q98": upper_array.tolist(),
        "center": ((lower_array + upper_array) * 0.5).tolist(),
        "extent": (upper_array - lower_array).tolist(),
        "stored_point_count": count,
        "diagnostics": {
            "applied": True,
            "fallback": False,
            "retained_point_count": count,
            "source_point_count": count,
        },
    }


def _hb_row(*, confidence=0.9, center=(0.0, 0.0, 2.0)):
    center_array = np.asarray(center, dtype=np.float64)
    extent = np.asarray([1.0, 1.0, 1.0], dtype=np.float64)
    rotation = np.eye(3, dtype=np.float64)
    signs = np.asarray(
        [
            [-1, -1, -1],
            [-1, -1, +1],
            [-1, +1, -1],
            [-1, +1, +1],
            [+1, -1, -1],
            [+1, -1, +1],
            [+1, +1, -1],
            [+1, +1, +1],
        ],
        dtype=np.float64,
    )
    corners = center_array + signs * extent * 0.5
    return {
        "valid": True,
        "confidence": confidence,
        "camera_depth": float(center_array[2]),
        "world_center": center_array.tolist(),
        "local_extent": extent.tolist(),
        "world_rotation": rotation.tolist(),
        "world_corners": corners.tolist(),
    }


def _source(frame_ordinal: int, *, confidence=0.9, center=(0.0, 0.0, 2.0)):
    frame_id = frame_ordinal * 25
    axis = np.asarray([-0.3, 0.0, 0.3], dtype=np.float64)
    points = np.stack(np.meshgrid(axis, axis, np.asarray([1.7, 2.0, 2.3])), axis=-1).reshape(-1, 3)
    points[:, 0] += center[0]
    points[:, 1] += center[1]
    points[:, 2] += center[2] - 2.0
    h0 = _aabb_row(
        (center[0] - 0.5, center[1] - 0.5, center[2] - 0.5),
        (center[0] + 0.5, center[1] + 0.5, center[2] + 0.5),
    )
    return F5SourceEvidence(
        source_id=f"scene0000_00/frame_{frame_id:06d}/raw_000",
        frame_id=frame_id,
        frame_ordinal=frame_ordinal,
        rank=0,
        hypotheses={
            "H0": copy.deepcopy(h0),
            "HL": copy.deepcopy(h0),
            "HLG": copy.deepcopy(h0),
            "HB": _hb_row(confidence=confidence, center=center),
        },
        points_world=points,
        tight_box_xyxy=np.asarray([286.0, 206.0, 354.0, 274.0]),
        camera_to_world=np.eye(4),
        intrinsic=np.asarray([[100.0, 0.0, 320.0], [0.0, 100.0, 240.0], [0.0, 0.0, 1.0]]),
        source_lineage_sha256="a" * 64,
    )


def test_hb_requires_two_committed_past_frames_and_never_changes_score():
    state = F5SelectorState()
    choices = []
    hashes = []
    for ordinal in range(3):
        query, commit = state.select_frame(
            frame_id=ordinal * 25,
            frame_ordinal=ordinal,
            sources=[_source(ordinal)],
        )
        row = query.rows[0]
        choices.append(row["selected_hypothesis"])
        hashes.append(row["result_sha256"])
        assert row["formal_score"] == 1.0
        assert canonical_result_sha256(row) == row["result_sha256"]
        assert commit.source_count == 1
        assert len(commit.buffer_after) <= 3
        assert query.maximum_accessed_frame_ordinal < ordinal
    assert choices == ["HLG", "HLG", "HB"]
    assert len(set(hashes)) == 3


def test_confidence_cannot_override_current_geometry_gate():
    state = F5SelectorState()
    for ordinal in range(2):
        state.select_frame(
            frame_id=ordinal * 25,
            frame_ordinal=ordinal,
            sources=[_source(ordinal)],
        )
    shifted = _source(2, confidence=1.0, center=(2.0, 0.0, 2.0))
    query, _ = state.select_frame(frame_id=50, frame_ordinal=2, sources=[shifted])
    assert query.rows[0]["selected_hypothesis"] != "HB"
    assert query.rows[0]["hb_abstention_reason"] in {
        "projection_iou",
        "history_count",
        "past_consistency",
    }


def test_exact_pending_query_is_required_for_commit():
    state = F5SelectorState()
    query = state.query_frame(frame_id=0, frame_ordinal=0, sources=[_source(0)])
    other_state = F5SelectorState()
    other = other_state.query_frame(frame_id=0, frame_ordinal=0, sources=[_source(0)])
    with pytest.raises(F5ContractError, match="exact pending"):
        state.commit_frame(other)
    state.commit_frame(query)


def test_query_row_tampering_is_rejected_at_commit():
    state = F5SelectorState()
    query = state.query_frame(frame_id=0, frame_ordinal=0, sources=[_source(0)])
    query.rows[0]["formal_score"] = 0.9
    with pytest.raises(F5ContractError, match="formal score"):
        state.commit_frame(query)


def test_source_id_frame_and_state_scene_are_bound():
    source = _source(0)
    with pytest.raises(F5ContractError, match="source_id frame"):
        F5SourceEvidence(
            source_id="scene0000_00/frame_999999/raw_000",
            frame_id=0,
            frame_ordinal=0,
            rank=0,
            hypotheses=source.hypotheses,
            points_world=source.points_world,
            tight_box_xyxy=source.tight_box_xyxy,
            camera_to_world=source.camera_to_world,
            intrinsic=source.intrinsic,
            source_lineage_sha256=source.source_lineage_sha256,
        )
    state = F5SelectorState()
    state.select_frame(frame_id=0, frame_ordinal=0, sources=[source])
    other = _source(1)
    other = F5SourceEvidence(
        source_id=other.source_id.replace("scene0000_00", "scene0001_00"),
        frame_id=other.frame_id,
        frame_ordinal=other.frame_ordinal,
        rank=other.rank,
        hypotheses=other.hypotheses,
        points_world=other.points_world,
        tight_box_xyxy=other.tight_box_xyxy,
        camera_to_world=other.camera_to_world,
        intrinsic=other.intrinsic,
        source_lineage_sha256=other.source_lineage_sha256,
    )
    with pytest.raises(F5ContractError, match="cross scene"):
        state.query_frame(frame_id=25, frame_ordinal=1, sources=[other])


def test_future_perturbation_does_not_change_prefix_hashes():
    def replay(final_center):
        state = F5SelectorState()
        ledger = []
        for ordinal in range(3):
            center = final_center if ordinal == 2 else (0.0, 0.0, 2.0)
            query, _ = state.select_frame(
                frame_id=ordinal * 25,
                frame_ordinal=ordinal,
                sources=[_source(ordinal, center=center)],
            )
            ledger.append(query.rows[0]["result_sha256"])
        return ledger

    first = replay((0.0, 0.0, 2.0))
    second = replay((1.0, 0.0, 2.0))
    assert first[:2] == second[:2]
    assert first[2] != second[2]


def test_buffer_is_bounded_to_three_successful_frames():
    state = F5SelectorState()
    for ordinal in range(8):
        _, commit = state.select_frame(
            frame_id=ordinal * 25,
            frame_ordinal=ordinal,
            sources=[_source(ordinal)],
        )
        assert len(commit.buffer_after) <= POLICY["max_buffered_successful_frames"]
        assert all(
            ordinal - frame["frame_ordinal"] <= 3 for frame in commit.buffer_after
        )


def test_h0_lineage_and_geometry_fail_closed():
    source = _source(0)
    bad = dict(source.hypotheses)
    bad["H0"] = dict(bad["H0"], q98=[-1.0, -1.0, -1.0])
    with pytest.raises(F5ContractError):
        F5SourceEvidence(
            source_id=source.source_id,
            frame_id=source.frame_id,
            frame_ordinal=source.frame_ordinal,
            rank=source.rank,
            hypotheses=bad,
            points_world=source.points_world,
            tight_box_xyxy=source.tight_box_xyxy,
            camera_to_world=source.camera_to_world,
            intrinsic=source.intrinsic,
            source_lineage_sha256=source.source_lineage_sha256,
        )
