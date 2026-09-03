from __future__ import annotations

import copy

import numpy as np
import pytest

from boxfusion.fastsam_f6_mvdc_selector import (
    F6ContractError,
    F6SelectorState,
    F6SourceEvidence,
    MAX_RAW_ARRAY_PAYLOAD_BYTES,
    POLICY,
    PROTOCOL_ID,
    SCHEMA,
    canonical_result_sha256,
)


def _aabb(center=(0.0, 0.0, 4.0), extent=(2.0, 2.0, 2.0), *, stored=64):
    center = np.asarray(center, dtype=np.float64)
    extent = np.asarray(extent, dtype=np.float64)
    lower = center - extent * 0.5
    upper = center + extent * 0.5
    return {
        "valid": True,
        "q02": lower.tolist(),
        "q98": upper.tolist(),
        "center": center.tolist(),
        "extent": extent.tolist(),
        "stored_point_count": stored,
    }


def _obb(center=(0.0, 0.0, 4.0), extent=(1.3, 1.3, 1.3), confidence=0.01):
    center = np.asarray(center, dtype=np.float64)
    extent = np.asarray(extent, dtype=np.float64)
    rotation = np.eye(3, dtype=np.float64)
    signs = np.asarray(
        [
            (-1, -1, -1),
            (-1, -1, 1),
            (-1, 1, -1),
            (-1, 1, 1),
            (1, -1, -1),
            (1, -1, 1),
            (1, 1, -1),
            (1, 1, 1),
        ],
        dtype=np.float64,
    )
    corners = center[None, :] + signs * extent[None, :] * 0.5
    return {
        "valid": True,
        "world_center": center.tolist(),
        "local_extent": extent.tolist(),
        "world_rotation": rotation.tolist(),
        "world_corners": corners.tolist(),
        "camera_depth": float(center[2]),
        "confidence": confidence,
    }


def _points(count=64):
    grid = np.linspace(-0.55, 0.55, 4)
    values = np.asarray(
        [(x, y, 4.0 + z) for x in grid for y in grid for z in grid],
        dtype=np.float64,
    )
    if count <= len(values):
        return values[:count]
    repeats = int(np.ceil(count / len(values)))
    return np.tile(values, (repeats, 1))[:count]


def _mask(x0=300, x1=340, y0=220, y1=260):
    mask = np.zeros((480, 640), dtype=np.uint8)
    mask[y0:y1, x0:x1] = 1
    return np.packbits(mask.reshape(-1), bitorder="little")


def _source(
    frame_id: int,
    ordinal: int,
    *,
    rank: int = 0,
    raw: int | None = None,
    center=(0.0, 0.0, 4.0),
    point_count: int = 64,
    confidence: float = 0.01,
    hb: bool = True,
    hlg: bool = False,
    mask_packbits=None,
):
    raw = rank if raw is None else raw
    points = _points(point_count) + (np.asarray(center) - np.asarray((0.0, 0.0, 4.0)))[None, :]
    hypotheses = {"H0": _aabb(center, stored=point_count)}
    if hlg:
        row = _aabb(center, (1.8, 1.8, 1.8), stored=point_count)
        row["diagnostics"] = {
            "applied": True,
            "fallback": False,
            "retained_point_count": point_count,
        }
        hypotheses["HLG"] = row
    if hb:
        hypotheses["HB"] = _obb(center, confidence=confidence)
    intrinsic = np.asarray(
        [[100.0, 0.0, 320.0], [0.0, 100.0, 240.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    return F6SourceEvidence(
        source_id=f"scene0000_00/frame_{frame_id:06d}/raw_{raw:03d}",
        frame_id=frame_id,
        frame_ordinal=ordinal,
        rank=rank,
        hypotheses=hypotheses,
        points_world=points,
        mask_packbits=_mask() if mask_packbits is None else mask_packbits,
        tight_box_xyxy=np.asarray([300.0, 220.0, 340.0, 260.0]),
        camera_to_world=np.eye(4, dtype=np.float64),
        intrinsic=intrinsic,
        source_lineage_sha256="a" * 64,
    )


def _advance(state: F6SelectorState, frame_id: int, ordinal: int, **kwargs):
    source = _source(frame_id, ordinal, **kwargs)
    query = state.query_frame(frame_id=frame_id, frame_ordinal=ordinal, sources=(source,))
    commit = state.commit_frame(query)
    return query, commit


def test_protocol_declares_shadow_gt_free_contract():
    assert SCHEMA == "boxfusion.fastsam_f6_mvdc_selector.v1"
    assert PROTOCOL_ID.startswith("F6-GT-FREE-PAST-ONLY")
    assert POLICY["observer_only"] is True
    assert POLICY["birth"] is False
    assert POLICY["native_prediction_access"] is False
    assert POLICY["training"] is False
    assert POLICY["online_learning"] is False
    assert POLICY["maximum_lookahead_frames"] == 0


def test_sampling_is_frozen_deterministic_and_immutable():
    source = _source(0, 0, point_count=1000)
    expected = np.floor((np.arange(256) + 0.5) * 1000 / 256).astype(np.int64)
    np.testing.assert_array_equal(source.points_world, _points(1000)[expected])
    assert source.original_point_count == 1000
    assert len(source.points_world) == 256
    assert len(source.sampled_mask_pixels_yx) == 256
    assert not source.points_world.flags.writeable
    assert not source.mask_packbits.flags.writeable
    assert not source.sampled_mask_pixels_yx.flags.writeable
    with pytest.raises(ValueError):
        source.points_world[0, 0] = 7.0


def test_first_two_views_strictly_fall_back_then_third_selects_hb():
    state = F6SelectorState()
    first, _ = _advance(state, 0, 0)
    second, _ = _advance(state, 25, 1)
    third_source = _source(50, 2)
    third = state.query_frame(frame_id=50, frame_ordinal=2, sources=(third_source,))
    assert first.rows[0]["selection_reason"] == "fewer_than_two_past_matches"
    assert second.rows[0]["selection_reason"] == "fewer_than_two_past_matches"
    row = third.rows[0]
    assert row["matched_past_frame_count"] == 2
    assert [item["frame_ordinal"] for item in row["matched_past"]] == [0, 1]
    assert row["base_hypothesis"] == "H0"
    assert row["selected_hypothesis"] == "HB"
    assert row["switched_from_base"] is True
    hb = row["candidate_evaluations"]["HB"]
    assert hb["gate"]["checks"]["two_of_three_support_passed"] is True
    assert hb["comparison"]["win_count"] >= 2
    assert hb["comparison"]["passed"] is True
    assert row["formal_score"] == 1.0
    assert canonical_result_sha256(row) == row["result_sha256"]


def test_hb_confidence_does_not_gate_or_rank_geometry():
    selected = []
    metric_hashes = []
    for confidence in (0.0, 1.0):
        state = F6SelectorState()
        _advance(state, 0, 0, confidence=confidence)
        _advance(state, 25, 1, confidence=confidence)
        query = state.query_frame(
            frame_id=50,
            frame_ordinal=2,
            sources=(_source(50, 2, confidence=confidence),),
        )
        row = query.rows[0]
        selected.append(row["selected_hypothesis"])
        metric_hashes.append(row["candidate_evaluations"]["HB"]["metrics"])
    assert selected == ["HB", "HB"]
    assert metric_hashes[0] == metric_hashes[1]


def test_query_before_commit_and_exact_token_are_enforced():
    state = F6SelectorState()
    source = _source(0, 0)
    query = state.query_frame(frame_id=0, frame_ordinal=0, sources=(source,))
    with pytest.raises(F6ContractError, match="previous F6 query"):
        state.query_frame(frame_id=25, frame_ordinal=1, sources=(_source(25, 1),))
    other_state = F6SelectorState()
    other = other_state.query_frame(frame_id=0, frame_ordinal=0, sources=(_source(0, 0),))
    with pytest.raises(F6ContractError, match="exact pending"):
        state.commit_frame(other)
    commit = state.commit_frame(query)
    assert commit.token == query.token
    assert state.seen_source_count == 1


def test_result_mutation_is_fatal_at_commit():
    state = F6SelectorState()
    query = state.query_frame(frame_id=0, frame_ordinal=0, sources=(_source(0, 0),))
    query.rows[0]["formal_score"] = 0.5
    with pytest.raises(F6ContractError, match="selection changed"):
        state.commit_frame(query)


def test_state_is_last_three_successful_frames_and_payload_is_exposed():
    state = F6SelectorState()
    commits = []
    for ordinal, frame_id in enumerate((0, 25, 50, 1000)):
        _, commit = _advance(state, frame_id, ordinal)
        commits.append(commit)
        assert commit.state_raw_array_payload_bytes == state.raw_array_payload_bytes
        assert state.raw_array_payload_bytes <= MAX_RAW_ARRAY_PAYLOAD_BYTES
    assert state.buffered_frame_count == 3
    assert [row["frame_ordinal"] for row in commits[-1].buffer_after] == [1, 2, 3]
    next_query = state.query_frame(frame_id=1025, frame_ordinal=4, sources=(_source(1025, 4),))
    assert [row["frame_ordinal"] for row in next_query.buffer_before] == [1, 2, 3]
    assert next_query.maximum_accessed_frame_ordinal == 3
    assert next_query.state_raw_array_payload_bytes == state.raw_array_payload_bytes


def test_mutual_best_is_per_past_frame_with_deterministic_ties():
    state = F6SelectorState()
    past = (_source(0, 0, rank=0, raw=0), _source(0, 0, rank=1, raw=1))
    first = state.query_frame(frame_id=0, frame_ordinal=0, sources=past)
    state.commit_frame(first)
    current = (_source(25, 1, rank=0, raw=0), _source(25, 1, rank=1, raw=1))
    query = state.query_frame(frame_id=25, frame_ordinal=1, sources=current)
    assert query.rows[0]["matched_past_frame_count"] == 1
    assert query.rows[0]["matched_past"][0]["rank"] == 0
    assert query.rows[1]["matched_past_frame_count"] == 0


def test_only_two_most_recent_distinct_matches_are_scored():
    state = F6SelectorState()
    _advance(state, 0, 0)
    _advance(state, 25, 1)
    _advance(state, 50, 2)
    query = state.query_frame(frame_id=75, frame_ordinal=3, sources=(_source(75, 3),))
    assert [row["frame_ordinal"] for row in query.rows[0]["matched_past"]] == [1, 2]
    metrics = query.rows[0]["base_metrics"]
    assert [row["frame_ordinal"] for row in metrics["per_view"]] == [3, 1, 2]


def test_hbase_reproduces_frozen_hlg_rule():
    state = F6SelectorState()
    query = state.query_frame(frame_id=0, frame_ordinal=0, sources=(_source(0, 0, hlg=True),))
    row = query.rows[0]
    assert row["base_hypothesis"] == "HLG"
    assert row["selected_hypothesis"] == "HLG"
    assert row["switched_from_base"] is False


def test_invalid_optional_hb_falls_back_but_invalid_h0_is_fatal():
    source = _source(0, 0)
    hypotheses = copy.deepcopy({key: dict(value) for key, value in source.hypotheses.items()})
    hypotheses["HB"]["camera_depth"] = 0.0
    replacement = F6SourceEvidence(
        source_id=source.source_id,
        frame_id=source.frame_id,
        frame_ordinal=source.frame_ordinal,
        rank=source.rank,
        hypotheses=hypotheses,
        points_world=source.points_world,
        mask_packbits=source.mask_packbits,
        tight_box_xyxy=source.tight_box_xyxy,
        camera_to_world=source.camera_to_world,
        intrinsic=source.intrinsic,
        source_lineage_sha256=source.source_lineage_sha256,
    )
    query = F6SelectorState().query_frame(frame_id=0, frame_ordinal=0, sources=(replacement,))
    assert query.rows[0]["selected_hypothesis"] == "H0"
    assert query.rows[0]["candidate_evaluations"]["HB"]["available"] is False

    broken = {"H0": _aabb(stored=64), "HB": _obb()}
    broken["H0"]["q98"][0] = broken["H0"]["q02"][0]
    with pytest.raises(F6ContractError, match="positive extents"):
        F6SourceEvidence(
            source_id=source.source_id,
            frame_id=source.frame_id,
            frame_ordinal=source.frame_ordinal,
            rank=source.rank,
            hypotheses=broken,
            points_world=source.points_world,
            mask_packbits=source.mask_packbits,
            tight_box_xyxy=source.tight_box_xyxy,
            camera_to_world=source.camera_to_world,
            intrinsic=source.intrinsic,
            source_lineage_sha256=source.source_lineage_sha256,
        )


def test_original_input_mutation_cannot_change_normalized_evidence():
    hypotheses = {"H0": _aabb(stored=64), "HB": _obb()}
    points = _points()
    packed = _mask()
    source = F6SourceEvidence(
        source_id="scene0000_00/frame_000000/raw_000",
        frame_id=0,
        frame_ordinal=0,
        rank=0,
        hypotheses=hypotheses,
        points_world=points,
        mask_packbits=packed,
        tight_box_xyxy=np.asarray([300.0, 220.0, 340.0, 260.0]),
        camera_to_world=np.eye(4),
        intrinsic=np.asarray([[100.0, 0.0, 320.0], [0.0, 100.0, 240.0], [0.0, 0.0, 1.0]]),
        source_lineage_sha256="b" * 64,
    )
    digest = source.input_evidence_sha256
    hypotheses["H0"]["q02"][0] = -99.0
    points[:] = 99.0
    packed[:] = 0
    assert source.input_evidence_sha256 == digest
    assert source.hypotheses["H0"]["q02"][0] == -1.0
    assert np.max(source.points_world) < 5.0
    assert np.count_nonzero(source.mask_packbits) > 0


def test_future_perturbation_cannot_change_earlier_result_hash():
    left = F6SelectorState()
    right = F6SelectorState()
    for state in (left, right):
        _advance(state, 0, 0)
        _advance(state, 25, 1)
    left_query = left.query_frame(frame_id=50, frame_ordinal=2, sources=(_source(50, 2),))
    right_query = right.query_frame(frame_id=50, frame_ordinal=2, sources=(_source(50, 2),))
    assert left_query.rows[0]["result_sha256"] == right_query.rows[0]["result_sha256"]
    left.commit_frame(left_query)
    right.commit_frame(right_query)
    _advance(left, 75, 3, center=(0.0, 0.0, 4.0))
    _advance(right, 75, 3, center=(3.0, 0.0, 4.0))
    assert left_query.rows[0]["result_sha256"] == right_query.rows[0]["result_sha256"]


def test_performance_caches_preserve_frozen_rows_and_tokens_exactly():
    """Lock the pre-optimization semantic receipts byte-for-byte."""

    expected = (
        (
            "45a5cf1aa27e268fa5fce7e8975a024005117e94c784c14bf6fdf9a730626497",
            "fa4ddb9db2346b828da19204115833dd79f8a34b5f539a1550ef86f852192421",
        ),
        (
            "b31a22aeae13eef0909b155a156dd24497fce07bcb6c19d1659e7d0149c41377",
            "eb36300f19120df535f7b6eda99dbe9a1ad36f5e91f1ebf600c0bebc9d2cf29a",
        ),
        (
            "e574f151841faae1259a0876e097ddbb4dc0d0b6970361fab3d9d40bd936e0c5",
            "660ac31844251dbe874ba1754d0fc8307d29cdce75c2cfcef0f63de681bac522",
        ),
    )
    state = F6SelectorState()
    observed = []
    for frame_id, ordinal in ((0, 0), (25, 1), (50, 2)):
        query = state.query_frame(
            frame_id=frame_id,
            frame_ordinal=ordinal,
            sources=(_source(frame_id, ordinal),),
        )
        commit = state.commit_frame(query)
        observed.append((query.rows[0]["result_sha256"], query.token))
        assert query.audit_hash_ns > 0
        assert query.audit_serialization_ns >= 0
        assert commit.audit_hash_ns > 0
        assert commit.audit_serialization_ns == 0
    assert tuple(observed) == expected


def test_invalid_mask_pose_lineage_and_source_order_are_fatal():
    with pytest.raises(F6ContractError, match="positive pixel"):
        _source(0, 0, mask_packbits=np.zeros(38400, dtype=np.uint8))
    source = _source(0, 0)
    with pytest.raises(F6ContractError, match="contiguous"):
        bad_rank = _source(0, 0, rank=1, raw=1)
        F6SelectorState().query_frame(frame_id=0, frame_ordinal=0, sources=(bad_rank,))
    with pytest.raises(F6ContractError, match="lineage"):
        F6SourceEvidence(
            source_id=source.source_id,
            frame_id=source.frame_id,
            frame_ordinal=source.frame_ordinal,
            rank=source.rank,
            hypotheses=source.hypotheses,
            points_world=source.points_world,
            mask_packbits=source.mask_packbits,
            tight_box_xyxy=source.tight_box_xyxy,
            camera_to_world=source.camera_to_world,
            intrinsic=source.intrinsic,
            source_lineage_sha256="invalid",
        )
