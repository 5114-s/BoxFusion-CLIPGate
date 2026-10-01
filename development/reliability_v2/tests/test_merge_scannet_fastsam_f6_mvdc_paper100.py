from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from boxfusion import fastsam_f6_mvdc_selector as core
import tools.merge_scannet_fastsam_f6_mvdc_paper100 as merger


def _aabb(extent: float = 2.0) -> dict[str, object]:
    center = np.asarray([0.0, 0.0, 4.0])
    size = np.full(3, extent)
    return {
        "valid": True, "q02": (center - size / 2).tolist(),
        "q98": (center + size / 2).tolist(), "center": center.tolist(),
        "extent": size.tolist(), "stored_point_count": 64,
    }


def _obb() -> dict[str, object]:
    center = np.asarray([0.0, 0.0, 4.0]); extent = np.full(3, 1.3)
    signs = np.asarray([
        (-1, -1, -1), (-1, -1, 1), (-1, 1, -1), (-1, 1, 1),
        (1, -1, -1), (1, -1, 1), (1, 1, -1), (1, 1, 1),
    ], dtype=np.float64)
    return {
        "valid": True, "world_center": center.tolist(), "local_extent": extent.tolist(),
        "world_rotation": np.eye(3).tolist(),
        "world_corners": (center[None] + signs * extent[None] / 2).tolist(),
        "camera_depth": 4.0, "confidence": 0.01,
    }


def _source(frame_id: int, ordinal: int) -> core.F6SourceEvidence:
    grid = np.linspace(-0.55, 0.55, 4)
    points = np.asarray([(x, y, 4.0 + z) for x in grid for y in grid for z in grid])
    mask = np.zeros((480, 640), dtype=np.uint8); mask[220:260, 300:340] = 1
    hypotheses = {"H0": _aabb(), "HL": None, "HLG": None, "HB": _obb()}
    return core.F6SourceEvidence(
        source_id=f"scene0000_00/frame_{frame_id:06d}/raw_000", frame_id=frame_id,
        frame_ordinal=ordinal, rank=0, hypotheses=hypotheses, points_world=points,
        mask_packbits=np.packbits(mask.reshape(-1), bitorder="little"),
        tight_box_xyxy=np.asarray([300.0, 220.0, 340.0, 260.0]),
        camera_to_world=np.eye(4),
        intrinsic=np.asarray([[100.0, 0.0, 320.0], [0.0, 100.0, 240.0], [0.0, 0.0, 1.0]]),
        source_lineage_sha256="a" * 64,
    )


def _switch_case() -> tuple[dict[str, object], dict[str, object], list[dict[str, object]]]:
    state = core.F6SelectorState()
    for ordinal, frame_id in enumerate((0, 25)):
        query = state.query_frame(frame_id=frame_id, frame_ordinal=ordinal, sources=(_source(frame_id, ordinal),))
        state.commit_frame(query)
    source = _source(50, 2)
    query = state.query_frame(frame_id=50, frame_ordinal=2, sources=(source,))
    row = dict(query.rows[0])
    assert row["switched_from_base"] is True
    f4_source = {
        "source_id": source.source_id, "source_lineage_sha256": source.source_lineage_sha256,
        "frame_id": source.frame_id, "frame_ordinal": source.frame_ordinal, "rank": source.rank,
        "hypotheses": {"H0": _aabb(), "HL": None, "HLG": None, "HB": _obb()},
    }
    return row, f4_source, [dict(value) for value in query.buffer_before]


def test_switch_requires_complete_three_view_rule_proof() -> None:
    row, source, buffer_before = _switch_case()
    selected, switched, evaluated = merger._verify_source_row(
        row, source, buffer_before=buffer_before, core=core
    )
    assert selected == "HB" and switched is True and evaluated is True
    tampered = copy.deepcopy(row)
    tampered["candidate_evaluations"]["HB"]["comparison"]["win_count"] = 1
    tampered["result_sha256"] = core.canonical_result_sha256(tampered)
    with pytest.raises(merger.F6MergeError, match="win/non-regression"):
        merger._verify_source_row(tampered, source, buffer_before=buffer_before, core=core)


def test_switch_rejects_missing_distinct_past_view_even_when_resealed() -> None:
    row, source, buffer_before = _switch_case()
    tampered = copy.deepcopy(row)
    tampered["matched_past"][1] = copy.deepcopy(tampered["matched_past"][0])
    tampered["result_sha256"] = core.canonical_result_sha256(tampered)
    with pytest.raises(merger.F6MergeError, match="distinct|unavailable"):
        merger._verify_source_row(tampered, source, buffer_before=buffer_before, core=core)


def test_frozen_four_decision_branches() -> None:
    good = {
        "contract": {"pass": True}, "switch_min_sources": {"pass": True},
        "switch_min_scenes": {"pass": True}, "switch_max_fraction": {"pass": True},
    }
    assert merger._decision_from_gates(good).startswith("retain_f6")
    failed = copy.deepcopy(good); failed["contract"]["pass"] = False
    assert merger._decision_from_gates(failed) == "discard_f6_selector"
    few = copy.deepcopy(good); few["switch_min_sources"]["pass"] = False
    assert merger._decision_from_gates(few) == "stop_f6_insufficient_multiview_switches"
    broad = copy.deepcopy(good); broad["switch_max_fraction"]["pass"] = False
    assert merger._decision_from_gates(broad) == "stop_f6_overbroad_switches"


def test_cli_exposes_no_forbidden_data_argument() -> None:
    options = {option for action in merger._parser()._actions for option in action.option_strings}
    assert not any(
        token in option
        for option in options
        for token in ("gt", "annotation", "oracle", "evaluator", "native", "prediction", "training")
    )


def test_buffer_keeps_three_successful_frames_across_failed_schedule_gaps() -> None:
    row = {
        "frame_id": 100,
        "frame_ordinal": 4,
        "source_ids": ["scene0000_00/frame_000100/raw_000"],
        "state_sha256": ["a" * 64],
        "raw_array_payload_bytes": 1024,
    }
    assert merger._validate_buffer(
        [row], current_ordinal=9, label="past-three-successes"
    )[0]["frame_ordinal"] == 4
    future = copy.deepcopy(row)
    future["frame_ordinal"] = 9
    with pytest.raises(merger.F6MergeError, match="causal"):
        merger._validate_buffer([future], current_ordinal=9, label="future")


def test_final_receipt_writer_is_create_only(tmp_path: Path) -> None:
    target = tmp_path / "final" / merger.OUTPUT_NAME
    first = {"schema": merger.MERGE_SCHEMA, "complete": True}
    assert len(merger._atomic_create_json(target, first)) == 64
    with pytest.raises(merger.F6MergeError, match="overwrite"):
        merger._atomic_create_json(target, first)
