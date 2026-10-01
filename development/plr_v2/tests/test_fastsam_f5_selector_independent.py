"""Independent no-GT contract tests for the frozen F5 selector.

The fixtures are entirely synthetic.  This module intentionally imports no
dataset annotation, evaluator, oracle, or native-prediction helper.
"""

from __future__ import annotations

import ast
import copy
from dataclasses import replace
import hashlib
import inspect
import json
from pathlib import Path

import numpy as np
import pytest

from boxfusion import fastsam_f5_selector as f5
from tools import merge_scannet_fastsam_f5_selector_paper100 as f5_merge
from tools import run_scannet_fastsam_f5_selector_paper100 as f5_runner


PROTOCOL_SHA256 = "2a6d62fa9d5912dc3871bbc485f44987565bda61b818722b3a4e6577d34a6afc"
PROTOCOL_PATH = (
    Path(__file__).resolve().parents[1]
    / "docs/F5_GT_FREE_GEOMETRY_SELECTOR_PROTOCOL_FREEZE.md"
)
CORE_PATH = Path(f5.__file__).resolve()
RUNNER_PATH = Path(f5_runner.__file__).resolve()
MERGE_PATH = Path(f5_merge.__file__).resolve()

_SIGNS = np.asarray(
    [
        [-1.0, -1.0, -1.0],
        [-1.0, -1.0, +1.0],
        [-1.0, +1.0, -1.0],
        [-1.0, +1.0, +1.0],
        [+1.0, -1.0, -1.0],
        [+1.0, -1.0, +1.0],
        [+1.0, +1.0, -1.0],
        [+1.0, +1.0, +1.0],
    ],
    dtype=np.float64,
)


def _aabb(
    *,
    center: tuple[float, float, float] = (0.0, 0.0, 2.0),
    extent: tuple[float, float, float] = (1.0, 1.0, 1.0),
    stored_point_count: int = 20,
    eligible: bool = False,
) -> dict:
    center_array = np.asarray(center, dtype=np.float64)
    extent_array = np.asarray(extent, dtype=np.float64)
    row = {
        "valid": True,
        "q02": (center_array - extent_array * 0.5).tolist(),
        "q98": (center_array + extent_array * 0.5).tolist(),
        "center": center_array.tolist(),
        "extent": extent_array.tolist(),
        "stored_point_count": stored_point_count,
    }
    if eligible:
        row["diagnostics"] = {
            "applied": True,
            "fallback": False,
            "retained_point_count": stored_point_count,
        }
    return row


def _hb(
    *,
    center: tuple[float, float, float] = (0.0, 0.0, 2.0),
    extent: tuple[float, float, float] = (1.0, 1.0, 1.0),
    confidence: object = 0.80,
    valid: bool = True,
) -> dict:
    center_array = np.asarray(center, dtype=np.float64)
    extent_array = np.asarray(extent, dtype=np.float64)
    rotation = np.eye(3, dtype=np.float64)
    corners = center_array[None, :] + _SIGNS * (extent_array[None, :] * 0.5)
    return {
        "valid": valid,
        "world_center": center_array.tolist(),
        "local_extent": extent_array.tolist(),
        "world_rotation": rotation.tolist(),
        "world_corners": corners.tolist(),
        "camera_depth": float(center_array[2]),
        "confidence": confidence,
    }


def _source(
    frame: int,
    *,
    scene: str = "scene0000_00",
    rank: int = 0,
    raw_index: int | None = None,
    center: tuple[float, float, float] = (0.0, 0.0, 2.0),
    clean_base: bool = True,
    hb_confidence: object = 0.80,
    points: np.ndarray | None = None,
    tight_box: np.ndarray | None = None,
    source_frame_in_id: int | None = None,
) -> f5.F5SourceEvidence:
    raw = rank if raw_index is None else raw_index
    id_frame = frame if source_frame_in_id is None else source_frame_in_id
    h0 = _aabb(center=center)
    hlg = _aabb(center=center, eligible=True) if clean_base else None
    hypotheses = {
        "H0": h0,
        "HL": None,
        "HLG": hlg,
        "HB": _hb(center=center, confidence=hb_confidence),
    }
    center_array = np.asarray(center, dtype=np.float64)
    if points is None:
        points = np.tile(center_array, (20, 1))
    if tight_box is None:
        # Projection of the unit OBB at z=2 through the intrinsic below.
        tight_box = np.asarray(
            [286.6666666666667, 206.66666666666666,
             353.3333333333333, 273.3333333333333],
            dtype=np.float64,
        )
    return f5.F5SourceEvidence(
        source_id=f"{scene}/frame_{id_frame:06d}/raw_{raw:03d}",
        frame_id=frame,
        frame_ordinal=frame,
        rank=rank,
        hypotheses=hypotheses,
        points_world=points,
        tight_box_xyxy=tight_box,
        camera_to_world=np.eye(4, dtype=np.float64),
        intrinsic=np.asarray(
            [[100.0, 0.0, 320.0], [0.0, 100.0, 240.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        ),
        source_lineage_sha256="a" * 64,
    )


def _three_frame_replay(
    *, hb_confidence: object = 0.80,
) -> tuple[f5.F5SelectorState, list[f5.F5FrameQuery]]:
    state = f5.F5SelectorState()
    queries = []
    for frame in range(3):
        query, _ = state.select_frame(
            frame_id=frame,
            frame_ordinal=frame,
            sources=(_source(frame, hb_confidence=hb_confidence),),
        )
        queries.append(query)
    return state, queries


def test_protocol_hash_id_policy_and_no_forbidden_import_or_io() -> None:
    assert hashlib.sha256(PROTOCOL_PATH.read_bytes()).hexdigest() == PROTOCOL_SHA256
    assert f5.PROTOCOL_ID == "F5-GT-FREE-PAST-ONLY-GEOMETRY-SELECTOR-PAPER100"
    assert f5.POLICY["maximum_lookahead_frames"] == 0
    assert f5.POLICY["formal_score"] == 1.0
    for key in (
        "ground_truth", "annotation", "evaluator", "native_prediction_access",
        "training", "online_learning", "birth", "native_output_mutation",
    ):
        assert f5.POLICY[key] is False

    tree = ast.parse(CORE_PATH.read_text(encoding="utf-8"), filename=str(CORE_PATH))
    imported = set()
    called = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.add(node.module or "")
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                called.add(node.func.id)
            elif isinstance(node.func, ast.Attribute):
                called.add(node.func.attr)
    forbidden_import_prefixes = (
        "evaluation", "tools.audit", "detectron2.data", "sklearn", "scipy.optimize",
    )
    assert not any(
        module.startswith(forbidden_import_prefixes) for module in imported
    )
    assert not ({"open", "read_text", "read_bytes", "iterdir", "glob", "rglob", "load", "save"} & called)

    for public in (f5.F5SelectorState.query_frame, f5.F5SelectorState.commit_frame,
                   f5.F5SelectorState.select_frame, f5.select_frame):
        parameters = set(inspect.signature(public).parameters)
        assert not parameters & {
            "gt", "annotation", "evaluator", "oracle", "native", "prediction",
            "future", "score", "class_name", "embedding",
        }


def test_runner_pins_protocol_has_no_forbidden_cli_and_is_create_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert f5_runner.EXPECTED_PROTOCOL_SHA256 == PROTOCOL_SHA256
    receipts = f5_runner._source_receipts()
    assert receipts["protocol"]["sha256"] == PROTOCOL_SHA256

    changed_protocol = tmp_path / "changed-protocol.md"
    changed_protocol.write_bytes(PROTOCOL_PATH.read_bytes() + b"\nchanged\n")
    monkeypatch.setattr(f5_runner, "PROTOCOL_PATH", changed_protocol)
    with pytest.raises(f5_runner.F5RunnerError, match="protocol hash"):
        f5_runner._source_receipts()

    options = {
        option
        for action in f5_runner._parser()._actions
        for option in action.option_strings
    }
    assert not any(
        token in option.lower()
        for option in options
        for token in (
            "gt", "annotation", "oracle", "evaluator", "native",
            "prediction", "clip", "class", "training", "optimizer",
        )
    )

    tree = ast.parse(RUNNER_PATH.read_text(encoding="utf-8"), filename=str(RUNNER_PATH))
    imports = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    imports.update(
        node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
    )
    forbidden = ("evaluation", "tools.audit", "detectron2.data", "scannet.load")
    assert not any(module.startswith(forbidden) for module in imports)

    output = tmp_path / "create-only.json"
    first = {"value": 1}
    f5_runner._atomic_create_json(output, first)
    with pytest.raises(f5_runner.F5RunnerError, match="overwrite"):
        f5_runner._atomic_create_json(output, {"value": 2})
    assert output.read_text(encoding="ascii").strip().endswith("}")
    assert json.loads(output.read_text(encoding="ascii")) == first


def test_merge_pins_protocol_source_has_no_forbidden_cli_and_exact_gates() -> None:
    assert f5_merge.PROTOCOL_SHA256 == PROTOCOL_SHA256
    source = MERGE_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(MERGE_PATH))
    imports = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    imports.update(
        node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
    )
    forbidden = ("evaluation", "tools.audit", "detectron2.data", "scannet.load")
    assert not any(module.startswith(forbidden) for module in imports)

    options = {
        option
        for action in f5_merge._parser()._actions
        for option in action.option_strings
    }
    assert not any(
        token in option.lower()
        for option in options
        for token in (
            "gt", "annotation", "oracle", "evaluator", "native",
            "prediction", "clip", "class", "training", "optimizer",
        )
    )
    merge_body = inspect.getsource(f5_merge.merge_f5)
    assert "merge_source_seal" in merge_body
    assert "F5 merge source changed during merge" in merge_body

    assert f5_merge._gate(25.0, "<=", 25.0)["pass"] is True
    assert f5_merge._gate(375.0, "<=", 375.0)["pass"] is True
    assert f5_merge._gate(833.33, "<", 833.33)["pass"] is False
    assert f5_merge._gate(0, "==", 0)["pass"] is True


def test_h0_fallback_is_source_preserving_exact_and_constant_score() -> None:
    source = _source(0, clean_base=False, hb_confidence=0.54)
    inputs_before = f5.canonical_json_sha256(source.hypotheses)
    query, commit = f5.F5SelectorState().select_frame(
        frame_id=0, frame_ordinal=0, sources=(source,)
    )
    assert commit.source_count == 1
    assert len(query.rows) == 1
    row = query.rows[0]
    assert row["base_hypothesis"] == "H0"
    assert row["selected_hypothesis"] == "H0"
    assert row["hb_abstention_reason"] == "confidence_threshold"
    assert row["selected_geometry"] == {
        "kind": "world_aabb",
        "hypothesis": "H0",
        "q02": [-0.5, -0.5, 1.5],
        "q98": [0.5, 0.5, 2.5],
        "center": [0.0, 0.0, 2.0],
        "extent": [1.0, 1.0, 1.0],
    }
    assert type(row["formal_score"]) is float and row["formal_score"] == 1.0
    assert row["result_sha256"] == f5.canonical_result_sha256(row)
    assert inputs_before == f5.canonical_json_sha256(source.hypotheses)


def test_hb_requires_two_distinct_committed_past_frames_and_copies_obb() -> None:
    _, queries = _three_frame_replay(hb_confidence=0.55)
    first, second, third = (query.rows[0] for query in queries)
    assert first["selected_hypothesis"] == "HLG"
    assert first["hb_abstention_reason"] == "history_count"
    assert first["matched_past_frame_count"] == 0
    assert second["selected_hypothesis"] == "HLG"
    assert second["hb_abstention_reason"] == "history_count"
    assert second["matched_past_frame_count"] == 1

    assert third["selected_hypothesis"] == "HB"
    assert third["hb_abstention_reason"] is None
    assert third["matched_past_frame_count"] == 2
    assert third["hb_consistent_past_frame_count"] == 2
    assert [row["frame_ordinal"] for row in third["matched_past"]] == [0, 1]
    assert len({row["frame_ordinal"] for row in third["matched_past"]}) == 2
    selected = third["selected_geometry"]
    input_hb = _source(2, hb_confidence=0.55).hypotheses["HB"]
    assert selected["world_center"] == input_hb["world_center"]
    assert selected["local_extent"] == input_hb["local_extent"]
    assert selected["world_rotation"] == input_hb["world_rotation"]
    assert selected["world_corners"] == input_hb["world_corners"]
    assert third["formal_score"] == 1.0


def test_current_geometry_and_projection_fail_closed_before_history() -> None:
    outside = np.tile(np.asarray([4.0, 4.0, 4.0]), (20, 1))
    source = _source(0, clean_base=False, points=outside)
    row = f5.F5SelectorState().query_frame(
        frame_id=0, frame_ordinal=0, sources=(source,)
    ).rows[0]
    assert row["selected_hypothesis"] == "H0"
    assert row["hb_abstention_reason"] == "exact_depth_support"

    source = _source(
        0,
        clean_base=False,
        tight_box=np.asarray([0.0, 0.0, 10.0, 10.0]),
    )
    row = f5.F5SelectorState().query_frame(
        frame_id=0, frame_ordinal=0, sources=(source,)
    ).rows[0]
    assert row["selected_hypothesis"] == "H0"
    assert row["hb_abstention_reason"] == "projection_iou"


def test_current_gate_abstention_still_seals_all_mutual_best_past_receipts() -> None:
    state = f5.F5SelectorState()
    for frame in range(2):
        state.select_frame(
            frame_id=frame,
            frame_ordinal=frame,
            sources=(_source(frame),),
        )
    query = state.query_frame(
        frame_id=2,
        frame_ordinal=2,
        sources=(_source(2, hb_confidence=0.54),),
    )
    row = query.rows[0]
    assert row["hb_abstention_reason"] == "confidence_threshold"
    assert row["matched_past_frame_count"] == 2
    assert [receipt["frame_ordinal"] for receipt in row["matched_past"]] == [0, 1]
    assert all(receipt["hb_consistency_evaluated"] is False for receipt in row["matched_past"])
    assert all(receipt["passed_hb_consistency"] is None for receipt in row["matched_past"])
    state.commit_frame(query)


def test_query_commit_requires_exact_unmodified_token_object() -> None:
    state = f5.F5SelectorState()
    query = state.query_frame(frame_id=0, frame_ordinal=0, sources=(_source(0),))
    forged = replace(query)
    with pytest.raises(f5.F5ContractError, match="exact pending"):
        state.commit_frame(forged)

    # A frozen dataclass is insufficient when its nested result mappings are
    # mutable.  Exact-token commit must re-hash and reject this in-place edit.
    query.rows[0]["formal_score"] = 0.5
    with pytest.raises(f5.F5ContractError, match="token|hash|changed|mutat"):
        state.commit_frame(query)


def test_source_id_frame_binding_duplicate_order_and_scene_isolation() -> None:
    with pytest.raises(f5.F5ContractError, match="source_id|frame"):
        _source(0, source_frame_in_id=999999)

    state = f5.F5SelectorState()
    duplicated = _source(0)
    with pytest.raises(f5.F5ContractError, match="duplicate|rank"):
        state.query_frame(
            frame_id=0, frame_ordinal=0, sources=(duplicated, duplicated)
        )

    with pytest.raises(f5.F5ContractError, match="order"):
        f5.F5SelectorState().query_frame(
            frame_id=0,
            frame_ordinal=0,
            sources=(_source(0, rank=1), _source(0, rank=0)),
        )

    # Sorted is not sufficient: the sealed per-frame ranks are exactly the
    # contiguous source-row ordinals, with neither duplicates nor gaps.
    with pytest.raises(f5.F5ContractError, match="rank|order"):
        f5.F5SelectorState().query_frame(
            frame_id=0,
            frame_ordinal=0,
            sources=(
                _source(0, rank=0, raw_index=0),
                _source(0, rank=0, raw_index=7),
            ),
        )
    with pytest.raises(f5.F5ContractError, match="rank|order"):
        f5.F5SelectorState().query_frame(
            frame_id=0,
            frame_ordinal=0,
            sources=(
                _source(0, rank=0, raw_index=0),
                _source(0, rank=2, raw_index=7),
            ),
        )

    state = f5.F5SelectorState()
    state.select_frame(frame_id=0, frame_ordinal=0, sources=(_source(0),))
    with pytest.raises(f5.F5ContractError, match="scene"):
        state.query_frame(
            frame_id=1,
            frame_ordinal=1,
            sources=(_source(1, scene="scene0001_00"),),
        )


def test_pending_query_hides_current_frame_and_buffer_is_strictly_bounded() -> None:
    state = f5.F5SelectorState()
    query = state.query_frame(frame_id=0, frame_ordinal=0, sources=(_source(0),))
    assert query.maximum_accessed_frame_ordinal == -1
    assert query.buffer_before == ()
    assert state.buffered_frame_count == 0
    with pytest.raises(f5.F5ContractError, match="not been committed"):
        state.query_frame(frame_id=1, frame_ordinal=1, sources=(_source(1),))
    state.commit_frame(query)

    for frame in range(1, 5):
        query, commit = state.select_frame(
            frame_id=frame,
            frame_ordinal=frame,
            sources=(_source(frame),),
        )
        assert query.maximum_accessed_frame_ordinal < frame
        assert len(commit.buffer_after) <= 3
    assert state.buffered_frame_count == 3


def test_ordinal_gap_excludes_expired_history_from_decision_and_receipt() -> None:
    state = f5.F5SelectorState()
    for frame in range(3):
        state.select_frame(
            frame_id=frame,
            frame_ordinal=frame,
            sources=(_source(frame),),
        )

    # The physical ring still has frames 0--2, but none lies in the frozen
    # [current-3, current-1] causal window for ordinal 10.  The sealed query
    # receipt must describe exactly the history the decision was allowed to
    # inspect, rather than leaking stale buffer contents into its token.
    query = state.query_frame(
        frame_id=10,
        frame_ordinal=10,
        sources=(_source(10),),
    )
    assert query.maximum_accessed_frame_ordinal == -1
    assert query.buffer_before == ()
    assert query.rows[0]["matched_past"] == []
    assert query.rows[0]["selected_hypothesis"] == "HLG"
    state.commit_frame(query)


def test_future_perturbation_and_independent_cpu_replays_preserve_prefix_hashes() -> None:
    def replay(future_center: tuple[float, float, float]) -> tuple[list[str], str]:
        state = f5.F5SelectorState()
        prefix = []
        for frame in range(3):
            query, _ = state.select_frame(
                frame_id=frame,
                frame_ordinal=frame,
                sources=(_source(frame),),
            )
            prefix.append(str(query.rows[0]["result_sha256"]))
        future, _ = state.select_frame(
            frame_id=3,
            frame_ordinal=3,
            sources=(_source(3, center=future_center),),
        )
        return prefix, str(future.rows[0]["result_sha256"])

    left_prefix, left_future = replay((0.0, 0.0, 2.0))
    right_prefix, right_future = replay((4.0, 0.0, 2.0))
    assert left_prefix == right_prefix
    assert left_future != right_future

    # A third replay uses independent object/array copies and must reproduce
    # the same ordered result ledger exactly.
    third_prefix, _ = replay(tuple(np.asarray([0.0, 0.0, 2.0]).copy()))
    assert third_prefix == left_prefix


def test_output_count_identity_hash_and_score_are_one_to_one() -> None:
    sources = (_source(0, rank=0), _source(0, rank=1, raw_index=7))
    source_ids = [source.source_id for source in sources]
    query, commit = f5.F5SelectorState().select_frame(
        frame_id=0, frame_ordinal=0, sources=sources
    )
    assert commit.source_count == len(sources) == len(query.rows)
    assert [row["source_id"] for row in query.rows] == source_ids
    assert len({row["source_id"] for row in query.rows}) == len(query.rows)
    for source, row in zip(sources, query.rows, strict=True):
        assert row["selected_hypothesis"] in {"H0", "HL", "HLG", "HB"}
        assert row["input_hypothesis_sha256"] == {
            name: f5.canonical_json_sha256(source.hypotheses.get(name))
            for name in ("H0", "HL", "HLG", "HB")
        }
        assert row["selected_geometry_sha256"] == f5.canonical_json_sha256(
            row["selected_geometry"]
        )
        assert row["result_sha256"] == f5.canonical_result_sha256(row)
        assert type(row["formal_score"]) is float and row["formal_score"] == 1.0
