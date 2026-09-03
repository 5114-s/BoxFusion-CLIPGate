from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import tools.audit_scannet_fastsam_f3_paper100_oracle as oracle
from tools.audit_scannet_boxer_unexplained_oracle import official_constant_evaluate


def _track(index: int, *, chosen: str | None = None) -> oracle.F3Track:
    geometry = np.asarray([0, 0, 0, 1, 1, 1], dtype=np.float64)
    return oracle.F3Track(
        scene_id="scene0001_00",
        track_id=index,
        source_ids=(f"source-{index}",),
        frame_ids=(index,),
        observation_count=1,
        confirmed=False,
        world_minmax={"B": geometry, "C": geometry, "SELECTOR": geometry if chosen else None},
        aligned_minmax={"B": geometry, "C": geometry, "SELECTOR": geometry if chosen else None},
        valid={"B": True, "C": True, "SELECTOR": chosen is not None},
        scores={"B": 0.4, "C": 0.5, "SELECTOR": 0.5 if chosen else None},
        selector_chosen=chosen,
    )


def test_grouped_matrix_is_one_row_per_track_and_ties_prefer_b():
    matrices = {
        "B": np.asarray([[0.9, 0.0], [0.7, 0.1]]),
        "C": np.asarray([[0.0, 0.9], [0.7, 0.8]]),
    }
    grouped = oracle.grouped_iou_matrix(matrices)
    np.testing.assert_array_equal(grouped, [[0.9, 0.9], [0.7, 0.8]])
    assert oracle.choose_hypothesis_for_edge(matrices, 1, 0) == ("B", 0.7)
    assert oracle.choose_hypothesis_for_edge(matrices, 1, 1) == ("C", 0.8)


def test_grouped_oracle_never_stacks_b_and_c_as_two_objects():
    matrices = {
        "B": np.asarray([[0.9, 0.0]]),
        "C": np.asarray([[0.0, 0.9]]),
        "SELECTOR": np.asarray([[0.9, 0.0]]),
    }
    baseline = official_constant_evaluate([np.empty((0, 2))], [2], 0.50)
    report = oracle.evaluate_f3_threshold(
        scenes=["scene0001_00"],
        native_iou=[np.empty((0, 2))],
        track_iou=[matrices],
        tracks=[[_track(0, chosen="B")]],
        gt_counts=[2],
        baseline_evaluation=baseline,
        threshold=0.50,
    )["identity_constrained_grouped"]
    assert report["candidate_maximum_matching_count"] == 1
    assert report["union_maximum_matching_count"] == 1
    assert report["gt_selected_track_suffix"]["selected_track_count"] == 1


def test_grouped_matching_uses_augmenting_path_without_reusing_track():
    matrices = {
        "B": np.asarray([[0.8, 0.7], [0.6, 0.0]]),
        "C": np.asarray([[0.0, 0.9], [0.0, 0.0]]),
        "SELECTOR": np.asarray([[0.8, 0.7], [0.6, 0.0]]),
    }
    baseline = official_constant_evaluate([np.empty((0, 2))], [2], 0.50)
    report = oracle.evaluate_f3_threshold(
        scenes=["scene0001_00"],
        native_iou=[np.empty((0, 2))],
        track_iou=[matrices],
        tracks=[[_track(0, chosen="B"), _track(1, chosen="B")]],
        gt_counts=[2],
        baseline_evaluation=baseline,
        threshold=0.50,
    )["identity_constrained_grouped"]
    assert report["candidate_maximum_matching_count"] == 2
    selected = report["gt_selected_track_suffix"]["per_scene_selection"]["scene0001_00"]
    assert len({row["track_id"] for row in selected}) == 2
    assert len({row["target_gt_index"] for row in selected}) == 2


def test_strict_threshold_excludes_exact_equality():
    matrices = {
        "B": np.asarray([[0.50]]),
        "C": np.asarray([[0.00]]),
        "SELECTOR": np.asarray([[0.50]]),
    }
    baseline = official_constant_evaluate([np.empty((0, 1))], [1], 0.50)
    result = oracle.evaluate_f3_threshold(
        scenes=["scene0001_00"],
        native_iou=[np.empty((0, 1))],
        track_iou=[matrices],
        tracks=[[_track(0, chosen="B")]],
        gt_counts=[1],
        baseline_evaluation=baseline,
        threshold=0.50,
    )
    assert result["strict_iou_comparison"] == ">"
    assert result["hypothesis_only"]["B"]["candidate_maximum_matching_count"] == 0


def test_fixed_selector_all_tracks_is_separate_and_reports_tp_fp():
    matrices = {
        "B": np.asarray([[0.9], [0.0]]),
        "C": np.asarray([[0.0], [0.0]]),
        "SELECTOR": np.asarray([[0.9], [0.0]]),
    }
    baseline = official_constant_evaluate([np.empty((0, 1))], [1], 0.50)
    result = oracle.evaluate_f3_threshold(
        scenes=["scene0001_00"],
        native_iou=[np.empty((0, 1))],
        track_iou=[matrices],
        tracks=[[_track(0, chosen="B"), _track(1, chosen="B")]],
        gt_counts=[1],
        baseline_evaluation=baseline,
        threshold=0.50,
    )["fixed_no_gt_selector"]["fixed_selector_all_tracks"]
    assert result["geometry_selection_uses_gt"] is False
    assert result["prediction_materialized"] is False
    assert result["appended_track_count"] == 2
    assert result["greedy_tp"] == 1
    assert result["false_positive"] == 1


def _minimal_h0_reports():
    f1 = {"per_threshold": {}}
    actual = {}
    for threshold in oracle.THRESHOLDS:
        key = f"{threshold:.2f}"
        common = {
            "candidate_maximum_matching_count": 2,
            "union_maximum_matching_count": 3,
            "additional_union_matching_over_native": 1,
        }
        f1["per_threshold"][key] = common | {
            "gt_selected_candidate_suffix": {
                "selected_candidate_count": 1,
                "official_evaluation": {"ap_points": 12.5},
                "delta_ap_points": 2.5,
            }
        }
        actual[key] = common | {
            "gt_selected_candidate_suffix": {
                "selected_candidate_count": 1,
                "official_evaluation": {"ap_points": 12.5},
                "delta_ap_points": 2.5,
            }
        }
    return f1, actual


def test_h0_f1_reproduction_passes_and_fails_closed():
    f1, actual = _minimal_h0_reports()
    checks = oracle.validate_h0_reproduces_f1(actual, f1)
    assert all(row["passed"] for row in checks.values())
    actual["0.50"]["union_maximum_matching_count"] = 4
    with pytest.raises(oracle.F3OracleError, match="failed to reproduce"):
        oracle.validate_h0_reproduces_f1(actual, f1)


def test_f3_retention_gate_passes_at_78_but_never_authorizes_birth():
    result = oracle.f3_retention_decision(
        grouped_ap50_additional_union_matches=78,
        h0_identity_passed=True,
        runtime_passed=True,
        causality_passed=True,
        final_geometry_capacity_passed=False,
        final_plus10_ap_passed=False,
    )
    assert result["actual_ap50_additional_match_gain_over_f1"] == 15
    assert result["ap50_retention_capacity_passed"] is True
    assert result["retain_f3_for_next_selector_filter_experiment"] is True
    assert result["authorize_active_birth"] is False


@pytest.mark.parametrize("additional,runtime,causal", [(77, True, True), (100, False, True), (100, True, False)])
def test_f3_retention_fails_capacity_runtime_or_causality(additional, runtime, causal):
    result = oracle.f3_retention_decision(
        grouped_ap50_additional_union_matches=additional,
        h0_identity_passed=True,
        runtime_passed=runtime,
        causality_passed=causal,
        final_geometry_capacity_passed=False,
        final_plus10_ap_passed=False,
    )
    assert result["retain_f3_for_next_selector_filter_experiment"] is False
    assert result["authorize_active_birth"] is False


def _runtime(passed: bool = True):
    gates = {}
    for name, (comparator, threshold) in oracle.RUNTIME_GATE_SPECS.items():
        if passed:
            actual = 0.0 if comparator == "==" else threshold * 0.5
        else:
            actual = 1.0 if comparator == "==" else threshold * 2.0
        gates[name] = {
            "actual": actual,
            "threshold": threshold,
            "comparator": comparator,
            "passed": passed,
        }
    return {"gates": gates, "overall_pass": passed}


def _contracts():
    return {
        "shadow_only": True,
        "birth_enabled": False,
        "ground_truth_access": False,
        "prediction_access": False,
        "evaluator_access": False,
        "native_output_mutation": False,
        "training": False,
        "online_learning": False,
        "future_frame_logical_access": False,
        "hl_hlg_access": False,
    }


def _causality():
    return {
        "prefix_invariance": True,
        "query_before_commit": True,
        "one_source_one_track": True,
        "maximum_logical_accessed_ordinal": True,
        "overall_pass": True,
    }


def test_runtime_gate_failure_is_valid_evidence_and_inconsistency_fails():
    checked = oracle._validate_runtime(_runtime(False))
    assert checked["overall_pass"] is False
    broken = _runtime(True)
    broken["gates"]["f3_incremental_mean_ms"]["passed"] = False
    with pytest.raises(oracle.F3OracleError, match="inconsistent"):
        oracle._validate_runtime(broken)


def test_merge_receipt_schema_and_runtime_contract():
    scenes = ["scene0001_00"]
    receipt = {
        "schema": oracle.F3_RECEIPT_SCHEMA,
        "protocol_id": oracle.F3_PROTOCOL_ID,
        "complete": True,
        "overall_pass": True,
        "run_signature_sha256": "d" * 64,
        "contracts": _contracts(),
        "coverage": {
            "scene_order": scenes,
            "scene_count": 1,
            "keyframe_count": 3,
            "successful_frame_count": 3,
            "source_count": 3,
        },
        "integrity": {"overall_pass": True},
        "causality": _causality(),
        "runtime": _runtime(True),
        "scenes": [
            {
                "scene_id": scenes[0],
                "index": 0,
                "sidecar": {"path": "scene0001_00.json", "sha256": "a" * 64},
                "counts": {},
            }
        ],
    }
    rows, signature, runtime_pass, runtime = oracle._validate_f3_receipt(
        receipt,
        scenes,
        expected_scene_count=1,
        expected_keyframe_count=3,
        expected_successful_frame_count=3,
        expected_source_count=3,
    )
    assert list(rows) == scenes
    assert signature == "d" * 64
    assert runtime_pass is True
    assert runtime["overall_pass"] is True


def _hypothesis(valid=True):
    if not valid:
        return {
            "valid": False,
            "reason": "unavailable",
            "q02": None,
            "q98": None,
            "center": None,
            "extent": None,
            "score": None,
            "valid_fold_count": 0,
            "fold_ious": [],
        }
    return {
        "valid": True,
        "reason": "valid",
        "q02": [0.0, 0.0, 0.0],
        "q98": [1.0, 1.0, 1.0],
        "center": [0.5, 0.5, 0.5],
        "extent": [1.0, 1.0, 1.0],
        "score": 0.3,
        "valid_fold_count": 2,
        "fold_ious": [0.2, 0.4],
    }


def test_selector_independently_rechecks_c_stability_and_safety():
    b = _hypothesis(True)
    b["score"] = 0.3
    b["fold_ious"] = [0.2, 0.4]
    c = _hypothesis(True)
    c.update(
        {
            "score": 0.4,
            "fold_ious": [0.3, 0.5],
            "loo_full_aabb_ious": [0.5, 0.6],
            "center_shift_from_b_m": 0.0,
            "extent_ratios": [1.0, 1.0, 1.0],
            "volume_ratio": 1.0,
        }
    )
    parsed = {
        name: oracle._hypothesis_geometry(value, np.eye(4), name)
        for name, value in {"B": b, "C": c}.items()
    }
    selector = {
        "chosen": "C",
        "reason": "C_valid_and_gain_at_least_0.03",
        "q02": c["q02"],
        "q98": c["q98"],
        "center": c["center"],
        "extent": c["extent"],
        "score": c["score"],
    }
    chosen, *_ = oracle._validate_selector(
        selector,
        hypotheses={"B": b, "C": c},
        parsed=parsed,
        alignment=np.eye(4),
        label="selector",
    )
    assert chosen == "C"
    c["loo_full_aabb_ious"] = [0.2]
    with pytest.raises(oracle.F3OracleError, match="stability"):
        oracle._validate_selector(
            selector,
            hypotheses={"B": b, "C": c},
            parsed=parsed,
            alignment=np.eye(4),
            label="selector",
        )


def _write_sidecars(tmp_path: Path):
    scene = "scene0001_00"
    f0_frames = []
    f3_frames = []
    sources = []
    for ordinal, frame_id in enumerate((0, 25, 50)):
        raw_index = ordinal + 3
        source = f"{scene}/frame_{frame_id:06d}/raw_{raw_index:03d}"
        sources.append(source)
        f0_frames.append(
            {
                "frame_id": frame_id,
                "frame_ordinal": ordinal,
                "successful": True,
                "funnel": {"candidates": [{"raw_index": raw_index}]},
            }
        )
        f3_frames.append(
            {
                "frame_id": frame_id,
                "ordinal": ordinal,
                "successful": True,
                "source_ids": [source],
                "assignments": [
                    {
                        "source_id": source,
                        "track_id": 0,
                        "action": "create" if ordinal == 0 else "match",
                    }
                ],
                "retired_ids": [],
                "max_logical_accessed_ordinal": ordinal,
                "f3_core_ms": 1.0,
            }
        )
    f0 = {"schema": oracle.F0_SCENE_SCHEMA, "scene_id": scene, "frames": f0_frames}
    f0_path = tmp_path / "f0.json"
    f0_path.write_text(json.dumps(f0, sort_keys=True), encoding="utf-8")

    b = _hypothesis(True)
    c = _hypothesis(False)
    signature = "c" * 64
    f3 = {
        "schema": oracle.F3_SCENE_SCHEMA,
        "protocol_id": oracle.F3_PROTOCOL_ID,
        "complete": True,
        "scene_id": scene,
        "scene_index": 0,
        "run_signature_sha256": signature,
        "contracts": _contracts(),
        "inputs": {},
        "counts": {
            "keyframe_count": 3,
            "successful_frame_count": 3,
            "source_count": 3,
            "track_count": 1,
            "confirmed_track_count": 1,
            "selected_track_count": 1,
        },
        "causality": _causality(),
        "runtime": {"f3_core_mean_ms": 1.0},
        "frames": f3_frames,
        "tracks": [
            {
                "track_id": 0,
                "source_ids": sources,
                "frame_ids": [0, 25, 50],
                "observation_count": 3,
                "retained_source_ids": sources,
                "retained_frame_ids": [0, 25, 50],
                "retained_observation_count": 3,
                "confirmed": True,
                "hypotheses": {"B": b, "C": c},
                "selector": {
                    "chosen": "B",
                    "reason": "B_valid",
                    "q02": b["q02"],
                    "q98": b["q98"],
                    "center": b["center"],
                    "extent": b["extent"],
                    "score": b["score"],
                },
            }
        ],
    }
    f3_path = tmp_path / "f3.json"
    f3_path.write_text(json.dumps(f3, sort_keys=False), encoding="utf-8")
    digest = hashlib.sha256(f3_path.read_bytes()).hexdigest()
    return f0_path, f3_path, digest, signature, f3


def test_scene_loader_proves_source_coverage_causality_and_selector_copy(tmp_path: Path):
    f0_path, f3_path, digest, signature, _ = _write_sidecars(tmp_path)
    tracks, counts, diagnostics = oracle._load_f3_tracks(
        path=f3_path,
        f0_path=f0_path,
        scene="scene0001_00",
        scene_index=0,
        alignment=np.eye(4),
        receipt_sidecar_sha256=digest,
        run_signature_sha256=signature,
    )
    assert counts["source_count"] == 3
    assert len(tracks) == 1
    assert tracks[0].selector_chosen == "B"
    assert diagnostics["max_logical_accessed_ordinal"] == 2


def test_scene_loader_rejects_future_access_and_selector_drift(tmp_path: Path):
    f0_path, f3_path, _, signature, f3 = _write_sidecars(tmp_path)
    f3["frames"][0]["max_logical_accessed_ordinal"] = 1
    f3_path.write_text(json.dumps(f3), encoding="utf-8")
    digest = hashlib.sha256(f3_path.read_bytes()).hexdigest()
    with pytest.raises(oracle.F3OracleError, match="future-frame"):
        oracle._load_f3_tracks(
            path=f3_path,
            f0_path=f0_path,
            scene="scene0001_00",
            scene_index=0,
            alignment=np.eye(4),
            receipt_sidecar_sha256=digest,
            run_signature_sha256=signature,
        )

    _, f3_path, _, signature, f3 = _write_sidecars(tmp_path)
    f3["tracks"][0]["selector"]["q98"] = [1.1, 1.0, 1.0]
    f3["tracks"][0]["selector"]["center"] = [0.55, 0.5, 0.5]
    f3["tracks"][0]["selector"]["extent"] = [1.1, 1.0, 1.0]
    f3_path.write_text(json.dumps(f3), encoding="utf-8")
    digest = hashlib.sha256(f3_path.read_bytes()).hexdigest()
    with pytest.raises(oracle.F3OracleError, match="does not exactly copy"):
        oracle._load_f3_tracks(
            path=f3_path,
            f0_path=f0_path,
            scene="scene0001_00",
            scene_index=0,
            alignment=np.eye(4),
            receipt_sidecar_sha256=digest,
            run_signature_sha256=signature,
        )


def test_scene_loader_allows_same_frame_created_id_permutation(tmp_path: Path):
    scene = "scene0001_00"
    source_rank0 = f"{scene}/frame_000000/raw_009"
    source_rank1 = f"{scene}/frame_000000/raw_003"
    f0 = {
        "schema": oracle.F0_SCENE_SCHEMA,
        "scene_id": scene,
        "frames": [
            {
                "frame_id": 0,
                "frame_ordinal": 0,
                "successful": True,
                "funnel": {"candidates": [{"raw_index": 9}, {"raw_index": 3}]},
            }
        ],
    }
    f0_path = tmp_path / "same_frame_f0.json"
    f0_path.write_text(json.dumps(f0), encoding="utf-8")
    invalid = _hypothesis(False)
    tracks = []
    # The core allocates IDs in lexical source order (raw_003 then raw_009),
    # while the public assignment ledger stays in F0 rank order.
    for track_id, source in ((0, source_rank1), (1, source_rank0)):
        tracks.append(
            {
                "track_id": track_id,
                "source_ids": [source],
                "frame_ids": [0],
                "observation_count": 1,
                "retained_source_ids": [source],
                "retained_frame_ids": [0],
                "retained_observation_count": 1,
                "confirmed": False,
                "hypotheses": {"B": dict(invalid), "C": dict(invalid)},
                "selector": {
                    "chosen": None,
                    "reason": "abstain",
                    "q02": None,
                    "q98": None,
                    "center": None,
                    "extent": None,
                    "score": None,
                },
            }
        )
    signature = "e" * 64
    f3 = {
        "schema": oracle.F3_SCENE_SCHEMA,
        "protocol_id": oracle.F3_PROTOCOL_ID,
        "complete": True,
        "scene_id": scene,
        "scene_index": 0,
        "run_signature_sha256": signature,
        "contracts": _contracts(),
        "inputs": {},
        "counts": {
            "keyframe_count": 1,
            "successful_frame_count": 1,
            "source_count": 2,
            "track_count": 2,
            "confirmed_track_count": 0,
            "selected_track_count": 0,
        },
        "causality": _causality(),
        "runtime": {},
        "frames": [
            {
                "frame_id": 0,
                "ordinal": 0,
                "successful": True,
                "source_ids": [source_rank0, source_rank1],
                "assignments": [
                    {"source_id": source_rank0, "track_id": 1, "action": "created"},
                    {"source_id": source_rank1, "track_id": 0, "action": "created"},
                ],
                "retired_ids": [],
                "max_logical_accessed_ordinal": 0,
                "f3_core_ms": 0.1,
            }
        ],
        "tracks": tracks,
    }
    f3_path = tmp_path / "same_frame_f3.json"
    f3_path.write_text(json.dumps(f3), encoding="utf-8")
    digest = hashlib.sha256(f3_path.read_bytes()).hexdigest()
    loaded, counts, _ = oracle._load_f3_tracks(
        path=f3_path,
        f0_path=f0_path,
        scene=scene,
        scene_index=0,
        alignment=np.eye(4),
        receipt_sidecar_sha256=digest,
        run_signature_sha256=signature,
    )
    assert [track.track_id for track in loaded] == [0, 1]
    assert counts["source_count"] == 2


def test_grouped_matrix_rejects_wrong_order_and_shape():
    with pytest.raises(oracle.F3OracleError, match="order"):
        oracle.grouped_iou_matrix({"C": np.zeros((1, 1)), "B": np.zeros((1, 1))})
    with pytest.raises(oracle.F3OracleError, match="identical"):
        oracle.grouped_iou_matrix({"B": np.zeros((1, 1)), "C": np.zeros((2, 1))})


def test_output_is_create_only_and_main_refuses_overwrite(tmp_path: Path, monkeypatch):
    protected = tmp_path / "inputs"
    protected.mkdir()
    with pytest.raises(oracle.F3OracleError, match="protected"):
        oracle._validate_output_path(protected / "report.json", [protected])
    with pytest.raises(oracle.F3OracleError, match="suffix"):
        oracle._validate_output_path(tmp_path / "report.txt", [protected])

    out = tmp_path / "reports" / "f3.json"
    fake = {
        "schema": oracle.SCHEMA,
        "totals": {"scene_count": 100},
        "decision": {"overall_pass": False},
    }
    monkeypatch.setattr(oracle, "audit_scannet_fastsam_f3_paper100_oracle", lambda **_: fake)
    assert oracle.main(["--out", str(out)]) == 0
    assert json.loads(out.read_text(encoding="utf-8")) == fake
    before = out.read_bytes()
    with pytest.raises(oracle.F3OracleError, match="overwrite"):
        oracle.main(["--out", str(out)])
    assert out.read_bytes() == before
