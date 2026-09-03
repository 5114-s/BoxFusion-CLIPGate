from __future__ import annotations

import json
import hashlib
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
for _directory in (ROOT / "tools", ROOT / "tests"):
    if str(_directory) not in sys.path:
        sys.path.insert(0, str(_directory))

import merge_scannet_fastsam_f2_paper100 as f2_merger
from test_run_scannet_fastsam_f2_paper100 import _prepare_f2

import tools.audit_scannet_fastsam_f2_paper100_oracle as oracle
from tools.audit_scannet_boxer_unexplained_oracle import official_constant_evaluate
from tools.audit_scannet_fastsam_f2_paper100_oracle import (
    F2OracleError,
    F2Source,
    _load_f2_sources,
    _validate_runtime_gates,
    _validate_output_path,
    _validate_f2_receipt,
    choose_hypothesis_for_edge,
    evaluate_f2_threshold,
    f2_retention_decision,
    grouped_iou_matrix,
    validate_h0_reproduces_f1,
)


def _source(index: int) -> F2Source:
    geometry = {name: np.zeros(6, dtype=np.float64) for name in oracle.HYPOTHESES}
    return F2Source(
        scene_id="scene0001_00",
        frame_id=0,
        frame_ordinal=0,
        candidate_index=index,
        rank=index,
        raw_index=index,
        source_id=f"scene0001_00/frame_000000/raw_{index:03d}",
        world_minmax=geometry,
        aligned_minmax=geometry,
        applied={"H0": True, "HL": True, "HLG": True},
    )


def test_grouped_iou_uses_one_row_per_source_not_one_row_per_hypothesis():
    # One physical source has a valid edge to GT0 through H0 and to GT1 through
    # HL. Stacking hypotheses would incorrectly claim two matches. Grouping
    # leaves one source row and therefore permits only one match.
    matrices = {
        "H0": np.asarray([[0.90, 0.00]]),
        "HL": np.asarray([[0.00, 0.90]]),
        "HLG": np.asarray([[0.00, 0.00]]),
    }
    grouped = grouped_iou_matrix(matrices)
    np.testing.assert_array_equal(grouped, [[0.90, 0.90]])

    baseline = official_constant_evaluate([np.empty((0, 2))], [2], 0.50)
    report = evaluate_f2_threshold(
        scenes=["scene0001_00"],
        native_iou=[np.empty((0, 2))],
        hypothesis_iou=[matrices],
        sources=[[_source(0)]],
        gt_counts=[2],
        baseline_evaluation=baseline,
        threshold=0.50,
    )
    grouped_report = report["identity_constrained_grouped"]
    assert grouped_report["candidate_maximum_matching_count"] == 1
    assert grouped_report["union_maximum_matching_count"] == 1
    assert grouped_report["gt_selected_candidate_suffix"]["selected_source_count"] == 1


def test_grouped_matching_can_use_augmenting_path_without_reusing_source():
    matrices = {
        # source0 can reach both GTs, source1 can only reach GT0. The maximum
        # matching must move source0 to GT1 and still use each source once.
        "H0": np.asarray([[0.80, 0.70], [0.60, 0.00]]),
        "HL": np.asarray([[0.00, 0.90], [0.00, 0.00]]),
        "HLG": np.asarray([[0.00, 0.00], [0.00, 0.00]]),
    }
    baseline = official_constant_evaluate([np.empty((0, 2))], [2], 0.50)
    report = evaluate_f2_threshold(
        scenes=["scene0001_00"],
        native_iou=[np.empty((0, 2))],
        hypothesis_iou=[matrices],
        sources=[[_source(0), _source(1)]],
        gt_counts=[2],
        baseline_evaluation=baseline,
        threshold=0.50,
    )["identity_constrained_grouped"]
    assert report["candidate_maximum_matching_count"] == 2
    selection = report["gt_selected_candidate_suffix"]["per_scene_selection"]["scene0001_00"]
    assert len({row["source_id"] for row in selection}) == 2
    assert len({row["target_gt_index"] for row in selection}) == 2


def test_choose_hypothesis_uses_highest_target_iou_and_h0_tie_priority():
    matrices = {
        "H0": np.asarray([[0.7, 0.6]]),
        "HL": np.asarray([[0.7, 0.8]]),
        "HLG": np.asarray([[0.7, 0.9]]),
    }
    assert choose_hypothesis_for_edge(matrices, 0, 0) == ("H0", 0.7)
    assert choose_hypothesis_for_edge(matrices, 0, 1) == ("HLG", 0.9)


def test_exact_threshold_is_not_an_edge_and_group_contains_h0_edges():
    matrices = {
        "H0": np.asarray([[0.50, 0.60]]),
        "HL": np.asarray([[0.90, 0.00]]),
        "HLG": np.asarray([[0.00, 0.00]]),
    }
    baseline = official_constant_evaluate([np.empty((0, 2))], [2], 0.50)
    report = evaluate_f2_threshold(
        scenes=["scene0001_00"],
        native_iou=[np.empty((0, 2))],
        hypothesis_iou=[matrices],
        sources=[[_source(0)]],
        gt_counts=[2],
        baseline_evaluation=baseline,
        threshold=0.50,
    )
    grouped = report["identity_constrained_grouped"]
    assert grouped["edges_vs_H0"]["lost_gt_edges"] == 0
    assert grouped["edges_vs_H0"]["gained_gt_edges"] == 1
    assert report["strict_iou_comparison"] == ">"


def _minimal_f1_and_f2_reports():
    f1 = {"per_threshold": {}}
    f2 = {}
    for threshold in oracle.THRESHOLDS:
        key = f"{threshold:.2f}"
        expected = {
            "candidate_maximum_matching_count": 2,
            "union_maximum_matching_count": 3,
            "additional_union_matching_over_native": 1,
            "gt_selected_candidate_suffix": {
                "selected_candidate_count": 1,
                "official_evaluation": {"ap_points": 12.5},
                "delta_ap_points": 2.5,
            },
        }
        actual = {
            "hypothesis_only": {
                "H0": {
                    "candidate_maximum_matching_count": 2,
                    "union_maximum_matching_count": 3,
                    "additional_union_matching_over_native": 1,
                    "gt_selected_candidate_suffix": {
                        "selected_source_count": 1,
                        "official_evaluation": {"ap_points": 12.5},
                        "delta_ap_points": 2.5,
                    },
                }
            }
        }
        f1["per_threshold"][key] = expected
        f2[key] = actual
    return f1, f2


def test_h0_reproduction_check_passes_and_fails_closed():
    f1, f2 = _minimal_f1_and_f2_reports()
    checks = validate_h0_reproduces_f1(f2, f1)
    assert all(row["passed"] for row in checks.values())
    f2["0.50"]["hypothesis_only"]["H0"]["union_maximum_matching_count"] = 4
    with pytest.raises(F2OracleError, match="H0 failed"):
        validate_h0_reproduces_f1(f2, f1)


def test_f2_retention_gate_passes_at_78_and_never_authorizes_birth():
    decision = f2_retention_decision(
        grouped_ap50_additional_union_matches=78,
        h0_identity_passed=True,
        merged_runtime_overall_pass=True,
        later_joint_active_capacity_passed=True,
    )
    assert decision["actual_ap50_additional_match_gain_over_f1"] == 15
    assert decision["ap50_retention_capacity_passed"] is True
    assert decision["retain_f2_geometry_for_f3"] is True
    assert decision["f3_shadow_geometry_input"] == (
        "F2_H0_HL_HLG_hypothesis_set_for_GT_free_F3_selection"
    )
    assert decision["grouped_oracle_geometry_exported"] is False
    assert decision["authorize_grouped_oracle_geometry"] is False
    assert decision["authorize_active_birth"] is False


def test_f2_retention_gate_discards_at_77():
    decision = f2_retention_decision(
        grouped_ap50_additional_union_matches=77,
        h0_identity_passed=True,
        merged_runtime_overall_pass=True,
        later_joint_active_capacity_passed=False,
    )
    assert decision["actual_ap50_additional_match_gain_over_f1"] == 14
    assert decision["ap50_retention_capacity_passed"] is False
    assert decision["retain_f2_geometry_for_f3"] is False
    assert decision["f3_shadow_geometry_input"] == "F1_H0_only"


def test_f2_retention_gate_fails_when_merged_runtime_fails():
    decision = f2_retention_decision(
        grouped_ap50_additional_union_matches=100,
        h0_identity_passed=True,
        merged_runtime_overall_pass=False,
        later_joint_active_capacity_passed=True,
    )
    assert decision["ap50_retention_capacity_passed"] is True
    assert decision["merged_runtime_overall_pass"] is False
    assert decision["retain_f2_geometry_for_f3"] is False
    assert decision["authorize_active_birth"] is False


def _runtime_gates(passed: bool = True):
    actual = 1.0 if passed else 3.0
    items = {
        name: {
            "actual": actual,
            "comparator": "<=",
            "threshold": 2.0,
            "passed": passed,
        }
        for name in oracle.REQUIRED_RUNTIME_GATES
    }
    return items | {
        "runtime": {
            "overall_pass": passed,
            "gate_names": list(oracle.REQUIRED_RUNTIME_GATES),
        },
        "overall_pass": passed,
    }


def test_runtime_gate_failure_is_valid_evidence_not_an_oracle_abort():
    gates = _runtime_gates(False)
    validated = _validate_runtime_gates({"gates": gates}, overall_pass=False)
    assert validated["runtime"]["overall_pass"] is False
    assert validated["overall_pass"] is False


def test_runtime_gate_receipt_fails_closed_on_inconsistent_boolean():
    gates = _runtime_gates(True)
    gates["provider_runtime_p95_ms"]["passed"] = False
    with pytest.raises(F2OracleError, match="inconsistent"):
        _validate_runtime_gates({"gates": gates}, overall_pass=False)


def test_final_merge_schema_contract_matches_oracle_loader():
    scenes = [f"scene{index:04d}_00" for index in range(100)]
    gates = _runtime_gates(True)
    receipt = {
        "schema": oracle.F2_RECEIPT_SCHEMA,
        "protocol_id": oracle.F2_PROTOCOL_ID,
        "complete": True,
        "overall_pass": True,
        "run_signature_sha256": "d" * 64,
        "coverage": {"scene_count": 100, "scene_order": scenes},
        "totals": {
            "scene_count": 100,
            "keyframe_count": oracle.EXPECTED["keyframe_count"],
            "successful_frame_count": oracle.EXPECTED["successful_frame_count"],
            "source_count": oracle.EXPECTED["candidate_count"],
            "identity_verified_source_count": oracle.EXPECTED["candidate_count"],
        },
        "contracts": {
            "shadow_only": True,
            "birth_enabled": False,
            "ground_truth_access": False,
            "prediction_access": False,
            "evaluator_access": False,
            "native_output_mutation": False,
            "training": False,
            "f0_exact_replay_required": True,
        },
        "gates": gates,
        "scenes": [
            {
                "scene_id": scene,
                "scene_index": index,
                "sidecar": {"path": f"{scene}.json", "sha256": "a" * 64},
                "evidence_npz": {"path": f"{scene}.npz", "sha256": "b" * 64},
            }
            for index, scene in enumerate(scenes)
        ],
    }
    rows, signature, overall, validated_gates = _validate_f2_receipt(receipt, scenes)
    assert list(rows) == scenes
    assert signature == "d" * 64
    assert overall is True
    assert validated_gates["runtime"]["overall_pass"] is True


def test_oracle_validator_accepts_actual_merge_produced_receipt(tmp_path: Path):
    manifest, _f0_receipt, _calls = _prepare_f2(tmp_path)
    receipt = f2_merger.merge_f2(
        shard_paths=(tmp_path / "f2/shards/shard-000-of-001.json",),
        scene_list_path=Path(manifest["scene_list"]["path"]),
        output_dir=tmp_path / "f2/final",
        _expected_scene_count=1,
    )
    scene = receipt["coverage"]["scene_order"][0]
    rows, _signature, overall, gates = _validate_f2_receipt(
        receipt,
        [scene],
        expected_scene_count=1,
        expected_keyframe_count=2,
        expected_successful_frame_count=2,
        expected_source_count=2,
    )
    assert list(rows) == [scene]
    assert overall is True
    assert gates["runtime"]["overall_pass"] is True


def test_grouped_matrix_rejects_wrong_order_and_shape():
    with pytest.raises(F2OracleError, match="order"):
        grouped_iou_matrix(
            {
                "HL": np.zeros((1, 1)),
                "H0": np.zeros((1, 1)),
                "HLG": np.zeros((1, 1)),
            }
        )
    with pytest.raises(F2OracleError, match="identical"):
        grouped_iou_matrix(
            {
                "H0": np.zeros((1, 1)),
                "HL": np.zeros((2, 1)),
                "HLG": np.zeros((1, 1)),
            }
        )


def _write_minimal_f0_f2_sidecars(tmp_path: Path):
    digest_a = "a" * 64
    digest_b = "b" * 64
    f0_candidate = {
        "rank": 0,
        "raw_index": 3,
        "mask_sha256": digest_a,
        "points_and_voxel_keys_sha256": digest_b,
        "world_q02": [0.0, 0.0, 0.0],
        "world_q98": [1.0, 1.0, 1.0],
    }
    f0 = {
        "schema": oracle.F0_SCENE_SCHEMA,
        "scene_id": "scene0001_00",
        "frames": [
            {
                "frame_id": 0,
                "frame_ordinal": 0,
                "successful": True,
                "funnel": {"candidates": [f0_candidate]},
            }
        ],
    }
    f0_path = tmp_path / "f0.json"
    f0_path.write_text(json.dumps(f0, sort_keys=True), encoding="utf-8")

    hypothesis = {
        "valid": True,
        "q02": [0.0, 0.0, 0.0],
        "q98": [1.0, 1.0, 1.0],
        "center": [0.5, 0.5, 0.5],
        "extent": [1.0, 1.0, 1.0],
        "diagnostics": {"applied": True},
    }
    signature = "c" * 64
    f2 = {
        "schema": oracle.F2_SCENE_SCHEMA,
        "protocol_id": oracle.F2_PROTOCOL_ID,
        "complete": True,
        "scene_id": "scene0001_00",
        "scene_index": 0,
        "run_signature_sha256": signature,
        "frames": [
            {
                "frame_id": 0,
                "frame_ordinal": 0,
                "successful": True,
                "sources": [
                    {
                        "source_id": "scene0001_00/frame_000000/raw_003",
                        "candidate_index": 0,
                        "rank": 0,
                        "raw_index": 3,
                        "mask_sha256": digest_a,
                        "points_and_voxel_keys_sha256": digest_b,
                        "f0_world_q02": [0.0, 0.0, 0.0],
                        "f0_world_q98": [1.0, 1.0, 1.0],
                        "hypotheses": {
                            "H0": dict(hypothesis),
                            "HL": dict(hypothesis),
                            "HLG": dict(hypothesis),
                        },
                    }
                ],
            }
        ],
    }
    f2_path = tmp_path / "f2.json"
    f2_path.write_text(json.dumps(f2, sort_keys=True), encoding="utf-8")
    sidecar_sha = hashlib.sha256(f2_path.read_bytes()).hexdigest()
    return f0_path, f2_path, sidecar_sha, signature, f2


def test_f2_loader_verifies_source_identity_and_h0_bitwise(tmp_path: Path):
    f0_path, f2_path, sidecar_sha, signature, _ = _write_minimal_f0_f2_sidecars(tmp_path)
    sources, keyframes, successful = _load_f2_sources(
        path=f2_path,
        f0_path=f0_path,
        scene="scene0001_00",
        scene_index=0,
        alignment=np.eye(4),
        receipt_sidecar_sha256=sidecar_sha,
        run_signature_sha256=signature,
    )
    assert (keyframes, successful, len(sources)) == (1, 1, 1)
    assert sources[0].source_id == "scene0001_00/frame_000000/raw_003"
    np.testing.assert_array_equal(sources[0].world_minmax["H0"], [0, 0, 0, 1, 1, 1])


def test_f2_loader_rejects_h0_geometry_drift(tmp_path: Path):
    f0_path, f2_path, _, signature, f2 = _write_minimal_f0_f2_sidecars(tmp_path)
    f2["frames"][0]["sources"][0]["hypotheses"]["H0"].update(
        {
            "q02": [0.1, 0.0, 0.0],
            "center": [0.55, 0.5, 0.5],
            "extent": [0.9, 1.0, 1.0],
        }
    )
    f2_path.write_text(json.dumps(f2, sort_keys=True), encoding="utf-8")
    sidecar_sha = hashlib.sha256(f2_path.read_bytes()).hexdigest()
    with pytest.raises(F2OracleError, match="bitwise reproduce"):
        _load_f2_sources(
            path=f2_path,
            f0_path=f0_path,
            scene="scene0001_00",
            scene_index=0,
            alignment=np.eye(4),
            receipt_sidecar_sha256=sidecar_sha,
            run_signature_sha256=signature,
        )


def test_output_is_create_only_and_outside_inputs(tmp_path: Path):
    protected = tmp_path / "inputs"
    protected.mkdir()
    with pytest.raises(F2OracleError, match="protected"):
        _validate_output_path(protected / "report.json", [protected])
    existing = tmp_path / "existing.json"
    existing.write_text("sealed", encoding="utf-8")
    with pytest.raises(F2OracleError, match="overwrite"):
        _validate_output_path(existing, [protected])
    with pytest.raises(F2OracleError, match="suffix"):
        _validate_output_path(tmp_path / "report.txt", [protected])


def test_main_writes_once_and_refuses_overwrite(tmp_path, monkeypatch):
    out = tmp_path / "reports" / "f2.json"
    fake_report = {
        "schema": oracle.SCHEMA,
        "totals": {"scene_count": 100},
        "decision": {"overall_pass": False},
    }
    monkeypatch.setattr(
        oracle, "audit_scannet_fastsam_f2_paper100_oracle", lambda **_: fake_report
    )
    assert oracle.main(["--out", str(out)]) == 0
    assert json.loads(out.read_text(encoding="utf-8")) == fake_report
    before = out.read_bytes()
    with pytest.raises(F2OracleError, match="overwrite"):
        oracle.main(["--out", str(out)])
    assert out.read_bytes() == before
