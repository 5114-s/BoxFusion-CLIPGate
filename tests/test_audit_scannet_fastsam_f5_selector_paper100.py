from __future__ import annotations

import copy
import json
import math
from pathlib import Path

import numpy as np
import pytest

import tools.audit_scannet_fastsam_f5_selector_paper100 as audit
from tools.audit_scannet_boxer_unexplained_oracle import official_constant_evaluate


@pytest.fixture(scope="module")
def official_eval_det():
    return audit._load_official_eval_det(
        audit._official_dependency_paths(audit.DEFAULT_OFFICIAL_EVALUATOR)
    )


def test_default_f5_receipt_matches_the_create_only_merge_name() -> None:
    assert audit.DEFAULT_F5_RECEIPT.name == "F5_GT_FREE_SELECTOR_PAPER100.json"


def _aabb(lower=(0.0, 0.0, 0.0), upper=(1.0, 1.0, 1.0)):
    lo = np.asarray(lower, dtype=float)
    hi = np.asarray(upper, dtype=float)
    return {
        "valid": True,
        "q02": lo.tolist(),
        "q98": hi.tolist(),
        "center": ((lo + hi) / 2.0).tolist(),
        "extent": (hi - lo).tolist(),
    }


def _hb(theta=math.pi / 4.0):
    rotation = np.asarray(
        [[math.cos(theta), -math.sin(theta), 0.0],
         [math.sin(theta), math.cos(theta), 0.0], [0.0, 0.0, 1.0]],
        dtype=float,
    )
    center = np.asarray([2.0, 3.0, 4.0])
    extent = np.asarray([4.0, 1.0, 2.0])
    corners = center + (audit._SIGNS * (extent / 2.0)) @ rotation.T
    return {
        "valid": True,
        "world_center": center.tolist(),
        "local_extent": extent.tolist(),
        "world_rotation": rotation.tolist(),
        "world_corners": corners.tolist(),
    }


def _source(selected="H0"):
    hypotheses = {"H0": _aabb(), "HL": _aabb(), "HLG": _aabb(), "HB": _hb()}
    f4 = {
        "source_id": "source-0",
        "source_lineage_sha256": "1" * 64,
        "scene_index": 0,
        "frame_id": 25,
        "frame_ordinal": 0,
        "rank": 0,
        "candidate_index": 0,
        "hypotheses": hypotheses,
    }
    geometry, _ = audit._expected_selected_geometry(selected, hypotheses[selected])
    f5 = {
        "source_id": "source-0",
        "source_lineage_sha256": "1" * 64,
        "frame_id": 25,
        "frame_ordinal": 0,
        "rank": 0,
        "input_hypothesis_sha256": {
            name: audit._canonical_json_sha256(row) for name, row in hypotheses.items()
        },
        "selected_hypothesis": selected,
        "selected_geometry": geometry,
        "selected_geometry_sha256": audit._canonical_json_sha256(geometry),
        "formal_score": 1.0,
    }
    f5["result_sha256"] = audit._canonical_json_sha256(f5)
    return f4, f5


def _selected(name: str, index: int) -> audit.SelectedSource:
    return audit.SelectedSource(
        scene_id="scene0000_00", scene_index=0, frame_id=index,
        frame_ordinal=index, rank=0, source_id=f"s{index}",
        source_lineage_sha256=f"{index + 1:064x}",
        result_sha256=f"{index + 100:064x}",
        selected_hypothesis=name, world_corners=np.zeros((8, 3)),
    )


def test_selected_geometry_is_exact_copy_and_row_hash_is_sealed() -> None:
    f4, f5 = _source("H0")
    result = audit.validate_selected_source(
        scene="scene0000_00", scene_index=0, frame_id=25,
        frame_ordinal=0, rank=0, f4_source=f4, f5_source=f5,
    )
    assert result.selected_hypothesis == "H0"
    changed = copy.deepcopy(f5)
    changed["selected_geometry"]["q98"][0] += 0.01
    changed["result_sha256"] = audit._canonical_json_sha256(
        {key: value for key, value in changed.items() if key != "result_sha256"}
    )
    with pytest.raises(audit.F5EvaluationError, match="exact F4 copy"):
        audit.validate_selected_source(
            scene="scene0000_00", scene_index=0, frame_id=25,
            frame_ordinal=0, rank=0, f4_source=f4, f5_source=changed,
        )


def test_selected_geometry_hash_rejects_bool_numeric_alias() -> None:
    f4, f5 = _source("H0")
    f5["selected_geometry"]["q02"][0] = False
    f5["result_sha256"] = audit._canonical_json_sha256(
        {key: value for key, value in f5.items() if key != "result_sha256"}
    )
    with pytest.raises(audit.F5EvaluationError, match="geometry hash"):
        audit.validate_selected_source(
            scene="scene0000_00", scene_index=0, frame_id=25,
            frame_ordinal=0, rank=0, f4_source=f4, f5_source=f5,
        )


def test_formal_score_rejects_json_boolean_true() -> None:
    f4, f5 = _source("H0")
    f5["formal_score"] = True
    f5["result_sha256"] = audit._canonical_json_sha256(
        {key: value for key, value in f5.items() if key != "result_sha256"}
    )
    with pytest.raises(audit.F5EvaluationError, match="formal score"):
        audit.validate_selected_source(
            scene="scene0000_00", scene_index=0, frame_id=25,
            frame_ordinal=0, rank=0, f4_source=f4, f5_source=f5,
        )


def test_hb_axis_alignment_transforms_obb_corners_before_minmax() -> None:
    f4, f5 = _source("HB")
    result = audit.validate_selected_source(
        scene="scene0000_00", scene_index=0, frame_id=25,
        frame_ordinal=0, rank=0, f4_source=f4, f5_source=f5,
    )
    alignment = np.eye(4)
    alignment[:3, :3] = np.asarray(f4["hypotheses"]["HB"]["world_rotation"]).T
    correct = audit._align_corners(result.world_corners, alignment, "HB")
    lo = result.world_corners.min(axis=0)
    hi = result.world_corners.max(axis=0)
    naive = audit._align_corners(audit._aabb_corners(lo, hi, "envelope"), alignment, "naive")
    np.testing.assert_allclose(correct[3:] - correct[:3], [4.0, 1.0, 2.0])
    assert not np.allclose(correct, naive)


def test_threshold_report_capacity_split_hb_retention_and_suffix(official_eval_det) -> None:
    scenes = ["scene0000_00"]
    native = [np.asarray([[0.9, 0.0, 0.0, 0.0]])]
    selected = [np.asarray([
        [0.0, 0.9, 0.0, 0.0],
        [0.0, 0.0, 0.9, 0.0],
        [0.0, 0.0, 0.0, 0.9],
    ])]
    sources = [[_selected("H0", 0), _selected("HB", 1), _selected("HLG", 2)]]
    baseline = official_constant_evaluate(native, [4], 0.5)
    report = audit.evaluate_selected_threshold(
        scenes=scenes, native_iou=native, selected_iou=selected,
        selected_sources=sources, gt_counts=[4], baseline_evaluation=baseline,
        threshold=0.5, f4_g4_additional_union_matches=4,
        official_eval_det=official_eval_det,
    )
    assert report["additional_union_matching_over_native"] == 3
    assert report["selected_hypothesis_split"]["H0"]["additional_union_matching_over_native"] == 1
    assert report["selected_hypothesis_split"]["HB"]["additional_union_matching_over_native"] == 1
    assert report["hb_selected_matching"]["matched_native_unmatched_gt_count"] == 1
    assert report["f4_g4_capacity"]["retained_fraction"] == pytest.approx(0.75)
    suffix = report["gt_selected_constructive_suffix"]
    assert suffix["selected_source_count"] == 3
    assert suffix["formal_score"] == 1.0
    assert suffix["official_evaluation"]["greedy_tp"] == 4
    json.dumps(report, allow_nan=False)


def test_authenticated_official_eval_det_matches_constant_reference(official_eval_det) -> None:
    assert Path(official_eval_det.eval_det_cls.__code__.co_filename).resolve() == (
        audit.DEFAULT_OFFICIAL_EVALUATOR.parent / "utils/eval_det.py"
    ).resolve()
    matrices = [np.asarray([[0.9, 0.0], [0.0, 0.8], [0.0, 0.0]])]
    result = audit._authenticated_official_constant_evaluate(
        matrices, [2], 0.5, official_eval_det
    )
    reference = official_constant_evaluate(matrices, [2], 0.5)
    assert result["authenticated_official_eval_det"] is True
    assert result["ap_points"] == pytest.approx(reference["ap_points"], abs=1e-12)
    assert result["official_eval_det_ap_points"] == pytest.approx(
        reference["ap_points"], abs=1e-12
    )


def test_evaluation_snapshot_reuses_prevalidated_selector_seals(tmp_path: Path) -> None:
    scene = "scene0000_00"
    selector_snapshot = {
        "fixed": {"f5_receipt": {"path": "/sealed/f5.json", "sha256": "1" * 64}},
        "scenes": {
            scene: {
                "f4_sidecar": {"path": "/sealed/f4-scene.json", "sha256": "2" * 64},
                "f5_sidecar": {"path": "/sealed/f5-scene.json", "sha256": "3" * 64},
            }
        },
    }
    evaluation_fixed = tmp_path / "f4-report.json"
    evaluation_fixed.write_text("{}", encoding="ascii")
    baseline = tmp_path / "baseline"
    gt = tmp_path / "gt"
    scans = tmp_path / "scans" / scene
    baseline.mkdir()
    gt.mkdir()
    scans.mkdir(parents=True)
    (baseline / f"{scene}_boxes.pkl").write_bytes(b"native")
    (gt / f"{scene}_bbox.npy").write_bytes(b"gt")
    (scans / f"{scene}.txt").write_text("alignment", encoding="ascii")
    result = audit._evaluation_snapshot(
        scenes=[scene], fixed={"f4_report": evaluation_fixed},
        selector_snapshot=selector_snapshot, baseline_root=baseline,
        gt_root=gt, scan_root=tmp_path / "scans",
    )
    assert result["fixed"]["f5_receipt"] == selector_snapshot["fixed"]["f5_receipt"]
    assert result["scenes"][scene]["f4_sidecar"] == selector_snapshot["scenes"][scene]["f4_sidecar"]
    assert result["scenes"][scene]["f5_sidecar"] == selector_snapshot["scenes"][scene]["f5_sidecar"]


def _historical_lineage_fixture():
    scenes = ["s0", "s1"]
    snapshot = {
        "fixed": {
            "scene_list": {"sha256": "1" * 64},
            "official_evaluator": {"sha256": "2" * 64},
            "f4_receipt": {"sha256": "3" * 64},
        },
        "scenes": {
            scene: {
                "f4_sidecar": {"sha256": f"{10 + index:064x}"},
                "native": {"sha256": f"{20 + index:064x}"},
                "gt": {"sha256": f"{30 + index:064x}"},
                "alignment": {"sha256": f"{40 + index:064x}"},
            }
            for index, scene in enumerate(scenes)
        },
    }
    fixed_files = {
        "scene_list": {"path": "scene.txt", "sha256": "1" * 64},
        "official_evaluator": {"path": "eval.py", "sha256": "2" * 64},
        "f4_receipt": {"path": "f4.json", "sha256": "3" * 64},
    }
    ledger_names = {
        "f4_sidecars": "f4_sidecar",
        "native_predictions": "native",
        "ground_truth": "gt",
        "axis_alignment": "alignment",
    }
    ledgers = {}
    for historical, current in ledger_names.items():
        entries = [[scene, snapshot["scenes"][scene][current]["sha256"]] for scene in scenes]
        ledgers[historical] = {
            "entries": entries,
            "sha256": audit._canonical_json_sha256(entries),
        }
    seal = {"fixed_files": fixed_files, "ordered_scene_ledgers": ledgers}
    report = {"input_sha256_before": seal, "input_sha256_after": copy.deepcopy(seal)}
    return scenes, snapshot, report


def test_historical_f4_lineage_crosscheck_rejects_replaced_gt() -> None:
    scenes, snapshot, report = _historical_lineage_fixture()
    audit._validate_historical_f4_input_lineage(report, snapshot=snapshot, scenes=scenes)
    snapshot["scenes"]["s1"]["gt"]["sha256"] = "f" * 64
    with pytest.raises(audit.F5EvaluationError, match="historical F4 ledger"):
        audit._validate_historical_f4_input_lineage(report, snapshot=snapshot, scenes=scenes)


def _synthetic_f5_receipt(scenes):
    gates = {
        name: {"pass": True, "passed": True}
        for name in audit.F5_GATE_NAMES
    }
    receipt = {
        "schema": audit.F5_MERGE_SCHEMA,
        "protocol_id": audit.F5_PROTOCOL_ID,
        "protocol_sha256": audit.F5_PROTOCOL_SHA256,
        "complete": True,
        "overall_pass": True,
        "decision": "retain_f5_for_one_separately_sealed_evaluation_only",
        "native_output_mutation_count": 0,
        "source_addition_or_removal_count": 0,
        "score_rank_semantic_mutation_count": 0,
        "forbidden_access_count": 0,
        "training_or_online_learning_count": 0,
        "birth_count": 0,
        "coverage": {
            "scene_count": 100,
            "scene_order": list(scenes),
            "exact_source_partition": True,
            "exact_source_order": True,
            "source_ids_sha256": audit.EXPECTED_SOURCE_IDS_SHA256,
            "source_lineage_sha256": audit.EXPECTED_SOURCE_LINEAGE_SHA256,
            "result_ledger_sha256": audit.EXPECTED_RESULT_LEDGER_SHA256,
        },
        "totals": {
            "keyframe_count": audit.EXPECTED["keyframe_count"],
            "successful_frame_count": audit.EXPECTED["successful_frame_count"],
            "source_count": audit.EXPECTED["source_count"],
            "identity_verified_source_count": audit.EXPECTED["source_count"],
        },
        "gates": gates,
        "causality": {"overall_pass": True},
        "determinism": {"overall_pass": True},
        "runtime": {"overall_pass": True},
        "evaluation_authorization": {
            "allowed": True, "birth_authorized": False, "deployment_authorized": False,
        },
        "selection": {
            "formal_score": 1.0,
            "selected_h0_count": 0,
            "selected_hl_count": 0,
            "selected_hlg_count": audit.EXPECTED["source_count"],
            "selected_hb_count": 0,
        },
        "scenes": [
            {
                "scene_id": scene,
                "scene_index": index,
                "sidecar": {"path": f"/{scene}.json", "sha256": "a" * 64},
            }
            for index, scene in enumerate(scenes)
        ],
    }
    receipt["content_sha256"] = audit._canonical_json_sha256(receipt)
    return receipt


def test_merge_validation_rejects_fabricated_gate_ledger() -> None:
    scenes = [f"scene{index:04d}_00" for index in range(100)]
    receipt = _synthetic_f5_receipt(scenes)
    assert len(audit._validate_merge(
        receipt, schema=audit.F5_MERGE_SCHEMA, protocol_id=audit.F5_PROTOCOL_ID,
        scenes=scenes, label="F5",
    )) == 100
    receipt["gates"] = {"fabricated_only_gate": {"pass": True, "passed": True}}
    receipt["content_sha256"] = audit._content_hash(receipt)
    with pytest.raises(audit.F5EvaluationError, match="frozen gate ledger"):
        audit._validate_merge(
            receipt, schema=audit.F5_MERGE_SCHEMA, protocol_id=audit.F5_PROTOCOL_ID,
            scenes=scenes, label="F5",
        )


def test_capacity_cannot_exceed_sealed_f4_g4(official_eval_det) -> None:
    native = [np.empty((0, 1))]
    selected = [np.asarray([[0.9]])]
    baseline = official_constant_evaluate(native, [1], 0.5)
    with pytest.raises(audit.F5EvaluationError, match="exceeds"):
        audit.evaluate_selected_threshold(
            scenes=["s"], native_iou=native, selected_iou=selected,
            selected_sources=[[_selected("H0", 0)]], gt_counts=[1],
            baseline_evaluation=baseline, threshold=0.5,
            f4_g4_additional_union_matches=0,
            official_eval_det=official_eval_det,
        )


def test_decision_requires_144_matches_and_plus10_at_every_threshold() -> None:
    rows = {
        f"{threshold:.2f}": {
            "additional_union_matching_over_native": 144,
            "gt_selected_constructive_suffix": {"delta_ap_points": 10.0},
        }
        for threshold in audit.THRESHOLDS
    }
    assert audit.f5_decision(
        per_threshold=rows, no_gt_merge_passed=True, baseline_passed=True
    )["overall_pass"] is True
    rows["0.50"]["additional_union_matching_over_native"] = 143
    assert audit.f5_decision(
        per_threshold=rows, no_gt_merge_passed=True, baseline_passed=True
    )["overall_pass"] is False


def test_main_is_create_only(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    report = {"decision": {"overall_pass": False}, "value": 1}
    monkeypatch.setattr(audit, "audit_scannet_fastsam_f5_selector_paper100", lambda **_: report)
    out = tmp_path / "report.json"
    assert audit.main(["--out", str(out)]) == 0
    assert json.loads(out.read_text()) == report
    with pytest.raises(audit.F5EvaluationError, match="overwrite"):
        audit.main(["--out", str(out)])
