from __future__ import annotations

import copy
import math
from pathlib import Path

import numpy as np
import pytest

import tools.audit_scannet_fastsam_f6_mvdc_paper100 as audit
from tools.audit_scannet_boxer_unexplained_oracle import official_constant_evaluate


@pytest.fixture(scope="module")
def official_eval_det():
    return audit._load_official_eval_det(
        audit._official_dependency_paths(audit.DEFAULT_OFFICIAL_EVALUATOR)
    )


def _aabb(lower=(0.0, 0.0, 0.0), upper=(1.0, 1.0, 1.0)):
    lo = np.asarray(lower, dtype=float)
    hi = np.asarray(upper, dtype=float)
    return {
        "valid": True,
        "q02": lo.tolist(),
        "q98": hi.tolist(),
        "center": ((lo + hi) * 0.5).tolist(),
        "extent": (hi - lo).tolist(),
    }


def _hb(theta=math.pi / 4.0):
    rotation = np.asarray(
        [
            [math.cos(theta), -math.sin(theta), 0.0],
            [math.sin(theta), math.cos(theta), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    center = np.asarray([2.0, 3.0, 4.0])
    extent = np.asarray([4.0, 1.0, 2.0])
    corners = center + (audit._SIGNS * (extent * 0.5)) @ rotation.T
    return {
        "valid": True,
        "world_center": center.tolist(),
        "local_extent": extent.tolist(),
        "world_rotation": rotation.tolist(),
        "world_corners": corners.tolist(),
        "camera_depth": 2.0,
    }


def _source(selected="H0", base="H0"):
    hypotheses = {
        "H0": _aabb(),
        "HL": _aabb((0.1, 0.1, 0.1), (0.9, 0.9, 0.9)),
        "HLG": _aabb((0.05, 0.05, 0.05), (0.95, 0.95, 0.95)),
        "HB": _hb(),
    }
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
    selected_geometry, _ = audit._expected_selected_geometry(selected, hypotheses[selected])
    base_geometry, _ = audit._expected_selected_geometry(base, hypotheses[base])
    switched = selected != base
    f6 = {
        "schema": audit.F6_SOURCE_SCHEMA,
        "protocol_id": audit.F6_PROTOCOL_ID,
        "mode": "shadow",
        "source_id": "source-0",
        "source_lineage_sha256": "1" * 64,
        "input_evidence_sha256": "2" * 64,
        "frame_id": 25,
        "frame_ordinal": 0,
        "rank": 0,
        "input_hypothesis_sha256": {
            name: audit._canonical_json_sha256(row) for name, row in hypotheses.items()
        },
        "base_hypothesis": base,
        "base_geometry": base_geometry,
        "base_geometry_sha256": audit._canonical_json_sha256(base_geometry),
        "matched_past_frame_count": 2 if switched else 0,
        "selected_hypothesis": selected,
        "selected_geometry": selected_geometry,
        "selected_geometry_sha256": audit._canonical_json_sha256(selected_geometry),
        "switched_from_base": switched,
        "selection_reason": "non_base_candidate_won" if switched else "fewer_than_two_past_matches",
        "formal_score": 1.0,
        "maximum_lookahead_frames": 0,
        "observer_only": True,
        "birth_applied": False,
        "native_output_mutation_applied": False,
    }
    f6["result_sha256"] = audit._canonical_json_sha256(f6)
    return f4, f6


def _selected(name: str, index: int, *, switched: bool) -> audit.SelectedSource:
    return audit.SelectedSource(
        scene_id="scene0000_00",
        scene_index=0,
        frame_id=index,
        frame_ordinal=index,
        rank=0,
        source_id=f"s{index}",
        source_lineage_sha256=f"{index + 1:064x}",
        result_sha256=f"{index + 100:064x}",
        selected_hypothesis=name,
        base_hypothesis="H0",
        switched_from_base=switched,
        world_corners=np.zeros((8, 3)),
        base_world_corners=np.zeros((8, 3)),
    )


def test_default_f6_receipt_and_frozen_hashes() -> None:
    assert audit.DEFAULT_F6_RECEIPT.name == "F6_GT_FREE_MVDC_PAPER100.json"
    assert audit._sha256(audit.DEFAULT_F6_RECEIPT) == audit.EXPECTED_F6_RECEIPT_SHA256
    assert audit._sha256(audit.DEFAULT_EVALUATION_PROTOCOL) == audit.EVALUATION_PROTOCOL_SHA256
    assert audit._sha256(audit.SHARED_EVALUATOR_SOURCE) == audit.EXPECTED_SHARED_EVALUATOR_SHA256


def test_selected_and_base_geometries_are_exact_f4_copies() -> None:
    f4, f6 = _source("HL", "H0")
    result = audit.validate_selected_source(
        scene="scene0000_00",
        scene_index=0,
        frame_id=25,
        frame_ordinal=0,
        rank=0,
        f4_source=f4,
        f6_source=f6,
    )
    assert result.selected_hypothesis == "HL"
    assert result.base_hypothesis == "H0"
    assert result.switched_from_base is True
    changed = copy.deepcopy(f6)
    changed["selected_geometry"]["q98"][0] += 0.01
    changed["result_sha256"] = audit._canonical_json_sha256(
        {key: value for key, value in changed.items() if key != "result_sha256"}
    )
    with pytest.raises(audit.F6EvaluationError, match="exact F4 copy"):
        audit.validate_selected_source(
            scene="scene0000_00",
            scene_index=0,
            frame_id=25,
            frame_ordinal=0,
            rank=0,
            f4_source=f4,
            f6_source=changed,
        )


def test_switch_ledger_and_formal_score_fail_closed() -> None:
    f4, f6 = _source("HL", "H0")
    changed = copy.deepcopy(f6)
    changed["switched_from_base"] = False
    changed["result_sha256"] = audit._canonical_json_sha256(
        {key: value for key, value in changed.items() if key != "result_sha256"}
    )
    with pytest.raises(audit.F6EvaluationError, match="base/switch ledger"):
        audit.validate_selected_source(
            scene="scene0000_00", scene_index=0, frame_id=25, frame_ordinal=0,
            rank=0, f4_source=f4, f6_source=changed,
        )
    changed = copy.deepcopy(f6)
    changed["formal_score"] = True
    changed["result_sha256"] = audit._canonical_json_sha256(
        {key: value for key, value in changed.items() if key != "result_sha256"}
    )
    with pytest.raises(audit.F6EvaluationError, match="formal score"):
        audit.validate_selected_source(
            scene="scene0000_00", scene_index=0, frame_id=25, frame_ordinal=0,
            rank=0, f4_source=f4, f6_source=changed,
        )


def test_hb_uses_true_corners_before_axis_aligned_envelope() -> None:
    f4, f6 = _source("HB", "H0")
    result = audit.validate_selected_source(
        scene="scene0000_00", scene_index=0, frame_id=25, frame_ordinal=0,
        rank=0, f4_source=f4, f6_source=f6,
    )
    alignment = np.eye(4)
    alignment[:3, :3] = np.asarray(f4["hypotheses"]["HB"]["world_rotation"]).T
    correct = audit._align_corners(result.world_corners, alignment, "HB")
    lower = result.world_corners.min(axis=0)
    upper = result.world_corners.max(axis=0)
    envelope_corners = (lower + upper)[None, :] * 0.5 + audit._SIGNS * ((upper - lower)[None, :] * 0.5)
    naive = audit._align_corners(envelope_corners, alignment, "naive")
    np.testing.assert_allclose(correct[3:] - correct[:3], [4.0, 1.0, 2.0])
    assert not np.allclose(correct, naive)


def _synthetic_f6_receipt(scenes):
    selected_counts = {"h0": 1000, "hl": 1000, "hlg": 50134, "hb": 165}
    totals = {
        "keyframe_count": audit.EXPECTED["keyframe_count"],
        "successful_frame_count": audit.EXPECTED["successful_frame_count"],
        "source_count": audit.EXPECTED["source_count"],
        "identity_verified_source_count": audit.EXPECTED["source_count"],
        "multiview_evaluated_source_count": 1000,
        "switch_count": 165,
        "fallback_count": audit.EXPECTED["source_count"] - 165,
        **{f"selected_{name}_count": count for name, count in selected_counts.items()},
    }
    rows = [
        {
            "scene_id": scene,
            "scene_index": index,
            "sidecar": {"path": f"/{scene}.json", "sha256": f"{index + 1:064x}"},
            "causality": {"overall_pass": True, "maximum_lookahead_frames": 0},
            "determinism": {"passed": True},
            "prefix_replay": {"passed": True},
            "bounded_state": {"overall_pass": True},
        }
        for index, scene in enumerate(scenes)
    ]
    receipt = {
        "schema": audit.F6_MERGE_SCHEMA,
        "protocol_id": audit.F6_PROTOCOL_ID,
        "protocol_sha256": audit.F6_PROTOCOL_SHA256,
        "complete": True,
        "overall_pass": True,
        "decision": "retain_f6_for_one_separately_sealed_evaluation_only",
        "coverage": {
            "scene_count": 100,
            "scene_order": list(scenes),
            "exact_source_partition": True,
            "exact_source_order": True,
            "source_ids_sha256": audit.EXPECTED_SOURCE_IDS_SHA256,
            "source_lineage_sha256": audit.EXPECTED_SOURCE_LINEAGE_SHA256,
            "result_ledger_sha256": audit.EXPECTED_RESULT_LEDGER_SHA256,
        },
        "totals": totals,
        "gates": {
            name: {"pass": True, "passed": True} for name in audit.F6_GATE_NAMES
        },
        "runtime": {"overall_pass": True},
        "bounded_state": {"overall_pass": True},
        "evaluation_authorization": {
            "allowed": True, "birth_authorized": False, "deployment_authorized": False,
        },
        "contracts": {
            "shadow_only": True,
            "selector_only": True,
            **{
                name: False for name in (
                    "ground_truth_access", "annotation_access", "evaluator_access",
                    "prediction_access", "future_frame_access", "native_output_mutation",
                    "source_addition_or_removal", "score_or_rank_mutation",
                    "semantic_or_clip_access", "birth_enabled", "training", "online_learning",
                )
            },
        },
        "selection": {
            **{f"selected_{name}_count": count for name, count in selected_counts.items()},
            "formal_score": 1.0,
            "switch_count": 165,
            "fallback_count": audit.EXPECTED["source_count"] - 165,
            "complete_three_view_switch_proof_count": 165,
        },
        "inputs": {"f4_receipt": {"sha256": audit.EXPECTED_F4_RECEIPT_SHA256}},
        "scenes": rows,
        "birth_count": 0,
        "native_output_mutation_count": 0,
        "source_addition_or_removal_count": 0,
        "score_rank_semantic_mutation_count": 0,
        "forbidden_access_count": 0,
        "training_or_online_learning_count": 0,
    }
    receipt["content_sha256"] = audit._content_hash(receipt)
    return receipt


def test_no_gt_merge_requires_every_frozen_gate() -> None:
    scenes = [f"scene{index:04d}_00" for index in range(100)]
    receipt = _synthetic_f6_receipt(scenes)
    assert len(audit._validate_f6_merge(receipt, scenes=scenes)) == 100
    changed = copy.deepcopy(receipt)
    changed["gates"]["switch_min_sources"]["pass"] = False
    changed["content_sha256"] = audit._content_hash(changed)
    with pytest.raises(audit.F6EvaluationError, match="gate ledger"):
        audit._validate_f6_merge(changed, scenes=scenes)


def test_threshold_report_has_hypothesis_base_switch_and_all_base_counterfactual(
    official_eval_det,
) -> None:
    scenes = ["scene0000_00"]
    native = [np.asarray([[0.9, 0.0, 0.0, 0.0]])]
    selected = [
        np.asarray(
            [
                [0.0, 0.9, 0.0, 0.0],
                [0.0, 0.0, 0.9, 0.0],
                [0.0, 0.0, 0.0, 0.9],
            ]
        )
    ]
    all_base = [
        np.asarray(
            [
                [0.0, 0.9, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.9],
            ]
        )
    ]
    sources = [[
        _selected("H0", 0, switched=False),
        _selected("HB", 1, switched=True),
        _selected("HLG", 2, switched=False),
    ]]
    baseline = official_constant_evaluate(native, [4], 0.5)
    report = audit.evaluate_selected_threshold(
        scenes=scenes,
        native_iou=native,
        selected_iou=selected,
        all_base_iou=all_base,
        selected_sources=sources,
        gt_counts=[4],
        baseline_evaluation=baseline,
        threshold=0.5,
        f4_g4_additional_union_matches=4,
        official_eval_det=official_eval_det,
    )
    assert report["additional_union_matching_over_native"] == 3
    assert report["selected_hypothesis_split"]["HB"]["additional_union_matching_over_native"] == 1
    assert report["selected_base_switch_split"]["switch"]["selected_source_count"] == 1
    assert report["all_base_counterfactual"]["switch_replacement_delta_union_matching"] == 1
    assert report["f4_g4_capacity"]["f6_retained_additional_union_matches"] == 3
    assert report["oracle_only"] is True and report["deployable"] is False
    suffix = report["gt_selected_constructive_suffix"]
    assert suffix["threshold_specific_gt_selection"] is True
    assert suffix["shared_detection_list_across_iou_thresholds"] is False


def test_f6_decision_requires_both_gates_at_every_threshold() -> None:
    per_threshold = {
        key: {
            "additional_union_matching_over_native": 144,
            "gt_selected_constructive_suffix": {"delta_ap_points": 10.0},
        }
        for key in ("0.15", "0.25", "0.50")
    }
    passed = audit.f6_decision(
        per_threshold=per_threshold, no_gt_merge_passed=True, baseline_passed=True
    )
    assert passed["overall_pass"] is True
    assert passed["active_birth_authorized"] is False
    assert passed["result"] == "retain_f6_authorize_f7_high_precision_birth_shadow_only"
    per_threshold["0.50"]["additional_union_matching_over_native"] = 143
    failed = audit.f6_decision(
        per_threshold=per_threshold, no_gt_merge_passed=True, baseline_passed=True
    )
    assert failed["overall_pass"] is False
    assert failed["result"] == "discard_f6_multiview_selector_for_plus10_route"
