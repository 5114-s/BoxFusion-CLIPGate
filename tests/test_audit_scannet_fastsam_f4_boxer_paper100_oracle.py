from __future__ import annotations

import copy
import importlib.util
import json
import math
from pathlib import Path
import sys

import numpy as np
import pytest

import tools.audit_scannet_fastsam_f4_boxer_paper100_oracle as oracle
from tools.audit_scannet_boxer_unexplained_oracle import official_constant_evaluate


def _source(source_id: str, index: int) -> oracle.F4Source:
    box = np.asarray([0.0, 0.0, 0.0, 1.0, 1.0, 1.0], dtype=np.float64)
    return oracle.F4Source(
        scene_id="scene0000_00",
        scene_index=0,
        frame_id=index,
        frame_ordinal=index,
        candidate_index=0,
        rank=0,
        raw_index=index,
        source_id=source_id,
        mask_sha256=f"{index:064x}",
        points_and_voxel_keys_sha256=f"{index + 1:064x}",
        tight_box_xyxy=(0.0, 0.0, 10.0, 10.0),
        world_minmax={name: box for name in oracle.HYPOTHESES},
        aligned_minmax={name: box for name in oracle.HYPOTHESES},
        valid={name: True for name in oracle.HYPOTHESES},
    )


def _hb_row(
    *, center=(1.0, 2.0, 3.0), extent=(2.0, 4.0, 6.0), rotation=None
):
    rotation = np.eye(3) if rotation is None else np.asarray(rotation, dtype=np.float64)
    center_array = np.asarray(center, dtype=np.float64)
    extent_array = np.asarray(extent, dtype=np.float64)
    corners = oracle._SIGNS * (extent_array[None, :] / 2.0)
    corners = corners @ rotation.T + center_array[None, :]
    return {
        "valid": True,
        "abstention_reason": None,
        "validity": {
            "finite_center": True,
            "finite_extent": True,
            "finite_rotation": True,
            "finite_corners": True,
            "positive_extent": True,
            "right_handed_orthonormal": True,
            "in_front": True,
            "orthogonality_error": 0.0,
            "determinant": 1.0,
            "rotation_correction_max_abs": 0.0,
            "reasons": [],
        },
        "world_center": center_array.tolist(),
        "local_extent": extent_array.tolist(),
        "world_rotation": rotation.tolist(),
        "world_corners": corners.tolist(),
        "camera_depth": 2.0,
        # Confidence is deliberately below the historical 0.5 threshold.  F4
        # must still keep valid geometry.
        "confidence": 0.01,
    }


def test_hb_axis_aligns_obb_corners_before_minmax() -> None:
    theta = math.pi / 4.0
    rotation = np.asarray(
        [
            [math.cos(theta), -math.sin(theta), 0.0],
            [math.sin(theta), math.cos(theta), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    row = _hb_row(extent=(4.0, 1.0, 2.0), rotation=rotation)
    alignment = np.eye(4)
    alignment[:3, :3] = rotation.T

    valid, world, aligned = oracle.validate_boxer_hypothesis(
        row, alignment, "source.HB"
    )

    assert valid is True
    assert world is not None and aligned is not None
    np.testing.assert_allclose(aligned[3:] - aligned[:3], [4.0, 1.0, 2.0])
    # Expanding to a world AABB first and then rotating that AABB is wider.
    naive = oracle._align_world_aabb(world, alignment, "naive")
    assert not np.allclose(naive, aligned)


def test_hb_confidence_does_not_filter_and_corner_identity_is_strict() -> None:
    row = _hb_row()
    valid, _, _ = oracle.validate_boxer_hypothesis(row, np.eye(4), "HB")
    assert valid is True

    forged = copy.deepcopy(row)
    forged["world_corners"][0][0] += 0.1
    with pytest.raises(oracle.F4OracleError, match="world_corners disagree"):
        oracle.validate_boxer_hypothesis(forged, np.eye(4), "HB")


def test_invalid_hb_abstains_without_exposing_a_geometry() -> None:
    row = {
        "valid": False,
        "abstention_reason": "nonfinite_center",
        "validity": {
            "finite_center": False,
            "finite_extent": True,
            "finite_rotation": True,
            "finite_corners": True,
            "positive_extent": True,
            "right_handed_orthonormal": True,
            "in_front": True,
            "orthogonality_error": 0.0,
            "determinant": 1.0,
            "rotation_correction_max_abs": 0.0,
            "reasons": ["nonfinite_center"],
        },
        "confidence": 0.9,
    }
    assert oracle.validate_boxer_hypothesis(row, np.eye(4), "HB") == (
        False,
        None,
        None,
    )


def test_grouped_matrix_has_one_row_per_source_and_strict_edges() -> None:
    matrices = {
        "H0": np.asarray([[0.5, 0.2], [0.0, 0.0]]),
        "HL": np.asarray([[0.5, 0.1], [0.0, 0.0]]),
        "HLG": np.asarray([[0.4, 0.3], [0.0, 0.0]]),
        "HB": np.asarray([[0.8, 0.9], [0.0, 0.7]]),
    }
    grouped = oracle.grouped_iou_matrix(matrices, oracle.HYPOTHESES)
    assert grouped.shape == (2, 2)
    np.testing.assert_allclose(grouped, [[0.8, 0.9], [0.0, 0.7]])
    # H0/HL tie retains H0.  The frozen full tie order retains H0 too.
    assert oracle.choose_hypothesis_for_edge(matrices, 0, 0, ("H0", "HL")) == (
        "H0",
        0.5,
    )


def test_evaluate_f4_uses_source_identity_and_reports_hb_gain() -> None:
    scenes = ["scene0000_00"]
    gt_counts = [3]
    native = [np.asarray([[0.9, 0.0, 0.0]], dtype=np.float64)]
    baseline = official_constant_evaluate(native, gt_counts, 0.5)
    sources = [[_source("source-0", 0), _source("source-1", 1)]]
    matrices = [
        {
            "H0": np.asarray([[0.0, 0.8, 0.0], [0.0, 0.0, 0.0]]),
            "HL": np.asarray([[0.0, 0.8, 0.0], [0.0, 0.0, 0.0]]),
            "HLG": np.asarray([[0.0, 0.8, 0.0], [0.0, 0.0, 0.0]]),
            # source-0 has two alternative GT edges but can match only one;
            # source-1 is needed for the second grouped match.
            "HB": np.asarray([[0.0, 0.8, 0.9], [0.0, 0.0, 0.7]]),
        }
    ]

    report = oracle.evaluate_f4_threshold(
        scenes=scenes,
        native_iou=native,
        hypothesis_iou=matrices,
        sources=sources,
        gt_counts=gt_counts,
        baseline_evaluation=baseline,
        threshold=0.5,
    )

    gbase = report["identity_constrained_gbase"]
    g4 = report["identity_constrained_g4"]
    assert gbase["additional_union_matching_over_native"] == 1
    assert g4["additional_union_matching_over_native"] == 2
    assert g4["candidate_maximum_matching_count"] == 2
    assert report["g4_minus_gbase"]["additional_union_matching_gain"] == 1
    assert g4["source_can_match_at_most_one_gt"] is True
    suffix = g4["gt_selected_candidate_suffix"]
    assert suffix["formal_score"] == 1.0
    assert suffix["deployable"] is False
    assert suffix["oracle_only"] is True
    assert suffix["hypothesis_tie_order"] == ["H0", "HL", "HLG", "HB"]
    selected = suffix["per_scene_selection"]["scene0000_00"]
    assert len({row["source_id"] for row in selected}) == len(selected) == 2


def _historical_block(value: int, ap: float) -> dict:
    return {
        "candidate_maximum_matching_count": value,
        "union_maximum_matching_count": value + 1,
        "native_maximum_matching_count": 1,
        "additional_union_matching_over_native": value,
        "gt_selected_candidate_suffix": {
            "selected_source_count": value,
            "chosen_hypothesis_counts": {
                "H0": value,
                "HL": 0,
                "HLG": 0,
                "HB": 0,
            },
            "official_evaluation": {"ap_points": ap},
            "delta_ap_points": ap - 1.0,
        },
    }


def test_f2_reproduction_validates_all_three_hypotheses_and_gbase(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(oracle, "THRESHOLDS", (0.5,))
    blocks = {
        "H0": _historical_block(1, 2.0),
        "HL": _historical_block(2, 3.0),
        "HLG": _historical_block(3, 4.0),
    }
    # F2 has no HB key in its chosen-count schema.
    expected_blocks = copy.deepcopy(blocks)
    for block in expected_blocks.values():
        block["gt_selected_candidate_suffix"]["chosen_hypothesis_counts"].pop("HB")
    gbase = _historical_block(4, 5.0)
    expected_gbase = copy.deepcopy(gbase)
    expected_gbase["gt_selected_candidate_suffix"]["chosen_hypothesis_counts"].pop(
        "HB"
    )
    per_threshold = {
        "0.50": {
            "hypothesis_only": blocks,
            "identity_constrained_gbase": gbase,
        }
    }
    f2 = {
        "per_threshold": {
            "0.50": {
                "hypothesis_only": expected_blocks,
                "identity_constrained_grouped": expected_gbase,
            }
        }
    }
    # The oracle normalizes absent HB counts in historical F2 below.
    checks = oracle.validate_f2_reproduction(per_threshold, f2)
    assert checks["0.50"]["passed"] is True

    forged = copy.deepcopy(per_threshold)
    forged["0.50"]["hypothesis_only"]["HLG"][
        "additional_union_matching_over_native"
    ] += 1
    with pytest.raises(oracle.F4OracleError, match="HLG failed"):
        oracle.validate_f2_reproduction(forged, f2)


def test_fixed_stopping_decision_never_authorizes_birth(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(oracle, "THRESHOLDS", (0.15, 0.25, 0.5))
    passing = {}
    for threshold in oracle.THRESHOLDS:
        passing[oracle._threshold_key(threshold)] = {
            "identity_constrained_g4": {
                "additional_union_matching_over_native": 144,
                "gt_selected_candidate_suffix": {"delta_ap_points": 10.0},
            }
        }
    decision = oracle.f4_stopping_decision(
        per_threshold=passing,
        integrity_passed=True,
        causality_passed=True,
        runtime_passed=True,
    )
    assert decision["overall_pass"] is True
    assert decision["retain_f4_for_preregistered_gt_free_selector"] is True
    assert decision["authorize_active_birth"] is False

    failing = copy.deepcopy(passing)
    failing["0.50"]["identity_constrained_g4"][
        "additional_union_matching_over_native"
    ] = 143
    decision = oracle.f4_stopping_decision(
        per_threshold=failing,
        integrity_passed=True,
        causality_passed=True,
        runtime_passed=True,
    )
    assert decision["overall_pass"] is False
    assert decision["result"] == "discard_f4_shadow"


def test_create_only_output_path_rejects_existing_and_protected(tmp_path: Path) -> None:
    # Added once the CLI helpers below are available; keeping it here guards
    # against accidentally weakening create-only behavior during integration.
    output = tmp_path / "result.json"
    oracle._validate_output_path(output, (tmp_path / "protected",))
    output.write_text("{}", encoding="utf-8")
    with pytest.raises(oracle.F4OracleError, match="overwrite"):
        oracle._validate_output_path(output, ())


def _runtime_gates() -> dict:
    actuals = {
        "f4_incremental_warm_p95_ms": 99.0,
        "replay_composed_warm_p95_ms": 349.0,
        "replay_composed_warm_max_ms": 833.32,
        "replay_composed_mean_per_source_frame_ms": 13.9,
        "gap25_warm_deadline_miss_count": 0,
        "cuda_peak_memory_bytes": 1024,
    }
    return {
        name: {
            "actual": actuals[name],
            "comparator": comparator,
            "threshold": threshold,
            "pass": True,
            "passed": True,
        }
        for name, (comparator, threshold) in oracle.RUNTIME_GATE_SPECS.items()
    }


def _merge_receipt(sidecar: Path) -> dict:
    gates = _runtime_gates()
    for name, actual, threshold in (
        ("integrity_complete", 1, 1),
        ("exact_keyframes", 1, 1),
        ("exact_successful_frames", 1, 1),
        ("exact_sources", 1, 1),
        ("native_output_mutation_count", 0, 0),
    ):
        gates[name] = {
            "actual": actual,
            "comparator": "==",
            "threshold": threshold,
            "pass": True,
            "passed": True,
        }
    receipt = {
        "schema": oracle.F4_RECEIPT_SCHEMA,
        "protocol_id": oracle.F4_PROTOCOL_ID,
        "complete": True,
        "overall_pass": True,
        "run_signature_sha256": "a" * 64,
        "contracts": {
            "shadow_only": True,
            "birth_enabled": False,
            "native_output_mutation": False,
            "gt_access": False,
            "prediction_access": False,
            "evaluator_access": False,
            "future_frame_access": False,
            "training": False,
            "online_learning": False,
        },
        "coverage": {
            "scene_count": 1,
            "scene_order": ["scene0000_00"],
            "keyframe_count": 1,
            "successful_frame_count": 1,
            "source_count": 1,
            "exact_source_partition": True,
            "exact_source_order": True,
            "source_ids_sha256": "c" * 64,
            "source_lineage_sha256": "d" * 64,
        },
        "totals": {
            "keyframe_count": 1,
            "successful_frame_count": 1,
            "source_count": 1,
            "identity_verified_source_count": 1,
            "provider_forward_count": 1,
            "valid_hb_count": 1,
            "invalid_hb_count": 0,
        },
        "runtime": {
            "overall_pass": True,
            "gates": {name: copy.deepcopy(gates[name]) for name in oracle.RUNTIME_GATE_SPECS},
            "f4_incremental_warm_ms": {
                "count": 1, "mean": 99.0, "p50": 99.0, "p95": 99.0, "max": 99.0
            },
            "replay_composed_warm_ms": {
                "count": 1, "mean": 347.5, "p50": 347.5, "p95": 349.0, "max": 833.32
            },
            "replay_composed_all_ms": {
                "count": 1, "mean": 347.5, "p50": 347.5, "p95": 349.0, "max": 833.32
            },
            "replay_composed_mean_per_source_frame_ms": 13.9,
            "gap25_warm_deadline_miss_count": 0,
            "gap25_all_deadline_miss_count": 1,
            "cuda_peak_memory_bytes": 1024,
            "cold_model_load_excluded": True,
            "warmup_forward_count_per_shard": [1, 1],
        },
        "gates": gates,
        "causality": {
            "overall_pass": True,
            "current_frame_only": True,
            "maximum_lookahead_frames": 0,
            "maximum_logical_accessed_ordinal": True,
            "source_order_identity": True,
            "future_frame_access": False,
            "provider_called_only_for_nonempty_successful_frames": True,
            "first_three_nonempty_forwards_per_shard_excluded_only_from_warm_distributions": True,
        },
        "scenes": [
            {
                "scene_id": "scene0000_00",
                "scene_index": 0,
                "sidecar": {"path": str(sidecar), "sha256": "b" * 64},
            }
        ],
        "native_output_mutation_count": 0,
        "oracle_authorization": {
            "allowed": True,
            "scope": "separate_post_seal_f4_geometry_capacity_oracle_only",
            "active_birth_authorized": False,
        },
    }
    receipt["content_sha256"] = oracle._canonical_json_sha256(receipt)
    return receipt


def test_merge_validator_recomputes_runtime_and_causality_before_gt(
    tmp_path: Path,
) -> None:
    receipt = _merge_receipt(tmp_path / "scene0000_00.json")
    rows, signature, runtime, causality = oracle._validate_f4_receipt(
        receipt,
        ["scene0000_00"],
        expected_scene_count=1,
        expected_keyframe_count=1,
        expected_successful_frame_count=1,
        expected_source_count=1,
    )
    assert tuple(rows) == ("scene0000_00",)
    assert signature == "a" * 64
    assert runtime["overall_pass"] is True
    assert causality["overall_pass"] is True

    forged = copy.deepcopy(receipt)
    forged["gates"]["f4_incremental_warm_p95_ms"]["actual"] = 101.0
    forged["content_sha256"] = oracle._canonical_json_sha256(
        {key: value for key, value in forged.items() if key != "content_sha256"}
    )
    with pytest.raises(oracle.F4OracleError, match="inconsistent F4 merge gate"):
        oracle._validate_f4_receipt(
            forged,
            ["scene0000_00"],
            expected_scene_count=1,
            expected_keyframe_count=1,
            expected_successful_frame_count=1,
            expected_source_count=1,
        )

    forged = copy.deepcopy(receipt)
    forged["causality"]["future_frame_access"] = True
    forged["content_sha256"] = oracle._canonical_json_sha256(
        {key: value for key, value in forged.items() if key != "content_sha256"}
    )
    with pytest.raises(oracle.F4OracleError, match="causality contract"):
        oracle._validate_f4_receipt(
            forged,
            ["scene0000_00"],
            expected_scene_count=1,
            expected_keyframe_count=1,
            expected_successful_frame_count=1,
            expected_source_count=1,
        )


def _write_source_fixture(tmp_path: Path) -> tuple[Path, Path, Path, str]:
    scene = "scene0000_00"
    source_id = f"{scene}/frame_000000/raw_003"
    h0 = {
        "valid": True,
        "q02": [0.0, 0.0, 0.0],
        "q98": [1.0, 1.0, 1.0],
        "center": [0.5, 0.5, 0.5],
        "extent": [1.0, 1.0, 1.0],
        "diagnostics": {"applied": True},
    }
    base_hypotheses = {"H0": h0, "HL": copy.deepcopy(h0), "HLG": copy.deepcopy(h0)}
    f2_source = {
        "source_id": source_id,
        "candidate_index": 0,
        "rank": 0,
        "raw_index": 3,
        "confidence": 0.75,
        "stored_point_count": 4,
        "mask_sha256": "1" * 64,
        "points_and_voxel_keys_sha256": "2" * 64,
        "f0_world_q02": [0.0, 0.0, 0.0],
        "f0_world_q98": [1.0, 1.0, 1.0],
        "hypotheses": base_hypotheses,
        "f2_receipt": {"result_sha256": "9" * 64},
    }
    f0_candidate = {
        "rank": 0,
        "raw_index": 3,
        "confidence": 0.75,
        "stored_point_count": 4,
        "mask_sha256": "1" * 64,
        "points_and_voxel_keys_sha256": "2" * 64,
        "tight_box_xyxy": [10, 20, 30, 40],
        "world_q02": [0.0, 0.0, 0.0],
        "world_q98": [1.0, 1.0, 1.0],
    }
    hb = _hb_row()
    hb.update(
        {
            "row_index": 0,
            "source_id": source_id,
            "input_tight_box_xyxy": [10, 20, 30, 40],
            "abstention_reason": None,
            "logvar": [0.0],
            "raw_params": [0.0],
            "provider_result_sha256": "3" * 64,
        }
    )
    hb["result_sha256"] = oracle._canonical_json_sha256(hb)
    f0_mask = {
        "rank": 0,
        "raw_index": 3,
        "mask_sha256": "1" * 64,
        "selected": True,
        "decision": "selected",
        "tight_box_xyxy": [10, 20, 30, 40],
    }
    identity = {
        "scene_index": 0,
        "frame_ordinal": 0,
        "frame_id": 0,
        "rank": 0,
        "raw_index": 3,
        "mask_sha256": "1" * 64,
        "points_and_voxel_keys_sha256": "2" * 64,
        "source_id": source_id,
    }
    f0_lineage = {
        "candidate_sha256": oracle._canonical_json_sha256(f0_candidate),
        "mask_diagnostic_sha256": oracle._canonical_json_sha256(f0_mask),
        "provider_box_ignored": True,
    }
    f2_lineage = {
        "source_sha256": oracle._canonical_json_sha256(f2_source),
        "f2_receipt_result_sha256": "9" * 64,
    }
    tight = [10.0, 20.0, 30.0, 40.0]
    sealed_hash = oracle._canonical_json_sha256(base_hypotheses)
    join_hash = oracle._canonical_json_sha256(
        {"identity": identity, "f0": f0_lineage, "f2": f2_lineage, "tight_box_xyxy": tight}
    )
    lineage_hash = oracle._canonical_json_sha256(
        {
            "identity": identity,
            "join_sha256": join_hash,
            "sealed_f2_hypotheses_sha256": sealed_hash,
            "hb_result_sha256": hb["result_sha256"],
        }
    )
    f4_source = {
        "source_id": source_id,
        "scene_index": 0,
        "frame_ordinal": 0,
        "frame_id": 0,
        "rank": 0,
        "raw_index": 3,
        "candidate_index": 0,
        "mask_sha256": "1" * 64,
        "points_and_voxel_keys_sha256": "2" * 64,
        "tight_box_xyxy": tight,
        "f0_source_lineage": f0_lineage,
        "f2_source_lineage": f2_lineage,
        "sealed_f2_hypotheses_sha256": sealed_hash,
        "hypotheses": {**copy.deepcopy(base_hypotheses), "HB": hb},
        "join_sha256": join_hash,
        "source_lineage_sha256": lineage_hash,
    }
    f0 = {
        "schema": oracle.F0_SCENE_SCHEMA,
        "scene_id": scene,
        "frames": [
            {
                "frame_id": 0,
                "frame_ordinal": 0,
                "successful": True,
                "funnel": {"candidates": [f0_candidate], "masks": [f0_mask]},
            }
        ],
    }
    f2 = {
        "schema": oracle.F2_SCENE_SCHEMA,
        "protocol_id": oracle.F2_PROTOCOL_ID,
        "scene_id": scene,
        "frames": [
            {
                "frame_id": 0,
                "frame_ordinal": 0,
                "successful": True,
                "sources": [f2_source],
            }
        ],
    }
    f0_path = tmp_path / "f0.json"
    f2_path = tmp_path / "f2.json"
    f4_path = tmp_path / "f4.json"
    f0_path.write_text(json.dumps(f0, sort_keys=True), encoding="utf-8")
    f2_path.write_text(json.dumps(f2, sort_keys=True), encoding="utf-8")
    f4 = {
        "schema": oracle.F4_SCENE_SCHEMA,
        "protocol_id": oracle.F4_PROTOCOL_ID,
        "complete": True,
        "scene_id": scene,
        "scene_index": 0,
        "run_signature_sha256": "a" * 64,
        "contracts": _merge_receipt(tmp_path / "unused")["contracts"],
        "inputs": {
            "f2_sidecar": {"path": str(f2_path.resolve()), "sha256": oracle._sha256(f2_path)},
            "f0_sidecar": {"path": str(f0_path.resolve()), "sha256": oracle._sha256(f0_path)},
            "frozen_inputs_before_sha256": "6" * 64,
            "frozen_inputs_after_sha256": "6" * 64,
            "model_receipts_sha256": "7" * 64,
        },
        "frames": [
            {
                "frame_id": 0,
                "frame_ordinal": 0,
                "successful": True,
                "current_only": True,
                "max_accessed_frame_ordinal": 0,
                "provider_invoked": True,
                "sources": [f4_source],
            }
        ],
        "counts": {
            "keyframe_count": 1,
            "successful_frame_count": 1,
            "source_count": 1,
            "provider_forward_count": 1,
            "valid_hb_count": 1,
            "invalid_hb_count": 0,
        },
        "runtime": {},
        "source_ids_sha256": oracle._canonical_json_sha256([source_id]),
        "source_lineage_sha256": oracle._canonical_json_sha256([lineage_hash]),
        "native_output_mutation_count": 0,
    }
    f4["content_sha256"] = oracle._canonical_json_sha256(f4)
    f4_path.write_text(json.dumps(f4, sort_keys=True), encoding="utf-8")
    return f0_path, f2_path, f4_path, oracle._sha256(f4_path)


def test_pre_gt_source_loader_rejoins_f0_f2_and_validates_hypothesis_hashes(
    tmp_path: Path,
) -> None:
    f0, f2, f4, sha = _write_source_fixture(tmp_path)
    rows, keyframes, successful = oracle._load_f4_sources_pre_gt(
        path=f4,
        f2_path=f2,
        f0_path=f0,
        scene="scene0000_00",
        scene_index=0,
        receipt_sidecar_sha256=sha,
        run_signature_sha256="a" * 64,
    )
    assert (keyframes, successful, len(rows)) == (1, 1, 1)
    assert rows[0].source_id.endswith("raw_003")

    payload = json.loads(f4.read_text(encoding="utf-8"))
    payload["frames"][0]["sources"][0]["hypotheses"]["HLG"]["q98"][0] = 2.0
    payload["content_sha256"] = oracle._canonical_json_sha256(
        {key: value for key, value in payload.items() if key != "content_sha256"}
    )
    f4.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    with pytest.raises(oracle.F4OracleError, match="not exact F2 copies"):
        oracle._load_f4_sources_pre_gt(
            path=f4,
            f2_path=f2,
            f0_path=f0,
            scene="scene0000_00",
            scene_index=0,
            receipt_sidecar_sha256=oracle._sha256(f4),
            run_signature_sha256="a" * 64,
        )


def test_pre_gt_loader_accepts_sealed_nonfinite_rotation_abstention(
    tmp_path: Path,
) -> None:
    f0, f2, f4, _ = _write_source_fixture(tmp_path)
    payload = json.loads(f4.read_text(encoding="utf-8"))
    source = payload["frames"][0]["sources"][0]
    hb = source["hypotheses"]["HB"]
    hb.update(
        {
            "valid": False,
            "abstention_reason": "nonfinite_rotation",
            "world_corners": None,
            "world_center": None,
            "local_extent": None,
            "world_rotation": None,
            "camera_depth": None,
            "validity": {
                "finite_center": True,
                "finite_extent": True,
                "finite_rotation": False,
                "finite_corners": False,
                "positive_extent": True,
                "right_handed_orthonormal": False,
                "in_front": True,
                "orthogonality_error": None,
                "determinant": None,
                "rotation_correction_max_abs": None,
                "reasons": ["nonfinite_rotation", "nonfinite_corners"],
            },
        }
    )
    hb.pop("result_sha256")
    hb["result_sha256"] = oracle._canonical_json_sha256(hb)
    identity = {
        key: source[key]
        for key in (
            "scene_index",
            "frame_ordinal",
            "frame_id",
            "rank",
            "raw_index",
            "mask_sha256",
            "points_and_voxel_keys_sha256",
            "source_id",
        )
    }
    source["source_lineage_sha256"] = oracle._canonical_json_sha256(
        {
            "identity": identity,
            "join_sha256": source["join_sha256"],
            "sealed_f2_hypotheses_sha256": source[
                "sealed_f2_hypotheses_sha256"
            ],
            "hb_result_sha256": hb["result_sha256"],
        }
    )
    payload["counts"]["valid_hb_count"] = 0
    payload["counts"]["invalid_hb_count"] = 1
    payload["source_lineage_sha256"] = oracle._canonical_json_sha256(
        [source["source_lineage_sha256"]]
    )
    payload.pop("content_sha256")
    payload["content_sha256"] = oracle._canonical_json_sha256(payload)
    f4.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")

    rows, _, _ = oracle._load_f4_sources_pre_gt(
        path=f4,
        f2_path=f2,
        f0_path=f0,
        scene="scene0000_00",
        scene_index=0,
        receipt_sidecar_sha256=oracle._sha256(f4),
        run_signature_sha256="a" * 64,
    )
    assert rows[0].hb["valid"] is False


def test_main_writes_once_with_constant_score_oracle_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    report = {
        "schema": oracle.SCHEMA,
        "totals": {"source_count": 1},
        "decision": {"authorize_active_birth": False},
    }
    monkeypatch.setattr(
        oracle, "audit_scannet_fastsam_f4_boxer_paper100_oracle", lambda **_: report
    )
    output = tmp_path / "oracle.json"
    assert oracle.main(["--out", str(output)]) == 0
    assert json.loads(output.read_text(encoding="utf-8")) == report
    with pytest.raises(oracle.F4OracleError, match="overwrite"):
        oracle.main(["--out", str(output)])


def test_runner_merge_to_oracle_validators_share_the_exact_schema(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Exercise the real runner/merge writers without opening GT or predictions."""

    root = Path(__file__).resolve().parents[1]
    for directory in (root / "tools", root / "tests"):
        if str(directory) not in sys.path:
            sys.path.insert(0, str(directory))
    helper_spec = importlib.util.spec_from_file_location(
        "f4_runner_test_helpers",
        root / "tests/test_run_scannet_fastsam_f4_boxer_paper100.py",
    )
    assert helper_spec is not None and helper_spec.loader is not None
    helpers = importlib.util.module_from_spec(helper_spec)
    helper_spec.loader.exec_module(helpers)
    import merge_scannet_fastsam_f4_boxer_paper100 as merger

    original_infer = helpers._FakeProvider.infer_batch

    def schema_complete_infer(self, **kwargs):
        result = original_infer(self, **kwargs)
        for row in result["rows"]:
            row["validity"] = {
                "finite_center": True,
                "finite_extent": True,
                "finite_rotation": True,
                "positive_extent": True,
                "right_handed_orthonormal": True,
                "in_front": True,
                "finite_corners": True,
                "orthogonality_error": 0.0,
                "determinant": 1.0,
                "rotation_correction_max_abs": 0.0,
                "reasons": [],
            }
            row["result_sha256"] = "e" * 64
        return result

    monkeypatch.setattr(helpers._FakeProvider, "infer_batch", schema_complete_infer)
    inputs = helpers._prepare_inputs(tmp_path)
    helpers._run_shard(inputs, 0, [])
    helpers._run_shard(inputs, 1, [])
    shards = (
        inputs["output"] / "shards/shard-000-of-002.json",
        inputs["output"] / "shards/shard-001-of-002.json",
    )
    merger.merge_f4(
        shard_paths=shards,
        output_dir=tmp_path / "final",
        expected_scene_count=2,
        expected_keyframes=2,
        expected_successful_frames=2,
        expected_sources=2,
    )
    receipt = json.loads(
        (tmp_path / "final/F4_FASTSAM_BOXER_PAPER100.json").read_text(
            encoding="utf-8"
        )
    )
    scenes = ["scene0000_00", "scene0001_00"]
    rows, signature, runtime, causality = oracle._validate_f4_receipt(
        receipt,
        scenes,
        expected_scene_count=2,
        expected_keyframe_count=2,
        expected_successful_frame_count=2,
        expected_source_count=2,
    )
    assert runtime["overall_pass"] and causality["overall_pass"]
    for scene_index, scene in enumerate(scenes):
        f4_path = Path(rows[scene]["sidecar"]["path"])
        f4_scene = json.loads(f4_path.read_text(encoding="utf-8"))
        loaded, keyframes, successful = oracle._load_f4_sources_pre_gt(
            path=f4_path,
            f2_path=Path(f4_scene["inputs"]["f2_sidecar"]["path"]),
            f0_path=Path(f4_scene["inputs"]["f0_sidecar"]["path"]),
            scene=scene,
            scene_index=scene_index,
            receipt_sidecar_sha256=rows[scene]["sidecar"]["sha256"],
            run_signature_sha256=signature,
        )
        assert (keyframes, successful, len(loaded)) == (1, 1, 1)
