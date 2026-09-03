from __future__ import annotations

import hashlib
import json
import math
import pickle

import numpy as np
import pytest
import torch

from boxfusion.p1_geometry_refiner import (
    P1G_ARCHITECTURE,
    P1G_CHECKPOINT_SCHEMA,
    P1G_REGRESSION_ENCODING,
    P1GeometryRegressionHead,
)
from boxfusion.p1_spatial_residual import (
    NativeSparseResidualProposalHead,
)
from boxfusion.residual_proposal import (
    P1_DIAGNOSTIC_SCHEMA,
    P1_FEATURE_NAMES,
    P1S_HEAD_SCHEMA,
    ResidualProposalConfig,
)
from tools.evaluate_p1g_candidate_audit import (
    FrozenCandidates,
    _bounded_refined_box,
    build_go_no_go,
    evaluate,
    novel_threshold_crossings,
    refinable_iou_quality,
    scene_bootstrap_novel_recall_delta,
    validate_candidate_identity,
    validate_stage_scene_binding,
)


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _file_sha(path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def _candidate(candidate_id: str) -> FrozenCandidates:
    return FrozenCandidates(
        candidate_ids=np.asarray([candidate_id], dtype=np.str_),
        frame_ids=np.asarray([0], dtype=np.int64),
        provider_steps=np.asarray([0], dtype=np.int64),
        boxes_world=np.asarray([[0, 0, 0, 1, 1, 1]], dtype=np.float64),
        scores=np.asarray([0.9], dtype=np.float64),
    )


def test_bounded_decoder_and_candidate_identity_fail_closed():
    refined = _bounded_refined_box(
        np.zeros(3),
        np.asarray([8.0, -8.0, 0.0, 8.0, -8.0, 0.0]),
        np.zeros(6),
        max_center_offset=1.0,
        min_box_extent=0.1,
        max_box_extent=4.0,
        adapter_epsilon=1e-6,
    )
    np.testing.assert_allclose(
        refined[:3], [1.0, -1.0, 0.0], rtol=1e-3, atol=1e-3
    )
    np.testing.assert_allclose(
        refined[3:],
        [4.0, 0.1, 1.0],
        rtol=2e-3,
        atol=2e-3,
    )

    identity = validate_candidate_identity(
        _candidate("scene0000_00:000000:0:0:0"),
        _candidate("scene0000_00:000000:0:0:0"),
    )
    assert identity["passes"] is True
    with pytest.raises(ValueError, match="identity contract"):
        validate_candidate_identity(
            _candidate("scene0000_00:000000:0:0:0"),
            _candidate("scene0000_00:000000:1:0:0"),
        )


def _gate_thresholds(
    *,
    delta_tp: int,
    delta_recall: float,
    up: int,
    down: int,
) -> dict:
    return {
        "0.25": {
            "raw": {"novel_recall": 0.20},
            "refined": {"novel_recall": 0.20},
            "delta_novel_recall": 0.0,
            "delta_novel_true_positives": 0,
            "crossings": {"up": 0, "down": 0, "net": 0},
        },
        "0.50": {
            "raw": {"novel_recall": 0.01},
            "refined": {"novel_recall": 0.01 + delta_recall},
            "delta_novel_recall": delta_recall,
            "delta_novel_true_positives": delta_tp,
            "crossings": {
                "up": up,
                "down": down,
                "net": up - down,
            },
        },
    }


def test_unique_novel_crossing_is_one_to_one_and_module20_gate():
    empty_boxes = np.empty((0, 6), dtype=np.float64)
    empty_scores = np.empty((0,), dtype=np.float64)
    raw = np.asarray([[0.7, -0.5, -0.5, 1.7, 0.5, 0.5]])
    refined = np.asarray([[-0.3, -0.5, -0.5, 0.7, 0.5, 0.5]])
    gt = np.asarray([[-0.5, -0.5, -0.5, 0.5, 0.5, 0.5]])
    crossing = novel_threshold_crossings(
        baseline_boxes=empty_boxes,
        baseline_scores=empty_scores,
        raw_boxes=raw,
        refined_boxes=refined,
        candidate_scores=np.asarray([0.9]),
        gt_boxes=gt,
        threshold=0.50,
    )
    assert crossing["up"] == 1
    assert crossing["down"] == 0
    assert crossing["net"] == 1

    # One refined candidate cannot claim both GT rows.
    duplicate_gt = np.repeat(gt, 2, axis=0)
    crossing = novel_threshold_crossings(
        baseline_boxes=empty_boxes,
        baseline_scores=empty_scores,
        raw_boxes=raw,
        refined_boxes=refined,
        candidate_scores=np.asarray([0.9]),
        gt_boxes=duplicate_gt,
        threshold=0.50,
    )
    assert crossing["up"] == 1
    assert crossing["refined_score_ordered_novel_tp"] == 1

    thresholds = _gate_thresholds(
        delta_tp=2, delta_recall=0.005, up=3, down=1
    )
    gate = build_go_no_go(
        stage="module20",
        thresholds=thresholds,
        total_gt=100,
        scene_count=20,
        refinement_quality={
            "matched_count": 10,
            "median_delta_iou": 0.02,
            "harm_rate": 0.12,
        },
        bootstrap_ci=None,
        safety_identity=True,
        candidates_bounded=True,
        refiner_seconds_per_scene=0.10,
        refiner_p95_seconds_per_scene=0.20,
    )
    assert gate["passes"] is True
    assert gate["decision"] == "GO_FRESH50_AUDIT"

    gate = build_go_no_go(
        stage="module20",
        thresholds=thresholds,
        total_gt=100,
        scene_count=20,
        refinement_quality={
            "matched_count": 10,
            "median_delta_iou": 0.02,
            "harm_rate": 0.12,
        },
        bootstrap_ci=None,
        safety_identity=True,
        candidates_bounded=True,
        refiner_seconds_per_scene=0.151,
        refiner_p95_seconds_per_scene=0.20,
    )
    assert gate["passes"] is False
    assert gate["decision"] == "STOP_P1G1_MODULE_AUDIT"


def test_fresh50_gate_requires_ci_ratio_and_quality():
    thresholds = _gate_thresholds(
        delta_tp=10, delta_recall=0.01, up=12, down=2
    )
    kwargs = {
        "stage": "fresh50",
        "thresholds": thresholds,
        "total_gt": 1_000,
        "scene_count": 50,
        "refinement_quality": {
            "matched_count": 100,
            "median_delta_iou": 0.03,
            "harm_rate": 0.10,
        },
        "bootstrap_ci": {"lower": 0.001, "upper": 0.02},
        "safety_identity": True,
        "candidates_bounded": True,
        "refiner_seconds_per_scene": 0.10,
        "refiner_p95_seconds_per_scene": 0.20,
    }
    gate = build_go_no_go(**kwargs)
    assert gate["passes"] is True
    assert gate["decision"] == "GO_ONE_SHOT_VAL10_OBSERVER"

    failed = dict(kwargs)
    failed["bootstrap_ci"] = {"lower": 0.0, "upper": 0.02}
    gate = build_go_no_go(**failed)
    assert gate["passes"] is False
    assert gate["decision"] == "STOP_P1G1"


def test_refinable_quality_and_scene_bootstrap_are_deterministic():
    raw = np.asarray([[-0.5, -0.5, -0.5, 0.3, 0.5, 0.5]])
    refined = np.asarray([[-0.5, -0.5, -0.5, 0.5, 0.5, 0.5]])
    gt = np.asarray([[-0.5, -0.5, -0.5, 0.5, 0.5, 0.5]])
    quality, deltas = refinable_iou_quality(
        raw_boxes=raw,
        refined_boxes=refined,
        candidate_scores=np.asarray([0.9]),
        candidate_ids=np.asarray(["candidate"]),
        gt_boxes=gt,
    )
    assert quality["matched_count"] == 1
    assert quality["median_delta_iou"] == pytest.approx(deltas[0])
    assert quality["harm_rate"] == 0.0

    rows = [
        {
            "ground_truth_count": 10,
            "raw_novel_true_positives": 1,
            "refined_novel_true_positives": 2,
        }
        for _ in range(50)
    ]
    first = scene_bootstrap_novel_recall_delta(rows, resamples=100)
    second = scene_bootstrap_novel_recall_delta(rows, resamples=100)
    assert first == second
    assert first["lower"] == pytest.approx(0.1)


def test_stage_scene_binding_separates_module20_from_fresh50():
    p1s_provenance = {
        "train_scene_ids": ["scene0000_00"],
        "scene_summaries": [{"scene_id": "scene0002_00"}],
    }
    p1g_provenance = {
        "fit_scene_ids": ["scene0000_00"],
        "cal_scene_ids": ["scene0001_00"],
        "audit_scene_ids": ["scene0002_00"],
        "audit_scene_list_sha256": _sha("module-list"),
    }
    module = validate_stage_scene_binding(
        stage="module20",
        scenes=["scene0002_00"],
        scene_list_sha256=_sha("module-list"),
        p1s_provenance=p1s_provenance,
        p1g_provenance=p1g_provenance,
    )
    assert module["status"] == "p1g_checkpoint_exact_module20_binding"

    fresh = validate_stage_scene_binding(
        stage="fresh50",
        scenes=["scene0100_00"],
        scene_list_sha256=_sha("fresh-list"),
        p1s_provenance=p1s_provenance,
        p1g_provenance=p1g_provenance,
    )
    assert fresh["status"] == (
        "fresh50_external_hash_and_disjoint_provenance"
    )
    with pytest.raises(ValueError, match="overlaps frozen"):
        validate_stage_scene_binding(
            stage="fresh50",
            scenes=["scene0002_00"],
            scene_list_sha256=_sha("bad-fresh-list"),
            p1s_provenance=p1s_provenance,
            p1g_provenance=p1g_provenance,
        )


def _write_synthetic_artifacts(tmp_path):
    scene_id = "scene0002_00"
    diagnostics = tmp_path / "diagnostics"
    predictions = tmp_path / "predictions"
    gt_root = tmp_path / "gt"
    scans = tmp_path / "scans"
    diagnostics.mkdir()
    predictions.mkdir()
    gt_root.mkdir()
    (scans / scene_id).mkdir(parents=True)
    scene_list = tmp_path / "audit.txt"
    scene_list.write_text(scene_id + "\n", encoding="utf-8")

    source_config = ResidualProposalConfig(
        enabled=True,
        observer_only=True,
        mutate=False,
        collect_diagnostics=True,
        mode="collect",
        collect_voxel_inputs=True,
    ).validated()
    np.savez_compressed(
        diagnostics / f"{scene_id}_tracks.npz",
        scene_id=np.asarray(scene_id),
        p1_schema=np.asarray(P1_DIAGNOSTIC_SCHEMA),
        p1_stage=np.asarray("P1"),
        p1_enabled=np.asarray(True),
        p1_observer_only=np.asarray(True),
        p1_uses_ground_truth=np.asarray(False),
        p1_mutation_enabled=np.asarray(False),
        p1_applied_count=np.asarray(0, dtype=np.int64),
        p1_complete=np.asarray(True),
        p1_class_agnostic=np.asarray(True),
        p1_regression_dim=np.asarray(6, dtype=np.int64),
        p1_config_json=np.asarray(
            json.dumps(source_config.to_dict(), sort_keys=True)
        ),
        p1_feature_names=np.asarray(P1_FEATURE_NAMES, dtype=np.str_),
        p1_step_frame_ids=np.asarray([0], dtype=np.int64),
        p1_step_provider_steps=np.asarray([0], dtype=np.int64),
        p1_step_voxel_counts=np.asarray([1], dtype=np.int64),
        p1_step_input_point_counts=np.asarray([1], dtype=np.int64),
        p1_step_explained_point_counts=np.asarray([0], dtype=np.int64),
        p1_step_residual_point_counts=np.asarray([1], dtype=np.int64),
        p1_voxel_offsets=np.asarray([0, 1], dtype=np.int64),
        p1_voxel_coords=np.asarray([[0, 0, 0]], dtype=np.int32),
        p1_voxel_centers=np.asarray([[0.7, 0.0, 0.0]], dtype=np.float32),
        p1_voxel_features=np.zeros(
            (1, len(P1_FEATURE_NAMES)), dtype=np.float32
        ),
        p1_voxel_point_counts=np.asarray([1], dtype=np.int32),
    )
    with (predictions / f"{scene_id}_boxes.pkl").open("wb") as handle:
        pickle.dump([[]], handle)
    np.save(
        gt_root / f"{scene_id}_bbox.npy",
        np.asarray([[0, 0, 0, 1, 1, 1]], dtype=np.float32),
    )
    identity = np.eye(4, dtype=np.float64).reshape(-1)
    (scans / scene_id / f"{scene_id}.txt").write_text(
        "axisAlignment = " + " ".join(str(value) for value in identity) + "\n",
        encoding="utf-8",
    )

    b6_sha = _sha("b6")
    p1s = NativeSparseResidualProposalHead(
        input_dim=len(P1_FEATURE_NAMES),
        hidden_dim=8,
        regression_dim=6,
    )
    with torch.no_grad():
        for parameter in p1s.parameters():
            parameter.zero_()
        p1s.objectness.bias.fill_(10.0)
    p1s_path = tmp_path / "p1s.pt"
    torch.save(
        {
            "schema": P1S_HEAD_SCHEMA,
            "variant": "P1S",
            "head_architecture": "native_sparse_context_v1",
            "target_assignment_scope": "snapshot_inside_only",
            "model_config": p1s.model_config(),
            "feature_names": list(P1_FEATURE_NAMES),
            "state_dict": p1s.state_dict(),
            "training_config": {
                "target_assignment_scope": "snapshot_inside_only"
            },
            "provenance": {
                "train_scene_ids": ["scene0000_00"],
                "forbidden_overlap": [],
                "b6_checkpoint_sha256": b6_sha,
                "train_scene_list_sha256": _sha("p1s-train"),
                "forbidden_scene_list_sha256": _sha("p1s-forbidden"),
                "scene_summaries": [
                    {
                        "scene_id": scene_id,
                        "diagnostic_sha256": _file_sha(
                            diagnostics / f"{scene_id}_tracks.npz"
                        ),
                        "prediction_sha256": _file_sha(
                            predictions / f"{scene_id}_boxes.pkl"
                        ),
                        "ground_truth_sha256": _file_sha(
                            gt_root / f"{scene_id}_bbox.npy"
                        ),
                        "axis_alignment_sha256": _file_sha(
                            scans / scene_id / f"{scene_id}.txt"
                        ),
                    }
                ],
            },
        },
        p1s_path,
    )
    p1s_sha = _file_sha(p1s_path)

    p1g = P1GeometryRegressionHead(hidden_dim=8)
    with torch.no_grad():
        p1g.correction.weight.zero_()
        p1g.correction.bias.copy_(
            torch.tensor(
                [
                    math.atanh(-0.5),
                    0,
                    0,
                    0,
                    0,
                    0,
                ],
                dtype=torch.float32,
            )
        )
    p1g_path = tmp_path / "p1g.pt"
    p1g_model_config = p1g.model_config(
        max_center_offset=1.0,
        min_box_extent=0.08,
        max_box_extent=4.0,
    )
    torch.save(
        {
            "schema": P1G_CHECKPOINT_SCHEMA,
            "architecture": P1G_ARCHITECTURE,
            "regression_encoding": P1G_REGRESSION_ENCODING,
            "initialization": (
                "zero_residual_correction_function_preserving_v2"
            ),
            "observer_only": True,
            "uses_ground_truth": False,
            "class_agnostic": True,
            "semantic_features": False,
            "model_config": p1g_model_config,
            "decoder_config": {
                "encoding": P1G_REGRESSION_ENCODING,
                "adapter_epsilon": p1g_model_config["adapter_epsilon"],
                "max_center_offset": p1g_model_config[
                    "max_center_offset"
                ],
                "min_box_extent": p1g_model_config["min_box_extent"],
                "max_box_extent": p1g_model_config["max_box_extent"],
            },
            "state_dict": p1g.state_dict(),
            "provenance": {
                "p1s_checkpoint_sha256": p1s_sha,
                "fit_scene_ids": ["scene0000_00"],
                "cal_scene_ids": ["scene0001_00"],
                "audit_scene_ids": [scene_id],
                "forbidden_overlap": [],
                "fit_scene_list_sha256": _sha("fit"),
                "cal_scene_list_sha256": _sha("cal"),
                "audit_scene_list_sha256": _file_sha(scene_list),
                "forbidden_scene_list_sha256": _sha("forbidden"),
                "dataset_fingerprint_sha256": _sha("dataset"),
            },
        },
        p1g_path,
    )
    return {
        "scene": scene_id,
        "scene_list": scene_list,
        "diagnostics": diagnostics,
        "predictions": predictions,
        "gt": gt_root,
        "scans": scans,
        "p1s": p1s_path,
        "p1g": p1g_path,
    }


def test_end_to_end_cpu_replays_raw_nms_then_replaces_same_candidate(tmp_path):
    artifacts = _write_synthetic_artifacts(tmp_path)
    report = evaluate(
        stage="module20",
        scene_list=artifacts["scene_list"],
        p1s_checkpoint=artifacts["p1s"],
        p1g_checkpoint=artifacts["p1g"],
        source_diagnostics_root=artifacts["diagnostics"],
        prediction_root=artifacts["predictions"],
        gt_root=artifacts["gt"],
        scans_root=artifacts["scans"],
        device="cpu",
        maximum_refiner_seconds_per_scene=10.0,
        maximum_refiner_p95_seconds_per_scene=10.0,
    )
    assert report["training_performed"] is False
    assert report["runtime_scope"] == "correction_forward_decode_only"
    assert report["full_live_runtime_verified"] is False
    assert report["full_online_activation_authorized"] is False
    assert report["candidate_count"] == 1
    assert report["candidate_identity"]["passes"] is True
    assert report["thresholds"]["0.25"]["raw"]["novel_true_positives"] == 0
    assert (
        report["thresholds"]["0.25"]["refined"]["novel_true_positives"]
        == 1
    )
    assert report["thresholds"]["0.50"]["crossings"] == {
        "up": 1,
        "down": 0,
        "net": 1,
    }
    # A one-scene fixture cannot satisfy module20 completeness or +2 AP50 TP.
    assert report["go_no_go"]["passes"] is False
    assert report["go_no_go"]["decision"] == "STOP_P1G1_MODULE_AUDIT"


def test_audit_fails_closed_when_scene_list_not_checkpoint_bound(tmp_path):
    artifacts = _write_synthetic_artifacts(tmp_path)
    artifacts["scene_list"].write_text(
        "scene0003_00\n", encoding="utf-8"
    )
    with pytest.raises((FileNotFoundError, ValueError)):
        evaluate(
            stage="module20",
            scene_list=artifacts["scene_list"],
            p1s_checkpoint=artifacts["p1s"],
            p1g_checkpoint=artifacts["p1g"],
            source_diagnostics_root=artifacts["diagnostics"],
            prediction_root=artifacts["predictions"],
            gt_root=artifacts["gt"],
            scans_root=artifacts["scans"],
            device="cpu",
        )
