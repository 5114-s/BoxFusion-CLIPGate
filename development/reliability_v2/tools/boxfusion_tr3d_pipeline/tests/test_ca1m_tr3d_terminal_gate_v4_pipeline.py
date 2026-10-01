from __future__ import annotations

import json
import hashlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from boxfusion.ca1m_tr3d_terminal_gate_v4 import (
    BENEFIT_TARGET,
    FEATURE_NAMES,
    FEATURE_SCHEMA,
    GATE_TRAIN_FOLDS,
    LOCKED_INTERNAL_FOLDS,
    MAX_REPLACEMENTS,
    POLICY_SCHEMA,
    PREREGISTRATION_SCHEMA,
    QUALITY_TARGET,
    SELECTION_RULE,
    THRESHOLD_DEV_FOLDS,
    build_terminal_gate_features_v4,
    load_gate_policy_v4,
    materialize_geometry_only,
    select_terminal_replacements_v4,
    preregistration_code_records,
    preregistration_science_contract,
    validate_candidate_evidence_artifact,
    write_binding_create_only,
)
from boxfusion.ca1m_tr3d_terminal_v4 import sha256_file
from tools.materialize_ca1m_tr3d_terminal_active_v4 import _validate_training_report
from tools.build_ca1m_tr3d_benefit_dataset_v4 import SCORE_SOURCE, labeled_scene
from tools.train_ca1m_tr3d_benefit_gate_v4 import (
    REPORT_SCHEMA,
    official_ca_ap,
    score_tie_audit,
)


def _box(center: tuple[float, float, float], size: tuple[float, float, float]) -> np.ndarray:
    center_value = np.asarray(center, np.float32)
    half = np.asarray(size, np.float32) * 0.5
    return np.asarray([
        center_value + np.asarray((x, y, z), np.float32) * half
        for x in (-1.0, 1.0)
        for y in (-1.0, 1.0)
        for z in (-1.0, 1.0)
    ], np.float32)


def _feature_inputs():
    anchors = np.asarray([
        _box((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
        _box((4.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
    ], np.float32)
    candidates = np.asarray([
        _box((0.05, 0.0, 0.0), (1.1, 1.0, 1.0)),
        _box((0.10, 0.0, 0.0), (0.9, 1.0, 1.0)),
        _box((8.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
    ], np.float32)
    candidate_scores = np.asarray((0.9, 0.8, 0.7), np.float32)
    proposal = {
        "summary": SimpleNamespace(scene_id="42000000", point_count=1000),
        "candidate_corners_world": candidates,
        "candidate_scores": candidate_scores,
        "candidate_point_count": np.asarray((50, 30, 10), np.int64),
    }
    overlay = {
        "summary": SimpleNamespace(scene_id="42000000"),
        "anchor_corners": anchors,
        "active_anchor_scores": np.asarray((0.85, 0.65), np.float32),
        "candidate_corners_world": candidates.copy(),
        "candidate_scores": candidate_scores.copy(),
        "best_anchor_indices": np.asarray((0, 0, 1), np.int64),
        "best_anchor_iou": np.asarray((0.75, 0.65, 0.0), np.float32),
        "best_anchor_center_distance_m": np.asarray((0.05, 0.10, 4.0), np.float32),
        "near_mask": np.asarray((True, True, False), np.bool_),
    }
    anchor_native = np.full((2, 14), 0.2, np.float32)
    anchor_native[:, 0] = np.asarray((0.31, 0.41), np.float32)
    candidate_native = np.full((3, 14), 0.3, np.float32)
    candidate_native[:, 0] = candidate_scores
    oof = np.asarray((0.44, 0.54), np.float32)
    return proposal, overlay, anchor_native, candidate_native, oof


def test_v4_40d_features_force_oof_anchor_scores_and_preserve_candidate_rows():
    proposal, overlay, anchor_native, candidate_native, oof = _feature_inputs()
    batch = build_terminal_gate_features_v4(
        proposal=proposal,
        overlay=overlay,
        anchor_native_evidence=anchor_native,
        anchor_native_detector_scores=anchor_native[:, 0],
        candidate_native_evidence=candidate_native,
        anchor_scores=oof,
        score_source=SCORE_SOURCE,
    )
    assert batch.schema == FEATURE_SCHEMA
    assert batch.features.shape == (2, 40)
    assert np.array_equal(batch.candidate_rows, np.asarray((0, 1), np.int64))
    assert np.array_equal(batch.anchor_indices, np.asarray((0, 0), np.int64))
    assert np.array_equal(batch.features[:, 0], np.asarray((0.44, 0.44), np.float32))
    assert np.allclose(batch.features[:, 28], np.asarray((0.46, 0.36), np.float32))
    assert not batch.features.flags.writeable
    with pytest.raises(ValueError, match="score source"):
        build_terminal_gate_features_v4(
            proposal=proposal, overlay=overlay,
            anchor_native_evidence=anchor_native,
            anchor_native_detector_scores=anchor_native[:, 0],
            candidate_native_evidence=candidate_native,
            anchor_scores=oof, score_source="legacy_deploy_score",
        )


def test_v4_world_aabb_targets_use_strict_quality_and_same_gt_gain():
    anchor = np.asarray([_box((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))], np.float32)
    candidates = np.asarray([
        _box((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
        _box((5.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
    ], np.float32)
    gt = np.asarray([_box((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))], np.float64)
    result = labeled_scene(
        anchor_corners=anchor,
        anchor_scores=np.asarray((0.4,), np.float32),
        candidate_corners=candidates,
        candidate_rows=np.asarray((0, 1), np.int64),
        anchor_indices=np.asarray((0, 0), np.int64),
        gt_corners=gt,
    )
    assert result["quality25_target"].tolist() == [True, False]
    assert result["benefit05_target"].tolist() == [False, False]
    # With a single GT, even the zero-IoU candidate has the same argmax target;
    # it is rejected by quality/gain rather than being mislabeled a switch.
    assert result["target_switch"].tolist() == [False, False]


def test_official_ca_ap_is_global_duplicate_aware_and_strict():
    result = official_ca_ap(
        scene_ids=np.asarray(("42000000", "42000000", "42000000")),
        scores=np.asarray((0.9, 0.8, 0.7), np.float32),
        best_iou=np.asarray((0.30, 0.90, 0.25), np.float64),
        best_gt=np.asarray((0, 0, 1), np.int64),
        gt_counts=np.asarray((2,), np.int64),
    )
    assert result["iou_0.25"]["tp"] == 1
    assert result["iou_0.25"]["fp"] == 2
    assert result["iou_0.50"]["tp"] == 1
    # IoU exactly 0.25 is excluded by the official strict-greater gate.
    assert result["iou_0.25"]["fn"] == 1


def test_official_ca_ap_bound_scores_have_explicit_no_tie_audit():
    audit = score_tie_audit(np.asarray((0.3, 0.2, 0.1), np.float32))
    assert audit["tied_score_value_count"] == 0
    assert audit["rows_in_ties"] == 0
    assert audit["default_quicksort_equals_stable_order"] is True
    tied = score_tie_audit(np.asarray((0.3, 0.3, 0.1), np.float32))
    assert tied["tied_score_value_count"] == 1
    assert tied["rows_in_ties"] == 2


def test_candidate_evidence_binding_recomputes_file_sha(tmp_path: Path):
    scene = "42000000"
    artifact = tmp_path / f"{scene}_ca1m_tr3d_candidate_evidence_v4.npz"
    artifact.write_bytes(b"sealed-candidate-evidence")
    artifact.chmod(0o444)
    digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
    assert validate_candidate_evidence_artifact(
        artifact, scene=scene, expected_sha256=digest, expected_root=tmp_path
    ) == artifact.resolve()
    with pytest.raises(ValueError, match="SHA256 differs"):
        validate_candidate_evidence_artifact(
            artifact, scene=scene, expected_sha256="0" * 64, expected_root=tmp_path
        )


def _preregistration(path: Path) -> dict[str, object]:
    payload = {
        "schema": PREREGISTRATION_SCHEMA,
        "complete": True,
        "train_only": True,
        "sealed_before_first_gt_join": True,
        "ground_truth_access": False,
        "validation_ground_truth_access": False,
        "validation_prediction_access": False,
        "official_validation_comparable": False,
        "locked_internal_fold1_gt_access": False,
        "fit_fold_ids": list(GATE_TRAIN_FOLDS),
        "threshold_dev_fold_ids": list(THRESHOLD_DEV_FOLDS),
        "locked_internal_fold_ids": list(LOCKED_INTERNAL_FOLDS),
        "anchor_score_source": SCORE_SOURCE,
        "deploy_b6_scores_used_for_stacked_training": False,
        "feature_schema": FEATURE_SCHEMA,
        "feature_names": list(FEATURE_NAMES),
        "quality_target": QUALITY_TARGET,
        "benefit_target": BENEFIT_TARGET,
        "selection_rule": SELECTION_RULE,
        "science": preregistration_science_contract(),
        "code": preregistration_code_records(),
        "upstream": {},
    }
    write_binding_create_only(path, payload)
    return payload


def test_materializer_requires_training_report_policy_threshold_chain(tmp_path: Path):
    preregistration_path = tmp_path / "preregistration.json"
    preregistration = _preregistration(preregistration_path)
    dataset = tmp_path / "dataset.npz"
    dataset.write_bytes(b"dataset")
    dataset.chmod(0o444)
    dataset_manifest = tmp_path / "dataset.manifest.json"
    dataset_manifest.write_text("{}")
    dataset_manifest.chmod(0o444)
    binding = tmp_path / "binding.json"
    binding.write_text("{}")
    binding.chmod(0o444)
    policy = tmp_path / "policy.json"
    policy_payload = {
        "training_binding_sha256": sha256_file(binding),
        "preregistration_manifest_sha256": sha256_file(preregistration_path),
        "dataset_sha256": sha256_file(dataset),
        "dataset_manifest_sha256": sha256_file(dataset_manifest),
        "source_code_sha256": preregistration["code"]["trainer"]["sha256"],
        "quality25": {"threshold": 0.4},
        "benefit05": {"threshold": 0.5},
    }
    write_binding_create_only(policy, policy_payload)
    report_payload = {
        "schema": REPORT_SCHEMA,
        "complete": True,
        "train_only": True,
        "threshold_dev_gate_passed": True,
        "failure_action": None,
        "locked_internal_fold1_accessed": False,
        "validation_ground_truth_access": False,
        "validation_prediction_access": False,
        "formal_canonical103_authorized": False,
        "eligible_operating_point_count": 1,
        "chosen_operating_point": {
            "quality_threshold": 0.4,
            "benefit_threshold": 0.5,
            "gate": {"pass": True},
        },
        "policy": {"path": str(policy), "sha256": sha256_file(policy)},
        "dataset": {"path": str(dataset), "sha256": sha256_file(dataset)},
        "dataset_manifest_sha256": sha256_file(dataset_manifest),
        "training_binding_sha256": sha256_file(binding),
        "preregistration_manifest": {
            "path": str(preregistration_path),
            "sha256": sha256_file(preregistration_path),
        },
        "source_code_sha256": preregistration["code"]["trainer"]["sha256"],
    }
    report = tmp_path / "training_report.json"
    write_binding_create_only(report, report_payload)
    assert _validate_training_report(
        report,
        policy_path=policy,
        dataset_path=dataset,
        dataset_manifest_path=dataset_manifest,
        binding_path=binding,
        preregistration_record={
            "path": str(preregistration_path),
            "sha256": sha256_file(preregistration_path),
            "schema": PREREGISTRATION_SCHEMA,
        },
    )[0] == report.resolve()
    drift = dict(report_payload)
    drift["chosen_operating_point"] = dict(report_payload["chosen_operating_point"])
    drift["chosen_operating_point"]["quality_threshold"] = 0.6
    drift_report = tmp_path / "training_report_drift.json"
    write_binding_create_only(drift_report, drift)
    with pytest.raises(ValueError, match="report/policy chain"):
        _validate_training_report(
            drift_report,
            policy_path=policy,
            dataset_path=dataset,
            dataset_manifest_path=dataset_manifest,
            binding_path=binding,
            preregistration_record={
                "path": str(preregistration_path),
                "sha256": sha256_file(preregistration_path),
                "schema": PREREGISTRATION_SCHEMA,
            },
        )


def _policy(path: Path) -> None:
    head = {
        "mean": [0.0] * len(FEATURE_NAMES),
        "scale": [1.0] * len(FEATURE_NAMES),
        "weights": [0.0] * len(FEATURE_NAMES),
        "bias": 0.0,
        "threshold": 0.4,
    }
    payload = {
        "schema": POLICY_SCHEMA,
        "complete": True,
        "train_only": True,
        "validation_ground_truth_access": False,
        "validation_prediction_access": False,
        "official_validation_comparable": False,
        "formal_canonical103_authorized": False,
        "threshold_dev_gate_passed": True,
        "feature_schema": FEATURE_SCHEMA,
        "feature_names": list(FEATURE_NAMES),
        "quality_target": QUALITY_TARGET,
        "benefit_target": BENEFIT_TARGET,
        "selection_rule": SELECTION_RULE,
        "fit_fold_ids": list(GATE_TRAIN_FOLDS),
        "threshold_dev_fold_ids": list(THRESHOLD_DEV_FOLDS),
        "locked_internal_fold_ids": list(LOCKED_INTERNAL_FOLDS),
        "anchor_score_source": SCORE_SOURCE,
        "deploy_b6_scores_used_for_stacked_training": False,
        "training_binding_sha256": "1" * 64,
        "preregistration_manifest_sha256": "5" * 64,
        "dataset_sha256": "2" * 64,
        "dataset_manifest_sha256": "3" * 64,
        "quality25": head,
        "benefit05": head,
        "max_replacements_per_scene": MAX_REPLACEMENTS,
        "source_code_sha256": "4" * 64,
    }
    path.write_text(json.dumps(payload))
    path.chmod(0o444)


def test_v4_policy_selection_and_materialization_are_geometry_only(tmp_path: Path):
    policy_path = tmp_path / "policy.json"
    _policy(policy_path)
    policy = load_gate_policy_v4(
        policy_path, expected_training_binding_sha256="1" * 64
    )
    proposal, overlay, anchor_native, candidate_native, oof = _feature_inputs()
    batch = build_terminal_gate_features_v4(
        proposal=proposal, overlay=overlay,
        anchor_native_evidence=anchor_native,
        anchor_native_detector_scores=anchor_native[:, 0],
        candidate_native_evidence=candidate_native,
        anchor_scores=oof, score_source=SCORE_SOURCE,
    )
    selection = select_terminal_replacements_v4(batch, policy)
    assert selection.anchor_indices.tolist() == [0]
    assert selection.candidate_rows.tolist() == [0]
    result = materialize_geometry_only(
        anchor_corners=overlay["anchor_corners"],
        anchor_scores=oof,
        candidate_corners=proposal["candidate_corners_world"],
        anchor_indices=selection.anchor_indices,
        candidate_rows=selection.candidate_rows,
    )
    assert np.array_equal(result.scores, oof)
    assert np.array_equal(result.corners[0], proposal["candidate_corners_world"][0])
    assert np.array_equal(result.corners[1], overlay["anchor_corners"][1])


def test_v4_cli_surface_has_no_fold1_or_validation_input():
    root = Path(__file__).resolve().parents[1]
    dataset = (root / "tools/build_ca1m_tr3d_benefit_dataset_v4.py").read_text()
    trainer = (root / "tools/train_ca1m_tr3d_benefit_gate_v4.py").read_text()
    materializer = (root / "tools/materialize_ca1m_tr3d_terminal_active_v4.py").read_text()
    assert "--official-val" not in dataset
    assert "--partition" not in dataset
    assert "locked_internal_fold1_gt_access" in dataset
    assert "set(np.unique(folds).tolist()) != {0, 2, 3, 4}" in trainer
    assert '"formal_canonical103_authorized": False' in materializer
    assert "--training-report" in materializer
