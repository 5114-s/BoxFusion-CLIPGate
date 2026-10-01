from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from boxfusion.ca1m_tr3d_terminal_gate_v4 import (
    B6_CHECKPOINT_MANIFEST_SCHEMA,
    B6_CHECKPOINT_SCHEMA,
    B6_OOF_MANIFEST_SCHEMA,
    B6_OOF_SCHEMA,
    CONFIG_SCHEMA,
    load_oof_row_scores,
    materialize_geometry_only,
    validate_ready,
    validate_static_config,
    write_binding_create_only,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config/ca1m_tr3d_benefit_gate_train100_v4.json"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_checked_in_gate_v4_contract_is_static_but_operational_modes_fail_closed():
    _, cfg = validate_static_config(CONFIG)
    assert cfg["schema"] == CONFIG_SCHEMA
    assert cfg["run_authorized"] is False
    assert cfg["split"]["gate_train_folds"] == [2, 3, 4]
    assert cfg["split"]["threshold_dev_folds"] == [0]
    assert cfg["split"]["locked_internal_check_folds"] == [1]
    assert cfg["split"]["deploy_b6_scores_allowed_for_stacked_training"] is False
    target = Path(cfg["outputs"]["binding_manifest"])
    before = target.exists()
    with pytest.raises(PermissionError, match="still pending"):
        validate_ready(CONFIG)
    assert target.exists() is before


def test_static_contract_rejects_legacy_paths_and_split_drift(tmp_path: Path):
    cfg = json.loads(CONFIG.read_text())
    legacy = copy.deepcopy(cfg)
    legacy["prerequisites"]["candidate_evidence_manifest"] = {
        "path": "/tmp/ca1m_tr3d_benefit_gate_v1/manifest.json",
        "sha256": "1" * 64,
        "schema": "x",
    }
    legacy_path = tmp_path / "legacy.json"
    legacy_path.write_text(json.dumps(legacy))
    with pytest.raises(ValueError, match="forbidden legacy artifact"):
        validate_static_config(legacy_path)

    drift = copy.deepcopy(cfg)
    drift["split"]["gate_train_folds"] = [1, 2, 3]
    drift_path = tmp_path / "drift.json"
    drift_path.write_text(json.dumps(drift))
    with pytest.raises(ValueError, match="split/OOF"):
        validate_static_config(drift_path)


def _oof_fixture(tmp_path: Path):
    checkpoint = tmp_path / "b6-v2.npz"
    checkpoint.write_bytes(b"checkpoint")
    checkpoint.chmod(0o444)
    checkpoint_manifest = tmp_path / "b6-v2.manifest.json"
    checkpoint_manifest.write_text("{}")
    checkpoint_manifest.chmod(0o444)

    scenes = np.asarray([f"{42_000_000 + index:08d}" for index in range(100)])
    folds = np.asarray([index % 5 for index in range(100)], dtype=np.int8)
    recipe = {
        "schema": "boxfusion.ca1m_native_b6_oof_recipe.v2",
        "heldout_rule": "model_fold_k_trained_on_all_scene_folds_except_k",
    }
    recipe_json = json.dumps(recipe, separators=(",", ":"), sort_keys=True)
    recipe_sha = hashlib.sha256(recipe_json.encode()).hexdigest()
    fold_hashes = [str(index) * 64 for index in range(1, 6)]
    arrays = {
        "schema": np.asarray(B6_OOF_SCHEMA),
        "complete": np.asarray(True, np.bool_),
        "train_only": np.asarray(True, np.bool_),
        "scene_group_oof": np.asarray(True, np.bool_),
        "validation_ground_truth_access": np.asarray(False, np.bool_),
        "validation_prediction_access": np.asarray(False, np.bool_),
        "official_validation_comparable": np.asarray(False, np.bool_),
        "each_row_model_excludes_scene": np.asarray(True, np.bool_),
        "fold_count": np.asarray(5, np.int8),
        "dataset_sha256": np.asarray("6" * 64),
        "dataset_manifest_sha256": np.asarray("7" * 64),
        "split_namespace": np.asarray("boxfusion.ca1m-native-b6.scene-folds.v1"),
        "feature_names": np.asarray(("detector_score",)),
        "scene_ids": scenes,
        "fold_ids": folds,
        "heldout_model_fold_ids": folds.copy(),
        "source_row_indices": np.zeros(100, np.int64),
        "dataset_row_positions": np.arange(100, dtype=np.int64),
        "detector_scores": np.full(100, 0.4, np.float32),
        "raw_oof_outputs": np.full((100, 4), 0.5, np.float32),
        "monotonic_oof_components": np.full((100, 4), 0.5, np.float32),
        "quality_oof_scores": np.full(100, 0.5, np.float32),
        "deployment_blend_oof_scores": np.full(100, 0.46, np.float32),
        "fold_model_sha256": np.asarray(fold_hashes),
        "recipe_json": np.asarray(recipe_json),
        "recipe_sha256": np.asarray(recipe_sha),
    }
    sidecar = tmp_path / "oof-v2.npz"
    np.savez_compressed(sidecar, **arrays)
    sidecar.chmod(0o444)
    folds_manifest = []
    for fold in range(5):
        heldout = scenes[folds == fold].tolist()
        training = scenes[folds != fold].tolist()
        folds_manifest.append({
            "heldout_fold": fold,
            "model_sha256": fold_hashes[fold],
            "heldout_scene_ids": heldout,
            "training_scene_ids": training,
            "training_excludes_every_heldout_scene": True,
        })
    manifest = {
        "schema": B6_OOF_MANIFEST_SCHEMA,
        "complete": True,
        "train_only": True,
        "scene_group_oof": True,
        "validation_ground_truth_access": False,
        "validation_prediction_access": False,
        "official_validation_comparable": False,
        "each_row_model_excludes_scene": True,
        "row_count": 100,
        "scene_count": 100,
        "dataset": {"sha256": "6" * 64, "manifest_sha256": "7" * 64},
        "artifact": {"path": str(sidecar), "sha256": _sha(sidecar), "schema": B6_OOF_SCHEMA},
        "deployment_checkpoint": {
            "path": str(checkpoint), "sha256": _sha(checkpoint),
            "schema": B6_CHECKPOINT_SCHEMA,
        },
        "checkpoint_manifest": {
            "path": str(checkpoint_manifest),
            "schema": B6_CHECKPOINT_MANIFEST_SCHEMA,
            "binds_this_sidecar": True,
        },
        "split": {
            "namespace": "boxfusion.ca1m-native-b6.scene-folds.v1",
            "fold_count": 5,
            "all_fold_oof": True,
            "gate_train_folds": [2, 3, 4],
            "threshold_dev_folds": [0],
            "locked_internal_check_folds": [1],
            "folds": folds_manifest,
        },
        "fold_model_sha256": fold_hashes,
        "recipe": recipe,
        "recipe_sha256": recipe_sha,
    }
    manifest_path = tmp_path / "oof-v2.manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    manifest_path.chmod(0o444)
    return (
        checkpoint,
        checkpoint_manifest,
        {"path": str(sidecar), "sha256": _sha(sidecar), "schema": B6_OOF_SCHEMA},
        {
            "path": str(manifest_path), "sha256": _sha(manifest_path),
            "schema": B6_OOF_MANIFEST_SCHEMA,
        },
        sidecar,
    )


def test_oof_loader_accepts_exact_60_20_20_cross_fit_and_rejects_fold_leakage(tmp_path: Path):
    checkpoint, checkpoint_manifest, sidecar_record, manifest_record, sidecar = _oof_fixture(tmp_path)
    values, manifest = load_oof_row_scores(
        sidecar_record, manifest_record,
        checkpoint=checkpoint, checkpoint_manifest=checkpoint_manifest,
    )
    assert len(values["scene_ids"]) == 100
    assert manifest["each_row_model_excludes_scene"] is True

    sidecar.chmod(0o644)
    with np.load(sidecar, allow_pickle=False) as archive:
        changed = {name: np.array(archive[name], copy=True) for name in archive.files}
    changed["heldout_model_fold_ids"][0] = 1
    np.savez_compressed(sidecar, **changed)
    sidecar.chmod(0o444)
    sidecar_record["sha256"] = _sha(sidecar)
    manifest_path = Path(manifest_record["path"])
    manifest_path.chmod(0o644)
    payload = json.loads(manifest_path.read_text())
    payload["artifact"]["sha256"] = _sha(sidecar)
    manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    manifest_path.chmod(0o444)
    manifest_record["sha256"] = _sha(manifest_path)
    with pytest.raises(ValueError, match="row identity"):
        load_oof_row_scores(
            sidecar_record, manifest_record,
            checkpoint=checkpoint, checkpoint_manifest=checkpoint_manifest,
        )


def test_binding_writer_is_create_only(tmp_path: Path):
    target = tmp_path / "binding.json"
    write_binding_create_only(target, {"schema": "x", "complete": True})
    original = target.read_bytes()
    assert target.stat().st_mode & 0o222 == 0
    with pytest.raises(FileExistsError):
        write_binding_create_only(target, {"schema": "changed"})
    assert target.read_bytes() == original


def test_v4_materialization_changes_only_selected_geometry():
    anchors = np.arange(2 * 8 * 3, dtype=np.float32).reshape(2, 8, 3)
    candidates = np.full((2, 8, 3), 77.0, dtype=np.float32)
    scores = np.asarray((0.8, 0.3), dtype=np.float32)
    frozen_anchors = anchors.copy()
    frozen_scores = scores.copy()
    result = materialize_geometry_only(
        anchor_corners=anchors,
        anchor_scores=scores,
        candidate_corners=candidates,
        anchor_indices=np.asarray((1,), dtype=np.int64),
        candidate_rows=np.asarray((0,), dtype=np.int64),
    )
    assert np.array_equal(result.corners[0], anchors[0])
    assert np.array_equal(result.corners[1], candidates[0])
    assert np.array_equal(result.scores, scores)
    assert not result.corners.flags.writeable
    assert not result.scores.flags.writeable
    assert np.array_equal(anchors, frozen_anchors)
    assert np.array_equal(scores, frozen_scores)

    with pytest.raises(ValueError, match="selection is invalid"):
        materialize_geometry_only(
            anchor_corners=anchors, anchor_scores=scores,
            candidate_corners=candidates,
            anchor_indices=np.asarray((1, 1), dtype=np.int64),
            candidate_rows=np.asarray((0, 1), dtype=np.int64),
        )
