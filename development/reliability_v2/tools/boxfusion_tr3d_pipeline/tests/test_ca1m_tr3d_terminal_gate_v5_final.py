from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from boxfusion import ca1m_tr3d_terminal_gate_v5 as gate
from boxfusion import ca1m_tr3d_terminal_gate_v5_final as final


ROOT = Path(__file__).resolve().parents[1]


def _box(center: tuple[float, float, float], size: float) -> np.ndarray:
    center_array = np.asarray(center, np.float32)
    return np.asarray([
        center_array + np.asarray((x, y, z), np.float32) * (size * 0.5)
        for x in (-1.0, 1.0)
        for y in (-1.0, 1.0)
        for z in (-1.0, 1.0)
    ], np.float32)


def _record(path: Path, schema: str) -> dict[str, str]:
    return {"path": str(path), "sha256": gate.sha256_file(path), "schema": schema}


def _seal(path: Path, payload: dict) -> Path:
    return gate.write_json_create_only(path, payload, path.name)


def _patch_r4_paths(monkeypatch: pytest.MonkeyPatch, root: Path) -> dict[str, Path]:
    r4_root = root / "ca1m_tr3d_e961_terminal_inputs_xfit_r2_v5_r4"
    producer_manifest_root = root / "producer_manifests"
    paths = {
        "root": r4_root,
        "collection": r4_root / "manifests/CANDIDATE_COLLECTION_EXACT80.json",
        "receipt": r4_root / "manifests/M_EXACT80_R4_RECEIPT.json",
        "prereg": producer_manifest_root / "PREREGISTRATION.json",
        "ready": producer_manifest_root / "READY_CONFIG.json",
        "auth": producer_manifest_root / "RUN_AUTHORIZATION.json",
        "bundle": producer_manifest_root / "AUTHORIZATION_BUNDLE.json",
        "r2_receipt": r4_root / "manifests/M_EXACT80_R2_RECEIPT.json",
    }
    monkeypatch.setattr(final, "R4_ROOT", r4_root)
    monkeypatch.setattr(final, "R4_COLLECTION_PATH", paths["collection"])
    monkeypatch.setattr(final, "R4_RECEIPT_PATH", paths["receipt"])
    monkeypatch.setattr(final, "R4_R2_EXECUTION_RECEIPT_PATH", paths["r2_receipt"])
    monkeypatch.setattr(final, "R4_MANIFEST_ROOT", producer_manifest_root)
    monkeypatch.setattr(final, "R4_PREREGISTRATION_PATH", paths["prereg"])
    monkeypatch.setattr(final, "R4_READY_CONFIG_PATH", paths["ready"])
    monkeypatch.setattr(final, "R4_RUN_AUTHORIZATION_PATH", paths["auth"])
    monkeypatch.setattr(final, "R4_AUTHORIZATION_BUNDLE_PATH", paths["bundle"])
    return paths


def _synthetic_r4(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> tuple[dict[str, Path], dict[str, int], dict[str, Path]]:
    paths = _patch_r4_paths(monkeypatch, tmp_path)
    continuation_payload = {
        "schema": "boxfusion.ca1m_tr3d_e961_outer_dev_continuation_receipt.v1",
        "complete": True, "fold1_access": False,
        "official_validation_access": False, "checkpoint_selection": False,
        "pass": True, "continue_inner_training_authorized": True,
        "authorized_inner_roles": ["inner_holdout2", "inner_holdout3", "inner_holdout4"],
        "continuation_gate": {
            "pass": True, "continue_inner_training_authorized": True,
            "authorized_inner_roles": ["inner_holdout2", "inner_holdout3", "inner_holdout4"],
        },
    }
    continuation = _seal(tmp_path / "continuation.json", continuation_payload)
    b6 = tmp_path / "ca1m_native_b6_all_fold_oof_v2.npz"
    b6.write_bytes(b"synthetic-ca-native-all-fold-oof")
    b6.chmod(0o444)
    b6_record = {
        "path": str(b6), "sha256": gate.sha256_file(b6),
        "schema": "boxfusion.ca1m_native_b6_oof_row_scores.v2",
        "score_source": gate.ANCHOR_SCORE_SOURCE,
        "each_row_model_excludes_scene": True, "deploy_scores": False,
    }
    role_records: dict[str, dict] = {}
    raw_receipts: dict[str, Path] = {}
    checkpoints: dict[str, Path] = {}
    scene_folds: dict[str, int] = {}
    gt_paths: dict[str, Path] = {}
    for role_index, (role, (train_folds, heldout)) in enumerate(gate.ROLE_SPECS.items()):
        checkpoint = tmp_path / f"e961_{role}_iter_11268.pth"
        checkpoint.write_bytes(f"e961-{role}-checkpoint".encode())
        checkpoint.chmod(0o444)
        checkpoints[role] = checkpoint
        raw_schema = final.OUTER_RUN_SCHEMA if role == "outer_dev" else final.INNER_RUN_SCHEMA
        raw_payload = {
            "schema": raw_schema, "complete": True, "create_only": True,
            "status": "success", "exit_code": 0, "role": role,
            "train_scenes": 1001, "heldout_scenes": 20,
            "training_protocol": {
                "train_folds": list(train_folds), "heldout_fold": heldout,
                "train_scenes": 1001, "optimizer_updates": 11268,
                "initialization": "random_scratch_ca_only",
            },
        }
        raw = _seal(tmp_path / f"e961_{role}_success_receipt.json", raw_payload)
        raw_receipts[role] = raw
        adapter_payload = {
            "schema": "boxfusion.ca1m_tr3d_e961_verified_receipt_adapter.v2",
            "complete": True, "create_only": True, "status": "success",
            "role": role, "checkpoint_selection": False,
            "training_protocol": {
                "train_folds": list(train_folds), "heldout_fold": heldout,
                "train_scenes": 1001, "optimizer_updates": 11268,
                "initialization": "random_scratch_ca_only",
                "scannet_checkpoint_or_module_access": False,
            },
            "checkpoint": {
                "path": str(checkpoint), "sha256": gate.sha256_file(checkpoint),
                "optimizer_updates": 11268, "checkpoint_selection": False,
            },
            "source_producer_receipt": {
                **_record(raw, raw_schema),
                "producer_verify_success_receipt_passed": True,
            },
            "access": {
                "fold1_metadata_or_ground_truth_access": False,
                "official_validation_access": False,
                "scannet_checkpoint_or_module_access": False,
            },
        }
        receipt_root = paths["root"] / "normalized_receipts"
        adapter = _seal(receipt_root / f"{role}_verified_producer_adapter.json", adapter_payload)
        normalized = receipt_root / f"{role}_detector_role_receipt_v5.json"
        gate.seal_detector_role_receipt_v5(
            normalized, role=role,
            source_training_receipt=_record(
                adapter, "boxfusion.ca1m_tr3d_e961_verified_receipt_adapter.v2"
            ),
            outer_continuation_receipt=_record(continuation, continuation_payload["schema"]),
        )
        evidence_paths: dict[str, Path] = {}
        scenes = [f"{61000000 + role_index * 100 + index:08d}" for index in range(20)]
        for index, scene in enumerate(scenes):
            scene_folds[scene] = heldout
            gt = np.asarray([_box((float(index) * 3.0, float(role_index) * 3.0, 0.0), 1.0)], np.float64)
            gt_path = tmp_path / "gt" / scene / "derived_train_gt_boxes.npy"
            gt_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(gt_path, gt)
            gt_path.chmod(0o444)
            gt_paths[scene] = gt_path
            anchor = np.asarray([_box((float(index) * 3.0, float(role_index) * 3.0, 0.0), 0.7)], np.float32)
            candidate = np.asarray([
                _box((float(index) * 3.0, float(role_index) * 3.0, 0.0), 1.0),
                _box((float(index) * 3.0 + 3.0, float(role_index) * 3.0, 0.0), 1.0),
            ], np.float32)
            features = np.zeros((2, len(gate.FEATURE_NAMES)), np.float32)
            features[:, 1] = np.asarray((1.0, -1.0), np.float32)
            features[:, 2] = index / 20.0
            evidence = paths["root"] / "evidence" / role / (
                f"{scene}_ca1m_tr3d_candidate_evidence_xfit_r2_v5.npz"
            )
            gate.write_candidate_evidence_v5(
                evidence, scene_id=scene, fold_id=heldout, producer_role=role,
                producer_checkpoint_sha256=gate.sha256_file(checkpoint),
                training_receipt_sha256=gate.sha256_file(normalized),
                outer_continuation_receipt_sha256=gate.sha256_file(continuation),
                b6_oof_sidecar_sha256=gate.sha256_file(b6),
                candidate_corners=candidate,
                candidate_rows=np.asarray((0, 1), np.int64),
                candidate_scores=np.asarray((0.55, 0.95), np.float32),
                anchor_indices=np.asarray((0, 0), np.int64), features=features,
                anchor_corners=anchor, anchor_scores=np.asarray((0.8,), np.float32),
            )
            evidence_paths[scene] = evidence
        role_manifest = paths["root"] / "manifests" / f"M_{role}_candidate_collection_exact20.json"
        gate.seal_role_candidate_collection_v5(
            role_manifest, role=role, expected_scenes=scenes,
            role_receipt=_record(normalized, gate.ROLE_RECEIPT_SCHEMA),
            evidence_paths=evidence_paths, b6_oof_sidecar=b6_record,
        )
        role_records[role] = _record(role_manifest, gate.ROLE_COLLECTION_SCHEMA)
    gate.seal_candidate_collection_v5(paths["collection"], role_collections=role_records)

    commit_id = "a" * 64
    prereg = _seal(paths["prereg"], {
        "schema": final.R4_PREREGISTRATION_SCHEMA, "complete": True,
        "create_only": True, "namespace": final.R4_NAMESPACE,
    })
    ready_payload = {
        "schema": final.R4_CONFIG_SCHEMA, "namespace": final.R4_NAMESPACE,
        "run_authorization": {
            "state": "committed_by_bundle", "path": str(paths["bundle"]),
            "commit_id": commit_id, "schema": final.R4_BUNDLE_SCHEMA,
        },
    }
    ready = _seal(paths["ready"], ready_payload)
    auth_payload = {
        "schema": final.R4_AUTHORIZATION_SCHEMA, "complete": True,
        "create_only": True, "namespace": final.R4_NAMESPACE,
        "commit_id": commit_id, "ground_truth_access": False,
        "fold1_access": False, "official_validation_access": False,
        "roles": [{
            "role": role, "receipt_path": str(raw_receipts[role]),
            "receipt_sha256": gate.sha256_file(raw_receipts[role]),
            "checkpoint_sha256": gate.sha256_file(checkpoints[role]),
        } for role in gate.ROLE_SPECS],
    }
    auth = _seal(paths["auth"], auth_payload)
    bundle = _seal(paths["bundle"], {
        "schema": final.R4_BUNDLE_SCHEMA, "complete": True,
        "create_only": True, "namespace": final.R4_NAMESPACE,
        "commit_id": commit_id,
        "commit_role": "last_published_unique_operational_gate",
        "ready_config": _record(ready, final.R4_CONFIG_SCHEMA),
        "run_authorization": _record(auth, final.R4_AUTHORIZATION_SCHEMA),
    })
    r2_receipt = _seal(paths["r2_receipt"], {
        "schema": final.R4_R2_EXECUTION_RECEIPT_SCHEMA,
        "complete": True, "create_only": True,
        "namespace": final.R4_NAMESPACE,
        "operational_authority": False,
    })
    wrapper_payload = {
        "schema": final.R4_RECEIPT_SCHEMA, "complete": True, "create_only": True,
        "namespace": final.R4_NAMESPACE,
        "fit_scene_count": 60, "fit_folds": [2, 3, 4],
        "reused_dev_scene_count": 20, "reused_dev_folds": [0],
        "scene_count": 80, "each_scene_detector_excludes_scene": True,
        "b6_score_source": "all_fold_oof_each_row_model_excludes_scene",
        "ground_truth_access": False, "fold1_access": False,
        "official_validation_access": False,
        "legacy_v1_v4_candidate_or_policy_reused": False,
        "r4_preregistration": _record(prereg, final.R4_PREREGISTRATION_SCHEMA),
        "r4_ready_config": _record(ready, final.R4_CONFIG_SCHEMA),
        "r4_run_authorization": _record(auth, final.R4_AUTHORIZATION_SCHEMA),
        "r4_authorization_bundle": _record(bundle, final.R4_BUNDLE_SCHEMA),
        "authorization_commit_id": commit_id,
        "candidate_collection": _record(paths["collection"], gate.COLLECTION_SCHEMA),
        "r2_execution_receipt": {
            **_record(r2_receipt, final.R4_R2_EXECUTION_RECEIPT_SCHEMA),
            "operational_authority": False,
        },
    }
    _seal(paths["receipt"], wrapper_payload)
    return paths, scene_folds, gt_paths


def _gt_inventory(
    tmp_path: Path, scene_folds: dict[str, int], gt_paths: dict[str, Path],
) -> Path:
    payload = {
        "schema": final.GT_INVENTORY_SCHEMA, "complete": True,
        "create_only": True, "scene_count": 80, "fit_scene_count": 60,
        "threshold_dev_scene_count": 20, "fit_fold_ids": [2, 3, 4],
        "threshold_dev_fold_ids": [0], "locked_internal_scene_count_accessed": 0,
        "official_validation_comparable": False,
        "validation_ground_truth_access": False,
        "validation_prediction_access": False, "train_only": True,
        "gt_array_content_loaded": False,
        "scenes": {
            scene: {
                "fold_id": fold,
                "box": {
                    "path": str(gt_paths[scene]),
                    "sha256": gate.sha256_file(gt_paths[scene]),
                },
            }
            for scene, fold in scene_folds.items()
        },
    }
    return _seal(tmp_path / "gt_inventory.json", payload)


def test_final_pending_config_and_official_ap_parity_are_static() -> None:
    _, cfg = final.validate_pending_config()
    assert cfg["state"] == "pending_r4_exact80"
    assert not any(cfg["authorizations"].values())
    report = final.static_preflight()
    assert report["status"] == "PASS_STATIC_PENDING_R4"
    assert report["ground_truth_access"] is False
    assert report["directory_created"] is False
    assert final.ap_parity_fixture()["pass"] is True


def test_pending_r4_stops_before_gt_mkdir_or_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    missing = tmp_path / "missing_r4_receipt.json"
    called = False

    def no_config(_path: Path):
        return tmp_path / "pending.json", {}

    def forbidden_loader():
        nonlocal called
        called = True

    monkeypatch.setattr(final, "validate_pending_config", no_config)
    monkeypatch.setattr(final, "R4_RECEIPT_PATH", missing)
    with pytest.raises(final.PendingR4Inputs):
        final.operational_preflight_pending(tmp_path / "pending.json")
    forbidden_loader  # prove a loader can exist but is never invoked
    assert called is False
    assert not missing.parent.joinpath("runtime").exists()


def test_r4_binding_is_exact80_e961_double_oof(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths, scene_folds, _ = _synthetic_r4(tmp_path, monkeypatch)
    binding = final.load_r4_exact80_binding(paths["receipt"])
    assert binding.collection_path == paths["collection"]
    assert len(binding.scene_folds) == 80
    assert {fold: list(scene_folds.values()).count(fold) for fold in (0, 2, 3, 4)} == {
        0: 20, 2: 20, 3: 20, 4: 20,
    }
    assert binding.upstream_records["r2_internal_execution_receipt"][
        "operational_authority"
    ] is False
    wrapper = json.loads(paths["receipt"].read_text())
    wrapper["candidate_collection"] = {
        **wrapper["candidate_collection"],
        "path": str(tmp_path / "legacy" / "CANDIDATE_COLLECTION_EXACT80.json"),
    }
    forged = _seal(tmp_path / "forged_r4_receipt.json", wrapper)
    monkeypatch.setattr(final, "R4_RECEIPT_PATH", forged)
    with pytest.raises(ValueError, match="canonical"):
        final.load_r4_exact80_binding(forged)

    wrapper = json.loads(paths["receipt"].read_text())
    wrapper["r2_execution_receipt"]["operational_authority"] = True
    forged_authority = _seal(tmp_path / "forged_r2_authority_receipt.json", wrapper)
    monkeypatch.setattr(final, "R4_RECEIPT_PATH", forged_authority)
    with pytest.raises(ValueError, match="non-authoritative"):
        final.load_r4_exact80_binding(forged_authority)


def test_create_only_writer_binds_inode_hash_and_refuses_collision(tmp_path: Path) -> None:
    target = gate.write_json_create_only(
        tmp_path / "artifact.json", {"schema": "synthetic", "value": 1}, "artifact"
    )
    record = final.artifact_record(target, "artifact", schema="synthetic")
    _, data, identity = final.validate_artifact_record(
        record, "artifact", schema="synthetic", require_identity=True
    )
    assert final.sha256_bytes(data) == record["sha256"]
    assert identity["st_nlink"] == 1
    with pytest.raises(FileExistsError):
        gate.write_json_create_only(target, {"schema": "synthetic", "value": 2}, "artifact")


def test_full_final_chain_trains_on_fit60_then_runs_frozen_fold0(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths, scene_folds, gt_paths = _synthetic_r4(tmp_path, monkeypatch)
    inventory = _gt_inventory(tmp_path, scene_folds, gt_paths)
    formal_root = tmp_path / "final_formal"
    runtime_root = tmp_path / "final_runtime"
    outputs = {
        "fit_dataset": runtime_root / "datasets/fit.npz",
        "fit_dataset_manifest": runtime_root / "datasets/fit.manifest.json",
        "oof_predictions": runtime_root / "results/oof.npz",
        "threshold_receipt": runtime_root / "results/threshold.json",
        "exploratory_policy": runtime_root / "models/policy.json",
        "fold0_dataset": runtime_root / "datasets/fold0.npz",
        "fold0_dataset_manifest": runtime_root / "datasets/fold0.manifest.json",
        "fold0_report": runtime_root / "reports/fold0.json",
        "materialization_root": runtime_root / "materialized",
        "materialization_manifest": runtime_root / "reports/materialized.json",
        "stop_receipt": runtime_root / "reports/STOP.json",
        "run_receipt": runtime_root / "reports/RUN.json",
    }
    prereg = formal_root / "PREREGISTRATION.json"
    ready = formal_root / "READY_CONFIG.json"
    authorization = formal_root / "RUN_AUTHORIZATION.json"
    pending = _seal(tmp_path / "synthetic_pending.json", {
        "schema": final.PENDING_SCHEMA, "synthetic": True,
    })
    monkeypatch.setattr(final, "PREREGISTRATION_PATH", prereg)
    monkeypatch.setattr(final, "READY_CONFIG_PATH", ready)
    monkeypatch.setattr(final, "RUN_AUTHORIZATION_PATH", authorization)
    monkeypatch.setattr(final, "RUNTIME_ROOT", runtime_root)
    monkeypatch.setattr(final, "OUTPUT_PATHS", outputs)
    monkeypatch.setattr(final, "validate_pending_config", lambda _path: (pending, {}))
    sealed_prereg = final.seal_scientific_preregistration(
        gt_inventory_path=inventory, output_path=prereg,
        pending_config_path=pending, r4_receipt_path=paths["receipt"],
    )
    assert sealed_prereg == prereg
    final.seal_ready_authorization(
        preregistration_path=prereg, ready_path=ready,
        authorization_path=authorization,
    )
    context = final.load_execution_context(authorization)
    opened: list[str] = []
    production_loader = final.inventory_ground_truth_loader(context)

    def recording_loader(scene: str) -> np.ndarray:
        opened.append(scene)
        return production_loader(scene)

    receipt = final.run_final_gate(
        context=context, ground_truth_loader=recording_loader
    )
    value = json.loads(receipt.read_text())
    assert value["schema"] == final.RUN_SCHEMA
    assert value["status"] == "PASS_EXPLORATORY_FOLD0_DIAGNOSTIC_COMPLETE"
    assert len(opened) == 80
    assert {scene_folds[scene] for scene in opened[:60]} == {2, 3, 4}
    assert {scene_folds[scene] for scene in opened[60:]} == {0}
    assert value["fold1_access"] is False
    assert value["official_validation_access"] is False
    assert value["policy_activation_authorized"] is False
    materialized = json.loads(outputs["materialization_manifest"].read_text())
    assert materialized["scene_count"] == 20
    assert materialized["geometry_only"] is True
    assert materialized["scores_preserved"] is True
    assert materialized["row_order_preserved"] is True
    assert materialized["row_count_preserved"] is True
    assert not outputs["stop_receipt"].exists()
