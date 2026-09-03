from __future__ import annotations

import json
import fcntl
import inspect
import os
from dataclasses import replace
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from boxfusion import ca1m_tr3d_terminal_gate_v5 as gate
from boxfusion import ca1m_tr3d_terminal_gate_v5_final_r4 as final


ROOT = Path(__file__).resolve().parents[1]
PREFLIGHT = ROOT / "tools/preflight_ca1m_tr3d_terminal_gate_v5_final_r4.py"
RUNNER = ROOT / "tools/run_ca1m_tr3d_terminal_gate_v5_final_r4.py"
LEGACY_RUNNER = ROOT / "tools/run_ca1m_tr3d_terminal_gate_v5_final.py"
R2_RUNNER = ROOT / "tools/run_ca1m_tr3d_terminal_gate_v5_final_r2.py"
R3_RUNNER = ROOT / "tools/run_ca1m_tr3d_terminal_gate_v5_final_r3.py"


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
    path.parent.mkdir(parents=True, exist_ok=True)
    return final._exclusive_bytes_fuse(
        path, gate._canonical_json(payload), path.name,
    )


def _patch_r6_paths(monkeypatch: pytest.MonkeyPatch, root: Path) -> dict[str, Path]:
    r6_root = root / "ca1m_tr3d_e961_terminal_inputs_xfit_r2_v5_r6"
    producer_manifest_root = root / "producer_manifests"
    paths = {
        "root": r6_root,
        "collection": r6_root / "manifests/CANDIDATE_COLLECTION_EXACT80.json",
        "receipt": r6_root / "manifests/M_EXACT80_R6_RECEIPT.json",
        "prereg": producer_manifest_root / "PREREGISTRATION.json",
        "ready": producer_manifest_root / "READY_CONFIG.json",
        "auth": producer_manifest_root / "RUN_AUTHORIZATION.json",
        "bundle": producer_manifest_root / "AUTHORIZATION_BUNDLE.json",
        "r2_receipt": r6_root / "manifests/M_EXACT80_R2_RECEIPT.json",
        "config": producer_manifest_root / "R6_PENDING_CONFIG.json",
        "core": producer_manifest_root / "r6_core.py",
    }
    monkeypatch.setattr(final, "R6_ROOT", r6_root)
    monkeypatch.setattr(final, "R6_COLLECTION_PATH", paths["collection"])
    monkeypatch.setattr(final, "R6_RECEIPT_PATH", paths["receipt"])
    monkeypatch.setattr(final, "R6_R2_EXECUTION_RECEIPT_PATH", paths["r2_receipt"])
    monkeypatch.setattr(final, "R6_MANIFEST_ROOT", producer_manifest_root)
    monkeypatch.setattr(final, "R6_PREREGISTRATION_PATH", paths["prereg"])
    monkeypatch.setattr(final, "R6_READY_CONFIG_PATH", paths["ready"])
    monkeypatch.setattr(final, "R6_RUN_AUTHORIZATION_PATH", paths["auth"])
    monkeypatch.setattr(final, "R6_AUTHORIZATION_BUNDLE_PATH", paths["bundle"])
    monkeypatch.setattr(final, "R6_CONFIG_PATH", paths["config"])
    monkeypatch.setattr(final, "R6_CORE_PATH", paths["core"])
    return paths


def _synthetic_r6(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> tuple[dict[str, Path], dict[str, int], dict[str, Path]]:
    """Scope the Python-3.8 fixture writer to construction only."""

    def fixture_write(path: Path, payload: bytes, name: str) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        return final._exclusive_bytes_fuse(path, payload, name)

    with monkeypatch.context() as scoped:
        scoped.setattr(gate, "write_bytes_create_only", fixture_write)
        return _synthetic_r6_impl(tmp_path, monkeypatch)


def _synthetic_r6_impl(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> tuple[dict[str, Path], dict[str, int], dict[str, Path]]:
    paths = _patch_r6_paths(monkeypatch, tmp_path)
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
    r6_config = _seal(paths["config"], {
        "schema": final.R6_CONFIG_SCHEMA, "namespace": final.R6_NAMESPACE,
    })
    paths["core"].parent.mkdir(parents=True, exist_ok=True)
    paths["core"].write_text("R6_SYNTHETIC_CORE = True\n")
    paths["core"].chmod(0o444)
    prereg = _seal(paths["prereg"], {
        "schema": final.R6_PREREGISTRATION_SCHEMA, "complete": True,
        "create_only": True, "static_only": True, "namespace": final.R6_NAMESPACE,
        "pending_config": _record(r6_config, final.R6_CONFIG_SCHEMA),
        "implementation": {
            "current_core": {
                "path": str(paths["core"]),
                "sha256": gate.sha256_file(paths["core"]),
            },
        },
    })
    monkeypatch.setattr(final, "R6_PREREGISTRATION_SHA256", gate.sha256_file(prereg))
    monkeypatch.setattr(final, "R6_CONFIG_SHA256", gate.sha256_file(r6_config))
    monkeypatch.setattr(final, "R6_CORE_SHA256", gate.sha256_file(paths["core"]))
    ready_payload = {
        "schema": final.R6_CONFIG_SCHEMA, "namespace": final.R6_NAMESPACE,
        "preregistration": {
            "path": str(prereg), "schema": final.R6_PREREGISTRATION_SCHEMA,
        },
        "implementation": {
            "current_core": {
                "path": str(paths["core"]),
                "sha256": gate.sha256_file(paths["core"]),
            },
        },
        "outputs": {"namespace_root": str(paths["root"])},
        "run_authorization": {
            "state": "committed_by_bundle", "path": str(paths["bundle"]),
            "commit_id": commit_id, "schema": final.R6_BUNDLE_SCHEMA,
        },
    }
    ready = _seal(paths["ready"], ready_payload)
    auth_payload = {
        "schema": final.R6_AUTHORIZATION_SCHEMA, "complete": True,
        "create_only": True, "namespace": final.R6_NAMESPACE,
        "commit_id": commit_id, "ground_truth_access": False,
        "fold1_access": False, "official_validation_access": False,
        "formal_gpu_run_started": False,
        "pending_config_sha256": gate.sha256_file(r6_config),
        "preregistration": {
            "path": str(prereg), "sha256": gate.sha256_file(prereg),
        },
        "roles": [{
            "role": role, "receipt_path": str(raw_receipts[role]),
            "receipt_sha256": gate.sha256_file(raw_receipts[role]),
            "checkpoint_sha256": gate.sha256_file(checkpoints[role]),
        } for role in gate.ROLE_SPECS],
    }
    auth = _seal(paths["auth"], auth_payload)
    bundle = _seal(paths["bundle"], {
        "schema": final.R6_BUNDLE_SCHEMA, "complete": True,
        "create_only": True, "namespace": final.R6_NAMESPACE,
        "commit_id": commit_id,
        "commit_role": "last_published_unique_operational_gate",
        "ready_config": _record(ready, final.R6_CONFIG_SCHEMA),
        "run_authorization": _record(auth, final.R6_AUTHORIZATION_SCHEMA),
        "ground_truth_access": False, "gpu_started": False,
    })
    r2_receipt = _seal(paths["r2_receipt"], {
        "schema": final.R6_R2_EXECUTION_RECEIPT_SCHEMA,
        "complete": True, "create_only": True,
        "namespace": final.R6_NAMESPACE,
        "operational_authority": False,
    })
    wrapper_payload = {
        "schema": final.R6_RECEIPT_SCHEMA, "complete": True, "create_only": True,
        "namespace": final.R6_NAMESPACE,
        "fit_scene_count": 60, "fit_folds": [2, 3, 4],
        "reused_dev_scene_count": 20, "reused_dev_folds": [0],
        "scene_count": 80, "each_scene_detector_excludes_scene": True,
        "b6_score_source": "all_fold_oof_each_row_model_excludes_scene",
        "ground_truth_access": False, "fold1_access": False,
        "official_validation_access": False,
        "legacy_v1_v4_candidate_or_policy_reused": False,
        "r6_preregistration": _record(prereg, final.R6_PREREGISTRATION_SCHEMA),
        "r6_ready_config": _record(ready, final.R6_CONFIG_SCHEMA),
        "r6_run_authorization": _record(auth, final.R6_AUTHORIZATION_SCHEMA),
        "r6_authorization_bundle": _record(bundle, final.R6_BUNDLE_SCHEMA),
        "authorization_commit_id": commit_id,
        "candidate_collection": _record(paths["collection"], gate.COLLECTION_SCHEMA),
        "r2_execution_receipt": {
            **_record(r2_receipt, final.R6_R2_EXECUTION_RECEIPT_SCHEMA),
            "operational_authority": False,
        },
    }
    _seal(paths["receipt"], wrapper_payload)
    return paths, scene_folds, gt_paths


def _gt_inventory(
    tmp_path: Path, scene_folds: dict[str, int], gt_paths: dict[str, Path],
) -> tuple[Path, dict[str, object]]:
    shadow_root = tmp_path / "gt"
    source_root = tmp_path / "synthetic_source_gt"
    source_dataset = _seal(tmp_path / "source_dataset.manifest.json", {
        "schema": "synthetic.ca1m.train.dataset", "complete": True,
    })
    oof_sidecar = tmp_path / "synthetic_b6_oof.npz"
    oof_sidecar.write_bytes(b"synthetic-oof-sidecar")
    oof_sidecar.chmod(0o444)
    scenes: dict[str, dict] = {}
    for scene, fold in scene_folds.items():
        box_path = gt_paths[scene]
        manifest_path = shadow_root / scene / "derived_train_gt_manifest.json"
        box_sha = gate.sha256_file(box_path)
        manifest = _seal(manifest_path, {
            "schema": "boxfusion.ca1m_native_b6_train_scene.v1",
            "scene_id": scene, "source_split": "train", "train_only": True,
            "validation_ground_truth_access": False,
            "validation_scene_overlap": False,
            "official_validation_comparable": False,
            "paper_validation_claim_permitted": False,
            "derived_train_gt": True,
            "derived_train_gt_artifact": "derived_train_gt_boxes.npy",
            "derived_train_gt_sha256": box_sha,
            "compat_after_filter_sha256": box_sha,
            "artifacts": {"after_filter_boxes.npy": {"sha256": box_sha}},
            "output_scene": str(source_root / scene),
            "storage_filesystem_policy": {
                "filesystem_type": "fuseblk", "posix_mode_enforceable": False,
                "artifact_integrity_contract": "regular_no_symlink_sha256_create_only",
            },
        })
        scenes[scene] = {
            "fold_id": fold,
            "box": {
                "mode": "0o444", "path": str(box_path), "sha256": box_sha,
                "source_mode": "0o777",
                "source_path": str(source_root / scene / "derived_train_gt_boxes.npy"),
            },
            "manifest": {
                "mode": "0o444", "path": str(manifest_path),
                "sha256": gate.sha256_file(manifest), "source_mode": "0o777",
                "source_path": str(source_root / scene / "derived_train_gt_manifest.json"),
            },
        }
    payload = {
        "schema": final.GT_INVENTORY_SCHEMA, "complete": True,
        "create_only": True, "file_count": 160,
        "scene_count": 80, "fit_scene_count": 60,
        "threshold_dev_scene_count": 20, "fit_fold_ids": [2, 3, 4],
        "threshold_dev_fold_ids": [0], "locked_internal_fold_ids": [1],
        "locked_internal_scene_count_accessed": 0,
        "official_validation_comparable": False,
        "validation_ground_truth_access": False,
        "validation_prediction_access": False, "train_only": True,
        "gt_array_content_loaded": False,
        "inventory_sha256": "b" * 64,
        "opaque_source_bytes_hashed_and_copied": True,
        "shadow_files_read_only": True, "source_bytes_mutated": False,
        "output_root": str(shadow_root), "source_root": str(source_root),
        "source_dataset_manifest": {
            "path": str(source_dataset), "sha256": gate.sha256_file(source_dataset),
        },
        "oof_sidecar": {
            "path": str(oof_sidecar), "sha256": gate.sha256_file(oof_sidecar),
        },
        "scenes": scenes,
    }
    inventory = _seal(tmp_path / "gt_inventory.json", payload)
    return inventory, {
        "shadow_root": shadow_root, "source_root": source_root,
        "source_dataset": source_dataset, "oof_sidecar": oof_sidecar,
        "content_sha": payload["inventory_sha256"],
    }


def _patch_gt_inventory(
    monkeypatch: pytest.MonkeyPatch, inventory: Path, metadata: dict[str, object],
) -> None:
    monkeypatch.setattr(final, "GT_INVENTORY_PATH", inventory)
    monkeypatch.setattr(final, "GT_INVENTORY_SHA256", gate.sha256_file(inventory))
    monkeypatch.setattr(final, "GT_INVENTORY_CONTENT_SHA256", metadata["content_sha"])
    monkeypatch.setattr(final, "GT_SHADOW_ROOT", metadata["shadow_root"])
    monkeypatch.setattr(final, "GT_SOURCE_ROOT", metadata["source_root"])
    monkeypatch.setattr(final, "GT_SOURCE_DATASET_MANIFEST_PATH", metadata["source_dataset"])
    monkeypatch.setattr(
        final, "GT_SOURCE_DATASET_MANIFEST_SHA256",
        gate.sha256_file(metadata["source_dataset"]),
    )
    monkeypatch.setattr(final, "GT_OOF_SIDECAR_PATH", metadata["oof_sidecar"])
    monkeypatch.setattr(
        final, "GT_OOF_SIDECAR_SHA256", gate.sha256_file(metadata["oof_sidecar"]),
    )


def test_final_pending_config_and_official_ap_parity_are_static() -> None:
    _, cfg = final.validate_pending_config()
    assert cfg["state"] == "pending_r6_exact80"
    assert not any(cfg["authorizations"].values())
    report = final.static_preflight()
    assert report["status"] == "PASS_STATIC_PENDING_R6"
    assert report["ground_truth_access"] is False
    assert report["directory_created"] is False
    assert report["legacy_final_invalidated"] is True
    assert final.ap_parity_fixture()["pass"] is True
    assert final.sha256_file(final.R6_PREREGISTRATION_PATH) == final.R6_PREREGISTRATION_SHA256
    assert final.sha256_file(final.R6_CONFIG_PATH) == final.R6_CONFIG_SHA256
    assert final.sha256_file(final.R6_CORE_PATH) == final.R6_CORE_SHA256
    assert final.GT_INVENTORY_PATH == (
        ROOT
        / "manifests/ca1m_tr3d_benefit_gate_final_base_v4/"
        "derived_train_gt_shadow_inventory_v1.json"
    )
    assert final.sha256_file(final.GT_INVENTORY_PATH) == final.GT_INVENTORY_SHA256


def test_static_protocol_builder_does_not_open_scene_gt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        final, "validate_gt_inventory_metadata",
        lambda *_args, **_kwargs: pytest.fail("scene GT inventory validator reached"),
    )
    monkeypatch.setattr(
        final.np, "load", lambda *_args, **_kwargs: pytest.fail("GT array opened"),
    )
    payload = final.build_static_protocol_payload()
    assert payload["static_science_protocol"] is True
    assert payload["operational_authority"] is False
    assert not any(payload["access_at_seal"].values())
    assert payload["future_r6"]["static_preregistration"]["sha256"] == (
        final.R6_PREREGISTRATION_SHA256
    )
    assert payload["annotation_inventory_receipt"]["sha256"] == (
        final.GT_INVENTORY_SHA256
    )
    assert payload["invalidated_predecessor"]["sha256"] == (
        final.R3_PROTOCOL_INVALID_SHA256
    )


def test_static_protocol_is_create_only_and_rederives_current_bytes() -> None:
    source, payload, data, _ = final.validate_static_protocol()
    assert source == final.PROTOCOL_PATH
    assert data == final.canonical_json(payload)
    assert payload["sealed_before_r6_exact80_wrapper_exists"] is True
    assert payload["implementation"]["generic_gate_core"]["sha256"] == (
        final.GENERIC_GATE_SHA256
    )


def test_legacy_final_runner_is_tombstoned() -> None:
    result = subprocess.run(
        [sys.executable, str(LEGACY_RUNNER)], cwd=ROOT, capture_output=True,
    )
    assert result.returncode == 66
    assert result.stdout == b""
    assert json.loads(result.stderr)["status"] == "INVALIDATED_FINAL_R1_STATIC_BLOCK"
    r2 = subprocess.run(
        [sys.executable, str(R2_RUNNER)], cwd=ROOT, capture_output=True,
    )
    assert r2.returncode == 66
    assert r2.stdout == b""
    assert json.loads(r2.stderr)["status"] == (
        "INVALIDATED_FINAL_R2_RUNTIME_ROOT_TOCTOU"
    )
    r3 = subprocess.run(
        [sys.executable, str(R3_RUNNER)], cwd=ROOT, capture_output=True,
    )
    assert r3.returncode == 66
    assert r3.stdout == b""
    assert json.loads(r3.stderr)["status"] == (
        "INVALIDATED_FINAL_R3_EXECUTION_BOUNDARY"
    )


@pytest.mark.parametrize(
    "program,args",
    [(PREFLIGHT, ["--mode", "r6"]), (RUNNER, [])],
)
def test_pending_cli_exit3_stdout_empty_and_namespace_absent(
    program: Path, args: list[str],
) -> None:
    result = subprocess.run(
        [sys.executable, str(program), *args], cwd=ROOT, capture_output=True,
    )
    assert result.returncode == 3
    assert result.stdout == b""
    assert json.loads(result.stderr)["status"].startswith("BLOCKED_PENDING")
    assert not Path("/extra/ZhaoX").joinpath(final.NAMESPACE).exists()


def test_r4_r5_wrappers_are_not_addressable() -> None:
    for revision in ("r4", "r5"):
        path = Path("/extra/ZhaoX") / (
            f"ca1m_tr3d_e961_terminal_inputs_xfit_r2_v5_{revision}"
        ) / "manifests" / f"M_EXACT80_{revision.upper()}_RECEIPT.json"
        with pytest.raises(ValueError, match="canonical R6"):
            final.load_r6_exact80_binding(path)


def test_pending_r6_stops_before_gt_mkdir_or_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    missing = tmp_path / "missing_r6_receipt.json"
    called = False

    def no_config(_path: Path):
        return tmp_path / "pending.json", {}

    def forbidden_loader():
        nonlocal called
        called = True

    monkeypatch.setattr(final, "validate_pending_config", no_config)
    monkeypatch.setattr(final, "R6_RECEIPT_PATH", missing)
    with pytest.raises(final.PendingR6Inputs):
        final.operational_preflight_pending(tmp_path / "pending.json")
    forbidden_loader  # prove a loader can exist but is never invoked
    assert called is False
    assert not missing.parent.joinpath("runtime").exists()


def test_formal_entry_and_claim_are_zero_argument_and_reject_injection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert tuple(inspect.signature(final.run_final_gate).parameters) == ()
    assert tuple(inspect.signature(final._claim_runtime).parameters) == ()
    reached = {"claim": 0, "gt": 0, "writer": 0}

    def forbidden_claim() -> None:
        reached["claim"] += 1
        pytest.fail("injected formal call entered claim")

    monkeypatch.setattr(final, "_claim_runtime", forbidden_claim)
    monkeypatch.setattr(
        final.np, "load",
        lambda *_args, **_kwargs: reached.__setitem__("gt", reached["gt"] + 1),
    )
    monkeypatch.setattr(
        final, "_write_runtime_bytes_fuse",
        lambda *_args, **_kwargs: reached.__setitem__(
            "writer", reached["writer"] + 1
        ),
    )
    injected = object()
    with pytest.raises(TypeError):
        final.run_final_gate(injected)  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        final.run_final_gate(context=injected)  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        final._claim_runtime(injected)  # type: ignore[call-arg]
    assert reached == {"claim": 0, "gt": 0, "writer": 0}


def test_synthetic_r6_writer_is_restored_on_return_and_exception(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = gate.write_bytes_create_only
    _synthetic_r6(tmp_path / "success", monkeypatch)
    assert gate.write_bytes_create_only is original
    with monkeypatch.context() as scoped:
        scoped.setattr(
            sys.modules[__name__], "_patch_r6_paths",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("fixture stop")),
        )
        with pytest.raises(RuntimeError, match="fixture stop"):
            _synthetic_r6(tmp_path / "failure", monkeypatch)
    assert gate.write_bytes_create_only is original


def test_r6_binding_is_exact80_e961_double_oof(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths, scene_folds, _ = _synthetic_r6(tmp_path, monkeypatch)
    binding = final.load_r6_exact80_binding(paths["receipt"])
    assert binding.collection_path == paths["collection"]
    assert len(binding.scene_folds) == 80
    assert {fold: list(scene_folds.values()).count(fold) for fold in (0, 2, 3, 4)} == {
        0: 20, 2: 20, 3: 20, 4: 20,
    }
    assert binding.upstream_records["r2_internal_execution_receipt"][
        "operational_authority"
    ] is False
    frozen_prereg_sha = final.R6_PREREGISTRATION_SHA256
    monkeypatch.setattr(final, "R6_PREREGISTRATION_SHA256", "0" * 64)
    with pytest.raises(PermissionError, match="frozen CODE-PASS SHA"):
        final.load_r6_exact80_binding(paths["receipt"])
    monkeypatch.setattr(final, "R6_PREREGISTRATION_SHA256", frozen_prereg_sha)
    wrapper = json.loads(paths["receipt"].read_text())
    wrapper["candidate_collection"] = {
        **wrapper["candidate_collection"],
        "path": str(tmp_path / "legacy" / "CANDIDATE_COLLECTION_EXACT80.json"),
    }
    forged = _seal(tmp_path / "forged_r6_receipt.json", wrapper)
    monkeypatch.setattr(final, "R6_RECEIPT_PATH", forged)
    with pytest.raises(ValueError, match="canonical"):
        final.load_r6_exact80_binding(forged)

    wrapper = json.loads(paths["receipt"].read_text())
    wrapper["r2_execution_receipt"]["operational_authority"] = True
    forged_authority = _seal(tmp_path / "forged_r2_authority_receipt.json", wrapper)
    monkeypatch.setattr(final, "R6_RECEIPT_PATH", forged_authority)
    with pytest.raises(ValueError, match="non-authoritative"):
        final.load_r6_exact80_binding(forged_authority)


def test_r6_committed_extra_artifacts_accept_mode777_but_generic_does_not(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths, _, _ = _synthetic_r6(tmp_path, monkeypatch)
    for artifact in paths["root"].rglob("*"):
        if artifact.is_file():
            artifact.chmod(0o777)
    with pytest.raises(ValueError, match="read-only"):
        gate.load_candidate_collection_v5(paths["collection"])
    binding = final.load_r6_exact80_binding(paths["receipt"])
    assert binding.collection_path == paths["collection"]
    assert binding.wrapper_identity["st_ino"] == paths["receipt"].stat().st_ino


def test_arbitrary_absolute_gt_inventory_is_rejected_before_box_open(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths, scene_folds, gt_paths = _synthetic_r6(tmp_path, monkeypatch)
    inventory, _ = _gt_inventory(tmp_path, scene_folds, gt_paths)
    binding = final.load_r6_exact80_binding(paths["receipt"])
    with pytest.raises(ValueError, match="canonical frozen"):
        final.validate_gt_inventory_metadata(inventory, r6_binding=binding)


def _claim_context(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> tuple[final.ExecutionContext, Path]:
    parent = tmp_path / "host"
    parent.mkdir()
    runtime = parent / final.NAMESPACE
    outputs = {
        key: runtime / (
            "materialized" if key == "materialization_root"
            else f"dir_{key}/{key}.bin"
        )
        for key in final.OUTPUT_PATHS
    }
    prereg = _seal(tmp_path / "claim_prereg.json", {"schema": "synthetic.prereg"})
    authorization = _seal(tmp_path / "claim_authorization.json", {"schema": "synthetic.auth"})
    ready = _seal(tmp_path / "claim_ready.json", {"schema": "synthetic.ready"})
    dummy = _seal(tmp_path / "dummy.json", {"schema": "synthetic.dummy"})
    r6 = final.R6Exact80Binding(
        wrapper_path=dummy, wrapper_sha256=gate.sha256_file(dummy),
        wrapper_identity=final._deep_immutable(
            final.artifact_record(dummy, "dummy")["identity"]
        ),
        collection_path=dummy, collection_sha256=gate.sha256_file(dummy),
        collection_identity=final._deep_immutable(
            final.artifact_record(dummy, "dummy")["identity"]
        ),
        authorization_commit_id="a" * 64,
        scene_folds=final._deep_immutable({}),
        upstream_records=final._deep_immutable({}),
    )
    inventory = final.GTInventoryBinding(
        dummy, gate.sha256_file(dummy),
        final._deep_immutable(final.artifact_record(dummy, "dummy")["identity"]),
        final._deep_immutable({}),
    )
    context = final.ExecutionContext(
        authorization, gate.sha256_file(authorization),
        final._deep_immutable(
            final.artifact_record(authorization, "auth")["identity"]
        ),
        prereg, final._deep_immutable({}), ready, final._deep_immutable({}),
        r6, inventory, final._deep_immutable(outputs),
    )
    monkeypatch.setattr(final, "OUTPUT_PARENT_PATH", parent)
    monkeypatch.setattr(final, "RUNTIME_ROOT", runtime)
    monkeypatch.setattr(final, "RUN_CLAIM_PATH", parent / f".{final.NAMESPACE}.run.claim")
    monkeypatch.setattr(final, "RUN_AUTHORIZATION_PATH", authorization)
    monkeypatch.setattr(final, "OUTPUT_PATHS", outputs)
    monkeypatch.setattr(
        final, "load_execution_context", lambda *_args, **_kwargs: context,
    )
    return context, parent


def test_fresh_rederive_rejects_replaced_collection_folds_and_gt_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    context, _ = _claim_context(tmp_path, monkeypatch)
    with pytest.raises(TypeError):
        context.outputs["fit_dataset"] = tmp_path / "injected"  # type: ignore[index]
    with pytest.raises(TypeError):
        context.r6.scene_folds["61000000"] = 0  # type: ignore[index]
    with pytest.raises(TypeError):
        context.inventory.scene_rows["61000000"] = {}  # type: ignore[index]

    alias = _seal(tmp_path / "same_bytes_collection_alias.json", {"schema": "synthetic.dummy"})
    replaced_collection = replace(
        context,
        r6=replace(context.r6, collection_path=alias),
    )
    replaced_folds = replace(
        context,
        r6=replace(
            context.r6,
            scene_folds=final._deep_immutable({"61000000": 0}),
        ),
    )
    replaced_rows = replace(
        context,
        inventory=replace(
            context.inventory,
            scene_rows=final._deep_immutable({"61000000": {"box": {}}}),
        ),
    )
    for forged in (replaced_collection, replaced_folds, replaced_rows):
        with pytest.raises(RuntimeError, match="fresh canonical"):
            final.revalidate_execution_inputs(forged)


def test_gt_alias_same_sha_is_rejected_before_np_load(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    context, _ = _claim_context(tmp_path, monkeypatch)
    scene = "61000000"
    canonical_root = tmp_path / "canonical_gt"
    canonical_box = canonical_root / scene / "derived_train_gt_boxes.npy"
    canonical_box.parent.mkdir(parents=True)
    np.save(canonical_box, np.asarray([_box((0.0, 0.0, 0.0), 1.0)], np.float64))
    alias = tmp_path / "alias_gt.npy"
    alias.write_bytes(canonical_box.read_bytes())
    box_record = {
        "path": str(alias), "sha256": gate.sha256_file(alias),
    }
    forged = replace(
        context,
        r6=replace(
            context.r6,
            scene_folds=final._deep_immutable({scene: 2}),
        ),
        inventory=replace(
            context.inventory,
            scene_rows=final._deep_immutable({
                scene: {"fold_id": 2, "box": box_record, "manifest": {}}
            }),
        ),
    )
    monkeypatch.setattr(final, "GT_SHADOW_ROOT", canonical_root)
    monkeypatch.setattr(
        final, "load_execution_context", lambda *_args, **_kwargs: forged,
    )
    monkeypatch.setattr(final, "_revalidate_inventory_master", lambda _context: None)
    observed: list[Path] = []
    original_validate = final.validate_artifact_record

    def recording_validate(value, name, **kwargs):
        if name == f"CA-train GT boxes {scene}":
            observed.append(kwargs["canonical_path"])
        return original_validate(value, name, **kwargs)

    np_loads = 0

    def forbidden_np_load(*_args, **_kwargs):
        nonlocal np_loads
        np_loads += 1
        pytest.fail("np.load reached for noncanonical GT alias")

    monkeypatch.setattr(final, "validate_artifact_record", recording_validate)
    monkeypatch.setattr(final.np, "load", forbidden_np_load)
    capability = final._claim_runtime()
    try:
        loader = final._inventory_ground_truth_loader(forged, capability)
        with pytest.raises(ValueError, match="canonical"):
            loader(scene)
        assert observed == [canonical_box]
        assert np_loads == 0
    finally:
        final._release_run_capability(capability)


def test_run_claim_single_writer_fd_owner_and_restart_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    context, _ = _claim_context(tmp_path, monkeypatch)
    capability = final._claim_runtime()
    try:
        final._guard_run_capability(capability)
        with pytest.raises(final.RunClaimConsumed):
            final._claim_runtime()

        duplicated = os.dup(capability.writer_fd)
        try:
            duplicate_capability = final._RunCapability(
                capability.token, duplicated, capability.parent_fd,
                capability.runtime_fd,
            )
            final._guard_run_capability(duplicate_capability)
        finally:
            os.close(duplicated)

        independent = os.open(final.RUN_CLAIM_PATH, os.O_RDONLY)
        try:
            independent_capability = final._RunCapability(
                capability.token, independent, capability.parent_fd,
                capability.runtime_fd,
            )
            with pytest.raises(PermissionError, match="lock-owning"):
                final._guard_run_capability(independent_capability)
        finally:
            os.close(independent)

        creator_pid = os.getpid()
        with monkeypatch.context() as scoped:
            scoped.setattr(final.os, "getpid", lambda: creator_pid + 1)
            with pytest.raises(PermissionError, match="creator process"):
                final._guard_run_capability(capability)

        fcntl.flock(capability.writer_fd, fcntl.LOCK_UN)
        with pytest.raises(PermissionError, match="no longer held"):
            final._guard_run_capability(capability)
    finally:
        final._release_run_capability(capability)
    assert final.RUN_CLAIM_PATH.exists()
    with pytest.raises(final.RunClaimConsumed):
        final._claim_runtime()
    assert final.RUNTIME_ROOT.is_dir()


def test_runtime_root_injected_after_context_is_rejected_and_claim_consumed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    context, _ = _claim_context(tmp_path, monkeypatch)
    final.RUNTIME_ROOT.mkdir()
    with pytest.raises(FileExistsError, match="appeared after context"):
        final._claim_runtime()
    assert final.RUN_CLAIM_PATH.exists()
    with pytest.raises(final.RunClaimConsumed):
        final._claim_runtime()


def test_formal_entry_parent_swap_during_claim_writes_only_held_parent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    context, parent = _claim_context(tmp_path, monkeypatch)
    held_parent = tmp_path / "held_host_during_claim"
    original_loader = final.load_execution_context
    swapped = False
    gt_loader_calls = 0

    def swap_after_parent_is_held(*args, **kwargs):
        nonlocal swapped
        assert swapped is False
        parent.rename(held_parent)
        parent.mkdir()
        swapped = True
        return original_loader(*args, **kwargs)

    def forbidden_gt_loader(*_args, **_kwargs):
        nonlocal gt_loader_calls
        gt_loader_calls += 1
        pytest.fail("GT loader reached after output-parent swap")

    monkeypatch.setattr(final, "load_execution_context", swap_after_parent_is_held)
    monkeypatch.setattr(final, "_inventory_ground_truth_loader", forbidden_gt_loader)
    with pytest.raises(PermissionError, match="output-parent"):
        final.run_final_gate()
    assert swapped is True
    assert gt_loader_calls == 0
    assert list(parent.iterdir()) == []
    assert (held_parent / final.RUN_CLAIM_PATH.name).is_file()
    assert (held_parent / final.NAMESPACE).is_dir()
    assert final._RUN_AUTHORITIES == {}


def test_unregistered_runtime_subdirectory_is_never_reused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    context, _ = _claim_context(tmp_path, monkeypatch)
    capability = final._claim_runtime()
    target = context.outputs["fit_dataset"]
    target.parent.mkdir()
    try:
        with pytest.raises(PermissionError, match="not claim-owned"):
            final._write_runtime_bytes_fuse(
                target, b"blocked", "injected subdirectory output",
                runtime_root=final.RUNTIME_ROOT, capability=capability,
            )
        assert not target.exists()
    finally:
        final._release_run_capability(capability)


def test_claim_owned_runtime_fd_writer_accepts_fuse_mode777(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    context, _ = _claim_context(tmp_path, monkeypatch)
    capability = final._claim_runtime()
    target = context.outputs["fit_dataset"]
    original = final._exclusive_bytes_at_fd

    def fuse_mode777(
        parent_fd: int, leaf: str, payload: bytes, name: str,
    ) -> dict[str, int]:
        identity = original(parent_fd, leaf, payload, name)
        leaf_fd = os.open(leaf, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=parent_fd)
        try:
            os.fchmod(leaf_fd, 0o777)
        finally:
            os.close(leaf_fd)
        return identity

    monkeypatch.setattr(final, "_exclusive_bytes_at_fd", fuse_mode777)
    try:
        output = final._write_runtime_bytes_fuse(
            target, b"fuse-output", "claim-owned output",
            runtime_root=final.RUNTIME_ROOT, capability=capability,
        )
        assert output == target
        assert output.read_bytes() == b"fuse-output"
        assert output.stat().st_mode & 0o777 == 0o777
        final._guard_run_capability(capability)
        with pytest.raises(FileExistsError):
            final._write_runtime_bytes_fuse(
                target, b"collision", "claim-owned output",
                runtime_root=final.RUNTIME_ROOT, capability=capability,
            )
    finally:
        final._release_run_capability(capability)


def test_runtime_root_swap_after_guard_before_open_writes_only_held_inode_and_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    context, parent = _claim_context(tmp_path, monkeypatch)
    capability = final._claim_runtime()
    runtime = final.RUNTIME_ROOT
    moved = parent / f"{final.NAMESPACE}.held"
    target = context.outputs["fit_dataset"]
    original_dup = final.os.dup
    swapped = False

    def swap_on_runtime_dup(descriptor: int) -> int:
        nonlocal swapped
        if descriptor == capability.runtime_fd and not swapped:
            swapped = True
            runtime.rename(moved)
            runtime.mkdir()
        return original_dup(descriptor)

    monkeypatch.setattr(final.os, "dup", swap_on_runtime_dup)
    try:
        with pytest.raises(PermissionError, match="runtime-root"):
            final._write_runtime_bytes_fuse(
                target, b"held-root-only", "root swap output",
                runtime_root=runtime, capability=capability,
            )
        assert swapped is True
        assert not target.exists()
        assert (moved / target.relative_to(runtime)).read_bytes() == b"held-root-only"
    finally:
        final._release_run_capability(capability)
        monkeypatch.setattr(final.os, "dup", original_dup)
        runtime.rmdir()
        moved.rename(runtime)


def test_output_parent_swap_after_writer_guard_targets_only_held_runtime_fd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    context, parent = _claim_context(tmp_path, monkeypatch)
    capability = final._claim_runtime()
    held_parent = tmp_path / "held_host_before_writer_open"
    target = context.outputs["fit_dataset"]
    original_dup = final.os.dup
    swapped = False

    def swap_parent_on_runtime_dup(descriptor: int) -> int:
        nonlocal swapped
        if descriptor == capability.runtime_fd and not swapped:
            swapped = True
            parent.rename(held_parent)
            parent.mkdir()
        return original_dup(descriptor)

    monkeypatch.setattr(final.os, "dup", swap_parent_on_runtime_dup)
    try:
        with pytest.raises(PermissionError, match="output-parent"):
            final._write_runtime_bytes_fuse(
                target, b"held-parent-only", "parent swap output",
                runtime_root=final.RUNTIME_ROOT, capability=capability,
            )
        assert swapped is True
        assert list(parent.iterdir()) == []
        assert not target.exists()
        held_output = (
            held_parent / final.NAMESPACE / target.relative_to(final.RUNTIME_ROOT)
        )
        assert held_output.read_bytes() == b"held-parent-only"
    finally:
        final._release_run_capability(capability)


def test_run_claim_parent_symlink_swap_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    context, parent = _claim_context(tmp_path, monkeypatch)
    capability = final._claim_runtime()
    moved = tmp_path / "held_host"
    replacement = tmp_path / "replacement_host"
    replacement.mkdir()
    parent.rename(moved)
    parent.symlink_to(replacement, target_is_directory=True)
    try:
        with pytest.raises((PermissionError, FileNotFoundError, NotADirectoryError, OSError)):
            final._guard_run_capability(capability)
    finally:
        final._release_run_capability(capability)
        parent.unlink()
        moved.rename(parent)


def test_create_only_writer_binds_inode_hash_and_refuses_collision(tmp_path: Path) -> None:
    target = final._exclusive_bytes_fuse(
        tmp_path / "artifact.json",
        gate._canonical_json({"schema": "synthetic", "value": 1}), "artifact",
    )
    record = final.artifact_record(target, "artifact", schema="synthetic")
    _, data, identity = final.validate_artifact_record(
        record, "artifact", schema="synthetic", require_identity=True
    )
    assert final.sha256_bytes(data) == record["sha256"]
    assert identity["st_nlink"] == 1
    with pytest.raises(FileExistsError):
        final._exclusive_bytes_fuse(
            target, gate._canonical_json({"schema": "synthetic", "value": 2}),
            "artifact",
        )


def test_canonical_json_preserves_plain_bytes_and_thaws_deep_authority() -> None:
    payload = {
        "z": [True, None, 2.5],
        "a": {"nested": [1, "two"]},
    }
    expected = b'{"a":{"nested":[1,"two"]},"z":[true,null,2.5]}\n'
    assert final.canonical_json(payload) == expected
    frozen = final._deep_immutable(payload)
    assert final.canonical_json(frozen) == expected
    assert json.loads(final.canonical_json(frozen)) == payload
    with pytest.raises(TypeError, match="unsupported type"):
        final.canonical_json({"bad": Path("not-json")})


def test_full_final_chain_trains_on_fit60_then_runs_frozen_fold0(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths, scene_folds, gt_paths = _synthetic_r6(tmp_path, monkeypatch)
    inventory, gt_metadata = _gt_inventory(tmp_path, scene_folds, gt_paths)
    _patch_gt_inventory(monkeypatch, inventory, gt_metadata)
    formal_root = tmp_path / "final_formal"
    output_parent = tmp_path / "formal_output_parent"
    output_parent.mkdir()
    runtime_root = output_parent / final.NAMESPACE
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
    protocol = formal_root / "PREREGISTRATION_PROTOCOL.json"
    pending = _seal(tmp_path / "synthetic_pending.json", {
        "schema": final.PENDING_SCHEMA, "synthetic": True,
    })
    monkeypatch.setattr(final, "PREREGISTRATION_PATH", prereg)
    monkeypatch.setattr(final, "PROTOCOL_PATH", protocol)
    monkeypatch.setattr(final, "READY_CONFIG_PATH", ready)
    monkeypatch.setattr(final, "RUN_AUTHORIZATION_PATH", authorization)
    monkeypatch.setattr(final, "MANIFEST_ROOT", formal_root)
    monkeypatch.setattr(final, "OUTPUT_PARENT_PATH", output_parent)
    monkeypatch.setattr(final, "RUNTIME_ROOT", runtime_root)
    monkeypatch.setattr(final, "CANONICAL_RUNTIME_ROOT", runtime_root)
    monkeypatch.setattr(
        final, "RUN_CLAIM_PATH",
        output_parent / f".{final.NAMESPACE}.run.claim",
    )
    monkeypatch.setattr(final, "OUTPUT_PATHS", outputs)
    monkeypatch.setattr(final, "validate_pending_config", lambda _path: (pending, {}))
    final._publish_json_replay_safe(
        protocol, final.build_static_protocol_payload(pending),
        "synthetic static protocol",
    )
    sealed_prereg = final.seal_scientific_preregistration(
        gt_inventory_path=inventory, output_path=prereg,
        pending_config_path=pending, r6_receipt_path=paths["receipt"],
    )
    assert sealed_prereg == prereg
    final.seal_ready_authorization(
        preregistration_path=prereg, ready_path=ready,
        authorization_path=authorization,
    )
    context = final.load_execution_context(authorization)
    assert context.preregistration["opaque_gt_bytes_hashed_at_seal"] is True
    assert context.preregistration["gt_arrays_decoded_at_seal"] is False
    assert context.preregistration["annotation_inventory"][
        "opaque_box_bytes_hashed_at_seal"
    ] is True
    opened: list[str] = []
    original_validate = final.validate_artifact_record

    def record_internal_gt(value, name, **kwargs):
        prefix = "CA-train GT boxes "
        if name.startswith(prefix):
            scene = name[len(prefix):]
            opened.append(scene)
            assert kwargs.get("canonical_path") == (
                final.GT_SHADOW_ROOT / scene / "derived_train_gt_boxes.npy"
            )
        return original_validate(value, name, **kwargs)

    original_writer = final._exclusive_bytes_at_fd

    def fuse_mode777(
        parent_fd: int, leaf: str, payload: bytes, name: str,
    ) -> dict[str, int]:
        identity = original_writer(parent_fd, leaf, payload, name)
        leaf_fd = os.open(leaf, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=parent_fd)
        try:
            os.fchmod(leaf_fd, 0o777)
        finally:
            os.close(leaf_fd)
        return identity

    monkeypatch.setattr(final, "validate_artifact_record", record_internal_gt)
    monkeypatch.setattr(final, "_exclusive_bytes_at_fd", fuse_mode777)
    receipt = final.run_final_gate()
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
    for name, output in outputs.items():
        if name != "materialization_root" and output.exists():
            assert output.stat().st_mode & 0o777 == 0o777
    assert final.RUN_CLAIM_PATH.is_file()
