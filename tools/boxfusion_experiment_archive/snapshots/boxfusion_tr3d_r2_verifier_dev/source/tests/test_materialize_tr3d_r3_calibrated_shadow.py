from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import pickle
from types import SimpleNamespace

import numpy as np
import pytest

from boxfusion.tr3d_r3_active import active_config_sha256
from boxfusion.tr3d_r3_calibrator import R3VetoCalibrator, calibrator_sha256
from tools import materialize_tr3d_r3_calibrated_shadow as materializer


def _corners(offset: float) -> np.ndarray:
    signs = np.asarray(
        [[x, y, z] for x in (0, 1) for y in (0, 1) for z in (0, 1)],
        dtype=np.float32,
    )
    return np.ascontiguousarray(signs + np.float32(offset))


def _model(*, authorized: bool) -> R3VetoCalibrator:
    coefficients = np.zeros((3, 6), dtype=np.float64)
    coefficients[0, 2] = 2.0
    coefficients[2, 2] = -2.0
    return R3VetoCalibrator(
        feature_mean=np.zeros(6, dtype=np.float64),
        feature_scale=np.ones(6, dtype=np.float64),
        coefficients=coefficients,
        intercept=np.asarray([-1.0, -10.0, 1.0], dtype=np.float64),
        activation_authorized=authorized,
        dataset_sha256="1" * 64,
        scene_list_sha256="2" * 64,
        metadata={
            "train_gate_pass": authorized,
            "train_only": True,
            "inference_lineage_contract": {
                "prefix_id": "p100",
                "parent_checkpoint_sha256": "5" * 64,
                "parent_config_sha256": "6" * 64,
                "r3_config_sha256": "7" * 64,
                "r3_code_sha256": "8" * 64,
                "primary_active_config_sha256": active_config_sha256(),
                "training_prefix_manifest_sha256": "a" * 64,
                "anchor_distribution_contract": {
                    "score_threshold": 0.4,
                    "minimum_extent_m": 0.4,
                    "quality_detector_blend": 0.4,
                    "selective_gate": {
                        "max_center_shift_m": 0.1,
                        "min_volume_ratio": 0.5,
                        "max_volume_ratio": 2.0,
                    },
                    "quality_checkpoint_sha256": "b" * 64,
                    "yoloe_checkpoint_sha256": "c" * 64,
                },
            },
        },
    )


def _write_model(path: Path, *, authorized: bool) -> None:
    path.write_text(
        json.dumps(_model(authorized=authorized).as_dict()), encoding="utf-8"
    )
    path.chmod(0o444)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_calibrated_adapter_is_veto_only_and_preserves_invariants() -> None:
    source = [[(0, _corners(0), 0.2), (1, _corners(3), 0.2)]]
    cache = SimpleNamespace(
        anchor_count=2,
        proposal_ids=np.asarray([10, 20], dtype=np.int64),
        proposal_corners_world=np.stack([_corners(8), _corners(9)]),
        anchor_index=np.asarray([0, 1], dtype=np.int64),
        tr3d_score=np.asarray([0.8, 0.8], dtype=np.float32),
        anchor_score=np.asarray([0.2, 0.2], dtype=np.float32),
        anchor_iou=np.asarray([0.8, 0.2], dtype=np.float32),
        center_distance_over_anchor_diagonal=np.asarray([0.1, 0.5], dtype=np.float32),
        volume_ratio=np.asarray([1.0, 2.0], dtype=np.float32),
        point_density_m3=np.asarray([9.0, 3.0], dtype=np.float32),
    )
    model = _model(authorized=True)
    output, summary = materializer._materialize_payload(source, cache, model)
    assert summary["calibrator_sha256"] == calibrator_sha256(model)
    assert summary["primary_count"] == 2
    assert summary["accepted_count"] == 1
    assert summary["vetoed_count"] == 1
    assert materializer.raw._validate_invariants(source, output)["changed_rows"] == 1
    np.testing.assert_array_equal(output[0][0][1], cache.proposal_corners_world[0])
    np.testing.assert_array_equal(output[0][1][1], source[0][1][1])


def test_unauthorized_model_fails_before_output_namespace_is_created(
    tmp_path: Path,
) -> None:
    scenes = [f"scene{index:04d}_00" for index in range(10)]
    scene_list = tmp_path / "scenes.txt"
    scene_list.write_text("".join(f"{scene}\n" for scene in scenes), encoding="utf-8")
    model = tmp_path / "unauthorized.json"
    _write_model(model, authorized=False)
    output_root = tmp_path / "predictions"
    manifest = tmp_path / "manifest.json"
    args = argparse.Namespace(
        frozen_manifest=tmp_path / "not-read.json",
        r3_export_report=tmp_path / "not-read-export.json",
        r3_cache_root=tmp_path / "not-read-cache",
        calibrator_model=model,
        scene_list=scene_list,
        scans_root=tmp_path / "not-read-scans",
        output_root=output_root,
        manifest=manifest,
        prefix_id="p100",
        resume=False,
    )
    with pytest.raises(PermissionError, match="train-only gate"):
        materializer.materialize(args)
    assert not output_root.exists()
    assert not manifest.exists()


def test_authorized_model_requires_positive_train_gate_attestation(
    tmp_path: Path,
) -> None:
    payload = _model(authorized=True).as_dict()
    del payload["metadata"]["train_gate_pass"]
    model = tmp_path / "bad-attestation.json"
    model.write_text(json.dumps(payload), encoding="utf-8")
    model.chmod(0o444)
    with pytest.raises(PermissionError, match="attestation"):
        materializer._load_authorized_calibrator(model)


def test_inference_lineage_mismatch_fails_closed() -> None:
    model = _model(authorized=True)
    export = {
        "expected_parent_checkpoint_sha256": "0" * 64,
        "expected_parent_config_sha256": "6" * 64,
        "r3_config_sha256": "7" * 64,
        "r3_code_sha256": "8" * 64,
    }
    frozen = {
        "metadata": {
            "score_threshold": 0.4,
            "minimum_extent_m": 0.4,
            "quality_detector_blend": 0.4,
            "selective_gate": {
                "max_center_shift_m": 0.1,
                "min_volume_ratio": 0.5,
                "max_volume_ratio": 2.0,
            },
        },
        "artifacts": {
            "quality_checkpoint": {"sha256": "b" * 64},
            "yoloe_checkpoint": {"sha256": "c" * 64},
        },
    }
    with pytest.raises(ValueError, match="parent_checkpoint_sha256"):
        materializer._validate_inference_lineage(model, export, frozen, "p100")


def test_manifest_binds_model_and_r3_lineage_in_isolated_create_only_tree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    scenes = [f"scene{index:04d}_00" for index in range(10)]
    frozen_root = tmp_path / "frozen"
    r3_root = tmp_path / "r3"
    scans_root = tmp_path / "scans"
    frozen_root.mkdir()
    r3_root.mkdir()
    scans_root.mkdir()
    source_bytes: dict[str, bytes] = {}
    sidecar_hashes: dict[str, str] = {}
    for index, scene in enumerate(scenes):
        source_path = frozen_root / f"{scene}_boxes.pkl"
        with source_path.open("wb") as handle:
            pickle.dump(
                [[(0, _corners(float(index)), 0.2)]],
                handle,
                protocol=materializer.raw.PICKLE_PROTOCOL,
            )
        source_bytes[scene] = source_path.read_bytes()
        sidecar = r3_root / scene / "p100.npz"
        sidecar.parent.mkdir()
        sidecar.write_bytes(scene.encode("utf-8"))
        sidecar_hashes[scene] = _sha(sidecar)
    scene_list = tmp_path / "scenes.txt"
    scene_list.write_text("".join(f"{scene}\n" for scene in scenes), encoding="utf-8")
    frozen_manifest = tmp_path / "frozen.json"
    frozen_manifest.write_text("{}\n", encoding="utf-8")
    export_path = tmp_path / "export.json"
    export_path.write_text("{}\n", encoding="utf-8")
    model_path = tmp_path / "authorized.json"
    _write_model(model_path, authorized=True)
    snapshot = {
        "anchor_name": "g0-test",
        "prediction_tree_sha256": "3" * 64,
        "artifact_tree_sha256": "4" * 64,
        "scene_list_sha256": _sha(scene_list),
    }
    verified = {
        **snapshot,
        "scene_ids": scenes,
        "reference_result_root": str(frozen_root),
        "metadata": {
            "score_threshold": 0.4,
            "minimum_extent_m": 0.4,
            "quality_detector_blend": 0.4,
            "selective_gate": {
                "max_center_shift_m": 0.1,
                "min_volume_ratio": 0.5,
                "max_volume_ratio": 2.0,
            },
        },
        "artifacts": {
            "quality_checkpoint": {"sha256": "b" * 64},
            "yoloe_checkpoint": {"sha256": "c" * 64},
        },
    }
    export = {
        "frozen_manifest": str(frozen_manifest),
        "frozen_manifest_sha256": _sha(frozen_manifest),
        "frozen_prediction_tree_sha256": snapshot["prediction_tree_sha256"],
        "parent_cache_root": str(tmp_path / "parent"),
        "prefix_manifest": str(tmp_path / "prefix.jsonl"),
        "r3_cache_root": str(r3_root),
        "scene_list": str(scene_list),
        "scans_root": str(scans_root),
        "prefix_id": "p100",
        "expected_parent_checkpoint_sha256": "5" * 64,
        "expected_parent_config_sha256": "6" * 64,
        "r3_config": {"r2a_enabled": False, "r2b_enabled": False},
        "r3_config_sha256": "7" * 64,
        "r3_code_sha256": "8" * 64,
        "parent_evidence_hashes": {},
        "scenes": [
            {"scene_id": scene, "r3_sidecar_sha256": sidecar_hashes[scene]}
            for scene in scenes
        ],
    }
    monkeypatch.setattr(
        materializer.raw, "verify_frozen_anchor_manifest", lambda _: dict(verified)
    )
    monkeypatch.setattr(
        materializer.raw, "_load_export", lambda _path, _scenes: dict(export)
    )
    monkeypatch.setattr(materializer, "_code_hash", lambda: "9" * 64)
    monkeypatch.setattr(
        materializer.raw,
        "_load_bound_cache",
        lambda **kwargs: SimpleNamespace(
            anchor_count=1,
            proposal_ids=np.asarray([1], dtype=np.int64),
            proposal_corners_world=np.stack([_corners(100)]),
            anchor_index=np.asarray([0], dtype=np.int64),
            tr3d_score=np.asarray([0.8], dtype=np.float32),
            anchor_score=np.asarray([0.2], dtype=np.float32),
            anchor_iou=np.asarray([0.8], dtype=np.float32),
            center_distance_over_anchor_diagonal=np.asarray([0.1], dtype=np.float32),
            volume_ratio=np.asarray([1.0], dtype=np.float32),
            point_density_m3=np.asarray([9.0], dtype=np.float32),
        ),
    )
    output_root = tmp_path / "calibrated"
    manifest_path = tmp_path / "manifest.json"
    manifest = materializer.materialize(
        argparse.Namespace(
            frozen_manifest=frozen_manifest,
            r3_export_report=export_path,
            r3_cache_root=r3_root,
            calibrator_model=model_path,
            scene_list=scene_list,
            scans_root=scans_root,
            output_root=output_root,
            manifest=manifest_path,
            prefix_id="p100",
            resume=False,
        )
    )
    assert manifest["train_gate_activation_authorized"] is True
    assert manifest["labels_scores_order_count_unchanged"] is True
    assert manifest["calibrator_model_file_sha256"] == _sha(model_path)
    assert manifest["calibrator_sha256"] == calibrator_sha256(
        _model(authorized=True)
    )
    assert manifest["r3_export_report_sha256"] == _sha(export_path)
    assert manifest["r3_config_sha256"] == "7" * 64
    assert manifest["calibrator_lineage_compatibility"]["compatible"] is True
    assert manifest["counts"]["accepted_replacements"] == 10
    assert manifest["counts"]["vetoed_replacements"] == 0
    assert manifest_path.stat().st_mode & 0o222 == 0
    assert all(path.stat().st_mode & 0o222 == 0 for path in output_root.iterdir())
    assert all(
        (frozen_root / f"{scene}_boxes.pkl").read_bytes() == source_bytes[scene]
        for scene in scenes
    )
