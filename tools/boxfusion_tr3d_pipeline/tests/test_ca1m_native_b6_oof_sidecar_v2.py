from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import train_ca1m_native_b6_quality as training  # noqa: E402
from train_ca1m_native_b6_quality import (  # noqa: E402
    OOF_ROW_SCORE_MANIFEST_SCHEMA,
    OOF_ROW_SCORE_SCHEMA,
    _fold_model_sha256,
    _oof_sidecar,
    _publish_transaction,
    parser,
)


def _args() -> argparse.Namespace:
    return argparse.Namespace(
        hidden_dims=(4, 2), epochs=3, learning_rate=0.01, l2_weight=0.001,
        iou_loss_weight=1.0, threshold_loss_weight=1.0,
        monotonic_loss_weight=0.1, seed=17,
        ranking_weights=(0.1, 0.2, 0.3, 0.4), detector_blend=0.4,
    )


def _models() -> dict[int, tuple[tuple[np.ndarray, ...], tuple[np.ndarray, ...], np.ndarray, np.ndarray]]:
    result = {}
    for fold in range(5):
        result[fold] = (
            (np.full((2, 4), fold + 1, np.float32),),
            (np.full((2,), fold + 2, np.float32),),
            np.full((4,), 0.1 * fold, np.float32),
            np.full((4,), 1.0 + fold, np.float32),
        )
    return result


def _inputs(tmp_path: Path):
    scenes = np.asarray([f"4200000{fold}" for fold in range(5)])
    folds = np.arange(5, dtype=np.int8)
    dataset = tmp_path / "dataset.npz"
    dataset.write_bytes(b"sealed-dataset")
    dataset_manifest_path = tmp_path / "dataset.manifest.json"
    dataset_manifest = {
        "scenes": [
            {"scene_id": str(scene), "fold_id": int(fold)}
            for scene, fold in zip(scenes, folds)
        ]
    }
    dataset_manifest_path.write_text(json.dumps(dataset_manifest))
    values = {
        "scene_ids": scenes,
        "fold_ids": folds,
        "row_indices": np.zeros(5, dtype=np.int64),
        "prediction_scores": np.linspace(0.1, 0.5, 5, dtype=np.float32),
        "split_namespace": np.asarray("boxfusion.ca1m-native-b6.scene-folds.v1"),
        "feature_names": np.asarray(("a", "b", "c", "d")),
    }
    raw = np.tile(np.asarray((0.2, 0.3, 0.4, 0.5)), (5, 1))
    return values, dataset, dataset_manifest_path, dataset_manifest, raw


def test_oof_sidecar_binds_each_row_to_model_excluding_its_scene(tmp_path: Path):
    values, dataset, dataset_manifest_path, manifest, raw = _inputs(tmp_path)
    arrays, report = _oof_sidecar(
        values=values,
        dataset_path=dataset,
        dataset_manifest_path=dataset_manifest_path,
        dataset_manifest=manifest,
        models=_models(),
        oof_outputs=raw,
        oof_components=raw,
        quality_oof_scores=np.linspace(0.2, 0.6, 5),
        deployment_oof_scores=np.linspace(0.3, 0.7, 5),
        ranking_weights=np.asarray((0.1, 0.2, 0.3, 0.4)),
        args=_args(),
    )
    assert arrays["schema"].item() == OOF_ROW_SCORE_SCHEMA
    assert report["schema"] == OOF_ROW_SCORE_MANIFEST_SCHEMA
    assert np.array_equal(arrays["heldout_model_fold_ids"], values["fold_ids"])
    assert report["split"]["gate_train_folds"] == [2, 3, 4]
    for row in report["split"]["folds"]:
        assert set(row["heldout_scene_ids"]).isdisjoint(row["training_scene_ids"])
        assert row["training_excludes_every_heldout_scene"] is True
    assert report["fold_model_sha256"] == [
        _fold_model_sha256(_models()[fold]) for fold in range(5)
    ]


def test_oof_sidecar_rejects_row_fold_different_from_scene_fold(tmp_path: Path):
    values, dataset, dataset_manifest_path, manifest, raw = _inputs(tmp_path)
    values["fold_ids"] = np.asarray((1, 1, 2, 3, 4), dtype=np.int8)
    with pytest.raises(ValueError, match="row folds differ"):
        _oof_sidecar(
            values=values, dataset_path=dataset,
            dataset_manifest_path=dataset_manifest_path,
            dataset_manifest=manifest, models=_models(),
            oof_outputs=raw, oof_components=raw,
            quality_oof_scores=np.linspace(0.2, 0.6, 5),
            deployment_oof_scores=np.linspace(0.3, 0.7, 5),
            ranking_weights=np.asarray((0.1, 0.2, 0.3, 0.4)), args=_args(),
        )


def test_oof_cli_is_optional_for_backward_compatibility():
    args = parser().parse_args(
        ["--dataset", "d.npz", "--dataset-manifest", "d.json", "--preflight"]
    )
    assert args.oof_output is None
    assert args.oof_manifest_output is None


def test_four_artifact_transaction_rolls_back_only_owned_inode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    targets = tuple(tmp_path / f"artifact-{index}" for index in range(4))
    original = training._create_only
    calls = 0

    def fail_second(path: Path, data: bytes):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("injected publish failure")
        return original(path, data)

    monkeypatch.setattr(training, "_create_only", fail_second)
    with pytest.raises(RuntimeError, match="injected"):
        _publish_transaction(tuple((path, f"value-{i}".encode()) for i, path in enumerate(targets)))
    assert not any(path.exists() for path in targets)

    protected = targets[0]
    protected.write_bytes(b"preexisting-user-bytes")
    with pytest.raises(FileExistsError, match="already exists"):
        _publish_transaction(tuple((path, b"new") for path in targets))
    assert protected.read_bytes() == b"preexisting-user-bytes"
