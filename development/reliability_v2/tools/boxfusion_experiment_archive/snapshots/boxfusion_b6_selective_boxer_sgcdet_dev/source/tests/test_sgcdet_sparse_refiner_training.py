"""CPU tests for strict SGCDet sparse-refiner data and training contracts."""

from __future__ import annotations

import hashlib

import numpy as np
import pytest

from boxfusion.quality_score import QUALITY_FEATURE_NAMES
from boxfusion.sgcdet_local_sparse_refiner import (
    SGCDetInspiredLocalSparseRefiner,
    SGCDetLocalSparseRefinerConfig,
    load_sgcdet_sparse_refiner_checkpoint,
)
from tools.build_sgcdet_sparse_refiner_dataset import (
    METADATA_KEYS,
    SAMPLE_KEYS,
    convert_joint_arrays,
)
from tools.train_sgcdet_sparse_refiner import (
    deterministic_scene_holdout,
    load_sgcdet_sparse_refiner_dataset,
    sgcdet_sparse_refiner_loss,
    train_sgcdet_sparse_refiner,
)


def _joint_arrays(scene_ids: np.ndarray) -> dict[str, np.ndarray]:
    count = int(len(scene_ids))
    geometry = np.arange(count) % 2 == 0
    points = np.zeros((count, 5, 128, 3), dtype=np.float32)
    masks = np.zeros((count, 5, 128), dtype=np.bool_)
    masks[:, 0, :8] = True
    points[:, 0, :8, 0] = np.linspace(-0.3, 0.3, 8, dtype=np.float32)
    view_mask = masks.any(axis=2)
    view_features = np.zeros((count, 5, 9), dtype=np.float32)
    view_features[:, 0, :] = 0.5
    boxes = np.zeros((count, 6), dtype=np.float32)
    boxes[:, 3:6] = 1.0
    quality = np.full(
        (count, len(QUALITY_FEATURE_NAMES)), 0.5, dtype=np.float32
    )
    residual = np.zeros((count, 6), dtype=np.float32)
    residual[geometry, 0] = 0.1
    baseline = np.where(geometry, 0.40, 0.60).astype(np.float32)
    target = np.full(count, 0.60, dtype=np.float32)
    runtime = np.ones(count, dtype=np.bool_)
    identity = ~geometry
    candidate = np.ones(count, dtype=np.bool_)
    gt = np.zeros((count, 6), dtype=np.float32)
    gt[:, 0] = 0.30
    gt[:, 3:6] = 1.0
    return {
        "joint_points_local": points,
        "joint_point_mask": masks,
        "joint_view_features": view_features,
        "joint_view_mask": view_mask,
        "joint_local_boxes": boxes,
        "joint_quality_features": quality,
        "target_residual": residual,
        "geometry_mask": geometry.astype(np.bool_),
        "scene_ids": scene_ids.astype(np.str_),
        "result_indices": np.arange(count, dtype=np.int64),
        "track_ids": np.arange(100, 100 + count, dtype=np.int64),
        "matched_gt_index": np.zeros(count, dtype=np.int64),
        "original_iou": baseline,
        "refined_iou": target,
        "iou_gain": np.maximum(target - baseline, 0.0).astype(np.float32),
        "cross_iou50": (runtime & ~identity & candidate),
        "ap50_weight": np.where(geometry, 4.0, 1.0).astype(np.float32),
        "runtime_eligible": runtime,
        "identity_tp50": identity,
        "candidate_oracle_tp50": candidate,
        "aligned_basis": np.tile(
            np.eye(3, dtype=np.float32), (count, 1, 1)
        ),
        "original_aligned_center": np.zeros((count, 3), dtype=np.float32),
        "matched_gt_box": gt,
        "max_center_fraction": np.asarray(0.15, dtype=np.float32),
        "max_log_dimension_residual": np.asarray(
            np.log(1.25), dtype=np.float32
        ),
        "source_dataset_sha256": np.asarray("a" * 64),
    }


def _safe_dataset(tmp_path):
    scenes = np.repeat(
        np.asarray(
            [
                "scene0000_00",
                "scene0001_00",
                "scene0002_00",
                "scene0003_00",
            ]
        ),
        2,
    )
    output = convert_joint_arrays(
        _joint_arrays(scenes),
        joint_sha256="b" * 64,
        forbidden_scenes=["scene9999_00"],
    )
    path = tmp_path / "sparse.npz"
    np.savez_compressed(path, **output)
    return path, output


def test_sparse_dataset_schema_is_pickle_free_and_loadable(tmp_path):
    path, output = _safe_dataset(tmp_path)
    assert set(output) == SAMPLE_KEYS | METADATA_KEYS
    assert not any(value.dtype.hasobject for value in output.values())
    loaded = load_sgcdet_sparse_refiner_dataset(path)
    assert loaded.points_local.shape == (8, 5, 128, 3)
    assert loaded.geometry_mask.sum() == 4
    assert loaded.source_joint_dataset_sha256 == "b" * 64


def test_builder_rejects_validation_scene_overlap():
    scenes = np.repeat(
        np.asarray(["scene0000_00", "scene0001_00"]), 2
    )
    with pytest.raises(ValueError, match="overlaps forbidden"):
        convert_joint_arrays(
            _joint_arrays(scenes),
            joint_sha256="b" * 64,
            forbidden_scenes=["scene0001_00"],
        )


def test_loader_rejects_object_dtype(tmp_path):
    _, output = _safe_dataset(tmp_path)
    output["scene_ids"] = output["scene_ids"].astype(object)
    path = tmp_path / "unsafe.npz"
    np.savez_compressed(path, **output)
    with pytest.raises(ValueError, match="object"):
        load_sgcdet_sparse_refiner_dataset(path)


def test_scene_holdout_is_deterministic_and_has_no_scene_overlap():
    scene_ids = np.repeat(
        np.asarray(
            [f"scene{index:04d}_00" for index in range(10)]
        ),
        3,
    )
    first = deterministic_scene_holdout(scene_ids, 0.2, 17)
    second = deterministic_scene_holdout(scene_ids, 0.2, 17)
    assert np.array_equal(first[0], second[0])
    assert np.array_equal(first[1], second[1])
    assert set(scene_ids[first[0]]).isdisjoint(set(scene_ids[first[1]]))
    assert len(set(scene_ids[first[1]])) == 2


def _fake_loss_batch(torch):
    count = 2
    boxes = torch.zeros((count, 6), dtype=torch.float32)
    boxes[:, 3:6] = 1.0
    gt = boxes.clone()
    gt[:, 0] = 0.30
    geometry = torch.tensor([True, False])
    output = {
        "center_residual_fraction": torch.zeros(
            (count, 3), dtype=torch.float32, requires_grad=True
        ),
        "log_dimension_residual": torch.zeros(
            (count, 3), dtype=torch.float32, requires_grad=True
        ),
        "candidate_iou": torch.full(
            (count,), 0.5, dtype=torch.float32, requires_grad=True
        ),
        "improvement_probability": torch.full(
            (count,), 0.5, dtype=torch.float32, requires_grad=True
        ),
        "uncertainty": torch.full(
            (count,), 0.5, dtype=torch.float32, requires_grad=True
        ),
        "coarse_occupancy_logits": torch.zeros(
            (count, 8), dtype=torch.float32, requires_grad=True
        ),
        "coarse_occupancy_targets": torch.tensor(
            [[1, 0, 0, 0, 0, 0, 0, 0]] * count, dtype=torch.float32
        ),
        "occupancy_logits": torch.zeros(
            (count, 16), dtype=torch.float32, requires_grad=True
        ),
        "occupancy_targets": torch.tensor(
            [[1, 1] + [0] * 14] * count, dtype=torch.float32
        ),
    }
    kwargs = {
        "target_residual": torch.zeros((count, 6), dtype=torch.float32),
        "geometry_mask": geometry,
        "local_boxes": boxes,
        "matched_gt_index": torch.zeros(count, dtype=torch.int64),
        "baseline_iou": torch.tensor([0.40, 0.60]),
        "target_iou": torch.tensor([0.60, 0.60]),
        "aligned_basis": torch.eye(3).repeat(count, 1, 1),
        "original_aligned_center": torch.zeros((count, 3)),
        "matched_gt_box": gt,
        "iou_gain": torch.tensor([0.20, 0.0]),
        "cross_iou50": torch.tensor([True, False]),
        "ap50_weight": torch.tensor([4.0, 1.0]),
        "runtime_eligible": torch.ones(count, dtype=torch.bool),
        "identity_tp50": torch.tensor([False, True]),
    }
    return output, kwargs


def test_loss_has_occupancy_and_geometry_positive_only():
    torch = pytest.importorskip("torch")
    output, kwargs = _fake_loss_batch(torch)
    loss, metrics = sgcdet_sparse_refiner_loss(output, **kwargs)
    assert torch.isfinite(loss)
    assert metrics["occupancy_loss"] > 0.0
    loss.backward()
    assert output["occupancy_logits"].grad is not None

    # A negative row must not alter residual regression supervision.
    changed = dict(kwargs)
    changed["target_residual"] = kwargs["target_residual"].clone()
    changed["target_residual"][1] = 0.15
    _, changed_metrics = sgcdet_sparse_refiner_loss(output, **changed)
    torch.testing.assert_close(
        metrics["residual_loss"], changed_metrics["residual_loss"]
    )


def test_identity_checkpoint_is_atomic_and_scene_provenanced(tmp_path):
    torch = pytest.importorskip("torch")
    path, _ = _safe_dataset(tmp_path)
    output = tmp_path / "identity.pt"
    result = train_sgcdet_sparse_refiner(
        path,
        output,
        identity_only=True,
        validation_fraction=0.5,
        seed=9,
    )
    assert output.is_file()
    assert not output.with_name(output.name + ".tmp").exists()
    assert result["identity_only"] is True
    assert set(result["train_scenes"]).isdisjoint(
        set(result["validation_scenes"])
    )
    try:
        checkpoint = torch.load(output, map_location="cpu", weights_only=True)
    except TypeError:
        checkpoint = torch.load(output, map_location="cpu")
    assert checkpoint["metadata"]["scene_leakage"] is False
    assert checkpoint["metadata"]["training_dataset_sha256"] == hashlib.sha256(
        path.read_bytes()
    ).hexdigest()
    model = SGCDetInspiredLocalSparseRefiner()
    load_sgcdet_sparse_refiner_checkpoint(model, output, map_location="cpu")
    model.eval()
    data = load_sgcdet_sparse_refiner_dataset(path)
    with torch.no_grad():
        prediction = model(
            torch.from_numpy(data.points_local[:1]),
            torch.from_numpy(data.point_mask[:1]),
            torch.from_numpy(data.local_boxes[:1]),
            torch.from_numpy(data.quality_features[:1]),
            torch.from_numpy(data.view_features[:1]),
            torch.from_numpy(data.view_mask[:1]),
        )
    torch.testing.assert_close(
        prediction["center_residual_fraction"],
        torch.zeros_like(prediction["center_residual_fraction"]),
        atol=0.0,
        rtol=0.0,
    )
    torch.testing.assert_close(
        prediction["log_dimension_residual"],
        torch.zeros_like(prediction["log_dimension_residual"]),
        atol=0.0,
        rtol=0.0,
    )


def test_one_epoch_cpu_training_smoke(tmp_path):
    pytest.importorskip("torch")
    path, _ = _safe_dataset(tmp_path)
    output = tmp_path / "trained.pt"
    config = SGCDetLocalSparseRefinerConfig(
        coarse_hidden_dim=8,
        coarse_embedding_dim=8,
        occupancy_hidden_dim=8,
        selected_hidden_dim=8,
        selected_embedding_dim=8,
        head_hidden_dim=8,
    )
    result = train_sgcdet_sparse_refiner(
        path,
        output,
        config=config,
        epochs=1,
        batch_size=4,
        validation_fraction=0.5,
        seed=11,
    )
    assert output.is_file()
    assert result["identity_only"] is False
    assert np.isfinite(result["best_validation_loss"])
    assert result["scene_leakage"] is False
