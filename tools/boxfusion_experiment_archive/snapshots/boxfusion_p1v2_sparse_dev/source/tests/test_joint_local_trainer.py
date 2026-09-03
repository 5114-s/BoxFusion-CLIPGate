"""CPU-only tests for realized-candidate joint losses and checkpoints."""

from __future__ import annotations

import hashlib

import numpy as np
import pytest

import tools.train_joint_local_head as trainer
from boxfusion.joint_local_head import (
    JointLocalHeadConfig,
    MultiViewJointLocalHead,
    load_joint_local_head_checkpoint,
)
from boxfusion.quality_score import QUALITY_FEATURE_NAMES


def _fake_output(torch, center_fraction):
    batch = int(center_fraction.shape[0])
    log_dimensions = torch.zeros(
        (batch, 3), dtype=torch.float32, requires_grad=True
    )
    quality = torch.tensor(
        [[[0.50, 0.80, 0.60, 0.40]] * 2] * batch,
        dtype=torch.float32,
        requires_grad=True,
    )
    log_variance = torch.zeros(
        (batch, 2), dtype=torch.float32, requires_grad=True
    )
    return {
        "center_residual_fraction": center_fraction,
        "log_dimension_residual": log_dimensions,
        "improvement_probability": torch.full(
            (batch,), 0.5, dtype=torch.float32, requires_grad=True
        ),
        "quality_components": quality,
        "quality_log_variance": log_variance,
        "quality_uncertainty": torch.exp(0.5 * log_variance),
    }


def _loss_kwargs(torch, batch=2):
    boxes = torch.zeros((batch, 6), dtype=torch.float32)
    boxes[:, 3:6] = 1.0
    gt = torch.zeros((batch, 6), dtype=torch.float32)
    gt[:, 0] = 0.30
    gt[:, 3:6] = 1.0
    return {
        "target_residual": torch.zeros((batch, 6), dtype=torch.float32),
        "geometry_mask": torch.ones(batch, dtype=torch.bool),
        "local_boxes": boxes,
        "matched_gt_index": torch.zeros(batch, dtype=torch.int64),
        "original_iou": torch.full((batch,), 0.40),
        "refined_iou": torch.full((batch,), 0.60),
        "aligned_basis": torch.eye(3).repeat(batch, 1, 1),
        "original_aligned_center": torch.zeros((batch, 3)),
        "matched_gt_box": gt,
        "iou_gain_target": torch.full((batch,), 0.20),
        "cross_iou50": torch.ones(batch, dtype=torch.bool),
        "ap50_weight": torch.ones(batch),
        "runtime_eligible": torch.ones(batch, dtype=torch.bool),
        "identity_tp50": torch.zeros(batch, dtype=torch.bool),
    }


def test_candidate_quality_uses_current_realized_iou_and_ap50_crossing():
    torch = pytest.importorskip("torch")
    good_center = torch.zeros(
        (2, 3), dtype=torch.float32, requires_grad=True
    )
    bad_center = torch.zeros(
        (2, 3), dtype=torch.float32, requires_grad=True
    )
    bad_center = bad_center.clone()
    bad_center[:, 0] = -0.15
    kwargs = _loss_kwargs(torch)
    good_loss, good_metrics = trainer.joint_local_loss(
        _fake_output(torch, good_center), **kwargs
    )
    bad_loss, bad_metrics = trainer.joint_local_loss(
        _fake_output(torch, bad_center), **kwargs
    )
    assert good_metrics["realized_candidate_iou"] > 0.50
    assert bad_metrics["realized_candidate_iou"] < 0.50
    assert good_metrics["cross_iou50_loss"] == pytest.approx(0.0)
    assert bad_metrics["cross_iou50_loss"] > 0.0
    assert bad_loss > good_loss

    # Oracle refined_iou is diagnostic only; candidate quality is tied to the
    # current model residual's realized IoU and therefore total loss is
    # invariant to this oracle field.
    changed = dict(kwargs)
    changed["refined_iou"] = torch.full((2,), 0.95)
    changed_loss, changed_metrics = trainer.joint_local_loss(
        _fake_output(
            torch,
            torch.zeros(
                (2, 3), dtype=torch.float32, requires_grad=True
            ),
        ),
        **changed,
    )
    torch.testing.assert_close(changed_loss, good_loss)
    assert changed_metrics["oracle_realized_iou_gap"] != pytest.approx(
        float(good_metrics["oracle_realized_iou_gap"])
    )


def test_joint_uncertainty_loss_is_finite_and_backpropagates():
    torch = pytest.importorskip("torch")
    output = _fake_output(
        torch,
        torch.zeros((2, 3), dtype=torch.float32, requires_grad=True),
    )
    loss, metrics = trainer.joint_local_loss(
        output, **_loss_kwargs(torch)
    )
    assert torch.isfinite(loss)
    assert torch.isfinite(metrics["uncertainty_loss"])
    loss.backward()
    assert output["quality_log_variance"].grad is not None
    assert torch.isfinite(output["quality_log_variance"].grad).all()


def _synthetic_training_data() -> trainer.JointLocalTrainingData:
    sample_count = 8
    geometry = np.tile(
        np.asarray([True, False], dtype=np.bool_), 4
    )
    points = np.zeros((sample_count, 5, 128, 3), dtype=np.float32)
    point_mask = np.zeros((sample_count, 5, 128), dtype=np.bool_)
    point_mask[:, 0, :4] = True
    points[:, 0, :4, 0] = np.asarray(
        [-0.3, -0.1, 0.1, 0.3], dtype=np.float32
    )
    view_mask = point_mask.any(axis=2)
    view_features = np.zeros((sample_count, 5, 9), dtype=np.float32)
    view_features[:, 0] = np.asarray(
        [0.9, 0.8, 0.8, 0.7, 4.0 / 128.0, 0.0, 0.5, 0.5, 0.5],
        dtype=np.float32,
    )
    boxes = np.zeros((sample_count, 6), dtype=np.float32)
    boxes[:, 3:6] = 1.0
    quality = np.full(
        (sample_count, len(QUALITY_FEATURE_NAMES)),
        0.5,
        dtype=np.float32,
    )
    target = np.zeros((sample_count, 6), dtype=np.float32)
    target[geometry, 0] = 0.1
    original = np.where(geometry, 0.40, 0.60).astype(np.float32)
    refined = np.full(sample_count, 0.60, dtype=np.float32)
    gt = np.zeros((sample_count, 6), dtype=np.float32)
    gt[:, 0] = 0.30
    gt[:, 3:6] = 1.0
    scene_ids = np.repeat(
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
    training_hash = hashlib.sha256(
        (
            "\n".join(sorted(np.unique(scene_ids).tolist())) + "\n"
        ).encode("utf-8")
    ).hexdigest()
    return trainer.JointLocalTrainingData(
        points_local=points,
        point_mask=point_mask,
        view_features=view_features,
        view_mask=view_mask,
        local_boxes=boxes,
        quality_features=quality,
        target_residual=target,
        geometry_mask=geometry,
        scene_ids=scene_ids,
        matched_gt_index=np.zeros(sample_count, dtype=np.int64),
        original_iou=original,
        refined_iou=refined,
        aligned_basis=np.tile(
            np.eye(3, dtype=np.float32), (sample_count, 1, 1)
        ),
        original_aligned_center=np.zeros(
            (sample_count, 3), dtype=np.float32
        ),
        matched_gt_box=gt,
        iou_gain=np.maximum(refined - original, 0.0).astype(np.float32),
        cross_iou50=geometry.copy(),
        ap50_weight=np.where(geometry, 6.0, 1.0).astype(np.float32),
        runtime_eligible=geometry.copy(),
        identity_tp50=~geometry,
        candidate_oracle_tp50=np.ones(
            sample_count, dtype=np.bool_
        ),
        max_center_fraction=0.15,
        max_log_dimension_residual=float(np.log(1.25)),
        source_dataset_sha256="1" * 64,
        forbidden_scene_count=1,
        forbidden_scene_sha256="2" * 64,
        training_scene_sha256=training_hash,
        points_per_view=128,
    )


def test_cpu_training_smoke_writes_strict_checkpoint(
    tmp_path, monkeypatch
):
    torch = pytest.importorskip("torch")
    data = _synthetic_training_data()
    dataset = tmp_path / "synthetic.npz"
    dataset.write_bytes(b"deterministic synthetic dataset marker")
    monkeypatch.setattr(
        trainer, "load_joint_local_dataset", lambda _: data
    )
    output = tmp_path / "joint.pt"
    config = JointLocalHeadConfig(
        point_hidden_dim=8,
        point_embedding_dim=8,
        view_embedding_dim=8,
        head_hidden_dim=8,
    )
    result = trainer.train_joint_local_head(
        dataset,
        output,
        config=config,
        epochs=2,
        batch_size=4,
        validation_fraction=0.5,
        seed=17,
    )
    assert output.is_file()
    assert result["scene_leakage"] is False
    assert set(result["train_scenes"]).isdisjoint(
        result["validation_scenes"]
    )
    assert result["best_epoch"] in {0, 1}
    model = MultiViewJointLocalHead(config).cpu()
    metadata = load_joint_local_head_checkpoint(
        model, output, map_location="cpu"
    )
    assert metadata["device"] == "cpu"
    assert metadata["points_per_view"] == 128
    assert (
        metadata["candidate_quality_target"]
        == "realized_candidate_iou_detached"
    )
    assert metadata["scene_leakage"] is False
    assert all(not parameter.is_cuda for parameter in model.parameters())

