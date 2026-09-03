import hashlib
import inspect
from dataclasses import replace

import numpy as np
import pytest

import tools.train_oriented_box_refiner as trainer
from boxfusion.oriented_box_refiner import OrientedBoxRefinerConfig
from boxfusion.quality_score import QUALITY_FEATURE_NAMES
from tools.build_oriented_refiner_dataset import (
    AP50_DATASET_FORMAT_VERSION,
    DATASET_SCHEMA,
    STRICT_PROVENANCE_EXPECTED,
    TARGET_LINE_SEARCH_ALPHAS,
    V2_METADATA_KEYS,
    V2_SAMPLE_KEYS,
)


def test_local_net_proxy_uses_one_denominator_and_penalizes_many_drops():
    # The rejected formulation was 1/1 - 50/100 = +0.5.  Net events over
    # one eligible denominator must instead correctly be negative.
    value = trainer.local_net_tp50_proxy(
        cross_success_count=1,
        drop50_count=50,
        eligible_matched_count=101,
    )
    assert value == pytest.approx(-49.0 / 101.0)
    assert value < 0.0


def test_ap50_importance_is_used_once_in_loss_not_in_sampling():
    torch = pytest.importorskip("torch")
    assert "sample_weights" not in inspect.signature(
        trainer.balanced_epoch_indices
    ).parameters

    output = {
        "center_residual_fraction": torch.tensor(
            [[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]]
        ),
        "log_dimension_residual": torch.zeros((2, 3)),
        "quality": torch.tensor([0.9, 0.9]),
    }
    common = {
        "target_residual": torch.zeros((2, 6)),
        "quality_target": torch.tensor([0.95, 0.95]),
        "geometry_mask": torch.tensor([True, True]),
        "objective": "ap50",
        "local_boxes": torch.tensor(
            [
                [0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
                [0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
            ]
        ),
        "original_iou": torch.tensor([0.4, 0.4]),
        "aligned_basis": torch.eye(3).repeat(2, 1, 1),
        "original_aligned_center": torch.zeros((2, 3)),
        "matched_gt_box": torch.tensor(
            [
                [0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
                [0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
            ]
        ),
        "iou_gain_target": torch.tensor([0.6, 0.6]),
        "cross_iou50": torch.tensor([True, True]),
        "ap50_weight": torch.tensor([1.0, 3.0]),
        "runtime_eligible": torch.tensor([True, True]),
        "identity_tp50": torch.tensor([False, False]),
        "candidate_oracle_tp50": torch.tensor([True, True]),
        "center_weight": 1.0,
        "dimension_weight": 0.0,
        "quality_weight": 0.0,
        "iou_gain_weight": 0.0,
        "cross_iou50_weight": 0.0,
    }
    loss, metrics = trainer.oriented_refiner_loss(output, **common)

    # Smooth-L1 per-sample values are .5 and 1.5.  One application of
    # weights [1,3] gives (0.5*1 + 1.5*3) / 4 = 1.25.  Squared weighting
    # would be 1.4 and must never appear.
    assert metrics["center_loss"] == pytest.approx(1.25)
    assert loss == pytest.approx(1.25)
    assert loss != pytest.approx(1.4)


def test_local_tp50_proxy_masks_full_scene_flags_by_runtime_eligibility():
    torch = pytest.importorskip("torch")
    _, metrics = trainer.oriented_refiner_loss(
        {
            "center_residual_fraction": torch.tensor(
                [[2.0, 0.0, 0.0], [2.0, 0.0, 0.0]]
            ),
            "log_dimension_residual": torch.zeros((2, 3)),
            "quality": torch.tensor([0.9, 0.9]),
        },
        target_residual=torch.zeros((2, 6)),
        quality_target=torch.zeros(2),
        geometry_mask=torch.tensor([False, False]),
        objective="ap50",
        local_boxes=torch.tensor(
            [
                [0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
                [0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
            ]
        ),
        original_iou=torch.tensor([1.0, 1.0]),
        aligned_basis=torch.eye(3).repeat(2, 1, 1),
        original_aligned_center=torch.zeros((2, 3)),
        matched_gt_box=torch.tensor(
            [
                [0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
                [0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
            ]
        ),
        iou_gain_target=torch.zeros(2),
        cross_iou50=torch.tensor([False, False]),
        ap50_weight=torch.ones(2),
        runtime_eligible=torch.tensor([True, False]),
        # Scene-aware evaluator flags are intentionally retained on the
        # ineligible second row.  Only the proxy/event mask excludes it.
        identity_tp50=torch.tensor([True, True]),
        candidate_oracle_tp50=torch.tensor([True, True]),
    )
    assert metrics["eligible_matched_count"] == 1
    assert metrics["identity_tp50_count"] == 1
    assert metrics["drop50_count"] == 1


@pytest.mark.parametrize("objective", ["improvement", "ap50"])
def test_schema_v2_loader_accepts_strict_k5_for_both_objectives(
    tmp_path, objective
):
    sample_count = 4
    point_count = 4
    geometry = np.asarray([True, False, True, False], dtype=np.bool_)
    local_boxes = np.zeros((sample_count, 6), dtype=np.float32)
    local_boxes[:, 3:6] = 1.0
    matched_gt_box = np.zeros((sample_count, 6), dtype=np.float32)
    matched_gt_box[geometry, 3:6] = 1.0
    identity = np.asarray([False, True, False, True], dtype=np.bool_)
    candidate = np.asarray([True, True, True, True], dtype=np.bool_)
    runtime_eligible = np.asarray(
        [True, False, True, False], dtype=np.bool_
    )
    training_scenes = ("scene0000_00", "scene0001_00")
    training_digest = hashlib.sha256(
        ("\n".join(training_scenes) + "\n").encode("utf-8")
    ).hexdigest()
    arrays = {
        "points_local": np.zeros(
            (sample_count, point_count, 3), dtype=np.float32
        ),
        "point_mask": np.ones(
            (sample_count, point_count), dtype=np.bool_
        ),
        "local_boxes": local_boxes,
        "quality_features": np.full(
            (sample_count, len(QUALITY_FEATURE_NAMES)),
            0.5,
            dtype=np.float32,
        ),
        "target_residual": np.zeros(
            (sample_count, 6), dtype=np.float32
        ),
        "quality_target": (
            np.asarray([0.95, 0.0, 0.95, 0.0], dtype=np.float32)
            if objective == "ap50"
            else geometry.astype(np.float32)
        ),
        "geometry_mask": geometry,
        "scene_ids": np.asarray(
            [
                "scene0000_00",
                "scene0000_00",
                "scene0001_00",
                "scene0001_00",
            ]
        ),
        "original_iou": np.asarray(
            [0.4, 0.6, 0.4, 0.6], dtype=np.float32
        ),
        "refined_iou": np.asarray(
            [0.6, 0.6, 0.6, 0.6], dtype=np.float32
        ),
        "matched_gt_index": np.asarray([0, 1, 0, 1], dtype=np.int64),
        "target_center_local_unclipped": np.zeros(
            (sample_count, 3), dtype=np.float32
        ),
        "target_dimensions_local_unclipped": np.ones(
            (sample_count, 3), dtype=np.float32
        ),
        "basis_world": np.tile(
            np.eye(3, dtype=np.float32), (sample_count, 1, 1)
        ),
        "result_indices": np.asarray([0, 1, 0, 1], dtype=np.int64),
        "track_ids": np.asarray([0, 1, 0, 1], dtype=np.int64),
        "aligned_basis": np.tile(
            np.eye(3, dtype=np.float32), (sample_count, 1, 1)
        ),
        "original_aligned_center": np.zeros(
            (sample_count, 3), dtype=np.float32
        ),
        "matched_gt_box": matched_gt_box,
        "iou_gain": np.asarray(
            [0.2, 0.0, 0.2, 0.0], dtype=np.float32
        ),
        "cross_iou50": (
            runtime_eligible & ~identity & candidate
        ),
        "near_iou50": np.full(sample_count, 0.5, dtype=np.float32),
        "ap50_weight": np.full(sample_count, 2.0, dtype=np.float32),
        "runtime_eligible": runtime_eligible,
        "selected_view_counts": np.asarray(
            [5, 1, 5, 1], dtype=np.int64
        ),
        # The ineligible rows deliberately retain the full scene-level flags.
        "identity_tp50": identity,
        "candidate_oracle_tp50": candidate,
        "schema": np.asarray(DATASET_SCHEMA),
        "format_version": np.asarray(
            AP50_DATASET_FORMAT_VERSION, dtype=np.int64
        ),
        "coordinate_frame": np.asarray("box_local"),
        "quality_feature_names": np.asarray(
            QUALITY_FEATURE_NAMES, dtype=np.str_
        ),
        "max_center_fraction": np.asarray(0.15, dtype=np.float32),
        "max_log_dimension_residual": np.asarray(
            np.log(1.25), dtype=np.float32
        ),
        "objective": np.asarray(objective),
        "strict_k5_diagnostics": np.asarray(True, dtype=np.bool_),
        "expected_top_k_views": np.asarray(5, dtype=np.int64),
        "min_runtime_views": np.asarray(2, dtype=np.int64),
        "min_runtime_points": np.asarray(128, dtype=np.int64),
        "runtime_minimum_extent": np.asarray(0.4, dtype=np.float32),
        "near_iou50_band": np.asarray(0.15, dtype=np.float32),
        "gain_cap": np.asarray(0.25, dtype=np.float32),
        "gain_sample_weight": np.asarray(2.0, dtype=np.float32),
        "cross_iou50_sample_weight": np.asarray(4.0, dtype=np.float32),
        "near_iou50_sample_weight": np.asarray(2.0, dtype=np.float32),
        "min_match_iou": np.asarray(0.15, dtype=np.float32),
        "improvement_epsilon": np.asarray(1e-4, dtype=np.float32),
        "target_line_search_alphas": np.asarray(
            TARGET_LINE_SEARCH_ALPHAS, dtype=np.float32
        ),
        "forbidden_scene_count": np.asarray(1, dtype=np.int64),
        "forbidden_scene_sha256": np.asarray("0" * 64),
        "training_scene_count": np.asarray(2, dtype=np.int64),
        "training_scene_sha256": np.asarray(training_digest),
    }
    arrays.update(
        {
            name: np.asarray(value)
            for name, value in STRICT_PROVENANCE_EXPECTED.items()
        }
    )
    assert set(arrays) == set(V2_SAMPLE_KEYS | V2_METADATA_KEYS)
    path = tmp_path / f"{objective}.npz"
    np.savez_compressed(path, **arrays)

    loaded = trainer.load_oriented_refiner_dataset(path)
    assert loaded.objective == objective
    np.testing.assert_array_equal(loaded.identity_tp50, identity)
    np.testing.assert_array_equal(
        loaded.candidate_oracle_tp50, candidate
    )


def _minimal_improvement_data():
    sample_count = 4
    point_count = 4
    geometry = np.asarray([True, False, True, False], dtype=np.bool_)
    local_boxes = np.zeros((sample_count, 6), dtype=np.float32)
    local_boxes[:, 3:6] = 1.0
    matched_gt_box = np.zeros((sample_count, 6), dtype=np.float32)
    matched_gt_box[:, 3:6] = 1.0
    return trainer.OrientedRefinerTrainingData(
        points_local=np.zeros(
            (sample_count, point_count, 3), dtype=np.float32
        ),
        point_mask=np.ones(
            (sample_count, point_count), dtype=np.bool_
        ),
        local_boxes=local_boxes,
        quality_features=np.full(
            (sample_count, len(QUALITY_FEATURE_NAMES)),
            0.5,
            dtype=np.float32,
        ),
        target_residual=np.zeros((sample_count, 6), dtype=np.float32),
        quality_target=geometry.astype(np.float32),
        geometry_mask=geometry,
        scene_ids=np.asarray(
            [
                "scene0000_00",
                "scene0000_00",
                "scene0001_00",
                "scene0001_00",
            ]
        ),
        matched_gt_index=np.asarray([0, -1, 0, -1], dtype=np.int64),
        original_iou=np.asarray([0.4, 0.0, 0.4, 0.0], dtype=np.float32),
        refined_iou=np.asarray([0.6, 0.0, 0.6, 0.0], dtype=np.float32),
        aligned_basis=np.tile(
            np.eye(3, dtype=np.float32), (sample_count, 1, 1)
        ),
        original_aligned_center=np.zeros(
            (sample_count, 3), dtype=np.float32
        ),
        matched_gt_box=matched_gt_box,
        iou_gain=np.asarray([0.2, 0.0, 0.2, 0.0], dtype=np.float32),
        cross_iou50=np.asarray([True, False, True, False]),
        near_iou50=np.zeros(sample_count, dtype=np.float32),
        ap50_weight=np.ones(sample_count, dtype=np.float32),
        runtime_eligible=np.ones(sample_count, dtype=np.bool_),
        identity_tp50=np.zeros(sample_count, dtype=np.bool_),
        candidate_oracle_tp50=np.asarray(
            [True, False, True, False], dtype=np.bool_
        ),
        objective="improvement",
        max_center_fraction=0.15,
        max_log_dimension_residual=float(np.log(1.25)),
    )


def test_returned_metrics_and_best_epoch_match_selected_checkpoint(
    tmp_path, monkeypatch
):
    torch = pytest.importorskip("torch")
    data = _minimal_improvement_data()
    monkeypatch.setattr(
        trainer, "load_oriented_refiner_dataset", lambda _: data
    )

    epoch_metrics = iter(
        [
            {"loss": 10.0, "marker": 100.0},
            {"loss": 3.0, "marker": 30.0},
            {"loss": 20.0, "marker": 200.0},
            {"loss": 1.0, "marker": 10.0},
            {"loss": 30.0, "marker": 300.0},
            {"loss": 2.0, "marker": 20.0},
        ]
    )

    def fake_run_epoch(*args, **kwargs):
        return next(epoch_metrics)

    monkeypatch.setattr(trainer, "_run_epoch", fake_run_epoch)
    checkpoint = tmp_path / "best.pt"
    result = trainer.train_oriented_box_refiner(
        "ignored.npz",
        checkpoint,
        objective="improvement",
        config=OrientedBoxRefinerConfig(
            point_hidden_dim=8,
            point_embedding_dim=8,
            head_hidden_dim=8,
        ),
        epochs=3,
        batch_size=2,
        validation_fraction=0.5,
        seed=7,
    )

    assert checkpoint.is_file()
    assert result["best_epoch"] == 1
    assert result["best_validation_loss"] == pytest.approx(1.0)
    assert result["train"] == {"loss": 20.0, "marker": 200.0}
    assert result["validation"] == {"loss": 1.0, "marker": 10.0}
    assert result["train"] != {"loss": 30.0, "marker": 300.0}
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert set(payload) == {
        "schema",
        "format_version",
        "coordinate_frame",
        "config",
        "state_dict",
    }


def test_ap50_checkpoint_uses_validation_loss_not_ungated_proxy(
    tmp_path, monkeypatch
):
    pytest.importorskip("torch")
    data = replace(_minimal_improvement_data(), objective="ap50")
    monkeypatch.setattr(
        trainer, "load_oriented_refiner_dataset", lambda _: data
    )
    epoch_metrics = iter(
        [
            {"loss": 10.0, "marker": 100.0},
            {
                "loss": 1.0,
                "marker": 10.0,
                "local_net_tp50_proxy": 0.0,
            },
            {"loss": 20.0, "marker": 200.0},
            {
                "loss": 2.0,
                "marker": 20.0,
                "local_net_tp50_proxy": 1.0,
            },
        ]
    )
    monkeypatch.setattr(
        trainer, "_run_epoch", lambda *args, **kwargs: next(epoch_metrics)
    )
    result = trainer.train_oriented_box_refiner(
        "ignored.npz",
        tmp_path / "best_ap50.pt",
        objective="ap50",
        config=OrientedBoxRefinerConfig(
            point_hidden_dim=8,
            point_embedding_dim=8,
            head_hidden_dim=8,
        ),
        epochs=2,
        batch_size=2,
        validation_fraction=0.5,
        seed=7,
    )
    assert result["best_epoch"] == 0
    assert result["best_validation_loss"] == pytest.approx(1.0)
    assert result["best_validation_ap50_proxy"] == pytest.approx(0.0)
    assert result["validation"]["marker"] == pytest.approx(10.0)
    assert "diagnostic only" in result["checkpoint_selection_note"]


def test_ap50_equality_boundary_is_not_counted_as_success(monkeypatch):
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(
        trainer,
        "differentiable_aligned_aabb_iou",
        lambda *args, **kwargs: torch.tensor([0.50]),
    )
    _, metrics = trainer.oriented_refiner_loss(
        {
            "center_residual_fraction": torch.zeros((1, 3)),
            "log_dimension_residual": torch.zeros((1, 3)),
            "quality": torch.tensor([0.9]),
        },
        target_residual=torch.zeros((1, 6)),
        quality_target=torch.tensor([0.95]),
        geometry_mask=torch.tensor([True]),
        objective="ap50",
        local_boxes=torch.tensor(
            [[0.0, 0.0, 0.0, 1.0, 1.0, 1.0]]
        ),
        original_iou=torch.tensor([0.4]),
        aligned_basis=torch.eye(3).unsqueeze(0),
        original_aligned_center=torch.zeros((1, 3)),
        matched_gt_box=torch.tensor(
            [[0.0, 0.0, 0.0, 1.0, 1.0, 1.0]]
        ),
        iou_gain_target=torch.tensor([0.1]),
        cross_iou50=torch.tensor([True]),
        ap50_weight=torch.ones(1),
        runtime_eligible=torch.tensor([True]),
        identity_tp50=torch.tensor([False]),
        candidate_oracle_tp50=torch.tensor([True]),
    )
    assert trainer.AP50_LOSS_TARGET > trainer.AP50_EVALUATOR_THRESHOLD
    assert metrics["cross_iou50_loss"] > 0.0
    assert metrics["cross50_success_count"] == 0
