"""Tests for the leakage-safe P1G geometry trainer."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from boxfusion.residual_proposal import P1S_HEAD_SCHEMA, P1_FEATURE_NAMES
from boxfusion.p1_geometry_refiner import load_p1g_checkpoint
from tools.train_p1v2_residual_head import (
    P1V2TrainingData,
    SceneTrainingContext,
)
from tools import train_p1g_geometry_refiner as trainer


def _sha(character: str = "a") -> str:
    return character * 64


def _encoded_dataset() -> trainer.EncodedPositiveDataset:
    scenes = []
    generator = torch.Generator().manual_seed(7)
    for index, scene_id in enumerate(
        ("scene0000_00", "scene0001_00", "scene0002_00")
    ):
        hidden = torch.randn((4, 5), generator=generator)
        anchors = torch.full((4, 3), float(index))
        targets = torch.cat(
            (
                anchors + 0.1,
                torch.ones((4, 3)) * (1.0 + 0.1 * index),
            ),
            dim=1,
        )
        scenes.append(
            trainer.EncodedPositiveScene(
                scene_id=scene_id,
                hidden=hidden,
                frozen_p1s_raw_regression=torch.zeros((4, 6)),
                anchor_centers=anchors,
                target_boxes_aligned=targets,
                axis_alignment=torch.eye(4).reshape(1, 4, 4).repeat(
                    len(hidden), 1, 1
                ),
            )
        )
    return trainer.EncodedPositiveDataset(tuple(scenes), hidden_dim=5)


def _valid_metrics() -> dict:
    cal = {
        "role": "cal",
        "positive_anchor_count": 4,
        "decoded_aligned_fraction_iou_gt_0p5": 0.25,
        "decoded_aligned_mean_iou": 0.4,
    }
    audit = dict(cal, role="audit")
    return {
        "selection": {
            "primary": trainer.P1G_SELECTION_PRIMARY,
            "secondary": trainer.P1G_SELECTION_SECONDARY,
            "comparison": "lexicographic_max",
            "best_epoch": 0,
            "best_key": [0.25, 0.4],
            "audit_used_for_selection": False,
        },
        "fit_positive_anchor_count": 4,
        "cal_positive_anchor_count": 4,
        "audit_positive_anchor_count": 4,
        "best_calibration": cal,
        "audit": audit,
        "audit_evaluation_count": 1,
        "history": [],
    }


def _valid_checkpoint_parts(tmp_path: Path, hidden_dim: int = 5):
    source = tmp_path / "source.pt"
    source.write_bytes(b"source")
    scene_list = tmp_path / "scenes.txt"
    scene_list.write_text("scene0000_00\n", encoding="utf-8")
    source_p1s = {
        "checkpoint": str(source),
        "checkpoint_sha256": trainer.sha256_file(source),
        "schema": P1S_HEAD_SCHEMA,
        "model_config_sha256": _sha("b"),
        "dataset_fingerprint_sha256": _sha("c"),
    }
    training_config = {
        "schema": trainer.P1G_TRAINING_SCHEMA,
        "frozen_encoder": True,
        "frozen_objectness": True,
        "trainable_parameters": [
            "correction.weight",
            "correction.bias",
        ],
    }
    overlaps = {
        "fit_cal": [],
        "fit_audit": [],
        "cal_audit": [],
        "fit_full_val": [],
        "cal_full_val": [],
        "audit_full_val": [],
    }
    provenance = {
        "p1s_checkpoint_sha256": source_p1s["checkpoint_sha256"],
        "forbidden_overlap": [],
        "forbidden_scene_list_sha256": trainer.sha256_file(scene_list),
        "dataset_fingerprint_sha256": _sha("c"),
        "train_scene_list": str(scene_list),
        "train_scene_list_sha256": trainer.sha256_file(scene_list),
        "train_scene_ids": [
            "scene0000_00",
            "scene0001_00",
            "scene0002_00",
        ],
        "fit_scene_list": str(scene_list),
        "fit_scene_list_sha256": trainer.sha256_file(scene_list),
        "fit_scene_ids": ["scene0000_00"],
        "cal_scene_list": str(scene_list),
        "cal_scene_list_sha256": trainer.sha256_file(scene_list),
        "cal_scene_ids": ["scene0001_00"],
        "audit_scene_list": str(scene_list),
        "audit_scene_list_sha256": trainer.sha256_file(scene_list),
        "audit_scene_ids": ["scene0002_00"],
        "full_val_scene_list": str(scene_list),
        "full_val_scene_list_sha256": trainer.sha256_file(scene_list),
        "full_val_scene_ids": ["scene0700_00"],
        "full_val_scene_count": 1,
        "split_overlaps": overlaps,
        "unused_train_scene_ids": [],
        "b6_checkpoint": str(source),
        "b6_checkpoint_sha256": trainer.sha256_file(source),
        "source_collection_binding": {"verified": True},
        "diagnostics_root": "/diagnostics",
        "prediction_root": "/predictions",
        "gt_root": "/gt",
        "scans_root": "/scans",
    }
    head = trainer.P1GeometryRegressionHead(hidden_dim)
    decoder = {
        "encoding": trainer.P1G_REGRESSION_ENCODING,
        "adapter_epsilon": 1e-6,
        "max_center_offset": 1.0,
        "min_box_extent": 0.08,
        "max_box_extent": 4.0,
    }
    return head, decoder, training_config, source_p1s, provenance


def test_geometry_head_is_exactly_zero_initialized_residual_correction():
    head = trainer.P1GeometryRegressionHead(hidden_dim=5)
    assert tuple(name for name, _ in head.named_parameters()) == (
        "correction.weight",
        "correction.bias",
    )
    correction = head(torch.randn((7, 5)))
    torch.testing.assert_close(correction, torch.zeros_like(correction))


def test_split_protocol_requires_pairwise_and_complete_val_disjointness():
    clean = trainer.validate_split_protocol(
        dataset_scene_ids=(
            "scene0000_00",
            "scene0001_00",
            "scene0002_00",
        ),
        fit_scene_ids=("scene0000_00",),
        cal_scene_ids=("scene0001_00",),
        audit_scene_ids=("scene0002_00",),
        full_val_scene_ids=("scene0700_00",),
    )
    assert all(value == [] for value in clean.values())

    with pytest.raises(ValueError, match="fit_cal"):
        trainer.validate_split_protocol(
            dataset_scene_ids=("scene0000_00", "scene0001_00"),
            fit_scene_ids=("scene0000_00",),
            cal_scene_ids=("scene0000_00",),
            audit_scene_ids=("scene0001_00",),
            full_val_scene_ids=("scene0700_00",),
        )
    with pytest.raises(ValueError, match="audit_full_val"):
        trainer.validate_split_protocol(
            dataset_scene_ids=(
                "scene0000_00",
                "scene0001_00",
                "scene0002_00",
            ),
            fit_scene_ids=("scene0000_00",),
            cal_scene_ids=("scene0001_00",),
            audit_scene_ids=("scene0002_00",),
            full_val_scene_ids=("scene0002_00",),
        )


class _ContextAwareFrozenEncoder(nn.Module):
    hidden_dim = 2

    def __init__(self) -> None:
        super().__init__()
        self.marker = nn.Parameter(torch.zeros(()), requires_grad=False)
        self.regression = nn.Linear(2, 6)
        with torch.no_grad():
            self.regression.weight.zero_()
            self.regression.bias.copy_(
                torch.tensor([0.1, -0.2, 0.3, 0.0, 0.1, -0.1])
            )
        for parameter in self.regression.parameters():
            parameter.requires_grad_(False)
        self.seen_rows: list[int] = []

    def encode(
        self, features: torch.Tensor, coordinates: torch.Tensor
    ) -> torch.Tensor:
        self.seen_rows.append(len(features))
        # Every row depends on the complete snapshot mean.  This makes a
        # positive-before-encode implementation observably incorrect.
        mean = features[:, :1].mean(dim=0, keepdim=True)
        return torch.cat((features[:, :1], mean.expand(len(features), 1)), 1)


def test_precompute_encodes_complete_snapshot_before_positive_selection(
    tmp_path: Path,
):
    scene = SceneTrainingContext(
        scene_id="scene0000_00",
        features=np.asarray(
            [[1.0] + [0.0] * 13, [3.0] + [0.0] * 13], dtype=np.float32
        ),
        coordinates=np.asarray([[0, 0, 0], [1, 0, 0]], dtype=np.int32),
        centers_world=np.asarray(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32
        ),
        offsets=np.asarray([0, 2], dtype=np.int64),
        frame_ids=np.asarray([0], dtype=np.int64),
        provider_steps=np.asarray([0], dtype=np.int64),
        objectness=np.asarray([1.0, 0.0], dtype=np.float32),
        regression=np.asarray(
            # Deliberately unrelated legacy enclosing-world target.  P1G
            # must ignore this row and recover original aligned GT index 1.
            [[9.0, 8.0, 7.0, 1.0, 1.0, 1.0], [0.0] * 6],
            dtype=np.float32,
        ),
        assigned_gt=np.asarray([0, -1], dtype=np.int64),
        loss_mask=np.asarray([True, True]),
        feature_names=tuple(P1_FEATURE_NAMES),
        voxel_size=0.08,
        diagnostic_path=tmp_path / "diagnostic.npz",
    )
    data = P1V2TrainingData(
        scenes=(scene,),
        feature_names=tuple(P1_FEATURE_NAMES),
        scene_summaries=(
            {
                "scene_id": "scene0000_00",
                "residual_ground_truth_indices": [1],
            },
        ),
        dataset_fingerprint_sha256=_sha("d"),
    )
    gt_root = tmp_path / "gt"
    gt_root.mkdir()
    np.save(
        gt_root / "scene0000_00_bbox.npy",
        np.asarray(
            [
                [20.0, 20.0, 20.0, 3.0, 3.0, 3.0],
                [4.0, 5.0, 6.0, 0.5, 1.0, 2.0],
            ],
            dtype=np.float32,
        ),
    )
    scans_root = tmp_path / "scans"
    scene_root = scans_root / "scene0000_00"
    scene_root.mkdir(parents=True)
    scene_root.joinpath("scene0000_00.txt").write_text(
        "axisAlignment = "
        "0 -1 0 2 1 0 0 3 0 0 1 4 0 0 0 1\n",
        encoding="utf-8",
    )
    encoder = _ContextAwareFrozenEncoder()
    encoded = trainer.precompute_positive_hidden(
        data,
        encoder,
        gt_root=gt_root,
        scans_root=scans_root,
    )

    assert encoder.seen_rows == [2]
    assert encoded.positive_count == 1
    torch.testing.assert_close(
        encoded.scenes[0].hidden, torch.tensor([[1.0, 2.0]])
    )
    torch.testing.assert_close(
        encoded.scenes[0].frozen_p1s_raw_regression,
        torch.tensor([[0.1, -0.2, 0.3, 0.0, 0.1, -0.1]]),
    )
    torch.testing.assert_close(
        encoded.scenes[0].target_boxes_aligned,
        torch.tensor([[4.0, 5.0, 6.0, 0.5, 1.0, 2.0]]),
    )
    torch.testing.assert_close(
        encoded.scenes[0].axis_alignment,
        torch.tensor(
            [
                [
                    [0.0, -1.0, 0.0, 2.0],
                    [1.0, 0.0, 0.0, 3.0],
                    [0.0, 0.0, 1.0, 4.0],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            ]
        ),
    )


def test_epoch_selection_is_cal_lexicographic_and_audit_runs_once(
    monkeypatch,
):
    calls: list[str] = []
    calibrations = iter(
        (
            (0.25, 0.80),
            (0.50, 0.20),
            (0.50, 0.30),
        )
    )

    def fake_evaluate(head, tensors, *, role, **kwargs):
        del head, kwargs
        calls.append(role)
        if role == "cal":
            fraction, mean = next(calibrations)
        else:
            fraction, mean = (0.99, 0.99)
        return {
            "role": role,
            "positive_anchor_count": len(tensors[0]),
            "decoded_aligned_fraction_iou_gt_0p5": fraction,
            "decoded_aligned_mean_iou": mean,
        }

    monkeypatch.setattr(trainer, "evaluate_geometry", fake_evaluate)
    _, metrics = trainer.train_geometry_refiner(
        _encoded_dataset(),
        fit_scene_ids=("scene0000_00",),
        cal_scene_ids=("scene0001_00",),
        audit_scene_ids=("scene0002_00",),
        epochs=3,
        batch_size=4,
        seed=3,
    )

    assert calls == ["cal", "cal", "cal", "audit"]
    assert metrics["selection"]["best_epoch"] == 2
    assert metrics["selection"]["best_key"] == [0.5, 0.3]
    assert metrics["selection"]["audit_used_for_selection"] is False
    assert metrics["audit_evaluation_count"] == 1
    assert (
        metrics["audit"]["decoded_aligned_fraction_iou_gt_0p5"] == 0.99
    )


def test_real_small_training_returns_finite_metrics_for_correction_only():
    encoded = _encoded_dataset()
    head, metrics = trainer.train_geometry_refiner(
        encoded,
        fit_scene_ids=("scene0000_00",),
        cal_scene_ids=("scene0001_00",),
        audit_scene_ids=("scene0002_00",),
        epochs=2,
        batch_size=2,
        learning_rate=1e-2,
        seed=19,
    )
    assert all(
        bool(torch.isfinite(parameter).all())
        for parameter in head.parameters()
    )
    assert tuple(name for name, _ in head.named_parameters()) == (
        "correction.weight",
        "correction.bias",
    )
    assert len(metrics["history"]) == 2
    assert metrics["audit_evaluation_count"] == 1


def test_checkpoint_round_trip_binds_schema_hashes_metrics_and_state(
    tmp_path: Path,
):
    (
        head,
        decoder,
        training_config,
        source_p1s,
        provenance,
    ) = _valid_checkpoint_parts(tmp_path)
    output = trainer.save_p1g_checkpoint(
        tmp_path / "p1g.pt",
        head=head,
        decoder_config=decoder,
        training_config=training_config,
        source_p1s=source_p1s,
        provenance=provenance,
        metrics=_valid_metrics(),
    )
    loaded, payload, digest = load_p1g_checkpoint(
        output,
        expected_p1s_checkpoint_sha256=source_p1s[
            "checkpoint_sha256"
        ],
    )

    assert payload["schema"] == trainer.P1G_CHECKPOINT_SCHEMA
    assert digest == trainer.sha256_file(output)
    assert len(digest) == 64
    assert payload["source_p1s"]["checkpoint_sha256"] == (
        source_p1s["checkpoint_sha256"]
    )
    assert payload["metrics"]["audit_evaluation_count"] == 1
    for name, value in head.state_dict().items():
        torch.testing.assert_close(loaded.state_dict()[name], value)


@pytest.mark.parametrize(
    "mutation",
    (
        lambda payload: payload.update(schema="wrong"),
        lambda payload: payload["provenance"].update(
            audit_scene_ids=["scene0000_00"]
        ),
        lambda payload: payload.update(uses_ground_truth=True),
        lambda payload: payload["provenance"].update(
            p1s_checkpoint_sha256="not-a-sha"
        ),
    ),
)
def test_strict_checkpoint_loader_rejects_tampering(
    tmp_path: Path, mutation
):
    (
        head,
        decoder,
        training_config,
        source_p1s,
        provenance,
    ) = _valid_checkpoint_parts(tmp_path)
    clean = tmp_path / "clean.pt"
    trainer.save_p1g_checkpoint(
        clean,
        head=head,
        decoder_config=decoder,
        training_config=training_config,
        source_p1s=source_p1s,
        provenance=provenance,
        metrics=_valid_metrics(),
    )
    payload = torch.load(clean, map_location="cpu", weights_only=False)
    mutation(payload)
    tampered = tmp_path / "tampered.pt"
    torch.save(payload, tampered)
    with pytest.raises(ValueError):
        load_p1g_checkpoint(
            tampered,
            expected_p1s_checkpoint_sha256=source_p1s[
                "checkpoint_sha256"
            ],
        )
