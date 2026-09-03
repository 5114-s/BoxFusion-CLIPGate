"""Synthetic contract tests for the controlled P1R/P1S trainer."""

from __future__ import annotations

import json
import hashlib
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from boxfusion.p1_spatial_residual import (  # noqa: E402
    P1_SPATIAL_ARCHITECTURE,
    NativeSparseResidualProposalHead,
)
from boxfusion.residual_proposal import (  # noqa: E402
    P1_DIAGNOSTIC_SCHEMA,
    P1_FEATURE_DIM,
    P1_FEATURE_NAMES,
    P1_HEAD_SCHEMA,
)
from tools.train_p1v2_residual_head import (  # noqa: E402
    P1R_ARCHITECTURE,
    P1R_CHECKPOINT_SCHEMA,
    P1S_CHECKPOINT_SCHEMA,
    TARGET_ASSIGNMENT_SCOPE,
    P1V2TrainingData,
    SceneTrainingContext,
    assign_snapshot_inside_targets,
    deterministic_scene_partition,
    load_scene_context,
    save_checkpoint,
    train_variant,
    validate_external_split,
    validate_source_collection_provenance,
)


def _diagnostic(path: Path, **updates) -> Path:
    coordinates = np.asarray(
        [
            [0, 0, 0],
            [1, 0, 0],
            [2, 0, 0],
            [0, 0, 0],
            [0, 1, 0],
        ],
        dtype=np.int32,
    )
    voxel_size = 0.08
    payload = {
        "scene_id": np.asarray("scene0001_00"),
        "p1_schema": np.asarray(P1_DIAGNOSTIC_SCHEMA),
        "p1_stage": np.asarray("P1"),
        "p1_profile": np.asarray("p1_residual_proposal_observer"),
        "p1_enabled": np.asarray(True, dtype=bool),
        "p1_observer_only": np.asarray(True, dtype=bool),
        "p1_uses_ground_truth": np.asarray(False, dtype=bool),
        "p1_mutation_enabled": np.asarray(False, dtype=bool),
        "p1_applied_count": np.asarray(0, dtype=np.int64),
        "p1_complete": np.asarray(True, dtype=bool),
        "p1_class_agnostic": np.asarray(True, dtype=bool),
        "p1_regression_dim": np.asarray(6, dtype=np.int64),
        "p1_config_json": np.asarray(
            json.dumps(
                {
                    "mode": "collect",
                    "collect_voxel_inputs": True,
                    "observer_only": True,
                    "mutate": False,
                    "voxel_size": voxel_size,
                }
            )
        ),
        "p1_feature_names": np.asarray(P1_FEATURE_NAMES, dtype=np.str_),
        "p1_step_frame_ids": np.asarray([0, 5], dtype=np.int64),
        "p1_step_provider_steps": np.asarray([0, 1], dtype=np.int64),
        "p1_step_voxel_counts": np.asarray([3, 2], dtype=np.int64),
        "p1_voxel_offsets": np.asarray([0, 3, 5], dtype=np.int64),
        "p1_voxel_coords": coordinates,
        "p1_voxel_centers": (
            (coordinates.astype(np.float32) + 0.5) * voxel_size
        ),
        "p1_voxel_features": np.zeros(
            (len(coordinates), P1_FEATURE_DIM), dtype=np.float32
        ),
    }
    payload.update(updates)
    np.savez_compressed(path, **payload)
    return path


def test_loader_preserves_legacy_full_context_and_step_offsets(tmp_path):
    path = _diagnostic(tmp_path / "scene0001_00_tracks.npz")
    scene = load_scene_context(path, expected_scene_id="scene0001_00")
    assert scene.features.shape == (5, P1_FEATURE_DIM)
    assert scene.coordinates.shape == (5, 3)
    assert scene.offsets.tolist() == [0, 3, 5]
    assert scene.frame_ids.tolist() == [0, 5]
    assert scene.provider_steps.tolist() == [0, 1]
    assert scene.snapshot_slice(0) == slice(0, 3)
    assert scene.snapshot_slice(1) == slice(3, 5)

    unsafe = _diagnostic(
        tmp_path / "unsafe.npz",
        p1_mutation_enabled=np.asarray(True, dtype=bool),
    )
    with pytest.raises(ValueError, match="unsafe p1_mutation_enabled"):
        load_scene_context(unsafe)

    broken_offsets = _diagnostic(
        tmp_path / "broken.npz",
        p1_voxel_offsets=np.asarray([0, 2, 5], dtype=np.int64),
    )
    with pytest.raises(ValueError, match="counts disagree"):
        load_scene_context(broken_offsets)


def test_snapshot_targets_never_fall_back_to_outside_voxels():
    centers = np.asarray(
        [[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [4.0, 4.0, 4.0]],
        dtype=np.float32,
    )
    visible_box = np.asarray(
        [[0.05, 0.0, 0.0, 0.4, 0.4, 0.4]], dtype=np.float32
    )
    visible = assign_snapshot_inside_targets(
        centers, visible_box, topk=1
    )
    assert int(visible.objectness.sum()) == 1
    assert visible.assigned_gt[2] == -1

    invisible_box = np.asarray(
        [[10.0, 10.0, 10.0, 0.2, 0.2, 0.2]], dtype=np.float32
    )
    invisible = assign_snapshot_inside_targets(
        centers, invisible_box, topk=6
    )
    assert not np.any(invisible.objectness)
    assert np.all(invisible.assigned_gt == -1)


def test_scene_splits_are_disjoint_and_reject_external_leakage():
    training, validation = deterministic_scene_partition(
        ("scene0001_00", "scene0002_00", "scene0003_00"),
        validation_fraction=1.0 / 3.0,
        seed=7,
    )
    assert training
    assert validation
    assert set(training).isdisjoint(validation)
    assert set(training) | set(validation) == {
        "scene0001_00",
        "scene0002_00",
        "scene0003_00",
    }
    with pytest.raises(ValueError, match="leakage"):
        validate_external_split(
            ("scene0001_00",), ("scene0001_00",)
        )


def test_source_witness_binds_b6_and_every_frozen_input(tmp_path):
    scene = "scene0001_00"
    diagnostics = tmp_path / "diagnostics"
    predictions = tmp_path / "predictions"
    ground_truth = tmp_path / "gt"
    for directory in (diagnostics, predictions, ground_truth):
        directory.mkdir()
    diagnostic = _diagnostic(diagnostics / f"{scene}_tracks.npz")
    prediction = predictions / f"{scene}_boxes.pkl"
    target = ground_truth / f"{scene}_bbox.npy"
    prediction.write_bytes(b"frozen-prediction")
    target.write_bytes(b"frozen-ground-truth")
    train_list = tmp_path / "train.txt"
    forbidden_list = tmp_path / "val.txt"
    b6 = tmp_path / "b6.npz"
    train_list.write_text(scene + "\n", encoding="utf-8")
    forbidden_list.write_text("scene0002_00\n", encoding="utf-8")
    b6.write_bytes(b"frozen-b6")

    def sha(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()
    witness = tmp_path / "legacy-p1.pt"
    torch.save(
        {
            "schema": P1_HEAD_SCHEMA,
            "provenance": {
                "train_scene_ids": [scene],
                "forbidden_overlap": [],
                "b6_checkpoint_sha256": sha(b6),
                "train_scene_list_sha256": sha(train_list),
                "forbidden_scene_list_sha256": sha(forbidden_list),
                "scene_summaries": [
                    {
                        "scene_id": scene,
                        "diagnostic_sha256": sha(diagnostic),
                        "prediction_sha256": sha(prediction),
                        "ground_truth_sha256": sha(target),
                    }
                ],
            },
        },
        witness,
    )
    report = validate_source_collection_provenance(
        witness,
        scenes=(scene,),
        diagnostics_root=diagnostics,
        prediction_root=predictions,
        gt_root=ground_truth,
        train_scene_list=train_list,
        forbidden_scene_list=forbidden_list,
        b6_checkpoint=b6,
    )
    assert report["verified"] is True
    assert report["b6_checkpoint_sha256"] == sha(b6)

    prediction.write_bytes(b"changed-prediction")
    with pytest.raises(ValueError, match="prediction_sha256"):
        validate_source_collection_provenance(
            witness,
            scenes=(scene,),
            diagnostics_root=diagnostics,
            prediction_root=predictions,
            gt_root=ground_truth,
            train_scene_list=train_list,
            forbidden_scene_list=forbidden_list,
            b6_checkpoint=b6,
        )


def _scene(scene_id: str, shift: int) -> SceneTrainingContext:
    # Two complete four-voxel snapshots.  Only rows 0/4 and a subset of
    # negatives contribute to the loss, while every forward still receives
    # all four context rows.
    base_coordinates = np.asarray(
        [[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0]],
        dtype=np.int32,
    )
    coordinates = np.concatenate(
        (base_coordinates, base_coordinates + np.asarray([0, 1, 0])),
        axis=0,
    )
    features = np.zeros((8, P1_FEATURE_DIM), dtype=np.float32)
    features[:, 0] = -1.0
    features[[0, 4], 0] = 1.0 + 0.1 * shift
    features[:, 1] = np.linspace(0.0, 1.0, 8, dtype=np.float32)
    objectness = np.zeros(8, dtype=np.float32)
    objectness[[0, 4]] = 1.0
    regression = np.zeros((8, 6), dtype=np.float32)
    regression[[0, 4], 3:] = np.log(
        np.asarray([0.4, 0.5, 0.6], dtype=np.float32)
    )
    loss_mask = np.zeros(8, dtype=bool)
    loss_mask[[0, 1, 2, 4, 5, 6]] = True
    return SceneTrainingContext(
        scene_id=scene_id,
        features=features,
        coordinates=coordinates,
        centers_world=(coordinates.astype(np.float32) + 0.5) * 0.08,
        offsets=np.asarray([0, 4, 8], dtype=np.int64),
        frame_ids=np.asarray([0, 5], dtype=np.int64),
        provider_steps=np.asarray([0, 1], dtype=np.int64),
        objectness=objectness,
        regression=regression,
        assigned_gt=np.where(objectness > 0.5, 0, -1).astype(np.int64),
        loss_mask=loss_mask,
        feature_names=tuple(P1_FEATURE_NAMES),
        voxel_size=0.08,
        diagnostic_path=Path(f"/synthetic/{scene_id}_tracks.npz"),
    )


@pytest.mark.parametrize(
    "variant,expected_schema,expected_architecture",
    [
        ("P1R", P1R_CHECKPOINT_SCHEMA, P1R_ARCHITECTURE),
        ("P1S", P1S_CHECKPOINT_SCHEMA, P1_SPATIAL_ARCHITECTURE),
    ],
)
def test_tiny_training_and_checkpoint_contract(
    tmp_path, variant, expected_schema, expected_architecture
):
    data = P1V2TrainingData(
        scenes=(
            _scene("scene0001_00", 0),
            _scene("scene0002_00", 1),
        ),
        feature_names=tuple(P1_FEATURE_NAMES),
        scene_summaries=(),
        dataset_fingerprint_sha256="a" * 64,
    )
    model, metrics = train_variant(
        data,
        variant=variant,
        hidden_dim=8,
        validation_fraction=0.5,
        epochs=2,
        learning_rate=1e-2,
        weight_decay=0.0,
        snapshots_per_optimizer_step=2,
        seed=3,
        device="cpu",
    )
    assert set(metrics["training_scenes"]).isdisjoint(
        metrics["validation_scenes"]
    )
    training_config = {
        "schema": "synthetic",
        "target_assignment_scope": TARGET_ASSIGNMENT_SCOPE,
    }
    provenance = {
        "train_scene_ids": ["scene0001_00", "scene0002_00"],
        "optimization_train_scene_ids": metrics["training_scenes"],
        "optimization_validation_scene_ids": metrics["validation_scenes"],
        "forbidden_overlap": [],
        "dataset_fingerprint_sha256": "a" * 64,
    }
    output = save_checkpoint(
        tmp_path / f"{variant.lower()}.pt",
        model=model,
        variant=variant,
        hidden_dim=8,
        feature_names=P1_FEATURE_NAMES,
        training_config=training_config,
        metrics=metrics,
        provenance=provenance,
    )
    try:
        checkpoint = torch.load(
            output, map_location="cpu", weights_only=False
        )
    except TypeError:
        checkpoint = torch.load(output, map_location="cpu")
    assert checkpoint["schema"] == expected_schema
    assert checkpoint["head_architecture"] == expected_architecture
    assert (
        checkpoint["training_config"]["target_assignment_scope"]
        == TARGET_ASSIGNMENT_SCOPE
    )
    assert tuple(checkpoint["feature_names"]) == tuple(P1_FEATURE_NAMES)
    assert checkpoint["provenance"] == provenance
    if variant == "P1R":
        assert (
            checkpoint["model_config"]["head_architecture"]
            == P1R_ARCHITECTURE
        )
    else:
        assert (
            checkpoint["model_config"]["architecture"]
            == P1_SPATIAL_ARCHITECTURE
        )
        rebuilt = NativeSparseResidualProposalHead.from_model_config(
            checkpoint["model_config"]
        )
        rebuilt.load_state_dict(checkpoint["state_dict"], strict=True)
