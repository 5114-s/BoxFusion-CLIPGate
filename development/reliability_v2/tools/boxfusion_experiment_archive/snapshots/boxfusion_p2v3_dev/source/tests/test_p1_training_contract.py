"""Train/runtime provenance closure for the P1 residual head."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from boxfusion.residual_proposal import (  # noqa: E402
    P1_DIAGNOSTIC_SCHEMA,
    P1_FEATURE_DIM,
    P1_FEATURE_NAMES,
    P1_HEAD_SCHEMA,
    ResidualProposalConfig,
    ResidualVoxelProposalHead,
    load_residual_proposal_head,
)
from tools.train_p1_residual_head import load_scene_voxels  # noqa: E402


def _diagnostic(path: Path, **updates) -> Path:
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
        "p1_feature_names": np.asarray(P1_FEATURE_NAMES, dtype=np.str_),
        "p1_voxel_features": np.zeros(
            (2, P1_FEATURE_DIM), dtype=np.float32
        ),
        "p1_voxel_centers": np.zeros((2, 3), dtype=np.float32),
        "p1_voxel_offsets": np.asarray([0, 2], dtype=np.int64),
    }
    payload.update(updates)
    np.savez_compressed(path, **payload)
    return path


def test_training_loader_requires_exact_read_only_14d_contract(tmp_path):
    path = _diagnostic(tmp_path / "valid.npz")
    loaded = load_scene_voxels(path, expected_scene_id="scene0001_00")
    assert loaded.features.shape == (2, P1_FEATURE_DIM)
    assert loaded.feature_names == P1_FEATURE_NAMES

    unsafe = _diagnostic(
        tmp_path / "unsafe.npz",
        p1_mutation_enabled=np.asarray(True, dtype=bool),
    )
    with pytest.raises(ValueError, match="unsafe p1_mutation_enabled"):
        load_scene_voxels(unsafe)

    wrong_schema = _diagnostic(
        tmp_path / "wrong_schema.npz",
        p1_feature_names=np.asarray(
            [f"feature_{index}" for index in range(P1_FEATURE_DIM)],
            dtype=np.str_,
        ),
    )
    with pytest.raises(ValueError, match="feature schema"):
        load_scene_voxels(wrong_schema)


def _checkpoint(path: Path, *, b6_sha: str, scenes=None) -> Path:
    head = ResidualVoxelProposalHead(
        input_dim=P1_FEATURE_DIM, hidden_dim=8
    )
    payload = {
        "schema": P1_HEAD_SCHEMA,
        "feature_names": list(P1_FEATURE_NAMES),
        "model_config": {
            "input_dim": P1_FEATURE_DIM,
            "hidden_dim": 8,
            "regression_dim": 6,
        },
        "state_dict": head.state_dict(),
        "provenance": {
            "train_scene_ids": scenes or ["scene0001_00"],
            "forbidden_overlap": [],
            "train_scene_list_sha256": "1" * 64,
            "forbidden_scene_list_sha256": "2" * 64,
            "b6_checkpoint_sha256": b6_sha,
        },
    }
    torch.save(payload, path)
    return path


def test_runtime_checkpoint_is_bound_to_exact_frozen_b6(tmp_path):
    b6_sha = "a" * 64
    checkpoint = _checkpoint(tmp_path / "p1.pt", b6_sha=b6_sha)
    config = ResidualProposalConfig(
        enabled=True,
        mode="infer",
        checkpoint=str(checkpoint),
        hidden_dim=8,
    ).validated()
    model, checkpoint_sha, _ = load_residual_proposal_head(
        checkpoint,
        expected_config=config,
        device="cpu",
        expected_b6_checkpoint_sha256=b6_sha,
    )
    assert isinstance(model, ResidualVoxelProposalHead)
    assert len(checkpoint_sha) == 64

    with pytest.raises(ValueError, match="different frozen B6"):
        load_residual_proposal_head(
            checkpoint,
            expected_config=config,
            device="cpu",
            expected_b6_checkpoint_sha256="b" * 64,
        )


@pytest.mark.parametrize(
    "b6_sha,scenes",
    [
        ("z" * 64, ["scene0001_00"]),
        ("a" * 64, ["not-a-scene"]),
        ("a" * 64, ["scene0001_00", "scene0001_00"]),
    ],
)
def test_runtime_rejects_forged_provenance(tmp_path, b6_sha, scenes):
    checkpoint = _checkpoint(
        tmp_path / "forged.pt", b6_sha=b6_sha, scenes=scenes
    )
    config = ResidualProposalConfig(
        enabled=True,
        mode="infer",
        checkpoint=str(checkpoint),
        hidden_dim=8,
    ).validated()
    with pytest.raises(ValueError, match="provenance"):
        load_residual_proposal_head(
            checkpoint,
            expected_config=config,
            device="cpu",
            expected_b6_checkpoint_sha256="a" * 64,
        )
