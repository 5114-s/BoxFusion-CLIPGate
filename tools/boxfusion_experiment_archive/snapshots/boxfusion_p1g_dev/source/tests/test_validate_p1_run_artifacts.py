"""Exact artifact-set validation for resumable P1 runs."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest

from boxfusion.residual_proposal import (
    P1_DIAGNOSTIC_SCHEMA,
    P1_FEATURE_DIM,
    P1_FEATURE_NAMES,
)
from tools.validate_p1_run_artifacts import validate


def _diagnostic(path: Path, scene: str, checkpoint_sha: str) -> None:
    np.savez_compressed(
        path,
        scene_id=np.asarray(scene),
        p1_schema=np.asarray(P1_DIAGNOSTIC_SCHEMA),
        p1_stage=np.asarray("P1"),
        p1_profile=np.asarray("p1_residual_proposal_observer"),
        p1_enabled=np.asarray(True, dtype=bool),
        p1_observer_only=np.asarray(True, dtype=bool),
        p1_uses_ground_truth=np.asarray(False, dtype=bool),
        p1_mutation_enabled=np.asarray(False, dtype=bool),
        p1_applied_count=np.asarray(0, dtype=np.int64),
        p1_complete=np.asarray(True, dtype=bool),
        p1_class_agnostic=np.asarray(True, dtype=bool),
        p1_regression_dim=np.asarray(6, dtype=np.int64),
        p1_checkpoint_sha256=np.asarray(checkpoint_sha),
        p1_feature_names=np.asarray(P1_FEATURE_NAMES, dtype=np.str_),
        p1_voxel_features=np.empty(
            (0, P1_FEATURE_DIM), dtype=np.float32
        ),
        p1_voxel_centers=np.empty((0, 3), dtype=np.float32),
        p1_voxel_offsets=np.asarray([0], dtype=np.int64),
    )


def test_validator_requires_exact_pairs_and_checkpoint(tmp_path):
    scene = "scene0001_00"
    scene_list = tmp_path / "scenes.txt"
    scene_list.write_text(scene + "\n", encoding="utf-8")
    predictions = tmp_path / "predictions"
    diagnostics = tmp_path / "diagnostics"
    predictions.mkdir()
    diagnostics.mkdir()
    (predictions / f"{scene}_boxes.pkl").write_bytes(b"prediction")
    checkpoint = tmp_path / "p1.pt"
    checkpoint.write_bytes(b"checkpoint")
    checkpoint_sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    _diagnostic(
        diagnostics / f"{scene}_tracks.npz", scene, checkpoint_sha
    )

    report = validate(
        scene_list=scene_list,
        prediction_root=predictions,
        diagnostics_root=diagnostics,
        require_checkpoint=True,
        expected_checkpoint=checkpoint,
    )
    assert report["scene_count"] == 1
    assert report["checkpoint_sha256"] == checkpoint_sha

    (predictions / "scene0002_00_boxes.pkl").write_bytes(b"extra")
    with pytest.raises(ValueError, match="prediction set mismatch"):
        validate(
            scene_list=scene_list,
            prediction_root=predictions,
            diagnostics_root=diagnostics,
            require_checkpoint=True,
            expected_checkpoint=checkpoint,
        )
