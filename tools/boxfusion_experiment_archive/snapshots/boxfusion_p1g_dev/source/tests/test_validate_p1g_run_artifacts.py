"""Safety, ancestry, and fail-open contracts for complete P1G runs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from boxfusion.p1_multiview_geometry import (
    P1G_DIAGNOSTIC_SCHEMA,
    P1G_PROFILE,
    P1G_SOURCE,
)
from boxfusion.residual_proposal import P1_DIAGNOSTIC_SCHEMA
import tools.validate_p1g_run_artifacts as validator


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _corners(boxes: np.ndarray) -> np.ndarray:
    signs = np.asarray(
        [
            [-1, -1, -1],
            [-1, -1, 1],
            [-1, 1, -1],
            [-1, 1, 1],
            [1, -1, -1],
            [1, -1, 1],
            [1, 1, -1],
            [1, 1, 1],
        ],
        dtype=np.float32,
    )
    return (
        boxes[:, None, :3]
        + 0.5 * signs[None] * boxes[:, None, 3:]
    ).astype(np.float32)


def _payload(scene: str, checkpoint_sha: str) -> dict[str, np.ndarray]:
    parent_boxes = np.asarray(
        [
            [0.0, 0.0, 0.0, 1.0, 0.8, 0.6],
            [2.0, 0.0, 0.0, 0.7, 0.9, 0.5],
        ],
        dtype=np.float32,
    )
    refined_boxes = np.array(parent_boxes, copy=True)
    refined_boxes[0, 0] += 0.1
    parent_ids = np.asarray(
        [
            f"{scene}:000005:0:0:0",
            f"{scene}:000005:1:0:0",
        ],
        dtype=np.str_,
    )
    scores = np.asarray([0.8, 0.6], dtype=np.float32)
    config = {
        "enabled": True,
        "observer_only": True,
        "mutate": False,
        "collect_diagnostics": True,
        "association_iou": 0.1,
        "crop_scale": 1.35,
        "top_k_views": 3,
        "view_diversity_weight": 0.25,
        "max_points_per_view": 128,
        "max_candidates": 2,
        "proposal": {},
    }
    return {
        "scene_id": np.asarray(scene),
        "p1_schema": np.asarray(P1_DIAGNOSTIC_SCHEMA),
        "p1_stage": np.asarray("P1S"),
        "p1_profile": np.asarray("p1s_native_sparse_context_observer"),
        "p1_checkpoint_sha256": np.asarray(checkpoint_sha),
        "p1_head_architecture": np.asarray("native_sparse_context_v1"),
        "p1_target_assignment_scope": np.asarray(
            "snapshot_inside_only"
        ),
        "p1_enabled": np.asarray(True, dtype=bool),
        "p1_observer_only": np.asarray(True, dtype=bool),
        "p1_uses_ground_truth": np.asarray(False, dtype=bool),
        "p1_reads_semantic_labels": np.asarray(False, dtype=bool),
        "p1_mutation_enabled": np.asarray(False, dtype=bool),
        "p1_applied_count": np.asarray(0, dtype=np.int64),
        "p1_complete": np.asarray(True, dtype=bool),
        "p1_class_agnostic": np.asarray(True, dtype=bool),
        "p1_candidate_ids": parent_ids,
        "p1_candidate_boxes": parent_boxes,
        "p1_candidate_corners": _corners(parent_boxes),
        "p1_candidate_scores": scores,
        "p1g_schema": np.asarray(P1G_DIAGNOSTIC_SCHEMA),
        "p1g_stage": np.asarray("P1G"),
        "p1g_profile": np.asarray(P1G_PROFILE),
        "p1g_parent_stage": np.asarray("P1S"),
        "p1g_parent_checkpoint_sha256": np.asarray(checkpoint_sha),
        "p1g_enabled": np.asarray(True, dtype=bool),
        "p1g_observer_only": np.asarray(True, dtype=bool),
        "p1g_uses_ground_truth": np.asarray(False, dtype=bool),
        "p1g_reads_semantic_labels": np.asarray(False, dtype=bool),
        "p1g_mutation_enabled": np.asarray(False, dtype=bool),
        "p1g_applied_count": np.asarray(0, dtype=np.int64),
        "p1g_complete": np.asarray(True, dtype=bool),
        "p1g_class_agnostic": np.asarray(True, dtype=bool),
        "p1g_regression_dim": np.asarray(6, dtype=np.int64),
        "p1g_config_json": np.asarray(
            json.dumps(config, sort_keys=True)
        ),
        "p1g_runtime_seconds": np.asarray(0.03, dtype=np.float64),
        "p1g_failure_count": np.asarray(0, dtype=np.int64),
        "p1g_parent_candidate_ids": parent_ids,
        "p1g_refined_candidate_ids": np.asarray(
            [f"{value}:p1g" for value in parent_ids.tolist()],
            dtype=np.str_,
        ),
        "p1g_parent_boxes": parent_boxes,
        "p1g_parent_corners": _corners(parent_boxes),
        "p1g_refined_boxes": refined_boxes,
        "p1g_refined_corners": _corners(refined_boxes),
        "p1g_candidate_scores": scores,
        "p1g_candidate_applied": np.zeros(2, dtype=bool),
        "p1g_is_candidate": np.asarray([True, False], dtype=bool),
        "p1g_reasons": np.asarray(
            ["candidate", "identity_insufficient_views"], dtype=np.str_
        ),
        "p1g_sources": np.asarray([P1G_SOURCE, P1G_SOURCE], dtype=np.str_),
        "p1g_matched_view_counts": np.asarray([3, 1], dtype=np.int64),
        "p1g_selected_view_counts": np.asarray([2, 1], dtype=np.int64),
        "p1g_selected_frame_ids": np.asarray(
            [[5, 10, -1], [5, -1, -1]], dtype=np.int64
        ),
        "p1g_cropped_point_counts": np.asarray([128, 64], dtype=np.int64),
        "p1g_face_residuals": np.zeros((2, 3, 2), dtype=np.float32),
        "p1g_face_support": np.full(
            (2, 3, 2), 0.8, dtype=np.float32
        ),
        "p1g_face_uncertainty": np.full(
            (2, 3, 2), 0.1, dtype=np.float32
        ),
        "p1g_face_supported": np.ones((2, 3, 2), dtype=bool),
        "p1g_feature_vectors": np.zeros((2, 48), dtype=np.float32),
        "p1g_step_total_seconds": np.asarray(
            [0.01, 0.01], dtype=np.float64
        ),
    }


def _artifacts(tmp_path: Path) -> dict[str, object]:
    scene = "scene0001_00"
    scene_list = tmp_path / "scenes.txt"
    scene_list.write_text(scene + "\n", encoding="utf-8")
    predictions = tmp_path / "predictions"
    diagnostics = tmp_path / "diagnostics"
    predictions.mkdir()
    diagnostics.mkdir()
    (predictions / f"{scene}_boxes.pkl").write_bytes(
        b"same-run formal prediction"
    )
    checkpoint = tmp_path / "p1s.pt"
    torch.save(
        {
            "model_config": {
                "head_architecture": "native_sparse_context_v1"
            },
            "training_config": {
                "target_assignment_scope": "snapshot_inside_only"
            },
        },
        checkpoint,
    )
    payload = _payload(scene, _sha(checkpoint))
    diagnostic = diagnostics / f"{scene}_tracks.npz"
    np.savez_compressed(diagnostic, **payload)
    return {
        "scene": scene,
        "scene_list": scene_list,
        "predictions": predictions,
        "diagnostics": diagnostics,
        "checkpoint": checkpoint,
        "diagnostic": diagnostic,
        "payload": payload,
    }


def _rewrite(artifacts: dict[str, object], **updates: np.ndarray) -> None:
    payload = dict(artifacts["payload"])
    payload.update(updates)
    np.savez_compressed(artifacts["diagnostic"], **payload)


def _validate(artifacts: dict[str, object]) -> dict[str, object]:
    return validator.validate(
        scene_list=artifacts["scene_list"],
        prediction_root=artifacts["predictions"],
        diagnostics_root=artifacts["diagnostics"],
        expected_p1s_checkpoint=artifacts["checkpoint"],
    )


def test_valid_run_is_bound_to_canonical_runtime_contract(tmp_path):
    artifacts = _artifacts(tmp_path)
    report = _validate(artifacts)

    assert validator.P1G_SCHEMA == P1G_DIAGNOSTIC_SCHEMA
    assert validator.P1G_PROFILE == P1G_PROFILE
    assert report["ok"] is True
    assert report["scene_count"] == 1
    assert report["parents"] == 2
    assert report["candidates"] == 1
    assert report["matched_multiview"] == 1
    assert report["selected_multiview"] == 1
    assert report["failures"] == 0
    assert report["checkpoint_sha256"] == _sha(artifacts["checkpoint"])


@pytest.mark.parametrize(
    ("key", "value", "message"),
    [
        (
            "p1_checkpoint_sha256",
            np.asarray("0" * 64),
            "p1_checkpoint_sha256",
        ),
        (
            "p1g_parent_checkpoint_sha256",
            np.asarray("1" * 64),
            "p1g_parent_checkpoint_sha256",
        ),
        (
            "p1g_profile",
            np.asarray("stale_p1g_profile"),
            "p1g_profile",
        ),
        (
            "p1_profile",
            np.asarray("stale_parent_profile"),
            "p1_profile",
        ),
    ],
)
def test_checkpoint_or_profile_drift_is_rejected(
    tmp_path, key, value, message
):
    artifacts = _artifacts(tmp_path)
    _rewrite(artifacts, **{key: value})
    with pytest.raises(ValueError, match=message):
        _validate(artifacts)


@pytest.mark.parametrize(
    ("key", "mutate", "message"),
    [
        (
            "p1g_parent_candidate_ids",
            lambda value: value[::-1],
            "parents disagree with frozen P1S",
        ),
        (
            "p1g_refined_candidate_ids",
            lambda value: np.asarray(
                ["wrong:p1g", value[1]], dtype=np.str_
            ),
            "invalid one-to-one P1G child ids",
        ),
        (
            "p1g_parent_boxes",
            lambda value: value + np.asarray(
                [0.01, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32
            ),
            "parents disagree with frozen P1S",
        ),
    ],
)
def test_parent_child_lineage_must_be_exact(
    tmp_path, key, mutate, message
):
    artifacts = _artifacts(tmp_path)
    value = np.array(artifacts["payload"][key], copy=True)
    _rewrite(artifacts, **{key: mutate(value)})
    with pytest.raises(ValueError, match=message):
        _validate(artifacts)


@pytest.mark.parametrize(
    ("key", "value", "message"),
    [
        (
            "p1g_mutation_enabled",
            np.asarray(True, dtype=bool),
            "unsafe p1g_mutation_enabled",
        ),
        (
            "p1g_applied_count",
            np.asarray(1, dtype=np.int64),
            "P1G mutated formal output",
        ),
        (
            "p1g_candidate_applied",
            np.asarray([False, True], dtype=bool),
            "P1G rows cannot be applied",
        ),
    ],
)
def test_any_mutation_path_is_rejected(tmp_path, key, value, message):
    artifacts = _artifacts(tmp_path)
    _rewrite(artifacts, **{key: value})
    with pytest.raises(ValueError, match=message):
        _validate(artifacts)


def test_rejected_row_must_be_exact_identity(tmp_path):
    artifacts = _artifacts(tmp_path)
    refined = np.array(
        artifacts["payload"]["p1g_refined_boxes"], copy=True
    )
    refined[1, 0] += 0.2
    _rewrite(
        artifacts,
        p1g_refined_boxes=refined,
        p1g_refined_corners=_corners(refined),
    )
    with pytest.raises(ValueError, match="rejected P1G row did not fail open"):
        _validate(artifacts)


def test_refined_box_and_corner_aliases_must_agree(tmp_path):
    artifacts = _artifacts(tmp_path)
    refined = np.array(
        artifacts["payload"]["p1g_refined_boxes"], copy=True
    )
    refined[0, 0] += 0.2
    _rewrite(artifacts, p1g_refined_boxes=refined)
    with pytest.raises(ValueError, match="refined box/corner aliases disagree"):
        _validate(artifacts)


def test_candidate_flag_must_match_reason_and_geometry(tmp_path):
    artifacts = _artifacts(tmp_path)
    _rewrite(
        artifacts,
        p1g_is_candidate=np.asarray([False, False], dtype=bool),
    )
    with pytest.raises(ValueError, match="candidate flags disagree"):
        _validate(artifacts)
