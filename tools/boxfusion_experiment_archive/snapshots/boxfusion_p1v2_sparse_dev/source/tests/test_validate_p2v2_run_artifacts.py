"""Safety and shape contracts for complete P2-v2 observer runs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from boxfusion.occupancy_topk import P2_DIAGNOSTIC_SCHEMA
from boxfusion.p2_local_mask_geometry import (
    P2V2_DIAGNOSTIC_SCHEMA,
    P2V2_SOURCE,
)
from boxfusion.residual_proposal import P1_DIAGNOSTIC_SCHEMA
from tools.validate_p2v2_run_artifacts import validate


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _payload(
    scene: str,
    p1_sha: str,
    p2_sha: str,
) -> dict[str, np.ndarray]:
    box = np.asarray([[0.0, 0.0, 0.0, 0.4, 0.5, 0.6]],
                     dtype=np.float32)
    corners = np.asarray(
        [
            [
                [-0.2, -0.25, -0.3],
                [-0.2, -0.25, 0.3],
                [-0.2, 0.25, -0.3],
                [-0.2, 0.25, 0.3],
                [0.2, -0.25, -0.3],
                [0.2, -0.25, 0.3],
                [0.2, 0.25, -0.3],
                [0.2, 0.25, 0.3],
            ]
        ],
        dtype=np.float32,
    )
    config = {
        "enabled": True,
        "observer_only": True,
        "mutate": False,
        "collect_diagnostics": True,
        "maximum_masks_per_step": 64,
        "maximum_components_per_mask": 8,
        "max_candidates_per_step": 16,
        "max_scene_candidates": 64,
    }
    return {
        "scene_id": np.asarray(scene),
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
        "p1_checkpoint_sha256": np.asarray(p1_sha),
        "p1_step_frame_ids": np.asarray([5], dtype=np.int64),
        "p1_step_provider_steps": np.asarray([1], dtype=np.int64),
        "p2_schema": np.asarray(P2_DIAGNOSTIC_SCHEMA),
        "p2_stage": np.asarray("P2"),
        "p2_profile": np.asarray("p2_occupancy_topk_observer"),
        "p2_enabled": np.asarray(True, dtype=bool),
        "p2_observer_only": np.asarray(True, dtype=bool),
        "p2_uses_ground_truth": np.asarray(False, dtype=bool),
        "p2_mutation_enabled": np.asarray(False, dtype=bool),
        "p2_applied_count": np.asarray(0, dtype=np.int64),
        "p2_complete": np.asarray(True, dtype=bool),
        "p2_class_agnostic": np.asarray(True, dtype=bool),
        "p2_checkpoint_sha256": np.asarray(p2_sha),
        "p2_step_frame_ids": np.asarray([5], dtype=np.int64),
        "p2_step_provider_steps": np.asarray([1], dtype=np.int64),
        "p2_step_input_voxel_counts": np.asarray([4], dtype=np.int64),
        "p2_step_eligible_voxel_counts": np.asarray(
            [3], dtype=np.int64
        ),
        "p2_step_selected_voxel_counts": np.asarray(
            [1], dtype=np.int64
        ),
        "p2_step_candidate_counts": np.asarray([0], dtype=np.int64),
        "p2_step_seconds": np.asarray([0.01], dtype=np.float64),
        "p2_candidate_ids": np.empty((0,), dtype=np.str_),
        "p2_candidate_boxes": np.empty((0, 6), dtype=np.float32),
        "p2_candidate_corners": np.empty(
            (0, 8, 3), dtype=np.float32
        ),
        "p2_candidate_objectness": np.empty((0,), dtype=np.float32),
        "p2_candidate_occupancy_scores": np.empty(
            (0,), dtype=np.float32
        ),
        "p2_candidate_occupancy_ranks": np.empty(
            (0,), dtype=np.int64
        ),
        "p2v2_schema": np.asarray(P2V2_DIAGNOSTIC_SCHEMA),
        "p2v2_stage": np.asarray("P2V2"),
        "p2v2_profile": np.asarray(
            "p2v2_local_component_mask_rgbd_observer"
        ),
        "p2v2_enabled": np.asarray(True, dtype=bool),
        "p2v2_observer_only": np.asarray(True, dtype=bool),
        "p2v2_uses_ground_truth": np.asarray(False, dtype=bool),
        "p2v2_reads_semantic_labels": np.asarray(
            False, dtype=bool
        ),
        "p2v2_mutation_enabled": np.asarray(False, dtype=bool),
        "p2v2_applied_count": np.asarray(0, dtype=np.int64),
        "p2v2_complete": np.asarray(True, dtype=bool),
        "p2v2_source": np.asarray(P2V2_SOURCE),
        "p2v2_mask_provider": np.asarray("yoloe"),
        "p2v2_parent_p2_checkpoint_sha256": np.asarray(p2_sha),
        "p2v2_config_json": np.asarray(
            json.dumps(config, sort_keys=True)
        ),
        "p2v2_step_frame_ids": np.asarray([5], dtype=np.int64),
        "p2v2_step_provider_steps": np.asarray([1], dtype=np.int64),
        "p2v2_step_selected_voxel_counts": np.asarray(
            [1], dtype=np.int64
        ),
        "p2v2_step_occupancy_component_counts": np.asarray(
            [1], dtype=np.int64
        ),
        "p2v2_step_mask_observation_counts": np.asarray(
            [1], dtype=np.int64
        ),
        "p2v2_step_mask_component_counts": np.asarray(
            [1], dtype=np.int64
        ),
        "p2v2_step_eligible_pair_counts": np.asarray(
            [1], dtype=np.int64
        ),
        "p2v2_step_candidate_counts": np.asarray(
            [1], dtype=np.int64
        ),
        "p2v2_step_seconds": np.asarray([0.02], dtype=np.float64),
        "p2v2_step_failed": np.asarray([False], dtype=bool),
        "p2v2_step_errors": np.asarray([""], dtype=np.str_),
        "p2v2_candidate_ids": np.asarray(
            ["p2v2:0123456789abcdef"], dtype=np.str_
        ),
        "p2v2_parent_p2_candidate_ids": np.asarray(
            ["p1:5:1:2:3"], dtype=np.str_
        ),
        "p2v2_mask_source_ids": np.asarray(
            ["scene0001_00:5:0"], dtype=np.str_
        ),
        "p2v2_candidate_boxes": box,
        "p2v2_candidate_corners": corners,
        "p2v2_candidate_parent_boxes": box.copy(),
        "p2v2_candidate_scores": np.asarray(
            [0.8], dtype=np.float32
        ),
        "p2v2_candidate_parent_objectness": np.asarray(
            [0.7], dtype=np.float32
        ),
        "p2v2_candidate_occupancy_scores": np.asarray(
            [0.8], dtype=np.float32
        ),
        "p2v2_candidate_mask_scores": np.asarray(
            [0.9], dtype=np.float32
        ),
        "p2v2_candidate_valid_depth_ratios": np.asarray(
            [0.8], dtype=np.float32
        ),
        "p2v2_candidate_component_point_counts": np.asarray(
            [40], dtype=np.int64
        ),
        "p2v2_candidate_component_voxel_counts": np.asarray(
            [12], dtype=np.int64
        ),
        "p2v2_candidate_selected_voxels_inside": np.asarray(
            [1], dtype=np.int64
        ),
        "p2v2_candidate_anchor_inside": np.asarray(
            [True], dtype=bool
        ),
        "p2v2_candidate_parent_iou": np.asarray(
            [0.7], dtype=np.float32
        ),
        "p2v2_candidate_normalized_center_distance": np.asarray(
            [0.1], dtype=np.float32
        ),
        "p2v2_candidate_extent_ratios": np.asarray(
            [[1.0, 1.0, 1.0]], dtype=np.float32
        ),
        "p2v2_candidate_center_shift_ratios": np.asarray(
            [[0.0, 0.0, 0.0]], dtype=np.float32
        ),
        "p2v2_candidate_applied": np.asarray([False], dtype=bool),
    }


def _artifacts(tmp_path: Path):
    scene = "scene0001_00"
    scene_list = tmp_path / "scenes.txt"
    scene_list.write_text(scene + "\n", encoding="utf-8")
    predictions = tmp_path / "predictions"
    diagnostics = tmp_path / "diagnostics"
    predictions.mkdir()
    diagnostics.mkdir()
    # Deliberately opaque, non-pickle bytes: validation only requires the
    # same-run formal artifact to exist and does not compare independent runs.
    (predictions / f"{scene}_boxes.pkl").write_bytes(
        b"nonempty formal output"
    )
    p1_checkpoint = tmp_path / "p1.pt"
    p2_checkpoint = tmp_path / "p2.pt"
    p1_checkpoint.write_bytes(b"frozen p1")
    p2_checkpoint.write_bytes(b"frozen p2")
    payload = _payload(
        scene, _sha(p1_checkpoint), _sha(p2_checkpoint)
    )
    diagnostic = diagnostics / f"{scene}_tracks.npz"
    np.savez_compressed(diagnostic, **payload)
    return (
        scene_list,
        predictions,
        diagnostics,
        p1_checkpoint,
        p2_checkpoint,
        diagnostic,
        payload,
    )


def _validate(artifacts):
    (
        scene_list,
        predictions,
        diagnostics,
        p1_checkpoint,
        p2_checkpoint,
        _,
        _,
    ) = artifacts
    return validate(
        scene_list=scene_list,
        prediction_root=predictions,
        diagnostics_root=diagnostics,
        expected_p1_checkpoint=p1_checkpoint,
        expected_p2_checkpoint=p2_checkpoint,
    )


def _rewrite(diagnostic: Path, payload, **updates) -> None:
    changed = dict(payload)
    changed.update(updates)
    np.savez_compressed(diagnostic, **changed)


def test_valid_observer_artifacts_pass_without_cross_run_bytes(
    tmp_path: Path,
) -> None:
    artifacts = _artifacts(tmp_path)
    report = _validate(artifacts)
    assert report["scene_count"] == 1
    assert report["p2v2_step_count"] == 1
    assert report["p2v2_pre_scene_nms_candidate_count"] == 1
    assert report["p2v2_scene_candidate_count"] == 1
    assert (
        report["formal_output_safety"][
            "cross_run_pickle_byte_equality_required"
        ]
        is False
    )


@pytest.mark.parametrize(
    ("key", "value", "message"),
    [
        (
            "p2v2_reads_semantic_labels",
            np.asarray(True, dtype=bool),
            "unsafe p2v2_reads_semantic_labels",
        ),
        (
            "p2v2_mutation_enabled",
            np.asarray(True, dtype=bool),
            "unsafe p2v2_mutation_enabled",
        ),
        (
            "p2v2_applied_count",
            np.asarray(1, dtype=np.int64),
            "mutated formal output",
        ),
        (
            "p2v2_candidate_applied",
            np.asarray([True], dtype=bool),
            "unsafe P2-v2 candidate flags",
        ),
    ],
)
def test_unsafe_observer_contract_is_rejected(
    tmp_path: Path,
    key: str,
    value: np.ndarray,
    message: str,
) -> None:
    artifacts = _artifacts(tmp_path)
    diagnostic, payload = artifacts[-2:]
    _rewrite(diagnostic, payload, **{key: value})
    with pytest.raises(ValueError, match=message):
        _validate(artifacts)


def test_parent_p2_checkpoint_mismatch_is_rejected(
    tmp_path: Path,
) -> None:
    artifacts = _artifacts(tmp_path)
    diagnostic, payload = artifacts[-2:]
    _rewrite(
        diagnostic,
        payload,
        p2v2_parent_p2_checkpoint_sha256=np.asarray("0" * 64),
    )
    with pytest.raises(ValueError, match="parent P2 checkpoint mismatch"):
        _validate(artifacts)


def test_p2v2_steps_must_align_exactly_with_p2(tmp_path: Path) -> None:
    artifacts = _artifacts(tmp_path)
    diagnostic, payload = artifacts[-2:]
    _rewrite(
        diagnostic,
        payload,
        p2v2_step_frame_ids=np.asarray([6], dtype=np.int64),
    )
    with pytest.raises(ValueError, match="scheduling is not aligned"):
        _validate(artifacts)


def test_candidate_ranges_and_scene_nms_counts_are_checked(
    tmp_path: Path,
) -> None:
    artifacts = _artifacts(tmp_path)
    diagnostic, payload = artifacts[-2:]
    _rewrite(
        diagnostic,
        payload,
        p2v2_candidate_mask_scores=np.asarray(
            [1.1], dtype=np.float32
        ),
    )
    with pytest.raises(
        ValueError, match="invalid p2v2_candidate_mask_scores"
    ):
        _validate(artifacts)

    _rewrite(
        diagnostic,
        payload,
        p2v2_step_candidate_counts=np.asarray([0], dtype=np.int64),
    )
    with pytest.raises(
        ValueError, match="scene NMS candidate count is impossible"
    ):
        _validate(artifacts)


def test_semantic_candidate_fields_are_forbidden(tmp_path: Path) -> None:
    artifacts = _artifacts(tmp_path)
    diagnostic, payload = artifacts[-2:]
    _rewrite(
        diagnostic,
        payload,
        p2v2_candidate_labels=np.asarray(["chair"], dtype=np.str_),
    )
    with pytest.raises(
        ValueError, match="semantic P2-v2 candidate field"
    ):
        _validate(artifacts)
