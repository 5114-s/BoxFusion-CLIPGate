"""Contracts for the offline P2-v2 recall/go-no-go report."""

from __future__ import annotations

import json

import numpy as np
import pytest

from boxfusion.p2_local_mask_geometry import (
    P2V2_DIAGNOSTIC_SCHEMA,
    P2V2_SOURCE,
)
from tests.test_report_p2_occupancy_recall import (
    _box,
    _scene_list,
    _write_scene_assets,
)
from tools.report_p1_residual_recall import center_size_to_corners
from tools.report_p2v2_local_geometry_recall import evaluate


def _add_p2v2(
    diagnostics,
    scene: str,
    *,
    applied: bool = False,
) -> None:
    path = diagnostics / f"{scene}_tracks.npz"
    with np.load(path, allow_pickle=False) as source:
        payload = {
            key: np.array(source[key], copy=True) for key in source.files
        }
    box = _box(15.0)[None]
    parent = _box(20.0)[None]
    payload.update(
        {
            "p2v2_schema": np.asarray(P2V2_DIAGNOSTIC_SCHEMA),
            "p2v2_stage": np.asarray("P2V2"),
            "p2v2_profile": np.asarray(
                "p2v2_local_component_mask_rgbd_observer"
            ),
            "p2v2_enabled": np.asarray(True, dtype=bool),
            "p2v2_observer_only": np.asarray(True, dtype=bool),
            "p2v2_uses_ground_truth": np.asarray(False, dtype=bool),
            "p2v2_reads_semantic_labels": np.asarray(False, dtype=bool),
            "p2v2_mutation_enabled": np.asarray(False, dtype=bool),
            "p2v2_applied_count": np.asarray(
                int(applied), dtype=np.int64
            ),
            "p2v2_complete": np.asarray(True, dtype=bool),
            "p2v2_source": np.asarray(P2V2_SOURCE),
            "p2v2_mask_provider": np.asarray("test-provider"),
            "p2v2_parent_p2_checkpoint_sha256": np.asarray("2" * 64),
            "p2v2_config_json": np.asarray(
                json.dumps(
                    {
                        "enabled": True,
                        "observer_only": True,
                        "mutate": False,
                        "collect_diagnostics": True,
                    }
                )
            ),
            "p2v2_step_frame_ids": np.asarray([0], dtype=np.int64),
            "p2v2_step_provider_steps": np.asarray([1], dtype=np.int64),
            "p2v2_step_selected_voxel_counts": np.asarray(
                [5], dtype=np.int64
            ),
            "p2v2_step_occupancy_component_counts": np.asarray(
                [2], dtype=np.int64
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
            "p2v2_step_candidate_counts": np.asarray([1], dtype=np.int64),
            "p2v2_step_seconds": np.asarray([0.02], dtype=np.float64),
            "p2v2_step_failed": np.asarray([False], dtype=bool),
            "p2v2_step_errors": np.asarray([""], dtype=np.str_),
            "p2v2_candidate_ids": np.asarray(["p2v2:new"], dtype=np.str_),
            "p2v2_parent_p2_candidate_ids": np.asarray(
                ["p2_false"], dtype=np.str_
            ),
            "p2v2_mask_source_ids": np.asarray(["mask:1"], dtype=np.str_),
            "p2v2_candidate_boxes": box,
            "p2v2_candidate_corners": center_size_to_corners(box),
            "p2v2_candidate_parent_boxes": parent,
            "p2v2_candidate_scores": np.asarray([0.8], dtype=np.float32),
            "p2v2_candidate_parent_objectness": np.asarray(
                [0.2], dtype=np.float32
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
                [100], dtype=np.int64
            ),
            "p2v2_candidate_component_voxel_counts": np.asarray(
                [20], dtype=np.int64
            ),
            "p2v2_candidate_selected_voxels_inside": np.asarray(
                [2], dtype=np.int64
            ),
            "p2v2_candidate_anchor_inside": np.asarray([True], dtype=bool),
            "p2v2_candidate_parent_iou": np.asarray(
                [0.0], dtype=np.float32
            ),
            "p2v2_candidate_normalized_center_distance": np.asarray(
                [1.0], dtype=np.float32
            ),
            "p2v2_candidate_extent_ratios": np.ones(
                (1, 3), dtype=np.float32
            ),
            "p2v2_candidate_center_shift_ratios": np.ones(
                (1, 3), dtype=np.float32
            ),
            "p2v2_candidate_applied": np.asarray(
                [applied], dtype=bool
            ),
        }
    )
    np.savez_compressed(path, **payload)


def test_report_measures_incremental_geometry_and_gate(tmp_path):
    scene = "scene0001_00"
    predictions, diagnostics, gt_root, scans = _write_scene_assets(
        tmp_path, scene
    )
    np.save(
        gt_root / f"{scene}_bbox.npy",
        np.stack((_box(0.0), _box(5.0), _box(10.0), _box(15.0))),
    )
    _add_p2v2(diagnostics, scene)
    report = evaluate(
        scenes=(scene,),
        prediction_root=predictions,
        diagnostics_root=diagnostics,
        gt_root=gt_root,
        scans_root=scans,
    )
    row = report["thresholds"]["0.50"]
    assert row["baseline"]["true_positives"] == 3
    assert row["p2v2_incremental"]["true_positives"] == 1
    assert row["p2v2_incremental"]["precision"] == pytest.approx(1.0)
    assert row["combined"]["recall"] == pytest.approx(1.0)
    assert report["runtime_seconds"]["p2v2_incremental"] == pytest.approx(
        0.02
    )
    assert report["go_no_go"]["passed"] is True
    assert report["go_no_go"]["decision"] == "GO_TO_P3"


def test_report_rejects_any_formal_application(tmp_path):
    scene = "scene0001_00"
    predictions, diagnostics, gt_root, scans = _write_scene_assets(
        tmp_path, scene
    )
    _add_p2v2(diagnostics, scene, applied=True)
    with pytest.raises(ValueError, match="mutated formal output"):
        evaluate(
            scenes=(scene,),
            prediction_root=predictions,
            diagnostics_root=diagnostics,
            gt_root=gt_root,
            scans_root=scans,
        )
