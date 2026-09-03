"""Contracts for the offline P2-v3 reliability-fusion recall report."""

from __future__ import annotations

import json

import numpy as np
import pytest

from tests.test_report_p2_occupancy_recall import (
    _box,
    _write_scene_assets,
)
from tests.test_report_p2v2_local_geometry_recall import _add_p2v2
from tools.report_p1_residual_recall import center_size_to_corners
from tools.report_p2v3_reliability_fusion_recall import evaluate
from tools.validate_p2v3_run_artifacts import (
    P2V3_DIAGNOSTIC_SCHEMA,
    P2V3_PROFILE,
    P2V3_RELIABILITY_CONTRACT,
    P2V3_SOURCE,
)


def _replace_p2v2_component(
    diagnostics,
    scene: str,
    component_box: np.ndarray,
) -> None:
    path = diagnostics / f"{scene}_tracks.npz"
    with np.load(path, allow_pickle=False) as source:
        payload = {
            key: np.array(source[key], copy=True) for key in source.files
        }
    box = np.asarray(component_box, dtype=np.float32).reshape(1, 6)
    payload["p2v2_candidate_boxes"] = box
    payload["p2v2_candidate_corners"] = center_size_to_corners(box)
    np.savez_compressed(path, **payload)


def _add_p2v3(
    diagnostics,
    scene: str,
    *,
    component_box: np.ndarray,
    fused_box: np.ndarray,
    applied: bool = False,
) -> None:
    path = diagnostics / f"{scene}_tracks.npz"
    with np.load(path, allow_pickle=False) as source:
        payload = {
            key: np.array(source[key], copy=True) for key in source.files
        }
    component = np.asarray(component_box, dtype=np.float32).reshape(1, 6)
    parent = _box(15.0).reshape(1, 6)
    fused = np.asarray(fused_box, dtype=np.float32).reshape(1, 6)
    bounded = np.asarray([0.8], dtype=np.float32)
    payload.update(
        {
            "p2v3_schema": np.asarray(P2V3_DIAGNOSTIC_SCHEMA),
            "p2v3_stage": np.asarray("P2V3"),
            "p2v3_profile": np.asarray(P2V3_PROFILE),
            "p2v3_source": np.asarray(P2V3_SOURCE),
            "p2v3_parent_p2v2_schema": payload["p2v2_schema"],
            "p2v3_reliability_contract": np.asarray(
                P2V3_RELIABILITY_CONTRACT
            ),
            "p2v3_parent_p2v2_source": payload["p2v2_source"],
            "p2v3_enabled": np.asarray(True, dtype=bool),
            "p2v3_observer_only": np.asarray(True, dtype=bool),
            "p2v3_uses_ground_truth": np.asarray(False, dtype=bool),
            "p2v3_reads_semantic_labels": np.asarray(False, dtype=bool),
            "p2v3_mutation_enabled": np.asarray(False, dtype=bool),
            "p2v3_applied_count": np.asarray(
                int(applied), dtype=np.int64
            ),
            "p2v3_complete": np.asarray(True, dtype=bool),
            "p2v3_parent_p2_checkpoint_sha256": payload[
                "p2_checkpoint_sha256"
            ],
            "p2v3_config_json": np.asarray(
                json.dumps(
                    {
                        "enabled": True,
                        "observer_only": True,
                        "mutate": False,
                        "collect_diagnostics": True,
                        "minimum_component_weight": 0.35,
                        "maximum_component_weight": 0.85,
                        "max_candidates_per_step": 16,
                        "max_scene_candidates": 64,
                    }
                )
            ),
            "p2v3_step_frame_ids": np.asarray([0], dtype=np.int64),
            "p2v3_step_provider_steps": np.asarray([1], dtype=np.int64),
            "p2v3_step_input_candidate_counts": np.asarray(
                [1], dtype=np.int64
            ),
            "p2v3_step_eligible_candidate_counts": np.asarray(
                [1], dtype=np.int64
            ),
            "p2v3_step_candidate_counts": np.asarray(
                [1], dtype=np.int64
            ),
            "p2v3_step_seconds": np.asarray([0.01], dtype=np.float64),
            "p2v3_step_failed": np.asarray([False], dtype=bool),
            "p2v3_step_errors": np.asarray([""], dtype=np.str_),
            "p2v3_candidate_ids": np.asarray(
                ["p2v3:new"], dtype=np.str_
            ),
            "p2v3_parent_p2v2_candidate_ids": np.asarray(
                ["p2v2:new"], dtype=np.str_
            ),
            "p2v3_parent_p2_candidate_ids": np.asarray(
                ["p2_false"], dtype=np.str_
            ),
            "p2v3_mask_source_ids": np.asarray(
                ["mask:1"], dtype=np.str_
            ),
            "p2v3_candidate_component_boxes": component,
            "p2v3_candidate_component_corners": center_size_to_corners(
                component
            ),
            "p2v3_candidate_parent_boxes": parent,
            "p2v3_candidate_parent_corners": center_size_to_corners(parent),
            "p2v3_candidate_fused_boxes": fused,
            "p2v3_candidate_fused_corners": center_size_to_corners(fused),
            "p2v3_candidate_scores": bounded,
            "p2v3_candidate_component_weights": np.asarray(
                [0.7], dtype=np.float32
            ),
            "p2v3_candidate_center_component_weights": np.full(
                (1, 3), 0.35, dtype=np.float32
            ),
            "p2v3_candidate_extent_component_weights": np.full(
                (1, 3), 0.6, dtype=np.float32
            ),
            "p2v3_candidate_component_reliabilities": bounded,
            "p2v3_candidate_parent_reliabilities": np.asarray(
                [0.4], dtype=np.float32
            ),
            "p2v3_candidate_mask_reliabilities": bounded,
            "p2v3_candidate_depth_reliabilities": bounded,
            "p2v3_candidate_support_reliabilities": bounded,
            "p2v3_candidate_agreement_reliabilities": bounded,
            "p2v3_candidate_applied": np.asarray(
                [applied], dtype=bool
            ),
        }
    )
    np.savez_compressed(path, **payload)


def _make_scene(tmp_path, *, applied: bool = False):
    scene = "scene0001_00"
    predictions, diagnostics, gt_root, scans = _write_scene_assets(
        tmp_path, scene
    )
    # B6/P1/P2 cover 0, 5 and 10.  The fourth target at 15 is the only
    # uncovered object.  The paired P2-v2 component misses it, while the
    # P2-v3 fused geometry reaches it.
    np.save(
        gt_root / f"{scene}_bbox.npy",
        np.stack((_box(0.0), _box(5.0), _box(10.0), _box(15.0))),
    )
    _add_p2v2(diagnostics, scene)
    component = _box(16.5)
    _replace_p2v2_component(diagnostics, scene, component)
    _add_p2v3(
        diagnostics,
        scene,
        component_box=component,
        fused_box=_box(15.525),
        applied=applied,
    )
    return scene, predictions, diagnostics, gt_root, scans


def test_report_uses_frozen_baseline_and_paired_geometry_control(tmp_path):
    scene, predictions, diagnostics, gt_root, scans = _make_scene(tmp_path)
    report = evaluate(
        scenes=(scene,),
        prediction_root=predictions,
        diagnostics_root=diagnostics,
        gt_root=gt_root,
        scans_root=scans,
    )
    row = report["thresholds"]["0.50"]
    assert row["baseline"]["true_positives"] == 3
    assert (
        row["paired_p2v2_component_control"]["true_positives"] == 0
    )
    assert row["p2v3_fused_incremental"]["true_positives"] == 1
    assert row["p2v3_fused_incremental"]["precision"] == pytest.approx(
        1.0
    )
    assert row["paired_delta"][
        "fused_minus_component_true_positives"
    ] == 1
    assert row["paired_delta"][
        "component_to_fused_cross_up_count"
    ] == 1
    assert row["combined"]["recall"] == pytest.approx(1.0)
    assert report["go_no_go"]["baseline"] == "b6_p1_p2_union"
    assert (
        report["go_no_go"][
            "paired_component_control_is_gate_baseline"
        ]
        is False
    )
    assert report["go_no_go"]["passed"] is True
    assert report["go_no_go"]["decision"] == "GO_TO_P3"
    assert report["runtime_seconds"]["p2v3_incremental"] == pytest.approx(
        0.01
    )
    assert report["reliability_summary"]["component_weight"][
        "q50"
    ] == pytest.approx(0.7)


def test_report_rejects_p2v3_formal_application(tmp_path):
    scene, predictions, diagnostics, gt_root, scans = _make_scene(
        tmp_path, applied=True
    )
    with pytest.raises(ValueError, match="mutated formal output"):
        evaluate(
            scenes=(scene,),
            prediction_root=predictions,
            diagnostics_root=diagnostics,
            gt_root=gt_root,
            scans_root=scans,
        )


def test_report_rejects_component_that_is_not_the_paired_p2v2_box(
    tmp_path,
):
    scene, predictions, diagnostics, gt_root, scans = _make_scene(tmp_path)
    path = diagnostics / f"{scene}_tracks.npz"
    with np.load(path, allow_pickle=False) as source:
        payload = {
            key: np.array(source[key], copy=True) for key in source.files
        }
    changed = _box(14.0).reshape(1, 6)
    payload["p2v3_candidate_component_boxes"] = changed
    payload["p2v3_candidate_component_corners"] = center_size_to_corners(
        changed
    )
    changed_fused = _box(14.65).reshape(1, 6)
    payload["p2v3_candidate_fused_boxes"] = changed_fused
    payload["p2v3_candidate_fused_corners"] = center_size_to_corners(
        changed_fused
    )
    np.savez_compressed(path, **payload)
    with pytest.raises(ValueError, match="disagrees with P2-v2 parent"):
        evaluate(
            scenes=(scene,),
            prediction_root=predictions,
            diagnostics_root=diagnostics,
            gt_root=gt_root,
            scans_root=scans,
        )
