"""Contracts for the strict observer-only P1G geometry report."""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pytest

from tools.report_p1_residual_recall import center_size_to_corners
from tools.report_p1g_geometry import (
    P1G_DIAGNOSTIC_SCHEMA,
    P1G_PROFILE,
    REPORT_SCHEMA,
    build_report,
    load_p1g_scene,
)


def _box(center_x: float, extent: float = 2.0) -> np.ndarray:
    return np.asarray(
        [center_x, 0.0, 0.0, extent, extent, extent],
        dtype=np.float32,
    )


def _write_predictions(path: Path) -> None:
    corners = center_size_to_corners(_box(0.0)[None])[0].astype(
        np.float32
    )
    with path.open("wb") as handle:
        pickle.dump(
            [[(0, corners, 0.95)]],
            handle,
            protocol=pickle.HIGHEST_PROTOCOL,
        )


def _write_diagnostic(
    path: Path,
    *,
    scene: str,
    parent_boxes: np.ndarray,
    refined_boxes: np.ndarray,
) -> None:
    count = len(parent_boxes)
    parent_corners = center_size_to_corners(parent_boxes).astype(
        np.float32
    )
    refined_corners = center_size_to_corners(refined_boxes).astype(
        np.float32
    )
    parent_ids = np.asarray(
        [f"{scene}:000001:{index}:0:0" for index in range(count)],
        dtype=np.str_,
    )
    scores = np.linspace(0.80, 0.60, count, dtype=np.float32)
    candidate_seconds = np.full(count, 0.02, dtype=np.float64)
    np.savez_compressed(
        path,
        scene_id=np.asarray(scene),
        p1_schema=np.asarray(
            "boxfusion.p1.residual_proposal_observer.v1"
        ),
        p1_stage=np.asarray("P1S"),
        p1_profile=np.asarray("p1s_native_sparse_context_observer"),
        p1_enabled=np.asarray(True, dtype=bool),
        p1_observer_only=np.asarray(True, dtype=bool),
        p1_uses_ground_truth=np.asarray(False, dtype=bool),
        p1_reads_semantic_labels=np.asarray(False, dtype=bool),
        p1_mutation_enabled=np.asarray(False, dtype=bool),
        p1_applied_count=np.asarray(0, dtype=np.int64),
        p1_complete=np.asarray(True, dtype=bool),
        p1_class_agnostic=np.asarray(True, dtype=bool),
        p1_head_architecture=np.asarray("native_sparse_context_v1"),
        p1_target_assignment_scope=np.asarray("snapshot_inside_only"),
        p1_candidate_ids=parent_ids,
        p1_candidate_boxes=np.asarray(parent_boxes, dtype=np.float32),
        p1_candidate_corners=parent_corners,
        p1_candidate_scores=scores,
        p1_step_voxelize_seconds=np.asarray([0.10], dtype=np.float64),
        p1_step_head_seconds=np.asarray([0.15], dtype=np.float64),
        p1_step_nms_seconds=np.asarray([0.05], dtype=np.float64),
        p1g_schema=np.asarray(P1G_DIAGNOSTIC_SCHEMA),
        p1g_stage=np.asarray("P1G"),
        p1g_profile=np.asarray(P1G_PROFILE),
        p1g_parent_stage=np.asarray("P1S"),
        p1g_enabled=np.asarray(True, dtype=bool),
        p1g_observer_only=np.asarray(True, dtype=bool),
        p1g_uses_ground_truth=np.asarray(False, dtype=bool),
        p1g_reads_semantic_labels=np.asarray(False, dtype=bool),
        p1g_mutation_enabled=np.asarray(False, dtype=bool),
        p1g_applied_count=np.asarray(0, dtype=np.int64),
        p1g_complete=np.asarray(True, dtype=bool),
        p1g_class_agnostic=np.asarray(True, dtype=bool),
        p1g_regression_dim=np.asarray(6, dtype=np.int64),
        p1g_parent_checkpoint_sha256=np.asarray("a" * 64),
        p1g_config_json=np.asarray(
            json.dumps(
                {
                    "enabled": True,
                    "observer_only": True,
                    "mutate": False,
                },
                sort_keys=True,
            )
        ),
        p1g_runtime_seconds=np.asarray(0.10, dtype=np.float64),
        p1g_failure_count=np.asarray(0, dtype=np.int64),
        p1g_parent_candidate_ids=parent_ids,
        p1g_refined_candidate_ids=np.asarray(
            [f"p1g:{value}" for value in parent_ids.tolist()],
            dtype=np.str_,
        ),
        p1g_parent_boxes=np.asarray(parent_boxes, dtype=np.float32),
        p1g_parent_corners=parent_corners,
        p1g_refined_boxes=np.asarray(refined_boxes, dtype=np.float32),
        p1g_refined_corners=refined_corners,
        p1g_candidate_scores=scores,
        p1g_candidate_applied=np.zeros(count, dtype=bool),
        p1g_is_candidate=np.ones(count, dtype=bool),
        p1g_reasons=np.asarray(["candidate"] * count, dtype=np.str_),
        p1g_sources=np.asarray(
            ["p1_multiview_geometry"] * count, dtype=np.str_
        ),
        p1g_matched_view_counts=np.full(count, 3, dtype=np.int64),
        p1g_selected_view_counts=np.full(count, 2, dtype=np.int64),
        p1g_selected_frame_ids=np.tile(
            np.asarray([1, 2, -1, -1, -1], dtype=np.int64),
            (count, 1),
        ),
        p1g_cropped_point_counts=np.full(count, 128, dtype=np.int64),
        p1g_face_residuals=np.zeros((count, 3, 2), dtype=np.float32),
        p1g_face_support=np.ones((count, 3, 2), dtype=np.float32),
        p1g_face_uncertainty=np.ones(
            (count, 3, 2), dtype=np.float32
        ),
        p1g_face_supported=np.ones((count, 3, 2), dtype=bool),
        p1g_feature_vectors=np.zeros((count, 48), dtype=np.float32),
        p1g_step_total_seconds=np.full(
            count, 0.08 / count, dtype=np.float64
        ),
        p1g_candidate_seconds=candidate_seconds,
    )


def _assets(
    tmp_path: Path,
    *,
    parent_boxes: np.ndarray | None = None,
    refined_boxes: np.ndarray | None = None,
) -> dict[str, Path | str]:
    scene = "scene0001_00"
    prediction_root = tmp_path / "predictions"
    diagnostics_root = tmp_path / "diagnostics"
    gt_root = tmp_path / "gt"
    scans_root = tmp_path / "scans"
    for directory in (
        prediction_root,
        diagnostics_root,
        gt_root,
        scans_root / scene,
    ):
        directory.mkdir(parents=True)
    scene_list = tmp_path / "scenes.txt"
    scene_list.write_text(scene + "\n", encoding="utf-8")
    transform = " ".join(
        str(float(value)) for value in np.eye(4).reshape(-1)
    )
    (scans_root / scene / f"{scene}.txt").write_text(
        f"axisAlignment = {transform}\n", encoding="utf-8"
    )
    np.save(
        gt_root / f"{scene}_bbox.npy",
        np.stack(
            (
                _box(0.0),
                _box(5.0),
                _box(10.0),
                _box(15.0),
            )
        ),
    )
    _write_predictions(prediction_root / f"{scene}_boxes.pkl")
    if parent_boxes is None:
        parent_boxes = np.stack(
            (_box(5.0, 2.8), _box(10.0, 2.0), _box(15.0, 2.8))
        )
    if refined_boxes is None:
        refined_boxes = np.stack(
            (_box(5.0, 2.0), _box(10.0, 2.5), _box(15.0, 2.8))
        )
    _write_diagnostic(
        diagnostics_root / f"{scene}_tracks.npz",
        scene=scene,
        parent_boxes=parent_boxes,
        refined_boxes=refined_boxes,
    )
    return {
        "scene": scene,
        "scene_list": scene_list,
        "prediction_root": prediction_root,
        "diagnostics_root": diagnostics_root,
        "diagnostic": diagnostics_root / f"{scene}_tracks.npz",
        "gt_root": gt_root,
        "scans_root": scans_root,
    }


def _report(paths, **updates):
    options = {
        "minimum_novel_tp50": 2,
        "maximum_p1g_runtime_seconds_per_scene": 0.18,
        "maximum_total_runtime_seconds_per_scene": 0.80,
        "maximum_candidates_per_scene": 256.0,
    }
    options.update(updates)
    return build_report(
        scene_list=paths["scene_list"],
        prediction_root=paths["prediction_root"],
        diagnostics_root=paths["diagnostics_root"],
        gt_root=paths["gt_root"],
        scans_root=paths["scans_root"],
        **options,
    )


def _rewrite(path: Path, **updates) -> None:
    with np.load(path, allow_pickle=False) as source:
        payload = {
            key: np.array(source[key], copy=True) for key in source.files
        }
    payload.update(updates)
    np.savez_compressed(path, **payload)


def test_report_compares_parent_refined_oracle_and_geometry_crossings(
    tmp_path,
):
    paths = _assets(tmp_path)
    report = _report(paths)

    assert report["schema"] == REPORT_SCHEMA
    assert report["observer_only"] is True
    assert report["candidate_count"] == 3
    assert report["ground_truth_count"] == 4
    row25 = report["thresholds"]["0.25"]
    assert row25["parent"]["novel_true_positives"] == 3
    assert row25["refined"]["novel_true_positives"] == 3
    row50 = report["thresholds"]["0.50"]
    assert row50["b6"]["candidate_true_positives"] == 1
    assert row50["parent"]["novel_true_positives"] == 1
    assert row50["refined"]["novel_true_positives"] == 2
    assert row50["oracle"]["novel_true_positives"] == 2
    assert row50["refined_minus_parent"]["novel_true_positives"] == 1

    geometry = report["geometry"]
    assert geometry["threshold_crossings"]["0.50"] == {
        "up": 1,
        "down": 0,
        "above": 1,
        "below": 1,
    }
    assert geometry["improved"] == 1
    assert geometry["harmed"] == 1
    assert geometry["unchanged"] == 1
    assert geometry["severe_harm_le_minus_0p05"] == 1
    assert report["identity_vs_refined_oracle"]["uses_ground_truth"] is True
    assert report["identity_vs_refined_oracle"]["deployable"] is False
    assert report["identity_vs_refined_oracle"]["selected_refined"] == 1
    assert report["go_no_go"]["passes"] is True
    assert report["diagnosis"]["classification"] == "production_effective"
    json.dumps(report, allow_nan=False)


def test_oracle_separates_parameter_problem_from_method_limit(tmp_path):
    # Production refines the exact parent away from IoU 0.50.  The identity
    # branch of the diagnostic oracle recovers it, so this is not evidence
    # that association/evidence is intrinsically insufficient.
    parent = np.stack(
        (_box(5.0, 2.8), _box(10.0, 2.0), _box(15.0, 2.8))
    )
    refined = np.stack(
        (_box(5.0, 2.8), _box(10.0, 2.8), _box(15.0, 2.8))
    )
    paths = _assets(
        tmp_path / "parameter",
        parent_boxes=parent,
        refined_boxes=refined,
    )
    report = _report(paths, minimum_novel_tp50=1)
    assert (
        report["thresholds"]["0.50"]["refined"][
            "novel_true_positives"
        ]
        == 0
    )
    assert (
        report["thresholds"]["0.50"]["oracle"][
            "novel_true_positives"
        ]
        == 1
    )
    assert report["go_no_go"]["passes"] is False
    assert (
        report["diagnosis"]["classification"]
        == "parameter_or_internal_gate_problem"
    )

    # Neither identity nor refined geometry reaches IoU 0.50.  Adjusting a
    # raw/refined selector cannot solve this evidence family.
    weak = np.stack(
        (_box(5.0, 2.8), _box(10.0, 2.8), _box(15.0, 2.8))
    )
    paths = _assets(
        tmp_path / "method",
        parent_boxes=weak,
        refined_boxes=weak,
    )
    report = _report(paths, minimum_novel_tp50=1)
    assert (
        report["thresholds"]["0.50"]["oracle"][
            "novel_true_positives"
        ]
        == 0
    )
    assert (
        report["diagnosis"]["classification"]
        == "association_or_evidence_method_problem"
    )


@pytest.mark.parametrize(
    "update,message",
    [
        (
            {"p1g_mutation_enabled": np.asarray(True, dtype=bool)},
            "unsafe p1g_mutation_enabled",
        ),
        (
            {"p1g_applied_count": np.asarray(1, dtype=np.int64)},
            "applied formal output",
        ),
        (
            {
                "p1g_candidate_applied": np.asarray(
                    [True, False, False], dtype=bool
                )
            },
            "candidate rows mutated formal output",
        ),
    ],
)
def test_loader_rejects_any_observer_mutation(tmp_path, update, message):
    paths = _assets(tmp_path)
    _rewrite(paths["diagnostic"], **update)
    with pytest.raises(ValueError, match=message):
        load_p1g_scene(paths["diagnostic"], scene_id=paths["scene"])


def test_loader_rejects_parent_reordering_or_geometry_change(tmp_path):
    paths = _assets(tmp_path / "ids")
    with np.load(paths["diagnostic"], allow_pickle=False) as source:
        ids = np.array(source["p1g_parent_candidate_ids"], copy=True)
    _rewrite(paths["diagnostic"], p1g_parent_candidate_ids=ids[::-1])
    with pytest.raises(ValueError, match="parent IDs/order"):
        load_p1g_scene(paths["diagnostic"], scene_id=paths["scene"])

    paths = _assets(tmp_path / "geometry")
    with np.load(paths["diagnostic"], allow_pickle=False) as source:
        corners = np.array(source["p1g_parent_corners"], copy=True)
    corners[0, 0, 0] += 0.01
    _rewrite(paths["diagnostic"], p1g_parent_corners=corners)
    with pytest.raises(ValueError, match="parent corners changed"):
        load_p1g_scene(paths["diagnostic"], scene_id=paths["scene"])


def test_loader_rejects_misaligned_geometry_evidence(tmp_path):
    paths = _assets(tmp_path)
    _rewrite(
        paths["diagnostic"],
        p1g_selected_view_counts=np.asarray([2, 3, 2], dtype=np.int64),
    )
    with pytest.raises(ValueError, match="selected frame IDs disagree"):
        load_p1g_scene(paths["diagnostic"], scene_id=paths["scene"])


def test_runtime_and_candidate_limits_are_operational_not_method_failure(
    tmp_path,
):
    paths = _assets(tmp_path)
    report = _report(
        paths,
        maximum_p1g_runtime_seconds_per_scene=0.05,
    )
    assert report["go_no_go"]["p1g_runtime_passes"] is False
    assert report["go_no_go"]["passes"] is False
    # Geometry and its GT-only selector pass.  Runtime/implementation tuning
    # is not evidence that the association/evidence method lacks potential.
    assert (
        report["diagnosis"]["classification"]
        == "parameter_or_internal_gate_problem"
    )


def test_report_requires_exact_prediction_and_diagnostic_scene_sets(
    tmp_path,
):
    paths = _assets(tmp_path)
    (paths["prediction_root"] / "scene9999_00_boxes.pkl").write_bytes(b"x")
    with pytest.raises(ValueError, match="prediction scene set mismatch"):
        _report(paths)
