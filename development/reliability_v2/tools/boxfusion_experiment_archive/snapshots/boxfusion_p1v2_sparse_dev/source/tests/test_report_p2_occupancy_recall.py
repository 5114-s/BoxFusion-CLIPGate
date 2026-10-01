"""Contracts for the strict P2 occupancy proposal-recall report."""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pytest

from boxfusion.occupancy_topk import (
    P2_DIAGNOSTIC_SCHEMA,
    P2_SOURCE,
)
from boxfusion.residual_proposal import (
    P1_DIAGNOSTIC_SCHEMA,
    P1_FEATURE_NAMES,
)
from tools.report_p1_residual_recall import center_size_to_corners
from tools.report_p2_occupancy_recall import (
    build_report,
    load_predictions,
    main,
)


def _box(center_x: float) -> np.ndarray:
    return np.asarray(
        [center_x, 0.0, 0.0, 2.0, 2.0, 2.0], dtype=np.float32
    )


def _write_scene_assets(
    root: Path,
    scene: str,
    *,
    p1_sha: str = "1" * 64,
    p2_sha: str = "2" * 64,
    diagnostic_updates: dict | None = None,
) -> tuple[Path, Path, Path, Path]:
    predictions = root / "predictions"
    diagnostics = root / "diagnostics"
    gt_root = root / "gt"
    scans = root / "scans"
    predictions.mkdir(parents=True, exist_ok=True)
    diagnostics.mkdir(parents=True, exist_ok=True)
    gt_root.mkdir(parents=True, exist_ok=True)
    (scans / scene).mkdir(parents=True, exist_ok=True)

    b6_box = _box(0.0)
    with (predictions / f"{scene}_boxes.pkl").open("wb") as handle:
        pickle.dump(
            [[("object", center_size_to_corners(b6_box[None])[0], 0.90)]],
            handle,
        )
    np.save(
        gt_root / f"{scene}_bbox.npy",
        np.stack((_box(0.0), _box(5.0), _box(10.0))),
    )
    identity = " ".join(str(value) for value in np.eye(4).reshape(-1))
    (scans / scene / f"{scene}.txt").write_text(
        f"axisAlignment = {identity}\n", encoding="utf-8"
    )

    p1_boxes = np.stack((_box(5.0), _box(0.0)))
    p1_corners = center_size_to_corners(p1_boxes).astype(np.float32)
    p2_boxes = np.stack((_box(5.0), _box(10.0), _box(20.0)))
    p2_corners = center_size_to_corners(p2_boxes).astype(np.float32)
    payload = {
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
        "p1_regression_dim": np.asarray(6, dtype=np.int64),
        "p1_checkpoint_sha256": np.asarray(p1_sha),
        "p1_config_json": np.asarray(
            json.dumps(
                {
                    "enabled": True,
                    "observer_only": True,
                    "mutate": False,
                }
            )
        ),
        "p1_feature_names": np.asarray(P1_FEATURE_NAMES, dtype=np.str_),
        "p1_step_frame_ids": np.asarray([0], dtype=np.int64),
        "p1_step_provider_steps": np.asarray([1], dtype=np.int64),
        "p1_step_voxelize_seconds": np.asarray([0.10], dtype=np.float64),
        "p1_step_head_seconds": np.asarray([0.20], dtype=np.float64),
        "p1_step_nms_seconds": np.asarray([0.30], dtype=np.float64),
        "p1_candidate_ids": np.asarray(
            ["shared", "p1_duplicate_b6"], dtype=np.str_
        ),
        "p1_candidate_frame_ids": np.asarray([0, 0], dtype=np.int64),
        "p1_candidate_boxes": p1_boxes,
        "p1_candidate_corners": p1_corners,
        "p1_candidate_scores": np.asarray([0.80, 0.70], dtype=np.float32),
        "p1_candidate_objectness": np.asarray(
            [0.80, 0.70], dtype=np.float32
        ),
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
        "p2_source": np.asarray(P2_SOURCE),
        "p2_checkpoint_sha256": np.asarray(p2_sha),
        "p2_config_json": np.asarray(
            json.dumps(
                {
                    "enabled": True,
                    "observer_only": True,
                    "mutate": False,
                }
            )
        ),
        "p2_feature_names": np.asarray(P1_FEATURE_NAMES, dtype=np.str_),
        "p2_step_frame_ids": np.asarray([0], dtype=np.int64),
        "p2_step_provider_steps": np.asarray([1], dtype=np.int64),
        "p2_step_input_voxel_counts": np.asarray([10], dtype=np.int64),
        "p2_step_eligible_voxel_counts": np.asarray([8], dtype=np.int64),
        "p2_step_selected_voxel_counts": np.asarray([5], dtype=np.int64),
        "p2_step_candidate_counts": np.asarray([3], dtype=np.int64),
        "p2_step_seconds": np.asarray([0.05], dtype=np.float64),
        "p2_candidate_ids": np.asarray(
            ["shared", "p2_new", "p2_false"], dtype=np.str_
        ),
        "p2_candidate_boxes": p2_boxes,
        "p2_candidate_corners": p2_corners,
        "p2_candidate_objectness": np.asarray(
            [0.80, 0.65, 0.20], dtype=np.float32
        ),
        "p2_candidate_occupancy_scores": np.asarray(
            [0.95, 0.90, 0.80], dtype=np.float32
        ),
        "p2_candidate_occupancy_ranks": np.asarray(
            [0, 1, 2], dtype=np.int64
        ),
    }
    if diagnostic_updates:
        payload.update(diagnostic_updates)
    np.savez_compressed(
        diagnostics / f"{scene}_tracks.npz", **payload
    )
    return predictions, diagnostics, gt_root, scans


def _scene_list(root: Path, *scenes: str) -> Path:
    path = root / "scenes.txt"
    path.write_text("".join(f"{scene}\n" for scene in scenes), encoding="utf-8")
    return path


def test_report_has_p1_p2_fixed_streams_unions_novel_precision_and_runtime(
    tmp_path,
):
    scene = "scene0001_00"
    predictions, diagnostics, gt_root, scans = _write_scene_assets(
        tmp_path, scene
    )
    report = build_report(
        scene_list=_scene_list(tmp_path, scene),
        prediction_root=predictions,
        diagnostics_root=diagnostics,
        gt_root=gt_root,
        scans_root=scans,
        thresholds=(0.50,),
    )

    assert report["observer_only"] is True
    assert report["safety"]["validated"] is True
    assert report["safety"]["mutation_enabled"] is False
    assert report["candidate_counts"] == {
        "b6": 1,
        "p1_only": 2,
        "p2_only": 3,
        "b6_p1_union": 3,
        "b6_p2_union": 4,
        "p1_p2_union": 4,
        "b6_p1_p2_union": 5,
        "p1_p2_shared_candidate_ids": 1,
    }
    row = report["thresholds"]["0.50"]
    assert row["sources"]["b6"]["true_positives"] == 1
    assert row["sources"]["p1_only"]["true_positives"] == 2
    assert row["sources"]["p2_only"]["true_positives"] == 2
    assert row["sources"]["b6_p1_union"]["true_positives"] == 2
    assert row["sources"]["b6_p2_union"]["true_positives"] == 3
    assert row["sources"]["p1_p2_union"]["true_positives"] == 3
    assert row["sources"]["b6_p1_p2_union"]["true_positives"] == 3
    assert row["novel"]["p1_vs_b6"]["true_positives"] == 1
    assert row["novel"]["p1_vs_b6"]["precision"] == pytest.approx(0.5)
    assert row["novel"]["p2_vs_b6"]["true_positives"] == 2
    assert row["novel"]["p2_vs_b6"]["precision"] == pytest.approx(2 / 3)
    assert row["novel"]["p2_vs_b6_p1"]["true_positives"] == 1
    assert row["novel"]["p2_vs_b6_p1"]["precision"] == pytest.approx(
        1 / 3
    )
    assert report["runtime_seconds"]["p1"] == pytest.approx(0.60)
    assert report["runtime_seconds"]["p2_incremental"] == pytest.approx(
        0.05
    )
    assert report["runtime_seconds"]["p2_total"] == pytest.approx(0.65)


@pytest.mark.parametrize(
    "updates,match",
    [
        (
            {"p2_mutation_enabled": np.asarray(True, dtype=bool)},
            "unsafe p2_mutation_enabled",
        ),
        (
            {"p2_applied_count": np.asarray(1, dtype=np.int64)},
            "p2 mutated formal output",
        ),
        (
            {"p2_uses_ground_truth": np.asarray(True, dtype=bool)},
            "unsafe p2_uses_ground_truth",
        ),
        (
            {"p2_observer_only": np.asarray(False, dtype=bool)},
            "unsafe p2_observer_only",
        ),
        (
            {"p2_complete": np.asarray(False, dtype=bool)},
            "unsafe p2_complete",
        ),
        (
            {"p2_mutation_enabled": np.asarray(0, dtype=np.int64)},
            "must be Boolean",
        ),
    ],
)
def test_report_rejects_every_unsafe_p2_contract(tmp_path, updates, match):
    scene = "scene0001_00"
    predictions, diagnostics, gt_root, scans = _write_scene_assets(
        tmp_path, scene, diagnostic_updates=updates
    )
    with pytest.raises(ValueError, match=match):
        build_report(
            scene_list=_scene_list(tmp_path, scene),
            prediction_root=predictions,
            diagnostics_root=diagnostics,
            gt_root=gt_root,
            scans_root=scans,
        )


def test_report_rejects_impossible_step_counts(tmp_path):
    scene = "scene0001_00"
    predictions, diagnostics, gt_root, scans = _write_scene_assets(
        tmp_path,
        scene,
        diagnostic_updates={
            "p2_step_selected_voxel_counts": np.asarray(
                [9], dtype=np.int64
            )
        },
    )
    with pytest.raises(ValueError, match="impossible P2 step count"):
        build_report(
            scene_list=_scene_list(tmp_path, scene),
            prediction_root=predictions,
            diagnostics_root=diagnostics,
            gt_root=gt_root,
            scans_root=scans,
        )


def test_shared_p1_p2_id_must_represent_the_same_frozen_proposal(tmp_path):
    scene = "scene0001_00"
    changed_boxes = np.stack((_box(6.0), _box(10.0), _box(20.0)))
    predictions, diagnostics, gt_root, scans = _write_scene_assets(
        tmp_path,
        scene,
        diagnostic_updates={
            "p2_candidate_boxes": changed_boxes,
            "p2_candidate_corners": center_size_to_corners(
                changed_boxes
            ).astype(np.float32),
        },
    )
    with pytest.raises(ValueError, match="shared candidate ID disagrees"):
        build_report(
            scene_list=_scene_list(tmp_path, scene),
            prediction_root=predictions,
            diagnostics_root=diagnostics,
            gt_root=gt_root,
            scans_root=scans,
        )


def test_report_rejects_checkpoint_mixing_across_scenes(tmp_path):
    first = "scene0001_00"
    second = "scene0002_00"
    predictions, diagnostics, gt_root, scans = _write_scene_assets(
        tmp_path, first
    )
    _write_scene_assets(tmp_path, second, p2_sha="3" * 64)
    with pytest.raises(ValueError, match="SHA changed across P2 scenes"):
        build_report(
            scene_list=_scene_list(tmp_path, first, second),
            prediction_root=predictions,
            diagnostics_root=diagnostics,
            gt_root=gt_root,
            scans_root=scans,
        )


def test_cli_writes_the_same_json_it_prints(tmp_path, capsys):
    scene = "scene0001_00"
    predictions, diagnostics, gt_root, scans = _write_scene_assets(
        tmp_path, scene
    )
    scene_list = _scene_list(tmp_path, scene)
    output = tmp_path / "report.json"
    assert (
        main(
            [
                "--scene-list",
                str(scene_list),
                "--prediction-root",
                str(predictions),
                "--diagnostics-root",
                str(diagnostics),
                "--gt-root",
                str(gt_root),
                "--scans-root",
                str(scans),
                "--thresholds",
                "0.5",
                "--output",
                str(output),
            ]
        )
        == 0
    )
    printed = json.loads(capsys.readouterr().out)
    saved = json.loads(output.read_text(encoding="utf-8"))
    assert printed == saved
    assert saved["schema"] == "boxfusion.p2_occupancy_recall_report.v1"


def test_real_detection_major_pickle_layout_is_supported(tmp_path):
    scene = "scene0001_00"
    predictions, diagnostics, gt_root, scans = _write_scene_assets(
        tmp_path, scene
    )
    prediction = predictions / f"{scene}_boxes.pkl"
    with prediction.open("rb") as handle:
        one_batch = pickle.load(handle)
    with prediction.open("wb") as handle:
        pickle.dump(one_batch[0], handle)

    report = build_report(
        scene_list=_scene_list(tmp_path, scene),
        prediction_root=predictions,
        diagnostics_root=diagnostics,
        gt_root=gt_root,
        scans_root=scans,
        thresholds=(0.50,),
    )
    assert report["candidate_counts"]["b6"] == 1
    assert report["thresholds"]["0.50"]["b6_recall"] == pytest.approx(
        1 / 3
    )


def test_one_batch_with_exactly_three_detections_is_not_misparsed(tmp_path):
    prediction = tmp_path / "three.pkl"
    detections = [
        ("object", center_size_to_corners(_box(value)[None])[0], 0.9)
        for value in (0.0, 5.0, 10.0)
    ]
    with prediction.open("wb") as handle:
        pickle.dump([detections], handle)
    loaded = load_predictions(prediction)
    assert loaded.corners_world.shape == (3, 8, 3)
    np.testing.assert_allclose(loaded.scores, [0.9, 0.9, 0.9])
