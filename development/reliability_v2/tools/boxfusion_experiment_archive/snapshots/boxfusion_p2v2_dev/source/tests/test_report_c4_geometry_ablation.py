import json
import pickle
from pathlib import Path

import numpy as np
import pytest

from tools.report_c4_geometry_ablation import (
    C4_DIAGNOSTIC_SCHEMA,
    REPORT_SCHEMA,
    build_report,
    load_c4_diagnostics,
)


_SIGNS = np.asarray(
    [
        [-1.0, -1.0, -1.0],
        [-1.0, -1.0, 1.0],
        [-1.0, 1.0, -1.0],
        [-1.0, 1.0, 1.0],
        [1.0, -1.0, -1.0],
        [1.0, -1.0, 1.0],
        [1.0, 1.0, -1.0],
        [1.0, 1.0, 1.0],
    ],
    dtype=np.float64,
)


def _corners(center, extent):
    return (
        np.asarray(center, dtype=np.float64)[None]
        + _SIGNS * (0.5 * np.asarray(extent, dtype=np.float64)[None])
    )


def _write_predictions(path, detections):
    with path.open("wb") as handle:
        pickle.dump([detections], handle, protocol=pickle.HIGHEST_PROTOCOL)


def _synthetic_c4_observer(tmp_path):
    scene = "scene0000_00"
    pred_root = tmp_path / "predictions"
    diagnostics_root = tmp_path / "diagnostics"
    gt_root = tmp_path / "gt"
    scan_root = tmp_path / "scans"
    for root in (pred_root, diagnostics_root, gt_root, scan_root):
        root.mkdir()

    scene_scan_root = scan_root / scene
    scene_scan_root.mkdir()
    transform_text = " ".join(
        str(float(value)) for value in np.eye(4).reshape(-1)
    )
    (scene_scan_root / f"{scene}.txt").write_text(
        f"axisAlignment = {transform_text}\n", encoding="utf-8"
    )
    scene_list = tmp_path / "scenes.txt"
    scene_list.write_text(scene + "\n", encoding="utf-8")

    np.save(
        gt_root / f"{scene}_bbox.npy",
        np.asarray(
            [
                [0.0, 0.0, 0.0, 2.0, 2.0, 2.0, 3.0],
                [5.0, 0.0, 0.0, 2.0, 2.0, 2.0, 4.0],
            ],
            dtype=np.float32,
        ),
    )
    _write_predictions(
        pred_root / f"{scene}_boxes.pkl",
        [
            (0, _corners([0.0, 0.0, 0.0], [2.8, 2.8, 2.8]), 0.90),
            (0, _corners([5.0, 0.0, 0.0], [2.0, 2.0, 2.0]), 0.80),
        ],
    )
    diagnostic_path = diagnostics_root / f"{scene}_tracks.npz"
    np.savez_compressed(
        diagnostic_path,
        scene_id=np.asarray(scene),
        c4_diagnostics_schema=np.asarray(C4_DIAGNOSTIC_SCHEMA),
        c4_enabled=np.asarray(True, dtype=np.bool_),
        c4_mutation_enabled=np.asarray(False, dtype=np.bool_),
        c4_fail_open=np.asarray(False, dtype=np.bool_),
        c4_error=np.asarray(""),
        c4_result_indices=np.asarray([0, 1], dtype=np.int64),
        c4_stable_ids=np.asarray([10, 11], dtype=np.int64),
        c4_scores=np.asarray([0.90, 0.80], dtype=np.float64),
        c4_attempted=np.asarray([True, True], dtype=np.bool_),
        c4_proposed=np.asarray([True, True], dtype=np.bool_),
        c4_verified=np.asarray([True, False], dtype=np.bool_),
        c4_applied=np.asarray([False, False], dtype=np.bool_),
        c4_reason=np.asarray(["verified_observer", "projection_drop"]),
        c4_source=np.asarray(["global", "global"]),
        c4_label=np.asarray(["cabinet", "chair"]),
        c4_normalized_label=np.asarray(["cabinet", "chair"]),
        c4_original_boxes=np.asarray(
            [
                [0.0, 0.0, 0.0, 2.8, 2.8, 2.8],
                [5.0, 0.0, 0.0, 2.0, 2.0, 2.0],
            ],
            dtype=np.float32,
        ),
        c4_candidate_boxes=np.asarray(
            [
                # Deliberately differs from c4_candidate_corners.  The v2
                # reporter must preserve yaw and use corners for evaluation;
                # the six-dimensional row is diagnostics-only.
                [0.0, 0.0, 0.0, 4.0, 4.0, 4.0],
                [5.0, 0.0, 0.0, 4.0, 4.0, 4.0],
            ],
            dtype=np.float32,
        ),
        c4_original_corners=np.stack(
            [
                _corners([0.0, 0.0, 0.0], [2.8, 2.8, 2.8]),
                _corners([5.0, 0.0, 0.0], [2.0, 2.0, 2.0]),
            ]
        ).astype(np.float32),
        c4_candidate_corners=np.stack(
            [
                _corners([0.0, 0.0, 0.0], [2.0, 2.0, 2.0]),
                _corners([5.0, 0.0, 0.0], [4.0, 4.0, 4.0]),
            ]
        ).astype(np.float32),
    )
    return {
        "scene": scene,
        "pred_root": pred_root,
        "diagnostics_root": diagnostics_root,
        "diagnostic_path": diagnostic_path,
        "gt_root": gt_root,
        "scan_root": scan_root,
        "scene_list": scene_list,
    }


def test_report_simulates_verified_replacement_with_frozen_scores(tmp_path):
    paths = _synthetic_c4_observer(tmp_path)
    report = build_report(
        pred_root=paths["pred_root"],
        diagnostics_root=paths["diagnostics_root"],
        scene_list=paths["scene_list"],
        gt_root=paths["gt_root"],
        scan_root=paths["scan_root"],
    )

    assert report["schema"] == REPORT_SCHEMA
    assert report["diagnostic_schema"] == C4_DIAGNOSTIC_SCHEMA
    assert report["scene_count"] == 1
    assert report["flow"]["attempted"] == 2
    assert report["flow"]["proposed"] == 2
    assert report["flow"]["verified"] == 1
    assert report["flow"]["applied"] == 0
    assert report["flow"]["rejection_reasons"] == {"projection_drop": 1}
    preservation = dict(report["score_preservation"])
    maximum_corner_delta = preservation.pop("maximum_absolute_corner_delta")
    assert preservation == {
        "rows_checked": 2,
        "scores_equal_with_atol_1e-7": True,
        "maximum_absolute_score_delta": 0.0,
        "corner_rows_checked": 2,
        "corners_equal_pointwise_with_atol_1e-6": True,
        "corners_equal_by_aabb_iou_atol_1e-6": True,
        "prediction_count_unchanged": True,
        "prediction_order_unchanged": True,
    }
    assert maximum_corner_delta == pytest.approx(0.0, abs=3e-8)
    assert (
        report["geometry"]["axis_alignment_source"]
        == "diagnostic_oriented_corners"
    )
    assert report["geometry"]["six_dimensional_boxes_role"] == "diagnostics_only"

    ap50 = report["thresholds"]["0.50"]
    assert ap50["observer"]["ap"] == pytest.approx(0.25)
    assert ap50["verified_replacement"]["ap"] == 1.0
    assert ap50["delta"]["ap"] == pytest.approx(0.75)
    assert report["geometry"]["improved"] == 1
    assert report["geometry"]["harmed"] == 1
    assert report["geometry"]["threshold_crossings"]["0.50"] == {
        "up": 1,
        "down": 1,
        "above": 0,
        "below": 0,
    }
    assert report["classes"]["cabinet"]["improved"] == 1
    assert report["classes"]["chair"]["harmed"] == 1

    first = report["candidate_rows"][0]
    assert first["verified"] is True
    assert first["original_best_gt"]["iou"] == pytest.approx(
        8.0 / (2.8**3)
    )
    assert first["candidate_best_gt"] == {"index": 0, "iou": 1.0}
    assert first["candidate_box_world"][3:] == [4.0, 4.0, 4.0]
    assert np.asarray(first["candidate_corners_world"]).shape == (8, 3)
    assert first["threshold_crossing"]["0.50"] == "up"
    assert first["original_export_box_iou"] == pytest.approx(1.0)

    # The returned report is strict JSON (no NaN/Infinity extensions).
    json.dumps(report, allow_nan=False)


def test_loader_fails_fast_on_schema_or_mutation(tmp_path):
    paths = _synthetic_c4_observer(tmp_path)
    diagnostic_path = Path(paths["diagnostic_path"])
    with np.load(diagnostic_path, allow_pickle=False) as archive:
        values = {name: np.array(archive[name], copy=True) for name in archive}

    values["c4_diagnostics_schema"] = np.asarray("old_schema")
    np.savez_compressed(diagnostic_path, **values)
    with pytest.raises(ValueError, match="unsupported C4 schema"):
        load_c4_diagnostics(
            diagnostic_path, expected_scene_id=paths["scene"]
        )

    values["c4_diagnostics_schema"] = np.asarray(C4_DIAGNOSTIC_SCHEMA)
    values["c4_mutation_enabled"] = np.asarray(True, dtype=np.bool_)
    np.savez_compressed(diagnostic_path, **values)
    with pytest.raises(ValueError, match="mutation must be disabled"):
        load_c4_diagnostics(
            diagnostic_path, expected_scene_id=paths["scene"]
        )


def test_report_rejects_original_corner_export_disagreement(tmp_path):
    paths = _synthetic_c4_observer(tmp_path)
    diagnostic_path = Path(paths["diagnostic_path"])
    with np.load(diagnostic_path, allow_pickle=False) as archive:
        values = {name: np.array(archive[name], copy=True) for name in archive}
    values["c4_original_corners"][0, 0, 0] += 0.01
    np.savez_compressed(diagnostic_path, **values)

    with pytest.raises(ValueError, match="disagree point-wise"):
        build_report(
            pred_root=paths["pred_root"],
            diagnostics_root=paths["diagnostics_root"],
            scene_list=paths["scene_list"],
            gt_root=paths["gt_root"],
            scan_root=paths["scan_root"],
        )


def test_loader_rejects_missing_or_malformed_oriented_corners(tmp_path):
    paths = _synthetic_c4_observer(tmp_path)
    diagnostic_path = Path(paths["diagnostic_path"])
    with np.load(diagnostic_path, allow_pickle=False) as archive:
        values = {name: np.array(archive[name], copy=True) for name in archive}

    missing = dict(values)
    del missing["c4_candidate_corners"]
    np.savez_compressed(diagnostic_path, **missing)
    with pytest.raises(ValueError, match="c4_candidate_corners"):
        load_c4_diagnostics(
            diagnostic_path, expected_scene_id=paths["scene"]
        )

    values["c4_candidate_corners"] = np.zeros((2, 6), dtype=np.float32)
    np.savez_compressed(diagnostic_path, **values)
    with pytest.raises(ValueError, match=r"shape \[2,8,3\]"):
        load_c4_diagnostics(
            diagnostic_path, expected_scene_id=paths["scene"]
        )


def test_heldout_exclusion_fails_when_no_scene_remains(tmp_path):
    paths = _synthetic_c4_observer(tmp_path)
    with pytest.raises(ValueError, match="No held-out scenes"):
        build_report(
            pred_root=paths["pred_root"],
            diagnostics_root=paths["diagnostics_root"],
            scene_list=paths["scene_list"],
            exclude_scene_list=paths["scene_list"],
            gt_root=paths["gt_root"],
            scan_root=paths["scan_root"],
        )


def test_offline_report_is_not_referenced_by_runtime():
    runtime_source = (
        Path(__file__).resolve().parents[1]
        / "boxfusion"
        / "online_refinement.py"
    ).read_text(encoding="utf-8")
    assert "report_c4_geometry_ablation" not in runtime_source
