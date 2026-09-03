import json
import pickle
from pathlib import Path

import numpy as np
import pytest

from tools.report_c2_geometry_ablation import (
    REPORT_SCHEMA,
    build_report,
    load_c2_diagnostics,
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


def _synthetic_c2_ablation(tmp_path):
    scene = "scene0000_00"
    baseline_root = tmp_path / "baseline"
    c2_root = tmp_path / "c2"
    diagnostics_root = tmp_path / "diagnostics"
    gt_root = tmp_path / "gt"
    scan_root = tmp_path / "scans"
    for root in (
        baseline_root,
        c2_root,
        diagnostics_root,
        gt_root,
        scan_root,
    ):
        root.mkdir()

    scene_scan_root = scan_root / scene
    scene_scan_root.mkdir()
    transform_text = " ".join(
        str(float(value)) for value in np.eye(4).reshape(-1)
    )
    (scene_scan_root / f"{scene}.txt").write_text(
        f"axisAlignment = {transform_text}\n",
        encoding="utf-8",
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
    baseline = [
        (0, _corners([0.0, 0.0, 0.0], [2.8, 2.8, 2.8]), 0.90),
        (0, _corners([5.0, 0.0, 0.0], [2.0, 2.0, 2.0]), 0.80),
    ]
    c2 = [
        (0, _corners([0.0, 0.0, 0.0], [2.0, 2.0, 2.0]), 0.90),
        (0, _corners([5.0, 0.0, 0.0], [2.0, 2.0, 2.0]), 0.80),
    ]
    _write_predictions(
        baseline_root / f"{scene}_boxes.pkl",
        baseline,
    )
    _write_predictions(c2_root / f"{scene}_boxes.pkl", c2)

    summary = {
        "c2_attempted": 2,
        "c2_proposed": 2,
        "c2_verified": 1,
        "c2_applied": 1,
        "c2_seconds": 0.01,
        "c2_rejections": {"reprojection_delta": 1},
    }
    diagnostic_path = diagnostics_root / f"{scene}_tracks.npz"
    np.savez_compressed(
        diagnostic_path,
        scene_id=np.asarray(scene),
        result_indices=np.asarray([0, 1], dtype=np.int64),
        track_ids=np.asarray([-1, -2], dtype=np.int64),
        labels=np.asarray(["cabinet", "chair"]),
        c2_attempted=np.asarray([True, True], dtype=np.bool_),
        c2_proposed=np.asarray([True, True], dtype=np.bool_),
        c2_verified=np.asarray([True, False], dtype=np.bool_),
        c2_applied=np.asarray([True, False], dtype=np.bool_),
        c2_reason=np.asarray(["accepted", "reprojection_delta"]),
        c2_branch=np.asarray(["solid", "solid"]),
        c2_original_boxes=np.asarray(
            [
                [0.0, 0.0, 0.0, 2.8, 2.8, 2.8],
                [5.0, 0.0, 0.0, 2.0, 2.0, 2.0],
            ],
            dtype=np.float32,
        ),
        c2_candidate_boxes=np.asarray(
            [
                [0.0, 0.0, 0.0, 2.0, 2.0, 2.0],
                [5.0, 0.0, 0.0, 2.2, 2.2, 2.2],
            ],
            dtype=np.float32,
        ),
        summary_json=np.asarray(json.dumps(summary)),
    )
    return {
        "scene": scene,
        "baseline_root": baseline_root,
        "c2_root": c2_root,
        "diagnostics_root": diagnostics_root,
        "diagnostic_path": diagnostic_path,
        "gt_root": gt_root,
        "scan_root": scan_root,
        "scene_list": scene_list,
    }


def test_report_pairs_real_score_ap_flow_and_applied_iou(tmp_path):
    paths = _synthetic_c2_ablation(tmp_path)
    report = build_report(
        baseline_pred_root=paths["baseline_root"],
        c2_pred_root=paths["c2_root"],
        c2_diagnostics_root=paths["diagnostics_root"],
        scene_list=paths["scene_list"],
        gt_root=paths["gt_root"],
        scan_root=paths["scan_root"],
    )

    assert report["schema"] == REPORT_SCHEMA
    assert report["scene_count"] == 1
    assert report["pairing"] == {
        "detection_counts_equal": True,
        "scores_equal_by_position": True,
        "maximum_absolute_score_delta": 0.0,
    }
    assert report["thresholds"]["0.15"]["baseline"]["ap"] == 1.0
    assert report["thresholds"]["0.15"]["c2"]["ap"] == 1.0
    assert report["thresholds"]["0.25"]["delta"]["ap"] == 0.0
    assert report["thresholds"]["0.50"]["baseline"]["ap"] == pytest.approx(
        0.25
    )
    assert report["thresholds"]["0.50"]["c2"]["ap"] == 1.0
    assert report["thresholds"]["0.50"]["delta"]["ap"] == pytest.approx(
        0.75
    )

    assert report["c2_flow"]["attempted"] == 2
    assert report["c2_flow"]["proposed"] == 2
    assert report["c2_flow"]["verified"] == 1
    assert report["c2_flow"]["applied"] == 1
    assert report["c2_flow"]["rejections"] == {
        "reprojection_delta": 1
    }

    assert len(report["applied_rows"]) == 1
    row = report["applied_rows"][0]
    assert row["scene_id"] == paths["scene"]
    assert row["prediction_index"] == 0
    assert row["branch"] == "solid"
    assert row["original_best_gt"]["index"] == 0
    assert row["original_best_gt"]["iou"] == pytest.approx(
        8.0 / (2.8**3)
    )
    assert row["candidate_best_gt"] == {"index": 0, "iou": 1.0}
    assert row["same_original_best_gt"]["candidate_iou"] == 1.0
    assert row["threshold_crossing"] == {
        "0.15": "above",
        "0.25": "above",
        "0.50": "up",
    }
    assert row["baseline_export_box_iou"] == pytest.approx(1.0)
    assert row["c2_export_box_iou"] == pytest.approx(1.0)


def test_diagnostics_reject_applied_row_without_verification(tmp_path):
    paths = _synthetic_c2_ablation(tmp_path)
    diagnostic_path = Path(paths["diagnostic_path"])
    with np.load(diagnostic_path, allow_pickle=False) as payload:
        values = {name: np.array(payload[name], copy=True) for name in payload}
    values["c2_verified"][0] = False
    np.savez_compressed(diagnostic_path, **values)

    with pytest.raises(ValueError, match="applied rows must be verified"):
        load_c2_diagnostics(
            diagnostic_path,
            expected_scene_id=paths["scene"],
        )


def test_offline_report_is_not_referenced_by_runtime():
    runtime_source = (
        Path(__file__).resolve().parents[1]
        / "boxfusion"
        / "online_refinement.py"
    ).read_text(encoding="utf-8")
    assert "report_c2_geometry_ablation" not in runtime_source
