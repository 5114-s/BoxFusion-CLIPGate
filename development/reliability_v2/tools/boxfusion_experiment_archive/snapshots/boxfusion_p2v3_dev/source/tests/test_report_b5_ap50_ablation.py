import json
import pickle

import numpy as np
import pytest

from tools.report_b5_ap50_ablation import (
    build_paired_report,
    load_predictions,
    write_score_locked_predictions,
)


def _rotation_z(degrees):
    angle = np.deg2rad(degrees)
    return np.asarray(
        [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def _corners(center, dimensions, basis):
    signs = np.asarray(
        [
            [-1.0, -1.0, -1.0],
            [1.0, -1.0, -1.0],
            [1.0, 1.0, -1.0],
            [-1.0, 1.0, -1.0],
            [-1.0, -1.0, 1.0],
            [1.0, -1.0, 1.0],
            [1.0, 1.0, 1.0],
            [-1.0, 1.0, 1.0],
        ],
        dtype=np.float64,
    )
    return (
        np.asarray(center, dtype=np.float64)
        + (signs * (0.5 * np.asarray(dimensions))) @ basis.T
    ).astype(np.float32)


def _write_predictions(path, detections):
    with path.open("wb") as handle:
        pickle.dump([detections], handle, protocol=pickle.HIGHEST_PROTOCOL)


def _synthetic_ablation(tmp_path):
    scene = "scene0000_00"
    identity_root = tmp_path / "identity"
    candidate_root = tmp_path / "candidate"
    diagnostics_root = tmp_path / "diagnostics"
    scan_root = tmp_path / "scans"
    gt_root = tmp_path / "gt"
    for root in (
        identity_root,
        candidate_root,
        diagnostics_root,
        scan_root,
        gt_root,
    ):
        root.mkdir()
    scene_root = scan_root / scene
    scene_root.mkdir()
    identity_transform = np.eye(4)
    flattened = " ".join(
        str(value) for value in identity_transform.reshape(-1)
    )
    (scene_root / f"{scene}.txt").write_text(
        f"axisAlignment = {flattened}\n",
        encoding="utf-8",
    )
    scene_list = tmp_path / "scenes.txt"
    scene_list.write_text(scene + "\n", encoding="utf-8")

    basis = _rotation_z(17.0)
    centers = (
        np.asarray([0.0, 0.0, 0.0]),
        np.asarray([10.0, 0.0, 0.0]),
        np.asarray([20.0, 0.0, 0.0]),
    )
    gt = np.asarray(
        [
            [0.0, 0.0, 0.0, 2.0, 2.0, 2.0, 3.0],
            [10.0, 0.0, 0.0, 2.0, 2.0, 2.0, 3.0],
            [20.0, 0.0, 0.0, 2.0, 2.0, 2.0, 3.0],
        ],
        dtype=np.float32,
    )
    np.save(gt_root / f"{scene}_bbox.npy", gt)

    identity_detections = [
        (0, _corners(centers[0], [2.6, 2.6, 2.6], basis), 0.90),
        (0, _corners(centers[1], [2.0, 2.0, 2.0], basis), 0.80),
        (1, _corners(centers[2], [2.0, 2.0, 2.0], basis), 0.70),
    ]
    candidate_detections = [
        (0, _corners(centers[0], [2.0, 2.0, 2.0], basis), 0.89),
        (0, _corners(centers[1], [3.0, 3.0, 3.0], basis), 0.81),
        (1, _corners(centers[2], [2.0, 2.0, 2.0], basis), 0.70),
    ]
    _write_predictions(
        identity_root / f"{scene}_boxes.pkl", identity_detections
    )
    _write_predictions(
        candidate_root / f"{scene}_boxes.pkl", candidate_detections
    )

    summary = {
        "neural_refits_attempted": 4,
        "neural_refits_accepted": 2,
        "neural_refits_quality_rejected": 1,
        "neural_refits_gate_rejected": 1,
        "neural_refits_invalid_orientation": 0,
    }
    np.savez_compressed(
        diagnostics_root / f"{scene}_tracks.npz",
        scene_id=np.asarray(scene),
        result_indices=np.asarray([0, 1, 2], dtype=np.int64),
        refit_applied=np.asarray([True, True, False], dtype=np.bool_),
        summary_json=np.asarray(json.dumps(summary)),
    )
    return {
        "scene": scene,
        "identity_root": identity_root,
        "candidate_root": candidate_root,
        "diagnostics_root": diagnostics_root,
        "scan_root": scan_root,
        "gt_root": gt_root,
        "scene_list": scene_list,
    }


def test_paired_report_measures_final_refits_crossings_ap_and_yaw(tmp_path):
    paths = _synthetic_ablation(tmp_path)
    report = build_paired_report(
        identity_pred_root=paths["identity_root"],
        candidate_pred_root=paths["candidate_root"],
        candidate_diagnostics_root=paths["diagnostics_root"],
        scene_list=paths["scene_list"],
        scan_root=paths["scan_root"],
        gt_root=paths["gt_root"],
    )

    assert report["pairing"]["count_equal"] is True
    assert report["pairing"]["label_order_equal"] is True
    assert report["pairing"]["score_rank_order_equal"] is True
    assert report["pairing"]["position_pairing_contract_valid"] is True
    assert report["scores"]["exact_changed_count"] == 2
    assert report["scores"]["maximum_absolute_delta"] == pytest.approx(0.01)

    assert report["runtime"]["cumulative"] == {
        "attempted": 4,
        "accepted": 2,
        "quality_rejected": 1,
        "gate_rejected": 1,
        "invalid_orientation": 0,
    }
    assert report["runtime"]["final_applied"] == 2
    assert report["accepted_iou"]["improved"] == 1
    assert report["accepted_iou"]["worsened"] == 1
    assert report["accepted_iou"]["equal"] == 0
    assert report["crossings"]["0.50"]["up"] == 1
    assert report["crossings"]["0.50"]["down"] == 1
    assert report["crossings"]["0.50"]["up_scene_count"] == 1
    assert report["crossings"]["0.50"]["down_scene_count"] == 1
    assert report["yaw"]["invalid_box_count"] == 0
    assert report["yaw"]["changed_box_count"] == 0
    assert report["yaw"]["minimum_absolute_axis_cosine"] == pytest.approx(
        1.0, abs=1e-6
    )

    for method in (
        "identity",
        "candidate_native_scores",
        "candidate_identity_scores",
    ):
        assert set(report["metrics"][method]) == {"0.15", "0.25", "0.50"}
        assert 0.0 <= report["metrics"][method]["0.50"]["ap"] <= 1.0
    assert report["metrics"]["candidate_identity_scores"] is not None


def test_score_lock_writes_new_root_without_changing_inputs(tmp_path):
    paths = _synthetic_ablation(tmp_path)
    output_root = tmp_path / "score_locked"
    candidate_before = (
        paths["candidate_root"] / f"{paths['scene']}_boxes.pkl"
    ).read_bytes()

    result = write_score_locked_predictions(
        identity_pred_root=paths["identity_root"],
        candidate_pred_root=paths["candidate_root"],
        scene_list=paths["scene_list"],
        output_root=output_root,
    )

    assert result == output_root.resolve()
    assert (
        paths["candidate_root"] / f"{paths['scene']}_boxes.pkl"
    ).read_bytes() == candidate_before
    identity = load_predictions(
        paths["identity_root"] / f"{paths['scene']}_boxes.pkl"
    )
    candidate = load_predictions(
        paths["candidate_root"] / f"{paths['scene']}_boxes.pkl"
    )
    locked = load_predictions(output_root / f"{paths['scene']}_boxes.pkl")
    assert [item.score for item in locked.detections] == [
        item.score for item in identity.detections
    ]
    for locked_item, candidate_item in zip(
        locked.detections, candidate.detections
    ):
        np.testing.assert_array_equal(
            locked_item.corners, candidate_item.corners
        )

    with pytest.raises(FileExistsError):
        write_score_locked_predictions(
            identity_pred_root=paths["identity_root"],
            candidate_pred_root=paths["candidate_root"],
            scene_list=paths["scene_list"],
            output_root=output_root,
        )
    with pytest.raises(ValueError, match="outside both input roots"):
        write_score_locked_predictions(
            identity_pred_root=paths["identity_root"],
            candidate_pred_root=paths["candidate_root"],
            scene_list=paths["scene_list"],
            output_root=paths["candidate_root"],
        )


def test_score_lock_refuses_ambiguous_pairing(tmp_path):
    paths = _synthetic_ablation(tmp_path)
    candidate_path = (
        paths["candidate_root"] / f"{paths['scene']}_boxes.pkl"
    )
    candidate = load_predictions(candidate_path)
    shortened = [
        (
            item.raw_label,
            item.raw_corners,
            item.raw_score,
        )
        for item in candidate.detections[:-1]
    ]
    _write_predictions(candidate_path, shortened)

    report = build_paired_report(
        identity_pred_root=paths["identity_root"],
        candidate_pred_root=paths["candidate_root"],
        candidate_diagnostics_root=paths["diagnostics_root"],
        scene_list=paths["scene_list"],
        scan_root=paths["scan_root"],
        gt_root=paths["gt_root"],
    )
    assert report["pairing"]["count_equal"] is False
    assert report["pairing"]["position_pairing_contract_valid"] is False
    assert report["metrics"]["candidate_identity_scores"] is None

    with pytest.raises(ValueError, match="detection count differs"):
        write_score_locked_predictions(
            identity_pred_root=paths["identity_root"],
            candidate_pred_root=paths["candidate_root"],
            scene_list=paths["scene_list"],
            output_root=tmp_path / "must_not_exist",
        )
    assert not (tmp_path / "must_not_exist").exists()
