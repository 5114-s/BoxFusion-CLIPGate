import hashlib
import json
import pickle
from pathlib import Path

import numpy as np
import pytest

from tools.report_trifusion_oracles import (
    CORNER_FRAME,
    GEOMETRY_CANDIDATE_SCHEMA,
    REPORT_SCHEMA,
    SUPPLEMENTAL_CANDIDATE_SCHEMA,
    _validate_output_path,
    build_report,
    load_geometry_candidates,
    load_supplemental_candidates,
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


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _synthetic_oracle_inputs(tmp_path):
    scene = "scene0000_00"
    pred_root = tmp_path / "predictions"
    geometry_root = tmp_path / "geometry"
    supplemental_root = tmp_path / "supplemental"
    gt_root = tmp_path / "gt"
    scan_root = tmp_path / "scans"
    for root in (
        pred_root,
        geometry_root,
        supplemental_root,
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
        f"axisAlignment = {transform_text}\n", encoding="utf-8"
    )
    scene_list = tmp_path / "scenes.txt"
    scene_list.write_text(scene + "\n", encoding="utf-8")

    np.save(
        gt_root / f"{scene}_bbox.npy",
        np.asarray(
            [
                [0.0, 0.0, 0.0, 2.0, 2.0, 2.0, 1.0],
                [5.0, 0.0, 0.0, 2.0, 2.0, 2.0, 1.0],
                [10.0, 0.0, 0.0, 2.0, 2.0, 2.0, 1.0],
            ],
            dtype=np.float32,
        ),
    )
    prediction_path = pred_root / f"{scene}_boxes.pkl"
    prediction_corners = [
        _corners([0.0, 0.0, 0.0], [4.0, 4.0, 4.0]),
        _corners([5.0, 0.0, 0.0], [2.0, 2.0, 2.0]),
        _corners([30.0, 0.0, 0.0], [2.0, 2.0, 2.0]),
    ]
    _write_predictions(
        prediction_path,
        [
            (0, prediction_corners[0], 0.90),
            (0, prediction_corners[1], 0.10),
            (0, prediction_corners[2], 0.80),
        ],
    )

    geometry_path = (
        geometry_root / f"{scene}_geometry_candidates.npz"
    )
    np.savez_compressed(
        geometry_path,
        schema=np.asarray(GEOMETRY_CANDIDATE_SCHEMA),
        scene_id=np.asarray(scene),
        corner_frame=np.asarray(CORNER_FRAME),
        prediction_indices=np.asarray([0], dtype=np.int64),
        original_corners=np.asarray(
            [prediction_corners[0]], dtype=np.float64
        ),
        candidate_offsets=np.asarray([0, 2], dtype=np.int64),
        candidate_corners=np.asarray(
            [
                # Verified geometry clears .15/.25, but not .50.
                _corners([0.0, 0.0, 0.0], [3.0, 3.0, 3.0]),
                # The all-candidate oracle must inspect the second choice.
                _corners([0.0, 0.0, 0.0], [2.0, 2.0, 2.0]),
            ],
            dtype=np.float64,
        ),
        candidate_ids=np.asarray(["msr-0", "occupancy-0"]),
        candidate_sources=np.asarray(["msr", "occupancy_msr"]),
        candidate_valid=np.asarray([True, True], dtype=np.bool_),
        candidate_verified=np.asarray([True, False], dtype=np.bool_),
    )

    supplemental_path = (
        supplemental_root / f"{scene}_supplemental_candidates.json"
    )
    supplemental_path.write_text(
        json.dumps(
            {
                "schema": SUPPLEMENTAL_CANDIDATE_SCHEMA,
                "scene_id": scene,
                "corner_frame": CORNER_FRAME,
                "candidates": [
                    {
                        "candidate_id": "graph-10",
                        "source": "missing_graph",
                        "corners": _corners(
                            [10.0, 0.0, 0.0], [2.0, 2.0, 2.0]
                        ).tolist(),
                        "score": 0.01,
                        "label": "chair",
                    },
                    {
                        "candidate_id": "graph-10-duplicate",
                        "source": "missing_graph",
                        "corners": _corners(
                            [10.0, 0.0, 0.0], [2.0, 2.0, 2.0]
                        ).tolist(),
                        "score": 0.99,
                        "label": "table",
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    return {
        "scene": scene,
        "pred_root": pred_root,
        "prediction_path": prediction_path,
        "geometry_root": geometry_root,
        "geometry_path": geometry_path,
        "supplemental_root": supplemental_root,
        "supplemental_path": supplemental_path,
        "gt_root": gt_root,
        "scan_root": scan_root,
        "scene_list": scene_list,
    }


def test_oracles_report_geometry_union_ordering_targets_and_immutability(
    tmp_path,
):
    paths = _synthetic_oracle_inputs(tmp_path)
    before_digest = _sha256(paths["prediction_path"])
    before_stat = paths["prediction_path"].stat()

    report = build_report(
        pred_root=paths["pred_root"],
        scene_list=paths["scene_list"],
        gt_root=paths["gt_root"],
        scan_root=paths["scan_root"],
        geometry_candidates_root=paths["geometry_root"],
        supplemental_candidates_root=paths["supplemental_root"],
    )

    assert report["schema"] == REPORT_SCHEMA
    assert report["paper_plus_10_targets_ap_percent"] == {
        "0.15": 47.46,
        "0.25": 41.36,
        "0.50": 23.41,
    }
    ap50 = report["thresholds"]["0.50"]
    assert ap50["baseline"]["ap"] == pytest.approx(1.0 / 9.0)
    assert ap50["baseline"]["recall"] == pytest.approx(1.0 / 3.0)
    assert ap50["baseline"]["precision"] == pytest.approx(1.0 / 3.0)
    assert ap50["best_box_oracle"]["ap"] == pytest.approx(2.0 / 3.0)
    assert (
        ap50["best_box_verified_only_oracle"]["ap"]
        == pytest.approx(1.0 / 3.0)
    )

    # Two supplemental boxes cover the same GT. One-to-one matching may use
    # only one of them, so recall reaches three GT rather than four matches.
    union = ap50["proposal_union_oracle"]
    assert union["predictions"] == 5
    assert union["maximum_matches"] == 3
    assert union["true_positives"] == 3
    assert union["recall"] == 1.0
    assert union["ap"] == 1.0
    assert union["precision"] == pytest.approx(3.0 / 5.0)
    assert (
        union["score_ordering"]
        == "matched_iou_descending_then_stable_scene_prediction_order"
    )
    assert sum(union["selection_counts"].values()) == 3
    assert union["selection_counts"]["geometry:occupancy_msr"] == 1
    assert union["selection_counts"]["original"] == 1
    assert union["selection_counts"]["supplemental:missing_graph"] == 1

    # At .25 the verified 3 m candidate is sufficient (IoU 8/27).
    assert (
        report["thresholds"]["0.25"][
            "best_box_verified_only_oracle"
        ]["maximum_matches"]
        == 2
    )
    target_gap = (
        23.41 - report["thresholds"]["0.50"]["baseline"]["ap_percent"]
    )
    assert report["ceilings_and_gaps"]["0.50"]["baseline"][
        "gap_to_target_percentage_points"
    ] == pytest.approx(target_gap)
    assert report["candidate_inventory"]["supplemental"][
        "score_rows_ignored_by_oracle"
    ] == 2
    assert report["candidate_inventory"]["supplemental"][
        "label_rows_ignored_by_class_agnostic_evaluation"
    ] == 2
    assert report["protocol"]["supplemental_scores_used"] is False
    assert report["protocol"]["supplemental_labels_used"] is False
    assert "GT-conditioned" in report["oracle_disclaimer"]
    json.dumps(report, allow_nan=False)

    repeated = build_report(
        pred_root=paths["pred_root"],
        scene_list=paths["scene_list"],
        gt_root=paths["gt_root"],
        scan_root=paths["scan_root"],
        geometry_candidates_root=paths["geometry_root"],
        supplemental_candidates_root=paths["supplemental_root"],
    )
    assert repeated == report
    assert _sha256(paths["prediction_path"]) == before_digest
    after_stat = paths["prediction_path"].stat()
    assert after_stat.st_mtime_ns == before_stat.st_mtime_ns
    assert after_stat.st_size == before_stat.st_size


def test_geometry_loader_rejects_schema_offsets_and_verified_invalid(
    tmp_path,
):
    paths = _synthetic_oracle_inputs(tmp_path)
    geometry_path = Path(paths["geometry_path"])
    with np.load(geometry_path, allow_pickle=False) as payload:
        values = {
            name: np.array(payload[name], copy=True) for name in payload
        }

    values["schema"] = np.asarray("wrong.geometry.schema")
    np.savez_compressed(geometry_path, **values)
    with pytest.raises(ValueError, match="unsupported schema"):
        load_geometry_candidates(
            geometry_path, expected_scene_id=paths["scene"]
        )

    values["schema"] = np.asarray(GEOMETRY_CANDIDATE_SCHEMA)
    values["candidate_offsets"] = np.asarray([0, 1], dtype=np.int64)
    np.savez_compressed(geometry_path, **values)
    with pytest.raises(ValueError, match="candidate_offsets"):
        load_geometry_candidates(
            geometry_path, expected_scene_id=paths["scene"]
        )

    values["candidate_offsets"] = np.asarray([0, 2], dtype=np.int64)
    values["candidate_valid"] = np.asarray(
        [True, False], dtype=np.bool_
    )
    values["candidate_verified"] = np.asarray(
        [True, True], dtype=np.bool_
    )
    np.savez_compressed(geometry_path, **values)
    with pytest.raises(ValueError, match="must also be valid"):
        load_geometry_candidates(
            geometry_path, expected_scene_id=paths["scene"]
        )


def test_report_rejects_scene_corner_and_corner_frame_mismatches(tmp_path):
    paths = _synthetic_oracle_inputs(tmp_path)
    supplemental_path = Path(paths["supplemental_path"])
    supplemental = json.loads(
        supplemental_path.read_text(encoding="utf-8")
    )
    supplemental["scene_id"] = "scene9999_00"
    supplemental_path.write_text(
        json.dumps(supplemental), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="does not match"):
        load_supplemental_candidates(
            supplemental_path, expected_scene_id=paths["scene"]
        )

    supplemental["scene_id"] = paths["scene"]
    supplemental["corner_frame"] = "axis_aligned_scannet"
    supplemental_path.write_text(
        json.dumps(supplemental), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="corner_frame"):
        load_supplemental_candidates(
            supplemental_path, expected_scene_id=paths["scene"]
        )

    # Restore the supplemental file and break the geometry/export integrity
    # check while retaining valid box corners.
    supplemental["corner_frame"] = CORNER_FRAME
    supplemental_path.write_text(
        json.dumps(supplemental), encoding="utf-8"
    )
    geometry_path = Path(paths["geometry_path"])
    with np.load(geometry_path, allow_pickle=False) as payload:
        values = {
            name: np.array(payload[name], copy=True) for name in payload
        }
    values["original_corners"][0] += np.asarray([0.01, 0.0, 0.0])
    np.savez_compressed(geometry_path, **values)
    with pytest.raises(ValueError, match="disagree point-wise"):
        build_report(
            pred_root=paths["pred_root"],
            scene_list=paths["scene_list"],
            gt_root=paths["gt_root"],
            scan_root=paths["scan_root"],
            geometry_candidates_root=paths["geometry_root"],
            supplemental_candidates_root=paths["supplemental_root"],
        )


def test_optional_flags_default_true_and_output_cannot_touch_inputs(
    tmp_path,
):
    scene = "scene0001_00"
    supplemental_path = tmp_path / "supplemental.json"
    supplemental_path.write_text(
        json.dumps(
            {
                "schema": SUPPLEMENTAL_CANDIDATE_SCHEMA,
                "scene_id": scene,
                "candidates": [
                    {
                        "candidate_id": 7,
                        "source": "graph",
                        "corners": _corners(
                            [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]
                        ).tolist(),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    loaded = load_supplemental_candidates(
        supplemental_path, expected_scene_id=scene
    )
    assert loaded.candidate_ids == ("7",)
    assert loaded.candidate_valid.tolist() == [True]
    assert loaded.candidate_verified.tolist() == [True]
    assert loaded.candidate_labels is None
    assert loaded.candidate_scores is None

    supplemental_npz = tmp_path / "supplemental.npz"
    np.savez_compressed(
        supplemental_npz,
        schema=np.asarray(SUPPLEMENTAL_CANDIDATE_SCHEMA),
        scene_id=np.asarray(scene),
        candidate_corners=np.asarray(
            [
                _corners(
                    [1.0, 0.0, 0.0], [1.0, 1.0, 1.0]
                )
            ],
            dtype=np.float64,
        ),
        candidate_ids=np.asarray(["npz-8"]),
        candidate_sources=np.asarray(["graph"]),
        candidate_scores=np.asarray([0.25], dtype=np.float64),
        candidate_labels=np.asarray(["cabinet"]),
    )
    loaded_npz = load_supplemental_candidates(
        supplemental_npz, expected_scene_id=scene
    )
    assert loaded_npz.candidate_valid.tolist() == [True]
    assert loaded_npz.candidate_verified.tolist() == [True]
    assert loaded_npz.candidate_scores.tolist() == [0.25]
    assert loaded_npz.candidate_labels == ("cabinet",)

    pred_root = tmp_path / "predictions"
    pred_root.mkdir()
    with pytest.raises(ValueError, match="must not be inside"):
        _validate_output_path(
            pred_root / "scene0001_00_boxes.pkl",
            pred_root=pred_root,
            geometry_root=None,
            supplemental_root=None,
        )


def test_offline_oracle_is_not_referenced_by_runtime():
    runtime_source = (
        Path(__file__).resolve().parents[1]
        / "boxfusion"
        / "online_refinement.py"
    ).read_text(encoding="utf-8")
    assert "report_trifusion_oracles" not in runtime_source
