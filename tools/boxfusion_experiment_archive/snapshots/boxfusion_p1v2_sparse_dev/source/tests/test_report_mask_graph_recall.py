import json
import pickle

import numpy as np
import pytest

from tools.report_mask_graph_recall import build_report, main


def _axis_alignment():
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = np.asarray(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
    )
    transform[:3, 3] = np.asarray([4.0, -2.0, 1.0])
    return transform


def _world_box_from_aligned(center, dimensions, transform):
    aligned_center = np.asarray(center, dtype=np.float64)
    rotation = transform[:3, :3]
    world_center = rotation.T @ (aligned_center - transform[:3, 3])
    aligned_dimensions = np.asarray(dimensions, dtype=np.float64)
    world_dimensions = np.abs(rotation).T @ aligned_dimensions
    return np.concatenate((world_center, world_dimensions))


def _corners(box):
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
        dtype=np.float64,
    )
    value = np.asarray(box, dtype=np.float64)
    return value[:3] + signs * (0.5 * value[3:6])


def _synthetic_inputs(tmp_path):
    scene = "scene0000_00"
    diagnostics_root = tmp_path / "diagnostics"
    gt_root = tmp_path / "gt"
    scans_root = tmp_path / "scans"
    pred_root = tmp_path / "predictions"
    for root in (diagnostics_root, gt_root, scans_root, pred_root):
        root.mkdir()
    transform = _axis_alignment()
    scene_scan_root = scans_root / scene
    scene_scan_root.mkdir()
    flattened = " ".join(str(value) for value in transform.reshape(-1))
    (scene_scan_root / f"{scene}.txt").write_text(
        f"axisAlignment = {flattened}\n", encoding="utf-8"
    )
    scene_list = tmp_path / "scenes.txt"
    scene_list.write_text(scene + "\n", encoding="utf-8")

    aligned_boxes = np.asarray(
        [
            [0.0, 0.0, 0.0, 2.0, 1.0, 1.0],
            [5.0, 0.0, 0.0, 1.0, 2.0, 1.0],
            [10.0, 0.0, 0.0, 2.0, 3.0, 1.0],
        ],
        dtype=np.float64,
    )
    gt = np.concatenate(
        (aligned_boxes, np.ones((3, 1), dtype=np.float64)), axis=1
    )
    np.save(gt_root / f"{scene}_bbox.npy", gt.astype(np.float32))

    graph_boxes = np.stack(
        [
            _world_box_from_aligned(
                row[:3], row[3:6], transform
            )
            for row in aligned_boxes
        ]
        + [
            _world_box_from_aligned(
                [30.0, 0.0, 0.0], [1.0, 1.0, 1.0], transform
            )
        ]
    ).astype(np.float32)
    np.savez_compressed(
        diagnostics_root / f"{scene}_tracks.npz",
        scene_id=np.asarray(scene),
        graph_component_track_ids=np.asarray([0, 1, 2, 3]),
        graph_component_states=np.asarray(
            ["active", "expired", "archived", "active"]
        ),
        graph_component_boxes=graph_boxes,
        graph_component_track_confirmed=np.asarray(
            [True, True, True, False]
        ),
        graph_component_confirmed=np.asarray(
            [True, True, True, False]
        ),
    )

    baseline_box = _world_box_from_aligned(
        aligned_boxes[0, :3], aligned_boxes[0, 3:6], transform
    )
    detections = [(0, _corners(baseline_box), 0.9)]
    with (pred_root / f"{scene}_boxes.pkl").open("wb") as handle:
        pickle.dump([detections], handle, protocol=pickle.HIGHEST_PROTOCOL)
    return {
        "scene": scene,
        "diagnostics_root": diagnostics_root,
        "gt_root": gt_root,
        "scans_root": scans_root,
        "pred_root": pred_root,
        "scene_list": scene_list,
    }


def test_report_separates_graph_lifecycle_and_incremental_recall(tmp_path):
    paths = _synthetic_inputs(tmp_path)
    report = build_report(
        diagnostics_root=paths["diagnostics_root"],
        gt_root=paths["gt_root"],
        scans_root=paths["scans_root"],
        scene_list=paths["scene_list"],
        pred_root=paths["pred_root"],
    )

    assert report["scene_count"] == 1
    assert report["state_counts"] == {
        "active": 2,
        "archived": 1,
        "expired": 1,
    }
    assert report["graph"]["all"]["proposal_count"] == 4
    assert report["graph"]["confirmed"]["proposal_count"] == 3
    assert report["graph"]["confirmed_live"]["proposal_count"] == 2
    assert report["graph"]["confirmed_active"]["proposal_count"] == 1
    assert report["graph"]["confirmed_archived"]["proposal_count"] == 1
    for key in ("0.15", "0.25", "0.50"):
        assert (
            report["graph"]["all"]["thresholds"][key]["recall"]
            == pytest.approx(1.0)
        )
        assert (
            report["graph"]["confirmed"]["thresholds"][key]["recall"]
            == pytest.approx(1.0)
        )
        assert (
            report["graph"]["confirmed_live"]["thresholds"][key]["recall"]
            == pytest.approx(2.0 / 3.0)
        )
        assert (
            report["baseline"]["thresholds"][key]["recall"]
            == pytest.approx(1.0 / 3.0)
        )
        combined = report["baseline_plus_graph"]["confirmed_live"]
        assert combined["thresholds"][key]["recall"] == pytest.approx(
            2.0 / 3.0
        )
        assert combined["increment_vs_baseline"][key] == {
            "incremental_matched_ground_truth": 1,
            "recall_gain": pytest.approx(1.0 / 3.0),
        }


def test_report_rejects_embedded_scene_mismatch(tmp_path):
    paths = _synthetic_inputs(tmp_path)
    diagnostic = (
        paths["diagnostics_root"] / f"{paths['scene']}_tracks.npz"
    )
    with np.load(diagnostic, allow_pickle=False) as payload:
        values = {name: payload[name] for name in payload.files}
    values["scene_id"] = np.asarray("scene9999_99")
    np.savez_compressed(diagnostic, **values)

    with pytest.raises(ValueError, match="scene mismatch"):
        build_report(
            diagnostics_root=paths["diagnostics_root"],
            gt_root=paths["gt_root"],
            scans_root=paths["scans_root"],
            scene_list=paths["scene_list"],
        )


def test_report_rejects_misaligned_graph_shapes(tmp_path):
    paths = _synthetic_inputs(tmp_path)
    diagnostic = (
        paths["diagnostics_root"] / f"{paths['scene']}_tracks.npz"
    )
    with np.load(diagnostic, allow_pickle=False) as payload:
        values = {name: payload[name] for name in payload.files}
    values["graph_component_boxes"] = np.zeros((3, 6), dtype=np.float32)
    np.savez_compressed(diagnostic, **values)

    with pytest.raises(ValueError, match=r"shape \[4,6\]"):
        build_report(
            diagnostics_root=paths["diagnostics_root"],
            gt_root=paths["gt_root"],
            scans_root=paths["scans_root"],
            scene_list=paths["scene_list"],
        )


def test_cli_writes_json_without_baseline(tmp_path, capsys):
    paths = _synthetic_inputs(tmp_path)
    output = tmp_path / "reports" / "mask_graph.json"
    status = main(
        [
            "--diagnostics-root",
            str(paths["diagnostics_root"]),
            "--gt-root",
            str(paths["gt_root"]),
            "--scans-root",
            str(paths["scans_root"]),
            "--scene-list",
            str(paths["scene_list"]),
            "--output-json",
            str(output),
        ]
    )
    assert status == 0
    written = json.loads(output.read_text(encoding="utf-8"))
    printed = json.loads(capsys.readouterr().out)
    assert written == printed
    assert "baseline" not in written
    assert written["graph"]["confirmed_live"]["proposal_count"] == 2
