from __future__ import annotations

import csv
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import run_scannet_target_first_mobilesam_masklift_full100 as module


class _Memory:
    @staticmethod
    def voxel_downsample(points, _voxel_size):
        return np.asarray(points, dtype=np.float32)

    @staticmethod
    def deterministic_bounded_sample(points, cap):
        return np.asarray(points, dtype=np.float32)[:cap]


def _write_csv(path: Path, columns: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def test_neighbor_consensus_uses_tested_core() -> None:
    points = [
        np.asarray([[0.001, 0.001, 0.001]], dtype=np.float32),
        np.asarray([[0.051, 0.001, 0.001]], dtype=np.float32),
        np.asarray([[2.0, 2.0, 2.0]], dtype=np.float32),
    ]
    result = module._fuse_three_view_points(points, object_memory=_Memory())
    assert result["supported_voxel_count"] == 2
    assert result["per_view_supported_voxels"] == [1, 1, 0]
    assert result["per_view_neighborhood_supported_voxels"] == [2, 2, 0]
    assert result["points"].shape == (2, 3)


def test_pi_periodic_yaw_mean_handles_equivalent_box_axes() -> None:
    def quaternion(yaw: float) -> np.ndarray:
        return np.asarray([np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)])

    yaw = module._circular_pi_yaw(
        np.stack([quaternion(0.01), quaternion(np.pi - 0.01), quaternion(0.0)])
    )
    assert abs(yaw) < 0.02


def test_plan_only_is_target_first_normalized_exact_and_top4(tmp_path: Path) -> None:
    scene = "scene0000_00"
    scene_list = tmp_path / "scenes.txt"
    scene_list.write_text(scene + "\n", encoding="utf-8")
    receipt = tmp_path / "receipt.json"
    receipt.write_text(
        json.dumps(
            {
                "schema": module.EXPECTED_RECEIPT_SCHEMA,
                "gt_access": False,
                "evaluator_access": False,
                "scenes": {scene: {"world_offset_xyz": [1.0, 2.0, 3.0]}},
            }
        ),
        encoding="utf-8",
    )
    schedule_root = tmp_path / "schedule"
    schedule_path = schedule_root / scene / "manifest.json"
    schedule_path.parent.mkdir(parents=True)
    schedule_path.write_text(json.dumps({"recorded_frame_ids": [0]}), encoding="utf-8")
    frames = tmp_path / "rgbd" / scene / "frames"
    (frames / "pose").mkdir(parents=True)
    pose = np.eye(4)
    pose[:3, 3] = [1.0, 2.0, 3.0]
    np.savetxt(frames / "pose/0.txt", pose)
    (frames / "intrinsic").mkdir()
    np.savetxt(frames / "intrinsic/intrinsic_depth.txt", np.eye(4))

    raw_columns = [
        "time_ns",
        "tx_world_object",
        "ty_world_object",
        "tz_world_object",
        "qw_world_object",
        "qx_world_object",
        "qy_world_object",
        "qz_world_object",
        "scale_x",
        "scale_y",
        "scale_z",
        "name",
        "instance",
        "sem_id",
        "prob",
    ]
    owl_columns = [
        "time_ns",
        "frame_id",
        "sensor",
        "device",
        "img_width",
        "img_height",
        "x1",
        "y1",
        "x2",
        "y2",
        "name",
        "instance",
        "sem_id",
        "prob",
    ]
    names = ["Chair", "chair", "chair", "chair", "chair", "wall"]
    scores = [0.6, 0.9, 0.8, 0.7, 0.5, 0.99]
    raw_rows = []
    owl_rows = []
    for instance, (name, score) in enumerate(zip(names, scores)):
        raw_rows.append(
            {
                "time_ns": 0,
                "tx_world_object": 0,
                "ty_world_object": 0,
                "tz_world_object": 0,
                "qw_world_object": 1,
                "qx_world_object": 0,
                "qy_world_object": 0,
                "qz_world_object": 0,
                "scale_x": 1,
                "scale_y": 1,
                "scale_z": 1,
                "name": name.lower(),
                "instance": instance,
                "sem_id": 10 + instance,
                "prob": score,
            }
        )
        owl_rows.append(
            {
                "time_ns": 0,
                "frame_id": 0,
                "sensor": "color",
                "device": "x",
                "img_width": 960,
                "img_height": 960,
                "x1": 100,
                "y1": 100,
                "x2": 300,
                "y2": 400,
                "name": name,
                "instance": instance,
                "sem_id": 10 + instance,
                "prob": score,
            }
        )
    raw_root = tmp_path / "raw"
    _write_csv(raw_root / "boxer_raw" / scene / "boxer_3dbbs.csv", raw_columns, raw_rows)
    _write_csv(raw_root / "boxer_raw" / scene / "owl_2dbbs.csv", owl_columns, owl_rows)

    plan = module.run_shadow(
        receipt_manifest_path=receipt,
        raw_log_root=raw_root,
        schedule_root=schedule_root,
        scene_root=tmp_path / "rgbd",
        scene_list_path=scene_list,
        baseline_root=tmp_path / "unused-baseline",
        checkpoint=tmp_path / "unused-checkpoint",
        output_root=tmp_path / "unused-output",
        device="cuda:0",
        expected_scene_count=1,
        plan_only=True,
    )
    assert plan["target_prompt_count"] == 4
    assert plan["target_prompt_frame_count"] == 1
    assert plan["selection_source"] == "all_raw_rows_not_old_receipt_membership"
