from __future__ import annotations

import csv
import hashlib
import json
import pickle
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from tools.audit_scannet_boxer_unexplained_oracle import (
    CSV_FIELDS,
    Candidate,
    OracleAuditError,
    gate_candidates_for_frame,
    main,
    official_constant_evaluate,
    signed_voxel_centroids,
    strict_maximum_matching,
    validate_shadow_seal,
)


SCENE = "scene0001_00"


def _corners(center, size):
    center = np.asarray(center, dtype=np.float64)
    half = np.asarray(size, dtype=np.float64) / 2.0
    signs = np.asarray(
        [
            [-1, -1, -1],
            [1, -1, -1],
            [1, 1, -1],
            [-1, 1, -1],
            [-1, -1, 1],
            [1, -1, 1],
            [1, 1, 1],
            [-1, 1, 1],
        ],
        dtype=np.float64,
    )
    return center + signs * half


def _candidate(center=(0.0, 0.0, 1.0), size=(0.8, 0.8, 0.2), row=0):
    center = np.asarray(center, dtype=np.float64)
    size = np.asarray(size, dtype=np.float64)
    corners = _corners(center, size)
    return Candidate(
        scene_id=SCENE,
        source="raw",
        row_index=row,
        frame_id=0,
        instance_id=row,
        name="object",
        probability=0.75,
        center_world=center,
        rotation_world_object=np.eye(3),
        size=size,
        corners_world=corners,
        aligned_minmax=np.concatenate((corners.min(0), corners.max(0))),
    )


def _write_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _csv_row(center=(0.0, 0.0, 1.0), size=(0.8, 0.8, 0.2)):
    return {
        "time_ns": "0",
        "tx_world_object": str(center[0]),
        "ty_world_object": str(center[1]),
        "tz_world_object": str(center[2]),
        "qw_world_object": "1",
        "qx_world_object": "0",
        "qy_world_object": "0",
        "qz_world_object": "0",
        "scale_x": str(size[0]),
        "scale_y": str(size[1]),
        "scale_z": str(size[2]),
        "name": "generic_object",
        "instance": "7",
        "sem_id": "99",
        "prob": "0.75",
    }


def test_signed_floor_voxelization_and_depth_gate():
    points = np.asarray(
        [[x, y, 1.0] for y in (-0.15, -0.05, 0.05, 0.15) for x in (-0.15, -0.05, 0.05, 0.15)],
        dtype=np.float64,
    )
    keys, _ = signed_voxel_centroids(
        np.asarray([[-0.001, 0.0, 0.0], [0.001, 0.0, 0.0]])
    )
    assert keys[:, 0].tolist() == [-1, 0]

    candidate = _candidate(size=(0.5, 0.5, 0.2))
    accepted, rows = gate_candidates_for_frame(
        [candidate], points, np.empty((0, 6), dtype=np.float64)
    )
    assert accepted == [candidate]
    assert rows[0].candidate_voxels == 16
    assert rows[0].unexplained_voxels == 16
    assert rows[0].unexplained_ratio == 1.0

    accepted, rows = gate_candidates_for_frame(
        [candidate], points, np.asarray([[-1.0, -1.0, 0.5, 1.0, 1.0, 1.5]])
    )
    assert accepted == []
    assert rows[0].reason == "insufficient_unexplained_voxels"
    assert rows[0].unexplained_voxels == 0


def test_strict_matching_and_official_greedy_do_not_accept_boundary():
    iou = np.asarray([[0.50]], dtype=np.float64)
    assert strict_maximum_matching(iou, 0.50) == []
    assert strict_maximum_matching(iou, 0.49) == [(0, 0)]

    boundary = official_constant_evaluate([iou], [1], 0.50)
    assert boundary["greedy_tp"] == 0
    assert boundary["ap"] == 0.0
    above = official_constant_evaluate([iou], [1], 0.49)
    assert above["greedy_tp"] == 1
    assert above["ap"] == pytest.approx(1.0 / 1.000001)


def test_maximum_matching_handles_candidate_collision():
    matrix = np.asarray(
        [
            [0.9, 0.8],
            [0.7, 0.0],
        ],
        dtype=np.float64,
    )
    pairs = strict_maximum_matching(matrix, 0.5)
    assert len(pairs) == 2
    assert {gt for _, gt in pairs} == {0, 1}


def _make_end_to_end_tree(root: Path):
    boxer_root = root / "boxer"
    schedule_root = root / "schedule"
    baseline_root = root / "baseline"
    gt_root = root / "gt"
    scan_root = root / "scans"
    rgbd_root = root / "rgbd"
    scene_list = root / "scenes.txt"
    scene_list.write_text(f"{SCENE}\n", encoding="utf-8")
    manifest_dir = schedule_root / SCENE
    manifest_dir.mkdir(parents=True)
    (manifest_dir / "manifest.json").write_text(
        json.dumps(
            {
                "schema": "boxfusion.cutr_postfilter_cache.v3",
                "namespace": "unit-test-sealed-v3",
                "scene_id": SCENE,
                "record_count": 1,
                "recorded_frame_ids": [0],
                "schedule": {
                    "dataset_length": 1,
                    "gap": 25,
                    "terminal_policy": "upstream_boxfusion_early_exit_v1",
                },
                "records": [{"frame_id": 0}],
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    scene_scan = scan_root / SCENE
    scene_scan.mkdir(parents=True)
    identity = " ".join(str(value) for value in np.eye(4).reshape(-1))
    (scene_scan / f"{SCENE}.txt").write_text(
        f"axisAlignment = {identity}\n", encoding="utf-8"
    )

    offset = np.asarray([10.0, 20.0, 30.0])
    candidate_center = offset + np.asarray([0.0, 0.0, 1.0])
    baseline_center = offset + np.asarray([2.0, 0.0, 1.0])
    size = np.asarray([0.8, 0.8, 0.2])
    gt_root.mkdir()
    np.save(
        gt_root / f"{SCENE}_bbox.npy",
        np.asarray(
            [
                [*baseline_center, *size, 3.0],
                [*candidate_center, *size, 4.0],
            ],
            dtype=np.float64,
        ),
    )

    baseline_root.mkdir()
    baseline_path = baseline_root / f"{SCENE}_boxes.pkl"
    with baseline_path.open("wb") as handle:
        pickle.dump([[(0, _corners(baseline_center, size), 0.73)]], handle)

    frames = rgbd_root / SCENE / "frames"
    for folder in ("color", "depth", "pose", "intrinsic"):
        (frames / folder).mkdir(parents=True, exist_ok=True)
    (frames / "color" / "0.jpg").write_bytes(b"schedule-only")
    pose = np.eye(4)
    pose[:3, 3] = offset
    np.savetxt(frames / "pose" / "0.txt", pose)
    intrinsics = np.eye(4)
    intrinsics[0, 0] = 20.0
    intrinsics[1, 1] = 20.0
    intrinsics[0, 2] = 6.0
    intrinsics[1, 2] = 6.0
    np.savetxt(frames / "intrinsic" / "intrinsic_depth.txt", intrinsics)
    Image.fromarray(np.full((16, 16), 1000, dtype=np.uint16)).save(
        frames / "depth" / "0.png"
    )

    rows = [_csv_row()]
    _write_csv(boxer_root / SCENE / "boxer_3dbbs.csv", rows)
    _write_csv(boxer_root / SCENE / "boxer_3dbbs_tracked.csv", rows)
    return {
        "boxer_root": boxer_root,
        "schedule_root": schedule_root,
        "baseline_root": baseline_root,
        "gt_root": gt_root,
        "scan_root": scan_root,
        "rgbd_root": rgbd_root,
        "scene_list": scene_list,
        "baseline_path": baseline_path,
        "offset": offset,
    }


def test_end_to_end_oracle_restores_offset_gates_depth_and_keeps_native(tmp_path):
    paths = _make_end_to_end_tree(tmp_path)
    output = tmp_path / "reports" / "oracle.json"
    before = paths["baseline_path"].read_bytes()
    assert (
        main(
            [
                "--boxer-root",
                str(paths["boxer_root"]),
                "--schedule-root",
                str(paths["schedule_root"]),
                "--baseline-root",
                str(paths["baseline_root"]),
                "--scene-list",
                str(paths["scene_list"]),
                "--gt-root",
                str(paths["gt_root"]),
                "--scan-root",
                str(paths["scan_root"]),
                "--scene-rgbd-root",
                str(paths["rgbd_root"]),
                "--out",
                str(output),
            ]
        )
        == 0
    )
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["oracle_only"] is True
    assert report["deployable"] is False
    assert report["gt_used"] is True
    assert report["birth_enabled"] is False
    assert report["native_predictions_modified"] is False
    assert report["scenes"][SCENE]["world_offset_restored"] == paths[
        "offset"
    ].tolist()
    assert report["pools"]["raw"]["candidate_count"] == 1
    assert report["pools"]["depth_gated"]["candidate_count"] == 1
    assert report["pools"]["tracked"]["candidate_count"] == 1
    for pool in ("raw", "depth_gated", "tracked"):
        at_50 = report["pools"][pool]["per_threshold"]["0.50"]
        assert at_50["coverage_gt_count"] == 1
        assert at_50["maximum_matching_count"] == 1
        assert at_50["baseline_unmatched_recovered_count"] == 1
        assert at_50["native_maximum_matching_count"] == 1
        assert at_50["native_union_pool_maximum_matching_count"] == 2
        assert at_50["union_recoverable_gt_count_over_native_maximum_matching"] == 1
        assert at_50["necessary_headroom_for_plus10"]["passes"] is True
        suffix = report["gt_selected_fixed_native_prefix_suffix"][pool][
            "per_threshold"
        ]["0.50"]
        assert suffix["oracle_only"] is True
        assert suffix["constructive_counterfactual"] is True
        assert suffix["mathematical_upper_bound"] is False
        assert suffix["selected_candidate_count"] == 1
        assert suffix["delta_greedy_tp"] == 1
    assert paths["baseline_path"].read_bytes() == before
    assert report["promotion"]["constructive_counterfactual_passes_all_thresholds"] is True
    assert report["promotion"]["birth_may_be_enabled"] is False


def test_cli_refuses_to_write_inside_native_prediction_root(tmp_path):
    paths = _make_end_to_end_tree(tmp_path)
    with pytest.raises(OracleAuditError, match="protected input root"):
        main(
            [
                "--boxer-root",
                str(paths["boxer_root"]),
                "--schedule-root",
                str(paths["schedule_root"]),
                "--baseline-root",
                str(paths["baseline_root"]),
                "--scene-list",
                str(paths["scene_list"]),
                "--gt-root",
                str(paths["gt_root"]),
                "--scan-root",
                str(paths["scan_root"]),
                "--scene-rgbd-root",
                str(paths["rgbd_root"]),
                "--out",
                str(paths["baseline_root"] / "oracle.json"),
            ]
        )


def test_sealed_manifest_filters_extra_csv_frames_and_records_zero_rows(tmp_path):
    paths = _make_end_to_end_tree(tmp_path)
    manifest_path = paths["schedule_root"] / SCENE / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["record_count"] = 2
    manifest["recorded_frame_ids"] = [0, 25]
    manifest["schedule"]["dataset_length"] = 26
    manifest["records"] = [{"frame_id": 0}, {"frame_id": 25}]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    raw_path = paths["boxer_root"] / SCENE / "boxer_3dbbs.csv"
    extra = _csv_row()
    extra["time_ns"] = "50"
    _write_csv(raw_path, [_csv_row(), extra])
    output = tmp_path / "oracle_schedule.json"
    main(
        [
            "--boxer-root",
            str(paths["boxer_root"]),
            "--schedule-root",
            str(paths["schedule_root"]),
            "--baseline-root",
            str(paths["baseline_root"]),
            "--scene-list",
            str(paths["scene_list"]),
            "--gt-root",
            str(paths["gt_root"]),
            "--scan-root",
            str(paths["scan_root"]),
            "--scene-rgbd-root",
            str(paths["rgbd_root"]),
            "--out",
            str(output),
        ]
    )
    report = json.loads(output.read_text(encoding="utf-8"))
    scene = report["scenes"][SCENE]
    assert scene["sealed_schedule_frame_count"] == 2
    assert scene["raw_candidate_count_before_schedule_filter"] == 2
    assert scene["raw_candidate_count"] == 1
    assert scene["raw_missing_frame_ids_treated_as_zero_candidates"] == [25]
    assert scene["raw_extra_frame_ids_excluded"] == [50]
    assert scene["raw_extra_candidate_count_excluded"] == 1
    assert scene["tracked_pool_schedule_clean"] is False
    assert report["pools"]["tracked"]["schedule_clean"] is False


def test_shadow_seal_validation_joins_candidates_by_stable_source_row(tmp_path):
    boxer_root = tmp_path / "boxer"
    schedule_root = tmp_path / "schedule"
    raw_path = boxer_root / SCENE / "boxer_3dbbs.csv"
    tracked_path = boxer_root / SCENE / "boxer_3dbbs_tracked.csv"
    manifest_path = schedule_root / SCENE / "manifest.json"
    _write_csv(raw_path, [_csv_row(), _csv_row(center=(0.1, 0.0, 1.0))])
    _write_csv(tracked_path, [_csv_row(), _csv_row(center=(0.1, 0.0, 1.0))])
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text("{}", encoding="utf-8")

    raw = [_candidate(row=0), _candidate(center=(0.1, 0.0, 1.0), row=1)]
    tracked = [replace(candidate, source="tracked") for candidate in raw]
    order = [1, 0]
    npz_path = tmp_path / "boxer_shadow_candidates.npz"
    np.savez(
        npz_path,
        per_view_center_world=np.stack([raw[index].center_world for index in order]),
        per_view_extent_xyz=np.stack([raw[index].size for index in order]),
        per_view_frame_id=np.asarray([raw[index].frame_id for index in order]),
        per_view_quaternion_wxyz=np.tile([1.0, 0.0, 0.0, 0.0], (2, 1)),
        per_view_scene_index=np.zeros(2, dtype=np.int64),
        per_view_source_instance_id=np.asarray(
            [raw[index].instance_id for index in order]
        ),
        per_view_source_row=np.asarray([raw[index].row_index for index in order]),
        per_view_source_score=np.asarray([raw[index].probability for index in order]),
        scene_ids=np.asarray([SCENE]),
        tracked_center_world=np.stack(
            [tracked[index].center_world for index in order]
        ),
        tracked_extent_xyz=np.stack([tracked[index].size for index in order]),
        tracked_instance_id=np.asarray(
            [tracked[index].instance_id for index in order]
        ),
        tracked_quaternion_wxyz=np.tile([1.0, 0.0, 0.0, 0.0], (2, 1)),
        tracked_scene_index=np.zeros(2, dtype=np.int64),
        tracked_source_row=np.asarray(
            [tracked[index].row_index for index in order]
        ),
        tracked_source_score=np.asarray(
            [tracked[index].probability for index in order]
        ),
    )
    json_path = tmp_path / "boxer_shadow_candidates.json"
    json_path.write_text(
        json.dumps(
            {
                "schema": "boxfusion.owl_boxer_shadow_candidates.v1",
                "mode": "shadow",
                "coordinate_frame": "scannet_world",
                "gt_access": False,
                "gt_access_guard": (
                    "BOXFUSION_SHADOW_GT_ACCESS=forbidden annotation_path=None"
                ),
                "gt_access_guard_verified": True,
                "output_inert": True,
                "birth": False,
                "native_before_after_identity": True,
                "native_clip_unchanged": True,
                "semantic_source_exported": False,
                "npz_file": npz_path.name,
                "npz_sha256": _sha256(npz_path),
                "scene_count": 1,
                "per_view_candidate_count": 2,
                "tracked_candidate_count": 2,
                "candidate_content_sha256": "fixture-content",
                "native_identity_ledger_sha256": "fixture-native",
                "assets_and_protocol": {
                    "profile": "clean_in2",
                    "detector": "owl",
                    "taxonomy": "lvisplus",
                    "taxonomy_count": 1220,
                    "start_n": 1,
                    "skip_n": 25,
                    "threshold_2d": 0.25,
                    "threshold_3d": 0.5,
                    "nms_iou_2d": 0.5,
                    **{
                        key: "0" * 64
                        for key in (
                            "boxer_checkpoint_sha256",
                            "boxernet_source_sha256",
                            "dinov3_checkpoint_sha256",
                            "owl_checkpoint_sha256",
                            "owl_text_cache_sha256",
                            "owl_wrapper_sha256",
                            "run_boxer_sha256",
                            "taxonomy_sha256",
                        )
                    },
                },
                "scenes": [
                    {
                        "scene_id": SCENE,
                        "scene_index": 0,
                        "gt_access_guard_verified": True,
                        "tracked_schedule_clean": True,
                        "per_view_extra_schedule_rows_excluded": 0,
                        "per_view_kept_rows": 2,
                        "tracked_kept_rows": 2,
                        "world_offset_xyz": [0.0, 0.0, 0.0],
                        "sealed_schedule_manifest_sha256": _sha256(manifest_path),
                        "inputs": {
                            "boxer_3dbbs_csv_sha256": _sha256(raw_path),
                            "boxer_3dbbs_tracked_csv_sha256": _sha256(tracked_path),
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    verified = validate_shadow_seal(
        json_path=json_path,
        npz_path=npz_path,
        scenes=[SCENE],
        raw_candidates=[raw],
        tracked_candidates=[tracked],
        boxer_root=boxer_root,
        schedule_root=schedule_root,
    )
    assert verified["verified"] is True
    assert verified["npz_sha256"] == _sha256(npz_path)
