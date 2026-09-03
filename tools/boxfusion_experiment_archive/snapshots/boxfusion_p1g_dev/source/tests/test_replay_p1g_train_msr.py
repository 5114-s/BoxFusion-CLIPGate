"""Synthetic contracts for the train-only P1G replay/MSR audit."""

from __future__ import annotations

import json
import math
import pickle
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from boxfusion.p1_spatial_residual import (
    NativeSparseResidualProposalHead,
)
from boxfusion.residual_proposal import (
    P1_DIAGNOSTIC_SCHEMA,
    P1_FEATURE_NAMES,
    P1S_HEAD_SCHEMA,
    ResidualProposalConfig,
)
from tools.replay_p1g_train_msr import (
    OUTPUT_SCHEMA,
    actual_iou_diagnostics,
    backproject_depth,
    bounded_face_oracle,
    feasibility_sweep,
    file_sha256,
    load_frozen_parents,
    replay_scene,
    validate_scene_partition,
)


SCENE = "scene0001_00"
FORBIDDEN = "scene9999_00"


def _write_legacy_collect(path: Path) -> None:
    coordinates_one = np.asarray(
        [[0, 0, 12], [1, 0, 12], [0, 1, 12], [1, 1, 12]],
        dtype=np.int32,
    )
    coordinates = np.concatenate((coordinates_one, coordinates_one), axis=0)
    centers = (coordinates.astype(np.float32) + 0.5) * 0.08
    features = np.zeros((len(coordinates), len(P1_FEATURE_NAMES)), np.float32)
    features[:, 0] = 0.5
    config = ResidualProposalConfig(
        enabled=True,
        observer_only=True,
        mutate=False,
        collect_diagnostics=True,
        mode="collect",
        collect_voxel_inputs=True,
        voxel_size=0.08,
    ).validated()
    np.savez_compressed(
        path,
        scene_id=np.asarray(SCENE),
        p1_schema=np.asarray(P1_DIAGNOSTIC_SCHEMA),
        p1_stage=np.asarray("P1"),
        p1_profile=np.asarray("p1_residual_proposal_observer"),
        p1_enabled=np.asarray(True, dtype=bool),
        p1_observer_only=np.asarray(True, dtype=bool),
        p1_uses_ground_truth=np.asarray(False, dtype=bool),
        p1_mutation_enabled=np.asarray(False, dtype=bool),
        p1_applied_count=np.asarray(0, dtype=np.int64),
        p1_complete=np.asarray(True, dtype=bool),
        p1_class_agnostic=np.asarray(True, dtype=bool),
        p1_regression_dim=np.asarray(6, dtype=np.int64),
        p1_config_json=np.asarray(
            json.dumps(
                config.to_dict(),
                sort_keys=True,
                separators=(",", ":"),
            )
        ),
        p1_feature_names=np.asarray(P1_FEATURE_NAMES, dtype=np.str_),
        p1_step_frame_ids=np.asarray([0, 1], dtype=np.int64),
        p1_step_provider_steps=np.asarray([0, 1], dtype=np.int64),
        p1_step_voxel_counts=np.asarray([4, 4], dtype=np.int64),
        p1_voxel_offsets=np.asarray([0, 4, 8], dtype=np.int64),
        p1_voxel_coords=coordinates,
        p1_voxel_centers=centers,
        p1_voxel_features=features,
        p1_voxel_point_counts=np.full(len(coordinates), 8, dtype=np.int32),
    )


def _write_frames(frames_root: Path) -> None:
    frame_root = frames_root / SCENE / "frames"
    for name in ("depth", "pose", "intrinsic"):
        (frame_root / name).mkdir(parents=True, exist_ok=True)
    intrinsic = np.eye(4, dtype=np.float64)
    intrinsic[0, 0] = intrinsic[1, 1] = 24.0
    intrinsic[0, 2] = intrinsic[1, 2] = 15.5
    np.savetxt(frame_root / "intrinsic" / "intrinsic_depth.txt", intrinsic)
    depth = np.full((32, 32), 1000, dtype=np.uint16)
    for frame_id in (0, 1):
        Image.fromarray(depth).save(
            frame_root / "depth" / f"{frame_id}.png"
        )
        pose = np.eye(4, dtype=np.float64)
        pose[0, 3] = 0.02 * frame_id
        np.savetxt(frame_root / "pose" / f"{frame_id}.txt", pose)


def _synthetic_tree(tmp_path: Path) -> dict[str, Path]:
    diagnostics = tmp_path / "diagnostics"
    predictions = tmp_path / "predictions"
    gt_root = tmp_path / "gt"
    scans = tmp_path / "scans"
    frames = tmp_path / "frames"
    output = tmp_path / "output"
    for root in (diagnostics, predictions, gt_root):
        root.mkdir(parents=True)
    (scans / SCENE).mkdir(parents=True)
    diagnostic = diagnostics / f"{SCENE}_tracks.npz"
    prediction = predictions / f"{SCENE}_boxes.pkl"
    gt = gt_root / f"{SCENE}_bbox.npy"
    alignment = scans / SCENE / f"{SCENE}.txt"
    _write_legacy_collect(diagnostic)
    with prediction.open("wb") as handle:
        pickle.dump([[]], handle)
    np.save(
        gt,
        np.asarray([[0.04, 0.04, 1.0, 1.0, 1.0, 1.0, 1]], np.float32),
        allow_pickle=False,
    )
    alignment.write_text(
        "axisAlignment = "
        + " ".join(str(value) for value in np.eye(4).reshape(-1))
        + "\n",
        encoding="utf-8",
    )
    _write_frames(frames)

    b6 = tmp_path / "b6.npz"
    np.savez(b6, marker=np.asarray(1, dtype=np.int64))
    scene_list = tmp_path / "train.txt"
    forbidden_list = tmp_path / "val.txt"
    scene_list.write_text(SCENE + "\n", encoding="utf-8")
    forbidden_list.write_text(FORBIDDEN + "\n", encoding="utf-8")

    head = NativeSparseResidualProposalHead(
        input_dim=len(P1_FEATURE_NAMES),
        hidden_dim=8,
        regression_dim=6,
    )
    with torch.no_grad():
        for parameter in head.parameters():
            parameter.zero_()
        head.objectness.bias.fill_(4.0)
        head.regression.bias.copy_(
            torch.tensor(
                [0.0, 0.0, 0.0, math.log(1.0), math.log(1.0), math.log(1.0)]
            )
        )
    p1s = tmp_path / "p1s.pt"
    torch.save(
        {
            "schema": P1S_HEAD_SCHEMA,
            "variant": "P1S",
            "head_architecture": "native_sparse_context_v1",
            "target_assignment_scope": "snapshot_inside_only",
            "model_config": head.model_config(),
            "feature_names": list(P1_FEATURE_NAMES),
            "state_dict": head.state_dict(),
            "training_config": {
                "target_assignment_scope": "snapshot_inside_only"
            },
            "provenance": {
                "train_scene_ids": [SCENE],
                "forbidden_overlap": [],
                "b6_checkpoint_sha256": file_sha256(b6),
                "train_scene_list_sha256": file_sha256(scene_list),
                "forbidden_scene_list_sha256": file_sha256(forbidden_list),
                "scene_summaries": [
                    {
                        "scene_id": SCENE,
                        "diagnostic_sha256": file_sha256(diagnostic),
                        "prediction_sha256": file_sha256(prediction),
                        "ground_truth_sha256": file_sha256(gt),
                        "axis_alignment_sha256": file_sha256(alignment),
                    }
                ],
            },
        },
        p1s,
    )
    return {
        "diagnostics": diagnostics,
        "predictions": predictions,
        "gt": gt_root,
        "scans": scans,
        "frames": frames,
        "output": output,
        "b6": b6,
        "p1s": p1s,
        "scene_list": scene_list,
        "forbidden_list": forbidden_list,
    }


def test_scene_partition_rejects_validation_leakage() -> None:
    validate_scene_partition([SCENE], [FORBIDDEN])
    with pytest.raises(ValueError, match="overlaps forbidden"):
        validate_scene_partition([SCENE], [SCENE])


def test_backproject_depth_uses_pose_and_metric_scale() -> None:
    depth = np.asarray([[1000, 0], [2000, 10000]], dtype=np.uint16)
    intrinsic = np.asarray(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    )
    pose = np.eye(4)
    pose[:3, 3] = [1.0, 2.0, 3.0]
    points, ratio = backproject_depth(
        depth,
        intrinsic,
        pose,
        stride=1,
        depth_scale=1000.0,
        min_depth=0.1,
        max_depth=8.0,
    )
    assert ratio == pytest.approx(0.5)
    assert points.shape == (2, 3)
    np.testing.assert_allclose(points[0], [1.0, 2.0, 4.0])
    np.testing.assert_allclose(points[1], [1.0, 4.0, 5.0])


def test_face_oracle_sweep_exposes_bound_limited_geometry() -> None:
    candidate = np.asarray([0.0, 0.0, 0.0, 1.0, 1.0, 1.0])
    target = np.asarray([0.30, 0.0, 0.0, 1.0, 1.0, 1.0])
    conservative = bounded_face_oracle(candidate, target, 0.18)
    relaxed = bounded_face_oracle(candidate, target, 0.50)
    conservative_iou = actual_iou_diagnostics(
        parent_boxes=candidate[None],
        refined_boxes=conservative[None],
        gt_boxes=target[None],
        baseline_boxes=np.empty((0, 6)),
        covered_iou=0.15,
    )["refined_same_gt_iou"][0]
    relaxed_iou = actual_iou_diagnostics(
        parent_boxes=candidate[None],
        refined_boxes=relaxed[None],
        gt_boxes=target[None],
        baseline_boxes=np.empty((0, 6)),
        covered_iou=0.15,
    )["refined_same_gt_iou"][0]
    assert relaxed_iou > conservative_iou
    assert relaxed_iou == pytest.approx(1.0)
    sweep = feasibility_sweep(
        candidate_boxes=candidate[None],
        gt_boxes=target[None],
        baseline_boxes=np.empty((0, 6)),
        face_limits=(0.18, 0.50),
        covered_iou=0.15,
        initial_min_iou=0.15,
        initial_max_iou=0.99,
    )
    assert sweep["refined_iou"].shape == (1, 2)
    assert sweep["refined_iou"][0, 1] > sweep["refined_iou"][0, 0]


def test_synthetic_scene_replay_writes_pickle_free_provenance(
    tmp_path: Path,
) -> None:
    tree = _synthetic_tree(tmp_path)
    parents = load_frozen_parents(
        tree["p1s"],
        tree["b6"],
        requested_scenes=[SCENE],
        forbidden_scene_list=tree["forbidden_list"],
        score_threshold=0.05,
        max_scene_candidates=16,
    )
    summary = replay_scene(
        scene_id=SCENE,
        diagnostics_root=tree["diagnostics"],
        prediction_root=tree["predictions"],
        gt_root=tree["gt"],
        scans_root=tree["scans"],
        frames_root=tree["frames"],
        parents=parents,
        output_root=tree["output"],
        face_limits=(0.18, 0.25, 0.50, 0.75),
        covered_iou=0.15,
        depth_stride=1,
        depth_scale=1000.0,
        min_depth=0.15,
        max_depth=8.0,
        explained_margin=0.05,
        p1g_config={
            "enabled": True,
            "observer_only": True,
            "mutate": False,
            "collect_diagnostics": True,
            "association_iou": 0.10,
            "crop_scale": 1.35,
            "top_k_views": 2,
            "view_diversity_weight": 0.25,
            "max_points_per_view": 128,
            "max_candidates": 16,
            "proposal": {
                "min_views": 2,
                "min_points_per_view": 8,
                "min_total_points": 16,
                "fine_min_view_consensus": 2,
                "min_component_views": 2,
                "min_component_points": 8,
                "face_min_views": 2,
                "face_min_points_per_view": 2,
                "maximum_face_shift_ratio": 0.18,
                "minimum_extent_ratio": 0.70,
                "maximum_extent_ratio": 1.25,
                "maximum_center_shift_ratio": 0.15,
            },
        },
    )
    output = Path(summary["output"])
    assert output.is_file()
    with np.load(output, allow_pickle=False) as archive:
        assert archive["schema"].item() == OUTPUT_SCHEMA
        assert archive["observer_only"].item() is True
        assert archive["mutation_enabled"].item() is False
        assert archive["applied_count"].item() == 0
        assert archive["p1s_checkpoint_sha256"].item() == file_sha256(
            tree["p1s"]
        )
        assert len(archive["candidate_ids"]) > 0
        assert archive["candidate_boxes"].shape == archive[
            "refined_boxes"
        ].shape
        assert archive["feasibility_refined_iou"].shape[1] == 4
        assert all(
            not archive[name].dtype.hasobject for name in archive.files
        )
        provenance = json.loads(archive["provenance_json"].item())
        assert provenance["diagnostic_sha256"] == file_sha256(
            tree["diagnostics"] / f"{SCENE}_tracks.npz"
        )
        assert provenance["geometry_source"] == (
            "scheduled_depth_minus_frozen_final_b6_boxes"
        )


def test_parent_loader_rejects_wrong_b6(tmp_path: Path) -> None:
    tree = _synthetic_tree(tmp_path)
    wrong_b6 = tmp_path / "wrong_b6.npz"
    np.savez(wrong_b6, marker=np.asarray(2, dtype=np.int64))
    with pytest.raises(ValueError, match="different B6"):
        load_frozen_parents(
            tree["p1s"],
            wrong_b6,
            requested_scenes=[SCENE],
            forbidden_scene_list=tree["forbidden_list"],
        )
