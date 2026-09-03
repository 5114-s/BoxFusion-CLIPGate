from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from boxfusion.capf import (
    CAPF,
    RAY_FREE_SPACE,
    RAY_INVALID,
    RAY_OCCLUDED,
    RAY_SURFACE,
    box_to_local_faces,
    classify_rays,
    local_faces_to_box,
    resolve_capf_config,
    _box_corners,
)


def _enabled_cfg(**overrides):
    values = {
        "enabled": True,
        "max_ray_samples": 9,
        "min_valid_depth_samples": 4,
        "min_surface_rays": 4,
        "min_reference_rays": 4,
        "min_loss_improvement": 0.001,
        "min_face_visibility_cosine": 0.50,
        "max_accepted_faces": 1,
    }
    values.update(overrides)
    return {"capf": values}


def _rotation_z(angle):
    cosine, sine = np.cos(angle), np.sin(angle)
    return np.asarray(
        [[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def _surface_grid(z):
    values = (-0.30, 0.0, 0.30)
    return np.asarray(
        [[x, y, z] for y in values for x in values], dtype=np.float64
    )


def _refine(points_z):
    capf = CAPF(_enabled_cfg())
    anchor = np.asarray([0.0, 0.0, 2.0, 2.0, 2.0, 2.0], dtype=np.float32)
    observations = np.repeat(anchor[None], 3, axis=0)
    # Only source view 0 proposes a changed camera-facing z- face: 1.00 -> 0.93.
    observations[0, 2] = 1.965
    observations[0, 5] = 2.07
    rotations = np.repeat(np.eye(3, dtype=np.float32)[None], 3, axis=0)
    poses = np.repeat(np.eye(4, dtype=np.float32)[None], 3, axis=0)
    points = np.stack([_surface_grid(z) for z in points_z]).astype(np.float32)
    valid = np.ones(points.shape[:2], dtype=bool)

    snapshots = tuple(
        value.copy() for value in (anchor, observations, rotations, poses, points, valid)
    )
    result = capf.refine(
        anchor_box_xyzlhw=anchor,
        anchor_rotation=rotations[0],
        observation_boxes_xyzlhw=observations,
        observation_rotations=rotations,
        camera_poses=poses,
        surface_points_world=points,
        surface_valid=valid,
        frame_ids=np.asarray([0, 25, 50]),
    )
    for actual, expected in zip(
        (anchor, observations, rotations, poses, points, valid), snapshots
    ):
        np.testing.assert_array_equal(actual, expected)
    return anchor, rotations[0], result


def test_missing_config_is_disabled_and_invalid_protocol_fails_fast():
    config = resolve_capf_config({})
    assert config["enabled"] is False
    assert config["min_views"] == 3

    with pytest.raises(ValueError):
        resolve_capf_config({"capf": {"enabled": True, "min_views": 2}})
    with pytest.raises(ValueError):
        resolve_capf_config(
            {"capf": {"surface_weight": 0.8, "free_space_weight": 0.3}}
        )
    with pytest.raises(ValueError):
        resolve_capf_config(
            {"capf": {"max_ray_samples": 8, "min_valid_depth_samples": 9}}
        )


def test_disabled_refinement_is_an_exact_no_op():
    anchor = np.asarray([1.0, 2.0, 3.0, 2.0, 1.0, 0.5], dtype=np.float32)
    rotation = _rotation_z(0.25).astype(np.float32)
    result = CAPF({}).refine(
        anchor_box_xyzlhw=anchor,
        anchor_rotation=rotation,
        observation_boxes_xyzlhw=np.empty((0, 6)),
        observation_rotations=np.empty((0, 3, 3)),
        camera_poses=np.empty((0, 4, 4)),
        surface_points_world=np.empty((0, 9, 3)),
        surface_valid=np.empty((0, 9), dtype=bool),
    )
    assert not result.accepted
    assert result.reason == "disabled"
    np.testing.assert_array_equal(result.box_xyzlhw, anchor)
    np.testing.assert_array_equal(result.rotation, rotation)


def test_directed_faces_round_trip_and_one_face_changes_one_extent():
    reference = np.asarray([1.0, -2.0, 3.0, 2.0, 4.0, 6.0])
    rotation = _rotation_z(np.deg2rad(30.0))
    faces = box_to_local_faces(reference)
    np.testing.assert_allclose(
        local_faces_to_box(reference, rotation, faces), reference, atol=1.0e-12
    )

    changed = faces.copy()
    changed[0, 1] += 0.20
    output = local_faces_to_box(reference, rotation, changed)
    expected_center = reference[:3] + 0.10 * rotation[:, 0]
    np.testing.assert_allclose(output[:3], expected_center, atol=1.0e-12)
    np.testing.assert_allclose(output[3:], [2.20, 4.0, 6.0], atol=1.0e-12)
    np.testing.assert_array_equal(changed[1:], faces[1:])


def test_ray_taxonomy_surface_occluded_free_space_and_invalid():
    labels, entries, measured = classify_rays(
        np.asarray([0.0, 0.0, 2.0, 2.0, 2.0, 2.0]),
        np.eye(3),
        np.zeros(3),
        np.asarray(
            [
                [0.0, 0.0, 1.0],
                [0.0, 0.0, 0.8],
                [0.0, 0.0, 1.3],
                [0.0, 0.0, 0.0],
            ]
        ),
        np.asarray([True, True, True, False]),
        surface_band_m=0.05,
        occlusion_margin_m=0.05,
        free_space_margin_m=0.05,
    )
    np.testing.assert_array_equal(
        labels,
        [RAY_SURFACE, RAY_OCCLUDED, RAY_FREE_SPACE, RAY_INVALID],
    )
    assert entries[0] == pytest.approx(1.0)
    assert measured[2] == pytest.approx(1.3)
    assert np.isnan(entries[3])


def test_good_heldout_views_accept_one_face_and_keep_rotation_exact():
    # Source-view depth is intentionally unrelated: view 0 generates the face,
    # while only views 1 and 2 are allowed to validate it.
    anchor, rotation, result = _refine([1.50, 0.93, 0.93])
    assert result.accepted, result.reason
    assert result.reason == "accepted"
    assert len(result.updates) == 1
    update = result.updates[0]
    assert update.face_index == 4
    assert update.source_view == 0
    assert update.heldout_views == (1, 2)
    assert update.median_loss_improvement > 0.0
    np.testing.assert_allclose(
        result.box_xyzlhw,
        [0.0, 0.0, 1.965, 2.0, 2.0, 2.07],
        atol=1.0e-6,
    )
    np.testing.assert_array_equal(result.rotation, rotation)
    assert result.box_xyzlhw.dtype == anchor.dtype


def test_bad_heldout_views_reject_and_roll_back_bit_exact():
    anchor, rotation, result = _refine([0.93, 1.00, 1.00])
    assert not result.accepted
    assert result.reason == "no_heldout_improvement"
    np.testing.assert_array_equal(result.box_xyzlhw, anchor)
    np.testing.assert_array_equal(result.rotation, rotation)
    assert result.updates == ()


class _DummyInstances:
    def __init__(self):
        self.pred_boxes = torch.tensor(
            [[1.0, 1.0, 5.0, 5.0], [2.0, 2.0, 6.0, 6.0]],
            dtype=torch.float32,
        )
        self.scores = torch.tensor([0.91, 0.63], dtype=torch.float32)
        self.categories = np.asarray(["chair", "table"])
        self.init_id = torch.tensor([10, 11], dtype=torch.int64)

    def __len__(self):
        return int(self.pred_boxes.shape[0])


def test_observation_attachment_only_adds_fixed_evidence_fields():
    instances = _DummyInstances()
    protected = {
        "pred_boxes": instances.pred_boxes.clone(),
        "scores": instances.scores.clone(),
        "categories": instances.categories.copy(),
        "init_id": instances.init_id.clone(),
    }
    capf = CAPF(_enabled_cfg())
    depth = torch.full((8, 8), 1.5, dtype=torch.float32)
    intrinsic = torch.tensor(
        [[4.0, 0.0, 4.0], [0.0, 4.0, 4.0], [0.0, 0.0, 1.0]],
        dtype=torch.float32,
    )
    capf.attach_observations(
        instances,
        depth_m=depth,
        intrinsics=intrinsic,
        image_height=8,
        image_width=8,
        camera_to_world=torch.eye(4),
    )

    assert instances.capf_surface_points_world.shape == (2, 9, 3)
    assert instances.capf_surface_points_world.dtype == torch.float32
    assert instances.capf_surface_valid.shape == (2, 9)
    assert instances.capf_surface_valid.dtype == torch.bool
    assert torch.all(
        instances.capf_surface_points_world[..., 2][instances.capf_surface_valid]
        == 1.5
    )
    torch.testing.assert_close(instances.pred_boxes, protected["pred_boxes"])
    torch.testing.assert_close(instances.scores, protected["scores"])
    np.testing.assert_array_equal(instances.categories, protected["categories"])
    torch.testing.assert_close(instances.init_id, protected["init_id"])


def test_oracle_shadow_writes_terminal_candidate_bank_without_geometry_change(
    tmp_path,
):
    config = _enabled_cfg(
        oracle_shadow=True,
        oracle_diagnostics_dir=str(tmp_path),
    )
    capf = CAPF(config)
    anchor = np.asarray([0.0, 0.0, 2.0, 2.0, 2.0, 2.0])
    rotation = np.eye(3)
    capf._record_oracle_snapshot(
        track_key=[1, 2],
        anchor_box=anchor,
        rotation=rotation,
        proposals=[(4, 0, -0.9)],
        selected_box=anchor,
        selected_updates=(),
    )
    output = capf.write_oracle_diagnostics(
        scene_id="scene0000_00",
        final_track_keys=[[1, 2, 3]],
        final_corners_world=_box_corners(anchor, rotation)[None],
        final_scores=[0.75],
    )
    payload = __import__("json").loads(output.read_text(encoding="utf-8"))
    assert payload["gt_access"] is False
    assert payload["online_writeback"] is False
    assert payload["candidate_snapshot_count"] == 1
    row = payload["rows"][0]
    assert row["track_key"] == [1, 2, 3]
    snapshot = row["candidate_snapshot"]
    assert snapshot["track_key"] == [1, 2]
    assert snapshot["face_options"][0]["face_index"] == 4


def test_released_route_config_enables_only_capf_after_boxer_topk3():
    config_path = (
        Path(__file__).resolve().parents[1]
        / "config"
        / "scannet_t05_boxer_capf_topk3_real_score05.yaml"
    )
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert config["lifting"]["backend"] == "boxer"
    assert config["lifting"]["boxer"]["mode"] == "active"
    assert "boxer_gsa" not in config["lifting"]
    assert "boxer_mvpr" not in config["lifting"]
    assert config["association"]["appearance_gate"]["enabled"] is False
    fusion = config["box_fusion"]
    assert fusion["reliable_views"]["enabled"] is True
    assert fusion["reliable_views"]["top_k"] == 3
    assert fusion["reliable_views"]["min_views"] == 3
    assert fusion["capf"]["enabled"] is True
    assert "vapf_lite" not in fusion
    assert "maskdepth_pfo" not in fusion
