import json

import numpy as np
import torch

from boxfusion.boxer_lifter import (
    BoxerLiftingAdapter,
    BoxerLiftingConfig,
    boxer_world_to_boxfusion_camera,
    deterministic_sdp_from_depth,
    geometry_hash,
    project_rotations_to_so3,
    project_centers,
    protected_proposal_hashes,
)
from boxfusion.boxes import GeneralInstance3DBoxes
from boxfusion.instances import Instances3D


def _rotation_z(angle):
    c = np.cos(angle)
    s = np.sin(angle)
    return torch.tensor(
        [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]],
        dtype=torch.float32,
    )


def _make_instances():
    instances = Instances3D((480, 640))
    instances.pred_boxes = torch.tensor(
        [[10.0, 20.0, 110.0, 220.0], [30.0, 40.0, 130.0, 240.0]]
    )
    instances.scores = torch.tensor([0.81, 0.63])
    instances.pred_classes = torch.tensor([0, 0])
    instances.pred_logits = torch.tensor([[1.2], [0.8]])
    instances.object_desc = torch.arange(8, dtype=torch.float32).reshape(2, 4)
    instances.pred_boxes_3d = GeneralInstance3DBoxes(
        torch.tensor(
            [
                [0.1, 0.2, 2.0, 1.0, 1.2, 0.8],
                [-0.3, 0.1, 3.0, 0.6, 0.7, 0.9],
            ]
        ),
        torch.eye(3).repeat(2, 1, 1),
    )
    instances.pred_proj_xy = torch.tensor([[100.0, 120.0], [200.0, 220.0]])
    return instances


def _config(tmp_path, mode):
    return BoxerLiftingConfig(
        mode=mode,
        apply_stage="post_filter",
        official_root="/does/not/load/in/fake/test",
        checkpoint="/does/not/load/in/fake/test.ckpt",
        expected_commit="test",
        checkpoint_sha256="",
        dinov3_sha256="",
        precision="float32",
        use_sdp=True,
        sdp_samples=100,
        seed=0,
        diagnostics_dir=str(tmp_path),
    )


class _FakeAdapter(BoxerLiftingAdapter):
    def _make_datum(self, **kwargs):
        boxes = kwargs["boxes_xyxy"].detach().float().cpu()
        return {}, {
            "image_np": np.zeros((4, 6, 3), dtype=np.uint8),
            "depth_np": np.ones((2, 3), dtype=np.float32),
            "image_K_np": np.array(
                [[100.0, 0.0, 20.0], [0.0, 100.0, 30.0], [0.0, 0.0, 1.0]],
                dtype=np.float32,
            ),
            "depth_K_np": np.eye(3, dtype=np.float32),
            "pose_np": np.eye(4, dtype=np.float32),
            "scaled_K": np.eye(3, dtype=np.float32),
            "boxer_boxes": boxes[:, [0, 2, 1, 3]],
            "sdp_seed": 123,
        }

    def _forward(self, datum, camera_to_world):
        logvar = torch.tensor([[0.1], [0.2]])
        return {
            "xyz_dims": torch.tensor(
                [
                    [0.2, 0.3, 2.5, 1.1, 1.3, 0.9],
                    [-0.4, 0.2, 3.5, 0.7, 0.8, 1.0],
                ]
            ),
            "rotations": _rotation_z(0.2).repeat(2, 1, 1),
            "confidence": (1.0 / (1.0 + torch.exp(logvar))).reshape(-1),
            "logvar": logvar,
            "raw_params": torch.zeros(2, 7),
            "runtime_ms": torch.tensor(12.5, dtype=torch.float64),
            "rotation_correction_max_abs": torch.tensor(
                0.0,
                dtype=torch.float64,
            ),
        }


def test_world_to_camera_round_trip():
    rotation_wc = _rotation_z(0.63)
    translation_wc = torch.tensor([1.5, -0.7, 2.1])
    camera_to_world = torch.eye(4)
    camera_to_world[:3, :3] = rotation_wc
    camera_to_world[:3, 3] = translation_wc

    center_camera = torch.tensor([[0.2, -0.4, 2.5], [-0.6, 0.1, 3.2]])
    rotation_camera_object = _rotation_z(-0.31).repeat(2, 1, 1)
    center_world = torch.einsum(
        "ij,nj->ni", rotation_wc, center_camera
    ) + translation_wc
    rotation_world_object = torch.einsum(
        "ij,njk->nik", rotation_wc, rotation_camera_object
    )
    dims = torch.tensor([[1.0, 2.0, 3.0], [0.4, 0.5, 0.6]])

    xyz_dims, recovered_rotation = boxer_world_to_boxfusion_camera(
        center_world,
        dims,
        rotation_world_object,
        camera_to_world,
    )
    torch.testing.assert_close(xyz_dims[:, :3], center_camera)
    torch.testing.assert_close(xyz_dims[:, 3:], dims)
    torch.testing.assert_close(recovered_rotation, rotation_camera_object)


def test_deterministic_sdp_does_not_touch_global_numpy_rng():
    depth = np.ones((80, 100), dtype=np.float32) * 2.0
    K = np.array(
        [[100.0, 0.0, 50.0], [0.0, 100.0, 40.0], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    pose = np.eye(4, dtype=np.float32)

    np.random.seed(17)
    before = np.random.random(4)
    first = deterministic_sdp_from_depth(
        depth, K, pose, num_samples=100, seed=91
    )
    after = np.random.random(4)

    np.random.seed(17)
    expected_before = np.random.random(4)
    expected_after = np.random.random(4)
    np.testing.assert_array_equal(before, expected_before)
    np.testing.assert_array_equal(after, expected_after)
    second = deterministic_sdp_from_depth(
        depth, K, pose, num_samples=100, seed=91
    )
    torch.testing.assert_close(first, second, rtol=0.0, atol=0.0)


def test_project_centers():
    xyz = torch.tensor([[0.0, 0.0, 2.0], [1.0, -1.0, 4.0]])
    K = torch.tensor(
        [[100.0, 0.0, 20.0], [0.0, 80.0, 30.0], [0.0, 0.0, 1.0]]
    )
    uv = project_centers(xyz, K)
    torch.testing.assert_close(
        uv,
        torch.tensor([[20.0, 30.0], [45.0, 10.0]]),
    )


def test_mixed_precision_rotation_is_projected_to_so3():
    rotation = _rotation_z(0.4).repeat(2, 1, 1)
    rotation[0, 0, 0] += 0.008
    rotation[1, :, 2] *= -1.0
    projected = project_rotations_to_so3(rotation)
    identity = torch.eye(3).repeat(2, 1, 1)
    torch.testing.assert_close(
        projected.transpose(-1, -2) @ projected,
        identity,
        atol=1e-5,
        rtol=1e-5,
    )
    torch.testing.assert_close(
        torch.linalg.det(projected),
        torch.ones(2),
        atol=1e-5,
        rtol=1e-5,
    )


def test_observer_is_output_identity(tmp_path):
    instances = _make_instances()
    protected_before = protected_proposal_hashes(instances)
    geometry_before = geometry_hash(instances)
    adapter = _FakeAdapter(_config(tmp_path, "observer"), device="cpu")

    result = adapter.apply(
        instances,
        image=np.zeros((4, 6, 3), dtype=np.uint8),
        depth=np.ones((2, 3), dtype=np.float32),
        image_K=np.eye(3, dtype=np.float32),
        depth_K=np.eye(3, dtype=np.float32),
        camera_to_world=np.eye(4, dtype=np.float32),
        scene_id="scene_test",
        frame_id=0,
    )

    assert result is instances
    assert protected_proposal_hashes(result) == protected_before
    assert geometry_hash(result) == geometry_before
    rows = [
        json.loads(line)
        for line in (tmp_path / "scene_test_boxer_lifting.jsonl")
        .read_text()
        .splitlines()
    ]
    assert rows[0]["mutation_enabled"] is False
    assert rows[0]["applied_count"] == 0
    assert rows[0]["attempt_id"] == "primary"


def test_active_changes_only_lifting_geometry(tmp_path):
    instances = _make_instances()
    protected_before = protected_proposal_hashes(instances)
    geometry_before = geometry_hash(instances)
    projected_centers_before = instances.pred_proj_xy.clone()
    adapter = _FakeAdapter(_config(tmp_path, "active"), device="cpu")

    result = adapter.apply(
        instances,
        image=np.zeros((4, 6, 3), dtype=np.uint8),
        depth=np.ones((2, 3), dtype=np.float32),
        image_K=np.eye(3, dtype=np.float32),
        depth_K=np.eye(3, dtype=np.float32),
        camera_to_world=np.eye(4, dtype=np.float32),
        scene_id="scene_test",
        frame_id=0,
    )

    assert protected_proposal_hashes(result) == protected_before
    assert geometry_hash(result) != geometry_before
    torch.testing.assert_close(
        result.pred_proj_xy,
        projected_centers_before,
        rtol=0.0,
        atol=0.0,
    )
    assert len(result) == 2
    torch.testing.assert_close(
        result.pred_boxes_3d.tensor[:, :3],
        torch.tensor([[0.2, 0.3, 2.5], [-0.4, 0.2, 3.5]]),
    )


def test_empty_attempt_is_audited_without_loading_boxer(tmp_path):
    instances = _make_instances()[torch.tensor([False, False])]
    adapter = _FakeAdapter(_config(tmp_path, "observer"), device="cpu")

    result = adapter.apply(
        instances,
        image=np.zeros((4, 6, 3), dtype=np.uint8),
        depth=np.ones((2, 3), dtype=np.float32),
        image_K=np.eye(3, dtype=np.float32),
        depth_K=np.eye(3, dtype=np.float32),
        camera_to_world=np.eye(4, dtype=np.float32),
        scene_id="scene_empty",
        frame_id=0,
        attempt_id="retry",
    )

    assert len(result) == 0
    row = json.loads(
        (tmp_path / "scene_empty_boxer_lifting.jsonl").read_text().strip()
    )
    assert row["count"] == 0
    assert row["attempt_id"] == "retry"
    assert row["applied_count"] == 0
