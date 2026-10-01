import json

import numpy as np
import pytest
import torch

from boxfusion.boxer_lifter import (
    BoxerLiftingAdapter,
    BoxerLiftingConfig,
    geometry_hash,
    project_rotations_to_so3,
    protected_proposal_hashes,
)
from boxfusion.boxes import GeneralInstance3DBoxes
from boxfusion.instances import Instances3D


ROW_COUNT = 8


def _config(tmp_path, mode="active", *, selective=True):
    return BoxerLiftingConfig(
        mode=mode,
        apply_stage="post_filter",
        official_root="/unused",
        checkpoint="/unused/checkpoint.ckpt",
        expected_commit="test",
        checkpoint_sha256="",
        dinov3_sha256="",
        precision="float32",
        use_sdp=False,
        sdp_samples=0,
        seed=0,
        diagnostics_dir=str(tmp_path),
        selective_gate_enabled=selective,
        selective_max_center_shift_m=0.10,
        selective_min_volume_ratio=0.50,
        selective_max_volume_ratio=2.00,
    )


def _instances():
    instances = Instances3D((480, 640))
    instances.pred_boxes = torch.arange(
        ROW_COUNT * 4, dtype=torch.float32
    ).reshape(ROW_COUNT, 4)
    instances.scores = torch.linspace(0.41, 0.90, ROW_COUNT)
    instances.pred_classes = torch.arange(ROW_COUNT)
    instances.pred_logits = torch.arange(
        ROW_COUNT * 2, dtype=torch.float32
    ).reshape(ROW_COUNT, 2)
    instances.object_desc = torch.arange(
        ROW_COUNT * 3, dtype=torch.float32
    ).reshape(ROW_COUNT, 3)
    cutr = torch.tensor(
        [[0.0, 0.0, 2.0, 1.0, 1.0, 1.0]] * ROW_COUNT,
        dtype=torch.float32,
    )
    # A malformed CuTR row cannot safely be replaced because the ratio has no
    # meaningful denominator.  Selective Boxer must preserve it exactly.
    cutr[7, 3] = -1.0
    instances.pred_boxes_3d = GeneralInstance3DBoxes(
        cutr,
        torch.eye(3).repeat(ROW_COUNT, 1, 1),
    )
    instances.pred_proj_xy = torch.arange(
        ROW_COUNT * 2, dtype=torch.float32
    ).reshape(ROW_COUNT, 2)
    return instances


def _boxer_prediction():
    xyz_dims = torch.tensor(
        [[0.0, 0.0, 2.0, 1.0, 1.0, 1.0]] * ROW_COUNT,
        dtype=torch.float32,
    )
    # Inclusive threshold boundaries: both rows must be eligible.
    xyz_dims[0, 0] = 0.10
    xyz_dims[0, 3] = 0.50
    xyz_dims[1, 3] = 2.00
    # One failure for each scalar gate.
    xyz_dims[2, 0] = 0.1001
    xyz_dims[3, 3] = 0.4999
    xyz_dims[4, 3] = 2.0001
    # Per-row invalid candidate and invalid rotation must fall back, not raise.
    xyz_dims[5, 0] = float("nan")
    rotations = torch.eye(3).repeat(ROW_COUNT, 1, 1)
    rotations[6] = 0.0
    return {
        "xyz_dims": xyz_dims,
        "rotations": rotations,
        "confidence": torch.full((ROW_COUNT,), 0.5),
        "logvar": torch.zeros(ROW_COUNT, 1),
        "raw_params": torch.zeros(ROW_COUNT, 7),
        "runtime_ms": torch.tensor(3.0, dtype=torch.float64),
        "rotation_correction_max_abs": torch.tensor(
            0.0, dtype=torch.float64
        ),
    }


class _SelectiveFakeAdapter(BoxerLiftingAdapter):
    def _make_datum(self, **kwargs):
        boxes = kwargs["boxes_xyxy"].detach().float().cpu()
        return {}, {
            "image_np": np.zeros((8, 12, 3), dtype=np.uint8),
            "depth_np": np.ones((4, 6), dtype=np.float32),
            "image_K_np": np.eye(3, dtype=np.float32),
            "depth_K_np": np.eye(3, dtype=np.float32),
            "pose_np": np.eye(4, dtype=np.float32),
            "scaled_K": np.eye(3, dtype=np.float32),
            "boxer_boxes": boxes[:, [0, 2, 1, 3]],
            "sdp_seed": None,
        }

    def _forward(self, datum, camera_to_world):
        return _boxer_prediction()


def _apply(adapter, instances, scene="scene_selective"):
    return adapter.apply(
        instances,
        image=np.zeros((8, 12, 3), dtype=np.uint8),
        depth=np.ones((4, 6), dtype=np.float32),
        image_K=np.eye(3, dtype=np.float32),
        depth_K=np.eye(3, dtype=np.float32),
        camera_to_world=np.eye(4, dtype=np.float32),
        scene_id=scene,
        frame_id=25,
    )


def _diagnostic(tmp_path, scene="scene_selective"):
    return json.loads(
        (tmp_path / f"{scene}_boxer_lifting.jsonl").read_text().strip()
    )


def test_active_replaces_only_eligible_rows_and_preserves_proposals(tmp_path):
    instances = _instances()
    cutr_xyz = instances.pred_boxes_3d.tensor.clone()
    cutr_rotations = instances.pred_boxes_3d.R.clone()
    protected_before = protected_proposal_hashes(instances)
    projected_before = instances.pred_proj_xy.clone()
    boxer = _boxer_prediction()
    adapter = _SelectiveFakeAdapter(_config(tmp_path), device="cpu")

    result = _apply(adapter, instances)

    assert protected_proposal_hashes(result) == protected_before
    torch.testing.assert_close(
        result.pred_proj_xy, projected_before, rtol=0.0, atol=0.0
    )
    assert len(result) == ROW_COUNT
    torch.testing.assert_close(
        result.pred_boxes_3d.tensor[:2],
        boxer["xyz_dims"][:2],
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        result.pred_boxes_3d.tensor[2:],
        cutr_xyz[2:],
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        result.pred_boxes_3d.R[:2],
        boxer["rotations"][:2],
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        result.pred_boxes_3d.R[2:],
        cutr_rotations[2:],
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        result.boxer_aleatoric_confidence,
        boxer["confidence"],
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        result.boxer_aleatoric_logvar,
        boxer["logvar"].reshape(-1),
        rtol=0.0,
        atol=0.0,
    )
    assert result.boxer_geometry_applied.tolist() == (
        [True, True] + [False] * 6
    )

    row = _diagnostic(tmp_path)
    assert row["gate_accepted"] == [True, True] + [False] * 6
    assert row["eligible_count"] == row["applied_count"] == 2
    assert row["fallback_count"] == 6
    assert row["gate_rejection_counts"]["center_shift"] == 1
    assert row["gate_rejection_counts"]["volume_low"] == 1
    assert row["gate_rejection_counts"]["volume_high"] == 1
    assert row["gate_rejection_counts"]["boxer_invalid"] == 2
    assert row["gate_rejection_counts"]["cutr_invalid"] == 1
    assert row["center_shift_m"][5] is None
    assert row["actual_xyz_dims_camera"] == row["selective_xyz_dims_camera"]
    assert "evaluated/eligible/fallback=8/2/6" in adapter.summary()


def test_observer_computes_gate_but_keeps_all_geometry_bit_exact(tmp_path):
    instances = _instances()
    geometry_before = geometry_hash(instances)
    xyz_before = instances.pred_boxes_3d.tensor.clone()
    rotations_before = instances.pred_boxes_3d.R.clone()
    adapter = _SelectiveFakeAdapter(
        _config(tmp_path, mode="observer"), device="cpu"
    )

    result = _apply(adapter, instances, scene="scene_observer")

    assert geometry_hash(result) == geometry_before
    torch.testing.assert_close(
        result.pred_boxes_3d.tensor, xyz_before, rtol=0.0, atol=0.0
    )
    torch.testing.assert_close(
        result.pred_boxes_3d.R, rotations_before, rtol=0.0, atol=0.0
    )
    row = _diagnostic(tmp_path, scene="scene_observer")
    assert row["eligible_count"] == 2
    assert row["applied_count"] == 0
    assert row["fallback_count"] == 6
    assert row["actual_geometry_sha256"] == geometry_before
    assert not result.boxer_geometry_applied.any()


def test_empty_selective_attempt_does_not_call_model(tmp_path):
    instances = _instances()[torch.zeros(ROW_COUNT, dtype=torch.bool)]
    adapter = _SelectiveFakeAdapter(_config(tmp_path), device="cpu")

    result = _apply(adapter, instances, scene="scene_empty")

    assert len(result) == 0
    assert result.has("boxer_aleatoric_confidence")
    assert result.has("boxer_aleatoric_logvar")
    assert result.has("boxer_geometry_applied")
    assert result.boxer_aleatoric_confidence.shape == (0,)
    assert result.boxer_aleatoric_logvar.shape == (0,)
    assert result.boxer_geometry_applied.shape == (0,)
    assert result.boxer_geometry_applied.dtype == torch.bool
    row = _diagnostic(tmp_path, scene="scene_empty")
    assert row["selective_gate_enabled"] is True
    assert row["eligible_count"] == 0
    assert row["applied_count"] == 0
    assert row["fallback_count"] == 0


def test_rotation_projection_does_not_abort_or_hide_nonfinite_row():
    rotations = torch.eye(3).repeat(2, 1, 1)
    rotations[1, 0, 0] = float("nan")

    projected = project_rotations_to_so3(rotations)

    torch.testing.assert_close(projected[0], torch.eye(3))
    assert torch.isnan(projected[1, 0, 0])


def test_selective_mapping_and_post_filter_contract(tmp_path):
    base = {
        "mode": "active",
        "apply_stage": "post_filter",
        "diagnostics_dir": str(tmp_path),
        "selective_gate": {
            "enabled": True,
            "max_center_shift_m": 0.10,
            "min_volume_ratio": 0.50,
            "max_volume_ratio": 2.00,
        },
    }
    config = BoxerLiftingConfig.from_mapping(base, code_root=str(tmp_path))
    assert config.selective_gate_enabled is True
    assert config.selective_max_center_shift_m == 0.10
    assert config.selective_min_volume_ratio == 0.50
    assert config.selective_max_volume_ratio == 2.00

    invalid = dict(base)
    invalid["apply_stage"] = "pre_filter"
    with pytest.raises(ValueError, match="restricted to post_filter"):
        BoxerLiftingConfig.from_mapping(invalid, code_root=str(tmp_path))


@pytest.mark.parametrize(
    "gate, message",
    [
        ({"enabled": "yes"}, "enabled must be boolean"),
        ({"enabled": True, "max_center_shift_m": -0.1}, "non-negative"),
        ({"enabled": True, "min_volume_ratio": 0.0}, "must be positive"),
        (
            {
                "enabled": True,
                "min_volume_ratio": 2.0,
                "max_volume_ratio": 1.0,
            },
            "greater than or equal",
        ),
        ({"enabled": True, "max_volume_ratio": float("inf")}, "finite"),
    ],
)
def test_invalid_selective_config_is_rejected(tmp_path, gate, message):
    with pytest.raises(ValueError, match=message):
        BoxerLiftingConfig.from_mapping(
            {
                "mode": "active",
                "apply_stage": "post_filter",
                "diagnostics_dir": str(tmp_path),
                "selective_gate": gate,
            },
            code_root=str(tmp_path),
        )
