from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from boxfusion import fastsam_boxer_f4_shadow as f4
from tools.run_scannet_fastsam_f4_boxer_paper100 import _normalize_hb


def _inputs(count=2):
    rgb = np.zeros((480, 640, 3), dtype=np.uint8)
    rgb[..., 0] = 17
    depth = np.full((480, 640), 2.0, dtype=np.float32)
    K = np.asarray(
        [[577.87, 0.0, 319.5], [0.0, 577.87, 239.5], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    pose = np.eye(4, dtype=np.float32)
    base = np.asarray(
        [[10, 20, 110, 220], [300, 100, 500, 400]], dtype=np.float32
    )
    if count <= 2:
        boxes = base[:count].copy()
    else:
        boxes = np.stack(
            [np.asarray([i, i, i + 10, i + 20], dtype=np.float32) for i in range(count)]
        )
    source_ids = tuple(
        f"scene0001_00/frame_000025/raw_{index:03d}" for index in range(count)
    )
    return rgb, depth, K, pose, boxes, source_ids


class _FakeObbs:
    def __init__(self, centers, extents, rotations, confidence):
        self.bb3_center_world = torch.as_tensor(centers, dtype=torch.float32)
        self.bb3_diagonal = torch.as_tensor(extents, dtype=torch.float32)
        self.T_world_object = SimpleNamespace(
            R=torch.as_tensor(np.asarray(rotations), dtype=torch.float32)
        )
        self.prob = torch.as_tensor(confidence, dtype=torch.float32).reshape(-1, 1)

    def __len__(self):
        return int(self.bb3_center_world.shape[0])

    def cpu(self):
        return self


class _FakeModel(torch.nn.Module):
    def __init__(self, centers, extents, rotations, confidence):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(()))
        self.hw = 960
        self.obbs = _FakeObbs(centers, extents, rotations, confidence)
        count = len(self.obbs)
        self.logvar = torch.arange(count, dtype=torch.float32).reshape(1, count, 1)
        self.raw_params = torch.arange(count * 7, dtype=torch.float32).reshape(
            1, count, 7
        )
        self.forward_calls = 0

    def forward(self, datum):
        self.forward_calls += 1
        return {
            "obbs_pr_w": [self.obbs],
            "obbs_pr_logvar": self.logvar,
            "obbs_pr_params": self.raw_params,
        }


class _FakeAdapter:
    def __init__(self, model, *, corrupt_mapping=False):
        self.model = model
        self.calls = 0
        self.corrupt_mapping = corrupt_mapping
        self.last_boxes_xyxy = None
        self.last_boxer_boxes = None

    def _make_datum(
        self,
        *,
        image,
        depth,
        boxes_xyxy,
        image_K,
        depth_K,
        camera_to_world,
        scene_id,
        frame_id,
    ):
        self.calls += 1
        boxes = boxes_xyxy.detach().cpu().float()
        boxer = torch.stack(
            (
                boxes[:, 0] * 1.5,
                boxes[:, 2] * 1.5,
                boxes[:, 1] * 2.0,
                boxes[:, 3] * 2.0,
            ),
            dim=-1,
        )
        if self.corrupt_mapping:
            boxer = boxer + 1.0
        self.last_boxes_xyxy = boxes.clone()
        self.last_boxer_boxes = boxer.clone()
        return {"bb2d": boxer}, {"boxer_boxes": boxer}


def _provider(
    centers=None,
    extents=None,
    rotations=None,
    confidence=None,
    *,
    adapter=None,
):
    centers = np.asarray(
        centers if centers is not None else [[1.0, 2.0, 3.0], [-1.0, 0.5, 4.0]],
        dtype=np.float32,
    )
    count = len(centers)
    extents = np.asarray(
        extents if extents is not None else np.tile([2.0, 4.0, 6.0], (count, 1)),
        dtype=np.float32,
    )
    rotations = np.asarray(
        rotations if rotations is not None else np.tile(np.eye(3), (count, 1, 1)),
        dtype=np.float32,
    )
    confidence = np.asarray(
        confidence if confidence is not None else np.linspace(0.01, 0.99, count),
        dtype=np.float32,
    )
    model = _FakeModel(centers, extents, rotations, confidence)
    adapter = adapter or _FakeAdapter(model)
    provider = f4.FrozenFastSAMBoxerF4Provider(
        adapter,
        device="cpu",
        precision="float32",
        model_load_ms=12.5,
        asset_validation_ms=3.25,
        frozen_receipts={
            "formal": False,
            "boxer_repository": {"commit": "fake", "clean": True},
            # Constructor must keep process-local timing out of the stable
            # cross-shard model identity even if a caller supplies it.
            "model_load_ms": 999.0,
            "asset_validation_ms": 888.0,
        },
        torch_module=torch,
    )
    return provider, adapter, model


def _infer(provider, count=2, *, pose=None):
    rgb, depth, K, default_pose, boxes, source_ids = _inputs(count)
    return provider.infer_batch(
        "scene0001_00",
        25,
        rgb,
        depth,
        K,
        default_pose if pose is None else pose,
        boxes,
        source_ids,
    )


def test_policy_is_frozen_output_inert_and_has_no_confidence_filter():
    assert dict(f4.POLICY) == {
        "input_boxes": "sealed_fastsam_tight_box_xyxy",
        "input_image_shape": (480, 640, 3),
        "boxer_image_shape": (960, 960),
        "boxer_box_convention": "xmin_xmax_ymin_ymax",
        "frame_batch": (0, 16),
        "sdp_enabled": True,
        "sdp_samples": 10000,
        "seed": 0,
        "confidence_filter": False,
        "training": False,
        "online_learning": False,
        "ground_truth": False,
        "prediction_access": False,
        "evaluator_access": False,
        "history": False,
        "birth": False,
        "native_output_mutation": False,
    }
    with pytest.raises(TypeError):
        f4.POLICY["confidence_filter"] = True


def test_one_frame_batch_preserves_rows_source_ids_and_exact_box_mapping():
    provider, adapter, model = _provider(confidence=[0.0, 1.0])
    rgb, depth, K, pose, boxes, source_ids = _inputs(2)
    originals = tuple(value.copy() for value in (rgb, depth, K, pose, boxes))
    result = provider.infer_batch(
        "scene0001_00", 25, rgb, depth, K, pose, boxes, source_ids
    )

    assert adapter.calls == 1
    assert model.forward_calls == 1
    assert provider.model_forward_count == 1
    assert [row.source_id for row in result.rows] == list(source_ids)
    assert [row.row_index for row in result.rows] == [0, 1]
    np.testing.assert_array_equal(
        adapter.last_boxer_boxes.numpy(),
        [[15.0, 165.0, 40.0, 440.0], [450.0, 750.0, 200.0, 800.0]],
    )
    assert result.rows[0].confidence == pytest.approx(0.0)
    assert result.rows[0].valid
    assert result.rows[1].confidence == pytest.approx(1.0)
    assert result.rows[1].valid
    assert result.diagnostics.source_count == 2
    assert result.diagnostics.valid_count == 2
    assert result.diagnostics.invalid_count == 0
    assert result.diagnostics.model_forward_calls == 1
    assert result.diagnostics.model_load_ms == pytest.approx(12.5)
    assert result.diagnostics.asset_validation_ms == pytest.approx(3.25)
    assert result.diagnostics.total_ms >= result.diagnostics.forward_ms
    for value, original in zip((rgb, depth, K, pose, boxes), originals):
        np.testing.assert_array_equal(value, original)


def test_model_is_explicitly_eval_and_all_parameters_are_frozen():
    provider, _, model = _provider()
    assert not model.training
    assert all(not parameter.requires_grad for parameter in model.parameters())
    assert provider.frozen_receipts["model_eval"] is True
    assert provider.frozen_receipts["model_parameters_frozen"] is True
    assert "model_load_ms" not in provider.frozen_receipts
    assert "asset_validation_ms" not in provider.frozen_receipts
    with pytest.raises(TypeError):
        provider.frozen_receipts["model_eval"] = False
    with pytest.raises(TypeError):
        provider.frozen_receipts["boxer_repository"]["clean"] = False

    model.weight.requires_grad_(True)
    with pytest.raises(f4.F4ContractError, match="trainable"):
        _infer(provider)


def test_world_obb_corners_preserve_orientation_not_an_expanded_world_aabb():
    rotation = np.asarray(
        [[[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]],
        dtype=np.float32,
    )
    provider, _, _ = _provider(
        centers=[[10.0, 20.0, 3.0]],
        extents=[[2.0, 4.0, 6.0]],
        rotations=rotation,
        confidence=[0.2],
    )
    row = _infer(provider, 1).rows[0]
    expected = row.world_center + (f4.CORNER_SIGNS * [1.0, 2.0, 3.0]) @ rotation[0].T
    np.testing.assert_allclose(row.world_corners, expected, atol=1e-7)
    np.testing.assert_allclose(row.local_extent, [2.0, 4.0, 6.0])
    # A 90-degree rotation swaps the world X/Y envelope but not local extent.
    np.testing.assert_allclose(np.ptp(row.world_corners, axis=0), [4.0, 2.0, 6.0])


def test_camera_depth_uses_exact_pose_while_geometry_stays_absolute_world():
    pose = np.eye(4, dtype=np.float32)
    pose[:3, 3] = [4.0, 5.0, 6.0]
    provider, _, _ = _provider(
        centers=[[4.0, 5.0, 8.5]],
        extents=[[1.0, 1.0, 1.0]],
        rotations=[np.eye(3)],
        confidence=[0.5],
    )
    row = _infer(provider, 1, pose=pose).rows[0]
    np.testing.assert_array_equal(row.world_center, [4.0, 5.0, 8.5])
    assert row.camera_depth == pytest.approx(2.5)
    assert row.valid


def test_fixed_so3_projection_accepts_finite_drift_and_seals_correction():
    drift = np.asarray(
        [[[1.0, 0.02, 0.0], [-0.01, 0.99, 0.0], [0.0, 0.0, 1.01]]],
        dtype=np.float32,
    )
    provider, _, _ = _provider(
        centers=[[0.0, 0.0, 2.0]],
        extents=[[1.0, 2.0, 3.0]],
        rotations=drift,
        confidence=[0.1],
    )
    row = _infer(provider, 1).rows[0]
    assert row.valid
    np.testing.assert_allclose(row.world_rotation.T @ row.world_rotation, np.eye(3), atol=1e-6)
    assert np.linalg.det(row.world_rotation) == pytest.approx(1.0, abs=1e-6)
    assert row.validity.rotation_correction_max_abs > 0.0


@pytest.mark.parametrize(
    "centers,extents,rotations,expected_reason",
    [
        ([[np.nan, 0.0, 2.0]], [[1.0, 1.0, 1.0]], [np.eye(3)], "nonfinite_center"),
        ([[0.0, 0.0, 2.0]], [[1.0, -1.0, 1.0]], [np.eye(3)], "nonpositive_extent"),
        ([[0.0, 0.0, -1.0]], [[1.0, 1.0, 1.0]], [np.eye(3)], "not_in_front"),
        (
            [[0.0, 0.0, 2.0]],
            [[1.0, 1.0, 1.0]],
            [np.asarray([[np.nan, 0, 0], [0, 1, 0], [0, 0, 1]])],
            "nonfinite_rotation",
        ),
    ],
)
def test_invalid_hb_abstains_without_dropping_or_confidence_filtering(
    centers, extents, rotations, expected_reason
):
    provider, _, _ = _provider(
        centers=centers,
        extents=extents,
        rotations=rotations,
        confidence=[0.999],
    )
    result = _infer(provider, 1)
    assert len(result.rows) == 1
    assert not result.rows[0].valid
    assert expected_reason in result.rows[0].validity.reasons
    assert result.diagnostics.valid_count == 0
    assert result.diagnostics.invalid_count == 1
    # Exercise the actual scene-runner boundary. Invalid model numerics must
    # become an explicit source-preserving abstention, not non-standard NaN
    # JSON that aborts the create-only receipt.
    normalized = _normalize_hb(
        result.rows[0],
        source_id=result.rows[0].source_id,
        row_index=0,
        tight_box_xyxy=result.rows[0].input_tight_box_xyxy,
    )
    assert normalized["valid"] is False
    assert normalized["source_id"] == result.rows[0].source_id
    assert normalized["world_corners"] is None
    assert normalized["world_center"] is None
    assert normalized["local_extent"] is None
    assert normalized["world_rotation"] is None
    assert normalized["camera_depth"] is None
    # Geometry abstention does not turn the frozen model diagnostics into a
    # selector: finite values are still sealed, while non-finite values are
    # covered by the test below and become null.
    assert normalized["confidence"] == pytest.approx(0.999)
    assert normalized["logvar"] == [0.0]
    assert normalized["raw_params"] == [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    json.dumps(normalized, allow_nan=False)


def test_nonfinite_optional_model_diagnostics_are_null_without_filtering():
    provider, _, model = _provider(
        centers=[[0.0, 0.0, 2.0]],
        extents=[[1.0, 1.0, 1.0]],
        rotations=[np.eye(3)],
        confidence=[0.999],
    )
    model.obbs.prob[0, 0] = float("nan")
    model.logvar[0, 0, 0] = float("inf")
    model.raw_params[0, 0, 0] = float("nan")

    result = _infer(provider, 1)
    row = result.rows[0]
    assert row.valid is True
    assert row.confidence is None
    assert row.logvar is None
    assert row.raw_params is None
    normalized = _normalize_hb(
        row,
        source_id=row.source_id,
        row_index=0,
        tight_box_xyxy=row.input_tight_box_xyxy,
    )
    assert normalized["valid"] is True
    assert normalized["confidence"] is None
    assert normalized["logvar"] is None
    assert normalized["raw_params"] is None
    json.dumps(normalized, allow_nan=False)


def test_empty_batch_does_not_construct_datum_or_invoke_boxer():
    provider, adapter, model = _provider(
        centers=np.empty((0, 3), dtype=np.float32),
        extents=np.empty((0, 3), dtype=np.float32),
        rotations=np.empty((0, 3, 3), dtype=np.float32),
        confidence=np.empty((0,), dtype=np.float32),
    )
    result = _infer(provider, 0)
    assert result.rows == ()
    assert adapter.calls == 0
    assert model.forward_calls == 0
    assert result.diagnostics.model_forward_calls == 0
    assert result.diagnostics.total_ms == 0.0


def test_row_count_mismatch_fails_closed():
    provider, _, _ = _provider(
        centers=[[0.0, 0.0, 2.0]],
        extents=[[1.0, 1.0, 1.0]],
        rotations=[np.eye(3)],
        confidence=[0.5],
    )
    with pytest.raises(f4.F4ContractError, match="row count differs"):
        _infer(provider, 2)


def test_adapter_cannot_switch_from_frozen_tight_box_mapping():
    model = _FakeModel(
        [[0.0, 0.0, 2.0]], [[1.0, 1.0, 1.0]], [np.eye(3)], [0.5]
    )
    adapter = _FakeAdapter(model, corrupt_mapping=True)
    provider, _, _ = _provider(
        centers=[[0.0, 0.0, 2.0]],
        extents=[[1.0, 1.0, 1.0]],
        rotations=[np.eye(3)],
        confidence=[0.5],
        adapter=adapter,
    )
    with pytest.raises(f4.F4ContractError, match="tight-box mapping"):
        _infer(provider, 1)


def test_hashes_and_numeric_outputs_are_deterministic_and_deeply_readonly():
    provider, _, _ = _provider()
    first = _infer(provider)
    second = _infer(provider)
    assert first.input_sha256 == second.input_sha256
    assert first.result_sha256 == second.result_sha256
    assert [row.result_sha256 for row in first.rows] == [
        row.result_sha256 for row in second.rows
    ]
    for left, right in zip(first.rows, second.rows):
        np.testing.assert_array_equal(left.world_corners, right.world_corners)
        for array in (
            left.input_tight_box_xyxy,
            left.world_corners,
            left.world_center,
            left.local_extent,
            left.world_rotation,
            left.logvar,
            left.raw_params,
        ):
            assert not array.flags.writeable
            with pytest.raises(ValueError):
                array.flat[0] = 999


@pytest.mark.parametrize(
    "mutator,match",
    [
        (lambda values: values.__setitem__(0, np.zeros((480, 640, 3), dtype=np.float32)), "rgb"),
        (lambda values: values.__setitem__(1, np.zeros((479, 640), dtype=np.float32)), "depth"),
        (lambda values: values.__setitem__(2, np.eye(4, dtype=np.float32)), "K"),
        (lambda values: values.__setitem__(3, np.zeros((4, 4), dtype=np.float32)), "homogeneous"),
        (
            lambda values: values.__setitem__(
                4, np.asarray([[10, 20, 640, 220], [20, 20, 40, 40]], dtype=np.float32)
            ),
            "tight-box bounds",
        ),
        (
            lambda values: values.__setitem__(
                5,
                (
                    "scene0001_00/frame_000026/raw_000",
                    "scene0001_00/frame_000025/raw_001",
                ),
            ),
            "differs",
        ),
    ],
)
def test_input_contract_fails_closed(mutator, match):
    provider, _, _ = _provider()
    values = list(_inputs(2))
    mutator(values)
    with pytest.raises(ValueError, match=match):
        provider.infer_batch("scene0001_00", 25, *values)


def test_sixteen_rows_are_allowed_but_seventeen_fail_before_model_call():
    centers = np.tile([0.0, 0.0, 2.0], (16, 1))
    provider, _, model = _provider(
        centers=centers,
        extents=np.tile([1.0, 1.0, 1.0], (16, 1)),
        rotations=np.tile(np.eye(3), (16, 1, 1)),
        confidence=np.zeros(16),
    )
    assert len(_infer(provider, 16).rows) == 16
    rgb, depth, K, pose, boxes, source_ids = _inputs(17)
    with pytest.raises(ValueError, match="sixteen-source cap"):
        provider.infer_batch(
            "scene0001_00", 25, rgb, depth, K, pose, boxes, source_ids
        )
    assert model.forward_calls == 1


def test_protocol_hash_constants_are_exact_lowercase_sha256():
    assert f4.PROTOCOL_ID == "F4-FASTSAM-BOXER-GEOMETRY-SHADOW-PAPER100"
    assert f4.SCHEMA == "boxfusion.fastsam_boxer_f4_shadow.v1"
    for value in (
        f4.BOXER_CHECKPOINT_SHA256,
        f4.DINOV3_CHECKPOINT_SHA256,
        f4.BOXERNET_SOURCE_SHA256,
        f4.ADAPTER_SOURCE_SHA256,
    ):
        assert len(value) == 64
        assert value == value.lower()
        int(value, 16)
