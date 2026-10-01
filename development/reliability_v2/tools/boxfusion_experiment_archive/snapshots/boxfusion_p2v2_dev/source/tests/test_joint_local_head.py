from dataclasses import dataclass

import numpy as np
import pytest

from boxfusion.joint_local_head import (
    JOINT_LOCAL_HEAD_COORDINATE_FRAME,
    JOINT_LOCAL_HEAD_FORMAT_VERSION,
    JOINT_LOCAL_HEAD_INPUT_SCHEMA,
    JOINT_LOCAL_HEAD_OUTPUT_SCHEMA,
    JOINT_LOCAL_HEAD_SCHEMA,
    JOINT_QUALITY_COMPONENT_NAMES,
    JOINT_VIEW_FEATURE_DIM,
    JointLocalHeadConfig,
    MultiViewJointLocalHead,
    build_joint_local_head,
    load_joint_local_head_checkpoint,
    make_joint_local_head_checkpoint,
    prepare_joint_view_inputs,
)


@dataclass(frozen=True)
class _ViewRecord:
    points_world: np.ndarray
    quality: float
    confidence: float
    valid_depth_ratio: float
    projection_mask_iou: float
    camera_position: np.ndarray | None


def _records():
    return (
        _ViewRecord(
            points_world=np.asarray(
                [
                    [1.0, 2.0, 3.0],
                    [2.0, 2.0, 3.0],
                    [1.0, 3.0, 3.0],
                ],
                dtype=np.float32,
            ),
            quality=0.72,
            confidence=0.90,
            valid_depth_ratio=0.80,
            projection_mask_iou=1.0,
            camera_position=np.asarray([1.0, 2.0, 1.0], dtype=np.float32),
        ),
        _ViewRecord(
            points_world=np.asarray(
                [[1.0, 2.0, 3.0], [1.0, 2.0, 4.0]],
                dtype=np.float32,
            ),
            quality=0.40,
            confidence=0.80,
            valid_depth_ratio=0.50,
            projection_mask_iou=1.0,
            camera_position=None,
        ),
    )


def _torch_inputs(torch, *, batch=2, views=3, points=5):
    points_local = torch.zeros(batch, views, points, 3)
    point_mask = torch.zeros(batch, views, points, dtype=torch.bool)
    view_mask = torch.zeros(batch, views, dtype=torch.bool)
    for batch_index in range(batch):
        point_mask[batch_index, 0, :3] = True
        point_mask[batch_index, 1, :2] = True
        view_mask[batch_index, :2] = True
        points_local[batch_index, 0, :3] = torch.tensor(
            [[-0.2, 0.1, 0.0], [0.3, -0.1, 0.2], [0.0, 0.2, -0.3]]
        )
        points_local[batch_index, 1, :2] = torch.tensor(
            [[0.1, 0.0, 0.4], [-0.1, 0.3, 0.2]]
        )
    view_features = torch.zeros(batch, views, JOINT_VIEW_FEATURE_DIM)
    view_features[:, :2] = 0.5
    local_boxes = torch.tensor(
        [[0.0, 0.0, 0.0, 1.0, 2.0, 3.0]] * batch,
        dtype=torch.float32,
    )
    quality_features = torch.full((batch, 12), 0.5)
    return (
        points_local,
        point_mask,
        view_features,
        view_mask,
        local_boxes,
        quality_features,
    )


def test_joint_config_validates_fixed_schemas_and_bounds():
    config = JointLocalHeadConfig().validated()
    assert config.point_feature_dim == 3
    assert config.view_feature_dim == JOINT_VIEW_FEATURE_DIM == 9
    assert config.quality_feature_dim == 12
    assert sum(config.ranking_weights) == pytest.approx(1.0)
    assert 0.0 < config.default_improvement_probability < 0.5

    with pytest.raises(ValueError, match="point_feature_dim"):
        JointLocalHeadConfig(point_feature_dim=4).validated()
    with pytest.raises(ValueError, match="view_feature_dim"):
        JointLocalHeadConfig(view_feature_dim=8).validated()
    with pytest.raises(TypeError, match="integer"):
        JointLocalHeadConfig(point_hidden_dim=True).validated()
    with pytest.raises(ValueError, match="summing to 1"):
        JointLocalHeadConfig(
            ranking_weights=(0.1, 0.2, 0.3, 0.5)
        ).validated()
    with pytest.raises(ValueError, match="minimum_log_variance"):
        JointLocalHeadConfig(
            minimum_log_variance=2.0,
            maximum_log_variance=2.0,
        ).validated()


def test_prepare_joint_views_preserves_view_axis_and_local_frame():
    center = np.asarray([1.0, 2.0, 3.0])
    # local x=world y, local y=-world x, local z=world z
    basis = np.asarray(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
    )
    output = prepare_joint_view_inputs(
        _records(),
        frame_center=center,
        frame_basis=basis,
        max_views=3,
        points_per_view=4,
    )

    assert output.points_local.shape == (3, 4, 3)
    assert output.point_mask.shape == (3, 4)
    assert output.view_features.shape == (3, 9)
    assert output.view_mask.tolist() == [True, True, False]
    assert output.point_mask.sum(axis=1).tolist() == [3, 2, 0]
    np.testing.assert_allclose(
        output.points_local[0, :3],
        np.asarray([[0, 0, 0], [0, -1, 0], [1, 0, 0]]),
    )
    np.testing.assert_allclose(
        output.view_features[0, :6],
        [0.72, 0.90, 0.80, 1.0, 0.75, 1.0],
    )
    # Missing cameras use the neutral direction with a false validity bit.
    np.testing.assert_allclose(output.view_features[1, 5:], [0.0, 0.5, 0.5, 0.5])
    np.testing.assert_array_equal(output.points_local[2], 0.0)
    np.testing.assert_array_equal(output.view_features[2], 0.0)


def test_prepare_joint_views_is_deterministic_and_rejects_bad_frames():
    kwargs = {
        "frame_center": np.asarray([1.0, 2.0, 3.0]),
        "frame_basis": np.eye(3),
        "max_views": 2,
        "points_per_view": 2,
    }
    first = prepare_joint_view_inputs(_records(), **kwargs)
    second = prepare_joint_view_inputs(_records(), **kwargs)
    for name in ("points_local", "point_mask", "view_features", "view_mask"):
        np.testing.assert_array_equal(getattr(first, name), getattr(second, name))

    with pytest.raises(ValueError, match="orthonormal"):
        prepare_joint_view_inputs(
            _records(), **{**kwargs, "frame_basis": np.ones((3, 3))}
        )
    with pytest.raises(ValueError, match="right-handed"):
        prepare_joint_view_inputs(
            _records(),
            **{
                **kwargs,
                "frame_basis": np.diag([1.0, 1.0, -1.0]),
            },
        )
    with pytest.raises(ValueError, match="positive"):
        prepare_joint_view_inputs(_records(), **{**kwargs, "max_views": 0})


def test_joint_head_safe_initialization_is_identity_and_monotonic():
    torch = pytest.importorskip("torch")
    config = JointLocalHeadConfig(
        default_improvement_probability=0.02,
        default_iou_probability=0.10,
    )
    model = MultiViewJointLocalHead(config).cpu().eval()
    inputs = _torch_inputs(torch)
    with torch.no_grad():
        output = model(*inputs)

    assert set(output) == {
        "center_residual",
        "center_residual_fraction",
        "log_dimension_residual",
        "improvement_probability",
        "quality_components",
        "ranking_scores",
        "quality_log_variance",
        "quality_uncertainty",
        "view_attention",
    }
    torch.testing.assert_close(output["center_residual"], torch.zeros(2, 3))
    torch.testing.assert_close(
        output["center_residual_fraction"], torch.zeros(2, 3)
    )
    torch.testing.assert_close(
        output["log_dimension_residual"], torch.zeros(2, 3)
    )
    torch.testing.assert_close(
        output["improvement_probability"],
        torch.full((2,), 0.02),
        atol=1e-6,
        rtol=1e-5,
    )
    assert output["quality_components"].shape == (
        2,
        2,
        len(JOINT_QUALITY_COMPONENT_NAMES),
    )
    components = output["quality_components"]
    assert torch.all(components[..., 1] >= components[..., 2])
    assert torch.all(components[..., 2] >= components[..., 3])
    torch.testing.assert_close(
        output["view_attention"].sum(dim=1), torch.ones(2)
    )
    torch.testing.assert_close(
        output["view_attention"][:, 2], torch.zeros(2)
    )


def test_joint_head_outputs_are_hard_bounded():
    torch = pytest.importorskip("torch")
    config = JointLocalHeadConfig(
        max_center_fraction=0.1,
        max_log_dimension_residual=0.2,
    )
    model = MultiViewJointLocalHead(config).cpu().eval()
    with torch.no_grad():
        model.geometry_layer.bias.copy_(
            torch.tensor([100.0, -100.0, 100.0, 100.0, -100.0, 100.0])
        )
        model.quality_layer.bias.copy_(
            torch.tensor([100.0, 100.0, 100.0, 100.0] * 2)
        )
    inputs = _torch_inputs(torch, batch=1)
    with torch.no_grad():
        output = model(*inputs)
    dimensions = inputs[4][:, 3:6]
    assert torch.all(
        output["center_residual"].abs()
        <= dimensions * config.max_center_fraction
    )
    assert torch.all(
        output["log_dimension_residual"].abs()
        <= config.max_log_dimension_residual
    )
    for name in (
        "improvement_probability",
        "quality_components",
        "ranking_scores",
    ):
        assert torch.all((output[name] >= 0.0) & (output[name] <= 1.0))


def test_joint_head_masks_all_point_and_view_padding():
    torch = pytest.importorskip("torch")
    torch.manual_seed(19)
    model = MultiViewJointLocalHead().cpu().eval()
    with torch.no_grad():
        for layer in (
            model.geometry_layer,
            model.improvement_layer,
            model.quality_layer,
            model.log_variance_layer,
        ):
            layer.weight.normal_(0.0, 0.1)
            layer.bias.normal_(0.0, 0.1)
    first = list(_torch_inputs(torch, batch=1))
    second = [value.clone() for value in first]
    # Invalid points in otherwise valid views and the complete padded view must
    # be invisible to both point pooling and view attention.
    second[0][:, 0, 3:] = 1000.0
    second[0][:, 1, 2:] = -1000.0
    second[0][:, 2] = 500.0
    second[2][:, 2] = 1.0
    with torch.no_grad():
        output_first = model(*first)
        output_second = model(*second)
    for name in output_first:
        torch.testing.assert_close(output_first[name], output_second[name])


def test_joint_head_strictly_validates_aligned_masks_and_ranges():
    torch = pytest.importorskip("torch")
    model = MultiViewJointLocalHead().cpu()
    inputs = list(_torch_inputs(torch, batch=1))

    bad = [value.clone() for value in inputs]
    bad[3][:] = False
    with pytest.raises(ValueError, match="valid view"):
        model(*bad)

    bad = [value.clone() for value in inputs]
    bad[3][:, 1] = False
    with pytest.raises(ValueError, match="exactly equal"):
        model(*bad)

    bad = [value.clone() for value in inputs]
    bad[2][:, 0, 0] = 2.0
    with pytest.raises(ValueError, match=r"\[0,1\]"):
        model(*bad)

    bad = [value.clone() for value in inputs]
    bad[4][:, 3] = 0.0
    with pytest.raises(ValueError, match="positive"):
        model(*bad)

    bad = [value.clone() for value in inputs]
    bad[5] = bad[5].double()
    with pytest.raises(ValueError, match="one dtype"):
        model(*bad)


def test_joint_checkpoint_round_trip_strict_schema_and_builder(tmp_path):
    torch = pytest.importorskip("torch")
    config = JointLocalHeadConfig(
        point_hidden_dim=16,
        point_embedding_dim=24,
        view_embedding_dim=20,
        head_hidden_dim=18,
    )
    source = MultiViewJointLocalHead(config)
    with torch.no_grad():
        source.geometry_layer.weight.fill_(0.125)
    payload = make_joint_local_head_checkpoint(
        source, metadata={"train_split": "scannet_train"}
    )
    assert set(payload) == {
        "schema",
        "format_version",
        "coordinate_frame",
        "input_schema",
        "output_schema",
        "config",
        "state_dict",
        "metadata",
    }
    assert payload["schema"] == JOINT_LOCAL_HEAD_SCHEMA
    assert payload["format_version"] == JOINT_LOCAL_HEAD_FORMAT_VERSION
    assert payload["coordinate_frame"] == JOINT_LOCAL_HEAD_COORDINATE_FRAME
    assert payload["input_schema"] == JOINT_LOCAL_HEAD_INPUT_SCHEMA
    assert payload["output_schema"] == JOINT_LOCAL_HEAD_OUTPUT_SCHEMA

    path = tmp_path / "joint.pt"
    torch.save(payload, path)
    target = MultiViewJointLocalHead(config)
    metadata = load_joint_local_head_checkpoint(target, path)
    assert metadata == {"train_split": "scannet_train"}
    torch.testing.assert_close(
        target.geometry_layer.weight, source.geometry_layer.weight
    )

    loaded = build_joint_local_head(
        enabled=True,
        checkpoint_path=path,
        config=config,
        device="cpu",
    )
    assert isinstance(loaded, MultiViewJointLocalHead)
    assert loaded.training is False
    assert build_joint_local_head(
        enabled=False,
        checkpoint_path=None,
    ) is None

    with pytest.raises(ValueError, match="checkpoint_path"):
        build_joint_local_head(enabled=True, checkpoint_path=None)
    with pytest.raises(ValueError, match="architecture"):
        load_joint_local_head_checkpoint(
            MultiViewJointLocalHead(
                JointLocalHeadConfig(point_hidden_dim=17)
            ),
            path,
        )

