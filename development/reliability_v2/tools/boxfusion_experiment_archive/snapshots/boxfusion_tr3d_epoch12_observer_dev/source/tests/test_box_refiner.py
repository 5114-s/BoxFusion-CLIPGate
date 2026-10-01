import copy

import numpy as np
import pytest

from boxfusion.box_refiner import (
    BoxRefinerConfig,
    PointNetBoxRefiner,
    apply_box_residual_numpy,
    build_box_refiner,
    load_box_refiner_checkpoint,
)


def test_box_refiner_config_rejects_invalid_bounds_and_dimensions():
    with pytest.raises(ValueError, match="at least xyz"):
        BoxRefinerConfig(point_feature_dim=2).validated()
    with pytest.raises(ValueError, match="positive"):
        BoxRefinerConfig(point_hidden_dim=0).validated()
    with pytest.raises(ValueError, match="finite"):
        BoxRefinerConfig(max_center_fraction=np.nan).validated()
    with pytest.raises(TypeError, match="integer"):
        BoxRefinerConfig(quality_feature_dim=True).validated()


def test_apply_box_residual_is_bounded_and_does_not_mutate_inputs():
    boxes = np.asarray(
        [[1.0, 2.0, 3.0, 2.0, 4.0, 6.0, 0.75]],
        dtype=np.float32,
    )
    center = np.asarray([[100.0, -100.0, 0.25]], dtype=np.float32)
    log_dims = np.asarray([[10.0, -10.0, 0.0]], dtype=np.float32)
    original_boxes = boxes.copy()
    original_center = center.copy()
    original_log_dims = log_dims.copy()

    refined = apply_box_residual_numpy(
        boxes,
        center,
        log_dims,
        max_center_fraction=0.25,
        max_abs_log_dimension_residual=np.log(2.0),
    )

    np.testing.assert_allclose(refined[0, :3], [1.5, 1.0, 3.25])
    np.testing.assert_allclose(refined[0, 3:6], [4.0, 2.0, 6.0])
    assert refined[0, 6] == pytest.approx(0.75)
    np.testing.assert_array_equal(boxes, original_boxes)
    np.testing.assert_array_equal(center, original_center)
    np.testing.assert_array_equal(log_dims, original_log_dims)


def test_apply_box_residual_supports_one_box_and_preserves_float_dtype():
    box = np.asarray([0, 0, 0, 1, 2, 3], dtype=np.float32)
    refined = apply_box_residual_numpy(
        box,
        np.asarray([0.1, 0.2, 0.3]),
        np.zeros(3),
    )
    assert refined.shape == (6,)
    assert refined.dtype == np.float32
    np.testing.assert_allclose(refined[:3], [0.1, 0.2, 0.3])


@pytest.mark.parametrize(
    "boxes, center, log_dims, message",
    [
        (
            np.ones((2, 5)),
            np.zeros((2, 3)),
            np.zeros((2, 3)),
            "boxes",
        ),
        (
            np.asarray([[0, 0, 0, 1, 0, 1]], dtype=float),
            np.zeros((1, 3)),
            np.zeros((1, 3)),
            "dimensions",
        ),
        (
            np.ones((2, 6)),
            np.zeros((1, 3)),
            np.zeros((2, 3)),
            "center_residual",
        ),
        (
            np.ones((1, 6)),
            np.asarray([[np.nan, 0, 0]]),
            np.zeros((1, 3)),
            "finite",
        ),
    ],
)
def test_apply_box_residual_rejects_malformed_inputs(
    boxes, center, log_dims, message
):
    with pytest.raises((TypeError, ValueError), match=message):
        apply_box_residual_numpy(boxes, center, log_dims)


def test_disabled_refiner_does_not_require_or_touch_checkpoint():
    missing = "/this/path/is/intentionally/missing/refiner.pt"
    assert build_box_refiner(enabled=False, checkpoint_path=missing) is None
    assert build_box_refiner(enabled=np.bool_(False)) is None


def test_enabled_refiner_requires_checkpoint():
    with pytest.raises(ValueError, match="checkpoint_path"):
        build_box_refiner(enabled=True)
    with pytest.raises(TypeError, match="Boolean"):
        build_box_refiner(enabled=1)


def test_pointnet_refiner_cpu_shapes_ranges_and_identity_initialization():
    torch = pytest.importorskip("torch")
    config = BoxRefinerConfig(
        max_center_fraction=0.2,
        max_log_dimension_residual=np.log(1.25),
    )
    model = PointNetBoxRefiner(config).cpu().eval()
    points = torch.randn(2, 9, 3)
    boxes = torch.tensor(
        [[0.0, 0.0, 0.0, 1.0, 2.0, 3.0]]
        * 2,
        dtype=torch.float32,
    )
    features = torch.full(
        (2, config.quality_feature_dim), 0.5, dtype=torch.float32
    )
    with torch.no_grad():
        output = model(points, boxes, features)

    assert set(output) == {
        "center_residual",
        "center_residual_fraction",
        "log_dimension_residual",
        "quality",
    }
    assert output["center_residual"].shape == (2, 3)
    assert output["log_dimension_residual"].shape == (2, 3)
    assert output["quality"].shape == (2,)
    torch.testing.assert_close(
        output["center_residual"], torch.zeros(2, 3)
    )
    torch.testing.assert_close(
        output["log_dimension_residual"], torch.zeros(2, 3)
    )
    torch.testing.assert_close(
        output["quality"], torch.full((2,), 0.5)
    )


def test_pointnet_outputs_remain_bounded_for_extreme_logits():
    torch = pytest.importorskip("torch")
    config = BoxRefinerConfig(
        max_center_fraction=0.1,
        max_log_dimension_residual=0.2,
    )
    model = PointNetBoxRefiner(config).cpu().eval()
    with torch.no_grad():
        model.output_layer.bias.copy_(
            torch.tensor([100, -100, 100, 100, -100, 100, 100.0])
        )
    points = torch.zeros(1, 3, 3)
    boxes = torch.tensor(
        [[0.0, 0.0, 0.0, 2.0, 4.0, 6.0]]
    )
    features = torch.zeros(1, config.quality_feature_dim)
    with torch.no_grad():
        output = model(points, boxes, features)

    limits = boxes[:, 3:6] * config.max_center_fraction
    assert torch.all(output["center_residual"].abs() <= limits)
    assert torch.all(
        output["log_dimension_residual"].abs()
        <= config.max_log_dimension_residual
    )
    assert torch.all(
        (output["quality"] >= 0.0) & (output["quality"] <= 1.0)
    )


def test_masked_padding_does_not_change_pointnet_output():
    torch = pytest.importorskip("torch")
    torch.manual_seed(7)
    model = PointNetBoxRefiner().cpu().eval()
    with torch.no_grad():
        model.output_layer.weight.normal_(0.0, 0.1)
        model.output_layer.bias.normal_(0.0, 0.1)

    shared = torch.tensor(
        [[0.1, 0.2, 0.3], [-0.3, 0.2, 0.1]], dtype=torch.float32
    )
    points = torch.stack(
        [
            torch.cat((shared, torch.zeros(3, 3))),
            torch.cat((shared, torch.full((3, 3), 1000.0))),
        ]
    )
    mask = torch.tensor(
        [[True, True, False, False, False]] * 2
    )
    boxes = torch.tensor(
        [[0.0, 0.0, 0.0, 1.0, 1.0, 1.0]] * 2
    )
    features = torch.full((2, 12), 0.5)
    with torch.no_grad():
        output = model(points, boxes, features, mask)
    for value in output.values():
        torch.testing.assert_close(value[0], value[1])


def test_pointnet_strictly_validates_features_masks_and_boxes():
    torch = pytest.importorskip("torch")
    model = PointNetBoxRefiner().cpu()
    points = torch.zeros(1, 3, 3)
    boxes = torch.tensor([[0.0, 0.0, 0.0, 1.0, 1.0, 1.0]])
    features = torch.zeros(1, 12)

    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        model(points, boxes, features + 2.0)
    with pytest.raises(ValueError, match="valid point"):
        model(
            points,
            boxes,
            features,
            torch.zeros(1, 3, dtype=torch.bool),
        )
    bad_boxes = boxes.clone()
    bad_boxes[:, 3] = 0.0
    with pytest.raises(ValueError, match="positive"):
        model(points, bad_boxes, features)
    with pytest.raises(TypeError, match="Boolean"):
        model(points, boxes, features, torch.ones(1, 3))


def test_checkpoint_round_trip_and_production_builder(tmp_path):
    torch = pytest.importorskip("torch")
    config = BoxRefinerConfig(point_hidden_dim=16, point_embedding_dim=24)
    source = PointNetBoxRefiner(config)
    with torch.no_grad():
        source.output_layer.bias.fill_(0.125)
    checkpoint = tmp_path / "refiner.pt"
    torch.save(
        {
            "state_dict": source.state_dict(),
            "config": config.architecture_dict(),
        },
        checkpoint,
    )

    target = PointNetBoxRefiner(config)
    load_box_refiner_checkpoint(target, checkpoint)
    for source_value, target_value in zip(
        source.state_dict().values(), target.state_dict().values()
    ):
        torch.testing.assert_close(source_value, target_value)

    loaded = build_box_refiner(
        enabled=True,
        checkpoint_path=checkpoint,
        config=config,
        device="cpu",
    )
    assert loaded is not None
    assert loaded.training is False


def test_checkpoint_rejects_missing_parameters_and_config_mismatch(
    tmp_path,
):
    torch = pytest.importorskip("torch")
    model = PointNetBoxRefiner()
    incomplete_state = copy.copy(model.state_dict())
    incomplete_state.pop(next(iter(incomplete_state)))
    incomplete_path = tmp_path / "incomplete.pt"
    torch.save(incomplete_state, incomplete_path)
    with pytest.raises(ValueError, match="incompatible"):
        load_box_refiner_checkpoint(model, incomplete_path)

    mismatch_path = tmp_path / "mismatch.pt"
    mismatch_config = model.config.architecture_dict()
    mismatch_config["point_hidden_dim"] += 1
    torch.save(
        {
            "state_dict": model.state_dict(),
            "config": mismatch_config,
        },
        mismatch_path,
    )
    with pytest.raises(ValueError, match="does not match"):
        load_box_refiner_checkpoint(model, mismatch_path)
