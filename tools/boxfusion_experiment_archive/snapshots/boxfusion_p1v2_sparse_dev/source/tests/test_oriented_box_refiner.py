import copy

import numpy as np
import pytest

from boxfusion.oriented_box_refiner import (
    ORIENTED_BOX_REFINER_COORDINATE_FRAME,
    ORIENTED_BOX_REFINER_FORMAT_VERSION,
    ORIENTED_BOX_REFINER_SCHEMA,
    OrientedBoxRefinerConfig,
    PointNetOrientedBoxRefiner,
    apply_local_box_residual_numpy,
    build_oriented_box_refiner,
    load_oriented_box_refiner_checkpoint,
    make_oriented_box_refiner_checkpoint,
)


def test_oriented_config_validates_fixed_schema_and_safe_defaults():
    config = OrientedBoxRefinerConfig().validated()
    assert config.point_feature_dim == 3
    assert config.quality_feature_dim == 12
    assert config.max_center_fraction == pytest.approx(0.15)
    assert 0.0 < config.default_quality_probability < 0.5

    with pytest.raises(ValueError, match="equal 3"):
        OrientedBoxRefinerConfig(point_feature_dim=4).validated()
    with pytest.raises(ValueError, match="equal 12"):
        OrientedBoxRefinerConfig(quality_feature_dim=11).validated()
    with pytest.raises(TypeError, match="integer"):
        OrientedBoxRefinerConfig(point_hidden_dim=True).validated()
    with pytest.raises(ValueError, match="positive"):
        OrientedBoxRefinerConfig(max_center_fraction=0.0).validated()
    with pytest.raises(ValueError, match=r"\(0, 0.5\)"):
        OrientedBoxRefinerConfig(
            default_quality_probability=0.5
        ).validated()


def test_apply_local_residual_is_bounded_pure_and_preserves_metadata():
    boxes = np.asarray(
        [[0.1, -0.2, 0.3, 2.0, 4.0, 6.0, 37.0]],
        dtype=np.float32,
    )
    center = np.asarray([[100.0, -100.0, 0.25]], dtype=np.float32)
    log_dimension = np.asarray([[10.0, -10.0, 0.0]], dtype=np.float32)
    boxes_before = boxes.copy()
    center_before = center.copy()
    log_before = log_dimension.copy()

    refined = apply_local_box_residual_numpy(
        boxes,
        center,
        log_dimension,
        max_center_fraction=0.1,
        max_abs_log_dimension_residual=np.log(2.0),
    )

    np.testing.assert_allclose(refined[0, :3], [0.3, -0.6, 0.55])
    np.testing.assert_allclose(refined[0, 3:6], [4.0, 2.0, 6.0])
    assert refined[0, 6] == pytest.approx(37.0)
    assert refined.dtype == np.float32
    np.testing.assert_array_equal(boxes, boxes_before)
    np.testing.assert_array_equal(center, center_before)
    np.testing.assert_array_equal(log_dimension, log_before)


def test_apply_local_residual_supports_single_integer_box():
    refined = apply_local_box_residual_numpy(
        np.asarray([0, 0, 0, 1, 2, 3]),
        np.asarray([0.1, 0.2, 0.3]),
        np.zeros(3),
    )
    assert refined.shape == (6,)
    assert refined.dtype == np.float32
    np.testing.assert_allclose(refined, [0.1, 0.2, 0.3, 1, 2, 3])


@pytest.mark.parametrize(
    "boxes, center, log_dimension, message",
    [
        (
            np.ones((2, 5)),
            np.zeros((2, 3)),
            np.zeros((2, 3)),
            "local_boxes",
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
            np.zeros((1, 3)),
            np.asarray([[np.nan, 0, 0]]),
            "finite",
        ),
    ],
)
def test_apply_local_residual_rejects_malformed_inputs(
    boxes, center, log_dimension, message
):
    with pytest.raises((TypeError, ValueError), match=message):
        apply_local_box_residual_numpy(boxes, center, log_dimension)


def test_oriented_pointnet_identity_initialization_and_default_reject():
    torch = pytest.importorskip("torch")
    config = OrientedBoxRefinerConfig(
        default_quality_probability=0.02
    )
    model = PointNetOrientedBoxRefiner(config).cpu().eval()
    points = torch.randn(2, 11, 3)
    boxes = torch.tensor(
        [[0.0, 0.0, 0.0, 1.0, 2.0, 3.0]] * 2,
        dtype=torch.float32,
    )
    quality_features = torch.full((2, 12), 0.5)
    with torch.no_grad():
        output = model(points, boxes, quality_features)

    assert set(output) == {
        "center_residual",
        "center_residual_fraction",
        "log_dimension_residual",
        "quality",
    }
    torch.testing.assert_close(
        output["center_residual"], torch.zeros(2, 3)
    )
    torch.testing.assert_close(
        output["center_residual_fraction"], torch.zeros(2, 3)
    )
    torch.testing.assert_close(
        output["log_dimension_residual"], torch.zeros(2, 3)
    )
    torch.testing.assert_close(
        output["quality"],
        torch.full((2,), 0.02),
        rtol=1e-5,
        atol=1e-6,
    )
    assert torch.all(output["quality"] < 0.5)


def test_oriented_pointnet_outputs_are_hard_bounded():
    torch = pytest.importorskip("torch")
    config = OrientedBoxRefinerConfig(
        max_center_fraction=0.1,
        max_log_dimension_residual=0.2,
    )
    model = PointNetOrientedBoxRefiner(config).cpu().eval()
    with torch.no_grad():
        model.output_layer.bias.copy_(
            torch.tensor([100, -100, 100, 100, -100, 100, 100.0])
        )
    points = torch.zeros(1, 3, 3)
    boxes = torch.tensor(
        [[0.0, 0.0, 0.0, 2.0, 4.0, 6.0]],
        dtype=torch.float32,
    )
    features = torch.zeros(1, 12)
    with torch.no_grad():
        output = model(points, boxes, features)

    center_limit = boxes[:, 3:6] * config.max_center_fraction
    assert torch.all(output["center_residual"].abs() <= center_limit)
    assert torch.all(
        output["log_dimension_residual"].abs()
        <= config.max_log_dimension_residual
    )
    assert torch.all(
        (output["quality"] >= 0.0) & (output["quality"] <= 1.0)
    )


def test_oriented_pointnet_mask_excludes_padding_from_max_and_mean():
    torch = pytest.importorskip("torch")
    torch.manual_seed(17)
    model = PointNetOrientedBoxRefiner().cpu().eval()
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
    mask = torch.tensor([[True, True, False, False, False]] * 2)
    boxes = torch.tensor(
        [[0.0, 0.0, 0.0, 1.0, 2.0, 3.0]] * 2,
        dtype=torch.float32,
    )
    features = torch.full((2, 12), 0.5)
    with torch.no_grad():
        output = model(points, boxes, features, mask)
    for value in output.values():
        torch.testing.assert_close(value[0], value[1])


def test_oriented_pointnet_strictly_validates_inputs():
    torch = pytest.importorskip("torch")
    model = PointNetOrientedBoxRefiner().cpu()
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
    with pytest.raises(TypeError, match="Boolean"):
        model(points, boxes, features, torch.ones(1, 3))
    bad_boxes = boxes.clone()
    bad_boxes[:, 3] = 0.0
    with pytest.raises(ValueError, match="positive"):
        model(points, bad_boxes, features)


def test_oriented_checkpoint_round_trip_and_builder(tmp_path):
    torch = pytest.importorskip("torch")
    config = OrientedBoxRefinerConfig(
        point_hidden_dim=16,
        point_embedding_dim=24,
        head_hidden_dim=20,
    )
    source = PointNetOrientedBoxRefiner(config)
    with torch.no_grad():
        source.output_layer.weight.fill_(0.125)
    payload = make_oriented_box_refiner_checkpoint(source)
    assert set(payload) == {
        "schema",
        "format_version",
        "coordinate_frame",
        "config",
        "state_dict",
    }
    assert payload["schema"] == ORIENTED_BOX_REFINER_SCHEMA
    assert (
        payload["format_version"]
        == ORIENTED_BOX_REFINER_FORMAT_VERSION
    )
    assert (
        payload["coordinate_frame"]
        == ORIENTED_BOX_REFINER_COORDINATE_FRAME
    )
    checkpoint = tmp_path / "b5v2.pt"
    torch.save(payload, checkpoint)

    target = PointNetOrientedBoxRefiner(config)
    load_oriented_box_refiner_checkpoint(target, checkpoint)
    for source_value, target_value in zip(
        source.state_dict().values(), target.state_dict().values()
    ):
        torch.testing.assert_close(source_value, target_value)

    loaded = build_oriented_box_refiner(
        enabled=True,
        checkpoint_path=checkpoint,
        config=config,
        device="cpu",
    )
    assert loaded is not None
    assert loaded.training is False


@pytest.mark.parametrize(
    "field, value, message",
    [
        ("schema", "wrong.schema", "schema"),
        ("format_version", 99, "format_version"),
        ("coordinate_frame", "world_aabb", "coordinate_frame"),
    ],
)
def test_oriented_checkpoint_rejects_wrong_metadata(
    tmp_path, field, value, message
):
    torch = pytest.importorskip("torch")
    model = PointNetOrientedBoxRefiner()
    payload = make_oriented_box_refiner_checkpoint(model)
    payload[field] = value
    checkpoint = tmp_path / f"wrong_{field}.pt"
    torch.save(payload, checkpoint)
    with pytest.raises(ValueError, match=message):
        load_oriented_box_refiner_checkpoint(model, checkpoint)


def test_oriented_checkpoint_rejects_extra_keys_config_and_state(
    tmp_path,
):
    torch = pytest.importorskip("torch")
    model = PointNetOrientedBoxRefiner()

    extra = make_oriented_box_refiner_checkpoint(model)
    extra["training_note"] = "not part of the runtime schema"
    extra_path = tmp_path / "extra.pt"
    torch.save(extra, extra_path)
    with pytest.raises(ValueError, match="strict schema"):
        load_oriented_box_refiner_checkpoint(model, extra_path)

    mismatched = make_oriented_box_refiner_checkpoint(model)
    mismatched["config"] = copy.copy(mismatched["config"])
    mismatched["config"]["point_hidden_dim"] += 1
    mismatch_path = tmp_path / "mismatch.pt"
    torch.save(mismatched, mismatch_path)
    with pytest.raises(ValueError, match="does not match"):
        load_oriented_box_refiner_checkpoint(model, mismatch_path)

    incomplete = make_oriented_box_refiner_checkpoint(model)
    incomplete["state_dict"] = copy.copy(incomplete["state_dict"])
    incomplete["state_dict"].pop(next(iter(incomplete["state_dict"])))
    incomplete_path = tmp_path / "incomplete.pt"
    torch.save(incomplete, incomplete_path)
    with pytest.raises(ValueError, match="incompatible"):
        load_oriented_box_refiner_checkpoint(model, incomplete_path)


def test_oriented_builder_disabled_is_dependency_and_checkpoint_free():
    missing = "/this/path/is/intentionally/missing/b5v2.pt"
    assert (
        build_oriented_box_refiner(
            enabled=False, checkpoint_path=missing
        )
        is None
    )
    assert build_oriented_box_refiner(enabled=np.bool_(False)) is None
    with pytest.raises(TypeError, match="Boolean"):
        build_oriented_box_refiner(enabled=1)
    with pytest.raises(ValueError, match="checkpoint_path"):
        build_oriented_box_refiner(enabled=True)
