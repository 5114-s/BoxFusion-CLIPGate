import json

import numpy as np
import pytest

from boxfusion.box_refiner import (
    BoxRefinerConfig,
    PointNetBoxRefiner,
    load_box_refiner_checkpoint,
)
from boxfusion.quality_score import (
    QUALITY_FEATURE_DIM,
    QUALITY_FEATURE_NAMES,
    LinearQualityScorer,
    load_quality_scorer,
)
from tools.train_box_refiner import (
    box_refiner_loss,
    deterministic_split as box_split,
    load_box_refiner_dataset,
    main as box_main,
    train_box_refiner,
)
from tools.train_quality_calibrator import (
    deterministic_split as quality_split,
    fit_linear_quality_scorer,
    load_quality_dataset,
    main as quality_main,
    train_quality_calibrator,
)


def _write_box_dataset(path, *, sample_count=8, point_count=12):
    rng = np.random.default_rng(11)
    boxes = np.empty((sample_count, 6), dtype=np.float32)
    boxes[:, :3] = rng.uniform(-1.0, 1.0, (sample_count, 3))
    boxes[:, 3:6] = rng.uniform(0.5, 2.0, (sample_count, 3))
    targets = boxes.copy()
    targets[:, :3] += 0.04 * boxes[:, 3:6]
    targets[:, 3:6] *= np.asarray([1.05, 0.95, 1.02])
    points = (
        targets[:, None, :3]
        + rng.uniform(
            -0.5, 0.5, (sample_count, point_count, 3)
        ).astype(np.float32)
        * targets[:, None, 3:6]
    ).astype(np.float32)
    point_mask = np.ones((sample_count, point_count), dtype=np.bool_)
    point_mask[::2, -2:] = False
    features = rng.uniform(
        0.0, 1.0, (sample_count, QUALITY_FEATURE_DIM)
    ).astype(np.float32)
    target_iou = np.linspace(0.2, 0.9, sample_count, dtype=np.float32)
    np.savez(
        path,
        points=points,
        point_mask=point_mask,
        boxes=boxes,
        quality_features=features,
        target_boxes=targets,
        target_iou=target_iou,
        feature_names=np.asarray(QUALITY_FEATURE_NAMES),
    )
    return boxes, targets


def _write_quality_dataset(path, *, target_key="target_iou"):
    rng = np.random.default_rng(19)
    features = rng.uniform(0.0, 1.0, (80, QUALITY_FEATURE_DIM)).astype(
        np.float32
    )
    if target_key == "target_binary":
        targets = (features[:, 0] + features[:, 3] > 1.0).astype(
            np.float32
        )
    else:
        targets = np.clip(
            0.65 * features[:, 0] + 0.35 * features[:, 3],
            0.0,
            1.0,
        ).astype(np.float32)
    arrays = {
        "quality_features": features,
        target_key: targets,
        "feature_names": np.asarray(QUALITY_FEATURE_NAMES),
    }
    np.savez(path, **arrays)
    return features, targets


def test_box_dataset_validation_and_deterministic_split(tmp_path):
    path = tmp_path / "box.npz"
    _write_box_dataset(path)
    data = load_box_refiner_dataset(path)
    assert data.points.shape == (8, 12, 3)
    assert data.quality_features.shape == (8, QUALITY_FEATURE_DIM)
    assert data.point_mask.dtype == np.bool_
    np.testing.assert_allclose(data.target_iou, np.linspace(0.2, 0.9, 8))

    first = box_split(8, 0.25, 7)
    second = box_split(8, 0.25, 7)
    np.testing.assert_array_equal(first[0], second[0])
    np.testing.assert_array_equal(first[1], second[1])
    assert len(first[0]) == 6
    assert len(first[1]) == 2
    assert not set(first[0]) & set(first[1])
    with pytest.raises(ValueError, match="strictly"):
        box_split(8, 0.0, 7)


@pytest.mark.parametrize(
    "mutator, message",
    [
        (
            lambda arrays: arrays.update(
                points=np.empty((0, 4, 3), dtype=np.float32),
                point_mask=np.empty((0, 4), dtype=np.bool_),
                boxes=np.empty((0, 6), dtype=np.float32),
                quality_features=np.empty(
                    (0, QUALITY_FEATURE_DIM), dtype=np.float32
                ),
                target_boxes=np.empty((0, 6), dtype=np.float32),
                target_iou=np.empty(0, dtype=np.float32),
            ),
            "at least one sample",
        ),
        (
            lambda arrays: arrays["points"].__setitem__((0, 0, 0), np.nan),
            "finite",
        ),
        (
            lambda arrays: arrays.update(
                point_mask=arrays["point_mask"].astype(np.uint8)
            ),
            "Boolean",
        ),
        (
            lambda arrays: arrays["quality_features"].__setitem__((0, 0), 2.0),
            r"\[0, 1\]",
        ),
        (
            lambda arrays: arrays.update(
                feature_names=np.asarray(tuple(reversed(QUALITY_FEATURE_NAMES)))
            ),
            "schema/order",
        ),
    ],
)
def test_box_dataset_rejects_empty_nan_shape_dtype_and_schema(
    tmp_path, mutator, message
):
    valid = tmp_path / "valid.npz"
    _write_box_dataset(valid)
    with np.load(valid, allow_pickle=False) as archive:
        arrays = {key: archive[key].copy() for key in archive.files}
    mutator(arrays)
    malformed = tmp_path / "malformed.npz"
    np.savez(malformed, **arrays)
    with pytest.raises((TypeError, ValueError), match=message):
        load_box_refiner_dataset(malformed)


def test_box_refiner_loss_has_all_targets_and_backpropagates():
    torch = pytest.importorskip("torch")
    torch.manual_seed(3)
    config = BoxRefinerConfig(
        point_hidden_dim=8,
        point_embedding_dim=8,
        head_hidden_dim=8,
    )
    model = PointNetBoxRefiner(config)
    points = torch.randn(3, 7, 3)
    boxes = torch.tensor(
        [[0.0, 0.0, 0.0, 1.0, 1.5, 2.0]] * 3
    )
    targets = boxes.clone()
    targets[:, 0] += 0.1
    targets[:, 3] *= 1.1
    features = torch.full((3, QUALITY_FEATURE_DIM), 0.5)
    output = model(points, boxes, features)
    quality_target = torch.tensor([0.2, 0.5, 0.9])
    loss, metrics = box_refiner_loss(
        output, boxes, targets, quality_target
    )
    assert set(metrics) == {
        "loss",
        "center_loss",
        "dimension_loss",
        "iou_loss",
        "quality_loss",
        "mean_iou",
    }
    assert torch.isfinite(loss)
    loss.backward()
    assert model.output_layer.weight.grad is not None
    assert torch.isfinite(model.output_layer.weight.grad).all()


def test_box_training_writes_strict_runtime_checkpoint(tmp_path):
    torch = pytest.importorskip("torch")
    dataset = tmp_path / "box.npz"
    checkpoint = tmp_path / "refiner.pt"
    _write_box_dataset(dataset, sample_count=10, point_count=8)
    config = BoxRefinerConfig(
        point_hidden_dim=8,
        point_embedding_dim=8,
        head_hidden_dim=8,
    )
    result = train_box_refiner(
        dataset,
        checkpoint,
        config=config,
        epochs=2,
        batch_size=4,
        validation_fraction=0.2,
        seed=23,
    )
    assert checkpoint.is_file()
    assert result["train_samples"] == 8
    assert result["validation_samples"] == 2
    assert np.isfinite(result["validation"]["loss"])

    try:
        raw = torch.load(checkpoint, map_location="cpu", weights_only=True)
    except TypeError:
        raw = torch.load(checkpoint, map_location="cpu")
    assert set(raw) == {"state_dict", "config"}
    assert raw["config"] == config.architecture_dict()
    target = PointNetBoxRefiner(config)
    assert load_box_refiner_checkpoint(target, checkpoint) is target


def test_quality_dataset_accepts_iou_and_binary_targets(tmp_path):
    iou_path = tmp_path / "iou.npz"
    _write_quality_dataset(iou_path, target_key="target_iou")
    iou = load_quality_dataset(iou_path)
    assert iou.target_kind == "iou"
    assert iou.features.shape == (80, QUALITY_FEATURE_DIM)

    binary_path = tmp_path / "binary.npz"
    _write_quality_dataset(binary_path, target_key="target_binary")
    binary = load_quality_dataset(binary_path)
    assert binary.target_kind == "binary"
    assert set(np.unique(binary.targets)) <= {0.0, 1.0}
    with pytest.raises(ValueError, match="incompatible"):
        load_quality_dataset(binary_path, target_kind="iou")

    first = quality_split(80, 0.2, 5)
    second = quality_split(80, 0.2, 5)
    np.testing.assert_array_equal(first[0], second[0])
    np.testing.assert_array_equal(first[1], second[1])


def test_refiner_builder_archive_can_train_both_heads(tmp_path):
    path = tmp_path / "shared_refiner_dataset.npz"
    _write_box_dataset(path)
    refiner_data = load_box_refiner_dataset(path)
    quality_data = load_quality_dataset(path)
    assert refiner_data.sample_count == quality_data.sample_count == 8
    np.testing.assert_array_equal(
        refiner_data.quality_features, quality_data.features
    )
    np.testing.assert_array_equal(
        refiner_data.target_iou, quality_data.targets
    )


@pytest.mark.parametrize(
    "arrays, message",
    [
        (
            {
                "quality_features": np.empty(
                    (0, QUALITY_FEATURE_DIM), dtype=np.float32
                ),
                "target_iou": np.empty(0, dtype=np.float32),
            },
            "at least one sample",
        ),
        (
            {
                "quality_features": np.zeros(
                    (3, QUALITY_FEATURE_DIM), dtype=np.float32
                ),
                "target_iou": np.asarray([0.0, np.nan, 1.0]),
            },
            "finite",
        ),
        (
            {
                "quality_features": np.zeros(
                    (3, QUALITY_FEATURE_DIM - 1), dtype=np.float32
                ),
                "target_iou": np.zeros(3, dtype=np.float32),
            },
            "shape",
        ),
        (
            {
                "quality_features": np.zeros(
                    (3, QUALITY_FEATURE_DIM), dtype=np.float32
                ),
                "target_binary": np.asarray([0.0, 0.5, 1.0]),
            },
            "only 0 and 1",
        ),
    ],
)
def test_quality_dataset_rejects_empty_nan_shape_and_bad_binary(
    tmp_path, arrays, message
):
    path = tmp_path / "bad.npz"
    arrays["feature_names"] = np.asarray(QUALITY_FEATURE_NAMES)
    np.savez(path, **arrays)
    with pytest.raises((TypeError, ValueError), match=message):
        load_quality_dataset(path, require_two_samples=False)


def test_quality_fit_and_checkpoint_are_loader_compatible(tmp_path):
    dataset = tmp_path / "quality.npz"
    output = tmp_path / "linear.npz"
    features, targets = _write_quality_dataset(
        dataset, target_key="target_binary"
    )
    initial_bce = np.log(2.0)
    result = train_quality_calibrator(
        dataset,
        output,
        epochs=300,
        learning_rate=0.05,
        l2_weight=1e-4,
        validation_fraction=0.2,
        seed=29,
    )
    assert result["target_kind"] == "binary"
    assert result["train"]["bce"] < initial_bce
    assert output.is_file()
    with np.load(output, allow_pickle=False) as archive:
        assert set(archive.files) == {"feature_names", "weight", "bias"}
    scorer = load_quality_scorer(output, method="linear")
    assert isinstance(scorer, LinearQualityScorer)
    scores = scorer(features)
    assert scores[targets == 1].mean() > scores[targets == 0].mean()


def test_quality_fit_is_deterministic_and_rejects_invalid_values():
    features = np.zeros((4, QUALITY_FEATURE_DIM), dtype=np.float64)
    features[:, 0] = np.asarray([0.0, 0.25, 0.75, 1.0])
    targets = np.asarray([0.0, 0.0, 1.0, 1.0])
    first = fit_linear_quality_scorer(
        features, targets, epochs=20, learning_rate=0.1
    )
    second = fit_linear_quality_scorer(
        features, targets, epochs=20, learning_rate=0.1
    )
    np.testing.assert_array_equal(first[0], second[0])
    assert first[1] == second[1]
    with pytest.raises(ValueError, match="positive"):
        fit_linear_quality_scorer(
            features, targets, epochs=0, learning_rate=0.1
        )


def test_cli_reports_validation_errors_and_emits_json(tmp_path, capsys):
    with pytest.raises(SystemExit) as error:
        quality_main(
            [
                "--input",
                str(tmp_path / "missing.npz"),
                "--output",
                str(tmp_path / "out.npz"),
            ]
        )
    assert error.value.code == 2
    assert "not found" in capsys.readouterr().err

    dataset = tmp_path / "box.npz"
    output = tmp_path / "box.pt"
    _write_box_dataset(dataset)
    assert (
        box_main(
            [
                "--input",
                str(dataset),
                "--output",
                str(output),
                "--epochs",
                "1",
                "--batch-size",
                "8",
                "--point-hidden-dim",
                "8",
                "--point-embedding-dim",
                "8",
                "--head-hidden-dim",
                "8",
            ]
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["output"] == str(output)
