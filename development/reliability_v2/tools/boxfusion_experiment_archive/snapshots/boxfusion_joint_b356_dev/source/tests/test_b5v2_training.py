import math
import pickle

import numpy as np
import pytest

from boxfusion.oriented_box_refiner import (
    OrientedBoxRefinerConfig,
    PointNetOrientedBoxRefiner,
    load_oriented_box_refiner_checkpoint,
)
from boxfusion.quality_score import QUALITY_FEATURE_NAMES
from tools.build_oriented_refiner_dataset import (
    DATASET_FORMAT_VERSION,
    DATASET_SCHEMA,
    BuildConfig,
    build_oriented_refiner_dataset,
    nonnegative_least_squares_3x3,
    oriented_box_frame,
)
from tools.train_oriented_box_refiner import (
    balanced_epoch_indices,
    deterministic_scene_split,
    load_oriented_refiner_dataset,
    oriented_refiner_loss,
    train_oriented_box_refiner,
)


def _rotation_z(degrees):
    angle = np.deg2rad(degrees)
    return np.asarray(
        [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )


def _corners(center, dimensions, basis):
    signs = np.asarray(
        [
            [-1.0, -1.0, -1.0],
            [1.0, -1.0, -1.0],
            [1.0, 1.0, -1.0],
            [-1.0, 1.0, -1.0],
            [-1.0, -1.0, 1.0],
            [1.0, -1.0, 1.0],
            [1.0, 1.0, 1.0],
            [-1.0, 1.0, 1.0],
        ]
    )
    return center + (signs * (0.5 * dimensions)) @ basis.T


def _write_axis_alignment(root, scene, transform):
    scene_root = root / scene
    scene_root.mkdir(parents=True)
    flattened = " ".join(str(value) for value in transform.reshape(-1))
    (scene_root / f"{scene}.txt").write_text(
        f"axisAlignment = {flattened}\n"
    )


def test_oriented_builder_projects_gt_into_local_frame_and_uses_nnls(tmp_path):
    scene = "scene0000_00"
    diagnostics_root = tmp_path / "diagnostics"
    prediction_root = tmp_path / "predictions"
    scan_root = tmp_path / "scans"
    gt_root = tmp_path / "gt"
    for root in (diagnostics_root, prediction_root, scan_root, gt_root):
        root.mkdir()
    scene_list = tmp_path / "train.txt"
    scene_list.write_text(scene + "\n")

    basis_world = _rotation_z(20.0)
    rotation = _rotation_z(30.0)
    translation = np.asarray([4.0, -2.0, 0.5])
    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = translation
    _write_axis_alignment(scan_root, scene, transform)

    positive_center = np.asarray([1.0, 2.0, 0.5])
    negative_center = np.asarray([20.0, 0.0, 0.5])
    original_dimensions = np.asarray([2.0, 1.0, 1.0])
    positive_corners = _corners(
        positive_center, original_dimensions, basis_world
    )
    negative_corners = _corners(
        negative_center, original_dimensions, basis_world
    )
    with (prediction_root / f"{scene}_boxes.pkl").open("wb") as handle:
        pickle.dump(
            [
                [
                    (0, positive_corners.astype(np.float32), 0.8),
                    (0, negative_corners.astype(np.float32), 0.7),
                ]
            ],
            handle,
        )

    aligned_basis = rotation @ basis_world
    aligned_positive_center = rotation @ positive_center + translation
    expected_center_local = np.asarray([0.2, -0.1, 0.05])
    expected_dimensions_local = np.asarray([1.8, 0.9, 1.0])
    gt_center = aligned_positive_center + aligned_basis @ expected_center_local
    gt_dimensions = np.abs(aligned_basis) @ expected_dimensions_local
    np.save(
        gt_root / f"{scene}_bbox.npy",
        np.asarray(
            [
                np.concatenate((gt_center, gt_dimensions, [3.0])),
            ],
            dtype=np.float32,
        ),
    )

    # Diagnostics deliberately reverse the result order.  Geometry points
    # differ from the fallback points to prove that the preferred source is
    # selected.
    point_count = 5
    geometry_points = np.zeros((2, point_count, 3), dtype=np.float32)
    geometry_points[0, :2] = negative_center + np.asarray(
        [[0.2, 0.0, 0.0], [-0.2, 0.0, 0.0]]
    )
    geometry_points[1, :2] = positive_center + np.asarray(
        [[0.3, 0.0, 0.0], [-0.3, 0.0, 0.0]]
    )
    point_mask = np.zeros((2, point_count), dtype=np.bool_)
    point_mask[:, :2] = True
    np.savez(
        diagnostics_root / f"{scene}_tracks.npz",
        scene_id=np.asarray(scene),
        quality_features=np.full((2, 12), 0.5, dtype=np.float32),
        quality_feature_names=np.asarray(QUALITY_FEATURE_NAMES),
        points=np.zeros_like(geometry_points),
        point_mask=point_mask,
        geometry_points=geometry_points,
        geometry_point_mask=point_mask,
        result_indices=np.asarray([1, 0], dtype=np.int64),
        track_ids=np.asarray([11, 12], dtype=np.int64),
    )

    output = tmp_path / "b5v2.npz"
    summary = build_oriented_refiner_dataset(
        BuildConfig(
            diagnostics_root=diagnostics_root,
            prediction_root=prediction_root,
            scan_root=scan_root,
            gt_root=gt_root,
            scene_list=scene_list,
            output=output,
        )
    )
    assert summary.samples == 2
    assert summary.geometry_positives == 1
    assert summary.quality_negatives == 1
    with np.load(output, allow_pickle=False) as archive:
        # Row one maps to prediction zero, the positive OBB.
        np.testing.assert_allclose(
            archive["target_center_local_unclipped"][1],
            expected_center_local,
            atol=2e-6,
        )
        np.testing.assert_allclose(
            archive["target_dimensions_local_unclipped"][1],
            expected_dimensions_local,
            atol=2e-6,
        )
        np.testing.assert_allclose(
            archive["local_boxes"][1, 3:],
            original_dimensions,
            atol=2e-6,
        )
        assert archive["geometry_mask"].tolist() == [False, True]
        assert archive["quality_target"].tolist() == [0.0, 1.0]
        assert archive["refined_iou"][1] > archive["original_iou"][1]
        # Preferred geometry points became nonzero local evidence, while
        # canonical padding remains exactly zero.
        assert np.any(archive["points_local"][1, :2] != 0.0)
        np.testing.assert_array_equal(archive["points_local"][:, 2:], 0.0)


def test_oriented_frame_and_nnls_are_exact_and_nonnegative():
    basis = _rotation_z(37.0)
    center = np.asarray([2.0, -4.0, 1.0])
    dimensions = np.asarray([2.5, 1.2, 0.7])
    recovered_center, recovered_dimensions, recovered_basis = (
        oriented_box_frame(_corners(center, dimensions, basis))
    )
    np.testing.assert_allclose(recovered_center, center, atol=1e-10)
    np.testing.assert_allclose(recovered_dimensions, dimensions, atol=1e-10)
    np.testing.assert_allclose(recovered_basis, basis, atol=1e-10)

    matrix = np.asarray(
        [[1.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    )
    # Unconstrained x would be negative; NNLS must deactivate it.
    result = nonnegative_least_squares_3x3(
        matrix, np.asarray([0.2, 1.0, 2.0])
    )
    assert (result >= 0.0).all()
    np.testing.assert_allclose(result, [0.0, 0.6, 2.0], atol=1e-8)


def _write_training_archive(path, scene_count=4):
    rng = np.random.default_rng(3)
    samples_per_scene = 2
    count = scene_count * samples_per_scene
    point_count = 6
    geometry_mask = np.tile(np.asarray([True, False]), scene_count)
    points = rng.normal(0.0, 0.2, (count, point_count, 3)).astype(np.float32)
    point_mask = np.ones((count, point_count), dtype=np.bool_)
    point_mask[:, -1] = False
    points[:, -1] = 0.0
    boxes = np.zeros((count, 6), dtype=np.float32)
    boxes[:, 3:] = 1.0
    target = np.zeros((count, 6), dtype=np.float32)
    target[geometry_mask, 0] = 0.05
    target[geometry_mask, 3] = np.log(1.1)
    scene_ids = np.repeat(
        np.asarray(
            [f"scene{index:04d}_00" for index in range(scene_count)]
        ),
        samples_per_scene,
    )
    np.savez(
        path,
        schema=np.asarray(DATASET_SCHEMA),
        format_version=np.asarray(DATASET_FORMAT_VERSION, dtype=np.int64),
        coordinate_frame=np.asarray("box_local"),
        quality_feature_names=np.asarray(QUALITY_FEATURE_NAMES),
        max_center_fraction=np.asarray(0.15, dtype=np.float32),
        max_log_dimension_residual=np.asarray(
            np.log(1.25), dtype=np.float32
        ),
        points_local=points,
        point_mask=point_mask,
        local_boxes=boxes,
        quality_features=rng.uniform(
            0.0, 1.0, (count, 12)
        ).astype(np.float32),
        target_residual=target,
        quality_target=geometry_mask.astype(np.float32),
        geometry_mask=geometry_mask,
        scene_ids=scene_ids,
        original_iou=np.full(count, 0.3, dtype=np.float32),
        refined_iou=np.where(geometry_mask, 0.5, 0.3).astype(np.float32),
        matched_gt_index=np.zeros(count, dtype=np.int64),
        target_center_local_unclipped=np.zeros(
            (count, 3), dtype=np.float32
        ),
        target_dimensions_local_unclipped=np.ones(
            (count, 3), dtype=np.float32
        ),
        basis_world=np.tile(np.eye(3, dtype=np.float32), (count, 1, 1)),
        result_indices=np.tile(
            np.arange(samples_per_scene, dtype=np.int64), scene_count
        ),
        track_ids=np.arange(count, dtype=np.int64),
    )


def test_scene_split_and_balanced_sampling_have_no_leakage(tmp_path):
    path = tmp_path / "training.npz"
    _write_training_archive(path)
    data = load_oriented_refiner_dataset(path)
    train, validation = deterministic_scene_split(
        data.scene_ids, validation_fraction=0.25, seed=17
    )
    assert not (
        set(data.scene_ids[train]) & set(data.scene_ids[validation])
    )
    balanced = balanced_epoch_indices(train, data.geometry_mask, seed=19)
    labels = data.geometry_mask[balanced]
    assert int(labels.sum()) * 2 == len(labels)
    # Oversampling never pulls samples from held-out scenes.
    assert set(balanced).issubset(set(train))


def test_negative_samples_do_not_supervise_geometry():
    torch = pytest.importorskip("torch")
    center = torch.zeros((2, 3), requires_grad=True)
    dimensions = torch.zeros((2, 3), requires_grad=True)
    quality_logits = torch.zeros(2, requires_grad=True)
    target = torch.tensor(
        [[0.1, 0.0, 0.0, 0.05, 0.0, 0.0], [9.0] * 6]
    )
    loss, _ = oriented_refiner_loss(
        {
            "center_residual_fraction": center,
            "log_dimension_residual": dimensions,
            "quality": torch.sigmoid(quality_logits),
        },
        target,
        torch.tensor([1.0, 0.0]),
        torch.tensor([True, False]),
    )
    loss.backward()
    np.testing.assert_array_equal(center.grad[1].detach().numpy(), 0.0)
    np.testing.assert_array_equal(dimensions.grad[1].detach().numpy(), 0.0)
    assert quality_logits.grad[1] != 0.0


def test_training_writes_strict_checkpoint_with_scene_split(tmp_path):
    pytest.importorskip("torch")
    dataset = tmp_path / "training.npz"
    checkpoint = tmp_path / "b5v2.pt"
    _write_training_archive(dataset)
    config = OrientedBoxRefinerConfig(
        point_hidden_dim=8,
        point_embedding_dim=8,
        head_hidden_dim=8,
    )
    result = train_oriented_box_refiner(
        dataset,
        checkpoint,
        config=config,
        epochs=2,
        batch_size=4,
        validation_fraction=0.25,
        seed=23,
    )
    assert checkpoint.is_file()
    assert result["scene_leakage"] is False
    assert not (
        set(result["train_scenes"]) & set(result["validation_scenes"])
    )
    assert math.isfinite(result["best_validation_loss"])
    model = PointNetOrientedBoxRefiner(config)
    assert load_oriented_box_refiner_checkpoint(model, checkpoint) is model
