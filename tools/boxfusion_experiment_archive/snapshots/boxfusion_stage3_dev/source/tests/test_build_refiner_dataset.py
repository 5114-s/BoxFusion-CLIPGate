import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest


SOURCE = (
    Path(__file__).resolve().parents[1]
    / "tools"
    / "build_refiner_dataset.py"
)
SPEC = importlib.util.spec_from_file_location("build_refiner_dataset", SOURCE)
builder = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = builder
SPEC.loader.exec_module(builder)


def write_axis_alignment(scan_root, scene, transform):
    scene_root = scan_root / scene
    scene_root.mkdir(parents=True)
    values = " ".join(str(float(value)) for value in transform.reshape(-1))
    (scene_root / f"{scene}.txt").write_text(
        f"axisAlignment = {values}\n"
    )


def write_diagnostics(
    root,
    scene,
    boxes,
    *,
    points=None,
    point_mask=None,
    quality_features=None,
    scores=None,
    stored_scene=True,
    filename=None,
):
    boxes = np.asarray(boxes, dtype=np.float32)
    count = len(boxes)
    if points is None:
        points = np.zeros((count, 3, 3), dtype=np.float32)
        points[:, 0, :] = boxes[:, :3]
        points[:, 1, :] = boxes[:, :3] + 0.1
    if point_mask is None:
        point_mask = np.tile(
            np.asarray([True, True, False]), (count, 1)
        )
    if quality_features is None:
        quality_features = np.full((count, 12), 0.5, dtype=np.float32)
    if scores is None:
        scores = np.full(count, 0.8, dtype=np.float32)
    payload = {
        "boxes": boxes,
        "scores": scores,
        "quality_features": quality_features,
        "points": points,
        "point_mask": point_mask,
    }
    if stored_scene:
        payload["scene_id"] = np.asarray(scene)
    np.savez(root / (filename or f"{scene}.npz"), **payload)


def base_tree(tmp_path, scenes):
    diagnostics_root = tmp_path / "diagnostics"
    scans_root = tmp_path / "scans"
    gt_root = tmp_path / "gt"
    diagnostics_root.mkdir()
    scans_root.mkdir()
    gt_root.mkdir()
    scene_list = tmp_path / "scenes.txt"
    scene_list.write_text("\n".join(scenes) + "\n")
    return diagnostics_root, scans_root, gt_root, scene_list


def config(tmp_path, diagnostics, scans, gt, scene_list, **overrides):
    values = {
        "diagnostics_root": diagnostics,
        "scan_root": scans,
        "gt_root": gt,
        "scene_list": scene_list,
        "output": tmp_path / "training.npz",
        "min_iou": 0.5,
        "include_negatives": False,
    }
    values.update(overrides)
    return builder.BuildConfig(**values)


def test_axis_alignment_transforms_points_and_aabb_enclosure_exactly():
    transform = np.asarray(
        [
            [0.0, -1.0, 0.0, 10.0],
            [1.0, 0.0, 0.0, 20.0],
            [0.0, 0.0, 1.0, 30.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    points = np.asarray([[1.0, 2.0, 3.0]])
    np.testing.assert_allclose(
        builder.transform_points(points, transform),
        [[8.0, 21.0, 33.0]],
    )
    boxes = np.asarray([[1.0, 2.0, 3.0, 2.0, 4.0, 6.0]])
    aligned = builder.align_center_size_boxes(boxes, transform)
    np.testing.assert_allclose(
        aligned, [[8.0, 21.0, 33.0, 4.0, 2.0, 6.0]]
    )


def test_build_positive_dataset_axis_aligns_predictions_and_points(tmp_path):
    scene = "scene0000_00"
    diagnostics, scans, gt, scene_list = base_tree(tmp_path, [scene])
    transform = np.eye(4)
    transform[:3, 3] = [10.0, 0.0, 0.0]
    write_axis_alignment(scans, scene, transform)
    boxes = np.asarray(
        [
            [0.0, 0.0, 0.0, 2.0, 2.0, 2.0],
            [20.0, 0.0, 0.0, 2.0, 2.0, 2.0],
        ]
    )
    points = np.asarray(
        [
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [99.0, 99.0, 99.0]],
            [[20.0, 0.0, 0.0], [21.0, 0.0, 0.0], [88.0, 88.0, 88.0]],
        ],
        dtype=np.float32,
    )
    write_diagnostics(diagnostics, scene, boxes, points=points)
    np.save(
        gt / f"{scene}_bbox.npy",
        np.asarray([[10.0, 0.0, 0.0, 2.0, 2.0, 2.0, 3.0]]),
    )

    summary = builder.build_refiner_dataset(
        config(tmp_path, diagnostics, scans, gt, scene_list)
    )
    assert summary.input_predictions == 2
    assert summary.positives == 1
    assert summary.negatives == 0
    assert summary.output_samples == 1
    with np.load(summary.output, allow_pickle=False) as result:
        assert set(result.files) == {
            "points",
            "point_mask",
            "boxes",
            "quality_features",
            "target_boxes",
            "target_iou",
        }
        np.testing.assert_allclose(
            result["boxes"], [[10.0, 0.0, 0.0, 2.0, 2.0, 2.0]]
        )
        np.testing.assert_allclose(result["target_boxes"], result["boxes"])
        np.testing.assert_allclose(result["target_iou"], [1.0])
        np.testing.assert_allclose(
            result["points"][0, :2],
            [[10.0, 0.0, 0.0], [11.0, 0.0, 0.0]],
        )
        # Padding is canonical zero rather than transformed garbage.
        np.testing.assert_array_equal(result["points"][0, 2], 0.0)
        assert result["point_mask"].dtype == np.bool_
        assert result["boxes"].dtype == np.float32


def test_include_negatives_uses_identity_regression_target(tmp_path):
    scene = "scene0001_00"
    diagnostics, scans, gt, scene_list = base_tree(tmp_path, [scene])
    write_axis_alignment(scans, scene, np.eye(4))
    boxes = np.asarray(
        [
            [0.0, 0.0, 0.0, 2.0, 2.0, 2.0],
            [10.0, 0.0, 0.0, 2.0, 2.0, 2.0],
        ]
    )
    write_diagnostics(diagnostics, scene, boxes)
    np.save(
        gt / f"{scene}_bbox.npy",
        np.asarray([[0.0, 0.0, 0.0, 2.0, 2.0, 2.0]]),
    )
    summary = builder.build_refiner_dataset(
        config(
            tmp_path,
            diagnostics,
            scans,
            gt,
            scene_list,
            include_negatives=True,
        )
    )
    assert summary.positives == 1
    assert summary.negatives == 1
    with np.load(summary.output, allow_pickle=False) as result:
        assert len(result["boxes"]) == 2
        np.testing.assert_allclose(result["target_iou"], [1.0, 0.0])
        np.testing.assert_allclose(
            result["target_boxes"][1], result["boxes"][1]
        )


def test_multiple_scenes_preserve_scene_and_proposal_order(tmp_path):
    scenes = ["scene0002_00", "scene0003_00"]
    diagnostics, scans, gt, scene_list = base_tree(tmp_path, scenes)
    for index, scene in enumerate(scenes):
        write_axis_alignment(scans, scene, np.eye(4))
        box = np.asarray(
            [[float(index * 5), 0.0, 0.0, 1.0, 1.0, 1.0]]
        )
        write_diagnostics(
            diagnostics,
            scene,
            box,
            stored_scene=False,
            filename=f"{scene}_runtime_diagnostics.npz",
        )
        np.save(gt / f"{scene}_bbox.npy", box)
    summary = builder.build_refiner_dataset(
        config(tmp_path, diagnostics, scans, gt, scene_list)
    )
    with np.load(summary.output, allow_pickle=False) as result:
        np.testing.assert_allclose(result["boxes"][:, 0], [0.0, 5.0])


@pytest.mark.parametrize(
    "mutation, error_type, message",
    [
        (
            {"quality_features": np.zeros((1, 11), dtype=np.float32)},
            ValueError,
            "quality_features",
        ),
        (
            {"point_mask": np.ones((1, 3), dtype=np.uint8)},
            TypeError,
            "Boolean",
        ),
        (
            {"scores": np.asarray([1.1], dtype=np.float32)},
            ValueError,
            r"\[0, 1\]",
        ),
        (
            {
                "points": np.asarray(
                    [[[np.nan, 0.0, 0.0], [0.0, 0.0, 0.0]]]
                ),
                "point_mask": np.asarray([[True, False]]),
            },
            ValueError,
            "finite",
        ),
    ],
)
def test_diagnostic_validation_is_strict(
    tmp_path, mutation, error_type, message
):
    scene = "scene0004_00"
    path = tmp_path / f"{scene}.npz"
    defaults = {
        "scene_id": np.asarray(scene),
        "boxes": np.asarray([[0, 0, 0, 1, 1, 1]], dtype=np.float32),
        "scores": np.asarray([0.5], dtype=np.float32),
        "quality_features": np.zeros((1, 12), dtype=np.float32),
        "points": np.zeros((1, 3, 3), dtype=np.float32),
        "point_mask": np.asarray([[True, False, False]]),
    }
    defaults.update(mutation)
    np.savez(path, **defaults)
    with pytest.raises(error_type, match=message):
        builder.load_scene_diagnostics(path, scene)


def test_scene_mismatch_and_duplicate_scene_list_fail(tmp_path):
    path = tmp_path / "scene0005_00.npz"
    write_diagnostics(
        tmp_path,
        "scene0006_00",
        [[0, 0, 0, 1, 1, 1]],
        filename=path.name,
    )
    with pytest.raises(ValueError, match="does not match"):
        builder.load_scene_diagnostics(path, "scene0005_00")

    scene_list = tmp_path / "duplicates.txt"
    scene_list.write_text("scene0005_00\nscene0005_00\n")
    with pytest.raises(ValueError, match="Duplicate"):
        builder.read_scene_ids(scene_list)


def test_no_positive_samples_does_not_write_output(tmp_path):
    scene = "scene0007_00"
    diagnostics, scans, gt, scene_list = base_tree(tmp_path, [scene])
    write_axis_alignment(scans, scene, np.eye(4))
    write_diagnostics(
        diagnostics,
        scene,
        [[0.0, 0.0, 0.0, 1.0, 1.0, 1.0]],
    )
    np.save(
        gt / f"{scene}_bbox.npy",
        [[20.0, 0.0, 0.0, 1.0, 1.0, 1.0]],
    )
    cfg = config(tmp_path, diagnostics, scans, gt, scene_list)
    with pytest.raises(ValueError, match="No samples"):
        builder.build_refiner_dataset(cfg)
    assert not cfg.output.exists()
