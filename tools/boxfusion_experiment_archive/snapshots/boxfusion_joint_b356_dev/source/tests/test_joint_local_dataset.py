"""Strict CPU tests for exact P=128 joint diagnostic injection."""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest

from boxfusion.quality_score import QUALITY_FEATURE_NAMES
from tools.build_joint_local_dataset import (
    JOINT_LOCAL_DATASET_SCHEMA,
    JOINT_METADATA_KEYS,
    JOINT_SAMPLE_KEYS,
    REFINER_QUALITY_INDEX,
    RUNTIME_QUALITY_FEATURE_SOURCE,
    JointDatasetBuildConfig,
    build_joint_local_dataset,
)
from tools.build_oriented_refiner_dataset import (
    AP50_DATASET_FORMAT_VERSION,
    DATASET_SCHEMA,
    STRICT_PROVENANCE_EXPECTED,
    TARGET_LINE_SEARCH_ALPHAS,
    V2_METADATA_KEYS,
    V2_SAMPLE_KEYS,
)
from tools.train_joint_local_head import load_joint_local_dataset


SCENES = ("scene0000_00", "scene0001_00")
FORBIDDEN = ("scene0700_00",)


def _scene_digest(scenes) -> str:
    canonical = "\n".join(sorted(set(scenes))) + "\n"
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _provenance_payload() -> dict[str, np.ndarray]:
    result = {
        name: np.asarray(value)
        for name, value in STRICT_PROVENANCE_EXPECTED.items()
    }
    result["summary_json"] = np.asarray(
        json.dumps(STRICT_PROVENANCE_EXPECTED, sort_keys=True)
    )
    return result


def _diagnostic_payload(
    scene: str,
    *,
    point_count: int = 128,
    omit_joint_field: str | None = None,
    legacy_refiner_quality: bool = False,
) -> dict[str, np.ndarray]:
    observations = 2
    aggregate_points = np.zeros(
        (observations, 512, 3), dtype=np.float32
    )
    aggregate_mask = np.zeros((observations, 512), dtype=np.bool_)
    aggregate_mask[:, :128] = True
    aggregate_points[0, :128, 0] = np.linspace(
        -0.5, 0.5, 128, dtype=np.float32
    )
    aggregate_points[1, :128, 1] = np.linspace(
        -0.4, 0.4, 128, dtype=np.float32
    )
    gate_points = np.zeros(
        (observations, 8192, 3), dtype=np.float32
    )
    gate_mask = np.zeros((observations, 8192), dtype=np.bool_)
    gate_mask[:, :256] = True
    local_boxes = np.zeros((observations, 6), dtype=np.float32)
    local_boxes[:, 3:6] = np.asarray([1.0, 1.2, 0.8], dtype=np.float32)
    centers = np.zeros((observations, 3), dtype=np.float64)
    centers[:, 2] = 3.0
    basis = np.tile(
        np.eye(3, dtype=np.float64), (observations, 1, 1)
    )
    view_valid = np.zeros((observations, 5), dtype=np.bool_)
    view_valid[:, :2] = True
    frame_ids = np.full((observations, 5), -1, dtype=np.int64)
    frame_ids[:, :2] = np.asarray([10, 20], dtype=np.int64)
    view_scores = np.full((observations, 5), np.nan, dtype=np.float32)
    view_scores[:, :2] = np.asarray([0.9, 0.8], dtype=np.float32)
    bboxes = np.full(
        (observations, 5, 4), np.nan, dtype=np.float32
    )
    bboxes[:, :2] = np.asarray([20, 20, 60, 60], dtype=np.float32)
    intrinsics = np.full(
        (observations, 5, 3, 3), np.nan, dtype=np.float32
    )
    intrinsics[:, :2] = np.asarray(
        [[60, 0, 50], [0, 60, 50], [0, 0, 1]], dtype=np.float32
    )
    poses = np.full(
        (observations, 5, 4, 4), np.nan, dtype=np.float32
    )
    poses[:, :2] = np.eye(4, dtype=np.float32)
    image_shapes = np.full(
        (observations, 5, 2), -1, dtype=np.int64
    )
    image_shapes[:, :2] = np.asarray([100, 100], dtype=np.int64)
    quality = np.full(
        (observations, len(QUALITY_FEATURE_NAMES)),
        0.5,
        dtype=np.float32,
    )
    joint_quality = quality.copy()
    if legacy_refiner_quality:
        quality[:, REFINER_QUALITY_INDEX] = 0.0

    joint_points = np.zeros(
        (observations, 5, point_count, 3), dtype=np.float32
    )
    joint_mask = np.zeros(
        (observations, 5, point_count), dtype=np.bool_
    )
    if point_count == 128:
        joint_mask[:, 0, :4] = True
        joint_mask[:, 1, :3] = True
        joint_points[:, 0, :4, 0] = np.asarray(
            [-0.3, -0.1, 0.1, 0.3], dtype=np.float32
        )
        joint_points[:, 1, :3, 1] = np.asarray(
            [-0.2, 0.0, 0.2], dtype=np.float32
        )
    else:
        # Deliberately valid-looking P=512 tensors for the fail-fast test.
        joint_mask[:, 0, :4] = True
        joint_mask[:, 1, :3] = True
    joint_view_mask = joint_mask.any(axis=2)
    joint_view_features = np.zeros(
        (observations, 5, 9), dtype=np.float32
    )
    joint_view_features[:, 0] = np.asarray(
        [0.9, 0.8, 0.75, 0.7, 4.0 / point_count, 1.0, 0.5, 0.5, 1.0],
        dtype=np.float32,
    )
    joint_view_features[:, 1] = np.asarray(
        [0.8, 0.7, 0.65, 0.6, 3.0 / point_count, 0.0, 0.5, 0.5, 0.5],
        dtype=np.float32,
    )
    payload = {
        "scene_id": np.asarray(scene),
        "quality_features": quality.copy(),
        "quality_feature_names": np.asarray(
            QUALITY_FEATURE_NAMES, dtype=np.str_
        ),
        "result_indices": np.asarray([0, 1], dtype=np.int64),
        "track_ids": np.asarray([7, 8], dtype=np.int64),
        "box_refiner_points_local": aggregate_points,
        "box_refiner_point_mask": aggregate_mask,
        "box_refiner_local_boxes": local_boxes.copy(),
        "box_refiner_frame_valid": np.ones(
            observations, dtype=np.bool_
        ),
        "box_refiner_gate_points_local": gate_points,
        "box_refiner_gate_point_mask": gate_mask,
        "box_refiner_frame_centers": centers.copy(),
        "box_refiner_frame_basis": basis.copy(),
        "box_refiner_view_valid": view_valid.copy(),
        "box_refiner_view_frame_ids": frame_ids.copy(),
        "box_refiner_view_scores": view_scores,
        "box_refiner_view_bboxes": bboxes,
        "box_refiner_view_intrinsics": intrinsics,
        "box_refiner_view_camera_to_world": poses,
        "box_refiner_view_image_shapes": image_shapes,
        "selected_view_counts": np.asarray([2, 2], dtype=np.int64),
        "selected_view_frame_ids": frame_ids.copy(),
        "top_k_view_valid": view_valid.copy(),
        "joint_points_local": joint_points,
        "joint_point_mask": joint_mask,
        "joint_view_features": joint_view_features,
        "joint_view_mask": joint_view_mask,
        "joint_local_boxes": local_boxes.copy(),
        "joint_quality_features": joint_quality,
        "joint_frame_center": centers.copy(),
        "joint_frame_centers": centers.copy(),
        "joint_frame_basis": basis.copy(),
        "joint_input_valid": np.ones(
            observations, dtype=np.bool_
        ),
    }
    payload.update(_provenance_payload())
    if omit_joint_field is not None:
        payload.pop(omit_joint_field)
    return payload


def _strict_source_arrays() -> dict[str, np.ndarray]:
    sample_count = 4
    points = np.concatenate(
        [
            _diagnostic_payload(scene)["box_refiner_points_local"]
            for scene in SCENES
        ],
        axis=0,
    )
    point_mask = np.concatenate(
        [
            _diagnostic_payload(scene)["box_refiner_point_mask"]
            for scene in SCENES
        ],
        axis=0,
    )
    local_boxes = np.concatenate(
        [
            _diagnostic_payload(scene)["box_refiner_local_boxes"]
            for scene in SCENES
        ],
        axis=0,
    )
    quality = np.full(
        (sample_count, len(QUALITY_FEATURE_NAMES)),
        0.5,
        dtype=np.float32,
    )
    geometry = np.asarray([True, False, True, False], dtype=np.bool_)
    runtime_eligible = geometry.copy()
    identity = np.asarray([False, True, False, True], dtype=np.bool_)
    candidate = np.ones(sample_count, dtype=np.bool_)
    original_iou = np.asarray(
        [0.40, 0.60, 0.40, 0.60], dtype=np.float32
    )
    refined_iou = np.asarray(
        [0.60, 0.60, 0.60, 0.60], dtype=np.float32
    )
    target = np.zeros((sample_count, 6), dtype=np.float32)
    target[geometry, 0] = 0.1
    matched_gt = np.zeros((sample_count, 6), dtype=np.float32)
    matched_gt[:, 3:6] = 1.0
    arrays = {
        "points_local": points,
        "point_mask": point_mask,
        "local_boxes": local_boxes,
        "quality_features": quality,
        "target_residual": target,
        "quality_target": np.asarray(
            [0.95, 0.0, 0.95, 0.0], dtype=np.float32
        ),
        "geometry_mask": geometry,
        "scene_ids": np.asarray(
            [SCENES[0], SCENES[0], SCENES[1], SCENES[1]],
            dtype=np.str_,
        ),
        "original_iou": original_iou,
        "refined_iou": refined_iou,
        "matched_gt_index": np.zeros(sample_count, dtype=np.int64),
        "target_center_local_unclipped": np.zeros(
            (sample_count, 3), dtype=np.float32
        ),
        "target_dimensions_local_unclipped": local_boxes[:, 3:6].copy(),
        "basis_world": np.tile(
            np.eye(3, dtype=np.float32), (sample_count, 1, 1)
        ),
        "result_indices": np.asarray([0, 1, 0, 1], dtype=np.int64),
        "track_ids": np.asarray([7, 8, 7, 8], dtype=np.int64),
        "aligned_basis": np.tile(
            np.eye(3, dtype=np.float32), (sample_count, 1, 1)
        ),
        "original_aligned_center": np.zeros(
            (sample_count, 3), dtype=np.float32
        ),
        "matched_gt_box": matched_gt,
        "iou_gain": np.asarray(
            [0.20, 0.0, 0.20, 0.0], dtype=np.float32
        ),
        "cross_iou50": geometry.copy(),
        "near_iou50": np.asarray(
            [0.33, 0.33, 0.33, 0.33], dtype=np.float32
        ),
        "ap50_weight": np.asarray(
            [6.0, 1.0, 6.0, 1.0], dtype=np.float32
        ),
        "runtime_eligible": runtime_eligible,
        "selected_view_counts": np.full(
            sample_count, 2, dtype=np.int64
        ),
        "identity_tp50": identity,
        "candidate_oracle_tp50": candidate,
        "schema": np.asarray(DATASET_SCHEMA),
        "format_version": np.asarray(
            AP50_DATASET_FORMAT_VERSION, dtype=np.int64
        ),
        "coordinate_frame": np.asarray("box_local"),
        "quality_feature_names": np.asarray(
            QUALITY_FEATURE_NAMES, dtype=np.str_
        ),
        "max_center_fraction": np.asarray(0.15, dtype=np.float32),
        "max_log_dimension_residual": np.asarray(
            np.log(1.25), dtype=np.float32
        ),
        "objective": np.asarray("ap50"),
        "strict_k5_diagnostics": np.asarray(True, dtype=np.bool_),
        "expected_top_k_views": np.asarray(5, dtype=np.int64),
        "min_runtime_views": np.asarray(2, dtype=np.int64),
        "min_runtime_points": np.asarray(128, dtype=np.int64),
        "runtime_minimum_extent": np.asarray(0.4, dtype=np.float32),
        "near_iou50_band": np.asarray(0.15, dtype=np.float32),
        "gain_cap": np.asarray(0.25, dtype=np.float32),
        "gain_sample_weight": np.asarray(2.0, dtype=np.float32),
        "cross_iou50_sample_weight": np.asarray(4.0, dtype=np.float32),
        "near_iou50_sample_weight": np.asarray(2.0, dtype=np.float32),
        "min_match_iou": np.asarray(0.15, dtype=np.float32),
        "improvement_epsilon": np.asarray(1e-4, dtype=np.float32),
        "target_line_search_alphas": np.asarray(
            TARGET_LINE_SEARCH_ALPHAS, dtype=np.float32
        ),
        "forbidden_scene_count": np.asarray(
            len(FORBIDDEN), dtype=np.int64
        ),
        "forbidden_scene_sha256": np.asarray(
            _scene_digest(FORBIDDEN)
        ),
        "training_scene_count": np.asarray(
            len(SCENES), dtype=np.int64
        ),
        "training_scene_sha256": np.asarray(_scene_digest(SCENES)),
    }
    arrays.update(
        {
            name: np.asarray(value)
            for name, value in STRICT_PROVENANCE_EXPECTED.items()
        }
    )
    assert set(arrays) == V2_SAMPLE_KEYS | V2_METADATA_KEYS
    return arrays


def _fixture_tree(tmp_path, *, legacy_refiner_quality: bool = False):
    diagnostics = tmp_path / "diagnostics"
    diagnostics.mkdir()
    for scene in SCENES:
        np.savez_compressed(
            diagnostics / f"{scene}_tracks.npz",
            **_diagnostic_payload(
                scene,
                legacy_refiner_quality=legacy_refiner_quality,
            ),
        )
    source = tmp_path / "b5_v2.npz"
    source_arrays = _strict_source_arrays()
    if legacy_refiner_quality:
        source_arrays["quality_features"][
            :, REFINER_QUALITY_INDEX
        ] = 0.0
    np.savez_compressed(source, **source_arrays)
    forbidden = tmp_path / "val.txt"
    forbidden.write_text("\n".join(FORBIDDEN) + "\n")
    output = tmp_path / "joint.npz"
    return source, diagnostics, forbidden, output


def test_exact_p128_joint_diagnostics_are_copied_without_reconstruction(
    tmp_path,
):
    source, diagnostics, forbidden, output = _fixture_tree(tmp_path)
    summary = build_joint_local_dataset(
        JointDatasetBuildConfig(
            b5_dataset=source,
            diagnostics_root=diagnostics,
            forbidden_scene_list=forbidden,
            output=output,
        )
    )
    assert summary.samples == 4
    assert summary.scenes == 2
    assert summary.points_per_view == 128
    assert summary.legacy_refiner_quality_normalized_rows == 0
    with np.load(output, allow_pickle=False) as archive:
        assert set(archive.files) == JOINT_SAMPLE_KEYS | JOINT_METADATA_KEYS
        assert archive["schema"].item() == JOINT_LOCAL_DATASET_SCHEMA
        assert (
            archive["quality_feature_source"].item()
            == RUNTIME_QUALITY_FEATURE_SOURCE
        )
        assert (
            archive["legacy_refiner_quality_normalized_rows"].item()
            == 0
        )
        expected_points = np.concatenate(
            [
                _diagnostic_payload(scene)["joint_points_local"]
                for scene in SCENES
            ],
            axis=0,
        )
        np.testing.assert_array_equal(
            archive["joint_points_local"], expected_points
        )
        expected_features = np.concatenate(
            [
                _diagnostic_payload(scene)["joint_view_features"]
                for scene in SCENES
            ],
            axis=0,
        )
        np.testing.assert_array_equal(
            archive["joint_view_features"], expected_features
        )
        for name in archive.files:
            assert not archive[name].dtype.hasobject
    loaded = load_joint_local_dataset(output)
    assert loaded.points_local.shape == (4, 5, 128, 3)
    assert loaded.points_per_view == 128
    assert np.all(
        loaded.quality_features[:, REFINER_QUALITY_INDEX] == 0.5
    )


def test_legacy_disabled_refiner_sentinel_is_fail_closed_normalized(
    tmp_path,
):
    source, diagnostics, forbidden, output = _fixture_tree(
        tmp_path, legacy_refiner_quality=True
    )
    summary = build_joint_local_dataset(
        JointDatasetBuildConfig(
            b5_dataset=source,
            diagnostics_root=diagnostics,
            forbidden_scene_list=forbidden,
            output=output,
        )
    )
    assert summary.legacy_refiner_quality_normalized_rows == 4
    with np.load(output, allow_pickle=False) as archive:
        assert np.all(
            archive["quality_features"][:, REFINER_QUALITY_INDEX] == 0.0
        )
        assert np.all(
            archive["joint_quality_features"][
                :, REFINER_QUALITY_INDEX
            ]
            == 0.5
        )
        assert (
            archive["legacy_refiner_quality_normalized_rows"].item()
            == 4
        )
    loaded = load_joint_local_dataset(output)
    assert np.all(
        loaded.quality_features[:, REFINER_QUALITY_INDEX] == 0.5
    )


@pytest.mark.parametrize(
    "column,source_value,joint_value",
    [
        (0, 0.5, 0.6),
        (REFINER_QUALITY_INDEX, 0.1, 0.5),
        (REFINER_QUALITY_INDEX, 0.0, 0.4),
        (REFINER_QUALITY_INDEX, 0.5, 0.0),
    ],
)
def test_quality_migration_rejects_every_unapproved_difference(
    tmp_path, column, source_value, joint_value
):
    source, diagnostics, forbidden, output = _fixture_tree(tmp_path)
    source_arrays = _strict_source_arrays()
    source_arrays["quality_features"][:, column] = source_value
    np.savez_compressed(source, **source_arrays)
    for scene in SCENES:
        payload = _diagnostic_payload(scene)
        payload["quality_features"][:, column] = source_value
        payload["joint_quality_features"][:, column] = joint_value
        np.savez_compressed(
            diagnostics / f"{scene}_tracks.npz", **payload
        )
    with pytest.raises(
        (TypeError, ValueError),
        match="quality_features|refiner_quality",
    ):
        build_joint_local_dataset(
            JointDatasetBuildConfig(
                b5_dataset=source,
                diagnostics_root=diagnostics,
                forbidden_scene_list=forbidden,
                output=output,
            )
        )
    assert not output.exists()


def test_loader_revalidates_quality_migration_metadata(tmp_path):
    source, diagnostics, forbidden, output = _fixture_tree(
        tmp_path, legacy_refiner_quality=True
    )
    build_joint_local_dataset(
        JointDatasetBuildConfig(
            b5_dataset=source,
            diagnostics_root=diagnostics,
            forbidden_scene_list=forbidden,
            output=output,
        )
    )
    with np.load(output, allow_pickle=False) as archive:
        arrays = {
            name: np.asarray(archive[name]).copy()
            for name in archive.files
        }
    arrays["legacy_refiner_quality_normalized_rows"] = np.asarray(
        3, dtype=np.int64
    )
    tampered = tmp_path / "tampered_joint.npz"
    np.savez_compressed(tampered, **arrays)
    with pytest.raises(ValueError, match="metadata is inconsistent"):
        load_joint_local_dataset(tampered)


@pytest.mark.parametrize(
    "point_count,missing,error",
    [
        (512, None, r"\[N,5,128,3\]"),
        (128, "joint_view_features", "missing joint Top-K fields"),
    ],
)
def test_wrong_p_or_missing_exact_joint_field_fails_fast(
    tmp_path, point_count, missing, error
):
    source, diagnostics, forbidden, output = _fixture_tree(tmp_path)
    scene = SCENES[0]
    np.savez_compressed(
        diagnostics / f"{scene}_tracks.npz",
        **_diagnostic_payload(
            scene,
            point_count=point_count,
            omit_joint_field=missing,
        ),
    )
    with pytest.raises((TypeError, ValueError), match=error):
        build_joint_local_dataset(
            JointDatasetBuildConfig(
                b5_dataset=source,
                diagnostics_root=diagnostics,
                forbidden_scene_list=forbidden,
                output=output,
            )
        )
    assert not output.exists()


def test_forbidden_validation_hash_cannot_be_substituted(tmp_path):
    source, diagnostics, _, output = _fixture_tree(tmp_path)
    wrong = tmp_path / "wrong_val.txt"
    wrong.write_text("scene0701_00\n")
    with pytest.raises(ValueError, match="does not match"):
        build_joint_local_dataset(
            JointDatasetBuildConfig(
                b5_dataset=source,
                diagnostics_root=diagnostics,
                forbidden_scene_list=wrong,
                output=output,
            )
        )
    assert not output.exists()
