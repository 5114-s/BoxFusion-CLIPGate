import inspect
from types import MappingProxyType

import numpy as np
import pytest

from boxfusion import fastsam_residual_shadow as f0


HEIGHT = 480
WIDTH = 640


def camera_inputs(focal=100.0):
    intrinsics = np.asarray(
        [[focal, 0.0, WIDTH / 2.0], [0.0, focal, HEIGHT / 2.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    return intrinsics, np.eye(4, dtype=np.float64)


def rectangle(top, left, height, width):
    mask = np.zeros((HEIGHT, WIDTH), dtype=bool)
    mask[top : top + height, left : left + width] = True
    return mask


def run(masks, confidences=None, depth=None, boxes=None, focal=100.0, pose=None):
    masks = np.asarray(masks)
    if confidences is None:
        confidences = np.full(len(masks), 0.8, dtype=np.float32)
    if depth is None:
        depth = np.full((HEIGHT, WIDTH), 2.0, dtype=np.float32)
    if boxes is None:
        boxes = np.empty((0, 4), dtype=np.float32)
    intrinsics, default_pose = camera_inputs(focal=focal)
    return f0.select_and_lift_residual_masks(
        masks=masks,
        confidences=np.asarray(confidences),
        depth_m=depth,
        explained_boxes_xyxy=boxes,
        intrinsics=intrinsics,
        camera_to_world=default_pose if pose is None else pose,
    )


def test_policy_is_sealed_to_preregistered_f0_values():
    assert isinstance(f0.POLICY, MappingProxyType)
    assert f0.POLICY == {
        "mask_shape": (480, 640),
        "mask_pixels": (200, 122880),
        "min_tight_box_side_px": 16,
        "max_tight_box_aspect": 6.0,
        "min_valid_depth_ratio": 0.5,
        "min_residual_pixels": 200,
        "min_residual_ratio": 0.2,
        "explained_box_expand_px": 4.0,
        "sort": (
            "-confidence",
            "-residual_ratio",
            "-residual_pixels",
            "tight_box_xyxy",
            "mask_sha256",
        ),
        "dedup_mask_iou": 0.8,
        "dedup_smaller_containment": 0.9,
        "top_k": 16,
        "depth_m": (0.1, 6.0),
        "mask_edge_margin_px": 1,
        "depth_edge_connectivity": 4,
        "depth_edge_jump_m": 0.15,
        "voxel_size_m": 0.02,
        "min_unique_voxels": 16,
        "max_stored_points": 2048,
        "world_quantiles": (0.02, 0.98),
        "min_world_aabb_extent_m": 0.02,
        "lift_support": "full_mask_not_residual",
    }
    with pytest.raises(TypeError):
        f0.POLICY["top_k"] = 17


def test_expanded_cutr_union_only_gates_residual_and_lift_uses_full_mask():
    mask = rectangle(100, 100, 40, 40)
    # Expanded inclusive box covers x=96..123.  Within this mask that is 24 of
    # 40 columns, leaving 640 residual pixels (40%) for eligibility.
    boxes = np.asarray([[100.0, 100.0, 119.0, 139.0]], dtype=np.float32)
    result = run(np.stack([mask]), boxes=boxes)

    assert len(result.candidates) == 1
    candidate = result.candidates[0]
    assert candidate.pixel_count == 1600
    assert candidate.residual_pixel_count == 640
    assert candidate.residual_ratio == pytest.approx(0.4)
    # Full-mask erosion is 38x38.  If lifting incorrectly used the residual it
    # could not retain all 1,444 support pixels.
    assert candidate.support_pixel_count == 38 * 38
    assert result.diagnostics.explained_union_pixels == 28 * 48


def test_residual_pixel_and_ratio_thresholds_are_both_enforced():
    mask = rectangle(100, 100, 40, 40)
    # Expanded box covers 35 columns inside the mask, leaving 200 pixels.  The
    # absolute threshold passes but 12.5% fails the ratio threshold.
    ratio_fail_box = np.asarray([[100.0, 100.0, 130.0, 139.0]], dtype=np.float32)
    result = run(np.stack([mask]), boxes=ratio_fail_box)
    assert result.masks[0].residual_pixel_count == 200
    assert result.masks[0].reason == "residual_ratio_too_low"

    # Complete expanded coverage leaves no residual and fails the earlier
    # absolute residual-pixel gate.
    complete_box = np.asarray([[100.0, 100.0, 139.0, 139.0]], dtype=np.float32)
    result = run(np.stack([mask]), boxes=complete_box)
    assert result.masks[0].residual_pixel_count == 0
    assert result.masks[0].reason == "residual_pixels_too_few"


def test_residual_is_measured_only_over_valid_metric_mask_support():
    mask = rectangle(100, 100, 40, 40)
    depth = np.full((HEIGHT, WIDTH), 2.0, dtype=np.float32)
    # The right half has invalid depth.  The explained box covers 12 of the 20
    # valid columns after expansion, leaving 320/800 valid residual pixels.
    depth[100:140, 120:140] = 0.0
    boxes = np.asarray([[100.0, 100.0, 107.0, 139.0]], dtype=np.float32)
    result = run(np.stack([mask]), depth=depth, boxes=boxes)
    row = result.masks[0]
    assert row.valid_pixel_count == 800
    assert row.valid_ratio == pytest.approx(0.5)
    assert row.residual_pixel_count == 320
    assert row.residual_ratio == pytest.approx(0.4)
    assert len(result.candidates) == 1


def test_sort_is_deterministic_and_higher_ranked_duplicate_is_representative():
    left = rectangle(80, 40, 30, 30)
    right = rectangle(80, 140, 30, 30)
    right_duplicate = right.copy()
    masks = np.stack([right, right_duplicate, left])
    result = run(masks, confidences=np.asarray([0.7, 0.9, 0.9]))

    # Same confidence/residual measurements: conventional tight xyxy order
    # puts the left mask first.  The 0.9 duplicate represents the 0.7 mask.
    assert [item.raw_index for item in result.candidates] == [2, 1]
    assert result.masks[0].reason == "duplicate"
    assert result.masks[0].duplicate_of_raw_index == 1
    assert result.diagnostics.deduplicated_count == 1
    assert result.diagnostics.post_dedup_count == 2


def test_smaller_containment_deduplicates_even_below_iou_threshold():
    large = rectangle(80, 80, 32, 32)
    small = rectangle(81, 81, 30, 30)
    # IoU is 900/1024 < .9 but the smaller mask is fully contained.
    result = run(np.stack([large, small]), confidences=np.asarray([0.9, 0.8]))
    assert len(result.candidates) == 1
    assert result.masks[1].reason == "duplicate"
    assert result.masks[1].duplicate_of_raw_index == 0


def test_top16_cap_is_applied_before_lifting_and_fully_accounted():
    masks = []
    for index in range(17):
        row = index // 8
        col = index % 8
        masks.append(rectangle(20 + row * 50, 20 + col * 70, 24, 24))
    confidences = np.linspace(0.99, 0.50, 17, dtype=np.float64)
    result = run(np.stack(masks), confidences=confidences)

    assert len(result.candidates) == 16
    assert [item.raw_index for item in result.candidates] == list(range(16))
    assert not result.masks[16].lifted
    assert result.masks[16].reason == "top_k_cap"
    assert result.diagnostics.lifting_eligible_count == 16
    assert result.diagnostics.cap_rejected_count == 1
    assert result.diagnostics.rejection_counts == {"selected": 16, "top_k_cap": 1}


def test_top16_boundary_does_not_backfill_after_geometry_failure():
    masks = []
    for index in range(17):
        row = index // 8
        col = index % 8
        masks.append(rectangle(20 + row * 50, 20 + col * 70, 24, 24))
    depth = np.full((HEIGHT, WIDTH), 2.0, dtype=np.float32)
    # Keep the highest-confidence mask valid in 2D, but place it on a very near
    # constant-depth patch so its 2 cm lift has fewer than 16 unique voxels.
    # Raw index 16 remains a cap drop rather than crossing the frozen boundary.
    depth[20:44, 20:44] = 0.10
    confidences = np.linspace(0.99, 0.50, 17, dtype=np.float64)
    result = run(np.stack(masks), confidences=confidences, depth=depth)
    assert result.masks[0].reason == "too_few_unique_voxels"
    assert result.masks[16].reason == "top_k_cap"
    assert len(result.candidates) == 15


def test_full_mask_lift_applies_margin_depth_edges_voxels_cap_and_aabb_floor():
    mask = rectangle(100, 100, 100, 100)
    depth = np.full((HEIGHT, WIDTH), 2.0, dtype=np.float32)
    # Both endpoints of this four-neighbour discontinuity are removed.
    depth[:, 150:] = 2.30
    result = run(np.stack([mask]), depth=depth)
    candidate = result.candidates[0]

    # Erosion yields 98x98; the jump removes interior columns 149 and 150.
    assert candidate.support_pixel_count == 98 * 96
    assert candidate.voxel_count > 2048
    assert candidate.stored_point_count == 2048
    assert candidate.points_world.shape == (2048, 3)
    assert candidate.voxel_keys.shape == (2048, 3)
    assert len(candidate.points_sha256) == 64
    assert len(np.unique(candidate.voxel_keys, axis=0)) == 2048
    assert np.all(candidate.world_extent >= 0.02)
    np.testing.assert_allclose(
        candidate.world_q98 - candidate.world_q02, candidate.world_extent
    )


def test_too_few_unique_voxels_is_an_ordinary_diagnosed_rejection():
    mask = rectangle(100, 100, 20, 20)
    result = run(np.stack([mask]), focal=100_000.0)
    assert not result.candidates
    assert result.masks[0].support_pixel_count == 18 * 18
    assert result.masks[0].voxel_count < 16
    assert result.masks[0].reason == "too_few_unique_voxels"


def test_inputs_are_unchanged_outputs_are_deeply_readonly_and_pose_is_used():
    masks = np.stack([rectangle(100, 100, 40, 40)])
    confidences = np.asarray([0.8], dtype=np.float32)
    depth = np.full((HEIGHT, WIDTH), 2.0, dtype=np.float32)
    boxes = np.empty((0, 4), dtype=np.float32)
    intrinsics, pose = camera_inputs()
    pose[:3, 3] = [1.0, -2.0, 3.0]
    values = (masks, confidences, depth, boxes, intrinsics, pose)
    originals = tuple(value.copy() for value in values)

    result = f0.select_and_lift_residual_masks(
        masks=masks,
        confidences=confidences,
        depth_m=depth,
        explained_boxes_xyxy=boxes,
        intrinsics=intrinsics,
        camera_to_world=pose,
    )

    for value, original in zip(values, originals):
        np.testing.assert_array_equal(value, original)
    candidate = result.candidates[0]
    assert candidate.world_center[2] > 4.0
    for array in (
        candidate.tight_box_xyxy,
        candidate.points_world,
        candidate.voxel_keys,
        candidate.world_q02,
        candidate.world_q98,
        candidate.world_center,
        candidate.world_extent,
        result.masks[0].tight_box_xyxy,
    ):
        assert not array.flags.writeable
        with pytest.raises(ValueError):
            array.setflags(write=True)


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"masks": np.zeros((1, 479, 640), dtype=bool)}, "shape"),
        ({"masks": np.full((1, 480, 640), 0.5)}, "binary"),
        ({"confidences": np.asarray([1.1])}, r"\[0,1\]"),
        ({"depth_m": np.zeros((480, 639))}, "depth_m"),
        ({"explained_boxes_xyxy": np.zeros((1, 5))}, "shape"),
        (
            {"explained_boxes_xyxy": np.asarray([[2.0, 0.0, 1.0, 3.0]])},
            "x2>x1",
        ),
        ({"intrinsics": np.eye(2)}, "intrinsics"),
        ({"camera_to_world": np.zeros((4, 4))}, "camera_to_world"),
    ],
)
def test_invalid_structural_inputs_fail_closed(override, message):
    intrinsics, pose = camera_inputs()
    values = {
        "masks": np.stack([rectangle(100, 100, 40, 40)]),
        "confidences": np.asarray([0.8]),
        "depth_m": np.full((HEIGHT, WIDTH), 2.0),
        "explained_boxes_xyxy": np.empty((0, 4)),
        "intrinsics": intrinsics,
        "camera_to_world": pose,
    }
    values.update(override)
    with pytest.raises(ValueError, match=message):
        f0.select_and_lift_residual_masks(**values)


def test_public_core_exposes_no_gt_training_rgb_semantics_or_tracking_inputs():
    parameters = inspect.signature(f0.select_and_lift_residual_masks).parameters
    assert tuple(parameters) == (
        "masks",
        "confidences",
        "depth_m",
        "explained_boxes_xyxy",
        "intrinsics",
        "camera_to_world",
    )
    forbidden = {
        "gt",
        "ground_truth",
        "labels",
        "rgb",
        "model",
        "checkpoint",
        "clip",
        "semantic",
        "history",
        "tracks",
    }
    assert not (forbidden & set(parameters))
