import inspect
from types import MappingProxyType

import numpy as np
import pytest

from boxfusion import udc_mobilesam as udc


def camera_inputs(height=128, width=160, focal=100.0):
    intrinsics = np.asarray(
        [[focal, 0.0, width / 2.0], [0.0, focal, height / 2.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    return intrinsics, np.eye(4, dtype=np.float64)


def paint_grid_component(depth, top, left, height, width, base=1.5):
    for row in range(height):
        for col in range(width):
            # Both stride-neighbour changes remain well below the 0.15 m edge
            # threshold, while q02/q98 retains non-degenerate world depth.
            depth[(top + row) * 4, (left + col) * 4] = base + 0.02 * row + 0.01 * col


def run(depth, boxes=None, focal=100.0, pose=None):
    intrinsics, default_pose = camera_inputs(*depth.shape, focal=focal)
    return udc.generate_udc_prompts(
        depth_m=depth,
        explained_boxes_xyxy=(
            np.empty((0, 4), dtype=np.float32) if boxes is None else boxes
        ),
        intrinsics=intrinsics,
        camera_to_world=default_pose if pose is None else pose,
    )


def test_policy_is_frozen_to_preregistered_values():
    assert isinstance(udc.POLICY, MappingProxyType)
    assert udc.POLICY == {
        "pixel_stride": 4,
        "min_depth_m": 0.1,
        "max_depth_m": 6.0,
        "depth_edge_jump_m": 0.15,
        "explained_box_expand_px": 4.0,
        "component_connectivity": 8,
        "min_component_grid_pixels": 24,
        "max_component_grid_pixels": 5000,
        "min_bbox_grid_size": 5,
        "reject_touches_at_least_borders": 2,
        "world_quantiles": (0.02, 0.98),
        "min_world_extent_m": 0.05,
        "max_world_extent_m": 2.5,
        "max_world_diagonal_m": 3.0,
        "max_world_volume_m3": 4.0,
        "top_k": 2,
        "voxel_size_m": 0.05,
    }
    with pytest.raises(TypeError):
        udc.POLICY["top_k"] = 3


def test_largest_top2_are_stable_and_every_component_is_diagnosed():
    depth = np.zeros((128, 160), dtype=np.float32)
    paint_grid_component(depth, 4, 4, 8, 8, 1.4)    # 64 pixels
    paint_grid_component(depth, 4, 24, 7, 8, 1.8)   # 56 pixels
    paint_grid_component(depth, 20, 5, 6, 8, 2.2)   # 48 pixels

    first = run(depth)
    second = run(depth.copy())

    assert len(first.prompts) == 2
    assert [item.grid_pixel_count for item in first.prompts] == [64, 56]
    assert first.prompts[0].source_pixels_yx.shape == (64, 2)
    np.testing.assert_array_equal(first.prompts[0].source_pixels_yx[0], [16, 16])
    np.testing.assert_array_equal(first.prompts[0].source_pixels_yx[-1], [44, 44])
    assert first.prompts[0].voxel_keys.ndim == 2
    assert first.prompts[0].voxel_keys.shape[1] == 3
    assert first.prompts[0].voxel_keys.dtype == np.int64
    assert len(np.unique(first.prompts[0].voxel_keys, axis=0)) == len(
        first.prompts[0].voxel_keys
    )
    np.testing.assert_array_equal(first.boxes_xyxy, second.boxes_xyxy)
    np.testing.assert_array_equal(first.boxes_xyxy, [[16, 16, 47, 47], [96, 16, 127, 43]])
    assert first.diagnostics.component_count == 3
    assert first.diagnostics.eligible_component_count == 3
    assert first.diagnostics.selected_component_count == 2
    assert first.diagnostics.rejection_counts == {"selected": 2, "top_k_cap": 1}
    assert sum(item.selected for item in first.components) == 2


def test_equal_area_tie_break_is_top_then_left_not_input_or_label_accident():
    depth = np.zeros((128, 160), dtype=np.float32)
    paint_grid_component(depth, 18, 22, 6, 6, 1.4)
    paint_grid_component(depth, 4, 24, 6, 6, 1.6)
    paint_grid_component(depth, 4, 5, 6, 6, 1.8)
    result = run(depth)
    np.testing.assert_array_equal(
        result.boxes_xyxy,
        [[20, 16, 43, 39], [96, 16, 119, 39]],
    )


def test_explained_box_expands_four_pixels_in_depth_coordinates():
    depth = np.zeros((128, 160), dtype=np.float32)
    paint_grid_component(depth, 8, 10, 6, 6, 1.5)
    # Component sample centres cover x=[40,60], y=[32,52].  Four-pixel
    # expansion makes this deliberately contracted box cover all centres.
    box = np.asarray([[44.0, 36.0, 56.0, 48.0]], dtype=np.float64)
    result = run(depth, box)
    assert not result.prompts
    assert result.diagnostics.explained_valid_grid_pixels == 36
    assert result.diagnostics.residual_grid_pixels == 0
    assert result.diagnostics.component_count == 0


@pytest.mark.parametrize("split_column", [61, 62, 63, 64])
def test_depth_domain_and_all_stride_phase_jump_edges_are_removed(split_column):
    depth = np.full((128, 160), 2.0, dtype=np.float32)
    depth[:, split_column:] = 2.30
    depth[0, 0] = 0.09
    depth[4, 0] = 6.01
    result = run(depth)

    # Every phase of a discontinuity relative to the stride-four lattice must
    # reject both neighbouring sampled columns.  In particular, split 62 lies
    # between samples 60 and 64 and neither endpoint was marked by the old
    # undilated full-resolution endpoint mask.
    assert result.diagnostics.edge_rejected_grid_pixels >= 64
    assert result.diagnostics.valid_depth_grid_pixels == 1280 - 2
    assert result.diagnostics.residual_grid_pixels <= 1280 - 2 - 64
    assert result.diagnostics.rejection_counts["touches_multiple_borders"] >= 1


def test_coarse_grid_barrier_marks_both_samples_for_nonmultiple_jump():
    sampled = np.full((3, 5), 2.0, dtype=np.float64)
    sampled[:, 2:] = 2.30
    edge = udc._stride_grid_edge_mask(sampled)
    np.testing.assert_array_equal(edge[:, 1:3], True)
    assert not edge[:, 0].any()
    assert not edge[:, 3:].any()


def test_component_pixel_bbox_border_and_world_extent_filters():
    depth = np.zeros((160, 200), dtype=np.float32)
    paint_grid_component(depth, 7, 12, 4, 6, 1.4)   # bbox height 4
    paint_grid_component(depth, 0, 0, 6, 6, 1.8)    # touches top and left
    paint_grid_component(depth, 15, 20, 3, 5, 2.2)  # fewer than 24
    paint_grid_component(depth, 25, 30, 6, 6, 2.5)  # valid pixels, tiny metric xy

    result = run(depth, focal=10_000.0)
    reasons = {item.reason for item in result.components}
    assert "bbox_too_small" in reasons
    assert "touches_multiple_borders" in reasons
    assert "too_few_grid_pixels" in reasons
    assert "world_extent_too_small" in reasons
    assert not result.prompts


def test_above_5000_grid_pixels_is_rejected_before_metric_filters():
    depth = np.zeros((480, 480), dtype=np.float32)
    paint_grid_component(depth, 10, 10, 71, 71, 1.0)
    result = run(depth, focal=500.0)
    assert len(result.components) == 1
    assert result.components[0].grid_pixel_count == 5041
    assert result.components[0].reason == "too_many_grid_pixels"
    assert not result.prompts


def test_inputs_are_unchanged_output_arrays_are_deeply_readonly_and_pose_is_used():
    depth = np.zeros((128, 160), dtype=np.float32)
    paint_grid_component(depth, 6, 8, 7, 7, 1.5)
    boxes = np.empty((0, 4), dtype=np.float64)
    intrinsics, pose = camera_inputs(*depth.shape)
    pose[:3, 3] = [1.0, -2.0, 3.0]
    originals = [value.copy() for value in (depth, boxes, intrinsics, pose)]

    result = udc.generate_residual_box_prompts(
        depth_m=depth,
        explained_boxes_xyxy=boxes,
        intrinsics=intrinsics,
        camera_to_world=pose,
    )

    for value, original in zip((depth, boxes, intrinsics, pose), originals):
        np.testing.assert_array_equal(value, original)
    assert len(result.prompts) == 1
    assert result.prompts[0].world_q02[2] > 4.0
    for array in (
        result.boxes_xyxy,
        result.prompts[0].box_xyxy,
        result.prompts[0].source_pixels_yx,
        result.prompts[0].voxel_keys,
        result.prompts[0].world_extent,
        result.components[0].grid_bbox_xywh,
        result.components[0].depth_q02_q98_m,
    ):
        assert not array.flags.writeable
        with pytest.raises(ValueError):
            array.setflags(write=True)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"depth_m": [[1.0]]}, "depth_m"),
        ({"depth_m": np.zeros((2, 2, 1))}, "depth_m"),
        ({"explained_boxes_xyxy": np.zeros((2, 5))}, "shape"),
        ({"explained_boxes_xyxy": np.asarray([[2.0, 0.0, 1.0, 3.0]])}, "x2>x1"),
        ({"intrinsics": np.eye(2)}, "intrinsics"),
        ({"camera_to_world": np.zeros((4, 4))}, "camera_to_world"),
    ],
)
def test_invalid_input_structure_fails_closed(overrides, message):
    depth = np.zeros((32, 32), dtype=np.float32)
    intrinsics, pose = camera_inputs(*depth.shape)
    values = dict(
        depth_m=depth,
        explained_boxes_xyxy=np.empty((0, 4), dtype=np.float32),
        intrinsics=intrinsics,
        camera_to_world=pose,
    )
    values.update(overrides)
    with pytest.raises(ValueError, match=message):
        udc.generate_udc_prompts(**values)


def test_public_core_has_no_gt_training_rgb_or_model_inputs():
    parameters = inspect.signature(udc.generate_residual_box_prompts).parameters
    assert tuple(parameters) == (
        "depth_m",
        "explained_boxes_xyxy",
        "intrinsics",
        "camera_to_world",
    )
    assert not ({"gt", "ground_truth", "labels", "rgb", "model", "checkpoint"} & set(parameters))
