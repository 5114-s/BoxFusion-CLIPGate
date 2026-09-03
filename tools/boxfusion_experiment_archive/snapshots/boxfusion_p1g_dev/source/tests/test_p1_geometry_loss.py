"""Contracts for the independent P1G geometry objective."""

from __future__ import annotations

import math

import pytest
import torch

from boxfusion.p1_geometry_loss import (
    P1G_DEFAULT_ADAPTER_EPSILON,
    decode_bounded_aabb,
    decode_p1g_residual_aabb,
    p1_geometry_aligned_loss,
    p1_geometry_loss,
    p1g_residual_geometry_aligned_loss,
    p1s_raw_to_bounded_logits,
    paired_aabb_giou,
    paired_aabb_iou,
    world_aabb_to_aligned_aabb,
)


def _old_p1s_clip_exp_decode(
    raw: torch.Tensor,
    anchors: torch.Tensor,
    *,
    max_center_offset: float,
    min_box_extent: float,
    max_box_extent: float,
) -> torch.Tensor:
    center = anchors + torch.clamp(
        raw[:, :3],
        min=-max_center_offset,
        max=max_center_offset,
    )
    extent = torch.exp(
        torch.clamp(
            raw[:, 3:],
            min=math.log(min_box_extent),
            max=math.log(max_box_extent),
        )
    )
    return torch.cat((center, extent), dim=1)


def test_zero_correction_preserves_old_p1s_decode_away_from_bounds():
    raw = torch.tensor(
        [
            [-0.75, 0.20, 0.95, math.log(0.1), 0.0, math.log(3.9)],
            [0.00, -0.40, 0.60, math.log(0.5), math.log(2.0), 0.3],
        ],
        dtype=torch.float64,
    )
    anchors = torch.tensor(
        [[2.0, -1.0, 0.5], [-3.0, 4.0, 1.0]], dtype=torch.float64
    )
    correction = torch.zeros_like(raw)
    expected = _old_p1s_clip_exp_decode(
        raw,
        anchors,
        max_center_offset=1.0,
        min_box_extent=0.08,
        max_box_extent=4.0,
    )
    observed = decode_p1g_residual_aabb(
        raw,
        correction,
        anchors,
        max_center_offset=1.0,
        min_box_extent=0.08,
        max_box_extent=4.0,
    )
    torch.testing.assert_close(observed, expected, rtol=1e-12, atol=1e-12)


def test_zero_correction_preserves_old_p1s_boundary_within_adapter_epsilon():
    epsilon = P1G_DEFAULT_ADAPTER_EPSILON
    minimum, maximum, offset = 0.08, 4.0, 1.5
    raw = torch.tensor(
        [[-100.0, 100.0, 0.0, -100.0, 100.0, math.log(minimum)]],
        dtype=torch.float64,
    )
    anchors = torch.zeros((1, 3), dtype=torch.float64)
    expected = _old_p1s_clip_exp_decode(
        raw,
        anchors,
        max_center_offset=offset,
        min_box_extent=minimum,
        max_box_extent=maximum,
    )
    observed = decode_p1g_residual_aabb(
        raw,
        torch.zeros_like(raw),
        anchors,
        max_center_offset=offset,
        min_box_extent=minimum,
        max_box_extent=maximum,
        adapter_epsilon=epsilon,
    )
    center_error_normalized = torch.abs(
        (observed[:, :3] - expected[:, :3]) / offset
    )
    log_range = math.log(maximum) - math.log(minimum)
    log_extent_error_normalized = torch.abs(
        (torch.log(observed[:, 3:]) - torch.log(expected[:, 3:]))
        / log_range
    )
    assert float(center_error_normalized.max()) <= epsilon * 1.01
    assert float(log_extent_error_normalized.max()) <= epsilon * 1.01
    assert bool(torch.isfinite(p1s_raw_to_bounded_logits(raw)).all())


def test_residual_aligned_loss_backpropagates_only_through_correction():
    frozen = torch.tensor(
        [[0.2, -0.1, 0.3, 0.0, math.log(0.5), math.log(2.0)]],
        dtype=torch.float64,
    )
    correction = torch.zeros_like(frozen, requires_grad=True)
    anchors = torch.zeros((1, 3), dtype=torch.float64)
    alignment = torch.eye(4, dtype=torch.float64)
    target = torch.tensor(
        [[0.0, 0.0, 0.0, 1.0, 1.0, 1.0]], dtype=torch.float64
    )
    loss = p1g_residual_geometry_aligned_loss(
        frozen,
        correction,
        anchors,
        target,
        alignment,
    )
    loss.backward()
    assert correction.grad is not None
    assert bool(torch.isfinite(correction.grad).all())
    assert bool(torch.any(correction.grad != 0.0))
    assert frozen.grad is None


def test_bounded_decode_has_expected_midpoint_and_limits():
    raw = torch.tensor(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [100.0, -100.0, 100.0, -100.0, 100.0, -100.0],
        ],
        dtype=torch.float64,
    )
    anchors = torch.tensor(
        [[1.0, 2.0, 3.0], [-2.0, 0.5, 7.0]],
        dtype=torch.float64,
    )
    decoded = decode_bounded_aabb(
        raw,
        anchors,
        max_center_offset=2.0,
        min_box_extent=0.25,
        max_box_extent=4.0,
    )

    torch.testing.assert_close(decoded[0, :3], anchors[0])
    torch.testing.assert_close(
        decoded[0, 3:],
        torch.full((3,), 1.0, dtype=torch.float64),
    )
    assert bool(torch.all(decoded[:, 3:] >= 0.25))
    assert bool(torch.all(decoded[:, 3:] <= 4.0))
    torch.testing.assert_close(
        decoded[1, :3],
        anchors[1] + torch.tensor([2.0, -2.0, 2.0]),
    )
    torch.testing.assert_close(
        decoded[1, 3:],
        torch.tensor([0.25, 4.0, 0.25], dtype=torch.float64),
    )


def test_paired_iou_and_giou_known_values():
    first = torch.tensor(
        [[0.0, 0.0, 0.0, 1.0, 1.0, 1.0]], dtype=torch.float64
    )
    identical = first.clone()
    disjoint = torch.tensor(
        [[2.0, 0.0, 0.0, 1.0, 1.0, 1.0]], dtype=torch.float64
    )

    torch.testing.assert_close(
        paired_aabb_iou(first, identical), torch.ones(1, dtype=torch.float64)
    )
    torch.testing.assert_close(
        paired_aabb_giou(first, identical),
        torch.ones(1, dtype=torch.float64),
    )
    torch.testing.assert_close(
        paired_aabb_iou(first, disjoint),
        torch.zeros(1, dtype=torch.float64),
    )
    torch.testing.assert_close(
        paired_aabb_giou(first, disjoint),
        torch.tensor([-1.0 / 3.0], dtype=torch.float64),
    )


def test_geometry_loss_is_zero_for_exact_decoded_box():
    raw = torch.zeros((2, 6), dtype=torch.float64)
    anchors = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, -2.0, 3.0]], dtype=torch.float64
    )
    target = torch.cat(
        (anchors, torch.ones((2, 3), dtype=torch.float64)), dim=1
    )
    loss = p1_geometry_loss(
        raw,
        anchors,
        target,
        min_box_extent=0.25,
        max_box_extent=4.0,
    )
    torch.testing.assert_close(
        loss, torch.zeros((), dtype=torch.float64), atol=1e-12, rtol=0.0
    )


def test_geometry_loss_is_differentiable_with_finite_nonzero_gradients():
    torch.manual_seed(23)
    raw = torch.randn((4, 6), dtype=torch.float64, requires_grad=True)
    anchors = torch.randn((4, 3), dtype=torch.float64)
    targets = torch.cat(
        (
            anchors + 0.2 * torch.randn((4, 3), dtype=torch.float64),
            0.4 + 1.6 * torch.rand((4, 3), dtype=torch.float64),
        ),
        dim=1,
    )
    loss = p1_geometry_loss(
        raw,
        anchors,
        targets,
        max_center_offset=1.0,
        min_box_extent=0.08,
        max_box_extent=4.0,
    )
    loss.backward()

    assert loss.ndim == 0
    assert raw.grad is not None
    assert bool(torch.isfinite(raw.grad).all())
    assert bool(torch.any(raw.grad != 0.0))


def test_world_aabb_alignment_matches_corner_enclosure_formula():
    angle = math.pi / 4.0
    cosine, sine = math.cos(angle), math.sin(angle)
    alignment = torch.tensor(
        [
            [cosine, -sine, 0.0, 1.0],
            [sine, cosine, 0.0, -2.0],
            [0.0, 0.0, 1.0, 0.5],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=torch.float64,
    )
    boxes = torch.tensor(
        [[2.0, 3.0, 4.0, 2.0, 1.0, 3.0]], dtype=torch.float64
    )
    transformed = world_aabb_to_aligned_aabb(boxes, alignment)
    expected_center = (
        alignment[:3, :3] @ boxes[0, :3] + alignment[:3, 3]
    )
    expected_extent = torch.abs(alignment[:3, :3]) @ boxes[0, 3:]
    torch.testing.assert_close(transformed[0, :3], expected_center)
    torch.testing.assert_close(transformed[0, 3:], expected_extent)


def test_aligned_geometry_loss_is_zero_for_exact_transformed_box():
    raw = torch.zeros((1, 6), dtype=torch.float64, requires_grad=True)
    anchor = torch.tensor([[2.0, 3.0, 4.0]], dtype=torch.float64)
    alignment = torch.tensor(
        [
            [0.0, -1.0, 0.0, 1.0],
            [1.0, 0.0, 0.0, -2.0],
            [0.0, 0.0, 1.0, 0.5],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=torch.float64,
    )
    world = decode_bounded_aabb(
        raw,
        anchor,
        min_box_extent=0.25,
        max_box_extent=4.0,
    )
    target = world_aabb_to_aligned_aabb(world.detach(), alignment)
    loss = p1_geometry_aligned_loss(
        raw,
        anchor,
        target,
        alignment,
        min_box_extent=0.25,
        max_box_extent=4.0,
    )
    torch.testing.assert_close(
        loss, torch.zeros((), dtype=torch.float64), atol=1e-12, rtol=0.0
    )
    loss.backward()
    assert raw.grad is not None
    assert bool(torch.isfinite(raw.grad).all())


def test_empty_inputs_have_stable_shapes_and_differentiable_zero():
    raw = torch.empty((0, 6), requires_grad=True)
    anchors = torch.empty((0, 3))
    targets = torch.empty((0, 6))

    decoded = decode_bounded_aabb(raw, anchors)
    assert decoded.shape == (0, 6)
    assert paired_aabb_iou(decoded, targets).shape == (0,)
    assert paired_aabb_giou(decoded, targets).shape == (0,)
    assert p1_geometry_loss(
        raw, anchors, targets, reduction="none"
    ).shape == (0,)

    loss = p1_geometry_loss(raw, anchors, targets, reduction="mean")
    assert loss.shape == ()
    assert float(loss) == 0.0
    loss.backward()
    assert raw.grad is not None
    assert raw.grad.shape == raw.shape


@pytest.mark.parametrize(
    ("operation", "error_type"),
    [
        (
            lambda: decode_bounded_aabb(
                torch.zeros((2, 5)), torch.zeros((2, 3))
            ),
            ValueError,
        ),
        (
            lambda: decode_bounded_aabb(
                torch.zeros((2, 6)), torch.zeros((1, 3))
            ),
            ValueError,
        ),
        (
            lambda: decode_bounded_aabb(
                torch.zeros((2, 6), dtype=torch.float64),
                torch.zeros((2, 3), dtype=torch.float32),
            ),
            TypeError,
        ),
        (
            lambda: decode_bounded_aabb(
                torch.zeros((1, 6)), torch.full((1, 3), math.nan)
            ),
            ValueError,
        ),
        (
            lambda: decode_bounded_aabb(
                torch.zeros((1, 6)), torch.zeros((1, 3)), max_center_offset=0
            ),
            ValueError,
        ),
        (
            lambda: decode_bounded_aabb(
                torch.zeros((1, 6)),
                torch.zeros((1, 3)),
                min_box_extent=1.0,
                max_box_extent=1.0,
            ),
            ValueError,
        ),
        (
            lambda: paired_aabb_iou(
                torch.tensor([[0.0, 0.0, 0.0, -1.0, 1.0, 1.0]]),
                torch.ones((1, 6)),
            ),
            ValueError,
        ),
        (
            lambda: p1_geometry_loss(
                torch.zeros((1, 6)),
                torch.zeros((1, 3)),
                torch.tensor([[0.0, 0.0, 0.0, 1.0, 1.0, 0.0]]),
            ),
            ValueError,
        ),
        (
            lambda: p1_geometry_loss(
                torch.zeros((1, 6)),
                torch.zeros((1, 3)),
                torch.ones((1, 6)),
                smooth_l1_weight=-0.1,
            ),
            ValueError,
        ),
        (
            lambda: p1_geometry_loss(
                torch.zeros((1, 6)),
                torch.zeros((1, 3)),
                torch.ones((1, 6)),
                reduction="median",
            ),
            ValueError,
        ),
        (
            lambda: world_aabb_to_aligned_aabb(
                torch.ones((1, 6)),
                torch.diag(torch.tensor([2.0, 1.0, 1.0, 1.0])),
            ),
            ValueError,
        ),
    ],
)
def test_strict_input_validation(operation, error_type):
    with pytest.raises(error_type):
        operation()
