"""Differentiable, decoder-consistent geometry objectives for P1G.

All boxes in this module use the axis-aligned centre-size convention
``[cx, cy, cz, dx, dy, dz]``.  The bounded decoder deliberately applies the
same constraints during training and inference:

* centre offsets are ``max_center_offset * tanh(raw_offset)``;
* log extents are mapped into ``[log(min_extent), log(max_extent)]`` with a
  sigmoid.

The functions are dependency-free apart from PyTorch and operate on paired
rows rather than constructing an ``N x M`` IoU matrix.
"""

from __future__ import annotations

import math
from typing import Literal

import torch
from torch.nn import functional as F


Reduction = Literal["none", "mean", "sum"]
P1G_DEFAULT_ADAPTER_EPSILON = 1e-6


def _require_float_tensor(
    value: torch.Tensor,
    *,
    name: str,
    columns: int,
) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if value.ndim != 2 or value.shape[1] != columns:
        raise ValueError(
            f"{name} must have shape [N,{columns}], got {tuple(value.shape)}"
        )
    if not value.is_floating_point():
        raise TypeError(f"{name} must use a floating dtype")
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} must contain only finite values")
    return value


def _require_same_tensor_contract(
    first: torch.Tensor,
    second: torch.Tensor,
    *,
    first_name: str,
    second_name: str,
) -> None:
    if len(first) != len(second):
        raise ValueError(
            f"{first_name} and {second_name} must have the same row count"
        )
    if first.device != second.device:
        raise ValueError(
            f"{first_name} and {second_name} must use the same device"
        )
    if first.dtype != second.dtype:
        raise TypeError(
            f"{first_name} and {second_name} must use the same dtype"
        )


def _finite_scalar(
    value: float,
    *,
    name: str,
    positive: bool = False,
    non_negative: bool = False,
) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise TypeError(f"{name} must be a real scalar") from error
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    if positive and result <= 0.0:
        raise ValueError(f"{name} must be positive")
    if non_negative and result < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return result


def _validate_center_size_boxes(
    boxes: torch.Tensor, *, name: str
) -> torch.Tensor:
    values = _require_float_tensor(boxes, name=name, columns=6)
    if bool(torch.any(values[:, 3:] <= 0.0)):
        raise ValueError(f"{name} must have strictly positive extents")
    return values


def decode_bounded_aabb(
    raw_regression: torch.Tensor,
    anchor_centers: torch.Tensor,
    *,
    max_center_offset: float = 1.0,
    min_box_extent: float = 0.08,
    max_box_extent: float = 4.0,
) -> torch.Tensor:
    """Decode raw P1 geometry into bounded centre-size AABBs.

    Args:
        raw_regression: Floating ``[N,6]`` tensor.  The first three values are
            unconstrained centre-offset logits and the final three are
            unconstrained log-extent logits.
        anchor_centers: Floating ``[N,3]`` world-frame voxel centres.
        max_center_offset: Symmetric per-axis offset bound in metres.
        min_box_extent: Inclusive lower extent bound in metres.
        max_box_extent: Inclusive upper extent bound in metres.
    """

    raw = _require_float_tensor(
        raw_regression, name="raw_regression", columns=6
    )
    anchors = _require_float_tensor(
        anchor_centers, name="anchor_centers", columns=3
    )
    _require_same_tensor_contract(
        raw,
        anchors,
        first_name="raw_regression",
        second_name="anchor_centers",
    )
    offset_bound = _finite_scalar(
        max_center_offset, name="max_center_offset", positive=True
    )
    minimum = _finite_scalar(
        min_box_extent, name="min_box_extent", positive=True
    )
    maximum = _finite_scalar(
        max_box_extent, name="max_box_extent", positive=True
    )
    if maximum <= minimum:
        raise ValueError("max_box_extent must exceed min_box_extent")

    center = anchors + offset_bound * torch.tanh(raw[:, :3])
    log_minimum = raw.new_tensor(math.log(minimum))
    log_range = raw.new_tensor(math.log(maximum) - math.log(minimum))
    log_extent = log_minimum + torch.sigmoid(raw[:, 3:]) * log_range
    extent = torch.exp(log_extent)
    return torch.cat((center, extent), dim=1)


def p1s_raw_to_bounded_logits(
    frozen_p1s_raw_regression: torch.Tensor,
    *,
    max_center_offset: float = 1.0,
    min_box_extent: float = 0.08,
    max_box_extent: float = 4.0,
    adapter_epsilon: float = P1G_DEFAULT_ADAPTER_EPSILON,
) -> torch.Tensor:
    """Map the frozen P1S clip/exp output into P1G bounded logits.

    P1S decodes centre deltas with ``clip(raw, -M, M)`` and extents with
    ``exp(clip(raw_log_extent, log(min), log(max)))``.  P1G applies a learned
    correction in the bounded tanh/sigmoid logit space.  This adapter is the
    function-preserving bridge between those two parameterizations:

    * a zero correction reproduces the old P1S decoder exactly away from a
      bound;
    * at a hard bound it differs only by ``adapter_epsilon`` in normalized
      decoder space, keeping inverse tanh/logit finite.

    The frozen P1S values are runtime inputs, not trainable parameters.
    """

    raw = _require_float_tensor(
        frozen_p1s_raw_regression,
        name="frozen_p1s_raw_regression",
        columns=6,
    )
    offset_bound = _finite_scalar(
        max_center_offset, name="max_center_offset", positive=True
    )
    minimum = _finite_scalar(
        min_box_extent, name="min_box_extent", positive=True
    )
    maximum = _finite_scalar(
        max_box_extent, name="max_box_extent", positive=True
    )
    if maximum <= minimum:
        raise ValueError("max_box_extent must exceed min_box_extent")
    epsilon = _finite_scalar(
        adapter_epsilon, name="adapter_epsilon", positive=True
    )
    if epsilon >= 0.5:
        raise ValueError("adapter_epsilon must be smaller than 0.5")

    normalized_center = torch.clamp(
        raw[:, :3] / offset_bound,
        min=-1.0 + epsilon,
        max=1.0 - epsilon,
    )
    center_logits = torch.atanh(normalized_center)

    log_minimum = raw.new_tensor(math.log(minimum))
    log_range = raw.new_tensor(math.log(maximum) - math.log(minimum))
    clipped_log_extent = torch.clamp(
        raw[:, 3:],
        min=float(math.log(minimum)),
        max=float(math.log(maximum)),
    )
    normalized_log_extent = torch.clamp(
        (clipped_log_extent - log_minimum) / log_range,
        min=epsilon,
        max=1.0 - epsilon,
    )
    extent_logits = torch.logit(normalized_log_extent)
    return torch.cat((center_logits, extent_logits), dim=1)


def decode_p1g_residual_aabb(
    frozen_p1s_raw_regression: torch.Tensor,
    residual_correction: torch.Tensor,
    anchor_centers: torch.Tensor,
    *,
    max_center_offset: float = 1.0,
    min_box_extent: float = 0.08,
    max_box_extent: float = 4.0,
    adapter_epsilon: float = P1G_DEFAULT_ADAPTER_EPSILON,
) -> torch.Tensor:
    """Decode frozen P1S geometry plus a learned P1G logit correction.

    This is the single P1G v2 decode entry point shared by training,
    calibration, audit and runtime evaluation.  ``residual_correction`` must
    be zero at initialization, making the untrained P1G head function
    preserving with respect to the frozen P1S clip/exp decoder.
    """

    frozen = _require_float_tensor(
        frozen_p1s_raw_regression,
        name="frozen_p1s_raw_regression",
        columns=6,
    )
    correction = _require_float_tensor(
        residual_correction,
        name="residual_correction",
        columns=6,
    )
    anchors = _require_float_tensor(
        anchor_centers, name="anchor_centers", columns=3
    )
    _require_same_tensor_contract(
        frozen,
        correction,
        first_name="frozen_p1s_raw_regression",
        second_name="residual_correction",
    )
    _require_same_tensor_contract(
        frozen,
        anchors,
        first_name="frozen_p1s_raw_regression",
        second_name="anchor_centers",
    )
    base_logits = p1s_raw_to_bounded_logits(
        frozen,
        max_center_offset=max_center_offset,
        min_box_extent=min_box_extent,
        max_box_extent=max_box_extent,
        adapter_epsilon=adapter_epsilon,
    )
    return decode_bounded_aabb(
        base_logits + correction,
        anchors,
        max_center_offset=max_center_offset,
        min_box_extent=min_box_extent,
        max_box_extent=max_box_extent,
    )


def _center_size_to_minmax(boxes: torch.Tensor) -> torch.Tensor:
    half_extent = 0.5 * boxes[:, 3:]
    return torch.cat(
        (boxes[:, :3] - half_extent, boxes[:, :3] + half_extent),
        dim=1,
    )


def world_aabb_to_aligned_aabb(
    world_boxes: torch.Tensor,
    axis_alignment: torch.Tensor,
) -> torch.Tensor:
    """Transform world AABBs into enclosing aligned-frame AABBs.

    ``axis_alignment`` may be one shared ``[4,4]`` transform or one transform
    per box as ``[N,4,4]``.  For a rigid transform, the enclosing AABB has
    centre ``R c + t`` and extent ``abs(R) s``.  This is the differentiable
    equivalent of transforming all eight corners and taking min/max.
    """

    boxes = _validate_center_size_boxes(world_boxes, name="world_boxes")
    if not isinstance(axis_alignment, torch.Tensor):
        raise TypeError("axis_alignment must be a torch.Tensor")
    if not axis_alignment.is_floating_point():
        raise TypeError("axis_alignment must use a floating dtype")
    if axis_alignment.device != boxes.device:
        raise ValueError(
            "world_boxes and axis_alignment must use the same device"
        )
    if axis_alignment.dtype != boxes.dtype:
        raise TypeError(
            "world_boxes and axis_alignment must use the same dtype"
        )
    if axis_alignment.shape == (4, 4):
        transforms = axis_alignment.unsqueeze(0).expand(len(boxes), -1, -1)
    elif axis_alignment.shape == (len(boxes), 4, 4):
        transforms = axis_alignment
    else:
        raise ValueError(
            "axis_alignment must have shape [4,4] or [N,4,4]"
        )
    if not bool(torch.isfinite(transforms).all()):
        raise ValueError("axis_alignment must contain only finite values")
    if len(boxes) and not bool(
        torch.allclose(
            transforms[:, 3, :],
            transforms.new_tensor((0.0, 0.0, 0.0, 1.0))
            .reshape(1, 4)
            .expand(len(boxes), -1),
            rtol=1e-5,
            atol=1e-5,
        )
    ):
        raise ValueError("axis_alignment must be homogeneous")
    rotation = transforms[:, :3, :3]
    identity = torch.eye(
        3, dtype=boxes.dtype, device=boxes.device
    ).unsqueeze(0)
    gram = torch.bmm(rotation.transpose(1, 2), rotation)
    if len(boxes) and not bool(
        torch.allclose(
            gram,
            identity.expand(len(boxes), -1, -1),
            rtol=5e-4,
            atol=5e-4,
        )
    ):
        raise ValueError("axis_alignment rotation must be rigid")
    center = torch.bmm(
        rotation, boxes[:, :3].unsqueeze(-1)
    ).squeeze(-1) + transforms[:, :3, 3]
    extent = torch.bmm(
        torch.abs(rotation), boxes[:, 3:].unsqueeze(-1)
    ).squeeze(-1)
    return torch.cat((center, extent), dim=1)


def _paired_iou_terms(
    boxes_a: torch.Tensor,
    boxes_b: torch.Tensor,
    *,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    first = _validate_center_size_boxes(boxes_a, name="boxes_a")
    second = _validate_center_size_boxes(boxes_b, name="boxes_b")
    _require_same_tensor_contract(
        first,
        second,
        first_name="boxes_a",
        second_name="boxes_b",
    )
    epsilon = _finite_scalar(eps, name="eps", positive=True)
    first_minmax = _center_size_to_minmax(first)
    second_minmax = _center_size_to_minmax(second)

    intersection_extent = torch.clamp_min(
        torch.minimum(first_minmax[:, 3:], second_minmax[:, 3:])
        - torch.maximum(first_minmax[:, :3], second_minmax[:, :3]),
        0.0,
    )
    intersection = torch.prod(intersection_extent, dim=1)
    volume_first = torch.prod(first[:, 3:], dim=1)
    volume_second = torch.prod(second[:, 3:], dim=1)
    union = volume_first + volume_second - intersection
    iou = intersection / torch.clamp_min(union, epsilon)

    enclosure_extent = torch.clamp_min(
        torch.maximum(first_minmax[:, 3:], second_minmax[:, 3:])
        - torch.minimum(first_minmax[:, :3], second_minmax[:, :3]),
        0.0,
    )
    enclosure = torch.prod(enclosure_extent, dim=1)
    return iou, union, torch.clamp_min(enclosure, epsilon)


def paired_aabb_iou(
    boxes_a: torch.Tensor,
    boxes_b: torch.Tensor,
    *,
    eps: float = 1e-7,
) -> torch.Tensor:
    """Return elementwise 3D IoU for paired centre-size AABBs."""

    iou, _, _ = _paired_iou_terms(boxes_a, boxes_b, eps=eps)
    return iou


def paired_aabb_giou(
    boxes_a: torch.Tensor,
    boxes_b: torch.Tensor,
    *,
    eps: float = 1e-7,
) -> torch.Tensor:
    """Return elementwise generalized 3D IoU for paired AABBs."""

    iou, union, enclosure = _paired_iou_terms(
        boxes_a, boxes_b, eps=eps
    )
    return iou - (enclosure - union) / enclosure


def p1_geometry_loss(
    raw_regression: torch.Tensor,
    anchor_centers: torch.Tensor,
    target_boxes: torch.Tensor,
    *,
    max_center_offset: float = 1.0,
    min_box_extent: float = 0.08,
    max_box_extent: float = 4.0,
    smooth_l1_weight: float = 0.1,
    smooth_l1_beta: float = 0.1,
    eps: float = 1e-7,
    reduction: Reduction = "mean",
) -> torch.Tensor:
    """Compute decoder-consistent GIoU plus a small Smooth-L1 term.

    The Smooth-L1 component operates on decoded centre-size boxes, so neither
    component asks the network to reproduce values that inference later
    clips.  For an empty batch, ``mean`` and ``sum`` return a differentiable
    scalar zero while ``none`` returns an empty vector.
    """

    targets = _validate_center_size_boxes(
        target_boxes, name="target_boxes"
    )
    raw = _require_float_tensor(
        raw_regression, name="raw_regression", columns=6
    )
    anchors = _require_float_tensor(
        anchor_centers, name="anchor_centers", columns=3
    )
    _require_same_tensor_contract(
        raw,
        anchors,
        first_name="raw_regression",
        second_name="anchor_centers",
    )
    _require_same_tensor_contract(
        raw,
        targets,
        first_name="raw_regression",
        second_name="target_boxes",
    )
    weight = _finite_scalar(
        smooth_l1_weight,
        name="smooth_l1_weight",
        non_negative=True,
    )
    beta = _finite_scalar(
        smooth_l1_beta, name="smooth_l1_beta", positive=True
    )
    _finite_scalar(eps, name="eps", positive=True)
    if reduction not in {"none", "mean", "sum"}:
        raise ValueError("reduction must be one of: none, mean, sum")

    decoded = decode_bounded_aabb(
        raw,
        anchors,
        max_center_offset=max_center_offset,
        min_box_extent=min_box_extent,
        max_box_extent=max_box_extent,
    )
    giou_term = 1.0 - paired_aabb_giou(decoded, targets, eps=eps)
    smooth_term = F.smooth_l1_loss(
        decoded,
        targets,
        beta=beta,
        reduction="none",
    ).mean(dim=1)
    per_box = giou_term + weight * smooth_term
    if reduction == "none":
        return per_box
    if len(per_box) == 0:
        return raw.sum() * 0.0
    if reduction == "sum":
        return per_box.sum()
    return per_box.mean()


def p1_geometry_aligned_loss(
    raw_regression: torch.Tensor,
    anchor_centers_world: torch.Tensor,
    target_boxes_aligned: torch.Tensor,
    axis_alignment: torch.Tensor,
    *,
    max_center_offset: float = 1.0,
    min_box_extent: float = 0.08,
    max_box_extent: float = 4.0,
    smooth_l1_weight: float = 0.1,
    smooth_l1_beta: float = 0.1,
    eps: float = 1e-7,
    reduction: Reduction = "mean",
) -> torch.Tensor:
    """Optimize the exact aligned-AABB geometry used by ScanNet evaluation.

    The P1 runtime still emits a world-frame AABB and never reads
    ``axisAlignment``.  Alignment is used only by this offline train-only
    objective so the head is not supervised toward the inferior
    inverse-transform-then-enclose world target.
    """

    targets = _validate_center_size_boxes(
        target_boxes_aligned, name="target_boxes_aligned"
    )
    raw = _require_float_tensor(
        raw_regression, name="raw_regression", columns=6
    )
    anchors = _require_float_tensor(
        anchor_centers_world, name="anchor_centers_world", columns=3
    )
    _require_same_tensor_contract(
        raw,
        anchors,
        first_name="raw_regression",
        second_name="anchor_centers_world",
    )
    _require_same_tensor_contract(
        raw,
        targets,
        first_name="raw_regression",
        second_name="target_boxes_aligned",
    )
    weight = _finite_scalar(
        smooth_l1_weight,
        name="smooth_l1_weight",
        non_negative=True,
    )
    beta = _finite_scalar(
        smooth_l1_beta, name="smooth_l1_beta", positive=True
    )
    _finite_scalar(eps, name="eps", positive=True)
    if reduction not in {"none", "mean", "sum"}:
        raise ValueError("reduction must be one of: none, mean, sum")
    decoded_world = decode_bounded_aabb(
        raw,
        anchors,
        max_center_offset=max_center_offset,
        min_box_extent=min_box_extent,
        max_box_extent=max_box_extent,
    )
    decoded_aligned = world_aabb_to_aligned_aabb(
        decoded_world, axis_alignment
    )
    giou_term = 1.0 - paired_aabb_giou(
        decoded_aligned, targets, eps=eps
    )
    smooth_term = F.smooth_l1_loss(
        decoded_aligned,
        targets,
        beta=beta,
        reduction="none",
    ).mean(dim=1)
    per_box = giou_term + weight * smooth_term
    if reduction == "none":
        return per_box
    if len(per_box) == 0:
        return raw.sum() * 0.0
    if reduction == "sum":
        return per_box.sum()
    return per_box.mean()


def p1g_residual_geometry_aligned_loss(
    frozen_p1s_raw_regression: torch.Tensor,
    residual_correction: torch.Tensor,
    anchor_centers_world: torch.Tensor,
    target_boxes_aligned: torch.Tensor,
    axis_alignment: torch.Tensor,
    *,
    max_center_offset: float = 1.0,
    min_box_extent: float = 0.08,
    max_box_extent: float = 4.0,
    adapter_epsilon: float = P1G_DEFAULT_ADAPTER_EPSILON,
    smooth_l1_weight: float = 0.1,
    smooth_l1_beta: float = 0.1,
    eps: float = 1e-7,
    reduction: Reduction = "mean",
) -> torch.Tensor:
    """Aligned ScanNet loss for function-preserving P1G v2 corrections."""

    frozen = _require_float_tensor(
        frozen_p1s_raw_regression,
        name="frozen_p1s_raw_regression",
        columns=6,
    )
    correction = _require_float_tensor(
        residual_correction,
        name="residual_correction",
        columns=6,
    )
    anchors = _require_float_tensor(
        anchor_centers_world, name="anchor_centers_world", columns=3
    )
    targets = _validate_center_size_boxes(
        target_boxes_aligned, name="target_boxes_aligned"
    )
    for other, other_name in (
        (correction, "residual_correction"),
        (anchors, "anchor_centers_world"),
        (targets, "target_boxes_aligned"),
    ):
        _require_same_tensor_contract(
            frozen,
            other,
            first_name="frozen_p1s_raw_regression",
            second_name=other_name,
        )
    weight = _finite_scalar(
        smooth_l1_weight,
        name="smooth_l1_weight",
        non_negative=True,
    )
    beta = _finite_scalar(
        smooth_l1_beta, name="smooth_l1_beta", positive=True
    )
    _finite_scalar(eps, name="eps", positive=True)
    if reduction not in {"none", "mean", "sum"}:
        raise ValueError("reduction must be one of: none, mean, sum")

    decoded_world = decode_p1g_residual_aabb(
        frozen,
        correction,
        anchors,
        max_center_offset=max_center_offset,
        min_box_extent=min_box_extent,
        max_box_extent=max_box_extent,
        adapter_epsilon=adapter_epsilon,
    )
    decoded_aligned = world_aabb_to_aligned_aabb(
        decoded_world, axis_alignment
    )
    giou_term = 1.0 - paired_aabb_giou(
        decoded_aligned, targets, eps=eps
    )
    smooth_term = F.smooth_l1_loss(
        decoded_aligned,
        targets,
        beta=beta,
        reduction="none",
    ).mean(dim=1)
    per_box = giou_term + weight * smooth_term
    if reduction == "none":
        return per_box
    if len(per_box) == 0:
        return correction.sum() * 0.0
    if reduction == "sum":
        return per_box.sum()
    return per_box.mean()


__all__ = [
    "P1G_DEFAULT_ADAPTER_EPSILON",
    "decode_bounded_aabb",
    "decode_p1g_residual_aabb",
    "p1_geometry_aligned_loss",
    "p1_geometry_loss",
    "p1g_residual_geometry_aligned_loss",
    "p1s_raw_to_bounded_logits",
    "paired_aabb_giou",
    "paired_aabb_iou",
    "world_aabb_to_aligned_aabb",
]
