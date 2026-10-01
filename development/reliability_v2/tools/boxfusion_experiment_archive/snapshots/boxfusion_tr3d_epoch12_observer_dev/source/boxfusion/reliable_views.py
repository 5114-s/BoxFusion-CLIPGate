"""Reliable-view selection and weighted box initialization.

This module is intentionally NumPy-only so the selection policy can be tested
without importing PyCUDA or constructing the full BoxFusion model.
"""

from __future__ import annotations

from typing import Dict, Mapping, Tuple

import numpy as np


DEFAULT_RELIABLE_VIEW_CONFIG = {
    "enabled": False,
    "top_k": 3,
    "min_views": 3,
    "confidence_power": 1.0,
    "area_power": 0.25,
    "area_reference_ratio": 0.02,
    "projection_iou_power": 0.50,
    "geometry_consistency_power": 0.50,
    "center_sigma": 0.75,
    "size_sigma": 0.50,
    "minimum_box_diagonal": 0.10,
    "minimum_weight": 0.05,
}


def stable_unique(values: np.ndarray) -> np.ndarray:
    """Return first-occurrence unique integer values without reordering."""

    values = np.asarray(values, dtype=np.int64).reshape(-1)
    seen = set()
    result = []
    for value in values.tolist():
        if value not in seen:
            seen.add(value)
            result.append(value)
    return np.asarray(result, dtype=np.int64)


def valid_reliable_view_mask(
    boxes_3d: np.ndarray,
    rotations: np.ndarray,
    scores: np.ndarray,
    detector_boxes_2d: np.ndarray,
    projected_corners: np.ndarray,
    camera_poses: np.ndarray,
) -> np.ndarray:
    """Hard-filter observations that would make initialization/CUDA unsafe."""

    boxes = np.asarray(boxes_3d)
    rotations = np.asarray(rotations)
    scores = np.asarray(scores).reshape(-1)
    detector_boxes = np.asarray(detector_boxes_2d)
    corners = np.asarray(projected_corners)
    camera_poses = np.asarray(camera_poses)

    num_views = boxes.shape[0]
    expected_shapes = (
        (boxes.shape, (num_views, 6), "boxes_3d"),
        (rotations.shape, (num_views, 3, 3), "rotations"),
        (detector_boxes.shape, (num_views, 4), "detector_boxes_2d"),
        (corners.shape, (num_views, 8, 2), "projected_corners"),
        (camera_poses.shape, (num_views, 4, 4), "camera_poses"),
    )
    for actual, expected, name in expected_shapes:
        if actual != expected:
            raise ValueError(
                f"{name} must have shape {expected}, received {actual}"
            )
    if scores.shape[0] != num_views:
        raise ValueError("scores must have one value per view")

    valid = np.isfinite(boxes).all(axis=1)
    valid &= (boxes[:, 3:] > 0.0).all(axis=1)
    valid &= np.isfinite(rotations).all(axis=(1, 2))
    valid &= np.isfinite(scores)
    valid &= scores > 0.0
    valid &= np.isfinite(detector_boxes).all(axis=1)
    valid &= np.isfinite(corners).all(axis=(1, 2))
    valid &= np.isfinite(camera_poses).all(axis=(1, 2))
    for view_index in np.flatnonzero(valid):
        try:
            world_to_camera = np.linalg.inv(camera_poses[view_index])
        except np.linalg.LinAlgError:
            valid[view_index] = False
            continue
        center_homogeneous = np.concatenate(
            [boxes[view_index, :3], np.ones(1)]
        )
        center_camera = world_to_camera @ center_homogeneous
        if not np.isfinite(center_camera).all() or center_camera[2] <= 1e-3:
            valid[view_index] = False
    return valid


def resolve_reliable_view_config(
    box_fusion_cfg: Mapping,
) -> Dict[str, float]:
    """Return a validated reliable-view configuration.

    The caller passes the complete ``box_fusion`` mapping. Missing
    ``reliable_views`` configuration means disabled, which preserves the
    released BoxFusion behavior.
    """

    raw_cfg = box_fusion_cfg.get("reliable_views", {})
    cfg = dict(DEFAULT_RELIABLE_VIEW_CONFIG)
    cfg.update(raw_cfg)

    cfg["enabled"] = bool(cfg["enabled"])
    cfg["top_k"] = int(cfg["top_k"])
    cfg["min_views"] = int(cfg["min_views"])
    for key in (
        "confidence_power",
        "area_power",
        "area_reference_ratio",
        "projection_iou_power",
        "geometry_consistency_power",
        "center_sigma",
        "size_sigma",
        "minimum_box_diagonal",
        "minimum_weight",
    ):
        cfg[key] = float(cfg[key])

    if cfg["top_k"] < 1:
        raise ValueError("reliable_views.top_k must be at least 1")
    if cfg["min_views"] < 1:
        raise ValueError("reliable_views.min_views must be at least 1")
    if (
        cfg["confidence_power"] < 0.0
        or cfg["area_power"] < 0.0
        or cfg["projection_iou_power"] < 0.0
        or cfg["geometry_consistency_power"] < 0.0
    ):
        raise ValueError("reliable-view powers must be non-negative")
    for key in (
        "area_reference_ratio",
        "center_sigma",
        "size_sigma",
        "minimum_box_diagonal",
        "minimum_weight",
    ):
        if cfg[key] <= 0.0:
            raise ValueError(f"reliable_views.{key} must be positive")
    if cfg["minimum_weight"] > 1.0:
        raise ValueError("reliable_views.minimum_weight must not exceed 1")

    return cfg


def _validate_view_arrays(
    boxes_3d: np.ndarray,
    scores: np.ndarray,
    detector_boxes_2d: np.ndarray,
    projected_corners: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    boxes_3d = np.asarray(boxes_3d, dtype=np.float64)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    detector_boxes_2d = np.asarray(
        detector_boxes_2d, dtype=np.float64
    )
    projected_corners = np.asarray(
        projected_corners, dtype=np.float64
    )

    if boxes_3d.ndim != 2 or boxes_3d.shape[1] != 6:
        raise ValueError("boxes_3d must have shape [N, 6]")
    if projected_corners.ndim != 3 or projected_corners.shape[1:] != (8, 2):
        raise ValueError("projected_corners must have shape [N, 8, 2]")
    if detector_boxes_2d.ndim != 2 or detector_boxes_2d.shape[1] != 4:
        raise ValueError("detector_boxes_2d must have shape [N, 4]")
    if not (
        boxes_3d.shape[0]
        == scores.shape[0]
        == detector_boxes_2d.shape[0]
        == projected_corners.shape[0]
    ):
        raise ValueError("all reliable-view inputs must have the same length")
    if boxes_3d.shape[0] == 0:
        raise ValueError("reliable-view selection requires at least one view")

    boxes_3d = np.nan_to_num(
        boxes_3d, nan=0.0, posinf=0.0, neginf=0.0
    )
    scores = np.nan_to_num(scores, nan=0.0, posinf=1.0, neginf=0.0)
    detector_boxes_2d = np.nan_to_num(
        detector_boxes_2d, nan=0.0, posinf=0.0, neginf=0.0
    )
    projected_corners = np.nan_to_num(
        projected_corners, nan=0.0, posinf=0.0, neginf=0.0
    )
    return boxes_3d, scores, detector_boxes_2d, projected_corners


def projected_area_ratio(
    projected_corners: np.ndarray,
    image_height: int,
    image_width: int,
) -> np.ndarray:
    """Compute the clipped 2D bounding-rectangle area of every projection."""

    if image_height <= 0 or image_width <= 0:
        raise ValueError("image dimensions must be positive")

    corners = np.asarray(projected_corners, dtype=np.float64)
    x = np.clip(corners[..., 0], 0.0, float(image_width))
    y = np.clip(corners[..., 1], 0.0, float(image_height))
    widths = np.maximum(x.max(axis=1) - x.min(axis=1), 0.0)
    heights = np.maximum(y.max(axis=1) - y.min(axis=1), 0.0)
    image_area = float(image_height * image_width)
    return np.clip((widths * heights) / image_area, 0.0, 1.0)


def detector_projection_iou(
    detector_boxes_2d: np.ndarray,
    projected_corners: np.ndarray,
    image_height: int,
    image_width: int,
) -> np.ndarray:
    """IoU between Cubify's 2D box and its 3D box projection."""

    detector_boxes = np.asarray(detector_boxes_2d, dtype=np.float64)
    corners = np.asarray(projected_corners, dtype=np.float64)

    detector_x1 = np.clip(
        np.minimum(detector_boxes[:, 0], detector_boxes[:, 2]),
        0.0,
        float(image_width),
    )
    detector_y1 = np.clip(
        np.minimum(detector_boxes[:, 1], detector_boxes[:, 3]),
        0.0,
        float(image_height),
    )
    detector_x2 = np.clip(
        np.maximum(detector_boxes[:, 0], detector_boxes[:, 2]),
        0.0,
        float(image_width),
    )
    detector_y2 = np.clip(
        np.maximum(detector_boxes[:, 1], detector_boxes[:, 3]),
        0.0,
        float(image_height),
    )

    projected_x = np.clip(
        corners[..., 0], 0.0, float(image_width)
    )
    projected_y = np.clip(
        corners[..., 1], 0.0, float(image_height)
    )
    projected_x1 = projected_x.min(axis=1)
    projected_y1 = projected_y.min(axis=1)
    projected_x2 = projected_x.max(axis=1)
    projected_y2 = projected_y.max(axis=1)

    intersection_width = np.maximum(
        np.minimum(detector_x2, projected_x2)
        - np.maximum(detector_x1, projected_x1),
        0.0,
    )
    intersection_height = np.maximum(
        np.minimum(detector_y2, projected_y2)
        - np.maximum(detector_y1, projected_y1),
        0.0,
    )
    intersection = intersection_width * intersection_height
    detector_area = np.maximum(detector_x2 - detector_x1, 0.0) * np.maximum(
        detector_y2 - detector_y1, 0.0
    )
    projected_area = np.maximum(projected_x2 - projected_x1, 0.0) * np.maximum(
        projected_y2 - projected_y1, 0.0
    )
    union = detector_area + projected_area - intersection
    return np.divide(
        intersection,
        union,
        out=np.zeros_like(intersection),
        where=union > 1e-12,
    )


def geometric_consistency(
    boxes_3d: np.ndarray,
    center_sigma: float,
    size_sigma: float,
    minimum_box_diagonal: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Measure robust center and size agreement with the observation median."""

    boxes = np.asarray(boxes_3d, dtype=np.float64)
    centers = boxes[:, :3]
    dimensions = np.maximum(np.abs(boxes[:, 3:]), 1e-6)

    median_center = np.median(centers, axis=0)
    median_dimensions = np.median(np.sort(dimensions, axis=1), axis=0)
    reference_diagonal = max(
        float(np.median(np.linalg.norm(dimensions, axis=1))),
        minimum_box_diagonal,
    )

    center_error = (
        np.linalg.norm(centers - median_center[None, :], axis=1)
        / reference_diagonal
    )
    log_size_ratio = np.log(
        np.sort(dimensions, axis=1)
        / np.maximum(median_dimensions[None, :], 1e-6)
    )
    size_error = np.linalg.norm(log_size_ratio, axis=1) / np.sqrt(3.0)

    consistency = np.exp(
        -0.5 * np.square(center_error / center_sigma)
        -0.5 * np.square(size_error / size_sigma)
    )
    return consistency, center_error, size_error


def compute_reliable_view_weights(
    boxes_3d: np.ndarray,
    scores: np.ndarray,
    detector_boxes_2d: np.ndarray,
    projected_corners: np.ndarray,
    image_height: int,
    image_width: int,
    cfg: Mapping,
) -> Dict[str, np.ndarray]:
    """Compute confidence, visibility and geometry reliability components."""

    boxes, scores, detector_boxes, corners = _validate_view_arrays(
        boxes_3d,
        scores,
        detector_boxes_2d,
        projected_corners,
    )

    confidence = np.power(
        np.clip(scores, 0.0, 1.0),
        cfg["confidence_power"],
    )
    area_ratio = projected_area_ratio(
        corners, image_height, image_width
    )
    area_quality = np.power(
        np.clip(
            area_ratio / cfg["area_reference_ratio"],
            0.0,
            1.0,
        ),
        cfg["area_power"],
    )
    projection_iou = detector_projection_iou(
        detector_boxes,
        corners,
        image_height,
        image_width,
    )
    projection_quality = np.power(
        np.clip(projection_iou, 0.0, 1.0),
        cfg["projection_iou_power"],
    )
    consistency, center_error, size_error = geometric_consistency(
        boxes,
        center_sigma=cfg["center_sigma"],
        size_sigma=cfg["size_sigma"],
        minimum_box_diagonal=cfg["minimum_box_diagonal"],
    )
    geometry_quality = np.power(
        np.clip(consistency, 0.0, 1.0),
        cfg["geometry_consistency_power"],
    )

    raw_weights = (
        confidence
        * area_quality
        * projection_quality
        * geometry_quality
    )
    weights = np.maximum(raw_weights, cfg["minimum_weight"])
    weights = np.nan_to_num(
        weights,
        nan=cfg["minimum_weight"],
        posinf=1.0,
        neginf=cfg["minimum_weight"],
    )

    return {
        "weights": weights.astype(np.float32),
        "confidence": confidence.astype(np.float32),
        "area_ratio": area_ratio.astype(np.float32),
        "area_quality": area_quality.astype(np.float32),
        "projection_iou": projection_iou.astype(np.float32),
        "projection_quality": projection_quality.astype(np.float32),
        "geometry_consistency": consistency.astype(np.float32),
        "geometry_quality": geometry_quality.astype(np.float32),
        "center_error": center_error.astype(np.float32),
        "size_error": size_error.astype(np.float32),
    }


def select_top_k_reliable_views(
    boxes_3d: np.ndarray,
    scores: np.ndarray,
    detector_boxes_2d: np.ndarray,
    projected_corners: np.ndarray,
    image_height: int,
    image_width: int,
    cfg: Mapping,
) -> Dict[str, np.ndarray]:
    """Select a stable reliability-ranked Top-K subset."""

    components = compute_reliable_view_weights(
        boxes_3d,
        scores,
        detector_boxes_2d,
        projected_corners,
        image_height,
        image_width,
        cfg,
    )
    num_views = components["weights"].shape[0]
    keep_count = min(
        num_views,
        max(int(cfg["top_k"]), int(cfg["min_views"])),
    )

    # Reliability is primary, real confidence breaks minimum-weight ties, and
    # temporal order is the final deterministic tie-break.
    ranked = np.lexsort(
        (
            np.arange(num_views, dtype=np.int64),
            -components["confidence"],
            -components["weights"],
        )
    ).astype(np.int64)
    selected = ranked[:keep_count]
    selected_mask = np.zeros(num_views, dtype=bool)
    selected_mask[selected] = True

    components.update(
        {
            "ranked_indices": ranked,
            "selected_indices": selected,
            "selected_mask": selected_mask,
            "selected_weights": components["weights"][selected],
        }
    )
    return components


def weighted_box_initialization(
    boxes_3d: np.ndarray,
    rotations: np.ndarray,
    weights: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """Initialize center/size by reliability and rotation from the best view."""

    boxes = np.asarray(boxes_3d, dtype=np.float64)
    rotations = np.asarray(rotations, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64).reshape(-1)

    if boxes.ndim != 2 or boxes.shape[1] != 6:
        raise ValueError("boxes_3d must have shape [N, 6]")
    if rotations.shape != (boxes.shape[0], 3, 3):
        raise ValueError("rotations must have shape [N, 3, 3]")
    if weights.shape[0] != boxes.shape[0]:
        raise ValueError("weights must have one value per box")
    if boxes.shape[0] == 0:
        raise ValueError("weighted initialization requires at least one box")
    if not np.isfinite(boxes).all():
        raise ValueError("boxes_3d must contain only finite values")
    if not np.isfinite(rotations).all():
        raise ValueError("rotations must contain only finite values")
    if np.any(boxes[:, 3:] <= 0.0):
        raise ValueError("box dimensions must be positive")

    weights = np.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0)
    weights = np.maximum(weights, 0.0)
    if float(weights.sum()) <= 1e-12:
        weights = np.ones_like(weights)
    normalized_weights = weights / weights.sum()

    best_box = int(np.argmax(weights))
    initial_box = np.zeros(6, dtype=np.float64)
    initial_box[:3] = np.sum(
        boxes[:, :3] * normalized_weights[:, None], axis=0
    )

    best_size = boxes[best_box, 3:]
    best_axis_order = np.argsort(best_size)
    restore_best_axes = np.argsort(best_axis_order)
    canonical_sizes = np.sort(boxes[:, 3:], axis=1)
    aligned_sizes = canonical_sizes[:, restore_best_axes]
    initial_box[3:] = np.sum(
        aligned_sizes * normalized_weights[:, None], axis=0
    )

    return (
        initial_box.astype(np.float32),
        rotations[best_box].astype(np.float32),
        best_box,
    )
