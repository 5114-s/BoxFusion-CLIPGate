"""Past-only Diverse Top-K SAM3 mask/depth confirmation for birth candidates.

The module is deliberately independent of BoxFusion output mutation.  It
consumes one immutable world-space OBB, frozen SAM3 proposal caches, metric
depth and camera calibration.  View eligibility is causal: a candidate may
only inspect frames at or before its confirmation frame.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence, TypeAlias

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class SAM3BirthConfig:
    top_k: int = 5
    diversity_weight: float = 0.15
    diversity_translation_reference_m: float = 0.50
    diversity_ray_reference_deg: float = 20.0
    min_projected_area_pixels: float = 25.0
    min_bbox_iou: float = 0.02
    min_mask_containment: float = 0.10
    min_box_coverage: float = 0.10
    min_valid_depth_pixels: int = 24
    min_depth_m: float = 0.10
    max_depth_m: float = 8.0
    depth_scale: float = 1000.0
    max_depth_points: int = 4096
    box_expansion: float = 1.25
    voxel_size_m: float = 0.05
    min_component_points: int = 16
    min_inside_expanded_fraction: float = 0.15
    min_component_inside_fraction: float = 0.20
    strong_mask_score: float = 0.50
    min_strong_views: int = 2
    min_total_component_points: int = 64
    min_mean_inside_expanded: float = 0.25

    def __post_init__(self) -> None:
        for name in (
            "top_k", "min_valid_depth_pixels", "max_depth_points",
            "min_component_points", "min_strong_views",
            "min_total_component_points",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or int(value) < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name in (
            "diversity_translation_reference_m", "diversity_ray_reference_deg",
            "min_projected_area_pixels", "min_depth_m", "max_depth_m",
            "depth_scale", "box_expansion", "voxel_size_m",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if self.max_depth_m <= self.min_depth_m or self.box_expansion < 1.0:
            raise ValueError("invalid depth interval or box expansion")
        for name in (
            "diversity_weight", "min_bbox_iou", "min_mask_containment",
            "min_box_coverage", "min_inside_expanded_fraction",
            "min_component_inside_fraction", "strong_mask_score",
            "min_mean_inside_expanded",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must lie in [0,1]")

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class SAM3TeacherView:
    frame_id: int
    intrinsics: np.ndarray
    camera_to_world: np.ndarray
    depth_path: Path
    proposal_path: Path
    image_shape: tuple[int, int]

    def __post_init__(self) -> None:
        if isinstance(self.frame_id, bool) or int(self.frame_id) < 0:
            raise ValueError("frame_id must be nonnegative")
        intrinsic = np.asarray(self.intrinsics, dtype=np.float64)
        if intrinsic.shape == (4, 4):
            intrinsic = intrinsic[:3, :3]
        pose = np.asarray(self.camera_to_world, dtype=np.float64)
        if (
            intrinsic.shape != (3, 3) or pose.shape != (4, 4)
            or not np.isfinite(intrinsic).all() or not np.isfinite(pose).all()
        ):
            raise ValueError("invalid SAM3 view calibration")
        height, width = self.image_shape
        if height < 1 or width < 1:
            raise ValueError("image_shape must be positive")
        object.__setattr__(self, "frame_id", int(self.frame_id))
        object.__setattr__(self, "intrinsics", intrinsic)
        object.__setattr__(self, "camera_to_world", pose)
        object.__setattr__(self, "depth_path", Path(self.depth_path))
        object.__setattr__(self, "proposal_path", Path(self.proposal_path))
        object.__setattr__(self, "image_shape", (int(height), int(width)))


@dataclass(frozen=True)
class SAM3MemoryTeacherView:
    """One bounded live SAM3 observation kept entirely in memory.

    The offline route stores proposal masks in an authenticated NPZ.  A live
    provider cannot round-trip through that terminal cache, so this companion
    type carries the exact same proposal tensors as packed bits.  Depth and
    proposal arrays are detached and immutable; no path or later frame can be
    consulted while a candidate is confirmed.
    """

    frame_id: int
    intrinsics: np.ndarray
    camera_to_world: np.ndarray
    depth_m: np.ndarray
    masks_packbits: np.ndarray
    scores: np.ndarray
    labels: np.ndarray
    image_shape: tuple[int, int]

    def __post_init__(self) -> None:
        if isinstance(self.frame_id, bool) or int(self.frame_id) < 0:
            raise ValueError("frame_id must be nonnegative")
        intrinsic = np.asarray(self.intrinsics, dtype=np.float64)
        if intrinsic.shape == (4, 4):
            intrinsic = intrinsic[:3, :3]
        pose = np.asarray(self.camera_to_world, dtype=np.float64)
        height, width = (int(self.image_shape[0]), int(self.image_shape[1]))
        depth = np.asarray(self.depth_m, dtype=np.float32)
        packed = np.asarray(self.masks_packbits, dtype=np.uint8)
        scores = np.asarray(self.scores, dtype=np.float64)
        labels = np.asarray(self.labels)
        packed_width = (height * width + 7) // 8
        if (
            intrinsic.shape != (3, 3)
            or pose.shape != (4, 4)
            or not np.isfinite(intrinsic).all()
            or not np.isfinite(pose).all()
            or height < 1
            or width < 1
            or depth.shape != (height, width)
            or not np.isfinite(depth).all()
            or np.any(depth < 0.0)
            or packed.ndim != 2
            or packed.shape[1:] != (packed_width,)
            or scores.shape != (len(packed),)
            or labels.shape != (len(packed),)
            or not np.isfinite(scores).all()
            or np.any((scores < 0.0) | (scores > 1.0))
        ):
            raise ValueError("invalid live SAM3 view")
        intrinsic = np.frombuffer(
            np.ascontiguousarray(intrinsic).tobytes(), dtype=np.float64
        ).reshape(3, 3)
        pose = np.frombuffer(
            np.ascontiguousarray(pose).tobytes(), dtype=np.float64
        ).reshape(4, 4)
        depth = np.frombuffer(
            np.ascontiguousarray(depth).tobytes(), dtype=np.float32
        ).reshape(height, width)
        packed = np.frombuffer(
            np.ascontiguousarray(packed).tobytes(), dtype=np.uint8
        ).reshape(packed.shape)
        scores = np.frombuffer(
            np.ascontiguousarray(scores).tobytes(), dtype=np.float64
        ).reshape(scores.shape)
        labels = np.asarray(tuple(str(value) for value in labels), dtype=str)
        labels.setflags(write=False)
        object.__setattr__(self, "frame_id", int(self.frame_id))
        object.__setattr__(self, "intrinsics", intrinsic)
        object.__setattr__(self, "camera_to_world", pose)
        object.__setattr__(self, "depth_m", depth)
        object.__setattr__(self, "masks_packbits", packed)
        object.__setattr__(self, "scores", scores)
        object.__setattr__(self, "labels", labels)
        object.__setattr__(self, "image_shape", (height, width))


TeacherView: TypeAlias = SAM3TeacherView | SAM3MemoryTeacherView


@dataclass(frozen=True)
class ProjectedTeacherView:
    view: TeacherView
    bbox_xyxy: np.ndarray
    area_pixels: float
    reliability: float


def _corners(value: object) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (8, 3) or not np.isfinite(result).all():
        raise ValueError("candidate corners must be finite [8,3]")
    if np.any(np.ptp(result, axis=0) <= 0.0):
        raise ValueError("candidate must have positive AABB extent")
    return result


def _obb_axes(corners: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Recover the ordered Boxer axes from its stable eight-corner layout."""

    vectors = np.stack(
        (corners[4] - corners[0], corners[2] - corners[0], corners[1] - corners[0]),
        axis=0,
    )
    dimensions = np.linalg.norm(vectors, axis=1)
    if np.any(dimensions <= 1.0e-8):
        raise ValueError("candidate contains a degenerate OBB edge")
    axes = vectors / dimensions[:, None]
    if not np.allclose(axes @ axes.T, np.eye(3), rtol=0.0, atol=2.0e-3):
        raise ValueError("candidate Boxer axes are not orthogonal")
    return corners.mean(axis=0), axes, dimensions


def _project_bbox(
    corners: np.ndarray, view: TeacherView
) -> tuple[np.ndarray, float] | None:
    pose = view.camera_to_world
    camera = (corners - pose[:3, 3][None]) @ pose[:3, :3]
    if np.any(camera[:, 2] <= 1.0e-3):
        return None
    intrinsic = view.intrinsics
    uv = camera[:, :2] / camera[:, 2:3]
    uv[:, 0] = intrinsic[0, 0] * uv[:, 0] + intrinsic[0, 2]
    uv[:, 1] = intrinsic[1, 1] * uv[:, 1] + intrinsic[1, 2]
    height, width = view.image_shape
    bbox = np.asarray(
        [uv[:, 0].min(), uv[:, 1].min(), uv[:, 0].max(), uv[:, 1].max()],
        dtype=np.float64,
    )
    bbox[[0, 2]] = np.clip(bbox[[0, 2]], 0.0, float(width))
    bbox[[1, 3]] = np.clip(bbox[[1, 3]], 0.0, float(height))
    area = float(max(bbox[2] - bbox[0], 0.0) * max(bbox[3] - bbox[1], 0.0))
    if area <= 0.0:
        return None
    return bbox, area


def select_past_diverse_views(
    candidate_corners: object,
    confirmation_frame_id: int,
    views: Sequence[TeacherView],
    config: SAM3BirthConfig,
) -> tuple[ProjectedTeacherView, ...]:
    """Greedily select reliability-first, candidate-specific diverse past views."""

    corners = _corners(candidate_corners)
    visible: list[ProjectedTeacherView] = []
    for view in views:
        if view.frame_id > int(confirmation_frame_id):
            continue
        projection = _project_bbox(corners, view)
        if projection is None or projection[1] < config.min_projected_area_pixels:
            continue
        height, width = view.image_shape
        reliability = min(
            projection[1] / max(0.02 * float(height * width), 1.0), 1.0
        )
        visible.append(
            ProjectedTeacherView(view, projection[0], projection[1], reliability)
        )
    if not visible:
        return ()

    ranked = sorted(
        range(len(visible)),
        key=lambda index: (
            -visible[index].reliability,
            -visible[index].area_pixels,
            visible[index].view.frame_id,
        ),
    )
    selected = [ranked[0]]
    remaining = set(ranked[1:])
    candidate_center = corners.mean(axis=0)
    while remaining and len(selected) < config.top_k:
        best_index = None
        best_key = None
        for candidate in sorted(remaining):
            camera_center = visible[candidate].view.camera_to_world[:3, 3]
            candidate_ray = candidate_center - camera_center
            candidate_norm = max(float(np.linalg.norm(candidate_ray)), 1.0e-12)
            novelty_values = []
            for prior in selected:
                prior_center = visible[prior].view.camera_to_world[:3, 3]
                prior_ray = candidate_center - prior_center
                prior_norm = max(float(np.linalg.norm(prior_ray)), 1.0e-12)
                translation = float(np.linalg.norm(camera_center - prior_center))
                cosine = float(
                    np.clip(np.dot(candidate_ray, prior_ray) / (candidate_norm * prior_norm), -1.0, 1.0)
                )
                ray_angle = float(np.degrees(np.arccos(cosine)))
                novelty_values.append(
                    0.5 * min(
                        translation / config.diversity_translation_reference_m, 1.0
                    )
                    + 0.5 * min(
                        ray_angle / config.diversity_ray_reference_deg, 1.0
                    )
                )
            novelty = min(novelty_values)
            combined = (
                (1.0 - config.diversity_weight) * visible[candidate].reliability
                + config.diversity_weight * novelty
            )
            key = (
                combined,
                visible[candidate].reliability,
                visible[candidate].area_pixels,
                -visible[candidate].view.frame_id,
            )
            if best_key is None or key > best_key:
                best_key = key
                best_index = candidate
        assert best_index is not None
        selected.append(best_index)
        remaining.remove(best_index)
    return tuple(visible[index] for index in selected)


def _bbox_mask_metrics(
    bbox_xyxy: np.ndarray, mask: np.ndarray
) -> tuple[float, float, float]:
    binary = np.asarray(mask, dtype=np.bool_)
    height, width = binary.shape
    x1 = int(np.clip(np.floor(bbox_xyxy[0]), 0, width))
    y1 = int(np.clip(np.floor(bbox_xyxy[1]), 0, height))
    x2 = int(np.clip(np.ceil(bbox_xyxy[2]), 0, width))
    y2 = int(np.clip(np.ceil(bbox_xyxy[3]), 0, height))
    if x2 <= x1 or y2 <= y1:
        return 0.0, 0.0, 0.0
    mask_area = int(np.count_nonzero(binary))
    box_area = int((x2 - x1) * (y2 - y1))
    intersection = int(np.count_nonzero(binary[y1:y2, x1:x2]))
    union = mask_area + box_area - intersection
    return (
        float(intersection / union) if union else 0.0,
        float(intersection / mask_area) if mask_area else 0.0,
        float(intersection / box_area) if box_area else 0.0,
    )


def _backproject_mask(
    mask: np.ndarray,
    depth_m: np.ndarray,
    view: TeacherView,
    config: SAM3BirthConfig,
) -> tuple[int, np.ndarray]:
    valid = (
        np.asarray(mask, dtype=np.bool_)
        & np.isfinite(depth_m)
        & (depth_m >= config.min_depth_m)
        & (depth_m <= config.max_depth_m)
    )
    rows, cols = np.nonzero(valid)
    full_count = int(len(rows))
    if full_count == 0:
        return 0, np.empty((0, 3), dtype=np.float64)
    if full_count > config.max_depth_points:
        keep = np.unique(
            np.linspace(0, full_count - 1, config.max_depth_points, dtype=np.int64)
        )
        rows, cols = rows[keep], cols[keep]
    z = depth_m[rows, cols].astype(np.float64, copy=False)
    intrinsic = view.intrinsics
    camera = np.column_stack(
        (
            (cols - intrinsic[0, 2]) * z / intrinsic[0, 0],
            (rows - intrinsic[1, 2]) * z / intrinsic[1, 1],
            z,
        )
    )
    pose = view.camera_to_world
    world = camera @ pose[:3, :3].T + pose[:3, 3]
    return full_count, world


def _points_inside_obb(
    points: np.ndarray,
    center: np.ndarray,
    axes: np.ndarray,
    dimensions: np.ndarray,
    scale: float,
) -> np.ndarray:
    local = (points - center[None]) @ axes.T
    return np.all(np.abs(local) <= 0.5 * dimensions[None] * scale + 1.0e-6, axis=1)


def _largest_voxel_component(
    points: np.ndarray,
    inside_original: np.ndarray,
    voxel_size: float,
) -> tuple[int, float, float]:
    if len(points) == 0:
        return 0, 0.0, 0.0
    keys = np.floor(points / voxel_size).astype(np.int64)
    unique, inverse, counts = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
    original_counts = np.bincount(
        inverse, weights=inside_original.astype(np.float64), minlength=len(unique)
    )
    lookup = {tuple(key.tolist()): row for row, key in enumerate(unique)}
    visited = np.zeros(len(unique), dtype=np.bool_)
    offsets = [
        (dx, dy, dz)
        for dx in (-1, 0, 1)
        for dy in (-1, 0, 1)
        for dz in (-1, 0, 1)
        if (dx, dy, dz) != (0, 0, 0)
    ]
    best_rows: list[int] = []
    best_key = None
    for start in range(len(unique)):
        if visited[start]:
            continue
        visited[start] = True
        stack = [start]
        component: list[int] = []
        while stack:
            row = stack.pop()
            component.append(row)
            x, y, z = unique[row]
            for dx, dy, dz in offsets:
                neighbor = lookup.get((int(x + dx), int(y + dy), int(z + dz)))
                if neighbor is not None and not visited[neighbor]:
                    visited[neighbor] = True
                    stack.append(neighbor)
        point_count = int(counts[component].sum())
        original_count = float(original_counts[component].sum())
        key = (original_count, point_count, -min(component))
        if best_key is None or key > best_key:
            best_key = key
            best_rows = component
    best_points = int(counts[best_rows].sum())
    best_original = float(original_counts[best_rows].sum())
    return (
        best_points,
        best_original / max(best_points, 1),
        best_points / max(len(points), 1),
    )


def _evaluate_view(
    corners: np.ndarray,
    projected: ProjectedTeacherView,
    config: SAM3BirthConfig,
) -> dict[str, object]:
    view = projected.view
    if isinstance(view, SAM3MemoryTeacherView):
        depth_m = np.asarray(view.depth_m, dtype=np.float32)
        height, width = view.image_shape
        masks = np.unpackbits(
            view.masks_packbits,
            axis=1,
            count=height * width,
            bitorder="little",
        ).reshape(len(view.masks_packbits), height, width).astype(np.bool_, copy=False)
        scores = np.asarray(view.scores, dtype=np.float64)
        labels = np.asarray(view.labels)
        image_shape = view.image_shape
    else:
        depth_raw = np.asarray(Image.open(view.depth_path))
        if depth_raw.ndim != 2 or depth_raw.shape != view.image_shape:
            raise ValueError(f"invalid depth image: {view.depth_path}")
        depth_m = depth_raw.astype(np.float32) / config.depth_scale
        with np.load(view.proposal_path, allow_pickle=False) as cache:
            masks = np.asarray(cache["masks"], dtype=np.bool_)
            scores = np.asarray(cache["scores"], dtype=np.float64)
            labels = np.asarray(cache["labels"])
            image_shape = tuple(
                int(value) for value in np.asarray(cache["image_shape"]).tolist()
            )
    if (
        image_shape != view.image_shape or masks.ndim != 3
        or masks.shape[1:] != view.image_shape or scores.shape != (len(masks),)
        or labels.shape != (len(masks),)
    ):
        raise ValueError(f"invalid SAM3 cache: {view.proposal_path}")

    center, axes, dimensions = _obb_axes(corners)
    best = None
    for index, (mask, score) in enumerate(zip(masks, scores)):
        bbox_iou, containment, coverage = _bbox_mask_metrics(
            projected.bbox_xyxy, mask
        )
        if not (
            bbox_iou >= config.min_bbox_iou
            or (
                containment >= config.min_mask_containment
                and coverage >= config.min_box_coverage
            )
        ):
            continue
        valid_count, points = _backproject_mask(mask, depth_m, view, config)
        if len(points):
            inside_original = _points_inside_obb(
                points, center, axes, dimensions, 1.0
            )
            inside_expanded = _points_inside_obb(
                points, center, axes, dimensions, config.box_expansion
            )
            original_fraction = float(np.mean(inside_original))
            expanded_fraction = float(np.mean(inside_expanded))
            component_count, component_inside, component_fraction = (
                _largest_voxel_component(
                    points[inside_expanded],
                    inside_original[inside_expanded],
                    config.voxel_size_m,
                )
            )
        else:
            original_fraction = expanded_fraction = 0.0
            component_count = 0
            component_inside = component_fraction = 0.0
        evidence = (
            0.15 * float(score)
            + 0.15 * bbox_iou
            + 0.15 * containment
            + 0.15 * coverage
            + 0.15 * expanded_fraction
            + 0.10 * component_inside
            + 0.05 * component_fraction
            + 0.05 * min(valid_count / config.min_valid_depth_pixels, 1.0)
            + 0.05 * min(component_count / config.min_component_points, 1.0)
        )
        record = {
            "frame_id": view.frame_id,
            "sam3_index": index,
            "sam3_label": str(labels[index]),
            "sam3_score": float(score),
            "bbox_iou": bbox_iou,
            "mask_containment": containment,
            "box_coverage": coverage,
            "valid_depth_pixels": valid_count,
            "sampled_depth_points": len(points),
            "inside_original_fraction": original_fraction,
            "inside_expanded_fraction": expanded_fraction,
            "component_point_count": component_count,
            "component_inside_fraction": component_inside,
            "component_fraction": component_fraction,
            "evidence_score": evidence,
        }
        rank = (evidence, float(score), -index)
        if best is None or rank > best[0]:
            best = (rank, record)
    if best is None:
        return {
            "frame_id": view.frame_id,
            "projected_area_pixels": projected.area_pixels,
            "reliability": projected.reliability,
            "matched": False,
            "strong": False,
        }
    result = best[1]
    strong = (
        result["sam3_score"] >= config.strong_mask_score
        and result["mask_containment"] >= config.min_mask_containment
        and result["box_coverage"] >= config.min_box_coverage
        and result["valid_depth_pixels"] >= config.min_valid_depth_pixels
        and result["inside_expanded_fraction"]
        >= config.min_inside_expanded_fraction
        and result["component_point_count"] >= config.min_component_points
        and result["component_inside_fraction"]
        >= config.min_component_inside_fraction
    )
    result.update(
        {
            "projected_area_pixels": projected.area_pixels,
            "reliability": projected.reliability,
            "matched": True,
            "strong": bool(strong),
        }
    )
    return result


def confirm_candidate(
    candidate_corners: object,
    confirmation_frame_id: int,
    views: Sequence[TeacherView],
    config: SAM3BirthConfig | None = None,
) -> dict[str, object]:
    """Return a serializable, no-GT high-precision mask/depth decision."""

    resolved = config or SAM3BirthConfig()
    corners = _corners(candidate_corners)
    selected = select_past_diverse_views(
        corners, confirmation_frame_id, views, resolved
    )
    records = [_evaluate_view(corners, view, resolved) for view in selected]
    strong = [row for row in records if row["strong"]]
    total_component = int(sum(int(row["component_point_count"]) for row in strong))
    mean_inside = (
        float(np.mean([float(row["inside_expanded_fraction"]) for row in strong]))
        if strong else 0.0
    )
    passed = (
        len(strong) >= resolved.min_strong_views
        and total_component >= resolved.min_total_component_points
        and mean_inside >= resolved.min_mean_inside_expanded
    )
    return {
        "confirmation_frame_id": int(confirmation_frame_id),
        "past_only": True,
        "available_past_view_count": sum(
            view.frame_id <= int(confirmation_frame_id) for view in views
        ),
        "selected_view_count": len(selected),
        "selected_frame_ids": [view.view.frame_id for view in selected],
        "matched_view_count": sum(bool(row["matched"]) for row in records),
        "strong_view_count": len(strong),
        "total_component_points": total_component,
        "mean_strong_inside_expanded": mean_inside,
        "mask_depth_pass": bool(passed),
        "views": records,
    }


__all__ = [
    "ProjectedTeacherView",
    "SAM3BirthConfig",
    "SAM3TeacherView",
    "SAM3MemoryTeacherView",
    "confirm_candidate",
    "select_past_diverse_views",
]
