"""C2 multi-view Mask-RGBD confirmation for C1 residual TR3D tracks.

The module is deliberately prediction-agnostic.  It consumes immutable C1
rows, cached instance masks, ScanNet depth/calibration, and one candidate yaw
box.  It never consumes ground truth, CLIP labels, or writes detections.

The cached SAM3/YOLOE label is retained only as a diagnostic string.  Every
matching and confirmation decision below is class agnostic and geometric.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Sequence

import numpy as np

from .supplemental_proposals import SupplementalProposal
from .tr3d_r2_geometry import (
    compose_depth_camera_to_world,
    project_yaw_obb_to_depth,
)
from .tr3d_r5_spgroup_observer import points_in_yaw_box


GATE_NAMES = ("mask_any", "mask1", "mask2", "mask2_depth", "mask3_strict")


@dataclass(frozen=True)
class C2MaskRGBDConfig:
    """Pre-registered class-agnostic C2 observer thresholds."""

    source_budget: int = 10
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
    mask2_min_total_component_points: int = 64
    mask2_min_mean_inside_expanded: float = 0.25
    mask3_min_total_component_points: int = 96
    mask3_min_mean_inside_expanded: float = 0.30

    def __post_init__(self) -> None:
        integer_names = (
            "source_budget", "min_valid_depth_pixels", "max_depth_points",
            "min_component_points", "mask2_min_total_component_points",
            "mask3_min_total_component_points",
        )
        for name in integer_names:
            value = getattr(self, name)
            if not isinstance(value, (int, np.integer)) or int(value) < 1:
                raise ValueError(f"{name} must be a positive integer")
        positive_names = (
            "min_projected_area_pixels", "min_depth_m", "max_depth_m",
            "depth_scale", "box_expansion", "voxel_size_m",
        )
        for name in positive_names:
            value = float(getattr(self, name))
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if self.max_depth_m <= self.min_depth_m:
            raise ValueError("max_depth_m must exceed min_depth_m")
        if self.box_expansion < 1.0:
            raise ValueError("box_expansion must be at least one")
        fraction_names = (
            "min_bbox_iou", "min_mask_containment", "min_box_coverage",
            "min_inside_expanded_fraction", "min_component_inside_fraction",
            "strong_mask_score", "mask2_min_mean_inside_expanded",
            "mask3_min_mean_inside_expanded",
        )
        for name in fraction_names:
            value = float(getattr(self, name))
            if not np.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must lie in [0,1]")

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class C2Frame:
    frame_id: int
    depth_meters: np.ndarray
    intrinsics: np.ndarray
    depth_camera_to_world: np.ndarray
    proposals: tuple[SupplementalProposal, ...]
    cache_sha256: str


@dataclass(frozen=True)
class C2SceneObservation:
    frame_ids: np.ndarray
    projected_valid: np.ndarray
    projected_area_pixels: np.ndarray
    best_mask_index: np.ndarray
    best_mask_score: np.ndarray
    best_mask_label: np.ndarray
    bbox_iou: np.ndarray
    mask_containment: np.ndarray
    box_coverage: np.ndarray
    valid_depth_pixels: np.ndarray
    sampled_depth_points: np.ndarray
    inside_original_fraction: np.ndarray
    inside_expanded_fraction: np.ndarray
    component_point_count: np.ndarray
    component_inside_fraction: np.ndarray
    component_fraction: np.ndarray
    evidence_score: np.ndarray
    view_matched: np.ndarray
    view_strong: np.ndarray
    projected_view_count: np.ndarray
    matched_view_count: np.ndarray
    strong_view_count: np.ndarray
    total_component_points: np.ndarray
    mean_strong_inside_expanded: np.ndarray
    max_evidence_score: np.ndarray
    gate_mask: np.ndarray


def _readonly(value: object, dtype: np.dtype) -> np.ndarray:
    result = np.ascontiguousarray(np.asarray(value, dtype=dtype))
    result.setflags(write=False)
    return result


def _bbox_mask_metrics(
    bbox_xyxy: np.ndarray, mask: np.ndarray
) -> tuple[float, float, float]:
    binary = np.asarray(mask, dtype=np.bool_)
    if binary.ndim != 2:
        raise ValueError("mask must be [H,W]")
    box = np.asarray(bbox_xyxy, dtype=np.float64)
    if box.shape != (4,) or not np.isfinite(box).all():
        raise ValueError("bbox must contain four finite values")
    height, width = binary.shape
    x1 = int(np.clip(np.floor(box[0]), 0, width))
    y1 = int(np.clip(np.floor(box[1]), 0, height))
    x2 = int(np.clip(np.ceil(box[2]), 0, width))
    y2 = int(np.clip(np.ceil(box[3]), 0, height))
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


def _bounded_indices(count: int, maximum: int) -> np.ndarray:
    if count <= maximum:
        return np.arange(count, dtype=np.int64)
    return np.unique(np.linspace(0, count - 1, maximum, dtype=np.int64))


def _backproject_mask(
    mask: np.ndarray,
    depth_meters: np.ndarray,
    intrinsics: np.ndarray,
    camera_to_world: np.ndarray,
    config: C2MaskRGBDConfig,
) -> tuple[int, np.ndarray]:
    binary = np.asarray(mask, dtype=np.bool_)
    depth = np.asarray(depth_meters, dtype=np.float64)
    if binary.shape != depth.shape or binary.ndim != 2:
        raise ValueError("mask and depth must share shape [H,W]")
    valid = (
        binary & np.isfinite(depth)
        & (depth >= config.min_depth_m) & (depth <= config.max_depth_m)
    )
    rows, cols = np.nonzero(valid)
    full_count = int(len(rows))
    if full_count == 0:
        return 0, np.empty((0, 3), dtype=np.float64)
    keep = _bounded_indices(full_count, config.max_depth_points)
    rows = rows[keep]
    cols = cols[keep]
    z = depth[rows, cols]
    intrinsic = np.asarray(intrinsics, dtype=np.float64)
    pose = np.asarray(camera_to_world, dtype=np.float64)
    if intrinsic.shape == (4, 4):
        intrinsic = intrinsic[:3, :3]
    if intrinsic.shape != (3, 3) or pose.shape != (4, 4):
        raise ValueError("invalid intrinsics or camera pose")
    pixels = np.column_stack((cols, rows, np.ones(len(rows), dtype=np.float64)))
    rays = pixels @ np.linalg.inv(intrinsic).T
    if np.any(np.abs(rays[:, 2]) <= 1e-12):
        raise ValueError("intrinsics produced a zero-z ray")
    camera = rays * (z / rays[:, 2])[:, None]
    world = camera @ pose[:3, :3].T + pose[:3, 3]
    if not np.isfinite(world).all():
        raise ValueError("mask depth backprojection produced non-finite points")
    return full_count, world


def _largest_voxel_component(
    points: np.ndarray,
    inside_original: np.ndarray,
    voxel_size: float,
) -> tuple[int, float, float]:
    """Return points, original-box fraction, and support fraction of best component."""

    if len(points) == 0:
        return 0, 0.0, 0.0
    keys = np.floor(np.asarray(points, dtype=np.float64) / voxel_size).astype(np.int64)
    unique, inverse, counts = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
    original_counts = np.bincount(
        inverse, weights=np.asarray(inside_original, dtype=np.float64), minlength=len(unique)
    )
    key_to_row = {tuple(key.tolist()): row for row, key in enumerate(unique)}
    visited = np.zeros(len(unique), dtype=np.bool_)
    best_key: tuple[float, int, tuple[int, int, int]] | None = None
    best_rows: list[int] = []
    offsets = [
        (dx, dy, dz)
        for dx in (-1, 0, 1)
        for dy in (-1, 0, 1)
        for dz in (-1, 0, 1)
        if (dx, dy, dz) != (0, 0, 0)
    ]
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
                neighbor = key_to_row.get((int(x + dx), int(y + dy), int(z + dz)))
                if neighbor is not None and not visited[neighbor]:
                    visited[neighbor] = True
                    stack.append(neighbor)
        point_count = int(counts[component].sum())
        original_count = float(original_counts[component].sum())
        first_key = tuple(int(v) for v in unique[min(component)].tolist())
        rank_key = (original_count, point_count, tuple(-v for v in first_key))
        if best_key is None or rank_key > best_key:
            best_key = rank_key
            best_rows = component
    best_points = int(counts[best_rows].sum())
    best_original = float(original_counts[best_rows].sum())
    return (
        best_points,
        best_original / max(best_points, 1),
        best_points / max(len(points), 1),
    )


def _proposal_evidence(
    proposal: SupplementalProposal,
    projected_bbox: np.ndarray,
    box_world: np.ndarray,
    frame: C2Frame,
    config: C2MaskRGBDConfig,
) -> tuple[float, ...] | None:
    bbox_iou, containment, coverage = _bbox_mask_metrics(
        projected_bbox, proposal.mask
    )
    if not (
        bbox_iou >= config.min_bbox_iou
        or (
            containment >= config.min_mask_containment
            and coverage >= config.min_box_coverage
        )
    ):
        return None
    valid_count, points = _backproject_mask(
        proposal.mask,
        frame.depth_meters,
        frame.intrinsics,
        frame.depth_camera_to_world,
        config,
    )
    if len(points):
        inside_original = points_in_yaw_box(points, box_world, scale=1.0)
        inside_expanded = points_in_yaw_box(
            points, box_world, scale=config.box_expansion
        )
        original_fraction = float(np.mean(inside_original))
        expanded_fraction = float(np.mean(inside_expanded))
        local_points = points[inside_expanded]
        local_original = inside_original[inside_expanded]
        component_count, component_inside, component_fraction = _largest_voxel_component(
            local_points, local_original, config.voxel_size_m
        )
    else:
        original_fraction = expanded_fraction = 0.0
        component_count = 0
        component_inside = component_fraction = 0.0
    depth_sufficiency = min(valid_count / max(config.min_valid_depth_pixels, 1), 1.0)
    component_sufficiency = min(
        component_count / max(config.min_component_points, 1), 1.0
    )
    evidence = (
        0.15 * float(proposal.score)
        + 0.15 * bbox_iou
        + 0.15 * containment
        + 0.15 * coverage
        + 0.15 * expanded_fraction
        + 0.10 * component_inside
        + 0.05 * component_fraction
        + 0.05 * depth_sufficiency
        + 0.05 * component_sufficiency
    )
    return (
        evidence, bbox_iou, containment, coverage, float(valid_count),
        float(len(points)), original_fraction, expanded_fraction,
        float(component_count), component_inside, component_fraction,
    )


def observe_scene(
    boxes_world: object,
    frames: Sequence[C2Frame],
    config: C2MaskRGBDConfig,
) -> C2SceneObservation:
    boxes = np.asarray(boxes_world, dtype=np.float64)
    if boxes.ndim != 2 or boxes.shape[1] != 7:
        raise ValueError("boxes_world must be [N,7]")
    if not np.isfinite(boxes).all() or np.any(boxes[:, 3:6] <= 0):
        raise ValueError("boxes_world contains invalid boxes")
    frame_values = tuple(frames)
    frame_ids = np.asarray([frame.frame_id for frame in frame_values], dtype=np.int64)
    if len(np.unique(frame_ids)) != len(frame_ids) or np.any(frame_ids < 0):
        raise ValueError("frame ids must be unique and nonnegative")
    p, f = len(boxes), len(frame_values)
    projected = np.zeros((p, f), dtype=np.bool_)
    projected_area = np.zeros((p, f), dtype=np.float32)
    best_index = np.full((p, f), -1, dtype=np.int32)
    best_score = np.zeros((p, f), dtype=np.float32)
    labels = np.full((p, f), "", dtype="<U64")
    metric_arrays = [np.zeros((p, f), dtype=np.float32) for _ in range(9)]
    (
        bbox_iou, containment, coverage, valid_pixels, sampled_points,
        inside_original, inside_expanded, component_points, component_inside,
    ) = metric_arrays
    component_fraction = np.zeros((p, f), dtype=np.float32)
    evidence = np.zeros((p, f), dtype=np.float32)
    matched = np.zeros((p, f), dtype=np.bool_)
    strong = np.zeros((p, f), dtype=np.bool_)

    for candidate_row, box in enumerate(boxes):
        for frame_row, frame in enumerate(frame_values):
            projection = project_yaw_obb_to_depth(
                box, frame.intrinsics, frame.depth_camera_to_world,
                frame.depth_meters.shape,
            )
            if projection is None or projection.area_pixels < config.min_projected_area_pixels:
                continue
            projected[candidate_row, frame_row] = True
            projected_area[candidate_row, frame_row] = projection.area_pixels
            best: tuple[tuple[float, float, int], int, tuple[float, ...]] | None = None
            for proposal_row, proposal in enumerate(frame.proposals):
                values = _proposal_evidence(
                    proposal, projection.bbox_xyxy, box, frame, config
                )
                if values is None:
                    continue
                rank = (values[0], float(proposal.score), -proposal_row)
                if best is None or rank > best[0]:
                    best = (rank, proposal_row, values)
            if best is None:
                continue
            _, proposal_row, values = best
            proposal = frame.proposals[proposal_row]
            (
                evidence_value, iou_value, containment_value, coverage_value,
                valid_value, sampled_value, original_value, expanded_value,
                component_value, component_inside_value, component_fraction_value,
            ) = values
            best_index[candidate_row, frame_row] = proposal_row
            best_score[candidate_row, frame_row] = proposal.score
            labels[candidate_row, frame_row] = proposal.label or ""
            bbox_iou[candidate_row, frame_row] = iou_value
            containment[candidate_row, frame_row] = containment_value
            coverage[candidate_row, frame_row] = coverage_value
            valid_pixels[candidate_row, frame_row] = valid_value
            sampled_points[candidate_row, frame_row] = sampled_value
            inside_original[candidate_row, frame_row] = original_value
            inside_expanded[candidate_row, frame_row] = expanded_value
            component_points[candidate_row, frame_row] = component_value
            component_inside[candidate_row, frame_row] = component_inside_value
            component_fraction[candidate_row, frame_row] = component_fraction_value
            evidence[candidate_row, frame_row] = evidence_value
            matched[candidate_row, frame_row] = True
            strong[candidate_row, frame_row] = (
                proposal.score >= config.strong_mask_score
                and containment_value >= config.min_mask_containment
                and coverage_value >= config.min_box_coverage
                and valid_value >= config.min_valid_depth_pixels
                and expanded_value >= config.min_inside_expanded_fraction
                and component_value >= config.min_component_points
                and component_inside_value >= config.min_component_inside_fraction
            )

    projected_count = projected.sum(axis=1, dtype=np.int32)
    matched_count = matched.sum(axis=1, dtype=np.int32)
    strong_count = strong.sum(axis=1, dtype=np.int32)
    total_component = np.where(strong, component_points, 0.0).sum(axis=1).astype(np.int32)
    mean_inside = np.divide(
        np.where(strong, inside_expanded, 0.0).sum(axis=1),
        strong_count,
        out=np.zeros(p, dtype=np.float64),
        where=strong_count > 0,
    ).astype(np.float32)
    maximum_evidence = evidence.max(axis=1, initial=0.0)
    gates = np.stack(
        (
            matched_count >= 1,
            strong_count >= 1,
            strong_count >= 2,
            (strong_count >= 2)
            & (total_component >= config.mask2_min_total_component_points)
            & (mean_inside >= config.mask2_min_mean_inside_expanded),
            (strong_count >= 3)
            & (total_component >= config.mask3_min_total_component_points)
            & (mean_inside >= config.mask3_min_mean_inside_expanded),
        ),
        axis=1,
    )
    return C2SceneObservation(
        frame_ids=_readonly(frame_ids, np.int64),
        projected_valid=_readonly(projected, np.bool_),
        projected_area_pixels=_readonly(projected_area, np.float32),
        best_mask_index=_readonly(best_index, np.int32),
        best_mask_score=_readonly(best_score, np.float32),
        best_mask_label=_readonly(labels, labels.dtype),
        bbox_iou=_readonly(bbox_iou, np.float32),
        mask_containment=_readonly(containment, np.float32),
        box_coverage=_readonly(coverage, np.float32),
        valid_depth_pixels=_readonly(valid_pixels, np.float32),
        sampled_depth_points=_readonly(sampled_points, np.float32),
        inside_original_fraction=_readonly(inside_original, np.float32),
        inside_expanded_fraction=_readonly(inside_expanded, np.float32),
        component_point_count=_readonly(component_points, np.float32),
        component_inside_fraction=_readonly(component_inside, np.float32),
        component_fraction=_readonly(component_fraction, np.float32),
        evidence_score=_readonly(evidence, np.float32),
        view_matched=_readonly(matched, np.bool_),
        view_strong=_readonly(strong, np.bool_),
        projected_view_count=_readonly(projected_count, np.int32),
        matched_view_count=_readonly(matched_count, np.int32),
        strong_view_count=_readonly(strong_count, np.int32),
        total_component_points=_readonly(total_component, np.int32),
        mean_strong_inside_expanded=_readonly(mean_inside, np.float32),
        max_evidence_score=_readonly(maximum_evidence, np.float32),
        gate_mask=_readonly(gates, np.bool_),
    )


__all__ = [
    "C2Frame", "C2MaskRGBDConfig", "C2SceneObservation", "GATE_NAMES",
    "observe_scene",
]
