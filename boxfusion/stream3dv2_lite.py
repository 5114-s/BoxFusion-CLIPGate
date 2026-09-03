"""Training-free, past-only Stream3Dv2-inspired track refinement.

This module is intentionally dataset and evaluator agnostic.  It consumes a
small, already-associated window of automatic-mask RGB-D lifts and one frozen
Boxer hypothesis per view.  The implementation keeps the parts of
Stream3Dv2 that can be reproduced from sealed BoxFusion evidence:

* MVF-lite: 5 cm mask denoising, greedy key-voxel set cover and component
  selection touching the latest observation;
* SDS-lite: semantic consistency is supplied separately from frozen SAM3
  observations and is used as continuous evidence rather than a hard veto;
* PMR-lite: a bounded 26-neighbour voxel graph, multi-view seeds and five
  propagation rounds produce a dominant point-cloud manifold;
* robust OBB hypotheses: the Boxer HB medoid, an HB-axis PMR fit and a
  gravity-aligned yaw PMR fit are compared without annotations.

It is a lightweight adapter, not a verbatim reproduction of Stream3Dv2.  It
does not import predictions, annotations, ground truth or an evaluator and it
owns no trainable parameters.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping, Sequence

import numpy as np


VOXEL_SIZE_M = 0.05
LOCAL_WINDOW_KEYFRAMES = 20
MAX_RETAINED_VIEWS = 5
KEY_POINT_RATE = 0.05
MAX_KEY_POINTS = 512
MASK_MERGE_IOU = 0.20
MASK_MERGE_CONTAINMENT = 0.60
MASK_MERGE_ND_MAX = 0.50
PMR_ITERATIONS = 5
MIN_COMPONENT_VOXELS = 16
MIN_OUTPUT_EXTENT_M = 0.05
MAX_OUTPUT_EXTENT_M = 3.50
MAX_OUTPUT_VOLUME_M3 = 12.0
PMR_CENTER_SHIFT_MAX_M = 0.50
PMR_EXTENT_RATIO_RANGE = (0.50, 2.00)
PMR_VOLUME_RATIO_RANGE = (0.25, 4.00)

_CORNER_SIGNS = np.asarray(
    [
        (-1.0, -1.0, -1.0),
        (-1.0, -1.0, +1.0),
        (-1.0, +1.0, -1.0),
        (-1.0, +1.0, +1.0),
        (+1.0, -1.0, -1.0),
        (+1.0, -1.0, +1.0),
        (+1.0, +1.0, -1.0),
        (+1.0, +1.0, +1.0),
    ],
    dtype=np.float64,
)
_NEIGHBOURS = tuple(
    (dx, dy, dz)
    for dx in (-1, 0, 1)
    for dy in (-1, 0, 1)
    for dz in (-1, 0, 1)
    if (dx, dy, dz) != (0, 0, 0)
)


def _finite_points(value: object) -> np.ndarray:
    points = np.asarray(value, dtype=np.float64)
    if points.ndim != 2 or points.shape[1:] != (3,) or not np.isfinite(points).all():
        raise ValueError("points must be finite [N,3]")
    if not len(points):
        raise ValueError("points must be non-empty")
    return np.ascontiguousarray(points)


def _finite_corners(value: object) -> np.ndarray:
    corners = np.asarray(value, dtype=np.float64)
    if (
        corners.shape != (8, 3)
        or not np.isfinite(corners).all()
        or np.any(np.ptp(corners, axis=0) <= 0.0)
    ):
        raise ValueError("corners must be a finite, non-degenerate [8,3] array")
    return np.ascontiguousarray(corners)


def _voxel_keys(points: np.ndarray) -> np.ndarray:
    return np.unique(np.floor(points / VOXEL_SIZE_M).astype(np.int64), axis=0)


def _key_tuples(keys: np.ndarray) -> set[tuple[int, int, int]]:
    return {(int(row[0]), int(row[1]), int(row[2])) for row in keys}


def _largest_component(keys: np.ndarray, preferred: set[tuple[int, int, int]] | None = None) -> np.ndarray:
    """Return indices of the best bounded 26-neighbour component."""

    if not len(keys):
        return np.empty((0,), dtype=np.int64)
    lookup = {tuple(map(int, key)): index for index, key in enumerate(keys)}
    visited: set[int] = set()
    best: list[int] = []
    best_rank: tuple[int, int, int] | None = None
    preferred = preferred or set()
    for start in range(len(keys)):
        if start in visited:
            continue
        visited.add(start)
        stack = [start]
        component: list[int] = []
        preferred_count = 0
        while stack:
            index = stack.pop()
            component.append(index)
            key = tuple(map(int, keys[index]))
            preferred_count += int(key in preferred)
            x, y, z = key
            for dx, dy, dz in _NEIGHBOURS:
                neighbor = lookup.get((x + dx, y + dy, z + dz))
                if neighbor is not None and neighbor not in visited:
                    visited.add(neighbor)
                    stack.append(neighbor)
        rank = (preferred_count, len(component), -min(component))
        if best_rank is None or rank > best_rank:
            best_rank = rank
            best = component
    return np.asarray(sorted(best), dtype=np.int64)


def aabb_overlap(left: object, right: object) -> tuple[float, float, float]:
    lc = _finite_corners(left)
    rc = _finite_corners(right)
    ll, lu = lc.min(axis=0), lc.max(axis=0)
    rl, ru = rc.min(axis=0), rc.max(axis=0)
    intersection = float(np.prod(np.maximum(np.minimum(lu, ru) - np.maximum(ll, rl), 0.0)))
    lv = float(np.prod(lu - ll))
    rv = float(np.prod(ru - rl))
    union = lv + rv - intersection
    return (
        0.0 if union <= 0.0 else intersection / union,
        0.0 if lv <= 0.0 else intersection / lv,
        0.0 if rv <= 0.0 else intersection / rv,
    )


def normalized_center_distance(left: object, right: object) -> float:
    lc = _finite_corners(left)
    rc = _finite_corners(right)
    scale = max(
        float(np.linalg.norm(np.ptp(lc, axis=0))),
        float(np.linalg.norm(np.ptp(rc, axis=0))),
        0.02,
    )
    return float(np.linalg.norm(lc.mean(axis=0) - rc.mean(axis=0)) / scale)


def _obb_axes(corners: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    vectors = np.stack(
        (corners[4] - corners[0], corners[2] - corners[0], corners[1] - corners[0]),
        axis=0,
    )
    dimensions = np.linalg.norm(vectors, axis=1)
    if np.any(dimensions <= 1.0e-8):
        raise ValueError("degenerate OBB")
    axes = vectors / dimensions[:, None]
    if not np.allclose(axes @ axes.T, np.eye(3), rtol=0.0, atol=3.0e-3):
        # Evaluator consumes the corners, so a stable AABB fallback is safer
        # than repairing an unknown corner layout.
        lower, upper = corners.min(axis=0), corners.max(axis=0)
        axes = np.eye(3, dtype=np.float64)
        dimensions = upper - lower
        return (lower + upper) * 0.5, axes, dimensions
    return corners.mean(axis=0), axes, dimensions


def points_inside_obb(points: object, corners: object, scale: float = 1.0) -> np.ndarray:
    rows = _finite_points(points)
    box = _finite_corners(corners)
    center, axes, dimensions = _obb_axes(box)
    local = (rows - center[None]) @ axes.T
    return np.all(np.abs(local) <= 0.5 * dimensions[None] * float(scale) + 1.0e-6, axis=1)


def _obb_from_axes(points: np.ndarray, axes: np.ndarray) -> np.ndarray:
    projected = points @ axes.T
    lower, upper = np.quantile(projected, [0.02, 0.98], axis=0)
    extent = np.maximum(upper - lower, MIN_OUTPUT_EXTENT_M)
    center_local = (lower + upper) * 0.5
    center_world = center_local @ axes
    local = _CORNER_SIGNS * (extent[None] * 0.5)
    return np.ascontiguousarray(center_world[None] + local @ axes)


def _yaw_obb(points: np.ndarray) -> np.ndarray:
    centered = points[:, :2] - np.median(points[:, :2], axis=0, keepdims=True)
    covariance = centered.T @ centered / max(len(centered), 1)
    try:
        _, vectors = np.linalg.eigh(covariance)
        primary = vectors[:, -1]
    except np.linalg.LinAlgError:
        primary = np.asarray([1.0, 0.0], dtype=np.float64)
    if primary[0] < 0.0 or (primary[0] == 0.0 and primary[1] < 0.0):
        primary = -primary
    secondary = np.asarray([-primary[1], primary[0]], dtype=np.float64)
    axes = np.asarray(
        [
            [primary[0], primary[1], 0.0],
            [secondary[0], secondary[1], 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    return _obb_from_axes(points, axes)


def _guarded_hb_fit(points: np.ndarray, hb: np.ndarray) -> np.ndarray | None:
    hb_center, axes, hb_extent = _obb_axes(hb)
    candidate = _obb_from_axes(points, axes)
    center, _, extent = _obb_axes(candidate)
    extent_ratio = extent / np.maximum(hb_extent, 1.0e-6)
    volume_ratio = float(np.prod(extent) / max(np.prod(hb_extent), 1.0e-9))
    if (
        float(np.linalg.norm(center - hb_center)) > PMR_CENTER_SHIFT_MAX_M
        or np.any(extent_ratio < PMR_EXTENT_RATIO_RANGE[0])
        or np.any(extent_ratio > PMR_EXTENT_RATIO_RANGE[1])
        or not PMR_VOLUME_RATIO_RANGE[0] <= volume_ratio <= PMR_VOLUME_RATIO_RANGE[1]
    ):
        return None
    return candidate


def _geometric_mean(values: Sequence[float]) -> float:
    rows = np.asarray([float(value) for value in values], dtype=np.float64)
    if not len(rows) or not np.isfinite(rows).all():
        raise ValueError("quality terms must be finite and non-empty")
    rows = np.clip(rows, 1.0e-4, 1.0)
    return float(np.exp(np.mean(np.log(rows))))


@dataclass(frozen=True)
class TrackView:
    source_id: str
    frame_id: int
    frame_ordinal: int
    mask_confidence: float
    hb_confidence: float
    points_world: np.ndarray
    hb_corners: np.ndarray

    def __post_init__(self) -> None:
        if not isinstance(self.source_id, str) or not self.source_id:
            raise ValueError("source_id must be non-empty")
        if int(self.frame_id) < 0 or int(self.frame_ordinal) < 0:
            raise ValueError("frame identity must be non-negative")
        for name in ("mask_confidence", "hb_confidence"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must lie in [0,1]")
            object.__setattr__(self, name, value)
        points = _finite_points(self.points_world)
        corners = _finite_corners(self.hb_corners)
        points.setflags(write=False)
        corners.setflags(write=False)
        object.__setattr__(self, "frame_id", int(self.frame_id))
        object.__setattr__(self, "frame_ordinal", int(self.frame_ordinal))
        object.__setattr__(self, "points_world", points)
        object.__setattr__(self, "hb_corners", corners)


@dataclass(frozen=True)
class TrackGeometry:
    source_ids: tuple[str, ...]
    frame_ids: tuple[int, ...]
    decision_frame_id: int
    decision_frame_ordinal: int
    selected_source_ids: tuple[str, ...]
    hb_source_id: str
    hypotheses: Mapping[str, np.ndarray]
    hypothesis_quality: Mapping[str, float]
    chosen_hypothesis: str
    corners: np.ndarray
    refined_points: np.ndarray
    distinct_view_count: int
    set_cover_fraction: float
    median_pairwise_hb_iou: float
    median_pairwise_hb_containment: float
    hb_center_rms_m: float
    point_inside_hb_fraction: float
    pmr_seed_fraction: float
    pmr_retained_fraction: float
    mask_confidence_mean: float
    hb_confidence_mean: float
    preliminary_score: float


def _view_overlap(left: TrackView, right: TrackView) -> tuple[float, float, float]:
    iou, left_in_right, right_in_left = aabb_overlap(left.hb_corners, right.hb_corners)
    return iou, max(left_in_right, right_in_left), normalized_center_distance(
        left.hb_corners, right.hb_corners
    )


def _set_cover(views: Sequence[TrackView]) -> tuple[list[int], float]:
    key_sets = [_key_tuples(_voxel_keys(view.points_world)) for view in views]
    union = sorted(set().union(*key_sets))
    if not union:
        return [len(views) - 1], 0.0
    step = max(int(round(1.0 / KEY_POINT_RATE)), 1)
    sampled = union[::step]
    if len(sampled) > MAX_KEY_POINTS:
        indices = np.linspace(0, len(sampled) - 1, MAX_KEY_POINTS, dtype=np.int64)
        sampled = [sampled[int(index)] for index in indices]
    uncovered = set(sampled)
    chosen: list[int] = []
    while uncovered:
        remaining = [index for index in range(len(views)) if index not in chosen]
        if not remaining:
            break
        winner = max(
            remaining,
            key=lambda index: (
                len(uncovered.intersection(key_sets[index])),
                views[index].mask_confidence,
                views[index].hb_confidence,
                views[index].frame_ordinal,
                -index,
            ),
        )
        gain = uncovered.intersection(key_sets[winner])
        if not gain:
            break
        chosen.append(winner)
        uncovered.difference_update(gain)
    if not chosen:
        chosen = [len(views) - 1]
    return chosen, 1.0 - len(uncovered) / max(len(sampled), 1)


def _component_touching_latest(views: Sequence[TrackView], selected: Sequence[int]) -> list[int]:
    if len(selected) <= 1:
        return list(selected)
    adjacency: dict[int, set[int]] = {index: set() for index in selected}
    for left_position, left in enumerate(selected):
        for right in selected[left_position + 1 :]:
            iou, containment, nd = _view_overlap(views[left], views[right])
            if nd <= MASK_MERGE_ND_MAX and (
                iou >= MASK_MERGE_IOU or containment >= MASK_MERGE_CONTAINMENT
            ):
                adjacency[left].add(right)
                adjacency[right].add(left)
    latest = max(range(len(views)), key=lambda index: (views[index].frame_ordinal, index))
    anchor = max(
        selected,
        key=lambda index: (
            _view_overlap(views[index], views[latest])[0],
            _view_overlap(views[index], views[latest])[1],
            -_view_overlap(views[index], views[latest])[2],
            views[index].frame_ordinal,
        ),
    )
    reached = {anchor}
    stack = [anchor]
    while stack:
        current = stack.pop()
        for neighbor in adjacency[current]:
            if neighbor not in reached:
                reached.add(neighbor)
                stack.append(neighbor)
    return [index for index in selected if index in reached]


def _pmr_refine(views: Sequence[TrackView]) -> tuple[np.ndarray, float, float]:
    all_points = np.concatenate([view.points_world for view in views], axis=0)
    all_keys = np.floor(all_points / VOXEL_SIZE_M).astype(np.int64)
    unique_keys, inverse = np.unique(all_keys, axis=0, return_inverse=True)
    support = np.zeros(len(unique_keys), dtype=np.int16)
    preferred: set[tuple[int, int, int]] = set()
    for view in views:
        view_keys = _voxel_keys(view.points_world)
        view_set = _key_tuples(view_keys)
        preferred.update(view_set)
        lookup = {tuple(map(int, key)): index for index, key in enumerate(unique_keys)}
        for key in view_set:
            index = lookup.get(key)
            if index is not None:
                support[index] += 1
    seed_indices = np.flatnonzero(support >= min(2, len(views)))
    seed_fraction = float(len(seed_indices) / max(len(unique_keys), 1))
    if len(seed_indices):
        retained = set(int(value) for value in seed_indices)
        lookup = {tuple(map(int, key)): index for index, key in enumerate(unique_keys)}
        frontier = set(retained)
        for _ in range(PMR_ITERATIONS):
            next_frontier: set[int] = set()
            for index in frontier:
                x, y, z = map(int, unique_keys[index])
                for dx, dy, dz in _NEIGHBOURS:
                    neighbor = lookup.get((x + dx, y + dy, z + dz))
                    if neighbor is not None and neighbor not in retained:
                        retained.add(neighbor)
                        next_frontier.add(neighbor)
            if not next_frontier:
                break
            frontier = next_frontier
        candidate_indices = np.asarray(sorted(retained), dtype=np.int64)
        component_local = _largest_component(
            unique_keys[candidate_indices],
            preferred={tuple(map(int, unique_keys[index])) for index in seed_indices},
        )
        retained_indices = candidate_indices[component_local]
    else:
        retained_indices = _largest_component(unique_keys, preferred)
    if len(retained_indices) < MIN_COMPONENT_VOXELS:
        retained_indices = np.arange(len(unique_keys), dtype=np.int64)
    keep_voxels = np.zeros(len(unique_keys), dtype=np.bool_)
    keep_voxels[retained_indices] = True
    keep_points = keep_voxels[inverse]
    refined = np.ascontiguousarray(all_points[keep_points])
    if len(refined) < MIN_COMPONENT_VOXELS:
        refined = np.ascontiguousarray(all_points)
    return refined, seed_fraction, float(len(refined) / max(len(all_points), 1))


def _hypothesis_quality(
    corners: np.ndarray,
    points: np.ndarray,
    views: Sequence[TrackView],
    *,
    prior: float,
) -> float:
    inside = float(np.mean(points_inside_obb(points, corners, 1.0)))
    agreements = [aabb_overlap(corners, view.hb_corners)[0] for view in views]
    agreement = float(np.median(agreements)) if agreements else 0.0
    lower, upper = np.quantile(points, [0.02, 0.98], axis=0)
    point_volume = float(np.prod(np.maximum(upper - lower, MIN_OUTPUT_EXTENT_M)))
    _, _, extent = _obb_axes(corners)
    box_volume = float(np.prod(extent))
    compactness = math.exp(-abs(math.log(max(box_volume, 1.0e-9) / max(point_volume, 1.0e-9))))
    return _geometric_mean((inside, max(agreement, 0.01), compactness, prior))


def build_track_geometry(
    views: Sequence[TrackView],
    *,
    preferred_hb_source_id: str | None = None,
    local_window_keyframes: int = LOCAL_WINDOW_KEYFRAMES,
) -> TrackGeometry:
    if not views:
        raise ValueError("track must contain at least one view")
    ordered = sorted(views, key=lambda row: (row.frame_ordinal, row.source_id))
    if len({row.source_id for row in ordered}) != len(ordered):
        raise ValueError("track source identities must be unique")
    decision_ordinal = max(row.frame_ordinal for row in ordered)
    lower_ordinal = decision_ordinal - int(local_window_keyframes) + 1
    local = [row for row in ordered if row.frame_ordinal >= lower_ordinal]
    if len(local) > MAX_RETAINED_VIEWS:
        local = local[-MAX_RETAINED_VIEWS:]
    selected_indices, cover_fraction = _set_cover(local)
    selected_indices = _component_touching_latest(local, selected_indices)
    selected = [local[index] for index in selected_indices]
    if not selected:
        selected = [local[-1]]

    if preferred_hb_source_id is not None:
        preferred = next(
            (row for row in selected if row.source_id == preferred_hb_source_id), None
        )
    else:
        preferred = None
    if preferred is None:
        medoid_rows = []
        for index, view in enumerate(selected):
            comparisons = [
                _view_overlap(view, other)
                for other in selected
                if other.source_id != view.source_id
            ]
            median_iou = float(np.median([row[0] for row in comparisons])) if comparisons else 0.0
            median_containment = (
                float(np.median([row[1] for row in comparisons])) if comparisons else 0.0
            )
            median_nd = float(np.median([row[2] for row in comparisons])) if comparisons else 0.0
            medoid_rows.append(
                ((median_iou, median_containment, -median_nd, view.hb_confidence, -index), view)
            )
        preferred = max(medoid_rows, key=lambda row: row[0])[1]

    refined, seed_fraction, retained_fraction = _pmr_refine(selected)
    hb = np.array(preferred.hb_corners, copy=True)
    hypotheses: dict[str, np.ndarray] = {"HB": hb}
    hb_fit = _guarded_hb_fit(refined, hb)
    if hb_fit is not None:
        hypotheses["PMR_HB"] = hb_fit
    yaw = _yaw_obb(refined)
    _, _, yaw_extent = _obb_axes(yaw)
    yaw_volume = float(np.prod(yaw_extent))
    if np.all(yaw_extent <= MAX_OUTPUT_EXTENT_M) and yaw_volume <= MAX_OUTPUT_VOLUME_M3:
        hypotheses["PMR_YAW"] = yaw
    priors = {"HB": 1.0, "PMR_HB": 0.90, "PMR_YAW": 0.80}
    qualities = {
        name: _hypothesis_quality(corners, refined, selected, prior=priors[name])
        for name, corners in hypotheses.items()
    }
    chosen_name = max(
        qualities,
        key=lambda name: (qualities[name], priors[name], name == "HB"),
    )
    chosen = hypotheses[chosen_name]

    comparisons = []
    centers = np.stack([row.hb_corners.mean(axis=0) for row in selected])
    for left_index, left in enumerate(selected):
        for right in selected[left_index + 1 :]:
            comparisons.append(_view_overlap(left, right))
    median_iou = float(np.median([row[0] for row in comparisons])) if comparisons else 0.0
    median_containment = (
        float(np.median([row[1] for row in comparisons])) if comparisons else 0.0
    )
    center_rms = float(
        np.sqrt(np.mean(np.sum((centers - centers.mean(axis=0)) ** 2, axis=1)))
    )
    inside_hb = float(np.mean(points_inside_obb(refined, hb, 1.0)))
    mask_confidence = float(np.mean([row.mask_confidence for row in selected]))
    hb_confidence = float(np.mean([row.hb_confidence for row in selected]))
    view_term = min(len({row.frame_id for row in selected}) / 3.0, 1.0)
    agreement_term = median_iou if len(selected) > 1 else 0.05
    preliminary = _geometric_mean(
        (
            mask_confidence,
            hb_confidence,
            max(view_term, 0.05),
            max(cover_fraction, 0.05),
            max(agreement_term, 0.05),
            max(inside_hb, 0.05),
            max(retained_fraction, 0.05),
            qualities[chosen_name],
        )
    )
    for value in hypotheses.values():
        value.setflags(write=False)
    chosen.setflags(write=False)
    refined.setflags(write=False)
    return TrackGeometry(
        source_ids=tuple(row.source_id for row in local),
        frame_ids=tuple(row.frame_id for row in local),
        decision_frame_id=max(row.frame_id for row in ordered),
        decision_frame_ordinal=decision_ordinal,
        selected_source_ids=tuple(row.source_id for row in selected),
        hb_source_id=preferred.source_id,
        hypotheses=hypotheses,
        hypothesis_quality=qualities,
        chosen_hypothesis=chosen_name,
        corners=chosen,
        refined_points=refined,
        distinct_view_count=len({row.frame_id for row in selected}),
        set_cover_fraction=cover_fraction,
        median_pairwise_hb_iou=median_iou,
        median_pairwise_hb_containment=median_containment,
        hb_center_rms_m=center_rms,
        point_inside_hb_fraction=inside_hb,
        pmr_seed_fraction=seed_fraction,
        pmr_retained_fraction=retained_fraction,
        mask_confidence_mean=mask_confidence,
        hb_confidence_mean=hb_confidence,
        preliminary_score=preliminary,
    )


def summarize_semantic_evidence(receipt: Mapping[str, Any]) -> dict[str, Any]:
    records_raw = receipt.get("views")
    records = [row for row in records_raw if isinstance(row, Mapping)] if isinstance(records_raw, list) else []
    matched = [row for row in records if row.get("matched") is True]
    strong = [row for row in matched if row.get("strong") is True]
    labels = [str(row.get("sam3_label")) for row in matched if row.get("sam3_label") is not None]
    if labels:
        counts = {label: labels.count(label) for label in sorted(set(labels))}
        dominant_label, dominant_count = max(counts.items(), key=lambda row: (row[1], row[0]))
        consistency = dominant_count / len(labels)
    else:
        dominant_label, dominant_count, consistency = None, 0, 0.0
    scores = [float(row.get("sam3_score", 0.0)) for row in matched]
    containments = [float(row.get("mask_containment", 0.0)) for row in matched]
    coverages = [float(row.get("box_coverage", 0.0)) for row in matched]
    evidences = [float(row.get("evidence_score", 0.0)) for row in matched]
    selected_count = int(receipt.get("selected_view_count", len(records)))
    matched_fraction = len(matched) / max(selected_count, 1)
    strong_fraction = len(strong) / max(selected_count, 1)
    if matched:
        quality = _geometric_mean(
            (
                max(float(np.median(scores)), 0.01),
                max(consistency, 0.01),
                max(float(np.median(containments)), 0.01),
                max(float(np.median(coverages)), 0.01),
                max(float(np.median(evidences)), 0.01),
                max(matched_fraction, 0.01),
                max(strong_fraction, 0.01),
            )
        )
    else:
        quality = 0.02
    return {
        "selected_view_count": selected_count,
        "matched_view_count": len(matched),
        "strong_view_count": len(strong),
        "dominant_label": dominant_label,
        "dominant_label_votes": dominant_count,
        "label_consistency": consistency,
        "median_sam3_score": float(np.median(scores)) if scores else 0.0,
        "median_mask_containment": float(np.median(containments)) if containments else 0.0,
        "median_box_coverage": float(np.median(coverages)) if coverages else 0.0,
        "median_evidence_score": float(np.median(evidences)) if evidences else 0.0,
        "semantic_quality": quality,
    }


def continuous_evidence_score(
    geometry: TrackGeometry,
    semantic: Mapping[str, Any] | None,
    *,
    duplication_risk: float = 0.0,
) -> float:
    risk = float(np.clip(duplication_risk, 0.0, 1.0))
    if semantic is None:
        base = geometry.preliminary_score
    else:
        semantic_quality = float(semantic.get("semantic_quality", 0.02))
        base = _geometric_mean((geometry.preliminary_score, max(semantic_quality, 0.01)))
    return float(np.clip(base * math.sqrt(max(1.0 - risk, 1.0e-4)), 0.0, 1.0))


def policy_receipt() -> dict[str, Any]:
    return {
        "voxel_size_m": VOXEL_SIZE_M,
        "local_window_keyframes": LOCAL_WINDOW_KEYFRAMES,
        "max_retained_views": MAX_RETAINED_VIEWS,
        "key_point_rate": KEY_POINT_RATE,
        "max_key_points": MAX_KEY_POINTS,
        "mask_merge_iou": MASK_MERGE_IOU,
        "mask_merge_containment": MASK_MERGE_CONTAINMENT,
        "mask_merge_nd_max": MASK_MERGE_ND_MAX,
        "pmr_iterations": PMR_ITERATIONS,
        "pmr_center_shift_max_m": PMR_CENTER_SHIFT_MAX_M,
        "pmr_extent_ratio_range": list(PMR_EXTENT_RATIO_RANGE),
        "pmr_volume_ratio_range": list(PMR_VOLUME_RATIO_RANGE),
        "score": "equal-weight geometric means; no learned/calibrated parameters",
    }


__all__ = [
    "LOCAL_WINDOW_KEYFRAMES",
    "TrackGeometry",
    "TrackView",
    "aabb_overlap",
    "build_track_geometry",
    "continuous_evidence_score",
    "normalized_center_distance",
    "points_inside_obb",
    "policy_receipt",
    "summarize_semantic_evidence",
]
