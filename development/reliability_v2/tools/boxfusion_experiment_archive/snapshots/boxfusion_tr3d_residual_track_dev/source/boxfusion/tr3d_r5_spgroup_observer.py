"""Paired local grouping evidence for R3 anchor/candidate boxes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .spgroup_feature_cache import SPGroupFeatureSidecar
from .spgroup_partition_cache import SPGroupPartition


METRIC_NAMES = (
    "mesh_vertex_count",
    "partition_group_count",
    "feature_group_count",
    "dominant_group_fraction",
    "normalized_partition_entropy",
    "partition_completeness",
    "feature_coverage",
    "embedding_cohesion",
    "boundary_feature_contrast",
    "vote_dispersion",
    "center_dispersion_over_box_diagonal",
)


@dataclass(frozen=True)
class R5SPGroupObservation:
    metrics: np.ndarray
    metric_valid: np.ndarray
    candidate_minus_anchor: np.ndarray


def points_in_yaw_box(points: np.ndarray, box: np.ndarray, scale: float = 1.0) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    box = np.asarray(box, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or box.shape != (7,):
        raise ValueError("points_in_yaw_box expects [N,3] points and [7] box")
    if not np.isfinite(points).all() or not np.isfinite(box).all() or np.any(box[3:6] <= 0):
        raise ValueError("box/points are non-finite or box extent is invalid")
    delta = points - box[:3]
    cosine, sine = np.cos(box[6]), np.sin(box[6])
    local_x = cosine * delta[:, 0] + sine * delta[:, 1]
    local_y = -sine * delta[:, 0] + cosine * delta[:, 1]
    local = np.column_stack((local_x, local_y, delta[:, 2]))
    return np.all(np.abs(local) <= box[3:6] * (0.5 * float(scale)) + 1e-6, axis=1)


def _safe_entropy(weights: np.ndarray) -> float:
    if len(weights) <= 1:
        return 0.0
    positive = weights[weights > 0]
    return float(-(positive * np.log(positive)).sum() / np.log(len(weights)))


def _unalign(points: np.ndarray, alignment: np.ndarray) -> np.ndarray:
    inverse = np.linalg.inv(alignment)
    return points @ inverse[:3, :3].T + inverse[:3, 3]


def _box_metrics(
    partition: SPGroupPartition,
    feature_sidecar: SPGroupFeatureSidecar,
    box: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    values = np.zeros(len(METRIC_NAMES), dtype=np.float32)
    valid = np.zeros(len(METRIC_NAMES), dtype=np.bool_)
    inside = points_in_yaw_box(partition.vertices_unaligned, box)
    vertex_count = int(np.count_nonzero(inside))
    values[0], valid[0] = vertex_count, True
    if vertex_count == 0:
        return values, valid

    ids = partition.superpoint_ids.astype(np.int64, copy=False)
    total = np.bincount(ids, minlength=partition.superpoint_count)
    inner = np.bincount(ids[inside], minlength=partition.superpoint_count)
    present = np.flatnonzero(inner)
    inner_counts = inner[present].astype(np.float64)
    weights = inner_counts / float(vertex_count)
    values[1], valid[1] = len(present), True
    values[3], valid[3] = float(weights.max()), True
    values[4], valid[4] = _safe_entropy(weights), True
    coverage = inner_counts / np.maximum(total[present], 1)
    values[5], valid[5] = float(np.sum(weights * coverage)), True

    features = feature_sidecar.features
    mapping = {int(identifier): row for row, identifier in enumerate(features.superpoint_ids.tolist())}
    selected = [(identifier, mapping[int(identifier)]) for identifier in present if int(identifier) in mapping]
    values[2], valid[2] = len(selected), True
    if not selected:
        return values, valid
    selected_ids = np.asarray([item[0] for item in selected], dtype=np.int64)
    selected_rows = np.asarray([item[1] for item in selected], dtype=np.int64)
    selected_inner = inner[selected_ids].astype(np.float64)
    selected_weights = selected_inner / selected_inner.sum()
    values[6], valid[6] = float(selected_inner.sum() / vertex_count), True

    embeddings = features.embeddings[selected_rows].astype(np.float64)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normalized = embeddings / np.maximum(norms, 1e-12)
    centroid = np.sum(selected_weights[:, None] * normalized, axis=0)
    centroid /= max(float(np.linalg.norm(centroid)), 1e-12)
    similarities = normalized @ centroid
    cohesion = float(np.sum(selected_weights * similarities))
    values[7], valid[7] = cohesion, True

    centers_unaligned = _unalign(features.centers_aligned.astype(np.float64), partition.axis_alignment)
    expanded = points_in_yaw_box(centers_unaligned, box, scale=1.5)
    original = points_in_yaw_box(centers_unaligned, box, scale=1.0)
    outside_rows = np.flatnonzero(expanded & ~original)
    if len(outside_rows):
        outside = features.embeddings[outside_rows].astype(np.float64)
        outside /= np.maximum(np.linalg.norm(outside, axis=1, keepdims=True), 1e-12)
        values[8], valid[8] = cohesion - float(np.max(outside @ centroid)), True

    vote_std = np.linalg.norm(features.vote_offset_std[selected_rows].astype(np.float64), axis=1)
    values[9], valid[9] = float(np.sum(selected_weights * vote_std)), True
    selected_centers = centers_unaligned[selected_rows]
    center = np.sum(selected_weights[:, None] * selected_centers, axis=0)
    dispersion = np.linalg.norm(selected_centers - center, axis=1)
    diagonal = max(float(np.linalg.norm(np.asarray(box[3:6], dtype=np.float64))), 1e-8)
    values[10], valid[10] = float(np.sum(selected_weights * dispersion) / diagonal), True
    return values, valid


def observe_pairs(
    partition: SPGroupPartition,
    feature_sidecar: SPGroupFeatureSidecar,
    anchor_boxes: np.ndarray,
    candidate_boxes: np.ndarray,
) -> R5SPGroupObservation:
    anchors = np.asarray(anchor_boxes, dtype=np.float32)
    candidates = np.asarray(candidate_boxes, dtype=np.float32)
    if anchors.shape != candidates.shape or anchors.ndim != 2 or anchors.shape[1] != 7:
        raise ValueError("paired boxes must both be [P,7]")
    if partition.scene_id != feature_sidecar.scene_id:
        raise ValueError("partition/feature scene identity mismatch")
    metrics = np.zeros((len(anchors), 2, len(METRIC_NAMES)), dtype=np.float32)
    valid = np.zeros_like(metrics, dtype=np.bool_)
    for row, (anchor, candidate) in enumerate(zip(anchors, candidates)):
        metrics[row, 0], valid[row, 0] = _box_metrics(partition, feature_sidecar, anchor)
        metrics[row, 1], valid[row, 1] = _box_metrics(partition, feature_sidecar, candidate)
    both = valid[:, 0] & valid[:, 1]
    delta = metrics[:, 1] - metrics[:, 0]
    delta[~both] = 0.0
    return R5SPGroupObservation(metrics=metrics, metric_valid=valid, candidate_minus_anchor=delta)
