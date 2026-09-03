"""Observer-only TR3D candidates near frozen G0 anchors.

R3 does not refine, replace, score, or emit a detection.  It only associates
the exact R2a/R2b proposal rows with frozen G0 prediction rows and records the
evidence required by a later ground-truth-only counterfactual audit.

Association is performed in ScanNet's axis-aligned coordinate system.  The
``axisAlignment`` matrix is ordinary scene input metadata used by the
detector/evaluator, not a ground-truth annotation.  A row belongs to the
fixed ``near`` split iff its maximum anchor AABB IoU is strictly greater than
0.15.  This split is intentionally a constant rather than a tunable
validation parameter.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import pickle

import numpy as np


TR3D_R3_NEAR_ANCHOR_IOU = 0.15
DEPTH_EVIDENCE_DIM = 4


def load_frozen_anchor_prediction(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Load a trusted, manifest-pinned BoxFusion prediction pickle."""

    source = Path(path)
    with source.open("rb") as handle:
        payload = pickle.load(handle)  # noqa: S301 - content-addressed local output
    if (
        not isinstance(payload, (list, tuple))
        or len(payload) != 1
        or not isinstance(payload[0], (list, tuple))
    ):
        raise ValueError(f"{source}: malformed BoxFusion prediction")
    corners: list[np.ndarray] = []
    scores: list[float] = []
    for index, detection in enumerate(payload[0]):
        if not isinstance(detection, (list, tuple)) or len(detection) != 3:
            raise ValueError(f"{source}: malformed detection {index}")
        geometry = np.asarray(detection[1], dtype=np.float64)
        score = float(detection[2])
        if geometry.shape != (8, 3) or not np.isfinite(geometry).all():
            raise ValueError(f"{source}: invalid corners {index}")
        if not math.isfinite(score):
            raise ValueError(f"{source}: invalid score {index}")
        corners.append(geometry)
        scores.append(score)
    return (
        np.stack(corners) if corners else np.empty((0, 8, 3), dtype=np.float64),
        np.asarray(scores, dtype=np.float64),
    )


def load_axis_alignment_input_metadata(path: str | Path) -> np.ndarray:
    """Read ScanNet ``axisAlignment`` from ordinary scene input metadata."""

    source = Path(path)
    values = None
    for line in source.read_text(encoding="utf-8").splitlines():
        if line.strip().startswith("axisAlignment"):
            values = np.fromstring(line.split("=", 1)[1], sep=" ")
            break
    if values is None or values.size != 16 or not np.isfinite(values).all():
        raise ValueError(f"{source}: missing/invalid axisAlignment input metadata")
    return _homogeneous_matrix(values.reshape(4, 4))


def _readonly(value: np.ndarray, dtype: np.dtype | None = None) -> np.ndarray:
    result = np.ascontiguousarray(value, dtype=dtype)
    result.setflags(write=False)
    return result


def _homogeneous_matrix(value: object) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64)
    if (
        matrix.shape != (4, 4)
        or not np.isfinite(matrix).all()
        or not np.allclose(matrix[3], [0, 0, 0, 1], rtol=0.0, atol=1e-8)
    ):
        raise ValueError("axis_alignment must be a finite homogeneous [4,4] matrix")
    return matrix


def axis_aligned_minmax(
    corners_world: object, axis_alignment: object
) -> np.ndarray:
    """Transform unaligned-world corners and return axis-aligned min/max."""

    corners = np.asarray(corners_world, dtype=np.float64)
    if corners.ndim != 3 or corners.shape[1:] != (8, 3):
        raise ValueError("corners_world must have shape [N,8,3]")
    if not np.isfinite(corners).all():
        raise ValueError("corners_world must be finite")
    matrix = _homogeneous_matrix(axis_alignment)
    aligned = corners @ matrix[:3, :3].T + matrix[None, None, :3, 3]
    if not len(aligned):
        return np.empty((0, 6), dtype=np.float64)
    result = np.concatenate((aligned.min(axis=1), aligned.max(axis=1)), axis=1)
    if np.any(result[:, 3:] <= result[:, :3]):
        raise ValueError("corners must define positive-volume aligned AABBs")
    return result


def pairwise_aabb_iou(left: object, right: object) -> np.ndarray:
    lhs = np.asarray(left, dtype=np.float64)
    rhs = np.asarray(right, dtype=np.float64)
    if lhs.ndim != 2 or lhs.shape[1] != 6:
        raise ValueError("left boxes must have shape [N,6]")
    if rhs.ndim != 2 or rhs.shape[1] != 6:
        raise ValueError("right boxes must have shape [M,6]")
    if not np.isfinite(lhs).all() or not np.isfinite(rhs).all():
        raise ValueError("AABB coordinates must be finite")
    if (len(lhs) and np.any(lhs[:, 3:] <= lhs[:, :3])) or (
        len(rhs) and np.any(rhs[:, 3:] <= rhs[:, :3])
    ):
        raise ValueError("AABBs must have positive volume")
    if not len(lhs) or not len(rhs):
        return np.zeros((len(lhs), len(rhs)), dtype=np.float64)
    size = np.maximum(
        np.minimum(lhs[:, None, 3:], rhs[None, :, 3:])
        - np.maximum(lhs[:, None, :3], rhs[None, :, :3]),
        0.0,
    )
    intersection = np.prod(size, axis=2)
    lhs_volume = np.prod(lhs[:, 3:] - lhs[:, :3], axis=1)
    rhs_volume = np.prod(rhs[:, 3:] - rhs[:, :3], axis=1)
    union = lhs_volume[:, None] + rhs_volume[None] - intersection
    return np.divide(
        intersection,
        union,
        out=np.zeros_like(intersection),
        where=union > 0,
    )


@dataclass(frozen=True)
class TR3DR3NearObservation:
    proposal_ids: np.ndarray
    lineage_ids: np.ndarray
    proposal_corners_world: np.ndarray
    anchor_index: np.ndarray
    anchor_iou: np.ndarray
    center_distance_m: np.ndarray
    center_distance_over_anchor_diagonal: np.ndarray
    volume_ratio: np.ndarray
    tr3d_score: np.ndarray
    anchor_score: np.ndarray
    point_count: np.ndarray
    point_density_m3: np.ndarray
    r2a_evidence_available: np.ndarray
    r2a_depth_evidence: np.ndarray
    r2a_depth_quality: np.ndarray
    r2a_view_count: np.ndarray
    r2a_point_count: np.ndarray
    r2b_feature_available: np.ndarray
    r2b_multiview_available: np.ndarray
    r2b_feature_view_count: np.ndarray
    r2b_pairwise_cosine_count: np.ndarray
    r2b_pairwise_cosine_mean: np.ndarray
    r2b_pairwise_cosine_median: np.ndarray
    r2b_pairwise_cosine_min: np.ndarray
    r2b_pairwise_cosine_max: np.ndarray
    r2b_pairwise_cosine_std: np.ndarray

    @property
    def proposal_count(self) -> int:
        return int(np.asarray(self.proposal_ids).shape[0])


def observe_anchor_near_candidates(
    *,
    proposal_ids: object,
    lineage_ids: object,
    proposal_corners_world: object,
    tr3d_score: object,
    point_count: object,
    anchor_corners_world: object,
    anchor_score: object,
    axis_alignment: object,
    r2a_evidence_available: object,
    r2a_depth_evidence: object,
    r2a_view_count: object,
    r2a_point_count: object,
    r2b_feature_view_count: object,
    r2b_pairwise_cosine_count: object,
    r2b_pairwise_cosine_mean: object,
    r2b_pairwise_cosine_median: object,
    r2b_pairwise_cosine_min: object,
    r2b_pairwise_cosine_max: object,
    r2b_pairwise_cosine_std: object,
    near_anchor_iou: float = TR3D_R3_NEAR_ANCHOR_IOU,
) -> TR3DR3NearObservation:
    """Associate exact parent proposal rows with frozen G0 anchors.

    Maximum AABB IoU is the primary association key.  Exact IoU ties use the
    nearest aligned-box centre and then the stable anchor index.  Consequently
    zero-overlap rows still have a deterministic nearest anchor association,
    though they can never enter the fixed near split.
    """

    if not math.isclose(
        float(near_anchor_iou),
        TR3D_R3_NEAR_ANCHOR_IOU,
        rel_tol=0.0,
        abs_tol=0.0,
    ):
        raise ValueError("R3 near-anchor IoU split is frozen at 0.15")

    ids = np.asarray(proposal_ids)
    lineage = np.asarray(lineage_ids)
    corners = np.asarray(proposal_corners_world)
    scores = np.asarray(tr3d_score)
    points = np.asarray(point_count)
    depth = np.asarray(r2a_depth_evidence)
    depth_available = np.asarray(r2a_evidence_available)
    depth_views = np.asarray(r2a_view_count)
    depth_points = np.asarray(r2a_point_count)
    feature_views = np.asarray(r2b_feature_view_count)
    cosine_count = np.asarray(r2b_pairwise_cosine_count)
    cosine_arrays = {
        "mean": np.asarray(r2b_pairwise_cosine_mean),
        "median": np.asarray(r2b_pairwise_cosine_median),
        "min": np.asarray(r2b_pairwise_cosine_min),
        "max": np.asarray(r2b_pairwise_cosine_max),
        "std": np.asarray(r2b_pairwise_cosine_std),
    }
    count = int(ids.shape[0]) if ids.ndim == 1 else -1
    expected_vectors = {
        "lineage_ids": lineage,
        "tr3d_score": scores,
        "point_count": points,
        "r2a_evidence_available": depth_available,
        "r2a_view_count": depth_views,
        "r2a_point_count": depth_points,
        "r2b_feature_view_count": feature_views,
        "r2b_pairwise_cosine_count": cosine_count,
        **{f"r2b_pairwise_cosine_{name}": value for name, value in cosine_arrays.items()},
    }
    if count < 0 or corners.shape != (count, 8, 3) or depth.shape != (
        count,
        DEPTH_EVIDENCE_DIM,
    ):
        raise ValueError("proposal evidence rows must align with proposal_ids")
    for name, value in expected_vectors.items():
        if value.shape != (count,):
            raise ValueError(f"{name} must have shape [N]")
    if ids.dtype != np.int64 or lineage.dtype != np.int64:
        raise ValueError("proposal_ids and lineage_ids must be int64")
    if len(np.unique(ids)) != count or len(np.unique(lineage)) != count:
        raise ValueError("proposal_ids and lineage_ids must be unique")
    if points.dtype != np.int32 or depth_views.dtype != np.int32:
        raise ValueError("point_count and r2a_view_count must be int32")
    if depth_points.dtype != np.int64:
        raise ValueError("r2a_point_count must be int64")
    if depth_available.dtype != np.bool_:
        raise ValueError("r2a_evidence_available must be bool")
    if feature_views.dtype != np.int32 or cosine_count.dtype != np.int32:
        raise ValueError("R2b view/pair counts must be int32")
    numeric = [corners, scores, depth, *cosine_arrays.values()]
    if any(not np.isfinite(value).all() for value in numeric):
        raise ValueError("proposal evidence must be finite")
    if (
        np.any(points < 0)
        or np.any(depth_views < 0)
        or np.any(depth_points < 0)
        or np.any(feature_views < 0)
        or np.any(cosine_count < 0)
    ):
        raise ValueError("evidence counts must be nonnegative")
    if np.any(depth < 0.0) or np.any(depth > 1.0):
        raise ValueError("R2a depth evidence must lie in [0,1]")
    if (
        np.any(depth[~depth_available] != 0)
        or np.any(depth_views[~depth_available] != 0)
        or np.any(depth_points[~depth_available] != 0)
    ):
        raise ValueError("missing R2a rows must use zero evidence sentinels")
    if np.any(cosine_count != feature_views * (feature_views - 1) // 2):
        raise ValueError("R2b pair count disagrees with feature view count")
    no_pairs = cosine_count == 0
    if any(np.any(value[no_pairs] != 0) for value in cosine_arrays.values()):
        raise ValueError("R2b rows without pairs must use zero cosine sentinels")
    if any(
        np.any(value < -1.0) or np.any(value > 1.0)
        for name, value in cosine_arrays.items()
        if name != "std"
    ) or np.any(cosine_arrays["std"] < 0.0) or np.any(
        cosine_arrays["std"] > 1.0
    ):
        raise ValueError("R2b cosine statistics are outside their physical range")

    anchor_corners = np.asarray(anchor_corners_world, dtype=np.float64)
    anchor_scores = np.asarray(anchor_score, dtype=np.float64)
    if anchor_corners.ndim != 3 or anchor_corners.shape[1:] != (8, 3):
        raise ValueError("anchor_corners_world must have shape [A,8,3]")
    if anchor_scores.shape != (len(anchor_corners),):
        raise ValueError("anchor_score must have shape [A]")
    if not np.isfinite(anchor_scores).all():
        raise ValueError("anchor scores must be finite")

    candidate_boxes = axis_aligned_minmax(corners, axis_alignment)
    anchor_boxes = axis_aligned_minmax(anchor_corners, axis_alignment)
    if not count or not len(anchor_boxes):
        selected = np.empty(0, dtype=np.int64)
        associated = np.empty(0, dtype=np.int64)
        best_iou = np.empty(0, dtype=np.float64)
        distances = np.empty(0, dtype=np.float64)
    else:
        iou = pairwise_aabb_iou(candidate_boxes, anchor_boxes)
        candidate_centres = (candidate_boxes[:, :3] + candidate_boxes[:, 3:]) * 0.5
        anchor_centres = (anchor_boxes[:, :3] + anchor_boxes[:, 3:]) * 0.5
        all_distances = np.linalg.norm(
            candidate_centres[:, None] - anchor_centres[None, :, :], axis=2
        )
        associated = np.empty(count, dtype=np.int64)
        best_iou = iou.max(axis=1)
        for row in range(count):
            tied = np.flatnonzero(iou[row] == best_iou[row])
            local_distances = all_distances[row, tied]
            associated[row] = int(tied[int(np.argmin(local_distances))])
        distances = all_distances[np.arange(count), associated]
        selected = np.flatnonzero(best_iou > TR3D_R3_NEAR_ANCHOR_IOU)

    if len(selected):
        selected_anchor = associated[selected]
        anchor_size = anchor_boxes[selected_anchor, 3:] - anchor_boxes[selected_anchor, :3]
        anchor_diagonal = np.linalg.norm(anchor_size, axis=1)
        candidate_volume = np.prod(
            candidate_boxes[selected, 3:] - candidate_boxes[selected, :3], axis=1
        )
        anchor_volume = np.prod(anchor_size, axis=1)
        ratio = candidate_volume / anchor_volume
        density = points[selected].astype(np.float64) / candidate_volume
        distance_ratio = distances[selected] / anchor_diagonal
        selected_anchor_scores = anchor_scores[selected_anchor]
        selected_iou = best_iou[selected]
    else:
        selected_anchor = np.empty(0, dtype=np.int64)
        ratio = density = distance_ratio = selected_anchor_scores = selected_iou = np.empty(
            0, dtype=np.float64
        )
        distances = np.empty(0, dtype=np.float64)

    selected_depth = depth[selected].astype(np.float64, copy=False)
    depth_quality = np.clip(
        selected_depth[:, 0]
        / np.maximum(1.0 - selected_depth[:, 3], 1e-6),
        0.0,
        1.0,
    )
    selected_feature_views = feature_views[selected]
    selected_pair_count = cosine_count[selected]
    return TR3DR3NearObservation(
        proposal_ids=_readonly(ids[selected], np.int64),
        lineage_ids=_readonly(lineage[selected], np.int64),
        proposal_corners_world=_readonly(corners[selected], np.float32),
        anchor_index=_readonly(selected_anchor, np.int64),
        anchor_iou=_readonly(selected_iou, np.float32),
        center_distance_m=_readonly(
            distances[selected] if len(selected) else distances, np.float32
        ),
        center_distance_over_anchor_diagonal=_readonly(distance_ratio, np.float32),
        volume_ratio=_readonly(ratio, np.float32),
        tr3d_score=_readonly(scores[selected], np.float32),
        anchor_score=_readonly(selected_anchor_scores, np.float32),
        point_count=_readonly(points[selected], np.int32),
        point_density_m3=_readonly(density, np.float32),
        r2a_evidence_available=_readonly(depth_available[selected], np.bool_),
        r2a_depth_evidence=_readonly(selected_depth, np.float32),
        r2a_depth_quality=_readonly(depth_quality, np.float32),
        r2a_view_count=_readonly(depth_views[selected], np.int32),
        r2a_point_count=_readonly(depth_points[selected], np.int64),
        r2b_feature_available=_readonly(selected_feature_views > 0, np.bool_),
        r2b_multiview_available=_readonly(selected_pair_count > 0, np.bool_),
        r2b_feature_view_count=_readonly(selected_feature_views, np.int32),
        r2b_pairwise_cosine_count=_readonly(selected_pair_count, np.int32),
        r2b_pairwise_cosine_mean=_readonly(cosine_arrays["mean"][selected], np.float32),
        r2b_pairwise_cosine_median=_readonly(cosine_arrays["median"][selected], np.float32),
        r2b_pairwise_cosine_min=_readonly(cosine_arrays["min"][selected], np.float32),
        r2b_pairwise_cosine_max=_readonly(cosine_arrays["max"][selected], np.float32),
        r2b_pairwise_cosine_std=_readonly(cosine_arrays["std"][selected], np.float32),
    )
