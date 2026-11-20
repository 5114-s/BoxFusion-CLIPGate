"""Causal track-level 7D box fusion with independent held-out acceptance.

The fitting API consumes only frozen FastSAM/F2 evidence and at most two
Boxer observations from earlier frames.  A later view selects one of three
OBB hypotheses and a still later view accepts or rejects the frozen result.
No selection or acceptance observation is ever fused back into the box.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Mapping, Sequence

import numpy as np

from boxfusion.stream3dv2_lite import TrackGeometry, aabb_overlap, points_inside_obb


IMAGE_HEIGHT = 480
IMAGE_WIDTH = 640
MASK_PACKED_BYTES = IMAGE_HEIGHT * IMAGE_WIDTH // 8
MAX_POINTS_PER_VIEW = 512
MAX_POOLED_POINTS = 2_048
_EPS = 1.0e-8
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


def _readonly(value: object, dtype: np.dtype, shape: tuple[int, ...] | None = None) -> np.ndarray:
    array = np.asarray(value, dtype=dtype)
    if shape is not None and array.shape != shape:
        raise ValueError(f"array must have shape {shape}, got {array.shape}")
    packed = np.ascontiguousarray(array).tobytes()
    return np.frombuffer(packed, dtype=dtype).reshape(array.shape)


def _finite_probability(value: object, label: str) -> float:
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(f"{label} must lie in [0,1]")
    return result


def _wrap(value: float | np.ndarray, period: float = np.pi):
    return (value + 0.5 * period) % period - 0.5 * period


def _rz(angle: float) -> np.ndarray:
    cosine, sine = math.cos(angle), math.sin(angle)
    return np.asarray(
        [[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def _safe_spd(value: np.ndarray, floor: float = 1.0e-6) -> np.ndarray:
    matrix = 0.5 * (np.asarray(value, dtype=np.float64) + np.asarray(value, dtype=np.float64).T)
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    eigenvalues = np.maximum(eigenvalues, floor)
    return (eigenvectors * eigenvalues[None, :]) @ eigenvectors.T


def _corners(center: np.ndarray, extent: np.ndarray, rotation: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(
        center[None] + (_CORNER_SIGNS * (extent[None] * 0.5)) @ rotation.T,
        dtype=np.float64,
    )


def _axis_aligned_corners(lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    return _corners((lower + upper) * 0.5, upper - lower, np.eye(3, dtype=np.float64))


def _fit_points(points: np.ndarray, rotation: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    local = points @ rotation
    lower, upper = np.quantile(local, [0.02, 0.98], axis=0)
    extent = np.maximum(upper - lower, 0.05)
    center = ((lower + upper) * 0.5) @ rotation.T
    return center, extent, _corners(center, extent, rotation)


def _pca_yaw(points: np.ndarray) -> float:
    xy = points[:, :2] - np.median(points[:, :2], axis=0, keepdims=True)
    covariance = xy.T @ xy / max(len(xy), 1)
    try:
        _, vectors = np.linalg.eigh(covariance)
        primary = vectors[:, -1]
    except np.linalg.LinAlgError:
        primary = np.asarray([1.0, 0.0], dtype=np.float64)
    return float(math.atan2(float(primary[1]), float(primary[0])))


def _geometric_mean(values: Sequence[float]) -> float:
    rows = np.clip(np.asarray(values, dtype=np.float64), 1.0e-6, 1.0)
    return float(np.exp(np.mean(np.log(rows))))


@dataclass(frozen=True)
class TrackEvidenceView:
    source_id: str
    frame_id: int
    frame_ordinal: int
    mask_confidence: float
    residual_ratio: float
    valid_ratio: float
    tight_box_xyxy: np.ndarray
    mask_packbits: np.ndarray
    points_world: np.ndarray
    world_q02: np.ndarray
    world_q98: np.ndarray
    intrinsics: np.ndarray
    camera_to_world: np.ndarray
    hb_center: np.ndarray | None = None
    hb_extent: np.ndarray | None = None
    hb_rotation: np.ndarray | None = None
    hb_confidence: float | None = None
    covariance_7d: np.ndarray | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.source_id, str) or not self.source_id:
            raise ValueError("source_id must be non-empty")
        if int(self.frame_id) < 0 or int(self.frame_ordinal) < 0:
            raise ValueError("frame identity must be non-negative")
        points = np.asarray(self.points_world, dtype=np.float64)
        if points.ndim != 2 or points.shape[1:] != (3,) or not len(points):
            raise ValueError("points_world must be non-empty [N,3]")
        if not np.isfinite(points).all():
            raise ValueError("points_world must be finite")
        if len(points) > MAX_POINTS_PER_VIEW:
            indices = np.linspace(0, len(points) - 1, MAX_POINTS_PER_VIEW, dtype=np.int64)
            points = points[indices]
        lower = np.asarray(self.world_q02, dtype=np.float64)
        upper = np.asarray(self.world_q98, dtype=np.float64)
        if lower.shape != (3,) or upper.shape != (3,) or np.any(upper <= lower):
            raise ValueError("world bounds must be non-degenerate [3]")
        K = np.asarray(self.intrinsics, dtype=np.float64)
        if K.shape == (4, 4):
            K = K[:3, :3]
        pose = np.asarray(self.camera_to_world, dtype=np.float64)
        if K.shape != (3, 3) or pose.shape != (4, 4):
            raise ValueError("intrinsics/pose shapes are invalid")
        if not np.isfinite(K).all() or not np.isfinite(pose).all():
            raise ValueError("intrinsics/pose must be finite")
        packed = np.asarray(self.mask_packbits)
        if packed.dtype != np.uint8 or packed.shape != (MASK_PACKED_BYTES,):
            raise ValueError("mask_packbits must be uint8[38400]")
        box = np.asarray(self.tight_box_xyxy, dtype=np.float64)
        if box.shape != (4,) or not np.isfinite(box).all() or box[2] <= box[0] or box[3] <= box[1]:
            raise ValueError("tight_box_xyxy must be a valid [4] box")
        hb_values = (self.hb_center, self.hb_extent, self.hb_rotation, self.hb_confidence, self.covariance_7d)
        if any(value is None for value in hb_values) and not all(value is None for value in hb_values):
            raise ValueError("HB fields must be all present or all absent")
        if self.hb_center is not None:
            center = np.asarray(self.hb_center, dtype=np.float64)
            extent = np.asarray(self.hb_extent, dtype=np.float64)
            rotation = np.asarray(self.hb_rotation, dtype=np.float64)
            covariance = np.asarray(self.covariance_7d, dtype=np.float64)
            if center.shape != (3,) or extent.shape != (3,) or rotation.shape != (3, 3):
                raise ValueError("HB geometry shapes are invalid")
            if covariance.shape != (7, 7) or not np.isfinite(covariance).all():
                raise ValueError("covariance_7d must be finite [7,7]")
            if np.any(extent <= 0.0) or not np.isfinite(center).all() or not np.isfinite(rotation).all():
                raise ValueError("HB geometry must be finite and positive")
            object.__setattr__(self, "hb_center", _readonly(center, np.float64, (3,)))
            object.__setattr__(self, "hb_extent", _readonly(extent, np.float64, (3,)))
            object.__setattr__(self, "hb_rotation", _readonly(rotation, np.float64, (3, 3)))
            object.__setattr__(self, "hb_confidence", _finite_probability(self.hb_confidence, "hb_confidence"))
            object.__setattr__(self, "covariance_7d", _readonly(_safe_spd(covariance), np.float64, (7, 7)))
        object.__setattr__(self, "frame_id", int(self.frame_id))
        object.__setattr__(self, "frame_ordinal", int(self.frame_ordinal))
        object.__setattr__(self, "mask_confidence", _finite_probability(self.mask_confidence, "mask_confidence"))
        object.__setattr__(self, "residual_ratio", _finite_probability(self.residual_ratio, "residual_ratio"))
        object.__setattr__(self, "valid_ratio", _finite_probability(self.valid_ratio, "valid_ratio"))
        object.__setattr__(self, "tight_box_xyxy", _readonly(box, np.float64, (4,)))
        object.__setattr__(self, "mask_packbits", _readonly(packed, np.uint8, (MASK_PACKED_BYTES,)))
        object.__setattr__(self, "points_world", _readonly(points, np.float64))
        object.__setattr__(self, "world_q02", _readonly(lower, np.float64, (3,)))
        object.__setattr__(self, "world_q98", _readonly(upper, np.float64, (3,)))
        object.__setattr__(self, "intrinsics", _readonly(K, np.float64, (3, 3)))
        object.__setattr__(self, "camera_to_world", _readonly(pose, np.float64, (4, 4)))

    @property
    def has_hb(self) -> bool:
        return self.hb_center is not None

    @property
    def camera_position(self) -> np.ndarray:
        return self.camera_to_world[:3, 3]

    @property
    def hb_corners(self) -> np.ndarray | None:
        if not self.has_hb:
            return None
        assert self.hb_center is not None and self.hb_extent is not None and self.hb_rotation is not None
        return _corners(self.hb_center, self.hb_extent, self.hb_rotation)


def pack_mask(mask: object) -> np.ndarray:
    array = np.asarray(mask)
    if array.shape != (IMAGE_HEIGHT, IMAGE_WIDTH):
        raise ValueError("mask must have shape [480,640]")
    if not np.all((array == 0) | (array == 1)):
        raise ValueError("mask must be binary")
    return np.packbits(np.asarray(array, dtype=np.bool_).reshape(-1), bitorder="little")


def _projected_box(corners: np.ndarray, view: TrackEvidenceView) -> np.ndarray | None:
    world_to_camera = np.linalg.inv(view.camera_to_world)
    homogeneous = np.column_stack((corners, np.ones(8, dtype=np.float64)))
    camera = (homogeneous @ world_to_camera.T)[:, :3]
    if not np.all(camera[:, 2] > 1.0e-3):
        return None
    pixels_h = camera @ view.intrinsics.T
    pixels = pixels_h[:, :2] / pixels_h[:, 2:3]
    box = np.asarray(
        [pixels[:, 0].min(), pixels[:, 1].min(), pixels[:, 0].max(), pixels[:, 1].max()],
        dtype=np.float64,
    )
    box[[0, 2]] = np.clip(box[[0, 2]], 0.0, float(IMAGE_WIDTH))
    box[[1, 3]] = np.clip(box[[1, 3]], 0.0, float(IMAGE_HEIGHT))
    return None if box[2] <= box[0] or box[3] <= box[1] else box


def _mask_metrics(box: np.ndarray | None, packed: np.ndarray) -> tuple[float, float]:
    if box is None:
        return 0.0, 0.0
    mask = np.unpackbits(packed, bitorder="little", count=IMAGE_HEIGHT * IMAGE_WIDTH).reshape(IMAGE_HEIGHT, IMAGE_WIDTH)
    x1 = max(0, min(IMAGE_WIDTH, int(np.floor(box[0]))))
    y1 = max(0, min(IMAGE_HEIGHT, int(np.floor(box[1]))))
    x2 = max(0, min(IMAGE_WIDTH, int(np.ceil(box[2]))))
    y2 = max(0, min(IMAGE_HEIGHT, int(np.ceil(box[3]))))
    box_area = max(x2 - x1, 0) * max(y2 - y1, 0)
    mask_area = int(np.count_nonzero(mask))
    intersection = int(np.count_nonzero(mask[y1:y2, x1:x2]))
    union = box_area + mask_area - intersection
    return (
        0.0 if union <= 0 else intersection / union,
        0.0 if mask_area <= 0 else intersection / mask_area,
    )


def _ray_depth_metrics(
    center: np.ndarray,
    extent: np.ndarray,
    rotation: np.ndarray,
    view: TrackEvidenceView,
) -> tuple[float, float]:
    points = np.asarray(view.points_world, dtype=np.float64)
    origin = view.camera_position
    rays = points - origin[None]
    observed = np.linalg.norm(rays, axis=1)
    valid_ray = observed > 1.0e-5
    directions = rays / np.maximum(observed[:, None], 1.0e-5)
    origin_local = (origin - center) @ rotation
    direction_local = directions @ rotation
    half = extent * 0.5
    parallel = np.abs(direction_local) <= 1.0e-9
    outside_parallel = parallel & (
        (origin_local[None] < -half[None])
        | (origin_local[None] > half[None])
    )
    safe_direction = np.where(parallel, 1.0, direction_local)
    t1 = (-half[None] - origin_local[None]) / safe_direction
    t2 = (half[None] - origin_local[None]) / safe_direction
    t1 = np.where(parallel, -np.inf, t1)
    t2 = np.where(parallel, np.inf, t2)
    near = np.max(np.minimum(t1, t2), axis=1)
    far = np.min(np.maximum(t1, t2), axis=1)
    intersects = (
        valid_ray
        & ~np.any(outside_parallel, axis=1)
        & (far >= np.maximum(near, 0.0))
    )
    not_occluded = intersects & (observed >= near - 0.05)
    denominator = max(int(np.count_nonzero(not_occluded)), 1)
    support = np.count_nonzero(not_occluded & (observed <= far + 0.05)) / denominator
    free = np.count_nonzero(not_occluded & (observed > far + 0.10)) / denominator
    return float(support), float(free)


@dataclass(frozen=True)
class HeldoutReceipt:
    source_id: str
    mask_box_iou: float
    mask_containment: float
    point_inside_fraction: float
    depth_support_fraction: float
    free_space_fraction: float
    quality: float
    passed: bool
    reasons: tuple[str, ...]


def evaluate_heldout(corners: np.ndarray, view: TrackEvidenceView) -> HeldoutReceipt:
    center, extent, rotation = corners_to_params(corners)
    projected = _projected_box(corners, view)
    mask_iou, containment = _mask_metrics(projected, view.mask_packbits)
    inside = float(np.mean(points_inside_obb(view.points_world, corners, scale=1.10)))
    support, free = _ray_depth_metrics(center, extent, rotation, view)
    quality = _geometric_mean(
        (mask_iou, containment, inside, support, max(1.0 - free, 1.0e-6))
    )
    return HeldoutReceipt(
        source_id=view.source_id,
        mask_box_iou=mask_iou,
        mask_containment=containment,
        point_inside_fraction=inside,
        depth_support_fraction=support,
        free_space_fraction=free,
        quality=quality,
        passed=False,
        reasons=(),
    )


def corners_to_params(corners: object) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    box = np.asarray(corners, dtype=np.float64)
    if box.shape != (8, 3) or not np.isfinite(box).all():
        raise ValueError("corners must be finite [8,3]")
    vectors = np.stack((box[4] - box[0], box[2] - box[0], box[1] - box[0]), axis=1)
    extent = np.linalg.norm(vectors, axis=0)
    if np.any(extent <= 1.0e-6):
        raise ValueError("corners are degenerate")
    rotation = vectors / extent[None]
    if np.linalg.det(rotation) < 0.0:
        rotation[:, -1] *= -1.0
    u, _, vh = np.linalg.svd(rotation)
    rotation = u @ vh
    return box.mean(axis=0), extent, rotation


def attach_boxer_observation(
    view: TrackEvidenceView,
    *,
    center: object,
    extent: object,
    rotation: object,
    confidence: object,
) -> TrackEvidenceView:
    center_np = np.asarray(center, dtype=np.float64)
    extent_np = np.asarray(extent, dtype=np.float64)
    rotation_np = np.asarray(rotation, dtype=np.float64)
    hb_confidence = _finite_probability(confidence, "hb_confidence")
    if center_np.shape != (3,) or extent_np.shape != (3,) or rotation_np.shape != (3, 3):
        raise ValueError("Boxer geometry shapes are invalid")
    world_to_camera = np.linalg.inv(view.camera_to_world)
    camera_points = np.column_stack((view.points_world, np.ones(len(view.points_world)))) @ world_to_camera.T
    depths = camera_points[:, 2]
    valid_depth = depths[np.isfinite(depths) & (depths > 0.1)]
    if len(valid_depth):
        median = float(np.median(valid_depth))
        depth_mad = float(1.4826 * np.median(np.abs(valid_depth - median)))
    else:
        depth_mad = 0.35
    hb_corners = _corners(center_np, extent_np, rotation_np)
    projection_iou, _ = _mask_metrics(_projected_box(hb_corners, view), view.mask_packbits)
    border = np.mean(
        [
            view.tight_box_xyxy[0] <= 1.0,
            view.tight_box_xyxy[1] <= 1.0,
            view.tight_box_xyxy[2] >= IMAGE_WIDTH - 2.0,
            view.tight_box_xyxy[3] >= IMAGE_HEIGHT - 2.0,
        ]
    )
    view_world = view.camera_position - center_np
    view_world /= max(float(np.linalg.norm(view_world)), _EPS)
    local_view = np.abs(rotation_np.T @ view_world)
    local_view /= max(float(np.linalg.norm(local_view)), _EPS)
    score_scale = 1.0 + 0.5 * (1.0 - math.sqrt(view.mask_confidence * hb_confidence))
    lateral_sigma = score_scale * np.clip(
        0.03 + 0.05 * (1.0 - view.valid_ratio) + 0.03 * border + 0.03 * (1.0 - projection_iou),
        0.025,
        0.20,
    )
    axial_sigma = score_scale * np.clip(
        0.05 + 0.75 * depth_mad + 0.08 * (1.0 - view.residual_ratio),
        0.04,
        0.50,
    )
    center_camera = np.diag([lateral_sigma**2, lateral_sigma**2, axial_sigma**2])
    center_world_cov = view.camera_to_world[:3, :3] @ center_camera @ view.camera_to_world[:3, :3].T
    # State components 3:6 are log-extents.  These uncertainty terms are
    # therefore dimensionless log-scale sigmas (not metre-valued size error):
    # every input term below is a ratio, confidence, border fraction, or view
    # direction component.
    log_size_sigma = np.clip(
        score_scale
        * (0.09 + 0.18 * (1.0 - view.valid_ratio) + 0.12 * border + 0.12 * (1.0 - projection_iou) + 0.16 * local_view),
        0.07,
        0.50,
    )
    square = 1.0 - abs(float(extent_np[0] - extent_np[1])) / max(float(np.max(extent_np[:2])), 1.0e-3)
    yaw_sigma = math.radians(
        float(np.clip(score_scale * (10.0 + 30.0 * square + 15.0 * border + 10.0 * (1.0 - projection_iou)), 8.0, 75.0))
    )
    covariance = np.zeros((7, 7), dtype=np.float64)
    covariance[:3, :3] = center_world_cov
    covariance[3:6, 3:6] = np.diag(log_size_sigma**2)
    covariance[6, 6] = yaw_sigma**2
    return replace(
        view,
        hb_center=center_np,
        hb_extent=extent_np,
        hb_rotation=rotation_np,
        hb_confidence=hb_confidence,
        covariance_7d=_safe_spd(covariance),
    )


@dataclass(frozen=True)
class TrackFitStatistics:
    center_rms_m: float
    log_size_mad_max: float
    yaw_mad_deg: float
    max_whitened_residual: float
    robust_downweighted: int
    camera_baseline_m: float
    view_ray_angle_deg: float
    max_center_std_m: float
    max_normalized_center_std: float
    max_log_size_std: float
    yaw_std_deg: float
    median_pairwise_hb_iou: float
    median_pairwise_hb_containment: float


def _fuse_hb(
    views: Sequence[TrackEvidenceView],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, TrackFitStatistics, str]:
    rows = [view for view in views if view.has_hb]
    if len(rows) < 2:
        raise ValueError("track fusion requires at least two Boxer views")
    yaw_variance = np.asarray([view.covariance_7d[6, 6] for view in rows])
    reference_index = int(np.argmin(yaw_variance))
    reference = rows[reference_index]
    assert reference.hb_rotation is not None
    reference_yaw = math.atan2(float(reference.hb_rotation[1, 0]), float(reference.hb_rotation[0, 0]))
    states = []
    covariances = []
    rotations = []
    for view in rows:
        assert view.hb_center is not None and view.hb_extent is not None and view.hb_rotation is not None and view.covariance_7d is not None
        yaw = math.atan2(float(view.hb_rotation[1, 0]), float(view.hb_rotation[0, 0]))
        extent = np.array(view.hb_extent, copy=True)
        covariance = np.array(view.covariance_7d, copy=True)
        direct = abs(float(_wrap(yaw - reference_yaw)))
        swapped = abs(float(_wrap(yaw + 0.5 * np.pi - reference_yaw)))
        if swapped < direct:
            yaw += 0.5 * np.pi
            extent[[0, 1]] = extent[[1, 0]]
            permutation = np.arange(7)
            permutation[[3, 4]] = permutation[[4, 3]]
            covariance = covariance[np.ix_(permutation, permutation)]
        yaw = reference_yaw + float(_wrap(yaw - reference_yaw))
        states.append(np.concatenate((view.hb_center, np.log(np.maximum(extent, 1.0e-3)), [yaw])))
        covariances.append(_safe_spd(covariance))
        rotations.append(view.hb_rotation)
    states_np = np.stack(states)
    cov_np = np.stack(covariances)
    order = np.argsort([view.frame_ordinal for view in rows], kind="stable")
    mean = states_np[order[0]].copy()
    fused_cov = cov_np[order[0]].copy()
    maximum_whitened = 0.0
    robust = 0
    effective_count = 1
    for index in order[1:]:
        observation = states_np[index].copy()
        observation[6] = mean[6] + float(_wrap(observation[6] - mean[6]))
        innovation = observation - mean
        innovation[6] = float(_wrap(innovation[6]))
        innovation_cov = _safe_spd(fused_cov + cov_np[index])
        whitened = float(np.sqrt(max(innovation @ np.linalg.solve(innovation_cov, innovation), 0.0)))
        maximum_whitened = max(maximum_whitened, whitened)
        huber = min(1.0, 3.0 / max(whitened, _EPS))
        observation_cov = cov_np[index] / max(huber**2, 1.0e-4)
        robust += int(huber < 1.0)
        omega = effective_count / (effective_count + 1.0)
        prior_info = np.linalg.inv(fused_cov)
        observation_info = np.linalg.inv(observation_cov)
        information = omega * prior_info + (1.0 - omega) * observation_info
        fused_cov = _safe_spd(np.linalg.inv(information))
        mean = fused_cov @ (omega * prior_info @ mean + (1.0 - omega) * observation_info @ observation)
        mean[6] = reference_yaw + float(_wrap(mean[6] - reference_yaw))
        effective_count += 1
    extent = np.exp(mean[3:6])
    rotation = _rz(float(_wrap(mean[6] - reference_yaw))) @ reference.hb_rotation
    center_scatter = states_np[:, :3] - np.mean(states_np[:, :3], axis=0, keepdims=True)
    center_rms = float(np.sqrt(np.mean(np.sum(center_scatter**2, axis=1))))
    log_size_mad = float(np.max(np.median(np.abs(states_np[:, 3:6] - np.median(states_np[:, 3:6], axis=0)), axis=0)))
    yaw_delta = _wrap(states_np[:, 6] - np.median(states_np[:, 6]))
    yaw_mad = math.degrees(float(np.median(np.abs(yaw_delta))))
    centers = [view.camera_position for view in rows]
    baseline = max((float(np.linalg.norm(a - b)) for index, a in enumerate(centers) for b in centers[index + 1 :]), default=0.0)
    rays = []
    for view in rows:
        ray = mean[:3] - view.camera_position
        ray /= max(float(np.linalg.norm(ray)), _EPS)
        rays.append(ray)
    angle = max(
        (
            math.degrees(math.acos(float(np.clip(a @ b, -1.0, 1.0))))
            for index, a in enumerate(rays)
            for b in rays[index + 1 :]
        ),
        default=0.0,
    )
    overlaps = []
    hb_corners = [view.hb_corners for view in rows]
    for index, left in enumerate(hb_corners):
        for right in hb_corners[index + 1 :]:
            assert left is not None and right is not None
            overlaps.append(aabb_overlap(left, right))
    std = np.sqrt(np.maximum(np.diag(fused_cov), 0.0))
    # The center block is expressed in world XYZ, whereas ``extent`` is in
    # the OBB's local axes.  Normalize like with like; dividing world-axis
    # standard deviations by local dimensions is not rotation invariant.
    center_cov_local = _safe_spd(rotation.T @ fused_cov[:3, :3] @ rotation)
    center_std_local = np.sqrt(np.maximum(np.diag(center_cov_local), 0.0))
    statistics = TrackFitStatistics(
        center_rms_m=center_rms,
        log_size_mad_max=log_size_mad,
        yaw_mad_deg=yaw_mad,
        max_whitened_residual=maximum_whitened,
        robust_downweighted=robust,
        camera_baseline_m=baseline,
        view_ray_angle_deg=angle,
        max_center_std_m=float(np.max(center_std_local)),
        max_normalized_center_std=float(
            np.max(center_std_local / np.maximum(extent, 0.05))
        ),
        max_log_size_std=float(np.max(std[3:6])),
        yaw_std_deg=math.degrees(float(std[6])),
        median_pairwise_hb_iou=float(np.median([row[0] for row in overlaps])) if overlaps else 0.0,
        median_pairwise_hb_containment=float(np.median([max(row[1:]) for row in overlaps])) if overlaps else 0.0,
    )
    return mean[:3], extent, rotation, fused_cov, statistics, reference.source_id


@dataclass(frozen=True)
class FrozenTrackGeometry:
    geometry: TrackGeometry
    covariance_7d: np.ndarray
    fit_statistics: TrackFitStatistics
    fit_source_ids: tuple[str, ...]
    f4_view_count: int
    selection_receipt: HeldoutReceipt
    second_best_quality: float
    near_square: bool


def build_and_select_geometry(
    fitting_views: Sequence[TrackEvidenceView],
    selection_view: TrackEvidenceView,
) -> FrozenTrackGeometry:
    ordered = sorted(fitting_views, key=lambda row: (row.frame_ordinal, row.source_id))
    if len({view.frame_id for view in ordered}) < 3:
        raise ValueError("fitting requires three distinct past views")
    if selection_view.frame_ordinal <= max(view.frame_ordinal for view in ordered):
        raise ValueError("selection view must be later than every fitting view")
    center, extent, rotation, fused_covariance, statistics, reference_source = _fuse_hb(ordered)
    points = np.concatenate([view.points_world for view in ordered], axis=0)
    if len(points) > MAX_POOLED_POINTS:
        indices = np.linspace(0, len(points) - 1, MAX_POOLED_POINTS, dtype=np.int64)
        points = points[indices]
    fused = _corners(center, extent, rotation)
    perpendicular = _corners(center, extent, _rz(0.5 * np.pi) @ rotation)
    pca_rotation = _rz(_pca_yaw(points))
    _, _, pca = _fit_points(points, pca_rotation)
    hypotheses = {"H_FUSED": fused, "H_PERP": perpendicular, "H_PCA": pca}
    priors = {"H_FUSED": 1.0, "H_PERP": 0.95, "H_PCA": 0.90}
    receipts = {name: evaluate_heldout(corners, selection_view) for name, corners in hypotheses.items()}
    training_inside = {
        name: float(np.mean(points_inside_obb(points, corners, scale=1.05)))
        for name, corners in hypotheses.items()
    }
    qualities = {
        name: _geometric_mean((receipt.quality, training_inside[name], priors[name]))
        for name, receipt in receipts.items()
    }
    ranked = sorted(qualities, key=lambda name: (qualities[name], priors[name], name), reverse=True)
    chosen_name = ranked[0]
    chosen = hypotheses[chosen_name]
    chosen_receipt = replace(receipts[chosen_name], quality=qualities[chosen_name])
    hb_rows = [view for view in ordered if view.has_hb]
    hb_confidence = float(np.mean([view.hb_confidence for view in hb_rows]))
    mask_confidence = float(np.mean([view.mask_confidence for view in ordered]))
    inside = training_inside[chosen_name]
    preliminary = _geometric_mean(
        (
            mask_confidence,
            hb_confidence,
            max(inside, 0.01),
            max(chosen_receipt.quality, 0.01),
            math.exp(-statistics.center_rms_m / 0.25),
            math.exp(-statistics.log_size_mad_max / math.log(1.5)),
        )
    )
    geometry = TrackGeometry(
        source_ids=tuple(view.source_id for view in ordered) + (selection_view.source_id,),
        frame_ids=tuple(view.frame_id for view in ordered) + (selection_view.frame_id,),
        decision_frame_id=selection_view.frame_id,
        decision_frame_ordinal=selection_view.frame_ordinal,
        selected_source_ids=tuple(view.source_id for view in ordered),
        hb_source_id=reference_source,
        hypotheses={name: _readonly(value, np.float64, (8, 3)) for name, value in hypotheses.items()},
        hypothesis_quality=qualities,
        chosen_hypothesis=chosen_name,
        corners=_readonly(chosen, np.float64, (8, 3)),
        refined_points=_readonly(points, np.float64),
        distinct_view_count=len({view.frame_id for view in ordered}) + 1,
        set_cover_fraction=1.0,
        median_pairwise_hb_iou=statistics.median_pairwise_hb_iou,
        median_pairwise_hb_containment=statistics.median_pairwise_hb_containment,
        hb_center_rms_m=statistics.center_rms_m,
        point_inside_hb_fraction=inside,
        pmr_seed_fraction=1.0,
        pmr_retained_fraction=1.0,
        mask_confidence_mean=mask_confidence,
        hb_confidence_mean=hb_confidence,
        preliminary_score=preliminary,
    )
    dimensions = extent[:2]
    near_square = 1.0 - abs(float(dimensions[0] - dimensions[1])) / max(float(np.max(dimensions)), 1.0e-3) >= 0.80
    return FrozenTrackGeometry(
        geometry=geometry,
        covariance_7d=_readonly(_safe_spd(fused_covariance), np.float64, (7, 7)),
        fit_statistics=statistics,
        fit_source_ids=tuple(view.source_id for view in ordered),
        f4_view_count=len(hb_rows),
        selection_receipt=chosen_receipt,
        second_best_quality=qualities[ranked[1]],
        near_square=near_square,
    )


@dataclass(frozen=True)
class AcceptanceConfig:
    min_total_views: int = 5
    min_f4_views: int = 2
    min_view_ray_angle_deg: float = 12.0
    min_camera_baseline_m: float = 0.20
    max_center_rms_m: float = 0.15
    max_log_size_mad: float = math.log(1.30)
    max_yaw_mad_deg: float = 25.0
    max_normalized_center_std: float = 0.25
    max_center_std_m: float = 0.18
    max_log_size_std: float = 0.30
    min_mask_box_iou: float = 0.15
    min_mask_containment: float = 0.60
    min_point_inside: float = 0.70
    min_depth_support: float = 0.55
    max_free_space: float = 0.10
    min_quality: float = 0.55
    min_hypothesis_margin: float = 0.03


@dataclass(frozen=True)
class TrackFusionResult:
    geometry: TrackGeometry
    covariance_7d: np.ndarray
    chosen_hypothesis: str
    fit_source_ids: tuple[str, ...]
    selection_receipt: HeldoutReceipt
    acceptance_receipt: HeldoutReceipt
    absolute_pass: bool
    evidence_score: float
    reasons: tuple[str, ...]


def accept_frozen_geometry(
    frozen: FrozenTrackGeometry,
    acceptance_view: TrackEvidenceView,
    *,
    total_distinct_views: int,
    config: AcceptanceConfig | None = None,
) -> TrackFusionResult:
    cfg = config or AcceptanceConfig()
    if acceptance_view.frame_ordinal <= frozen.geometry.decision_frame_ordinal:
        raise ValueError("acceptance view must be later than the selection view")
    receipt = evaluate_heldout(frozen.geometry.corners, acceptance_view)
    stats = frozen.fit_statistics
    checks = {
        "total_views": total_distinct_views >= cfg.min_total_views,
        "f4_views": frozen.f4_view_count >= cfg.min_f4_views,
        "view_ray_angle": stats.view_ray_angle_deg >= cfg.min_view_ray_angle_deg,
        "camera_baseline": stats.camera_baseline_m >= cfg.min_camera_baseline_m,
        "center_rms": stats.center_rms_m <= cfg.max_center_rms_m,
        "log_size_mad": stats.log_size_mad_max <= cfg.max_log_size_mad,
        "yaw_mad": frozen.near_square or stats.yaw_mad_deg <= cfg.max_yaw_mad_deg,
        "normalized_center_std": stats.max_normalized_center_std <= cfg.max_normalized_center_std,
        "center_std": stats.max_center_std_m <= cfg.max_center_std_m,
        "log_size_std": stats.max_log_size_std <= cfg.max_log_size_std,
        "selection_iou": frozen.selection_receipt.mask_box_iou >= cfg.min_mask_box_iou,
        "selection_containment": frozen.selection_receipt.mask_containment >= cfg.min_mask_containment,
        "selection_inside": frozen.selection_receipt.point_inside_fraction >= cfg.min_point_inside,
        "selection_depth": frozen.selection_receipt.depth_support_fraction >= cfg.min_depth_support,
        "selection_free": frozen.selection_receipt.free_space_fraction <= cfg.max_free_space,
        "selection_quality": frozen.selection_receipt.quality >= cfg.min_quality,
        "hypothesis_margin": frozen.near_square
        or frozen.selection_receipt.quality - frozen.second_best_quality >= cfg.min_hypothesis_margin,
        "acceptance_iou": receipt.mask_box_iou >= cfg.min_mask_box_iou,
        "acceptance_containment": receipt.mask_containment >= cfg.min_mask_containment,
        "acceptance_inside": receipt.point_inside_fraction >= cfg.min_point_inside,
        "acceptance_depth": receipt.depth_support_fraction >= cfg.min_depth_support,
        "acceptance_free": receipt.free_space_fraction <= cfg.max_free_space,
        "acceptance_quality": receipt.quality >= cfg.min_quality,
    }
    reasons = tuple(name for name, passed in checks.items() if not passed)
    accepted_receipt = replace(receipt, passed=not reasons, reasons=reasons)
    selection_receipt = replace(
        frozen.selection_receipt,
        passed=all(value for name, value in checks.items() if name.startswith("selection_") or name == "hypothesis_margin"),
        reasons=tuple(name for name in reasons if name.startswith("selection_") or name == "hypothesis_margin"),
    )
    evidence_score = _geometric_mean(
        (
            frozen.geometry.preliminary_score,
            selection_receipt.quality,
            accepted_receipt.quality,
            math.exp(-stats.center_rms_m / max(cfg.max_center_rms_m, _EPS)),
        )
    )
    geometry = replace(
        frozen.geometry,
        decision_frame_id=acceptance_view.frame_id,
        decision_frame_ordinal=acceptance_view.frame_ordinal,
        preliminary_score=evidence_score,
    )
    return TrackFusionResult(
        geometry=geometry,
        covariance_7d=frozen.covariance_7d,
        chosen_hypothesis=geometry.chosen_hypothesis,
        fit_source_ids=frozen.fit_source_ids,
        selection_receipt=selection_receipt,
        acceptance_receipt=accepted_receipt,
        absolute_pass=not reasons,
        evidence_score=evidence_score,
        reasons=reasons,
    )


__all__ = [
    "AcceptanceConfig",
    "FrozenTrackGeometry",
    "HeldoutReceipt",
    "TrackEvidenceView",
    "TrackFitStatistics",
    "TrackFusionResult",
    "accept_frozen_geometry",
    "attach_boxer_observation",
    "build_and_select_geometry",
    "corners_to_params",
    "evaluate_heldout",
    "pack_mask",
]
