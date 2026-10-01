"""Pure NumPy geometry primitives for the TR3D R2a observer.

This module is deliberately independent from CLIP, proposal scoring, and the
active BoxFusion output path.  It projects a world-frame yaw OBB into a depth
camera, intersects sampled camera rays with that OBB, and classifies real
metric depth as supporting, occluding, free-space violating, or invalid.

Coordinate convention
---------------------
``box_world`` is ``[cx, cy, cz, dx, dy, dz, yaw]``.  Positive ``yaw`` rotates
the box-local x axis towards world +y.  Camera rays are parameterized with
camera z equal to one, so their intersection parameters are directly
comparable with z-depth images.  ``depth_camera_to_world`` must therefore be
rigid.  For ScanNet it is constructed as ``pose @ extrinsic_depth``.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral, Real
from typing import Optional, Sequence

import numpy as np


_CORNER_SIGNS = np.asarray(
    [
        [-1.0, -1.0, -1.0],
        [-1.0, -1.0, 1.0],
        [-1.0, 1.0, -1.0],
        [-1.0, 1.0, 1.0],
        [1.0, -1.0, -1.0],
        [1.0, -1.0, 1.0],
        [1.0, 1.0, -1.0],
        [1.0, 1.0, 1.0],
    ],
    dtype=np.float64,
)


def _readonly(value: np.ndarray, dtype: np.dtype) -> np.ndarray:
    result = np.ascontiguousarray(value, dtype=dtype)
    result.setflags(write=False)
    return result


def _finite_real(name: str, value: object, *, minimum: float) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real number")
    result = float(value)
    if not np.isfinite(result) or result < minimum:
        raise ValueError(f"{name} must be finite and at least {minimum}")
    return result


def _positive_int(name: str, value: object) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be a positive integer")
    result = int(value)
    if result < 1:
        raise ValueError(f"{name} must be a positive integer")
    return result


def _image_shape(value: Sequence[int]) -> tuple[int, int]:
    if not isinstance(value, Sequence) or len(value) != 2:
        raise ValueError("image_shape must be (height, width)")
    return (
        _positive_int("image height", value[0]),
        _positive_int("image width", value[1]),
    )


def _box(value: object) -> np.ndarray:
    box = np.asarray(value, dtype=np.float64)
    if box.shape != (7,) or not np.isfinite(box).all():
        raise ValueError("box_world must be a finite [7] yaw OBB")
    if np.any(box[3:6] <= 0.0):
        raise ValueError("box_world dimensions must be positive")
    return np.ascontiguousarray(box)


def _intrinsics(value: object) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape == (4, 4):
        matrix = matrix[:3, :3]
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        raise ValueError("intrinsics must be finite [3,3] or [4,4]")
    if matrix[0, 0] <= 0.0 or matrix[1, 1] <= 0.0:
        raise ValueError("intrinsics focal lengths must be positive")
    determinant = float(np.linalg.det(matrix))
    if not np.isfinite(determinant) or abs(determinant) <= 1e-12:
        raise ValueError("intrinsics must be invertible")
    return np.ascontiguousarray(matrix)


def _rigid_transform(value: object, name: str) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError(f"{name} must be a finite [4,4] transform")
    if not np.allclose(
        matrix[3], [0.0, 0.0, 0.0, 1.0], rtol=0.0, atol=1e-8
    ):
        raise ValueError(f"{name} must be homogeneous")
    rotation = matrix[:3, :3]
    determinant = float(np.linalg.det(rotation))
    if (
        not np.isfinite(determinant)
        or not np.isclose(determinant, 1.0, rtol=0.0, atol=1e-4)
        or not np.allclose(
            rotation.T @ rotation,
            np.eye(3, dtype=np.float64),
            rtol=0.0,
            atol=1e-4,
        )
    ):
        raise ValueError(f"{name} must contain a proper rigid rotation")
    return np.ascontiguousarray(matrix)


def compose_depth_camera_to_world(
    pose: object, extrinsic_depth: object
) -> np.ndarray:
    """Compose ScanNet sensor-to-world pose with depth-to-sensor extrinsic."""

    sensor_to_world = _rigid_transform(pose, "pose")
    depth_to_sensor = _rigid_transform(
        extrinsic_depth, "extrinsic_depth"
    )
    result = sensor_to_world @ depth_to_sensor
    return _readonly(
        _rigid_transform(result, "depth_camera_to_world"), np.float64
    )


def yaw_obb_corners_world(box_world: object) -> np.ndarray:
    """Return the eight world corners of one ``[cxyz,dxyz,yaw]`` OBB."""

    box = _box(box_world)
    local = _CORNER_SIGNS * (0.5 * box[3:6])
    cosine = float(np.cos(box[6]))
    sine = float(np.sin(box[6]))
    rotation = np.asarray(
        [[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    return _readonly(local @ rotation.T + box[None, :3], np.float64)


def _cross_2d(origin: np.ndarray, left: np.ndarray, right: np.ndarray) -> float:
    return float(
        (left[0] - origin[0]) * (right[1] - origin[1])
        - (left[1] - origin[1]) * (right[0] - origin[0])
    )


def _convex_hull(points: np.ndarray) -> np.ndarray:
    unique = np.unique(np.asarray(points, dtype=np.float64), axis=0)
    if len(unique) <= 2:
        return unique
    ordered = unique[np.lexsort((unique[:, 1], unique[:, 0]))]
    lower: list[np.ndarray] = []
    for point in ordered:
        while len(lower) >= 2 and _cross_2d(lower[-2], lower[-1], point) <= 0:
            lower.pop()
        lower.append(point)
    upper: list[np.ndarray] = []
    for point in ordered[::-1]:
        while len(upper) >= 2 and _cross_2d(upper[-2], upper[-1], point) <= 0:
            upper.pop()
        upper.append(point)
    return np.asarray(lower[:-1] + upper[:-1], dtype=np.float64)


def _clip_polygon_axis(
    polygon: np.ndarray, axis: int, bound: float, keep_greater: bool
) -> np.ndarray:
    if len(polygon) == 0:
        return polygon

    def inside(point: np.ndarray) -> bool:
        return bool(
            point[axis] >= bound if keep_greater else point[axis] <= bound
        )

    output: list[np.ndarray] = []
    start = polygon[-1]
    start_inside = inside(start)
    for end in polygon:
        end_inside = inside(end)
        if start_inside != end_inside:
            denominator = float(end[axis] - start[axis])
            if abs(denominator) <= 1e-15:
                raise ValueError("degenerate projected polygon edge")
            amount = float((bound - start[axis]) / denominator)
            intersection = start + amount * (end - start)
            intersection[axis] = bound
            output.append(intersection)
        if end_inside:
            output.append(end.copy())
        start = end
        start_inside = end_inside
    return np.asarray(output, dtype=np.float64).reshape(-1, 2)


def _clipped_polygon(points: np.ndarray, height: int, width: int) -> np.ndarray:
    polygon = _convex_hull(points)
    if len(polygon) < 3:
        return np.empty((0, 2), dtype=np.float64)
    for axis, bound, keep_greater in (
        (0, 0.0, True),
        (0, float(width), False),
        (1, 0.0, True),
        (1, float(height), False),
    ):
        polygon = _clip_polygon_axis(
            polygon, axis, bound, keep_greater
        )
        if len(polygon) < 3:
            return np.empty((0, 2), dtype=np.float64)
    return polygon


def _polygon_area(polygon: np.ndarray) -> float:
    if len(polygon) < 3:
        return 0.0
    x = polygon[:, 0]
    y = polygon[:, 1]
    return float(
        0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
    )


@dataclass(frozen=True)
class OBBDepthProjection:
    """A fully visible yaw OBB projection in one depth camera."""

    corners_world: np.ndarray
    corners_camera: np.ndarray
    pixels: np.ndarray
    bbox_xyxy: np.ndarray
    area_pixels: float
    area_ratio: float


def project_yaw_obb_to_depth(
    box_world: object,
    intrinsics: object,
    depth_camera_to_world: object,
    image_shape: Sequence[int],
    *,
    near_clip: float = 1e-3,
) -> Optional[OBBDepthProjection]:
    """Project an OBB, returning ``None`` for a non-usable depth view.

    A box touching or crossing the camera near plane is rejected rather than
    partially clipped.  This is intentional fail-closed observer behavior.
    """

    box = _box(box_world)
    intrinsic = _intrinsics(intrinsics)
    camera_to_world = _rigid_transform(
        depth_camera_to_world, "depth_camera_to_world"
    )
    height, width = _image_shape(image_shape)
    near = _finite_real("near_clip", near_clip, minimum=0.0)
    if near <= 0.0:
        raise ValueError("near_clip must be positive")

    corners_world = yaw_obb_corners_world(box)
    world_to_camera = np.linalg.inv(camera_to_world)
    corners_camera = (
        corners_world @ world_to_camera[:3, :3].T
        + world_to_camera[None, :3, 3]
    )
    if not np.isfinite(corners_camera).all():
        raise ValueError("world-to-camera projection produced non-finite values")
    if np.any(corners_camera[:, 2] <= near):
        return None

    homogeneous = corners_camera @ intrinsic.T
    pixels = homogeneous[:, :2] / homogeneous[:, 2:3]
    if not np.isfinite(pixels).all():
        raise ValueError("depth projection produced non-finite pixels")
    polygon = _clipped_polygon(pixels, height, width)
    area_pixels = _polygon_area(polygon)
    if area_pixels <= 0.0:
        return None
    minimum = np.maximum(pixels.min(axis=0), [0.0, 0.0])
    maximum = np.minimum(pixels.max(axis=0), [float(width), float(height)])
    if np.any(maximum <= minimum):
        return None
    bbox = np.concatenate((minimum, maximum))
    return OBBDepthProjection(
        corners_world=_readonly(corners_world, np.float64),
        corners_camera=_readonly(corners_camera, np.float64),
        pixels=_readonly(pixels, np.float64),
        bbox_xyxy=_readonly(bbox, np.float64),
        area_pixels=area_pixels,
        area_ratio=float(area_pixels / float(height * width)),
    )


@dataclass(frozen=True)
class RayOBBIntersections:
    """Forward ray intervals; misses carry zero near/far values."""

    t_near: np.ndarray
    t_far: np.ndarray
    intersects: np.ndarray


def intersect_rays_with_yaw_obb(
    ray_origins_world: object,
    ray_directions_world: object,
    box_world: object,
    *,
    parallel_epsilon: float = 1e-12,
) -> RayOBBIntersections:
    """Intersect one or many forward world rays with a world yaw OBB."""

    box = _box(box_world)
    directions = np.asarray(ray_directions_world, dtype=np.float64)
    if directions.ndim == 1:
        directions = directions[None, :]
    if (
        directions.ndim != 2
        or directions.shape[1] != 3
        or not np.isfinite(directions).all()
    ):
        raise ValueError("ray_directions_world must be finite [N,3]")
    if np.any(np.linalg.norm(directions, axis=1) <= 1e-12):
        raise ValueError("ray directions must be non-zero")

    origins = np.asarray(ray_origins_world, dtype=np.float64)
    if origins.shape == (3,):
        origins = np.broadcast_to(origins, directions.shape).copy()
    if origins.shape != directions.shape or not np.isfinite(origins).all():
        raise ValueError("ray_origins_world must be finite [3] or [N,3]")
    epsilon = _finite_real(
        "parallel_epsilon", parallel_epsilon, minimum=0.0
    )
    if epsilon <= 0.0:
        raise ValueError("parallel_epsilon must be positive")

    cosine = float(np.cos(box[6]))
    sine = float(np.sin(box[6]))
    world_to_local = np.asarray(
        [[cosine, sine, 0.0], [-sine, cosine, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    local_origins = (origins - box[None, :3]) @ world_to_local.T
    local_directions = directions @ world_to_local.T
    half = 0.5 * box[3:6]

    count = len(directions)
    near = np.full(count, -np.inf, dtype=np.float64)
    far = np.full(count, np.inf, dtype=np.float64)
    possible = np.ones(count, dtype=bool)
    for axis in range(3):
        origin = local_origins[:, axis]
        direction = local_directions[:, axis]
        parallel = np.abs(direction) <= epsilon
        possible &= ~(
            parallel & ((origin < -half[axis]) | (origin > half[axis]))
        )
        active = ~parallel
        first = np.empty(count, dtype=np.float64)
        second = np.empty(count, dtype=np.float64)
        first.fill(-np.inf)
        second.fill(np.inf)
        first[active] = (-half[axis] - origin[active]) / direction[active]
        second[active] = (half[axis] - origin[active]) / direction[active]
        near = np.maximum(near, np.minimum(first, second))
        far = np.minimum(far, np.maximum(first, second))

    intersects = (
        possible
        & np.isfinite(near)
        & np.isfinite(far)
        & (far >= np.maximum(near, 0.0))
        & (far >= 0.0)
    )
    output_near = np.where(intersects, np.maximum(near, 0.0), 0.0)
    output_far = np.where(intersects, far, 0.0)
    return RayOBBIntersections(
        t_near=_readonly(output_near, np.float64),
        t_far=_readonly(output_far, np.float64),
        intersects=_readonly(intersects, np.bool_),
    )


@dataclass(frozen=True)
class DepthRayClassification:
    """Per-sampled-pixel R2a evidence for one candidate and depth view."""

    projection: OBBDepthProjection
    rows: np.ndarray
    cols: np.ndarray
    support: np.ndarray
    occluded: np.ndarray
    free_space: np.ndarray
    invalid: np.ndarray
    support_points_world: np.ndarray

    @property
    def sample_count(self) -> int:
        return int(self.rows.shape[0])

    @property
    def support_count(self) -> int:
        return int(np.count_nonzero(self.support))

    @property
    def occluded_count(self) -> int:
        return int(np.count_nonzero(self.occluded))

    @property
    def free_space_count(self) -> int:
        return int(np.count_nonzero(self.free_space))

    @property
    def invalid_count(self) -> int:
        return int(np.count_nonzero(self.invalid))

    @property
    def classified_count(self) -> int:
        return self.support_count + self.occluded_count + self.free_space_count

    def _ratio(self, count: int) -> float:
        return float(count / self.classified_count) if self.classified_count else 0.0

    @property
    def support_ratio(self) -> float:
        return self._ratio(self.support_count)

    @property
    def occluded_ratio(self) -> float:
        return self._ratio(self.occluded_count)

    @property
    def free_space_ratio(self) -> float:
        return self._ratio(self.free_space_count)

    @property
    def invalid_ratio(self) -> float:
        return float(self.invalid_count / self.sample_count) if self.sample_count else 0.0


def classify_depth_rays(
    depth_meters: object,
    box_world: object,
    intrinsics: object,
    depth_camera_to_world: object,
    *,
    pixel_stride: int = 4,
    margin: float = 0.05,
    min_depth: float = 0.10,
    max_depth: float = 8.0,
    near_clip: float = 1e-3,
) -> Optional[DepthRayClassification]:
    """Classify sampled projected pixels against a yaw OBB.

    Non-finite, non-positive, or out-of-range depth is classified ``invalid``.
    Rays that fall inside the projected bounding rectangle but miss the OBB
    are also invalid.  All other samples form an exact partition into:

    * ``support``: observed z lies within ``[near-margin, far+margin]``;
    * ``occluded``: observed z is before ``near-margin``;
    * ``free_space``: observed z is beyond ``far+margin``.
    """

    depth = np.asarray(depth_meters)
    if depth.ndim != 2 or min(depth.shape) < 1:
        raise ValueError("depth_meters must have shape [H,W]")
    if not np.issubdtype(depth.dtype, np.number):
        raise ValueError("depth_meters must be numeric")
    depth = depth.astype(np.float64, copy=False)
    stride = _positive_int("pixel_stride", pixel_stride)
    tolerance = _finite_real("margin", margin, minimum=0.0)
    minimum_depth = _finite_real("min_depth", min_depth, minimum=0.0)
    maximum_depth = _finite_real("max_depth", max_depth, minimum=0.0)
    if maximum_depth <= minimum_depth:
        raise ValueError("max_depth must exceed min_depth")

    box = _box(box_world)
    intrinsic = _intrinsics(intrinsics)
    camera_to_world = _rigid_transform(
        depth_camera_to_world, "depth_camera_to_world"
    )
    projection = project_yaw_obb_to_depth(
        box,
        intrinsic,
        camera_to_world,
        depth.shape,
        near_clip=near_clip,
    )
    if projection is None:
        return None

    height, width = depth.shape
    x1, y1, x2, y2 = projection.bbox_xyxy
    row_min = max(0, int(np.ceil(y1)))
    row_max = min(height - 1, int(np.floor(y2)))
    col_min = max(0, int(np.ceil(x1)))
    col_max = min(width - 1, int(np.floor(x2)))
    if row_min <= row_max:
        grid_rows = np.arange(
            row_min, row_max + 1, stride, dtype=np.int64
        )
    else:
        # A valid sub-pixel projection may contain no integer sample.  Use
        # one deterministic nearest centre pixel rather than silently losing
        # the view because of the global stride phase.
        grid_rows = np.asarray(
            [int(np.clip(np.rint((y1 + y2) * 0.5), 0, height - 1))],
            dtype=np.int64,
        )
    if col_min <= col_max:
        grid_cols = np.arange(
            col_min, col_max + 1, stride, dtype=np.int64
        )
    else:
        grid_cols = np.asarray(
            [int(np.clip(np.rint((x1 + x2) * 0.5), 0, width - 1))],
            dtype=np.int64,
        )
    cols, rows = np.meshgrid(grid_cols, grid_rows)
    rows = rows.reshape(-1)
    cols = cols.reshape(-1)

    pixels = np.column_stack(
        (cols.astype(np.float64), rows.astype(np.float64), np.ones(len(rows)))
    )
    rays_camera = pixels @ np.linalg.inv(intrinsic).T
    if (
        not np.isfinite(rays_camera).all()
        or np.any(np.abs(rays_camera[:, 2]) <= 1e-12)
    ):
        raise ValueError("intrinsics produced invalid camera rays")
    rays_camera /= rays_camera[:, 2:3]
    rays_world = rays_camera @ camera_to_world[:3, :3].T
    origins_world = camera_to_world[:3, 3]
    intersections = intersect_rays_with_yaw_obb(
        origins_world, rays_world, box
    )

    observed = depth[rows, cols]
    valid_depth = (
        np.isfinite(observed)
        & (observed >= minimum_depth)
        & (observed <= maximum_depth)
    )
    usable = valid_depth & intersections.intersects
    support = usable & (
        observed >= intersections.t_near - tolerance
    ) & (observed <= intersections.t_far + tolerance)
    occluded = usable & (observed < intersections.t_near - tolerance)
    free_space = usable & (observed > intersections.t_far + tolerance)
    invalid = ~(support | occluded | free_space)
    if not np.all(support | occluded | free_space | invalid):
        raise AssertionError("depth classifications must form a partition")
    support_points = (
        origins_world[None, :]
        + rays_world[support] * observed[support, None]
    )
    if not np.isfinite(support_points).all():
        raise ValueError("support backprojection produced non-finite points")

    return DepthRayClassification(
        projection=projection,
        rows=_readonly(rows, np.int64),
        cols=_readonly(cols, np.int64),
        support=_readonly(support, np.bool_),
        occluded=_readonly(occluded, np.bool_),
        free_space=_readonly(free_space, np.bool_),
        invalid=_readonly(invalid, np.bool_),
        support_points_world=_readonly(support_points, np.float32),
    )


def stable_top_k_view_indices(
    reliability: object,
    top_k: int,
    *,
    frame_ids: Optional[object] = None,
    valid_mask: Optional[object] = None,
) -> np.ndarray:
    """Return deterministic reliability-ranked view indices.

    Reliability is descending, frame id is the first tie-break, and original
    array position is the final stable tie-break.  Invalid views are excluded.
    """

    scores = np.asarray(reliability, dtype=np.float64)
    if scores.ndim != 1 or not np.isfinite(scores).all():
        raise ValueError("reliability must be a finite 1D array")
    count = len(scores)
    keep = _positive_int("top_k", top_k)

    if frame_ids is None:
        frames = np.arange(count, dtype=np.int64)
    else:
        raw_frames = np.asarray(frame_ids)
        if raw_frames.shape != (count,) or not np.issubdtype(
            raw_frames.dtype, np.integer
        ):
            raise ValueError("frame_ids must be an integer [N] array")
        frames = raw_frames.astype(np.int64, copy=False)
        if np.any(frames < 0):
            raise ValueError("frame_ids must be non-negative")

    if valid_mask is None:
        valid = np.ones(count, dtype=bool)
    else:
        raw_valid = np.asarray(valid_mask)
        if raw_valid.shape != (count,) or raw_valid.dtype != np.dtype(np.bool_):
            raise ValueError("valid_mask must be a Boolean [N] array")
        valid = raw_valid
    candidates = np.flatnonzero(valid)
    if not len(candidates):
        return _readonly(np.empty(0, dtype=np.int64), np.int64)
    order = np.lexsort(
        (
            candidates,
            frames[candidates],
            -scores[candidates],
        )
    )
    return _readonly(candidates[order[:keep]], np.int64)


__all__ = [
    "DepthRayClassification",
    "OBBDepthProjection",
    "RayOBBIntersections",
    "classify_depth_rays",
    "compose_depth_camera_to_world",
    "intersect_rays_with_yaw_obb",
    "project_yaw_obb_to_depth",
    "stable_top_k_view_indices",
    "yaw_obb_corners_world",
]
