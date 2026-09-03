"""Pure NumPy geometry for a training-free MV3DIS-lite depth guide.

The helpers in this module are deliberately stateless.  They do not import
Torch, CLIP, QIM, PUF, or BoxFusion association code, and they expose no
mutation API.  A current RGB-D proposal can be reduced to at most 64 world
points, then a historical point guide can be projected into a later frame to
measure the visibility and depth-consistency terms from MV3DIS.

Coordinate convention
---------------------
``T_wc`` maps depth-camera coordinates to world coordinates.  Depth is metric
camera-z (not Euclidean ray length), and ``K`` is a conventional pinhole
intrinsic matrix.  Pixel lookup is deterministic nearest-neighbour with
half-pixel ties rounded upward: ``floor(value + 0.5)``.  A geometrically valid
projection is first required to satisfy ``0 <= u < W`` and ``0 <= v < H``;
the resulting nearest index is then clipped to the closest image-edge pixel.

The proposal sampler uses the true intersection between the raw 2D box and
the convex projection of the 3D OBB.  It places one sample at each cell centre
of a fixed 8 x 8 grid over that intersection's bounding rectangle, retains
only centres inside the intersection polygon, rounds and de-duplicates pixel
indices, back-projects valid 0.1--8 metre depth, and finally retains only
points inside the *oriented* box.  Fewer than 16 surviving points fail closed.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations, product
from numbers import Real
from typing import Optional

import numpy as np


GRID_SIZE = 8
MIN_GUIDE_POINTS = 16
MAX_GUIDE_POINTS = 64
MAX_BATCH_PROPOSALS = 256
MIN_DEPTH_M = 0.10
MAX_DEPTH_M = 8.0
DEPTH_ALPHA = 0.05

# Frozen to the existing BoxFusion/TR3D depth-projection convention.
_NEAR_CLIP_M = 1e-3
_GEOMETRY_TOLERANCE = 1e-6
_OBB_NORMAL_TOLERANCE = 5e-6
_CORNER_TRIPLES = np.asarray(list(combinations(range(8), 3)), dtype=np.int64)
_CORNER_SIGNS = np.asarray(
    list(product((-1.0, 1.0), repeat=3)),
    dtype=np.float64,
)
_GRID_FRACTIONS = np.column_stack(
    (
        np.tile(
            (np.arange(GRID_SIZE, dtype=np.float64) + 0.5) / GRID_SIZE,
            GRID_SIZE,
        ),
        np.repeat(
            (np.arange(GRID_SIZE, dtype=np.float64) + 0.5) / GRID_SIZE,
            GRID_SIZE,
        ),
    )
)


def _readonly(value: object, dtype: np.dtype) -> np.ndarray:
    result = np.array(value, dtype=dtype, order="C", copy=True)
    result.setflags(write=False)
    return result


def _readonly_owned(value: object, dtype: np.dtype) -> np.ndarray:
    """Freeze an array allocated wholly inside this module without re-copying."""

    result = np.ascontiguousarray(value, dtype=dtype)
    result.setflags(write=False)
    return result


def _numeric_array(value: object, name: str) -> np.ndarray:
    raw = np.asarray(value)
    if raw.dtype.kind not in "iuf":
        raise ValueError(f"{name} must be numeric")
    return raw


def _depth_image(value: object) -> np.ndarray:
    raw = _numeric_array(value, "depth_m")
    if raw.ndim != 2 or raw.shape[0] < 1 or raw.shape[1] < 1:
        raise ValueError("depth_m must be a non-empty numeric [H,W] image")
    # Invalid/non-finite sensor samples are legal and are filtered per pixel.
    # Keep the sensor dtype and avoid copying the full frame once per proposal;
    # all arithmetic below is read-only and sampled values promote to float64.
    return np.ascontiguousarray(raw)


def _intrinsics(value: object, image_shape: tuple[int, int]) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64)
    if not np.isfinite(matrix).all():
        raise ValueError("K must be finite [3,3] or [4,4]")
    if matrix.shape == (4, 4):
        matrix = matrix[:3, :3]
    if matrix.shape != (3, 3):
        raise ValueError("K must be finite [3,3] or [4,4]")
    if matrix[0, 0] <= 0.0 or matrix[1, 1] <= 0.0:
        raise ValueError("K focal lengths must be positive")
    if np.max(np.abs(matrix[2] - [0.0, 0.0, 1.0])) > 1e-10:
        raise ValueError("K must use the pinhole last row [0,0,1]")
    height, width = image_shape
    if not (0.0 <= matrix[0, 2] < width and 0.0 <= matrix[1, 2] < height):
        raise ValueError("K principal point must lie in the depth image")
    determinant = float(np.linalg.det(matrix))
    if not np.isfinite(determinant) or abs(determinant) <= 1e-12:
        raise ValueError("K must be invertible")
    return np.ascontiguousarray(matrix)


def _rigid_transform(value: object) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError("T_wc must be a finite [4,4] transform")
    if np.max(np.abs(matrix[3] - [0.0, 0.0, 0.0, 1.0])) > 1e-8:
        raise ValueError("T_wc must be homogeneous")
    rotation = matrix[:3, :3]
    if (
        np.max(np.abs(rotation.T @ rotation - np.eye(3))) > 1e-5
        or abs(float(np.linalg.det(rotation)) - 1.0) > 1e-5
    ):
        raise ValueError("T_wc must contain a proper rigid rotation")
    return np.ascontiguousarray(matrix)


def _box_xyxy(value: object, name: str) -> np.ndarray:
    box = np.asarray(value, dtype=np.float64)
    if box.shape != (4,) or not np.isfinite(box).all():
        raise ValueError(f"{name} must be a finite [4] xyxy box")
    if box[2] <= box[0] or box[3] <= box[1]:
        raise ValueError(f"{name} must have x2>x1 and y2>y1")
    return np.ascontiguousarray(box)


def _points_world(value: object) -> np.ndarray:
    points = np.asarray(value, dtype=np.float64)
    if points.ndim != 2 or points.shape[1:] != (3,):
        raise ValueError("points_world must have shape [N,3]")
    if not MIN_GUIDE_POINTS <= len(points) <= MAX_GUIDE_POINTS:
        raise ValueError(
            f"points_world must contain {MIN_GUIDE_POINTS}..{MAX_GUIDE_POINTS} points"
        )
    if not np.isfinite(points).all():
        raise ValueError("points_world must contain only finite values")
    return np.ascontiguousarray(points)


def _obb_corners(value: object) -> np.ndarray:
    corners = np.asarray(value, dtype=np.float64)
    if corners.shape != (8, 3) or not np.isfinite(corners).all():
        raise ValueError("obb_corners_world must be finite [8,3]")
    pairwise = np.max(
        np.abs(corners[:, None, :] - corners[None, :, :]), axis=2
    )
    pairwise[np.diag_indices(8)] = np.inf
    if np.any(pairwise == 0.0):
        raise ValueError("obb_corners_world must contain eight unique corners")
    return np.ascontiguousarray(corners)


def _finite_positive(name: str, value: object) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite positive real number")
    result = float(value)
    if not np.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be a finite positive real number")
    return result


def _cross_2d(origin: np.ndarray, left: np.ndarray, right: np.ndarray) -> float:
    return float(
        (left[0] - origin[0]) * (right[1] - origin[1])
        - (left[1] - origin[1]) * (right[0] - origin[0])
    )


def _convex_hull(points: np.ndarray) -> np.ndarray:
    values = np.asarray(points, dtype=np.float64)
    ordered = values[np.lexsort((values[:, 1], values[:, 0]))]
    unique_rows: list[np.ndarray] = []
    for point in ordered:
        if (
            not unique_rows
            or np.max(np.abs(point - unique_rows[-1])) > _GEOMETRY_TOLERANCE
        ):
            unique_rows.append(point)
    if len(unique_rows) < 3:
        return np.empty((0, 2), dtype=np.float64)
    ordered = np.asarray(unique_rows, dtype=np.float64)
    lower: list[np.ndarray] = []
    for point in ordered:
        while len(lower) >= 2 and _cross_2d(lower[-2], lower[-1], point) <= 0.0:
            lower.pop()
        lower.append(point)
    upper: list[np.ndarray] = []
    for point in ordered[::-1]:
        while len(upper) >= 2 and _cross_2d(upper[-2], upper[-1], point) <= 0.0:
            upper.pop()
        upper.append(point)
    return np.asarray(lower[:-1] + upper[:-1], dtype=np.float64)


def _clip_polygon_axis(
    polygon: np.ndarray, axis: int, bound: float, keep_greater: bool
) -> np.ndarray:
    if len(polygon) == 0:
        return polygon

    def inside(point: np.ndarray) -> bool:
        if keep_greater:
            return bool(point[axis] >= bound)
        return bool(point[axis] <= bound)

    output: list[np.ndarray] = []
    start = polygon[-1]
    start_inside = inside(start)
    for end in polygon:
        end_inside = inside(end)
        if start_inside != end_inside:
            denominator = float(end[axis] - start[axis])
            if abs(denominator) <= 1e-15:
                raise ValueError("degenerate polygon clipping edge")
            amount = float((bound - start[axis]) / denominator)
            intersection = start + amount * (end - start)
            intersection[axis] = bound
            output.append(intersection)
        if end_inside:
            output.append(end.copy())
        start = end
        start_inside = end_inside
    return np.asarray(output, dtype=np.float64).reshape(-1, 2)


def _clip_polygon_to_rectangle(
    polygon: np.ndarray, x1: float, y1: float, x2: float, y2: float
) -> np.ndarray:
    result = polygon
    for axis, bound, keep_greater in (
        (0, x1, True),
        (0, x2, False),
        (1, y1, True),
        (1, y2, False),
    ):
        result = _clip_polygon_axis(result, axis, bound, keep_greater)
        if len(result) < 3:
            return np.empty((0, 2), dtype=np.float64)
    # Clipping exactly through an existing vertex can emit that vertex twice.
    # Canonicalize adjacent duplicates while preserving polygon traversal.
    cleaned: list[np.ndarray] = []
    for point in result:
        if not cleaned or np.max(np.abs(point - cleaned[-1])) > _GEOMETRY_TOLERANCE:
            cleaned.append(point.copy())
    if (
        len(cleaned) > 1
        and np.max(np.abs(cleaned[0] - cleaned[-1])) <= _GEOMETRY_TOLERANCE
    ):
        cleaned.pop()
    if len(cleaned) < 3:
        return np.empty((0, 2), dtype=np.float64)
    return np.asarray(cleaned, dtype=np.float64)


def _polygon_area_signed(polygon: np.ndarray) -> float:
    x, y = polygon[:, 0], polygon[:, 1]
    return float(
        0.5
        * (
            np.dot(x[:-1], y[1:])
            + x[-1] * y[0]
            - np.dot(y[:-1], x[1:])
            - y[-1] * x[0]
        )
    )


def _inside_convex_polygon(points: np.ndarray, polygon: np.ndarray) -> np.ndarray:
    if len(polygon) < 3:
        return np.zeros(len(points), dtype=bool)
    orientation = 1.0 if _polygon_area_signed(polygon) >= 0.0 else -1.0
    edge_start = polygon
    edge_end = np.concatenate((polygon[1:], polygon[:1]), axis=0)
    edge = edge_end - edge_start
    relative = points[:, None, :] - edge_start[None, :, :]
    cross = edge[None, :, 0] * relative[:, :, 1] - edge[None, :, 1] * relative[:, :, 0]
    return np.all(orientation * cross >= -_GEOMETRY_TOLERANCE, axis=1)


def _obb_halfspaces(corners: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Recover the six true OBB face half-spaces without corner ordering."""

    scale = float(np.max(np.ptp(corners, axis=0)))
    if not np.isfinite(scale) or scale <= _GEOMETRY_TOLERANCE:
        raise ValueError("obb_corners_world is degenerate")
    tolerance = max(_GEOMETRY_TOLERANCE, scale * _GEOMETRY_TOLERANCE)
    center = corners.mean(axis=0)
    # Work in center-relative coordinates.  Comparing world-plane offsets
    # directly is numerically unstable for float32 corners: a tiny difference
    # between two fitted normals is multiplied by the (potentially large)
    # world translation and can split one physical face into two planes.
    centered = corners - center[None, :]
    triples = centered[_CORNER_TRIPLES]
    first = triples[:, 0]
    normals = np.cross(triples[:, 1] - first, triples[:, 2] - first)
    norms = np.linalg.norm(normals, axis=1)
    nondegenerate = norms > tolerance * tolerance
    normals = normals[nondegenerate]
    first = first[nondegenerate]
    norms = norms[nondegenerate]
    normals /= norms[:, None]
    # Outward orientation makes the box interior n.x + d <= 0.
    inward = np.einsum("ij,ij->i", normals, -first) > 0.0
    normals[inward] *= -1.0
    centered_offsets = -np.einsum("ij,ij->i", normals, first)
    signed = normals @ centered.T + centered_offsets[:, None]
    supporting = (
        (np.max(signed, axis=1) <= tolerance)
        & (np.count_nonzero(np.abs(signed) <= tolerance, axis=1) == 4)
    )
    candidate_normals = normals[supporting]
    candidate_offsets = centered_offsets[supporting]
    if len(candidate_normals) == 0:
        raise ValueError("obb_corners_world does not define a non-degenerate OBB")
    normal_close = np.max(
        np.abs(candidate_normals[:, None, :] - candidate_normals[None, :, :]),
        axis=2,
    ) <= _OBB_NORMAL_TOLERANCE
    offset_close = (
        np.abs(candidate_offsets[:, None] - candidate_offsets[None, :])
        <= tolerance
    )
    equivalent = normal_close & offset_close
    covered = np.zeros(len(candidate_normals), dtype=bool)
    unique_indices: list[int] = []
    for index in range(len(candidate_normals)):
        if not covered[index]:
            unique_indices.append(index)
            covered |= equivalent[index]
    normals = candidate_normals[unique_indices]
    centered_offsets = candidate_offsets[unique_indices]
    if len(normals) != 6:
        raise ValueError("obb_corners_world does not define a non-degenerate OBB")
    # A rectangular OBB has three mutually orthogonal axes and two opposing
    # supporting planes per axis.  Validate this rather than accepting an
    # arbitrary convex hexahedron as an OBB.
    unused = set(range(6))
    axes: list[np.ndarray] = []
    while unused:
        index = min(unused)
        unused.remove(index)
        matches = [
            other
            for other in sorted(unused)
            if abs(float(np.dot(normals[index], normals[other])) + 1.0) <= 1e-5
        ]
        if len(matches) != 1:
            raise ValueError("obb_corners_world faces must form opposite pairs")
        unused.remove(matches[0])
        axes.append(normals[index])
    axis_matrix = np.stack(axes)
    if np.max(
        np.abs(np.abs(axis_matrix @ axis_matrix.T) - np.eye(3))
    ) > 1e-5:
        raise ValueError("obb_corners_world axes must be orthogonal")
    offsets = centered_offsets - normals @ center
    return normals, offsets, tolerance


def _points_inside_halfspaces(
    points: np.ndarray,
    normals: np.ndarray,
    offsets: np.ndarray,
    tolerance: float,
) -> np.ndarray:
    signed = points @ normals.T + offsets[None, :]
    return np.all(signed <= tolerance, axis=1)


def _world_to_camera(points: np.ndarray, T_wc: np.ndarray) -> np.ndarray:
    rotation = T_wc[:3, :3]
    translation = T_wc[:3, 3]
    # For a rigid transform, inverse rotation is its transpose.  With row
    # vectors this is (world - t) @ R.
    result = (points - translation[None, :]) @ rotation
    if not np.isfinite(result).all():
        raise ValueError("world-to-camera projection produced non-finite values")
    return result


def _project_camera(points_camera: np.ndarray, K: np.ndarray) -> np.ndarray:
    homogeneous = points_camera @ K.T
    pixels = np.full((len(points_camera), 2), np.nan, dtype=np.float64)
    positive = points_camera[:, 2] > _NEAR_CLIP_M
    pixels[positive] = (
        homogeneous[positive, :2] / homogeneous[positive, 2:3]
    )
    return pixels


def _nearest_pixel_indices(
    pixels: np.ndarray, height: int, width: int
) -> np.ndarray:
    # Callers mask non-finite/out-of-image projections before indexing.  Zero
    # is used here only as a safe placeholder for those rows.
    safe = np.where(np.isfinite(pixels), pixels, 0.0)
    indices = np.floor(safe + 0.5).astype(np.int64)
    indices[:, 0] = np.clip(indices[:, 0], 0, width - 1)
    indices[:, 1] = np.clip(indices[:, 1], 0, height - 1)
    return indices


@dataclass(frozen=True)
class DepthGuideSample:
    """A deterministic current-frame proposal guide in world coordinates."""

    pixels_xy: np.ndarray
    points_camera: np.ndarray
    points_world: np.ndarray
    intersection_polygon_xy: np.ndarray
    sampled_cell_count: int
    unique_pixel_count: int
    valid_depth_count: int


@dataclass(frozen=True)
class DepthGuideMetrics:
    """Frozen MV3DIS-lite forward and proposal-conditioned measurements."""

    pixels_xy: np.ndarray
    pixel_indices_xy: np.ndarray
    projected_depth_m: np.ndarray
    measured_depth_m: np.ndarray
    valid_depth: np.ndarray
    i_vis: np.ndarray
    w_d: np.ndarray
    v_f: float
    d_f: float
    q_f: float
    inside_proposal: Optional[np.ndarray]
    v_b: Optional[float]
    d_b: Optional[float]
    affinity_a: Optional[float]


@dataclass(frozen=True)
class _PreparedDepthFrame:
    depth_m: np.ndarray
    K: np.ndarray
    K_inv: np.ndarray
    T_wc: np.ndarray
    height: int
    width: int


def _prepare_depth_frame(
    depth_m: object, K: object, T_wc: object
) -> _PreparedDepthFrame:
    depth = _depth_image(depth_m)
    height, width = depth.shape
    intrinsic = _intrinsics(K, (height, width))
    camera_to_world = _rigid_transform(T_wc)
    inverse_intrinsic = np.linalg.inv(intrinsic)
    if not np.isfinite(inverse_intrinsic).all():
        raise ValueError("K inverse must be finite")
    return _PreparedDepthFrame(
        depth_m=depth,
        K=intrinsic,
        K_inv=np.ascontiguousarray(inverse_intrinsic),
        T_wc=camera_to_world,
        height=height,
        width=width,
    )


def _finite_batch(
    value: object, name: str, shape_tail: tuple[int, ...]
) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != len(shape_tail) + 1 or array.shape[1:] != shape_tail:
        suffix = ",".join(str(item) for item in shape_tail)
        raise ValueError(f"{name} must have shape [N,{suffix}]")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    return np.ascontiguousarray(array)


def _validate_structured_obb_batch(
    raw_boxes_xyxy: object,
    obb_centers_world: object,
    obb_dimensions: object,
    obb_rotations_world: object,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    boxes = _finite_batch(raw_boxes_xyxy, "raw_boxes_xyxy", (4,))
    centers = _finite_batch(obb_centers_world, "obb_centers_world", (3,))
    dimensions = _finite_batch(obb_dimensions, "obb_dimensions", (3,))
    rotations = _finite_batch(
        obb_rotations_world, "obb_rotations_world", (3, 3)
    )
    count = len(boxes)
    if count > MAX_BATCH_PROPOSALS:
        raise ValueError(
            f"structured OBB batch must not exceed {MAX_BATCH_PROPOSALS} proposals"
        )
    if len(centers) != count or len(dimensions) != count or len(rotations) != count:
        raise ValueError("structured OBB batch inputs must have the same length")
    if np.any(boxes[:, 2] <= boxes[:, 0]) or np.any(
        boxes[:, 3] <= boxes[:, 1]
    ):
        raise ValueError("raw_boxes_xyxy rows must have x2>x1 and y2>y1")
    if np.any(dimensions <= 0.0):
        raise ValueError("obb_dimensions must be positive")
    if count:
        gram = np.einsum("pji,pjk->pik", rotations, rotations)
        orthogonal_error = np.max(
            np.abs(gram - np.eye(3, dtype=np.float64)[None, :, :]),
            axis=(1, 2),
        )
        determinants = np.linalg.det(rotations)
        if np.any(orthogonal_error > 1e-5) or np.any(
            np.abs(determinants - 1.0) > 1e-5
        ):
            raise ValueError(
                "obb_rotations_world must contain proper rigid rotations"
            )
    return boxes, centers, dimensions, rotations


def _points_inside_structured_obb(
    points_world: np.ndarray,
    center_world: np.ndarray,
    half_dimensions: np.ndarray,
    rotation_world: np.ndarray,
    tolerance: float,
) -> np.ndarray:
    # GeneralInstance3DBoxes uses p_world = p_local @ R.T + center.
    local = (points_world - center_world[None, :]) @ rotation_world
    return np.all(np.abs(local) <= half_dimensions[None, :] + tolerance, axis=1)


def _sample_depth_guide_points_prepared(
    frame: _PreparedDepthFrame,
    raw_box: np.ndarray,
    corners_world: np.ndarray,
    *,
    corner_projection: Optional[tuple[np.ndarray, np.ndarray]] = None,
    halfspaces: Optional[tuple[np.ndarray, np.ndarray, float]] = None,
    structured_obb: Optional[
        tuple[np.ndarray, np.ndarray, np.ndarray, float]
    ] = None,
) -> Optional[DepthGuideSample]:
    if (halfspaces is None) == (structured_obb is None):
        raise AssertionError("exactly one OBB membership representation is required")

    depth = frame.depth_m
    height, width = frame.height, frame.width
    intrinsic = frame.K
    camera_to_world = frame.T_wc
    if corner_projection is None:
        corners_camera = _world_to_camera(corners_world, camera_to_world)
        projected = _project_camera(corners_camera, intrinsic)
    else:
        corners_camera, projected = corner_projection
    if np.any(corners_camera[:, 2] <= _NEAR_CLIP_M):
        return None
    if not np.isfinite(projected).all():
        return None
    polygon = _convex_hull(projected)
    if len(polygon) < 3:
        return None
    x1 = max(float(raw_box[0]), 0.0)
    y1 = max(float(raw_box[1]), 0.0)
    x2 = min(float(raw_box[2]), float(width))
    y2 = min(float(raw_box[3]), float(height))
    if x2 <= x1 or y2 <= y1:
        return None
    polygon = _clip_polygon_to_rectangle(polygon, x1, y1, x2, y2)
    if len(polygon) < 3 or abs(_polygon_area_signed(polygon)) <= 1e-9:
        return None

    minimum = polygon.min(axis=0)
    maximum = polygon.max(axis=0)
    if np.any(maximum <= minimum):
        return None
    cell_centers = minimum[None, :] + _GRID_FRACTIONS * (
        maximum - minimum
    )[None, :]
    cell_centers = cell_centers[_inside_convex_polygon(cell_centers, polygon)]
    sampled_cell_count = int(len(cell_centers))
    if sampled_cell_count < MIN_GUIDE_POINTS:
        return None

    pixel_indices = _nearest_pixel_indices(cell_centers, height, width)
    # Restore first-occurrence order after np.unique's lexicographic grouping.
    # This is equivalent to the previous row-major set loop and considerably
    # cheaper in the batched P10 online path.
    if not len(pixel_indices):
        return None
    pixel_keys = pixel_indices[:, 1] * width + pixel_indices[:, 0]
    _, first_indices = np.unique(pixel_keys, return_index=True)
    pixels = pixel_indices[np.sort(first_indices)][:MAX_GUIDE_POINTS]
    unique_pixel_count = int(len(pixels))
    if unique_pixel_count < MIN_GUIDE_POINTS:
        return None

    measured = depth[pixels[:, 1], pixels[:, 0]]
    valid_depth = (
        np.isfinite(measured)
        & (measured >= MIN_DEPTH_M)
        & (measured <= MAX_DEPTH_M)
    )
    valid_depth_count = int(np.count_nonzero(valid_depth))
    if valid_depth_count < MIN_GUIDE_POINTS:
        return None
    pixels = pixels[valid_depth]
    measured = measured[valid_depth]

    homogeneous_pixels = np.column_stack(
        (pixels.astype(np.float64), np.ones(len(pixels), dtype=np.float64))
    )
    rays = homogeneous_pixels @ frame.K_inv.T
    if np.max(np.abs(rays[:, 2] - 1.0)) > 1e-8:
        raise ValueError("K inverse rays must use unit camera-z")
    points_camera = rays * measured[:, None]
    points_world = (
        points_camera @ camera_to_world[:3, :3].T
        + camera_to_world[None, :3, 3]
    )
    if halfspaces is not None:
        obb_normals, obb_offsets, obb_tolerance = halfspaces
        inside = _points_inside_halfspaces(
            points_world, obb_normals, obb_offsets, obb_tolerance
        )
    else:
        assert structured_obb is not None
        center_world, half_dimensions, rotation_world, obb_tolerance = (
            structured_obb
        )
        inside = _points_inside_structured_obb(
            points_world,
            center_world,
            half_dimensions,
            rotation_world,
            obb_tolerance,
        )
    pixels = pixels[inside]
    points_camera = points_camera[inside]
    points_world = points_world[inside]
    if len(points_world) < MIN_GUIDE_POINTS:
        return None
    if len(points_world) > MAX_GUIDE_POINTS:
        pixels = pixels[:MAX_GUIDE_POINTS]
        points_camera = points_camera[:MAX_GUIDE_POINTS]
        points_world = points_world[:MAX_GUIDE_POINTS]

    return DepthGuideSample(
        pixels_xy=_readonly_owned(pixels, np.int64),
        points_camera=_readonly_owned(points_camera, np.float64),
        points_world=_readonly_owned(points_world, np.float64),
        intersection_polygon_xy=_readonly_owned(polygon, np.float64),
        sampled_cell_count=sampled_cell_count,
        unique_pixel_count=unique_pixel_count,
        valid_depth_count=valid_depth_count,
    )


def sample_depth_guide_points(
    depth_m: object,
    K: object,
    T_wc: object,
    raw_box_xyxy: object,
    obb_corners_world: object,
) -> Optional[DepthGuideSample]:
    """Extract one fixed 8x8/min16/max64 proposal depth guide.

    Invalid caller inputs raise ``ValueError``.  A valid but unusable view
    (near-plane crossing, empty polygon, insufficient valid depth, or fewer
    than 16 points inside the oriented box) returns ``None`` fail-closed.

    This corner-order-independent entry point is retained for diagnostics and
    compatibility.  The online proposal path should use
    :func:`sample_depth_guide_points_batch`, which consumes BoxFusion's native
    center/dimensions/rotation representation without fitting planes.
    """

    frame = _prepare_depth_frame(depth_m, K, T_wc)
    raw_box = _box_xyxy(raw_box_xyxy, "raw_box_xyxy")
    corners_world = _obb_corners(obb_corners_world)
    # Validate OBB geometry even if this particular view later fails.
    halfspaces = _obb_halfspaces(corners_world)
    return _sample_depth_guide_points_prepared(
        frame,
        raw_box,
        corners_world,
        halfspaces=halfspaces,
    )


def sample_depth_guide_points_batch(
    depth_m: object,
    K: object,
    T_wc: object,
    raw_boxes_xyxy: object,
    obb_centers_world: object,
    obb_dimensions: object,
    obb_rotations_world: object,
) -> tuple[Optional[DepthGuideSample], ...]:
    """Fast structured-OBB sampler for one online keyframe.

    The four proposal arrays align on their first dimension and may contain at
    most 256 rows.  ``obb_dimensions`` contains full positive side lengths;
    ``obb_rotations_world`` contains the local-to-world rotation matrices used
    by :class:`GeneralInstance3DBoxes`.  Frame validation and ``K`` inversion
    occur once per batch.  Invalid caller arrays raise ``ValueError`` before
    any row is sampled; geometrically unusable rows return ``None``.
    """

    frame = _prepare_depth_frame(depth_m, K, T_wc)
    boxes, centers, dimensions, rotations = _validate_structured_obb_batch(
        raw_boxes_xyxy,
        obb_centers_world,
        obb_dimensions,
        obb_rotations_world,
    )
    if not len(boxes):
        return ()
    half_dimensions = 0.5 * dimensions
    corners_world = np.einsum(
        "pni,pji->pnj",
        _CORNER_SIGNS[None, :, :] * half_dimensions[:, None, :],
        rotations,
    ) + centers[:, None, :]
    corners_camera = np.einsum(
        "pni,ij->pnj",
        corners_world - frame.T_wc[None, None, :3, 3],
        frame.T_wc[:3, :3],
    )
    if not np.isfinite(corners_camera).all():
        raise ValueError("world-to-camera batch projection produced non-finite values")
    homogeneous = np.einsum("pni,ji->pnj", corners_camera, frame.K)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        projected = homogeneous[:, :, :2] / homogeneous[:, :, 2:3]
    results = []
    for index in range(len(boxes)):
        tolerance = max(
            _GEOMETRY_TOLERANCE,
            float(np.max(dimensions[index])) * _GEOMETRY_TOLERANCE,
        )
        results.append(
            _sample_depth_guide_points_prepared(
                frame,
                boxes[index],
                corners_world[index],
                corner_projection=(
                    corners_camera[index],
                    projected[index],
                ),
                structured_obb=(
                    centers[index],
                    half_dimensions[index],
                    rotations[index],
                    tolerance,
                ),
            )
        )
    return tuple(results)


def project_guide_metrics(
    points_world: object,
    depth_m: object,
    K: object,
    T_wc: object,
    proposal_box_xyxy: object | None = None,
    alpha: float = DEPTH_ALPHA,
) -> DepthGuideMetrics:
    """Project a historical guide and compute MV3DIS-lite metrics.

    For guide ``G`` and MV3DIS visibility/weight arrays ``Ivis`` and ``wd``::

        Vf = sum(Ivis) / |G|
        Df = sum(Ivis * wd) / max(sum(Ivis), 1)
        Qf = Vf * Df

    If a current raw proposal box ``bp`` is supplied, the backward branch is::

        Vb = sum(Ivis * inside_bp) / max(sum(Ivis), 1)
        Db = sum(Ivis * inside_bp * wd) / max(sum(Ivis * inside_bp), 1)
        a  = Vb * Db

    ``Ivis`` uses the paper's strict relative-depth test
    ``abs(z-d) < alpha*d``.  Invalid depth and off-image projections receive
    zero visibility and zero weight.
    """

    guide = _points_world(points_world)
    depth = _depth_image(depth_m)
    height, width = depth.shape
    intrinsic = _intrinsics(K, (height, width))
    camera_to_world = _rigid_transform(T_wc)
    relative_tolerance = _finite_positive("alpha", alpha)
    proposal = (
        None
        if proposal_box_xyxy is None
        else _box_xyxy(proposal_box_xyxy, "proposal_box_xyxy")
    )

    points_camera = _world_to_camera(guide, camera_to_world)
    projected_depth = points_camera[:, 2]
    pixels = _project_camera(points_camera, intrinsic)
    inside_image = (
        (projected_depth > _NEAR_CLIP_M)
        & np.isfinite(pixels).all(axis=1)
        & (pixels[:, 0] >= 0.0)
        & (pixels[:, 0] < width)
        & (pixels[:, 1] >= 0.0)
        & (pixels[:, 1] < height)
    )
    pixel_indices = _nearest_pixel_indices(pixels, height, width)
    measured = depth[pixel_indices[:, 1], pixel_indices[:, 0]]
    valid_depth = (
        inside_image
        & np.isfinite(measured)
        & (measured >= MIN_DEPTH_M)
        & (measured <= MAX_DEPTH_M)
    )
    error = np.zeros(len(guide), dtype=np.float64)
    error[valid_depth] = np.abs(
        projected_depth[valid_depth] - measured[valid_depth]
    )
    scale = np.zeros(len(guide), dtype=np.float64)
    scale[valid_depth] = relative_tolerance * measured[valid_depth]
    i_vis = valid_depth & (error < scale)
    w_d = np.zeros(len(guide), dtype=np.float64)
    w_d[valid_depth] = np.maximum(
        0.0, 1.0 - error[valid_depth] / scale[valid_depth]
    )

    visible_count = int(np.count_nonzero(i_vis))
    v_f = float(visible_count / len(guide))
    d_f = float(np.sum(w_d[i_vis], dtype=np.float64) / max(visible_count, 1))
    q_f = float(v_f * d_f)

    inside_proposal: Optional[np.ndarray]
    v_b: Optional[float]
    d_b: Optional[float]
    affinity_a: Optional[float]
    if proposal is None:
        inside_proposal = None
        v_b = d_b = affinity_a = None
    else:
        inside_proposal = (
            np.isfinite(pixels).all(axis=1)
            & (pixels[:, 0] >= proposal[0])
            & (pixels[:, 0] <= proposal[2])
            & (pixels[:, 1] >= proposal[1])
            & (pixels[:, 1] <= proposal[3])
        )
        backward = i_vis & inside_proposal
        backward_count = int(np.count_nonzero(backward))
        v_b = float(backward_count / max(visible_count, 1))
        d_b = float(
            np.sum(w_d[backward], dtype=np.float64) / max(backward_count, 1)
        )
        affinity_a = float(v_b * d_b)
        inside_proposal = _readonly(inside_proposal, np.bool_)

    # Keep public scalar metrics numerically within their mathematical range.
    scalars = [v_f, d_f, q_f]
    scalars.extend(value for value in (v_b, d_b, affinity_a) if value is not None)
    if not all(np.isfinite(value) and 0.0 <= value <= 1.0 for value in scalars):
        raise AssertionError("depth-guide metrics escaped [0,1]")

    return DepthGuideMetrics(
        pixels_xy=_readonly(pixels, np.float64),
        pixel_indices_xy=_readonly(pixel_indices, np.int64),
        projected_depth_m=_readonly(projected_depth, np.float64),
        measured_depth_m=_readonly(measured, np.float64),
        valid_depth=_readonly(valid_depth, np.bool_),
        i_vis=_readonly(i_vis, np.bool_),
        w_d=_readonly(w_d, np.float64),
        v_f=v_f,
        d_f=d_f,
        q_f=q_f,
        inside_proposal=inside_proposal,
        v_b=v_b,
        d_b=d_b,
        affinity_a=affinity_a,
    )


__all__ = [
    "DEPTH_ALPHA",
    "GRID_SIZE",
    "MAX_BATCH_PROPOSALS",
    "MAX_DEPTH_M",
    "MAX_GUIDE_POINTS",
    "MIN_DEPTH_M",
    "MIN_GUIDE_POINTS",
    "DepthGuideMetrics",
    "DepthGuideSample",
    "project_guide_metrics",
    "sample_depth_guide_points",
    "sample_depth_guide_points_batch",
]
