"""GT-free geometry and budget helpers for the frozen Step 1c audit."""
from __future__ import annotations

from collections import Counter

import numpy as np


def validate_grids(rgb_shape, depth_m, k_rgb, k_depth):
    """Require registered optical frames; load_frame depth is already metres."""
    depth_m = np.asarray(depth_m)
    if depth_m.ndim != 2 or not np.issubdtype(depth_m.dtype, np.floating):
        raise ValueError('Expected HxW floating-point depth in metres from load_frame')
    height, width = rgb_shape[:2]
    dh, dw = depth_m.shape
    if min(height, width, dh, dw) <= 0:
        raise ValueError('Empty RGB/depth grid')
    if not np.allclose(np.diag([dw / width, dh / height, 1.0]) @ k_rgb,
                       k_depth, atol=1e-5, rtol=0):
        raise ValueError('RGB/depth require explicit optical-frame registration')
    return int(height), int(width)


def anchor_world_points(centers, ids, depth_m, pose, k_rgb, k_depth, rgb_shape):
    """Sample a 5x5 patch on the DEPTH grid, then lift the RGB ray."""
    height, width = validate_grids(rgb_shape, depth_m, k_rgb, k_depth)
    dh, dw = depth_m.shape
    k_inv = np.linalg.inv(k_rgb)
    rgb_to_depth = k_depth @ k_inv
    kept, worlds, depths = [], [], []
    stats = Counter({'tested': len(ids), 'no_valid_depth': 0,
                     'out_of_range': 0, 'outside_image': 0})
    for idx in ids:
        u, v = centers[idx]
        if not (np.isfinite([u, v]).all() and 0 <= u < width and 0 <= v < height):
            stats['outside_image'] += 1
            continue
        mapped = rgb_to_depth @ np.array([u, v, 1.0])
        du, dv = mapped[:2] / mapped[2]
        if not (0 <= du < dw and 0 <= dv < dh):
            stats['outside_image'] += 1
            continue
        x, y = int(du), int(dv)
        window = depth_m[max(0, y - 2):min(dh, y + 3), max(0, x - 2):min(dw, x + 3)]
        valid = window[np.isfinite(window) & (window > 0)]
        if not valid.size:
            stats['no_valid_depth'] += 1
            continue
        # load_frame has already converted millimetres to metres. Do not rescale.
        d = float(np.median(valid))
        if not 0.2 < d < 12.0:
            stats['out_of_range'] += 1
            continue
        ray = k_inv @ np.array([u, v, 1.0])
        world = (pose @ np.r_[ray * d, 1.0])[:3]
        if not np.isfinite(world).all():
            raise ValueError('Nonfinite world point')
        kept.append(int(idx))
        worlds.append(world)
        depths.append(d)
    stats['valid_world_points'] = len(kept)
    return (np.asarray(kept, dtype=np.int64), np.asarray(worlds, dtype=float).reshape(-1, 3),
            np.asarray(depths, dtype=float), stats)


def recurrence_counts(worlds, source_frame, frames, world_to_camera, k_rgb,
                      image_shapes, residual_trees, radius=30.0):
    """Each other frame votes at most once, irrespective of anchor density."""
    hits = np.zeros(len(worlds), dtype=np.int64)
    if not len(worlds):
        return hits
    homogeneous = np.column_stack((worlds, np.ones(len(worlds))))
    for frame in frames:
        tree = residual_trees[frame]
        if frame == source_frame or tree is None:
            continue
        cam = (homogeneous @ world_to_camera[frame].T)[:, :3]
        projected = cam @ k_rgb.T
        ids = np.flatnonzero(np.isfinite(projected).all(1) & (cam[:, 2] > 1e-3))
        pixel = projected[ids, :2] / projected[ids, 2:3]
        height, width = image_shapes[frame]
        inside = ((pixel[:, 0] >= 0) & (pixel[:, 0] < width)
                  & (pixel[:, 1] >= 0) & (pixel[:, 1] < height))
        ids, pixel = ids[inside], pixel[inside]
        if len(ids):
            distances, _ = tree.query(pixel, k=1)
            hits[ids] += distances <= radius
    return hits


def top_m(ids, scores, budget):
    """Rank eligible anchors once; every smaller budget is an exact prefix."""
    ids = np.asarray(ids, dtype=np.int64)
    if budget < 0 or not np.isfinite(np.asarray(scores)[ids]).all():
        raise ValueError('Invalid budget or anchor scores')
    return ids[np.argsort(-np.asarray(scores)[ids], kind='mergesort')][:budget]


def add_clusters(clusters, frame, anchor_ids, corners, scores, voxel_size=0.3):
    """Accumulate scene voxels with DISTINCT frames and a GT-free exemplar."""
    for anchor, box, score in zip(anchor_ids, corners, scores):
        key = tuple(np.floor(box.mean(0) / voxel_size).astype(np.int64).tolist())
        state = clusters.setdefault(key, {'frames': set(), 'observations': 0,
                                          'rank': None, 'box': None})
        state['frames'].add(int(frame))
        state['observations'] += 1
        rank = (-float(score), int(frame), int(anchor))
        if state['rank'] is None or rank < state['rank']:
            state['rank'], state['box'] = rank, box.copy()


def update_coverage(support, best_box, missed_ids, corners, gt, frame):
    """Offline only: global GT IDs must not be confused with column indices."""
    from tools.ca1m_prenms_query_core import box_iou3d

    ious = box_iou3d(corners, gt[np.asarray(missed_ids, dtype=np.int64)])
    for column, gt_id in enumerate(missed_ids):
        good = np.flatnonzero(ious[:, column] > .15)
        if len(good):
            support[int(gt_id)].add(int(frame))
            best = int(good[np.argmax(ious[good, column])])
            if ious[best, column] > best_box[int(gt_id)][0]:
                best_box[int(gt_id)] = (float(ious[best, column]), corners[best].copy())
