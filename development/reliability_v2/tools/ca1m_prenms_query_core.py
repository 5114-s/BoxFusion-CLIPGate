"""Pure NumPy primitives for the frozen CA-1M pre-NMS query experiment.

This module has no detector, filesystem, GT, or score-calibration dependency.
PendingMemory must only receive the normal detector stream: query/control
recoveries are counterfactual outputs, not evidence to commit back to memory.
"""

from __future__ import annotations

import copy

import numpy as np


def _matrix(value, tail, name):
    array = np.asarray(value, dtype=np.float64)
    if array.size == 0:
        return np.empty((0, *tail), dtype=np.float64)
    if array.ndim != 1 + len(tail) or array.shape[1:] != tuple(tail):
        raise ValueError(f"{name} must have shape (N, {', '.join(map(str, tail))})")
    return array


def box_iou2d(a, b):
    """Pairwise IoU; invalid/degenerate boxes have zero overlap."""
    a, b = _matrix(a, (4,), "a"), _matrix(b, (4,), "b")
    valid_a = np.isfinite(a).all(1) & (a[:, 2:] > a[:, :2]).all(1)
    valid_b = np.isfinite(b).all(1) & (b[:, 2:] > b[:, :2]).all(1)
    a = np.where(valid_a[:, None], a, 0.0)
    b = np.where(valid_b[:, None], b, 0.0)
    inter = np.prod(np.maximum(0.0, np.minimum(a[:, None, 2:], b[None, :, 2:])
                               - np.maximum(a[:, None, :2], b[None, :, :2])), axis=-1)
    union = np.prod(a[:, 2:] - a[:, :2], axis=1)[:, None]
    union = union + np.prod(b[:, 2:] - b[:, :2], axis=1)[None, :] - inter
    result = np.zeros_like(inter)
    np.divide(inter, union, out=result, where=union > 0.0)
    return result


def box_iou3d(a, b):
    """Pairwise AABB IoU from world-space corners, not oriented-box IoU."""
    a, b = _matrix(a, (8, 3), "a"), _matrix(b, (8, 3), "b")
    valid_a, valid_b = np.isfinite(a).all((1, 2)), np.isfinite(b).all((1, 2))
    a = np.where(valid_a[:, None, None], a, 0.0)
    b = np.where(valid_b[:, None, None], b, 0.0)
    alo, ahi, blo, bhi = a.min(1), a.max(1), b.min(1), b.max(1)
    inter = np.prod(np.maximum(0.0, np.minimum(ahi[:, None], bhi[None])
                               - np.maximum(alo[:, None], blo[None])), axis=-1)
    union = np.prod(ahi - alo, axis=1)[:, None]
    union = union + np.prod(bhi - blo, axis=1)[None, :] - inter
    result = np.zeros_like(inter)
    np.divide(inter, union, out=result, where=union > 0.0)
    return result


def project_box(corners, camera_to_world, K, width, height):
    """Return a clipped xyxy projection, or None for unsafe/empty projection.

    Any corner on or behind the near plane rejects the entire box.  The input
    pose maps camera to world, and both focal lengths and principal point from
    K are used.  This is a conservative ROI, not a visibility detector.
    """
    c = np.asarray(corners, dtype=np.float64)
    pose = np.asarray(camera_to_world, dtype=np.float64)
    intrinsics = np.asarray(K, dtype=np.float64)
    if c.shape != (8, 3) or pose.shape != (4, 4) or intrinsics.shape != (3, 3):
        raise ValueError("expected corners (8,3), camera_to_world (4,4), K (3,3)")
    if (not np.isfinite(c).all() or not np.isfinite(pose).all()
            or not np.isfinite(intrinsics).all() or not np.isfinite([width, height]).all()
            or width <= 0 or height <= 0 or intrinsics[0, 0] <= 0 or intrinsics[1, 1] <= 0):
        return None
    try:
        world_to_camera = np.linalg.inv(pose)
    except np.linalg.LinAlgError:
        return None
    camera = np.column_stack((c, np.ones(8))) @ world_to_camera.T
    if not np.isfinite(camera).all() or np.any(camera[:, 2] <= 1e-4):
        return None
    pixel = camera[:, :3] @ intrinsics.T
    if np.any(pixel[:, 2] <= 1e-8) or not np.isfinite(pixel).all():
        return None
    xy = pixel[:, :2] / pixel[:, 2:3]
    bounds = np.concatenate((xy.min(0), xy.max(0)))
    bounds[[0, 2]] = np.clip(bounds[[0, 2]], 0.0, float(width))
    bounds[[1, 3]] = np.clip(bounds[[1, 3]], 0.0, float(height))
    return bounds if np.all(bounds[2:] > bounds[:2]) else None


def normalize_features(X):
    """L2-normalize the final dimension; zero/invalid vectors become zeros."""
    array = np.asarray(X, dtype=np.float64)
    if array.ndim not in (1, 2):
        raise ValueError("features must be a vector or a matrix")
    finite = np.isfinite(array).all(axis=-1, keepdims=True)
    clean = np.where(finite, array, 0.0)
    norm = np.linalg.norm(clean, axis=-1, keepdims=True)
    result = np.zeros_like(clean)
    np.divide(clean, norm, out=result, where=norm > 1e-12)
    return result.astype(np.float32)


def _nms_ids(boxes, ids, values, limit):
    ordered = ids[np.lexsort((ids, -values))]
    kept = []
    for index in ordered:
        if kept and np.any(box_iou2d(boxes[index:index + 1], boxes[kept])[0] > 0.7):
            continue
        kept.append(int(index))
        if len(kept) >= limit:
            break
    return np.asarray(kept, dtype=np.int64)


def select_local(raw_boxes, raw_scores, raw_embeddings, projection, prototypes,
                 excluded_ids, limit=1):
    """Query/control select from exactly the same geometry-filtered anchor pool.

    The ROI expands each projection side by 25% of its width/height.  Query
    ranks by maximum prototype cosine, control by normal detector score.  Both
    have stable raw-anchor-ID ties and identical 0.7-IoU NMS.  No threshold is
    applied to the normal score, and cosine is not interpreted as probability.
    """
    boxes = _matrix(raw_boxes, (4,), "raw_boxes")
    scores = np.asarray(raw_scores, dtype=np.float64)
    embeddings = np.asarray(raw_embeddings, dtype=np.float64)
    if scores.shape != (len(boxes),) or embeddings.ndim != 2 or len(embeddings) != len(boxes):
        raise ValueError("raw scores/embeddings must align with raw boxes")
    if not isinstance(limit, (int, np.integer)) or limit < 0:
        raise ValueError("limit must be a nonnegative integer")
    proto = np.asarray(prototypes, dtype=np.float64)
    if proto.size == 0:
        proto = np.empty((0, embeddings.shape[1]), dtype=np.float64)
    elif proto.ndim == 1:
        proto = proto[None]
    if proto.ndim != 2 or proto.shape[1] != embeddings.shape[1]:
        raise ValueError("prototype dimension must match raw embeddings")
    roi = np.asarray(projection, dtype=np.float64)
    empty = np.empty(0, dtype=np.int64)
    result = dict(pool_ids=empty.copy(), query_ids=empty.copy(), control_ids=empty.copy(),
                  query_cosines=np.empty(0, dtype=np.float32))
    if roi.shape != (4,):
        raise ValueError("projection must have shape (4,)")
    if not np.isfinite(roi).all() or np.any(roi[2:] <= roi[:2]):
        return result
    centers = (boxes[:, :2] + boxes[:, 2:]) / 2.0
    margin = (roi[2:] - roi[:2]) * 0.25
    normalized = normalize_features(embeddings)
    valid = (np.isfinite(boxes).all(1) & (boxes[:, 2:] > boxes[:, :2]).all(1)
             & np.isfinite(scores) & np.isfinite(embeddings).all(1)
             & (np.linalg.norm(normalized, axis=1) > 0.0)
             & (centers >= roi[:2] - margin).all(1)
             & (centers <= roi[2:] + margin).all(1)
             & (box_iou2d(boxes, roi[None])[:, 0] >= 0.05))
    excluded = np.asarray(excluded_ids, dtype=np.int64).reshape(-1)
    valid &= ~np.isin(np.arange(len(boxes)), excluded)
    pool = np.flatnonzero(valid).astype(np.int64)
    result["pool_ids"] = pool
    if len(pool) == 0 or limit == 0:
        return result
    result["control_ids"] = _nms_ids(boxes, pool, scores[pool], limit)
    normalized_proto = normalize_features(proto)
    normalized_proto = normalized_proto[np.linalg.norm(normalized_proto, axis=1) > 0.0]
    if len(normalized_proto):
        cosine = np.clip(normalized @ normalized_proto.T, -1.0, 1.0).max(axis=1)
        result["query_ids"] = _nms_ids(boxes, pool, cosine[pool], limit)
        result["query_cosines"] = cosine[result["query_ids"]].astype(np.float32)
    return result


class PendingMemory:
    """Bounded normal-stream memory; queries cannot become confirmation evidence.

    Call eligible(t) before commit(t, ...).  Returned dictionaries are copies.
    obs/frames contain at most max_obs normal observations; features contains
    at most two normalized visual prototypes.  Confirmation is latched at
    three distinct frames, independently of retained-history length.
    """

    def __init__(self, max_tracks=64, ttl=10, max_obs=3):
        for name, value in (("max_tracks", max_tracks), ("ttl", ttl), ("max_obs", max_obs)):
            if not isinstance(value, (int, np.integer)) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        self.max_tracks, self.ttl, self.max_obs = int(max_tracks), int(ttl), int(max_obs)
        self._tracks = {}
        self._next_id = 0
        self._clock = -1
        self._last_commit = None
        self._feature_dim = None

    def _advance(self, ordinal):
        if not isinstance(ordinal, (int, np.integer)) or ordinal < 0:
            raise ValueError("ordinal must be a nonnegative integer")
        if ordinal < self._clock:
            raise ValueError("past-only memory cannot move backwards")
        self._clock = int(ordinal)
        self._tracks = {i: t for i, t in self._tracks.items()
                        if ordinal - t["last_ord"] <= self.ttl}

    def eligible(self, ordinal, limit=8):
        if not isinstance(limit, (int, np.integer)) or limit < 0:
            raise ValueError("limit must be a nonnegative integer")
        self._advance(ordinal)
        candidates = [t for t in self._tracks.values()
                      if not t["confirmed"] and 1 <= t["frame_count"] <= 2
                      and t["last_ord"] < ordinal]
        candidates.sort(key=lambda t: (-t["frame_count"], -t["score"], -t["last_ord"], t["id"]))
        return copy.deepcopy(candidates[:limit])

    def commit(self, ordinal, frame_id, corners, embeddings, scores):
        """Commit only NORMAL detector observations, never query/control outputs."""
        boxes = _matrix(corners, (8, 3), "corners")
        feature = np.asarray(embeddings, dtype=np.float64)
        score = np.asarray(scores, dtype=np.float64)
        if feature.ndim != 2 or len(feature) != len(boxes) or score.shape != (len(boxes),):
            raise ValueError("normal scores/embeddings must align with corners")
        if self._feature_dim is not None and feature.shape[1] != self._feature_dim:
            raise ValueError("feature dimension cannot change within a scene")
        if not isinstance(frame_id, (int, np.integer)) or frame_id < 0:
            raise ValueError("frame_id must be a nonnegative integer")
        if self._last_commit is not None:
            last_ord, last_frame = self._last_commit
            if ordinal == last_ord and frame_id == last_frame:
                return dict(added=0, updated=0, skipped=len(boxes), duplicate_commit=True)
            if ordinal <= last_ord or frame_id <= last_frame:
                raise ValueError("commits require strictly increasing ordinal/frame_id")
        self._advance(ordinal)
        self._feature_dim = feature.shape[1]
        self._last_commit = (int(ordinal), int(frame_id))
        feature = normalize_features(feature)
        valid = (np.isfinite(boxes).all((1, 2)) & np.isfinite(score)
                 & (boxes.max(1) > boxes.min(1)).all(1)
                 & (np.linalg.norm(feature, axis=1) > 0.0))
        rows = np.flatnonzero(valid)
        rows = rows[np.lexsort((rows, -score[rows]))]
        stats = dict(added=0, updated=0, skipped=int((~valid).sum()), duplicate_commit=False)
        used = set()
        for row in rows:
            center = boxes[row].mean(0)
            matches = []
            for track_id, track in self._tracks.items():
                old_box = track["obs"][-1]
                distance = float(np.linalg.norm(center - old_box.mean(0)))
                if distance <= 0.5:
                    overlap = float(box_iou3d(boxes[row:row + 1], old_box[None])[0, 0])
                    if overlap >= 0.1:
                        matches.append((-overlap, distance, track_id))
            if matches:
                track_id = min(matches)[2]
                if track_id in used:
                    stats["skipped"] += 1
                    continue
                track = self._tracks[track_id]
                stats["updated"] += 1
            else:
                if len(self._tracks) >= self.max_tracks:
                    victim = min(self._tracks.values(),
                                 key=lambda t: (t["last_ord"], t["score"], -t["id"]))
                    if victim["last_ord"] == ordinal and victim["score"] >= score[row]:
                        stats["skipped"] += 1
                        continue
                    del self._tracks[victim["id"]]
                track_id = self._next_id
                self._next_id += 1
                track = dict(id=track_id, obs=[], frames=[], features=[], feature_scores=[],
                             score=float(score[row]), last_ord=int(ordinal), frame_count=0,
                             confirmed=False)
                self._tracks[track_id] = track
                stats["added"] += 1
            used.add(track_id)
            track["obs"].append(boxes[row].copy())
            track["frames"].append(int(frame_id))
            track["obs"] = track["obs"][-self.max_obs:]
            track["frames"] = track["frames"][-self.max_obs:]
            track["frame_count"] += 1
            track["confirmed"] = track["confirmed"] or track["frame_count"] >= 3
            track["last_ord"] = int(ordinal)
            track["score"] = max(track["score"], float(score[row]))
            if not any(float(np.dot(p, feature[row])) >= 0.999999 for p in track["features"]):
                track["features"].append(feature[row].copy())
                track["feature_scores"].append(float(score[row]))
                order = sorted(range(len(track["features"])),
                               key=lambda i: (-track["feature_scores"][i], i))[:2]
                track["features"] = [track["features"][i] for i in order]
                track["feature_scores"] = [track["feature_scores"][i] for i in order]
        return stats
