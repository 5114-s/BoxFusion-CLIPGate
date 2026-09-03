"""Fail-closed CA-1M E961 adapter for the v3 incremental L6 protocol.

The adapter is deliberately construction-only: it does not create a worker.
It preserves the exact R6/P ``world_to_local`` float64 C-order bytes and rejects
even numerically equivalent byte drift, including +0.0 versus -0.0 changes.
"""

from __future__ import annotations

import hashlib
import math
from typing import Any

import numpy as np

from .tr3d_incremental_online import TR3DProviderResult


def _strict_transform(value: Any, name: str) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise ValueError(f"{name} must be an ndarray")
    if value.dtype != np.dtype(np.float64):
        raise ValueError(f"{name} must have exact float64 dtype")
    if value.shape != (4, 4) or not value.flags.c_contiguous:
        raise ValueError(f"{name} must be a C-contiguous [4,4] matrix")
    if (
        not np.isfinite(value).all()
        or not np.array_equal(value[3], np.array([0.0, 0.0, 0.0, 1.0]))
    ):
        raise ValueError(f"{name} must be finite homogeneous [4,4]")
    return value


class CA1ME961IncrementalProviderV3:
    """Adapt the pinned CA worker without relaxing transform identity."""

    def __init__(self, worker: Any, *, world_to_local: Any) -> None:
        transform = _strict_transform(world_to_local, "R6/P world_to_local")
        preserved = transform.copy(order="C")
        preserved.setflags(write=False)
        self.worker = worker
        self.world_to_local = preserved
        self._world_to_local_bytes = preserved.tobytes(order="C")

    def infer(
        self, *, scene_id: str, prefix_id: str,
        points_world_xyzrgb: np.ndarray, axis_align_matrix: np.ndarray,
    ) -> TR3DProviderResult:
        supplied = _strict_transform(
            axis_align_matrix, "incremental observer axis_align_matrix",
        )
        if supplied.tobytes(order="C") != self._world_to_local_bytes:
            raise ValueError(
                "incremental observer transform bytes differ from R6/P world_to_local"
            )
        points = np.ascontiguousarray(points_world_xyzrgb, dtype=np.float32)
        if points.ndim != 2 or points.shape[1] != 6 or not np.isfinite(points).all():
            raise ValueError("incremental observer points must be finite [N,6]")
        result = self.worker.infer(
            scene_id=str(scene_id), prefix_id=str(prefix_id),
            points_world_xyzrgb=points, world_to_local=self.world_to_local,
        )
        corners = np.asarray(result.corners_world, dtype=np.float32)
        scores = np.asarray(result.scores, dtype=np.float32)
        labels_source = np.asarray(result.labels)
        counts_source = np.asarray(result.point_counts)
        if labels_source.dtype.kind not in "iu" or counts_source.dtype.kind not in "iu":
            raise ValueError("CA worker labels/counts must retain integer dtype")
        labels = np.ascontiguousarray(labels_source, dtype=np.int64)
        counts = np.ascontiguousarray(counts_source, dtype=np.int64)
        runtime_source = np.asarray(result.model_runtime_s)
        if runtime_source.shape != () or runtime_source.dtype.kind == "b":
            raise ValueError("CA worker runtime must be a numeric non-bool scalar")
        runtime = float(runtime_source)
        expected_sha = hashlib.sha256(points.tobytes(order="C")).hexdigest()
        if (
            getattr(result, "adapter_mode", None) != "genuine"
            or getattr(result, "source_points_sha256", None) != expected_sha
            or corners.ndim != 3 or corners.shape[1:] != (8, 3)
            or scores.shape != (len(corners),)
            or labels.shape != (len(corners),)
            or counts.shape != (len(corners),)
            or not np.isfinite(corners).all()
            or not np.isfinite(scores).all()
            or np.any((scores < 0.0) | (scores > 1.0))
            or np.any(labels != 0)
            or np.any(counts < 0)
            or not math.isfinite(runtime) or runtime < 0.0
        ):
            raise ValueError("CA E961 incremental provider result differs")
        return TR3DProviderResult(
            np.ascontiguousarray(corners), np.ascontiguousarray(scores),
            runtime, counts,
        )


__all__ = ["CA1ME961IncrementalProviderV3"]
