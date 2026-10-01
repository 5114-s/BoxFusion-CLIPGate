"""Pure protocol adapter from CA-1M worker results to incremental TR3D.

Construction does not start a worker.  The caller must supply the canonical
R6-bound worker and the per-scene ``world_to_local`` matrix recorded by P.
"""

from __future__ import annotations

import hashlib
import math
from typing import Any

import numpy as np

from .tr3d_incremental_online import TR3DProviderResult


class CA1ME961IncrementalProviderV2:
    def __init__(self, worker: Any, *, world_to_local: Any) -> None:
        transform = np.asarray(world_to_local, dtype=np.float64)
        if (
            transform.shape != (4, 4)
            or not np.isfinite(transform).all()
            or not np.allclose(transform[3], [0.0, 0.0, 0.0, 1.0], atol=1e-8)
        ):
            raise ValueError("E961 world_to_local must be finite homogeneous [4,4]")
        self.worker = worker
        self.world_to_local = np.ascontiguousarray(transform)

    def infer(
        self, *, scene_id: str, prefix_id: str,
        points_world_xyzrgb: np.ndarray, axis_align_matrix: np.ndarray,
    ) -> TR3DProviderResult:
        points = np.ascontiguousarray(points_world_xyzrgb, dtype=np.float32)
        supplied = np.asarray(axis_align_matrix, dtype=np.float64)
        if supplied.shape != (4, 4) or not np.array_equal(supplied, self.world_to_local):
            raise ValueError("incremental observer transform differs from R6 P world_to_local")
        result = self.worker.infer(
            scene_id=str(scene_id), prefix_id=str(prefix_id),
            points_world_xyzrgb=points, world_to_local=self.world_to_local,
        )
        corners = np.asarray(result.corners_world, dtype=np.float32)
        scores = np.asarray(result.scores, dtype=np.float32)
        labels = np.asarray(result.labels)
        counts = np.asarray(result.point_counts, dtype=np.int64)
        runtime = float(result.model_runtime_s)
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
            runtime, np.ascontiguousarray(counts),
        )


__all__ = ["CA1ME961IncrementalProviderV2"]
