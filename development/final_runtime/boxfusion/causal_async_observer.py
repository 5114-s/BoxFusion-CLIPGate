"""Bounded asynchronous observer for immutable causal proposal receipts.

The observer deliberately has no output-mutation API.  Each task owns a copy
of an already committed S3R receipt and is permanently bound to its scene,
confirmation frame, candidate identity and memory version.  Worker code can
therefore neither read future frames nor rebind a late result to newer state.
"""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import asdict, dataclass
import hashlib
import math
from typing import Any

import numpy as np

from boxfusion.s3r_receipt_tracker import S3RReceipt


SCHEMA = "boxfusion.causal_async_observer.v1"


@dataclass(frozen=True)
class CausalAsyncObserverConfig:
    max_workers: int = 2
    max_pending_tasks: int = 32
    max_results: int = 1024
    max_result_lag_keyframes: int = 4

    def __post_init__(self) -> None:
        for name, ceiling in (
            ("max_workers", 8),
            ("max_pending_tasks", 256),
            ("max_results", 1024),
            ("max_result_lag_keyframes", 64),
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"{name} must be an integer")
            minimum = 0 if name == "max_result_lag_keyframes" else 1
            if value < minimum or value > ceiling:
                raise ValueError(f"{name} must lie in [{minimum},{ceiling}]")

    def as_dict(self) -> dict[str, int]:
        return asdict(self)


@dataclass(frozen=True)
class CausalObserverTask:
    serial: int
    scene_id: str
    candidate_id: int
    enqueue_frame_id: int
    enqueue_keyframe_step: int
    memory_version: int
    evidence_frame_ids: tuple[int, int, int]
    evidence_source_rows: tuple[int, int, int]
    evidence_scores: tuple[float, float, float]
    evidence_corners: np.ndarray

    def __post_init__(self) -> None:
        for name in (
            "serial", "candidate_id", "enqueue_frame_id",
            "enqueue_keyframe_step", "memory_version",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if not self.scene_id or len(self.scene_id) > 128:
            raise ValueError("scene_id must be nonempty and bounded")
        frames = tuple(int(value) for value in self.evidence_frame_ids)
        rows = tuple(int(value) for value in self.evidence_source_rows)
        scores = tuple(float(value) for value in self.evidence_scores)
        if (
            len(frames) != 3 or frames != tuple(sorted(frames))
            or len(set(frames)) != 3 or frames[-1] != self.enqueue_frame_id
            or len(rows) != 3 or len(set(rows)) != 3
            or len(scores) != 3
            or any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in scores)
        ):
            raise ValueError("task evidence must be three distinct causal rows")
        corners = np.asarray(self.evidence_corners, dtype=np.float64)
        if corners.shape != (3, 8, 3) or not np.isfinite(corners).all():
            raise ValueError("evidence_corners must be finite [3,8,3]")
        immutable = np.frombuffer(
            np.ascontiguousarray(corners).tobytes(), dtype=np.float64
        ).reshape(3, 8, 3)
        object.__setattr__(self, "evidence_frame_ids", frames)
        object.__setattr__(self, "evidence_source_rows", rows)
        object.__setattr__(self, "evidence_scores", scores)
        object.__setattr__(self, "evidence_corners", immutable)


@dataclass(frozen=True)
class CausalObserverResult:
    serial: int
    scene_id: str
    candidate_id: int
    enqueue_frame_id: int
    enqueue_keyframe_step: int
    memory_version: int
    evidence_frame_ids: tuple[int, int, int]
    evidence_source_rows: tuple[int, int, int]
    evidence_sha256: str
    mean_score: float
    median_pairwise_center_distance_m: float
    median_aabb_extent_m: float

    def to_json_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class _Pending:
    task: CausalObserverTask
    future: Future[CausalObserverResult]


def _sha256_array(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes())
    return digest.hexdigest()


def _execute(task: CausalObserverTask) -> CausalObserverResult:
    """Pure worker: consume only the immutable task payload."""

    if max(task.evidence_frame_ids) > task.enqueue_frame_id:
        raise RuntimeError("future evidence reached asynchronous worker")
    centers = 0.5 * (
        task.evidence_corners.min(axis=1) + task.evidence_corners.max(axis=1)
    )
    distances = np.linalg.norm(centers[:, None] - centers[None, :], axis=2)
    pairwise = distances[np.triu_indices(3, 1)]
    extents = task.evidence_corners.max(axis=1) - task.evidence_corners.min(axis=1)
    return CausalObserverResult(
        serial=task.serial,
        scene_id=task.scene_id,
        candidate_id=task.candidate_id,
        enqueue_frame_id=task.enqueue_frame_id,
        enqueue_keyframe_step=task.enqueue_keyframe_step,
        memory_version=task.memory_version,
        evidence_frame_ids=task.evidence_frame_ids,
        evidence_source_rows=task.evidence_source_rows,
        evidence_sha256=_sha256_array(task.evidence_corners),
        mean_score=float(np.mean(task.evidence_scores)),
        median_pairwise_center_distance_m=float(np.median(pairwise)),
        median_aabb_extent_m=float(np.median(extents)),
    )


class BoundedCausalAsyncObserver:
    """Fail-closed bounded scheduler whose results are diagnostic only."""

    def __init__(
        self,
        scene_id: str,
        config: CausalAsyncObserverConfig | None = None,
    ) -> None:
        self.scene_id = str(scene_id)
        self.config = config or CausalAsyncObserverConfig()
        if not self.scene_id or len(self.scene_id) > 128:
            raise ValueError("scene_id must be nonempty and bounded")
        self._executor = ThreadPoolExecutor(
            max_workers=self.config.max_workers,
            thread_name_prefix="boxfusion-c1",
        )
        self._pending: list[_Pending] = []
        self._results: list[CausalObserverResult] = []
        self._drops: list[dict[str, Any]] = []
        self._serial = 0
        self._last_memory_version = 0
        self._last_poll_step = -1
        self._submitted = 0
        self._closed = False
        self._peak_pending = 0

    def _drop(self, task: CausalObserverTask, reason: str) -> None:
        self._drops.append(
            {
                "serial": task.serial,
                "candidate_id": task.candidate_id,
                "enqueue_frame_id": task.enqueue_frame_id,
                "enqueue_keyframe_step": task.enqueue_keyframe_step,
                "memory_version": task.memory_version,
                "reason": reason,
            }
        )

    def submit(
        self,
        receipt: S3RReceipt,
        *,
        keyframe_step: int,
        memory_version: int,
    ) -> bool:
        if self._closed:
            raise RuntimeError("observer is closed")
        if memory_version < self._last_memory_version:
            raise ValueError("memory_version must be monotonic")
        self._last_memory_version = memory_version
        self._serial += 1
        task = CausalObserverTask(
            serial=self._serial,
            scene_id=self.scene_id,
            candidate_id=int(receipt.track_id),
            enqueue_frame_id=int(receipt.confirmation_frame_id),
            enqueue_keyframe_step=int(keyframe_step),
            memory_version=int(memory_version),
            evidence_frame_ids=receipt.evidence_frame_ids,
            evidence_source_rows=receipt.evidence_source_rows,
            evidence_scores=receipt.evidence_scores,
            evidence_corners=receipt.evidence_corners,
        )
        if len(self._pending) >= self.config.max_pending_tasks:
            self._drop(task, "queue_capacity")
            return False
        if len(self._results) >= self.config.max_results:
            self._drop(task, "result_capacity")
            return False
        self._pending.append(_Pending(task=task, future=self._executor.submit(_execute, task)))
        self._submitted += 1
        self._peak_pending = max(self._peak_pending, len(self._pending))
        return True

    def poll(self, keyframe_step: int, *, block: bool = False) -> None:
        if self._closed:
            raise RuntimeError("observer is closed")
        if keyframe_step < self._last_poll_step:
            raise ValueError("poll keyframe_step must be monotonic")
        self._last_poll_step = keyframe_step
        retained: list[_Pending] = []
        for pending in self._pending:
            task = pending.task
            lag = keyframe_step - task.enqueue_keyframe_step
            if lag > self.config.max_result_lag_keyframes:
                pending.future.cancel()
                self._drop(task, "expired")
                continue
            if not block and not pending.future.done():
                retained.append(pending)
                continue
            try:
                result = pending.future.result()
            except Exception as error:
                self._drop(task, f"worker_error:{type(error).__name__}")
                continue
            if len(self._results) >= self.config.max_results:
                self._drop(task, "result_capacity")
            else:
                self._results.append(result)
        self._pending = retained

    def close(self, final_keyframe_step: int) -> None:
        if self._closed:
            return
        self.poll(final_keyframe_step, block=True)
        self._executor.shutdown(wait=True, cancel_futures=True)
        self._closed = True

    def result_rows(self) -> tuple[dict[str, Any], ...]:
        if not self._closed:
            raise RuntimeError("close observer before reading results")
        return tuple(
            result.to_json_dict()
            for result in sorted(self._results, key=lambda row: row.serial)
        )

    def drop_rows(self) -> tuple[dict[str, Any], ...]:
        if not self._closed:
            raise RuntimeError("close observer before reading drops")
        return tuple(sorted(self._drops, key=lambda row: int(row["serial"])))

    def summary(self) -> dict[str, Any]:
        if not self._closed:
            raise RuntimeError("close observer before reading summary")
        return {
            "schema": SCHEMA,
            "observer_only": True,
            "output_inert": True,
            "active_authorized": False,
            "output_mutation_applied": False,
            "past_only": True,
            "immutable_task_payload": True,
            "late_result_rebinding": False,
            "bounded_queue": True,
            "config": self.config.as_dict(),
            "submitted_tasks": self._submitted,
            "completed_results": len(self._results),
            "dropped_tasks": len(self._drops),
            "peak_pending_tasks": self._peak_pending,
        }


__all__ = [
    "BoundedCausalAsyncObserver",
    "CausalAsyncObserverConfig",
    "CausalObserverResult",
    "CausalObserverTask",
    "SCHEMA",
]
