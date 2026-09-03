"""Fail-closed persistent worker client for CA-1M terminal TR3D.

This is intentionally separate from the generic/ScanNet client.  The CA-1M
observer cache needs additional provenance (the local TR3D boxes, one-class
labels, and an input point hash) and a real, non-blocking response timeout.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from multiprocessing import shared_memory
import os
from pathlib import Path
import selectors
import subprocess
import time
from typing import Any, Sequence

import numpy as np


_PREFIX = "BOXFUSION_TR3D_RESPONSE "
_PREFIX_BYTES = _PREFIX.encode("ascii")
MAX_STARTUP_TIMEOUT_S = 600.0
STARTUP_ABORT_GRACE_S = 5.0
INFERENCE_TIMEOUT_S = 120.0


@dataclass(frozen=True)
class CA1MTR3DWorkerResult:
    corners_world: np.ndarray
    boxes_local: np.ndarray
    scores: np.ndarray
    labels: np.ndarray
    point_counts: np.ndarray
    model_runtime_s: float
    source_points_sha256: str
    adapter_mode: str


class CA1MTR3DWorker:
    """Keep the official model resident and validate every returned field."""

    def __init__(
        self,
        *,
        python: str,
        worker_script: str,
        runtime_root: str,
        config: str,
        checkpoint: str,
        project_root: str,
        vendor_root: str,
        startup_timeout_s: float,
        device: str = "cuda:0",
        extra_args: Sequence[str] = (),
    ) -> None:
        timeout = float(startup_timeout_s)
        if (
            isinstance(startup_timeout_s, bool)
            or not math.isfinite(timeout)
            or timeout <= 0.0
            or timeout > MAX_STARTUP_TIMEOUT_S
        ):
            raise ValueError("startup timeout must be finite in (0,600] seconds")
        self.startup_timeout_s = timeout
        command = [
            str(Path(python).resolve()),
            str(Path(worker_script).resolve()),
            "--runtime-root",
            str(Path(runtime_root).resolve()),
            "--config",
            str(Path(config).resolve()),
            "--checkpoint",
            str(Path(checkpoint).resolve()),
            "--project-root",
            str(Path(project_root).resolve()),
            "--vendor-root",
            str(Path(vendor_root).resolve()),
            "--device",
            device,
            "--startup-timeout-s",
            str(timeout),
            *extra_args,
        ]
        environment = dict(os.environ)
        for name in ("PYTHONPATH", "PYTHONHOME", "LD_LIBRARY_PATH"):
            environment.pop(name, None)
        environment["PYTHONNOUSERSITE"] = "1"
        # Never mix selector readiness on the OS pipe with TextIOWrapper's
        # private read-ahead buffer.  OpenMMLab writes ordinary stdout lines
        # before our READY record; text readline() could prefetch READY, return
        # only the preceding line, then leave selector waiting on an empty fd.
        self._stdout_buffer = bytearray()
        self.process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=None,
            text=False,
            bufsize=0,
            env=environment,
        )
        if self.process.stdout is None:
            self._abort_and_reap()
            raise RuntimeError("CA-1M TR3D worker stdout is unavailable")
        os.set_blocking(self.process.stdout.fileno(), False)
        try:
            ready = self._response(timeout_s=self.startup_timeout_s)
        except BaseException:
            # During initialization the child has not entered its stdin loop,
            # so sending a JSON close request cannot work.  Abort directly and
            # reap it within a fixed grace period; never leave a CUDA worker
            # behind after a bounded cold-start failure.
            self._abort_and_reap()
            raise
        if ready.get("status") != "ready":
            self.close()
            raise RuntimeError(f"CA-1M TR3D worker failed to initialize: {ready}")
        self.adapter_mode = "synthetic" if ready.get("synthetic") is True else "genuine"
        self.startup_s = float(ready.get("startup_s", -1.0))
        if (
            not math.isfinite(self.startup_s)
            or self.startup_s < 0.0
            or self.startup_s > self.startup_timeout_s
        ):
            self.close()
            raise ValueError("CA-1M TR3D worker returned invalid startup duration")

    def _response(self, *, timeout_s: float) -> dict[str, Any]:
        stdout = self.process.stdout
        if stdout is None:
            raise RuntimeError("CA-1M TR3D worker stdout is unavailable")
        selector = selectors.DefaultSelector()
        selector.register(stdout, selectors.EVENT_READ)
        deadline = time.monotonic() + float(timeout_s)
        try:
            while True:
                while b"\n" in self._stdout_buffer:
                    raw, _, remainder = self._stdout_buffer.partition(b"\n")
                    self._stdout_buffer[:] = remainder
                    line = raw.rstrip(b"\r")
                    if line.startswith(_PREFIX_BYTES):
                        return json.loads(line[len(_PREFIX_BYTES) :].decode("utf-8"))
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    raise TimeoutError("CA-1M TR3D worker response timed out")
                if not selector.select(remaining):
                    raise TimeoutError("CA-1M TR3D worker response timed out")
                try:
                    chunk = os.read(stdout.fileno(), 65536)
                except BlockingIOError:
                    continue
                if not chunk:
                    raise RuntimeError(
                        f"CA-1M TR3D worker exited with {self.process.poll()}"
                    )
                self._stdout_buffer.extend(chunk)
        finally:
            selector.close()

    def _abort_and_reap(self) -> None:
        """Boundedly terminate and reap a worker that cannot service stdin."""

        process = getattr(self, "process", None)
        if process is None or process.poll() is not None:
            if process is not None:
                process.wait()
            return
        process.terminate()
        try:
            process.wait(timeout=STARTUP_ABORT_GRACE_S)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=STARTUP_ABORT_GRACE_S)

    def infer(
        self,
        *,
        scene_id: str,
        prefix_id: str,
        points_world_xyzrgb: np.ndarray,
        world_to_local: np.ndarray,
    ) -> CA1MTR3DWorkerResult:
        points = np.ascontiguousarray(points_world_xyzrgb, dtype=np.float32)
        if (
            points.ndim != 2
            or points.shape[1] != 6
            or not len(points)
            or not np.isfinite(points).all()
        ):
            raise ValueError("CA-1M TR3D input must be finite float32 [N,6]")
        transform = np.asarray(world_to_local, dtype=np.float64)
        if transform.shape != (4, 4) or not np.isfinite(transform).all():
            raise ValueError("world_to_local must be finite [4,4]")
        source_sha = hashlib.sha256(points.tobytes(order="C")).hexdigest()
        memory = shared_memory.SharedMemory(create=True, size=points.nbytes)
        try:
            shared = np.ndarray(points.shape, dtype=np.float32, buffer=memory.buf)
            shared[:] = points
            request = {
                "command": "infer",
                "scene_id": scene_id,
                "prefix_id": prefix_id,
                "shared_memory": memory.name,
                "shape": list(points.shape),
                "world_to_local": transform.tolist(),
                # Retain the old key only so the pinned adapter's input name
                # cannot accidentally change the mathematical transform.
                "axis_align_matrix": transform.tolist(),
                "source_points_sha256": source_sha,
            }
            if self.process.stdin is None:
                raise RuntimeError("CA-1M TR3D worker stdin is unavailable")
            self.process.stdin.write(
                (json.dumps(request, separators=(",", ":")) + "\n").encode("utf-8")
            )
            self.process.stdin.flush()
            try:
                response = self._response(timeout_s=INFERENCE_TIMEOUT_S)
            except TimeoutError:
                # The worker may be inside an uninterruptible CUDA/OpenMMLab
                # call and cannot consume a close request.  Reap it now rather
                # than relying on context-manager cleanup after propagation.
                self._abort_and_reap()
                raise
            if response.get("status") != "ok":
                raise RuntimeError(f"CA-1M TR3D worker inference failed: {response}")
            if response.get("scene_id") != scene_id:
                raise ValueError("CA-1M TR3D worker scene response mismatch")
            if response.get("prefix_id") != prefix_id:
                raise ValueError("CA-1M TR3D worker prefix response mismatch")
            if response.get("source_points_sha256") != source_sha:
                raise ValueError("CA-1M TR3D cross-process point hash mismatch")
            if response.get("adapter_mode") != self.adapter_mode:
                raise ValueError("CA-1M TR3D adapter mode response mismatch")
            corners = np.asarray(response["corners_world"], dtype=np.float32).reshape(
                -1, 8, 3
            )
            boxes = np.asarray(response["boxes_local"], dtype=np.float32).reshape(-1, 7)
            scores = np.asarray(response["scores"], dtype=np.float32)
            labels = np.asarray(response["labels"], dtype=np.int64)
            counts = np.asarray(response["point_counts"], dtype=np.int64)
            rows = len(corners)
            if (
                boxes.shape != (rows, 7)
                or scores.shape != (rows,)
                or labels.shape != (rows,)
                or counts.shape != (rows,)
                or not np.isfinite(corners).all()
                or not np.isfinite(boxes).all()
                or not np.isfinite(scores).all()
                or np.any(boxes[:, 3:6] <= 0.0)
                or np.any(scores < 0.0)
                or np.any(scores > 1.0)
                or np.any(labels != 0)
                or np.any(counts < 0)
            ):
                raise ValueError("CA-1M TR3D worker returned malformed proposals")
            runtime = float(response["model_runtime_s"])
            if not np.isfinite(runtime) or runtime < 0.0:
                raise ValueError("CA-1M TR3D worker returned invalid runtime")
            return CA1MTR3DWorkerResult(
                corners_world=np.ascontiguousarray(corners),
                boxes_local=np.ascontiguousarray(boxes),
                scores=np.ascontiguousarray(scores),
                labels=np.ascontiguousarray(labels),
                point_counts=np.ascontiguousarray(counts),
                model_runtime_s=runtime,
                source_points_sha256=source_sha,
                adapter_mode=self.adapter_mode,
            )
        finally:
            memory.close()
            memory.unlink()

    def close(self) -> None:
        process = getattr(self, "process", None)
        if process is None or process.poll() is not None:
            return
        try:
            if process.stdin is None:
                raise RuntimeError("CA-1M TR3D worker stdin is unavailable")
            process.stdin.write(b'{"command":"close"}\n')
            process.stdin.flush()
            process.wait(timeout=10.0)
        except Exception:
            process.terminate()
            try:
                process.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5.0)

    def __enter__(self) -> "CA1MTR3DWorker":
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()


__all__ = [
    "CA1MTR3DWorker",
    "CA1MTR3DWorkerResult",
    "INFERENCE_TIMEOUT_S",
    "STARTUP_ABORT_GRACE_S",
    "MAX_STARTUP_TIMEOUT_S",
]
