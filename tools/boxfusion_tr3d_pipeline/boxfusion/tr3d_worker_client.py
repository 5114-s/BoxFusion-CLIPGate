"""Persistent cross-environment TR3D worker client using shared memory."""

from __future__ import annotations

import json
from multiprocessing import shared_memory
import os
from pathlib import Path
import subprocess
import time
from typing import Any, Sequence

import numpy as np

from .tr3d_incremental_online import TR3DProviderResult


_PREFIX = "BOXFUSION_TR3D_RESPONSE "


class PersistentTR3DWorker:
    """Keep the official TR3D model resident outside the BoxFusion env."""

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
        device: str = "cuda:0",
        extra_args: Sequence[str] = (),
    ) -> None:
        command = [
            str(Path(python).resolve()), str(Path(worker_script).resolve()),
            "--runtime-root", str(Path(runtime_root).resolve()),
            "--config", str(Path(config).resolve()),
            "--checkpoint", str(Path(checkpoint).resolve()),
            "--project-root", str(Path(project_root).resolve()),
            "--vendor-root", str(Path(vendor_root).resolve()),
            "--device", device, *extra_args,
        ]
        environment = dict(os.environ)
        for name in ("PYTHONPATH", "PYTHONHOME", "LD_LIBRARY_PATH"):
            environment.pop(name, None)
        environment["PYTHONNOUSERSITE"] = "1"
        self.process = subprocess.Popen(
            command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=None, text=True, bufsize=1, env=environment,
        )
        ready = self._response(timeout_s=180.0)
        if ready.get("status") != "ready":
            self.close()
            raise RuntimeError(f"TR3D worker failed to initialize: {ready}")

    def _response(self, *, timeout_s: float) -> dict[str, Any]:
        assert self.process.stdout is not None
        started = time.monotonic()
        while time.monotonic() - started < timeout_s:
            line = self.process.stdout.readline()
            if line == "":
                raise RuntimeError(f"TR3D worker exited with {self.process.poll()}")
            if line.startswith(_PREFIX):
                return json.loads(line[len(_PREFIX):])
        raise TimeoutError("TR3D worker response timed out")

    def infer(
        self,
        *,
        scene_id: str,
        prefix_id: str,
        points_world_xyzrgb: np.ndarray,
        axis_align_matrix: np.ndarray,
    ) -> TR3DProviderResult:
        points = np.ascontiguousarray(points_world_xyzrgb, dtype=np.float32)
        if points.ndim != 2 or points.shape[1] != 6 or not len(points):
            raise ValueError("TR3D worker input must be non-empty float32 [N,6]")
        memory = shared_memory.SharedMemory(create=True, size=points.nbytes)
        try:
            shared = np.ndarray(points.shape, dtype=np.float32, buffer=memory.buf)
            shared[:] = points
            request = {
                "command": "infer", "scene_id": scene_id, "prefix_id": prefix_id,
                "shared_memory": memory.name, "shape": list(points.shape),
                "axis_align_matrix": np.asarray(axis_align_matrix, dtype=np.float64).tolist(),
            }
            assert self.process.stdin is not None
            self.process.stdin.write(json.dumps(request, separators=(",", ":")) + "\n")
            self.process.stdin.flush()
            response = self._response(timeout_s=120.0)
            if response.get("status") != "ok":
                raise RuntimeError(f"TR3D worker inference failed: {response}")
            return TR3DProviderResult(
                np.asarray(response["corners_world"], dtype=np.float32).reshape(-1, 8, 3),
                np.asarray(response["scores"], dtype=np.float32),
                float(response["model_runtime_s"]),
                np.asarray(response["point_counts"], dtype=np.int64),
            )
        finally:
            memory.close(); memory.unlink()

    def close(self) -> None:
        process = getattr(self, "process", None)
        if process is None or process.poll() is not None:
            return
        try:
            assert process.stdin is not None
            process.stdin.write('{"command":"close"}\n'); process.stdin.flush()
            process.wait(timeout=10.0)
        except Exception:
            process.terminate()
            try: process.wait(timeout=5.0)
            except subprocess.TimeoutExpired: process.kill()

    def __enter__(self) -> "PersistentTR3DWorker":
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()


__all__ = ["PersistentTR3DWorker"]
