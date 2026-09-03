#!/usr/bin/env python3
"""Persistent official-TR3D worker for CA-1M terminal observation.

This worker is intentionally separate from ``tr3d_online_worker.py``.  The
older worker serializes a ScanNet-only cache and therefore rejects CA-1M's
numeric scene identifiers.  Here the official model still runs unchanged;
only its local-frame boxes are converted back to CA-1M world corners before
returning them to the observer process.
"""

from __future__ import annotations

import argparse
import faulthandler
import hashlib
import json
from multiprocessing import resource_tracker, shared_memory
from pathlib import Path
import re
import sys
import time
import traceback

import numpy as np


PREFIX = "BOXFUSION_TR3D_RESPONSE "
SCENE_RE = re.compile(r"^[0-9]{8}$")
CORNER_SIGNS = np.asarray(
    [
        [-1, -1, -1],
        [-1, -1, 1],
        [-1, 1, -1],
        [-1, 1, 1],
        [1, -1, -1],
        [1, -1, 1],
        [1, 1, -1],
        [1, 1, 1],
    ],
    dtype=np.float64,
)


def respond(payload: dict) -> None:
    print(PREFIX + json.dumps(payload, separators=(",", ":")), flush=True)


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--runtime-root", type=Path, required=True)
    value.add_argument("--config", type=Path, required=True)
    value.add_argument("--checkpoint", type=Path, required=True)
    value.add_argument("--project-root", type=Path, required=True)
    value.add_argument("--vendor-root", type=Path, required=True)
    value.add_argument("--device", default="cuda:0")
    value.add_argument("--startup-timeout-s", type=float, required=True)
    value.add_argument("--score-threshold", type=float, default=0.01)
    value.add_argument("--max-proposals", type=int, default=256)
    value.add_argument("--synthetic", action="store_true")
    return value


def _homogeneous(value: object) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64)
    if (
        matrix.shape != (4, 4)
        or not np.isfinite(matrix).all()
        or not np.allclose(matrix[3], [0.0, 0.0, 0.0, 1.0], atol=1e-8)
    ):
        raise ValueError("world_to_local must be homogeneous [4,4]")
    rotation = matrix[:3, :3]
    if (
        not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6)
        or not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-6)
    ):
        raise ValueError("world_to_local must contain a proper rotation")
    return matrix


def _world_corners(boxes_local: np.ndarray, world_to_local: np.ndarray) -> np.ndarray:
    boxes = np.asarray(boxes_local, dtype=np.float64)
    if boxes.ndim != 2 or boxes.shape[1] not in {6, 7}:
        raise ValueError("TR3D boxes must have shape [N,6/7]")
    if boxes.shape[1] == 6:
        boxes = np.concatenate(
            (boxes, np.zeros((len(boxes), 1), dtype=np.float64)), axis=1
        )
    local = CORNER_SIGNS[None] * (0.5 * boxes[:, None, 3:6])
    cosine, sine = np.cos(boxes[:, 6]), np.sin(boxes[:, 6])
    corners = np.empty_like(local)
    corners[:, :, 0] = local[:, :, 0] * cosine[:, None] - local[:, :, 1] * sine[:, None]
    corners[:, :, 1] = local[:, :, 0] * sine[:, None] + local[:, :, 1] * cosine[:, None]
    corners[:, :, 2] = local[:, :, 2]
    corners += boxes[:, None, :3]
    local_to_world = np.linalg.inv(world_to_local)
    corners = corners @ local_to_world[:3, :3].T + local_to_world[None, None, :3, 3]
    return np.ascontiguousarray(corners, dtype=np.float32)


def main() -> int:
    args = parser().parse_args()
    if not 0.0 <= args.score_threshold <= 1.0:
        raise ValueError("score-threshold must be in [0,1]")
    if args.max_proposals < 1:
        raise ValueError("max-proposals must be positive")
    if args.startup_timeout_s != 600.0:
        raise ValueError("formal CA-1M TR3D startup timeout must equal 600 seconds")
    sys.path.insert(0, str(args.runtime_root.resolve()))
    startup_started = time.monotonic()
    print(
        f"CA1M_TR3D_STARTUP phase=adapter_import_start timeout_s={args.startup_timeout_s:g}",
        file=sys.stderr,
        flush=True,
    )
    # If a pinned CUDA/OpenMMLab call remains in C/Python for a long cold
    # start, preserve a host-visible stack every two minutes.  This writes
    # diagnostics only; it neither interrupts nor changes model execution.
    faulthandler.dump_traceback_later(120.0, repeat=True, file=sys.stderr)
    try:
        from boxfusion.tr3d_inference import (  # type: ignore
            OfficialMMDet3DTR3DAdapter,
            SyntheticTR3DAdapter,
            _proposal_point_count,
            _validate_output,
        )

        print(
            "CA1M_TR3D_STARTUP phase=adapter_construct_start",
            file=sys.stderr,
            flush=True,
        )

        adapter = (
            SyntheticTR3DAdapter()
            if args.synthetic
            else OfficialMMDet3DTR3DAdapter(
                config_path=args.config,
                checkpoint_path=args.checkpoint,
                device=args.device,
                project_root=args.project_root,
                vendor_root=args.vendor_root,
            )
        )
        faulthandler.cancel_dump_traceback_later()
        startup_s = time.monotonic() - startup_started
        print(
            f"CA1M_TR3D_STARTUP phase=ready elapsed_s={startup_s:.3f}",
            file=sys.stderr,
            flush=True,
        )
        respond(
            {
                "status": "ready",
                "synthetic": bool(args.synthetic),
                "startup_s": startup_s,
            }
        )
    except Exception as error:
        faulthandler.cancel_dump_traceback_later()
        respond(
            {
                "status": "error",
                "error": repr(error),
                "traceback": traceback.format_exc(),
            }
        )
        return 1

    for line in sys.stdin:
        memory = None
        infer_trace_active = False
        scene_id = "unknown"

        def infer_phase(name: str) -> None:
            print(
                f"CA1M_TR3D_INFER scene={scene_id} phase={name}",
                file=sys.stderr,
                flush=True,
            )

        try:
            request = json.loads(line)
            if request.get("command") == "close":
                respond({"status": "closed"})
                return 0
            if request.get("command") != "infer":
                raise ValueError("unknown worker command")
            scene_id = str(request.get("scene_id", ""))
            infer_phase("request")
            if SCENE_RE.fullmatch(scene_id) is None:
                raise ValueError(f"invalid CA-1M scene id: {scene_id!r}")
            prefix_id = str(request.get("prefix_id", ""))
            if prefix_id != "p100_gap20":
                raise ValueError(f"invalid CA-1M terminal prefix: {prefix_id!r}")
            shape = tuple(int(value) for value in request["shape"])
            if len(shape) != 2 or shape[1] != 6 or shape[0] < 1:
                raise ValueError("shared point shape must be [N,6]")
            world_to_local = _homogeneous(request["world_to_local"])
            legacy_transform = _homogeneous(request["axis_align_matrix"])
            if not np.array_equal(world_to_local, legacy_transform):
                raise ValueError("world/local transform aliases disagree")
            memory = shared_memory.SharedMemory(name=str(request["shared_memory"]))
            resource_tracker.unregister(memory._name, "shared_memory")
            points = np.array(
                np.ndarray(shape, dtype=np.float32, buffer=memory.buf), copy=True
            )
            source_points_sha256 = hashlib.sha256(
                points.tobytes(order="C")
            ).hexdigest()
            if request.get("source_points_sha256") != source_points_sha256:
                raise ValueError("cross-process source point hash mismatch")
            faulthandler.dump_traceback_later(30.0, repeat=True, file=sys.stderr)
            infer_trace_active = True
            output = adapter.infer(
                points,
                world_to_local,
                phase_callback=infer_phase,
            )
            boxes_all, scores_all, labels_all = _validate_output(output)
            selected = np.flatnonzero(scores_all >= args.score_threshold)
            selected = selected[
                np.argsort(-scores_all[selected], kind="stable")[: args.max_proposals]
            ]
            boxes = boxes_all[selected]
            if boxes.shape[1] == 6:
                boxes = np.concatenate(
                    (boxes, np.zeros((len(boxes), 1), dtype=np.float32)), axis=1
                )
            boxes = np.ascontiguousarray(boxes, dtype=np.float32)
            scores = scores_all[selected]
            labels = labels_all[selected]
            if np.any(labels != 0):
                raise ValueError("official TR3D output is not class-agnostic")
            points_local = (
                points[:, :3] @ world_to_local[:3, :3].T
                + world_to_local[None, :3, 3]
            )
            point_counts = _proposal_point_count(points_local, boxes)
            corners_world = _world_corners(boxes, world_to_local)
            infer_phase("response")
            respond(
                {
                    "status": "ok",
                    "scene_id": scene_id,
                    "prefix_id": prefix_id,
                    "adapter_mode": "synthetic" if args.synthetic else "genuine",
                    "corners_world": corners_world.tolist(),
                    "boxes_local": boxes.tolist(),
                    "scores": scores.tolist(),
                    "labels": labels.tolist(),
                    "point_counts": np.asarray(point_counts, dtype=np.int64).tolist(),
                    "model_runtime_s": float(output.runtime_s),
                    "source_points_sha256": source_points_sha256,
                }
            )
        except Exception as error:
            infer_phase("response")
            respond(
                {
                    "status": "error",
                    "error": repr(error),
                    "traceback": traceback.format_exc(),
                }
            )
        finally:
            if infer_trace_active:
                faulthandler.cancel_dump_traceback_later()
            if memory is not None:
                memory.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
