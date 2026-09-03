#!/usr/bin/env python3
"""Persistent official-TR3D worker for the online BoxFusion process."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
from multiprocessing import resource_tracker, shared_memory
from pathlib import Path
import sys
import traceback

import numpy as np


PREFIX = "BOXFUSION_TR3D_RESPONSE "


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
    value.add_argument("--score-threshold", type=float, default=0.01)
    value.add_argument("--max-proposals", type=int, default=256)
    value.add_argument("--synthetic", action="store_true")
    return value


def main() -> int:
    args = parser().parse_args()
    # Force the official runtime package to win over this script's repository.
    sys.path.insert(0, str(args.runtime_root.resolve()))
    try:
        from boxfusion.tr3d_inference import (
            OfficialMMDet3DTR3DAdapter, SyntheticTR3DAdapter,
            make_tr3d_residual_cache_from_aligned, _proposal_point_count,
        )
        if args.synthetic:
            adapter = SyntheticTR3DAdapter()
        else:
            adapter = OfficialMMDet3DTR3DAdapter(
                config_path=args.config, checkpoint_path=args.checkpoint,
                device=args.device, project_root=args.project_root,
                vendor_root=args.vendor_root,
            )
        respond({"status": "ready", "synthetic": bool(args.synthetic)})
    except Exception as error:
        respond({"status": "error", "error": repr(error), "traceback": traceback.format_exc()})
        return 1

    for line in sys.stdin:
        memory = None
        try:
            request = json.loads(line)
            if request.get("command") == "close":
                respond({"status": "closed"})
                return 0
            if request.get("command") != "infer":
                raise ValueError("unknown worker command")
            shape = tuple(int(value) for value in request["shape"])
            if len(shape) != 2 or shape[1] != 6 or shape[0] < 1:
                raise ValueError("shared point shape must be [N,6]")
            memory = shared_memory.SharedMemory(name=str(request["shared_memory"]))
            # The creator process owns unlink.  Prevent this attached worker's
            # resource tracker from racing the creator during shutdown.
            resource_tracker.unregister(memory._name, "shared_memory")
            points = np.array(np.ndarray(shape, dtype=np.float32, buffer=memory.buf), copy=True)
            axis = np.asarray(request["axis_align_matrix"], dtype=np.float64)
            output = adapter.infer(points, axis)
            selected = np.flatnonzero(output.scores_3d >= args.score_threshold)
            selected = selected[np.argsort(-output.scores_3d[selected], kind="stable")[: args.max_proposals]]
            boxes = output.boxes_aligned[selected]
            scores = output.scores_3d[selected]
            labels = output.labels_3d[selected]
            points_aligned = (
                points[:, :3] @ axis[:3, :3].T
                + axis[None, :3, 3]
            )
            point_counts = _proposal_point_count(points_aligned, boxes)
            digest = hashlib.sha256(points.tobytes(order="C")).hexdigest()
            cache = make_tr3d_residual_cache_from_aligned(
                scene_id=str(request["scene_id"]), prefix_id=str(request["prefix_id"]),
                prefix_fraction=1.0, boxes_aligned=boxes, scores_3d=scores,
                labels_3d=labels, unaligned_to_aligned=axis,
                checkpoint_sha256="0" * 64, config_sha256="0" * 64,
                source_scene_sha256=digest, runtime_s=output.runtime_s,
                num_input_points=len(points), point_count=point_counts,
            )
            respond({
                "status": "ok", "corners_world": cache.corners_world.tolist(),
                "scores": cache.scores_3d.tolist(),
                "point_counts": cache.point_count.tolist(),
                "model_runtime_s": float(output.runtime_s),
            })
        except Exception as error:
            respond({"status": "error", "error": repr(error), "traceback": traceback.format_exc()})
        finally:
            if memory is not None:
                memory.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
