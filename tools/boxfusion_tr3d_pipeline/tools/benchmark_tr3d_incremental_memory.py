#!/usr/bin/env python3
"""Benchmark causal voxel-memory updates on a real ScanNet stream."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from boxfusion.tr3d_incremental_online import IncrementalTR3DConfig, IncrementalTR3DObserver, TR3DProviderResult
from tools.tr3d_data import discover_frame_bundle, load_matrix, valid_pose


class EmptyProvider:
    def infer(self, **_kwargs):
        return TR3DProviderResult(
            np.empty((0, 8, 3), dtype=np.float32),
            np.empty((0,), dtype=np.float32), 0.0,
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frame-root", type=Path, required=True)
    parser.add_argument("--scene-id", required=True)
    parser.add_argument("--frame-gap", type=int, default=25)
    parser.add_argument("--pixel-stride", type=int, default=4)
    parser.add_argument("--voxel-size", type=float, default=0.03)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    bundle = discover_frame_bundle(args.frame_root, args.scene_id)
    intrinsic = load_matrix(bundle.intrinsic_depth)[:3, :3]
    extrinsic = load_matrix(bundle.extrinsic_depth)
    common = sorted(set(bundle.depth) & set(bundle.color) & set(bundle.pose))
    frame_ids = common[:: args.frame_gap]
    config = IncrementalTR3DConfig(
        pixel_stride=args.pixel_stride, voxel_size_m=args.voxel_size,
        warmup_keyframes=3, inference_interval_keyframes=5,
    )
    observer = IncrementalTR3DObserver(config, EmptyProvider())
    observer.reset_scene(args.scene_id, np.eye(4))
    wall = []
    accepted = []
    for frame_id in frame_ids:
        pose = valid_pose(bundle.pose[frame_id])
        if pose is None:
            continue
        depth = np.asarray(Image.open(bundle.depth[frame_id]), dtype=np.float32) / 1000.0
        image = np.asarray(Image.open(bundle.color[frame_id]).convert("RGB"))
        started = time.perf_counter()
        observer.process_keyframe(
            scene_id=args.scene_id, depth=depth, image=image,
            intrinsics=intrinsic, camera_to_world=pose @ extrinsic,
            source_timestamp=frame_id,
        )
        wall.append((time.perf_counter() - started) * 1000.0)
        accepted.append(frame_id)
    summary = observer.finalize()
    report = {
        "schema": "boxfusion.tr3d_incremental_memory_benchmark.v1",
        "scene_id": args.scene_id, "frames": len(accepted),
        "frame_ids": accepted, "memory_voxels": summary["memory_voxels"],
        "update_ms": {
            "mean": float(np.mean(wall)), "median": float(np.median(wall)),
            "p95": float(np.quantile(wall, 0.95)), "max": float(np.max(wall)),
        },
        "provider_calls_scheduled": summary["provider_calls"],
        "provider_is_empty_benchmark": True,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
