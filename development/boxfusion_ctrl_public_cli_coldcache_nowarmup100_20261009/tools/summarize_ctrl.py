#!/usr/bin/env python3
"""Summarize the completed BoxFusion-Ctrl efficiency arm."""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-list", type=Path, required=True)
    parser.add_argument("--receipt-root", type=Path, required=True)
    parser.add_argument("--arm", choices=("ctrl",), default="ctrl")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    scenes = [
        line.strip().split()[0]
        for line in args.scene_list.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if len(scenes) != 100:
        raise ValueError(f"expected 100 scenes, found {len(scenes)}")
    rows = []
    for scene in scenes:
        path = args.receipt_root / args.arm / scene / "runtime.json"
        row = json.loads(path.read_text(encoding="utf-8"))
        if row["arm"] != args.arm or row["scene"] != scene:
            raise ValueError(f"receipt identity mismatch: {path}")
        rows.append(row)
    frames = sum(int(row["raw_frames"]) for row in rows)
    seconds = sum(float(row["cost_seconds"]) for row in rows)
    result = {
        "arm": args.arm,
        "scene_count": len(rows),
        "raw_frames": frames,
        "total_seconds": seconds,
        "frame_weighted_fps": frames / seconds,
        "mean_scene_fps": statistics.fmean(float(row["fps"]) for row in rows),
        "median_scene_fps": statistics.median(float(row["fps"]) for row in rows),
        "peak_allocated_bytes_max": max(
            int(row["peak_allocated_bytes"]) for row in rows
        ),
    }
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    report = args.output.with_name("REPORT.md")
    report.write_text(
        "# BoxFusion-Ctrl public-CLI cold-cache FPS\n\n"
        "Route: active Boxer lifting + Reliable-View Top-K=3 + PFO; "
        "online candidate map, PLR, CALR and MVSR disabled.\n\n"
        "Protocol: ScanNet-100, one RTX 3090, gap=25, native score "
        "threshold=0.5, verified cold scene files, no synthetic warm-up, "
        "public demo `time.time()` boundary.\n\n"
        "| Scenes | Raw frames | Timed seconds | Frame-weighted FPS | "
        "Peak allocated memory |\n"
        "|---:|---:|---:|---:|---:|\n"
        f"| {result['scene_count']} | {result['raw_frames']} | "
        f"{result['total_seconds']:.3f} | "
        f"{result['frame_weighted_fps']:.4f} | "
        f"{result['peak_allocated_bytes_max'] / (1024 ** 3):.3f} GiB |\n",
        encoding="utf-8",
    )
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
