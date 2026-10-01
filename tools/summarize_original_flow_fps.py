#!/usr/bin/env python3
"""Summarize released BoxFusion versus current full original-flow FPS."""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


def describe(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)
    return {
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "p25": ordered[int(0.25 * (len(ordered) - 1))],
        "minimum": ordered[0],
        "maximum": ordered[-1],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-list", type=Path, required=True)
    parser.add_argument("--receipt-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    scenes = [
        line.strip()
        for line in args.scene_list.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    rows = []
    for index, scene in enumerate(scenes):
        official_path = args.receipt_root / "official" / scene / "runtime.json"
        full_path = args.receipt_root / "full" / scene / "runtime.json"
        official = json.loads(official_path.read_text(encoding="utf-8"))
        full = json.loads(full_path.read_text(encoding="utf-8"))
        if official.get("scene") != scene or full.get("scene") != scene:
            raise RuntimeError(f"scene mismatch for {scene}")
        if not full.get("original_frame_loop") or full.get("children_enabled") is not False:
            raise RuntimeError(f"invalid full receipt for {scene}")
        if not full.get("cuda_synchronized"):
            raise RuntimeError(f"unsynchronized full receipt for {scene}")
        oframes = int(official["consumed_raw_frames"])
        fframes = int(full["raw_frames"])
        if oframes != fframes:
            raise RuntimeError(f"frame mismatch for {scene}: {oframes} != {fframes}")
        oseconds = float(official["run_seconds"])
        fseconds = float(full["cost_seconds"])
        rows.append({
            "scene": scene,
            "index": index,
            "first_arm": "official" if index % 2 == 0 else "full",
            "raw_frames": oframes,
            "official_seconds": oseconds,
            "full_seconds": fseconds,
            "official_fps": oframes / oseconds,
            "full_fps": fframes / fseconds,
            "retention": oseconds / fseconds,
            "official_peak_bytes": int(official["peak_allocated_bytes"]),
            "full_peak_bytes": int(full["peak_allocated_bytes"]),
        })
    frames = sum(row["raw_frames"] for row in rows)
    official_seconds = sum(row["official_seconds"] for row in rows)
    full_seconds = sum(row["full_seconds"] for row in rows)
    official_first = [r["retention"] for r in rows if r["first_arm"] == "official"]
    full_first = [r["retention"] for r in rows if r["first_arm"] == "full"]
    order_medians = {
        "official_first": statistics.median(official_first),
        "full_first": statistics.median(full_first),
    }
    order_gap = abs(order_medians["official_first"] - order_medians["full_first"]) / statistics.median([*official_first, *full_first])
    summary = {
        "schema": "boxfusion.original_flow_official_vs_full_fps.v1",
        "scene_count": len(rows),
        "protocol": {
            "scene_set": "ScanNet official100",
            "gpu_count_per_arm": 1,
            "scene_order": "alternating paired arms",
            "scene_cache_prewarmed": False,
            "per_raw_frame_preparation": True,
            "cuda_synchronized_at_timing_boundaries": True,
            "aggregation": "sum(raw_frames) / sum(stream_seconds)",
            "model_initialization": "excluded",
            "visualization": False,
            "proposal_replay": False,
            "final_output_serialization": "included",
        },
        "aggregate": {
            "raw_frames": frames,
            "official_seconds": official_seconds,
            "full_seconds": full_seconds,
            "official_fps": frames / official_seconds,
            "full_fps": frames / full_seconds,
            "throughput_retention": official_seconds / full_seconds,
            "official_scene_fps": describe([r["official_fps"] for r in rows]),
            "full_scene_fps": describe([r["full_fps"] for r in rows]),
            "paired_retention": describe([r["retention"] for r in rows]),
            "order_conditioned_median_retention": order_medians,
            "relative_order_gap": order_gap,
            "order_check_pass": order_gap <= 0.05,
            "official_peak_allocated_bytes_max": max(r["official_peak_bytes"] for r in rows),
            "full_peak_allocated_bytes_max": max(r["full_peak_bytes"] for r in rows),
        },
        "per_scene": rows,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    a = summary["aggregate"]
    status = "PASS" if a["order_check_pass"] else "FAIL"
    report = f"""# Original-flow single-GPU FPS benchmark

The released BoxFusion snapshot and the current strict-causal no-child full
model use the same ScanNet official100 scene list, keyframe gap, single RTX
3090, per-raw-frame packaging/device transfer/preprocessing, CUDA-synchronized
timing boundaries, and alternating paired order. No explicit cache prewarm is
performed. Model loading is excluded; stream input, inference, mapping, online
M1/M2 updates, and final serialization are included.

| Arm | Frame-weighted FPS | Mean scene FPS | Median | P25 | Minimum |
|---|---:|---:|---:|---:|---:|
| Released BoxFusion | {a['official_fps']:.4f} | {a['official_scene_fps']['mean']:.4f} | {a['official_scene_fps']['median']:.4f} | {a['official_scene_fps']['p25']:.4f} | {a['official_scene_fps']['minimum']:.4f} |
| Full online M1-P+M1-A+M2 (no child) | {a['full_fps']:.4f} | {a['full_scene_fps']['mean']:.4f} | {a['full_scene_fps']['median']:.4f} | {a['full_scene_fps']['p25']:.4f} | {a['full_scene_fps']['minimum']:.4f} |

Frame-weighted throughput retention: **{100*a['throughput_retention']:.2f}%**.
Median paired-scene retention: **{100*a['paired_retention']['median']:.2f}%**.

Order-bias check: **{status}** (relative gap {100*a['relative_order_gap']:.2f}%;
official-first median {100*a['order_conditioned_median_retention']['official_first']:.2f}%,
full-first median {100*a['order_conditioned_median_retention']['full_first']:.2f}%).
Paper-reported ~20 FPS is contextual only; the table reports the local matched
reproduction on this machine.
"""
    (args.output / "REPORT.md").write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
