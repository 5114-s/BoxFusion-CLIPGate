#!/usr/bin/env python3
"""Summarize synchronized, cache-prewarmed control/full FPS receipts."""
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
        arms = {}
        for arm in ("control", "full"):
            path = args.receipt_root / arm / scene / "runtime.json"
            value = json.loads(path.read_text(encoding="utf-8"))
            required = {
                "arm": arm,
                "scene": scene,
                "gpu_count": 1,
                "cuda_synchronized": True,
                "scene_cache_prewarmed": True,
                "final_output_serialization_included": True,
                "model_initialization_excluded": True,
                "visualization": False,
                "proposal_replay": False,
            }
            for key, expected in required.items():
                if value.get(key) != expected:
                    raise RuntimeError(
                        f"{path}: {key}={value.get(key)!r}, expected {expected!r}"
                    )
            arms[arm] = value
        if arms["control"]["raw_frames"] != arms["full"]["raw_frames"]:
            raise RuntimeError(f"frame mismatch for {scene}")
        control_seconds = float(arms["control"]["cost_seconds"])
        full_seconds = float(arms["full"]["cost_seconds"])
        rows.append({
            "scene": scene,
            "index": index,
            "first_arm": "control" if index % 2 == 0 else "full",
            "raw_frames": int(arms["control"]["raw_frames"]),
            "control_seconds": control_seconds,
            "full_seconds": full_seconds,
            "control_fps": float(arms["control"]["fps"]),
            "full_fps": float(arms["full"]["fps"]),
            "retention": control_seconds / full_seconds,
            "control_peak_bytes": int(arms["control"]["peak_allocated_bytes"]),
            "full_peak_bytes": int(arms["full"]["peak_allocated_bytes"]),
        })

    frames = sum(row["raw_frames"] for row in rows)
    control_seconds = sum(row["control_seconds"] for row in rows)
    full_seconds = sum(row["full_seconds"] for row in rows)
    control_first = [row["retention"] for row in rows if row["first_arm"] == "control"]
    full_first = [row["retention"] for row in rows if row["first_arm"] == "full"]
    order_medians = {
        "control_first": statistics.median(control_first),
        "full_first": statistics.median(full_first),
    }
    order_gap = abs(
        order_medians["control_first"] - order_medians["full_first"]
    ) / statistics.median([*control_first, *full_first])
    summary = {
        "schema": "boxfusion.current_code_single_gpu_matched_fps.v1",
        "scene_count": len(rows),
        "protocol": {
            "gpu_count_per_arm": 1,
            "same_current_code": True,
            "only_full_enables_online_candidate_map": True,
            "scene_order": "alternating paired arms",
            "scene_cache_prewarmed_before_each_arm": True,
            "cuda_synchronized_at_timing_boundaries": True,
            "aggregation": "sum(raw_frames) / sum(stream_seconds)",
            "model_initialization": "excluded",
            "visualization": False,
            "proposal_replay": False,
            "per_keyframe_disk_snapshots": False,
            "final_output_serialization": "included",
        },
        "aggregate": {
            "raw_frames": frames,
            "control_seconds": control_seconds,
            "full_seconds": full_seconds,
            "control_fps": frames / control_seconds,
            "full_fps": frames / full_seconds,
            "throughput_retention": control_seconds / full_seconds,
            "control_scene_fps": describe([row["control_fps"] for row in rows]),
            "full_scene_fps": describe([row["full_fps"] for row in rows]),
            "paired_retention": describe([row["retention"] for row in rows]),
            "order_conditioned_median_retention": order_medians,
            "relative_order_gap": order_gap,
            "order_check_pass": order_gap <= 0.05,
            "control_peak_allocated_bytes_max": max(
                row["control_peak_bytes"] for row in rows
            ),
            "full_peak_allocated_bytes_max": max(
                row["full_peak_bytes"] for row in rows
            ),
        },
        "per_scene": rows,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    a = summary["aggregate"]
    status = "PASS" if a["order_check_pass"] else "FAIL"
    report = f"""# Current-code single-GPU matched FPS benchmark

The control and full method use the same current native code, strengthened
Top-K+Boxer base, ScanNet official100 scenes, and one RTX GPU. The only method
difference is whether the online candidate map (M1-P, M1-A, and M2) is enabled.
Each arm prewarms the same scene files outside timing. CUDA is synchronized at
both timing boundaries. Model loading is excluded; live RGB-D input,
inference, native mapping, online updates, and final serialization are included.

| Arm | Frame-weighted FPS | Mean scene FPS | Median | P25 | Minimum |
|---|---:|---:|---:|---:|---:|
| Current strengthened base | {a['control_fps']:.4f} | {a['control_scene_fps']['mean']:.4f} | {a['control_scene_fps']['median']:.4f} | {a['control_scene_fps']['p25']:.4f} | {a['control_scene_fps']['minimum']:.4f} |
| Full M1-P+M1-A+M2 | {a['full_fps']:.4f} | {a['full_scene_fps']['mean']:.4f} | {a['full_scene_fps']['median']:.4f} | {a['full_scene_fps']['p25']:.4f} | {a['full_scene_fps']['minimum']:.4f} |

Frame-weighted throughput retention: **{100*a['throughput_retention']:.2f}%**.
Median paired-scene retention: **{100*a['paired_retention']['median']:.2f}%**.

Order-bias check: **{status}** (relative gap {100*a['relative_order_gap']:.2f}%;
control-first median {100*a['order_conditioned_median_retention']['control_first']:.2f}%,
full-first median {100*a['order_conditioned_median_retention']['full_first']:.2f}%).
The result is suitable for a paired overhead claim only when this check passes.
"""
    (args.output / "REPORT.md").write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
