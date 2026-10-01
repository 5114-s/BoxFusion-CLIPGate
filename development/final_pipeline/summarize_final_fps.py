#!/usr/bin/env python3
"""Summarize matched single-GPU FPS and online-state latency diagnostics."""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-list", type=Path, required=True)
    parser.add_argument("--receipt-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    scenes = [r.strip() for r in args.scene_list.read_text().splitlines()
              if r.strip() and not r.lstrip().startswith("#")]
    rows, diagnostics = [], []
    for index, scene in enumerate(scenes):
        arms = {
            arm: json.loads((args.receipt_root / arm / scene / "runtime.json").read_text())
            for arm in ("control", "full")
        }
        if arms["control"]["raw_frames"] != arms["full"]["raw_frames"]:
            raise RuntimeError(f"frame mismatch: {scene}")
        rows.append({
            "scene": scene, "index": index,
            "frames": int(arms["control"]["raw_frames"]),
            "control_seconds": float(arms["control"]["cost_seconds"]),
            "full_seconds": float(arms["full"]["cost_seconds"]),
            "control_peak": int(arms["control"]["peak_allocated_bytes"]),
            "full_peak": int(arms["full"]["peak_allocated_bytes"]),
        })
        path = args.receipt_root / "full" / scene / "full_diagnostics" / f"{scene}.json"
        d = json.loads(path.read_text())
        if not d["state"].get("selected_calr_v1", False):
            raise RuntimeError(f"not final CALR-v1 runtime: {scene}")
        diagnostics.append(d)

    frames = sum(r["frames"] for r in rows)
    control_seconds = sum(r["control_seconds"] for r in rows)
    full_seconds = sum(r["full_seconds"] for r in rows)
    keyframes = sum(int(d["keyframe_end_to_end"]["count"]) for d in diagnostics)
    deadline_misses = sum(int(d["deadline_misses"]) for d in diagnostics)

    def weighted_state(path):
        count = 0; total = 0.0; scene_p95 = []
        for d in diagnostics:
            value = d["state"]
            for key in path:
                value = value[key]
            n = int(value["count"]); count += n; total += n * float(value["mean_ms"])
            scene_p95.append(float(value["p95_ms"]))
        return {"count": count, "mean_ms": total / count, "scene_p95_median_ms": statistics.median(scene_p95), "scene_p95_max_ms": max(scene_p95)}

    latency = {
        "plr": weighted_state(("plr_timing_and_stability", "current")),
        "calr": weighted_state(("calr_timing",)),
        "mvsr": weighted_state(("mvsr_timing",)),
        "all_state": weighted_state(("timing",)),
    }
    e2e_mean = sum(
        int(d["keyframe_end_to_end"]["count"]) * float(d["keyframe_end_to_end"]["mean_ms"])
        for d in diagnostics
    ) / keyframes
    result = {
        "schema": "boxfusion.recar3d.final_runtime.v1",
        "scene_count": len(scenes),
        "raw_frames": frames,
        "control_fps": frames / control_seconds,
        "full_fps": frames / full_seconds,
        "throughput_retention": control_seconds / full_seconds,
        "peak_gpu_bytes": {"control": max(r["control_peak"] for r in rows), "full": max(r["full_peak"] for r in rows)},
        "keyframes": keyframes,
        "keyframe_end_to_end_mean_ms": e2e_mean,
        "scene_p95_keyframe_latency_median_ms": statistics.median(float(d["keyframe_end_to_end"]["p95_ms"]) for d in diagnostics),
        "scene_p95_keyframe_latency_max_ms": max(float(d["keyframe_end_to_end"]["p95_ms"]) for d in diagnostics),
        "deadline_misses": deadline_misses,
        "deadline_miss_fraction": deadline_misses / keyframes,
        "state_latency": latency,
        "additional_trainable_parameters": 0,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    gib = 1024 ** 3
    lines = [
        "# Final ReCaR-3D single-GPU matched runtime", "",
        "Control and Full use the same final-runtime source and alternating paired scene order.", "",
        "| Arm | Frame-weighted FPS | Peak allocated GPU memory (GiB) |",
        "|---|---:|---:|",
        f"| Strengthened native base | {result['control_fps']:.4f} | {result['peak_gpu_bytes']['control']/gib:.3f} |",
        f"| ReCaR-3D | {result['full_fps']:.4f} | {result['peak_gpu_bytes']['full']/gib:.3f} |",
        "",
        f"Throughput retention: **{100*result['throughput_retention']:.2f}%**; deadline misses: **{deadline_misses}/{keyframes}**.",
        "",
        "| Online state | Mean ms/keyframe | Median scene p95 | Maximum scene p95 |",
        "|---|---:|---:|---:|",
    ]
    for name in ("plr", "calr", "mvsr", "all_state"):
        row = latency[name]
        lines.append(f"| {name.upper()} | {row['mean_ms']:.3f} | {row['scene_p95_median_ms']:.3f} | {row['scene_p95_max_ms']:.3f} |")
    (args.output / "REPORT.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
