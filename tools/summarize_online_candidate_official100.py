#!/usr/bin/env python3
"""Validate and summarize the strict-causal online official100 run."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
from pathlib import Path


RECEIPT = re.compile(
    r"Online runtime receipt \| raw_frames=(\d+) "
    r"cost_seconds=([0-9.]+) fps=([0-9.]+)"
)


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-list", type=Path, required=True)
    parser.add_argument("--logs", type=Path, required=True)
    parser.add_argument("--diagnostics", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    scenes = [
        line.strip()
        for line in args.scene_list.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if len(scenes) != 100:
        raise RuntimeError(f"expected 100 scenes, found {len(scenes)}")

    rows: list[dict[str, object]] = []
    for scene in scenes:
        log_path = args.logs / f"{scene}.log"
        diagnostic_path = args.diagnostics / f"{scene}.json"
        prediction_path = args.predictions / f"{scene}_boxes.pkl"
        if not log_path.is_file() or not diagnostic_path.is_file():
            raise RuntimeError(f"missing runtime artifact for {scene}")
        if not prediction_path.is_file() or prediction_path.stat().st_size == 0:
            raise RuntimeError(f"missing prediction for {scene}")
        text = log_path.read_text(encoding="utf-8", errors="replace")
        matches = RECEIPT.findall(text)
        if len(matches) != 1:
            raise RuntimeError(f"expected one exact runtime receipt for {scene}")
        raw_frames = int(matches[0][0])
        seconds = float(matches[0][1])
        fps = float(matches[0][2])
        if raw_frames <= 0 or seconds <= 0 or not math.isclose(
            fps, raw_frames / seconds, rel_tol=1e-8, abs_tol=1e-8
        ):
            raise RuntimeError(f"invalid runtime receipt for {scene}")

        diagnostic = json.loads(diagnostic_path.read_text(encoding="utf-8"))
        required = {
            "strictly_causal": True,
            "online_incremental": True,
            "scene_end_inference": False,
            "asynchronous": True,
            "provider_preloaded_before_stream": True,
        }
        for key, expected in required.items():
            if diagnostic.get(key) is not expected:
                raise RuntimeError(f"{scene}: {key} != {expected!r}")
        if diagnostic["state"]["m1p"].get("use_children") is not False:
            raise RuntimeError(f"{scene}: child evidence was not disabled")
        if int(diagnostic.get("pending_at_close", -1)) < 0:
            raise RuntimeError(f"{scene}: invalid close state")

        end_to_end = diagnostic["keyframe_end_to_end"]
        rows.append(
            {
                "scene": scene,
                "raw_frames": raw_frames,
                "cost_seconds": seconds,
                "fps": fps,
                "keyframes": int(end_to_end["count"]),
                "keyframe_p50_ms": float(end_to_end["p50_ms"]),
                "keyframe_p95_ms": float(end_to_end["p95_ms"]),
                "keyframe_max_ms": float(end_to_end["max_ms"]),
                "deadline_misses": int(diagnostic["deadline_misses"]),
                "queue_max": int(diagnostic["max_queue_depth"]),
                "close_drain_ms": float(diagnostic["close_drain_ms"]),
            }
        )

    total_frames = sum(int(row["raw_frames"]) for row in rows)
    total_seconds = sum(float(row["cost_seconds"]) for row in rows)
    scene_fps = [float(row["fps"]) for row in rows]
    total_keyframes = sum(int(row["keyframes"]) for row in rows)
    deadline_misses = sum(int(row["deadline_misses"]) for row in rows)
    summary = {
        "schema": "boxfusion.strict_causal_online_official100.v1",
        "protocol": {
            "scene_count": 100,
            "aggregation": "sum(raw_frames) / sum(cost_seconds)",
            "gpu_layout": "native cuda:0; frozen evidence provider cuda:1",
            "m1p_evidence": "lifted post-NMS proposals only; children disabled",
            "model_initialization": "excluded before stream clock",
            "included": [
                "raw-frame RGB-D I/O",
                "native candidate generation, association, and fusion",
                "fresh frozen WeDetect-Uni and shared Boxer evidence",
                "strict-causal M1-P, M1-A, and M2 updates",
                "per-keyframe online-map atomic serialization",
                "queue drain of already submitted keyframes",
            ],
            "scene_end_inference": False,
        },
        "totals": {
            "raw_frames": total_frames,
            "cost_seconds": total_seconds,
            "frame_weighted_fps": total_frames / total_seconds,
            "keyframes": total_keyframes,
            "deadline_misses": deadline_misses,
            "deadline_miss_rate": deadline_misses / max(total_keyframes, 1),
        },
        "scene_fps": {
            "mean": statistics.fmean(scene_fps),
            "median": statistics.median(scene_fps),
            "p25": percentile(scene_fps, 0.25),
            "minimum": min(scene_fps),
            "maximum": max(scene_fps),
        },
        "per_scene": rows,
    }

    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    with (args.output / "scenes.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    totals = summary["totals"]
    distribution = summary["scene_fps"]
    report = f"""# Strict-causal online M1-P + M1-A + M2 official100 runtime

All 100 scenes passed the causal runtime checks: incremental prefix updates,
no future-frame access, no scene-end inference, and no child evidence in M1-P.

| Metric | Result |
|---|---:|
| Total raw frames | {totals['raw_frames']} |
| Total timed seconds | {totals['cost_seconds']:.6f} |
| **Frame-weighted FPS** | **{totals['frame_weighted_fps']:.4f}** |
| Mean scene FPS | {distribution['mean']:.4f} |
| Median scene FPS | {distribution['median']:.4f} |
| P25 scene FPS | {distribution['p25']:.4f} |
| Minimum scene FPS | {distribution['minimum']:.4f} |
| Keyframes | {totals['keyframes']} |
| Deadline misses | {totals['deadline_misses']} ({100.0 * totals['deadline_miss_rate']:.2f}%) |

The primary throughput is `sum(raw frames) / sum(timed seconds)`. The method
uses two RTX 3090 GPUs: native BoxFusion on GPU 0 and the frozen evidence
provider on GPU 1. Model/checkpoint initialization is excluded; all online
method updates and per-keyframe output serialization are included.
"""
    (args.output / "REPORT.md").write_text(report, encoding="utf-8")
    print(json.dumps(summary["totals"], ensure_ascii=False, indent=2))
    print(json.dumps(summary["scene_fps"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
