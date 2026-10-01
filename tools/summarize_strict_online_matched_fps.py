#!/usr/bin/env python3
"""Summarize paired official BoxFusion and strict-online runtime receipts."""
from __future__ import annotations

import argparse
import json
import re
import statistics
from pathlib import Path


RECEIPT = re.compile(
    r"Online runtime receipt \| raw_frames=(\d+) "
    r"cost_seconds=([0-9.]+) fps=([0-9.]+)"
)


def quantiles(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)
    return {
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "p25": ordered[int(0.25 * (len(ordered) - 1))],
        "minimum": ordered[0],
        "maximum": ordered[-1],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene-list", type=Path, required=True)
    parser.add_argument("--official-root", type=Path, required=True)
    parser.add_argument("--method-logs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    scenes = [line.strip() for line in args.scene_list.read_text().splitlines()
              if line.strip() and not line.startswith("#")]
    rows = []
    for scene in scenes:
        official = json.loads(
            (args.official_root / scene / "runtime.json").read_text()
        )
        text = (args.method_logs / f"{scene}.log").read_text(errors="replace")
        matches = RECEIPT.findall(text)
        if len(matches) != 1:
            raise RuntimeError(f"expected one method receipt for {scene}")
        frames, seconds, fps = matches[0]
        frames = int(frames)
        seconds = float(seconds)
        fps = float(fps)
        if frames != int(official["consumed_raw_frames"]):
            raise RuntimeError(f"frame mismatch for {scene}")
        rows.append({
            "scene": scene,
            "frames": frames,
            "official_seconds": float(official["run_seconds"]),
            "official_fps": float(official["raw_fps"]),
            "method_seconds": seconds,
            "method_fps": fps,
            "retention": fps / float(official["raw_fps"]),
        })

    frames = sum(row["frames"] for row in rows)
    official_seconds = sum(row["official_seconds"] for row in rows)
    method_seconds = sum(row["method_seconds"] for row in rows)
    summary = {
        "schema": "boxfusion.strict_online_single_gpu_matched_fps.v1",
        "scene_count": len(rows),
        "protocol": {
            "gpu_count_per_arm": 1,
            "scene_order": "alternating paired arms",
            "aggregation": "sum(raw_frames) / sum(stream_seconds)",
            "model_initialization": "excluded",
            "visualization": False,
            "proposal_replay": False,
            "per_keyframe_disk_snapshots": False,
        },
        "aggregate": {
            "raw_frames": frames,
            "official_seconds": official_seconds,
            "method_seconds": method_seconds,
            "official_fps": frames / official_seconds,
            "method_fps": frames / method_seconds,
            "throughput_retention": official_seconds / method_seconds,
            "official_scene_fps": quantiles(
                [row["official_fps"] for row in rows]
            ),
            "method_scene_fps": quantiles(
                [row["method_fps"] for row in rows]
            ),
            "paired_retention": quantiles(
                [row["retention"] for row in rows]
            ),
        },
        "per_scene": rows,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    a = summary["aggregate"]
    (args.output / "REPORT.md").write_text(f"""# Single-GPU matched FPS benchmark

Frozen official BoxFusion and the strict-causal no-child method were run
serially and in alternating order on the same RTX GPU and the same 100 scenes.
Both arms exclude model initialization and evaluation, disable visualization
and proposal replay, and include live RGB-D I/O, inference, mapping, final
output serialization, and their respective online processing.

| Arm | Frame-weighted FPS | Mean scene FPS | Median | P25 | Minimum |
|---|---:|---:|---:|---:|---:|
| Official BoxFusion | {a['official_fps']:.4f} | {a['official_scene_fps']['mean']:.4f} | {a['official_scene_fps']['median']:.4f} | {a['official_scene_fps']['p25']:.4f} | {a['official_scene_fps']['minimum']:.4f} |
| Strict-causal M1-P+A+M2 | {a['method_fps']:.4f} | {a['method_scene_fps']['mean']:.4f} | {a['method_scene_fps']['median']:.4f} | {a['method_scene_fps']['p25']:.4f} | {a['method_scene_fps']['minimum']:.4f} |

Frame-weighted throughput retention: **{100*a['throughput_retention']:.2f}%**.
Median paired-scene retention: **{100*a['paired_retention']['median']:.2f}%**.
""")


if __name__ == "__main__":
    main()
