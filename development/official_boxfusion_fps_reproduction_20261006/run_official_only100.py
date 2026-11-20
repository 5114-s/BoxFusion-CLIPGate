#!/usr/bin/env python3
"""Repeat the released BoxFusion ScanNet-100 throughput measurement.

This runner deliberately measures only the frozen original BoxFusion control.
It uses the released ScanNet configuration, disables the interactive Rerun
viewer, excludes model construction, and does not explicitly prewarm scene
files.  Every scene runs in a fresh process so GPU peak-memory and timing
receipts remain independently auditable and the run can safely resume.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[2]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def valid_receipt(path: Path, scene: str) -> bool:
    if not path.is_file():
        return False
    try:
        receipt = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return (
        receipt.get("scene") == scene
        and int(receipt.get("consumed_raw_frames", 0)) > 0
        and float(receipt.get("run_seconds", 0.0)) > 0.0
    )


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("empty percentile input")
    position = q * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def summarize(output: Path, scenes: list[str]) -> dict:
    rows = []
    for scene in scenes:
        receipt_path = output / "scenes" / scene / "runtime.json"
        if not valid_receipt(receipt_path, scene):
            continue
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        rows.append(
            {
                "scene": scene,
                "frames": int(receipt["consumed_raw_frames"]),
                "seconds": float(receipt["run_seconds"]),
                "fps": float(receipt["raw_fps"]),
                "peak_allocated_bytes": int(receipt["peak_allocated_bytes"]),
            }
        )
    result: dict = {"completed_scenes": len(rows), "scenes": rows}
    if rows:
        frames = sum(row["frames"] for row in rows)
        seconds = sum(row["seconds"] for row in rows)
        fps = [row["fps"] for row in rows]
        result["aggregate"] = {
            "frames": frames,
            "seconds": seconds,
            "frame_weighted_fps": frames / seconds,
            "scene_fps_mean": statistics.fmean(fps),
            "scene_fps_median": statistics.median(fps),
            "scene_fps_p25": percentile(fps, 0.25),
            "scene_fps_p75": percentile(fps, 0.75),
            "scene_fps_min": min(fps),
            "scene_fps_max": max(fps),
            "peak_allocated_bytes_max": max(
                row["peak_allocated_bytes"] for row in rows
            ),
        }
    write_json(output / "summary.partial.json", result)
    return result


def write_report(output: Path, summary: dict, protocol: dict) -> None:
    aggregate = summary["aggregate"]
    report = f"""# Original BoxFusion ScanNet-100 FPS reproduction

This is an original-BoxFusion-only repeat on the released uniformly sampled
ScanNet validation list. It uses one local RTX 3090, gap=25, score threshold
0.5, no proposal replay, no interactive visualization, no explicit page-cache
prewarm, and excludes model construction and metric evaluation. RGB-D loading,
per-frame preprocessing, CuTR inference, association/PFO fusion, and final
serialization are inside the measured execution path.

| Statistic | FPS |
|---|---:|
| Frame-weighted throughput | {aggregate['frame_weighted_fps']:.4f} |
| Mean scene throughput | {aggregate['scene_fps_mean']:.4f} |
| Median scene throughput | {aggregate['scene_fps_median']:.4f} |
| Scene P25 / P75 | {aggregate['scene_fps_p25']:.4f} / {aggregate['scene_fps_p75']:.4f} |

Processed {aggregate['frames']:,} raw frames from {summary['completed_scenes']}
scenes. Peak PyTorch allocated GPU memory was
{aggregate['peak_allocated_bytes_max'] / 2**30:.3f} GiB.

The paper reports the rounded reference value 20 FPS on an Intel i9-14900K and
RTX 3090 Ti. This machine uses `{protocol['cpu']}` and the GPU recorded in each
scene receipt, so numerical identity is not expected; the purpose of this run
is to reproduce the released software protocol without tuning timing to a
target value.
"""
    (output / "REPORT.md").write_text(report, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument("--limit", type=int, default=100)
    args = parser.parse_args()

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / "logs").mkdir(exist_ok=True)
    source = (
        ROOT
        / "reports/paper_matched_runtime_full100_20260916"
        / "official_source_06ea629"
    ).resolve()
    benchmark = (
        ROOT
        / "development/reliability_v2/tools"
        / "benchmark_official_boxfusion_scene.py"
    ).resolve()
    scene_list = (
        ROOT / "evaluation/data_util/meta_data/scannetv2_val.txt"
    ).resolve()
    scenes = scene_list.read_text(encoding="utf-8").split()[: args.limit]
    if len(scenes) != args.limit:
        raise RuntimeError(
            f"requested {args.limit} scenes but list contains {len(scenes)}"
        )
    for required in (
        source / "demo.py",
        source / "config/scannet.yaml",
        benchmark,
        scene_list,
    ):
        if not required.is_file():
            raise FileNotFoundError(required)

    protocol = {
        "schema": "boxfusion.original_only.scannet100.v1",
        "source_commit": "06ea629",
        "source_demo_sha256": sha256(source / "demo.py"),
        "source_config_sha256": sha256(source / "config/scannet.yaml"),
        "benchmark_sha256": sha256(benchmark),
        "scene_list": str(scene_list),
        "scene_list_sha256": sha256(scene_list),
        "scenes": scenes,
        "device_index": args.device_index,
        "gpu_count": 1,
        "gap": 25,
        "score_threshold": 0.5,
        "visualization": False,
        "scene_cache_prewarmed": False,
        "model_initialization_excluded": True,
        "metric_evaluation_excluded": True,
        "proposal_replay": False,
        "cpu": platform.processor() or platform.machine(),
    }
    protocol_path = output / "protocol.json"
    if protocol_path.exists():
        previous = json.loads(protocol_path.read_text(encoding="utf-8"))
        if previous != protocol:
            raise RuntimeError("existing protocol differs; refusing to mix runs")
    else:
        write_json(protocol_path, protocol)

    started = time.perf_counter()
    for index, scene in enumerate(scenes, start=1):
        scene_output = output / "scenes" / scene
        receipt = scene_output / "runtime.json"
        if valid_receipt(receipt, scene):
            print(f"SKIP {index}/{len(scenes)} {scene}", flush=True)
            continue
        scene_output.mkdir(parents=True, exist_ok=True)
        log_path = output / "logs" / f"{scene}.log"
        command = [
            sys.executable,
            str(benchmark),
            "--source-root",
            str(source),
            "--repository-root",
            str(ROOT),
            "--scene",
            scene,
            "--output-root",
            str(scene_output),
            "--device-index",
            str(args.device_index),
        ]
        print(f"RUN {index}/{len(scenes)} {scene}", flush=True)
        with log_path.open("w", encoding="utf-8") as handle:
            completed = subprocess.run(
                command,
                cwd=ROOT,
                stdout=handle,
                stderr=subprocess.STDOUT,
                check=False,
            )
        if completed.returncode != 0:
            tail = "\n".join(
                log_path.read_text(errors="replace").splitlines()[-40:]
            )
            raise RuntimeError(
                f"scene {scene} failed with {completed.returncode}:\n{tail}"
            )
        current = summarize(output, scenes)
        row = current["scenes"][-1]
        print(
            f"DONE {index}/{len(scenes)} {scene} "
            f"frames={row['frames']} fps={row['fps']:.3f}",
            flush=True,
        )

    summary = summarize(output, scenes)
    summary["completed"] = summary["completed_scenes"] == len(scenes)
    summary["wall_seconds_this_invocation"] = time.perf_counter() - started
    write_json(output / "summary.json", summary)
    if not summary["completed"]:
        raise RuntimeError("run ended with incomplete scene receipts")
    write_report(output, summary, protocol)
    print(json.dumps(summary["aggregate"], indent=2), flush=True)


if __name__ == "__main__":
    main()
