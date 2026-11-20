#!/usr/bin/env python3
"""Run final no-visibility ReCaR-3D under the BoxFusion reproduction protocol."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
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
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return (
        value.get("scene") == scene
        and value.get("arm") == "recar"
        and int(value.get("raw_frames", 0)) > 0
        and float(value.get("cost_seconds", 0.0)) > 0.0
        and value.get("scene_cache_prewarmed") is False
        and value.get("visualization") is False
    )


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    position = q * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def summarize(output: Path, scenes: list[str]) -> dict:
    rows = []
    for scene in scenes:
        path = output / "scenes" / scene / "runtime.json"
        if not valid_receipt(path, scene):
            continue
        receipt = json.loads(path.read_text(encoding="utf-8"))
        rows.append(
            {
                "scene": scene,
                "frames": int(receipt["raw_frames"]),
                "seconds": float(receipt["cost_seconds"]),
                "fps": float(receipt["fps"]),
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


def write_report(
    output: Path,
    summary: dict,
    original_summary: dict,
) -> None:
    recar = summary["aggregate"]
    original = original_summary["aggregate"]
    if recar["frames"] != original["frames"]:
        raise RuntimeError("aggregate raw-frame count differs from original run")
    retention = recar["frame_weighted_fps"] / original["frame_weighted_fps"]
    report = f"""# Final no-visibility ReCaR-3D FPS reproduction

The final ReCaR-3D configuration was measured under the same explicit protocol
as the original BoxFusion reproduction: the identical ScanNet-100 scene order,
gap=25, native score threshold 0.5, one RTX 3090, no explicit scene-cache
prewarm, no interactive visualization or proposal replay, and CUDA-synchronized
timing. Model construction and metric evaluation are excluded; RGB-D loading,
per-frame preprocessing, inference, association/fusion, all ReCaR stages, and
final serialization are included.

The final ReCaR route is PLR-v1 + CALR-v1 + three-factor MVSR (proposal score,
2D overlap, 3D consistency; no visibility) with historical-maximum aggregation.
All MVSR audit/shadow variants are disabled during timing.

| Method | Frame-weighted FPS | Scene mean | Scene median | Peak allocated memory |
|---|---:|---:|---:|---:|
| Original BoxFusion reproduction | {original['frame_weighted_fps']:.4f} | {original['scene_fps_mean']:.4f} | {original['scene_fps_median']:.4f} | {original['peak_allocated_bytes_max'] / 2**30:.3f} GiB |
| Final ReCaR-3D, no visibility | {recar['frame_weighted_fps']:.4f} | {recar['scene_fps_mean']:.4f} | {recar['scene_fps_median']:.4f} | {recar['peak_allocated_bytes_max'] / 2**30:.3f} GiB |

Frame-weighted throughput retention relative to the local original reproduction:
**{100.0 * retention:.2f}%**. Both rows contain {recar['frames']:,} raw frames.

The two runs share the explicit software timing protocol, but neither run
forces or clears the operating-system page cache. Accordingly, the absolute
cross-run difference should be reported together with this cache-state caveat;
the separate cache-symmetric benchmark remains the stronger pure-overhead
measurement.
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
    source = (ROOT / "development/final_controls").resolve()
    config = (
        source
        / "experiments/mvsr_novis_temporal_20261006/config.yaml"
    ).resolve()
    benchmark = (
        ROOT
        / "development/final_efficiency_original_vs_recar_20261005/tools"
        / "benchmark_scene.py"
    ).resolve()
    scene_list = (
        ROOT / "evaluation/data_util/meta_data/scannetv2_val.txt"
    ).resolve()
    original_root = (
        ROOT / "reports/original_boxfusion_fps_reproduction100_20261006"
    ).resolve()
    original_summary_path = original_root / "summary.json"
    scenes = scene_list.read_text(encoding="utf-8").split()[: args.limit]
    original_summary = json.loads(
        original_summary_path.read_text(encoding="utf-8")
    )
    original_scenes = [row["scene"] for row in original_summary["scenes"]]
    if scenes != original_scenes[: args.limit]:
        raise RuntimeError("scene order differs from original reproduction")

    protocol = {
        "schema": "recar3d.final_novis.original_protocol.scannet100.v1",
        "source_root": str(source),
        "source_demo_sha256": sha256(source / "demo.py"),
        "source_reliability_candidate_map_sha256": sha256(
            source / "boxfusion/reliability_candidate_map.py"
        ),
        "source_native_reliability_reranker_sha256": sha256(
            source / "boxfusion/native_reliability_reranker.py"
        ),
        "source_calr_v1_sha256": sha256(
            source / "boxfusion/m1_anchor_online.py"
        ),
        "source_config": str(config),
        "source_config_sha256": sha256(config),
        "benchmark_sha256": sha256(benchmark),
        "scene_list": str(scene_list),
        "scene_list_sha256": sha256(scene_list),
        "scenes": scenes,
        "original_protocol": str(original_root / "protocol.json"),
        "original_summary": str(original_summary_path),
        "device_index": args.device_index,
        "gpu_count": 1,
        "gap": 25,
        "native_score_threshold": 0.5,
        "visualization": False,
        "scene_cache_prewarmed": False,
        "model_initialization_excluded": True,
        "metric_evaluation_excluded": True,
        "proposal_replay": False,
        "plr": "v1",
        "calr": "v1",
        "mvsr_strength_variant": "no_visibility",
        "mvsr_temporal_aggregation": "max",
        "mvsr_audit_variants_timed": False,
    }
    protocol_path = output / "protocol.json"
    if protocol_path.exists():
        if json.loads(protocol_path.read_text(encoding="utf-8")) != protocol:
            raise RuntimeError("existing protocol differs; refusing mixed run")
    else:
        write_json(protocol_path, protocol)

    started = time.perf_counter()
    child_environment = os.environ.copy()
    child_environment["CUDA_VISIBLE_DEVICES"] = str(args.device_index)
    child_environment["PYTHONHASHSEED"] = "0"
    child_environment["PYTHONNOUSERSITE"] = "1"
    child_environment["PYTHONDONTWRITEBYTECODE"] = "1"
    child_environment["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    child_environment["HF_HUB_OFFLINE"] = "1"
    child_environment["OMP_NUM_THREADS"] = "8"
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
            "--arm",
            "recar",
            "--scene",
            scene,
            "--source-root",
            str(source),
            "--repository-root",
            str(ROOT),
            "--output-root",
            str(scene_output),
            "--config",
            str(config),
            "--device-index",
            "0",
        ]
        print(f"RUN {index}/{len(scenes)} {scene}", flush=True)
        with log_path.open("w", encoding="utf-8") as handle:
            completed = subprocess.run(
                command,
                cwd=ROOT,
                env=child_environment,
                stdout=handle,
                stderr=subprocess.STDOUT,
                check=False,
            )
        if completed.returncode != 0:
            tail = "\n".join(
                log_path.read_text(errors="replace").splitlines()[-50:]
            )
            raise RuntimeError(
                f"scene {scene} failed with {completed.returncode}:\n{tail}"
            )
        current = summarize(output, scenes)
        row = current["scenes"][-1]
        expected_frames = int(original_summary["scenes"][index - 1]["frames"])
        if row["frames"] != expected_frames:
            raise RuntimeError(
                f"raw-frame mismatch for {scene}: "
                f"ReCaR={row['frames']} original={expected_frames}"
            )
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
    write_report(output, summary, original_summary)
    print(json.dumps(summary["aggregate"], indent=2), flush=True)


if __name__ == "__main__":
    main()
