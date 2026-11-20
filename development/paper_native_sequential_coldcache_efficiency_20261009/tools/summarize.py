#!/usr/bin/env python3
"""Summarize the exact-final paired BoxFusion versus ReCaR benchmark."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import statistics


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def weighted_mean(rows, path):
    total = 0.0
    count = 0
    for row in rows:
        node = row
        for key in path:
            node = node[key]
        n = int(node["count"])
        total += float(node["mean_ms"]) * n
        count += n
    return (total / count if count else 0.0), count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-list", type=Path, required=True)
    parser.add_argument("--receipt-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    scenes = [
        line.strip().split()[0]
        for line in args.scene_list.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if len(scenes) != 100:
        raise ValueError(f"expected 100 scenes, found {len(scenes)}")
    receipts = {"original": [], "recar": []}
    diagnostics = []
    per_scene = []
    for scene in scenes:
        pair = {}
        for arm in receipts:
            path = args.receipt_root / arm / scene / "runtime.json"
            row = json.loads(path.read_text(encoding="utf-8"))
            if row["scene"] != scene or row["arm"] != arm:
                raise ValueError(f"receipt identity mismatch: {path}")
            receipts[arm].append(row)
            pair[arm] = row
        if pair["original"]["raw_frames"] != pair["recar"]["raw_frames"]:
            raise ValueError(f"frame mismatch for {scene}")
        diag_path = (
            args.receipt_root
            / "recar"
            / scene
            / "online_diagnostics"
            / f"{scene}.json"
        )
        diag = json.loads(diag_path.read_text(encoding="utf-8"))
        diagnostics.append(diag)
        per_scene.append(
            {
                "scene": scene,
                "raw_frames": int(pair["original"]["raw_frames"]),
                "original_fps": float(pair["original"]["fps"]),
                "recar_fps": float(pair["recar"]["fps"]),
                "retention": float(pair["recar"]["fps"])
                / float(pair["original"]["fps"]),
                "recar_keyframes": int(diag["keyframe_end_to_end"]["count"]),
                "recar_keyframe_p95_ms": float(
                    diag["keyframe_end_to_end"]["p95_ms"]
                ),
                "recar_deadline_misses": int(diag["deadline_misses"]),
            }
        )

    arm_summary = {}
    for arm, rows in receipts.items():
        frames = sum(int(row["raw_frames"]) for row in rows)
        seconds = sum(float(row["cost_seconds"]) for row in rows)
        arm_summary[arm] = {
            "raw_frames": frames,
            "total_seconds": seconds,
            "frame_weighted_fps": frames / seconds,
            "mean_scene_fps": statistics.fmean(float(r["fps"]) for r in rows),
            "median_scene_fps": statistics.median(float(r["fps"]) for r in rows),
            "peak_allocated_bytes_max": max(
                int(r["peak_allocated_bytes"]) for r in rows
            ),
            "peak_allocated_gib_max": max(
                int(r["peak_allocated_bytes"]) for r in rows
            )
            / (1024**3),
        }

    keyframe_mean, keyframe_count = weighted_mean(
        diagnostics, ("keyframe_end_to_end",)
    )
    state_mean, state_count = weighted_mean(
        diagnostics, ("state", "timing")
    )
    plr_mean, plr_count = weighted_mean(
        diagnostics, ("state", "plr_timing_and_stability", "current")
    )
    calr_mean, calr_count = weighted_mean(
        diagnostics, ("state", "calr_timing")
    )
    mvsr_mean, mvsr_count = weighted_mean(
        diagnostics, ("state", "mvsr_timing")
    )
    counts = {keyframe_count, state_count, plr_count, calr_count, mvsr_count}
    if len(counts) != 1:
        raise ValueError(f"keyframe timing counts differ: {sorted(counts)}")
    deadline_misses = sum(int(row["deadline_misses"]) for row in diagnostics)
    deadline_ms = {float(row["keyframe_deadline_ms"]) for row in diagnostics}
    if len(deadline_ms) != 1:
        raise ValueError("deadline differs across scenes")
    p95_median = statistics.median(
        float(row["keyframe_end_to_end"]["p95_ms"])
        for row in diagnostics
    )
    p50_scene_median = statistics.median(
        float(row["keyframe_end_to_end"]["p50_ms"])
        for row in diagnostics
    )

    retention = (
        arm_summary["recar"]["frame_weighted_fps"]
        / arm_summary["original"]["frame_weighted_fps"]
    )
    result = {
        "schema": "boxfusion.original_vs_final_recar.sequential_cold_efficiency100.v1",
        "phase_order": ["recar", "original"],
        "phase_transition": "gpu_idle_and_temperature_at_most_initial_plus_2c",
        "scene_count": len(scenes),
        "scene_list": str(args.scene_list.resolve()),
        "scene_list_sha256": sha256(args.scene_list.resolve()),
        "arms": arm_summary,
        "throughput_retention": retention,
        "recar_keyframe": {
            "count": keyframe_count,
            "mean_end_to_end_ms": keyframe_mean,
            "median_of_scene_p50_ms": p50_scene_median,
            "median_of_scene_p95_ms": p95_median,
            "deadline_ms": next(iter(deadline_ms)),
            "deadline_misses": deadline_misses,
            "deadline_miss_rate": deadline_misses / keyframe_count,
            "state_total_mean_ms": state_mean,
            "plr_mean_ms": plr_mean,
            "calr_mean_ms": calr_mean,
            "mvsr_mean_ms": mvsr_mean,
        },
        "memory_measurement": "maximum over scenes of torch.cuda.max_memory_allocated",
        "per_scene": per_scene,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    o = arm_summary["original"]
    r = arm_summary["recar"]
    k = result["recar_keyframe"]
    report = f"""# Official-source BoxFusion vs final ReCaR-3D: verified cold-page-cache timing

Both arms use the same ScanNet-100 scene order, gap=25, score threshold 0.5,
one RTX 3090, and the public BoxFusion demo's native wall-clock boundary. ReCaR
is measured as a complete 100-scene pass first. After GPU 0 is idle and cools
to within 2 degrees C of its initial temperature, frozen original BoxFusion is
measured as a separate complete pass. This avoids immediately reusing the same
scene in a second arm. Before every scene, all RGB, depth, pose, and intrinsic
files are evicted with
`POSIX_FADV_DONTNEED`; `fincore` must report zero resident bytes immediately
before timing. There is no CUDA synchronization, eager raw-frame preparation,
or proposal replay. Model initialization, visualization, metric evaluation, and
final output serialization are outside the timed interval.
RGB-D loading, per-frame preprocessing/device transfer, inference, native
association/fusion, and online ReCaR updates are included.

This is a controlled cold-page-cache benchmark. The paper does not document a
cache-dropping procedure, so its 20 FPS value and this stricter cold-cache result
share the public timing boundary but are not claimed to have identical storage
state.

The ReCaR arm is the snapshotted final route: PLR-v1 + CALR-v1 + composite
three-factor MVSR without visibility, using historical-maximum aggregation.
All diagnostic and shadow variants are disabled during timing.

| Method | FPS | Retention | Peak allocated GPU memory |
|---|---:|---:|---:|
| Frozen original BoxFusion | {o['frame_weighted_fps']:.4f} | 100.00% | {o['peak_allocated_gib_max']:.3f} GiB |
| Final ReCaR-3D | {r['frame_weighted_fps']:.4f} | {retention*100:.2f}% | {r['peak_allocated_gib_max']:.3f} GiB |

ReCaR keyframe diagnostics:

- End-to-end mean: {k['mean_end_to_end_ms']:.3f} ms
- Median of per-scene p50: {k['median_of_scene_p50_ms']:.3f} ms
- Median of per-scene p95: {k['median_of_scene_p95_ms']:.3f} ms
- Deadline: {k['deadline_ms']:.3f} ms
- Deadline misses: {k['deadline_misses']}/{k['count']} ({k['deadline_miss_rate']*100:.2f}%)
- State total: {k['state_total_mean_ms']:.3f} ms/keyframe
- PLR / CALR / MVSR: {k['plr_mean_ms']:.3f} / {k['calr_mean_ms']:.3f} / {k['mvsr_mean_ms']:.3f} ms/keyframe

Peak memory is PyTorch allocated memory (`torch.cuda.max_memory_allocated`),
not `nvidia-smi` process memory. The published 20 FPS / 7.0 GB values are an
RTX 3090 Ti ScanNet reference. This run uses a verified cold input-file cache
state and cannot reproduce other undocumented author-machine system state.
"""
    (args.output / "REPORT.md").write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
