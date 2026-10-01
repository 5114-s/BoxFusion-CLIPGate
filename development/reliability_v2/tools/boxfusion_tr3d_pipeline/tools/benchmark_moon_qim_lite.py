#!/usr/bin/env python3
"""Deterministic synthetic microbenchmark for Moon-QIM-lite."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from time import perf_counter_ns

import numpy as np

PIPELINE_ROOT = Path(__file__).resolve().parents[1]
if str(PIPELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(PIPELINE_ROOT))

from boxfusion.moon_qim_lite import MoonQIMLiteObserver


_SIGNS = np.asarray(
    [
        [-1, -1, -1],
        [-1, -1, 1],
        [-1, 1, -1],
        [-1, 1, 1],
        [1, -1, -1],
        [1, -1, 1],
        [1, 1, -1],
        [1, 1, 1],
    ],
    dtype=np.float32,
)


def boxes(centers: np.ndarray, sizes: np.ndarray) -> np.ndarray:
    return (
        centers[:, None, :]
        + _SIGNS[None, :, :] * sizes[:, None, :] / 2.0
    ).astype(np.float32, copy=False)


def percentile(values, q):
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tracks", type=int, default=128)
    parser.add_argument("--proposals", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--report", type=Path, default=None)
    args = parser.parse_args()
    for name in ("tracks", "proposals", "iterations"):
        if getattr(args, name) < 1:
            parser.error(f"--{name} must be positive")
    if args.warmup < 0:
        parser.error("--warmup must be non-negative")
    if args.seed < 0:
        parser.error("--seed must be non-negative")

    rng = np.random.default_rng(args.seed)
    centers = rng.uniform(
        low=np.asarray([-5.0, -5.0, 0.25]),
        high=np.asarray([5.0, 5.0, 2.5]),
        size=(args.tracks, 3),
    ).astype(np.float32)
    sizes = rng.uniform(0.20, 1.25, size=(args.tracks, 3)).astype(
        np.float32
    )
    track_ids = np.arange(args.tracks, dtype=np.int64)
    observer = MoonQIMLiteObserver(
        {
            "enabled": True,
            "observer_only": True,
            "voxel_size_m": 0.30,
            "samples_per_axis": 3,
            "neighbor_radius": 1,
            "max_candidates_per_query": 8,
            "max_tracks": max(args.tracks, 1),
            "track_ttl_keyframes": 80,
            "max_postings_per_key": 32,
        }
    )
    observer.update(
        scene_id="benchmark",
        frame_id=0,
        track_ids=track_ids,
        track_corners_world=boxes(centers, sizes),
    )

    query_times = []
    update_times = []
    candidate_counts = []
    total_steps = args.warmup + args.iterations
    for step in range(total_steps):
        selected = rng.integers(0, args.tracks, size=args.proposals)
        proposal_centers = centers[selected] + rng.normal(
            0.0, 0.04, size=(args.proposals, 3)
        ).astype(np.float32)
        proposal_sizes = sizes[selected] * rng.uniform(
            0.95, 1.05, size=(args.proposals, 3)
        ).astype(np.float32)
        frame_id = step + 1
        batch = observer.query(
            scene_id="benchmark",
            frame_id=frame_id,
            proposal_ids=(
                np.arange(args.proposals, dtype=np.int64)
                + frame_id * args.proposals
                + args.tracks
            ),
            proposal_corners_world=boxes(proposal_centers, proposal_sizes),
        )
        start = perf_counter_ns()
        observer.update(
            scene_id="benchmark",
            frame_id=frame_id,
            track_ids=track_ids,
            track_corners_world=boxes(centers, sizes),
        )
        update_ms = (perf_counter_ns() - start) / 1e6
        if step >= args.warmup:
            query_times.append(batch.query_ms)
            update_times.append(update_ms)
            candidate_counts.extend(len(row) for row in batch.candidates)

    combined = np.asarray(query_times) + np.asarray(update_times)
    snapshot = observer.snapshot()
    report = {
        "schema": "boxfusion.moon_qim_lite_microbenchmark.v1",
        "tracks": args.tracks,
        "proposals": args.proposals,
        "warmup": args.warmup,
        "iterations": args.iterations,
        "seed": args.seed,
        "query_ms": {
            "p50": percentile(query_times, 50),
            "p95": percentile(query_times, 95),
            "max": float(np.max(query_times)),
        },
        "update_ms": {
            "p50": percentile(update_times, 50),
            "p95": percentile(update_times, 95),
            "max": float(np.max(update_times)),
        },
        "combined_ms": {
            "p50": percentile(combined, 50),
            "p95": percentile(combined, 95),
            "max": float(np.max(combined)),
        },
        "candidates_per_query": {
            "mean": float(np.mean(candidate_counts)),
            "p95": percentile(candidate_counts, 95),
            "max": int(np.max(candidate_counts)),
        },
        "retained": {
            "tracks": len(snapshot["track_ids"]),
            "keys": snapshot["key_count"],
            "postings": snapshot["posting_count"],
        },
        "training_free": True,
        "causal": True,
    }
    serialized = json.dumps(report, indent=2, sort_keys=True)
    print(serialized)
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(serialized + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
