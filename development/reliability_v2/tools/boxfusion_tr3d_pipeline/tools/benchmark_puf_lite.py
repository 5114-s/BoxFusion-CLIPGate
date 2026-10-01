#!/usr/bin/env python3
"""Deterministic CPU microbenchmark for PUF-lite shortlist and fallback."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

PIPELINE_ROOT = Path(__file__).resolve().parents[1]
if str(PIPELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(PIPELINE_ROOT))

from boxfusion.moon_qim_lite import QIMCandidate, QIMQueryBatch
from boxfusion.puf_lite import PUFLiteShadowObserver, box_geometry_likelihood


SIGNS = np.asarray(
    [
        [-1, -1, -1], [-1, -1, 1], [-1, 1, -1], [-1, 1, 1],
        [1, -1, -1], [1, -1, 1], [1, 1, -1], [1, 1, 1],
    ],
    dtype=np.float32,
)


def boxes(centers: np.ndarray, sizes: np.ndarray) -> np.ndarray:
    return centers[:, None, :] + SIGNS[None, :, :] * sizes[:, None, :] / 2.0


def percentile(values, q):
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def candidate(
    track_id: int, proposal_box: np.ndarray, track_box: np.ndarray
) -> QIMCandidate:
    geometry = box_geometry_likelihood(
        proposal_box, track_box, shared_key_fraction=1.0
    )
    return QIMCandidate(
        track_id=int(track_id),
        shared_key_count=3,
        shared_key_fraction=1.0,
        center_distance_m=float(
            np.linalg.norm(
                np.mean(proposal_box, axis=0) - np.mean(track_box, axis=0)
            )
        ),
        aabb_iou=geometry.aabb_iou,
        age_keyframes=0,
        active_at_last_commit=True,
    )


def run_case(
    *,
    name: str,
    track_ids: np.ndarray,
    track_boxes: np.ndarray,
    selected: np.ndarray,
    proposal_boxes: np.ndarray,
    iterations: int,
    warmup: int,
    use_qim: bool,
) -> dict[str, object]:
    observer = PUFLiteShadowObserver(
        {
            "enabled": True,
            "observer_only": True,
            "top_k": 3,
            "birth_likelihood": 0.4,
            "center_sigma": 0.5,
            "center_margin_m": 0.05,
            "shared_key_power": 1.0,
            "max_tracks": max(len(track_ids), 1),
            "exhaustive_fallback": True,
            "probability_tolerance": 1e-12,
            "epsilon": 1e-9,
            "max_diagnostic_examples": 0,
        }
    )
    samples = []
    total = warmup + iterations
    for step in range(total):
        frame_id = step + 1
        proposal_ids = tuple(
            int(value)
            for value in (
                np.arange(len(proposal_boxes), dtype=np.int64)
                + frame_id * len(proposal_boxes)
                + len(track_ids)
            )
        )
        rows = (
            tuple(
                (
                    candidate(
                        int(track_ids[index]),
                        proposal_boxes[proposal_index],
                        track_boxes[index],
                    ),
                )
                for proposal_index, index in enumerate(selected)
            )
            if use_qim
            else tuple(() for _ in selected)
        )
        query = QIMQueryBatch(
            scene_id=name,
            frame_id=frame_id,
            proposal_ids=proposal_ids,
            candidates=rows,
            history_max_frame_id=frame_id - 1,
            query_ms=0.0,
        )
        result = observer.query(
            qim_batch=query,
            proposal_corners_world=proposal_boxes,
            active_track_ids=track_ids,
            active_track_corners_world=track_boxes,
        )
        observer.observe_native_targets(
            result,
            [(int(track_ids[index]),) for index in selected],
        )
        if step >= warmup:
            samples.append(result.query_ms)
    summary = observer.summary()
    return {
        "name": name,
        "query_ms": {
            "p50": percentile(samples, 50),
            "p95": percentile(samples, 95),
            "max": float(np.max(samples)),
        },
        "fallback_trigger_rate": summary["fallback_trigger_rate"],
        "fallback_rescue_rate": summary["fallback_rescue_rate"],
        "scored_per_query": (
            summary["exhaustive_tracks_scored"] / summary["queries"]
        ),
        "invalid_rows": summary["invalid_rows"],
        "nonfinite_probability_rows": summary["nonfinite_probability_rows"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tracks", type=int, default=128)
    parser.add_argument("--proposals", type=int, default=32)
    parser.add_argument("--iterations", type=int, default=300)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gap", type=int, default=25)
    parser.add_argument("--report", type=Path, default=None)
    args = parser.parse_args()
    for name in ("tracks", "proposals", "iterations", "gap"):
        if getattr(args, name) < 1:
            parser.error(f"--{name} must be positive")
    if args.warmup < 0 or args.seed < 0:
        parser.error("--warmup and --seed must be non-negative")

    rng = np.random.default_rng(args.seed)
    centers = rng.uniform(
        np.asarray([-5.0, -5.0, 0.25]),
        np.asarray([5.0, 5.0, 2.5]),
        size=(args.tracks, 3),
    ).astype(np.float32)
    sizes = rng.uniform(0.2, 1.2, size=(args.tracks, 3)).astype(np.float32)
    track_boxes = boxes(centers, sizes).astype(np.float32)
    track_ids = np.arange(args.tracks, dtype=np.int64)
    selected = rng.integers(0, args.tracks, size=args.proposals)
    proposal_boxes = boxes(
        centers[selected]
        + rng.normal(0.0, 0.02, size=(args.proposals, 3)).astype(np.float32),
        sizes[selected],
    ).astype(np.float32)

    shortlist = run_case(
        name="shortlist",
        track_ids=track_ids,
        track_boxes=track_boxes,
        selected=selected,
        proposal_boxes=proposal_boxes,
        iterations=args.iterations,
        warmup=args.warmup,
        use_qim=True,
    )
    fallback = run_case(
        name="fallback",
        track_ids=track_ids,
        track_boxes=track_boxes,
        selected=selected,
        proposal_boxes=proposal_boxes,
        iterations=args.iterations,
        warmup=args.warmup,
        use_qim=False,
    )
    report = {
        "schema": "boxfusion.puf_lite_microbenchmark.v1",
        "tracks": args.tracks,
        "proposals": args.proposals,
        "iterations": args.iterations,
        "warmup": args.warmup,
        "seed": args.seed,
        "gap": args.gap,
        "shortlist": shortlist,
        "fallback": fallback,
        "fallback_p95_ms_per_input_frame": fallback["query_ms"]["p95"] / args.gap,
        "training_free": True,
        "causal": True,
        "semantic_access": False,
    }
    serialized = json.dumps(report, indent=2, sort_keys=True)
    print(serialized)
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(serialized + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
