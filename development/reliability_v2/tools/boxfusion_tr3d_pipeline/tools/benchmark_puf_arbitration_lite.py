#!/usr/bin/env python3
"""Deterministic CPU microbenchmark for PUF arbitration-lite."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

PIPELINE_ROOT = Path(__file__).resolve().parents[1]
if str(PIPELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(PIPELINE_ROOT))

from boxfusion.puf_arbitration_lite import PUFArbitrationLiteObserver
from boxfusion.puf_lite import (
    PUFCandidatePosterior,
    PUFProposalDecision,
    PUFQueryBatch,
)


def candidate(track_id: int, probability: float) -> PUFCandidatePosterior:
    return PUFCandidatePosterior(
        track_id=track_id,
        global_row=track_id,
        source="qim",
        qim_rank=0,
        containment=1.0,
        aabb_iou=1.0,
        overlap_support=1.0,
        center_support=1.0,
        shared_key_fraction=1.0,
        likelihood=1.0,
        probability=probability,
    )


def track_row(
    proposal_id: int,
    track_id: int,
    probability: float,
    conflict: bool,
) -> PUFProposalDecision:
    return PUFProposalDecision(
        proposal_id=proposal_id,
        valid=True,
        actionable=not conflict,
        invalid_reason="same_track_conflict" if conflict else None,
        conflict=conflict,
        qim_candidate_track_ids=(track_id,),
        candidates=(candidate(track_id, probability),),
        birth_probability=1.0 - probability,
        predicted_birth=False,
        predicted_track_id=track_id,
        predicted_global_row=track_id,
        fallback_triggered=False,
        fallback_rescued=False,
        exhaustive_ms=0.0,
        normalization_error=0.0,
    )


def percentile(values, q):
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--proposals", type=int, default=64)
    parser.add_argument("--group-size", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--gap", type=int, default=25)
    parser.add_argument("--report", type=Path, default=None)
    args = parser.parse_args()
    for name in ("proposals", "group_size", "iterations", "gap"):
        if getattr(args, name) < 1:
            parser.error(f"--{name} must be positive")
    if args.proposals > 256:
        parser.error("--proposals must not exceed the configured cap 256")
    if args.warmup < 0:
        parser.error("--warmup must be non-negative")

    observer = PUFArbitrationLiteObserver(
        {
            "enabled": True,
            "observer_only": True,
            "track_min_probability": 0.70,
            "track_min_margin": 0.20,
            "birth_min_probability": 0.70,
            "birth_min_margin": 0.20,
            "conflict_min_owner_gap": 0.10,
            "max_proposals": 256,
            "probability_tolerance": 1e-12,
            "max_diagnostic_examples": 0,
        }
    )
    samples = []
    total = args.warmup + args.iterations
    for step in range(total):
        frame_id = step + 1
        rows = []
        targets = []
        for index in range(args.proposals):
            track_id = index // args.group_size
            group_offset = index % args.group_size
            # A clear owner followed by conservative loser(s).
            probability = 0.72 if group_offset == 0 else 0.55
            conflict = args.group_size > 1
            proposal_id = frame_id * args.proposals + index
            rows.append(
                track_row(proposal_id, track_id, probability, conflict)
            )
            targets.append((track_id,))
        puf_batch = PUFQueryBatch(
            scene_id="benchmark",
            frame_id=frame_id,
            history_max_frame_id=frame_id - 1,
            proposal_ids=tuple(row.proposal_id for row in rows),
            rows=tuple(rows),
            query_ms=0.0,
        )
        result = observer.query(puf_batch=puf_batch)
        observer.observe_native_targets(result, targets)
        if step >= args.warmup:
            samples.append(result.query_ms)

    summary = observer.summary()
    report = {
        "schema": "boxfusion.puf_arbitration_lite_microbenchmark.v1",
        "proposals": args.proposals,
        "group_size": args.group_size,
        "iterations": args.iterations,
        "warmup": args.warmup,
        "gap": args.gap,
        "query_ms": {
            "p50": percentile(samples, 50),
            "p95": percentile(samples, 95),
            "max": float(np.max(samples)),
        },
        "query_p95_ms_per_input_frame": percentile(samples, 95) / args.gap,
        "duplicate_selected_tracks": summary["duplicate_selected_tracks"],
        "conflict_group_resolution_rate": summary[
            "conflict_group_resolution_rate"
        ],
        "selective_precision": summary["selective_precision"],
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
