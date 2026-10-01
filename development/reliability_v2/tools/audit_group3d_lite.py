#!/usr/bin/env python3
"""Deterministic parity and bounded-latency audit for Group3D-lite.

The audit never touches BoxFusion outputs.  It compares the optimized raw and
prepared matchers with the independent reference matcher, then times the
preregistered worst-cap prepared path.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import statistics
import time

import numpy as np

from boxfusion import group3d_lite as fast
from boxfusion import group3d_lite_oracle as oracle


def _grid(count: int, offset: tuple[int, int, int]) -> np.ndarray:
    base = np.stack(
        (
            np.arange(count, dtype=np.int64),
            np.zeros(count, dtype=np.int64),
            np.zeros(count, dtype=np.int64),
        ),
        axis=1,
    )
    return base + np.asarray(offset, dtype=np.int64)


def _proposal(identifier: int, points: np.ndarray, score: float) -> dict:
    return {"id": identifier, "score": score, "voxels": points}


def _track(identifier: int, views: list[np.ndarray]) -> dict:
    return {"id": identifier, "views": views}


def _random_case(rng: np.random.Generator) -> tuple[list[dict], list[dict], np.ndarray]:
    proposals = []
    for proposal_id in range(int(rng.integers(0, 14))):
        points = rng.integers(
            -40,
            41,
            size=(int(rng.integers(0, 45)), 3),
            dtype=np.int64,
        )
        if len(points) and bool(rng.integers(0, 2)):
            points = np.concatenate((points, points[: min(3, len(points))]), axis=0)
        proposals.append(_proposal(proposal_id, points, float(rng.normal())))
    tracks = []
    for track_id in range(int(rng.integers(0, 20))):
        views = [
            rng.integers(
                -40,
                41,
                size=(int(rng.integers(0, 45)), 3),
                dtype=np.int64,
            )
            for _ in range(int(rng.integers(1, 4)))
        ]
        tracks.append(_track(track_id, views))

    # Every case has one isolated negative-coordinate accepted association, so
    # parity cannot pass merely because both implementations always abstain.
    anchor = _grid(16, (-10_000, -31, -17))
    proposals.append(_proposal(1_000_000, anchor.copy(), 1_000_000.0))
    tracks.append(_track(2_000_000, [anchor.copy()]))
    mask = rng.integers(0, 2, size=len(tracks), dtype=np.int8).astype(bool)
    mask[-1] = True
    return proposals, tracks, mask


def _run_parity(trials: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    started = time.perf_counter()
    accepted = 0
    for trial in range(trials):
        proposals, tracks, mask = _random_case(rng)
        expected = oracle.match_voxels(proposals, tracks, mask)
        raw = fast.match_voxels(proposals, tracks, mask)
        prepared_tracks = fast.prepare_track_snapshot(tracks)
        prepared_proposals = fast.prepare_proposals(proposals)
        if prepared_tracks.snapshot is None or prepared_proposals.batch is None:
            raise AssertionError("valid randomized input failed preparation")
        prepared = fast.match_prepared_proposals(
            prepared_proposals.batch,
            prepared_tracks.snapshot,
            mask,
        )
        if raw != expected:
            raise AssertionError("raw parity mismatch at trial %d" % trial)
        if prepared != expected:
            raise AssertionError("prepared parity mismatch at trial %d" % trial)
        accepted += len(expected.associations)
    return {
        "trials": trials,
        "paths_checked": trials * 2,
        "accepted_associations": accepted,
        "seconds": time.perf_counter() - started,
    }


def _percentiles(samples: list[float]) -> dict:
    return {
        "median_ms": statistics.median(samples),
        "p95_ms": float(np.percentile(samples, 95)),
        "max_ms": max(samples),
    }


def _run_benchmark(warmup: int, iterations: int) -> dict:
    # Every broad-phase AABB survives: 64 x 1024 checks. Exact work remains
    # hard-capped to eight tracks per proposal.
    proposals = [
        _proposal(
            proposal_id,
            np.vstack((_grid(8, (0, 0, 0)), _grid(8, (6300, 0, 0)))),
            float(64 - proposal_id),
        )
        for proposal_id in range(64)
    ]
    tracks = [
        _track(
            track_id,
            [np.vstack((_grid(8, (0, 0, 0)), _grid(8, (6300, 0, 0))))],
        )
        for track_id in range(1024)
    ]
    mask = np.ones(1024, dtype=bool)
    snapshot = fast.prepare_track_snapshot(tracks).snapshot
    batch = fast.prepare_proposals(proposals).batch
    if snapshot is None or batch is None:
        raise AssertionError("benchmark preparation failed")
    for _ in range(warmup):
        fast.match_prepared_proposals(batch, snapshot, mask)
    samples = []
    candidate_pairs = None
    for _ in range(iterations):
        started = time.perf_counter_ns()
        result = fast.match_prepared_proposals(batch, snapshot, mask)
        samples.append((time.perf_counter_ns() - started) / 1_000_000.0)
        candidate_pairs = result.diagnostics.candidate_pairs
    metrics = _percentiles(samples)
    metrics.update(
        {
            "warmup": warmup,
            "iterations": iterations,
            "candidate_pairs": candidate_pairs,
            "broad_phase_pairs": 64 * 1024,
        }
    )
    return metrics


def _sha256(module) -> str:
    return hashlib.sha256(pathlib.Path(module.__file__).read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260823)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=30)
    args = parser.parse_args()
    if args.trials <= 0 or args.warmup < 0 or args.iterations <= 0:
        parser.error("trials/iterations must be positive and warmup non-negative")

    started = time.perf_counter()
    report = {
        "schema_version": 1,
        "fast_sha256": _sha256(fast),
        "oracle_sha256": _sha256(oracle),
        "parity": _run_parity(args.trials, args.seed),
        "worst_cap_prepared_match": _run_benchmark(args.warmup, args.iterations),
    }
    report["total_seconds"] = time.perf_counter() - started
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
