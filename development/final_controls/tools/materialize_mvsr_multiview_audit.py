#!/usr/bin/env python3
"""Materialize matched ScanNet MVSR view-count and shuffled controls."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import pickle
from pathlib import Path

import numpy as np


COMPONENTS = {
    "native": "native_factor_no_visibility_native",
    "single_view": "native_factor_no_visibility_max_ge1",
    "two_view": "native_factor_no_visibility_max_ge2",
    "three_view": "native_factor_no_visibility_max_ge3",
}


def read(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with path.open("rb") as handle:
        rows = pickle.load(handle)[0]
    boxes = np.asarray([row[1] for row in rows], dtype=np.float64).reshape(-1, 8, 3)
    scores = np.asarray([row[2] for row in rows], dtype=np.float64).reshape(-1)
    return boxes, scores


def write(path: Path, boxes: np.ndarray, scores: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    payload = [[(0, np.array(box, copy=True), float(score)) for box, score in zip(boxes, scores)]]
    with temporary.open("wb") as handle:
        pickle.dump(payload, handle)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def logit(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, 1e-4, 1.0 - 1e-4)
    return np.log(clipped / (1.0 - clipped))


def sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-values))


def scene_seed(scene: str, seed: int) -> int:
    digest = hashlib.sha256(f"{seed}:{scene}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little", signed=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-list", type=Path, required=True)
    parser.add_argument("--components", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-scenes", type=int, default=100)
    parser.add_argument("--shuffle-seed", type=int, default=20261012)
    args = parser.parse_args()

    scenes = [row.strip() for row in args.scene_list.read_text().splitlines() if row.strip() and not row.startswith("#")]
    if len(scenes) != args.expected_scenes:
        raise RuntimeError(f"expected {args.expected_scenes} scenes, got {len(scenes)}")
    counts = {arm: 0 for arm in (*COMPONENTS, "shuffled_negative")}
    activated = {arm: 0 for arm in ("single_view", "two_view", "three_view", "shuffled_negative")}
    scene_rows = {}
    for scene in scenes:
        loaded = {
            arm: read(args.components / component / f"{scene}_boxes.pkl")
            for arm, component in COMPONENTS.items()
        }
        native_boxes, native_scores = loaded["native"]
        for arm, (boxes, scores) in loaded.items():
            if not np.array_equal(boxes, native_boxes):
                raise RuntimeError(f"{scene}: geometry differs for {arm}")
            if len(scores) != len(native_scores):
                raise RuntimeError(f"{scene}: row count differs for {arm}")
            write(args.output / arm / f"{scene}_boxes.pkl", boxes, scores)
            counts[arm] += len(scores)
            if arm != "native":
                activated[arm] += int(np.count_nonzero(scores > native_scores + 1e-12))

        single_scores = loaded["single_view"][1]
        correction = logit(single_scores) - logit(native_scores)
        rng = np.random.default_rng(scene_seed(scene, args.shuffle_seed))
        permuted = correction[rng.permutation(len(correction))]
        shuffled_scores = sigmoid(logit(native_scores) + permuted)
        write(args.output / "shuffled_negative" / f"{scene}_boxes.pkl", native_boxes, shuffled_scores)
        counts["shuffled_negative"] += len(shuffled_scores)
        activated["shuffled_negative"] += int(np.count_nonzero(permuted > 1e-12))
        scene_rows[scene] = {
            "native_rows": len(native_scores),
            "activated": {
                arm: int(np.count_nonzero(scores > native_scores + 1e-12))
                for arm, (_, scores) in loaded.items() if arm != "native"
            },
            "shuffle_nonzero_corrections": int(np.count_nonzero(permuted > 1e-12)),
        }

    if len(set(counts.values())) != 1:
        raise RuntimeError(f"arm row counts differ: {counts}")
    manifest = {
        "schema": "boxfusion.scannet100.mvsr_multiview_audit.v1",
        "protocol": {
            "view_definition": "distinct matched keyframes since the last native-geometry reset",
            "view_arms": "historical maximum support activated after >=1, >=2, or >=3 matched keyframes",
            "shuffled_negative": "within-scene fixed-seed permutation of the final logit correction across native identities",
            "shuffle_seed": args.shuffle_seed,
            "geometry_and_row_count_fixed": True,
            "visibility_used": False,
        },
        "arm_counts": counts,
        "activated_rows": activated,
        "scenes": scene_rows,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
