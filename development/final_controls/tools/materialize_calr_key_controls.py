#!/usr/bin/env python3
"""Materialize current-prefix CALR key controls on ScanNet official100.

The three arms share the final strict-online no-child PLR+MVSR prefix.  The
two controls add exactly the same number of boxes per scene as current CALR.
No ground truth is read by this program.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import pickle
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from boxfusion.m1_anchor_online import deterministic_tail_score
VOXEL_M = 0.3


def add_clusters(clusters, frame, anchor_ids, corners, scores, voxel_size=0.3):
    """Build the GT-free strongest-exemplar voxel control."""
    for anchor, box, score in zip(anchor_ids, corners, scores):
        key = tuple(np.floor(box.mean(0) / voxel_size).astype(np.int64).tolist())
        state = clusters.setdefault(
            key,
            {"frames": set(), "observations": 0, "rank": None, "box": None},
        )
        state["frames"].add(int(frame))
        state["observations"] += 1
        rank = (-float(score), int(frame), int(anchor))
        if state["rank"] is None or rank < state["rank"]:
            state["rank"], state["box"] = rank, box.copy()


def read(path: Path) -> list[tuple[int, np.ndarray, float]]:
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    if not isinstance(payload, (list, tuple)) or len(payload) != 1:
        raise RuntimeError(f"invalid prediction container: {path}")
    rows = []
    for row in payload[0]:
        if len(row) != 3 or int(row[0]) != 0:
            raise RuntimeError(f"invalid class-agnostic prediction: {path}")
        rows.append((0, np.asarray(row[1], dtype=np.float64).reshape(8, 3),
                     float(row[2])))
    return rows


def write(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump([[tuple(row) for row in rows]], handle)


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            value.update(chunk)
    return value.hexdigest()


def same_rows(left, right) -> bool:
    return len(left) == len(right) and all(
        a[0] == b[0] and np.array_equal(a[1], b[1]) and a[2] == b[2]
        for a, b in zip(left, right)
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-list", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--full", type=Path, required=True)
    parser.add_argument("--anchor-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    scenes = [line.strip() for line in args.scene_list.read_text().splitlines()
              if line.strip() and not line.lstrip().startswith("#")]
    if len(scenes) != 100 or len(set(scenes)) != 100:
        raise RuntimeError("expected the official 100 unique ScanNet scenes")

    arms = ("lower_threshold", "strongest_matched_budget", "full_calr")
    for arm in arms:
        (args.output / arm).mkdir(parents=True, exist_ok=True)
    totals = {arm: {"boxes": 0, "births": 0} for arm in arms}
    per_scene = {}
    hashes = {}

    for scene in scenes:
        baseline_path = args.baseline / f"{scene}_boxes.pkl"
        full_path = args.full / f"{scene}_boxes.pkl"
        baseline = read(baseline_path)
        full = read(full_path)
        if len(full) < len(baseline) or not same_rows(
            baseline, full[:len(baseline)]
        ):
            raise RuntimeError(f"CALR changed its current prefix: {scene}")
        q = len(full) - len(baseline)

        cache_path = args.anchor_cache / f"{scene}.npz"
        with np.load(cache_path, allow_pickle=False) as values:
            if "anchor_corners" in values.files:
                boxes = np.asarray(values["anchor_corners"], dtype=np.float64)
                scores = np.asarray(values["anchor_scores"], dtype=np.float64)
                keyframes = np.asarray(values["frame_ids"], dtype=np.int64)
                lengths = np.asarray(values["anchor_lengths"], dtype=np.int64)
                frames = np.repeat(keyframes, lengths)
                anchor_ids = np.asarray(values["anchor_ids"], dtype=np.int64)
            else:
                boxes = np.asarray(values["corners_raw"], dtype=np.float64)
                scores = np.asarray(values["scores"], dtype=np.float64)
                frames = np.asarray(values["frame_ids"], dtype=np.int64)
                anchor_ids = np.asarray(values["anchor_ids"], dtype=np.int64)
        if not (len(boxes) == len(scores) == len(frames) == len(anchor_ids)):
            raise RuntimeError(f"anchor cache is misaligned: {scene}")
        clusters = {}
        for frame in np.unique(frames):
            keep_frame = np.flatnonzero(frames == frame)
            add_clusters(
                clusters,
                int(frame),
                anchor_ids[keep_frame],
                boxes[keep_frame],
                scores[keep_frame],
                VOXEL_M,
            )
        order = np.argsort(-scores, kind="stable")
        if len(order) < q:
            raise RuntimeError(f"low-threshold pool smaller than budget: {scene}")
        lower_rows = [
            (0, boxes[index], float(scores[index])) for index in order[:q]
        ]

        states = sorted(clusters.values(), key=lambda row: row["rank"])
        if len(states) < q:
            raise RuntimeError(f"matched simple pool smaller than budget: {scene}")
        strongest_rows = []
        for state in states[:q]:
            raw_score, frame_id, anchor_id = (
                -float(state["rank"][0]),
                int(state["rank"][1]),
                int(state["rank"][2]),
            )
            strongest_rows.append((
                0,
                np.asarray(state["box"], dtype=np.float64),
                deterministic_tail_score(
                    raw_score, scene, frame_id, anchor_id
                ),
            ))

        outputs = {
            "lower_threshold": baseline + lower_rows,
            "strongest_matched_budget": baseline + strongest_rows,
            "full_calr": full,
        }
        scene_counts = {}
        for arm, rows in outputs.items():
            write(args.output / arm / f"{scene}_boxes.pkl", rows)
            births = len(rows) - len(baseline)
            if births != q:
                raise RuntimeError(f"budget mismatch in {arm}: {scene}")
            totals[arm]["boxes"] += len(rows)
            totals[arm]["births"] += births
            scene_counts[arm] = births
        per_scene[scene] = scene_counts
        for path in (baseline_path, full_path, cache_path):
            hashes[str(path.resolve())] = digest(path)

    manifest = {
        "schema": "boxfusion.calr.current_key_controls.v1",
        "scene_count": len(scenes),
        "baseline": "current strict-online no-child PLR+MVSR",
        "budgets_match_full_calr_per_scene": True,
        "control_uses_ground_truth": False,
        "control_uses_complete_scene_anchor_cache": True,
        "control_uses_full_calr_terminal_birth_count_as_budget": True,
        "control_strictly_online_output": False,
        "full_calr_strictly_online": True,
        "parameters": {
            "voxel_m": VOXEL_M,
            "anchor_budget_per_keyframe": 300,
            "simple_control": (
                "same anchor pool, 0.3 m voxel exemplar and deterministic "
                "tail score; no distinct-keyframe confirmation; terminal "
                "selection from the complete-scene cache at a matched budget"
            ),
        },
        "arms": totals,
        "per_scene_births": per_scene,
        "input_sha256": hashes,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(totals, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
