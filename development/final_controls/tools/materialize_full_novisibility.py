#!/usr/bin/env python3
"""Build paired full ReCaR-3D arms with/without MVSR visibility."""
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

from boxfusion.m1_anchor_online import OnlineAnchorRecovery


def read(path: Path) -> list[tuple[int, np.ndarray, float]]:
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    if not isinstance(payload, (list, tuple)) or len(payload) != 1:
        raise RuntimeError(f"invalid prediction payload: {path}")
    return [
        (int(label), np.asarray(box, dtype=np.float64).reshape(8, 3), float(score))
        for label, box, score in payload[0]
    ]


def write(path: Path, rows: list[tuple[int, np.ndarray, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump([[tuple(row) for row in rows]], handle)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def same_geometry(left, right) -> bool:
    return len(left) == len(right) and all(
        a[0] == b[0] and np.array_equal(a[1], b[1])
        for a, b in zip(left, right)
    )


def replay_calr_v1(scene: str, path: Path):
    state = OnlineAnchorRecovery(
        scene, voxel_m=0.3, min_views=3, max_active=4096, max_births=640
    )
    with np.load(path, allow_pickle=False) as values:
        frame_ids = np.asarray(values["frame_ids"], dtype=np.int64)
        lengths = np.asarray(values["anchor_lengths"], dtype=np.int64)
        anchor_ids = np.asarray(values["anchor_ids"], dtype=np.int64)
        corners = np.asarray(values["anchor_corners"], dtype=np.float64)
        scores = np.asarray(values["anchor_scores"], dtype=np.float64)
    offsets = np.r_[0, np.cumsum(lengths)]
    for ordinal, frame_id in enumerate(frame_ids):
        low, high = int(offsets[ordinal]), int(offsets[ordinal + 1])
        state.update(
            ordinal, int(frame_id), anchor_ids[low:high],
            corners[low:high], scores[low:high],
        )
    return [
        (0, np.asarray(row["box"], dtype=np.float64), float(row["score"]))
        for row in state.rows()
    ], state.diagnostics()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-list", type=Path, required=True)
    parser.add_argument("--components", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    scenes = [
        row.strip() for row in args.scene_list.read_text().splitlines()
        if row.strip() and not row.lstrip().startswith("#")
    ]
    if len(scenes) != 100 or len(set(scenes)) != 100:
        raise RuntimeError("expected the official 100 unique ScanNet scenes")
    if (args.output / "manifest.json").exists():
        raise RuntimeError("output manifest exists; refusing to overwrite")

    totals = {"four_factor": 0, "no_visibility": 0, "plr": 0, "calr_v1": 0}
    per_scene = {}
    hashes = {}
    for scene in scenes:
        paths = {
            "four_factor": args.components / "native_max" / f"{scene}_boxes.pkl",
            "no_visibility": args.components / "native_factor_no_visibility" / f"{scene}_boxes.pkl",
            "plr": args.components / "births_plr_v1" / f"{scene}_boxes.pkl",
            "evidence": args.evidence / f"{scene}.npz",
        }
        native4 = read(paths["four_factor"])
        native3 = read(paths["no_visibility"])
        plr = read(paths["plr"])
        if not same_geometry(native4, native3):
            raise RuntimeError(f"MVSR changed native geometry: {scene}")
        calr, diagnostics = replay_calr_v1(scene, paths["evidence"])
        outputs = {
            "four_factor": native4 + plr + calr,
            "no_visibility": native3 + plr + calr,
        }
        for name, rows in outputs.items():
            write(args.output / name / f"{scene}_boxes.pkl", rows)
            totals[name] += len(rows)
        totals["plr"] += len(plr)
        totals["calr_v1"] += len(calr)
        per_scene[scene] = {
            "native": len(native4), "plr": len(plr), "calr_v1": len(calr),
            "calr_diagnostics": diagnostics,
        }
        for path in paths.values():
            hashes[str(path.resolve())] = digest(path)

    manifest = {
        "schema": "boxfusion.full_novisibility_paired.v1",
        "scene_count": len(scenes),
        "single_online_evidence_pass": True,
        "strictly_causal": True,
        "native_geometry_count_labels_fixed": True,
        "recovered_boxes_identical_between_arms": True,
        "calr_version": "v1",
        "totals": totals,
        "per_scene": per_scene,
        "input_sha256": hashes,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(totals, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
