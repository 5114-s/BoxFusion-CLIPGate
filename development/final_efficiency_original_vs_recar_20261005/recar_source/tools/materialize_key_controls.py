#!/usr/bin/env python3
"""Materialize paired MVSR and PLR controls from one causal online pass."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import pickle

import numpy as np


def read(path: Path) -> list[tuple[int, np.ndarray, float]]:
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    if not isinstance(payload, (list, tuple)) or len(payload) != 1:
        raise RuntimeError(f"invalid prediction container: {path}")
    return [
        (int(label), np.asarray(box, dtype=np.float64).reshape(8, 3), float(score))
        for label, box, score in payload[0]
    ]


def write(path: Path, rows: list[tuple[int, np.ndarray, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump([[(label, np.array(box, copy=True), score) for label, box, score in rows]], handle)


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            value.update(chunk)
    return value.hexdigest()


def same_geometry(left, right) -> bool:
    if len(left) != len(right):
        return False
    return all(a[0] == b[0] and np.array_equal(a[1], b[1]) for a, b in zip(left, right))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-list", type=Path, required=True)
    parser.add_argument("--components", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    scenes = [
        row.strip() for row in args.scene_list.read_text().splitlines()
        if row.strip() and not row.lstrip().startswith("#")
    ]
    if len(scenes) != 100 or len(set(scenes)) != 100:
        raise RuntimeError("expected the unique official100 scene list")
    if (args.output / "manifest.json").exists():
        raise RuntimeError("output manifest exists; refusing to overwrite")

    arms = (
        "native", "mvsr_raw_iou_max", "mvsr_composite_max",
        "plr_prefix", "plr_direct", "plr_current",
        "calr_prefix", "calr_full",
        "full_raw_iou", "full_composite",
    )
    totals = {arm: 0 for arm in arms}
    per_scene = {}
    hashes = {}
    for scene in scenes:
        paths = {
            "native": args.components / "native_native" / f"{scene}_boxes.pkl",
            "raw": args.components / "native_raw_iou_max" / f"{scene}_boxes.pkl",
            "composite": args.components / "native_max" / f"{scene}_boxes.pkl",
            "plr_direct": args.components / "births_plr_direct" / f"{scene}_boxes.pkl",
            "plr_current": args.components / "births_plr_v1" / f"{scene}_boxes.pkl",
            "calr": args.components / "births_calr_v1" / f"{scene}_boxes.pkl",
        }
        rows = {name: read(path) for name, path in paths.items()}
        if not same_geometry(rows["native"], rows["raw"]):
            raise RuntimeError(f"raw-IoU MVSR changed native geometry: {scene}")
        if not same_geometry(rows["native"], rows["composite"]):
            raise RuntimeError(f"composite MVSR changed native geometry: {scene}")
        if len(rows["plr_direct"]) != len(rows["plr_current"]):
            raise RuntimeError(f"PLR matched budget failed: {scene}")

        scene_arms = {
            "native": rows["native"],
            "mvsr_raw_iou_max": rows["raw"],
            "mvsr_composite_max": rows["composite"],
            # PLR controls share Native + CALR-v1 + MVSR-v2 Max.
            "plr_prefix": rows["composite"] + rows["calr"],
            "plr_direct": rows["composite"] + rows["plr_direct"] + rows["calr"],
            "plr_current": rows["composite"] + rows["plr_current"] + rows["calr"],
            # CALR controls share Native + PLR + MVSR-v2 Max.
            "calr_prefix": rows["composite"] + rows["plr_current"],
            "calr_full": rows["composite"] + rows["plr_current"] + rows["calr"],
            "full_raw_iou": rows["raw"] + rows["plr_current"] + rows["calr"],
            "full_composite": rows["composite"] + rows["plr_current"] + rows["calr"],
        }
        for arm, output in scene_arms.items():
            write(args.output / arm / f"{scene}_boxes.pkl", output)
            totals[arm] += len(output)
        per_scene[scene] = {name: len(value) for name, value in rows.items()}
        for path in paths.values():
            hashes[str(path.resolve())] = digest(path)

    manifest = {
        "schema": "boxfusion.recar3d.key_controls.v1",
        "scene_count": 100,
        "single_online_pass": True,
        "strictly_causal": True,
        "future_frames_used": False,
        "children_used": False,
        "native_geometry_count_labels_fixed": True,
        "plr_budget_matched_per_scene": True,
        "arm_counts": totals,
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
