#!/usr/bin/env python3
"""Materialize PLR/CALR/MVSR factorial arms from one component run."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import pickle

import numpy as np


ARMS = {
    "base": (False, False, False),
    "p": (True, False, False),
    "a": (False, True, False),
    "m2": (False, False, True),
    "p_a": (True, True, False),
    "p_m2": (True, False, True),
    "a_m2": (False, True, True),
    "p_a_m2": (True, True, True),
}


def read(path: Path):
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    if not isinstance(payload, (tuple, list)) or len(payload) != 1:
        raise RuntimeError(f"invalid prediction container: {path}")
    return [
        (int(label), np.asarray(box, dtype=np.float64).reshape(8, 3), float(score))
        for label, box, score in payload[0]
    ]


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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-list", type=Path, required=True)
    parser.add_argument("--expected-scenes", type=int, required=True)
    parser.add_argument("--components", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    scenes = [
        row.strip() for row in args.scene_list.read_text().splitlines()
        if row.strip() and not row.lstrip().startswith("#")
    ]
    if len(scenes) != args.expected_scenes or len(set(scenes)) != len(scenes):
        raise RuntimeError("scene-list cardinality mismatch")
    if (args.output / "manifest.json").exists():
        raise RuntimeError("factorial manifest already exists")

    counts = {name: 0 for name in ARMS}
    source_counts = {name: 0 for name in ("native", "plr", "calr")}
    per_scene = {}
    hashes = {}
    for scene in scenes:
        paths = {
            "native": args.components / "native_native" / f"{scene}_boxes.pkl",
            "mvsr": args.components / "native_max" / f"{scene}_boxes.pkl",
            "plr": args.components / "births_plr_v1" / f"{scene}_boxes.pkl",
            "calr": args.components / "births_calr_v1" / f"{scene}_boxes.pkl",
        }
        rows = {name: read(path) for name, path in paths.items()}
        if len(rows["native"]) != len(rows["mvsr"]):
            raise RuntimeError(f"MVSR row-count change: {scene}")
        for left, right in zip(rows["native"], rows["mvsr"]):
            if left[0] != right[0] or not np.array_equal(left[1], right[1]):
                raise RuntimeError(f"MVSR geometry/label change: {scene}")
        for arm, (use_plr, use_calr, use_mvsr) in ARMS.items():
            output = list(rows["mvsr"] if use_mvsr else rows["native"])
            if use_plr:
                output.extend(rows["plr"])
            if use_calr:
                output.extend(rows["calr"])
            write(args.output / arm / f"{scene}_boxes.pkl", output)
            counts[arm] += len(output)
        for name in source_counts:
            source_counts[name] += len(rows[name])
        per_scene[scene] = {name: len(value) for name, value in rows.items()}
        for path in paths.values():
            hashes[str(path.resolve())] = digest(path)

    manifest = {
        "schema": "boxfusion.recar3d.factorial.v2",
        "scene_count": len(scenes),
        "single_online_pass": True,
        "paired_from_one_run": True,
        "strictly_causal": True,
        "children_enabled": False,
        "native_geometry_count_labels_fixed_for_mvsr": True,
        "components": {
            "plr": "current PLR",
            "calr": "CALR-v1",
            "mvsr": "MVSR-v2 composite-support max",
        },
        "arm_counts": counts,
        "component_counts": source_counts,
        "per_scene": per_scene,
        "input_sha256": hashes,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps({"arm_counts": counts, "component_counts": source_counts}, indent=2))


if __name__ == "__main__":
    main()
