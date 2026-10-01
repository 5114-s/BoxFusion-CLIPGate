#!/usr/bin/env python3
"""Materialize the ReCaR-3D factorial arms for the extra50 holdout scenes.

Same arm definitions and row order as
``tools/materialize_recar3d_final_factorial.py`` (the locked official100
materializer): an arm is MVSR-or-native rows followed by optional PLR birth
rows and optional CALR birth rows.  Only the expected scene count differs
(50 pre-registered holdout scenes instead of official100).
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import pickle

import numpy as np


EXPECTED_SCENES = 50
ARMS = {
    "base": (False, False, False),
    "plr": (True, False, False),
    "calr": (False, True, False),
    "mvsr": (False, False, True),
    "plr_calr": (True, True, False),
    "plr_mvsr": (True, False, True),
    "calr_mvsr": (False, True, True),
    "full": (True, True, True),
}


def read(path: Path) -> list[tuple[int, np.ndarray, float]]:
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    if not isinstance(payload, (list, tuple)) or len(payload) != 1:
        raise RuntimeError(f"invalid prediction container: {path}")
    rows = []
    for label, box, score in payload[0]:
        rows.append((int(label), np.asarray(box, dtype=np.float64).reshape(8, 3), float(score)))
    return rows


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
    if len(scenes) != EXPECTED_SCENES or len(set(scenes)) != EXPECTED_SCENES:
        raise RuntimeError("expected the unique ScanNet extra50 scene list")
    if (args.output / "manifest.json").exists():
        raise RuntimeError("output manifest already exists; refusing to overwrite")

    counts = {name: 0 for name in ARMS}
    component_counts = {"native": 0, "plr": 0, "calr": 0}
    hashes: dict[str, str] = {}
    per_scene: dict[str, dict[str, int]] = {}
    for scene in scenes:
        paths = {
            "native": args.components / "native_native" / f"{scene}_boxes.pkl",
            "mvsr": args.components / "native_max" / f"{scene}_boxes.pkl",
            "plr": args.components / "births_plr_v1" / f"{scene}_boxes.pkl",
            "calr": args.components / "births_calr_v1" / f"{scene}_boxes.pkl",
        }
        rows = {name: read(path) for name, path in paths.items()}
        native_boxes = np.asarray([row[1] for row in rows["native"]])
        mvsr_boxes = np.asarray([row[1] for row in rows["mvsr"]])
        if native_boxes.shape != mvsr_boxes.shape or not np.array_equal(native_boxes, mvsr_boxes):
            raise RuntimeError(f"MVSR changed native geometry/count: {scene}")
        if [row[0] for row in rows["native"]] != [row[0] for row in rows["mvsr"]]:
            raise RuntimeError(f"MVSR changed native labels: {scene}")

        for arm, (use_plr, use_calr, use_mvsr) in ARMS.items():
            output = list(rows["mvsr"] if use_mvsr else rows["native"])
            if use_plr:
                output.extend(rows["plr"])
            if use_calr:
                output.extend(rows["calr"])
            write(args.output / arm / f"{scene}_boxes.pkl", output)
            counts[arm] += len(output)
        component_counts["native"] += len(rows["native"])
        component_counts["plr"] += len(rows["plr"])
        component_counts["calr"] += len(rows["calr"])
        per_scene[scene] = {name: len(value) for name, value in rows.items()}
        for path in paths.values():
            hashes[str(path.resolve())] = digest(path)

    manifest = {
        "schema": "boxfusion.recar3d.extra50_factorial.v1",
        "scene_count": EXPECTED_SCENES,
        "single_online_pass": True,
        "within_run_pairing": True,
        "strictly_causal": True,
        "children_used": False,
        "native_geometry_count_labels_fixed_for_mvsr": True,
        "components": {
            "plr": "current PLR-v1",
            "calr": "CALR-v1",
            "mvsr": "MVSR-v2 composite-support max",
        },
        "arm_definitions": {name: list(value) for name, value in ARMS.items()},
        "arm_counts": counts,
        "component_counts": component_counts,
        "per_scene": per_scene,
        "input_sha256": hashes,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"arm_counts": counts, "component_counts": component_counts}, indent=2))


if __name__ == "__main__":
    main()
