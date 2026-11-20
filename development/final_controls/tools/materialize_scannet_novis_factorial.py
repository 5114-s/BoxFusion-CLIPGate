#!/usr/bin/env python3
"""Materialize the paired ScanNet no-visibility eight-arm factorial."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import pickle

import numpy as np


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


def read(path: Path):
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    return [
        (int(label), np.asarray(box, dtype=np.float64).reshape(8, 3), float(score))
        for label, box, score in payload[0]
    ]


def write(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump([[tuple(row) for row in rows]], handle)


def same(left, right) -> bool:
    return len(left) == len(right) and all(
        a[0] == b[0] and np.array_equal(a[1], b[1]) and a[2] == b[2]
        for a, b in zip(left, right)
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-list", type=Path, required=True)
    parser.add_argument("--components", type=Path, required=True)
    parser.add_argument("--paired-full-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    scenes = [
        row.strip() for row in args.scene_list.read_text().splitlines()
        if row.strip() and not row.lstrip().startswith("#")
    ]
    if len(scenes) != 100 or (args.output / "manifest.json").exists():
        raise RuntimeError("expected fresh official100 output")
    counts = {name: 0 for name in ARMS}
    components = {name: 0 for name in ("native", "plr", "calr")}
    for scene in scenes:
        native = read(args.components / "native_native" / f"{scene}_boxes.pkl")
        mvsr = read(args.components / "native_factor_no_visibility" / f"{scene}_boxes.pkl")
        plr = read(args.components / "births_plr_v1" / f"{scene}_boxes.pkl")
        paired = read(args.paired_full_root / "four_factor" / f"{scene}_boxes.pkl")
        if len(paired) < len(native) + len(plr):
            raise RuntimeError(f"paired full is too short: {scene}")
        calr = paired[len(native) + len(plr):]
        for arm, (use_plr, use_calr, use_mvsr) in ARMS.items():
            rows = list(mvsr if use_mvsr else native)
            if use_plr:
                rows.extend(plr)
            if use_calr:
                rows.extend(calr)
            write(args.output / arm / f"{scene}_boxes.pkl", rows)
            counts[arm] += len(rows)
        expected = read(args.paired_full_root / "no_visibility" / f"{scene}_boxes.pkl")
        if not same(read(args.output / "full" / f"{scene}_boxes.pkl"), expected):
            raise RuntimeError(f"full reconstruction mismatch: {scene}")
        components["native"] += len(native)
        components["plr"] += len(plr)
        components["calr"] += len(calr)
    manifest = {
        "schema": "boxfusion.scannet100.novis_factorial.v1",
        "scene_count": 100,
        "strictly_causal": True,
        "paired_from_one_online_pass": True,
        "mvsr": "composite maximum without visibility",
        "calr": "CALR-v1",
        "arm_counts": counts,
        "component_counts": components,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
