#!/usr/bin/env python3
"""Rebuild the ScanNet key controls with the final no-visibility MVSR scores."""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np


Row = tuple[int, np.ndarray, float]


def read(path: Path) -> list[Row]:
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    if not isinstance(payload, (list, tuple)) or len(payload) != 1:
        raise RuntimeError(f"invalid prediction container: {path}")
    return [
        (int(label), np.asarray(box, dtype=np.float64).reshape(8, 3), float(score))
        for label, box, score in payload[0]
    ]


def write(path: Path, rows: list[Row]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(
            [[(label, np.array(box, copy=True), score) for label, box, score in rows]],
            handle,
        )


def same_rows(left: list[Row], right: list[Row]) -> bool:
    return len(left) == len(right) and all(
        a[0] == b[0] and np.array_equal(a[1], b[1]) and a[2] == b[2]
        for a, b in zip(left, right)
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-list", type=Path, required=True)
    parser.add_argument("--new-components", type=Path, required=True)
    parser.add_argument("--new-factorial", type=Path, required=True)
    parser.add_argument("--old-components", type=Path, required=True)
    parser.add_argument("--old-key-arms", type=Path, required=True)
    parser.add_argument("--old-calr-arms", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    scenes = [
        line.strip()
        for line in args.scene_list.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if len(scenes) != 100 or len(set(scenes)) != 100:
        raise RuntimeError("expected the unique ScanNet official-100 scene list")
    if (args.output / "manifest.json").exists():
        raise RuntimeError("output manifest exists; refusing to overwrite")

    arms = ("plr_direct", "plr_current", "calr_lower", "calr_simple", "calr_full")
    totals = {arm: 0 for arm in arms}
    birth_totals = {"plr": 0, "calr": 0}
    per_scene: dict[str, dict[str, int]] = {}

    for scene in scenes:
        filename = f"{scene}_boxes.pkl"
        native = read(args.new_components / "native_factor_no_visibility" / filename)
        plr = read(args.new_components / "births_plr_v1" / filename)
        full = read(args.new_factorial / "full" / filename)
        calr = full[len(native) + len(plr) :]
        if not same_rows(full, native + plr + calr):
            raise RuntimeError(f"new full decomposition failed: {scene}")

        old_plr = read(args.old_components / "births_plr_v1" / filename)
        direct = read(args.old_components / "births_plr_direct" / filename)
        if len(plr) != len(old_plr) or len(direct) != len(plr):
            raise RuntimeError(f"PLR per-scene budget mismatch: {scene}")

        old_prefix = read(args.old_key_arms / "calr_prefix" / filename)
        lower_full = read(args.old_calr_arms / "lower_threshold" / filename)
        simple_full = read(args.old_calr_arms / "strongest_matched_budget" / filename)
        lower = lower_full[len(old_prefix) :]
        simple = simple_full[len(old_prefix) :]
        if len(lower) != len(calr) or len(simple) != len(calr):
            raise RuntimeError(f"CALR per-scene budget mismatch: {scene}")

        outputs = {
            # PLR controls share final MVSR and final online CALR-v1.
            "plr_direct": native + direct + calr,
            "plr_current": full,
            # CALR controls share final MVSR and final online PLR.
            "calr_lower": native + plr + lower,
            "calr_simple": native + plr + simple,
            "calr_full": full,
        }
        for arm, rows in outputs.items():
            write(args.output / arm / filename, rows)
            totals[arm] += len(rows)
        birth_totals["plr"] += len(plr)
        birth_totals["calr"] += len(calr)
        per_scene[scene] = {
            "native": len(native),
            "plr": len(plr),
            "calr": len(calr),
        }

    if not all(totals[arm] == totals["plr_current"] for arm in arms):
        raise RuntimeError(f"aggregate output budgets differ: {totals}")

    manifest = {
        "schema": "boxfusion.scannet100.novis_key_controls.v1",
        "scene_count": 100,
        "mvsr": "composite maximum without visibility",
        "plr": "v1",
        "calr": "v1",
        "other_recovery_branch_enabled": True,
        "matched_budget_per_scene": True,
        "controls_use_ground_truth": False,
        "arm_counts": totals,
        "added_boxes": birth_totals,
        "per_scene": per_scene,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
