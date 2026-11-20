#!/usr/bin/env python3
"""Materialize the canonical 1,787-native-box ScanNet factorial.

Base, MVSR, and PLR are taken from the same 1,787-box online pass.  The
selected CALR-v1 records are recovered from the frozen 1,785-box factorial.
This reuse is valid because CALR-v1 neither reads nor deduplicates against the
native map; the manifest records the cross-run provenance explicitly.
"""
from __future__ import annotations

import argparse
import hashlib
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
        pickle.dump(
            [[(label, np.array(box, copy=True), score) for label, box, score in rows]],
            handle,
        )


def same(left, right) -> bool:
    return len(left) == len(right) and all(
        a[0] == b[0] and np.array_equal(a[1], b[1]) and a[2] == b[2]
        for a, b in zip(left, right)
    )


def same_geometry(left, right) -> bool:
    return len(left) == len(right) and all(
        a[0] == b[0] and np.array_equal(a[1], b[1])
        for a, b in zip(left, right)
    )


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            value.update(chunk)
    return value.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-list", type=Path, required=True)
    parser.add_argument("--components-1787", type=Path, required=True)
    parser.add_argument("--frozen-factorial-1785", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    scenes = [
        row.strip()
        for row in args.scene_list.read_text(encoding="utf-8").splitlines()
        if row.strip() and not row.lstrip().startswith("#")
    ]
    if len(scenes) != 100 or len(set(scenes)) != 100:
        raise RuntimeError("expected the unique ScanNet official100 scene list")
    if args.output.exists() and any(args.output.iterdir()):
        raise RuntimeError(f"refusing to overwrite non-empty output: {args.output}")

    counts = {name: 0 for name in ARMS}
    component_counts = {"native": 0, "plr": 0, "calr": 0}
    per_scene: dict[str, dict[str, int]] = {}
    hashes: dict[str, str] = {}

    for scene in scenes:
        new_paths = {
            "native": args.components_1787 / "native_native" / f"{scene}_boxes.pkl",
            "mvsr": (
                args.components_1787
                / "native_factor_no_visibility_max"
                / f"{scene}_boxes.pkl"
            ),
            "plr": args.components_1787 / "births_plr_v1" / f"{scene}_boxes.pkl",
        }
        old_paths = {
            "prefix": args.frozen_factorial_1785 / "plr_mvsr" / f"{scene}_boxes.pkl",
            "full": args.frozen_factorial_1785 / "full" / f"{scene}_boxes.pkl",
        }
        rows = {name: read(path) for name, path in new_paths.items()}
        old_prefix = read(old_paths["prefix"])
        old_full = read(old_paths["full"])
        if len(old_full) < len(old_prefix) or not same(
            old_prefix, old_full[: len(old_prefix)]
        ):
            raise RuntimeError(f"old full does not preserve PLR+MVSR prefix: {scene}")
        rows["calr"] = old_full[len(old_prefix) :]

        if not same_geometry(rows["native"], rows["mvsr"]):
            raise RuntimeError(f"MVSR changed native geometry/count/labels: {scene}")

        for component in ("native", "mvsr", "plr", "calr"):
            write(
                args.output / "components" / component / f"{scene}_boxes.pkl",
                rows[component],
            )

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
        for path in (*new_paths.values(), *old_paths.values()):
            hashes[str(path.resolve())] = digest(path)

    if component_counts != {"native": 1787, "plr": 1101, "calr": 23172}:
        raise RuntimeError(f"unexpected component counts: {component_counts}")

    manifest = {
        "schema": "boxfusion.scannet100.canonical_1787_factorial.v1",
        "scene_count": 100,
        "strictly_causal": True,
        "native_box_count": 1787,
        "native_plr_same_online_pass": True,
        "native_mvsr_same_online_pass": True,
        "calr_v1_cross_run_reuse": True,
        "calr_v1_native_map_independent": True,
        "support_factors": [
            "proposal_score",
            "2d_overlap",
            "3d_geometry_consistency",
        ],
        "visibility_included": False,
        "temporal_aggregation": "historical maximum",
        "arm_definitions": {name: list(value) for name, value in ARMS.items()},
        "arm_counts": counts,
        "component_counts": component_counts,
        "per_scene": per_scene,
        "input_sha256": hashes,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"arm_counts": counts, "component_counts": component_counts}, indent=2))


if __name__ == "__main__":
    main()
