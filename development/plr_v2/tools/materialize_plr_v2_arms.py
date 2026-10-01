#!/usr/bin/env python3
"""Materialize matched PLR stages on the frozen CALR-v1 + MVSR-v2-max prefix."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import pickle

import numpy as np


PLR_COMPONENTS = {
    "current": "births_plr_v1",
    "assoc": "births_plr_assoc",
    "reliability": "births_plr_reliability",
    "score": "births_plr_score",
}


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
        pickle.dump(
            [[(label, np.array(box, copy=True), score) for label, box, score in rows]],
            handle,
        )


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene-list", type=Path, required=True)
    parser.add_argument("--components", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    scenes = [
        line.strip()
        for line in args.scene_list.read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if len(scenes) != 100 or len(set(scenes)) != 100:
        raise RuntimeError("expected the unique official100 scene list")

    arms = ["prefix", *(f"prefix_plr_{name}" for name in PLR_COMPONENTS)]
    counts = {name: 0 for name in arms}
    component_counts = {"native_max": 0, "births_calr_v1": 0}
    component_counts.update({value: 0 for value in PLR_COMPONENTS.values()})
    hashes: dict[str, str] = {}
    per_scene = {}
    for scene in scenes:
        native_path = args.components / "native_max" / f"{scene}_boxes.pkl"
        calr_path = args.components / "births_calr_v1" / f"{scene}_boxes.pkl"
        native = read(native_path)
        calr = read(calr_path)
        plr = {
            name: read(args.components / component / f"{scene}_boxes.pkl")
            for name, component in PLR_COMPONENTS.items()
        }
        prefix = native + calr
        scene_arms = {"prefix": prefix}
        scene_arms.update(
            {f"prefix_plr_{name}": prefix + rows for name, rows in plr.items()}
        )
        for arm, rows in scene_arms.items():
            write(args.output / arm / f"{scene}_boxes.pkl", rows)
            counts[arm] += len(rows)
        component_counts["native_max"] += len(native)
        component_counts["births_calr_v1"] += len(calr)
        for name, component in PLR_COMPONENTS.items():
            component_counts[component] += len(plr[name])
        per_scene[scene] = {
            "native": len(native),
            "calr_v1": len(calr),
            **{f"plr_{name}": len(rows) for name, rows in plr.items()},
        }
        for component in ("native_max", "births_calr_v1", *PLR_COMPONENTS.values()):
            path = args.components / component / f"{scene}_boxes.pkl"
            hashes[str(path.resolve())] = digest(path)

    payload = {
        "schema": "boxfusion.plr_v2.matched_arms.v1",
        "scene_count": len(scenes),
        "scenes": scenes,
        "single_online_pass": True,
        "within_run_pairing": True,
        "fixed_prefix": "native_max + calr_v1",
        "post_nms_proposals_only": True,
        "children_used": False,
        "max_plr_outputs_per_scene": 12,
        "cross_branch_dedup": False,
        "component_counts": component_counts,
        "arm_counts": counts,
        "per_scene": per_scene,
        "input_sha256": hashes,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "manifest.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"component_counts": component_counts, "arm_counts": counts}, indent=2))


if __name__ == "__main__":
    main()

