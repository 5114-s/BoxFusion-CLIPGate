#!/usr/bin/env python3
"""Materialize matched MVSR/CALR/PLR screening arms from one online pass."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import pickle

import numpy as np


MODES = ("native", "first", "mean", "max", "ema", "diverse_max", "reliability")


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
        pickle.dump([[ (label, np.array(box, copy=True), score) for label, box, score in rows ]], handle)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene-list", type=Path, required=True)
    parser.add_argument("--components", type=Path, required=True)
    parser.add_argument("--v4-factorial", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    scenes = [line.strip() for line in args.scene_list.read_text().splitlines()
              if line.strip() and not line.lstrip().startswith("#")]
    if len(scenes) != 100 or len(set(scenes)) != 100:
        raise RuntimeError("expected official100 scene list")

    arms = ["mvsr_v4_rawmax"] + [f"mvsr_{mode}" for mode in MODES]
    arms += ["calr_v1_full", "calr_v2_full", "calr_v1_matched", "calr_v2_matched"]
    arms += [f"calr_v1_mvsr_{mode}" for mode in MODES]
    arms += [f"calr_v2_mvsr_{mode}" for mode in MODES]
    arms += [f"calr_v1_mvsr_{mode}_matched" for mode in MODES]
    arms += [f"calr_v2_mvsr_{mode}_matched" for mode in MODES]
    arms += [f"calr_v2_mvsr_{mode}_plr_v1" for mode in MODES]
    counts = {arm: 0 for arm in arms}
    component_counts = {name: 0 for name in (
        "births_calr_v1", "births_calr_v2", "births_plr_v1"
    )}
    hashes: dict[str, str] = {}
    per_scene = {}
    for scene in scenes:
        native = {
            mode: read(args.components / f"native_{mode}" / f"{scene}_boxes.pkl")
            for mode in MODES
        }
        v1 = read(args.components / "births_calr_v1" / f"{scene}_boxes.pkl")
        v2 = read(args.components / "births_calr_v2" / f"{scene}_boxes.pkl")
        plr = read(args.components / "births_plr_v1" / f"{scene}_boxes.pkl")
        v4_base = read(args.v4_factorial / "base" / f"{scene}_boxes.pkl")
        v4_rawmax = read(args.v4_factorial / "m2" / f"{scene}_boxes.pkl")
        boxes = np.asarray([row[1] for row in native["native"]])
        base_boxes = np.asarray([row[1] for row in v4_base])
        base_scores = np.asarray([row[2] for row in v4_base])
        current_scores = np.asarray([row[2] for row in native["native"]])
        if (
            boxes.shape != base_boxes.shape
            or not np.allclose(boxes, base_boxes, rtol=0.0, atol=1e-6)
            or not np.allclose(current_scores, base_scores, rtol=0.0, atol=1e-8)
        ):
            raise RuntimeError(f"v4/v2 native base mismatch: {scene}")
        rawmax_boxes = np.asarray([row[1] for row in v4_rawmax])
        if boxes.shape != rawmax_boxes.shape or not np.allclose(
            boxes, rawmax_boxes, rtol=0.0, atol=1e-6
        ):
            raise RuntimeError(f"v4 raw-max geometry/count mismatch: {scene}")
        for mode in MODES:
            mode_boxes = np.asarray([row[1] for row in native[mode]])
            if boxes.shape != mode_boxes.shape or not np.array_equal(boxes, mode_boxes):
                raise RuntimeError(f"MVSR geometry changed: {scene}/{mode}")
        matched = min(len(v1), len(v2))
        scene_arms = {"mvsr_v4_rawmax": v4_rawmax}
        scene_arms.update({f"mvsr_{mode}": native[mode] for mode in MODES})
        scene_arms.update({
            "calr_v1_full": native["native"] + v1,
            "calr_v2_full": native["native"] + v2,
            "calr_v1_matched": native["native"] + v1[:matched],
            "calr_v2_matched": native["native"] + v2[:matched],
        })
        for mode in MODES:
            scene_arms[f"calr_v1_mvsr_{mode}"] = native[mode] + v1
            scene_arms[f"calr_v2_mvsr_{mode}"] = native[mode] + v2
            scene_arms[f"calr_v1_mvsr_{mode}_matched"] = native[mode] + v1[:matched]
            scene_arms[f"calr_v2_mvsr_{mode}_matched"] = native[mode] + v2[:matched]
            scene_arms[f"calr_v2_mvsr_{mode}_plr_v1"] = native[mode] + v2 + plr
        for arm, rows in scene_arms.items():
            write(args.output / arm / f"{scene}_boxes.pkl", rows)
            counts[arm] += len(rows)
        component_counts["births_calr_v1"] += len(v1)
        component_counts["births_calr_v2"] += len(v2)
        component_counts["births_plr_v1"] += len(plr)
        per_scene[scene] = {
            "native": len(native["native"]), "calr_v1": len(v1),
            "calr_v2": len(v2), "plr_v1": len(plr), "matched_calr": matched,
        }
        for name in [*(f"native_{mode}" for mode in MODES),
                     "births_calr_v1", "births_calr_v2", "births_plr_v1"]:
            path = args.components / name / f"{scene}_boxes.pkl"
            hashes[str(path.resolve())] = digest(path)
    report = {
        "schema": "boxfusion.reliability_v2.screening.v1",
        "scene_count": len(scenes),
        "scenes": scenes,
        "single_online_pass": True,
        "v4_native_base_parity_required": True,
        "mvsr_geometry_and_count_fixed": True,
        "cross_branch_dedup": False,
        "component_counts": component_counts,
        "arm_counts": counts,
        "per_scene": per_scene,
        "input_sha256": hashes,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "manifest.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"component_counts": component_counts, "arm_counts": counts}, indent=2))


if __name__ == "__main__":
    main()
