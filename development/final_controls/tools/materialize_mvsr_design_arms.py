#!/usr/bin/env python3
"""Materialize fixed-geometry MVSR factor and temporal-aggregation arms."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import pickle

import numpy as np


COMPONENTS = {
    "native": "native_native",
    "only_2d_iou": "native_raw_iou_max",
    "full": "native_max",
    "no_proposal_score": "native_factor_no_proposal_score",
    "no_visibility": "native_factor_no_visibility",
    "no_center_consistency": "native_factor_no_center_consistency",
    "no_scale_consistency": "native_factor_no_scale_consistency",
    "no_overlap_consistency": "native_factor_no_overlap_consistency",
    "latest": "native_latest",
    "mean": "native_mean",
    "ema": "native_ema",
    "top3_mean": "native_top3_mean",
    "maximum": "native_max",
}


def read(path: Path):
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    if not isinstance(payload, (tuple, list)) or len(payload) != 1:
        raise RuntimeError(f"invalid prediction payload: {path}")
    return [
        (int(label), np.asarray(box, dtype=np.float64), float(score))
        for label, box, score in payload[0]
    ]


def write(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump([[tuple(row) for row in rows]], handle)


def geometry(rows):
    return [(label, box.tobytes()) for label, box, _ in rows]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-list", type=Path, required=True)
    parser.add_argument("--components", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-scenes", type=int, default=100)
    args = parser.parse_args()
    scenes = [
        row.strip() for row in args.scene_list.read_text().splitlines()
        if row.strip() and not row.lstrip().startswith("#")
    ][: args.expected_scenes]
    if len(scenes) != args.expected_scenes:
        raise RuntimeError("scene-count mismatch")
    manifest_path = args.output / "manifest.json"
    if manifest_path.exists():
        raise RuntimeError("output manifest exists; refusing to overwrite")

    counts = {name: 0 for name in COMPONENTS}
    hashes = {}
    for scene in scenes:
        rows = {}
        for name, component in COMPONENTS.items():
            path = args.components / component / f"{scene}_boxes.pkl"
            rows[name] = read(path)
            hashes[str(path.resolve())] = digest(path)
        reference = geometry(rows["native"])
        for name, value in rows.items():
            if geometry(value) != reference:
                raise RuntimeError(f"{name} changed native geometry in {scene}")
            write(args.output / name / f"{scene}_boxes.pkl", value)
            counts[name] += len(value)

    manifest = {
        "schema": "boxfusion.mvsr_design_ablation.v1",
        "scene_count": len(scenes),
        "single_online_pass": True,
        "strictly_causal": True,
        "native_geometry_count_labels_fixed": True,
        "top_q": 3,
        "arm_counts": counts,
        "input_sha256": hashes,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(counts, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
