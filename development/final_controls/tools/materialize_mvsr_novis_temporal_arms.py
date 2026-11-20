#!/usr/bin/env python3
"""Materialize fixed-geometry temporal arms for final three-factor MVSR."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import pickle

import numpy as np


COMPONENTS = {
    "native": "native_native",
    "latest": "native_factor_no_visibility_latest",
    "mean": "native_factor_no_visibility_mean",
    "ema": "native_factor_no_visibility_ema",
    "top3_mean": "native_factor_no_visibility_top3_mean",
    "maximum": "native_factor_no_visibility_max",
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
        row.strip()
        for row in args.scene_list.read_text(encoding="utf-8").splitlines()
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
        "schema": "boxfusion.mvsr_no_visibility_temporal.v1",
        "scene_count": len(scenes),
        "single_online_pass": True,
        "strictly_causal": True,
        "support_factors": [
            "proposal_score", "2d_overlap", "3d_geometry_consistency"
        ],
        "visibility_included": False,
        "native_geometry_count_labels_fixed": True,
        "top_q": 3,
        "arm_counts": counts,
        "input_sha256": hashes,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(counts, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
