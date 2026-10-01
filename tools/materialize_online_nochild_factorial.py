#!/usr/bin/env python3
"""Materialize the paired 2x2x2 M1-P/M1-A/M2 arms from one online run."""

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
    "p_a": (True, True, False),
    "m2": (False, False, True),
    "p_m2": (True, False, True),
    "a_m2": (False, True, True),
    "p_a_m2": (True, True, True),
}


def read(path: Path) -> list[tuple[int, np.ndarray, float]]:
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    if not isinstance(payload, (list, tuple)) or len(payload) != 1:
        raise RuntimeError(f"invalid prediction container: {path}")
    rows = []
    for row in payload[0]:
        if len(row) != 3 or int(row[0]) != 0:
            raise RuntimeError(f"invalid class-agnostic row: {path}")
        box = np.asarray(row[1], dtype=np.float64).reshape(8, 3)
        score = float(row[2])
        if not np.isfinite(box).all() or not np.isfinite(score):
            raise RuntimeError(f"non-finite prediction row: {path}")
        rows.append((0, box, score))
    return rows


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            value.update(chunk)
    return value.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-list", type=Path, required=True)
    parser.add_argument("--native", type=Path, required=True)
    parser.add_argument("--full", type=Path, required=True)
    parser.add_argument("--diagnostics", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-scenes", type=int, default=100)
    args = parser.parse_args()

    scenes = [line.strip() for line in args.scene_list.read_text().splitlines()
              if line.strip() and not line.lstrip().startswith("#")]
    if len(scenes) != args.expected_scenes or len(set(scenes)) != len(scenes):
        raise RuntimeError(
            f"paired materialization requires {args.expected_scenes} unique scenes"
        )
    for arm in ARMS:
        (args.output / arm).mkdir(parents=True, exist_ok=True)

    totals = {arm: 0 for arm in ARMS}
    components = {"native": 0, "m1p": 0, "m1a": 0}
    score_changes = 0
    hashes: dict[str, str] = {}
    per_scene = {}
    for scene in scenes:
        native_path = args.native / f"{scene}_boxes.pkl"
        full_path = args.full / f"{scene}_boxes.pkl"
        diagnostic_path = args.diagnostics / f"{scene}.json"
        native = read(native_path)
        full = read(full_path)
        diagnostic = json.loads(diagnostic_path.read_text())
        if diagnostic["state"]["m1p"]["use_children"] is not False:
            raise RuntimeError(f"child evidence enabled: {scene}")
        if diagnostic.get("scene_end_inference") is not False:
            raise RuntimeError(f"scene-end inference detected: {scene}")
        if diagnostic.get("scene_end_assembly") is not True:
            raise RuntimeError(f"terminal assembly missing: {scene}")
        counts = diagnostic["state"]["terminal_counts"]
        n, p, a = (int(counts[key]) for key in ("native", "m1p", "m1a"))
        if n != len(native) or n + p + a != len(full):
            raise RuntimeError(f"component count mismatch: {scene}")
        native_m2, births_p, births_a = full[:n], full[n:n + p], full[n + p:]
        if n:
            native_boxes = np.stack([row[1] for row in native])
            m2_boxes = np.stack([row[1] for row in native_m2])
            if not np.array_equal(native_boxes, m2_boxes):
                raise RuntimeError(f"M2 changed native geometry: {scene}")
        score_changes += sum(left[2] != right[2]
                             for left, right in zip(native, native_m2))
        for arm, (with_p, with_a, with_m2) in ARMS.items():
            rows = list(native_m2 if with_m2 else native)
            if with_p:
                rows.extend(births_p)
            if with_a:
                rows.extend(births_a)
            destination = args.output / arm / f"{scene}_boxes.pkl"
            with destination.open("wb") as handle:
                pickle.dump([rows], handle)
            totals[arm] += len(rows)
        components["native"] += n
        components["m1p"] += p
        components["m1a"] += a
        per_scene[scene] = {"native": n, "m1p": p, "m1a": a}
        for path in (native_path, full_path, diagnostic_path):
            hashes[str(path.resolve())] = digest(path)

    report = {
        "schema": "boxfusion.strict_causal_nochild.factorial.v1",
        "scene_count": len(scenes),
        "paired_from_one_run": True,
        "children_enabled": False,
        "components": components,
        "arm_counts": totals,
        "native_scores_changed_by_m2": score_changes,
        "per_scene": per_scene,
        "input_sha256": hashes,
    }
    (args.output / "manifest.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps({"components": components, "arm_counts": totals,
                      "native_scores_changed_by_m2": score_changes}, indent=2))


if __name__ == "__main__":
    main()
