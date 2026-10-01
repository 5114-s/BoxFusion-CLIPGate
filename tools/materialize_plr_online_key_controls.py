#!/usr/bin/env python3
"""Pair final-online PLR key controls on one canonical native map."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import pickle

import numpy as np


def read(path: Path) -> list[tuple[int, np.ndarray, float]]:
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    if not isinstance(payload, (list, tuple)) or len(payload) != 1:
        raise RuntimeError(f"invalid prediction container: {path}")
    rows = []
    for row in payload[0]:
        if len(row) != 3 or int(row[0]) != 0:
            raise RuntimeError(f"invalid class-agnostic prediction: {path}")
        rows.append((0, np.asarray(row[1], dtype=np.float64).reshape(8, 3),
                     float(row[2])))
    return rows


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


def same_rows(left, right) -> bool:
    return len(left) == len(right) and all(
        a[0] == b[0]
        and np.array_equal(a[1], b[1])
        and a[2] == b[2]
        for a, b in zip(left, right)
    )


def split(full_path: Path, diagnostic_path: Path):
    rows = read(full_path)
    diagnostic = json.loads(diagnostic_path.read_text(encoding="utf-8"))
    counts = diagnostic["state"]["terminal_counts"]
    n, p, a = (int(counts[key]) for key in ("native", "m1p", "m1a"))
    if n + p + a != len(rows):
        raise RuntimeError(f"component-count mismatch: {full_path}")
    return rows[n:n + p], diagnostic


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-list", type=Path, required=True)
    parser.add_argument("--canonical-native", type=Path, required=True)
    parser.add_argument("--canonical-full", type=Path, required=True)
    parser.add_argument("--canonical-diagnostics", type=Path, required=True)
    parser.add_argument("--direct-native", type=Path, required=True)
    parser.add_argument("--direct-full", type=Path, required=True)
    parser.add_argument("--direct-diagnostics", type=Path, required=True)
    parser.add_argument("--no-dedup-native", type=Path, required=True)
    parser.add_argument("--no-dedup-full", type=Path, required=True)
    parser.add_argument("--no-dedup-diagnostics", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    scenes = [line.strip() for line in args.scene_list.read_text().splitlines()
              if line.strip() and not line.lstrip().startswith("#")]
    if len(scenes) != 100 or len(set(scenes)) != 100:
        raise RuntimeError("expected the official 100 unique ScanNet scenes")

    arms = ("direct_matched_budget", "without_native_dedup", "full_plr")
    for arm in arms:
        (args.output / arm).mkdir(parents=True, exist_ok=True)
    totals = {arm: {"boxes": 0, "births": 0} for arm in arms}
    per_scene = {}
    hashes = {}

    for scene in scenes:
        canonical_native_path = args.canonical_native / f"{scene}_boxes.pkl"
        canonical_native = read(canonical_native_path)
        full_births, full_diag = split(
            args.canonical_full / f"{scene}_boxes.pkl",
            args.canonical_diagnostics / f"{scene}.json",
        )

        variants = {}
        for name, native_root, full_root, diagnostic_root, expected_mode in (
            ("direct_matched_budget", args.direct_native, args.direct_full,
             args.direct_diagnostics, "direct_matched_budget"),
            ("without_native_dedup", args.no_dedup_native, args.no_dedup_full,
             args.no_dedup_diagnostics, "no_native_dedup"),
        ):
            variant_native_path = native_root / f"{scene}_boxes.pkl"
            variant_native = read(variant_native_path)
            if not same_rows(canonical_native, variant_native):
                raise RuntimeError(
                    f"native map changed in {name}: {scene}"
                )
            births, diagnostic = split(
                full_root / f"{scene}_boxes.pkl",
                diagnostic_root / f"{scene}.json",
            )
            state = diagnostic["state"]["m1p"]
            if state.get("ablation_mode") != expected_mode:
                raise RuntimeError(f"wrong PLR mode for {name}: {scene}")
            if state.get("use_children") is not False:
                raise RuntimeError(f"child evidence enabled: {scene}")
            variants[name] = births
            for path in (
                variant_native_path,
                full_root / f"{scene}_boxes.pkl",
                diagnostic_root / f"{scene}.json",
            ):
                hashes[str(path.resolve())] = digest(path)

        if len(variants["direct_matched_budget"]) != len(full_births):
            raise RuntimeError(
                f"matched-budget count differs in {scene}: "
                f"{len(variants['direct_matched_budget'])} vs {len(full_births)}"
            )
        variants["full_plr"] = full_births

        scene_counts = {}
        for arm, births in variants.items():
            output = canonical_native + births
            write(args.output / arm / f"{scene}_boxes.pkl", output)
            totals[arm]["boxes"] += len(output)
            totals[arm]["births"] += len(births)
            scene_counts[arm] = len(births)
        per_scene[scene] = scene_counts
        for path in (
            canonical_native_path,
            args.canonical_full / f"{scene}_boxes.pkl",
            args.canonical_diagnostics / f"{scene}.json",
        ):
            hashes[str(path.resolve())] = digest(path)

    manifest = {
        "schema": "boxfusion.plr.online_key_controls.v1",
        "scene_count": len(scenes),
        "children_enabled": False,
        "canonical_native_map_shared": True,
        "direct_budget_matches_full_per_scene": True,
        "arms": totals,
        "per_scene_births": per_scene,
        "input_sha256": hashes,
    }
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(totals, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
