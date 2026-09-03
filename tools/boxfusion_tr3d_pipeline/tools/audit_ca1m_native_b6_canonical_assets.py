#!/usr/bin/env python3
"""Audit historical canonical103 C0 and legacy cache without using either."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import tempfile
from pathlib import Path

import numpy as np


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def scenes(path: Path) -> list[str]:
    rows = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    if len(rows) != 103 or len(rows) != len(set(rows)) or any(not row.isdigit() for row in rows):
        raise ValueError("expected frozen canonical103 scene list")
    return rows


def scores(path: Path) -> list[float]:
    with path.open("rb") as handle:
        value = pickle.load(handle)
        if handle.read(1):
            raise ValueError(f"trailing bytes in {path}")
    if not isinstance(value, list) or len(value) != 1 or not isinstance(value[0], list):
        raise ValueError(f"invalid prediction batch: {path}")
    result = []
    for row in value[0]:
        if not isinstance(row, tuple) or len(row) != 3:
            raise ValueError(f"invalid prediction row: {path}")
        label, corners, score = row
        corners = np.asarray(corners)
        score = float(score)
        if int(label) != 0 or corners.shape != (8, 3) or not np.isfinite(corners).all() or not 0 <= score <= 1:
            raise ValueError(f"invalid prediction value: {path}")
        result.append(score)
    return result


def write(path: Path, value: dict) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-list", type=Path, required=True)
    parser.add_argument("--c0-root", type=Path, required=True)
    parser.add_argument("--legacy-cache-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    scene_rows = scenes(args.scene_list)
    expected = {f"{scene}_boxes.pkl" for scene in scene_rows}
    actual = {path.name for path in args.c0_root.glob("*_boxes.pkl") if path.is_file()}
    if actual != expected:
        raise ValueError("historical C0 does not contain exactly canonical103")
    digest = hashlib.sha256()
    all_scores: list[float] = []
    for scene in scene_rows:
        path = args.c0_root / f"{scene}_boxes.pkl"
        digest.update(f"{scene}\t{sha256(path)}\n".encode())
        all_scores.extend(scores(path))
    if len(all_scores) < 2 or np.ptp(np.asarray(all_scores)) <= 0:
        raise ValueError("historical C0 does not contain varying real scores")

    cached = sorted(
        path.name for path in args.legacy_cache_root.iterdir()
        if path.is_dir() and not path.is_symlink() and path.name.isdigit()
    ) if args.legacy_cache_root.is_dir() else []
    cache_fingerprints: set[str] = set()
    for scene in cached:
        manifest_path = args.legacy_cache_root / scene / "manifest.json"
        value = json.loads(manifest_path.read_text())
        if value.get("scene_id") != scene or value.get("namespace") != args.legacy_cache_root.name:
            raise ValueError(f"legacy cache manifest drift: {manifest_path}")
        cache_fingerprints.add(str(value.get("producer_fingerprint")))
    report = {
        "schema": "boxfusion.ca1m_native_b6_canonical103_existing_asset_audit.v1",
        "ok": True,
        "ground_truth_access": False,
        "evaluation_invoked": False,
        "historical_c0": {
            "role": "audit_only_not_consumed_by_new_collection",
            "scenes": 103,
            "prediction_rows": len(all_scores),
            "score_min": float(min(all_scores)),
            "score_max": float(max(all_scores)),
            "real_score_variation": True,
            "collection_sha256": digest.hexdigest(),
        },
        "legacy_cache": {
            "role": "audit_only_never_reused",
            "canonical103_coverage": len(set(cached) & set(scene_rows)),
            "scene_directories": len(cached),
            "producer_fingerprints": sorted(cache_fingerprints),
            "reusable_for_new_collection": False,
            "reason": "incomplete coverage and mandatory independent namespace",
        },
    }
    write(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
