#!/usr/bin/env python3
"""Finalize and audit the isolated non-canonical CA-1M derived107 overlay."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path

import cv2
import numpy as np


SCHEMA = "boxfusion.ca1m_derived107_overlay.v1"
DERIVED_FLAGS = {
    "derived": True,
    "official_comparable": False,
    "paper_claim_permitted": False,
}


def rows(path: Path) -> tuple[str, ...]:
    values = tuple(line.strip() for line in path.read_text().splitlines() if line.strip())
    if len(values) != len(set(values)):
        raise ValueError(f"duplicate scene IDs in {path}")
    return values


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def audit_scene(scene: Path, expected_id: str) -> dict:
    manifest_path = scene / "derived_gt_manifest.json"
    payload = json.loads(manifest_path.read_text())
    if payload.get("scene_id") != expected_id:
        raise ValueError(f"{expected_id}: manifest scene ID mismatch")
    for key, expected in DERIVED_FLAGS.items():
        if payload.get(key) is not expected:
            raise ValueError(f"{expected_id}: manifest must state {key}={expected}")
    required = (
        "K_depth.txt", "K_rgb.txt", "all_poses.npy", "T_gravity.npy",
        "instances.json", "after_filter_boxes.npy",
    )
    hashes: dict[str, str] = {}
    for name in required:
        path = scene / name
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"{expected_id}: missing regular artifact {name}")
        hashes[name] = sha256(path)
    poses = np.load(scene / "all_poses.npy")
    gravity = np.load(scene / "T_gravity.npy")
    boxes = np.load(scene / "after_filter_boxes.npy")
    rgb = sorted((scene / "rgb").glob("*.png"), key=lambda p: int(p.stem))
    depth = sorted((scene / "depth").glob("*.png"), key=lambda p: int(p.stem))
    instances = sorted((scene / "instances").glob("*.json"), key=lambda p: int(p.stem))
    frame_count = int(payload["counts"]["frames"])
    if not (len(rgb) == len(depth) == len(instances) == len(poses) == len(gravity) == frame_count):
        raise ValueError(f"{expected_id}: RGB-D/pose/gravity/instance count mismatch")
    if boxes.dtype != np.float64 or boxes.ndim != 3 or boxes.shape[1:] != (8, 3):
        raise ValueError(f"{expected_id}: invalid derived GT array {boxes.dtype} {boxes.shape}")
    if len(boxes) != int(payload["counts"]["kept_boxes"]):
        raise ValueError(f"{expected_id}: GT count differs from manifest")
    if hashes["after_filter_boxes.npy"] != payload.get("gt_sha256"):
        raise ValueError(f"{expected_id}: GT SHA256 differs from manifest")
    first_depth = cv2.imread(str(depth[0]), cv2.IMREAD_UNCHANGED)
    first_rgb = cv2.imread(str(rgb[0]), cv2.IMREAD_UNCHANGED)
    if first_depth is None or first_depth.ndim != 2 or first_depth.dtype != np.uint16:
        raise ValueError(f"{expected_id}: invalid first depth frame")
    if first_rgb is None or first_rgb.ndim != 3:
        raise ValueError(f"{expected_id}: invalid first RGB frame")
    if first_rgb.shape[:2] != (2 * first_depth.shape[0], 2 * first_depth.shape[1]):
        raise ValueError(f"{expected_id}: RGB/depth dimensions disagree")
    return {
        "scene_id": expected_id,
        "source_kind": payload["source_kind"],
        "frames": frame_count,
        "raw_boxes": int(payload["counts"]["raw_boxes"]),
        "frustum_boxes": int(payload["counts"]["frustum_boxes"]),
        "derived_gt_boxes": int(len(boxes)),
        "surface_points": int(payload["counts"]["surface_points"]),
        "scene_manifest": str(manifest_path.resolve()),
        "scene_manifest_sha256": sha256(manifest_path),
        "artifact_sha256": hashes,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--overlay-root", type=Path, required=True)
    parser.add_argument("--canonical-root", type=Path, required=True)
    parser.add_argument("--full-list", type=Path, required=True)
    parser.add_argument("--canonical-list", type=Path, required=True)
    parser.add_argument("--derived-list", type=Path, required=True)
    args = parser.parse_args()

    overlay = args.overlay_root.resolve()
    canonical_root = args.canonical_root.resolve()
    full = rows(args.full_list)
    canonical = rows(args.canonical_list)
    derived = rows(args.derived_list)
    if len(full) != 107 or len(canonical) != 103 or len(derived) != 4:
        raise ValueError("scene partition must be exactly 107=103+4")
    if set(full) != set(canonical) | set(derived) or set(canonical) & set(derived):
        raise ValueError("scene partition does not exactly cover official full107")
    overlay.mkdir(parents=True, exist_ok=True)

    for scene_id in canonical:
        source = canonical_root / scene_id
        target = overlay / scene_id
        if not source.is_dir() or source.is_symlink():
            raise ValueError(f"{scene_id}: invalid canonical source")
        if target.is_symlink():
            if target.resolve() != source.resolve():
                raise ValueError(f"{scene_id}: existing overlay link has wrong target")
        elif target.exists():
            raise ValueError(f"{scene_id}: refusing non-symlink canonical overlay entry")
        else:
            target.symlink_to(source, target_is_directory=True)

    derived_reports = []
    for scene_id in derived:
        scene = overlay / scene_id
        if not scene.is_dir() or scene.is_symlink():
            raise ValueError(f"{scene_id}: derived scene must be a physical directory")
        derived_reports.append(audit_scene(scene, scene_id))

    numeric = {
        entry.name for entry in overlay.iterdir()
        if entry.name.isdigit() and (entry.is_dir() or entry.is_symlink())
    }
    if numeric != set(full):
        raise ValueError(
            f"overlay exact-set mismatch: missing={sorted(set(full)-numeric)}, "
            f"unexpected={sorted(numeric-set(full))}"
        )
    payload = {
        "schema": SCHEMA,
        **DERIVED_FLAGS,
        "protocol": "derived107_noncanonical_internal_diagnostic",
        "scene_count": 107,
        "canonical_scene_count": 103,
        "derived_scene_count": 4,
        "official_gt_scenes": 103,
        "derived_gt_scenes": list(derived),
        "overlay_root": str(overlay),
        "canonical_root": str(canonical_root),
        "scene_lists": {
            "full107": {"path": str(args.full_list.resolve()), "sha256": sha256(args.full_list)},
            "canonical103": {"path": str(args.canonical_list.resolve()), "sha256": sha256(args.canonical_list)},
            "derived4": {"path": str(args.derived_list.resolve()), "sha256": sha256(args.derived_list)},
        },
        "proxy_validation": {
            "reference_scenes": ["42898811", "42897552", "47333923"],
            "precision": 0.98084,
            "recall": 0.99225,
            "jaccard": 0.97338,
            "disclosure": "reference agreement does not make derived GT author GT",
        },
        "derived_scene_reports": derived_reports,
        "scene_manifests": {
            row["scene_id"]: {
                "path": row["scene_manifest"],
                "sha256": row["scene_manifest_sha256"],
            }
            for row in derived_reports
        },
    }
    manifest = overlay / "derived_gt_manifest.json"
    atomic_json(manifest, payload)
    print(json.dumps({
        "ok": True,
        "scene_count": 107,
        "canonical_symlinks": 103,
        "derived_scene_counts": {
            row["scene_id"]: row["derived_gt_boxes"] for row in derived_reports
        },
        "manifest": str(manifest),
        **DERIVED_FLAGS,
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
