#!/usr/bin/env python3
"""Fail-closed audit of one Apple-tar-derived CA-1M processed scene."""

from __future__ import annotations

import argparse
import json
import os
import tarfile
import math
from pathlib import Path

import cv2
import numpy as np

from convert_ca1m_apple_tar import (
    HF_REQUIRED,
    SCHEMA,
    atomic_json,
    decode_png,
    depth_shape,
    expected_raw_metadata,
    frame_ids,
    infer_rot90,
    infer_rot90_aspect_clockwise_to_portrait,
    load_orientation_policy,
    member_bytes,
    parse_json_member,
    sha256,
    validate_hf_metadata,
    validate_poses,
)


AUDIT_SCHEMA = "boxfusion.ca1m_apple_conversion_audit.v1"


def numeric_pngs(directory: Path) -> tuple[Path, ...]:
    if not directory.is_dir() or directory.is_symlink():
        raise ValueError(f"invalid image directory: {directory}")
    rows = []
    for path in directory.iterdir():
        if path.suffix.lower() != ".png":
            continue
        if not path.is_file() or path.is_symlink() or not path.stem.isdigit():
            raise ValueError(f"invalid PNG entry: {path}")
        rows.append(path)
    return tuple(sorted(rows, key=lambda path: int(path.stem)))


def audit(args: argparse.Namespace) -> dict:
    scene_dir = args.scene_dir.resolve()
    tar_path = args.tar.resolve()
    metadata_scene = args.metadata_scene.resolve()
    scene_id = scene_dir.name
    if not scene_dir.is_dir() or scene_dir.is_symlink():
        raise ValueError(f"invalid processed scene directory: {scene_dir}")
    if metadata_scene.name != scene_id:
        raise ValueError("scene/metadata IDs disagree")
    for name in (*HF_REQUIRED, "conversion_manifest.json"):
        path = scene_dir / name
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"missing/non-regular converted file: {path}")
    manifest = json.loads((scene_dir / "conversion_manifest.json").read_text())
    if manifest.get("schema") != SCHEMA or manifest.get("scene_id") != scene_id:
        raise ValueError("conversion manifest schema/scene mismatch")
    if sha256(tar_path) != manifest.get("apple_tar_sha256"):
        raise ValueError("Apple tar SHA256 differs from conversion manifest")
    orientation_rule, orientation_policy = load_orientation_policy(
        args.orientation_policy, scene_id, tar_path
    )
    if manifest.get("orientation_policy") != orientation_policy:
        raise ValueError("orientation policy provenance differs from conversion manifest")
    if manifest.get("orientation_rule") != orientation_rule:
        raise ValueError("orientation rule differs from conversion manifest")

    rgb_paths = numeric_pngs(scene_dir / "rgb")
    depth_paths = numeric_pngs(scene_dir / "depth")
    with tarfile.open(tar_path, mode="r:") as archive:
        ids = frame_ids(archive, scene_id)
        expected_names = tuple(range(len(ids)))
        if tuple(int(path.stem) for path in rgb_paths) != expected_names:
            raise ValueError("RGB frame indices are not contiguous 0..N-1")
        if tuple(int(path.stem) for path in depth_paths) != expected_names:
            raise ValueError("depth frame indices are not contiguous 0..N-1")
        shapes = tuple(depth_shape(archive, scene_id, frame_id) for frame_id in ids)
        poses = np.stack(
            [
                parse_json_member(archive, f"{scene_id}/{frame_id}.gt/RT.json")
                for frame_id in ids
            ]
        )
        validate_poses(poses, scene_id)
        if orientation_rule["method"] == "pose_continuity":
            rotations, orientation = infer_rot90(
                poses, shapes, float(orientation_rule["min_margin_degrees"])
            )
            orientation["method"] = "pose_continuity"
        else:
            rotations, orientation = infer_rot90_aspect_clockwise_to_portrait(shapes)
        if orientation != manifest.get("orientation"):
            raise ValueError("recomputed cardinal rotations differ from manifest")
        expected = expected_raw_metadata(
            archive,
            scene_id,
            ids,
            shapes,
            orientation["target_orientation"],
        )
        metadata_hashes = validate_hf_metadata(metadata_scene, expected, scene_id)
        for name in HF_REQUIRED:
            if sha256(scene_dir / name) != metadata_hashes[name]:
                raise ValueError(f"converted {name} is not a byte copy of HF metadata")

        if args.pixel_check == "all":
            indices = range(len(ids))
        elif args.pixel_check == "sample":
            indices = sorted(
                set(
                    [0, len(ids) // 2, len(ids) - 1]
                    + [
                        index
                        for index in range(1, len(ids))
                        if rotations[index] != rotations[index - 1]
                    ]
                )
            )
        else:
            indices = ()
        pixel_rows = 0
        for index in indices:
            frame_id = ids[index]
            k = int(rotations[index])
            for suffix, output in (
                (".wide/image.png", rgb_paths[index]),
                (".gt/depth.png", depth_paths[index]),
            ):
                raw = member_bytes(archive, f"{scene_id}/{frame_id}{suffix}")
                observed = output.read_bytes()
                if k == 0:
                    if raw != observed:
                        raise ValueError(f"{output}: k=0 is not a byte-exact raw copy")
                else:
                    raw_image = decode_png(raw, str(output))
                    output_image = decode_png(observed, str(output))
                    if not np.array_equal(np.rot90(raw_image, k), output_image):
                        raise ValueError(f"{output}: rotated pixels disagree with raw tar")
                pixel_rows += 1

    report = {
        "schema": AUDIT_SCHEMA,
        "ok": True,
        "scene_id": scene_id,
        "scene_dir": str(scene_dir),
        "apple_tar": str(tar_path),
        "frame_count": len(ids),
        "pixel_check": args.pixel_check,
        "pixel_rows_checked": pixel_rows,
        "metadata_byte_exact": list(HF_REQUIRED),
        "orientation": orientation,
        "live_scene_access": "none",
    }
    if args.output:
        output = args.output.resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        if output.exists() or output.is_symlink():
            raise FileExistsError(output)
        atomic_json(output, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return report


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--scene-dir", type=Path, required=True)
    result.add_argument("--tar", type=Path, required=True)
    result.add_argument("--metadata-scene", type=Path, required=True)
    result.add_argument("--pixel-check", choices=("all", "sample", "none"), default="all")
    result.add_argument("--min-orientation-margin-degrees", type=float, default=30.0)
    result.add_argument("--orientation-policy", type=Path)
    result.add_argument("--output", type=Path)
    return result


def main() -> int:
    args = parser().parse_args()
    if not math.isfinite(args.min_orientation_margin_degrees) or args.min_orientation_margin_degrees < 0:
        raise ValueError("orientation margin must be finite and non-negative")
    audit(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
