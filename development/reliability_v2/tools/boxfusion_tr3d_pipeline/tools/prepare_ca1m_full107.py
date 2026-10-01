#!/usr/bin/env python3
"""Audit processed CA-1M data against the frozen official 107-scene list."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path

import numpy as np


REQUIRED_FILES = (
    "K_depth.txt",
    "K_rgb.txt",
    "all_poses.npy",
    "after_filter_boxes.npy",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def png_count(path: Path) -> int:
    if not path.is_dir():
        return 0
    return sum(
        entry.is_file() and entry.name.lower().endswith(".png")
        for entry in os.scandir(path)
    )


def inspect_scene(path: Path) -> dict:
    missing = [name for name in REQUIRED_FILES if not (path / name).is_file()]
    rgb_count = png_count(path / "rgb")
    depth_count = png_count(path / "depth")
    pose_count = None
    pose_error = None
    if (path / "all_poses.npy").is_file():
        try:
            poses = np.load(path / "all_poses.npy", mmap_mode="r")
            pose_count = int(poses.reshape(-1, 4, 4).shape[0])
        except Exception as exc:  # fail closed and retain the exact reason
            pose_error = f"{type(exc).__name__}: {exc}"
    complete = (
        not missing
        and pose_error is None
        and pose_count is not None
        and pose_count > 0
        and rgb_count == pose_count
        and depth_count == pose_count
    )
    return {
        "scene_id": path.name,
        "complete": complete,
        "rgb_frames": rgb_count,
        "depth_frames": depth_count,
        "pose_frames": pose_count,
        "missing": missing,
        "pose_error": pose_error,
    }


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", dir=path.parent, delete=False, encoding="utf-8"
    ) as handle:
        handle.write(text)
        temporary = Path(handle.name)
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--scene-list", type=Path, required=True)
    parser.add_argument("--report-output", type=Path, required=True)
    parser.add_argument("--expected-scenes", type=int, default=107)
    parser.add_argument(
        "--allow-unlisted-scenes",
        action="store_true",
        help=(
            "Audit an exact scene-list subset while allowing other numeric scene "
            "directories to remain in the source data root. The report still lists "
            "them and the evaluation view must remain exact-set."
        ),
    )
    args = parser.parse_args()

    data_root = args.data_root.resolve()
    if not data_root.is_dir():
        parser.error(f"CA-1M data root is absent: {data_root}")
    present_paths = sorted(
        (path for path in data_root.iterdir() if path.is_dir() and path.name.isdigit()),
        key=lambda path: path.name,
    )
    expected_ids = tuple(
        row.strip()
        for row in args.scene_list.read_text(encoding="utf-8").splitlines()
        if row.strip()
    )
    if (
        len(expected_ids) != args.expected_scenes
        or len(set(expected_ids)) != args.expected_scenes
        or any(not scene.isdigit() for scene in expected_ids)
    ):
        raise ValueError(
            f"official scene list must contain {args.expected_scenes} unique numeric IDs"
        )
    expected_set = set(expected_ids)
    present_set = {path.name for path in present_paths}
    rows = [inspect_scene(data_root / scene) for scene in expected_ids]
    complete = [row["scene_id"] for row in rows if row["complete"]]
    incomplete = [row for row in rows if not row["complete"]]
    partial_present = [row for row in incomplete if row["scene_id"] in present_set]
    absent = sorted(expected_set - present_set)
    unexpected = sorted(present_set - expected_set)
    ready = (
        len(complete) == args.expected_scenes
        and (args.allow_unlisted_scenes or not unexpected)
    )
    report = {
        "schema": "boxfusion.ca1m_full107_preparation.v1",
        "ready": ready,
        "data_root": str(data_root),
        "expected_scenes": args.expected_scenes,
        "official_scene_list": str(args.scene_list.resolve()),
        "official_scene_list_sha256": sha256(args.scene_list),
        "present_scene_directories": len(present_paths),
        "complete_scenes": len(complete),
        "incomplete_present_scenes": partial_present,
        "complete_scene_ids": complete,
        "absent_scene_ids": absent,
        "unexpected_numeric_scene_ids": unexpected,
        "allow_unlisted_scenes": args.allow_unlisted_scenes,
    }
    atomic_text(args.report_output, json.dumps(report, indent=2, sort_keys=True) + "\n")

    print(
        json.dumps(
            {
                "ready": ready,
                "present": len(present_paths),
                "complete": len(complete),
                "expected": args.expected_scenes,
                "partial_present": len(partial_present),
                "absent": len(absent),
                "report": str(args.report_output.resolve()),
            },
            indent=2,
        )
    )
    if not ready:
        print(
            "CA-1M is not ready for the official 107-scene evaluation. "
            "Resume the Hugging Face download and run this audit again."
        )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
