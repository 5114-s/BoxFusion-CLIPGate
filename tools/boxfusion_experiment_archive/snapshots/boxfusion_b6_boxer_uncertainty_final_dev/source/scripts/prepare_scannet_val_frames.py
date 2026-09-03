#!/usr/bin/env python3
"""Extract posed RGB-D frames for an explicit BoxFusion ScanNetV2 scene list.

This uses the ScanNet `SensorData.py` reader that already exists in
`/extra/ZhaoX/scannet_data` on this machine. It processes only the explicitly
provided list and skips scenes whose `color/`, `depth/`, `pose/`, and
`intrinsic/` directories already contain files.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def count_files(path: Path, suffix: str) -> int:
    if not path.is_dir():
        return 0
    return sum(1 for p in path.iterdir() if p.is_file() and p.name.endswith(suffix))


def scene_is_extracted(scene_dir: Path) -> bool:
    color_count = count_files(scene_dir / "color", ".jpg")
    depth_count = count_files(scene_dir / "depth", ".png")
    pose_count = count_files(scene_dir / "pose", ".txt")
    return (
        color_count > 0
        and color_count == depth_count == pose_count
        and (scene_dir / "intrinsic" / "intrinsic_depth.txt").exists()
    )


def load_val_scenes(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scannet-root",
        default="/extra/ZhaoX/scannet_data",
        help="Directory containing ScanNet SensorData.py and scans/<scene>/<scene>.sens",
    )
    parser.add_argument(
        "--val-list",
        default="evaluation/data_util/meta_data/scannetv2_val.txt",
        help="BoxFusion ScanNetV2 validation scene list",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional maximum number of frames per scene; 0 means all frames",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print what would be processed",
    )
    args = parser.parse_args()

    scannet_root = Path(args.scannet_root).resolve()
    scans_root = scannet_root / "scans"
    val_scenes = load_val_scenes(Path(args.val_list))

    sys.path.insert(0, str(scannet_root))
    from SensorData import SensorData  # type: ignore

    missing_sens: list[str] = []
    todo: list[str] = []
    done: list[str] = []

    for scene in val_scenes:
        scene_dir = scans_root / scene
        sens_file = scene_dir / f"{scene}.sens"
        if not sens_file.exists():
            missing_sens.append(scene)
            continue
        if scene_is_extracted(scene_dir):
            done.append(scene)
        else:
            todo.append(scene)

    print(f"selected scenes: {len(val_scenes)}")
    print(f"already extracted: {len(done)}")
    print(f"to extract: {len(todo)}")
    print(f"missing .sens: {len(missing_sens)}")
    if missing_sens:
        print("missing:", " ".join(missing_sens[:20]))
        return 2

    if args.dry_run:
        if todo:
            print("first todo:", " ".join(todo[:20]))
        return 0

    for idx, scene in enumerate(todo, start=1):
        scene_dir = scans_root / scene
        sens_file = scene_dir / f"{scene}.sens"
        print(f"[{idx}/{len(todo)}] extracting {scene}: {sens_file}", flush=True)

        sd = SensorData(str(sens_file))
        if args.limit and args.limit > 0:
            sd.frames = sd.frames[: args.limit]

        (scene_dir / "depth").mkdir(exist_ok=True)
        (scene_dir / "color").mkdir(exist_ok=True)
        (scene_dir / "pose").mkdir(exist_ok=True)
        (scene_dir / "intrinsic").mkdir(exist_ok=True)

        sd.export_depth_images(str(scene_dir / "depth"))
        sd.export_color_images(str(scene_dir / "color"))
        sd.export_poses(str(scene_dir / "pose"))
        sd.export_intrinsics(str(scene_dir / "intrinsic"))

        print(
            f"[{idx}/{len(todo)}] done {scene}: "
            f"color={count_files(scene_dir / 'color', '.jpg')} "
            f"depth={count_files(scene_dir / 'depth', '.png')} "
            f"pose={count_files(scene_dir / 'pose', '.txt')}",
            flush=True,
        )

    print("ScanNet frame extraction complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
