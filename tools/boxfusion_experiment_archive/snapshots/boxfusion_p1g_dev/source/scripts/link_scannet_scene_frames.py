#!/usr/bin/env python3
"""Create an isolated ``<scene>/frames`` view of extracted ScanNet scenes."""

from __future__ import annotations

import argparse
from pathlib import Path


FRAME_DIRECTORIES = ("color", "depth", "pose", "intrinsic")


def read_scenes(path: Path) -> list[str]:
    scenes = [line.strip() for line in path.read_text().splitlines()]
    scenes = [scene for scene in scenes if scene]
    if not scenes:
        raise ValueError(f"empty scene list: {path}")
    if len(set(scenes)) != len(scenes):
        raise ValueError(f"duplicate scene id in {path}")
    return scenes


def ensure_link(link: Path, target: Path) -> None:
    target = target.resolve()
    if not target.is_dir():
        raise FileNotFoundError(target)
    if link.is_symlink():
        if link.resolve() == target:
            return
        link.unlink()
    elif link.exists():
        raise FileExistsError(
            f"refusing to replace non-symlink frame path: {link}"
        )
    link.symlink_to(target)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-list", type=Path, required=True)
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path("/extra/ZhaoX/scannet_data/scans.sens"),
        help="Extracted ScanNet scans/<scene> root",
    )
    parser.add_argument(
        "--frames-root",
        type=Path,
        default=Path(
            "/data/ZhaoX/OVM3D-Dett/boxfusion_b6_dev/data/scannet_train"
        ),
        help="Isolated BoxFusion <scene>/frames output root",
    )
    args = parser.parse_args()

    scenes = read_scenes(args.scene_list)
    linked = 0
    for scene in scenes:
        frames = args.frames_root / scene / "frames"
        frames.mkdir(parents=True, exist_ok=True)
        for name in FRAME_DIRECTORIES:
            ensure_link(
                frames / name,
                args.source_root / scene / name,
            )
            linked += 1
    print(
        f"Linked {linked} frame directories for {len(scenes)} scenes "
        f"under {args.frames_root}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
