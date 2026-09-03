#!/usr/bin/env python3
"""Create BoxFusion ScanNetV2 val `scene/frames/*` symlinks.

The local ScanNet data in /extra is laid out as:
  scans/<scene>/color
  scans/<scene>/depth
  scans/<scene>/pose
  scans/<scene>/intrinsic

BoxFusion's ScanNet loader expects:
  <root>/<scene>/frames/color
  <root>/<scene>/frames/depth
  <root>/<scene>/frames/pose
  <root>/<scene>/frames/intrinsic

This script creates lightweight symlinks under the repository so no images are
copied.
"""

from __future__ import annotations

from pathlib import Path


VAL_LIST = Path("evaluation/data_util/meta_data/scannetv2_val.txt")
SOURCE_ROOT = Path("/extra/ZhaoX/scannet_data/scans")
LINK_ROOT = Path("data/scannet_val")


def relink(link: Path, target: Path) -> None:
    if link.exists() or link.is_symlink():
        if link.is_symlink() and link.readlink() == target:
            return
        link.unlink()
    link.symlink_to(target)


def main() -> int:
    scenes = [line.strip() for line in VAL_LIST.read_text().splitlines() if line.strip()]
    missing: list[str] = []
    linked = 0
    for scene in scenes:
        src_scene = SOURCE_ROOT / scene
        if not (src_scene / f"{scene}.sens").exists():
            missing.append(scene)
            continue

        frames = LINK_ROOT / scene / "frames"
        frames.mkdir(parents=True, exist_ok=True)
        for name in ("color", "depth", "pose", "intrinsic"):
            relink(frames / name, src_scene / name)
            linked += 1

    print(f"scenes: {len(scenes)}")
    print(f"linked dirs: {linked}")
    print(f"missing .sens scenes: {len(missing)}")
    if missing:
        print("missing:", " ".join(missing[:20]))
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
