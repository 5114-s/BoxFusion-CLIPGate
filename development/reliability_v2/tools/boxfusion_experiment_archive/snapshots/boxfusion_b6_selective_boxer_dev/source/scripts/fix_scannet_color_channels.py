#!/usr/bin/env python3
"""Create a ScanNet frame tree with corrected RGB channel order.

The local SensorData.py used on this machine decoded JPEGs with OpenCV
(BGR) and saved them with imageio (expects RGB), so the exported color jpgs
have red/blue swapped. This script creates a BoxFusion-compatible tree whose
color images are channel-swapped back, while depth/pose/intrinsic are symlinked.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import cv2


def relink(link: Path, target: Path) -> None:
    if link.exists() or link.is_symlink():
        if link.is_symlink() and link.readlink() == target:
            return
        link.unlink()
    link.symlink_to(target)


def fix_one_image(src: Path, dst: Path, overwrite: bool) -> bool:
    if dst.exists() and not overwrite:
        return False
    img = cv2.imread(str(src), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"failed to read {src}")
    # The current exported jpg stores BGR values as RGB pixels. cv2.imread
    # returns [stored_B, stored_G, stored_R] == [true_R, true_G, true_B].
    # cv2.imwrite expects BGR, so swap once before writing a correct RGB jpg.
    cv2.imwrite(str(dst), img[:, :, ::-1], [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    return True


def fix_scene(scene: str, source_root: Path, output_root: Path, overwrite: bool, workers: int) -> None:
    src_scene = source_root / scene
    out_frames = output_root / scene / "frames"
    out_color = out_frames / "color"
    out_color.mkdir(parents=True, exist_ok=True)

    for name in ("depth", "pose", "intrinsic"):
        relink(out_frames / name, src_scene / name)

    color_files = sorted((src_scene / "color").glob("*.jpg"), key=lambda p: int(p.stem))
    tasks = [(src, out_color / src.name) for src in color_files]
    fixed = 0
    if workers <= 1:
        for src, dst in tasks:
            fixed += int(fix_one_image(src, dst, overwrite))
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(fix_one_image, src, dst, overwrite) for src, dst in tasks]
            for future in as_completed(futures):
                fixed += int(future.result())
    print(f"{scene}: fixed {fixed}/{len(color_files)} color frames -> {out_color}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", default="/extra/ZhaoX/scannet_data/scans")
    parser.add_argument("--output-root", default="data/scannet_val_rgbfix")
    parser.add_argument("--val-list", default="evaluation/data_util/meta_data/scannetv2_val.txt")
    parser.add_argument("--scenes", nargs="*", default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()

    source_root = Path(args.source_root)
    output_root = Path(args.output_root)
    if args.scenes:
        scenes = args.scenes
    else:
        scenes = [line.strip() for line in Path(args.val_list).read_text().splitlines() if line.strip()]

    for scene in scenes:
        fix_scene(scene, source_root, output_root, args.overwrite, args.workers)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
