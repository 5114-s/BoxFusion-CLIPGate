#!/usr/bin/env python3
"""Convert a sampled MultiScan sequence to BoxFusion's ScanNet-style layout."""

from __future__ import annotations

import argparse
import json
import zlib
from pathlib import Path

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("scene_dir", type=Path)
    parser.add_argument("output_root", type=Path)
    parser.add_argument("--stride", type=int, default=120)
    parser.add_argument("--max-frames", type=int, default=16)
    return parser.parse_args()


def select_indices(num_frames: int, stride: int, max_frames: int) -> list[int]:
    if stride < 1 or max_frames < 1:
        raise ValueError("stride and max-frames must be positive")
    indices = list(range(0, num_frames, stride))
    if len(indices) > max_frames:
        positions = np.linspace(0, len(indices) - 1, max_frames, dtype=int)
        indices = [indices[int(position)] for position in positions]
    return sorted(set(indices))


def extract_depth_frames(
    source: Path,
    selected: set[int],
    output_dir: Path,
    height: int,
    width: int,
    unit: str,
) -> None:
    frame_bytes = height * width * 2
    decompressor = zlib.decompressobj(-zlib.MAX_WBITS)
    pending = bytearray()
    frame_index = 0

    def consume(data: bytes) -> None:
        nonlocal frame_index
        pending.extend(data)
        while len(pending) >= frame_bytes:
            raw = bytes(pending[:frame_bytes])
            del pending[:frame_bytes]
            if frame_index in selected:
                dtype = np.float16 if unit == "m" else np.uint16
                depth = np.frombuffer(raw, dtype=dtype).reshape(height, width)
                if unit == "m":
                    depth_m = depth.astype(np.float32)
                elif unit == "mm":
                    depth_m = depth.astype(np.float32) / 1000.0
                else:
                    raise ValueError(f"unsupported depth unit: {unit}")
                depth_m[~np.isfinite(depth_m)] = 0.0
                depth_mm = np.clip(np.rint(depth_m * 1000.0), 0, 65535).astype(np.uint16)
                if not cv2.imwrite(str(output_dir / f"{frame_index}.png"), depth_mm):
                    raise RuntimeError(f"failed to write depth frame {frame_index}")
            frame_index += 1

    with source.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            consume(decompressor.decompress(block))
        consume(decompressor.flush())
    if pending:
        raise ValueError(f"trailing decompressed depth bytes: {len(pending)}")


def extract_color_frames(source: Path, selected: set[int], output_dir: Path) -> None:
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open video: {source}")
    last = max(selected)
    index = 0
    while index <= last:
        ok, frame = capture.read()
        if not ok:
            capture.release()
            raise RuntimeError(f"video ended at frame {index}, expected at least {last + 1}")
        if index in selected:
            if not cv2.imwrite(str(output_dir / f"{index}.jpg"), frame):
                capture.release()
                raise RuntimeError(f"failed to write color frame {index}")
        index += 1
    capture.release()


def main() -> None:
    args = parse_args()
    scene_dir = args.scene_dir.resolve()
    scene_id = scene_dir.name
    prefix = scene_dir / scene_id
    with (prefix.with_suffix(".json")).open(encoding="utf-8") as handle:
        metadata = json.load(handle)
    with (prefix.with_suffix(".jsonl")).open(encoding="utf-8") as handle:
        cameras = [json.loads(line) for line in handle if line.strip()]

    color_stream, depth_stream = metadata["streams"][:2]
    num_frames = min(
        int(color_stream["number_of_frames"]),
        int(depth_stream["number_of_frames"]),
        len(cameras),
    )
    indices = select_indices(num_frames, args.stride, args.max_frames)
    selected = set(indices)
    color_h, color_w = map(int, color_stream["resolution"])
    depth_h, depth_w = map(int, depth_stream["resolution"])

    scene_out = args.output_root.resolve() / scene_id
    frames_out = scene_out / "frames"
    color_out = frames_out / "color"
    depth_out = frames_out / "depth"
    pose_out = frames_out / "pose"
    intrinsic_out = frames_out / "intrinsic"
    for directory in (color_out, depth_out, pose_out, intrinsic_out):
        directory.mkdir(parents=True, exist_ok=True)

    extract_color_frames(prefix.with_suffix(".mp4"), selected, color_out)
    extract_depth_frames(
        prefix.with_suffix(".depth.zlib"),
        selected,
        depth_out,
        depth_h,
        depth_w,
        str(metadata.get("depth_unit", "m")),
    )

    depth_intrinsics = []
    flip_yz = np.diag([1.0, -1.0, -1.0, 1.0])
    scale = np.diag([depth_w / color_w, depth_h / color_h, 1.0])
    for index in indices:
        camera = cameras[index]
        c2w_arkit = np.asarray(camera["transform"], dtype=np.float64).reshape(4, 4).T
        c2w = c2w_arkit @ flip_yz
        c2w /= c2w[3, 3]
        np.savetxt(pose_out / f"{index}.txt", c2w, fmt="%.10g")
        k_color = np.asarray(camera["intrinsics"], dtype=np.float64).reshape(3, 3).T
        depth_intrinsics.append(scale @ k_color)

    k_depth = np.median(np.stack(depth_intrinsics), axis=0)
    intrinsic4 = np.eye(4, dtype=np.float64)
    intrinsic4[:3, :3] = k_depth
    np.savetxt(intrinsic_out / "intrinsic_depth.txt", intrinsic4, fmt="%.10g")
    np.savetxt(frames_out / "K_depth.txt", k_depth, fmt="%.10g")
    manifest = {
        "schema": "boxfusion.multiscan_sample.v1",
        "scene_id": scene_id,
        "source": str(scene_dir),
        "source_frame_count": num_frames,
        "selected_indices": indices,
        "color_resolution": [color_h, color_w],
        "depth_resolution": [depth_h, depth_w],
        "depth_unit": metadata.get("depth_unit", "m"),
        "output_depth_unit": "mm_uint16_png",
        "pose_convention": "camera_to_world_opencv",
    }
    (scene_out / "multiscan_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
