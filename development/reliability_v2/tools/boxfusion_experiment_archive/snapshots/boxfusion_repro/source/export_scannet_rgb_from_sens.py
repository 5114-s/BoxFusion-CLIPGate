#!/usr/bin/env python3
"""Export ScanNet RGB frames from .sens files using the official RGB path.

The official ScanNet SensorData.py decodes the JPEG payload with imageio and
writes the resulting RGB array with imageio.  This Python 3 implementation
keeps that behavior, but streams one frame at a time so a whole .sens file
does not need to be held in memory.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import io
import json
import os
from pathlib import Path
import struct
import time

import imageio
import imageio.v2 as imageio_v2
import numpy as np
import PIL


SENS_VERSION = 4
JPEG_COMPRESSION_TYPE = 2
OFFICIAL_SENSORDATA_URL = (
    "https://github.com/ScanNet/ScanNet/blob/master/"
    "SensReader/python/SensorData.py"
)


def read_exact(handle, size: int) -> bytes:
    data = handle.read(size)
    if len(data) != size:
        raise EOFError(f"Expected {size} bytes, received {len(data)}")
    return data


def unpack(handle, fmt: str):
    size = struct.calcsize(fmt)
    return struct.unpack(fmt, read_exact(handle, size))


def read_header(handle) -> dict:
    version = unpack(handle, "<I")[0]
    if version != SENS_VERSION:
        raise ValueError(f"Unsupported .sens version {version}")

    sensor_name_size = unpack(handle, "<Q")[0]
    sensor_name = read_exact(handle, sensor_name_size).decode("utf-8")

    matrices = []
    for _ in range(4):
        matrices.append(
            np.asarray(unpack(handle, "<" + "f" * 16), dtype=np.float32)
            .reshape(4, 4)
            .tolist()
        )

    color_compression = unpack(handle, "<i")[0]
    depth_compression = unpack(handle, "<i")[0]
    color_width = unpack(handle, "<I")[0]
    color_height = unpack(handle, "<I")[0]
    depth_width = unpack(handle, "<I")[0]
    depth_height = unpack(handle, "<I")[0]
    depth_shift = unpack(handle, "<f")[0]
    num_frames = unpack(handle, "<Q")[0]

    if color_compression != JPEG_COMPRESSION_TYPE:
        raise ValueError(
            f"Expected JPEG color compression ({JPEG_COMPRESSION_TYPE}), "
            f"received {color_compression}"
        )

    return {
        "version": version,
        "sensor_name": sensor_name,
        "intrinsic_color": matrices[0],
        "extrinsic_color": matrices[1],
        "intrinsic_depth": matrices[2],
        "extrinsic_depth": matrices[3],
        "color_compression_type": color_compression,
        "depth_compression_type": depth_compression,
        "color_width": color_width,
        "color_height": color_height,
        "depth_width": depth_width,
        "depth_height": depth_height,
        "depth_shift": depth_shift,
        "num_frames": num_frames,
    }


def read_frame(handle) -> tuple[bytes, int]:
    read_exact(handle, 16 * 4)  # camera_to_world
    read_exact(handle, 8)  # timestamp_color
    read_exact(handle, 8)  # timestamp_depth
    color_size = unpack(handle, "<Q")[0]
    depth_size = unpack(handle, "<Q")[0]
    color_data = read_exact(handle, color_size)
    handle.seek(depth_size, os.SEEK_CUR)
    return color_data, depth_size


def replace_symlink(link_path: Path, target_path: Path) -> None:
    if link_path.is_symlink():
        if Path(os.readlink(link_path)) == target_path:
            return
        link_path.unlink()
    elif link_path.exists():
        raise FileExistsError(
            f"{link_path} exists and is not a symlink; refusing to replace it"
        )
    link_path.symlink_to(target_path, target_is_directory=True)


def image_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_export(color_dir: Path, header: dict) -> dict:
    expected = int(header["num_frames"])
    paths = sorted(color_dir.glob("*.jpg"), key=lambda p: int(p.stem))
    ids = [int(path.stem) for path in paths]
    if ids != list(range(expected)):
        missing = sorted(set(range(expected)) - set(ids))
        raise RuntimeError(
            f"RGB frame IDs are not contiguous: count={len(ids)}, "
            f"expected={expected}, first_missing={missing[:10]}"
        )

    sample_ids = sorted({0, expected // 2, expected - 1})
    sample_hashes = {}
    expected_shape = (
        int(header["color_height"]),
        int(header["color_width"]),
        3,
    )
    for frame_id in sample_ids:
        path = color_dir / f"{frame_id}.jpg"
        rgb = imageio_v2.imread(path)
        if rgb.shape != expected_shape:
            raise RuntimeError(
                f"{path} has shape {rgb.shape}, expected {expected_shape}"
            )
        sample_hashes[str(frame_id)] = image_sha256(path)

    return {
        "frame_count": len(paths),
        "sample_sha256": sample_hashes,
    }


def export_scene(
    scene: str,
    source_root: str,
    output_root: str,
    overwrite: bool,
    max_frames: int | None,
) -> dict:
    source_scene = Path(source_root) / scene
    sens_path = source_scene / f"{scene}.sens"
    output_scene = Path(output_root) / scene
    frames_dir = output_scene / "frames"
    color_dir = frames_dir / "color"
    marker_path = output_scene / ".sens_rgb_complete.json"

    if not sens_path.is_file():
        raise FileNotFoundError(sens_path)

    if marker_path.is_file() and not overwrite and max_frames is None:
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        if (
            marker.get("sens_size_bytes") == sens_path.stat().st_size
            and marker.get("scene") == scene
        ):
            return {
                "scene": scene,
                "status": "already_complete",
                "frame_count": marker["validation"]["frame_count"],
                "seconds": 0.0,
            }

    color_dir.mkdir(parents=True, exist_ok=True)
    for name in ("depth", "pose", "intrinsic"):
        target = source_scene / name
        if not target.is_dir():
            raise FileNotFoundError(target)
        replace_symlink(frames_dir / name, target)

    start = time.time()
    written = 0
    skipped = 0

    with sens_path.open("rb") as handle:
        header = read_header(handle)
        frame_limit = int(header["num_frames"])
        if max_frames is not None:
            frame_limit = min(frame_limit, max_frames)

        for frame_id in range(int(header["num_frames"])):
            color_data, _ = read_frame(handle)
            if frame_id >= frame_limit:
                continue

            output_path = color_dir / f"{frame_id}.jpg"
            if output_path.is_file() and output_path.stat().st_size > 0 and not overwrite:
                skipped += 1
                continue

            # Official ScanNet path: imageio.imread(JPEG bytes) returns RGB,
            # followed by imageio.imwrite(..., RGB array).
            rgb = imageio_v2.imread(io.BytesIO(color_data))
            expected_shape = (
                int(header["color_height"]),
                int(header["color_width"]),
                3,
            )
            if rgb.shape != expected_shape:
                raise RuntimeError(
                    f"{sens_path} frame {frame_id}: decoded shape {rgb.shape}, "
                    f"expected {expected_shape}"
                )

            temp_path = color_dir / f"{frame_id}.tmp.jpg"
            imageio_v2.imwrite(temp_path, rgb)
            os.replace(temp_path, output_path)
            written += 1

    elapsed = time.time() - start
    if max_frames is not None:
        return {
            "scene": scene,
            "status": "smoke_complete",
            "frame_count": frame_limit,
            "written": written,
            "skipped": skipped,
            "seconds": elapsed,
        }

    validation = validate_export(color_dir, header)
    marker = {
        "scene": scene,
        "source_sens": str(sens_path),
        "sens_size_bytes": sens_path.stat().st_size,
        "sens_mtime_ns": sens_path.stat().st_mtime_ns,
        "official_sensordata_url": OFFICIAL_SENSORDATA_URL,
        "decoder": "imageio.v2.imread(BytesIO(jpeg_payload))",
        "writer": "imageio.v2.imwrite(path, rgb)",
        "imageio_version": imageio.__version__,
        "pillow_version": PIL.__version__,
        "header": header,
        "validation": validation,
        "elapsed_seconds": elapsed,
    }
    temporary_marker = output_scene / ".sens_rgb_complete.tmp.json"
    temporary_marker.write_text(
        json.dumps(marker, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary_marker, marker_path)

    return {
        "scene": scene,
        "status": "exported",
        "frame_count": validation["frame_count"],
        "written": written,
        "skipped": skipped,
        "seconds": elapsed,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--scene-list", required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-frames", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    scenes = [
        line.strip()
        for line in Path(args.scene_list).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if args.limit is not None:
        scenes = scenes[: args.limit]
    if len(scenes) != len(set(scenes)):
        raise ValueError("Scene list contains duplicates")

    print(
        f"Exporting {len(scenes)} scene(s) with {args.workers} worker(s) "
        f"from {args.source_root} to {args.output_root}",
        flush=True,
    )
    results = []
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=max(1, args.workers)
    ) as executor:
        future_to_scene = {
            executor.submit(
                export_scene,
                scene,
                args.source_root,
                args.output_root,
                args.overwrite,
                args.max_frames,
            ): scene
            for scene in scenes
        }
        for index, future in enumerate(
            concurrent.futures.as_completed(future_to_scene), start=1
        ):
            scene = future_to_scene[future]
            result = future.result()
            results.append(result)
            print(
                f"[{index}/{len(scenes)}] {scene}: "
                f"{result['status']}, frames={result['frame_count']}, "
                f"seconds={result['seconds']:.1f}",
                flush=True,
            )

    total_frames = sum(int(result["frame_count"]) for result in results)
    total_seconds = sum(float(result["seconds"]) for result in results)
    print(
        f"Done: scenes={len(results)}, frames={total_frames}, "
        f"sum_worker_seconds={total_seconds:.1f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
