#!/usr/bin/env python3
"""Prepare sampled MultiScan RGB-D scans for BoxFusion evaluation.

The release stores ARKit camera poses in the capture world frame while the
mesh and object OBBs use the aligned scan frame.  This adapter applies the
inverse of ``coordinate_transform`` to every camera-to-world pose.  It also
warps every sampled RGB/depth pair to a fixed median intrinsic matrix because
MultiScan records autofocus-varying intrinsics but BoxFusion consumes one
intrinsic matrix per sequence.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import zlib

import cv2
import numpy as np


SIGNS = np.asarray(
    [[x, y, z] for x in (-1.0, 1.0) for y in (-1.0, 1.0) for z in (-1.0, 1.0)],
    dtype=np.float64,
)
STRUCTURAL = {"beam", "ceiling", "floor", "pillar", "remove", "wall", "windowsill"}
SCANNET18_MAP = {
    "basin": "sink",
    "bed": "bed",
    "cabinet": "cabinet",
    "chair": "chair",
    "curtain": "curtain",
    "door": "door",
    "fridge": "refrigerator",
    "painting": "picture",
    "picture": "picture",
    "refrigerator": "refrigerator",
    "shower": "shower curtain",
    "sink": "sink",
    "sliding_door": "door",
    "sofa": "sofa",
    "table": "table",
    "toilet": "toilet",
    "wall_cabinet": "cabinet",
    "wardrobe": "cabinet",
    "window": "window",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def corners_from_bounds(low: np.ndarray, high: np.ndarray) -> np.ndarray:
    return np.asarray(
        [[x, y, z] for x in (low[0], high[0]) for y in (low[1], high[1]) for z in (low[2], high[2])],
        dtype=np.float64,
    )


def valid_bounds(obb: dict) -> tuple[np.ndarray, np.ndarray]:
    low = np.asarray(obb["min"], dtype=np.float64)
    high = np.asarray(obb["max"], dtype=np.float64)
    if low.shape != (3,) or high.shape != (3,) or not np.isfinite([low, high]).all():
        raise ValueError("invalid MultiScan OBB bounds")
    if np.any(high <= low):
        raise ValueError(f"degenerate MultiScan OBB bounds: {low} / {high}")
    return low, high


def write_depth_frames(
    source: Path,
    selected_to_output: dict[int, int],
    output_dir: Path,
    height: int,
    width: int,
    homographies: dict[int, np.ndarray],
    output_size: tuple[int, int],
) -> None:
    frame_bytes = height * width * 2
    decompressor = zlib.decompressobj(-zlib.MAX_WBITS)
    pending = bytearray()
    frame_index = 0
    last = max(selected_to_output)

    def consume(data: bytes) -> None:
        nonlocal frame_index
        pending.extend(data)
        while len(pending) >= frame_bytes and frame_index <= last:
            raw = bytes(pending[:frame_bytes])
            del pending[:frame_bytes]
            output_index = selected_to_output.get(frame_index)
            if output_index is not None:
                depth_m = np.frombuffer(raw, dtype=np.float16).reshape(height, width).astype(np.float32)
                depth_m[~np.isfinite(depth_m)] = 0.0
                depth_mm = np.clip(np.rint(depth_m * 1000.0), 0, 65535).astype(np.uint16)
                warped = cv2.warpPerspective(
                    depth_mm,
                    homographies[frame_index],
                    output_size,
                    flags=cv2.INTER_NEAREST,
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=0,
                )
                if not cv2.imwrite(str(output_dir / f"{output_index}.png"), warped):
                    raise RuntimeError(f"failed to write depth frame {frame_index}")
            frame_index += 1

    with source.open("rb") as handle:
        while frame_index <= last:
            block = handle.read(1 << 20)
            if not block:
                consume(decompressor.flush())
                break
            consume(decompressor.decompress(block))
    if frame_index <= last:
        raise ValueError(f"depth stream ended at {frame_index}; expected frame {last}")


def write_color_frames(
    source: Path,
    selected_to_output: dict[int, int],
    output_dir: Path,
    homographies: dict[int, np.ndarray],
    output_size: tuple[int, int],
) -> None:
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open video: {source}")
    last = max(selected_to_output)
    index = 0
    try:
        while index <= last:
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError(f"video ended at frame {index}; expected frame {last}")
            output_index = selected_to_output.get(index)
            if output_index is not None:
                warped = cv2.warpPerspective(
                    frame,
                    homographies[index],
                    output_size,
                    flags=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=0,
                )
                if not cv2.imwrite(
                    str(output_dir / f"{output_index}.jpg"),
                    warped,
                    [cv2.IMWRITE_JPEG_QUALITY, 95],
                ):
                    raise RuntimeError(f"failed to write color frame {index}")
            index += 1
    finally:
        capture.release()


def stream_by_type(metadata: dict, stream_type: str) -> dict:
    matches = [stream for stream in metadata["streams"] if stream.get("type") == stream_type]
    if len(matches) != 1:
        raise ValueError(f"expected one {stream_type} stream, found {len(matches)}")
    return matches[0]


def prepare_scene(
    scene_dir: Path,
    target: Path,
    alias: str,
    stride: int,
    output_width: int,
    output_height: int,
) -> dict:
    scene_id = scene_dir.name
    prefix = scene_dir / scene_id
    metadata_path = prefix.with_suffix(".json")
    cameras_path = prefix.with_suffix(".jsonl")
    align_path = prefix.with_suffix(".align.json")
    annotations_path = prefix.with_suffix(".annotations.json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    cameras = [json.loads(line) for line in cameras_path.read_text(encoding="utf-8").splitlines() if line]
    color_stream = stream_by_type(metadata, "color_camera")
    depth_stream = stream_by_type(metadata, "lidar_sensor")
    color_h, color_w = map(int, color_stream["resolution"])
    depth_h, depth_w = map(int, depth_stream["resolution"])
    count = min(int(color_stream["number_of_frames"]), int(depth_stream["number_of_frames"]), len(cameras))
    source_indices = list(range(0, count, stride))
    if source_indices[-1] != count - 1:
        source_indices.append(count - 1)
    selected_to_output = {source: output for output, source in enumerate(source_indices)}

    k_color = {
        index: np.asarray(cameras[index]["intrinsics"], dtype=np.float64).reshape(3, 3).T
        for index in source_indices
    }
    k_color_scaled = np.stack(
        [np.diag([output_width / color_w, output_height / color_h, 1.0]) @ k_color[index] for index in source_indices]
    )
    target_k = np.median(k_color_scaled, axis=0)
    color_homographies = {index: target_k @ np.linalg.inv(k_color[index]) for index in source_indices}
    depth_scale = np.diag([depth_w / color_w, depth_h / color_h, 1.0])
    depth_homographies = {
        index: target_k @ np.linalg.inv(depth_scale @ k_color[index]) for index in source_indices
    }

    frames = target / "frames"
    color_out, depth_out = frames / "color", frames / "depth"
    pose_out, intrinsic_out = frames / "pose", frames / "intrinsic"
    for directory in (color_out, depth_out, pose_out, intrinsic_out):
        directory.mkdir(parents=True, exist_ok=True)
    output_size = (output_width, output_height)
    write_color_frames(prefix.with_suffix(".mp4"), selected_to_output, color_out, color_homographies, output_size)
    write_depth_frames(
        prefix.with_suffix(".depth.zlib"),
        selected_to_output,
        depth_out,
        depth_h,
        depth_w,
        depth_homographies,
        output_size,
    )

    alignment = np.asarray(
        json.loads(align_path.read_text(encoding="utf-8"))["coordinate_transform"], dtype=np.float64
    ).reshape(4, 4).T
    capture_to_aligned = np.linalg.inv(alignment)
    flip_yz = np.diag([1.0, -1.0, -1.0, 1.0])
    for output_index, source_index in enumerate(source_indices):
        c2w_arkit = np.asarray(cameras[source_index]["transform"], dtype=np.float64).reshape(4, 4).T
        c2w_aligned = capture_to_aligned @ c2w_arkit @ flip_yz
        c2w_aligned /= c2w_aligned[3, 3]
        np.savetxt(pose_out / f"{output_index}.txt", c2w_aligned, fmt="%.10g")
    intrinsic4 = np.eye(4, dtype=np.float64)
    intrinsic4[:3, :3] = target_k
    np.savetxt(intrinsic_out / "intrinsic_depth.txt", intrinsic4, fmt="%.10g")
    np.savetxt(frames / "K_depth.txt", target_k, fmt="%.10g")

    annotations = json.loads(annotations_path.read_text(encoding="utf-8"))
    nonstructural, scannet18 = [], []
    selected_labels, ignored_labels, ignored_missing_obb = [], [], []
    for obj in annotations.get("objects", []):
        raw_label = str(obj.get("label", "")).strip().casefold()
        label = raw_label.rsplit(".", 1)[0]
        if not isinstance(obj.get("obb"), dict):
            ignored_missing_obb.append(raw_label)
            continue
        low, high = valid_bounds(obj["obb"])
        box = corners_from_bounds(low, high)
        if label in STRUCTURAL:
            ignored_labels.append(raw_label)
            continue
        nonstructural.append(box)
        selected_labels.append(raw_label)
        if label in SCANNET18_MAP:
            scannet18.append(box)
    nonstructural_array = np.stack(nonstructural) if nonstructural else np.zeros((0, 8, 3), dtype=np.float64)
    scannet18_array = np.stack(scannet18) if scannet18 else np.zeros((0, 8, 3), dtype=np.float64)
    np.save(target / "gt_aabb_nonstructural.npy", nonstructural_array)
    np.save(target / "gt_aabb_scannet18.npy", scannet18_array)

    focal_values = k_color_scaled[:, [0, 1], [0, 1]]
    row = {
        "alias": alias,
        "scan_id": scene_id,
        "frames": len(source_indices),
        "source_frames": count,
        "source_stride": stride,
        "source_indices": source_indices,
        "raw_color_size": [color_w, color_h],
        "raw_depth_size": [depth_w, depth_h],
        "target_size": [output_width, output_height],
        "target_intrinsic": target_k.tolist(),
        "sampled_focal_range": [focal_values.min(axis=0).tolist(), focal_values.max(axis=0).tolist()],
        "pose_conversion": "inverse(coordinate_transform) @ c2w_arkit @ diag(1,-1,-1,1)",
        "gt_nonstructural": len(nonstructural_array),
        "gt_scannet18": len(scannet18_array),
        "selected_labels": selected_labels,
        "ignored_structural_labels": ignored_labels,
        "ignored_missing_obb_labels": ignored_missing_obb,
        "metadata_sha256": sha256(metadata_path),
        "camera_sha256": sha256(cameras_path),
        "alignment_sha256": sha256(align_path),
        "annotations_sha256": sha256(annotations_path),
    }
    (target / "metadata.json").write_text(json.dumps(row, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--scene-list", type=Path)
    parser.add_argument("--stride", type=int, default=60, help="source frames between online observations")
    parser.add_argument("--alias-start", type=int, default=9000)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.stride < 1 or args.width < 1 or args.height < 1:
        raise ValueError("stride and target dimensions must be positive")

    raw_root = args.raw_root.resolve()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    if args.scene_list:
        scene_ids = [line.strip() for line in args.scene_list.read_text().splitlines() if line.strip() and not line.startswith("#")]
    else:
        scene_ids = sorted(path.name for path in raw_root.glob("scene_*_00") if path.is_dir())
    if not scene_ids:
        raise ValueError("no MultiScan scenes selected")

    rows = []
    for index, scene_id in enumerate(scene_ids):
        alias = f"scene{args.alias_start + index:04d}_00"
        scene_dir, target = raw_root / scene_id, output_root / alias
        metadata_file = target / "metadata.json"
        if args.resume and metadata_file.is_file():
            row = json.loads(metadata_file.read_text(encoding="utf-8"))
            if row.get("alias") == alias and row.get("scan_id") == scene_id:
                rows.append(row)
                print(f"{alias} <- {scene_id}: resumed frames={row['frames']}", flush=True)
                continue
        if target.exists() and not (args.overwrite or args.resume):
            raise FileExistsError(f"refusing to overwrite {target}")
        if target.exists():
            shutil.rmtree(target)
        row = prepare_scene(scene_dir, target, alias, args.stride, args.width, args.height)
        rows.append(row)
        print(
            f"{alias} <- {scene_id}: frames={row['frames']} "
            f"GT(nonstructural/scannet18)={row['gt_nonstructural']}/{row['gt_scannet18']}",
            flush=True,
        )

    manifest = {
        "schema": "boxfusion.multiscan_preflight.v1",
        "protocol": "MultiScan aligned-world class-agnostic enclosing-AABB AP",
        "protocols": {
            "nonstructural": "all annotated object instances except beam/ceiling/floor/pillar/remove/wall/windowsill",
            "scannet18": "conservative MultiScan-label mapping to ScanNet18 object classes",
        },
        "coordinate_frame": "MultiScan aligned mesh/annotation frame",
        "image_conversion": "per-frame projective warp to median intrinsics at fixed 640x480",
        "depth_unit": "millimetres uint16 on disk; BoxFusion loader divides by 1000",
        "scans": rows,
    }
    (output_root / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output_root / "scenes.txt").write_text("".join(row["alias"] + "\n" for row in rows), encoding="utf-8")
    print(f"wrote {output_root / 'manifest.json'}", flush=True)


if __name__ == "__main__":
    main()
