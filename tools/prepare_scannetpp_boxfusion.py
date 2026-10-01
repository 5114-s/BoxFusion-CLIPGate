#!/usr/bin/env python3
"""Convert decoded ScanNet++ v2 iPhone streams to BoxFusion's ScanNet contract.

The converter deliberately keeps the official aligned world frame.  It uses
``aligned_pose``/``aligned_poses`` camera-to-world matrices, resizes the aligned
RGB and LiDAR depth images to 640x480, and derives class-agnostic world-space
AABBs from annotated mesh vertices belonging to official instance classes.

The input is the *decoded* official layout.  Run ScanNet++'s
``iphone.prepare_iphone_data`` first so every requested scene has
``iphone/rgb/frame_*.jpg`` and ``iphone/depth/frame_*.png``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
from typing import Any, Iterable

import cv2
import numpy as np
from plyfile import PlyData


FRAME_RE = re.compile(r"(\d+)(?=\.[^.]+$)")
POSE_FRAME_KEYS = ("frame_id", "frameId", "frame", "id", "index")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_ids(path: Path) -> list[str]:
    ids = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    ids = [value for value in ids if value and not value.startswith("#")]
    if not ids or len(ids) != len(set(ids)):
        raise ValueError(f"{path}: scene list must be non-empty and unique")
    invalid = [value for value in ids if re.fullmatch(r"[0-9a-f]{10}", value) is None]
    if invalid:
        raise ValueError(f"{path}: invalid ScanNet++ scene IDs: {invalid[:3]}")
    return ids


def locate_data_root(root: Path, first_scene: str) -> Path:
    candidates = (root, root / "data")
    for candidate in candidates:
        if (candidate / first_scene).is_dir():
            return candidate
    raise FileNotFoundError(
        f"cannot find {first_scene} below {root}; expected <root>/data/<scene> "
        "or <root>/<scene>"
    )


def locate_instance_classes(root: Path, data_root: Path, explicit: Path | None) -> Path:
    candidates = [] if explicit is None else [explicit]
    candidates += [root / "metadata" / "instance_classes.txt", data_root.parent / "metadata" / "instance_classes.txt"]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(
        "missing official metadata/instance_classes.txt; download ScanNet++ metadata "
        "or pass --instance-classes"
    )


def indexed_files(directory: Path, suffixes: tuple[str, ...]) -> dict[int, Path]:
    result: dict[int, Path] = {}
    if not directory.is_dir():
        return result
    for path in directory.iterdir():
        if not path.is_file() or path.suffix.lower() not in suffixes:
            continue
        match = FRAME_RE.search(path.name)
        if match is None:
            continue
        frame_id = int(match.group(1))
        if frame_id in result:
            raise ValueError(f"duplicate frame {frame_id} in {directory}")
        result[frame_id] = path
    return result


def matrix(value: Any, shape: tuple[int, int], name: str) -> np.ndarray:
    if isinstance(value, str):
        array = np.fromstring(value.replace(",", " "), sep=" ", dtype=np.float64)
    else:
        array = np.asarray(value, dtype=np.float64)
    if array.size != shape[0] * shape[1]:
        raise ValueError(f"{name}: expected {shape[0] * shape[1]} values, got {array.size}")
    array = array.reshape(shape)
    if not np.isfinite(array).all():
        raise ValueError(f"{name}: contains non-finite values")
    return array


def frame_id_from_record(key: Any, record: dict[str, Any], fallback: int) -> int:
    for candidate in POSE_FRAME_KEYS:
        if candidate in record:
            value = record[candidate]
            if isinstance(value, str):
                match = re.search(r"\d+", value)
                if match is not None:
                    return int(match.group(0))
            if isinstance(value, (int, np.integer)):
                return int(value)
    match = re.search(r"\d+", str(key))
    return int(match.group(0)) if match is not None else fallback


def parse_pose_json(path: Path) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray]]:
    """Support both released per-frame JSON and documented aggregate JSON."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    poses: dict[int, np.ndarray] = {}
    intrinsics: dict[int, np.ndarray] = {}

    if isinstance(payload, dict) and "aligned_poses" in payload:
        pose_values = payload["aligned_poses"]
        intrinsic_values = payload.get("intrinsic", payload.get("intrinsics"))
        items: Iterable[tuple[Any, Any]]
        items = pose_values.items() if isinstance(pose_values, dict) else enumerate(pose_values)
        for fallback, (key, value) in enumerate(items):
            record = value if isinstance(value, dict) else {}
            pose_value = record.get("aligned_pose", record.get("pose", value))
            frame_id = frame_id_from_record(key, record, fallback)
            poses[frame_id] = matrix(pose_value, (4, 4), f"{path}:{frame_id}:aligned_pose")
            intrinsic_value = record.get("intrinsic")
            if intrinsic_value is None and intrinsic_values is not None:
                if isinstance(intrinsic_values, dict):
                    intrinsic_value = intrinsic_values.get(str(key), intrinsic_values.get(key))
                else:
                    intrinsic_value = intrinsic_values
            if intrinsic_value is not None:
                intrinsics[frame_id] = matrix(intrinsic_value, (3, 3), f"{path}:{frame_id}:intrinsic")
    elif isinstance(payload, dict):
        for fallback, (key, value) in enumerate(payload.items()):
            if not isinstance(value, dict):
                continue
            pose_value = value.get("aligned_pose", value.get("alignedPose"))
            if pose_value is None:
                continue
            frame_id = frame_id_from_record(key, value, fallback)
            poses[frame_id] = matrix(pose_value, (4, 4), f"{path}:{frame_id}:aligned_pose")
            intrinsic_value = value.get("intrinsic", value.get("intrinsics"))
            if intrinsic_value is not None:
                intrinsics[frame_id] = matrix(intrinsic_value, (3, 3), f"{path}:{frame_id}:intrinsic")
    else:
        raise ValueError(f"{path}: unsupported JSON root {type(payload).__name__}")

    if not poses:
        raise ValueError(f"{path}: no aligned camera-to-world poses found")
    if not intrinsics:
        raise ValueError(f"{path}: no RGB intrinsics found")
    return poses, intrinsics


def validate_pose(pose: np.ndarray, context: str) -> None:
    if not np.allclose(pose[3], [0.0, 0.0, 0.0, 1.0], atol=1e-4):
        raise ValueError(f"{context}: invalid homogeneous bottom row {pose[3].tolist()}")
    rotation = pose[:3, :3]
    error = float(np.linalg.norm(rotation.T @ rotation - np.eye(3), ord="fro"))
    determinant = float(np.linalg.det(rotation))
    if error > 2e-2 or not 0.98 <= determinant <= 1.02:
        raise ValueError(f"{context}: non-rigid pose (orth_error={error:.4g}, det={determinant:.4g})")


def aabb_corners(low: np.ndarray, high: np.ndarray) -> np.ndarray:
    return np.asarray(
        [[x, y, z] for x in (low[0], high[0]) for y in (low[1], high[1]) for z in (low[2], high[2])],
        dtype=np.float64,
    )


def build_ground_truth(
    mesh_path: Path,
    segments_path: Path,
    annotation_path: Path,
    instance_classes: set[str],
) -> tuple[np.ndarray, list[dict[str, Any]], list[str], np.ndarray]:
    ply = PlyData.read(str(mesh_path))
    vertex = ply["vertex"].data
    vertices = np.column_stack([vertex[axis] for axis in ("x", "y", "z")]).astype(np.float64)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or not np.isfinite(vertices).all():
        raise ValueError(f"{mesh_path}: invalid mesh vertices")
    segment_ids = np.asarray(json.loads(segments_path.read_text(encoding="utf-8"))["segIndices"], dtype=np.int64)
    if len(segment_ids) != len(vertices):
        raise ValueError(
            f"{segments_path}: {len(segment_ids)} segment IDs for {len(vertices)} mesh vertices"
        )
    groups = json.loads(annotation_path.read_text(encoding="utf-8")).get("segGroups", [])
    boxes, labels, ignored = [], [], []
    for group in groups:
        label_raw = str(group.get("label", "")).strip()
        label = label_raw.casefold()
        if label not in instance_classes:
            ignored.append(label_raw)
            continue
        ids = np.asarray(group.get("segments", []), dtype=np.int64)
        mask = np.isin(segment_ids, ids)
        points = vertices[mask]
        if len(points) == 0:
            raise ValueError(
                f"{annotation_path}: instance {group.get('objectId', group.get('id'))} "
                "has no mesh vertices"
            )
        low, high = points.min(axis=0), points.max(axis=0)
        if np.any(high <= low):
            ignored.append(label_raw + " [degenerate]")
            continue
        boxes.append(aabb_corners(low, high))
        labels.append(
            {
                "object_id": group.get("objectId", group.get("id")),
                "label": label_raw,
                "vertices": int(len(points)),
                "extent_xyz": (high - low).tolist(),
            }
        )
    ground_truth = np.stack(boxes) if boxes else np.zeros((0, 8, 3), dtype=np.float64)
    mesh_bounds = np.stack([vertices.min(axis=0), vertices.max(axis=0)])
    return ground_truth, labels, sorted(set(ignored)), mesh_bounds


def select_intrinsic(
    frame_ids: list[int],
    intrinsics: dict[int, np.ndarray],
    rgb_size: tuple[int, int],
    target_size: tuple[int, int],
) -> tuple[np.ndarray, float]:
    available = []
    last: np.ndarray | None = None
    for frame_id in frame_ids:
        if frame_id in intrinsics:
            last = intrinsics[frame_id]
        if last is None:
            following = [key for key in intrinsics if key >= frame_id]
            if following:
                last = intrinsics[min(following)]
        if last is None:
            raise ValueError(f"frame {frame_id}: no intrinsic at or near this frame")
        available.append(last)
    values = np.stack(available)
    median = np.median(values, axis=0)
    normalized = np.maximum(np.abs(median), 1.0)
    max_relative_drift = float(np.max(np.abs(values - median) / normalized))
    rgb_width, rgb_height = rgb_size
    target_width, target_height = target_size
    result = median.copy()
    result[0] *= target_width / rgb_width
    result[1] *= target_height / rgb_height
    result[2] = [0.0, 0.0, 1.0]
    if result[0, 0] <= 0 or result[1, 1] <= 0:
        raise ValueError("non-positive focal length")
    return result, max_relative_drift


def convert_scene(
    scene_id: str,
    alias: str,
    source: Path,
    target: Path,
    instance_classes: set[str],
    frame_stride: int,
    max_frames: int | None,
) -> dict[str, Any]:
    rgb_files = indexed_files(source / "iphone" / "rgb", (".jpg", ".jpeg", ".png"))
    depth_files = indexed_files(source / "iphone" / "depth", (".png",))
    if not rgb_files or not depth_files:
        raise FileNotFoundError(
            f"{scene_id}: decoded iphone/rgb or iphone/depth missing; run the official "
            "iphone.prepare_iphone_data extractor first"
        )
    pose_path = source / "iphone" / "pose_intrinsic_imu.json"
    poses, intrinsics = parse_pose_json(pose_path)
    complete = sorted(set(rgb_files) & set(depth_files) & set(poses))
    if not complete:
        raise ValueError(f"{scene_id}: no synchronized RGB-depth-aligned-pose frames")
    selected = complete[::frame_stride]
    if max_frames is not None:
        selected = selected[:max_frames]
    if not selected:
        raise ValueError(f"{scene_id}: frame selection is empty")

    first_rgb = cv2.imread(str(rgb_files[selected[0]]), cv2.IMREAD_COLOR)
    first_depth = cv2.imread(str(depth_files[selected[0]]), cv2.IMREAD_UNCHANGED)
    if first_rgb is None or first_depth is None or first_depth.ndim != 2:
        raise ValueError(f"{scene_id}: failed to decode first RGB-D frame")
    rgb_height, rgb_width = first_rgb.shape[:2]
    depth_height, depth_width = first_depth.shape
    target_size = (640, 480)
    target_k, intrinsic_drift = select_intrinsic(
        selected, intrinsics, (rgb_width, rgb_height), target_size
    )

    frames = target / "frames"
    for name in ("color", "depth", "pose", "intrinsic"):
        (frames / name).mkdir(parents=True, exist_ok=True)
    intrinsic4 = np.eye(4, dtype=np.float64)
    intrinsic4[:3, :3] = target_k
    np.savetxt(frames / "intrinsic" / "intrinsic_depth.txt", intrinsic4, fmt="%.10g")
    np.savetxt(frames / "K_depth.txt", target_k, fmt="%.10g")

    valid_depth_fractions = []
    for output_id, frame_id in enumerate(selected):
        color = cv2.imread(str(rgb_files[frame_id]), cv2.IMREAD_COLOR)
        depth = cv2.imread(str(depth_files[frame_id]), cv2.IMREAD_UNCHANGED)
        if color is None or color.shape[:2] != (rgb_height, rgb_width):
            raise ValueError(f"{scene_id}/frame_{frame_id}: inconsistent RGB shape")
        if depth is None or depth.ndim != 2 or depth.shape != (depth_height, depth_width):
            raise ValueError(f"{scene_id}/frame_{frame_id}: inconsistent depth shape")
        if depth.dtype != np.uint16:
            raise ValueError(f"{scene_id}/frame_{frame_id}: expected uint16 millimetre depth, got {depth.dtype}")
        pose = poses[frame_id]
        validate_pose(pose, f"{scene_id}/frame_{frame_id}")
        color_out = cv2.resize(color, target_size, interpolation=cv2.INTER_AREA)
        depth_out = cv2.resize(depth, target_size, interpolation=cv2.INTER_NEAREST)
        valid_depth_fractions.append(float(np.count_nonzero(depth_out) / depth_out.size))
        if not cv2.imwrite(str(frames / "color" / f"{output_id}.jpg"), color_out, [cv2.IMWRITE_JPEG_QUALITY, 95]):
            raise RuntimeError(f"failed to write {scene_id} color frame {output_id}")
        if not cv2.imwrite(str(frames / "depth" / f"{output_id}.png"), depth_out):
            raise RuntimeError(f"failed to write {scene_id} depth frame {output_id}")
        np.savetxt(frames / "pose" / f"{output_id}.txt", pose, fmt="%.10g")

    scans = source / "scans"
    mesh_path = scans / "mesh_aligned_0.05.ply"
    segments_path = scans / "segments.json"
    annotation_path = scans / "segments_anno.json"
    for required in (mesh_path, segments_path, annotation_path):
        if not required.is_file():
            raise FileNotFoundError(required)
    ground_truth, gt_labels, ignored_labels, mesh_bounds = build_ground_truth(
        mesh_path, segments_path, annotation_path, instance_classes
    )
    np.save(target / "gt_aabb_instances.npy", ground_truth)

    row: dict[str, Any] = {
        "alias": alias,
        "scene_id": scene_id,
        "frames": len(selected),
        "source_frame_ids": selected,
        "complete_source_frames": len(complete),
        "frame_stride": frame_stride,
        "raw_rgb_size": [rgb_width, rgb_height],
        "raw_depth_size": [depth_width, depth_height],
        "target_size": list(target_size),
        "target_intrinsic": target_k.tolist(),
        "intrinsic_max_relative_drift": intrinsic_drift,
        "valid_depth_fraction_min": min(valid_depth_fractions),
        "valid_depth_fraction_median": float(np.median(valid_depth_fractions)),
        "gt_instances": len(ground_truth),
        "gt_labels": gt_labels,
        "ignored_noninstance_labels": ignored_labels,
        "mesh_bounds": mesh_bounds.tolist(),
        "pose_semantics": "aligned camera-to-world in official mesh coordinates; +Z camera forward",
        "source_sha256": {
            "pose_intrinsic_imu.json": sha256(pose_path),
            "mesh_aligned_0.05.ply": sha256(mesh_path),
            "segments.json": sha256(segments_path),
            "segments_anno.json": sha256(annotation_path),
        },
    }
    (target / "metadata.json").write_text(json.dumps(row, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--scene-list", type=Path, required=True)
    parser.add_argument("--instance-classes", type=Path)
    parser.add_argument("--alias-start", type=int, default=9000)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.frame_stride < 1:
        parser.error("--frame-stride must be >= 1")
    if args.max_frames is not None and args.max_frames < 1:
        parser.error("--max-frames must be >= 1")

    requested_ids = read_ids(args.scene_list.resolve())
    if args.limit is not None:
        requested_ids = requested_ids[: args.limit]
    raw_root = args.raw_root.resolve()
    data_root = locate_data_root(raw_root, requested_ids[0])
    classes_path = locate_instance_classes(raw_root, data_root, args.instance_classes)
    instance_classes = {
        line.strip().casefold()
        for line in classes_path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    }
    if not instance_classes:
        raise ValueError(f"{classes_path}: no instance classes")
    missing = [scene_id for scene_id in requested_ids if not (data_root / scene_id).is_dir()]
    if missing:
        raise FileNotFoundError(f"missing {len(missing)} requested ScanNet++ scenes; first: {missing[:3]}")

    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    rows = []
    for index, scene_id in enumerate(requested_ids):
        alias = f"scene{args.alias_start + index:04d}_00"
        target = output_root / alias
        if args.resume and (target / "metadata.json").is_file():
            row = json.loads((target / "metadata.json").read_text(encoding="utf-8"))
            if row.get("alias") == alias and row.get("scene_id") == scene_id:
                rows.append(row)
                print(f"{alias} <- {scene_id}: resumed frames={row['frames']} gt={row['gt_instances']}", flush=True)
                continue
        if target.exists() and not args.overwrite:
            raise FileExistsError(f"refusing to overwrite {target}")
        with tempfile.TemporaryDirectory(prefix=f".{alias}.", dir=output_root) as temp_text:
            staging = Path(temp_text) / alias
            staging.mkdir()
            row = convert_scene(
                scene_id,
                alias,
                data_root / scene_id,
                staging,
                instance_classes,
                args.frame_stride,
                args.max_frames,
            )
            if target.exists():
                shutil.rmtree(target)
            os.replace(staging, target)
        rows.append(row)
        print(f"{alias} <- {scene_id}: frames={row['frames']} gt={row['gt_instances']}", flush=True)

    manifest = {
        "protocol": "ScanNet++ v2 nvs_sem_val instance-class class-agnostic world-AABB AP",
        "split_file": str(args.scene_list.resolve()),
        "split_sha256": sha256(args.scene_list.resolve()),
        "instance_classes_file": str(classes_path),
        "instance_classes_sha256": sha256(classes_path),
        "coordinate_frame": "official aligned mesh world frame",
        "image_conversion": "aligned RGB and depth independently resized to 640x480",
        "depth_unit": "millimetres on disk; BoxFusion divides by 1000",
        "frame_stride": args.frame_stride,
        "scans": rows,
    }
    (output_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output_root / "scenes.txt").write_text(
        "".join(row["alias"] + "\n" for row in rows), encoding="utf-8"
    )
    print(f"wrote {output_root / 'manifest.json'}", flush=True)


if __name__ == "__main__":
    main()
