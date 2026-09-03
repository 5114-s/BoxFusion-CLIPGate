#!/usr/bin/env python3
"""Build one isolated, explicitly non-canonical CA-1M scene.

This tool is only for the four CA-1M validation scenes whose author-published
``after_filter_boxes.npy`` is absent.  It reconstructs the processed RGB-D
layout from an official Apple tar and derives, rather than invents, the missing
filtered GT.  The output is never promoted into the canonical data root.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import tarfile
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
from scipy.spatial import KDTree

import convert_ca1m_apple_tar as apple


SCHEMA = "boxfusion.ca1m_derived_missing_scene.v1"
MISSING4 = {"45663164", "47115469", "47331311", "47332000"}


def atomic_bytes(path: Path, payload: bytes) -> None:
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_json(path: Path, payload: dict) -> None:
    atomic_bytes(path, (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode())


def save_npy(path: Path, value: np.ndarray) -> None:
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("xb") as handle:
        np.save(handle, value)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def threshold_name(value: float) -> str:
    return f"after_filter_boxes.threshold_{value:.3f}".replace(".", "p") + ".npy"


def parse_thresholds(text: str) -> tuple[float, ...]:
    values = tuple(float(row.strip()) for row in text.split(",") if row.strip())
    if not values or any(not math.isfinite(x) or x <= 0 for x in values):
        raise argparse.ArgumentTypeError("thresholds must be positive finite values")
    if len(values) != len(set(values)):
        raise argparse.ArgumentTypeError("thresholds must be unique")
    return values


def frustum_mask(
    corners: np.ndarray,
    intrinsic: np.ndarray,
    poses: np.ndarray,
    image_shape: tuple[int, int],
    near: float,
    far: float,
) -> np.ndarray:
    """Author protocol: retain a box when at least six corners are visible."""
    height, width = image_shape
    homogeneous = np.concatenate(
        (corners, np.ones((len(corners), 8, 1), dtype=corners.dtype)), axis=-1
    )
    visible = np.zeros((len(corners), 8), dtype=bool)
    fx, fy = intrinsic[0, 0], intrinsic[1, 1]
    cx, cy = intrinsic[0, 2], intrinsic[1, 2]
    for pose in poses:
        camera = homogeneous @ np.linalg.inv(pose).T
        x, y, z = camera[..., 0], camera[..., 1], camera[..., 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            u = (fx * x / z + cx).astype(np.int64)
            v = (fy * y / z + cy).astype(np.int64)
        visible |= (
            (z > near)
            & (z < far)
            & (u >= 0)
            & (u < width)
            & (v >= 0)
            & (v < height)
        )
    return visible.sum(axis=1) >= 6


def first_voxel_surface(
    archive: tarfile.TarFile,
    scene_id: str,
    ids: tuple[str, ...],
    rotations: np.ndarray,
    intrinsic: np.ndarray,
    poses: np.ndarray,
    pixel_stride: int,
    voxel_size: float,
    depth_scale: float,
    max_depth: float,
) -> np.ndarray:
    """Create a deterministic first-in-acquisition-order world surface proxy."""
    first: dict[tuple[int, int, int], tuple[float, float, float]] = {}
    for frame_index, (frame_id, rotation) in enumerate(zip(ids, rotations)):
        raw = apple.member_bytes(
            archive, f"{scene_id}/{frame_id}.gt/depth.png"
        )
        depth = apple.decode_png(raw, f"{scene_id}/{frame_id} depth")
        if int(rotation):
            depth = np.ascontiguousarray(np.rot90(depth, int(rotation)))
        depth = depth.astype(np.float64) / depth_scale
        vv = np.arange(0, depth.shape[0], pixel_stride, dtype=np.int64)
        uu = np.arange(0, depth.shape[1], pixel_stride, dtype=np.int64)
        grid_u, grid_v = np.meshgrid(uu, vv, indexing="xy")
        z = depth[grid_v, grid_u]
        good = np.isfinite(z) & (z > 0.0) & (z < max_depth)
        if not good.any():
            continue
        u = grid_u[good].astype(np.float64)
        v = grid_v[good].astype(np.float64)
        z = z[good]
        camera = np.column_stack(
            (
                (u - intrinsic[0, 2]) * z / intrinsic[0, 0],
                (v - intrinsic[1, 2]) * z / intrinsic[1, 1],
                z,
                np.ones_like(z),
            )
        )
        world = (camera @ poses[frame_index].T)[:, :3]
        keys = np.floor(world / voxel_size).astype(np.int64)
        # Pixel arrays are row-major and frames follow the sorted raw-frame order.
        for key, point in zip(keys, world):
            packed = (int(key[0]), int(key[1]), int(key[2]))
            if packed not in first:
                first[packed] = (float(point[0]), float(point[1]), float(point[2]))
    if not first:
        raise ValueError(f"{scene_id}: surface proxy is empty")
    return np.asarray(tuple(first.values()), dtype=np.float32)


def proximity_indices(
    corners: np.ndarray,
    candidate_indices: np.ndarray,
    surface: np.ndarray,
    threshold: float,
) -> np.ndarray:
    """Author protocol: >=4 corners have strictly less than threshold distance."""
    tree = KDTree(surface)
    kept: list[int] = []
    for index in candidate_indices:
        distances, _ = tree.query(corners[int(index)], k=1)
        if int(np.count_nonzero(distances < threshold)) >= 4:
            kept.append(int(index))
    return np.asarray(kept, dtype=np.int64)


def build(args: argparse.Namespace) -> dict:
    tar_path = args.tar.resolve()
    match = re.search(r"ca1m-val-(\d+)\.tar$", tar_path.name)
    scene_id = args.scene_id or (match.group(1) if match else None)
    if scene_id not in MISSING4:
        raise ValueError(f"scene must be one of the frozen missing four: {sorted(MISSING4)}")
    if not tar_path.is_file() or tar_path.is_symlink():
        raise ValueError(f"missing/non-regular Apple tar: {tar_path}")
    staging_root = args.staging_root.resolve()
    staging_root.mkdir(parents=True, exist_ok=True)
    output = staging_root / scene_id
    building = staging_root / f".{scene_id}.building.{os.getpid()}"
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"refusing to overwrite output scene: {output}")
    if building.exists() or building.is_symlink():
        raise FileExistsError(building)
    author_mesh = args.author_mesh.resolve() if args.author_mesh else None
    if author_mesh is not None and (not author_mesh.is_file() or author_mesh.is_symlink()):
        raise ValueError(f"invalid author mesh: {author_mesh}")
    source_kind = "author_mesh" if author_mesh is not None else "voxelized_depth_surface_proxy"
    if (scene_id == "45663164") != (author_mesh is not None):
        raise ValueError(
            "the frozen protocol requires the published author mesh only for "
            "45663164 and the depth-surface proxy for the other three scenes"
        )
    if args.primary_threshold not in args.sensitivity_thresholds:
        raise ValueError("primary threshold must be included in sensitivity thresholds")

    building.mkdir()
    (building / "rgb").mkdir()
    (building / "depth").mkdir()
    (building / "instances").mkdir()
    try:
        with tarfile.open(tar_path, mode="r:") as archive:
            ids = apple.frame_ids(archive, scene_id)
            shapes = tuple(apple.depth_shape(archive, scene_id, fid) for fid in ids)
            raw_poses = np.stack(
                [apple.parse_json_member(archive, f"{scene_id}/{fid}.gt/RT.json") for fid in ids]
            )
            apple.validate_poses(raw_poses, scene_id)
            orientation_rule, orientation_policy = apple.load_orientation_policy(
                args.orientation_policy, scene_id, tar_path
            )
            if orientation_rule["method"] == "pose_continuity":
                rotations, orientation = apple.infer_rot90(
                    raw_poses, shapes, float(orientation_rule["min_margin_degrees"])
                )
                orientation["method"] = "pose_continuity"
            else:
                rotations, orientation = apple.infer_rot90_aspect_clockwise_to_portrait(shapes)
            metadata = apple.expected_raw_metadata(
                archive, scene_id, ids, shapes, orientation["target_orientation"]
            )
            intrinsic = np.asarray(metadata["K_depth.txt"], dtype=np.float64)

            target_rgb_shape = None
            target_depth_shape = None
            for index, (frame_id, rotation) in enumerate(zip(ids, rotations)):
                rgb_raw = apple.member_bytes(archive, f"{scene_id}/{frame_id}.wide/image.png")
                depth_raw = apple.member_bytes(archive, f"{scene_id}/{frame_id}.gt/depth.png")
                rgb, rgb_shape = apple.encoded_rotated_png(
                    rgb_raw, int(rotation), f"{scene_id}/{frame_id} RGB"
                )
                depth, depth_shape = apple.encoded_rotated_png(
                    depth_raw, int(rotation), f"{scene_id}/{frame_id} depth"
                )
                if target_rgb_shape is None:
                    target_rgb_shape, target_depth_shape = rgb_shape[:2], depth_shape[:2]
                if rgb_shape[:2] != target_rgb_shape or depth_shape[:2] != target_depth_shape:
                    raise ValueError(f"{scene_id}: inconsistent rotated frame dimensions")
                atomic_bytes(building / "rgb" / f"{index}.png", rgb)
                atomic_bytes(building / "depth" / f"{index}.png", depth)
                frame_instances = apple.member_bytes(
                    archive, f"{scene_id}/{frame_id}.wide/instances.json"
                )
                atomic_bytes(building / "instances" / f"{index}.json", frame_instances)

            world_name = f"{scene_id}/world.gt/instances.json"
            world_payload = apple.member_bytes(archive, world_name)
            atomic_bytes(building / "instances.json", world_payload)
            instance_rows = json.loads(world_payload)
            corners = np.stack([np.asarray(row["corners"], dtype=np.float64) for row in instance_rows])
            poses = np.asarray(metadata["all_poses.npy"], dtype=np.float64)
            gravity = np.asarray(metadata["T_gravity.npy"])
            save_npy(building / "all_poses.npy", poses)
            save_npy(building / "T_gravity.npy", gravity)
            np.savetxt(building / "K_rgb.txt", metadata["K_rgb.txt"])
            np.savetxt(building / "K_depth.txt", intrinsic)

            assert target_depth_shape is not None
            visible = frustum_mask(
                corners, intrinsic, poses, target_depth_shape, args.near, args.far
            )
            candidate_indices = np.flatnonzero(visible)
            if author_mesh is not None:
                shutil.copyfile(author_mesh, building / "mesh.ply")
                # Match the author's filter_gt_boxes.py exactly: it loads the
                # PLY as a point cloud and queries those point coordinates.
                point_cloud = o3d.io.read_point_cloud(str(author_mesh))
                surface = np.asarray(point_cloud.points, dtype=np.float64)
                if not len(surface) or not np.isfinite(surface).all():
                    raise ValueError(f"{scene_id}: author mesh has invalid vertices")
                surface_artifact = "mesh.ply"
            else:
                surface = first_voxel_surface(
                    archive,
                    scene_id,
                    ids,
                    rotations,
                    intrinsic,
                    poses,
                    args.pixel_stride,
                    args.voxel_size,
                    args.depth_scale,
                    args.max_depth,
                )
                save_npy(building / "surface_proxy.npy", surface)
                surface_artifact = "surface_proxy.npy"

        sensitivity: dict[str, dict] = {}
        main_indices = None
        for threshold in args.sensitivity_thresholds:
            indices = proximity_indices(corners, candidate_indices, surface, threshold)
            name = threshold_name(threshold)
            # Keep float64, matching np.asarray(JSON corners) + np.save in the
            # released author preprocessing script.
            save_npy(building / name, corners[indices])
            sensitivity[f"{threshold:.3f}"] = {
                "artifact": name,
                "kept_count": int(len(indices)),
                "kept_indices": indices.tolist(),
            }
            if math.isclose(threshold, args.primary_threshold, abs_tol=1e-12):
                main_indices = indices
        if main_indices is None:
            main_indices = proximity_indices(
                corners, candidate_indices, surface, args.primary_threshold
            )
        save_npy(building / "after_filter_boxes.npy", corners[main_indices])

        artifacts = {}
        for path in sorted(p for p in building.iterdir() if p.is_file()):
            artifacts[path.name] = {"sha256": apple.sha256(path), "bytes": path.stat().st_size}
        manifest = {
            "schema": SCHEMA,
            "derived": True,
            "official_comparable": False,
            "paper_claim_permitted": False,
            "scene_id": scene_id,
            "output_scene": str(output),
            "source_kind": source_kind,
            "apple_tar": {"path": str(tar_path), "sha256": apple.sha256(tar_path)},
            "author_mesh": (
                {"path": str(author_mesh), "sha256": apple.sha256(author_mesh)}
                if author_mesh is not None
                else None
            ),
            "orientation": orientation,
            "orientation_policy": orientation_policy,
            "parameters": {
                "frustum_min_visible_corners": 6,
                "near_m": args.near,
                "far_m": args.far,
                "proximity_min_corners": 4,
                "proximity_comparison": "strict_less_than",
                "primary_threshold_m": args.primary_threshold,
                "sensitivity_thresholds_m": list(args.sensitivity_thresholds),
                "pixel_stride": args.pixel_stride,
                "voxel_size_m": args.voxel_size,
                "depth_scale": args.depth_scale,
                "max_depth_m_exclusive": args.max_depth,
                "voxel_representative": "first_in_frame_row_major_acquisition_order",
            },
            "counts": {
                "frames": len(ids),
                "raw_boxes": len(corners),
                "frustum_boxes": int(len(candidate_indices)),
                "kept_boxes": int(len(main_indices)),
                "surface_points": int(len(surface)),
            },
            "frustum_indices": candidate_indices.tolist(),
            "kept_indices": main_indices.tolist(),
            "sensitivity": sensitivity,
            "artifacts": artifacts,
            "gt_sha256": artifacts["after_filter_boxes.npy"]["sha256"],
        }
        atomic_json(building / "derived_gt_manifest.json", manifest)
        os.replace(building, output)
        print(json.dumps(manifest, indent=2, sort_keys=True))
        return manifest
    except Exception:
        if building.exists():
            failed = staging_root / f".{scene_id}.failed.{os.getpid()}"
            if not failed.exists():
                os.replace(building, failed)
        raise


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--tar", type=Path, required=True)
    result.add_argument("--scene-id")
    result.add_argument("--staging-root", type=Path, required=True)
    result.add_argument("--author-mesh", type=Path)
    result.add_argument("--orientation-policy", type=Path)
    result.add_argument("--near", type=float, default=0.1)
    result.add_argument("--far", type=float, default=100.0)
    result.add_argument("--pixel-stride", type=int, default=4)
    result.add_argument("--voxel-size", type=float, default=0.02)
    result.add_argument("--depth-scale", type=float, default=1000.0)
    result.add_argument("--max-depth", type=float, default=10.0)
    result.add_argument("--primary-threshold", type=float, default=0.10)
    result.add_argument(
        "--sensitivity-thresholds", type=parse_thresholds, default=parse_thresholds("0.08,0.10,0.12")
    )
    return result


def main() -> int:
    args = parser().parse_args()
    scalars = (args.near, args.far, args.voxel_size, args.depth_scale, args.max_depth, args.primary_threshold)
    if any(not math.isfinite(x) or x <= 0 for x in scalars):
        raise ValueError("all numeric geometry parameters must be positive and finite")
    if args.near >= args.far or args.pixel_stride <= 0:
        raise ValueError("invalid near/far or pixel stride")
    build(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
