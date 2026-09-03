#!/usr/bin/env python3
"""Build one isolated CA-1M train scene for CA-1M-native B6 training.

The builder is deliberately single-scene and fail-closed.  It consumes one
official Apple *train* tar selected by the frozen train100 manifest, never
reads validation ground truth, and never writes to the validation/live data
root.  RGB-D conversion follows the audited Apple converter.  The training GT
is explicitly marked as derived: world.gt boxes are filtered with a
deterministic depth-surface proxy and the author's frustum/corner-proximity
rules.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import stat
import tarfile
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial import cKDTree

import convert_ca1m_apple_tar as apple


SCHEMA = "boxfusion.ca1m_native_b6_train_scene.v1"
SUBSET_SCHEMA = "boxfusion.ca1m_native_b6_train_subset.v1"
PER_FRAME_K_SCHEMA = "boxfusion.ca1m_native_b6_per_frame_intrinsics.v1"
FRAME_MANIFEST_SCHEMA = "boxfusion.ca1m_native_b6_train_frames.v1"
TRAIN_TAR_RE = re.compile(r"ca1m-train-([0-9]{8})\.tar$")
VAL_URL_RE = re.compile(
    r"https://ml-site\.cdn-apple\.com/datasets/ca1m/val/ca1m-val-([0-9]{8})\.tar"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json(payload: Any) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")


def atomic_bytes(path: Path, payload: bytes) -> None:
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_json(path: Path, payload: Any) -> None:
    atomic_bytes(path, canonical_json(payload))


def save_npy(path: Path, value: np.ndarray) -> None:
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("xb") as handle:
        np.save(handle, value, allow_pickle=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def save_npz(path: Path, **values: np.ndarray) -> None:
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}.npz")
    with temporary.open("xb") as handle:
        np.savez(handle, **values)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def save_txt(path: Path, value: np.ndarray) -> None:
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("x", encoding="utf-8") as handle:
        np.savetxt(handle, value)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def require_regular(path: Path, label: str) -> None:
    try:
        mode = path.lstat().st_mode
    except FileNotFoundError as exc:
        raise ValueError(f"missing {label}: {path}") from exc
    if path.is_symlink() or not stat.S_ISREG(mode) or path.stat().st_size <= 0:
        raise ValueError(f"{label} must be a non-empty regular file: {path}")


def storage_filesystem_policy(path: Path) -> dict[str, Any]:
    """Describe whether the destination can represent POSIX write bits."""
    resolved = path.resolve()
    best: tuple[int, str, str, list[str]] | None = None
    try:
        lines = Path("/proc/self/mountinfo").read_text(encoding="utf-8").splitlines()
    except OSError:
        lines = []
    for line in lines:
        if " - " not in line:
            continue
        left, right = line.split(" - ", 1)
        before, after = left.split(), right.split()
        if len(before) < 6 or len(after) < 3:
            continue
        mount = Path(
            before[4]
            .replace("\\040", " ")
            .replace("\\011", "\t")
            .replace("\\012", "\n")
            .replace("\\134", "\\")
        )
        try:
            resolved.relative_to(mount)
        except ValueError:
            continue
        row = (len(str(mount)), str(mount), after[0], before[5].split(","))
        if best is None or row[0] > best[0]:
            best = row
    if best is None:
        return {
            "mount_point": None,
            "filesystem_type": "unknown",
            "mount_options": [],
            "posix_mode_enforceable": False,
            "artifact_integrity_contract": "regular_no_symlink_sha256_create_only",
        }
    _, mount_point, filesystem, options = best
    lacks_posix_modes = filesystem.lower() in {
        "fuseblk",
        "ntfs",
        "ntfs3",
        "vfat",
        "msdos",
        "exfat",
    }
    return {
        "mount_point": mount_point,
        "filesystem_type": filesystem,
        "mount_options": options,
        "posix_mode_enforceable": not lacks_posix_modes,
        "artifact_integrity_contract": (
            "regular_no_symlink_sha256_create_only"
            if lacks_posix_modes
            else "regular_no_symlink_sha256_create_only_and_no_write_bits"
        ),
    }


def load_train_contract(
    subset_manifest: Path, val_url_list: Path, tar_path: Path, scene_id: str | None
) -> tuple[str, dict[str, Any], dict[str, Any], set[str]]:
    require_regular(subset_manifest, "frozen train subset manifest")
    require_regular(val_url_list, "official validation URL list")
    require_regular(tar_path, "official Apple train tar")
    matched = TRAIN_TAR_RE.fullmatch(tar_path.name)
    if matched is None:
        raise ValueError(f"train tar name is not canonical: {tar_path.name}")
    inferred = matched.group(1)
    selected_scene = scene_id or inferred
    if not re.fullmatch(r"[0-9]{8}", selected_scene) or selected_scene != inferred:
        raise ValueError("explicit, filename, and inferred train scene IDs disagree")

    manifest = json.loads(subset_manifest.read_text(encoding="utf-8"))
    if manifest.get("schema") != SUBSET_SCHEMA:
        raise ValueError("unsupported frozen subset manifest schema")
    safety = manifest.get("safety_contract", {})
    if safety.get("train_only") is not True:
        raise ValueError("subset manifest is not train-only")
    if safety.get("validation_ground_truth_access") is not False:
        raise ValueError("subset manifest permits validation ground-truth access")
    if int(safety.get("validation_scene_overlap_count", -1)) != 0:
        raise ValueError("subset manifest reports validation overlap")
    source = manifest.get("source", {})
    if source.get("train_val_overlap") != []:
        raise ValueError("source train/validation lists overlap")
    if source.get("val_url_list_sha256") != sha256_file(val_url_list):
        raise ValueError("official validation URL list hash differs from frozen manifest")
    entries = manifest.get("entries")
    if not isinstance(entries, list):
        raise ValueError("subset manifest entries are invalid")
    rows = [row for row in entries if row.get("scene_id") == selected_scene]
    if len(rows) != 1:
        raise ValueError(f"{selected_scene}: scene is not uniquely frozen in train subset")
    entry = rows[0]
    if entry.get("tar_name") != tar_path.name:
        raise ValueError("frozen tar name differs from input")
    if f"/train/{tar_path.name}" not in str(entry.get("url", "")):
        raise ValueError("frozen entry is not an official train URL")

    val_ids: set[str] = set()
    for line_number, raw in enumerate(
        val_url_list.read_text(encoding="utf-8").splitlines(), 1
    ):
        value = raw.strip()
        if not value:
            continue
        match = VAL_URL_RE.fullmatch(value)
        if match is None:
            raise ValueError(f"invalid official validation URL at line {line_number}")
        if match.group(1) in val_ids:
            raise ValueError("duplicate validation scene ID")
        val_ids.add(match.group(1))
    if not val_ids:
        raise ValueError("official validation URL list is empty")
    if selected_scene in val_ids:
        raise ValueError(f"train scene overlaps validation split: {selected_scene}")
    return selected_scene, manifest, entry, val_ids


def validate_tar_members(archive: tarfile.TarFile, scene_id: str) -> None:
    members = archive.getmembers()
    if not members:
        raise ValueError("empty train tar")
    names: set[str] = set()
    prefix = f"{scene_id}/"
    for member in members:
        name = member.name
        if (
            name in names
            or name.startswith("/")
            or ".." in Path(name).parts
            or not name.startswith(prefix)
            or not member.isfile()
        ):
            raise ValueError(f"unsafe or unsupported tar member: {name}")
        names.add(name)
    world = f"{scene_id}/world.gt/instances.json"
    if world not in names:
        raise ValueError(f"missing train-only world GT: {world}")


def validate_world_instances(payload: bytes, scene_id: str) -> tuple[list[dict], np.ndarray]:
    rows = json.loads(payload)
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"{scene_id}: world instances must be a non-empty list")
    corners: list[np.ndarray] = []
    seen_ids: set[str] = set()
    for index, row in enumerate(rows):
        if not isinstance(row, dict) or not isinstance(row.get("id"), str):
            raise ValueError(f"{scene_id}: malformed world instance row {index}")
        if row["id"] in seen_ids:
            raise ValueError(f"{scene_id}: duplicate world instance ID")
        seen_ids.add(row["id"])
        value = np.asarray(row.get("corners"), dtype=np.float64)
        if value.shape != (8, 3) or not np.isfinite(value).all():
            raise ValueError(f"{scene_id}: invalid world corners at row {index}")
        corners.append(value)
    return rows, np.stack(corners)


def processed_intrinsics(
    raw: np.ndarray,
    raw_shapes: tuple[tuple[int, int], ...],
    rotations: np.ndarray,
    *,
    image_scale: int = 1,
) -> np.ndarray:
    """Normalize Apple per-frame K into the converter's unified orientation.

    Apple's principal-point convention uses image dimensions (rather than the
    last integer pixel index) when the encoded device orientation flips.  This
    transformation reproduces the target-orientation K observed in adjacent
    Apple frames while retaining each frame's own calibration.
    """
    values = np.asarray(raw, dtype=np.float64)
    if values.shape != (len(raw_shapes), 3, 3) or len(rotations) != len(values):
        raise ValueError("per-frame intrinsic normalization cardinality mismatch")
    result = values.copy()
    for index, (shape, rotation) in enumerate(zip(raw_shapes, rotations)):
        height, width = int(shape[0]) * image_scale, int(shape[1]) * image_scale
        k = int(rotation) % 4
        fx, fy = values[index, 0, 0], values[index, 1, 1]
        cx, cy = values[index, 0, 2], values[index, 1, 2]
        if k == 0:
            continue
        if k == 2:
            result[index] = ((fx, 0.0, width - cx), (0.0, fy, height - cy), (0.0, 0.0, 1.0))
        else:
            # For Apple's positive-focal K convention, both odd cardinal
            # rotations swap axes and mirror each principal coordinate about
            # the corresponding raw image dimension.  Rotation direction is
            # carried by rot90/RT, not by a negative focal length.  This is
            # independently checked against released mixed-orientation scenes.
            result[index] = ((fy, 0.0, height - cy), (0.0, fx, width - cx), (0.0, 0.0, 1.0))
    if (
        not np.isfinite(result).all()
        or np.any(result[:, 0, 0] <= 0)
        or np.any(result[:, 1, 1] <= 0)
        or not np.allclose(result[:, 2], (0.0, 0.0, 1.0), atol=1e-12, rtol=0)
    ):
        raise ValueError("normalized per-frame intrinsics are invalid")
    return result


def frustum_indices(
    corners: np.ndarray,
    intrinsic: np.ndarray,
    poses: np.ndarray,
    image_shape: tuple[int, int],
    near: float,
    far: float,
) -> np.ndarray:
    """Author filter: a box has at least six corners visible across frames."""
    height, width = image_shape
    homogeneous = np.concatenate(
        (corners, np.ones((len(corners), 8, 1), dtype=np.float64)), axis=-1
    )
    visible = np.zeros((len(corners), 8), dtype=bool)
    intrinsics = np.asarray(intrinsic, dtype=np.float64)
    if intrinsics.shape == (3, 3):
        intrinsics = np.repeat(intrinsics[None], len(poses), axis=0)
    if intrinsics.shape != (len(poses), 3, 3):
        raise ValueError("frustum per-frame intrinsic cardinality mismatch")
    for pose, frame_k in zip(poses, intrinsics):
        fx, fy = frame_k[0, 0], frame_k[1, 1]
        cx, cy = frame_k[0, 2], frame_k[1, 2]
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
    return np.flatnonzero(visible.sum(axis=1) >= 6).astype(np.int64)


def first_voxel_surface(
    archive: tarfile.TarFile,
    scene_id: str,
    frame_ids: tuple[str, ...],
    rotations: np.ndarray,
    intrinsic: np.ndarray,
    poses: np.ndarray,
    pixel_stride: int,
    voxel_size: float,
    depth_scale: float,
    max_depth: float,
) -> np.ndarray:
    """Deterministic first-in-frame/row-major depth surface proxy."""
    first: dict[tuple[int, int, int], tuple[float, float, float]] = {}
    intrinsics = np.asarray(intrinsic, dtype=np.float64)
    if intrinsics.shape == (3, 3):
        intrinsics = np.repeat(intrinsics[None], len(frame_ids), axis=0)
    if intrinsics.shape != (len(frame_ids), 3, 3):
        raise ValueError("surface per-frame intrinsic cardinality mismatch")
    for frame_index, (frame_id, rotation) in enumerate(zip(frame_ids, rotations)):
        raw = apple.member_bytes(archive, f"{scene_id}/{frame_id}.gt/depth.png")
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
        frame_k = intrinsics[frame_index]
        camera = np.column_stack(
            (
                (u - frame_k[0, 2]) * z / frame_k[0, 0],
                (v - frame_k[1, 2]) * z / frame_k[1, 1],
                z,
                np.ones_like(z),
            )
        )
        world = (camera @ poses[frame_index].T)[:, :3]
        keys = np.floor(world / voxel_size).astype(np.int64)
        for key, point in zip(keys, world):
            packed = int(key[0]), int(key[1]), int(key[2])
            if packed not in first:
                first[packed] = float(point[0]), float(point[1]), float(point[2])
    if not first:
        raise ValueError(f"{scene_id}: depth surface proxy is empty")
    return np.asarray(tuple(first.values()), dtype=np.float32)


def proximity_indices(
    corners: np.ndarray,
    candidates: np.ndarray,
    surface: np.ndarray,
    threshold: float,
) -> np.ndarray:
    """Author filter: >=4 box corners have strict distance < threshold."""
    tree = cKDTree(np.asarray(surface, dtype=np.float64))
    kept: list[int] = []
    for index in candidates:
        distances, _ = tree.query(corners[int(index)], k=1)
        if int(np.count_nonzero(distances < threshold)) >= 4:
            kept.append(int(index))
    return np.asarray(kept, dtype=np.int64)


def artifact_inventory(root: Path) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for path in sorted(value for value in root.rglob("*") if value.is_file()):
        relative = str(path.relative_to(root))
        result[relative] = {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
    return result


def _preflight_tar(tar_path: Path, scene_id: str) -> tuple[int, int]:
    with tarfile.open(tar_path, mode="r:") as archive:
        validate_tar_members(archive, scene_id)
        ids = apple.frame_ids(archive, scene_id)
        world = apple.member_bytes(archive, f"{scene_id}/world.gt/instances.json")
        rows, _ = validate_world_instances(world, scene_id)
    return len(ids), len(rows)


def build(args: argparse.Namespace) -> dict[str, Any]:
    tar_path = args.tar.resolve()
    subset_manifest = args.subset_manifest.resolve()
    val_url_list = args.val_url_list.resolve()
    scene_id, subset, entry, val_ids = load_train_contract(
        subset_manifest, val_url_list, tar_path, args.scene_id
    )
    frame_count, raw_box_count = _preflight_tar(tar_path, scene_id)
    common = {
        "schema": SCHEMA,
        "mode": args.mode,
        "scene_id": scene_id,
        "source_split": "train",
        "train_only": True,
        "validation_scene_overlap": False,
        "validation_ground_truth_access": False,
        "derived_train_gt": True,
        "official_validation_comparable": False,
        "paper_validation_claim_permitted": False,
        "preflight": {"frames": frame_count, "raw_world_boxes": raw_box_count},
    }
    if args.mode == "preflight":
        common.update(
            {
                "output_created": False,
                "source_tar": {"path": str(tar_path), "bytes": tar_path.stat().st_size},
                "frozen_subset_manifest": {
                    "path": str(subset_manifest),
                    "sha256": sha256_file(subset_manifest),
                    "rank": int(entry["rank"]),
                },
                "validation_scene_count": len(val_ids),
            }
        )
        print(json.dumps(common, indent=2, sort_keys=True))
        return common

    output_root = args.output_root.resolve()
    if output_root.is_symlink():
        raise ValueError(f"refusing symlink output root: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    output = output_root / scene_id
    building = output_root / f".{scene_id}.building.{os.getpid()}"
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"refusing to overwrite train scene: {output}")
    if building.exists() or building.is_symlink():
        raise FileExistsError(building)

    building.mkdir(mode=0o755)
    (building / "rgb").mkdir()
    (building / "depth").mkdir()
    try:
        tar_hash = sha256_file(tar_path)
        with tarfile.open(tar_path, mode="r:") as archive:
            validate_tar_members(archive, scene_id)
            ids = apple.frame_ids(archive, scene_id)
            shapes = tuple(apple.depth_shape(archive, scene_id, value) for value in ids)
            poses = np.stack(
                [apple.parse_json_member(archive, f"{scene_id}/{value}.gt/RT.json") for value in ids]
            )
            gravity = np.stack(
                [apple.parse_json_member(archive, f"{scene_id}/{value}.wide/T_gravity.json") for value in ids]
            )
            depth_k_raw = np.stack(
                [apple.parse_json_member(archive, f"{scene_id}/{value}.gt/depth/K.json") for value in ids]
            )
            rgb_k_raw = np.stack(
                [apple.parse_json_member(archive, f"{scene_id}/{value}.gt/image/K.json") for value in ids]
            )
            apple.validate_poses(poses, scene_id)
            if (
                gravity.shape not in {(len(ids), 3, 3), (len(ids), 4, 4)}
                or not np.isfinite(gravity).all()
            ):
                raise ValueError(f"{scene_id}: invalid per-frame gravity array")
            for label, values in (("depth", depth_k_raw), ("RGB", rgb_k_raw)):
                if values.shape != (len(ids), 3, 3) or not np.isfinite(values).all():
                    raise ValueError(f"{scene_id}: invalid per-frame {label} intrinsics")

            orientation_rule, orientation_policy = apple.load_orientation_policy(
                args.orientation_policy, scene_id, tar_path
            )
            rotations, orientation = apple.infer_orientation(
                poses, shapes, orientation_rule
            )
            depth_k_processed = processed_intrinsics(
                depth_k_raw, shapes, rotations, image_scale=1
            )
            rgb_k_processed = processed_intrinsics(
                rgb_k_raw, shapes, rotations, image_scale=2
            )
            metadata = apple.expected_raw_metadata(
                archive, scene_id, ids, shapes, orientation["target_orientation"]
            )
            depth_k = np.asarray(metadata["K_depth.txt"], dtype=np.float64)
            rgb_k = np.asarray(metadata["K_rgb.txt"], dtype=np.float64)

            target_depth_shape: tuple[int, int] | None = None
            target_rgb_shape: tuple[int, int] | None = None
            frame_rows: list[dict[str, Any]] = []
            for index, (frame_id, rotation) in enumerate(zip(ids, rotations)):
                rgb_raw = apple.member_bytes(archive, f"{scene_id}/{frame_id}.wide/image.png")
                depth_raw = apple.member_bytes(archive, f"{scene_id}/{frame_id}.gt/depth.png")
                rgb, rgb_shape = apple.encoded_rotated_png(
                    rgb_raw, int(rotation), f"{scene_id}/{frame_id} RGB"
                )
                depth, depth_shape = apple.encoded_rotated_png(
                    depth_raw, int(rotation), f"{scene_id}/{frame_id} depth"
                )
                if len(rgb_shape) != 3 or len(depth_shape) != 2:
                    raise ValueError(f"{scene_id}/{frame_id}: invalid RGB-D channels")
                if target_depth_shape is None:
                    target_depth_shape = tuple(depth_shape[:2])
                    target_rgb_shape = tuple(rgb_shape[:2])
                if tuple(depth_shape[:2]) != target_depth_shape or tuple(rgb_shape[:2]) != target_rgb_shape:
                    raise ValueError(f"{scene_id}: inconsistent converted image dimensions")
                if target_rgb_shape != (2 * target_depth_shape[0], 2 * target_depth_shape[1]):
                    raise ValueError(f"{scene_id}: RGB dimensions are not 2x depth")
                rgb_name, depth_name = f"rgb/{index}.png", f"depth/{index}.png"
                atomic_bytes(building / rgb_name, rgb)
                atomic_bytes(building / depth_name, depth)
                frame_rows.append(
                    {
                        "processed_index": index,
                        "raw_frame_id": frame_id,
                        "rot90_ccw": int(rotation),
                        "raw_depth_shape": list(shapes[index]),
                        "output_depth_shape": list(target_depth_shape),
                        "rgb": rgb_name,
                        "depth": depth_name,
                    }
                )

            assert target_depth_shape is not None and target_rgb_shape is not None
            world_payload = apple.member_bytes(archive, f"{scene_id}/world.gt/instances.json")
            instance_rows, corners = validate_world_instances(world_payload, scene_id)
            candidates = frustum_indices(
                corners,
                depth_k_processed,
                poses,
                target_depth_shape,
                args.near,
                args.far,
            )
            surface = first_voxel_surface(
                archive,
                scene_id,
                ids,
                rotations,
                depth_k_processed,
                poses,
                args.pixel_stride,
                args.voxel_size,
                args.depth_scale,
                args.max_depth,
            )
            kept = proximity_indices(corners, candidates, surface, args.proximity_threshold)

        save_npy(building / "all_poses.npy", poses)
        save_npy(building / "T_gravity.npy", gravity)
        save_txt(building / "K_depth.txt", depth_k)
        save_txt(building / "K_rgb.txt", rgb_k)
        save_npy(building / "K_depth_per_frame.npy", depth_k_processed)
        save_npz(
            building / "per_frame_intrinsics.npz",
            schema=np.asarray(PER_FRAME_K_SCHEMA),
            raw_frame_ids=np.asarray(ids),
            rot90_ccw=rotations.astype(np.int8),
            depth_intrinsics_raw=depth_k_raw,
            rgb_intrinsics_raw=rgb_k_raw,
            depth_intrinsics_processed=depth_k_processed,
            rgb_intrinsics_processed=rgb_k_processed,
            raw_depth_shapes=np.asarray(shapes, dtype=np.int32),
            output_depth_shape=np.asarray(target_depth_shape, dtype=np.int32),
            output_rgb_shape=np.asarray(target_rgb_shape, dtype=np.int32),
        )
        atomic_json(
            building / "per_frame_intrinsics_manifest.json",
            {
                "schema": PER_FRAME_K_SCHEMA,
                "scene_id": scene_id,
                "frame_count": len(ids),
                "artifact": "per_frame_intrinsics.npz",
                "raw_values_preserved": True,
                "boxfusion_static_compatibility_policy": (
                    "K_depth.txt/K_rgb.txt retain the audited Apple converter's "
                    "target-orientation scene mean"
                ),
                "native_per_frame_policy": (
                    "K_depth_per_frame.npy contains each raw Apple K normalized "
                    "to the processed unified image orientation"
                ),
                "validation_ground_truth_access": False,
            },
        )
        atomic_json(
            building / "frame_manifest.json",
            {
                "schema": FRAME_MANIFEST_SCHEMA,
                "scene_id": scene_id,
                "frame_count": len(ids),
                "mapping": frame_rows,
            },
        )
        atomic_bytes(building / "instances.json", world_payload)
        save_npy(building / "surface_proxy.npy", surface)
        save_npy(building / "frustum_indices.npy", candidates)
        save_npy(building / "kept_indices.npy", kept)
        derived_path = building / "derived_train_gt_boxes.npy"
        save_npy(derived_path, corners[kept])
        os.link(derived_path, building / "after_filter_boxes.npy")
        atomic_json(
            building / "derived_train_gt_instances.json",
            [instance_rows[int(index)] for index in kept],
        )

        artifacts = artifact_inventory(building)
        derived_hash = artifacts["derived_train_gt_boxes.npy"]["sha256"]
        compatibility_hash = artifacts["after_filter_boxes.npy"]["sha256"]
        if derived_hash != compatibility_hash:
            raise AssertionError("derived GT compatibility artifact bytes differ")
        manifest = dict(common)
        manifest.update(
            {
                "mode": "build",
                "output_scene": str(output),
                "source_tar": {
                    "path": str(tar_path),
                    "bytes": tar_path.stat().st_size,
                    "sha256": tar_hash,
                    "member_policy": "read-only regular files under exact scene prefix",
                    "world_gt_member": f"{scene_id}/world.gt/instances.json",
                },
                "frozen_subset_manifest": {
                    "path": str(subset_manifest),
                    "sha256": sha256_file(subset_manifest),
                    "rank": int(entry["rank"]),
                    "selection_key_sha256": entry["selection_key_sha256"],
                },
                "official_validation_url_list": {
                    "path": str(val_url_list),
                    "sha256": sha256_file(val_url_list),
                    "scene_count": len(val_ids),
                },
                "orientation": orientation,
                "orientation_policy": orientation_policy,
                "orientation_rule": orientation_rule,
                "parameters": {
                    "frustum_min_visible_corners": 6,
                    "frustum_intrinsics": "processed_per_frame_depth_K",
                    "near_m": args.near,
                    "far_m": args.far,
                    "surface_proxy_pixel_stride": args.pixel_stride,
                    "surface_proxy_intrinsics": "processed_per_frame_depth_K",
                    "surface_proxy_voxel_size_m": args.voxel_size,
                    "depth_scale": args.depth_scale,
                    "max_depth_m_exclusive": args.max_depth,
                    "surface_voxel_representative": "first_in_frame_row_major_acquisition_order",
                    "proximity_threshold_m_exclusive": args.proximity_threshold,
                    "proximity_min_corners": 4,
                },
                "counts": {
                    "frames": len(ids),
                    "raw_world_boxes": len(corners),
                    "frustum_boxes": len(candidates),
                    "derived_train_gt_boxes": len(kept),
                    "surface_proxy_points": len(surface),
                },
                "frustum_indices": candidates.tolist(),
                "kept_indices": kept.tolist(),
                "derived_train_gt_artifact": "derived_train_gt_boxes.npy",
                "boxfusion_compatibility_gt_artifact": "after_filter_boxes.npy",
                "derived_train_gt_sha256": derived_hash,
                "compat_after_filter_sha256": compatibility_hash,
                "per_frame_intrinsics_artifact": "per_frame_intrinsics.npz",
                "boxfusion_per_frame_depth_intrinsics_artifact": "K_depth_per_frame.npy",
                "storage_filesystem_policy": storage_filesystem_policy(output_root),
                "artifacts": artifacts,
            }
        )
        atomic_json(building / "derived_train_gt_manifest.json", manifest)
        for path in building.rglob("*"):
            if path.is_file():
                path.chmod(0o444)
        for path in sorted((value for value in building.rglob("*") if value.is_dir()), reverse=True):
            path.chmod(0o555)
        building.chmod(0o555)
        os.replace(building, output)
        print(json.dumps(manifest, indent=2, sort_keys=True))
        return manifest
    except Exception:
        if building.exists():
            failed = output_root / f".{scene_id}.failed.{os.getpid()}"
            if not failed.exists():
                os.replace(building, failed)
        raise


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--tar", type=Path, required=True)
    result.add_argument("--scene-id")
    result.add_argument("--subset-manifest", type=Path, required=True)
    result.add_argument("--val-url-list", type=Path, required=True)
    result.add_argument("--output-root", type=Path, required=True)
    result.add_argument("--mode", choices=("preflight", "build"), default="preflight")
    result.add_argument("--orientation-policy", type=Path)
    result.add_argument("--near", type=float, default=0.10)
    result.add_argument("--far", type=float, default=100.0)
    result.add_argument("--pixel-stride", type=int, default=4)
    result.add_argument("--voxel-size", type=float, default=0.02)
    result.add_argument("--depth-scale", type=float, default=1000.0)
    result.add_argument("--max-depth", type=float, default=10.0)
    result.add_argument("--proximity-threshold", type=float, default=0.10)
    return result


def main() -> int:
    args = parser().parse_args()
    positive = (
        args.near,
        args.far,
        args.voxel_size,
        args.depth_scale,
        args.max_depth,
        args.proximity_threshold,
    )
    if any(not math.isfinite(value) or value <= 0 for value in positive):
        raise ValueError("geometry parameters must be positive and finite")
    if args.near >= args.far or args.pixel_stride <= 0:
        raise ValueError("invalid near/far or pixel stride")
    build(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
