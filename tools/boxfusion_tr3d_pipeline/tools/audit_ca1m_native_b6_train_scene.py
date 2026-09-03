#!/usr/bin/env python3
"""Audit one isolated CA-1M-native B6 train scene against its source tar."""

from __future__ import annotations

import argparse
import json
import os
import stat
import tarfile
from pathlib import Path
from typing import Any

import cv2
import numpy as np

import build_ca1m_native_b6_train_scene as native
import convert_ca1m_apple_tar as apple


AUDIT_SCHEMA = "boxfusion.ca1m_native_b6_train_scene_audit.v1"


def _regular(path: Path, label: str) -> None:
    native.require_regular(path, label)


def _load_png(path: Path, label: str) -> np.ndarray:
    _regular(path, label)
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"cannot decode {label}: {path}")
    return image


def _write_create_only(path: Path, payload: dict[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    native.atomic_json(path, payload)
    path.chmod(0o444)


def audit(args: argparse.Namespace) -> dict[str, Any]:
    scene_dir = args.scene_dir.resolve()
    if not scene_dir.is_dir() or scene_dir.is_symlink():
        raise ValueError(f"invalid train scene directory: {scene_dir}")
    manifest_path = scene_dir / "derived_train_gt_manifest.json"
    _regular(manifest_path, "derived train GT manifest")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != native.SCHEMA:
        raise ValueError("unsupported train scene manifest schema")
    scene_id = manifest.get("scene_id")
    if scene_dir.name != scene_id:
        raise ValueError("scene directory/manifest ID mismatch")
    required_safety = {
        "source_split": "train",
        "train_only": True,
        "validation_scene_overlap": False,
        "validation_ground_truth_access": False,
        "derived_train_gt": True,
        "official_validation_comparable": False,
        "paper_validation_claim_permitted": False,
    }
    for key, expected in required_safety.items():
        if manifest.get(key) != expected:
            raise ValueError(f"invalid safety field {key}: {manifest.get(key)!r}")

    tar_path = Path(manifest["source_tar"]["path"]).resolve()
    subset_manifest = Path(manifest["frozen_subset_manifest"]["path"]).resolve()
    val_url_list = Path(manifest["official_validation_url_list"]["path"]).resolve()
    contract_scene, _, entry, val_ids = native.load_train_contract(
        subset_manifest, val_url_list, tar_path, scene_id
    )
    if contract_scene != scene_id or scene_id in val_ids:
        raise ValueError("train-only split contract failed")
    if native.sha256_file(tar_path) != manifest["source_tar"]["sha256"]:
        raise ValueError("source tar SHA256 mismatch")
    if native.sha256_file(subset_manifest) != manifest["frozen_subset_manifest"]["sha256"]:
        raise ValueError("subset manifest SHA256 mismatch")
    if native.sha256_file(val_url_list) != manifest["official_validation_url_list"]["sha256"]:
        raise ValueError("validation URL list SHA256 mismatch")
    if int(entry["rank"]) != int(manifest["frozen_subset_manifest"]["rank"]):
        raise ValueError("frozen subset rank mismatch")
    if args.orientation_policy is not None:
        expected_rule, expected_policy = apple.load_orientation_policy(
            args.orientation_policy.resolve(), scene_id, tar_path
        )
        if expected_rule != manifest.get("orientation_rule"):
            raise ValueError("scene orientation rule differs from frozen policy")
        if expected_policy["override_applied"] is True and expected_policy != manifest.get(
            "orientation_policy"
        ):
            raise ValueError("scene orientation override provenance differs from frozen policy")

    expected_artifacts = manifest.get("artifacts")
    if not isinstance(expected_artifacts, dict):
        raise ValueError("artifact inventory missing")
    observed_paths = {
        str(path.relative_to(scene_dir))
        for path in scene_dir.rglob("*")
        if path.is_file() and path.name != manifest_path.name
    }
    if observed_paths != set(expected_artifacts):
        missing = sorted(set(expected_artifacts) - observed_paths)
        extra = sorted(observed_paths - set(expected_artifacts))
        raise ValueError(f"artifact inventory differs; missing={missing}, extra={extra}")
    storage_policy = native.storage_filesystem_policy(scene_dir)
    mode_enforceable = bool(storage_policy["posix_mode_enforceable"])
    for relative, metadata in expected_artifacts.items():
        path = scene_dir / relative
        _regular(path, f"artifact {relative}")
        if path.stat().st_size != int(metadata["bytes"]):
            raise ValueError(f"artifact byte count mismatch: {relative}")
        if native.sha256_file(path) != metadata["sha256"]:
            raise ValueError(f"artifact SHA256 mismatch: {relative}")
        if mode_enforceable and path.stat().st_mode & (
            stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH
        ):
            raise ValueError(f"artifact is writable: {relative}")

    derived_path = scene_dir / "derived_train_gt_boxes.npy"
    compatibility_path = scene_dir / "after_filter_boxes.npy"
    if derived_path.stat().st_ino != compatibility_path.stat().st_ino:
        raise ValueError("BoxFusion compatibility GT is not a hard link to derived GT")
    derived = np.load(derived_path, allow_pickle=False)
    compatibility = np.load(compatibility_path, allow_pickle=False)
    if (
        derived.shape != compatibility.shape
        or not np.array_equal(derived, compatibility)
        or derived.ndim != 3
        or derived.shape[1:] != (8, 3)
        or not np.isfinite(derived).all()
    ):
        raise ValueError("derived/compatibility GT arrays differ or are invalid")
    if native.sha256_file(derived_path) != manifest["derived_train_gt_sha256"]:
        raise ValueError("derived train GT manifest hash mismatch")

    poses = np.load(scene_dir / "all_poses.npy", allow_pickle=False)
    gravity = np.load(scene_dir / "T_gravity.npy", allow_pickle=False)
    depth_k = np.loadtxt(scene_dir / "K_depth.txt")
    rgb_k = np.loadtxt(scene_dir / "K_rgb.txt")
    frame_manifest = json.loads((scene_dir / "frame_manifest.json").read_text())
    k_manifest = json.loads((scene_dir / "per_frame_intrinsics_manifest.json").read_text())
    if frame_manifest.get("schema") != native.FRAME_MANIFEST_SCHEMA:
        raise ValueError("unsupported frame manifest schema")
    if k_manifest.get("schema") != native.PER_FRAME_K_SCHEMA:
        raise ValueError("unsupported per-frame K manifest schema")

    with np.load(scene_dir / "per_frame_intrinsics.npz", allow_pickle=False) as data:
        if str(data["schema"].item()) != native.PER_FRAME_K_SCHEMA:
            raise ValueError("per-frame K NPZ schema mismatch")
        raw_ids = tuple(str(value) for value in data["raw_frame_ids"].tolist())
        rotations_saved = np.asarray(data["rot90_ccw"], dtype=np.int8)
        depth_k_raw_saved = np.asarray(data["depth_intrinsics_raw"])
        rgb_k_raw_saved = np.asarray(data["rgb_intrinsics_raw"])
        depth_processed_saved = np.asarray(data["depth_intrinsics_processed"])
        rgb_processed_saved = np.asarray(data["rgb_intrinsics_processed"])
        raw_shapes_saved = np.asarray(data["raw_depth_shapes"], dtype=np.int32)
        output_depth_shape = tuple(int(value) for value in data["output_depth_shape"])
        output_rgb_shape = tuple(int(value) for value in data["output_rgb_shape"])

    rows = frame_manifest.get("mapping")
    if not isinstance(rows, list) or len(rows) != len(raw_ids):
        raise ValueError("frame mapping count mismatch")
    if [row.get("processed_index") for row in rows] != list(range(len(rows))):
        raise ValueError("processed frame indices are not contiguous")
    if tuple(str(row.get("raw_frame_id")) for row in rows) != raw_ids:
        raise ValueError("raw frame mapping differs from per-frame K NPZ")

    parameters = manifest["parameters"]
    with tarfile.open(tar_path, mode="r:") as archive:
        native.validate_tar_members(archive, scene_id)
        ids = apple.frame_ids(archive, scene_id)
        if ids != raw_ids:
            raise ValueError("raw tar frame IDs differ from converted mapping")
        shapes = tuple(apple.depth_shape(archive, scene_id, value) for value in ids)
        raw_shapes = np.asarray(shapes, dtype=np.int32)
        raw_poses = np.stack(
            [apple.parse_json_member(archive, f"{scene_id}/{value}.gt/RT.json") for value in ids]
        )
        raw_gravity = np.stack(
            [apple.parse_json_member(archive, f"{scene_id}/{value}.wide/T_gravity.json") for value in ids]
        )
        depth_k_raw = np.stack(
            [apple.parse_json_member(archive, f"{scene_id}/{value}.gt/depth/K.json") for value in ids]
        )
        rgb_k_raw = np.stack(
            [apple.parse_json_member(archive, f"{scene_id}/{value}.gt/image/K.json") for value in ids]
        )
        rule = manifest["orientation_rule"]
        rotations, orientation = apple.infer_orientation(raw_poses, shapes, rule)
        if orientation != manifest["orientation"]:
            raise ValueError("recomputed orientation differs from manifest")
        metadata = apple.expected_raw_metadata(
            archive, scene_id, ids, shapes, orientation["target_orientation"]
        )
        if not np.array_equal(raw_poses, poses) or not np.array_equal(raw_gravity, gravity):
            raise ValueError("pose/gravity arrays are not exact raw values")
        if not np.array_equal(depth_k_raw, depth_k_raw_saved) or not np.array_equal(rgb_k_raw, rgb_k_raw_saved):
            raise ValueError("raw per-frame intrinsics were not preserved exactly")
        if not np.array_equal(raw_shapes, raw_shapes_saved) or not np.array_equal(rotations, rotations_saved):
            raise ValueError("raw shapes/rotations differ from per-frame manifest")
        depth_processed = native.processed_intrinsics(
            depth_k_raw, shapes, rotations, image_scale=1
        )
        rgb_processed = native.processed_intrinsics(
            rgb_k_raw, shapes, rotations, image_scale=2
        )
        if not np.array_equal(depth_processed, depth_processed_saved) or not np.array_equal(
            rgb_processed, rgb_processed_saved
        ):
            raise ValueError("processed per-frame intrinsics differ from normalization policy")
        if not np.array_equal(
            depth_processed,
            np.load(scene_dir / "K_depth_per_frame.npy", allow_pickle=False),
        ):
            raise ValueError("BoxFusion per-frame depth K sidecar differs from preserved values")
        if not np.allclose(depth_k, metadata["K_depth.txt"], rtol=0, atol=1e-12):
            raise ValueError("BoxFusion depth K differs from converter policy")
        if not np.allclose(rgb_k, metadata["K_rgb.txt"], rtol=0, atol=1e-12):
            raise ValueError("BoxFusion RGB K differs from converter policy")

        if args.pixel_check == "all":
            pixel_indices = range(len(ids))
        elif args.pixel_check == "sample":
            pixel_indices = sorted(
                set(
                    [0, len(ids) // 2, len(ids) - 1]
                    + [index for index in range(1, len(ids)) if rotations[index] != rotations[index - 1]]
                )
            )
        else:
            pixel_indices = ()
        pixel_rows = 0
        for index in pixel_indices:
            frame_id, rotation = ids[index], int(rotations[index])
            for member_suffix, output_relative in (
                (".wide/image.png", f"rgb/{index}.png"),
                (".gt/depth.png", f"depth/{index}.png"),
            ):
                raw = apple.decode_png(
                    apple.member_bytes(archive, f"{scene_id}/{frame_id}{member_suffix}"),
                    f"raw {scene_id}/{frame_id}{member_suffix}",
                )
                observed = _load_png(scene_dir / output_relative, output_relative)
                expected = np.rot90(raw, rotation) if rotation else raw
                if not np.array_equal(expected, observed):
                    raise ValueError(f"converted pixels differ: {output_relative}")
                pixel_rows += 1

        world_payload = apple.member_bytes(archive, f"{scene_id}/world.gt/instances.json")
        rows_raw, corners = native.validate_world_instances(world_payload, scene_id)
        if (scene_dir / "instances.json").read_bytes() != world_payload:
            raise ValueError("world instances are not a byte-exact tar copy")
        candidates = native.frustum_indices(
            corners,
            depth_processed,
            poses,
            output_depth_shape,
            float(parameters["near_m"]),
            float(parameters["far_m"]),
        )
        surface_artifact = np.load(scene_dir / "surface_proxy.npy", allow_pickle=False)
        if args.geometry_check == "full":
            surface = native.first_voxel_surface(
                archive,
                scene_id,
                ids,
                rotations,
                depth_processed,
                poses,
                int(parameters["surface_proxy_pixel_stride"]),
                float(parameters["surface_proxy_voxel_size_m"]),
                float(parameters["depth_scale"]),
                float(parameters["max_depth_m_exclusive"]),
            )
            if not np.array_equal(surface, surface_artifact):
                raise ValueError("surface proxy is not exactly reproducible")
        else:
            surface = surface_artifact
        kept = native.proximity_indices(
            corners,
            candidates,
            surface,
            float(parameters["proximity_threshold_m_exclusive"]),
        )

    if not np.array_equal(candidates, np.load(scene_dir / "frustum_indices.npy", allow_pickle=False)):
        raise ValueError("frustum indices differ from recomputation")
    if not np.array_equal(kept, np.load(scene_dir / "kept_indices.npy", allow_pickle=False)):
        raise ValueError("kept indices differ from recomputation")
    if candidates.tolist() != manifest["frustum_indices"] or kept.tolist() != manifest["kept_indices"]:
        raise ValueError("manifest GT indices differ from recomputation")
    if not np.array_equal(derived, corners[kept]):
        raise ValueError("derived train GT boxes differ from selected world boxes")
    selected_rows = json.loads((scene_dir / "derived_train_gt_instances.json").read_text())
    if selected_rows != [rows_raw[int(index)] for index in kept]:
        raise ValueError("derived train GT instance metadata differs")
    if output_rgb_shape != (2 * output_depth_shape[0], 2 * output_depth_shape[1]):
        raise ValueError("output RGB-D shape relation is invalid")
    counts = manifest["counts"]
    expected_counts = {
        "frames": len(raw_ids),
        "raw_world_boxes": len(corners),
        "frustum_boxes": len(candidates),
        "derived_train_gt_boxes": len(kept),
        "surface_proxy_points": len(surface_artifact),
    }
    if counts != expected_counts:
        raise ValueError(f"manifest counts differ: {counts} vs {expected_counts}")

    report = {
        "schema": AUDIT_SCHEMA,
        "ok": True,
        "scene_id": scene_id,
        "train_only": True,
        "validation_scene_overlap": False,
        "validation_ground_truth_access": False,
        "derived_train_gt": True,
        "official_validation_comparable": False,
        "geometry_check": args.geometry_check,
        "pixel_check": args.pixel_check,
        "pixel_rows_checked": pixel_rows,
        "counts": expected_counts,
        "source_tar_sha256": manifest["source_tar"]["sha256"],
        "derived_train_gt_sha256": manifest["derived_train_gt_sha256"],
        "per_frame_intrinsics_preserved": True,
        "boxfusion_layout_complete": True,
        "storage_filesystem_policy": storage_policy,
    }
    if args.output:
        _write_create_only(args.output.resolve(), report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return report


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--scene-dir", type=Path, required=True)
    result.add_argument("--geometry-check", choices=("full", "artifact"), default="full")
    result.add_argument("--pixel-check", choices=("all", "sample", "none"), default="sample")
    result.add_argument("--orientation-policy", type=Path)
    result.add_argument("--output", type=Path)
    return result


def main() -> int:
    audit(parser().parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
