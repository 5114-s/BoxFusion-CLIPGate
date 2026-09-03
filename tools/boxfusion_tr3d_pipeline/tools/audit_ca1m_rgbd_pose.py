#!/usr/bin/env python3
"""Fail-closed CA-1M RGB-D/pose/gravity audit without reading predictions."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import tempfile

import cv2
import numpy as np


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def scenes(path: Path) -> tuple[str, ...]:
    rows = tuple(
        line.split()[0] for line in path.read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    )
    if not rows or len(rows) != len(set(rows)):
        raise ValueError("scene list is empty or contains duplicates")
    return rows


def audit(args: argparse.Namespace) -> dict:
    rows = []
    for scene in scenes(args.scene_list.resolve()):
        root = args.data_root.resolve() / scene
        files = {
            "depth_intrinsics": root / "K_depth.txt",
            "rgb_intrinsics": root / "K_rgb.txt",
            "poses": root / "all_poses.npy",
            "ground_truth": root / "after_filter_boxes.npy",
        }
        for path in files.values():
            if not path.is_file() or path.is_symlink():
                raise ValueError(f"{scene}: missing/non-regular {path}")
        rgb = sorted((root / "rgb").glob("*.png"), key=lambda path: int(path.stem))
        depth = sorted((root / "depth").glob("*.png"), key=lambda path: int(path.stem))
        poses = np.load(files["poses"]).reshape(-1, 4, 4).astype(np.float64)
        gt = np.load(files["ground_truth"]).reshape(-1, 8, 3).astype(np.float64)
        K = np.loadtxt(files["depth_intrinsics"]).reshape(3, 3).astype(np.float64)
        if len(rgb) != len(depth) or len(rgb) != len(poses) or len(rgb) < args.min_frames:
            raise ValueError(f"{scene}: RGB/depth/pose count mismatch")
        if not np.isfinite(poses).all() or not np.isfinite(gt).all() or not np.isfinite(K).all():
            raise ValueError(f"{scene}: non-finite calibration/geometry")
        if not np.allclose(poses[:, 3], [0, 0, 0, 1], atol=1e-6):
            raise ValueError(f"{scene}: poses are not homogeneous")
        rotations = poses[:, :3, :3]
        orthogonal_error = float(np.max(np.abs(rotations.transpose(0, 2, 1) @ rotations - np.eye(3))))
        determinant = np.linalg.det(rotations)
        if orthogonal_error > 1e-3 or np.max(np.abs(determinant - 1.0)) > 1e-3:
            raise ValueError(f"{scene}: pose rotation is not proper rigid")
        middle = cv2.imread(str(depth[len(depth) // 2]), cv2.IMREAD_UNCHANGED)
        if middle is None or middle.ndim != 2:
            raise ValueError(f"{scene}: unreadable depth")
        positive = middle[middle > 0]
        if not len(positive):
            raise ValueError(f"{scene}: empty metric depth")
        camera_min, camera_max = poses[:, :3, 3].min(0), poses[:, :3, 3].max(0)
        gt_min, gt_max = gt.reshape(-1, 3).min(0), gt.reshape(-1, 3).max(0)
        # Camera and GT must occupy compatible world-coordinate slabs.  This
        # rejects accidental W2C poses and missing large global translations.
        gap = np.maximum(np.maximum(camera_min - gt_max, gt_min - camera_max), 0.0)
        if float(np.linalg.norm(gap)) > 2.5:
            raise ValueError(f"{scene}: pose/GT world coordinates disagree")
        rows.append({
            "scene_id": scene, "frames": len(poses), "gt_boxes": len(gt),
            "depth_shape": list(middle.shape),
            "depth_median_m": float(np.median(positive) / args.depth_scale),
            "pose_rotation_orthogonal_error": orthogonal_error,
            "pose_rotation_determinant_minmax": [float(determinant.min()), float(determinant.max())],
            "camera_world_min": camera_min.tolist(), "camera_world_max": camera_max.tolist(),
            "gt_world_min": gt_min.tolist(), "gt_world_max": gt_max.tolist(),
            "depth_intrinsics": K.tolist(),
            "input_sha256": {name: sha256(path) for name, path in files.items()},
        })
    return {
        "schema": "boxfusion.ca1m_rgbd_pose_audit.v1", "complete": True,
        "ok": True, "prediction_access": False, "ground_truth_used_for_coordinate_audit_only": True,
        "pose_convention": "camera_to_world", "world_frame": "ca1m_gravity_aligned",
        "axis_alignment": "identity", "depth_scale": args.depth_scale,
        "scene_list": str(args.scene_list.resolve()),
        "scene_list_sha256": sha256(args.scene_list.resolve()),
        "scene_count": len(rows), "scenes": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--scene-list", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--depth-scale", type=float, default=1000.0)
    parser.add_argument("--min-frames", type=int, default=1)
    args = parser.parse_args()
    result = audit(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError(args.output)
    with tempfile.NamedTemporaryFile("w", dir=args.output.parent, delete=False) as handle:
        handle.write(json.dumps(result, indent=2, sort_keys=True) + "\n")
        temporary = Path(handle.name)
    temporary.replace(args.output)
    args.output.chmod(0o444)
    print(json.dumps({key: result[key] for key in ("ok", "scene_count", "pose_convention", "world_frame", "axis_alignment")}, indent=2))


if __name__ == "__main__":
    main()
