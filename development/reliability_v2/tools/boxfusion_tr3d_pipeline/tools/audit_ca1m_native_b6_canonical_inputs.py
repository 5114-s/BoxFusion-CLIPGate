#!/usr/bin/env python3
"""GT-free input audit for the frozen CA-1M canonical-103 observer run."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import tempfile
from pathlib import Path
from urllib.parse import urlsplit

import cv2
import numpy as np


SCHEMA = "boxfusion.ca1m_native_b6_canonical103_input_audit.v1"
VAL_RE = re.compile(r"^/datasets/ca1m/val/ca1m-val-([0-9]{8})\.tar$")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def regular(path: Path, label: str) -> None:
    try:
        mode = path.lstat().st_mode
    except FileNotFoundError as error:
        raise ValueError(f"missing {label}: {path}") from error
    if path.is_symlink() or not stat.S_ISREG(mode) or path.stat().st_size <= 0:
        raise ValueError(f"{label} must be a non-empty regular file: {path}")


def scene_ids(path: Path, expected: int) -> list[str]:
    regular(path, "frozen scene list")
    rows = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    if (
        len(rows) != expected
        or len(rows) != len(set(rows))
        or any(re.fullmatch(r"[0-9]{8}", row) is None for row in rows)
    ):
        raise ValueError(f"scene list must contain exactly {expected} unique numeric IDs")
    return rows


def official_ids(path: Path) -> list[str]:
    regular(path, "official CA-1M validation URL list")
    rows: list[str] = []
    for line_number, raw in enumerate(path.read_text().splitlines(), 1):
        url = raw.strip()
        if not url:
            continue
        parsed = urlsplit(url)
        matched = VAL_RE.fullmatch(parsed.path)
        if (
            parsed.scheme != "https"
            or parsed.netloc != "ml-site.cdn-apple.com"
            or parsed.query
            or parsed.fragment
            or matched is None
        ):
            raise ValueError(f"invalid official URL at line {line_number}")
        rows.append(matched.group(1))
    if len(rows) != 107 or len(rows) != len(set(rows)):
        raise ValueError("official validation URL list must resolve to 107 unique scenes")
    return rows


def load_contract(scene_list: Path, excluded_list: Path, official_list: Path) -> list[str]:
    canonical = scene_ids(scene_list, 103)
    excluded = scene_ids(excluded_list, 4)
    official = official_ids(official_list)
    if set(canonical) & set(excluded):
        raise ValueError("canonical and excluded scene lists overlap")
    if set(canonical) | set(excluded) != set(official):
        raise ValueError("canonical103 plus excluded4 does not equal official validation107")
    return canonical


def numbered_pngs(root: Path) -> list[Path]:
    if not root.is_dir() or root.is_symlink():
        raise ValueError(f"missing/non-regular image directory: {root}")
    paths = list(root.glob("*.png"))
    if any(path.is_symlink() or not path.is_file() or not path.stem.isdigit() for path in paths):
        raise ValueError(f"non-numeric, symlink, or irregular PNG input: {root}")
    return sorted(paths, key=lambda path: int(path.stem))


def matrix(path: Path) -> np.ndarray:
    regular(path, path.name)
    value = np.loadtxt(path).reshape(3, 3).astype(np.float64)
    if not np.isfinite(value).all():
        raise ValueError(f"non-finite camera matrix: {path}")
    return value


def validate_k(value: np.ndarray, image_shape: tuple[int, int], label: str) -> None:
    height, width = image_shape
    if (
        value[0, 0] <= 0
        or value[1, 1] <= 0
        or not np.allclose(value[2], [0, 0, 1], atol=1e-8, rtol=0)
        or not (-0.5 <= value[0, 2] <= width - 0.5)
        or not (-0.5 <= value[1, 2] <= height - 0.5)
    ):
        raise ValueError(f"invalid intrinsics for {label}")


def audit_scene(data_root: Path, scene: str) -> dict:
    root = data_root / scene
    if not root.is_dir() or root.is_symlink():
        raise ValueError(f"missing/non-regular canonical scene: {root}")
    rgb = numbered_pngs(root / "rgb")
    depth = numbered_pngs(root / "depth")
    if not rgb or [path.stem for path in rgb] != [path.stem for path in depth]:
        raise ValueError(f"{scene}: RGB/depth frame IDs disagree")

    pose_path = root / "all_poses.npy"
    gravity_path = root / "T_gravity.npy"
    regular(pose_path, "camera poses")
    regular(gravity_path, "gravity rotations")
    poses = np.load(pose_path, allow_pickle=False)
    gravity = np.load(gravity_path, allow_pickle=False)
    if (
        poses.shape != (len(rgb), 4, 4)
        or gravity.shape != (len(rgb), 3, 3)
        or not np.isfinite(poses).all()
        or not np.isfinite(gravity).all()
        or not np.allclose(poses[:, 3], [0, 0, 0, 1], atol=1e-6)
    ):
        raise ValueError(f"{scene}: pose/gravity cardinality or geometry is invalid")

    sample_ids = sorted({0, len(rgb) // 2, len(rgb) - 1})
    rgb_shapes: set[tuple[int, int]] = set()
    depth_shapes: set[tuple[int, int]] = set()
    for index in sample_ids:
        rgb_value = cv2.imread(str(rgb[index]), cv2.IMREAD_COLOR)
        depth_value = cv2.imread(str(depth[index]), cv2.IMREAD_UNCHANGED)
        if rgb_value is None or depth_value is None or depth_value.ndim != 2:
            raise ValueError(f"{scene}: unreadable sampled RGB-D frame {index}")
        rgb_shapes.add(tuple(rgb_value.shape[:2]))
        depth_shapes.add(tuple(depth_value.shape))
    if len(rgb_shapes) != 1 or len(depth_shapes) != 1:
        raise ValueError(f"{scene}: sampled frame shapes are inconsistent")

    depth_k_path = root / "K_depth.txt"
    rgb_k_path = root / "K_rgb.txt"
    depth_k, rgb_k = matrix(depth_k_path), matrix(rgb_k_path)
    validate_k(depth_k, next(iter(depth_shapes)), f"{scene}/depth")
    validate_k(rgb_k, next(iter(rgb_shapes)), f"{scene}/rgb")
    return {
        "scene_id": scene,
        "frames": len(rgb),
        "frame_id_first_last": [rgb[0].stem, rgb[-1].stem],
        "sampled_frame_indices": sample_ids,
        "rgb_shape": list(next(iter(rgb_shapes))),
        "depth_shape": list(next(iter(depth_shapes))),
        "metadata_sha256": {
            "K_depth.txt": sha256(depth_k_path),
            "K_rgb.txt": sha256(rgb_k_path),
            "all_poses.npy": sha256(pose_path),
            "T_gravity.npy": sha256(gravity_path),
        },
    }


def write_exclusive(path: Path, value: dict) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-list", type=Path, required=True)
    parser.add_argument("--excluded-scene-list", type=Path, required=True)
    parser.add_argument("--official-url-list", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()

    canonical = load_contract(
        args.scene_list, args.excluded_scene_list, args.official_url_list
    )
    present = [scene for scene in canonical if (args.data_root / scene).is_dir()]
    common = {
        "schema": SCHEMA,
        "ok": True,
        "preflight_only": bool(args.preflight),
        "scene_list": str(args.scene_list.resolve()),
        "scene_list_sha256": sha256(args.scene_list),
        "excluded_scene_list_sha256": sha256(args.excluded_scene_list),
        "official_url_list_sha256": sha256(args.official_url_list),
        "expected_scenes": 103,
        "data_root": str(args.data_root.resolve(strict=False)),
        "scene_directories_present": len(present),
        "scene_directories_missing": len(canonical) - len(present),
        "ground_truth_access": False,
        "evaluation_invoked": False,
        "prediction_access": False,
        "accessed_input_modalities": [
            "rgb", "depth", "K_depth", "K_rgb", "camera_to_world", "gravity"
        ],
        "forbidden_inputs_opened": [],
    }
    if args.preflight:
        if len(present) != len(canonical):
            raise ValueError("canonical103 data root is incomplete")
        result = common
    else:
        result = {
            **common,
            "audited_scenes": 103,
            "scenes": [audit_scene(args.data_root, scene) for scene in canonical],
        }
    if args.output is not None:
        write_exclusive(args.output, result)
    summary_keys = (
        "schema", "ok", "preflight_only", "expected_scenes",
        "scene_directories_present", "ground_truth_access", "evaluation_invoked",
    )
    print(json.dumps({key: result[key] for key in summary_keys}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
