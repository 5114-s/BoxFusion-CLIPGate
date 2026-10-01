#!/usr/bin/env python3
"""Fail-closed, GT-free input audit for CA-1M native-B6 train collection."""

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


SCHEMA = "boxfusion.ca1m_native_b6_train_input_audit.v1"
MANIFEST_SCHEMA = "boxfusion.ca1m_native_b6_train_subset.v1"
VAL_RE = re.compile(r"^/datasets/ca1m/val/ca1m-val-([0-9]{8})\.tar$")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def regular(path: Path, label: str) -> None:
    try:
        mode = path.lstat().st_mode
    except FileNotFoundError as exc:
        raise ValueError(f"missing {label}: {path}") from exc
    if path.is_symlink() or not stat.S_ISREG(mode) or path.stat().st_size <= 0:
        raise ValueError(f"{label} must be a non-empty regular file: {path}")


def load_contract(manifest_path: Path, val_url_list: Path) -> tuple[dict, list[str]]:
    regular(manifest_path, "frozen train subset manifest")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != MANIFEST_SCHEMA:
        raise ValueError("unexpected train subset manifest schema")
    entries = manifest.get("entries")
    if not isinstance(entries, list) or not entries:
        raise ValueError("train subset manifest has no entries")
    scene_ids = [row.get("scene_id") for row in entries]
    if (
        any(not isinstance(scene, str) or not re.fullmatch(r"[0-9]{8}", scene) for scene in scene_ids)
        or len(scene_ids) != len(set(scene_ids))
    ):
        raise ValueError("invalid/duplicate scene IDs in train subset manifest")
    safety = manifest.get("safety_contract", {})
    if (
        safety.get("train_only") is not True
        or safety.get("validation_ground_truth_access") is not False
        or int(safety.get("validation_scene_overlap_count", -1)) != 0
        or safety.get("training_started") is not False
    ):
        raise ValueError("frozen subset safety contract is not train-only")
    expected_digest = hashlib.sha256(
        ("\n".join(scene_ids) + "\n").encode("ascii")
    ).hexdigest()
    if manifest.get("selection", {}).get("scene_ids_sha256") != expected_digest:
        raise ValueError("train subset scene-list digest disagrees")

    regular(val_url_list, "official validation URL list")
    val_ids: list[str] = []
    for line_number, raw in enumerate(val_url_list.read_text().splitlines(), 1):
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
            raise ValueError(f"invalid official validation URL at line {line_number}")
        val_ids.append(matched.group(1))
    if not val_ids or len(val_ids) != len(set(val_ids)):
        raise ValueError("official validation URL IDs are empty or duplicate")
    overlap = sorted(set(scene_ids) & set(val_ids))
    if overlap:
        raise ValueError("train collection overlaps validation IDs: " + ",".join(overlap))
    return manifest, scene_ids


def numbered_pngs(root: Path) -> list[Path]:
    if not root.is_dir() or root.is_symlink():
        raise ValueError(f"missing/non-regular image directory: {root}")
    paths = list(root.glob("*.png"))
    if any(path.is_symlink() or not path.stem.isdigit() for path in paths):
        raise ValueError(f"non-numeric/symlink PNG input: {root}")
    return sorted(paths, key=lambda path: int(path.stem))


def matrix(path: Path, shape: tuple[int, ...]) -> np.ndarray:
    regular(path, path.name)
    value = np.loadtxt(path).reshape(shape).astype(np.float64)
    if not np.isfinite(value).all():
        raise ValueError(f"non-finite matrix: {path}")
    return value


def validate_k(k: np.ndarray, shape: tuple[int, int], label: str) -> None:
    height, width = shape
    if k.shape != (3, 3) or not np.isfinite(k).all():
        raise ValueError(f"{label}: invalid 3x3 intrinsics")
    if k[0, 0] <= 0 or k[1, 1] <= 0 or abs(k[2, 2] - 1.0) > 1e-6:
        raise ValueError(f"{label}: invalid focal length/homogeneous scale")
    if not (-0.5 <= k[0, 2] <= width - 0.5 and -0.5 <= k[1, 2] <= height - 0.5):
        raise ValueError(f"{label}: principal point lies outside image")


def optional_per_frame_k(
    scene_root: Path,
    static_depth: np.ndarray,
    frames: int,
) -> dict:
    # CA1MDataset consumes depth-camera intrinsics after resizing RGB to the
    # depth image.  The train-scene builder therefore publishes exactly the
    # loader-facing K_depth_per_frame.npy sidecar.  Raw/processed per-frame RGB
    # intrinsics remain preserved in per_frame_intrinsics.npz for provenance,
    # but requiring a second loader sidecar here would reject valid data for an
    # artifact the inference path never reads.
    depth_path = scene_root / "K_depth_per_frame.npy"
    if depth_path.is_symlink():
        raise ValueError(f"{scene_root.name}: refusing symlink per-frame depth K")
    if not depth_path.is_file():
        return {
            "mode": "static_scene_intrinsics_v1",
            "per_frame_sidecars_present": False,
            "variation_detected": False,
            "loader_behavior": "static_K_depth_txt",
        }
    values = np.load(depth_path, allow_pickle=False).astype(np.float64)
    if values.shape != (frames, 3, 3) or not np.isfinite(values).all():
        raise ValueError(f"{scene_root.name}: invalid depth per-frame K shape")
    deviation = float(np.max(np.abs(values - static_depth[None])))
    if np.any(values[:, 0, 0] <= 0.0) or np.any(values[:, 1, 1] <= 0.0):
        raise ValueError(f"{scene_root.name}: depth per-frame K has invalid focal length")
    if not np.allclose(
            values[:, 2, :], np.asarray([0.0, 0.0, 1.0]),
            atol=1e-8, rtol=0.0):
        raise ValueError(f"{scene_root.name}: depth per-frame K has invalid homogeneous row")
    return {
        "mode": (
            "per_frame_intrinsics_v1"
            if deviation > 1e-6
            else "per_frame_sidecars_static_equivalent_v1"
        ),
        "per_frame_sidecars_present": True,
        "variation_detected": bool(deviation > 1e-6),
        "max_abs_from_static": {"depth": deviation},
        "loader_behavior": "K_depth_per_frame_npy",
        "rgb_intrinsics_behavior": "K_rgb.txt_static; RGB resized to depth grid",
    }


def audit_scene(data_root: Path, scene: str) -> dict:
    root = data_root / scene
    if not root.is_dir() or root.is_symlink():
        raise ValueError(f"missing/non-regular train scene: {root}")
    rgb = numbered_pngs(root / "rgb")
    depth = numbered_pngs(root / "depth")
    if not rgb or [path.stem for path in rgb] != [path.stem for path in depth]:
        raise ValueError(f"{scene}: RGB/depth frame IDs disagree")
    poses_path = root / "all_poses.npy"
    gravity_path = root / "T_gravity.npy"
    regular(poses_path, "camera poses")
    regular(gravity_path, "gravity transforms")
    poses = np.load(poses_path, allow_pickle=False).reshape(-1, 4, 4).astype(np.float64)
    gravity = np.load(gravity_path, allow_pickle=False)
    if len(poses) != len(rgb) or not np.isfinite(poses).all():
        raise ValueError(f"{scene}: pose/RGB cardinality or finiteness mismatch")
    if gravity.shape[0] != len(rgb) or not np.isfinite(gravity).all():
        raise ValueError(f"{scene}: gravity/RGB cardinality or finiteness mismatch")
    if not np.allclose(poses[:, 3], [0, 0, 0, 1], atol=1e-6):
        raise ValueError(f"{scene}: poses are not homogeneous camera-to-world matrices")
    sample_depth = cv2.imread(str(depth[0]), cv2.IMREAD_UNCHANGED)
    sample_rgb = cv2.imread(str(rgb[0]), cv2.IMREAD_COLOR)
    if sample_depth is None or sample_depth.ndim != 2 or sample_rgb is None:
        raise ValueError(f"{scene}: first RGB-D frame is unreadable")
    depth_k = matrix(root / "K_depth.txt", (3, 3))
    rgb_k = matrix(root / "K_rgb.txt", (3, 3))
    validate_k(depth_k, tuple(sample_depth.shape), f"{scene}/depth")
    validate_k(rgb_k, tuple(sample_rgb.shape[:2]), f"{scene}/rgb")
    intrinsics = optional_per_frame_k(root, depth_k, len(rgb))
    return {
        "scene_id": scene,
        "frames": len(rgb),
        "frame_id_first_last": [rgb[0].stem, rgb[-1].stem],
        "depth_shape": list(sample_depth.shape),
        "rgb_shape": list(sample_rgb.shape[:2]),
        "intrinsics_contract": intrinsics,
        "metadata_sha256": {
            "K_depth.txt": sha256(root / "K_depth.txt"),
            "K_rgb.txt": sha256(root / "K_rgb.txt"),
            "all_poses.npy": sha256(poses_path),
            "T_gravity.npy": sha256(gravity_path),
        },
    }


def create_json(path: Path, value: dict) -> None:
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
        path.chmod(0o444)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--val-url-list", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    manifest, scene_ids = load_contract(args.manifest, args.val_url_list)
    root_exists = args.data_root.is_dir() and not args.data_root.is_symlink()
    present = [scene for scene in scene_ids if (args.data_root / scene).is_dir()]
    if args.preflight:
        result = {
            "schema": SCHEMA,
            "ok": True,
            "preflight_only": True,
            "manifest_sha256": sha256(args.manifest),
            "scene_ids_sha256": manifest["selection"]["scene_ids_sha256"],
            "expected_scenes": len(scene_ids),
            "data_root": str(args.data_root.resolve(strict=False)),
            "data_root_exists": root_exists,
            "scene_directories_present": len(present),
            "scene_directories_missing": len(scene_ids) - len(present),
            "validation_url_ids_checked": manifest["source"]["val_scene_count"],
            "validation_overlap": 0,
            "validation_ground_truth_access": False,
            "training_started": False,
        }
    else:
        if not root_exists:
            raise ValueError(f"missing/non-regular train data root: {args.data_root}")
        scenes = [audit_scene(args.data_root, scene) for scene in scene_ids]
        result = {
            "schema": SCHEMA,
            "ok": True,
            "preflight_only": False,
            "manifest_sha256": sha256(args.manifest),
            "scene_ids_sha256": manifest["selection"]["scene_ids_sha256"],
            "expected_scenes": len(scene_ids),
            "audited_scenes": len(scenes),
            "data_root": str(args.data_root.resolve()),
            "validation_overlap": 0,
            "validation_ground_truth_access": False,
            "ground_truth_files_read": [],
            "prediction_access": False,
            "intrinsics_policy": "loader_consumes_optional_K_depth_per_frame_npy_v1",
            "scenes": scenes,
        }
    if args.output is not None:
        create_json(args.output, result)
    print(json.dumps(result if args.preflight else {k: result[k] for k in (
        "schema", "ok", "expected_scenes", "audited_scenes", "validation_overlap",
        "validation_ground_truth_access", "intrinsics_policy")}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
