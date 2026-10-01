#!/usr/bin/env python3
"""Seal exact100 lineage and training-point parity for terminal TR3D v4.

This one-time, read-only protocol audit may inspect the old sealed native-B6
``used_frame_ids`` solely as a lineage oracle.  It also rebuilds proposal
points from allowed processed RGB-D inputs and compares the resulting local
float32 array byte-for-byte with the point cloud used to train CA-native TR3D.
No annotation, prediction, evaluator, model, or GPU is opened.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
import sys

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from boxfusion.ca1m_tr3d_checkpoint_binding import regular_directory, regular_file  # noqa: E402
from boxfusion.ca1m_tr3d_terminal import terminal_world_to_local  # noqa: E402
from boxfusion.ca1m_tr3d_terminal_v4 import (  # noqa: E402
    derive_demo_gap20_early_finalize_frame_ids,
    sha256_array,
    sha256_file,
)
from tools.run_ca1m_tr3d_proposal_cache_v4 import _points, _scene_inputs  # noqa: E402


SCHEMA = "boxfusion.ca1m_tr3d_v4_lineage_training_point_parity.v1"
SCENE_LIST_SHA = "35321e9942dc5d512db2952b9ca6228b1291127e0c13fd92aa458f2d7eb7f9fd"


def _create_only(path: Path, value: dict[str, Any]) -> Path:
    target = path.resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=f".{target.name}.", suffix=".tmp", dir=target.parent, delete=False
        ) as handle:
            temporary = Path(handle.name)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, target)
        except FileExistsError as error:
            raise FileExistsError(f"refusing to overwrite parity receipt: {target}") from error
        target.chmod(0o444)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return target


def _scenes(path: Path) -> tuple[str, ...]:
    source = regular_file(path, "exact100 scene list")
    if sha256_file(source) != SCENE_LIST_SHA:
        raise ValueError("train100 scene list SHA256 differs")
    values = tuple(row.strip() for row in source.read_text().splitlines() if row.strip())
    if len(values) != 100 or len(set(values)) != 100:
        raise ValueError("parity audit requires exact100 scenes")
    return values


def _oracle(path: Path, scene: str) -> tuple[np.ndarray, str]:
    source = regular_file(path, "sealed lineage-oracle diagnostic")
    with np.load(source, allow_pickle=False) as archive:
        if set(("scene_id", "used_frame_ids")) - set(archive.files):
            raise ValueError(f"{scene}: lineage oracle lacks required arrays")
        stored = np.asarray(archive["scene_id"])
        frames = np.array(archive["used_frame_ids"], copy=True)
    if stored.shape != () or str(stored.item()) != scene:
        raise ValueError(f"{scene}: lineage oracle scene differs")
    if frames.dtype != np.dtype(np.int64) or frames.ndim != 1:
        raise ValueError(f"{scene}: lineage oracle array differs")
    return frames, sha256_file(source)


def audit(args: argparse.Namespace) -> dict[str, Any]:
    scene_list = regular_file(args.scene_list, "train100 scene list")
    scenes = _scenes(scene_list)
    data_root = regular_directory(args.data_root, "processed train100 RGB-D root")
    oracle_root = regular_directory(args.oracle_root, "sealed native-B6 lineage oracle root")
    point_root = regular_directory(args.training_point_root, "CA-native TR3D training point root")
    implementation_paths = {
        "audit": Path(__file__),
        "proposal_runner": ROOT / "tools/run_ca1m_tr3d_proposal_cache_v4.py",
        "v4_contract": ROOT / "boxfusion/ca1m_tr3d_terminal_v4.py",
        "backprojection": ROOT / "boxfusion/tr3d_incremental_online.py",
        "terminal_geometry": ROOT / "boxfusion/ca1m_tr3d_terminal.py",
        "training_converter": Path(
            "/data/ZhaoX/OVM3D-Dett/boxfusion_tr3d_dev/tools/prepare_tr3d_ca1m.py"
        ),
    }
    implementation = {
        name: {"path": str(regular_file(path, f"implementation {name}")), "sha256": sha256_file(path)}
        for name, path in sorted(implementation_paths.items())
    }
    rows: dict[str, Any] = {}
    total_frames = 0
    total_used = 0
    total_points = 0
    for index, scene in enumerate(scenes, 1):
        scene_root = regular_directory(data_root / scene, f"processed scene {scene}")
        rgb, depth, poses, intrinsics, derived = _scene_inputs(scene_root)
        oracle, oracle_sha = _oracle(
            oracle_root / f"{scene}_ca1m_native_b6.npz", scene
        )
        if not np.array_equal(derived, oracle):
            raise ValueError(f"{scene}: demo-loop lineage differs from sealed oracle")
        expected = derive_demo_gap20_early_finalize_frame_ids(len(rgb))
        if not np.array_equal(expected, derived):
            raise ValueError(f"{scene}: independent lineage derivations differ")
        world = _points(
            rgb=rgb,
            depth=depth,
            poses=poses,
            intrinsics=intrinsics,
            frames=derived,
            pixel_stride=4,
            min_depth=0.10,
            max_depth=6.0,
            voxel_size=0.01,
            depth_scale=1000.0,
        )
        world_to_local = terminal_world_to_local(poses[int(derived[0])])
        local = np.array(world, dtype=np.float32, order="C", copy=True)
        local[:, :3] += np.asarray(world_to_local[:3, 3], dtype=np.float32)
        training_path = regular_file(point_root / f"{scene}.bin", "CA TR3D training point")
        if training_path.stat().st_size % (6 * np.dtype(np.float32).itemsize):
            raise ValueError(f"{scene}: training point bytes are not float32 [N,6]")
        training = np.fromfile(training_path, dtype=np.float32).reshape(-1, 6)
        if not np.array_equal(local, training):
            difference = (
                float(np.max(np.abs(local.astype(np.float64) - training.astype(np.float64))))
                if local.shape == training.shape and local.size
                else None
            )
            raise ValueError(
                f"{scene}: v4 point array differs from CA training points "
                f"(v4={local.shape}, train={training.shape}, max_abs={difference})"
            )
        local_sha = hashlib.sha256(local.tobytes(order="C")).hexdigest()
        training_sha = sha256_file(training_path)
        if local_sha != training_sha:
            raise ValueError(f"{scene}: equal point arrays have different bytes")
        rows[scene] = {
            "frame_count": len(rgb),
            "used_frame_count": len(derived),
            "used_frame_ids_sha256": sha256_array(derived),
            "last_reachable_keyframe": int(derived[-1]),
            "lineage_oracle_diagnostic_sha256": oracle_sha,
            "world_point_count": len(world),
            "world_point_array_sha256": sha256_array(world),
            "local_point_array_sha256": local_sha,
            "training_point_file_sha256": training_sha,
            "lineage_equal": True,
            "array_equal": True,
            "byte_equal": True,
        }
        total_frames += len(rgb)
        total_used += len(derived)
        total_points += len(world)
        print(
            f"[{index:03d}/100] {scene}: lineage={len(derived)}, points={len(world)}, parity=PASS",
            flush=True,
        )
    return {
        "schema": SCHEMA,
        "complete": True,
        "create_only": True,
        "ground_truth_access": False,
        "validation_ground_truth_access": False,
        "prediction_access": False,
        "evaluator_access": False,
        "gpu_started": False,
        "model_started": False,
        "scene_count": len(scenes),
        "lineage_parity_scene_count": sum(row["lineage_equal"] for row in rows.values()),
        "point_array_parity_scene_count": sum(row["array_equal"] for row in rows.values()),
        "point_byte_parity_scene_count": sum(row["byte_equal"] for row in rows.values()),
        "counts": {
            "processed_frames": total_frames,
            "reachable_demo_keyframes": total_used,
            "training_distribution_points": total_points,
        },
        "lineage_contract": {
            "schema": "boxfusion.ca1m_demo_gap20_early_finalize_lineage.v1",
            "simulation": "increment_then_finalize_if_count_eq_N_minus_1_or_count_plus_20_gt_N_minus_1",
            "include_last": False,
            "sealed_native_b6_diagnostic_role": "one_time_protocol_oracle_only",
            "proposal_runtime_dependency": False,
        },
        "point_contract": {
            "source": "raw_processed_rgb_depth_pose_intrinsics",
            "pixel_stride": 4,
            "depth_range_m": [0.1, 6.0],
            "world_voxel_size_m": 0.01,
            "local_transform": "negative_first_reachable_camera_translation",
            "resize_or_orientation_transform": "none_in_training_converter_or_v4_builder",
            "proof": "recomputed local float32 arrays equal CA-native training .bin arrays exactly",
        },
        "allowed_files_opened": [
            "scene_ids.txt", "rgb/<reachable>.png", "depth/<reachable>.png",
            "all_poses.npy", "K_depth_per_frame.npy_or_K_depth.txt",
            "old_native_b6_diagnostic_used_frame_ids_only", "CA_TR3D_training_points.bin",
            "implementation_sources",
        ],
        "forbidden_files_opened": {
            "derived_train_gt_boxes.npy": False,
            "derived_train_gt_manifest.json": False,
            "instances.json": False,
            "after_filter_boxes.npy": False,
            "prediction_pkl": False,
            "validation_annotation": False,
        },
        "scene_list": {"path": str(scene_list), "sha256": sha256_file(scene_list)},
        "data_root": str(data_root),
        "lineage_oracle_root": str(oracle_root),
        "training_point_root": str(point_root),
        "implementation": implementation,
        "scenes": rows,
    }


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument(
        "--scene-list",
        type=Path,
        default=ROOT / "manifests/ca1m_native_b6_train100_v1/scene_ids.txt",
    )
    value.add_argument(
        "--data-root",
        type=Path,
        default=Path("/extra/ZhaoX/boxfusion_ca1m_native_b6_train100_v1"),
    )
    value.add_argument(
        "--oracle-root",
        type=Path,
        default=ROOT / "diagnostics/ca1m_native_b6_train100_v1/native_b6",
    )
    value.add_argument(
        "--training-point-root",
        type=Path,
        default=Path("/extra/ZhaoX/tr3d_ca1m_train100_v1/points/full"),
    )
    value.add_argument("--output", type=Path, required=True)
    return value


def main() -> int:
    args = parser().parse_args()
    report = audit(args)
    target = _create_only(args.output, report)
    print(json.dumps({"complete": True, "output": str(target), "sha256": sha256_file(target)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
