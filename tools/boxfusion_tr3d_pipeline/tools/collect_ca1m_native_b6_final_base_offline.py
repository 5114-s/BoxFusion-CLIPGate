#!/usr/bin/env python3
"""Query sealed CA-1M final-base boxes with train RGB-D evidence offline.

This tool deliberately does not rerun CuTR, Selective Boxer, CLIP association,
or BoxFusion.  The sealed final-base prediction is the sole geometry/score
authority.  We reconstruct exactly the CA1MDataset depth/K/pose keyframes and
invoke the observer directly, producing a create-only diagnostic and receipt.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import tempfile
from typing import Any, Mapping

import cv2
import numpy as np
import torch
import yaml

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

from boxfusion.ca1m_native_b6_observer import (
    CA1MNativeB6Config,
    CA1MNativeB6Observer,
    FEATURE_NAMES,
)
from boxfusion.orientation import (
    ImageOrientation,
    ROT_K,
    get_orientation,
    rotate_K,
)
from finalize_ca1m_native_b6_final_base_v2 import (
    FINAL_BASE_SCHEMA,
    load_final_base_manifest,
    prediction,
    regular,
    sha256,
)


CONFIG_SCHEMA = "boxfusion.ca1m_native_b6_final_base_offline_config.v2"
RECEIPT_SCHEMA = "boxfusion.ca1m_native_b6_final_base_offline_receipt.v2"
SCENE = re.compile(r"^[0-9]{8}$")


def array_sha(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def create_json(path: Path, value: Mapping[str, Any]) -> None:
    target = Path(path)
    if target.exists() or target.is_symlink():
        raise FileExistsError(f"refusing existing offline receipt: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
    descriptor, name = tempfile.mkstemp(
        dir=target.parent, prefix=f".{target.name}.", suffix=".tmp"
    )
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, target)
        target.chmod(0o444)
    except FileExistsError as error:
        raise FileExistsError(f"refusing existing offline receipt: {target}") from error
    finally:
        temporary.unlink(missing_ok=True)


def compare_diagnostics(existing: Path, recomputed: Path) -> None:
    """Require exact semantic identity for an interrupted orphan diagnostic."""

    left_path = regular(existing, "orphan offline diagnostic")
    right_path = regular(recomputed, "recomputed offline diagnostic")
    with np.load(left_path, allow_pickle=False) as left, np.load(
        right_path, allow_pickle=False
    ) as right:
        if set(left.files) != set(right.files):
            raise ValueError("orphan diagnostic schema differs from recomputation")
        for name in left.files:
            left_value = np.asarray(left[name])
            right_value = np.asarray(right[name])
            if name == "summary_json":
                left_summary = json.loads(str(left_value.item()))
                right_summary = json.loads(str(right_value.item()))
                # Wall-clock time is the sole intentionally non-deterministic
                # field.  Every result count and contract field remains exact.
                left_summary.pop("observer_seconds", None)
                right_summary.pop("observer_seconds", None)
                if left_summary != right_summary:
                    raise ValueError("orphan diagnostic summary differs from recomputation")
            elif not np.array_equal(left_value, right_value):
                raise ValueError(
                    f"orphan diagnostic field {name} differs from recomputation"
                )


def run_observer(
    *,
    scene: str,
    diagnostic_root: Path,
    decoded: list[tuple[int, np.ndarray, np.ndarray, np.ndarray, str]],
    corners: np.ndarray,
    scores: np.ndarray,
    observer_cfg: Mapping[str, Any],
) -> Any:
    observer = CA1MNativeB6Observer(
        CA1MNativeB6Config(
            enabled=True,
            diagnostics_root=str(diagnostic_root),
            top_k=int(observer_cfg["top_k_views"]),
            pixel_stride=int(observer_cfg["pixel_stride"]),
            margin=float(observer_cfg["depth_margin_m"]),
            min_depth=float(observer_cfg["min_depth_m"]),
            max_depth=float(observer_cfg["max_depth_m"]),
            near_clip=float(observer_cfg["near_clip_m"]),
            max_cached_keyframes=int(observer_cfg["max_cached_keyframes"]),
        )
    )
    for frame_id, depth, intrinsic, pose, _ in decoded:
        observer.record_keyframe(
            scene_id=scene,
            frame_id=frame_id,
            source_frame_id=str(frame_id),
            depth_meters=depth,
            intrinsics=intrinsic,
            camera_to_world=pose,
        )
    return observer.finalize(
        scene_id=scene,
        corners=corners.copy(),
        scores=scores.copy(),
        stable_ids=np.arange(len(scores), dtype=np.int64),
    )


def load_config(path: Path) -> tuple[dict[str, Any], Path]:
    source = regular(path, "offline observer config")
    value = yaml.safe_load(source.read_text())
    if not isinstance(value, dict) or value.get("schema") != CONFIG_SCHEMA:
        raise ValueError("unsupported offline observer config")
    if value.get("dataset") != "CA1M":
        raise ValueError("offline observer config must use CA1M")
    data = value.get("data") or {}
    if data != {
        "gap": 20,
        "start": 0,
        "depth_scale": 1000.0,
        "image_height": 384,
        "image_width": 512,
    }:
        raise ValueError("offline CA frame protocol disagrees")
    source_anchor = value.get("source_anchor") or {}
    if source_anchor != {
        "split": "train100",
        "geometry_authority": "sealed_final_base_prediction",
        "required_modules": {
            "selective_boxer_g0": True,
            "clip_appearance_gate": True,
            "reliable_view_top_k": 3,
        },
        "cross_run_replay_required": False,
        "cross_run_exact_identity_required": False,
    }:
        raise ValueError("offline source-anchor contract disagrees")
    observer = value.get("observer") or {}
    if observer != {
        "top_k_views": 5,
        "pixel_stride": 4,
        "depth_margin_m": 0.05,
        "min_depth_m": 0.10,
        "max_depth_m": 8.0,
        "near_clip_m": 0.001,
        "max_cached_keyframes": 256,
        "stable_id_policy": "sealed_prediction_row_index",
    }:
        raise ValueError("offline native-B6 observer protocol disagrees")
    safety = value.get("safety") or {}
    if safety != {
        "train_only": True,
        "prediction_mutation_authorized": False,
        "ground_truth_access": False,
        "validation_ground_truth_access": False,
        "validation_prediction_access": False,
        "evaluator_invoked": False,
        "rgb_pixels_accessed": False,
        "old_native_b6_diagnostics_reused": False,
        "old_native_b6_checkpoint_reused": False,
    }:
        raise ValueError("offline safety contract disagrees")
    return value, source


def _numeric_files(root: Path, suffix: str, label: str) -> dict[int, Path]:
    raw = Path(root)
    if raw.is_symlink():
        raise ValueError(f"{label} root must not be a symlink: {raw}")
    folder = raw.resolve()
    if not folder.is_dir() or folder.is_symlink():
        raise ValueError(f"{label} root is missing/unsafe: {folder}")
    result: dict[int, Path] = {}
    for item in folder.iterdir():
        if not item.is_file() or item.suffix.lower() != suffix:
            continue
        if not item.stem.isdigit():
            raise ValueError(f"{label} has a non-numeric frame: {item}")
        frame_id = int(item.stem)
        if frame_id in result:
            raise ValueError(f"{label} has duplicate frame id {frame_id}")
        result[frame_id] = regular(item, f"{label} frame")
    if not result:
        raise ValueError(f"{label} frame inventory is empty")
    return result


def selected_frame_ids(frame_count: int, gap: int) -> tuple[int, ...]:
    """Reproduce demo.py's record/finalize order, including its early stop.

    ``demo.py`` records before incrementing ``count`` and then finalizes after
    the increment once the next gap would exceed the scene tail.  Consequently
    it normally does *not* consume the physical terminal frame.  This explicit
    simulation is less error-prone than a closed-form endpoint and is frozen by
    parity against the sealed v1 train100 observer diagnostics.
    """

    if frame_count < 1 or gap < 1:
        raise ValueError("frame_count/gap must be positive")
    selected: list[int] = []
    count = 0
    while count < frame_count:
        if count % gap == 0 or count == frame_count - 1:
            selected.append(count)
        count += 1
        if count == frame_count - 1 or count + gap > frame_count - 1:
            break
    return tuple(selected)


def _scene_inputs(scene_root: Path, cfg: Mapping[str, Any]) -> dict[str, Any]:
    raw = Path(scene_root)
    if raw.is_symlink():
        raise ValueError(f"CA train scene root must not be a symlink: {raw}")
    root = raw.resolve()
    if not root.is_dir() or root.is_symlink():
        raise ValueError(f"CA train scene root is missing/unsafe: {root}")
    rgb = _numeric_files(root / "rgb", ".png", "CA train RGB")
    depth = _numeric_files(root / "depth", ".png", "CA train depth")
    poses_path = regular(root / "all_poses.npy", "CA train poses")
    scene_k_path = regular(root / "K_depth.txt", "CA train scene intrinsics")
    frame_k_path = regular(
        root / "K_depth_per_frame.npy", "CA train per-frame intrinsics"
    )
    poses = np.asarray(np.load(poses_path, allow_pickle=False), dtype=np.float64).reshape(-1, 4, 4)
    frame_k = np.asarray(np.load(frame_k_path, allow_pickle=False), dtype=np.float64)
    scene_k = np.asarray(np.loadtxt(scene_k_path), dtype=np.float64).reshape(3, 3)
    count = len(poses)
    expected = set(range(count))
    if set(rgb) != expected or set(depth) != expected:
        raise ValueError("CA train RGB/depth frame IDs do not match contiguous poses")
    if frame_k.shape != (count, 3, 3):
        raise ValueError("CA train per-frame intrinsics do not match frame count")
    if (
        not np.isfinite(poses).all()
        or not np.isfinite(scene_k).all()
        or not np.isfinite(frame_k).all()
    ):
        raise ValueError("CA train pose/intrinsic arrays contain non-finite values")
    if not np.allclose(frame_k[:, 2], [0.0, 0.0, 1.0], rtol=0, atol=1e-8):
        raise ValueError("CA train per-frame intrinsics have invalid homogeneous rows")
    gap = int((cfg.get("data") or {})["gap"])
    selected = selected_frame_ids(count, gap)
    return {
        "root": root,
        "rgb": rgb,
        "depth": depth,
        "poses_path": poses_path,
        "poses": poses,
        "scene_k_path": scene_k_path,
        "scene_k": scene_k,
        "frame_k_path": frame_k_path,
        "frame_k": frame_k,
        "frame_count": count,
        "selected": selected,
    }


def load_observer_frame(
    inputs: Mapping[str, Any], frame_id: int, cfg: Mapping[str, Any]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    data = cfg["data"]
    depth_path = inputs["depth"][frame_id]
    raw = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
    if raw is None or raw.ndim != 2 or not np.issubdtype(raw.dtype, np.integer):
        raise ValueError(f"failed to decode integer CA depth image: {depth_path}")
    depth = raw.astype(np.float32) / float(data["depth_scale"])
    raw_height, raw_width = depth.shape
    source_intrinsic64 = np.asarray(
        inputs["frame_k"][frame_id], dtype=np.float64
    )
    fx, fy = float(source_intrinsic64[0, 0]), float(source_intrinsic64[1, 1])
    cx, cy = float(source_intrinsic64[0, 2]), float(source_intrinsic64[1, 2])
    if not (-0.5 <= cx <= raw_width - 0.5 and -0.5 <= cy <= raw_height - 0.5):
        raise ValueError(f"frame {frame_id}: principal point lies outside raw depth")
    scene_k = inputs["scene_k"]
    if float(scene_k[0, 2]) < float(scene_k[1, 2]):
        image_height = int(data["image_width"])
        image_width = int(data["image_height"])
    else:
        image_height = int(data["image_height"])
        image_width = int(data["image_width"])
    # CA1MDataset uses cv2.resize's default INTER_LINEAR unconditionally.
    resized = cv2.resize(depth, (image_width, image_height))
    pose32 = np.asarray(inputs["poses"][frame_id], dtype=np.float32).reshape(4, 4)
    pose_tensor = torch.from_numpy(pose32.copy())[None]
    orientation = ImageOrientation(int(get_orientation(pose_tensor)[-1].item()))
    target = ImageOrientation.UPRIGHT
    # CA1MDataset rebuilds the matrix from fx/fy/cx/cy rather than retaining
    # any source off-diagonal terms.  Keep that exact construction here.
    intrinsic_tensor = torch.tensor(
        [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]]
    )[None]
    oriented_k = rotate_K(
        intrinsic_tensor, orientation, (image_width, image_height), target=target
    )[0]
    oriented_depth = torch.rot90(
        torch.from_numpy(np.ascontiguousarray(resized))[None],
        ROT_K[(orientation, target)],
        dims=(-2, -1),
    )[0]
    result_depth = np.array(oriented_depth.numpy(), dtype=np.float32, order="C", copy=True)
    result_k = np.array(oriented_k.numpy(), dtype=np.float64, order="C", copy=True)
    result_pose = np.array(pose32, dtype=np.float64, order="C", copy=True)
    if not np.isfinite(result_depth).all():
        raise ValueError(f"frame {frame_id}: oriented depth contains non-finite values")
    return result_depth, result_k, result_pose, orientation.name.lower()


def collect(args: argparse.Namespace) -> dict[str, Any]:
    scene = str(args.scene)
    if SCENE.fullmatch(scene) is None:
        raise ValueError(f"invalid CA-1M scene id: {scene!r}")
    cfg, config_path = load_config(args.config)
    final_manifest, final_manifest_path = load_final_base_manifest(args.final_base_manifest)
    if scene not in final_manifest["per_scene"]:
        raise ValueError(f"{scene}: absent from sealed final-base manifest")
    final_root = Path(args.final_base_root)
    if final_root.is_symlink() or not final_root.resolve().is_dir():
        raise ValueError("sealed final-base root is missing/unsafe")
    final_root = final_root.resolve()
    final_anchor = regular(final_root / f"{scene}_boxes.pkl", "sealed final-base anchor")
    anchor_sha = sha256(final_anchor)
    if final_manifest["per_scene"][scene].get("active_prediction_sha256") != anchor_sha:
        raise ValueError(f"{scene}: sealed final-base anchor differs from manifest")
    corners, scores, rows = prediction(final_anchor)
    inputs = _scene_inputs(Path(args.data_root) / scene, cfg)
    if len(inputs["selected"]) > int(cfg["observer"]["max_cached_keyframes"]):
        raise ValueError(f"{scene}: selected keyframes exceed observer cache limit")

    input_records: list[dict[str, Any]] = []
    decoded: list[tuple[int, np.ndarray, np.ndarray, np.ndarray, str]] = []
    for frame_id in inputs["selected"]:
        depth, intrinsic, pose, orientation = load_observer_frame(inputs, frame_id, cfg)
        depth_path = inputs["depth"][frame_id]
        input_records.append(
            {
                "frame_id": frame_id,
                "source_frame_id": str(frame_id),
                "depth_path": str(depth_path),
                "depth_sha256": sha256(depth_path),
                "oriented_depth_array_sha256": array_sha(depth),
                "oriented_intrinsics_array_sha256": array_sha(intrinsic),
                "camera_to_world_array_sha256": array_sha(pose),
                "source_orientation": orientation,
            }
        )
        decoded.append((frame_id, depth, intrinsic, pose, orientation))

    base_report = {
        "schema": RECEIPT_SCHEMA,
        "complete": args.mode == "run",
        "mode": args.mode,
        "scene_id": scene,
        "train_only": True,
        "prediction_mutation_authorized": False,
        "ground_truth_access": False,
        "validation_ground_truth_access": False,
        "validation_prediction_access": False,
        "evaluator_invoked": False,
        "rgb_pixels_accessed": False,
        "cross_run_boxfusion_replay_invoked": False,
        "cross_run_exact_identity_required": False,
        "old_native_b6_diagnostics_reused": False,
        "old_native_b6_checkpoint_reused": False,
        "geometry_authority": "sealed_final_base_prediction",
        "source_modules": {
            "selective_boxer_g0": True,
            "clip_appearance_gate": True,
            "reliable_view_top_k": 3,
            "b6_evidence_top_k": 5,
        },
        "stable_id_policy": "sealed_prediction_row_index",
        "prediction_rows": rows,
        "source_final_base": {
            "path": str(final_anchor),
            "sha256": anchor_sha,
            "manifest_path": str(final_manifest_path),
            "manifest_sha256": sha256(final_manifest_path),
            "manifest_schema": FINAL_BASE_SCHEMA,
        },
        "offline_config": {"path": str(config_path), "sha256": sha256(config_path)},
        "frame_protocol": {
            "frame_count": inputs["frame_count"],
            "gap": cfg["data"]["gap"],
            "lineage": "demo.py record-before-increment then early-finalize v1",
            "physical_terminal_frame_policy": "not_forced",
            "used_frame_ids": list(inputs["selected"]),
            "depth_resize": "cv2.INTER_LINEAR",
            "orientation": "CA1MDataset pose-derived rotate_K/torch.rot90 to upright",
            "pose_dtype": "source float64 -> online-equivalent float32 -> observer float64",
            "intrinsics_dtype": "source float64 -> online-equivalent torch.float32 -> observer float64",
        },
        "input_files": {
            "all_poses": {
                "path": str(inputs["poses_path"]),
                "sha256": sha256(inputs["poses_path"]),
            },
            "scene_intrinsics": {
                "path": str(inputs["scene_k_path"]),
                "sha256": sha256(inputs["scene_k_path"]),
            },
            "per_frame_intrinsics": {
                "path": str(inputs["frame_k_path"]),
                "sha256": sha256(inputs["frame_k_path"]),
            },
            "rgb_inventory_count": len(inputs["rgb"]),
            "rgb_pixels_accessed": False,
            "selected_depth": input_records,
        },
    }
    if args.mode == "preflight":
        return base_report

    diagnostic_root = Path(args.diagnostics_root)
    diagnostic_path = diagnostic_root / f"{scene}_ca1m_native_b6.npz"
    receipt_path = Path(args.receipt)
    if receipt_path.exists() or receipt_path.is_symlink():
        raise FileExistsError(f"refusing existing offline receipt: {receipt_path}")
    if diagnostic_path.is_symlink():
        raise ValueError(f"refusing symlink offline diagnostic: {diagnostic_path}")
    observer_cfg = cfg["observer"]
    recovered_orphan = diagnostic_path.exists()
    if recovered_orphan:
        regular(diagnostic_path, "orphan offline diagnostic")
        with tempfile.TemporaryDirectory(prefix=f"b6-offline-{scene}-") as name:
            temporary_root = Path(name)
            summary = run_observer(
                scene=scene,
                diagnostic_root=temporary_root,
                decoded=decoded,
                corners=corners,
                scores=scores,
                observer_cfg=observer_cfg,
            )
            recomputed = temporary_root / diagnostic_path.name
            compare_diagnostics(diagnostic_path, recomputed)
    else:
        summary = run_observer(
            scene=scene,
            diagnostic_root=diagnostic_root,
            decoded=decoded,
            corners=corners,
            scores=scores,
            observer_cfg=observer_cfg,
        )
        if Path(summary.diagnostic_path).resolve() != diagnostic_path.resolve():
            raise RuntimeError("offline observer wrote an unexpected diagnostic path")
    if sha256(final_anchor) != anchor_sha:
        raise RuntimeError("sealed final-base anchor changed during offline observation")
    for record in input_records:
        if sha256(Path(record["depth_path"])) != record["depth_sha256"]:
            raise RuntimeError("selected CA depth changed during offline observation")
    base_report["diagnostic"] = {
        "path": str(diagnostic_path.resolve()),
        "sha256": sha256(diagnostic_path),
        "schema": "boxfusion.ca1m_native_b6_observer.v1",
        "feature_names": list(FEATURE_NAMES),
        "valid_evidence_rows": summary.valid_evidence_rows,
        "projectable_rows": summary.projectable_rows,
    }
    base_report["diagnostic_recovery"] = {
        "preexisting_orphan": recovered_orphan,
        "semantic_recomputation_exact": recovered_orphan,
        "runtime_only_field_ignored": (
            "summary_json.observer_seconds" if recovered_orphan else None
        ),
    }
    base_report["receipt_path"] = str(receipt_path.resolve())
    create_json(receipt_path, base_report)
    return base_report


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--mode", choices=("preflight", "run"), default="preflight")
    result.add_argument("--scene", required=True)
    result.add_argument("--config", type=Path, required=True)
    result.add_argument("--data-root", type=Path, required=True)
    result.add_argument("--final-base-root", type=Path, required=True)
    result.add_argument("--final-base-manifest", type=Path, required=True)
    result.add_argument("--diagnostics-root", type=Path, required=True)
    result.add_argument("--receipt", type=Path, required=True)
    return result


def main() -> int:
    report = collect(parser().parse_args())
    print(
        json.dumps(
            {
                "schema": report["schema"],
                "mode": report["mode"],
                "scene_id": report["scene_id"],
                "complete": report["complete"],
                "prediction_rows": report["prediction_rows"],
                "used_frames": len(report["frame_protocol"]["used_frame_ids"]),
                "diagnostic": (report.get("diagnostic") or {}).get("path"),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
