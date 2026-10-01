#!/usr/bin/env python3
"""Run GT-free terminal TR3D observation on processed CA-1M scenes.

The tool consumes the exact keyframe IDs sealed by the native-B6 collection,
recomputes the authorized CA-1M B6 scores, builds a causal terminal XYZRGB
cloud, and invokes the genuine one-class TR3D checkpoint through a persistent
worker.  It creates only immutable observer NPZ files; no prediction writer
or ground-truth path is present.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import pickle
import sys
from typing import Any

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from boxfusion.ca1m_native_b6_score import (  # noqa: E402
    load_ca1m_native_b6_scorer,
    load_native_observer_diagnostic,
    sha256_file as b6_sha256_file,
)
from boxfusion.ca1m_tr3d_terminal import (  # noqa: E402
    TerminalObserverSummary,
    associate_terminal_candidates,
    observation_payload,
    sha256_array,
    sha256_file,
    terminal_world_to_local,
    validate_homogeneous,
    validate_scene_id,
    voxel_downsample_first,
    write_npz_create_only,
)
from boxfusion.ca1m_tr3d_worker_client import CA1MTR3DWorker  # noqa: E402
from boxfusion.tr3d_incremental_online import backproject_rgbd  # noqa: E402


EXPECTED_CHECKPOINT_SHA256 = (
    "a484fd79093aa3004f4d2984e7ad8763c5d5cbec7edd04172d775643a8436448"
)
EXPECTED_CONFIG_SHA256 = (
    "e74b29335f32baa6595bcc84a9b3e4fdd14b92a7044abd408a44de95fc360dc4"
)


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--scene-list", type=Path, required=True)
    value.add_argument("--scene", action="append", default=[])
    value.add_argument("--data-root", type=Path, required=True)
    value.add_argument("--anchor-root", type=Path, required=True)
    value.add_argument("--materialized-active-root", type=Path)
    value.add_argument("--native-b6-diagnostics-root", type=Path, required=True)
    value.add_argument("--native-b6-checkpoint", type=Path, required=True)
    value.add_argument("--native-b6-manifest", type=Path, required=True)
    value.add_argument("--output-root", type=Path, required=True)
    value.add_argument("--worker-python", type=Path, required=True)
    value.add_argument(
        "--worker-script",
        type=Path,
        default=ROOT / "tools/ca1m_tr3d_terminal_worker.py",
    )
    value.add_argument("--runtime-root", type=Path, required=True)
    value.add_argument("--tr3d-config", type=Path, required=True)
    value.add_argument("--tr3d-checkpoint", type=Path, required=True)
    value.add_argument("--tr3d-project-root", type=Path, required=True)
    value.add_argument("--tr3d-vendor-root", type=Path, required=True)
    value.add_argument("--device", default="cuda:0")
    value.add_argument("--pixel-stride", type=int, default=4)
    value.add_argument("--voxel-size", type=float, default=0.01)
    value.add_argument("--min-depth", type=float, default=0.10)
    value.add_argument("--max-depth", type=float, default=6.0)
    value.add_argument("--near-iou", type=float, default=0.15)
    value.add_argument("--score-threshold", type=float, default=0.01)
    value.add_argument("--max-proposals", type=int, default=256)
    value.add_argument("--synthetic", action="store_true")
    return value


def _regular(path: Path, name: str) -> Path:
    if path.is_symlink():
        raise ValueError(f"{name} must not be a symlink: {path}")
    resolved = path.resolve()
    if not resolved.is_file() or resolved.is_symlink() or resolved.stat().st_size <= 0:
        raise FileNotFoundError(f"missing regular {name}: {resolved}")
    return resolved


def _directory(path: Path, name: str) -> Path:
    if path.is_symlink():
        raise ValueError(f"{name} must not be a symlink: {path}")
    resolved = path.resolve()
    if not resolved.is_dir() or resolved.is_symlink():
        raise FileNotFoundError(f"missing {name}: {resolved}")
    return resolved


def _executable(path: Path, name: str) -> Path:
    """Resolve the environment's conventional ``python -> pythonX`` link."""

    resolved = path.resolve()
    if not resolved.is_file() or not os.access(resolved, os.X_OK):
        raise FileNotFoundError(f"missing executable {name}: {resolved}")
    return resolved


def _code_manifest(args: argparse.Namespace) -> str:
    sources = {
        "runner": Path(__file__),
        "worker": args.worker_script,
        "terminal_core": ROOT / "boxfusion/ca1m_tr3d_terminal.py",
        "worker_client": ROOT / "boxfusion/ca1m_tr3d_worker_client.py",
        "native_b6_score": ROOT / "boxfusion/ca1m_native_b6_score.py",
        "rgbd_backprojection": ROOT / "boxfusion/tr3d_incremental_online.py",
        "official_adapter": args.runtime_root / "boxfusion/tr3d_inference.py",
    }
    files = {
        name: sha256_file(_regular(path, f"code source {name}"))
        for name, path in sorted(sources.items())
    }
    return json.dumps(
        {"schema": "boxfusion.ca1m_tr3d_terminal_code_manifest.v1", "files": files},
        sort_keys=True,
    )


def _read_scenes(path: Path, selected: list[str]) -> tuple[str, ...]:
    source = _regular(path, "scene list")
    scenes = tuple(
        line.strip()
        for line in source.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    )
    if not scenes or len(scenes) != len(set(scenes)):
        raise ValueError("scene list must be non-empty and duplicate-free")
    for scene in scenes:
        validate_scene_id(scene)
    if selected:
        requested = tuple(validate_scene_id(scene) for scene in selected)
        if len(requested) != len(set(requested)):
            raise ValueError("--scene values must be unique")
        missing = sorted(set(requested) - set(scenes))
        if missing:
            raise ValueError(f"requested scenes are absent from scene list: {missing}")
        return tuple(scene for scene in scenes if scene in set(requested))
    return scenes


def _prediction(path: Path) -> tuple[np.ndarray, np.ndarray]:
    source = _regular(path, "anchor prediction")
    with source.open("rb") as handle:
        payload = pickle.load(handle)  # noqa: S301 - local sealed artifact
        if handle.read(1):
            raise ValueError(f"prediction has trailing bytes: {source}")
    if (
        not isinstance(payload, list)
        or len(payload) != 1
        or not isinstance(payload[0], list)
    ):
        raise ValueError(f"prediction must contain one list batch: {source}")
    corners: list[np.ndarray] = []
    scores: list[float] = []
    for index, row in enumerate(payload[0]):
        if (
            not isinstance(row, tuple)
            or len(row) != 3
            or type(row[0]) is not int
            or row[0] != 0
        ):
            raise ValueError(f"invalid prediction row {index}: {source}")
        corner = np.asarray(row[1])
        score = float(row[2])
        if corner.dtype != np.dtype(np.float32) or corner.shape != (8, 3):
            raise ValueError(f"prediction corners must be float32 [8,3]: {source}")
        if not np.isfinite(corner).all() or not np.isfinite(score):
            raise ValueError(f"prediction row is non-finite: {source}")
        corners.append(np.array(corner, dtype=np.float32, order="C", copy=True))
        scores.append(score)
    return (
        np.stack(corners).astype(np.float32, copy=False)
        if corners
        else np.empty((0, 8, 3), dtype=np.float32),
        np.asarray(scores, dtype=np.float32),
    )


def _used_frame_ids(path: Path, scene_id: str) -> np.ndarray:
    source = _regular(path, "native-B6 diagnostic")
    with np.load(source, allow_pickle=False) as archive:
        if "scene_id" not in archive.files or "used_frame_ids" not in archive.files:
            raise ValueError(f"native-B6 diagnostic lacks frame lineage: {source}")
        stored_scene = np.asarray(archive["scene_id"])
        frames = np.array(archive["used_frame_ids"], copy=True)
    if stored_scene.shape != () or str(stored_scene.item()) != scene_id:
        raise ValueError(f"native-B6 diagnostic scene mismatch: {source}")
    if (
        frames.dtype != np.dtype(np.int64)
        or frames.ndim != 1
        or not len(frames)
        or np.any(frames < 0)
        or len(np.unique(frames)) != len(frames)
        or np.any(np.diff(frames) <= 0)
    ):
        raise ValueError(f"native-B6 used_frame_ids are invalid: {source}")
    return np.ascontiguousarray(frames)


def _numeric_files(root: Path, name: str) -> dict[int, Path]:
    directory = _directory(root, name)
    result: dict[int, Path] = {}
    for path in directory.iterdir():
        if path.is_symlink() or not path.is_file() or path.suffix.lower() != ".png":
            continue
        try:
            index = int(path.stem)
        except ValueError:
            continue
        if index in result:
            raise ValueError(f"duplicate numeric frame {index}: {directory}")
        result[index] = path.resolve()
    if not result:
        raise ValueError(f"no numeric PNG frames: {directory}")
    return result


def _scene_inputs(scene_root: Path, used_frames: np.ndarray) -> tuple[
    dict[int, Path], dict[int, Path], np.ndarray, np.ndarray
]:
    rgb = _numeric_files(scene_root / "rgb", "RGB directory")
    depth = _numeric_files(scene_root / "depth", "depth directory")
    poses = np.load(_regular(scene_root / "all_poses.npy", "all_poses"), allow_pickle=False)
    poses = np.asarray(poses, dtype=np.float64)
    if poses.ndim != 3 or poses.shape[1:] != (4, 4) or not np.isfinite(poses).all():
        raise ValueError(f"invalid all_poses.npy: {scene_root}")
    per_frame_path = scene_root / "K_depth_per_frame.npy"
    if per_frame_path.exists() or per_frame_path.is_symlink():
        intrinsics = np.load(_regular(per_frame_path, "per-frame intrinsics"), allow_pickle=False)
        intrinsics = np.asarray(intrinsics, dtype=np.float64)
    else:
        intrinsic = np.loadtxt(_regular(scene_root / "K_depth.txt", "K_depth"))
        intrinsic = np.asarray(intrinsic, dtype=np.float64).reshape(3, 3)
        intrinsics = np.broadcast_to(intrinsic, (len(poses), 3, 3)).copy()
    if intrinsics.shape != (len(poses), 3, 3) or not np.isfinite(intrinsics).all():
        raise ValueError(f"intrinsics/pose count mismatch: {scene_root}")
    for frame in used_frames.tolist():
        if frame >= len(poses) or frame not in rgb or frame not in depth:
            raise ValueError(f"missing used frame {frame}: {scene_root}")
        validate_homogeneous(poses[frame], f"pose[{frame}]")
    return rgb, depth, poses, intrinsics


def _terminal_points(
    *,
    rgb: dict[int, Path],
    depth: dict[int, Path],
    poses: np.ndarray,
    intrinsics: np.ndarray,
    used_frames: np.ndarray,
    pixel_stride: int,
    min_depth: float,
    max_depth: float,
    voxel_size: float,
) -> np.ndarray:
    parts: list[np.ndarray] = []
    for frame in used_frames.tolist():
        depth_raw = np.asarray(Image.open(depth[frame]))
        image = np.asarray(Image.open(rgb[frame]).convert("RGB"))
        if depth_raw.ndim != 2:
            raise ValueError(f"depth must be single-channel: {depth[frame]}")
        depth_m = depth_raw.astype(np.float32) / 1000.0
        points = backproject_rgbd(
            depth_m,
            image,
            intrinsics[frame],
            poses[frame],
            pixel_stride=pixel_stride,
            min_depth_m=min_depth,
            max_depth_m=max_depth,
        )
        parts.append(points)
    if not parts or not any(len(value) for value in parts):
        raise ValueError("terminal RGB-D prefix produced no valid points")
    points = np.concatenate(parts, axis=0)
    points = voxel_downsample_first(points, voxel_size)
    if not len(points):
        raise ValueError("terminal voxelization removed all points")
    return points


def run(args: argparse.Namespace) -> dict[str, Any]:
    scenes = _read_scenes(args.scene_list, list(args.scene))
    data_root = _directory(args.data_root, "processed CA-1M root")
    anchor_root = _directory(args.anchor_root, "anchor root")
    materialized_active_root = (
        _directory(args.materialized_active_root, "materialized active root")
        if args.materialized_active_root is not None
        else None
    )
    native_root = _directory(args.native_b6_diagnostics_root, "native-B6 diagnostic root")
    if args.output_root.is_symlink():
        raise ValueError("output root must not be a symlink")
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    _executable(args.worker_python, "TR3D worker Python")
    for path, name in (
        (args.worker_script, "CA-1M TR3D worker"),
        (args.tr3d_config, "TR3D config"),
        (args.tr3d_checkpoint, "TR3D checkpoint"),
        (args.native_b6_checkpoint, "native-B6 checkpoint"),
        (args.native_b6_manifest, "native-B6 manifest"),
    ):
        _regular(path, name)
    _directory(args.runtime_root, "TR3D runtime root")
    _directory(args.tr3d_project_root, "TR3D project root")
    _directory(args.tr3d_vendor_root, "TR3D vendor root")
    checkpoint_sha = sha256_file(args.tr3d_checkpoint)
    config_sha = sha256_file(args.tr3d_config)
    if checkpoint_sha != EXPECTED_CHECKPOINT_SHA256:
        raise ValueError("frozen ScanNet TR3D checkpoint SHA256 mismatch")
    if config_sha != EXPECTED_CONFIG_SHA256:
        raise ValueError("frozen ScanNet TR3D config SHA256 mismatch")
    b6_checkpoint_sha = b6_sha256_file(args.native_b6_checkpoint.resolve())
    b6_manifest_sha = b6_sha256_file(args.native_b6_manifest.resolve())
    scorer = load_ca1m_native_b6_scorer(
        args.native_b6_checkpoint,
        args.native_b6_manifest,
        require_activation_authorized=True,
    )
    if args.pixel_stride < 1:
        raise ValueError("pixel-stride must be positive")
    if not 0.0 < args.min_depth < args.max_depth:
        raise ValueError("invalid depth range")
    if args.voxel_size != 0.01:
        raise ValueError("first CA-1M transfer probe freezes voxel-size=0.01")
    frozen_protocol = (4, 0.01, 0.10, 6.0, 0.15, 0.01, 256)
    observed_protocol = (
        args.pixel_stride,
        args.voxel_size,
        args.min_depth,
        args.max_depth,
        args.near_iou,
        args.score_threshold,
        args.max_proposals,
    )
    if observed_protocol != frozen_protocol:
        raise ValueError(
            f"first CA-1M transfer protocol is frozen at {frozen_protocol}, "
            f"observed {observed_protocol}"
        )
    code_manifest_json = _code_manifest(args)
    code_manifest_sha = hashlib.sha256(code_manifest_json.encode("utf-8")).hexdigest()

    extra_args = [
        "--score-threshold", str(args.score_threshold),
        "--max-proposals", str(args.max_proposals),
    ]
    if args.synthetic:
        extra_args.append("--synthetic")
    reports: dict[str, Any] = {}
    with CA1MTR3DWorker(
        python=str(args.worker_python),
        worker_script=str(args.worker_script),
        runtime_root=str(args.runtime_root),
        config=str(args.tr3d_config),
        checkpoint=str(args.tr3d_checkpoint),
        project_root=str(args.tr3d_project_root),
        vendor_root=str(args.tr3d_vendor_root),
        startup_timeout_s=600.0,
        device=str(args.device),
        extra_args=extra_args,
    ) as worker:
        for scene in scenes:
            target = output_root / f"{scene}_ca1m_tr3d_terminal.npz"
            if target.exists() or target.is_symlink():
                raise FileExistsError(f"refusing existing terminal observer artifact: {target}")
            scene_root = _directory(data_root / scene, f"processed scene {scene}")
            anchor_path = _regular(anchor_root / f"{scene}_boxes.pkl", "G0 anchor")
            native_path = _regular(
                native_root / f"{scene}_ca1m_native_b6.npz", "native-B6 diagnostic"
            )
            corners, detector_scores = _prediction(anchor_path)
            evidence = load_native_observer_diagnostic(
                native_path,
                scene_id=scene,
                corners=corners,
                scores=detector_scores,
            )
            active_scores = np.asarray(
                scorer.predict(evidence["features"], detector_scores).scores,
                dtype=np.float32,
            )
            materialized_active_verified = False
            if materialized_active_root is not None:
                materialized_corners, materialized_scores = _prediction(
                    _regular(
                        materialized_active_root / f"{scene}_boxes.pkl",
                        "materialized active prediction",
                    )
                )
                if not np.array_equal(materialized_corners, corners):
                    raise ValueError("materialized active corners differ from G0 anchor")
                if not np.array_equal(materialized_scores, active_scores):
                    raise ValueError("materialized active scores differ from recomputation")
                materialized_active_verified = True
            used_frames = _used_frame_ids(native_path, scene)
            rgb, depth, poses, intrinsics = _scene_inputs(scene_root, used_frames)
            points = _terminal_points(
                rgb=rgb,
                depth=depth,
                poses=poses,
                intrinsics=intrinsics,
                used_frames=used_frames,
                pixel_stride=args.pixel_stride,
                min_depth=args.min_depth,
                max_depth=args.max_depth,
                voxel_size=args.voxel_size,
            )
            world_to_local = terminal_world_to_local(poses[int(used_frames[0])])
            result = worker.infer(
                scene_id=scene,
                prefix_id="p100_gap20",
                points_world_xyzrgb=points,
                world_to_local=world_to_local,
            )
            candidate_corners = np.ascontiguousarray(result.corners_world, dtype=np.float32)
            candidate_scores = np.ascontiguousarray(result.scores, dtype=np.float32)
            candidate_point_count = np.ascontiguousarray(
                result.point_counts,
                dtype=np.int64,
            )
            if result.source_points_sha256 != hashlib.sha256(
                points.tobytes(order="C")
            ).hexdigest():
                raise ValueError("worker/input source point SHA256 mismatch")
            if len(candidate_scores) and np.any(np.diff(candidate_scores) > 0.0):
                raise ValueError("TR3D candidates must be sorted by descending score")
            association = associate_terminal_candidates(
                anchor_corners=corners,
                anchor_scores=active_scores,
                candidate_corners=candidate_corners,
                candidate_scores=candidate_scores,
                near_iou=args.near_iou,
            )
            source_points_sha = hashlib.sha256(points.tobytes(order="C")).hexdigest()
            summary = TerminalObserverSummary(
                scene_id=scene,
                anchor_count=len(corners),
                candidate_count=len(candidate_corners),
                near_candidate_count=int(association.near_mask.sum()),
                represented_anchor_count=len(association.represented_anchor_indices),
                legacy_rule_selected_count=len(
                    association.legacy_rule_selected_candidate_rows
                ),
                used_frame_count=len(used_frames),
                point_count=len(points),
                model_runtime_s=float(result.model_runtime_s),
                source_anchor_prediction_sha256=sha256_file(anchor_path),
                active_anchor_scores_sha256=sha256_array(active_scores),
                native_b6_diagnostic_sha256=sha256_file(native_path),
                native_b6_checkpoint_sha256=b6_checkpoint_sha,
                native_b6_manifest_sha256=b6_manifest_sha,
                source_points_sha256=source_points_sha,
                checkpoint_sha256=checkpoint_sha,
                config_sha256=config_sha,
                code_manifest_sha256=code_manifest_sha,
                adapter_mode=result.adapter_mode,
                prefix_id="p100_gap20",
                device=str(args.device),
                pixel_stride=int(args.pixel_stride),
                voxel_size_m=float(args.voxel_size),
                min_depth_m=float(args.min_depth),
                max_depth_m=float(args.max_depth),
                near_iou=float(args.near_iou),
                score_threshold=float(args.score_threshold),
                max_proposals=int(args.max_proposals),
                materialized_active_verified=materialized_active_verified,
            )
            payload = observation_payload(
                summary=summary,
                used_frame_ids=used_frames,
                world_to_local=world_to_local,
                anchor_corners=corners,
                anchor_scores=active_scores,
                candidate_corners=candidate_corners,
                candidate_scores=candidate_scores,
                candidate_point_count=candidate_point_count,
                candidate_boxes_local=result.boxes_local,
                candidate_labels=result.labels,
                association=association,
                code_manifest_json=code_manifest_json,
            )
            write_npz_create_only(target, payload)
            reports[scene] = summary.as_dict()
            print(
                "CA-1M terminal TR3D observer | "
                f"scene={scene}, frames={len(used_frames)}, points={len(points)}, "
                f"anchors/candidates/near/legacy={len(corners)}/"
                f"{len(candidate_corners)}/{int(association.near_mask.sum())}/"
                f"{len(association.legacy_rule_selected_candidate_rows)}, "
                f"model_ms={result.model_runtime_s * 1000.0:.3f}",
                flush=True,
            )
    return {
        "schema": "boxfusion.ca1m_tr3d_terminal_observer_run.v1",
        "complete": True,
        "observer_only": True,
        "mutation_enabled": False,
        "ground_truth_access": False,
        "scene_count": len(scenes),
        "scenes": reports,
    }


def main() -> int:
    result = run(parser().parse_args())
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
