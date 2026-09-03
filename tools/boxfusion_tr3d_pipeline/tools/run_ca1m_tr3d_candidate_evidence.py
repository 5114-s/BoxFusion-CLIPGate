#!/usr/bin/env python3
"""Collect GT-free CA-1M depth evidence for terminal-TR3D candidates.

The terminal observer cache contains genuine TR3D geometry but only a sparse
point-count statistic.  This tool applies the already frozen CA-1M native-B6
depth/free-space observer to *every* candidate, using exactly the causal
keyframes sealed in the terminal cache.  It never reads train or validation
ground truth and cannot mutate a prediction.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
from typing import Any

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
import sys

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from boxfusion.ca1m_native_b6_observer import (  # noqa: E402
    CA1MNativeB6Config,
    CA1MNativeB6Observer,
)
from boxfusion.ca1m_tr3d_terminal import SCHEMA as TERMINAL_SCHEMA  # noqa: E402


SCENE_RE = re.compile(r"^[0-9]{8}$")


def _regular(path: Path, name: str) -> Path:
    if path.is_symlink():
        raise ValueError(f"{name} must not be a symlink: {path}")
    value = path.resolve()
    if not value.is_file() or value.is_symlink() or value.stat().st_size <= 0:
        raise FileNotFoundError(f"missing regular {name}: {value}")
    return value


def _directory(path: Path, name: str) -> Path:
    if path.is_symlink():
        raise ValueError(f"{name} must not be a symlink: {path}")
    value = path.resolve()
    if not value.is_dir() or value.is_symlink():
        raise FileNotFoundError(f"missing {name}: {value}")
    return value


def _scenes(path: Path, selected: list[str]) -> tuple[str, ...]:
    source = _regular(path, "scene list")
    all_scenes = tuple(row.strip() for row in source.read_text().splitlines() if row.strip())
    if (
        not all_scenes
        or len(all_scenes) != len(set(all_scenes))
        or any(SCENE_RE.fullmatch(scene) is None for scene in all_scenes)
    ):
        raise ValueError("scene list is empty, duplicate, or malformed")
    if not selected:
        return all_scenes
    if len(selected) != len(set(selected)) or set(selected) - set(all_scenes):
        raise ValueError("selected scenes are duplicate or outside the frozen list")
    return tuple(scene for scene in all_scenes if scene in set(selected))


def _numeric_pngs(path: Path, name: str) -> dict[int, Path]:
    directory = _directory(path, name)
    result: dict[int, Path] = {}
    for item in directory.iterdir():
        if item.is_symlink() or not item.is_file() or item.suffix.lower() != ".png":
            continue
        try:
            frame = int(item.stem)
        except ValueError:
            continue
        if frame in result:
            raise ValueError(f"duplicate numeric frame {frame}: {directory}")
        result[frame] = item.resolve()
    if not result:
        raise ValueError(f"no numeric PNGs in {directory}")
    return result


def _scene_metadata(scene_root: Path) -> tuple[dict[int, Path], np.ndarray, np.ndarray]:
    depth = _numeric_pngs(scene_root / "depth", "depth directory")
    poses = np.asarray(
        np.load(_regular(scene_root / "all_poses.npy", "all_poses"), allow_pickle=False),
        dtype=np.float64,
    )
    if poses.ndim != 3 or poses.shape[1:] != (4, 4) or not np.isfinite(poses).all():
        raise ValueError(f"invalid pose array: {scene_root}")
    per_frame = scene_root / "K_depth_per_frame.npy"
    if per_frame.exists() or per_frame.is_symlink():
        intrinsics = np.asarray(
            np.load(_regular(per_frame, "per-frame intrinsics"), allow_pickle=False),
            dtype=np.float64,
        )
    else:
        intrinsic = np.asarray(
            np.loadtxt(_regular(scene_root / "K_depth.txt", "K_depth")),
            dtype=np.float64,
        ).reshape(3, 3)
        intrinsics = np.broadcast_to(intrinsic, (len(poses), 3, 3)).copy()
    if intrinsics.shape != (len(poses), 3, 3) or not np.isfinite(intrinsics).all():
        raise ValueError(f"intrinsics/pose count mismatch: {scene_root}")
    return depth, poses, intrinsics


def _terminal_cache(path: Path, scene: str) -> dict[str, np.ndarray]:
    source = _regular(path, "terminal observer cache")
    with np.load(source, allow_pickle=False) as archive:
        required = {
            "schema", "complete", "observer_only", "mutation_enabled",
            "ground_truth_access", "scene_id", "adapter_mode", "used_frame_ids",
            "candidate_corners", "candidate_scores",
        }
        if not required.issubset(set(archive.files)):
            raise ValueError(f"terminal cache is missing required fields: {source}")
        values = {name: np.array(archive[name], copy=True) for name in required}
    scalar = {
        "schema": TERMINAL_SCHEMA,
        "complete": True,
        "observer_only": True,
        "mutation_enabled": False,
        "ground_truth_access": False,
        "scene_id": scene,
        "adapter_mode": "genuine",
    }
    for name, expected in scalar.items():
        value = np.asarray(values[name])
        if value.shape != () or value.item() != expected:
            raise ValueError(f"terminal cache field {name} disagrees for {scene}")
    corners = values["candidate_corners"]
    scores = values["candidate_scores"]
    frames = values["used_frame_ids"]
    if (
        corners.dtype != np.float32
        or corners.shape != (len(corners), 8, 3)
        or scores.dtype != np.float32
        or scores.shape != (len(corners),)
        or not np.isfinite(corners).all()
        or not np.isfinite(scores).all()
        or np.any(scores < 0.0)
        or np.any(scores > 1.0)
    ):
        raise ValueError(f"terminal candidate arrays are invalid for {scene}")
    if (
        frames.dtype != np.int64
        or frames.ndim != 1
        or not len(frames)
        or np.any(frames < 0)
        or np.any(np.diff(frames) <= 0)
    ):
        raise ValueError(f"terminal used_frame_ids are invalid for {scene}")
    return {"corners": corners, "scores": scores, "frames": frames}


def run(args: argparse.Namespace) -> dict[str, Any]:
    scenes = _scenes(args.scene_list, list(args.scene))
    data_root = _directory(args.data_root, "processed CA-1M train root")
    cache_root = _directory(args.terminal_cache_root, "terminal cache root")
    if args.output_root.is_symlink():
        raise ValueError("candidate evidence output root must not be a symlink")
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    reports: dict[str, Any] = {}
    for scene in scenes:
        target = output_root / f"{scene}_ca1m_native_b6.npz"
        if target.exists() or target.is_symlink():
            raise FileExistsError(f"refusing existing candidate evidence: {target}")
        terminal = _terminal_cache(
            cache_root / f"{scene}_ca1m_tr3d_terminal.npz", scene
        )
        scene_root = _directory(data_root / scene, f"processed scene {scene}")
        depth, poses, intrinsics = _scene_metadata(scene_root)
        config = CA1MNativeB6Config(
            enabled=True,
            diagnostics_root=str(output_root),
            top_k=5,
            pixel_stride=4,
            margin=0.05,
            min_depth=0.10,
            max_depth=8.0,
            near_clip=1e-3,
            max_cached_keyframes=256,
        )
        observer = CA1MNativeB6Observer(config)
        for frame in terminal["frames"].tolist():
            if frame >= len(poses) or frame not in depth:
                raise ValueError(f"candidate evidence frame {frame} is missing for {scene}")
            depth_m = np.asarray(Image.open(depth[frame]), dtype=np.float32) / 1000.0
            observer.record_keyframe(
                scene_id=scene,
                frame_id=frame,
                source_frame_id=str(frame),
                depth_meters=depth_m,
                intrinsics=intrinsics[frame],
                camera_to_world=poses[frame],
            )
        summary = observer.finalize(
            scene_id=scene,
            corners=terminal["corners"],
            scores=terminal["scores"],
            stable_ids=np.arange(len(terminal["corners"]), dtype=np.int64),
        )
        reports[scene] = {
            "candidate_rows": len(terminal["corners"]),
            "valid_evidence_rows": summary.valid_evidence_rows,
            "frame_count": summary.frame_count,
            "observer_seconds": summary.observer_seconds,
            "diagnostic_path": summary.diagnostic_path,
        }
        print(CA1MNativeB6Observer.summary_text(summary), flush=True)
    return {
        "schema": "boxfusion.ca1m_tr3d_candidate_evidence_collection.v1",
        "complete": True,
        "ground_truth_access": False,
        "scene_count": len(scenes),
        "scenes": reports,
    }


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--scene-list", type=Path, required=True)
    value.add_argument("--scene", action="append", default=[])
    value.add_argument("--data-root", type=Path, required=True)
    value.add_argument("--terminal-cache-root", type=Path, required=True)
    value.add_argument("--output-root", type=Path, required=True)
    return value


if __name__ == "__main__":
    print(json.dumps(run(parser().parse_args()), sort_keys=True))
