#!/usr/bin/env python3
"""Build the anchor-free CA-native TR3D proposal cache for terminal v4.

The entry point accepts no anchor, B6, raw model-config, or raw checkpoint
argument.  It derives the reachable final-base demo-loop frame IDs directly
from processed train100 RGB-D and resolves the model only through the sealed
CA-1M scratch-training binding.  The checked-in config currently keeps GPU
execution unauthorized; this implementation is for the later explicit P run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import sys
from typing import Any

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from boxfusion.ca1m_tr3d_checkpoint_binding import (  # noqa: E402
    load_checkpoint_binding,
    regular_directory,
    regular_file,
)
from boxfusion.ca1m_tr3d_inference_contract import (  # noqa: E402
    validate_ca1m_point_inference_config,
)
from boxfusion.ca1m_tr3d_terminal import (  # noqa: E402
    terminal_world_to_local,
    validate_homogeneous,
    voxel_downsample_first,
)
from boxfusion.ca1m_tr3d_terminal_v4 import (  # noqa: E402
    PREFIX_ID,
    ProposalCacheSummary,
    derive_demo_gap20_early_finalize_frame_ids,
    frame_lineage_json,
    load_proposal_cache,
    proposal_cache_payload,
    sha256_bytes,
    sha256_file,
    write_npz_create_only,
)
from boxfusion.ca1m_tr3d_worker_client import CA1MTR3DWorker  # noqa: E402
from boxfusion.tr3d_incremental_online import backproject_rgbd  # noqa: E402
from tools.preflight_ca1m_tr3d_terminal_train100_v4 import (  # noqa: E402
    validate_config,
)


SCENE = re.compile(r"^[0-9]{8}$")


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument(
        "--collection-config",
        type=Path,
        default=ROOT / "config/ca1m_tr3d_terminal_train100_v4_p3.json",
    )
    value.add_argument("--scene", action="append", default=[])
    value.add_argument("--device", default="cuda:0")
    return value


def _config(path: Path) -> tuple[Path, dict[str, Any]]:
    source = regular_file(path, "terminal-v4 config")
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError("terminal-v4 config is not JSON") from error
    if not isinstance(value, dict):
        raise ValueError("terminal-v4 config must be an object")
    return source, value


def _numeric_pngs(path: Path, name: str) -> dict[int, Path]:
    root = regular_directory(path, name)
    result: dict[int, Path] = {}
    for item in root.iterdir():
        if item.is_symlink() or not item.is_file() or item.suffix.lower() != ".png":
            continue
        try:
            index = int(item.stem)
        except ValueError:
            continue
        if index < 0 or index in result:
            raise ValueError(f"invalid/duplicate numeric frame in {root}")
        result[index] = item.resolve()
    if set(result) != set(range(len(result))) or not result:
        raise ValueError(f"{name} must contain contiguous 0..N-1 PNG frames")
    return result


def _scene_inputs(scene_root: Path) -> tuple[
    dict[int, Path], dict[int, Path], np.ndarray, np.ndarray, np.ndarray
]:
    rgb = _numeric_pngs(scene_root / "rgb", "processed RGB directory")
    depth = _numeric_pngs(scene_root / "depth", "processed depth directory")
    if set(rgb) != set(depth):
        raise ValueError(f"RGB/depth frame sets differ: {scene_root}")
    poses = np.asarray(
        np.load(regular_file(scene_root / "all_poses.npy", "processed poses"), allow_pickle=False),
        dtype=np.float64,
    )
    if poses.shape != (len(rgb), 4, 4) or not np.isfinite(poses).all():
        raise ValueError(f"processed pose/frame count differs: {scene_root}")
    per_frame = scene_root / "K_depth_per_frame.npy"
    if per_frame.exists() or per_frame.is_symlink():
        intrinsics = np.asarray(
            np.load(regular_file(per_frame, "per-frame depth intrinsics"), allow_pickle=False),
            dtype=np.float64,
        )
    else:
        intrinsic = np.asarray(
            np.loadtxt(regular_file(scene_root / "K_depth.txt", "depth intrinsics")),
            dtype=np.float64,
        ).reshape(3, 3)
        intrinsics = np.broadcast_to(intrinsic, (len(rgb), 3, 3)).copy()
    if intrinsics.shape != (len(rgb), 3, 3) or not np.isfinite(intrinsics).all():
        raise ValueError(f"processed intrinsics/frame count differs: {scene_root}")
    frames = derive_demo_gap20_early_finalize_frame_ids(len(rgb))
    for frame in frames.tolist():
        validate_homogeneous(poses[frame], f"pose[{frame}]")
        intrinsic = intrinsics[frame]
        if (
            intrinsic.shape != (3, 3)
            or intrinsic[0, 0] <= 0.0
            or intrinsic[1, 1] <= 0.0
            or not np.allclose(intrinsic[2], [0.0, 0.0, 1.0], atol=1e-6)
        ):
            raise ValueError(f"invalid depth intrinsics at frame {frame}: {scene_root}")
    return rgb, depth, poses, intrinsics, frames


def _points(
    *,
    rgb: dict[int, Path],
    depth: dict[int, Path],
    poses: np.ndarray,
    intrinsics: np.ndarray,
    frames: np.ndarray,
    pixel_stride: int,
    min_depth: float,
    max_depth: float,
    voxel_size: float,
    depth_scale: float,
) -> np.ndarray:
    parts: list[np.ndarray] = []
    for frame in frames.tolist():
        depth_raw = np.asarray(Image.open(depth[frame]))
        image = np.asarray(Image.open(rgb[frame]).convert("RGB"))
        if depth_raw.ndim != 2 or image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"malformed processed RGB-D at frame {frame}")
        # Match the CA scratch-training converter exactly: keep the raw RGB
        # and depth resolutions.  ``backproject_rgbd`` performs the same
        # nearest-neighbour color lookup into RGB for each sampled depth pixel.
        parts.append(
            backproject_rgbd(
                depth_raw.astype(np.float32) / float(depth_scale),
                image,
                intrinsics[frame],
                poses[frame],
                pixel_stride=pixel_stride,
                min_depth_m=min_depth,
                max_depth_m=max_depth,
            )
        )
    if not parts or not any(len(part) for part in parts):
        raise ValueError("demo early-finalize RGB-D sequence produced no valid points")
    result = voxel_downsample_first(np.concatenate(parts, axis=0), voxel_size)
    if not len(result):
        raise ValueError("voxelization removed every proposal input point")
    return np.ascontiguousarray(result, dtype=np.float32)


def _selected_scenes(path: Path, selected: list[str]) -> tuple[str, ...]:
    source = regular_file(path, "train100 scene list")
    scenes = tuple(row.strip() for row in source.read_text().splitlines() if row.strip())
    if len(scenes) != 100 or len(set(scenes)) != 100 or any(SCENE.fullmatch(x) is None for x in scenes):
        raise ValueError("proposal stage requires exact CA train100 scene list")
    if selected:
        requested = tuple(str(value) for value in selected)
        if len(requested) != len(set(requested)) or any(value not in scenes for value in requested):
            raise ValueError("--scene selection differs from train100 contract")
        wanted = set(requested)
        return tuple(scene for scene in scenes if scene in wanted)
    return scenes


def _code_manifest(
    *,
    config_path: Path,
    inference_config_path: Path,
    binding_path: Path,
    runtime_root: Path,
    worker_script: Path,
) -> str:
    sources = {
        "proposal_runner_v4": Path(__file__),
        "terminal_v4_core": ROOT / "boxfusion/ca1m_tr3d_terminal_v4.py",
        "ca_checkpoint_binding_core": ROOT / "boxfusion/ca1m_tr3d_checkpoint_binding.py",
        "terminal_geometry_core": ROOT / "boxfusion/ca1m_tr3d_terminal.py",
        "rgbd_backprojection": ROOT / "boxfusion/tr3d_incremental_online.py",
        "worker_client": ROOT / "boxfusion/ca1m_tr3d_worker_client.py",
        "worker": worker_script,
        "official_adapter": runtime_root / "boxfusion/tr3d_inference.py",
        "ca_point_inference_contract": ROOT / "boxfusion/ca1m_tr3d_inference_contract.py",
        "ca_point_inference_config": inference_config_path,
        "collection_config": config_path,
        "checkpoint_binding": binding_path,
    }
    rows = {name: sha256_file(regular_file(path, f"v4 code source {name}")) for name, path in sorted(sources.items())}
    return json.dumps(
        {"schema": "boxfusion.ca1m_tr3d_proposal_code_manifest.v4", "files": rows},
        separators=(",", ":"),
        sort_keys=True,
    )


def _parity_receipt(cfg: dict[str, Any]) -> tuple[Path, dict[str, Any]]:
    record = cfg.get("distribution_parity") or {}
    source = regular_file(Path(str(record.get("receipt", ""))), "v4 point-parity receipt")
    if sha256_file(source) != record.get("receipt_sha256"):
        raise ValueError("v4 point-parity receipt SHA256 differs")
    value = json.loads(source.read_text(encoding="utf-8"))
    if (
        value.get("schema")
        != "boxfusion.ca1m_tr3d_v4_lineage_training_point_parity.v1"
        or value.get("complete") is not True
        or value.get("point_array_parity_scene_count") != 100
        or value.get("point_byte_parity_scene_count") != 100
        or value.get("ground_truth_access") is not False
    ):
        raise ValueError("v4 point-parity receipt contract differs")
    return source, value


def _build_scene_points(
    *,
    data_root: Path,
    scene: str,
    processed: dict[str, Any],
    protocol: dict[str, Any],
) -> tuple[dict[int, Path], np.ndarray, np.ndarray, np.ndarray]:
    scene_root = regular_directory(data_root / scene, f"processed scene {scene}")
    rgb, depth, poses, intrinsics, frames = _scene_inputs(scene_root)
    points = _points(
        rgb=rgb,
        depth=depth,
        poses=poses,
        intrinsics=intrinsics,
        frames=frames,
        pixel_stride=int(protocol["pixel_stride"]),
        min_depth=float(protocol["min_depth_m"]),
        max_depth=float(protocol["max_depth_m"]),
        voxel_size=float(protocol["voxel_size_m"]),
        depth_scale=float(processed["depth_scale"]),
    )
    return rgb, poses, frames, points


def run(args: argparse.Namespace) -> dict[str, Any]:
    config_path, cfg = _config(args.collection_config)
    # This check occurs before resolving an executable or constructing a worker.
    if (cfg.get("proposal_stage") or {}).get("run_authorized") is not True:
        raise PermissionError("terminal-v4 proposal GPU run is not authorized")
    preflight = validate_config(config_path)
    if preflight["proposal_stage_runtime_authorized"] is not True:
        raise PermissionError("terminal-v4 proposal runtime preflight is not authorized")
    binding_cfg = cfg["ca_native_tr3d_binding"]
    binding = load_checkpoint_binding(Path(binding_cfg["path"]))
    inference_cfg = cfg["ca_native_tr3d_inference"]
    inference_config_path = regular_file(
        Path(inference_cfg["path"]), "CA-only point-inference config"
    )
    validate_ca1m_point_inference_config(
        inference_path=inference_config_path,
        inference_sha256=str(inference_cfg["sha256"]),
        effective_training_path=binding.effective_config_path,
        effective_training_sha256=binding.effective_config_sha256,
    )
    runtime = cfg["tr3d_runtime"]
    protocol = cfg["protocol"]
    processed = cfg["processed_rgbd"]
    stage = cfg["proposal_stage"]
    scenes = _selected_scenes(Path(cfg["scene_contract"]["path"]), list(args.scene))
    data_root = regular_directory(Path(processed["root"]), "processed train100 RGB-D root")
    output_root = Path(stage["output_root"])
    if output_root.is_symlink():
        raise ValueError("v4 proposal output root must not be a symlink")
    output_root = output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    worker_python = Path(runtime["worker_python"]).resolve()
    if not worker_python.is_file() or not os.access(worker_python, os.X_OK):
        raise FileNotFoundError(f"missing TR3D worker Python: {worker_python}")
    worker_script = regular_file(Path(runtime["worker_script"]), "CA TR3D worker")
    runtime_root = regular_directory(Path(runtime["runtime_root"]), "TR3D runtime root")
    project_root = regular_directory(Path(runtime["project_root"]), "TR3D project root")
    vendor_root = regular_directory(Path(runtime["vendor_root"]), "TR3D vendor root")
    code_json = _code_manifest(
        config_path=config_path,
        inference_config_path=inference_config_path,
        binding_path=binding.manifest_path,
        runtime_root=runtime_root,
        worker_script=worker_script,
    )
    code_sha = sha256_bytes(code_json.encode())
    reports: dict[str, Any] = {}
    pending: list[str] = []
    _parity_path, parity = _parity_receipt(cfg)
    parity_scenes = parity.get("scenes") or {}
    for scene in scenes:
        target = output_root / f"{scene}_ca1m_tr3d_proposals_v4.npz"
        if target.exists() or target.is_symlink():
            loaded = load_proposal_cache(
                target,
                expected_scene=scene,
                expected_binding_sha256=binding.manifest_sha256,
            )
            expected_point_sha = (parity_scenes.get(scene) or {}).get(
                "world_point_array_sha256"
            )
            if loaded["summary"].source_points_sha256 != expected_point_sha:
                raise ValueError(f"{scene}: resumed proposal cache differs from training distribution")
            reports[scene] = {**loaded["summary"].as_dict(), "resumed": True}
        else:
            pending.append(scene)
    if pending:
        # Fail before constructing the GPU worker unless every pending scene's
        # current RGB-D reconstruction still exactly matches the CA scratch
        # training distribution sealed by the parity receipt.
        for scene in pending:
            _rgb, _poses, _frames, verified_points = _build_scene_points(
                data_root=data_root,
                scene=scene,
                processed=processed,
                protocol=protocol,
            )
            verified_sha = hashlib.sha256(
                verified_points.tobytes(order="C")
            ).hexdigest()
            expected_sha = (parity_scenes.get(scene) or {}).get(
                "world_point_array_sha256"
            )
            if verified_sha != expected_sha:
                raise ValueError(
                    f"{scene}: current proposal points differ from CA scratch training distribution"
                )
            print(f"CA-1M v4 CPU distribution precheck | scene={scene}, parity=PASS", flush=True)
        with CA1MTR3DWorker(
            python=str(worker_python),
            worker_script=str(worker_script),
            runtime_root=str(runtime_root),
            config=str(inference_config_path),
            checkpoint=str(binding.checkpoint_path),
            project_root=str(project_root),
            vendor_root=str(vendor_root),
            startup_timeout_s=float(runtime["startup_timeout_s"]),
            device=str(args.device),
            extra_args=(
                "--score-threshold", str(protocol["score_threshold"]),
                "--max-proposals", str(protocol["max_proposals"]),
            ),
        ) as worker:
            if worker.adapter_mode != "genuine":
                raise ValueError("formal v4 proposal stage forbids synthetic TR3D")
            for scene in pending:
                rgb, poses, frames, points = _build_scene_points(
                    data_root=data_root,
                    scene=scene,
                    processed=processed,
                    protocol=protocol,
                )
                point_sha = hashlib.sha256(points.tobytes(order="C")).hexdigest()
                if point_sha != (parity_scenes.get(scene) or {}).get(
                    "world_point_array_sha256"
                ):
                    raise ValueError(f"{scene}: proposal point parity changed after CPU precheck")
                world_to_local = terminal_world_to_local(poses[int(frames[0])])
                result = worker.infer(
                    scene_id=scene,
                    prefix_id=PREFIX_ID,
                    points_world_xyzrgb=points,
                    world_to_local=world_to_local,
                )
                if result.source_points_sha256 != point_sha:
                    raise ValueError("worker/input proposal point SHA256 differs")
                lineage = frame_lineage_json(scene, len(rgb))
                summary = ProposalCacheSummary(
                    scene_id=scene,
                    frame_count=len(rgb),
                    used_frame_count=len(frames),
                    point_count=len(points),
                    candidate_count=len(result.scores),
                    model_runtime_s=float(result.model_runtime_s),
                    source_points_sha256=point_sha,
                    frame_lineage_sha256=sha256_bytes(lineage.encode()),
                    checkpoint_binding_sha256=binding.manifest_sha256,
                    checkpoint_sha256=binding.checkpoint_sha256,
                    config_sha256=str(inference_cfg["sha256"]),
                    code_manifest_sha256=code_sha,
                    adapter_mode=result.adapter_mode,
                    device=str(args.device),
                )
                payload = proposal_cache_payload(
                    summary=summary,
                    used_frame_ids=frames,
                    world_to_local=world_to_local,
                    candidate_corners_world=result.corners_world,
                    candidate_scores=result.scores,
                    candidate_point_count=result.point_counts,
                    candidate_boxes_local=result.boxes_local,
                    candidate_labels=result.labels,
                    frame_lineage=lineage,
                    code_manifest=code_json,
                )
                target = output_root / f"{scene}_ca1m_tr3d_proposals_v4.npz"
                write_npz_create_only(target, payload)
                reports[scene] = {**summary.as_dict(), "resumed": False}
    return {
        "schema": "boxfusion.ca1m_tr3d_proposal_run.v4",
        "complete": True,
        "stage": "P",
        "ground_truth_access": False,
        "anchor_access": False,
        "b6_access": False,
        "scene_count": len(scenes),
        "resumed_count": sum(bool(row["resumed"]) for row in reports.values()),
        "scenes": reports,
    }


def main() -> int:
    print(json.dumps(run(parser().parse_args()), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
