#!/usr/bin/env python3
"""Build immutable, label-free SPGroup3D mesh partition sidecars."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from typing import Any, Sequence

import numpy as np
import open3d as o3d
import torch

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from boxfusion.spgroup_partition_cache import (  # noqa: E402
    SCHEMA, SPGroupPartition, canonical_sha256, load_partition,
    read_axis_alignment, sha256_file, write_partition,
)


REPORT_SCHEMA = "boxfusion.spgroup3d_partition_export.v1"


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--scans-root", type=Path, required=True)
    value.add_argument("--scene-list", type=Path, required=True)
    value.add_argument("--output-root", type=Path, required=True)
    value.add_argument("--segmentator-root", type=Path, required=True)
    value.add_argument("--expected-segmentator-commit", required=True)
    value.add_argument("--expected-segmentator-binary-sha256", required=True)
    value.add_argument("--official-root", type=Path, required=True)
    value.add_argument("--expected-commit", required=True)
    value.add_argument("--report", type=Path, required=True)
    value.add_argument("--resume", action="store_true")
    return value


def _scenes(path: Path) -> list[str]:
    result = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not result or len(result) != len(set(result)):
        raise ValueError(f"{path}: scene list is empty or contains duplicates")
    return result


def _commit(root: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
    ).strip()


def _tree_clean(root: Path) -> bool:
    return not subprocess.check_output(
        ["git", "-C", str(root), "status", "--porcelain"], text=True
    ).strip()


def _write_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(report, indent=2, sort_keys=True) + "\n").encode()
    temporary: str | None = None
    try:
        descriptor, temporary = tempfile.mkstemp(
            dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
        )
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
        path.chmod(0o444)
    except FileExistsError as error:
        raise FileExistsError(f"immutable partition report exists: {path}") from error
    finally:
        if temporary is not None:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass


def export(args: argparse.Namespace) -> dict[str, Any]:
    official_root = args.official_root.resolve()
    actual_commit = _commit(official_root)
    if actual_commit != args.expected_commit:
        raise ValueError(f"official SPGroup3D commit mismatch: {actual_commit}")
    if not _tree_clean(official_root):
        raise ValueError("official SPGroup3D source tree is dirty")

    segmentator_root = args.segmentator_root.resolve()
    segmentator_commit = _commit(segmentator_root)
    if segmentator_commit != args.expected_segmentator_commit:
        raise ValueError(f"segmentator commit mismatch: {segmentator_commit}")
    segmentator_binary = segmentator_root / "csrc" / "build" / "libsegmentator.so"
    if not segmentator_binary.is_file():
        raise FileNotFoundError(f"compiled segmentator binding is absent: {segmentator_binary}")
    segmentator_binary_sha = sha256_file(segmentator_binary)
    if segmentator_binary_sha != args.expected_segmentator_binary_sha256:
        raise ValueError(f"segmentator binary SHA256 mismatch: {segmentator_binary_sha}")
    segmentator_sources = [
        segmentator_root / "main.py",
        segmentator_root / "__init__.py",
        segmentator_root / "csrc" / "segmentator.cpp",
        segmentator_root / "csrc" / "CMakeLists.txt",
    ]
    segmentator_source_sha = canonical_sha256(
        [{"path": str(path), "sha256": sha256_file(path)} for path in segmentator_sources]
    )
    segmentator_status = subprocess.check_output(
        ["git", "-C", str(segmentator_root), "status", "--porcelain"], text=True
    ).splitlines()
    if str(segmentator_root.parent) not in sys.path:
        sys.path.insert(0, str(segmentator_root.parent))
    from segmentator import segment_mesh  # type: ignore  # noqa: E402

    config = {
        "schema": "boxfusion.spgroup3d_partition_config.v1",
        "observer_only": True,
        "mutation_enabled": False,
        "applied_count": 0,
        "ground_truth_access": False,
        "clip_access": False,
        "semantic_head_used": False,
        "source": "official_SPGroup3D_segmentator.segment_mesh",
        "official_commit": actual_commit,
        "segmentator_commit": segmentator_commit,
        "segmentator_source_sha256": segmentator_source_sha,
        "segmentator_binary_sha256": segmentator_binary_sha,
        "segmentator_local_patch_disclosure": segmentator_status,
        "coordinate_frames": ["scannet_unaligned_world", "scannet_axis_aligned"],
        "raw_mesh_topology_used": True,
        "online_eligible": False,
        "online_blocker": "offline reconstructed ScanNet mesh topology",
    }
    config_sha = canonical_sha256(config)
    scene_reports: list[dict[str, Any]] = []
    for position, scene in enumerate(_scenes(args.scene_list.resolve()), start=1):
        scene_root = args.scans_root.resolve() / scene
        mesh_path = scene_root / f"{scene}_vh_clean_2.ply"
        meta_path = scene_root / f"{scene}.txt"
        if not mesh_path.is_file() or not meta_path.is_file():
            raise FileNotFoundError(f"{scene}: missing raw ScanNet mesh or metadata")
        target = args.output_root.resolve() / scene / "mesh_partition.npz"
        resumed = target.exists()
        start = time.perf_counter()
        if resumed:
            if not args.resume:
                raise FileExistsError(f"immutable partition exists: {target}")
            value = load_partition(target)
            metadata = value.metadata
            if (
                value.scene_id != scene
                or metadata.get("mesh_sha256") != sha256_file(mesh_path)
                or metadata.get("meta_sha256") != sha256_file(meta_path)
                or metadata.get("config_sha256") != config_sha
            ):
                raise ValueError(f"{scene}: resumed partition provenance mismatch")
        else:
            mesh = o3d.io.read_triangle_mesh(str(mesh_path))
            vertices = np.asarray(mesh.vertices, dtype=np.float32)
            faces = np.asarray(mesh.triangles, dtype=np.int64)
            colors = np.asarray(mesh.vertex_colors, dtype=np.float32)
            if colors.shape != vertices.shape:
                raise ValueError(f"{mesh_path}: vertex colors are absent")
            superpoints = segment_mesh(
                torch.from_numpy(vertices.copy()), torch.from_numpy(faces.copy())
            ).cpu().numpy().astype(np.int64, copy=False)
            _, superpoints = np.unique(superpoints, return_inverse=True)
            axis_alignment = read_axis_alignment(meta_path)
            aligned = (
                vertices @ axis_alignment[:3, :3].T + axis_alignment[:3, 3]
            ).astype(np.float32)
            metadata = {
                **config,
                "schema": SCHEMA,
                "scene_id": scene,
                "config_sha256": config_sha,
                "mesh_path": str(mesh_path),
                "mesh_sha256": sha256_file(mesh_path),
                "meta_path": str(meta_path),
                "meta_sha256": sha256_file(meta_path),
                "vertex_count": int(vertices.shape[0]),
                "face_count": int(faces.shape[0]),
                "superpoint_count": int(np.unique(superpoints).size),
            }
            value = SPGroupPartition(
                scene_id=scene,
                vertices_unaligned=vertices,
                vertices_aligned=aligned,
                colors=colors,
                faces=faces,
                superpoint_ids=superpoints.astype(np.int32),
                axis_alignment=axis_alignment,
                metadata=metadata,
            )
            write_partition(target, value)
        elapsed = time.perf_counter() - start
        row = {
            "scene_id": scene,
            "resumed": resumed,
            "partition": str(target),
            "partition_sha256": sha256_file(target),
            "vertices": value.vertex_count,
            "faces": int(value.faces.shape[0]),
            "superpoints": value.superpoint_count,
            "wall_s": elapsed,
        }
        scene_reports.append(row)
        print(
            f"[{position}] {scene}: vertices={row['vertices']}, "
            f"superpoints={row['superpoints']}, wall={elapsed:.3f}s",
            flush=True,
        )
    report = {
        "schema": REPORT_SCHEMA,
        "observer_only": True,
        "mutation_enabled": False,
        "applied_count": 0,
        "ground_truth_access": False,
        "clip_access": False,
        "semantic_head_used": False,
        "official_commit": actual_commit,
        "official_tree_clean": True,
        "segmentator_commit": segmentator_commit,
        "segmentator_source_sha256": segmentator_source_sha,
        "segmentator_binary_sha256": segmentator_binary_sha,
        "segmentator_local_patch_disclosure": segmentator_status,
        "partition_config": config,
        "partition_config_sha256": config_sha,
        "scene_count": len(scene_reports),
        "scenes": scene_reports,
    }
    _write_report(args.report.resolve(), report)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    report = export(args)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
