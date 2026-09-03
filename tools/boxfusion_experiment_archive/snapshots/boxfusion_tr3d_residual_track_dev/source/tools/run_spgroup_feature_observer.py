#!/usr/bin/env python3
"""Extract official pretrained SPGroup3D grouping features, observer-only."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from typing import Any, Sequence

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from boxfusion.spgroup_feature_cache import (  # noqa: E402
    SCHEMA, SPGroupFeatureSidecar, load_feature_sidecar, write_feature_sidecar,
)
from boxfusion.spgroup_official_adapter import (  # noqa: E402
    OFFICIAL_CHECKPOINT_SHA256, OFFICIAL_COMMIT, OfficialSPGroupEncoder,
)
from boxfusion.spgroup_partition_cache import (  # noqa: E402
    canonical_sha256, load_partition, sha256_file,
)


REPORT_SCHEMA = "boxfusion.spgroup3d_group_feature_export.v1"


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--partition-root", type=Path, required=True)
    value.add_argument("--feature-root", type=Path, required=True)
    value.add_argument("--scene-list", type=Path, required=True)
    value.add_argument("--active-prediction-root", type=Path, required=True)
    value.add_argument("--official-root", type=Path, required=True)
    value.add_argument("--checkpoint", type=Path, required=True)
    value.add_argument("--expected-commit", default=OFFICIAL_COMMIT)
    value.add_argument("--expected-checkpoint-sha256", default=OFFICIAL_CHECKPOINT_SHA256)
    value.add_argument("--device", default="cuda")
    value.add_argument("--resume", action="store_true")
    value.add_argument("--report", type=Path, required=True)
    return value


def _scenes(path: Path) -> list[str]:
    scenes = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not scenes or len(scenes) != len(set(scenes)):
        raise ValueError(f"{path}: invalid scene list")
    return scenes


def _prediction_tree(root: Path, scenes: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for scene in scenes:
        path = root / f"{scene}_boxes.pkl"
        digest.update(scene.encode())
        digest.update(bytes.fromhex(sha256_file(path)))
    return digest.hexdigest()


def _source_hash(root: Path) -> str:
    paths = [
        root / "projects" / "spgroup" / "biresnet.py",
        root / "projects" / "spgroup" / "Superpoint_encoder.py",
        root / "projects" / "configs" / "SPGroup_scannet.py",
    ]
    return canonical_sha256([{"path": str(path), "sha256": sha256_file(path)} for path in paths])


def _write(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(report, indent=2, sort_keys=True) + "\n").encode()
    temporary: str | None = None
    try:
        descriptor, temporary = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload); handle.flush(); os.fsync(handle.fileno())
        os.link(temporary, path); path.chmod(0o444)
    except FileExistsError as error:
        raise FileExistsError(f"immutable feature report exists: {path}") from error
    finally:
        if temporary is not None:
            try: os.unlink(temporary)
            except FileNotFoundError: pass


def export(args: argparse.Namespace) -> dict[str, Any]:
    scenes = _scenes(args.scene_list.resolve())
    official_root = args.official_root.resolve()
    commit = subprocess.check_output(["git", "-C", str(official_root), "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(["git", "-C", str(official_root), "status", "--porcelain"], text=True).strip()
    if commit != args.expected_commit or dirty:
        raise ValueError("official SPGroup3D source commit/tree verification failed")
    source_sha = _source_hash(official_root)
    config = {
        "schema": "boxfusion.spgroup3d_group_feature_config.v1",
        "observer_only": True, "mutation_enabled": False, "applied_count": 0,
        "ground_truth_access": False, "clip_access": False,
        "semantic_head_used": False,
        "official_commit": commit, "official_source_sha256": source_sha,
        "checkpoint_sha256": args.expected_checkpoint_sha256,
        "backbone": "BiResNet", "grouping_encoder": "SSG",
        "voxel_size_m": 0.02, "local_k": 8,
        "feature_channels": [64, 128, 128], "embedding_dim": 390,
        "geometry_aware_voting": True, "superpoint_voxel_fusion": True,
        "raw_mesh_topology_used": True, "online_eligible": False,
    }
    config_sha = canonical_sha256(config)
    tree_before = _prediction_tree(args.active_prediction_root.resolve(), scenes)
    encoder = OfficialSPGroupEncoder(
        official_root, args.checkpoint.resolve(),
        expected_checkpoint_sha256=args.expected_checkpoint_sha256,
        device=args.device,
    )
    rows = []
    for position, scene in enumerate(scenes, start=1):
        partition_path = args.partition_root.resolve() / scene / "mesh_partition.npz"
        partition_sha = sha256_file(partition_path)
        target = args.feature_root.resolve() / scene / "group_features.npz"
        resumed = target.exists()
        start = time.perf_counter()
        if resumed:
            if not args.resume:
                raise FileExistsError(f"immutable feature sidecar exists: {target}")
            sidecar = load_feature_sidecar(target)
            if (
                sidecar.scene_id != scene
                or sidecar.metadata.get("partition_sha256") != partition_sha
                or sidecar.metadata.get("config_sha256") != config_sha
            ):
                raise ValueError(f"{scene}: resumed feature provenance mismatch")
        else:
            partition = load_partition(partition_path)
            features = encoder.encode(
                partition.vertices_aligned, partition.colors, partition.superpoint_ids
            )
            metadata = {
                **config, "schema": SCHEMA, "scene_id": scene,
                "partition_sha256": partition_sha,
                "partition_config_sha256": partition.metadata["config_sha256"],
                "config_sha256": config_sha,
                "group_count": int(len(features.superpoint_ids)),
            }
            sidecar = SPGroupFeatureSidecar(scene_id=scene, features=features, metadata=metadata)
            write_feature_sidecar(target, sidecar)
        wall = time.perf_counter() - start
        rows.append({
            "scene_id": scene, "resumed": resumed, "sidecar": str(target),
            "sidecar_sha256": sha256_file(target),
            "group_count": int(len(sidecar.features.superpoint_ids)), "wall_s": wall,
        })
        print(f"[{position}/{len(scenes)}] {scene}: groups={rows[-1]['group_count']}, wall={wall:.3f}s", flush=True)
    tree_after = _prediction_tree(args.active_prediction_root.resolve(), scenes)
    if tree_before != tree_after:
        raise RuntimeError("R3 active prediction tree changed during SPGroup3D observer")
    report = {
        "schema": REPORT_SCHEMA, "observer_only": True, "mutation_enabled": False,
        "applied_count": 0, "ground_truth_access": False, "clip_access": False,
        "semantic_head_used": False, "active_prediction_identity_ok": True,
        "active_prediction_tree_sha256_before": tree_before,
        "active_prediction_tree_sha256_after": tree_after,
        "feature_config": config, "feature_config_sha256": config_sha,
        "scene_count": len(rows), "scenes": rows,
    }
    _write(args.report.resolve(), report)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    report = export(parser().parse_args(argv))
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
