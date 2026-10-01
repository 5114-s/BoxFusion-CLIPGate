#!/usr/bin/env python3
"""Bind official SPGroup3D grouping evidence to immutable R3 pairs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import time
from typing import Any, Sequence

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from boxfusion.spgroup_feature_cache import load_feature_sidecar  # noqa: E402
from boxfusion.spgroup_partition_cache import (  # noqa: E402
    canonical_sha256, load_partition, sha256_file,
)
from boxfusion.tr3d_r4_smov_cache import load_r4_depth_sidecar  # noqa: E402
from boxfusion.tr3d_r5_spgroup_cache import (  # noqa: E402
    SCHEMA, load_r5_sidecar, write_r5_sidecar,
)
from boxfusion.tr3d_r5_spgroup_observer import METRIC_NAMES, observe_pairs  # noqa: E402


REPORT_SCHEMA = "boxfusion.tr3d_r5_spgroup_export.v1"


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--r4-depth-cache-root", type=Path, required=True)
    value.add_argument("--partition-root", type=Path, required=True)
    value.add_argument("--feature-root", type=Path, required=True)
    value.add_argument("--output-root", type=Path, required=True)
    value.add_argument("--scene-list", type=Path, required=True)
    value.add_argument("--active-prediction-root", type=Path, required=True)
    value.add_argument("--prefix-id", default="p100")
    value.add_argument("--resume", action="store_true")
    value.add_argument("--report", type=Path, required=True)
    return value


def _scenes(path: Path) -> list[str]:
    scenes = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not scenes or len(scenes) != len(set(scenes)):
        raise ValueError(f"{path}: invalid scene list")
    return scenes


def _tree(root: Path, scenes: Sequence[str]) -> str:
    return canonical_sha256([
        {"scene_id": scene, "sha256": sha256_file(root / f"{scene}_boxes.pkl")}
        for scene in scenes
    ])


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
        raise FileExistsError(f"immutable R5 report exists: {path}") from error
    finally:
        if temporary is not None:
            try: os.unlink(temporary)
            except FileNotFoundError: pass


def export(args: argparse.Namespace) -> dict[str, Any]:
    scenes = _scenes(args.scene_list.resolve())
    config = {
        "schema": "boxfusion.tr3d_r5_spgroup_config.v1",
        "observer_only": True, "mutation_enabled": False, "applied_count": 0,
        "ground_truth_access": False, "clip_access": False,
        "semantic_head_used": False,
        "paired_source": "R3_selected_replacement_anchor_and_candidate",
        "metric_names": list(METRIC_NAMES),
        "box_geometry": "yaw_oriented_world_unaligned",
        "boundary_context_scale": 1.5,
        "active_rule": None,
        "raw_mesh_topology_used": True, "online_eligible": False,
    }
    config_sha = canonical_sha256(config)
    tree_before = _tree(args.active_prediction_root.resolve(), scenes)
    rows = []
    total_pairs = total_supported = total_feature_pairs = 0
    for position, scene in enumerate(scenes, start=1):
        depth_path = args.r4_depth_cache_root.resolve() / scene / f"{args.prefix_id}.r4d.npz"
        partition_path = args.partition_root.resolve() / scene / "mesh_partition.npz"
        feature_path = args.feature_root.resolve() / scene / "group_features.npz"
        source_hashes = {
            "r4_depth_sidecar_sha256": sha256_file(depth_path),
            "partition_sha256": sha256_file(partition_path),
            "feature_sidecar_sha256": sha256_file(feature_path),
        }
        target = args.output_root.resolve() / scene / f"{args.prefix_id}.r5g.npz"
        resumed = target.exists()
        start = time.perf_counter()
        if resumed:
            if not args.resume:
                raise FileExistsError(f"immutable R5 sidecar exists: {target}")
            sidecar = load_r5_sidecar(target)
            metadata = sidecar["metadata"]
            if (
                sidecar["scene_id"] != scene
                or metadata.get("config_sha256") != config_sha
                or any(metadata.get(key) != value for key, value in source_hashes.items())
            ):
                raise ValueError(f"{scene}: resumed R5 provenance mismatch")
            pair_count = len(sidecar["proposal_ids"])
            supported = int((sidecar["metrics"][:, :, 0] > 0).all(axis=1).sum())
            feature_pairs = int(sidecar["metric_valid"][:, :, 7].all(axis=1).sum())
        else:
            depth = load_r4_depth_sidecar(depth_path)
            partition = load_partition(partition_path)
            feature = load_feature_sidecar(feature_path)
            if depth.scene_id != scene or partition.scene_id != scene or feature.scene_id != scene:
                raise ValueError(f"{scene}: input sidecar identity mismatch")
            if feature.metadata.get("partition_sha256") != source_hashes["partition_sha256"]:
                raise ValueError(f"{scene}: feature/partition lineage mismatch")
            observation = observe_pairs(
                partition, feature, depth.anchor_boxes_world, depth.candidate_boxes_world
            )
            metadata = {
                **config, **source_hashes, "schema": SCHEMA, "scene_id": scene,
                "prefix_id": args.prefix_id, "config_sha256": config_sha,
                "proposal_id_order_sha256": hashlib.sha256(depth.proposal_ids.tobytes()).hexdigest(),
                "anchor_index_order_sha256": hashlib.sha256(depth.anchor_indices.tobytes()).hexdigest(),
            }
            write_r5_sidecar(
                target, scene_id=scene, proposal_ids=depth.proposal_ids,
                anchor_indices=depth.anchor_indices, observation=observation, metadata=metadata,
            )
            pair_count = len(depth.proposal_ids)
            supported = int((observation.metrics[:, :, 0] > 0).all(axis=1).sum())
            feature_pairs = int(observation.metric_valid[:, :, 7].all(axis=1).sum())
        wall = time.perf_counter() - start
        total_pairs += pair_count; total_supported += supported; total_feature_pairs += feature_pairs
        rows.append({
            "scene_id": scene, "resumed": resumed, "sidecar": str(target),
            "sidecar_sha256": sha256_file(target), "pair_count": pair_count,
            "mesh_supported_pairs": supported, "learned_feature_pairs": feature_pairs,
            "wall_s": wall,
        })
        print(
            f"[{position}/{len(scenes)}] {scene}: pairs={pair_count}, "
            f"mesh={supported}, learned={feature_pairs}, wall={wall:.3f}s", flush=True,
        )
    tree_after = _tree(args.active_prediction_root.resolve(), scenes)
    if tree_before != tree_after:
        raise RuntimeError("R3 active predictions changed during R5 observer")
    report = {
        "schema": REPORT_SCHEMA, "observer_only": True, "mutation_enabled": False,
        "applied_count": 0, "ground_truth_access": False, "clip_access": False,
        "semantic_head_used": False, "active_prediction_identity_ok": True,
        "active_prediction_tree_sha256_before": tree_before,
        "active_prediction_tree_sha256_after": tree_after,
        "config": config, "config_sha256": config_sha,
        "scene_count": len(scenes), "pair_count": total_pairs,
        "mesh_supported_pair_count": total_supported,
        "learned_feature_pair_count": total_feature_pairs,
        "scenes": rows,
    }
    _write(args.report.resolve(), report)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    report = export(parser().parse_args(argv))
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
