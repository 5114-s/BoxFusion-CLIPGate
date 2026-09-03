#!/usr/bin/env python3
"""Export paired Boxer-DINO consistency for immutable R4-D sidecars."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Any, Sequence

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from boxfusion.tr3d_r2_observer import TR3DR2ObserverConfig  # noqa: E402
from boxfusion.tr3d_r2_provenance import (  # noqa: E402
    canonical_json_sha256, code_artifact_tree_sha256, frame_artifact_tree,
    load_prefix_manifest, load_resolved_poses, sha256_file,
)
from boxfusion.tr3d_r2b_dino import (  # noqa: E402
    BOXER_DINO_MODEL, BoxerDINOv3Config, BoxerDINOv3DenseEncoder,
)
from boxfusion.tr3d_r2b_observer import TR3DR2BFrameBundle  # noqa: E402
from boxfusion.tr3d_r4_smov_cache import load_r4_depth_sidecar  # noqa: E402
from boxfusion.tr3d_r4_smov_feature import (  # noqa: E402
    observe_r3_replacement_pair_features,
)
from boxfusion.tr3d_r4_smov_feature_cache import (  # noqa: E402
    load_r4_feature_sidecar, make_r4_feature_sidecar,
    write_r4_feature_sidecar,
)
from tools.tr3d_data import discover_frame_bundle, read_scene_list  # noqa: E402


REPORT_SCHEMA = "boxfusion.tr3d_r4_smov_feature_export.v1"
POSE_SOURCE = "scannet_g0_resolved_pose_v1"


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--r4-depth-cache-root", type=Path, required=True)
    value.add_argument("--r4-feature-cache-root", type=Path, required=True)
    value.add_argument("--prefix-manifest", type=Path, required=True)
    value.add_argument("--frames-root", type=Path, required=True)
    value.add_argument("--scene-list", type=Path, required=True)
    value.add_argument("--active-prediction-root", type=Path, required=True)
    value.add_argument("--official-boxer-root", type=Path, required=True)
    value.add_argument("--expected-boxer-commit", required=True)
    value.add_argument("--expected-dino-sha256", required=True)
    value.add_argument("--prefix-id", default="p100")
    value.add_argument("--input-height", type=int, default=960)
    value.add_argument("--input-width", type=int, default=960)
    value.add_argument("--precision", choices=("float32", "bfloat16"), default="bfloat16")
    value.add_argument("--device", choices=("cuda", "cpu", "mps"), default="cuda")
    value.add_argument("--min-support-points", type=int, default=2)
    value.add_argument("--min-feature-cells", type=int, default=1)
    value.add_argument("--shard-index", type=int, default=0)
    value.add_argument("--num-shards", type=int, default=1)
    value.add_argument("--resume", action="store_true")
    value.add_argument("--report", type=Path, required=True)
    return value


def _tree_hash(root: Path, scenes: Sequence[str]) -> str:
    return canonical_json_sha256(
        [
            {"name": f"{scene}_boxes.pkl", "sha256": sha256_file(root / f"{scene}_boxes.pkl")}
            for scene in scenes
        ]
    )


def _configuration(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "schema": "boxfusion.tr3d_r4_smov_feature_config.v1",
        "observer_only": True, "mutation_enabled": False, "applied_count": 0,
        "ground_truth_access": False, "clip_access": False,
        "feature_model": BOXER_DINO_MODEL,
        "official_boxer_commit": args.expected_boxer_commit,
        "dino_checkpoint_sha256": args.expected_dino_sha256,
        "input_shape": [args.input_height, args.input_width],
        "precision": args.precision, "device": args.device,
        "view_source": "exact_r4d_common_topk",
        "region_source": "paired_metric_depth_support_points",
        "pooling": "unique_dense_cells_equal_weight_l2",
        "minimum_support_points": args.min_support_points,
        "minimum_feature_cells": args.min_feature_cells,
        "standalone_encoder_used_for_observer": True,
        "online_activation_requires_selective_boxer_dino0_reuse": True,
        "second_online_backbone_forbidden": True,
    }


def _write_report(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
    temporary: str | None = None
    try:
        descriptor, temporary = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
        path.chmod(0o444)
    except FileExistsError as error:
        raise FileExistsError(f"immutable R4-F report exists: {path}") from error
    finally:
        if temporary is not None:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass


def export(args: argparse.Namespace) -> dict[str, Any]:
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("shard index must lie in [0,num-shards)")
    all_scenes = read_scene_list(args.scene_list.resolve())
    scenes = [scene for index, scene in enumerate(all_scenes) if index % args.num_shards == args.shard_index]
    rows = load_prefix_manifest(args.prefix_manifest.resolve(), prefix_id=args.prefix_id)
    config_payload = _configuration(args)
    config_sha = canonical_json_sha256(config_payload)
    code_sha = code_artifact_tree_sha256(
        (
            _ROOT / "boxfusion" / "tr3d_r2_geometry.py",
            _ROOT / "boxfusion" / "tr3d_r2b_observer.py",
            _ROOT / "boxfusion" / "tr3d_r2b_dino.py",
            _ROOT / "boxfusion" / "tr3d_r4_smov_feature.py",
            _ROOT / "boxfusion" / "tr3d_r4_smov_feature_cache.py",
            args.official_boxer_root.resolve() / "boxernet" / "dinov3_wrapper.py",
            Path(__file__),
        )
    )
    encoder = BoxerDINOv3DenseEncoder(
        BoxerDINOv3Config(
            official_root=str(args.official_boxer_root.resolve()),
            expected_commit=args.expected_boxer_commit,
            checkpoint_sha256=args.expected_dino_sha256,
            input_height=args.input_height,
            input_width=args.input_width,
            precision=args.precision,
            device=args.device,
        )
    )
    # Verify immutable assets before any scene output is created.
    if encoder.verified_commit != args.expected_boxer_commit or encoder.verified_checkpoint_sha256 != args.expected_dino_sha256:
        raise RuntimeError("Boxer/DINO asset verification failed")
    depth_config = TR3DR2ObserverConfig(
        image_shape=(480, 640), pose_source=POSE_SOURCE, top_k=5,
        pixel_stride=4, depth_scale=1000.0, margin=0.05,
        min_depth=0.10, max_depth=8.0, near_clip=1e-3,
    )
    tree_before = _tree_hash(args.active_prediction_root.resolve(), scenes)
    scene_reports: list[dict[str, Any]] = []
    resumed_count = 0
    total_pairs = 0
    for position, scene in enumerate(scenes, start=1):
        row = rows[scene]
        if Path(str(row.get("source_frames_root", ""))).resolve() != args.frames_root.resolve():
            raise ValueError(f"{scene}: frames root provenance mismatch")
        depth_path = args.r4_depth_cache_root.resolve() / scene / f"{args.prefix_id}.r4d.npz"
        depth_sha = sha256_file(depth_path)
        depth_sidecar = load_r4_depth_sidecar(depth_path)
        if depth_sidecar.scene_id != scene or depth_sidecar.prefix_id != args.prefix_id:
            raise ValueError(f"{scene}: R4-D identity mismatch")
        bundle = discover_frame_bundle(args.frames_root.resolve(), scene)
        frame_tree_sha, _ = frame_artifact_tree(row, bundle)
        if frame_tree_sha != depth_sidecar.frame_artifact_tree_sha256:
            raise ValueError(f"{scene}: R4-D frame provenance mismatch")
        poses = load_resolved_poses(row, bundle)
        used = [int(value) for value in row["used_frame_ids"]]
        feature_bundle = TR3DR2BFrameBundle(
            scene_id=scene, pose_source=POSE_SOURCE,
            color={frame: bundle.color[frame] for frame in used},
            depth={frame: bundle.depth[frame] for frame in used},
            pose=poses,
            intrinsic_depth=bundle.intrinsic_depth,
            intrinsic_color=bundle.intrinsic_color,
            extrinsic_depth=bundle.extrinsic_depth,
            extrinsic_color=bundle.extrinsic_color,
        )
        target = args.r4_feature_cache_root.resolve() / scene / f"{args.prefix_id}.r4f.npz"
        was_resumed = target.exists()
        if was_resumed:
            if not args.resume:
                raise FileExistsError(f"immutable R4-F sidecar exists: {target}")
            sidecar = load_r4_feature_sidecar(target)
            if (
                sidecar.r4_depth_sidecar_sha256 != depth_sha
                or sidecar.frame_artifact_tree_sha256 != frame_tree_sha
                or sidecar.r4_feature_config_sha256 != config_sha
                or sidecar.r4_feature_code_sha256 != code_sha
            ):
                raise ValueError(f"{scene}: resumed R4-F provenance mismatch")
            resumed_count += 1
        else:
            observation = observe_r3_replacement_pair_features(
                anchor_boxes_world=depth_sidecar.anchor_boxes_world,
                candidate_boxes_world=depth_sidecar.candidate_boxes_world,
                proposal_ids=depth_sidecar.proposal_ids,
                anchor_indices=depth_sidecar.anchor_indices,
                topk_frame_ids=depth_sidecar.topk_frame_ids,
                topk_view_valid=depth_sidecar.topk_view_valid,
                frame_bundle=feature_bundle,
                depth_config=depth_config,
                encode_rgb=encoder,
                min_support_points=args.min_support_points,
                min_feature_cells=args.min_feature_cells,
            )
            sidecar = make_r4_feature_sidecar(
                observation=observation, prefix_id=args.prefix_id,
                r4_depth_sidecar_sha256=depth_sha,
                frame_artifact_tree_sha256=frame_tree_sha,
                r4_feature_config_sha256=config_sha,
                r4_feature_code_sha256=code_sha,
                official_boxer_commit=args.expected_boxer_commit,
                dino_checkpoint_sha256=args.expected_dino_sha256,
            )
            write_r4_feature_sidecar(target, sidecar)
        total_pairs += sidecar.pair_count
        scene_reports.append(
            {
                "scene_id": scene, "selected_pairs": sidecar.pair_count,
                "valid_anchor_feature_views": int(sidecar.aggregate_feature_view_count[:, 0].sum()),
                "valid_candidate_feature_views": int(sidecar.aggregate_feature_view_count[:, 1].sum()),
                "feature_runtime_s": sidecar.feature_runtime_s,
                "geometry_runtime_s": sidecar.geometry_runtime_s,
                "total_runtime_s": sidecar.total_runtime_s,
                "resumed": was_resumed, "sidecar": str(target),
            }
        )
        print(f"[{position}/{len(scenes)}] {scene}: pairs={sidecar.pair_count}, feature_views={scene_reports[-1]['valid_anchor_feature_views']}/{scene_reports[-1]['valid_candidate_feature_views']}, runtime={sidecar.total_runtime_s:.3f}s", flush=True)
    tree_after = _tree_hash(args.active_prediction_root.resolve(), scenes)
    if tree_before != tree_after:
        raise RuntimeError("active R3 prediction tree changed during R4-F observer")
    return {
        "schema": REPORT_SCHEMA, "observer_only": True, "mutation_enabled": False,
        "applied_count": 0, "ground_truth_access": False, "clip_access": False,
        "active_prediction_identity_ok": True,
        "active_prediction_tree_sha256_before": tree_before,
        "active_prediction_tree_sha256_after": tree_after,
        "scene_count": len(scenes), "selected_pair_count": total_pairs,
        "resumed_scene_count": resumed_count,
        "feature_config": config_payload, "feature_config_sha256": config_sha,
        "feature_code_sha256": code_sha,
        "standalone_feature_runtime_not_online_latency": True,
        "online_activation_requires_existing_boxer_dino0_reuse": True,
        "scenes": scene_reports,
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    report = export(args)
    _write_report(args.report.resolve(), report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
