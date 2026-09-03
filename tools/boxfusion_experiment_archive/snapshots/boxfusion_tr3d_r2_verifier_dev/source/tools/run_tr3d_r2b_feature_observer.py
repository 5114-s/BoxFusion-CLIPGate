#!/usr/bin/env python3
"""Export immutable R2b DINO feature evidence for an exact R2a run.

This is an observer-only process.  It never reads ground truth or CLIP text
features and cannot write BoxFusion predictions.  Validation-time output is
diagnostic only; any score calibration belongs on a frozen ScanNet-train
split.
"""

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

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from boxfusion.tr3d_r2_cache import (  # noqa: E402
    load_tr3d_r2_cache,
    tr3d_r2_cache_path,
)
from boxfusion.tr3d_r2_observer import TR3DR2ObserverConfig  # noqa: E402
from boxfusion.tr3d_r2_provenance import (  # noqa: E402
    canonical_json_sha256,
    code_artifact_tree_sha256,
    frame_artifact_tree,
    load_prefix_manifest,
    load_resolved_poses,
    sha256_file,
)
from boxfusion.tr3d_r2b_cache import (  # noqa: E402
    load_tr3d_r2b_cache,
    make_tr3d_r2b_cache,
    tr3d_r2b_cache_path,
    write_tr3d_r2b_cache,
)
from boxfusion.tr3d_r2b_dino import (  # noqa: E402
    BOXER_DINO_MODEL,
    BoxerDINOv3Config,
    BoxerDINOv3DenseEncoder,
)
from boxfusion.tr3d_r2b_observer import (  # noqa: E402
    TR3DR2BFrameBundle,
    observe_tr3d_r2b_scene,
)
from boxfusion.tr3d_residual_cache import (  # noqa: E402
    tr3d_residual_cache_path,
)
from tools.run_tr3d_r2_observer import (  # noqa: E402
    REPORT_SCHEMA as R2A_REPORT_SCHEMA,
    _code_hash as current_r2a_code_hash,
    _load_bound_parent,
)
from tools.tr3d_data import discover_frame_bundle, read_scene_list  # noqa: E402


REPORT_SCHEMA = "boxfusion.tr3d_r2b_feature_observer_export.v1"
FEATURE_CONFIG_SCHEMA = "boxfusion.tr3d_r2b_dinov3_feature_config.v1"
RESOLVED_POSE_SOURCE = "scannet_g0_resolved_pose_v1"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-cache-root", type=Path, required=True)
    parser.add_argument("--r2a-cache-root", type=Path, required=True)
    parser.add_argument("--r2a-export-report", type=Path, required=True)
    parser.add_argument("--r2b-cache-root", type=Path, required=True)
    parser.add_argument("--prefix-manifest", type=Path, required=True)
    parser.add_argument("--frames-root", type=Path, required=True)
    parser.add_argument("--scene-list", type=Path, required=True)
    parser.add_argument(
        "--allow-scene-subset",
        action="store_true",
        help="allow an ordered smoke-test subset of the exact R2a scene set",
    )
    parser.add_argument("--prefix-id", default="p100")
    parser.add_argument("--expected-parent-checkpoint-sha256", required=True)
    parser.add_argument("--expected-parent-config-sha256", required=True)
    parser.add_argument("--official-boxer-root", type=Path, required=True)
    parser.add_argument("--expected-boxer-commit", required=True)
    parser.add_argument("--expected-dino-sha256", required=True)
    parser.add_argument("--input-height", type=int, default=960)
    parser.add_argument("--input-width", type=int, default=960)
    parser.add_argument(
        "--precision", choices=("float32", "bfloat16"), default="bfloat16"
    )
    parser.add_argument("--device", choices=("cuda", "cpu", "mps"), default="cuda")
    parser.add_argument("--min-support-points", type=int, default=2)
    parser.add_argument("--min-feature-cells", type=int, default=1)
    parser.add_argument("--feature-storage", choices=("float16", "float32"), default="float16")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--report", type=Path)
    return parser


def _quantiles(values: object) -> dict[str, float | None]:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if not len(array):
        return {"q10": None, "q50": None, "q90": None}
    q10, q50, q90 = np.quantile(array, (0.10, 0.50, 0.90))
    return {"q10": float(q10), "q50": float(q50), "q90": float(q90)}


def _feature_config(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "schema": FEATURE_CONFIG_SCHEMA,
        "observer_only": True,
        "mutation_enabled": False,
        "applied_count": 0,
        "ground_truth_access": False,
        "clip_access": False,
        "clip_semantics_unchanged": True,
        "feature_model": BOXER_DINO_MODEL,
        "official_boxer_root": str(args.official_boxer_root.resolve()),
        "official_boxer_commit": args.expected_boxer_commit,
        "feature_checkpoint_sha256": args.expected_dino_sha256,
        "preprocessing": "selective_boxer_cv2_linear_stretch_uint8_to_unit_v1",
        "input_shape": [args.input_height, args.input_width],
        "precision": args.precision,
        "device_type": args.device,
        "view_source": "exact_r2a_topk",
        "region_source": "r2a_metric_depth_support_points",
        "rgb_projection": "pose_at_extrinsic_color_intrinsic_color",
        "pooling": "unique_dense_cells_equal_weight_then_l2",
        "minimum_support_points": args.min_support_points,
        "minimum_feature_cells": args.min_feature_cells,
        "feature_storage": args.feature_storage,
        "dense_map_retention": "one_frame_streaming",
    }


def _feature_code_hash(args: argparse.Namespace) -> str:
    official_wrapper = (
        args.official_boxer_root.resolve()
        / "boxernet"
        / "dinov3_wrapper.py"
    )
    return code_artifact_tree_sha256(
        (
            _ROOT / "boxfusion" / "tr3d_r2_geometry.py",
            _ROOT / "boxfusion" / "tr3d_r2b_observer.py",
            _ROOT / "boxfusion" / "tr3d_r2b_dino.py",
            _ROOT / "boxfusion" / "tr3d_r2b_cache.py",
            official_wrapper,
            Path(__file__),
        )
    )


def _load_r2a_report(args: argparse.Namespace, scenes: list[str]) -> dict[str, Any]:
    path = args.r2a_export_report.resolve()
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("schema") != R2A_REPORT_SCHEMA:
        raise ValueError("unsupported R2a export report schema")
    if not report.get("observer_only") or report.get("mutation_enabled"):
        raise ValueError("R2a report violates observer-only contract")
    if int(report.get("applied_count", -1)) != 0:
        raise ValueError("R2a report applied_count must be zero")
    if report.get("ground_truth_access") or report.get("clip_enabled"):
        raise ValueError("R2a report accessed forbidden GT/CLIP inputs")
    expected_paths = {
        "parent_cache_root": args.parent_cache_root,
        "r2_cache_root": args.r2a_cache_root,
        "prefix_manifest": args.prefix_manifest,
        "frames_root": args.frames_root,
    }
    for name, expected in expected_paths.items():
        if Path(str(report.get(name, ""))).resolve() != expected.resolve():
            raise ValueError(f"R2a report {name} mismatch")
    if report.get("prefix_id") != args.prefix_id:
        raise ValueError("R2a report prefix id mismatch")
    report_scenes = [str(row.get("scene_id")) for row in report.get("scenes", [])]
    if int(report.get("scene_count", -1)) != len(report_scenes):
        raise ValueError("R2a report scene count mismatch")
    if args.allow_scene_subset:
        selected = [scene for scene in report_scenes if scene in set(scenes)]
        if selected != scenes or len(set(scenes)) != len(scenes):
            raise ValueError("requested smoke scenes are not an ordered R2a subset")
    elif report_scenes != scenes or Path(str(report.get("scene_list", ""))).resolve() != args.scene_list.resolve():
        raise ValueError("R2a report ordered scene set/list mismatch")
    if canonical_json_sha256(report.get("r2_config")) != report.get("r2_config_sha256"):
        raise ValueError("R2a report config hash mismatch")
    if current_r2a_code_hash() != report.get("r2_code_sha256"):
        raise ValueError("current R2a code differs from the parent export")
    return report


def _write_create_only(path: Path, encoded: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as handle:
            temporary_name = handle.name
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary_name, path)
        path.chmod(0o444)
    except FileExistsError as error:
        raise FileExistsError(f"immutable R2b report exists: {path}") from error
    finally:
        if temporary_name is not None:
            try:
                os.unlink(temporary_name)
            except FileNotFoundError:
                pass


def export(args: argparse.Namespace) -> dict[str, Any]:
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("shard index must lie in [0,num_shards)")
    all_scenes = read_scene_list(args.scene_list.resolve())
    r2a_report = _load_r2a_report(args, all_scenes)
    scenes = [
        scene for index, scene in enumerate(all_scenes)
        if index % args.num_shards == args.shard_index
    ]
    manifest = load_prefix_manifest(
        args.prefix_manifest.resolve(), prefix_id=args.prefix_id
    )
    if any(scene not in manifest for scene in scenes):
        raise ValueError("prefix manifest is missing a requested R2b scene")

    feature_payload = _feature_config(args)
    feature_config_sha = canonical_json_sha256(feature_payload)
    feature_code_sha = _feature_code_hash(args)
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
    # Verify bytes before touching any scene.  Model loading remains lazy, so
    # a fully resumed shard does not consume GPU memory.
    checkpoint_sha = encoder.verified_checkpoint_sha256
    if checkpoint_sha != args.expected_dino_sha256:
        raise RuntimeError("verified DINO hash changed unexpectedly")

    parent_r2_config_sha = str(r2a_report["r2_config_sha256"])
    parent_r2_code_sha = str(r2a_report["r2_code_sha256"])
    r2config = r2a_report["r2_config"]
    depth_config = TR3DR2ObserverConfig(
        image_shape=tuple(r2config["image_shape"]),
        pose_source=str(r2config["pose_source"]),
        top_k=int(r2config["top_k"]),
        pixel_stride=int(r2config["pixel_stride"]),
        depth_scale=float(r2config["depth_scale"]),
        margin=float(r2config["margin_m"]),
        min_depth=float(r2config["min_depth_m"]),
        max_depth=float(r2config["max_depth_m"]),
        near_clip=float(r2config["near_clip_m"]),
    )
    storage_dtype = np.float16 if args.feature_storage == "float16" else np.float32
    scene_reports: list[dict[str, Any]] = []
    total_wall = total_feature = total_geometry = 0.0
    total_proposals = total_feature_views = total_pairs = 0
    resumed_count = 0

    for position, scene_id in enumerate(scenes, start=1):
        scene_started = time.perf_counter()
        row = manifest[scene_id]
        used_ids = [int(value) for value in row["used_frame_ids"]]
        manifest_sha = canonical_json_sha256(row)
        bundle = discover_frame_bundle(args.frames_root.resolve(), scene_id)
        frame_tree_sha, _ = frame_artifact_tree(row, bundle)
        parent_path = tr3d_residual_cache_path(
            args.parent_cache_root.resolve(), scene_id, args.prefix_id
        )
        parent = _load_bound_parent(
            parent_path,
            row,
            args.prefix_manifest.resolve(),
            expected_scene_id=scene_id,
            expected_prefix_id=args.prefix_id,
            expected_checkpoint_sha256=args.expected_parent_checkpoint_sha256,
            expected_config_sha256=args.expected_parent_config_sha256,
        )
        r2a_path = tr3d_r2_cache_path(
            args.r2a_cache_root.resolve(), scene_id, args.prefix_id
        )
        r2a = load_tr3d_r2_cache(
            r2a_path,
            parent_cache_path=parent_path,
            expected_prefix_manifest_row_sha256=manifest_sha,
            expected_frame_artifact_tree_sha256=frame_tree_sha,
            expected_r2_config_sha256=parent_r2_config_sha,
            expected_r2_code_sha256=parent_r2_code_sha,
            expected_scene_id=scene_id,
            expected_prefix_id=args.prefix_id,
            expected_prefix_fraction=float(row["fraction"]),
            expected_allowed_frame_ids=used_ids,
        )
        target = tr3d_r2b_cache_path(
            args.r2b_cache_root.resolve(), scene_id, args.prefix_id
        )
        observation = None
        if target.exists():
            if not args.resume:
                raise FileExistsError(f"immutable R2b cache exists: {target}")
            sidecar = load_tr3d_r2b_cache(
                target,
                parent_r2a_cache_path=r2a_path,
                parent_tr3d_cache_path=parent_path,
                expected_parent_prefix_manifest_row_sha256=manifest_sha,
                expected_parent_frame_artifact_tree_sha256=frame_tree_sha,
                expected_parent_r2_config_sha256=parent_r2_config_sha,
                expected_parent_r2_code_sha256=parent_r2_code_sha,
                expected_feature_checkpoint_sha256=checkpoint_sha,
                expected_feature_config_sha256=feature_config_sha,
                expected_feature_code_sha256=feature_code_sha,
                expected_scene_id=scene_id,
                expected_prefix_id=args.prefix_id,
                expected_prefix_fraction=float(row["fraction"]),
            )
            resumed_count += 1
        else:
            resolved_poses = load_resolved_poses(row, bundle)
            frame_bundle = TR3DR2BFrameBundle(
                scene_id=scene_id,
                pose_source=RESOLVED_POSE_SOURCE,
                color={frame_id: bundle.color[frame_id] for frame_id in used_ids},
                depth={frame_id: bundle.depth[frame_id] for frame_id in used_ids},
                pose=resolved_poses,
                intrinsic_depth=bundle.intrinsic_depth,
                intrinsic_color=bundle.intrinsic_color,
                extrinsic_depth=bundle.extrinsic_depth,
                extrinsic_color=bundle.extrinsic_color,
            )
            parent_index = {
                int(value): index
                for index, value in enumerate(parent.proposal_ids)
            }
            try:
                parent_rows = np.asarray(
                    [parent_index[int(value)] for value in r2a.proposal_ids],
                    dtype=np.int64,
                )
            except KeyError as error:  # R2a loader should already prevent it.
                raise RuntimeError("R2a proposal is absent from TR3D parent") from error
            observation = observe_tr3d_r2b_scene(
                boxes_world=parent.boxes_world[parent_rows],
                proposal_ids=r2a.proposal_ids,
                topk_frame_ids=r2a.topk_frame_ids,
                topk_view_valid=r2a.topk_view_valid,
                frame_bundle=frame_bundle,
                depth_config=depth_config,
                encode_rgb=encoder,
                min_support_points=args.min_support_points,
                min_feature_cells=args.min_feature_cells,
            )
            if not np.array_equal(observation.proposal_ids, r2a.proposal_ids):
                raise RuntimeError("R2b observer changed proposal row identity")
            sidecar = make_tr3d_r2b_cache(
                parent_r2a_cache_path=r2a_path,
                parent_tr3d_cache_path=parent_path,
                parent_prefix_manifest_row_sha256=manifest_sha,
                parent_frame_artifact_tree_sha256=frame_tree_sha,
                parent_r2_config_sha256=parent_r2_config_sha,
                parent_r2_code_sha256=parent_r2_code_sha,
                feature_checkpoint_sha256=checkpoint_sha,
                feature_config_sha256=feature_config_sha,
                feature_code_sha256=feature_code_sha,
                per_view_feature_valid=observation.feature_view_valid,
                per_view_feature_count=observation.per_view_feature_count,
                per_view_feature_vector=observation.per_view_features.astype(
                    storage_dtype, copy=False
                ),
                runtime_s=observation.total_runtime_s,
            )
            write_tr3d_r2b_cache(
                target,
                sidecar,
                parent_r2a_cache_path=r2a_path,
                parent_tr3d_cache_path=parent_path,
            )

        scene_wall = time.perf_counter() - scene_started
        valid_pairs = sidecar.pairwise_cosine_count > 0
        feature_views = int(sidecar.aggregate_view_count.sum())
        pair_count = int(sidecar.pairwise_cosine_count.sum())
        scene_report = {
            "scene_id": scene_id,
            "proposal_count": sidecar.proposal_count,
            "feature_view_count": feature_views,
            "proposal_with_feature_count": int(np.count_nonzero(sidecar.aggregate_view_count > 0)),
            "proposal_with_multiview_feature_count": int(np.count_nonzero(valid_pairs)),
            "pairwise_feature_pair_count": pair_count,
            "pairwise_cosine_mean": _quantiles(sidecar.pairwise_cosine_mean[valid_pairs]),
            "feature_cell_count": _quantiles(sidecar.per_view_feature_count[sidecar.per_view_feature_valid]),
            "unique_encoded_frame_count": (
                int(len(observation.encoded_frame_ids)) if observation is not None else None
            ),
            "observer_feature_runtime_s": (
                float(observation.feature_runtime_s) if observation is not None else None
            ),
            "observer_geometry_runtime_s": (
                float(observation.geometry_runtime_s) if observation is not None else None
            ),
            "observer_total_runtime_s": float(sidecar.runtime_s),
            "end_to_end_scene_wall_s": float(scene_wall),
            "non_observer_wall_s": float(max(0.0, scene_wall - sidecar.runtime_s)),
            "resumed": observation is None,
            "r2b_sidecar": str(target),
            "r2b_sidecar_sha256": sha256_file(target),
        }
        scene_reports.append(scene_report)
        total_wall += scene_wall
        total_proposals += sidecar.proposal_count
        total_feature_views += feature_views
        total_pairs += pair_count
        if observation is not None:
            total_feature += observation.feature_runtime_s
            total_geometry += observation.geometry_runtime_s
        print(
            f"[{position}/{len(scenes)}] {scene_id}: proposals={sidecar.proposal_count}, "
            f"feature_views={feature_views}, multiview={scene_report['proposal_with_multiview_feature_count']}, "
            f"wall={scene_wall:.3f}s",
            flush=True,
        )

    return {
        "schema": REPORT_SCHEMA,
        "observer_only": True,
        "mutation_enabled": False,
        "applied_count": 0,
        "ground_truth_access": False,
        "clip_access": False,
        "clip_semantics_unchanged": True,
        "parent_cache_root": str(args.parent_cache_root.resolve()),
        "r2a_cache_root": str(args.r2a_cache_root.resolve()),
        "r2a_export_report": str(args.r2a_export_report.resolve()),
        "r2b_cache_root": str(args.r2b_cache_root.resolve()),
        "prefix_manifest": str(args.prefix_manifest.resolve()),
        "frames_root": str(args.frames_root.resolve()),
        "scene_list": str(args.scene_list.resolve()),
        "prefix_id": args.prefix_id,
        "input_hashes": {
            "r2a_export_report_sha256": sha256_file(args.r2a_export_report.resolve()),
            "prefix_manifest_sha256": sha256_file(args.prefix_manifest.resolve()),
            "scene_list_sha256": sha256_file(args.scene_list.resolve()),
        },
        "parent_r2_config_sha256": parent_r2_config_sha,
        "parent_r2_code_sha256": parent_r2_code_sha,
        "feature_config": feature_payload,
        "feature_config_sha256": feature_config_sha,
        "feature_code_sha256": feature_code_sha,
        "feature_checkpoint_sha256": checkpoint_sha,
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
        "scene_count": len(scene_reports),
        "proposal_count": total_proposals,
        "feature_view_count": total_feature_views,
        "pairwise_feature_pair_count": total_pairs,
        "resumed_scene_count": resumed_count,
        "summed_feature_compute_s": float(total_feature),
        "summed_geometry_compute_s": float(total_geometry),
        "summed_end_to_end_scene_wall_s": float(total_wall),
        "runtime_note": (
            "offline observer measurement; feature maps are streamed one frame "
            "at a time. Online cost requires reuse of Selective Boxer dino0 "
            "and must be measured separately before activation"
        ),
        "scenes": scene_reports,
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = export(args)
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.report is not None:
        _write_create_only(args.report.resolve(), encoded)
    print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
