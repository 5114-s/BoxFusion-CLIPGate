#!/usr/bin/env python3
"""Build immutable R2a real-depth/free-space evidence sidecars.

This command is observer-only.  It reads a frozen parent TR3D cache and the
strict frozen-G0 trajectory manifest, then writes a separate evidence NPZ.  It
does not read GT, CLIP, or BoxFusion predictions and has no active-output path.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys
import tempfile
from typing import Any, Sequence

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from boxfusion.tr3d_r2_cache import (  # noqa: E402
    depth_evidence_fractions,
    load_tr3d_r2_cache,
    make_tr3d_r2_cache,
    tr3d_r2_cache_path,
    write_tr3d_r2_cache,
)
from boxfusion.tr3d_r2_observer import (  # noqa: E402
    TR3DR2FrameBundle,
    TR3DR2ObserverConfig,
    observe_tr3d_r2_scene,
)
from boxfusion.tr3d_r2_provenance import (  # noqa: E402
    canonical_json_sha256,
    code_artifact_tree_sha256,
    frame_artifact_tree,
    load_prefix_manifest,
    load_resolved_poses,
    sha256_file,
)
from boxfusion.tr3d_residual_cache import (  # noqa: E402
    load_tr3d_residual_cache,
    tr3d_residual_cache_path,
)
from tools.tr3d_data import (  # noqa: E402
    discover_frame_bundle,
    read_scene_list,
)


REPORT_SCHEMA = "boxfusion.tr3d_r2a_observer_export.v1"
R2_CONFIG_SCHEMA = "boxfusion.tr3d_r2a_config.v1"
RESOLVED_POSE_SOURCE = "scannet_g0_resolved_pose_v1"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-cache-root", type=Path, required=True)
    parser.add_argument("--r2-cache-root", type=Path, required=True)
    parser.add_argument("--prefix-manifest", type=Path, required=True)
    parser.add_argument("--frames-root", type=Path, required=True)
    parser.add_argument("--scene-list", type=Path, required=True)
    parser.add_argument("--prefix-id", default="p100")
    parser.add_argument("--expected-parent-checkpoint-sha256")
    parser.add_argument("--expected-parent-config-sha256")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--pixel-stride", type=int, default=4)
    parser.add_argument("--depth-scale", type=float, default=1000.0)
    parser.add_argument("--margin", type=float, default=0.05)
    parser.add_argument("--min-depth", type=float, default=0.10)
    parser.add_argument("--max-depth", type=float, default=8.0)
    parser.add_argument("--near-clip", type=float, default=1e-3)
    parser.add_argument("--image-height", type=int, default=480)
    parser.add_argument("--image-width", type=int, default=640)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--report", type=Path)
    return parser


def _quantiles(values: np.ndarray) -> dict[str, float | None]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if not len(finite):
        return {"q10": None, "q50": None, "q90": None}
    q10, q50, q90 = np.quantile(finite, (0.10, 0.50, 0.90))
    return {"q10": float(q10), "q50": float(q50), "q90": float(q90)}


def _r2_config(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "schema": R2_CONFIG_SCHEMA,
        "observer_only": True,
        "mutation_enabled": False,
        "applied_count": 0,
        "prefix_id": args.prefix_id,
        "pose_source": RESOLVED_POSE_SOURCE,
        "top_k": args.top_k,
        "pixel_stride": args.pixel_stride,
        "depth_scale": args.depth_scale,
        "margin_m": args.margin,
        "min_depth_m": args.min_depth,
        "max_depth_m": args.max_depth,
        "near_clip_m": args.near_clip,
        "image_shape": [args.image_height, args.image_width],
        "view_ranking": "projected_area_desc_frame_id_asc_stable",
        "depth_classes": [
            "support",
            "occluded",
            "free_space",
            "invalid",
        ],
        "feature_consistency_enabled": False,
        "clip_enabled": False,
        "ground_truth_access": False,
    }


def _code_hash() -> str:
    return code_artifact_tree_sha256(
        (
            _ROOT / "boxfusion" / "tr3d_r2_geometry.py",
            _ROOT / "boxfusion" / "tr3d_r2_observer.py",
            _ROOT / "boxfusion" / "tr3d_r2_cache.py",
            _ROOT / "boxfusion" / "tr3d_r2_provenance.py",
            Path(__file__),
        )
    )


def _source_root_contract(row: dict[str, Any], frames_root: Path) -> None:
    recorded = row.get("source_frames_root")
    if not isinstance(recorded, str):
        raise ValueError("prefix manifest lacks source_frames_root")
    if Path(recorded).resolve() != frames_root.resolve():
        raise ValueError(
            "R2a frames root differs from the exported frozen-G0 root"
        )


def _prefix_point_contract(
    row: dict[str, Any], manifest_path: Path
) -> tuple[Path, int, str]:
    """Bind one manifest row to the exact exported XYZRGB point bytes."""

    raw_path = row.get("point_path")
    if not isinstance(raw_path, str) or not raw_path:
        raise ValueError("prefix manifest lacks point_path")
    point_path = Path(raw_path)
    if not point_path.is_absolute():
        point_path = manifest_path.resolve().parent / point_path
    point_path = point_path.resolve()
    if not point_path.is_file():
        raise FileNotFoundError(point_path)
    point_count = row.get("point_count")
    if (
        isinstance(point_count, bool)
        or not isinstance(point_count, int)
        or point_count < 0
    ):
        raise ValueError("prefix manifest point_count must be nonnegative")
    expected_bytes = point_count * 6 * np.dtype(np.float32).itemsize
    if point_path.stat().st_size != expected_bytes:
        raise ValueError(
            "prefix point file size disagrees with point_count XYZRGB layout"
        )
    return point_path, point_count, sha256_file(point_path)


def _load_bound_parent(
    parent_path: Path,
    row: dict[str, Any],
    manifest_path: Path,
    *,
    expected_scene_id: str,
    expected_prefix_id: str,
    expected_checkpoint_sha256: str | None = None,
    expected_config_sha256: str | None = None,
):
    """Load a parent only when it describes this exact prefix point file."""

    _, point_count, point_sha256 = _prefix_point_contract(
        row, manifest_path
    )
    parent = load_tr3d_residual_cache(
        parent_path,
        expected_scene_id=expected_scene_id,
        expected_prefix_id=expected_prefix_id,
        expected_checkpoint_sha256=expected_checkpoint_sha256,
        expected_config_sha256=expected_config_sha256,
        expected_source_scene_sha256=point_sha256,
    )
    if parent.num_input_points != point_count:
        raise ValueError(
            "parent num_input_points disagrees with prefix point_count"
        )
    fraction = row.get("fraction")
    if (
        isinstance(fraction, bool)
        or not isinstance(fraction, (int, float))
        or not math.isclose(
            parent.prefix_fraction,
            float(fraction),
            rel_tol=0.0,
            abs_tol=0.0,
        )
    ):
        raise ValueError(
            "parent prefix fraction disagrees with prefix manifest"
        )
    return parent


def export(args: argparse.Namespace) -> dict[str, Any]:
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("shard index must lie in [0,num_shards)")
    scenes = read_scene_list(args.scene_list.resolve())
    scenes = [
        scene for index, scene in enumerate(scenes)
        if index % args.num_shards == args.shard_index
    ]
    manifest = load_prefix_manifest(
        args.prefix_manifest.resolve(), prefix_id=args.prefix_id
    )
    missing = [scene for scene in scenes if scene not in manifest]
    if missing:
        raise ValueError(
            "prefix manifest is missing requested scenes: "
            + ", ".join(missing[:8])
        )

    config_payload = _r2_config(args)
    config = TR3DR2ObserverConfig(
        image_shape=(args.image_height, args.image_width),
        pose_source=RESOLVED_POSE_SOURCE,
        top_k=args.top_k,
        pixel_stride=args.pixel_stride,
        depth_scale=args.depth_scale,
        margin=args.margin,
        min_depth=args.min_depth,
        max_depth=args.max_depth,
        near_clip=args.near_clip,
    )
    config_sha = canonical_json_sha256(config_payload)
    code_sha = _code_hash()
    scene_reports: list[dict[str, Any]] = []
    total_runtime = 0.0
    total_proposals = 0
    resumed = 0

    for position, scene_id in enumerate(scenes, start=1):
        row = manifest[scene_id]
        _source_root_contract(row, args.frames_root)
        parent_path = tr3d_residual_cache_path(
            args.parent_cache_root.resolve(), scene_id, args.prefix_id
        )
        parent = _load_bound_parent(
            parent_path,
            row,
            args.prefix_manifest.resolve(),
            expected_scene_id=scene_id,
            expected_prefix_id=args.prefix_id,
            expected_checkpoint_sha256=(
                args.expected_parent_checkpoint_sha256
            ),
            expected_config_sha256=args.expected_parent_config_sha256,
        )
        bundle = discover_frame_bundle(args.frames_root.resolve(), scene_id)
        manifest_sha = canonical_json_sha256(row)
        frame_tree_sha, _ = frame_artifact_tree(row, bundle)
        resolved_poses = load_resolved_poses(row, bundle)
        used_ids = [int(value) for value in row["used_frame_ids"]]
        frame_bundle = TR3DR2FrameBundle(
            scene_id=scene_id,
            pose_source=RESOLVED_POSE_SOURCE,
            depth={frame_id: bundle.depth[frame_id] for frame_id in used_ids},
            pose=resolved_poses,
            intrinsic_depth=bundle.intrinsic_depth,
            extrinsic_depth=bundle.extrinsic_depth,
        )
        observer_manifest = {
            "scene_id": scene_id,
            "used_frame_ids": used_ids,
            "pose_source": RESOLVED_POSE_SOURCE,
        }
        target = tr3d_r2_cache_path(
            args.r2_cache_root.resolve(), scene_id, args.prefix_id
        )
        if target.exists():
            if not args.resume:
                raise FileExistsError(f"immutable R2a cache exists: {target}")
            cached = load_tr3d_r2_cache(
                target,
                parent_cache_path=parent_path,
                expected_prefix_manifest_row_sha256=manifest_sha,
                expected_frame_artifact_tree_sha256=frame_tree_sha,
                expected_r2_config_sha256=config_sha,
                expected_r2_code_sha256=code_sha,
                expected_scene_id=scene_id,
                expected_prefix_id=args.prefix_id,
                expected_prefix_fraction=float(row["fraction"]),
                expected_allowed_frame_ids=used_ids,
            )
            observation = None
            sidecar = cached
            resumed += 1
        else:
            observation = observe_tr3d_r2_scene(
                boxes_world=parent.boxes_world,
                proposal_ids=parent.proposal_ids,
                prefix_manifest=observer_manifest,
                frame_bundle=frame_bundle,
                config=config,
            )
            per_view_evidence = depth_evidence_fractions(
                observation.per_view_depth_counts
            )
            aggregate_evidence = depth_evidence_fractions(
                observation.aggregate_depth_counts
            )
            sidecar = make_tr3d_r2_cache(
                parent_cache_path=parent_path,
                prefix_manifest_row_sha256=manifest_sha,
                frame_artifact_tree_sha256=frame_tree_sha,
                r2_config_sha256=config_sha,
                r2_code_sha256=code_sha,
                proposal_ids=observation.proposal_ids,
                lineage_ids=observation.proposal_ids,
                topk_frame_ids=observation.topk_frame_ids,
                topk_view_valid=observation.topk_view_valid,
                topk_projected_area_pixels=(
                    observation.topk_projected_area_pixels
                ),
                topk_projected_area_fraction=(
                    observation.topk_projected_area_fraction
                ),
                per_view_depth_evidence=per_view_evidence,
                per_view_depth_counts=observation.per_view_depth_counts,
                per_view_point_count=observation.per_view_point_count,
                aggregate_depth_evidence=aggregate_evidence,
                aggregate_depth_counts=observation.aggregate_depth_counts,
                aggregate_view_count=observation.aggregate_view_count,
                aggregate_point_count=observation.aggregate_point_count,
                runtime_s=observation.runtime_s,
                expected_allowed_frame_ids=used_ids,
            )
            write_tr3d_r2_cache(
                target,
                sidecar,
                parent_cache_path=parent_path,
                expected_allowed_frame_ids=used_ids,
            )

        visible = sidecar.aggregate_view_count > 0
        evidence = sidecar.aggregate_depth_evidence
        report = {
            "scene_id": scene_id,
            "proposal_count": sidecar.proposal_count,
            "visible_proposals": int(np.count_nonzero(visible)),
            "invisible_proposals": int(np.count_nonzero(~visible)),
            "selected_views": int(sidecar.aggregate_view_count.sum()),
            "sampled_pixels": int(sidecar.aggregate_point_count.sum()),
            "support_fraction": _quantiles(evidence[visible, 0]),
            "occluded_fraction": _quantiles(evidence[visible, 1]),
            "free_space_fraction": _quantiles(evidence[visible, 2]),
            "invalid_fraction": _quantiles(evidence[visible, 3]),
            "runtime_s": float(sidecar.runtime_s),
            "resumed": bool(observation is None),
            "sidecar": str(target),
        }
        scene_reports.append(report)
        total_runtime += float(sidecar.runtime_s)
        total_proposals += sidecar.proposal_count
        print(
            f"[{position}/{len(scenes)}] {scene_id}: "
            f"proposals={sidecar.proposal_count}, "
            f"visible={report['visible_proposals']}, "
            f"views={report['selected_views']}, "
            f"runtime={sidecar.runtime_s:.3f}s",
            flush=True,
        )

    return {
        "schema": REPORT_SCHEMA,
        "observer_only": True,
        "mutation_enabled": False,
        "applied_count": 0,
        "ground_truth_access": False,
        "clip_enabled": False,
        "feature_consistency_enabled": False,
        "parent_cache_root": str(args.parent_cache_root.resolve()),
        "r2_cache_root": str(args.r2_cache_root.resolve()),
        "prefix_manifest": str(args.prefix_manifest.resolve()),
        "frames_root": str(args.frames_root.resolve()),
        "scene_list": str(args.scene_list.resolve()),
        "prefix_id": args.prefix_id,
        "r2_config": config_payload,
        "r2_config_sha256": config_sha,
        "r2_code_sha256": code_sha,
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
        "scene_count": len(scene_reports),
        "proposal_count": total_proposals,
        "resumed_scene_count": resumed,
        "summed_scene_runtime_s": total_runtime,
        "scenes": scene_reports,
    }


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
        raise FileExistsError(
            f"immutable R2a export report exists: {path}"
        ) from error
    finally:
        if temporary_name is not None:
            try:
                os.unlink(temporary_name)
            except FileNotFoundError:
                pass


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = export(args)
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.report is not None:
        path = args.report.resolve()
        cache_root = args.r2_cache_root.resolve()
        if path == cache_root or cache_root in path.parents:
            raise ValueError("report must not be written inside immutable cache")
        _write_create_only(path, encoded)
    print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
