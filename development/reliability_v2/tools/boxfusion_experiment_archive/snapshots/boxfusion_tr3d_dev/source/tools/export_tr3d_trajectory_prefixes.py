#!/usr/bin/env python3
"""Export BoxFusion RGB-D trajectory prefixes for online-aligned TR3D training.

Prefix points remain in the original ScanNet world frame. The generated info
rows retain ``axis_align_matrix`` so the official TR3D ``GlobalAlignment``
transform is applied exactly once. Ground-truth boxes with insufficient
observed-prefix point support are removed; unseen full-scene boxes are never
silently retained as impossible positives.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.tr3d_data import (PREFIX_SCHEMA, dump_json_atomic,
                             dump_pickle_atomic, export_scene_prefixes,
                             foreground_metainfo, index_info_rows, load_info,
                             read_scene_list, write_jsonl)


def fraction(value: str) -> float:
    parsed = float(value)
    if not (0 < parsed <= 1):
        raise argparse.ArgumentTypeError("fraction must be in (0, 1]")
    return parsed


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    prepared_root = project_root / "data" / "tr3d_scannet"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--prepared-root", type=Path, default=prepared_root)
    parser.add_argument(
        "--frames-root",
        type=Path,
        default=Path("/extra/ZhaoX/scannet_data/scans.sens"),
        help="Accepts direct scene/{color,depth,pose,intrinsic} or nested "
             "scene/frames/{color,depth,pose,intrinsic} layouts.")
    parser.add_argument(
        "--scene-list",
        type=Path,
        default=prepared_root / "splits" / "trajectory_available_train.txt")
    parser.add_argument(
        "--source-info",
        type=Path,
        default=Path("/extra/ZhaoX/scannet_data/scannet_infos_train.pkl"))
    parser.add_argument(
        "--source-points",
        type=Path,
        default=Path("/extra/ZhaoX/scannet_data/points"))
    parser.add_argument(
        "--output-info-name",
        default="scannet_infos_prefix_train_foreground.pkl")
    parser.add_argument(
        "--manifest-name", default="trajectory_prefix_train.jsonl")
    parser.add_argument(
        "--fractions",
        type=fraction,
        nargs="+",
        default=(0.25, 0.5, 0.75, 1.0))
    parser.add_argument("--frame-stride", type=int, default=25)
    parser.add_argument("--pixel-stride", type=int, default=4)
    parser.add_argument("--voxel-size", type=float, default=0.01)
    parser.add_argument("--depth-scale", type=float, default=1000.0)
    parser.add_argument("--min-depth", type=float, default=0.1)
    parser.add_argument("--max-depth", type=float, default=6.0)
    parser.add_argument("--min-observed-points", type=int, default=20)
    parser.add_argument(
        "--min-visibility-fraction",
        type=float,
        default=0.0,
        help="Optional observed/full in-box point fraction threshold.")
    parser.add_argument(
        "--manifest-only",
        action="store_true",
        help="Write the deterministic schedule without decoding images.")
    parser.add_argument(
        "--skip-missing-scenes",
        action="store_true",
        help="Record missing frame scenes instead of failing the whole export.")
    parser.add_argument(
        "--max-scenes",
        type=int,
        default=None,
        help="Optional deterministic prefix of scene-list entries for smoke tests.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    scenes = read_scene_list(args.scene_list.resolve())
    if args.max_scenes is not None:
        if args.max_scenes < 1:
            raise ValueError("--max-scenes must be positive")
        scenes = scenes[:args.max_scenes]
    source_meta, source_rows = load_info(args.source_info.resolve())
    source_index = index_info_rows(source_rows)
    if not set(scenes) <= set(source_index):
        missing = sorted(set(scenes) - set(source_index))
        raise ValueError(
            "source info is missing scenes: " + ", ".join(missing[:10]))

    all_rows: List[dict] = []
    all_manifests: List[dict] = []
    errors: List[dict] = []
    for position, scene in enumerate(scenes, start=1):
        try:
            rows, manifests = export_scene_prefixes(
                scene_id=scene,
                frame_root=args.frames_root.resolve(),
                source_row=source_index[scene],
                output_root=args.prepared_root.resolve(),
                fractions=args.fractions,
                frame_stride=args.frame_stride,
                pixel_stride=args.pixel_stride,
                voxel_size=args.voxel_size,
                depth_scale=args.depth_scale,
                min_depth=args.min_depth,
                max_depth=args.max_depth,
                min_observed_points=args.min_observed_points,
                full_points_path=(
                    args.source_points.resolve() / f"{scene}.bin"),
                min_visibility_fraction=args.min_visibility_fraction,
                manifest_only=args.manifest_only,
            )
        except (FileNotFoundError, ValueError) as error:
            if not args.skip_missing_scenes:
                raise
            errors.append({"scene_id": scene, "error": str(error)})
            continue
        all_rows.extend(rows)
        all_manifests.extend(manifests)
        print(
            f"[{position}/{len(scenes)}] {scene}: "
            f"{len(manifests)} prefixes", flush=True)

    manifest_path = (
        args.prepared_root.resolve() / "manifests" / args.manifest_name)
    write_jsonl(manifest_path, all_manifests)
    if not args.manifest_only:
        info = {
            "metainfo": foreground_metainfo(source_meta),
            "data_list": all_rows,
        }
        info_path = (
            args.prepared_root.resolve()
            / "annotations" / args.output_info_name)
        dump_pickle_atomic(info_path, info)
    summary = {
        "schema": PREFIX_SCHEMA,
        "scene_list": str(args.scene_list.resolve()),
        "scene_count_requested": len(scenes),
        "scene_count_exported": len(
            {item["scene_id"] for item in all_manifests}),
        "prefix_count": len(all_manifests),
        "annotation_row_count": len(all_rows),
        "manifest_only": args.manifest_only,
        "coordinate_frame": "world_unaligned",
        "network_frame_after_pipeline": "scannet_axis_aligned",
        "visibility_rule": {
            "min_observed_points": args.min_observed_points,
            "min_visibility_fraction": args.min_visibility_fraction,
        },
        "errors": errors,
    }
    summary_path = manifest_path.with_suffix(".summary.json")
    dump_json_atomic(summary_path, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
