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
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.tr3d_data import (PREFIX_SCHEMA, backproject_rgbd,
                             build_prefix_info_row, discover_frame_bundle,
                             dump_json_atomic, dump_pickle_atomic,
                             filter_prefix_instances, foreground_metainfo,
                             index_info_rows, load_info, load_matrix,
                             prefix_tag, read_points_bin, read_scene_list,
                             voxel_downsample_first, write_jsonl)


BOXFUSION_CLOCK_POLICY = "g0_post_frame_tail_guard_v1"
BOXFUSION_POSE_POLICY = "previous_valid_inf_only_v1"
BOXFUSION_SOURCE_TIMESTAMP = "zero_based_scannet_dataset_index"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def boxfusion_prefix_schedule(
        frame_ids: Sequence[int],
        *,
        fractions: Sequence[float] = (0.25, 0.5, 0.75, 1.0),
        frame_stride: int = 25,
) -> List[dict]:
    """Reproduce the frozen-G0 keyframe clock exactly.

    ``ScannetDataset`` emits a zero-based ``meta.timestamp`` for each sorted
    source frame.  G0 selects timestamps divisible by ``gap`` and terminates
    *after* processing the first frame for which ``count + gap`` would cross
    the final dataset index.  Consequently the protected tail is never added
    as an extra keyframe.  This differs deliberately from the legacy generic
    prefix scheduler, which always appended the final source frame.
    """
    if frame_stride <= 0:
        raise ValueError("frame_stride must be positive")
    ordered = list(sorted(frame_ids))
    if not ordered:
        raise ValueError("frame_ids is empty")
    if len(set(ordered)) != len(ordered):
        raise ValueError("frame_ids contains duplicates")

    # demo.py increments count after processing a frame and then exits when
    # (count + gap) > len(dataset) - 1.  At least frame zero is processed even
    # for a sequence no longer than the guard interval.
    processed_frame_count = max(1, len(ordered) - frame_stride)
    source_timestamps = list(range(0, processed_frame_count, frame_stride))
    sampled_frame_ids = [ordered[index] for index in source_timestamps]

    result: List[dict] = []
    previous_count = 0
    for value in fractions:
        fraction_value = float(value)
        if not (0 < fraction_value <= 1):
            raise ValueError("prefix fractions must be in (0, 1]")
        count = max(
            1,
            min(
                len(sampled_frame_ids),
                math.ceil(len(sampled_frame_ids) * fraction_value),
            ),
        )
        if count <= previous_count:
            continue
        result.append({
            "fraction": fraction_value,
            "sampled_frame_count": count,
            "frame_ids": sampled_frame_ids[:count],
            "source_timestamps": source_timestamps[:count],
            "last_frame_id": sampled_frame_ids[count - 1],
            "last_source_timestamp": source_timestamps[count - 1],
            "source_frame_count": len(ordered),
            "processed_frame_count": processed_frame_count,
        })
        previous_count = count
    if result[-1]["sampled_frame_count"] != len(sampled_frame_ids):
        result.append({
            "fraction": 1.0,
            "sampled_frame_count": len(sampled_frame_ids),
            "frame_ids": sampled_frame_ids,
            "source_timestamps": source_timestamps,
            "last_frame_id": sampled_frame_ids[-1],
            "last_source_timestamp": source_timestamps[-1],
            "source_frame_count": len(ordered),
            "processed_frame_count": processed_frame_count,
        })
    return result


def _aligned_frame_ids(bundle) -> List[int]:
    """Fail closed unless RGB, depth, and pose have the G0 positional map."""
    color_ids = sorted(bundle.color)
    depth_ids = sorted(bundle.depth)
    pose_ids = sorted(bundle.pose)
    if color_ids != depth_ids or color_ids != pose_ids:
        raise ValueError(
            f"{bundle.scene_id}: RGB/depth/pose frame ids are not aligned; "
            "an exact ScannetDataset timestamp map cannot be constructed"
        )
    if not color_ids:
        raise ValueError(f"{bundle.scene_id}: no aligned RGB-D-pose frames")
    return color_ids


def _resolve_boxfusion_poses(
        bundle,
        frame_ids: Sequence[int],
) -> Tuple[Dict[int, np.ndarray], Dict[int, dict]]:
    """Resolve poses using the exact frozen-G0 ``load_poses`` policy.

    G0 walks every pose in source order before inference.  A pose containing
    infinity reuses the immediately preceding valid pose; importantly, that
    predecessor may be a non-keyframe.  NaN and singular matrices are left
    untouched here because the frozen loader only tests ``np.isinf``.
    """
    resolved: Dict[int, np.ndarray] = {}
    provenance: Dict[int, dict] = {}
    last_valid_pose: Optional[np.ndarray] = None
    last_valid_timestamp: Optional[int] = None
    last_valid_frame_id: Optional[int] = None

    for source_timestamp, frame_id in enumerate(frame_ids):
        input_path = Path(bundle.pose[frame_id])
        input_pose = load_matrix(input_path)
        carried_forward = bool(np.isinf(input_pose).any())
        if not carried_forward:
            last_valid_pose = input_pose
            last_valid_timestamp = source_timestamp
            last_valid_frame_id = frame_id
        elif last_valid_pose is None:
            raise ValueError(
                f"{bundle.scene_id}: first pose at source timestamp "
                f"{source_timestamp} contains infinity and G0 has no "
                "previous valid pose to carry forward"
            )

        assert last_valid_pose is not None
        assert last_valid_timestamp is not None
        assert last_valid_frame_id is not None
        resolved[source_timestamp] = last_valid_pose.copy()
        resolved_path = Path(bundle.pose[last_valid_frame_id])
        provenance[source_timestamp] = {
            "source_timestamp": source_timestamp,
            "frame_id": frame_id,
            "input_pose_frame_id": frame_id,
            "input_pose_path": str(input_path.resolve()),
            "input_pose_sha256": _sha256_file(input_path),
            "pose_resolution": (
                "carry_forward" if carried_forward else "direct"
            ),
            "resolved_pose_source_timestamp": last_valid_timestamp,
            "resolved_pose_frame_id": last_valid_frame_id,
            "resolved_pose_path": str(resolved_path.resolve()),
            "resolved_pose_sha256": _sha256_file(resolved_path),
        }
    return resolved, provenance


def export_boxfusion_scene_prefixes(
        *,
        scene_id: str,
        frame_root: Path,
        source_row: Mapping,
        output_root: Path,
        fractions: Sequence[float] = (0.25, 0.5, 0.75, 1.0),
        frame_stride: int = 25,
        pixel_stride: int = 4,
        voxel_size: float = 0.01,
        depth_scale: float = 1000.0,
        min_depth: float = 0.1,
        max_depth: float = 6.0,
        min_observed_points: int = 20,
        full_points_path: Optional[Path] = None,
        min_visibility_fraction: float = 0.0,
        manifest_only: bool = False,
) -> Tuple[List[dict], List[dict]]:
    """Export trajectory prefixes with the frozen-G0 temporal contract."""
    bundle = discover_frame_bundle(frame_root, scene_id)
    frame_ids = _aligned_frame_ids(bundle)
    schedule = boxfusion_prefix_schedule(
        frame_ids, fractions=fractions, frame_stride=frame_stride)
    resolved_poses, pose_provenance = _resolve_boxfusion_poses(
        bundle, frame_ids)

    intrinsic_depth = load_matrix(bundle.intrinsic_depth)
    intrinsic_color = load_matrix(bundle.intrinsic_color)
    extrinsic_depth = load_matrix(bundle.extrinsic_depth)
    extrinsic_color = load_matrix(bundle.extrinsic_color)
    axis_align = np.asarray(
        source_row.get("axis_align_matrix", np.eye(4)), dtype=np.float64)
    if axis_align.shape != (4, 4):
        raise ValueError(f"{scene_id}: invalid axis_align_matrix")

    full_points = None
    if min_visibility_fraction > 0 and not manifest_only:
        if full_points_path is None:
            raise ValueError(
                f"{scene_id}: full points are required for visibility filtering")
        full_points = read_points_bin(full_points_path)

    exported_rows: List[dict] = []
    manifests: List[dict] = []
    frame_cache: Dict[int, np.ndarray] = {}
    source_root = frame_root.resolve()
    scene_frame_root = bundle.frame_root.resolve()
    for item in schedule:
        tag = prefix_tag(item["fraction"])
        relative_point_path = str(
            Path("prefixes") / scene_id / f"{scene_id}__{tag}.bin")
        absolute_point_path = output_root / "points" / relative_point_path
        selected_provenance = [
            pose_provenance[timestamp]
            for timestamp in item["source_timestamps"]
        ]
        metadata = {
            "schema": PREFIX_SCHEMA,
            "scene_id": scene_id,
            "tag": tag,
            "fraction": item["fraction"],
            "frame_stride": frame_stride,
            "tail_guard_frames": frame_stride,
            "clock_policy": BOXFUSION_CLOCK_POLICY,
            "pose_policy": BOXFUSION_POSE_POLICY,
            "source_timestamp_semantics": BOXFUSION_SOURCE_TIMESTAMP,
            "source_frames_root": str(source_root),
            "source_scene_frame_root": str(scene_frame_root),
            "source_frame_count": item["source_frame_count"],
            "processed_frame_count": item["processed_frame_count"],
            "pixel_stride": pixel_stride,
            "voxel_size": voxel_size,
            "depth_scale": depth_scale,
            "coordinate_frame": "world_unaligned",
            "network_frame_after_pipeline": "scannet_axis_aligned",
            "axis_align_matrix": axis_align.tolist(),
            "sampled_frame_count": item["sampled_frame_count"],
            "first_frame_id": item["frame_ids"][0],
            "last_frame_id": item["last_frame_id"],
            "frame_ids": item["frame_ids"],
            "source_timestamps": item["source_timestamps"],
            "last_source_timestamp": item["last_source_timestamp"],
            # Carry-forward makes every scheduled frame usable.  Keep both the
            # legacy field and the exact timestamp mapping in every manifest,
            # including manifest-only schedules.
            "used_frame_ids": item["frame_ids"],
            "used_source_timestamps": item["source_timestamps"],
            "pose_provenance": selected_provenance,
            "point_path": str(absolute_point_path),
            "min_observed_points": min_observed_points,
            "min_visibility_fraction": min_visibility_fraction,
        }
        if manifest_only:
            metadata.update({
                "status": "planned",
                "point_count": None,
                "source_instance_count": len(source_row.get("instances", [])),
                "kept_instance_count": None,
            })
            manifests.append(metadata)
            continue

        parts: List[np.ndarray] = []
        for source_timestamp, frame_id in zip(
                item["source_timestamps"], item["frame_ids"]):
            if source_timestamp not in frame_cache:
                frame_cache[source_timestamp] = backproject_rgbd(
                    depth_path=bundle.depth[frame_id],
                    color_path=bundle.color[frame_id],
                    pose=resolved_poses[source_timestamp],
                    intrinsic_depth=intrinsic_depth,
                    intrinsic_color=intrinsic_color,
                    extrinsic_depth=extrinsic_depth,
                    extrinsic_color=extrinsic_color,
                    depth_scale=depth_scale,
                    min_depth=min_depth,
                    max_depth=max_depth,
                    pixel_stride=pixel_stride,
                )
            if len(frame_cache[source_timestamp]):
                parts.append(frame_cache[source_timestamp])
        if not parts:
            raise ValueError(f"{scene_id}/{tag}: every selected depth is empty")
        prefix_points = voxel_downsample_first(
            np.concatenate(parts, axis=0), voxel_size)
        absolute_point_path.parent.mkdir(parents=True, exist_ok=True)
        prefix_points.astype(np.float32).tofile(absolute_point_path)

        kept, support = filter_prefix_instances(
            source_row.get("instances", []),
            prefix_points,
            axis_align,
            min_observed_points=min_observed_points,
            full_world_points=full_points,
            min_visibility_fraction=min_visibility_fraction,
        )
        metadata.update({
            "status": "exported",
            "point_count": int(len(prefix_points)),
            "source_instance_count": len(source_row.get("instances", [])),
            "kept_instance_count": len(kept),
            "instance_support": support,
        })
        exported_rows.append(build_prefix_info_row(
            source_row,
            relative_point_path=relative_point_path,
            instances=kept,
            prefix_metadata=metadata,
        ))
        manifests.append(metadata)
    return exported_rows, manifests


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
            rows, manifests = export_boxfusion_scene_prefixes(
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
        "clock_policy": BOXFUSION_CLOCK_POLICY,
        "pose_policy": BOXFUSION_POSE_POLICY,
        "source_timestamp_semantics": BOXFUSION_SOURCE_TIMESTAMP,
        "tail_guard_frames": args.frame_stride,
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
