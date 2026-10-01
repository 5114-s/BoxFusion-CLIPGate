#!/usr/bin/env python3
"""Prepare leak-free ScanNet metadata for class-agnostic TR3D.

This command is metadata-only: source ScanNet arrays remain read-only and are
referenced through directory symlinks. It creates deterministic train,
calibration and audit subsets exclusively from the official ScanNet train
split, plus a separately named official-validation annotation file.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.tr3d_data import (DATASET_SCHEMA, build_foreground_info,
                             deterministic_partition, dump_json_atomic,
                             dump_pickle_atomic, ensure_directory_link,
                             index_info_rows, load_info, read_scene_list,
                             scene_id_from_info, sha256_lines, write_jsonl,
                             write_scene_list)


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path("/extra/ZhaoX/scannet_data"),
        help="Existing MMDetection3D-formatted ScanNet root (read-only).")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=project_root / "data" / "tr3d_scannet",
        help="Prepared metadata/link root used by the TR3D configs.")
    parser.add_argument("--calibration-size", type=int, default=100)
    parser.add_argument("--audit-size", type=int, default=100)
    parser.add_argument(
        "--seed", default="boxfusion-genuine-tr3d-v1",
        help="Stable string seed recorded in the split contract.")
    parser.add_argument(
        "--frames-root",
        type=Path,
        default=None,
        help="Per-scene RGB-D root. Defaults to SOURCE_ROOT/scans.sens; both "
             "scene/frames/{color,depth,pose,intrinsic} and direct "
             "scene/{color,depth,pose,intrinsic} layouts are accepted.")
    parser.add_argument(
        "--no-links",
        action="store_true",
        help="Validate and write metadata without creating source symlinks.")
    return parser.parse_args()


def validate_source_assets(source_root: Path) -> Dict[str, Path]:
    required = {
        "train_list": source_root / "meta_data" / "scannetv2_train.txt",
        "val_list": source_root / "meta_data" / "scannetv2_val.txt",
        "train_info": source_root / "scannet_infos_train.pkl",
        "val_info": source_root / "scannet_infos_val.pkl",
        "points": source_root / "points",
        "instance_mask": source_root / "instance_mask",
        "semantic_mask": source_root / "semantic_mask",
    }
    missing = [str(path) for path in required.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "ScanNet source is incomplete: " + ", ".join(missing))
    return required


def discover_frame_scenes(frames_root: Path) -> List[str]:
    if not frames_root.is_dir():
        return []
    scenes = []
    for path in frames_root.iterdir():
        if not path.is_dir():
            continue
        candidate = path / "frames" if (path / "frames").is_dir() else path
        if all((candidate / name).is_dir()
               for name in ("color", "depth", "pose", "intrinsic")):
            scenes.append(path.name)
    return sorted(scenes)


def main() -> int:
    args = parse_args()
    source_root = args.source_root.resolve()
    output_root = args.output_root.resolve()
    frames_root = (
        args.frames_root.resolve()
        if args.frames_root is not None
        else (source_root / "scans.sens").resolve())
    assets = validate_source_assets(source_root)
    official_train = read_scene_list(assets["train_list"])
    official_val = read_scene_list(assets["val_list"])
    overlap = sorted(set(official_train) & set(official_val))
    if overlap:
        raise ValueError(
            "official ScanNet train/val lists overlap: " + ", ".join(overlap))

    partitions = deterministic_partition(
        official_train,
        forbidden_scenes=official_val,
        calibration_size=args.calibration_size,
        audit_size=args.audit_size,
        seed=args.seed,
    )
    train_meta, train_rows = load_info(assets["train_info"])
    val_meta, val_rows = load_info(assets["val_info"])
    train_index = index_info_rows(train_rows)
    val_index = index_info_rows(val_rows)
    if set(train_index) != set(official_train):
        missing = sorted(set(official_train) - set(train_index))
        extra = sorted(set(train_index) - set(official_train))
        raise ValueError(
            "train info/list mismatch: "
            f"missing={missing[:10]}, extra={extra[:10]}")
    if set(val_index) != set(official_val):
        missing = sorted(set(official_val) - set(val_index))
        extra = sorted(set(val_index) - set(official_val))
        raise ValueError(
            "val info/list mismatch: "
            f"missing={missing[:10]}, extra={extra[:10]}")

    split_root = output_root / "splits"
    annotation_root = output_root / "annotations"
    for name, scenes in partitions.items():
        write_scene_list(split_root / f"{name}.txt", scenes)
        value = build_foreground_info(
            train_meta, train_rows, scenes, point_path_prefix="full")
        dump_pickle_atomic(
            annotation_root / f"scannet_infos_{name}_foreground.pkl", value)
    write_scene_list(split_root / "official_val.txt", sorted(official_val))
    official_val_info = build_foreground_info(
        val_meta, val_rows, official_val, point_path_prefix="full")
    dump_pickle_atomic(
        annotation_root / "scannet_infos_official_val_foreground.pkl",
        official_val_info,
    )

    if not args.no_links:
        ensure_directory_link(output_root / "points" / "full", assets["points"])
        ensure_directory_link(output_root / "instance_mask",
                              assets["instance_mask"])
        ensure_directory_link(output_root / "semantic_mask",
                              assets["semantic_mask"])

    assignment = {
        scene: split
        for split, scenes in partitions.items()
        for scene in scenes
    }
    frame_scenes = discover_frame_scenes(frames_root)
    invalid_frame_scenes = sorted(
        set(frame_scenes) - set(official_train) - set(official_val))
    if invalid_frame_scenes:
        raise ValueError(
            "RGB-D root includes scenes outside official ScanNet train/val: "
            + ", ".join(invalid_frame_scenes[:10]))
    train_frame_scenes = sorted(set(frame_scenes) & set(official_train))
    available_by_split: Dict[str, List[str]] = {
        split: sorted(scene for scene in train_frame_scenes
                      if assignment.get(scene) == split)
        for split in partitions
    }
    for split, scenes in available_by_split.items():
        write_scene_list(
            split_root / f"trajectory_available_{split}.txt", scenes)

    manifest_rows = []
    for source_split, scenes in (
            ("official_train", sorted(official_train)),
            ("official_val", sorted(official_val))):
        index = train_index if source_split == "official_train" else val_index
        for scene in scenes:
            row = index[scene]
            axis = row.get("axis_align_matrix")
            manifest_rows.append({
                "scene_id": scene,
                "official_split": source_split,
                "experiment_split": assignment.get(scene),
                "has_boxfusion_frames": scene in set(frame_scenes),
                "source_point_path": str(assets["points"] / f"{scene}.bin"),
                "source_instance_mask_path":
                    str(assets["instance_mask"] / f"{scene}.bin"),
                "source_semantic_mask_path":
                    str(assets["semantic_mask"] / f"{scene}.bin"),
                "source_instance_count": len(row.get("instances", [])),
                "coordinate_frame": "world_unaligned",
                "box_coordinate_frame": "scannet_axis_aligned",
                "axis_align_matrix": axis,
            })
    write_jsonl(output_root / "scene_manifest.jsonl", manifest_rows)

    contract = {
        "schema": DATASET_SCHEMA,
        "source_root": str(source_root),
        "output_root": str(output_root),
        "seed": args.seed,
        "coordinate_contract": {
            "stored_points": "world_unaligned",
            "stored_detection_boxes": "scannet_axis_aligned",
            "training_transform": "GlobalAlignment exactly once",
            "inference_export_requirement":
                "inverse axis_align_matrix before comparison/fusion with "
                "BoxFusion world_unaligned boxes",
        },
        "counts": {
            **{name: len(scenes) for name, scenes in partitions.items()},
            "official_val": len(official_val),
            "trajectory_discovered_total": len(frame_scenes),
            "trajectory_available_train_total": len(train_frame_scenes),
            **{
                f"trajectory_available_{name}": len(scenes)
                for name, scenes in available_by_split.items()
            },
        },
        "scene_list_sha256": {
            **{
                name: sha256_lines(scenes)
                for name, scenes in partitions.items()
            },
            "official_val": sha256_lines(sorted(official_val)),
        },
        "forbidden_training_scenes": {
            "file": str(split_root / "official_val.txt"),
            "count": len(official_val),
            "sha256": sha256_lines(sorted(official_val)),
        },
        "links_created": not args.no_links,
        "frames_root": str(frames_root),
    }
    dump_json_atomic(output_root / "DATASET_CONTRACT.json", contract)
    print(json.dumps(contract, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
