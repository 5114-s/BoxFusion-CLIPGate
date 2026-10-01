"""Data utilities for the isolated class-agnostic TR3D experiment.

The coordinate contract is deliberately explicit:

* ScanNet ``points/*.bin`` and RGB-D trajectory-prefix exports contain XYZ in
  the original (unaligned) ScanNet world frame.
* Every annotation row retains ScanNet's ``axis_align_matrix``.
* The inherited TR3D pipeline applies ``GlobalAlignment`` exactly once.
* ScanNet detection boxes in the info files are already in the aligned frame.

Keeping this contract in one small, dependency-light module makes it possible
to validate the data before importing MMDetection3D or MinkowskiEngine.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import pickle
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Mapping, MutableMapping
from typing import Optional, Sequence, Tuple

import numpy as np


SCENE_RE = re.compile(r"^scene\d{4}_\d{2}$")
DATASET_SCHEMA = "boxfusion.tr3d.scannet_foreground.v1"
PREFIX_SCHEMA = "boxfusion.tr3d.trajectory_prefix.v1"


def read_scene_list(path: Path) -> List[str]:
    """Read and validate a ScanNet scene list."""
    scenes: List[str] = []
    seen = set()
    for line_number, raw in enumerate(path.read_text().splitlines(), start=1):
        scene = raw.strip()
        if not scene or scene.startswith("#"):
            continue
        if not SCENE_RE.fullmatch(scene):
            raise ValueError(
                f"{path}:{line_number}: invalid ScanNet scene id {scene!r}")
        if scene in seen:
            raise ValueError(f"{path}: duplicate scene id {scene}")
        seen.add(scene)
        scenes.append(scene)
    if not scenes:
        raise ValueError(f"{path}: scene list is empty")
    return scenes


def write_scene_list(path: Path, scenes: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{scene}\n" for scene in scenes))


def sha256_lines(lines: Sequence[str]) -> str:
    payload = "".join(f"{line}\n" for line in lines).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def deterministic_partition(
        train_scenes: Sequence[str],
        *,
        forbidden_scenes: Iterable[str] = (),
        calibration_size: int = 100,
        audit_size: int = 100,
        seed: str = "boxfusion-genuine-tr3d-v1",
) -> Dict[str, List[str]]:
    """Partition ScanNet-train deterministically without touching validation.

    Ordering is based on SHA-256 instead of Python's randomized ``hash``.
    Consequently the result is invariant to process, host, locale, and the
    ordering of the input list.
    """
    scenes = sorted(set(train_scenes))
    if len(scenes) != len(train_scenes):
        raise ValueError("training scene list contains duplicate ids")
    forbidden = set(forbidden_scenes)
    overlap = sorted(set(scenes) & forbidden)
    if overlap:
        raise ValueError(
            "training source overlaps forbidden validation scenes: "
            + ", ".join(overlap[:10]))
    if calibration_size < 0 or audit_size < 0:
        raise ValueError("split sizes must be non-negative")
    if calibration_size + audit_size >= len(scenes):
        raise ValueError(
            "calibration_size + audit_size must leave at least one train scene")

    def rank(scene: str) -> Tuple[bytes, str]:
        value = f"{seed}\0{scene}".encode("utf-8")
        return hashlib.sha256(value).digest(), scene

    ranked = sorted(scenes, key=rank)
    calibration = sorted(ranked[:calibration_size])
    audit = sorted(ranked[calibration_size:calibration_size + audit_size])
    train = sorted(ranked[calibration_size + audit_size:])
    result = {"train": train, "calibration": calibration, "audit": audit}
    assert not (set(train) & set(calibration))
    assert not (set(train) & set(audit))
    assert not (set(calibration) & set(audit))
    assert set().union(*map(set, result.values())) == set(scenes)
    return result


def load_info(path: Path) -> Tuple[dict, List[dict]]:
    """Load MMDetection3D v1 info and normalize legacy list-form metadata."""
    with path.open("rb") as handle:
        value = pickle.load(handle)
    if isinstance(value, dict) and isinstance(value.get("data_list"), list):
        return copy.deepcopy(value.get("metainfo", {})), value["data_list"]
    if isinstance(value, list):
        return {}, value
    raise ValueError(
        f"{path}: expected an MMDetection3D info dict or legacy list")


def scene_id_from_info(row: Mapping) -> str:
    lidar = row.get("lidar_points", {})
    relative = lidar.get("lidar_path") or row.get("pts_path")
    if not relative:
        raise ValueError("annotation row has no lidar path")
    name = Path(str(relative)).stem
    # Prefix rows use e.g. scene0006_00__p025.bin.
    scene = name.split("__", 1)[0]
    if not SCENE_RE.fullmatch(scene):
        raise ValueError(f"cannot recover ScanNet scene id from {relative!r}")
    return scene


def index_info_rows(rows: Sequence[dict]) -> Dict[str, dict]:
    result: Dict[str, dict] = {}
    for row in rows:
        scene = scene_id_from_info(row)
        if scene in result:
            raise ValueError(f"annotation info contains duplicate scene {scene}")
        result[scene] = row
    return result


def collapse_row_to_foreground(
        row: Mapping,
        *,
        point_path_prefix: Optional[str] = "full",
) -> dict:
    """Deep-copy one ScanNet row and map all detection instances to label 0."""
    output = copy.deepcopy(dict(row))
    for instance in output.get("instances", []):
        if "bbox_3d" not in instance:
            raise ValueError("instance is missing bbox_3d")
        # Avoid the substring ``label``: Det3DDataset treats every instance
        # field containing it as a training label and applies label_mapping.
        instance["source_category_id"] = int(
            instance.get("bbox_label_3d", -1))
        instance["bbox_label_3d"] = 0

    if point_path_prefix is not None:
        lidar = output.setdefault("lidar_points", {})
        relative = lidar.get("lidar_path") or output.get("pts_path")
        if not relative:
            raise ValueError("annotation row has no lidar path")
        basename = Path(str(relative)).name
        rewritten = str(Path(point_path_prefix) / basename)
        lidar["lidar_path"] = rewritten
        output.pop("pts_path", None)
    output["coordinate_frame"] = "world_unaligned"
    output["box_coordinate_frame"] = "scannet_axis_aligned"
    return output


def foreground_metainfo(source_metainfo: Mapping) -> dict:
    output = copy.deepcopy(dict(source_metainfo))
    output["categories"] = {"foreground": 0}
    output["classes"] = ("foreground",)
    output["task"] = "class_agnostic_3d_detection"
    output["schema"] = DATASET_SCHEMA
    return output


def build_foreground_info(
        source_metainfo: Mapping,
        source_rows: Sequence[dict],
        scene_ids: Sequence[str],
        *,
        point_path_prefix: Optional[str] = "full",
) -> dict:
    index = index_info_rows(source_rows)
    missing = sorted(set(scene_ids) - set(index))
    if missing:
        raise ValueError(
            "annotation info is missing requested scenes: "
            + ", ".join(missing[:10]))
    rows = [
        collapse_row_to_foreground(
            index[scene], point_path_prefix=point_path_prefix)
        for scene in sorted(scene_ids)
    ]
    return {
        "metainfo": foreground_metainfo(source_metainfo),
        "data_list": rows,
    }


def dump_pickle_atomic(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        pickle.dump(value, handle, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(temporary, path)


def ensure_directory_link(link: Path, target: Path) -> None:
    """Create a directory symlink, refusing to overwrite a different target."""
    target = target.resolve()
    link.parent.mkdir(parents=True, exist_ok=True)
    if link.is_symlink():
        if link.resolve() != target:
            raise ValueError(f"{link} already points to {link.resolve()}")
        return
    if link.exists():
        raise ValueError(f"{link} already exists and is not a symlink")
    link.symlink_to(target, target_is_directory=True)


def dump_json_atomic(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def write_jsonl(path: Path, rows: Iterable[Mapping]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), sort_keys=True) + "\n")
    os.replace(temporary, path)


def numeric_frame_files(directory: Path,
                        suffixes: Sequence[str]) -> Dict[int, Path]:
    result: Dict[int, Path] = {}
    allowed = {suffix.lower() for suffix in suffixes}
    if not directory.is_dir():
        return result
    for path in directory.iterdir():
        if not path.is_file() or path.suffix.lower() not in allowed:
            continue
        try:
            frame_id = int(path.stem)
        except ValueError:
            continue
        result[frame_id] = path
    return result


@dataclass(frozen=True)
class FrameBundle:
    scene_id: str
    frame_root: Path
    color: Mapping[int, Path]
    depth: Mapping[int, Path]
    pose: Mapping[int, Path]
    intrinsic_depth: Path
    intrinsic_color: Path
    extrinsic_depth: Path
    extrinsic_color: Path

    @property
    def common_ids(self) -> List[int]:
        return sorted(set(self.color) & set(self.depth) & set(self.pose))


def discover_frame_bundle(frame_root: Path, scene_id: str) -> FrameBundle:
    scene_root = frame_root / scene_id
    frames = scene_root / "frames"
    if not frames.is_dir():
        # Also accept a root that already points at per-scene frame folders.
        frames = scene_root
    intrinsic = frames / "intrinsic"
    bundle = FrameBundle(
        scene_id=scene_id,
        frame_root=frames,
        color=numeric_frame_files(frames / "color", (".jpg", ".jpeg", ".png")),
        depth=numeric_frame_files(frames / "depth", (".png", ".tiff")),
        pose=numeric_frame_files(frames / "pose", (".txt",)),
        intrinsic_depth=intrinsic / "intrinsic_depth.txt",
        intrinsic_color=intrinsic / "intrinsic_color.txt",
        extrinsic_depth=intrinsic / "extrinsic_depth.txt",
        extrinsic_color=intrinsic / "extrinsic_color.txt",
    )
    missing = [
        str(path) for path in (
            bundle.intrinsic_depth,
            bundle.intrinsic_color,
            bundle.extrinsic_depth,
            bundle.extrinsic_color,
        ) if not path.is_file()
    ]
    if missing:
        raise FileNotFoundError(
            f"{scene_id}: missing calibration files: {', '.join(missing)}")
    if not bundle.common_ids:
        raise ValueError(f"{scene_id}: no common RGB/depth/pose frame ids")
    return bundle


def prefix_schedule(
        frame_ids: Sequence[int],
        *,
        fractions: Sequence[float] = (0.25, 0.5, 0.75, 1.0),
        frame_stride: int = 25,
) -> List[dict]:
    if frame_stride <= 0:
        raise ValueError("frame_stride must be positive")
    if not frame_ids:
        raise ValueError("frame_ids is empty")
    sampled = list(sorted(frame_ids))[::frame_stride]
    if sampled[-1] != max(frame_ids):
        sampled.append(max(frame_ids))
    result = []
    previous_count = 0
    for fraction in fractions:
        if not (0 < fraction <= 1):
            raise ValueError("prefix fractions must be in (0, 1]")
        count = max(1, min(len(sampled), math.ceil(len(sampled) * fraction)))
        if count <= previous_count:
            continue
        result.append({
            "fraction": float(fraction),
            "sampled_frame_count": count,
            "frame_ids": sampled[:count],
            "last_frame_id": sampled[count - 1],
        })
        previous_count = count
    if result[-1]["sampled_frame_count"] != len(sampled):
        result.append({
            "fraction": 1.0,
            "sampled_frame_count": len(sampled),
            "frame_ids": sampled,
            "last_frame_id": sampled[-1],
        })
    return result


def load_matrix(path: Path) -> np.ndarray:
    matrix = np.loadtxt(path, dtype=np.float64)
    if matrix.shape != (4, 4):
        raise ValueError(f"{path}: expected a 4x4 matrix, got {matrix.shape}")
    return matrix


def valid_pose(path: Path) -> Optional[np.ndarray]:
    pose = load_matrix(path)
    if not np.all(np.isfinite(pose)):
        return None
    if abs(np.linalg.det(pose[:3, :3])) < 1e-8:
        return None
    return pose


def transform_xyz(xyz: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError("xyz must have shape (N, 3)")
    return xyz @ matrix[:3, :3].T + matrix[:3, 3]


def _load_image(path: Path) -> np.ndarray:
    # Pillow is already a BoxFusion runtime dependency and avoids a hard OpenCV
    # import in data-only validation and unit tests.
    from PIL import Image
    return np.asarray(Image.open(path))


def backproject_rgbd(
        *,
        depth_path: Path,
        color_path: Path,
        pose: np.ndarray,
        intrinsic_depth: np.ndarray,
        intrinsic_color: np.ndarray,
        extrinsic_depth: np.ndarray,
        extrinsic_color: np.ndarray,
        depth_scale: float = 1000.0,
        min_depth: float = 0.1,
        max_depth: float = 6.0,
        pixel_stride: int = 4,
) -> np.ndarray:
    """Back-project one ScanNet RGB-D frame into unaligned world XYZRGB."""
    if depth_scale <= 0 or pixel_stride <= 0:
        raise ValueError("depth_scale and pixel_stride must be positive")
    depth_raw = _load_image(depth_path)
    color = _load_image(color_path)
    if depth_raw.ndim != 2:
        raise ValueError(f"{depth_path}: expected a single-channel depth image")
    if color.ndim != 3 or color.shape[2] < 3:
        raise ValueError(f"{color_path}: expected an RGB image")

    rows = np.arange(0, depth_raw.shape[0], pixel_stride, dtype=np.int64)
    cols = np.arange(0, depth_raw.shape[1], pixel_stride, dtype=np.int64)
    u, v = np.meshgrid(cols, rows)
    z = depth_raw[v, u].astype(np.float64) / depth_scale
    valid = np.isfinite(z) & (z >= min_depth) & (z <= max_depth)
    if not np.any(valid):
        return np.zeros((0, 6), dtype=np.float32)
    u = u[valid].astype(np.float64)
    v = v[valid].astype(np.float64)
    z = z[valid]
    fx, fy = intrinsic_depth[0, 0], intrinsic_depth[1, 1]
    cx, cy = intrinsic_depth[0, 2], intrinsic_depth[1, 2]
    depth_xyz = np.stack(
        ((u - cx) * z / fx, (v - cy) * z / fy, z), axis=1)

    # ScanNet calibration extrinsics map camera coordinates to the common
    # sensor frame. Poses map that common camera/sensor frame to world.
    sensor_xyz = transform_xyz(depth_xyz, extrinsic_depth)
    world_xyz = transform_xyz(sensor_xyz, pose)

    color_from_sensor = np.linalg.inv(extrinsic_color)
    color_xyz = transform_xyz(sensor_xyz, color_from_sensor)
    positive_z = color_xyz[:, 2] > 1e-6
    cu = np.rint(
        intrinsic_color[0, 0] * color_xyz[:, 0] / color_xyz[:, 2]
        + intrinsic_color[0, 2]).astype(np.int64)
    cv = np.rint(
        intrinsic_color[1, 1] * color_xyz[:, 1] / color_xyz[:, 2]
        + intrinsic_color[1, 2]).astype(np.int64)
    inside = (
        positive_z & (cu >= 0) & (cu < color.shape[1])
        & (cv >= 0) & (cv < color.shape[0]))
    rgb = np.zeros((len(world_xyz), 3), dtype=np.float64)
    rgb[inside] = color[cv[inside], cu[inside], :3]
    return np.concatenate((world_xyz, rgb), axis=1).astype(np.float32)


def voxel_downsample_first(points: np.ndarray, voxel_size: float) -> np.ndarray:
    """Deterministically retain the earliest point in every world voxel."""
    if voxel_size <= 0:
        raise ValueError("voxel_size must be positive")
    if points.ndim != 2 or points.shape[1] != 6:
        raise ValueError("points must have shape (N, 6)")
    if not len(points):
        return points.astype(np.float32, copy=False)
    keys = np.floor(points[:, :3] / voxel_size).astype(np.int64)
    _, first = np.unique(keys, axis=0, return_index=True)
    # Preserve acquisition order instead of lexicographic voxel order.
    return points[np.sort(first)].astype(np.float32, copy=False)


def points_inside_axis_aligned_boxes(
        aligned_xyz: np.ndarray,
        boxes: np.ndarray,
        *,
        tolerance: float = 0.01,
) -> np.ndarray:
    """Return observed point support for aligned ``[cxyz, dxyz]`` boxes."""
    if boxes.size == 0:
        return np.zeros((0,), dtype=np.int64)
    boxes = np.asarray(boxes, dtype=np.float64)
    if boxes.ndim != 2 or boxes.shape[1] < 6:
        raise ValueError("boxes must have shape (M, >=6)")
    counts = np.zeros((len(boxes),), dtype=np.int64)
    # Chunk boxes rather than points: ScanNet normally has only a few dozen.
    for index, box in enumerate(boxes):
        half = np.maximum(box[3:6] / 2.0 + tolerance, 0)
        counts[index] = np.count_nonzero(
            np.all(np.abs(aligned_xyz - box[:3]) <= half, axis=1))
    return counts


def prefix_tag(fraction: float) -> str:
    value = int(round(fraction * 100))
    return f"p{value:03d}"


def filter_prefix_instances(
        source_instances: Sequence[Mapping],
        prefix_world_points: np.ndarray,
        axis_align_matrix: np.ndarray,
        *,
        min_observed_points: int = 20,
        full_world_points: Optional[np.ndarray] = None,
        min_visibility_fraction: float = 0.0,
        box_tolerance: float = 0.01,
) -> Tuple[List[dict], List[dict]]:
    """Retain only boxes that have sufficient observed-prefix point support."""
    if min_observed_points < 1:
        raise ValueError("min_observed_points must be positive")
    if not (0 <= min_visibility_fraction <= 1):
        raise ValueError("min_visibility_fraction must be in [0, 1]")
    boxes = np.asarray(
        [instance["bbox_3d"][:6] for instance in source_instances],
        dtype=np.float64)
    if not len(boxes):
        return [], []
    prefix_aligned = transform_xyz(
        np.asarray(prefix_world_points[:, :3]), axis_align_matrix)
    observed = points_inside_axis_aligned_boxes(
        prefix_aligned, boxes, tolerance=box_tolerance)
    full_counts: Optional[np.ndarray] = None
    if min_visibility_fraction > 0:
        if full_world_points is None:
            raise ValueError(
                "full_world_points is required for visibility filtering")
        full_aligned = transform_xyz(
            np.asarray(full_world_points[:, :3]), axis_align_matrix)
        full_counts = points_inside_axis_aligned_boxes(
            full_aligned, boxes, tolerance=box_tolerance)

    kept: List[dict] = []
    diagnostics: List[dict] = []
    for index, instance in enumerate(source_instances):
        full_count = (
            int(full_counts[index]) if full_counts is not None else None)
        fraction = (
            float(observed[index] / max(full_count, 1))
            if full_count is not None else None)
        accepted = int(observed[index]) >= min_observed_points
        if fraction is not None:
            accepted = accepted and fraction >= min_visibility_fraction
        diagnostic = {
            "instance_index": index,
            "observed_point_count": int(observed[index]),
            "full_point_count": full_count,
            "visibility_fraction": fraction,
            "accepted": bool(accepted),
        }
        diagnostics.append(diagnostic)
        if accepted:
            item = copy.deepcopy(dict(instance))
            item["source_category_id"] = int(
                item.get("bbox_label_3d", -1))
            item["bbox_label_3d"] = 0
            item["prefix_observed_point_count"] = int(observed[index])
            if full_count is not None:
                item["prefix_full_point_count"] = full_count
                item["prefix_visibility_fraction"] = fraction
            kept.append(item)
    return kept, diagnostics


def read_points_bin(path: Path, dimensions: int = 6) -> np.ndarray:
    values = np.fromfile(path, dtype=np.float32)
    if len(values) % dimensions:
        raise ValueError(
            f"{path}: float count {len(values)} is not divisible by {dimensions}")
    return values.reshape(-1, dimensions)


def build_prefix_info_row(
        source_row: Mapping,
        *,
        relative_point_path: str,
        instances: Sequence[Mapping],
        prefix_metadata: Mapping,
) -> dict:
    output = copy.deepcopy(dict(source_row))
    output["lidar_points"] = {
        "num_pts_feats": 6,
        "lidar_path": relative_point_path,
    }
    output.pop("pts_path", None)
    output["instances"] = copy.deepcopy(list(instances))
    output["coordinate_frame"] = "world_unaligned"
    output["box_coordinate_frame"] = "scannet_axis_aligned"
    output["trajectory_prefix"] = copy.deepcopy(dict(prefix_metadata))
    # These paths are retained for ScanNetDataset.parse_data_info. The TR3D
    # detection pipeline never loads them for prefix samples.
    scene = scene_id_from_info(source_row)
    output.setdefault("pts_semantic_mask_path", f"{scene}.bin")
    output.setdefault("pts_instance_mask_path", f"{scene}.bin")
    return output


def export_scene_prefixes(
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
    """Export one scene's trajectory prefixes and annotation rows."""
    bundle = discover_frame_bundle(frame_root, scene_id)
    schedule = prefix_schedule(
        bundle.common_ids, fractions=fractions, frame_stride=frame_stride)
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
    for item in schedule:
        tag = prefix_tag(item["fraction"])
        relative_point_path = str(
            Path("prefixes") / scene_id / f"{scene_id}__{tag}.bin")
        absolute_point_path = output_root / "points" / relative_point_path
        metadata = {
            "schema": PREFIX_SCHEMA,
            "scene_id": scene_id,
            "tag": tag,
            "fraction": item["fraction"],
            "frame_stride": frame_stride,
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
        used_frame_ids: List[int] = []
        for frame_id in item["frame_ids"]:
            if frame_id not in frame_cache:
                pose = valid_pose(bundle.pose[frame_id])
                if pose is None:
                    frame_cache[frame_id] = np.zeros((0, 6), dtype=np.float32)
                else:
                    frame_cache[frame_id] = backproject_rgbd(
                        depth_path=bundle.depth[frame_id],
                        color_path=bundle.color[frame_id],
                        pose=pose,
                        intrinsic_depth=intrinsic_depth,
                        intrinsic_color=intrinsic_color,
                        extrinsic_depth=extrinsic_depth,
                        extrinsic_color=extrinsic_color,
                        depth_scale=depth_scale,
                        min_depth=min_depth,
                        max_depth=max_depth,
                        pixel_stride=pixel_stride,
                    )
            if len(frame_cache[frame_id]):
                parts.append(frame_cache[frame_id])
                used_frame_ids.append(frame_id)
        if not parts:
            raise ValueError(f"{scene_id}/{tag}: every selected pose was invalid")
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
            "used_frame_ids": used_frame_ids,
            "point_count": int(len(prefix_points)),
            "source_instance_count": len(source_row.get("instances", [])),
            "kept_instance_count": len(kept),
            "instance_support": support,
        })
        exported_rows.append(
            build_prefix_info_row(
                source_row,
                relative_point_path=relative_point_path,
                instances=kept,
                prefix_metadata=metadata,
            ))
        manifests.append(metadata)
    return exported_rows, manifests
