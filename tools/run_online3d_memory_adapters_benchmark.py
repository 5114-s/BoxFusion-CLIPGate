#!/usr/bin/env python3
"""Run the released Online3D Memory-based Adapters detector scene by scene.

This runner keeps the model's temporal memories alive while moving only one
RGB-D frame to the GPU at a time.  That is equivalent to the authors' online
test loop, but avoids materialising a complete (and sometimes 2,000-frame)
CA-1M sequence as one GPU tensor.

The terminal scene map is written in the BoxFusion evaluator format::

    [[(class_id, world_corners_8x3, confidence), ...]]

The released detector is closed-vocabulary.  For the class-agnostic protocol
we retain its predicted class only as provenance and apply one final
class-agnostic AABB NMS after collapsing the categories.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ONLINE3D_ROOT = ROOT / "external" / "Online3D_official_20261108"
DEFAULT_CONFIG = (
    DEFAULT_ONLINE3D_ROOT
    / "configs/fcaf3d_online/online3dv3FF_8x2_scannet-3d-18class.py"
)
DEFAULT_CHECKPOINT = DEFAULT_ONLINE3D_ROOT / "checkpoints/FCAF3D-SV+Ours.pth"
DEFAULT_SCANNET_ROOT = Path("/extra/ZhaoX/scannet_data/scans")
DEFAULT_SCANNET_LIST = (
    ROOT / "evaluation/data_util/meta_data/scannetv2_val.txt"
)
DEFAULT_CA1M_ROOT = Path("/extra/ZhaoX/boxfusion_ca1m")

SCANNET_INTRINSIC = np.array(
    [[577.870605, 0.0, 319.5], [0.0, 577.870605, 239.5], [0.0, 0.0, 1.0]],
    dtype=np.float32,
)
IMAGE_MEAN = np.array([103.530, 116.280, 123.675], dtype=np.float32)
IMAGE_STD = np.ones(3, dtype=np.float32)


class InvalidDepthFrame(ValueError):
    """Raised when a frame contains no finite positive depth samples."""


@dataclass(frozen=True)
class Frame:
    frame_id: int
    image_path: Path
    depth_path: Path
    camera_to_world: np.ndarray
    depth_intrinsic: np.ndarray
    image_intrinsic: np.ndarray
    resize_image_to: tuple[int, int] | None


def numeric_paths(directory: Path, suffixes: Sequence[str]) -> dict[int, Path]:
    suffix_set = {suffix.lower() for suffix in suffixes}
    paths: dict[int, Path] = {}
    for path in directory.iterdir():
        if not path.is_file() or path.suffix.lower() not in suffix_set:
            continue
        try:
            frame_id = int(path.stem)
        except ValueError:
            continue
        paths.setdefault(frame_id, path)
    return paths


def read_scene_list(path: Path) -> list[str]:
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def discover_ca1m_scenes(data_root: Path) -> list[str]:
    scenes = []
    for path in data_root.iterdir():
        if (
            path.is_dir()
            and path.name.isdigit()
            and (path / "after_filter_boxes.npy").is_file()
            and (path / "all_poses.npy").is_file()
            and (path / "K_depth.txt").is_file()
            and (path / "K_rgb.txt").is_file()
        ):
            scenes.append(path.name)
    return sorted(scenes)


def load_axis_alignment(scene_root: Path) -> np.ndarray:
    matrix = np.eye(4, dtype=np.float32)
    metadata = scene_root / f"{scene_root.name}.txt"
    for line in metadata.read_text(encoding="utf-8").splitlines():
        if line.startswith("axisAlignment"):
            values = line.split("=", 1)[1].split()
            matrix = np.asarray(values, dtype=np.float32).reshape(4, 4)
            break
    return matrix


def load_scannet_frames(scene_root: Path, stride: int) -> list[Frame]:
    images = numeric_paths(scene_root / "color", (".jpg", ".jpeg", ".png"))
    depths = numeric_paths(scene_root / "depth", (".png",))
    poses = numeric_paths(scene_root / "pose", (".txt",))
    alignment = load_axis_alignment(scene_root)
    frames = []
    for frame_id in sorted(set(images) & set(depths) & set(poses)):
        if frame_id % stride:
            continue
        pose = np.loadtxt(poses[frame_id]).astype(np.float32).reshape(4, 4)
        pose = alignment @ pose
        if not np.all(np.isfinite(pose)):
            continue
        frames.append(
            Frame(
                frame_id=frame_id,
                image_path=images[frame_id],
                depth_path=depths[frame_id],
                camera_to_world=pose,
                depth_intrinsic=SCANNET_INTRINSIC,
                image_intrinsic=SCANNET_INTRINSIC,
                resize_image_to=(640, 480),
            )
        )
    return frames


def load_ca1m_frames(scene_root: Path, stride: int) -> list[Frame]:
    images = numeric_paths(scene_root / "rgb", (".png", ".jpg", ".jpeg"))
    depths = numeric_paths(scene_root / "depth", (".png",))
    poses = np.load(scene_root / "all_poses.npy", mmap_mode="r").reshape(-1, 4, 4)
    depth_intrinsic = np.loadtxt(scene_root / "K_depth.txt").astype(np.float32)[:3, :3]
    image_intrinsic = np.loadtxt(scene_root / "K_rgb.txt").astype(np.float32)[:3, :3]
    per_frame_path = scene_root / "K_depth_per_frame.npy"
    per_frame_intrinsics = (
        np.load(per_frame_path, mmap_mode="r") if per_frame_path.is_file() else None
    )
    frames = []
    for frame_id in sorted(set(images) & set(depths)):
        if frame_id % stride or frame_id >= len(poses):
            continue
        pose = np.asarray(poses[frame_id], dtype=np.float32).reshape(4, 4)
        if not np.all(np.isfinite(pose)):
            continue
        frame_depth_intrinsic = (
            np.asarray(per_frame_intrinsics[frame_id], dtype=np.float32)[:3, :3]
            if per_frame_intrinsics is not None and frame_id < len(per_frame_intrinsics)
            else depth_intrinsic
        )
        frames.append(
            Frame(
                frame_id=frame_id,
                image_path=images[frame_id],
                depth_path=depths[frame_id],
                camera_to_world=pose,
                depth_intrinsic=frame_depth_intrinsic,
                image_intrinsic=image_intrinsic,
                resize_image_to=None,
            )
        )
    return frames


def deterministic_seed(scene: str, frame_id: int) -> int:
    digest = hashlib.sha256(f"{scene}:{frame_id}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little")


def load_rgb(frame: Frame) -> np.ndarray:
    with Image.open(frame.image_path) as handle:
        image = handle.convert("RGB")
        if frame.resize_image_to is not None and image.size != frame.resize_image_to:
            image = image.resize(frame.resize_image_to, Image.Resampling.BILINEAR)
        return np.asarray(image, dtype=np.uint8)


def backproject_frame(
    frame: Frame,
    image_rgb: np.ndarray,
    scene: str,
    num_points: int,
) -> np.ndarray:
    with Image.open(frame.depth_path) as handle:
        depth_raw = np.asarray(handle)
    depth = depth_raw.astype(np.float32) / 1000.0
    height, width = depth.shape

    if image_rgb.shape[:2] != (height, width):
        color_image = Image.fromarray(image_rgb).resize(
            (width, height), Image.Resampling.BILINEAR
        )
        point_colors = np.asarray(color_image, dtype=np.uint8)
    else:
        point_colors = image_rgb

    valid = np.isfinite(depth) & (depth > 0)
    rows, cols = np.nonzero(valid)
    if len(rows) == 0:
        raise InvalidDepthFrame(f"no valid depth pixels in {frame.depth_path}")
    z = depth[rows, cols]
    intrinsic = frame.depth_intrinsic
    x = (cols.astype(np.float32) - intrinsic[0, 2]) * z / intrinsic[0, 0]
    y = (rows.astype(np.float32) - intrinsic[1, 2]) * z / intrinsic[1, 1]
    camera_points = np.stack((x, y, z, np.ones_like(z)), axis=1)
    world_points = camera_points @ frame.camera_to_world.T
    colors = point_colors[rows, cols].astype(np.float32) / 255.0
    points = np.concatenate((world_points[:, :3], colors), axis=1).astype(np.float32)

    rng = np.random.default_rng(deterministic_seed(scene, frame.frame_id))
    indices = rng.choice(len(points), size=num_points, replace=len(points) < num_points)
    return np.ascontiguousarray(points[indices])


def prepare_image(mmcv, image_rgb: np.ndarray):
    image_bgr = np.ascontiguousarray(image_rgb[:, :, ::-1])
    original_height, original_width = image_bgr.shape[:2]
    resized, _ = mmcv.imrescale(
        image_bgr, (1333, 600), return_scale=True, interpolation="bilinear"
    )
    resized_height, resized_width = resized.shape[:2]
    scale_factor = np.array(
        [
            resized_width / original_width,
            resized_height / original_height,
            resized_width / original_width,
            resized_height / original_height,
        ],
        dtype=np.float32,
    )
    normalized = mmcv.imnormalize(resized, IMAGE_MEAN, IMAGE_STD, to_rgb=False)
    padded = mmcv.impad_to_multiple(normalized, 32, pad_val=0)
    tensor = np.ascontiguousarray(padded.transpose(2, 0, 1), dtype=np.float32)
    return tensor, scale_factor, resized.shape, padded.shape


def projection_matrix(intrinsic: np.ndarray, camera_to_world: np.ndarray) -> np.ndarray:
    intrinsic4 = np.eye(4, dtype=np.float32)
    intrinsic4[:3, :3] = intrinsic
    return intrinsic4 @ np.linalg.inv(camera_to_world).astype(np.float32)


def aabb_iou_one_to_many(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    box_min, box_max = box.min(axis=0), box.max(axis=0)
    boxes_min, boxes_max = boxes.min(axis=1), boxes.max(axis=1)
    intersection = np.maximum(
        np.minimum(box_max, boxes_max) - np.maximum(box_min, boxes_min), 0.0
    ).prod(axis=1)
    box_volume = np.maximum(box_max - box_min, 0.0).prod()
    boxes_volume = np.maximum(boxes_max - boxes_min, 0.0).prod(axis=1)
    return intersection / np.maximum(box_volume + boxes_volume - intersection, 1e-12)


def class_agnostic_nms(corners: np.ndarray, scores: np.ndarray, threshold: float) -> np.ndarray:
    if len(scores) == 0:
        return np.empty(0, dtype=np.int64)
    order = np.argsort(-scores, kind="stable")
    keep = []
    while len(order):
        current = int(order[0])
        keep.append(current)
        if len(order) == 1:
            break
        remaining = order[1:]
        ious = aabb_iou_one_to_many(corners[current], corners[remaining])
        order = remaining[ious <= threshold]
    return np.asarray(keep, dtype=np.int64)


def atomic_pickle_dump(value: object, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="wb", dir=output_path.parent, prefix=f".{output_path.name}.", delete=False
    ) as handle:
        temporary = Path(handle.name)
        pickle.dump(value, handle, protocol=pickle.HIGHEST_PROTOCOL)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, output_path)


def atomic_json_dump(value: object, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        dir=output_path.parent,
        prefix=f".{output_path.name}.",
        encoding="utf-8",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, output_path)


def valid_prediction(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size == 0:
        return False
    try:
        with path.open("rb") as handle:
            value = pickle.load(handle)
        return isinstance(value, list) and len(value) == 1 and isinstance(value[0], list)
    except (OSError, EOFError, pickle.UnpicklingError):
        return False


def import_online3d(online3d_root: Path):
    sys.path.insert(0, str(online3d_root.resolve()))
    import mmcv
    import torch
    from mmcv import Config
    from mmcv.runner import load_checkpoint
    from mmdet3d.core.bbox import Box3DMode, DepthInstance3DBoxes
    from mmdet3d.models import build_model

    return (
        mmcv,
        torch,
        Config,
        load_checkpoint,
        Box3DMode,
        DepthInstance3DBoxes,
        build_model,
    )


def build_model(online3d_root: Path, config_path: Path, checkpoint_path: Path, device: str):
    previous = Path.cwd()
    os.chdir(online3d_root)
    try:
        (
            mmcv,
            torch,
            Config,
            load_checkpoint,
            box3d_mode,
            depth_boxes,
            model_builder,
        ) = import_online3d(online3d_root)
        config = Config.fromfile(str(config_path))
        config.model.train_cfg = None
        model = model_builder(config.model, test_cfg=config.get("test_cfg"))
        checkpoint = load_checkpoint(model, str(checkpoint_path), map_location="cpu")
        model.CLASSES = checkpoint.get("meta", {}).get("CLASSES", ())
        model.eval().to(device)
    finally:
        os.chdir(previous)
    return mmcv, torch, box3d_mode, depth_boxes, model


def run_scene(
    *,
    mmcv,
    torch,
    box3d_mode,
    depth_boxes,
    model,
    dataset: str,
    scene: str,
    scene_root: Path,
    output_root: Path,
    stride: int,
    num_points: int,
    max_frames: int | None,
    class_agnostic_nms_threshold: float,
    device: str,
) -> dict[str, object]:
    output_path = output_root / f"{scene}_boxes.pkl"
    metadata_path = output_root / f"{scene}.json"
    if valid_prediction(output_path):
        return {"scene": scene, "status": "reused", "output": str(output_path)}

    frames = (
        load_scannet_frames(scene_root, stride)
        if dataset == "scannet"
        else load_ca1m_frames(scene_root, stride)
    )
    if max_frames is not None:
        frames = frames[:max_frames]
    if not frames:
        raise RuntimeError(f"no valid frames for {dataset}/{scene}")

    if hasattr(model, "img_memory"):
        model.img_memory.reset()
    if hasattr(model, "memory"):
        model.memory.reset()
    bbox_data_list = []
    terminal = None
    processed_frames = 0
    skipped_invalid_depth = 0
    started = time.perf_counter()

    with torch.inference_mode():
        for index, frame in enumerate(frames, start=1):
            image_rgb = load_rgb(frame)
            try:
                point_array = backproject_frame(frame, image_rgb, scene, num_points)
            except InvalidDepthFrame as error:
                skipped_invalid_depth += 1
                print(
                    f"[{dataset}/{scene}] skipping frame {frame.frame_id}: {error}",
                    flush=True,
                )
                continue
            image_array, scale_factor, resized_shape, padded_shape = prepare_image(
                mmcv, image_rgb
            )
            points = torch.from_numpy(point_array).to(device=device, non_blocking=True)
            image = torch.from_numpy(image_array).unsqueeze(0).to(
                device=device, non_blocking=True
            )
            metadata = {
                "depth2img": projection_matrix(
                    frame.image_intrinsic, frame.camera_to_world
                ),
                "scale_factor": scale_factor,
                "img_shape": resized_shape,
                "pad_shape": padded_shape,
                "box_type_3d": depth_boxes,
                "box_mode_3d": box3d_mode.DEPTH,
                "sample_idx": scene,
                "num_frames": 1,
            }
            current_features = model.extract_feat([points], image, [metadata])
            bbox_list, bbox_data_list = model.neck_with_head.forward_test(
                current_features, bbox_data_list, [metadata]
            )
            terminal = bbox_list[0]
            processed_frames += 1
            del current_features, bbox_list, points, image
            if index == 1 or index % 25 == 0 or index == len(frames):
                elapsed = time.perf_counter() - started
                print(
                    f"[{dataset}/{scene}] {index}/{len(frames)} frames "
                    f"elapsed={elapsed:.1f}s",
                    flush=True,
                )

    if terminal is None:
        corners = np.empty((0, 8, 3), dtype=np.float64)
        scores_np = np.empty(0, dtype=np.float64)
        labels_np = np.empty(0, dtype=np.int64)
        keep = np.empty(0, dtype=np.int64)
    else:
        boxes, scores, labels = terminal
        corners = boxes.corners.detach().cpu().numpy().astype(np.float64)
        scores_np = scores.detach().cpu().numpy().astype(np.float64)
        labels_np = labels.detach().cpu().numpy().astype(np.int64)
        keep = class_agnostic_nms(corners, scores_np, class_agnostic_nms_threshold)
    predictions = [
        (int(labels_np[index]), corners[index], float(scores_np[index]))
        for index in keep
    ]
    atomic_pickle_dump([predictions], output_path)
    elapsed = time.perf_counter() - started
    metadata = {
        "schema": "boxfusion.memory_based_adapters.scene.v1",
        "dataset": dataset,
        "scene": scene,
        "frames": processed_frames,
        "frames_discovered": len(frames),
        "frames_processed": processed_frames,
        "frames_skipped_invalid_depth": skipped_invalid_depth,
        "stride": stride,
        "points_per_frame": num_points,
        "terminal_boxes_before_class_agnostic_nms": int(len(scores_np)),
        "terminal_boxes": len(predictions),
        "class_agnostic_nms_iou": class_agnostic_nms_threshold,
        "elapsed_seconds": elapsed,
        "fps": processed_frames / elapsed,
        "closed_vocabulary_model": True,
        "output": str(output_path),
    }
    atomic_json_dump(metadata, metadata_path)
    return {**metadata, "status": "completed"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("scannet", "ca1m"), required=True)
    parser.add_argument("--scene", help="run exactly one scene")
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--scene-list", type=Path, default=DEFAULT_SCANNET_LIST)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--online3d-root", type=Path, default=DEFAULT_ONLINE3D_ROOT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--stride", type=int)
    parser.add_argument("--num-points", type=int, default=5000)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--class-agnostic-nms-iou", type=float, default=0.5)
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="build the model and load the checkpoint, then exit without data/GPU inference",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not (0 <= args.shard_index < args.num_shards):
        raise ValueError("shard-index must be in [0, num-shards)")
    for required in (args.online3d_root, args.config, args.checkpoint):
        if not required.exists():
            raise FileNotFoundError(required)

    model_device = "cpu" if args.verify_only else args.device
    mmcv, torch, box3d_mode, depth_boxes, model = build_model(
        args.online3d_root, args.config, args.checkpoint, model_device
    )
    print(
        f"environment verified: torch={torch.__version__} mmcv={mmcv.__version__} "
        f"model={type(model).__name__} device={model_device}",
        flush=True,
    )
    if args.verify_only:
        return 0

    data_root = args.data_root or (
        DEFAULT_SCANNET_ROOT if args.dataset == "scannet" else DEFAULT_CA1M_ROOT
    )
    stride = args.stride or (20 if args.dataset == "scannet" else 1)
    scenes = (
        read_scene_list(args.scene_list)
        if args.dataset == "scannet"
        else discover_ca1m_scenes(data_root)
    )
    expected = 100 if args.dataset == "scannet" else 107
    if len(scenes) != expected:
        raise RuntimeError(f"expected {expected} {args.dataset} scenes, found {len(scenes)}")
    if args.scene:
        if args.scene not in scenes:
            raise ValueError(f"scene {args.scene!r} is not in the frozen scene set")
        scenes = [args.scene]
    else:
        scenes = scenes[args.shard_index :: args.num_shards]

    args.output_root.mkdir(parents=True, exist_ok=True)
    run_manifest = {
        "schema": "boxfusion.memory_based_adapters.run.v1",
        "dataset": args.dataset,
        "scenes": scenes,
        "stride": stride,
        "points_per_frame": args.num_points,
        "num_shards": args.num_shards,
        "shard_index": args.shard_index,
        "online3d_root": str(args.online3d_root.resolve()),
        "config": str(args.config.resolve()),
        "checkpoint": str(args.checkpoint.resolve()),
    }
    atomic_json_dump(
        run_manifest, args.output_root / f"run_shard_{args.shard_index:02d}.json"
    )
    for index, scene in enumerate(scenes, start=1):
        print(f"[{index}/{len(scenes)}] starting {args.dataset}/{scene}", flush=True)
        result = run_scene(
            mmcv=mmcv,
            torch=torch,
            box3d_mode=box3d_mode,
            depth_boxes=depth_boxes,
            model=model,
            dataset=args.dataset,
            scene=scene,
            scene_root=data_root / scene,
            output_root=args.output_root,
            stride=stride,
            num_points=args.num_points,
            max_frames=args.max_frames,
            class_agnostic_nms_threshold=args.class_agnostic_nms_iou,
            device=args.device,
        )
        print(json.dumps(result, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
