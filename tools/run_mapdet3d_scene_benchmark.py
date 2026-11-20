#!/usr/bin/env python3
"""Run Map-Det3D as a scene-level BoxFusion-protocol baseline.

This runner intentionally performs inference with the released CA-1M
checkpoint.  It does not train on the ScanNetV2-100 or CA-1M-107 evaluation
scenes.  Terminal scene tracks are exported in the pickle format consumed by
the existing BoxFusion evaluators::

    [[(class_id, world_corners_8x3, confidence), ...]]

ScanNet follows the official Map-Det3D conversion script and samples every
25th valid RGB frame.  CA-1M uses every RGB frame, as in the official CA-1M
conversion script.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from PIL import Image, ImageOps


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MAPDET_ROOT = ROOT / "external" / "Map-Det3D_official_20261108"
DEFAULT_SCANNET_ROOT = Path("/extra/ZhaoX/scannet_data/scans")
DEFAULT_CA1M_ROOT = Path("/extra/ZhaoX/boxfusion_ca1m")
DEFAULT_TORCH_HOME = ROOT / "models" / "mapdet3d_torch_cache"
DEFAULT_SCANNET_LIST = (
    ROOT / "evaluation" / "data_util" / "meta_data" / "scannetv2_val.txt"
)
DEFAULT_MODEL_REVISION = "753d600abdcaf4f28c8789b3ca4536487b2819c1"
MAP_ANYTHING_REVISION = "a1d87e9086706fb9974f3be5a3e3a0ca5401c5aa"
MAPDET_SOURCE_REVISION = "665e054ff4fb74e3363f72fb3f534c8addf60338"
VIS4D_CUDA_OPS_REVISION = "d2478d80045b62f85be19235981a63ea9bc8db66"
DINOV2_SOURCE_REVISION = "7764ea0f912e53c92e82eb78a2a1631e92725fc8"


@dataclass(frozen=True)
class Frame:
    """One posed RGB frame."""

    image_path: Path
    pose: np.ndarray


def numeric_paths(directory: Path, suffixes: Sequence[str]) -> dict[int, Path]:
    """Return numeric-stem files in *directory*, indexed by integer stem."""
    suffix_set = {suffix.lower() for suffix in suffixes}
    paths: dict[int, Path] = {}
    for path in directory.iterdir():
        if path.is_file() and path.suffix.lower() in suffix_set:
            try:
                frame_id = int(path.stem)
            except ValueError:
                continue
            paths.setdefault(frame_id, path)
    return paths


def read_scene_list(path: Path) -> list[str]:
    """Read a newline-delimited scene list."""
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def discover_ca1m_scenes(data_root: Path) -> list[str]:
    """Discover exactly the processed CA-1M scenes that have evaluation GT."""
    scenes = []
    for path in data_root.iterdir():
        if (
            path.is_dir()
            and path.name.isdigit()
            and (path / "after_filter_boxes.npy").is_file()
            and (path / "all_poses.npy").is_file()
        ):
            scenes.append(path.name)
    return sorted(scenes)


def load_scannet_frames(scene_root: Path, stride: int) -> tuple[list[Frame], np.ndarray]:
    """Load finite-pose ScanNet frames using the official sampling rule."""
    images = numeric_paths(scene_root / "color", (".jpg", ".jpeg", ".png"))
    poses = numeric_paths(scene_root / "pose", (".txt",))
    valid: list[Frame] = []
    for raw_id in sorted(set(images) & set(poses)):
        pose = np.loadtxt(poses[raw_id]).astype(np.float32).reshape(4, 4)
        if np.all(np.isfinite(pose)):
            valid.append(Frame(images[raw_id], pose))

    intrinsics = np.loadtxt(
        scene_root / "intrinsic" / "intrinsic_color.txt"
    ).astype(np.float32)[:3, :3]
    return valid[::stride], intrinsics


def load_ca1m_frames(scene_root: Path, stride: int) -> tuple[list[Frame], np.ndarray]:
    """Load processed CA-1M RGB frames and camera-to-world poses."""
    images = numeric_paths(scene_root / "rgb", (".png", ".jpg", ".jpeg"))
    poses = np.load(scene_root / "all_poses.npy").astype(np.float32).reshape(-1, 4, 4)
    common_ids = [frame_id for frame_id in sorted(images) if frame_id < len(poses)]
    valid = [
        Frame(images[frame_id], poses[frame_id])
        for frame_id in common_ids
        if np.all(np.isfinite(poses[frame_id]))
    ]
    intrinsics = np.loadtxt(scene_root / "K_rgb.txt").astype(np.float32)[:3, :3]
    return valid[::stride], intrinsics


def atomic_pickle_dump(value: object, output_path: Path) -> None:
    """Atomically write a pickle so interrupted scenes are never considered done."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="wb", dir=output_path.parent, prefix=f".{output_path.name}.", delete=False
    ) as handle:
        temp_path = Path(handle.name)
        pickle.dump(value, handle, protocol=pickle.HIGHEST_PROTOCOL)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp_path, output_path)


def atomic_json_dump(value: object, output_path: Path) -> None:
    """Atomically write human-readable run metadata."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        dir=output_path.parent,
        prefix=f".{output_path.name}.",
        encoding="utf-8",
        delete=False,
    ) as handle:
        temp_path = Path(handle.name)
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp_path, output_path)


def prepare_imports(mapdet_root: Path):
    """Import Map-Det3D from the pinned official checkout."""
    sys.path.insert(0, str(mapdet_root.resolve()))
    import torch
    from huggingface_hub import hf_hub_download
    from mapdet3d.data.const import AxisMode
    from mapdet3d.model.mapanything import MapAnything
    from mapdet3d.model.mapdet3d import MapDet3D
    from mapdet3d.op.box3d import boxes3d_to_corners
    from mapdet3d.op.mapdet3d.head import RoI2Det

    return (
        torch,
        hf_hub_download,
        AxisMode,
        MapAnything,
        MapDet3D,
        boxes3d_to_corners,
        RoI2Det,
    )


def load_model(
    mapdet_root: Path,
    model_id: str,
    model_revision: str,
    cache_dir: Path,
    score_threshold: float,
):
    """Load the release checkpoint without downloading overwritten backbones."""
    (
        torch,
        hf_hub_download,
        axis_mode,
        map_anything,
        map_det3d,
        boxes3d_to_corners,
        roi2det,
    ) = prepare_imports(mapdet_root)

    # Match the authors' official inference app and ScanNet evaluation config.
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("highest")

    config_path = hf_hub_download(
        "facebook/map-anything",
        "config.json",
        revision=MAP_ANYTHING_REVISION,
        cache_dir=str(cache_dir),
    )
    with open(config_path, encoding="utf-8") as handle:
        config = json.load(handle)
    # The final Map-Det3D checkpoint overwrites all encoder parameters.  This
    # is the low-download loader used by the authors' official HF demo.
    config["encoder_config"]["torch_hub_pretrained"] = False
    original_from_pretrained = map_anything.from_pretrained
    map_anything.from_pretrained = classmethod(
        lambda cls, *unused_args, **unused_kwargs: cls(**config)
    )
    try:
        model = map_det3d.from_pretrained(
            model_id, revision=model_revision, cache_dir=str(cache_dir)
        )
    finally:
        map_anything.from_pretrained = original_from_pretrained

    model.track_whole_scene = True
    model.roi2det = roi2det(
        nms=True, score_threshold=score_threshold, iou_threshold=0.5
    )
    model.eval().to("cuda")
    return torch, axis_mode, boxes3d_to_corners, model


def validate_existing_prediction(path: Path) -> bool:
    """Return true only for a readable BoxFusion-format result."""
    if not path.is_file() or path.stat().st_size == 0:
        return False
    try:
        with path.open("rb") as handle:
            value = pickle.load(handle)
        return isinstance(value, list) and len(value) == 1 and isinstance(value[0], list)
    except (OSError, EOFError, pickle.UnpicklingError):
        return False


def run_scene(
    *,
    torch,
    axis_mode,
    boxes3d_to_corners,
    model,
    dataset: str,
    scene: str,
    data_root: Path,
    output_root: Path,
    stride: int,
    max_frames: int | None,
    model_id: str,
    model_revision: str,
) -> dict[str, object]:
    """Run one scene and save terminal tracks."""
    scene_root = data_root / scene
    if dataset == "scannet":
        frames, intrinsics_np = load_scannet_frames(scene_root, stride)
    else:
        frames, intrinsics_np = load_ca1m_frames(scene_root, stride)
    if max_frames is not None:
        frames = frames[:max_frames]
    if not frames:
        raise RuntimeError(f"{scene}: no valid frames")

    # frame_id=0 also resets all sliding-window caches and the track graph.
    intrinsics = torch.from_numpy(intrinsics_np).to("cuda")
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    terminal = None

    with torch.inference_mode():
        for frame_id, frame in enumerate(frames):
            with Image.open(frame.image_path) as raw_image:
                image = ImageOps.exif_transpose(raw_image).convert("RGB")
                image_np = np.asarray(image, dtype=np.float32)[None]
            image_tensor = torch.from_numpy(
                np.ascontiguousarray(image_np.transpose(0, 3, 1, 2))
            ).to("cuda")
            extrinsics = torch.from_numpy(frame.pose).to("cuda")
            with torch.autocast("cuda", dtype=torch.bfloat16):
                terminal = model(
                    images=[image_tensor],
                    intrinsics=[intrinsics],
                    extrinsics=[extrinsics],
                    frame_ids=[frame_id],
                )

    assert terminal is not None
    boxes = terminal.boxes3d[0].detach().float()
    scores = terminal.scores[0].detach().float().cpu().numpy()
    corners = boxes3d_to_corners(boxes, axis_mode.ROS).cpu().numpy()
    rows = [
        # Plain lists keep prediction pickles portable between the NumPy 2.x
        # inference environment and the NumPy 1.x BoxFusion evaluators.
        (0, corner.astype(np.float32, copy=False).tolist(), float(score))
        for corner, score in zip(corners, scores)
    ]
    output_path = output_root / f"{scene}_boxes.pkl"
    atomic_pickle_dump([rows], output_path)

    elapsed = time.perf_counter() - start
    metadata = {
        "dataset": dataset,
        "scene": scene,
        "model_id": model_id,
        "model_revision": model_revision,
        "map_anything_revision": MAP_ANYTHING_REVISION,
        "mapdet_source_revision": MAPDET_SOURCE_REVISION,
        "vis4d_cuda_ops_revision": VIS4D_CUDA_OPS_REVISION,
        "dinov2_source_revision": DINOV2_SOURCE_REVISION,
        "num_input_frames": len(frames),
        "frame_stride": stride,
        "num_terminal_tracks": len(rows),
        "elapsed_seconds": elapsed,
        "fps_including_io": len(frames) / elapsed,
        "peak_gpu_memory_gib": torch.cuda.max_memory_allocated() / 1024**3,
        "prediction_path": str(output_path),
    }
    atomic_json_dump(metadata, output_root / "metadata" / f"{scene}.json")
    print(json.dumps(metadata, sort_keys=True), flush=True)
    return metadata


def iter_selected_scenes(
    scenes: Sequence[str], shard_index: int, num_shards: int, max_scenes: int | None
) -> Iterable[str]:
    """Select a deterministic interleaved shard."""
    selected = list(scenes[shard_index::num_shards])
    if max_scenes is not None:
        selected = selected[:max_scenes]
    return selected


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("scannet", "ca1m"), required=True)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--scene-list", type=Path)
    parser.add_argument("--scene", action="append", dest="scenes")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--mapdet-root", type=Path, default=DEFAULT_MAPDET_ROOT)
    parser.add_argument("--model-id", default="RoyYang0714/Map-Det3D")
    parser.add_argument("--model-revision", default=DEFAULT_MODEL_REVISION)
    parser.add_argument(
        "--model-cache", type=Path, default=ROOT / "models" / "mapdet3d_hf_cache"
    )
    parser.add_argument("--torch-home", type=Path, default=DEFAULT_TORCH_HOME)
    parser.add_argument("--score-threshold", type=float, default=0.25)
    parser.add_argument(
        "--frame-stride",
        type=int,
        help="default: ScanNet=25 (official), CA-1M=1 (official)",
    )
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--max-scenes", type=int)
    parser.add_argument("--max-frames", type=int, help="smoke tests only")
    parser.add_argument("--no-resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("require 0 <= shard-index < num-shards")

    if args.dataset == "scannet":
        data_root = args.data_root or DEFAULT_SCANNET_ROOT
        scene_list = args.scene_list or DEFAULT_SCANNET_LIST
        scenes = args.scenes or read_scene_list(scene_list)
        stride = args.frame_stride or 25
    else:
        data_root = args.data_root or DEFAULT_CA1M_ROOT
        scenes = args.scenes or (
            read_scene_list(args.scene_list)
            if args.scene_list
            else discover_ca1m_scenes(data_root)
        )
        stride = args.frame_stride or 1

    if stride < 1:
        raise ValueError("frame stride must be positive")
    missing = [scene for scene in scenes if not (data_root / scene).is_dir()]
    if missing:
        raise FileNotFoundError(f"missing {len(missing)} scenes, first: {missing[:5]}")

    selected = list(
        iter_selected_scenes(
            scenes, args.shard_index, args.num_shards, args.max_scenes
        )
    )
    pending = [
        scene
        for scene in selected
        if args.no_resume
        or not validate_existing_prediction(args.output_root / f"{scene}_boxes.pkl")
    ]
    print(
        json.dumps(
            {
                "dataset": args.dataset,
                "total_scenes": len(scenes),
                "selected_scenes": len(selected),
                "pending_scenes": len(pending),
                "shard_index": args.shard_index,
                "num_shards": args.num_shards,
                "frame_stride": stride,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    if not pending:
        return

    args.model_cache.mkdir(parents=True, exist_ok=True)
    # Isolate this run from user-modified global Torch Hub caches.  DINOv2 is
    # used only to construct the encoder architecture; the Map-Det3D release
    # checkpoint overwrites its parameters.
    os.environ["TORCH_HOME"] = str(args.torch_home.resolve())
    torch, axis_mode, boxes3d_to_corners, model = load_model(
        args.mapdet_root,
        args.model_id,
        args.model_revision,
        args.model_cache,
        args.score_threshold,
    )
    for scene in pending:
        run_scene(
            torch=torch,
            axis_mode=axis_mode,
            boxes3d_to_corners=boxes3d_to_corners,
            model=model,
            dataset=args.dataset,
            scene=scene,
            data_root=data_root,
            output_root=args.output_root,
            stride=stride,
            max_frames=args.max_frames,
            model_id=args.model_id,
            model_revision=args.model_revision,
        )
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
