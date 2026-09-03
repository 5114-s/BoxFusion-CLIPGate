#!/usr/bin/env python3
"""Build a strict, auditable SAM3 teacher cache for ScanNet.

This tool deliberately runs SAM3 *offline*.  Its output is the same safe NPZ
format consumed by ``StrictCacheProposalProvider`` during BoxFusion inference.
The cache key is bound to the exact oriented RGB array, logical frame id and
immutable namespace, so a cache generated with different RGB extraction or
frame scheduling cannot be replayed silently.

The frame schedule mirrors ``demo.py`` rather than selecting the last frame
conventionally.  In particular, the current driver exits after processing a
frame when ``count == N - 1 or count + gap > N - 1`` (where ``count`` has
already been incremented).  Consequently it normally never processes the
physical last frame.  Supplemental inference is then run every
``proposal_interval`` BoxFusion keyframes.

SAM3 and torch are imported lazily.  The scheduling, RGB preprocessing,
proposal normalization/deduplication and cache-writing helpers are therefore
unit-testable on CPU without installing or loading SAM3.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import re
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np


TOOL_ROOT = Path(__file__).resolve().parents[1]
if str(TOOL_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOL_ROOT))

from boxfusion.supplemental_proposals import (  # noqa: E402
    NpzProposalCache,
    SupplementalProposal,
    proposal_cache_key,
)


SCANNET18_PROMPTS: Tuple[str, ...] = (
    "cabinet",
    "bed",
    "chair",
    "sofa",
    "table",
    "door",
    "window",
    "bookshelf",
    "picture",
    "counter",
    "desk",
    "curtain",
    "refrigerator",
    "shower curtain",
    "toilet",
    "sink",
    "bathtub",
    "garbage bin",
)

_SCENE_RE = re.compile(r"^scene\d{4}_\d{2}$")
_PROVIDER_CALL_RE = re.compile(r"\bprovider_calls=(\d+)\b")
_ORIENTATION_NAMES = ("upright", "left", "upside_down", "right")
_ROT90_TO_UPRIGHT = (0, -1, 2, 1)
_MANIFEST_SCHEMA = "boxfusion_scannet_sam3_teacher_cache_v1"
_RUNTIME_RGB_MANIFEST_SCHEMA = "boxfusion_scannet_runtime_rgb_v1"


def _sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_text_file(path: Path) -> str:
    return _sha256_file(path, chunk_size=1024 * 1024)


def _rgb_content_sha256(image: np.ndarray) -> str:
    """Hash the exact array contract used by ``proposal_cache_key``."""

    rgb = np.asarray(image)
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"runtime RGB must have shape (H,W,3), got {rgb.shape}")
    contiguous = np.ascontiguousarray(rgb)
    digest = hashlib.sha256()
    digest.update(str(contiguous.shape).encode("ascii"))
    digest.update(contiguous.dtype.str.encode("ascii"))
    digest.update(contiguous.tobytes())
    return digest.hexdigest()


def _write_npy_atomic(path: Path, image: np.ndarray) -> None:
    """Losslessly write one runtime RGB array without exposing partial files."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Optional[Path] = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w+b",
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=str(path.parent),
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            np.save(handle, np.ascontiguousarray(image), allow_pickle=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(str(temporary), str(path))
    except Exception:
        if temporary is not None:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass
        raise


def _runtime_rgb_manifest_path(
    runtime_rgb_dir: Path,
    shard_index: int,
    num_shards: int,
) -> Path:
    return (
        runtime_rgb_dir
        / "manifests"
        / f"runtime_rgb_shard_{shard_index:03d}_of_{num_shards:03d}.json"
    )


def _runtime_rgb_frame_path(
    runtime_rgb_dir: Path,
    scene_id: str,
    frame_index: int,
) -> Path:
    return runtime_rgb_dir / "frames" / scene_id / f"{int(frame_index):06d}.npy"


def read_scene_list(path: os.PathLike[str] | str) -> List[str]:
    """Read a non-empty, duplicate-free, ordered ScanNet scene list."""

    scene_path = Path(path)
    if not scene_path.is_file():
        raise FileNotFoundError(f"Missing ScanNet scene list: {scene_path}")
    scenes = [
        line.strip()
        for line in scene_path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if not scenes:
        raise ValueError(f"ScanNet scene list is empty: {scene_path}")
    invalid = [scene for scene in scenes if not _SCENE_RE.fullmatch(scene)]
    if invalid:
        raise ValueError(f"Invalid ScanNet scene id: {invalid[0]!r}")
    if len(set(scenes)) != len(scenes):
        duplicate = next(
            scene for scene, count in Counter(scenes).items() if count > 1
        )
        raise ValueError(f"Duplicate ScanNet scene id: {duplicate}")
    return scenes


def scheduled_frame_indices(
    frame_count: int,
    *,
    gap: int = 25,
    proposal_interval: int = 5,
) -> List[int]:
    """Return physical frame indices used by the current BoxFusion provider.

    This is an intentional simulation of the driver, including its early-exit
    condition.  It must not be replaced by logic that appends ``N - 1``.
    """

    if not isinstance(frame_count, (int, np.integer)) or int(frame_count) < 1:
        raise ValueError("frame_count must be a positive integer")
    if not isinstance(gap, (int, np.integer)) or int(gap) < 1:
        raise ValueError("gap must be a positive integer")
    if (
        not isinstance(proposal_interval, (int, np.integer))
        or int(proposal_interval) < 1
    ):
        raise ValueError("proposal_interval must be a positive integer")

    total = int(frame_count)
    stride = int(gap)
    provider_stride = int(proposal_interval)
    count = 0
    keyframe_index = 0
    selected: List[int] = []
    while count < total:
        if count % stride == 0 or count == total - 1:
            if keyframe_index % provider_stride == 0:
                selected.append(count)
            keyframe_index += 1

        count += 1
        if count == total - 1 or (count + stride) > total - 1:
            break
    return selected


def orientation_index_from_pose(camera_to_world: np.ndarray) -> int:
    """Match ``ImageOrientation(get_orientation(RT))`` using NumPy only."""

    pose = np.asarray(camera_to_world, dtype=np.float64)
    if pose.shape != (4, 4):
        raise ValueError(f"pose must have shape (4, 4), received {pose.shape}")
    if not np.isfinite(pose).all():
        raise ValueError("pose must contain only finite values")
    z_vector = pose[2, :3]
    candidates = np.asarray(
        (
            (0.0, -1.0, 0.0),
            (-1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (1.0, 0.0, 0.0),
        ),
        dtype=np.float64,
    )
    return int(np.argmax(candidates @ z_vector))


def orient_rgb_upright(
    image: np.ndarray,
    orientation_index: int,
) -> np.ndarray:
    """Match ``rotate_tensor(..., current, UPRIGHT)`` for an HWC RGB image."""

    rgb = np.asarray(image)
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"image must have shape (H, W, 3), got {rgb.shape}")
    if orientation_index not in range(4):
        raise ValueError("orientation_index must be one of 0, 1, 2, 3")
    oriented = np.rot90(
        rgb,
        k=_ROT90_TO_UPRIGHT[int(orientation_index)],
        axes=(0, 1),
    )
    return np.ascontiguousarray(oriented)


def _numeric_paths(directory: Path, suffix: str) -> List[Path]:
    paths = list(directory.glob(f"*{suffix}"))
    try:
        return sorted(paths, key=lambda path: int(path.stem))
    except ValueError as error:
        raise ValueError(
            f"Non-numeric ScanNet frame name under {directory}"
        ) from error


def load_scannet_pose_sequence(pose_paths: Sequence[Path]) -> List[np.ndarray]:
    """Load poses with the exact ScanNet loader's last-valid inf fallback."""

    poses: List[np.ndarray] = []
    last_valid: Optional[np.ndarray] = None
    for path in pose_paths:
        try:
            pose = np.loadtxt(str(path), dtype=np.float64).reshape(4, 4)
        except (OSError, ValueError) as error:
            raise ValueError(f"Invalid ScanNet pose file: {path}") from error
        if not np.isinf(pose).any():
            if not np.isfinite(pose).all():
                raise ValueError(f"ScanNet pose contains NaN: {path}")
            last_valid = pose
        elif last_valid is not None:
            pose = last_valid.copy()
        else:
            raise ValueError(
                f"First ScanNet pose is infinite and has no fallback: {path}"
            )
        poses.append(np.asarray(pose, dtype=np.float64))
    return poses


def discover_scannet_scene(
    frames_root: os.PathLike[str] | str,
    scene_id: str,
) -> Dict[str, Any]:
    """Discover the independently sorted streams used by ``ScannetDataset``."""

    if not _SCENE_RE.fullmatch(scene_id):
        raise ValueError(f"Invalid ScanNet scene id: {scene_id!r}")
    frames_dir = Path(frames_root) / scene_id / "frames"
    color_paths = _numeric_paths(frames_dir / "color", ".jpg")
    depth_paths = _numeric_paths(frames_dir / "depth", ".png")
    pose_paths = _numeric_paths(frames_dir / "pose", ".txt")
    if not color_paths:
        raise FileNotFoundError(f"No ScanNet RGB frames under {frames_dir}")
    # The production loader indexes these streams in parallel.  Fail early
    # rather than allowing a subtle cache/driver mismatch.
    if not (
        len(color_paths) == len(depth_paths) == len(pose_paths)
    ):
        raise ValueError(
            f"ScanNet stream length mismatch for {scene_id}: "
            f"color={len(color_paths)}, depth={len(depth_paths)}, "
            f"pose={len(pose_paths)}"
        )
    return {
        "frames_dir": frames_dir,
        "color_paths": color_paths,
        "depth_paths": depth_paths,
        "pose_paths": pose_paths,
        "poses": load_scannet_pose_sequence(pose_paths),
        "frame_count": len(color_paths),
    }


def load_scannet_rgb(
    color_path: os.PathLike[str] | str,
    depth_path: os.PathLike[str] | str,
    camera_to_world: np.ndarray,
    *,
    configured_height: int = 480,
    configured_width: int = 640,
) -> Tuple[np.ndarray, int]:
    """Reproduce the RGB array passed from ScanNet into ``demo.py`` exactly."""

    if configured_height < 1 or configured_width < 1:
        raise ValueError("configured image dimensions must be positive")
    try:
        import cv2
    except ImportError as error:
        raise RuntimeError("OpenCV is required for ScanNet RGB preprocessing") from error

    color = cv2.imread(os.fspath(color_path), cv2.IMREAD_COLOR)
    if color is None:
        raise ValueError(f"OpenCV could not read ScanNet RGB: {color_path}")
    depth = cv2.imread(os.fspath(depth_path), cv2.IMREAD_UNCHANGED)
    if depth is None or depth.ndim != 2:
        raise ValueError(f"OpenCV could not read ScanNet depth: {depth_path}")

    # These are deliberately the same OpenCV operations and defaults used by
    # ScannetDataset: BGR->RGB, resize color to depth, then reshape to cfg H/W.
    color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
    depth_height, depth_width = depth.shape
    color = cv2.resize(color, (depth_width, depth_height))
    expected_values = int(configured_height) * int(configured_width) * 3
    if color.size != expected_values:
        raise ValueError(
            "ScanNet depth/config size mismatch: "
            f"depth={(depth_height, depth_width)}, "
            f"config={(configured_height, configured_width)}"
        )
    color = np.asarray(color).reshape(
        (int(configured_height), int(configured_width), 3)
    )
    orientation = orientation_index_from_pose(camera_to_world)
    return orient_rgb_upright(color, orientation), orientation


def stage_runtime_rgb(
    *,
    args: argparse.Namespace,
    scenes: Sequence[str],
    selected_scenes: Sequence[Tuple[int, str]],
    scene_sources: Mapping[str, Mapping[str, Any]],
    full_schedule: Sequence[Tuple[str, int]],
    schedule: Sequence[Tuple[str, int]],
) -> Path:
    """Export exact BoxFusion-runtime pixels as lossless, audited NPY files."""

    if args.runtime_rgb_dir is None:
        raise ValueError("--stage-runtime-rgb requires --runtime-rgb-dir")
    try:
        import cv2
    except ImportError as error:
        raise RuntimeError(
            "OpenCV is required in the BoxFusion runtime RGB exporter"
        ) from error
    if not str(cv2.__version__).startswith("4.6."):
        raise RuntimeError(
            "Runtime RGB staging requires the boxfusion2 OpenCV 4.6 decoder; "
            f"loaded cv2 {cv2.__version__} from {cv2.__file__}"
        )

    runtime_dir = args.runtime_rgb_dir.resolve()
    manifest_path = _runtime_rgb_manifest_path(
        runtime_dir, args.shard_index, args.num_shards
    )
    if manifest_path.exists():
        if args.resume:
            _, _, records = load_runtime_rgb_manifest(
                args=args,
                scenes=scenes,
                selected_scenes=selected_scenes,
                full_schedule=full_schedule,
                schedule=schedule,
            )
            for scene, frame_index in schedule:
                source = scene_sources[scene]
                logical_frame_id = f"{scene}:{frame_index}"
                load_staged_runtime_rgb(
                    runtime_rgb_dir=runtime_dir,
                    record=records[logical_frame_id],
                    namespace=args.namespace.strip(),
                    logical_frame_id=logical_frame_id,
                    source_paths={
                        "color": source["color_paths"][frame_index],
                        "depth": source["depth_paths"][frame_index],
                        "pose": source["pose_paths"][frame_index],
                    },
                )
            print(f"Verified resumed runtime RGB manifest: {manifest_path}")
            return manifest_path
        raise FileExistsError(
            "Refusing to overwrite immutable runtime RGB manifest: "
            f"{manifest_path}"
        )

    frame_records: List[Dict[str, Any]] = []
    print(
        "Staging BoxFusion runtime RGB | "
        f"opencv={cv2.__version__}, frames={len(schedule)}, "
        f"shard={args.shard_index}/{args.num_shards}"
    )
    for position, (scene, frame_index) in enumerate(schedule, start=1):
        source = scene_sources[scene]
        color_path = source["color_paths"][frame_index]
        depth_path = source["depth_paths"][frame_index]
        pose_path = source["pose_paths"][frame_index]
        image, orientation_index = load_scannet_rgb(
            color_path,
            depth_path,
            source["poses"][frame_index],
            configured_height=args.configured_height,
            configured_width=args.configured_width,
        )
        if image.dtype != np.uint8:
            raise TypeError(
                f"Runtime RGB must be uint8, received {image.dtype} for "
                f"{scene}:{frame_index}"
            )
        image = np.ascontiguousarray(image)
        logical_frame_id = f"{scene}:{frame_index}"
        frame_path = _runtime_rgb_frame_path(
            runtime_dir, scene, frame_index
        )
        if frame_path.exists():
            if not args.resume:
                raise FileExistsError(
                    f"Refusing to overwrite staged runtime RGB: {frame_path}"
                )
            prior = np.load(frame_path, allow_pickle=False)
            if not np.array_equal(prior, image):
                raise ValueError(
                    "Interrupted runtime RGB staging contains different "
                    f"pixels: {frame_path}"
                )
        else:
            _write_npy_atomic(frame_path, image)
        reloaded = np.load(frame_path, allow_pickle=False)
        if not np.array_equal(reloaded, image):
            raise RuntimeError(
                f"Lossless runtime RGB round-trip failed: {frame_path}"
            )
        record = {
            "logical_frame_id": logical_frame_id,
            "scene_id": scene,
            "frame_index": int(frame_index),
            "path": str(frame_path.resolve()),
            "shape": list(image.shape),
            "dtype": image.dtype.str,
            "orientation": _ORIENTATION_NAMES[orientation_index],
            "rgb_content_sha256": _rgb_content_sha256(image),
            "npy_sha256": _sha256_file(frame_path),
            "proposal_cache_key": proposal_cache_key(
                args.namespace.strip(), logical_frame_id, image
            ),
            "sources": {
                "color": {
                    "path": str(color_path.resolve()),
                    "sha256": _sha256_file(color_path),
                },
                "depth": {
                    "path": str(depth_path.resolve()),
                    "sha256": _sha256_file(depth_path),
                },
                "pose": {
                    "path": str(pose_path.resolve()),
                    "sha256": _sha256_file(pose_path),
                },
            },
        }
        frame_records.append(record)
        print(
            f"[{position}/{len(schedule)}] staged {logical_frame_id} "
            f"sha256={record['rgb_content_sha256'][:16]}"
        )

    manifest = {
        "schema": _RUNTIME_RGB_MANIFEST_SCHEMA,
        "complete": True,
        "created_unix_seconds": time.time(),
        "exporter": {
            "python": sys.executable,
            "python_version": sys.version.split()[0],
            "numpy_version": np.__version__,
            "opencv_version": cv2.__version__,
            "opencv_path": str(Path(cv2.__file__).resolve()),
        },
        "scene_list": {
            "path": str(args.scene_list.resolve()),
            "sha256": _sha256_text_file(args.scene_list),
            "all_scene_count": len(scenes),
            "selected_scene_ids": [scene for _, scene in selected_scenes],
        },
        "frames_root": str(args.frames_root.resolve()),
        "namespace": args.namespace.strip(),
        "settings": {
            "configured_height": int(args.configured_height),
            "configured_width": int(args.configured_width),
            "gap": int(args.gap),
            "proposal_interval": int(args.proposal_interval),
            "max_frames": args.max_frames,
        },
        "shard": {
            "index": int(args.shard_index),
            "count": int(args.num_shards),
            "pre_limit_provider_calls": len(full_schedule),
            "executed_provider_calls": len(schedule),
        },
        "frames": frame_records,
    }
    _write_json_atomic(manifest_path, manifest)
    print(f"Runtime RGB manifest: {manifest_path}")
    return manifest_path


def load_runtime_rgb_manifest(
    *,
    args: argparse.Namespace,
    scenes: Sequence[str],
    selected_scenes: Sequence[Tuple[int, str]],
    full_schedule: Sequence[Tuple[str, int]],
    schedule: Sequence[Tuple[str, int]],
) -> Tuple[Path, Dict[str, Any], Dict[str, Mapping[str, Any]]]:
    """Load and validate the immutable BoxFusion-runtime staging contract."""

    if args.runtime_rgb_dir is None:
        raise ValueError(
            "SAM3 cache generation/verification requires --runtime-rgb-dir; "
            "JPEG decoding inside the SAM3 environment is forbidden"
        )
    runtime_dir = args.runtime_rgb_dir.resolve()
    manifest_path = _runtime_rgb_manifest_path(
        runtime_dir, args.shard_index, args.num_shards
    )
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Missing immutable runtime RGB manifest: {manifest_path}"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_global = {
        "schema": _RUNTIME_RGB_MANIFEST_SCHEMA,
        "complete": True,
        "scene_list_sha256": _sha256_text_file(args.scene_list),
        "selected_scene_ids": [scene for _, scene in selected_scenes],
        "frames_root": str(args.frames_root.resolve()),
        "namespace": args.namespace.strip(),
        "settings": {
            "configured_height": int(args.configured_height),
            "configured_width": int(args.configured_width),
            "gap": int(args.gap),
            "proposal_interval": int(args.proposal_interval),
            "max_frames": args.max_frames,
        },
        "shard": {
            "index": int(args.shard_index),
            "count": int(args.num_shards),
            "pre_limit_provider_calls": len(full_schedule),
            "executed_provider_calls": len(schedule),
        },
    }
    actual_global = {
        "schema": manifest.get("schema"),
        "complete": manifest.get("complete"),
        "scene_list_sha256": manifest.get("scene_list", {}).get("sha256"),
        "selected_scene_ids": manifest.get("scene_list", {}).get(
            "selected_scene_ids"
        ),
        "frames_root": manifest.get("frames_root"),
        "namespace": manifest.get("namespace"),
        "settings": manifest.get("settings"),
        "shard": manifest.get("shard"),
    }
    if actual_global != expected_global:
        raise ValueError(
            "Runtime RGB manifest disagrees with the requested immutable "
            f"schedule/configuration: {manifest_path}"
        )
    exporter = manifest.get("exporter", {})
    if not str(exporter.get("opencv_version", "")).startswith("4.6."):
        raise ValueError(
            "Runtime RGB manifest was not exported by OpenCV 4.6: "
            f"{manifest_path}"
        )
    records = manifest.get("frames")
    if not isinstance(records, list):
        raise ValueError(f"Invalid runtime RGB frames manifest: {manifest_path}")
    by_id: Dict[str, Mapping[str, Any]] = {}
    for record in records:
        if not isinstance(record, Mapping):
            raise ValueError(f"Invalid runtime RGB frame record: {manifest_path}")
        logical_frame_id = str(record.get("logical_frame_id", ""))
        if not logical_frame_id or logical_frame_id in by_id:
            raise ValueError(
                "Missing or duplicate logical frame id in runtime RGB "
                f"manifest: {logical_frame_id!r}"
            )
        by_id[logical_frame_id] = record
    expected_ids = [f"{scene}:{index}" for scene, index in schedule]
    if list(by_id) != expected_ids:
        raise ValueError(
            "Runtime RGB manifest frame order/content disagrees with the "
            f"requested schedule: {manifest_path}"
        )
    return manifest_path, manifest, by_id


def load_staged_runtime_rgb(
    *,
    runtime_rgb_dir: Path,
    record: Mapping[str, Any],
    namespace: str,
    logical_frame_id: str,
    source_paths: Mapping[str, Path],
) -> Tuple[np.ndarray, int]:
    """Fail-closed load of one lossless runtime RGB array and its provenance."""

    if record.get("logical_frame_id") != logical_frame_id:
        raise ValueError(f"Runtime RGB logical frame mismatch: {logical_frame_id}")
    frame_path = Path(str(record.get("path", ""))).resolve()
    runtime_root = runtime_rgb_dir.resolve()
    try:
        frame_path.relative_to(runtime_root)
    except ValueError as error:
        raise ValueError(
            f"Runtime RGB path escapes staging root: {frame_path}"
        ) from error
    if frame_path.suffix != ".npy" or not frame_path.is_file():
        raise FileNotFoundError(f"Missing staged runtime RGB: {frame_path}")
    if _sha256_file(frame_path) != record.get("npy_sha256"):
        raise ValueError(f"Staged runtime RGB file hash mismatch: {frame_path}")
    for name, source_path in source_paths.items():
        source_record = record.get("sources", {}).get(name, {})
        if source_record.get("path") != str(source_path.resolve()):
            raise ValueError(
                f"Runtime RGB {name} source path mismatch: {logical_frame_id}"
            )
        if _sha256_file(source_path) != source_record.get("sha256"):
            raise ValueError(
                f"Runtime RGB {name} source hash mismatch: {logical_frame_id}"
            )
    image = np.load(frame_path, allow_pickle=False)
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(
            f"Invalid staged runtime RGB dtype/shape at {frame_path}: "
            f"{image.dtype} {image.shape}"
        )
    image = np.ascontiguousarray(image)
    if list(image.shape) != record.get("shape"):
        raise ValueError(f"Runtime RGB shape manifest mismatch: {frame_path}")
    if image.dtype.str != record.get("dtype"):
        raise ValueError(f"Runtime RGB dtype manifest mismatch: {frame_path}")
    if _rgb_content_sha256(image) != record.get("rgb_content_sha256"):
        raise ValueError(f"Runtime RGB content hash mismatch: {frame_path}")
    if proposal_cache_key(namespace, logical_frame_id, image) != record.get(
        "proposal_cache_key"
    ):
        raise ValueError(f"Runtime RGB proposal key mismatch: {frame_path}")
    orientation_name = str(record.get("orientation", ""))
    if orientation_name not in _ORIENTATION_NAMES:
        raise ValueError(f"Invalid runtime RGB orientation: {orientation_name}")
    return image, _ORIENTATION_NAMES.index(orientation_name)


def _to_numpy(value: Any, name: str) -> np.ndarray:
    if value is None:
        raise ValueError(f"SAM3 output is missing {name}")
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    # NumPy has no native bfloat16 dtype.  SAM3 outputs can inherit the CUDA
    # autocast dtype, so promote floating tensors before crossing the boundary.
    if (
        hasattr(value, "is_floating_point")
        and callable(value.is_floating_point)
        and value.is_floating_point()
        and hasattr(value, "float")
    ):
        value = value.float()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def normalize_sam3_output(
    output: Mapping[str, Any],
    *,
    label: str,
    image_shape: Tuple[int, int],
    mask_threshold: float = 0.5,
    min_mask_pixels: int = 64,
    max_per_class: int = 32,
) -> List[SupplementalProposal]:
    """Convert one SAM3 text-prompt output to strict cache proposals."""

    if not isinstance(output, Mapping):
        raise TypeError("SAM3 output must be a mapping")
    clean_label = str(label).strip()
    if not clean_label:
        raise ValueError("SAM3 label must be non-empty")
    height, width = (int(image_shape[0]), int(image_shape[1]))
    if height < 1 or width < 1:
        raise ValueError("image_shape must be positive (H, W)")
    if not np.isfinite(mask_threshold) or not 0.0 <= mask_threshold <= 1.0:
        raise ValueError("mask_threshold must be in [0, 1]")
    if min_mask_pixels < 1 or max_per_class < 1:
        raise ValueError("mask limits must be positive")

    boxes = _to_numpy(output.get("boxes"), "boxes").astype(
        np.float32, copy=False
    )
    scores = _to_numpy(output.get("scores"), "scores").astype(
        np.float32, copy=False
    )
    masks_value = output.get("masks_logits")
    if masks_value is None:
        masks_value = output.get("masks")
    masks = _to_numpy(masks_value, "masks").copy()
    if masks.ndim == 4 and masks.shape[1] == 1:
        masks = masks[:, 0]
    if masks.ndim != 3:
        raise ValueError(
            f"SAM3 masks must have shape (N,H,W) or (N,1,H,W), got {masks.shape}"
        )
    if (
        boxes.ndim != 2
        or boxes.shape[1:] != (4,)
        or scores.ndim != 1
        or boxes.shape[0] != scores.shape[0]
        or masks.shape[0] != scores.shape[0]
    ):
        raise ValueError("SAM3 boxes, scores and masks have inconsistent counts")
    if masks.shape[1:] != (height, width):
        raise ValueError(
            f"SAM3 mask shape {masks.shape[1:]} does not match {(height, width)}"
        )

    binary_masks = (
        masks.astype(np.bool_, copy=False)
        if np.issubdtype(masks.dtype, np.bool_)
        else np.asarray(masks >= float(mask_threshold), dtype=np.bool_)
    )
    order = np.argsort(-scores, kind="stable")
    proposals: List[SupplementalProposal] = []
    for index in order:
        score = float(scores[index])
        bbox = np.asarray(boxes[index], dtype=np.float32).copy()
        mask = np.asarray(binary_masks[index], dtype=np.bool_)
        if (
            not np.isfinite(score)
            or score < 0.0
            or score > 1.0
            or not np.isfinite(bbox).all()
            or int(mask.sum(dtype=np.int64)) < int(min_mask_pixels)
        ):
            continue
        bbox[0::2] = np.clip(bbox[0::2], 0.0, float(width))
        bbox[1::2] = np.clip(bbox[1::2], 0.0, float(height))
        if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
            continue
        proposals.append(
            SupplementalProposal(
                bbox=bbox,
                score=score,
                mask=mask,
                label=clean_label,
                feature=None,
            )
        )
        if len(proposals) >= int(max_per_class):
            break
    return proposals


def _mask_iou(
    first: SupplementalProposal,
    second: SupplementalProposal,
    first_area: int,
    second_area: int,
) -> float:
    x1 = max(int(np.floor(first.bbox[0])), int(np.floor(second.bbox[0])), 0)
    y1 = max(int(np.floor(first.bbox[1])), int(np.floor(second.bbox[1])), 0)
    x2 = min(
        int(np.ceil(first.bbox[2])),
        int(np.ceil(second.bbox[2])),
        first.mask.shape[1],
    )
    y2 = min(
        int(np.ceil(first.bbox[3])),
        int(np.ceil(second.bbox[3])),
        first.mask.shape[0],
    )
    if x2 <= x1 or y2 <= y1:
        return 0.0
    intersection = int(
        np.logical_and(
            first.mask[y1:y2, x1:x2],
            second.mask[y1:y2, x1:x2],
        ).sum(dtype=np.int64)
    )
    if intersection <= 0:
        return 0.0
    union = int(first_area) + int(second_area) - intersection
    return float(intersection / union) if union > 0 else 0.0


def deduplicate_proposals(
    proposals: Sequence[SupplementalProposal],
    *,
    duplicate_mask_iou: float = 0.8,
    max_proposals: int = 128,
) -> List[SupplementalProposal]:
    """Score-ordered, deterministic cross-prompt mask-IoU deduplication."""

    if (
        not np.isfinite(duplicate_mask_iou)
        or not 0.0 <= duplicate_mask_iou <= 1.0
    ):
        raise ValueError("duplicate_mask_iou must be in [0, 1]")
    if max_proposals < 1:
        raise ValueError("max_proposals must be positive")
    values = list(proposals)
    for index, proposal in enumerate(values):
        if not isinstance(proposal, SupplementalProposal):
            raise TypeError(
                f"proposals[{index}] must be a SupplementalProposal"
            )
    indexed = sorted(
        enumerate(values),
        key=lambda pair: (-float(pair[1].score), pair[0]),
    )
    kept: List[SupplementalProposal] = []
    kept_areas: List[int] = []
    for _, proposal in indexed:
        area = int(proposal.mask.sum(dtype=np.int64))
        duplicate = any(
            _mask_iou(proposal, prior, area, prior_area)
            > float(duplicate_mask_iou)
            for prior, prior_area in zip(kept, kept_areas)
        )
        if duplicate:
            continue
        kept.append(proposal)
        kept_areas.append(area)
        if len(kept) >= int(max_proposals):
            break
    return kept


def store_frame_proposals(
    cache: NpzProposalCache,
    namespace: str,
    scene_id: str,
    raw_frame_id: int | str,
    image: np.ndarray,
    proposals: Sequence[SupplementalProposal],
) -> Tuple[str, Path]:
    """Write one StrictCache-compatible frame using the public cache API."""

    if not isinstance(cache, NpzProposalCache):
        raise TypeError("cache must be an NpzProposalCache")
    if not _SCENE_RE.fullmatch(scene_id):
        raise ValueError(f"Invalid ScanNet scene id: {scene_id!r}")
    rgb = np.asarray(image)
    frame_id = f"{scene_id}:{raw_frame_id}"
    key = proposal_cache_key(namespace, frame_id, rgb)
    cache.store(key, proposals, image_shape=rgb.shape[:2])
    return key, cache.path_for_key(key)


def observer_provider_calls(
    observer_log_root: os.PathLike[str] | str,
    scene_id: str,
) -> int:
    """Read the final per-scene provider call count from an observer log."""

    root = Path(observer_log_root)
    candidates = (root / f"{scene_id}.log", root / "scenes" / f"{scene_id}.log")
    log_path = next((path for path in candidates if path.is_file()), None)
    if log_path is None:
        raise FileNotFoundError(
            f"Missing observer scene log for {scene_id} under {root}"
        )
    matches = _PROVIDER_CALL_RE.findall(
        log_path.read_text(encoding="utf-8", errors="replace")
    )
    if not matches:
        raise ValueError(f"No provider_calls summary in {log_path}")
    return int(matches[-1])


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Optional[Path] = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=str(path.parent),
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(str(temporary), str(path))
    except Exception:
        if temporary is not None:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass
        raise


class Sam3Teacher:
    """Small lazy adapter around the repository's official SAM3 image API."""

    def __init__(
        self,
        *,
        sam3_root: Path,
        checkpoint: Path,
        bpe_path: Optional[Path],
        device: str,
        confidence_threshold: float,
        resolution: int,
        precision: str,
        mask_threshold: float,
        min_mask_pixels: int,
        max_per_class: int,
        prompts: Sequence[str],
    ) -> None:
        try:
            import torch
            from PIL import Image
        except ImportError as error:
            raise RuntimeError("torch and Pillow are required for SAM3") from error
        sam3_path = str(sam3_root.resolve())
        if sam3_path not in sys.path:
            sys.path.insert(0, sam3_path)
        try:
            from sam3.model.sam3_image_processor import Sam3Processor
            from sam3.model_builder import build_sam3_image_model
        except ImportError as error:
            raise RuntimeError(
                f"Could not import the existing SAM3 API from {sam3_root}"
            ) from error

        requested_device = torch.device(device)
        if requested_device.type != "cuda":
            raise ValueError("SAM3 teacher generation currently requires CUDA")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable in the SAM3 process")
        if precision == "bf16" and not torch.cuda.is_bf16_supported():
            raise RuntimeError("The selected CUDA device does not support bfloat16")

        # The local SAM3 builder checks for the literal string "cuda" before
        # moving the model.  Move it explicitly afterwards so cuda:0 remains
        # correct inside a CUDA_VISIBLE_DEVICES worker.
        model = build_sam3_image_model(
            bpe_path=(None if bpe_path is None else str(bpe_path)),
            device="cuda",
            eval_mode=True,
            checkpoint_path=str(checkpoint),
            load_from_HF=False,
            enable_segmentation=True,
            enable_inst_interactivity=False,
            compile=False,
        )
        model = model.to(requested_device).eval()
        self.processor = Sam3Processor(
            model,
            resolution=int(resolution),
            device=str(requested_device),
            confidence_threshold=float(confidence_threshold),
        )
        self.torch = torch
        self.Image = Image
        self.device = requested_device
        self.precision = precision
        self.mask_threshold = float(mask_threshold)
        self.min_mask_pixels = int(min_mask_pixels)
        self.max_per_class = int(max_per_class)
        self.prompts = tuple(prompts)

    def predict(self, image: np.ndarray) -> List[SupplementalProposal]:
        pil_image = self.Image.fromarray(np.asarray(image, dtype=np.uint8), "RGB")
        autocast = (
            self.torch.autocast(
                device_type="cuda",
                dtype=self.torch.bfloat16,
            )
            if self.precision == "bf16"
            else contextlib.nullcontext()
        )
        proposals: List[SupplementalProposal] = []
        with self.torch.inference_mode(), autocast:
            state = self.processor.set_image(pil_image)
            for prompt in self.prompts:
                output = self.processor.set_text_prompt(
                    state=state,
                    prompt=prompt,
                )
                proposals.extend(
                    normalize_sam3_output(
                        output,
                        label=prompt,
                        image_shape=image.shape[:2],
                        mask_threshold=self.mask_threshold,
                        min_mask_pixels=self.min_mask_pixels,
                        max_per_class=self.max_per_class,
                    )
                )
        return proposals


def _parse_prompts(values: Optional[Sequence[str]]) -> Tuple[str, ...]:
    if not values:
        return SCANNET18_PROMPTS
    prompts: List[str] = []
    for value in values:
        prompts.extend(part.strip() for part in value.split(","))
    prompts = [prompt for prompt in prompts if prompt]
    if not prompts:
        raise ValueError("--prompts resolved to an empty list")
    if len(set(prompts)) != len(prompts):
        raise ValueError("--prompts must not contain duplicates")
    return tuple(prompts)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate an offline ScanNet SAM3 proposal cache compatible with "
            "StrictCacheProposalProvider."
        )
    )
    parser.add_argument(
        "--scene-list",
        type=Path,
        default=TOOL_ROOT
        / "evaluation/data_util/meta_data/scannetv2_val_ablation10_even.txt",
    )
    parser.add_argument(
        "--frames-root",
        type=Path,
        default=Path(
            "/data/ZhaoX/BoxFusion/upstream_clean/scannet_readme_frames"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=TOOL_ROOT / "cache/sam3_scannet18_ablation10_v1",
    )
    parser.add_argument(
        "--namespace",
        default="sam3-scannet18-gap25-interval5-v1",
        help="Immutable cache namespace; changing inference settings requires a new value.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Local facebook/sam3 image checkpoint (sam3.pt).",
    )
    parser.add_argument("--bpe-path", type=Path, default=None)
    parser.add_argument(
        "--sam3-root",
        type=Path,
        default=TOOL_ROOT.parent
        / "third_party/WildDet3D/third_party/sam3",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--precision", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument("--resolution", type=int, default=1008)
    parser.add_argument("--confidence-threshold", type=float, default=0.35)
    parser.add_argument("--mask-threshold", type=float, default=0.50)
    parser.add_argument("--duplicate-mask-iou", type=float, default=0.80)
    parser.add_argument("--min-mask-pixels", type=int, default=64)
    parser.add_argument("--max-per-class", type=int, default=32)
    parser.add_argument("--max-proposals", type=int, default=128)
    parser.add_argument("--prompts", action="append", default=None)
    parser.add_argument("--gap", type=int, default=25)
    parser.add_argument("--proposal-interval", type=int, default=5)
    parser.add_argument("--configured-height", type=int, default=480)
    parser.add_argument("--configured-width", type=int, default=640)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse only entries whose strict RGB-bound cache key and shape match.",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Fail on any missing/incompatible cache entry without loading SAM3.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Limit scheduled frames in this shard (for a smoke test).",
    )
    parser.add_argument(
        "--observer-log-root",
        type=Path,
        default=None,
        help="Require each simulated provider count to match an observer scene log.",
    )
    parser.add_argument(
        "--runtime-rgb-dir",
        type=Path,
        default=None,
        help=(
            "Lossless runtime RGB staging root. Actual SAM3 generation and "
            "strict verification require this OpenCV-4.6-exported staging."
        ),
    )
    parser.add_argument(
        "--stage-runtime-rgb",
        action="store_true",
        help=(
            "Decode and orient ScanNet RGB with the BoxFusion runtime, write "
            "lossless NPY files plus an immutable manifest, then exit."
        ),
    )
    parser.add_argument(
        "--expected-provider-calls",
        type=int,
        default=None,
        help="Require the pre-limit provider-call total for this shard.",
    )
    parser.add_argument("--metadata-path", type=Path, default=None)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Audit scheduling/preprocessing/cache keys without loading SAM3 or writing NPZ files.",
    )
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    if not args.namespace.strip():
        raise ValueError("--namespace must be non-empty")
    for name in (
        "resolution",
        "min_mask_pixels",
        "max_per_class",
        "max_proposals",
        "gap",
        "proposal_interval",
        "configured_height",
        "configured_width",
        "num_shards",
    ):
        if int(getattr(args, name)) < 1:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.shard_index < 0 or args.shard_index >= args.num_shards:
        raise ValueError("--shard-index must satisfy 0 <= index < num_shards")
    if args.max_frames is not None and args.max_frames < 1:
        raise ValueError("--max-frames must be positive")
    selected_modes = sum(
        int(value)
        for value in (args.verify_only, args.dry_run, args.stage_runtime_rgb)
    )
    if selected_modes > 1:
        raise ValueError(
            "--verify-only, --dry-run and --stage-runtime-rgb are mutually "
            "exclusive"
        )
    if (
        args.expected_provider_calls is not None
        and args.expected_provider_calls < 0
    ):
        raise ValueError("--expected-provider-calls must be non-negative")
    for name in (
        "confidence_threshold",
        "mask_threshold",
        "duplicate_mask_iou",
    ):
        value = float(getattr(args, name))
        if not np.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError(f"--{name.replace('_', '-')} must be in [0, 1]")
    if not args.frames_root.is_dir():
        raise FileNotFoundError(f"Missing ScanNet frames root: {args.frames_root}")
    if (
        not (args.verify_only or args.stage_runtime_rgb)
        and not args.sam3_root.is_dir()
    ):
        raise FileNotFoundError(f"Missing existing SAM3 source: {args.sam3_root}")
    if not (args.verify_only or args.stage_runtime_rgb) and (
        args.checkpoint is None or not args.checkpoint.is_file()
    ):
        raise FileNotFoundError(
            f"Missing local SAM3 checkpoint: {args.checkpoint}"
        )
    if (
        not (args.verify_only or args.stage_runtime_rgb)
        and args.bpe_path is not None
        and not args.bpe_path.is_file()
    ):
        raise FileNotFoundError(f"Missing SAM3 BPE vocabulary: {args.bpe_path}")
    if args.stage_runtime_rgb and args.runtime_rgb_dir is None:
        raise ValueError("--stage-runtime-rgb requires --runtime-rgb-dir")
    if (
        not (args.stage_runtime_rgb or args.dry_run)
        and args.runtime_rgb_dir is None
    ):
        raise ValueError(
            "SAM3 generation/verification requires --runtime-rgb-dir"
        )


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    _validate_args(args)
    prompts = _parse_prompts(args.prompts)
    scenes = read_scene_list(args.scene_list)
    selected_scenes = [
        (index, scene)
        for index, scene in enumerate(scenes)
        if index % args.num_shards == args.shard_index
    ]
    if not selected_scenes:
        raise ValueError("This shard contains no ScanNet scenes")

    scene_sources: Dict[str, Dict[str, Any]] = {}
    full_schedule: List[Tuple[str, int]] = []
    scene_audit: List[Dict[str, Any]] = []
    for source_index, scene in selected_scenes:
        source = discover_scannet_scene(args.frames_root, scene)
        indices = scheduled_frame_indices(
            source["frame_count"],
            gap=args.gap,
            proposal_interval=args.proposal_interval,
        )
        expected = None
        if args.observer_log_root is not None:
            expected = observer_provider_calls(args.observer_log_root, scene)
            if expected != len(indices):
                raise ValueError(
                    f"{scene} schedule mismatch: simulated={len(indices)}, "
                    f"observer={expected}"
                )
        scene_sources[scene] = source
        full_schedule.extend((scene, frame_index) for frame_index in indices)
        scene_audit.append(
            {
                "scene_id": scene,
                "source_scene_list_index": int(source_index),
                "frame_count": int(source["frame_count"]),
                "scheduled_provider_calls": len(indices),
                "observer_provider_calls": expected,
                "scheduled_frame_indices": indices,
            }
        )

    if (
        args.expected_provider_calls is not None
        and len(full_schedule) != args.expected_provider_calls
    ):
        raise ValueError(
            "Provider-call total mismatch: "
            f"simulated={len(full_schedule)}, "
            f"expected={args.expected_provider_calls}"
        )
    schedule = (
        full_schedule
        if args.max_frames is None
        else full_schedule[: args.max_frames]
    )
    if args.stage_runtime_rgb:
        stage_runtime_rgb(
            args=args,
            scenes=scenes,
            selected_scenes=selected_scenes,
            scene_sources=scene_sources,
            full_schedule=full_schedule,
            schedule=schedule,
        )
        return 0

    runtime_manifest_path: Optional[Path] = None
    runtime_manifest: Optional[Dict[str, Any]] = None
    runtime_records: Dict[str, Mapping[str, Any]] = {}
    if not args.dry_run:
        (
            runtime_manifest_path,
            runtime_manifest,
            runtime_records,
        ) = load_runtime_rgb_manifest(
            args=args,
            scenes=scenes,
            selected_scenes=selected_scenes,
            full_schedule=full_schedule,
            schedule=schedule,
        )

    output_dir = args.output_dir.resolve()
    metadata_path = (
        args.metadata_path.resolve()
        if args.metadata_path is not None
        else output_dir
        / "manifests"
        / (
            f"sam3_teacher_shard_{args.shard_index:03d}"
            f"_of_{args.num_shards:03d}.json"
        )
    )
    checkpoint_record = None
    if args.checkpoint is not None:
        checkpoint_stat = args.checkpoint.stat()
        checkpoint_record = {
            "path": str(args.checkpoint.resolve()),
            "size_bytes": int(checkpoint_stat.st_size),
            "mtime_ns": int(checkpoint_stat.st_mtime_ns),
            "sha256": (
                None
                if args.verify_only
                else _sha256_file(args.checkpoint)
            ),
        }
    manifest: Dict[str, Any] = {
        "schema": _MANIFEST_SCHEMA,
        "complete": False,
        "created_unix_seconds": time.time(),
        "tool": str(Path(__file__).resolve()),
        "scene_list": {
            "path": str(args.scene_list.resolve()),
            "sha256": _sha256_text_file(args.scene_list),
            "all_scene_count": len(scenes),
            "selected_scene_count": len(selected_scenes),
        },
        "frames_root": str(args.frames_root.resolve()),
        "runtime_rgb": (
            None
            if runtime_manifest_path is None
            else {
                "directory": str(args.runtime_rgb_dir.resolve()),
                "manifest_path": str(runtime_manifest_path),
                "manifest_sha256": _sha256_file(runtime_manifest_path),
                "exporter": runtime_manifest.get("exporter"),
            }
        ),
        "output_dir": str(output_dir),
        "namespace": args.namespace.strip(),
        "checkpoint": checkpoint_record,
        "bpe_path": (
            None
            if args.bpe_path is None
            else {
                "path": str(args.bpe_path.resolve()),
                "sha256": _sha256_file(args.bpe_path),
            }
        ),
        "sam3_root": str(args.sam3_root.resolve()),
        "prompts": list(prompts),
        "settings": {
            "device": args.device,
            "precision": args.precision,
            "resolution": args.resolution,
            "confidence_threshold": args.confidence_threshold,
            "mask_threshold": args.mask_threshold,
            "duplicate_mask_iou": args.duplicate_mask_iou,
            "min_mask_pixels": args.min_mask_pixels,
            "max_per_class": args.max_per_class,
            "max_proposals": args.max_proposals,
            "gap": args.gap,
            "proposal_interval": args.proposal_interval,
            "configured_height": args.configured_height,
            "configured_width": args.configured_width,
            "resume": args.resume,
            "verify_only": args.verify_only,
            "dry_run": args.dry_run,
            "max_frames": args.max_frames,
        },
        "shard": {
            "index": args.shard_index,
            "count": args.num_shards,
            "pre_limit_provider_calls": len(full_schedule),
            "executed_provider_calls": len(schedule),
        },
        "scenes": [
            {
                key: value
                for key, value in row.items()
                if key != "observer_provider_calls"
            }
            for row in scene_audit
        ],
        "frames": [],
        "summary": {},
    }

    # The content key intentionally does not encode detector settings: the
    # explicit namespace represents that provenance.  Guard the namespace
    # contract with an immutable per-shard sidecar so --resume cannot mix
    # checkpoints, prompts, thresholds, schedules, or precision modes.
    provenance = {
        "schema": "boxfusion_scannet_sam3_teacher_provenance_v1",
        "scene_list": manifest["scene_list"],
        "selected_scene_ids": [scene for _, scene in selected_scenes],
        "frames_root": manifest["frames_root"],
        "runtime_rgb": manifest["runtime_rgb"],
        "namespace": manifest["namespace"],
        "checkpoint": manifest["checkpoint"],
        "bpe_path": manifest["bpe_path"],
        "sam3_root": manifest["sam3_root"],
        "prompts": manifest["prompts"],
        "settings": {
            key: value
            for key, value in manifest["settings"].items()
            if key not in {"resume", "verify_only", "dry_run", "max_frames"}
        },
        "shard": {
            "index": args.shard_index,
            "count": args.num_shards,
            "pre_limit_provider_calls": len(full_schedule),
        },
        "scenes": scene_audit,
    }
    provenance_path = (
        output_dir
        / "manifests"
        / (
            f"provenance_shard_{args.shard_index:03d}"
            f"_of_{args.num_shards:03d}.json"
        )
    )
    if not (args.verify_only or args.dry_run):
        if provenance_path.exists():
            previous = json.loads(
                provenance_path.read_text(encoding="utf-8")
            )
            if previous != provenance:
                raise ValueError(
                    "Refusing to mix incompatible SAM3 teacher cache "
                    f"provenance: {provenance_path}"
                )
            if not args.resume:
                raise FileExistsError(
                    f"{provenance_path} exists; use --resume only for the "
                    "same immutable configuration or choose a fresh output"
                )
        elif args.resume:
            raise FileNotFoundError(
                "Cannot safely resume without immutable provenance: "
                f"{provenance_path}"
            )
        else:
            _write_json_atomic(provenance_path, provenance)

    cache = NpzProposalCache(
        output_dir,
        write_enabled=not (args.dry_run or args.verify_only),
    )
    teacher: Optional[Sam3Teacher] = None
    status_counts: Counter[str] = Counter()
    orientation_counts: Counter[str] = Counter()
    proposal_total = 0
    started = time.time()
    print(
        "ScanNet SAM3 teacher cache | "
        f"scenes={len(selected_scenes)}, frames={len(schedule)}, "
        f"shard={args.shard_index}/{args.num_shards}, "
        f"namespace={args.namespace}"
    )

    for position, (scene, frame_index) in enumerate(schedule, start=1):
        source = scene_sources[scene]
        color_path = source["color_paths"][frame_index]
        depth_path = source["depth_paths"][frame_index]
        pose_path = source["pose_paths"][frame_index]
        logical_frame_id = f"{scene}:{frame_index}"
        if args.dry_run:
            image, orientation_index = load_scannet_rgb(
                color_path,
                depth_path,
                source["poses"][frame_index],
                configured_height=args.configured_height,
                configured_width=args.configured_width,
            )
        else:
            image, orientation_index = load_staged_runtime_rgb(
                runtime_rgb_dir=args.runtime_rgb_dir,
                record=runtime_records[logical_frame_id],
                namespace=args.namespace.strip(),
                logical_frame_id=logical_frame_id,
                source_paths={
                    "color": color_path,
                    "depth": depth_path,
                    "pose": pose_path,
                },
            )
        key = proposal_cache_key(args.namespace.strip(), logical_frame_id, image)
        cache_path = cache.path_for_key(key)
        status = "dry_run"
        proposals: List[SupplementalProposal] = []

        cached = cache.load(
            key, expected_image_shape=image.shape[:2]
        ) if (args.resume or args.verify_only) else None
        if args.verify_only and cached is None:
            raise FileNotFoundError(
                "Missing or image-incompatible strict SAM3 teacher cache "
                f"entry for {logical_frame_id}: {cache_path}"
            )
        if cached is not None:
            proposals = cached
            status = "verified" if args.verify_only else "resumed"
        elif cache_path.exists() and not args.resume:
            raise FileExistsError(
                f"Refusing existing cache entry without --resume: {cache_path}"
            )
        elif not args.dry_run:
            if teacher is None:
                teacher = Sam3Teacher(
                    sam3_root=args.sam3_root,
                    checkpoint=args.checkpoint,
                    bpe_path=args.bpe_path,
                    device=args.device,
                    confidence_threshold=args.confidence_threshold,
                    resolution=args.resolution,
                    precision=args.precision,
                    mask_threshold=args.mask_threshold,
                    min_mask_pixels=args.min_mask_pixels,
                    max_per_class=args.max_per_class,
                    prompts=prompts,
                )
            raw_proposals = teacher.predict(image)
            proposals = deduplicate_proposals(
                raw_proposals,
                duplicate_mask_iou=args.duplicate_mask_iou,
                max_proposals=args.max_proposals,
            )
            stored_key, stored_path = store_frame_proposals(
                cache,
                args.namespace.strip(),
                scene,
                frame_index,
                image,
                proposals,
            )
            if stored_key != key or stored_path != cache_path:
                raise RuntimeError("Cache writer and audit key/path disagree")
            status = "written"

        label_counts = Counter(
            proposal.label or "<none>" for proposal in proposals
        )
        status_counts[status] += 1
        orientation_name = _ORIENTATION_NAMES[orientation_index]
        orientation_counts[orientation_name] += 1
        proposal_total += len(proposals)
        frame_record = {
            "scene_id": scene,
            "frame_index": int(frame_index),
            "source_color": str(color_path.resolve()),
            "source_depth": str(depth_path.resolve()),
            "source_pose": str(pose_path.resolve()),
            "logical_frame_id": logical_frame_id,
            "orientation": orientation_name,
            "image_shape": list(image.shape),
            "image_dtype": image.dtype.str,
            "runtime_rgb_content_sha256": _rgb_content_sha256(image),
            "cache_key": key,
            "cache_path": str(cache_path),
            "status": status,
            "proposal_count": len(proposals),
            "label_counts": dict(sorted(label_counts.items())),
            "score_min": (
                None if not proposals else min(float(p.score) for p in proposals)
            ),
            "score_max": (
                None if not proposals else max(float(p.score) for p in proposals)
            ),
        }
        manifest["frames"].append(frame_record)
        print(
            f"[{position}/{len(schedule)}] {logical_frame_id} "
            f"orientation={orientation_name} proposals={len(proposals)} "
            f"status={status}"
        )

    manifest["complete"] = True
    manifest["completed_unix_seconds"] = time.time()
    manifest["summary"] = {
        "duration_seconds": time.time() - started,
        "status_counts": dict(sorted(status_counts.items())),
        "orientation_counts": dict(sorted(orientation_counts.items())),
        "proposal_total": int(proposal_total),
        "cache_files_expected": int(len(schedule)),
    }
    _write_json_atomic(metadata_path, manifest)
    print(f"Audit manifest: {metadata_path}")
    print(
        "SAM3 teacher cache complete | "
        f"frames={len(schedule)}, proposals={proposal_total}, "
        f"status={dict(status_counts)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
