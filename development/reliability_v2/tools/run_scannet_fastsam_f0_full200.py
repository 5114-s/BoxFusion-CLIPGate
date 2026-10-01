#!/usr/bin/env python3
"""Run frozen FastSAM-x residual automatic masks as an inert ScanNet shadow.

F0 is deliberately a capacity experiment, not a detector variant.  A safe
``torch.load`` deserializes each complete sealed gap-25 CuTR payload, but F0
indexes and uses only its current-frame ``pred_boxes`` field.  It then runs a
frozen class-agnostic FastSAM provider on the current BGR frame and passes its
masks to :mod:`boxfusion.fastsam_residual_shadow`.  The executable writes
auditable JSON sidecars only: it has no tracking, birth, semantic, CLIP,
terminal-native-prediction, annotation, evaluator, or training surface.

The exact 200-scene list is deterministically sharded by original list index.
Each scene resolves to exactly one of the two immutable CuTR-v2 schedule roots.
Scene receipts and shard manifests are published atomically with create-only
semantics so concurrent shards and crash-safe ``--resume`` are possible.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict, is_dataclass
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import resource
import sys
import tempfile
import time
import platform
from functools import lru_cache
from typing import Any, Callable, Mapping, Sequence

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if os.fspath(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, os.fspath(REPOSITORY_ROOT))

from boxfusion import fastsam_residual_shadow as f0_core  # noqa: E402


PROTOCOL_ID = "F0-frozen-FastSAM-x-residual-automatic-mask-shadow-full200"
SCENE_SCHEMA = "boxfusion.scannet_fastsam_f0_full200.scene.v1"
SHARD_SCHEMA = "boxfusion.scannet_fastsam_f0_full200.shard.v1"
EXPECTED_CACHE_SCHEMA = "boxfusion.cutr_postfilter_cache.v2"
EXPECTED_CACHE_NAMESPACE = "scannet-score05-gap25-postfilter-v2"
EXPECTED_DELTA_CACHE_NAMESPACE = "cutr-score05-gap25-v2-extra100"
EXPECTED_CACHE_NAMESPACES = frozenset(
    (EXPECTED_CACHE_NAMESPACE, EXPECTED_DELTA_CACHE_NAMESPACE)
)
OLD100_PRODUCER_FINGERPRINT = (
    "ba44e29386d2c2f76bb927e00f02b62cfc5ee4f188a94408c32ce91757f4462d"
)
# Exact full-extra100 build-plan fingerprints for the accepted 2/4/8-way
# deterministic shardings.  The tuple is (num_shards, shard_index); a delta
# scene must also occupy that shard in the sealed extra100 scene order.
DELTA_PRODUCER_FINGERPRINTS: Mapping[str, tuple[int, int]] = {
    # Current sparse-reader full-extra100 two-way build plans.
    "1589802fd762b69015f6fd06f8ad88826888874540be7b8ad4b27b2d566cd316": (2, 0),
    "af2314cba8b47f43d1379655d6a9b809a155108fa2a69791ec500c5ac86f5a34": (2, 1),
    # Earlier frozen dense-reader plans retained for the already-sealed scenes.
    "ded58bd1ef087fd9d5e40ee040b0c74293786f189aa64ff8ea84fb3bae04f1c2": (2, 0),
    "ee6816e93a5171a1c528f96145e4d0699611538edcc28730e6e1909521cfa592": (2, 1),
    "1452e65a32e411df84e442cbd68152798fe5d66d3744fdae389e55f0f41b7f19": (4, 0),
    "459bd2b68507c83ff7f4bd07a42494c85ac41711a13683e9ffa3fe24481cc2b6": (4, 1),
    "7a277aef08ef98d4cc8ab934bffa63c34fd536816570f78751faf7d3e93ad223": (4, 2),
    "675bfa770d4ddbc780af4379b123e2efe16280b245e5d82fb039425ea5d50a70": (4, 3),
    "0a5444c55adec2688fb8f6d6ed94edeea50ab0626e1d083400d0d36ff2e79ac2": (8, 0),
    "caedd46b2f29dce9093bf00688ca35425a9d14c72c273970e596515b9a64df93": (8, 1),
    "fcd0fb2a140b17b0fb051ba3709a41a473a70a067964cc59003a59375dc2eb86": (8, 2),
    "15af86b0788078764373035b4672f9d10504ff9f7b81693cfe460a374e95a1fc": (8, 3),
    "7b458adb99904bb0a7a7531f93d040ce70566fc0233c971411c1a01242ce1f8c": (8, 4),
    "2de76023e4587a6c89bf9b6806ae01ca34caacfc9628deee88e61c5ed93114d5": (8, 5),
    "77bd1609de0569dca2d2a9eba0688ab92130b91cdb6a23ec5fcff90dbc0e40bc": (8, 6),
    "f09c678c411a526ffc7233e322b2308dc5218e475d7890a92a11daff0f57b095": (8, 7),
}
EXPECTED_SCENE_COUNT = 200
EXPECTED_FULL200_KEYFRAME_COUNT = 12_941
EXPECTED_SCENE_LIST_SHA256 = (
    "0e7e722d3e93ec4b721f12293a3f1e98ca62d475b42cc8b9d491878a897e9bd1"
)
EXPECTED_CHECKPOINT_SHA256 = (
    "c0be4e7ddbe4c15333d15a859c676d053c486d0a746a3be6a7a9790d52a9b6d7"
)
EXPECTED_TORCH_VERSION = "2.6.0+cu124"
EXPECTED_TORCH_CUDA_VERSION = "12.4"
EXPECTED_OPENCV_VERSION = "4.6.0"
EXPECTED_ULTRALYTICS_VERSION = "8.4.105"
EXPECTED_GPU_NAME = "NVIDIA GeForce RTX 3090"
EXPECTED_COMPUTE_CAPABILITY = (8, 6)
EXPECTED_GPU_UUID_BY_LOGICAL_DEVICE: Mapping[str, str] = {
    "cuda:0": "GPU-97755ff7-98ad-196d-1250-21eb5c95149d",
    "cuda:1": "GPU-2715f5df-abd1-cb90-a32b-770881114397",
}
IMAGE_HEIGHT = 480
IMAGE_WIDTH = 640
PROVIDER_MAX_DET = 100
SHARD_WARMUP_SUCCESSFUL_CALLS = 3
EXPECTED_NON_UPRIGHT_KEYFRAMES: Mapping[tuple[str, int], int] = {
    ("scene0246_00", 1900): 1,  # LEFT, producer torch.rot90 k=-1
    ("scene0426_00", 2200): 3,  # RIGHT, producer torch.rot90 k=+1
}
EXPECTED_EXECUTION_CENSUS_SHA256 = (
    "c306d37296b3dcbea7266202eb0ca86482cf32175f7909c0b1d97ea696e46b53"
)
# Exact current-pose/orientation execution ledger for the production one- or
# two-way deterministic full200 split.  Rewarm calls are deliberately absent.
EXPECTED_EXECUTION_COUNTS: Mapping[tuple[int, int], Mapping[str, int]] = {
    (1, 0): {
        "keyframes": 12_941,
        "invalid_pose_frames": 229,
        "non_upright_producer_frames": 2,
        "successful_frames": 12_710,
    },
    (2, 0): {
        "keyframes": 6_460,
        "invalid_pose_frames": 121,
        "non_upright_producer_frames": 0,
        "successful_frames": 6_339,
    },
    (2, 1): {
        "keyframes": 6_481,
        "invalid_pose_frames": 108,
        "non_upright_producer_frames": 2,
        "successful_frames": 6_371,
    },
}

DEFAULT_SCENE_LIST = (
    REPOSITORY_ROOT / "evaluation/data_util/meta_data/scannetv2_val_f0_full200.txt"
)
DEFAULT_SCENE_ROOT = Path("/extra/ZhaoX/scannet_data/scans")
DEFAULT_CHECKPOINT = Path(
    "/data/ZhaoX/RoboFusion/RoboFusion-master/focalconvsamfusion/OpenPCDet/"
    "checkpoints/FastSAM.pt"
)
DEFAULT_SCHEDULE_ROOTS = (
    Path(
        "/data/ZhaoX/OVM3D-Dett/boxfusion_boxer_dev/cache/cutr_proposals/"
        "scannet-score05-gap25-postfilter-v2"
    ),
    REPOSITORY_ROOT
    / "cache/f0_fastsam_full200/cutr-score05-gap25-v2-extra100",
)
DEFAULT_OUTPUT_ROOT = REPOSITORY_ROOT / "logs/scannet_fastsam_f0_full200_score05"

MINIMUM_FULL200_ACCEPTED_LIFTS = 1_500
MINIMUM_FULL200_CANDIDATE_SCENES = 160
MAXIMUM_CAP_SATURATION_RATIO = 0.25


class F0RunnerError(RuntimeError):
    """Raised when an immutable input or F0 shadow contract is violated."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_json_sha256(value: object) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _regular_file(path: Path, label: str) -> Path:
    if path.is_symlink() or not path.is_file():
        raise F0RunnerError(f"{label} must be a regular non-symlink file: {path}")
    return path.resolve()


def _read_json(path: Path, label: str) -> dict[str, Any]:
    source = _regular_file(path, label)
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise F0RunnerError(f"invalid {label}: {source}") from error
    if not isinstance(value, dict):
        raise F0RunnerError(f"{label} must contain a JSON object: {source}")
    return value


def _read_scene_list(path: Path, expected_scene_count: int) -> tuple[str, ...]:
    source = _regular_file(path, "F0 scene list")
    rows = tuple(
        line.strip()
        for line in source.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    )
    if len(rows) != expected_scene_count or len(set(rows)) != len(rows):
        raise F0RunnerError(
            f"expected {expected_scene_count} unique scenes, found {len(rows)}"
        )
    if any("/" in row or row in {".", ".."} for row in rows):
        raise F0RunnerError("scene list contains an unsafe scene identifier")
    return rows


def _validate_shard(shard_index: int, num_shards: int) -> None:
    if num_shards < 1:
        raise F0RunnerError("num-shards must be positive")
    if shard_index < 0 or shard_index >= num_shards:
        raise F0RunnerError("shard-index must be in [0,num-shards)")


@lru_cache(maxsize=1)
def _sealed_extra100_scene_indices() -> Mapping[str, int]:
    rows = _read_scene_list(DEFAULT_SCENE_LIST, EXPECTED_SCENE_COUNT)
    if _sha256(DEFAULT_SCENE_LIST) != EXPECTED_SCENE_LIST_SHA256:
        raise F0RunnerError("sealed extra100 scene-list SHA-256 differs")
    return {scene: index for index, scene in enumerate(rows[100:])}


def _validate_producer_fingerprint(
    namespace: str, scene: str, fingerprint: object
) -> None:
    if not isinstance(fingerprint, str) or len(fingerprint) != 64:
        raise F0RunnerError(f"CuTR producer fingerprint is invalid for {scene}")
    if namespace == EXPECTED_CACHE_NAMESPACE:
        if fingerprint != OLD100_PRODUCER_FINGERPRINT:
            raise F0RunnerError(f"old100 CuTR producer fingerprint differs for {scene}")
        return
    shard = DELTA_PRODUCER_FINGERPRINTS.get(fingerprint)
    scene_index = _sealed_extra100_scene_indices().get(scene)
    if shard is None or scene_index is None or scene_index % shard[0] != shard[1]:
        raise F0RunnerError(f"delta CuTR producer plan differs for {scene}")


def _read_schedule_manifest(path: Path, scene: str, root: Path) -> dict[str, Any]:
    manifest = _read_json(path, f"CuTR schedule manifest {scene}")
    frames = manifest.get("recorded_frame_ids")
    records = manifest.get("records")
    if manifest.get("namespace") not in EXPECTED_CACHE_NAMESPACES:
        raise F0RunnerError(f"CuTR namespace differs for {scene}")
    if manifest.get("schema") != EXPECTED_CACHE_SCHEMA:
        raise F0RunnerError(f"CuTR manifest schema differs for {scene}")
    if manifest.get("scene_id") != scene:
        raise F0RunnerError(f"CuTR manifest scene identity differs for {scene}")
    _validate_producer_fingerprint(
        str(manifest["namespace"]), scene, manifest.get("producer_fingerprint")
    )
    schedule = manifest.get("schedule")
    if (
        not isinstance(schedule, dict)
        or isinstance(schedule.get("dataset_length"), bool)
        or not isinstance(schedule.get("dataset_length"), int)
        or schedule["dataset_length"] < 0
        or schedule.get("gap") != 25
        or schedule.get("terminal_policy") != "upstream_boxfusion_early_exit_v1"
    ):
        raise F0RunnerError(f"CuTR manifest schedule contract differs for {scene}")
    expected_frames = list(range(0, max(schedule["dataset_length"] - 25, 0), 25))
    if (
        not isinstance(frames, list)
        or not frames
        or any(isinstance(item, bool) or not isinstance(item, int) for item in frames)
        or frames != sorted(set(frames))
        or frames != expected_frames
        or any(right - left != 25 for left, right in zip(frames, frames[1:]))
        or not isinstance(records, list)
        or len(records) != len(frames)
        or manifest.get("record_count") != len(frames)
    ):
        raise F0RunnerError(f"invalid gap-25 CuTR schedule for {scene}")

    by_frame: dict[int, dict[str, Any]] = {}
    for frame_id, record in zip(frames, records):
        if not isinstance(record, dict) or record.get("frame_id") != frame_id:
            raise F0RunnerError(f"CuTR record order differs for {scene}")
        count = record.get("count")
        digest = record.get("sha256")
        protected = record.get("protected_hashes")
        input_signature = record.get("input_signature")
        if (
            isinstance(count, bool)
            or not isinstance(count, int)
            or count < 0
            or not isinstance(digest, str)
            or len(digest) != 64
            or not isinstance(protected, dict)
            or not isinstance(protected.get("pred_boxes"), str)
            or len(protected["pred_boxes"]) != 64
            or not isinstance(input_signature, dict)
            or set(input_signature)
            != {"camera_to_world", "depth", "depth_K", "image", "image_K"}
            or any(
                not isinstance(value, str) or len(value) != 64
                for value in input_signature.values()
            )
        ):
            raise F0RunnerError(f"invalid CuTR record receipt for {scene}/{frame_id}")
        by_frame[frame_id] = record
    if manifest.get("proposal_count") != sum(row["count"] for row in records):
        raise F0RunnerError(f"CuTR proposal total differs for {scene}")
    source = path.resolve()
    return {
        "scene_id": scene,
        "root": root.resolve(),
        "path": source,
        "sha256": _sha256(source),
        "manifest": manifest,
        "frames": tuple(frames),
        "records": by_frame,
    }


def _resolve_schedules(
    schedule_roots: Sequence[Path], scenes: Sequence[str]
) -> dict[str, dict[str, Any]]:
    roots = tuple(Path(root).resolve() for root in schedule_roots)
    if len(roots) < 1 or len(set(roots)) != len(roots):
        raise F0RunnerError("schedule roots must be a non-empty unique sequence")
    schedules: dict[str, dict[str, Any]] = {}
    for scene in scenes:
        matches = [root / scene / "manifest.json" for root in roots if (root / scene / "manifest.json").exists() or (root / scene / "manifest.json").is_symlink()]
        if len(matches) != 1:
            raise F0RunnerError(
                f"scene {scene} must resolve to exactly one schedule root, found {len(matches)}"
            )
        match = matches[0]
        schedules[scene] = _read_schedule_manifest(match, scene, match.parents[1])
    return schedules


def _tensor_sha256(value: Any) -> str:
    """Match the CuTR-v2 cache tensor hash without importing detector code."""

    try:
        import torch
    except ImportError as error:  # pragma: no cover - production dependency
        raise F0RunnerError("PyTorch is unavailable for CuTR cache verification") from error
    if not isinstance(value, torch.Tensor):
        raise F0RunnerError("CuTR pred_boxes must be a tensor")
    tensor = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode("utf-8"))
    digest.update(b"\0")
    digest.update(repr(tuple(tensor.shape)).encode("ascii"))
    digest.update(b"\0")
    digest.update(tensor.view(torch.uint8).numpy().tobytes(order="C"))
    return digest.hexdigest()


def _array_sha256(value: Any) -> str:
    """Mirror proposal_cache._array_sha256 for NumPy and torch inputs."""

    try:
        import torch
    except ImportError as error:  # pragma: no cover - production dependency
        raise F0RunnerError("PyTorch is unavailable for input binding") from error
    if isinstance(value, torch.Tensor):
        return _tensor_sha256(value)
    array = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("utf-8"))
    digest.update(b"\0")
    digest.update(repr(tuple(array.shape)).encode("ascii"))
    digest.update(b"\0")
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _producer_pose(
    scene_root: Path, scene: str, frame_id: int
) -> tuple[np.ndarray, int, Path]:
    """Reproduce ScannetDataset's past-raw-frame pose used only for cache binding.

    This producer pose is never supplied to F0 geometry when the current raw
    pose is invalid.  It exists solely to authenticate the already-built CuTR
    cache's five-field input signature.
    """

    pose_root = scene_root / scene / "pose"
    for source_frame_id in range(frame_id, -1, -1):
        path = _regular_file(
            pose_root / f"{source_frame_id}.txt", "CuTR producer pose frame"
        )
        try:
            value = np.loadtxt(path, dtype=np.float64).reshape(4, 4)
        except (OSError, ValueError) as error:
            raise F0RunnerError(
                f"invalid CuTR producer pose {scene}/{source_frame_id}"
            ) from error
        # The released loader tests only infinities (not general rigid-pose
        # validity) before carrying the latest raw-frame pose forward.
        if not np.isinf(value).any():
            return np.ascontiguousarray(value), source_frame_id, path
    raise F0RunnerError(f"CuTR producer has no prior finite pose for {scene}/{frame_id}")


def _producer_orientation(pose: np.ndarray) -> int:
    axes = np.asarray(
        [[0.0, -1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]],
        dtype=np.float64,
    )
    return int(np.argmax(axes @ np.asarray(pose, dtype=np.float64)[2, :3]))


def _rotated_image_intrinsic(intrinsic: Any, orientation: int) -> Any:
    """Mirror ImageMeasurementInfo.orient(..., UPRIGHT) exactly."""

    import torch

    matrix = intrinsic.detach().cpu().contiguous()
    if orientation == 0:
        return matrix.clone()
    if orientation in (1, 3):
        return torch.stack(
            (
                torch.stack((matrix[1, 1], matrix[0, 1], matrix[1, 2])),
                torch.stack((matrix[1, 0], matrix[0, 0], matrix[0, 2])),
                torch.stack((matrix[2, 0], matrix[2, 1], matrix[2, 2])),
            )
        ).contiguous()
    if orientation == 2:
        return torch.stack(
            (
                torch.stack((matrix[0, 0], matrix[0, 1], 640.0 - matrix[0, 2])),
                torch.stack((matrix[1, 0], matrix[1, 1], 480.0 - matrix[1, 2])),
                torch.stack((matrix[2, 0], matrix[2, 1], matrix[2, 2])),
            )
        ).contiguous()
    raise F0RunnerError("CuTR producer orientation is outside [0,3]")


def _reconstruct_cutr_input_signature(
    *,
    bgr: np.ndarray,
    depth_mm: np.ndarray,
    intrinsic: np.ndarray,
    scene_root: Path,
    scene: str,
    frame_id: int,
) -> tuple[dict[str, str], dict[str, Any]]:
    """Reconstruct all five proposal-cache inputs with producer dtypes/shapes."""

    try:
        import cv2
        import torch
    except ImportError as error:  # pragma: no cover - production dependency
        raise F0RunnerError("OpenCV/PyTorch unavailable for CuTR input binding") from error
    pose, pose_source_frame_id, pose_path = _producer_pose(
        scene_root, scene, frame_id
    )
    orientation = _producer_orientation(pose)
    rotation_k = {0: 0, 1: -1, 2: 2, 3: 1}[orientation]

    image_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    image_tensor = torch.from_numpy(
        np.ascontiguousarray(np.moveaxis(image_rgb, -1, 0))
    )
    depth_tensor = torch.from_numpy(
        np.ascontiguousarray(depth_mm.astype(np.float32) / 1000.0)
    ).float()
    if rotation_k:
        image_tensor = torch.rot90(image_tensor, rotation_k, dims=(-2, -1))
        depth_tensor = torch.rot90(depth_tensor, rotation_k, dims=(-2, -1))
    image = np.ascontiguousarray(np.moveaxis(image_tensor.numpy(), 0, -1))
    depth_tensor = depth_tensor.contiguous()
    intrinsic_tensor = torch.from_numpy(
        np.ascontiguousarray(intrinsic.astype(np.float32))
    ).float().contiguous()
    image_intrinsic = _rotated_image_intrinsic(intrinsic_tensor, orientation)
    pose_tensor = torch.from_numpy(
        np.ascontiguousarray(pose.astype(np.float32))
    ).float().contiguous()
    signature = {
        "image": _array_sha256(image),
        "depth": _array_sha256(depth_tensor),
        "image_K": _array_sha256(image_intrinsic),
        "depth_K": _array_sha256(intrinsic_tensor),
        "camera_to_world": _array_sha256(pose_tensor),
    }
    metadata = {
        "producer_orientation": orientation,
        "producer_rotation_k": rotation_k,
        "producer_image_shape": list(image.shape),
        "producer_depth_shape": list(depth_tensor.shape),
        "producer_pose_source_frame_id": pose_source_frame_id,
        "producer_pose_path": os.fspath(pose_path),
        "producer_pose_sha256": _sha256(pose_path),
    }
    return signature, metadata


def _load_cutr_boxes(
    schedule: Mapping[str, Any], scene: str, frame_id: int
) -> tuple[Path, np.ndarray, str, tuple[int, int], Mapping[str, str]]:
    record = schedule["records"][frame_id]
    path = _regular_file(
        Path(schedule["root"]) / scene / f"frame_{frame_id:06d}.pt",
        f"CuTR cache {scene}/{frame_id}",
    )
    digest = _sha256(path)
    if digest != record["sha256"]:
        raise F0RunnerError(f"CuTR cache hash changed for {scene}/{frame_id}")
    try:
        import torch

        payload = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as error:
        raise F0RunnerError(f"could not safely load CuTR cache {scene}/{frame_id}") from error
    if not isinstance(payload, dict) or payload.get("schema") != EXPECTED_CACHE_SCHEMA:
        raise F0RunnerError(f"CuTR cache schema differs for {scene}/{frame_id}")
    image_size = tuple(payload.get("image_size", ()))
    if image_size not in ((IMAGE_HEIGHT, IMAGE_WIDTH), (IMAGE_WIDTH, IMAGE_HEIGHT)):
        raise F0RunnerError(f"CuTR cache image size differs for {scene}/{frame_id}")
    fields = payload.get("fields")
    metadata = payload.get("field_metadata")
    if not isinstance(fields, dict) or not isinstance(metadata, dict):
        raise F0RunnerError(f"CuTR cache fields are invalid for {scene}/{frame_id}")
    if "pred_boxes" not in fields or not isinstance(metadata.get("pred_boxes"), dict):
        raise F0RunnerError(f"CuTR pred_boxes are missing for {scene}/{frame_id}")
    count = record["count"]
    if payload.get("count") != count:
        raise F0RunnerError(f"CuTR cache count differs for {scene}/{frame_id}")
    tensor_hash = _tensor_sha256(fields["pred_boxes"])
    # ``field_metadata`` hashes the serialized tensor itself.  The separate
    # protected-field hash has a detector-level canonicalization and is only
    # compared byte-for-byte between the payload and its sealed manifest.
    if (
        tensor_hash != metadata["pred_boxes"].get("sha256")
        or payload.get("protected_hashes", {}).get("pred_boxes")
        != record["protected_hashes"]["pred_boxes"]
    ):
        raise F0RunnerError(f"CuTR pred_boxes hash differs for {scene}/{frame_id}")
    for key in ("attempt_id", "input_signature", "geometry_sha256"):
        if payload.get(key) != record.get(key):
            raise F0RunnerError(f"CuTR {key} receipt differs for {scene}/{frame_id}")
    raw = fields["pred_boxes"]
    boxes = np.asarray(raw.detach().cpu().numpy(), dtype=np.float32)
    if (
        boxes.shape != (count, 4)
        or not np.isfinite(boxes).all()
        or (len(boxes) and (np.any(boxes[:, 2] <= boxes[:, 0]) or np.any(boxes[:, 3] <= boxes[:, 1])))
    ):
        raise F0RunnerError(f"invalid CuTR pred_boxes for {scene}/{frame_id}")
    return (
        path,
        np.ascontiguousarray(boxes),
        digest,
        image_size,
        dict(record["input_signature"]),
    )


def _load_intrinsic(scene_root: Path, scene: str) -> tuple[Path, np.ndarray]:
    path = _regular_file(
        scene_root / scene / "intrinsic/intrinsic_depth.txt",
        f"depth intrinsic {scene}",
    )
    try:
        value = np.loadtxt(path, dtype=np.float64)
    except (OSError, ValueError) as error:
        raise F0RunnerError(f"invalid depth intrinsic for {scene}") from error
    if value.shape != (4, 4):
        raise F0RunnerError(f"depth intrinsic must be 4x4 for {scene}")
    intrinsic = value[:3, :3]
    if (
        not np.isfinite(intrinsic).all()
        or intrinsic[0, 0] <= 0.0
        or intrinsic[1, 1] <= 0.0
        or not (0.0 <= intrinsic[0, 2] < IMAGE_WIDTH)
        or not (0.0 <= intrinsic[1, 2] < IMAGE_HEIGHT)
        or abs(float(np.linalg.det(intrinsic))) <= 1e-12
    ):
        raise F0RunnerError(f"depth intrinsic is not registered to 480x640 for {scene}")
    return path, np.ascontiguousarray(intrinsic)


def _valid_pose(value: np.ndarray) -> bool:
    return bool(
        value.shape == (4, 4)
        and np.isfinite(value).all()
        and np.allclose(value[3], [0.0, 0.0, 0.0, 1.0], atol=1e-6, rtol=0.0)
        and np.allclose(value[:3, :3].T @ value[:3, :3], np.eye(3), atol=1e-3, rtol=0.0)
        and math.isclose(float(np.linalg.det(value[:3, :3])), 1.0, abs_tol=1e-3)
    )


def _read_pose(path: Path) -> np.ndarray | None:
    try:
        value = np.loadtxt(_regular_file(path, "pose frame"), dtype=np.float64)
    except (OSError, ValueError, F0RunnerError):
        return None
    return np.ascontiguousarray(value) if _valid_pose(value) else None


def _decode_frame(
    scene_root: Path, scene: str, frame_id: int
) -> tuple[np.ndarray, np.ndarray, dict[str, Path]]:
    try:
        import cv2
    except ImportError as error:  # pragma: no cover - production dependency
        raise F0RunnerError("OpenCV is unavailable") from error
    root = scene_root / scene
    paths = {
        "rgb": _regular_file(root / "color" / f"{frame_id}.jpg", "RGB frame"),
        "depth": _regular_file(root / "depth" / f"{frame_id}.png", "depth frame"),
        "pose": _regular_file(root / "pose" / f"{frame_id}.txt", "pose frame"),
    }
    bgr = cv2.imread(os.fspath(paths["rgb"]), cv2.IMREAD_COLOR)
    depth = cv2.imread(os.fspath(paths["depth"]), cv2.IMREAD_UNCHANGED)
    if bgr is None or depth is None:
        raise F0RunnerError(f"could not decode RGB-D {scene}/{frame_id}")
    if depth.shape != (IMAGE_HEIGHT, IMAGE_WIDTH) or depth.dtype != np.uint16:
        raise F0RunnerError(f"depth must be uint16 [480,640]: {scene}/{frame_id}")
    bgr = cv2.resize(bgr, (IMAGE_WIDTH, IMAGE_HEIGHT), interpolation=cv2.INTER_LINEAR)
    return np.ascontiguousarray(bgr), np.ascontiguousarray(depth), paths


def _percentiles(values: Sequence[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    if not len(array):
        return {"sample_count": 0, "mean_ms": 0.0, "p50_ms": 0.0, "p95_ms": 0.0}
    return {
        "sample_count": int(len(array)),
        "mean_ms": float(array.mean()),
        "p50_ms": float(np.quantile(array, 0.50)),
        "p95_ms": float(np.quantile(array, 0.95)),
    }


def _json_timing(value: object) -> dict[str, object]:
    if value is None:
        return {}
    if is_dataclass(value):
        value = asdict(value)
    elif hasattr(value, "to_dict"):
        value = value.to_dict()
    if not isinstance(value, Mapping):
        raise F0RunnerError("FastSAM provider timing must be a dataclass or mapping")
    output: dict[str, object] = {}
    for key, item in value.items():
        name = str(key)
        if isinstance(item, (str, bool)) or item is None:
            output[name] = item
            continue
        if not isinstance(item, (int, float)):
            raise F0RunnerError("FastSAM provider timing contains a non-scalar value")
        number = float(item)
        if not math.isfinite(number) or number < 0.0:
            raise F0RunnerError("FastSAM provider timing contains an invalid value")
        output[name] = int(item) if isinstance(item, int) else number
        if name.endswith("_seconds"):
            output[name[: -len("_seconds")] + "_ms"] = number * 1000.0
    return dict(sorted(output.items()))


def _provider_predict(
    provider: Any, bgr: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    result = provider.predict(bgr)
    try:
        masks = np.asarray(result.masks)
        confidences = np.asarray(result.confidences)
        boxes = np.asarray(result.boxes_xyxy)
        timing = _json_timing(result.timing)
    except AttributeError as error:
        raise F0RunnerError("FastSAM provider result schema differs") from error
    if masks.shape[1:] != (IMAGE_HEIGHT, IMAGE_WIDTH) or masks.ndim != 3:
        raise F0RunnerError("FastSAM provider masks must have shape [N,480,640]")
    count = len(masks)
    if count > PROVIDER_MAX_DET:
        raise F0RunnerError("FastSAM provider result exceeds frozen max_det=100")
    if confidences.shape != (count,) or boxes.shape != (count, 4):
        raise F0RunnerError("FastSAM provider output counts differ")
    return (
        np.ascontiguousarray(masks),
        np.ascontiguousarray(confidences),
        np.ascontiguousarray(boxes),
        timing,
    )


def _checkpoint_metadata(provider: Any) -> dict[str, Any]:
    checkpoint = getattr(provider, "checkpoint", None)
    if checkpoint is None:
        raise F0RunnerError("FastSAM provider has no frozen checkpoint receipt")
    if is_dataclass(checkpoint):
        checkpoint = asdict(checkpoint)
    elif not isinstance(checkpoint, Mapping):
        checkpoint = {
            key: getattr(checkpoint, key)
            for key in ("path", "byte_count", "sha256")
            if hasattr(checkpoint, key)
        }
    path_value = checkpoint.get("path")
    byte_count = checkpoint.get("byte_count")
    digest = checkpoint.get("sha256")
    if not isinstance(path_value, (str, os.PathLike)):
        raise F0RunnerError("FastSAM checkpoint receipt has no path")
    path = _regular_file(Path(path_value), "FastSAM checkpoint")
    if (
        isinstance(byte_count, bool)
        or not isinstance(byte_count, int)
        or byte_count != path.stat().st_size
        or not isinstance(digest, str)
        or digest != _sha256(path)
    ):
        raise F0RunnerError("FastSAM checkpoint receipt differs from the file")
    return {"path": os.fspath(path), "bytes": byte_count, "sha256": digest}


def _provider_source(provider: Any) -> dict[str, str]:
    try:
        path = _regular_file(Path(inspect.getfile(type(provider))), "FastSAM provider source")
    except (TypeError, OSError) as error:
        raise F0RunnerError("could not locate FastSAM provider source") from error
    return {"path": os.fspath(path), "sha256": _sha256(path)}


def _default_provider_factory(checkpoint: Path, device: str) -> Any:
    from boxfusion.fastsam_automatic_provider import (  # noqa: PLC0415
        FrozenFastSAMAutomaticMaskProvider,
    )

    return FrozenFastSAMAutomaticMaskProvider(checkpoint, device=device)


def _environment_receipt(device: str, *, production: bool) -> dict[str, Any]:
    try:
        import cv2
        import torch
        import ultralytics
    except ImportError as error:  # pragma: no cover - production dependency
        raise F0RunnerError("F0 runtime dependencies are unavailable") from error
    receipt: dict[str, Any] = {
        "production_cuda_required": production,
        "dependency_injected_provider": not production,
        "conda_environment": os.environ.get("CONDA_DEFAULT_ENV"),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "opencv_version": cv2.__version__,
        "ultralytics_version": ultralytics.__version__,
        "device": str(device),
        "cuda_available": bool(torch.cuda.is_available()),
        "gpu_name": None,
        "gpu_uuid": None,
        "compute_capability": None,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "cuda_synchronization_contract": (
            "provider synchronizes CUDA before and after every prediction; "
            "host masks bound the following CPU residual core"
        ),
    }
    if not production:
        return receipt
    if (
        not isinstance(device, str)
        or not device.startswith("cuda:")
        or not device[len("cuda:") :].isdigit()
    ):
        raise F0RunnerError("production F0 device must be an explicit cuda:N")
    if (
        torch.__version__ != EXPECTED_TORCH_VERSION
        or torch.version.cuda != EXPECTED_TORCH_CUDA_VERSION
        or cv2.__version__ != EXPECTED_OPENCV_VERSION
        or ultralytics.__version__ != EXPECTED_ULTRALYTICS_VERSION
    ):
        raise F0RunnerError("production F0 software environment differs")
    if os.environ.get("CONDA_DEFAULT_ENV") != "boxfusion-online":
        raise F0RunnerError("production F0 conda environment differs")
    if os.environ.get("CUDA_VISIBLE_DEVICES") is not None:
        raise F0RunnerError("production F0 requires the unremapped two-GPU namespace")
    if not torch.cuda.is_available():
        raise F0RunnerError("production F0 CUDA is unavailable")
    index = int(device.split(":", 1)[1])
    if index >= torch.cuda.device_count():
        raise F0RunnerError("production F0 CUDA device index is unavailable")
    properties = torch.cuda.get_device_properties(index)
    capability = (int(properties.major), int(properties.minor))
    if properties.name != EXPECTED_GPU_NAME or capability != EXPECTED_COMPUTE_CAPABILITY:
        raise F0RunnerError("production F0 GPU identity differs from RTX3090 sm86")
    gpu_uuid = f"GPU-{str(properties.uuid).removeprefix('GPU-')}"
    if EXPECTED_GPU_UUID_BY_LOGICAL_DEVICE.get(device) != gpu_uuid:
        raise F0RunnerError("production F0 GPU UUID/logical-device binding differs")
    receipt.update(
        {
            "gpu_name": properties.name,
            "gpu_uuid": gpu_uuid,
            "compute_capability": list(capability),
        }
    )
    return receipt


def _validate_provider_timing_environment(
    timing: Mapping[str, object], environment: Mapping[str, object]
) -> None:
    if not environment.get("production_cuda_required"):
        return
    if (
        timing.get("device") != environment.get("device")
        or timing.get("cuda_synchronized") is not True
    ):
        raise F0RunnerError("FastSAM provider timing violated CUDA sync/device contract")


def _candidate_json(candidate: Any) -> dict[str, Any]:
    # Intentionally omit points_world and voxel_keys.  Their sealed joint hash,
    # support counts, and robust world geometry are sufficient for F0 audit.
    return {
        "raw_index": int(candidate.raw_index),
        "rank": int(candidate.rank),
        "confidence": float(candidate.confidence),
        "mask_sha256": candidate.mask_sha256,
        "tight_box_xyxy": candidate.tight_box_xyxy.tolist(),
        "pixel_count": int(candidate.pixel_count),
        "valid_pixel_count": int(candidate.valid_pixel_count),
        "residual_pixel_count": int(candidate.residual_pixel_count),
        "residual_ratio": float(candidate.residual_ratio),
        "valid_ratio": float(candidate.valid_ratio),
        "support_pixel_count": int(candidate.support_pixel_count),
        "voxel_count": int(candidate.voxel_count),
        "stored_point_count": int(candidate.stored_point_count),
        "points_and_voxel_keys_sha256": candidate.points_sha256,
        "world_q02": candidate.world_q02.tolist(),
        "world_q98": candidate.world_q98.tolist(),
        "world_center": candidate.world_center.tolist(),
        "world_extent": candidate.world_extent.tolist(),
    }


def _mask_json(mask: Any, provider_box: np.ndarray) -> dict[str, Any]:
    return {
        "raw_index": int(mask.raw_index),
        "confidence": float(mask.confidence),
        "mask_sha256": mask.mask_sha256,
        "provider_box_xyxy": np.asarray(provider_box, dtype=np.float64).tolist(),
        "tight_box_xyxy": mask.tight_box_xyxy.tolist(),
        "pixel_count": int(mask.pixel_count),
        "valid_pixel_count": int(mask.valid_pixel_count),
        "residual_pixel_count": int(mask.residual_pixel_count),
        "residual_ratio": float(mask.residual_ratio),
        "valid_ratio": float(mask.valid_ratio),
        "support_pixel_count": int(mask.support_pixel_count),
        "voxel_count": int(mask.voxel_count),
        "pre_dedup_eligible": bool(mask.pre_dedup_eligible),
        "deduplicated": bool(mask.deduplicated),
        "lifted": bool(mask.lifted),
        "selected": bool(mask.selected),
        "rank": None if mask.rank is None else int(mask.rank),
        "duplicate_of_raw_index": (
            None
            if mask.duplicate_of_raw_index is None
            else int(mask.duplicate_of_raw_index)
        ),
        "decision": mask.reason,
    }


def _frame_funnel(result: Any, provider_boxes: np.ndarray) -> dict[str, Any]:
    diagnostics = result.diagnostics
    return {
        "input_mask_count": int(diagnostics.input_mask_count),
        "input_explained_box_count": int(diagnostics.input_explained_box_count),
        "explained_union_pixels": int(diagnostics.explained_union_pixels),
        "pre_dedup_eligible_count": int(diagnostics.pre_dedup_eligible_count),
        "deduplicated_count": int(diagnostics.deduplicated_count),
        "post_dedup_count": int(diagnostics.post_dedup_count),
        "lifting_eligible_count": int(diagnostics.lifting_eligible_count),
        "selected_count": int(diagnostics.selected_count),
        "cap_rejected_count": int(diagnostics.cap_rejected_count),
        "rejection_counts": dict(diagnostics.rejection_counts),
        "masks": [
            _mask_json(mask, provider_boxes[mask.raw_index]) for mask in result.masks
        ],
        "candidates": [_candidate_json(candidate) for candidate in result.candidates],
    }


def _gpu_peak_bytes(device: str) -> int:
    try:
        import torch

        if str(device).startswith("cuda") and torch.cuda.is_available():
            return int(torch.cuda.max_memory_allocated(torch.device(device)))
    except Exception:
        pass
    return 0


def _reset_gpu_peak(device: str) -> None:
    try:
        import torch

        if str(device).startswith("cuda") and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(torch.device(device))
    except Exception:
        pass


def _accumulate_funnel_counts(
    counts: Counter[str], funnel: Mapping[str, Any]
) -> None:
    """Accumulate capacity counters with distinct core/provider saturation."""

    counts["successful_frames"] += 1
    counts["raw_masks"] += int(funnel["input_mask_count"])
    counts["pre_dedup_eligible_masks"] += int(funnel["pre_dedup_eligible_count"])
    counts["deduplicated_masks"] += int(funnel["deduplicated_count"])
    counts["lifting_eligible_masks"] += int(funnel["lifting_eligible_count"])
    counts["accepted_lifts"] += int(funnel["selected_count"])
    counts["cap_rejected_masks"] += int(funnel["cap_rejected_count"])
    # Exactly 16 selected masks is not evidence of truncation: it may be the
    # complete eligible set.  Only a diagnosed post-lift Top-K rejection is.
    if int(funnel["cap_rejected_count"]) > 0:
        counts["cap_saturated_frames"] += 1
    # This is a separate upstream source-cap diagnostic, not the core Top-K
    # capacity gate.  The provider itself fails if it ever returns >100.
    if int(funnel["input_mask_count"]) == PROVIDER_MAX_DET:
        counts["provider_max_det_saturated_frames"] += 1


def _resume_rewarm(
    *,
    provider: Any,
    environment: Mapping[str, object],
    scene_root: Path,
    scene: str,
    frame_id: int,
    completed_scene_count: int,
    pending_scene_count: int,
) -> dict[str, Any]:
    """Physically warm a cold resumed provider without creating F0 evidence."""

    try:
        import cv2
    except ImportError as error:  # pragma: no cover
        raise F0RunnerError("OpenCV is unavailable for resume rewarm") from error
    rgb_path = _regular_file(
        scene_root / scene / "color" / f"{frame_id}.jpg",
        "resume rewarm RGB frame",
    )
    bgr = cv2.imread(os.fspath(rgb_path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise F0RunnerError(f"could not decode resume rewarm RGB {scene}/{frame_id}")
    bgr = np.ascontiguousarray(
        cv2.resize(bgr, (IMAGE_WIDTH, IMAGE_HEIGHT), interpolation=cv2.INTER_LINEAR)
    )
    calls: list[dict[str, Any]] = []
    for ordinal in range(SHARD_WARMUP_SUCCESSFUL_CALLS):
        started = time.perf_counter()
        try:
            masks, confidences, boxes, timing = _provider_predict(provider, bgr)
            _validate_provider_timing_environment(timing, environment)
        except Exception as error:
            raise F0RunnerError(
                f"resume rewarm provider call {ordinal} failed for {scene}/{frame_id}"
            ) from error
        calls.append(
            {
                "ordinal": ordinal,
                "success": True,
                "wall_ms": float((time.perf_counter() - started) * 1000.0),
                "raw_mask_count": int(len(masks)),
                "masks_sha256": _array_sha256(masks),
                "confidences_sha256": _array_sha256(confidences),
                "boxes_xyxy_sha256": _array_sha256(boxes),
                "provider_timing": timing,
            }
        )
    return {
        "required": True,
        "reason": "cold_resume_with_completed_prefix_and_pending_suffix",
        "completed_scene_count": completed_scene_count,
        "pending_scene_count": pending_scene_count,
        "scene_id": scene,
        "frame_id": int(frame_id),
        "rgb_path": os.fspath(rgb_path),
        "rgb_sha256": _sha256(rgb_path),
        "call_count": len(calls),
        "all_successful": True,
        "excluded_from_scene_counts": True,
        "excluded_from_capacity": True,
        "excluded_from_runtime_distributions": True,
        "calls": calls,
    }


def _process_scene(
    *,
    scene: str,
    scene_index: int,
    schedule: Mapping[str, Any],
    scene_root: Path,
    provider: Any,
    device: str,
    run_signature: str,
    warmup_state: dict[str, int],
    environment: Mapping[str, object],
    checkpoint: Mapping[str, object],
    sources: Mapping[str, Mapping[str, str]],
) -> dict[str, Any]:
    intrinsic_path, intrinsic = _load_intrinsic(scene_root, scene)
    frames: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    rejection_counts: Counter[str] = Counter()
    runtime_samples: defaultdict[str, list[float]] = defaultdict(list)
    provider_internal_samples: defaultdict[str, list[float]] = defaultdict(list)
    provider_reported_gpu_peak_bytes = 0
    _reset_gpu_peak(device)

    for ordinal, frame_id in enumerate(schedule["frames"]):
        total_started = time.perf_counter()
        decode_started = time.perf_counter()
        bgr, depth_mm, paths = _decode_frame(scene_root, scene, frame_id)
        decode_ms = (time.perf_counter() - decode_started) * 1000.0

        cache_started = time.perf_counter()
        (
            cache_path,
            cutr_boxes,
            cache_sha256,
            cache_image_size,
            sealed_input_signature,
        ) = _load_cutr_boxes(schedule, scene, frame_id)
        cache_ms = (time.perf_counter() - cache_started) * 1000.0

        current_pose = _read_pose(paths["pose"])
        reconstructed_signature, producer_metadata = _reconstruct_cutr_input_signature(
            bgr=bgr,
            depth_mm=depth_mm,
            intrinsic=intrinsic,
            scene_root=scene_root,
            scene=scene,
            frame_id=frame_id,
        )
        if reconstructed_signature != sealed_input_signature:
            changed = sorted(
                key
                for key in reconstructed_signature
                if reconstructed_signature[key] != sealed_input_signature.get(key)
            )
            raise F0RunnerError(
                f"CuTR input signature differs for {scene}/{frame_id}: {changed}"
            )
        producer_shape = tuple(producer_metadata["producer_image_shape"][:2])
        if producer_shape != cache_image_size:
            raise F0RunnerError(
                f"CuTR producer/cache image shape differs for {scene}/{frame_id}"
            )
        orientation = int(producer_metadata["producer_orientation"])
        expected_orientation = EXPECTED_NON_UPRIGHT_KEYFRAMES.get((scene, frame_id), 0)
        if orientation != expected_orientation:
            raise F0RunnerError(
                f"CuTR producer orientation census differs for {scene}/{frame_id}: "
                f"{orientation} != {expected_orientation}"
            )
        expected_image_size = (
            (IMAGE_HEIGHT, IMAGE_WIDTH)
            if orientation == 0
            else (IMAGE_WIDTH, IMAGE_HEIGHT)
        )
        if cache_image_size != expected_image_size:
            raise F0RunnerError(
                f"CuTR cache orientation/image size differs for {scene}/{frame_id}"
            )
        frame: dict[str, Any] = {
            "frame_id": int(frame_id),
            "frame_ordinal": ordinal,
            "inputs": {
                "rgb_path": os.fspath(paths["rgb"]),
                "rgb_sha256": _sha256(paths["rgb"]),
                "depth_path": os.fspath(paths["depth"]),
                "depth_sha256": _sha256(paths["depth"]),
                "pose_path": os.fspath(paths["pose"]),
                "pose_sha256": _sha256(paths["pose"]),
                "cutr_cache_path": os.fspath(cache_path),
                "cutr_cache_sha256": cache_sha256,
                "cutr_box_count": int(len(cutr_boxes)),
                "cutr_cache_image_size": list(cache_image_size),
                "cutr_input_signature": reconstructed_signature,
                **producer_metadata,
                "current_pose_valid": current_pose is not None,
                "f0_pose_source_frame_id": frame_id if current_pose is not None else None,
                "f0_pose_forward_filled": False,
            },
            "successful": False,
            "abstention": None,
            "provider_timing": {},
            "funnel": None,
        }
        counts["keyframes"] += 1
        counts["cutr_boxes"] += len(cutr_boxes)
        abstention: str | None = None
        if current_pose is None:
            counts["invalid_pose_frames"] += 1
            abstention = "invalid_current_pose"
        elif orientation != 0:
            # Exactly two full200 keyframes are LEFT/RIGHT in the CuTR
            # producer and therefore have 640x480 boxes.  The frozen FastSAM
            # and F0 core contracts are 480x640; abstaining avoids an unsealed
            # coordinate transform while preserving exact cache validation.
            counts["non_upright_producer_frames"] += 1
            abstention = "non_upright_cache_coordinate_frame"
        if abstention is not None:
            rejection_counts[abstention] += 1
            receipt_total_ms = (time.perf_counter() - total_started) * 1000.0
            frame.update(
                {
                    "abstention": abstention,
                    "runtime": {
                        "decode_ms": decode_ms,
                        "cache_ms": cache_ms,
                        "provider_ms": 0.0,
                        "core_ms": 0.0,
                        "complete_ms": 0.0,
                        "receipt_total_ms": receipt_total_ms,
                        "provider_call_index_in_shard": None,
                        "warmup_excluded": False,
                    },
                }
            )
            frames.append(frame)
            runtime_samples["decode_ms"].append(decode_ms)
            runtime_samples["cache_ms"].append(cache_ms)
            runtime_samples["receipt_total_ms"].append(receipt_total_ms)
            continue

        provider_started = time.perf_counter()
        masks, confidences, provider_boxes, internal_timing = _provider_predict(
            provider, bgr
        )
        _validate_provider_timing_environment(internal_timing, environment)
        provider_reported_gpu_peak_bytes = max(
            provider_reported_gpu_peak_bytes,
            int(internal_timing.get("max_memory_allocated_bytes", 0)),
        )
        provider_ms = (time.perf_counter() - provider_started) * 1000.0
        core_started = time.perf_counter()
        try:
            result = f0_core.select_and_lift_residual_masks(
                masks=masks,
                confidences=confidences,
                depth_m=depth_mm.astype(np.float32) / 1000.0,
                explained_boxes_xyxy=cutr_boxes,
                intrinsics=intrinsic,
                camera_to_world=current_pose,
            )
        except ValueError as error:
            raise F0RunnerError(f"F0 core rejected sealed input {scene}/{frame_id}") from error
        core_ms = (time.perf_counter() - core_started) * 1000.0
        complete_ms = (time.perf_counter() - provider_started) * 1000.0
        funnel = _frame_funnel(result, provider_boxes)
        receipt_total_ms = (time.perf_counter() - total_started) * 1000.0
        provider_call_index = int(warmup_state["successful_provider_calls"])
        warmup_excluded = provider_call_index < SHARD_WARMUP_SUCCESSFUL_CALLS
        warmup_state["successful_provider_calls"] = provider_call_index + 1
        frame.update(
            {
                "successful": True,
                "provider_timing": internal_timing,
                "funnel": funnel,
                "runtime": {
                    "decode_ms": decode_ms,
                    "cache_ms": cache_ms,
                    "provider_ms": provider_ms,
                    "core_ms": core_ms,
                    "complete_ms": complete_ms,
                    "receipt_total_ms": receipt_total_ms,
                    "provider_call_index_in_shard": provider_call_index,
                    "warmup_excluded": warmup_excluded,
                },
            }
        )
        frames.append(frame)

        _accumulate_funnel_counts(counts, funnel)
        if warmup_excluded:
            counts["warmup_excluded_successful_frames"] += 1
        rejection_counts.update(funnel["rejection_counts"])
        runtime_samples["decode_ms"].append(decode_ms)
        runtime_samples["cache_ms"].append(cache_ms)
        runtime_samples["receipt_total_ms"].append(receipt_total_ms)
        if not warmup_excluded:
            runtime_samples["provider_ms"].append(provider_ms)
            runtime_samples["core_ms"].append(core_ms)
            runtime_samples["complete_ms"].append(complete_ms)
            for key, value in internal_timing.items():
                if key.endswith("_ms") and isinstance(value, (int, float)):
                    provider_internal_samples[key].append(float(value))

    for required in (
        "keyframes",
        "successful_frames",
        "invalid_pose_frames",
        "non_upright_producer_frames",
        "cutr_boxes",
        "raw_masks",
        "pre_dedup_eligible_masks",
        "deduplicated_masks",
        "lifting_eligible_masks",
        "accepted_lifts",
        "cap_rejected_masks",
        "cap_saturated_frames",
        "provider_max_det_saturated_frames",
        "warmup_excluded_successful_frames",
    ):
        counts.setdefault(required, 0)
    cap_ratio = (
        counts["cap_saturated_frames"] / counts["successful_frames"]
        if counts["successful_frames"]
        else 0.0
    )
    provider_cap_ratio = (
        counts["provider_max_det_saturated_frames"] / counts["successful_frames"]
        if counts["successful_frames"]
        else 0.0
    )
    summary = {
        "counts": dict(sorted(counts.items())),
        "rejection_counts": dict(sorted(rejection_counts.items())),
        "candidate_scene": counts["accepted_lifts"] > 0,
        "cap_saturation_ratio": float(cap_ratio),
        "provider_max_det_saturation_ratio": float(provider_cap_ratio),
        "runtime": {
            key: _percentiles(values) for key, values in sorted(runtime_samples.items())
        },
        "provider_internal_runtime": {
            key: _percentiles(values)
            for key, values in sorted(provider_internal_samples.items())
        },
        "cpu_peak_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024),
        "gpu_peak_memory_bytes": max(
            provider_reported_gpu_peak_bytes, _gpu_peak_bytes(device)
        ),
    }
    return {
        "schema": SCENE_SCHEMA,
        "protocol_id": PROTOCOL_ID,
        "run_signature_sha256": run_signature,
        "complete": True,
        "scene_id": scene,
        "scene_index": scene_index,
        "frame_id_ledger_sha256": _frame_id_ledger_sha256(schedule["frames"]),
        "environment_sha256": _canonical_json_sha256(environment),
        "checkpoint": dict(checkpoint),
        "sources": {key: dict(value) for key, value in sources.items()},
        "schedule": {
            "root": os.fspath(schedule["root"]),
            "manifest_path": os.fspath(schedule["path"]),
            "manifest_sha256": schedule["sha256"],
            "namespace": schedule["manifest"]["namespace"],
            "keyframe_count": len(schedule["frames"]),
            "proposal_count": schedule["manifest"]["proposal_count"],
        },
        "intrinsic": {
            "path": os.fspath(intrinsic_path),
            "sha256": _sha256(intrinsic_path),
        },
        "contracts": {
            "current_frame_rgb_depth_and_cutr_only": True,
            "current_pose_required_no_forward_fill": True,
            "history_or_tracking": False,
            "shadow_only": True,
            "no_output_affecting": True,
            "birth_enabled": False,
            "ground_truth_access": False,
            "terminal_native_prediction_access": False,
            "cutr_current_pred_boxes_access": True,
            "cutr_nonbox_field_use": False,
            "clip_or_semantic_use": False,
            "cutr_payload_deserialization_scope": "full_safe_payload",
        },
        "frames": frames,
        "summary": summary,
    }


def _atomic_create_json(path: Path, value: Mapping[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="ascii") as handle:
            json.dump(
                value,
                handle,
                indent=2,
                sort_keys=True,
                ensure_ascii=True,
                allow_nan=False,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as error:
            raise F0RunnerError(f"refusing to overwrite output: {path}") from error
        return _sha256(path)
    finally:
        temporary.unlink(missing_ok=True)


def _frame_id_ledger_sha256(frame_ids: Sequence[int]) -> str:
    return _canonical_json_sha256([int(frame_id) for frame_id in frame_ids])


def _recompute_scene_summary(frames: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    counts: Counter[str] = Counter()
    rejections: Counter[str] = Counter()
    runtime_samples: defaultdict[str, list[float]] = defaultdict(list)
    provider_internal: defaultdict[str, list[float]] = defaultdict(list)
    for frame in frames:
        counts["keyframes"] += 1
        counts["cutr_boxes"] += int(frame["inputs"]["cutr_box_count"])
        runtime = frame["runtime"]
        for key in ("decode_ms", "cache_ms", "receipt_total_ms"):
            runtime_samples[key].append(float(runtime[key]))
        if frame.get("successful") is True:
            funnel = frame["funnel"]
            _accumulate_funnel_counts(counts, funnel)
            rejections.update(funnel["rejection_counts"])
            if runtime["warmup_excluded"]:
                counts["warmup_excluded_successful_frames"] += 1
            else:
                for key in ("provider_ms", "core_ms", "complete_ms"):
                    runtime_samples[key].append(float(runtime[key]))
                for key, value in frame.get("provider_timing", {}).items():
                    if key.endswith("_ms") and isinstance(value, (int, float)):
                        provider_internal[key].append(float(value))
        else:
            reason = str(frame.get("abstention"))
            rejections[reason] += 1
            if reason == "invalid_current_pose":
                counts["invalid_pose_frames"] += 1
            elif reason == "non_upright_cache_coordinate_frame":
                counts["non_upright_producer_frames"] += 1
            else:
                raise F0RunnerError(f"unknown resumed frame abstention: {reason}")
    for required in (
        "keyframes",
        "successful_frames",
        "invalid_pose_frames",
        "non_upright_producer_frames",
        "cutr_boxes",
        "raw_masks",
        "pre_dedup_eligible_masks",
        "deduplicated_masks",
        "lifting_eligible_masks",
        "accepted_lifts",
        "cap_rejected_masks",
        "cap_saturated_frames",
        "provider_max_det_saturated_frames",
        "warmup_excluded_successful_frames",
    ):
        counts.setdefault(required, 0)
    successful = counts["successful_frames"]
    return {
        "counts": dict(sorted(counts.items())),
        "rejection_counts": dict(sorted(rejections.items())),
        "candidate_scene": counts["accepted_lifts"] > 0,
        "cap_saturation_ratio": (
            counts["cap_saturated_frames"] / successful if successful else 0.0
        ),
        "provider_max_det_saturation_ratio": (
            counts["provider_max_det_saturated_frames"] / successful
            if successful
            else 0.0
        ),
        "runtime": {
            key: _percentiles(values) for key, values in sorted(runtime_samples.items())
        },
        "provider_internal_runtime": {
            key: _percentiles(values)
            for key, values in sorted(provider_internal.items())
        },
    }


def _resume_scene(
    path: Path,
    *,
    scene: str,
    scene_index: int,
    run_signature: str,
    schedule: Mapping[str, Any],
    provider_call_start: int,
    environment: Mapping[str, object],
    checkpoint: Mapping[str, object],
    sources: Mapping[str, Mapping[str, str]],
) -> tuple[dict[str, Any], str, int]:
    receipt = _read_json(path, f"resumed F0 scene receipt {scene}")
    if (
        receipt.get("schema") != SCENE_SCHEMA
        or receipt.get("protocol_id") != PROTOCOL_ID
        or receipt.get("run_signature_sha256") != run_signature
        or receipt.get("scene_id") != scene
        or receipt.get("scene_index") != scene_index
        or receipt.get("complete") is not True
        or receipt.get("schedule", {}).get("manifest_sha256") != schedule["sha256"]
        or receipt.get("environment_sha256") != _canonical_json_sha256(environment)
        or receipt.get("checkpoint") != dict(checkpoint)
        or receipt.get("sources")
        != {key: dict(value) for key, value in sources.items()}
    ):
        raise F0RunnerError(f"resumed scene receipt contract differs: {path}")
    frames = receipt.get("frames")
    if not isinstance(frames, list) or any(
        not isinstance(row, dict) or isinstance(row.get("frame_id"), bool) or not isinstance(row.get("frame_id"), int)
        for row in frames
    ) or len({row["frame_id"] for row in frames}) != len(frames):
        raise F0RunnerError(f"resumed scene frame ledger is invalid: {path}")
    frame_ids = [row["frame_id"] for row in frames]
    if (
        frame_ids != list(schedule["frames"])
        or receipt.get("frame_id_ledger_sha256")
        != _frame_id_ledger_sha256(frame_ids)
    ):
        raise F0RunnerError(f"resumed scene frame schedule differs: {path}")
    recomputed = _recompute_scene_summary(frames)
    summary = receipt.get("summary")
    if not isinstance(summary, dict) or any(
        summary.get(key) != recomputed[key]
        for key in (
            "counts",
            "rejection_counts",
            "candidate_scene",
            "cap_saturation_ratio",
            "provider_max_det_saturation_ratio",
            "runtime",
            "provider_internal_runtime",
        )
    ):
        raise F0RunnerError(f"resumed scene summary differs from frames: {path}")
    successful = [row for row in frames if row.get("successful") is True]
    call_indices = [
        row.get("runtime", {}).get("provider_call_index_in_shard")
        for row in successful
    ]
    expected_indices = list(
        range(provider_call_start, provider_call_start + len(successful))
    )
    if call_indices != expected_indices or any(
        row.get("runtime", {}).get("warmup_excluded")
        != (index < SHARD_WARMUP_SUCCESSFUL_CALLS)
        for row, index in zip(successful, expected_indices)
    ):
        raise F0RunnerError(f"resumed scene shard-global warmup differs: {path}")
    return receipt, _sha256(path), len(successful)


def _scene_manifest_row(
    receipt: Mapping[str, Any], sidecar_path: Path, sidecar_sha256: str
) -> dict[str, Any]:
    summary = receipt["summary"]
    schedule = receipt["schedule"]
    return {
        "scene_id": receipt["scene_id"],
        "scene_index": receipt["scene_index"],
        "sidecar_path": os.fspath(sidecar_path.resolve()),
        "sidecar_sha256": sidecar_sha256,
        "frame_id_ledger_sha256": receipt["frame_id_ledger_sha256"],
        "schedule_root": schedule["root"],
        "schedule_path": schedule["manifest_path"],
        "schedule_sha256": schedule["manifest_sha256"],
        "keyframe_count": schedule["keyframe_count"],
        "counts": summary["counts"],
        "runtime": summary["runtime"],
        "cpu_peak_rss_bytes": summary["cpu_peak_rss_bytes"],
        "gpu_peak_memory_bytes": summary["gpu_peak_memory_bytes"],
    }


def _aggregate_scene_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    totals: Counter[str] = Counter()
    for row in rows:
        totals.update({key: int(value) for key, value in row["counts"].items()})
    candidate_scene_count = sum(row["counts"]["accepted_lifts"] > 0 for row in rows)
    successful = totals["successful_frames"]
    cap_ratio = totals["cap_saturated_frames"] / successful if successful else 0.0
    provider_cap_ratio = (
        totals["provider_max_det_saturated_frames"] / successful
        if successful
        else 0.0
    )
    return {
        **dict(sorted(totals.items())),
        "candidate_scene_count": candidate_scene_count,
        "cap_saturation_ratio": float(cap_ratio),
        "provider_max_det_saturation_ratio": float(provider_cap_ratio),
    }


def _signature_payload(
    *,
    plan: Mapping[str, Any],
    full_scenes: Sequence[str],
    schedules: Mapping[str, Mapping[str, Any]],
    scene_root: Path,
    checkpoint: Mapping[str, Any],
    sources: Mapping[str, Mapping[str, str]],
    environment: Mapping[str, object],
) -> dict[str, Any]:
    # The protocol signature is shared across the two deterministic shards.
    # Per-shard logical device and physical UUID remain independently sealed
    # in ``environment``/``environment_sha256`` and are validated by merge;
    # including them here would make cuda:0 and cuda:1 signatures unequal.
    environment_protocol = {
        key: value
        for key, value in environment.items()
        if key not in {"device", "gpu_uuid"}
    }
    return {
        "protocol_id": PROTOCOL_ID,
        "scene_list_sha256": plan["scene_list"]["sha256"],
        "scene_order": list(full_scenes),
        "scene_root": os.fspath(scene_root.resolve()),
        "schedule_manifests": [
            {
                "scene_id": scene,
                "root": os.fspath(schedules[scene]["root"]),
                "sha256": schedules[scene]["sha256"],
                "producer_fingerprint": schedules[scene]["manifest"][
                    "producer_fingerprint"
                ],
            }
            for scene in full_scenes
        ],
        "checkpoint": dict(checkpoint),
        "sources": {key: dict(value) for key, value in sources.items()},
        "environment_protocol": environment_protocol,
        "core_schema": f0_core.SCHEMA,
        "core_policy": dict(f0_core.POLICY),
    }


def _validate_recorded_identity(
    *,
    checkpoint: object,
    sources: object,
    strict_default_provider: bool,
) -> tuple[dict[str, Any], dict[str, dict[str, str]]]:
    if not isinstance(checkpoint, dict) or not isinstance(sources, dict):
        raise F0RunnerError("recorded F0 execution identity is invalid")
    checkpoint_path = _regular_file(Path(checkpoint.get("path", "")), "FastSAM checkpoint")
    normalized_checkpoint = {
        "path": os.fspath(checkpoint_path),
        "bytes": checkpoint_path.stat().st_size,
        "sha256": _sha256(checkpoint_path),
    }
    if normalized_checkpoint != checkpoint:
        raise F0RunnerError("recorded FastSAM checkpoint identity differs")
    normalized_sources: dict[str, dict[str, str]] = {}
    for name in ("runner", "core", "provider"):
        row = sources.get(name)
        if not isinstance(row, dict):
            raise F0RunnerError(f"recorded {name} source identity is invalid")
        path = _regular_file(Path(row.get("path", "")), f"recorded {name} source")
        normalized_sources[name] = {"path": os.fspath(path), "sha256": _sha256(path)}
        if normalized_sources[name] != row:
            raise F0RunnerError(f"recorded {name} source identity differs")
    if (
        Path(normalized_sources["runner"]["path"]) != Path(__file__).resolve()
        or Path(normalized_sources["core"]["path"]) != Path(f0_core.__file__).resolve()
    ):
        raise F0RunnerError("recorded runner/core source path differs")
    if strict_default_provider:
        from boxfusion import fastsam_automatic_provider as provider_module

        if (
            checkpoint_path != DEFAULT_CHECKPOINT.resolve()
            or normalized_checkpoint["sha256"] != EXPECTED_CHECKPOINT_SHA256
            or Path(normalized_sources["provider"]["path"])
            != Path(provider_module.__file__).resolve()
        ):
            raise F0RunnerError("recorded production FastSAM identity differs")
    return normalized_checkpoint, normalized_sources


def _existing_completed_manifest(
    path: Path,
    *,
    plan: Mapping[str, Any],
    full_scenes: Sequence[str],
    schedules: Mapping[str, Mapping[str, Any]],
    scene_root: Path,
    environment: Mapping[str, object],
    strict_default_provider: bool,
    shard_index: int,
    num_shards: int,
    selected_indices: Sequence[int],
    selected_scenes: Sequence[str],
) -> dict[str, Any]:
    value = _read_json(path, "resumed F0 shard manifest")
    if (
        value.get("schema") != SHARD_SCHEMA
        or value.get("shard", {}).get("index") != shard_index
        or value.get("shard", {}).get("count") != num_shards
        or value.get("shard", {}).get("scene_order") != list(selected_scenes)
        or value.get("scene_list") != plan["scene_list"]
        or value.get("full200_keyframe_count") != plan["full200_keyframe_count"]
        or value.get("environment") != dict(environment)
        or value.get("environment_sha256")
        != _canonical_json_sha256(environment)
        or value.get("complete") is not True
    ):
        raise F0RunnerError(f"resumed shard manifest contract differs: {path}")
    checkpoint, sources = _validate_recorded_identity(
        checkpoint=value.get("checkpoint"),
        sources=value.get("sources"),
        strict_default_provider=strict_default_provider,
    )
    run_signature = _canonical_json_sha256(
        _signature_payload(
            plan=plan,
            full_scenes=full_scenes,
            schedules=schedules,
            scene_root=scene_root,
            checkpoint=checkpoint,
            sources=sources,
            environment=environment,
        )
    )
    if value.get("run_signature_sha256") != run_signature:
        raise F0RunnerError(f"resumed shard run signature differs: {path}")
    rows = value.get("scenes")
    if not isinstance(rows, list) or len(rows) != len(selected_scenes):
        raise F0RunnerError(f"resumed shard scene rows differ: {path}")
    provider_call_start = 0
    for scene_index, scene, row in zip(selected_indices, selected_scenes, rows):
        if not isinstance(row, dict) or row.get("scene_id") != scene:
            raise F0RunnerError(f"resumed shard scene order differs: {path}")
        sidecar = _regular_file(Path(row["sidecar_path"]), "resumed F0 sidecar")
        receipt, sidecar_sha256, successful_calls = _resume_scene(
            sidecar,
            scene=scene,
            scene_index=scene_index,
            run_signature=run_signature,
            schedule=schedules[scene],
            provider_call_start=provider_call_start,
            environment=environment,
            checkpoint=checkpoint,
            sources=sources,
        )
        expected_row = _scene_manifest_row(receipt, sidecar, sidecar_sha256)
        if any(row.get(key) != expected_row[key] for key in expected_row):
            raise F0RunnerError(f"resumed shard scene receipt differs: {scene}")
        provider_call_start += successful_calls
    return value


def run_shadow(
    *,
    schedule_roots: Sequence[Path],
    scene_root: Path,
    scene_list_path: Path,
    output_root: Path,
    device: str,
    shard_index: int = 0,
    num_shards: int = 1,
    resume: bool = False,
    plan_only: bool = False,
    provider_factory: Callable[[Path, str], Any] | None = None,
    _expected_scene_count: int = EXPECTED_SCENE_COUNT,
) -> dict[str, Any]:
    """Execute one deterministic F0 shard or return its read-only plan."""

    _validate_shard(shard_index, num_shards)
    full_scenes = _read_scene_list(scene_list_path, _expected_scene_count)
    if (
        _expected_scene_count == EXPECTED_SCENE_COUNT
        and _sha256(scene_list_path) != EXPECTED_SCENE_LIST_SHA256
    ):
        raise F0RunnerError("F0 full200 scene-list SHA-256 differs")
    schedules = _resolve_schedules(schedule_roots, full_scenes)
    full_keyframe_count = sum(len(row["frames"]) for row in schedules.values())
    if (
        _expected_scene_count == EXPECTED_SCENE_COUNT
        and full_keyframe_count != EXPECTED_FULL200_KEYFRAME_COUNT
    ):
        raise F0RunnerError(
            "F0 full200 keyframe count differs: "
            f"{full_keyframe_count} != {EXPECTED_FULL200_KEYFRAME_COUNT}"
        )
    selected_indices = tuple(
        index for index in range(len(full_scenes)) if index % num_shards == shard_index
    )
    selected_scenes = tuple(full_scenes[index] for index in selected_indices)
    plan = {
        "schema": SHARD_SCHEMA,
        "protocol_id": PROTOCOL_ID,
        "mode": "plan_only" if plan_only else "shadow",
        "scene_list": {
            "path": os.fspath(scene_list_path.resolve()),
            "sha256": _sha256(scene_list_path),
            "exact_scene_count": len(full_scenes),
        },
        "shard": {
            "index": shard_index,
            "count": num_shards,
            "scene_indices": list(selected_indices),
            "scene_order": list(selected_scenes),
        },
        "full200_keyframe_count": full_keyframe_count,
        "shard_keyframe_count": sum(len(schedules[scene]["frames"]) for scene in selected_scenes),
        "schedule_roots": [os.fspath(Path(root).resolve()) for root in schedule_roots],
        "scene_root": os.fspath(scene_root.resolve()),
        "expected_non_upright_keyframes": [
            {"scene_id": scene, "frame_id": frame_id, "orientation": orientation}
            for (scene, frame_id), orientation in EXPECTED_NON_UPRIGHT_KEYFRAMES.items()
            if scene in selected_scenes
        ],
        "expected_execution_census": {
            "sha256": EXPECTED_EXECUTION_CENSUS_SHA256,
            "counts": (
                dict(EXPECTED_EXECUTION_COUNTS[(num_shards, shard_index)])
                if _expected_scene_count == EXPECTED_SCENE_COUNT
                and (num_shards, shard_index) in EXPECTED_EXECUTION_COUNTS
                else None
            ),
        },
    }
    if plan_only:
        print(json.dumps(plan, sort_keys=True), flush=True)
        return plan

    if (
        provider_factory is None
        and _expected_scene_count == EXPECTED_SCENE_COUNT
        and num_shards == 2
        and device != f"cuda:{shard_index}"
    ):
        raise F0RunnerError(
            "production two-way F0 requires shard 0 on cuda:0 and shard 1 on cuda:1"
        )

    output = output_root.resolve()
    if output_root.is_symlink():
        raise F0RunnerError(f"output root cannot be a symlink: {output_root}")
    scenes_dir = output / "scenes"
    shards_dir = output / "shards"
    manifest_path = shards_dir / f"shard-{shard_index:03d}-of-{num_shards:03d}.json"
    if manifest_path.exists() and not resume:
        raise F0RunnerError(f"refusing to overwrite output: {manifest_path}")
    environment = _environment_receipt(
        device, production=provider_factory is None
    )
    if manifest_path.exists():
        return _existing_completed_manifest(
            manifest_path,
            plan=plan,
            full_scenes=full_scenes,
            schedules=schedules,
            scene_root=scene_root.resolve(),
            environment=environment,
            strict_default_provider=provider_factory is None,
            shard_index=shard_index,
            num_shards=num_shards,
            selected_indices=selected_indices,
            selected_scenes=selected_scenes,
        )

    sidecar_paths = tuple(scenes_dir / f"{scene}.json" for scene in selected_scenes)
    existing_flags = tuple(path.exists() or path.is_symlink() for path in sidecar_paths)
    if any(existing_flags) and not resume:
        first = sidecar_paths[existing_flags.index(True)]
        raise F0RunnerError(f"refusing to overwrite output: {first}")
    completed_prefix_count = 0
    while (
        completed_prefix_count < len(existing_flags)
        and existing_flags[completed_prefix_count]
    ):
        completed_prefix_count += 1
    if any(existing_flags[completed_prefix_count:]):
        raise F0RunnerError("resume scene sidecars must form an exact completed prefix")
    pending_count = len(selected_scenes) - completed_prefix_count

    provider: Any | None = None
    if pending_count:
        factory = provider_factory or _default_provider_factory
        provider = factory(DEFAULT_CHECKPOINT, device)
        checkpoint = _checkpoint_metadata(provider)
        if provider_factory is None and (
            Path(checkpoint["path"]) != DEFAULT_CHECKPOINT.resolve()
            or checkpoint["sha256"] != EXPECTED_CHECKPOINT_SHA256
        ):
            raise F0RunnerError("FastSAM provider checkpoint differs from frozen F0")
        sources = {
            "runner": {
                "path": os.fspath(Path(__file__).resolve()),
                "sha256": _sha256(Path(__file__).resolve()),
            },
            "core": {
                "path": os.fspath(Path(f0_core.__file__).resolve()),
                "sha256": _sha256(Path(f0_core.__file__).resolve()),
            },
            "provider": _provider_source(provider),
        }
    elif completed_prefix_count:
        identity = _read_json(sidecar_paths[0], "completed F0 scene identity")
        checkpoint, sources = _validate_recorded_identity(
            checkpoint=identity.get("checkpoint"),
            sources=identity.get("sources"),
            strict_default_provider=provider_factory is None,
        )
    else:
        raise F0RunnerError("empty shard has no provider-free execution identity")

    run_signature = _canonical_json_sha256(
        _signature_payload(
            plan=plan,
            full_scenes=full_scenes,
            schedules=schedules,
            scene_root=scene_root.resolve(),
            checkpoint=checkpoint,
            sources=sources,
            environment=environment,
        )
    )
    scene_rows: list[dict[str, Any]] = []
    warmup_state = {"successful_provider_calls": 0}
    shard_runtime_started = time.perf_counter()
    for ordinal in range(completed_prefix_count):
        scene_index = selected_indices[ordinal]
        scene = selected_scenes[ordinal]
        sidecar_path = sidecar_paths[ordinal]
        receipt, sidecar_sha256, resumed_successful_calls = _resume_scene(
            sidecar_path,
            scene=scene,
            scene_index=scene_index,
            run_signature=run_signature,
            schedule=schedules[scene],
            provider_call_start=warmup_state["successful_provider_calls"],
            environment=environment,
            checkpoint=checkpoint,
            sources=sources,
        )
        warmup_state["successful_provider_calls"] += resumed_successful_calls
        row = _scene_manifest_row(receipt, sidecar_path, sidecar_sha256)
        row["resumed"] = True
        scene_rows.append(row)

    if completed_prefix_count and pending_count:
        first_pending_scene = selected_scenes[completed_prefix_count]
        first_pending_frame = schedules[first_pending_scene]["frames"][0]
        assert provider is not None
        resume_rewarm = _resume_rewarm(
            provider=provider,
            environment=environment,
            scene_root=scene_root.resolve(),
            scene=first_pending_scene,
            frame_id=first_pending_frame,
            completed_scene_count=completed_prefix_count,
            pending_scene_count=pending_count,
        )
    else:
        resume_rewarm = {
            "required": False,
            "reason": (
                "no_pending_scene"
                if not pending_count
                else "fresh_or_resume_without_completed_scene"
            ),
            "completed_scene_count": completed_prefix_count,
            "pending_scene_count": pending_count,
            "call_count": 0,
            "all_successful": True,
            "excluded_from_scene_counts": True,
            "excluded_from_capacity": True,
            "excluded_from_runtime_distributions": True,
            "calls": [],
        }

    for ordinal in range(completed_prefix_count, len(selected_scenes)):
        scene_index = selected_indices[ordinal]
        scene = selected_scenes[ordinal]
        sidecar_path = sidecar_paths[ordinal]
        assert provider is not None
        receipt = _process_scene(
            scene=scene,
            scene_index=scene_index,
            schedule=schedules[scene],
            scene_root=scene_root.resolve(),
            provider=provider,
            device=device,
            run_signature=run_signature,
            warmup_state=warmup_state,
            environment=environment,
            checkpoint=checkpoint,
            sources=sources,
        )
        sidecar_sha256 = _atomic_create_json(sidecar_path, receipt)
        row = _scene_manifest_row(receipt, sidecar_path, sidecar_sha256)
        row["resumed"] = False
        scene_rows.append(row)
        counts = row["counts"]
        print(
            f"[{ordinal + 1}/{len(selected_scenes)}] {scene}: "
            f"frames={counts['keyframes']} masks={counts['raw_masks']} "
            f"lifts={counts['accepted_lifts']} cap={counts['cap_saturated_frames']} "
            "written",
            flush=True,
        )

    # Fail if source/checkpoint files changed while the long run was active.
    for row in sources.values():
        if _sha256(_regular_file(Path(row["path"]), "frozen source")) != row["sha256"]:
            raise F0RunnerError(f"frozen source changed during run: {row['path']}")
    if _sha256(_regular_file(Path(checkpoint["path"]), "FastSAM checkpoint")) != checkpoint["sha256"]:
        raise F0RunnerError("FastSAM checkpoint changed during run")
    for scene in selected_scenes:
        schedule = schedules[scene]
        if _sha256(_regular_file(schedule["path"], "CuTR schedule manifest")) != schedule["sha256"]:
            raise F0RunnerError(f"CuTR schedule changed during run: {scene}")

    totals = _aggregate_scene_rows(scene_rows)
    expected_non_upright_count = sum(
        scene in selected_scenes for scene, _frame_id in EXPECTED_NON_UPRIGHT_KEYFRAMES
    )
    if totals["non_upright_producer_frames"] != expected_non_upright_count:
        raise F0RunnerError(
            "observed non-upright abstention count differs from sealed census"
        )
    if warmup_state["successful_provider_calls"] != totals["successful_frames"]:
        raise F0RunnerError("provider call ledger differs from successful frame total")
    expected_execution_counts = plan["expected_execution_census"]["counts"]
    if expected_execution_counts is not None and any(
        totals[key] != expected
        for key, expected in expected_execution_counts.items()
    ):
        raise F0RunnerError("shard execution counts differ from sealed full200 census")
    shard_fraction = len(selected_scenes) / EXPECTED_SCENE_COUNT
    scaled_minimum_lifts = math.ceil(MINIMUM_FULL200_ACCEPTED_LIFTS * shard_fraction)
    scaled_minimum_scenes = math.ceil(MINIMUM_FULL200_CANDIDATE_SCENES * shard_fraction)
    capacity = {
        "full200_reference": {
            "minimum_accepted_lifts": MINIMUM_FULL200_ACCEPTED_LIFTS,
            "minimum_candidate_scenes": MINIMUM_FULL200_CANDIDATE_SCENES,
            "maximum_cap_saturation_ratio": MAXIMUM_CAP_SATURATION_RATIO,
        },
        "shard_scaled_reference": {
            "minimum_accepted_lifts": scaled_minimum_lifts,
            "minimum_candidate_scenes": scaled_minimum_scenes,
        },
        "accepted_lifts_pass": totals["accepted_lifts"] >= scaled_minimum_lifts,
        "candidate_scene_coverage_pass": totals["candidate_scene_count"] >= scaled_minimum_scenes,
        "cap_saturation_pass": totals["cap_saturation_ratio"] <= MAXIMUM_CAP_SATURATION_RATIO,
    }
    capacity["pass"] = all(
        capacity[key]
        for key in (
            "accepted_lifts_pass",
            "candidate_scene_coverage_pass",
            "cap_saturation_pass",
        )
    )
    contracts = {
        "shadow_only": True,
        "no_output_affecting": True,
        "birth_enabled": False,
        "ground_truth_access": False,
        "annotation_access": False,
        "evaluator_access": False,
        "terminal_native_prediction_access": False,
        "terminal_native_prediction_mutation": False,
        "terminal_prediction_pickle_write": False,
        "cutr_current_pred_boxes_access": True,
        "cutr_nonbox_field_use": False,
        "cutr_payload_deserialization_scope": "full_safe_payload",
        "clip_or_semantic_use": False,
        "tracking_or_history": False,
        "training": False,
        "online_learning": False,
        "external_pretraining_frozen": True,
        "current_pose_required_no_forward_fill": True,
    }
    manifest = {
        **plan,
        "mode": "shadow",
        "complete": True,
        "run_signature_sha256": run_signature,
        "environment": environment,
        "environment_sha256": _canonical_json_sha256(environment),
        "contracts": contracts,
        "policy": {
            "core_schema": f0_core.SCHEMA,
            "core": dict(f0_core.POLICY),
            "provider": "frozen FastSAM-x automatic masks; fixed adapter policy",
            "provider_max_det": PROVIDER_MAX_DET,
            "schedule": (
                "sealed CuTR-v2 manifest gap25; torch.load weights_only safely "
                "deserializes the full payload, while F0 indexes/uses pred_boxes only"
            ),
            "pose": "current raw pose required; invalid current pose abstains",
            "orientation": {
                "census_full200": {"UPRIGHT": 12939, "LEFT": 1, "RIGHT": 1},
                "non_upright_keyframes": plan["expected_non_upright_keyframes"],
                "non_upright_action": "abstain_without_provider_or_core",
            },
            "runtime": {
                "warmup_excluded_successful_provider_calls_per_shard": SHARD_WARMUP_SUCCESSFUL_CALLS,
                "complete_ms_scope": "provider_predict_plus_residual_core_wall",
                "receipt_total_ms_scope": "decode_cache_hash_provider_core_and_receipt",
                "invalid_pose_complete_ms": 0.0,
            },
            "capacity_accounting": {
                "cap_saturated_frame": "core_cap_rejected_count_gt_zero",
                "provider_max_det_saturated_frame": "raw_mask_count_eq_100",
            },
        },
        "checkpoint": checkpoint,
        "sources": sources,
        "resume_rewarm_calls": int(resume_rewarm["call_count"]),
        "resume_rewarm": resume_rewarm,
        "scenes": scene_rows,
        "totals": totals,
        "capacity": capacity,
        "runtime": {
            "wall_seconds": float(time.perf_counter() - shard_runtime_started),
            "cpu_peak_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024),
            "gpu_peak_memory_bytes": max(
                (int(row["gpu_peak_memory_bytes"]) for row in scene_rows), default=0
            ),
        },
        "conclusion_guardrail": (
            "F0 measures no-GT residual automatic-mask capacity and geometry only. "
            "It cannot establish AP and it does not create or modify a prediction."
        ),
    }
    _atomic_create_json(manifest_path, manifest)
    print(f"Saved: {manifest_path}", flush=True)
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run frozen FastSAM-x F0 residual-mask full200 shadow"
    )
    parser.add_argument(
        "--schedule-root",
        action="append",
        type=Path,
        dest="schedule_roots",
        help="repeat exactly once per composite CuTR-v2 schedule root",
    )
    parser.add_argument("--scene-root", type=Path, default=DEFAULT_SCENE_ROOT)
    parser.add_argument("--scene-list", type=Path, default=DEFAULT_SCENE_LIST)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--plan-only", action="store_true")
    return parser


def main() -> None:
    args = _parser().parse_args()
    roots = tuple(args.schedule_roots) if args.schedule_roots else DEFAULT_SCHEDULE_ROOTS
    run_shadow(
        schedule_roots=roots,
        scene_root=args.scene_root,
        scene_list_path=args.scene_list,
        output_root=args.output_root,
        device=args.device,
        shard_index=args.shard_index,
        num_shards=args.num_shards,
        resume=args.resume,
        plan_only=args.plan_only,
    )


if __name__ == "__main__":
    main()
