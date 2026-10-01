#!/usr/bin/env python3
"""Run the fresh, exact-frame, no-GT S3R H10 Boxer shadow provider.

This is deliberately not a wrapper around Boxer's ScanNet loader or command
line entry point.  A manifest-driven reader opens exactly one scheduled frame
at a time, verifies all input bytes, constructs the inference datum
synchronously, and returns control only after the frame transaction has been
durably committed.  The frozen OWLv2 and BoxerNet models are constructed once
for the complete ten-scene process.

The output is a shadow receipt only.  It cannot modify the native T05 prefix,
run an evaluator, track instances, or enable births.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import importlib
import io
import json
import math
import os
from pathlib import Path
import random
import resource
import stat
import subprocess
import sys
import time
from typing import Any, Callable, Mapping, Protocol, Sequence

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if os.fspath(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, os.fspath(REPOSITORY_ROOT))

from boxfusion.s3r_h10_provider_core import (  # noqa: E402
    ExactScheduleBundle,
    FrameTransaction,
    SceneSchedule,
    ScheduledFrame,
    parse_exact_schedule_bundle,
)


RUN_SCHEMA = "boxfusion.s3r_h10_fresh_boxer_provider_run.v1"
SCHEDULE_PATH = REPOSITORY_ROOT / "docs" / "data" / "S3R_H10_EXACT_SCHEDULE_V2.json"
EXPECTED_SCHEDULE_SHA256 = (
    "1ce565a65510b80d69a0402fe7a40ea89920625f6a81147d42f9232f7a7761e9"
)
HOLDOUT_LIST_PATH = (
    REPOSITORY_ROOT
    / "evaluation"
    / "data_util"
    / "meta_data"
    / "scannetv2_boxer_past3_s1_holdout10.txt"
)
EXPECTED_HOLDOUT_LIST_SHA256 = (
    "8965d0534ed3028f85d8b0ea7227d348a6faa1387b858ddf42c3183bd9ebdf90"
)
SCENE_ROOT = REPOSITORY_ROOT / "upstream_clean" / "scannet_readme_frames"
FORMAL_T05_ROOT = REPOSITORY_ROOT / "results" / "scannet_topk_fusion_score05"
SHADOW_OUTPUT_ROOT = REPOSITORY_ROOT / "logs"

BOXER_ROOT = Path("/data/ZhaoX/OVM3D-Dett/boxfusion_boxer_dev/third_party/boxer")
EXPECTED_BOXER_COMMIT = "1f86542dc342a4b1d474c87c97c5d1d6566d9148"
OWL_CHECKPOINT = Path(
    "/data/ZhaoX/OVM3D-Dett/third_party/boxer/ckpts/" "owlv2-base-patch16-ensemble.pt"
)
OWL_TEXT_CACHE = Path(
    "/data/ZhaoX/OVM3D-Dett/third_party/boxer/ckpts/"
    "owlv2-base-patch16-ensemble_textemb_878186d327b0.pt"
)
BOXER_CHECKPOINT_RELPATH = "ckpts/boxernet_hw960in2x6d768-c88128f8.ckpt"
DINO_CHECKPOINT_RELPATH = "ckpts/dinov3_vits16plus_pretrain_lvd1689m-4057cbaa.pth"

EXPECTED_EXTERNAL_CODE_SHA256: Mapping[str, str] = {
    "run_boxer.py": "8ff93e62881db5bd4d0fb20cbddfb5767ec2c4a941e873672c3acf603ecdad1b",
    "boxernet/boxernet.py": "a8009c1c0932aaab98bb074a2a4c50e55a3fbdfc3c6cb1afc9e1aef0e5324130",
    "owl/owl_wrapper.py": "7cf26a25bba1e67d8d8230ef47eb8288a48a728eda27d846e4f57bc6d4b6c628",
    "owl/clip_tokenizer.py": "39ac9e78731d91d0e50be80ac5ab1a2045ab28ab41e07ce35017b0eaa677dfe3",
    "owl/lvisplus_classes.csv": "3d6fd6fedb15ec5ea2f8ae80d2a5da310e64bece64aa38bb14f16cb7ac05cb3e",
    "utils/taxonomy.py": "42f26d270d6305c6cf3dbddc1635c4e7473837b6bd7bcc1654d8b43bf2018ec7",
    "utils/tw/camera.py": "dd31d0df949b2e937e81e76d994e40680e8c6412b7e424b22d8a5b43207521cf",
    "utils/tw/pose.py": "61091d10b5ecbb2720bf86ee78da21d8b0059ee45c03a5b127a4816606004703",
    # These two files are frozen as negative provenance: this runner neither
    # imports nor instantiates their asynchronous/dataset loader classes.
    "loaders/base_loader.py": "93e3e1fb600960b3f8dfcd9091a745787ccd6be258ef9ae54d08bebb3107839d",
    "loaders/scannet_loader.py": "93a451d70cd57ba01290e152ef5d7b95d4de7f0a835a010f3437b55242b9d4bf",
}
EXPECTED_CHECKPOINT_SHA256: Mapping[str, str] = {
    "owl_checkpoint": "14aa78ffe7b13e5b3ebf55845bc9a07e339a095cfd88f4c4e8f726b38ce1ebbf",
    "owl_text_cache": "59193fc014d381b2200edf1c1e6dc86324edb55a067189d3e84226a184185283",
    "boxer_checkpoint": "d5a30b348a8f5b0e5990ff3aa0e8f473ce77d860da22586322e7f47abc83ca6f",
    "dino_checkpoint": "4057cbaaad8c16657adb09d6815f28d4164eeba30532fde23f0d17313124caea",
}

THRESHOLD_2D = 0.25
THRESHOLD_3D = 0.50
NMS_IOU_2D = 0.50
IMAGE_HW = 960
TAXONOMY = "lvisplus"
EXPECTED_PROMPT_COUNT = 1220
SEED = 0
PRECISION = "bfloat16"
MAX_FRAME_FILE_BYTES = 64 * 1024 * 1024
MAX_MATRIX_FILE_BYTES = 64 * 1024
MAX_T05_FILE_BYTES = 2 * 1024 * 1024 * 1024
MAX_PROVIDER_CONTRACT_BYTES = 4 * 1024 * 1024

REQUIRED_ENVIRONMENT: Mapping[str, str] = {
    "OPENBLAS_NUM_THREADS": "1",
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "PYTHONHASHSEED": "0",
    "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
    "PYTHONNOUSERSITE": "1",
    "PYTHONDONTWRITEBYTECODE": "1",
}


class FreshProviderError(RuntimeError):
    """A frozen provider, isolation, or exact-input invariant failed."""


@dataclass(frozen=True)
class FrameDatum:
    scene_id: str
    frame_id: int
    boxer_datum: Mapping[str, Any]
    world_offset_absolute: np.ndarray


@dataclass(frozen=True)
class RawBoxRows:
    center: np.ndarray
    extent: np.ndarray
    quaternion: np.ndarray
    score: np.ndarray

    @classmethod
    def empty(cls) -> "RawBoxRows":
        return cls(
            center=np.empty((0, 3), dtype=np.float64),
            extent=np.empty((0, 3), dtype=np.float64),
            quaternion=np.empty((0, 4), dtype=np.float64),
            score=np.empty((0,), dtype=np.float64),
        )


class DatumBuilder(Protocol):
    def __call__(
        self,
        *,
        color_bytes: bytes,
        depth_bytes: bytes,
        intrinsic: np.ndarray,
        pose_absolute: np.ndarray,
        world_offset_absolute: np.ndarray,
        resize: int,
        frame_id: int,
    ) -> Mapping[str, Any]: ...


class Provider(Protocol):
    image_hw: int

    def reset_scene_seed(self, scene_id: str) -> None: ...

    def infer(self, frame: FrameDatum) -> RawBoxRows: ...

    def synchronize(self) -> None: ...

    def provenance(self) -> Mapping[str, Any]: ...


def _sha256_path(path: Path, *, max_bytes: int | None = None) -> str:
    """Hash one explicitly named file without deserializing its content."""

    absolute = Path(os.path.abspath(os.fspath(path)))
    try:
        before = os.stat(absolute, follow_symlinks=True)
    except OSError as error:
        raise FreshProviderError(
            f"cannot stat frozen file {absolute}: {error}"
        ) from error
    if not stat.S_ISREG(before.st_mode):
        raise FreshProviderError(f"frozen input is not a regular file: {absolute}")
    if max_bytes is not None and before.st_size > max_bytes:
        raise FreshProviderError(f"frozen input exceeds byte cap: {absolute}")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    descriptor = os.open(absolute, flags)
    digest = hashlib.sha256()
    total = 0
    try:
        opened = os.fstat(descriptor)
        if (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino):
            raise FreshProviderError(f"frozen input identity changed: {absolute}")
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            total += len(chunk)
            if max_bytes is not None and total > max_bytes:
                raise FreshProviderError(f"frozen input exceeds byte cap: {absolute}")
            digest.update(chunk)
        after = os.fstat(descriptor)
        if (after.st_size, after.st_mtime_ns) != (opened.st_size, opened.st_mtime_ns):
            raise FreshProviderError(f"frozen input changed while hashing: {absolute}")
    finally:
        os.close(descriptor)
    if total != before.st_size:
        raise FreshProviderError(f"short read while hashing frozen input: {absolute}")
    return digest.hexdigest()


def _read_exact_bytes(
    path: Path, expected_sha256: str, *, max_bytes: int, label: str
) -> bytes:
    """Open one manifest-named leaf, verify identity and hash, then return bytes."""

    absolute = Path(os.path.abspath(os.fspath(path)))
    try:
        before = os.lstat(absolute)
    except OSError as error:
        raise FreshProviderError(f"cannot stat {label}: {absolute}: {error}") from error
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
        raise FreshProviderError(
            f"{label} must be a non-symlink regular file: {absolute}"
        )
    if before.st_size > max_bytes:
        raise FreshProviderError(f"{label} exceeds byte cap: {absolute}")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(absolute, flags)
    try:
        opened = os.fstat(descriptor)
        if (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino):
            raise FreshProviderError(f"{label} identity changed: {absolute}")
        chunks: list[bytes] = []
        remaining = max_bytes + 1
        while remaining:
            chunk = os.read(descriptor, min(1024 * 1024, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        payload = b"".join(chunks)
        after = os.fstat(descriptor)
        if (
            len(payload) > max_bytes
            or len(payload) != opened.st_size
            or (after.st_size, after.st_mtime_ns)
            != (opened.st_size, opened.st_mtime_ns)
        ):
            raise FreshProviderError(f"{label} changed or exceeded cap: {absolute}")
    finally:
        os.close(descriptor)
    actual = hashlib.sha256(payload).hexdigest()
    if actual != expected_sha256:
        raise FreshProviderError(
            f"{label} SHA-256 mismatch: expected {expected_sha256}, observed {actual}"
        )
    return payload


def _load_text_matrix(payload: bytes, label: str) -> np.ndarray:
    try:
        value = np.loadtxt(io.BytesIO(payload), dtype=np.float64)
    except (OSError, ValueError) as error:
        raise FreshProviderError(f"cannot parse {label} matrix") from error
    if value.shape != (4, 4) or not np.isfinite(value).all():
        raise FreshProviderError(f"{label} must be a finite 4x4 matrix")
    return np.ascontiguousarray(value)


def _module_file(module: Any, expected_root: Path, label: str) -> Path:
    source = getattr(module, "__file__", None)
    if not isinstance(source, str):
        raise FreshProviderError(f"{label} has no concrete source file")
    path = Path(source).resolve(strict=True)
    root = expected_root.resolve(strict=True)
    try:
        path.relative_to(root)
    except ValueError as error:
        raise FreshProviderError(
            f"{label} imported outside frozen Boxer root: {path}"
        ) from error
    return path


def _import_external_module(boxer_root: Path, name: str) -> Any:
    root_text = os.fspath(boxer_root.resolve(strict=True))
    if root_text not in sys.path:
        sys.path.insert(0, root_text)
    module = importlib.import_module(name)
    _module_file(module, boxer_root, name)
    return module


def _build_boxer_datum(
    *,
    boxer_root: Path,
    color_bytes: bytes,
    depth_bytes: bytes,
    intrinsic: np.ndarray,
    pose_absolute: np.ndarray,
    world_offset_absolute: np.ndarray,
    resize: int,
    frame_id: int,
) -> Mapping[str, Any]:
    """Reproduce the released ScanNet frame math without a dataset loader."""

    cv2 = importlib.import_module("cv2")
    torch = importlib.import_module("torch")
    camera_module = _import_external_module(boxer_root, "utils.tw.camera")
    pose_module = _import_external_module(boxer_root, "utils.tw.pose")

    color_bgr = cv2.imdecode(
        np.frombuffer(color_bytes, dtype=np.uint8), cv2.IMREAD_COLOR
    )
    if color_bgr is None or color_bgr.ndim != 3 or color_bgr.shape[2] != 3:
        raise FreshProviderError(
            f"frame {frame_id} color payload is not a 3-channel image"
        )
    color_rgb = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)
    source_h, source_w = color_rgb.shape[:2]
    if source_h <= 0 or source_w <= 0:
        raise FreshProviderError(f"frame {frame_id} color image is empty")
    scale_x = float(resize) / float(source_w)
    scale_y = float(resize) / float(source_h)
    color_rgb = cv2.resize(color_rgb, (resize, resize), interpolation=cv2.INTER_LINEAR)
    color_rgb = np.ascontiguousarray(color_rgb)

    depth_raw = cv2.imdecode(
        np.frombuffer(depth_bytes, dtype=np.uint8), cv2.IMREAD_UNCHANGED
    )
    if depth_raw is None or depth_raw.ndim != 2:
        raise FreshProviderError(f"frame {frame_id} depth payload is not a 2D image")
    depth_m = depth_raw.astype(np.float32) / 1000.0
    if depth_m.shape != (resize, resize):
        depth_m = cv2.resize(depth_m, (resize, resize), interpolation=cv2.INTER_NEAREST)
    depth_m = np.ascontiguousarray(depth_m, dtype=np.float32)

    fx = float(intrinsic[0, 0]) * scale_x
    fy = float(intrinsic[1, 1]) * scale_y
    cx = float(intrinsic[0, 2]) * scale_x
    cy = float(intrinsic[1, 2]) * scale_y
    if (
        not all(math.isfinite(value) for value in (fx, fy, cx, cy))
        or fx <= 0
        or fy <= 0
    ):
        raise FreshProviderError("invalid scaled camera intrinsics")

    image = torch.from_numpy(color_rgb).permute(2, 0, 1).float().unsqueeze(0) / 255.0
    transform = np.array(pose_absolute, dtype=np.float64, order="C", copy=True)
    transform[:3, 3] -= world_offset_absolute
    try:
        np.linalg.inv(transform)
    except np.linalg.LinAlgError as error:
        raise FreshProviderError(f"frame {frame_id} pose is singular") from error
    transform32 = transform.astype(np.float32)

    camera_to_rig = torch.tensor(
        [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0], dtype=torch.float32
    )
    camera_values = torch.tensor(
        [resize, resize, fx, fy, cx, cy, -1, 1e-3, resize, resize],
        dtype=torch.float32,
    )
    camera = camera_module.CameraTW(torch.cat([camera_values, camera_to_rig])).float()
    pose_values = torch.tensor(
        [*transform32[:3, :3].reshape(-1), *transform32[:3, 3]],
        dtype=torch.float32,
    )
    world_from_rig = pose_module.PoseTW(pose_values)

    # This is the released BaseLoader static math, kept synchronous here.  The
    # global NumPy seed is reset exactly once at each scene boundary.
    height, width = depth_m.shape
    target_count = 10_000
    step = max(1, int(np.sqrt(height * width / (target_count * 2))))
    ys_grid, xs_grid = np.mgrid[0:height:step, 0:width:step]
    ys_flat = ys_grid.ravel()
    xs_flat = xs_grid.ravel()
    depths = depth_m[ys_flat, xs_flat]
    valid = depths > 0
    ys_valid = ys_flat[valid]
    xs_valid = xs_flat[valid]
    depths_valid = depths[valid]
    if len(ys_valid) > target_count:
        chosen = np.random.choice(len(ys_valid), size=target_count, replace=False)
        ys_valid = ys_valid[chosen]
        xs_valid = xs_valid[chosen]
        depths_valid = depths_valid[chosen]
    if len(ys_valid):
        x3d = (xs_valid.astype(np.float32) - cx) / fx * depths_valid
        y3d = (ys_valid.astype(np.float32) - cy) / fy * depths_valid
        camera_points = np.stack([x3d, y3d, depths_valid], axis=-1)
        world_points = camera_points @ transform32[:3, :3].T + transform32[:3, 3]
        sdp_world = torch.from_numpy(np.ascontiguousarray(world_points)).float()
        if len(sdp_world) < target_count:
            padding = torch.full(
                (target_count - len(sdp_world), 3), float("nan"), dtype=torch.float32
            )
            sdp_world = torch.cat([sdp_world, padding], dim=0)
    else:
        sdp_world = torch.zeros((0, 3), dtype=torch.float32)

    return {
        "img0": image,
        "cam0": camera,
        "T_world_rig0": world_from_rig,
        "sdp_w": sdp_world,
        "time_ns0": int(frame_id),
    }


class ManifestScanNetFrameReader:
    """Strict synchronous reader over ``bundle.ordered_frames`` only."""

    def __init__(
        self,
        bundle: ExactScheduleBundle,
        scene_root: Path,
        *,
        boxer_root: Path = BOXER_ROOT,
        resize: int = IMAGE_HW,
        datum_builder: DatumBuilder | None = None,
    ) -> None:
        self.bundle = bundle
        self.scene_root = Path(os.path.abspath(os.fspath(scene_root)))
        self.boxer_root = Path(os.path.abspath(os.fspath(boxer_root)))
        self.resize = int(resize)
        if self.resize != IMAGE_HW:
            raise FreshProviderError(f"reader resize must remain {IMAGE_HW}")
        self._ordered = bundle.ordered_frames
        self._next_index = 0
        self._active_scene: str | None = None
        self._intrinsic: np.ndarray | None = None
        self._world_offset: np.ndarray | None = None
        if datum_builder is None:
            self._datum_builder: DatumBuilder = lambda **kwargs: _build_boxer_datum(
                boxer_root=self.boxer_root, **kwargs
            )
        else:
            self._datum_builder = datum_builder

    @property
    def completed_frame_count(self) -> int:
        return self._next_index

    def _start_scene(self, scene: SceneSchedule) -> None:
        scene_dir = self.scene_root / scene.scene_id
        intrinsic_payload = _read_exact_bytes(
            scene_dir / scene.intrinsic_color_relpath,
            scene.intrinsic_color_sha256,
            max_bytes=MAX_MATRIX_FILE_BYTES,
            label=f"{scene.scene_id} intrinsic",
        )
        intrinsic = _load_text_matrix(intrinsic_payload, f"{scene.scene_id} intrinsic")
        self._active_scene = scene.scene_id
        self._intrinsic = intrinsic
        self._world_offset = None

    def read(self, scene: SceneSchedule, frame: ScheduledFrame) -> FrameDatum:
        if self._next_index >= len(self._ordered):
            raise FreshProviderError("reader has no remaining scheduled frame")
        expected_scene, expected_frame = self._ordered[self._next_index]
        if scene is not expected_scene or frame is not expected_frame:
            raise FreshProviderError(
                "reader request is not the exact next schedule object: "
                f"expected {expected_scene.scene_id}/{expected_frame.frame_id}"
            )
        if self._active_scene != scene.scene_id:
            self._start_scene(scene)
        if self._intrinsic is None:
            raise FreshProviderError("intrinsic was not initialized")

        scene_dir = self.scene_root / scene.scene_id
        color_payload = _read_exact_bytes(
            scene_dir / frame.color_relpath,
            frame.color_sha256,
            max_bytes=MAX_FRAME_FILE_BYTES,
            label=f"{scene.scene_id}/{frame.frame_id} color",
        )
        depth_payload = _read_exact_bytes(
            scene_dir / frame.depth_relpath,
            frame.depth_sha256,
            max_bytes=MAX_FRAME_FILE_BYTES,
            label=f"{scene.scene_id}/{frame.frame_id} depth",
        )
        pose_payload = _read_exact_bytes(
            scene_dir / frame.pose_relpath,
            frame.pose_sha256,
            max_bytes=MAX_MATRIX_FILE_BYTES,
            label=f"{scene.scene_id}/{frame.frame_id} pose",
        )
        pose_absolute = _load_text_matrix(
            pose_payload, f"{scene.scene_id}/{frame.frame_id} pose"
        )
        if self._world_offset is None:
            if frame is not scene.frames[0]:
                raise FreshProviderError(
                    "world offset can only be set by the first exact frame"
                )
            self._world_offset = np.array(
                pose_absolute[:3, 3], dtype=np.float64, copy=True
            )
        world_offset = np.array(self._world_offset, dtype=np.float64, copy=True)
        boxer_datum = self._datum_builder(
            color_bytes=color_payload,
            depth_bytes=depth_payload,
            intrinsic=np.array(self._intrinsic, copy=True),
            pose_absolute=pose_absolute,
            world_offset_absolute=world_offset,
            resize=self.resize,
            frame_id=frame.frame_id,
        )
        self._next_index += 1
        world_offset.setflags(write=False)
        return FrameDatum(
            scene_id=scene.scene_id,
            frame_id=frame.frame_id,
            boxer_datum=boxer_datum,
            world_offset_absolute=world_offset,
        )


class FrozenBoxerProvider:
    """One process-wide frozen OWLv2 -> BoxerNet inference stack."""

    image_hw = IMAGE_HW

    def __init__(self, boxer_root: Path, boxer_checkpoint: Path, owl_checkpoint: Path):
        self.boxer_root = boxer_root.resolve(strict=True)
        self.device = "cuda"
        torch = importlib.import_module("torch")
        if not torch.cuda.is_available():
            raise FreshProviderError("formal frozen provider requires CUDA")
        if not torch.cuda.is_bf16_supported():
            raise FreshProviderError("formal frozen provider requires CUDA bfloat16")
        self.torch = torch

        taxonomy_module = _import_external_module(self.boxer_root, "utils.taxonomy")
        owl_module = _import_external_module(self.boxer_root, "owl.owl_wrapper")
        tokenizer_module = _import_external_module(
            self.boxer_root, "owl.clip_tokenizer"
        )
        boxernet_module = _import_external_module(self.boxer_root, "boxernet.boxernet")
        owl_module._CKPT_PATH = os.fspath(owl_checkpoint)
        tokenizer_module._CKPT_PATH = os.fspath(owl_checkpoint)
        labels = taxonomy_module.load_text_labels([TAXONOMY])
        if len(labels) != EXPECTED_PROMPT_COUNT:
            raise FreshProviderError(
                f"{TAXONOMY} prompt count changed: {len(labels)} != {EXPECTED_PROMPT_COUNT}"
            )
        self.labels = tuple(labels)
        self.owl = owl_module.OwlWrapper(
            self.device,
            text_prompts=list(self.labels),
            min_confidence=THRESHOLD_2D,
            precision=PRECISION,
            warmup=True,
            nms_iou_threshold=NMS_IOU_2D,
        )
        self.boxernet = boxernet_module.BoxerNet.load_from_checkpoint(
            os.fspath(boxer_checkpoint), device=self.device
        )
        if int(self.boxernet.hw) != IMAGE_HW:
            raise FreshProviderError(
                f"Boxer checkpoint image size changed: {self.boxernet.hw} != {IMAGE_HW}"
            )
        self.synchronize()

    def synchronize(self) -> None:
        self.torch.cuda.synchronize()

    def reset_scene_seed(self, scene_id: str) -> None:
        del scene_id
        random.seed(SEED)
        np.random.seed(SEED)
        self.torch.manual_seed(SEED)
        self.torch.cuda.manual_seed_all(SEED)

    def infer(self, frame: FrameDatum) -> RawBoxRows:
        torch = self.torch
        datum = dict(frame.boxer_datum)
        image_255 = datum["img0"].clone() * 255.0
        bb2d, scores2d, label_ids, _ = self.owl.forward(
            image_255, resize_to_HW=(IMAGE_HW, IMAGE_HW)
        )
        count_2d = int(bb2d.shape[0])
        if tuple(bb2d.shape) != (count_2d, 4):
            raise FreshProviderError("OWLv2 returned malformed 2D boxes")
        if tuple(scores2d.shape) != (count_2d,) or int(len(label_ids)) != count_2d:
            raise FreshProviderError("OWLv2 returned inconsistent result lengths")
        if count_2d == 0:
            return RawBoxRows.empty()

        datum["bb2d"] = bb2d
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            outputs = self.boxernet.forward(datum)
        predicted = outputs["obbs_pr_w"].cpu()[0]
        if len(predicted) != count_2d:
            raise FreshProviderError("BoxerNet row count differs from OWLv2")
        scores3d = predicted.prob.squeeze(-1).clone()
        keep = scores3d >= THRESHOLD_3D
        filtered = predicted[keep].clone()
        scores3d = scores3d[keep]
        scores2d_kept = scores2d[keep]
        mean_scores = (scores2d_kept + scores3d) / 2.0
        count = int(len(filtered))
        if count == 0:
            return RawBoxRows.empty()

        center_recentered = filtered.T_world_object.t.numpy().astype(np.float64)
        center_absolute = center_recentered + frame.world_offset_absolute.reshape(1, 3)
        quaternion = filtered.T_world_object.q.numpy().astype(np.float64)
        extent = filtered.bb3_diagonal.numpy().astype(np.float64)
        score = mean_scores.numpy().astype(np.float64)
        return RawBoxRows(
            center=np.ascontiguousarray(center_absolute),
            extent=np.ascontiguousarray(extent),
            quaternion=np.ascontiguousarray(quaternion),
            score=np.ascontiguousarray(score),
        )

    def provenance(self) -> Mapping[str, Any]:
        properties = self.torch.cuda.get_device_properties(
            self.torch.cuda.current_device()
        )
        return {
            "device": "cuda",
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "cuda_device_name": properties.name,
            "cuda_device_total_memory_bytes": int(properties.total_memory),
            "torch_version": str(self.torch.__version__),
            "cuda_version": str(self.torch.version.cuda),
            "owl_instance_count": 1,
            "boxernet_instance_count": 1,
            "prompt_count": len(self.labels),
            "boxer_image_hw": int(self.boxernet.hw),
            "owl_use_bfloat16": bool(self.owl.use_bfloat16),
            "cuda_max_memory_allocated_bytes": int(
                self.torch.cuda.max_memory_allocated()
            ),
            "cuda_max_memory_reserved_bytes": int(
                self.torch.cuda.max_memory_reserved()
            ),
        }


def _validate_environment() -> dict[str, str | None]:
    observed = {name: os.environ.get(name) for name in REQUIRED_ENVIRONMENT}
    mismatches = {
        name: {"expected": expected, "observed": observed[name]}
        for name, expected in REQUIRED_ENVIRONMENT.items()
        if observed[name] != expected
    }
    if mismatches:
        raise FreshProviderError(
            "formal environment must be pinned before Python starts: "
            + json.dumps(mismatches, sort_keys=True)
        )
    return observed


def _git_text(boxer_root: Path, *arguments: str) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", os.fspath(boxer_root), *arguments],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise FreshProviderError(
            f"cannot audit frozen Boxer git checkout: {error}"
        ) from error
    return result.stdout.strip()


def _validate_provider_contract(path: Path, expected_sha256: str) -> Path:
    if (
        not isinstance(expected_sha256, str)
        or len(expected_sha256) != 64
        or any(character not in "0123456789abcdef" for character in expected_sha256)
    ):
        raise FreshProviderError(
            "expected provider-contract SHA-256 must be 64 lowercase hex characters"
        )
    absolute = Path(os.path.abspath(os.fspath(path)))
    try:
        metadata = os.lstat(absolute)
    except OSError as error:
        raise FreshProviderError(f"missing provider contract: {absolute}") from error
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise FreshProviderError("provider contract must be a non-symlink regular file")
    digest = _sha256_path(absolute, max_bytes=MAX_PROVIDER_CONTRACT_BYTES)
    if digest != expected_sha256:
        raise FreshProviderError(
            f"provider contract SHA-256 mismatch: expected {expected_sha256}, observed {digest}"
        )
    return absolute


def _validate_frozen_assets(
    schedule_path: Path,
    boxer_root: Path,
    provider_contract: Path,
    expected_provider_contract_sha256: str,
) -> tuple[dict[str, dict[str, Any]], dict[str, Path]]:
    provider_contract = _validate_provider_contract(
        provider_contract, expected_provider_contract_sha256
    )
    if (
        _sha256_path(schedule_path, max_bytes=32 * 1024 * 1024)
        != EXPECTED_SCHEDULE_SHA256
    ):
        raise FreshProviderError("exact H10 schedule bytes differ from frozen V2")
    commit = _git_text(boxer_root, "rev-parse", "HEAD")
    if commit != EXPECTED_BOXER_COMMIT:
        raise FreshProviderError(f"Boxer commit differs: {commit}")
    if _git_text(boxer_root, "status", "--porcelain"):
        raise FreshProviderError("Boxer checkout is not clean")

    paths: dict[str, Path] = {
        "schedule": schedule_path,
        "holdout_list": HOLDOUT_LIST_PATH,
        "provider_contract": provider_contract,
        "owl_checkpoint": OWL_CHECKPOINT,
        "owl_text_cache": OWL_TEXT_CACHE,
        "boxer_checkpoint": boxer_root / BOXER_CHECKPOINT_RELPATH,
        "dino_checkpoint": boxer_root / DINO_CHECKPOINT_RELPATH,
        "runner_source": Path(__file__).resolve(strict=True),
        "provider_core_source": REPOSITORY_ROOT
        / "boxfusion"
        / "s3r_h10_provider_core.py",
    }
    for relative in EXPECTED_EXTERNAL_CODE_SHA256:
        paths[f"boxer_code:{relative}"] = boxer_root / relative

    expected: dict[str, str | None] = {
        "schedule": EXPECTED_SCHEDULE_SHA256,
        "holdout_list": EXPECTED_HOLDOUT_LIST_SHA256,
        "provider_contract": expected_provider_contract_sha256,
        "owl_checkpoint": EXPECTED_CHECKPOINT_SHA256["owl_checkpoint"],
        "owl_text_cache": EXPECTED_CHECKPOINT_SHA256["owl_text_cache"],
        "boxer_checkpoint": EXPECTED_CHECKPOINT_SHA256["boxer_checkpoint"],
        "dino_checkpoint": EXPECTED_CHECKPOINT_SHA256["dino_checkpoint"],
        "runner_source": None,
        "provider_core_source": None,
    }
    for relative, digest in EXPECTED_EXTERNAL_CODE_SHA256.items():
        expected[f"boxer_code:{relative}"] = digest

    ledger: dict[str, dict[str, Any]] = {}
    for name, path in paths.items():
        digest = _sha256_path(path)
        required = expected[name]
        if required is not None and digest != required:
            raise FreshProviderError(
                f"frozen asset {name} differs: expected {required}, observed {digest}"
            )
        ledger[name] = {
            "path": os.fspath(path),
            "sha256_before": digest,
            "expected_sha256": required,
        }
    return ledger, paths


def _rehash_assets(
    ledger: Mapping[str, Mapping[str, Any]], paths: Mapping[str, Path]
) -> dict[str, str]:
    after: dict[str, str] = {}
    for name, path in paths.items():
        digest = _sha256_path(path)
        if digest != ledger[name]["sha256_before"]:
            raise FreshProviderError(f"frozen asset changed during inference: {name}")
        after[name] = digest
    return after


def _hash_formal_t05(bundle: ExactScheduleBundle) -> dict[str, str]:
    ledger: dict[str, str] = {}
    for scene in bundle.scenes:
        path = REPOSITORY_ROOT / scene.formal_t05_relpath
        digest = _sha256_path(path, max_bytes=MAX_T05_FILE_BYTES)
        if digest != scene.formal_t05_sha256:
            raise FreshProviderError(f"formal T05 hash mismatch for {scene.scene_id}")
        ledger[scene.scene_id] = digest
    return ledger


def _rehash_exact_frame_inputs(
    bundle: ExactScheduleBundle, scene_root: Path
) -> dict[str, Any]:
    """Re-verify every manifest-named input after the complete stream."""

    digest = hashlib.sha256()
    file_count = 0
    for scene in bundle.scenes:
        scene_dir = scene_root / scene.scene_id
        entries = [
            (
                "intrinsic",
                scene.intrinsic_color_relpath,
                scene.intrinsic_color_sha256,
                MAX_MATRIX_FILE_BYTES,
            )
        ]
        for frame in scene.frames:
            entries.extend(
                (
                    (
                        f"{frame.frame_id}:color",
                        frame.color_relpath,
                        frame.color_sha256,
                        MAX_FRAME_FILE_BYTES,
                    ),
                    (
                        f"{frame.frame_id}:depth",
                        frame.depth_relpath,
                        frame.depth_sha256,
                        MAX_FRAME_FILE_BYTES,
                    ),
                    (
                        f"{frame.frame_id}:pose",
                        frame.pose_relpath,
                        frame.pose_sha256,
                        MAX_MATRIX_FILE_BYTES,
                    ),
                )
            )
        for role, relative, expected, maximum in entries:
            _read_exact_bytes(
                scene_dir / relative,
                expected,
                max_bytes=maximum,
                label=f"post-stream {scene.scene_id}/{role}",
            )
            digest.update(scene.scene_id.encode("ascii"))
            digest.update(b"\0")
            digest.update(role.encode("ascii"))
            digest.update(b"\0")
            digest.update(relative.encode("ascii"))
            digest.update(b"\0")
            digest.update(expected.encode("ascii"))
            digest.update(b"\n")
            file_count += 1
    expected_count = len(bundle.scenes) + 3 * bundle.valid_frame_count
    if file_count != expected_count:
        raise FreshProviderError("post-stream exact-input file count differs")
    return {
        "verified_file_count": file_count,
        "expected_file_count": expected_count,
        "exact_input_ledger_sha256": digest.hexdigest(),
    }


def _canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        + "\n"
    ).encode("ascii")


def _percentiles(values: Sequence[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or not len(array) or not np.isfinite(array).all():
        raise FreshProviderError("runtime ledger is empty or non-finite")
    return {
        "p50_seconds": float(np.percentile(array, 50)),
        "p95_seconds": float(np.percentile(array, 95)),
        "max_seconds": float(np.max(array)),
    }


def _validate_raw_rows(rows: RawBoxRows) -> RawBoxRows:
    if not isinstance(rows, RawBoxRows):
        raise FreshProviderError("provider must return RawBoxRows")
    center = np.asarray(rows.center, dtype=np.float64)
    extent = np.asarray(rows.extent, dtype=np.float64)
    quaternion = np.asarray(rows.quaternion, dtype=np.float64)
    score = np.asarray(rows.score, dtype=np.float64)
    count = len(center)
    if (
        center.shape != (count, 3)
        or extent.shape != (count, 3)
        or quaternion.shape != (count, 4)
        or score.shape != (count,)
    ):
        raise FreshProviderError("provider raw arrays have malformed shapes")
    if not all(
        np.isfinite(value).all() for value in (center, extent, quaternion, score)
    ):
        raise FreshProviderError("provider raw arrays contain non-finite values")
    if count > 2048:
        raise FreshProviderError("provider raw row cap exceeded")
    if count and np.max(np.abs(center)) > 10_000.0:
        raise FreshProviderError("provider returned an implausible absolute center")
    if np.any(extent <= 0.0) or (count and np.max(extent) > 100.0):
        raise FreshProviderError("provider returned an invalid extent")
    if np.any(score < 0.0) or np.any(score > 1.0):
        raise FreshProviderError("provider returned a score outside [0,1]")
    quaternion_norm = np.linalg.norm(quaternion, axis=1)
    if count and not np.allclose(quaternion_norm, np.ones(count), rtol=0.0, atol=5e-3):
        raise FreshProviderError("provider returned a non-unit OBB quaternion")
    if count:
        quaternion = quaternion / quaternion_norm[:, None]
    return RawBoxRows(
        center=np.ascontiguousarray(center),
        extent=np.ascontiguousarray(extent),
        quaternion=np.ascontiguousarray(quaternion),
        score=np.ascontiguousarray(score),
    )


def _execute_stream(
    *,
    bundle: ExactScheduleBundle,
    output_root: Path,
    provider_factory: Callable[[], Provider],
    reader_factory: Callable[[Provider], ManifestScanNetFrameReader],
    environment: Mapping[str, Any],
    asset_ledger: Mapping[str, Mapping[str, Any]],
    t05_before: Mapping[str, str],
    t05_after_fn: Callable[[], Mapping[str, str]],
    immutable_recheck_fn: Callable[[], Mapping[str, str]],
    frame_input_recheck_fn: Callable[[], Mapping[str, Any]],
    clock: Callable[[], float] = time.perf_counter,
    transaction_factory: Callable[[Path, ExactScheduleBundle], Any] = FrameTransaction,
) -> dict[str, Any]:
    """Execute one complete ordered stream; dependency injection supports tests."""

    cold_start_begin = clock()
    provider = provider_factory()
    provider.synchronize()
    cold_start_seconds = float(clock() - cold_start_begin)
    if not math.isfinite(cold_start_seconds) or cold_start_seconds < 0:
        raise FreshProviderError("cold-start timer is invalid")
    reader = reader_factory(provider)
    runtime_rows: list[dict[str, Any]] = []
    total_raw_rows = 0
    active_scene: str | None = None

    transaction = transaction_factory(output_root, bundle)
    try:
        for scene, frame in bundle.ordered_frames:
            if active_scene != scene.scene_id:
                provider.reset_scene_seed(scene.scene_id)
                active_scene = scene.scene_id
            token = transaction.begin(scene.scene_id, frame.frame_id)
            provider.synchronize()
            frame_begin = clock()
            datum = reader.read(scene, frame)
            raw = _validate_raw_rows(provider.infer(datum))
            provider.synchronize()
            precommit_seconds = float(clock() - frame_begin)
            if not math.isfinite(precommit_seconds) or precommit_seconds < 0:
                raise FreshProviderError("precommit timer is invalid")
            count = len(raw.center)
            commit = transaction.commit(
                token,
                center=raw.center,
                extent=raw.extent,
                quaternion=raw.quaternion,
                score=raw.score,
                source_row=np.arange(count, dtype=np.int64),
                runtime_seconds=precommit_seconds,
            )
            end_to_end_seconds = float(clock() - frame_begin)
            if (
                not math.isfinite(end_to_end_seconds)
                or end_to_end_seconds < precommit_seconds
            ):
                raise FreshProviderError("end-to-end frame timer is invalid")
            runtime_rows.append(
                {
                    "scene_id": scene.scene_id,
                    "frame_id": frame.frame_id,
                    "row_count": int(commit.row_count),
                    "precommit_compute_seconds": precommit_seconds,
                    "end_to_end_seconds": end_to_end_seconds,
                }
            )
            total_raw_rows += int(commit.row_count)

        if reader.completed_frame_count != bundle.valid_frame_count:
            raise FreshProviderError("reader did not consume every exact valid frame")
        if transaction.completed_frame_count != bundle.valid_frame_count:
            raise FreshProviderError(
                "transaction did not commit every exact valid frame"
            )
        t05_after = dict(t05_after_fn())
        if dict(t05_before) != t05_after:
            raise FreshProviderError("formal T05 files changed during shadow inference")
        frame_input_recheck = dict(frame_input_recheck_fn())
        immutable_after = dict(immutable_recheck_fn())

        precommit_values = [row["precommit_compute_seconds"] for row in runtime_rows]
        end_to_end_values = [row["end_to_end_seconds"] for row in runtime_rows]
        if len(end_to_end_values) < 2:
            raise FreshProviderError("runtime ledger needs a cold frame and warm frames")
        cold_first_frame = runtime_rows[0]
        warm_end_to_end_values = end_to_end_values[1:]
        peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # Linux reports ru_maxrss in KiB.  This formal runner is Linux-only
        # because FrameTransaction relies on POSIX fsync/link semantics.
        process_peak_rss_bytes = int(peak_rss) * 1024
        provenance: dict[str, Any] = {
            "schema": RUN_SCHEMA,
            "audit_complete": True,
            "shadow_only": True,
            "birth_enabled": False,
            "ap_evaluated": False,
            "gt_used": False,
            "target_dataset_training_used": False,
            "schedule": {
                "schema": bundle.schema,
                "sha256": bundle.sha256,
                "scene_order": list(bundle.scene_order),
                "valid_frame_count": bundle.valid_frame_count,
                "raw_frame_count": bundle.raw_frame_count,
                "excluded_frame_count": bundle.raw_frame_count
                - bundle.valid_frame_count,
            },
            "provider_contract": {
                "model_process_count": 1,
                "owl_instance_count": 1,
                "boxernet_instance_count": 1,
                "taxonomy": TAXONOMY,
                "prompt_count": EXPECTED_PROMPT_COUNT,
                "threshold_2d": THRESHOLD_2D,
                "nms_iou_2d": NMS_IOU_2D,
                "threshold_3d": THRESHOLD_3D,
                "score_rule": "mean(owl_2d_score,boxer_3d_score)_after_3d_threshold",
                "image_hw": [IMAGE_HW, IMAGE_HW],
                "precision": PRECISION,
                "seed": SEED,
                "temporal_state": False,
                "prefetch": False,
                "frame_directory_enumeration": False,
                "coordinate_convention": (
                    "absolute_scannet_world=center_boxer_recentered+"
                    "translation_of_first_valid_exact_schedule_pose;"
                    "extent_unchanged;Hamilton_wxyz_quaternion_l2_normalized"
                ),
            },
            "model_runtime": dict(provider.provenance()),
            "environment": dict(environment),
            "frozen_assets": {
                name: {**dict(record), "sha256_after": immutable_after[name]}
                for name, record in asset_ledger.items()
            },
            "formal_t05": {
                "deserialized": False,
                "before_sha256": dict(t05_before),
                "after_sha256": t05_after,
                "byte_identical": True,
            },
            "frame_inputs": {
                "before_each_frame_read_verified": True,
                "after_complete_stream_verified": True,
                "frame_inputs_before_read_and_after_stream_verified": True,
                **frame_input_recheck,
            },
            "runtime": {
                "cold_start_model_load_and_warmup_seconds": cold_start_seconds,
                "cold_first_frame": dict(cold_first_frame),
                "cold_first_frame_end_to_end_seconds": cold_first_frame[
                    "end_to_end_seconds"
                ],
                "cold_start_total_seconds": cold_start_seconds
                + cold_first_frame["end_to_end_seconds"],
                "precommit_compute_definition": (
                    "current-frame verified reads + synchronous datum construction + "
                    "OWL + Boxer + CUDA synchronize; excludes persistence"
                ),
                "end_to_end_definition": (
                    "precommit compute + frame NPZ fsync + frame-directory fsync + "
                    "journal fsync"
                ),
                "precommit_compute_summary": _percentiles(precommit_values),
                "all_frame_end_to_end_summary": _percentiles(end_to_end_values),
                "warm_frame_end_to_end_summary": _percentiles(
                    warm_end_to_end_values
                ),
                "warm_frame_count": len(warm_end_to_end_values),
                "deadline_uses": (
                    "warm_frame_end_to_end_summary_after_global_first_committed_frame"
                ),
                "process_peak_rss_bytes": process_peak_rss_bytes,
                "integrated_realtime_qualified": False,
                "frames": runtime_rows,
            },
            "output": {
                "committed_frame_count": bundle.valid_frame_count,
                "raw_row_count": total_raw_rows,
                "empty_frame_count": sum(row["row_count"] == 0 for row in runtime_rows),
                "native_prediction_mutation": False,
                "tracked_csv_created": False,
            },
        }
        provenance_payload = _canonical_json_bytes(provenance)
        provenance_hash = transaction.publish_run_provenance(provenance_payload)
        final_seal = transaction.seal(run_provenance_sha256=provenance_hash)
        return {
            "provenance": provenance,
            "provenance_sha256": provenance_hash,
            "final_seal": final_seal,
        }
    finally:
        transaction.close()


def _validated_shadow_output(path: Path) -> Path:
    absolute = Path(os.path.abspath(os.fspath(path)))
    allowed = Path(os.path.abspath(os.fspath(SHADOW_OUTPUT_ROOT)))
    try:
        relative = absolute.relative_to(allowed)
    except ValueError as error:
        raise FreshProviderError(
            f"output must be below the shadow log root: {allowed}"
        ) from error
    if not relative.parts:
        raise FreshProviderError("output cannot equal the shadow log root")
    native = Path(os.path.abspath(os.fspath(FORMAL_T05_ROOT)))
    try:
        absolute.relative_to(native)
    except ValueError:
        pass
    else:  # pragma: no cover - native lies outside the allowed root by construction
        raise FreshProviderError("output cannot overlap formal T05")
    return absolute


def run_fresh_provider(
    output_root: Path,
    *,
    provider_contract: Path,
    expected_provider_contract_sha256: str,
) -> dict[str, Any]:
    environment = _validate_environment()
    output = _validated_shadow_output(output_root)
    asset_ledger, asset_paths = _validate_frozen_assets(
        SCHEDULE_PATH,
        BOXER_ROOT,
        provider_contract,
        expected_provider_contract_sha256,
    )
    bundle = parse_exact_schedule_bundle(SCHEDULE_PATH)
    if bundle.sha256 != EXPECTED_SCHEDULE_SHA256:
        raise FreshProviderError("parsed schedule hash differs from frozen V2")
    t05_before = _hash_formal_t05(bundle)
    boxer_checkpoint = BOXER_ROOT / BOXER_CHECKPOINT_RELPATH

    return _execute_stream(
        bundle=bundle,
        output_root=output,
        provider_factory=lambda: FrozenBoxerProvider(
            BOXER_ROOT, boxer_checkpoint, OWL_CHECKPOINT
        ),
        reader_factory=lambda provider: ManifestScanNetFrameReader(
            bundle,
            SCENE_ROOT,
            boxer_root=BOXER_ROOT,
            resize=provider.image_hw,
        ),
        environment=environment,
        asset_ledger=asset_ledger,
        t05_before=t05_before,
        t05_after_fn=lambda: _hash_formal_t05(bundle),
        immutable_recheck_fn=lambda: _rehash_assets(asset_ledger, asset_paths),
        frame_input_recheck_fn=lambda: _rehash_exact_frame_inputs(bundle, SCENE_ROOT),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fresh no-GT exact-frame S3R H10 Boxer shadow provider"
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="fresh create-only directory below the repository logs/ root",
    )
    parser.add_argument(
        "--provider-contract",
        type=Path,
        required=True,
        help="frozen preregistration/contract file (hashed only, never parsed)",
    )
    parser.add_argument(
        "--expected-provider-contract-sha256",
        required=True,
        help="predeclared lowercase SHA-256 of --provider-contract",
    )
    args = parser.parse_args()
    result = run_fresh_provider(
        args.output_root,
        provider_contract=args.provider_contract,
        expected_provider_contract_sha256=args.expected_provider_contract_sha256,
    )
    print(
        json.dumps(
            {
                "output_root": os.fspath(args.output_root),
                "provenance_sha256": result["provenance_sha256"],
                "completed_frame_count": result["final_seal"]["completed_frame_count"],
                "shadow_only": True,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
