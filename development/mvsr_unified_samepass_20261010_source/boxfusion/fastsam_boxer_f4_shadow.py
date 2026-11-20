"""Frozen BoxerNet geometry hypotheses for sealed FastSAM sources.

This module is the model-facing core of the F4 shadow experiment.  It accepts
one current RGB-D frame and the sealed FastSAM ``tight_box_xyxy`` rows, invokes
the released BoxerNet exactly once for the non-empty frame, and returns one
row-aligned world-frame OBB hypothesis (``HB``) per source identity.

The module deliberately has no annotation, evaluator, native-prediction,
semantic, association, score-selection, or birth dependency.  Boxer
confidence and uncertainty are diagnostics only; every output row is retained
and geometry validity is decided without a probability threshold.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import dataclass
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from types import MappingProxyType
from typing import Any, Callable, Mapping, Optional, Sequence

import numpy as np


SCHEMA = "boxfusion.fastsam_boxer_f4_shadow.v1"
MODE = "shadow"
PROTOCOL_ID = "F4-FASTSAM-BOXER-GEOMETRY-SHADOW-PAPER100"

IMAGE_HEIGHT = 480
IMAGE_WIDTH = 640
BOXER_IMAGE_HW = 960
MAX_SOURCES_PER_FRAME = 16
SDP_SAMPLES = 10_000
SEED = 0
CAMERA_DEPTH_EPSILON_M = 1e-4
ROTATION_TOLERANCE = 5e-3

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ADAPTER_PATH = (
    REPOSITORY_ROOT
    / "tools/boxfusion_tr3d_pipeline/boxfusion/boxer_lifter.py"
)
DEFAULT_BOXER_ROOT = Path(
    "/data/ZhaoX/OVM3D-Dett/boxfusion_boxer_dev/third_party/boxer"
)
DEFAULT_BOXER_CHECKPOINT = (
    DEFAULT_BOXER_ROOT / "ckpts/boxernet_hw960in2x6d768-c88128f8.ckpt"
)
DEFAULT_DINOV3_CHECKPOINT = (
    DEFAULT_BOXER_ROOT
    / "ckpts/dinov3_vits16plus_pretrain_lvd1689m-4057cbaa.pth"
)
DEFAULT_BOXERNET_SOURCE = DEFAULT_BOXER_ROOT / "boxernet/boxernet.py"

BOXER_REPOSITORY_COMMIT = "1f86542dc342a4b1d474c87c97c5d1d6566d9148"
BOXER_CHECKPOINT_SHA256 = (
    "d5a30b348a8f5b0e5990ff3aa0e8f473ce77d860da22586322e7f47abc83ca6f"
)
DINOV3_CHECKPOINT_SHA256 = (
    "4057cbaaad8c16657adb09d6815f28d4164eeba30532fde23f0d17313124caea"
)
BOXERNET_SOURCE_SHA256 = (
    "a8009c1c0932aaab98bb074a2a4c50e55a3fbdfc3c6cb1afc9e1aef0e5324130"
)
ADAPTER_SOURCE_SHA256 = (
    "3e82d49512de4abe61d033c2cca903993a83587d2ea56080ff71e42c2c7372a4"
)

_SOURCE_ID_RE = re.compile(
    r"^(?P<scene>scene[0-9]{4}_[0-9]{2})/"
    r"frame_(?P<frame>[0-9]{6})/raw_(?P<raw>[0-9]{3})$"
)
_SCENE_ID_RE = re.compile(r"^scene[0-9]{4}_[0-9]{2}$")
_HEX = frozenset("0123456789abcdef")

# Fixed object-local corner order.  The oracle transforms these eight corners
# and only then computes an axis-aligned envelope.
CORNER_SIGNS = np.asarray(
    [
        [-1.0, -1.0, -1.0],
        [-1.0, -1.0, +1.0],
        [-1.0, +1.0, -1.0],
        [-1.0, +1.0, +1.0],
        [+1.0, -1.0, -1.0],
        [+1.0, -1.0, +1.0],
        [+1.0, +1.0, -1.0],
        [+1.0, +1.0, +1.0],
    ],
    dtype=np.float64,
)
CORNER_SIGNS.setflags(write=False)

POLICY: Mapping[str, object] = MappingProxyType(
    {
        "input_boxes": "sealed_fastsam_tight_box_xyxy",
        "input_image_shape": (IMAGE_HEIGHT, IMAGE_WIDTH, 3),
        "boxer_image_shape": (BOXER_IMAGE_HW, BOXER_IMAGE_HW),
        "boxer_box_convention": "xmin_xmax_ymin_ymax",
        "frame_batch": (0, MAX_SOURCES_PER_FRAME),
        "sdp_enabled": True,
        "sdp_samples": SDP_SAMPLES,
        "seed": SEED,
        "confidence_filter": False,
        "training": False,
        "online_learning": False,
        "ground_truth": False,
        "prediction_access": False,
        "evaluator_access": False,
        "history": False,
        "birth": False,
        "native_output_mutation": False,
    }
)


class F4ContractError(RuntimeError):
    """A frozen F4 model, input, identity, or output contract was violated."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(8 * 1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _valid_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in _HEX for character in value)
    )


def _readonly(
    value: object,
    dtype: np.dtype,
    shape: Optional[tuple[int, ...]] = None,
) -> np.ndarray:
    array = np.asarray(value, dtype=dtype)
    if shape is not None and array.shape != shape:
        raise ValueError(f"array must have shape {shape}, got {array.shape}")
    contiguous = np.ascontiguousarray(array)
    return np.frombuffer(contiguous.tobytes(), dtype=dtype).reshape(contiguous.shape)


def _as_numpy(value: Any, dtype: Optional[np.dtype] = None) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    # NumPy has no bfloat16 dtype.  Boxer emits its uncertainty/raw-head
    # diagnostics in the active autocast dtype, so materialize floating torch
    # tensors as float32 before crossing the NumPy boundary.  Geometry tensors
    # are already explicitly float32, and this conversion is diagnostic-only.
    if hasattr(value, "is_floating_point") and value.is_floating_point():
        value = value.float()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    result = np.asarray(value)
    if dtype is not None:
        result = result.astype(dtype, copy=False)
    return np.ascontiguousarray(result)


def _hash_array(digest: "hashlib._Hash", label: str, value: object) -> None:
    array = np.ascontiguousarray(np.asarray(value))
    digest.update(label.encode("utf-8"))
    digest.update(b"\0")
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(b"\0")
    digest.update(repr(tuple(array.shape)).encode("ascii"))
    digest.update(b"\0")
    digest.update(array.tobytes(order="C"))
    digest.update(b"\0")


def _hash_text(digest: "hashlib._Hash", label: str, value: str) -> None:
    payload = value.encode("utf-8")
    digest.update(label.encode("utf-8"))
    digest.update(b"\0")
    digest.update(len(payload).to_bytes(8, "little"))
    digest.update(payload)


def _input_digest(
    *,
    scene_id: str,
    frame_id: int,
    rgb: np.ndarray,
    depth_m: np.ndarray,
    K: np.ndarray,
    camera_to_world: np.ndarray,
    boxes_xyxy: np.ndarray,
    source_ids: Sequence[str],
) -> str:
    digest = hashlib.sha256()
    _hash_text(digest, "schema", SCHEMA)
    _hash_text(digest, "scene_id", scene_id)
    _hash_array(digest, "frame_id", np.asarray([frame_id], dtype=np.int64))
    _hash_array(digest, "rgb", rgb)
    _hash_array(digest, "depth_m", depth_m)
    _hash_array(digest, "K", K)
    _hash_array(digest, "camera_to_world", camera_to_world)
    _hash_array(digest, "boxes_xyxy", boxes_xyxy)
    for index, source_id in enumerate(source_ids):
        _hash_text(digest, f"source_id[{index}]", source_id)
    return digest.hexdigest()


def _source_id(scene_id: str, frame_id: int, value: object) -> str:
    if not isinstance(value, str):
        raise ValueError("source_ids must contain strings")
    match = _SOURCE_ID_RE.fullmatch(value)
    if match is None:
        raise ValueError(f"invalid canonical source_id: {value!r}")
    if match.group("scene") != scene_id or int(match.group("frame")) != frame_id:
        raise ValueError("source_id scene/frame differs from the current frame")
    return value


def _obb_corners(
    center_world: np.ndarray,
    local_extent: np.ndarray,
    rotation_world_object: np.ndarray,
) -> np.ndarray:
    local = CORNER_SIGNS * (local_extent.reshape(1, 3) * 0.5)
    return center_world.reshape(1, 3) + local @ rotation_world_object.T


@dataclass(frozen=True)
class HBValidity:
    """Fixed, confidence-independent HB validity decision."""

    finite_center: bool
    finite_extent: bool
    finite_rotation: bool
    positive_extent: bool
    right_handed_orthonormal: bool
    in_front: bool
    finite_corners: bool
    orthogonality_error: Optional[float]
    determinant: Optional[float]
    rotation_correction_max_abs: Optional[float]
    reasons: tuple[str, ...]

    def __post_init__(self) -> None:
        for value in (
            self.orthogonality_error,
            self.determinant,
            self.rotation_correction_max_abs,
        ):
            if value is not None and (
                not isinstance(value, (int, float, np.integer, np.floating))
                or not math.isfinite(float(value))
            ):
                raise ValueError("validity measurements must be finite numeric values or None")
        expected = []
        if not self.finite_center:
            expected.append("nonfinite_center")
        if not self.finite_extent:
            expected.append("nonfinite_extent")
        if not self.finite_rotation:
            expected.append("nonfinite_rotation")
        if self.finite_extent and not self.positive_extent:
            expected.append("nonpositive_extent")
        if self.finite_rotation and not self.right_handed_orthonormal:
            expected.append("invalid_rotation")
        if not self.in_front:
            expected.append("not_in_front")
        if not self.finite_corners:
            expected.append("nonfinite_corners")
        if self.reasons != tuple(expected):
            raise ValueError("validity reasons are not the frozen ordered decision")

    @property
    def valid(self) -> bool:
        return len(self.reasons) == 0


@dataclass(frozen=True)
class HBGeometryRow:
    """One immutable, row-bound frozen Boxer geometry hypothesis."""

    source_id: str
    row_index: int
    input_tight_box_xyxy: np.ndarray
    world_corners: Optional[np.ndarray]
    world_center: Optional[np.ndarray]
    local_extent: Optional[np.ndarray]
    world_rotation: Optional[np.ndarray]
    camera_depth: Optional[float]
    confidence: Optional[float]
    logvar: Optional[np.ndarray]
    raw_params: Optional[np.ndarray]
    valid: bool
    validity: HBValidity
    result_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.source_id, str) or _SOURCE_ID_RE.fullmatch(self.source_id) is None:
            raise ValueError("HB source_id is not canonical")
        if isinstance(self.row_index, bool) or int(self.row_index) < 0:
            raise ValueError("HB row_index must be nonnegative")
        box = _readonly(self.input_tight_box_xyxy, np.float64, (4,))
        corners = (
            None
            if self.world_corners is None
            else _readonly(self.world_corners, np.float64, (8, 3))
        )
        center = (
            None
            if self.world_center is None
            else _readonly(self.world_center, np.float64, (3,))
        )
        extent = (
            None
            if self.local_extent is None
            else _readonly(self.local_extent, np.float64, (3,))
        )
        rotation = (
            None
            if self.world_rotation is None
            else _readonly(self.world_rotation, np.float64, (3, 3))
        )
        logvar = None if self.logvar is None else _readonly(self.logvar, np.float64)
        raw_params = (
            None if self.raw_params is None else _readonly(self.raw_params, np.float64)
        )
        if bool(self.valid) != self.validity.valid:
            raise ValueError("HB valid flag differs from its diagnostic decision")
        if not _valid_sha256(self.result_sha256):
            raise ValueError("HB result_sha256 must be lowercase SHA-256")
        if self.valid:
            if any(
                value is None for value in (corners, center, extent, rotation)
            ) or self.camera_depth is None:
                raise ValueError("valid HB must expose complete finite geometry")
            assert corners is not None
            assert center is not None
            assert extent is not None
            assert rotation is not None
            expected = _obb_corners(center, extent, rotation)
            if not np.allclose(corners, expected, rtol=0.0, atol=1e-9):
                raise ValueError("valid HB corners differ from center/extent/rotation")
        object.__setattr__(self, "row_index", int(self.row_index))
        object.__setattr__(self, "input_tight_box_xyxy", box)
        object.__setattr__(self, "world_corners", corners)
        object.__setattr__(self, "world_center", center)
        object.__setattr__(self, "local_extent", extent)
        object.__setattr__(self, "world_rotation", rotation)
        object.__setattr__(self, "logvar", logvar)
        object.__setattr__(self, "raw_params", raw_params)


@dataclass(frozen=True)
class F4FrameDiagnostics:
    source_count: int
    valid_count: int
    invalid_count: int
    input_hash_ms: float
    datum_ms: float
    forward_ms: float
    conversion_ms: float
    total_ms: float
    model_load_ms: float
    asset_validation_ms: float
    cuda_synchronized: bool
    cuda_memory_allocated_before_bytes: int
    cuda_memory_allocated_after_bytes: int
    cuda_memory_reserved_after_bytes: int
    cuda_max_memory_allocated_bytes: int
    cuda_max_memory_reserved_bytes: int
    model_eval: bool
    model_parameters_frozen: bool
    model_forward_calls: int

    def __post_init__(self) -> None:
        counts = (self.source_count, self.valid_count, self.invalid_count)
        if any(isinstance(value, bool) or int(value) < 0 for value in counts):
            raise ValueError("F4 diagnostic counts must be nonnegative integers")
        if self.valid_count + self.invalid_count != self.source_count:
            raise ValueError("F4 valid/invalid counts do not partition sources")
        timings = (
            self.input_hash_ms,
            self.datum_ms,
            self.forward_ms,
            self.conversion_ms,
            self.total_ms,
            self.model_load_ms,
            self.asset_validation_ms,
        )
        if any(not math.isfinite(float(value)) or float(value) < 0.0 for value in timings):
            raise ValueError("F4 timings must be finite and nonnegative")
        memory = (
            self.cuda_memory_allocated_before_bytes,
            self.cuda_memory_allocated_after_bytes,
            self.cuda_memory_reserved_after_bytes,
            self.cuda_max_memory_allocated_bytes,
            self.cuda_max_memory_reserved_bytes,
        )
        if any(isinstance(value, bool) or int(value) < 0 for value in memory):
            raise ValueError("F4 CUDA byte counts must be nonnegative integers")
        if self.model_forward_calls not in (0, 1):
            raise ValueError("F4 performs zero or one model forward per frame")
        if self.source_count == 0 and self.model_forward_calls != 0:
            raise ValueError("empty F4 frame must not invoke Boxer")
        if self.source_count > 0 and self.model_forward_calls != 1:
            raise ValueError("non-empty F4 frame must invoke Boxer exactly once")
        if not self.model_eval or not self.model_parameters_frozen:
            raise ValueError("F4 model must remain eval-only and frozen")


@dataclass(frozen=True)
class F4BatchResult:
    """One output-inert current-frame F4 batch."""

    scene_id: str
    frame_id: int
    rows: tuple[HBGeometryRow, ...]
    diagnostics: F4FrameDiagnostics
    input_sha256: str
    result_sha256: str
    schema: str = SCHEMA
    mode: str = MODE
    protocol_id: str = PROTOCOL_ID

    def __post_init__(self) -> None:
        if self.schema != SCHEMA or self.mode != MODE or self.protocol_id != PROTOCOL_ID:
            raise ValueError("F4 result schema/mode/protocol differs")
        if _SCENE_ID_RE.fullmatch(self.scene_id) is None:
            raise ValueError("invalid F4 scene_id")
        if isinstance(self.frame_id, bool) or int(self.frame_id) < 0:
            raise ValueError("invalid F4 frame_id")
        if len(self.rows) != self.diagnostics.source_count:
            raise ValueError("F4 result rows differ from diagnostic source count")
        if tuple(row.row_index for row in self.rows) != tuple(range(len(self.rows))):
            raise ValueError("F4 rows are not in exact input order")
        if len({row.source_id for row in self.rows}) != len(self.rows):
            raise ValueError("F4 result contains duplicate source identities")
        for digest in (self.input_sha256, self.result_sha256):
            if not _valid_sha256(digest):
                raise ValueError("F4 result digests must be lowercase SHA-256")
        object.__setattr__(self, "frame_id", int(self.frame_id))


def _row_digest(
    *,
    source_id: str,
    row_index: int,
    box: np.ndarray,
    corners: np.ndarray,
    center: np.ndarray,
    extent: np.ndarray,
    rotation: np.ndarray,
    camera_depth: float,
    confidence: float,
    logvar: np.ndarray,
    raw_params: np.ndarray,
    validity: HBValidity,
) -> str:
    digest = hashlib.sha256()
    _hash_text(digest, "source_id", source_id)
    _hash_array(digest, "row_index", np.asarray([row_index], dtype=np.int64))
    for label, value in (
        ("input_tight_box_xyxy", box),
        ("world_corners", corners),
        ("world_center", center),
        ("local_extent", extent),
        ("world_rotation", rotation),
        ("camera_depth", np.asarray([camera_depth], dtype=np.float64)),
        ("confidence", np.asarray([confidence], dtype=np.float64)),
        ("logvar", logvar),
        ("raw_params", raw_params),
        (
            "validity_flags",
            np.asarray(
                [
                    validity.finite_center,
                    validity.finite_extent,
                    validity.finite_rotation,
                    validity.positive_extent,
                    validity.right_handed_orthonormal,
                    validity.in_front,
                    validity.finite_corners,
                ],
                dtype=np.uint8,
            ),
        ),
        (
            "validity_metrics",
            np.asarray(
                [
                    np.nan
                    if validity.orthogonality_error is None
                    else validity.orthogonality_error,
                    np.nan if validity.determinant is None else validity.determinant,
                    np.nan
                    if validity.rotation_correction_max_abs is None
                    else validity.rotation_correction_max_abs,
                ],
                dtype=np.float64,
            ),
        ),
    ):
        _hash_array(digest, label, value)
    for reason in validity.reasons:
        _hash_text(digest, "reason", reason)
    return digest.hexdigest()


def _batch_digest(input_sha256: str, rows: Sequence[HBGeometryRow]) -> str:
    digest = hashlib.sha256()
    _hash_text(digest, "input_sha256", input_sha256)
    for row in rows:
        _hash_text(digest, "row_result_sha256", row.result_sha256)
    return digest.hexdigest()


class FrozenFastSAMBoxerF4Provider:
    """Strict row-preserving adapter around one already-loaded Boxer model."""

    def __init__(
        self,
        adapter: Any,
        *,
        device: str,
        precision: str = "bfloat16",
        model_load_ms: float = 0.0,
        asset_validation_ms: float = 0.0,
        frozen_receipts: Optional[Mapping[str, Any]] = None,
        torch_module: Any = None,
        world_to_camera: Optional[Callable[..., Any]] = None,
        so3_projector: Optional[Callable[[Any], Any]] = None,
        rng_context_factory: Optional[Callable[..., Any]] = None,
    ) -> None:
        if precision not in ("float32", "bfloat16"):
            raise ValueError("precision must be float32 or bfloat16")
        if getattr(adapter, "model", None) is None:
            raise ValueError("F4 adapter must own an already-loaded model")
        if torch_module is None:
            import torch as torch_module  # type: ignore[no-redef]

        self._torch = torch_module
        self._adapter = adapter
        self._model = adapter.model
        self.device = "cuda" if str(device).startswith("cuda") else str(device)
        self.precision = precision
        self.model_load_ms = float(model_load_ms)
        self.asset_validation_ms = float(asset_validation_ms)
        self._world_to_camera = world_to_camera
        self._so3_projector = so3_projector
        self._rng_context_factory = rng_context_factory
        self._model_forward_count = 0
        self._feature_cache_hit_count = 0
        self._freeze_model()

        receipts = dict(frozen_receipts or {})
        # Per-process timings are deliberately excluded from the immutable
        # model identity.  They remain available through provider attributes
        # and every frame diagnostic, but including them here would make two
        # otherwise identical GPU shards produce different run signatures.
        receipts.pop("model_load_ms", None)
        receipts.pop("asset_validation_ms", None)
        receipts.setdefault("formal", False)
        receipts["model_eval"] = True
        receipts["model_parameters_frozen"] = True
        receipts["device"] = self.device
        receipts["precision"] = self.precision
        receipts["boxer_image_hw"] = int(getattr(self._model, "hw", BOXER_IMAGE_HW))
        receipts["sdp_enabled"] = True
        receipts["sdp_samples"] = SDP_SAMPLES
        receipts["seed"] = SEED
        self._frozen_receipts = _freeze_mapping(receipts)

    @property
    def frozen_receipts(self) -> Mapping[str, Any]:
        return self._frozen_receipts

    @property
    def model_forward_count(self) -> int:
        return self._model_forward_count

    def _freeze_model(self) -> None:
        if hasattr(self._model, "eval"):
            self._model.eval()
        if not hasattr(self._model, "parameters"):
            raise F4ContractError("Boxer model parameters are not inspectable")
        for parameter in self._model.parameters():
            parameter.requires_grad_(False)
        self._assert_model_frozen()

    def _assert_model_frozen(self) -> None:
        if bool(getattr(self._model, "training", True)):
            raise F4ContractError("Boxer model left eval mode")
        if any(bool(parameter.requires_grad) for parameter in self._model.parameters()):
            raise F4ContractError("Boxer model contains trainable parameters")

    def _cuda_sync(self) -> None:
        if self.device == "cuda":
            self._torch.cuda.synchronize()

    def _cuda_bytes(self, name: str) -> int:
        if self.device != "cuda":
            return 0
        value = int(getattr(self._torch.cuda, name)())
        if value < 0:
            raise F4ContractError(f"CUDA {name} returned a negative byte count")
        return value

    def _convert_world_to_camera(
        self,
        centers: Any,
        extents: Any,
        rotations: Any,
        camera_to_world: np.ndarray,
    ) -> tuple[Any, Any]:
        torch = self._torch
        pose = torch.from_numpy(camera_to_world).float()
        if self._world_to_camera is not None:
            return self._world_to_camera(centers, extents, rotations, pose)
        rotation_world_camera = pose[:3, :3]
        translation_world_camera = pose[:3, 3]
        rotation_camera_world = rotation_world_camera.transpose(0, 1)
        center_camera = torch.einsum(
            "ij,nj->ni", rotation_camera_world, centers - translation_world_camera[None]
        )
        rotation_camera_object = torch.einsum(
            "ij,njk->nik", rotation_camera_world, rotations
        )
        return torch.cat((center_camera, extents), dim=-1), rotation_camera_object

    def _project_so3(self, rotations: Any) -> Any:
        if self._so3_projector is not None:
            return self._so3_projector(rotations)
        torch = self._torch
        rotations = rotations.float()
        finite = torch.isfinite(rotations).all(dim=(1, 2))
        result = rotations.clone()
        if torch.any(finite):
            u, _, vh = torch.linalg.svd(rotations[finite])
            candidate = torch.matmul(u, vh)
            signs = torch.ones(
                (candidate.shape[0], 3), dtype=rotations.dtype, device=rotations.device
            )
            signs[:, -1] = torch.where(
                torch.linalg.det(candidate) < 0.0,
                -1.0,
                1.0,
            )
            result[finite] = torch.matmul(torch.matmul(u, torch.diag_embed(signs)), vh)
        return result

    def infer_batch(
        self,
        scene_id: str,
        frame_id: int,
        rgb: object,
        depth_m: object,
        K: object,
        camera_to_world: object,
        boxes_xyxy: object,
        source_ids: Sequence[str],
    ) -> F4BatchResult:
        """Infer one HB row for every sealed current-frame source.

        The positional signature is intentional: runners may use this method
        through a small duck-typed fake provider without importing Boxer.
        """

        if _SCENE_ID_RE.fullmatch(str(scene_id)) is None:
            raise ValueError("scene_id must match sceneNNNN_NN")
        scene_id = str(scene_id)
        if isinstance(frame_id, (bool, np.bool_)) or not isinstance(
            frame_id, (int, np.integer)
        ):
            raise ValueError("frame_id must be an integer")
        frame_id = int(frame_id)
        if frame_id < 0 or frame_id > 999_999:
            raise ValueError("frame_id is outside the canonical six-digit range")

        rgb_array = np.asarray(rgb)
        if rgb_array.shape != (IMAGE_HEIGHT, IMAGE_WIDTH, 3) or rgb_array.dtype != np.uint8:
            raise ValueError("rgb must be uint8 RGB with shape [480,640,3]")
        depth_array = np.asarray(depth_m)
        if depth_array.shape != (IMAGE_HEIGHT, IMAGE_WIDTH) or not np.issubdtype(
            depth_array.dtype, np.floating
        ):
            raise ValueError("depth_m must be a floating [480,640] metric map")
        depth_array = np.ascontiguousarray(depth_array, dtype=np.float32)
        if not np.isfinite(depth_array).all() or np.any(depth_array < 0.0):
            raise ValueError("depth_m must be finite and nonnegative")
        K_array = np.ascontiguousarray(np.asarray(K, dtype=np.float32))
        if K_array.shape != (3, 3) or not np.isfinite(K_array).all():
            raise ValueError("K must be a finite 3x3 matrix")
        if K_array[0, 0] <= 0.0 or K_array[1, 1] <= 0.0:
            raise ValueError("K focal lengths must be positive")
        pose_array = np.ascontiguousarray(np.asarray(camera_to_world, dtype=np.float32))
        if pose_array.shape != (4, 4) or not np.isfinite(pose_array).all():
            raise ValueError("camera_to_world must be a finite 4x4 matrix")
        if not np.allclose(pose_array[3], [0.0, 0.0, 0.0, 1.0], atol=1e-6, rtol=0.0):
            raise ValueError("camera_to_world has an invalid homogeneous row")
        rotation_world_camera = pose_array[:3, :3]
        if not np.allclose(
            rotation_world_camera.T @ rotation_world_camera,
            np.eye(3),
            atol=5e-3,
            rtol=0.0,
        ) or not np.isclose(np.linalg.det(rotation_world_camera), 1.0, atol=5e-3):
            raise ValueError("camera_to_world rotation is not right-handed orthonormal")

        boxes = np.ascontiguousarray(np.asarray(boxes_xyxy, dtype=np.float32))
        if boxes.ndim != 2 or boxes.shape[1:] != (4,):
            raise ValueError("boxes_xyxy must have shape [N,4]")
        count = int(boxes.shape[0])
        if count > MAX_SOURCES_PER_FRAME:
            raise ValueError("F4 frame exceeds the sealed sixteen-source cap")
        if len(source_ids) != count:
            raise ValueError("source_ids count differs from boxes_xyxy")
        source_ids_tuple = tuple(
            _source_id(scene_id, frame_id, value) for value in source_ids
        )
        if len(set(source_ids_tuple)) != count:
            raise ValueError("source_ids must be unique within a frame")
        if not np.isfinite(boxes).all():
            raise ValueError("boxes_xyxy must be finite")
        if count:
            if (
                np.any(boxes[:, 0] < 0.0)
                or np.any(boxes[:, 1] < 0.0)
                or np.any(boxes[:, 2] >= IMAGE_WIDTH)
                or np.any(boxes[:, 3] >= IMAGE_HEIGHT)
                or np.any(boxes[:, 2] <= boxes[:, 0])
                or np.any(boxes[:, 3] <= boxes[:, 1])
            ):
                raise ValueError("boxes_xyxy violate the sealed 640x480 tight-box bounds")

        rgb_copy = np.ascontiguousarray(rgb_array).copy()
        depth_copy = depth_array.copy()
        K_copy = K_array.copy()
        pose_copy = pose_array.copy()
        boxes_copy = boxes.copy()
        hash_started = time.perf_counter()
        input_sha256 = _input_digest(
            scene_id=scene_id,
            frame_id=frame_id,
            rgb=rgb_copy,
            depth_m=depth_copy,
            K=K_copy,
            camera_to_world=pose_copy,
            boxes_xyxy=boxes_copy,
            source_ids=source_ids_tuple,
        )
        input_hash_ms = (time.perf_counter() - hash_started) * 1000.0
        self._assert_model_frozen()
        allocated_before = self._cuda_bytes("memory_allocated")

        if count == 0:
            diagnostics = F4FrameDiagnostics(
                source_count=0,
                valid_count=0,
                invalid_count=0,
                input_hash_ms=input_hash_ms,
                datum_ms=0.0,
                forward_ms=0.0,
                conversion_ms=0.0,
                total_ms=0.0,
                model_load_ms=self.model_load_ms,
                asset_validation_ms=self.asset_validation_ms,
                cuda_synchronized=self.device == "cuda",
                cuda_memory_allocated_before_bytes=allocated_before,
                cuda_memory_allocated_after_bytes=self._cuda_bytes("memory_allocated"),
                cuda_memory_reserved_after_bytes=self._cuda_bytes("memory_reserved"),
                cuda_max_memory_allocated_bytes=self._cuda_bytes("max_memory_allocated"),
                cuda_max_memory_reserved_bytes=self._cuda_bytes("max_memory_reserved"),
                model_eval=True,
                model_parameters_frozen=True,
                model_forward_calls=0,
            )
            return F4BatchResult(
                scene_id=scene_id,
                frame_id=frame_id,
                rows=(),
                diagnostics=diagnostics,
                input_sha256=input_sha256,
                result_sha256=_batch_digest(input_sha256, ()),
            )

        online_started = time.perf_counter()
        datum_started = time.perf_counter()
        torch_boxes = self._torch.from_numpy(boxes_copy.copy()).float()
        datum, metadata = self._adapter._make_datum(
            image=rgb_copy,
            depth=depth_copy,
            boxes_xyxy=torch_boxes,
            image_K=K_copy,
            depth_K=K_copy,
            camera_to_world=pose_copy,
            scene_id=scene_id,
            frame_id=frame_id,
        )
        datum_ms = (time.perf_counter() - datum_started) * 1000.0
        expected_xxyy = np.stack(
            (
                boxes_copy[:, 0] * (BOXER_IMAGE_HW / IMAGE_WIDTH),
                boxes_copy[:, 2] * (BOXER_IMAGE_HW / IMAGE_WIDTH),
                boxes_copy[:, 1] * (BOXER_IMAGE_HW / IMAGE_HEIGHT),
                boxes_copy[:, 3] * (BOXER_IMAGE_HW / IMAGE_HEIGHT),
            ),
            axis=-1,
        )
        observed_xxyy = _as_numpy(metadata.get("boxer_boxes"), np.float32)
        if observed_xxyy.shape != (count, 4) or not np.array_equal(
            observed_xxyy, expected_xxyy.astype(np.float32)
        ):
            raise F4ContractError("adapter changed the frozen tight-box mapping")

        self._assert_model_frozen()
        forward_started = time.perf_counter()
        cache_enabled = bool(
            getattr(
                getattr(self._adapter, "config", None),
                "cache_image_features",
                False,
            )
        )
        if cache_enabled:
            output, _, cache_hit = self._adapter.forward_raw_with_feature_cache(
                datum,
                scene_id=scene_id,
                frame_id=frame_id,
                encoder_input_sha256=metadata["encoder_input_sha256"],
            )
            self._feature_cache_hit_count += int(cache_hit)
        else:
            rng_context = (
                self._rng_context_factory(include_cuda=self.device == "cuda")
                if self._rng_context_factory is not None
                else nullcontext()
            )
            self._cuda_sync()
            if self.device == "mps" or self.precision == "float32":
                precision_context = nullcontext()
            else:
                precision_context = self._torch.autocast(
                    device_type=self.device, dtype=self._torch.bfloat16
                )
            with rng_context, self._torch.inference_mode(), precision_context:
                output = self._model.forward(datum)
            self._cuda_sync()
        forward_ms = (time.perf_counter() - forward_started) * 1000.0
        self._model_forward_count += 1
        self._assert_model_frozen()

        conversion_started = time.perf_counter()
        if not isinstance(output, Mapping) or "obbs_pr_w" not in output:
            raise F4ContractError("Boxer output lacks obbs_pr_w")
        try:
            obbs_world = output["obbs_pr_w"][0].cpu()
            output_count = len(obbs_world)
            centers_t = obbs_world.bb3_center_world.float()
            extents_t = obbs_world.bb3_diagonal.float()
            rotations_raw_t = obbs_world.T_world_object.R.float()
            confidence_t = obbs_world.prob.squeeze(-1).float()
        except Exception as error:
            raise F4ContractError("Boxer returned malformed world OBB rows") from error
        if output_count != count:
            raise F4ContractError(
                f"Boxer output row count differs: {output_count} != {count}"
            )
        if tuple(centers_t.shape) != (count, 3) or tuple(extents_t.shape) != (count, 3):
            raise F4ContractError("Boxer center/extent shapes differ from [N,3]")
        if tuple(rotations_raw_t.shape) != (count, 3, 3):
            raise F4ContractError("Boxer rotation shape differs from [N,3,3]")
        if tuple(confidence_t.shape) != (count,):
            raise F4ContractError("Boxer confidence shape differs from [N]")
        for key in ("obbs_pr_logvar", "obbs_pr_params"):
            if key not in output:
                raise F4ContractError(f"Boxer output lacks {key}")
        logvar = _as_numpy(output["obbs_pr_logvar"][0], np.float64)
        raw_params = _as_numpy(output["obbs_pr_params"][0], np.float64)
        if logvar.ndim == 0 or logvar.shape[0] != count:
            raise F4ContractError("Boxer logvar rows differ from input")
        if raw_params.ndim == 0 or raw_params.shape[0] != count:
            raise F4ContractError("Boxer raw parameter rows differ from input")

        xyz_dims_camera_t, rotations_camera_raw_t = self._convert_world_to_camera(
            centers_t, extents_t, rotations_raw_t, pose_copy
        )
        rotations_camera_t = self._project_so3(rotations_camera_raw_t)
        pose_rotation_t = self._torch.from_numpy(pose_copy[:3, :3]).float()
        rotations_world_t = self._torch.einsum(
            "ij,njk->nik", pose_rotation_t, rotations_camera_t
        )

        centers = _as_numpy(centers_t, np.float64)
        extents = _as_numpy(extents_t, np.float64)
        rotations_raw = _as_numpy(rotations_raw_t, np.float64)
        rotations_camera_raw = _as_numpy(rotations_camera_raw_t, np.float64)
        rotations_camera = _as_numpy(rotations_camera_t, np.float64)
        rotations_world = _as_numpy(rotations_world_t, np.float64)
        xyz_dims_camera = _as_numpy(xyz_dims_camera_t, np.float64)
        confidence = _as_numpy(confidence_t, np.float64)

        rows = []
        for index, source_id_value in enumerate(source_ids_tuple):
            center = centers[index]
            extent = extents[index]
            raw_rotation = rotations_raw[index]
            rotation = rotations_world[index]
            camera_rotation = rotations_camera[index]
            corners = _obb_corners(center, extent, rotation)
            finite_center = bool(np.isfinite(center).all())
            finite_extent = bool(np.isfinite(extent).all())
            finite_rotation = bool(
                np.isfinite(raw_rotation).all()
                and np.isfinite(camera_rotation).all()
                and np.isfinite(rotation).all()
            )
            positive_extent = bool(finite_extent and np.all(extent > 0.0))
            if finite_rotation:
                orthogonality_error = float(
                    np.max(np.abs(camera_rotation.T @ camera_rotation - np.eye(3)))
                )
                determinant = float(np.linalg.det(camera_rotation))
                right_handed = bool(
                    math.isfinite(orthogonality_error)
                    and math.isfinite(determinant)
                    and orthogonality_error <= ROTATION_TOLERANCE
                    and abs(determinant - 1.0) <= ROTATION_TOLERANCE
                )
                correction = float(
                    np.max(np.abs(camera_rotation - rotations_camera_raw[index]))
                )
            else:
                orthogonality_error = None
                determinant = None
                correction = None
                right_handed = False
            camera_depth = float(xyz_dims_camera[index, 2])
            in_front = bool(math.isfinite(camera_depth) and camera_depth > CAMERA_DEPTH_EPSILON_M)
            finite_corners = bool(np.isfinite(corners).all())
            reasons = []
            if not finite_center:
                reasons.append("nonfinite_center")
            if not finite_extent:
                reasons.append("nonfinite_extent")
            if not finite_rotation:
                reasons.append("nonfinite_rotation")
            if finite_extent and not positive_extent:
                reasons.append("nonpositive_extent")
            if finite_rotation and not right_handed:
                reasons.append("invalid_rotation")
            if not in_front:
                reasons.append("not_in_front")
            if not finite_corners:
                reasons.append("nonfinite_corners")
            validity = HBValidity(
                finite_center=finite_center,
                finite_extent=finite_extent,
                finite_rotation=finite_rotation,
                positive_extent=positive_extent,
                right_handed_orthonormal=right_handed,
                in_front=in_front,
                finite_corners=finite_corners,
                orthogonality_error=orthogonality_error,
                determinant=determinant,
                rotation_correction_max_abs=correction,
                reasons=tuple(reasons),
            )
            row_hash = _row_digest(
                source_id=source_id_value,
                row_index=index,
                box=boxes_copy[index].astype(np.float64),
                corners=corners,
                center=center,
                extent=extent,
                rotation=rotation,
                camera_depth=camera_depth,
                confidence=float(confidence[index]),
                logvar=np.asarray(logvar[index], dtype=np.float64),
                raw_params=np.asarray(raw_params[index], dtype=np.float64),
                validity=validity,
            )
            rows.append(
                HBGeometryRow(
                    source_id=source_id_value,
                    row_index=index,
                    input_tight_box_xyxy=boxes_copy[index],
                    world_corners=corners if finite_corners else None,
                    world_center=center if finite_center else None,
                    local_extent=extent if finite_extent else None,
                    world_rotation=rotation if finite_rotation else None,
                    camera_depth=camera_depth if math.isfinite(camera_depth) else None,
                    confidence=(
                        float(confidence[index])
                        if math.isfinite(float(confidence[index]))
                        else None
                    ),
                    logvar=(
                        logvar[index] if np.isfinite(logvar[index]).all() else None
                    ),
                    raw_params=(
                        raw_params[index]
                        if np.isfinite(raw_params[index]).all()
                        else None
                    ),
                    valid=validity.valid,
                    validity=validity,
                    result_sha256=row_hash,
                )
            )
        conversion_ms = (time.perf_counter() - conversion_started) * 1000.0
        total_ms = (time.perf_counter() - online_started) * 1000.0

        # Adapter/model code receives detached copies.  This second digest is a
        # fail-closed guard against accidental in-place changes to the inputs
        # that define this result.
        after_digest = _input_digest(
            scene_id=scene_id,
            frame_id=frame_id,
            rgb=rgb_copy,
            depth_m=depth_copy,
            K=K_copy,
            camera_to_world=pose_copy,
            boxes_xyxy=boxes_copy,
            source_ids=source_ids_tuple,
        )
        if after_digest != input_sha256:
            raise F4ContractError("F4 adapter mutated a frozen current-frame input")

        valid_count = sum(row.valid for row in rows)
        diagnostics = F4FrameDiagnostics(
            source_count=count,
            valid_count=valid_count,
            invalid_count=count - valid_count,
            input_hash_ms=input_hash_ms,
            datum_ms=datum_ms,
            forward_ms=forward_ms,
            conversion_ms=conversion_ms,
            total_ms=total_ms,
            model_load_ms=self.model_load_ms,
            asset_validation_ms=self.asset_validation_ms,
            cuda_synchronized=self.device == "cuda",
            cuda_memory_allocated_before_bytes=allocated_before,
            cuda_memory_allocated_after_bytes=self._cuda_bytes("memory_allocated"),
            cuda_memory_reserved_after_bytes=self._cuda_bytes("memory_reserved"),
            cuda_max_memory_allocated_bytes=self._cuda_bytes("max_memory_allocated"),
            cuda_max_memory_reserved_bytes=self._cuda_bytes("max_memory_reserved"),
            model_eval=True,
            model_parameters_frozen=True,
            model_forward_calls=1,
        )
        rows_tuple = tuple(rows)
        return F4BatchResult(
            scene_id=scene_id,
            frame_id=frame_id,
            rows=rows_tuple,
            diagnostics=diagnostics,
            input_sha256=input_sha256,
            result_sha256=_batch_digest(input_sha256, rows_tuple),
        )


def _freeze_mapping(value: Mapping[str, Any]) -> Mapping[str, Any]:
    frozen = {}
    for key, item in value.items():
        if isinstance(item, Mapping):
            frozen[str(key)] = _freeze_mapping(item)
        elif isinstance(item, list):
            frozen[str(key)] = tuple(item)
        else:
            frozen[str(key)] = item
    return MappingProxyType(frozen)


def _load_frozen_adapter_module(path: Path) -> Any:
    name = "_boxfusion_f4_frozen_boxer_lifter"
    existing = sys.modules.get(name)
    if existing is not None:
        if Path(existing.__file__).resolve() != path.resolve():
            raise F4ContractError("frozen Boxer adapter module path changed")
        return existing
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise F4ContractError("cannot construct frozen Boxer adapter import")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(name, None)
        raise
    return module


def _git(boxer_root: Path, *arguments: str) -> str:
    try:
        completed = subprocess.run(
            ["git", "-C", os.fspath(boxer_root), *arguments],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise F4ContractError(f"cannot audit frozen Boxer repository: {error}") from error
    return completed.stdout.strip()


def create_frozen_boxer_provider(
    device: str = "cuda",
    *,
    boxer_root: Path | str = DEFAULT_BOXER_ROOT,
    checkpoint: Path | str = DEFAULT_BOXER_CHECKPOINT,
    adapter_path: Path | str = DEFAULT_ADAPTER_PATH,
) -> FrozenFastSAMBoxerF4Provider:
    """Verify, load and freeze the exact released BoxerNet used by F4."""

    boxer_root = Path(boxer_root).resolve(strict=True)
    checkpoint = Path(checkpoint).resolve(strict=True)
    adapter_path = Path(adapter_path).resolve(strict=True)
    dino_path = (boxer_root / "ckpts" / DEFAULT_DINOV3_CHECKPOINT.name).resolve(strict=True)
    boxernet_source = (boxer_root / "boxernet/boxernet.py").resolve(strict=True)

    validation_started = time.perf_counter()
    observed = {
        "adapter": _sha256_file(adapter_path),
        "boxernet_source": _sha256_file(boxernet_source),
        "boxernet_checkpoint": _sha256_file(checkpoint),
        "dinov3_checkpoint": _sha256_file(dino_path),
    }
    expected = {
        "adapter": ADAPTER_SOURCE_SHA256,
        "boxernet_source": BOXERNET_SOURCE_SHA256,
        "boxernet_checkpoint": BOXER_CHECKPOINT_SHA256,
        "dinov3_checkpoint": DINOV3_CHECKPOINT_SHA256,
    }
    if observed != expected:
        raise F4ContractError(
            "frozen Boxer asset SHA-256 differs: "
            + json.dumps({key: {"expected": expected[key], "observed": observed[key]} for key in expected if expected[key] != observed[key]}, sort_keys=True)
        )
    commit = _git(boxer_root, "rev-parse", "HEAD")
    if commit != BOXER_REPOSITORY_COMMIT:
        raise F4ContractError(f"Boxer repository commit differs: {commit}")
    status = _git(boxer_root, "status", "--porcelain")
    if status:
        raise F4ContractError("Boxer repository worktree is not clean")
    asset_validation_ms = (time.perf_counter() - validation_started) * 1000.0

    adapter_module = _load_frozen_adapter_module(adapter_path)
    mapping = {
        "mode": "observer",
        "apply_stage": "post_filter",
        "official_root": os.fspath(boxer_root),
        "checkpoint": os.fspath(checkpoint),
        "expected_commit": BOXER_REPOSITORY_COMMIT,
        "checkpoint_sha256": BOXER_CHECKPOINT_SHA256,
        "dinov3_sha256": DINOV3_CHECKPOINT_SHA256,
        "precision": "bfloat16",
        "use_sdp": True,
        "sdp_samples": SDP_SAMPLES,
        "seed": SEED,
        # F4 never calls the adapter diagnostic writer.
        "diagnostics_dir": os.devnull,
        "selective_gate": {"enabled": False},
    }
    config = adapter_module.BoxerLiftingConfig.from_mapping(
        mapping, code_root=os.fspath(REPOSITORY_ROOT)
    )
    adapter = adapter_module.BoxerLiftingAdapter(config, device=device)
    load_started = time.perf_counter()
    adapter._load_model()
    model_load_ms = (time.perf_counter() - load_started) * 1000.0
    model = adapter.model
    if int(getattr(model, "hw", -1)) != BOXER_IMAGE_HW:
        raise F4ContractError("Boxer checkpoint image size differs from 960")
    parameter_count = sum(int(parameter.numel()) for parameter in model.parameters())
    receipts = {
        "formal": str(device).startswith("cuda"),
        "boxer_repository": {
            "path": os.fspath(boxer_root),
            "commit": commit,
            "clean": True,
        },
        "boxernet_checkpoint": {
            "path": os.fspath(checkpoint),
            "sha256": observed["boxernet_checkpoint"],
        },
        "dinov3_checkpoint": {
            "path": os.fspath(dino_path),
            "sha256": observed["dinov3_checkpoint"],
        },
        "boxernet_source": {
            "path": os.fspath(boxernet_source),
            "sha256": observed["boxernet_source"],
        },
        "adapter_source": {
            "path": os.fspath(adapter_path),
            "sha256": observed["adapter"],
        },
        "parameter_count": parameter_count,
    }
    return FrozenFastSAMBoxerF4Provider(
        adapter,
        device=device,
        precision="bfloat16",
        model_load_ms=model_load_ms,
        asset_validation_ms=asset_validation_ms,
        frozen_receipts=receipts,
        world_to_camera=adapter_module.boxer_world_to_boxfusion_camera,
        so3_projector=adapter_module.project_rotations_to_so3,
        rng_context_factory=adapter_module._preserve_rng_state,
    )


def run_scene0568_three_row_smoke(device: str = "cuda") -> Mapping[str, Any]:
    """Run the frozen no-GT three-row smoke used to validate the F4 route."""

    import cv2

    sidecar_path = (
        REPOSITORY_ROOT
        / "logs/scannet_fastsam_f0_full200_score05/scenes/scene0568_00.json"
    )
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    frame = next(
        row
        for row in sidecar["frames"]
        if row.get("successful")
        and row.get("funnel")
        and len(row["funnel"]["candidates"]) >= 3
    )
    inputs = frame["inputs"]
    paths_and_hashes = (
        (Path(inputs["rgb_path"]), inputs["rgb_sha256"]),
        (Path(inputs["depth_path"]), inputs["depth_sha256"]),
        (Path(inputs["pose_path"]), inputs["pose_sha256"]),
        (Path(sidecar["intrinsic"]["path"]), sidecar["intrinsic"]["sha256"]),
    )
    for path, expected_hash in paths_and_hashes:
        if _sha256_file(path) != expected_hash:
            raise F4ContractError(f"smoke input changed: {path}")
    bgr = cv2.imread(inputs["rgb_path"], cv2.IMREAD_COLOR)
    if bgr is None:
        raise F4ContractError("could not decode smoke RGB")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    if rgb.shape[:2] != (IMAGE_HEIGHT, IMAGE_WIDTH):
        rgb = cv2.resize(rgb, (IMAGE_WIDTH, IMAGE_HEIGHT), interpolation=cv2.INTER_LINEAR)
    depth_raw = cv2.imread(inputs["depth_path"], cv2.IMREAD_UNCHANGED)
    if depth_raw is None:
        raise F4ContractError("could not decode smoke depth")
    if depth_raw.shape != (IMAGE_HEIGHT, IMAGE_WIDTH):
        depth_raw = cv2.resize(
            depth_raw, (IMAGE_WIDTH, IMAGE_HEIGHT), interpolation=cv2.INTER_NEAREST
        )
    depth_m = depth_raw.astype(np.float32) / 1000.0
    K = np.loadtxt(sidecar["intrinsic"]["path"], dtype=np.float32)[:3, :3]
    pose = np.loadtxt(inputs["pose_path"], dtype=np.float32)
    candidates = frame["funnel"]["candidates"][:3]
    boxes = np.asarray([row["tight_box_xyxy"] for row in candidates], dtype=np.float32)
    frame_id = int(frame["frame_id"])
    source_ids = tuple(
        f"{sidecar['scene_id']}/frame_{frame_id:06d}/raw_{int(row['raw_index']):03d}"
        for row in candidates
    )
    provider = create_frozen_boxer_provider(device=device)
    result = provider.infer_batch(
        sidecar["scene_id"], frame_id, rgb, depth_m, K, pose, boxes, source_ids
    )
    return MappingProxyType(
        {
            "scene_id": result.scene_id,
            "frame_id": result.frame_id,
            "source_ids": tuple(row.source_id for row in result.rows),
            "valid": tuple(row.valid for row in result.rows),
            "confidence": tuple(row.confidence for row in result.rows),
            "world_center": tuple(tuple(row.world_center.tolist()) for row in result.rows),
            "local_extent": tuple(tuple(row.local_extent.tolist()) for row in result.rows),
            "model_load_ms": result.diagnostics.model_load_ms,
            "datum_ms": result.diagnostics.datum_ms,
            "forward_ms": result.diagnostics.forward_ms,
            "conversion_ms": result.diagnostics.conversion_ms,
            "total_ms": result.diagnostics.total_ms,
            "cuda_max_memory_allocated_bytes": result.diagnostics.cuda_max_memory_allocated_bytes,
        }
    )


def _main() -> int:
    parser = argparse.ArgumentParser(description="Frozen F4 Boxer geometry shadow core")
    parser.add_argument("--smoke-scene0568-three-rows", action="store_true")
    parser.add_argument("--device", default="cuda")
    arguments = parser.parse_args()
    if not arguments.smoke_scene0568_three_rows:
        parser.error("only the explicit no-GT three-row smoke is supported")
    value = run_scene0568_three_row_smoke(device=arguments.device)
    print(json.dumps(dict(value), sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())


__all__ = [
    "ADAPTER_SOURCE_SHA256",
    "BOXERNET_SOURCE_SHA256",
    "BOXER_CHECKPOINT_SHA256",
    "BOXER_REPOSITORY_COMMIT",
    "CORNER_SIGNS",
    "DINOV3_CHECKPOINT_SHA256",
    "F4BatchResult",
    "F4ContractError",
    "F4FrameDiagnostics",
    "FrozenFastSAMBoxerF4Provider",
    "HBGeometryRow",
    "HBValidity",
    "MODE",
    "POLICY",
    "PROTOCOL_ID",
    "SCHEMA",
    "create_frozen_boxer_provider",
    "run_scene0568_three_row_smoke",
]
