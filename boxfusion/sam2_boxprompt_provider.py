"""Frozen SAM2.1 image box-prompt provider for N0 shadow experiments.

The provider deliberately exposes only current-frame, semantic-free image
segmentation.  One RGB frame is embedded exactly once and between one and
sixteen source boxes are decoded in one call.  For every source box, the
multimask hypothesis with the largest frozen predicted-IoU value is selected;
``numpy.argmax`` makes an exact tie choose the lowest hypothesis index.

The SAM2 imports, asset verification, checkpoint load, and CUDA allocation are
all lazy.  An empty box batch returns an explicit empty result without loading
SAM2.  The image predictor is reset after every non-empty call, including
failed calls, so no image embedding or prompt state crosses frame boundaries.
This module never accepts source scores, classes, IDs, history, ground truth,
or native predictions and therefore cannot filter or reorder source boxes.
"""

from __future__ import annotations

from contextlib import ExitStack, nullcontext
from dataclasses import dataclass
import hashlib
import importlib
import os
from pathlib import Path
import sys
import threading
import time
from typing import Any, Callable

import numpy as np


SCHEMA = "boxfusion.sam2_boxprompt_provider.n0.v1"
IMAGE_HEIGHT = 480
IMAGE_WIDTH = 640
MAX_BOXES_PER_FRAME = 16
MULTIMASK_HYPOTHESES = 3

EXPECTED_SAM2_SOURCE_FILE_COUNT = 23
EXPECTED_SAM2_SOURCE_TREE_SHA256 = (
    "cc5a594bab1508ab69cbedfbb83ba8e226f848dd142a3deba8c195ee1e2469cf"
)
EXPECTED_SAM2_CONFIG_SHA256 = (
    "545e4325aa5c19a1615d43c946b07276ed4c57214eacf1437e38fa3d9374f636"
)
EXPECTED_SAM2_CHECKPOINT_BYTES = 898_083_611
EXPECTED_SAM2_CHECKPOINT_SHA256 = (
    "2647878d5dfa5098f2f8649825738a9345572bae2d4350a2468587ece47dd318"
)
SDPA_COMPATIBILITY_POLICY_ID = (
    "N0A-TORCH251-SDPA-KERNEL-COMPAT-OLD-FLAGS-EXACT-V1"
)
EXPECTED_TORCH_VERSION = "2.5.1+cu121"
EXPECTED_SAM2_TRANSFORMER_RELATIVE_PATH = "sam2/modeling/sam/transformer.py"
EXPECTED_SAM2_TRANSFORMER_SHA256 = (
    "17aac13abc8f73023f6be4b78af708df9f9f254964729421b1ba60e72a9011c1"
)
EXPECTED_TORCH_ATTENTION_SHA256 = (
    "32f2d016ba9292c182ef4e3ffa1c8b4143d16e99f9d01ffc4999366a5a342374"
)
_SDPA_PATCH_MARKER = "__boxfusion_n0a_sdpa_compatibility_policy_id__"


class SAM2BoxPromptError(RuntimeError):
    """A frozen asset, SAM2 runtime, or decoder output violated the contract."""


@dataclass(frozen=True)
class SAM2BoxPromptProductionConfig:
    """Exact local SAM2.1 Hiera-L production identity and inference policy."""

    source_root: Path = Path(
        "/data/ZhaoX/OVM3D-Dett/third_party/Grounded-SAM-2"
    )
    config_name: str = "configs/sam2.1/sam2.1_hiera_l.yaml"
    checkpoint_path: Path = Path(
        "/data/ZhaoX/OVM3D-Dett/third_party/Grounded-SAM-2/"
        "checkpoints/sam2.1_hiera_large.pt"
    )
    source_file_glob: str = "sam2/**/*.py"
    source_file_count: int = EXPECTED_SAM2_SOURCE_FILE_COUNT
    source_tree_sha256: str = EXPECTED_SAM2_SOURCE_TREE_SHA256
    config_sha256: str = EXPECTED_SAM2_CONFIG_SHA256
    checkpoint_bytes: int = EXPECTED_SAM2_CHECKPOINT_BYTES
    checkpoint_sha256: str = EXPECTED_SAM2_CHECKPOINT_SHA256
    device: str = "cuda"
    apply_postprocessing: bool = True
    autocast_dtype: str = "bfloat16"
    multimask_output: bool = True
    return_logits: bool = False
    normalize_coords: bool = True
    mask_threshold: float = 0.0
    max_boxes_per_frame: int = MAX_BOXES_PER_FRAME
    multimask_hypotheses: int = MULTIMASK_HYPOTHESES
    sdpa_compatibility_policy_id: str = SDPA_COMPATIBILITY_POLICY_ID
    torch_version: str = EXPECTED_TORCH_VERSION
    transformer_relative_path: str = EXPECTED_SAM2_TRANSFORMER_RELATIVE_PATH
    transformer_sha256: str = EXPECTED_SAM2_TRANSFORMER_SHA256
    torch_attention_sha256: str = EXPECTED_TORCH_ATTENTION_SHA256


PRODUCTION_CONFIG = SAM2BoxPromptProductionConfig()


@dataclass(frozen=True)
class SAM2BoxPromptTiming:
    """Per-frame synchronized latency and peak allocator observation."""

    encoder_ms: float
    decoder_and_host_mask_ms: float
    complete_ms: float
    cuda_synchronized: bool
    peak_allocated_memory_bytes: int

    def __post_init__(self) -> None:
        phases = (
            float(self.encoder_ms),
            float(self.decoder_and_host_mask_ms),
            float(self.complete_ms),
        )
        if not np.isfinite(phases).all() or any(value < 0.0 for value in phases):
            raise ValueError("SAM2 timing values must be finite and non-negative")
        phase_sum = phases[0] + phases[1]
        tolerance = max(1e-6, abs(phases[2]) * 1e-9)
        if abs(phase_sum - phases[2]) > tolerance:
            raise ValueError("SAM2 complete_ms must equal encoder plus decoder/host time")
        if not isinstance(self.cuda_synchronized, bool):
            raise ValueError("cuda_synchronized must be a bool")
        if (
            not isinstance(self.peak_allocated_memory_bytes, int)
            or isinstance(self.peak_allocated_memory_bytes, bool)
            or self.peak_allocated_memory_bytes < 0
        ):
            raise ValueError("peak_allocated_memory_bytes must be a non-negative int")
        if not self.cuda_synchronized and self.peak_allocated_memory_bytes != 0:
            raise ValueError("an unsynchronized timing cannot report CUDA peak memory")


EMPTY_TIMING = SAM2BoxPromptTiming(
    encoder_ms=0.0,
    decoder_and_host_mask_ms=0.0,
    complete_ms=0.0,
    cuda_synchronized=False,
    peak_allocated_memory_bytes=0,
)


@dataclass(frozen=True)
class SAM2BoxPromptResult:
    """Immutable, source-order-preserving result for one current RGB frame."""

    masks: np.ndarray
    selected_hypothesis_indices: np.ndarray
    predicted_ious: np.ndarray
    all_predicted_ious: np.ndarray
    timing: SAM2BoxPromptTiming

    def __post_init__(self) -> None:
        masks = _readonly_array(self.masks, np.bool_)
        selected = _readonly_array(self.selected_hypothesis_indices, np.int64)
        ious = _readonly_array(self.predicted_ious, np.float32)
        all_ious = _readonly_array(self.all_predicted_ious, np.float32)
        if masks.ndim != 3:
            raise ValueError("masks must have shape [N,H,W]")
        count = int(masks.shape[0])
        if selected.shape != (count,):
            raise ValueError("selected_hypothesis_indices must have shape [N]")
        if ious.shape != (count,):
            raise ValueError("predicted_ious must have shape [N]")
        if all_ious.shape != (count, MULTIMASK_HYPOTHESES):
            raise ValueError("all_predicted_ious must have shape [N,3]")
        if not isinstance(self.timing, SAM2BoxPromptTiming):
            raise ValueError("timing must be a SAM2BoxPromptTiming")
        if count:
            if np.any((selected < 0) | (selected >= MULTIMASK_HYPOTHESES)):
                raise ValueError("selected_hypothesis_indices are outside [0,3)")
            expected = all_ious[np.arange(count, dtype=np.int64), selected]
            if not np.array_equal(ious, expected):
                raise ValueError("predicted_ious differ from the selected all_predicted_ious")
        object.__setattr__(self, "masks", masks)
        object.__setattr__(self, "selected_hypothesis_indices", selected)
        object.__setattr__(self, "predicted_ious", ious)
        object.__setattr__(self, "all_predicted_ious", all_ious)

    @property
    def count(self) -> int:
        return int(self.masks.shape[0])

    @property
    def selected_indices(self) -> np.ndarray:
        """Read-only short alias for receipt writers."""

        return self.selected_hypothesis_indices


def _readonly_array(value: object, dtype: np.dtype[Any]) -> np.ndarray:
    array = np.ascontiguousarray(np.asarray(value, dtype=dtype))
    return np.frombuffer(array.tobytes(), dtype=dtype).reshape(array.shape)


def _hash_regular_file(path: Path, *, role: str) -> tuple[Path, int, str]:
    """Resolve and hash one stable regular file, allowing a requested symlink."""

    try:
        resolved = path.expanduser().resolve(strict=True)
        before = resolved.stat()
    except (OSError, RuntimeError) as error:
        raise SAM2BoxPromptError(f"missing SAM2 {role}: {path}") from error
    if not resolved.is_file():
        raise SAM2BoxPromptError(f"SAM2 {role} is not a regular file: {resolved}")
    digest = hashlib.sha256()
    try:
        with resolved.open("rb") as handle:
            for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(block)
        after = resolved.stat()
    except OSError as error:
        raise SAM2BoxPromptError(f"could not hash SAM2 {role}: {resolved}") from error
    before_identity = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
    after_identity = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    if before_identity != after_identity:
        raise SAM2BoxPromptError(f"SAM2 {role} changed while hashing: {resolved}")
    return resolved, int(after.st_size), digest.hexdigest()


def _source_tree_identity(source_root: Path) -> tuple[int, str]:
    """Hash all ``sam2/**/*.py`` files using a documented deterministic ledger.

    The aggregate equals ``sha256sum`` over the C-locale sorted lines
    ``<file-sha256>  <relative-path>\\n``.  Symlinked source files fail closed.
    """

    package_root = source_root / "sam2"
    try:
        candidates = sorted(
            package_root.rglob("*.py"),
            key=lambda item: item.relative_to(source_root).as_posix(),
        )
    except (OSError, RuntimeError) as error:
        raise SAM2BoxPromptError(f"could not enumerate SAM2 source: {source_root}") from error
    digest = hashlib.sha256()
    count = 0
    for candidate in candidates:
        try:
            if candidate.is_symlink() or not candidate.is_file():
                raise SAM2BoxPromptError(
                    f"SAM2 source entry is not a plain regular file: {candidate}"
                )
            relative = candidate.relative_to(source_root).as_posix()
        except (OSError, RuntimeError, ValueError) as error:
            if isinstance(error, SAM2BoxPromptError):
                raise
            raise SAM2BoxPromptError(f"invalid SAM2 source entry: {candidate}") from error
        _, _, file_sha256 = _hash_regular_file(candidate, role=f"source {relative}")
        digest.update(f"{file_sha256}  {relative}\n".encode("ascii"))
        count += 1
    return count, digest.hexdigest()


@dataclass(frozen=True)
class _VerifiedAssets:
    source_root: Path
    config_path: Path
    checkpoint_path: Path


def _verify_production_assets(
    config: SAM2BoxPromptProductionConfig,
) -> _VerifiedAssets:
    try:
        source_root = config.source_root.expanduser().resolve(strict=True)
    except (OSError, RuntimeError) as error:
        raise SAM2BoxPromptError(
            f"missing SAM2 source root: {config.source_root}"
        ) from error
    if not source_root.is_dir():
        raise SAM2BoxPromptError(f"SAM2 source root is not a directory: {source_root}")
    count, source_sha256 = _source_tree_identity(source_root)
    if count != config.source_file_count:
        raise SAM2BoxPromptError(
            f"SAM2 source file count differs: {count} != {config.source_file_count}"
        )
    if source_sha256 != config.source_tree_sha256:
        raise SAM2BoxPromptError(
            "SAM2 source tree SHA-256 differs: "
            f"{source_sha256} != {config.source_tree_sha256}"
        )

    config_path, _, config_sha256 = _hash_regular_file(
        source_root / "sam2" / config.config_name,
        role="config",
    )
    if config_sha256 != config.config_sha256:
        raise SAM2BoxPromptError(
            "SAM2 config SHA-256 differs: "
            f"{config_sha256} != {config.config_sha256}"
        )
    checkpoint_path, checkpoint_bytes, checkpoint_sha256 = _hash_regular_file(
        config.checkpoint_path,
        role="checkpoint",
    )
    if checkpoint_bytes != config.checkpoint_bytes:
        raise SAM2BoxPromptError(
            "SAM2 checkpoint byte count differs: "
            f"{checkpoint_bytes} != {config.checkpoint_bytes}"
        )
    if checkpoint_sha256 != config.checkpoint_sha256:
        raise SAM2BoxPromptError(
            "SAM2 checkpoint SHA-256 differs: "
            f"{checkpoint_sha256} != {config.checkpoint_sha256}"
        )
    return _VerifiedAssets(
        source_root=source_root,
        config_path=config_path,
        checkpoint_path=checkpoint_path,
    )


def _import_from_verified_root(source_root: Path) -> tuple[object, object, object]:
    """Import torch and SAM2 lazily, failing if another SAM2 tree won the import."""

    source_text = os.fspath(source_root)
    inserted = source_text not in sys.path
    if inserted:
        sys.path.insert(0, source_text)
    try:
        torch_module = importlib.import_module("torch")
        sam2_module = importlib.import_module("sam2")
        build_module = importlib.import_module("sam2.build_sam")
        predictor_module = importlib.import_module("sam2.sam2_image_predictor")
    except (ImportError, RuntimeError) as error:
        raise SAM2BoxPromptError("could not import the verified SAM2 runtime") from error
    finally:
        if inserted:
            try:
                sys.path.remove(source_text)
            except ValueError:
                pass
    try:
        imported_root = Path(sam2_module.__file__).resolve(strict=True).parent.parent
    except (AttributeError, OSError, RuntimeError, TypeError) as error:
        raise SAM2BoxPromptError("could not identify the imported SAM2 source") from error
    if imported_root != source_root:
        raise SAM2BoxPromptError(
            f"imported SAM2 source differs: {imported_root} != {source_root}"
        )
    return torch_module, build_module.build_sam2, predictor_module.SAM2ImagePredictor


def _sdpa_backend_names(
    *,
    old_gpu: bool,
    use_flash_attention: bool,
    math_kernel_on: bool,
    dropout_p: float,
) -> tuple[str, ...]:
    """Translate the deprecated torch 2.5 SDP flags without changing semantics."""

    if not all(
        isinstance(value, bool)
        for value in (old_gpu, use_flash_attention, math_kernel_on)
    ):
        raise SAM2BoxPromptError("SAM2 SDP backend flags must be exact bools")
    if (
        isinstance(dropout_p, bool)
        or not isinstance(dropout_p, (int, float))
        or not np.isfinite(float(dropout_p))
        or float(dropout_p) < 0.0
    ):
        raise SAM2BoxPromptError("SAM2 SDP dropout must be finite and non-negative")
    result: list[str] = []
    if use_flash_attention:
        result.append("FLASH_ATTENTION")
    if old_gpu:
        result.append("EFFICIENT_ATTENTION")
    if (old_gpu and float(dropout_p) > 0.0) or math_kernel_on:
        result.append("MATH")
    # torch.backends.cuda.sdp_kernel leaves enable_cudnn=True by default.  The
    # compatibility mapping must retain that fourth flag even when Flash wins.
    result.append("CUDNN_ATTENTION")
    return tuple(result)


def _install_sdpa_compatibility_patch(
    transformer_module: object,
    attention_module: object,
) -> Callable[[float], object]:
    """Idempotently replace only SAM2's deprecated SDP context factory."""

    current = getattr(transformer_module, "sdp_kernel_context", None)
    marker = getattr(current, _SDPA_PATCH_MARKER, None)
    if marker is not None:
        if marker != SDPA_COMPATIBILITY_POLICY_ID:
            raise SAM2BoxPromptError("an incompatible SAM2 SDP patch is installed")
        return current
    if not callable(current):
        raise SAM2BoxPromptError("SAM2 transformer SDP context factory is absent")
    flags = {}
    for name in ("OLD_GPU", "USE_FLASH_ATTN", "MATH_KERNEL_ON", "ALLOW_ALL_KERNELS"):
        value = getattr(transformer_module, name, None)
        if not isinstance(value, bool):
            raise SAM2BoxPromptError(f"SAM2 transformer {name} flag differs")
        flags[name] = value
    sdpa_kernel = getattr(attention_module, "sdpa_kernel", None)
    backend_enum = getattr(attention_module, "SDPBackend", None)
    if not callable(sdpa_kernel) or backend_enum is None:
        raise SAM2BoxPromptError("torch 2.5 SDPA compatibility API is absent")
    for name in (
        "FLASH_ATTENTION",
        "EFFICIENT_ATTENTION",
        "MATH",
        "CUDNN_ATTENTION",
    ):
        if getattr(backend_enum, name, None) is None:
            raise SAM2BoxPromptError(f"torch 2.5 SDP backend {name} is absent")

    def compatible_sdp_kernel_context(dropout_p: float) -> object:
        runtime_flags = {
            name: getattr(transformer_module, name, None)
            for name in (
                "OLD_GPU",
                "USE_FLASH_ATTN",
                "MATH_KERNEL_ON",
                "ALLOW_ALL_KERNELS",
            )
        }
        if any(not isinstance(value, bool) for value in runtime_flags.values()):
            raise SAM2BoxPromptError("SAM2 transformer runtime SDP flag differs")
        allow_all = runtime_flags["ALLOW_ALL_KERNELS"]
        if allow_all:
            return nullcontext()
        names = _sdpa_backend_names(
            old_gpu=runtime_flags["OLD_GPU"],
            use_flash_attention=runtime_flags["USE_FLASH_ATTN"],
            math_kernel_on=runtime_flags["MATH_KERNEL_ON"],
            dropout_p=dropout_p,
        )
        backends = [getattr(backend_enum, name) for name in names]
        return sdpa_kernel(backends)

    setattr(
        compatible_sdp_kernel_context,
        _SDPA_PATCH_MARKER,
        SDPA_COMPATIBILITY_POLICY_ID,
    )
    setattr(transformer_module, "sdp_kernel_context", compatible_sdp_kernel_context)
    installed = getattr(transformer_module, "sdp_kernel_context", None)
    if (
        installed is not compatible_sdp_kernel_context
        or getattr(installed, _SDPA_PATCH_MARKER, None)
        != SDPA_COMPATIBILITY_POLICY_ID
    ):
        raise SAM2BoxPromptError("SAM2 SDP compatibility patch installation failed")
    return compatible_sdp_kernel_context


def _install_verified_production_sdpa_patch(
    *, source_root: Path, torch_module: object
) -> dict[str, object]:
    """Authenticate the exact SAM2/torch modules, install, then verify the patch."""

    if str(getattr(torch_module, "__version__", "")) != EXPECTED_TORCH_VERSION:
        raise SAM2BoxPromptError("SAM2 SDP patch requires the frozen torch 2.5.1 build")
    try:
        transformer_module = importlib.import_module(
            "sam2.modeling.sam.transformer"
        )
        attention_module = importlib.import_module("torch.nn.attention")
    except (ImportError, RuntimeError) as error:
        raise SAM2BoxPromptError("could not import the frozen SDP compatibility API") from error
    transformer_path, _, transformer_sha = _hash_regular_file(
        Path(str(getattr(transformer_module, "__file__", ""))),
        role="transformer compatibility source",
    )
    expected_transformer_path = (
        source_root / EXPECTED_SAM2_TRANSFORMER_RELATIVE_PATH
    ).resolve(strict=True)
    if (
        transformer_path != expected_transformer_path
        or transformer_sha != EXPECTED_SAM2_TRANSFORMER_SHA256
    ):
        raise SAM2BoxPromptError("SAM2 transformer compatibility source differs")
    attention_path, _, attention_sha = _hash_regular_file(
        Path(str(getattr(attention_module, "__file__", ""))),
        role="torch attention compatibility source",
    )
    if attention_sha != EXPECTED_TORCH_ATTENTION_SHA256:
        raise SAM2BoxPromptError("torch attention compatibility source differs")
    installed = _install_sdpa_compatibility_patch(
        transformer_module, attention_module
    )
    if (
        getattr(transformer_module, "sdp_kernel_context", None) is not installed
        or getattr(installed, _SDPA_PATCH_MARKER, None)
        != SDPA_COMPATIBILITY_POLICY_ID
    ):
        raise SAM2BoxPromptError("verified SAM2 SDP compatibility patch differs")
    return {
        "policy_id": SDPA_COMPATIBILITY_POLICY_ID,
        "torch_version": EXPECTED_TORCH_VERSION,
        "transformer_path": os.fspath(transformer_path),
        "transformer_sha256": transformer_sha,
        "torch_attention_path": os.fspath(attention_path),
        "torch_attention_sha256": attention_sha,
        "deprecated_future_warning_authenticated": False,
    }


def _build_production_predictor(
    config: SAM2BoxPromptProductionConfig,
) -> tuple[object, object]:
    assets = _verify_production_assets(config)
    torch_module, build_sam2, predictor_class = _import_from_verified_root(
        assets.source_root
    )
    try:
        model = build_sam2(
            config_file=config.config_name,
            ckpt_path=os.fspath(assets.checkpoint_path),
            device=config.device,
            mode="eval",
            apply_postprocessing=config.apply_postprocessing,
        )
        model.eval()
        model.requires_grad_(False)
        predictor = predictor_class(model)
    except Exception as error:
        raise SAM2BoxPromptError("could not construct frozen SAM2 image predictor") from error
    try:
        if bool(model.training) or any(
            bool(parameter.requires_grad) for parameter in model.parameters()
        ):
            raise SAM2BoxPromptError("SAM2 model is not frozen")
    except SAM2BoxPromptError:
        raise
    except Exception as error:
        raise SAM2BoxPromptError("could not inspect frozen SAM2 model") from error
    _install_verified_production_sdpa_patch(
        source_root=assets.source_root, torch_module=torch_module
    )
    return predictor, torch_module


def _to_numpy(value: object, *, role: str) -> np.ndarray:
    try:
        detached = value.detach() if hasattr(value, "detach") else value
        host = detached.cpu() if hasattr(detached, "cpu") else detached
        materialized = host.numpy() if hasattr(host, "numpy") else host
        return np.array(materialized, copy=True)
    except (AttributeError, TypeError, ValueError, RuntimeError) as error:
        raise SAM2BoxPromptError(f"could not materialize SAM2 {role}") from error


def _validate_image(image_rgb: object) -> np.ndarray:
    if not isinstance(image_rgb, np.ndarray):
        raise ValueError("image_rgb must be a numpy array")
    if (
        image_rgb.dtype != np.uint8
        or image_rgb.shape != (IMAGE_HEIGHT, IMAGE_WIDTH, 3)
    ):
        raise ValueError("image_rgb must have dtype uint8 and shape [480,640,3]")
    return np.array(image_rgb, dtype=np.uint8, order="C", copy=True)


def _validate_boxes(
    boxes_xyxy: object,
    *,
    height: int,
    width: int,
    max_boxes: int,
) -> np.ndarray:
    if boxes_xyxy is None:
        return np.empty((0, 4), dtype=np.float32)
    try:
        boxes = np.asarray(boxes_xyxy, dtype=np.float32)
    except (TypeError, ValueError) as error:
        raise ValueError("boxes_xyxy must be finite XYXY coordinates") from error
    if boxes.shape == (0,):
        boxes = np.empty((0, 4), dtype=np.float32)
    if boxes.ndim != 2 or boxes.shape[1:] != (4,):
        raise ValueError("boxes_xyxy must have shape [N,4]")
    count = int(boxes.shape[0])
    if count > max_boxes:
        raise ValueError(f"boxes_xyxy supports at most {max_boxes} boxes per frame")
    if not np.isfinite(boxes).all():
        raise ValueError("boxes_xyxy must be finite")
    if count:
        x1, y1, x2, y2 = boxes.T
        if (
            np.any(x1 < 0.0)
            or np.any(y1 < 0.0)
            # F0 prompt boxes use inclusive pixel coordinates.  The last
            # legal coordinates are therefore width-1 and height-1; accepting
            # x2==width or y2==height would silently switch to half-open box
            # semantics at the provider boundary.
            or np.any(x2 >= float(width))
            or np.any(y2 >= float(height))
            or np.any(x2 <= x1)
            or np.any(y2 <= y1)
        ):
            raise ValueError("boxes_xyxy must be non-empty original-image coordinates")
    return np.array(boxes, dtype=np.float32, order="C", copy=True)


def _select_multimask_outputs(
    raw_output: object,
    *,
    box_count: int,
    height: int,
    width: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if not isinstance(raw_output, (tuple, list)) or len(raw_output) != 3:
        raise SAM2BoxPromptError("SAM2 predict must return masks, IoUs, and low-res masks")
    masks = _to_numpy(raw_output[0], role="masks")
    ious = _to_numpy(raw_output[1], role="predicted IoUs")
    if box_count == 1 and masks.ndim == 3:
        masks = masks[None, ...]
    if box_count == 1 and ious.ndim == 1:
        ious = ious[None, ...]
    if (
        masks.ndim != 4
        or masks.shape[0] != box_count
        or masks.shape[2:] != (height, width)
        or masks.shape[1] != MULTIMASK_HYPOTHESES
        or ious.shape != masks.shape[:2]
    ):
        raise SAM2BoxPromptError(
            "SAM2 multimask output shape differs: "
            f"masks={masks.shape}, ious={ious.shape}"
        )
    if not (
        np.issubdtype(masks.dtype, np.bool_)
        or np.issubdtype(masks.dtype, np.number)
    ) or not np.isfinite(masks).all():
        raise SAM2BoxPromptError("SAM2 masks must be finite boolean or numeric values")
    if np.any((masks != 0) & (masks != 1)):
        raise SAM2BoxPromptError(
            "SAM2 return_logits=False masks must contain exact binary {0,1} values"
        )
    if not np.issubdtype(ious.dtype, np.number) or not np.isfinite(ious).all():
        raise SAM2BoxPromptError("SAM2 predicted IoUs must be finite numeric values")

    # np.argmax is stable toward the first occurrence, which freezes the exact
    # lowest-index tie break required by the N0 protocol.
    selected = np.argmax(ious, axis=1).astype(np.int64, copy=False)
    rows = np.arange(box_count, dtype=np.int64)
    # Production asks SAM2 for already-thresholded masks
    # (return_logits=False).  Numeric fixtures are accepted only when every
    # value is exactly 0 or 1, so conversion cannot hide logits or malformed
    # probabilities behind a second provider-side threshold.
    selected_masks = masks[rows, selected].astype(np.bool_, copy=False)
    selected_ious = ious[rows, selected]
    return (
        np.asarray(selected_masks, dtype=np.bool_),
        np.asarray(selected, dtype=np.int64),
        np.asarray(selected_ious, dtype=np.float32),
        np.asarray(ious, dtype=np.float32),
    )


class FrozenSAM2BoxPromptProvider:
    """Lazy, frozen, current-frame-only SAM2.1 box-prompt decoder."""

    def __init__(
        self,
        *,
        config: SAM2BoxPromptProductionConfig = PRODUCTION_CONFIG,
        predictor_factory: Callable[[SAM2BoxPromptProductionConfig], object] | None = None,
        torch_module: object | None = None,
        clock_ns: Callable[[], int] = time.perf_counter_ns,
    ) -> None:
        if config.max_boxes_per_frame != MAX_BOXES_PER_FRAME:
            raise ValueError(
                f"production max_boxes_per_frame must remain {MAX_BOXES_PER_FRAME}"
            )
        if (
            config.multimask_output is not True
            or config.return_logits is not False
            or config.normalize_coords is not True
            or config.mask_threshold != 0.0
            or config.multimask_hypotheses != MULTIMASK_HYPOTHESES
            or config.sdpa_compatibility_policy_id
            != SDPA_COMPATIBILITY_POLICY_ID
            or config.torch_version != EXPECTED_TORCH_VERSION
            or config.transformer_relative_path
            != EXPECTED_SAM2_TRANSFORMER_RELATIVE_PATH
            or config.transformer_sha256 != EXPECTED_SAM2_TRANSFORMER_SHA256
            or config.torch_attention_sha256
            != EXPECTED_TORCH_ATTENTION_SHA256
        ):
            raise ValueError("SAM2 box-prompt selection policy differs")
        self._config = config
        self._predictor_factory = predictor_factory
        self._predictor: object | None = None
        self._torch: object | None = torch_module
        self._clock_ns = clock_ns
        self._poisoned = False
        self._lock = threading.Lock()

    @property
    def config(self) -> SAM2BoxPromptProductionConfig:
        return self._config

    @property
    def loaded(self) -> bool:
        return self._predictor is not None

    def _ensure_predictor(self) -> object:
        if self._poisoned:
            raise SAM2BoxPromptError("SAM2 provider is poisoned after failed state cleanup")
        if self._predictor is not None:
            return self._predictor
        try:
            if self._predictor_factory is None:
                predictor, torch_module = _build_production_predictor(self._config)
                self._torch = torch_module
            else:
                predictor = self._predictor_factory(self._config)
        except SAM2BoxPromptError:
            raise
        except Exception as error:
            raise SAM2BoxPromptError("could not lazily construct SAM2 predictor") from error
        for method in ("set_image", "predict", "reset_predictor"):
            if not callable(getattr(predictor, method, None)):
                raise SAM2BoxPromptError(f"SAM2 predictor lacks callable {method}")
        self._predictor = predictor
        return predictor

    def _enter_inference_contexts(self, stack: ExitStack) -> None:
        if self._torch is None:
            return
        try:
            stack.enter_context(self._torch.inference_mode())
            device_type = self._config.device.split(":", 1)[0].lower()
            if device_type == "cuda":
                dtype = getattr(self._torch, self._config.autocast_dtype)
                stack.enter_context(
                    self._torch.autocast(device_type="cuda", dtype=dtype)
                )
        except Exception as error:
            raise SAM2BoxPromptError("could not enter frozen SAM2 inference context") from error

    def _cuda_timing_enabled(self) -> bool:
        return self._torch is not None and self._config.device.split(":", 1)[0].lower() == "cuda"

    def _cuda_synchronize(self, *, phase: str) -> None:
        try:
            self._torch.cuda.synchronize(self._config.device)
        except Exception as error:
            raise SAM2BoxPromptError(
                f"could not synchronize CUDA {phase}"
            ) from error

    def _clock(self) -> int:
        try:
            value = int(self._clock_ns())
        except Exception as error:
            raise SAM2BoxPromptError("could not read SAM2 provider clock") from error
        if value < 0:
            raise SAM2BoxPromptError("SAM2 provider clock returned a negative value")
        return value

    def predict(
        self,
        image_rgb: np.ndarray,
        boxes_xyxy: object,
    ) -> SAM2BoxPromptResult:
        """Decode source-order-preserving masks for one current RGB frame.

        ``boxes_xyxy`` are original-image pixel coordinates.  ``None``, ``[]``,
        and an array of shape ``[0,4]`` all mean an empty source batch.  That
        case returns empty arrays and deliberately does not load SAM2.
        """

        image = _validate_image(image_rgb)
        height, width = image.shape[:2]
        boxes = _validate_boxes(
            boxes_xyxy,
            height=height,
            width=width,
            max_boxes=self._config.max_boxes_per_frame,
        )
        count = int(boxes.shape[0])
        if count == 0:
            return SAM2BoxPromptResult(
                masks=np.empty((0, height, width), dtype=np.bool_),
                selected_hypothesis_indices=np.empty((0,), dtype=np.int64),
                predicted_ious=np.empty((0,), dtype=np.float32),
                all_predicted_ious=np.empty(
                    (0, MULTIMASK_HYPOTHESES), dtype=np.float32
                ),
                timing=EMPTY_TIMING,
            )

        with self._lock:
            predictor = self._ensure_predictor()
            cuda_synchronized = self._cuda_timing_enabled()
            if cuda_synchronized:
                self._cuda_synchronize(phase="before set_image")
                try:
                    self._torch.cuda.reset_peak_memory_stats(self._config.device)
                except Exception as error:
                    raise SAM2BoxPromptError(
                        "could not reset CUDA peak memory before set_image"
                    ) from error
            started_ns = self._clock()
            encoder_finished_ns: int | None = None
            raw_output: object | None = None
            inference_error: Exception | None = None
            try:
                with ExitStack() as stack:
                    self._enter_inference_contexts(stack)
                    # Exactly one embedding and one batched decoder call for
                    # this frame.  Copies prevent third-party mutation of the
                    # caller's image and source boxes.
                    predictor.set_image(image)
                    if cuda_synchronized:
                        self._cuda_synchronize(phase="after set_image")
                    encoder_finished_ns = self._clock()
                    raw_output = predictor.predict(
                        point_coords=None,
                        point_labels=None,
                        box=boxes,
                        mask_input=None,
                        multimask_output=True,
                        return_logits=False,
                        normalize_coords=True,
                    )
            except Exception as error:
                inference_error = error
            try:
                predictor.reset_predictor()
            except Exception as cleanup_error:
                self._poisoned = True
                raise SAM2BoxPromptError(
                    "SAM2 predictor state cleanup failed; provider is poisoned"
                ) from cleanup_error
            if inference_error is not None:
                if isinstance(inference_error, SAM2BoxPromptError):
                    raise inference_error
                raise SAM2BoxPromptError("SAM2 box-prompt inference failed") from inference_error
            if encoder_finished_ns is None:
                raise SAM2BoxPromptError("SAM2 encoder timing boundary is absent")
            masks, selected, selected_ious, all_ious = _select_multimask_outputs(
                raw_output,
                box_count=count,
                height=height,
                width=width,
            )
            if cuda_synchronized:
                self._cuda_synchronize(phase="at complete frame end")
            finished_ns = self._clock()
            if finished_ns < encoder_finished_ns or encoder_finished_ns < started_ns:
                raise SAM2BoxPromptError("SAM2 provider clock moved backwards")
            if cuda_synchronized:
                try:
                    peak_allocated = int(
                        self._torch.cuda.max_memory_allocated(self._config.device)
                    )
                except Exception as error:
                    raise SAM2BoxPromptError(
                        "could not read CUDA peak allocated memory"
                    ) from error
                if peak_allocated < 0:
                    raise SAM2BoxPromptError(
                        "CUDA peak allocated memory returned a negative value"
                    )
            else:
                peak_allocated = 0
            encoder_ms = (encoder_finished_ns - started_ns) / 1_000_000.0
            decoder_and_host_ms = (
                finished_ns - encoder_finished_ns
            ) / 1_000_000.0
            complete_ms = (finished_ns - started_ns) / 1_000_000.0
            timing = SAM2BoxPromptTiming(
                encoder_ms=encoder_ms,
                decoder_and_host_mask_ms=decoder_and_host_ms,
                complete_ms=complete_ms,
                cuda_synchronized=cuda_synchronized,
                peak_allocated_memory_bytes=peak_allocated,
            )
            return SAM2BoxPromptResult(
                masks=masks,
                selected_hypothesis_indices=selected,
                predicted_ious=selected_ious,
                all_predicted_ious=all_ious,
                timing=timing,
            )

    __call__ = predict
