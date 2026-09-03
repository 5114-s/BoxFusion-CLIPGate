"""Frozen, class-agnostic FastSAM automatic-mask provider for F0.

The provider has one deliberately narrow interface: one BGR ``uint8`` image
in and automatic binary masks, confidences, and 2D boxes out.  It never reads
or returns class IDs, names, prompts, native detections, history, tracking, or
birth decisions.  Checkpoint identity and every prediction argument are
fixed so an F0 shadow receipt can be audited independently of downstream
residual geometry.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import stat
import threading
import time
from types import MappingProxyType
from typing import Any, Callable, Mapping

import numpy as np


SCHEMA = "boxfusion.fastsam_automatic_provider.f0.v1"
EXPECTED_ULTRALYTICS_VERSION = "8.4.105"
EXPECTED_CHECKPOINT_BYTES = 144_943_063
EXPECTED_CHECKPOINT_SHA256 = (
    "c0be4e7ddbe4c15333d15a859c676d053c486d0a746a3be6a7a9790d52a9b6d7"
)

IMAGE_SHAPE = (480, 640, 3)
MASK_SHAPE = (480, 640)
MASK_THRESHOLD = 0.5

PREDICT_POLICY: Mapping[str, object] = MappingProxyType(
    {
        "imgsz": 1024,
        "conf": 0.25,
        "iou": 0.90,
        "max_det": 100,
        "agnostic_nms": True,
        "retina_masks": True,
        "classes": None,
        "augment": False,
        "half": False,
        "batch": 1,
        "verbose": False,
        "save": False,
        "stream": False,
        "source_container": "one_element_list",
        "source_color_order": "BGR",
        "source_dtype": "uint8",
        "source_shape": IMAGE_SHAPE,
        "mask_threshold": MASK_THRESHOLD,
        "semantic_outputs": None,
    }
)


class FastSAMProviderError(RuntimeError):
    """A checkpoint, model, CUDA, or FastSAM result violated the F0 contract."""


@dataclass(frozen=True)
class FastSAMCheckpointIdentity:
    """Exact on-disk identity of the frozen FastSAM-x checkpoint."""

    path: str
    byte_count: int
    sha256: str


@dataclass(frozen=True)
class FastSAMProviderTiming:
    """Synchronized provider latency and torch allocator observations.

    Allocator values are point observations except for the two explicitly
    named peaks.  They are zero on a CPU device.  CUDA synchronization bounds
    the prediction interval; it is not a continuous device-wide profiler.
    """

    device: str
    started_ns: int
    prediction_finished_ns: int
    finished_ns: int
    prediction_seconds: float
    extraction_seconds: float
    total_seconds: float
    cuda_synchronized: bool
    memory_allocated_before_bytes: int
    memory_allocated_after_bytes: int
    memory_reserved_before_bytes: int
    memory_reserved_after_bytes: int
    max_memory_allocated_bytes: int
    max_memory_reserved_bytes: int


@dataclass(frozen=True)
class FastSAMAutomaticMaskResult:
    """Immutable, semantic-free output for one image."""

    masks: np.ndarray
    confidences: np.ndarray
    boxes_xyxy: np.ndarray
    timing: FastSAMProviderTiming

    def __post_init__(self) -> None:
        masks = _readonly_array(self.masks, np.bool_)
        confidences = _readonly_array(self.confidences, np.float32)
        boxes = _readonly_array(self.boxes_xyxy, np.float32)
        count = int(masks.shape[0]) if masks.ndim == 3 else -1
        if masks.shape != (count, *MASK_SHAPE):
            raise ValueError("masks must have shape [N,480,640]")
        if confidences.shape != (count,):
            raise ValueError("confidences must have shape [N]")
        if boxes.shape != (count, 4):
            raise ValueError("boxes_xyxy must have shape [N,4]")
        object.__setattr__(self, "masks", masks)
        object.__setattr__(self, "confidences", confidences)
        object.__setattr__(self, "boxes_xyxy", boxes)

    @property
    def conf(self) -> np.ndarray:
        """Short, read-only compatibility alias; no class data is attached."""

        return self.confidences

    @property
    def boxes(self) -> np.ndarray:
        """Short, read-only compatibility alias for ``boxes_xyxy``."""

        return self.boxes_xyxy

    @property
    def count(self) -> int:
        return int(self.masks.shape[0])


def _readonly_array(value: object, dtype: np.dtype[Any]) -> np.ndarray:
    array = np.ascontiguousarray(np.asarray(value, dtype=dtype))
    # A bytes-backed array cannot be made writeable again, unlike merely
    # setting ndarray.flags.writeable=False on an owned mutable allocation.
    return np.frombuffer(array.tobytes(), dtype=dtype).reshape(array.shape)


def _validate_checkpoint(
    checkpoint_path: str | Path,
    *,
    expected_bytes: int = EXPECTED_CHECKPOINT_BYTES,
    expected_sha256: str = EXPECTED_CHECKPOINT_SHA256,
) -> FastSAMCheckpointIdentity:
    """Validate a non-symlink regular file and hash it through one descriptor.

    The keyword overrides exist only to make this private primitive testable
    with a small fixture.  The public provider always calls it with the frozen
    constants above.
    """

    candidate = Path(checkpoint_path).expanduser()
    try:
        before = candidate.lstat()
    except OSError as error:
        raise FastSAMProviderError(f"FastSAM checkpoint is missing: {candidate}") from error
    if not stat.S_ISREG(before.st_mode):
        raise FastSAMProviderError(
            f"FastSAM checkpoint must be a non-symlink regular file: {candidate}"
        )
    if before.st_size != expected_bytes:
        raise FastSAMProviderError(
            f"FastSAM checkpoint byte count differs: {before.st_size} != {expected_bytes}"
        )

    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = -1
    digest = hashlib.sha256()
    try:
        descriptor = os.open(candidate, flags)
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode):
            raise FastSAMProviderError("opened FastSAM checkpoint is not a regular file")
        if (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino):
            raise FastSAMProviderError("FastSAM checkpoint changed while opening")
        while True:
            chunk = os.read(descriptor, 8 * 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
        after = os.fstat(descriptor)
    except FastSAMProviderError:
        raise
    except OSError as error:
        raise FastSAMProviderError(f"could not read FastSAM checkpoint: {candidate}") from error
    finally:
        if descriptor >= 0:
            os.close(descriptor)

    before_identity = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
    after_identity = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    if before_identity != after_identity:
        raise FastSAMProviderError("FastSAM checkpoint changed while hashing")
    actual_sha256 = digest.hexdigest()
    if actual_sha256 != expected_sha256:
        raise FastSAMProviderError(
            "FastSAM checkpoint SHA-256 differs: "
            f"{actual_sha256} != {expected_sha256}"
        )
    return FastSAMCheckpointIdentity(
        path=str(candidate.resolve(strict=True)),
        byte_count=int(after.st_size),
        sha256=actual_sha256,
    )


def _default_model_factory(checkpoint_path: str) -> object:
    try:
        import ultralytics
        from ultralytics import FastSAM
    except ImportError as error:
        raise FastSAMProviderError("ultralytics with FastSAM is unavailable") from error
    if ultralytics.__version__ != EXPECTED_ULTRALYTICS_VERSION:
        raise FastSAMProviderError(
            "ultralytics version differs: "
            f"{ultralytics.__version__} != {EXPECTED_ULTRALYTICS_VERSION}"
        )
    return FastSAM(checkpoint_path)


def _to_numpy(value: object, *, role: str) -> np.ndarray:
    try:
        detached = value.detach() if hasattr(value, "detach") else value
        host = detached.cpu() if hasattr(detached, "cpu") else detached
        array = host.numpy() if hasattr(host, "numpy") else np.asarray(host)
        return np.array(array, copy=True)
    except (AttributeError, TypeError, ValueError, RuntimeError) as error:
        raise FastSAMProviderError(f"could not materialize FastSAM {role}") from error


def _extract_result(raw_results: object) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not isinstance(raw_results, (list, tuple)) or len(raw_results) != 1:
        raise FastSAMProviderError("FastSAM must return exactly one result for one source")
    result = raw_results[0]
    try:
        orig_shape = tuple(int(value) for value in result.orig_shape)
    except (AttributeError, TypeError, ValueError) as error:
        raise FastSAMProviderError("FastSAM result has no valid orig_shape") from error
    if orig_shape != MASK_SHAPE:
        raise FastSAMProviderError(
            f"FastSAM orig_shape differs: {orig_shape} != {MASK_SHAPE}"
        )

    boxes_object = getattr(result, "boxes", None)
    if boxes_object is None:
        boxes = np.empty((0, 4), dtype=np.float32)
        confidences = np.empty((0,), dtype=np.float32)
    else:
        if not hasattr(boxes_object, "xyxy") or not hasattr(boxes_object, "conf"):
            raise FastSAMProviderError("FastSAM boxes lack xyxy or conf")
        boxes = _to_numpy(boxes_object.xyxy, role="boxes.xyxy")
        confidences = _to_numpy(boxes_object.conf, role="boxes.conf")
        if boxes.ndim != 2 or boxes.shape[1:] != (4,):
            raise FastSAMProviderError("FastSAM boxes.xyxy must have shape [N,4]")
        if confidences.ndim != 1 or confidences.shape[0] != boxes.shape[0]:
            raise FastSAMProviderError("FastSAM boxes.conf must have shape [N]")
        if boxes.shape[0] > int(PREDICT_POLICY["max_det"]):
            raise FastSAMProviderError("FastSAM result exceeds frozen max_det")
        if not np.isfinite(boxes).all() or not np.isfinite(confidences).all():
            raise FastSAMProviderError("FastSAM boxes and confidences must be finite")
        if np.any((confidences < 0.0) | (confidences > 1.0)):
            raise FastSAMProviderError("FastSAM confidences must be within [0,1]")
        if boxes.size:
            x1, y1, x2, y2 = boxes.T
            if (
                np.any(x1 < 0.0)
                or np.any(y1 < 0.0)
                or np.any(x2 > MASK_SHAPE[1])
                or np.any(y2 > MASK_SHAPE[0])
                or np.any(x2 <= x1)
                or np.any(y2 <= y1)
            ):
                raise FastSAMProviderError("FastSAM boxes are outside image bounds")

    masks_object = getattr(result, "masks", None)
    count = int(boxes.shape[0])
    if masks_object is None:
        if count != 0:
            raise FastSAMProviderError("FastSAM returned boxes without masks")
        masks = np.empty((0, *MASK_SHAPE), dtype=np.bool_)
    else:
        if not hasattr(masks_object, "data"):
            raise FastSAMProviderError("FastSAM masks lack data")
        masks_float = _to_numpy(masks_object.data, role="masks.data")
        if masks_float.shape != (count, *MASK_SHAPE):
            raise FastSAMProviderError(
                "FastSAM masks.data must have shape [N,480,640] matching boxes"
            )
        if not np.isfinite(masks_float).all():
            raise FastSAMProviderError("FastSAM masks must be finite")
        if np.any((masks_float < 0.0) | (masks_float > 1.0)):
            raise FastSAMProviderError("FastSAM masks must be within [0,1]")
        masks = masks_float >= MASK_THRESHOLD

    return (
        np.asarray(masks, dtype=np.bool_),
        np.asarray(confidences, dtype=np.float32),
        np.asarray(boxes, dtype=np.float32),
    )


class FrozenFastSAMAutomaticMaskProvider:
    """Own one verified, frozen FastSAM-x model and run exact F0 inference."""

    def __init__(
        self,
        checkpoint_path: str | Path,
        *,
        device: str,
        model_factory: Callable[[str], object] | None = None,
        torch_module: object | None = None,
        clock_ns: Callable[[], int] = time.perf_counter_ns,
    ) -> None:
        if not isinstance(device, str) or not device.strip():
            raise ValueError("device must be a non-empty string")
        self._device = device.strip()
        self._clock_ns = clock_ns
        self._checkpoint = _validate_checkpoint(checkpoint_path)
        if torch_module is None:
            try:
                import torch as torch_module_imported
            except ImportError as error:
                raise FastSAMProviderError("torch is unavailable") from error
            torch_module = torch_module_imported
        self._torch = torch_module
        factory = _default_model_factory if model_factory is None else model_factory
        try:
            self._model = factory(self._checkpoint.path)
        except FastSAMProviderError:
            raise
        except Exception as error:
            raise FastSAMProviderError("could not construct frozen FastSAM") from error
        self._freeze_model()
        self._lock = threading.Lock()

        self._is_cuda = self._device.lower().startswith("cuda")
        self._torch_device: object = self._device
        if self._is_cuda:
            try:
                if not bool(self._torch.cuda.is_available()):
                    raise FastSAMProviderError("CUDA device requested but CUDA is unavailable")
                self._torch_device = self._torch.device(self._device)
            except FastSAMProviderError:
                raise
            except Exception as error:
                raise FastSAMProviderError(f"invalid CUDA device: {self._device}") from error

    @property
    def checkpoint(self) -> FastSAMCheckpointIdentity:
        return self._checkpoint

    @property
    def device(self) -> str:
        return self._device

    def _freeze_model(self) -> None:
        if not callable(getattr(self._model, "eval", None)) or not callable(
            getattr(self._model, "requires_grad_", None)
        ):
            raise FastSAMProviderError("FastSAM model cannot be frozen")
        try:
            self._model.eval()
            self._model.requires_grad_(False)
        except Exception as error:
            raise FastSAMProviderError("could not freeze FastSAM model") from error
        self._assert_model_frozen()

    def _assert_model_frozen(self) -> None:
        if bool(getattr(self._model, "training", True)):
            raise FastSAMProviderError("FastSAM model is not in eval mode")
        if not callable(getattr(self._model, "parameters", None)):
            raise FastSAMProviderError("FastSAM model parameters are not inspectable")
        try:
            if any(bool(parameter.requires_grad) for parameter in self._model.parameters()):
                raise FastSAMProviderError("FastSAM model has trainable parameters")
        except FastSAMProviderError:
            raise
        except Exception as error:
            raise FastSAMProviderError("could not inspect FastSAM parameters") from error

    def _cuda_memory(self, name: str) -> int:
        try:
            value = getattr(self._torch.cuda, name)(self._torch_device)
            result = int(value)
        except Exception as error:
            raise FastSAMProviderError(f"could not read CUDA {name}") from error
        if result < 0:
            raise FastSAMProviderError(f"CUDA {name} returned a negative value")
        return result

    def infer_bgr(self, image_bgr: np.ndarray) -> FastSAMAutomaticMaskResult:
        """Run the exact, class-agnostic automatic-mask call for one frame."""

        if not isinstance(image_bgr, np.ndarray):
            raise ValueError("image_bgr must be a numpy array")
        if image_bgr.dtype != np.uint8 or image_bgr.shape != IMAGE_SHAPE:
            raise ValueError("image_bgr must have dtype uint8 and shape [480,640,3]")
        # Own the source supplied to a third-party library and preserve BGR
        # bytes exactly.  No channel conversion or normalization occurs here.
        source = np.array(image_bgr, dtype=np.uint8, order="C", copy=True)

        with self._lock:
            self._assert_model_frozen()
            if self._is_cuda:
                try:
                    self._torch.cuda.synchronize(self._torch_device)
                    self._torch.cuda.reset_peak_memory_stats(self._torch_device)
                except Exception as error:
                    raise FastSAMProviderError("could not initialize CUDA timing") from error
                allocated_before = self._cuda_memory("memory_allocated")
                reserved_before = self._cuda_memory("memory_reserved")
            else:
                allocated_before = 0
                reserved_before = 0

            started_ns = int(self._clock_ns())
            try:
                with self._torch.inference_mode():
                    raw_results = self._model.predict(
                        source=[source],
                        imgsz=1024,
                        conf=0.25,
                        iou=0.90,
                        max_det=100,
                        agnostic_nms=True,
                        retina_masks=True,
                        classes=None,
                        augment=False,
                        half=False,
                        batch=1,
                        device=self._device,
                        verbose=False,
                        save=False,
                        stream=False,
                    )
                if self._is_cuda:
                    self._torch.cuda.synchronize(self._torch_device)
            except Exception as error:
                # Make a best effort to leave CUDA timing at a synchronization
                # boundary, while preserving the original provider failure.
                if self._is_cuda:
                    try:
                        self._torch.cuda.synchronize(self._torch_device)
                    except Exception:
                        pass
                raise FastSAMProviderError("FastSAM automatic-mask prediction failed") from error
            prediction_finished_ns = int(self._clock_ns())
            self._assert_model_frozen()

            masks, confidences, boxes = _extract_result(raw_results)
            finished_ns = int(self._clock_ns())
            if not (started_ns <= prediction_finished_ns <= finished_ns):
                raise FastSAMProviderError("provider clock is not monotonic")

            if self._is_cuda:
                allocated_after = self._cuda_memory("memory_allocated")
                reserved_after = self._cuda_memory("memory_reserved")
                max_allocated = self._cuda_memory("max_memory_allocated")
                max_reserved = self._cuda_memory("max_memory_reserved")
            else:
                allocated_after = 0
                reserved_after = 0
                max_allocated = 0
                max_reserved = 0

            timing = FastSAMProviderTiming(
                device=self._device,
                started_ns=started_ns,
                prediction_finished_ns=prediction_finished_ns,
                finished_ns=finished_ns,
                prediction_seconds=(prediction_finished_ns - started_ns) / 1e9,
                extraction_seconds=(finished_ns - prediction_finished_ns) / 1e9,
                total_seconds=(finished_ns - started_ns) / 1e9,
                cuda_synchronized=self._is_cuda,
                memory_allocated_before_bytes=allocated_before,
                memory_allocated_after_bytes=allocated_after,
                memory_reserved_before_bytes=reserved_before,
                memory_reserved_after_bytes=reserved_after,
                max_memory_allocated_bytes=max_allocated,
                max_memory_reserved_bytes=max_reserved,
            )
            return FastSAMAutomaticMaskResult(
                masks=masks,
                confidences=confidences,
                boxes_xyxy=boxes,
                timing=timing,
            )

    predict = infer_bgr
    __call__ = infer_bgr
