"""Frozen, bounded, causal SAM2.1 video-mask provider for N0 shadow tests.

The provider owns exactly one object and one short track per call.  Observation
zero is an externally frozen binary seed mask.  For every later observation it
first asks SAM2 to predict from the already committed prefix and materializes
that prediction; only afterwards is the current frozen mask committed as a
correction for future observations.  This query-before-commit ordering makes
current-mask leakage impossible at the API boundary.

No scores, labels, detector predictions, ground truth, or evaluator objects are
accepted.  Production tracks are restricted to three through five observations
so the online state is explicitly bounded.
"""

from __future__ import annotations

from contextlib import ExitStack, nullcontext, redirect_stdout
from dataclasses import dataclass
import importlib
import io
import os
import time
from typing import Callable

import numpy as np

from boxfusion import sam2_boxprompt_provider as image_provider


SCHEMA = "boxfusion.sam2_video_track_provider.n0.v1"
PROTOCOL_ID = "N0-SAM2-PAST-ONLY-QUERY-BEFORE-COMMIT-V1"
IMAGE_HEIGHT = image_provider.IMAGE_HEIGHT
IMAGE_WIDTH = image_provider.IMAGE_WIDTH
MIN_TRACK_OBSERVATIONS = 3
MAX_TRACK_OBSERVATIONS = 5
OBJECT_ID = 1


class SAM2VideoTrackError(RuntimeError):
    """The frozen SAM2 runtime or its causal output violated the contract."""


@dataclass(frozen=True)
class SAM2VideoTrackConfig:
    """Frozen production identity plus bounded video inference choices."""

    image_config: image_provider.SAM2BoxPromptProductionConfig = (
        image_provider.PRODUCTION_CONFIG
    )
    min_track_observations: int = MIN_TRACK_OBSERVATIONS
    max_track_observations: int = MAX_TRACK_OBSERVATIONS
    object_id: int = OBJECT_ID
    offload_video_to_cpu: bool = False
    offload_state_to_cpu: bool = False
    mask_logit_threshold: float = 0.0


PRODUCTION_CONFIG = SAM2VideoTrackConfig()


@dataclass(frozen=True)
class SAM2VideoObservationTiming:
    add_frame_ms: float
    infer_ms: float
    commit_ms: float
    complete_ms: float

    def __post_init__(self) -> None:
        values = np.asarray(
            [self.add_frame_ms, self.infer_ms, self.commit_ms, self.complete_ms],
            dtype=np.float64,
        )
        if not np.isfinite(values).all() or np.any(values < 0.0):
            raise ValueError("SAM2 video timing must be finite and non-negative")
        expected = float(values[:3].sum())
        if abs(expected - float(values[3])) > max(1e-6, abs(expected) * 1e-9):
            raise ValueError("complete_ms must equal add_frame+infer+commit")


@dataclass(frozen=True)
class SAM2VideoTrackResult:
    """Immutable observation-order output and causal access receipt."""

    masks: np.ndarray
    timings: tuple[SAM2VideoObservationTiming, ...]
    predicted_flags: tuple[bool, ...]
    committed_flags: tuple[bool, ...]
    maximum_lookahead_observations: int
    max_state_observations: int
    cuda_synchronized: bool
    peak_allocated_memory_bytes: int

    def __post_init__(self) -> None:
        masks = np.ascontiguousarray(np.asarray(self.masks, dtype=np.bool_))
        if masks.ndim != 3 or masks.shape[1:] != (IMAGE_HEIGHT, IMAGE_WIDTH):
            raise ValueError("masks must have shape [N,480,640]")
        count = int(len(masks))
        if not MIN_TRACK_OBSERVATIONS <= count <= MAX_TRACK_OBSERVATIONS:
            raise ValueError("track result observation count is outside [3,5]")
        if len(self.timings) != count:
            raise ValueError("timings must align one-to-one with masks")
        if self.predicted_flags != (False,) + (True,) * (count - 1):
            raise ValueError("only the first mask may be a non-predicted seed")
        if self.committed_flags != (True,) * count:
            raise ValueError("every frozen observation mask must be committed")
        if self.maximum_lookahead_observations != 0:
            raise ValueError("SAM2 video result must have zero lookahead")
        if self.max_state_observations != count or count > MAX_TRACK_OBSERVATIONS:
            raise ValueError("SAM2 video state bound differs")
        if not isinstance(self.cuda_synchronized, bool):
            raise ValueError("cuda_synchronized must be bool")
        if (
            not isinstance(self.peak_allocated_memory_bytes, int)
            or isinstance(self.peak_allocated_memory_bytes, bool)
            or self.peak_allocated_memory_bytes < 0
        ):
            raise ValueError("peak_allocated_memory_bytes must be non-negative int")
        masks = np.frombuffer(masks.tobytes(), dtype=np.bool_).reshape(masks.shape)
        object.__setattr__(self, "masks", masks)


def _validate_track_inputs(
    images_rgb: object,
    frozen_masks: object,
    config: SAM2VideoTrackConfig,
) -> tuple[np.ndarray, np.ndarray]:
    if not isinstance(images_rgb, np.ndarray):
        raise ValueError("images_rgb must be a numpy array")
    if images_rgb.dtype != np.uint8 or images_rgb.ndim != 4:
        raise ValueError("images_rgb must have dtype uint8 and shape [N,480,640,3]")
    count = int(images_rgb.shape[0])
    if images_rgb.shape[1:] != (IMAGE_HEIGHT, IMAGE_WIDTH, 3):
        raise ValueError("images_rgb must have shape [N,480,640,3]")
    if not config.min_track_observations <= count <= config.max_track_observations:
        raise ValueError("track input observation count is outside the frozen bound")
    try:
        masks = np.asarray(frozen_masks)
    except (TypeError, ValueError) as error:
        raise ValueError("frozen_masks must be binary [N,480,640]") from error
    if masks.shape != (count, IMAGE_HEIGHT, IMAGE_WIDTH):
        raise ValueError("frozen_masks must align with images_rgb")
    if not (
        np.issubdtype(masks.dtype, np.bool_)
        or np.issubdtype(masks.dtype, np.integer)
    ) or np.any((masks != 0) & (masks != 1)):
        raise ValueError("frozen_masks must contain exact binary values")
    if np.any(np.count_nonzero(masks, axis=(1, 2)) == 0):
        raise ValueError("every frozen mask must be non-empty")
    return (
        np.array(images_rgb, dtype=np.uint8, order="C", copy=True),
        np.array(masks, dtype=np.bool_, order="C", copy=True),
    )


def _build_production_video_predictor(
    config: SAM2VideoTrackConfig,
) -> tuple[object, object]:
    image_config = config.image_config
    assets = image_provider._verify_production_assets(image_config)
    torch_module, _, _ = image_provider._import_from_verified_root(assets.source_root)
    try:
        build_module = importlib.import_module("sam2.build_sam")
        builder = getattr(build_module, "build_sam2_video_predictor")
        predictor = builder(
            config_file=image_config.config_name,
            ckpt_path=os.fspath(assets.checkpoint_path),
            device=image_config.device,
            mode="eval",
            apply_postprocessing=image_config.apply_postprocessing,
        )
        predictor.eval()
        predictor.requires_grad_(False)
    except Exception as error:
        raise SAM2VideoTrackError(
            "could not construct frozen SAM2 video predictor"
        ) from error
    try:
        if bool(predictor.training) or any(
            bool(parameter.requires_grad) for parameter in predictor.parameters()
        ):
            raise SAM2VideoTrackError("SAM2 video predictor is not frozen")
    except SAM2VideoTrackError:
        raise
    except Exception as error:
        raise SAM2VideoTrackError("could not inspect frozen SAM2 video model") from error
    image_provider._install_verified_production_sdpa_patch(
        source_root=assets.source_root,
        torch_module=torch_module,
    )
    return predictor, torch_module


def _to_mask(value: object, *, threshold: float) -> np.ndarray:
    try:
        detached = value.detach() if hasattr(value, "detach") else value
        host = detached.cpu() if hasattr(detached, "cpu") else detached
        materialized = host.numpy() if hasattr(host, "numpy") else host
        logits = np.asarray(materialized)
    except (AttributeError, TypeError, ValueError, RuntimeError) as error:
        raise SAM2VideoTrackError("could not materialize SAM2 video logits") from error
    while logits.ndim > 2 and logits.shape[0] == 1:
        logits = logits[0]
    if logits.shape != (IMAGE_HEIGHT, IMAGE_WIDTH) or not np.isfinite(logits).all():
        raise SAM2VideoTrackError(
            f"SAM2 video logits must reduce to [480,640], got {logits.shape}"
        )
    return np.asarray(logits > threshold, dtype=np.bool_)


class FrozenSAM2VideoTrackProvider:
    """Lazy frozen short-track provider enforcing query-before-commit."""

    def __init__(
        self,
        *,
        config: SAM2VideoTrackConfig = PRODUCTION_CONFIG,
        predictor_factory: Callable[[SAM2VideoTrackConfig], tuple[object, object]]
        | None = None,
        clock_ns: Callable[[], int] = time.perf_counter_ns,
    ) -> None:
        if (
            config.min_track_observations != MIN_TRACK_OBSERVATIONS
            or config.max_track_observations != MAX_TRACK_OBSERVATIONS
            or config.object_id != OBJECT_ID
            or config.mask_logit_threshold != 0.0
        ):
            raise ValueError("SAM2 video production policy differs")
        self._config = config
        self._factory = predictor_factory or _build_production_video_predictor
        self._clock_ns = clock_ns
        self._predictor: object | None = None
        self._torch: object | None = None

    def _ensure_loaded(self) -> None:
        if self._predictor is None:
            predictor, torch_module = self._factory(self._config)
            self._predictor = predictor
            self._torch = torch_module

    def _sync(self) -> None:
        if self._torch is None:
            return
        cuda = getattr(self._torch, "cuda", None)
        if cuda is not None and bool(cuda.is_available()):
            cuda.synchronize()

    def _context(self) -> ExitStack:
        stack = ExitStack()
        if self._torch is not None:
            inference_mode = getattr(self._torch, "inference_mode", None)
            stack.enter_context(inference_mode() if callable(inference_mode) else nullcontext())
            image_config = self._config.image_config
            if str(image_config.device).startswith("cuda"):
                autocast = getattr(self._torch, "autocast", None)
                dtype = getattr(self._torch, image_config.autocast_dtype, None)
                if callable(autocast) and dtype is not None:
                    stack.enter_context(autocast(device_type="cuda", dtype=dtype))
        return stack

    def predict_track(
        self,
        *,
        images_rgb: np.ndarray,
        frozen_masks: np.ndarray,
    ) -> SAM2VideoTrackResult:
        images, corrections = _validate_track_inputs(
            images_rgb, frozen_masks, self._config
        )
        self._ensure_loaded()
        assert self._predictor is not None
        predictor = self._predictor
        self._sync()
        cuda = getattr(self._torch, "cuda", None) if self._torch is not None else None
        cuda_active = bool(cuda is not None and cuda.is_available())
        if cuda_active:
            cuda.reset_peak_memory_stats()

        outputs: list[np.ndarray] = []
        timings: list[SAM2VideoObservationTiming] = []
        predicted: list[bool] = []
        committed: list[bool] = []
        state: dict[str, object] | None = None
        try:
            with self._context():
                # The verified fork prints one informational line in streaming mode.
                # Suppress only that line; receipts record the mode explicitly.
                with redirect_stdout(io.StringIO()):
                    state = predictor.init_state(
                        video_path=None,
                        offload_video_to_cpu=self._config.offload_video_to_cpu,
                        offload_state_to_cpu=self._config.offload_state_to_cpu,
                    )
                state["video_height"] = IMAGE_HEIGHT
                state["video_width"] = IMAGE_WIDTH

                for index, (image, correction) in enumerate(zip(images, corrections)):
                    start = self._clock_ns()
                    returned_index = int(predictor.add_new_frame(state, image))
                    self._sync()
                    after_add = self._clock_ns()
                    if returned_index != index or int(state["num_frames"]) != index + 1:
                        raise SAM2VideoTrackError("SAM2 streaming frame index differs")

                    if index == 0:
                        outputs.append(np.array(correction, copy=True))
                        predicted.append(False)
                        after_infer = after_add
                    else:
                        inferred_index, object_ids, logits = predictor.infer_single_frame(
                            state, index
                        )
                        self._sync()
                        after_infer = self._clock_ns()
                        if int(inferred_index) != index or list(object_ids) != [OBJECT_ID]:
                            raise SAM2VideoTrackError("SAM2 inferred object identity differs")
                        outputs.append(
                            _to_mask(logits, threshold=self._config.mask_logit_threshold)
                        )
                        predicted.append(True)

                    committed_index, object_ids, _ = predictor.add_new_mask(
                        state,
                        index,
                        OBJECT_ID,
                        correction,
                    )
                    self._sync()
                    after_commit = self._clock_ns()
                    if int(committed_index) != index or list(object_ids) != [OBJECT_ID]:
                        raise SAM2VideoTrackError("SAM2 committed object identity differs")
                    committed.append(True)
                    add_ms = (after_add - start) / 1.0e6
                    infer_ms = (after_infer - after_add) / 1.0e6
                    commit_ms = (after_commit - after_infer) / 1.0e6
                    timings.append(
                        SAM2VideoObservationTiming(
                            add_frame_ms=add_ms,
                            infer_ms=infer_ms,
                            commit_ms=commit_ms,
                            complete_ms=add_ms + infer_ms + commit_ms,
                        )
                    )
        except SAM2VideoTrackError:
            raise
        except Exception as error:
            raise SAM2VideoTrackError("SAM2 causal track inference failed") from error
        finally:
            if state is not None:
                try:
                    predictor.reset_state(state)
                except Exception:
                    pass

        peak = int(cuda.max_memory_allocated()) if cuda_active else 0
        return SAM2VideoTrackResult(
            masks=np.stack(outputs, axis=0),
            timings=tuple(timings),
            predicted_flags=tuple(predicted),
            committed_flags=tuple(committed),
            maximum_lookahead_observations=0,
            max_state_observations=len(outputs),
            cuda_synchronized=cuda_active,
            peak_allocated_memory_bytes=peak,
        )

    def production_receipt(self) -> dict[str, object]:
        config = self._config.image_config
        return {
            "schema": SCHEMA,
            "protocol_id": PROTOCOL_ID,
            "model": "SAM2.1-Hiera-L-video",
            "source_tree_sha256": config.source_tree_sha256,
            "config_sha256": config.config_sha256,
            "checkpoint_sha256": config.checkpoint_sha256,
            "checkpoint_bytes": config.checkpoint_bytes,
            "torch_version": config.torch_version,
            "device": config.device,
            "autocast_dtype": config.autocast_dtype,
            "mask_logit_threshold": self._config.mask_logit_threshold,
            "min_track_observations": self._config.min_track_observations,
            "max_track_observations": self._config.max_track_observations,
            "object_id": self._config.object_id,
            "query_before_commit": True,
            "maximum_lookahead_observations": 0,
            "training": False,
            "online_learning": False,
            "semantics": False,
            "ground_truth": False,
        }
