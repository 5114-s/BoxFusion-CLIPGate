"""Optional 2D instance proposal and mask providers.

The released BoxFusion path does not import or construct any detector through
this module unless ``supplemental_proposals.enabled`` is explicitly set.  The
module itself is NumPy-only; the optional YOLOE dependency is imported lazily
on the first inference request.

Images passed to providers are RGB ``numpy.ndarray`` objects with shape
``[H, W, 3]``.  Every returned mask is resized to that image's exact ``[H, W]``
shape, which makes cached proposals safe to consume during RGB-D
back-projection.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Mapping,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    Union,
    runtime_checkable,
)

import numpy as np


DEFAULT_SUPPLEMENTAL_PROPOSAL_CONFIG: Dict[str, Any] = {
    "enabled": False,
    "provider": "yoloe",
    "mode": "prompt_free",
    "checkpoint": "yoloe-11s-seg-pf.pt",
    "prompts": (),
    "confidence": 0.25,
    "iou": 0.70,
    "image_size": 640,
    "max_detections": 300,
    "mask_threshold": 0.50,
    "agnostic_nms": True,
    "cache": {
        "enabled": False,
        "directory": None,
        "write": True,
    },
}

_CACHE_FORMAT_VERSION = 1


def _readonly_copy(value: np.ndarray, dtype: Any) -> np.ndarray:
    array = np.array(value, dtype=dtype, copy=True)
    array.setflags(write=False)
    return array


@dataclass(frozen=True)
class SupplementalProposal:
    """One supplemental instance observation in image coordinates.

    ``bbox`` uses ``[x1, y1, x2, y2]`` ordering.  ``mask`` is binary and has
    the full source-image resolution.  ``label`` and ``feature`` may be absent
    for class-agnostic or detector-only providers.
    """

    bbox: np.ndarray
    score: float
    mask: np.ndarray
    label: Optional[str] = None
    feature: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        bbox = np.asarray(self.bbox)
        if bbox.shape != (4,):
            raise ValueError(
                "SupplementalProposal.bbox must have shape (4,), "
                f"received {bbox.shape}"
            )
        if not np.issubdtype(bbox.dtype, np.number):
            raise TypeError("SupplementalProposal.bbox must be numeric")
        if not np.isfinite(bbox).all():
            raise ValueError("SupplementalProposal.bbox must be finite")
        bbox = _readonly_copy(bbox, np.float32)
        if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
            raise ValueError(
                "SupplementalProposal.bbox must have positive width and height"
            )

        score = float(self.score)
        if not np.isfinite(score):
            raise ValueError("SupplementalProposal.score must be finite")
        if score < 0.0 or score > 1.0:
            raise ValueError(
                "SupplementalProposal.score must be in the closed interval [0, 1]"
            )

        mask = np.asarray(self.mask)
        if mask.ndim != 2:
            raise ValueError(
                "SupplementalProposal.mask must have shape (H, W), "
                f"received {mask.shape}"
            )
        if mask.shape[0] < 1 or mask.shape[1] < 1:
            raise ValueError(
                "SupplementalProposal.mask dimensions must be positive"
            )
        if not (
            np.issubdtype(mask.dtype, np.bool_)
            or np.issubdtype(mask.dtype, np.number)
        ):
            raise TypeError(
                "SupplementalProposal.mask must be boolean or numeric"
            )
        if np.issubdtype(mask.dtype, np.number):
            if not np.isfinite(mask).all():
                raise ValueError("SupplementalProposal.mask must be finite")
            unique = np.unique(mask)
            if not np.isin(unique, (0, 1, False, True)).all():
                raise ValueError(
                    "SupplementalProposal.mask must contain only binary values"
                )
        mask = _readonly_copy(mask, np.bool_)

        label = self.label
        if label is not None:
            if not isinstance(label, str):
                raise TypeError(
                    "SupplementalProposal.label must be a string or None"
                )
            label = label.strip()
            if not label:
                raise ValueError(
                    "SupplementalProposal.label cannot be an empty string"
                )

        feature = self.feature
        if feature is not None:
            feature = np.asarray(feature)
            if feature.ndim != 1 or feature.shape[0] < 1:
                raise ValueError(
                    "SupplementalProposal.feature must be a non-empty 1D array"
                )
            if not np.issubdtype(feature.dtype, np.number):
                raise TypeError(
                    "SupplementalProposal.feature must be numeric"
                )
            if not np.isfinite(feature).all():
                raise ValueError(
                    "SupplementalProposal.feature must be finite"
                )
            feature = _readonly_copy(feature, np.float32)

        object.__setattr__(self, "bbox", bbox)
        object.__setattr__(self, "score", score)
        object.__setattr__(self, "mask", mask)
        object.__setattr__(self, "label", label)
        object.__setattr__(self, "feature", feature)


ProposalBatch = List[List[SupplementalProposal]]


@runtime_checkable
class ProposalProvider(Protocol):
    """Interface implemented by supplemental proposal backends."""

    def predict(
        self,
        images: Sequence[np.ndarray],
        *,
        frame_ids: Optional[Sequence[str]] = None,
    ) -> ProposalBatch:
        """Return one proposal list per input image."""


def _validate_image(image: np.ndarray, index: int) -> np.ndarray:
    if not isinstance(image, np.ndarray):
        raise TypeError(
            f"images[{index}] must be a numpy.ndarray, "
            f"received {type(image).__name__}"
        )
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(
            f"images[{index}] must have shape (H, W, 3), "
            f"received {image.shape}"
        )
    if image.shape[0] < 1 or image.shape[1] < 1:
        raise ValueError(f"images[{index}] dimensions must be positive")
    if not (
        np.issubdtype(image.dtype, np.integer)
        or np.issubdtype(image.dtype, np.floating)
    ):
        raise TypeError(f"images[{index}] must have a numeric dtype")
    if np.issubdtype(image.dtype, np.floating) and not np.isfinite(image).all():
        raise ValueError(f"images[{index}] must be finite")
    return image


def _validate_frame_ids(
    frame_ids: Optional[Sequence[str]], count: int
) -> Optional[List[str]]:
    if frame_ids is None:
        return None
    values = list(frame_ids)
    if len(values) != count:
        raise ValueError(
            "frame_ids must contain exactly one identifier per image"
        )
    for index, value in enumerate(values):
        if not isinstance(value, str) or not value.strip():
            raise ValueError(
                f"frame_ids[{index}] must be a non-empty string"
            )
    return [value.strip() for value in values]


class DisabledProposalProvider:
    """No-op provider used whenever the feature is not explicitly enabled."""

    def predict(
        self,
        images: Sequence[np.ndarray],
        *,
        frame_ids: Optional[Sequence[str]] = None,
    ) -> ProposalBatch:
        image_list = list(images)
        _validate_frame_ids(frame_ids, len(image_list))
        return [[] for _ in image_list]


def _as_numpy(value: Any, name: str) -> np.ndarray:
    if value is None:
        raise ValueError(f"YOLOE result is missing {name}")
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def _resize_mask_nearest(mask: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
    output_height, output_width = shape
    if mask.shape == shape:
        return mask
    input_height, input_width = mask.shape
    if input_height < 1 or input_width < 1:
        raise ValueError("YOLOE returned an empty mask dimension")
    y_indices = np.minimum(
        (np.arange(output_height) * input_height // output_height),
        input_height - 1,
    )
    x_indices = np.minimum(
        (np.arange(output_width) * input_width // output_width),
        input_width - 1,
    )
    return mask[np.ix_(y_indices, x_indices)]


def _resolve_label(names: Any, class_index: int) -> Optional[str]:
    if names is None:
        return None
    if isinstance(names, Mapping):
        label = names.get(class_index, names.get(str(class_index)))
    elif isinstance(names, Sequence) and not isinstance(names, (str, bytes)):
        label = names[class_index] if class_index < len(names) else None
    else:
        raise TypeError("YOLOE result.names must be a mapping or sequence")
    if label is None:
        return None
    return str(label)


class YOLOEProposalProvider:
    """Lazy YOLOE adapter supporting text and prompt-free segmentation."""

    def __init__(
        self,
        *,
        checkpoint: Union[str, os.PathLike],
        device: str,
        mode: str = "prompt_free",
        prompts: Sequence[str] = (),
        confidence: float = 0.25,
        iou: float = 0.70,
        image_size: int = 640,
        max_detections: int = 300,
        mask_threshold: float = 0.50,
        agnostic_nms: bool = True,
        model_factory: Optional[Callable[[str], Any]] = None,
    ) -> None:
        checkpoint_value = os.fspath(checkpoint)
        if not checkpoint_value:
            raise ValueError("YOLOE checkpoint must be a non-empty path or ID")
        if not isinstance(device, str) or not device.strip():
            raise ValueError("YOLOE device must be a non-empty string")

        normalized_mode = str(mode).strip().lower().replace("-", "_")
        if normalized_mode not in ("text", "prompt_free"):
            raise ValueError(
                "YOLOE mode must be either 'text' or 'prompt_free'"
            )

        prompt_values = list(prompts)
        for index, prompt in enumerate(prompt_values):
            if not isinstance(prompt, str) or not prompt.strip():
                raise ValueError(
                    f"YOLOE prompts[{index}] must be a non-empty string"
                )
        prompt_values = [prompt.strip() for prompt in prompt_values]
        if normalized_mode == "text" and not prompt_values:
            raise ValueError("YOLOE text mode requires at least one prompt")
        if normalized_mode == "prompt_free" and prompt_values:
            raise ValueError(
                "YOLOE prompt_free mode must not define text prompts"
            )

        confidence_value = float(confidence)
        iou_value = float(iou)
        mask_threshold_value = float(mask_threshold)
        for name, value in (
            ("confidence", confidence_value),
            ("iou", iou_value),
            ("mask_threshold", mask_threshold_value),
        ):
            if not np.isfinite(value) or value < 0.0 or value > 1.0:
                raise ValueError(f"YOLOE {name} must be finite and in [0, 1]")
        image_size_value = int(image_size)
        max_detections_value = int(max_detections)
        if image_size_value < 1:
            raise ValueError("YOLOE image_size must be positive")
        if max_detections_value < 1:
            raise ValueError("YOLOE max_detections must be positive")

        self.checkpoint = checkpoint_value
        self.device = device.strip()
        self.mode = normalized_mode
        self.prompts = tuple(prompt_values)
        self.confidence = confidence_value
        self.iou = iou_value
        self.image_size = image_size_value
        self.max_detections = max_detections_value
        self.mask_threshold = mask_threshold_value
        self.agnostic_nms = bool(agnostic_nms)
        self._model_factory = model_factory
        self._model: Optional[Any] = None

    @staticmethod
    def _import_yoloe() -> Callable[[str], Any]:
        try:
            from ultralytics import YOLOE
        except (ImportError, ModuleNotFoundError) as error:
            raise RuntimeError(
                "The YOLOE supplemental proposal provider is enabled, but "
                "YOLOE is unavailable. Install the official YOLOE-compatible "
                "Ultralytics package in a separate environment or disable "
                "supplemental_proposals."
            ) from error
        return YOLOE

    def _get_model(self) -> Any:
        if self._model is not None:
            return self._model
        factory = self._model_factory or self._import_yoloe()
        model = factory(self.checkpoint)
        if self.mode == "text":
            if not hasattr(model, "set_classes"):
                raise RuntimeError(
                    "Loaded YOLOE model does not support text prompts "
                    "(missing set_classes)"
                )
            embeddings = None
            if hasattr(model, "get_text_pe"):
                embeddings = model.get_text_pe(list(self.prompts))
            if embeddings is None:
                model.set_classes(list(self.prompts))
            else:
                model.set_classes(list(self.prompts), embeddings)
        self._model = model
        return model

    def _extract_result(
        self, result: Any, image_shape: Tuple[int, int]
    ) -> List[SupplementalProposal]:
        boxes_object = getattr(result, "boxes", None)
        if boxes_object is None:
            return []

        boxes = _as_numpy(getattr(boxes_object, "xyxy", None), "boxes.xyxy")
        scores = _as_numpy(getattr(boxes_object, "conf", None), "boxes.conf")
        classes = _as_numpy(getattr(boxes_object, "cls", None), "boxes.cls")
        if boxes.ndim != 2 or boxes.shape[1] != 4:
            raise ValueError(
                "YOLOE boxes.xyxy must have shape (N, 4), "
                f"received {boxes.shape}"
            )
        scores = scores.reshape(-1)
        classes = classes.reshape(-1)
        count = boxes.shape[0]
        if scores.shape != (count,) or classes.shape != (count,):
            raise ValueError(
                "YOLOE boxes, confidence, and class arrays must have "
                "matching lengths"
            )
        if count == 0:
            return []
        if (
            not np.isfinite(boxes).all()
            or not np.isfinite(scores).all()
            or not np.isfinite(classes).all()
        ):
            raise ValueError("YOLOE box outputs must be finite")

        masks_object = getattr(result, "masks", None)
        if masks_object is None:
            raise ValueError(
                "YOLOE segmentation result has boxes but no masks; use a "
                "segmentation checkpoint such as '*-seg.pt'"
            )
        masks = _as_numpy(getattr(masks_object, "data", None), "masks.data")
        if masks.ndim != 3 or masks.shape[0] != count:
            raise ValueError(
                "YOLOE masks.data must have shape (N, H, W) matching boxes, "
                f"received {masks.shape}"
            )
        if not np.isfinite(masks).all():
            raise ValueError("YOLOE masks must be finite")

        image_height, image_width = image_shape
        proposals: List[SupplementalProposal] = []
        names = getattr(result, "names", None)
        for index in range(count):
            class_value = float(classes[index])
            class_index = int(round(class_value))
            if class_index < 0 or abs(class_value - class_index) > 1e-5:
                raise ValueError(
                    "YOLOE class IDs must be non-negative integers"
                )
            bbox = np.asarray(boxes[index], dtype=np.float32).copy()
            bbox[[0, 2]] = np.clip(bbox[[0, 2]], 0.0, float(image_width))
            bbox[[1, 3]] = np.clip(bbox[[1, 3]], 0.0, float(image_height))
            resized_mask = _resize_mask_nearest(
                np.asarray(masks[index]), image_shape
            )
            binary_mask = resized_mask >= self.mask_threshold
            proposals.append(
                SupplementalProposal(
                    bbox=bbox,
                    score=float(scores[index]),
                    mask=binary_mask,
                    label=_resolve_label(names, class_index),
                    feature=None,
                )
            )
        return proposals

    def predict(
        self,
        images: Sequence[np.ndarray],
        *,
        frame_ids: Optional[Sequence[str]] = None,
    ) -> ProposalBatch:
        image_list = list(images)
        _validate_frame_ids(frame_ids, len(image_list))
        validated = [
            _validate_image(image, index)
            for index, image in enumerate(image_list)
        ]
        if not validated:
            return []

        # Ultralytics treats NumPy images as BGR.  The provider contract is RGB.
        model_inputs = [np.ascontiguousarray(image[..., ::-1]) for image in validated]
        model = self._get_model()
        results = model.predict(
            source=model_inputs,
            device=self.device,
            conf=self.confidence,
            iou=self.iou,
            imgsz=self.image_size,
            max_det=self.max_detections,
            agnostic_nms=self.agnostic_nms,
            verbose=False,
        )
        results = list(results)
        if len(results) != len(validated):
            raise ValueError(
                "YOLOE returned a different number of results than input images"
            )
        return [
            self._extract_result(result, image.shape[:2])
            for result, image in zip(results, validated)
        ]


class NpzProposalCache:
    """Safe, pickle-free, atomic NPZ cache for proposal batches."""

    def __init__(
        self,
        directory: Union[str, os.PathLike],
        *,
        write_enabled: bool = True,
    ) -> None:
        directory_value = os.fspath(directory)
        if not directory_value:
            raise ValueError("Proposal cache directory must be non-empty")
        self.directory = Path(directory_value)
        self.write_enabled = bool(write_enabled)

    def path_for_key(self, key: str) -> Path:
        if not isinstance(key, str) or not key:
            raise ValueError("Proposal cache key must be a non-empty string")
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
        return self.directory / f"{digest}.npz"

    def load(
        self,
        key: str,
        *,
        expected_image_shape: Optional[Tuple[int, int]] = None,
    ) -> Optional[List[SupplementalProposal]]:
        path = self.path_for_key(key)
        if not path.is_file():
            return None
        try:
            with np.load(str(path), allow_pickle=False) as archive:
                required = {
                    "format_version",
                    "image_shape",
                    "boxes",
                    "scores",
                    "masks",
                    "labels",
                    "label_present",
                    "feature_values",
                    "feature_offsets",
                }
                missing = required.difference(archive.files)
                if missing:
                    raise ValueError(
                        f"missing arrays: {', '.join(sorted(missing))}"
                    )
                version = int(np.asarray(archive["format_version"]).reshape(()))
                image_shape_array = np.asarray(
                    archive["image_shape"], dtype=np.int64
                ).copy()
                boxes = np.asarray(archive["boxes"], dtype=np.float32).copy()
                scores = np.asarray(archive["scores"], dtype=np.float32).copy()
                masks = np.asarray(archive["masks"]).copy()
                labels = np.asarray(archive["labels"]).copy()
                label_present = np.asarray(
                    archive["label_present"], dtype=np.bool_
                ).copy()
                feature_values = np.asarray(
                    archive["feature_values"], dtype=np.float32
                ).copy()
                feature_offsets = np.asarray(
                    archive["feature_offsets"], dtype=np.int64
                ).copy()
        except (OSError, ValueError, TypeError, KeyError) as error:
            raise ValueError(
                f"Invalid supplemental proposal cache file: {path}"
            ) from error

        if version != _CACHE_FORMAT_VERSION:
            raise ValueError(
                f"Unsupported proposal cache version {version} in {path}"
            )
        if image_shape_array.shape != (2,) or (image_shape_array < 1).any():
            raise ValueError(f"Invalid image shape in proposal cache: {path}")
        image_shape = tuple(int(value) for value in image_shape_array)
        if (
            expected_image_shape is not None
            and tuple(expected_image_shape) != image_shape
        ):
            return None

        count = boxes.shape[0] if boxes.ndim == 2 else -1
        if (
            boxes.shape != (count, 4)
            or scores.shape != (count,)
            or masks.shape != (count,) + image_shape
            or labels.shape != (count,)
            or label_present.shape != (count,)
            or feature_values.ndim != 1
            or feature_offsets.shape != (count + 1,)
            or feature_offsets[0] != 0
            or feature_offsets[-1] != feature_values.shape[0]
            or (np.diff(feature_offsets) < 0).any()
        ):
            raise ValueError(
                f"Inconsistent arrays in proposal cache file: {path}"
            )

        proposals: List[SupplementalProposal] = []
        for index in range(count):
            start = int(feature_offsets[index])
            end = int(feature_offsets[index + 1])
            feature = feature_values[start:end] if end > start else None
            label = str(labels[index]) if label_present[index] else None
            proposals.append(
                SupplementalProposal(
                    bbox=boxes[index],
                    score=float(scores[index]),
                    mask=masks[index],
                    label=label,
                    feature=feature,
                )
            )
        return proposals

    def store(
        self,
        key: str,
        proposals: Sequence[SupplementalProposal],
        *,
        image_shape: Tuple[int, int],
    ) -> bool:
        if not self.write_enabled:
            return False
        path = self.path_for_key(key)
        if (
            len(image_shape) != 2
            or int(image_shape[0]) < 1
            or int(image_shape[1]) < 1
        ):
            raise ValueError("Proposal cache image_shape must be positive (H, W)")
        normalized_shape = (int(image_shape[0]), int(image_shape[1]))
        values = list(proposals)
        for index, proposal in enumerate(values):
            if not isinstance(proposal, SupplementalProposal):
                raise TypeError(
                    f"proposals[{index}] must be a SupplementalProposal"
                )
            if proposal.mask.shape != normalized_shape:
                raise ValueError(
                    f"proposals[{index}].mask shape {proposal.mask.shape} "
                    f"does not match image shape {normalized_shape}"
                )

        count = len(values)
        boxes = np.empty((count, 4), dtype=np.float32)
        scores = np.empty((count,), dtype=np.float32)
        masks = np.empty((count,) + normalized_shape, dtype=np.bool_)
        label_present = np.zeros((count,), dtype=np.bool_)
        label_values: List[str] = []
        feature_parts: List[np.ndarray] = []
        feature_offsets = [0]
        for index, proposal in enumerate(values):
            boxes[index] = proposal.bbox
            scores[index] = proposal.score
            masks[index] = proposal.mask
            label_present[index] = proposal.label is not None
            label_values.append(proposal.label or "")
            if proposal.feature is not None:
                feature_parts.append(np.asarray(proposal.feature, dtype=np.float32))
                feature_offsets.append(
                    feature_offsets[-1] + proposal.feature.shape[0]
                )
            else:
                feature_offsets.append(feature_offsets[-1])

        labels = np.asarray(label_values, dtype=np.str_)
        if feature_parts:
            feature_values = np.concatenate(feature_parts).astype(
                np.float32, copy=False
            )
        else:
            feature_values = np.empty((0,), dtype=np.float32)

        path.parent.mkdir(parents=True, exist_ok=True)
        temp_path: Optional[Path] = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w+b",
                prefix=f".{path.stem}.",
                suffix=".tmp",
                dir=str(path.parent),
                delete=False,
            ) as temporary:
                temp_path = Path(temporary.name)
                np.savez_compressed(
                    temporary,
                    format_version=np.asarray(
                        _CACHE_FORMAT_VERSION, dtype=np.int64
                    ),
                    image_shape=np.asarray(normalized_shape, dtype=np.int64),
                    boxes=boxes,
                    scores=scores,
                    masks=masks,
                    labels=labels,
                    label_present=label_present,
                    feature_values=feature_values,
                    feature_offsets=np.asarray(
                        feature_offsets, dtype=np.int64
                    ),
                )
                temporary.flush()
                os.fsync(temporary.fileno())
            os.replace(str(temp_path), str(path))
        except Exception:
            if temp_path is not None:
                try:
                    temp_path.unlink()
                except FileNotFoundError:
                    pass
            raise
        return True


class CachedProposalProvider:
    """Read-through cache that batches only cache misses."""

    def __init__(
        self,
        provider: ProposalProvider,
        cache: NpzProposalCache,
        *,
        namespace: str = "",
    ) -> None:
        self.provider = provider
        self.cache = cache
        self.namespace = str(namespace)

    def _key(self, frame_id: str, image: np.ndarray) -> str:
        """Bind a logical frame id to the actual RGB payload.

        The content digest prevents a cache entry generated with a different
        ``data.start`` or frame-sampling policy from being silently reused.
        """

        contiguous = np.ascontiguousarray(image)
        digest = hashlib.sha256()
        digest.update(str(contiguous.shape).encode("ascii"))
        digest.update(contiguous.dtype.str.encode("ascii"))
        digest.update(contiguous.tobytes())
        return (
            f"{self.namespace}:{frame_id}:"
            f"{digest.hexdigest()[:20]}"
        )

    def predict(
        self,
        images: Sequence[np.ndarray],
        *,
        frame_ids: Optional[Sequence[str]] = None,
    ) -> ProposalBatch:
        image_list = list(images)
        validated = [
            _validate_image(image, index)
            for index, image in enumerate(image_list)
        ]
        identifiers = _validate_frame_ids(frame_ids, len(validated))
        if identifiers is None:
            return self.provider.predict(validated)

        output: List[Optional[List[SupplementalProposal]]] = [
            None for _ in validated
        ]
        missing_indices: List[int] = []
        for index, (image, frame_id) in enumerate(
            zip(validated, identifiers)
        ):
            cached = self.cache.load(
                self._key(frame_id, image),
                expected_image_shape=image.shape[:2],
            )
            if cached is None:
                missing_indices.append(index)
            else:
                output[index] = cached

        if missing_indices:
            missing_images = [validated[index] for index in missing_indices]
            missing_ids = [identifiers[index] for index in missing_indices]
            predicted = self.provider.predict(
                missing_images, frame_ids=missing_ids
            )
            if len(predicted) != len(missing_indices):
                raise ValueError(
                    "Proposal provider returned a different number of batches "
                    "than requested"
                )
            for index, proposals in zip(missing_indices, predicted):
                output[index] = proposals
                self.cache.store(
                    self._key(identifiers[index], validated[index]),
                    proposals,
                    image_shape=validated[index].shape[:2],
                )

        if any(proposals is None for proposals in output):
            raise RuntimeError("Proposal cache failed to resolve every image")
        return [proposals for proposals in output if proposals is not None]


def resolve_supplemental_proposal_config(
    cfg: Optional[Mapping[str, Any]],
) -> Dict[str, Any]:
    """Resolve either a full BoxFusion config or the provider subsection."""

    if cfg is None:
        raw: Mapping[str, Any] = {}
    elif not isinstance(cfg, Mapping):
        raise TypeError("supplemental proposal config must be a mapping")
    elif "supplemental_proposals" in cfg:
        nested = cfg.get("supplemental_proposals")
        if nested is None:
            raw = {}
        elif not isinstance(nested, Mapping):
            raise TypeError("supplemental_proposals must be a mapping")
        else:
            raw = nested
    else:
        raw = cfg

    resolved = dict(DEFAULT_SUPPLEMENTAL_PROPOSAL_CONFIG)
    resolved["cache"] = dict(DEFAULT_SUPPLEMENTAL_PROPOSAL_CONFIG["cache"])
    for key, value in raw.items():
        if key == "cache":
            if value is None:
                continue
            if not isinstance(value, Mapping):
                raise TypeError("supplemental_proposals.cache must be a mapping")
            resolved["cache"].update(value)
        else:
            resolved[key] = value

    resolved["enabled"] = bool(resolved["enabled"])
    resolved["provider"] = str(resolved["provider"]).strip().lower()
    resolved["mode"] = str(resolved["mode"]).strip().lower().replace("-", "_")
    if "prompts" in raw and "classes" in raw:
        if tuple(raw["prompts"]) != tuple(raw["classes"]):
            raise ValueError(
                "supplemental_proposals.prompts and classes cannot disagree"
            )
    if "prompts" in raw:
        prompts = raw["prompts"]
    elif "classes" in raw:
        prompts = raw["classes"]
    else:
        prompts = resolved["prompts"]
    if isinstance(prompts, (str, bytes)):
        raise TypeError(
            "supplemental_proposals.prompts must be a sequence of strings"
        )
    resolved["prompts"] = tuple(prompts)
    resolved["cache"]["enabled"] = bool(resolved["cache"]["enabled"])
    resolved["cache"]["write"] = bool(resolved["cache"]["write"])
    return resolved


def _provider_namespace(cfg: Mapping[str, Any]) -> str:
    payload = {
        key: cfg[key]
        for key in (
            "provider",
            "mode",
            "checkpoint",
            "prompts",
            "confidence",
            "iou",
            "image_size",
            "max_detections",
            "mask_threshold",
            "agnostic_nms",
        )
    }
    serialized = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), default=str
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:16]


def build_provider(
    cfg: Optional[Mapping[str, Any]], device: str
) -> ProposalProvider:
    """Build a disabled, YOLOE, or cached YOLOE provider.

    Missing configuration is deliberately equivalent to ``enabled: false``.
    Building a YOLOE provider remains dependency-free; the optional dependency
    is imported only when ``predict`` is first called.
    """

    resolved = resolve_supplemental_proposal_config(cfg)
    if not resolved["enabled"]:
        return DisabledProposalProvider()
    if resolved["provider"] != "yoloe":
        raise ValueError(
            "Unsupported supplemental proposal provider "
            f"{resolved['provider']!r}; expected 'yoloe'"
        )

    provider: ProposalProvider = YOLOEProposalProvider(
        checkpoint=resolved["checkpoint"],
        device=device,
        mode=resolved["mode"],
        prompts=resolved["prompts"],
        confidence=resolved["confidence"],
        iou=resolved["iou"],
        image_size=resolved["image_size"],
        max_detections=resolved["max_detections"],
        mask_threshold=resolved["mask_threshold"],
        agnostic_nms=resolved["agnostic_nms"],
    )
    cache_cfg = resolved["cache"]
    if cache_cfg["enabled"]:
        directory = cache_cfg.get("directory")
        if directory is None or not os.fspath(directory):
            raise ValueError(
                "supplemental_proposals.cache.directory is required when "
                "cache is enabled"
            )
        provider = CachedProposalProvider(
            provider,
            NpzProposalCache(
                directory,
                write_enabled=cache_cfg["write"],
            ),
            namespace=_provider_namespace(resolved),
        )
    return provider


__all__ = [
    "CachedProposalProvider",
    "DEFAULT_SUPPLEMENTAL_PROPOSAL_CONFIG",
    "DisabledProposalProvider",
    "NpzProposalCache",
    "ProposalBatch",
    "ProposalProvider",
    "SupplementalProposal",
    "YOLOEProposalProvider",
    "build_provider",
    "resolve_supplemental_proposal_config",
]
