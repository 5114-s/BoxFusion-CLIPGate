#!/usr/bin/env python3
"""Persistent SAM3 subprocess for the strict live Stream3Dv2-lite path.

The parent passes one end of a Unix ``socketpair`` with ``pass_fds``.  This
process loads SAM3 once, accepts exact uint8 RGB arrays, evaluates the frozen
ScanNet18 text vocabulary, performs class-agnostic mask-IoU deduplication and
returns a compact packbits payload.  It never reads a frame cache, a future
frame, BoxFusion predictions, annotations or evaluator state.
"""

from __future__ import annotations

import argparse
import contextlib
from dataclasses import dataclass
import os
from pathlib import Path
import pickle
import struct
import sys
import time
import traceback
from typing import Any, List, Mapping, Optional, Sequence, Tuple

import numpy as np


TOOL_ROOT = Path(__file__).resolve().parents[1]
if str(TOOL_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOL_ROOT))

from boxfusion.live_sam3_client import (  # noqa: E402
    _PROTOCOL_VERSION,
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

_BYTE_POPCOUNT = np.unpackbits(
    np.arange(256, dtype=np.uint8)[:, None], axis=1
).sum(axis=1, dtype=np.uint16)
_PACKET_HEADER = struct.Struct("!Q")
_MAX_PACKET_BYTES = 128 * 1024 * 1024


def _fd_write_all(fd: int, value: bytes) -> None:
    view = memoryview(value)
    while view:
        written = os.write(fd, view)
        if written <= 0:
            raise EOFError("socket fd closed while sending a packet")
        view = view[written:]


def _fd_send_packet(fd: int, payload: Mapping[str, Any]) -> None:
    body = pickle.dumps(dict(payload), protocol=4)
    if len(body) < 1 or len(body) > _MAX_PACKET_BYTES:
        raise ValueError("worker packet size is invalid")
    _fd_write_all(fd, _PACKET_HEADER.pack(len(body)) + body)


def _fd_read_exact(fd: int, size: int) -> bytes:
    chunks = bytearray()
    while len(chunks) < size:
        chunk = os.read(fd, size - len(chunks))
        if not chunk:
            raise EOFError("socket fd closed while receiving a packet")
        chunks.extend(chunk)
    return bytes(chunks)


def _fd_recv_packet(fd: int) -> Mapping[str, Any]:
    (size,) = _PACKET_HEADER.unpack(_fd_read_exact(fd, _PACKET_HEADER.size))
    if size < 1 or size > _MAX_PACKET_BYTES:
        raise ValueError("worker request packet size is invalid")
    payload = pickle.loads(_fd_read_exact(fd, int(size)))
    if not isinstance(payload, dict):
        raise TypeError("worker request must be a mapping")
    return payload


@dataclass(frozen=True)
class _PackedProposal:
    mask: np.ndarray
    pixels: int
    score: float
    label: str
    box: np.ndarray
    ordinal: int


def _to_numpy(value: Any, name: str) -> np.ndarray:
    if value is None:
        raise ValueError("SAM3 output is missing {}".format(name))
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
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


def _normalize_prompt_output(
    output: Mapping[str, Any],
    label: str,
    image_shape: Tuple[int, int],
    mask_threshold: float,
    min_mask_pixels: int,
    max_per_prompt: int,
    first_ordinal: int,
) -> List[_PackedProposal]:
    if not isinstance(output, Mapping):
        raise TypeError("SAM3 output must be a mapping")
    height, width = image_shape
    boxes = _to_numpy(output.get("boxes"), "boxes").astype(np.float32, copy=False)
    scores = _to_numpy(output.get("scores"), "scores").astype(np.float32, copy=False)
    masks_value = output.get("masks_logits")
    if masks_value is None:
        masks_value = output.get("masks")
    masks = _to_numpy(masks_value, "masks")
    if masks.ndim == 4 and masks.shape[1] == 1:
        masks = masks[:, 0]
    if masks.ndim != 3 or masks.shape[1:] != (height, width):
        raise ValueError("SAM3 masks do not match the input image shape")
    if (
        boxes.ndim != 2
        or boxes.shape[1:] != (4,)
        or scores.ndim != 1
        or boxes.shape[0] != scores.shape[0]
        or masks.shape[0] != scores.shape[0]
    ):
        raise ValueError("SAM3 boxes, scores and masks have inconsistent counts")
    order = np.argsort(-scores, kind="stable")
    proposals: List[_PackedProposal] = []
    for source_index in order:
        score = float(scores[source_index])
        box = np.asarray(boxes[source_index], dtype=np.float32).copy()
        mask_value = masks[source_index]
        mask = (
            np.asarray(mask_value, dtype=np.bool_)
            if np.issubdtype(mask_value.dtype, np.bool_)
            else np.asarray(mask_value >= float(mask_threshold), dtype=np.bool_)
        )
        pixels = int(mask.sum(dtype=np.int64))
        if (
            not np.isfinite(score)
            or score < 0.0
            or score > 1.0
            or not np.isfinite(box).all()
            or pixels < int(min_mask_pixels)
        ):
            continue
        box[0::2] = np.clip(box[0::2], 0.0, float(width))
        box[1::2] = np.clip(box[1::2], 0.0, float(height))
        if box[2] <= box[0] or box[3] <= box[1]:
            continue
        packed = np.packbits(mask.reshape(-1), bitorder="little")
        proposals.append(
            _PackedProposal(
                mask=np.ascontiguousarray(packed, dtype=np.uint8),
                pixels=pixels,
                score=score,
                label=str(label),
                box=box,
                ordinal=first_ordinal + len(proposals),
            )
        )
        if len(proposals) >= int(max_per_prompt):
            break
    return proposals


def _mask_iou(first: _PackedProposal, second: _PackedProposal) -> float:
    # A disjoint tight box proves zero mask intersection and avoids the packed
    # popcount.  For overlapping boxes, full packed intersection preserves the
    # exact cache-builder union definition.
    if (
        min(float(first.box[2]), float(second.box[2]))
        <= max(float(first.box[0]), float(second.box[0]))
        or min(float(first.box[3]), float(second.box[3]))
        <= max(float(first.box[1]), float(second.box[1]))
    ):
        return 0.0
    intersection = int(
        _BYTE_POPCOUNT[np.bitwise_and(first.mask, second.mask)].sum(dtype=np.int64)
    )
    if intersection <= 0:
        return 0.0
    union = first.pixels + second.pixels - intersection
    return float(intersection / union) if union > 0 else 0.0


def _deduplicate(
    proposals: Sequence[_PackedProposal],
    duplicate_mask_iou: float,
    max_proposals: int,
) -> List[_PackedProposal]:
    ordered = sorted(proposals, key=lambda value: (-value.score, value.ordinal))
    kept: List[_PackedProposal] = []
    for proposal in ordered:
        if any(
            _mask_iou(proposal, prior) > float(duplicate_mask_iou)
            for prior in kept
        ):
            continue
        kept.append(proposal)
        if len(kept) >= int(max_proposals):
            break
    return kept


class _Sam3Backend:
    def __init__(self, args: argparse.Namespace) -> None:
        try:
            import torch
            from PIL import Image
        except ImportError as error:
            raise RuntimeError("torch and Pillow are required by SAM3") from error
        sam3_root = Path(args.sam3_root).resolve()
        checkpoint = Path(args.checkpoint).resolve()
        if not sam3_root.is_dir():
            raise FileNotFoundError("missing SAM3 source: {}".format(sam3_root))
        if not checkpoint.is_file():
            raise FileNotFoundError("missing SAM3 checkpoint: {}".format(checkpoint))
        if str(sam3_root) not in sys.path:
            sys.path.insert(0, str(sam3_root))
        try:
            from sam3.model.sam3_image_processor import Sam3Processor
            from sam3.model_builder import build_sam3_image_model
        except ImportError as error:
            raise RuntimeError("could not import SAM3 from {}".format(sam3_root)) from error
        device = torch.device(args.device)
        if device.type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("live SAM3 requires an available CUDA device")
        if args.precision == "bf16" and not torch.cuda.is_bf16_supported():
            raise RuntimeError("selected CUDA device does not support bfloat16")
        model = build_sam3_image_model(
            bpe_path=args.bpe_path,
            device="cuda",
            eval_mode=True,
            checkpoint_path=str(checkpoint),
            load_from_HF=False,
            enable_segmentation=True,
            enable_inst_interactivity=False,
            compile=False,
        )
        self.model = model.to(device).eval()
        self.processor = Sam3Processor(
            self.model,
            resolution=int(args.resolution),
            device=str(device),
            confidence_threshold=float(args.confidence_threshold),
        )
        self.torch = torch
        self.Image = Image
        self.device = device
        self.precision = str(args.precision)
        self.mask_threshold = float(args.mask_threshold)
        self.min_mask_pixels = int(args.min_mask_pixels)
        self.max_per_prompt = int(args.max_per_prompt)
        self.duplicate_mask_iou = float(args.duplicate_mask_iou)
        self.max_proposals = int(args.max_proposals)

    def predict(self, image: np.ndarray) -> Tuple[List[_PackedProposal], float]:
        pil_image = self.Image.fromarray(image, "RGB")
        autocast = (
            self.torch.autocast(device_type="cuda", dtype=self.torch.bfloat16)
            if self.precision == "bf16"
            else contextlib.nullcontext()
        )
        start_event = self.torch.cuda.Event(enable_timing=True)
        end_event = self.torch.cuda.Event(enable_timing=True)
        self.torch.cuda.synchronize(self.device)
        start_event.record()
        proposals: List[_PackedProposal] = []
        ordinal = 0
        with self.torch.inference_mode(), autocast:
            state = self.processor.set_image(pil_image)
            for prompt in SCANNET18_PROMPTS:
                output = self.processor.set_text_prompt(state=state, prompt=prompt)
                prompt_proposals = _normalize_prompt_output(
                    output,
                    prompt,
                    image.shape[:2],
                    self.mask_threshold,
                    self.min_mask_pixels,
                    self.max_per_prompt,
                    ordinal,
                )
                proposals.extend(prompt_proposals)
                ordinal += len(prompt_proposals)
        end_event.record()
        end_event.synchronize()
        gpu_runtime_ms = float(start_event.elapsed_time(end_event))
        kept = _deduplicate(
            proposals,
            self.duplicate_mask_iou,
            self.max_proposals,
        )
        return kept, gpu_runtime_ms


class _FakeBackend:
    """Small deterministic backend used only by the no-GPU protocol tests."""

    def __init__(self, args: argparse.Namespace) -> None:
        self.delay_s = float(args.fake_delay_ms) / 1000.0
        self.duplicate_mask_iou = float(args.duplicate_mask_iou)
        self.max_proposals = int(args.max_proposals)

    @staticmethod
    def _proposal(
        mask: np.ndarray,
        score: float,
        label: str,
        box: Sequence[float],
        ordinal: int,
    ) -> _PackedProposal:
        return _PackedProposal(
            mask=np.packbits(mask.reshape(-1), bitorder="little"),
            pixels=int(mask.sum(dtype=np.int64)),
            score=float(score),
            label=label,
            box=np.asarray(box, dtype=np.float32),
            ordinal=ordinal,
        )

    def predict(self, image: np.ndarray) -> Tuple[List[_PackedProposal], float]:
        if self.delay_s:
            time.sleep(self.delay_s)
        if np.all(image == np.uint8(255)):
            return [], 0.0
        height, width = image.shape[:2]
        left = np.zeros((height, width), dtype=np.bool_)
        right = np.zeros((height, width), dtype=np.bool_)
        left[:, : max(width // 2, 1)] = True
        right[:, min(width // 2, width - 1) :] = True
        proposals = (
            self._proposal(left, 0.91, "chair", (0, 0, max(width // 2, 1), height), 0),
            self._proposal(left, 0.84, "table", (0, 0, max(width // 2, 1), height), 1),
            self._proposal(
                right,
                0.77,
                "cabinet",
                (min(width // 2, width - 1), 0, width, height),
                2,
            ),
        )
        return _deduplicate(proposals, self.duplicate_mask_iou, self.max_proposals), 0.0


def _result_payload(
    request_id: int,
    image_shape: Tuple[int, int],
    proposals: Sequence[_PackedProposal],
    gpu_runtime_ms: float,
    worker_runtime_ms: float,
) -> Mapping[str, Any]:
    if proposals:
        masks = b"".join(bytes(value.mask) for value in proposals)
        boxes = np.stack([value.box for value in proposals]).astype(np.float32).tolist()
    else:
        masks = b""
        boxes = []
    return {
        "type": "result",
        "protocol_version": _PROTOCOL_VERSION,
        "request_id": int(request_id),
        "image_shape": tuple(int(value) for value in image_shape),
        "count": len(proposals),
        "masks_packbits": masks,
        "scores": [float(value.score) for value in proposals],
        "labels": [value.label for value in proposals],
        "boxes": boxes,
        "gpu_runtime_ms": float(gpu_runtime_ms),
        "worker_runtime_ms": float(worker_runtime_ms),
    }


def _run(fd: int, args: argparse.Namespace) -> None:
    load_started = time.perf_counter()
    backend = _FakeBackend(args) if args.backend == "fake" else _Sam3Backend(args)
    _fd_send_packet(
        fd,
        {
            "type": "ready",
            "protocol_version": _PROTOCOL_VERSION,
            "backend": args.backend,
            "model_load_ms": (time.perf_counter() - load_started) * 1000.0,
            "max_pending": 1,
            "max_proposals": int(args.max_proposals),
            "prompts": SCANNET18_PROMPTS,
        },
    )
    while True:
        request = _fd_recv_packet(fd)
        message_type = request.get("type")
        if message_type == "shutdown":
            _fd_send_packet(fd, {"type": "bye", "protocol_version": _PROTOCOL_VERSION})
            return
        request_id = int(request.get("request_id", -1))
        if message_type != "infer":
            _fd_send_packet(
                fd,
                {
                    "type": "error",
                    "request_id": request_id,
                    "error": "unsupported worker message type",
                },
            )
            continue
        try:
            shape = tuple(int(value) for value in request.get("shape", ()))
            rgb_bytes = request.get("rgb")
            if len(shape) != 3 or shape[2] != 3 or min(shape) < 1:
                raise ValueError("request RGB shape is invalid")
            if shape[0] * shape[1] > int(args.max_image_pixels):
                raise ValueError("request RGB exceeds max_image_pixels")
            if not isinstance(rgb_bytes, bytes) or len(rgb_bytes) != int(np.prod(shape)):
                raise ValueError("request RGB byte count is invalid")
            image = np.frombuffer(rgb_bytes, dtype=np.uint8).reshape(shape)
            started = time.perf_counter()
            proposals, gpu_runtime_ms = backend.predict(image)
            worker_runtime_ms = (time.perf_counter() - started) * 1000.0
            _fd_send_packet(
                fd,
                _result_payload(
                    request_id,
                    (shape[0], shape[1]),
                    proposals,
                    gpu_runtime_ms,
                    worker_runtime_ms,
                ),
            )
        except Exception as error:
            _fd_send_packet(
                fd,
                {
                    "type": "error",
                    "request_id": request_id,
                    "error": "{}: {}".format(type(error).__name__, error),
                    "traceback": traceback.format_exc(limit=8),
                },
            )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fd", type=int, required=True)
    parser.add_argument("--backend", choices=("sam3", "fake"), default="sam3")
    parser.add_argument("--sam3-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--bpe-path", default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--precision", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument("--resolution", type=int, default=1008)
    parser.add_argument("--confidence-threshold", type=float, default=0.50)
    parser.add_argument("--mask-threshold", type=float, default=0.50)
    parser.add_argument("--duplicate-mask-iou", type=float, default=0.90)
    parser.add_argument("--min-mask-pixels", type=int, default=100)
    parser.add_argument("--max-per-prompt", type=int, default=10)
    parser.add_argument("--max-proposals", type=int, default=64)
    parser.add_argument("--max-image-pixels", type=int, default=4_194_304)
    parser.add_argument("--fake-delay-ms", type=float, default=0.0)
    return parser


def main() -> int:
    args = _parser().parse_args()
    fd = int(args.fd)
    try:
        if not 1 <= int(args.max_proposals) <= 64:
            raise ValueError("max_proposals must be in [1, 64]")
        _run(fd, args)
        return 0
    except Exception as error:
        try:
            _fd_send_packet(
                fd,
                {
                    "type": "fatal",
                    "protocol_version": _PROTOCOL_VERSION,
                    "error": "{}: {}".format(type(error).__name__, error),
                    "traceback": traceback.format_exc(limit=8),
                },
            )
        except Exception:
            pass
        return 1
    finally:
        try:
            os.close(fd)
        except OSError:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
