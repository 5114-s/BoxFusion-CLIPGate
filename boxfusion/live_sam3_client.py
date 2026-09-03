"""Bounded asynchronous client for the live SAM3 image worker.

The SAM3 environment is intentionally isolated from the BoxFusion runtime.
One persistent child process owns the model and communicates with the parent
through a private ``socketpair`` inherited with ``pass_fds``.  The public
client permits exactly one in-flight RGB frame; callers therefore get a
deterministic drop instead of an unbounded inference queue.

This module has no torch or SAM3 import and is safe to unit-test on CPU.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import os
from pathlib import Path
import pickle
import select
import socket
import struct
import subprocess
import time
from typing import Any, Dict, Mapping, Optional, Tuple

import numpy as np


_PACKET_HEADER = struct.Struct("!Q")
_DEFAULT_MAX_PACKET_BYTES = 128 * 1024 * 1024
_PROTOCOL_VERSION = 1


class LiveSAM3Error(RuntimeError):
    """Base error for the live SAM3 process boundary."""


class LiveSAM3ProtocolError(LiveSAM3Error):
    """The worker violated the local socket protocol."""


class LiveSAM3WorkerError(LiveSAM3Error):
    """SAM3 failed to initialize or infer a frame."""


def _encode_packet(payload: Mapping[str, Any]) -> bytes:
    body = pickle.dumps(dict(payload), protocol=4)
    if len(body) > _DEFAULT_MAX_PACKET_BYTES:
        raise LiveSAM3ProtocolError("packet exceeds the hard size limit")
    return _PACKET_HEADER.pack(len(body)) + body


def _send_packet(sock: socket.socket, payload: Mapping[str, Any]) -> None:
    view = memoryview(_encode_packet(payload))
    while view:
        written = os.write(sock.fileno(), view)
        if written <= 0:
            raise EOFError("socket closed while sending a packet")
        view = view[written:]


def _recv_exact(sock: socket.socket, size: int) -> bytes:
    chunks = bytearray()
    while len(chunks) < size:
        chunk = os.read(sock.fileno(), size - len(chunks))
        if not chunk:
            raise EOFError("socket closed while receiving a packet")
        chunks.extend(chunk)
    return bytes(chunks)


def _recv_packet_blocking(
    sock: socket.socket,
    max_packet_bytes: int = _DEFAULT_MAX_PACKET_BYTES,
) -> Dict[str, Any]:
    header = _recv_exact(sock, _PACKET_HEADER.size)
    (size,) = _PACKET_HEADER.unpack(header)
    if size < 1 or size > int(max_packet_bytes):
        raise LiveSAM3ProtocolError("invalid packet length: {}".format(size))
    payload = pickle.loads(_recv_exact(sock, int(size)))
    if not isinstance(payload, dict):
        raise LiveSAM3ProtocolError("packet payload must be a mapping")
    return payload


def _pop_packet(
    buffer: bytearray,
    max_packet_bytes: int = _DEFAULT_MAX_PACKET_BYTES,
) -> Optional[Dict[str, Any]]:
    if len(buffer) < _PACKET_HEADER.size:
        return None
    (size,) = _PACKET_HEADER.unpack(buffer[: _PACKET_HEADER.size])
    if size < 1 or size > int(max_packet_bytes):
        raise LiveSAM3ProtocolError("invalid packet length: {}".format(size))
    total = _PACKET_HEADER.size + int(size)
    if len(buffer) < total:
        return None
    body = bytes(buffer[_PACKET_HEADER.size : total])
    del buffer[:total]
    payload = pickle.loads(body)
    if not isinstance(payload, dict):
        raise LiveSAM3ProtocolError("packet payload must be a mapping")
    return payload


@dataclass(frozen=True)
class LiveSAM3Config:
    """Configuration for one persistent SAM3 subprocess.

    ``enabled=False`` is a hard off switch: no socket and no process are
    created.  ``worker_backend='fake'`` exists only for CPU protocol tests.
    Production callers should retain the default ``sam3`` backend.
    """

    enabled: bool = False
    python_executable: str = "/home/admin1/miniconda3/envs/sam3/bin/python"
    worker_path: Optional[str] = None
    sam3_root: str = "/data/ZhaoX/Group3D/third_party/sam3"
    checkpoint: str = "/data/ZhaoX/Group3D/checkpoints/sam3/sam3.pt"
    bpe_path: Optional[str] = None
    device: str = "cuda:0"
    precision: str = "bf16"
    resolution: int = 1008
    confidence_threshold: float = 0.50
    mask_threshold: float = 0.50
    duplicate_mask_iou: float = 0.90
    min_mask_pixels: int = 100
    max_per_prompt: int = 10
    max_proposals: int = 64
    max_image_pixels: int = 4_194_304
    startup_timeout_s: float = 300.0
    close_timeout_s: float = 5.0
    late_after_s: Optional[float] = 2.0
    drop_late_results: bool = True
    raise_worker_errors: bool = False
    worker_backend: str = "sam3"
    fake_delay_ms: float = 0.0

    def validate(self) -> None:
        if self.precision not in ("bf16", "fp32"):
            raise ValueError("precision must be 'bf16' or 'fp32'")
        if self.worker_backend not in ("sam3", "fake"):
            raise ValueError("worker_backend must be 'sam3' or 'fake'")
        if not 1 <= int(self.max_proposals) <= 64:
            raise ValueError("max_proposals must be in [1, 64]")
        if int(self.max_per_prompt) < 1 or int(self.min_mask_pixels) < 1:
            raise ValueError("mask proposal limits must be positive")
        if int(self.max_image_pixels) < 1:
            raise ValueError("max_image_pixels must be positive")
        for name, value in (
            ("confidence_threshold", self.confidence_threshold),
            ("mask_threshold", self.mask_threshold),
            ("duplicate_mask_iou", self.duplicate_mask_iou),
        ):
            if not np.isfinite(value) or not 0.0 <= float(value) <= 1.0:
                raise ValueError("{} must be finite and in [0, 1]".format(name))
        for name, value in (
            ("startup_timeout_s", self.startup_timeout_s),
            ("close_timeout_s", self.close_timeout_s),
        ):
            if not np.isfinite(value) or float(value) <= 0.0:
                raise ValueError("{} must be finite and positive".format(name))
        if self.late_after_s is not None and (
            not np.isfinite(self.late_after_s) or float(self.late_after_s) <= 0.0
        ):
            raise ValueError("late_after_s must be positive or None")
        if not np.isfinite(self.fake_delay_ms) or float(self.fake_delay_ms) < 0.0:
            raise ValueError("fake_delay_ms must be finite and non-negative")


@dataclass(frozen=True)
class LiveSAM3Result:
    request_id: int
    context: Mapping[str, Any]
    image_shape: Tuple[int, int]
    masks_packbits: np.ndarray
    scores: np.ndarray
    labels: Tuple[str, ...]
    boxes: np.ndarray
    gpu_runtime_ms: float
    worker_runtime_ms: float
    end_to_end_runtime_ms: float
    late: bool

    @property
    def count(self) -> int:
        return int(self.scores.shape[0])

    def unpack_masks(self) -> np.ndarray:
        """Return a newly allocated ``[N,H,W]`` boolean array."""

        height, width = self.image_shape
        flat = np.unpackbits(
            self.masks_packbits,
            axis=1,
            count=height * width,
            bitorder="little",
        )
        return flat.reshape(self.count, height, width).astype(np.bool_, copy=False)


@dataclass(frozen=True)
class LiveSAM3Stats:
    enabled: bool
    started: bool
    closed: bool
    healthy: bool
    worker_pid: Optional[int]
    queue_depth: int
    max_queue_depth: int
    submitted: int
    completed: int
    delivered: int
    drop_count: int
    dropped_pending: int
    dropped_disabled: int
    dropped_late: int
    late_count: int
    worker_error_count: int
    gpu_runtime_ms_total: float
    worker_runtime_ms_total: float
    end_to_end_runtime_ms_total: float
    last_gpu_runtime_ms: Optional[float]
    last_worker_runtime_ms: Optional[float]
    last_end_to_end_runtime_ms: Optional[float]
    last_error: Optional[str]


@dataclass
class _Pending:
    request_id: int
    context: Dict[str, Any]
    image_shape: Tuple[int, int]
    submitted_at: float


class LiveSAM3Client:
    """Single-pending asynchronous SAM3 client.

    The child is started explicitly with :meth:`start` or lazily by the first
    accepted :meth:`submit`.  ``poll(0)`` never waits for inference; ``drain``
    is the blocking shutdown/test primitive.
    """

    def __init__(self, config: Optional[LiveSAM3Config] = None) -> None:
        self.config = config or LiveSAM3Config()
        self.config.validate()
        self._sock: Optional[socket.socket] = None
        self._process: Optional[subprocess.Popen] = None
        self._recv_buffer = bytearray()
        self._pending: Optional[_Pending] = None
        self._next_request_id = 1
        self._started = False
        self._closed = False
        self._healthy = True
        self._submitted = 0
        self._completed = 0
        self._delivered = 0
        self._dropped_pending = 0
        self._dropped_disabled = 0
        self._dropped_late = 0
        self._late_count = 0
        self._worker_error_count = 0
        self._max_queue_depth = 0
        self._gpu_runtime_ms_total = 0.0
        self._worker_runtime_ms_total = 0.0
        self._end_to_end_runtime_ms_total = 0.0
        self._last_gpu_runtime_ms: Optional[float] = None
        self._last_worker_runtime_ms: Optional[float] = None
        self._last_end_to_end_runtime_ms: Optional[float] = None
        self._last_error: Optional[str] = None

    @property
    def enabled(self) -> bool:
        return bool(self.config.enabled)

    @property
    def pending(self) -> bool:
        return self._pending is not None

    @property
    def worker_pid(self) -> Optional[int]:
        return None if self._process is None else int(self._process.pid)

    def _worker_path(self) -> Path:
        if self.config.worker_path is not None:
            return Path(self.config.worker_path).resolve()
        return (
            Path(__file__).resolve().parents[1]
            / "tools"
            / "run_stream3dv2_live_sam3_worker.py"
        )

    def _command(self, child_fd: int) -> list:
        command = [
            str(self.config.python_executable),
            str(self._worker_path()),
            "--fd",
            str(int(child_fd)),
            "--backend",
            self.config.worker_backend,
            "--sam3-root",
            str(self.config.sam3_root),
            "--checkpoint",
            str(self.config.checkpoint),
            "--device",
            self.config.device,
            "--precision",
            self.config.precision,
            "--resolution",
            str(int(self.config.resolution)),
            "--confidence-threshold",
            repr(float(self.config.confidence_threshold)),
            "--mask-threshold",
            repr(float(self.config.mask_threshold)),
            "--duplicate-mask-iou",
            repr(float(self.config.duplicate_mask_iou)),
            "--min-mask-pixels",
            str(int(self.config.min_mask_pixels)),
            "--max-per-prompt",
            str(int(self.config.max_per_prompt)),
            "--max-proposals",
            str(int(self.config.max_proposals)),
            "--max-image-pixels",
            str(int(self.config.max_image_pixels)),
            "--fake-delay-ms",
            repr(float(self.config.fake_delay_ms)),
        ]
        if self.config.bpe_path is not None:
            command.extend(("--bpe-path", str(self.config.bpe_path)))
        return command

    def start(self) -> bool:
        if self._closed:
            raise LiveSAM3Error("client is closed")
        if not self.enabled:
            return False
        if self._started:
            return True
        worker_path = self._worker_path()
        if not worker_path.is_file():
            raise FileNotFoundError("missing SAM3 worker: {}".format(worker_path))
        parent_sock, child_sock = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            environment = os.environ.copy()
            environment["PYTHONUNBUFFERED"] = "1"
            self._process = subprocess.Popen(
                self._command(child_sock.fileno()),
                pass_fds=(child_sock.fileno(),),
                close_fds=True,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                env=environment,
            )
            child_sock.close()
            child_sock = None
            parent_sock.setblocking(True)
            readable, _, _ = select.select(
                [parent_sock], [], [], float(self.config.startup_timeout_s)
            )
            if not readable:
                raise LiveSAM3WorkerError("timed out while loading the SAM3 worker")
            hello = _recv_packet_blocking(parent_sock)
            if hello.get("type") != "ready":
                message = str(hello.get("error", "worker did not become ready"))
                raise LiveSAM3WorkerError(message)
            if int(hello.get("protocol_version", -1)) != _PROTOCOL_VERSION:
                raise LiveSAM3ProtocolError("worker protocol version mismatch")
            parent_sock.setblocking(False)
            self._sock = parent_sock
            self._started = True
            return True
        except Exception:
            parent_sock.close()
            if child_sock is not None:
                child_sock.close()
            self._stop_process()
            raise

    def submit(
        self,
        rgb: np.ndarray,
        context: Optional[Mapping[str, Any]] = None,
    ) -> Optional[int]:
        """Submit one exact uint8 RGB frame, or deterministically drop it.

        ``None`` means the feature is disabled or the sole pending slot is
        occupied.  Accepted frames return a monotonically increasing request
        identifier.
        """

        if self._closed:
            raise LiveSAM3Error("client is closed")
        if not self.enabled:
            self._dropped_disabled += 1
            return None
        if self._pending is not None:
            self._dropped_pending += 1
            return None
        image = np.asarray(rgb)
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("rgb must have dtype uint8 and shape [H,W,3]")
        height, width = int(image.shape[0]), int(image.shape[1])
        if height < 1 or width < 1 or height * width > int(self.config.max_image_pixels):
            raise ValueError("rgb pixel count is outside the configured bound")
        if not self._started:
            self.start()
        if self._sock is None:
            raise LiveSAM3WorkerError("worker socket is unavailable")
        request_id = self._next_request_id
        self._next_request_id += 1
        clean_context = dict(context or {})
        payload = {
            "type": "infer",
            "protocol_version": _PROTOCOL_VERSION,
            "request_id": request_id,
            "shape": (height, width, 3),
            "rgb": np.ascontiguousarray(image).tobytes(),
            "context": clean_context,
        }
        submitted_at = time.monotonic()
        try:
            self._sock.setblocking(True)
            _send_packet(self._sock, payload)
        except Exception as error:
            self._healthy = False
            self._last_error = "submit failed: {}".format(error)
            raise LiveSAM3WorkerError(self._last_error) from error
        finally:
            if self._sock is not None:
                self._sock.setblocking(False)
        self._pending = _Pending(
            request_id=request_id,
            context=clean_context,
            image_shape=(height, width),
            submitted_at=submitted_at,
        )
        self._submitted += 1
        self._max_queue_depth = max(self._max_queue_depth, 1)
        return request_id

    def _receive_until(self, timeout_s: float) -> Optional[Dict[str, Any]]:
        if self._sock is None:
            return None
        packet = _pop_packet(self._recv_buffer)
        if packet is not None:
            return packet
        timeout_s = max(float(timeout_s), 0.0)
        deadline = time.monotonic() + timeout_s
        first = True
        while True:
            wait_s = max(deadline - time.monotonic(), 0.0)
            if not first and wait_s <= 0.0:
                return None
            first = False
            readable, _, _ = select.select([self._sock], [], [], wait_s)
            if not readable:
                return None
            saw_eof = False
            while True:
                try:
                    chunk = os.read(self._sock.fileno(), 1024 * 1024)
                except BlockingIOError:
                    break
                if not chunk:
                    saw_eof = True
                    break
                self._recv_buffer.extend(chunk)
                packet = _pop_packet(self._recv_buffer)
                if packet is not None:
                    return packet
            if saw_eof:
                raise LiveSAM3WorkerError("SAM3 worker closed its socket")
            if timeout_s == 0.0:
                return None

    def _record_worker_error(self, payload: Mapping[str, Any]) -> None:
        self._worker_error_count += 1
        self._healthy = False
        self._last_error = str(payload.get("error", "unknown SAM3 worker error"))

    def _decode_result(
        self,
        payload: Mapping[str, Any],
        pending: _Pending,
        end_to_end_ms: float,
        late: bool,
    ) -> LiveSAM3Result:
        if int(payload.get("request_id", -1)) != pending.request_id:
            raise LiveSAM3ProtocolError("response request_id mismatch")
        shape = tuple(int(value) for value in payload.get("image_shape", ()))
        if shape != pending.image_shape:
            raise LiveSAM3ProtocolError("response image shape mismatch")
        count = int(payload.get("count", -1))
        if count < 0 or count > int(self.config.max_proposals) or count > 64:
            raise LiveSAM3ProtocolError("response proposal count is invalid")
        packed_width = (shape[0] * shape[1] + 7) // 8
        packed_bytes = payload.get("masks_packbits", b"")
        if not isinstance(packed_bytes, bytes) or len(packed_bytes) != count * packed_width:
            raise LiveSAM3ProtocolError("response mask byte count is invalid")
        masks = np.frombuffer(packed_bytes, dtype=np.uint8).copy().reshape(count, packed_width)
        scores = np.asarray(payload.get("scores", ()), dtype=np.float32)
        boxes = np.asarray(payload.get("boxes", ()), dtype=np.float32)
        if count == 0 and boxes.size == 0:
            boxes = boxes.reshape(0, 4)
        labels = tuple(str(value) for value in payload.get("labels", ()))
        if scores.shape != (count,) or boxes.shape != (count, 4) or len(labels) != count:
            raise LiveSAM3ProtocolError("response proposal arrays are inconsistent")
        if not np.isfinite(scores).all() or not np.isfinite(boxes).all():
            raise LiveSAM3ProtocolError("response contains non-finite values")
        if np.any(scores < 0.0) or np.any(scores > 1.0):
            raise LiveSAM3ProtocolError("response scores are outside [0, 1]")
        if any(not label for label in labels):
            raise LiveSAM3ProtocolError("response contains an empty label")
        gpu_ms = float(payload.get("gpu_runtime_ms", float("nan")))
        worker_ms = float(payload.get("worker_runtime_ms", float("nan")))
        if not np.isfinite(gpu_ms) or gpu_ms < 0.0:
            raise LiveSAM3ProtocolError("invalid synchronized GPU runtime")
        if not np.isfinite(worker_ms) or worker_ms < 0.0:
            raise LiveSAM3ProtocolError("invalid worker runtime")
        masks.setflags(write=False)
        scores.setflags(write=False)
        boxes.setflags(write=False)
        return LiveSAM3Result(
            request_id=pending.request_id,
            context=dict(pending.context),
            image_shape=shape,
            masks_packbits=masks,
            scores=scores,
            labels=labels,
            boxes=boxes,
            gpu_runtime_ms=gpu_ms,
            worker_runtime_ms=worker_ms,
            end_to_end_runtime_ms=float(end_to_end_ms),
            late=bool(late),
        )

    def poll(self, timeout_s: float = 0.0) -> Optional[LiveSAM3Result]:
        """Return the pending result when ready; otherwise return ``None``."""

        if self._pending is None:
            return None
        if not np.isfinite(timeout_s) or float(timeout_s) < 0.0:
            raise ValueError("timeout_s must be finite and non-negative")
        pending = self._pending
        try:
            payload = self._receive_until(float(timeout_s))
        except Exception as error:
            self._pending = None
            self._record_worker_error({"error": str(error)})
            if self.config.raise_worker_errors:
                raise
            return None
        if payload is None:
            return None
        self._pending = None
        elapsed_ms = (time.monotonic() - pending.submitted_at) * 1000.0
        if payload.get("type") == "error":
            self._record_worker_error(payload)
            if self.config.raise_worker_errors:
                raise LiveSAM3WorkerError(self._last_error or "SAM3 inference failed")
            return None
        if payload.get("type") != "result":
            error = LiveSAM3ProtocolError("unexpected worker packet type")
            self._record_worker_error({"error": str(error)})
            if self.config.raise_worker_errors:
                raise error
            return None
        late = (
            self.config.late_after_s is not None
            and elapsed_ms > float(self.config.late_after_s) * 1000.0
        )
        try:
            result = self._decode_result(payload, pending, elapsed_ms, late)
        except Exception as error:
            self._record_worker_error({"error": str(error)})
            if self.config.raise_worker_errors:
                raise
            return None
        self._completed += 1
        self._gpu_runtime_ms_total += result.gpu_runtime_ms
        self._worker_runtime_ms_total += result.worker_runtime_ms
        self._end_to_end_runtime_ms_total += result.end_to_end_runtime_ms
        self._last_gpu_runtime_ms = result.gpu_runtime_ms
        self._last_worker_runtime_ms = result.worker_runtime_ms
        self._last_end_to_end_runtime_ms = result.end_to_end_runtime_ms
        if late:
            self._late_count += 1
            if self.config.drop_late_results:
                self._dropped_late += 1
                return None
        self._delivered += 1
        return result

    def drain(self, timeout_s: Optional[float] = None) -> Optional[LiveSAM3Result]:
        """Wait for the sole pending request, bounded by ``timeout_s``."""

        if self._pending is None:
            return None
        if timeout_s is not None and (
            not np.isfinite(timeout_s) or float(timeout_s) < 0.0
        ):
            raise ValueError("timeout_s must be non-negative or None")
        deadline = None if timeout_s is None else time.monotonic() + float(timeout_s)
        while self._pending is not None:
            wait_s = 0.1
            if deadline is not None:
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    return None
                wait_s = min(wait_s, remaining)
            result = self.poll(wait_s)
            if result is not None:
                return result
        return None

    def snapshot(self) -> LiveSAM3Stats:
        dropped = self._dropped_pending + self._dropped_disabled + self._dropped_late
        return LiveSAM3Stats(
            enabled=self.enabled,
            started=self._started,
            closed=self._closed,
            healthy=self._healthy,
            worker_pid=self.worker_pid,
            queue_depth=int(self._pending is not None),
            max_queue_depth=self._max_queue_depth,
            submitted=self._submitted,
            completed=self._completed,
            delivered=self._delivered,
            drop_count=dropped,
            dropped_pending=self._dropped_pending,
            dropped_disabled=self._dropped_disabled,
            dropped_late=self._dropped_late,
            late_count=self._late_count,
            worker_error_count=self._worker_error_count,
            gpu_runtime_ms_total=self._gpu_runtime_ms_total,
            worker_runtime_ms_total=self._worker_runtime_ms_total,
            end_to_end_runtime_ms_total=self._end_to_end_runtime_ms_total,
            last_gpu_runtime_ms=self._last_gpu_runtime_ms,
            last_worker_runtime_ms=self._last_worker_runtime_ms,
            last_end_to_end_runtime_ms=self._last_end_to_end_runtime_ms,
            last_error=self._last_error,
        )

    def diagnostics(self) -> Dict[str, Any]:
        return asdict(self.snapshot())

    def _stop_process(self) -> None:
        process = self._process
        self._process = None
        if process is None:
            return
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=float(self.config.close_timeout_s))
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=float(self.config.close_timeout_s))

    def close(self, drain: bool = True) -> None:
        if self._closed:
            return
        if drain and self._pending is not None:
            self.drain(float(self.config.close_timeout_s))
        if self._sock is not None and self._pending is None and self._started:
            try:
                self._sock.setblocking(True)
                _send_packet(
                    self._sock,
                    {"type": "shutdown", "protocol_version": _PROTOCOL_VERSION},
                )
                readable, _, _ = select.select(
                    [self._sock], [], [], float(self.config.close_timeout_s)
                )
                reply = (
                    _recv_packet_blocking(self._sock)
                    if readable
                    else {"type": "shutdown_timeout"}
                )
                if reply.get("type") != "bye":
                    self._healthy = False
                    self._last_error = "worker did not acknowledge shutdown"
            except (OSError, EOFError, LiveSAM3Error):
                pass
        if self._sock is not None:
            self._sock.close()
            self._sock = None
        self._stop_process()
        self._pending = None
        self._closed = True

    def __enter__(self) -> "LiveSAM3Client":
        self.start()
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        self.close(drain=exc_type is None)

    def __del__(self) -> None:
        try:
            self.close(drain=False)
        except Exception:
            pass


__all__ = [
    "LiveSAM3Client",
    "LiveSAM3Config",
    "LiveSAM3Error",
    "LiveSAM3ProtocolError",
    "LiveSAM3Result",
    "LiveSAM3Stats",
    "LiveSAM3WorkerError",
]
