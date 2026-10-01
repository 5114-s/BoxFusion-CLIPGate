#!/usr/bin/env python3
"""Create-only N0a SAM2 image-mask lifting shadow over the sealed F0 extra100.

The runner authenticates the complete extra100 universe before selecting a
shard.  For each successful non-empty current frame it opens only the exact
RGB, depth, pose, and intrinsic paths sealed by F0, invokes one frozen SAM2
box-prompt batch, and passes each selected mask to ``sam2_masklift_n0a``.
Ground truth, native predictions, evaluators, semantics, history, and future
frames are neither arguments nor dependencies of this program.

Scene JSON and NPZ evidence are create-only.  A completed matching scene (or
shard) can be authenticated and reused on resume, but is never overwritten.
Test-only provider and frame-loader injection is available only when the
production 100-scene constants are not requested.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, is_dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import random
import subprocess
import sys
import tempfile
import time
from typing import Any, Callable, Mapping, Sequence
import warnings

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if os.fspath(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, os.fspath(REPOSITORY_ROOT))

PROTOCOL_ID = "N0A-FROZEN-SAM2-IMAGE-BOXPROMPT-MASKLIFT-EXTRA100-SHADOW"
SCENE_SCHEMA = "boxfusion.scannet_sam2_image_masklift_n0a_extra100.scene.v2"
SHARD_SCHEMA = "boxfusion.scannet_sam2_image_masklift_n0a_extra100.shard.v2"
EVIDENCE_SCHEMA = "boxfusion.scannet_sam2_image_masklift_n0a_extra100.evidence.v1"

EXPECTED_F0_SCHEMA = "boxfusion.scannet_fastsam_f0_full200.merge.v1"
EXPECTED_F0_SCENE_SCHEMA = "boxfusion.scannet_fastsam_f0_full200.scene.v1"
EXPECTED_F0_PROTOCOL = "F0-frozen-FastSAM-x-residual-automatic-mask-shadow-full200"

EXPECTED_PROTOCOL_SHA256 = (
    "a0cf925eac638993f7458d6f4debd79b0113553bd5174eb544d4eea9334f307b"
)
EXPECTED_CORE_SHA256 = (
    "80897a977d5694fec6322dadfd94dd6f6fb1bdf9af87b6055b4939df5bf4dced"
)
EXPECTED_PROVIDER_SHA256 = (
    "cbe1ba2cdb0853f49b2ab780c9feb0cea72a9f926700e82882857cc361f6f32e"
)
EXPECTED_F0_RECEIPT_SHA256 = (
    "07249ead31ad150cb43d7a35f4c922ac70a8a2f95bcf0fcd24f61f944c1e58a1"
)
EXPECTED_FULL200_SCENE_LIST_SHA256 = (
    "0e7e722d3e93ec4b721f12293a3f1e98ca62d475b42cc8b9d491878a897e9bd1"
)
EXPECTED_EXTRA100_SCENE_LIST_SHA256 = (
    "f28e6997b2f50799020cf827edfe6a1520b4afe8e17de7c5564004208b8a2287"
)
EXPECTED_EXTRA100_FRAME_LEDGER_SHA256 = (
    "f4fa82ce8a1513262fe10278eed54a33874df00c1cea0964c8afb3945b137818"
)
EXPECTED_EXTRA100_SOURCE_LEDGER_SHA256 = (
    "1f03cc600de29930d3b314588326f35a7f0fcd995ab2700341a2469d8bbbcb00"
)
EXPECTED_EXTRA100_SIDECAR_LEDGER_SHA256 = (
    "0471aa066706ed6ccd17da58bf986fb3d7434d65833c5d01d23dcac976957834"
)

EXPECTED_COHORT_START = 100
EXPECTED_SCENES = 100
EXPECTED_KEYFRAMES = 6_124
EXPECTED_SUCCESSFUL_FRAMES = 5_984
EXPECTED_SOURCES = 46_090
EXPECTED_PROVIDER_FORWARDS = 5_739
EXPECTED_SUCCESSFUL_EMPTY_FRAMES = 245
EXPECTED_AUTHENTICATED_WARNINGS = 11_478
SEALED_CENSUS_KEYS = (
    "keyframe_count",
    "successful_frame_count",
    "source_count",
    "provider_forward_count",
)
TOTAL_COUNT_KEYS = (
    *SEALED_CENSUS_KEYS,
    "valid_hs_count",
    "invalid_hs_count",
    "nontrivial_hs_count",
    "authenticated_warning_count",
)
SCENE_EXCLUDED_RUNTIME_KEYS = (
    "input_pre_rehash_ms",
    "intrinsic_decode_ms",
    "input_end_rehash_ms",
    "evidence_npz_compression_write_ms",
    "scene_json_serialization_write_ms",
)
MAX_SOURCES_PER_FRAME = 16
WARMUP_FORWARD_COUNT = 3
SOURCE_FRAME_STRIDE = 25.0
MASK_PACKED_BYTES = 480 * 640 // 8
MAX_STORED_POINTS = 2_048
MAX_CUDA_BYTES = 4 * 1024**3
WARNING_POLICY_ID = "N0A-WARN-V2-EXACT-2XCUMSUM-POSENC-143-144"
EXPECTED_WARNING_SOURCE_PATH = Path(
    "/data/ZhaoX/OVM3D-Dett/third_party/Grounded-SAM-2/"
    "sam2/modeling/position_encoding.py"
)
EXPECTED_WARNING_SOURCE_RELATIVE_PATH = "sam2/modeling/position_encoding.py"
EXPECTED_WARNING_SOURCE_SHA256 = (
    "14ae89d7ae68f61e2ffcba09eb171d8df9a7298332d4da99036d703294f89ec1"
)
EXPECTED_WARNING_LINES = (143, 144)
EXPECTED_WARNING_MESSAGE = (
    "cumsum_cuda_kernel does not have a deterministic implementation, but you "
    "set 'torch.use_deterministic_algorithms(True, warn_only=True)'. You can "
    "file an issue at https://github.com/pytorch/pytorch/issues to help us "
    "prioritize adding deterministic support for this operation. (Triggered "
    "internally at ../aten/src/ATen/Context.cpp:91.)"
)
EXPECTED_WARNING_MESSAGE_SHA256 = (
    "ed71c50715686ffdf28200dc9deb5f46c8d1f641a112c5050777b9401be90fd8"
)

MIN_VALID_SOURCE_COUNT = 36_872
MIN_VALID_SCENE_COUNT = 90
MIN_NONTRIVIAL_SOURCE_COUNT = 1_440
MIN_NONTRIVIAL_SCENE_COUNT = 50

DEFAULT_F0_RECEIPT = (
    REPOSITORY_ROOT
    / "logs/scannet_fastsam_f0_full200_score05/final/F0_FASTSAM_FULL200.json"
)
DEFAULT_FULL200_SCENE_LIST = (
    REPOSITORY_ROOT / "evaluation/data_util/meta_data/scannetv2_val_f0_full200.txt"
)
DEFAULT_OUTPUT_ROOT = (
    REPOSITORY_ROOT / "logs/scannet_sam2_n0a_extra100_score05_v2_strictwarn"
)
PERMANENTLY_INVALID_V1_OUTPUT_ROOT = (
    REPOSITORY_ROOT / "logs/scannet_sam2_n0a_extra100_score05"
)
PROTOCOL_PATH = (
    REPOSITORY_ROOT / "docs/N0A_SAM2_IMAGE_MASKLIFT_EXTRA100_PROTOCOL_FREEZE.md"
)

CONTRACTS: Mapping[str, bool] = {
    "shadow_only": True,
    "birth_enabled": False,
    "active_authorized": False,
    "native_prediction_access": False,
    "native_output_mutation": False,
    "ground_truth_access": False,
    "gt_access": False,
    "annotation_access": False,
    "evaluator_access": False,
    "future_frame_access": False,
    "history_or_tracking": False,
    "class_clip_or_semantic_use": False,
    "training": False,
    "online_learning": False,
}


class N0ARunnerError(RuntimeError):
    """A sealed input, provider result, output, or frozen gate differed."""


def _resolved_output_root(path: Path) -> Path:
    try:
        return path.expanduser().resolve(strict=False)
    except (OSError, RuntimeError) as error:
        raise N0ARunnerError(f"could not resolve N0a output root: {path}") from error


def _validate_production_output_root(output_root: Path, *, resume: bool) -> Path:
    """Reject the invalid v1 namespace and require a clean --no-resume root."""

    resolved = _resolved_output_root(output_root)
    invalid = _resolved_output_root(PERMANENTLY_INVALID_V1_OUTPUT_ROOT)
    if resolved == invalid or invalid in resolved.parents:
        raise N0ARunnerError(
            "the permanently invalid v1 output root and its descendants are forbidden"
        )
    if not resume:
        if output_root.is_symlink():
            raise N0ARunnerError("production --no-resume output root must not be a symlink")
        if output_root.exists():
            if not output_root.is_dir():
                raise N0ARunnerError(
                    "production --no-resume output root must be absent or an empty directory"
                )
            try:
                first_entry = next(output_root.iterdir(), None)
            except OSError as error:
                raise N0ARunnerError(
                    "could not inspect production --no-resume output root"
                ) from error
            if first_entry is not None:
                raise N0ARunnerError(
                    "production --no-resume requires a new empty v2 output root"
                )
    return resolved


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_json_bytes(value: object, *, sort_keys: bool = True) -> bytes:
    return json.dumps(
        value,
        sort_keys=sort_keys,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _canonical_json_sha256(value: object, *, sort_keys: bool = True) -> str:
    return hashlib.sha256(
        _canonical_json_bytes(value, sort_keys=sort_keys)
    ).hexdigest()


def _valid_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _regular_file(path: Path, label: str, suffix: str | None = None) -> Path:
    if path.is_symlink() or not path.is_file():
        raise N0ARunnerError(f"{label} must be a regular non-symlink file: {path}")
    result = path.resolve()
    if suffix is not None and result.suffix.lower() != suffix:
        raise N0ARunnerError(f"{label} must be a {suffix} file: {result}")
    if result.suffix.lower() in {".pkl", ".pickle"}:
        raise N0ARunnerError(f"prediction pickle input is forbidden: {result}")
    return result


def _warning_policy_source() -> Path:
    """Authenticate the sole source file allowed to emit production warnings."""

    source = _regular_file(
        EXPECTED_WARNING_SOURCE_PATH, "frozen SAM2 warning source", ".py"
    )
    if (
        os.fspath(source) != os.fspath(EXPECTED_WARNING_SOURCE_PATH)
        or _sha256(source) != EXPECTED_WARNING_SOURCE_SHA256
        or hashlib.sha256(EXPECTED_WARNING_MESSAGE.encode("utf-8")).hexdigest()
        != EXPECTED_WARNING_MESSAGE_SHA256
    ):
        raise N0ARunnerError("frozen deterministic-warning policy differs")
    return source


def _warning_policy_receipt(source: Path) -> dict[str, Any]:
    return {
        "policy_id": WARNING_POLICY_ID,
        "expected_count_per_nonempty_forward": 2,
        "category": "builtins.UserWarning",
        "message_type": "builtins.UserWarning",
        "source_path": os.fspath(source),
        "source_relative_path": EXPECTED_WARNING_SOURCE_RELATIVE_PATH,
        "source_sha256": EXPECTED_WARNING_SOURCE_SHA256,
        "ordered_lines": list(EXPECTED_WARNING_LINES),
        "message_sha256": EXPECTED_WARNING_MESSAGE_SHA256,
    }


def _warning_evidence_receipt() -> dict[str, Any]:
    return {
        "policy_id": WARNING_POLICY_ID,
        "count": len(EXPECTED_WARNING_LINES),
        "ordered_lines": list(EXPECTED_WARNING_LINES),
        "source_sha256": EXPECTED_WARNING_SOURCE_SHA256,
        "message_sha256": EXPECTED_WARNING_MESSAGE_SHA256,
    }


def _validate_forward_warnings(
    caught: Sequence[object], *, source: Path
) -> dict[str, Any]:
    """Fail closed unless one forward emitted the exact frozen warning pair."""

    if len(caught) != len(EXPECTED_WARNING_LINES):
        observed = []
        for warning_row in caught:
            message = getattr(warning_row, "message", None)
            message_text = str(message)
            observed.append(
                {
                    "category": getattr(
                        getattr(warning_row, "category", None), "__name__", None
                    ),
                    "message_type": type(message).__name__,
                    "filename": getattr(warning_row, "filename", None),
                    "lineno": getattr(warning_row, "lineno", None),
                    "message_sha256": hashlib.sha256(
                        message_text.encode("utf-8")
                    ).hexdigest(),
                }
            )
        raise N0ARunnerError(
            "each non-empty SAM2 forward must emit exactly two warnings; "
            f"observed_count={len(caught)}; observed="
            f"{json.dumps(observed, sort_keys=True, allow_nan=False)}"
        )
    for warning_row, expected_line in zip(caught, EXPECTED_WARNING_LINES):
        category = getattr(warning_row, "category", None)
        message = getattr(warning_row, "message", None)
        filename = getattr(warning_row, "filename", None)
        lineno = getattr(warning_row, "lineno", None)
        if category is not UserWarning or type(message) is not UserWarning:
            raise N0ARunnerError("SAM2 warning category or message type differs")
        if str(message) != EXPECTED_WARNING_MESSAGE:
            raise N0ARunnerError("SAM2 warning message differs")
        if not isinstance(filename, str) or filename != os.fspath(source):
            raise N0ARunnerError("SAM2 warning source path differs")
        warning_source = _regular_file(
            Path(filename), "captured SAM2 warning source", ".py"
        )
        if warning_source != source or lineno != expected_line:
            raise N0ARunnerError("SAM2 warning source line differs")
    return _warning_evidence_receipt()


def _validate_scene_warning_evidence(
    receipt: Mapping[str, Any], *, warning_policy: Mapping[str, Any]
) -> None:
    """Authenticate the per-forward distribution, not only an aggregate."""

    frames = receipt.get("frames")
    counts = receipt.get("counts")
    if (
        receipt.get("warning_policy") != warning_policy
        or not isinstance(frames, list)
        or not isinstance(counts, Mapping)
    ):
        raise N0ARunnerError("scene warning policy receipt differs")
    forward_count = 0
    warning_count = 0
    expected_evidence = _warning_evidence_receipt()
    for frame in frames:
        if not isinstance(frame, Mapping):
            raise N0ARunnerError("scene warning frame receipt differs")
        invoked = frame.get("provider_invoked") is True
        frame_count = frame.get("authenticated_warning_count")
        expected_count = len(EXPECTED_WARNING_LINES) if invoked else 0
        if frame_count != expected_count:
            raise N0ARunnerError("scene per-frame warning count differs")
        runtime = frame.get("runtime")
        if invoked:
            if (
                not isinstance(runtime, Mapping)
                or runtime.get("deterministic_warning_evidence")
                != expected_evidence
            ):
                raise N0ARunnerError("scene per-forward warning evidence differs")
            forward_count += 1
            warning_count += expected_count
        elif runtime is not None:
            raise N0ARunnerError("non-provider frame unexpectedly has runtime")
    if (
        counts.get("provider_forward_count") != forward_count
        or counts.get("authenticated_warning_count") != warning_count
        or warning_count != 2 * forward_count
    ):
        raise N0ARunnerError("scene warning/forward aggregate differs")


def _read_json(path: Path, label: str) -> tuple[Path, dict[str, Any]]:
    source = _regular_file(path, label, ".json")
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise N0ARunnerError(f"invalid {label}: {source}") from error
    if not isinstance(value, dict):
        raise N0ARunnerError(f"{label} must contain one JSON object")
    return source, value


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
            raise N0ARunnerError(f"refusing to overwrite output: {path}") from error
        return _sha256(path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_create_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            np.savez_compressed(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as error:
            raise N0ARunnerError(f"refusing to overwrite output: {path}") from error
        return _sha256(path)
    finally:
        temporary.unlink(missing_ok=True)


class _EvidenceSpool:
    """Frame-drained offline evidence writer backed by temporary files."""

    def __init__(self, parent: Path) -> None:
        parent.mkdir(parents=True, exist_ok=True)
        self._directory = tempfile.TemporaryDirectory(prefix="n0a-evidence-", dir=parent)
        root = Path(self._directory.name)
        self._mask_path = root / "masks.bin"
        self._points_path = root / "points.bin"
        self._keys_path = root / "keys.bin"
        self._mask_handle = self._mask_path.open("w+b")
        self._points_handle = self._points_path.open("w+b")
        self._keys_handle = self._keys_path.open("w+b")
        self.source_count = 0
        self.point_count = 0

    def __enter__(self) -> "_EvidenceSpool":
        return self

    def __exit__(self, _exc_type, _exc, _traceback) -> None:
        for handle in (self._mask_handle, self._points_handle, self._keys_handle):
            if not handle.closed:
                handle.close()
        self._directory.cleanup()

    def append(
        self, mask_packbits: np.ndarray, points_world: np.ndarray, voxel_keys: np.ndarray
    ) -> tuple[int, int]:
        mask = np.ascontiguousarray(mask_packbits, dtype=np.uint8)
        points = np.ascontiguousarray(points_world, dtype=np.float64)
        keys = np.ascontiguousarray(voxel_keys, dtype=np.int64)
        if mask.shape != (MASK_PACKED_BYTES,):
            raise N0ARunnerError("N0a packed mask shape differs")
        if (
            points.ndim != 2
            or points.shape[1:] != (3,)
            or keys.shape != points.shape
            or len(points) > MAX_STORED_POINTS
        ):
            raise N0ARunnerError("N0a core bounded evidence shape differs")
        start = self.point_count
        stop = start + len(points)
        self._mask_handle.write(mask.tobytes(order="C"))
        self._points_handle.write(points.astype("<f8", copy=False).tobytes(order="C"))
        self._keys_handle.write(keys.astype("<i8", copy=False).tobytes(order="C"))
        self.source_count += 1
        self.point_count = stop
        return start, stop

    def arrays(
        self,
        *,
        point_offsets: Sequence[int],
        frame_ordinals: Sequence[int],
        frame_ids: Sequence[int],
        ranks: Sequence[int],
        raw_indices: Sequence[int],
        selected_indices: Sequence[int],
        selected_ious: Sequence[float],
        all_ious: Sequence[Sequence[float]],
        result_hashes: Sequence[str],
    ) -> dict[str, np.ndarray]:
        source_count = self.source_count
        if not (
            len(point_offsets) == source_count + 1
            and len(frame_ordinals)
            == len(frame_ids)
            == len(ranks)
            == len(raw_indices)
            == len(selected_indices)
            == len(selected_ious)
            == len(all_ious)
            == len(result_hashes)
            == source_count
            and int(point_offsets[-1]) == self.point_count
        ):
            raise N0ARunnerError("N0a evidence arrays have different source counts")
        for handle in (self._mask_handle, self._points_handle, self._keys_handle):
            handle.flush()
            os.fsync(handle.fileno())
        masks = (
            np.memmap(
                self._mask_path,
                mode="r",
                dtype=np.uint8,
                shape=(source_count, MASK_PACKED_BYTES),
            )
            if source_count
            else np.empty((0, MASK_PACKED_BYTES), dtype=np.uint8)
        )
        points = (
            np.memmap(
                self._points_path,
                mode="r",
                dtype="<f8",
                shape=(self.point_count, 3),
            )
            if self.point_count
            else np.empty((0, 3), dtype=np.float64)
        )
        keys = (
            np.memmap(
                self._keys_path,
                mode="r",
                dtype="<i8",
                shape=(self.point_count, 3),
            )
            if self.point_count
            else np.empty((0, 3), dtype=np.int64)
        )
        return {
            "schema_utf8": np.frombuffer(EVIDENCE_SCHEMA.encode("utf-8"), dtype=np.uint8),
            "mask_packbits": masks,
            "points_world": points,
            "voxel_keys": keys,
            "point_offsets": np.asarray(point_offsets, dtype=np.int64),
            "frame_ordinals": np.asarray(frame_ordinals, dtype=np.int64),
            "frame_ids": np.asarray(frame_ids, dtype=np.int64),
            "ranks": np.asarray(ranks, dtype=np.int64),
            "raw_indices": np.asarray(raw_indices, dtype=np.int64),
            "selected_hypothesis_indices": np.asarray(selected_indices, dtype=np.int64),
            "predicted_ious": np.asarray(selected_ious, dtype=np.float32),
            "all_predicted_ious": np.asarray(all_ious, dtype=np.float32).reshape(source_count, 3),
            "result_sha256_ascii": np.asarray(result_hashes, dtype="S64"),
        }


def _jsonable(value: object, label: str) -> Any:
    if is_dataclass(value):
        value = asdict(value)
    elif isinstance(value, Path):
        value = os.fspath(value)
    elif isinstance(value, np.ndarray):
        value = value.tolist()
    elif isinstance(value, np.generic):
        value = value.item()
    elif not isinstance(
        value, (Mapping, list, tuple, str, int, float, bool, type(None))
    ):
        if hasattr(value, "__dict__"):
            value = vars(value)
        else:
            raise N0ARunnerError(f"{label} is not JSON serializable")
    if isinstance(value, Mapping):
        return {
            str(key): _jsonable(item, f"{label}.{key}")
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_jsonable(item, f"{label}[]") for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        raise N0ARunnerError(f"{label} contains a non-finite number")
    return value


def _distribution(values: Sequence[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    if not len(array):
        return {"count": 0, "mean": 0.0, "p50": 0.0, "p95": 0.0, "max": 0.0}
    if not np.isfinite(array).all() or np.any(array < 0.0):
        raise N0ARunnerError("runtime samples must be finite and non-negative")
    return {
        "count": int(len(array)),
        "mean": float(array.mean()),
        "p50": float(np.quantile(array, 0.50)),
        "p95": float(np.quantile(array, 0.95)),
        "max": float(array.max()),
    }


def _number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise N0ARunnerError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise N0ARunnerError(f"{label} must be finite and non-negative")
    return result


def _rehash_reference(value: object, label: str, suffix: str | None = None) -> Path:
    if (
        not isinstance(value, Mapping)
        or not isinstance(value.get("path"), str)
        or not _valid_sha256(value.get("sha256"))
    ):
        raise N0ARunnerError(f"{label} seal is absent")
    path = _regular_file(Path(str(value["path"])), label, suffix)
    if _sha256(path) != value["sha256"]:
        raise N0ARunnerError(f"{label} rehash differs")
    return path


def _seal(path: Path, label: str, suffix: str | None = None) -> dict[str, str]:
    source = _regular_file(path, label, suffix)
    return {"path": os.fspath(source), "sha256": _sha256(source)}


def _load_intrinsic(
    reference: object,
) -> tuple[Path, np.ndarray, float, float]:
    rehash_started = time.perf_counter_ns()
    path = _rehash_reference(reference, "sealed depth intrinsic", ".txt")
    rehash_ms = (time.perf_counter_ns() - rehash_started) / 1.0e6
    decode_started = time.perf_counter_ns()
    try:
        matrix = np.loadtxt(path, dtype=np.float64)
    except (OSError, ValueError) as error:
        raise N0ARunnerError("sealed depth intrinsic cannot be decoded") from error
    if matrix.shape == (4, 4):
        matrix = matrix[:3, :3]
    if (
        matrix.shape != (3, 3)
        or not np.isfinite(matrix).all()
        or matrix[0, 0] <= 0.0
        or matrix[1, 1] <= 0.0
        or not (0.0 <= matrix[0, 2] < 640.0)
        or not (0.0 <= matrix[1, 2] < 480.0)
    ):
        raise N0ARunnerError("sealed depth intrinsic is invalid")
    decode_ms = (time.perf_counter_ns() - decode_started) / 1.0e6
    return path, np.ascontiguousarray(matrix, dtype=np.float64), rehash_ms, decode_ms


def _default_frame_loader(
    rgb_path: Path, depth_path: Path, pose_path: Path
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    try:
        import cv2  # type: ignore
    except ImportError as error:  # pragma: no cover - production dependency
        raise N0ARunnerError("OpenCV is required for N0a production") from error
    bgr = cv2.imread(os.fspath(rgb_path), cv2.IMREAD_COLOR)
    depth_raw = cv2.imread(os.fspath(depth_path), cv2.IMREAD_UNCHANGED)
    if bgr is None or depth_raw is None:
        raise N0ARunnerError("sealed current RGB/depth cannot be decoded")
    if bgr.ndim != 3 or bgr.shape[2] != 3 or bgr.dtype != np.uint8:
        raise N0ARunnerError("sealed RGB must decode as uint8 BGR[H,W,3]")
    if (
        depth_raw.shape != (480, 640)
        or depth_raw.ndim != 2
        or not np.issubdtype(depth_raw.dtype, np.integer)
    ):
        raise N0ARunnerError("sealed ScanNet depth must be uint millimetres[480,640]")
    rgb_native = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb_native, (640, 480), interpolation=cv2.INTER_LINEAR)
    depth_m = depth_raw.astype(np.float64) / 1000.0
    try:
        pose = np.loadtxt(pose_path, dtype=np.float64)
    except (OSError, ValueError) as error:
        raise N0ARunnerError("sealed current pose cannot be decoded") from error
    if pose.shape != (4, 4) or not np.isfinite(pose).all():
        raise N0ARunnerError("sealed current pose must be finite [4,4]")
    return (
        np.ascontiguousarray(rgb, dtype=np.uint8),
        np.ascontiguousarray(depth_m, dtype=np.float64),
        np.ascontiguousarray(pose, dtype=np.float64),
    )


def _source_id(scene_id: str, frame_id: int, raw_index: int) -> str:
    return f"{scene_id}/frame_{frame_id:06d}/raw_{raw_index:03d}"


def _selected_mask_map(funnel: Mapping[str, Any], label: str) -> dict[tuple[Any, ...], Mapping[str, Any]]:
    masks = funnel.get("masks")
    if not isinstance(masks, list):
        raise N0ARunnerError(f"{label} F0 mask diagnostics are absent")
    result: dict[tuple[Any, ...], Mapping[str, Any]] = {}
    for mask in masks:
        if isinstance(mask, Mapping) and mask.get("selected") is True:
            key = (mask.get("rank"), mask.get("raw_index"), mask.get("mask_sha256"))
            if key in result:
                raise N0ARunnerError(f"{label} selected F0 masks are ambiguous")
            result[key] = mask
    return result


def _validate_candidate(
    candidate: object,
    *,
    selected_masks: Mapping[tuple[Any, ...], Mapping[str, Any]],
    expected_rank: int,
    label: str,
) -> Mapping[str, Any]:
    if not isinstance(candidate, Mapping):
        raise N0ARunnerError(f"{label} candidate row is invalid")
    rank = candidate.get("rank")
    raw_index = candidate.get("raw_index")
    mask_sha = candidate.get("mask_sha256")
    points_sha = candidate.get("points_and_voxel_keys_sha256")
    tight_box = candidate.get("tight_box_xyxy")
    if (
        rank != expected_rank
        or isinstance(raw_index, bool)
        or not isinstance(raw_index, int)
        or not _valid_sha256(mask_sha)
        or not _valid_sha256(points_sha)
    ):
        raise N0ARunnerError(f"{label} candidate identity differs")
    mask = selected_masks.get((rank, raw_index, mask_sha))
    if (
        not isinstance(mask, Mapping)
        or mask.get("decision") != "selected"
        or mask.get("tight_box_xyxy") != tight_box
    ):
        raise N0ARunnerError(f"{label} candidate/mask join differs")
    try:
        box = np.asarray(tight_box, dtype=np.float64)
        q02 = np.asarray(candidate.get("world_q02"), dtype=np.float64)
        q98 = np.asarray(candidate.get("world_q98"), dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as error:
        raise N0ARunnerError(f"{label} candidate geometry is invalid") from error
    if (
        box.shape != (4,)
        or not np.isfinite(box).all()
        or box[0] < 0.0
        or box[1] < 0.0
        or box[2] >= 640.0
        or box[3] >= 480.0
        or box[2] <= box[0]
        or box[3] <= box[1]
        or q02.shape != (3,)
        or q98.shape != (3,)
        or not np.isfinite(q02).all()
        or not np.isfinite(q98).all()
        or np.any(q98 - q02 < 0.02 - 1.0e-12)
    ):
        raise N0ARunnerError(f"{label} candidate geometry is invalid")
    return candidate


def _frame_source_rows(
    *,
    scene_id: str,
    scene_index: int,
    frame: Mapping[str, Any],
) -> list[dict[str, Any]]:
    frame_ordinal = frame.get("frame_ordinal")
    frame_id = frame.get("frame_id")
    funnel = frame.get("funnel")
    if not isinstance(frame_ordinal, int) or not isinstance(frame_id, int):
        raise N0ARunnerError(f"{scene_id} frame identity is invalid")
    if not isinstance(funnel, Mapping):
        return []
    candidates = funnel.get("candidates")
    if not isinstance(candidates, list) or len(candidates) > MAX_SOURCES_PER_FRAME:
        raise N0ARunnerError(f"{scene_id}/{frame_id} candidate census differs")
    selected_masks = _selected_mask_map(funnel, f"{scene_id}/{frame_id}")
    if len(selected_masks) != len(candidates):
        raise N0ARunnerError(f"{scene_id}/{frame_id} selected-mask census differs")
    result: list[dict[str, Any]] = []
    for rank, raw in enumerate(candidates):
        candidate = _validate_candidate(
            raw,
            selected_masks=selected_masks,
            expected_rank=rank,
            label=f"{scene_id}/{frame_id}/{rank}",
        )
        raw_index = int(candidate["raw_index"])
        identity = {
            "scene_index": scene_index,
            "scene_id": scene_id,
            "frame_ordinal": frame_ordinal,
            "frame_id": frame_id,
            "rank": rank,
            "raw_index": raw_index,
            "mask_sha256": candidate["mask_sha256"],
            "points_and_voxel_keys_sha256": candidate[
                "points_and_voxel_keys_sha256"
            ],
            "source_id": _source_id(scene_id, frame_id, raw_index),
        }
        h0 = {
            "valid": True,
            "world_q02": candidate["world_q02"],
            "world_q98": candidate["world_q98"],
            "world_center": candidate.get("world_center"),
            "world_extent": candidate.get("world_extent"),
        }
        result.append(
            {
                "identity": identity,
                "h0": h0,
                "tight_box_xyxy": [float(value) for value in candidate["tight_box_xyxy"]],
                "f0_candidate_sha256": _canonical_json_sha256(candidate),
                "f0_mask_diagnostic_sha256": _canonical_json_sha256(
                    selected_masks[(rank, raw_index, candidate["mask_sha256"])]
                ),
            }
        )
    return result


def _load_f0_universe(
    *,
    f0_receipt_path: Path,
    full200_scene_list_path: Path,
    cohort_start: int,
    expected_scene_count: int,
    expected_keyframes: int | None,
    expected_successful_frames: int | None,
    expected_sources: int | None,
) -> tuple[dict[str, Any], dict[str, Any], tuple[str, ...], tuple[dict[str, Any], ...], dict[str, Any]]:
    production = cohort_start == EXPECTED_COHORT_START and expected_scene_count == EXPECTED_SCENES
    receipt_path, receipt = _read_json(f0_receipt_path, "sealed F0 full200 merge")
    receipt_sha = _sha256(receipt_path)
    if production and receipt_sha != EXPECTED_F0_RECEIPT_SHA256:
        raise N0ARunnerError("sealed production F0 receipt SHA-256 differs")
    rows = receipt.get("scenes")
    coverage = receipt.get("coverage")
    if (
        receipt.get("schema") != EXPECTED_F0_SCHEMA
        or receipt.get("protocol_id") != EXPECTED_F0_PROTOCOL
        or receipt.get("complete") is not True
        or receipt.get("overall_pass") is not True
        or not isinstance(rows, list)
        or not isinstance(coverage, Mapping)
    ):
        raise N0ARunnerError("sealed F0 merge contract differs")
    if cohort_start < 0 or expected_scene_count < 1 or cohort_start + expected_scene_count > len(rows):
        raise N0ARunnerError("requested F0 cohort is outside the sealed receipt")

    list_path = _regular_file(full200_scene_list_path, "sealed F0 full200 scene list", ".txt")
    list_sha = _sha256(list_path)
    if production and list_sha != EXPECTED_FULL200_SCENE_LIST_SHA256:
        raise N0ARunnerError("sealed full200 scene-list SHA-256 differs")
    listed_all = tuple(
        line.strip()
        for line in list_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    if len(listed_all) != len(rows) or len(set(listed_all)) != len(listed_all):
        raise N0ARunnerError("sealed full200 scene-list census differs")
    selected_rows = tuple(dict(row) for row in rows[cohort_start : cohort_start + expected_scene_count])
    scenes = listed_all[cohort_start : cohort_start + expected_scene_count]
    if [row.get("scene_id") for row in selected_rows] != list(scenes):
        raise N0ARunnerError("F0 receipt and scene-list order differ")
    if [row.get("scene_index") for row in selected_rows] != list(
        range(cohort_start, cohort_start + expected_scene_count)
    ):
        raise N0ARunnerError("F0 original scene indices differ")
    extra_list_bytes = ("\n".join(scenes) + "\n").encode("utf-8")
    extra_list_sha = hashlib.sha256(extra_list_bytes).hexdigest()
    if production and extra_list_sha != EXPECTED_EXTRA100_SCENE_LIST_SHA256:
        raise N0ARunnerError("derived extra100 scene-list SHA-256 differs")

    frame_ledger: list[list[Any]] = []
    source_ledger: list[list[Any]] = []
    sidecar_ledger: list[list[Any]] = []
    total_keyframes = 0
    total_successful = 0
    total_sources = 0
    total_provider_forwards = 0
    source_ids: set[str] = set()
    scene_census: dict[str, dict[str, int]] = {}
    for expected_index, (scene_id, merged_row) in enumerate(
        zip(scenes, selected_rows), start=cohort_start
    ):
        sidecar_path = _rehash_reference(
            merged_row.get("sidecar"), f"{scene_id} sealed F0 sidecar", ".json"
        )
        _, sidecar = _read_json(sidecar_path, f"{scene_id} sealed F0 sidecar")
        if (
            sidecar.get("schema") != EXPECTED_F0_SCENE_SCHEMA
            or sidecar.get("protocol_id") != EXPECTED_F0_PROTOCOL
            or sidecar.get("complete") is not True
            or sidecar.get("scene_id") != scene_id
            or sidecar.get("scene_index") != expected_index
        ):
            raise N0ARunnerError(f"{scene_id} sealed F0 sidecar contract differs")
        frames = sidecar.get("frames")
        if not isinstance(frames, list):
            raise N0ARunnerError(f"{scene_id} F0 frame ledger is absent")
        scene_successful = 0
        scene_sources = 0
        scene_nonempty = 0
        for ordinal, frame in enumerate(frames):
            if (
                not isinstance(frame, Mapping)
                or frame.get("frame_ordinal") != ordinal
                or not isinstance(frame.get("frame_id"), int)
            ):
                raise N0ARunnerError(f"{scene_id} F0 frame order differs")
            frame_id = int(frame["frame_id"])
            frame_ledger.append([expected_index, scene_id, ordinal, frame_id])
            successful = frame.get("successful") is True
            scene_successful += int(successful)
            sources = _frame_source_rows(
                scene_id=scene_id, scene_index=expected_index, frame=frame
            )
            if sources and not successful:
                raise N0ARunnerError(f"{scene_id}/{frame_id} abstained frame has sources")
            scene_sources += len(sources)
            scene_nonempty += int(bool(sources))
            for source in sources:
                identity = source["identity"]
                source_id = str(identity["source_id"])
                if source_id in source_ids:
                    raise N0ARunnerError("derived N0a source IDs are not unique")
                source_ids.add(source_id)
                source_ledger.append(
                    [
                        expected_index,
                        scene_id,
                        ordinal,
                        frame_id,
                        identity["rank"],
                        identity["raw_index"],
                        identity["mask_sha256"],
                        identity["points_and_voxel_keys_sha256"],
                        [int(value) for value in source["tight_box_xyxy"]],
                    ]
                )
        total_keyframes += len(frames)
        total_successful += scene_successful
        total_sources += scene_sources
        total_provider_forwards += scene_nonempty
        scene_census[scene_id] = {
            "scene_index": expected_index,
            "keyframe_count": len(frames),
            "successful_frame_count": scene_successful,
            "source_count": scene_sources,
            "provider_forward_count": scene_nonempty,
        }
        sidecar_ledger.append(
            [
                expected_index,
                scene_id,
                sidecar_path.name,
                str(merged_row["sidecar"]["sha256"]),
            ]
        )

    census = {
        "scene_count": len(scenes),
        "keyframe_count": total_keyframes,
        "successful_frame_count": total_successful,
        "source_count": total_sources,
        "provider_forward_count": total_provider_forwards,
        "successful_empty_frame_count": total_successful - total_provider_forwards,
    }
    if len(source_ids) != total_sources:
        raise N0ARunnerError("derived N0a source-ID census differs")
    for key, expected in (
        ("keyframe_count", expected_keyframes),
        ("successful_frame_count", expected_successful_frames),
        ("source_count", expected_sources),
    ):
        if expected is not None and census[key] != expected:
            raise N0ARunnerError(f"sealed F0 cohort {key} differs")
    frame_sha = _canonical_json_sha256(frame_ledger, sort_keys=False)
    source_sha = _canonical_json_sha256(source_ledger, sort_keys=False)
    sidecar_sha = _canonical_json_sha256(sidecar_ledger, sort_keys=False)
    if production and (
        census
        != {
            "scene_count": EXPECTED_SCENES,
            "keyframe_count": EXPECTED_KEYFRAMES,
            "successful_frame_count": EXPECTED_SUCCESSFUL_FRAMES,
            "source_count": EXPECTED_SOURCES,
            "provider_forward_count": EXPECTED_PROVIDER_FORWARDS,
            "successful_empty_frame_count": EXPECTED_SUCCESSFUL_EMPTY_FRAMES,
        }
        or frame_sha != EXPECTED_EXTRA100_FRAME_LEDGER_SHA256
        or source_sha != EXPECTED_EXTRA100_SOURCE_LEDGER_SHA256
        or sidecar_sha != EXPECTED_EXTRA100_SIDECAR_LEDGER_SHA256
    ):
        raise N0ARunnerError("sealed production extra100 identity differs")
    seals = {
        "full200_scene_list": {"path": os.fspath(list_path), "sha256": list_sha},
        "derived_cohort_scene_list_sha256": extra_list_sha,
        "frame_ledger_sha256": frame_sha,
        "source_ledger_sha256": source_sha,
        "sidecar_ledger_sha256": sidecar_sha,
        "census": census,
        "scene_census": scene_census,
    }
    return (
        {
            "path": os.fspath(receipt_path),
            "sha256": receipt_sha,
            "run_signature_sha256": receipt.get("run_signature_sha256"),
        },
        seals,
        tuple(scenes),
        selected_rows,
        scene_census,
    )


def _provider_result_arrays(
    result: object, *, source_count: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    def field(name: str, alias: str | None = None) -> object:
        if isinstance(result, Mapping):
            value = result.get(name)
            if value is None and alias is not None:
                value = result.get(alias)
            return value
        value = getattr(result, name, None)
        if value is None and alias is not None:
            value = getattr(result, alias, None)
        return value

    masks = np.asarray(field("masks"))
    selected = np.asarray(
        field("selected_hypothesis_indices", "selected_indices")
    )
    selected_ious = np.asarray(field("predicted_ious"))
    all_ious = np.asarray(field("all_predicted_ious"))
    raw_timing = field("timing")
    if masks.dtype != np.bool_ or masks.shape != (source_count, 480, 640):
        raise N0ARunnerError("SAM2 provider masks must be bool[N,480,640]")
    if selected.shape != (source_count,) or selected.dtype.kind not in "iu":
        raise N0ARunnerError("SAM2 selected indices must be integer[N]")
    selected = np.asarray(selected, dtype=np.int64)
    if np.any((selected < 0) | (selected >= 3)):
        raise N0ARunnerError("SAM2 selected hypothesis is outside [0,3)")
    if selected_ious.shape != (source_count,) or all_ious.shape != (source_count, 3):
        raise N0ARunnerError("SAM2 predicted-IoU shapes differ")
    selected_ious = np.asarray(selected_ious, dtype=np.float32)
    all_ious = np.asarray(all_ious, dtype=np.float32)
    if not np.isfinite(selected_ious).all() or not np.isfinite(all_ious).all():
        raise N0ARunnerError("SAM2 predicted IoUs must be finite")
    expected = all_ious[np.arange(source_count), selected]
    if not np.array_equal(selected_ious, expected):
        raise N0ARunnerError("SAM2 selected IoUs differ from all predicted IoUs")
    if not np.array_equal(selected, np.argmax(all_ious, axis=1)):
        raise N0ARunnerError("SAM2 selection violates max-IoU/lowest-index tie rule")
    timing = _jsonable(raw_timing, "SAM2 provider timing")
    required_timing = {
        "encoder_ms",
        "decoder_and_host_mask_ms",
        "complete_ms",
        "cuda_synchronized",
        "peak_allocated_memory_bytes",
    }
    if not isinstance(timing, Mapping) or set(timing) != required_timing:
        raise N0ARunnerError("SAM2 provider timing contract differs")
    encoder_ms = _number(timing["encoder_ms"], "SAM2 encoder_ms")
    decoder_ms = _number(
        timing["decoder_and_host_mask_ms"], "SAM2 decoder_and_host_mask_ms"
    )
    complete_ms = _number(timing["complete_ms"], "SAM2 complete_ms")
    synchronized = timing["cuda_synchronized"]
    peak_bytes = timing["peak_allocated_memory_bytes"]
    tolerance = max(1.0e-6, abs(complete_ms) * 1.0e-9)
    if (
        not isinstance(synchronized, bool)
        or isinstance(peak_bytes, bool)
        or not isinstance(peak_bytes, int)
        or peak_bytes < 0
        or (not synchronized and peak_bytes != 0)
        or abs(encoder_ms + decoder_ms - complete_ms) > tolerance
    ):
        raise N0ARunnerError("SAM2 provider timing values differ")
    timing = {
        "encoder_ms": encoder_ms,
        "decoder_and_host_mask_ms": decoder_ms,
        "complete_ms": complete_ms,
        "cuda_synchronized": synchronized,
        "peak_allocated_memory_bytes": peak_bytes,
    }
    return (
        np.ascontiguousarray(masks),
        selected,
        selected_ious,
        all_ious,
        timing,
    )


def _aabb_iou(lower_a: np.ndarray, upper_a: np.ndarray, lower_b: np.ndarray, upper_b: np.ndarray) -> float:
    intersection = np.maximum(
        np.minimum(upper_a, upper_b) - np.maximum(lower_a, lower_b), 0.0
    )
    intersection_volume = float(np.prod(intersection))
    volume_a = float(np.prod(upper_a - lower_a))
    volume_b = float(np.prod(upper_b - lower_b))
    union = volume_a + volume_b - intersection_volume
    return intersection_volume / union if union > 0.0 else 0.0


def _is_nontrivial(result: object) -> tuple[bool, float | None, float | None]:
    if getattr(result, "valid", False) is not True:
        return False, None, None
    h0 = getattr(result, "h0")
    hs = getattr(result, "hs")
    h0_q02 = np.asarray(h0.q02, dtype=np.float64)
    h0_q98 = np.asarray(h0.q98, dtype=np.float64)
    hs_q02 = np.asarray(hs.q02, dtype=np.float64)
    hs_q98 = np.asarray(hs.q98, dtype=np.float64)
    iou = _aabb_iou(h0_q02, h0_q98, hs_q02, hs_q98)
    face_displacement = float(
        np.max(np.abs(np.concatenate((h0_q02 - hs_q02, h0_q98 - hs_q98))))
    )
    return iou < 0.90 or face_displacement >= 0.05, iou, face_displacement


def _cuda_peak_allocated_bytes() -> int:
    try:
        import torch

        if torch.cuda.is_available():
            return int(torch.cuda.max_memory_allocated())
    except (ImportError, RuntimeError):
        pass
    return 0


def _source_receipts(
    core_module: object,
    provider_module: object,
    injected_provider: object | None,
    injected_frame_loader: object | None,
) -> dict[str, Any]:
    if _sha256(PROTOCOL_PATH) != EXPECTED_PROTOCOL_SHA256:
        raise N0ARunnerError("N0a frozen protocol SHA-256 differs")
    core_path = _regular_file(Path(str(core_module.__file__)), "N0a mask-lift core")
    provider_path = _regular_file(
        Path(str(provider_module.__file__)), "N0a SAM2 provider"
    )
    if _sha256(core_path) != EXPECTED_CORE_SHA256:
        raise N0ARunnerError("N0a frozen mask-lift core SHA-256 differs")
    if _sha256(provider_path) != EXPECTED_PROVIDER_SHA256:
        raise N0ARunnerError("N0a frozen SAM2 provider SHA-256 differs")
    result: dict[str, Any] = {
        "runner": _seal(Path(__file__).resolve(), "N0a runner source"),
        "protocol": _seal(PROTOCOL_PATH, "N0a protocol"),
        "core": _seal(core_path, "N0a mask-lift core"),
        "provider": _seal(provider_path, "N0a SAM2 provider"),
    }
    if injected_provider is not None:
        result["provider_factory"] = {
            "injected_for_test": True,
            "module": getattr(
                injected_provider, "__module__", type(injected_provider).__module__
            ),
            "qualname": getattr(
                injected_provider,
                "__qualname__",
                type(injected_provider).__qualname__,
            ),
        }
    if injected_frame_loader is not None:
        result["frame_loader"] = {
            "injected_for_test": True,
            "module": getattr(
                injected_frame_loader,
                "__module__",
                type(injected_frame_loader).__module__,
            ),
            "qualname": getattr(
                injected_frame_loader,
                "__qualname__",
                type(injected_frame_loader).__qualname__,
            ),
        }
    return result


def _production_preflight(provider_module: object) -> dict[str, Any]:
    """Freeze runtime policy and return a complete same-device receipt."""

    if os.environ.get("PYTHONHASHSEED") != "0":
        raise N0ARunnerError("production requires external PYTHONHASHSEED=0")
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    try:
        import cv2  # type: ignore
        import hydra  # type: ignore
        import omegaconf  # type: ignore
        import PIL  # type: ignore
        import torch
        import torchvision  # type: ignore
    except ImportError as error:  # pragma: no cover - production environment
        raise N0ARunnerError("N0a production environment dependency is absent") from error

    versions = {
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "torchvision": str(torchvision.__version__),
        "numpy": np.__version__,
        "opencv": str(cv2.__version__),
        "hydra": str(hydra.__version__),
        "omegaconf": str(omegaconf.__version__),
        "pillow": str(PIL.__version__),
    }
    expected_versions = {
        "python": "3.10.19",
        "torch": "2.5.1+cu121",
        "torchvision": "0.20.1+cu121",
        "numpy": "2.2.6",
        "opencv": "4.13.0",
        "hydra": "1.3.2",
        "omegaconf": "2.3.0",
        "pillow": "12.0.0",
    }
    if versions != expected_versions or os.environ.get("CONDA_DEFAULT_ENV") != "gsam2_env":
        raise N0ARunnerError("production must run in the exact frozen gsam2_env")
    if not torch.cuda.is_available():
        raise N0ARunnerError("production N0a requires CUDA")

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    except (AttributeError, RuntimeError) as error:
        raise N0ARunnerError("could not freeze deterministic CUDA policy") from error
    if (
        not torch.are_deterministic_algorithms_enabled()
        or not torch.is_deterministic_algorithms_warn_only_enabled()
        or torch.backends.cudnn.benchmark
        or not torch.backends.cudnn.deterministic
        or torch.backends.cuda.matmul.allow_tf32
        or torch.backends.cudnn.allow_tf32
    ):
        raise N0ARunnerError("deterministic CUDA policy verification failed")

    logical_index = int(torch.cuda.current_device())
    properties = torch.cuda.get_device_properties(logical_index)
    capability = [int(properties.major), int(properties.minor)]
    name = str(properties.name)
    if "RTX 3090" not in name or capability != [8, 6]:
        raise N0ARunnerError("production N0a requires an RTX 3090 (compute 8.6)")
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,uuid,name",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise N0ARunnerError("could not obtain the production GPU UUID") from error
    gpu_rows: list[tuple[int, str, str]] = []
    for line in completed.stdout.splitlines():
        fields = [field.strip() for field in line.split(",", 2)]
        if len(fields) != 3 or not fields[0].isdigit() or not fields[1].startswith("GPU-"):
            raise N0ARunnerError("nvidia-smi GPU identity output differs")
        gpu_rows.append((int(fields[0]), fields[1], fields[2]))
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is None or not visible.strip():
        physical_token = str(logical_index)
    else:
        tokens = [token.strip() for token in visible.split(",")]
        if logical_index >= len(tokens):
            raise N0ARunnerError("CUDA_VISIBLE_DEVICES cannot identify the active GPU")
        physical_token = tokens[logical_index]
    if physical_token.isdigit():
        matches = [row for row in gpu_rows if row[0] == int(physical_token)]
    elif physical_token.startswith("GPU-"):
        matches = [row for row in gpu_rows if row[1] == physical_token]
    else:
        raise N0ARunnerError("MIG/ambiguous CUDA device selection is not frozen for N0a")
    if len(matches) != 1 or "RTX 3090" not in matches[0][2]:
        raise N0ARunnerError("active CUDA device UUID could not be authenticated")

    provider_config = getattr(provider_module, "PRODUCTION_CONFIG", None)
    if provider_config is None:
        raise N0ARunnerError("frozen SAM2 production configuration is absent")
    return {
        "conda_environment": "gsam2_env",
        "versions": versions,
        "gpu": {
            "logical_index": logical_index,
            "physical_index": matches[0][0],
            "uuid": matches[0][1],
            "name": name,
            "compute_capability": capability,
            "total_memory_bytes": int(properties.total_memory),
        },
        "determinism": {
            "seed": 0,
            "pythonhashseed": os.environ["PYTHONHASHSEED"],
            "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"],
            "deterministic_algorithms": True,
            "deterministic_algorithms_warn_only": True,
            "registered_nondeterministic_warning": "aten::cumsum_cuda",
            "warning_policy_id": WARNING_POLICY_ID,
            "expected_warning_count_per_nonempty_forward": 2,
            "bitwise_replay_required": True,
            "cudnn_benchmark": False,
            "cudnn_deterministic": True,
            "cuda_matmul_tf32": False,
            "cudnn_tf32": False,
        },
        "provider_config": _jsonable(provider_config, "SAM2 production config"),
    }


def _resume_scene(
    *,
    output_root: Path,
    scene_id: str,
    scene_index: int,
    run_signature: str,
    f0_sidecar_sha: str,
    warning_policy: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]] | None:
    json_path = output_root / "scenes" / f"{scene_id}.json"
    npz_path = output_root / "scenes" / f"{scene_id}.evidence.npz"
    if not json_path.exists() and not npz_path.exists():
        return None
    if not json_path.is_file() or not npz_path.is_file():
        raise N0ARunnerError(f"partial/orphaned resume output for {scene_id}")
    _, receipt = _read_json(json_path, f"resumed {scene_id} N0a scene")
    evidence = receipt.get("evidence_npz")
    if (
        receipt.get("schema") != SCENE_SCHEMA
        or receipt.get("protocol_id") != PROTOCOL_ID
        or receipt.get("complete") is not True
        or receipt.get("scene_id") != scene_id
        or receipt.get("scene_index") != scene_index
        or receipt.get("run_signature_sha256") != run_signature
        or receipt.get("warning_policy") != warning_policy
        or not isinstance(evidence, Mapping)
        or evidence.get("path") != os.fspath(npz_path.resolve())
        or evidence.get("sha256") != _sha256(npz_path)
        or receipt.get("inputs", {}).get("f0_sidecar", {}).get("sha256") != f0_sidecar_sha
    ):
        raise N0ARunnerError(f"resumed scene authentication differs: {scene_id}")
    content_sha = receipt.get("content_sha256")
    payload = dict(receipt)
    payload.pop("content_sha256", None)
    if content_sha != _canonical_json_sha256(payload):
        raise N0ARunnerError(f"resumed scene content hash differs: {scene_id}")
    _validate_scene_warning_evidence(receipt, warning_policy=warning_policy)
    receipt_runtime = receipt.get("runtime")
    excluded = (
        receipt_runtime.get("excluded_offline_reporting")
        if isinstance(receipt_runtime, Mapping)
        else None
    )
    if not isinstance(excluded, Mapping):
        raise N0ARunnerError(f"resumed scene runtime reporting differs: {scene_id}")
    row = {
        "scene_id": scene_id,
        "scene_index": scene_index,
        "sidecar": {"path": os.fspath(json_path.resolve()), "sha256": _sha256(json_path)},
        "evidence_npz": dict(evidence),
        "counts": receipt["counts"],
        "runtime": receipt["runtime"],
        "source_ids_sha256": receipt["source_ids_sha256"],
        "source_lineage_sha256": receipt["source_lineage_sha256"],
        "excluded_runtime_reporting": {
            **dict(excluded),
            "scene_json_serialization_write_ms": 0.0,
            "scene_json_write_measurement_unavailable_after_partial_resume": True,
        },
        "resumed": True,
    }
    return row, receipt


def _process_scene(
    *,
    merged_row: Mapping[str, Any],
    provider: object,
    lift_mask: Callable[..., object],
    frame_loader: Callable[[Path, Path, Path], tuple[np.ndarray, np.ndarray, np.ndarray]],
    output_root: Path,
    evidence_spool: _EvidenceSpool,
    call_index: list[int],
    run_signature: str,
    warning_source: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    scene_id = merged_row.get("scene_id")
    scene_index = merged_row.get("scene_index")
    if not isinstance(scene_id, str) or not isinstance(scene_index, int):
        raise N0ARunnerError("F0 merged scene identity is invalid")
    sidecar_rehash_started = time.perf_counter_ns()
    sidecar_path = _rehash_reference(
        merged_row.get("sidecar"), f"{scene_id} sealed F0 sidecar", ".json"
    )
    input_pre_rehash_ms = (
        time.perf_counter_ns() - sidecar_rehash_started
    ) / 1.0e6
    _, scene = _read_json(sidecar_path, f"{scene_id} sealed F0 sidecar")
    if (
        scene.get("schema") != EXPECTED_F0_SCENE_SCHEMA
        or scene.get("protocol_id") != EXPECTED_F0_PROTOCOL
        or scene.get("complete") is not True
        or scene.get("scene_id") != scene_id
        or scene.get("scene_index") != scene_index
    ):
        raise N0ARunnerError(f"{scene_id} F0 sidecar contract differs")
    (
        intrinsic_path,
        intrinsic,
        intrinsic_pre_rehash_ms,
        intrinsic_decode_ms,
    ) = _load_intrinsic(scene.get("intrinsic"))
    input_pre_rehash_ms += intrinsic_pre_rehash_ms
    input_seals: list[dict[str, Any]] = [
        {"kind": "f0_sidecar", "path": os.fspath(sidecar_path), "sha256": _sha256(sidecar_path)},
        {"kind": "intrinsic", "path": os.fspath(intrinsic_path), "sha256": _sha256(intrinsic_path)},
    ]
    frames = scene.get("frames")
    if not isinstance(frames, list):
        raise N0ARunnerError(f"{scene_id} F0 frames are absent")

    output_frames: list[dict[str, Any]] = []
    source_ids: list[str] = []
    lineage_hashes: list[str] = []
    point_offsets = [0]
    evidence_frame_ordinals: list[int] = []
    evidence_frame_ids: list[int] = []
    evidence_ranks: list[int] = []
    evidence_raw_indices: list[int] = []
    evidence_selected_indices: list[int] = []
    evidence_selected_ious: list[float] = []
    evidence_all_ious: list[list[float]] = []
    evidence_result_hashes: list[str] = []
    incremental_all: list[float] = []
    incremental_warm: list[float] = []
    composed_all: list[float] = []
    composed_warm: list[float] = []
    successful_count = 0
    provider_forward_count = 0
    valid_count = 0
    nontrivial_count = 0
    provider_peak_bytes = 0
    authenticated_warning_count = 0

    for ordinal, frame in enumerate(frames):
        if (
            not isinstance(frame, Mapping)
            or frame.get("frame_ordinal") != ordinal
            or not isinstance(frame.get("frame_id"), int)
        ):
            raise N0ARunnerError(f"{scene_id} frame order differs")
        frame_id = int(frame["frame_id"])
        successful = frame.get("successful") is True
        sources = _frame_source_rows(scene_id=scene_id, scene_index=scene_index, frame=frame)
        if not successful:
            if sources:
                raise N0ARunnerError(f"{scene_id}/{frame_id} abstained frame has sources")
            output_frames.append(
                {
                    "frame_ordinal": ordinal,
                    "frame_id": frame_id,
                    "successful": False,
                    "abstention": _jsonable(frame.get("abstention"), "F0 abstention"),
                    "current_only": True,
                    "provider_invoked": False,
                    "authenticated_warning_count": 0,
                    "sources": [],
                    "runtime": None,
                }
            )
            continue
        successful_count += 1
        if not sources:
            output_frames.append(
                {
                    "frame_ordinal": ordinal,
                    "frame_id": frame_id,
                    "successful": True,
                    "abstention": None,
                    "current_only": True,
                    "max_accessed_frame_ordinal": ordinal,
                    "provider_invoked": False,
                    "authenticated_warning_count": 0,
                    "sources": [],
                    "runtime": None,
                }
            )
            continue

        inputs = frame.get("inputs")
        if not isinstance(inputs, Mapping):
            raise N0ARunnerError(f"{scene_id}/{frame_id} sealed inputs are absent")
        if (
            inputs.get("current_pose_valid") is not True
            or inputs.get("f0_pose_forward_filled") is not False
            or inputs.get("producer_orientation") != 0
            or inputs.get("producer_rotation_k") != 0
            or inputs.get("producer_depth_shape") != [480, 640]
            or inputs.get("producer_image_shape") != [480, 640, 3]
        ):
            raise N0ARunnerError(f"{scene_id}/{frame_id} current-frame contract differs")
        rgb_path = _regular_file(Path(str(inputs.get("rgb_path", ""))), "sealed RGB")
        depth_path = _regular_file(Path(str(inputs.get("depth_path", ""))), "sealed depth", ".png")
        pose_path = _regular_file(Path(str(inputs.get("pose_path", ""))), "sealed pose", ".txt")
        frame_rehash_started = time.perf_counter_ns()
        for kind, path, expected_sha in (
            ("rgb", rgb_path, inputs.get("rgb_sha256")),
            ("depth", depth_path, inputs.get("depth_sha256")),
            ("pose", pose_path, inputs.get("pose_sha256")),
        ):
            if not _valid_sha256(expected_sha) or _sha256(path) != expected_sha:
                raise N0ARunnerError(f"{scene_id}/{frame_id} sealed {kind} rehash differs")
            input_seals.append(
                {
                    "kind": kind,
                    "frame_ordinal": ordinal,
                    "frame_id": frame_id,
                    "path": os.fspath(path),
                    "sha256": expected_sha,
                }
            )
        input_pre_rehash_ms += (
            time.perf_counter_ns() - frame_rehash_started
        ) / 1.0e6

        decode_started = time.perf_counter_ns()
        rgb, depth_m, camera_to_world = frame_loader(rgb_path, depth_path, pose_path)
        decode_ms = (time.perf_counter_ns() - decode_started) / 1.0e6
        if rgb.shape != (480, 640, 3) or rgb.dtype != np.uint8:
            raise N0ARunnerError("frame loader must return uint8 RGB[480,640,3]")
        if depth_m.shape != (480, 640) or depth_m.dtype.kind not in "f":
            raise N0ARunnerError("frame loader must return metric floating depth[480,640]")
        if camera_to_world.shape != (4, 4) or not np.isfinite(camera_to_world).all():
            raise N0ARunnerError("frame loader must return a finite [4,4] pose")
        boxes = np.asarray([source["tight_box_xyxy"] for source in sources], dtype=np.float32)
        infer = getattr(provider, "predict", None)
        if not callable(infer):
            infer = provider if callable(provider) else None
        if not callable(infer):
            raise N0ARunnerError("SAM2 provider lacks predict(image_rgb, boxes_xyxy)")
        provider_started = time.perf_counter_ns()
        with warnings.catch_warnings(record=True) as caught_warnings:
            warnings.simplefilter("always")
            provider_result = infer(
                np.ascontiguousarray(rgb), np.ascontiguousarray(boxes)
            )
        provider_ms = (time.perf_counter_ns() - provider_started) / 1.0e6
        warning_evidence = _validate_forward_warnings(
            caught_warnings, source=warning_source
        )
        frame_authenticated_warning_count = int(warning_evidence["count"])
        authenticated_warning_count += frame_authenticated_warning_count
        (
            masks,
            selected_indices,
            selected_ious,
            all_ious,
            provider_timing,
        ) = _provider_result_arrays(provider_result, source_count=len(sources))
        provider_peak_bytes = max(
            provider_peak_bytes,
            int(provider_timing["peak_allocated_memory_bytes"]),
        )
        lift_ms = 0.0
        offline_evidence_ms = 0.0
        output_sources: list[dict[str, Any]] = []
        frame_valid = 0
        frame_nontrivial = 0
        for row_index, source in enumerate(sources):
            identity = source["identity"]
            source_lift_started = time.perf_counter_ns()
            result = lift_mask(
                f0_source_identity=identity,
                selected_mask=masks[row_index],
                depth_m=np.ascontiguousarray(depth_m, dtype=np.float64),
                intrinsics=intrinsic,
                camera_to_world=np.ascontiguousarray(camera_to_world, dtype=np.float64),
                h0=source["h0"],
            )
            lift_ms += (time.perf_counter_ns() - source_lift_started) / 1.0e6
            offline_started = time.perf_counter_ns()
            if getattr(result, "source_id", None) != identity["source_id"]:
                raise N0ARunnerError("N0a core changed the F0 source identity")
            receipt = result.to_receipt()
            nontrivial, h0_hs_iou, face_displacement = _is_nontrivial(result)
            evidence_index = evidence_spool.source_count
            points = np.asarray(result.points_world, dtype=np.float64)
            keys = np.asarray(result.voxel_keys, dtype=np.int64)
            start, stop = evidence_spool.append(
                np.asarray(result.mask_packbits, dtype=np.uint8), points, keys
            )
            point_offsets.append(stop)
            evidence_frame_ordinals.append(ordinal)
            evidence_frame_ids.append(frame_id)
            evidence_ranks.append(int(identity["rank"]))
            evidence_raw_indices.append(int(identity["raw_index"]))
            evidence_selected_indices.append(int(selected_indices[row_index]))
            evidence_selected_ious.append(float(selected_ious[row_index]))
            evidence_all_ious.append([float(value) for value in all_ious[row_index]])
            evidence_result_hashes.append(str(result.result_sha256))
            lineage = _canonical_json_sha256(
                {
                    "identity": identity,
                    "f0_candidate_sha256": source["f0_candidate_sha256"],
                    "f0_mask_diagnostic_sha256": source["f0_mask_diagnostic_sha256"],
                    "n0a_result_sha256": result.result_sha256,
                    "sam2_selected_hypothesis_index": int(selected_indices[row_index]),
                    "sam2_all_predicted_ious": evidence_all_ious[-1],
                }
            )
            source_ids.append(identity["source_id"])
            lineage_hashes.append(lineage)
            frame_valid += int(bool(result.valid))
            frame_nontrivial += int(nontrivial)
            output_sources.append(
                {
                    **identity,
                    "prompt_tight_box_xyxy": source["tight_box_xyxy"],
                    "sam2": {
                        "selected_hypothesis_index": int(selected_indices[row_index]),
                        "predicted_iou": float(selected_ious[row_index]),
                        "all_predicted_ious": evidence_all_ious[-1],
                        "used_for_detection_score": False,
                    },
                    "n0a_receipt": receipt,
                    "nontrivial_vs_h0": nontrivial,
                    "h0_hs_iou3d": h0_hs_iou,
                    "maximum_face_displacement_m": face_displacement,
                    "evidence_index": evidence_index,
                    "point_offset": [start, stop],
                    "source_lineage_sha256": lineage,
                }
            )
            offline_evidence_ms += (
                time.perf_counter_ns() - offline_started
            ) / 1.0e6
        incremental_ms = decode_ms + provider_ms + lift_ms
        inherited_runtime = frame.get("runtime")
        if not isinstance(inherited_runtime, Mapping):
            raise N0ARunnerError(f"{scene_id}/{frame_id} sealed F0 runtime is absent")
        inherited_complete_ms = _number(
            inherited_runtime.get("complete_ms"), "sealed F0 complete_ms"
        )
        composed_ms = inherited_complete_ms + incremental_ms
        current_call = call_index[0]
        warmup_excluded = current_call < WARMUP_FORWARD_COUNT
        cold_provider_load_and_first_forward_ms = (
            provider_ms if current_call == 0 else 0.0
        )
        call_index[0] += 1
        provider_forward_count += 1
        valid_count += frame_valid
        nontrivial_count += frame_nontrivial
        incremental_all.append(incremental_ms)
        composed_all.append(composed_ms)
        if not warmup_excluded:
            incremental_warm.append(incremental_ms)
            composed_warm.append(composed_ms)
        output_frames.append(
            {
                "frame_ordinal": ordinal,
                "frame_id": frame_id,
                "successful": True,
                "abstention": None,
                "current_only": True,
                "max_accessed_frame_ordinal": ordinal,
                "provider_invoked": True,
                "provider_forward_count": 1,
                "authenticated_warning_count": frame_authenticated_warning_count,
                "input": {
                    "rgb": {"path": os.fspath(rgb_path), "sha256": inputs["rgb_sha256"]},
                    "depth": {"path": os.fspath(depth_path), "sha256": inputs["depth_sha256"]},
                    "pose": {"path": os.fspath(pose_path), "sha256": inputs["pose_sha256"]},
                    "intrinsic": {"path": os.fspath(intrinsic_path), "sha256": _sha256(intrinsic_path)},
                    "rgb_color_order": "RGB_after_exact_BGR_to_RGB",
                    "box_source": "sealed_F0_candidate.tight_box_xyxy",
                },
                "sources": output_sources,
                "runtime": {
                    "provider_call_index_in_shard": current_call,
                    "n0a_warmup_excluded": warmup_excluded,
                    "decode_ms": decode_ms,
                    "sam2_provider_ms": provider_ms,
                    "cold_provider_load_and_first_forward_ms": (
                        cold_provider_load_and_first_forward_ms
                    ),
                    "cold_provider_metric_includes_first_forward": current_call == 0,
                    "sam2_provider_timing": provider_timing,
                    "deterministic_warning_evidence": warning_evidence,
                    "masklift_ms": lift_ms,
                    "n0a_incremental_ms": incremental_ms,
                    "offline_evidence_buffer_ms_excluded": offline_evidence_ms,
                    "sealed_f0_complete_ms": inherited_complete_ms,
                    "replay_composed_ms": composed_ms,
                    "replay_composed_ms_per_source_frame": composed_ms / SOURCE_FRAME_STRIDE,
                    "gap25_deadline_missed": composed_ms >= 833.33,
                    "gap25_deadline_missed_warm": (not warmup_excluded) and composed_ms >= 833.33,
                },
            }
        )
        # No image embedding, selected masks, RGB-D array, or lifted result
        # crosses the current-frame boundary.  The spool is output-only,
        # temporary offline evidence and is not queried by later frames.
        del (
            provider_result,
            masks,
            selected_indices,
            selected_ious,
            all_ious,
            rgb,
            depth_m,
            camera_to_world,
            boxes,
            result,
            points,
            keys,
            receipt,
            caught_warnings,
        )

    before_hash = _canonical_json_sha256(input_seals)
    input_end_rehash_started = time.perf_counter_ns()
    for seal in input_seals:
        path = _regular_file(Path(str(seal["path"])), f"{scene_id} frozen input after replay")
        if _sha256(path) != seal["sha256"]:
            raise N0ARunnerError(f"{scene_id} frozen input changed during replay")
    input_end_rehash_ms = (
        time.perf_counter_ns() - input_end_rehash_started
    ) / 1.0e6
    after_hash = _canonical_json_sha256(input_seals)
    if before_hash != after_hash:
        raise N0ARunnerError(f"{scene_id} frozen-input aggregate changed")

    arrays = evidence_spool.arrays(
        point_offsets=point_offsets,
        frame_ordinals=evidence_frame_ordinals,
        frame_ids=evidence_frame_ids,
        ranks=evidence_ranks,
        raw_indices=evidence_raw_indices,
        selected_indices=evidence_selected_indices,
        selected_ious=evidence_selected_ious,
        all_ious=evidence_all_ious,
        result_hashes=evidence_result_hashes,
    )
    evidence_path = output_root / "scenes" / f"{scene_id}.evidence.npz"
    evidence_write_started = time.perf_counter_ns()
    evidence_sha = _atomic_create_npz(evidence_path, arrays)
    evidence_npz_compression_write_ms = (
        time.perf_counter_ns() - evidence_write_started
    ) / 1.0e6
    evidence_bytes = evidence_path.stat().st_size
    cuda_peak = max(_cuda_peak_allocated_bytes(), provider_peak_bytes)
    if authenticated_warning_count != 2 * provider_forward_count:
        raise N0ARunnerError(
            f"{scene_id} authenticated warning/forward count differs"
        )
    counts = {
        "keyframe_count": len(frames),
        "successful_frame_count": successful_count,
        "source_count": len(source_ids),
        "provider_forward_count": provider_forward_count,
        "valid_hs_count": valid_count,
        "invalid_hs_count": len(source_ids) - valid_count,
        "nontrivial_hs_count": nontrivial_count,
        "authenticated_warning_count": authenticated_warning_count,
    }
    runtime = {
        "n0a_incremental_all_ms": _distribution(incremental_all),
        "n0a_incremental_warm_ms": _distribution(incremental_warm),
        "replay_composed_all_ms": _distribution(composed_all),
        "replay_composed_warm_ms": _distribution(composed_warm),
        "gap25_all_deadline_miss_count": int(sum(value >= 833.33 for value in composed_all)),
        "gap25_warm_deadline_miss_count": int(sum(value >= 833.33 for value in composed_warm)),
        "cuda_peak_memory_bytes": cuda_peak,
        "excluded_offline_reporting": {
            "input_pre_rehash_ms": input_pre_rehash_ms,
            "intrinsic_decode_ms": intrinsic_decode_ms,
            "input_end_rehash_ms": input_end_rehash_ms,
            "evidence_npz_compression_write_ms": evidence_npz_compression_write_ms,
            "included_in_online_or_warm_distributions": False,
        },
    }
    receipt: dict[str, Any] = {
        "schema": SCENE_SCHEMA,
        "protocol_id": PROTOCOL_ID,
        "complete": True,
        "scene_id": scene_id,
        "scene_index": scene_index,
        "run_signature_sha256": run_signature,
        "contracts": dict(CONTRACTS),
        "warning_policy": _warning_policy_receipt(warning_source),
        "inputs": {
            "f0_sidecar": input_seals[0],
            "intrinsic": input_seals[1],
            "frozen_inputs_before_sha256": before_hash,
            "frozen_inputs_after_sha256": after_hash,
        },
        "evidence_npz": {
            "path": os.fspath(evidence_path.resolve()),
            "sha256": evidence_sha,
            "byte_count": evidence_bytes,
            "schema": EVIDENCE_SCHEMA,
        },
        "frames": output_frames,
        "counts": counts,
        "runtime": runtime,
        "source_ids_sha256": _canonical_json_sha256(source_ids),
        "source_lineage_sha256": _canonical_json_sha256(lineage_hashes),
        "native_output_mutation_count": 0,
        "bounded_state": {
            "cross_frame_model_or_object_state": False,
            "current_frame_arrays_released_before_next_frame": True,
            "evidence_spool_is_output_only_offline_state": True,
            "maximum_sources_per_current_frame": MAX_SOURCES_PER_FRAME,
            "maximum_stored_points_per_source": MAX_STORED_POINTS,
        },
    }
    _validate_scene_warning_evidence(
        receipt, warning_policy=_warning_policy_receipt(warning_source)
    )
    receipt["content_sha256"] = _canonical_json_sha256(receipt)
    json_path = output_root / "scenes" / f"{scene_id}.json"
    scene_json_write_started = time.perf_counter_ns()
    json_sha = _atomic_create_json(json_path, receipt)
    scene_json_serialization_write_ms = (
        time.perf_counter_ns() - scene_json_write_started
    ) / 1.0e6
    row = {
        "scene_id": scene_id,
        "scene_index": scene_index,
        "sidecar": {"path": os.fspath(json_path.resolve()), "sha256": json_sha},
        "evidence_npz": receipt["evidence_npz"],
        "counts": counts,
        "runtime": runtime,
        "source_ids_sha256": receipt["source_ids_sha256"],
        "source_lineage_sha256": receipt["source_lineage_sha256"],
        "excluded_runtime_reporting": {
            **runtime["excluded_offline_reporting"],
            "scene_json_serialization_write_ms": scene_json_serialization_write_ms,
        },
        "resumed": False,
    }
    return row, receipt


def _runtime_gates(warm_incremental: Sequence[float], warm_composed: Sequence[float], *, deadline_misses: int, cuda_peak: int) -> dict[str, Any]:
    inc = _distribution(warm_incremental)
    composed = _distribution(warm_composed)
    mean_per_raw = float(composed["mean"]) / SOURCE_FRAME_STRIDE if composed["count"] else 0.0
    gates = {
        "n0a_incremental_warm_p95_ms": {"actual": inc["p95"], "threshold": 250.0, "comparator": "<=", "passed": inc["p95"] <= 250.0},
        "replay_composed_warm_p95_ms": {"actual": composed["p95"], "threshold": 500.0, "comparator": "<=", "passed": composed["p95"] <= 500.0},
        "replay_composed_warm_max_ms": {"actual": composed["max"], "threshold": 833.33, "comparator": "<", "passed": composed["max"] < 833.33},
        "replay_composed_mean_per_raw_frame_ms": {"actual": mean_per_raw, "threshold": 20.0, "comparator": "<=", "passed": mean_per_raw <= 20.0},
        "gap25_warm_deadline_miss_count": {"actual": deadline_misses, "threshold": 0, "comparator": "==", "passed": deadline_misses == 0},
        "cuda_peak_memory_bytes": {"actual": cuda_peak, "threshold": MAX_CUDA_BYTES, "comparator": "<=", "passed": cuda_peak <= MAX_CUDA_BYTES},
    }
    return {"gates": gates, "overall_pass": all(row["passed"] for row in gates.values())}


def _strict_nonnegative_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise N0ARunnerError(f"{label} must be a non-negative int")
    return value


def _recompute_scene_counts(receipt: Mapping[str, Any]) -> dict[str, int]:
    """Recompute a resumed scene census from its authenticated frame rows."""

    frames = receipt.get("frames")
    if not isinstance(frames, list):
        raise N0ARunnerError("completed scene frames are absent")
    successful_count = 0
    source_count = 0
    provider_forward_count = 0
    valid_count = 0
    nontrivial_count = 0
    warning_count = 0
    for ordinal, frame in enumerate(frames):
        if (
            not isinstance(frame, Mapping)
            or frame.get("frame_ordinal") != ordinal
            or not isinstance(frame.get("successful"), bool)
            or not isinstance(frame.get("provider_invoked"), bool)
        ):
            raise N0ARunnerError("completed scene frame identity/status differs")
        sources = frame.get("sources")
        if not isinstance(sources, list):
            raise N0ARunnerError("completed scene frame sources are absent")
        invoked = frame["provider_invoked"]
        if invoked != bool(sources):
            raise N0ARunnerError("completed scene provider/source distribution differs")
        successful_count += int(frame["successful"])
        provider_forward_count += int(invoked)
        warning_count += _strict_nonnegative_int(
            frame.get("authenticated_warning_count"),
            "completed frame authenticated_warning_count",
        )
        for source in sources:
            if not isinstance(source, Mapping):
                raise N0ARunnerError("completed scene source row differs")
            core_receipt = source.get("n0a_receipt")
            if (
                not isinstance(core_receipt, Mapping)
                or not isinstance(core_receipt.get("valid"), bool)
                or not isinstance(source.get("nontrivial_vs_h0"), bool)
            ):
                raise N0ARunnerError("completed scene source validity differs")
            valid = bool(core_receipt["valid"])
            nontrivial = bool(source["nontrivial_vs_h0"])
            if nontrivial and not valid:
                raise N0ARunnerError("invalid completed HS cannot be nontrivial")
            source_count += 1
            valid_count += int(valid)
            nontrivial_count += int(nontrivial)
    return {
        "keyframe_count": len(frames),
        "successful_frame_count": successful_count,
        "source_count": source_count,
        "provider_forward_count": provider_forward_count,
        "valid_hs_count": valid_count,
        "invalid_hs_count": source_count - valid_count,
        "nontrivial_hs_count": nontrivial_count,
        "authenticated_warning_count": warning_count,
    }


def _scene_excluded_runtime(row: Mapping[str, Any], receipt: Mapping[str, Any]) -> dict[str, float]:
    reported = row.get("excluded_runtime_reporting")
    runtime = receipt.get("runtime")
    receipt_reported = (
        runtime.get("excluded_offline_reporting")
        if isinstance(runtime, Mapping)
        else None
    )
    if not isinstance(reported, Mapping) or not isinstance(receipt_reported, Mapping):
        raise N0ARunnerError("completed scene excluded runtime reporting is absent")
    if (
        reported.get("included_in_online_or_warm_distributions") is not False
        or receipt_reported.get("included_in_online_or_warm_distributions") is not False
    ):
        raise N0ARunnerError("excluded scene runtime was marked as online/warm")
    result: dict[str, float] = {}
    for key in SCENE_EXCLUDED_RUNTIME_KEYS:
        value = _number(reported.get(key), f"completed scene {key}")
        if key != "scene_json_serialization_write_ms":
            receipt_value = _number(
                receipt_reported.get(key), f"completed receipt {key}"
            )
            if value != receipt_value:
                raise N0ARunnerError(f"completed scene {key} differs from receipt")
        result[key] = value
    return result


def _completed_manifest(
    path: Path,
    *,
    output_root: Path,
    run_signature: str,
    signature_payload: Mapping[str, Any],
    shard_index: int,
    num_shards: int,
    warning_policy: Mapping[str, Any],
    expected_scenes: Sequence[Mapping[str, Any]],
) -> dict[str, Any] | None:
    if not path.exists():
        return None
    _, manifest = _read_json(path, "completed N0a shard manifest")
    if (
        manifest.get("schema") != SHARD_SCHEMA
        or manifest.get("protocol_id") != PROTOCOL_ID
        or manifest.get("complete") is not True
        or manifest.get("run_signature_sha256") != run_signature
        or manifest.get("signature_payload_sha256") != run_signature
        or manifest.get("shard_index") != shard_index
        or manifest.get("num_shards") != num_shards
        or manifest.get("warning_policy") != warning_policy
        or manifest.get("contracts") != signature_payload.get("contracts")
        or manifest.get("source_receipts")
        != signature_payload.get("source_receipts")
        or manifest.get("provider_config") != signature_payload.get("provider_config")
    ):
        raise N0ARunnerError("completed shard manifest differs")
    environment = manifest.get("environment")
    inputs = manifest.get("inputs")
    if (
        not isinstance(environment, Mapping)
        or environment.get("preflight") != signature_payload.get("preflight")
        or not isinstance(inputs, Mapping)
        or inputs.get("f0_receipt") != signature_payload.get("f0_receipt")
        or not isinstance(inputs.get("universe"), Mapping)
        or {
            key: value
            for key, value in inputs["universe"].items()
            if key != "scene_census"
        }
        != signature_payload.get("universe")
    ):
        raise N0ARunnerError("completed shard signed inputs/environment differ")
    content = manifest.get("content_sha256")
    payload = dict(manifest)
    payload.pop("content_sha256", None)
    if content != _canonical_json_sha256(payload):
        raise N0ARunnerError("completed shard content hash differs")
    rows = manifest.get("scenes")
    if (
        not isinstance(rows, list)
        or not rows
        or len(rows) != len(expected_scenes)
    ):
        raise N0ARunnerError("completed shard scene census differs")
    recomputed_totals = {key: 0 for key in TOTAL_COUNT_KEYS}
    recomputed_excluded = {key: 0.0 for key in SCENE_EXCLUDED_RUNTIME_KEYS}
    recomputed_cold_provider_ms = 0.0
    recomputed_cold_provider_rows = 0
    expected_shard = {key: 0 for key in SEALED_CENSUS_KEYS}
    output_scenes_root = (_resolved_output_root(output_root) / "scenes").resolve(
        strict=False
    )
    for row, expected in zip(rows, expected_scenes):
        if not isinstance(row, Mapping) or not isinstance(expected, Mapping):
            raise N0ARunnerError("completed shard scene row differs")
        scene_id = expected.get("scene_id")
        scene_index = expected.get("scene_index")
        f0_sidecar_sha = expected.get("f0_sidecar_sha256")
        expected_counts = expected.get("counts")
        if (
            not isinstance(scene_id, str)
            or isinstance(scene_index, bool)
            or not isinstance(scene_index, int)
            or not _valid_sha256(f0_sidecar_sha)
            or not isinstance(expected_counts, Mapping)
            or row.get("scene_id") != scene_id
            or row.get("scene_index") != scene_index
        ):
            raise N0ARunnerError("completed shard scene identity/order differs")
        expected_json = (output_scenes_root / f"{scene_id}.json").resolve(
            strict=False
        )
        expected_npz = (output_scenes_root / f"{scene_id}.evidence.npz").resolve(
            strict=False
        )
        if (
            not isinstance(row.get("sidecar"), Mapping)
            or row["sidecar"].get("path") != os.fspath(expected_json)
            or not isinstance(row.get("evidence_npz"), Mapping)
            or row["evidence_npz"].get("path") != os.fspath(expected_npz)
        ):
            raise N0ARunnerError("completed shard contains a cross-root scene reference")
        resumed = _resume_scene(
            output_root=output_root,
            scene_id=scene_id,
            scene_index=scene_index,
            run_signature=run_signature,
            f0_sidecar_sha=str(f0_sidecar_sha),
            warning_policy=warning_policy,
        )
        if resumed is None:
            raise N0ARunnerError("completed shard scene files are absent")
        authenticated_row, receipt = resumed
        for key in (
            "scene_id",
            "scene_index",
            "sidecar",
            "evidence_npz",
            "counts",
            "runtime",
            "source_ids_sha256",
            "source_lineage_sha256",
        ):
            if row.get(key) != authenticated_row.get(key):
                raise N0ARunnerError(f"completed shard scene {key} differs")
        scene_counts = _recompute_scene_counts(receipt)
        if receipt.get("counts") != scene_counts:
            raise N0ARunnerError("completed scene counts do not recompute")
        for key in SEALED_CENSUS_KEYS:
            expected_value = _strict_nonnegative_int(
                expected_counts.get(key), f"expected completed scene {key}"
            )
            if scene_counts[key] != expected_value:
                raise N0ARunnerError("completed scene differs from sealed F0 census")
            expected_shard[key] += expected_value
        for key in TOTAL_COUNT_KEYS:
            recomputed_totals[key] += scene_counts[key]
        for key, value in _scene_excluded_runtime(row, receipt).items():
            recomputed_excluded[key] += value
        for frame in receipt["frames"]:
            runtime = frame.get("runtime")
            if (
                isinstance(runtime, Mapping)
                and runtime.get("cold_provider_metric_includes_first_forward")
                is True
            ):
                recomputed_cold_provider_rows += 1
                recomputed_cold_provider_ms += _number(
                    runtime.get("cold_provider_load_and_first_forward_ms"),
                    "completed cold provider load/first forward runtime",
                )

    totals = manifest.get("totals")
    if not isinstance(totals, Mapping) or dict(totals) != recomputed_totals:
        raise N0ARunnerError("completed shard totals do not recompute")
    if (
        manifest.get("expected_shard_census") != expected_shard
        or recomputed_totals["authenticated_warning_count"]
        != 2 * recomputed_totals["provider_forward_count"]
    ):
        raise N0ARunnerError("completed shard sealed census/warning formula differs")
    excluded = manifest.get("excluded_runtime_reporting")
    if not isinstance(excluded, Mapping):
        raise N0ARunnerError("completed shard excluded runtime reporting is absent")
    scene_aggregate = excluded.get("scene_aggregate_ms")
    if (
        excluded.get("included_in_online_or_warm_distributions") is not False
        or not isinstance(scene_aggregate, Mapping)
    ):
        raise N0ARunnerError("completed shard excluded runtime reporting differs")
    for key, expected_value in recomputed_excluded.items():
        if _number(scene_aggregate.get(key), f"completed shard aggregate {key}") != expected_value:
            raise N0ARunnerError("completed shard excluded runtime aggregate differs")
    sealed_universe_ms = _number(
        excluded.get("sealed_universe_pre_authentication_ms"),
        "completed sealed-universe pre-authentication runtime",
    )
    global_end_ms = _number(
        excluded.get("global_input_end_rehash_ms"),
        "completed global input end-rehash runtime",
    )
    if (
        _number(
            excluded.get("cold_provider_initialization_ms"),
            "completed cold provider initialization runtime",
        )
        < 0.0
        or excluded.get("cold_model_load_is_combined_with_first_forward") is not True
        or _number(
            excluded.get("cold_model_provider_load_and_first_forward_ms"),
            "completed cold provider load/first forward runtime",
        )
        != recomputed_cold_provider_ms
        or recomputed_cold_provider_rows
        != int(recomputed_totals["provider_forward_count"] > 0)
        or _number(
            excluded.get("input_pre_rehash_total_ms"),
            "completed total input pre-rehash runtime",
        )
        != sealed_universe_ms + recomputed_excluded["input_pre_rehash_ms"]
        or _number(
            excluded.get("input_end_rehash_total_ms"),
            "completed total input end-rehash runtime",
        )
        != global_end_ms + recomputed_excluded["input_end_rehash_ms"]
    ):
        raise N0ARunnerError("completed shard excluded runtime totals differ")
    manifest["manifest_path"] = os.fspath(path.resolve())
    manifest["manifest_sha256"] = _sha256(path)
    manifest["resumed_complete"] = True
    manifest["unsealed_return_only_runtime"] = {
        "shard_manifest_json_serialization_write_ms": 0.0,
        "included_in_online_or_warm_distributions": False,
        "authorizing_sealed_metric": False,
        "write_performed_by_this_resume": False,
    }
    return manifest


def run_n0a(
    *,
    f0_receipt_path: Path = DEFAULT_F0_RECEIPT,
    full200_scene_list_path: Path = DEFAULT_FULL200_SCENE_LIST,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    shard_index: int,
    num_shards: int,
    cohort_start: int = EXPECTED_COHORT_START,
    expected_scene_count: int = EXPECTED_SCENES,
    expected_keyframes: int | None = EXPECTED_KEYFRAMES,
    expected_successful_frames: int | None = EXPECTED_SUCCESSFUL_FRAMES,
    expected_sources: int | None = EXPECTED_SOURCES,
    provider_factory: Callable[[], object] | None = None,
    frame_loader: Callable[[Path, Path, Path], tuple[np.ndarray, np.ndarray, np.ndarray]] | None = None,
    plan_only: bool = False,
    resume: bool = True,
) -> dict[str, Any]:
    """Run one source-index-modulo shard after authenticating the full cohort."""

    if num_shards < 1 or shard_index not in range(num_shards):
        raise N0ARunnerError("shard-index must be in [0,num-shards)")
    production = cohort_start == EXPECTED_COHORT_START and expected_scene_count == EXPECTED_SCENES
    if production and (provider_factory is not None or frame_loader is not None):
        raise N0ARunnerError("production N0a forbids injected provider/frame loader")
    output_root = Path(output_root)
    if production:
        _validate_production_output_root(output_root, resume=resume)
    universe_authentication_started = time.perf_counter_ns()
    f0_seal, universe, scenes, rows, scene_census = _load_f0_universe(
        f0_receipt_path=Path(f0_receipt_path),
        full200_scene_list_path=Path(full200_scene_list_path),
        cohort_start=cohort_start,
        expected_scene_count=expected_scene_count,
        expected_keyframes=expected_keyframes,
        expected_successful_frames=expected_successful_frames,
        expected_sources=expected_sources,
    )
    sealed_universe_pre_authentication_ms = (
        time.perf_counter_ns() - universe_authentication_started
    ) / 1.0e6
    assigned_positions = tuple(
        position
        for position, row in enumerate(rows)
        if int(row["scene_index"]) % num_shards == shard_index
    )
    expected_completed_scenes = tuple(
        {
            "scene_id": scenes[position],
            "scene_index": int(rows[position]["scene_index"]),
            "f0_sidecar_sha256": str(rows[position]["sidecar"]["sha256"]),
            "counts": dict(scene_census[scenes[position]]),
        }
        for position in assigned_positions
    )
    plan = {
        "protocol_id": PROTOCOL_ID,
        "scene_schema": SCENE_SCHEMA,
        "shard_schema": SHARD_SCHEMA,
        "warning_policy_id": WARNING_POLICY_ID,
        "shard_index": shard_index,
        "num_shards": num_shards,
        "full_universe_authenticated": True,
        "universe_census": universe["census"],
        "scene_count": len(assigned_positions),
        "scene_indices": [int(rows[position]["scene_index"]) for position in assigned_positions],
        "scene_ids": [scenes[position] for position in assigned_positions],
        "f0_receipt": f0_seal,
        "output_root": os.fspath(_resolved_output_root(output_root)),
        "contracts": dict(CONTRACTS),
    }
    if plan_only:
        return plan

    from boxfusion import sam2_boxprompt_provider as provider_module
    from boxfusion import sam2_masklift_n0a as core_module

    sources_receipt = _source_receipts(
        core_module, provider_module, provider_factory, frame_loader
    )
    preflight = (
        _production_preflight(provider_module)
        if production
        else {
            "test_injection_mode": True,
            "production_environment_verified": False,
            "python_version": platform.python_version(),
            "numpy_version": np.__version__,
        }
    )
    provider_config = _jsonable(
        getattr(provider_module, "PRODUCTION_CONFIG", {}), "SAM2 production config"
    )
    warning_source = _warning_policy_source()
    warning_policy = _warning_policy_receipt(warning_source)
    signature_payload = {
        "protocol_id": PROTOCOL_ID,
        "protocol_sha256": EXPECTED_PROTOCOL_SHA256,
        "f0_receipt": f0_seal,
        "universe": {key: value for key, value in universe.items() if key != "scene_census"},
        "scene_order": list(scenes),
        "source_receipts": sources_receipt,
        "provider_config": provider_config,
        "warning_policy": warning_policy,
        "preflight": preflight,
        "contracts": dict(CONTRACTS),
        "shard_index": shard_index,
        "num_shards": num_shards,
    }
    run_signature = _canonical_json_sha256(signature_payload)
    manifest_path = output_root / "shards" / f"shard-{shard_index:03d}-of-{num_shards:03d}.json"
    if resume:
        completed = _completed_manifest(
            manifest_path,
            output_root=output_root,
            run_signature=run_signature,
            signature_payload=signature_payload,
            shard_index=shard_index,
            num_shards=num_shards,
            warning_policy=warning_policy,
            expected_scenes=expected_completed_scenes,
        )
        if completed is not None:
            return completed
    if production:
        partial_outputs = [
            output_root / "scenes" / f"{scenes[position]}{suffix}"
            for position in assigned_positions
            for suffix in (".json", ".evidence.npz")
            if (output_root / "scenes" / f"{scenes[position]}{suffix}").exists()
        ]
        if partial_outputs:
            raise N0ARunnerError(
                "production partial-scene resume is forbidden; use a new output root"
            )

    provider: object | None = None
    loader = frame_loader or _default_frame_loader
    lift_mask = core_module.lift_sam2_mask
    scene_rows: list[dict[str, Any]] = []
    scene_receipts: list[dict[str, Any]] = []
    call_index = [0]
    provider_initialization_ms = 0.0
    for position in assigned_positions:
        merged_row = rows[position]
        scene_id = scenes[position]
        resumed = (
            _resume_scene(
                output_root=output_root,
                scene_id=scene_id,
                scene_index=int(merged_row["scene_index"]),
                run_signature=run_signature,
                f0_sidecar_sha=str(merged_row["sidecar"]["sha256"]),
                warning_policy=warning_policy,
            )
            if resume
            else None
        )
        if resumed is not None:
            row, receipt = resumed
        else:
            if provider is None:
                provider_initialization_started = time.perf_counter_ns()
                if provider_factory is None:
                    provider = provider_module.FrozenSAM2BoxPromptProvider()
                else:
                    provider = provider_factory()
                provider_initialization_ms += (
                    time.perf_counter_ns() - provider_initialization_started
                ) / 1.0e6
            with _EvidenceSpool(output_root / ".evidence-spool") as evidence_spool:
                row, receipt = _process_scene(
                    merged_row=merged_row,
                    provider=provider,
                    lift_mask=lift_mask,
                    frame_loader=loader,
                    output_root=output_root,
                    evidence_spool=evidence_spool,
                    call_index=call_index,
                    run_signature=run_signature,
                    warning_source=warning_source,
                )
        expected = scene_census[scene_id]
        for key in (
            "keyframe_count",
            "successful_frame_count",
            "source_count",
            "provider_forward_count",
        ):
            if int(row["counts"][key]) != expected[key]:
                raise N0ARunnerError(f"{scene_id} completed {key} differs from sealed F0")
        scene_rows.append(row)
        scene_receipts.append(receipt)

    global_input_seals: list[dict[str, str]] = [
        {"role": "f0_full200_receipt", **f0_seal},
        {"role": "f0_full200_scene_list", **universe["full200_scene_list"]},
    ]
    for role in ("runner", "protocol", "core", "provider"):
        seal = sources_receipt[role]
        global_input_seals.append({"role": role, **seal})
    global_input_seals.append(
        {
            "role": "warning_source",
            "path": os.fspath(warning_source),
            "sha256": EXPECTED_WARNING_SOURCE_SHA256,
        }
    )
    for merged_row in rows:
        sidecar = merged_row["sidecar"]
        global_input_seals.append(
            {
                "role": f"f0_sidecar:{merged_row['scene_id']}",
                "path": str(sidecar["path"]),
                "sha256": str(sidecar["sha256"]),
            }
        )
    global_before_sha = _canonical_json_sha256(global_input_seals)
    global_input_end_rehash_started = time.perf_counter_ns()
    for seal in global_input_seals:
        _rehash_reference(seal, f"global end rehash {seal['role']}")
    global_input_end_rehash_ms = (
        time.perf_counter_ns() - global_input_end_rehash_started
    ) / 1.0e6
    global_after_sha = _canonical_json_sha256(global_input_seals)
    if global_after_sha != global_before_sha:
        raise N0ARunnerError("global frozen-input aggregate changed during replay")
    global_input_integrity = {
        "seal_count": len(global_input_seals),
        "before_sha256": global_before_sha,
        "after_sha256": global_after_sha,
        "passed": True,
        "seals": global_input_seals,
        "sam2_asset_end_rehash": (
            "pending_mandatory_merge_replay_audit_provider_authenticated_before_masks"
        ),
    }

    totals = {
        key: int(sum(int(row["counts"][key]) for row in scene_rows))
        for key in TOTAL_COUNT_KEYS
    }
    expected_shard = {
        key: int(
            sum(scene_census[scenes[position]][key] for position in assigned_positions)
        )
        for key in SEALED_CENSUS_KEYS
    }
    if any(totals[key] != value for key, value in expected_shard.items()):
        raise N0ARunnerError("completed shard census differs from authenticated universe")
    full_universe_in_this_shard = len(assigned_positions) == expected_scene_count
    expected_shard_warning_count = 2 * totals["provider_forward_count"]
    if totals["authenticated_warning_count"] != expected_shard_warning_count:
        raise N0ARunnerError("completed shard warning/forward count differs")
    if (
        production
        and full_universe_in_this_shard
        and (
            totals["provider_forward_count"] != EXPECTED_PROVIDER_FORWARDS
            or totals["authenticated_warning_count"]
            != EXPECTED_AUTHENTICATED_WARNINGS
        )
    ):
        raise N0ARunnerError("full production warning census differs")
    warm_incremental = [
        float(frame["runtime"]["n0a_incremental_ms"])
        for scene in scene_receipts
        for frame in scene["frames"]
        if isinstance(frame.get("runtime"), Mapping)
        and frame["runtime"].get("n0a_warmup_excluded") is False
    ]
    warm_composed = [
        float(frame["runtime"]["replay_composed_ms"])
        for scene in scene_receipts
        for frame in scene["frames"]
        if isinstance(frame.get("runtime"), Mapping)
        and frame["runtime"].get("n0a_warmup_excluded") is False
    ]
    deadline_misses = int(
        sum(scene["runtime"]["gap25_warm_deadline_miss_count"] for scene in scene_receipts)
    )
    cuda_peak = max(
        (int(scene["runtime"]["cuda_peak_memory_bytes"]) for scene in scene_receipts),
        default=0,
    )
    runtime_gates = _runtime_gates(
        warm_incremental,
        warm_composed,
        deadline_misses=deadline_misses,
        cuda_peak=cuda_peak,
    )
    valid_scene_count = sum(int(row["counts"]["valid_hs_count"]) > 0 for row in scene_rows)
    nontrivial_scene_count = sum(
        int(row["counts"]["nontrivial_hs_count"]) > 0 for row in scene_rows
    )
    capacity_gates = {
        "merge_only": not full_universe_in_this_shard,
        "valid_hs_count": {
            "actual": totals["valid_hs_count"],
            "threshold": MIN_VALID_SOURCE_COUNT,
            "comparator": ">=",
            "passed": totals["valid_hs_count"] >= MIN_VALID_SOURCE_COUNT,
        },
        "valid_scene_count": {
            "actual": valid_scene_count,
            "threshold": MIN_VALID_SCENE_COUNT,
            "comparator": ">=",
            "passed": valid_scene_count >= MIN_VALID_SCENE_COUNT,
        },
        "nontrivial_hs_count": {
            "actual": totals["nontrivial_hs_count"],
            "threshold": MIN_NONTRIVIAL_SOURCE_COUNT,
            "comparator": ">=",
            "passed": totals["nontrivial_hs_count"] >= MIN_NONTRIVIAL_SOURCE_COUNT,
        },
        "nontrivial_scene_count": {
            "actual": nontrivial_scene_count,
            "threshold": MIN_NONTRIVIAL_SCENE_COUNT,
            "comparator": ">=",
            "passed": nontrivial_scene_count >= MIN_NONTRIVIAL_SCENE_COUNT,
        },
    }
    capacity_pass = (
        all(row["passed"] for key, row in capacity_gates.items() if key != "merge_only")
        if full_universe_in_this_shard
        else None
    )
    shard_source_ids = [
        str(source["source_id"])
        for scene in scene_receipts
        for frame in scene["frames"]
        for source in frame["sources"]
    ]
    replay_sample_source_ids = [
        source_id
        for source_id in shard_source_ids
        if int.from_bytes(
            hashlib.sha256(source_id.encode("ascii")).digest()[:2], "big"
        )
        < 0x0290
    ]
    decision = "awaiting_complete_extra100_merge_and_mandatory_replay"
    determinism_gates = {
        "status": "pending_create_only_merge_replay_receipt",
        "overall_pass": None,
        "registered_warn_only_operation": "aten::cumsum_cuda",
        "warning_policy": warning_policy,
        "authenticated_warning_count": totals["authenticated_warning_count"],
        "expected_warning_count": expected_shard_warning_count,
        "warning_count_formula": "2 * provider_forward_count",
        "per_forward_exact_warning_pair_passed": True,
        "half_prefix_exact_source_result_replay": None,
        "fresh_same_gpu_one_percent_exact_replay": None,
        "one_percent_selector": (
            "big_endian_first_two_bytes_of_sha256(source_id_ASCII)_lt_0x0290"
        ),
        "shard_selected_source_count": len(replay_sample_source_ids),
        "shard_selected_source_ids_sha256": _canonical_json_sha256(
            replay_sample_source_ids
        ),
        "shard_selected_source_ids": replay_sample_source_ids,
        "future_frame_perturbation_invariance": None,
        "n0b_or_gt_stage_authorized": False,
    }
    scene_excluded_runtime = {
        key: float(
            sum(
                _number(
                    row.get("excluded_runtime_reporting", {}).get(key),
                    f"scene excluded runtime {key}",
                )
                for row in scene_rows
            )
        )
        for key in SCENE_EXCLUDED_RUNTIME_KEYS
    }
    cold_provider_rows = [
        frame["runtime"]
        for scene in scene_receipts
        for frame in scene["frames"]
        if isinstance(frame.get("runtime"), Mapping)
        and frame["runtime"].get("cold_provider_metric_includes_first_forward")
        is True
    ]
    expected_cold_row_count = int(totals["provider_forward_count"] > 0)
    if len(cold_provider_rows) != expected_cold_row_count:
        raise N0ARunnerError("cold provider first-forward reporting differs")
    cold_provider_load_and_first_forward_ms = float(
        sum(
            _number(
                row.get("cold_provider_load_and_first_forward_ms"),
                "cold provider load/first forward runtime",
            )
            for row in cold_provider_rows
        )
    )
    excluded_runtime_reporting = {
        "included_in_online_or_warm_distributions": False,
        "cold_provider_initialization_ms": provider_initialization_ms,
        "cold_model_provider_load_and_first_forward_ms": (
            cold_provider_load_and_first_forward_ms
        ),
        "cold_model_load_is_combined_with_first_forward": True,
        "sealed_universe_pre_authentication_ms": (
            sealed_universe_pre_authentication_ms
        ),
        "global_input_end_rehash_ms": global_input_end_rehash_ms,
        "scene_aggregate_ms": scene_excluded_runtime,
        "input_pre_rehash_total_ms": (
            sealed_universe_pre_authentication_ms
            + scene_excluded_runtime["input_pre_rehash_ms"]
        ),
        "input_end_rehash_total_ms": (
            global_input_end_rehash_ms
            + scene_excluded_runtime["input_end_rehash_ms"]
        ),
        "shard_manifest_json_write_reporting": (
            "measured_after_seal_and_returned_out_of_band_to_avoid_self_reference"
        ),
    }

    manifest: dict[str, Any] = {
        "schema": SHARD_SCHEMA,
        "protocol_id": PROTOCOL_ID,
        "complete": True,
        "shard_index": shard_index,
        "num_shards": num_shards,
        "run_signature_sha256": run_signature,
        "signature_payload_sha256": _canonical_json_sha256(signature_payload),
        "contracts": dict(CONTRACTS),
        "inputs": {"f0_receipt": f0_seal, "universe": universe},
        "global_input_integrity": global_input_integrity,
        "source_receipts": sources_receipt,
        "provider_config": provider_config,
        "warning_policy": warning_policy,
        "environment": {
            "preflight": preflight,
            "platform": platform.platform(),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
        "scenes": scene_rows,
        "totals": totals,
        "expected_shard_census": expected_shard,
        "runtime": {
            "n0a_incremental_warm_ms": _distribution(warm_incremental),
            "replay_composed_warm_ms": _distribution(warm_composed),
            "gap25_warm_deadline_miss_count": deadline_misses,
            "cuda_peak_memory_bytes": cuda_peak,
        },
        "excluded_runtime_reporting": excluded_runtime_reporting,
        "runtime_gates": runtime_gates,
        "runtime_gates_preliminary": True,
        "capacity_gates": capacity_gates,
        "capacity_gates_overall_pass": capacity_pass,
        "capacity_gates_preliminary": True,
        "determinism_gates": determinism_gates,
        "decision": decision,
        "resumed_scene_count": sum(bool(row.get("resumed")) for row in scene_rows),
        "native_output_mutation_count": 0,
    }
    manifest["content_sha256"] = _canonical_json_sha256(manifest)
    shard_manifest_write_started = time.perf_counter_ns()
    manifest_sha = _atomic_create_json(manifest_path, manifest)
    shard_manifest_json_serialization_write_ms = (
        time.perf_counter_ns() - shard_manifest_write_started
    ) / 1.0e6
    manifest["manifest_path"] = os.fspath(manifest_path.resolve())
    manifest["manifest_sha256"] = manifest_sha
    manifest["unsealed_return_only_runtime"] = {
        "shard_manifest_json_serialization_write_ms": (
            shard_manifest_json_serialization_write_ms
        ),
        "included_in_online_or_warm_distributions": False,
        "authorizing_sealed_metric": False,
    }
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--f0-receipt", type=Path, default=DEFAULT_F0_RECEIPT)
    parser.add_argument(
        "--full200-scene-list", type=Path, default=DEFAULT_FULL200_SCENE_LIST
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--num-shards", type=int, required=True)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    return parser


def main() -> None:
    args = _parser().parse_args()
    result = run_n0a(
        f0_receipt_path=args.f0_receipt,
        full200_scene_list_path=args.full200_scene_list,
        output_root=args.output_root,
        shard_index=args.shard_index,
        num_shards=args.num_shards,
        plan_only=args.plan_only,
        resume=not args.no_resume,
    )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
