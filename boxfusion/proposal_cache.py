"""Immutable CuTR post-filter proposal cache for paired online ablations.

Controlled E2/Sshadow/Graw/Gclean experiments must compare the same ordered
CuTR proposal rows.  Record mode saves the final camera-frame proposals after
all legacy filters and immediately consumes a deserialized copy.  Replay mode
loads those exact rows and does not execute the CuTR forward pass.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import re
import tempfile
import copy
from dataclasses import dataclass
import fcntl
from numbers import Integral, Real
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple

import numpy as np
import torch

from boxfusion.boxes import BoxDOF, GeneralInstance3DBoxes
from boxfusion.instances import Instances3D


SCHEMA = "boxfusion.cutr_postfilter_cache.v3"
INDEX_SCHEMA = "boxfusion.cutr_postfilter_cache.index.v1"
TERMINAL_POLICY = "upstream_boxfusion_early_exit_v1"
EXPECTED_FIELDS = (
    "scores",
    "pred_classes",
    "pred_boxes",
    "pred_logits",
    "pred_boxes_3d",
    "object_desc",
    "pred_proj_xy",
)
VALID_ATTEMPTS = ("primary", "retry")
_INPUT_FIELDS = (
    "image",
    "depth",
    "image_K",
    "depth_K",
    "camera_to_world",
)
_SCENE_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
_HEX_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

_PAYLOAD_KEYS = {
    "schema",
    "scene_id",
    "frame_id",
    "namespace",
    "producer_fingerprint",
    "image_size",
    "field_names",
    "fields",
    "field_metadata",
    "count",
    "attempt_id",
    "input_signature",
    "rng_state",
    "rng_sha256",
    "protected_hashes",
    "geometry_sha256",
    "proposal_contract_sha256",
}
_INDEX_RECEIPT_KEYS = {
    "schema",
    "scene_id",
    "namespace",
    "producer_fingerprint",
    "manifest",
    "manifest_sha256",
    "prediction_sha256",
    "record_count",
    "proposal_count",
}

__all__ = [
    "INDEX_SCHEMA",
    "ProposalCache",
    "ProposalCacheConfig",
    "ProposalCacheError",
    "SCHEMA",
    "TERMINAL_POLICY",
    "build_proposal_cache",
    "geometry_hash",
    "protected_proposal_hashes",
]


class ProposalCacheError(RuntimeError):
    """Raised when an immutable proposal-cache contract is broken."""


def _strict_int(name: str, value: Any, *, minimum: int = 0) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise ProposalCacheError(f"{name} must be an integer")
    result = int(value)
    if result < minimum:
        raise ProposalCacheError(f"{name} must be at least {minimum}")
    return result


def _strict_scene_id(scene_id: Any) -> str:
    if not isinstance(scene_id, str) or _SCENE_RE.fullmatch(scene_id) is None:
        raise ProposalCacheError(f"Invalid scene ID: {scene_id!r}")
    return scene_id


def _strict_fingerprint(name: str, value: Any) -> str:
    if not isinstance(value, str) or _HEX_SHA256_RE.fullmatch(value) is None:
        raise ProposalCacheError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _strict_sha256(name: str, value: Any) -> str:
    if not isinstance(value, str) or _HEX_SHA256_RE.fullmatch(value) is None:
        raise ProposalCacheError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _strict_hash_mapping(
    name: str,
    value: Any,
    *,
    keys: tuple[str, ...],
) -> Dict[str, str]:
    if not isinstance(value, Mapping) or set(value) != set(keys):
        raise ProposalCacheError(f"Invalid {name} schema")
    return {
        key: _strict_sha256(f"{name}.{key}", value[key])
        for key in keys
    }


def _validate_index_receipt(
    value: Any,
    *,
    scene_id: str,
    namespace: str,
    producer_fingerprint: str,
) -> Dict[str, Any]:
    if (
        not isinstance(value, dict)
        or set(value) != _INDEX_RECEIPT_KEYS
        or value.get("schema") != INDEX_SCHEMA
        or value.get("scene_id") != scene_id
        or value.get("namespace") != namespace
        or value.get("producer_fingerprint") != producer_fingerprint
        or value.get("manifest") != f"{scene_id}/manifest.json"
    ):
        raise ProposalCacheError(
            f"Invalid proposal-cache index receipt: {scene_id}"
        )
    _strict_sha256("receipt manifest hash", value.get("manifest_sha256"))
    _strict_sha256("receipt prediction hash", value.get("prediction_sha256"))
    _strict_int("receipt record_count", value.get("record_count"))
    _strict_int("receipt proposal_count", value.get("proposal_count"))
    return dict(value)


def _canonical_json_sha256(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(8 * 1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _load_json_file(
    path: Path,
    *,
    expected_sha256: Optional[str] = None,
    label: str,
) -> Dict[str, Any]:
    try:
        data = path.read_bytes()
    except OSError as error:
        raise ProposalCacheError(f"Could not read {label}: {path}") from error
    if expected_sha256 is not None:
        _strict_sha256(f"expected {label} hash", expected_sha256)
        if hashlib.sha256(data).hexdigest() != expected_sha256:
            raise ProposalCacheError(f"{label} hash mismatch: {path}")
    try:
        value = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ProposalCacheError(f"Invalid {label}: {path}") from error
    if not isinstance(value, dict):
        raise ProposalCacheError(f"Invalid {label}: {path}")
    return value


def _canonical_tensor(value: torch.Tensor) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise ProposalCacheError(f"Expected a tensor, received {type(value)}")
    return value.detach().cpu().contiguous().clone()


def _tensor_sha256(value: torch.Tensor) -> str:
    value = _canonical_tensor(value)
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("utf-8"))
    digest.update(b"\0")
    digest.update(repr(tuple(value.shape)).encode("ascii"))
    digest.update(b"\0")
    digest.update(value.view(torch.uint8).numpy().tobytes(order="C"))
    return digest.hexdigest()


def _tensor_metadata(value: torch.Tensor) -> Dict[str, Any]:
    canonical = _canonical_tensor(value)
    return {
        "dtype": str(canonical.dtype),
        "shape": list(canonical.shape),
        "sha256": _tensor_sha256(canonical),
    }


def _rng_contract(state: Mapping[str, Any]) -> Dict[str, Any]:
    try:
        python_state = state["python"]
        numpy_state = state["numpy"]
        if not isinstance(python_state, Mapping) or set(python_state) != {
            "version",
            "state",
            "gauss",
        }:
            raise ProposalCacheError("Invalid cached Python RNG schema")
        if not isinstance(numpy_state, Mapping) or set(numpy_state) != {
            "algorithm",
            "state",
            "position",
            "has_gauss",
            "cached_gaussian",
        }:
            raise ProposalCacheError("Invalid cached NumPy RNG schema")
        if set(state) != {"torch_cpu", "torch_cuda", "python", "numpy"}:
            raise ProposalCacheError("Invalid cached RNG state schema")
        if not isinstance(numpy_state["algorithm"], str):
            raise ProposalCacheError("Invalid cached NumPy RNG algorithm")
        if isinstance(numpy_state["cached_gaussian"], (bool, np.bool_)) or not isinstance(
            numpy_state["cached_gaussian"], Real
        ):
            raise ProposalCacheError("Invalid cached NumPy Gaussian value")
        result = {
            "torch_cpu": _tensor_metadata(state["torch_cpu"]),
            "torch_cuda": (
                None
                if state.get("torch_cuda") is None
                else _tensor_metadata(state["torch_cuda"])
            ),
            "python": {
                "version": _strict_int(
                    "cached Python RNG version", python_state["version"]
                ),
                "state": _tensor_metadata(python_state["state"]),
                "gauss": python_state["gauss"],
            },
            "numpy": {
                "algorithm": numpy_state["algorithm"],
                "state": _tensor_metadata(numpy_state["state"]),
                "position": _strict_int(
                    "cached NumPy RNG position", numpy_state["position"]
                ),
                "has_gauss": _strict_int(
                    "cached NumPy RNG has_gauss", numpy_state["has_gauss"]
                ),
                "cached_gaussian": float(numpy_state["cached_gaussian"]),
            },
        }
    except (KeyError, TypeError, ValueError, OverflowError) as error:
        raise ProposalCacheError("Invalid cached RNG state schema") from error
    if (
        result["torch_cpu"]["dtype"] != "torch.uint8"
        or len(result["torch_cpu"]["shape"]) != 1
        or result["torch_cpu"]["shape"][0] <= 0
    ):
        raise ProposalCacheError("Invalid cached Torch CPU RNG state")
    if result["torch_cuda"] is not None and (
        result["torch_cuda"]["dtype"] != "torch.uint8"
        or len(result["torch_cuda"]["shape"]) != 1
        or result["torch_cuda"]["shape"][0] <= 0
    ):
        raise ProposalCacheError("Invalid cached Torch CUDA RNG state")
    if (
        result["python"]["version"] != 3
        or result["python"]["state"]["dtype"] != "torch.int64"
        or result["python"]["state"]["shape"] != [625]
        or (
            result["python"]["gauss"] is not None
            and (
                isinstance(result["python"]["gauss"], (bool, np.bool_))
                or not isinstance(result["python"]["gauss"], (int, float))
                or not math.isfinite(float(result["python"]["gauss"]))
            )
        )
    ):
        raise ProposalCacheError("Invalid cached Python RNG state")
    if (
        result["numpy"]["state"]["dtype"] != "torch.int64"
        or result["numpy"]["state"]["shape"] != [624]
        or not 0 <= result["numpy"]["position"] <= 624
        or result["numpy"]["has_gauss"] not in (0, 1)
        or not math.isfinite(result["numpy"]["cached_gaussian"])
    ):
        raise ProposalCacheError("Invalid cached NumPy RNG state")
    if result["numpy"]["algorithm"] != "MT19937":
        raise ProposalCacheError("Unsupported cached NumPy RNG algorithm")
    return result


def _rng_sha256(state: Mapping[str, Any]) -> str:
    return _canonical_json_sha256(_rng_contract(state))


def _array_sha256(value: Any) -> str:
    if isinstance(value, torch.Tensor):
        return _tensor_sha256(value)
    array = np.ascontiguousarray(np.asarray(value))
    if array.dtype.kind not in "biufc":
        raise ProposalCacheError(
            f"Proposal-cache inputs must be numeric, received {array.dtype}"
        )
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("utf-8"))
    digest.update(b"\0")
    digest.update(repr(tuple(array.shape)).encode("ascii"))
    digest.update(b"\0")
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _shape_of(value: Any) -> Tuple[int, ...]:
    if isinstance(value, torch.Tensor):
        return tuple(int(item) for item in value.shape)
    try:
        return tuple(int(item) for item in np.asarray(value).shape)
    except Exception as error:
        raise ProposalCacheError("Invalid proposal-cache input array") from error


def _input_signature(inputs: Mapping[str, Any]) -> Dict[str, str]:
    if not isinstance(inputs, Mapping) or set(inputs) != set(_INPUT_FIELDS):
        actual = tuple(inputs.keys()) if isinstance(inputs, Mapping) else type(inputs)
        raise ProposalCacheError(
            f"Unexpected proposal-cache input schema: {actual}"
        )
    shapes = {name: _shape_of(inputs[name]) for name in _INPUT_FIELDS}
    if len(shapes["image"]) != 3 or shapes["image"][2] not in (1, 3, 4):
        raise ProposalCacheError("image must have shape HxWxC")
    if len(shapes["depth"]) != 2:
        raise ProposalCacheError("depth must have shape HxW")
    if shapes["image_K"] != (3, 3) or shapes["depth_K"] != (3, 3):
        raise ProposalCacheError("image_K and depth_K must be 3x3")
    if shapes["camera_to_world"] != (4, 4):
        raise ProposalCacheError("camera_to_world must be 4x4")
    return {name: _array_sha256(inputs[name]) for name in _INPUT_FIELDS}


def _field_hash(instances: Any, name: str) -> Optional[str]:
    if not instances.has(name):
        return None
    value = instances.get(name)
    if isinstance(value, torch.Tensor):
        return _tensor_sha256(value)
    if hasattr(value, "tensor"):
        pieces = [_tensor_sha256(value.tensor)]
        if hasattr(value, "R"):
            pieces.append(_tensor_sha256(value.R))
        return hashlib.sha256("|".join(pieces).encode("ascii")).hexdigest()
    if isinstance(value, np.ndarray):
        return _array_sha256(value)
    return hashlib.sha256(repr(value).encode("utf-8")).hexdigest()


def protected_proposal_hashes(instances: Any) -> Dict[str, Optional[str]]:
    """Hash fields whose values/order must remain exactly frozen."""

    names = (
        "pred_boxes",
        "scores",
        "pred_classes",
        "pred_logits",
        "object_desc",
    )
    return {name: _field_hash(instances, name) for name in names}


def geometry_hash(instances: Any) -> str:
    boxes = instances.pred_boxes_3d
    return hashlib.sha256(
        (
            _tensor_sha256(boxes.tensor)
            + "|"
            + _tensor_sha256(boxes.R)
        ).encode("ascii")
    ).hexdigest()


def _validate_proposal_shapes(instances: Any) -> int:
    """Validate the fixed CuTR row schema without coercing values or dtypes."""

    count = int(len(instances))
    tensor_shapes = {
        name: tuple(int(item) for item in instances.get(name).shape)
        for name in EXPECTED_FIELDS
        if name != "pred_boxes_3d"
    }
    if (
        tensor_shapes["scores"] != (count,)
        or tensor_shapes["pred_classes"] != (count,)
        or tensor_shapes["pred_boxes"] != (count, 4)
        or len(tensor_shapes["pred_logits"]) != 2
        or tensor_shapes["pred_logits"][0] != count
        or len(tensor_shapes["object_desc"]) != 2
        or tensor_shapes["object_desc"][0] != count
        or tensor_shapes["pred_proj_xy"] != (count, 2)
    ):
        raise ProposalCacheError(
            f"Unexpected CuTR tensor shapes: count={count}, shapes={tensor_shapes}"
        )
    boxes = instances.pred_boxes_3d
    if (
        tuple(int(item) for item in boxes.tensor.shape) != (count, 6)
        or tuple(int(item) for item in boxes.R.shape) != (count, 3, 3)
    ):
        raise ProposalCacheError("Unexpected CuTR 3D-box tensor shapes")
    return count


def _proposal_contract_sha256_from_parts(
    *,
    image_size: Any,
    field_names: Any,
    field_metadata: Mapping[str, Any],
    count: int,
    protected_hashes: Mapping[str, Any],
    geometry_sha256: str,
) -> str:
    try:
        contract = {
            "image_size": [int(item) for item in image_size],
            "field_names": [str(item) for item in field_names],
            "field_metadata": dict(field_metadata),
            "count": int(count),
            "protected_hashes": dict(protected_hashes),
            "geometry_sha256": str(geometry_sha256),
        }
    except (TypeError, ValueError, OverflowError) as error:
        raise ProposalCacheError("Invalid proposal contract metadata") from error
    return _canonical_json_sha256(contract)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _install_create_only(temporary_path: Path, final_path: Path) -> None:
    """Atomically install a same-filesystem temp without replacing a winner."""

    try:
        os.link(temporary_path, final_path)
    except FileExistsError as error:
        raise ProposalCacheError(
            f"Refusing to overwrite immutable cache artifact: {final_path}"
        ) from error
    _fsync_directory(final_path.parent)


def _atomic_json_write(path: Path, value: Mapping[str, Any]) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        _install_create_only(temporary_path, path)
        path.chmod(0o444)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


@dataclass(frozen=True)
class ProposalCacheConfig:
    mode: str
    root: Path
    namespace: str
    baseline_prediction_root: Optional[Path] = None
    require_index: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.mode, str) or self.mode not in {
            "disabled",
            "record",
            "replay",
        }:
            raise ValueError("proposal-cache mode must be disabled, record, or replay")
        if isinstance(self.root, bool) or not isinstance(
            self.root, (str, os.PathLike)
        ):
            raise ValueError("proposal-cache root must be a filesystem path")
        object.__setattr__(
            self,
            "root",
            Path(os.path.abspath(os.path.expanduser(os.fspath(self.root)))),
        )
        if not isinstance(self.namespace, str):
            raise ValueError("proposal-cache namespace must be a string")
        namespace = self.namespace.strip()
        if self.mode != "disabled" and not namespace:
            raise ValueError("proposal-cache namespace is required")
        if namespace and (
            namespace in (".", "..")
            or Path(namespace).name != namespace
            or os.sep in namespace
            or (os.altsep is not None and os.altsep in namespace)
            or _SCENE_RE.fullmatch(namespace) is None
        ):
            raise ValueError("proposal-cache namespace must be one safe path component")
        object.__setattr__(self, "namespace", namespace)
        if self.baseline_prediction_root is not None:
            if isinstance(self.baseline_prediction_root, bool) or not isinstance(
                self.baseline_prediction_root, (str, os.PathLike)
            ):
                raise ValueError(
                    "proposal-cache baseline_prediction_root must be a path"
                )
            object.__setattr__(
                self,
                "baseline_prediction_root",
                Path(
                    os.path.abspath(
                        os.path.expanduser(os.fspath(self.baseline_prediction_root))
                    )
                ),
            )
        if self.mode == "replay" and self.baseline_prediction_root is None:
            raise ValueError(
                "proposal-cache replay requires baseline_prediction_root"
            )
        if not isinstance(self.require_index, (bool, np.bool_)):
            raise ValueError("proposal-cache require_index must be boolean")
        object.__setattr__(self, "require_index", bool(self.require_index))
        if self.require_index and self.mode != "replay":
            raise ValueError("proposal-cache require_index is replay-only")

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> "ProposalCacheConfig":
        if not isinstance(mapping, Mapping):
            raise ValueError("lifting.proposal_cache must be a mapping")
        allowed = {
            "mode",
            "root",
            "namespace",
            "baseline_prediction_root",
            "require_index",
        }
        unknown = sorted(set(mapping) - allowed)
        if unknown:
            raise ValueError(
                "unknown lifting.proposal_cache key(s): " + ", ".join(unknown)
            )
        mode_raw = mapping.get("mode", "disabled")
        if not isinstance(mode_raw, str):
            raise ValueError("lifting.proposal_cache.mode must be a string")
        mode = mode_raw.lower()
        if mode not in ("disabled", "record", "replay"):
            raise ValueError(
                "lifting.proposal_cache.mode must be disabled, record, or replay"
            )
        root_raw = mapping.get("root", "cache/cutr_postfilter")
        if isinstance(root_raw, bool) or not isinstance(
            root_raw, (str, os.PathLike)
        ):
            raise ValueError("lifting.proposal_cache.root must be a path")
        root = Path(
            os.path.abspath(
                os.path.expanduser(
                    os.fspath(root_raw)
                )
            )
        )
        namespace_raw = mapping.get("namespace", "")
        if not isinstance(namespace_raw, str):
            raise ValueError("lifting.proposal_cache.namespace must be a string")
        namespace = namespace_raw.strip()
        if mode != "disabled" and not namespace:
            raise ValueError(
                "lifting.proposal_cache.namespace is required for record/replay"
            )
        if (
            mode != "disabled"
            and (
                namespace in (".", "..")
                or Path(namespace).name != namespace
                or os.sep in namespace
            )
        ):
            raise ValueError(
                "lifting.proposal_cache.namespace must be one path component"
            )
        baseline_raw = mapping.get("baseline_prediction_root", "")
        if isinstance(baseline_raw, bool) or not isinstance(
            baseline_raw, (str, os.PathLike)
        ):
            raise ValueError(
                "lifting.proposal_cache.baseline_prediction_root must be a path"
            )
        baseline_text = os.fspath(baseline_raw).strip()
        baseline = (
            None
            if not baseline_text
            else Path(os.path.abspath(os.path.expanduser(baseline_text)))
        )
        require_index = mapping.get("require_index", False)
        if not isinstance(require_index, (bool, np.bool_)):
            raise ValueError("lifting.proposal_cache.require_index must be boolean")
        return cls(
            mode=mode,
            root=root,
            namespace=namespace,
            baseline_prediction_root=baseline,
            require_index=bool(require_index),
        )


class ProposalCache:
    """Record once and replay byte-identical ordered CuTR proposal rows."""

    def __init__(self, config: ProposalCacheConfig, device: torch.device):
        self.config = config
        self.device = torch.device(device)
        self._records: Dict[str, Dict[int, Dict[str, Any]]] = {}
        self._manifests: Dict[str, Dict[str, Any]] = {}
        self._consumed: Dict[str, list[int]] = {}
        self._schedules: Dict[str, Dict[str, int]] = {}
        self._scene_locks: Dict[str, int] = {}
        self._verified_index: Optional[Dict[str, Any]] = None
        required_environment = {
            "record": "BOXFUSION_PROPOSAL_CACHE_PRODUCER_FINGERPRINT",
            "replay": "BOXFUSION_PROPOSAL_CACHE_EXPECTED_FINGERPRINT",
        }.get(self.mode)
        if required_environment and not os.environ.get(
            required_environment, ""
        ).strip():
            raise ProposalCacheError(
                f"Missing required environment variable {required_environment}"
            )
        self._fingerprint = (
            ""
            if required_environment is None
            else _strict_fingerprint(
                required_environment,
                os.environ.get(required_environment, ""),
            )
        )

    @property
    def mode(self) -> str:
        return self.config.mode

    @property
    def is_record(self) -> bool:
        return self.mode == "record"

    @property
    def is_replay(self) -> bool:
        return self.mode == "replay"

    def _scene_root(self, scene_id: str) -> Path:
        scene_id = _strict_scene_id(scene_id)
        return self.config.root / self.config.namespace / scene_id

    def _namespace_root(self) -> Path:
        return self.config.root / self.config.namespace

    def _locks_root(self) -> Path:
        return self._namespace_root() / ".locks"

    def _index_records_root(self) -> Path:
        return self._namespace_root() / "index_records"

    def _index_path(self) -> Path:
        return self._namespace_root() / "index.json"

    def _frame_path(self, scene_id: str, frame_id: int) -> Path:
        frame_id = _strict_int("frame_id", frame_id)
        return self._scene_root(scene_id) / f"frame_{frame_id:06d}.pt"

    def _manifest_path(self, scene_id: str) -> Path:
        return self._scene_root(scene_id) / "manifest.json"

    def _acquire_scene_lock(self, scene_id: str) -> None:
        if not self.is_record or scene_id in self._scene_locks:
            return
        if self._index_path().exists():
            raise ProposalCacheError(
                f"Proposal-cache namespace is already sealed: {self.config.namespace}"
            )
        locks_root = self._locks_root()
        locks_root.mkdir(parents=True, exist_ok=True)
        lock_path = locks_root / f"{scene_id}.lock"
        descriptor = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            os.close(descriptor)
            raise ProposalCacheError(
                f"Another writer owns proposal-cache scene {scene_id}"
            ) from error
        if self._manifest_path(scene_id).exists():
            fcntl.flock(descriptor, fcntl.LOCK_UN)
            os.close(descriptor)
            raise ProposalCacheError(
                f"Proposal-cache scene is already sealed: {scene_id}"
            )
        self._scene_locks[scene_id] = descriptor

    def _release_scene_lock(self, scene_id: str) -> None:
        descriptor = self._scene_locks.pop(scene_id, None)
        if descriptor is not None:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
            os.close(descriptor)

    def close(self) -> None:
        for scene_id in tuple(self._scene_locks):
            self._release_scene_lock(scene_id)

    @staticmethod
    def _expected_frame_ids(dataset_length: int, gap: int) -> list[int]:
        dataset_length = _strict_int("dataset_length", dataset_length, minimum=1)
        gap = _strict_int("gap", gap, minimum=1)
        final_native_frame = max(0, dataset_length - gap - 1)
        return list(range(0, final_native_frame + 1, gap))

    def bind_scene(self, scene_id: str, *, dataset_length: int, gap: int) -> None:
        scene_id = _strict_scene_id(scene_id)
        schedule = {
            "dataset_length": _strict_int(
                "dataset_length", dataset_length, minimum=1
            ),
            "gap": _strict_int("gap", gap, minimum=1),
        }
        previous = self._schedules.setdefault(scene_id, schedule)
        if previous != schedule:
            raise ProposalCacheError(
                f"Scene schedule changed for {scene_id}: {previous} != {schedule}"
            )
        if self.is_record:
            self._acquire_scene_lock(scene_id)

    def _capture_rng_state(self) -> Dict[str, Any]:
        python_state = random.getstate()
        numpy_state = np.random.get_state()
        state: Dict[str, Any] = {
            "torch_cpu": torch.get_rng_state().cpu().clone(),
            "torch_cuda": None,
            "python": {
                "version": int(python_state[0]),
                "state": torch.tensor(python_state[1], dtype=torch.int64),
                "gauss": python_state[2],
            },
            "numpy": {
                "algorithm": str(numpy_state[0]),
                "state": torch.from_numpy(numpy_state[1].astype(np.int64)),
                "position": int(numpy_state[2]),
                "has_gauss": int(numpy_state[3]),
                "cached_gaussian": float(numpy_state[4]),
            },
        }
        if self.device.type == "cuda":
            state["torch_cuda"] = torch.cuda.get_rng_state(self.device).cpu().clone()
        return state

    def _restore_rng_state(self, state: Mapping[str, Any]) -> None:
        _rng_contract(state)
        cuda_state = state.get("torch_cuda")
        if cuda_state is not None and self.device.type != "cuda":
            raise ProposalCacheError(
                "A CUDA proposal cache cannot be replayed on a CPU device"
            )
        python_state = state["python"]
        python_tuple = (
            int(python_state["version"]),
            tuple(int(value) for value in python_state["state"].tolist()),
            python_state["gauss"],
        )
        numpy_state = state["numpy"]
        numpy_tuple = (
            str(numpy_state["algorithm"]),
            numpy_state["state"].cpu().numpy().astype(np.uint32),
            int(numpy_state["position"]),
            int(numpy_state["has_gauss"]),
            float(numpy_state["cached_gaussian"]),
        )

        # Validate every backend on isolated generators before mutating any
        # process-global RNG state.  A malformed cache therefore fails closed
        # without leaving Python/NumPy/Torch at different logical points.
        try:
            random.Random().setstate(python_tuple)
            np.random.RandomState().set_state(numpy_tuple)
            torch.Generator(device="cpu").set_state(state["torch_cpu"].cpu())
            if cuda_state is not None:
                torch.Generator(device=self.device).set_state(cuda_state.cpu())
        except Exception as error:
            raise ProposalCacheError("Invalid cached RNG backend state") from error

        random.setstate(python_tuple)
        np.random.set_state(numpy_tuple)
        torch.set_rng_state(state["torch_cpu"].cpu())
        if cuda_state is not None:
            torch.cuda.set_rng_state(cuda_state.cpu(), self.device)

    def _serialize(
        self,
        instances: Instances3D,
        *,
        scene_id: str,
        frame_id: int,
        attempt_id: str,
        input_signature: Mapping[str, str],
        rng_state: Mapping[str, Any],
    ) -> Dict[str, Any]:
        if attempt_id not in VALID_ATTEMPTS:
            raise ProposalCacheError(f"Invalid CuTR attempt ID: {attempt_id}")
        field_names = tuple(instances.get_fields().keys())
        if field_names != EXPECTED_FIELDS:
            raise ProposalCacheError(
                "Unexpected CuTR field order/schema: "
                f"expected={EXPECTED_FIELDS}, actual={field_names}"
            )
        fields: Dict[str, Any] = {}
        metadata: Dict[str, Any] = {}
        for name in EXPECTED_FIELDS:
            value = instances.get(name)
            if name == "pred_boxes_3d":
                if not isinstance(value, GeneralInstance3DBoxes):
                    raise ProposalCacheError(
                        "pred_boxes_3d is not GeneralInstance3DBoxes"
                    )
                if not isinstance(value.dof, BoxDOF):
                    raise ProposalCacheError("pred_boxes_3d has an invalid BoxDOF")
                fields[name] = {
                    "tensor": _canonical_tensor(value.tensor),
                    "rotation": _canonical_tensor(value.R),
                    "box_dim": _strict_int(
                        "CuTR box_dim", value.box_dim, minimum=1
                    ),
                    "dof": value.dof.name,
                }
                metadata[name] = {
                    "tensor": _tensor_metadata(value.tensor),
                    "rotation": _tensor_metadata(value.R),
                    "box_dim": _strict_int(
                        "CuTR box_dim", value.box_dim, minimum=1
                    ),
                    "dof": value.dof.name,
                }
            elif isinstance(value, torch.Tensor):
                fields[name] = _canonical_tensor(value)
                metadata[name] = _tensor_metadata(value)
            else:
                raise ProposalCacheError(
                    f"Unsupported cached field {name}: {type(value)}"
                )
        count = _validate_proposal_shapes(instances)
        for name in EXPECTED_FIELDS:
            if len(instances.get(name)) != count:
                raise ProposalCacheError(
                    f"CuTR field {name} has an inconsistent row count"
                )
        canonical_rng_state = {
            "torch_cpu": _canonical_tensor(rng_state["torch_cpu"]),
            "torch_cuda": (
                None
                if rng_state.get("torch_cuda") is None
                else _canonical_tensor(rng_state["torch_cuda"])
            ),
            "python": {
                "version": int(rng_state["python"]["version"]),
                "state": _canonical_tensor(rng_state["python"]["state"]),
                "gauss": rng_state["python"]["gauss"],
            },
            "numpy": {
                "algorithm": str(rng_state["numpy"]["algorithm"]),
                "state": _canonical_tensor(rng_state["numpy"]["state"]),
                "position": int(rng_state["numpy"]["position"]),
                "has_gauss": int(rng_state["numpy"]["has_gauss"]),
                "cached_gaussian": float(
                    rng_state["numpy"]["cached_gaussian"]
                ),
            },
        }
        if (
            not isinstance(instances.image_size, (list, tuple))
            or len(instances.image_size) != 2
        ):
            raise ProposalCacheError("CuTR image_size must contain height and width")
        image_size = tuple(
            _strict_int("CuTR image_size", value, minimum=1)
            for value in instances.image_size
        )
        protected_hashes = protected_proposal_hashes(instances)
        geometry_sha256 = geometry_hash(instances)
        proposal_contract_sha256 = _proposal_contract_sha256_from_parts(
            image_size=image_size,
            field_names=field_names,
            field_metadata=metadata,
            count=count,
            protected_hashes=protected_hashes,
            geometry_sha256=geometry_sha256,
        )
        return {
            "schema": SCHEMA,
            "scene_id": scene_id,
            "frame_id": frame_id,
            "namespace": self.config.namespace,
            "producer_fingerprint": self._fingerprint,
            "image_size": image_size,
            "field_names": list(field_names),
            "fields": fields,
            "field_metadata": metadata,
            "count": count,
            "attempt_id": attempt_id,
            "input_signature": dict(input_signature),
            "rng_state": canonical_rng_state,
            "rng_sha256": _rng_sha256(canonical_rng_state),
            "protected_hashes": protected_hashes,
            "geometry_sha256": geometry_sha256,
            "proposal_contract_sha256": proposal_contract_sha256,
        }

    def _validate_tensor(
        self,
        value: Any,
        metadata: Mapping[str, Any],
        *,
        field_name: str,
    ) -> torch.Tensor:
        if not isinstance(value, torch.Tensor):
            raise ProposalCacheError(f"Cached {field_name} is not a tensor")
        if (
            not isinstance(metadata, Mapping)
            or set(metadata) != {"dtype", "shape", "sha256"}
            or not isinstance(metadata.get("dtype"), str)
            or not isinstance(metadata.get("shape"), list)
            or any(
                isinstance(item, (bool, np.bool_))
                or not isinstance(item, Integral)
                or int(item) < 0
                for item in metadata.get("shape", [])
            )
        ):
            raise ProposalCacheError(
                f"Cached tensor metadata schema is invalid for {field_name}"
            )
        _strict_sha256(
            f"cached tensor hash for {field_name}", metadata.get("sha256")
        )
        actual = _tensor_metadata(value)
        if actual != dict(metadata):
            raise ProposalCacheError(
                f"Cached tensor metadata/hash mismatch for {field_name}"
            )
        return value

    def _deserialize(
        self,
        payload: Mapping[str, Any],
        *,
        expected_scene_id: Optional[str] = None,
        expected_frame_id: Optional[int] = None,
    ) -> Instances3D:
        if not isinstance(payload, Mapping) or set(payload) != _PAYLOAD_KEYS:
            raise ProposalCacheError("Cached payload schema keys are invalid")
        if payload.get("schema") != SCHEMA:
            raise ProposalCacheError(
                f"Unexpected proposal-cache schema: {payload.get('schema')}"
            )
        scene_id = _strict_scene_id(payload.get("scene_id"))
        frame_id = _strict_int("payload frame_id", payload.get("frame_id"))
        if expected_scene_id is not None and scene_id != expected_scene_id:
            raise ProposalCacheError("Cached payload scene_id mismatch")
        if expected_frame_id is not None and frame_id != expected_frame_id:
            raise ProposalCacheError("Cached payload frame_id mismatch")
        if payload.get("namespace") != self.config.namespace:
            raise ProposalCacheError("Cached payload namespace mismatch")
        if payload.get("producer_fingerprint") != self._fingerprint:
            raise ProposalCacheError("Cached payload producer fingerprint mismatch")
        if payload.get("attempt_id") not in VALID_ATTEMPTS:
            raise ProposalCacheError("Cached payload attempt_id is invalid")
        _strict_hash_mapping(
            "cached input signature",
            payload.get("input_signature"),
            keys=_INPUT_FIELDS,
        )
        _strict_hash_mapping(
            "cached protected hashes",
            payload.get("protected_hashes"),
            keys=(
                "pred_boxes",
                "scores",
                "pred_classes",
                "pred_logits",
                "object_desc",
            ),
        )
        _strict_sha256("cached geometry hash", payload.get("geometry_sha256"))
        _strict_sha256(
            "cached proposal contract hash",
            payload.get("proposal_contract_sha256"),
        )
        _strict_sha256("cached RNG hash", payload.get("rng_sha256"))
        count = _strict_int("cached proposal count", payload.get("count"))
        field_names = tuple(payload.get("field_names", ()))
        if field_names != EXPECTED_FIELDS:
            raise ProposalCacheError(
                f"Cached field schema mismatch: {field_names}"
            )
        fields = payload.get("fields")
        metadata = payload.get("field_metadata")
        if not isinstance(fields, dict) or tuple(fields.keys()) != EXPECTED_FIELDS:
            raise ProposalCacheError("Cached field payload is incomplete/reordered")
        if not isinstance(metadata, dict) or tuple(metadata.keys()) != EXPECTED_FIELDS:
            raise ProposalCacheError("Cached field metadata is incomplete/reordered")

        image_size = payload["image_size"]
        if (
            not isinstance(image_size, (tuple, list))
            or len(image_size) != 2
            or any(
                isinstance(item, (bool, np.bool_))
                or not isinstance(item, Integral)
                or int(item) <= 0
                for item in image_size
            )
        ):
            raise ProposalCacheError("Cached image_size is invalid")
        instances = Instances3D(tuple(int(item) for item in image_size))
        for name in EXPECTED_FIELDS:
            value = fields[name]
            field_metadata = metadata[name]
            if name == "pred_boxes_3d":
                if (
                    not isinstance(value, Mapping)
                    or set(value) != {"tensor", "rotation", "box_dim", "dof"}
                    or not isinstance(field_metadata, Mapping)
                    or set(field_metadata)
                    != {"tensor", "rotation", "box_dim", "dof"}
                ):
                    raise ProposalCacheError("Cached 3D-box payload is invalid")
                tensor = self._validate_tensor(
                    value["tensor"],
                    field_metadata["tensor"],
                    field_name="pred_boxes_3d.tensor",
                )
                rotation = self._validate_tensor(
                    value["rotation"],
                    field_metadata["rotation"],
                    field_name="pred_boxes_3d.rotation",
                )
                value_box_dim = _strict_int(
                    "cached box_dim", value["box_dim"], minimum=1
                )
                metadata_box_dim = _strict_int(
                    "cached metadata box_dim",
                    field_metadata["box_dim"],
                    minimum=1,
                )
                if (
                    value_box_dim != metadata_box_dim
                    or not isinstance(value["dof"], str)
                    or not isinstance(field_metadata["dof"], str)
                    or value["dof"] != field_metadata["dof"]
                ):
                    raise ProposalCacheError("Cached 3D-box metadata changed")
                try:
                    dof = BoxDOF[str(value["dof"])]
                except KeyError as error:
                    raise ProposalCacheError("Unknown cached box DOF") from error
                value = GeneralInstance3DBoxes(
                    tensor.to(self.device),
                    rotation.to(self.device),
                    box_dim=value_box_dim,
                    dof=dof,
                )
            else:
                if not isinstance(field_metadata, Mapping):
                    raise ProposalCacheError(
                        f"Cached tensor metadata is invalid for {name}"
                    )
                value = self._validate_tensor(
                    value,
                    field_metadata,
                    field_name=name,
                ).to(self.device)
            instances.set(name, value)

        if _validate_proposal_shapes(instances) != count:
            raise ProposalCacheError("Cached proposal count changed on load")
        if protected_proposal_hashes(instances) != payload["protected_hashes"]:
            raise ProposalCacheError("Cached protected fields changed on load")
        if geometry_hash(instances) != payload["geometry_sha256"]:
            raise ProposalCacheError("Cached geometry changed on load")
        actual_contract = _proposal_contract_sha256_from_parts(
            image_size=image_size,
            field_names=field_names,
            field_metadata=metadata,
            count=count,
            protected_hashes=payload["protected_hashes"],
            geometry_sha256=payload["geometry_sha256"],
        )
        if actual_contract != payload["proposal_contract_sha256"]:
            raise ProposalCacheError("Cached proposal contract hash mismatch")
        if _rng_sha256(payload["rng_state"]) != payload["rng_sha256"]:
            raise ProposalCacheError("Cached RNG state hash mismatch")
        return instances

    @staticmethod
    def _load_payload(
        path: Path,
        *,
        expected_sha256: Optional[str] = None,
    ) -> Dict[str, Any]:
        try:
            with path.open("rb") as handle:
                if expected_sha256 is not None:
                    _strict_sha256("expected payload hash", expected_sha256)
                    digest = hashlib.sha256()
                    while True:
                        block = handle.read(8 * 1024 * 1024)
                        if not block:
                            break
                        digest.update(block)
                    if digest.hexdigest() != expected_sha256:
                        raise ProposalCacheError(
                            f"Cached keyframe hash mismatch: {path}"
                        )
                    handle.seek(0)
                payload = torch.load(
                    handle,
                    map_location="cpu",
                    weights_only=True,
                )
        except ProposalCacheError:
            raise
        except Exception as error:
            raise ProposalCacheError(
                f"Could not safely load proposal cache payload: {path}"
            ) from error
        if not isinstance(payload, dict):
            raise ProposalCacheError(f"Invalid proposal cache payload: {path}")
        return payload

    def record(
        self,
        scene_id: str,
        frame_id: int,
        instances: Instances3D,
        *,
        attempt_id: str,
        inputs: Mapping[str, Any],
    ) -> Instances3D:
        """Write an immutable event and return its canonical round-trip copy."""

        if not self.is_record:
            raise ProposalCacheError("record() called outside record mode")
        scene_id = _strict_scene_id(scene_id)
        frame_id = _strict_int("frame_id", frame_id)
        schedule = self._schedules.get(scene_id)
        if schedule is None or scene_id not in self._scene_locks:
            raise ProposalCacheError(
                f"Scene must be bound and writer-locked before record: {scene_id}"
            )
        scene_records = self._records.setdefault(scene_id, {})
        if frame_id in scene_records:
            raise ProposalCacheError(
                f"Duplicate cached frame: {scene_id}/{frame_id}"
            )
        expected_frames = self._expected_frame_ids(
            schedule["dataset_length"], schedule["gap"]
        )
        expected_index = len(scene_records)
        if (
            expected_index >= len(expected_frames)
            or expected_frames[expected_index] != frame_id
        ):
            raise ProposalCacheError(
                f"Out-of-order/unexpected record frame: {scene_id}/{frame_id}"
            )
        rng_state = self._capture_rng_state()
        input_signature = _input_signature(inputs)
        payload = self._serialize(
            instances,
            scene_id=scene_id,
            frame_id=frame_id,
            attempt_id=attempt_id,
            input_signature=input_signature,
            rng_state=rng_state,
        )

        scene_root = self._scene_root(scene_id)
        scene_root.mkdir(parents=True, exist_ok=True)
        final_path = self._frame_path(scene_id, frame_id)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{final_path.name}.",
            suffix=".tmp",
            dir=scene_root,
        )
        os.close(descriptor)
        temporary_path = Path(temporary_name)
        try:
            torch.save(payload, temporary_path)
            with temporary_path.open("rb") as handle:
                os.fsync(handle.fileno())
            _install_create_only(temporary_path, final_path)
            final_path.chmod(0o444)
        finally:
            if temporary_path.exists():
                temporary_path.unlink()

        installed_sha256 = _sha256_file(final_path)
        roundtrip_payload = self._load_payload(
            final_path,
            expected_sha256=installed_sha256,
        )
        canonical_instances = self._deserialize(
            roundtrip_payload,
            expected_scene_id=scene_id,
            expected_frame_id=frame_id,
        )
        if (
            roundtrip_payload["proposal_contract_sha256"]
            != payload["proposal_contract_sha256"]
            or roundtrip_payload["rng_sha256"] != payload["rng_sha256"]
        ):
            raise ProposalCacheError("Recorded payload changed during installation")
        self._restore_rng_state(roundtrip_payload["rng_state"])
        scene_records[frame_id] = {
            "frame_id": frame_id,
            "attempt_id": attempt_id,
            "count": int(payload["count"]),
            "sha256": installed_sha256,
            "input_signature": input_signature,
            "protected_hashes": payload["protected_hashes"],
            "geometry_sha256": payload["geometry_sha256"],
            "proposal_contract_sha256": payload[
                "proposal_contract_sha256"
            ],
            "rng_sha256": payload["rng_sha256"],
        }
        return canonical_instances

    def finalize(self, scene_id: str, prediction_path: str | Path) -> Path:
        """Seal a recorded scene only after its prediction was saved."""

        if not self.is_record:
            raise ProposalCacheError("finalize() called outside record mode")
        scene_id = _strict_scene_id(scene_id)
        records = self._records.get(scene_id)
        if not records:
            raise ProposalCacheError(
                f"No proposal-cache records were written for {scene_id}"
            )
        schedule = self._schedules.get(scene_id)
        if schedule is None:
            raise ProposalCacheError(f"Scene schedule was not bound: {scene_id}")
        if scene_id not in self._scene_locks:
            raise ProposalCacheError(f"Scene writer lock is not held: {scene_id}")
        expected_frames = self._expected_frame_ids(
            schedule["dataset_length"], schedule["gap"]
        )
        actual_frames = sorted(records)
        if actual_frames != expected_frames:
            raise ProposalCacheError(
                f"Incomplete proposal-cache scene {scene_id}: "
                f"expected={expected_frames}, recorded={actual_frames}"
            )
        prediction_path = Path(prediction_path)
        if not prediction_path.is_file():
            raise ProposalCacheError(
                f"Cannot seal cache before prediction exists: {prediction_path}"
            )
        expected_prediction_name = f"{scene_id}_boxes.pkl"
        if prediction_path.name != expected_prediction_name:
            raise ProposalCacheError(
                "Prediction filename does not match proposal-cache scene: "
                f"expected={expected_prediction_name}, actual={prediction_path.name}"
            )
        producer_fingerprint = self._fingerprint
        ordered_records = [records[key] for key in sorted(records)]
        manifest = {
            "schema": SCHEMA,
            "scene_id": scene_id,
            "namespace": self.config.namespace,
            "producer_fingerprint": producer_fingerprint,
            "schedule": {
                **schedule,
                # This string deliberately freezes the upstream early-exit
                # behavior instead of silently changing BoxFusion traversal.
                "terminal_policy": TERMINAL_POLICY,
            },
            "records": ordered_records,
            "recorded_frame_ids": [row["frame_id"] for row in ordered_records],
            "record_count": len(ordered_records),
            "proposal_count": sum(int(row["count"]) for row in ordered_records),
            "nonempty_frame_count": sum(
                int(row["count"] > 0) for row in ordered_records
            ),
            "prediction_file": prediction_path.name,
            "prediction_sha256": _sha256_file(prediction_path),
        }
        final_path = self._manifest_path(scene_id)
        _atomic_json_write(final_path, manifest)
        manifest_sha256 = _sha256_file(final_path)
        index_records_root = self._index_records_root()
        index_records_root.mkdir(parents=True, exist_ok=True)
        receipt_path = index_records_root / f"{scene_id}.json"
        _atomic_json_write(
            receipt_path,
            {
                "schema": INDEX_SCHEMA,
                "scene_id": scene_id,
                "namespace": self.config.namespace,
                "producer_fingerprint": producer_fingerprint,
                "manifest": str(final_path.relative_to(self._namespace_root())),
                "manifest_sha256": manifest_sha256,
                "prediction_sha256": manifest["prediction_sha256"],
                "record_count": manifest["record_count"],
                "proposal_count": manifest["proposal_count"],
            },
        )
        self._scene_root(scene_id).chmod(0o555)
        self._release_scene_lock(scene_id)
        return final_path

    def seal_index(self, scene_ids: list[str] | tuple[str, ...]) -> Path:
        """Create the immutable namespace index after every scene is sealed."""

        if not self.is_record:
            raise ProposalCacheError("seal_index() called outside record mode")
        if self._index_path().exists():
            raise ProposalCacheError(
                f"Proposal-cache namespace is already sealed: {self.config.namespace}"
            )
        if self._scene_locks:
            raise ProposalCacheError(
                "Cannot seal proposal-cache index while a scene writer is active"
            )
        if not isinstance(scene_ids, (list, tuple)) or not scene_ids:
            raise ProposalCacheError("scene_ids must be a non-empty ordered list")
        normalized = [_strict_scene_id(value) for value in scene_ids]
        if len(set(normalized)) != len(normalized):
            raise ProposalCacheError("scene_ids must be unique")
        locks_root = self._locks_root()
        locks_root.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(
            locks_root / "index.lock", os.O_CREAT | os.O_RDWR, 0o600
        )
        try:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise ProposalCacheError("Another writer is sealing the index") from error
            rows = []
            for scene_id in normalized:
                receipt_path = self._index_records_root() / f"{scene_id}.json"
                if not receipt_path.is_file():
                    raise ProposalCacheError(
                        f"Missing proposal-cache index receipt: {scene_id}"
                    )
                receipt = _load_json_file(
                    receipt_path,
                    label="proposal-cache index receipt",
                )
                receipt = _validate_index_receipt(
                    receipt,
                    scene_id=scene_id,
                    namespace=self.config.namespace,
                    producer_fingerprint=self._fingerprint,
                )
                manifest_path = self._manifest_path(scene_id)
                if (
                    not manifest_path.is_file()
                    or _sha256_file(manifest_path)
                    != receipt.get("manifest_sha256")
                ):
                    raise ProposalCacheError(
                        f"Manifest differs from index receipt: {scene_id}"
                    )
                sealed_manifest = _load_json_file(
                    manifest_path,
                    expected_sha256=receipt["manifest_sha256"],
                    label="proposal-cache index manifest",
                )
                if (
                    receipt["prediction_sha256"]
                    != sealed_manifest.get("prediction_sha256")
                    or receipt["record_count"]
                    != sealed_manifest.get("record_count")
                    or receipt["proposal_count"]
                    != sealed_manifest.get("proposal_count")
                ):
                    raise ProposalCacheError(
                        f"Index receipt summary differs from manifest: {scene_id}"
                    )
                rows.append(receipt)
            aggregate = _canonical_json_sha256({"scenes": rows})
            index = {
                "schema": INDEX_SCHEMA,
                "namespace": self.config.namespace,
                "producer_fingerprint": self._fingerprint,
                "scene_ids": normalized,
                "scene_count": len(normalized),
                "scenes": rows,
                "aggregate_sha256": aggregate,
            }
            path = self._index_path()
            _atomic_json_write(path, index)
            self._index_records_root().chmod(0o555)
            _fsync_directory(self._namespace_root())
            self._namespace_root().chmod(0o555)
            _fsync_directory(self._namespace_root().parent)
            return path
        finally:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
            os.close(descriptor)

    def verify_index(
        self, expected_scene_ids: Optional[list[str] | tuple[str, ...]] = None
    ) -> Dict[str, Any]:
        """Validate the sealed namespace index and every referenced manifest."""

        if self._verified_index is not None:
            index = self._verified_index
        else:
            path = self._index_path()
            if not path.is_file():
                raise ProposalCacheError(f"Missing proposal-cache index: {path}")
            index = _load_json_file(path, label="proposal-cache index")
            required = {
                "schema",
                "namespace",
                "producer_fingerprint",
                "scene_ids",
                "scene_count",
                "scenes",
                "aggregate_sha256",
            }
            if (
                not isinstance(index, dict)
                or set(index) != required
                or index.get("schema") != INDEX_SCHEMA
                or index.get("namespace") != self.config.namespace
                or index.get("producer_fingerprint") != self._fingerprint
            ):
                raise ProposalCacheError(f"Invalid proposal-cache index: {path}")
            scene_ids = index.get("scene_ids")
            rows = index.get("scenes")
            if (
                not isinstance(scene_ids, list)
                or not isinstance(rows, list)
                or any(not isinstance(row, dict) for row in rows)
                or len(rows) != len(scene_ids)
            ):
                raise ProposalCacheError("Proposal-cache index contents are inconsistent")
            normalized_scene_ids = [_strict_scene_id(value) for value in scene_ids]
            if (
                len(normalized_scene_ids) != len(set(normalized_scene_ids))
                or _strict_int("index scene_count", index.get("scene_count"))
                != len(normalized_scene_ids)
                or [row.get("scene_id") for row in rows] != normalized_scene_ids
                or _canonical_json_sha256({"scenes": rows})
                != _strict_sha256(
                    "index aggregate hash", index.get("aggregate_sha256")
                )
            ):
                raise ProposalCacheError("Proposal-cache index contents are inconsistent")
            for scene_id, row in zip(normalized_scene_ids, rows):
                row = _validate_index_receipt(
                    row,
                    scene_id=scene_id,
                    namespace=self.config.namespace,
                    producer_fingerprint=self._fingerprint,
                )
                manifest_path = self._manifest_path(scene_id)
                if (
                    not manifest_path.is_file()
                    or _sha256_file(manifest_path)
                    != row["manifest_sha256"]
                ):
                    raise ProposalCacheError(
                        f"Proposal-cache index manifest mismatch: {scene_id}"
                    )
                sealed_manifest = _load_json_file(
                    manifest_path,
                    expected_sha256=row["manifest_sha256"],
                    label="proposal-cache index manifest",
                )
                if (
                    row["prediction_sha256"]
                    != sealed_manifest.get("prediction_sha256")
                    or row["record_count"]
                    != sealed_manifest.get("record_count")
                    or row["proposal_count"]
                    != sealed_manifest.get("proposal_count")
                ):
                    raise ProposalCacheError(
                        f"Proposal-cache index summary mismatch: {scene_id}"
                    )
            self._verified_index = index
        if expected_scene_ids is not None:
            if not isinstance(expected_scene_ids, (list, tuple)):
                raise ProposalCacheError(
                    "expected_scene_ids must be an ordered list or tuple"
                )
            normalized = [_strict_scene_id(value) for value in expected_scene_ids]
            if len(normalized) != len(set(normalized)):
                raise ProposalCacheError("expected_scene_ids must be unique")
            if index["scene_ids"] != normalized:
                raise ProposalCacheError("Proposal-cache index scene list mismatch")
        return copy.deepcopy(index)

    def _load_manifest(self, scene_id: str) -> Dict[str, Any]:
        scene_id = _strict_scene_id(scene_id)
        if scene_id in self._manifests:
            return self._manifests[scene_id]
        expected_manifest_sha256: Optional[str] = None
        if self.config.require_index:
            index = self.verify_index()
            if scene_id not in index["scene_ids"]:
                raise ProposalCacheError(
                    f"Scene is absent from proposal-cache index: {scene_id}"
                )
            expected_manifest_sha256 = next(
                row["manifest_sha256"]
                for row in index["scenes"]
                if row["scene_id"] == scene_id
            )
        path = self._manifest_path(scene_id)
        if not path.is_file():
            raise ProposalCacheError(f"Missing cache manifest: {path}")
        manifest = _load_json_file(
            path,
            expected_sha256=expected_manifest_sha256,
            label="cache manifest",
        )
        manifest_keys = {
            "schema",
            "scene_id",
            "namespace",
            "producer_fingerprint",
            "schedule",
            "records",
            "recorded_frame_ids",
            "record_count",
            "proposal_count",
            "nonempty_frame_count",
            "prediction_file",
            "prediction_sha256",
        }
        if (
            not isinstance(manifest, dict)
            or set(manifest) != manifest_keys
            or manifest.get("schema") != SCHEMA
            or manifest.get("scene_id") != scene_id
            or manifest.get("namespace") != self.config.namespace
        ):
            raise ProposalCacheError(f"Invalid cache manifest: {path}")
        expected_fingerprint = self._fingerprint
        if (
            not expected_fingerprint
            or manifest.get("producer_fingerprint") != expected_fingerprint
        ):
            raise ProposalCacheError(
                "Proposal-cache producer fingerprint mismatch"
            )
        records = manifest.get("records", [])
        required_record_keys = {
            "frame_id",
            "attempt_id",
            "count",
            "sha256",
            "input_signature",
            "protected_hashes",
            "geometry_sha256",
            "proposal_contract_sha256",
            "rng_sha256",
        }
        if not isinstance(records, list) or any(
            not isinstance(row, dict) or set(row) != required_record_keys
            for row in records
        ):
            raise ProposalCacheError("Invalid proposal-cache record schema")
        derived_frames = [
            _strict_int("manifest frame_id", row["frame_id"])
            for row in records
        ]
        counts = [
            _strict_int("manifest proposal count", row["count"])
            for row in records
        ]
        by_frame = {frame: row for frame, row in zip(derived_frames, records)}
        derived_proposals = sum(counts)
        derived_nonempty = sum(int(count > 0) for count in counts)
        for row in records:
            if row["attempt_id"] not in VALID_ATTEMPTS:
                raise ProposalCacheError("Invalid proposal-cache record values")
            _strict_hash_mapping(
                "manifest input signature",
                row["input_signature"],
                keys=_INPUT_FIELDS,
            )
            _strict_hash_mapping(
                "manifest protected hashes",
                row["protected_hashes"],
                keys=(
                    "pred_boxes",
                    "scores",
                    "pred_classes",
                    "pred_logits",
                    "object_desc",
                ),
            )
            for name in (
                "sha256",
                "geometry_sha256",
                "proposal_contract_sha256",
                "rng_sha256",
            ):
                _strict_sha256(f"manifest {name}", row[name])
        record_count = _strict_int(
            "manifest record_count", manifest.get("record_count")
        )
        proposal_count = _strict_int(
            "manifest proposal_count", manifest.get("proposal_count")
        )
        nonempty_frame_count = _strict_int(
            "manifest nonempty_frame_count",
            manifest.get("nonempty_frame_count"),
        )
        if (
            len(by_frame) != len(records)
            or derived_frames != sorted(by_frame)
            or record_count != len(records)
            or manifest.get("recorded_frame_ids") != derived_frames
            or proposal_count != derived_proposals
            or nonempty_frame_count != derived_nonempty
        ):
            raise ProposalCacheError("Invalid/duplicate frame IDs in cache manifest")
        schedule = self._schedules.get(scene_id)
        manifest_schedule = manifest.get("schedule", {})
        if (
            schedule is None
            or not isinstance(manifest_schedule, Mapping)
            or set(manifest_schedule)
            != {"dataset_length", "gap", "terminal_policy"}
            or _strict_int(
                "manifest dataset_length",
                manifest_schedule.get("dataset_length"),
                minimum=1,
            )
            != schedule["dataset_length"]
            or _strict_int(
                "manifest gap", manifest_schedule.get("gap"), minimum=1
            )
            != schedule["gap"]
            or manifest_schedule.get("terminal_policy") != TERMINAL_POLICY
        ):
            raise ProposalCacheError(
                f"Current scene schedule differs from cache: {scene_id}"
            )
        if (
            not isinstance(manifest.get("prediction_file"), str)
            or not manifest["prediction_file"]
            or manifest["prediction_file"] in (".", "..")
            or Path(manifest["prediction_file"]).name
            != manifest["prediction_file"]
        ):
            raise ProposalCacheError("Invalid cached prediction identity")
        _strict_sha256(
            "manifest prediction hash", manifest.get("prediction_sha256")
        )
        expected_frames = self._expected_frame_ids(
            schedule["dataset_length"], schedule["gap"]
        )
        if derived_frames != expected_frames:
            raise ProposalCacheError(
                f"Cache manifest has incomplete keyframe schedule: {scene_id}"
            )
        manifest["_by_frame"] = by_frame
        self._manifests[scene_id] = manifest
        return manifest

    def replay(
        self,
        scene_id: str,
        frame_id: int,
        *,
        inputs: Mapping[str, Any],
    ) -> Tuple[Instances3D, str]:
        if not self.is_replay:
            raise ProposalCacheError("replay() called outside replay mode")
        scene_id = _strict_scene_id(scene_id)
        manifest = self._load_manifest(scene_id)
        frame_id = _strict_int("frame_id", frame_id)
        consumed = self._consumed.setdefault(scene_id, [])
        if frame_id in consumed:
            raise ProposalCacheError(
                f"Duplicate cache consumption: {scene_id}/{frame_id}"
            )
        expected_index = len(consumed)
        expected_frames = manifest["recorded_frame_ids"]
        if expected_index >= len(expected_frames) or int(
            expected_frames[expected_index]
        ) != frame_id:
            raise ProposalCacheError(
                f"Out-of-order/unexpected cache frame: {scene_id}/{frame_id}"
            )
        record = manifest["_by_frame"].get(frame_id)
        if record is None:
            raise ProposalCacheError(
                f"Missing cached keyframe: {scene_id}/{frame_id}"
            )
        if _input_signature(inputs) != record["input_signature"]:
            raise ProposalCacheError(
                f"Current RGB-D/calibration input differs: {scene_id}/{frame_id}"
            )
        path = self._frame_path(scene_id, frame_id)
        if not path.is_file():
            raise ProposalCacheError(f"Cached keyframe is missing: {path}")
        payload = self._load_payload(
            path,
            expected_sha256=record["sha256"],
        )
        if (
            payload.get("attempt_id") != record["attempt_id"]
            or payload.get("input_signature") != record["input_signature"]
            or payload.get("protected_hashes") != record["protected_hashes"]
            or payload.get("geometry_sha256") != record["geometry_sha256"]
            or payload.get("proposal_contract_sha256")
            != record["proposal_contract_sha256"]
            or payload.get("rng_sha256") != record["rng_sha256"]
        ):
            raise ProposalCacheError(
                f"Cached event metadata mismatch: {scene_id}/{frame_id}"
            )
        instances = self._deserialize(
            payload,
            expected_scene_id=scene_id,
            expected_frame_id=frame_id,
        )
        if int(len(instances)) != int(record["count"]):
            raise ProposalCacheError(
                f"Manifest count mismatch: {scene_id}/{frame_id}"
            )
        self._restore_rng_state(payload["rng_state"])
        consumed.append(frame_id)
        return instances, str(record["attempt_id"])

    def verify_replay_complete(
        self,
        scene_id: str,
        *,
        baseline_prediction_path: Optional[str | Path] = None,
    ) -> None:
        """Fail closed if replay did not consume the complete recorded schedule."""

        if not self.is_replay:
            raise ProposalCacheError(
                "verify_replay_complete() called outside replay mode"
            )
        scene_id = _strict_scene_id(scene_id)
        manifest = self._load_manifest(scene_id)
        consumed = self._consumed.get(scene_id, [])
        expected = [int(value) for value in manifest["recorded_frame_ids"]]
        if consumed != expected:
            raise ProposalCacheError(
                f"Incomplete proposal replay for {scene_id}: "
                f"expected={expected}, consumed={consumed}"
            )
        if self.config.baseline_prediction_root is None:
            raise ProposalCacheError(
                "Replay has no configured baseline prediction root"
            )
        configured_prediction_path = (
            self.config.baseline_prediction_root / manifest["prediction_file"]
        )
        if baseline_prediction_path is None:
            baseline_prediction_path = configured_prediction_path
        else:
            baseline_prediction_path = Path(baseline_prediction_path)
            if (
                baseline_prediction_path.resolve()
                != configured_prediction_path.resolve()
            ):
                raise ProposalCacheError(
                    f"Baseline prediction path is outside the frozen root: {scene_id}"
                )
        baseline_prediction_path = Path(baseline_prediction_path)
        if (
            not baseline_prediction_path.is_file()
            or baseline_prediction_path.name != manifest["prediction_file"]
            or _sha256_file(baseline_prediction_path)
            != manifest["prediction_sha256"]
        ):
            raise ProposalCacheError(
                f"Frozen X0 prediction does not match cache: {scene_id}"
            )


def build_proposal_cache(
    cfg: Mapping[str, Any],
    *,
    device: torch.device,
) -> Optional[ProposalCache]:
    if not isinstance(cfg, Mapping):
        raise ValueError("BoxFusion config must be a mapping")
    lifting = cfg.get("lifting", {})
    if not isinstance(lifting, Mapping):
        raise ValueError("lifting must be a mapping")
    mapping = lifting.get("proposal_cache", {})
    config = ProposalCacheConfig.from_mapping(mapping)
    if config.mode == "disabled":
        return None
    backend_value = lifting.get("backend", "cutr")
    if not isinstance(backend_value, str):
        raise ValueError("lifting.backend must be a string")
    backend = backend_value.lower()
    if config.mode == "record" and backend != "cutr":
        raise ValueError("Proposal cache record mode requires CuTR backend")
    if config.mode == "replay" and backend not in {"cutr", "boxer"}:
        raise ValueError(
            "Proposal cache replay mode requires CuTR or Boxer backend"
        )
    return ProposalCache(config, device=device)
