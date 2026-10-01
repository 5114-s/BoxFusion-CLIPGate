"""Immutable CuTR post-filter proposal cache for paired lifting ablations.

The controlled Boxer experiment must compare the same ordered CuTR proposal
rows.  X0 records the final camera-frame proposals after all legacy filters and
immediately consumes a deserialized copy.  X1/X2 replay those exact rows and do
not execute the CuTR forward pass.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple

import numpy as np
import torch

from boxfusion.boxer_lifter import geometry_hash, protected_proposal_hashes
from boxfusion.boxes import BoxDOF, GeneralInstance3DBoxes
from boxfusion.instances import Instances3D


SCHEMA = "boxfusion.cutr_postfilter_cache.v2"
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


class ProposalCacheError(RuntimeError):
    """Raised when an immutable proposal-cache contract is broken."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(8 * 1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


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


def _array_sha256(value: Any) -> str:
    if isinstance(value, torch.Tensor):
        return _tensor_sha256(value)
    array = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("utf-8"))
    digest.update(b"\0")
    digest.update(repr(tuple(array.shape)).encode("ascii"))
    digest.update(b"\0")
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _input_signature(inputs: Mapping[str, Any]) -> Dict[str, str]:
    required = (
        "image",
        "depth",
        "image_K",
        "depth_K",
        "camera_to_world",
    )
    if tuple(inputs.keys()) != required:
        raise ProposalCacheError(
            f"Unexpected proposal-cache input schema: {tuple(inputs.keys())}"
        )
    return {name: _array_sha256(inputs[name]) for name in required}


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
        os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


@dataclass(frozen=True)
class ProposalCacheConfig:
    mode: str
    root: Path
    namespace: str

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> "ProposalCacheConfig":
        mode = str(mapping.get("mode", "disabled")).lower()
        if mode not in ("disabled", "record", "replay"):
            raise ValueError(
                "lifting.proposal_cache.mode must be disabled, record, or replay"
            )
        root = Path(
            os.path.abspath(
                os.path.expanduser(
                    str(mapping.get("root", "cache/cutr_postfilter"))
                )
            )
        )
        namespace = str(mapping.get("namespace", "")).strip()
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
        return cls(mode=mode, root=root, namespace=namespace)


class ProposalCache:
    """Record once in X0 and replay byte-identical CuTR rows in X1/X2."""

    def __init__(self, config: ProposalCacheConfig, device: torch.device):
        self.config = config
        self.device = torch.device(device)
        self._records: Dict[str, Dict[int, Dict[str, Any]]] = {}
        self._manifests: Dict[str, Dict[str, Any]] = {}
        self._consumed: Dict[str, list[int]] = {}
        self._schedules: Dict[str, Dict[str, int]] = {}
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
        if os.sep in scene_id or scene_id in ("", ".", ".."):
            raise ProposalCacheError(f"Invalid scene ID: {scene_id!r}")
        return self.config.root / self.config.namespace / scene_id

    def _frame_path(self, scene_id: str, frame_id: int) -> Path:
        return self._scene_root(scene_id) / f"frame_{int(frame_id):06d}.pt"

    def _manifest_path(self, scene_id: str) -> Path:
        return self._scene_root(scene_id) / "manifest.json"

    def bind_scene(self, scene_id: str, *, dataset_length: int, gap: int) -> None:
        schedule = {
            "dataset_length": int(dataset_length),
            "gap": int(gap),
        }
        if schedule["dataset_length"] <= 0 or schedule["gap"] <= 0:
            raise ProposalCacheError(f"Invalid scene schedule: {schedule}")
        previous = self._schedules.setdefault(scene_id, schedule)
        if previous != schedule:
            raise ProposalCacheError(
                f"Scene schedule changed for {scene_id}: {previous} != {schedule}"
            )

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
        python_state = state["python"]
        random.setstate(
            (
                int(python_state["version"]),
                tuple(int(value) for value in python_state["state"].tolist()),
                python_state["gauss"],
            )
        )
        numpy_state = state["numpy"]
        np.random.set_state(
            (
                str(numpy_state["algorithm"]),
                numpy_state["state"].cpu().numpy().astype(np.uint32),
                int(numpy_state["position"]),
                int(numpy_state["has_gauss"]),
                float(numpy_state["cached_gaussian"]),
            )
        )
        torch.set_rng_state(state["torch_cpu"].cpu())
        cuda_state = state.get("torch_cuda")
        if cuda_state is not None:
            if self.device.type != "cuda":
                raise ProposalCacheError(
                    "A CUDA proposal cache cannot be replayed on a CPU device"
                )
            torch.cuda.set_rng_state(cuda_state.cpu(), self.device)

    @staticmethod
    def _serialize(
        instances: Instances3D,
        *,
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
                fields[name] = {
                    "tensor": _canonical_tensor(value.tensor),
                    "rotation": _canonical_tensor(value.R),
                    "box_dim": int(value.box_dim),
                    "dof": value.dof.name,
                }
                metadata[name] = {
                    "tensor": _tensor_metadata(value.tensor),
                    "rotation": _tensor_metadata(value.R),
                    "box_dim": int(value.box_dim),
                    "dof": value.dof.name,
                }
            elif isinstance(value, torch.Tensor):
                fields[name] = _canonical_tensor(value)
                metadata[name] = _tensor_metadata(value)
            else:
                raise ProposalCacheError(
                    f"Unsupported cached field {name}: {type(value)}"
                )
        count = int(len(instances))
        for name in EXPECTED_FIELDS:
            if len(instances.get(name)) != count:
                raise ProposalCacheError(
                    f"CuTR field {name} has an inconsistent row count"
                )
        return {
            "schema": SCHEMA,
            "image_size": tuple(int(x) for x in instances.image_size),
            "field_names": list(field_names),
            "fields": fields,
            "field_metadata": metadata,
            "count": count,
            "attempt_id": attempt_id,
            "input_signature": dict(input_signature),
            "rng_state": {
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
            },
            "protected_hashes": protected_proposal_hashes(instances),
            "geometry_sha256": geometry_hash(instances),
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
        actual = _tensor_metadata(value)
        if actual != dict(metadata):
            raise ProposalCacheError(
                f"Cached tensor metadata/hash mismatch for {field_name}"
            )
        return value

    def _deserialize(self, payload: Mapping[str, Any]) -> Instances3D:
        if payload.get("schema") != SCHEMA:
            raise ProposalCacheError(
                f"Unexpected proposal-cache schema: {payload.get('schema')}"
            )
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

        instances = Instances3D(tuple(payload["image_size"]))
        for name in EXPECTED_FIELDS:
            value = fields[name]
            field_metadata = metadata[name]
            if name == "pred_boxes_3d":
                if set(value) != {"tensor", "rotation", "box_dim", "dof"}:
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
                if (
                    int(value["box_dim"]) != int(field_metadata["box_dim"])
                    or str(value["dof"]) != str(field_metadata["dof"])
                ):
                    raise ProposalCacheError("Cached 3D-box metadata changed")
                try:
                    dof = BoxDOF[str(value["dof"])]
                except KeyError as error:
                    raise ProposalCacheError("Unknown cached box DOF") from error
                value = GeneralInstance3DBoxes(
                    tensor.to(self.device),
                    rotation.to(self.device),
                    box_dim=int(value["box_dim"]),
                    dof=dof,
                )
            else:
                value = self._validate_tensor(
                    value,
                    field_metadata,
                    field_name=name,
                ).to(self.device)
            instances.set(name, value)

        if int(len(instances)) != int(payload["count"]):
            raise ProposalCacheError("Cached proposal count changed on load")
        if protected_proposal_hashes(instances) != payload["protected_hashes"]:
            raise ProposalCacheError("Cached protected fields changed on load")
        if geometry_hash(instances) != payload["geometry_sha256"]:
            raise ProposalCacheError("Cached geometry changed on load")
        return instances

    @staticmethod
    def _load_payload(path: Path) -> Dict[str, Any]:
        try:
            payload = torch.load(path, map_location="cpu", weights_only=True)
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
        frame_id = int(frame_id)
        scene_records = self._records.setdefault(scene_id, {})
        if frame_id in scene_records:
            raise ProposalCacheError(
                f"Duplicate cached frame: {scene_id}/{frame_id}"
            )
        rng_state = self._capture_rng_state()
        input_signature = _input_signature(inputs)
        payload = self._serialize(
            instances,
            attempt_id=attempt_id,
            input_signature=input_signature,
            rng_state=rng_state,
        )

        scene_root = self._scene_root(scene_id)
        scene_root.mkdir(parents=True, exist_ok=True)
        final_path = self._frame_path(scene_id, frame_id)
        if final_path.exists():
            raise ProposalCacheError(
                f"Refusing to overwrite immutable cache frame: {final_path}"
            )
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
            os.replace(temporary_path, final_path)
        finally:
            if temporary_path.exists():
                temporary_path.unlink()

        roundtrip_payload = self._load_payload(final_path)
        canonical_instances = self._deserialize(roundtrip_payload)
        self._restore_rng_state(roundtrip_payload["rng_state"])
        scene_records[frame_id] = {
            "frame_id": frame_id,
            "attempt_id": attempt_id,
            "count": int(payload["count"]),
            "sha256": _sha256_file(final_path),
            "input_signature": input_signature,
            "protected_hashes": payload["protected_hashes"],
            "geometry_sha256": payload["geometry_sha256"],
        }
        return canonical_instances

    def finalize(self, scene_id: str, prediction_path: str | Path) -> Path:
        """Seal a recorded scene only after its prediction was saved."""

        if not self.is_record:
            raise ProposalCacheError("finalize() called outside record mode")
        records = self._records.get(scene_id)
        if not records:
            raise ProposalCacheError(
                f"No proposal-cache records were written for {scene_id}"
            )
        schedule = self._schedules.get(scene_id)
        if schedule is None:
            raise ProposalCacheError(f"Scene schedule was not bound: {scene_id}")
        prediction_path = Path(prediction_path)
        if not prediction_path.is_file():
            raise ProposalCacheError(
                f"Cannot seal cache before prediction exists: {prediction_path}"
            )
        producer_fingerprint = os.environ.get(
            "BOXFUSION_PROPOSAL_CACHE_PRODUCER_FINGERPRINT", ""
        ).strip()
        if not producer_fingerprint:
            raise ProposalCacheError(
                "Missing proposal-cache producer fingerprint"
            )
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
                "terminal_policy": "upstream_boxfusion_early_exit_v1",
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
        if final_path.exists():
            raise ProposalCacheError(
                f"Refusing to overwrite immutable cache manifest: {final_path}"
            )
        _atomic_json_write(final_path, manifest)
        return final_path

    def _load_manifest(self, scene_id: str) -> Dict[str, Any]:
        if scene_id in self._manifests:
            return self._manifests[scene_id]
        path = self._manifest_path(scene_id)
        if not path.is_file():
            raise ProposalCacheError(f"Missing cache manifest: {path}")
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if (
            manifest.get("schema") != SCHEMA
            or manifest.get("scene_id") != scene_id
            or manifest.get("namespace") != self.config.namespace
        ):
            raise ProposalCacheError(f"Invalid cache manifest: {path}")
        expected_fingerprint = os.environ.get(
            "BOXFUSION_PROPOSAL_CACHE_EXPECTED_FINGERPRINT", ""
        ).strip()
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
        }
        if any(set(row) != required_record_keys for row in records):
            raise ProposalCacheError("Invalid proposal-cache record schema")
        by_frame = {int(row["frame_id"]): row for row in records}
        derived_frames = [int(row["frame_id"]) for row in records]
        derived_proposals = sum(int(row["count"]) for row in records)
        derived_nonempty = sum(int(row["count"] > 0) for row in records)
        if (
            len(by_frame) != len(records)
            or derived_frames != sorted(by_frame)
            or int(manifest.get("record_count", -1)) != len(records)
            or manifest.get("recorded_frame_ids") != derived_frames
            or int(manifest.get("proposal_count", -1)) != derived_proposals
            or int(manifest.get("nonempty_frame_count", -1))
            != derived_nonempty
        ):
            raise ProposalCacheError("Invalid/duplicate frame IDs in cache manifest")
        schedule = self._schedules.get(scene_id)
        manifest_schedule = manifest.get("schedule", {})
        if (
            schedule is None
            or int(manifest_schedule.get("dataset_length", -1))
            != schedule["dataset_length"]
            or int(manifest_schedule.get("gap", -1)) != schedule["gap"]
            or manifest_schedule.get("terminal_policy")
            != "upstream_boxfusion_early_exit_v1"
        ):
            raise ProposalCacheError(
                f"Current scene schedule differs from cache: {scene_id}"
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
        manifest = self._load_manifest(scene_id)
        frame_id = int(frame_id)
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
        if not path.is_file() or _sha256_file(path) != record["sha256"]:
            raise ProposalCacheError(f"Cached keyframe hash mismatch: {path}")
        payload = self._load_payload(path)
        if (
            payload.get("attempt_id") != record["attempt_id"]
            or payload.get("input_signature") != record["input_signature"]
        ):
            raise ProposalCacheError(
                f"Cached event metadata mismatch: {scene_id}/{frame_id}"
            )
        instances = self._deserialize(payload)
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
        baseline_prediction_path: str | Path,
    ) -> None:
        """Fail closed if replay did not consume the complete recorded schedule."""

        if not self.is_replay:
            raise ProposalCacheError(
                "verify_replay_complete() called outside replay mode"
            )
        manifest = self._load_manifest(scene_id)
        consumed = self._consumed.get(scene_id, [])
        expected = [int(value) for value in manifest["recorded_frame_ids"]]
        if consumed != expected:
            raise ProposalCacheError(
                f"Incomplete proposal replay for {scene_id}: "
                f"expected={expected}, consumed={consumed}"
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
    mapping = cfg.get("lifting", {}).get("proposal_cache", {})
    config = ProposalCacheConfig.from_mapping(mapping)
    if config.mode == "disabled":
        return None
    backend = str(cfg.get("lifting", {}).get("backend", "cutr")).lower()
    if config.mode == "record" and backend != "cutr":
        raise ValueError("Proposal cache record mode requires CuTR backend")
    if config.mode == "replay" and backend != "boxer":
        raise ValueError("Proposal cache replay mode requires Boxer backend")
    return ProposalCache(config, device=device)
