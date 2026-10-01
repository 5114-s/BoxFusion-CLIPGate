#!/usr/bin/env python3
"""Fail-closed, GT-free audit for OpenBox-SMOV R2 shadow runs.

The audit binds the terminal shadow sidecar to the prediction actually saved
by ``demo.py``, verifies every counterfactual receipt and safety bound, and
checks the frozen online budgets and measured latency.  It never materializes
counterfactual predictions and has no ground-truth input.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import pickle
import re
import sys
import tempfile
from types import SimpleNamespace
from typing import Any, Mapping, Sequence
import zipfile

import numpy as np


_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from boxfusion.tr3d_r4_smov_observer import corners_to_yaw_boxes  # noqa: E402
from tools.tr3d_data import read_scene_list  # noqa: E402


SUMMARY_SCHEMA_V1 = "boxfusion.openbox_smov_r2_shadow.v1"
SUMMARY_SCHEMA = "boxfusion.openbox_smov_r2_shadow.v2"
REPORT_SCHEMA_V1 = "boxfusion.openbox_smov_r2_shadow_audit.v1"
REPORT_SCHEMA = "boxfusion.openbox_smov_r2_shadow_audit.v2"
SUMMARY_PREFIX = "OpenBox-SMOV R2 shadow JSON | "
SIDECAR_SUFFIX = "_openbox_smov_r2_shadow.npz"
PREDICTION_SUFFIX = "_boxes.pkl"
_FPS_RE = re.compile(
    r"Average FPS:\s*([0-9]+(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?)"
)
_MAX_LOG_BYTES = 64 * 1024 * 1024
_MAX_PREDICTION_BYTES = 64 * 1024 * 1024
_MAX_SIDECAR_COMPRESSED_BYTES = 64 * 1024 * 1024
_MAX_SIDECAR_UNCOMPRESSED_BYTES = 32 * 1024 * 1024


# Pin the complete S0 experiment contract.  ``diagnostics.root`` is the one
# host-specific value and is validated separately.
EXPECTED_CONFIG_V1: dict[str, object] = {
    "enabled": True,
    "observer_only": True,
    "pixel_stride": 4,
    "min_depth_m": 0.10,
    "max_depth_m": 8.0,
    "depth_edge_m": 0.15,
    "component_jump_m": 0.15,
    "min_component_pixels": 16,
    "voxel_size_m": 0.05,
    "max_points_per_view": 512,
    "max_points_per_track": 1024,
    "max_validation_rays_per_view": 1024,
    "max_views_per_track": 5,
    "max_tracks": 1024,
    "max_proposals_per_keyframe": 64,
    "min_views": 3,
    "min_points": 192,
    "translation_gap_m": 0.80,
    "rotation_gap_deg": 30.0,
    "lower_quantile": 0.02,
    "upper_quantile": 0.98,
    "minimum_extent_m": 0.05,
    "max_center_shift_diagonal": 0.60,
    "min_extent_ratio": 0.35,
    "max_extent_ratio": 2.50,
    "depth_margin_m": 0.05,
    "near_clip_m": 1e-3,
    "max_diagnostics": 1024,
    "timing_window": 4096,
}

EXPECTED_CONFIG: dict[str, object] = {
    **EXPECTED_CONFIG_V1,
    "face_front_dot": 0.25,
    "face_weak_dot": 0.05,
    "face_band_fraction": 0.10,
    "face_band_max_m": 0.20,
    "min_face_points": 8,
    "min_face_weak_points": 4,
    "face_extension_fraction": 0.25,
    "face_extension_min_m": 0.05,
    "face_extension_max_m": 0.30,
    "max_face_candidates_per_fit": 4,
}

_FALSE_ATTESTATIONS = (
    "active_authorized",
    "training_invoked",
    "online_learning",
    "ground_truth_access",
    "clip_access",
    "semantic_access",
    "checkpoint_access",
    "future_frame_access",
    "full_scene_reconstruction",
    "native_outputs_mutated",
    "counterfactual_geometry_applied",
)
_COUNTERS = (
    "keyframes",
    "proposals",
    "proposal_cap_drops",
    "valid_fragments",
    "invalid_fragments",
    "accepted_views",
    "same_frame_duplicates",
    "track_capacity_drops",
    "retired_tracks",
    "prepare_failures",
    "would_replace",
    "active_tracks_at_close",
)
_SIDECAR_FIELDS = {
    "schema",
    "native_corners",
    "native_scores",
    "stable_ids",
    "counterfactual_corners",
    "would_replace_mask",
    "receipts_json",
}
_RECEIPT_FIELDS_V1 = {
    "native_index",
    "stable_id",
    "reason",
    "hypothesis",
    "view_frame_ids",
    "native_corners",
    "candidate_corners",
    "native_projection_iou",
    "candidate_projection_iou",
    "native_support",
    "candidate_support",
    "native_free_space",
    "candidate_free_space",
    "center_shift_m",
    "volume_ratio",
    "would_replace",
}

_RECEIPT_FIELDS = _RECEIPT_FIELDS_V1 | {
    "face_extension_signs",
    "face_extension_delta_m",
    "face_strong_mask",
    "face_weak_mask",
}


@dataclass(frozen=True)
class AuditContract:
    name: str
    summary_schema: str
    report_schema: str
    expected_config: Mapping[str, object]
    receipt_fields: frozenset[str]
    allowed_hypotheses: frozenset[str]
    visibility_metadata: bool


_V1_HYPOTHESES = frozenset(
    {"native_yaw_quantile", "pca_yaw_quantile"}
)
_V2_HYPOTHESES = frozenset(
    f"{yaw}+{recipe}"
    for yaw in ("native_yaw_quantile", "pca_yaw_quantile")
    for recipe in ("base", "face_x", "face_y", "face_xy")
)
AUDIT_CONTRACTS = {
    "r2-v1": AuditContract(
        name="r2-v1",
        summary_schema=SUMMARY_SCHEMA_V1,
        report_schema=REPORT_SCHEMA_V1,
        expected_config=EXPECTED_CONFIG_V1,
        receipt_fields=frozenset(_RECEIPT_FIELDS_V1),
        allowed_hypotheses=_V1_HYPOTHESES,
        visibility_metadata=False,
    ),
    "visibility-v2": AuditContract(
        name="visibility-v2",
        summary_schema=SUMMARY_SCHEMA,
        report_schema=REPORT_SCHEMA,
        expected_config=EXPECTED_CONFIG,
        receipt_fields=frozenset(_RECEIPT_FIELDS),
        allowed_hypotheses=_V2_HYPOTHESES,
        visibility_metadata=True,
    ),
}


class AuditError(ValueError):
    """One audited artifact violated the frozen shadow contract."""


def _require(condition: object, message: str) -> None:
    if not bool(condition):
        raise AuditError(message)


def _regular(path: Path, label: str, maximum_bytes: int) -> Path:
    if path.is_symlink():
        raise FileNotFoundError(f"{label} must not be a symlink: {path}")
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{label} is not a regular file: {path}")
    size = path.stat().st_size
    if size < 1 or size > maximum_bytes:
        raise AuditError(f"{label} has invalid size {size}: {path}")
    return path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class _PredictionUnpickler(pickle.Unpickler):
    """Allow only NumPy's ndarray reconstruction helpers."""

    _ALLOWED = {
        ("numpy", "dtype"),
        ("numpy", "ndarray"),
        ("numpy.core.multiarray", "_reconstruct"),
        ("numpy.core.multiarray", "scalar"),
        ("numpy.core.numeric", "_frombuffer"),
        ("numpy._core.multiarray", "_reconstruct"),
        ("numpy._core.multiarray", "scalar"),
        ("numpy._core.numeric", "_frombuffer"),
    }

    def find_class(self, module: str, name: str) -> Any:
        if (module, name) not in self._ALLOWED:
            raise pickle.UnpicklingError(
                f"forbidden prediction pickle global {module}.{name}"
            )
        return super().find_class(module, name)


def _prediction(path: Path) -> tuple[np.ndarray, np.ndarray]:
    path = _regular(path, "native prediction", _MAX_PREDICTION_BYTES)
    try:
        with path.open("rb") as handle:
            payload = _PredictionUnpickler(handle).load()
    except (pickle.PickleError, AttributeError, EOFError, ImportError) as error:
        raise AuditError(f"malformed native prediction: {path}") from error
    _require(
        type(payload) is list
        and len(payload) == 1
        and type(payload[0]) is list,
        f"non-canonical prediction container: {path}",
    )
    corners: list[np.ndarray] = []
    scores: list[float] = []
    for index, row in enumerate(payload[0]):
        _require(
            type(row) is tuple
            and len(row) == 3
            and type(row[0]) is int
            and row[0] == 0,
            f"malformed prediction row {index}: {path}",
        )
        geometry = np.asarray(row[1])
        _require(
            geometry.shape == (8, 3)
            and geometry.dtype == np.float32
            and np.isfinite(geometry).all(),
            f"invalid prediction geometry row {index}: {path}",
        )
        try:
            score = float(row[2])
        except (TypeError, ValueError, OverflowError) as error:
            raise AuditError(
                f"invalid prediction score row {index}: {path}"
            ) from error
        _require(math.isfinite(score), f"non-finite score row {index}: {path}")
        corners.append(np.array(geometry, dtype=np.float32, order="C", copy=True))
        scores.append(score)
    return (
        np.stack(corners)
        if corners
        else np.empty((0, 8, 3), dtype=np.float32),
        np.asarray(scores, dtype=np.float64),
    )


def _read_log(path: Path, *, require_summary: bool) -> tuple[dict[str, Any] | None, float]:
    path = _regular(path, "scene log", _MAX_LOG_BYTES)
    text = path.read_text(encoding="utf-8")
    summaries = []
    for line in text.splitlines():
        position = line.find(SUMMARY_PREFIX)
        if position < 0:
            continue
        try:
            value = json.loads(line[position + len(SUMMARY_PREFIX) :])
        except json.JSONDecodeError as error:
            raise AuditError(f"invalid R2 summary JSON in {path}") from error
        _require(isinstance(value, dict), f"R2 summary is not an object: {path}")
        summaries.append(value)
    if require_summary:
        _require(len(summaries) == 1, f"expected exactly one R2 summary in {path}")
    else:
        _require(not summaries, f"control log unexpectedly contains R2 summary: {path}")
    fps_values = []
    for match in _FPS_RE.finditer(text):
        fps = float(match.group(1))
        _require(math.isfinite(fps) and fps > 0.0, f"invalid FPS in {path}")
        fps_values.append(fps)
    _require(len(fps_values) == 1, f"expected exactly one Average FPS in {path}")
    return (summaries[0] if summaries else None), fps_values[0]


def _sidecar(path: Path, contract: AuditContract) -> dict[str, Any]:
    path = _regular(path, "R2 sidecar", _MAX_SIDECAR_COMPRESSED_BYTES)
    try:
        with zipfile.ZipFile(path) as archive:
            infos = archive.infolist()
            _require(
                {Path(info.filename).stem for info in infos} == _SIDECAR_FIELDS
                and len(infos) == len(_SIDECAR_FIELDS)
                and all(Path(info.filename).suffix == ".npy" for info in infos),
                f"unexpected R2 sidecar members: {path}",
            )
            expanded = sum(info.file_size for info in infos)
            _require(
                expanded <= _MAX_SIDECAR_UNCOMPRESSED_BYTES,
                f"R2 sidecar exceeds expanded-size budget: {path}",
            )
        with np.load(path, allow_pickle=False) as archive:
            _require(set(archive.files) == _SIDECAR_FIELDS, f"bad sidecar fields: {path}")
            schema = np.asarray(archive["schema"])
            native = np.array(archive["native_corners"], copy=True)
            scores = np.array(archive["native_scores"], copy=True)
            stable_ids = np.array(archive["stable_ids"], copy=True)
            counterfactual = np.array(archive["counterfactual_corners"], copy=True)
            mask = np.array(archive["would_replace_mask"], copy=True)
            receipt_bytes = np.asarray(archive["receipts_json"])
    except (OSError, ValueError, zipfile.BadZipFile) as error:
        raise AuditError(f"invalid R2 sidecar: {path}") from error
    _require(
        schema.shape == ()
        and schema.dtype.kind in "US"
        and schema.item() == contract.summary_schema,
        f"wrong R2 sidecar schema: {path}",
    )
    count = len(native) if native.ndim == 3 else -1
    _require(
        native.shape == (count, 8, 3)
        and native.dtype == np.float32
        and scores.shape == (count,)
        and scores.dtype == np.float32
        and stable_ids.shape == (count,)
        and stable_ids.dtype == np.int64
        and counterfactual.shape == native.shape
        and counterfactual.dtype == np.float32
        and mask.shape == (count,)
        and mask.dtype == np.bool_
        and np.isfinite(native).all()
        and np.isfinite(scores).all()
        and np.isfinite(counterfactual).all()
        and np.all(stable_ids >= 0)
        and len(np.unique(stable_ids)) == count,
        f"invalid or misaligned R2 sidecar arrays: {path}",
    )
    _require(
        receipt_bytes.ndim == 1 and receipt_bytes.dtype == np.uint8,
        f"invalid R2 receipt byte array: {path}",
    )
    try:
        receipts = json.loads(receipt_bytes.tobytes().decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise AuditError(f"invalid R2 receipt JSON: {path}") from error
    _require(
        isinstance(receipts, list) and len(receipts) == count,
        f"R2 receipts do not align with native rows: {path}",
    )
    return {
        "native_corners": native,
        "native_scores": scores,
        "stable_ids": stable_ids,
        "counterfactual_corners": counterfactual,
        "would_replace_mask": mask,
        "receipts": receipts,
    }


def _nonnegative_int(value: object, label: str) -> int:
    _require(type(value) is int and value >= 0, f"{label} must be a nonnegative integer")
    return int(value)


def _metric(value: object, label: str) -> float:
    _require(
        type(value) in (int, float) and not isinstance(value, bool),
        f"{label} must be a number",
    )
    result = float(value)
    _require(math.isfinite(result), f"{label} must be finite")
    return result


def _timing(
    value: object,
    label: str,
    *,
    require_positive: bool,
    p95_limit: float,
    max_limit: float,
) -> dict[str, float]:
    _require(
        isinstance(value, Mapping) and set(value) == {"mean_ms", "p95_ms", "max_ms"},
        f"{label} has the wrong timing fields",
    )
    result = {key: _metric(value[key], f"{label}.{key}") for key in value}
    _require(all(item >= 0.0 for item in result.values()), f"{label} is negative")
    _require(result["p95_ms"] <= result["max_ms"] + 1e-12, f"{label} p95 exceeds max")
    if require_positive:
        _require(result["mean_ms"] > 0.0 and result["max_ms"] > 0.0, f"{label} was not measured")
    _require(result["p95_ms"] <= p95_limit, f"{label} p95 budget exceeded")
    _require(result["max_ms"] <= max_limit, f"{label} max budget exceeded")
    return result


def _validate_config(
    value: object, scene: str, contract: AuditContract
) -> dict[str, Any]:
    _require(isinstance(value, Mapping), f"{scene}: missing effective_config")
    _require(
        set(value) == set(contract.expected_config) | {"diagnostics"},
        f"{scene}: effective_config key set drifted",
    )
    for key, expected in contract.expected_config.items():
        _require(value[key] == expected, f"{scene}: effective_config.{key} drifted")
    diagnostics = value["diagnostics"]
    _require(
        isinstance(diagnostics, Mapping)
        and set(diagnostics) == {"root"}
        and isinstance(diagnostics["root"], str)
        and bool(diagnostics["root"].strip()),
        f"{scene}: diagnostics.root is not a non-empty path",
    )
    return dict(value)


def _validate_receipts(
    scene: str,
    artifact: Mapping[str, Any],
    config: Mapping[str, Any],
    contract: AuditContract,
) -> None:
    native = artifact["native_corners"]
    counterfactual = artifact["counterfactual_corners"]
    stable_ids = artifact["stable_ids"]
    mask = artifact["would_replace_mask"]
    receipts = artifact["receipts"]
    for index, receipt in enumerate(receipts):
        _require(
            isinstance(receipt, Mapping)
            and set(receipt) == set(contract.receipt_fields),
            f"{scene}:{index}: malformed receipt",
        )
        _require(
            receipt["native_index"] == index
            and receipt["stable_id"] == int(stable_ids[index])
            and type(receipt["would_replace"]) is bool
            and receipt["would_replace"] == bool(mask[index]),
            f"{scene}:{index}: receipt identity/mask mismatch",
        )
        receipt_native = np.asarray(receipt["native_corners"], dtype=np.float32)
        _require(
            receipt_native.shape == (8, 3)
            and np.array_equal(receipt_native, native[index]),
            f"{scene}:{index}: receipt native geometry mismatch",
        )
        frames = receipt["view_frame_ids"]
        _require(
            isinstance(frames, list)
            and all(type(frame) is int and frame >= 0 for frame in frames)
            and frames == sorted(set(frames)),
            f"{scene}:{index}: non-causal or duplicate view frame IDs",
        )
        candidate_raw = receipt["candidate_corners"]
        if candidate_raw is None:
            _require(not mask[index], f"{scene}:{index}: replacement lacks candidate")
            candidate = None
            _require(
                receipt["hypothesis"] is None,
                f"{scene}:{index}: empty candidate has a hypothesis",
            )
        else:
            candidate = np.asarray(candidate_raw, dtype=np.float32)
            _require(
                candidate.shape == (8, 3) and np.isfinite(candidate).all(),
                f"{scene}:{index}: invalid candidate geometry",
            )
            _require(
                receipt["hypothesis"] in contract.allowed_hypotheses,
                f"{scene}:{index}: unknown hypothesis",
            )

        if contract.visibility_metadata:
            metadata_names = (
                "face_extension_signs",
                "face_extension_delta_m",
                "face_strong_mask",
                "face_weak_mask",
            )
            metadata = [receipt[name] for name in metadata_names]
            if candidate is None:
                _require(
                    all(value is None for value in metadata),
                    f"{scene}:{index}: empty candidate has face metadata",
                )
            else:
                signs_raw, deltas_raw, strong_raw, weak_raw = metadata
                _require(
                    isinstance(signs_raw, list)
                    and len(signs_raw) == 2
                    and all(
                        type(value) is int and value in (-1, 0, 1)
                        for value in signs_raw
                    ),
                    f"{scene}:{index}: invalid face extension signs",
                )
                _require(
                    isinstance(deltas_raw, list) and len(deltas_raw) == 2,
                    f"{scene}:{index}: invalid face extension delta",
                )
                deltas = [
                    _metric(value, f"{scene}:{index}:face_delta")
                    for value in deltas_raw
                ]
                _require(
                    isinstance(strong_raw, list)
                    and isinstance(weak_raw, list)
                    and len(strong_raw) == len(weak_raw) == 4
                    and all(type(value) is bool for value in strong_raw)
                    and all(type(value) is bool for value in weak_raw),
                    f"{scene}:{index}: invalid face masks",
                )
                _require(
                    all(not strong or weak for strong, weak in zip(strong_raw, weak_raw)),
                    f"{scene}:{index}: strong face is not weak-visible",
                )
                recipe = str(receipt["hypothesis"]).rsplit("+", 1)[-1]
                expected_axes = {
                    "base": (False, False),
                    "face_x": (True, False),
                    "face_y": (False, True),
                    "face_xy": (True, True),
                }[recipe]
                for axis, active in enumerate(expected_axes):
                    sign = int(signs_raw[axis])
                    delta = float(deltas[axis])
                    if not active:
                        _require(
                            sign == 0 and delta == 0.0,
                            f"{scene}:{index}: recipe/axis metadata mismatch",
                        )
                        continue
                    _require(
                        sign != 0
                        and float(config["face_extension_min_m"]) <= delta
                        <= float(config["face_extension_max_m"]),
                        f"{scene}:{index}: active face extension is out of bounds",
                    )
                    negative, positive = ((0, 1), (2, 3))[axis]
                    unseen = positive if sign > 0 else negative
                    anchor = negative if sign > 0 else positive
                    _require(
                        not weak_raw[unseen] and strong_raw[anchor],
                        f"{scene}:{index}: extension lacks visible-anchor/unseen-face evidence",
                    )

        if mask[index]:
            _require(
                receipt["reason"] == "loo_improved"
                and receipt["hypothesis"] in contract.allowed_hypotheses
                and len(frames) >= int(config["min_views"])
                and candidate is not None
                and np.array_equal(candidate, counterfactual[index]),
                f"{scene}:{index}: replacement is not a complete LOO receipt",
            )
        else:
            _require(
                np.array_equal(counterfactual[index], native[index]),
                f"{scene}:{index}: unselected geometry changed",
            )

        metric_names = (
            "native_projection_iou",
            "candidate_projection_iou",
            "native_support",
            "candidate_support",
            "native_free_space",
            "candidate_free_space",
            "center_shift_m",
            "volume_ratio",
        )
        present = [receipt[name] is not None for name in metric_names]
        _require(all(present) or not any(present), f"{scene}:{index}: partial metrics")
        if not any(present):
            _require(candidate is None, f"{scene}:{index}: candidate lacks metrics")
            continue
        metrics = {name: _metric(receipt[name], f"{scene}:{index}:{name}") for name in metric_names}
        for name in metric_names[:6]:
            _require(0.0 <= metrics[name] <= 1.0, f"{scene}:{index}:{name} outside [0,1]")
        dominates = (
            metrics["candidate_projection_iou"] >= metrics["native_projection_iou"]
            and metrics["candidate_support"] >= metrics["native_support"]
            and metrics["candidate_free_space"] <= metrics["native_free_space"]
            and (
                metrics["candidate_projection_iou"]
                > metrics["native_projection_iou"] + 1e-9
                or metrics["candidate_support"] > metrics["native_support"] + 1e-9
                or metrics["candidate_free_space"]
                < metrics["native_free_space"] - 1e-9
            )
        )
        _require(dominates == bool(mask[index]), f"{scene}:{index}: LOO dominance mismatch")
        _require(candidate is not None, f"{scene}:{index}: metrics lack candidate")

        native_box = corners_to_yaw_boxes(native[index : index + 1])[0]
        candidate_box = corners_to_yaw_boxes(candidate[None, ...])[0]
        shift = float(np.linalg.norm(candidate_box[:3] - native_box[:3]))
        native_dims = np.asarray([*sorted(native_box[3:5]), native_box[5]])
        candidate_dims = np.asarray([*sorted(candidate_box[3:5]), candidate_box[5]])
        ratios = candidate_dims / native_dims
        volume_ratio = float(np.prod(candidate_box[3:6]) / np.prod(native_box[3:6]))
        _require(
            shift <= float(config["max_center_shift_diagonal"]) * float(np.linalg.norm(native_box[3:6])) + 1e-5
            and np.all(ratios >= float(config["min_extent_ratio"]) - 1e-5)
            and np.all(ratios <= float(config["max_extent_ratio"]) + 1e-5),
            f"{scene}:{index}: counterfactual violates geometric safety",
        )
        _require(
            math.isclose(metrics["center_shift_m"], shift, rel_tol=1e-5, abs_tol=1e-5)
            and math.isclose(metrics["volume_ratio"], volume_ratio, rel_tol=1e-5, abs_tol=1e-5),
            f"{scene}:{index}: receipt safety metrics disagree with geometry",
        )


def _audit_scene(
    *,
    scene: str,
    prediction_root: Path,
    diagnostics_root: Path,
    log_root: Path,
    anchor_root: Path | None,
    control_log_root: Path | None,
    limits: SimpleNamespace,
    contract: AuditContract,
) -> dict[str, Any]:
    prediction_path = prediction_root / f"{scene}{PREDICTION_SUFFIX}"
    sidecar_path = diagnostics_root / f"{scene}{SIDECAR_SUFFIX}"
    log_path = log_root / f"{scene}.log"
    summary, fps = _read_log(log_path, require_summary=True)
    assert summary is not None
    _require(
        summary.get("schema") == contract.summary_schema,
        f"{scene}: wrong summary schema",
    )
    _require(summary.get("scene_id") == scene, f"{scene}: summary scene mismatch")
    _require(summary.get("enabled") is True, f"{scene}: observer was disabled")
    _require(summary.get("observer_only") is True, f"{scene}: observer_only is false")
    _require(summary.get("closed") is True, f"{scene}: observer did not close")
    for key in _FALSE_ATTESTATIONS:
        _require(summary.get(key) is False, f"{scene}: unsafe attestation {key}")
    config = _validate_config(
        summary.get("effective_config"), scene, contract
    )

    counters = {key: _nonnegative_int(summary.get(key), f"{scene}:{key}") for key in _COUNTERS}
    _require(
        counters["valid_fragments"] + counters["invalid_fragments"] == counters["proposals"],
        f"{scene}: proposal accounting mismatch",
    )
    _require(
        counters["accepted_views"] <= counters["valid_fragments"]
        and counters["active_tracks_at_close"] <= int(config["max_tracks"]),
        f"{scene}: memory/accounting bound violated",
    )
    _require(
        counters["proposal_cap_drops"] <= limits.max_proposal_cap_drops,
        f"{scene}: proposal cap-drop budget exceeded",
    )
    _require(
        counters["track_capacity_drops"] <= limits.max_track_capacity_drops,
        f"{scene}: track capacity-drop budget exceeded",
    )
    core = _timing(
        summary.get("core_timing"),
        f"{scene}:core_timing",
        require_positive=counters["keyframes"] > 0,
        p95_limit=limits.max_core_p95_ms,
        max_limit=limits.max_core_max_ms,
    )
    wrapper = _timing(
        summary.get("wrapper_timing"),
        f"{scene}:wrapper_timing",
        require_positive=counters["keyframes"] > 0,
        p95_limit=limits.max_wrapper_p95_ms,
        max_limit=limits.max_wrapper_max_ms,
    )
    _require(fps >= limits.min_average_fps, f"{scene}: online FPS floor missed")

    prediction_corners, prediction_scores = _prediction(prediction_path)
    artifact = _sidecar(sidecar_path, contract)
    native = artifact["native_corners"]
    scores = artifact["native_scores"]
    _require(
        np.array_equal(prediction_corners, native)
        and np.array_equal(prediction_scores.astype(np.float32), scores),
        f"{scene}: saved native prediction is not sidecar-identical",
    )
    _validate_receipts(scene, artifact, config, contract)
    mask = artifact["would_replace_mask"]
    count = len(native)
    replacements = np.flatnonzero(mask).astype(np.int64)
    terminal = summary.get("terminal")
    _require(isinstance(terminal, Mapping), f"{scene}: terminal summary missing")
    _require(
        terminal.get("native_count") == count
        and terminal.get("counterfactual_count") == int(len(replacements))
        and terminal.get("would_replace_native_indices") == replacements.tolist()
        and terminal.get("would_replace_stable_ids")
        == artifact["stable_ids"][mask].tolist()
        and terminal.get("native_export_mutated") is False
        and terminal.get("counterfactual_geometry_applied") is False
        and counters["would_replace"] == int(len(replacements)),
        f"{scene}: terminal summary disagrees with sidecar",
    )
    summary_receipts = summary.get("receipts")
    expected_summary_receipts = artifact["receipts"][: int(config["max_diagnostics"])]
    _require(
        summary_receipts == expected_summary_receipts,
        f"{scene}: log/sidecar receipt mismatch",
    )

    prediction_sha = _sha256(_regular(prediction_path, "native prediction", _MAX_PREDICTION_BYTES))
    anchor_equal = None
    anchor_sha = None
    if anchor_root is not None:
        anchor_path = _regular(
            anchor_root / f"{scene}{PREDICTION_SUFFIX}",
            "same-run native anchor",
            _MAX_PREDICTION_BYTES,
        )
        anchor_sha = _sha256(anchor_path)
        anchor_equal = anchor_sha == prediction_sha
        _require(anchor_equal, f"{scene}: same-run native prediction bytes differ")

    control_fps = None
    fps_ratio = None
    if control_log_root is not None:
        _, control_fps = _read_log(control_log_root / f"{scene}.log", require_summary=False)
        fps_ratio = fps / control_fps
        _require(fps_ratio >= limits.min_paired_fps_ratio, f"{scene}: paired FPS ratio missed")
    return {
        "scene_id": scene,
        "prediction_sha256": prediction_sha,
        "sidecar_sha256": _sha256(_regular(sidecar_path, "R2 sidecar", _MAX_SIDECAR_COMPRESSED_BYTES)),
        "log_sha256": _sha256(_regular(log_path, "scene log", _MAX_LOG_BYTES)),
        "native_rows": count,
        "would_replace": int(len(replacements)),
        "average_fps": fps,
        "control_average_fps": control_fps,
        "paired_fps_ratio": fps_ratio,
        "core_timing": core,
        "wrapper_timing": wrapper,
        "anchor_sha256": anchor_sha,
        "anchor_byte_identity": anchor_equal,
        "proposal_cap_drops": counters["proposal_cap_drops"],
        "track_capacity_drops": counters["track_capacity_drops"],
    }


def _write_create_only(path: Path, payload: Mapping[str, Any]) -> Path:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
    temporary_name: str | None = None
    try:
        descriptor, temporary_name = tempfile.mkstemp(
            dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
        )
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary_name, path)
        except FileExistsError as error:
            raise FileExistsError(f"refusing existing R2 audit report: {path}") from error
        path.chmod(0o444)
    finally:
        if temporary_name is not None:
            try:
                os.unlink(temporary_name)
            except FileNotFoundError:
                pass
    return path


def audit(args: argparse.Namespace) -> dict[str, Any]:
    contract_name = getattr(args, "contract", "visibility-v2")
    try:
        contract = AUDIT_CONTRACTS[contract_name]
    except KeyError as error:
        raise AuditError(f"unknown R2 audit contract: {contract_name}") from error
    limits = SimpleNamespace(
        min_average_fps=_metric(args.min_average_fps, "min_average_fps"),
        min_paired_fps_ratio=_metric(args.min_paired_fps_ratio, "min_paired_fps_ratio"),
        max_core_p95_ms=_metric(args.max_core_p95_ms, "max_core_p95_ms"),
        max_core_max_ms=_metric(args.max_core_max_ms, "max_core_max_ms"),
        max_wrapper_p95_ms=_metric(args.max_wrapper_p95_ms, "max_wrapper_p95_ms"),
        max_wrapper_max_ms=_metric(args.max_wrapper_max_ms, "max_wrapper_max_ms"),
        max_proposal_cap_drops=_nonnegative_int(args.max_proposal_cap_drops, "max_proposal_cap_drops"),
        max_track_capacity_drops=_nonnegative_int(args.max_track_capacity_drops, "max_track_capacity_drops"),
    )
    _require(
        limits.min_average_fps > 0.0
        and limits.min_paired_fps_ratio > 0.0
        and limits.max_core_p95_ms > 0.0
        and limits.max_core_max_ms >= limits.max_core_p95_ms
        and limits.max_wrapper_p95_ms > 0.0
        and limits.max_wrapper_max_ms >= limits.max_wrapper_p95_ms,
        "invalid audit thresholds",
    )
    scenes = read_scene_list(args.scene_list.resolve())
    def directory(path: Path, label: str) -> Path:
        _require(not path.is_symlink(), f"{label} must not be a symlink")
        resolved = path.resolve()
        _require(resolved.is_dir(), f"{label} is not a regular directory")
        return resolved

    roots = {
        "prediction_root": directory(args.prediction_root, "prediction_root"),
        "diagnostics_root": directory(args.diagnostics_root, "diagnostics_root"),
        "log_root": directory(args.log_root, "log_root"),
    }
    anchor_root = (
        None if args.anchor_root is None else directory(args.anchor_root, "anchor_root")
    )
    control_log_root = (
        None
        if args.control_log_root is None
        else directory(args.control_log_root, "control_log_root")
    )
    rows = [
        _audit_scene(
            scene=scene,
            prediction_root=roots["prediction_root"],
            diagnostics_root=roots["diagnostics_root"],
            log_root=roots["log_root"],
            anchor_root=anchor_root,
            control_log_root=control_log_root,
            limits=limits,
            contract=contract,
        )
        for scene in scenes
    ]
    report = {
        "schema": contract.report_schema,
        "contract": contract.name,
        "expected_summary_schema": contract.summary_schema,
        "expected_sidecar_schema": contract.summary_schema,
        "passed": True,
        "ground_truth_access": False,
        "counterfactual_predictions_materialized": False,
        "native_prediction_identity_checked": True,
        "same_run_anchor_byte_identity_checked": anchor_root is not None,
        "paired_realtime_checked": control_log_root is not None,
        "frozen_memory_and_compute_caps_checked": True,
        "loo_receipt_dominance_checked": True,
        "scene_list": str(args.scene_list.resolve()),
        "scene_count": len(rows),
        "native_rows": sum(row["native_rows"] for row in rows),
        "would_replace": sum(row["would_replace"] for row in rows),
        "minimum_average_fps": min(row["average_fps"] for row in rows),
        "minimum_paired_fps_ratio": (
            min(row["paired_fps_ratio"] for row in rows)
            if control_log_root is not None
            else None
        ),
        "limits": vars(limits),
        "scenes": rows,
    }
    _write_create_only(args.report, report)
    return report


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--scene-list", type=Path, required=True)
    value.add_argument(
        "--contract",
        choices=tuple(AUDIT_CONTRACTS),
        default="visibility-v2",
        help="one frozen artifact contract; mixed v1/v2 scene lists fail",
    )
    value.add_argument("--prediction-root", type=Path, required=True)
    value.add_argument("--diagnostics-root", type=Path, required=True)
    value.add_argument("--log-root", type=Path, required=True)
    value.add_argument("--anchor-root", type=Path)
    value.add_argument("--control-log-root", type=Path)
    value.add_argument("--report", type=Path, required=True)
    value.add_argument("--min-average-fps", type=float, default=20.0)
    value.add_argument("--min-paired-fps-ratio", type=float, default=0.95)
    value.add_argument("--max-core-p95-ms", type=float, default=10.0)
    value.add_argument("--max-core-max-ms", type=float, default=50.0)
    value.add_argument("--max-wrapper-p95-ms", type=float, default=10.0)
    value.add_argument("--max-wrapper-max-ms", type=float, default=50.0)
    value.add_argument("--max-proposal-cap-drops", type=int, default=0)
    value.add_argument("--max-track-capacity-drops", type=int, default=0)
    return value


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    report = audit(args)
    print(
        "OpenBox-SMOV R2 audit passed:",
        f"scenes={report['scene_count']}",
        f"native_rows={report['native_rows']}",
        f"would_replace={report['would_replace']}",
        f"min_fps={report['minimum_average_fps']:.2f}",
    )
    print("Immutable report:", args.report.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
