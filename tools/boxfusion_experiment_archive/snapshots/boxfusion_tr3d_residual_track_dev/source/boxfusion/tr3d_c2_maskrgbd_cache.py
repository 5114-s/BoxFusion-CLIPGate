"""Immutable, pickle-free C2 Mask-RGBD observer sidecars."""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile
from typing import Mapping

import numpy as np

from .tr3d_c2_maskrgbd_observer import (
    C2MaskRGBDConfig,
    C2SceneObservation,
    GATE_NAMES,
)


SCHEMA = "boxfusion.tr3d_c2_maskrgbd_confirmation.v1"
_SCENE_RE = re.compile(r"^scene[0-9]{4}_[0-9]{2}$")
_PREFIX_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
_SHA_RE = re.compile(r"^[0-9a-f]{64}$")


def canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sidecar_path(root: str | os.PathLike[str], scene_id: str, prefix_id: str) -> Path:
    if _SCENE_RE.fullmatch(scene_id) is None or _PREFIX_RE.fullmatch(prefix_id) is None:
        raise ValueError("invalid C2 scene or prefix")
    return Path(root) / scene_id / f"{prefix_id}.c2-maskrgbd.npz"


def _readonly(value: object, dtype: np.dtype) -> np.ndarray:
    result = np.ascontiguousarray(np.asarray(value, dtype=dtype))
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class TR3DC2MaskRGBDCache:
    scene_id: str
    prefix_id: str
    c1_sidecar_sha256: str
    parent_cache_sha256: str
    anchor_prediction_sha256: str
    teacher_manifest_set_sha256: str
    runtime_manifest_set_sha256: str
    scene_frame_input_sha256: str
    config_sha256: str
    code_sha256: str
    config_json: str
    source_c1_rows: np.ndarray
    source_ranks: np.ndarray
    proposal_ids: np.ndarray
    parent_rows: np.ndarray
    c1_track_scores: np.ndarray
    frame_cache_sha256: np.ndarray
    observation: C2SceneObservation
    runtime_s: float

    @property
    def candidate_count(self) -> int:
        return int(len(self.proposal_ids))

    def as_payload(self) -> dict[str, np.ndarray]:
        observation = self.observation
        payload = {
            "schema": np.asarray(SCHEMA),
            "complete": np.asarray(True, dtype=np.bool_),
            "observer_only": np.asarray(True, dtype=np.bool_),
            "mutation_enabled": np.asarray(False, dtype=np.bool_),
            "applied_count": np.asarray(0, dtype=np.int64),
            "ground_truth_access": np.asarray(False, dtype=np.bool_),
            "clip_access": np.asarray(False, dtype=np.bool_),
            "clip_semantics_unchanged": np.asarray(True, dtype=np.bool_),
            "teacher_labels_used_for_gate": np.asarray(False, dtype=np.bool_),
            "scene_id": np.asarray(self.scene_id),
            "prefix_id": np.asarray(self.prefix_id),
            "c1_sidecar_sha256": np.asarray(self.c1_sidecar_sha256),
            "parent_cache_sha256": np.asarray(self.parent_cache_sha256),
            "anchor_prediction_sha256": np.asarray(self.anchor_prediction_sha256),
            "teacher_manifest_set_sha256": np.asarray(self.teacher_manifest_set_sha256),
            "runtime_manifest_set_sha256": np.asarray(self.runtime_manifest_set_sha256),
            "scene_frame_input_sha256": np.asarray(self.scene_frame_input_sha256),
            "config_sha256": np.asarray(self.config_sha256),
            "code_sha256": np.asarray(self.code_sha256),
            "config_json": np.asarray(self.config_json),
            "gate_names": np.asarray(GATE_NAMES),
            "source_c1_rows": np.asarray(self.source_c1_rows, dtype=np.int64),
            "source_ranks": np.asarray(self.source_ranks, dtype=np.int32),
            "proposal_ids": np.asarray(self.proposal_ids, dtype=np.int64),
            "parent_rows": np.asarray(self.parent_rows, dtype=np.int64),
            "c1_track_scores": np.asarray(self.c1_track_scores, dtype=np.float32),
            "frame_cache_sha256": np.asarray(self.frame_cache_sha256),
            "runtime_s": np.asarray(self.runtime_s, dtype=np.float64),
        }
        for name in C2SceneObservation.__dataclass_fields__:
            payload[name] = np.asarray(getattr(observation, name))
        return payload


def _scalar(values: Mapping[str, np.ndarray], name: str):
    value = np.asarray(values[name])
    if value.shape != () or value.dtype.hasobject:
        raise ValueError(f"{name} must be a non-object scalar")
    return value.item()


def validate_payload(values: Mapping[str, np.ndarray]) -> TR3DC2MaskRGBDCache:
    if str(_scalar(values, "schema")) != SCHEMA:
        raise ValueError("unsupported C2 schema")
    if not bool(_scalar(values, "complete")):
        raise ValueError("incomplete C2 sidecar")
    if not bool(_scalar(values, "observer_only")) or bool(_scalar(values, "mutation_enabled")):
        raise ValueError("C2 observer contract violation")
    if int(_scalar(values, "applied_count")) != 0:
        raise ValueError("C2 applied_count must be zero")
    if bool(_scalar(values, "ground_truth_access")) or bool(_scalar(values, "clip_access")):
        raise ValueError("C2 exporter must not access GT or CLIP")
    if not bool(_scalar(values, "clip_semantics_unchanged")):
        raise ValueError("C2 must preserve CLIP semantics")
    if bool(_scalar(values, "teacher_labels_used_for_gate")):
        raise ValueError("C2 teacher labels must be diagnostic-only")
    scene_id = str(_scalar(values, "scene_id"))
    prefix_id = str(_scalar(values, "prefix_id"))
    if _SCENE_RE.fullmatch(scene_id) is None or _PREFIX_RE.fullmatch(prefix_id) is None:
        raise ValueError("invalid C2 scene or prefix")
    hashes: dict[str, str] = {}
    for name in (
        "c1_sidecar_sha256", "parent_cache_sha256", "anchor_prediction_sha256",
        "teacher_manifest_set_sha256", "runtime_manifest_set_sha256",
        "scene_frame_input_sha256", "config_sha256", "code_sha256",
    ):
        value = str(_scalar(values, name))
        if _SHA_RE.fullmatch(value) is None:
            raise ValueError(f"invalid {name}")
        hashes[name] = value
    config_json = str(_scalar(values, "config_json"))
    config_payload = json.loads(config_json)
    config = C2MaskRGBDConfig(**config_payload)
    if canonical_json(config.as_dict()) != config_json:
        raise ValueError("C2 config JSON is not canonical")
    if sha256_bytes(config_json.encode("utf-8")) != hashes["config_sha256"]:
        raise ValueError("C2 config hash mismatch")
    if not np.array_equal(np.asarray(values["gate_names"]), np.asarray(GATE_NAMES)):
        raise ValueError("C2 gate names mismatch")

    proposal_ids = np.asarray(values["proposal_ids"])
    if proposal_ids.ndim != 1 or proposal_ids.dtype != np.int64:
        raise ValueError("proposal_ids must be int64 [P]")
    p = len(proposal_ids)
    if len(np.unique(proposal_ids)) != p or p > config.source_budget:
        raise ValueError("C2 proposal ids are duplicate or exceed source budget")
    frame_ids = np.asarray(values["frame_ids"])
    if frame_ids.ndim != 1 or frame_ids.dtype != np.int64:
        raise ValueError("frame_ids must be int64 [F]")
    f = len(frame_ids)
    if len(np.unique(frame_ids)) != f:
        raise ValueError("frame ids must be unique")

    def exact(name: str, dtype: np.dtype, shape: tuple[int, ...]) -> np.ndarray:
        array = np.asarray(values[name])
        if array.dtype != np.dtype(dtype) or array.shape != shape:
            raise ValueError(f"{name} must be {np.dtype(dtype)} {shape}")
        return array

    source_rows = exact("source_c1_rows", np.int64, (p,))
    source_ranks = exact("source_ranks", np.int32, (p,))
    parent_rows = exact("parent_rows", np.int64, (p,))
    scores = exact("c1_track_scores", np.float32, (p,))
    cache_hashes = np.asarray(values["frame_cache_sha256"])
    if cache_hashes.shape != (f,) or cache_hashes.dtype.kind != "U":
        raise ValueError("frame cache hashes must be unicode [F]")
    if any(_SHA_RE.fullmatch(str(value)) is None for value in cache_hashes.tolist()):
        raise ValueError("invalid frame cache hash")
    if np.any(source_rows < 0) or len(np.unique(source_rows)) != p:
        raise ValueError("invalid C1 source rows")
    if not np.array_equal(source_ranks, np.arange(1, p + 1, dtype=np.int32)):
        raise ValueError("source ranks must be contiguous one-based ranks")
    if np.any(parent_rows < 0) or len(np.unique(parent_rows)) != p:
        raise ValueError("invalid parent rows")
    if not np.isfinite(scores).all():
        raise ValueError("C1 scores must be finite")

    bool_fields = ("projected_valid", "view_matched", "view_strong")
    int_fields = {
        "best_mask_index": np.int32,
    }
    float_fields = (
        "projected_area_pixels", "best_mask_score", "bbox_iou",
        "mask_containment", "box_coverage", "valid_depth_pixels",
        "sampled_depth_points", "inside_original_fraction",
        "inside_expanded_fraction", "component_point_count",
        "component_inside_fraction", "component_fraction", "evidence_score",
    )
    arrays: dict[str, np.ndarray] = {}
    for name in bool_fields:
        arrays[name] = exact(name, np.bool_, (p, f))
    for name, dtype in int_fields.items():
        arrays[name] = exact(name, dtype, (p, f))
    for name in float_fields:
        arrays[name] = exact(name, np.float32, (p, f))
        if not np.isfinite(arrays[name]).all():
            raise ValueError(f"{name} must be finite")
    labels = np.asarray(values["best_mask_label"])
    if labels.shape != (p, f) or labels.dtype.kind != "U":
        raise ValueError("best mask labels must be diagnostic unicode [P,F]")
    projected_count = exact("projected_view_count", np.int32, (p,))
    matched_count = exact("matched_view_count", np.int32, (p,))
    strong_count = exact("strong_view_count", np.int32, (p,))
    total_component = exact("total_component_points", np.int32, (p,))
    mean_inside = exact("mean_strong_inside_expanded", np.float32, (p,))
    max_evidence = exact("max_evidence_score", np.float32, (p,))
    gates = exact("gate_mask", np.bool_, (p, len(GATE_NAMES)))
    if np.any(arrays["view_strong"] & ~arrays["view_matched"]):
        raise ValueError("strong views must be matched views")
    if np.any(arrays["view_matched"] & ~arrays["projected_valid"]):
        raise ValueError("matched views must be projected views")
    if not np.array_equal(arrays["view_matched"], arrays["best_mask_index"] >= 0):
        raise ValueError("matched-view and best-mask identity mismatch")
    if np.any(arrays["projected_area_pixels"][arrays["projected_valid"]] < config.min_projected_area_pixels):
        raise ValueError("projected view violates minimum projected area")
    nonnegative = (
        "projected_area_pixels", "best_mask_score", "bbox_iou",
        "mask_containment", "box_coverage", "valid_depth_pixels",
        "sampled_depth_points", "inside_original_fraction",
        "inside_expanded_fraction", "component_point_count",
        "component_inside_fraction", "component_fraction", "evidence_score",
    )
    if any(np.any(arrays[name] < 0) for name in nonnegative):
        raise ValueError("C2 evidence values must be nonnegative")
    fractions = (
        "best_mask_score", "bbox_iou", "mask_containment", "box_coverage",
        "inside_original_fraction", "inside_expanded_fraction",
        "component_inside_fraction", "component_fraction", "evidence_score",
    )
    if any(np.any(arrays[name] > 1.0 + 1e-6) for name in fractions):
        raise ValueError("C2 normalized evidence must not exceed one")
    expected_strong = (
        arrays["view_matched"]
        & (arrays["best_mask_score"] >= config.strong_mask_score)
        & (arrays["mask_containment"] >= config.min_mask_containment)
        & (arrays["box_coverage"] >= config.min_box_coverage)
        & (arrays["valid_depth_pixels"] >= config.min_valid_depth_pixels)
        & (arrays["inside_expanded_fraction"] >= config.min_inside_expanded_fraction)
        & (arrays["component_point_count"] >= config.min_component_points)
        & (arrays["component_inside_fraction"] >= config.min_component_inside_fraction)
    )
    if not np.array_equal(arrays["view_strong"], expected_strong):
        raise ValueError("C2 strong-view decision mismatch")
    if not np.array_equal(projected_count, arrays["projected_valid"].sum(1, dtype=np.int32)):
        raise ValueError("projected view count mismatch")
    if not np.array_equal(matched_count, arrays["view_matched"].sum(1, dtype=np.int32)):
        raise ValueError("matched view count mismatch")
    if not np.array_equal(strong_count, arrays["view_strong"].sum(1, dtype=np.int32)):
        raise ValueError("strong view count mismatch")
    expected_total = np.where(
        arrays["view_strong"], arrays["component_point_count"], 0.0
    ).sum(1).astype(np.int32)
    if not np.array_equal(total_component, expected_total):
        raise ValueError("component point aggregate mismatch")
    expected_mean = np.divide(
        np.where(arrays["view_strong"], arrays["inside_expanded_fraction"], 0.0).sum(1),
        strong_count,
        out=np.zeros(p, dtype=np.float64), where=strong_count > 0,
    ).astype(np.float32)
    if not np.allclose(mean_inside, expected_mean, rtol=0, atol=1e-7):
        raise ValueError("mean strong inside fraction mismatch")
    expected_max = arrays["evidence_score"].max(1, initial=0.0)
    if not np.allclose(max_evidence, expected_max, rtol=0, atol=1e-7):
        raise ValueError("maximum evidence mismatch")
    expected_gates = np.stack(
        (
            matched_count >= 1,
            strong_count >= 1,
            strong_count >= 2,
            (strong_count >= 2)
            & (total_component >= config.mask2_min_total_component_points)
            & (mean_inside >= config.mask2_min_mean_inside_expanded),
            (strong_count >= 3)
            & (total_component >= config.mask3_min_total_component_points)
            & (mean_inside >= config.mask3_min_mean_inside_expanded),
        ), axis=1,
    )
    if not np.array_equal(gates, expected_gates):
        raise ValueError("C2 gate decision mismatch")
    runtime = float(_scalar(values, "runtime_s"))
    if not math.isfinite(runtime) or runtime < 0:
        raise ValueError("invalid C2 runtime")
    observation = C2SceneObservation(
        frame_ids=_readonly(frame_ids, np.int64),
        projected_valid=_readonly(arrays["projected_valid"], np.bool_),
        projected_area_pixels=_readonly(arrays["projected_area_pixels"], np.float32),
        best_mask_index=_readonly(arrays["best_mask_index"], np.int32),
        best_mask_score=_readonly(arrays["best_mask_score"], np.float32),
        best_mask_label=_readonly(labels, labels.dtype),
        bbox_iou=_readonly(arrays["bbox_iou"], np.float32),
        mask_containment=_readonly(arrays["mask_containment"], np.float32),
        box_coverage=_readonly(arrays["box_coverage"], np.float32),
        valid_depth_pixels=_readonly(arrays["valid_depth_pixels"], np.float32),
        sampled_depth_points=_readonly(arrays["sampled_depth_points"], np.float32),
        inside_original_fraction=_readonly(arrays["inside_original_fraction"], np.float32),
        inside_expanded_fraction=_readonly(arrays["inside_expanded_fraction"], np.float32),
        component_point_count=_readonly(arrays["component_point_count"], np.float32),
        component_inside_fraction=_readonly(arrays["component_inside_fraction"], np.float32),
        component_fraction=_readonly(arrays["component_fraction"], np.float32),
        evidence_score=_readonly(arrays["evidence_score"], np.float32),
        view_matched=_readonly(arrays["view_matched"], np.bool_),
        view_strong=_readonly(arrays["view_strong"], np.bool_),
        projected_view_count=_readonly(projected_count, np.int32),
        matched_view_count=_readonly(matched_count, np.int32),
        strong_view_count=_readonly(strong_count, np.int32),
        total_component_points=_readonly(total_component, np.int32),
        mean_strong_inside_expanded=_readonly(mean_inside, np.float32),
        max_evidence_score=_readonly(max_evidence, np.float32),
        gate_mask=_readonly(gates, np.bool_),
    )
    return TR3DC2MaskRGBDCache(
        scene_id=scene_id, prefix_id=prefix_id, config_json=config_json,
        source_c1_rows=_readonly(source_rows, np.int64),
        source_ranks=_readonly(source_ranks, np.int32),
        proposal_ids=_readonly(proposal_ids, np.int64),
        parent_rows=_readonly(parent_rows, np.int64),
        c1_track_scores=_readonly(scores, np.float32),
        frame_cache_sha256=_readonly(cache_hashes, cache_hashes.dtype),
        observation=observation, runtime_s=runtime, **hashes,
    )


def write_sidecar(path: str | os.PathLike[str], cache: TR3DC2MaskRGBDCache) -> str:
    canonical = validate_payload(cache.as_payload())
    buffer = BytesIO()
    np.savez_compressed(buffer, **canonical.as_payload())
    encoded = buffer.getvalue()
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=f".{target.name}.", suffix=".tmp", dir=target.parent, delete=False
        ) as handle:
            temporary_name = handle.name
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary_name, target)
        target.chmod(0o444)
    except FileExistsError as error:
        raise FileExistsError(f"immutable C2 sidecar exists: {target}") from error
    finally:
        if temporary_name is not None:
            try:
                os.unlink(temporary_name)
            except FileNotFoundError:
                pass
    return sha256_file(target)


def load_sidecar(path: str | os.PathLike[str]) -> TR3DC2MaskRGBDCache:
    with np.load(Path(path), allow_pickle=False) as archive:
        payload = {name: archive[name] for name in archive.files}
    return validate_payload(payload)


__all__ = [
    "SCHEMA", "TR3DC2MaskRGBDCache", "canonical_json", "load_sidecar",
    "sha256_bytes", "sha256_file", "sidecar_path", "validate_payload",
    "write_sidecar",
]
