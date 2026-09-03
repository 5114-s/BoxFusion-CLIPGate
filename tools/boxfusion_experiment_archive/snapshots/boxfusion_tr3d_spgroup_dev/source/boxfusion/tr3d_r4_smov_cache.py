"""Immutable observer-only sidecars for paired terminal-R3 depth evidence."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from io import BytesIO
import math
import os
from pathlib import Path
import re
import tempfile
from typing import Mapping

import numpy as np

from .tr3d_r4_smov_observer import R4PairedDepthObservation


TR3D_R4_DEPTH_SCHEMA = "boxfusion.tr3d_r4_smov_depth_observer.v1"
_SCENE_RE = re.compile(r"^scene[0-9]{4}_[0-9]{2}$")
_PREFIX_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
_SHA_RE = re.compile(r"^[0-9a-f]{64}$")
_FIELDS = frozenset(
    {
        "schema", "complete", "observer_only", "mutation_enabled",
        "applied_count", "ground_truth_access", "clip_access",
        "feature_consistency_enabled", "scene_id", "prefix_id",
        "final_source_timestamp", "parent_cache_sha256",
        "prefix_manifest_row_sha256", "frame_artifact_tree_sha256",
        "r3_diagnostic_sha256", "input_geometry_sha256",
        "input_scores_sha256", "r4_config_sha256", "r4_code_sha256",
        "proposal_ids", "anchor_indices", "tr3d_scores", "anchor_scores",
        "anchor_iou", "anchor_boxes_world", "candidate_boxes_world",
        "topk_frame_ids", "topk_view_valid",
        "topk_projected_area_pixels", "topk_projected_area_fraction",
        "per_view_depth_counts", "per_view_depth_evidence",
        "per_view_point_count", "aggregate_depth_counts",
        "aggregate_depth_evidence", "aggregate_view_count",
        "aggregate_point_count", "candidate_minus_anchor_evidence",
        "runtime_s",
    }
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha(value: str, name: str) -> str:
    if not isinstance(value, str) or _SHA_RE.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA256")
    return value


def _readonly(value: object, dtype: np.dtype) -> np.ndarray:
    result = np.ascontiguousarray(value, dtype=dtype)
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class R4DepthSidecar:
    scene_id: str
    prefix_id: str
    final_source_timestamp: int
    parent_cache_sha256: str
    prefix_manifest_row_sha256: str
    frame_artifact_tree_sha256: str
    r3_diagnostic_sha256: str
    input_geometry_sha256: str
    input_scores_sha256: str
    r4_config_sha256: str
    r4_code_sha256: str
    proposal_ids: np.ndarray
    anchor_indices: np.ndarray
    tr3d_scores: np.ndarray
    anchor_scores: np.ndarray
    anchor_iou: np.ndarray
    anchor_boxes_world: np.ndarray
    candidate_boxes_world: np.ndarray
    topk_frame_ids: np.ndarray
    topk_view_valid: np.ndarray
    topk_projected_area_pixels: np.ndarray
    topk_projected_area_fraction: np.ndarray
    per_view_depth_counts: np.ndarray
    per_view_depth_evidence: np.ndarray
    per_view_point_count: np.ndarray
    aggregate_depth_counts: np.ndarray
    aggregate_depth_evidence: np.ndarray
    aggregate_view_count: np.ndarray
    aggregate_point_count: np.ndarray
    candidate_minus_anchor_evidence: np.ndarray
    runtime_s: float

    @property
    def pair_count(self) -> int:
        return int(self.proposal_ids.shape[0])

    def as_npz_payload(self) -> dict[str, np.ndarray]:
        return {
            "schema": np.asarray(TR3D_R4_DEPTH_SCHEMA),
            "complete": np.asarray(True, dtype=np.bool_),
            "observer_only": np.asarray(True, dtype=np.bool_),
            "mutation_enabled": np.asarray(False, dtype=np.bool_),
            "applied_count": np.asarray(0, dtype=np.int64),
            "ground_truth_access": np.asarray(False, dtype=np.bool_),
            "clip_access": np.asarray(False, dtype=np.bool_),
            "feature_consistency_enabled": np.asarray(False, dtype=np.bool_),
            "scene_id": np.asarray(self.scene_id),
            "prefix_id": np.asarray(self.prefix_id),
            "final_source_timestamp": np.asarray(
                self.final_source_timestamp, dtype=np.int64
            ),
            "parent_cache_sha256": np.asarray(self.parent_cache_sha256),
            "prefix_manifest_row_sha256": np.asarray(
                self.prefix_manifest_row_sha256
            ),
            "frame_artifact_tree_sha256": np.asarray(
                self.frame_artifact_tree_sha256
            ),
            "r3_diagnostic_sha256": np.asarray(self.r3_diagnostic_sha256),
            "input_geometry_sha256": np.asarray(self.input_geometry_sha256),
            "input_scores_sha256": np.asarray(self.input_scores_sha256),
            "r4_config_sha256": np.asarray(self.r4_config_sha256),
            "r4_code_sha256": np.asarray(self.r4_code_sha256),
            "proposal_ids": np.asarray(self.proposal_ids, dtype=np.int64),
            "anchor_indices": np.asarray(self.anchor_indices, dtype=np.int64),
            "tr3d_scores": np.asarray(self.tr3d_scores, dtype=np.float32),
            "anchor_scores": np.asarray(self.anchor_scores, dtype=np.float32),
            "anchor_iou": np.asarray(self.anchor_iou, dtype=np.float32),
            "anchor_boxes_world": np.asarray(
                self.anchor_boxes_world, dtype=np.float32
            ),
            "candidate_boxes_world": np.asarray(
                self.candidate_boxes_world, dtype=np.float32
            ),
            "topk_frame_ids": np.asarray(self.topk_frame_ids, dtype=np.int64),
            "topk_view_valid": np.asarray(
                self.topk_view_valid, dtype=np.bool_
            ),
            "topk_projected_area_pixels": np.asarray(
                self.topk_projected_area_pixels, dtype=np.float32
            ),
            "topk_projected_area_fraction": np.asarray(
                self.topk_projected_area_fraction, dtype=np.float32
            ),
            "per_view_depth_counts": np.asarray(
                self.per_view_depth_counts, dtype=np.int32
            ),
            "per_view_depth_evidence": np.asarray(
                self.per_view_depth_evidence, dtype=np.float32
            ),
            "per_view_point_count": np.asarray(
                self.per_view_point_count, dtype=np.int32
            ),
            "aggregate_depth_counts": np.asarray(
                self.aggregate_depth_counts, dtype=np.int64
            ),
            "aggregate_depth_evidence": np.asarray(
                self.aggregate_depth_evidence, dtype=np.float32
            ),
            "aggregate_view_count": np.asarray(
                self.aggregate_view_count, dtype=np.int32
            ),
            "aggregate_point_count": np.asarray(
                self.aggregate_point_count, dtype=np.int64
            ),
            "candidate_minus_anchor_evidence": np.asarray(
                self.candidate_minus_anchor_evidence, dtype=np.float32
            ),
            "runtime_s": np.asarray(self.runtime_s, dtype=np.float64),
        }


def make_r4_depth_sidecar(
    *,
    observation: R4PairedDepthObservation,
    scene_id: str,
    prefix_id: str,
    final_source_timestamp: int,
    parent_cache_sha256: str,
    prefix_manifest_row_sha256: str,
    frame_artifact_tree_sha256: str,
    r3_diagnostic_sha256: str,
    input_geometry_sha256: str,
    input_scores_sha256: str,
    r4_config_sha256: str,
    r4_code_sha256: str,
    tr3d_scores: object,
    anchor_scores: object,
    anchor_iou: object,
    anchor_boxes_world: object,
    candidate_boxes_world: object,
) -> R4DepthSidecar:
    if observation.scene_id != scene_id:
        raise ValueError("observation scene mismatch")
    count = observation.pair_count
    sidecar = R4DepthSidecar(
        scene_id=scene_id,
        prefix_id=prefix_id,
        final_source_timestamp=int(final_source_timestamp),
        parent_cache_sha256=_sha(parent_cache_sha256, "parent_cache_sha256"),
        prefix_manifest_row_sha256=_sha(
            prefix_manifest_row_sha256, "prefix_manifest_row_sha256"
        ),
        frame_artifact_tree_sha256=_sha(
            frame_artifact_tree_sha256, "frame_artifact_tree_sha256"
        ),
        r3_diagnostic_sha256=_sha(
            r3_diagnostic_sha256, "r3_diagnostic_sha256"
        ),
        input_geometry_sha256=_sha(
            input_geometry_sha256, "input_geometry_sha256"
        ),
        input_scores_sha256=_sha(input_scores_sha256, "input_scores_sha256"),
        r4_config_sha256=_sha(r4_config_sha256, "r4_config_sha256"),
        r4_code_sha256=_sha(r4_code_sha256, "r4_code_sha256"),
        proposal_ids=_readonly(observation.proposal_ids, np.int64),
        anchor_indices=_readonly(observation.anchor_indices, np.int64),
        tr3d_scores=_readonly(tr3d_scores, np.float32),
        anchor_scores=_readonly(anchor_scores, np.float32),
        anchor_iou=_readonly(anchor_iou, np.float32),
        anchor_boxes_world=_readonly(anchor_boxes_world, np.float32),
        candidate_boxes_world=_readonly(candidate_boxes_world, np.float32),
        topk_frame_ids=_readonly(observation.topk_frame_ids, np.int64),
        topk_view_valid=_readonly(observation.topk_view_valid, np.bool_),
        topk_projected_area_pixels=_readonly(
            observation.topk_projected_area_pixels, np.float32
        ),
        topk_projected_area_fraction=_readonly(
            observation.topk_projected_area_fraction, np.float32
        ),
        per_view_depth_counts=_readonly(
            observation.per_view_depth_counts, np.int32
        ),
        per_view_depth_evidence=_readonly(
            observation.per_view_depth_evidence, np.float32
        ),
        per_view_point_count=_readonly(
            observation.per_view_point_count, np.int32
        ),
        aggregate_depth_counts=_readonly(
            observation.aggregate_depth_counts, np.int64
        ),
        aggregate_depth_evidence=_readonly(
            observation.aggregate_depth_evidence, np.float32
        ),
        aggregate_view_count=_readonly(
            observation.aggregate_view_count, np.int32
        ),
        aggregate_point_count=_readonly(
            observation.aggregate_point_count, np.int64
        ),
        candidate_minus_anchor_evidence=_readonly(
            observation.candidate_minus_anchor_evidence, np.float32
        ),
        runtime_s=float(observation.runtime_s),
    )
    _validate_sidecar(sidecar)
    if any(
        np.asarray(value).shape != (count,)
        for value in (sidecar.tr3d_scores, sidecar.anchor_scores, sidecar.anchor_iou)
    ):
        raise ValueError("score/IoU arrays must have shape [N]")
    return sidecar


def _fractions(counts: np.ndarray) -> np.ndarray:
    totals = counts.sum(axis=-1, dtype=np.int64)
    output = np.zeros(counts.shape, dtype=np.float64)
    np.divide(counts, totals[..., None], out=output, where=totals[..., None] > 0)
    return output.astype(np.float32)


def _validate_sidecar(value: R4DepthSidecar) -> None:
    if _SCENE_RE.fullmatch(value.scene_id) is None:
        raise ValueError("invalid scene_id")
    if _PREFIX_RE.fullmatch(value.prefix_id) is None:
        raise ValueError("invalid prefix_id")
    if value.final_source_timestamp < 0:
        raise ValueError("final_source_timestamp must be nonnegative")
    if not math.isfinite(value.runtime_s) or value.runtime_s < 0.0:
        raise ValueError("runtime_s must be finite and nonnegative")
    count = value.pair_count
    if (
        value.anchor_indices.shape != (count,)
        or len(np.unique(value.proposal_ids)) != count
        or len(np.unique(value.anchor_indices)) != count
        or np.any(value.proposal_ids < 0)
        or np.any(value.anchor_indices < 0)
    ):
        raise ValueError("proposal/anchor ids must be unique and nonnegative")
    topk = value.topk_frame_ids.shape[1]
    expected = {
        "topk_view_valid": (count, topk),
        "topk_projected_area_pixels": (count, topk, 2),
        "topk_projected_area_fraction": (count, topk, 2),
        "per_view_depth_counts": (count, topk, 2, 4),
        "per_view_depth_evidence": (count, topk, 2, 4),
        "per_view_point_count": (count, topk, 2),
        "aggregate_depth_counts": (count, 2, 4),
        "aggregate_depth_evidence": (count, 2, 4),
        "aggregate_view_count": (count,),
        "aggregate_point_count": (count, 2),
        "candidate_minus_anchor_evidence": (count, 4),
        "anchor_boxes_world": (count, 7),
        "candidate_boxes_world": (count, 7),
    }
    for name, shape in expected.items():
        if np.asarray(getattr(value, name)).shape != shape:
            raise ValueError(f"{name} must have shape {shape}")
    if topk < 1:
        raise ValueError("Top-K dimension must be positive")
    valid = value.topk_view_valid
    if topk > 1 and np.any((~valid[:, :-1]) & valid[:, 1:]):
        raise ValueError("valid views must occupy leading slots")
    if np.any(value.topk_frame_ids[valid] < 0) or np.any(
        value.topk_frame_ids[~valid] != -1
    ):
        raise ValueError("Top-K frame sentinel mismatch")
    if np.any(value.per_view_depth_counts < 0):
        raise ValueError("depth counts must be nonnegative")
    if not np.array_equal(
        value.per_view_point_count,
        value.per_view_depth_counts.sum(axis=3, dtype=np.int32),
    ):
        raise ValueError("per-view point counts disagree")
    if not np.array_equal(
        value.aggregate_depth_counts,
        value.per_view_depth_counts.sum(axis=1, dtype=np.int64),
    ):
        raise ValueError("aggregate depth counts disagree")
    if not np.array_equal(
        value.aggregate_point_count,
        value.aggregate_depth_counts.sum(axis=2, dtype=np.int64),
    ):
        raise ValueError("aggregate point counts disagree")
    if not np.array_equal(
        value.aggregate_view_count,
        valid.sum(axis=1, dtype=np.int32),
    ):
        raise ValueError("aggregate view counts disagree")
    if not np.array_equal(
        value.per_view_depth_evidence,
        _fractions(value.per_view_depth_counts),
    ) or not np.array_equal(
        value.aggregate_depth_evidence,
        _fractions(value.aggregate_depth_counts),
    ):
        raise ValueError("depth evidence is not canonical counts/total")
    expected_delta = (
        value.aggregate_depth_evidence[:, 1]
        - value.aggregate_depth_evidence[:, 0]
    ).astype(np.float32)
    if not np.array_equal(value.candidate_minus_anchor_evidence, expected_delta):
        raise ValueError("paired depth delta disagrees")
    arrays = value.as_npz_payload()
    numeric = [item for item in arrays.values() if item.dtype.kind in "f"]
    if any(not np.isfinite(item).all() for item in numeric):
        raise ValueError("R4 sidecar contains non-finite values")


def write_r4_depth_sidecar(path: str | os.PathLike[str], value: R4DepthSidecar) -> None:
    """Atomically create one immutable sidecar; overwrite is forbidden."""

    _validate_sidecar(value)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    buffer = BytesIO()
    np.savez(buffer, **value.as_npz_payload())
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", prefix=f".{target.name}.", suffix=".tmp",
            dir=target.parent, delete=False,
        ) as handle:
            temporary = handle.name
            handle.write(buffer.getvalue())
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, target)
        target.chmod(0o444)
    except FileExistsError as error:
        raise FileExistsError(f"immutable R4 sidecar exists: {target}") from error
    finally:
        if temporary is not None:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass


def load_r4_depth_sidecar(path: str | os.PathLike[str]) -> R4DepthSidecar:
    source = Path(path)
    if not source.is_file() or source.is_symlink():
        raise FileNotFoundError(source)
    raw = source.read_bytes()
    with np.load(BytesIO(raw), allow_pickle=False) as archive:
        values = {name: np.array(archive[name], copy=True) for name in archive.files}
    if frozenset(values) != _FIELDS or any(v.dtype.hasobject for v in values.values()):
        raise ValueError("R4 sidecar fields disagree or contain object arrays")

    def scalar(name: str, dtype: np.dtype):
        item = np.asarray(values[name])
        if item.shape != () or item.dtype != np.dtype(dtype):
            raise ValueError(f"{name} has invalid scalar dtype")
        return item.item()

    def text(name: str) -> str:
        item = np.asarray(values[name])
        if item.shape != () or item.dtype.hasobject:
            raise ValueError(f"{name} has invalid text scalar")
        result = item.item()
        if isinstance(result, bytes):
            result = result.decode("utf-8")
        if not isinstance(result, str):
            raise ValueError(f"{name} has invalid text scalar")
        return result

    if text("schema") != TR3D_R4_DEPTH_SCHEMA:
        raise ValueError("unsupported R4 sidecar schema")
    contracts = {
        "complete": True,
        "observer_only": True,
        "mutation_enabled": False,
        "ground_truth_access": False,
        "clip_access": False,
        "feature_consistency_enabled": False,
    }
    for name, expected in contracts.items():
        if bool(scalar(name, np.bool_)) is not expected:
            raise ValueError(f"R4 safety contract failed: {name}")
    if int(scalar("applied_count", np.int64)) != 0:
        raise ValueError("R4 applied_count must be zero")
    kwargs = {
        "scene_id": text("scene_id"),
        "prefix_id": text("prefix_id"),
        "final_source_timestamp": int(scalar("final_source_timestamp", np.int64)),
        "runtime_s": float(scalar("runtime_s", np.float64)),
    }
    for name in (
        "parent_cache_sha256", "prefix_manifest_row_sha256",
        "frame_artifact_tree_sha256", "r3_diagnostic_sha256",
        "input_geometry_sha256", "input_scores_sha256", "r4_config_sha256",
        "r4_code_sha256",
    ):
        kwargs[name] = _sha(text(name), name)
    for name in _FIELDS - {
        "schema", "complete", "observer_only", "mutation_enabled",
        "applied_count", "ground_truth_access", "clip_access",
        "feature_consistency_enabled", "scene_id", "prefix_id",
        "final_source_timestamp", "runtime_s",
        "parent_cache_sha256", "prefix_manifest_row_sha256",
        "frame_artifact_tree_sha256", "r3_diagnostic_sha256",
        "input_geometry_sha256", "input_scores_sha256", "r4_config_sha256",
        "r4_code_sha256",
    }:
        kwargs[name] = values[name]
    sidecar = R4DepthSidecar(**kwargs)
    _validate_sidecar(sidecar)
    return sidecar


__all__ = [
    "R4DepthSidecar", "TR3D_R4_DEPTH_SCHEMA", "load_r4_depth_sidecar",
    "make_r4_depth_sidecar", "write_r4_depth_sidecar",
]
