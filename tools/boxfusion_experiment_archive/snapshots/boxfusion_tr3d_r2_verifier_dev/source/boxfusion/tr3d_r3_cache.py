"""Immutable R3 anchor-near correction-observer sidecars.

Each sidecar is a derived, observer-only view of one exact TR3D parent and a
content-addressed frozen G0 prediction.  R2a depth and R2b DINO evidence are
optional: when absent their exact parent hashes are empty and every evidence
array uses a validated zero sentinel.  Loading recomputes the complete R3
observation from the authoritative parents and rejects any disagreement.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from io import BytesIO
import hashlib
import math
import os
from pathlib import Path
import re
import tempfile
from typing import Mapping

import numpy as np

from .frozen_anchor_manifest import verify_frozen_anchor_manifest
from .tr3d_r2_cache import load_tr3d_r2_cache
from .tr3d_r2b_cache import load_tr3d_r2b_cache
from .tr3d_r3_observer import (
    DEPTH_EVIDENCE_DIM,
    TR3D_R3_NEAR_ANCHOR_IOU,
    TR3DR3NearObservation,
    load_axis_alignment_input_metadata,
    load_frozen_anchor_prediction,
    observe_anchor_near_candidates,
)
from .tr3d_residual_cache import load_tr3d_residual_cache


TR3D_R3_CACHE_SCHEMA = "boxfusion.tr3d_r3_anchor_near_correction.v1"
TR3D_R3_ASSOCIATION_FRAME = "scannet_axis_aligned_input_metadata"
TR3D_R3_GEOMETRY_FRAME = "scannet_unaligned_world"

_SCENE_RE = re.compile(r"^scene[0-9]{4}_[0-9]{2}$")
_PREFIX_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

_PROVENANCE_FIELDS = (
    "parent_tr3d_cache_sha256",
    "parent_checkpoint_sha256",
    "parent_config_sha256",
    "parent_source_scene_sha256",
    "parent_axis_alignment_sha256",
    "parent_r2a_cache_sha256",
    "parent_r2b_cache_sha256",
    "parent_prefix_manifest_row_sha256",
    "parent_frame_artifact_tree_sha256",
    "parent_r2_config_sha256",
    "parent_r2_code_sha256",
    "parent_feature_checkpoint_sha256",
    "parent_feature_config_sha256",
    "parent_feature_code_sha256",
    "frozen_anchor_manifest_sha256",
    "frozen_anchor_prediction_sha256",
    "frozen_anchor_prediction_tree_sha256",
    "axis_alignment_metadata_sha256",
    "r3_config_sha256",
    "r3_code_sha256",
)

_ARRAY_DTYPES: dict[str, np.dtype] = {
    "proposal_ids": np.dtype(np.int64),
    "lineage_ids": np.dtype(np.int64),
    "proposal_corners_world": np.dtype(np.float32),
    "anchor_index": np.dtype(np.int64),
    "anchor_iou": np.dtype(np.float32),
    "center_distance_m": np.dtype(np.float32),
    "center_distance_over_anchor_diagonal": np.dtype(np.float32),
    "volume_ratio": np.dtype(np.float32),
    "tr3d_score": np.dtype(np.float32),
    "anchor_score": np.dtype(np.float32),
    "point_count": np.dtype(np.int32),
    "point_density_m3": np.dtype(np.float32),
    "r2a_evidence_available": np.dtype(np.bool_),
    "r2a_depth_evidence": np.dtype(np.float32),
    "r2a_depth_quality": np.dtype(np.float32),
    "r2a_view_count": np.dtype(np.int32),
    "r2a_point_count": np.dtype(np.int64),
    "r2b_feature_available": np.dtype(np.bool_),
    "r2b_multiview_available": np.dtype(np.bool_),
    "r2b_feature_view_count": np.dtype(np.int32),
    "r2b_pairwise_cosine_count": np.dtype(np.int32),
    "r2b_pairwise_cosine_mean": np.dtype(np.float32),
    "r2b_pairwise_cosine_median": np.dtype(np.float32),
    "r2b_pairwise_cosine_min": np.dtype(np.float32),
    "r2b_pairwise_cosine_max": np.dtype(np.float32),
    "r2b_pairwise_cosine_std": np.dtype(np.float32),
}

_FIELDS = frozenset(
    {
        "schema",
        "complete",
        "observer_only",
        "mutation_enabled",
        "applied_count",
        "scene_id",
        "sample_idx",
        "prefix_id",
        "prefix_fraction",
        "association_frame",
        "proposal_geometry_frame",
        "axis_alignment",
        "near_anchor_iou",
        "anchor_count",
        "parent_r2a_available",
        "parent_r2b_available",
        "runtime_s",
        *_PROVENANCE_FIELDS,
        *_ARRAY_DTYPES,
    }
)


@dataclass(frozen=True)
class TR3DR3NearCache:
    scene_id: str
    prefix_id: str
    prefix_fraction: float
    axis_alignment: np.ndarray
    near_anchor_iou: float
    anchor_count: int
    parent_r2a_available: bool
    parent_r2b_available: bool
    parent_tr3d_cache_sha256: str
    parent_checkpoint_sha256: str
    parent_config_sha256: str
    parent_source_scene_sha256: str
    parent_axis_alignment_sha256: str
    parent_r2a_cache_sha256: str
    parent_r2b_cache_sha256: str
    parent_prefix_manifest_row_sha256: str
    parent_frame_artifact_tree_sha256: str
    parent_r2_config_sha256: str
    parent_r2_code_sha256: str
    parent_feature_checkpoint_sha256: str
    parent_feature_config_sha256: str
    parent_feature_code_sha256: str
    frozen_anchor_manifest_sha256: str
    frozen_anchor_prediction_sha256: str
    frozen_anchor_prediction_tree_sha256: str
    axis_alignment_metadata_sha256: str
    r3_config_sha256: str
    r3_code_sha256: str
    proposal_ids: np.ndarray
    lineage_ids: np.ndarray
    proposal_corners_world: np.ndarray
    anchor_index: np.ndarray
    anchor_iou: np.ndarray
    center_distance_m: np.ndarray
    center_distance_over_anchor_diagonal: np.ndarray
    volume_ratio: np.ndarray
    tr3d_score: np.ndarray
    anchor_score: np.ndarray
    point_count: np.ndarray
    point_density_m3: np.ndarray
    r2a_evidence_available: np.ndarray
    r2a_depth_evidence: np.ndarray
    r2a_depth_quality: np.ndarray
    r2a_view_count: np.ndarray
    r2a_point_count: np.ndarray
    r2b_feature_available: np.ndarray
    r2b_multiview_available: np.ndarray
    r2b_feature_view_count: np.ndarray
    r2b_pairwise_cosine_count: np.ndarray
    r2b_pairwise_cosine_mean: np.ndarray
    r2b_pairwise_cosine_median: np.ndarray
    r2b_pairwise_cosine_min: np.ndarray
    r2b_pairwise_cosine_max: np.ndarray
    r2b_pairwise_cosine_std: np.ndarray
    runtime_s: float = 0.0

    @property
    def sample_idx(self) -> str:
        return f"{self.scene_id}:{self.prefix_id}"

    @property
    def proposal_count(self) -> int:
        return int(np.asarray(self.proposal_ids).shape[0])

    def as_npz_payload(self) -> dict[str, np.ndarray]:
        payload: dict[str, np.ndarray] = {
            "schema": np.asarray(TR3D_R3_CACHE_SCHEMA),
            "complete": np.asarray(True, dtype=np.bool_),
            "observer_only": np.asarray(True, dtype=np.bool_),
            "mutation_enabled": np.asarray(False, dtype=np.bool_),
            "applied_count": np.asarray(0, dtype=np.int64),
            "scene_id": np.asarray(self.scene_id),
            "sample_idx": np.asarray(self.sample_idx),
            "prefix_id": np.asarray(self.prefix_id),
            "prefix_fraction": np.asarray(self.prefix_fraction, dtype=np.float64),
            "association_frame": np.asarray(TR3D_R3_ASSOCIATION_FRAME),
            "proposal_geometry_frame": np.asarray(TR3D_R3_GEOMETRY_FRAME),
            "axis_alignment": np.asarray(self.axis_alignment, dtype=np.float64),
            "near_anchor_iou": np.asarray(self.near_anchor_iou, dtype=np.float64),
            "anchor_count": np.asarray(self.anchor_count, dtype=np.int64),
            "parent_r2a_available": np.asarray(self.parent_r2a_available, dtype=np.bool_),
            "parent_r2b_available": np.asarray(self.parent_r2b_available, dtype=np.bool_),
            "runtime_s": np.asarray(self.runtime_s, dtype=np.float64),
        }
        payload.update({name: np.asarray(getattr(self, name)) for name in _PROVENANCE_FIELDS})
        payload.update(
            {
                name: np.asarray(getattr(self, name), dtype=dtype)
                for name, dtype in _ARRAY_DTYPES.items()
            }
        )
        return payload


def sha256_file(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tr3d_r3_cache_path(
    root: str | os.PathLike[str], scene_id: str, prefix_id: str = "full"
) -> Path:
    if _SCENE_RE.fullmatch(scene_id) is None:
        raise ValueError(f"invalid ScanNet scene id: {scene_id!r}")
    if _PREFIX_RE.fullmatch(prefix_id) is None:
        raise ValueError(f"invalid prefix_id: {prefix_id!r}")
    return Path(root) / scene_id / f"{prefix_id}.npz"


def _sha(value: str, name: str, *, optional: bool = False) -> str:
    result = str(value)
    if optional and result == "":
        return result
    if result != result.lower() or _SHA256_RE.fullmatch(result) is None:
        raise ValueError(f"{name} must be a lowercase SHA256")
    return result


def _scalar(values: Mapping[str, np.ndarray], name: str, dtype: np.dtype):
    value = np.asarray(values[name])
    if value.shape != () or value.dtype != np.dtype(dtype):
        raise ValueError(f"{name} must be a {np.dtype(dtype)} scalar")
    return value.item()


def _text(values: Mapping[str, np.ndarray], name: str) -> str:
    value = np.asarray(values[name])
    if value.shape != () or value.dtype.hasobject:
        raise ValueError(f"{name} must be a non-object string scalar")
    result = value.item()
    if isinstance(result, bytes):
        result = result.decode("utf-8")
    if not isinstance(result, str):
        raise ValueError(f"{name} must be a string scalar")
    return result


def _readonly(value: np.ndarray) -> np.ndarray:
    result = np.ascontiguousarray(value)
    result.setflags(write=False)
    return result


def _anchor_contract(
    *,
    manifest_path: Path,
    prediction_path: Path,
    scene_id: str,
) -> tuple[dict, str, str]:
    manifest = verify_frozen_anchor_manifest(manifest_path)
    relative = f"{scene_id}_boxes.pkl"
    expected = manifest["prediction_files"].get(relative)
    if expected is None:
        raise ValueError("R3 scene is absent from frozen anchor manifest")
    if prediction_path.resolve() != (
        Path(manifest["reference_result_root"]).resolve() / relative
    ):
        raise ValueError("R3 anchor prediction path is not the frozen path")
    observed = sha256_file(prediction_path)
    if observed != expected:
        raise ValueError("R3 frozen anchor prediction bytes changed")
    return manifest, sha256_file(manifest_path), observed


def _empty_parent_evidence(parent) -> dict[str, np.ndarray]:
    count = parent.proposal_count
    return {
        "lineage_ids": np.array(parent.proposal_ids, dtype=np.int64, copy=True),
        "r2a_evidence_available": np.zeros(count, dtype=np.bool_),
        "r2a_depth_evidence": np.zeros((count, DEPTH_EVIDENCE_DIM), dtype=np.float32),
        "r2a_view_count": np.zeros(count, dtype=np.int32),
        "r2a_point_count": np.zeros(count, dtype=np.int64),
        "r2b_feature_view_count": np.zeros(count, dtype=np.int32),
        "r2b_pairwise_cosine_count": np.zeros(count, dtype=np.int32),
        "r2b_pairwise_cosine_mean": np.zeros(count, dtype=np.float32),
        "r2b_pairwise_cosine_median": np.zeros(count, dtype=np.float32),
        "r2b_pairwise_cosine_min": np.zeros(count, dtype=np.float32),
        "r2b_pairwise_cosine_max": np.zeros(count, dtype=np.float32),
        "r2b_pairwise_cosine_std": np.zeros(count, dtype=np.float32),
    }


def _derive_observation(
    *,
    parent,
    r2a,
    r2b,
    anchor_corners_world: np.ndarray,
    anchor_scores: np.ndarray,
    axis_alignment: np.ndarray,
) -> TR3DR3NearObservation:
    evidence = _empty_parent_evidence(parent)
    parent_index = {int(value): index for index, value in enumerate(parent.proposal_ids)}
    if r2a is not None:
        rows = np.asarray([parent_index[int(value)] for value in r2a.proposal_ids], dtype=np.int64)
        evidence["r2a_evidence_available"][rows] = True
        evidence["r2a_depth_evidence"][rows] = r2a.aggregate_depth_evidence
        evidence["r2a_view_count"][rows] = r2a.aggregate_view_count
        evidence["r2a_point_count"][rows] = r2a.aggregate_point_count
    if r2b is not None:
        rows = np.asarray([parent_index[int(value)] for value in r2b.proposal_ids], dtype=np.int64)
        evidence["r2b_feature_view_count"][rows] = r2b.aggregate_view_count
        evidence["r2b_pairwise_cosine_count"][rows] = r2b.pairwise_cosine_count
        for name in ("mean", "median", "min", "max", "std"):
            evidence[f"r2b_pairwise_cosine_{name}"][rows] = getattr(
                r2b, f"pairwise_cosine_{name}"
            )
    return observe_anchor_near_candidates(
        proposal_ids=parent.proposal_ids,
        proposal_corners_world=parent.corners_world,
        tr3d_score=parent.scores_3d,
        point_count=parent.point_count,
        anchor_corners_world=anchor_corners_world,
        anchor_score=anchor_scores,
        axis_alignment=axis_alignment,
        **evidence,
    )


def _verified_parents(
    *,
    parent_tr3d_cache_path: Path,
    parent_r2a_cache_path: Path | None,
    parent_r2b_cache_path: Path | None,
    expected_scene_id: str,
    expected_prefix_id: str,
    expected_checkpoint_sha256: str,
    expected_config_sha256: str,
    expected_prefix_manifest_row_sha256: str,
    expected_frame_artifact_tree_sha256: str,
    expected_r2_config_sha256: str,
    expected_r2_code_sha256: str,
    expected_feature_checkpoint_sha256: str,
    expected_feature_config_sha256: str,
    expected_feature_code_sha256: str,
):
    parent = load_tr3d_residual_cache(
        parent_tr3d_cache_path,
        expected_scene_id=expected_scene_id,
        expected_prefix_id=expected_prefix_id,
        expected_checkpoint_sha256=expected_checkpoint_sha256,
        expected_config_sha256=expected_config_sha256,
    )
    r2a = r2b = None
    if parent_r2b_cache_path is not None and parent_r2a_cache_path is None:
        raise ValueError("R2b parent requires its exact R2a parent")
    if parent_r2a_cache_path is not None:
        required = {
            "prefix manifest row": expected_prefix_manifest_row_sha256,
            "frame artifact tree": expected_frame_artifact_tree_sha256,
            "R2 config": expected_r2_config_sha256,
            "R2 code": expected_r2_code_sha256,
        }
        if any(not value for value in required.values()):
            raise ValueError("present R2a parent requires complete external provenance")
        r2a = load_tr3d_r2_cache(
            parent_r2a_cache_path,
            parent_cache_path=parent_tr3d_cache_path,
            expected_prefix_manifest_row_sha256=expected_prefix_manifest_row_sha256,
            expected_frame_artifact_tree_sha256=expected_frame_artifact_tree_sha256,
            expected_r2_config_sha256=expected_r2_config_sha256,
            expected_r2_code_sha256=expected_r2_code_sha256,
            expected_scene_id=expected_scene_id,
            expected_prefix_id=expected_prefix_id,
            expected_prefix_fraction=parent.prefix_fraction,
        )
    if parent_r2b_cache_path is not None:
        required = {
            "feature checkpoint": expected_feature_checkpoint_sha256,
            "feature config": expected_feature_config_sha256,
            "feature code": expected_feature_code_sha256,
        }
        if any(not value for value in required.values()):
            raise ValueError("present R2b parent requires complete feature provenance")
        r2b = load_tr3d_r2b_cache(
            parent_r2b_cache_path,
            parent_r2a_cache_path=parent_r2a_cache_path,
            parent_tr3d_cache_path=parent_tr3d_cache_path,
            expected_parent_prefix_manifest_row_sha256=expected_prefix_manifest_row_sha256,
            expected_parent_frame_artifact_tree_sha256=expected_frame_artifact_tree_sha256,
            expected_parent_r2_config_sha256=expected_r2_config_sha256,
            expected_parent_r2_code_sha256=expected_r2_code_sha256,
            expected_feature_checkpoint_sha256=expected_feature_checkpoint_sha256,
            expected_feature_config_sha256=expected_feature_config_sha256,
            expected_feature_code_sha256=expected_feature_code_sha256,
            expected_scene_id=expected_scene_id,
            expected_prefix_id=expected_prefix_id,
            expected_prefix_fraction=parent.prefix_fraction,
        )
    return parent, r2a, r2b


def _validate_payload_arrays(values: Mapping[str, np.ndarray], count: int) -> dict[str, np.ndarray]:
    result: dict[str, np.ndarray] = {}
    for name, dtype in _ARRAY_DTYPES.items():
        value = np.asarray(values[name])
        shape = (count, 8, 3) if name == "proposal_corners_world" else (
            (count, DEPTH_EVIDENCE_DIM) if name == "r2a_depth_evidence" else (count,)
        )
        if value.dtype != dtype or value.shape != shape:
            raise ValueError(f"{name} must have dtype {dtype} and shape {shape}")
        result[name] = value
    return result


def validate_tr3d_r3_payload(
    values: Mapping[str, np.ndarray],
    *,
    parent_tr3d_cache_path: str | os.PathLike[str],
    frozen_anchor_manifest_path: str | os.PathLike[str],
    anchor_prediction_path: str | os.PathLike[str],
    anchor_corners_world: np.ndarray,
    anchor_scores: np.ndarray,
    axis_alignment_metadata_path: str | os.PathLike[str],
    expected_checkpoint_sha256: str,
    expected_config_sha256: str,
    expected_r3_config_sha256: str,
    expected_r3_code_sha256: str,
    parent_r2a_cache_path: str | os.PathLike[str] | None = None,
    parent_r2b_cache_path: str | os.PathLike[str] | None = None,
    expected_prefix_manifest_row_sha256: str = "",
    expected_frame_artifact_tree_sha256: str = "",
    expected_r2_config_sha256: str = "",
    expected_r2_code_sha256: str = "",
    expected_feature_checkpoint_sha256: str = "",
    expected_feature_config_sha256: str = "",
    expected_feature_code_sha256: str = "",
    expected_scene_id: str | None = None,
    expected_prefix_id: str | None = None,
) -> TR3DR3NearCache:
    if frozenset(values) != _FIELDS:
        raise ValueError(
            f"R3 cache fields disagree; missing={sorted(_FIELDS-frozenset(values))}, "
            f"unknown={sorted(frozenset(values)-_FIELDS)}"
        )
    if any(np.asarray(value).dtype.hasobject for value in values.values()):
        raise ValueError("R3 cache must not contain object arrays")
    if _text(values, "schema") != TR3D_R3_CACHE_SCHEMA:
        raise ValueError("unsupported R3 cache schema")
    if not bool(_scalar(values, "complete", np.bool_)):
        raise ValueError("R3 cache is incomplete")
    if not bool(_scalar(values, "observer_only", np.bool_)):
        raise ValueError("R3 cache is not observer-only")
    if bool(_scalar(values, "mutation_enabled", np.bool_)):
        raise ValueError("R3 cache enables mutation")
    if int(_scalar(values, "applied_count", np.int64)) != 0:
        raise ValueError("R3 observer applied_count must be zero")
    if _text(values, "association_frame") != TR3D_R3_ASSOCIATION_FRAME:
        raise ValueError("R3 association frame mismatch")
    if _text(values, "proposal_geometry_frame") != TR3D_R3_GEOMETRY_FRAME:
        raise ValueError("R3 proposal geometry frame mismatch")

    scene_id = _text(values, "scene_id")
    prefix_id = _text(values, "prefix_id")
    if _SCENE_RE.fullmatch(scene_id) is None or _PREFIX_RE.fullmatch(prefix_id) is None:
        raise ValueError("invalid R3 scene/prefix")
    if _text(values, "sample_idx") != f"{scene_id}:{prefix_id}":
        raise ValueError("R3 sample_idx mismatch")
    if expected_scene_id is not None and scene_id != expected_scene_id:
        raise ValueError("R3 scene id mismatch")
    if expected_prefix_id is not None and prefix_id != expected_prefix_id:
        raise ValueError("R3 prefix id mismatch")
    fraction = float(_scalar(values, "prefix_fraction", np.float64))
    near_iou = float(_scalar(values, "near_anchor_iou", np.float64))
    runtime_s = float(_scalar(values, "runtime_s", np.float64))
    anchor_count = int(_scalar(values, "anchor_count", np.int64))
    r2a_available = bool(_scalar(values, "parent_r2a_available", np.bool_))
    r2b_available = bool(_scalar(values, "parent_r2b_available", np.bool_))
    if not math.isfinite(fraction) or not 0 < fraction <= 1:
        raise ValueError("R3 prefix fraction must lie in (0,1]")
    if near_iou != TR3D_R3_NEAR_ANCHOR_IOU:
        raise ValueError("R3 near-anchor split must equal 0.15")
    if not math.isfinite(runtime_s) or runtime_s < 0 or anchor_count < 0:
        raise ValueError("invalid R3 runtime/anchor count")
    if r2b_available and not r2a_available:
        raise ValueError("R3 R2b availability requires R2a")
    matrix = np.asarray(values["axis_alignment"])
    if matrix.dtype != np.float64 or matrix.shape != (4, 4):
        raise ValueError("R3 axis_alignment must be float64 [4,4]")

    hashes = {name: _text(values, name) for name in _PROVENANCE_FIELDS}
    optional = {
        "parent_r2a_cache_sha256",
        "parent_r2b_cache_sha256",
        "parent_prefix_manifest_row_sha256",
        "parent_frame_artifact_tree_sha256",
        "parent_r2_config_sha256",
        "parent_r2_code_sha256",
        "parent_feature_checkpoint_sha256",
        "parent_feature_config_sha256",
        "parent_feature_code_sha256",
    }
    for name, value in hashes.items():
        _sha(value, name, optional=name in optional)
    expected_simple = {
        "parent_checkpoint_sha256": expected_checkpoint_sha256,
        "parent_config_sha256": expected_config_sha256,
        "r3_config_sha256": expected_r3_config_sha256,
        "r3_code_sha256": expected_r3_code_sha256,
    }
    for name, expected in expected_simple.items():
        if hashes[name] != _sha(expected, f"expected_{name}"):
            raise ValueError(f"R3 {name} provenance mismatch")

    parent_path = Path(parent_tr3d_cache_path)
    r2a_path = None if parent_r2a_cache_path is None else Path(parent_r2a_cache_path)
    r2b_path = None if parent_r2b_cache_path is None else Path(parent_r2b_cache_path)
    if r2a_available != (r2a_path is not None) or r2b_available != (r2b_path is not None):
        raise ValueError("R3 optional parent availability/path mismatch")
    parent_before = sha256_file(parent_path)
    r2a_before = sha256_file(r2a_path) if r2a_path else ""
    r2b_before = sha256_file(r2b_path) if r2b_path else ""
    parent, r2a, r2b = _verified_parents(
        parent_tr3d_cache_path=parent_path,
        parent_r2a_cache_path=r2a_path,
        parent_r2b_cache_path=r2b_path,
        expected_scene_id=scene_id,
        expected_prefix_id=prefix_id,
        expected_checkpoint_sha256=expected_checkpoint_sha256,
        expected_config_sha256=expected_config_sha256,
        expected_prefix_manifest_row_sha256=expected_prefix_manifest_row_sha256,
        expected_frame_artifact_tree_sha256=expected_frame_artifact_tree_sha256,
        expected_r2_config_sha256=expected_r2_config_sha256,
        expected_r2_code_sha256=expected_r2_code_sha256,
        expected_feature_checkpoint_sha256=expected_feature_checkpoint_sha256,
        expected_feature_config_sha256=expected_feature_config_sha256,
        expected_feature_code_sha256=expected_feature_code_sha256,
    )
    if parent_before != sha256_file(parent_path):
        raise RuntimeError("TR3D parent changed while R3 verified it")
    if r2a_path and r2a_before != sha256_file(r2a_path):
        raise RuntimeError("R2a parent changed while R3 verified it")
    if r2b_path and r2b_before != sha256_file(r2b_path):
        raise RuntimeError("R2b parent changed while R3 verified it")
    if fraction != parent.prefix_fraction:
        raise ValueError("R3 prefix fraction disagrees with TR3D parent")
    manifest, manifest_sha, prediction_sha = _anchor_contract(
        manifest_path=Path(frozen_anchor_manifest_path),
        prediction_path=Path(anchor_prediction_path),
        scene_id=scene_id,
    )
    metadata_path = Path(axis_alignment_metadata_path)
    if hashes["parent_tr3d_cache_sha256"] != parent_before:
        raise ValueError("R3 exact TR3D parent bytes mismatch")
    exact_provenance = {
        "parent_checkpoint_sha256": parent.checkpoint_sha256,
        "parent_config_sha256": parent.config_sha256,
        "parent_source_scene_sha256": parent.source_scene_sha256,
        "parent_axis_alignment_sha256": parent.axis_alignment_sha256,
        "parent_r2a_cache_sha256": r2a_before,
        "parent_r2b_cache_sha256": r2b_before,
        "parent_prefix_manifest_row_sha256": expected_prefix_manifest_row_sha256 if r2a_path else "",
        "parent_frame_artifact_tree_sha256": expected_frame_artifact_tree_sha256 if r2a_path else "",
        "parent_r2_config_sha256": expected_r2_config_sha256 if r2a_path else "",
        "parent_r2_code_sha256": expected_r2_code_sha256 if r2a_path else "",
        "parent_feature_checkpoint_sha256": expected_feature_checkpoint_sha256 if r2b_path else "",
        "parent_feature_config_sha256": expected_feature_config_sha256 if r2b_path else "",
        "parent_feature_code_sha256": expected_feature_code_sha256 if r2b_path else "",
        "frozen_anchor_manifest_sha256": manifest_sha,
        "frozen_anchor_prediction_sha256": prediction_sha,
        "frozen_anchor_prediction_tree_sha256": manifest["prediction_tree_sha256"],
        "axis_alignment_metadata_sha256": sha256_file(metadata_path),
    }
    for name, expected in exact_provenance.items():
        if hashes[name] != expected:
            raise ValueError(f"R3 {name} exact provenance mismatch")
    metadata_matrix = load_axis_alignment_input_metadata(metadata_path)
    if not np.array_equal(matrix, metadata_matrix):
        raise ValueError("R3 axis alignment differs from exact input metadata")
    identity = np.eye(4, dtype=np.float64)
    if not (
        np.allclose(matrix @ parent.aligned_to_unaligned, identity, rtol=0, atol=1e-10)
        and np.allclose(parent.aligned_to_unaligned @ matrix, identity, rtol=0, atol=1e-10)
    ):
        raise ValueError("R3 axis alignment disagrees with TR3D parent inverse")
    anchors = np.asarray(anchor_corners_world, dtype=np.float64)
    anchor_values = np.asarray(anchor_scores, dtype=np.float64)
    if anchors.shape != (anchor_count, 8, 3) or anchor_values.shape != (anchor_count,):
        raise ValueError("R3 anchor arrays disagree with anchor_count")
    pinned_corners, pinned_scores = load_frozen_anchor_prediction(
        anchor_prediction_path
    )
    if not np.array_equal(anchors, pinned_corners) or not np.array_equal(
        anchor_values, pinned_scores
    ):
        raise ValueError("R3 anchor arrays differ from manifest-pinned prediction")

    count = int(np.asarray(values["proposal_ids"]).shape[0])
    arrays = _validate_payload_arrays(values, count)
    derived = _derive_observation(
        parent=parent,
        r2a=r2a,
        r2b=r2b,
        anchor_corners_world=anchors,
        anchor_scores=anchor_values,
        axis_alignment=matrix,
    )
    for field in fields(TR3DR3NearObservation):
        name = field.name
        if not np.array_equal(arrays[name], getattr(derived, name)):
            raise ValueError(f"R3 {name} disagrees with authoritative parents")

    return TR3DR3NearCache(
        scene_id=scene_id,
        prefix_id=prefix_id,
        prefix_fraction=fraction,
        axis_alignment=_readonly(matrix),
        near_anchor_iou=near_iou,
        anchor_count=anchor_count,
        parent_r2a_available=r2a_available,
        parent_r2b_available=r2b_available,
        **hashes,
        **{name: _readonly(value) for name, value in arrays.items()},
        runtime_s=runtime_s,
    )


def make_tr3d_r3_cache(
    *,
    parent_tr3d_cache_path: str | os.PathLike[str],
    frozen_anchor_manifest_path: str | os.PathLike[str],
    anchor_prediction_path: str | os.PathLike[str],
    anchor_corners_world: np.ndarray,
    anchor_scores: np.ndarray,
    axis_alignment_metadata_path: str | os.PathLike[str],
    axis_alignment: np.ndarray,
    expected_checkpoint_sha256: str,
    expected_config_sha256: str,
    r3_config_sha256: str,
    r3_code_sha256: str,
    parent_r2a_cache_path: str | os.PathLike[str] | None = None,
    parent_r2b_cache_path: str | os.PathLike[str] | None = None,
    prefix_manifest_row_sha256: str = "",
    frame_artifact_tree_sha256: str = "",
    r2_config_sha256: str = "",
    r2_code_sha256: str = "",
    feature_checkpoint_sha256: str = "",
    feature_config_sha256: str = "",
    feature_code_sha256: str = "",
    runtime_s: float = 0.0,
) -> TR3DR3NearCache:
    parent_path = Path(parent_tr3d_cache_path)
    r2a_path = None if parent_r2a_cache_path is None else Path(parent_r2a_cache_path)
    r2b_path = None if parent_r2b_cache_path is None else Path(parent_r2b_cache_path)
    parent, r2a, r2b = _verified_parents(
        parent_tr3d_cache_path=parent_path,
        parent_r2a_cache_path=r2a_path,
        parent_r2b_cache_path=r2b_path,
        expected_scene_id=load_tr3d_residual_cache(parent_path).scene_id,
        expected_prefix_id=load_tr3d_residual_cache(parent_path).prefix_id,
        expected_checkpoint_sha256=expected_checkpoint_sha256,
        expected_config_sha256=expected_config_sha256,
        expected_prefix_manifest_row_sha256=prefix_manifest_row_sha256,
        expected_frame_artifact_tree_sha256=frame_artifact_tree_sha256,
        expected_r2_config_sha256=r2_config_sha256,
        expected_r2_code_sha256=r2_code_sha256,
        expected_feature_checkpoint_sha256=feature_checkpoint_sha256,
        expected_feature_config_sha256=feature_config_sha256,
        expected_feature_code_sha256=feature_code_sha256,
    )
    manifest, manifest_sha, prediction_sha = _anchor_contract(
        manifest_path=Path(frozen_anchor_manifest_path),
        prediction_path=Path(anchor_prediction_path),
        scene_id=parent.scene_id,
    )
    observation = _derive_observation(
        parent=parent,
        r2a=r2a,
        r2b=r2b,
        anchor_corners_world=np.asarray(anchor_corners_world),
        anchor_scores=np.asarray(anchor_scores),
        axis_alignment=np.asarray(axis_alignment),
    )
    hashes = {
        "parent_tr3d_cache_sha256": sha256_file(parent_path),
        "parent_checkpoint_sha256": parent.checkpoint_sha256,
        "parent_config_sha256": parent.config_sha256,
        "parent_source_scene_sha256": parent.source_scene_sha256,
        "parent_axis_alignment_sha256": parent.axis_alignment_sha256,
        "parent_r2a_cache_sha256": sha256_file(r2a_path) if r2a_path else "",
        "parent_r2b_cache_sha256": sha256_file(r2b_path) if r2b_path else "",
        "parent_prefix_manifest_row_sha256": prefix_manifest_row_sha256 if r2a_path else "",
        "parent_frame_artifact_tree_sha256": frame_artifact_tree_sha256 if r2a_path else "",
        "parent_r2_config_sha256": r2_config_sha256 if r2a_path else "",
        "parent_r2_code_sha256": r2_code_sha256 if r2a_path else "",
        "parent_feature_checkpoint_sha256": feature_checkpoint_sha256 if r2b_path else "",
        "parent_feature_config_sha256": feature_config_sha256 if r2b_path else "",
        "parent_feature_code_sha256": feature_code_sha256 if r2b_path else "",
        "frozen_anchor_manifest_sha256": manifest_sha,
        "frozen_anchor_prediction_sha256": prediction_sha,
        "frozen_anchor_prediction_tree_sha256": manifest["prediction_tree_sha256"],
        "axis_alignment_metadata_sha256": sha256_file(axis_alignment_metadata_path),
        "r3_config_sha256": _sha(r3_config_sha256, "r3_config_sha256"),
        "r3_code_sha256": _sha(r3_code_sha256, "r3_code_sha256"),
    }
    cache = TR3DR3NearCache(
        scene_id=parent.scene_id,
        prefix_id=parent.prefix_id,
        prefix_fraction=parent.prefix_fraction,
        axis_alignment=np.asarray(axis_alignment, dtype=np.float64),
        near_anchor_iou=TR3D_R3_NEAR_ANCHOR_IOU,
        anchor_count=len(anchor_scores),
        parent_r2a_available=r2a is not None,
        parent_r2b_available=r2b is not None,
        **hashes,
        **{field.name: getattr(observation, field.name) for field in fields(TR3DR3NearObservation)},
        runtime_s=float(runtime_s),
    )
    return validate_tr3d_r3_payload(
        cache.as_npz_payload(),
        parent_tr3d_cache_path=parent_path,
        frozen_anchor_manifest_path=frozen_anchor_manifest_path,
        anchor_prediction_path=anchor_prediction_path,
        anchor_corners_world=anchor_corners_world,
        anchor_scores=anchor_scores,
        axis_alignment_metadata_path=axis_alignment_metadata_path,
        expected_checkpoint_sha256=expected_checkpoint_sha256,
        expected_config_sha256=expected_config_sha256,
        expected_r3_config_sha256=r3_config_sha256,
        expected_r3_code_sha256=r3_code_sha256,
        parent_r2a_cache_path=r2a_path,
        parent_r2b_cache_path=r2b_path,
        expected_prefix_manifest_row_sha256=prefix_manifest_row_sha256,
        expected_frame_artifact_tree_sha256=frame_artifact_tree_sha256,
        expected_r2_config_sha256=r2_config_sha256,
        expected_r2_code_sha256=r2_code_sha256,
        expected_feature_checkpoint_sha256=feature_checkpoint_sha256,
        expected_feature_config_sha256=feature_config_sha256,
        expected_feature_code_sha256=feature_code_sha256,
    )


def load_tr3d_r3_cache(path: str | os.PathLike[str], **contract) -> TR3DR3NearCache:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(source)
    with np.load(source, allow_pickle=False) as archive:
        values = {name: np.array(archive[name], copy=True) for name in archive.files}
    return validate_tr3d_r3_payload(values, **contract)


def write_tr3d_r3_cache(
    path: str | os.PathLike[str],
    cache: TR3DR3NearCache,
    **contract,
) -> Path:
    """Atomically create an immutable R3 file; existing paths are refused."""

    canonical = validate_tr3d_r3_payload(cache.as_npz_payload(), **contract)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    buffer = BytesIO()
    np.savez_compressed(buffer, **canonical.as_npz_payload())
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{target.name}.",
            suffix=".tmp",
            dir=target.parent,
            delete=False,
        ) as handle:
            temporary_name = handle.name
            handle.write(buffer.getvalue())
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary_name, target)
        target.chmod(0o444)
    except FileExistsError as error:
        raise FileExistsError(f"immutable R3 cache exists: {target}") from error
    finally:
        if temporary_name is not None:
            try:
                os.unlink(temporary_name)
            except FileNotFoundError:
                pass
    return target
