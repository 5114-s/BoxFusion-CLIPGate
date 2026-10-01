"""Immutable R4-F paired DINO feature-consistency sidecars."""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
import math
import os
from pathlib import Path
import re
import tempfile
from typing import Mapping

import numpy as np

from .tr3d_r2b_observer import FEATURE_STAT_NAMES, feature_consistency_statistics
from .tr3d_r4_smov_feature import R4PairedFeatureObservation


TR3D_R4_FEATURE_SCHEMA = "boxfusion.tr3d_r4_smov_feature_observer.v1"
_SCENE_RE = re.compile(r"^scene[0-9]{4}_[0-9]{2}$")
_PREFIX_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
_SHA_RE = re.compile(r"^[0-9a-f]{64}$")
_FIELDS = frozenset(
    {
        "schema", "complete", "observer_only", "mutation_enabled",
        "applied_count", "ground_truth_access", "clip_access",
        "scene_id", "prefix_id", "r4_depth_sidecar_sha256",
        "frame_artifact_tree_sha256", "r4_feature_config_sha256",
        "r4_feature_code_sha256", "official_boxer_commit",
        "dino_checkpoint_sha256", "proposal_ids", "anchor_indices",
        "topk_frame_ids", "topk_view_valid", "feature_view_valid",
        "per_view_feature_count", "per_view_support_point_count",
        "per_view_features", "aggregate_feature_statistics",
        "aggregate_feature_view_count", "aggregate_feature_pair_count",
        "candidate_minus_anchor_statistics", "feature_runtime_s",
        "geometry_runtime_s", "total_runtime_s",
    }
)


def _sha(value: str, name: str) -> str:
    if not isinstance(value, str) or _SHA_RE.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA256")
    return value


def _readonly(value: object, dtype: np.dtype) -> np.ndarray:
    result = np.ascontiguousarray(value, dtype=dtype)
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class R4FeatureSidecar:
    scene_id: str
    prefix_id: str
    r4_depth_sidecar_sha256: str
    frame_artifact_tree_sha256: str
    r4_feature_config_sha256: str
    r4_feature_code_sha256: str
    official_boxer_commit: str
    dino_checkpoint_sha256: str
    proposal_ids: np.ndarray
    anchor_indices: np.ndarray
    topk_frame_ids: np.ndarray
    topk_view_valid: np.ndarray
    feature_view_valid: np.ndarray
    per_view_feature_count: np.ndarray
    per_view_support_point_count: np.ndarray
    per_view_features: np.ndarray
    aggregate_feature_statistics: np.ndarray
    aggregate_feature_view_count: np.ndarray
    aggregate_feature_pair_count: np.ndarray
    candidate_minus_anchor_statistics: np.ndarray
    feature_runtime_s: float
    geometry_runtime_s: float
    total_runtime_s: float

    @property
    def pair_count(self) -> int:
        return int(self.proposal_ids.shape[0])

    def as_npz_payload(self) -> dict[str, np.ndarray]:
        result = {
            "schema": np.asarray(TR3D_R4_FEATURE_SCHEMA),
            "complete": np.asarray(True, dtype=np.bool_),
            "observer_only": np.asarray(True, dtype=np.bool_),
            "mutation_enabled": np.asarray(False, dtype=np.bool_),
            "applied_count": np.asarray(0, dtype=np.int64),
            "ground_truth_access": np.asarray(False, dtype=np.bool_),
            "clip_access": np.asarray(False, dtype=np.bool_),
            "scene_id": np.asarray(self.scene_id),
            "prefix_id": np.asarray(self.prefix_id),
            "r4_depth_sidecar_sha256": np.asarray(self.r4_depth_sidecar_sha256),
            "frame_artifact_tree_sha256": np.asarray(self.frame_artifact_tree_sha256),
            "r4_feature_config_sha256": np.asarray(self.r4_feature_config_sha256),
            "r4_feature_code_sha256": np.asarray(self.r4_feature_code_sha256),
            "official_boxer_commit": np.asarray(self.official_boxer_commit),
            "dino_checkpoint_sha256": np.asarray(self.dino_checkpoint_sha256),
        }
        array_dtypes = {
            "proposal_ids": np.int64,
            "anchor_indices": np.int64,
            "topk_frame_ids": np.int64,
            "topk_view_valid": np.bool_,
            "feature_view_valid": np.bool_,
            "per_view_feature_count": np.int32,
            "per_view_support_point_count": np.int32,
            "per_view_features": np.float32,
            "aggregate_feature_statistics": np.float32,
            "aggregate_feature_view_count": np.int32,
            "aggregate_feature_pair_count": np.int32,
            "candidate_minus_anchor_statistics": np.float32,
        }
        result.update(
            {
                name: np.asarray(getattr(self, name), dtype=dtype)
                for name, dtype in array_dtypes.items()
            }
        )
        result.update(
            {
                name: np.asarray(getattr(self, name), dtype=np.float64)
                for name in (
                    "feature_runtime_s", "geometry_runtime_s", "total_runtime_s"
                )
            }
        )
        return result


def make_r4_feature_sidecar(
    *, observation: R4PairedFeatureObservation, prefix_id: str,
    r4_depth_sidecar_sha256: str, frame_artifact_tree_sha256: str,
    r4_feature_config_sha256: str, r4_feature_code_sha256: str,
    official_boxer_commit: str, dino_checkpoint_sha256: str,
) -> R4FeatureSidecar:
    if not isinstance(official_boxer_commit, str) or not re.fullmatch(
        r"[0-9a-f]{40}", official_boxer_commit
    ):
        raise ValueError("official_boxer_commit must be a lowercase git SHA")
    value = R4FeatureSidecar(
        scene_id=observation.scene_id,
        prefix_id=prefix_id,
        r4_depth_sidecar_sha256=_sha(r4_depth_sidecar_sha256, "r4_depth_sidecar_sha256"),
        frame_artifact_tree_sha256=_sha(frame_artifact_tree_sha256, "frame_artifact_tree_sha256"),
        r4_feature_config_sha256=_sha(r4_feature_config_sha256, "r4_feature_config_sha256"),
        r4_feature_code_sha256=_sha(r4_feature_code_sha256, "r4_feature_code_sha256"),
        official_boxer_commit=official_boxer_commit,
        dino_checkpoint_sha256=_sha(dino_checkpoint_sha256, "dino_checkpoint_sha256"),
        proposal_ids=_readonly(observation.proposal_ids, np.int64),
        anchor_indices=_readonly(observation.anchor_indices, np.int64),
        topk_frame_ids=_readonly(observation.topk_frame_ids, np.int64),
        topk_view_valid=_readonly(observation.topk_view_valid, np.bool_),
        feature_view_valid=_readonly(observation.feature_view_valid, np.bool_),
        per_view_feature_count=_readonly(observation.per_view_feature_count, np.int32),
        per_view_support_point_count=_readonly(observation.per_view_support_point_count, np.int32),
        per_view_features=_readonly(observation.per_view_features, np.float32),
        aggregate_feature_statistics=_readonly(observation.aggregate_feature_statistics, np.float32),
        aggregate_feature_view_count=_readonly(observation.aggregate_feature_view_count, np.int32),
        aggregate_feature_pair_count=_readonly(observation.aggregate_feature_pair_count, np.int32),
        candidate_minus_anchor_statistics=_readonly(observation.candidate_minus_anchor_statistics, np.float32),
        feature_runtime_s=float(observation.feature_runtime_s),
        geometry_runtime_s=float(observation.geometry_runtime_s),
        total_runtime_s=float(observation.total_runtime_s),
    )
    _validate(value)
    return value


def _validate(value: R4FeatureSidecar) -> None:
    if _SCENE_RE.fullmatch(value.scene_id) is None or _PREFIX_RE.fullmatch(value.prefix_id) is None:
        raise ValueError("invalid scene/prefix identity")
    if not re.fullmatch(r"[0-9a-f]{40}", value.official_boxer_commit):
        raise ValueError("invalid Boxer commit")
    for number in (value.feature_runtime_s, value.geometry_runtime_s, value.total_runtime_s):
        if not math.isfinite(number) or number < 0.0:
            raise ValueError("feature runtimes must be finite and nonnegative")
    count = value.pair_count
    frames = value.topk_frame_ids
    valid = value.topk_view_valid
    if frames.ndim != 2 or frames.shape[0] != count or valid.shape != frames.shape:
        raise ValueError("invalid Top-K shape")
    topk = frames.shape[1]
    dim = value.per_view_features.shape[-1]
    shapes = {
        "anchor_indices": (count,), "feature_view_valid": (count, topk, 2),
        "per_view_feature_count": (count, topk, 2),
        "per_view_support_point_count": (count, topk, 2),
        "per_view_features": (count, topk, 2, dim),
        "aggregate_feature_statistics": (count, 2, len(FEATURE_STAT_NAMES)),
        "aggregate_feature_view_count": (count, 2),
        "aggregate_feature_pair_count": (count, 2),
        "candidate_minus_anchor_statistics": (count, len(FEATURE_STAT_NAMES)),
    }
    for name, shape in shapes.items():
        if np.asarray(getattr(value, name)).shape != shape:
            raise ValueError(f"{name} must have shape {shape}")
    if dim < 1 and np.any(value.feature_view_valid):
        raise ValueError("valid feature views require a positive feature dimension")
    if np.any(value.feature_view_valid & ~valid[:, :, None]):
        raise ValueError("feature validity exceeds depth Top-K validity")
    if np.any(value.per_view_feature_count < 0) or np.any(value.per_view_support_point_count < 0):
        raise ValueError("feature/support counts must be nonnegative")
    if np.any(value.per_view_features[~value.feature_view_valid] != 0.0):
        raise ValueError("invalid feature slots must be exact zeros")
    if np.any(value.per_view_feature_count[~value.feature_view_valid] < 0):
        raise ValueError("invalid feature counts")
    recomputed = np.zeros_like(value.aggregate_feature_statistics)
    recomputed_pairs = np.zeros_like(value.aggregate_feature_pair_count)
    for pair in range(count):
        for role in range(2):
            stats, pairs = feature_consistency_statistics(
                value.per_view_features[pair, :, role],
                value.feature_view_valid[pair, :, role],
            )
            recomputed[pair, role] = stats
            recomputed_pairs[pair, role] = pairs
    if not np.array_equal(recomputed, value.aggregate_feature_statistics):
        raise ValueError("feature statistics disagree with stored vectors")
    if not np.array_equal(recomputed_pairs, value.aggregate_feature_pair_count):
        raise ValueError("feature pair counts disagree")
    if not np.array_equal(
        value.aggregate_feature_view_count,
        value.feature_view_valid.sum(axis=1, dtype=np.int32),
    ):
        raise ValueError("feature view counts disagree")
    delta = (recomputed[:, 1] - recomputed[:, 0]).astype(np.float32)
    if not np.array_equal(delta, value.candidate_minus_anchor_statistics):
        raise ValueError("paired feature delta disagrees")


def write_r4_feature_sidecar(path: str | os.PathLike[str], value: R4FeatureSidecar) -> None:
    _validate(value)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    buffer = BytesIO()
    np.savez_compressed(buffer, **value.as_npz_payload())
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(mode="wb", dir=target.parent, prefix=f".{target.name}.", suffix=".tmp", delete=False) as handle:
            temporary = handle.name
            handle.write(buffer.getvalue())
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, target)
        target.chmod(0o444)
    except FileExistsError as error:
        raise FileExistsError(f"immutable R4-F sidecar exists: {target}") from error
    finally:
        if temporary is not None:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass


def load_r4_feature_sidecar(path: str | os.PathLike[str]) -> R4FeatureSidecar:
    source = Path(path)
    if not source.is_file() or source.is_symlink():
        raise FileNotFoundError(source)
    with np.load(BytesIO(source.read_bytes()), allow_pickle=False) as archive:
        values = {name: np.array(archive[name], copy=True) for name in archive.files}
    if frozenset(values) != _FIELDS or any(item.dtype.hasobject for item in values.values()):
        raise ValueError("R4-F sidecar fields disagree or contain objects")

    def scalar(name: str, dtype: np.dtype):
        item = values[name]
        if item.shape != () or item.dtype != np.dtype(dtype):
            raise ValueError(f"invalid scalar {name}")
        return item.item()

    def text(name: str) -> str:
        item = values[name]
        if item.shape != () or item.dtype.hasobject:
            raise ValueError(f"invalid text {name}")
        result = item.item()
        return result.decode() if isinstance(result, bytes) else str(result)

    if text("schema") != TR3D_R4_FEATURE_SCHEMA:
        raise ValueError("unsupported R4-F schema")
    for name, expected in {
        "complete": True, "observer_only": True, "mutation_enabled": False,
        "ground_truth_access": False, "clip_access": False,
    }.items():
        if bool(scalar(name, np.bool_)) is not expected:
            raise ValueError(f"R4-F safety contract failed: {name}")
    if int(scalar("applied_count", np.int64)) != 0:
        raise ValueError("R4-F applied_count must be zero")
    excluded = {
        "schema", "complete", "observer_only", "mutation_enabled", "applied_count",
        "ground_truth_access", "clip_access", "scene_id", "prefix_id",
        "r4_depth_sidecar_sha256", "frame_artifact_tree_sha256",
        "r4_feature_config_sha256", "r4_feature_code_sha256",
        "official_boxer_commit", "dino_checkpoint_sha256",
        "feature_runtime_s", "geometry_runtime_s", "total_runtime_s",
    }
    kwargs = {name: values[name] for name in _FIELDS - excluded}
    kwargs.update(
        scene_id=text("scene_id"), prefix_id=text("prefix_id"),
        r4_depth_sidecar_sha256=_sha(text("r4_depth_sidecar_sha256"), "r4_depth_sidecar_sha256"),
        frame_artifact_tree_sha256=_sha(text("frame_artifact_tree_sha256"), "frame_artifact_tree_sha256"),
        r4_feature_config_sha256=_sha(text("r4_feature_config_sha256"), "r4_feature_config_sha256"),
        r4_feature_code_sha256=_sha(text("r4_feature_code_sha256"), "r4_feature_code_sha256"),
        official_boxer_commit=text("official_boxer_commit"),
        dino_checkpoint_sha256=_sha(text("dino_checkpoint_sha256"), "dino_checkpoint_sha256"),
        feature_runtime_s=float(scalar("feature_runtime_s", np.float64)),
        geometry_runtime_s=float(scalar("geometry_runtime_s", np.float64)),
        total_runtime_s=float(scalar("total_runtime_s", np.float64)),
    )
    result = R4FeatureSidecar(**kwargs)
    _validate(result)
    return result


__all__ = [
    "R4FeatureSidecar", "TR3D_R4_FEATURE_SCHEMA", "load_r4_feature_sidecar",
    "make_r4_feature_sidecar", "write_r4_feature_sidecar",
]
