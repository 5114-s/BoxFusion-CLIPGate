"""Immutable official SPGroup3D grouping-feature sidecars."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import tempfile
from typing import Any

import numpy as np

from .spgroup_official_adapter import SPGroupFeatures


SCHEMA = "boxfusion.spgroup3d_group_features.v1"


@dataclass(frozen=True)
class SPGroupFeatureSidecar:
    scene_id: str
    features: SPGroupFeatures
    metadata: dict[str, Any]


def validate_feature_sidecar(value: SPGroupFeatureSidecar) -> None:
    features = value.features
    groups = len(features.superpoint_ids)
    shapes = {
        "centers_aligned": (groups, 3),
        "embeddings": (groups, 390),
        "vote_offsets": (groups, 3),
        "vote_offset_std": (groups, 3),
        "voxel_counts": (groups,),
    }
    for name, shape in shapes.items():
        array = np.asarray(getattr(features, name))
        if array.shape != shape:
            raise ValueError(f"{name} must have shape {shape}, got {array.shape}")
        if not np.isfinite(array).all():
            raise ValueError(f"{name} contains non-finite values")
    if features.superpoint_ids.shape != (groups,):
        raise ValueError("superpoint_ids must be [G]")
    if groups and not np.all(features.superpoint_ids[1:] > features.superpoint_ids[:-1]):
        raise ValueError("superpoint_ids must be strictly increasing")
    if np.any(features.voxel_counts <= 0):
        raise ValueError("voxel_counts must be positive")
    if value.metadata.get("schema") != SCHEMA or value.metadata.get("scene_id") != value.scene_id:
        raise ValueError("feature metadata identity mismatch")
    for key in ("observer_only", "ground_truth_access", "clip_access", "semantic_head_used"):
        expected = key == "observer_only"
        if bool(value.metadata.get(key)) != expected:
            raise ValueError(f"invalid feature safety attestation: {key}")


def write_feature_sidecar(path: Path, value: SPGroupFeatureSidecar) -> None:
    validate_feature_sidecar(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: str | None = None
    try:
        descriptor, temporary = tempfile.mkstemp(
            dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
        )
        with os.fdopen(descriptor, "wb") as handle:
            np.savez_compressed(
                handle,
                schema=np.asarray(SCHEMA), complete=np.asarray(True),
                scene_id=np.asarray(value.scene_id),
                superpoint_ids=np.asarray(value.features.superpoint_ids, dtype=np.int32),
                centers_aligned=np.asarray(value.features.centers_aligned, dtype=np.float32),
                embeddings=np.asarray(value.features.embeddings, dtype=np.float32),
                vote_offsets=np.asarray(value.features.vote_offsets, dtype=np.float32),
                vote_offset_std=np.asarray(value.features.vote_offset_std, dtype=np.float32),
                voxel_counts=np.asarray(value.features.voxel_counts, dtype=np.int32),
                metadata_json=np.asarray(json.dumps(value.metadata, sort_keys=True, separators=(",", ":"))),
            )
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
        path.chmod(0o444)
    except FileExistsError as error:
        raise FileExistsError(f"immutable SPGroup3D feature sidecar exists: {path}") from error
    finally:
        if temporary is not None:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass


def load_feature_sidecar(path: Path) -> SPGroupFeatureSidecar:
    with np.load(path, allow_pickle=False) as archive:
        if str(archive["schema"].item()) != SCHEMA or not bool(archive["complete"].item()):
            raise ValueError(f"{path}: incomplete or incompatible feature sidecar")
        value = SPGroupFeatureSidecar(
            scene_id=str(archive["scene_id"].item()),
            features=SPGroupFeatures(
                superpoint_ids=np.asarray(archive["superpoint_ids"], dtype=np.int32),
                centers_aligned=np.asarray(archive["centers_aligned"], dtype=np.float32),
                embeddings=np.asarray(archive["embeddings"], dtype=np.float32),
                vote_offsets=np.asarray(archive["vote_offsets"], dtype=np.float32),
                vote_offset_std=np.asarray(archive["vote_offset_std"], dtype=np.float32),
                voxel_counts=np.asarray(archive["voxel_counts"], dtype=np.int32),
            ),
            metadata=json.loads(str(archive["metadata_json"].item())),
        )
    validate_feature_sidecar(value)
    return value
