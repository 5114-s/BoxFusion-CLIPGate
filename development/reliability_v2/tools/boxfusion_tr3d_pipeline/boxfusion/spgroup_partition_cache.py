"""Immutable ScanNet mesh-superpoint caches for the R5 observer.

This module deliberately contains no ground-truth, class-label, CLIP, or
prediction code.  It reproduces the partitioning input used by SPGroup3D and
stores enough provenance to reject stale or mixed-scene artifacts.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any

import numpy as np


SCHEMA = "boxfusion.spgroup3d_mesh_partition.v1"


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def read_axis_alignment(path: Path) -> np.ndarray:
    values: np.ndarray | None = None
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip().startswith("axisAlignment"):
            values = np.fromstring(line.split("=", 1)[1], sep=" ", dtype=np.float64)
            break
    if values is None or values.shape != (16,) or not np.isfinite(values).all():
        raise ValueError(f"{path}: missing or invalid axisAlignment")
    result = values.reshape(4, 4)
    if not np.allclose(result[3], (0.0, 0.0, 0.0, 1.0), atol=1e-7):
        raise ValueError(f"{path}: invalid homogeneous axisAlignment")
    return result


@dataclass(frozen=True)
class SPGroupPartition:
    scene_id: str
    vertices_unaligned: np.ndarray
    vertices_aligned: np.ndarray
    colors: np.ndarray
    faces: np.ndarray
    superpoint_ids: np.ndarray
    axis_alignment: np.ndarray
    metadata: dict[str, Any]

    @property
    def vertex_count(self) -> int:
        return int(self.vertices_unaligned.shape[0])

    @property
    def superpoint_count(self) -> int:
        return int(np.unique(self.superpoint_ids).size)


def validate_partition(value: SPGroupPartition) -> None:
    n = value.vertex_count
    if value.vertices_unaligned.shape != (n, 3):
        raise ValueError("vertices_unaligned must be [N,3]")
    if value.vertices_aligned.shape != (n, 3):
        raise ValueError("vertices_aligned must be [N,3]")
    if value.colors.shape != (n, 3):
        raise ValueError("colors must be [N,3]")
    if value.faces.ndim != 2 or value.faces.shape[1] != 3:
        raise ValueError("faces must be [F,3]")
    if value.superpoint_ids.shape != (n,):
        raise ValueError("superpoint_ids must be [N]")
    if value.axis_alignment.shape != (4, 4):
        raise ValueError("axis_alignment must be [4,4]")
    arrays = (value.vertices_unaligned, value.vertices_aligned, value.colors, value.axis_alignment)
    if any(not np.isfinite(array).all() for array in arrays):
        raise ValueError("partition contains non-finite values")
    if n == 0 or value.faces.shape[0] == 0:
        raise ValueError("partition mesh is empty")
    if value.faces.min(initial=0) < 0 or value.faces.max(initial=-1) >= n:
        raise ValueError("face index is outside vertex array")
    if value.superpoint_ids.min(initial=0) < 0:
        raise ValueError("superpoint ids must be non-negative")
    expected = np.arange(value.superpoint_count, dtype=np.int64)
    if not np.array_equal(np.unique(value.superpoint_ids), expected):
        raise ValueError("superpoint ids must be contiguous from zero")
    aligned = (
        value.vertices_unaligned @ value.axis_alignment[:3, :3].T
        + value.axis_alignment[:3, 3]
    )
    if not np.allclose(aligned, value.vertices_aligned, atol=2e-5, rtol=0.0):
        raise ValueError("aligned vertices disagree with axisAlignment")
    if value.metadata.get("schema") != SCHEMA:
        raise ValueError("partition metadata schema mismatch")
    if value.metadata.get("scene_id") != value.scene_id:
        raise ValueError("partition scene identity mismatch")
    if bool(value.metadata.get("ground_truth_access", True)):
        raise ValueError("partition must attest ground_truth_access=false")


def write_partition(path: Path, value: SPGroupPartition) -> None:
    validate_partition(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: str | None = None
    try:
        descriptor, temporary = tempfile.mkstemp(
            dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
        )
        with os.fdopen(descriptor, "wb") as handle:
            np.savez_compressed(
                handle,
                schema=np.asarray(SCHEMA),
                complete=np.asarray(True),
                scene_id=np.asarray(value.scene_id),
                vertices_unaligned=np.asarray(value.vertices_unaligned, dtype=np.float32),
                vertices_aligned=np.asarray(value.vertices_aligned, dtype=np.float32),
                colors=np.asarray(value.colors, dtype=np.float32),
                faces=np.asarray(value.faces, dtype=np.int32),
                superpoint_ids=np.asarray(value.superpoint_ids, dtype=np.int32),
                axis_alignment=np.asarray(value.axis_alignment, dtype=np.float64),
                metadata_json=np.asarray(
                    json.dumps(value.metadata, sort_keys=True, separators=(",", ":"))
                ),
            )
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
        path.chmod(0o444)
    except FileExistsError as error:
        raise FileExistsError(f"immutable SPGroup3D partition exists: {path}") from error
    finally:
        if temporary is not None:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass


def load_partition(path: Path) -> SPGroupPartition:
    with np.load(path, allow_pickle=False) as archive:
        if str(archive["schema"].item()) != SCHEMA or not bool(archive["complete"].item()):
            raise ValueError(f"{path}: incomplete or incompatible partition")
        value = SPGroupPartition(
            scene_id=str(archive["scene_id"].item()),
            vertices_unaligned=np.asarray(archive["vertices_unaligned"], dtype=np.float32),
            vertices_aligned=np.asarray(archive["vertices_aligned"], dtype=np.float32),
            colors=np.asarray(archive["colors"], dtype=np.float32),
            faces=np.asarray(archive["faces"], dtype=np.int32),
            superpoint_ids=np.asarray(archive["superpoint_ids"], dtype=np.int32),
            axis_alignment=np.asarray(archive["axis_alignment"], dtype=np.float64),
            metadata=json.loads(str(archive["metadata_json"].item())),
        )
    validate_partition(value)
    return value
