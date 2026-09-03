"""Immutable R5 paired grouping-evidence sidecars."""

from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
from typing import Any

import numpy as np

from .tr3d_r5_spgroup_observer import METRIC_NAMES, R5SPGroupObservation


SCHEMA = "boxfusion.tr3d_r5_spgroup_observer.v1"


def write_r5_sidecar(
    path: Path,
    *,
    scene_id: str,
    proposal_ids: np.ndarray,
    anchor_indices: np.ndarray,
    observation: R5SPGroupObservation,
    metadata: dict[str, Any],
) -> None:
    pair_count = len(proposal_ids)
    if anchor_indices.shape != (pair_count,):
        raise ValueError("anchor_indices shape mismatch")
    if observation.metrics.shape != (pair_count, 2, len(METRIC_NAMES)):
        raise ValueError("R5 metrics shape mismatch")
    if observation.metric_valid.shape != observation.metrics.shape:
        raise ValueError("R5 metric validity shape mismatch")
    if observation.candidate_minus_anchor.shape != (pair_count, len(METRIC_NAMES)):
        raise ValueError("R5 metric delta shape mismatch")
    if not np.isfinite(observation.metrics).all() or not np.isfinite(observation.candidate_minus_anchor).all():
        raise ValueError("R5 evidence contains non-finite values")
    required = {
        "schema": SCHEMA, "scene_id": scene_id, "observer_only": True,
        "mutation_enabled": False, "applied_count": 0,
        "ground_truth_access": False, "clip_access": False,
    }
    for key, expected in required.items():
        if metadata.get(key) != expected:
            raise ValueError(f"invalid R5 metadata: {key}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: str | None = None
    try:
        descriptor, temporary = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
        with os.fdopen(descriptor, "wb") as handle:
            np.savez_compressed(
                handle, schema=np.asarray(SCHEMA), complete=np.asarray(True),
                scene_id=np.asarray(scene_id),
                proposal_ids=np.asarray(proposal_ids, dtype=np.int64),
                anchor_indices=np.asarray(anchor_indices, dtype=np.int64),
                metric_names=np.asarray(METRIC_NAMES),
                metrics=np.asarray(observation.metrics, dtype=np.float32),
                metric_valid=np.asarray(observation.metric_valid, dtype=np.bool_),
                candidate_minus_anchor=np.asarray(observation.candidate_minus_anchor, dtype=np.float32),
                metadata_json=np.asarray(json.dumps(metadata, sort_keys=True, separators=(",", ":"))),
            )
            handle.flush(); os.fsync(handle.fileno())
        os.link(temporary, path); path.chmod(0o444)
    except FileExistsError as error:
        raise FileExistsError(f"immutable R5 sidecar exists: {path}") from error
    finally:
        if temporary is not None:
            try: os.unlink(temporary)
            except FileNotFoundError: pass


def load_r5_sidecar(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as archive:
        if str(archive["schema"].item()) != SCHEMA or not bool(archive["complete"].item()):
            raise ValueError(f"{path}: incomplete or incompatible R5 sidecar")
        result = {key: np.array(archive[key], copy=True) for key in (
            "proposal_ids", "anchor_indices", "metric_names", "metrics",
            "metric_valid", "candidate_minus_anchor",
        )}
        result["scene_id"] = str(archive["scene_id"].item())
        result["metadata"] = json.loads(str(archive["metadata_json"].item()))
    if tuple(result["metric_names"].tolist()) != METRIC_NAMES:
        raise ValueError(f"{path}: metric schema mismatch")
    return result
