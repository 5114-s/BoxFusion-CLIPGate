"""Deterministic, score-preserving R3 shadow-active geometry replacement.

The module is deliberately independent from ScanNet ground truth and CLIP.
It consumes one already validated R3 near-candidate cache and one trusted
BoxFusion prediction payload.  For every represented anchor it selects the
highest TR3D score (proposal id ascending breaks exact ties), then replaces
that anchor's geometry only when the selected TR3D score is strictly greater
than the frozen anchor score.

Candidate corners are copied directly from ``proposal_corners_world``.  They
are already in the same unaligned-world coordinate frame as the BoxFusion
payload; applying ScanNet ``axisAlignment`` here would transform them twice
during evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import pickle
from typing import Any, Protocol

import numpy as np


R3_ACTIVE_CONFIG_SCHEMA = "boxfusion.tr3d_r3_shadow_active_config.v1"
R3_ACTIVE_SUMMARY_SCHEMA = "boxfusion.tr3d_r3_shadow_active_summary.v1"


class R3NearCacheLike(Protocol):
    anchor_count: int
    proposal_ids: np.ndarray
    proposal_corners_world: np.ndarray
    anchor_index: np.ndarray
    tr3d_score: np.ndarray
    anchor_score: np.ndarray


@dataclass(frozen=True)
class R3ActiveSelection:
    anchor_index: int
    proposal_row: int
    proposal_id: int
    tr3d_score: float
    anchor_score: float
    geometry_changed: bool

    def as_dict(self) -> dict[str, int | float | bool]:
        return {
            "anchor_index": self.anchor_index,
            "proposal_row": self.proposal_row,
            "proposal_id": self.proposal_id,
            "tr3d_score": self.tr3d_score,
            "anchor_score": self.anchor_score,
            "geometry_changed": self.geometry_changed,
        }


@dataclass(frozen=True)
class R3ActiveSummary:
    prediction_count: int
    candidate_count: int
    represented_anchor_count: int
    selections: tuple[R3ActiveSelection, ...]
    selected_count: int
    changed_count: int
    noop: bool

    def as_dict(self) -> dict[str, Any]:
        """Return stable JSON-ready names used by the materializer manifest."""

        return {
            "schema": R3_ACTIVE_SUMMARY_SCHEMA,
            "prediction_count": self.prediction_count,
            "candidate_count": self.candidate_count,
            "represented_anchor_count": self.represented_anchor_count,
            "selected_count": self.selected_count,
            "applied_count": self.selected_count,
            "changed_count": self.changed_count,
            "byte_changed_count": self.changed_count,
            "noop": self.noop,
            "selections": [selection.as_dict() for selection in self.selections],
        }


def active_config() -> dict[str, Any]:
    """Return the frozen, threshold-free shadow-active policy."""

    return {
        "schema": R3_ACTIVE_CONFIG_SCHEMA,
        "ground_truth_access": False,
        "clip_access": False,
        "clip_semantics_unchanged": True,
        "candidate_population": "immutable_r3_anchor_near_cache",
        "per_anchor_selection": (
            "tr3d_score_descending_then_proposal_id_ascending"
        ),
        "replacement_condition": "selected_tr3d_score_strictly_gt_anchor_score",
        "output_mutation": "geometry_only",
        "preserved_fields": ["label", "score", "order", "count", "container_types"],
        "candidate_geometry_frame": "scannet_unaligned_world",
        "axis_alignment_applied_by_materializer": False,
    }


def active_code_sha256() -> str:
    """Hash the exact implementation bytes used by a materialization run."""

    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def active_config_sha256() -> str:
    encoded = json.dumps(
        active_config(),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _container(value: object, name: str) -> list | tuple:
    if type(value) not in {list, tuple}:
        raise ValueError(f"{name} must be a built-in list or tuple")
    return value


def _prediction_rows(payload: object) -> tuple[list | tuple, list | tuple]:
    outer = _container(payload, "prediction payload")
    if len(outer) != 1:
        raise ValueError("prediction payload must contain exactly one batch row")
    rows = _container(outer[0], "prediction batch row")
    for index, detection in enumerate(rows):
        if type(detection) not in {list, tuple} or len(detection) != 3:
            raise ValueError(f"prediction row {index} must be a built-in length-3 sequence")
        label, corners, score = detection
        if isinstance(label, bool) or not isinstance(label, (int, np.integer)):
            raise ValueError(f"prediction row {index} label must be an integer")
        if not isinstance(corners, np.ndarray):
            raise ValueError(f"prediction row {index} geometry must be a numpy array")
        if corners.dtype != np.float32 or corners.shape != (8, 3):
            raise ValueError(
                f"prediction row {index} geometry must be float32 [8,3]"
            )
        if not corners.flags.c_contiguous or not np.isfinite(corners).all():
            raise ValueError(
                f"prediction row {index} geometry must be finite and C-contiguous"
            )
        if type(score) is not float or not math.isfinite(score):
            raise ValueError(f"prediction row {index} score must be a finite Python float")
        if score < 0.0 or score > 1.0:
            raise ValueError(f"prediction row {index} score must lie in [0,1]")
    return outer, rows


def _cache_arrays(
    cache: R3NearCacheLike, prediction_count: int
) -> dict[str, np.ndarray]:
    if isinstance(cache.anchor_count, bool) or not isinstance(
        cache.anchor_count, (int, np.integer)
    ):
        raise ValueError("R3 cache anchor_count must be an integer")
    if int(cache.anchor_count) != prediction_count:
        raise ValueError("R3 cache anchor_count disagrees with prediction count")
    proposal_ids = np.asarray(cache.proposal_ids)
    if proposal_ids.dtype != np.int64 or proposal_ids.ndim != 1:
        raise ValueError("R3 proposal_ids must be int64 [N]")
    count = len(proposal_ids)
    corners = np.asarray(cache.proposal_corners_world)
    anchors = np.asarray(cache.anchor_index)
    scores = np.asarray(cache.tr3d_score)
    cached_anchor_scores = np.asarray(cache.anchor_score)
    if corners.dtype != np.float32 or corners.shape != (count, 8, 3):
        raise ValueError("R3 proposal_corners_world must be float32 [N,8,3]")
    if anchors.dtype != np.int64 or anchors.shape != (count,):
        raise ValueError("R3 anchor_index must be int64 [N]")
    if scores.dtype != np.float32 or scores.shape != (count,):
        raise ValueError("R3 tr3d_score must be float32 [N]")
    if cached_anchor_scores.dtype != np.float32 or cached_anchor_scores.shape != (
        count,
    ):
        raise ValueError("R3 anchor_score must be float32 [N]")
    if len(np.unique(proposal_ids)) != count:
        raise ValueError("R3 proposal_ids must be unique")
    if count and (np.any(anchors < 0) or np.any(anchors >= prediction_count)):
        raise ValueError("R3 anchor_index is out of prediction range")
    if not np.isfinite(corners).all():
        raise ValueError("R3 proposal geometry must be finite")
    extents = corners.max(axis=1) - corners.min(axis=1)
    if count and np.any(extents <= 0.0):
        raise ValueError("R3 proposal geometry must have positive extent")
    if (
        not np.isfinite(scores).all()
        or np.any(scores < 0.0)
        or np.any(scores > 1.0)
    ):
        raise ValueError("R3 TR3D scores must be finite in [0,1]")
    if (
        not np.isfinite(cached_anchor_scores).all()
        or np.any(cached_anchor_scores < 0.0)
        or np.any(cached_anchor_scores > 1.0)
    ):
        raise ValueError("R3 cached anchor scores must be finite in [0,1]")
    return {
        "proposal_ids": proposal_ids,
        "corners": corners,
        "anchor_index": anchors,
        "tr3d_score": scores,
        "anchor_score": cached_anchor_scores,
    }


def _rebuild_container(source: list | tuple, values: list[object]) -> list | tuple:
    return values if type(source) is list else tuple(values)


def _geometry_bytes_equal(left: np.ndarray, right: np.ndarray) -> bool:
    return bool(
        left.dtype == right.dtype
        and left.shape == right.shape
        and left.strides == right.strides
        and left.flags.c_contiguous == right.flags.c_contiguous
        and left.flags.f_contiguous == right.flags.f_contiguous
        and left.tobytes(order="A") == right.tobytes(order="A")
    )


def _select_rows(
    *,
    proposal_ids: np.ndarray,
    anchor_index: np.ndarray,
    tr3d_score: np.ndarray,
    prediction_scores: np.ndarray,
) -> tuple[int, ...]:
    selected: list[int] = []
    for anchor in np.unique(anchor_index):
        rows = np.flatnonzero(anchor_index == anchor)
        order = np.lexsort((proposal_ids[rows], -tr3d_score[rows]))
        row = int(rows[int(order[0])])
        if float(tr3d_score[row]) > float(prediction_scores[int(anchor)]):
            selected.append(row)
    return tuple(selected)


def primary_candidate_rows(
    source_payload: object,
    r3_cache: R3NearCacheLike,
) -> tuple[int, ...]:
    """Return the exact frozen-primary candidate rows without mutating output.

    This public, dependency-light adapter is shared by the train-only risk
    calibrator and the active materializer.  Keeping selection in one place is
    important: the authoritative rule compares the float32 TR3D score against
    the *actual Python-float* prediction score, not against the rounded score
    stored in the R3 sidecar.
    """

    _, source_rows = _prediction_rows(source_payload)
    arrays = _cache_arrays(r3_cache, len(source_rows))
    prediction_scores = np.asarray(
        [float(detection[2]) for detection in source_rows], dtype=np.float64
    )
    expected_cached_scores = prediction_scores[arrays["anchor_index"]].astype(
        np.float32
    )
    if not np.array_equal(arrays["anchor_score"], expected_cached_scores):
        raise ValueError("R3 cached anchor scores disagree with source prediction")
    return _select_rows(
        proposal_ids=arrays["proposal_ids"],
        anchor_index=arrays["anchor_index"],
        tr3d_score=arrays["tr3d_score"],
        prediction_scores=prediction_scores,
    )


def materialize_shadow_active_prediction(
    source_payload: object,
    r3_cache: R3NearCacheLike,
) -> tuple[list | tuple, R3ActiveSummary]:
    """Return a geometry-only shadow prediction and deterministic summary."""

    outer, source_rows = _prediction_rows(source_payload)
    arrays = _cache_arrays(r3_cache, len(source_rows))
    prediction_scores = np.asarray(
        [float(detection[2]) for detection in source_rows], dtype=np.float64
    )
    expected_cached_scores = prediction_scores[arrays["anchor_index"]].astype(
        np.float32
    )
    if not np.array_equal(arrays["anchor_score"], expected_cached_scores):
        raise ValueError("R3 cached anchor scores disagree with source prediction")

    selected = primary_candidate_rows(source_payload, r3_cache)
    output_rows = list(source_rows)
    records: list[R3ActiveSelection] = []
    for proposal_row in selected:
        anchor = int(arrays["anchor_index"][proposal_row])
        source_detection = source_rows[anchor]
        candidate = np.array(
            arrays["corners"][proposal_row], dtype=np.float32, order="C", copy=True
        )
        changed = not _geometry_bytes_equal(source_detection[1], candidate)
        replacement = [source_detection[0], candidate, source_detection[2]]
        output_rows[anchor] = _rebuild_container(source_detection, replacement)
        records.append(
            R3ActiveSelection(
                anchor_index=anchor,
                proposal_row=proposal_row,
                proposal_id=int(arrays["proposal_ids"][proposal_row]),
                tr3d_score=float(arrays["tr3d_score"][proposal_row]),
                anchor_score=float(prediction_scores[anchor]),
                geometry_changed=changed,
            )
        )

    output_inner = _rebuild_container(source_rows, output_rows)
    output = _rebuild_container(outer, [output_inner])
    changed_count = sum(record.geometry_changed for record in records)
    summary = R3ActiveSummary(
        prediction_count=len(source_rows),
        candidate_count=len(arrays["proposal_ids"]),
        represented_anchor_count=len(np.unique(arrays["anchor_index"])),
        selections=tuple(records),
        selected_count=len(records),
        changed_count=changed_count,
        noop=changed_count == 0,
    )
    return output, summary


def validate_shadow_active_prediction(
    source_payload: object,
    output_payload: object,
    r3_cache: R3NearCacheLike,
) -> R3ActiveSummary:
    """Recompute and strictly validate a materialized shadow prediction."""

    expected, summary = materialize_shadow_active_prediction(
        source_payload, r3_cache
    )
    expected_outer, expected_rows = _prediction_rows(expected)
    observed_outer, observed_rows = _prediction_rows(output_payload)
    if type(expected_outer) is not type(observed_outer) or type(
        expected_rows
    ) is not type(observed_rows):
        raise ValueError("shadow output changed prediction container types")
    if len(expected_rows) != len(observed_rows):
        raise ValueError("shadow output changed prediction count")
    for index, (wanted, observed) in enumerate(
        zip(expected_rows, observed_rows)
    ):
        if type(wanted) is not type(observed):
            raise ValueError(f"shadow output changed detection type at row {index}")
        if type(wanted[0]) is not type(observed[0]) or wanted[0] != observed[0]:
            raise ValueError(f"shadow output changed label at row {index}")
        if type(wanted[2]) is not type(observed[2]) or wanted[2] != observed[2]:
            raise ValueError(f"shadow output changed score at row {index}")
        if not _geometry_bytes_equal(wanted[1], observed[1]):
            raise ValueError(f"shadow output geometry mismatch at row {index}")
    return summary


def load_prediction_payload(path: str | Path) -> list | tuple:
    """Load and validate a trusted local BoxFusion prediction pickle."""

    source = Path(path)
    with source.open("rb") as handle:
        payload = pickle.load(handle)  # noqa: S301 - manifest-pinned local artifact
    outer, _ = _prediction_rows(payload)
    return outer


__all__ = [
    "R3_ACTIVE_CONFIG_SCHEMA",
    "R3_ACTIVE_SUMMARY_SCHEMA",
    "R3ActiveSelection",
    "R3ActiveSummary",
    "active_code_sha256",
    "active_config",
    "active_config_sha256",
    "load_prediction_payload",
    "materialize_shadow_active_prediction",
    "validate_shadow_active_prediction",
]
