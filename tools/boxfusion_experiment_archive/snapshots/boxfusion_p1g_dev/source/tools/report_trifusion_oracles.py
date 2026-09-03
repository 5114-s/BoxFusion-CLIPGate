#!/usr/bin/env python3
"""Deterministic, offline oracle analysis for the BoxFusion tri-fusion route.

The report deliberately uses ScanNet ground truth to measure upper bounds.  It
is not an inference component, a deployable scoring rule, or evidence that a
model generalizes to unseen data.  In particular, oracle-selected geometry and
oracle score ordering must never be fed back into validation-time predictions.

Prediction pickle files are trusted local BoxFusion experiment artifacts with
the layout ``[[(label, corners[8,3], score), ...]]``.  Do not run this program
on untrusted pickle files.

Geometry artifacts use ``boxfusion.trifusion.geometry_candidates.v1`` with
``scene_id``, unique ``prediction_indices[K]``, matching
``original_corners[K,8,3]``, ragged ``candidate_offsets[K+1]``, and flattened
``candidate_corners[C,8,3]``, ``candidate_ids[C]``, and
``candidate_sources[C]``. Supplemental artifacts use
``boxfusion.trifusion.supplemental_candidates.v1`` with flattened candidate
corners, IDs, and sources. Both schemas may include Boolean ``candidate_valid``
and ``candidate_verified`` (strictly all true when omitted); supplemental
scores and labels are optional diagnostics. JSON additionally supports nested
``predictions[].candidates`` and top-level ``candidates`` records.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.analyze_fused_oracle import (  # noqa: E402
    DEFAULT_THRESHOLDS,
    center_size_to_minmax,
    corners_to_minmax,
    load_axis_alignment,
    load_scene_predictions,
    maximum_matches,
    pairwise_aabb_iou,
    ranked_metrics,
    read_scene_ids,
    score_scene,
    transform_corners,
)


REPORT_SCHEMA = "boxfusion.trifusion.oracle_report"
REPORT_FORMAT_VERSION = 1
GEOMETRY_CANDIDATE_SCHEMA = (
    "boxfusion.trifusion.geometry_candidates.v1"
)
SUPPLEMENTAL_CANDIDATE_SCHEMA = (
    "boxfusion.trifusion.supplemental_candidates.v1"
)
CORNER_FRAME = "world_pre_axis_alignment"
PREDICTION_SUFFIX = "_boxes.pkl"
GEOMETRY_SUFFIX = "_geometry_candidates"
SUPPLEMENTAL_SUFFIX = "_supplemental_candidates"
THRESHOLDS = tuple(float(value) for value in DEFAULT_THRESHOLDS)
PAPER_PLUS_10_TARGET_AP_PERCENT = {
    0.15: 47.46,
    0.25: 41.36,
    0.50: 23.41,
}

_BASELINE = "baseline"
_BASELINE_ORACLE_SCORE = "baseline_oracle_score"
_BEST_ALL = "best_box_oracle"
_BEST_VERIFIED = "best_box_verified_only_oracle"
_UNION_ALL = "proposal_union_oracle"
_UNION_VERIFIED = "proposal_union_verified_only_oracle"
_METHODS = (
    _BASELINE,
    _BASELINE_ORACLE_SCORE,
    _BEST_ALL,
    _BEST_VERIFIED,
    _UNION_ALL,
    _UNION_VERIFIED,
)


@dataclass(frozen=True)
class GeometryCandidates:
    """Validated ragged geometry alternatives for exported predictions."""

    scene_id: str
    prediction_indices: np.ndarray
    original_corners: np.ndarray
    candidate_offsets: np.ndarray
    candidate_corners: np.ndarray
    candidate_ids: tuple[str, ...]
    candidate_sources: tuple[str, ...]
    candidate_valid: np.ndarray
    candidate_verified: np.ndarray


@dataclass(frozen=True)
class SupplementalCandidates:
    """Validated candidate proposals which are not exported predictions."""

    scene_id: str
    candidate_corners: np.ndarray
    candidate_ids: tuple[str, ...]
    candidate_sources: tuple[str, ...]
    candidate_valid: np.ndarray
    candidate_verified: np.ndarray
    candidate_scores: np.ndarray | None
    candidate_labels: tuple[str, ...] | None


def _threshold_key(value: float) -> str:
    return f"{float(value):.2f}"


def _readonly(array: np.ndarray, *, dtype: Any | None = None) -> np.ndarray:
    result = np.array(array, dtype=dtype, copy=True)
    result.setflags(write=False)
    return result


def _load_artifact(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.suffix.lower() == ".npz":
        with np.load(path, allow_pickle=False) as payload:
            return {
                name: np.array(payload[name], copy=True)
                for name in payload.files
            }
    if path.suffix.lower() == ".json":
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            raise ValueError(f"{path}: invalid JSON") from error
        if not isinstance(payload, Mapping):
            raise ValueError(f"{path}: JSON artifact must be an object")
        return dict(payload)
    raise ValueError(f"{path}: candidate artifact must be .npz or .json")


def _scalar_text(
    value: Any,
    *,
    name: str,
    path: Path,
    nonempty: bool = True,
) -> str:
    array = np.asarray(value)
    if array.shape != () or array.dtype.hasobject:
        raise ValueError(f"{path}: {name} must be a scalar string")
    scalar = array.item()
    if isinstance(scalar, bytes):
        scalar = scalar.decode("utf-8")
    if not isinstance(scalar, str) or (nonempty and not scalar):
        qualifier = "non-empty " if nonempty else ""
        raise ValueError(f"{path}: {name} must be a {qualifier}string")
    return scalar


def _require_schema_scene(
    payload: Mapping[str, Any],
    *,
    schema: str,
    expected_scene_id: str,
    path: Path,
) -> None:
    missing = {"schema", "scene_id"} - set(payload)
    if missing:
        raise ValueError(f"{path}: missing fields {sorted(missing)}")
    actual_schema = _scalar_text(
        payload["schema"], name="schema", path=path
    )
    if actual_schema != schema:
        raise ValueError(
            f"{path}: unsupported schema {actual_schema!r}; "
            f"expected {schema!r}"
        )
    scene_id = _scalar_text(
        payload["scene_id"], name="scene_id", path=path
    )
    if scene_id != expected_scene_id:
        raise ValueError(
            f"{path}: scene {scene_id!r} does not match "
            f"{expected_scene_id!r}"
        )
    if "corner_frame" in payload:
        corner_frame = _scalar_text(
            payload["corner_frame"], name="corner_frame", path=path
        )
        if corner_frame != CORNER_FRAME:
            raise ValueError(
                f"{path}: corner_frame {corner_frame!r} does not match "
                f"{CORNER_FRAME!r}"
            )


def _integer_rows(
    value: Any,
    *,
    name: str,
    path: Path,
    rows: int | None = None,
    nonnegative: bool = False,
) -> np.ndarray:
    array = np.asarray(value)
    if array.shape == (0,) and rows in {None, 0}:
        return np.empty(0, dtype=np.int64)
    if (
        array.ndim != 1
        or array.dtype.kind not in "iu"
        or (rows is not None and array.shape != (rows,))
    ):
        shape = "[N]" if rows is None else f"[{rows}]"
        raise ValueError(f"{path}: {name} must be integer {shape}")
    result = np.asarray(array, dtype=np.int64)
    if nonnegative and np.any(result < 0):
        raise ValueError(f"{path}: {name} must be non-negative")
    return result


def _boolean_rows(
    value: Any,
    *,
    name: str,
    path: Path,
    rows: int,
) -> np.ndarray:
    array = np.asarray(value)
    if rows == 0 and array.shape == (0,):
        return np.empty(0, dtype=bool)
    if array.shape != (rows,) or array.dtype != np.bool_:
        raise ValueError(
            f"{path}: {name} must have Boolean shape [{rows}]"
        )
    return np.asarray(array, dtype=bool)


def _corners(
    value: Any,
    *,
    name: str,
    path: Path,
    rows: int | None = None,
) -> np.ndarray:
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{path}: {name} must be numeric corners") from error
    if array.size == 0 and array.shape == (0,):
        array = np.empty((0, 8, 3), dtype=np.float64)
    if (
        array.ndim != 3
        or array.shape[1:] != (8, 3)
        or (rows is not None and array.shape[0] != rows)
    ):
        expected = "[N,8,3]" if rows is None else f"[{rows},8,3]"
        raise ValueError(f"{path}: {name} must have shape {expected}")
    if not np.isfinite(array).all():
        raise ValueError(f"{path}: {name} contains non-finite corners")
    if len(array):
        extents = array.max(axis=1) - array.min(axis=1)
        if np.any(extents <= 0.0):
            raise ValueError(f"{path}: {name} contains degenerate corners")
    return array


def _string_rows(
    value: Any,
    *,
    name: str,
    path: Path,
    rows: int,
    allow_empty: bool = False,
) -> tuple[str, ...]:
    array = np.asarray(value)
    if array.shape != (rows,) or array.dtype.hasobject:
        raise ValueError(
            f"{path}: {name} must have non-object shape [{rows}]"
        )
    output: list[str] = []
    for scalar in array.tolist():
        if isinstance(scalar, bytes):
            scalar = scalar.decode("utf-8")
        if not isinstance(scalar, str) or (not allow_empty and not scalar):
            qualifier = "" if allow_empty else " non-empty"
            raise ValueError(
                f"{path}: {name} entries must be{qualifier} strings"
            )
        output.append(scalar)
    return tuple(output)


def _identifier_rows(
    value: Any,
    *,
    name: str,
    path: Path,
    rows: int,
) -> tuple[str, ...]:
    array = np.asarray(value)
    if array.shape != (rows,) or array.dtype.hasobject:
        raise ValueError(
            f"{path}: {name} must have non-object shape [{rows}]"
        )
    output: list[str] = []
    for scalar in array.tolist():
        if isinstance(scalar, bytes):
            scalar = scalar.decode("utf-8")
        if isinstance(scalar, (bool, np.bool_)) or not isinstance(
            scalar, (str, int, np.integer)
        ):
            raise ValueError(
                f"{path}: {name} entries must be strings or integers"
            )
        normalized = str(scalar)
        if not normalized:
            raise ValueError(f"{path}: {name} entries must be non-empty")
        output.append(normalized)
    if len(set(output)) != len(output):
        raise ValueError(f"{path}: {name} entries must be unique")
    return tuple(output)


def _flatten_geometry_json(
    payload: Mapping[str, Any], *, path: Path
) -> dict[str, Any]:
    if "predictions" not in payload:
        return dict(payload)
    flat_fields = {
        "prediction_indices",
        "original_corners",
        "candidate_offsets",
        "candidate_corners",
        "candidate_ids",
        "candidate_sources",
        "candidate_valid",
        "candidate_verified",
    }
    ambiguous = flat_fields.intersection(payload)
    if ambiguous:
        raise ValueError(
            f"{path}: nested predictions cannot be combined with flat "
            f"geometry fields {sorted(ambiguous)}"
        )
    rows = payload["predictions"]
    if not isinstance(rows, list):
        raise ValueError(f"{path}: predictions must be a JSON array")
    prediction_indices: list[Any] = []
    original_corners: list[Any] = []
    offsets = [0]
    candidate_corners: list[Any] = []
    candidate_ids: list[Any] = []
    candidate_sources: list[Any] = []
    candidate_valid: list[Any] = []
    candidate_verified: list[Any] = []
    for row_index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise ValueError(
                f"{path}: predictions[{row_index}] must be an object"
            )
        required = {"prediction_index", "original_corners", "candidates"}
        missing = required - set(row)
        if missing:
            raise ValueError(
                f"{path}: predictions[{row_index}] missing "
                f"{sorted(missing)}"
            )
        candidates = row["candidates"]
        if not isinstance(candidates, list):
            raise ValueError(
                f"{path}: predictions[{row_index}].candidates "
                "must be an array"
            )
        prediction_indices.append(row["prediction_index"])
        original_corners.append(row["original_corners"])
        for candidate_index, candidate in enumerate(candidates):
            if not isinstance(candidate, Mapping):
                raise ValueError(
                    f"{path}: candidate {row_index}:{candidate_index} "
                    "must be an object"
                )
            missing_candidate = {
                "candidate_id",
                "source",
                "corners",
            } - set(candidate)
            if missing_candidate:
                raise ValueError(
                    f"{path}: candidate {row_index}:{candidate_index} "
                    f"missing {sorted(missing_candidate)}"
                )
            candidate_ids.append(candidate["candidate_id"])
            candidate_sources.append(candidate["source"])
            candidate_corners.append(candidate["corners"])
            candidate_valid.append(candidate.get("valid", True))
            candidate_verified.append(candidate.get("verified", True))
        offsets.append(len(candidate_ids))
    flattened = dict(payload)
    flattened.pop("predictions")
    flattened.update(
        {
            "prediction_indices": prediction_indices,
            "original_corners": original_corners,
            "candidate_offsets": offsets,
            "candidate_corners": candidate_corners,
            "candidate_ids": candidate_ids,
            "candidate_sources": candidate_sources,
            "candidate_valid": candidate_valid,
            "candidate_verified": candidate_verified,
        }
    )
    return flattened


def load_geometry_candidates(
    path: str | Path,
    *,
    expected_scene_id: str,
) -> GeometryCandidates:
    """Load the strict ragged geometry-candidate NPZ/JSON contract."""

    artifact_path = Path(path)
    raw = _load_artifact(artifact_path)
    _require_schema_scene(
        raw,
        schema=GEOMETRY_CANDIDATE_SCHEMA,
        expected_scene_id=expected_scene_id,
        path=artifact_path,
    )
    payload = _flatten_geometry_json(raw, path=artifact_path)
    required = {
        "prediction_indices",
        "original_corners",
        "candidate_offsets",
        "candidate_corners",
        "candidate_ids",
        "candidate_sources",
    }
    missing = required - set(payload)
    if missing:
        raise ValueError(
            f"{artifact_path}: missing geometry fields {sorted(missing)}"
        )

    prediction_indices = _integer_rows(
        payload["prediction_indices"],
        name="prediction_indices",
        path=artifact_path,
        nonnegative=True,
    )
    if len(np.unique(prediction_indices)) != len(prediction_indices):
        raise ValueError(
            f"{artifact_path}: prediction_indices must be unique"
        )
    rows = len(prediction_indices)
    original_corners = _corners(
        payload["original_corners"],
        name="original_corners",
        path=artifact_path,
        rows=rows,
    )
    candidate_corners = _corners(
        payload["candidate_corners"],
        name="candidate_corners",
        path=artifact_path,
    )
    candidate_count = len(candidate_corners)
    offsets = _integer_rows(
        payload["candidate_offsets"],
        name="candidate_offsets",
        path=artifact_path,
        rows=rows + 1,
        nonnegative=True,
    )
    if (
        offsets[0] != 0
        or offsets[-1] != candidate_count
        or np.any(offsets[1:] < offsets[:-1])
    ):
        raise ValueError(
            f"{artifact_path}: candidate_offsets must start at zero, be "
            "non-decreasing, and end at the candidate count"
        )
    candidate_ids = _identifier_rows(
        payload["candidate_ids"],
        name="candidate_ids",
        path=artifact_path,
        rows=candidate_count,
    )
    candidate_sources = _string_rows(
        payload["candidate_sources"],
        name="candidate_sources",
        path=artifact_path,
        rows=candidate_count,
    )
    candidate_valid = (
        np.ones(candidate_count, dtype=bool)
        if "candidate_valid" not in payload
        else _boolean_rows(
            payload["candidate_valid"],
            name="candidate_valid",
            path=artifact_path,
            rows=candidate_count,
        )
    )
    candidate_verified = (
        np.ones(candidate_count, dtype=bool)
        if "candidate_verified" not in payload
        else _boolean_rows(
            payload["candidate_verified"],
            name="candidate_verified",
            path=artifact_path,
            rows=candidate_count,
        )
    )
    if np.any(candidate_verified & ~candidate_valid):
        raise ValueError(
            f"{artifact_path}: verified candidates must also be valid"
        )
    return GeometryCandidates(
        scene_id=expected_scene_id,
        prediction_indices=_readonly(prediction_indices, dtype=np.int64),
        original_corners=_readonly(original_corners, dtype=np.float64),
        candidate_offsets=_readonly(offsets, dtype=np.int64),
        candidate_corners=_readonly(candidate_corners, dtype=np.float64),
        candidate_ids=candidate_ids,
        candidate_sources=candidate_sources,
        candidate_valid=_readonly(candidate_valid, dtype=bool),
        candidate_verified=_readonly(candidate_verified, dtype=bool),
    )


def _flatten_supplemental_json(
    payload: Mapping[str, Any], *, path: Path
) -> dict[str, Any]:
    if "candidates" not in payload:
        return dict(payload)
    flat_fields = {
        "candidate_corners",
        "candidate_ids",
        "candidate_sources",
        "candidate_valid",
        "candidate_verified",
        "candidate_scores",
        "candidate_labels",
    }
    ambiguous = flat_fields.intersection(payload)
    if ambiguous:
        raise ValueError(
            f"{path}: nested candidates cannot be combined with flat "
            f"supplemental fields {sorted(ambiguous)}"
        )
    rows = payload["candidates"]
    if not isinstance(rows, list):
        raise ValueError(f"{path}: candidates must be a JSON array")
    corners: list[Any] = []
    ids: list[Any] = []
    sources: list[Any] = []
    valid: list[Any] = []
    verified: list[Any] = []
    scores: list[Any] = []
    labels: list[Any] = []
    score_presence: list[bool] = []
    label_presence: list[bool] = []
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise ValueError(f"{path}: candidates[{index}] must be an object")
        missing = {"candidate_id", "source", "corners"} - set(row)
        if missing:
            raise ValueError(
                f"{path}: candidates[{index}] missing {sorted(missing)}"
            )
        ids.append(row["candidate_id"])
        sources.append(row["source"])
        corners.append(row["corners"])
        valid.append(row.get("valid", True))
        verified.append(row.get("verified", True))
        score_presence.append("score" in row)
        label_presence.append("label" in row)
        if "score" in row:
            scores.append(row["score"])
        if "label" in row:
            labels.append(row["label"])
    if score_presence and any(score_presence) and not all(score_presence):
        raise ValueError(
            f"{path}: supplemental scores must be present for all or no "
            "candidates"
        )
    if label_presence and any(label_presence) and not all(label_presence):
        raise ValueError(
            f"{path}: supplemental labels must be present for all or no "
            "candidates"
        )
    flattened = dict(payload)
    flattened.pop("candidates")
    flattened.update(
        {
            "candidate_corners": corners,
            "candidate_ids": ids,
            "candidate_sources": sources,
            "candidate_valid": valid,
            "candidate_verified": verified,
        }
    )
    if score_presence and all(score_presence):
        flattened["candidate_scores"] = scores
    if label_presence and all(label_presence):
        flattened["candidate_labels"] = labels
    return flattened


def load_supplemental_candidates(
    path: str | Path,
    *,
    expected_scene_id: str,
) -> SupplementalCandidates:
    """Load class-agnostic supplemental proposals from strict NPZ/JSON."""

    artifact_path = Path(path)
    raw = _load_artifact(artifact_path)
    _require_schema_scene(
        raw,
        schema=SUPPLEMENTAL_CANDIDATE_SCHEMA,
        expected_scene_id=expected_scene_id,
        path=artifact_path,
    )
    payload = _flatten_supplemental_json(raw, path=artifact_path)
    required = {
        "candidate_corners",
        "candidate_ids",
        "candidate_sources",
    }
    missing = required - set(payload)
    if missing:
        raise ValueError(
            f"{artifact_path}: missing supplemental fields "
            f"{sorted(missing)}"
        )
    candidate_corners = _corners(
        payload["candidate_corners"],
        name="candidate_corners",
        path=artifact_path,
    )
    rows = len(candidate_corners)
    candidate_ids = _identifier_rows(
        payload["candidate_ids"],
        name="candidate_ids",
        path=artifact_path,
        rows=rows,
    )
    candidate_sources = _string_rows(
        payload["candidate_sources"],
        name="candidate_sources",
        path=artifact_path,
        rows=rows,
    )
    candidate_valid = (
        np.ones(rows, dtype=bool)
        if "candidate_valid" not in payload
        else _boolean_rows(
            payload["candidate_valid"],
            name="candidate_valid",
            path=artifact_path,
            rows=rows,
        )
    )
    candidate_verified = (
        np.ones(rows, dtype=bool)
        if "candidate_verified" not in payload
        else _boolean_rows(
            payload["candidate_verified"],
            name="candidate_verified",
            path=artifact_path,
            rows=rows,
        )
    )
    if np.any(candidate_verified & ~candidate_valid):
        raise ValueError(
            f"{artifact_path}: verified candidates must also be valid"
        )
    candidate_scores: np.ndarray | None = None
    if "candidate_scores" in payload:
        try:
            scores = np.asarray(
                payload["candidate_scores"], dtype=np.float64
            )
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"{artifact_path}: candidate_scores must be numeric"
            ) from error
        if scores.shape != (rows,) or not np.isfinite(scores).all():
            raise ValueError(
                f"{artifact_path}: candidate_scores must be finite "
                f"shape [{rows}]"
            )
        candidate_scores = _readonly(scores, dtype=np.float64)
    candidate_labels: tuple[str, ...] | None = None
    if "candidate_labels" in payload:
        candidate_labels = _string_rows(
            payload["candidate_labels"],
            name="candidate_labels",
            path=artifact_path,
            rows=rows,
            allow_empty=True,
        )
    return SupplementalCandidates(
        scene_id=expected_scene_id,
        candidate_corners=_readonly(candidate_corners, dtype=np.float64),
        candidate_ids=candidate_ids,
        candidate_sources=candidate_sources,
        candidate_valid=_readonly(candidate_valid, dtype=bool),
        candidate_verified=_readonly(candidate_verified, dtype=bool),
        candidate_scores=candidate_scores,
        candidate_labels=candidate_labels,
    )


def _empty_geometry(scene_id: str) -> GeometryCandidates:
    return GeometryCandidates(
        scene_id=scene_id,
        prediction_indices=_readonly(
            np.empty(0, dtype=np.int64)
        ),
        original_corners=_readonly(
            np.empty((0, 8, 3), dtype=np.float64)
        ),
        candidate_offsets=_readonly(np.asarray([0], dtype=np.int64)),
        candidate_corners=_readonly(
            np.empty((0, 8, 3), dtype=np.float64)
        ),
        candidate_ids=(),
        candidate_sources=(),
        candidate_valid=_readonly(np.empty(0, dtype=bool)),
        candidate_verified=_readonly(np.empty(0, dtype=bool)),
    )


def _empty_supplemental(scene_id: str) -> SupplementalCandidates:
    return SupplementalCandidates(
        scene_id=scene_id,
        candidate_corners=_readonly(
            np.empty((0, 8, 3), dtype=np.float64)
        ),
        candidate_ids=(),
        candidate_sources=(),
        candidate_valid=_readonly(np.empty(0, dtype=bool)),
        candidate_verified=_readonly(np.empty(0, dtype=bool)),
        candidate_scores=None,
        candidate_labels=None,
    )


def _resolve_scene_artifact(
    root: str | Path,
    *,
    scene_id: str,
    suffix: str,
) -> Path:
    artifact_root = Path(root)
    if artifact_root.is_file():
        return artifact_root
    if not artifact_root.is_dir():
        raise FileNotFoundError(artifact_root)
    candidates = [
        artifact_root / f"{scene_id}{suffix}.npz",
        artifact_root / f"{scene_id}{suffix}.json",
    ]
    existing = [path for path in candidates if path.is_file()]
    if not existing:
        raise FileNotFoundError(candidates[0])
    if len(existing) != 1:
        raise ValueError(
            f"{scene_id}: ambiguous candidate artifacts "
            f"{[str(path) for path in existing]}"
        )
    return existing[0]


def _aligned_minmax(
    corners: np.ndarray, transform: np.ndarray
) -> np.ndarray:
    return corners_to_minmax(transform_corners(corners, transform))


def _candidate_iou(
    candidates: GeometryCandidates,
    *,
    transform: np.ndarray,
    gt_minmax: np.ndarray,
) -> np.ndarray:
    aligned = _aligned_minmax(candidates.candidate_corners, transform)
    return pairwise_aabb_iou(aligned, gt_minmax)


def _geometry_lookup(
    candidates: GeometryCandidates,
) -> dict[int, tuple[int, int]]:
    return {
        int(prediction_index): (
            int(candidates.candidate_offsets[row]),
            int(candidates.candidate_offsets[row + 1]),
        )
        for row, prediction_index in enumerate(
            candidates.prediction_indices.tolist()
        )
    }


def _best_box_envelope(
    original_iou: np.ndarray,
    geometry: GeometryCandidates,
    geometry_iou: np.ndarray,
    *,
    verified_only: bool,
) -> np.ndarray:
    envelope = np.array(original_iou, dtype=np.float64, copy=True)
    for row, prediction_index in enumerate(
        geometry.prediction_indices.tolist()
    ):
        start = int(geometry.candidate_offsets[row])
        stop = int(geometry.candidate_offsets[row + 1])
        eligible = geometry.candidate_valid[start:stop]
        if verified_only:
            eligible = eligible & geometry.candidate_verified[start:stop]
        local_indices = np.flatnonzero(eligible) + start
        if len(local_indices):
            envelope[prediction_index] = np.maximum(
                envelope[prediction_index],
                np.max(geometry_iou[local_indices], axis=0),
            )
    return envelope


def _oracle_records(
    iou: np.ndarray, threshold: float
) -> tuple[
    list[tuple[float, bool]],
    list[tuple[int, int, float]],
]:
    """Return maximum one-to-one matches ranked by their assigned IoU."""

    matched_predictions, matched_ground_truth = maximum_matches(
        iou, threshold
    )
    assignments = {
        int(prediction_index): (
            int(gt_index),
            float(iou[prediction_index, gt_index]),
        )
        for prediction_index, gt_index in zip(
            matched_predictions.tolist(),
            matched_ground_truth.tolist(),
        )
    }
    records = [
        (
            assignments[index][1] if index in assignments else -1.0,
            index in assignments,
        )
        for index in range(len(iou))
    ]
    pairs = [
        (prediction_index, gt_index, matched_iou)
        for prediction_index, (gt_index, matched_iou) in sorted(
            assignments.items()
        )
    ]
    if len({gt_index for _, gt_index, _ in pairs}) != len(pairs):
        raise AssertionError("oracle matching reused a ground-truth box")
    return records, pairs


def _choice_for_geometry_match(
    *,
    prediction_index: int,
    gt_index: int,
    original_iou: np.ndarray,
    geometry: GeometryCandidates,
    geometry_iou: np.ndarray,
    lookup: Mapping[int, tuple[int, int]],
    verified_only: bool,
) -> tuple[str, float]:
    best_source = "original"
    best_iou = float(original_iou[prediction_index, gt_index])
    best_candidate_id: str | None = None
    interval = lookup.get(prediction_index)
    if interval is None:
        return best_source, best_iou
    start, stop = interval
    for candidate_index in range(start, stop):
        if not geometry.candidate_valid[candidate_index]:
            continue
        if verified_only and not geometry.candidate_verified[candidate_index]:
            continue
        value = float(geometry_iou[candidate_index, gt_index])
        candidate_id = geometry.candidate_ids[candidate_index]
        improves_iou = value > best_iou
        wins_candidate_tie = bool(
            value == best_iou
            and best_candidate_id is not None
            and candidate_id < best_candidate_id
        )
        # An exact original/candidate tie deliberately retains the immutable
        # original. Candidate/candidate ties use stable IDs, independent of
        # serialization order.
        if improves_iou or wins_candidate_tie:
            best_iou = value
            best_candidate_id = candidate_id
            best_source = (
                "geometry:"
                + geometry.candidate_sources[candidate_index]
            )
    return best_source, best_iou


def _finite_summary(values: Sequence[float]) -> dict[str, Any]:
    if not values:
        return {
            "count": 0,
            "minimum": None,
            "mean": None,
            "maximum": None,
        }
    finite = [float(value) for value in values]
    return {
        "count": len(finite),
        "minimum": float(min(finite)),
        "mean": float(statistics.fmean(finite)),
        "maximum": float(max(finite)),
    }


def _metric(
    records: Sequence[tuple[float, bool]],
    *,
    predictions: int,
    ground_truth: int,
    maximum_matches_count: int,
    matched_ious: Sequence[float] = (),
    selections: Mapping[str, int] | None = None,
    score_ordering: str,
) -> dict[str, Any]:
    ap, recall, precision = ranked_metrics(records, ground_truth)
    return {
        "predictions": int(predictions),
        "ground_truth": int(ground_truth),
        "true_positives": int(sum(bool(match) for _, match in records)),
        "maximum_matches": int(maximum_matches_count),
        "ap": float(ap),
        "ap_percent": float(ap * 100.0),
        "recall": float(recall),
        "precision": float(precision),
        "final_precision": float(precision),
        "score_ordering": score_ordering,
        "matched_iou": _finite_summary(matched_ious),
        "selection_counts": dict(sorted((selections or {}).items())),
    }


def _accumulate_oracle(
    *,
    method: str,
    threshold: float,
    iou: np.ndarray,
    records: dict[str, dict[float, list[tuple[float, bool]]]],
    maximum_match_counts: dict[str, dict[float, int]],
    matched_ious: dict[str, dict[float, list[float]]],
) -> list[tuple[int, int, float]]:
    scene_records, pairs = _oracle_records(iou, threshold)
    records[method][threshold].extend(scene_records)
    maximum_match_counts[method][threshold] += len(pairs)
    matched_ious[method][threshold].extend(
        matched_iou for _, _, matched_iou in pairs
    )
    return pairs


def _validate_original_corner_pairing(
    *,
    scene_id: str,
    prediction_corners: np.ndarray,
    geometry: GeometryCandidates,
) -> None:
    if (
        len(geometry.prediction_indices)
        and int(np.max(geometry.prediction_indices))
        >= len(prediction_corners)
    ):
        raise ValueError(
            f"{scene_id}: geometry prediction index exceeds exported "
            "predictions"
        )
    exported = prediction_corners[geometry.prediction_indices]
    if not np.allclose(
        exported,
        geometry.original_corners,
        rtol=0.0,
        atol=1e-6,
    ):
        difference = (
            0.0
            if not len(exported)
            else float(np.max(np.abs(exported - geometry.original_corners)))
        )
        raise ValueError(
            f"{scene_id}: geometry original_corners disagree point-wise "
            f"with exported prediction corners (max delta {difference})"
        )


def build_report(
    *,
    pred_root: str | Path,
    scene_list: str | Path,
    gt_root: str | Path,
    scan_root: str | Path,
    geometry_candidates_root: str | Path | None = None,
    supplemental_candidates_root: str | Path | None = None,
    exclude_scene_list: str | Path | None = None,
) -> dict[str, Any]:
    """Build the class-agnostic tri-fusion oracle report without writes."""

    prediction_root = Path(pred_root)
    gt_path = Path(gt_root)
    scans_path = Path(scan_root)
    requested_scenes = read_scene_ids(Path(scene_list))
    excluded = (
        set()
        if exclude_scene_list is None
        else set(read_scene_ids(Path(exclude_scene_list)))
    )
    scenes = [
        scene for scene in requested_scenes if scene not in excluded
    ]
    if not scenes:
        raise ValueError("No analysis scenes remain after exclusions")

    records = {
        method: {threshold: [] for threshold in THRESHOLDS}
        for method in _METHODS
    }
    maximum_match_counts = {
        method: {threshold: 0 for threshold in THRESHOLDS}
        for method in _METHODS
    }
    matched_ious = {
        method: {threshold: [] for threshold in THRESHOLDS}
        for method in _METHODS
    }
    selections = {
        method: {threshold: Counter() for threshold in THRESHOLDS}
        for method in _METHODS
    }
    prediction_counts = {method: 0 for method in _METHODS}
    total_ground_truth = 0
    geometry_inventory = Counter()
    supplemental_inventory = Counter()
    geometry_sources: Counter[str] = Counter()
    supplemental_sources: Counter[str] = Counter()
    original_corner_rows_checked = 0
    supplemental_score_rows_ignored = 0
    supplemental_label_rows_ignored = 0
    scene_reports: list[dict[str, Any]] = []

    for scene_id in scenes:
        prediction_corners, prediction_scores = load_scene_predictions(
            prediction_root / f"{scene_id}{PREDICTION_SUFFIX}"
        )
        geometry = _empty_geometry(scene_id)
        if geometry_candidates_root is not None:
            geometry_path = _resolve_scene_artifact(
                geometry_candidates_root,
                scene_id=scene_id,
                suffix=GEOMETRY_SUFFIX,
            )
            geometry = load_geometry_candidates(
                geometry_path, expected_scene_id=scene_id
            )
        supplemental = _empty_supplemental(scene_id)
        if supplemental_candidates_root is not None:
            supplemental_path = _resolve_scene_artifact(
                supplemental_candidates_root,
                scene_id=scene_id,
                suffix=SUPPLEMENTAL_SUFFIX,
            )
            supplemental = load_supplemental_candidates(
                supplemental_path, expected_scene_id=scene_id
            )
        _validate_original_corner_pairing(
            scene_id=scene_id,
            prediction_corners=prediction_corners,
            geometry=geometry,
        )
        original_corner_rows_checked += len(geometry.prediction_indices)

        transform = load_axis_alignment(scans_path, scene_id)
        original_minmax = _aligned_minmax(
            prediction_corners, transform
        )
        gt_payload = np.load(
            gt_path / f"{scene_id}_bbox.npy", allow_pickle=False
        )
        gt_minmax = center_size_to_minmax(gt_payload)
        original_iou = pairwise_aabb_iou(original_minmax, gt_minmax)
        geometry_iou = _candidate_iou(
            geometry, transform=transform, gt_minmax=gt_minmax
        )
        supplemental_minmax = _aligned_minmax(
            supplemental.candidate_corners, transform
        )
        supplemental_iou = pairwise_aabb_iou(
            supplemental_minmax, gt_minmax
        )
        best_all = _best_box_envelope(
            original_iou,
            geometry,
            geometry_iou,
            verified_only=False,
        )
        best_verified = _best_box_envelope(
            original_iou,
            geometry,
            geometry_iou,
            verified_only=True,
        )
        supplemental_all_indices = np.asarray(
            sorted(
                np.flatnonzero(supplemental.candidate_valid).tolist(),
                key=lambda index: supplemental.candidate_ids[index],
            ),
            dtype=np.int64,
        )
        supplemental_verified_indices = np.asarray(
            sorted(
                np.flatnonzero(
                    supplemental.candidate_valid
                    & supplemental.candidate_verified
                ).tolist(),
                key=lambda index: supplemental.candidate_ids[index],
            ),
            dtype=np.int64,
        )
        union_all = np.concatenate(
            (best_all, supplemental_iou[supplemental_all_indices]),
            axis=0,
        )
        union_verified = np.concatenate(
            (
                best_verified,
                supplemental_iou[supplemental_verified_indices],
            ),
            axis=0,
        )

        base_count = len(prediction_scores)
        prediction_counts[_BASELINE] += base_count
        prediction_counts[_BASELINE_ORACLE_SCORE] += base_count
        prediction_counts[_BEST_ALL] += base_count
        prediction_counts[_BEST_VERIFIED] += base_count
        prediction_counts[_UNION_ALL] += len(union_all)
        prediction_counts[_UNION_VERIFIED] += len(union_verified)
        total_ground_truth += len(gt_minmax)

        geometry_inventory.update(
            {
                "prediction_rows": len(geometry.prediction_indices),
                "candidates": len(geometry.candidate_corners),
                "valid_candidates": int(
                    np.count_nonzero(geometry.candidate_valid)
                ),
                "verified_candidates": int(
                    np.count_nonzero(
                        geometry.candidate_valid
                        & geometry.candidate_verified
                    )
                ),
            }
        )
        geometry_sources.update(geometry.candidate_sources)
        supplemental_inventory.update(
            {
                "candidates": len(supplemental.candidate_corners),
                "valid_candidates": len(supplemental_all_indices),
                "verified_candidates": len(
                    supplemental_verified_indices
                ),
            }
        )
        supplemental_sources.update(supplemental.candidate_sources)
        if supplemental.candidate_scores is not None:
            supplemental_score_rows_ignored += len(
                supplemental.candidate_scores
            )
        if supplemental.candidate_labels is not None:
            supplemental_label_rows_ignored += len(
                supplemental.candidate_labels
            )

        geometry_lookup = _geometry_lookup(geometry)
        scene_matches: dict[str, dict[str, int]] = {}
        for threshold in THRESHOLDS:
            threshold_name = _threshold_key(threshold)
            real_records, _, baseline_maximum = score_scene(
                original_iou, prediction_scores, threshold
            )
            records[_BASELINE][threshold].extend(real_records)
            maximum_match_counts[_BASELINE][threshold] += int(
                baseline_maximum
            )

            baseline_pairs = _accumulate_oracle(
                method=_BASELINE_ORACLE_SCORE,
                threshold=threshold,
                iou=original_iou,
                records=records,
                maximum_match_counts=maximum_match_counts,
                matched_ious=matched_ious,
            )
            selections[_BASELINE_ORACLE_SCORE][threshold][
                "original"
            ] += len(baseline_pairs)

            all_pairs = _accumulate_oracle(
                method=_BEST_ALL,
                threshold=threshold,
                iou=best_all,
                records=records,
                maximum_match_counts=maximum_match_counts,
                matched_ious=matched_ious,
            )
            for prediction_index, gt_index, envelope_iou in all_pairs:
                source, choice_iou = _choice_for_geometry_match(
                    prediction_index=prediction_index,
                    gt_index=gt_index,
                    original_iou=original_iou,
                    geometry=geometry,
                    geometry_iou=geometry_iou,
                    lookup=geometry_lookup,
                    verified_only=False,
                )
                if not np.isclose(
                    envelope_iou, choice_iou, rtol=0.0, atol=1e-12
                ):
                    raise AssertionError(
                        "all-candidate envelope is not realizable"
                    )
                selections[_BEST_ALL][threshold][source] += 1

            verified_pairs = _accumulate_oracle(
                method=_BEST_VERIFIED,
                threshold=threshold,
                iou=best_verified,
                records=records,
                maximum_match_counts=maximum_match_counts,
                matched_ious=matched_ious,
            )
            for prediction_index, gt_index, envelope_iou in verified_pairs:
                source, choice_iou = _choice_for_geometry_match(
                    prediction_index=prediction_index,
                    gt_index=gt_index,
                    original_iou=original_iou,
                    geometry=geometry,
                    geometry_iou=geometry_iou,
                    lookup=geometry_lookup,
                    verified_only=True,
                )
                if not np.isclose(
                    envelope_iou, choice_iou, rtol=0.0, atol=1e-12
                ):
                    raise AssertionError(
                        "verified-candidate envelope is not realizable"
                    )
                selections[_BEST_VERIFIED][threshold][source] += 1

            union_all_pairs = _accumulate_oracle(
                method=_UNION_ALL,
                threshold=threshold,
                iou=union_all,
                records=records,
                maximum_match_counts=maximum_match_counts,
                matched_ious=matched_ious,
            )
            for proposal_index, gt_index, envelope_iou in union_all_pairs:
                if proposal_index < base_count:
                    source, choice_iou = _choice_for_geometry_match(
                        prediction_index=proposal_index,
                        gt_index=gt_index,
                        original_iou=original_iou,
                        geometry=geometry,
                        geometry_iou=geometry_iou,
                        lookup=geometry_lookup,
                        verified_only=False,
                    )
                else:
                    supplemental_index = int(
                        supplemental_all_indices[
                            proposal_index - base_count
                        ]
                    )
                    source = (
                        "supplemental:"
                        + supplemental.candidate_sources[
                            supplemental_index
                        ]
                    )
                    choice_iou = float(
                        supplemental_iou[supplemental_index, gt_index]
                    )
                if not np.isclose(
                    envelope_iou, choice_iou, rtol=0.0, atol=1e-12
                ):
                    raise AssertionError(
                        "proposal-union envelope is not realizable"
                    )
                selections[_UNION_ALL][threshold][source] += 1

            union_verified_pairs = _accumulate_oracle(
                method=_UNION_VERIFIED,
                threshold=threshold,
                iou=union_verified,
                records=records,
                maximum_match_counts=maximum_match_counts,
                matched_ious=matched_ious,
            )
            for (
                proposal_index,
                gt_index,
                envelope_iou,
            ) in union_verified_pairs:
                if proposal_index < base_count:
                    source, choice_iou = _choice_for_geometry_match(
                        prediction_index=proposal_index,
                        gt_index=gt_index,
                        original_iou=original_iou,
                        geometry=geometry,
                        geometry_iou=geometry_iou,
                        lookup=geometry_lookup,
                        verified_only=True,
                    )
                else:
                    supplemental_index = int(
                        supplemental_verified_indices[
                            proposal_index - base_count
                        ]
                    )
                    source = (
                        "supplemental:"
                        + supplemental.candidate_sources[
                            supplemental_index
                        ]
                    )
                    choice_iou = float(
                        supplemental_iou[supplemental_index, gt_index]
                    )
                if not np.isclose(
                    envelope_iou, choice_iou, rtol=0.0, atol=1e-12
                ):
                    raise AssertionError(
                        "verified proposal-union envelope is not realizable"
                    )
                selections[_UNION_VERIFIED][threshold][source] += 1

            scene_matches[threshold_name] = {
                _BASELINE: int(
                    sum(match for _, match in real_records)
                ),
                _BASELINE_ORACLE_SCORE: len(baseline_pairs),
                _BEST_ALL: len(all_pairs),
                _BEST_VERIFIED: len(verified_pairs),
                _UNION_ALL: len(union_all_pairs),
                _UNION_VERIFIED: len(union_verified_pairs),
            }
        scene_reports.append(
            {
                "scene_id": scene_id,
                "predictions": base_count,
                "ground_truth": len(gt_minmax),
                "geometry_candidates": len(
                    geometry.candidate_corners
                ),
                "supplemental_candidates": len(
                    supplemental.candidate_corners
                ),
                "matches": scene_matches,
            }
        )

    threshold_reports: dict[str, Any] = {}
    ceilings_and_gaps: dict[str, Any] = {}
    for threshold in THRESHOLDS:
        threshold_name = _threshold_key(threshold)
        target = PAPER_PLUS_10_TARGET_AP_PERCENT[threshold]
        method_reports: dict[str, Any] = {}
        for method in _METHODS:
            ordering = (
                "exported_prediction_score_descending_stable"
                if method == _BASELINE
                else (
                    "matched_iou_descending_then_stable_scene_"
                    "prediction_order"
                )
            )
            report = _metric(
                records[method][threshold],
                predictions=prediction_counts[method],
                ground_truth=total_ground_truth,
                maximum_matches_count=maximum_match_counts[method][
                    threshold
                ],
                matched_ious=matched_ious[method][threshold],
                selections=selections[method][threshold],
                score_ordering=ordering,
            )
            report["target_ap_percent"] = float(target)
            report["gap_to_target_percentage_points"] = float(
                target - report["ap_percent"]
            )
            report["meets_target"] = bool(
                report["ap_percent"] >= target
            )
            method_reports[method] = report
        threshold_reports[threshold_name] = method_reports
        baseline_percent = method_reports[_BASELINE]["ap_percent"]
        threshold_gaps: dict[str, Any] = {
            "target_ap_percent": float(target)
        }
        threshold_gaps.update(
            {
                method: {
                    "ap_percent": method_reports[method]["ap_percent"],
                    "gap_to_target_percentage_points": method_reports[
                        method
                    ]["gap_to_target_percentage_points"],
                    "gain_over_baseline_percentage_points": float(
                        method_reports[method]["ap_percent"]
                        - baseline_percent
                    ),
                    "meets_target": method_reports[method]["meets_target"],
                }
                for method in _METHODS
            }
        )
        ceilings_and_gaps[threshold_name] = threshold_gaps

    return {
        "schema": REPORT_SCHEMA,
        "format_version": REPORT_FORMAT_VERSION,
        "scene_count": len(scenes),
        "requested_scene_count": len(requested_scenes),
        "excluded_scenes": sorted(excluded.intersection(requested_scenes)),
        "thresholds": threshold_reports,
        "ceilings_and_gaps": ceilings_and_gaps,
        "paper_plus_10_targets_ap_percent": {
            _threshold_key(threshold): float(target)
            for threshold, target in PAPER_PLUS_10_TARGET_AP_PERCENT.items()
        },
        "candidate_inventory": {
            "geometry": {
                **dict(geometry_inventory),
                "sources": dict(sorted(geometry_sources.items())),
                "original_corner_rows_checked": int(
                    original_corner_rows_checked
                ),
            },
            "supplemental": {
                **dict(supplemental_inventory),
                "sources": dict(sorted(supplemental_sources.items())),
                "score_rows_ignored_by_oracle": int(
                    supplemental_score_rows_ignored
                ),
                "label_rows_ignored_by_class_agnostic_evaluation": int(
                    supplemental_label_rows_ignored
                ),
            },
        },
        "protocol": {
            "geometry": (
                "world-frame oriented corners are transformed by ScanNet "
                "axisAlignment and enclosed as aligned AABBs"
            ),
            "best_box": (
                "one original-or-eligible-candidate choice per exported "
                "prediction and assigned ground-truth box"
            ),
            "proposal_union": (
                "best-box prediction rows plus eligible supplemental rows; "
                "each prediction/proposal and GT is used at most once"
            ),
            "all_candidates_means": "candidate_valid == true",
            "verified_only_means": (
                "candidate_valid == true and candidate_verified == true"
            ),
            "oracle_score_ordering": (
                "threshold-specific one-to-one matched predictions are "
                "ranked by IoU with their assigned GT; unmatched rows follow"
            ),
            "supplemental_scores_used": False,
            "supplemental_labels_used": False,
            "prediction_artifacts_mutated": False,
        },
        "oracle_disclaimer": (
            "Retrospective GT-conditioned upper bounds only. Oracle geometry "
            "selection, one-to-one assignment, and matched-IoU ordering use "
            "ScanNet ground truth and therefore are not deployable inference "
            "results, do not establish validation-free gains, and must not "
            "be used to tune or rewrite prediction artifacts."
        ),
        "scenes": scene_reports,
    }


def _path_is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _validate_output_path(
    output: Path,
    *,
    pred_root: Path,
    geometry_root: Path | None,
    supplemental_root: Path | None,
) -> None:
    resolved_output = output.resolve(strict=False)
    protected_roots = [pred_root.resolve(strict=False)]
    protected_files: list[Path] = []
    for optional in (geometry_root, supplemental_root):
        if optional is not None:
            resolved = optional.resolve(strict=False)
            if optional.is_dir():
                protected_roots.append(resolved)
            else:
                protected_files.append(resolved)
    if resolved_output in protected_files:
        raise ValueError(
            f"report output {output} must not overwrite an input artifact"
        )
    for root in protected_roots:
        if resolved_output == root or _path_is_within(resolved_output, root):
            raise ValueError(
                f"report output {output} must not be inside an input "
                f"artifact root {root}"
            )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Offline ScanNet baseline, best-box, proposal-union, and "
            "matched-IoU score-order oracle report."
        )
    )
    parser.add_argument("--pred-root", type=Path, required=True)
    parser.add_argument("--scene-list", type=Path, required=True)
    parser.add_argument("--gt-root", type=Path, required=True)
    parser.add_argument("--scan-root", type=Path, required=True)
    parser.add_argument(
        "--geometry-candidates-root",
        "--geometry-root",
        dest="geometry_candidates_root",
        type=Path,
    )
    parser.add_argument(
        "--supplemental-candidates-root",
        "--supplemental-root",
        dest="supplemental_candidates_root",
        type=Path,
    )
    parser.add_argument("--exclude-scene-list", type=Path)
    parser.add_argument("--output", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report = build_report(
        pred_root=args.pred_root,
        scene_list=args.scene_list,
        gt_root=args.gt_root,
        scan_root=args.scan_root,
        geometry_candidates_root=args.geometry_candidates_root,
        supplemental_candidates_root=args.supplemental_candidates_root,
        exclude_scene_list=args.exclude_scene_list,
    )
    rendered = json.dumps(
        report, indent=2, sort_keys=True, allow_nan=False
    )
    print(rendered)
    if args.output is not None:
        _validate_output_path(
            args.output,
            pred_root=args.pred_root,
            geometry_root=args.geometry_candidates_root,
            supplemental_root=args.supplemental_candidates_root,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
