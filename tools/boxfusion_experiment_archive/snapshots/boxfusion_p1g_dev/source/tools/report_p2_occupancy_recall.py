#!/usr/bin/env python3
"""Evaluate the read-only P2 occupancy Top-K proposal stream.

The P2 diagnostic archive contains both the frozen P1 residual proposals and
the P2 occupancy-selected subset/alternative stream.  This tool reports,
without changing any formal BoxFusion prediction:

* B6, P1-only, and P2-only class-agnostic recall;
* B6+P1, B6+P2, P1+P2, and B6+P1+P2 union recall;
* P1/P2 novel precision against B6 and P2 incremental precision after B6+P1;
* candidate volume, P1/P2 overlap, and observer runtime.

Every P1/P2 safety field is mandatory and validated fail-closed.  Matching is
stable score-descending, one-to-one, and uses the ScanNet evaluator's strict
``IoU > threshold`` convention.  P2 candidates are ordered by the frozen P1
objectness score; occupancy remains the anchor selector, not a calibrated
detection score.

Prediction pickle files are trusted local experiment artifacts and must never
come from an untrusted source.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from boxfusion.occupancy_topk import (  # noqa: E402
    P2_DIAGNOSTIC_SCHEMA,
    P2_SOURCE,
)
from boxfusion.residual_proposal import (  # noqa: E402
    P1_DIAGNOSTIC_SCHEMA,
    P1_FEATURE_NAMES,
)
from tools.report_p1_residual_recall import (  # noqa: E402
    DEFAULT_THRESHOLDS,
    P1Candidates,
    Predictions,
    center_size_to_corners,
    corners_to_minmax,
    load_axis_alignment,
    load_gt_boxes,
    load_p1_candidates,
    pairwise_aabb_iou,
    read_scene_ids,
    score_ordered_match,
    transform_corners,
    validate_thresholds,
)


REPORT_SCHEMA = "boxfusion.p2_occupancy_recall_report.v1"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SOURCE_NAMES = (
    "b6",
    "p1_only",
    "p2_only",
    "b6_p1_union",
    "b6_p2_union",
    "p1_p2_union",
    "b6_p1_p2_union",
)
_NOVEL_NAMES = (
    "p1_vs_b6",
    "p2_vs_b6",
    "p2_vs_b6_p1",
)


@dataclass(frozen=True)
class P2Candidates:
    corners_world: np.ndarray
    objectness_scores: np.ndarray
    occupancy_scores: np.ndarray
    candidate_ids: np.ndarray
    occupancy_ranks: np.ndarray
    incremental_runtime_seconds: float


@dataclass(frozen=True)
class P2Diagnostic:
    p1: P1Candidates
    p2: P2Candidates
    p1_checkpoint_sha256: str
    p2_checkpoint_sha256: str


@dataclass(frozen=True)
class CandidateStream:
    boxes: np.ndarray
    scores: np.ndarray
    ids: np.ndarray

    def __post_init__(self) -> None:
        boxes = np.asarray(self.boxes, dtype=np.float64)
        scores = np.asarray(self.scores, dtype=np.float64)
        ids = np.asarray(self.ids)
        if (
            boxes.ndim != 2
            or boxes.shape[1] != 6
            or scores.shape != (len(boxes),)
            or ids.shape != (len(boxes),)
            or ids.dtype.hasobject
            or ids.dtype.kind not in {"i", "u", "U", "S"}
            or not np.isfinite(boxes).all()
            or not np.isfinite(scores).all()
            or (len(boxes) and np.any(boxes[:, 3:] <= boxes[:, :3]))
        ):
            raise ValueError("candidate stream arrays are invalid or misaligned")
        object.__setattr__(self, "boxes", np.ascontiguousarray(boxes))
        object.__setattr__(self, "scores", np.ascontiguousarray(scores))
        object.__setattr__(self, "ids", np.asarray(ids))

    def __len__(self) -> int:
        return int(len(self.boxes))

    @classmethod
    def empty(cls) -> "CandidateStream":
        return cls(
            boxes=np.empty((0, 6), dtype=np.float64),
            scores=np.empty((0,), dtype=np.float64),
            ids=np.empty((0,), dtype=np.str_),
        )


def _scalar(
    archive: Mapping[str, np.ndarray],
    key: str,
    path: Path,
) -> np.ndarray:
    if key not in archive:
        raise ValueError(f"{path}: missing {key}")
    value = np.asarray(archive[key])
    if value.shape != () or value.dtype.hasobject:
        raise ValueError(f"{path}: {key} must be a non-object scalar")
    return value


def _text(
    archive: Mapping[str, np.ndarray],
    key: str,
    path: Path,
) -> str:
    value = _scalar(archive, key, path).item()
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    if not isinstance(value, str):
        raise ValueError(f"{path}: {key} must be text")
    return value


def _boolean(
    archive: Mapping[str, np.ndarray],
    key: str,
    path: Path,
) -> bool:
    value = _scalar(archive, key, path)
    if value.dtype != np.dtype(bool):
        raise ValueError(f"{path}: {key} must be Boolean")
    return bool(value.item())


def _integer(
    archive: Mapping[str, np.ndarray],
    key: str,
    path: Path,
) -> int:
    value = _scalar(archive, key, path)
    if not np.issubdtype(value.dtype, np.integer):
        raise ValueError(f"{path}: {key} must be integer")
    return int(value.item())


def _json_mapping(
    archive: Mapping[str, np.ndarray],
    key: str,
    path: Path,
) -> Mapping[str, Any]:
    raw = _text(archive, key, path)
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as error:
        raise ValueError(f"{path}: malformed {key}") from error
    if not isinstance(value, Mapping):
        raise ValueError(f"{path}: {key} must encode a mapping")
    return value


def _require_read_only_config(
    archive: Mapping[str, np.ndarray],
    prefix: str,
    path: Path,
) -> None:
    config = _json_mapping(archive, f"{prefix}_config_json", path)
    expected = {
        "enabled": True,
        "observer_only": True,
        "mutate": False,
    }
    for key, value in expected.items():
        if config.get(key) is not value:
            raise ValueError(
                f"{path}: unsafe {prefix}_config_json.{key}"
            )


def _require_false_array_if_present(
    archive: Mapping[str, np.ndarray],
    key: str,
    path: Path,
) -> None:
    if key not in archive:
        return
    values = np.asarray(archive[key])
    if values.dtype != np.dtype(bool) or np.any(values):
        raise ValueError(f"{path}: unsafe {key}")


def _validate_safety(
    archive: Mapping[str, np.ndarray],
    path: Path,
    *,
    expected_scene_id: str,
) -> tuple[str, str]:
    expected_text = {
        "scene_id": expected_scene_id,
        "p1_schema": P1_DIAGNOSTIC_SCHEMA,
        "p1_stage": "P1",
        "p1_profile": "p1_residual_proposal_observer",
        "p2_schema": P2_DIAGNOSTIC_SCHEMA,
        "p2_stage": "P2",
        "p2_profile": "p2_occupancy_topk_observer",
        "p2_source": P2_SOURCE,
    }
    for key, expected in expected_text.items():
        if _text(archive, key, path) != expected:
            raise ValueError(f"{path}: invalid {key}")

    for prefix in ("p1", "p2"):
        expected_boolean = {
            f"{prefix}_enabled": True,
            f"{prefix}_observer_only": True,
            f"{prefix}_uses_ground_truth": False,
            f"{prefix}_mutation_enabled": False,
            f"{prefix}_complete": True,
            f"{prefix}_class_agnostic": True,
        }
        for key, expected in expected_boolean.items():
            if _boolean(archive, key, path) is not expected:
                raise ValueError(f"{path}: unsafe {key}")
        if _integer(archive, f"{prefix}_applied_count", path) != 0:
            raise ValueError(f"{path}: {prefix} mutated formal output")
        for suffix in ("applied", "candidate_applied"):
            _require_false_array_if_present(
                archive, f"{prefix}_{suffix}", path
            )
        _require_read_only_config(archive, prefix, path)

    if _integer(archive, "p1_regression_dim", path) != 6:
        raise ValueError(f"{path}: P1 must remain class-agnostic 6-D")
    for prefix in ("p1", "p2"):
        names = np.asarray(archive.get(f"{prefix}_feature_names"))
        if (
            names.ndim != 1
            or names.dtype.hasobject
            or names.dtype.kind not in {"U", "S"}
            or tuple(str(value) for value in names.tolist())
            != tuple(P1_FEATURE_NAMES)
        ):
            raise ValueError(f"{path}: invalid {prefix}_feature_names")

    p1_sha = _text(archive, "p1_checkpoint_sha256", path).lower()
    p2_sha = _text(archive, "p2_checkpoint_sha256", path).lower()
    if _SHA256.fullmatch(p1_sha) is None:
        raise ValueError(f"{path}: invalid P1 checkpoint SHA")
    if _SHA256.fullmatch(p2_sha) is None:
        raise ValueError(f"{path}: invalid P2 checkpoint SHA")
    return p1_sha, p2_sha


def _validate_step_arrays(
    archive: Mapping[str, np.ndarray], path: Path
) -> float:
    integer_names = (
        "p2_step_frame_ids",
        "p2_step_provider_steps",
        "p2_step_input_voxel_counts",
        "p2_step_eligible_voxel_counts",
        "p2_step_selected_voxel_counts",
        "p2_step_candidate_counts",
    )
    rows: dict[str, np.ndarray] = {}
    lengths: set[int] = set()
    for key in integer_names:
        if key not in archive:
            raise ValueError(f"{path}: missing {key}")
        values = np.asarray(archive[key])
        if values.ndim != 1 or not np.issubdtype(
            values.dtype, np.integer
        ):
            raise ValueError(f"{path}: {key} must be integer [S]")
        rows[key] = np.asarray(values, dtype=np.int64)
        lengths.add(len(values))
    if "p2_step_seconds" not in archive:
        raise ValueError(f"{path}: missing p2_step_seconds")
    seconds = np.asarray(archive["p2_step_seconds"])
    if (
        seconds.ndim != 1
        or not np.issubdtype(seconds.dtype, np.floating)
        or not np.isfinite(seconds).all()
        or np.any(seconds < 0.0)
    ):
        raise ValueError(f"{path}: p2_step_seconds must be non-negative [S]")
    lengths.add(len(seconds))
    if len(lengths) != 1:
        raise ValueError(f"{path}: P2 step arrays disagree in length")
    if not lengths or next(iter(lengths)) < 1:
        raise ValueError(f"{path}: P2 observer never executed")
    for p1_key, p2_key in (
        ("p1_step_frame_ids", "p2_step_frame_ids"),
        ("p1_step_provider_steps", "p2_step_provider_steps"),
    ):
        if p1_key not in archive:
            raise ValueError(f"{path}: missing {p1_key}")
        p1_values = np.asarray(archive[p1_key])
        if (
            p1_values.ndim != 1
            or not np.issubdtype(p1_values.dtype, np.integer)
            or not np.array_equal(p1_values, rows[p2_key])
        ):
            raise ValueError(f"{path}: P1/P2 scheduling is not aligned")

    for key in integer_names[2:]:
        if np.any(rows[key] < 0):
            raise ValueError(f"{path}: negative count in {key}")
    inputs = rows["p2_step_input_voxel_counts"]
    eligible = rows["p2_step_eligible_voxel_counts"]
    selected = rows["p2_step_selected_voxel_counts"]
    candidates = rows["p2_step_candidate_counts"]
    if (
        np.any(eligible > inputs)
        or np.any(selected > eligible)
        or np.any(candidates > selected)
    ):
        raise ValueError(f"{path}: impossible P2 step count relationship")
    return float(np.sum(seconds, dtype=np.float64))


def _load_p2_candidates(
    archive: Mapping[str, np.ndarray],
    path: Path,
) -> P2Candidates:
    required = (
        "p2_candidate_ids",
        "p2_candidate_boxes",
        "p2_candidate_corners",
        "p2_candidate_objectness",
        "p2_candidate_occupancy_scores",
        "p2_candidate_occupancy_ranks",
    )
    missing = [key for key in required if key not in archive]
    if missing:
        raise ValueError(f"{path}: missing {missing[0]}")
    ids = np.asarray(archive["p2_candidate_ids"])
    boxes = np.asarray(archive["p2_candidate_boxes"])
    corners = np.asarray(archive["p2_candidate_corners"])
    objectness = np.asarray(archive["p2_candidate_objectness"])
    occupancy = np.asarray(archive["p2_candidate_occupancy_scores"])
    ranks = np.asarray(archive["p2_candidate_occupancy_ranks"])
    count = len(ids)
    if (
        ids.ndim != 1
        or ids.dtype.hasobject
        or ids.dtype.kind not in {"U", "S"}
        or len(np.unique(ids)) != count
        or boxes.shape != (count, 6)
        or corners.shape != (count, 8, 3)
        or objectness.shape != (count,)
        or occupancy.shape != (count,)
        or ranks.shape != (count,)
        or not np.issubdtype(boxes.dtype, np.floating)
        or not np.issubdtype(corners.dtype, np.floating)
        or not np.issubdtype(objectness.dtype, np.floating)
        or not np.issubdtype(occupancy.dtype, np.floating)
        or not np.issubdtype(ranks.dtype, np.integer)
        or not np.isfinite(boxes).all()
        or not np.isfinite(corners).all()
        or not np.isfinite(objectness).all()
        or not np.isfinite(occupancy).all()
        or (count and np.any(boxes[:, 3:] <= 0.0))
        or np.any((objectness < 0.0) | (objectness > 1.0))
        or np.any((occupancy < 0.0) | (occupancy > 1.0))
        or np.any(ranks < 0)
    ):
        raise ValueError(f"{path}: invalid or misaligned P2 candidates")
    expected = corners_to_minmax(center_size_to_corners(boxes))
    observed = corners_to_minmax(corners)
    if not np.allclose(expected, observed, rtol=1e-5, atol=1e-5):
        raise ValueError(f"{path}: P2 box and corner aliases disagree")
    return P2Candidates(
        corners_world=np.asarray(corners, dtype=np.float64),
        objectness_scores=np.asarray(objectness, dtype=np.float64),
        occupancy_scores=np.asarray(occupancy, dtype=np.float64),
        candidate_ids=np.asarray(ids),
        occupancy_ranks=np.asarray(ranks, dtype=np.int64),
        incremental_runtime_seconds=_validate_step_arrays(archive, path),
    )


def load_p2_diagnostic(
    path: str | os.PathLike[str],
    *,
    expected_scene_id: str,
) -> P2Diagnostic:
    diagnostic_path = Path(path)
    if not diagnostic_path.is_file():
        raise FileNotFoundError(diagnostic_path)
    with np.load(diagnostic_path, allow_pickle=False) as source:
        archive = {
            key: np.array(source[key], copy=True) for key in source.files
        }
    for key, value in archive.items():
        if np.asarray(value).dtype.hasobject:
            raise ValueError(f"{diagnostic_path}: object dtype in {key}")
    p1_sha, p2_sha = _validate_safety(
        archive, diagnostic_path, expected_scene_id=expected_scene_id
    )
    p2 = _load_p2_candidates(archive, diagnostic_path)
    p1 = load_p1_candidates(
        diagnostic_path, expected_scene_id=expected_scene_id
    )
    if p1.mutation_enabled is not False or p1.applied_count != 0:
        raise ValueError(f"{diagnostic_path}: unsafe P1 candidate stream")
    return P2Diagnostic(
        p1=p1,
        p2=p2,
        p1_checkpoint_sha256=p1_sha,
        p2_checkpoint_sha256=p2_sha,
    )


def load_predictions(
    path: str | os.PathLike[str],
) -> Predictions:
    """Load both BoxFusion detection-major and one-batch pickle layouts."""

    prediction_path = Path(path)
    if not prediction_path.is_file():
        raise FileNotFoundError(prediction_path)
    with prediction_path.open("rb") as handle:
        payload = pickle.load(handle)  # noqa: S301 - trusted local artifact
    if not isinstance(payload, (list, tuple)):
        raise ValueError(f"{prediction_path}: invalid BoxFusion predictions")

    def is_detection(value: Any) -> bool:
        if not isinstance(value, (list, tuple)) or len(value) != 3:
            return False
        try:
            corners = np.asarray(value[1])
            score = float(value[2])
        except (TypeError, ValueError):
            return False
        return bool(corners.shape == (8, 3) and math.isfinite(score))

    if all(is_detection(value) for value in payload):
        detections = payload
    elif (
        len(payload) == 1
        and isinstance(payload[0], (list, tuple))
        and all(is_detection(value) for value in payload[0])
    ):
        detections = payload[0]
    else:
        raise ValueError(
            f"{prediction_path}: expected detection-major predictions or "
            "one batch containing detections"
        )

    corners: list[np.ndarray] = []
    scores: list[float] = []
    for index, detection in enumerate(detections):
        value = np.asarray(detection[1])
        if (
            value.shape != (8, 3)
            or not np.issubdtype(value.dtype, np.number)
            or not np.isfinite(value).all()
        ):
            raise ValueError(
                f"{prediction_path}: invalid corners at detection {index}"
            )
        try:
            score = float(detection[2])
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"{prediction_path}: invalid score at detection {index}"
            ) from error
        if not math.isfinite(score):
            raise ValueError(
                f"{prediction_path}: non-finite score at detection {index}"
            )
        corners.append(np.asarray(value, dtype=np.float64))
        scores.append(score)
    return Predictions(
        corners_world=(
            np.stack(corners)
            if corners
            else np.empty((0, 8, 3), dtype=np.float64)
        ),
        scores=np.asarray(scores, dtype=np.float64),
    )


def _normalise_ids(values: np.ndarray) -> np.ndarray:
    result: list[str] = []
    for value in np.asarray(values).tolist():
        if isinstance(value, bytes):
            value = value.decode("utf-8")
        if isinstance(value, (int, np.integer)):
            result.append(f"int:{int(value)}")
        else:
            result.append(f"text:{str(value)}")
    return np.asarray(result, dtype=np.str_)


def _stream(
    boxes: np.ndarray,
    scores: np.ndarray,
    ids: np.ndarray,
) -> CandidateStream:
    return CandidateStream(
        boxes=np.asarray(boxes, dtype=np.float64),
        scores=np.asarray(scores, dtype=np.float64),
        ids=_normalise_ids(ids),
    )


def _concatenate(
    named_streams: Sequence[tuple[str, CandidateStream]],
) -> CandidateStream:
    if not named_streams:
        return CandidateStream.empty()
    box_parts: list[np.ndarray] = []
    score_parts: list[np.ndarray] = []
    id_parts: list[np.ndarray] = []
    for name, stream in named_streams:
        box_parts.append(stream.boxes)
        score_parts.append(stream.scores)
        id_parts.append(
            np.asarray(
                [f"{name}:{value}" for value in stream.ids.tolist()],
                dtype=np.str_,
            )
        )
    return CandidateStream(
        boxes=np.concatenate(box_parts, axis=0),
        scores=np.concatenate(score_parts, axis=0),
        ids=np.concatenate(id_parts, axis=0),
    )


def _merge_p1_p2_unique(
    p1: CandidateStream,
    p2: CandidateStream,
) -> tuple[CandidateStream, int]:
    rows: list[tuple[np.ndarray, float, str]] = []
    locations: dict[str, int] = {}
    for prefix, stream in (("p1", p1), ("p2", p2)):
        for index, candidate_id in enumerate(stream.ids.tolist()):
            existing = locations.get(candidate_id)
            if existing is not None:
                previous_box, previous_score, _ = rows[existing]
                if not np.allclose(
                    previous_box,
                    stream.boxes[index],
                    rtol=1e-5,
                    atol=1e-5,
                ) or not math.isclose(
                    previous_score,
                    float(stream.scores[index]),
                    rel_tol=1e-5,
                    abs_tol=1e-6,
                ):
                    raise ValueError(
                        "P1/P2 shared candidate ID disagrees in geometry "
                        "or frozen objectness"
                    )
                continue
            locations[candidate_id] = len(rows)
            rows.append(
                (
                    np.asarray(stream.boxes[index], dtype=np.float64),
                    float(stream.scores[index]),
                    f"{prefix}:{candidate_id}",
                )
            )
    overlap_count = len(p1) + len(p2) - len(rows)
    if not rows:
        return CandidateStream.empty(), overlap_count
    return (
        CandidateStream(
            boxes=np.stack([row[0] for row in rows]),
            scores=np.asarray([row[1] for row in rows]),
            ids=np.asarray([row[2] for row in rows], dtype=np.str_),
        ),
        overlap_count,
    )


def _match(stream: CandidateStream, gt_boxes: np.ndarray, threshold: float):
    return score_ordered_match(
        pairwise_aabb_iou(stream.boxes, gt_boxes),
        stream.scores,
        float(threshold),
        tie_break_ids=stream.ids,
    )


def _novel_metrics(
    reference: CandidateStream,
    candidate: CandidateStream,
    gt_boxes: np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    reference_match = _match(reference, gt_boxes, threshold)
    covered = np.zeros(len(gt_boxes), dtype=bool)
    covered[reference_match.matched_gt] = True
    candidate_iou = pairwise_aabb_iou(candidate.boxes, gt_boxes)
    novel = score_ordered_match(
        candidate_iou,
        candidate.scores,
        float(threshold),
        allowed_gt=~covered,
        tie_break_ids=candidate.ids,
    )
    true_positives = novel.true_positive_count
    return {
        "true_positives": int(true_positives),
        "candidate_count": int(len(candidate)),
        "precision": (
            float(true_positives / len(candidate)) if len(candidate) else 0.0
        ),
        "recall_gain": float(true_positives / max(len(gt_boxes), 1)),
    }


def evaluate_scene(
    *,
    b6: CandidateStream,
    p1: CandidateStream,
    p2: CandidateStream,
    gt_boxes: np.ndarray,
    thresholds: Sequence[float],
) -> dict[str, Any]:
    p1_p2, overlap_count = _merge_p1_p2_unique(p1, p2)
    sources = {
        "b6": b6,
        "p1_only": p1,
        "p2_only": p2,
        "b6_p1_union": _concatenate((("b6", b6), ("p1", p1))),
        "b6_p2_union": _concatenate((("b6", b6), ("p2", p2))),
        "p1_p2_union": p1_p2,
        "b6_p1_p2_union": _concatenate(
            (("b6", b6), ("p1p2", p1_p2))
        ),
    }
    by_threshold: dict[str, Any] = {}
    for threshold in thresholds:
        key = f"{float(threshold):.2f}"
        source_metrics: dict[str, Any] = {}
        for name, stream in sources.items():
            match = _match(stream, gt_boxes, float(threshold))
            source_metrics[name] = {
                "true_positives": match.true_positive_count,
                "prediction_count": int(len(stream)),
                "recall": float(
                    match.true_positive_count / max(len(gt_boxes), 1)
                ),
            }
        novel_metrics = {
            "p1_vs_b6": _novel_metrics(
                b6, p1, gt_boxes, float(threshold)
            ),
            "p2_vs_b6": _novel_metrics(
                b6, p2, gt_boxes, float(threshold)
            ),
            "p2_vs_b6_p1": _novel_metrics(
                sources["b6_p1_union"],
                p2,
                gt_boxes,
                float(threshold),
            ),
        }
        by_threshold[key] = {
            "ground_truth_count": int(len(gt_boxes)),
            "sources": source_metrics,
            "novel": novel_metrics,
        }
    return {
        "ground_truth_count": int(len(gt_boxes)),
        "candidate_counts": {
            name: int(len(stream)) for name, stream in sources.items()
        },
        "p1_p2_shared_candidate_ids": int(overlap_count),
        "thresholds": by_threshold,
    }


def _quantiles(values: Sequence[np.ndarray]) -> dict[str, float | None]:
    nonempty = [np.asarray(row, dtype=np.float64) for row in values if len(row)]
    if not nonempty:
        return {
            name: None for name in ("q10", "q25", "q50", "q75", "q90")
        }
    merged = np.concatenate(nonempty)
    return {
        name: float(value)
        for name, value in zip(
            ("q10", "q25", "q50", "q75", "q90"),
            np.quantile(merged, (0.10, 0.25, 0.50, 0.75, 0.90)),
        )
    }


def _validate_exact_scene_set(
    root: Path,
    scenes: Sequence[str],
    *,
    suffix: str,
    role: str,
) -> None:
    expected = {f"{scene}{suffix}" for scene in scenes}
    actual = {
        path.name
        for path in root.glob(f"scene*{suffix}")
        if path.is_file()
    }
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ValueError(
            f"{role} scene set mismatch; missing={missing[:4]}, "
            f"extra={extra[:4]}"
        )


def evaluate(
    *,
    scenes: Sequence[str],
    prediction_root: str | os.PathLike[str],
    diagnostics_root: str | os.PathLike[str],
    gt_root: str | os.PathLike[str],
    scans_root: str | os.PathLike[str],
    thresholds: Sequence[float] = DEFAULT_THRESHOLDS,
) -> dict[str, Any]:
    thresholds = validate_thresholds(thresholds)
    scene_ids = tuple(str(scene) for scene in scenes)
    if not scene_ids or len(scene_ids) != len(set(scene_ids)):
        raise ValueError("scenes must be non-empty and unique")
    prediction_directory = Path(prediction_root)
    diagnostic_directory = Path(diagnostics_root)
    gt_directory = Path(gt_root)
    scans_directory = Path(scans_root)
    for role, root in (
        ("prediction", prediction_directory),
        ("diagnostics", diagnostic_directory),
        ("ground-truth", gt_directory),
        ("scans", scans_directory),
    ):
        if not root.is_dir():
            raise FileNotFoundError(f"{role} root not found: {root}")
    _validate_exact_scene_set(
        prediction_directory,
        scene_ids,
        suffix="_boxes.pkl",
        role="prediction",
    )
    _validate_exact_scene_set(
        diagnostic_directory,
        scene_ids,
        suffix="_tracks.npz",
        role="diagnostic",
    )

    totals = {
        f"{threshold:.2f}": {
            "sources": {name: 0 for name in _SOURCE_NAMES},
            "novel": {name: 0 for name in _NOVEL_NAMES},
        }
        for threshold in thresholds
    }
    total_gt = 0
    candidate_counts = {name: 0 for name in _SOURCE_NAMES}
    shared_candidate_ids = 0
    p1_runtime = 0.0
    p2_runtime = 0.0
    p1_scores: list[np.ndarray] = []
    p2_objectness: list[np.ndarray] = []
    p2_occupancy: list[np.ndarray] = []
    observed_p1_sha: set[str] = set()
    observed_p2_sha: set[str] = set()
    per_scene: dict[str, Any] = {}

    for scene_id in scene_ids:
        baseline = load_predictions(
            prediction_directory / f"{scene_id}_boxes.pkl"
        )
        diagnostic = load_p2_diagnostic(
            diagnostic_directory / f"{scene_id}_tracks.npz",
            expected_scene_id=scene_id,
        )
        alignment = load_axis_alignment(scans_directory, scene_id)
        b6 = _stream(
            corners_to_minmax(
                transform_corners(baseline.corners_world, alignment)
            ),
            baseline.scores,
            np.asarray(
                [f"b6:{index:06d}" for index in range(len(baseline.scores))],
                dtype=np.str_,
            ),
        )
        p1 = _stream(
            corners_to_minmax(
                transform_corners(
                    diagnostic.p1.corners_world, alignment
                )
            ),
            diagnostic.p1.scores,
            diagnostic.p1.candidate_ids,
        )
        p2 = _stream(
            corners_to_minmax(
                transform_corners(
                    diagnostic.p2.corners_world, alignment
                )
            ),
            diagnostic.p2.objectness_scores,
            diagnostic.p2.candidate_ids,
        )
        gt_boxes = load_gt_boxes(gt_directory / f"{scene_id}_bbox.npy")
        scene_report = evaluate_scene(
            b6=b6,
            p1=p1,
            p2=p2,
            gt_boxes=gt_boxes,
            thresholds=thresholds,
        )
        scene_report["runtime_seconds"] = {
            "p1": float(diagnostic.p1.runtime_seconds),
            "p2_incremental": float(
                diagnostic.p2.incremental_runtime_seconds
            ),
            "p2_total": float(
                diagnostic.p1.runtime_seconds
                + diagnostic.p2.incremental_runtime_seconds
            ),
        }
        per_scene[scene_id] = scene_report

        total_gt += len(gt_boxes)
        for name, count in scene_report["candidate_counts"].items():
            candidate_counts[name] += int(count)
        shared_candidate_ids += int(
            scene_report["p1_p2_shared_candidate_ids"]
        )
        p1_runtime += diagnostic.p1.runtime_seconds
        p2_runtime += diagnostic.p2.incremental_runtime_seconds
        p1_scores.append(diagnostic.p1.scores)
        p2_objectness.append(diagnostic.p2.objectness_scores)
        p2_occupancy.append(diagnostic.p2.occupancy_scores)
        observed_p1_sha.add(diagnostic.p1_checkpoint_sha256)
        observed_p2_sha.add(diagnostic.p2_checkpoint_sha256)

        for key, threshold_row in scene_report["thresholds"].items():
            for name in _SOURCE_NAMES:
                totals[key]["sources"][name] += int(
                    threshold_row["sources"][name]["true_positives"]
                )
            for name in _NOVEL_NAMES:
                totals[key]["novel"][name] += int(
                    threshold_row["novel"][name]["true_positives"]
                )

    if len(observed_p1_sha) != 1 or len(observed_p2_sha) != 1:
        raise ValueError("checkpoint SHA changed across P2 scenes")

    threshold_report: dict[str, Any] = {}
    for key, total in totals.items():
        sources: dict[str, Any] = {}
        for name, true_positives in total["sources"].items():
            sources[name] = {
                "true_positives": int(true_positives),
                "prediction_count": int(candidate_counts[name]),
                "ground_truth_count": int(total_gt),
                "recall": float(true_positives / max(total_gt, 1)),
            }
        novel_denominators = {
            "p1_vs_b6": candidate_counts["p1_only"],
            "p2_vs_b6": candidate_counts["p2_only"],
            "p2_vs_b6_p1": candidate_counts["p2_only"],
        }
        novel: dict[str, Any] = {}
        for name, true_positives in total["novel"].items():
            denominator = int(novel_denominators[name])
            novel[name] = {
                "true_positives": int(true_positives),
                "candidate_count": denominator,
                "ground_truth_count": int(total_gt),
                "precision": (
                    float(true_positives / denominator)
                    if denominator
                    else 0.0
                ),
                "recall_gain": float(
                    true_positives / max(total_gt, 1)
                ),
            }
        threshold_report[key] = {
            "ground_truth_count": int(total_gt),
            "sources": sources,
            "novel": novel,
            # Flat aliases make experiment tables easy to generate without
            # weakening the structured source/novel contract above.
            "b6_recall": sources["b6"]["recall"],
            "p1_only_recall": sources["p1_only"]["recall"],
            "p2_only_recall": sources["p2_only"]["recall"],
            "b6_p1_union_recall": sources["b6_p1_union"]["recall"],
            "b6_p2_union_recall": sources["b6_p2_union"]["recall"],
            "p1_p2_union_recall": sources["p1_p2_union"]["recall"],
            "b6_p1_p2_union_recall": sources[
                "b6_p1_p2_union"
            ]["recall"],
            "p1_novel_precision": novel["p1_vs_b6"]["precision"],
            "p2_novel_precision": novel["p2_vs_b6"]["precision"],
            "p2_incremental_precision_after_b6_p1": novel[
                "p2_vs_b6_p1"
            ]["precision"],
        }

    sources = {
        name: {
            (
                "prediction_count"
                if name == "b6" or "union" in name
                else "candidate_count"
            ): int(candidate_counts[name]),
            "thresholds": {
                key: row["sources"][name]
                for key, row in threshold_report.items()
            },
        }
        for name in _SOURCE_NAMES
    }
    scene_count = len(scene_ids)
    return {
        "schema": REPORT_SCHEMA,
        "matching_contract": (
            "class-agnostic, stable score-descending by B6 score/frozen P1 "
            "objectness, strict IoU > threshold, one-to-one per scene"
        ),
        "p2_score_contract": (
            "occupancy selects Top-K anchors; frozen P1 objectness orders "
            "decoded P2 proposals"
        ),
        "observer_only": True,
        "safety": {
            "validated": True,
            "uses_ground_truth_online": False,
            "mutation_enabled": False,
            "applied_count": 0,
            "p1_checkpoint_sha256": next(iter(observed_p1_sha)),
            "p2_checkpoint_sha256": next(iter(observed_p2_sha)),
        },
        "scene_count": int(scene_count),
        "ground_truth_count": int(total_gt),
        "candidate_counts": {
            **{name: int(value) for name, value in candidate_counts.items()},
            "p1_p2_shared_candidate_ids": int(shared_candidate_ids),
        },
        "candidate_counts_per_scene": {
            name: float(value / max(scene_count, 1))
            for name, value in candidate_counts.items()
        },
        "runtime_seconds": {
            "p1": float(p1_runtime),
            "p2_incremental": float(p2_runtime),
            "p2_total": float(p1_runtime + p2_runtime),
            "p1_per_scene": float(p1_runtime / max(scene_count, 1)),
            "p2_incremental_per_scene": float(
                p2_runtime / max(scene_count, 1)
            ),
            "p2_total_per_scene": float(
                (p1_runtime + p2_runtime) / max(scene_count, 1)
            ),
        },
        "score_quantiles": {
            "p1_objectness": _quantiles(p1_scores),
            "p2_objectness": _quantiles(p2_objectness),
            "p2_occupancy": _quantiles(p2_occupancy),
        },
        "thresholds": threshold_report,
        "sources": sources,
        # Convenient aliases for downstream notebooks.
        "b6": sources["b6"],
        "p1": sources["p1_only"],
        "p2": sources["p2_only"],
        "b6_p1_union": sources["b6_p1_union"],
        "b6_p2_union": sources["b6_p2_union"],
        "p1_p2_union": sources["p1_p2_union"],
        "full_union": sources["b6_p1_p2_union"],
        "per_scene": per_scene,
    }


def build_report(
    *,
    diagnostics_root: str | os.PathLike[str],
    gt_root: str | os.PathLike[str],
    scans_root: str | os.PathLike[str],
    scene_list: str | os.PathLike[str],
    prediction_root: str | os.PathLike[str] | None = None,
    pred_root: str | os.PathLike[str] | None = None,
    thresholds: Sequence[float] = DEFAULT_THRESHOLDS,
) -> dict[str, Any]:
    if prediction_root is None:
        prediction_root = pred_root
    elif pred_root is not None and Path(prediction_root) != Path(pred_root):
        raise ValueError("prediction_root and pred_root disagree")
    if prediction_root is None:
        raise ValueError("a frozen B6 prediction root is required")
    return evaluate(
        scenes=read_scene_ids(scene_list),
        prediction_root=prediction_root,
        diagnostics_root=diagnostics_root,
        gt_root=gt_root,
        scans_root=scans_root,
        thresholds=thresholds,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-list", required=True, type=Path)
    parser.add_argument("--prediction-root", required=True, type=Path)
    parser.add_argument("--diagnostics-root", required=True, type=Path)
    parser.add_argument("--gt-root", required=True, type=Path)
    parser.add_argument("--scans-root", required=True, type=Path)
    parser.add_argument(
        "--thresholds", nargs="+", type=float, default=DEFAULT_THRESHOLDS
    )
    parser.add_argument("--output", "--output-json", dest="output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = evaluate(
        scenes=read_scene_ids(args.scene_list),
        prediction_root=args.prediction_root,
        diagnostics_root=args.diagnostics_root,
        gt_root=args.gt_root,
        scans_root=args.scans_root,
        thresholds=args.thresholds,
    )
    rendered = json.dumps(report, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_name(args.output.name + ".tmp")
        temporary.write_text(rendered + "\n", encoding="utf-8")
        os.replace(temporary, args.output)
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
