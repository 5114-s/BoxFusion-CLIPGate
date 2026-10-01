#!/usr/bin/env python3
"""Read-only, CPU-only TriFusion counterfactual AP evaluation.

This program evaluates already-produced observer diagnostics.  It never loads
the AP50 gate checkpoint, gate-training examples, or any runtime model.  For
M3, a frozen B6 prediction is replaced only when the corresponding diagnostic
row is both geometry-verified and gate-accepted; its frozen B6 score and row
position are preserved exactly.  An optional, separate M1/M2 ablation appends
confirmed, valid, verified missing-instance candidates at one explicit fixed
score.

The evaluator reads ScanNet ground truth for the exact scene IDs in
``--scene-list``.  Its output is retrospective evaluation, not deployable
inference evidence.  In particular, do not tune a gate, a fixed supplemental
score, or any threshold on a held-out split with this report.

Prediction pickle files are trusted local BoxFusion artifacts with the exact
layout ``[[(label, corners[8,3], score), ...]]``.  Never load untrusted pickle
files.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import pickle
import re
import sys
from typing import Any, Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from boxfusion.local_occupancy_msr_refiner import (  # noqa: E402
    OCCUPANCY_MSR_FEATURE_NAMES,
)
from boxfusion.quality_score import QUALITY_FEATURE_NAMES  # noqa: E402
from tools.report_b5_ap50_ablation import (  # noqa: E402
    aligned_aabb,
    class_agnostic_ap,
    load_axis_alignment,
    load_gt_boxes,
)
REPORT_SCHEMA = "boxfusion.trifusion.gate_counterfactual_evaluation.v1"
DIAGNOSTIC_SCHEMA = (
    "boxfusion.trifusion.occupancy_msr_observer.v1"
)
GATE_DIAGNOSTIC_SCHEMA = (
    "boxfusion.trifusion.ap50_safety_observer.v1"
)
MISSING_DIAGNOSTIC_SCHEMA = (
    "boxfusion.trifusion.missing_graph_observer.v1"
)
SUPPLEMENTAL_CANDIDATE_SCHEMA = (
    "boxfusion.trifusion.supplemental_candidates.v1"
)
CORNER_FRAME = "world_pre_axis_alignment"
PREDICTION_SUFFIX = "_boxes.pkl"
DIAGNOSTIC_SUFFIX = "_tracks.npz"
SUPPLEMENTAL_SUFFIX = "_supplemental_candidates.npz"
THRESHOLDS = (0.15, 0.25, 0.50)
SCENE_ID_PATTERN = re.compile(r"^scene[0-9]{4}_[0-9]{2}$")

M3_FEATURE_NAMES = tuple(OCCUPANCY_MSR_FEATURE_NAMES)
GATE_FEATURE_NAMES = (
    tuple(f"b6_original_{name}" for name in QUALITY_FEATURE_NAMES)
    + ("b6_original_features_available",)
    + tuple(
        f"occupancy_msr_{name}" for name in OCCUPANCY_MSR_FEATURE_NAMES
    )
)

FROZEN_METHOD = "frozen_b6"
M3_METHOD = "m3_verified_gate_accepted"
SUPPLEMENTAL_METHOD = "frozen_b6_plus_confirmed_m1_m2"
COMBINED_METHOD = (
    "m3_verified_gate_accepted_plus_confirmed_m1_m2"
)


@dataclass(frozen=True)
class FrozenPredictions:
    corners: np.ndarray
    scores: np.ndarray


@dataclass(frozen=True)
class MissingObserverRows:
    candidate_ids: tuple[str, ...]
    candidate_sources: tuple[str, ...]
    corners: np.ndarray
    valid: np.ndarray
    verified: np.ndarray
    confirmed: np.ndarray


@dataclass(frozen=True)
class M3Diagnostics:
    result_indices: np.ndarray
    stable_ids: np.ndarray
    original_corners: np.ndarray
    candidate_corners: np.ndarray
    replace_mask: np.ndarray
    gate_evaluated: np.ndarray
    gate_accepted: np.ndarray
    candidate_verified: np.ndarray
    missing: MissingObserverRows | None


@dataclass(frozen=True)
class SupplementalRows:
    candidate_ids: tuple[str, ...]
    candidate_sources: tuple[str, ...]
    corners: np.ndarray
    eligible: np.ndarray
    artifact_score_rows: int


def apply_m3_counterfactual(
    frozen: FrozenPredictions,
    diagnostic: M3Diagnostics,
) -> FrozenPredictions:
    """Return a detached geometry-only M3 counterfactual."""

    corners = np.array(frozen.corners, copy=True)
    selected_rows = np.flatnonzero(diagnostic.replace_mask)
    result_indices = diagnostic.result_indices[selected_rows]
    corners[result_indices] = diagnostic.candidate_corners[selected_rows]
    scores = np.array(frozen.scores, copy=True)
    if not np.array_equal(scores, frozen.scores):
        raise AssertionError("M3 counterfactual changed frozen B6 scores")
    return FrozenPredictions(
        corners=_readonly(corners, dtype=np.float64),
        scores=_readonly(scores, dtype=np.float64),
    )


def _readonly(value: Any, *, dtype: Any) -> np.ndarray:
    result = np.array(value, dtype=dtype, copy=True)
    result.setflags(write=False)
    return result


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.suffix.lower() != ".npz":
        raise ValueError(f"{path}: expected an NPZ artifact")
    with np.load(path, allow_pickle=False) as archive:
        return {
            name: np.array(archive[name], copy=True)
            for name in archive.files
        }


def _require_fields(
    payload: Mapping[str, Any], fields: set[str], *, path: Path
) -> None:
    missing = fields - set(payload)
    if missing:
        raise ValueError(f"{path}: missing fields {sorted(missing)}")


def _scalar_text(value: Any, *, name: str, path: Path) -> str:
    array = np.asarray(value)
    if array.shape != () or array.dtype.hasobject:
        raise ValueError(f"{path}: {name} must be a scalar string")
    scalar = array.item()
    if isinstance(scalar, bytes):
        scalar = scalar.decode("utf-8")
    if not isinstance(scalar, str) or not scalar:
        raise ValueError(f"{path}: {name} must be a non-empty string")
    return scalar


def _scalar_bool(value: Any, *, name: str, path: Path) -> bool:
    array = np.asarray(value)
    if array.shape != () or array.dtype != np.bool_:
        raise ValueError(f"{path}: {name} must be a Boolean scalar")
    return bool(array.item())


def _scalar_int(value: Any, *, name: str, path: Path) -> int:
    array = np.asarray(value)
    if array.shape != () or array.dtype.kind not in "iu":
        raise ValueError(f"{path}: {name} must be an integer scalar")
    return int(array.item())


def _integer_rows(
    value: Any,
    *,
    name: str,
    path: Path,
    rows: int | None = None,
    nonnegative: bool = False,
) -> np.ndarray:
    array = np.asarray(value)
    if (
        array.ndim != 1
        or array.dtype.kind not in "iu"
        or (rows is not None and array.shape != (rows,))
    ):
        expected = "[N]" if rows is None else f"[{rows}]"
        raise ValueError(f"{path}: {name} must be integer {expected}")
    result = np.asarray(array, dtype=np.int64)
    if nonnegative and np.any(result < 0):
        raise ValueError(f"{path}: {name} must be non-negative")
    return result


def _boolean_rows(
    value: Any, *, name: str, path: Path, rows: int
) -> np.ndarray:
    array = np.asarray(value)
    if array.shape != (rows,) or array.dtype != np.bool_:
        raise ValueError(
            f"{path}: {name} must have Boolean shape [{rows}]"
        )
    return np.asarray(array, dtype=bool)


def _floating_rows(
    value: Any,
    *,
    name: str,
    path: Path,
    rows: int,
    finite: bool = True,
) -> np.ndarray:
    array = np.asarray(value)
    if array.shape != (rows,) or array.dtype.kind not in "fiu":
        raise ValueError(f"{path}: {name} must be numeric shape [{rows}]")
    result = np.asarray(array, dtype=np.float64)
    if finite and not np.isfinite(result).all():
        raise ValueError(f"{path}: {name} must be finite")
    return result


def _string_rows(
    value: Any,
    *,
    name: str,
    path: Path,
    rows: int,
    unique: bool = False,
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
        if not isinstance(scalar, str) or not scalar:
            raise ValueError(
                f"{path}: {name} entries must be non-empty strings"
            )
        output.append(scalar)
    if unique and len(set(output)) != len(output):
        raise ValueError(f"{path}: {name} entries must be unique")
    return tuple(output)


def _corners(
    value: Any, *, name: str, path: Path, rows: int | None = None
) -> np.ndarray:
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{path}: {name} must be numeric") from error
    if (
        array.ndim != 3
        or array.shape[1:] != (8, 3)
        or (rows is not None and array.shape[0] != rows)
    ):
        expected = "[N,8,3]" if rows is None else f"[{rows},8,3]"
        raise ValueError(f"{path}: {name} must have shape {expected}")
    if not np.isfinite(array).all():
        raise ValueError(f"{path}: {name} contains non-finite values")
    if len(array):
        extents = array.max(axis=1) - array.min(axis=1)
        if np.any(extents <= 0.0):
            raise ValueError(f"{path}: {name} contains degenerate boxes")
    return array


def _matrix(
    value: Any,
    *,
    name: str,
    path: Path,
    rows: int,
    columns: int,
) -> np.ndarray:
    array = np.asarray(value)
    if (
        array.shape != (rows, columns)
        or array.dtype.kind not in "fiu"
    ):
        raise ValueError(
            f"{path}: {name} must be numeric shape [{rows},{columns}]"
        )
    result = np.asarray(array, dtype=np.float64)
    if not np.isfinite(result).all():
        raise ValueError(f"{path}: {name} must be finite")
    return result


def _exact_array_match(
    left: np.ndarray,
    right: np.ndarray,
    *,
    name: str,
    path: Path,
) -> None:
    if left.shape != right.shape or not np.array_equal(left, right):
        raise ValueError(f"{path}: {name} does not exactly align")


def read_fixed_scene_ids(path: Path) -> list[str]:
    """Read an exact, duplicate-free ScanNet scene list."""

    if not path.is_file():
        raise FileNotFoundError(path)
    scenes = [line.strip() for line in path.read_text().splitlines()]
    scenes = [scene for scene in scenes if scene]
    if not scenes:
        raise ValueError(f"{path}: scene list is empty")
    if len(set(scenes)) != len(scenes):
        raise ValueError(f"{path}: duplicate scene IDs are forbidden")
    invalid = [scene for scene in scenes if not SCENE_ID_PATTERN.fullmatch(scene)]
    if invalid:
        raise ValueError(f"{path}: invalid ScanNet scene IDs {invalid}")
    return scenes


def _scene_list_sha256(scenes: Sequence[str]) -> str:
    canonical = "".join(f"{scene}\n" for scene in scenes).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def load_frozen_predictions(path: Path) -> FrozenPredictions:
    """Load and strictly validate one trusted frozen BoxFusion pickle."""

    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    if not isinstance(payload, (list, tuple)) or len(payload) != 1:
        raise ValueError(
            f"{path}: prediction pickle must have exactly one batch"
        )
    detections = payload[0]
    if not isinstance(detections, (list, tuple)):
        raise ValueError(f"{path}: prediction batch must be a sequence")
    corner_rows: list[np.ndarray] = []
    score_rows: list[float] = []
    for index, detection in enumerate(detections):
        if not isinstance(detection, (list, tuple)) or len(detection) != 3:
            raise ValueError(
                f"{path}: prediction {index} must be (label,corners,score)"
            )
        corner = np.asarray(detection[1], dtype=np.float64)
        if corner.shape != (8, 3):
            raise ValueError(
                f"{path}: prediction {index} corners must be [8,3]"
            )
        try:
            score = float(detection[2])
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"{path}: prediction {index} score is not numeric"
            ) from error
        corner_rows.append(corner)
        score_rows.append(score)
    corners = (
        np.stack(corner_rows)
        if corner_rows
        else np.empty((0, 8, 3), dtype=np.float64)
    )
    corners = _corners(
        corners, name="frozen prediction corners", path=path
    )
    scores = np.asarray(score_rows, dtype=np.float64)
    if (
        not np.isfinite(scores).all()
        or np.any(scores < 0.0)
        or np.any(scores > 1.0)
    ):
        raise ValueError(f"{path}: frozen scores must be finite in [0,1]")
    return FrozenPredictions(
        corners=_readonly(corners, dtype=np.float64),
        scores=_readonly(scores, dtype=np.float64),
    )


def _validate_config(payload: Mapping[str, Any], *, path: Path) -> None:
    raw = _scalar_text(
        payload["trifusion_config_json"],
        name="trifusion_config_json",
        path=path,
    )
    try:
        config = json.loads(raw)
    except json.JSONDecodeError as error:
        raise ValueError(
            f"{path}: trifusion_config_json is invalid JSON"
        ) from error
    if not isinstance(config, Mapping):
        raise ValueError(f"{path}: trifusion config must be an object")
    if (
        config.get("enabled") is not True
        or config.get("mutate") is not False
        or config.get("collect_diagnostics") is not True
    ):
        raise ValueError(
            f"{path}: TriFusion config is not enabled observer-only"
        )
    safety = config.get("safety_gate")
    if (
        not isinstance(safety, Mapping)
        or safety.get("enabled") is not True
        or safety.get("mutate") is not False
        or safety.get("collect_diagnostics") is not True
    ):
        raise ValueError(
            f"{path}: AP50 gate config is not enabled observer-only"
        )


def _validate_missing_observer(
    payload: Mapping[str, Any], *, scene_id: str, path: Path
) -> MissingObserverRows:
    required = {
        "trifusion_missing_diagnostics_schema",
        "trifusion_missing_enabled",
        "trifusion_missing_mutation_enabled",
        "trifusion_missing_candidate_ids",
        "trifusion_missing_sources",
        "trifusion_missing_corners",
        "trifusion_missing_valid",
        "trifusion_missing_verified",
        "trifusion_missing_confirmed",
        "trifusion_missing_applied",
    }
    _require_fields(payload, required, path=path)
    schema = _scalar_text(
        payload["trifusion_missing_diagnostics_schema"],
        name="trifusion_missing_diagnostics_schema",
        path=path,
    )
    if schema != MISSING_DIAGNOSTIC_SCHEMA:
        raise ValueError(
            f"{path}: unsupported missing observer schema {schema!r}"
        )
    if not _scalar_bool(
        payload["trifusion_missing_enabled"],
        name="trifusion_missing_enabled",
        path=path,
    ):
        raise ValueError(f"{path}: missing-instance observer was disabled")
    if _scalar_bool(
        payload["trifusion_missing_mutation_enabled"],
        name="trifusion_missing_mutation_enabled",
        path=path,
    ):
        raise ValueError(
            f"{path}: missing-instance observer allowed mutation"
        )
    corners = _corners(
        payload["trifusion_missing_corners"],
        name="trifusion_missing_corners",
        path=path,
    )
    rows = len(corners)
    track_ids = _integer_rows(
        payload["trifusion_missing_candidate_ids"],
        name="trifusion_missing_candidate_ids",
        path=path,
        rows=rows,
        nonnegative=True,
    )
    if len(np.unique(track_ids)) != rows:
        raise ValueError(f"{path}: missing candidate IDs are not unique")
    sources = _string_rows(
        payload["trifusion_missing_sources"],
        name="trifusion_missing_sources",
        path=path,
        rows=rows,
    )
    if any(source != "missing_graph" for source in sources):
        raise ValueError(f"{path}: non-M1/M2 supplemental source present")
    valid = _boolean_rows(
        payload["trifusion_missing_valid"],
        name="trifusion_missing_valid",
        path=path,
        rows=rows,
    )
    verified = _boolean_rows(
        payload["trifusion_missing_verified"],
        name="trifusion_missing_verified",
        path=path,
        rows=rows,
    )
    confirmed = _boolean_rows(
        payload["trifusion_missing_confirmed"],
        name="trifusion_missing_confirmed",
        path=path,
        rows=rows,
    )
    applied = _boolean_rows(
        payload["trifusion_missing_applied"],
        name="trifusion_missing_applied",
        path=path,
        rows=rows,
    )
    if np.any(verified & ~valid):
        raise ValueError(
            f"{path}: verified missing rows must also be valid"
        )
    if np.any(~confirmed):
        raise ValueError(
            f"{path}: exported missing rows must all be confirmed"
        )
    if np.any(applied):
        raise ValueError(
            f"{path}: missing-instance applied flags must all be false"
        )
    candidate_ids = tuple(
        f"{scene_id}:missing_graph:track:{int(track_id)}"
        for track_id in track_ids
    )
    return MissingObserverRows(
        candidate_ids=candidate_ids,
        candidate_sources=sources,
        corners=_readonly(corners, dtype=np.float64),
        valid=_readonly(valid, dtype=bool),
        verified=_readonly(verified, dtype=bool),
        confirmed=_readonly(confirmed, dtype=bool),
    )


def load_m3_diagnostics(
    path: Path,
    *,
    scene_id: str,
    frozen: FrozenPredictions,
    require_missing_observer: bool,
) -> M3Diagnostics:
    """Load diagnostics and prove result-index/stable-ID/frozen alignment."""

    payload = _load_npz(path)
    required = {
        "scene_id",
        "trifusion_diagnostics_schema",
        "trifusion_enabled",
        "trifusion_mutation_enabled",
        "trifusion_config_json",
        "trifusion_result_indices",
        "trifusion_stable_ids",
        "trifusion_feature_names",
        "trifusion_features",
        "trifusion_original_corners",
        "trifusion_candidate_corners",
        "trifusion_gate_diagnostics_schema",
        "trifusion_gate_enabled",
        "trifusion_gate_mutation_enabled",
        "trifusion_gate_feature_names",
        "trifusion_gate_features",
        "trifusion_gate_evaluated",
        "trifusion_gate_accepted",
        "trifusion_gate_reason",
        "trifusion_gate_lower_confidence_delta",
        "trifusion_gate_delta_mean",
        "trifusion_gate_delta_std",
        "trifusion_gate_improvement_probability",
        "trifusion_gate_harm_probability",
        "trifusion_gate_original_iou",
        "trifusion_gate_candidate_iou",
        "trifusion_gate_cross_iou25_probability",
        "trifusion_gate_cross_iou50_probability",
        "trifusion_candidate_valid",
        "trifusion_is_candidate",
        "trifusion_candidate_verified",
        "trifusion_applied",
        "trifusion_reason",
        "trifusion_source",
        "result_indices",
        "track_ids",
        "refit_candidate_corners",
        "scores",
    }
    _require_fields(payload, required, path=path)
    stored_scene = _scalar_text(
        payload["scene_id"], name="scene_id", path=path
    )
    if stored_scene != scene_id:
        raise ValueError(
            f"{path}: scene {stored_scene!r} != {scene_id!r}"
        )
    schema = _scalar_text(
        payload["trifusion_diagnostics_schema"],
        name="trifusion_diagnostics_schema",
        path=path,
    )
    if schema != DIAGNOSTIC_SCHEMA:
        raise ValueError(f"{path}: unsupported TriFusion schema {schema!r}")
    gate_schema = _scalar_text(
        payload["trifusion_gate_diagnostics_schema"],
        name="trifusion_gate_diagnostics_schema",
        path=path,
    )
    if gate_schema != GATE_DIAGNOSTIC_SCHEMA:
        raise ValueError(f"{path}: unsupported gate schema {gate_schema!r}")
    if not _scalar_bool(
        payload["trifusion_enabled"],
        name="trifusion_enabled",
        path=path,
    ):
        raise ValueError(f"{path}: TriFusion observer was disabled")
    if _scalar_bool(
        payload["trifusion_mutation_enabled"],
        name="trifusion_mutation_enabled",
        path=path,
    ):
        raise ValueError(f"{path}: TriFusion observer allowed mutation")
    if not _scalar_bool(
        payload["trifusion_gate_enabled"],
        name="trifusion_gate_enabled",
        path=path,
    ):
        raise ValueError(f"{path}: AP50 gate was disabled")
    if _scalar_bool(
        payload["trifusion_gate_mutation_enabled"],
        name="trifusion_gate_mutation_enabled",
        path=path,
    ):
        raise ValueError(f"{path}: AP50 gate allowed mutation")
    _validate_config(payload, path=path)

    output_indices = _integer_rows(
        payload["result_indices"],
        name="result_indices",
        path=path,
        nonnegative=True,
    )
    observed = len(output_indices)
    if len(np.unique(output_indices)) != observed or np.any(
        output_indices[1:] <= output_indices[:-1]
    ):
        raise ValueError(
            f"{path}: result_indices must be unique and increasing"
        )
    if observed and int(output_indices[-1]) >= len(frozen.scores):
        raise ValueError(
            f"{path}: result_indices exceed frozen prediction count"
        )
    output_stable_ids = _integer_rows(
        payload["track_ids"],
        name="track_ids",
        path=path,
        rows=observed,
    )
    if len(np.unique(output_stable_ids)) != observed:
        raise ValueError(f"{path}: track_ids must be unique")
    output_corners = _corners(
        payload["refit_candidate_corners"],
        name="refit_candidate_corners",
        path=path,
        rows=observed,
    )
    output_scores = _floating_rows(
        payload["scores"],
        name="scores",
        path=path,
        rows=observed,
    )
    _exact_array_match(
        output_corners,
        frozen.corners[output_indices],
        name="diagnostic output corners vs frozen prediction rows",
        path=path,
    )
    _exact_array_match(
        output_scores,
        frozen.scores[output_indices],
        name="diagnostic output scores vs frozen prediction rows",
        path=path,
    )

    result_indices = _integer_rows(
        payload["trifusion_result_indices"],
        name="trifusion_result_indices",
        path=path,
        nonnegative=True,
    )
    rows = len(result_indices)
    if len(np.unique(result_indices)) != rows or np.any(
        result_indices[1:] <= result_indices[:-1]
    ):
        raise ValueError(
            f"{path}: trifusion_result_indices must be unique and increasing"
        )
    if rows and int(result_indices[-1]) >= len(frozen.scores):
        raise ValueError(
            f"{path}: TriFusion result index exceeds frozen predictions"
        )
    stable_ids = _integer_rows(
        payload["trifusion_stable_ids"],
        name="trifusion_stable_ids",
        path=path,
        rows=rows,
    )
    if len(np.unique(stable_ids)) != rows:
        raise ValueError(f"{path}: trifusion_stable_ids must be unique")
    positions = np.searchsorted(output_indices, result_indices)
    if np.any(positions >= observed) or not np.array_equal(
        output_indices[positions], result_indices
    ):
        raise ValueError(
            f"{path}: TriFusion rows are absent from result_indices"
        )
    if not np.array_equal(output_stable_ids[positions], stable_ids):
        raise ValueError(
            f"{path}: TriFusion stable IDs do not align with result indices"
        )

    original_corners = _corners(
        payload["trifusion_original_corners"],
        name="trifusion_original_corners",
        path=path,
        rows=rows,
    )
    candidate_corners = _corners(
        payload["trifusion_candidate_corners"],
        name="trifusion_candidate_corners",
        path=path,
        rows=rows,
    )
    _exact_array_match(
        original_corners,
        frozen.corners[result_indices],
        name="M3 original corners vs frozen prediction rows",
        path=path,
    )

    feature_names = _string_rows(
        payload["trifusion_feature_names"],
        name="trifusion_feature_names",
        path=path,
        rows=len(M3_FEATURE_NAMES),
        unique=True,
    )
    if feature_names != M3_FEATURE_NAMES:
        raise ValueError(f"{path}: M3 feature schema/order mismatch")
    _matrix(
        payload["trifusion_features"],
        name="trifusion_features",
        path=path,
        rows=rows,
        columns=len(M3_FEATURE_NAMES),
    )
    gate_feature_names = _string_rows(
        payload["trifusion_gate_feature_names"],
        name="trifusion_gate_feature_names",
        path=path,
        rows=len(GATE_FEATURE_NAMES),
        unique=True,
    )
    if gate_feature_names != GATE_FEATURE_NAMES:
        raise ValueError(f"{path}: gate feature schema/order mismatch")
    _matrix(
        payload["trifusion_gate_features"],
        name="trifusion_gate_features",
        path=path,
        rows=rows,
        columns=len(GATE_FEATURE_NAMES),
    )

    valid = _boolean_rows(
        payload["trifusion_candidate_valid"],
        name="trifusion_candidate_valid",
        path=path,
        rows=rows,
    )
    is_candidate = _boolean_rows(
        payload["trifusion_is_candidate"],
        name="trifusion_is_candidate",
        path=path,
        rows=rows,
    )
    verified = _boolean_rows(
        payload["trifusion_candidate_verified"],
        name="trifusion_candidate_verified",
        path=path,
        rows=rows,
    )
    evaluated = _boolean_rows(
        payload["trifusion_gate_evaluated"],
        name="trifusion_gate_evaluated",
        path=path,
        rows=rows,
    )
    accepted = _boolean_rows(
        payload["trifusion_gate_accepted"],
        name="trifusion_gate_accepted",
        path=path,
        rows=rows,
    )
    applied = _boolean_rows(
        payload["trifusion_applied"],
        name="trifusion_applied",
        path=path,
        rows=rows,
    )
    if np.any(is_candidate & ~valid):
        raise ValueError(f"{path}: M3 candidates must also be valid")
    if np.any(verified & ~is_candidate):
        raise ValueError(f"{path}: verified M3 rows must be candidates")
    if not np.array_equal(evaluated, verified):
        raise ValueError(
            f"{path}: gate_evaluated must exactly align with verified M3"
        )
    if np.any(accepted & ~evaluated):
        raise ValueError(
            f"{path}: accepted M3 rows must have gate_evaluated=true"
        )
    if np.any(applied):
        raise ValueError(f"{path}: trifusion_applied must be all false")
    sources = _string_rows(
        payload["trifusion_source"],
        name="trifusion_source",
        path=path,
        rows=rows,
    )
    if any(source != "occupancy_msr" for source in sources):
        raise ValueError(f"{path}: unexpected M3 source")
    _string_rows(
        payload["trifusion_reason"],
        name="trifusion_reason",
        path=path,
        rows=rows,
    )
    _string_rows(
        payload["trifusion_gate_reason"],
        name="trifusion_gate_reason",
        path=path,
        rows=rows,
    )

    gate_numeric_names = (
        "trifusion_gate_lower_confidence_delta",
        "trifusion_gate_delta_mean",
        "trifusion_gate_delta_std",
        "trifusion_gate_improvement_probability",
        "trifusion_gate_harm_probability",
        "trifusion_gate_original_iou",
        "trifusion_gate_candidate_iou",
        "trifusion_gate_cross_iou25_probability",
        "trifusion_gate_cross_iou50_probability",
    )
    gate_values = {
        name: _floating_rows(
            payload[name],
            name=name,
            path=path,
            rows=rows,
            finite=False,
        )
        for name in gate_numeric_names
    }
    accepted_rows = np.flatnonzero(accepted)
    for name, values in gate_values.items():
        if not np.isfinite(values[accepted_rows]).all():
            raise ValueError(
                f"{path}: accepted rows require finite {name}"
            )
    probability_names = (
        "trifusion_gate_improvement_probability",
        "trifusion_gate_harm_probability",
        "trifusion_gate_original_iou",
        "trifusion_gate_candidate_iou",
        "trifusion_gate_cross_iou25_probability",
        "trifusion_gate_cross_iou50_probability",
    )
    for name in probability_names:
        values = gate_values[name][accepted_rows]
        if np.any((values < 0.0) | (values > 1.0)):
            raise ValueError(f"{path}: accepted {name} must be in [0,1]")
    if np.any(
        gate_values["trifusion_gate_delta_std"][accepted_rows] < 0.0
    ):
        raise ValueError(f"{path}: gate delta std must be non-negative")

    missing = (
        _validate_missing_observer(
            payload, scene_id=scene_id, path=path
        )
        if require_missing_observer
        else None
    )
    replace_mask = (
        valid & is_candidate & verified & evaluated & accepted
    )
    return M3Diagnostics(
        result_indices=_readonly(result_indices, dtype=np.int64),
        stable_ids=_readonly(stable_ids, dtype=np.int64),
        original_corners=_readonly(original_corners, dtype=np.float64),
        candidate_corners=_readonly(candidate_corners, dtype=np.float64),
        replace_mask=_readonly(replace_mask, dtype=bool),
        gate_evaluated=_readonly(evaluated, dtype=bool),
        gate_accepted=_readonly(accepted, dtype=bool),
        candidate_verified=_readonly(verified, dtype=bool),
        missing=missing,
    )


def load_confirmed_supplemental(
    path: Path,
    *,
    scene_id: str,
    missing: MissingObserverRows,
) -> SupplementalRows:
    """Validate a strict M1/M2 export against its source diagnostics."""

    payload = _load_npz(path)
    required = {
        "schema",
        "format_version",
        "scene_id",
        "corner_frame",
        "candidate_corners",
        "candidate_ids",
        "candidate_sources",
        "candidate_valid",
        "candidate_verified",
        "candidate_confirmed",
        "observer_only",
        "uses_ground_truth",
    }
    _require_fields(payload, required, path=path)
    schema = _scalar_text(payload["schema"], name="schema", path=path)
    if schema != SUPPLEMENTAL_CANDIDATE_SCHEMA:
        raise ValueError(
            f"{path}: unsupported supplemental schema {schema!r}"
        )
    if _scalar_int(
        payload["format_version"], name="format_version", path=path
    ) != 1:
        raise ValueError(f"{path}: unsupported supplemental format version")
    stored_scene = _scalar_text(
        payload["scene_id"], name="scene_id", path=path
    )
    if stored_scene != scene_id:
        raise ValueError(
            f"{path}: scene {stored_scene!r} != {scene_id!r}"
        )
    frame = _scalar_text(
        payload["corner_frame"], name="corner_frame", path=path
    )
    if frame != CORNER_FRAME:
        raise ValueError(f"{path}: unexpected corner frame {frame!r}")
    if not _scalar_bool(
        payload["observer_only"], name="observer_only", path=path
    ):
        raise ValueError(f"{path}: supplemental artifact is not observer-only")
    if _scalar_bool(
        payload["uses_ground_truth"], name="uses_ground_truth", path=path
    ):
        raise ValueError(
            f"{path}: supplemental artifact used ground truth"
        )

    corners = _corners(
        payload["candidate_corners"],
        name="candidate_corners",
        path=path,
    )
    rows = len(corners)
    candidate_ids = _string_rows(
        payload["candidate_ids"],
        name="candidate_ids",
        path=path,
        rows=rows,
        unique=True,
    )
    sources = _string_rows(
        payload["candidate_sources"],
        name="candidate_sources",
        path=path,
        rows=rows,
    )
    if any(source != "missing_graph" for source in sources):
        raise ValueError(f"{path}: non-M1/M2 supplemental source present")
    valid = _boolean_rows(
        payload["candidate_valid"],
        name="candidate_valid",
        path=path,
        rows=rows,
    )
    verified = _boolean_rows(
        payload["candidate_verified"],
        name="candidate_verified",
        path=path,
        rows=rows,
    )
    confirmed = _boolean_rows(
        payload["candidate_confirmed"],
        name="candidate_confirmed",
        path=path,
        rows=rows,
    )
    if np.any(verified & ~valid):
        raise ValueError(
            f"{path}: verified supplemental rows must also be valid"
        )
    for optional_applied in ("candidate_applied", "applied"):
        if optional_applied in payload:
            flags = _boolean_rows(
                payload[optional_applied],
                name=optional_applied,
                path=path,
                rows=rows,
            )
            if np.any(flags):
                raise ValueError(
                    f"{path}: supplemental applied flags must all be false"
                )

    if candidate_ids != missing.candidate_ids:
        raise ValueError(
            f"{path}: supplemental IDs do not align with diagnostics"
        )
    if sources != missing.candidate_sources:
        raise ValueError(
            f"{path}: supplemental sources do not align with diagnostics"
        )
    _exact_array_match(
        corners,
        missing.corners,
        name="supplemental corners vs missing observer rows",
        path=path,
    )
    for name, actual, expected in (
        ("valid", valid, missing.valid),
        ("verified", verified, missing.verified),
        ("confirmed", confirmed, missing.confirmed),
    ):
        _exact_array_match(
            actual,
            expected,
            name=f"supplemental {name} vs missing observer rows",
            path=path,
        )
    artifact_score_rows = 0
    if "candidate_scores" in payload:
        scores = _floating_rows(
            payload["candidate_scores"],
            name="candidate_scores",
            path=path,
            rows=rows,
        )
        if np.any((scores < 0.0) | (scores > 1.0)):
            raise ValueError(
                f"{path}: artifact candidate scores must be in [0,1]"
            )
        artifact_score_rows = rows
    eligible = valid & verified & confirmed
    return SupplementalRows(
        candidate_ids=candidate_ids,
        candidate_sources=sources,
        corners=_readonly(corners, dtype=np.float64),
        eligible=_readonly(eligible, dtype=bool),
        artifact_score_rows=artifact_score_rows,
    )


def _method_metrics(
    predictions: Mapping[
        str, Mapping[str, Sequence[tuple[np.ndarray, float]]]
    ],
    *,
    ground_truth: Mapping[str, np.ndarray],
) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for method, method_predictions in predictions.items():
        thresholds: dict[str, Any] = {}
        for threshold in THRESHOLDS:
            canonical = class_agnostic_ap(
                method_predictions, ground_truth, threshold
            )
            key = f"AP{int(round(threshold * 100)):02d}"
            thresholds[key] = {
                "iou_threshold": threshold,
                "predictions": int(
                    sum(
                        len(rows)
                        for rows in method_predictions.values()
                    )
                ),
                "ground_truth": int(canonical["ground_truth_count"]),
                "matched": int(canonical["true_positives"]),
                "false_positives": int(canonical["false_positives"]),
                "average_precision": float(canonical["ap"]),
                "ap_percent": float(canonical["ap"]) * 100.0,
                "recall": float(canonical["recall"]),
                "final_precision": float(canonical["precision"]),
            }
        output[method] = thresholds
    baseline = output[FROZEN_METHOD]
    for method, thresholds in output.items():
        for key, metric in thresholds.items():
            metric["delta_ap_percent_vs_frozen_b6"] = (
                metric["ap_percent"] - baseline[key]["ap_percent"]
            )
    return output


def evaluate(
    *,
    pred_root: Path,
    diagnostics_root: Path,
    scene_list: Path,
    gt_root: Path,
    scan_root: Path,
    supplemental_candidates_root: Path | None = None,
    supplemental_fixed_score: float | None = None,
) -> dict[str, Any]:
    """Evaluate exact fixed scenes without modifying any input artifact."""

    if (supplemental_candidates_root is None) != (
        supplemental_fixed_score is None
    ):
        raise ValueError(
            "supplemental root and explicit fixed score must be provided "
            "together"
        )
    if supplemental_fixed_score is not None and (
        not np.isfinite(supplemental_fixed_score)
        or supplemental_fixed_score < 0.0
        or supplemental_fixed_score > 1.0
    ):
        raise ValueError("supplemental fixed score must be finite in [0,1]")
    for name, root in (
        ("prediction", pred_root),
        ("diagnostics", diagnostics_root),
        ("ground-truth", gt_root),
        ("scan", scan_root),
    ):
        if not root.is_dir():
            raise FileNotFoundError(f"{name} root is not a directory: {root}")
    if (
        supplemental_candidates_root is not None
        and not supplemental_candidates_root.is_dir()
    ):
        raise FileNotFoundError(
            "supplemental root is not a directory: "
            f"{supplemental_candidates_root}"
        )

    scenes = read_fixed_scene_ids(scene_list)
    methods = [FROZEN_METHOD, M3_METHOD]
    if supplemental_candidates_root is not None:
        methods.extend((SUPPLEMENTAL_METHOD, COMBINED_METHOD))
    aligned_predictions: dict[
        str, dict[str, list[tuple[np.ndarray, float]]]
    ] = {
        method: {} for method in methods
    }
    ground_truth: dict[str, np.ndarray] = {}
    scene_reports: list[dict[str, Any]] = []
    totals = {
        "frozen_predictions": 0,
        "m3_rows": 0,
        "m3_verified": 0,
        "gate_evaluated": 0,
        "gate_accepted": 0,
        "m3_replacements": 0,
        "supplemental_rows": 0,
        "supplemental_eligible": 0,
        "supplemental_artifact_score_rows_ignored": 0,
    }

    for scene_id in scenes:
        prediction_path = pred_root / f"{scene_id}{PREDICTION_SUFFIX}"
        diagnostic_path = (
            diagnostics_root / f"{scene_id}{DIAGNOSTIC_SUFFIX}"
        )
        frozen = load_frozen_predictions(prediction_path)
        diagnostic = load_m3_diagnostics(
            diagnostic_path,
            scene_id=scene_id,
            frozen=frozen,
            require_missing_observer=(
                supplemental_candidates_root is not None
            ),
        )

        selected_rows = np.flatnonzero(diagnostic.replace_mask)
        selected_result_indices = diagnostic.result_indices[selected_rows]
        m3 = apply_m3_counterfactual(frozen, diagnostic)
        m3_corners = np.asarray(m3.corners)
        m3_scores = np.asarray(m3.scores)

        method_corners: dict[str, np.ndarray] = {
            FROZEN_METHOD: np.asarray(frozen.corners),
            M3_METHOD: m3_corners,
        }
        method_scores: dict[str, np.ndarray] = {
            FROZEN_METHOD: np.asarray(frozen.scores),
            M3_METHOD: m3_scores,
        }
        supplemental_rows = 0
        supplemental_eligible = 0
        if supplemental_candidates_root is not None:
            if diagnostic.missing is None:
                raise AssertionError("missing observer validation was skipped")
            supplemental_path = (
                supplemental_candidates_root
                / f"{scene_id}{SUPPLEMENTAL_SUFFIX}"
            )
            supplemental = load_confirmed_supplemental(
                supplemental_path,
                scene_id=scene_id,
                missing=diagnostic.missing,
            )
            order = np.asarray(
                sorted(
                    np.flatnonzero(supplemental.eligible).tolist(),
                    key=lambda index: supplemental.candidate_ids[index],
                ),
                dtype=np.int64,
            )
            supplemental_corners = supplemental.corners[order]
            fixed_scores = np.full(
                len(order),
                float(supplemental_fixed_score),
                dtype=np.float64,
            )
            method_corners[SUPPLEMENTAL_METHOD] = np.concatenate(
                (frozen.corners, supplemental_corners), axis=0
            )
            method_scores[SUPPLEMENTAL_METHOD] = np.concatenate(
                (frozen.scores, fixed_scores), axis=0
            )
            method_corners[COMBINED_METHOD] = np.concatenate(
                (m3_corners, supplemental_corners), axis=0
            )
            method_scores[COMBINED_METHOD] = np.concatenate(
                (m3_scores, fixed_scores), axis=0
            )
            supplemental_rows = len(supplemental.corners)
            supplemental_eligible = len(order)
            totals[
                "supplemental_artifact_score_rows_ignored"
            ] += supplemental.artifact_score_rows

        transform = load_axis_alignment(scan_root, scene_id)
        gt_boxes = load_gt_boxes(gt_root, scene_id)
        ground_truth[scene_id] = gt_boxes
        scene_matches: dict[str, dict[str, int]] = {}
        for method in methods:
            corners = method_corners[method]
            scores = method_scores[method]
            if len(corners) != len(scores):
                raise AssertionError(
                    f"{scene_id}: {method} corners/scores misaligned"
                )
            scene_predictions = [
                (aligned_aabb(corner, transform), float(score))
                for corner, score in zip(corners, scores)
            ]
            aligned_predictions[method][scene_id] = scene_predictions
            scene_matches[method] = {}
            for threshold in THRESHOLDS:
                scene_metric = class_agnostic_ap(
                    {scene_id: scene_predictions},
                    {scene_id: gt_boxes},
                    threshold,
                )
                scene_matches[method][
                    f"AP{int(round(threshold * 100)):02d}"
                ] = int(scene_metric["true_positives"])

        totals["frozen_predictions"] += len(frozen.scores)
        totals["m3_rows"] += len(diagnostic.result_indices)
        totals["m3_verified"] += int(
            np.count_nonzero(diagnostic.candidate_verified)
        )
        totals["gate_evaluated"] += int(
            np.count_nonzero(diagnostic.gate_evaluated)
        )
        totals["gate_accepted"] += int(
            np.count_nonzero(diagnostic.gate_accepted)
        )
        totals["m3_replacements"] += len(selected_rows)
        totals["supplemental_rows"] += supplemental_rows
        totals["supplemental_eligible"] += supplemental_eligible
        scene_reports.append(
            {
                "scene_id": scene_id,
                "frozen_predictions": len(frozen.scores),
                "ground_truth": len(gt_boxes),
                "m3_rows": len(diagnostic.result_indices),
                "m3_verified": int(
                    np.count_nonzero(diagnostic.candidate_verified)
                ),
                "gate_evaluated": int(
                    np.count_nonzero(diagnostic.gate_evaluated)
                ),
                "gate_accepted": int(
                    np.count_nonzero(diagnostic.gate_accepted)
                ),
                "m3_replacements": len(selected_rows),
                "m3_replaced_result_indices": (
                    selected_result_indices.astype(int).tolist()
                ),
                "m3_replaced_stable_ids": (
                    diagnostic.stable_ids[selected_rows]
                    .astype(int)
                    .tolist()
                ),
                "supplemental_rows": supplemental_rows,
                "supplemental_eligible": supplemental_eligible,
                "matches": scene_matches,
            }
        )

    metrics = _method_metrics(
        aligned_predictions,
        ground_truth=ground_truth,
    )
    return {
        "schema": REPORT_SCHEMA,
        "format_version": 1,
        "evaluation_only": True,
        "uses_ground_truth": True,
        "runtime_or_training_component": False,
        "gate_checkpoint_loaded": False,
        "gate_training_inputs_read": False,
        "prediction_artifacts_mutated": False,
        "diagnostic_artifacts_mutated": False,
        "cpu_only_numpy_evaluation": True,
        "scene_list": {
            "count": len(scenes),
            "sha256": _scene_list_sha256(scenes),
            "scene_ids": list(scenes),
            "selection": "exactly the listed scenes; no directory discovery",
        },
        "thresholds": list(THRESHOLDS),
        "methods": methods,
        "metrics": metrics,
        "inventory": {key: int(value) for key, value in totals.items()},
        "protocol": {
            "class_agnostic": True,
            "m3_selection": (
                "candidate_valid == true AND is_candidate == true AND "
                "candidate_verified == true AND "
                "trifusion_gate_evaluated == true AND "
                "trifusion_gate_accepted == true"
            ),
            "m3_score": (
                "exact frozen B6 score at the same result index"
            ),
            "m3_row_order": "exact frozen B6 row order",
            "supplemental_enabled": (
                supplemental_candidates_root is not None
            ),
            "supplemental_selection": (
                "candidate_confirmed AND candidate_valid AND "
                "candidate_verified"
            ),
            "supplemental_fixed_score": supplemental_fixed_score,
            "supplemental_artifact_scores_used": False,
            "supplemental_equal_score_order": (
                "candidate_id ascending, appended after frozen rows"
            ),
            "geometry": (
                "world_pre_axis_alignment corners are transformed by "
                "ScanNet axisAlignment and enclosed as aligned AABBs"
            ),
            "matching": (
                "repository canonical class-agnostic global score ranking, "
                "strict IoU > threshold, per-scene one-to-one matching, "
                "and continuous VOC AP"
            ),
            "input_contract": (
                "strict schemas; stable/result/frozen alignment; all M3 "
                "and missing-observer applied flags false"
            ),
        },
        "ground_truth_scope": {
            "files_read": [
                f"{scene_id}_bbox.npy" for scene_id in scenes
            ],
            "gate_training_ground_truth_read": False,
            "other_scene_ground_truth_read": False,
        },
        "heldout_warning": (
            "EVALUATION ONLY: this report reads ground truth for AP. Do not "
            "use its AP values to tune the gate, its thresholds, or the "
            "supplemental fixed score on a held-out split. Freeze every "
            "choice on a disjoint development split before held-out use."
        ),
        "scenes": scene_reports,
    }


def _path_is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _validate_report_output(
    output: Path,
    *,
    protected_roots: Sequence[Path],
    protected_files: Sequence[Path],
) -> None:
    resolved_output = output.resolve(strict=False)
    for path in protected_files:
        if resolved_output == path.resolve(strict=False):
            raise ValueError(
                f"report output must not overwrite input file {path}"
            )
    for root in protected_roots:
        resolved_root = root.resolve(strict=False)
        if resolved_output == resolved_root or _path_is_within(
            resolved_output, resolved_root
        ):
            raise ValueError(
                f"report output must not be inside input root {root}"
            )
    if output.exists():
        raise FileExistsError(
            f"refusing to overwrite existing report: {output}"
        )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Strict CPU-only AP15/AP25/AP50 counterfactual evaluation of "
            "TriFusion observer diagnostics."
        )
    )
    parser.add_argument("--pred-root", type=Path, required=True)
    parser.add_argument("--diagnostics-root", type=Path, required=True)
    parser.add_argument("--scene-list", type=Path, required=True)
    parser.add_argument("--gt-root", type=Path, required=True)
    parser.add_argument("--scan-root", type=Path, required=True)
    parser.add_argument(
        "--supplemental-candidates-root", type=Path
    )
    parser.add_argument("--supplemental-fixed-score", type=float)
    parser.add_argument(
        "--output",
        type=Path,
        help=(
            "Optional new JSON report path. Input roots remain read-only and "
            "an existing report is never overwritten."
        ),
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report = evaluate(
        pred_root=args.pred_root,
        diagnostics_root=args.diagnostics_root,
        scene_list=args.scene_list,
        gt_root=args.gt_root,
        scan_root=args.scan_root,
        supplemental_candidates_root=args.supplemental_candidates_root,
        supplemental_fixed_score=args.supplemental_fixed_score,
    )
    rendered = json.dumps(
        report, indent=2, sort_keys=True, allow_nan=False
    )
    print(rendered)
    if args.output is not None:
        protected_roots = [
            args.pred_root,
            args.diagnostics_root,
            args.gt_root,
            args.scan_root,
        ]
        if args.supplemental_candidates_root is not None:
            protected_roots.append(args.supplemental_candidates_root)
        _validate_report_output(
            args.output,
            protected_roots=protected_roots,
            protected_files=[args.scene_list],
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
