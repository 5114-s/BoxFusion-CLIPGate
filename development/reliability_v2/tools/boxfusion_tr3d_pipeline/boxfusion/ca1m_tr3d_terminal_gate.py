"""GT-free CA-1M terminal-TR3D quality/benefit selection.

This module is intentionally smaller than an active prediction hook.  It
consumes one immutable CA-1M terminal observer cache plus row-aligned native
14-D evidence for the anchors and TR3D candidates.  It may return replacement
indices, but it has no prediction writer and no ground-truth input.

The policy contains two scene-grouped, train-only logistic heads:

* ``quality25`` estimates whether a candidate reaches evaluator IoU 0.25;
* ``benefit05`` estimates an identity-preserving IoU improvement of 0.05.

Both heads must pass their frozen thresholds.  At most one candidate is
selected for each anchor, ordered deterministically by benefit probability,
quality probability, raw TR3D score, and candidate row.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Mapping

import numpy as np

from .ca1m_native_b6_observer import FEATURE_NAMES as NATIVE_FEATURE_NAMES
from .ca1m_tr3d_terminal import (
    BOX_MODE,
    COORDINATE_FRAME,
    CORNER_SEMANTICS,
    DEFAULT_NEAR_IOU,
    SCHEMA as TERMINAL_OBSERVER_SCHEMA,
    associate_terminal_candidates,
    world_aabb,
)


POLICY_SCHEMA = "boxfusion.ca1m_tr3d_terminal_gate_policy.v1"
FEATURE_SCHEMA = "boxfusion.ca1m_tr3d_terminal_gate_features.v1"
SELECTION_SCHEMA = "boxfusion.ca1m_tr3d_terminal_gate_selection.v1"
QUALITY_TARGET = "candidate_max_gt_iou_strict_gt_0.25"
BENEFIT_TARGET = "same_best_gt_and_same_gt_iou_gain_ge_0.05"
SELECTION_RULE = (
    "quality25_and_benefit05_then_per_anchor_"
    "benefit_quality_candidate_score_row_v1"
)

RELATION_FEATURE_NAMES = (
    "candidate_minus_anchor_score",
    "candidate_anchor_iou",
    "center_distance_over_anchor_diagonal",
    "log_candidate_over_anchor_volume",
    "extent_log_ratio_l2",
    "log1p_candidate_point_support",
    "log1p_candidate_point_density",
    "candidate_point_support_fraction",
    "candidate_global_rank_fraction",
    "candidate_anchor_group_rank_fraction",
    "log1p_anchor_group_size",
    "candidate_score_minus_best_sibling",
)
FEATURE_NAMES = (
    tuple(f"anchor_{name}" for name in NATIVE_FEATURE_NAMES)
    + tuple(f"candidate_{name}" for name in NATIVE_FEATURE_NAMES)
    + RELATION_FEATURE_NAMES
)

_SCENE_RE = re.compile(r"^[0-9]{8}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_EXPECTED_GATE_TRAIN_FOLDS = (2, 3, 4)
_EXPECTED_CALIBRATION_FOLDS = (0,)
_EXPECTED_ONE_TIME_AUDIT_FOLDS = (1,)


def _readonly(value: Any, dtype: np.dtype) -> np.ndarray:
    result = np.ascontiguousarray(np.asarray(value, dtype=dtype))
    result.setflags(write=False)
    return result


def _scalar(archive: Any, key: str) -> Any:
    try:
        value = np.asarray(archive[key])
    except (KeyError, TypeError) as error:
        raise ValueError(f"terminal cache is missing {key}") from error
    if value.shape != ():
        raise ValueError(f"terminal cache field {key} must be scalar")
    return value.item()


def _array(archive: Any, key: str) -> np.ndarray:
    try:
        return np.array(archive[key], copy=True)
    except (KeyError, TypeError) as error:
        raise ValueError(f"terminal cache is missing {key}") from error


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_native_evidence(
    value: Any, *, rows: int, label: str
) -> np.ndarray:
    evidence = np.asarray(value, dtype=np.float64)
    expected = (rows, len(NATIVE_FEATURE_NAMES))
    if evidence.shape != expected:
        raise ValueError(f"{label} native evidence must have shape {expected}")
    if (
        not np.isfinite(evidence).all()
        or np.any(evidence < 0.0)
        or np.any(evidence > 1.0)
    ):
        raise ValueError(f"{label} native evidence must be finite in [0,1]")
    return np.ascontiguousarray(evidence, dtype=np.float64)


@dataclass(frozen=True)
class _TerminalInputs:
    scene_id: str
    anchor_corners: np.ndarray
    anchor_scores: np.ndarray
    candidate_corners: np.ndarray
    candidate_scores: np.ndarray
    candidate_point_count: np.ndarray
    point_count: int
    best_anchor_indices: np.ndarray
    best_anchor_iou: np.ndarray
    best_anchor_center_distance_m: np.ndarray
    near_mask: np.ndarray
    materialized_active_verified: bool


def _terminal_inputs(archive: Any) -> _TerminalInputs:
    scalar_contract = {
        "schema": TERMINAL_OBSERVER_SCHEMA,
        "complete": True,
        "observer_only": True,
        "mutation_enabled": False,
        "ground_truth_access": False,
        "coordinate_frame": COORDINATE_FRAME,
        "box_mode": BOX_MODE,
        "corner_semantics": CORNER_SEMANTICS,
        "adapter_mode": "genuine",
    }
    for key, expected in scalar_contract.items():
        if _scalar(archive, key) != expected:
            raise ValueError(f"terminal cache field {key} violates the gate contract")
    scene_id = str(_scalar(archive, "scene_id"))
    if _SCENE_RE.fullmatch(scene_id) is None:
        raise ValueError("terminal cache has invalid CA-1M scene id")
    near_iou = float(_scalar(archive, "near_iou"))
    if not math.isclose(near_iou, DEFAULT_NEAR_IOU, rel_tol=0.0, abs_tol=0.0):
        raise ValueError("terminal cache near_iou differs from the frozen gate route")

    anchors = _array(archive, "anchor_corners")
    anchor_scores = _array(archive, "anchor_scores")
    candidates = _array(archive, "candidate_corners")
    candidate_scores = _array(archive, "candidate_scores")
    support = _array(archive, "candidate_point_count")
    best_anchor = _array(archive, "best_anchor_indices")
    best_iou = _array(archive, "best_anchor_iou")
    best_distance = _array(archive, "best_anchor_center_distance_m")
    near = _array(archive, "near_mask")
    labels = _array(archive, "candidate_labels")
    point_count = int(_scalar(archive, "point_count"))
    verified = bool(_scalar(archive, "materialized_active_verified"))
    anchor_count, candidate_count = len(anchors), len(candidates)
    if (
        anchors.dtype != np.float32
        or anchors.shape != (anchor_count, 8, 3)
        or anchor_scores.dtype != np.float32
        or anchor_scores.shape != (anchor_count,)
        or candidates.dtype != np.float32
        or candidates.shape != (candidate_count, 8, 3)
        or candidate_scores.dtype != np.float32
        or candidate_scores.shape != (candidate_count,)
        or support.dtype != np.int64
        or support.shape != (candidate_count,)
        or best_anchor.dtype != np.int64
        or best_anchor.shape != (candidate_count,)
        or best_iou.dtype != np.float32
        or best_iou.shape != (candidate_count,)
        or best_distance.dtype != np.float32
        or best_distance.shape != (candidate_count,)
        or near.dtype != np.bool_
        or near.shape != (candidate_count,)
        or labels.dtype != np.int64
        or labels.shape != (candidate_count,)
    ):
        raise ValueError("terminal cache array schema is incompatible with the gate")
    if (
        not np.isfinite(anchors).all()
        or not np.isfinite(anchor_scores).all()
        or not np.isfinite(candidates).all()
        or not np.isfinite(candidate_scores).all()
        or not np.isfinite(best_iou).all()
        or not np.isfinite(best_distance).all()
        or np.any(anchor_scores < 0.0)
        or np.any(anchor_scores > 1.0)
        or np.any(candidate_scores < 0.0)
        or np.any(candidate_scores > 1.0)
        or np.any(labels != 0)
        or point_count < 1
        or np.any(support < 0)
        or np.any(support > point_count)
    ):
        raise ValueError("terminal cache contains invalid gate inputs")
    world_aabb(anchors)
    world_aabb(candidates)
    association = associate_terminal_candidates(
        anchor_corners=anchors,
        anchor_scores=anchor_scores,
        candidate_corners=candidates,
        candidate_scores=candidate_scores,
        near_iou=near_iou,
    )
    for name, actual, expected in (
        ("best_anchor_indices", best_anchor, association.best_anchor_indices),
        ("best_anchor_iou", best_iou, association.best_anchor_iou),
        (
            "best_anchor_center_distance_m",
            best_distance,
            association.best_anchor_center_distance_m,
        ),
        ("near_mask", near, association.near_mask),
    ):
        if actual.dtype != expected.dtype or not np.array_equal(actual, expected):
            raise ValueError(f"terminal cache {name} differs from recomputation")
    return _TerminalInputs(
        scene_id=scene_id,
        anchor_corners=_readonly(anchors, np.float32),
        anchor_scores=_readonly(anchor_scores, np.float32),
        candidate_corners=_readonly(candidates, np.float32),
        candidate_scores=_readonly(candidate_scores, np.float32),
        candidate_point_count=_readonly(support, np.int64),
        point_count=point_count,
        best_anchor_indices=_readonly(best_anchor, np.int64),
        best_anchor_iou=_readonly(best_iou, np.float32),
        best_anchor_center_distance_m=_readonly(best_distance, np.float32),
        near_mask=_readonly(near, np.bool_),
        materialized_active_verified=verified,
    )


@dataclass(frozen=True)
class TerminalGateFeatureBatch:
    """Row-aligned GT-free features for terminal candidates near an anchor."""

    scene_id: str
    candidate_rows: np.ndarray
    anchor_indices: np.ndarray
    features: np.ndarray


def build_terminal_gate_features(
    archive: Any,
    *,
    anchor_native_evidence: Any,
    candidate_native_evidence: Any,
) -> TerminalGateFeatureBatch:
    """Build the frozen 40-D feature matrix without reading GT or images."""

    inputs = _terminal_inputs(archive)
    anchor_native = _validate_native_evidence(
        anchor_native_evidence, rows=len(inputs.anchor_corners), label="anchor"
    )
    candidate_native = _validate_native_evidence(
        candidate_native_evidence,
        rows=len(inputs.candidate_corners),
        label="candidate",
    )
    detector_feature = NATIVE_FEATURE_NAMES.index("detector_score")
    if not np.array_equal(
        candidate_native[:, detector_feature].astype(np.float32),
        inputs.candidate_scores,
    ):
        raise ValueError(
            "candidate native detector_score differs from terminal TR3D score"
        )
    candidate_rows = np.flatnonzero(inputs.near_mask).astype(np.int64)
    if not len(candidate_rows):
        return TerminalGateFeatureBatch(
            scene_id=inputs.scene_id,
            candidate_rows=_readonly(candidate_rows, np.int64),
            anchor_indices=_readonly(np.empty((0,), dtype=np.int64), np.int64),
            features=_readonly(
                np.empty((0, len(FEATURE_NAMES)), dtype=np.float32), np.float32
            ),
        )
    anchor_rows = inputs.best_anchor_indices[candidate_rows]
    if np.any(anchor_rows < 0) or np.any(anchor_rows >= len(inputs.anchor_corners)):
        raise ValueError("near terminal candidate has invalid anchor association")

    anchor_boxes = world_aabb(inputs.anchor_corners)
    candidate_boxes = world_aabb(inputs.candidate_corners)
    anchor_extent = anchor_boxes[:, 3:] - anchor_boxes[:, :3]
    candidate_extent = candidate_boxes[:, 3:] - candidate_boxes[:, :3]
    anchor_volume = np.prod(anchor_extent, axis=1)
    candidate_volume = np.prod(candidate_extent, axis=1)
    anchor_diagonal = np.linalg.norm(anchor_extent, axis=1)

    count = len(inputs.candidate_corners)
    global_order = np.lexsort(
        (np.arange(count, dtype=np.int64), -inputs.candidate_scores.astype(np.float64))
    )
    global_rank = np.empty(count, dtype=np.int64)
    global_rank[global_order] = np.arange(count, dtype=np.int64)
    global_denominator = max(count - 1, 1)

    group_rank_fraction = np.zeros(count, dtype=np.float64)
    group_size = np.zeros(count, dtype=np.int64)
    sibling_margin = np.zeros(count, dtype=np.float64)
    for anchor in np.unique(anchor_rows).tolist():
        rows = candidate_rows[anchor_rows == anchor]
        order = np.lexsort((rows, -inputs.candidate_scores[rows].astype(np.float64)))
        ordered = rows[order]
        denominator = max(len(ordered) - 1, 1)
        for rank, row in enumerate(ordered.tolist()):
            group_rank_fraction[row] = rank / denominator
            group_size[row] = len(ordered)
            siblings = rows[rows != row]
            sibling_margin[row] = (
                0.0
                if not len(siblings)
                else float(inputs.candidate_scores[row])
                - float(np.max(inputs.candidate_scores[siblings]))
            )

    relation = np.empty((len(candidate_rows), len(RELATION_FEATURE_NAMES)), np.float64)
    for output_row, (candidate, anchor) in enumerate(
        zip(candidate_rows.tolist(), anchor_rows.tolist())
    ):
        extent_ratio = np.maximum(candidate_extent[candidate], 1.0e-12) / np.maximum(
            anchor_extent[anchor], 1.0e-12
        )
        support = float(inputs.candidate_point_count[candidate])
        volume = max(float(candidate_volume[candidate]), 1.0e-12)
        relation[output_row] = (
            float(inputs.candidate_scores[candidate] - inputs.anchor_scores[anchor]),
            float(inputs.best_anchor_iou[candidate]),
            float(
                inputs.best_anchor_center_distance_m[candidate]
                / max(float(anchor_diagonal[anchor]), 1.0e-12)
            ),
            float(
                math.log(
                    max(float(candidate_volume[candidate]), 1.0e-12)
                    / max(float(anchor_volume[anchor]), 1.0e-12)
                )
            ),
            float(np.linalg.norm(np.log(extent_ratio))),
            math.log1p(support),
            math.log1p(support / volume),
            support / float(inputs.point_count),
            float(global_rank[candidate] / global_denominator),
            float(group_rank_fraction[candidate]),
            math.log1p(int(group_size[candidate])),
            float(sibling_margin[candidate]),
        )
    features = np.concatenate(
        (anchor_native[anchor_rows], candidate_native[candidate_rows], relation), axis=1
    )
    if features.shape != (len(candidate_rows), len(FEATURE_NAMES)):
        raise RuntimeError("terminal gate feature assembly changed shape")
    if not np.isfinite(features).all():
        raise ValueError("terminal gate features contain non-finite values")
    return TerminalGateFeatureBatch(
        scene_id=inputs.scene_id,
        candidate_rows=_readonly(candidate_rows, np.int64),
        anchor_indices=_readonly(anchor_rows, np.int64),
        features=_readonly(features, np.float32),
    )


def _stable_sigmoid(logits: np.ndarray) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float64)
    output = np.empty_like(values)
    positive = values >= 0.0
    output[positive] = 1.0 / (1.0 + np.exp(-np.minimum(values[positive], 700.0)))
    exponential = np.exp(np.maximum(values[~positive], -700.0))
    output[~positive] = exponential / (1.0 + exponential)
    return output


@dataclass(frozen=True)
class LogisticGate:
    weights: np.ndarray
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    bias: float
    probability_threshold: float

    def probabilities(self, features: Any) -> np.ndarray:
        values = np.asarray(features, dtype=np.float64)
        if values.ndim == 1:
            values = values[None]
        if values.ndim != 2 or values.shape[1] != len(FEATURE_NAMES):
            raise ValueError(
                f"terminal gate features must have shape [N,{len(FEATURE_NAMES)}]"
            )
        if not np.isfinite(values).all():
            raise ValueError("terminal gate features must be finite")
        standardized = (values - self.feature_mean) / self.feature_scale
        return np.asarray(
            _stable_sigmoid(standardized @ self.weights + self.bias),
            dtype=np.float64,
        )


def _logistic_gate(payload: Any, name: str) -> LogisticGate:
    if not isinstance(payload, Mapping):
        raise ValueError(f"terminal gate policy lacks {name}")
    expected = len(FEATURE_NAMES)
    weights = np.asarray(payload.get("weights", ()), dtype=np.float64)
    mean = np.asarray(payload.get("feature_mean", ()), dtype=np.float64)
    scale = np.asarray(payload.get("feature_scale", ()), dtype=np.float64)
    bias = float(payload.get("bias", float("nan")))
    threshold = float(payload.get("probability_threshold", float("nan")))
    if (
        weights.shape != (expected,)
        or mean.shape != (expected,)
        or scale.shape != (expected,)
        or not np.isfinite(weights).all()
        or not np.isfinite(mean).all()
        or not np.isfinite(scale).all()
        or np.any(scale <= 0.0)
        or not math.isfinite(bias)
        or not math.isfinite(threshold)
        or not 0.0 < threshold < 1.0
    ):
        raise ValueError(f"terminal gate policy has invalid {name}")
    return LogisticGate(
        weights=_readonly(weights, np.float64),
        feature_mean=_readonly(mean, np.float64),
        feature_scale=_readonly(scale, np.float64),
        bias=bias,
        probability_threshold=threshold,
    )


@dataclass(frozen=True)
class CA1MTerminalGatePolicy:
    path: Path
    sha256: str
    quality25_gate: LogisticGate
    benefit05_gate: LogisticGate
    max_replacements_per_scene: int
    near_iou: float
    training_data_sha256: str
    observer_audit_sha256: str
    training_scene_list_sha256: str
    forbidden_validation_scene_list_sha256: str
    activation_authorized: bool = True

    @classmethod
    def load(cls, path: str | Path) -> "CA1MTerminalGatePolicy":
        source = Path(path)
        if source.is_symlink():
            raise ValueError("terminal gate policy must not be a symlink")
        resolved = source.resolve()
        if (
            not resolved.is_file()
            or resolved.is_symlink()
            or resolved.stat().st_size <= 0
            or resolved.stat().st_mode & 0o222
        ):
            raise ValueError("terminal gate policy must be an immutable regular file")
        try:
            payload = json.loads(resolved.read_text(encoding="utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError("terminal gate policy must be valid JSON") from error
        if not isinstance(payload, Mapping) or payload.get("schema") != POLICY_SCHEMA:
            raise ValueError("unsupported terminal gate policy schema")
        boolean_contract = {
            "complete": True,
            "activation_authorized": True,
            "train_only": True,
            "scene_group_split": True,
            "ground_truth_used_only_for_training": True,
            "candidate_collection_ground_truth_access": False,
            "validation_predictions_used_for_training": False,
            "validation_scene_access": False,
            "one_time_audit_passed": True,
            "geometry_only": True,
            "preserve_anchor_scores": True,
            "preserve_row_order": True,
            "preserve_row_count": True,
            "clip_semantics_unchanged": True,
        }
        for key, expected in boolean_contract.items():
            if payload.get(key) is not expected:
                raise ValueError(f"terminal gate policy violates {key}")
        if (
            payload.get("dataset") != "ca1m"
            or payload.get("observer_schema") != TERMINAL_OBSERVER_SCHEMA
            or tuple(payload.get("native_feature_names", ())) != NATIVE_FEATURE_NAMES
            or tuple(payload.get("feature_names", ())) != FEATURE_NAMES
            or payload.get("feature_schema") != FEATURE_SCHEMA
            or payload.get("quality_target") != QUALITY_TARGET
            or payload.get("benefit_target") != BENEFIT_TARGET
            or payload.get("selection_rule") != SELECTION_RULE
            or tuple(payload.get("gate_train_fold_ids", ()))
            != _EXPECTED_GATE_TRAIN_FOLDS
            or tuple(payload.get("calibration_fold_ids", ()))
            != _EXPECTED_CALIBRATION_FOLDS
            or tuple(payload.get("one_time_audit_fold_ids", ()))
            != _EXPECTED_ONE_TIME_AUDIT_FOLDS
            or int(payload.get("validation_overlap_count", -1)) != 0
        ):
            raise ValueError("terminal gate policy provenance/schema contract failed")
        training = tuple(str(value) for value in payload.get("training_scene_ids", ()))
        forbidden = tuple(
            str(value) for value in payload.get("forbidden_validation_scene_ids", ())
        )
        if (
            not training
            or len(training) != len(set(training))
            or len(forbidden) < 100
            or len(forbidden) != len(set(forbidden))
            or any(_SCENE_RE.fullmatch(value) is None for value in training + forbidden)
            or set(training) & set(forbidden)
        ):
            raise ValueError("terminal gate policy has invalid train/validation scenes")
        hashes: dict[str, str] = {}
        for name in (
            "training_data_sha256",
            "observer_audit_sha256",
            "training_scene_list_sha256",
            "forbidden_validation_scene_list_sha256",
        ):
            digest = str(payload.get(name, ""))
            if _SHA256_RE.fullmatch(digest) is None:
                raise ValueError(f"terminal gate policy has invalid {name}")
            hashes[name] = digest
        near_iou = float(payload.get("near_iou", float("nan")))
        maximum = int(payload.get("max_replacements_per_scene", 0))
        if (
            not math.isfinite(near_iou)
            or not math.isclose(
                near_iou, DEFAULT_NEAR_IOU, rel_tol=0.0, abs_tol=0.0
            )
            or not 1 <= maximum <= 64
        ):
            raise ValueError("terminal gate policy has invalid selection limits")
        return cls(
            path=resolved,
            sha256=_sha256_file(resolved),
            quality25_gate=_logistic_gate(payload.get("quality25_gate"), "quality25_gate"),
            benefit05_gate=_logistic_gate(payload.get("benefit05_gate"), "benefit05_gate"),
            max_replacements_per_scene=maximum,
            near_iou=near_iou,
            training_data_sha256=hashes["training_data_sha256"],
            observer_audit_sha256=hashes["observer_audit_sha256"],
            training_scene_list_sha256=hashes["training_scene_list_sha256"],
            forbidden_validation_scene_list_sha256=hashes[
                "forbidden_validation_scene_list_sha256"
            ],
            activation_authorized=True,
        )


@dataclass(frozen=True)
class TerminalGateSelection:
    """Replacement identities only; this object cannot mutate predictions."""

    schema: str
    scene_id: str
    candidate_rows: np.ndarray
    anchor_indices: np.ndarray
    quality25_probabilities: np.ndarray
    benefit05_probabilities: np.ndarray
    evaluated_count: int
    eligible_count: int
    policy_sha256: str


def select_terminal_replacements(
    archive: Any,
    *,
    anchor_native_evidence: Any,
    candidate_native_evidence: Any,
    policy: CA1MTerminalGatePolicy,
) -> TerminalGateSelection:
    """Select authorized replacement indices without producing predictions."""

    if not isinstance(policy, CA1MTerminalGatePolicy) or not policy.activation_authorized:
        raise ValueError("terminal replacement requires an authorized policy")
    inputs = _terminal_inputs(archive)
    if not inputs.materialized_active_verified:
        raise ValueError("formal terminal selection requires verified active B6 anchors")
    if not math.isclose(
        policy.near_iou, DEFAULT_NEAR_IOU, rel_tol=0.0, abs_tol=0.0
    ):
        raise ValueError("terminal policy near_iou differs from the cache route")
    batch = build_terminal_gate_features(
        archive,
        anchor_native_evidence=anchor_native_evidence,
        candidate_native_evidence=candidate_native_evidence,
    )
    quality = policy.quality25_gate.probabilities(batch.features)
    benefit = policy.benefit05_gate.probabilities(batch.features)
    eligible = (
        (quality >= policy.quality25_gate.probability_threshold)
        & (benefit >= policy.benefit05_gate.probability_threshold)
    )
    eligible_rows = np.flatnonzero(eligible)
    chosen: list[int] = []
    for anchor in np.unique(batch.anchor_indices[eligible_rows]).tolist():
        local = eligible_rows[batch.anchor_indices[eligible_rows] == anchor]
        best = max(
            local.tolist(),
            key=lambda row: (
                float(benefit[row]),
                float(quality[row]),
                float(inputs.candidate_scores[int(batch.candidate_rows[row])]),
                -int(batch.candidate_rows[row]),
            ),
        )
        chosen.append(best)
    chosen.sort(
        key=lambda row: (
            -float(benefit[row]),
            -float(quality[row]),
            -float(inputs.candidate_scores[int(batch.candidate_rows[row])]),
            int(batch.anchor_indices[row]),
            int(batch.candidate_rows[row]),
        )
    )
    chosen = chosen[: policy.max_replacements_per_scene]
    chosen.sort(
        key=lambda row: (
            int(batch.anchor_indices[row]), int(batch.candidate_rows[row])
        )
    )
    selection = np.asarray(chosen, dtype=np.int64)
    return TerminalGateSelection(
        schema=SELECTION_SCHEMA,
        scene_id=batch.scene_id,
        candidate_rows=_readonly(batch.candidate_rows[selection], np.int64),
        anchor_indices=_readonly(batch.anchor_indices[selection], np.int64),
        quality25_probabilities=_readonly(quality[selection], np.float64),
        benefit05_probabilities=_readonly(benefit[selection], np.float64),
        evaluated_count=len(batch.candidate_rows),
        eligible_count=int(np.count_nonzero(eligible)),
        policy_sha256=policy.sha256,
    )


__all__ = [
    "BENEFIT_TARGET",
    "CA1MTerminalGatePolicy",
    "FEATURE_NAMES",
    "FEATURE_SCHEMA",
    "LogisticGate",
    "NATIVE_FEATURE_NAMES",
    "POLICY_SCHEMA",
    "QUALITY_TARGET",
    "RELATION_FEATURE_NAMES",
    "SELECTION_RULE",
    "SELECTION_SCHEMA",
    "TerminalGateFeatureBatch",
    "TerminalGateSelection",
    "build_terminal_gate_features",
    "select_terminal_replacements",
]
