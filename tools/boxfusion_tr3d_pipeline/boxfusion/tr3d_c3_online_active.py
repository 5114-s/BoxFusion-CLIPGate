"""Fail-closed online C3 candidate activation.

The online identity observer owns candidate/evidence collection.  This module
is the only inference-side component allowed to turn those immutable rows into
an append-only prediction branch.  Activation requires a train-only policy
checkpoint; validation-tuned or incomplete checkpoints are rejected.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping

import numpy as np

from .tr3d_c3_online_identity import PARENT_SCORE_ROUTE
from .tr3d_c3_online_identity import ROUTE as IDENTITY_ROUTE
from .tr3d_c3_online_identity import SCHEMA as IDENTITY_SCHEMA
from .tr3d_c3_online_identity import prediction_state_sha256
from .tr3d_c2_maskrgbd_cache import sha256_file
from .tr3d_residual_cache import (
    load_tr3d_residual_cache,
    tr3d_residual_cache_path,
)


POLICY_SCHEMA = "boxfusion.tr3d_c3_source_gate.v1"
RESULT_SCHEMA = "boxfusion.tr3d_c3_online_active_result.v1"
FEATURE_NAMES = (
    "source_rank",
    "c1_depth_dino_track_score",
    "projected_view_count",
    "matched_view_count",
    "strong_view_count",
    "log1p_total_component_points",
    "mean_strong_inside_expanded",
    "max_evidence_score",
    "matched_over_projected",
    "strong_over_matched",
)


def _readonly(value: Any, dtype: np.dtype) -> np.ndarray:
    result = np.ascontiguousarray(np.asarray(value, dtype=dtype))
    result.setflags(write=False)
    return result


def _sigmoid(value: float) -> float:
    if value >= 0.0:
        inverse = math.exp(-value)
        return 1.0 / (1.0 + inverse)
    exponent = math.exp(value)
    return exponent / (1.0 + exponent)


def _aabb_iou(corners: np.ndarray, anchors: np.ndarray) -> np.ndarray:
    if len(anchors) == 0:
        return np.empty((0,), dtype=np.float64)
    candidate_min = np.min(corners, axis=0).astype(np.float64)
    candidate_max = np.max(corners, axis=0).astype(np.float64)
    anchor_min = np.min(anchors, axis=1).astype(np.float64)
    anchor_max = np.max(anchors, axis=1).astype(np.float64)
    intersection = np.maximum(
        np.minimum(candidate_max, anchor_max)
        - np.maximum(candidate_min, anchor_min),
        0.0,
    )
    intersection_volume = np.prod(intersection, axis=1)
    candidate_volume = float(np.prod(candidate_max - candidate_min))
    anchor_volume = np.prod(anchor_max - anchor_min, axis=1)
    union = candidate_volume + anchor_volume - intersection_volume
    return np.divide(
        intersection_volume,
        union,
        out=np.zeros_like(intersection_volume),
        where=union > 0.0,
    )


@dataclass(frozen=True)
class C3SourceGatePolicy:
    path: Path
    sha256: str
    weights: np.ndarray
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    bias: float
    probability_threshold: float
    output_score_min: float
    output_score_max: float
    max_candidates_per_scene: int
    max_anchor_iou: float
    training_scene_list_sha256: str
    training_data_sha256: str
    route: str

    @classmethod
    def load(cls, path: str | Path) -> "C3SourceGatePolicy":
        source = Path(path).resolve()
        if (
            not source.is_file()
            or source.is_symlink()
            or source.stat().st_mode & 0o022
        ):
            raise ValueError(
                "C3 source-gate checkpoint must be a non-writable regular file"
            )
        payload = json.loads(source.read_text(encoding="utf-8"))
        forbidden = set(payload.get("forbidden_validation_scene_ids", ()))
        training = set(payload.get("training_scene_ids", ()))
        required_bools = {
            "complete": True,
            "activation_authorized": True,
            "train_only": True,
            "scene_group_oof": True,
            "ground_truth_used_only_for_training": True,
            "validation_predictions_used_for_training": False,
        }
        if payload.get("schema") != POLICY_SCHEMA:
            raise ValueError("unsupported C3 source-gate checkpoint schema")
        for key, expected in required_bools.items():
            if bool(payload.get(key)) is not expected:
                raise ValueError(f"C3 source-gate checkpoint violates {key}")
        if (
            payload.get("dataset") != "scannet"
            or payload.get("route") not in {IDENTITY_ROUTE, PARENT_SCORE_ROUTE}
            or tuple(payload.get("feature_names", ())) != FEATURE_NAMES
            or not training
            or training & forbidden
            or int(payload.get("validation_overlap_count", -1)) != 0
        ):
            raise ValueError("C3 source-gate training provenance is invalid")
        if len(forbidden) < 100:
            raise ValueError(
                "C3 source-gate checkpoint must enumerate the forbidden "
                "validation partition"
            )

        count = len(FEATURE_NAMES)
        weights = _readonly(payload.get("weights", ()), np.float64)
        mean = _readonly(payload.get("feature_mean", ()), np.float64)
        scale = _readonly(payload.get("feature_scale", ()), np.float64)
        if weights.shape != (count,) or mean.shape != (count,) or scale.shape != (count,):
            raise ValueError("C3 source-gate parameter shape mismatch")
        if (
            not np.isfinite(weights).all()
            or not np.isfinite(mean).all()
            or not np.isfinite(scale).all()
            or np.any(scale <= 0.0)
        ):
            raise ValueError("C3 source-gate parameters must be finite")

        bias = float(payload.get("bias", float("nan")))
        threshold = float(payload.get("probability_threshold", float("nan")))
        score_min = float(payload.get("output_score_min", float("nan")))
        score_max = float(payload.get("output_score_max", float("nan")))
        max_candidates = int(payload.get("max_candidates_per_scene", 0))
        max_anchor_iou = float(payload.get("max_anchor_iou", float("nan")))
        if (
            not all(
                math.isfinite(value)
                for value in (bias, threshold, score_min, score_max, max_anchor_iou)
            )
            or not 0.0 < threshold < 1.0
            or not 0.0 < score_min < score_max < 1.0
            or not 1 <= max_candidates <= 64
            or not 0.0 <= max_anchor_iou <= 0.5
        ):
            raise ValueError("C3 source-gate scalar policy is invalid")

        for name in ("training_scene_list_sha256", "training_data_sha256"):
            digest = str(payload.get(name, ""))
            if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
                raise ValueError(f"C3 source-gate checkpoint has invalid {name}")
        return cls(
            path=source,
            sha256=sha256_file(source),
            weights=weights,
            feature_mean=mean,
            feature_scale=scale,
            bias=bias,
            probability_threshold=threshold,
            output_score_min=score_min,
            output_score_max=score_max,
            max_candidates_per_scene=max_candidates,
            max_anchor_iou=max_anchor_iou,
            training_scene_list_sha256=str(payload["training_scene_list_sha256"]),
            training_data_sha256=str(payload["training_data_sha256"]),
            route=str(payload["route"]),
        )

    def probability(self, features: np.ndarray) -> float:
        values = np.asarray(features, dtype=np.float64)
        if values.shape != (len(FEATURE_NAMES),) or not np.isfinite(values).all():
            raise ValueError("C3 source-gate feature vector is invalid")
        standardized = (values - self.feature_mean) / self.feature_scale
        return _sigmoid(float(np.dot(standardized, self.weights) + self.bias))


def candidate_features(candidate: Mapping[str, Any]) -> np.ndarray:
    projected = float(candidate.get("projected_view_count", 0))
    matched = float(candidate.get("matched_view_count", 0))
    strong = float(candidate.get("strong_view_count", 0))
    component_points = float(candidate.get("total_component_points", 0))
    values = np.asarray(
        (
            float(candidate.get("source_rank", float("nan"))),
            float(candidate.get("c1_depth_dino_track_score", float("nan"))),
            projected,
            matched,
            strong,
            math.log1p(max(component_points, 0.0)),
            float(candidate.get("mean_strong_inside_expanded", float("nan"))),
            float(candidate.get("max_evidence_score", float("nan"))),
            matched / projected if projected > 0.0 else 0.0,
            strong / matched if matched > 0.0 else 0.0,
        ),
        dtype=np.float64,
    )
    if not np.isfinite(values).all():
        raise ValueError("C3 candidate contains non-finite gate features")
    return values


def write_active_summary_create_only(
    path: str | Path, summary: Mapping[str, Any]
) -> Path:
    """Persist an immutable per-scene active decision report."""

    target = Path(path).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(summary, indent=2, sort_keys=True) + "\n").encode()
    temporary: str | None = None
    try:
        descriptor, temporary = tempfile.mkstemp(
            dir=target.parent, prefix=f".{target.name}.", suffix=".tmp"
        )
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, target)
        target.chmod(0o444)
    except FileExistsError as error:
        raise FileExistsError(
            f"refusing existing C3 online active report: {target}"
        ) from error
    finally:
        if temporary is not None:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass
    return target


@dataclass(frozen=True)
class C3ActiveResult:
    corners: np.ndarray
    scores: np.ndarray
    summary: Mapping[str, Any]


class C3OnlineActiveAppender:
    """Append train-authorized C3 rows without modifying existing anchors."""

    def __init__(self, policy: C3SourceGatePolicy) -> None:
        self.policy = policy

    @classmethod
    def from_checkpoint(cls, path: str | Path) -> "C3OnlineActiveAppender":
        return cls(C3SourceGatePolicy.load(path))

    def apply(
        self,
        *,
        scene_id: str,
        identity_summary: Mapping[str, Any],
        anchor_corners: Any,
        anchor_scores: Any,
    ) -> C3ActiveResult:
        if (
            identity_summary.get("schema") != IDENTITY_SCHEMA
            or not identity_summary.get("complete")
            or not identity_summary.get("observer_only")
            or identity_summary.get("mutation_enabled")
            or int(identity_summary.get("applied_count", -1)) != 0
            or identity_summary.get("scene_id") != scene_id
            or identity_summary.get("route") != self.policy.route
            or identity_summary.get("ground_truth_access")
            or identity_summary.get("clip_access")
            or identity_summary.get("candidate_generation_is_live") is not False
        ):
            raise ValueError("C3 online identity summary failed the active contract")

        anchors = np.ascontiguousarray(np.asarray(anchor_corners, dtype=np.float32))
        scores = np.ascontiguousarray(np.asarray(anchor_scores, dtype=np.float32))
        if anchors.size == 0:
            anchors = np.empty((0, 8, 3), dtype=np.float32)
        if anchors.ndim != 3 or anchors.shape[1:] != (8, 3):
            raise ValueError("C3 active anchors must have shape [N,8,3]")
        if scores.shape != (len(anchors),) or not np.isfinite(anchors).all() or not np.isfinite(scores).all():
            raise ValueError("C3 active anchor state is malformed")
        if len(scores) == 0 or float(np.min(scores)) <= 0.0:
            raise ValueError("C3 active requires positive anchor scores")
        anchor_state = prediction_state_sha256(anchors, scores)
        if (
            identity_summary.get("prediction_state_before_sha256")
            != anchor_state
            or identity_summary.get("prediction_state_after_sha256")
            != anchor_state
            or int(identity_summary.get("prediction_count", -1))
            != len(anchors)
        ):
            raise ValueError("C3 active identity/anchor prediction mismatch")

        parent_path = Path(str(identity_summary.get("parent_cache", ""))).resolve()
        expected_parent = tr3d_residual_cache_path(
            parent_path.parents[1], scene_id, "p100"
        ).resolve()
        if parent_path != expected_parent or sha256_file(parent_path) != identity_summary.get("parent_cache_sha256"):
            raise ValueError("C3 active parent-cache lineage mismatch")
        with np.load(parent_path, allow_pickle=False) as archive:
            checkpoint_sha = str(np.asarray(archive["checkpoint_sha256"]).item())
            config_sha = str(np.asarray(archive["config_sha256"]).item())
        parent = load_tr3d_residual_cache(
            parent_path,
            expected_scene_id=scene_id,
            expected_prefix_id="p100",
            expected_checkpoint_sha256=checkpoint_sha,
            expected_config_sha256=config_sha,
        )

        selected: list[tuple[float, int, int, np.ndarray, np.ndarray, float]] = []
        reject_counts = {
            "online_gate": 0,
            "probability": 0,
            "anchor_iou": 0,
            "duplicate_identity": 0,
        }
        seen: set[str] = set()
        for candidate in identity_summary.get("candidates", ()):
            identity = str(candidate.get("identity_key", ""))
            if identity in seen:
                reject_counts["duplicate_identity"] += 1
                continue
            seen.add(identity)
            if not bool(candidate.get("online_yoloe_mask2_depth")):
                reject_counts["online_gate"] += 1
                continue
            parent_row = int(candidate.get("parent_row", -1))
            proposal_id = int(candidate.get("proposal_id", -1))
            if (
                parent_row < 0
                or parent_row >= len(parent.proposal_ids)
                or int(parent.proposal_ids[parent_row]) != proposal_id
            ):
                raise ValueError("C3 active candidate/parent identity mismatch")
            features = candidate_features(candidate)
            probability = self.policy.probability(features)
            if probability < self.policy.probability_threshold:
                reject_counts["probability"] += 1
                continue
            corners = np.ascontiguousarray(parent.corners_world[parent_row], dtype=np.float32)
            maximum_iou = float(np.max(_aabb_iou(corners, anchors), initial=0.0))
            if maximum_iou > self.policy.max_anchor_iou:
                reject_counts["anchor_iou"] += 1
                continue
            selected.append(
                (
                    probability,
                    int(candidate.get("source_rank", 2**31 - 1)),
                    proposal_id,
                    corners,
                    features,
                    maximum_iou,
                )
            )

        selected.sort(key=lambda row: (-row[0], row[1], row[2]))
        selected = selected[: self.policy.max_candidates_per_scene]
        anchor_floor = float(np.min(scores))
        score_ceiling = min(
            self.policy.output_score_max,
            float(np.nextafter(np.float32(anchor_floor), np.float32(0.0))),
        )
        if score_ceiling <= self.policy.output_score_min:
            raise ValueError("C3 active score band is not below the anchor floor")

        candidate_corners: list[np.ndarray] = []
        candidate_scores: list[float] = []
        accepted: list[dict[str, Any]] = []
        denominator = max(1.0 - self.policy.probability_threshold, np.finfo(float).eps)
        for probability, source_rank, proposal_id, corners, features, maximum_iou in selected:
            fraction = min(
                max((probability - self.policy.probability_threshold) / denominator, 0.0),
                1.0,
            )
            score = float(
                np.float32(
                    self.policy.output_score_min
                    + fraction * (score_ceiling - self.policy.output_score_min)
                )
            )
            candidate_corners.append(corners)
            candidate_scores.append(score)
            accepted.append(
                {
                    "proposal_id": proposal_id,
                    "source_rank": source_rank,
                    "probability": probability,
                    "score": score,
                    "maximum_anchor_iou": maximum_iou,
                    "features": {
                        name: float(features[index])
                        for index, name in enumerate(FEATURE_NAMES)
                    },
                }
            )

        appended_corners = (
            np.stack(candidate_corners).astype(np.float32, copy=False)
            if candidate_corners
            else np.empty((0, 8, 3), dtype=np.float32)
        )
        appended_scores = np.asarray(candidate_scores, dtype=np.float32)
        output_corners = np.ascontiguousarray(
            np.concatenate((anchors, appended_corners), axis=0), dtype=np.float32
        )
        output_scores = np.ascontiguousarray(
            np.concatenate((scores, appended_scores), axis=0), dtype=np.float32
        )
        summary = {
            "schema": RESULT_SCHEMA,
            "complete": True,
            "scene_id": scene_id,
            "active": True,
            "mutation_enabled": True,
            "append_only": True,
            "ground_truth_access": False,
            "clip_access": False,
            "clip_semantics_unchanged": True,
            "candidate_label": 0,
            "route": self.policy.route,
            "policy_checkpoint": str(self.policy.path),
            "policy_checkpoint_sha256": self.policy.sha256,
            "training_scene_list_sha256": self.policy.training_scene_list_sha256,
            "training_data_sha256": self.policy.training_data_sha256,
            "anchor_count": len(anchors),
            "evaluated_count": len(identity_summary.get("candidates", ())),
            "online_gate_count": int(identity_summary.get("online_selected_count", 0)),
            "applied_count": len(appended_corners),
            "output_count": len(output_corners),
            "anchor_score_floor": anchor_floor,
            "candidate_score_ceiling": score_ceiling,
            "reject_counts": reject_counts,
            "accepted": accepted,
        }
        return C3ActiveResult(output_corners, output_scores, summary)


__all__ = [
    "C3ActiveResult",
    "C3OnlineActiveAppender",
    "C3SourceGatePolicy",
    "FEATURE_NAMES",
    "POLICY_SCHEMA",
    "RESULT_SCHEMA",
    "candidate_features",
    "write_active_summary_create_only",
]
