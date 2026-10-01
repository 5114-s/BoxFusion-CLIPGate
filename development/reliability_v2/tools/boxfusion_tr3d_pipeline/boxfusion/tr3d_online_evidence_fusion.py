"""Train-authorized, low-overhead keep/replace/append decision head.

The head deliberately reuses evidence already produced by the online C3
observer.  It performs no image-backbone, point-backbone, mesh, CLIP, or GT
work at inference time.  A policy may only be activated when it was fitted on
the disjoint ScanNet training partition and passed scene-grouped OOF gates.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .tr3d_c2_maskrgbd_cache import sha256_file
from .tr3d_c3_online_active import FEATURE_NAMES as SOURCE_FEATURE_NAMES
from .tr3d_c3_online_active import candidate_features
from .tr3d_c3_online_identity import PARENT_SCORE_ROUTE, SCHEMA as IDENTITY_SCHEMA


POLICY_SCHEMA = "boxfusion.tr3d_online_evidence_fusion_policy.v1"
RESULT_SCHEMA = "boxfusion.tr3d_online_evidence_fusion_result.v1"
FEATURE_NAMES = SOURCE_FEATURE_NAMES + (
    "nearest_anchor_iou",
    "center_distance_over_anchor_diagonal",
    "log_candidate_over_anchor_volume",
    "candidate_tr3d_score",
    "nearest_anchor_score",
    "candidate_minus_anchor_score",
)


def _sigmoid(value: float) -> float:
    value = float(np.clip(value, -40.0, 40.0))
    return 1.0 / (1.0 + math.exp(-value))


def aabb_iou(corners: np.ndarray, anchors: np.ndarray) -> np.ndarray:
    candidate = np.asarray(corners, dtype=np.float64)
    rows = np.asarray(anchors, dtype=np.float64)
    if candidate.shape != (8, 3) or rows.ndim != 3 or rows.shape[1:] != (8, 3):
        raise ValueError("candidate/anchor corners must be [8,3]/[N,8,3]")
    if not len(rows):
        return np.empty((0,), dtype=np.float64)
    cmin, cmax = candidate.min(axis=0), candidate.max(axis=0)
    amin, amax = rows.min(axis=1), rows.max(axis=1)
    extent = np.maximum(np.minimum(cmax, amax) - np.maximum(cmin, amin), 0.0)
    intersection = np.prod(extent, axis=1)
    cvolume = float(np.prod(cmax - cmin))
    avolume = np.prod(amax - amin, axis=1)
    union = cvolume + avolume - intersection
    return np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0.0)


def fusion_features(
    candidate: Mapping[str, Any],
    candidate_corners: np.ndarray,
    candidate_score: float,
    anchor_corners: np.ndarray,
    anchor_score: float,
    nearest_iou: float,
) -> np.ndarray:
    c = np.asarray(candidate_corners, dtype=np.float64)
    a = np.asarray(anchor_corners, dtype=np.float64)
    cmin, cmax = c.min(axis=0), c.max(axis=0)
    amin, amax = a.min(axis=0), a.max(axis=0)
    ccenter, acenter = (cmin + cmax) * 0.5, (amin + amax) * 0.5
    cvolume = max(float(np.prod(cmax - cmin)), 1e-12)
    avolume = max(float(np.prod(amax - amin)), 1e-12)
    diagonal = max(float(np.linalg.norm(amax - amin)), 1e-12)
    extra = np.asarray(
        (
            nearest_iou,
            float(np.linalg.norm(ccenter - acenter) / diagonal),
            math.log(cvolume / avolume),
            candidate_score,
            anchor_score,
            candidate_score - anchor_score,
        ),
        dtype=np.float64,
    )
    values = np.concatenate((candidate_features(candidate), extra))
    if values.shape != (len(FEATURE_NAMES),) or not np.isfinite(values).all():
        raise ValueError("online evidence-fusion features are invalid")
    return values


@dataclass(frozen=True)
class LinearGate:
    weights: np.ndarray
    mean: np.ndarray
    scale: np.ndarray
    bias: float
    threshold: float

    def probability(self, features: np.ndarray) -> float:
        values = np.asarray(features, dtype=np.float64)
        return _sigmoid(float(((values - self.mean) / self.scale) @ self.weights + self.bias))


@dataclass(frozen=True)
class EvidenceFusionPolicy:
    path: Path
    sha256: str
    append: LinearGate
    replace: LinearGate
    max_append_anchor_iou: float
    min_replace_anchor_iou: float
    max_replace_anchor_iou: float
    output_score_min: float
    output_score_max: float
    max_appends_per_scene: int
    route: str
    activation_authorized: bool

    @staticmethod
    def _gate(payload: Mapping[str, Any], name: str) -> LinearGate:
        row = payload.get(name)
        if not isinstance(row, Mapping):
            raise ValueError(f"missing {name} gate")
        count = len(FEATURE_NAMES)
        weights = np.asarray(row.get("weights", ()), dtype=np.float64)
        mean = np.asarray(row.get("feature_mean", ()), dtype=np.float64)
        scale = np.asarray(row.get("feature_scale", ()), dtype=np.float64)
        threshold = float(row.get("probability_threshold", float("nan")))
        bias = float(row.get("bias", float("nan")))
        if (
            weights.shape != (count,) or mean.shape != (count,) or scale.shape != (count,)
            or not np.isfinite(weights).all() or not np.isfinite(mean).all()
            or not np.isfinite(scale).all() or np.any(scale <= 0.0)
            or not math.isfinite(bias) or not 0.0 < threshold < 1.0
        ):
            raise ValueError(f"invalid {name} gate")
        for value in (weights, mean, scale):
            value.setflags(write=False)
        return LinearGate(weights, mean, scale, bias, threshold)

    @classmethod
    def load(
        cls, path: str | Path, *, require_authorized: bool = True
    ) -> "EvidenceFusionPolicy":
        source = Path(path).resolve()
        if not source.is_file() or source.is_symlink() or source.stat().st_mode & 0o022:
            raise ValueError("evidence-fusion policy must be immutable")
        payload = json.loads(source.read_text(encoding="utf-8"))
        required = {
            "complete": True, "train_only": True,
            "scene_group_oof": True, "validation_predictions_used_for_training": False,
            "ground_truth_used_only_for_training": True,
        }
        if payload.get("schema") != POLICY_SCHEMA:
            raise ValueError("unsupported evidence-fusion policy schema")
        for key, expected in required.items():
            if bool(payload.get(key)) is not expected:
                raise ValueError(f"policy violates {key}")
        authorized = bool(payload.get("activation_authorized", False))
        if require_authorized and not authorized:
            raise ValueError("evidence-fusion policy is observer-only")
        if tuple(payload.get("feature_names", ())) != FEATURE_NAMES:
            raise ValueError("evidence-fusion feature schema mismatch")
        if payload.get("route") != PARENT_SCORE_ROUTE:
            raise ValueError("evidence-fusion policy route mismatch")
        training = set(payload.get("training_scene_ids", ()))
        forbidden = set(payload.get("forbidden_validation_scene_ids", ()))
        if not training or len(forbidden) < 100 or training & forbidden:
            raise ValueError("evidence-fusion policy provenance is invalid")
        scalars = {
            "max_append_anchor_iou": (0.0, 0.5),
            "min_replace_anchor_iou": (0.0, 1.0),
            "max_replace_anchor_iou": (0.0, 1.0),
            "output_score_min": (0.0, 1.0),
            "output_score_max": (0.0, 1.0),
        }
        parsed: dict[str, float] = {}
        for name, bounds in scalars.items():
            value = float(payload.get(name, float("nan")))
            if not math.isfinite(value) or not bounds[0] <= value <= bounds[1]:
                raise ValueError(f"invalid policy scalar {name}")
            parsed[name] = value
        if parsed["min_replace_anchor_iou"] > parsed["max_replace_anchor_iou"]:
            raise ValueError("invalid replacement IoU band")
        if not parsed["output_score_min"] < parsed["output_score_max"]:
            raise ValueError("invalid append score band")
        maximum = int(payload.get("max_appends_per_scene", 0))
        if not 1 <= maximum <= 32:
            raise ValueError("invalid max_appends_per_scene")
        return cls(
            source, sha256_file(source), cls._gate(payload, "append_gate"),
            cls._gate(payload, "replace_gate"),
            parsed["max_append_anchor_iou"], parsed["min_replace_anchor_iou"],
            parsed["max_replace_anchor_iou"], parsed["output_score_min"],
            parsed["output_score_max"], maximum, str(payload["route"]), authorized,
        )


@dataclass(frozen=True)
class EvidenceFusionResult:
    corners: np.ndarray
    scores: np.ndarray
    summary: Mapping[str, Any]


class OnlineEvidenceFusion:
    """Deterministic three-way decision head; observer mode is byte-identical."""

    def __init__(self, policy: EvidenceFusionPolicy, *, active: bool = False) -> None:
        self.policy = policy
        self.active = bool(active)

    @classmethod
    def from_checkpoint(cls, path: str | Path, *, active: bool = False) -> "OnlineEvidenceFusion":
        return cls(
            EvidenceFusionPolicy.load(path, require_authorized=active),
            active=active,
        )

    def apply(
        self,
        *,
        scene_id: str,
        identity_summary: Mapping[str, Any],
        parent_corners: np.ndarray,
        parent_scores: np.ndarray,
        anchor_corners: np.ndarray,
        anchor_scores: np.ndarray,
    ) -> EvidenceFusionResult:
        if (
            identity_summary.get("schema") != IDENTITY_SCHEMA
            or identity_summary.get("scene_id") != scene_id
            or identity_summary.get("route") != self.policy.route
            or not identity_summary.get("complete")
            or not identity_summary.get("observer_only")
            or identity_summary.get("ground_truth_access")
            or identity_summary.get("clip_access")
        ):
            raise ValueError("identity summary violates evidence-fusion contract")
        anchors = np.ascontiguousarray(anchor_corners, dtype=np.float32)
        scores = np.ascontiguousarray(anchor_scores, dtype=np.float32)
        parent_geometry = np.asarray(parent_corners, dtype=np.float32)
        parent_confidence = np.asarray(parent_scores, dtype=np.float32)
        if anchors.ndim != 3 or anchors.shape[1:] != (8, 3) or scores.shape != (len(anchors),):
            raise ValueError("invalid anchor prediction state")

        proposals: list[dict[str, Any]] = []
        for candidate in identity_summary.get("candidates", ()):
            if not bool(candidate.get("online_yoloe_mask2_depth")):
                continue
            row = int(candidate.get("parent_row", -1))
            if row < 0 or row >= len(parent_geometry):
                raise ValueError("candidate parent row outside cache")
            corners = parent_geometry[row]
            overlaps = aabb_iou(corners, anchors)
            nearest = int(np.argmax(overlaps)) if len(overlaps) else -1
            nearest_iou = float(overlaps[nearest]) if nearest >= 0 else 0.0
            nearest_corners = anchors[nearest] if nearest >= 0 else corners
            nearest_score = float(scores[nearest]) if nearest >= 0 else 0.0
            features = fusion_features(
                candidate, corners, float(parent_confidence[row]), nearest_corners,
                nearest_score, nearest_iou,
            )
            append_probability = self.policy.append.probability(features)
            replace_probability = self.policy.replace.probability(features)
            if (
                nearest_iou <= self.policy.max_append_anchor_iou
                and append_probability >= self.policy.append.threshold
            ):
                action = "append"
            elif (
                nearest >= 0
                and self.policy.min_replace_anchor_iou <= nearest_iou <= self.policy.max_replace_anchor_iou
                and replace_probability >= self.policy.replace.threshold
            ):
                action = "replace"
            else:
                action = "keep"
            proposals.append({
                "proposal_id": int(candidate.get("proposal_id", -1)),
                "parent_row": row, "nearest_anchor": nearest,
                "nearest_anchor_iou": nearest_iou,
                "append_probability": append_probability,
                "replace_probability": replace_probability,
                "action": action, "corners": corners,
            })

        # One candidate may replace an anchor; choose deterministically.
        replace_best: dict[int, dict[str, Any]] = {}
        for row in proposals:
            if row["action"] != "replace":
                continue
            anchor = int(row["nearest_anchor"])
            current = replace_best.get(anchor)
            key = (row["replace_probability"], -row["proposal_id"])
            if current is None or key > (current["replace_probability"], -current["proposal_id"]):
                replace_best[anchor] = row
        appends = sorted(
            (row for row in proposals if row["action"] == "append"),
            key=lambda row: (-row["append_probability"], row["proposal_id"]),
        )[: self.policy.max_appends_per_scene]

        output_corners = np.array(anchors, copy=True)
        output_scores = np.array(scores, copy=True)
        if self.active:
            for anchor, row in replace_best.items():
                output_corners[anchor] = row["corners"]
            if appends:
                floor = float(np.min(scores))
                ceiling = min(self.policy.output_score_max, float(np.nextafter(np.float32(floor), np.float32(0.0))))
                if ceiling <= self.policy.output_score_min:
                    raise ValueError("append score band is not below anchors")
                geometry, confidence = [], []
                denominator = max(1.0 - self.policy.append.threshold, 1e-12)
                for row in appends:
                    fraction = np.clip((row["append_probability"] - self.policy.append.threshold) / denominator, 0.0, 1.0)
                    geometry.append(row["corners"])
                    confidence.append(self.policy.output_score_min + fraction * (ceiling - self.policy.output_score_min))
                output_corners = np.concatenate((output_corners, np.asarray(geometry, dtype=np.float32)))
                output_scores = np.concatenate((output_scores, np.asarray(confidence, dtype=np.float32)))

        decisions = []
        for row in proposals:
            action = row["action"]
            if action == "replace" and replace_best.get(int(row["nearest_anchor"])) is not row:
                action = "keep_conflict"
            if action == "append" and row not in appends:
                action = "keep_budget"
            decision = {
                key: value for key, value in row.items() if key != "corners"
            }
            decision["action"] = action
            decisions.append(decision)
        summary = {
            "schema": RESULT_SCHEMA, "complete": True, "scene_id": scene_id,
            "active": self.active, "observer_only": not self.active,
            "mutation_enabled": self.active,
            "clip_access": False, "clip_semantics_unchanged": True,
            "extra_backbone_forwards": 0, "mesh_access": False,
            "anchor_count": len(anchors), "candidate_count": len(proposals),
            "replace_count": len(replace_best) if self.active else 0,
            "append_count": len(appends) if self.active else 0,
            "output_count": len(output_corners), "policy_sha256": self.policy.sha256,
            "decisions": decisions,
        }
        return EvidenceFusionResult(output_corners, output_scores, summary)


__all__ = [
    "EvidenceFusionPolicy", "EvidenceFusionResult", "FEATURE_NAMES",
    "OnlineEvidenceFusion", "POLICY_SCHEMA", "RESULT_SCHEMA", "aabb_iou",
    "fusion_features",
]
