"""CA-1M-only contracts for incremental/L6 novelty append.

This module intentionally contains no TR3D runner, prediction writer, or GT
loader.  It freezes the parts of the ScanNet L6 method that are safe to port:

* a train-only novelty probability;
* deterministic source-aware ranking; and
* distinct positive float32 scores below every sealed anchor score.

The policy schema and provenance are CA-1M specific.  A ScanNet policy, an old
CA terminal cache, or a naked TR3D checkpoint cannot satisfy this contract.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

import numpy as np


POLICY_SCHEMA = "boxfusion.ca1m_incremental_l6_policy.v1"
DATASET_SCHEMA = "boxfusion.ca1m_incremental_l6_dataset.v1"
OBSERVER_SCHEMA = "boxfusion.ca1m_incremental_l6_observer.v1"
ACTIVE_SCHEMA = "boxfusion.ca1m_incremental_l6_active.v1"
UPSTREAM_ROUTE = "ca1m_final_base_b6_v2_terminal_benefit_v2"
CA_TR3D_BINDING_SCHEMA = "boxfusion.tr3d.ca1m_checkpoint_binding.v1"
CA_TR3D_BINDING_SHA256 = (
    "19b8c3d12de8dd8d3ffff1413c6c6003a5ccb1a10cf213b972ebd43fa9db5043"
)
CA_TR3D_CHECKPOINT_SHA256 = (
    "d3ba6cc22f0a1a11ab47e55ccdd21c2ef4a84efaf3c6359b7e8231a6c8d3b4a7"
)

SPLIT_SHA256 = {
    "weights_train": "7f0a22c660f7f9bd44137f5049c694393e038f5ab97ec55053443bfc00967478",
    "threshold_dev": "9c886ca85ba599881797b25a49d2fc72dd136d255a245a09fe1cf17cbce735a7",
    "locked_internal_check": "d6238bae873c98737858ac3a84c0706091fa9a91113321ac9736a8d64de6d6b6",
    "train100": "35321e9942dc5d512db2952b9ca6228b1291127e0c13fd92aa458f2d7eb7f9fd",
    "official_validation_forbidden": "bd5f3fc66168114048a1b12addc45949c8f54f9c016b921bacfb6fe9e3e7dc2f",
}
SPLIT_FOLDS = {
    "weights_train": (2, 3, 4),
    "threshold_dev": (0,),
    "locked_internal_check": (1,),
}
SPLIT_COUNTS = {
    "weights_train": 60,
    "threshold_dev": 20,
    "locked_internal_check": 20,
    "train100": 100,
    "official_validation_forbidden": 107,
}

# These are the ScanNet L6 method signals, renamed only where the CA route has
# a different upstream source.  The learned gate may use them; runtime source
# ranking remains the frozen deterministic expression below.
FEATURE_NAMES = (
    "best_score",
    "score_mean",
    "score_std",
    "log_hit_count",
    "lifespan_fraction",
    "hit_rate",
    "center_jitter_m",
    "extent_jitter_relative",
    "mean_match_iou",
    "log_point_support",
    "log_point_density",
    "post_terminal_anchor_iou_max",
    "post_terminal_anchor_center_distance_m",
    "matched_post_terminal_anchor_score",
    "log_volume",
    "log_aspect_ratio",
    "first_call_fraction",
    "last_call_fraction",
    "visibility_quality_mean",
    "support_ratio_mean",
    "free_space_ratio_mean",
    "invalid_ratio_mean",
    "selected_geometry_fused",
)

SOURCE_RANK_FORMULA = (
    "novelty_probability+0.10*visibility01+0.08*support"
    "-0.18*free_space+0.04*fused_geometry"
)
SCORE_POLICY = "global_source_rank_float32_below_every_sealed_anchor_v1"

_SHA = re.compile(r"^[0-9a-f]{64}$")
_SCENE = re.compile(r"^[0-9]{8}$")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_aware_rank(
    *,
    novelty_probability: float,
    visibility_quality_mean: float,
    support_ratio_mean: float,
    free_space_ratio_mean: float,
    selected_geometry: str,
) -> float:
    """Return the frozen L6 source-aware rank with strict input checks."""

    probability = float(novelty_probability)
    visibility = float(visibility_quality_mean)
    support = float(support_ratio_mean)
    free_space = float(free_space_ratio_mean)
    if not all(math.isfinite(value) for value in (
        probability, visibility, support, free_space
    )):
        raise ValueError("L6 source-rank inputs must be finite")
    if not 0.0 <= probability <= 1.0:
        raise ValueError("novelty probability must be in [0,1]")
    if not -1.0 <= visibility <= 1.0:
        raise ValueError("visibility quality must be in [-1,1]")
    if not 0.0 <= support <= 1.0 or not 0.0 <= free_space <= 1.0:
        raise ValueError("support/free-space ratios must be in [0,1]")
    if selected_geometry not in {"raw", "fused"}:
        raise ValueError("selected geometry must be raw or fused")
    visibility01 = float(np.clip((visibility + 1.0) * 0.5, 0.0, 1.0))
    return float(
        probability
        + 0.10 * visibility01
        + 0.08 * support
        - 0.18 * free_space
        + (0.04 if selected_geometry == "fused" else 0.0)
    )


def assign_low_scores(
    entries: Sequence[tuple[int, int, float]], anchor_score_floor: float
) -> dict[tuple[int, int], float]:
    """Map global source rank to distinct float32 scores below all anchors."""

    floor = float(anchor_score_floor)
    if not math.isfinite(floor) or floor <= 0.0:
        raise ValueError("sealed anchor score floor must be positive and finite")
    ordered = sorted(entries, key=lambda row: (-row[2], row[0], row[1]))
    if any(
        isinstance(scene, bool)
        or isinstance(local, bool)
        or int(scene) != scene
        or int(local) != local
        or scene < 0
        or local < 0
        or not math.isfinite(float(rank))
        for scene, local, rank in ordered
    ):
        raise ValueError("invalid L6 rank entry")
    identities = [(int(row[0]), int(row[1])) for row in ordered]
    if len(identities) != len(set(identities)):
        raise ValueError("duplicate L6 candidate identity")
    if not ordered:
        return {}
    cap = np.float32(floor * 0.5)
    if not 0.0 < float(cap) < floor:
        raise ValueError("cannot construct a positive score band below anchors")
    result: dict[tuple[int, int], float] = {}
    previous = float("inf")
    count = len(ordered)
    for rank_index, (scene_index, local_index, _) in enumerate(ordered):
        score = float(
            np.float32(float(cap) * (count - rank_index) / (count + 1.0))
        )
        if not 0.0 < score < floor or not score < previous:
            raise ValueError("float32 quantization changed the global source rank")
        result[(int(scene_index), int(local_index))] = score
        previous = score
    return result


def validate_low_score_contract(
    anchor_scores: Any, candidate_scores: Any
) -> None:
    anchors = np.asarray(anchor_scores)
    candidates = np.asarray(candidate_scores)
    if anchors.ndim != 1 or candidates.ndim != 1:
        raise ValueError("anchor/candidate scores must be one-dimensional")
    if not np.isfinite(anchors).all() or not np.isfinite(candidates).all():
        raise ValueError("anchor/candidate scores must be finite")
    if not len(anchors) or np.any(anchors <= 0.0):
        raise ValueError("sealed anchors must have positive scores")
    if len(candidates) and (
        np.any(candidates <= 0.0)
        or float(candidates.max()) >= float(anchors.min())
    ):
        raise ValueError("every L6 score must be positive and below every anchor")


def _head(payload: Mapping[str, Any], name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    value = payload.get(name)
    if not isinstance(value, Mapping):
        raise ValueError(f"CA L6 policy lacks {name} head")
    mean = np.asarray(value.get("feature_mean"), dtype=np.float64)
    scale = np.asarray(value.get("feature_scale"), dtype=np.float64)
    weights = np.asarray(value.get("weights"), dtype=np.float64)
    bias = float(value.get("bias", math.nan))
    expected = (len(FEATURE_NAMES),)
    if (
        mean.shape != expected
        or scale.shape != expected
        or weights.shape != expected
        or not np.isfinite(mean).all()
        or not np.isfinite(scale).all()
        or not np.isfinite(weights).all()
        or not math.isfinite(bias)
        or np.any(scale <= 0.0)
    ):
        raise ValueError(f"CA L6 policy has invalid {name} tensors")
    return mean, scale, weights, bias


def _sigmoid(logit: float) -> float:
    if logit >= 0.0:
        return float(1.0 / (1.0 + math.exp(-min(logit, 700.0))))
    value = math.exp(max(logit, -700.0))
    return float(value / (1.0 + value))


@dataclass(frozen=True)
class CA1MIncrementalL6Policy:
    path: Path
    sha256: str
    novelty_mean: np.ndarray
    novelty_scale: np.ndarray
    novelty_weights: np.ndarray
    novelty_bias: float
    quality_mean: np.ndarray
    quality_scale: np.ndarray
    quality_weights: np.ndarray
    quality_bias: float
    novelty_threshold: float
    quality_threshold: float
    hard_max_anchor_iou: float
    max_candidates_per_scene: int

    @classmethod
    def load(cls, path: str | Path) -> "CA1MIncrementalL6Policy":
        raw = Path(path)
        if raw.is_symlink():
            raise ValueError("CA L6 policy must not be a symlink")
        source = raw.resolve()
        if (
            not source.is_file()
            or source.is_symlink()
            or source.stat().st_size <= 0
            or source.stat().st_mode & 0o222
        ):
            raise ValueError("CA L6 policy must be an immutable regular file")
        try:
            payload = json.loads(source.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            raise ValueError("CA L6 policy is not valid JSON") from error
        if not isinstance(payload, Mapping):
            raise ValueError("CA L6 policy must be an object")
        required = {
            "schema": POLICY_SCHEMA,
            "complete": True,
            "activation_authorized": True,
            "dataset": "ca1m_train100",
            "train_only": True,
            "official_validation_access": False,
            "validation_predictions_used_for_training": False,
            "validation_overlap_count": 0,
            "upstream_route": UPSTREAM_ROUTE,
            "terminal_anchor_cross_fitted": True,
            "source_rank_formula": SOURCE_RANK_FORMULA,
            "score_policy": SCORE_POLICY,
        }
        for name, expected in required.items():
            if payload.get(name) != expected:
                raise ValueError(f"CA L6 policy violates {name}")
        if tuple(payload.get("feature_names", ())) != FEATURE_NAMES:
            raise ValueError("CA L6 policy feature contract differs")
        binding = payload.get("ca_native_tr3d_binding")
        if not isinstance(binding, Mapping) or binding != {
            "schema": CA_TR3D_BINDING_SCHEMA,
            "sha256": CA_TR3D_BINDING_SHA256,
            "checkpoint_sha256": CA_TR3D_CHECKPOINT_SHA256,
            "initialization": "ca1m_random_scratch",
            "scannet_checkpoint_or_config_access": False,
        }:
            raise ValueError("CA L6 policy is not bound to the CA-scratch TR3D model")
        split = payload.get("split")
        if not isinstance(split, Mapping):
            raise ValueError("CA L6 policy lacks split provenance")
        for role, folds in SPLIT_FOLDS.items():
            row = split.get(role)
            if not isinstance(row, Mapping) or row != {
                "folds": list(folds),
                "scene_count": SPLIT_COUNTS[role],
                "scene_list_sha256": SPLIT_SHA256[role],
            }:
                raise ValueError(f"CA L6 policy has invalid {role} split")
        if split.get("train100_scene_list_sha256") != SPLIT_SHA256["train100"]:
            raise ValueError("CA L6 policy train100 identity differs")
        if (
            split.get("official_validation_scene_list_sha256")
            != SPLIT_SHA256["official_validation_forbidden"]
        ):
            raise ValueError("CA L6 policy forbidden-validation identity differs")
        upstream = payload.get("upstream")
        expected_upstream = {
            "final_base_manifest_sha256",
            "native_b6_v2_collection_manifest_sha256",
            "native_b6_v2_checkpoint_manifest_sha256",
            "terminal_v4_manifest_sha256",
            "terminal_benefit_v2_policy_sha256",
            "post_terminal_anchor_manifest_sha256",
        }
        if not isinstance(upstream, Mapping) or set(upstream) != expected_upstream:
            raise ValueError("CA L6 policy upstream provenance keys differ")
        if any(_SHA.fullmatch(str(upstream[name])) is None for name in upstream):
            raise ValueError("CA L6 policy upstream provenance lacks SHA256 bindings")
        counts = payload.get("sample_gate")
        if not isinstance(counts, Mapping) or any(
            int(counts.get(name, -1)) < minimum
            for name, minimum in {
                "weights_train_candidates": 120,
                "weights_train_novel25_positive": 20,
                "weights_train_novel25_negative": 20,
                "weights_train_novel50_positive": 10,
                "threshold_dev_candidates": 20,
                "threshold_dev_positive_scenes": 4,
                "locked_internal_candidates": 20,
                "locked_internal_positive_scenes": 4,
            }.items()
        ):
            raise ValueError("CA L6 policy did not pass the frozen sample gate")
        audit = payload.get("locked_internal_audit")
        if not isinstance(audit, Mapping) or (
            audit.get("consumed_once") is not True
            or audit.get("gate_passed") is not True
            or audit.get("official_validation_access") is not False
        ):
            raise ValueError("CA L6 locked internal audit did not authorize activation")
        novelty = _head(payload, "novelty25_head")
        quality = _head(payload, "quality50_head")
        novelty_threshold = float(payload.get("novelty_threshold", math.nan))
        quality_threshold = float(payload.get("quality_threshold", math.nan))
        hard_iou = float(payload.get("hard_max_post_terminal_anchor_iou", math.nan))
        maximum = int(payload.get("max_candidates_per_scene", -1))
        if (
            not 0.0 < novelty_threshold < 1.0
            or not 0.0 < quality_threshold < 1.0
            or not 0.0 <= hard_iou < 1.0
            or maximum < 1
            or maximum > 32
        ):
            raise ValueError("CA L6 policy thresholds/capacity are invalid")
        return cls(
            path=source,
            sha256=sha256_file(source),
            novelty_mean=novelty[0],
            novelty_scale=novelty[1],
            novelty_weights=novelty[2],
            novelty_bias=novelty[3],
            quality_mean=quality[0],
            quality_scale=quality[1],
            quality_weights=quality[2],
            quality_bias=quality[3],
            novelty_threshold=novelty_threshold,
            quality_threshold=quality_threshold,
            hard_max_anchor_iou=hard_iou,
            max_candidates_per_scene=maximum,
        )

    def probabilities(self, features: Any) -> tuple[float, float]:
        value = np.asarray(features, dtype=np.float64)
        if value.shape != (len(FEATURE_NAMES),) or not np.isfinite(value).all():
            raise ValueError("CA L6 features must be one finite feature row")
        novelty = _sigmoid(float(
            ((value - self.novelty_mean) / self.novelty_scale)
            @ self.novelty_weights
            + self.novelty_bias
        ))
        quality = _sigmoid(float(
            ((value - self.quality_mean) / self.quality_scale)
            @ self.quality_weights
            + self.quality_bias
        ))
        return novelty, quality


__all__ = [
    "ACTIVE_SCHEMA",
    "CA1MIncrementalL6Policy",
    "CA_TR3D_BINDING_SCHEMA",
    "CA_TR3D_BINDING_SHA256",
    "CA_TR3D_CHECKPOINT_SHA256",
    "DATASET_SCHEMA",
    "FEATURE_NAMES",
    "OBSERVER_SCHEMA",
    "POLICY_SCHEMA",
    "SCORE_POLICY",
    "SOURCE_RANK_FORMULA",
    "SPLIT_COUNTS",
    "SPLIT_FOLDS",
    "SPLIT_SHA256",
    "UPSTREAM_ROUTE",
    "assign_low_scores",
    "source_aware_rank",
    "validate_low_score_contract",
]
