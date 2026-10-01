"""Training-free PUF-style probability and birth observer for BoxFusion.

Only PUF's per-observation likelihood normalization and explicit birth state
are retained.  The semantic JSD/Dirichlet path is intentionally absent.  A
fixed AABB geometry approximation scores Moon-QIM-lite candidates, and a
bounded scan of the previous committed tracks is used only when the sparse
shortlist would otherwise predict birth.

This first integration is shadow-only: it never mutates BoxFusion tracks,
boxes, scores, categories, or CLIP features.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, replace
from numbers import Integral, Real
from time import perf_counter_ns
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np

from .moon_qim_lite import QIMCandidate, QIMQueryBatch


DEFAULT_PUF_LITE_CONFIG = {
    "enabled": False,
    "observer_only": True,
    "top_k": 3,
    # Frozen literature constant.  Do not tune it on BoxFusion evaluation GT.
    "birth_likelihood": 0.40,
    "center_sigma": 0.50,
    "center_margin_m": 0.05,
    "shared_key_power": 1.0,
    "max_tracks": 1024,
    "exhaustive_fallback": True,
    "probability_tolerance": 1e-12,
    "snapshot_tolerance": 1e-5,
    "epsilon": 1e-9,
    "max_diagnostic_examples": 64,
}


def _strict_bool(name: str, value: object) -> bool:
    if not isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a boolean")
    return bool(value)


def _strict_int(name: str, value: object, minimum: int) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer")
    result = int(value)
    if result < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return result


def _finite_float(
    name: str,
    value: object,
    minimum: float,
    *,
    strict_minimum: bool = False,
) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real number")
    result = float(value)
    invalid = result <= minimum if strict_minimum else result < minimum
    if not np.isfinite(result) or invalid:
        qualifier = "greater than" if strict_minimum else "at least"
        raise ValueError(f"{name} must be finite and {qualifier} {minimum}")
    return result


def resolve_puf_lite_config(
    config: Optional[Mapping[str, object]] = None,
) -> Dict[str, object]:
    """Return a strict PUF-lite configuration."""

    if config is None:
        config = {}
    if not isinstance(config, Mapping):
        raise ValueError("puf_lite config must be a mapping")
    unknown = sorted(set(config) - set(DEFAULT_PUF_LITE_CONFIG))
    if unknown:
        raise ValueError("Unknown puf_lite config key(s): " + ", ".join(unknown))
    resolved = dict(DEFAULT_PUF_LITE_CONFIG)
    resolved.update(config)
    resolved["enabled"] = _strict_bool("puf_lite.enabled", resolved["enabled"])
    resolved["observer_only"] = _strict_bool(
        "puf_lite.observer_only", resolved["observer_only"]
    )
    if resolved["enabled"] and not resolved["observer_only"]:
        raise ValueError(
            "puf_lite active association is not authorized; "
            "observer_only must remain true"
        )
    resolved["top_k"] = _strict_int("puf_lite.top_k", resolved["top_k"], 1)
    if resolved["top_k"] > 3:
        raise ValueError("puf_lite.top_k must not exceed 3")
    for key in ("birth_likelihood", "center_sigma", "center_margin_m", "epsilon"):
        resolved[key] = _finite_float(
            f"puf_lite.{key}", resolved[key], 0.0, strict_minimum=True
        )
    resolved["shared_key_power"] = _finite_float(
        "puf_lite.shared_key_power",
        resolved["shared_key_power"],
        0.0,
        strict_minimum=True,
    )
    resolved["probability_tolerance"] = _finite_float(
        "puf_lite.probability_tolerance",
        resolved["probability_tolerance"],
        0.0,
        strict_minimum=True,
    )
    resolved["snapshot_tolerance"] = _finite_float(
        "puf_lite.snapshot_tolerance",
        resolved["snapshot_tolerance"],
        0.0,
        strict_minimum=True,
    )
    if resolved["enabled"] and resolved["birth_likelihood"] != 0.40:
        raise ValueError(
            "enabled puf_lite must keep the frozen literature "
            "birth_likelihood=0.40"
        )
    if resolved["birth_likelihood"] > 1.0:
        raise ValueError("puf_lite.birth_likelihood must not exceed 1")
    if resolved["probability_tolerance"] > 1e-6:
        raise ValueError("puf_lite.probability_tolerance must not exceed 1e-6")
    resolved["max_tracks"] = _strict_int(
        "puf_lite.max_tracks", resolved["max_tracks"], 1
    )
    if resolved["max_tracks"] > 1024:
        raise ValueError("puf_lite.max_tracks must not exceed 1024")
    resolved["exhaustive_fallback"] = _strict_bool(
        "puf_lite.exhaustive_fallback", resolved["exhaustive_fallback"]
    )
    resolved["max_diagnostic_examples"] = _strict_int(
        "puf_lite.max_diagnostic_examples",
        resolved["max_diagnostic_examples"],
        0,
    )
    return resolved


def _as_numpy(value: object, name: str) -> np.ndarray:
    candidate = value
    if hasattr(candidate, "detach"):
        candidate = candidate.detach()
    if hasattr(candidate, "cpu"):
        candidate = candidate.cpu()
    if hasattr(candidate, "numpy"):
        candidate = candidate.numpy()
    try:
        return np.asarray(candidate)
    except Exception as error:
        raise ValueError(f"{name} cannot be converted to NumPy") from error


def _validated_ids(value: object, count: int, name: str) -> np.ndarray:
    ids = _as_numpy(value, name)
    if ids.ndim != 1 or len(ids) != count:
        raise ValueError(f"{name} must have shape [{count}]")
    if not np.issubdtype(ids.dtype, np.integer):
        raise ValueError(f"{name} must contain integers")
    ids = ids.astype(np.int64, copy=False)
    if np.any(ids < 0):
        raise ValueError(f"{name} must be non-negative")
    if len(np.unique(ids)) != len(ids):
        raise ValueError(f"{name} must be unique")
    return np.array(ids, dtype=np.int64, order="C", copy=True)


def _box_array(value: object, count: int, name: str) -> np.ndarray:
    corners = _as_numpy(value, name).astype(np.float64, copy=False)
    if corners.shape != (count, 8, 3):
        raise ValueError(f"{name} must have shape [{count}, 8, 3]")
    return np.array(corners, dtype=np.float64, order="C", copy=True)


@dataclass(frozen=True)
class GeometryLikelihood:
    containment: float
    aabb_iou: float
    overlap_support: float
    center_support: float
    shared_key_fraction: float
    likelihood: float


def normalize_puf_likelihoods(
    likelihoods: Sequence[float], birth_likelihood: float = 0.40
) -> Tuple[Tuple[float, ...], float]:
    """Apply PUF Eq. (6) directly, without a softmax."""

    birth = _finite_float(
        "birth_likelihood", birth_likelihood, 0.0, strict_minimum=True
    )
    values = []
    for index, value in enumerate(likelihoods):
        likelihood = _finite_float(f"likelihoods[{index}]", value, 0.0)
        if likelihood > 1.0:
            raise ValueError(f"likelihoods[{index}] must not exceed 1")
        values.append(likelihood)
    normalizer = birth + sum(values)
    probabilities = tuple(value / normalizer for value in values)
    return probabilities, birth / normalizer


def box_geometry_likelihood(
    proposal_corners: object,
    track_corners: object,
    *,
    shared_key_fraction: float,
    center_sigma: float = 0.50,
    center_margin_m: float = 0.05,
    shared_key_power: float = 1.0,
    epsilon: float = 1e-9,
) -> GeometryLikelihood:
    """Compute the fixed BoxFusion geometry approximation used by PUF-lite."""

    proposal = np.asarray(proposal_corners, dtype=np.float64)
    track = np.asarray(track_corners, dtype=np.float64)
    if proposal.shape != (8, 3) or track.shape != (8, 3):
        raise ValueError("proposal_corners and track_corners must have shape [8, 3]")
    if not np.isfinite(proposal).all() or not np.isfinite(track).all():
        raise ValueError("box corners must contain only finite values")
    q = _finite_float("shared_key_fraction", shared_key_fraction, 0.0)
    if q > 1.0:
        raise ValueError("shared_key_fraction must not exceed 1")
    sigma = _finite_float("center_sigma", center_sigma, 0.0, strict_minimum=True)
    margin = _finite_float(
        "center_margin_m", center_margin_m, 0.0, strict_minimum=True
    )
    power = _finite_float(
        "shared_key_power", shared_key_power, 0.0, strict_minimum=True
    )
    eps = _finite_float("epsilon", epsilon, 0.0, strict_minimum=True)

    proposal_lower, proposal_upper = np.min(proposal, axis=0), np.max(proposal, axis=0)
    track_lower, track_upper = np.min(track, axis=0), np.max(track, axis=0)
    proposal_size = proposal_upper - proposal_lower
    track_size = track_upper - track_lower
    if np.any(proposal_size <= eps) or np.any(track_size <= eps):
        raise ValueError("box AABB extents must be positive")
    proposal_volume = float(np.prod(proposal_size))
    track_volume = float(np.prod(track_size))
    intersection_size = np.maximum(
        np.minimum(proposal_upper, track_upper)
        - np.maximum(proposal_lower, track_lower),
        0.0,
    )
    intersection = float(np.prod(intersection_size))
    containment = float(np.clip(intersection / proposal_volume, 0.0, 1.0))
    union = proposal_volume + track_volume - intersection
    aabb_iou = float(np.clip(intersection / union, 0.0, 1.0)) if union > eps else 0.0
    overlap_support = float(np.sqrt(containment * aabb_iou))

    proposal_center = (proposal_lower + proposal_upper) / 2.0
    track_center = (track_lower + track_upper) / 2.0
    scale = sigma * 0.5 * (proposal_size + track_size) + margin
    standardized = (proposal_center - track_center) / scale
    center_support = float(np.exp(-0.5 * float(np.dot(standardized, standardized))))
    likelihood = overlap_support + (1.0 - overlap_support) * (q ** power) * center_support
    likelihood = float(np.clip(likelihood, 0.0, 1.0))
    if not np.isfinite(likelihood):
        raise ValueError("geometry likelihood is not finite")
    return GeometryLikelihood(
        containment=containment,
        aabb_iou=aabb_iou,
        overlap_support=overlap_support,
        center_support=center_support,
        shared_key_fraction=q,
        likelihood=likelihood,
    )


@dataclass(frozen=True)
class PUFCandidatePosterior:
    track_id: int
    global_row: int
    source: str
    qim_rank: Optional[int]
    containment: float
    aabb_iou: float
    overlap_support: float
    center_support: float
    shared_key_fraction: float
    likelihood: float
    probability: float


@dataclass(frozen=True)
class PUFProposalDecision:
    proposal_id: int
    valid: bool
    actionable: bool
    invalid_reason: Optional[str]
    conflict: bool
    qim_candidate_track_ids: Tuple[int, ...]
    candidates: Tuple[PUFCandidatePosterior, ...]
    birth_probability: Optional[float]
    predicted_birth: Optional[bool]
    predicted_track_id: Optional[int]
    predicted_global_row: Optional[int]
    fallback_triggered: bool
    fallback_rescued: bool
    exhaustive_ms: float
    normalization_error: Optional[float]


@dataclass(frozen=True)
class PUFQueryBatch:
    scene_id: str
    frame_id: int
    history_max_frame_id: Optional[int]
    proposal_ids: Tuple[int, ...]
    rows: Tuple[PUFProposalDecision, ...]
    query_ms: float


def _invalid_row(proposal_id: int, reason: str) -> PUFProposalDecision:
    return PUFProposalDecision(
        proposal_id=int(proposal_id),
        valid=False,
        actionable=False,
        invalid_reason=str(reason),
        conflict=False,
        qim_candidate_track_ids=(),
        candidates=(),
        birth_probability=None,
        predicted_birth=None,
        predicted_track_id=None,
        predicted_global_row=None,
        fallback_triggered=False,
        fallback_rescued=False,
        exhaustive_ms=0.0,
        normalization_error=None,
    )


class PUFLiteShadowObserver:
    """Causal, bounded, training-free PUF-lite shadow observer."""

    _LATENCY_WINDOW = 2048

    def __init__(self, config: Optional[Mapping[str, object]] = None):
        self.config = resolve_puf_lite_config(config)
        self.enabled = bool(self.config["enabled"])
        self.observer_only = bool(self.config["observer_only"])
        self._scene_id: Optional[str] = None
        self._last_query_frame_id: Optional[int] = None
        self._pending_batch: Optional[PUFQueryBatch] = None
        self._last_observed_batch: Optional[PUFQueryBatch] = None
        self._query_samples = deque(maxlen=self._LATENCY_WINDOW)
        self._exhaustive_samples = deque(maxlen=self._LATENCY_WINDOW)
        self._examples: list[dict[str, object]] = []
        self._stats = self._new_stats()

    @staticmethod
    def _new_stats() -> Dict[str, object]:
        return {
            "queries": 0,
            "proposals": 0,
            "valid_rows": 0,
            "invalid_rows": 0,
            "actionable_rows": 0,
            "same_track_conflicts": 0,
            "qim_candidates_retained": 0,
            "stale_candidates_dropped": 0,
            "fallback_triggers": 0,
            "fallback_rescues": 0,
            "exhaustive_tracks_scored": 0,
            "probability_rows": 0,
            "nonfinite_probability_rows": 0,
            "max_normalization_error": 0.0,
            "native_history_matches": 0,
            "native_births": 0,
            "native_ambiguous": 0,
            "ambiguous_qim_coverage_any": 0,
            "ambiguous_final_support_any": 0,
            "ambiguous_top1_in_target_set": 0,
            "native_unresolved": 0,
            "qim_target_coverage_at_3": 0,
            "post_fallback_target_coverage": 0,
            "top1_native_agreement": 0,
            "conditional_top1_agreement": 0,
            "conditional_top1_denominator": 0,
            "native_decision_agreement": 0,
            "predicted_births_evaluated": 0,
            "birth_true_positives": 0,
            "false_births": 0,
            "wrong_tracks": 0,
            "retrieval_misses": 0,
            "invalid_native_rows": 0,
            "nll_total": 0.0,
            "brier_total": 0.0,
            "proper_score_rows": 0,
            "query_ms_total": 0.0,
            "query_ms_max": 0.0,
            "exhaustive_ms_total": 0.0,
            "exhaustive_ms_max": 0.0,
            "pipeline_query_calls": 0,
            "pipeline_query_ms_total": 0.0,
            "pipeline_query_ms_max": 0.0,
            "pipeline_observe_calls": 0,
            "pipeline_observe_ms_total": 0.0,
            "pipeline_observe_ms_max": 0.0,
        }

    def reset_scene(self, scene_id: str) -> None:
        scene_id = str(scene_id)
        if not scene_id:
            raise ValueError("scene_id must not be empty")
        self._scene_id = scene_id
        self._last_query_frame_id = None
        self._pending_batch = None
        self._last_observed_batch = None
        self._query_samples.clear()
        self._exhaustive_samples.clear()
        self._examples.clear()
        self._stats = self._new_stats()

    def _bind_scene(self, scene_id: str) -> str:
        scene_id = str(scene_id)
        if not scene_id:
            raise ValueError("scene_id must not be empty")
        if self._scene_id is None:
            self.reset_scene(scene_id)
        elif scene_id != self._scene_id:
            raise ValueError(f"puf_lite is bound to {self._scene_id}, not {scene_id}")
        return scene_id

    def _geometry(
        self,
        proposal: np.ndarray,
        track: np.ndarray,
        q: float,
    ) -> GeometryLikelihood:
        return box_geometry_likelihood(
            proposal,
            track,
            shared_key_fraction=q,
            center_sigma=float(self.config["center_sigma"]),
            center_margin_m=float(self.config["center_margin_m"]),
            shared_key_power=float(self.config["shared_key_power"]),
            epsilon=float(self.config["epsilon"]),
        )

    def _geometry_many(
        self,
        proposal: np.ndarray,
        tracks: np.ndarray,
        shared_key_fractions: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Vectorized geometry evidence for the bounded exhaustive fallback."""

        count = len(tracks)
        if count == 0:
            empty = np.empty((0,), dtype=np.float64)
            return empty, empty, empty, empty, empty
        q = np.asarray(shared_key_fractions, dtype=np.float64)
        if q.shape != (count,) or not np.isfinite(q).all() or np.any(q < 0.0) or np.any(q > 1.0):
            raise ValueError("fallback shared-key fractions must lie in [0, 1]")
        proposal_lower = np.min(proposal, axis=0)
        proposal_upper = np.max(proposal, axis=0)
        proposal_size = proposal_upper - proposal_lower
        proposal_volume = float(np.prod(proposal_size))
        track_lower = np.min(tracks, axis=1)
        track_upper = np.max(tracks, axis=1)
        track_size = track_upper - track_lower
        track_volume = np.prod(track_size, axis=1)
        intersection_size = np.maximum(
            np.minimum(track_upper, proposal_upper[None, :])
            - np.maximum(track_lower, proposal_lower[None, :]),
            0.0,
        )
        intersection = np.prod(intersection_size, axis=1)
        containment = np.clip(intersection / proposal_volume, 0.0, 1.0)
        union = proposal_volume + track_volume - intersection
        iou = np.divide(
            intersection,
            union,
            out=np.zeros_like(intersection),
            where=union > float(self.config["epsilon"]),
        )
        iou = np.clip(iou, 0.0, 1.0)
        overlap = np.sqrt(containment * iou)
        proposal_center = (proposal_lower + proposal_upper) / 2.0
        track_center = (track_lower + track_upper) / 2.0
        scale = (
            float(self.config["center_sigma"])
            * 0.5
            * (track_size + proposal_size[None, :])
            + float(self.config["center_margin_m"])
        )
        standardized = (track_center - proposal_center[None, :]) / scale
        center = np.exp(-0.5 * np.sum(standardized * standardized, axis=1))
        likelihood = overlap + (1.0 - overlap) * (
            q ** float(self.config["shared_key_power"])
        ) * center
        likelihood = np.clip(likelihood, 0.0, 1.0)
        if not all(
            np.isfinite(value).all()
            for value in (containment, iou, overlap, center, likelihood)
        ):
            raise ValueError("fallback geometry likelihood is not finite")
        return containment, iou, overlap, center, likelihood

    @staticmethod
    def _validate_qim_candidate(candidate: object) -> QIMCandidate:
        if not isinstance(candidate, QIMCandidate):
            raise ValueError("QIM candidates must be QIMCandidate objects")
        if isinstance(candidate.track_id, (bool, np.bool_)) or not isinstance(
            candidate.track_id, Integral
        ) or int(candidate.track_id) < 0:
            raise ValueError("QIM candidate track_id must be non-negative")
        for name, value in (
            ("shared_key_fraction", candidate.shared_key_fraction),
            ("center_distance_m", candidate.center_distance_m),
            ("aabb_iou", candidate.aabb_iou),
        ):
            if not isinstance(value, Real) or not np.isfinite(float(value)):
                raise ValueError(f"QIM candidate {name} must be finite")
        if not 0.0 <= float(candidate.shared_key_fraction) <= 1.0:
            raise ValueError("QIM shared_key_fraction must lie in [0, 1]")
        if float(candidate.center_distance_m) < 0.0:
            raise ValueError("QIM center_distance_m must be non-negative")
        if not 0.0 <= float(candidate.aabb_iou) <= 1.0:
            raise ValueError("QIM aabb_iou must lie in [0, 1]")
        if isinstance(candidate.shared_key_count, (bool, np.bool_)) or not isinstance(
            candidate.shared_key_count, Integral
        ) or int(candidate.shared_key_count) < 0:
            raise ValueError("QIM shared_key_count must be non-negative")
        if isinstance(candidate.age_keyframes, (bool, np.bool_)) or not isinstance(
            candidate.age_keyframes, Integral
        ) or int(candidate.age_keyframes) < 0:
            raise ValueError("QIM age_keyframes must be non-negative")
        if not isinstance(candidate.active_at_last_commit, (bool, np.bool_)):
            raise ValueError("QIM active_at_last_commit must be boolean")
        return candidate

    def _score_row(
        self,
        *,
        proposal_id: int,
        proposal: np.ndarray,
        raw_candidates: Sequence[QIMCandidate],
        active_ids: np.ndarray,
        active_boxes: np.ndarray,
        id_to_row: Mapping[int, int],
    ) -> Tuple[PUFProposalDecision, int, int, int]:
        if not np.isfinite(proposal).all():
            return _invalid_row(proposal_id, "nonfinite_proposal_geometry"), 0, 0, 0
        proposal_size = np.max(proposal, axis=0) - np.min(proposal, axis=0)
        if np.any(proposal_size <= float(self.config["epsilon"])):
            return _invalid_row(proposal_id, "nonpositive_proposal_extent"), 0, 0, 0

        # Deduplicate by stable ID, preserving first QIM rank but retaining the
        # strongest q if malformed external input repeats an ID.
        by_id: Dict[int, Tuple[int, float]] = {}
        stale_drops = 0
        try:
            for rank, raw in enumerate(raw_candidates):
                candidate = self._validate_qim_candidate(raw)
                track_id = int(candidate.track_id)
                row = id_to_row.get(track_id)
                if row is None:
                    if bool(candidate.active_at_last_commit):
                        return (
                            _invalid_row(proposal_id, "active_candidate_registry_mismatch"),
                            stale_drops,
                            0,
                            0,
                        )
                    stale_drops += 1
                    continue
                if not bool(candidate.active_at_last_commit):
                    return (
                        _invalid_row(proposal_id, "candidate_liveness_mismatch"),
                        stale_drops,
                        0,
                        0,
                    )
                q = float(candidate.shared_key_fraction)
                previous = by_id.get(track_id)
                if previous is None:
                    by_id[track_id] = (rank, q)
                else:
                    by_id[track_id] = (previous[0], max(previous[1], q))
        except ValueError as error:
            return _invalid_row(proposal_id, str(error)), stale_drops, 0, 0

        ranked_qim = sorted(by_id.items(), key=lambda item: (item[1][0], item[0]))
        ranked_qim = ranked_qim[: int(self.config["top_k"])]
        qim_ids = tuple(track_id for track_id, _ in ranked_qim)
        qim_q = {track_id: rank_and_q[1] for track_id, rank_and_q in ranked_qim}
        scored: list[Tuple[int, int, str, Optional[int], GeometryLikelihood]] = []
        try:
            for track_id, (rank, q) in ranked_qim:
                global_row = int(id_to_row[track_id])
                evidence = self._geometry(proposal, active_boxes[global_row], q)
                raw = raw_candidates[rank]
                center_distance = float(
                    np.linalg.norm(
                        np.mean(proposal, axis=0)
                        - np.mean(active_boxes[global_row], axis=0)
                    )
                )
                tolerance = float(self.config["snapshot_tolerance"])
                if (
                    abs(center_distance - float(raw.center_distance_m))
                    > tolerance
                    or abs(evidence.aabb_iou - float(raw.aabb_iou))
                    > tolerance
                    or int(raw.age_keyframes) != 0
                ):
                    return (
                        _invalid_row(proposal_id, "candidate_snapshot_mismatch"),
                        stale_drops,
                        0,
                        0,
                    )
                scored.append((track_id, global_row, "qim", rank, evidence))
        except ValueError as error:
            return _invalid_row(proposal_id, str(error)), stale_drops, 0, 0

        birth_likelihood = float(self.config["birth_likelihood"])
        fallback_triggered = sum(item[4].likelihood for item in scored) < birth_likelihood
        fallback_rescued = False
        exhaustive_ms = 0.0
        exhaustive_tracks = 0
        if fallback_triggered and bool(self.config["exhaustive_fallback"]):
            exhaustive_start = perf_counter_ns()
            try:
                q_values = np.asarray(
                    [qim_q.get(int(track_id), 0.0) for track_id in active_ids],
                    dtype=np.float64,
                )
                containment, iou, overlap, center, likelihood = self._geometry_many(
                    proposal, active_boxes, q_values
                )
            except ValueError as error:
                return _invalid_row(proposal_id, str(error)), stale_drops, 0, 0
            exhaustive_tracks = len(active_ids)
            order = np.lexsort((active_ids, -likelihood))
            selected_rows = order[: int(self.config["top_k"])]
            scored = []
            for global_row_value in selected_rows:
                global_row = int(global_row_value)
                track_id = int(active_ids[global_row])
                evidence = GeometryLikelihood(
                    containment=float(containment[global_row]),
                    aabb_iou=float(iou[global_row]),
                    overlap_support=float(overlap[global_row]),
                    center_support=float(center[global_row]),
                    shared_key_fraction=float(q_values[global_row]),
                    likelihood=float(likelihood[global_row]),
                )
                scored.append(
                    (
                        track_id,
                        global_row,
                        "qim" if track_id in qim_q else "fallback",
                        by_id[track_id][0] if track_id in qim_q else None,
                        evidence,
                    )
                )
            exhaustive_ms = (perf_counter_ns() - exhaustive_start) / 1e6
            fallback_rescued = sum(
                item[4].likelihood for item in scored
            ) >= birth_likelihood

        scored = [item for item in scored if item[4].likelihood > 0.0]
        scored.sort(key=lambda item: (-item[4].likelihood, item[0]))
        try:
            candidate_probabilities, birth_probability = normalize_puf_likelihoods(
                [item[4].likelihood for item in scored], birth_likelihood
            )
        except ValueError:
            return _invalid_row(proposal_id, "invalid_probability_normalizer"), stale_drops, exhaustive_tracks, 0
        posteriors = tuple(
            PUFCandidatePosterior(
                track_id=track_id,
                global_row=global_row,
                source=source,
                qim_rank=qim_rank,
                containment=evidence.containment,
                aabb_iou=evidence.aabb_iou,
                overlap_support=evidence.overlap_support,
                center_support=evidence.center_support,
                shared_key_fraction=evidence.shared_key_fraction,
                likelihood=evidence.likelihood,
                probability=probability,
            )
            for (track_id, global_row, source, qim_rank, evidence), probability
            in zip(scored, candidate_probabilities)
        )
        probabilities = [item.probability for item in posteriors] + [birth_probability]
        normalization_error = abs(sum(probabilities) - 1.0)
        tolerance = float(self.config["probability_tolerance"])
        if (
            not all(np.isfinite(value) and 0.0 <= value <= 1.0 for value in probabilities)
            or normalization_error > tolerance
        ):
            return _invalid_row(proposal_id, "invalid_probability_row"), stale_drops, exhaustive_tracks, 1
        predicted_birth = bool(birth_probability > 0.5)
        winner = None if predicted_birth or not posteriors else posteriors[0]
        return (
            PUFProposalDecision(
                proposal_id=int(proposal_id),
                valid=True,
                actionable=True,
                invalid_reason=None,
                conflict=False,
                qim_candidate_track_ids=qim_ids,
                candidates=posteriors,
                birth_probability=float(birth_probability),
                predicted_birth=predicted_birth,
                predicted_track_id=None if winner is None else winner.track_id,
                predicted_global_row=None if winner is None else winner.global_row,
                fallback_triggered=fallback_triggered,
                fallback_rescued=fallback_rescued,
                exhaustive_ms=float(exhaustive_ms),
                normalization_error=float(normalization_error),
            ),
            stale_drops,
            exhaustive_tracks,
            0,
        )

    def query(
        self,
        *,
        qim_batch: QIMQueryBatch,
        proposal_corners_world: object,
        active_track_ids: object,
        active_track_corners_world: object,
    ) -> PUFQueryBatch:
        """Freeze PUF-lite posteriors before native association runs."""

        start = perf_counter_ns()
        if not self.enabled:
            raise RuntimeError("puf_lite observer is disabled")
        if self._pending_batch is not None:
            raise ValueError("previous PUF query must be observed first")
        if not isinstance(qim_batch, QIMQueryBatch):
            raise ValueError("qim_batch must be a QIMQueryBatch")
        scene_id = self._bind_scene(qim_batch.scene_id)
        frame_id = int(qim_batch.frame_id)
        if self._last_query_frame_id is not None and frame_id <= self._last_query_frame_id:
            raise ValueError("PUF query frame ids must be strictly increasing")
        if qim_batch.history_max_frame_id is not None and qim_batch.history_max_frame_id >= frame_id:
            raise ValueError("PUF query history must precede the current frame")
        proposal_ids = _validated_ids(
            np.asarray(qim_batch.proposal_ids), len(qim_batch.proposal_ids), "proposal_ids"
        )
        if len(qim_batch.candidates) != len(proposal_ids):
            raise ValueError("QIM candidates must align with proposal ids")
        proposals = _box_array(
            proposal_corners_world, len(proposal_ids), "proposal_corners_world"
        )
        active_ids_array = _as_numpy(active_track_ids, "active_track_ids")
        if active_ids_array.ndim != 1:
            raise ValueError("active_track_ids must be one-dimensional")
        active_ids = _validated_ids(
            active_ids_array, len(active_ids_array), "active_track_ids"
        )
        active_boxes = _box_array(
            active_track_corners_world,
            len(active_ids),
            "active_track_corners_world",
        )
        id_to_row = {int(track_id): row for row, track_id in enumerate(active_ids)}

        batch_invalid_reason = None
        if len(active_ids) > int(self.config["max_tracks"]):
            batch_invalid_reason = "active_track_cap_exceeded"
        elif len(active_boxes) and (
            not np.isfinite(active_boxes).all()
            or np.any(
                (np.max(active_boxes, axis=1) - np.min(active_boxes, axis=1))
                <= float(self.config["epsilon"])
            )
        ):
            batch_invalid_reason = "invalid_active_track_geometry"

        rows = []
        stale_drops = 0
        exhaustive_tracks = 0
        nonfinite_rows = 0
        if batch_invalid_reason is not None:
            rows = [_invalid_row(value, batch_invalid_reason) for value in proposal_ids]
        else:
            for proposal_id, proposal, candidates in zip(
                proposal_ids, proposals, qim_batch.candidates
            ):
                row, dropped, scanned, nonfinite = self._score_row(
                    proposal_id=int(proposal_id),
                    proposal=proposal,
                    raw_candidates=candidates,
                    active_ids=active_ids,
                    active_boxes=active_boxes,
                    id_to_row=id_to_row,
                )
                rows.append(row)
                stale_drops += dropped
                exhaustive_tracks += scanned
                nonfinite_rows += nonfinite

        # Simultaneous proposals targeting one historical row are not safe for
        # a future active override.  Preserve probabilities for diagnostics but
        # mark every member of the conflict non-actionable.
        selected: Dict[int, list[int]] = {}
        for index, row in enumerate(rows):
            if row.valid and row.predicted_track_id is not None:
                selected.setdefault(row.predicted_track_id, []).append(index)
        conflict_count = 0
        for indices in selected.values():
            if len(indices) < 2:
                continue
            conflict_count += len(indices)
            for index in indices:
                rows[index] = replace(
                    rows[index],
                    actionable=False,
                    invalid_reason="same_track_conflict",
                    conflict=True,
                )

        elapsed_ms = (perf_counter_ns() - start) / 1e6
        batch = PUFQueryBatch(
            scene_id=scene_id,
            frame_id=frame_id,
            history_max_frame_id=qim_batch.history_max_frame_id,
            proposal_ids=tuple(int(value) for value in proposal_ids),
            rows=tuple(rows),
            query_ms=float(elapsed_ms),
        )
        self._last_query_frame_id = frame_id
        self._pending_batch = batch
        self._stats["queries"] += 1
        self._stats["proposals"] += len(rows)
        self._stats["valid_rows"] += sum(row.valid for row in rows)
        self._stats["invalid_rows"] += sum(not row.valid for row in rows)
        self._stats["actionable_rows"] += sum(row.actionable for row in rows)
        self._stats["same_track_conflicts"] += conflict_count
        self._stats["qim_candidates_retained"] += sum(
            len(row.qim_candidate_track_ids) for row in rows
        )
        self._stats["stale_candidates_dropped"] += stale_drops
        self._stats["fallback_triggers"] += sum(row.fallback_triggered for row in rows)
        self._stats["fallback_rescues"] += sum(row.fallback_rescued for row in rows)
        self._stats["exhaustive_tracks_scored"] += exhaustive_tracks
        self._stats["probability_rows"] += sum(row.valid for row in rows)
        self._stats["nonfinite_probability_rows"] += nonfinite_rows
        self._stats["max_normalization_error"] = max(
            float(self._stats["max_normalization_error"]),
            max(
                (float(row.normalization_error) for row in rows if row.valid),
                default=0.0,
            ),
        )
        exhaustive_ms = sum(row.exhaustive_ms for row in rows)
        self._stats["query_ms_total"] += elapsed_ms
        self._stats["query_ms_max"] = max(float(self._stats["query_ms_max"]), elapsed_ms)
        self._stats["exhaustive_ms_total"] += exhaustive_ms
        self._stats["exhaustive_ms_max"] = max(
            float(self._stats["exhaustive_ms_max"]),
            max((row.exhaustive_ms for row in rows), default=0.0),
        )
        self._query_samples.append(float(elapsed_ms))
        self._exhaustive_samples.extend(
            row.exhaustive_ms for row in rows if row.fallback_triggered
        )
        return batch

    def _add_example(
        self,
        row: PUFProposalDecision,
        native_kind: str,
        native_track_id: Optional[int],
    ) -> None:
        if len(self._examples) >= int(self.config["max_diagnostic_examples"]):
            return
        self._examples.append(
            {
                "frame_id": self._pending_batch.frame_id if self._pending_batch else None,
                "proposal_id": row.proposal_id,
                "native_kind": native_kind,
                "native_track_id": native_track_id,
                "predicted_birth": row.predicted_birth,
                "predicted_track_id": row.predicted_track_id,
                "birth_probability": row.birth_probability,
                "fallback_triggered": row.fallback_triggered,
                "conflict": row.conflict,
                "invalid_reason": row.invalid_reason,
            }
        )

    def observe_native_targets(
        self,
        batch: PUFQueryBatch,
        native_target_track_ids: Sequence[Optional[Iterable[int]]],
    ) -> None:
        """Record diagnostics only; native targets never alter scoring state."""

        if batch is self._last_observed_batch:
            raise ValueError("PUF batch was already observed")
        if batch is not self._pending_batch:
            raise ValueError("native targets require the pending PUF batch")
        if len(native_target_track_ids) != len(batch.rows):
            raise ValueError("native targets must align with PUF proposals")
        epsilon = float(self.config["epsilon"])
        normalized_targets = []
        for raw_targets in native_target_track_ids:
            if raw_targets is None:
                normalized_targets.append(None)
                continue
            targets = set()
            for value in raw_targets:
                if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
                    raise ValueError("native target ids must contain integers")
                if int(value) < 0:
                    raise ValueError("native target ids must be non-negative")
                targets.add(int(value))
            normalized_targets.append(frozenset(targets))

        for row, targets in zip(batch.rows, normalized_targets):
            if targets is None:
                self._stats["native_unresolved"] += 1
                continue
            if len(targets) > 1:
                self._stats["native_ambiguous"] += 1
                probability_by_id = {
                    candidate.track_id: candidate.probability
                    for candidate in row.candidates
                }
                self._stats["ambiguous_qim_coverage_any"] += int(
                    bool(targets.intersection(row.qim_candidate_track_ids))
                )
                self._stats["ambiguous_final_support_any"] += int(
                    any(probability_by_id.get(target, 0.0) > 0.0 for target in targets)
                )
                self._stats["ambiguous_top1_in_target_set"] += int(
                    row.valid and row.predicted_track_id in targets
                )
                if row.valid and row.predicted_birth:
                    # The exact winner is ambiguous, but this is unambiguously
                    # a history match rather than a birth.
                    self._stats["predicted_births_evaluated"] += 1
                    self._stats["false_births"] += 1
                    self._add_example(row, "ambiguous_track", min(targets))
                continue
            if not row.valid:
                self._stats["invalid_native_rows"] += 1

            probability_by_id = {
                candidate.track_id: candidate.probability for candidate in row.candidates
            }
            probabilities = [candidate.probability for candidate in row.candidates]
            if row.birth_probability is not None:
                probabilities.append(row.birth_probability)

            if not targets:
                self._stats["native_births"] += 1
                if row.valid and row.predicted_birth:
                    self._stats["birth_true_positives"] += 1
                    self._stats["native_decision_agreement"] += 1
                elif row.valid:
                    self._add_example(row, "birth", None)
                if row.valid and row.predicted_birth:
                    self._stats["predicted_births_evaluated"] += 1
                if row.valid and row.birth_probability is not None:
                    target_probability = max(row.birth_probability, epsilon)
                    self._stats["nll_total"] += -float(np.log(target_probability))
                    self._stats["brier_total"] += (
                        sum(value * value for value in probabilities)
                        - 2.0 * row.birth_probability
                        + 1.0
                    )
                    self._stats["proper_score_rows"] += 1
                continue

            target = next(iter(targets))
            self._stats["native_history_matches"] += 1
            qim_covered = target in row.qim_candidate_track_ids
            final_covered = probability_by_id.get(target, 0.0) > 0.0
            self._stats["qim_target_coverage_at_3"] += int(qim_covered)
            self._stats["post_fallback_target_coverage"] += int(final_covered)
            correct = row.valid and row.predicted_track_id == target
            self._stats["top1_native_agreement"] += int(correct)
            if final_covered:
                self._stats["conditional_top1_denominator"] += 1
                self._stats["conditional_top1_agreement"] += int(correct)
            if correct:
                self._stats["native_decision_agreement"] += 1
            elif row.valid and row.predicted_birth:
                self._stats["false_births"] += 1
                self._stats["predicted_births_evaluated"] += 1
                self._add_example(row, "track", target)
            elif row.valid and final_covered:
                self._stats["wrong_tracks"] += 1
                self._add_example(row, "track", target)
            elif row.valid:
                self._stats["retrieval_misses"] += 1
                self._add_example(row, "track", target)
            if row.valid:
                target_probability = max(probability_by_id.get(target, 0.0), epsilon)
                self._stats["nll_total"] += -float(np.log(target_probability))
                self._stats["brier_total"] += (
                    sum(value * value for value in probabilities)
                    - 2.0 * probability_by_id.get(target, 0.0)
                    + 1.0
                )
                self._stats["proper_score_rows"] += 1

        self._last_observed_batch = batch
        self._pending_batch = None

    def record_pipeline_timing(
        self,
        *,
        query_ms: Optional[float] = None,
        observe_ms: Optional[float] = None,
    ) -> None:
        if query_ms is None and observe_ms is None:
            raise ValueError("at least one pipeline timing value is required")
        for stage, value in (("query", query_ms), ("observe", observe_ms)):
            if value is None:
                continue
            timing = _finite_float(f"pipeline_{stage}_ms", value, 0.0)
            self._stats[f"pipeline_{stage}_calls"] += 1
            self._stats[f"pipeline_{stage}_ms_total"] += timing
            self._stats[f"pipeline_{stage}_ms_max"] = max(
                float(self._stats[f"pipeline_{stage}_ms_max"]), timing
            )

    @staticmethod
    def _rate(numerator: int, denominator: int) -> Optional[float]:
        return numerator / denominator if denominator else None

    @staticmethod
    def _percentile(values: Sequence[float], q: float) -> float:
        return float(np.percentile(np.asarray(values, dtype=np.float64), q)) if values else 0.0

    def summary(self) -> Dict[str, object]:
        stats = dict(self._stats)
        matches = int(stats["native_history_matches"])
        births = int(stats["native_births"])
        evaluated = matches + births
        proper = int(stats["proper_score_rows"])
        triggers = int(stats["fallback_triggers"])
        queries = int(stats["queries"])
        pipeline_queries = int(stats["pipeline_query_calls"])
        pipeline_observes = int(stats["pipeline_observe_calls"])
        stats.update(
            {
                "schema": "boxfusion.puf_lite_shadow.v1",
                "enabled": self.enabled,
                "observer_only": self.observer_only,
                "training_free": True,
                "causal": True,
                "online_update": False,
                "semantic_access": False,
                "semantic_mutation": False,
                "ground_truth_access": False,
                "detector_score_access": False,
                "scene_id": self._scene_id,
                "effective_config": dict(self.config),
                "qim_target_coverage_at_3_rate": self._rate(
                    int(stats["qim_target_coverage_at_3"]), matches
                ),
                "post_fallback_target_coverage_rate": self._rate(
                    int(stats["post_fallback_target_coverage"]), matches
                ),
                "top1_native_agreement_rate": self._rate(
                    int(stats["top1_native_agreement"]), matches
                ),
                "conditional_top1_agreement_rate": self._rate(
                    int(stats["conditional_top1_agreement"]),
                    int(stats["conditional_top1_denominator"]),
                ),
                "ambiguous_qim_coverage_any_rate": self._rate(
                    int(stats["ambiguous_qim_coverage_any"]),
                    int(stats["native_ambiguous"]),
                ),
                "ambiguous_final_support_any_rate": self._rate(
                    int(stats["ambiguous_final_support_any"]),
                    int(stats["native_ambiguous"]),
                ),
                "ambiguous_top1_in_target_set_rate": self._rate(
                    int(stats["ambiguous_top1_in_target_set"]),
                    int(stats["native_ambiguous"]),
                ),
                "native_decision_agreement_rate": self._rate(
                    int(stats["native_decision_agreement"]), evaluated
                ),
                "birth_recall": self._rate(
                    int(stats["birth_true_positives"]), births
                ),
                "birth_precision": self._rate(
                    int(stats["birth_true_positives"]),
                    int(stats["predicted_births_evaluated"]),
                ),
                "fallback_trigger_rate": self._rate(triggers, int(stats["proposals"])),
                "fallback_rescue_rate": self._rate(
                    int(stats["fallback_rescues"]), triggers
                ),
                "invalid_rate": self._rate(
                    int(stats["invalid_rows"]), int(stats["proposals"])
                ),
                "same_track_conflict_rate": self._rate(
                    int(stats["same_track_conflicts"]), int(stats["proposals"])
                ),
                "nll_mean": float(stats["nll_total"]) / proper if proper else None,
                "brier_mean": float(stats["brier_total"]) / proper if proper else None,
                "query_ms_mean": float(stats["query_ms_total"]) / queries if queries else 0.0,
                "query_ms_p95": self._percentile(self._query_samples, 95),
                "exhaustive_ms_p95": self._percentile(self._exhaustive_samples, 95),
                "pipeline_query_ms_mean": (
                    float(stats["pipeline_query_ms_total"]) / pipeline_queries
                    if pipeline_queries else 0.0
                ),
                "pipeline_observe_ms_mean": (
                    float(stats["pipeline_observe_ms_total"]) / pipeline_observes
                    if pipeline_observes else 0.0
                ),
                "diagnostic_examples": tuple(self._examples),
            }
        )
        return stats

    def summary_line(self) -> str:
        summary = self.summary()

        def rate(value: object) -> str:
            return "nan" if value is None else f"{float(value):.4f}"

        return (
            "PUF-lite shadow summary | "
            f"queries/proposals={summary['queries']}/{summary['proposals']}, "
            f"valid/invalid/conflict={summary['valid_rows']}/"
            f"{summary['invalid_rows']}/{summary['same_track_conflicts']}, "
            f"native_match/birth/ambiguous/unresolved="
            f"{summary['native_history_matches']}/{summary['native_births']}/"
            f"{summary['native_ambiguous']}/{summary['native_unresolved']}, "
            f"QIMcov3/finalcov/top1={rate(summary['qim_target_coverage_at_3_rate'])}/"
            f"{rate(summary['post_fallback_target_coverage_rate'])}/"
            f"{rate(summary['top1_native_agreement_rate'])}, "
            f"birth_P/R={rate(summary['birth_precision'])}/"
            f"{rate(summary['birth_recall'])}, "
            f"fallback_trigger/rescue={rate(summary['fallback_trigger_rate'])}/"
            f"{rate(summary['fallback_rescue_rate'])}, "
            f"query_mean/p95/max_ms={summary['query_ms_mean']:.3f}/"
            f"{summary['query_ms_p95']:.3f}/{summary['query_ms_max']:.3f}"
        )


def build_puf_lite(config: Mapping[str, object]) -> PUFLiteShadowObserver:
    if not isinstance(config, Mapping):
        raise ValueError("application config must be a mapping")
    return PUFLiteShadowObserver(config.get("puf_lite", {}))


__all__ = [
    "DEFAULT_PUF_LITE_CONFIG",
    "GeometryLikelihood",
    "PUFCandidatePosterior",
    "PUFLiteShadowObserver",
    "PUFProposalDecision",
    "PUFQueryBatch",
    "box_geometry_likelihood",
    "build_puf_lite",
    "normalize_puf_likelihoods",
    "resolve_puf_lite_config",
]
