"""Causal multi-view reliability reranking for native BoxFusion rows.

This module is an isolated v2 implementation.  It does not replace the
frozen running-max reranker used by the current official100 experiments.
Each native row keeps a bounded set of directionally diverse observations.
The row geometry and row count are never changed.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from boxfusion.causal_reliability import CausalReliabilityState
from boxfusion.online_candidate_map import (
    EvidenceFrame,
    NativeFrame,
    _aabb_iou,
    _center,
    _iou2d_matrix,
)


def _project_with_visibility(
    corners_world: np.ndarray,
    camera_to_world: np.ndarray,
    intrinsic: np.ndarray,
    width: int,
    height: int,
) -> tuple[np.ndarray, float] | None:
    """Project a 3D box and return its clipped rectangle and visible fraction."""
    world_to_camera = np.linalg.inv(camera_to_world)
    camera = (
        corners_world @ world_to_camera[:3, :3].T
        + world_to_camera[:3, 3]
    )
    if np.any(camera[:, 2] < 0.1):
        return None
    uvw = camera @ intrinsic[:3, :3].T
    uv = uvw[:, :2] / uvw[:, 2:3]
    low, high = uv.min(axis=0), uv.max(axis=0)
    raw_extent = np.maximum(high - low, 0.0)
    raw_area = float(np.prod(raw_extent))
    if raw_area <= 0.0:
        return None
    clipped_low = np.maximum(low, [0.0, 0.0])
    clipped_high = np.minimum(high, [float(width), float(height)])
    clipped_extent = np.maximum(clipped_high - clipped_low, 0.0)
    clipped_area = float(np.prod(clipped_extent))
    if clipped_area < 4.0:
        return None
    rectangle = np.r_[clipped_low, clipped_high].astype(np.float64)
    return rectangle, float(np.clip(clipped_area / raw_area, 0.0, 1.0))


def _geometry_consistency(left: np.ndarray, right: np.ndarray) -> float:
    """Bounded center/scale/overlap agreement between two 3D AABBs."""
    left_extent = np.ptp(left, axis=0)
    right_extent = np.ptp(right, axis=0)
    reference = max(float(np.linalg.norm(left_extent)), 1e-6)
    center_distance = float(np.linalg.norm(_center(left) - _center(right)))
    center_score = math.exp(-center_distance / reference)
    scale_error = float(
        np.mean(np.abs(np.log((right_extent + 1e-6) / (left_extent + 1e-6))))
    )
    scale_score = math.exp(-scale_error)
    overlap = _aabb_iou(left, right)
    return float(np.clip((center_score + scale_score + overlap) / 3.0, 0.0, 1.0))


@dataclass
class _GeometryState:
    corners: np.ndarray
    last_seen_ordinal: int


class OnlineNativeReliabilityReranker:
    """Rerank native rows with bounded causal multi-view reliability.

    ``support_mode`` selects an online aggregation operator for controlled
    comparisons.  ``reliability`` uses the beta-inspired lower bound;
    ``first``, ``mean``, ``max``, ``ema`` and ``diverse_max`` expose matched
    ablations under the same matching and evidence-strength definition.
    """

    MODES = {"native", "first", "mean", "max", "ema", "diverse_max", "reliability"}

    def __init__(
        self,
        *,
        support_mode: str = "reliability",
        match_iou: float = 0.10,
        support_threshold: float = 0.50,
        logit_weight: float = 2.0,
        geometry_reset_iou: float = 0.20,
        exclusive_matching: bool = True,
        use_negative_evidence: bool = True,
        max_view_slots: int = 8,
        min_angular_separation_deg: float = 30.0,
        state_ttl_keyframes: int = 10,
        max_states: int = 4096,
        ema_decay: float = 0.8,
        negative_weight: float = 0.25,
        lower_bound_kappa: float = 1.0,
    ) -> None:
        if support_mode not in self.MODES:
            raise ValueError(f"unknown support_mode {support_mode!r}")
        for name, value in (
            ("match_iou", match_iou),
            ("support_threshold", support_threshold),
            ("geometry_reset_iou", geometry_reset_iou),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must lie in [0,1]")
        if not math.isfinite(logit_weight) or logit_weight < 0.0:
            raise ValueError("logit_weight must be finite and nonnegative")
        self.support_mode = support_mode
        self.match_iou = float(match_iou)
        self.support_threshold = float(support_threshold)
        self.logit_weight = float(logit_weight)
        self.geometry_reset_iou = float(geometry_reset_iou)
        self.exclusive_matching = bool(exclusive_matching)
        self.use_negative_evidence = bool(use_negative_evidence)
        self.state_ttl_keyframes = int(state_ttl_keyframes)
        self.max_states = int(max_states)
        self.reliability = CausalReliabilityState(
            max_view_slots=max_view_slots,
            min_angular_separation_deg=min_angular_separation_deg,
            state_ttl_keyframes=state_ttl_keyframes,
            max_states=max_states,
            ema_decay=ema_decay,
            negative_weight=negative_weight,
            lower_bound_kappa=lower_bound_kappa,
        )
        self.geometry: dict[int, _GeometryState] = {}
        self.frames_seen = 0
        self.last_frame_id = -1
        self.last_ids = np.empty(0, dtype=np.int64)
        self.last_scores = np.empty(0, dtype=np.float64)
        self.last_support = np.empty(0, dtype=np.float64)
        self.last_frame_strength = np.empty(0, dtype=np.float64)
        self.last_frame_iou2d = np.empty(0, dtype=np.float64)
        self.visible_updates = 0
        self.matched_updates = 0
        self.geometry_resets = 0

    @staticmethod
    def _score(native_score: float, support: float, threshold: float, weight: float) -> float:
        clipped = min(max(float(native_score), 1e-4), 1.0 - 1e-4)
        logit = math.log(clipped / (1.0 - clipped))
        logit += weight * max(0.0, float(support) - threshold)
        return 1.0 / (1.0 + math.exp(-logit))

    def _support(self, identity: int, mode: str | None = None) -> float:
        selected = self.support_mode if mode is None else str(mode)
        if selected not in self.MODES:
            raise ValueError(f"unknown support_mode {selected!r}")
        if selected == "native":
            return 0.0
        summary = self.reliability.summary(identity)
        key = {
            "first": "first",
            "mean": "mean_support",
            "max": "max",
            "ema": "ema",
            "diverse_max": "diverse_max",
            "reliability": "lower",
        }[selected]
        return float(summary[key])

    def _prune_geometry(self, ordinal: int, current_ids: set[int]) -> None:
        expired = [
            identity
            for identity, state in self.geometry.items()
            if identity not in current_ids
            and ordinal - state.last_seen_ordinal > self.state_ttl_keyframes
        ]
        for identity in expired:
            del self.geometry[identity]

    def update(
        self,
        ordinal: int,
        frame_id: int,
        native: NativeFrame,
        evidence: EvidenceFrame,
    ) -> np.ndarray:
        if ordinal != self.frames_seen:
            raise ValueError(f"expected ordinal {self.frames_seen}, got {ordinal}")
        if frame_id <= self.last_frame_id:
            raise ValueError("frame_id must increase strictly")
        if len(native.ids) > self.max_states:
            raise RuntimeError("max_states is smaller than the current native map")

        current_ids = {int(value) for value in native.ids}
        self.reliability.prune(ordinal, current_ids)
        self._prune_geometry(ordinal, current_ids)

        projected_rows: list[int] = []
        projected_boxes: list[np.ndarray] = []
        visibility = np.zeros(len(native.ids), dtype=np.float64)
        for row, corners in enumerate(native.corners):
            identity = int(native.ids[row])
            previous = self.geometry.get(identity)
            if previous is not None and _aabb_iou(previous.corners, corners) < self.geometry_reset_iou:
                self.reliability.reset_evidence(identity, ordinal=ordinal)
                self.geometry_resets += 1
            self.geometry[identity] = _GeometryState(
                corners=np.array(corners, copy=True), last_seen_ordinal=int(ordinal)
            )
            projection = _project_with_visibility(
                corners,
                native.camera_to_world,
                native.intrinsic,
                native.width,
                native.height,
            )
            if projection is None:
                continue
            rectangle, fraction = projection
            projected_rows.append(row)
            projected_boxes.append(rectangle)
            visibility[row] = fraction

        assigned = np.full(len(native.ids), -1, dtype=np.int64)
        frame_iou2d = np.zeros(len(native.ids), dtype=np.float64)
        if projected_boxes and len(evidence.proposal_boxes_2d):
            overlap = _iou2d_matrix(
                np.asarray(projected_boxes), evidence.proposal_boxes_2d
            )
            if self.exclusive_matching:
                pairs = [
                    (float(overlap[r, c]), r, c)
                    for r, c in zip(*np.where(overlap >= self.match_iou))
                ]
                pairs.sort(key=lambda value: (-value[0], value[1], value[2]))
                used_rows: set[int] = set()
                used_proposals: set[int] = set()
                for value, projected_row, proposal_row in pairs:
                    if projected_row in used_rows or proposal_row in used_proposals:
                        continue
                    used_rows.add(projected_row)
                    used_proposals.add(proposal_row)
                    native_row = projected_rows[projected_row]
                    assigned[native_row] = proposal_row
                    frame_iou2d[native_row] = value
            else:
                best = overlap.argmax(axis=1)
                values = overlap[np.arange(len(overlap)), best]
                for projected_row, (proposal_row, value) in enumerate(zip(best, values)):
                    if value < self.match_iou:
                        continue
                    native_row = projected_rows[projected_row]
                    assigned[native_row] = int(proposal_row)
                    frame_iou2d[native_row] = float(value)

        frame_strength = np.zeros(len(native.ids), dtype=np.float64)
        camera_origin = native.camera_to_world[:3, 3]
        for row in projected_rows:
            identity = int(native.ids[row])
            direction = _center(native.corners[row]) - camera_origin
            proposal_row = int(assigned[row])
            matched = proposal_row >= 0
            strength = 0.0
            if matched:
                quality = float(evidence.proposal_scores[proposal_row])
                geometry = _geometry_consistency(
                    native.corners[row], evidence.proposal_corners[proposal_row]
                )
                strength = float(
                    np.clip(
                        (quality * visibility[row] * frame_iou2d[row] * geometry) ** 0.25,
                        0.0,
                        1.0,
                    )
                )
                self.matched_updates += 1
            if matched or self.use_negative_evidence:
                self.reliability.update(
                    identity,
                    frame_id=frame_id,
                    ordinal=ordinal,
                    view_direction=direction,
                    strength=strength,
                    visibility=float(visibility[row]),
                    matched=matched,
                )
                self.visible_updates += 1
            frame_strength[row] = strength

        support = np.asarray(
            [self._support(int(identity)) for identity in native.ids], dtype=np.float64
        )
        if self.support_mode == "native":
            scores = np.array(native.scores, copy=True)
        else:
            scores = np.asarray(
                [
                    self._score(score, value, self.support_threshold, self.logit_weight)
                    for score, value in zip(native.scores, support)
                ],
                dtype=np.float64,
            )
        self.frames_seen += 1
        self.last_frame_id = int(frame_id)
        self.last_ids = np.array(native.ids, copy=True)
        self.last_scores = np.array(scores, copy=True)
        self.last_support = support
        self.last_frame_strength = frame_strength
        self.last_frame_iou2d = frame_iou2d
        return np.array(scores, copy=True)

    def materialize(
        self, native_ids: np.ndarray, native_scores: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        return self.materialize_mode(native_ids, native_scores, self.support_mode)

    def materialize_mode(
        self,
        native_ids: np.ndarray,
        native_scores: np.ndarray,
        mode: str,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Read one aggregation arm from the same accumulated online state."""
        if mode not in self.MODES:
            raise ValueError(f"unknown support_mode {mode!r}")
        ids = np.asarray(native_ids, dtype=np.int64).reshape(-1)
        scores = np.asarray(native_scores, dtype=np.float64).reshape(-1)
        if len(ids) != len(scores):
            raise ValueError("terminal native IDs and scores must align")
        support = np.asarray(
            [self._support(int(identity), mode) for identity in ids],
            dtype=np.float64,
        )
        if mode == "native":
            return np.array(scores, copy=True), support
        reranked = np.asarray(
            [
                self._score(score, value, self.support_threshold, self.logit_weight)
                for score, value in zip(scores, support)
            ]
        )
        return reranked, support

    def materialize_all(
        self, native_ids: np.ndarray, native_scores: np.ndarray
    ) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        """Materialize all matched aggregation controls without replay."""
        return {
            mode: self.materialize_mode(native_ids, native_scores, mode)
            for mode in sorted(self.MODES)
        }

    def diagnostics(self) -> dict[str, object]:
        return {
            "frames_seen": self.frames_seen,
            "support_mode": self.support_mode,
            "exclusive_matching": self.exclusive_matching,
            "use_negative_evidence": self.use_negative_evidence,
            "visible_updates": self.visible_updates,
            "matched_updates": self.matched_updates,
            "geometry_resets": self.geometry_resets,
            "reliability": self.reliability.diagnostics(),
        }
