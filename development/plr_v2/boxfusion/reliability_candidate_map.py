"""Isolated full online composition for the reliability-v2 experiments."""
from __future__ import annotations

import time
from typing import Mapping

import numpy as np

from boxfusion.m1_anchor_reliability_online import OnlineReliableAnchorRecovery
from boxfusion.m1_anchor_online import OnlineAnchorRecovery
from boxfusion.native_reliability_reranker import OnlineNativeReliabilityReranker
from boxfusion.plr_v2 import OnlineProposalRecoveryV2
from boxfusion.online_candidate_map import (
    EvidenceFrame,
    NativeFrame,
    OnlineMapSnapshot,
    OnlineProposalRecovery,
)


SCHEMA = "boxfusion.reliability_candidate_map.v2"


class ReliabilityCandidateMap:
    """Compose frozen PLR with reliability-gated CALR-v2 and MVSR-v2."""

    def __init__(
        self,
        scene_id: str,
        *,
        m1p: OnlineProposalRecovery,
        m1a: OnlineReliableAnchorRecovery,
        m2: OnlineNativeReliabilityReranker,
        m1a_control: OnlineAnchorRecovery | None = None,
        plr_controls: Mapping[str, OnlineProposalRecoveryV2] | None = None,
        cross_branch_dedup: bool = False,
    ) -> None:
        self.scene_id = str(scene_id)
        self.m1p = m1p
        self.m1a = m1a
        self.m2 = m2
        self.m1a_control = m1a_control
        self.plr_controls = dict(plr_controls or {})
        self.cross_branch_dedup = bool(cross_branch_dedup)
        self.frames_seen = 0
        self.last_snapshot: OnlineMapSnapshot | None = None
        self.update_seconds: list[float] = []
        self.plr_update_seconds: dict[str, list[float]] = {
            name: [] for name in ("current", *sorted(self.plr_controls))
        }
        self.plr_previous_outputs: dict[str, set[int]] = {
            name: set() for name in self.plr_update_seconds
        }
        self.plr_output_jaccard: dict[str, list[float]] = {
            name: [] for name in self.plr_update_seconds
        }
        self.terminal_readout = False
        self.terminal_counts: dict[str, int] | None = None

    @staticmethod
    def _boxes(rows: list[dict[str, object]]) -> np.ndarray:
        if not rows:
            return np.empty((0, 8, 3), dtype=np.float64)
        return np.asarray([row["box"] for row in rows], dtype=np.float64)

    @staticmethod
    def _assemble(
        frame_id: int,
        native: NativeFrame,
        native_scores: np.ndarray,
        native_support: np.ndarray,
        proposal_rows: list[dict[str, object]],
        anchor_rows: list[dict[str, object]],
    ) -> OnlineMapSnapshot:
        boxes = [np.array(value, copy=True) for value in native.corners]
        scores = np.asarray(native_scores, dtype=np.float64).tolist()
        sources = ["native"] * len(boxes)
        source_ids = [int(value) for value in native.ids]
        for row in proposal_rows:
            boxes.append(np.array(row["box"], copy=True))
            scores.append(float(row["score"]))
            sources.append("m1p")
            source_ids.append(int(row["track_id"]))
        for row in anchor_rows:
            boxes.append(np.array(row["box"], copy=True))
            scores.append(float(row["score"]))
            sources.append("m1a")
            source_ids.append(int(row["track_id"]))
        return OnlineMapSnapshot(
            frame_id=int(frame_id),
            native_ids=np.array(native.ids, copy=True),
            native_corners=np.array(native.corners, copy=True),
            native_scores=np.array(native_scores, copy=True),
            native_support=np.array(native_support, copy=True),
            boxes=(
                np.stack(boxes)
                if boxes
                else np.empty((0, 8, 3), dtype=np.float64)
            ),
            scores=np.asarray(scores, dtype=np.float64),
            sources=tuple(sources),
            source_ids=tuple(source_ids),
        )

    def update(
        self, frame_id: int, native: NativeFrame, evidence: EvidenceFrame
    ) -> OnlineMapSnapshot:
        started = time.perf_counter()
        ordinal = self.frames_seen
        plr_started = time.perf_counter()
        self.m1p.update(
            ordinal,
            int(frame_id),
            proposal_ids=evidence.proposal_ids,
            proposal_corners=evidence.proposal_corners,
            proposal_scores=evidence.proposal_scores,
            child_ids=evidence.child_ids,
            child_corners=evidence.child_corners,
            child_scores=evidence.child_scores,
            native_corners=native.corners,
        )
        self.plr_update_seconds["current"].append(time.perf_counter() - plr_started)
        for name, state in sorted(self.plr_controls.items()):
            plr_started = time.perf_counter()
            state.update(
                ordinal,
                int(frame_id),
                proposal_ids=evidence.proposal_ids,
                proposal_corners=evidence.proposal_corners,
                proposal_scores=evidence.proposal_scores,
                native_corners=native.corners,
                camera_to_world=native.camera_to_world,
            )
            self.plr_update_seconds[name].append(time.perf_counter() - plr_started)
        proposal_rows = self.m1p.rows()
        plr_rows = {
            "current": proposal_rows,
            **{
                name: state.rows()
                for name, state in sorted(self.plr_controls.items())
            },
        }
        for name, rows in plr_rows.items():
            current = {int(row["track_id"]) for row in rows}
            previous = self.plr_previous_outputs[name]
            union = current | previous
            self.plr_output_jaccard[name].append(
                1.0 if not union else len(current & previous) / len(union)
            )
            self.plr_previous_outputs[name] = current
        anchor_rows = self.m1a.update(
            ordinal,
            int(frame_id),
            evidence.anchor_ids,
            evidence.anchor_corners,
            evidence.anchor_scores,
            native.camera_to_world,
            native_corners=native.corners,
            proposal_recovery_corners=(
                self._boxes(proposal_rows)
                if self.cross_branch_dedup
                else np.empty((0, 8, 3), dtype=np.float64)
            ),
        )
        if self.m1a_control is not None:
            self.m1a_control.update(
                ordinal,
                int(frame_id),
                evidence.anchor_ids,
                evidence.anchor_corners,
                evidence.anchor_scores,
            )
        native_scores = self.m2.update(
            ordinal, int(frame_id), native, evidence
        )
        snapshot = self._assemble(
            frame_id,
            native,
            native_scores,
            self.m2.last_support,
            proposal_rows,
            anchor_rows,
        )
        self.frames_seen += 1
        self.last_snapshot = snapshot
        self.update_seconds.append(time.perf_counter() - started)
        return snapshot

    @staticmethod
    def _component(
        rows: list[dict[str, object]],
    ) -> tuple[np.ndarray, np.ndarray]:
        if not rows:
            return (
                np.empty((0, 8, 3), dtype=np.float64),
                np.empty(0, dtype=np.float64),
            )
        return (
            np.asarray([row["box"] for row in rows], dtype=np.float64),
            np.asarray([row["score"] for row in rows], dtype=np.float64),
        )

    def experiment_components(
        self,
        native_ids: np.ndarray,
        native_corners: np.ndarray,
        native_scores: np.ndarray,
    ) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        """Read matched experiment arms from one causal online execution."""
        ids = np.asarray(native_ids, dtype=np.int64).reshape(-1)
        corners = np.asarray(native_corners, dtype=np.float64).reshape(-1, 8, 3)
        scores = np.asarray(native_scores, dtype=np.float64).reshape(-1)
        if not (len(ids) == len(corners) == len(scores)):
            raise ValueError("terminal native arrays must align")
        components: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for mode, (updated, _) in self.m2.materialize_all(ids, scores).items():
            components[f"native_{mode}"] = (
                np.array(corners, copy=True), np.array(updated, copy=True)
            )
        components["births_plr_v1"] = self._component(
            self.m1p.terminal_rows(corners)
        )
        for name, state in sorted(self.plr_controls.items()):
            components[f"births_plr_{name}"] = self._component(
                state.terminal_rows(corners)
            )
        components["births_calr_v2"] = self._component(
            self.m1a.rows(native_corners=corners)
        )
        if self.m1a_control is not None:
            components["births_calr_v1"] = self._component(
                self.m1a_control.rows()
            )
        return components

    def materialize_terminal(
        self,
        frame_id: int,
        native_ids: np.ndarray,
        native_corners: np.ndarray,
        native_scores: np.ndarray,
    ) -> OnlineMapSnapshot:
        ids = np.asarray(native_ids, dtype=np.int64).reshape(-1)
        corners = np.asarray(native_corners, dtype=np.float64).reshape(-1, 8, 3)
        scores = np.asarray(native_scores, dtype=np.float64).reshape(-1)
        if not (len(ids) == len(corners) == len(scores)):
            raise ValueError("terminal native arrays must align")
        reranked, support = self.m2.materialize(ids, scores)
        terminal_native = NativeFrame(
            ids=ids,
            corners=corners,
            scores=scores,
            camera_to_world=np.eye(4),
            intrinsic=np.eye(3),
            width=1,
            height=1,
        )
        proposal_rows = self.m1p.terminal_rows(corners)
        anchor_rows = self.m1a.rows(
            native_corners=corners,
            proposal_recovery_corners=self._boxes(proposal_rows),
        )
        snapshot = self._assemble(
            frame_id,
            terminal_native,
            reranked,
            support,
            proposal_rows,
            anchor_rows,
        )
        self.last_snapshot = snapshot
        self.terminal_readout = True
        self.terminal_counts = {
            source: snapshot.sources.count(source)
            for source in ("native", "m1p", "m1a")
        }
        self.terminal_counts["total"] = len(snapshot.sources)
        return snapshot

    def diagnostics(self) -> dict[str, object]:
        milliseconds = np.asarray(self.update_seconds) * 1000.0
        timing = {
            "count": len(milliseconds),
            "mean_ms": float(milliseconds.mean()) if len(milliseconds) else 0.0,
            "p50_ms": float(np.percentile(milliseconds, 50)) if len(milliseconds) else 0.0,
            "p95_ms": float(np.percentile(milliseconds, 95)) if len(milliseconds) else 0.0,
            "max_ms": float(milliseconds.max()) if len(milliseconds) else 0.0,
        }
        plr_timing = {}
        for name, values in self.plr_update_seconds.items():
            samples = np.asarray(values, dtype=np.float64) * 1000.0
            stability = np.asarray(self.plr_output_jaccard[name], dtype=np.float64)
            plr_timing[name] = {
                "count": len(samples),
                "mean_ms": float(samples.mean()) if len(samples) else 0.0,
                "p50_ms": float(np.percentile(samples, 50)) if len(samples) else 0.0,
                "p95_ms": float(np.percentile(samples, 95)) if len(samples) else 0.0,
                "max_ms": float(samples.max()) if len(samples) else 0.0,
                "output_jaccard_mean": (
                    float(stability.mean()) if len(stability) else 1.0
                ),
                "output_jaccard_min": (
                    float(stability.min()) if len(stability) else 1.0
                ),
            }
        return {
            "schema": SCHEMA,
            "scene_id": self.scene_id,
            "strictly_causal": True,
            "online_incremental": True,
            "uses_future_frames": False,
            "uses_full_scene_cache": False,
            "uses_terminal_map": self.terminal_readout,
            "uses_terminal_map_for_inference": False,
            "births_feed_back_into_native_association": False,
            "frames_seen": self.frames_seen,
            "terminal_counts": self.terminal_counts,
            "timing": timing,
            "plr_timing_and_stability": plr_timing,
            "m1p": self.m1p.diagnostics(),
            "plr_controls": {
                name: state.diagnostics()
                for name, state in sorted(self.plr_controls.items())
            },
            "m1a": self.m1a.diagnostics(),
            "m1a_control": (
                None if self.m1a_control is None else self.m1a_control.diagnostics()
            ),
            "m2": self.m2.diagnostics(),
            "cross_branch_dedup": self.cross_branch_dedup,
        }


def build_reliability_candidate_map(
    scene_id: str, config: Mapping[str, object] | None = None
) -> ReliabilityCandidateMap:
    section = {} if config is None else dict(config)
    section.pop("variant", None)
    m1p = dict(section.get("m1p", {}) or {})
    m1a = dict(section.get("m1a", {}) or {})
    m2 = dict(section.get("m2", {}) or {})
    audit_controls = bool(section.pop("audit_controls", False))
    audit_plr_controls = bool(section.pop("audit_plr_controls", False))
    plr_v2 = dict(section.pop("plr_v2", {}) or {})
    cross_branch_dedup = bool(section.pop("cross_branch_dedup", False))
    return ReliabilityCandidateMap(
        scene_id,
        m1p=OnlineProposalRecovery(**m1p),
        m1a=OnlineReliableAnchorRecovery(scene_id, **m1a),
        m2=OnlineNativeReliabilityReranker(**m2),
        plr_controls=(
            {
                stage: OnlineProposalRecoveryV2(
                    stage=stage,
                    **m1p,
                    **plr_v2,
                )
                for stage in ("assoc", "reliability", "score")
            }
            if audit_plr_controls
            else None
        ),
        m1a_control=(
            OnlineAnchorRecovery(
                scene_id,
                voxel_m=float(m1a.get("voxel_m", 0.3)),
                min_views=int(m1a.get("min_views", 3)),
                max_active=int(m1a.get("max_active_tracks", 4096)),
                max_births=int(m1a.get("max_births", 640)),
            )
            if audit_controls
            else None
        ),
        cross_branch_dedup=cross_branch_dedup,
    )
