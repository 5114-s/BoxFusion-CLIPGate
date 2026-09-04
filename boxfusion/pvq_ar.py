"""PVQ-AR: view-indexed multi-prototype historical CLIP query for the local
rearrangement of ambiguous native 2D-matching edges.

Phase-1 contract (single-variable association experiment on top of the
Cbest route: Top-K3 + Boxer active + native real-score):

* Native intra-frame 3D NMS is untouched.
* Native 2D matching is untouched for unambiguous edges.
* Only ``proposal -> track`` assignment on Top-1/Top-2 ambiguous
  correspondence edges may change (local rearrangement), everything else
  falls back to the native decision (abstain).
* No proposal is added or removed, no score is modified, no Boxer
  single-frame corners are modified, no FastSAM/SAM/TSDF/point-cloud/
  covariance/global-Hungarian machinery is used.

Key differences from the retired global appearance gate and the failed
global causal Hungarian:

* The query is compared against up to ``K <= 4`` *view-compatible*
  historical prototypes of the candidate track (same side / similar
  visible face), selected by viewing-direction angular distance, instead
  of one aggregated global descriptor.
* A track without a compatible prototype is treated as missing evidence,
  never as negative evidence.
* Only a handful of ambiguous edges per keyframe are ever examined
  (bounded work, ``scene_event_cap`` safety cap).
"""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

DEFAULTS = {
    "enabled": False,
    "mode": "shadow",
    "max_prototypes": 4,
    "memory_per_track": 12,
    "ambiguity_margin": 0.10,
    "view_angle_max_deg": 60.0,
    "rearrange_margin": 0.05,
    "min_similarity": 0.50,
    "require_both_prototypes": True,
    "diagnostics_dir": None,
    "scene_event_cap": 4096,
    # Observer-only NMS merge logging (no behavior change).  Used to
    # quantify the absorbed-TP headroom of a potential NMS-stage PVQ-AR
    # before any retrieval logic is built for that hook point.
    "nms_observer": False,
    "nms_stage": None,
    # Observer-only per-keyframe box-map dump (world corners of every row
    # entering the spatial-association NMS).  Enables causal replay of
    # output-time modules (consensus support, birth dedup against the
    # map-so-far).  Never feeds back into association decisions.
    "keyframe_map_log": False,
}

NMS_STAGE_DEFAULTS = {
    "enabled": False,
    # Furniture-scale gate: the absorbed-TP headroom audit showed ~67% of
    # native NMS merges absorb unscored clutter (< 0.35 m).  Only
    # children at least this large are worth adjudicating.
    "min_child_dim": 0.35,
    # Above this parent-child IoU the geometry alone is decisive and no
    # appearance query is made.
    "iou_confident": 0.60,
    # Contrast decision rule (negative-prototype evidence): refuse the
    # native absorb only when some rival track's view-compatible
    # prototypes beat the parent's by at least this margin while also
    # clearing the absolute floor.
    "contrast_margin": 0.05,
    "min_similarity": 0.50,
    # When the parent exposes no compatible prototype at all, only refuse
    # if the geometric overlap is weak: high overlap means the boxes sit
    # on top of each other regardless of appearance memory.
    "iou_orphan_guard": 0.30,
    "max_events_per_keyframe": 16,
}


class PVQARConfigError(ValueError):
    """Raised for invalid association.pvq_ar configuration."""


@dataclass(frozen=True)
class Prototype:
    """One committed historical observation of a track."""

    init_id: int
    frame_id: int
    feature: np.ndarray  # [D] float32, L2-normalized CLIP image feature
    view_dir: np.ndarray  # [3] float64, camera-center -> box-center (world)
    score: float


@dataclass
class CandidateQuery:
    """Retrieval outcome for one candidate track of an ambiguity event."""

    track_row: int
    canonical_id: int
    iou: float
    margin: float
    score: float
    corners_world: np.ndarray  # [8, 3]
    compatible_count: int
    retrieved_count: int
    best_similarity: Optional[float]
    similarities: List[float]
    angles_deg: List[float]


def _resolve_nms_stage(raw) -> Dict:
    cfg = dict(NMS_STAGE_DEFAULTS)
    if raw is None:
        raw = {}
    if not hasattr(raw, "get"):
        raise PVQARConfigError("association.pvq_ar.nms_stage must be a mapping")
    unknown = set(raw) - set(NMS_STAGE_DEFAULTS)
    if unknown:
        raise PVQARConfigError(
            "unknown pvq_ar.nms_stage keys: " + ", ".join(sorted(unknown))
        )
    cfg.update(raw)
    cfg["enabled"] = bool(cfg["enabled"])
    if not cfg["enabled"]:
        return cfg
    for key, low, high in (
        ("min_child_dim", 0.0, 10.0),
        ("iou_confident", 0.0, 1.0),
        ("contrast_margin", 0.0, 1.0),
        ("min_similarity", -1.0, 1.0),
        ("iou_orphan_guard", 0.0, 1.0),
    ):
        value = float(cfg[key])
        if not low <= value <= high:
            raise PVQARConfigError(
                f"association.pvq_ar.nms_stage.{key} must be in [{low},{high}]"
            )
        cfg[key] = value
    cap = cfg["max_events_per_keyframe"]
    if not isinstance(cap, int) or isinstance(cap, bool) or cap < 1:
        raise PVQARConfigError(
            "association.pvq_ar.nms_stage.max_events_per_keyframe "
            "must be a positive int"
        )
    return cfg


def resolve_pvq_ar_config(raw) -> Dict:
    cfg = dict(DEFAULTS)
    if raw is None:
        raw = {}
    if not hasattr(raw, "get"):
        raise PVQARConfigError("association.pvq_ar must be a mapping")
    unknown = set(raw) - set(DEFAULTS)
    if unknown:
        raise PVQARConfigError(
            "unknown association.pvq_ar keys: " + ", ".join(sorted(unknown))
        )
    cfg.update(raw)
    cfg["enabled"] = bool(cfg["enabled"])
    if not cfg["enabled"]:
        return cfg
    if cfg["mode"] not in ("shadow", "active"):
        raise PVQARConfigError(
            "association.pvq_ar.mode must be 'shadow' or 'active'"
        )
    for key, low, high in (
        ("max_prototypes", 1, 4),
        ("memory_per_track", 1, 64),
    ):
        value = cfg[key]
        if not isinstance(value, int) or isinstance(value, bool):
            raise PVQARConfigError(f"association.pvq_ar.{key} must be an int")
        if not low <= value <= high:
            raise PVQARConfigError(
                f"association.pvq_ar.{key} must be in [{low},{high}]"
            )
    for key, low, high in (
        ("ambiguity_margin", 0.0, 1.0),
        ("view_angle_max_deg", 0.0, 180.0),
        ("rearrange_margin", 0.0, 1.0),
        ("min_similarity", -1.0, 1.0),
    ):
        value = float(cfg[key])
        if not low <= value <= high:
            raise PVQARConfigError(
                f"association.pvq_ar.{key} must be in [{low},{high}]"
            )
        cfg[key] = value
    cfg["require_both_prototypes"] = bool(cfg["require_both_prototypes"])
    cfg["nms_observer"] = bool(cfg["nms_observer"])
    cfg["keyframe_map_log"] = bool(cfg["keyframe_map_log"])
    cfg["nms_stage"] = _resolve_nms_stage(cfg["nms_stage"])
    diagnostics_dir = cfg["diagnostics_dir"]
    if not isinstance(diagnostics_dir, str) or not diagnostics_dir:
        raise PVQARConfigError(
            "association.pvq_ar.diagnostics_dir is required when enabled"
        )
    cap = cfg["scene_event_cap"]
    if not isinstance(cap, int) or isinstance(cap, bool) or cap < 1:
        raise PVQARConfigError("association.pvq_ar.scene_event_cap must be a positive int")
    return cfg


def _normalize_view_dir(direction: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(direction))
    if not np.isfinite(norm) or norm < 1.0e-9:
        raise ValueError("degenerate viewing direction")
    return (direction / norm).astype(np.float64)


def _angle_deg_between(first: np.ndarray, second: np.ndarray) -> float:
    cosine = float(np.dot(first, second))
    cosine = max(-1.0, min(1.0, cosine))
    return math.degrees(math.acos(cosine))


class PVQAR:
    """Shadow/active adjudicator for ambiguous correspondence edges."""

    def __init__(self, cfg) -> None:
        self.cfg = resolve_pvq_ar_config(
            (cfg.get("association", {}) or {}).get("pvq_ar", {})
            if hasattr(cfg, "get")
            else None
        )
        self.enabled = bool(self.cfg["enabled"])
        self.active = self.enabled and self.cfg["mode"] == "active"
        self.nms_stage = self.cfg.get("nms_stage") or _resolve_nms_stage(None)
        self.nms_stage_active = (
            self.enabled
            and self.active
            and bool(self.nms_stage["enabled"])
        )
        self.scene_id: Optional[str] = None
        self._observations: Dict[int, Prototype] = {}
        self._committed: Dict[int, List[Prototype]] = {}
        self._jsonl_path: Optional[str] = None
        self._jsonl_handle = None
        self._nms_jsonl_path: Optional[str] = None
        self._kfmap_jsonl_path: Optional[str] = None
        self._nms_jsonl_handle = None
        self._nms_records = 0
        self._nms_record_cap = 200000
        self._nms_ar_jsonl_path: Optional[str] = None
        self._nms_ar_jsonl_handle = None
        self._nms_ar_records = 0
        self._nms_ar_record_cap = 200000
        self._nms_events_this_keyframe = 0
        self.nms_stats = {
            "keyframes": 0,
            "events": 0,
            "gate_small_child": 0,
            "gate_confident_iou": 0,
            "gate_event_cap": 0,
            "decisions": {},
            "applied_refusals": 0,
            "retrieval_elapsed_ms": 0.0,
        }
        self.stats = {
            "keyframes": 0,
            "events": 0,
            "event_cap_hits": 0,
            "applied_rearrangements": 0,
            "shadow_rearrangements": 0,
            "retrieval_elapsed_ms": 0.0,
        }
        self._abstain_reasons: Dict[str, int] = {}

    # ------------------------------------------------------------------
    # Scene lifecycle
    # ------------------------------------------------------------------

    def bind_scene(self, scene_id: str) -> None:
        if not self.enabled:
            return
        if self.scene_id is not None and self.scene_id != scene_id:
            raise RuntimeError(
                "One PVQ-AR run cannot mix scenes: "
                f"{self.scene_id} != {scene_id}"
            )
        if self.scene_id == scene_id:
            return
        self.scene_id = scene_id
        self._observations = {}
        self._committed = {}
        self._close_jsonl()
        self._nms_records = 0
        root = self.cfg["diagnostics_dir"]
        os.makedirs(root, exist_ok=True)
        self._jsonl_path = os.path.join(root, f"{scene_id}_pvq_ar.jsonl")
        if self.cfg["nms_observer"]:
            self._kfmap_jsonl_path = os.path.join(
                root, f"{scene_id}_pvq_kfmap.jsonl"
            )
            self._nms_jsonl_path = os.path.join(
                root, f"{scene_id}_pvq_nms.jsonl"
            )
        else:
            self._nms_jsonl_path = None
        if self.nms_stage["enabled"]:
            self._nms_ar_jsonl_path = os.path.join(
                root, f"{scene_id}_pvq_nms_ar.jsonl"
            )
        else:
            self._nms_ar_jsonl_path = None

    def finalize(self) -> Dict:
        if not self.enabled:
            return {}
        self._close_jsonl()
        if self._nms_jsonl_handle is not None:
            self._nms_jsonl_handle.close()
            self._nms_jsonl_handle = None
        if self._nms_ar_jsonl_handle is not None:
            self._nms_ar_jsonl_handle.close()
            self._nms_ar_jsonl_handle = None
        summary = {
            "scene_id": self.scene_id,
            "mode": self.cfg["mode"],
            **{key: value for key, value in self.stats.items()},
            "abstain_reasons": dict(sorted(self._abstain_reasons.items())),
            "memory_tracks": len(self._committed),
            "memory_prototypes": sum(
                len(views) for views in self._committed.values()
            ),
        }
        if self._jsonl_path is not None:
            path = self._jsonl_path.replace("_pvq_ar.jsonl", "_pvq_ar_summary.json")
            with open(path, "w", encoding="utf-8") as handle:
                json.dump(summary, handle, indent=2, sort_keys=True)
        return summary

    def summary(self) -> str:
        if not self.enabled:
            return "PVQ-AR disabled"
        stats = self.stats
        text = (
            "PVQ-AR summary | "
            f"mode={self.cfg['mode']}, "
            f"keyframes={stats['keyframes']}, "
            f"events={stats['events']}, "
            f"applied={stats['applied_rearrangements']}, "
            f"shadow_rearrangements={stats['shadow_rearrangements']}, "
            f"event_cap_hits={stats['event_cap_hits']}, "
            f"retrieval_ms={stats['retrieval_elapsed_ms']:.3f}, "
            f"abstain={dict(sorted(self._abstain_reasons.items()))}"
        )
        if self.nms_stage["enabled"]:
            nms = self.nms_stats
            text += (
                " | NMS arbitration: "
                f"events={nms['events']}, "
                f"gate_small={nms['gate_small_child']}, "
                f"gate_confident={nms['gate_confident_iou']}, "
                f"gate_cap={nms['gate_event_cap']}, "
                f"decisions={dict(sorted(nms['decisions'].items()))}, "
                f"applied_refusals={nms['applied_refusals']}, "
                f"retrieval_ms={nms['retrieval_elapsed_ms']:.3f}"
            )
        return text

    # ------------------------------------------------------------------
    # Keyframe lifecycle
    # ------------------------------------------------------------------

    def begin_keyframe(self, box_manager) -> None:
        """Snapshot committed track memory before this keyframe commits.

        Must run after ``init_new_predictions`` but before STEP1 NMS so the
        snapshot only reflects observations from previous keyframes.
        """
        if not self.enabled:
            return
        self.stats["keyframes"] += 1
        self.nms_stats["keyframes"] += 1
        self._nms_events_this_keyframe = 0
        per_track_limit = int(self.cfg["memory_per_track"])
        committed: Dict[int, List[Prototype]] = {}
        for fusion_ids in box_manager.fusion_list:
            canonical = min(int(value) for value in fusion_ids)
            if canonical in committed:
                continue
            views: List[Prototype] = []
            for raw_init_id in sorted({int(value) for value in fusion_ids}):
                prototype = self._observations.get(raw_init_id)
                if prototype is not None:
                    views.append(prototype)
            views.sort(key=lambda view: view.frame_id)
            committed[canonical] = views[-per_track_limit:]
        self._committed = committed

    def record_observations(self, pred_instances) -> None:
        """Store this keyframe's proposal CLIP features for future queries."""
        if not self.enabled:
            return
        if not pred_instances.has("appearance_features"):
            raise RuntimeError(
                "PVQ-AR is enabled but appearance_features are absent; "
                "association feature extraction must run for every keyframe"
            )
        init_ids = pred_instances.init_id.detach().cpu().numpy()
        frame_ids = pred_instances.frame_id.detach().cpu().numpy()
        scores = pred_instances.scores.detach().cpu().numpy()
        features = pred_instances.appearance_features.detach().cpu().numpy()
        centers = (
            pred_instances.pred_boxes_3d.tensor.detach().cpu().numpy()[:, :3]
        )
        cam_centers = (
            pred_instances.cam_pose.detach().cpu().numpy()[:, :3, 3]
        )
        if features.ndim != 2 or features.shape[0] != len(init_ids):
            raise RuntimeError("PVQ-AR received malformed appearance features")
        for row in range(len(init_ids)):
            init_id = int(init_ids[row])
            if init_id in self._observations:
                raise RuntimeError(
                    f"PVQ-AR observed duplicate init_id {init_id}"
                )
            view_dir = _normalize_view_dir(
                np.asarray(centers[row], dtype=np.float64)
                - np.asarray(cam_centers[row], dtype=np.float64)
            )
            feature = np.asarray(features[row], dtype=np.float32)
            norm = float(np.linalg.norm(feature))
            if not np.isfinite(norm) or norm < 1.0e-9:
                continue
            self._observations[init_id] = Prototype(
                init_id=init_id,
                frame_id=int(frame_ids[row]),
                feature=feature / norm,
                view_dir=view_dir,
                score=float(scores[row]),
            )

    # ------------------------------------------------------------------
    # Adjudication
    # ------------------------------------------------------------------

    def _retrieve(
        self,
        canonical_id: int,
        query_feature: np.ndarray,
        query_view_dir: np.ndarray,
    ) -> CandidateQuery:
        views = self._committed.get(canonical_id, ())
        max_angle = float(self.cfg["view_angle_max_deg"])
        compatible = [
            (_angle_deg_between(view.view_dir, query_view_dir), view)
            for view in views
        ]
        compatible = [
            (angle, view)
            for angle, view in compatible
            if angle <= max_angle
        ]
        compatible.sort(key=lambda item: item[0])
        retrieved = compatible[: int(self.cfg["max_prototypes"])]
        similarities: List[float] = []
        angles: List[float] = []
        for angle, view in retrieved:
            similarities.append(float(np.dot(query_feature, view.feature)))
            angles.append(round(float(angle), 4))
        best = max(similarities) if similarities else None
        return CandidateQuery(
            track_row=-1,
            canonical_id=canonical_id,
            iou=float("nan"),
            margin=float("nan"),
            score=float("nan"),
            corners_world=np.zeros((8, 3), dtype=np.float64),
            compatible_count=len(compatible),
            retrieved_count=len(retrieved),
            best_similarity=best,
            similarities=[round(value, 6) for value in similarities],
            angles_deg=angles,
        )

    def adjudicate_ambiguity(
        self,
        *,
        frame_id: int,
        proposal_row: int,
        proposal_init_id: int,
        query_feature: np.ndarray,
        query_view_dir: np.ndarray,
        proposal_corners_world: np.ndarray,
        candidate_rows: Sequence[int],
        candidate_canonicals: Sequence[int],
        candidate_ious: Sequence[float],
        candidate_margins: Sequence[float],
        candidate_scores: Sequence[float],
        candidate_corners_world: Sequence[np.ndarray],
        box_manager=None,
    ) -> int:
        """Return 0 to keep the native Top-1 candidate, 1 for the Top-2.

        In shadow mode the return value is always 0 so the native decision
        and the saved predictions stay byte-identical to the baseline.
        """
        if len(candidate_rows) != 2:
            return 0
        started = time.perf_counter()
        query_feature = np.asarray(query_feature, dtype=np.float32).ravel()
        norm = float(np.linalg.norm(query_feature))
        if not np.isfinite(norm) or norm < 1.0e-9:
            self.stats["events"] += 1
            self._abstain_reasons["abstain_bad_query"] = (
                self._abstain_reasons.get("abstain_bad_query", 0) + 1
            )
            return 0
        query_feature = query_feature / norm
        query_view_dir = _normalize_view_dir(
            np.asarray(query_view_dir, dtype=np.float64)
        )

        queries = [
            self._retrieve(int(canonical), query_feature, query_view_dir)
            for canonical in candidate_canonicals
        ]
        for query, row, iou, margin, score, corners in zip(
            queries,
            candidate_rows,
            candidate_ious,
            candidate_margins,
            candidate_scores,
            candidate_corners_world,
        ):
            query.track_row = int(row)
            query.iou = float(iou)
            query.margin = float(margin)
            query.score = float(score)
            query.corners_world = np.asarray(
                corners, dtype=np.float64
            ).reshape(8, 3)

        native_sim = queries[0].best_similarity
        alternative_sim = queries[1].best_similarity
        if (
            self.cfg["require_both_prototypes"]
            and (native_sim is None or alternative_sim is None)
        ):
            reason = "abstain_missing_prototype"
            chosen = 0
        elif alternative_sim is None or alternative_sim < float(
            self.cfg["min_similarity"]
        ):
            reason = "abstain_low_similarity"
            chosen = 0
        elif (
            alternative_sim - (native_sim if native_sim is not None else -1.0)
        ) >= float(self.cfg["rearrange_margin"]):
            reason = "rearrange"
            chosen = 1
        else:
            reason = "native_better"
            chosen = 0

        applied = bool(self.active and chosen == 1)
        self.stats["events"] += 1
        if reason == "rearrange":
            if applied:
                self.stats["applied_rearrangements"] += 1
            else:
                self.stats["shadow_rearrangements"] += 1
        else:
            self._abstain_reasons[reason] = (
                self._abstain_reasons.get(reason, 0) + 1
            )
        self.stats["retrieval_elapsed_ms"] += (
            time.perf_counter() - started
        ) * 1000.0

        record = {
            "type": "ambiguity_event",
            "scene_id": self.scene_id,
            "frame_id": int(frame_id),
            "proposal_row": int(proposal_row),
            "proposal_init_id": int(proposal_init_id),
            "query_view_dir": [round(float(v), 6) for v in query_view_dir],
            "proposal_corners_world": np.asarray(
                proposal_corners_world, dtype=np.float64
            )
            .reshape(8, 3)
            .round(6)
            .tolist(),
            "ambiguity_margin_cfg": float(self.cfg["ambiguity_margin"]),
            "candidates": [
                {
                    "track_row": query.track_row,
                    "canonical_id": query.canonical_id,
                    "iou": round(query.iou, 6),
                    "margin": round(query.margin, 6),
                    "score": round(query.score, 6),
                    "compatible_count": query.compatible_count,
                    "retrieved_count": query.retrieved_count,
                    "best_similarity": (
                        None
                        if query.best_similarity is None
                        else round(query.best_similarity, 6)
                    ),
                    "prototype_similarities": query.similarities,
                    "prototype_angles_deg": query.angles_deg,
                    "corners_world": query.corners_world.round(6).tolist(),
                }
                for query in queries
            ],
            "reason": reason,
            "chosen": chosen,
            "applied": applied,
        }
        self._append_event(record)
        mode_tag = "ACTIVE" if self.active else "shadow"
        print(
            f"PVQ-AR {mode_tag} frame={frame_id} proposal={proposal_init_id} "
            f"native=track{queries[0].canonical_id}"
            f"(sim={queries[0].best_similarity}, "
            f"k={queries[0].retrieved_count}) "
            f"alt=track{queries[1].canonical_id}"
            f"(sim={queries[1].best_similarity}, "
            f"k={queries[1].retrieved_count}) "
            f"reason={reason} applied={applied}"
        )
        return 1 if applied else 0

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # NMS-stage arbitration (contrast-score decision rule)
    # ------------------------------------------------------------------

    def adjudicate_nms_absorb(
        self,
        *,
        keyframe_id: int,
        parent_row: int,
        child_row: int,
        parent_init_id: int,
        child_init_id: int,
        iou: float,
        parent_score: float,
        child_score: float,
        child_max_dim: float,
        query_feature: np.ndarray,
        query_view_dir: np.ndarray,
        parent_corners_world: np.ndarray,
        child_corners_world: np.ndarray,
        row_canonicals: Dict[int, int],
    ) -> bool:
        """Contrast-score arbitration of one native NMS absorb event.

        Returns True only in active mode when the merge should be refused
        (the child then survives independently and the native greedy loop
        can still assign it to a later, correct parent).  Shadow mode
        always returns False and only logs.
        """
        stage = self.nms_stage
        if not stage["enabled"]:
            return False
        if child_max_dim < float(stage["min_child_dim"]):
            self.nms_stats["gate_small_child"] += 1
            return False
        if iou >= float(stage["iou_confident"]):
            self.nms_stats["gate_confident_iou"] += 1
            return False
        if (
            self._nms_events_this_keyframe
            >= int(stage["max_events_per_keyframe"])
        ):
            self.nms_stats["gate_event_cap"] += 1
            return False
        self._nms_events_this_keyframe += 1
        self.nms_stats["events"] += 1

        started = time.perf_counter()
        feature = np.asarray(query_feature, dtype=np.float32).ravel()
        norm = float(np.linalg.norm(feature))
        if not np.isfinite(norm) or norm < 1.0e-9:
            self._record_nms_decision(
                "abstain_bad_query", keyframe_id, parent_row, child_row,
                parent_init_id, child_init_id, iou, parent_score,
                child_score, child_max_dim, None, None, None, None, None,
                None, None, parent_corners_world, child_corners_world,
            )
            return False
        feature = feature / norm
        try:
            view_dir = _normalize_view_dir(
                np.asarray(query_view_dir, dtype=np.float64)
            )
        except ValueError:
            self._record_nms_decision(
                "abstain_bad_query", keyframe_id, parent_row, child_row,
                parent_init_id, child_init_id, iou, parent_score,
                child_score, child_max_dim, None, None, None, None, None,
                None, None, parent_corners_world, child_corners_world,
            )
            return False

        parent_canonical = row_canonicals.get(parent_row)
        child_canonical = row_canonicals.get(child_row)

        # Negative-prototype contrast: the parent's claim is scored
        # against the best claim of every *other* live track.
        parent_query = (
            self._retrieve(parent_canonical, feature, view_dir)
            if parent_canonical is not None
            else None
        )
        best_rival = None  # (sim, row, canonical, retrieved, angles)
        for row, canonical in row_canonicals.items():
            if row in (parent_row, child_row):
                continue
            if canonical in (parent_canonical, child_canonical):
                continue
            rival = self._retrieve(canonical, feature, view_dir)
            if rival.best_similarity is None:
                continue
            if best_rival is None or (
                rival.best_similarity > best_rival[0]
            ):
                best_rival = (
                    rival.best_similarity,
                    row,
                    canonical,
                    rival.retrieved_count,
                    rival.angles_deg,
                )

        sim_parent = (
            parent_query.best_similarity
            if parent_query is not None
            else None
        )
        sim_rival = best_rival[0] if best_rival is not None else None
        parent_effective = sim_parent if sim_parent is not None else -1.0

        if sim_rival is None or sim_rival < float(stage["min_similarity"]):
            decision = "abstain_no_rival"
        elif (
            sim_parent is None
            and iou >= float(stage["iou_orphan_guard"])
        ):
            decision = "abstain_orphan_guard"
        elif (
            sim_rival - parent_effective >= float(stage["contrast_margin"])
        ):
            decision = "refuse_contest"
        else:
            decision = "abstain_parent_owns"

        applied = bool(
            self.nms_stage_active and decision == "refuse_contest"
        )
        if applied:
            self.nms_stats["applied_refusals"] += 1
        self.nms_stats["retrieval_elapsed_ms"] += (
            time.perf_counter() - started
        ) * 1000.0
        self._record_nms_decision(
            decision,
            keyframe_id,
            parent_row,
            child_row,
            parent_init_id,
            child_init_id,
            iou,
            parent_score,
            child_score,
            child_max_dim,
            sim_parent,
            sim_rival,
            best_rival[1] if best_rival else None,
            best_rival[2] if best_rival else None,
            best_rival[3] if best_rival else None,
            best_rival[4] if best_rival else None,
            parent_query.retrieved_count if parent_query else 0,
            parent_corners_world,
            child_corners_world,
        )
        return applied

    def _record_nms_decision(
        self,
        decision,
        keyframe_id,
        parent_row,
        child_row,
        parent_init_id,
        child_init_id,
        iou,
        parent_score,
        child_score,
        child_max_dim,
        sim_parent,
        sim_rival,
        rival_row,
        rival_canonical,
        rival_retrieved,
        rival_angles,
        parent_retrieved,
        parent_corners_world,
        child_corners_world,
    ):
        self.nms_stats["decisions"][decision] = (
            self.nms_stats["decisions"].get(decision, 0) + 1
        )
        if self._nms_ar_jsonl_path is None:
            return
        if self._nms_ar_records >= self._nms_ar_record_cap:
            return
        record = {
            "type": "nms_arbitration",
            "scene_id": self.scene_id,
            "keyframe_id": int(keyframe_id),
            "parent_row": int(parent_row),
            "child_row": int(child_row),
            "parent_init_id": int(parent_init_id),
            "child_init_id": int(child_init_id),
            "iou": round(float(iou), 6),
            "parent_score": round(float(parent_score), 6),
            "child_score": round(float(child_score), 6),
            "child_max_dim": round(float(child_max_dim), 6),
            "sim_parent": (
                None if sim_parent is None else round(float(sim_parent), 6)
            ),
            "parent_retrieved": int(parent_retrieved or 0),
            "sim_rival": (
                None if sim_rival is None else round(float(sim_rival), 6)
            ),
            "rival_row": None if rival_row is None else int(rival_row),
            "rival_canonical": (
                None if rival_canonical is None else int(rival_canonical)
            ),
            "rival_retrieved": (
                None if rival_retrieved is None else int(rival_retrieved)
            ),
            "rival_angles_deg": rival_angles,
            "decision": decision,
            "applied": bool(
                self.nms_stage_active and decision == "refuse_contest"
            ),
            "parent_corners_world": np.asarray(
                parent_corners_world, dtype=np.float64
            )
            .reshape(8, 3)
            .round(6)
            .tolist(),
            "child_corners_world": np.asarray(
                child_corners_world, dtype=np.float64
            )
            .reshape(8, 3)
            .round(6)
            .tolist(),
        }
        if self._nms_ar_jsonl_handle is None:
            self._nms_ar_jsonl_handle = open(
                self._nms_ar_jsonl_path, "w", encoding="utf-8"
            )
        self._nms_ar_jsonl_handle.write(
            json.dumps(record, sort_keys=True) + "\n"
        )
        self._nms_ar_records += 1
        print(
            f"PVQ-AR NMS {self.cfg['mode']} kf={keyframe_id} "
            f"parent={parent_init_id} child={child_init_id} "
            f"iou={iou:.3f} sim_p={record['sim_parent']} "
            f"sim_r={record['sim_rival']} -> {decision} "
            f"applied={record['applied']}"
        )

    def log_keyframe_map(
        self,
        *,
        keyframe_id: int,
        boxes: np.ndarray,
        scores,
        init_ids,
    ) -> None:
        """Observer-only dump of the box map entering this keyframe's SA.

        Causality anchor for output-time modules: consensus support and birth
        dedup must only ever see maps like this one (past + current keyframe).
        """
        if (
            not self.enabled
            or not self.cfg["keyframe_map_log"]
            or self._kfmap_jsonl_path is None
        ):
            return
        boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 8, 3)
        scores = np.asarray(scores, dtype=np.float64).ravel().tolist()
        init_ids = [int(v) for v in np.asarray(init_ids).ravel().tolist()]
        record = {
            "type": "kf_map",
            "scene_id": self.scene_id,
            "keyframe_id": int(keyframe_id),
            "n": int(len(boxes)),
            "boxes": np.round(boxes, 3).tolist(),
            "scores": [round(float(s), 4) for s in scores],
            "init_ids": init_ids,
        }
        with open(self._kfmap_jsonl_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, separators=(",", ":")) + "\n")

    def log_nms_merge(
        self,
        *,
        keyframe_id: int,
        parent_row: int,
        child_row: int,
        parent_init_id: int,
        child_init_id: int,
        parent_frame_id: int,
        child_frame_id: int,
        iou: float,
        parent_score: float,
        child_score: float,
        parent_corners_world: np.ndarray,
        child_corners_world: np.ndarray,
    ) -> None:
        """Observer-only log of one native NMS absorb decision (parent keeps,
        child merges in).  Never feeds back into association decisions."""
        if (
            not self.enabled
            or not self.cfg["nms_observer"]
            or self._nms_jsonl_path is None
        ):
            return
        if self._nms_records >= self._nms_record_cap:
            return
        record = {
            "type": "nms_merge",
            "scene_id": self.scene_id,
            "keyframe_id": int(keyframe_id),
            "parent_row": int(parent_row),
            "child_row": int(child_row),
            "parent_init_id": int(parent_init_id),
            "child_init_id": int(child_init_id),
            "parent_frame_id": int(parent_frame_id),
            "child_frame_id": int(child_frame_id),
            "iou": round(float(iou), 6),
            "parent_score": round(float(parent_score), 6),
            "child_score": round(float(child_score), 6),
            "parent_corners_world": np.asarray(
                parent_corners_world, dtype=np.float64
            )
            .reshape(8, 3)
            .round(6)
            .tolist(),
            "child_corners_world": np.asarray(
                child_corners_world, dtype=np.float64
            )
            .reshape(8, 3)
            .round(6)
            .tolist(),
        }
        if self._nms_jsonl_handle is None:
            self._nms_jsonl_handle = open(
                self._nms_jsonl_path, "w", encoding="utf-8"
            )
        self._nms_jsonl_handle.write(
            json.dumps(record, sort_keys=True) + "\n"
        )
        self._nms_records += 1

    def log_correspondence_edge(
        self,
        *,
        frame_id: int,
        proposal_init_id: int,
        top1_margin: float,
        top2_margin: Optional[float],
        top1_canonical: int,
        top2_canonical: Optional[int],
        accepted_count: int,
        ambiguous: bool,
    ) -> None:
        """Record every small-box correspondence edge for offline oracle.

        Observer-only: never feeds back into association decisions.  This
        lets the choice-set audit measure how often close runner-up tracks
        exist, including below the acceptance threshold, without widening
        the ambiguity definition the active module uses.
        """
        if not self.enabled or self._jsonl_path is None:
            return
        if self.stats["events"] > int(self.cfg["scene_event_cap"]):
            return
        record = {
            "type": "correspondence_edge",
            "scene_id": self.scene_id,
            "frame_id": int(frame_id),
            "proposal_init_id": int(proposal_init_id),
            "top1_margin": round(float(top1_margin), 6),
            "top2_margin": (
                None if top2_margin is None else round(float(top2_margin), 6)
            ),
            "top1_canonical": int(top1_canonical),
            "top2_canonical": (
                None if top2_canonical is None else int(top2_canonical)
            ),
            "accepted_count": int(accepted_count),
            "ambiguous": bool(ambiguous),
        }
        self._append_event(record)

    def _append_event(self, record: Dict) -> None:
        if self._jsonl_path is None:
            return
        if self.stats["events"] > int(self.cfg["scene_event_cap"]):
            self.stats["event_cap_hits"] += 1
            return
        if self._jsonl_handle is None:
            self._jsonl_handle = open(
                self._jsonl_path, "w", encoding="utf-8"
            )
        self._jsonl_handle.write(json.dumps(record, sort_keys=True) + "\n")
        self._jsonl_handle.flush()

    def _close_jsonl(self) -> None:
        if self._jsonl_handle is not None:
            self._jsonl_handle.close()
            self._jsonl_handle = None


def build_pvq_ar(cfg) -> PVQAR:
    return PVQAR(cfg)
