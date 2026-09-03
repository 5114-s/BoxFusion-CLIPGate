"""Online, prediction-identity C3 Mask-RGBD observer.

The frozen C3 engineering replay selects terminal TR3D candidates with the
route ``source_rank <= 5 AND mask2_depth``.  This module evaluates the same
route incrementally with the *runtime* proposal provider and ScanNet depth.
It deliberately has no API that can return or mutate detections.

Candidate generation and its depth/DINO ordering remain the immutable C1
terminal-cache evidence.  Only Mask-RGBD confirmation is replayed online;
there is no extra DINO or SAM3 forward pass in this observer.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
import time
from typing import Any, Mapping, Sequence

import numpy as np

from .supplemental_proposals import SupplementalProposal
from .tr3d_c2_maskrgbd_cache import (
    load_sidecar,
    sha256_file,
    sidecar_path,
)
from .tr3d_c2_maskrgbd_observer import (
    C2Frame,
    C2MaskRGBDConfig,
    GATE_NAMES,
    observe_scene,
)
from .tr3d_residual_cache import (
    load_tr3d_residual_cache,
    tr3d_residual_cache_path,
)


SCHEMA = "boxfusion.tr3d_c3_online_identity_observer.v1"
ROUTE = "source_rank<=5 AND mask2_depth"
REJECTION_TELEMETRY_SCHEMA = "boxfusion.tr3d_c3_strong_predicates.v1"
_STRONG_PREDICATES = (
    "mask_score",
    "mask_containment",
    "box_coverage",
    "valid_depth_pixels",
    "inside_expanded",
    "component_points",
    "component_inside",
)
_MATCHED_MAX_METRICS = (
    "best_mask_score",
    "bbox_iou",
    "mask_containment",
    "box_coverage",
    "valid_depth_pixels",
    "inside_expanded",
    "component_points",
    "component_inside",
)
_SCENE_RE = re.compile(r"^scene[0-9]{4}_[0-9]{2}$")
_PREFIX_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")


def _readonly(value: Any, dtype: np.dtype) -> np.ndarray:
    result = np.ascontiguousarray(np.asarray(value, dtype=dtype))
    result.setflags(write=False)
    return result


def _array_sha256(value: Any) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def prediction_state_sha256(corners: Any, scores: Any) -> str:
    geometry = np.ascontiguousarray(np.asarray(corners, dtype=np.float32))
    confidence = np.ascontiguousarray(np.asarray(scores, dtype=np.float32))
    if geometry.size == 0:
        geometry = np.empty((0, 8, 3), dtype=np.float32)
    if confidence.size == 0:
        confidence = np.empty((0,), dtype=np.float32)
    if geometry.ndim != 3 or geometry.shape[1:] != (8, 3):
        raise ValueError("prediction corners must have shape [N,8,3]")
    if confidence.shape != (len(geometry),):
        raise ValueError("prediction scores must have shape [N]")
    if not np.isfinite(geometry).all() or not np.isfinite(confidence).all():
        raise ValueError("prediction state must be finite")
    return hashlib.sha256(
        (_array_sha256(geometry) + "|" + _array_sha256(confidence)).encode("ascii")
    ).hexdigest()


@dataclass(frozen=True)
class C3OnlineIdentityConfig:
    enabled: bool = False
    c2_cache_root: str = ""
    parent_cache_root: str = ""
    diagnostics_root: str = ""
    prefix_id: str = "p100"
    source_rank_max: int = 5
    gate_name: str = "mask2_depth"

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | None) -> "C3OnlineIdentityConfig":
        raw = dict(value or {})
        config = cls(
            enabled=bool(raw.get("enabled", False)),
            c2_cache_root=os.fspath(raw.get("c2_cache_root", "")),
            parent_cache_root=os.fspath(raw.get("parent_cache_root", "")),
            diagnostics_root=os.fspath(raw.get("diagnostics_root", "")),
            prefix_id=str(raw.get("prefix_id", "p100")),
            source_rank_max=int(raw.get("source_rank_max", 5)),
            gate_name=str(raw.get("gate_name", "mask2_depth")),
        )
        if not config.enabled:
            return config
        if _PREFIX_RE.fullmatch(config.prefix_id) is None:
            raise ValueError("C3 online observer prefix_id is invalid")
        if config.source_rank_max != 5 or config.gate_name != "mask2_depth":
            raise ValueError(
                "C3 online identity observer freezes source_rank_max=5 and "
                "gate_name=mask2_depth"
            )
        for name in ("c2_cache_root", "parent_cache_root"):
            path = Path(getattr(config, name))
            if not path.is_dir() or path.is_symlink():
                raise FileNotFoundError(f"C3 online observer {name}: {path}")
        if not config.diagnostics_root.strip():
            raise ValueError("C3 online observer diagnostics_root is required")
        return config


@dataclass
class _SceneState:
    scene_id: str
    c2_path: Path
    c2_sha256: str
    parent_path: Path
    parent_sha256: str
    candidate_boxes: np.ndarray
    proposal_ids: np.ndarray
    parent_rows: np.ndarray
    source_ranks: np.ndarray
    c1_track_scores: np.ndarray
    frozen_selected: np.ndarray
    c2_config: C2MaskRGBDConfig
    frame_ids: list[int]
    projected_count: np.ndarray
    matched_count: np.ndarray
    strong_count: np.ndarray
    total_component_points: np.ndarray
    sum_strong_inside: np.ndarray
    max_evidence: np.ndarray
    matched_predicate_pass_count: np.ndarray
    matched_predicate_fail_count: np.ndarray
    matched_metric_max: np.ndarray
    proposal_count: int = 0
    runtime_s: float = 0.0


class C3OnlineIdentityObserver:
    """Incremental online confirmation with an immutable output contract."""

    def __init__(self, config: C3OnlineIdentityConfig) -> None:
        if not isinstance(config, C3OnlineIdentityConfig):
            raise TypeError("config must be C3OnlineIdentityConfig")
        self.config = config
        self.enabled = bool(config.enabled)
        self._state: _SceneState | None = None
        self._last_summary: dict[str, Any] | None = None

    @classmethod
    def from_config(cls, cfg: Mapping[str, Any]) -> "C3OnlineIdentityObserver":
        return cls(
            C3OnlineIdentityConfig.from_mapping(
                cfg.get("tr3d_c3_online_observer", {})
            )
        )

    def _load_scene(self, scene_id: str) -> _SceneState:
        if _SCENE_RE.fullmatch(scene_id) is None:
            raise ValueError(f"invalid ScanNet scene id: {scene_id!r}")
        c2_path = sidecar_path(
            self.config.c2_cache_root, scene_id, self.config.prefix_id
        )
        c2 = load_sidecar(c2_path)
        parent_path = tr3d_residual_cache_path(
            self.config.parent_cache_root, scene_id, self.config.prefix_id
        )
        parent_sha = sha256_file(parent_path)
        if parent_sha != c2.parent_cache_sha256:
            raise ValueError(f"{scene_id}: C2/parent cache SHA256 mismatch")
        with np.load(parent_path, allow_pickle=False) as archive:
            checkpoint_sha = str(np.asarray(archive["checkpoint_sha256"]).item())
            parent_config_sha = str(np.asarray(archive["config_sha256"]).item())
        parent = load_tr3d_residual_cache(
            parent_path,
            expected_scene_id=scene_id,
            expected_prefix_id=self.config.prefix_id,
            expected_checkpoint_sha256=checkpoint_sha,
            expected_config_sha256=parent_config_sha,
        )
        if not np.array_equal(parent.proposal_ids[c2.parent_rows], c2.proposal_ids):
            raise ValueError(f"{scene_id}: C2 proposal identity disagrees with parent")
        source = np.asarray(c2.source_ranks <= self.config.source_rank_max)
        gate_index = GATE_NAMES.index(self.config.gate_name)
        boxes = _readonly(parent.boxes_world[c2.parent_rows[source]], np.float32)
        proposal_ids = _readonly(c2.proposal_ids[source], np.int64)
        count = len(proposal_ids)
        config_payload = json.loads(c2.config_json)
        c2_config = C2MaskRGBDConfig(**config_payload)
        return _SceneState(
            scene_id=scene_id,
            c2_path=c2_path.resolve(),
            c2_sha256=sha256_file(c2_path),
            parent_path=parent_path.resolve(),
            parent_sha256=parent_sha,
            candidate_boxes=boxes,
            proposal_ids=proposal_ids,
            parent_rows=_readonly(c2.parent_rows[source], np.int64),
            source_ranks=_readonly(c2.source_ranks[source], np.int32),
            c1_track_scores=_readonly(c2.c1_track_scores[source], np.float32),
            frozen_selected=_readonly(
                c2.observation.gate_mask[source, gate_index], np.bool_
            ),
            c2_config=c2_config,
            frame_ids=[],
            projected_count=np.zeros(count, dtype=np.int32),
            matched_count=np.zeros(count, dtype=np.int32),
            strong_count=np.zeros(count, dtype=np.int32),
            total_component_points=np.zeros(count, dtype=np.int64),
            sum_strong_inside=np.zeros(count, dtype=np.float64),
            max_evidence=np.zeros(count, dtype=np.float32),
            matched_predicate_pass_count=np.zeros(
                (count, len(_STRONG_PREDICATES)), dtype=np.int32
            ),
            matched_predicate_fail_count=np.zeros(
                (count, len(_STRONG_PREDICATES)), dtype=np.int32
            ),
            matched_metric_max=np.zeros(
                (count, len(_MATCHED_MAX_METRICS)), dtype=np.float64
            ),
        )

    def reset_scene(self, scene_id: str) -> None:
        if not self.enabled:
            return
        if self._state is not None and self._state.scene_id != scene_id:
            if self._last_summary is None:
                raise RuntimeError(
                    "C3 online observer scene changed before finalization"
                )
            self._state = None
        if self._state is None:
            self._state = self._load_scene(scene_id)
            self._last_summary = None

    def observe_keyframe(
        self,
        *,
        scene_id: str,
        frame_id: int,
        proposals: Sequence[SupplementalProposal],
        depth: Any,
        intrinsics: Any,
        camera_to_world: Any,
    ) -> None:
        if not self.enabled:
            return
        self.reset_scene(scene_id)
        state = self._state
        assert state is not None
        source_frame = int(frame_id)
        if source_frame < 0 or source_frame in state.frame_ids:
            raise ValueError("C3 online observer frame ids must be unique and nonnegative")
        proposal_tuple = tuple(proposals)
        if any(not isinstance(item, SupplementalProposal) for item in proposal_tuple):
            raise TypeError("C3 online observer requires SupplementalProposal rows")
        started = time.perf_counter()
        frame = C2Frame(
            frame_id=source_frame,
            depth_meters=np.asarray(depth, dtype=np.float32),
            intrinsics=np.asarray(intrinsics, dtype=np.float64),
            depth_camera_to_world=np.asarray(camera_to_world, dtype=np.float64),
            proposals=proposal_tuple,
            cache_sha256="online-runtime-no-cache",
        )
        observation = observe_scene(
            state.candidate_boxes, (frame,), state.c2_config
        )
        state.frame_ids.append(source_frame)
        state.proposal_count += len(proposal_tuple)
        state.projected_count += observation.projected_view_count
        state.matched_count += observation.matched_view_count
        state.strong_count += observation.strong_view_count
        state.total_component_points += np.where(
            observation.view_strong[:, 0],
            observation.component_point_count[:, 0],
            0.0,
        ).astype(np.int64)
        state.sum_strong_inside += np.where(
            observation.view_strong[:, 0],
            observation.inside_expanded_fraction[:, 0],
            0.0,
        )
        state.max_evidence = np.maximum(
            state.max_evidence, observation.max_evidence_score
        )
        matched = np.asarray(observation.view_matched[:, 0], dtype=np.bool_)
        predicate_pass = np.stack(
            (
                observation.best_mask_score[:, 0]
                >= state.c2_config.strong_mask_score,
                observation.mask_containment[:, 0]
                >= state.c2_config.min_mask_containment,
                observation.box_coverage[:, 0]
                >= state.c2_config.min_box_coverage,
                observation.valid_depth_pixels[:, 0]
                >= state.c2_config.min_valid_depth_pixels,
                observation.inside_expanded_fraction[:, 0]
                >= state.c2_config.min_inside_expanded_fraction,
                observation.component_point_count[:, 0]
                >= state.c2_config.min_component_points,
                observation.component_inside_fraction[:, 0]
                >= state.c2_config.min_component_inside_fraction,
            ),
            axis=1,
        )
        state.matched_predicate_pass_count += (
            matched[:, None] & predicate_pass
        ).astype(np.int32)
        state.matched_predicate_fail_count += (
            matched[:, None] & ~predicate_pass
        ).astype(np.int32)
        metric_values = np.stack(
            (
                observation.best_mask_score[:, 0],
                observation.bbox_iou[:, 0],
                observation.mask_containment[:, 0],
                observation.box_coverage[:, 0],
                observation.valid_depth_pixels[:, 0],
                observation.inside_expanded_fraction[:, 0],
                observation.component_point_count[:, 0],
                observation.component_inside_fraction[:, 0],
            ),
            axis=1,
        ).astype(np.float64)
        state.matched_metric_max = np.maximum(
            state.matched_metric_max,
            np.where(matched[:, None], metric_values, 0.0),
        )
        state.runtime_s += time.perf_counter() - started

    @staticmethod
    def _online_gate(state: _SceneState) -> tuple[np.ndarray, np.ndarray]:
        mean_inside = np.divide(
            state.sum_strong_inside,
            state.strong_count,
            out=np.zeros_like(state.sum_strong_inside),
            where=state.strong_count > 0,
        )
        gate = (
            (state.strong_count >= 2)
            & (
                state.total_component_points
                >= state.c2_config.mask2_min_total_component_points
            )
            & (
                mean_inside
                >= state.c2_config.mask2_min_mean_inside_expanded
            )
        )
        return gate.astype(np.bool_), mean_inside.astype(np.float32)

    @staticmethod
    def _write_create_only(path: Path, payload: Mapping[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
        temporary_name: str | None = None
        try:
            descriptor, temporary_name = tempfile.mkstemp(
                dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
            )
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(encoded)
                handle.flush()
                os.fsync(handle.fileno())
            os.link(temporary_name, path)
            path.chmod(0o444)
        except FileExistsError as error:
            raise FileExistsError(
                f"refusing existing C3 online diagnostic: {path}"
            ) from error
        finally:
            if temporary_name is not None:
                try:
                    os.unlink(temporary_name)
                except FileNotFoundError:
                    pass

    def finalize(
        self,
        *,
        scene_id: str,
        prediction_corners: Any,
        prediction_scores: Any,
    ) -> Mapping[str, Any]:
        if not self.enabled:
            return {"enabled": False}
        self.reset_scene(scene_id)
        state = self._state
        assert state is not None
        online_selected, mean_inside = self._online_gate(state)
        frozen = state.frozen_selected
        intersection = frozen & online_selected
        union = frozen | online_selected
        prediction_sha = prediction_state_sha256(
            prediction_corners, prediction_scores
        )
        runtimes = (
            state.runtime_s / len(state.frame_ids)
            if state.frame_ids
            else 0.0
        )
        candidates: list[dict[str, Any]] = []
        for index in range(len(state.proposal_ids)):
            identity_key = (
                f"{scene_id}:{self.config.prefix_id}:"
                f"{state.parent_sha256}:{int(state.proposal_ids[index])}"
            )
            candidates.append(
                {
                    "identity_key": identity_key,
                    "proposal_id": int(state.proposal_ids[index]),
                    "parent_row": int(state.parent_rows[index]),
                    "source_rank": int(state.source_ranks[index]),
                    "c1_depth_dino_track_score": float(
                        state.c1_track_scores[index]
                    ),
                    "frozen_sam3_mask2_depth": bool(frozen[index]),
                    "online_yoloe_mask2_depth": bool(online_selected[index]),
                    "projected_view_count": int(state.projected_count[index]),
                    "matched_view_count": int(state.matched_count[index]),
                    "strong_view_count": int(state.strong_count[index]),
                    "total_component_points": int(
                        state.total_component_points[index]
                    ),
                    "mean_strong_inside_expanded": float(mean_inside[index]),
                    "max_evidence_score": float(state.max_evidence[index]),
                    "strong_predicate_pass_counts": {
                        name: int(state.matched_predicate_pass_count[index, row])
                        for row, name in enumerate(_STRONG_PREDICATES)
                    },
                    "strong_predicate_fail_counts": {
                        name: int(state.matched_predicate_fail_count[index, row])
                        for row, name in enumerate(_STRONG_PREDICATES)
                    },
                    "matched_metric_max": {
                        name: float(state.matched_metric_max[index, row])
                        for row, name in enumerate(_MATCHED_MAX_METRICS)
                    },
                }
            )
        true_positive = int(np.count_nonzero(intersection))
        online_count = int(np.count_nonzero(online_selected))
        frozen_count = int(np.count_nonzero(frozen))
        false_positive = int(np.count_nonzero(online_selected & ~frozen))
        false_negative = int(np.count_nonzero(frozen & ~online_selected))
        true_negative = int(np.count_nonzero(~online_selected & ~frozen))
        candidate_count = len(state.proposal_ids)
        agreement = true_positive + true_negative
        summary: dict[str, Any] = {
            "schema": SCHEMA,
            "complete": True,
            "enabled": True,
            "observer_only": True,
            "mutation_enabled": False,
            "applied_count": 0,
            "scene_id": scene_id,
            "route": ROUTE,
            "comparison_scope": "frozen_c1_top5_online_yoloe_c2_gate",
            "source_rank_max": self.config.source_rank_max,
            "gate_name": self.config.gate_name,
            "class_agnostic": True,
            "ground_truth_access": False,
            "clip_access": False,
            "clip_semantics_unchanged": True,
            "teacher_labels_used_for_gate": False,
            "online_sam3_forward": False,
            "online_dino_forward": False,
            "frozen_c1_depth_dino_ranking_reused": True,
            "online_confirmation_provider": "runtime_yoloe_mask_real_depth",
            "offline_reference_confirmation_provider": "sam3_teacher_mask_real_depth",
            "candidate_generation": "immutable_terminal_tr3d_p100_cache",
            "candidate_generation_is_live": False,
            "observer_latency_authoritative": True,
            "provider_latency_included": False,
            "end_to_end_live_latency_authoritative": False,
            "rejection_telemetry_schema": REJECTION_TELEMETRY_SCHEMA,
            "strong_predicate_fail_counts": {
                name: int(state.matched_predicate_fail_count[:, row].sum())
                for row, name in enumerate(_STRONG_PREDICATES)
            },
            "prediction_state_before_sha256": prediction_sha,
            "prediction_state_after_sha256": prediction_sha,
            "prediction_identity": True,
            "prediction_count": int(np.asarray(prediction_scores).size),
            "provider_calls_observed": len(state.frame_ids),
            "provider_proposals_observed": state.proposal_count,
            "frame_ids": state.frame_ids,
            "candidate_count": candidate_count,
            "exact_identity_joined_count": candidate_count,
            "missing_identity_count": 0,
            "identity_coverage": 1.0,
            "frozen_selected_count": frozen_count,
            "online_selected_count": online_count,
            "intersection_count": true_positive,
            "true_positive_count": true_positive,
            "true_negative_count": true_negative,
            "false_positive_count": false_positive,
            "false_negative_count": false_negative,
            "out_of_universe_selected_count": 0,
            "selection_exact_match": bool(np.array_equal(frozen, online_selected)),
            "scene_exact_set": bool(np.array_equal(frozen, online_selected)),
            "selection_precision_vs_frozen": (
                float(true_positive / online_count) if online_count else 0.0
            ),
            "selection_recall_vs_frozen": (
                float(true_positive / frozen_count) if frozen_count else 0.0
            ),
            "route_coverage_vs_frozen": (
                float(true_positive / frozen_count) if frozen_count else 0.0
            ),
            "decision_agreement_conditional": (
                float(agreement / candidate_count) if candidate_count else 1.0
            ),
            "decision_agreement_e2e": (
                float(agreement / candidate_count) if candidate_count else 1.0
            ),
            "selection_jaccard": (
                float(true_positive / np.count_nonzero(union))
                if np.any(union)
                else 1.0
            ),
            "observer_runtime_s": float(state.runtime_s),
            "observer_mean_runtime_ms_per_provider_call": float(runtimes * 1000.0),
            "c2_sidecar": str(state.c2_path),
            "c2_sidecar_sha256": state.c2_sha256,
            "parent_cache": str(state.parent_path),
            "parent_cache_sha256": state.parent_sha256,
            "candidates": candidates,
        }
        destination = (
            Path(self.config.diagnostics_root).resolve()
            / f"{scene_id}_c3_online_identity.json"
        )
        self._write_create_only(destination, summary)
        self._last_summary = summary
        return summary

    @staticmethod
    def summary_text(summary: Mapping[str, Any]) -> str:
        if not summary.get("enabled", False):
            return "C3 online identity observer disabled"
        return (
            "C3 online identity summary | applied=0, "
            f"frames/proposals={summary['provider_calls_observed']}/"
            f"{summary['provider_proposals_observed']}, "
            f"frozen/online/intersection={summary['frozen_selected_count']}/"
            f"{summary['online_selected_count']}/"
            f"{summary['intersection_count']}, "
            f"precision/recall/jaccard="
            f"{summary['selection_precision_vs_frozen']:.3f}/"
            f"{summary['selection_recall_vs_frozen']:.3f}/"
            f"{summary['selection_jaccard']:.3f}, "
            f"observer_ms/call="
            f"{summary['observer_mean_runtime_ms_per_provider_call']:.3f}"
        )


def build_c3_online_identity_observer(
    cfg: Mapping[str, Any],
) -> C3OnlineIdentityObserver:
    return C3OnlineIdentityObserver.from_config(cfg)


__all__ = [
    "C3OnlineIdentityConfig",
    "C3OnlineIdentityObserver",
    "ROUTE",
    "SCHEMA",
    "build_c3_online_identity_observer",
    "prediction_state_sha256",
]
