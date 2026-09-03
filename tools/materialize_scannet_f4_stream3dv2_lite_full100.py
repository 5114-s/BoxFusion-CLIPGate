#!/usr/bin/env python3
"""Materialize F4-HB + past-only Stream3Dv2-lite on real-score Cbest.

The route is a sealed-cache replay: FastSAM/F2 supplies automatic-mask point
lifts, F4 supplies frozen Boxer HB hypotheses, L2/L3B supplies causal track
identity and the no-GT HB medoid, and the frozen SAM3 teacher cache supplies
sparse semantic/depth observations.  No annotation or evaluator API is
accepted by this program.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, dataclass, replace
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np
import cv2


ROOT = Path(__file__).resolve().parents[1]
if os.fspath(ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, os.fspath(ROOT))

from boxfusion.sam3_diverse_maskdepth_birth import (  # noqa: E402
    SAM3BirthConfig,
    SAM3TeacherView,
    confirm_candidate,
)
from boxfusion.probabilistic_tsdf_boxba import (  # noqa: E402
    PITSDFBoxBAConfig,
    refine_causal_track,
)
from boxfusion.sam2_tsdf_mv3dis_shadow import (  # noqa: E402
    LiftedMaskView,
    lift_mask_view,
)
from boxfusion.stream3dv2_lite import (  # noqa: E402
    TrackGeometry,
    TrackView,
    aabb_overlap,
    build_track_geometry,
    continuous_evidence_score,
    normalized_center_distance,
    points_inside_obb,
    policy_receipt,
    summarize_semantic_evidence,
)
from tools.audit_scannet_l0_f3_f4_perview_paper100_oracle import (  # noqa: E402
    _json as _sealed_json,
    _sha as _sealed_sha,
    _source_map,
)
from tools.materialize_scannet_raw_boxer_past3_birth_full100 import (  # noqa: E402
    BirthMaterializationError,
    NativePrediction,
    _load_native_prediction,
    _regular_file,
    _scene_list,
    _sha256,
    _write_json,
    _write_pickle,
)
from tools.materialize_scannet_sam3_diverse_clip_birth_full100 import (  # noqa: E402
    _teacher_views,
)


SCHEMA = "boxfusion.scannet_f4_stream3dv2_lite_full100.v1"
MANIFEST_NAME = "F4_STREAM3DV2_LITE_FULL100.json"
PITSDF_SCHEMA = "boxfusion.scannet_pitsdf_boxba_full100.v1"
PITSDF_MANIFEST_NAME = "PITSDF_BOXBA_FULL100.json"
OFFICIAL_SCENE_LIST_SHA256 = (
    "4b18fc586f7ad60cb17f41ee7d1d8b0ab1a0782917b5ae3519dd8ec90e7744d5"
)
L2_SCHEMA = "boxfusion.scannet_l2_source_preserving_paper100.seal.v1"
L3B_SCHEMA = "boxfusion.scannet_l3b_hbmedoid_t1_selector_paper100.shadow.v1"
F2_SCENE_SCHEMA = "boxfusion.scannet_fastsam_f2_paper100.scene.v1"
F2_ARRAY_SCHEMA = "boxfusion.scannet_fastsam_f2_paper100.evidence.v1"
F4_SCENE_SCHEMA = "boxfusion.scannet_fastsam_f4_boxer_paper100.scene.v1"

NATIVE_NOVELTY_AABB_IOU = 0.10
NATIVE_MAX_BIDIRECTIONAL_CONTAINMENT = 0.50
SELF_NMS_AABB_IOU = 0.15
SELF_NMS_BIDIRECTIONAL_CONTAINMENT = 0.25
PRESELECT_BIRTHS_PER_SCENE = 24
PRESELECT_OVERLAYS_PER_SCENE = 8
MAX_BIRTHS_PER_SCENE = 6
MAX_OVERLAYS_PER_SCENE = 1
APPENDED_CLASS_ID = 0
EVALUATOR_CONFIDENCE_THRESHOLD = 0.05
SCORE_EPSILON = 1.0e-6


class RouteError(BirthMaterializationError):
    pass


@dataclass
class Candidate:
    scene: str
    track_id: int
    geometry: TrackGeometry
    native_index: int | None
    native_overlap: tuple[float, float, float]
    native_nd: float
    native_volume_ratio: float
    semantic_receipt: dict[str, Any] | None = None
    semantic: dict[str, Any] | None = None
    evidence_score: float = 0.0
    append_score: float | None = None
    pitsdf_receipt: dict[str, Any] | None = None

    @property
    def pre_rank(self) -> tuple[Any, ...]:
        risk = max(self.native_overlap)
        return (
            self.geometry.preliminary_score * math.sqrt(max(1.0 - risk, 1.0e-4)),
            self.geometry.distinct_view_count,
            self.geometry.median_pairwise_hb_iou,
            self.geometry.hb_confidence_mean,
            -self.track_id,
        )

    @property
    def final_rank(self) -> tuple[Any, ...]:
        semantic = self.semantic or {}
        return (
            self.evidence_score,
            int(semantic.get("strong_view_count", 0)),
            int(semantic.get("matched_view_count", 0)),
            self.geometry.distinct_view_count,
            self.geometry.preliminary_score,
            -self.track_id,
        )


def _read_json(path: Path, label: str) -> dict[str, Any]:
    _regular_file(path, label)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RouteError(f"invalid {label}: {path}") from error
    if not isinstance(value, dict):
        raise RouteError(f"{label} must contain a JSON object")
    return value


def _track_ledgers(l2_path: Path, l3b_path: Path, scenes: Sequence[str]) -> tuple[Any, Any]:
    l2 = _read_json(l2_path, "L2 seal")
    l3b = _read_json(l3b_path, "L3B seal")
    if (
        l2.get("schema") != L2_SCHEMA
        or l2.get("complete") is not True
        or l2.get("overall_pass") is not True
        or l2.get("contracts", {}).get("ground_truth_access") is not False
        or tuple(l2.get("scene_order", ())) != tuple(scenes)
        or l3b.get("schema") != L3B_SCHEMA
        or l3b.get("complete") is not True
        or l3b.get("overall_pass") is not True
        or l3b.get("contracts", {}).get("ground_truth_access") is not False
    ):
        raise RouteError("sealed L2/L3B contract differs")
    if len(l2.get("scenes", ())) != len(scenes) or len(l3b.get("scenes", ())) != len(scenes):
        raise RouteError("sealed L2/L3B scene census differs")
    return l2, l3b


def _f2_source_rows(sidecar: Mapping[str, Any], scene: str) -> list[Mapping[str, Any]]:
    if (
        sidecar.get("schema") != F2_SCENE_SCHEMA
        or sidecar.get("scene_id") != scene
        or sidecar.get("complete") is not True
        or sidecar.get("contracts", {}).get("ground_truth_access") is not False
    ):
        raise RouteError(f"sealed F2 scene contract differs: {scene}")
    rows: list[Mapping[str, Any]] = []
    for frame in sidecar.get("frames", ()):
        if not isinstance(frame, Mapping):
            raise RouteError(f"invalid F2 frame: {scene}")
        sources = frame.get("sources")
        if not isinstance(sources, list):
            raise RouteError(f"invalid F2 sources: {scene}")
        for source in sources:
            if not isinstance(source, Mapping):
                raise RouteError(f"invalid F2 source: {scene}")
            rows.append(source)
    return rows


def _native_relation(corners: np.ndarray, native: NativePrediction) -> tuple[int | None, tuple[float, float, float], float, float]:
    if not len(native.corners):
        return None, (0.0, 0.0, 0.0), float("inf"), 1.0
    relations = [aabb_overlap(corners, row) for row in native.corners]
    index = max(
        range(len(relations)),
        key=lambda row: (
            relations[row][0],
            max(relations[row][1:]),
            -normalized_center_distance(corners, native.corners[row]),
            -row,
        ),
    )
    relation = tuple(float(value) for value in relations[index])
    nd = normalized_center_distance(corners, native.corners[index])
    candidate_volume = float(np.prod(np.ptp(corners, axis=0)))
    native_volume = float(np.prod(np.ptp(native.corners[index], axis=0)))
    return index, relation, nd, candidate_volume / max(native_volume, 1.0e-9)


def _is_native_novel(overlap: tuple[float, float, float]) -> bool:
    return (
        overlap[0] < NATIVE_NOVELTY_AABB_IOU
        and overlap[1] < NATIVE_MAX_BIDIRECTIONAL_CONTAINMENT
        and overlap[2] < NATIVE_MAX_BIDIRECTIONAL_CONTAINMENT
    )


def _semantic_enrich(candidate: Candidate, views: Sequence[SAM3TeacherView], config: SAM3BirthConfig) -> None:
    receipt = confirm_candidate(
        candidate.geometry.corners,
        candidate.geometry.decision_frame_id,
        views,
        config,
    )
    summary = summarize_semantic_evidence(receipt)
    semantic_or_none = summary if int(summary["selected_view_count"]) > 0 else None
    candidate.semantic_receipt = receipt
    candidate.semantic = semantic_or_none
    candidate.evidence_score = continuous_evidence_score(
        candidate.geometry,
        semantic_or_none,
        duplication_risk=max(candidate.native_overlap) if _is_native_novel(candidate.native_overlap) else 0.0,
    )


def _overlay_safe(candidate: Candidate, native: NativePrediction) -> tuple[bool, dict[str, Any]]:
    semantic = candidate.semantic or {}
    if candidate.native_index is None:
        return False, {"reason": "no_native"}
    native_corners = native.corners[candidate.native_index]
    points = candidate.geometry.refined_points
    candidate_inside = float(np.mean(points_inside_obb(points, candidate.geometry.corners)))
    lower, upper = native_corners.min(axis=0), native_corners.max(axis=0)
    native_inside = float(np.mean(np.all((points >= lower) & (points <= upper), axis=1)))
    checks = {
        "overlap": candidate.native_overlap[0] >= 0.15
        or max(candidate.native_overlap[1:]) >= 0.60,
        "not_near_duplicate": candidate.native_overlap[0] < 0.85,
        "center": candidate.native_nd <= 0.25,
        "volume": 0.50 <= candidate.native_volume_ratio <= 2.00,
        "multiview_geometry": candidate.geometry.distinct_view_count >= 2,
        "semantic_views": int(semantic.get("matched_view_count", 0)) >= 2,
        "semantic_consistency": float(semantic.get("label_consistency", 0.0)) >= 0.50,
        "point_fit_improves": candidate_inside >= native_inside + 0.05,
    }
    return all(checks.values()), {
        "checks": checks,
        "candidate_point_inside": candidate_inside,
        "native_point_inside": native_inside,
    }


def _self_overlap(left: Candidate, right: Candidate) -> bool:
    iou, left_in_right, right_in_left = aabb_overlap(
        left.geometry.corners, right.geometry.corners
    )
    return (
        iou >= SELF_NMS_AABB_IOU
        or left_in_right >= SELF_NMS_BIDIRECTIONAL_CONTAINMENT
        or right_in_left >= SELF_NMS_BIDIRECTIONAL_CONTAINMENT
    )


def _select_births(candidates: Sequence[Candidate]) -> tuple[list[Candidate], list[dict[str, Any]]]:
    selected: list[Candidate] = []
    decisions: list[dict[str, Any]] = []
    for candidate in sorted(candidates, key=lambda row: row.final_rank, reverse=True):
        decision = "accepted"
        if any(_self_overlap(candidate, prior) for prior in selected):
            decision = "self_nms"
        elif len(selected) >= MAX_BIRTHS_PER_SCENE:
            decision = "scene_cap"
        if decision == "accepted":
            selected.append(candidate)
        decisions.append({"track_id": candidate.track_id, "decision": decision})
    return selected, decisions


def _score_births(candidates: Sequence[Candidate], native_floor: float) -> None:
    ranked = sorted(candidates, key=lambda row: row.final_rank, reverse=True)
    if not ranked:
        return
    scores = np.linspace(
        native_floor - SCORE_EPSILON,
        EVALUATOR_CONFIDENCE_THRESHOLD + SCORE_EPSILON,
        len(ranked),
        dtype=np.float64,
    )
    if max(scores) >= native_floor or min(scores) <= EVALUATOR_CONFIDENCE_THRESHOLD:
        raise RouteError("low-score suffix interval is invalid")
    for candidate, score in zip(ranked, scores):
        candidate.append_score = float(score)


def _replace_row(row: Any, corners: np.ndarray) -> Any:
    values = [row[0], np.ascontiguousarray(corners, dtype=np.float32), row[2]]
    return tuple(values) if isinstance(row, tuple) else values


def _output_payload(native: NativePrediction, overlays: Sequence[Candidate], births: Sequence[Candidate]) -> Any:
    rows = list(native.rows)
    overlay_indices: set[int] = set()
    for candidate in overlays:
        if candidate.native_index is None or candidate.native_index in overlay_indices:
            raise RouteError("overlay target identity differs")
        overlay_indices.add(candidate.native_index)
        rows[candidate.native_index] = _replace_row(
            rows[candidate.native_index], candidate.geometry.corners
        )
    for index, (before, after) in enumerate(zip(native.rows, rows)):
        if before[0] != after[0] or type(before[2]) is not type(after[2]) or float(before[2]) != float(after[2]):
            raise RouteError(f"overlay changed native label/score/order at row {index}")
        if index not in overlay_indices and (
            np.asarray(before[1]).dtype != np.asarray(after[1]).dtype
            or np.asarray(before[1]).tobytes() != np.asarray(after[1]).tobytes()
        ):
            raise RouteError(f"non-overlay native geometry changed at row {index}")
    ordered_births = sorted(births, key=lambda row: float(row.append_score), reverse=True)
    rows.extend(
        (
            APPENDED_CLASS_ID,
            np.ascontiguousarray(candidate.geometry.corners, dtype=np.float32),
            float(candidate.append_score),
        )
        for candidate in ordered_births
    )
    row_container = tuple(rows) if isinstance(native.rows, tuple) else rows
    return (row_container,) if isinstance(native.payload, tuple) else [row_container]


def _candidate_report(candidate: Candidate) -> dict[str, Any]:
    geometry = candidate.geometry
    return {
        "track_id": candidate.track_id,
        "decision_frame_id": geometry.decision_frame_id,
        "source_ids": list(geometry.source_ids),
        "selected_source_ids": list(geometry.selected_source_ids),
        "hb_source_id": geometry.hb_source_id,
        "chosen_hypothesis": geometry.chosen_hypothesis,
        "distinct_view_count": geometry.distinct_view_count,
        "set_cover_fraction": geometry.set_cover_fraction,
        "median_pairwise_hb_iou": geometry.median_pairwise_hb_iou,
        "median_pairwise_hb_containment": geometry.median_pairwise_hb_containment,
        "hb_center_rms_m": geometry.hb_center_rms_m,
        "point_inside_hb_fraction": geometry.point_inside_hb_fraction,
        "pmr_seed_fraction": geometry.pmr_seed_fraction,
        "pmr_retained_fraction": geometry.pmr_retained_fraction,
        "preliminary_score": geometry.preliminary_score,
        "semantic": candidate.semantic,
        "semantic_receipt": candidate.semantic_receipt,
        "native_index": candidate.native_index,
        "native_overlap": list(candidate.native_overlap),
        "native_nd": candidate.native_nd,
        "native_volume_ratio": candidate.native_volume_ratio,
        "evidence_score": candidate.evidence_score,
        "append_score": candidate.append_score,
        "pitsdf_boxba": candidate.pitsdf_receipt,
        "corners_world": geometry.corners.tolist(),
    }


@dataclass
class _PITSDFSceneEvidence:
    """Lazily reconstruct only the views used by preselected candidates."""

    scene: str
    frames_root: Path
    source_index: Mapping[str, int]
    sources: Mapping[str, Mapping[str, Any]]
    frame_ids: np.ndarray
    masks_packbits: np.ndarray
    mask_shape: tuple[int, int]
    mask_bitorder: str
    intrinsic: np.ndarray
    cache: dict[str, LiftedMaskView | None]
    receipts: dict[Path, str]

    def load(self, source_id: str) -> LiftedMaskView | None:
        if source_id in self.cache:
            return self.cache[source_id]
        if source_id not in self.source_index or source_id not in self.sources:
            raise RouteError(f"PI-TSDF source identity is absent: {source_id}")
        index = int(self.source_index[source_id])
        source = self.sources[source_id]
        frame_id = int(source["frame_id"])
        if int(self.frame_ids[index]) != frame_id:
            raise RouteError(f"PI-TSDF frame identity differs: {source_id}")
        height, width = self.mask_shape
        flat = np.unpackbits(
            self.masks_packbits[index],
            bitorder=self.mask_bitorder,
            count=height * width,
        )
        mask = flat.reshape(height, width).astype(np.bool_, copy=False)
        scene_frames = self.frames_root / self.scene / "frames"
        depth_path = scene_frames / "depth" / f"{frame_id}.png"
        pose_path = scene_frames / "pose" / f"{frame_id}.txt"
        if not depth_path.is_file() or not pose_path.is_file():
            raise RouteError(f"PI-TSDF RGB-D/pose input is absent: {self.scene}/{frame_id}")
        depth_raw = cv2.imread(os.fspath(depth_path), cv2.IMREAD_UNCHANGED)
        if depth_raw is None or depth_raw.shape != (height, width):
            raise RouteError(f"PI-TSDF depth input differs: {self.scene}/{frame_id}")
        depth_m = np.asarray(depth_raw, dtype=np.float64) / 1000.0
        try:
            pose = np.loadtxt(pose_path, dtype=np.float64)
        except (OSError, ValueError) as error:
            raise RouteError(f"PI-TSDF pose input differs: {self.scene}/{frame_id}") from error
        if pose.shape != (4, 4) or not np.isfinite(pose).all():
            raise RouteError(f"PI-TSDF pose matrix differs: {self.scene}/{frame_id}")
        self.receipts.setdefault(depth_path, _sha256(depth_path))
        self.receipts.setdefault(pose_path, _sha256(pose_path))
        lifted = lift_mask_view(
            source_id=source_id,
            frame_id=frame_id,
            mask=mask,
            depth_m=depth_m,
            intrinsic=self.intrinsic,
            camera_to_world=pose,
        )
        self.cache[source_id] = lifted
        return lifted


def _apply_pitsdf_boxba(
    candidate: Candidate,
    *,
    native: NativePrediction,
    evidence: _PITSDFSceneEvidence,
    config: PITSDFBoxBAConfig,
) -> None:
    views = [
        row
        for source_id in candidate.geometry.source_ids
        if (row := evidence.load(str(source_id))) is not None
    ]
    views.sort(key=lambda row: (row.frame_id, row.source_id))
    if len(views) < config.minimum_track_views:
        candidate.pitsdf_receipt = {
            "schema": "boxfusion.probabilistic_tsdf_boxba.v1",
            "attempted": False,
            "accepted": False,
            "reason": "too_few_lifted_views",
            "available_view_count": len(views),
            "required_view_count": config.minimum_track_views,
            "contracts": {
                "ground_truth_access": False,
                "past_only_fit": True,
                "rollback_on_gate_failure": True,
            },
        }
        return
    boxer: dict[str, np.ndarray] = {}
    for view in views:
        raw = evidence.sources[view.source_id].get("hypotheses", {}).get("HB")
        if isinstance(raw, Mapping) and raw.get("valid") is True:
            boxer[view.source_id] = np.asarray(raw["world_corners"], dtype=np.float64)
    result = refine_causal_track(
        views=views,
        boxer_corners_by_source=boxer,
        baseline_corners=candidate.geometry.corners,
        config=config,
    )
    receipt = result.get("receipt")
    if not isinstance(receipt, Mapping):
        raise RouteError(f"PI-TSDF receipt differs: {candidate.scene}/{candidate.track_id}")
    flattened_receipt = dict(receipt)
    acceptance = receipt.get("acceptance", {})
    candidate_metrics = (
        acceptance.get("candidate", {}) if isinstance(acceptance, Mapping) else {}
    )
    flattened_receipt.update(
        {
            "attempted": "boxba" in receipt,
            "accepted": bool(result.get("accepted", False)),
            "reason": str(result.get("reason", "unknown")),
            "heldout_candidate_loss": candidate_metrics.get("loss"),
        }
    )
    candidate.pitsdf_receipt = flattened_receipt
    if bool(result.get("accepted", False)):
        corners = np.ascontiguousarray(result["output_corners"], dtype=np.float64)
        consensus = np.ascontiguousarray(result["consensus_points"], dtype=np.float64)
        if corners.shape != (8, 3) or not np.isfinite(corners).all() or not len(consensus):
            raise RouteError(
                f"PI-TSDF accepted geometry differs: {candidate.scene}/{candidate.track_id}"
            )
        corners.setflags(write=False)
        consensus.setflags(write=False)
        hypotheses = dict(candidate.geometry.hypotheses)
        hypotheses["PI_TSDF_BOXBA"] = corners
        qualities = dict(candidate.geometry.hypothesis_quality)
        heldout_loss = float(flattened_receipt.get("heldout_candidate_loss", 1.0))
        qualities["PI_TSDF_BOXBA"] = float(np.clip(1.0 - heldout_loss, 0.0, 1.0))
        candidate.geometry = replace(
            candidate.geometry,
            hypotheses=hypotheses,
            hypothesis_quality=qualities,
            chosen_hypothesis="PI_TSDF_BOXBA",
            corners=corners,
            refined_points=consensus,
        )
    (
        candidate.native_index,
        candidate.native_overlap,
        candidate.native_nd,
        candidate.native_volume_ratio,
    ) = _native_relation(candidate.geometry.corners, native)


def _refine_and_repool(
    birth_preselected: Sequence[Candidate],
    overlay_preselected: Sequence[Candidate],
    *,
    native: NativePrediction,
    evidence: _PITSDFSceneEvidence,
    config: PITSDFBoxBAConfig,
) -> tuple[list[Candidate], list[Candidate], Counter[str]]:
    unique: dict[int, Candidate] = {}
    for candidate in tuple(birth_preselected) + tuple(overlay_preselected):
        unique.setdefault(candidate.track_id, candidate)
    counts: Counter[str] = Counter()
    for candidate in unique.values():
        _apply_pitsdf_boxba(
            candidate,
            native=native,
            evidence=evidence,
            config=config,
        )
        receipt = candidate.pitsdf_receipt or {}
        counts["attempted"] += int(bool(receipt.get("attempted", False)))
        counts["accepted"] += int(bool(receipt.get("accepted", False)))
        counts[f"reason:{receipt.get('reason', 'unknown')}"] += 1
    births = [row for row in unique.values() if _is_native_novel(row.native_overlap)]
    overlays = [row for row in unique.values() if not _is_native_novel(row.native_overlap)]
    return (
        sorted(births, key=lambda row: row.pre_rank, reverse=True)[
            :PRESELECT_BIRTHS_PER_SCENE
        ],
        sorted(overlays, key=lambda row: row.pre_rank, reverse=True)[
            :PRESELECT_OVERLAYS_PER_SCENE
        ],
        counts,
    )


def _process_scene(
    *,
    scene: str,
    scene_index: int,
    l2_scene: Mapping[str, Any],
    l3b_scene: Mapping[str, Any],
    f2_root: Path,
    native: NativePrediction,
    teacher_views: Sequence[SAM3TeacherView],
    sam3_config: SAM3BirthConfig,
    pitsdf_config: PITSDFBoxBAConfig | None = None,
    frames_root: Path | None = None,
) -> tuple[list[Candidate], list[Candidate], dict[str, Any], list[tuple[Path, str]]]:
    if (
        l2_scene.get("scene_id") != scene
        or l2_scene.get("scene_index") != scene_index
        or l3b_scene.get("scene_id") != scene
        or l3b_scene.get("scene_index") != scene_index
    ):
        raise RouteError(f"sealed track scene order differs: {scene}")
    tracks = l2_scene.get("tracks")
    selections = l3b_scene.get("selections")
    f4_receipt = l2_scene.get("f4")
    source_order = l2_scene.get("f4_source_order")
    if (
        not isinstance(tracks, list)
        or not isinstance(selections, list)
        or len(tracks) != len(selections)
        or not isinstance(f4_receipt, Mapping)
        or not isinstance(source_order, list)
    ):
        raise RouteError(f"sealed track ledger differs: {scene}")

    f4_path = _regular_file(Path(str(f4_receipt.get("path", ""))), f"F4 scene {scene}")
    if _sealed_sha(f4_path) != f4_receipt.get("sha256"):
        raise RouteError(f"sealed F4 hash differs: {scene}")
    f4 = _sealed_json(f4_path, f"F4 scene {scene}")
    if f4.get("schema") != F4_SCENE_SCHEMA or f4.get("complete") is not True:
        raise RouteError(f"sealed F4 contract differs: {scene}")
    sources = _source_map(f4, scene)
    if tuple(str(value) for value in source_order) != tuple(sources):
        raise RouteError(f"F4/L2 source order differs: {scene}")
    frame_ordinal: dict[int, int] = {}
    for ordinal, frame in enumerate(f4.get("frames", ())):
        if not isinstance(frame, Mapping) or frame.get("frame_ordinal") != ordinal:
            raise RouteError(f"F4 frame order differs: {scene}")
        frame_ordinal[int(frame["frame_id"])] = ordinal

    f2_sidecar_path = _regular_file(
        f2_root / "scenes" / f"{scene}.json", f"F2 sidecar {scene}"
    )
    f2_array_path = _regular_file(
        f2_root / "arrays" / f"{scene}.npz", f"F2 arrays {scene}"
    )
    f2_sidecar_sha = _sha256(f2_sidecar_path)
    f2_array_sha = _sha256(f2_array_path)
    declared_inputs = f4.get("inputs", {})
    if (
        declared_inputs.get("f2_sidecar", {}).get("sha256") != f2_sidecar_sha
        or declared_inputs.get("f2_evidence", {}).get("sha256") != f2_array_sha
    ):
        raise RouteError(f"F4/F2 sealed hash differs: {scene}")
    f2_sidecar = _read_json(f2_sidecar_path, f"F2 sidecar {scene}")
    f2_rows = _f2_source_rows(f2_sidecar, scene)
    f2_confidence = {
        str(row["source_id"]): float(row["confidence"])
        for row in f2_rows
    }
    if tuple(f2_confidence) != tuple(sources):
        raise RouteError(f"F2/F4 source identity differs: {scene}")

    birth_pool: list[Candidate] = []
    overlay_pool: list[Candidate] = []
    rejection_counts: Counter[str] = Counter()
    hypothesis_counts: Counter[str] = Counter()
    with np.load(f2_array_path, allow_pickle=False) as archive:
        if (
            str(archive["schema"].item()) != F2_ARRAY_SCHEMA
            or str(archive["scene_id"].item()) != scene
            or tuple(str(value) for value in archive["source_ids"]) != tuple(sources)
        ):
            raise RouteError(f"sealed F2 array identity differs: {scene}")
        points_world = np.asarray(archive["points_world"], dtype=np.float64)
        point_offsets = np.asarray(archive["point_offsets"], dtype=np.int64)
        hlg_offsets = np.asarray(archive["hlg_index_offsets"], dtype=np.int64)
        hlg_indices = np.asarray(archive["hlg_retained_indices"], dtype=np.int64)
        if pitsdf_config is not None:
            masks_packbits = np.array(archive["masks_packbits"], dtype=np.uint8, copy=True)
            archive_frame_ids = np.array(archive["frame_ids"], dtype=np.int64, copy=True)
            mask_shape = tuple(int(value) for value in archive["mask_shape"])
            mask_bitorder = str(archive["mask_bitorder"].item())
            if (
                mask_shape != (480, 640)
                or mask_bitorder not in {"little", "big"}
                or masks_packbits.shape != (len(sources), 38_400)
                or archive_frame_ids.shape != (len(sources),)
            ):
                raise RouteError(f"sealed PI-TSDF mask evidence differs: {scene}")
        source_index = {source_id: index for index, source_id in enumerate(sources)}
        if (
            point_offsets.shape != (len(sources) + 1,)
            or hlg_offsets.shape != (len(sources) + 1,)
            or point_offsets[-1] != len(points_world)
            or np.any(point_offsets[1:] < point_offsets[:-1])
            or np.any(hlg_offsets[1:] < hlg_offsets[:-1])
        ):
            raise RouteError(f"sealed F2 point offsets differ: {scene}")

        for expected_track_id, (track, selection) in enumerate(zip(tracks, selections)):
            if (
                not isinstance(track, Mapping)
                or track.get("track_id") != expected_track_id
                or not isinstance(selection, Mapping)
                or selection.get("track_id") != expected_track_id
                or selection.get("hypothesis") != "HB"
                or selection.get("past_only_at_decision") is not True
            ):
                raise RouteError(f"sealed track selection differs: {scene}/{expected_track_id}")
            retained = track.get("retained_source_ids")
            if not isinstance(retained, list) or not retained:
                rejection_counts["no_retained_sources"] += 1
                continue
            views: list[TrackView] = []
            try:
                for source_id_raw in retained:
                    source_id = str(source_id_raw)
                    index = source_index[source_id]
                    start, end = int(point_offsets[index]), int(point_offsets[index + 1])
                    hstart, hend = int(hlg_offsets[index]), int(hlg_offsets[index + 1])
                    local_indices = hlg_indices[hstart:hend]
                    if len(local_indices):
                        if int(local_indices.min()) < 0 or int(local_indices.max()) >= end - start:
                            raise RouteError(f"HLG local index differs: {source_id}")
                        points = points_world[start:end][local_indices]
                    else:
                        points = points_world[start:end]
                    if not len(points):
                        raise ValueError("empty F2 point lift")
                    source = sources[source_id]
                    hb = source.get("hypotheses", {}).get("HB")
                    if not isinstance(hb, Mapping) or hb.get("valid") is not True:
                        raise ValueError("invalid HB")
                    frame_id = int(source["frame_id"])
                    views.append(
                        TrackView(
                            source_id=source_id,
                            frame_id=frame_id,
                            frame_ordinal=frame_ordinal[frame_id],
                            mask_confidence=f2_confidence[source_id],
                            hb_confidence=float(hb["confidence"]),
                            points_world=points,
                            hb_corners=np.asarray(hb["world_corners"], dtype=np.float64),
                        )
                    )
                geometry = build_track_geometry(
                    views,
                    preferred_hb_source_id=str(selection.get("source_id", "")),
                )
            except (KeyError, TypeError, ValueError, np.linalg.LinAlgError):
                rejection_counts["invalid_track_geometry"] += 1
                continue
            hypothesis_counts[geometry.chosen_hypothesis] += 1
            native_index, overlap, nd, volume_ratio = _native_relation(
                geometry.corners, native
            )
            candidate = Candidate(
                scene=scene,
                track_id=expected_track_id,
                geometry=geometry,
                native_index=native_index,
                native_overlap=overlap,
                native_nd=nd,
                native_volume_ratio=volume_ratio,
            )
            if _is_native_novel(overlap):
                birth_pool.append(candidate)
            else:
                overlay_pool.append(candidate)

    birth_preselected = sorted(
        birth_pool, key=lambda row: row.pre_rank, reverse=True
    )[:PRESELECT_BIRTHS_PER_SCENE]
    overlay_preselected = sorted(
        overlay_pool, key=lambda row: row.pre_rank, reverse=True
    )[:PRESELECT_OVERLAYS_PER_SCENE]
    pitsdf_counts: Counter[str] = Counter()
    pitsdf_receipts: dict[Path, str] = {}
    if pitsdf_config is not None:
        if frames_root is None:
            raise RouteError("PI-TSDF frames root is required")
        intrinsic_raw = f2_sidecar.get("intrinsic")
        if not isinstance(intrinsic_raw, Mapping):
            raise RouteError(f"sealed PI-TSDF intrinsic receipt differs: {scene}")
        intrinsic_path = _regular_file(
            Path(str(intrinsic_raw.get("path", ""))), f"PI-TSDF intrinsic {scene}"
        )
        intrinsic_sha = _sha256(intrinsic_path)
        if intrinsic_sha != intrinsic_raw.get("sha256"):
            raise RouteError(f"sealed PI-TSDF intrinsic hash differs: {scene}")
        try:
            intrinsic_full = np.loadtxt(intrinsic_path, dtype=np.float64)
        except (OSError, ValueError) as error:
            raise RouteError(f"sealed PI-TSDF intrinsic differs: {scene}") from error
        if intrinsic_full.shape not in {(3, 3), (4, 4)}:
            raise RouteError(f"sealed PI-TSDF intrinsic shape differs: {scene}")
        intrinsic = np.ascontiguousarray(intrinsic_full[:3, :3], dtype=np.float64)
        evidence = _PITSDFSceneEvidence(
            scene=scene,
            frames_root=frames_root,
            source_index=source_index,
            sources=sources,
            frame_ids=archive_frame_ids,
            masks_packbits=masks_packbits,
            mask_shape=mask_shape,
            mask_bitorder=mask_bitorder,
            intrinsic=intrinsic,
            cache={},
            receipts={intrinsic_path: intrinsic_sha},
        )
        birth_preselected, overlay_preselected, pitsdf_counts = _refine_and_repool(
            birth_preselected,
            overlay_preselected,
            native=native,
            evidence=evidence,
            config=pitsdf_config,
        )
        pitsdf_receipts.update(evidence.receipts)
    evaluated: dict[int, Candidate] = {}
    for candidate in birth_preselected + overlay_preselected:
        if candidate.track_id not in evaluated:
            _semantic_enrich(candidate, teacher_views, sam3_config)
            evaluated[candidate.track_id] = candidate
    birth_selected, birth_decisions = _select_births(birth_preselected)

    overlays: list[Candidate] = []
    overlay_decisions: list[dict[str, Any]] = []
    used_native: set[int] = set()
    for candidate in sorted(overlay_preselected, key=lambda row: row.final_rank, reverse=True):
        safe, diagnostics = _overlay_safe(candidate, native)
        decision = "accepted"
        if not safe:
            decision = "safety_abstain"
        elif candidate.native_index in used_native:
            decision = "native_already_overlaid"
        elif len(overlays) >= MAX_OVERLAYS_PER_SCENE:
            decision = "scene_cap"
        if decision == "accepted":
            assert candidate.native_index is not None
            used_native.add(candidate.native_index)
            overlays.append(candidate)
        overlay_decisions.append(
            {"track_id": candidate.track_id, "decision": decision, **diagnostics}
        )

    report = {
        "track_count": len(tracks),
        "native_count": len(native.rows),
        "birth_pool_count": len(birth_pool),
        "overlay_pool_count": len(overlay_pool),
        "birth_preselected_count": len(birth_preselected),
        "overlay_preselected_count": len(overlay_preselected),
        "birth_count": len(birth_selected),
        "overlay_count": len(overlays),
        "rejection_counts": dict(rejection_counts),
        "chosen_hypothesis_counts": dict(hypothesis_counts),
        "pitsdf_boxba_counts": dict(sorted(pitsdf_counts.items())),
        "birth_decisions": birth_decisions,
        "overlay_decisions": overlay_decisions,
        "births": [_candidate_report(row) for row in birth_selected],
        "overlays": [_candidate_report(row) for row in overlays],
    }
    receipts = [
        (f4_path, _sha256(f4_path)),
        (f2_sidecar_path, f2_sidecar_sha),
        (f2_array_path, f2_array_sha),
        *sorted(pitsdf_receipts.items(), key=lambda row: os.fspath(row[0])),
    ]
    return birth_selected, overlays, report, receipts


def materialize(args: argparse.Namespace) -> dict[str, Any]:
    pitsdf_enabled = bool(getattr(args, "pitsdf_boxba", False))
    pitsdf_config = PITSDFBoxBAConfig() if pitsdf_enabled else None
    frames_root = (
        Path(getattr(args, "frames_root")).resolve() if pitsdf_enabled else None
    )
    scene_list = args.scene_list.resolve()
    scenes = list(_scene_list(scene_list, args.expected_scene_count))
    if args.expected_scene_count == 100 and _sha256(scene_list) != OFFICIAL_SCENE_LIST_SHA256:
        raise RouteError("official100 scene-list hash mismatch")
    selected_scenes = scenes
    if args.scene is not None:
        if args.scene not in scenes:
            raise RouteError(f"scene outside requested list: {args.scene}")
        selected_scenes = [args.scene]
    if args.max_scenes is not None:
        if args.max_scenes < 1:
            raise RouteError("max-scenes must be positive")
        selected_scenes = selected_scenes[: args.max_scenes]
    selected_set = set(selected_scenes)

    output_root = args.output_root.resolve()
    if output_root.exists() or output_root.is_symlink():
        raise RouteError(f"refusing to overwrite output root: {output_root}")
    baseline_root = args.baseline_root.resolve()
    f2_root = args.f2_root.resolve()
    teacher_root = args.teacher_root.resolve()
    for path, label in (
        (baseline_root, "baseline root"),
        (f2_root, "F2 root"),
        (teacher_root, "SAM3 teacher root"),
    ):
        if path.is_symlink() or not path.is_dir():
            raise RouteError(f"{label} must be a non-symlink directory: {path}")
    if pitsdf_enabled and (frames_root is None or not frames_root.is_dir()):
        raise RouteError(f"PI-TSDF frames root must be a directory: {frames_root}")

    l2, l3b = _track_ledgers(args.l2_seal.resolve(), args.l3b_seal.resolve(), scenes)
    teacher_by_scene, teacher_provenance = _teacher_views(
        teacher_root, selected_scenes, _sha256(scene_list)
    )
    sam3_config = SAM3BirthConfig()
    # The suffix contract is global, not scene-local.  Resolve the floor from
    # the complete requested baseline even for a one-scene smoke replay.
    global_native_scores: list[float] = []
    for scene in scenes:
        path = _regular_file(
            baseline_root / f"{scene}_boxes.pkl", f"native prediction {scene}"
        )
        global_native_scores.extend(
            float(row[2]) for row in _load_native_prediction(path).rows
        )
    if not global_native_scores:
        raise RouteError("native predictions contain no scores")
    native_floor = min(global_native_scores)

    natives: dict[str, NativePrediction] = {}
    native_hashes: dict[str, str] = {}
    births_by_scene: dict[str, list[Candidate]] = {}
    overlays_by_scene: dict[str, list[Candidate]] = {}
    scene_reports: dict[str, Any] = {}
    input_receipts: dict[str, str] = {}
    l2_scenes = l2["scenes"]
    l3b_scenes = l3b["scenes"]
    selected_positions = [(index, scene) for index, scene in enumerate(scenes) if scene in selected_set]
    for position, (scene_index, scene) in enumerate(selected_positions, 1):
        native_path = _regular_file(
            baseline_root / f"{scene}_boxes.pkl", f"native prediction {scene}"
        )
        native = _load_native_prediction(native_path)
        natives[scene] = native
        native_hashes[scene] = _sha256(native_path)
        births, overlays, report, receipts = _process_scene(
            scene=scene,
            scene_index=scene_index,
            l2_scene=l2_scenes[scene_index],
            l3b_scene=l3b_scenes[scene_index],
            f2_root=f2_root,
            native=native,
            teacher_views=teacher_by_scene[scene],
            sam3_config=sam3_config,
            pitsdf_config=pitsdf_config,
            frames_root=frames_root,
        )
        births_by_scene[scene] = births
        overlays_by_scene[scene] = overlays
        scene_reports[scene] = report
        input_receipts.update({os.fspath(path): digest for path, digest in receipts})
        print(
            f"[{position}/{len(selected_positions)}] {scene}: "
            f"tracks={report['track_count']} pre={report['birth_preselected_count']} "
            f"birth={len(births)} overlay={len(overlays)}",
            flush=True,
        )
    flat_births = [row for scene in selected_scenes for row in births_by_scene[scene]]
    _score_births(flat_births, native_floor)

    output_root.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{output_root.name}.stage-", dir=output_root.parent))
    output_hashes: dict[str, str] = {}
    try:
        for scene in selected_scenes:
            payload = _output_payload(
                natives[scene], overlays_by_scene[scene], births_by_scene[scene]
            )
            path = stage / f"{scene}_boxes.pkl"
            _write_pickle(path, payload)
            reloaded = _load_native_prediction(path)
            if len(reloaded.rows) != len(natives[scene].rows) + len(births_by_scene[scene]):
                raise RouteError(f"output row count differs: {scene}")
            output_hashes[scene] = _sha256(path)
        for scene in selected_scenes:
            scene_reports[scene]["births"] = [
                _candidate_report(row) for row in births_by_scene[scene]
            ]
            scene_reports[scene]["overlays"] = [
                _candidate_report(row) for row in overlays_by_scene[scene]
            ]
        aggregate_pitsdf: Counter[str] = Counter()
        for report in scene_reports.values():
            aggregate_pitsdf.update(report.get("pitsdf_boxba_counts", {}))
        manifest = {
            "schema": PITSDF_SCHEMA if pitsdf_enabled else SCHEMA,
            "complete": len(selected_scenes) == args.expected_scene_count,
            "scene_count": len(selected_scenes),
            "native_count": sum(len(row.rows) for row in natives.values()),
            "birth_count": len(flat_births),
            "overlay_count": sum(len(rows) for rows in overlays_by_scene.values()),
            "output_count": sum(len(row.rows) for row in natives.values()) + len(flat_births),
            "score_mode": "real_native_scores_plus_strict_lower_unique_suffix",
            "native_score_floor": native_floor,
            "training_free": True,
            "target_dataset_training": False,
            "online_learning": False,
            "gt_access": False,
            "annotation_access": False,
            "evaluator_access": False,
            "past_only": True,
            "bounded_memory": True,
            "semantic_scope": "frozen_SAM3_ScanNet18_teacher_cache_only; class-agnostic fallback otherwise",
            "pipeline": [
                "F4 FastSAM automatic-mask sources and frozen Boxer HB",
                "L2 query-before-commit causal track identity",
                "20 scheduled-keyframe MVF-lite set cover",
                "sparse past-only SAM3 SDS-lite continuous evidence",
                "five-step voxel-graph PMR-lite",
                "HB / PMR-HB / PMR-YAW hypothesis selection",
                *(
                    [
                        "bounded probabilistic instance TSDF with free-space negative evidence",
                        "past-fit 7D BoxBA and last-past-view held-out rollback",
                    ]
                    if pitsdf_enabled
                    else []
                ),
                "safe near-native geometry overlay",
                "truly-unmatched real-score low-score append",
            ],
            "policy": {
                **policy_receipt(),
                "sam3": sam3_config.as_dict(),
                "pitsdf_boxba": (
                    asdict(pitsdf_config) if pitsdf_config is not None else None
                ),
                "preselect_births_per_scene": PRESELECT_BIRTHS_PER_SCENE,
                "preselect_overlays_per_scene": PRESELECT_OVERLAYS_PER_SCENE,
                "max_births_per_scene": MAX_BIRTHS_PER_SCENE,
                "max_overlays_per_scene": MAX_OVERLAYS_PER_SCENE,
                "native_novelty_aabb_iou": NATIVE_NOVELTY_AABB_IOU,
                "native_max_bidirectional_containment": NATIVE_MAX_BIDIRECTIONAL_CONTAINMENT,
                "self_nms_aabb_iou": SELF_NMS_AABB_IOU,
                "self_nms_bidirectional_containment": SELF_NMS_BIDIRECTIONAL_CONTAINMENT,
            },
            "inputs": {
                "scene_list": os.fspath(scene_list),
                "scene_list_sha256": _sha256(scene_list),
                "baseline_root": os.fspath(baseline_root),
                "f2_root": os.fspath(f2_root),
                "frames_root": None if frames_root is None else os.fspath(frames_root),
                "l2_seal": os.fspath(args.l2_seal.resolve()),
                "l2_seal_sha256": _sha256(args.l2_seal.resolve()),
                "l3b_seal": os.fspath(args.l3b_seal.resolve()),
                "l3b_seal_sha256": _sha256(args.l3b_seal.resolve()),
                "teacher": teacher_provenance,
                "source_receipts": input_receipts,
            },
            "native_prediction_sha256": native_hashes,
            "output_prediction_sha256": output_hashes,
            "pitsdf_boxba_counts": dict(sorted(aggregate_pitsdf.items())),
            "scenes": scene_reports,
        }
        _write_json(
            stage / (PITSDF_MANIFEST_NAME if pitsdf_enabled else MANIFEST_NAME),
            manifest,
        )
        os.replace(stage, output_root)
        return manifest
    except Exception:
        if stage.exists():
            shutil.rmtree(stage)
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scene-list", type=Path,
        default=ROOT / "evaluation/data_util/meta_data/scannetv2_val.txt",
    )
    parser.add_argument("--expected-scene-count", type=int, default=100)
    parser.add_argument(
        "--baseline-root", type=Path,
        default=ROOT / "results/scannet_t05_boxer_replay_active_score05",
    )
    parser.add_argument(
        "--f2-root", type=Path,
        default=ROOT / "logs/scannet_fastsam_f2_paper100_score05",
    )
    parser.add_argument(
        "--l2-seal", type=Path,
        default=ROOT / "logs/scannet_l2_source_preserving_paper100_score05/final/L2_SOURCE_PRESERVING_PAPER100.json",
    )
    parser.add_argument(
        "--l3b-seal", type=Path,
        default=ROOT / "logs/scannet_l3b_hbmedoid_t1_selector_paper100_score05/final/L3B_HBMEDOID_T1_SELECTOR_PAPER100.json",
    )
    parser.add_argument(
        "--teacher-root", type=Path,
        default=Path(
            "/data/ZhaoX/OVM3D-Dett/boxfusion_maskgraph_dev/cache/sam3_teacher/"
            "sam3_teacher_full100_c050_frozen_v1"
        ),
    )
    parser.add_argument(
        "--output-root", type=Path,
        default=ROOT / "results/scannet_cbest_real_score_f4_stream3dv2_lite_score05",
    )
    parser.add_argument(
        "--frames-root",
        type=Path,
        default=ROOT / "upstream_clean/scannet_readme_frames",
    )
    parser.add_argument(
        "--pitsdf-boxba",
        action="store_true",
        help="enable bounded PI-TSDF, free-space 7D BoxBA and causal held-out rollback",
    )
    parser.add_argument("--scene")
    parser.add_argument("--max-scenes", type=int)
    return parser


def main() -> int:
    args = _parser().parse_args()
    result = materialize(args)
    print(
        json.dumps(
            {
                "schema": result["schema"],
                "scene_count": result["scene_count"],
                "native_count": result["native_count"],
                "birth_count": result["birth_count"],
                "overlay_count": result["overlay_count"],
                "output_root": os.fspath(args.output_root.resolve()),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
