#!/usr/bin/env python3
"""Materialize the frozen R15 -> OVIR/OpenBox/MH-OBB real-score route.

The input R15 sidecar already contains past-only three-view mask-depth tracks.
This program consumes only the 160 no-GT native-novel shadow tracks, associates
them query-before-commit in a bounded causal memory, refines their stored mask
points against local sensor-depth components, chooses one of four robust box
hypotheses, and appends the final candidates below every native score.

No annotation path or evaluator API exists in this program.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Mapping, Sequence

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if os.fspath(ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, os.fspath(ROOT))

from boxfusion.r15_ovir_openbox_lite import (  # noqa: E402
    CausalObservation,
    aabb_bounds,
    aabb_corners,
    aabb_overlap,
    build_causal_memories,
    openbox_refine,
    select_multi_hypothesis_obb,
    voxel_downsample,
)
from tools.materialize_scannet_raw_boxer_past3_birth_full100 import (  # noqa: E402
    BirthMaterializationError,
    NativePrediction,
    _assert_native_prefix,
    _load_native_prediction,
    _regular_file,
    _scene_list,
    _sha256,
    _write_json,
    _write_pickle,
)
from tools.materialize_scannet_target_first_mobilesam_birth_full100 import (  # noqa: E402
    load_masklift_sidecar,
)


SCHEMA = "boxfusion.scannet_r15_ovir_openbox_mhobb_full100.v1"
R15_SCHEMA = "boxfusion.scannet_target_first_mobilesam_masklift_full100.v1"
MANIFEST_NAME = "R15_OVIR_OPENBOX_MHOBB_FULL100.json"
PREDICTION_SUFFIX = "_boxes.pkl"
OFFICIAL_SCENE_LIST_SHA256 = (
    "4b18fc586f7ad60cb17f41ee7d1d8b0ab1a0782917b5ae3519dd8ec90e7744d5"
)

# Frozen before AP access.  Values are sensor-resolution or inherited R15
# contracts, not target-label fits.
DEPTH_STRIDE = 2
MIN_DEPTH_M = 0.10
MAX_DEPTH_M = 8.00
DEPTH_VOXEL_M = 0.05
ROI_EXPANSION_M = 0.20
OPENBOX_PROXIMITY_M = 0.075
OVIR_MAX_MEMORIES = 64
OVIR_MAX_IDLE_FRAMES = 1000
OVIR_MAX_POINTS_PER_MEMORY = 32768
OVIR_MAX_VIEWS_PER_MEMORY = 12
OVIR_MAX_SOURCES_PER_MEMORY = 16
OVIR_MIN_ASSOCIATION_SCORE = 0.12
NATIVE_NOVELTY_AABB_IOU = 0.10
NATIVE_MAX_BIDIRECTIONAL_CONTAINMENT = 0.50
SELF_NMS_AABB_IOU = 0.15
SELF_NMS_BIDIRECTIONAL_CONTAINMENT = 0.25
MAX_BIRTHS_PER_SCENE = 4
APPENDED_CLASS_ID = 0
EVALUATOR_CONFIDENCE_THRESHOLD = 0.05
SCORE_EPSILON = 1.0e-6


class RouteError(BirthMaterializationError):
    pass


@dataclass
class Candidate:
    scene: str
    memory_id: int
    source_track_indices: tuple[int, ...]
    source_track_ids: tuple[int, ...]
    confirmation_frame_id: int
    evidence_frame_ids: tuple[int, ...]
    target_group: str
    corners: np.ndarray
    chosen_hypothesis: str
    hypothesis_rows: tuple[dict[str, Any], ...]
    openbox: dict[str, Any]
    source_median_mask_iou: float
    source_min_evidence_score: float
    source_mean_score: float
    source_supported_voxels: int
    max_native_iou: float
    max_candidate_in_native: float
    max_native_in_candidate: float
    append_score: float | None = None

    @property
    def quality_key(self) -> tuple[float, float, float, float, int, int, str, int]:
        chosen = next(row for row in self.hypothesis_rows if row["name"] == self.chosen_hypothesis)
        return (
            -float(chosen["evidence_score"]),
            -float(self.openbox["mutual_harmonic"]),
            -self.source_median_mask_iou,
            -self.source_min_evidence_score,
            -self.source_supported_voxels,
            self.confirmation_frame_id,
            self.scene,
            self.memory_id,
        )


def _hash_array(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(b"\0")
    digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode("ascii"))
    digest.update(b"\0")
    digest.update(array.tobytes())
    return digest.hexdigest()


def _read_json(path: Path, label: str) -> dict[str, Any]:
    path = _regular_file(path, label)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RouteError(f"invalid {label}: {path}") from error
    if not isinstance(payload, dict):
        raise RouteError(f"{label} must contain an object")
    return payload


def _load_intrinsic(scene_root: Path, scene: str) -> tuple[Path, np.ndarray]:
    path = _regular_file(
        scene_root / scene / "frames/intrinsic/intrinsic_depth.txt",
        f"depth intrinsic {scene}",
    )
    matrix = np.loadtxt(path, dtype=np.float64)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise RouteError(f"invalid depth intrinsic: {scene}")
    return path, matrix[:3, :3]


def _load_depth_world(
    scene_root: Path,
    scene: str,
    frame_id: int,
    intrinsic: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    frame_root = scene_root / scene / "frames"
    depth_path = _regular_file(
        frame_root / "depth" / f"{frame_id}.png", f"depth {scene}/{frame_id}"
    )
    pose_path = _regular_file(
        frame_root / "pose" / f"{frame_id}.txt", f"pose {scene}/{frame_id}"
    )
    depth = cv2.imread(os.fspath(depth_path), cv2.IMREAD_UNCHANGED)
    try:
        pose = np.loadtxt(pose_path, dtype=np.float64)
    except (OSError, ValueError) as error:
        raise RouteError(f"invalid pose {scene}/{frame_id}") from error
    if (
        depth is None
        or depth.ndim != 2
        or depth.dtype != np.uint16
        or pose.shape != (4, 4)
        or not np.isfinite(pose).all()
    ):
        raise RouteError(f"invalid RGB-D geometry {scene}/{frame_id}")
    rows = np.arange(0, depth.shape[0], DEPTH_STRIDE, dtype=np.int32)
    cols = np.arange(0, depth.shape[1], DEPTH_STRIDE, dtype=np.int32)
    uu, vv = np.meshgrid(cols, rows)
    z = depth[vv, uu].astype(np.float64) / 1000.0
    valid = np.isfinite(z) & (z >= MIN_DEPTH_M) & (z <= MAX_DEPTH_M)
    z = z[valid]
    u = uu[valid].astype(np.float64)
    v = vv[valid].astype(np.float64)
    camera = np.column_stack(
        (
            (u - intrinsic[0, 2]) * z / intrinsic[0, 0],
            (v - intrinsic[1, 2]) * z / intrinsic[1, 1],
            z,
        )
    )
    world = camera @ pose[:3, :3].T + pose[:3, 3]
    points = voxel_downsample(world, DEPTH_VOXEL_M)
    return points, {
        "depth_path": os.fspath(depth_path),
        "depth_sha256": _sha256(depth_path),
        "pose_path": os.fspath(pose_path),
        "pose_sha256": _sha256(pose_path),
        "valid_depth_samples": int(len(z)),
        "world_voxel_count": int(len(points)),
    }


def _view_box(center: np.ndarray, extent: np.ndarray) -> np.ndarray:
    lower = np.asarray(center, dtype=np.float64) - np.asarray(extent, dtype=np.float64) * 0.5
    upper = np.asarray(center, dtype=np.float64) + np.asarray(extent, dtype=np.float64) * 0.5
    return aabb_corners(lower, upper)


def _native_overlap(corners: np.ndarray, native_corners: np.ndarray) -> tuple[float, float, float]:
    if not len(native_corners):
        return 0.0, 0.0, 0.0
    values = [aabb_overlap(corners, native) for native in native_corners]
    return tuple(float(max(row[index] for row in values)) for index in range(3))  # type: ignore[return-value]


def _passes_native_novelty(overlap: tuple[float, float, float]) -> bool:
    return (
        overlap[0] < NATIVE_NOVELTY_AABB_IOU
        and overlap[1] < NATIVE_MAX_BIDIRECTIONAL_CONTAINMENT
        and overlap[2] < NATIVE_MAX_BIDIRECTIONAL_CONTAINMENT
    )


def _self_overlaps(left: Candidate, right: Candidate) -> bool:
    iou, left_in_right, right_in_left = aabb_overlap(left.corners, right.corners)
    return (
        iou >= SELF_NMS_AABB_IOU
        or left_in_right >= SELF_NMS_BIDIRECTIONAL_CONTAINMENT
        or right_in_left >= SELF_NMS_BIDIRECTIONAL_CONTAINMENT
    )


def _load_route_inputs(
    scene_list: Path,
    sidecar_path: Path,
    expected_scene_count: int,
) -> tuple[list[str], Any, dict[str, Any], dict[str, np.ndarray]]:
    scenes = _scene_list(scene_list, expected_scene_count)
    if expected_scene_count == 100 and _sha256(scene_list) != OFFICIAL_SCENE_LIST_SHA256:
        raise RouteError("official100 scene-list hash mismatch")
    sidecar = load_masklift_sidecar(sidecar_path, exact_schema=R15_SCHEMA)
    manifest = _read_json(sidecar.path, "R15 sidecar manifest")
    scene_order = manifest.get("scene_order")
    if not isinstance(scene_order, list) or tuple(scene_order) != tuple(scenes):
        raise RouteError("R15 sidecar scene order differs from official scene list")
    if sidecar.npz_path is None:
        raise RouteError("R15 sidecar lacks sealed NPZ geometry")
    with np.load(sidecar.npz_path, allow_pickle=False) as archive:
        arrays = {name: np.asarray(archive[name]) for name in archive.files}
    required = {
        "proposal_frame_id",
        "proposal_lift_center_world",
        "proposal_lift_extent_xyz",
        "proposal_scene_index",
        "proposal_source_row",
        "track_accepted_shadow",
        "track_evidence_global_rows",
        "track_fused_obb_corners",
        "track_fused_point_offsets",
        "track_fused_points_world",
        "track_scene_index",
    }
    if not required.issubset(arrays):
        raise RouteError(f"R15 NPZ misses arrays: {sorted(required - set(arrays))}")
    if int(np.count_nonzero(arrays["track_accepted_shadow"])) != 160:
        raise RouteError("frozen R15 native-novel population is not 160")
    return scenes, sidecar, manifest, arrays


def _scene_observations(
    scene: str,
    scene_index: int,
    sidecar: Any,
    arrays: Mapping[str, np.ndarray],
) -> tuple[list[CausalObservation], dict[int, Any]]:
    track_indices = np.flatnonzero(
        (arrays["track_scene_index"] == scene_index)
        & arrays["track_accepted_shadow"].astype(bool)
    )
    all_scene_indices = np.flatnonzero(arrays["track_scene_index"] == scene_index)
    if len(all_scene_indices) != len(sidecar.receipts[scene]):
        raise RouteError(f"R15 scene receipt count mismatch: {scene}")
    local_by_global = {int(value): index for index, value in enumerate(all_scene_indices)}
    receipts: dict[int, Any] = {}
    observations: list[CausalObservation] = []
    offsets = arrays["track_fused_point_offsets"]
    for track_index_raw in track_indices:
        track_index = int(track_index_raw)
        receipt = sidecar.receipts[scene][local_by_global[track_index]]
        start, end = int(offsets[track_index]), int(offsets[track_index + 1])
        points = np.asarray(arrays["track_fused_points_world"][start:end], dtype=np.float32)
        if not len(points):
            raise RouteError(f"accepted R15 track has no points: {scene}/{receipt.track_id}")
        proposal_rows = np.asarray(arrays["track_evidence_global_rows"][track_index], dtype=np.int64)
        proposal_frames = tuple(int(value) for value in arrays["proposal_frame_id"][proposal_rows])
        proposal_sources = tuple(int(value) for value in arrays["proposal_source_row"][proposal_rows])
        if (
            proposal_frames != receipt.evidence_frame_ids
            or proposal_sources != receipt.evidence_source_rows
            or np.any(arrays["proposal_scene_index"][proposal_rows] != scene_index)
        ):
            raise RouteError(f"R15 evidence identity mismatch: {scene}/{receipt.track_id}")
        view_corners = tuple(
            _view_box(
                arrays["proposal_lift_center_world"][row],
                arrays["proposal_lift_extent_xyz"][row],
            )
            for row in proposal_rows
        )
        corners = np.asarray(arrays["track_fused_obb_corners"][track_index], dtype=np.float32)
        observations.append(
            CausalObservation(
                source_index=track_index,
                confirmation_frame_id=receipt.confirmation_frame_id,
                target_group=str(receipt.target_group),
                points=points,
                corners=corners,
                evidence_frame_ids=receipt.evidence_frame_ids,
                view_corners=view_corners,
            )
        )
        receipts[track_index] = receipt
    return observations, receipts


def _process_scene(
    *,
    scene: str,
    scene_index: int,
    scene_root: Path,
    sidecar: Any,
    arrays: Mapping[str, np.ndarray],
    native: NativePrediction,
) -> tuple[list[Candidate], dict[str, Any]]:
    observations, receipt_by_track = _scene_observations(scene, scene_index, sidecar, arrays)
    memories, causal_audit = build_causal_memories(
        observations,
        voxel_m=DEPTH_VOXEL_M,
        max_memories=OVIR_MAX_MEMORIES,
        max_idle_frames=OVIR_MAX_IDLE_FRAMES,
        max_points_per_memory=OVIR_MAX_POINTS_PER_MEMORY,
        max_views_per_memory=OVIR_MAX_VIEWS_PER_MEMORY,
        max_sources_per_memory=OVIR_MAX_SOURCES_PER_MEMORY,
        min_association_score=OVIR_MIN_ASSOCIATION_SCORE,
    )
    if not all(bool(row["query_before_commit"]) for row in causal_audit):
        raise RouteError(f"query-before-commit audit failed: {scene}")
    intrinsic_path, intrinsic = _load_intrinsic(scene_root, scene)
    frame_cache: dict[int, np.ndarray] = {}
    frame_ledger: dict[str, Any] = {}
    candidates: list[Candidate] = []
    memory_rows: list[dict[str, Any]] = []
    for memory in memories:
        original = np.asarray(memory.corners, dtype=np.float32)
        lower, upper = aabb_bounds(original)
        roi_lower = lower - ROI_EXPANSION_M
        roi_upper = upper + ROI_EXPANSION_M
        context_blocks: list[np.ndarray] = []
        context_frame_ids = tuple(sorted(set(memory.evidence_frame_ids)))
        if any(frame_id > memory.last_confirmation_frame_id for frame_id in context_frame_ids):
            raise RouteError(f"future context frame reached causal memory: {scene}/{memory.memory_id}")
        for frame_id in context_frame_ids:
            if frame_id not in frame_cache:
                points, ledger = _load_depth_world(
                    scene_root, scene, frame_id, intrinsic
                )
                frame_cache[frame_id] = points
                frame_ledger[str(frame_id)] = ledger
            points = frame_cache[frame_id]
            inside = np.all((points >= roi_lower) & (points <= roi_upper), axis=1)
            if np.any(inside):
                context_blocks.append(points[inside])
        context = (
            voxel_downsample(np.concatenate(context_blocks, axis=0), DEPTH_VOXEL_M)
            if context_blocks
            else np.empty((0, 3), dtype=np.float32)
        )
        refinement = openbox_refine(
            memory.points,
            context,
            voxel_m=DEPTH_VOXEL_M,
            proximity_m=OPENBOX_PROXIMITY_M,
            context_frame_ids=context_frame_ids,
            cutoff_frame_id=memory.last_confirmation_frame_id,
        )
        selection = select_multi_hypothesis_obb(
            memory.points,
            refinement.points,
            original,
            memory.view_corners,
        )
        overlap = _native_overlap(selection.corners, native.corners)
        source_receipts = [receipt_by_track[index] for index in memory.source_indices]
        candidate = Candidate(
            scene=scene,
            memory_id=memory.memory_id,
            source_track_indices=tuple(memory.source_indices),
            source_track_ids=tuple(int(row.track_id) for row in source_receipts),
            confirmation_frame_id=memory.last_confirmation_frame_id,
            evidence_frame_ids=tuple(sorted(set(memory.evidence_frame_ids))),
            target_group=memory.target_group,
            corners=selection.corners,
            chosen_hypothesis=selection.name,
            hypothesis_rows=tuple(dict(row) for row in selection.diagnostics),
            openbox={
                "context_point_count": refinement.context_point_count,
                "component_count": refinement.component_count,
                "accepted_component_count": refinement.accepted_component_count,
                "mask_to_context_fraction": refinement.mask_to_context_fraction,
                "context_to_mask_fraction": refinement.context_to_mask_fraction,
                "mutual_harmonic": refinement.mutual_harmonic,
                "mask_retained_fraction": refinement.mask_retained_fraction,
                "used_context": refinement.used_context,
                "refined_point_count": int(len(refinement.points)),
            },
            source_median_mask_iou=float(
                np.median([row.median_pairwise_mask_aabb_iou for row in source_receipts])
            ),
            source_min_evidence_score=float(
                min(row.min_evidence_score for row in source_receipts)
            ),
            source_mean_score=float(
                np.mean([row.raw_mean_score for row in source_receipts])
            ),
            source_supported_voxels=int(
                sum(row.supported_voxel_count for row in source_receipts)
            ),
            max_native_iou=overlap[0],
            max_candidate_in_native=overlap[1],
            max_native_in_candidate=overlap[2],
        )
        memory_rows.append(
            {
                "memory_id": memory.memory_id,
                "target_group": memory.target_group,
                "source_track_indices": list(memory.source_indices),
                "source_track_ids": list(candidate.source_track_ids),
                "first_confirmation_frame_id": memory.first_confirmation_frame_id,
                "last_confirmation_frame_id": memory.last_confirmation_frame_id,
                "evidence_frame_ids": list(candidate.evidence_frame_ids),
                "mask_point_count": int(len(memory.points)),
                "source_count_total": memory.source_count_total,
                "dropped_point_count": memory.dropped_point_count,
                "dropped_view_count": memory.dropped_view_count,
                "dropped_source_count": memory.dropped_source_count,
                "retired_frame_id": memory.retired_frame_id,
                "chosen_hypothesis": selection.name,
                "hypotheses": [dict(row) for row in selection.diagnostics],
                "openbox": candidate.openbox,
                "native_overlap": {
                    "max_aabb_iou": overlap[0],
                    "max_candidate_in_native": overlap[1],
                    "max_native_in_candidate": overlap[2],
                    "pass": _passes_native_novelty(overlap),
                },
            }
        )
        if _passes_native_novelty(overlap):
            candidates.append(candidate)

    ranked = sorted(candidates, key=lambda row: row.quality_key)
    selected: list[Candidate] = []
    decisions: list[dict[str, Any]] = []
    for candidate in ranked:
        decision = "accepted"
        if any(_self_overlaps(candidate, kept) for kept in selected):
            decision = "self_nms"
        elif len(selected) >= MAX_BIRTHS_PER_SCENE:
            decision = "scene_cap"
        if decision == "accepted":
            selected.append(candidate)
        decisions.append(
            {
                "memory_id": candidate.memory_id,
                "decision": decision,
                "source_track_ids": list(candidate.source_track_ids),
                "chosen_hypothesis": candidate.chosen_hypothesis,
                "quality_key": list(candidate.quality_key[:-2]),
            }
        )
    report = {
        "r15_native_novel_observations": len(observations),
        "ovir_memory_count": len(memories),
        "ovir_association_count": sum(row["decision"] == "associate" for row in causal_audit),
        "causal_audit": causal_audit,
        "intrinsic_path": os.fspath(intrinsic_path),
        "intrinsic_sha256": _sha256(intrinsic_path),
        "depth_frames": frame_ledger,
        "memory_rows": memory_rows,
        "post_refinement_native_novel_count": len(candidates),
        "birth_count": len(selected),
        "selection_decisions": decisions,
    }
    return selected, report


def _score_candidates(candidates: Sequence[Candidate], native_floor: float) -> None:
    if native_floor <= EVALUATOR_CONFIDENCE_THRESHOLD + 3 * SCORE_EPSILON:
        raise RouteError("native score floor leaves no safe evaluated suffix interval")
    ranked = sorted(candidates, key=lambda row: row.quality_key)
    if not ranked:
        return
    high = native_floor - SCORE_EPSILON
    low = EVALUATOR_CONFIDENCE_THRESHOLD + SCORE_EPSILON
    scores = np.linspace(high, low, len(ranked), dtype=np.float64)
    if len(ranked) > 1 and not np.all(scores[:-1] > scores[1:]):
        raise RouteError("append score mapping is not strictly ordered")
    for candidate, score in zip(ranked, scores):
        candidate.append_score = float(score)


def _augmented_payload(native: NativePrediction, candidates: Sequence[Candidate]) -> Any:
    suffix = [
        (
            APPENDED_CLASS_ID,
            np.ascontiguousarray(candidate.corners, dtype=np.float32),
            float(candidate.append_score),
        )
        for candidate in candidates
    ]
    rows = tuple(native.rows) + tuple(suffix) if isinstance(native.rows, tuple) else list(native.rows) + suffix
    output = (rows,) if isinstance(native.payload, tuple) else [rows]
    _assert_native_prefix(native.rows, output[0], "R15 OVIR/OpenBox output")
    return output


def materialize(args: argparse.Namespace) -> dict[str, Any]:
    scene_list = args.scene_list.resolve()
    sidecar_path = args.r15_sidecar.resolve()
    baseline_root = args.baseline_root.resolve()
    scene_root = args.scene_root.resolve()
    output_root = args.output_root.resolve()
    if output_root.exists() or output_root.is_symlink():
        raise RouteError(f"refusing to overwrite output root: {output_root}")
    for path, label in ((baseline_root, "baseline root"), (scene_root, "RGB-D root")):
        if path.is_symlink() or not path.is_dir():
            raise RouteError(f"{label} must be a non-symlink directory: {path}")
    scenes, sidecar, r15_manifest, arrays = _load_route_inputs(
        scene_list, sidecar_path, args.expected_scene_count
    )
    selected_scenes = scenes
    if args.scene is not None:
        if args.scene not in scenes:
            raise RouteError(f"scene outside official list: {args.scene}")
        selected_scenes = [args.scene]
    if args.max_scenes is not None:
        if args.max_scenes < 1:
            raise RouteError("max-scenes must be positive")
        selected_scenes = selected_scenes[: args.max_scenes]

    natives: dict[str, NativePrediction] = {}
    native_hashes: dict[str, str] = {}
    all_native_scores: list[float] = []
    selected_by_scene: dict[str, list[Candidate]] = {}
    scene_reports: dict[str, Any] = {}
    for position, scene in enumerate(selected_scenes, 1):
        native_path = _regular_file(
            baseline_root / f"{scene}{PREDICTION_SUFFIX}", f"native prediction {scene}"
        )
        native_hashes[scene] = _sha256(native_path)
        native = _load_native_prediction(native_path)
        natives[scene] = native
        all_native_scores.extend(float(row[2]) for row in native.rows)
        scene_index = scenes.index(scene)
        selected, report = _process_scene(
            scene=scene,
            scene_index=scene_index,
            scene_root=scene_root,
            sidecar=sidecar,
            arrays=arrays,
            native=native,
        )
        selected_by_scene[scene] = selected
        scene_reports[scene] = report
        print(
            f"[{position}/{len(selected_scenes)}] {scene}: "
            f"R15={report['r15_native_novel_observations']} "
            f"memory={report['ovir_memory_count']} birth={len(selected)}",
            flush=True,
        )

    if not all_native_scores:
        raise RouteError("baseline contains no native scores")
    native_floor = min(all_native_scores)
    flat_candidates = [row for scene in selected_scenes for row in selected_by_scene[scene]]
    _score_candidates(flat_candidates, native_floor)
    append_scores = [float(row.append_score) for row in flat_candidates]
    if append_scores and (
        max(append_scores) >= native_floor
        or min(append_scores) <= EVALUATOR_CONFIDENCE_THRESHOLD
        or len(set(append_scores)) != len(append_scores)
    ):
        raise RouteError("native-first append score contract failed")

    if args.plan_only:
        return {
            "scene_count": len(selected_scenes),
            "native_count": sum(len(value.rows) for value in natives.values()),
            "birth_count": len(flat_candidates),
            "native_score_floor": native_floor,
            "append_score_min": min(append_scores) if append_scores else None,
            "append_score_max": max(append_scores) if append_scores else None,
            "scenes": scene_reports,
        }

    output_root.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{output_root.name}.stage-", dir=output_root.parent))
    output_hashes: dict[str, str] = {}
    try:
        for scene in selected_scenes:
            output_path = stage / f"{scene}{PREDICTION_SUFFIX}"
            ordered_candidates = sorted(
                selected_by_scene[scene], key=lambda row: (-float(row.append_score), row.memory_id)
            )
            _write_pickle(output_path, _augmented_payload(natives[scene], ordered_candidates))
            reloaded = _load_native_prediction(output_path)
            _assert_native_prefix(natives[scene].rows, reloaded.rows, scene)
            if len(reloaded.rows) != len(natives[scene].rows) + len(ordered_candidates):
                raise RouteError(f"output suffix count mismatch: {scene}")
            if any(float(row[2]) >= native_floor for row in reloaded.rows[len(natives[scene].rows):]):
                raise RouteError(f"suffix crossed native floor: {scene}")
            output_hashes[scene] = _sha256(output_path)
            scene_reports[scene]["suffix"] = [
                {
                    "suffix_index": index,
                    "memory_id": candidate.memory_id,
                    "source_track_ids": list(candidate.source_track_ids),
                    "target_group": candidate.target_group,
                    "confirmation_frame_id": candidate.confirmation_frame_id,
                    "evidence_frame_ids": list(candidate.evidence_frame_ids),
                    "chosen_hypothesis": candidate.chosen_hypothesis,
                    "score": candidate.append_score,
                    "corners_world": candidate.corners.tolist(),
                }
                for index, candidate in enumerate(ordered_candidates)
            ]
        if _sha256(sidecar.path) != sidecar.sha256 or (
            sidecar.npz_path is not None and _sha256(sidecar.npz_path) != sidecar.npz_sha256
        ):
            raise RouteError("sealed R15 sidecar changed during materialization")
        manifest = {
            "schema": SCHEMA,
            "mode": "r15_ovir_openbox_mhobb_native_first_low_score_append",
            "training_free": True,
            "target_dataset_training": False,
            "online_learning": False,
            "external_pretraining_frozen": True,
            "past_only": True,
            "query_before_commit": True,
            "gt_access": False,
            "evaluator_access": False,
            "annotation_path_argument": False,
            "native_rows_are_unchanged_prefix": True,
            "native_clip_unchanged": True,
            "native_geometry_changed": False,
            "native_score_changed": False,
            "native_order_changed": False,
            "birth": True,
            "append_can_suppress_native": False,
            "score_mode": "native_real_score_plus_strict_lower_unique_suffix",
            "scene_count": len(selected_scenes),
            "native_count": sum(len(value.rows) for value in natives.values()),
            "r15_native_novel_observation_count": int(
                sum(row["r15_native_novel_observations"] for row in scene_reports.values())
            ),
            "ovir_memory_count": int(sum(row["ovir_memory_count"] for row in scene_reports.values())),
            "ovir_association_count": int(sum(row["ovir_association_count"] for row in scene_reports.values())),
            "openbox_context_used_count": int(
                sum(
                    memory["openbox"]["used_context"]
                    for report in scene_reports.values()
                    for memory in report["memory_rows"]
                )
            ),
            "birth_count": len(flat_candidates),
            "hypothesis_counts": dict(Counter(row.chosen_hypothesis for row in flat_candidates)),
            "native_score_floor": native_floor,
            "evaluator_confidence_threshold": EVALUATOR_CONFIDENCE_THRESHOLD,
            "minimum_append_score": min(append_scores) if append_scores else None,
            "maximum_append_score": max(append_scores) if append_scores else None,
            "append_scores_unique": len(set(append_scores)) == len(append_scores),
            "append_scores_strictly_below_all_native": not append_scores or max(append_scores) < native_floor,
            "frozen_policy": {
                "depth_stride": DEPTH_STRIDE,
                "depth_range_m": [MIN_DEPTH_M, MAX_DEPTH_M],
                "depth_voxel_m": DEPTH_VOXEL_M,
                "roi_expansion_m": ROI_EXPANSION_M,
                "openbox_proximity_m": OPENBOX_PROXIMITY_M,
                "openbox_component_policy": "26_neighbor_voxels_local_near_mask_mutual_support_clean_top2",
                "ovir_max_memories": OVIR_MAX_MEMORIES,
                "ovir_max_idle_frames": OVIR_MAX_IDLE_FRAMES,
                "ovir_max_points_per_memory": OVIR_MAX_POINTS_PER_MEMORY,
                "ovir_max_views_per_memory": OVIR_MAX_VIEWS_PER_MEMORY,
                "ovir_max_sources_per_memory": OVIR_MAX_SOURCES_PER_MEMORY,
                "ovir_min_association_score": OVIR_MIN_ASSOCIATION_SCORE,
                "ovir_association": "same_group_and_spatial_or_point_overlap_query_before_commit",
                "hypotheses": [
                    "r15_original",
                    "raw_yaw_refined",
                    "pca_yaw_refined",
                    "axis_aligned_refined",
                ],
                "hypothesis_selection": "cross_view_geometry_evidence_no_gt",
                "native_novelty_aabb_iou_gte_reject": NATIVE_NOVELTY_AABB_IOU,
                "native_bidirectional_containment_gte_reject": NATIVE_MAX_BIDIRECTIONAL_CONTAINMENT,
                "self_nms_aabb_iou_gte_reject": SELF_NMS_AABB_IOU,
                "self_nms_bidirectional_containment_gte_reject": SELF_NMS_BIDIRECTIONAL_CONTAINMENT,
                "max_births_per_scene": MAX_BIRTHS_PER_SCENE,
                "appended_class_id": APPENDED_CLASS_ID,
            },
            "inputs": {
                "scene_list": os.fspath(scene_list),
                "scene_list_sha256": _sha256(scene_list),
                "baseline_root": os.fspath(baseline_root),
                "r15_sidecar": os.fspath(sidecar.path),
                "r15_sidecar_sha256": sidecar.sha256,
                "r15_npz": os.fspath(sidecar.npz_path),
                "r15_npz_sha256": sidecar.npz_sha256,
                "r15_schema": R15_SCHEMA,
                "r15_accepted_shadow_count": int(np.count_nonzero(arrays["track_accepted_shadow"])),
                "rgbd_root": os.fspath(scene_root),
                "materializer": os.fspath(Path(__file__).resolve()),
                "materializer_sha256": _sha256(Path(__file__).resolve()),
                "geometry_module": os.fspath((ROOT / "boxfusion/r15_ovir_openbox_lite.py").resolve()),
                "geometry_module_sha256": _sha256(ROOT / "boxfusion/r15_ovir_openbox_lite.py"),
            },
            "native_prediction_sha256": native_hashes,
            "output_prediction_sha256": output_hashes,
            "scenes": scene_reports,
        }
        _write_json(stage / MANIFEST_NAME, manifest)
        if output_root.exists() or output_root.is_symlink():
            raise RouteError(f"refusing to overwrite output root: {output_root}")
        os.rename(stage, output_root)
        stage = None
        return manifest
    finally:
        if stage is not None:
            shutil.rmtree(stage, ignore_errors=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scene-list",
        type=Path,
        default=ROOT / "evaluation/data_util/meta_data/scannetv2_val.txt",
    )
    parser.add_argument(
        "--baseline-root",
        type=Path,
        default=ROOT / "results/scannet_t05_boxer_replay_active_score05",
    )
    parser.add_argument(
        "--r15-sidecar",
        type=Path,
        default=ROOT / "logs/scannet_target_first_mobilesam_masklift_full100_score05/TARGET_FIRST_MOBILESAM_MASKLIFT_FULL100.json",
    )
    parser.add_argument(
        "--scene-root",
        type=Path,
        default=ROOT / "upstream_clean/scannet_readme_frames",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "results/scannet_cbest_real_score_r15_ovir_openbox_mhobb_lowappend_score05",
    )
    parser.add_argument("--expected-scene-count", type=int, default=100)
    parser.add_argument("--scene")
    parser.add_argument("--max-scenes", type=int)
    parser.add_argument("--plan-only", action="store_true")
    return parser


def main() -> None:
    args = _parser().parse_args()
    manifest = materialize(args)
    print(
        json.dumps(
            {
                "scene_count": manifest["scene_count"],
                "native_count": manifest["native_count"],
                "birth_count": manifest["birth_count"],
                "native_score_floor": manifest["native_score_floor"],
                "minimum_append_score": manifest.get("minimum_append_score", manifest.get("append_score_min")),
                "maximum_append_score": manifest.get("maximum_append_score", manifest.get("append_score_max")),
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
