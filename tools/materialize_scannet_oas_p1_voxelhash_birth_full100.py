#!/usr/bin/env python3
"""Materialize OAS-P1 automatic-mask voxel-hash births below native scores.

The candidate stream is the sealed, class-agnostic FastSAM F2 automatic-mask
RGB-D lift.  Candidate association is query-before-commit and target-label
free.  Final Cbest boxes are read only after proposal construction to perform
conservative terminal duplicate suppression; therefore this program describes
an auditable terminal replay, not a fully integrated live native snapshot.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if os.fspath(ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, os.fspath(ROOT))

from boxfusion.oas_p1_voxelhash import (  # noqa: E402
    AutomaticMaskObservation,
    CausalVoxelHashTracker,
    InstanceProposal,
    aabb_overlap,
    build_instance_proposal,
    policy_receipt,
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


SCHEMA = "boxfusion.scannet_oas_p1_voxelhash_birth_full100.v1"
F2_MANIFEST_SCHEMA = "boxfusion.scannet_fastsam_f2_paper100.merge.v1"
F2_SCENE_SCHEMA = "boxfusion.scannet_fastsam_f2_paper100.scene.v1"
F2_ARRAY_SCHEMA = "boxfusion.scannet_fastsam_f2_paper100.evidence.v1"
MANIFEST_NAME = "OAS_P1_VOXELHASH_BIRTH_FULL100.json"
PREDICTION_SUFFIX = "_boxes.pkl"
OFFICIAL_SCENE_LIST_SHA256 = (
    "4b18fc586f7ad60cb17f41ee7d1d8b0ab1a0782917b5ae3519dd8ec90e7744d5"
)

# Frozen before this active AP run.  These are inherited from the native-first
# append contract already used by the real-score R15 route.
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
class BirthCandidate:
    scene: str
    proposal: InstanceProposal
    max_native_iou: float
    max_candidate_in_native: float
    max_native_in_candidate: float
    append_score: float | None = None

    @property
    def quality_key(self) -> tuple[Any, ...]:
        return self.proposal.quality_key + (self.scene, self.proposal.memory_id)


def _read_json(path: Path, label: str) -> dict[str, Any]:
    _regular_file(path, label)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RouteError(f"invalid {label}: {path}") from error
    if not isinstance(value, dict):
        raise RouteError(f"{label} must contain a JSON object")
    return value


def _hash_array(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(b"\0")
    digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode("ascii"))
    digest.update(b"\0")
    digest.update(array.tobytes())
    return digest.hexdigest()


def _load_f2_manifest(
    f2_root: Path,
    scenes: Sequence[str],
) -> tuple[Path, str, dict[str, Mapping[str, Any]], dict[str, Any]]:
    path = _regular_file(
        f2_root / "final/F2_FASTSAM_PAPER100.json", "sealed F2 manifest"
    )
    digest = _sha256(path)
    payload = _read_json(path, "sealed F2 manifest")
    contracts = payload.get("contracts")
    coverage = payload.get("coverage")
    if (
        payload.get("schema") != F2_MANIFEST_SCHEMA
        or payload.get("complete") is not True
        or payload.get("overall_pass") is not True
        or not isinstance(contracts, Mapping)
        or contracts.get("ground_truth_access") is not False
        or contracts.get("evaluator_access") is not False
        or contracts.get("training") is not False
        or not isinstance(coverage, Mapping)
        or tuple(coverage.get("scene_order", ())) != tuple(scenes)
        or int(coverage.get("source_count", -1)) != 52_299
    ):
        raise RouteError("sealed F2 manifest contract differs")
    scene_rows = payload.get("scenes")
    if not isinstance(scene_rows, list) or len(scene_rows) != len(scenes):
        raise RouteError("sealed F2 scene ledger differs")
    indexed: dict[str, Mapping[str, Any]] = {}
    for index, row in enumerate(scene_rows):
        if (
            not isinstance(row, Mapping)
            or row.get("scene_id") != scenes[index]
            or row.get("scene_index") != index
        ):
            raise RouteError("sealed F2 scene order differs")
        indexed[scenes[index]] = row
    return path, digest, indexed, payload


def _expected_source_rows(sidecar: Mapping[str, Any], scene: str) -> list[Mapping[str, Any]]:
    if (
        sidecar.get("schema") != F2_SCENE_SCHEMA
        or sidecar.get("scene_id", scene) != scene
        or sidecar.get("complete") is not True
        or sidecar.get("contracts", {}).get("ground_truth_access") is not False
        or sidecar.get("contracts", {}).get("evaluator_access") is not False
        or sidecar.get("contracts", {}).get("training") is not False
    ):
        raise RouteError(f"sealed F2 scene contract differs: {scene}")
    frames = sidecar.get("frames")
    if not isinstance(frames, list):
        raise RouteError(f"sealed F2 frame ledger missing: {scene}")
    result: list[Mapping[str, Any]] = []
    previous_frame = -1
    seen: set[str] = set()
    for frame in frames:
        if not isinstance(frame, Mapping) or type(frame.get("frame_id")) is not int:
            raise RouteError(f"invalid sealed F2 frame: {scene}")
        frame_id = int(frame["frame_id"])
        if frame_id <= previous_frame:
            raise RouteError(f"sealed F2 frames are not ordered: {scene}")
        previous_frame = frame_id
        sources = frame.get("sources")
        if not isinstance(sources, list):
            raise RouteError(f"invalid sealed F2 source ledger: {scene}/{frame_id}")
        for source in sources:
            if not isinstance(source, Mapping) or not isinstance(source.get("source_id"), str):
                raise RouteError(f"invalid sealed F2 source: {scene}/{frame_id}")
            source_id = str(source["source_id"])
            if source_id in seen:
                raise RouteError(f"duplicate sealed F2 source: {source_id}")
            if not source_id.startswith(f"{scene}/frame_{frame_id:06d}/raw_"):
                raise RouteError(f"sealed F2 source identity differs: {source_id}")
            seen.add(source_id)
            result.append(source)
    return result


def _scene_observations(
    *,
    scene: str,
    f2_root: Path,
    ledger: Mapping[str, Any],
) -> tuple[list[AutomaticMaskObservation], dict[str, Any], tuple[Path, str], tuple[Path, str]]:
    sidecar_path = _regular_file(f2_root / "scenes" / f"{scene}.json", f"F2 sidecar {scene}")
    array_path = _regular_file(f2_root / "arrays" / f"{scene}.npz", f"F2 arrays {scene}")
    sidecar_digest = _sha256(sidecar_path)
    array_digest = _sha256(array_path)
    declared_sidecar = ledger.get("sidecar")
    declared_array = ledger.get("evidence_npz")
    if (
        not isinstance(declared_sidecar, Mapping)
        or not isinstance(declared_array, Mapping)
        or declared_sidecar.get("sha256") != sidecar_digest
        or declared_array.get("sha256") != array_digest
    ):
        raise RouteError(f"sealed F2 per-scene hash differs: {scene}")
    sidecar = _read_json(sidecar_path, f"F2 sidecar {scene}")
    source_rows = _expected_source_rows(sidecar, scene)
    try:
        with np.load(array_path, allow_pickle=False) as archive:
            if (
                str(archive["schema"].item()) != F2_ARRAY_SCHEMA
                or str(archive["scene_id"].item()) != scene
                or archive["mask_shape"].tolist() != [480, 640]
                or str(archive["mask_bitorder"].item()) != "little"
            ):
                raise RouteError(f"sealed F2 array metadata differs: {scene}")
            source_ids = np.array(archive["source_ids"], copy=True)
            frame_ids = np.array(archive["frame_ids"], dtype=np.int64, copy=True)
            point_offsets = np.array(archive["point_offsets"], dtype=np.int64, copy=True)
            points_world = np.array(archive["points_world"], dtype=np.float64, copy=True)
            voxel_keys = np.asarray(archive["voxel_keys"])
            masks = np.asarray(archive["masks_packbits"])
    except (KeyError, OSError, ValueError) as error:
        if isinstance(error, RouteError):
            raise
        raise RouteError(f"could not decode sealed F2 arrays: {scene}") from error
    count = len(source_rows)
    if (
        source_ids.shape != (count,)
        or frame_ids.shape != (count,)
        or point_offsets.shape != (count + 1,)
        or point_offsets[0] != 0
        or np.any(point_offsets[1:] < point_offsets[:-1])
        or point_offsets[-1] != len(points_world)
        or points_world.ndim != 2
        or points_world.shape[1:] != (3,)
        or not np.isfinite(points_world).all()
        or voxel_keys.shape != points_world.shape
        or masks.shape != (count, 38_400)
        or masks.dtype != np.uint8
    ):
        raise RouteError(f"sealed F2 evidence shape differs: {scene}")
    expected_ids = tuple(str(row["source_id"]) for row in source_rows)
    expected_frames = tuple(
        int(str(row["source_id"]).split("/frame_")[1].split("/")[0])
        for row in source_rows
    )
    if tuple(str(value) for value in source_ids) != expected_ids or tuple(frame_ids) != expected_frames:
        raise RouteError(f"sealed F2 source ledger and arrays differ: {scene}")
    distinct_frames = tuple(sorted(set(expected_frames)))
    ordinal_by_frame = {frame_id: index for index, frame_id in enumerate(distinct_frames)}
    observations: list[AutomaticMaskObservation] = []
    invalid_lift_count = 0
    for index, row in enumerate(source_rows):
        confidence = float(row.get("confidence", float("nan")))
        start, end = int(point_offsets[index]), int(point_offsets[index + 1])
        try:
            observation = AutomaticMaskObservation(
                source_id=expected_ids[index],
                frame_id=expected_frames[index],
                frame_ordinal=ordinal_by_frame[expected_frames[index]],
                confidence=confidence,
                points_world=points_world[start:end],
            )
        except ValueError:
            invalid_lift_count += 1
            continue
        observations.append(observation)
    return (
        observations,
        {
            "sealed_source_count": count,
            "valid_observation_count": len(observations),
            "invalid_observation_count": invalid_lift_count,
            "frame_count": len(distinct_frames),
            "source_id_ledger_sha256": _hash_array(source_ids),
            "mask_packbits_shape": [int(value) for value in masks.shape],
            "raw_point_count": int(len(points_world)),
        },
        (sidecar_path, sidecar_digest),
        (array_path, array_digest),
    )


def _native_overlap(
    proposal: InstanceProposal,
    native: NativePrediction,
) -> tuple[float, float, float]:
    if not len(native.corners):
        return 0.0, 0.0, 0.0
    values = []
    for corners in native.corners:
        lower, upper = corners.min(axis=0), corners.max(axis=0)
        values.append(aabb_overlap(proposal.lower, proposal.upper, lower, upper))
    return tuple(float(max(row[index] for row in values)) for index in range(3))  # type: ignore[return-value]


def _passes_native_novelty(overlap: tuple[float, float, float]) -> bool:
    return (
        overlap[0] < NATIVE_NOVELTY_AABB_IOU
        and overlap[1] < NATIVE_MAX_BIDIRECTIONAL_CONTAINMENT
        and overlap[2] < NATIVE_MAX_BIDIRECTIONAL_CONTAINMENT
    )


def _self_overlaps(left: BirthCandidate, right: BirthCandidate) -> bool:
    overlap = aabb_overlap(
        left.proposal.lower,
        left.proposal.upper,
        right.proposal.lower,
        right.proposal.upper,
    )
    return (
        overlap[0] >= SELF_NMS_AABB_IOU
        or overlap[1] >= SELF_NMS_BIDIRECTIONAL_CONTAINMENT
        or overlap[2] >= SELF_NMS_BIDIRECTIONAL_CONTAINMENT
    )


def _process_scene(
    *,
    scene: str,
    f2_root: Path,
    f2_ledger: Mapping[str, Any],
    native: NativePrediction,
) -> tuple[list[BirthCandidate], dict[str, Any], tuple[Path, str], tuple[Path, str]]:
    observations, source_report, sidecar_receipt, array_receipt = _scene_observations(
        scene=scene, f2_root=f2_root, ledger=f2_ledger
    )
    tracker = CausalVoxelHashTracker()
    cursor = 0
    while cursor < len(observations):
        frame_id = observations[cursor].frame_id
        frame_ordinal = observations[cursor].frame_ordinal
        end = cursor
        while end < len(observations) and observations[end].frame_id == frame_id:
            end += 1
        tracker.process_frame(frame_id, frame_ordinal, observations[cursor:end])
        cursor = end
    audit = tracker.audit
    if not audit.query_before_commit or audit.same_frame_self_confirmation_count != 0:
        raise RouteError(f"causal voxel-hash audit failed: {scene}")

    proposals: list[InstanceProposal] = []
    rejected_reasons: Counter[str] = Counter()
    geometry_sources: Counter[str] = Counter()
    memory_rows: list[dict[str, Any]] = []
    for memory in tracker.memories:
        proposal = build_instance_proposal(memory)
        geometry_sources[proposal.geometry_source] += 1
        if not proposal.admissible:
            rejected_reasons[str(proposal.rejection_reason)] += 1
        else:
            proposals.append(proposal)
        if proposal.admissible:
            memory_rows.append(
                {
                    "memory_id": proposal.memory_id,
                    "source_ids": list(proposal.source_ids),
                    "frame_ids": list(proposal.frame_ids),
                    "source_count_total": proposal.source_count_total,
                    "median_association_harmonic": proposal.median_association_harmonic,
                    "mean_automask_confidence": proposal.mean_automask_confidence,
                    "consensus_voxel_count": proposal.consensus_voxel_count,
                    "consensus_view_iou_median": proposal.consensus_view_iou_median,
                    "consensus_loo_stability_iou_median": proposal.consensus_loo_stability_iou_median,
                    "geometry_source": proposal.geometry_source,
                }
            )

    candidates: list[BirthCandidate] = []
    native_rejected = 0
    for proposal in proposals:
        overlap = _native_overlap(proposal, native)
        if not _passes_native_novelty(overlap):
            native_rejected += 1
            continue
        candidates.append(
            BirthCandidate(
                scene=scene,
                proposal=proposal,
                max_native_iou=overlap[0],
                max_candidate_in_native=overlap[1],
                max_native_in_candidate=overlap[2],
            )
        )
    ranked = sorted(candidates, key=lambda row: row.quality_key)
    selected: list[BirthCandidate] = []
    self_nms_rejected = 0
    scene_cap_rejected = 0
    for candidate in ranked:
        if any(_self_overlaps(candidate, kept) for kept in selected):
            self_nms_rejected += 1
        elif len(selected) >= MAX_BIRTHS_PER_SCENE:
            scene_cap_rejected += 1
        else:
            selected.append(candidate)
    report = {
        **source_report,
        "tracker": asdict(audit),
        "memory_count": len(tracker.memories),
        "confirmed_admissible_proposal_count": len(proposals),
        "proposal_rejection_reasons": dict(rejected_reasons),
        "geometry_source_counts_all_memories": dict(geometry_sources),
        "terminal_native_dedup_rejected_count": native_rejected,
        "terminal_native_novel_candidate_count": len(candidates),
        "self_nms_rejected_count": self_nms_rejected,
        "scene_cap_rejected_count": scene_cap_rejected,
        "birth_count": len(selected),
        "admissible_memories": memory_rows,
    }
    return selected, report, sidecar_receipt, array_receipt


def _score_candidates(candidates: Sequence[BirthCandidate], native_floor: float) -> None:
    if native_floor <= EVALUATOR_CONFIDENCE_THRESHOLD + 3.0 * SCORE_EPSILON:
        raise RouteError("native score floor leaves no evaluated suffix interval")
    ranked = sorted(candidates, key=lambda row: row.quality_key)
    if not ranked:
        return
    scores = np.linspace(
        native_floor - SCORE_EPSILON,
        EVALUATOR_CONFIDENCE_THRESHOLD + SCORE_EPSILON,
        len(ranked),
        dtype=np.float64,
    )
    if len(scores) > 1 and not np.all(scores[:-1] > scores[1:]):
        raise RouteError("append score mapping is not strictly decreasing")
    for candidate, score in zip(ranked, scores):
        candidate.append_score = float(score)


def _augmented_payload(native: NativePrediction, candidates: Sequence[BirthCandidate]) -> Any:
    suffix = [
        (
            APPENDED_CLASS_ID,
            np.ascontiguousarray(row.proposal.corners, dtype=np.float32),
            float(row.append_score),
        )
        for row in candidates
    ]
    rows = tuple(native.rows) + tuple(suffix) if isinstance(native.rows, tuple) else list(native.rows) + suffix
    output = (rows,) if isinstance(native.payload, tuple) else [rows]
    _assert_native_prefix(native.rows, output[0], "OAS-P1 in-memory output")
    return output


def materialize(args: argparse.Namespace) -> dict[str, Any]:
    scene_list = args.scene_list.resolve()
    f2_root = args.f2_root.resolve()
    native_root = args.native_root.resolve()
    output_root = args.output_root.resolve()
    if output_root.exists() or output_root.is_symlink():
        raise RouteError(f"refusing to overwrite output root: {output_root}")
    for path, label in ((f2_root, "F2 root"), (native_root, "native root")):
        if path.is_symlink() or not path.is_dir():
            raise RouteError(f"{label} must be a non-symlink directory: {path}")
    scenes = list(_scene_list(scene_list, args.expected_scene_count))
    if args.expected_scene_count == 100 and _sha256(scene_list) != OFFICIAL_SCENE_LIST_SHA256:
        raise RouteError("official100 scene-list hash mismatch")
    f2_manifest_path, f2_manifest_sha, f2_ledgers, f2_manifest = _load_f2_manifest(
        f2_root, scenes
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
    selected_by_scene: dict[str, list[BirthCandidate]] = {}
    scene_reports: dict[str, Any] = {}
    source_receipts: list[tuple[Path, str]] = []
    for position, scene in enumerate(selected_scenes, 1):
        native_path = _regular_file(
            native_root / f"{scene}{PREDICTION_SUFFIX}", f"native prediction {scene}"
        )
        native_hashes[scene] = _sha256(native_path)
        native = _load_native_prediction(native_path)
        natives[scene] = native
        all_native_scores.extend(float(row[2]) for row in native.rows)
        selected, report, sidecar_receipt, array_receipt = _process_scene(
            scene=scene,
            f2_root=f2_root,
            f2_ledger=f2_ledgers[scene],
            native=native,
        )
        selected_by_scene[scene] = selected
        scene_reports[scene] = report
        source_receipts.extend((sidecar_receipt, array_receipt))
        print(
            f"[{position}/{len(selected_scenes)}] {scene}: "
            f"auto={report['valid_observation_count']} "
            f"memory={report['memory_count']} "
            f"admissible={report['confirmed_admissible_proposal_count']} "
            f"birth={len(selected)}",
            flush=True,
        )
    if not all_native_scores:
        raise RouteError("native predictions contain no scores")
    native_floor = min(all_native_scores)
    flat = [row for scene in selected_scenes for row in selected_by_scene[scene]]
    _score_candidates(flat, native_floor)
    append_scores = [float(row.append_score) for row in flat]
    if append_scores and (
        max(append_scores) >= native_floor
        or min(append_scores) <= EVALUATOR_CONFIDENCE_THRESHOLD
        or len(set(append_scores)) != len(append_scores)
    ):
        raise RouteError("native-first append score contract failed")

    if args.plan_only:
        return {
            "scene_count": len(selected_scenes),
            "native_count": sum(len(row.rows) for row in natives.values()),
            "birth_count": len(flat),
            "native_score_floor": native_floor,
            "scenes": scene_reports,
        }

    output_root.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{output_root.name}.stage-", dir=output_root.parent))
    output_hashes: dict[str, str] = {}
    try:
        for scene in selected_scenes:
            ordered = sorted(
                selected_by_scene[scene],
                key=lambda row: (-float(row.append_score), row.proposal.memory_id),
            )
            output_path = stage / f"{scene}{PREDICTION_SUFFIX}"
            _write_pickle(output_path, _augmented_payload(natives[scene], ordered))
            reloaded = _load_native_prediction(output_path)
            _assert_native_prefix(natives[scene].rows, reloaded.rows, scene)
            if len(reloaded.rows) != len(natives[scene].rows) + len(ordered):
                raise RouteError(f"output suffix count mismatch: {scene}")
            suffix_rows = reloaded.rows[len(natives[scene].rows) :]
            if any(
                float(row[2]) <= EVALUATOR_CONFIDENCE_THRESHOLD
                or float(row[2]) >= native_floor
                for row in suffix_rows
            ):
                raise RouteError(f"suffix score contract failed after reload: {scene}")
            output_hashes[scene] = _sha256(output_path)
            scene_reports[scene]["suffix"] = [
                {
                    "suffix_index": index,
                    "memory_id": row.proposal.memory_id,
                    "source_ids": list(row.proposal.source_ids),
                    "frame_ids": list(row.proposal.frame_ids),
                    "source_count_total": row.proposal.source_count_total,
                    "geometry_source": row.proposal.geometry_source,
                    "median_association_harmonic": row.proposal.median_association_harmonic,
                    "mean_automask_confidence": row.proposal.mean_automask_confidence,
                    "consensus_voxel_count": row.proposal.consensus_voxel_count,
                    "terminal_native_overlap": {
                        "aabb_iou": row.max_native_iou,
                        "candidate_in_native": row.max_candidate_in_native,
                        "native_in_candidate": row.max_native_in_candidate,
                    },
                    "score": row.append_score,
                    "corners_world": row.proposal.corners.tolist(),
                }
                for index, row in enumerate(ordered)
            ]
        if _sha256(f2_manifest_path) != f2_manifest_sha:
            raise RouteError("sealed F2 manifest changed during materialization")
        for path, digest in source_receipts:
            if _sha256(path) != digest:
                raise RouteError(f"sealed F2 source changed during materialization: {path}")
        core_path = ROOT / "boxfusion/oas_p1_voxelhash.py"
        manifest = {
            "schema": SCHEMA,
            "mode": "automatic_mask_causal_voxelhash_native_first_low_score_append",
            "onlineanyseg_full_reproduction": False,
            "onlineanyseg_inspired_lite": True,
            "training_free": True,
            "target_dataset_training": False,
            "online_learning": False,
            "external_automask_model_frozen": True,
            "gt_access": False,
            "evaluator_access": False,
            "annotation_path_argument": False,
            "candidate_generation_past_only": True,
            "candidate_generation_query_before_commit": True,
            "terminal_native_dedup": True,
            "strict_integrated_online_claim": False,
            "native_rows_are_unchanged_prefix": True,
            "native_geometry_changed": False,
            "native_score_changed": False,
            "native_order_changed": False,
            "birth": True,
            "score_mode": "native_real_score_plus_strict_lower_unique_suffix",
            "scene_count": len(selected_scenes),
            "native_count": sum(len(row.rows) for row in natives.values()),
            "automatic_mask_observation_count": int(
                sum(row["valid_observation_count"] for row in scene_reports.values())
            ),
            "voxel_memory_count": int(sum(row["memory_count"] for row in scene_reports.values())),
            "admissible_proposal_count": int(
                sum(row["confirmed_admissible_proposal_count"] for row in scene_reports.values())
            ),
            "birth_count": len(flat),
            "birth_geometry_source_counts": dict(
                Counter(row.proposal.geometry_source for row in flat)
            ),
            "native_score_floor": native_floor,
            "evaluator_confidence_threshold": EVALUATOR_CONFIDENCE_THRESHOLD,
            "minimum_append_score": min(append_scores) if append_scores else None,
            "maximum_append_score": max(append_scores) if append_scores else None,
            "append_scores_unique": len(append_scores) == len(set(append_scores)),
            "append_scores_strictly_below_all_native": not append_scores or max(append_scores) < native_floor,
            "frozen_policy": {
                **policy_receipt(),
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
                "native_root": os.fspath(native_root),
                "f2_root": os.fspath(f2_root),
                "f2_manifest": os.fspath(f2_manifest_path),
                "f2_manifest_sha256": f2_manifest_sha,
                "f2_source_count": int(f2_manifest["coverage"]["source_count"]),
                "materializer": os.fspath(Path(__file__).resolve()),
                "materializer_sha256": _sha256(Path(__file__).resolve()),
                "core": os.fspath(core_path.resolve()),
                "core_sha256": _sha256(core_path),
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
        "--f2-root",
        type=Path,
        default=ROOT / "logs/scannet_fastsam_f2_paper100_score05",
    )
    parser.add_argument(
        "--native-root",
        type=Path,
        default=ROOT / "results/scannet_t05_boxer_replay_active_score05",
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--expected-scene-count", type=int, default=100)
    parser.add_argument("--scene")
    parser.add_argument("--max-scenes", type=int)
    parser.add_argument("--plan-only", action="store_true")
    return parser


def main() -> None:
    manifest = materialize(_parser().parse_args())
    print(
        json.dumps(
            {
                "scene_count": manifest["scene_count"],
                "native_count": manifest["native_count"],
                "birth_count": manifest["birth_count"],
                "native_score_floor": manifest["native_score_floor"],
                "minimum_append_score": manifest.get("minimum_append_score"),
                "maximum_append_score": manifest.get("maximum_append_score"),
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
