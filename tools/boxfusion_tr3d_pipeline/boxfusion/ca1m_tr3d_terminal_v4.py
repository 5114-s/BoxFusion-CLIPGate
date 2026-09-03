"""Pure contracts for the two-stage CA-1M terminal-TR3D v4 route.

Stage P is an anchor-free, B6-free proposal cache produced from processed
train100 RGB-D.  Stage O is a later CPU-only association overlay.  Keeping the
schemas separate prevents a changed final-base/B6 anchor from invalidating the
expensive TR3D proposal inference.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
from typing import Any, Mapping

import numpy as np

from .ca1m_tr3d_terminal import (
    TerminalAssociation,
    aligned_boxes_to_world_corners,
    validate_homogeneous,
    validate_scene_id,
    write_npz_create_only,
)


PROPOSAL_SCHEMA = "boxfusion.ca1m_tr3d_anchor_free_proposal_cache.v4"
OVERLAY_SCHEMA = "boxfusion.ca1m_tr3d_terminal_cpu_overlay.v4"
FRAME_LINEAGE_SCHEMA = "boxfusion.ca1m_demo_gap20_early_finalize_lineage.v1"
PROPOSAL_STAGE = "P_anchor_free_ca_native_tr3d"
OVERLAY_STAGE = "O_cpu_final_base_b6_association"
# The pinned worker names this prefix ``p100_gap20``.  The independent frame
# lineage record additionally proves the demo loop's early-finalize behavior.
PREFIX_ID = "p100_gap20"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_array(value: Any) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    return sha256_bytes(array.tobytes(order="C"))


def derive_demo_gap20_early_finalize_frame_ids(
    frame_count: int, *, gap: int = 20, start: int = 0
) -> np.ndarray:
    """Reproduce the *reachable* final-base keyframes exactly.

    ``demo.py`` increments ``count`` and then finalizes when either
    ``count == N-1`` or ``count + gap > N-1``.  Consequently its apparent
    ``or count == len(dataset)-1`` keyframe branch is unreachable.  We model
    the loop instead of simplifying it to a misleading ``include_last`` rule.
    """

    if isinstance(frame_count, bool) or int(frame_count) != frame_count:
        raise ValueError("frame_count must be an integer")
    if isinstance(gap, bool) or int(gap) != gap or int(gap) != 20:
        raise ValueError("CA-1M final-base lineage freezes gap=20")
    if isinstance(start, bool) or int(start) != start or int(start) != 0:
        raise ValueError("CA-1M final-base lineage freezes start=0")
    count = int(frame_count)
    if count < 1:
        raise ValueError("frame_count must be positive")
    rows: list[int] = []
    index = 0
    while index < count:
        if index % 20 == 0:
            rows.append(index)
        index += 1
        if index == count - 1 or index + 20 > count - 1:
            break
    if not rows:
        raise RuntimeError("demo lineage simulation produced no keyframe")
    return np.asarray(rows, dtype=np.int64)


def frame_lineage_json(scene_id: str, frame_count: int) -> str:
    frames = derive_demo_gap20_early_finalize_frame_ids(frame_count)
    return json.dumps(
        {
            "schema": FRAME_LINEAGE_SCHEMA,
            "scene_id": validate_scene_id(scene_id),
            "source": "processed_train100_rgb_depth_pose_demo_loop_simulation",
            "start": 0,
            "gap": 20,
            "early_finalize_condition": "count_eq_N_minus_1_or_count_plus_gap_gt_N_minus_1_after_increment",
            "include_last": False,
            "frame_count": int(frame_count),
            "used_frame_ids": frames.tolist(),
            "anchor_access": False,
            "b6_access": False,
            "ground_truth_access": False,
        },
        separators=(",", ":"),
        sort_keys=True,
    )


def _sha(value: str, name: str) -> str:
    if _SHA256.fullmatch(str(value)) is None:
        raise ValueError(f"{name} must be a lowercase SHA256")
    return str(value)


def _scalar(archive: Mapping[str, Any], name: str) -> Any:
    if name not in archive:
        raise ValueError(f"proposal cache lacks {name}")
    value = np.asarray(archive[name])
    if value.shape != ():
        raise ValueError(f"{name} must be scalar")
    return value.item()


@dataclass(frozen=True)
class ProposalCacheSummary:
    scene_id: str
    frame_count: int
    used_frame_count: int
    point_count: int
    candidate_count: int
    model_runtime_s: float
    source_points_sha256: str
    frame_lineage_sha256: str
    checkpoint_binding_sha256: str
    checkpoint_sha256: str
    config_sha256: str
    code_manifest_sha256: str
    adapter_mode: str
    device: str
    pixel_stride: int = 4
    voxel_size_m: float = 0.01
    min_depth_m: float = 0.10
    max_depth_m: float = 6.0
    score_threshold: float = 0.01
    max_proposals: int = 256

    def as_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result.update(
            {
                "schema": PROPOSAL_SCHEMA,
                "complete": True,
                "stage": PROPOSAL_STAGE,
                "create_only": True,
                "resume_policy": "validate_complete_then_skip_else_fail",
                "observer_only": True,
                "mutation_enabled": False,
                "ground_truth_access": False,
                "anchor_access": False,
                "b6_access": False,
                "frame_lineage_schema": FRAME_LINEAGE_SCHEMA,
                "prefix_id": PREFIX_ID,
                "coordinate_frame": "ca1m_world",
                "box_mode": "class_agnostic_aligned_local_to_world_aabb_corners",
            }
        )
        return result


def proposal_cache_payload(
    *,
    summary: ProposalCacheSummary,
    used_frame_ids: Any,
    world_to_local: Any,
    candidate_corners_world: Any,
    candidate_scores: Any,
    candidate_point_count: Any,
    candidate_boxes_local: Any,
    candidate_labels: Any,
    frame_lineage: str,
    code_manifest: str,
) -> dict[str, np.ndarray]:
    scene = validate_scene_id(summary.scene_id)
    frames = np.asarray(used_frame_ids)
    corners = np.asarray(candidate_corners_world)
    scores = np.asarray(candidate_scores)
    point_counts = np.asarray(candidate_point_count)
    boxes_local = np.asarray(candidate_boxes_local)
    labels = np.asarray(candidate_labels)
    transform = validate_homogeneous(world_to_local, "world_to_local")
    expected_frames = derive_demo_gap20_early_finalize_frame_ids(summary.frame_count)
    if frames.dtype != np.dtype(np.int64) or not np.array_equal(frames, expected_frames):
        raise ValueError("used_frame_ids do not match reachable demo early-finalize lineage")
    if summary.used_frame_count != len(frames):
        raise ValueError("used_frame_count disagrees")
    if summary.point_count < 1:
        raise ValueError("proposal source cloud must be non-empty")
    if not math.isfinite(summary.model_runtime_s) or summary.model_runtime_s < 0.0:
        raise ValueError("model_runtime_s must be finite and non-negative")
    if corners.dtype != np.dtype(np.float32) or corners.shape != (
        summary.candidate_count,
        8,
        3,
    ):
        raise ValueError("candidate_corners_world must be float32 [C,8,3]")
    if scores.dtype != np.dtype(np.float32) or scores.shape != (
        summary.candidate_count,
    ):
        raise ValueError("candidate_scores must be float32 [C]")
    if point_counts.dtype != np.dtype(np.int64) or point_counts.shape != (
        summary.candidate_count,
    ):
        raise ValueError("candidate_point_count must be int64 [C]")
    if boxes_local.dtype != np.dtype(np.float32) or boxes_local.shape != (
        summary.candidate_count,
        7,
    ):
        raise ValueError("candidate_boxes_local must be float32 [C,7]")
    if labels.dtype != np.dtype(np.int64) or labels.shape != (
        summary.candidate_count,
    ):
        raise ValueError("candidate_labels must be int64 [C]")
    if (
        not np.isfinite(corners).all()
        or not np.isfinite(scores).all()
        or not np.isfinite(boxes_local).all()
        or np.any(scores < 0.0)
        or np.any(scores > 1.0)
        or np.any(point_counts < 0)
        or np.any(labels != 0)
    ):
        raise ValueError("proposal arrays violate finite/class-agnostic contract")
    if len(scores) and np.any(np.diff(scores) > 0.0):
        raise ValueError("candidate scores must be sorted descending")
    reconstructed = aligned_boxes_to_world_corners(boxes_local, transform)
    if not np.array_equal(reconstructed, corners):
        raise ValueError("local boxes do not reconstruct cached world corners")
    try:
        lineage_value = json.loads(frame_lineage)
        code_value = json.loads(code_manifest)
    except json.JSONDecodeError as error:
        raise ValueError("lineage/code manifest must be valid JSON") from error
    if json.dumps(lineage_value, separators=(",", ":"), sort_keys=True) != frame_lineage:
        raise ValueError("frame lineage JSON must be canonical")
    if json.dumps(code_value, separators=(",", ":"), sort_keys=True) != code_manifest:
        raise ValueError("code manifest JSON must be canonical")
    expected_lineage = frame_lineage_json(scene, summary.frame_count)
    if frame_lineage != expected_lineage:
        raise ValueError("frame lineage differs from direct demo-loop simulation")
    for value, expected, name in (
        (sha256_bytes(frame_lineage.encode()), summary.frame_lineage_sha256, "lineage"),
        (sha256_bytes(code_manifest.encode()), summary.code_manifest_sha256, "code"),
    ):
        if value != _sha(expected, f"{name} SHA256"):
            raise ValueError(f"{name} manifest SHA256 mismatch")
    for value, name in (
        (summary.source_points_sha256, "source points SHA256"),
        (summary.checkpoint_binding_sha256, "binding SHA256"),
        (summary.checkpoint_sha256, "checkpoint SHA256"),
        (summary.config_sha256, "config SHA256"),
    ):
        _sha(value, name)
    if summary.adapter_mode != "genuine" or not summary.device:
        raise ValueError("formal v4 proposal cache requires genuine TR3D/device")
    protocol = (
        summary.pixel_stride,
        summary.voxel_size_m,
        summary.min_depth_m,
        summary.max_depth_m,
        summary.score_threshold,
        summary.max_proposals,
    )
    if protocol != (4, 0.01, 0.10, 6.0, 0.01, 256):
        raise ValueError("proposal protocol differs from frozen v4 contract")
    metadata = summary.as_dict()
    return {
        "schema": np.asarray(PROPOSAL_SCHEMA),
        "complete": np.asarray(True, dtype=np.bool_),
        "stage": np.asarray(PROPOSAL_STAGE),
        "create_only": np.asarray(True, dtype=np.bool_),
        "ground_truth_access": np.asarray(False, dtype=np.bool_),
        "anchor_access": np.asarray(False, dtype=np.bool_),
        "b6_access": np.asarray(False, dtype=np.bool_),
        "scene_id": np.asarray(scene),
        "frame_count": np.asarray(summary.frame_count, dtype=np.int64),
        "used_frame_ids": np.ascontiguousarray(frames),
        "world_to_local": np.ascontiguousarray(transform, dtype=np.float64),
        "candidate_corners_world": np.ascontiguousarray(corners),
        "candidate_scores": np.ascontiguousarray(scores),
        "candidate_point_count": np.ascontiguousarray(point_counts),
        "candidate_boxes_local": np.ascontiguousarray(boxes_local),
        "candidate_labels": np.ascontiguousarray(labels),
        "frame_lineage_json": np.asarray(frame_lineage),
        "code_manifest_json": np.asarray(code_manifest),
        "summary_json": np.asarray(json.dumps(metadata, separators=(",", ":"), sort_keys=True)),
    }


def load_proposal_cache(
    path: Path,
    *,
    expected_scene: str | None = None,
    expected_binding_sha256: str | None = None,
) -> dict[str, Any]:
    if path.is_symlink():
        raise ValueError(f"proposal cache must not be a symlink: {path}")
    source = path.resolve()
    if not source.is_file() or source.is_symlink() or source.stat().st_size <= 0:
        raise FileNotFoundError(f"missing proposal cache: {source}")
    if source.stat().st_mode & 0o222:
        raise ValueError("proposal cache must be read-only")
    with np.load(source, allow_pickle=False) as archive:
        values = {name: np.array(archive[name], copy=True) for name in archive.files}
    expected_keys = {
        "schema", "complete", "stage", "create_only", "ground_truth_access",
        "anchor_access", "b6_access", "scene_id", "frame_count",
        "used_frame_ids", "world_to_local", "candidate_corners_world",
        "candidate_scores", "candidate_point_count", "candidate_boxes_local",
        "candidate_labels", "frame_lineage_json", "code_manifest_json",
        "summary_json",
    }
    if set(values) != expected_keys:
        raise ValueError("proposal cache key set differs")
    required_scalars = {
        "schema": PROPOSAL_SCHEMA,
        "complete": True,
        "stage": PROPOSAL_STAGE,
        "create_only": True,
        "ground_truth_access": False,
        "anchor_access": False,
        "b6_access": False,
    }
    for name, expected in required_scalars.items():
        if _scalar(values, name) != expected:
            raise ValueError(f"proposal cache scalar {name} differs")
    scene = validate_scene_id(str(_scalar(values, "scene_id")))
    if expected_scene is not None and scene != validate_scene_id(expected_scene):
        raise ValueError("proposal cache scene differs")
    try:
        metadata = json.loads(str(_scalar(values, "summary_json")))
    except json.JSONDecodeError as error:
        raise ValueError("proposal summary is not JSON") from error
    if metadata.get("schema") != PROPOSAL_SCHEMA or metadata.get("scene_id") != scene:
        raise ValueError("proposal summary contract differs")
    summary_fields = set(ProposalCacheSummary.__dataclass_fields__)
    try:
        summary = ProposalCacheSummary(**{name: metadata[name] for name in summary_fields})
    except (KeyError, TypeError) as error:
        raise ValueError("proposal summary fields differ") from error
    rebuilt = proposal_cache_payload(
        summary=summary,
        used_frame_ids=values["used_frame_ids"],
        world_to_local=values["world_to_local"],
        candidate_corners_world=values["candidate_corners_world"],
        candidate_scores=values["candidate_scores"],
        candidate_point_count=values["candidate_point_count"],
        candidate_boxes_local=values["candidate_boxes_local"],
        candidate_labels=values["candidate_labels"],
        frame_lineage=str(_scalar(values, "frame_lineage_json")),
        code_manifest=str(_scalar(values, "code_manifest_json")),
    )
    if set(rebuilt) != set(values):
        raise ValueError("proposal cache reconstruction differs")
    for name in rebuilt:
        if not np.array_equal(rebuilt[name], values[name]):
            raise ValueError(f"proposal cache field {name} differs from reconstruction")
    if (
        expected_binding_sha256 is not None
        and summary.checkpoint_binding_sha256 != _sha(
            expected_binding_sha256, "expected binding SHA256"
        )
    ):
        raise ValueError("proposal cache checkpoint binding differs")
    return {"path": source, "sha256": sha256_file(source), "summary": summary, **values}


@dataclass(frozen=True)
class OverlaySummary:
    scene_id: str
    anchor_count: int
    candidate_count: int
    near_candidate_count: int
    represented_anchor_count: int
    proposal_cache_sha256: str
    final_anchor_sha256: str
    final_anchor_manifest_sha256: str
    native_b6_diagnostic_sha256: str
    native_b6_collection_manifest_sha256: str
    native_b6_checkpoint_sha256: str
    native_b6_checkpoint_manifest_sha256: str
    active_anchor_scores_sha256: str
    near_iou: float = 0.15

    def as_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result.update(
            {
                "schema": OVERLAY_SCHEMA,
                "complete": True,
                "stage": OVERLAY_STAGE,
                "create_only": True,
                "cpu_only": True,
                "observer_only": True,
                "mutation_enabled": False,
                "ground_truth_access": False,
                "source_proposal_schema": PROPOSAL_SCHEMA,
                "legacy_rule_activation_authorized": False,
            }
        )
        return result


def overlay_payload(
    *,
    summary: OverlaySummary,
    anchor_corners: Any,
    anchor_scores: Any,
    proposal: Mapping[str, Any],
    association: TerminalAssociation,
) -> dict[str, np.ndarray]:
    scene = validate_scene_id(summary.scene_id)
    anchors = np.asarray(anchor_corners)
    scores = np.asarray(anchor_scores)
    candidates = np.asarray(proposal["candidate_corners_world"])
    candidate_scores = np.asarray(proposal["candidate_scores"])
    if anchors.dtype != np.dtype(np.float32) or anchors.shape != (
        summary.anchor_count,
        8,
        3,
    ):
        raise ValueError("anchor_corners must be float32 [A,8,3]")
    if scores.dtype != np.dtype(np.float32) or scores.shape != (summary.anchor_count,):
        raise ValueError("anchor_scores must be float32 [A]")
    if candidates.shape != (summary.candidate_count, 8, 3):
        raise ValueError("proposal candidate count differs")
    if not np.isfinite(anchors).all() or not np.isfinite(scores).all():
        raise ValueError("anchors/scores must be finite")
    if np.any(scores < 0.0) or np.any(scores > 1.0):
        raise ValueError("anchor scores must be in [0,1]")
    if association.near_mask.shape != (summary.candidate_count,):
        raise ValueError("association candidate count differs")
    if int(association.near_mask.sum()) != summary.near_candidate_count:
        raise ValueError("near candidate count differs")
    if len(association.represented_anchor_indices) != summary.represented_anchor_count:
        raise ValueError("represented anchor count differs")
    if not math.isfinite(summary.near_iou) or summary.near_iou != 0.15:
        raise ValueError("overlay freezes near_iou=0.15")
    for name in (
        "proposal_cache_sha256", "final_anchor_sha256",
        "final_anchor_manifest_sha256", "native_b6_diagnostic_sha256",
        "native_b6_collection_manifest_sha256", "native_b6_checkpoint_sha256",
        "native_b6_checkpoint_manifest_sha256", "active_anchor_scores_sha256",
    ):
        _sha(str(getattr(summary, name)), name)
    return {
        "schema": np.asarray(OVERLAY_SCHEMA),
        "complete": np.asarray(True, dtype=np.bool_),
        "stage": np.asarray(OVERLAY_STAGE),
        "cpu_only": np.asarray(True, dtype=np.bool_),
        "ground_truth_access": np.asarray(False, dtype=np.bool_),
        "scene_id": np.asarray(scene),
        "anchor_corners": np.ascontiguousarray(anchors),
        "active_anchor_scores": np.ascontiguousarray(scores),
        "candidate_corners_world": np.ascontiguousarray(candidates),
        "candidate_scores": np.ascontiguousarray(candidate_scores),
        "best_anchor_indices": np.ascontiguousarray(association.best_anchor_indices),
        "best_anchor_iou": np.ascontiguousarray(association.best_anchor_iou),
        "best_anchor_center_distance_m": np.ascontiguousarray(
            association.best_anchor_center_distance_m
        ),
        "near_mask": np.ascontiguousarray(association.near_mask),
        "represented_anchor_indices": np.ascontiguousarray(
            association.represented_anchor_indices
        ),
        "summary_json": np.asarray(
            json.dumps(summary.as_dict(), separators=(",", ":"), sort_keys=True)
        ),
    }


def load_overlay_cache(
    path: Path,
    *,
    expected_scene: str | None = None,
    expected_proposal_sha256: str | None = None,
) -> dict[str, Any]:
    if path.is_symlink():
        raise ValueError(f"overlay cache must not be a symlink: {path}")
    source = path.resolve()
    if not source.is_file() or source.is_symlink() or source.stat().st_size <= 0:
        raise FileNotFoundError(f"missing overlay cache: {source}")
    if source.stat().st_mode & 0o222:
        raise ValueError("overlay cache must be read-only")
    with np.load(source, allow_pickle=False) as archive:
        values = {name: np.array(archive[name], copy=True) for name in archive.files}
    expected_keys = {
        "schema", "complete", "stage", "cpu_only", "ground_truth_access",
        "scene_id", "anchor_corners", "active_anchor_scores",
        "candidate_corners_world", "candidate_scores", "best_anchor_indices",
        "best_anchor_iou", "best_anchor_center_distance_m", "near_mask",
        "represented_anchor_indices", "summary_json",
    }
    if set(values) != expected_keys:
        raise ValueError("overlay cache key set differs")
    for name, expected in {
        "schema": OVERLAY_SCHEMA,
        "complete": True,
        "stage": OVERLAY_STAGE,
        "cpu_only": True,
        "ground_truth_access": False,
    }.items():
        if _scalar(values, name) != expected:
            raise ValueError(f"overlay cache scalar {name} differs")
    scene = validate_scene_id(str(_scalar(values, "scene_id")))
    if expected_scene is not None and scene != validate_scene_id(expected_scene):
        raise ValueError("overlay cache scene differs")
    try:
        metadata = json.loads(str(_scalar(values, "summary_json")))
    except json.JSONDecodeError as error:
        raise ValueError("overlay summary is not JSON") from error
    summary_fields = set(OverlaySummary.__dataclass_fields__)
    try:
        summary = OverlaySummary(**{name: metadata[name] for name in summary_fields})
    except (KeyError, TypeError) as error:
        raise ValueError("overlay summary fields differ") from error
    association = TerminalAssociation(
        best_anchor_indices=np.asarray(values["best_anchor_indices"]),
        best_anchor_iou=np.asarray(values["best_anchor_iou"]),
        best_anchor_center_distance_m=np.asarray(
            values["best_anchor_center_distance_m"]
        ),
        near_mask=np.asarray(values["near_mask"]),
        represented_anchor_indices=np.asarray(values["represented_anchor_indices"]),
        legacy_rule_selected_candidate_rows=np.empty((0,), dtype=np.int64),
        legacy_rule_selected_anchor_indices=np.empty((0,), dtype=np.int64),
    )
    rebuilt = overlay_payload(
        summary=summary,
        anchor_corners=values["anchor_corners"],
        anchor_scores=values["active_anchor_scores"],
        proposal={
            "candidate_corners_world": values["candidate_corners_world"],
            "candidate_scores": values["candidate_scores"],
        },
        association=association,
    )
    for name in expected_keys:
        if not np.array_equal(rebuilt[name], values[name]):
            raise ValueError(f"overlay cache field {name} differs from reconstruction")
    if (
        expected_proposal_sha256 is not None
        and summary.proposal_cache_sha256
        != _sha(expected_proposal_sha256, "expected proposal SHA256")
    ):
        raise ValueError("overlay proposal-cache binding differs")
    return {"path": source, "sha256": sha256_file(source), "summary": summary, **values}


__all__ = [
    "FRAME_LINEAGE_SCHEMA", "OVERLAY_SCHEMA", "OVERLAY_STAGE", "OverlaySummary",
    "PREFIX_ID", "PROPOSAL_SCHEMA", "PROPOSAL_STAGE", "ProposalCacheSummary",
    "derive_demo_gap20_early_finalize_frame_ids", "frame_lineage_json", "load_overlay_cache",
    "load_proposal_cache", "overlay_payload", "proposal_cache_payload", "sha256_array", "sha256_bytes",
    "sha256_file", "write_npz_create_only",
]
