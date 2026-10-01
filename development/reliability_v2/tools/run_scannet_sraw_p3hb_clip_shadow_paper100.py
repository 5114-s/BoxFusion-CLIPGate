#!/usr/bin/env python3
"""Run the frozen SRAW-P3HB-CLIP-v1 causal shadow on paper100.

The runner is deliberately prediction-inert.  L2/F4 supplies identity and
geometry, the sealed F0 receipt supplies only the immutable score-0.5 CuTR
cache index, and the existing native 473-way CLIP assets score the three
causal crops.  No annotation, evaluator, or native prediction is accepted.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import tempfile
from typing import Any, Mapping, MutableMapping, Sequence

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if os.fspath(ROOT) not in sys.path:
    sys.path.insert(0, os.fspath(ROOT))

from tools.materialize_scannet_raw_boxer_past3_birth_full100 import (  # noqa: E402
    _aabb_overlap_matrices,
)
from tools.run_scannet_raw_boxer_clip_vocab_shadow_full100 import (  # noqa: E402
    EXPECTED_VOCABULARY_SIZE,
    TARGET_GROUP_ALIASES,
    _crop_rgb,
    _flush_batch,
    _load_clip_runtime,
    _load_text_features,
    _resolve_target_indices,
)
from tools.seal_scannet_l0_f3_f4_perview_paper100 import (  # noqa: E402
    F4_SCHEMA,
    SOURCE_RE,
)
from tools.seal_scannet_l2_source_preserving_paper100 import (  # noqa: E402
    PROTOCOL_ID as L2_PROTOCOL_ID,
    SCHEMA as L2_SCHEMA,
)


SCHEMA = "boxfusion.scannet_sraw_p3hb_clip_shadow_paper100.v1"
PROTOCOL_ID = "SRAW-P3HB-CLIP-V1"
F0_MERGE_SCHEMA = "boxfusion.scannet_fastsam_f0_full200.merge.v1"
F0_SCENE_SCHEMA = "boxfusion.scannet_fastsam_f0_full200.scene.v1"
CUTR_CACHE_SCHEMA = "boxfusion.cutr_postfilter_cache.v2"
CUTR_FIELD_NAMES = (
    "scores",
    "pred_classes",
    "pred_boxes",
    "pred_logits",
    "pred_boxes_3d",
    "object_desc",
    "pred_proj_xy",
)
CUTR_NAMESPACE = "scannet-score05-gap25-postfilter-v2"

DEFAULT_L2 = ROOT / "logs/scannet_l2_source_preserving_paper100_score05/final/L2_SOURCE_PRESERVING_PAPER100.json"
DEFAULT_F0 = ROOT / "logs/scannet_fastsam_f0_full200_score05/final/F0_FASTSAM_FULL200.json"
DEFAULT_PROTOCOL = ROOT / "docs/SRAW_P3HB_CLIP_V1_PROTOCOL_FREEZE.md"
DEFAULT_OUTPUT = ROOT / "logs/scannet_sraw_p3hb_clip_shadow_paper100_score05/SRAW_P3HB_CLIP_SHADOW_PAPER100.json"

GEOMETRY_POLICY = {
    "hb_confidence_gte": 0.55,
    "median_pairwise_hb_aabb_iou_gte": 0.25,
    "hb_center_rms_m_lte": 0.25,
    "selected_hb_each_local_extent_m_gte": 0.30,
    "selected_hb_h0_normalized_center_distance_lte": 0.50,
    "selected_hb_h0_volume_ratio": [0.25, 4.00],
    "selected_hb_h0_aabb_iou_gte": 0.20,
    "selected_hb_h0_bidirectional_max_containment_gte": 0.70,
}
SEMANTIC_POLICY = {
    "all_vocab_top1_target_votes_gte": 2,
    "same_target_alias_group_votes_gte": 2,
    "median_best_target_cosine_gte": 0.20,
    "median_target_non_target_margin_gte": -0.01,
}
NOVELTY_POLICY = {
    "cutr_aabb_iou_gte_reject": 0.10,
    "cutr_bidirectional_containment_gte_reject": 0.50,
    "past_birth_aabb_iou_gte_reject": 0.15,
    "past_birth_bidirectional_containment_gte_reject": 0.25,
    "max_births_per_scene": 2,
}
CONTRACTS: Mapping[str, bool] = {
    "shadow_only": True,
    "birth_enabled": False,
    "native_output_mutation": False,
    "ground_truth_access": False,
    "annotation_access": False,
    "evaluator_access": False,
    "future_frame_access": False,
    "training": False,
    "online_learning": False,
    "past_only": True,
}

_SIGNS = np.asarray(
    [
        (-1.0, -1.0, -1.0), (-1.0, -1.0, 1.0),
        (-1.0, 1.0, -1.0), (-1.0, 1.0, 1.0),
        (1.0, -1.0, -1.0), (1.0, -1.0, 1.0),
        (1.0, 1.0, -1.0), (1.0, 1.0, 1.0),
    ],
    dtype=np.float64,
)


class SRAWShadowError(RuntimeError):
    """Raised when a frozen input or causal invariant differs."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _regular(path: Path, label: str) -> Path:
    if path.is_symlink() or not path.is_file():
        raise SRAWShadowError(f"{label} must be a regular non-symlink file: {path}")
    return path.resolve()


def _json(path: Path, label: str) -> tuple[Path, dict[str, Any]]:
    source = _regular(path, label)
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SRAWShadowError(f"invalid {label}: {source}") from error
    if not isinstance(value, dict):
        raise SRAWShadowError(f"{label} must contain one JSON object")
    return source, value


class _InputLedger:
    def __init__(self) -> None:
        self.rows: dict[str, str] = {}

    def add(self, path: Path, label: str, expected: object | None = None) -> Path:
        source = _regular(path, label)
        digest = _sha256(source)
        if expected is not None and digest != expected:
            raise SRAWShadowError(f"{label} SHA-256 differs: {source}")
        key = os.fspath(source)
        if key in self.rows and self.rows[key] != digest:
            raise SRAWShadowError(f"input hash changed while reused: {source}")
        self.rows[key] = digest
        return source

    def verify(self) -> None:
        for name, digest in self.rows.items():
            if _sha256(Path(name)) != digest:
                raise SRAWShadowError(f"input hash drift detected: {name}")

    def receipt(self) -> dict[str, Any]:
        rows = [[name, digest] for name, digest in sorted(self.rows.items())]
        return {"file_count": len(rows), "ledger_sha256": _canonical_sha256(rows)}


def _number(value: object, label: str) -> float:
    if isinstance(value, bool):
        raise SRAWShadowError(f"{label} must be finite")
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise SRAWShadowError(f"{label} must be finite") from error
    if not math.isfinite(result):
        raise SRAWShadowError(f"{label} must be finite")
    return result


def _integer(value: object, label: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise SRAWShadowError(f"{label} must be an integer")
    result = int(value)
    if result < minimum:
        raise SRAWShadowError(f"{label} must be >= {minimum}")
    return result


def _array(value: object, shape: tuple[int, ...], label: str) -> np.ndarray:
    try:
        result = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as error:
        raise SRAWShadowError(f"{label} is not numeric") from error
    if result.shape != shape or not np.isfinite(result).all():
        raise SRAWShadowError(f"{label} must be finite shape {shape}")
    return np.ascontiguousarray(result)


def _aabb_corners(lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    return (lower + upper)[None, :] * 0.5 + _SIGNS * (upper - lower)[None, :] * 0.5


def _source_frame(source_id: str, scene_id: str) -> int:
    match = SOURCE_RE.fullmatch(source_id)
    if match is None or match["scene"] != scene_id:
        raise SRAWShadowError(f"invalid source identity: {source_id}")
    return int(match["frame"])


def _source_map(f4: Mapping[str, Any], scene_id: str) -> tuple[dict[str, Mapping[str, Any]], dict[int, Mapping[str, Any]]]:
    sources: dict[str, Mapping[str, Any]] = {}
    frames: dict[int, Mapping[str, Any]] = {}
    for ordinal, frame in enumerate(f4.get("frames", [])):
        if not isinstance(frame, Mapping):
            raise SRAWShadowError(f"invalid F4 frame: {scene_id}")
        frame_id = _integer(frame.get("frame_id"), "F4 frame_id")
        if frame.get("frame_ordinal") != ordinal or frame_id in frames:
            raise SRAWShadowError(f"F4 frame order differs: {scene_id}/{frame_id}")
        frames[frame_id] = frame
        for source in frame.get("sources", []):
            if not isinstance(source, Mapping) or not isinstance(source.get("source_id"), str):
                raise SRAWShadowError(f"invalid F4 source: {scene_id}/{frame_id}")
            source_id = str(source["source_id"])
            if source_id in sources or _source_frame(source_id, scene_id) != frame_id:
                raise SRAWShadowError(f"duplicate/misaligned F4 source: {source_id}")
            sources[source_id] = source
    return sources, frames


def _first_three_sources(
    track: Mapping[str, Any], source_order_index: Mapping[str, int],
    sources: Mapping[str, Mapping[str, Any]], scene_id: str,
) -> list[str]:
    raw = track.get("source_ids")
    if not isinstance(raw, list) or len(raw) != len(set(map(str, raw))):
        raise SRAWShadowError(f"invalid track source ledger: {scene_id}")
    normalized = [str(item) for item in raw]
    if any(item not in sources or item not in source_order_index for item in normalized):
        raise SRAWShadowError(f"track source absent from F4: {scene_id}")
    ordered = sorted(normalized, key=source_order_index.__getitem__)
    selected: list[str] = []
    seen_frames: set[int] = set()
    for source_id in ordered:
        frame_id = _source_frame(source_id, scene_id)
        if frame_id not in seen_frames:
            selected.append(source_id)
            seen_frames.add(frame_id)
        if len(selected) == 3:
            break
    selected_frames = [_source_frame(source_id, scene_id) for source_id in selected]
    if selected_frames != sorted(selected_frames) or len(selected_frames) != len(set(selected_frames)):
        raise SRAWShadowError(f"first-three source frames are not strictly causal: {scene_id}")
    return selected


def _hb(source: Mapping[str, Any], source_id: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    row = source.get("hypotheses", {}).get("HB")
    if not isinstance(row, Mapping) or row.get("valid") is not True:
        raise SRAWShadowError(f"valid HB missing: {source_id}")
    corners = _array(row.get("world_corners"), (8, 3), f"{source_id}.HB corners")
    center = _array(row.get("world_center"), (3,), f"{source_id}.HB center")
    extent = _array(row.get("local_extent"), (3,), f"{source_id}.HB extent")
    confidence = _number(row.get("confidence"), f"{source_id}.HB confidence")
    if np.any(np.ptp(corners, axis=0) <= 0.0) or np.any(extent <= 0.0):
        raise SRAWShadowError(f"degenerate HB: {source_id}")
    if not np.allclose(center, corners.mean(axis=0), rtol=0.0, atol=2e-5):
        raise SRAWShadowError(f"HB center/corners differ: {source_id}")
    return corners, center, extent, confidence


def _geometry_summary(source_ids: Sequence[str], sources: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    if len(source_ids) != 3:
        raise SRAWShadowError("P3 geometry requires exactly three sources")
    rows = [_hb(sources[source_id], source_id) for source_id in source_ids]
    corners = np.stack([row[0] for row in rows])
    iou, _, _ = _aabb_overlap_matrices(corners, corners)
    pairs = [float(iou[0, 1]), float(iou[0, 2]), float(iou[1, 2])]
    means = [float((iou[index].sum() - 1.0) / 2.0) for index in range(3)]
    medoid = max(range(3), key=lambda index: (means[index], rows[index][3], -index))
    selected_id = source_ids[medoid]
    selected_corners, _, selected_extent, selected_confidence = rows[medoid]
    h0 = sources[selected_id].get("hypotheses", {}).get("H0")
    if not isinstance(h0, Mapping) or h0.get("valid") is not True:
        raise SRAWShadowError(f"valid H0 missing: {selected_id}")
    h0_lower = _array(h0.get("q02"), (3,), f"{selected_id}.H0 q02")
    h0_upper = _array(h0.get("q98"), (3,), f"{selected_id}.H0 q98")
    if np.any(h0_upper <= h0_lower):
        raise SRAWShadowError(f"degenerate H0: {selected_id}")
    h0_corners = _aabb_corners(h0_lower, h0_upper)
    base_iou, hb_in_h0, h0_in_hb = _aabb_overlap_matrices(
        selected_corners[None], h0_corners[None]
    )
    hb_lower, hb_upper = selected_corners.min(0), selected_corners.max(0)
    hb_center, h0_center = (hb_lower + hb_upper) * 0.5, (h0_lower + h0_upper) * 0.5
    scale = max(float(np.linalg.norm(hb_upper - hb_lower)), float(np.linalg.norm(h0_upper - h0_lower)), 0.02)
    nd = float(np.linalg.norm(hb_center - h0_center) / scale)
    ratio = float(np.prod(hb_upper - hb_lower) / np.prod(h0_upper - h0_lower))
    containment = float(max(hb_in_h0[0, 0], h0_in_hb[0, 0]))
    centers = np.stack([row[1] for row in rows])
    center_rms = float(np.sqrt(np.mean(np.sum((centers - centers.mean(0)) ** 2, axis=1))))
    median_iou = float(np.median(pairs))
    confidences = [float(row[3]) for row in rows]
    checks = {
        "three_hb_confidences": all(value >= 0.55 for value in confidences),
        "median_pairwise_hb_aabb_iou": median_iou >= 0.25,
        "hb_center_rms_m": center_rms <= 0.25,
        "selected_hb_local_extent": bool(np.all(selected_extent >= 0.30)),
        "selected_hb_h0_normalized_center_distance": nd <= 0.50,
        "selected_hb_h0_volume_ratio": 0.25 <= ratio <= 4.00,
        "selected_hb_h0_overlap": float(base_iou[0, 0]) >= 0.20 or containment >= 0.70,
    }
    return {
        "gate_pass": all(checks.values()),
        "gate_checks": checks,
        "gate_rejection_reasons": [name for name, passed in checks.items() if not passed],
        "selected_source_id": selected_id,
        "selected_source_ordinal": medoid,
        "corners_world": selected_corners.tolist(),
        "hb_confidences": confidences,
        "selected_hb_confidence": selected_confidence,
        "pairwise_hb_aabb_ious": pairs,
        "median_pairwise_hb_aabb_iou": median_iou,
        "medoid_mean_hb_aabb_iou": means[medoid],
        "hb_center_rms_m": center_rms,
        "selected_hb_local_extent_m": selected_extent.tolist(),
        "selected_hb_h0_normalized_center_distance": nd,
        "selected_hb_h0_volume_ratio": ratio,
        "selected_hb_h0_aabb_iou": float(base_iou[0, 0]),
        "selected_hb_h0_bidirectional_max_containment": containment,
    }


def _semantic_summary(evidence: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if len(evidence) != 3:
        raise SRAWShadowError("semantic gate requires exactly three CLIP rows")
    target_votes = sum(bool(row.get("all_vocab_top1_is_target")) for row in evidence)
    groups: Counter[str] = Counter()
    for row in evidence:
        aliases = row.get("all_vocab_top1_target_alias_groups")
        if not isinstance(aliases, list):
            raise SRAWShadowError("all-vocabulary CLIP alias groups must be a list")
        if bool(row.get("all_vocab_top1_is_target")):
            if len(aliases) != 1:
                raise SRAWShadowError(
                    "target all-vocabulary top-1 must resolve to one alias group"
                )
            groups[str(aliases[0])] += 1
        elif aliases:
            raise SRAWShadowError(
                "non-target all-vocabulary top-1 cannot cast an alias vote"
            )
    if groups:
        target_group, group_votes = sorted(
            groups.items(), key=lambda item: (-item[1], item[0])
        )[0]
    else:
        target_group, group_votes = None, 0
    cosines = [_number(row.get("target_best_cosine"), "target cosine") for row in evidence]
    margins = [_number(row.get("target_non_target_margin"), "target margin") for row in evidence]
    median_cosine, median_margin = float(np.median(cosines)), float(np.median(margins))
    checks = {
        "all_vocab_top1_target_votes": target_votes >= 2,
        "same_target_alias_group_votes": group_votes >= 2,
        "median_best_target_cosine": median_cosine >= 0.20,
        "median_target_non_target_margin": median_margin >= -0.01,
    }
    return {
        "gate_pass": all(checks.values()),
        "gate_checks": checks,
        "gate_rejection_reasons": [name for name, passed in checks.items() if not passed],
        "all_vocab_top1_target_votes": target_votes,
        "target_group": target_group,
        "same_target_alias_group_votes": group_votes,
        "median_best_target_cosine": median_cosine,
        "median_target_non_target_margin": median_margin,
    }


def _overlap_max(left: np.ndarray, right: np.ndarray) -> tuple[float, float, float]:
    iou, left_containment, right_containment = _aabb_overlap_matrices(left, right)
    return (
        float(iou.max()) if iou.size else 0.0,
        float(left_containment.max()) if left_containment.size else 0.0,
        float(right_containment.max()) if right_containment.size else 0.0,
    )


def _admit_candidates(candidates: Sequence[MutableMapping[str, Any]], cutr_by_frame: Mapping[int, np.ndarray]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    accepted: list[dict[str, Any]] = []
    decisions: list[dict[str, Any]] = []
    accumulated: list[np.ndarray] = []
    cache_frames = sorted(cutr_by_frame)
    cache_index = 0
    ordered = sorted(
        candidates,
        key=lambda row: (
            row["confirmation_frame_id"],
            -row["geometry"]["medoid_mean_hb_aabb_iou"],
            -row["semantic"]["same_target_alias_group_votes"],
            -row["semantic"]["all_vocab_top1_target_votes"],
            -row["semantic"]["median_best_target_cosine"],
            -row["semantic"]["median_target_non_target_margin"],
            -row["geometry"]["selected_hb_confidence"],
            row["track_id"],
            row["geometry"]["selected_source_id"],
        ),
    )
    for candidate in ordered:
        confirmation = int(candidate["confirmation_frame_id"])
        evidence_frames = candidate.get("evidence_frame_ids")
        evidence_sources = candidate.get("evidence_source_ids")
        if (
            not isinstance(evidence_frames, list)
            or len(evidence_frames) != 3
            or evidence_frames != sorted(set(evidence_frames))
            or evidence_frames[2] != confirmation
            or not isinstance(evidence_sources, list)
            or len(evidence_sources) != 3
        ):
            raise SRAWShadowError("candidate first-three/confirmation contract differs")
        for source_id, frame_id in zip(evidence_sources, evidence_frames, strict=True):
            match = SOURCE_RE.fullmatch(str(source_id))
            if match is None or int(match["frame"]) != frame_id:
                raise SRAWShadowError("candidate source/frame evidence differs")
        if candidate["geometry"]["selected_source_id"] not in evidence_sources:
            raise SRAWShadowError("selected HB source is outside first-three evidence")
        while cache_index < len(cache_frames) and cache_frames[cache_index] <= confirmation:
            rows = cutr_by_frame[cache_frames[cache_index]]
            if len(rows):
                accumulated.append(rows)
            cache_index += 1
        candidate_corners = np.asarray(candidate["geometry"]["corners_world"], dtype=np.float64)[None]
        cutr = np.concatenate(accumulated, axis=0) if accumulated else np.empty((0, 8, 3), dtype=np.float64)
        cutr_iou, cutr_left, cutr_right = _overlap_max(candidate_corners, cutr)
        prior = np.stack([np.asarray(row["corners_world"], dtype=np.float64) for row in accepted]) if accepted else np.empty((0, 8, 3), dtype=np.float64)
        birth_iou, birth_left, birth_right = _overlap_max(candidate_corners, prior)
        decision = "accepted"
        if cutr_iou >= 0.10 or max(cutr_left, cutr_right) >= 0.50:
            decision = "cutr_overlap"
        elif birth_iou >= 0.15 or max(birth_left, birth_right) >= 0.25:
            decision = "past_birth_nms"
        elif len(accepted) >= 2:
            decision = "scene_cap"
        row = {
            "track_id": candidate["track_id"],
            "confirmation_frame_id": confirmation,
            "selected_source_id": candidate["geometry"]["selected_source_id"],
            "target_group": candidate["semantic"]["target_group"],
            "decision": decision,
            "max_cutr_aabb_iou": cutr_iou,
            "max_candidate_in_cutr_containment": cutr_left,
            "max_cutr_in_candidate_containment": cutr_right,
            "max_past_birth_aabb_iou": birth_iou,
            "max_candidate_in_past_birth_containment": birth_left,
            "max_past_birth_in_candidate_containment": birth_right,
        }
        decisions.append(row)
        if decision == "accepted":
            accepted.append(
                {
                    "track_id": candidate["track_id"],
                    "confirmation_frame_id": confirmation,
                    "selected_source_id": candidate["geometry"]["selected_source_id"],
                    "target_group": candidate["semantic"]["target_group"],
                    "corners_world": candidate["geometry"]["corners_world"],
                    "evidence_source_ids": list(candidate["evidence_source_ids"]),
                    "evidence_frame_ids": list(candidate["evidence_frame_ids"]),
                    "geometry": dict(candidate["geometry"]),
                    "semantic": dict(candidate["semantic"]),
                }
            )
    return accepted, decisions


def _torch_tensor_sha256(value: Any) -> str:
    try:
        import torch
    except ImportError as error:
        raise SRAWShadowError("torch is required for the CuTR cache") from error
    if not torch.is_tensor(value):
        raise SRAWShadowError("cached value is not a tensor")
    tensor = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode("utf-8"))
    digest.update(b"\0")
    digest.update(repr(tuple(tensor.shape)).encode("ascii"))
    digest.update(b"\0")
    digest.update(tensor.view(torch.uint8).numpy().tobytes(order="C"))
    return digest.hexdigest()


def _cache_manifest_records(
    frames: Sequence[Mapping[str, Any]], scene_id: str,
    schedule: Mapping[str, Any], ledger: _InputLedger
) -> tuple[Path, dict[int, Mapping[str, Any]]]:
    cache_paths = []
    for frame in frames:
        inputs = frame.get("inputs")
        if not isinstance(inputs, Mapping):
            raise SRAWShadowError(f"F0 frame inputs are absent: {scene_id}")
        cache_paths.append(Path(str(inputs.get("cutr_cache_path", ""))))
    if not cache_paths:
        raise SRAWShadowError(f"F0 cache frame ledger is empty: {scene_id}")
    roots = {path.parent for path in cache_paths}
    if len(roots) != 1:
        raise SRAWShadowError(f"F0 cache root changes within scene: {scene_id}")
    cache_root = next(iter(roots))
    scheduled_path = Path(str(schedule.get("manifest_path", "")))
    if (
        schedule.get("namespace") != CUTR_NAMESPACE
        or cache_root.name != scene_id
        or scheduled_path.resolve() != (cache_root / "manifest.json").resolve()
    ):
        raise SRAWShadowError(f"F0 CuTR schedule binding differs: {scene_id}")
    manifest_path = ledger.add(
        scheduled_path,
        f"CuTR manifest {scene_id}",
        schedule.get("manifest_sha256"),
    )
    _, manifest = _json(manifest_path, f"CuTR manifest {scene_id}")
    records = manifest.get("records")
    if (
        manifest.get("namespace") != CUTR_NAMESPACE
        or not isinstance(records, list)
        or manifest.get("record_count") != len(records)
    ):
        raise SRAWShadowError(f"CuTR manifest contract differs: {scene_id}")
    by_frame: dict[int, Mapping[str, Any]] = {}
    proposal_count = 0
    for row in records:
        if not isinstance(row, Mapping):
            raise SRAWShadowError(f"invalid CuTR manifest record: {scene_id}")
        frame_id = _integer(row.get("frame_id"), "CuTR manifest frame_id")
        count = _integer(row.get("count"), "CuTR manifest count")
        if frame_id in by_frame:
            raise SRAWShadowError(f"duplicate CuTR manifest frame: {scene_id}/{frame_id}")
        by_frame[frame_id] = row
        proposal_count += count
    if list(by_frame) != sorted(by_frame) or manifest.get("recorded_frame_ids") != list(by_frame) or manifest.get("proposal_count") != proposal_count:
        raise SRAWShadowError(f"CuTR manifest census differs: {scene_id}")
    return manifest_path, by_frame


def _load_cutr_world(
    frame: Mapping[str, Any], record: Mapping[str, Any], ledger: _InputLedger
) -> np.ndarray:
    inputs = frame.get("inputs")
    if not isinstance(inputs, Mapping):
        raise SRAWShadowError("F0 frame inputs are absent")
    cache = ledger.add(Path(str(inputs.get("cutr_cache_path", ""))), "CuTR cache", inputs.get("cutr_cache_sha256"))
    pose = ledger.add(Path(str(inputs.get("producer_pose_path", ""))), "CuTR producer pose", inputs.get("producer_pose_sha256"))
    try:
        import torch
        payload = torch.load(cache, map_location="cpu", weights_only=True)
    except Exception as error:
        raise SRAWShadowError(f"cannot safely load CuTR cache: {cache}") from error
    frame_id = _integer(frame.get("frame_id"), "F0 frame_id")
    producer_source = _integer(inputs.get("producer_pose_source_frame_id"), "CuTR producer pose source frame")
    orientation = _integer(inputs.get("producer_orientation"), "CuTR producer orientation")
    if producer_source > frame_id or orientation > 3 or pose.stem != str(producer_source):
        raise SRAWShadowError(f"CuTR producer-pose receipt differs: {cache}")
    if (
        record.get("frame_id") != frame_id
        or record.get("sha256") != inputs.get("cutr_cache_sha256")
        or record.get("count") != inputs.get("cutr_box_count")
    ):
        raise SRAWShadowError(f"CuTR manifest/F0 interlock differs: {cache}")
    if not isinstance(payload, Mapping) or payload.get("schema") != CUTR_CACHE_SCHEMA:
        raise SRAWShadowError(f"CuTR cache schema differs: {cache}")
    for key in ("count", "attempt_id", "input_signature", "protected_hashes", "geometry_sha256"):
        if payload.get(key) != record.get(key):
            raise SRAWShadowError(f"CuTR payload/manifest {key} differs: {cache}")
    fields = payload.get("fields")
    field_metadata = payload.get("field_metadata")
    if (
        tuple(payload.get("field_names", ())) != CUTR_FIELD_NAMES
        or not isinstance(fields, Mapping)
        or tuple(fields) != CUTR_FIELD_NAMES
        or not isinstance(field_metadata, Mapping)
        or tuple(field_metadata) != CUTR_FIELD_NAMES
    ):
        raise SRAWShadowError(f"CuTR field schema differs: {cache}")
    if inputs.get("cutr_input_signature") != record.get("input_signature"):
        raise SRAWShadowError(f"CuTR F0/manifest input signature differs: {cache}")
    box3d = fields.get("pred_boxes_3d") if isinstance(fields, Mapping) else None
    if not isinstance(box3d, Mapping):
        raise SRAWShadowError(f"CuTR 3D boxes absent: {cache}")
    tensor_raw, rotation_raw = box3d.get("tensor"), box3d.get("rotation")
    tensor = np.asarray(tensor_raw.detach().cpu().numpy(), dtype=np.float64)
    rotation = np.asarray(rotation_raw.detach().cpu().numpy(), dtype=np.float64)
    count = _integer(payload.get("count"), "CuTR count")
    if tensor.shape != (count, 6) or rotation.shape != (count, 3, 3) or not np.isfinite(tensor).all() or not np.isfinite(rotation).all() or (count and np.any(tensor[:, 3:6] <= 0.0)):
        raise SRAWShadowError(f"CuTR 3D geometry differs: {cache}")
    metadata = field_metadata.get("pred_boxes_3d", {})
    if _torch_tensor_sha256(tensor_raw) != metadata.get("tensor", {}).get("sha256") or _torch_tensor_sha256(rotation_raw) != metadata.get("rotation", {}).get("sha256"):
        raise SRAWShadowError(f"CuTR 3D tensor hash differs: {cache}")
    try:
        camera_to_world = np.loadtxt(pose, dtype=np.float64).reshape(4, 4)
    except (OSError, ValueError) as error:
        raise SRAWShadowError(f"invalid CuTR producer pose: {pose}") from error
    if not np.isfinite(camera_to_world).all() or not np.allclose(camera_to_world[3], [0, 0, 0, 1], rtol=0.0, atol=1e-6):
        raise SRAWShadowError(f"invalid CuTR producer pose: {pose}")
    pose_tensor = torch.from_numpy(np.ascontiguousarray(camera_to_world.astype(np.float32))).float().contiguous()
    if _torch_tensor_sha256(pose_tensor) != payload.get("input_signature", {}).get("camera_to_world"):
        raise SRAWShadowError(f"CuTR cache/producer pose binding differs: {cache}")
    local = _SIGNS[None] * tensor[:, None, 3:6] * 0.5
    camera = np.einsum("nkj,nij->nki", local, rotation) + tensor[:, None, :3]
    world = camera @ camera_to_world[:3, :3].T + camera_to_world[:3, 3]
    return np.ascontiguousarray(world)


def _atomic_create(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise SRAWShadowError(f"refusing to overwrite output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="ascii") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as error:
            raise SRAWShadowError(f"refusing to overwrite output: {path}") from error
    finally:
        temporary.unlink(missing_ok=True)


def _validate_roots(l2: Mapping[str, Any], f0: Mapping[str, Any], expected_scene_count: int) -> tuple[list[str], list[Mapping[str, Any]], dict[str, Mapping[str, Any]]]:
    scenes = l2.get("scene_order")
    rows = l2.get("scenes")
    if l2.get("schema") != L2_SCHEMA or l2.get("protocol_id") != L2_PROTOCOL_ID or l2.get("complete") is not True or l2.get("overall_pass") is not True or not isinstance(scenes, list) or len(scenes) != expected_scene_count or len(set(scenes)) != expected_scene_count or not isinstance(rows, list) or len(rows) != expected_scene_count:
        raise SRAWShadowError("L2 seal contract differs")
    for key in ("ground_truth_access", "annotation_access", "evaluator_access", "training", "online_learning"):
        if l2.get("contracts", {}).get(key) is not False:
            raise SRAWShadowError(f"L2 contract {key} differs")
    f0_order = f0.get("coverage", {}).get("scene_order")
    f0_rows = f0.get("scenes")
    if f0.get("schema") != F0_MERGE_SCHEMA or f0.get("complete") is not True or not isinstance(f0_order, list) or f0_order[:expected_scene_count] != scenes or not isinstance(f0_rows, list):
        raise SRAWShadowError("F0 sealed cache index contract differs")
    for key in ("ground_truth_access", "annotation_access", "evaluator_access", "training", "online_learning"):
        if f0.get("contracts", {}).get(key) is not False:
            raise SRAWShadowError(f"F0 contract {key} differs")
    by_scene = {str(row.get("scene_id")): row for row in f0_rows if isinstance(row, Mapping)}
    if any(scene not in by_scene for scene in scenes):
        raise SRAWShadowError("F0 cache index misses a paper100 scene")
    return [str(item) for item in scenes], rows, by_scene


def run_shadow(
    *, l2_seal: Path = DEFAULT_L2, f0_manifest: Path = DEFAULT_F0,
    protocol_path: Path = DEFAULT_PROTOCOL, clip_checkpoint: Path = ROOT / "models/open_clip_pytorch_model.bin",
    class_features: Path = ROOT / "data/class_features.pt", vocabulary_path: Path = ROOT / "data/panoptic_categories_nomerge.txt",
    output_path: Path = DEFAULT_OUTPUT, device: str = "cuda:0", batch_size: int = 32,
    expected_scene_count: int = 100, plan_only: bool = False,
) -> dict[str, Any]:
    if batch_size < 1 or expected_scene_count < 1:
        raise SRAWShadowError("batch size and scene count must be positive")
    if not plan_only and (output_path.exists() or output_path.is_symlink()):
        raise SRAWShadowError(f"refusing to overwrite output: {output_path}")
    ledger = _InputLedger()
    l2_path = ledger.add(l2_seal, "L2 seal")
    f0_path = ledger.add(f0_manifest, "F0 merged cache receipt")
    protocol = ledger.add(protocol_path, "frozen protocol")
    _, l2 = _json(l2_path, "L2 seal")
    _, f0 = _json(f0_path, "F0 merged cache receipt")
    scenes, l2_rows, f0_by_scene = _validate_roots(l2, f0, expected_scene_count)

    prepared_scenes: list[dict[str, Any]] = []
    eligible_count = geometry_pass_count = 0
    for scene_index, (scene_id, l2_row) in enumerate(zip(scenes, l2_rows, strict=True)):
        if not isinstance(l2_row, Mapping) or l2_row.get("scene_id") != scene_id or l2_row.get("scene_index") != scene_index:
            raise SRAWShadowError(f"L2 scene order differs: {scene_id}")
        f4_receipt = l2_row.get("f4")
        if not isinstance(f4_receipt, Mapping):
            raise SRAWShadowError(f"L2 F4 receipt absent: {scene_id}")
        f4_path = ledger.add(Path(str(f4_receipt.get("path", ""))), f"F4 {scene_id}", f4_receipt.get("sha256"))
        _, f4 = _json(f4_path, f"F4 {scene_id}")
        if f4.get("schema") != F4_SCHEMA or f4.get("complete") is not True or f4.get("contracts", {}).get("gt_access") is not False or f4.get("contracts", {}).get("prediction_access") is not False or f4.get("contracts", {}).get("evaluator_access") is not False:
            raise SRAWShadowError(f"F4 contract differs: {scene_id}")
        sources, frames = _source_map(f4, scene_id)
        order = l2_row.get("f4_source_order")
        tracks = l2_row.get("tracks")
        if not isinstance(order, list) or [str(item) for item in order] != list(sources) or not isinstance(tracks, list):
            raise SRAWShadowError(f"L2/F4 source order differs: {scene_id}")
        order_index = {source_id: index for index, source_id in enumerate(sources)}
        candidates: list[dict[str, Any]] = []
        for track_id, track in enumerate(tracks):
            if not isinstance(track, Mapping) or track.get("track_id") != track_id:
                raise SRAWShadowError(f"L2 track order differs: {scene_id}/{track_id}")
            evidence_ids = _first_three_sources(track, order_index, sources, scene_id)
            if len(evidence_ids) < 3:
                continue
            geometry = _geometry_summary(evidence_ids, sources)
            candidate = {
                "track_id": track_id,
                "evidence_source_ids": evidence_ids,
                "evidence_frame_ids": [_source_frame(item, scene_id) for item in evidence_ids],
                "confirmation_frame_id": _source_frame(evidence_ids[2], scene_id),
                "geometry": geometry,
            }
            candidates.append(candidate)
            eligible_count += 1
            geometry_pass_count += int(geometry["gate_pass"])
        f0_row = f0_by_scene[scene_id]
        sidecar_receipt = f0_row.get("sidecar")
        if not isinstance(sidecar_receipt, Mapping):
            raise SRAWShadowError(f"F0 sidecar receipt absent: {scene_id}")
        f0_scene_path = ledger.add(Path(str(sidecar_receipt.get("path", ""))), f"F0 scene {scene_id}", sidecar_receipt.get("sha256"))
        _, f0_scene = _json(f0_scene_path, f"F0 scene {scene_id}")
        if f0_scene.get("schema") != F0_SCENE_SCHEMA or f0_scene.get("complete") is not True:
            raise SRAWShadowError(f"F0 scene contract differs: {scene_id}")
        prepared_scenes.append({"scene_id": scene_id, "scene_index": scene_index, "candidates": candidates, "sources": sources, "frames": frames, "f4_path": f4_path, "f0_scene_path": f0_scene_path, "f0_scene": f0_scene})

    plan = {"schema": SCHEMA, "protocol_id": PROTOCOL_ID, "mode": "plan_only", "scene_count": len(scenes), "p3_eligible_track_count": eligible_count, "geometry_pass_track_count": geometry_pass_count, "clip_crop_count": geometry_pass_count * 3, "contracts": dict(CONTRACTS)}
    if plan_only:
        ledger.verify()
        print(json.dumps(plan, sort_keys=True), flush=True)
        return plan

    checkpoint = ledger.add(clip_checkpoint, "CLIP checkpoint")
    features_path = ledger.add(class_features, "CLIP class features")
    vocab_path = ledger.add(vocabulary_path, "native CLIP vocabulary")
    vocabulary = vocab_path.read_text(encoding="utf-8").splitlines()
    if len(vocabulary) != EXPECTED_VOCABULARY_SIZE:
        raise SRAWShadowError("native CLIP vocabulary size differs")
    target_groups, target_indices, non_target_indices = _resolve_target_indices(vocabulary)
    index_groups: dict[int, list[str]] = defaultdict(list)
    for group, indices in target_groups.items():
        for index in indices:
            index_groups[index].append(group)
    model, preprocess = _load_clip_runtime(checkpoint, device)
    text_features = _load_text_features(features_path, device)

    output_scenes: list[dict[str, Any]] = []
    total_accepted = 0
    decision_counts: Counter[str] = Counter()
    for scene_row in prepared_scenes:
        tasks: list[tuple[int, int, MutableMapping[str, Any], Sequence[float], Path]] = []
        for candidate in scene_row["candidates"]:
            if not candidate["geometry"]["gate_pass"]:
                continue
            evidence: list[dict[str, Any]] = []
            for evidence_index, source_id in enumerate(candidate["evidence_source_ids"]):
                source = scene_row["sources"][source_id]
                frame_id = candidate["evidence_frame_ids"][evidence_index]
                frame = scene_row["frames"][frame_id]
                rgb_receipt = frame.get("input", {}).get("rgb")
                if not isinstance(rgb_receipt, Mapping):
                    raise SRAWShadowError(f"sealed F4 RGB absent: {source_id}")
                rgb_path = ledger.add(Path(str(rgb_receipt.get("path", ""))), f"F4 RGB {source_id}", rgb_receipt.get("sha256"))
                row: dict[str, Any] = {"source_id": source_id, "frame_id": frame_id, "tight_box_xyxy": list(source.get("tight_box_xyxy", []))}
                evidence.append(row)
                tasks.append((frame_id, evidence_index, row, row["tight_box_xyxy"], rgb_path))
            candidate["clip_evidence"] = evidence
        tensors: list[Any] = []
        destinations: list[MutableMapping[str, Any]] = []
        cached_path: Path | None = None
        cached_rgb: np.ndarray | None = None
        for _, _, destination, bbox, rgb_path in sorted(tasks, key=lambda item: (item[0], item[2]["source_id"], item[1])):
            if rgb_path != cached_path:
                bgr = cv2.imread(os.fspath(rgb_path), cv2.IMREAD_COLOR)
                if bgr is None or bgr.ndim != 3 or bgr.shape[2] != 3:
                    raise SRAWShadowError(f"cannot decode F4 RGB: {rgb_path}")
                cached_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                cached_path = rgb_path
            assert cached_rgb is not None
            tensors.append(preprocess(_crop_rgb(cached_rgb, bbox)))
            destinations.append(destination)
            if len(tensors) >= batch_size:
                _flush_batch(tensors=tensors, destinations=destinations, model=model, text_features=text_features, vocabulary=vocabulary, target_indices=target_indices, non_target_indices=non_target_indices, target_index_groups=index_groups, device=device)
        _flush_batch(tensors=tensors, destinations=destinations, model=model, text_features=text_features, vocabulary=vocabulary, target_indices=target_indices, non_target_indices=non_target_indices, target_index_groups=index_groups, device=device)

        admitted: list[MutableMapping[str, Any]] = []
        pre_decisions: list[dict[str, Any]] = []
        for candidate in scene_row["candidates"]:
            if not candidate["geometry"]["gate_pass"]:
                pre_decisions.append({"track_id": candidate["track_id"], "confirmation_frame_id": candidate["confirmation_frame_id"], "decision": "geometry_gate", "reasons": candidate["geometry"]["gate_rejection_reasons"]})
                decision_counts["geometry_gate"] += 1
                continue
            semantic = _semantic_summary(candidate["clip_evidence"])
            candidate["semantic"] = semantic
            if not semantic["gate_pass"]:
                pre_decisions.append({"track_id": candidate["track_id"], "confirmation_frame_id": candidate["confirmation_frame_id"], "decision": "semantic_gate", "reasons": semantic["gate_rejection_reasons"]})
                decision_counts["semantic_gate"] += 1
                continue
            admitted.append(candidate)

        max_confirmation = max((row["confirmation_frame_id"] for row in admitted), default=-1)
        cutr_by_frame: dict[int, np.ndarray] = {}
        f0_frames = scene_row["f0_scene"].get("frames")
        if not isinstance(f0_frames, list):
            raise SRAWShadowError(f"F0 frames absent: {scene_row['scene_id']}")
        schedule = scene_row["f0_scene"].get("schedule")
        if not isinstance(schedule, Mapping):
            raise SRAWShadowError(
                f"F0 CuTR schedule absent: {scene_row['scene_id']}"
            )
        _, cache_records = _cache_manifest_records(
            f0_frames, scene_row["scene_id"], schedule, ledger
        )
        for ordinal, frame in enumerate(f0_frames):
            if not isinstance(frame, Mapping) or frame.get("frame_ordinal") != ordinal:
                raise SRAWShadowError(f"F0 frame order differs: {scene_row['scene_id']}")
            frame_id = _integer(frame.get("frame_id"), "F0 frame_id")
            if frame_id <= max_confirmation:
                record = cache_records.get(frame_id)
                if record is None:
                    raise SRAWShadowError(
                        f"CuTR manifest misses F0 frame: {scene_row['scene_id']}/{frame_id}"
                    )
                cutr_by_frame[frame_id] = _load_cutr_world(
                    frame, record, ledger
                )
        accepted, novelty_decisions = _admit_candidates(admitted, cutr_by_frame)
        for row in novelty_decisions:
            decision_counts[row["decision"]] += 1
        total_accepted += len(accepted)
        output_scenes.append({"scene_id": scene_row["scene_id"], "scene_index": scene_row["scene_index"], "p3_eligible_track_count": len(scene_row["candidates"]), "accepted_births": accepted, "decisions": pre_decisions + novelty_decisions})
        print(f"[{scene_row['scene_index'] + 1}/{len(scenes)}] {scene_row['scene_id']}: {len(accepted)} shadow births", flush=True)

    ledger.verify()
    output = {
        "schema": SCHEMA,
        "protocol_id": PROTOCOL_ID,
        "mode": "shadow_output_inert",
        "complete": True,
        "scene_count": len(scenes),
        "scene_order": scenes,
        "accepted_birth_count": total_accepted,
        "counts": {"p3_eligible_tracks": eligible_count, "geometry_pass_tracks": geometry_pass_count, "decisions": dict(sorted(decision_counts.items()))},
        "contracts": dict(CONTRACTS),
        "policy": {"causal_evidence": "first_three_distinct_source_frames", "geometry": GEOMETRY_POLICY, "semantic": SEMANTIC_POLICY, "novelty": NOVELTY_POLICY, "clip_model": "frozen_OpenCLIP_ViT-H-14_native_473_vocab", "equal_frame_order": ["geometry_support", "semantic_votes_and_cosine", "hb_confidence", "stable_track_source_identity"]},
        "inputs": {"l2_seal": os.fspath(l2_path), "l2_seal_sha256": _sha256(l2_path), "f0_manifest": os.fspath(f0_path), "f0_manifest_sha256": _sha256(f0_path), "protocol": os.fspath(protocol), "protocol_sha256": _sha256(protocol), "clip_checkpoint": os.fspath(checkpoint), "clip_checkpoint_sha256": _sha256(checkpoint), "class_features": os.fspath(features_path), "class_features_sha256": _sha256(features_path), "vocabulary": os.fspath(vocab_path), "vocabulary_sha256": _sha256(vocab_path), "consumed_input_ledger": ledger.receipt()},
        "scenes": output_scenes,
    }
    _atomic_create(output_path, output)
    print(f"Saved: {output_path}", flush=True)
    return output


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--l2-seal", type=Path, default=DEFAULT_L2)
    parser.add_argument("--f0-manifest", type=Path, default=DEFAULT_F0)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--clip-checkpoint", type=Path, default=ROOT / "models/open_clip_pytorch_model.bin")
    parser.add_argument("--class-features", type=Path, default=ROOT / "data/class_features.pt")
    parser.add_argument("--vocabulary", type=Path, default=ROOT / "data/panoptic_categories_nomerge.txt")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--expected-scene-count", type=int, default=100)
    parser.add_argument("--plan-only", action="store_true")
    return parser


def main() -> None:
    args = _parser().parse_args()
    run_shadow(l2_seal=args.l2_seal, f0_manifest=args.f0_manifest, protocol_path=args.protocol, clip_checkpoint=args.clip_checkpoint, class_features=args.class_features, vocabulary_path=args.vocabulary, output_path=args.output, device=args.device, batch_size=args.batch_size, expected_scene_count=args.expected_scene_count, plan_only=args.plan_only)


if __name__ == "__main__":
    main()
