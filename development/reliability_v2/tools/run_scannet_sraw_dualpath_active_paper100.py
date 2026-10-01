#!/usr/bin/env python3
"""Materialize SRAW dual-path HB births and run paper100-ready predictions.

This is the active counterpart for the frozen F0/F4 Boxer-HB candidate route:

* S path: one high-quality HB source, admitted at its own frame.
* P path: two causal HB sources from distinct frames, admitted at the second.

The runner never reads annotations or evaluator state.  It consumes sealed L2/F4
source identities, sealed F0 CuTR cache receipts, and the existing frozen native
CLIP vocabulary/features.  Native predictions are read only at materialization
time to preserve their row prefix and to avoid appending terminal duplicates.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import math
import os
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any, Mapping, MutableMapping, Sequence

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if os.fspath(ROOT) not in sys.path:
    sys.path.insert(0, os.fspath(ROOT))

from tools import run_scannet_sraw_p3hb_clip_shadow_paper100 as p3  # noqa: E402
from tools.materialize_scannet_raw_boxer_past3_birth_full100 import (  # noqa: E402
    APPENDED_CLASS_ID,
    APPENDED_SCORE,
    PREDICTION_SUFFIX,
    _assert_native_prefix,
    _load_native_prediction,
    _scene_list,
    _write_json,
    _write_pickle,
)
from tools.run_scannet_raw_boxer_clip_vocab_shadow_full100 import (  # noqa: E402
    EXPECTED_VOCABULARY_SIZE,
    _crop_rgb,
    _flush_batch,
    _load_clip_runtime,
    _load_text_features,
    _resolve_target_indices,
)
from tools.seal_scannet_l0_f3_f4_perview_paper100 import F4_SCHEMA, SOURCE_RE  # noqa: E402


SCHEMA = "boxfusion.scannet_sraw_dualpath_active_paper100.v1"
PROTOCOL_ID = "SRAW-DUALPATH-HB-CLIP-BIRTH-V1"

DEFAULT_OUTPUT_ROOT = ROOT / "results/scannet_sraw_dualpath_birth_score05"
DEFAULT_MANIFEST_NAME = "SRAW_DUALPATH_BIRTH_PAPER100.json"
DEFAULT_BASELINE_ROOT = ROOT / "results/scannet_t05_boxer_replay_active_score05"
DEFAULT_SCENE_LIST = ROOT / "evaluation/data_util/meta_data/scannetv2_val.txt"

S_POLICY = {
    "hb_confidence_gte": 0.70,
    "hb_each_local_extent_m_gte": 0.20,
    "hb_h0_normalized_center_distance_lte": 0.50,
    "hb_h0_volume_ratio": [0.25, 4.00],
    "hb_h0_aabb_iou_gte": 0.20,
    "hb_h0_bidirectional_max_containment_gte": 0.70,
}
P_POLICY = {
    "anchor_hb_confidence_gte": 0.65,
    "supporter_hb_confidence_gte": 0.55,
    "anchor_hb_each_local_extent_m_gte": 0.20,
    "anchor_hb_h0_normalized_center_distance_lte": 0.50,
    "anchor_hb_h0_volume_ratio": [0.25, 4.00],
    "anchor_hb_h0_aabb_iou_gte": 0.20,
    "anchor_hb_h0_bidirectional_max_containment_gte": 0.70,
    "pair_hb_aabb_iou_gte": 0.20,
    "pair_hb_bidirectional_max_containment_gte": 0.60,
    "pair_hb_center_distance_m_lte": 0.30,
}
SEMANTIC_POLICY = {
    "single_all_vocab_top1_target": True,
    "pair_all_vocab_top1_target_votes": 2,
    "pair_same_target_alias_group_votes": 2,
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
TERMINAL_POLICY = {
    "native_aabb_iou_gte_reject": 0.10,
    "native_bidirectional_containment_gte_reject": 0.50,
    "self_nms_aabb_iou_gte_reject": 0.15,
    "self_nms_bidirectional_containment_gte_reject": 0.25,
}
CONTRACTS: Mapping[str, bool] = {
    "birth_enabled": True,
    "native_prefix_preserved": True,
    "ground_truth_access": False,
    "annotation_access": False,
    "evaluator_access": False,
    "training": False,
    "online_learning": False,
    "past_only": True,
}


class DualPathActiveError(RuntimeError):
    """Raised when a frozen input or active-policy invariant differs."""


def _check_f4(scene_id: str, f4: Mapping[str, Any]) -> None:
    contracts = f4.get("contracts", {})
    if (
        f4.get("schema") != F4_SCHEMA
        or f4.get("complete") is not True
        or contracts.get("gt_access") is not False
        or contracts.get("prediction_access") is not False
        or contracts.get("evaluator_access") is not False
    ):
        raise DualPathActiveError(f"F4 contract differs: {scene_id}")


def _first_distinct_sources(
    track: Mapping[str, Any],
    order_index: Mapping[str, int],
    sources: Mapping[str, Mapping[str, Any]],
    scene_id: str,
    count: int,
) -> list[str]:
    raw = track.get("source_ids")
    if not isinstance(raw, list) or len(raw) != len(set(map(str, raw))):
        raise DualPathActiveError(f"invalid track source ledger: {scene_id}")
    normalized = [str(item) for item in raw]
    if any(item not in sources or item not in order_index for item in normalized):
        raise DualPathActiveError(f"track source absent from F4: {scene_id}")
    selected: list[str] = []
    seen_frames: set[int] = set()
    for source_id in sorted(normalized, key=order_index.__getitem__):
        frame_id = p3._source_frame(source_id, scene_id)
        if frame_id in seen_frames:
            continue
        selected.append(source_id)
        seen_frames.add(frame_id)
        if len(selected) == count:
            break
    frames = [p3._source_frame(source_id, scene_id) for source_id in selected]
    if frames != sorted(frames) or len(frames) != len(set(frames)):
        raise DualPathActiveError(f"causal source order differs: {scene_id}")
    return selected


def _source_metrics(
    source_id: str, sources: Mapping[str, Mapping[str, Any]]
) -> dict[str, Any]:
    source = sources[source_id]
    corners, center, extent, confidence = p3._hb(source, source_id)
    h0 = source.get("hypotheses", {}).get("H0")
    if not isinstance(h0, Mapping) or h0.get("valid") is not True:
        raise DualPathActiveError(f"valid H0 missing: {source_id}")
    h0_lower = p3._array(h0.get("q02"), (3,), f"{source_id}.H0 q02")
    h0_upper = p3._array(h0.get("q98"), (3,), f"{source_id}.H0 q98")
    if np.any(h0_upper <= h0_lower):
        raise DualPathActiveError(f"degenerate H0: {source_id}")
    h0_corners = p3._aabb_corners(h0_lower, h0_upper)
    iou, hb_in_h0, h0_in_hb = p3._aabb_overlap_matrices(
        corners[None], h0_corners[None]
    )
    hb_lower, hb_upper = corners.min(0), corners.max(0)
    hb_center = (hb_lower + hb_upper) * 0.5
    h0_center = (h0_lower + h0_upper) * 0.5
    scale = max(
        float(np.linalg.norm(hb_upper - hb_lower)),
        float(np.linalg.norm(h0_upper - h0_lower)),
        0.02,
    )
    ratio = float(np.prod(hb_upper - hb_lower) / np.prod(h0_upper - h0_lower))
    return {
        "source_id": source_id,
        "corners": corners,
        "center": center,
        "extent": extent,
        "confidence": float(confidence),
        "hb_h0_normalized_center_distance": float(np.linalg.norm(hb_center - h0_center) / scale),
        "hb_h0_volume_ratio": ratio,
        "hb_h0_aabb_iou": float(iou[0, 0]),
        "hb_h0_bidirectional_max_containment": float(max(hb_in_h0[0, 0], h0_in_hb[0, 0])),
    }


def _single_geometry(
    source_id: str, sources: Mapping[str, Mapping[str, Any]]
) -> dict[str, Any]:
    row = _source_metrics(source_id, sources)
    checks = {
        "hb_confidence": row["confidence"] >= S_POLICY["hb_confidence_gte"],
        "hb_each_local_extent": bool(np.all(row["extent"] >= S_POLICY["hb_each_local_extent_m_gte"])),
        "hb_h0_normalized_center_distance": row["hb_h0_normalized_center_distance"] <= S_POLICY["hb_h0_normalized_center_distance_lte"],
        "hb_h0_volume_ratio": S_POLICY["hb_h0_volume_ratio"][0] <= row["hb_h0_volume_ratio"] <= S_POLICY["hb_h0_volume_ratio"][1],
        "hb_h0_overlap": (
            row["hb_h0_aabb_iou"] >= S_POLICY["hb_h0_aabb_iou_gte"]
            or row["hb_h0_bidirectional_max_containment"] >= S_POLICY["hb_h0_bidirectional_max_containment_gte"]
        ),
    }
    return {
        "gate_pass": all(checks.values()),
        "gate_checks": checks,
        "gate_rejection_reasons": [name for name, passed in checks.items() if not passed],
        "selected_source_id": source_id,
        "corners_world": row["corners"].tolist(),
        "selected_hb_confidence": row["confidence"],
        "selected_hb_local_extent_m": row["extent"].tolist(),
        "selected_hb_h0_normalized_center_distance": row["hb_h0_normalized_center_distance"],
        "selected_hb_h0_volume_ratio": row["hb_h0_volume_ratio"],
        "selected_hb_h0_aabb_iou": row["hb_h0_aabb_iou"],
        "selected_hb_h0_bidirectional_max_containment": row["hb_h0_bidirectional_max_containment"],
        "support_score": row["confidence"],
    }


def _pair_geometry(
    source_ids: Sequence[str], sources: Mapping[str, Mapping[str, Any]]
) -> dict[str, Any]:
    if len(source_ids) != 2:
        raise DualPathActiveError("P path requires exactly two sources")
    rows = [_source_metrics(source_id, sources) for source_id in source_ids]
    anchor_index = max(range(2), key=lambda index: (rows[index]["confidence"], -index))
    support_index = 1 - anchor_index
    anchor = rows[anchor_index]
    supporter = rows[support_index]
    iou, left, right = p3._aabb_overlap_matrices(
        rows[0]["corners"][None], rows[1]["corners"][None]
    )
    pair_iou = float(iou[0, 0])
    pair_containment = float(max(left[0, 0], right[0, 0]))
    pair_center_distance = float(np.linalg.norm(rows[0]["center"] - rows[1]["center"]))
    checks = {
        "anchor_hb_confidence": anchor["confidence"] >= P_POLICY["anchor_hb_confidence_gte"],
        "supporter_hb_confidence": supporter["confidence"] >= P_POLICY["supporter_hb_confidence_gte"],
        "anchor_hb_each_local_extent": bool(np.all(anchor["extent"] >= P_POLICY["anchor_hb_each_local_extent_m_gte"])),
        "anchor_hb_h0_normalized_center_distance": anchor["hb_h0_normalized_center_distance"] <= P_POLICY["anchor_hb_h0_normalized_center_distance_lte"],
        "anchor_hb_h0_volume_ratio": P_POLICY["anchor_hb_h0_volume_ratio"][0] <= anchor["hb_h0_volume_ratio"] <= P_POLICY["anchor_hb_h0_volume_ratio"][1],
        "anchor_hb_h0_overlap": (
            anchor["hb_h0_aabb_iou"] >= P_POLICY["anchor_hb_h0_aabb_iou_gte"]
            or anchor["hb_h0_bidirectional_max_containment"] >= P_POLICY["anchor_hb_h0_bidirectional_max_containment_gte"]
        ),
        "pair_hb_overlap": (
            pair_iou >= P_POLICY["pair_hb_aabb_iou_gte"]
            or pair_containment >= P_POLICY["pair_hb_bidirectional_max_containment_gte"]
        ),
        "pair_hb_center_distance": pair_center_distance <= P_POLICY["pair_hb_center_distance_m_lte"],
    }
    return {
        "gate_pass": all(checks.values()),
        "gate_checks": checks,
        "gate_rejection_reasons": [name for name, passed in checks.items() if not passed],
        "selected_source_id": str(anchor["source_id"]),
        "selected_source_ordinal": anchor_index,
        "corners_world": anchor["corners"].tolist(),
        "hb_confidences": [rows[0]["confidence"], rows[1]["confidence"]],
        "selected_hb_confidence": anchor["confidence"],
        "supporter_hb_confidence": supporter["confidence"],
        "pair_hb_aabb_iou": pair_iou,
        "pair_hb_bidirectional_max_containment": pair_containment,
        "pair_hb_center_distance_m": pair_center_distance,
        "selected_hb_local_extent_m": anchor["extent"].tolist(),
        "selected_hb_h0_normalized_center_distance": anchor["hb_h0_normalized_center_distance"],
        "selected_hb_h0_volume_ratio": anchor["hb_h0_volume_ratio"],
        "selected_hb_h0_aabb_iou": anchor["hb_h0_aabb_iou"],
        "selected_hb_h0_bidirectional_max_containment": anchor["hb_h0_bidirectional_max_containment"],
        "support_score": pair_iou + pair_containment + min(rows[0]["confidence"], rows[1]["confidence"]),
    }


def _candidate_semantic(evidence: Sequence[Mapping[str, Any]], channel: str) -> dict[str, Any]:
    if channel == "S" and len(evidence) != 1:
        raise DualPathActiveError("S semantic gate requires one CLIP row")
    if channel == "P" and len(evidence) != 2:
        raise DualPathActiveError("P semantic gate requires two CLIP rows")
    target_votes = 0
    groups: Counter[str] = Counter()
    for row in evidence:
        aliases = row.get("all_vocab_top1_target_alias_groups")
        if not isinstance(aliases, list):
            raise DualPathActiveError("CLIP alias groups must be a list")
        if bool(row.get("all_vocab_top1_is_target")):
            if len(aliases) != 1:
                raise DualPathActiveError("target top-1 must resolve to one alias group")
            target_votes += 1
            groups[str(aliases[0])] += 1
        elif aliases:
            raise DualPathActiveError("non-target top-1 cannot cast an alias vote")
    group, group_votes = (None, 0)
    if groups:
        group, group_votes = sorted(groups.items(), key=lambda item: (-item[1], item[0]))[0]
    cosines = [p3._number(row.get("target_best_cosine"), "target cosine") for row in evidence]
    margins = [p3._number(row.get("target_non_target_margin"), "target margin") for row in evidence]
    median_cosine = float(np.median(cosines))
    median_margin = float(np.median(margins))
    if channel == "S":
        checks = {
            "all_vocab_top1_target": target_votes == 1,
            "single_target_alias_group": group_votes == 1,
            "median_best_target_cosine": median_cosine >= SEMANTIC_POLICY["median_best_target_cosine_gte"],
            "median_target_non_target_margin": median_margin >= SEMANTIC_POLICY["median_target_non_target_margin_gte"],
        }
    else:
        checks = {
            "all_vocab_top1_target_votes": target_votes >= SEMANTIC_POLICY["pair_all_vocab_top1_target_votes"],
            "same_target_alias_group_votes": group_votes >= SEMANTIC_POLICY["pair_same_target_alias_group_votes"],
            "median_best_target_cosine": median_cosine >= SEMANTIC_POLICY["median_best_target_cosine_gte"],
            "median_target_non_target_margin": median_margin >= SEMANTIC_POLICY["median_target_non_target_margin_gte"],
        }
    return {
        "gate_pass": all(checks.values()),
        "gate_checks": checks,
        "gate_rejection_reasons": [name for name, passed in checks.items() if not passed],
        "all_vocab_top1_target_votes": target_votes,
        "target_group": group,
        "same_target_alias_group_votes": group_votes,
        "median_best_target_cosine": median_cosine,
        "median_target_non_target_margin": median_margin,
    }


def _order_candidates(candidates: Sequence[MutableMapping[str, Any]]) -> list[MutableMapping[str, Any]]:
    channel_priority = {"P": 0, "S": 1}
    return sorted(
        candidates,
        key=lambda row: (
            int(row["confirmation_frame_id"]),
            channel_priority[str(row["channel"])],
            -int(row["semantic"]["same_target_alias_group_votes"]),
            -int(row["semantic"]["all_vocab_top1_target_votes"]),
            -float(row["semantic"]["median_best_target_cosine"]),
            -float(row["semantic"]["median_target_non_target_margin"]),
            -float(row["geometry"]["support_score"]),
            int(row["track_id"]),
            str(row["channel"]),
            str(row["geometry"]["selected_source_id"]),
        ),
    )


def _admit_causal(
    candidates: Sequence[MutableMapping[str, Any]],
    cutr_by_frame: Mapping[int, np.ndarray],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    accepted: list[dict[str, Any]] = []
    decisions: list[dict[str, Any]] = []
    accumulated: list[np.ndarray] = []
    cache_frames = sorted(cutr_by_frame)
    cache_index = 0
    for candidate in _order_candidates(candidates):
        confirmation = int(candidate["confirmation_frame_id"])
        while cache_index < len(cache_frames) and cache_frames[cache_index] <= confirmation:
            rows = cutr_by_frame[cache_frames[cache_index]]
            if len(rows):
                accumulated.append(rows)
            cache_index += 1
        corners = np.asarray(candidate["geometry"]["corners_world"], dtype=np.float64)[None]
        cutr = (
            np.concatenate(accumulated, axis=0)
            if accumulated
            else np.empty((0, 8, 3), dtype=np.float64)
        )
        cutr_iou, cutr_left, cutr_right = p3._overlap_max(corners, cutr)
        prior = (
            np.stack([np.asarray(row["corners_world"], dtype=np.float64) for row in accepted])
            if accepted
            else np.empty((0, 8, 3), dtype=np.float64)
        )
        birth_iou, birth_left, birth_right = p3._overlap_max(corners, prior)
        decision = "accepted"
        if cutr_iou >= NOVELTY_POLICY["cutr_aabb_iou_gte_reject"] or max(cutr_left, cutr_right) >= NOVELTY_POLICY["cutr_bidirectional_containment_gte_reject"]:
            decision = "cutr_overlap"
        elif birth_iou >= NOVELTY_POLICY["past_birth_aabb_iou_gte_reject"] or max(birth_left, birth_right) >= NOVELTY_POLICY["past_birth_bidirectional_containment_gte_reject"]:
            decision = "past_birth_nms"
        elif len(accepted) >= NOVELTY_POLICY["max_births_per_scene"]:
            decision = "scene_cap"
        row = {
            "channel": candidate["channel"],
            "track_id": candidate["track_id"],
            "confirmation_frame_id": confirmation,
            "evidence_source_ids": list(candidate["evidence_source_ids"]),
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
                    "channel": candidate["channel"],
                    "track_id": candidate["track_id"],
                    "confirmation_frame_id": confirmation,
                    "evidence_source_ids": list(candidate["evidence_source_ids"]),
                    "evidence_frame_ids": list(candidate["evidence_frame_ids"]),
                    "selected_source_id": candidate["geometry"]["selected_source_id"],
                    "target_group": candidate["semantic"]["target_group"],
                    "corners_world": candidate["geometry"]["corners_world"],
                    "geometry": dict(candidate["geometry"]),
                    "semantic": dict(candidate["semantic"]),
                }
            )
    return accepted, decisions


def _terminal_select(
    native_corners: np.ndarray, accepted: Sequence[Mapping[str, Any]]
) -> tuple[list[Mapping[str, Any]], list[dict[str, Any]]]:
    selected: list[Mapping[str, Any]] = []
    decisions: list[dict[str, Any]] = []
    for candidate in accepted:
        corners = np.asarray(candidate["corners_world"], dtype=np.float64)[None]
        native_iou, native_left, native_right = p3._overlap_max(corners, native_corners)
        prior = (
            np.stack([np.asarray(row["corners_world"], dtype=np.float64) for row in selected])
            if selected
            else np.empty((0, 8, 3), dtype=np.float64)
        )
        self_iou, self_left, self_right = p3._overlap_max(corners, prior)
        decision = "accepted"
        if native_iou >= TERMINAL_POLICY["native_aabb_iou_gte_reject"] or max(native_left, native_right) >= TERMINAL_POLICY["native_bidirectional_containment_gte_reject"]:
            decision = "terminal_native_overlap"
        elif self_iou >= TERMINAL_POLICY["self_nms_aabb_iou_gte_reject"] or max(self_left, self_right) >= TERMINAL_POLICY["self_nms_bidirectional_containment_gte_reject"]:
            decision = "terminal_self_nms"
        row = {
            "channel": candidate["channel"],
            "track_id": candidate["track_id"],
            "confirmation_frame_id": candidate["confirmation_frame_id"],
            "selected_source_id": candidate["selected_source_id"],
            "decision": decision,
            "max_native_aabb_iou": native_iou,
            "max_candidate_in_native_containment": native_left,
            "max_native_in_candidate_containment": native_right,
            "max_terminal_birth_aabb_iou": self_iou,
            "max_candidate_in_terminal_birth_containment": self_left,
            "max_terminal_birth_in_candidate_containment": self_right,
        }
        decisions.append(row)
        if decision == "accepted":
            selected.append(candidate)
    return selected, decisions


def _augment_payload(native: Any, selected: Sequence[Mapping[str, Any]]) -> list[Any] | tuple[Any, ...]:
    suffix = [
        (
            APPENDED_CLASS_ID,
            np.ascontiguousarray(row["corners_world"], dtype=np.float32),
            APPENDED_SCORE,
        )
        for row in selected
    ]
    rows: list[Any] | tuple[Any, ...]
    if isinstance(native.rows, tuple):
        rows = tuple(native.rows) + tuple(suffix)
    else:
        rows = list(native.rows) + suffix
    output: list[Any] | tuple[Any, ...]
    if isinstance(native.payload, tuple):
        output = (rows,)
    else:
        output = [rows]
    _assert_native_prefix(native.rows, output[0], "in-memory dualpath output")
    return output


def _prepare_candidates(
    scene_id: str,
    l2_row: Mapping[str, Any],
    ledger: p3._InputLedger,
) -> tuple[dict[str, Any], Counter[str]]:
    f4_receipt = l2_row.get("f4")
    if not isinstance(f4_receipt, Mapping):
        raise DualPathActiveError(f"L2 F4 receipt absent: {scene_id}")
    f4_path = ledger.add(Path(str(f4_receipt.get("path", ""))), f"F4 {scene_id}", f4_receipt.get("sha256"))
    _, f4 = p3._json(f4_path, f"F4 {scene_id}")
    _check_f4(scene_id, f4)
    sources, frames = p3._source_map(f4, scene_id)
    order = l2_row.get("f4_source_order")
    tracks = l2_row.get("tracks")
    if not isinstance(order, list) or [str(item) for item in order] != list(sources) or not isinstance(tracks, list):
        raise DualPathActiveError(f"L2/F4 source order differs: {scene_id}")
    order_index = {source_id: index for index, source_id in enumerate(sources)}
    candidates: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    for track_id, track in enumerate(tracks):
        if not isinstance(track, Mapping) or track.get("track_id") != track_id:
            raise DualPathActiveError(f"L2 track order differs: {scene_id}/{track_id}")
        first_one = _first_distinct_sources(track, order_index, sources, scene_id, 1)
        if first_one:
            frame_ids = [p3._source_frame(first_one[0], scene_id)]
            geometry = _single_geometry(first_one[0], sources)
            counts["S_total"] += 1
            counts[f"S_geometry_{'pass' if geometry['gate_pass'] else 'fail'}"] += 1
            if geometry["gate_pass"]:
                candidates.append(
                    {
                        "channel": "S",
                        "track_id": track_id,
                        "evidence_source_ids": first_one,
                        "evidence_frame_ids": frame_ids,
                        "confirmation_frame_id": frame_ids[0],
                        "geometry": geometry,
                    }
                )
        first_two = _first_distinct_sources(track, order_index, sources, scene_id, 2)
        if len(first_two) == 2:
            frame_ids = [p3._source_frame(item, scene_id) for item in first_two]
            geometry = _pair_geometry(first_two, sources)
            counts["P_total"] += 1
            counts[f"P_geometry_{'pass' if geometry['gate_pass'] else 'fail'}"] += 1
            if geometry["gate_pass"]:
                candidates.append(
                    {
                        "channel": "P",
                        "track_id": track_id,
                        "evidence_source_ids": first_two,
                        "evidence_frame_ids": frame_ids,
                        "confirmation_frame_id": frame_ids[-1],
                        "geometry": geometry,
                    }
                )
    return {
        "scene_id": scene_id,
        "sources": sources,
        "frames": frames,
        "candidates": candidates,
    }, counts


def _run_clip_for_scene(
    scene_row: MutableMapping[str, Any],
    *,
    ledger: p3._InputLedger,
    model: Any,
    preprocess: Any,
    text_features: Any,
    vocabulary: Sequence[str],
    target_indices: Sequence[int],
    non_target_indices: Sequence[int],
    index_groups: Mapping[int, Sequence[str]],
    device: str,
    batch_size: int,
) -> None:
    tasks: list[tuple[int, str, int, MutableMapping[str, Any], Sequence[float], Path]] = []
    sources: Mapping[str, Mapping[str, Any]] = scene_row["sources"]
    frames: Mapping[int, Mapping[str, Any]] = scene_row["frames"]
    for candidate in scene_row["candidates"]:
        evidence: list[dict[str, Any]] = []
        for evidence_index, source_id in enumerate(candidate["evidence_source_ids"]):
            frame_id = int(candidate["evidence_frame_ids"][evidence_index])
            source = sources[source_id]
            frame = frames[frame_id]
            rgb_receipt = frame.get("input", {}).get("rgb")
            if not isinstance(rgb_receipt, Mapping):
                raise DualPathActiveError(f"sealed F4 RGB absent: {source_id}")
            rgb_path = ledger.add(Path(str(rgb_receipt.get("path", ""))), f"F4 RGB {source_id}", rgb_receipt.get("sha256"))
            bbox = source.get("tight_box_xyxy")
            if not isinstance(bbox, list) or len(bbox) != 4:
                raise DualPathActiveError(f"tight 2D box missing: {source_id}")
            row: dict[str, Any] = {
                "source_id": source_id,
                "frame_id": frame_id,
                "tight_box_xyxy": list(bbox),
            }
            evidence.append(row)
            tasks.append((frame_id, source_id, evidence_index, row, row["tight_box_xyxy"], rgb_path))
        candidate["clip_evidence"] = evidence

    tensors: list[Any] = []
    destinations: list[MutableMapping[str, Any]] = []
    cached_path: Path | None = None
    cached_rgb: np.ndarray | None = None
    for _, source_id, _, destination, bbox, rgb_path in sorted(tasks, key=lambda item: (item[0], item[1], item[2])):
        if rgb_path != cached_path:
            bgr = cv2.imread(os.fspath(rgb_path), cv2.IMREAD_COLOR)
            if bgr is None or bgr.ndim != 3 or bgr.shape[2] != 3:
                raise DualPathActiveError(f"cannot decode F4 RGB: {rgb_path}")
            cached_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            cached_path = rgb_path
        assert cached_rgb is not None
        tensors.append(preprocess(_crop_rgb(cached_rgb, bbox)))
        destinations.append(destination)
        if len(tensors) >= batch_size:
            _flush_batch(
                tensors=tensors,
                destinations=destinations,
                model=model,
                text_features=text_features,
                vocabulary=vocabulary,
                target_indices=target_indices,
                non_target_indices=non_target_indices,
                target_index_groups=index_groups,
                device=device,
            )
    _flush_batch(
        tensors=tensors,
        destinations=destinations,
        model=model,
        text_features=text_features,
        vocabulary=vocabulary,
        target_indices=target_indices,
        non_target_indices=non_target_indices,
        target_index_groups=index_groups,
        device=device,
    )


def _load_cutr_until(
    scene_id: str,
    f0_scene: Mapping[str, Any],
    max_confirmation: int,
    ledger: p3._InputLedger,
) -> dict[int, np.ndarray]:
    if max_confirmation < 0:
        return {}
    f0_frames = f0_scene.get("frames")
    schedule = f0_scene.get("schedule")
    if not isinstance(f0_frames, list) or not isinstance(schedule, Mapping):
        raise DualPathActiveError(f"F0 scene cache contract differs: {scene_id}")
    _, records = p3._cache_manifest_records(f0_frames, scene_id, schedule, ledger)
    cutr_by_frame: dict[int, np.ndarray] = {}
    for ordinal, frame in enumerate(f0_frames):
        if not isinstance(frame, Mapping) or frame.get("frame_ordinal") != ordinal:
            raise DualPathActiveError(f"F0 frame order differs: {scene_id}")
        frame_id = p3._integer(frame.get("frame_id"), "F0 frame_id")
        if frame_id > max_confirmation:
            continue
        record = records.get(frame_id)
        if record is None:
            raise DualPathActiveError(f"CuTR manifest misses F0 frame: {scene_id}/{frame_id}")
        cutr_by_frame[frame_id] = p3._load_cutr_world(frame, record, ledger)
    return cutr_by_frame


def run_active(
    *,
    l2_seal: Path,
    f0_manifest: Path,
    baseline_root: Path,
    scene_list_path: Path,
    output_root: Path,
    clip_checkpoint: Path,
    class_features: Path,
    vocabulary_path: Path,
    device: str,
    batch_size: int,
    expected_scene_count: int,
) -> dict[str, Any]:
    if output_root.exists() or output_root.is_symlink():
        raise DualPathActiveError(f"refusing to overwrite output root: {output_root}")
    if batch_size < 1 or expected_scene_count < 1:
        raise DualPathActiveError("batch size and scene count must be positive")

    ledger = p3._InputLedger()
    l2_path = ledger.add(l2_seal, "L2 seal")
    f0_path = ledger.add(f0_manifest, "F0 merged cache receipt")
    _, l2 = p3._json(l2_path, "L2 seal")
    _, f0 = p3._json(f0_path, "F0 merged cache receipt")
    scenes, l2_rows, f0_by_scene = p3._validate_roots(l2, f0, expected_scene_count)
    official_scenes = _scene_list(scene_list_path, expected_scene_count)
    if tuple(scenes) != official_scenes:
        raise DualPathActiveError("L2 scene order differs from official paper100 list")

    prepared: list[dict[str, Any]] = []
    build_counts: Counter[str] = Counter()
    for scene_index, (scene_id, l2_row) in enumerate(zip(scenes, l2_rows, strict=True)):
        if not isinstance(l2_row, Mapping) or l2_row.get("scene_id") != scene_id or l2_row.get("scene_index") != scene_index:
            raise DualPathActiveError(f"L2 scene order differs: {scene_id}")
        scene_row, counts = _prepare_candidates(scene_id, l2_row, ledger)
        build_counts.update(counts)
        f0_row = f0_by_scene[scene_id]
        sidecar_receipt = f0_row.get("sidecar")
        if not isinstance(sidecar_receipt, Mapping):
            raise DualPathActiveError(f"F0 sidecar receipt absent: {scene_id}")
        f0_scene_path = ledger.add(Path(str(sidecar_receipt.get("path", ""))), f"F0 scene {scene_id}", sidecar_receipt.get("sha256"))
        _, f0_scene = p3._json(f0_scene_path, f"F0 scene {scene_id}")
        if f0_scene.get("schema") != p3.F0_SCENE_SCHEMA or f0_scene.get("complete") is not True:
            raise DualPathActiveError(f"F0 scene contract differs: {scene_id}")
        scene_row["scene_index"] = scene_index
        scene_row["f0_scene"] = f0_scene
        prepared.append(scene_row)

    checkpoint = ledger.add(clip_checkpoint, "CLIP checkpoint")
    features_path = ledger.add(class_features, "CLIP class features")
    vocab_path = ledger.add(vocabulary_path, "native CLIP vocabulary")
    vocabulary = vocab_path.read_text(encoding="utf-8").splitlines()
    if len(vocabulary) != EXPECTED_VOCABULARY_SIZE:
        raise DualPathActiveError("native CLIP vocabulary size differs")
    target_groups, target_indices, non_target_indices = _resolve_target_indices(vocabulary)
    index_groups: dict[int, list[str]] = defaultdict(list)
    for group, indices in target_groups.items():
        for index in indices:
            index_groups[index].append(group)
    model, preprocess = _load_clip_runtime(checkpoint, device)
    text_features = _load_text_features(features_path, device)

    output_parent = output_root.parent
    output_parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{output_root.name}.", dir=output_parent))
    scene_outputs: list[dict[str, Any]] = []
    decision_counts: Counter[str] = Counter()
    terminal_counts: Counter[str] = Counter()
    total_shadow_births = 0
    total_terminal_births = 0
    total_native_boxes = 0
    try:
        for scene_row in prepared:
            scene_id = str(scene_row["scene_id"])
            scene_index = int(scene_row["scene_index"])
            _run_clip_for_scene(
                scene_row,
                ledger=ledger,
                model=model,
                preprocess=preprocess,
                text_features=text_features,
                vocabulary=vocabulary,
                target_indices=target_indices,
                non_target_indices=non_target_indices,
                index_groups=index_groups,
                device=device,
                batch_size=batch_size,
            )

            semantic_ready: list[MutableMapping[str, Any]] = []
            pre_decisions: list[dict[str, Any]] = []
            for candidate in scene_row["candidates"]:
                semantic = _candidate_semantic(candidate["clip_evidence"], str(candidate["channel"]))
                candidate["semantic"] = semantic
                if not semantic["gate_pass"]:
                    decision_counts[f"{candidate['channel']}_semantic_gate"] += 1
                    pre_decisions.append(
                        {
                            "channel": candidate["channel"],
                            "track_id": candidate["track_id"],
                            "confirmation_frame_id": candidate["confirmation_frame_id"],
                            "decision": "semantic_gate",
                            "reasons": semantic["gate_rejection_reasons"],
                        }
                    )
                    continue
                semantic_ready.append(candidate)

            max_confirmation = max((int(row["confirmation_frame_id"]) for row in semantic_ready), default=-1)
            cutr_by_frame = _load_cutr_until(scene_id, scene_row["f0_scene"], max_confirmation, ledger)
            shadow_accepted, causal_decisions = _admit_causal(semantic_ready, cutr_by_frame)
            for row in causal_decisions:
                decision_counts[f"{row['channel']}_{row['decision']}"] += 1

            native_path = ledger.add(baseline_root / f"{scene_id}{PREDICTION_SUFFIX}", f"native prediction {scene_id}")
            native = _load_native_prediction(native_path)
            selected, terminal_decisions = _terminal_select(native.corners, shadow_accepted)
            for row in terminal_decisions:
                terminal_counts[f"{row['channel']}_{row['decision']}"] += 1

            payload = _augment_payload(native, selected)
            out_path = stage / f"{scene_id}{PREDICTION_SUFFIX}"
            _write_pickle(out_path, payload)
            reloaded = _load_native_prediction(out_path)
            _assert_native_prefix(native.rows, reloaded.rows, f"disk output {scene_id}")

            total_native_boxes += len(native.rows)
            total_shadow_births += len(shadow_accepted)
            total_terminal_births += len(selected)
            scene_outputs.append(
                {
                    "scene_id": scene_id,
                    "scene_index": scene_index,
                    "geometry_pass_candidates": len(scene_row["candidates"]),
                    "semantic_ready_candidates": len(semantic_ready),
                    "causal_shadow_births": len(shadow_accepted),
                    "terminal_births": len(selected),
                    "decisions": pre_decisions + causal_decisions,
                    "terminal_decisions": terminal_decisions,
                    "selected_births": selected,
                }
            )
            print(
                f"[{scene_index + 1}/{len(scenes)}] {scene_id}: "
                f"geom={len(scene_row['candidates'])} semantic={len(semantic_ready)} "
                f"shadow={len(shadow_accepted)} terminal={len(selected)}",
                flush=True,
            )

        ledger.verify()
        manifest = {
            "schema": SCHEMA,
            "protocol_id": PROTOCOL_ID,
            "complete": True,
            "scene_count": len(scenes),
            "scene_order": scenes,
            "counts": {
                "candidate_build": dict(sorted(build_counts.items())),
                "decisions": dict(sorted(decision_counts.items())),
                "terminal_decisions": dict(sorted(terminal_counts.items())),
                "native_box_count": total_native_boxes,
                "causal_shadow_birth_count": total_shadow_births,
                "terminal_birth_count": total_terminal_births,
            },
            "contracts": dict(CONTRACTS),
            "policy": {
                "score_thresh": 0.5,
                "final_output_score": APPENDED_SCORE,
                "S_path": S_POLICY,
                "P_path": P_POLICY,
                "semantic": SEMANTIC_POLICY,
                "causal_novelty": NOVELTY_POLICY,
                "terminal_filter": TERMINAL_POLICY,
                "clip_model": "frozen_OpenCLIP_ViT-H-14_native_473_vocab",
            },
            "inputs": {
                "l2_seal": os.fspath(l2_path),
                "l2_seal_sha256": p3._sha256(l2_path),
                "f0_manifest": os.fspath(f0_path),
                "f0_manifest_sha256": p3._sha256(f0_path),
                "baseline_root": os.fspath(baseline_root.resolve()),
                "clip_checkpoint": os.fspath(checkpoint),
                "clip_checkpoint_sha256": p3._sha256(checkpoint),
                "class_features": os.fspath(features_path),
                "class_features_sha256": p3._sha256(features_path),
                "vocabulary": os.fspath(vocab_path),
                "vocabulary_sha256": p3._sha256(vocab_path),
                "consumed_input_ledger": ledger.receipt(),
            },
            "scenes": scene_outputs,
        }
        _write_json(stage / DEFAULT_MANIFEST_NAME, manifest)
        os.rename(stage, output_root)
        print(f"Saved: {output_root}", flush=True)
        return manifest
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--l2-seal", type=Path, default=p3.DEFAULT_L2)
    parser.add_argument("--f0-manifest", type=Path, default=p3.DEFAULT_F0)
    parser.add_argument("--baseline-root", type=Path, default=DEFAULT_BASELINE_ROOT)
    parser.add_argument("--scene-list", type=Path, default=DEFAULT_SCENE_LIST)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--clip-checkpoint", type=Path, default=ROOT / "models/open_clip_pytorch_model.bin")
    parser.add_argument("--class-features", type=Path, default=ROOT / "data/class_features.pt")
    parser.add_argument("--vocabulary", type=Path, default=ROOT / "data/panoptic_categories_nomerge.txt")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--expected-scene-count", type=int, default=100)
    return parser


def main() -> None:
    args = _parser().parse_args()
    run_active(
        l2_seal=args.l2_seal,
        f0_manifest=args.f0_manifest,
        baseline_root=args.baseline_root,
        scene_list_path=args.scene_list,
        output_root=args.output_root,
        clip_checkpoint=args.clip_checkpoint,
        class_features=args.class_features,
        vocabulary_path=args.vocabulary,
        device=args.device,
        batch_size=args.batch_size,
        expected_scene_count=args.expected_scene_count,
    )


if __name__ == "__main__":
    main()
