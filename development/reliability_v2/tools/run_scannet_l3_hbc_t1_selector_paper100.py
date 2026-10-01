#!/usr/bin/env python3
"""Seal L3-HBC: a fixed GT-free track-level geometry selector for T1.

HBC means HB-prior with cross-view geometric consensus.  A one-view track
selects its frozen Boxer HB geometry.  A multi-view track ranks every retained
H0/HL/HLG/HB geometry by agreement with the other retained views, using the
same fixed history thresholds as F5.  The terminal choice is emitted only on
track retirement/end-of-stream, so every consulted observation is current or
past at decision time.

This runner is shadow-only.  It never opens GT, native predictions, semantics
or an evaluator and never creates a birth prediction.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any, Mapping

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if os.fspath(ROOT) not in sys.path:
    sys.path.insert(0, os.fspath(ROOT))

from tools.audit_scannet_l0_f3_f4_perview_paper100_oracle import (  # noqa: E402
    HYPOTHESES,
    _array,
    _json,
    _sha,
    _source_map,
)
from tools.seal_scannet_l0_f3_f4_perview_paper100 import (  # noqa: E402
    SOURCE_RE,
    _write,
)
from tools.seal_scannet_l2_source_preserving_paper100 import (  # noqa: E402
    PROTOCOL_ID as L2_PROTOCOL_ID,
    SCHEMA as L2_SCHEMA,
)


SCHEMA = "boxfusion.scannet_l3_hbc_t1_selector_paper100.shadow.v1"
PROTOCOL_ID = "L3-HBC-T1-GTFREE-TRACK-GEOMETRY-SELECTOR-PAPER100-V1"
DEFAULT_L2 = ROOT / "logs/scannet_l2_source_preserving_paper100_score05/final/L2_SOURCE_PRESERVING_PAPER100.json"
DEFAULT_OUT = ROOT / "logs/scannet_l3_hbc_t1_selector_paper100_score05/final/L3_HBC_T1_SELECTOR_PAPER100.json"

# Frozen before annotation/evaluator access.  These are inherited from F5.
HISTORY_IOU_MIN = 0.15
HISTORY_CONTAINMENT_MIN = 0.60
HISTORY_ND_MAX = 0.50
HYPOTHESIS_PRIORITY = {"H0": 0, "HL": 1, "HLG": 2, "HB": 3}


class L3SelectorError(RuntimeError):
    pass


def _aabb(row: Mapping[str, Any], name: str, label: str) -> np.ndarray:
    if row.get("valid") is not True:
        raise L3SelectorError(f"invalid sealed geometry: {label}")
    if name == "HB":
        corners = _array(row.get("world_corners"), (8, 3), f"{label}.world_corners")
        result = np.concatenate((corners.min(axis=0), corners.max(axis=0)))
    else:
        lower = _array(row.get("q02"), (3,), f"{label}.q02")
        upper = _array(row.get("q98"), (3,), f"{label}.q98")
        result = np.concatenate((lower, upper))
    if np.any(result[3:] <= result[:3]):
        raise L3SelectorError(f"degenerate sealed geometry: {label}")
    return result


def _metrics(left: np.ndarray, right: np.ndarray) -> tuple[float, float, float]:
    lower = np.maximum(left[:3], right[:3])
    upper = np.minimum(left[3:], right[3:])
    intersection = float(np.prod(np.maximum(upper - lower, 0.0)))
    left_volume = float(np.prod(left[3:] - left[:3]))
    right_volume = float(np.prod(right[3:] - right[:3]))
    union = left_volume + right_volume - intersection
    iou = 0.0 if union <= 0.0 else intersection / union
    containment = intersection / max(min(left_volume, right_volume), np.finfo(np.float64).eps)
    center_distance = float(np.linalg.norm((left[:3] + left[3:] - right[:3] - right[3:]) * 0.5))
    scale = max(
        float(np.linalg.norm(left[3:] - left[:3])),
        float(np.linalg.norm(right[3:] - right[:3])),
        0.02,
    )
    return iou, containment, center_distance / scale


def _hb_confidence(source: Mapping[str, Any], source_id: str) -> float:
    hb = source.get("hypotheses", {}).get("HB")
    value = hb.get("confidence") if isinstance(hb, Mapping) else None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise L3SelectorError(f"HB confidence missing: {source_id}")
    result = float(value)
    if not math.isfinite(result):
        raise L3SelectorError(f"HB confidence nonfinite: {source_id}")
    return result


def _choose(
    source_ids: list[str], source_map: Mapping[str, Mapping[str, Any]]
) -> dict[str, Any]:
    banks: list[dict[str, Any]] = []
    for source_ordinal, source_id in enumerate(source_ids):
        source = source_map[source_id]
        hypotheses = source.get("hypotheses")
        if not isinstance(hypotheses, Mapping) or set(hypotheses) != set(HYPOTHESES):
            raise L3SelectorError(f"hypothesis bank differs: {source_id}")
        confidence = _hb_confidence(source, source_id)
        bank = {
            "source_id": source_id,
            "source_ordinal": source_ordinal,
            "hb_confidence": confidence,
            "boxes": {
                name: _aabb(hypotheses[name], name, f"{source_id}.{name}")
                for name in HYPOTHESES
            },
        }
        banks.append(bank)

    if len(banks) == 1:
        bank = banks[0]
        return {
            "source_id": bank["source_id"],
            "hypothesis": "HB",
            "reason": "singleton_hb_prior",
            "supporting_other_view_count": 0,
            "other_view_count": 0,
            "median_best_iou": 0.0,
            "median_best_containment": 0.0,
            "median_best_normalized_center_distance": 0.0,
            "hb_confidence": bank["hb_confidence"],
        }

    candidates: list[tuple[tuple[float, ...], dict[str, Any]]] = []
    for bank in banks:
        for name in HYPOTHESES:
            box = bank["boxes"][name]
            best_by_view: list[tuple[int, float, float, float]] = []
            for other in banks:
                if other["source_id"] == bank["source_id"]:
                    continue
                alternatives: list[tuple[tuple[float, ...], tuple[int, float, float, float]]] = []
                for other_name in HYPOTHESES:
                    iou, containment, nd = _metrics(box, other["boxes"][other_name])
                    supported = int(
                        iou >= HISTORY_IOU_MIN
                        or (containment >= HISTORY_CONTAINMENT_MIN and nd <= HISTORY_ND_MAX)
                    )
                    value = (supported, iou, containment, nd)
                    key = (supported, iou, containment, -nd, HYPOTHESIS_PRIORITY[other_name])
                    alternatives.append((key, value))
                alternatives.sort(key=lambda item: item[0], reverse=True)
                best_by_view.append(alternatives[0][1])
            support = int(sum(item[0] for item in best_by_view))
            median_iou = float(np.median([item[1] for item in best_by_view]))
            median_containment = float(np.median([item[2] for item in best_by_view]))
            median_nd = float(np.median([item[3] for item in best_by_view]))
            rank = (
                float(support),
                median_iou,
                median_containment,
                -median_nd,
                float(HYPOTHESIS_PRIORITY[name]),
                float(bank["hb_confidence"]),
                -float(bank["source_ordinal"]),
            )
            candidates.append(
                (
                    rank,
                    {
                        "source_id": bank["source_id"],
                        "hypothesis": name,
                        "reason": "cross_view_consensus",
                        "supporting_other_view_count": support,
                        "other_view_count": len(best_by_view),
                        "median_best_iou": median_iou,
                        "median_best_containment": median_containment,
                        "median_best_normalized_center_distance": median_nd,
                        "hb_confidence": bank["hb_confidence"],
                    },
                )
            )
    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates[0][1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--l2-seal", type=Path, default=DEFAULT_L2)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    started = time.perf_counter()
    l2 = _json(args.l2_seal, "L2 seal")
    rows = l2.get("scenes")
    if (
        l2.get("schema") != L2_SCHEMA
        or l2.get("protocol_id") != L2_PROTOCOL_ID
        or l2.get("complete") is not True
        or l2.get("overall_pass") is not True
        or l2.get("contracts", {}).get("ground_truth_access") is not False
        or not isinstance(rows, list)
        or len(rows) != 100
    ):
        raise L3SelectorError("L2 no-GT seal contract differs")

    scene_rows: list[dict[str, Any]] = []
    hypothesis_counts: Counter[str] = Counter()
    track_length_counts: Counter[str] = Counter()
    total_tracks = total_sources = 0
    for expected_scene_index, row in enumerate(rows):
        if not isinstance(row, Mapping) or row.get("scene_index") != expected_scene_index:
            raise L3SelectorError("L2 scene order differs")
        scene = str(row.get("scene_id", ""))
        f4_receipt = row.get("f4")
        tracks = row.get("tracks")
        source_order = row.get("f4_source_order")
        if not isinstance(f4_receipt, Mapping) or not isinstance(tracks, list) or not isinstance(source_order, list):
            raise L3SelectorError(f"L2 scene ledger differs: {scene}")
        f4_path = Path(str(f4_receipt.get("path", "")))
        if _sha(f4_path) != f4_receipt.get("sha256"):
            raise L3SelectorError(f"F4 hash differs: {scene}")
        source_map_raw = _source_map(_json(f4_path, f"F4 scene {scene}"), scene)
        source_map: dict[str, Mapping[str, Any]] = dict(source_map_raw)
        if [str(item) for item in source_order] != list(source_map):
            raise L3SelectorError(f"F4 source order differs: {scene}")

        selections: list[dict[str, Any]] = []
        for expected_track_id, track in enumerate(tracks):
            if not isinstance(track, Mapping) or track.get("track_id") != expected_track_id:
                raise L3SelectorError(f"track order differs: {scene}")
            retained = track.get("retained_source_ids")
            all_sources = track.get("source_ids")
            if not isinstance(retained, list) or not retained or not isinstance(all_sources, list) or not all_sources:
                raise L3SelectorError(f"track source ledger differs: {scene}:{expected_track_id}")
            retained_ids = [str(item) for item in retained]
            all_ids = [str(item) for item in all_sources]
            if not set(retained_ids).issubset(all_ids) or any(item not in source_map for item in all_ids):
                raise L3SelectorError(f"track/source identity differs: {scene}:{expected_track_id}")
            frame_ids = []
            for source_id in all_ids:
                match = SOURCE_RE.fullmatch(source_id)
                if match is None or match["scene"] != scene:
                    raise L3SelectorError(f"source ID differs: {source_id}")
                frame_ids.append(int(match["frame"]))
            choice = _choose(retained_ids, source_map)
            chosen_match = SOURCE_RE.fullmatch(str(choice["source_id"]))
            if chosen_match is None:
                raise L3SelectorError("chosen source ID differs")
            choice.update(
                {
                    "track_id": expected_track_id,
                    "observation_count": len(all_ids),
                    "retained_observation_count": len(retained_ids),
                    "decision_frame_id": max(frame_ids),
                    "chosen_source_frame_id": int(chosen_match["frame"]),
                    "past_only_at_decision": int(chosen_match["frame"]) <= max(frame_ids),
                    "emit_event": "track_retirement_or_end_of_stream",
                }
            )
            if choice["past_only_at_decision"] is not True:
                raise L3SelectorError("selector used a future source")
            selections.append(choice)
            hypothesis_counts[str(choice["hypothesis"])] += 1
            length_key = "1" if len(retained_ids) == 1 else "2" if len(retained_ids) == 2 else "3plus"
            track_length_counts[length_key] += 1
        if len(selections) != row.get("mode_identity_counts", {}).get("T1"):
            raise L3SelectorError(f"T1 track census differs: {scene}")
        total_tracks += len(selections)
        total_sources += len(source_map)
        scene_rows.append(
            {
                "scene_id": scene,
                "scene_index": expected_scene_index,
                "f4": {"path": os.fspath(f4_path.resolve()), "sha256": _sha(f4_path)},
                "track_count": len(selections),
                "selections": selections,
            }
        )

    expected_counts = l2.get("counts", {})
    if total_tracks != expected_counts.get("mode_identity_counts", {}).get("T1") or total_sources != expected_counts.get("raw_source_count"):
        raise L3SelectorError("L3 census differs from L2")
    receipt = {
        "schema": SCHEMA,
        "protocol_id": PROTOCOL_ID,
        "complete": True,
        "overall_pass": True,
        "l2_seal": {"path": os.fspath(args.l2_seal.resolve()), "sha256": _sha(args.l2_seal)},
        "policy": {
            "singleton": "valid_HB",
            "multiview": "cross_view_consensus_then_HB_HLG_HL_H0_priority",
            "history_iou_min": HISTORY_IOU_MIN,
            "history_containment_min": HISTORY_CONTAINMENT_MIN,
            "history_normalized_center_distance_max": HISTORY_ND_MAX,
            "hypothesis_priority": ["HB", "HLG", "HL", "H0"],
            "max_retained_observations_per_track": 5,
            "emit_event": "track_retirement_or_end_of_stream",
        },
        "contracts": {
            "shadow_only": True,
            "selector_only": True,
            "birth_enabled": False,
            "native_output_mutation": False,
            "ground_truth_access": False,
            "annotation_access": False,
            "evaluator_access": False,
            "native_prediction_access": False,
            "semantic_or_clip_access": False,
            "future_frame_access": False,
            "training": False,
            "online_learning": False,
            "past_only": True,
            "bounded_track_memory": True,
        },
        "counts": {
            "scene_count": 100,
            "track_count": total_tracks,
            "raw_source_count": total_sources,
            "selected_hypothesis_counts": dict(sorted(hypothesis_counts.items())),
            "retained_track_length_counts": dict(sorted(track_length_counts.items())),
        },
        "runtime": {"wall_seconds": time.perf_counter() - started},
        "scenes": scene_rows,
        "conclusion_guardrail": "No GT and no AP; this shadow cannot authorize active birth.",
    }
    _write(args.out, receipt)
    print(json.dumps({"out": os.fspath(args.out), "counts": receipt["counts"], "runtime": receipt["runtime"]}, sort_keys=True))


if __name__ == "__main__":
    main()
