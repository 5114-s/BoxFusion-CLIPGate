#!/usr/bin/env python3
"""Seal L3B: fixed GT-free HB medoid geometry for every T1 track.

Every retained observation contributes exactly its frozen Boxer HB geometry.
Singleton tracks keep that HB.  Multi-view tracks choose the HB medoid with
the strongest fixed cross-view agreement.  Selection is evaluated only at
track retirement/end-of-stream and therefore consumes current/past evidence.
This program is shadow-only and never opens GT, predictions or an evaluator.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
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
    _json,
    _sha,
    _source_map,
)
from tools.run_scannet_l3_hbc_t1_selector_paper100 import (  # noqa: E402
    HISTORY_CONTAINMENT_MIN,
    HISTORY_IOU_MIN,
    HISTORY_ND_MAX,
    _aabb,
    _hb_confidence,
    _metrics,
)
from tools.seal_scannet_l0_f3_f4_perview_paper100 import (  # noqa: E402
    SOURCE_RE,
    _write,
)
from tools.seal_scannet_l2_source_preserving_paper100 import (  # noqa: E402
    PROTOCOL_ID as L2_PROTOCOL_ID,
    SCHEMA as L2_SCHEMA,
)


SCHEMA = "boxfusion.scannet_l3b_hbmedoid_t1_selector_paper100.shadow.v1"
PROTOCOL_ID = "L3B-HBMEDOID-T1-GTFREE-TRACK-GEOMETRY-SELECTOR-PAPER100-V1"
DEFAULT_L2 = ROOT / "logs/scannet_l2_source_preserving_paper100_score05/final/L2_SOURCE_PRESERVING_PAPER100.json"
DEFAULT_OUT = ROOT / "logs/scannet_l3b_hbmedoid_t1_selector_paper100_score05/final/L3B_HBMEDOID_T1_SELECTOR_PAPER100.json"


class L3BSelectorError(RuntimeError):
    pass


def _choose_hbmedoid(
    source_ids: list[str], source_map: Mapping[str, Mapping[str, Any]]
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for ordinal, source_id in enumerate(source_ids):
        source = source_map[source_id]
        hypotheses = source.get("hypotheses")
        if not isinstance(hypotheses, Mapping) or not isinstance(hypotheses.get("HB"), Mapping):
            raise L3BSelectorError(f"HB geometry is absent: {source_id}")
        rows.append(
            {
                "source_id": source_id,
                "source_ordinal": ordinal,
                "box": _aabb(hypotheses["HB"], "HB", f"{source_id}.HB"),
                "hb_confidence": _hb_confidence(source, source_id),
            }
        )
    if len(rows) == 1:
        winner = rows[0]
        return {
            "source_id": winner["source_id"],
            "hypothesis": "HB",
            "reason": "singleton_hb",
            "supporting_other_view_count": 0,
            "other_view_count": 0,
            "median_hb_iou": 0.0,
            "median_hb_containment": 0.0,
            "median_hb_normalized_center_distance": 0.0,
            "hb_confidence": winner["hb_confidence"],
        }

    candidates: list[tuple[tuple[float, ...], dict[str, Any]]] = []
    for row in rows:
        comparisons = [
            _metrics(row["box"], other["box"])
            for other in rows
            if other["source_id"] != row["source_id"]
        ]
        support = int(
            sum(
                iou >= HISTORY_IOU_MIN
                or (containment >= HISTORY_CONTAINMENT_MIN and nd <= HISTORY_ND_MAX)
                for iou, containment, nd in comparisons
            )
        )
        median_iou = float(np.median([item[0] for item in comparisons]))
        median_containment = float(np.median([item[1] for item in comparisons]))
        median_nd = float(np.median([item[2] for item in comparisons]))
        rank = (
            float(support),
            median_iou,
            median_containment,
            -median_nd,
            float(row["hb_confidence"]),
            -float(row["source_ordinal"]),
        )
        candidates.append(
            (
                rank,
                {
                    "source_id": row["source_id"],
                    "hypothesis": "HB",
                    "reason": "hb_cross_view_medoid",
                    "supporting_other_view_count": support,
                    "other_view_count": len(comparisons),
                    "median_hb_iou": median_iou,
                    "median_hb_containment": median_containment,
                    "median_hb_normalized_center_distance": median_nd,
                    "hb_confidence": row["hb_confidence"],
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
    scene_rows = l2.get("scenes")
    if (
        l2.get("schema") != L2_SCHEMA
        or l2.get("protocol_id") != L2_PROTOCOL_ID
        or l2.get("complete") is not True
        or l2.get("overall_pass") is not True
        or l2.get("contracts", {}).get("ground_truth_access") is not False
        or not isinstance(scene_rows, list)
        or len(scene_rows) != 100
    ):
        raise L3BSelectorError("L2 no-GT seal contract differs")

    sealed_scenes: list[dict[str, Any]] = []
    track_lengths: Counter[str] = Counter()
    total_tracks = total_sources = 0
    for scene_index, scene_row in enumerate(scene_rows):
        if not isinstance(scene_row, Mapping) or scene_row.get("scene_index") != scene_index:
            raise L3BSelectorError("L2 scene order differs")
        scene = str(scene_row.get("scene_id", ""))
        f4_receipt = scene_row.get("f4")
        tracks = scene_row.get("tracks")
        source_order = scene_row.get("f4_source_order")
        if not isinstance(f4_receipt, Mapping) or not isinstance(tracks, list) or not isinstance(source_order, list):
            raise L3BSelectorError(f"L2 scene ledger differs: {scene}")
        f4_path = Path(str(f4_receipt.get("path", "")))
        if _sha(f4_path) != f4_receipt.get("sha256"):
            raise L3BSelectorError(f"sealed F4 hash differs: {scene}")
        source_map_raw = _source_map(_json(f4_path, f"F4 scene {scene}"), scene)
        source_map: dict[str, Mapping[str, Any]] = dict(source_map_raw)
        if [str(item) for item in source_order] != list(source_map):
            raise L3BSelectorError(f"F4 source order differs: {scene}")

        selections: list[dict[str, Any]] = []
        for track_id, track in enumerate(tracks):
            if not isinstance(track, Mapping) or track.get("track_id") != track_id:
                raise L3BSelectorError(f"track order differs: {scene}")
            retained = track.get("retained_source_ids")
            all_sources = track.get("source_ids")
            if not isinstance(retained, list) or not retained or not isinstance(all_sources, list) or not all_sources:
                raise L3BSelectorError(f"track source ledger differs: {scene}:{track_id}")
            retained_ids = [str(item) for item in retained]
            all_ids = [str(item) for item in all_sources]
            if not set(retained_ids).issubset(all_ids) or any(item not in source_map for item in all_ids):
                raise L3BSelectorError(f"track/source identity differs: {scene}:{track_id}")
            all_frame_ids: list[int] = []
            for source_id in all_ids:
                match = SOURCE_RE.fullmatch(source_id)
                if match is None or match["scene"] != scene:
                    raise L3BSelectorError(f"source identity differs: {source_id}")
                all_frame_ids.append(int(match["frame"]))
            choice = _choose_hbmedoid(retained_ids, source_map)
            chosen_match = SOURCE_RE.fullmatch(str(choice["source_id"]))
            if chosen_match is None:
                raise L3BSelectorError("chosen source identity differs")
            decision_frame = max(all_frame_ids)
            choice.update(
                {
                    "track_id": track_id,
                    "observation_count": len(all_ids),
                    "retained_observation_count": len(retained_ids),
                    "decision_frame_id": decision_frame,
                    "chosen_source_frame_id": int(chosen_match["frame"]),
                    "past_only_at_decision": int(chosen_match["frame"]) <= decision_frame,
                    "emit_event": "track_retirement_or_end_of_stream",
                }
            )
            if choice["past_only_at_decision"] is not True:
                raise L3BSelectorError("future source selected")
            selections.append(choice)
            track_lengths["1" if len(retained_ids) == 1 else "2" if len(retained_ids) == 2 else "3plus"] += 1
        if len(selections) != scene_row.get("mode_identity_counts", {}).get("T1"):
            raise L3BSelectorError(f"T1 census differs: {scene}")
        total_tracks += len(selections)
        total_sources += len(source_map)
        sealed_scenes.append(
            {
                "scene_id": scene,
                "scene_index": scene_index,
                "f4": {"path": os.fspath(f4_path.resolve()), "sha256": _sha(f4_path)},
                "track_count": len(selections),
                "selections": selections,
            }
        )

    counts = l2.get("counts", {})
    if total_tracks != counts.get("mode_identity_counts", {}).get("T1") or total_sources != counts.get("raw_source_count"):
        raise L3BSelectorError("L3B census differs from L2")
    receipt = {
        "schema": SCHEMA,
        "protocol_id": PROTOCOL_ID,
        "complete": True,
        "overall_pass": True,
        "l2_seal": {"path": os.fspath(args.l2_seal.resolve()), "sha256": _sha(args.l2_seal)},
        "policy": {
            "geometry": "frozen_Boxer_HB_only",
            "singleton": "the_only_retained_HB",
            "multiview": "HB_cross_view_medoid",
            "history_iou_min": HISTORY_IOU_MIN,
            "history_containment_min": HISTORY_CONTAINMENT_MIN,
            "history_normalized_center_distance_max": HISTORY_ND_MAX,
            "tie_break": ["support", "median_iou", "median_containment", "negative_median_nd", "hb_confidence", "earliest_retained_source"],
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
            "selected_hypothesis_counts": {"HB": total_tracks},
            "retained_track_length_counts": dict(sorted(track_lengths.items())),
        },
        "runtime": {"wall_seconds": time.perf_counter() - started},
        "scenes": sealed_scenes,
        "conclusion_guardrail": "No GT and no AP; this shadow cannot authorize active birth.",
    }
    _write(args.out, receipt)
    print(json.dumps({"out": os.fspath(args.out), "counts": receipt["counts"], "runtime": receipt["runtime"]}, sort_keys=True))


if __name__ == "__main__":
    main()
