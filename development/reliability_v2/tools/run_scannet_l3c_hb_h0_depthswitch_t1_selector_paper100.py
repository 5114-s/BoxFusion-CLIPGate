#!/usr/bin/env python3
"""Seal L3C: L3B HB medoid with conservative F5 depth-to-H0 switches.

L3C keeps the fixed L3B HB choice unless the already-sealed, GT-free F5
current-view diagnostics contain a physical contradiction: insufficient RGB-D
point support, invalid projection, poor mask projection IoU, or strong
center/volume/overlap disagreement with the point-derived base box.  Low
Boxer confidence, insufficient history, and past-consistency failures never
trigger a switch.  A rejected HB switches to H0 from the same source.

This is a shadow-only selector.  It never reads GT, native predictions,
semantics or an evaluator and never creates a birth.
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


ROOT = Path(__file__).resolve().parents[1]
if os.fspath(ROOT) not in sys.path:
    sys.path.insert(0, os.fspath(ROOT))

from tools.audit_scannet_l0_f3_f4_perview_paper100_oracle import (  # noqa: E402
    _json,
    _sha,
    _source_map,
)
from tools.run_scannet_l3b_hbmedoid_t1_selector_paper100 import (  # noqa: E402
    PROTOCOL_ID as L3B_PROTOCOL_ID,
    SCHEMA as L3B_SCHEMA,
)
from tools.seal_scannet_l0_f3_f4_perview_paper100 import _write  # noqa: E402


SCHEMA = "boxfusion.scannet_l3c_hb_h0_depthswitch_t1_selector_paper100.shadow.v1"
PROTOCOL_ID = "L3C-HB-H0-DEPTHSWITCH-T1-GTFREE-PAPER100-V1"
F5_SCENE_SCHEMA = "boxfusion.scannet_fastsam_f5_gtfree_selector_paper100.scene.v1"
DEFAULT_L3B = ROOT / "logs/scannet_l3b_hbmedoid_t1_selector_paper100_score05/final/L3B_HBMEDOID_T1_SELECTOR_PAPER100.json"
DEFAULT_F5_ROOT = ROOT / "logs/scannet_fastsam_f5_selector_paper100_score05/scenes"
DEFAULT_OUT = ROOT / "logs/scannet_l3c_hb_h0_depthswitch_t1_selector_paper100_score05/final/L3C_HB_H0_DEPTHSWITCH_T1_SELECTOR_PAPER100.json"

# Frozen F5 current-view physical rejections.  Confidence/history are excluded.
SWITCH_REASONS = frozenset(
    {
        "validity",
        "confidence_domain",
        "point_count",
        "exact_depth_support",
        "expanded_depth_support",
        "projection_depth",
        "projection_iou",
        "center",
        "volume",
        "base_overlap",
    }
)
KEEP_HB_REASONS = frozenset(
    {None, "selected_hb", "confidence_threshold", "history_count", "past_consistency"}
)


class L3CSelectorError(RuntimeError):
    pass


def _f5_source_map(value: Mapping[str, Any], scene: str) -> tuple[list[str], dict[str, Mapping[str, Any]]]:
    if (
        value.get("schema") != F5_SCENE_SCHEMA
        or value.get("complete") is not True
        or value.get("contracts", {}).get("ground_truth_access") is not False
        or value.get("contracts", {}).get("future_frame_access") is not False
        or value.get("contracts", {}).get("training") is not False
        or value.get("contracts", {}).get("birth_enabled") is not False
    ):
        raise L3CSelectorError(f"F5 no-GT shadow contract differs: {scene}")
    order: list[str] = []
    result: dict[str, Mapping[str, Any]] = {}
    for frame in value.get("frames", []):
        if not isinstance(frame, Mapping):
            raise L3CSelectorError(f"invalid F5 frame: {scene}")
        for source in frame.get("sources", []):
            if not isinstance(source, Mapping) or not isinstance(source.get("source_id"), str):
                raise L3CSelectorError(f"invalid F5 source: {scene}")
            source_id = str(source["source_id"])
            if source_id in result:
                raise L3CSelectorError(f"duplicate F5 source: {source_id}")
            order.append(source_id)
            result[source_id] = source
    if len(result) != value.get("counts", {}).get("source_count"):
        raise L3CSelectorError(f"F5 source census differs: {scene}")
    return order, result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--l3b-shadow", type=Path, default=DEFAULT_L3B)
    parser.add_argument("--f5-root", type=Path, default=DEFAULT_F5_ROOT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    started = time.perf_counter()
    l3b = _json(args.l3b_shadow, "L3B shadow")
    scene_rows = l3b.get("scenes")
    if (
        l3b.get("schema") != L3B_SCHEMA
        or l3b.get("protocol_id") != L3B_PROTOCOL_ID
        or l3b.get("complete") is not True
        or l3b.get("overall_pass") is not True
        or l3b.get("contracts", {}).get("ground_truth_access") is not False
        or not isinstance(scene_rows, list)
        or len(scene_rows) != 100
    ):
        raise L3CSelectorError("L3B no-GT shadow contract differs")

    output_scenes: list[dict[str, Any]] = []
    switch_reasons: Counter[str] = Counter()
    selected_hypotheses: Counter[str] = Counter()
    total_tracks = total_sources = 0
    for scene_index, scene_row in enumerate(scene_rows):
        if not isinstance(scene_row, Mapping) or scene_row.get("scene_index") != scene_index:
            raise L3CSelectorError("L3B scene order differs")
        scene = str(scene_row.get("scene_id", ""))
        selections = scene_row.get("selections")
        f4_receipt = scene_row.get("f4")
        if not isinstance(selections, list) or not isinstance(f4_receipt, Mapping):
            raise L3CSelectorError(f"L3B scene ledger differs: {scene}")
        f4_path = Path(str(f4_receipt.get("path", "")))
        if _sha(f4_path) != f4_receipt.get("sha256"):
            raise L3CSelectorError(f"sealed F4 hash differs: {scene}")
        f4_sources = _source_map(_json(f4_path, f"F4 scene {scene}"), scene)
        f5_path = args.f5_root / f"{scene}.json"
        f5_value = _json(f5_path, f"F5 scene {scene}")
        f5_order, f5_sources = _f5_source_map(f5_value, scene)
        if f5_order != list(f4_sources):
            raise L3CSelectorError(f"F4/F5 source order differs: {scene}")

        output_selections: list[dict[str, Any]] = []
        for track_id, selection in enumerate(selections):
            if (
                not isinstance(selection, Mapping)
                or selection.get("track_id") != track_id
                or selection.get("hypothesis") != "HB"
                or selection.get("past_only_at_decision") is not True
            ):
                raise L3CSelectorError(f"L3B selection differs: {scene}:{track_id}")
            source_id = str(selection.get("source_id", ""))
            f5_source = f5_sources.get(source_id)
            f4_source = f4_sources.get(source_id)
            if not isinstance(f5_source, Mapping) or not isinstance(f4_source, Mapping):
                raise L3CSelectorError(f"chosen source absent: {source_id}")
            reason = f5_source.get("hb_abstention_reason")
            if reason not in SWITCH_REASONS and reason not in KEEP_HB_REASONS:
                raise L3CSelectorError(f"unexpected F5 HB reason: {source_id}:{reason}")
            switch = reason in SWITCH_REASONS
            chosen = "H0" if switch else "HB"
            hypotheses = f4_source.get("hypotheses")
            if not isinstance(hypotheses, Mapping) or hypotheses.get(chosen, {}).get("valid") is not True:
                raise L3CSelectorError(f"chosen L3C geometry invalid: {source_id}.{chosen}")
            output = dict(selection)
            output.update(
                {
                    "hypothesis": chosen,
                    "reason": "f5_physical_rejection_switch_to_h0" if switch else "keep_l3b_hb",
                    "l3b_hypothesis": "HB",
                    "f5_hb_abstention_reason": reason,
                    "f5_hb_diagnostics": f5_source.get("hb_diagnostics"),
                    "geometry_selection_uses_gt": False,
                }
            )
            output_selections.append(output)
            selected_hypotheses[chosen] += 1
            if switch:
                switch_reasons[str(reason)] += 1
        if len(output_selections) != scene_row.get("track_count"):
            raise L3CSelectorError(f"L3C track census differs: {scene}")
        total_tracks += len(output_selections)
        total_sources += len(f4_sources)
        output_scenes.append(
            {
                "scene_id": scene,
                "scene_index": scene_index,
                "f4": {"path": os.fspath(f4_path.resolve()), "sha256": _sha(f4_path)},
                "f5": {"path": os.fspath(f5_path.resolve()), "sha256": _sha(f5_path)},
                "track_count": len(output_selections),
                "selections": output_selections,
            }
        )

    expected = l3b.get("counts", {})
    if total_tracks != expected.get("track_count") or total_sources != expected.get("raw_source_count"):
        raise L3CSelectorError("L3C global census differs")
    receipt = {
        "schema": SCHEMA,
        "protocol_id": PROTOCOL_ID,
        "complete": True,
        "overall_pass": True,
        "l3b_shadow": {"path": os.fspath(args.l3b_shadow.resolve()), "sha256": _sha(args.l3b_shadow)},
        "policy": {
            "default_geometry": "L3B_HBMedoid",
            "fallback_geometry": "H0_from_same_source",
            "switch_reasons": sorted(SWITCH_REASONS),
            "explicit_non_switch_reasons": ["selected_hb", "confidence_threshold", "history_count", "past_consistency", None],
            "diagnostic_source": "sealed_F5_current_view_RGBD_projection_and_base_consistency",
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
            "selected_hypothesis_counts": dict(sorted(selected_hypotheses.items())),
            "switch_count": int(sum(switch_reasons.values())),
            "switch_reason_counts": dict(sorted(switch_reasons.items())),
        },
        "runtime": {"wall_seconds": time.perf_counter() - started},
        "scenes": output_scenes,
        "conclusion_guardrail": "No GT and no AP; this shadow cannot authorize active birth.",
    }
    _write(args.out, receipt)
    print(json.dumps({"out": os.fspath(args.out), "counts": receipt["counts"], "runtime": receipt["runtime"]}, sort_keys=True))


if __name__ == "__main__":
    main()
