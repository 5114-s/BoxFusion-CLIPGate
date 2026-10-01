#!/usr/bin/env python3
"""Seal L1: every F3 track with >=2 views and its full F4 view bank."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping


_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if os.fspath(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, os.fspath(_REPOSITORY_ROOT))

from tools.seal_scannet_l0_f3_f4_perview_paper100 import (
    F3_SCHEMA,
    F4_SCHEMA,
    HYPOTHESES,
    L0SealError,
    ROOT,
    SOURCE_RE,
    _json,
    _sha,
    _validate_hypothesis,
    _write,
)


SCHEMA = "boxfusion.scannet_l1_f3_2view_f4_perview_paper100.seal.v1"
PROTOCOL_ID = "L1-F3-MIN2VIEW-F4-PERVIEW-HYPOTHESIS-BANK-PAPER100-V1"
MIN_DISTINCT_VIEWS = 2
DEFAULT_F3_ROOT = ROOT / "logs/scannet_fastsam_f3_openbox_paper100_score05/scenes"
DEFAULT_F4_ROOT = ROOT / "logs/scannet_fastsam_f4_boxer_paper100_score05/scenes"
DEFAULT_OUT = ROOT / "logs/scannet_l1_f3_2view_f4_perview_paper100_score05/final/L1_F3_2VIEW_F4_PERVIEW_PAPER100.json"


class L1SealError(L0SealError):
    pass


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-list", type=Path, default=ROOT / "evaluation/data_util/meta_data/scannetv2_val.txt")
    parser.add_argument("--f3-root", type=Path, default=DEFAULT_F3_ROOT)
    parser.add_argument("--f4-root", type=Path, default=DEFAULT_F4_ROOT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    scenes = [line.strip() for line in args.scene_list.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(scenes) != 100 or len(set(scenes)) != 100:
        raise L1SealError("L1 requires exact unique paper100 scene list")

    scene_rows: list[dict[str, Any]] = []
    total_tracks = total_sources = total_geometries = 0
    confirmed_tracks = unconfirmed_two_view_tracks = 0
    hypothesis_counts = {name: 0 for name in HYPOTHESES}
    for scene_index, scene in enumerate(scenes):
        f3_path = args.f3_root / f"{scene}.json"
        f4_path = args.f4_root / f"{scene}.json"
        f3 = _json(f3_path, f"F3 scene {scene}")
        f4 = _json(f4_path, f"F4 scene {scene}")
        if (
            f3.get("schema") != F3_SCHEMA
            or f3.get("complete") is not True
            or f3.get("contracts", {}).get("ground_truth_access") is not False
            or f3.get("causality", {}).get("query_before_commit", {}).get("passed") is not True
            or f4.get("schema") != F4_SCHEMA
            or f4.get("complete") is not True
            or f4.get("contracts", {}).get("gt_access") is not False
            or f4.get("contracts", {}).get("native_output_mutation") is not False
        ):
            raise L1SealError(f"upstream shadow contract differs: {scene}")
        f4_sources: dict[str, Mapping[str, Any]] = {}
        for frame in f4.get("frames", []):
            if not isinstance(frame, Mapping):
                raise L1SealError(f"invalid F4 frame: {scene}")
            for source in frame.get("sources", []):
                if not isinstance(source, Mapping) or not isinstance(source.get("source_id"), str):
                    raise L1SealError(f"invalid F4 source: {scene}")
                source_id = str(source["source_id"])
                if source_id in f4_sources:
                    raise L1SealError(f"duplicate F4 source: {source_id}")
                f4_sources[source_id] = source

        tracks = []
        used_sources: set[str] = set()
        scene_confirmed = scene_unconfirmed = 0
        for track in f3.get("tracks", []):
            if not isinstance(track, Mapping):
                raise L1SealError(f"invalid F3 track row: {scene}")
            source_ids = track.get("retained_source_ids")
            frame_ids = track.get("retained_frame_ids")
            if not isinstance(source_ids, list) or not isinstance(frame_ids, list):
                raise L1SealError(f"F3 retained ledger differs: {scene}:{track.get('track_id')}")
            if len(source_ids) < MIN_DISTINCT_VIEWS:
                continue
            track_id = track.get("track_id")
            if (
                type(track_id) is not int
                or not MIN_DISTINCT_VIEWS <= len(source_ids) == len(frame_ids) <= 5
                or frame_ids != sorted(set(frame_ids))
                or type(track.get("confirmed")) is not bool
            ):
                raise L1SealError(f"eligible F3 track differs: {scene}:{track_id}")
            sources = []
            for source_id_raw, frame_id in zip(source_ids, frame_ids):
                source_id = str(source_id_raw)
                match = SOURCE_RE.fullmatch(source_id)
                if (
                    match is None
                    or match["scene"] != scene
                    or int(match["frame"]) != frame_id
                    or source_id in used_sources
                    or source_id not in f4_sources
                ):
                    raise L1SealError(f"F3/F4 retained identity differs: {source_id}")
                used_sources.add(source_id)
                source = f4_sources[source_id]
                hypotheses = source.get("hypotheses")
                if not isinstance(hypotheses, Mapping) or set(hypotheses) != set(HYPOTHESES):
                    raise L1SealError(f"F4 hypothesis bank differs: {source_id}")
                available = [
                    name
                    for name in HYPOTHESES
                    if _validate_hypothesis(hypotheses[name], name, source_id)
                ]
                if not available:
                    raise L1SealError(f"eligible source has no valid F4 geometry: {source_id}")
                for name in available:
                    hypothesis_counts[name] += 1
                total_geometries += len(available)
                sources.append(
                    {
                        "source_id": source_id,
                        "frame_id": frame_id,
                        "available_hypotheses": available,
                    }
                )
            is_confirmed = bool(track["confirmed"])
            scene_confirmed += int(is_confirmed)
            scene_unconfirmed += int(not is_confirmed)
            tracks.append(
                {
                    "track_id": track_id,
                    "f3_confirmed": is_confirmed,
                    "sources": sources,
                }
            )
        total_tracks += len(tracks)
        total_sources += len(used_sources)
        confirmed_tracks += scene_confirmed
        unconfirmed_two_view_tracks += scene_unconfirmed
        scene_rows.append(
            {
                "scene_id": scene,
                "scene_index": scene_index,
                "f3": {"path": os.fspath(f3_path.resolve()), "sha256": _sha(f3_path)},
                "f4": {"path": os.fspath(f4_path.resolve()), "sha256": _sha(f4_path)},
                "eligible_track_count": len(tracks),
                "confirmed_track_count": scene_confirmed,
                "unconfirmed_two_view_track_count": scene_unconfirmed,
                "retained_source_count": len(used_sources),
                "tracks": tracks,
            }
        )
    receipt = {
        "schema": SCHEMA,
        "protocol_id": PROTOCOL_ID,
        "complete": True,
        "overall_pass": True,
        "scene_count": 100,
        "scene_order": scenes,
        "counts": {
            "eligible_track_count": total_tracks,
            "confirmed_track_count": confirmed_tracks,
            "unconfirmed_two_view_track_count": unconfirmed_two_view_tracks,
            "retained_source_count": total_sources,
            "valid_geometry_count": total_geometries,
            "per_hypothesis": hypothesis_counts,
        },
        "contracts": {
            "shadow_only": True,
            "birth_enabled": False,
            "native_output_mutation": False,
            "ground_truth_access": False,
            "annotation_access": False,
            "evaluator_access": False,
            "training": False,
            "online_learning": False,
            "minimum_distinct_views": MIN_DISTINCT_VIEWS,
            "one_track_one_identity": True,
            "per_view_geometry_preserved": True,
            "sam2_access": False,
            "tsdf_access": False,
            "mv3dis_access": False,
        },
        "scenes": scene_rows,
        "conclusion_guardrail": "L1 is a no-GT seal with no AP and cannot authorize birth.",
    }
    _write(args.out, receipt)
    print(json.dumps({"out": os.fspath(args.out), "counts": receipt["counts"]}, sort_keys=True))


if __name__ == "__main__":
    main()
