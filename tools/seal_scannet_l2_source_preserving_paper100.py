#!/usr/bin/env python3
"""Seal nested L2 track/source identity universes without annotations."""

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
    ROOT,
    SOURCE_RE,
    _json,
    _sha,
    _validate_hypothesis,
    _write,
)


SCHEMA = "boxfusion.scannet_l2_source_preserving_paper100.seal.v1"
PROTOCOL_ID = "L2-F3-F4-TRACK-SOURCE-IDENTITY-STRATIFICATION-PAPER100-V1"
MODES = ("T2", "S2", "T1", "S1R", "SRAW")
DEFAULT_F3_ROOT = ROOT / "logs/scannet_fastsam_f3_openbox_paper100_score05/scenes"
DEFAULT_F4_ROOT = ROOT / "logs/scannet_fastsam_f4_boxer_paper100_score05/scenes"
DEFAULT_OUT = ROOT / "logs/scannet_l2_source_preserving_paper100_score05/final/L2_SOURCE_PRESERVING_PAPER100.json"


class L2SealError(RuntimeError):
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
        raise L2SealError("L2 requires exact unique paper100 scene list")

    scene_rows: list[dict[str, Any]] = []
    total_tracks = total_retained = total_raw_sources = total_geometries = 0
    totals_by_mode = {name: 0 for name in MODES}
    per_hypothesis = {name: 0 for name in HYPOTHESES}
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
            or f3.get("causality", {}).get("one_source_one_track", {}).get("passed") is not True
            or f4.get("schema") != F4_SCHEMA
            or f4.get("complete") is not True
            or f4.get("contracts", {}).get("gt_access") is not False
            or f4.get("contracts", {}).get("native_output_mutation") is not False
        ):
            raise L2SealError(f"upstream shadow contract differs: {scene}")

        f4_sources: dict[str, Mapping[str, Any]] = {}
        f4_source_order: list[str] = []
        for frame in f4.get("frames", []):
            if not isinstance(frame, Mapping):
                raise L2SealError(f"invalid F4 frame: {scene}")
            for source in frame.get("sources", []):
                if not isinstance(source, Mapping) or not isinstance(source.get("source_id"), str):
                    raise L2SealError(f"invalid F4 source: {scene}")
                source_id = str(source["source_id"])
                if source_id in f4_sources:
                    raise L2SealError(f"duplicate F4 source: {source_id}")
                hypotheses = source.get("hypotheses")
                if not isinstance(hypotheses, Mapping) or set(hypotheses) != set(HYPOTHESES):
                    raise L2SealError(f"F4 hypothesis bank differs: {source_id}")
                for name in HYPOTHESES:
                    if not _validate_hypothesis(hypotheses[name], name, source_id):
                        raise L2SealError(f"F4 raw source has invalid {name}: {source_id}")
                    per_hypothesis[name] += 1
                    total_geometries += 1
                f4_sources[source_id] = source
                f4_source_order.append(source_id)

        tracks = []
        all_f3_sources: set[str] = set()
        retained_sources: set[str] = set()
        two_view_retained_sources: set[str] = set()
        t2_count = 0
        for expected_track_id, track in enumerate(f3.get("tracks", [])):
            if (
                not isinstance(track, Mapping)
                or track.get("track_id") != expected_track_id
                or type(track.get("confirmed")) is not bool
            ):
                raise L2SealError(f"F3 track order differs: {scene}:{expected_track_id}")
            source_ids = track.get("source_ids")
            retained_ids = track.get("retained_source_ids")
            retained_frames = track.get("retained_frame_ids")
            if (
                not isinstance(source_ids, list)
                or not source_ids
                or not isinstance(retained_ids, list)
                or not isinstance(retained_frames, list)
                or not 1 <= len(retained_ids) == len(retained_frames) <= 5
                or retained_frames != sorted(set(retained_frames))
                or len(source_ids) != len(set(source_ids))
                or not set(retained_ids).issubset(set(source_ids))
            ):
                raise L2SealError(f"F3 source ledger differs: {scene}:{expected_track_id}")
            normalized_all = []
            for source_id_raw in source_ids:
                source_id = str(source_id_raw)
                match = SOURCE_RE.fullmatch(source_id)
                if (
                    match is None
                    or match["scene"] != scene
                    or source_id in all_f3_sources
                    or source_id not in f4_sources
                ):
                    raise L2SealError(f"F3/F4 all-source identity differs: {source_id}")
                all_f3_sources.add(source_id)
                normalized_all.append(source_id)
            normalized_retained = []
            for source_id_raw, frame_id in zip(retained_ids, retained_frames):
                source_id = str(source_id_raw)
                match = SOURCE_RE.fullmatch(source_id)
                if (
                    match is None
                    or int(match["frame"]) != frame_id
                    or source_id in retained_sources
                ):
                    raise L2SealError(f"F3 retained identity differs: {source_id}")
                retained_sources.add(source_id)
                normalized_retained.append(source_id)
            if len(normalized_retained) >= 2:
                t2_count += 1
                two_view_retained_sources.update(normalized_retained)
            tracks.append(
                {
                    "track_id": expected_track_id,
                    "f3_confirmed": bool(track["confirmed"]),
                    "source_ids": normalized_all,
                    "retained_source_ids": normalized_retained,
                    "retained_frame_ids": list(retained_frames),
                }
            )
        if all_f3_sources != set(f4_source_order):
            raise L2SealError(f"F3 track partition does not cover exact F4 sources: {scene}")
        mode_counts = {
            "T2": t2_count,
            "S2": len(two_view_retained_sources),
            "T1": len(tracks),
            "S1R": len(retained_sources),
            "SRAW": len(f4_source_order),
        }
        for name, count in mode_counts.items():
            totals_by_mode[name] += count
        total_tracks += len(tracks)
        total_retained += len(retained_sources)
        total_raw_sources += len(f4_source_order)
        scene_rows.append(
            {
                "scene_id": scene,
                "scene_index": scene_index,
                "f3": {"path": os.fspath(f3_path.resolve()), "sha256": _sha(f3_path)},
                "f4": {"path": os.fspath(f4_path.resolve()), "sha256": _sha(f4_path)},
                "mode_identity_counts": mode_counts,
                "f4_source_order": f4_source_order,
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
        "modes": {
            "T2": "track identity; retained views >=2",
            "S2": "source identity; retained sources from tracks with retained views >=2",
            "T1": "track identity; all tracks with retained views >=1",
            "S1R": "source identity; every retained F3 source",
            "SRAW": "source identity; every raw F4 source",
        },
        "counts": {
            "track_count": total_tracks,
            "retained_source_count": total_retained,
            "raw_source_count": total_raw_sources,
            "valid_geometry_count": total_geometries,
            "per_hypothesis": per_hypothesis,
            "mode_identity_counts": totals_by_mode,
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
            "per_view_geometry_preserved": True,
            "sam2_access": False,
            "tsdf_access": False,
            "mv3dis_access": False,
        },
        "scenes": scene_rows,
        "conclusion_guardrail": "L2 is a no-GT identity-universe seal with no AP.",
    }
    _write(args.out, receipt)
    print(json.dumps({"out": os.fspath(args.out), "counts": receipt["counts"]}, sort_keys=True))


if __name__ == "__main__":
    main()
