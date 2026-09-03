#!/usr/bin/env python3
"""Materialize C2: C1 plus native-first low-score confirmed births.

The candidate universe and SAM3/depth/CLIP decisions are frozen by the prior
no-GT observer manifest.  C2 additionally requires that the exact causal
receipt completed in C1.  Native rows remain an exact value/type prefix;
accepted candidates receive unique scores strictly between the evaluator
threshold and the global native score floor.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if os.fspath(ROOT) not in sys.path:
    sys.path.insert(0, os.fspath(ROOT))

from tools.materialize_scannet_raw_boxer_past3_birth_full100 import (  # noqa: E402
    PREDICTION_SUFFIX,
    _assert_native_prefix,
    _load_native_prediction,
    _regular_file,
    _scene_list,
    _write_json,
    _write_pickle,
)


SCHEMA = "boxfusion.scannet_c2_native_first_low_score_append_full100.v1"
C1_SCHEMA = "boxfusion.scannet_c1_causal_async_observer_full100.v1"
SAM3_SCHEMA = "boxfusion.scannet_sam3_diverse_clip_birth_full100.v1"
MANIFEST_NAME = "C2_NATIVE_FIRST_LOW_SCORE_APPEND_FULL100.json"
OFFICIAL_SCENE_LIST_SHA256 = (
    "4b18fc586f7ad60cb17f41ee7d1d8b0ab1a0782917b5ae3519dd8ec90e7744d5"
)
EVALUATOR_CONFIDENCE_THRESHOLD = 0.05
APPENDED_CLASS_ID = 0


class C2MaterializationError(RuntimeError):
    pass


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path, label: str) -> dict[str, Any]:
    _regular_file(path, label)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise C2MaterializationError(f"{label} root must be an object")
    return value


def _accepted_candidates(
    scenes: Sequence[str],
    sam3: Mapping[str, Any],
    c1_diagnostics_root: Path,
) -> tuple[list[dict[str, Any]], int]:
    accepted: list[dict[str, Any]] = []
    blocked_by_async = 0
    for scene in scenes:
        sam_scene = sam3["scenes"].get(scene)
        if not isinstance(sam_scene, dict):
            raise C2MaterializationError(f"missing SAM3 scene: {scene}")
        diagnostic = _read_json(
            c1_diagnostics_root / f"{scene}.json", "C1 scene diagnostic"
        )
        if (
            diagnostic.get("observer_only") is not True
            or diagnostic.get("output_mutation_applied") is not False
            or diagnostic.get("query_before_commit") is not True
        ):
            raise C2MaterializationError(f"invalid C1 diagnostic contract: {scene}")
        completed = {
            (
                int(row["candidate_id"]),
                tuple(int(value) for value in row["evidence_source_rows"]),
            ): row
            for row in diagnostic["results"]
        }
        dropped = {int(row["candidate_id"]) for row in diagnostic["drops"]}
        decisions = {
            int(row["track_id"]): row
            for row in sam_scene.get("decisions", [])
            if isinstance(row, dict)
        }
        for suffix in sam_scene.get("suffix", []):
            track_id = int(suffix["track_id"])
            decision = decisions.get(track_id)
            if (
                decision is None
                or decision.get("decision") != "accepted"
                or decision.get("clip_gate_pass") is not True
                or decision.get("sam3_mask_depth", {}).get("mask_depth_pass") is not True
            ):
                raise C2MaterializationError(
                    f"SAM3 accepted suffix contract mismatch: {scene}/{track_id}"
                )
            confirmation = int(suffix["confirmation_frame_id"])
            evidence_rows = tuple(int(value) for value in suffix["evidence_source_rows"])
            key = (track_id, evidence_rows)
            if key not in completed:
                if track_id in dropped:
                    blocked_by_async += 1
                    continue
                raise C2MaterializationError(
                    f"C1 lacks accepted receipt identity: {scene}/{track_id}"
                )
            c1_row = completed[key]
            mask_depth = decision["sam3_mask_depth"]
            if (
                int(c1_row["enqueue_frame_id"]) != confirmation
                or int(c1_row["memory_version"]) < 1
                or max(int(value) for value in mask_depth["selected_frame_ids"])
                > confirmation
            ):
                raise C2MaterializationError(
                    f"C2 causal timestamp mismatch: {scene}/{track_id}"
                )
            corners = np.asarray(suffix["corners_world"], dtype=np.float64)
            if (
                corners.shape != (8, 3)
                or not np.isfinite(corners).all()
                or np.any(np.ptp(corners, axis=0) <= 0.0)
            ):
                raise C2MaterializationError(
                    f"invalid accepted geometry: {scene}/{track_id}"
                )
            clip = decision.get("clip_summary") or {}
            rank_key = (
                -int(mask_depth["strong_view_count"]),
                -float(mask_depth["mean_strong_inside_expanded"]),
                -int(mask_depth["total_component_points"]),
                -float(clip.get("median_target_best_cosine", -1.0)),
                -float(clip.get("target_margin_median", -1.0)),
                scene,
                track_id,
            )
            accepted.append(
                {
                    "scene_id": scene,
                    "track_id": track_id,
                    "confirmation_frame_id": confirmation,
                    "evidence_frame_ids": [int(value) for value in suffix["evidence_frame_ids"]],
                    "evidence_source_rows": list(evidence_rows),
                    "corners": corners,
                    "rank_key": rank_key,
                    "strong_view_count": int(mask_depth["strong_view_count"]),
                    "mean_strong_inside_expanded": float(
                        mask_depth["mean_strong_inside_expanded"]
                    ),
                    "total_component_points": int(mask_depth["total_component_points"]),
                    "clip_median_target_cosine": float(
                        clip.get("median_target_best_cosine", -1.0)
                    ),
                    "clip_median_margin": float(clip.get("target_margin_median", -1.0)),
                }
            )
    accepted.sort(key=lambda row: row["rank_key"])
    return accepted, blocked_by_async


def materialize(args: argparse.Namespace) -> dict[str, Any]:
    scene_list = args.scene_list.resolve()
    scenes = _scene_list(scene_list, args.expected_scene_count)
    scene_list_sha = _sha256(scene_list)
    if args.expected_scene_count == 100 and scene_list_sha != OFFICIAL_SCENE_LIST_SHA256:
        raise C2MaterializationError("official100 scene-list hash mismatch")

    c1_root = args.c1_root.resolve()
    output_root = args.output_root.resolve()
    if c1_root.is_symlink() or not c1_root.is_dir():
        raise C2MaterializationError(f"invalid C1 root: {c1_root}")
    if output_root.exists() or output_root.is_symlink():
        raise C2MaterializationError(f"refusing to overwrite output root: {output_root}")
    c1_manifest_path = args.c1_manifest.resolve()
    sam3_manifest_path = args.sam3_manifest.resolve()
    c1 = _read_json(c1_manifest_path, "C1 manifest")
    sam3 = _read_json(sam3_manifest_path, "SAM3/CLIP manifest")
    if (
        c1.get("schema") != C1_SCHEMA
        or c1.get("observer_only") is not True
        or c1.get("output_inert") is not True
        or c1.get("native_predictions_byte_identical") is not True
        or c1.get("past_only") is not True
        or c1.get("gt_access") is not False
        or c1.get("evaluator_access") is not False
    ):
        raise C2MaterializationError("C1 manifest contract mismatch")
    if (
        sam3.get("schema") != SAM3_SCHEMA
        or sam3.get("past_only_confirmation") is not True
        or sam3.get("gt_access") is not False
        or sam3.get("evaluator_access") is not False
        or sam3.get("birth_count") != 1
    ):
        raise C2MaterializationError("SAM3/CLIP manifest contract mismatch")

    accepted, blocked_by_async = _accepted_candidates(
        scenes, sam3, c1_root / "observer_diagnostics"
    )
    natives: dict[str, Any] = {}
    native_hashes: dict[str, str] = {}
    all_native_scores: list[float] = []
    for scene in scenes:
        path = _regular_file(
            c1_root / f"{scene}{PREDICTION_SUFFIX}", "C1 prediction"
        )
        digest = _sha256(path)
        if (
            digest != c1["output_prediction_sha256"].get(scene)
            or digest != sam3["native_prediction_sha256"].get(scene)
        ):
            raise C2MaterializationError(f"C1/SAM3 native hash mismatch: {scene}")
        native = _load_native_prediction(path)
        natives[scene] = native
        native_hashes[scene] = digest
        all_native_scores.extend(float(row[2]) for row in native.rows)
    if not all_native_scores:
        raise C2MaterializationError("C1 contains no native scores")
    native_floor = min(all_native_scores)
    if native_floor <= EVALUATOR_CONFIDENCE_THRESHOLD:
        raise C2MaterializationError(
            "native score floor leaves no positive evaluator-visible suffix interval"
        )
    count = len(accepted)
    for rank, row in enumerate(accepted):
        fraction = (rank + 1) / (count + 1)
        score = native_floor - fraction * (
            native_floor - EVALUATOR_CONFIDENCE_THRESHOLD
        )
        if not EVALUATOR_CONFIDENCE_THRESHOLD < score < native_floor:
            raise C2MaterializationError("invalid low-score mapping")
        row["rank"] = rank
        row["append_score"] = float(score)

    by_scene: dict[str, list[dict[str, Any]]] = {scene: [] for scene in scenes}
    for row in accepted:
        by_scene[row["scene_id"]].append(row)
    output_root.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{output_root.name}.stage-", dir=output_root.parent))
    output_hashes: dict[str, str] = {}
    scene_reports: dict[str, Any] = {}
    total_rows = 0
    try:
        for position, scene in enumerate(scenes, 1):
            native = natives[scene]
            suffix_rows = [
                (
                    APPENDED_CLASS_ID,
                    np.ascontiguousarray(row["corners"], dtype=np.float32),
                    float(row["append_score"]),
                )
                for row in by_scene[scene]
            ]
            rows: list[Any] | tuple[Any, ...]
            if isinstance(native.rows, tuple):
                rows = tuple(native.rows) + tuple(suffix_rows)
            else:
                rows = list(native.rows) + suffix_rows
            payload: list[Any] | tuple[Any, ...]
            payload = (rows,) if isinstance(native.payload, tuple) else [rows]
            _assert_native_prefix(native.rows, rows, scene)
            output_path = stage / f"{scene}{PREDICTION_SUFFIX}"
            _write_pickle(output_path, payload)
            reloaded = _load_native_prediction(output_path)
            _assert_native_prefix(native.rows, reloaded.rows, scene)
            if len(reloaded.rows) != len(native.rows) + len(suffix_rows):
                raise C2MaterializationError(f"C2 row count mismatch: {scene}")
            if any(float(row[2]) >= native_floor for row in reloaded.rows[len(native.rows):]):
                raise C2MaterializationError(f"C2 suffix crossed native floor: {scene}")
            output_hashes[scene] = _sha256(output_path)
            scene_reports[scene] = {
                "native_rows": len(native.rows),
                "appended_rows": len(suffix_rows),
                "output_rows": len(reloaded.rows),
                "native_prefix_preserved": True,
                "suffix": [
                    {
                        key: value
                        for key, value in row.items()
                        if key not in ("corners", "rank_key")
                    }
                    | {"corners_world": row["corners"].tolist()}
                    for row in by_scene[scene]
                ],
            }
            total_rows += len(reloaded.rows)
            print(
                f"[{position}/{len(scenes)}] {scene}: native={len(native.rows)} "
                f"append={len(suffix_rows)} output={len(reloaded.rows)}",
                flush=True,
            )

        manifest = {
            "schema": SCHEMA,
            "mode": "c2_native_first_low_score_append",
            "scene_count": len(scenes),
            "native_rows": len(all_native_scores),
            "appended_rows": len(accepted),
            "output_rows": total_rows,
            "blocked_by_c1_async_timeout": blocked_by_async,
            "native_score_floor": native_floor,
            "evaluator_confidence_threshold": EVALUATOR_CONFIDENCE_THRESHOLD,
            "minimum_append_score": (
                min(row["append_score"] for row in accepted) if accepted else None
            ),
            "maximum_append_score": (
                max(row["append_score"] for row in accepted) if accepted else None
            ),
            "native_rows_are_unchanged_prefix": True,
            "native_first_nms": True,
            "append_can_suppress_native": False,
            "append_scores_strictly_below_all_native": True,
            "append_scores_unique": len({row["append_score"] for row in accepted}) == len(accepted),
            "geometry_overlay": False,
            "native_geometry_changed": False,
            "native_score_changed": False,
            "native_label_changed": False,
            "native_row_order_changed": False,
            "birth": True,
            "training_free": True,
            "online_learning": False,
            "gt_access": False,
            "annotation_access": False,
            "evaluator_access": False,
            "past_only": True,
            "ranking_policy": (
                "strong_views_desc,mean_inside_desc,component_points_desc,"
                "clip_cosine_desc,clip_margin_desc,scene,track"
            ),
            "score_mapping": (
                "native_floor-rank_fraction*(native_floor-conf_thresh)"
            ),
            "inputs": {
                "scene_list": os.fspath(scene_list),
                "scene_list_sha256": scene_list_sha,
                "c1_root": os.fspath(c1_root),
                "c1_manifest": os.fspath(c1_manifest_path),
                "c1_manifest_sha256": _sha256(c1_manifest_path),
                "sam3_manifest": os.fspath(sam3_manifest_path),
                "sam3_manifest_sha256": _sha256(sam3_manifest_path),
                "materializer": os.fspath(Path(__file__).resolve()),
                "materializer_sha256": _sha256(Path(__file__).resolve()),
            },
            "native_prediction_sha256": native_hashes,
            "output_prediction_sha256": output_hashes,
            "accepted_candidate_order": [
                {
                    key: value
                    for key, value in row.items()
                    if key not in ("corners", "rank_key")
                }
                for row in accepted
            ],
            "scenes": scene_reports,
        }
        _write_json(stage / MANIFEST_NAME, manifest)
        if output_root.exists() or output_root.is_symlink():
            raise C2MaterializationError(f"refusing existing output root: {output_root}")
        os.rename(stage, output_root)
        return manifest
    except Exception:
        if stage.exists():
            shutil.rmtree(stage)
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scene-list", type=Path,
        default=ROOT / "evaluation/data_util/meta_data/scannetv2_val.txt",
    )
    parser.add_argument("--expected-scene-count", type=int, default=100)
    parser.add_argument(
        "--c1-root", type=Path,
        default=ROOT / "results/scannet_cbest_real_score_c1_causal_async_observer_score05",
    )
    parser.add_argument(
        "--c1-manifest", type=Path,
        default=(
            ROOT / "results/scannet_cbest_real_score_c1_causal_async_observer_score05/"
            "C1_CAUSAL_ASYNC_OBSERVER_FULL100.json"
        ),
    )
    parser.add_argument(
        "--sam3-manifest", type=Path,
        default=(
            ROOT / "results/scannet_cbest_sam3_diverse_clip_birth_score05/"
            "SAM3_DIVERSE_CLIP_BIRTH_FULL100.json"
        ),
    )
    parser.add_argument(
        "--output-root", type=Path,
        default=ROOT / "results/scannet_cbest_real_score_c2_native_first_low_score_append_score05",
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    result = materialize(args)
    print(
        json.dumps(
            {
                key: result[key]
                for key in (
                    "schema", "scene_count", "native_rows", "appended_rows",
                    "output_rows", "blocked_by_c1_async_timeout",
                    "native_score_floor", "minimum_append_score",
                    "maximum_append_score",
                )
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
