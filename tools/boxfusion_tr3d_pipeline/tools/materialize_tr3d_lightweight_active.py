#!/usr/bin/env python3
"""GT-free source-aware materialization for the lightweight TR3D route."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from boxfusion.tr3d_c2_maskrgbd_cache import sha256_file  # noqa: E402
from boxfusion.tr3d_incremental_gate import IncrementalNoveltyPolicy  # noqa: E402
from boxfusion.tr3d_incremental_online import _aabb_iou  # noqa: E402
from tools.materialize_tr3d_c3_active import (  # noqa: E402
    _append_payload, _assign_candidate_scores, _load_prediction,
    _write_json_create_only, _write_pickle_create_only,
)
from tools.run_tr3d_c1_track_observer import _tree_snapshot  # noqa: E402
from tools.tr3d_data import read_scene_list  # noqa: E402


SCHEMA = "boxfusion.tr3d_lightweight_active.v1"


def _prediction_arrays(payload):
    rows = payload[0]
    corners = np.stack([row[1] for row in rows]) if rows else np.empty((0, 8, 3), np.float32)
    scores = np.asarray([row[2] for row in rows], np.float32)
    return np.ascontiguousarray(corners, np.float32), scores


def _load_diagnostic(path: Path, scene_id: str, stage: int) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"missing/non-regular lightweight diagnostic: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if (
        value.get("schema") != "boxfusion.tr3d_lightweight_online_observer.v1"
        or not value.get("complete") or not value.get("observer_only")
        or value.get("mutation_enabled") or value.get("ground_truth_access")
        or int(value.get("applied_count", -1)) != 0
        or value.get("scene_id") != scene_id
        or int(value.get("lightweight_stage", -1)) != stage
        or bool(value.get("async_latest_only")) != (stage >= 3)
    ):
        raise ValueError(f"{path}: lightweight observer safety contract failed")
    return value


def _source_rank(row: dict[str, Any], probability: float, stage: int) -> float:
    if stage < 6:
        return float(probability)
    visibility = float(np.clip((float(row["visibility_quality_mean"]) + 1.0) * 0.5, 0.0, 1.0))
    free_space = float(np.clip(float(row["free_space_ratio_mean"]), 0.0, 1.0))
    support = float(np.clip(float(row["support_ratio_mean"]), 0.0, 1.0))
    geometry_bonus = 0.04 if row.get("selected_geometry") == "fused" else 0.0
    return float(probability + 0.10 * visibility + 0.08 * support - 0.18 * free_space + geometry_bonus)


def _select(value: dict[str, Any], policy: IncrementalNoveltyPolicy, nms_iou: float, stage: int):
    ranked = []
    provider_calls = int(value["provider_calls"])
    for original in value["confirmed"]:
        row = dict(original)
        row["anchor_iou_max"] = float(row["selected_anchor_iou_max"])
        row["anchor_center_distance_m"] = float(row["selected_anchor_center_distance_m"])
        probability = policy.probability(row, provider_calls)
        if probability < policy.probability_threshold:
            continue
        if row["anchor_iou_max"] > policy.hard_max_anchor_iou:
            continue
        # SMOV-style fail-closed protection against boxes mostly contradicted
        # by observed free space.  This is prediction-time geometry only.
        if stage >= 4 and float(row["free_space_ratio_mean"]) > 0.45:
            continue
        corners = np.asarray(row["selected_corners_world"], np.float32)
        if corners.shape != (8, 3) or not np.isfinite(corners).all():
            raise ValueError("invalid selected lightweight geometry")
        rank = _source_rank(row, probability, stage)
        ranked.append((rank, probability, float(row["best_score"]), int(row["track_id"]), corners, row["selected_geometry"]))
    ranked.sort(key=lambda item: (-item[0], -item[1], -item[2], item[3]))
    kept = []
    for item in ranked:
        if kept and float(_aabb_iou(item[4][None], np.stack([saved[4] for saved in kept])).max(initial=0.0)) > nms_iou:
            continue
        kept.append(item)
        if len(kept) >= policy.max_candidates_per_scene:
            break
    return kept, len(ranked)


def materialize(args: argparse.Namespace) -> dict[str, Any]:
    scenes = read_scene_list(args.scene_list.resolve())
    policy = IncrementalNoveltyPolicy.load(args.policy)
    anchor_root = args.anchor_root.resolve()
    diagnostics_root = args.diagnostics_root.resolve()
    output_root = args.output_root.resolve()
    manifest_path = args.manifest.resolve()
    if output_root.exists() or manifest_path.exists():
        raise FileExistsError("refusing existing lightweight active namespace")
    before = _tree_snapshot(anchor_root, scenes)
    prepared, entries = [], []
    anchor_floor = float("inf")
    for scene_index, scene_id in enumerate(scenes):
        anchor_path = anchor_root / f"{scene_id}_boxes.pkl"
        payload = _load_prediction(anchor_path)
        anchor_corners, anchor_scores = _prediction_arrays(payload)
        diagnostic_path = diagnostics_root / f"{scene_id}_tr3d_incremental.json"
        diagnostic = _load_diagnostic(diagnostic_path, scene_id, args.stage)
        observed_corners = np.asarray(diagnostic["anchor_corners_world"], np.float32)
        observed_scores = np.asarray(diagnostic["anchor_scores"], np.float32)
        if not np.array_equal(observed_corners, anchor_corners) or not np.array_equal(observed_scores, anchor_scores):
            raise ValueError(f"{scene_id}: lightweight diagnostic anchor binding failed")
        selected, eligible = _select(
            diagnostic, policy, args.candidate_nms_iou, args.stage
        )
        for local, item in enumerate(selected):
            entries.append((scene_index, local, item[0]))
        if len(anchor_scores):
            anchor_floor = min(anchor_floor, float(anchor_scores.min()))
        prepared.append((scene_id, payload, anchor_path, diagnostic_path, selected, eligible))
    if not math.isfinite(anchor_floor) or anchor_floor <= 0.0:
        raise ValueError("anchor score floor must be positive")
    score_map = _assign_candidate_scores(entries, anchor_floor)
    reports = []
    for scene_index, (scene_id, payload, anchor_path, diagnostic_path, selected, eligible) in enumerate(prepared):
        corners = np.stack([item[4] for item in selected]) if selected else np.empty((0, 8, 3), np.float32)
        scores = [score_map[(scene_index, index)] for index in range(len(selected))]
        output = _append_payload(payload, np.ascontiguousarray(corners, np.float32), scores)
        output_path = output_root / f"{scene_id}_boxes.pkl"
        output_sha = _write_pickle_create_only(output_path, output)
        reports.append({
            "scene_id": scene_id, "anchor_prediction": str(anchor_path),
            "anchor_prediction_sha256": sha256_file(anchor_path),
            "diagnostic": str(diagnostic_path),
            "diagnostic_sha256": sha256_file(diagnostic_path),
            "anchor_count": len(payload[0]), "eligible_before_nms": eligible,
            "applied_count": len(selected), "output_count": len(output[0]),
            "track_ids": [item[3] for item in selected],
            "source_ranks": [item[0] for item in selected],
            "probabilities": [item[1] for item in selected],
            "selected_geometry": [item[5] for item in selected],
            "candidate_output_scores": scores,
            "output_prediction_sha256": output_sha,
        })
    after = _tree_snapshot(anchor_root, scenes)
    if before != after:
        raise RuntimeError("anchor tree changed during lightweight materialization")
    manifest = {
        "schema": SCHEMA, "complete": True, "active": True,
        "activation_authorized": True, "append_only": True,
        "ground_truth_access": False, "clip_access": False,
        "clip_semantics_unchanged": True, "class_agnostic": True,
        "anchor_rows_first_and_unchanged": True,
        "candidate_scores_below_every_anchor": True,
        "lightweight_stage": args.stage,
        "source_aware_ranking": args.stage >= 6,
        "modules": ["ovscan_depth_visibility", "zoo3d_diverse_topk", "async_incremental_tr3d", "smov3d_free_space", "insfusion_raw_fused_choice", "source_aware_low_score_append"][:args.stage],
        "policy_checkpoint": str(policy.path),
        "policy_checkpoint_sha256": policy.sha256,
        "probability_threshold": policy.probability_threshold,
        "hard_max_anchor_iou": policy.hard_max_anchor_iou,
        "max_candidates_per_scene": policy.max_candidates_per_scene,
        "candidate_nms_iou": args.candidate_nms_iou,
        "scene_list": str(args.scene_list.resolve()),
        "scene_list_sha256": sha256_file(args.scene_list.resolve()),
        "scene_count": len(scenes), "anchor_root": str(anchor_root),
        "diagnostics_root": str(diagnostics_root), "output_root": str(output_root),
        "anchor_score_floor": anchor_floor,
        "anchor_count": sum(row["anchor_count"] for row in reports),
        "eligible_before_nms": sum(row["eligible_before_nms"] for row in reports),
        "applied_count": sum(row["applied_count"] for row in reports),
        "output_count": sum(row["output_count"] for row in reports),
        "anchor_tree_before": before, "anchor_tree_after": after,
        "output_tree": _tree_snapshot(output_root, scenes), "scenes": reports,
    }
    _write_json_create_only(manifest_path, manifest)
    return manifest


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--scene-list", type=Path, required=True)
    value.add_argument("--policy", type=Path, required=True)
    value.add_argument("--anchor-root", type=Path, required=True)
    value.add_argument("--diagnostics-root", type=Path, required=True)
    value.add_argument("--output-root", type=Path, required=True)
    value.add_argument("--manifest", type=Path, required=True)
    value.add_argument("--candidate-nms-iou", type=float, default=0.25)
    value.add_argument("--stage", type=int, choices=range(1, 7), required=True)
    return value


if __name__ == "__main__":
    result = materialize(parser().parse_args())
    print(json.dumps({key: result[key] for key in ("scene_count", "anchor_count", "eligible_before_nms", "applied_count", "output_count")}, indent=2))
