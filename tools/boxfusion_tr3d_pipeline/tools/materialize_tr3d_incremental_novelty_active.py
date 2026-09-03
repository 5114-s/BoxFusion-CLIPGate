#!/usr/bin/env python3
"""Materialize train-authorized incremental TR3D tracks append-only.

The command is GT-free.  It consumes causal observer diagnostics, applies an
immutable train-only policy, removes duplicate candidates geometrically, and
places all supplemental scores below every terminal-R3 anchor score.
"""

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


SCHEMA = "boxfusion.tr3d_incremental_novelty_active.v1"


def _load_diagnostic(path: Path, scene_id: str) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"missing/non-regular incremental diagnostic: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if (
        value.get("schema") != "boxfusion.tr3d_incremental_online_observer.v3"
        or not value.get("complete")
        or not value.get("observer_only")
        or value.get("mutation_enabled")
        or value.get("ground_truth_access")
        or int(value.get("applied_count", -1)) != 0
        or value.get("coordinate_frame") != "world_unaligned"
        or value.get("scene_id") != scene_id
        or not isinstance(value.get("confirmed"), list)
    ):
        raise ValueError(f"{path}: incremental observer safety contract failed")
    return value


def _prediction_arrays(payload: list[list[tuple[int, np.ndarray, float]]]):
    rows = payload[0]
    corners = np.stack([row[1] for row in rows]) if rows else np.empty((0, 8, 3), np.float32)
    scores = np.asarray([row[2] for row in rows], dtype=np.float32)
    return np.ascontiguousarray(corners, dtype=np.float32), scores


def _select(value: dict[str, Any], policy: IncrementalNoveltyPolicy, nms_iou: float):
    ranked = []
    provider_calls = int(value["provider_calls"])
    for row in value["confirmed"]:
        probability = policy.probability(row, provider_calls)
        if probability < policy.probability_threshold:
            continue
        if float(row["anchor_iou_max"]) > policy.hard_max_anchor_iou:
            continue
        corners = np.ascontiguousarray(row["best_corners_world"], dtype=np.float32)
        if corners.shape != (8, 3) or not np.isfinite(corners).all():
            raise ValueError("selected incremental candidate has invalid geometry")
        ranked.append((probability, float(row["best_score"]), int(row["track_id"]), corners))
    ranked.sort(key=lambda item: (-item[0], -item[1], item[2]))
    retained = []
    for item in ranked:
        if retained:
            existing = np.stack([saved[3] for saved in retained])
            if float(_aabb_iou(item[3][None], existing).max(initial=0.0)) > nms_iou:
                continue
        retained.append(item)
        if len(retained) >= policy.max_candidates_per_scene:
            break
    return retained, len(ranked)


def materialize(args: argparse.Namespace) -> dict[str, Any]:
    scenes = read_scene_list(args.scene_list.resolve())
    if len(scenes) not in (1, 10, 100):
        raise ValueError("incremental active requires 1, 10, or 100 scenes")
    if not 0.0 <= args.candidate_nms_iou < 1.0:
        raise ValueError("candidate NMS IoU must be in [0,1)")
    policy = IncrementalNoveltyPolicy.load(args.policy)
    anchor_root = args.anchor_root.resolve()
    diagnostics_root = args.diagnostics_root.resolve()
    output_root = args.output_root.resolve()
    manifest_path = args.manifest.resolve()
    if output_root.exists() or manifest_path.exists():
        raise FileExistsError("refusing existing incremental active namespace")
    before = _tree_snapshot(anchor_root, scenes)
    prepared = []
    entries = []
    anchor_floor = float("inf")
    for scene_index, scene_id in enumerate(scenes):
        anchor_path = anchor_root / f"{scene_id}_boxes.pkl"
        payload = _load_prediction(anchor_path)
        anchor_corners, anchor_scores = _prediction_arrays(payload)
        diagnostic_path = diagnostics_root / f"{scene_id}_tr3d_incremental.json"
        diagnostic = _load_diagnostic(diagnostic_path, scene_id)
        observed_corners = np.asarray(diagnostic["anchor_corners_world"], dtype=np.float32)
        observed_scores = np.asarray(diagnostic["anchor_scores"], dtype=np.float32)
        if (
            observed_corners.shape != anchor_corners.shape
            or observed_scores.shape != anchor_scores.shape
            or not np.array_equal(observed_corners, anchor_corners)
            or not np.array_equal(observed_scores, anchor_scores)
            or int(diagnostic.get("anchor_count", -1)) != len(anchor_corners)
        ):
            raise ValueError(f"{scene_id}: diagnostic is not bound to the anchor prediction")
        selected, eligible = _select(diagnostic, policy, args.candidate_nms_iou)
        for local_index, item in enumerate(selected):
            entries.append((scene_index, local_index, item[0]))
        if len(anchor_scores):
            anchor_floor = min(anchor_floor, float(anchor_scores.min()))
        prepared.append({
            "scene_id": scene_id, "payload": payload, "anchor_path": anchor_path,
            "anchor_sha256": sha256_file(anchor_path), "diagnostic_path": diagnostic_path,
            "diagnostic_sha256": sha256_file(diagnostic_path), "eligible_count": eligible,
            "selected": selected,
        })
    if not math.isfinite(anchor_floor) or anchor_floor <= 0.0:
        raise ValueError("anchor score floor must be positive")
    score_map = _assign_candidate_scores(entries, anchor_floor)
    reports = []
    for scene_index, row in enumerate(prepared):
        selected = row["selected"]
        corners = np.stack([item[3] for item in selected]) if selected else np.empty((0, 8, 3), np.float32)
        scores = [score_map[(scene_index, index)] for index in range(len(selected))]
        output = _append_payload(row["payload"], np.ascontiguousarray(corners, dtype=np.float32), scores)
        output_path = output_root / f"{row['scene_id']}_boxes.pkl"
        output_sha = _write_pickle_create_only(output_path, output)
        reports.append({
            "scene_id": row["scene_id"], "anchor_prediction": str(row["anchor_path"]),
            "anchor_prediction_sha256": row["anchor_sha256"],
            "incremental_diagnostic": str(row["diagnostic_path"]),
            "incremental_diagnostic_sha256": row["diagnostic_sha256"],
            "anchor_count": len(row["payload"][0]), "eligible_before_nms": row["eligible_count"],
            "applied_count": len(selected), "output_count": len(output[0]),
            "track_ids": [item[2] for item in selected],
            "probabilities": [item[0] for item in selected],
            "candidate_output_scores": scores, "output_prediction_sha256": output_sha,
        })
    after = _tree_snapshot(anchor_root, scenes)
    if before != after:
        raise RuntimeError("anchor prediction tree changed during materialization")
    manifest = {
        "schema": SCHEMA, "complete": True, "active": True,
        "activation_authorized": True, "append_only": True,
        "ground_truth_access": False, "clip_access": False,
        "clip_semantics_unchanged": True, "class_agnostic": True,
        "anchor_rows_first_and_unchanged": True,
        "candidate_scores_below_every_anchor": True,
        "candidate_nms_iou": args.candidate_nms_iou,
        "policy_checkpoint": str(policy.path), "policy_checkpoint_sha256": policy.sha256,
        "probability_threshold": policy.probability_threshold,
        "hard_max_anchor_iou": policy.hard_max_anchor_iou,
        "max_candidates_per_scene": policy.max_candidates_per_scene,
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


def parser():
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--scene-list", type=Path, required=True)
    value.add_argument("--policy", type=Path, required=True)
    value.add_argument("--anchor-root", type=Path, required=True)
    value.add_argument("--diagnostics-root", type=Path, required=True)
    value.add_argument("--output-root", type=Path, required=True)
    value.add_argument("--manifest", type=Path, required=True)
    value.add_argument("--candidate-nms-iou", type=float, default=0.25)
    return value


if __name__ == "__main__":
    result = materialize(parser().parse_args())
    print(json.dumps(result, indent=2, sort_keys=True))
