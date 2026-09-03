#!/usr/bin/env python3
"""GT-only audit for C1 unmatched-TR3D cross-view evidence tracks."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Any, Sequence

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from boxfusion.tr3d_c1_track_cache import (  # noqa: E402
    GATE_NAMES,
    RESIDUAL_ANCHOR_IOU_MAX,
    load_sidecar,
    sha256_file,
    sidecar_path,
)
from boxfusion.tr3d_residual_cache import (  # noqa: E402
    load_tr3d_residual_cache,
    tr3d_residual_cache_path,
)
from tools.audit_tr3d_residual_observer import (  # noqa: E402
    _alignment,
    _gt_boxes,
    _load_b6,
    _minmax,
    _transform,
    _validate_alignment_provenance,
    maximum_cardinality,
    pairwise_iou,
)
from tools.run_tr3d_c1_track_observer import (  # noqa: E402
    REPORT_SCHEMA as EXPORT_SCHEMA,
    _code_hash,
    _tree_snapshot,
)
from tools.tr3d_data import read_scene_list  # noqa: E402


REPORT_SCHEMA = "boxfusion.tr3d_c1_track_gt_audit.v1"
IOU_THRESHOLDS = (0.15, 0.25, 0.50)
PER_SCENE_BUDGETS = (1, 3, 5, 10)


def _write_json_create_only(path: Path, payload: dict[str, Any]) -> None:
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", prefix=f".{path.name}.", suffix=".tmp",
            dir=path.parent, delete=False,
        ) as handle:
            temporary_name = handle.name
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary_name, path)
        path.chmod(0o444)
    except FileExistsError as error:
        raise FileExistsError(f"immutable C1 audit exists: {path}") from error
    finally:
        if temporary_name is not None:
            try:
                os.unlink(temporary_name)
            except FileNotFoundError:
                pass


def _load_export(path: Path) -> dict[str, Any]:
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("schema") != EXPORT_SCHEMA:
        raise ValueError("unsupported C1 export report")
    if not report.get("observer_only") or report.get("mutation_enabled"):
        raise ValueError("C1 export violates observer contract")
    if int(report.get("applied_count", -1)) != 0:
        raise ValueError("C1 export applied_count must be zero")
    if report.get("ground_truth_access") or report.get("clip_access"):
        raise ValueError("C1 exporter improperly accessed GT/CLIP")
    if report.get("cross_prefix_tracking"):
        raise ValueError("terminal C1 exporter mislabels cross-view evidence")
    if report.get("code_sha256") != _code_hash():
        raise ValueError("C1 code differs from immutable export")
    return report


def _self_duplicate_count(boxes: np.ndarray, scores: np.ndarray, threshold: float) -> int:
    if len(boxes) < 2:
        return 0
    iou = pairwise_iou(boxes, boxes)
    order = np.argsort(-np.asarray(scores), kind="stable")
    kept: list[int] = []
    duplicates = 0
    for row in order:
        if kept and float(iou[int(row), kept].max(initial=0.0)) > threshold:
            duplicates += 1
        else:
            kept.append(int(row))
    return duplicates


def _empty_totals() -> dict[str, Any]:
    return {
        "candidate_count": 0,
        "positive_scenes": {f"{t:.2f}": 0 for t in IOU_THRESHOLDS},
        "anchor_matches": {f"{t:.2f}": 0 for t in IOU_THRESHOLDS},
        "union_matches": {f"{t:.2f}": 0 for t in IOU_THRESHOLDS},
        "direct_hits": {f"{t:.2f}": 0 for t in IOU_THRESHOLDS},
        "self_duplicates": {f"{t:.2f}": 0 for t in IOU_THRESHOLDS},
    }


def _add_route(
    totals: dict[str, Any], *, anchor_iou: np.ndarray, candidate_iou: np.ndarray,
    candidate_boxes: np.ndarray, candidate_scores: np.ndarray,
) -> None:
    totals["candidate_count"] += len(candidate_iou)
    for threshold in IOU_THRESHOLDS:
        key = f"{threshold:.2f}"
        anchor = maximum_cardinality(anchor_iou, threshold)
        union = maximum_cardinality(np.concatenate((anchor_iou, candidate_iou), axis=0), threshold)
        totals["anchor_matches"][key] += anchor
        totals["union_matches"][key] += union
        totals["positive_scenes"][key] += int(union > anchor)
        totals["direct_hits"][key] += int(
            np.count_nonzero(candidate_iou.max(axis=1) > threshold)
            if candidate_iou.shape[1] else 0
        )
        totals["self_duplicates"][key] += _self_duplicate_count(
            candidate_boxes, candidate_scores, threshold
        )


def _finalize_route(raw: dict[str, Any], total_gt: int) -> dict[str, Any]:
    count = int(raw["candidate_count"])
    thresholds = {}
    for threshold in IOU_THRESHOLDS:
        key = f"{threshold:.2f}"
        anchor = int(raw["anchor_matches"][key])
        union = int(raw["union_matches"][key])
        direct = int(raw["direct_hits"][key])
        duplicate = int(raw["self_duplicates"][key])
        thresholds[key] = {
            "anchor_oracle_tp": anchor,
            "union_oracle_tp": union,
            "novel_oracle_tp": union - anchor,
            "anchor_oracle_recall": anchor / max(total_gt, 1),
            "union_oracle_recall": union / max(total_gt, 1),
            "delta_oracle_recall": (union - anchor) / max(total_gt, 1),
            "direct_gt_hits": direct,
            "independent_gt_hit_precision": direct / max(count, 1),
            "positive_scene_count": int(raw["positive_scenes"][key]),
            "self_duplicate_count": duplicate,
            "self_duplicate_rate": duplicate / max(count, 1),
        }
    return {"candidate_count": count, "thresholds": thresholds}


def audit(args: argparse.Namespace) -> dict[str, Any]:
    export = _load_export(args.export_report.resolve())
    scenes = read_scene_list(args.scene_list.resolve())
    if export.get("scene_count") != len(scenes):
        raise ValueError("scene count mismatch")
    active_root = args.active_prediction_root.resolve()
    if Path(export["active_prediction_root"]).resolve() != active_root:
        raise ValueError("active prediction root mismatch")
    before = _tree_snapshot(active_root, scenes)
    if before["tree_sha256"] != export["frozen_active_before"]["tree_sha256"]:
        raise ValueError("R3-active prediction tree differs from C1 export")
    exported_rows = {str(row["scene_id"]): row for row in export["scenes"]}
    route_names = ("all_unmatched",) + GATE_NAMES
    totals = {name: _empty_totals() for name in route_names}
    budget_totals = {
        f"top{budget}_per_scene": _empty_totals() for budget in PER_SCENE_BUDGETS
    }
    total_gt = total_anchor = total_parent = total_tracks = 0
    per_scene: list[dict[str, Any]] = []

    for scene_id in scenes:
        sidecar_file = sidecar_path(args.c1_cache_root.resolve(), scene_id, args.prefix_id)
        if sha256_file(sidecar_file) != exported_rows[scene_id]["sidecar_sha256"]:
            raise ValueError(f"{scene_id}: C1 sidecar hash mismatch")
        track = load_sidecar(sidecar_file)
        parent_path = tr3d_residual_cache_path(args.parent_cache_root.resolve(), scene_id, args.prefix_id)
        if sha256_file(parent_path) != track.parent_cache_sha256:
            raise ValueError(f"{scene_id}: parent cache hash mismatch")
        with np.load(parent_path, allow_pickle=False) as raw_parent:
            checkpoint_sha = str(np.asarray(raw_parent["checkpoint_sha256"]).item())
            config_sha = str(np.asarray(raw_parent["config_sha256"]).item())
        parent = load_tr3d_residual_cache(
            parent_path, expected_scene_id=scene_id, expected_prefix_id=args.prefix_id,
            expected_checkpoint_sha256=checkpoint_sha, expected_config_sha256=config_sha,
        )
        anchor_path = active_root / f"{scene_id}_boxes.pkl"
        if sha256_file(anchor_path) != track.anchor_prediction_sha256:
            raise ValueError(f"{scene_id}: active anchor hash mismatch")
        if not np.array_equal(parent.proposal_ids[track.parent_rows], track.proposal_ids):
            raise ValueError(f"{scene_id}: C1 parent-row identity mismatch")

        alignment = _alignment(args.scans_root.resolve(), scene_id)
        _validate_alignment_provenance(scene_id, alignment, parent.aligned_to_unaligned)
        all_candidate_boxes = _minmax(_transform(parent.corners_world, alignment))
        candidate_boxes = all_candidate_boxes[track.parent_rows]
        anchor_corners, _ = _load_b6(anchor_path)
        anchor_boxes = _minmax(_transform(anchor_corners, alignment))
        targets = _gt_boxes(args.gt_root.resolve() / f"{scene_id}_bbox.npy")
        anchor_vs_gt = pairwise_iou(anchor_boxes, targets)
        candidate_vs_gt = pairwise_iou(candidate_boxes, targets)
        candidate_vs_anchor = pairwise_iou(candidate_boxes, anchor_boxes)
        recomputed_max = (
            candidate_vs_anchor.max(axis=1)
            if candidate_vs_anchor.shape[1]
            else np.zeros(track.track_count, dtype=np.float64)
        )
        if np.any(recomputed_max > RESIDUAL_ANCHOR_IOU_MAX + 1e-9):
            raise ValueError(f"{scene_id}: C1 contains matched candidates")
        if not np.allclose(recomputed_max, track.max_anchor_iou, rtol=0, atol=1e-6):
            raise ValueError(f"{scene_id}: stored anchor IoU mismatch")

        masks = {"all_unmatched": np.ones(track.track_count, dtype=np.bool_)}
        masks.update({name: track.gate_mask[:, index] for index, name in enumerate(GATE_NAMES)})
        scene_routes = {}
        for name, mask in masks.items():
            _add_route(
                totals[name], anchor_iou=anchor_vs_gt,
                candidate_iou=candidate_vs_gt[mask], candidate_boxes=candidate_boxes[mask],
                candidate_scores=track.depth_feature_track_score[mask],
            )
            scene_routes[name] = int(np.count_nonzero(mask))
        ranking_order = np.argsort(-track.depth_feature_track_score, kind="stable")
        for budget in PER_SCENE_BUDGETS:
            selected = ranking_order[:budget]
            _add_route(
                budget_totals[f"top{budget}_per_scene"], anchor_iou=anchor_vs_gt,
                candidate_iou=candidate_vs_gt[selected], candidate_boxes=candidate_boxes[selected],
                candidate_scores=track.depth_feature_track_score[selected],
            )
        per_scene.append({
            "scene_id": scene_id, "ground_truth": len(targets),
            "active_anchor_predictions": len(anchor_boxes),
            "parent_proposals": parent.proposal_count, "unmatched_tracks": track.track_count,
            "gate_counts": scene_routes,
        })
        total_gt += len(targets)
        total_anchor += len(anchor_boxes)
        total_parent += parent.proposal_count
        total_tracks += track.track_count

    routes = {name: _finalize_route(raw, total_gt) for name, raw in totals.items()}
    budgets = {name: _finalize_route(raw, total_gt) for name, raw in budget_totals.items()}
    primary = routes["depth_feature2"]
    p15 = primary["thresholds"]["0.15"]
    p25 = primary["thresholds"]["0.25"]
    advance = (
        p15["delta_oracle_recall"] >= 0.05
        and p25["delta_oracle_recall"] >= 0.05
        and p25["novel_oracle_tp"] >= 10
        and p25["independent_gt_hit_precision"] >= 0.05
        and p25["positive_scene_count"] >= 3
    )
    after = _tree_snapshot(active_root, scenes)
    if before != after:
        raise RuntimeError("R3-active prediction tree changed during C1 audit")
    return {
        "schema": REPORT_SCHEMA,
        "observer_contract": {
            "observer_only": True, "mutation_enabled": False, "applied_count": 0,
            "standard_ap_unchanged_by_design": True,
            "frozen_active_verified_before_and_after": True,
            "before": before, "after": after,
        },
        "scope_disclosure": {
            "track_type": "multi-view evidence for one terminal-prefix proposal",
            "cross_prefix_tracking": False,
            "not_claimed": "temporal proposal association across p25/p50/p75/p100",
        },
        "scene_count": len(scenes),
        "counts": {
            "ground_truth": total_gt, "active_anchor_predictions": total_anchor,
            "parent_proposals": total_parent, "unmatched_tracks": total_tracks,
        },
        "routes": routes, "ranked_budgets": budgets, "per_scene": per_scene,
        "pre_registered_advance_gate": {
            "primary_route": "depth_feature2",
            "requirements": {
                "delta_oracle_recall_15_min": 0.05,
                "delta_oracle_recall_25_min": 0.05,
                "novel_oracle_tp25_min": 10,
                "independent_gt_hit_precision25_min": 0.05,
                "positive_scene_count25_min": 3,
            },
            "pass": bool(advance),
            "meaning": (
                "PASS only authorizes a separate C2 source-aware confirmation "
                "observer; it never authorizes active candidate output"
            ),
        },
        "input_hashes": {
            "export_report_sha256": sha256_file(args.export_report.resolve()),
            "scene_list_sha256": sha256_file(args.scene_list.resolve()),
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export-report", type=Path, required=True)
    parser.add_argument("--c1-cache-root", type=Path, required=True)
    parser.add_argument("--parent-cache-root", type=Path, required=True)
    parser.add_argument("--active-prediction-root", type=Path, required=True)
    parser.add_argument("--scene-list", type=Path, required=True)
    parser.add_argument("--prefix-id", default="p100")
    parser.add_argument("--gt-root", type=Path, default=Path("/data/ZhaoX/BoxFusion/evaluation/data_util/scannet_train_detection_data"))
    parser.add_argument("--scans-root", type=Path, default=Path("/extra/ZhaoX/scannet_data/scans"))
    parser.add_argument("--report", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = audit(args)
    _write_json_create_only(args.report.resolve(), report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
