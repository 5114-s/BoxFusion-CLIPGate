#!/usr/bin/env python3
"""Offline GT-only AP audit for the frozen C2 top5_mask2_depth route.

This tool never writes or rewrites prediction artifacts.  It appends C2
candidates in memory only and reports exploratory, post-hoc counterfactuals.
Ground truth is confined to this audit and no result authorizes active output.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any, Sequence

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from boxfusion.tr3d_c2_maskrgbd_cache import (  # noqa: E402
    load_sidecar,
    sha256_file,
    sidecar_path,
)
from boxfusion.tr3d_c2_maskrgbd_observer import GATE_NAMES  # noqa: E402
from boxfusion.tr3d_residual_cache import (  # noqa: E402
    load_tr3d_residual_cache,
    tr3d_residual_cache_path,
)
from tools.audit_tr3d_c2_maskrgbd_observer import (  # noqa: E402
    _partition_scene_ids,
    _write_json_create_only,
)
from tools.audit_tr3d_r4_verifier import (  # noqa: E402
    THRESHOLDS,
    _alignment,
    _ground_truth,
    _iou,
    _metrics,
    _minmax,
    _prediction,
    _voc_ap,
)
from tools.audit_tr3d_residual_observer import maximum_cardinality  # noqa: E402
from tools.run_tr3d_c1_track_observer import _tree_snapshot  # noqa: E402
from tools.run_tr3d_c2_maskrgbd_observer import (  # noqa: E402
    REPORT_SCHEMA as C2_EXPORT_SCHEMA,
    _code_hash as c2_code_hash,
)
from tools.tr3d_data import read_scene_list  # noqa: E402


REPORT_SCHEMA = "boxfusion.tr3d_c3_shadow_counterfactual.v1"
ROUTE_NAME = "top5_mask2_depth"
FIXED_LOW_SCORE = -1_000_000.0


def _delta(
    metrics: dict[str, dict[str, float | int]],
    anchor: dict[str, dict[str, float | int]],
) -> dict[str, dict[str, float | int]]:
    result: dict[str, dict[str, float | int]] = {}
    for threshold in THRESHOLDS:
        key = f"{threshold:.2f}"
        ap_delta = float(metrics[key]["average_precision"]) - float(
            anchor[key]["average_precision"]
        )
        result[key] = {
            "delta_average_precision": ap_delta,
            "delta_ap_percentage_points": 100.0 * ap_delta,
            "delta_matched_tp": int(metrics[key]["matched_tp"])
            - int(anchor[key]["matched_tp"]),
            "delta_final_recall": float(metrics[key]["final_recall"])
            - float(anchor[key]["final_recall"]),
        }
    return result


def _with_ap_percent(
    metrics: dict[str, dict[str, float | int]],
) -> dict[str, dict[str, float | int]]:
    return {
        key: {**row, "ap_percent": 100.0 * float(row["average_precision"])}
        for key, row in metrics.items()
    }


def _policy_rows(
    rows: Sequence[dict[str, Any]],
    candidate_scores: Sequence[np.ndarray] | None,
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for scene_index, row in enumerate(rows):
        if candidate_scores is None:
            boxes = row["anchor_boxes"]
            scores = row["anchor_scores"]
        else:
            boxes = np.concatenate(
                (row["anchor_boxes"], row["candidate_boxes"]), axis=0
            )
            scores = np.concatenate(
                (row["anchor_scores"], candidate_scores[scene_index]), axis=0
            )
        result.append(
            {
                "scene_id": row["scene_id"],
                "boxes": boxes,
                "scores": scores,
                "gt": row["gt"],
            }
        )
    return result


def _c1_rank_scores(rows: Sequence[dict[str, Any]]) -> list[np.ndarray]:
    anchor_values = [row["anchor_scores"] for row in rows if len(row["anchor_scores"])]
    anchor_floor = min(float(values.min()) for values in anchor_values) if anchor_values else 0.0
    result = [np.empty(len(row["candidate_boxes"]), dtype=np.float64) for row in rows]
    references = [
        (float(score), scene_index, local_index)
        for scene_index, row in enumerate(rows)
        for local_index, score in enumerate(row["c1_track_scores"])
    ]
    references.sort(key=lambda item: (-item[0], item[1], item[2]))
    for rank, (_, scene_index, local_index) in enumerate(references):
        result[scene_index][local_index] = anchor_floor - 1.0 - rank
    return result


def _ordered_metric(matches: Sequence[bool], total_gt: int) -> dict[str, float | int]:
    tp = np.asarray(matches, dtype=np.float64)
    fp = 1.0 - tp
    cumulative_tp = np.cumsum(tp)
    cumulative_fp = np.cumsum(fp)
    recall = cumulative_tp / float(total_gt + 1e-6)
    precision = cumulative_tp / np.maximum(
        cumulative_tp + cumulative_fp, np.finfo(np.float64).eps
    )
    return {
        "predictions": len(matches),
        "ground_truth": total_gt,
        "matched_tp": int(tp.sum()),
        "average_precision": _voc_ap(recall, precision) if len(tp) else 0.0,
        "final_precision": float(precision[-1]) if len(precision) else 0.0,
        "final_recall": float(recall[-1]) if len(recall) else 0.0,
    }


def _gt_oracle_metrics(
    rows: Sequence[dict[str, Any]],
    anchor_metrics: dict[str, dict[str, float | int]],
) -> dict[str, dict[str, float | int]]:
    """Candidate-only GT upper bound with every anchor ranked first."""

    output: dict[str, dict[str, float | int]] = {}
    total_gt = sum(len(row["gt"]) for row in rows)
    candidate_count = sum(len(row["candidate_boxes"]) for row in rows)
    anchor_records = [
        (float(score), scene_index, local_index)
        for scene_index, row in enumerate(rows)
        for local_index, score in enumerate(row["anchor_scores"])
    ]
    anchor_records.sort(key=lambda item: (-item[0], item[1], item[2]))
    for threshold in THRESHOLDS:
        key = f"{threshold:.2f}"
        used = [np.zeros(len(row["gt"]), dtype=np.bool_) for row in rows]
        ordered_anchor_matches: list[bool] = []
        for _, scene_index, local_index in anchor_records:
            row = rows[scene_index]
            overlaps = _iou(
                row["anchor_boxes"][local_index : local_index + 1], row["gt"]
            )[0]
            matched = False
            if len(overlaps):
                target = int(np.argmax(overlaps))
                matched = bool(
                    overlaps[target] > threshold and not used[scene_index][target]
                )
                if matched:
                    used[scene_index][target] = True
            ordered_anchor_matches.append(matched)
        novel_matches = 0
        for scene_index, row in enumerate(rows):
            remaining = row["gt"][~used[scene_index]]
            novel_matches += maximum_cardinality(
                _iou(row["candidate_boxes"], remaining), threshold
            )
        ordered = [
            *ordered_anchor_matches,
            *([True] * novel_matches),
            *([False] * (candidate_count - novel_matches)),
        ]
        metric = _ordered_metric(ordered, total_gt)
        if (
            int(anchor_metrics[key]["matched_tp"])
            != sum(ordered_anchor_matches)
        ):
            raise AssertionError("oracle anchor prefix disagrees with AP evaluator")
        metric["oracle_novel_candidate_tp"] = int(novel_matches)
        output[key] = metric
    return output


def _partition_report(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    anchor_rows = _policy_rows(rows, None)
    anchor_metrics = _metrics(anchor_rows, "boxes")
    if any(
        np.any(row["anchor_scores"] <= FIXED_LOW_SCORE)
        for row in rows
    ):
        raise ValueError("fixed low score does not rank below every anchor")
    fixed_scores = [
        np.full(len(row["candidate_boxes"]), FIXED_LOW_SCORE, dtype=np.float64)
        for row in rows
    ]
    fixed_metrics = _metrics(_policy_rows(rows, fixed_scores), "boxes")
    c1_scores = _c1_rank_scores(rows)
    c1_metrics = _metrics(_policy_rows(rows, c1_scores), "boxes")
    oracle_metrics = _gt_oracle_metrics(rows, anchor_metrics)
    candidate_count = sum(len(row["candidate_boxes"]) for row in rows)

    def route(
        metrics: dict[str, dict[str, float | int]], score_ordering: str
    ) -> dict[str, Any]:
        return {
            "candidate_count": candidate_count,
            "score_ordering": score_ordering,
            "metrics": _with_ap_percent(metrics),
            "delta_vs_anchor": _delta(metrics, anchor_metrics),
        }

    return {
        "scene_count": len(rows),
        "scene_ids": [str(row["scene_id"]) for row in rows],
        "ground_truth_count": sum(len(row["gt"]) for row in rows),
        "active_anchor_count": sum(len(row["anchor_boxes"]) for row in rows),
        "route_candidate_count": candidate_count,
        "routes": {
            "anchor": route(anchor_metrics, "exported_anchor_score_descending_stable"),
            "append_fixed_low": route(
                fixed_metrics,
                f"all anchors first; candidates tied at fixed score {FIXED_LOW_SCORE:g}",
            ),
            "append_c1_track_rank": route(
                c1_metrics,
                "all anchors first; candidates ordered by c1_track_score descending",
            ),
            "gt_oracle_upper_bound": route(
                oracle_metrics,
                "all anchors fixed first; threshold-specific maximum-cardinality "
                "candidate TPs before candidate FPs",
            ),
        },
    }


def audit(args: argparse.Namespace) -> dict[str, Any]:
    scene_list = args.scene_list.resolve()
    exclude_scene_list = args.exclude_scene_list.resolve()
    c2_export_path = args.c2_export_report.resolve()
    stable_hashes = {
        "scene_list": sha256_file(scene_list),
        "exclude_scene_list": sha256_file(exclude_scene_list),
        "c2_export_report": sha256_file(c2_export_path),
    }
    scenes = read_scene_list(scene_list)
    excluded = read_scene_list(exclude_scene_list)
    partition_ids = _partition_scene_ids(scenes, excluded)
    c2_export = json.loads(c2_export_path.read_text(encoding="utf-8"))
    if c2_export.get("schema") != C2_EXPORT_SCHEMA:
        raise ValueError("unsupported C2 export report")
    if (
        not c2_export.get("observer_only")
        or c2_export.get("mutation_enabled")
        or int(c2_export.get("applied_count", -1)) != 0
    ):
        raise ValueError("C2 export violates observer-only contract")
    if c2_export.get("ground_truth_access") or c2_export.get("clip_access"):
        raise ValueError("C2 export improperly accessed GT or CLIP")
    if c2_export.get("teacher_labels_used_for_gate"):
        raise ValueError("C2 export used teacher labels for gating")
    if c2_export.get("code_sha256") != c2_code_hash():
        raise ValueError("C2 code differs from immutable export")
    if int(c2_export.get("scene_count", -1)) != len(scenes):
        raise ValueError("C2 export scene count mismatch")
    export_rows = {str(row["scene_id"]): row for row in c2_export["scenes"]}
    if set(export_rows) != set(scenes) or len(export_rows) != len(scenes):
        raise ValueError("C2 export scene identities mismatch")

    active_root = args.active_prediction_root.resolve()
    before = _tree_snapshot(active_root, scenes)
    if before["tree_sha256"] != c2_export["frozen_active_before"]["tree_sha256"]:
        raise ValueError("R3 active tree differs from C2 export anchor")

    rows: list[dict[str, Any]] = []
    input_hashes: list[tuple[Path, str]] = []
    gate_index = GATE_NAMES.index("mask2_depth")
    for scene_id in scenes:
        c2_path = sidecar_path(args.c2_cache_root.resolve(), scene_id, args.prefix_id)
        c2_sha = sha256_file(c2_path)
        if c2_sha != export_rows[scene_id]["sidecar_sha256"]:
            raise ValueError(f"{scene_id}: C2 sidecar hash mismatch")
        c2 = load_sidecar(c2_path)
        parent_path = tr3d_residual_cache_path(
            args.parent_cache_root.resolve(), scene_id, args.prefix_id
        )
        parent_sha = sha256_file(parent_path)
        if parent_sha != c2.parent_cache_sha256:
            raise ValueError(f"{scene_id}: parent cache hash mismatch")
        with np.load(parent_path, allow_pickle=False) as raw:
            checkpoint_sha = str(np.asarray(raw["checkpoint_sha256"]).item())
            config_sha = str(np.asarray(raw["config_sha256"]).item())
        parent = load_tr3d_residual_cache(
            parent_path,
            expected_scene_id=scene_id,
            expected_prefix_id=args.prefix_id,
            expected_checkpoint_sha256=checkpoint_sha,
            expected_config_sha256=config_sha,
        )
        if not np.array_equal(parent.proposal_ids[c2.parent_rows], c2.proposal_ids):
            raise ValueError(f"{scene_id}: C2/parent identity mismatch")
        anchor_path = active_root / f"{scene_id}_boxes.pkl"
        if sha256_file(anchor_path) != c2.anchor_prediction_sha256:
            raise ValueError(f"{scene_id}: active prediction hash mismatch")
        alignment = _alignment(args.scans_root.resolve(), scene_id)
        anchor_corners, anchor_scores = _prediction(anchor_path)
        candidate_corners = parent.corners_world[c2.parent_rows]
        route_mask = (c2.source_ranks <= 5) & c2.observation.gate_mask[:, gate_index]
        c1_scores = np.asarray(c2.c1_track_scores[route_mask], dtype=np.float64)
        if not np.isfinite(c1_scores).all():
            raise ValueError(f"{scene_id}: non-finite C1 track score")
        rows.append(
            {
                "scene_id": scene_id,
                "anchor_boxes": _minmax(anchor_corners, alignment),
                "anchor_scores": anchor_scores,
                "candidate_boxes": _minmax(candidate_corners[route_mask], alignment),
                "c1_track_scores": c1_scores,
                "gt": _ground_truth(args.gt_root.resolve() / f"{scene_id}_bbox.npy"),
            }
        )
        input_hashes.extend(((c2_path, c2_sha), (parent_path, parent_sha)))

    by_id = {str(row["scene_id"]): row for row in rows}
    partitions = {
        name: _partition_report([by_id[scene_id] for scene_id in scene_ids])
        for name, scene_ids in partition_ids.items()
    }
    after = _tree_snapshot(active_root, scenes)
    if before != after:
        raise RuntimeError("R3 active prediction tree changed during C3 audit")
    for path, expected in input_hashes:
        if sha256_file(path) != expected:
            raise RuntimeError(f"input changed during C3 audit: {path}")
    for name, path in (
        ("scene_list", scene_list),
        ("exclude_scene_list", exclude_scene_list),
        ("c2_export_report", c2_export_path),
    ):
        if sha256_file(path) != stable_hashes[name]:
            raise RuntimeError(f"{name} changed during C3 audit")

    return {
        "schema": REPORT_SCHEMA,
        "route": ROUTE_NAME,
        "ground_truth_only_offline_audit": True,
        "inference_modules_ground_truth_access": False,
        "observer_only": True,
        "prediction_artifacts_mutated": False,
        "active_materialization_authorized": False,
        "exploratory": True,
        "post_hoc": True,
        "scene_list": str(scene_list),
        "exclude_scene_list": str(exclude_scene_list),
        "c2_export_report": str(c2_export_path),
        "input_sha256": stable_hashes,
        "decision_partition": "heldout",
        "partitions": partitions,
        "frozen_active_before": before,
        "frozen_active_after": after,
        "protocol": {
            "class_agnostic": True,
            "iou_thresholds": list(THRESHOLDS),
            "iou_comparison": "strictly greater than threshold",
            "ap": "VOC precision-envelope integration reused from R4 audit",
            "candidate_route": "source_rank<=5 AND mask2_depth",
            "fixed_low_score": FIXED_LOW_SCORE,
            "anchor_order_preserved_for_every_append_route": True,
            "gt_oracle_is_threshold_specific": True,
            "gt_oracle_is_not_deployable": True,
        },
        "decision": {
            "status": "EXPLORATORY_POST_HOC_DO_NOT_ACTIVATE",
            "active_materialization_authorized": False,
            "validation_tuning_authorized": False,
            "meaning": (
                "These counterfactuals measure route headroom only and can never "
                "authorize writing or activating C2/C3 predictions."
            ),
        },
    }


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--scene-list", type=Path, required=True)
    value.add_argument("--exclude-scene-list", type=Path, required=True)
    value.add_argument("--c2-export-report", type=Path, required=True)
    value.add_argument("--c2-cache-root", type=Path, required=True)
    value.add_argument("--parent-cache-root", type=Path, required=True)
    value.add_argument("--active-prediction-root", type=Path, required=True)
    value.add_argument("--scans-root", type=Path, required=True)
    value.add_argument("--gt-root", type=Path, required=True)
    value.add_argument("--prefix-id", default="p100")
    value.add_argument("--report", type=Path, required=True)
    return value


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    report = audit(args)
    _write_json_create_only(args.report.resolve(), report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
