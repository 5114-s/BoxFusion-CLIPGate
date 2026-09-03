#!/usr/bin/env python3
"""Ground-truth-only audit of immutable C2 Mask-RGBD sidecars."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Any

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
from tools.audit_tr3d_residual_observer import (  # noqa: E402
    _alignment,
    _gt_boxes,
    _load_b6,
    _minmax,
    _transform,
    maximum_cardinality,
    pairwise_iou,
)
from tools.run_tr3d_c1_track_observer import _tree_snapshot  # noqa: E402
from tools.run_tr3d_c2_maskrgbd_observer import (  # noqa: E402
    REPORT_SCHEMA as EXPORT_SCHEMA,
    _code_hash,
)
from tools.tr3d_data import read_scene_list  # noqa: E402


REPORT_SCHEMA = "boxfusion.tr3d_c2_maskrgbd_gt_audit.v1"
IOU_THRESHOLDS = (0.15, 0.25, 0.50)
ROUTE_NAMES = (
    "source_top3",
    "source_top5",
    "source_top10",
    *GATE_NAMES,
    "top5_mask1",
    "top5_mask2_depth",
)


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
        raise FileExistsError(f"immutable C2 audit exists: {path}") from error
    finally:
        if temporary_name is not None:
            try:
                os.unlink(temporary_name)
            except FileNotFoundError:
                pass


def _empty() -> dict[str, Any]:
    keys = {f"{threshold:.2f}": 0 for threshold in IOU_THRESHOLDS}
    return {
        "candidate_count": 0,
        "anchor_matches": dict(keys),
        "union_matches": dict(keys),
        "direct_hits": dict(keys),
        "positive_scenes": dict(keys),
        "duplicate_count": dict(keys),
    }


def _self_duplicates(boxes: np.ndarray, scores: np.ndarray, threshold: float) -> int:
    if len(boxes) < 2:
        return 0
    iou = pairwise_iou(boxes, boxes)
    order = np.argsort(-scores, kind="stable")
    kept: list[int] = []
    duplicates = 0
    for row in order:
        if kept and float(iou[int(row), kept].max(initial=0.0)) > threshold:
            duplicates += 1
        else:
            kept.append(int(row))
    return duplicates


def _add(
    raw: dict[str, Any],
    anchor_vs_gt: np.ndarray,
    candidate_vs_gt: np.ndarray,
    candidate_boxes: np.ndarray,
    candidate_scores: np.ndarray,
) -> None:
    raw["candidate_count"] += len(candidate_boxes)
    for threshold in IOU_THRESHOLDS:
        key = f"{threshold:.2f}"
        anchor = maximum_cardinality(anchor_vs_gt, threshold)
        union = maximum_cardinality(
            np.concatenate((anchor_vs_gt, candidate_vs_gt), axis=0), threshold
        )
        raw["anchor_matches"][key] += anchor
        raw["union_matches"][key] += union
        raw["direct_hits"][key] += int(
            np.count_nonzero(candidate_vs_gt.max(axis=1) > threshold)
            if candidate_vs_gt.shape[1] else 0
        )
        raw["positive_scenes"][key] += int(union > anchor)
        raw["duplicate_count"][key] += _self_duplicates(
            candidate_boxes, candidate_scores, threshold
        )


def _finalize(raw: dict[str, Any], total_gt: int) -> dict[str, Any]:
    count = int(raw["candidate_count"])
    thresholds = {}
    for threshold in IOU_THRESHOLDS:
        key = f"{threshold:.2f}"
        anchor = int(raw["anchor_matches"][key])
        union = int(raw["union_matches"][key])
        direct = int(raw["direct_hits"][key])
        duplicates = int(raw["duplicate_count"][key])
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
            "self_duplicate_count": duplicates,
            "self_duplicate_rate": duplicates / max(count, 1),
        }
    return {"candidate_count": count, "thresholds": thresholds}


def _decision(routes: dict[str, Any]) -> dict[str, Any]:
    """Evaluate the pre-registered C2 gate for one scene partition."""

    source = routes["source_top10"]
    primary = routes["mask2_depth"]
    source25 = source["thresholds"]["0.25"]
    primary25 = primary["thresholds"]["0.25"]
    primary50 = primary["thresholds"]["0.50"]
    checks = {
        "at_least_5_candidates": primary["candidate_count"] >= 5,
        "ap25_hit_precision_at_least_50pct": (
            primary25["independent_gt_hit_precision"] >= 0.50
        ),
        "ap25_precision_improves_source_by_10pt": (
            primary25["independent_gt_hit_precision"]
            >= source25["independent_gt_hit_precision"] + 0.10
        ),
        "at_least_3_novel_ap25_tp": primary25["novel_oracle_tp"] >= 3,
        "ap50_hit_precision_at_least_25pct": (
            primary50["independent_gt_hit_precision"] >= 0.25
        ),
    }
    return {
        "primary_route": "mask2_depth",
        "checks": checks,
        "pass": all(checks.values()),
        "meaning": (
            "PASS only authorizes a separate source-aware C3 shadow materializer; "
            "it does not authorize C2 active output or a validation-tuned gate."
        ),
    }


def _partition_scene_ids(
    scenes: list[str], excluded_scenes: list[str] | None
) -> dict[str, list[str]]:
    """Return the audited all partition and optional frozen held-out90 split."""

    all_scenes = list(scenes)
    if excluded_scenes is None:
        return {"all": all_scenes}
    if len(all_scenes) != 100:
        raise ValueError(
            "--exclude-scene-list requires exactly 100 scenes in --scene-list"
        )
    if len(excluded_scenes) != 10:
        raise ValueError("--exclude-scene-list must contain exactly 10 scenes")
    missing = sorted(set(excluded_scenes) - set(all_scenes))
    if missing:
        raise ValueError(
            "excluded scenes are absent from --scene-list: " + ", ".join(missing)
        )
    excluded = set(excluded_scenes)
    heldout = [scene_id for scene_id in all_scenes if scene_id not in excluded]
    if len(heldout) != 90:
        raise ValueError("main/excluded scene difference must contain 90 scenes")
    return {"all": all_scenes, "heldout": heldout}


def _finalize_partition(
    totals: dict[str, dict[str, Any]],
    *,
    scene_ids: list[str],
    total_gt: int,
    total_anchors: int,
) -> dict[str, Any]:
    routes = {
        name: _finalize(totals[name], total_gt) for name in ROUTE_NAMES
    }
    return {
        "scene_count": len(scene_ids),
        "scene_ids": list(scene_ids),
        "ground_truth_count": int(total_gt),
        "active_anchor_count": int(total_anchors),
        "routes": routes,
        "decision": _decision(routes),
    }


def audit(args: argparse.Namespace) -> dict[str, Any]:
    export = json.loads(args.export_report.read_text(encoding="utf-8"))
    if export.get("schema") != EXPORT_SCHEMA:
        raise ValueError("unsupported C2 export report")
    if not export.get("observer_only") or export.get("mutation_enabled"):
        raise ValueError("C2 export violates observer contract")
    if int(export.get("applied_count", -1)) != 0:
        raise ValueError("C2 export applied_count must be zero")
    if export.get("ground_truth_access") or export.get("clip_access"):
        raise ValueError("C2 export improperly accessed GT or CLIP")
    if export.get("teacher_labels_used_for_gate"):
        raise ValueError("C2 export used teacher labels for gating")
    if export.get("code_sha256") != _code_hash():
        raise ValueError("C2 code differs from immutable export")
    scenes = read_scene_list(args.scene_list.resolve())
    exclude_scene_list = getattr(args, "exclude_scene_list", None)
    exclude_scene_list_sha256 = (
        sha256_file(exclude_scene_list.resolve())
        if exclude_scene_list is not None
        else None
    )
    excluded_scenes = (
        read_scene_list(exclude_scene_list.resolve())
        if exclude_scene_list is not None
        else None
    )
    partition_scene_ids = _partition_scene_ids(scenes, excluded_scenes)
    partition_membership = {
        name: set(scene_ids) for name, scene_ids in partition_scene_ids.items()
    }
    if export.get("scene_count") != len(scenes):
        raise ValueError("scene count mismatch")
    active_root = args.active_prediction_root.resolve()
    before = _tree_snapshot(active_root, scenes)
    if before["tree_sha256"] != export["frozen_active_before"]["tree_sha256"]:
        raise ValueError("frozen active tree differs from C2 export")
    export_rows = {str(row["scene_id"]): row for row in export["scenes"]}

    partition_totals = {
        partition: {name: _empty() for name in ROUTE_NAMES}
        for partition in partition_scene_ids
    }
    partition_counts = {
        partition: {"ground_truth": 0, "active_anchors": 0}
        for partition in partition_scene_ids
    }
    per_scene: list[dict[str, Any]] = []
    label_counts: dict[str, int] = {}
    for scene_id in scenes:
        path = sidecar_path(args.c2_cache_root.resolve(), scene_id, args.prefix_id)
        if sha256_file(path) != export_rows[scene_id]["sidecar_sha256"]:
            raise ValueError(f"{scene_id}: C2 sidecar hash mismatch")
        c2 = load_sidecar(path)
        parent_path = tr3d_residual_cache_path(
            args.parent_cache_root.resolve(), scene_id, args.prefix_id
        )
        if sha256_file(parent_path) != c2.parent_cache_sha256:
            raise ValueError(f"{scene_id}: parent cache hash mismatch")
        with np.load(parent_path, allow_pickle=False) as raw:
            checkpoint_sha = str(np.asarray(raw["checkpoint_sha256"]).item())
            config_sha = str(np.asarray(raw["config_sha256"]).item())
        parent = load_tr3d_residual_cache(
            parent_path, expected_scene_id=scene_id, expected_prefix_id=args.prefix_id,
            expected_checkpoint_sha256=checkpoint_sha, expected_config_sha256=config_sha,
        )
        if not np.array_equal(parent.proposal_ids[c2.parent_rows], c2.proposal_ids):
            raise ValueError(f"{scene_id}: C2/parent identity mismatch")
        anchor_path = active_root / f"{scene_id}_boxes.pkl"
        if sha256_file(anchor_path) != c2.anchor_prediction_sha256:
            raise ValueError(f"{scene_id}: anchor prediction hash mismatch")
        alignment = _alignment(args.scans_root.resolve(), scene_id)
        candidate_boxes = _minmax(
            _transform(parent.corners_world[c2.parent_rows], alignment)
        )
        anchor_corners, _ = _load_b6(anchor_path)
        anchor_boxes = _minmax(_transform(anchor_corners, alignment))
        targets = _gt_boxes(args.gt_root.resolve() / f"{scene_id}_bbox.npy")
        anchor_vs_gt = pairwise_iou(anchor_boxes, targets)
        candidate_vs_gt = pairwise_iou(candidate_boxes, targets)
        evidence_scores = c2.observation.max_evidence_score
        masks: dict[str, np.ndarray] = {
            "source_top3": c2.source_ranks <= 3,
            "source_top5": c2.source_ranks <= 5,
            "source_top10": c2.source_ranks <= 10,
        }
        masks.update(
            {
                name: c2.observation.gate_mask[:, index]
                for index, name in enumerate(GATE_NAMES)
            }
        )
        masks["top5_mask1"] = masks["source_top5"] & masks["mask1"]
        masks["top5_mask2_depth"] = masks["source_top5"] & masks["mask2_depth"]
        scene_counts = {}
        for name, mask in masks.items():
            for partition, membership in partition_membership.items():
                if scene_id in membership:
                    _add(
                        partition_totals[partition][name],
                        anchor_vs_gt,
                        candidate_vs_gt[mask],
                        candidate_boxes[mask],
                        evidence_scores[mask],
                    )
            scene_counts[name] = int(np.count_nonzero(mask))
        for label in c2.observation.best_mask_label[c2.observation.view_matched].tolist():
            normalized = str(label) if str(label) else "<missing>"
            label_counts[normalized] = label_counts.get(normalized, 0) + 1
        per_scene.append(
            {
                "scene_id": scene_id,
                "ground_truth": len(targets),
                "active_anchors": len(anchor_boxes),
                "source_candidates": c2.candidate_count,
                "routes": scene_counts,
            }
        )
        for partition, membership in partition_membership.items():
            if scene_id in membership:
                partition_counts[partition]["ground_truth"] += len(targets)
                partition_counts[partition]["active_anchors"] += len(anchor_boxes)

    partitions = {
        partition: _finalize_partition(
            partition_totals[partition],
            scene_ids=scene_ids,
            total_gt=partition_counts[partition]["ground_truth"],
            total_anchors=partition_counts[partition]["active_anchors"],
        )
        for partition, scene_ids in partition_scene_ids.items()
    }
    decision_partition = "heldout" if "heldout" in partitions else "all"
    all_partition = partitions["all"]
    after = _tree_snapshot(active_root, scenes)
    if before != after:
        raise RuntimeError("frozen active prediction tree changed during C2 audit")
    if (
        exclude_scene_list is not None
        and sha256_file(exclude_scene_list.resolve()) != exclude_scene_list_sha256
    ):
        raise RuntimeError("exclude scene list changed during C2 audit")
    report = {
        "schema": REPORT_SCHEMA,
        "observer_only": True,
        "mutation_enabled": False,
        "applied_count": 0,
        "export_report": str(args.export_report.resolve()),
        "scene_list": str(args.scene_list.resolve()),
        "scene_count": len(scenes),
        "ground_truth_count": all_partition["ground_truth_count"],
        "active_anchor_count": all_partition["active_anchor_count"],
        # Keep the legacy top-level route table tied to the complete audited
        # scene list.  Only the authorization decision switches to heldout.
        "routes": all_partition["routes"],
        "teacher_label_diagnostics_only": dict(sorted(label_counts.items())),
        "decision_partition": decision_partition,
        "decision": partitions[decision_partition]["decision"],
        "partitions": partitions,
        "frozen_active_before": before,
        "frozen_active_after": after,
        "per_scene": per_scene,
    }
    if exclude_scene_list is not None:
        report["exclude_scene_list"] = str(exclude_scene_list.resolve())
        report["exclude_scene_list_sha256"] = exclude_scene_list_sha256
        report["excluded_scene_count"] = len(excluded_scenes or ())
        report["excluded_scene_ids"] = list(excluded_scenes or ())
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-list", type=Path, required=True)
    parser.add_argument(
        "--exclude-scene-list",
        type=Path,
        help=(
            "Frozen 10-scene development list to exclude from a 100-scene "
            "audit; reports all100 and authoritative heldout90 partitions"
        ),
    )
    parser.add_argument("--export-report", type=Path, required=True)
    parser.add_argument("--c2-cache-root", type=Path, required=True)
    parser.add_argument("--parent-cache-root", type=Path, required=True)
    parser.add_argument("--active-prediction-root", type=Path, required=True)
    parser.add_argument("--scans-root", type=Path, required=True)
    parser.add_argument("--gt-root", type=Path, required=True)
    parser.add_argument("--prefix-id", default="p100")
    parser.add_argument("--report", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = audit(args)
    _write_json_create_only(args.report.resolve(), report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
