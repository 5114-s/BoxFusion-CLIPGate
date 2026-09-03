#!/usr/bin/env python3
"""Audit R2b feature evidence against score-only at fixed budgets.

Ground truth is read only by this audit, never by either R2 observer.  The
fixed validation subset may reject a route but may not select a deployment
threshold or fit a score calibrator.
"""

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

from boxfusion.frozen_anchor_manifest import verify_frozen_anchor_manifest  # noqa: E402
from boxfusion.tr3d_r2_cache import load_tr3d_r2_cache, tr3d_r2_cache_path  # noqa: E402
from boxfusion.tr3d_r2_provenance import (  # noqa: E402
    canonical_json_sha256,
    frame_artifact_tree,
    load_prefix_manifest,
    sha256_file,
)
from boxfusion.tr3d_r2b_cache import (  # noqa: E402
    load_tr3d_r2b_cache,
    tr3d_r2b_cache_path,
)
from boxfusion.tr3d_residual_cache import tr3d_residual_cache_path  # noqa: E402
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
from tools.run_tr3d_r2_observer import (  # noqa: E402
    _code_hash as current_r2a_code_hash,
    _load_bound_parent,
)
from tools.run_tr3d_r2b_feature_observer import (  # noqa: E402
    REPORT_SCHEMA as R2B_EXPORT_SCHEMA,
    _feature_code_hash,
)
from tools.tr3d_data import discover_frame_bundle, read_scene_list  # noqa: E402


REPORT_SCHEMA = "boxfusion.tr3d_r2b_feature_audit.v1"
R2A_EXPORT_SCHEMA = "boxfusion.tr3d_r2a_observer_export.v1"
RESIDUAL_ANCHOR_IOU = 0.15
IOU_THRESHOLDS = (0.15, 0.25, 0.50)
FIXED_BUDGETS = (25, 50, 100, 200)


def _snapshot(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "anchor_name": payload["anchor_name"],
        "prediction_tree_sha256": payload["prediction_tree_sha256"],
        "artifact_tree_sha256": payload["artifact_tree_sha256"],
        "scene_list_sha256": payload["scene_list_sha256"],
    }


def binary_auc_ap(labels: object, scores: object) -> dict[str, float | int]:
    """AUC and non-interpolated AP with deterministic stable tie handling."""

    y = np.asarray(labels)
    s = np.asarray(scores, dtype=np.float64)
    if y.ndim != 1 or s.shape != y.shape or y.dtype != np.bool_ or not np.isfinite(s).all():
        raise ValueError("labels/scores must be boolean [N] and finite [N]")
    positives = s[y]
    negatives = s[~y]
    if not len(positives) or not len(negatives):
        raise ValueError("AUC requires both positive and negative rows")
    comparisons = positives[:, None] - negatives[None, :]
    auc = float(
        (np.count_nonzero(comparisons > 0) + 0.5 * np.count_nonzero(comparisons == 0))
        / comparisons.size
    )
    order = np.argsort(-s, kind="stable")
    ranked = y[order]
    precision = np.cumsum(ranked, dtype=np.int64) / np.arange(1, len(ranked) + 1)
    ap = float(precision[ranked].mean())
    return {
        "rows": int(len(y)),
        "positives": int(np.count_nonzero(y)),
        "negatives": int(np.count_nonzero(~y)),
        "auc": auc,
        "average_precision": ap,
    }


def _quantiles(values: object) -> dict[str, float | int | None]:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if not len(array):
        return {"count": 0, "q10": None, "q50": None, "q90": None}
    q10, q50, q90 = np.quantile(array, (0.10, 0.50, 0.90))
    return {
        "count": int(len(array)),
        "q10": float(q10),
        "q50": float(q50),
        "q90": float(q90),
    }


def _load_json(path: Path, schema: str, label: str) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != schema:
        raise ValueError(f"unsupported {label} schema")
    if not payload.get("observer_only") or payload.get("mutation_enabled"):
        raise ValueError(f"{label} violates observer-only contract")
    if int(payload.get("applied_count", -1)) != 0:
        raise ValueError(f"{label} applied_count must be zero")
    if payload.get("ground_truth_access"):
        raise ValueError(f"{label} improperly accessed ground truth")
    return payload


def _verify_reports(
    args: argparse.Namespace, scenes: list[str]
) -> tuple[dict[str, Any], dict[str, Any]]:
    r2a = _load_json(args.r2a_export_report.resolve(), R2A_EXPORT_SCHEMA, "R2a export")
    r2b = _load_json(args.r2b_export_report.resolve(), R2B_EXPORT_SCHEMA, "R2b export")
    if r2a.get("clip_enabled") or r2b.get("clip_access"):
        raise ValueError("observer export accessed CLIP")
    if not r2b.get("clip_semantics_unchanged"):
        raise ValueError("R2b did not preserve the CLIP semantic contract")
    shared_paths = {
        "parent_cache_root": args.parent_cache_root,
        "prefix_manifest": args.prefix_manifest,
        "frames_root": args.frames_root,
        "scene_list": args.scene_list,
    }
    for name, expected in shared_paths.items():
        if Path(str(r2a.get(name, ""))).resolve() != expected.resolve():
            raise ValueError(f"R2a {name} mismatch")
        if Path(str(r2b.get(name, ""))).resolve() != expected.resolve():
            raise ValueError(f"R2b {name} mismatch")
    extra_paths = {
        "r2_cache_root": (r2a, args.r2a_cache_root),
        "r2a_cache_root": (r2b, args.r2a_cache_root),
        "r2b_cache_root": (r2b, args.r2b_cache_root),
        "r2a_export_report": (r2b, args.r2a_export_report),
    }
    for name, (payload, expected) in extra_paths.items():
        if Path(str(payload.get(name, ""))).resolve() != expected.resolve():
            raise ValueError(f"export {name} mismatch")
    if r2a.get("prefix_id") != args.prefix_id or r2b.get("prefix_id") != args.prefix_id:
        raise ValueError("export prefix id mismatch")
    for label, payload in (("R2a", r2a), ("R2b", r2b)):
        ordered = [str(row.get("scene_id")) for row in payload.get("scenes", [])]
        if ordered != scenes or int(payload.get("scene_count", -1)) != len(scenes):
            raise ValueError(f"{label} ordered scene set mismatch")
    if canonical_json_sha256(r2a["r2_config"]) != r2a.get("r2_config_sha256"):
        raise ValueError("R2a config hash mismatch")
    if canonical_json_sha256(r2b["feature_config"]) != r2b.get("feature_config_sha256"):
        raise ValueError("R2b feature config hash mismatch")
    if current_r2a_code_hash() != r2a.get("r2_code_sha256"):
        raise ValueError("current R2a code differs from export")
    if _feature_code_hash(args) != r2b.get("feature_code_sha256"):
        raise ValueError("current R2b feature code differs from export")
    if sha256_file(args.r2a_export_report.resolve()) != r2b.get("input_hashes", {}).get("r2a_export_report_sha256"):
        raise ValueError("R2b parent export report bytes changed")
    return r2a, r2b


def _rank(records: list[dict[str, Any]], key: str, budget: int) -> set[tuple[str, int]]:
    ordered = sorted(
        records,
        key=lambda row: (-float(row[key]), row["scene_id"], int(row["proposal_id"])),
    )
    return {
        (str(row["scene_id"]), int(row["proposal_id"]))
        for row in ordered[: min(budget, len(ordered))]
    }


def _budget_metrics(
    scene_data: list[dict[str, Any]], selected: set[tuple[str, int]]
) -> dict[str, Any]:
    count = independent_tp50 = 0
    tp_scenes: set[str] = set()
    anchor_matches = {threshold: 0 for threshold in IOU_THRESHOLDS}
    union_matches = {threshold: 0 for threshold in IOU_THRESHOLDS}
    gt_count = 0
    for scene in scene_data:
        keys = [
            (scene["scene_id"], int(value))
            for value in scene["proposal_ids"]
        ]
        mask = np.asarray([key in selected for key in keys], dtype=np.bool_)
        count += int(np.count_nonzero(mask))
        chosen = scene["candidate_vs_gt"][mask]
        hit = (
            chosen.max(axis=1) > 0.50
            if chosen.shape[1]
            else np.zeros(len(chosen), dtype=np.bool_)
        )
        independent_tp50 += int(np.count_nonzero(hit))
        if np.any(hit):
            tp_scenes.add(scene["scene_id"])
        gt_count += int(scene["candidate_vs_gt"].shape[1])
        for threshold in IOU_THRESHOLDS:
            anchor_matches[threshold] += maximum_cardinality(
                scene["anchor_vs_gt"], threshold
            )
            union = np.concatenate((scene["anchor_vs_gt"], chosen), axis=0)
            union_matches[threshold] += maximum_cardinality(union, threshold)
    thresholds = {}
    for threshold in IOU_THRESHOLDS:
        novel = union_matches[threshold] - anchor_matches[threshold]
        thresholds[f"{threshold:.2f}"] = {
            "anchor_matches": anchor_matches[threshold],
            "union_matches": union_matches[threshold],
            "novel_oracle_tp": novel,
            "delta_oracle_recall": novel / gt_count if gt_count else 0.0,
        }
    return {
        "candidate_count": count,
        "independent_tp50": independent_tp50,
        "independent_precision50_upper_bound": independent_tp50 / count if count else 0.0,
        "independent_tp50_scene_coverage": len(tp_scenes),
        "thresholds": thresholds,
    }


def _route_pass(score: dict[str, Any], joint: dict[str, Any]) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    same_novel = (
        joint["thresholds"]["0.50"]["novel_oracle_tp"]
        == score["thresholds"]["0.50"]["novel_oracle_tp"]
    )
    precision_gain = (
        joint["independent_precision50_upper_bound"]
        - score["independent_precision50_upper_bound"]
    )
    if same_novel and precision_gain >= 0.05 - 1e-12:
        reasons.append("same novel TP50 with >=5pp independent precision gain")
    novel_gain = (
        joint["thresholds"]["0.50"]["novel_oracle_tp"]
        - score["thresholds"]["0.50"]["novel_oracle_tp"]
    )
    if novel_gain >= 2 and joint["independent_tp50_scene_coverage"] >= score["independent_tp50_scene_coverage"]:
        reasons.append(">=2 novel TP50 with nondecreasing scene coverage")
    return bool(reasons), reasons


def audit(args: argparse.Namespace) -> dict[str, Any]:
    before = verify_frozen_anchor_manifest(args.frozen_manifest.resolve())
    before_snapshot = _snapshot(before)
    scenes = read_scene_list(args.scene_list.resolve())
    r2a_report, r2b_report = _verify_reports(args, scenes)
    manifest = load_prefix_manifest(args.prefix_manifest.resolve(), prefix_id=args.prefix_id)
    r2a_config_sha = str(r2a_report["r2_config_sha256"])
    r2a_code_sha = str(r2a_report["r2_code_sha256"])
    feature_checkpoint_sha = str(r2b_report["feature_checkpoint_sha256"])
    feature_config_sha = str(r2b_report["feature_config_sha256"])
    feature_code_sha = str(r2b_report["feature_code_sha256"])
    r2b_rows = {str(row["scene_id"]): row for row in r2b_report["scenes"]}
    frozen_root = Path(before["reference_result_root"]).resolve()

    records: list[dict[str, Any]] = []
    scene_data: list[dict[str, Any]] = []
    feature_positive: list[np.ndarray] = []
    feature_negative: list[np.ndarray] = []
    total_gt = total_anchor = total_candidate = total_residual = 0
    for scene_id in scenes:
        row = manifest[scene_id]
        used_ids = [int(value) for value in row["used_frame_ids"]]
        bundle = discover_frame_bundle(args.frames_root.resolve(), scene_id)
        frame_tree_sha, _ = frame_artifact_tree(row, bundle)
        manifest_sha = canonical_json_sha256(row)
        parent_path = tr3d_residual_cache_path(args.parent_cache_root.resolve(), scene_id, args.prefix_id)
        parent = _load_bound_parent(
            parent_path,
            row,
            args.prefix_manifest.resolve(),
            expected_scene_id=scene_id,
            expected_prefix_id=args.prefix_id,
            expected_checkpoint_sha256=args.expected_parent_checkpoint_sha256,
            expected_config_sha256=args.expected_parent_config_sha256,
        )
        r2a_path = tr3d_r2_cache_path(args.r2a_cache_root.resolve(), scene_id, args.prefix_id)
        r2a = load_tr3d_r2_cache(
            r2a_path,
            parent_cache_path=parent_path,
            expected_prefix_manifest_row_sha256=manifest_sha,
            expected_frame_artifact_tree_sha256=frame_tree_sha,
            expected_r2_config_sha256=r2a_config_sha,
            expected_r2_code_sha256=r2a_code_sha,
            expected_scene_id=scene_id,
            expected_prefix_id=args.prefix_id,
            expected_prefix_fraction=float(row["fraction"]),
            expected_allowed_frame_ids=used_ids,
        )
        r2b_path = tr3d_r2b_cache_path(args.r2b_cache_root.resolve(), scene_id, args.prefix_id)
        if sha256_file(r2b_path) != r2b_rows[scene_id]["r2b_sidecar_sha256"]:
            raise ValueError(f"{scene_id}: R2b sidecar bytes changed after export")
        r2b = load_tr3d_r2b_cache(
            r2b_path,
            parent_r2a_cache_path=r2a_path,
            parent_tr3d_cache_path=parent_path,
            expected_parent_prefix_manifest_row_sha256=manifest_sha,
            expected_parent_frame_artifact_tree_sha256=frame_tree_sha,
            expected_parent_r2_config_sha256=r2a_config_sha,
            expected_parent_r2_code_sha256=r2a_code_sha,
            expected_feature_checkpoint_sha256=feature_checkpoint_sha,
            expected_feature_config_sha256=feature_config_sha,
            expected_feature_code_sha256=feature_code_sha,
            expected_scene_id=scene_id,
            expected_prefix_id=args.prefix_id,
            expected_prefix_fraction=float(row["fraction"]),
        )
        parent_index = {int(value): index for index, value in enumerate(parent.proposal_ids)}
        parent_rows = np.asarray([parent_index[int(value)] for value in r2a.proposal_ids], dtype=np.int64)
        transform = _alignment(args.scans_root.resolve(), scene_id)
        _validate_alignment_provenance(scene_id, transform, parent.aligned_to_unaligned)
        candidate_boxes = _minmax(_transform(parent.corners_world[parent_rows], transform))
        anchor_corners, _ = _load_b6(frozen_root / f"{scene_id}_boxes.pkl")
        anchor_boxes = _minmax(_transform(anchor_corners, transform))
        gt_boxes = _gt_boxes(args.gt_root.resolve() / f"{scene_id}_bbox.npy")
        candidate_vs_gt = pairwise_iou(candidate_boxes, gt_boxes)
        candidate_vs_anchor = pairwise_iou(candidate_boxes, anchor_boxes)
        anchor_vs_gt = pairwise_iou(anchor_boxes, gt_boxes)
        max_gt = candidate_vs_gt.max(axis=1) if candidate_vs_gt.shape[1] else np.zeros(len(candidate_boxes))
        max_anchor = candidate_vs_anchor.max(axis=1) if candidate_vs_anchor.shape[1] else np.zeros(len(candidate_boxes))
        residual = max_anchor <= RESIDUAL_ANCHOR_IOU
        available = r2b.pairwise_cosine_count > 0
        selected_population = residual & available
        evidence = r2a.aggregate_depth_evidence.astype(np.float64)
        depth_quality = np.clip(
            evidence[:, 0] / np.maximum(1.0 - evidence[:, 3], 1e-6),
            0.0,
            1.0,
        )
        feature_score = r2b.pairwise_cosine_mean.astype(np.float64)
        tr3d_score = parent.scores_3d[parent_rows].astype(np.float64)
        feature_quality = (feature_score + 1.0) * 0.5
        joint_score = (
            tr3d_score
            * (0.5 + 0.5 * feature_quality)
            * (0.75 + 0.25 * depth_quality)
        )
        positive = selected_population & (max_gt > 0.50)
        negative = selected_population & (max_gt <= 0.15)
        feature_positive.append(feature_score[positive])
        feature_negative.append(feature_score[negative])
        for index in np.flatnonzero(selected_population):
            records.append({
                "scene_id": scene_id,
                "proposal_id": int(r2a.proposal_ids[index]),
                "max_gt_iou": float(max_gt[index]),
                "tr3d_score": float(tr3d_score[index]),
                "feature_score": float(feature_score[index]),
                "depth_quality": float(depth_quality[index]),
                "joint_score": float(joint_score[index]),
            })
        scene_data.append({
            "scene_id": scene_id,
            "proposal_ids": r2a.proposal_ids,
            "candidate_vs_gt": candidate_vs_gt,
            "anchor_vs_gt": anchor_vs_gt,
        })
        total_gt += len(gt_boxes)
        total_anchor += len(anchor_boxes)
        total_candidate += len(candidate_boxes)
        total_residual += int(np.count_nonzero(residual))

    labels_all = np.asarray([row["max_gt_iou"] > 0.50 for row in records], dtype=np.bool_)
    clear = np.asarray([
        row["max_gt_iou"] > 0.50 or row["max_gt_iou"] <= 0.15 for row in records
    ], dtype=np.bool_)
    labels = labels_all[clear]
    ranking = {}
    for name in ("tr3d_score", "feature_score", "depth_quality", "joint_score"):
        scores = np.asarray([row[name] for row in records], dtype=np.float64)[clear]
        ranking[name] = binary_auc_ap(labels, scores)

    budget_report: dict[str, Any] = {}
    passed_budgets: list[int] = []
    for budget in FIXED_BUDGETS:
        score_selected = _rank(records, "tr3d_score", budget)
        joint_selected = _rank(records, "joint_score", budget)
        score_metrics = _budget_metrics(scene_data, score_selected)
        joint_metrics = _budget_metrics(scene_data, joint_selected)
        passed, reasons = _route_pass(score_metrics, joint_metrics)
        if passed:
            passed_budgets.append(budget)
        budget_report[str(budget)] = {
            "score_only": score_metrics,
            "score_depth_feature": joint_metrics,
            "increment": {
                "independent_tp50": joint_metrics["independent_tp50"] - score_metrics["independent_tp50"],
                "independent_precision50_pp": 100.0 * (
                    joint_metrics["independent_precision50_upper_bound"]
                    - score_metrics["independent_precision50_upper_bound"]
                ),
                "novel_oracle_tp50": (
                    joint_metrics["thresholds"]["0.50"]["novel_oracle_tp"]
                    - score_metrics["thresholds"]["0.50"]["novel_oracle_tp"]
                ),
            },
            "passes_incremental_gate": passed,
            "pass_reasons": reasons,
        }

    after = verify_frozen_anchor_manifest(args.frozen_manifest.resolve())
    after_snapshot = _snapshot(after)
    if before_snapshot != after_snapshot:
        raise RuntimeError("frozen G0 anchor changed during R2b audit")
    return {
        "schema": REPORT_SCHEMA,
        "observer_contract": {
            "observer_only": True,
            "mutation_enabled": False,
            "applied_count": 0,
            "ground_truth_used_by_audit_only": True,
            "clip_semantics_unchanged": True,
            "frozen_anchor_verified_before_and_after": True,
            "before": before_snapshot,
            "after": after_snapshot,
        },
        "purpose": (
            "fixed held-out diagnostic; no validation threshold selection or "
            "active score calibration is permitted"
        ),
        "anchor": {
            "name": before["anchor_name"],
            "metrics_percent": before["anchor_metrics_percent"],
        },
        "counts": {
            "scenes": len(scenes),
            "gt": total_gt,
            "anchor": total_anchor,
            "candidate": total_candidate,
            "residual": total_residual,
            "residual_with_multiview_feature": len(records),
            "clear_positive_iou50": int(np.count_nonzero(labels)),
            "clear_negative_iou15": int(np.count_nonzero(~labels)),
        },
        "fixed_joint_formula": (
            "score*(0.5+0.5*(cosine+1)/2)*"
            "(0.75+0.25*clip(support/(1-invalid),0,1))"
        ),
        "ranking_diagnostics": ranking,
        "feature_separation": {
            "positive_residual_iou50": _quantiles(np.concatenate(feature_positive)),
            "negative_residual_iou15": _quantiles(np.concatenate(feature_negative)),
        },
        "fixed_budget_comparison": budget_report,
        "decision": {
            "passes_pre_registered_incremental_gate": bool(passed_budgets),
            "passing_budgets": passed_budgets,
            "required": (
                ">=5pp precision at equal novel TP50, or >=2 novel TP50 at "
                "nondecreasing scene coverage, relative to score-only"
            ),
            "next_step": (
                "train-only source-aware calibration"
                if passed_budgets
                else "stop R2 active route; keep R2b as weak/negative ablation"
            ),
        },
        "input_hashes": {
            "r2a_export_report_sha256": sha256_file(args.r2a_export_report.resolve()),
            "r2b_export_report_sha256": sha256_file(args.r2b_export_report.resolve()),
            "prefix_manifest_sha256": sha256_file(args.prefix_manifest.resolve()),
            "scene_list_sha256": sha256_file(args.scene_list.resolve()),
        },
    }


def _write_create_only(path: Path, encoded: str) -> None:
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
        raise FileExistsError(f"immutable R2b audit report exists: {path}") from error
    finally:
        if temporary_name is not None:
            try:
                os.unlink(temporary_name)
            except FileNotFoundError:
                pass


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen-manifest", type=Path, required=True)
    parser.add_argument("--parent-cache-root", type=Path, required=True)
    parser.add_argument("--r2a-cache-root", type=Path, required=True)
    parser.add_argument("--r2a-export-report", type=Path, required=True)
    parser.add_argument("--r2b-cache-root", type=Path, required=True)
    parser.add_argument("--r2b-export-report", type=Path, required=True)
    parser.add_argument("--prefix-manifest", type=Path, required=True)
    parser.add_argument("--frames-root", type=Path, required=True)
    parser.add_argument("--scene-list", type=Path, required=True)
    parser.add_argument("--prefix-id", default="p100")
    parser.add_argument("--expected-parent-checkpoint-sha256", required=True)
    parser.add_argument("--expected-parent-config-sha256", required=True)
    parser.add_argument("--official-boxer-root", type=Path, required=True)
    parser.add_argument("--gt-root", type=Path, required=True)
    parser.add_argument("--scans-root", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = audit(args)
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    _write_create_only(args.report.resolve(), encoded)
    print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
