#!/usr/bin/env python3
"""Frozen-B6 identity and class-agnostic TR3D union-oracle audit.

The tool reads B6 predictions and immutable TR3D caches.  It never writes to
either root.  The frozen manifest is verified before and after evaluation, and
the report records both stable score-ordered recall and a maximum-cardinality
geometric oracle.

Pickle is executable serialization; only use trusted local BoxFusion outputs.
"""

from __future__ import annotations

import argparse
import json
import math
import pickle
from pathlib import Path
import sys
from typing import Any, Sequence

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from boxfusion.frozen_anchor_manifest import (  # noqa: E402
    verify_frozen_anchor_manifest,
)
from boxfusion.frozen_b6_manifest import read_scene_list
from boxfusion.tr3d_residual_cache import (
    TR3DResidualCache,
    load_tr3d_residual_cache,
    tr3d_residual_cache_path,
)


REPORT_SCHEMA = "boxfusion.tr3d_residual_union_oracle.v2"
DEFAULT_THRESHOLDS = (0.15, 0.25, 0.50)
DEFAULT_CANDIDATE_SCORE_THRESHOLDS = (
    0.01,
    0.05,
    0.10,
    0.20,
    0.30,
    0.40,
    0.50,
    0.60,
    0.70,
    0.80,
    0.90,
)
_SIGNS = np.asarray(
    [
        [-1, -1, -1],
        [-1, -1, 1],
        [-1, 1, -1],
        [-1, 1, 1],
        [1, -1, -1],
        [1, -1, 1],
        [1, 1, -1],
        [1, 1, 1],
    ],
    dtype=np.float64,
)


def _load_b6(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with path.open("rb") as handle:
        payload = pickle.load(handle)  # noqa: S301 - trusted local result
    if (
        not isinstance(payload, (list, tuple))
        or len(payload) != 1
        or not isinstance(payload[0], (list, tuple))
    ):
        raise ValueError(f"{path}: malformed BoxFusion prediction")
    corners: list[np.ndarray] = []
    scores: list[float] = []
    for index, detection in enumerate(payload[0]):
        if not isinstance(detection, (list, tuple)) or len(detection) != 3:
            raise ValueError(f"{path}: malformed detection {index}")
        value = np.asarray(detection[1])
        score = float(detection[2])
        if value.shape != (8, 3) or not np.isfinite(value).all():
            raise ValueError(f"{path}: invalid corners {index}")
        if not math.isfinite(score):
            raise ValueError(f"{path}: invalid score {index}")
        corners.append(np.asarray(value, dtype=np.float64))
        scores.append(score)
    return (
        np.stack(corners)
        if corners
        else np.empty((0, 8, 3), dtype=np.float64),
        np.asarray(scores, dtype=np.float64),
    )


def _tr3d_corners(boxes: np.ndarray) -> np.ndarray:
    values = np.asarray(boxes, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 7:
        raise ValueError("TR3D boxes must be [N,7]")
    local = _SIGNS[None] * (0.5 * values[:, None, 3:6])
    cosine = np.cos(values[:, 6])
    sine = np.sin(values[:, 6])
    result = np.empty_like(local)
    result[:, :, 0] = (
        local[:, :, 0] * cosine[:, None]
        - local[:, :, 1] * sine[:, None]
    )
    result[:, :, 1] = (
        local[:, :, 0] * sine[:, None]
        + local[:, :, 1] * cosine[:, None]
    )
    result[:, :, 2] = local[:, :, 2]
    return result + values[:, None, :3]


def _alignment(scans_root: Path, scene_id: str) -> np.ndarray:
    metadata = scans_root / scene_id / f"{scene_id}.txt"
    values = None
    for line in metadata.read_text(encoding="utf-8").splitlines():
        if line.strip().startswith("axisAlignment"):
            values = np.fromstring(line.split("=", 1)[1], sep=" ")
            break
    if values is None or values.size != 16 or not np.isfinite(values).all():
        raise ValueError(f"{metadata}: missing/invalid axisAlignment")
    transform = values.reshape(4, 4)
    if not np.allclose(transform[3], [0, 0, 0, 1], atol=1e-6):
        raise ValueError(f"{metadata}: non-homogeneous axisAlignment")
    return transform


def _validate_alignment_provenance(
    scene_id: str,
    unaligned_to_aligned: np.ndarray,
    aligned_to_unaligned: np.ndarray,
) -> None:
    """Validate the cached inverse without cross-runtime byte equality.

    The immutable parent-cache loader already verifies that its stored SHA
    hashes its exact matrix bytes.  Recomputing an inverse under another
    NumPy/LAPACK build may differ by a few ULPs, so the external contract is
    the inverse relationship in both multiplication directions.
    """

    identity = np.eye(4, dtype=np.float64)
    forward = np.asarray(unaligned_to_aligned, dtype=np.float64)
    inverse = np.asarray(aligned_to_unaligned, dtype=np.float64)
    if not (
        np.allclose(
            forward @ inverse,
            identity,
            rtol=0.0,
            atol=1e-10,
        )
        and np.allclose(
            inverse @ forward,
            identity,
            rtol=0.0,
            atol=1e-10,
        )
    ):
        raise ValueError(
            f"{scene_id}: cache inverse axisAlignment provenance mismatch"
        )


def _transform(corners: np.ndarray, transform: np.ndarray) -> np.ndarray:
    return (
        corners @ transform[:3, :3].T
        + transform[None, None, :3, 3]
    )


def _minmax(corners: np.ndarray) -> np.ndarray:
    if not len(corners):
        return np.empty((0, 6), dtype=np.float64)
    return np.concatenate((corners.min(1), corners.max(1)), axis=1)


def _gt_boxes(path: Path) -> np.ndarray:
    payload = np.load(path, allow_pickle=False)
    if payload.ndim != 2 or payload.shape[1] < 6:
        raise ValueError(f"{path}: GT must be [N,>=6]")
    boxes = np.asarray(payload[:, :6], dtype=np.float64)
    if not np.isfinite(boxes).all() or (
        len(boxes) and np.any(boxes[:, 3:] <= 0)
    ):
        raise ValueError(f"{path}: invalid GT boxes")
    return np.concatenate(
        (boxes[:, :3] - boxes[:, 3:] / 2, boxes[:, :3] + boxes[:, 3:] / 2),
        axis=1,
    )


def pairwise_iou(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    if not len(left) or not len(right):
        return np.zeros((len(left), len(right)), dtype=np.float64)
    size = np.maximum(
        np.minimum(left[:, None, 3:], right[None, :, 3:])
        - np.maximum(left[:, None, :3], right[None, :, :3]),
        0,
    )
    intersection = np.prod(size, axis=2)
    left_volume = np.prod(left[:, 3:] - left[:, :3], axis=1)
    right_volume = np.prod(right[:, 3:] - right[:, :3], axis=1)
    union = left_volume[:, None] + right_volume[None] - intersection
    return np.divide(
        intersection,
        union,
        out=np.zeros_like(intersection),
        where=union > 0,
    )


def score_match(iou: np.ndarray, scores: np.ndarray, threshold: float) -> int:
    available = np.ones(iou.shape[1], dtype=np.bool_)
    matched = 0
    for prediction in np.argsort(-scores, kind="stable"):
        candidates = np.flatnonzero(available)
        if not len(candidates):
            break
        local = iou[prediction, candidates]
        best = int(np.argmax(local))
        if float(local[best]) > threshold:
            available[int(candidates[best])] = False
            matched += 1
    return matched


def maximum_cardinality(iou: np.ndarray, threshold: float) -> int:
    """Maximum bipartite prediction/GT matches above ``threshold``."""

    adjacency = [
        np.flatnonzero(row > threshold).tolist()
        for row in np.asarray(iou)
    ]
    gt_owner = [-1] * iou.shape[1]

    def augment(prediction: int, seen: set[int]) -> bool:
        for gt in adjacency[prediction]:
            if gt in seen:
                continue
            seen.add(gt)
            if gt_owner[gt] < 0 or augment(gt_owner[gt], seen):
                gt_owner[gt] = prediction
                return True
        return False

    matched = 0
    # More constrained predictions first improves deterministic traversal,
    # while the augmenting-path algorithm still yields maximum cardinality.
    order = sorted(range(len(adjacency)), key=lambda row: (len(adjacency[row]), row))
    for prediction in order:
        matched += int(augment(prediction, set()))
    return matched


def _manifest_snapshot(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "anchor_name": payload["anchor_name"],
        "prediction_tree_sha256": payload["prediction_tree_sha256"],
        "artifact_tree_sha256": payload["artifact_tree_sha256"],
        "scene_list_sha256": payload["scene_list_sha256"],
    }


def _score_key(value: float) -> str:
    return f"{value:.4f}"


def _validate_candidate_score_thresholds(
    values: Sequence[float],
) -> tuple[float, ...]:
    result = tuple(float(value) for value in values)
    if not result or len(result) != len(set(result)):
        raise ValueError("candidate score thresholds must be non-empty/unique")
    if any(not math.isfinite(value) or value < 0 or value > 1 for value in result):
        raise ValueError("candidate score thresholds must be finite in [0,1]")
    return tuple(sorted(result))


def audit(
    *,
    manifest_path: Path,
    cache_root: Path,
    gt_root: Path,
    scans_root: Path,
    scene_list: Path | None,
    prefix_id: str,
    checkpoint_sha256: str,
    config_sha256: str,
    thresholds: Sequence[float] = DEFAULT_THRESHOLDS,
    candidate_score_thresholds: Sequence[float] = (
        DEFAULT_CANDIDATE_SCORE_THRESHOLDS
    ),
) -> dict[str, Any]:
    candidate_score_thresholds = _validate_candidate_score_thresholds(
        candidate_score_thresholds
    )
    before = verify_frozen_anchor_manifest(manifest_path)
    all_scenes = tuple(before["scene_ids"])
    scenes = all_scenes if scene_list is None else read_scene_list(scene_list)
    if not set(scenes).issubset(all_scenes):
        raise ValueError("audit scene list is not a frozen-B6 subset")
    prediction_root = Path(before["reference_result_root"])
    totals = {
        f"{threshold:.2f}": {
            "b6_score_tp": 0,
            "tr3d_score_tp": 0,
            "union_score_tp": 0,
            "b6_oracle_tp": 0,
            "tr3d_oracle_tp": 0,
            "union_oracle_tp": 0,
            "duplicate_candidates": 0,
            "candidate_gt_hits": 0,
            "novel_oracle_tp": 0,
            "novel_tp_scenes": 0,
        }
        for threshold in thresholds
    }
    total_gt = 0
    total_b6 = 0
    total_candidates = 0
    total_runtime = 0.0
    score_frontier_totals: dict[str, Any] = {
        _score_key(score_threshold): {
            "candidate_score_threshold": score_threshold,
            "candidate_count": 0,
            "thresholds": {
                f"{iou_threshold:.2f}": {
                    "union_score_tp": 0,
                    "union_oracle_tp": 0,
                    "novel_oracle_tp": 0,
                    "novel_tp_scenes": 0,
                    "candidate_gt_hits": 0,
                    "duplicate_candidates": 0,
                }
                for iou_threshold in thresholds
            },
        }
        for score_threshold in candidate_score_thresholds
    }
    per_scene: dict[str, Any] = {}
    for scene_id in scenes:
        baseline_corners, baseline_scores = _load_b6(
            prediction_root / f"{scene_id}_boxes.pkl"
        )
        residual: TR3DResidualCache = load_tr3d_residual_cache(
            tr3d_residual_cache_path(cache_root, scene_id, prefix_id),
            expected_scene_id=scene_id,
            expected_prefix_id=prefix_id,
            expected_checkpoint_sha256=checkpoint_sha256,
            expected_config_sha256=config_sha256,
        )
        transform = _alignment(scans_root, scene_id)
        _validate_alignment_provenance(
            scene_id,
            transform,
            residual.aligned_to_unaligned,
        )
        baseline_boxes = _minmax(_transform(baseline_corners, transform))
        candidate_boxes = _minmax(
            _transform(residual.corners_world, transform)
        )
        targets = _gt_boxes(gt_root / f"{scene_id}_bbox.npy")
        b6_iou = pairwise_iou(baseline_boxes, targets)
        candidate_iou = pairwise_iou(candidate_boxes, targets)
        union_iou = np.concatenate((b6_iou, candidate_iou), axis=0)
        union_scores = np.concatenate(
            (baseline_scores, residual.scores_3d.astype(np.float64))
        )
        candidate_b6_iou = pairwise_iou(candidate_boxes, baseline_boxes)
        candidate_scores = residual.scores_3d.astype(np.float64)
        scene_metrics: dict[str, Any] = {}
        for threshold in thresholds:
            key = f"{threshold:.2f}"
            b6_oracle = maximum_cardinality(b6_iou, threshold)
            candidate_oracle = maximum_cardinality(candidate_iou, threshold)
            union_oracle = maximum_cardinality(union_iou, threshold)
            novel = union_oracle - b6_oracle
            row = {
                "b6_score_tp": score_match(
                    b6_iou, baseline_scores, threshold
                ),
                "tr3d_score_tp": score_match(
                    candidate_iou,
                    candidate_scores,
                    threshold,
                ),
                "union_score_tp": score_match(
                    union_iou, union_scores, threshold
                ),
                "b6_oracle_tp": b6_oracle,
                "tr3d_oracle_tp": candidate_oracle,
                "union_oracle_tp": union_oracle,
                "duplicate_candidates": int(
                    np.sum(
                        np.max(candidate_b6_iou, axis=1, initial=0.0)
                        > threshold
                    )
                ),
                "candidate_gt_hits": int(
                    np.sum(
                        np.max(candidate_iou, axis=1, initial=0.0)
                        > threshold
                    )
                ),
                "novel_oracle_tp": novel,
                "novel_tp_scenes": int(novel > 0),
            }
            scene_metrics[key] = row
            for name, value in row.items():
                totals[key][name] += int(value)
        for candidate_score_threshold in candidate_score_thresholds:
            score_row = score_frontier_totals[
                _score_key(candidate_score_threshold)
            ]
            selected = candidate_scores >= candidate_score_threshold
            selected_iou = candidate_iou[selected]
            selected_b6_iou = candidate_b6_iou[selected]
            selected_scores = candidate_scores[selected]
            selected_union_iou = np.concatenate(
                (b6_iou, selected_iou), axis=0
            )
            selected_union_scores = np.concatenate(
                (baseline_scores, selected_scores)
            )
            score_row["candidate_count"] += int(np.count_nonzero(selected))
            for iou_threshold in thresholds:
                key = f"{iou_threshold:.2f}"
                b6_oracle = scene_metrics[key]["b6_oracle_tp"]
                union_oracle = maximum_cardinality(
                    selected_union_iou, iou_threshold
                )
                novel = union_oracle - b6_oracle
                frontier = score_row["thresholds"][key]
                frontier["union_score_tp"] += score_match(
                    selected_union_iou,
                    selected_union_scores,
                    iou_threshold,
                )
                frontier["union_oracle_tp"] += union_oracle
                frontier["novel_oracle_tp"] += novel
                frontier["novel_tp_scenes"] += int(novel > 0)
                frontier["candidate_gt_hits"] += int(
                    np.sum(
                        np.max(selected_iou, axis=1, initial=0.0)
                        > iou_threshold
                    )
                )
                frontier["duplicate_candidates"] += int(
                    np.sum(
                        np.max(selected_b6_iou, axis=1, initial=0.0)
                        > iou_threshold
                    )
                )
        per_scene[scene_id] = {
            "ground_truth_count": len(targets),
            "b6_predictions": len(baseline_boxes),
            "tr3d_candidates": residual.proposal_count,
            "runtime_s": residual.runtime_s,
            "thresholds": scene_metrics,
        }
        total_gt += len(targets)
        total_b6 += len(baseline_boxes)
        total_candidates += residual.proposal_count
        total_runtime += residual.runtime_s

    threshold_report: dict[str, Any] = {}
    for key, row in totals.items():
        threshold_report[key] = {
            **row,
            "ground_truth_count": total_gt,
            "b6_score_recall": row["b6_score_tp"] / max(total_gt, 1),
            "tr3d_score_recall": row["tr3d_score_tp"] / max(total_gt, 1),
            "union_score_recall": row["union_score_tp"] / max(total_gt, 1),
            "b6_oracle_recall": row["b6_oracle_tp"] / max(total_gt, 1),
            "tr3d_oracle_recall": row["tr3d_oracle_tp"] / max(total_gt, 1),
            "union_oracle_recall": row["union_oracle_tp"] / max(total_gt, 1),
            "union_oracle_recall_gain": (
                row["union_oracle_tp"] - row["b6_oracle_tp"]
            )
            / max(total_gt, 1),
            "novel_precision_upper_bound": row["novel_oracle_tp"]
            / max(total_candidates, 1),
            "candidate_gt_hit_rate": row["candidate_gt_hits"]
            / max(total_candidates, 1),
            "duplicate_rate": row["duplicate_candidates"]
            / max(total_candidates, 1),
        }
    score_frontier: dict[str, Any] = {}
    for score_key, raw in score_frontier_totals.items():
        candidate_count = int(raw["candidate_count"])
        score_thresholds: dict[str, Any] = {}
        for iou_key, row in raw["thresholds"].items():
            b6_oracle = totals[iou_key]["b6_oracle_tp"]
            score_thresholds[iou_key] = {
                **row,
                "union_score_recall": row["union_score_tp"]
                / max(total_gt, 1),
                "union_oracle_recall": row["union_oracle_tp"]
                / max(total_gt, 1),
                "union_oracle_recall_gain": (
                    row["union_oracle_tp"] - b6_oracle
                )
                / max(total_gt, 1),
                "novel_precision_upper_bound": row["novel_oracle_tp"]
                / max(candidate_count, 1),
                "candidate_gt_hit_rate": row["candidate_gt_hits"]
                / max(candidate_count, 1),
                "duplicate_rate": row["duplicate_candidates"]
                / max(candidate_count, 1),
            }
        score_frontier[score_key] = {
            "candidate_score_threshold": raw[
                "candidate_score_threshold"
            ],
            "candidate_count": candidate_count,
            "thresholds": score_thresholds,
        }

    after = verify_frozen_anchor_manifest(manifest_path)
    before_snapshot = _manifest_snapshot(before)
    after_snapshot = _manifest_snapshot(after)
    if before_snapshot != after_snapshot:
        raise RuntimeError("frozen B6 changed during TR3D observer audit")
    gain25 = threshold_report.get("0.25", {}).get(
        "union_oracle_recall_gain", 0.0
    )
    gain50 = threshold_report.get("0.50", {}).get(
        "union_oracle_recall_gain", 0.0
    )
    novel50 = threshold_report.get("0.50", {}).get("novel_oracle_tp", 0)
    scenes50 = threshold_report.get("0.50", {}).get("novel_tp_scenes", 0)
    precision50 = threshold_report.get("0.50", {}).get(
        "novel_precision_upper_bound", 0.0
    )
    return {
        "schema": REPORT_SCHEMA,
        "observer_contract": {
            "observer_only": True,
            "mutation_enabled": False,
            "applied_count": 0,
            "frozen_anchor_verified_before_and_after": True,
            # Kept for readers of the v1 report schema.
            "frozen_b6_verified_before_and_after": True,
            "before": before_snapshot,
            "after": after_snapshot,
        },
        "anchor": {
            "name": before["anchor_name"],
            "metrics_percent": before["anchor_metrics_percent"],
        },
        "frozen_anchor_manifest": str(manifest_path.resolve()),
        "cache_root": str(cache_root.resolve()),
        "prefix_id": prefix_id,
        "scene_count": len(scenes),
        "ground_truth_count": total_gt,
        "b6_prediction_count": total_b6,
        "tr3d_candidate_count": total_candidates,
        "tr3d_runtime_s": total_runtime,
        "thresholds": threshold_report,
        "score_frontier": {
            "purpose": (
                "diagnostic-only fixed grid; do not select a deployment "
                "threshold on validation scenes"
            ),
            "rows": score_frontier,
        },
        "continuation_gate": {
            "required_delta_recall_025": 0.08,
            "required_delta_recall_050": 0.05,
            "required_novel_tp50": 5,
            "required_novel_tp50_scenes": 3,
            "required_novel_precision_050": 0.15,
            "delta_recall_025": gain25,
            "delta_recall_050": gain50,
            "novel_tp50": novel50,
            "novel_tp50_scenes": scenes50,
            "novel_precision_050_upper_bound": precision50,
            "pass": bool(
                gain25 >= 0.08
                and gain50 >= 0.05
                and novel50 >= 5
                and scenes50 >= 3
                and precision50 >= 0.15
            ),
        },
        "per_scene": per_scene,
    }


def build_parser() -> argparse.ArgumentParser:
    root = _ROOT
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=root / "manifests" / "frozen_b6_full100.json",
    )
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument(
        "--gt-root",
        type=Path,
        default=Path(
            "/data/ZhaoX/BoxFusion/evaluation/data_util/"
            "scannet_train_detection_data"
        ),
    )
    parser.add_argument(
        "--scans-root",
        type=Path,
        default=Path("/extra/ZhaoX/scannet_data/scans"),
    )
    parser.add_argument("--scene-list", type=Path)
    parser.add_argument("--prefix-id", default="full")
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--config-sha256", required=True)
    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=list(DEFAULT_THRESHOLDS),
    )
    parser.add_argument(
        "--candidate-score-thresholds",
        type=float,
        nargs="+",
        default=list(DEFAULT_CANDIDATE_SCORE_THRESHOLDS),
    )
    parser.add_argument("--report", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report_path = args.report.resolve()
    manifest = verify_frozen_anchor_manifest(args.manifest)
    frozen_root = Path(manifest["reference_result_root"]).resolve()
    if report_path == frozen_root or frozen_root in report_path.parents:
        raise ValueError("audit report must not be written inside frozen B6")
    report = audit(
        manifest_path=args.manifest,
        cache_root=args.cache_root,
        gt_root=args.gt_root,
        scans_root=args.scans_root,
        scene_list=args.scene_list,
        prefix_id=args.prefix_id,
        checkpoint_sha256=args.checkpoint_sha256,
        config_sha256=args.config_sha256,
        thresholds=args.thresholds,
        candidate_score_thresholds=args.candidate_score_thresholds,
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
