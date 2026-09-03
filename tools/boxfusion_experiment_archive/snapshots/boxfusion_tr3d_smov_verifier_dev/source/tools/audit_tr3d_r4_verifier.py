#!/usr/bin/env python3
"""GT-only counterfactual audit for frozen R4 observer evidence.

Ground truth is confined to this offline tool.  It never writes predictions
and evaluates pre-registered, zero-threshold paired-dominance rules only.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import pickle
from pathlib import Path
import sys
import tempfile
from typing import Any, Sequence

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from boxfusion.tr3d_r2_provenance import sha256_file  # noqa: E402
from boxfusion.tr3d_r2b_observer import FEATURE_STAT_NAMES  # noqa: E402
from boxfusion.tr3d_r4_smov_cache import load_r4_depth_sidecar  # noqa: E402
from boxfusion.tr3d_r4_smov_feature_cache import load_r4_feature_sidecar  # noqa: E402
from tools.tr3d_data import read_scene_list  # noqa: E402


SCHEMA = "boxfusion.tr3d_r4_smov_counterfactual_audit.v1"
THRESHOLDS = (0.15, 0.25, 0.50)


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--r4-depth-cache-root", type=Path, required=True)
    value.add_argument("--r4-feature-cache-root", type=Path, required=True)
    value.add_argument("--same-run-baseline-root", type=Path, required=True)
    value.add_argument("--active-prediction-root", type=Path, required=True)
    value.add_argument("--scene-list", type=Path, required=True)
    value.add_argument("--gt-root", type=Path, required=True)
    value.add_argument("--scans-root", type=Path, required=True)
    value.add_argument("--prefix-id", default="p100")
    value.add_argument("--report", type=Path, required=True)
    return value


def _prediction(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with path.open("rb") as handle:
        payload = pickle.load(handle)  # noqa: S301 - local experiment artifact
    if not isinstance(payload, (list, tuple)) or len(payload) != 1 or not isinstance(payload[0], (list, tuple)):
        raise ValueError(f"{path}: malformed prediction")
    corners, scores = [], []
    for index, row in enumerate(payload[0]):
        if not isinstance(row, (list, tuple)) or len(row) != 3:
            raise ValueError(f"{path}: malformed row {index}")
        geometry = np.asarray(row[1], dtype=np.float64)
        score = float(row[2])
        if geometry.shape != (8, 3) or not np.isfinite(geometry).all() or not math.isfinite(score):
            raise ValueError(f"{path}: invalid row {index}")
        corners.append(geometry)
        scores.append(score)
    return (
        np.stack(corners) if corners else np.empty((0, 8, 3), dtype=np.float64),
        np.asarray(scores, dtype=np.float64),
    )


def _alignment(root: Path, scene: str) -> np.ndarray:
    path = root / scene / f"{scene}.txt"
    values = None
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip().startswith("axisAlignment"):
            values = np.fromstring(line.split("=", 1)[1], sep=" ")
            break
    if values is None or values.size != 16 or not np.isfinite(values).all():
        raise ValueError(f"{path}: invalid axisAlignment")
    result = values.reshape(4, 4)
    if not np.allclose(result[3], [0, 0, 0, 1], atol=1e-6):
        raise ValueError(f"{path}: non-homogeneous axisAlignment")
    return result


def _minmax(corners: np.ndarray, transform: np.ndarray) -> np.ndarray:
    if not len(corners):
        return np.empty((0, 6), dtype=np.float64)
    aligned = corners @ transform[:3, :3].T + transform[None, None, :3, 3]
    return np.concatenate((aligned.min(axis=1), aligned.max(axis=1)), axis=1)


def _ground_truth(path: Path) -> np.ndarray:
    value = np.load(path, allow_pickle=False)
    if value.ndim != 2 or value.shape[1] < 6:
        raise ValueError(f"{path}: GT must be [N,>=6]")
    boxes = np.asarray(value[:, :6], dtype=np.float64)
    if not np.isfinite(boxes).all() or (len(boxes) and np.any(boxes[:, 3:] <= 0)):
        raise ValueError(f"{path}: invalid GT")
    return np.concatenate((boxes[:, :3] - boxes[:, 3:] / 2, boxes[:, :3] + boxes[:, 3:] / 2), axis=1)


def _iou(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    if not len(left) or not len(right):
        return np.zeros((len(left), len(right)), dtype=np.float64)
    lower = np.maximum(left[:, None, :3], right[None, :, :3])
    upper = np.minimum(left[:, None, 3:], right[None, :, 3:])
    intersection = np.prod(np.maximum(upper - lower, 0.0), axis=2)
    lv = np.prod(left[:, 3:] - left[:, :3], axis=1)
    rv = np.prod(right[:, 3:] - right[:, :3], axis=1)
    union = lv[:, None] + rv[None] - intersection
    return np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)


def _voc_ap(recall: np.ndarray, precision: np.ndarray) -> float:
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))
    for index in range(mpre.size - 1, 0, -1):
        mpre[index - 1] = max(mpre[index - 1], mpre[index])
    changing = np.flatnonzero(mrec[1:] != mrec[:-1])
    return float(np.sum((mrec[changing + 1] - mrec[changing]) * mpre[changing + 1]))


def _metric(scenes: Sequence[dict[str, Any]], threshold: float, boxes_key: str) -> dict[str, float | int]:
    records: list[tuple[float, int, str, int, np.ndarray]] = []
    targets: dict[str, np.ndarray] = {}
    total_gt = 0
    for order, scene in enumerate(scenes):
        boxes = np.asarray(scene[boxes_key], dtype=np.float64)
        scores = np.asarray(scene["scores"], dtype=np.float64)
        gt = np.asarray(scene["gt"], dtype=np.float64)
        targets[scene["scene_id"]] = gt
        total_gt += len(gt)
        records.extend((float(score), order, scene["scene_id"], row, boxes[row]) for row, score in enumerate(scores))
    records.sort(key=lambda item: (-item[0], item[1], item[3]))
    used = {scene: np.zeros(len(gt), dtype=np.bool_) for scene, gt in targets.items()}
    tp = np.zeros(len(records), dtype=np.float64)
    fp = np.ones(len(records), dtype=np.float64)
    for index, (_, _, scene, _, box) in enumerate(records):
        gt = targets[scene]
        if not len(gt):
            continue
        overlaps = _iou(box[None], gt)[0]
        target = int(np.argmax(overlaps))
        if overlaps[target] > threshold and not used[scene][target]:
            used[scene][target] = True
            tp[index], fp[index] = 1.0, 0.0
    ctp, cfp = np.cumsum(tp), np.cumsum(fp)
    recall = ctp / float(total_gt + 1e-6)
    precision = ctp / np.maximum(ctp + cfp, np.finfo(np.float64).eps)
    return {
        "predictions": len(records), "ground_truth": total_gt,
        "matched_tp": int(tp.sum()),
        "average_precision": _voc_ap(recall, precision) if len(records) else 0.0,
        "final_precision": float(precision[-1]) if len(precision) else 0.0,
        "final_recall": float(recall[-1]) if len(recall) else 0.0,
    }


def _metrics(scenes: Sequence[dict[str, Any]], boxes_key: str) -> dict[str, dict[str, float | int]]:
    return {f"{threshold:.2f}": _metric(scenes, threshold, boxes_key) for threshold in THRESHOLDS}


def _significant(values: np.ndarray, direction: str) -> bool:
    if len(values) < 2 or not np.isfinite(values).all():
        return False
    mean = float(values.mean())
    standard_error = float(values.std(ddof=1) / np.sqrt(len(values)))
    if direction == "lower":
        return mean + 1.96 * standard_error < 0.0
    if direction == "higher":
        return mean - 1.96 * standard_error > 0.0
    raise AssertionError(direction)


def _rules(depth, feature) -> dict[str, np.ndarray]:
    count = depth.pair_count
    support = np.zeros(count, dtype=np.bool_)
    free_space = np.zeros(count, dtype=np.bool_)
    dino = np.zeros(count, dtype=np.bool_)
    minimum_index = FEATURE_STAT_NAMES.index("pairwise_minimum")
    maximum_index = FEATURE_STAT_NAMES.index("pairwise_maximum")
    for row in range(count):
        valid = depth.topk_view_valid[row]
        enough = (
            np.count_nonzero(valid) >= 2
            and np.all(depth.aggregate_point_count[row] >= 64)
        )
        if enough:
            evidence = depth.per_view_depth_evidence[row, valid]
            support[row] = _significant(evidence[:, 1, 0] - evidence[:, 0, 0], "lower")
            free_space[row] = _significant(evidence[:, 1, 2] - evidence[:, 0, 2], "higher")
        feature_enough = np.all(feature.aggregate_feature_view_count[row] >= 2)
        if feature_enough:
            anchor_min = feature.aggregate_feature_statistics[row, 0, minimum_index]
            candidate_max = feature.aggregate_feature_statistics[row, 1, maximum_index]
            dino[row] = bool(candidate_max < anchor_min)
    return {
        "r4_d_support": support,
        "r4_fs_support_and_free_space": support & free_space,
        "r4_f_full_dominance": support & free_space & dino,
    }


def _write(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(report, indent=2, sort_keys=True) + "\n").encode()
    temporary: str | None = None
    try:
        descriptor, temporary = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
        path.chmod(0o444)
    except FileExistsError as error:
        raise FileExistsError(f"immutable R4 audit exists: {path}") from error
    finally:
        if temporary is not None:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass


def audit(args: argparse.Namespace) -> dict[str, Any]:
    scenes = read_scene_list(args.scene_list.resolve())
    rows: list[dict[str, Any]] = []
    sample_scene: list[int] = []
    sample_local: list[int] = []
    rule_parts: dict[str, list[np.ndarray]] = {}
    for scene_index, scene in enumerate(scenes):
        baseline_corners, baseline_scores = _prediction(args.same_run_baseline_root.resolve() / f"{scene}_boxes.pkl")
        active_corners, active_scores = _prediction(args.active_prediction_root.resolve() / f"{scene}_boxes.pkl")
        if baseline_scores.shape != active_scores.shape or not np.array_equal(baseline_scores, active_scores):
            raise ValueError(f"{scene}: baseline/active scores differ")
        transform = _alignment(args.scans_root.resolve(), scene)
        baseline = _minmax(baseline_corners, transform)
        raw = _minmax(active_corners, transform)
        depth_path = args.r4_depth_cache_root.resolve() / scene / f"{args.prefix_id}.r4d.npz"
        feature_path = args.r4_feature_cache_root.resolve() / scene / f"{args.prefix_id}.r4f.npz"
        depth = load_r4_depth_sidecar(depth_path)
        feature = load_r4_feature_sidecar(feature_path)
        if (
            feature.r4_depth_sidecar_sha256 != sha256_file(depth_path)
            or not np.array_equal(feature.proposal_ids, depth.proposal_ids)
            or not np.array_equal(feature.anchor_indices, depth.anchor_indices)
            or not np.array_equal(feature.topk_frame_ids, depth.topk_frame_ids)
            or not np.array_equal(feature.topk_view_valid, depth.topk_view_valid)
        ):
            raise ValueError(f"{scene}: R4-D/R4-F lineage mismatch")
        indices = np.asarray(depth.anchor_indices, dtype=np.int64)
        if len(indices) and not np.allclose(raw[indices], _minmax(active_corners[indices], transform), atol=0.0, rtol=0.0):
            raise AssertionError("active geometry self-check failed")
        scene_rules = _rules(depth, feature)
        for name, mask in scene_rules.items():
            rule_parts.setdefault(name, []).append(mask)
        sample_scene.extend([scene_index] * len(indices))
        sample_local.extend(range(len(indices)))
        rows.append(
            {
                "scene_id": scene, "baseline": baseline, "raw": raw,
                "scores": baseline_scores,
                "gt": _ground_truth(args.gt_root.resolve() / f"{scene}_bbox.npy"),
                "anchor_indices": indices,
            }
        )
    sample_scene_array = np.asarray(sample_scene, dtype=np.int64)
    sample_local_array = np.asarray(sample_local, dtype=np.int64)
    raw_metrics = _metrics(rows, "raw")
    baseline_metrics = _metrics(rows, "baseline")
    raw_tp = np.asarray([raw_metrics[f"{value:.2f}"]["matched_tp"] for value in THRESHOLDS])
    raw_ap = np.asarray([raw_metrics[f"{value:.2f}"]["average_precision"] for value in THRESHOLDS])
    labels = np.full(len(sample_scene_array), 1, dtype=np.int8)  # 0 gain, 1 neutral, 2 harm
    tp_delta = np.zeros((len(labels), 3), dtype=np.int8)
    ap_delta = np.zeros((len(labels), 3), dtype=np.float64)
    for sample, (scene_index, local) in enumerate(zip(sample_scene_array.tolist(), sample_local_array.tolist())):
        row = rows[scene_index]
        key = f"without_{sample}"
        for current in rows:
            current[key] = np.array(current["raw"], copy=True)
        anchor = int(row["anchor_indices"][local])
        row[key][anchor] = row["baseline"][anchor]
        metrics = _metrics(rows, key)
        without_tp = np.asarray([metrics[f"{value:.2f}"]["matched_tp"] for value in THRESHOLDS])
        without_ap = np.asarray([metrics[f"{value:.2f}"]["average_precision"] for value in THRESHOLDS])
        tp_delta[sample] = (raw_tp - without_tp).astype(np.int8)
        ap_delta[sample] = raw_ap - without_ap
        if np.any(tp_delta[sample] < 0) or np.any(ap_delta[sample] < -1e-12):
            labels[sample] = 2
        elif tp_delta[sample, 2] > 0 and np.all(tp_delta[sample, :2] >= 0) and ap_delta[sample, 2] > 1e-12 and np.all(ap_delta[sample, :2] >= -1e-12):
            labels[sample] = 0
        for current in rows:
            del current[key]
    rules_report: dict[str, Any] = {}
    for name, parts in rule_parts.items():
        veto = np.concatenate(parts) if parts else np.zeros(0, dtype=np.bool_)
        for row in rows:
            row[name] = np.array(row["raw"], copy=True)
        offset = 0
        for row in rows:
            count = len(row["anchor_indices"])
            local_veto = veto[offset:offset + count]
            indices = row["anchor_indices"][local_veto]
            row[name][indices] = row["baseline"][indices]
            offset += count
        veto_count = int(veto.sum())
        harm = labels == 2
        gain = labels == 0
        veto_harm = int(np.count_nonzero(veto & harm))
        veto_gain = int(np.count_nonzero(veto & gain))
        rules_report[name] = {
            "veto_count": veto_count,
            "veto_harm": veto_harm,
            "veto_gain": veto_gain,
            "veto_precision": float(veto_harm / veto_count) if veto_count else 0.0,
            "harm_veto_recall": float(veto_harm / np.count_nonzero(harm)) if np.any(harm) else 0.0,
            "gain_retention": float(1.0 - veto_gain / np.count_nonzero(gain)) if np.any(gain) else 1.0,
            "metrics": _metrics(rows, name),
        }
    # Perfect leave-one-out harm veto is an optimistic diagnostic upper bound.
    for row in rows:
        row["perfect_harm_veto"] = np.array(row["raw"], copy=True)
    for sample in np.flatnonzero(labels == 2):
        scene_index, local = int(sample_scene_array[sample]), int(sample_local_array[sample])
        row = rows[scene_index]
        anchor = int(row["anchor_indices"][local])
        row["perfect_harm_veto"][anchor] = row["baseline"][anchor]
    report = {
        "schema": SCHEMA, "ground_truth_access": True,
        "inference_modules_ground_truth_access": False,
        "observer_only": True, "active_materialization_authorized": False,
        "scene_count": len(rows), "replacement_count": len(labels),
        "label_counts": {
            "gain": int(np.count_nonzero(labels == 0)),
            "neutral": int(np.count_nonzero(labels == 1)),
            "harm": int(np.count_nonzero(labels == 2)),
        },
        "baseline_metrics": baseline_metrics, "raw_r3_metrics": raw_metrics,
        "perfect_harm_veto_metrics": _metrics(rows, "perfect_harm_veto"),
        "rules": rules_report,
        "fixed_rule": {
            "minimum_common_depth_views": 2, "minimum_rays_per_role": 64,
            "confidence_multiplier": 1.96,
            "support_condition": "mean(delta)+1.96*SE<0",
            "free_space_condition": "mean(delta)-1.96*SE>0",
            "feature_condition": "candidate_pairwise_max<anchor_pairwise_min",
            "validation_tuned_thresholds": False,
        },
        "activation_gate": {
            "gain_retention_min": 0.97, "harm_veto_recall_min": 0.25,
            "veto_precision_min": 0.50, "delta_ap50_min": 0.005,
            "max_ap15_ap25_loss": 0.001,
        },
        "decision": "OBSERVER_ONLY_DO_NOT_ACTIVATE",
    }
    return report


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    report = audit(args)
    _write(args.report.resolve(), report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
