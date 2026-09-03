#!/usr/bin/env python3
"""Measure train-only terminal-TR3D replacement headroom after cache sealing.

This tool is intentionally separate from candidate collection.  It reads the
derived train GT only after a GT-free observer audit has been sealed.  The
``oracle`` result is a diagnostic upper bound, never a deployable policy.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import pickle
import re
import tempfile
from urllib.parse import urlsplit
from typing import Any, Mapping

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
import sys

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from boxfusion.ca1m_tr3d_terminal import (  # noqa: E402
    associate_terminal_candidates,
    pairwise_world_aabb_iou,
    sha256_file,
    world_aabb,
)


THRESHOLDS = (0.15, 0.25, 0.50)


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--scene-list", type=Path, required=True)
    value.add_argument("--subset-manifest", type=Path, required=True)
    value.add_argument("--train-dataset", type=Path, required=True)
    value.add_argument("--official-val-list", type=Path, required=True)
    value.add_argument("--data-root", type=Path, required=True)
    value.add_argument("--observer-root", type=Path, required=True)
    value.add_argument("--observer-audit", type=Path, required=True)
    value.add_argument("--output", type=Path, required=True)
    return value


def _regular(path: Path, name: str) -> Path:
    if path.is_symlink():
        raise ValueError(f"{name} must not be a symlink: {path}")
    result = path.resolve()
    if not result.is_file() or result.is_symlink() or result.stat().st_size <= 0:
        raise FileNotFoundError(f"missing regular {name}: {result}")
    return result


def _scenes(path: Path) -> tuple[str, ...]:
    source = _regular(path, "train probe scene list")
    result = tuple(row.strip() for row in source.read_text().splitlines() if row.strip())
    if (
        not result
        or len(result) != len(set(result))
        or any(re.fullmatch(r"[0-9]{8}", scene) is None for scene in result)
    ):
        raise ValueError("train probe scene list is invalid")
    return result


def _val_ids(path: Path) -> set[str]:
    source = _regular(path, "official CA-1M validation URL list")
    result: set[str] = set()
    for row in source.read_text().splitlines():
        if not row.strip():
            continue
        parsed = urlsplit(row.strip())
        matched = re.fullmatch(
            r"/datasets/ca1m/val/ca1m-val-([0-9]{8})\.tar", parsed.path
        )
        if matched is None:
            raise ValueError("official validation list contains an unexpected URL")
        result.add(matched.group(1))
    if not result:
        raise ValueError("official validation list is empty")
    return result


def _voc_ap(recall: np.ndarray, precision: np.ndarray) -> float:
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))
    for index in range(mpre.size - 1, 0, -1):
        mpre[index - 1] = max(mpre[index - 1], mpre[index])
    changed = np.flatnonzero(mrec[1:] != mrec[:-1])
    return float(np.sum((mrec[changed + 1] - mrec[changed]) * mpre[changed + 1]))


def _targets(corners: np.ndarray, gt: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if not len(corners):
        return np.empty((0,), np.float64), np.empty((0,), np.int64)
    if not len(gt):
        return np.zeros(len(corners), np.float64), np.full(len(corners), -1, np.int64)
    iou = pairwise_world_aabb_iou(corners, gt)
    matched = np.argmax(iou, axis=1).astype(np.int64)
    return iou[np.arange(len(corners)), matched], matched


def _metrics(
    predictions: Mapping[str, tuple[np.ndarray, np.ndarray]],
    ground_truth: Mapping[str, np.ndarray],
) -> dict[str, dict[str, float | int]]:
    rows: list[tuple[str, int, float, float, int]] = []
    positives = sum(len(value) for value in ground_truth.values())
    for scene in sorted(predictions):
        corners, scores = predictions[scene]
        target, matched = _targets(corners, ground_truth[scene])
        rows.extend(
            (scene, index, float(scores[index]), float(target[index]), int(matched[index]))
            for index in range(len(corners))
        )
    score = np.asarray([row[2] for row in rows], dtype=np.float64)
    order = np.argsort(-score)
    result: dict[str, dict[str, float | int]] = {}
    for threshold in THRESHOLDS:
        tp = np.zeros(len(rows), dtype=np.float64)
        fp = np.zeros(len(rows), dtype=np.float64)
        detected: set[tuple[str, int]] = set()
        for rank, input_row in enumerate(order.tolist()):
            scene, _, _, iou, gt_index = rows[input_row]
            key = (scene, gt_index)
            if iou > threshold and gt_index >= 0 and key not in detected:
                tp[rank] = 1.0
                detected.add(key)
            else:
                fp[rank] = 1.0
        tp_cumulative = np.cumsum(tp)
        fp_cumulative = np.cumsum(fp)
        recall = tp_cumulative / float(positives + 1e-6)
        precision = tp_cumulative / np.maximum(
            tp_cumulative + fp_cumulative, np.finfo(np.float64).eps
        )
        final_tp = int(tp_cumulative[-1]) if len(tp_cumulative) else 0
        final_fp = int(fp_cumulative[-1]) if len(fp_cumulative) else 0
        result[f"iou_{threshold:.2f}"] = {
            "ap": _voc_ap(recall, precision),
            "precision": float(precision[-1]) if len(precision) else 0.0,
            "recall": float(recall[-1]) if len(recall) else 0.0,
            "tp": final_tp,
            "fp": final_fp,
            "fn": int(positives - final_tp),
        }
    return result


def _prediction_from_observer(archive: Any) -> tuple[np.ndarray, np.ndarray]:
    corners = np.array(archive["anchor_corners"], copy=True)
    scores = np.array(archive["anchor_scores"], copy=True)
    if corners.dtype != np.float32 or scores.dtype != np.float32:
        raise ValueError("observer anchor dtype mismatch")
    return corners, scores


def _write_create_only(path: Path, payload: dict[str, Any]) -> None:
    target = path.resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", prefix=f".{target.name}.", dir=target.parent, delete=False
        ) as handle:
            temporary = handle.name
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, target)
        target.chmod(0o444)
    except FileExistsError as error:
        raise FileExistsError(f"refusing to overwrite train probe report: {target}") from error
    finally:
        if temporary is not None:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass


def run(args: argparse.Namespace) -> dict[str, Any]:
    scenes = _scenes(args.scene_list)
    subset = json.loads(_regular(args.subset_manifest, "probe subset manifest").read_text())
    if (
        subset.get("schema") != "boxfusion.ca1m_tr3d_train_probe_subset.v1"
        or subset.get("train_only") is not True
        or subset.get("validation_scene_access") is not False
        or subset.get("ground_truth_access_during_candidate_collection") is not False
        or subset.get("scene_count") != len(scenes)
        or subset.get("scene_ids_sha256")
        != hashlib.sha256(("\n".join(scenes) + "\n").encode()).hexdigest()
    ):
        raise ValueError("train probe subset manifest contract mismatch")
    if set(scenes) & _val_ids(args.official_val_list):
        raise ValueError("train probe scenes overlap official validation")
    dataset_path = _regular(args.train_dataset, "frozen native-B6 train dataset")
    if sha256_file(dataset_path) != subset.get("dataset_sha256"):
        raise ValueError("frozen train dataset SHA256 mismatch")
    with np.load(dataset_path, allow_pickle=False) as dataset:
        dataset_scenes = np.asarray(dataset["scene_ids"], dtype=np.str_)
        folds = np.asarray(dataset["fold_ids"], dtype=np.int64)
        expected = set(dataset_scenes[folds == int(subset["fold_id"])].tolist())
    if set(scenes) != expected:
        raise ValueError("probe list is not the exact frozen dataset fold")
    audit_path = _regular(args.observer_audit, "sealed GT-free observer audit")
    audit = json.loads(audit_path.read_text())
    if (
        audit.get("schema") != "boxfusion.ca1m_tr3d_terminal_observer_audit.v1"
        or audit.get("ok") is not True
        or audit.get("ground_truth_access") is not False
        or audit.get("scene_count") != len(scenes)
        or set((audit.get("scenes") or {}).keys()) != set(scenes)
    ):
        raise ValueError("observer audit is incomplete or not GT-free")
    actual_cache = {
        path.name.removesuffix("_ca1m_tr3d_terminal.npz")
        for path in args.observer_root.iterdir()
        if path.is_file() and not path.is_symlink() and path.name.endswith("_ca1m_tr3d_terminal.npz")
    }
    if actual_cache != set(scenes):
        raise ValueError("observer root is not the exact sealed train probe set")

    baseline: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    legacy: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    oracle: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    ground_truth: dict[str, np.ndarray] = {}
    per_scene: dict[str, Any] = {}
    improved_pairs = 0
    improved_scenes = 0
    total_gain = 0.0
    cache_hashes: dict[str, str] = {}
    gt_hashes: dict[str, str] = {}
    for scene in scenes:
        cache_path = args.observer_root / f"{scene}_ca1m_tr3d_terminal.npz"
        if sha256_file(cache_path) != audit["scenes"][scene]["artifact_sha256"]:
            raise ValueError(f"observer artifact changed after audit: {scene}")
        gt_path = _regular(
            args.data_root / scene / "derived_train_gt_boxes.npy", "derived train GT"
        )
        gt = np.load(gt_path, allow_pickle=False)
        gt = np.asarray(gt, dtype=np.float64)
        if gt.ndim != 3 or gt.shape[1:] != (8, 3) or not np.isfinite(gt).all():
            raise ValueError(f"invalid derived train GT: {scene}")
        world_aabb(gt)
        ground_truth[scene] = gt
        gt_hashes[scene] = sha256_file(gt_path)
        cache_hashes[scene] = sha256_file(cache_path)
        with np.load(cache_path, allow_pickle=False) as archive:
            anchors, scores = _prediction_from_observer(archive)
            candidates = np.array(archive["candidate_corners"], copy=True)
            candidate_scores = np.array(archive["candidate_scores"], copy=True)
            association = associate_terminal_candidates(
                anchor_corners=anchors,
                anchor_scores=scores,
                candidate_corners=candidates,
                candidate_scores=candidate_scores,
                near_iou=0.15,
            )
            legacy_corners = anchors.copy()
            for candidate_row, anchor_row in zip(
                association.legacy_rule_selected_candidate_rows.tolist(),
                association.legacy_rule_selected_anchor_indices.tolist(),
            ):
                legacy_corners[anchor_row] = candidates[candidate_row]
            oracle_corners = anchors.copy()
            anchor_target, _ = _targets(anchors, gt)
            selected_gain: list[float] = []
            for anchor_row in association.represented_anchor_indices.tolist():
                candidate_rows = np.flatnonzero(
                    association.near_mask
                    & (association.best_anchor_indices == anchor_row)
                )
                candidate_target, _ = _targets(candidates[candidate_rows], gt)
                if len(candidate_target):
                    best_local = int(np.argmax(candidate_target))
                    gain = float(candidate_target[best_local] - anchor_target[anchor_row])
                    if gain > 0.0:
                        oracle_corners[anchor_row] = candidates[int(candidate_rows[best_local])]
                        selected_gain.append(gain)
            scene_improved = sum(gain >= 0.05 for gain in selected_gain)
            if scene_improved:
                improved_scenes += 1
            improved_pairs += scene_improved
            total_gain += sum(selected_gain)
            baseline[scene] = (anchors, scores)
            legacy[scene] = (legacy_corners, scores)
            oracle[scene] = (oracle_corners, scores)
            per_scene[scene] = {
                "anchors": len(anchors),
                "candidates": len(candidates),
                "near_candidates": int(association.near_mask.sum()),
                "represented_anchors": len(association.represented_anchor_indices),
                "legacy_replacements": len(
                    association.legacy_rule_selected_candidate_rows
                ),
                "oracle_positive_replacements": len(selected_gain),
                "oracle_improvements_ge_0_05": scene_improved,
                "oracle_iou_gain_sum": sum(selected_gain),
                "gt_boxes": len(gt),
            }

    metrics = {
        "baseline": _metrics(baseline, ground_truth),
        "legacy_raw_score_rule_diagnostic": _metrics(legacy, ground_truth),
        "oracle_replacement_headroom": _metrics(oracle, ground_truth),
    }
    delta: dict[str, dict[str, float]] = {}
    for name in ("legacy_raw_score_rule_diagnostic", "oracle_replacement_headroom"):
        delta[name] = {
            key: float(metrics[name][key]["ap"] - metrics["baseline"][key]["ap"])
            for key in metrics["baseline"]
        }
    oracle_delta = delta["oracle_replacement_headroom"]
    continue_gate = {
        "oracle_delta_ap25_nonnegative": oracle_delta["iou_0.25"] >= 0.0,
        "oracle_delta_ap50_at_least_0_01": oracle_delta["iou_0.50"] >= 0.01,
        "improved_pairs_ge_10": improved_pairs >= 10,
        "improved_scenes_ge_5": improved_scenes >= 5,
    }
    continue_gate["pass"] = all(continue_gate.values())
    result = {
        "schema": "boxfusion.ca1m_tr3d_train_probe_report.v1",
        "complete": True,
        "train_only": True,
        "official_validation_comparable": False,
        "validation_scene_access": False,
        "candidate_collection_ground_truth_access": False,
        "ground_truth_access_after_cache_seal": True,
        "observer_audit_sha256": sha256_file(audit_path),
        "scene_count": len(scenes),
        "prediction_count": sum(len(value[0]) for value in baseline.values()),
        "gt_count": sum(len(value) for value in ground_truth.values()),
        "cache_hashes": cache_hashes,
        "derived_gt_hashes": gt_hashes,
        "metrics": metrics,
        "ap_delta": delta,
        "oracle_geometry": {
            "improvements_ge_0_05": improved_pairs,
            "scenes_with_improvement_ge_0_05": improved_scenes,
            "positive_iou_gain_sum": total_gain,
        },
        "continue_to_ca_native_selector_gate": continue_gate,
        "legacy_rule_activation_authorized": False,
        "oracle_is_not_deployable": True,
        "per_scene": per_scene,
    }
    _write_create_only(args.output, result)
    return result


def main() -> int:
    result = run(parser().parse_args())
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
