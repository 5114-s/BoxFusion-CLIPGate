#!/usr/bin/env python3
"""GT-only counterfactual audit for pre-registered R5 grouping vetoes."""

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

from boxfusion.tr3d_r5_spgroup_cache import load_r5_sidecar  # noqa: E402
from boxfusion.tr3d_r5_spgroup_observer import METRIC_NAMES  # noqa: E402
from tools.audit_tr3d_r4_verifier import (  # noqa: E402
    THRESHOLDS, _alignment, _ground_truth, _iou, _metrics, _minmax, _prediction,
)
from tools.tr3d_data import read_scene_list  # noqa: E402


SCHEMA = "boxfusion.tr3d_r5_spgroup_counterfactual_audit.v1"


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--r5-cache-root", type=Path, required=True)
    value.add_argument("--same-run-baseline-root", type=Path, required=True)
    value.add_argument("--active-prediction-root", type=Path, required=True)
    value.add_argument("--scene-list", type=Path, required=True)
    value.add_argument("--gt-root", type=Path, required=True)
    value.add_argument("--scans-root", type=Path, required=True)
    value.add_argument("--prefix-id", default="p100")
    value.add_argument("--report", type=Path, required=True)
    return value


def _write(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(report, indent=2, sort_keys=True) + "\n").encode()
    temporary: str | None = None
    try:
        descriptor, temporary = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded); handle.flush(); os.fsync(handle.fileno())
        os.link(temporary, path); path.chmod(0o444)
    except FileExistsError as error:
        raise FileExistsError(f"immutable R5 audit exists: {path}") from error
    finally:
        if temporary is not None:
            try: os.unlink(temporary)
            except FileNotFoundError: pass


def _delta(metrics: dict[str, dict[str, float | int]], base: dict[str, dict[str, float | int]]) -> dict[str, float]:
    return {
        f"AP{int(threshold * 100)}":
        float(metrics[f"{threshold:.2f}"]["average_precision"])
        - float(base[f"{threshold:.2f}"]["average_precision"])
        for threshold in THRESHOLDS
    }


def audit(args: argparse.Namespace) -> dict[str, Any]:
    metric = {name: index for index, name in enumerate(METRIC_NAMES)}
    rows: list[dict[str, Any]] = []
    rules_per_scene: dict[str, list[np.ndarray]] = {
        "partition_strict_veto": [], "learned_strict_veto": [],
    }
    selected_iou: dict[str, list[tuple[float, float]]] = {key: [] for key in rules_per_scene}
    pair_count = 0
    for scene in read_scene_list(args.scene_list.resolve()):
        baseline_corners, baseline_scores = _prediction(args.same_run_baseline_root.resolve() / f"{scene}_boxes.pkl")
        active_corners, active_scores = _prediction(args.active_prediction_root.resolve() / f"{scene}_boxes.pkl")
        if baseline_scores.shape != active_scores.shape or not np.array_equal(baseline_scores, active_scores):
            raise ValueError(f"{scene}: baseline/active scores differ")
        transform = _alignment(args.scans_root.resolve(), scene)
        baseline = _minmax(baseline_corners, transform)
        active = _minmax(active_corners, transform)
        gt = _ground_truth(args.gt_root.resolve() / f"{scene}_bbox.npy")
        sidecar = load_r5_sidecar(args.r5_cache_root.resolve() / scene / f"{args.prefix_id}.r5g.npz")
        indices = np.asarray(sidecar["anchor_indices"], dtype=np.int64)
        if len(indices) and (indices.min() < 0 or indices.max() >= len(active)):
            raise ValueError(f"{scene}: R5 anchor index outside prediction array")
        values = np.asarray(sidecar["metrics"], dtype=np.float64)
        valid = np.asarray(sidecar["metric_valid"], dtype=np.bool_)
        delta = np.asarray(sidecar["candidate_minus_anchor"], dtype=np.float64)
        support = np.all(values[:, :, metric["mesh_vertex_count"]] >= 32, axis=1)
        partition = (
            support
            & valid[:, :, metric["partition_completeness"]].all(axis=1)
            & valid[:, :, metric["normalized_partition_entropy"]].all(axis=1)
            & valid[:, :, metric["center_dispersion_over_box_diagonal"]].all(axis=1)
            & (delta[:, metric["partition_completeness"]] < 0.0)
            & (delta[:, metric["normalized_partition_entropy"]] > 0.0)
            & (delta[:, metric["center_dispersion_over_box_diagonal"]] > 0.0)
        )
        learned = (
            partition
            & valid[:, :, metric["embedding_cohesion"]].all(axis=1)
            & valid[:, :, metric["vote_dispersion"]].all(axis=1)
            & (delta[:, metric["embedding_cohesion"]] < 0.0)
            & (delta[:, metric["vote_dispersion"]] > 0.0)
        )
        scene_rules = {"partition_strict_veto": partition, "learned_strict_veto": learned}
        for name, veto in scene_rules.items():
            rules_per_scene[name].append(veto)
            if len(indices) and len(gt):
                anchor_iou = _iou(baseline[indices], gt).max(axis=1)
                candidate_iou = _iou(active[indices], gt).max(axis=1)
                selected_iou[name].extend(zip(anchor_iou[veto].tolist(), candidate_iou[veto].tolist()))
        rows.append({
            "scene_id": scene, "baseline": baseline, "active": active,
            "scores": active_scores, "gt": gt, "anchor_indices": indices,
            "rules": scene_rules,
        })
        pair_count += len(indices)

    baseline_metrics = _metrics(rows, "baseline")
    active_metrics = _metrics(rows, "active")
    rules_report: dict[str, Any] = {}
    for name in rules_per_scene:
        key = f"counterfactual_{name}"
        selected = 0
        for row in rows:
            row[key] = np.array(row["active"], copy=True)
            mask = row["rules"][name]
            indices = row["anchor_indices"][mask]
            row[key][indices] = row["baseline"][indices]
            selected += int(mask.sum())
        result = _metrics(rows, key)
        pairs = selected_iou[name]
        crossing_gain = sum(anchor < 0.5 <= candidate for anchor, candidate in pairs)
        crossing_loss = sum(candidate < 0.5 <= anchor for anchor, candidate in pairs)
        rules_report[name] = {
            "veto_count": selected,
            "metrics": result,
            "delta_vs_active": _delta(result, active_metrics),
            "selected_candidate_cross50_gain_if_kept": crossing_gain,
            "selected_candidate_cross50_loss_if_kept": crossing_loss,
            "selected_candidate_cross50_net_if_kept": crossing_gain - crossing_loss,
            "activation_authorized": False,
        }
        for row in rows:
            del row[key]

    report = {
        "schema": SCHEMA, "ground_truth_only_offline_audit": True,
        "observer_mutation_enabled": False, "observer_applied_count": 0,
        "pre_registered_rules": {
            "partition_strict_veto": "support>=32 and candidate completeness lower, entropy higher, normalized center dispersion higher",
            "learned_strict_veto": "partition_strict_veto and candidate embedding cohesion lower and vote dispersion higher",
            "all_comparisons": "zero threshold; no validation-fitted constants",
        },
        "scene_count": len(rows), "pair_count": pair_count,
        "baseline_metrics": baseline_metrics, "active_r3_metrics": active_metrics,
        "active_delta_vs_baseline": _delta(active_metrics, baseline_metrics),
        "rules": rules_report,
        "decision": "OBSERVER_ONLY_REQUIRES_HELDOUT_CONFIRMATION",
    }
    _write(args.report.resolve(), report)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    report = audit(parser().parse_args(argv))
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
