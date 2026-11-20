#!/usr/bin/env python3
"""Compute CALR output multiplier, cross-branch overlap, and GT multiplicity."""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.audit_final_ledger import scannet_inputs  # noqa: E402
from tools.true_fusion_audit_core import aabb_iou  # noqa: E402


THRESHOLDS = (0.15, 0.25, 0.50)


def read(path: Path) -> np.ndarray:
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    rows = payload[0] if isinstance(payload, (list, tuple)) and len(payload) == 1 else payload
    boxes = [np.asarray(row[1], dtype=np.float64).reshape(8, 3) for row in rows]
    return np.stack(boxes) if boxes else np.empty((0, 8, 3), dtype=np.float64)


def summary(values) -> dict[str, float | int | None]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(len(array)),
        "mean": None if not len(array) else float(array.mean()),
        "median": None if not len(array) else float(np.percentile(array, 50)),
        "p90": None if not len(array) else float(np.percentile(array, 90)),
        "p95": None if not len(array) else float(np.percentile(array, 95)),
        "max": None if not len(array) else float(array.max()),
    }


def overlap_fraction(left: np.ndarray, right: np.ndarray, threshold: float) -> tuple[int, int, float]:
    if not len(left):
        return 0, 0, 0.0
    hits = np.zeros(len(left), dtype=bool) if not len(right) else aabb_iou(left, right).max(axis=1) > threshold
    return int(hits.sum()), int(len(left)), float(hits.mean())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--components", type=Path, required=True)
    parser.add_argument("--native-components", type=Path)
    parser.add_argument("--plr-components", type=Path)
    parser.add_argument("--calr-components", type=Path)
    parser.add_argument("--native-branch", default="native_factor_no_visibility")
    parser.add_argument("--plr-branch", default="births_plr_v1")
    parser.add_argument("--calr-branch", default="births_calr_v1")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    native_components = args.native_components or args.components
    plr_components = args.plr_components or args.components
    calr_components = args.calr_components or args.components
    gts, alignments, transform = scannet_inputs()
    scenes = list(gts)
    totals = {"native": 0, "plr": 0, "calr": 0, "full": 0, "gt": 0}
    scene_calr_native_ratio, scene_full_native_ratio, scene_calr_per_gt = [], [], []
    cross = {str(t): {key: [0, 0] for key in ("calr_to_plr", "plr_to_calr", "calr_to_native", "calr_to_prefix")} for t in THRESHOLDS}
    multiplicity = {str(t): {branch: [] for branch in ("native", "plr", "calr", "full")} for t in THRESHOLDS}
    unique = {str(t): {"calr_covered_gt": 0, "calr_unique_beyond_native_plr": 0} for t in THRESHOLDS}
    per_scene = {}
    for scene in scenes:
        filename = f"{scene}_boxes.pkl"
        branches = {
            "native": read(native_components / args.native_branch / filename),
            "plr": read(plr_components / args.plr_branch / filename),
            "calr": read(calr_components / args.calr_branch / filename),
        }
        branches = {name: transform(boxes, alignments[scene]) for name, boxes in branches.items()}
        branches["full"] = np.concatenate((branches["native"], branches["plr"], branches["calr"]), axis=0)
        prefix = np.concatenate((branches["native"], branches["plr"]), axis=0)
        counts = {name: len(boxes) for name, boxes in branches.items()}
        counts["gt"] = len(gts[scene])
        for name in totals:
            totals[name] += counts[name]
        scene_calr_native_ratio.append(counts["calr"] / max(counts["native"], 1))
        scene_full_native_ratio.append(counts["full"] / max(counts["native"], 1))
        scene_calr_per_gt.append(counts["calr"] / max(counts["gt"], 1))
        for threshold in THRESHOLDS:
            key = str(threshold)
            for label, left, right in (
                ("calr_to_plr", branches["calr"], branches["plr"]),
                ("plr_to_calr", branches["plr"], branches["calr"]),
                ("calr_to_native", branches["calr"], branches["native"]),
                ("calr_to_prefix", branches["calr"], prefix),
            ):
                hit, total, _ = overlap_fraction(left, right, threshold)
                cross[key][label][0] += hit
                cross[key][label][1] += total
            branch_counts = {}
            for name, boxes in branches.items():
                values = np.zeros(len(gts[scene]), dtype=np.int64) if not len(boxes) or not len(gts[scene]) else (aabb_iou(boxes, gts[scene]) > threshold).sum(axis=0)
                multiplicity[key][name].extend(values.tolist())
                branch_counts[name] = values
            calr_covered = branch_counts["calr"] > 0
            prefix_covered = (branch_counts["native"] + branch_counts["plr"]) > 0
            unique[key]["calr_covered_gt"] += int(calr_covered.sum())
            unique[key]["calr_unique_beyond_native_plr"] += int((calr_covered & ~prefix_covered).sum())
        per_scene[scene] = counts
    cross_report = {}
    for threshold, labels in cross.items():
        cross_report[threshold] = {
            label: {"overlapping": values[0], "total": values[1], "fraction": values[0] / max(values[1], 1)}
            for label, values in labels.items()
        }
    multiplicity_report = {}
    for threshold, branches in multiplicity.items():
        multiplicity_report[threshold] = {}
        for name, values in branches.items():
            array = np.asarray(values, dtype=np.int64)
            matched = array[array > 0]
            multiplicity_report[threshold][name] = {
                "all_gt": summary(array), "matched_gt": summary(matched),
                "gt_zero": int(np.count_nonzero(array == 0)),
                "gt_one": int(np.count_nonzero(array == 1)),
                "gt_ge2": int(np.count_nonzero(array >= 2)),
                "excess_predictions_beyond_one_per_gt": int(np.maximum(array - 1, 0).sum()),
            }
    result = {
        "schema": "boxfusion.scannet100.calr_output_duplication.v1",
        "component_source": {
            "native": str(native_components / args.native_branch),
            "plr": str(plr_components / args.plr_branch),
            "calr": str(calr_components / args.calr_branch),
        },
        "thresholds": list(THRESHOLDS), "totals": totals,
        "output_multiplier": {
            "global_calr_over_native": totals["calr"] / max(totals["native"], 1),
            "global_full_over_native": totals["full"] / max(totals["native"], 1),
            "per_scene_calr_over_native": summary(scene_calr_native_ratio),
            "per_scene_full_over_native": summary(scene_full_native_ratio),
            "per_scene_calr_per_gt": summary(scene_calr_per_gt),
        },
        "cross_branch_overlap": cross_report,
        "gt_prediction_multiplicity": multiplicity_report,
        "unique_gt_contribution": unique,
        "per_scene_counts": per_scene,
        "note": "Overlap and multiplicity are geometry-only; scores are not thresholded.",
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "results.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    lines = [
        "# ScanNet-100 CALR output volume and duplication audit", "",
        f"Native/PLR/CALR/full boxes: {totals['native']:,}/{totals['plr']:,}/{totals['calr']:,}/{totals['full']:,}.",
        f"Global CALR/native multiplier: {result['output_multiplier']['global_calr_over_native']:.2f}×; full/native: {result['output_multiplier']['global_full_over_native']:.2f}×.",
        "", "| IoU | CALR→PLR overlap | CALR→native overlap | CALR→prefix overlap | Full GT multiplicity mean | GT with ≥2 predictions | CALR-unique GT |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for threshold in THRESHOLDS:
        key = str(threshold)
        c = cross_report[key]
        m = multiplicity_report[key]["full"]
        lines.append(
            f"| {threshold:.2f} | {100*c['calr_to_plr']['fraction']:.2f}% | {100*c['calr_to_native']['fraction']:.2f}% | "
            f"{100*c['calr_to_prefix']['fraction']:.2f}% | {m['all_gt']['mean']:.2f} | {m['gt_ge2']:,} | {unique[key]['calr_unique_beyond_native_plr']:,} |"
        )
    (args.output / "REPORT.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
