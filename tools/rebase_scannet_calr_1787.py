#!/usr/bin/env python3
"""Rebase frozen CALR controls onto the canonical 1,787-box prefix."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from tools.analyze_calr_causal_controls import (
    ranked_flags,
    read_rows,
    truncate_predictions,
    write_rows,
)
from tools.analyze_scannet_recovery_behavior import (
    prediction,
    quality_counts,
    select_fraction,
)
from tools.audit_final_ledger import scannet_inputs
from tools.true_fusion_audit_core import class_agnostic_ap


THRESHOLDS = (0.15, 0.25, 0.50)
CAUSAL_ARMS = ("streaming_topk", "nA1", "nA2", "nA3")
FRACTIONS = (0.0, 0.25, 0.50, 0.75, 1.0)
TOPK = (50, 100, 200)
FP_PER_100_GT = (1, 5, 10, 20)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-prefix", type=Path, required=True)
    parser.add_argument("--new-prefix", type=Path, required=True)
    parser.add_argument("--calr-component", type=Path, required=True)
    parser.add_argument("--old-causal-predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise RuntimeError(f"refusing to overwrite non-empty output: {args.output}")

    gts, alignments, transform = scannet_inputs()
    scenes = list(gts)
    predictions = {
        "causal": {arm: {} for arm in CAUSAL_ARMS},
        "budget": {str(fraction): {} for fraction in FRACTIONS},
    }
    counts = {
        "causal": {arm: {"prefix": 0, "added": 0, "total": 0} for arm in CAUSAL_ARMS},
        "budget": {str(fraction): {"prefix": 0, "added": 0, "total": 0} for fraction in FRACTIONS},
    }
    quality = {
        "causal": {
            arm: {str(t): {"calr": {"tp": 0, "fp": 0, "duplicate": 0}} for t in THRESHOLDS}
            for arm in CAUSAL_ARMS
        },
        "budget": {
            str(fraction): {str(t): {"calr": {"tp": 0, "fp": 0, "duplicate": 0}} for t in THRESHOLDS}
            for fraction in FRACTIONS
        },
    }

    for scene in scenes:
        filename = f"{scene}_boxes.pkl"
        old_prefix = read_rows(args.old_prefix / filename)
        new_prefix = read_rows(args.new_prefix / filename)
        final_calr = read_rows(args.calr_component / filename)

        for arm in CAUSAL_ARMS:
            old_full = read_rows(args.old_causal_predictions / arm / filename)
            if len(old_full) < len(old_prefix):
                raise RuntimeError(f"short old causal prediction: {arm}/{scene}")
            added = old_full[len(old_prefix) :]
            if arm == "nA3":
                if len(added) != len(final_calr) or any(
                    a[0] != b[0]
                    or not np.array_equal(a[1], b[1])
                    or a[2] != b[2]
                    for a, b in zip(added, final_calr)
                ):
                    raise RuntimeError(f"nA3 CALR suffix mismatch: {scene}")
            rows = new_prefix + added
            write_rows(args.output / "predictions" / "causal" / arm / filename, rows)
            boxes, scores = prediction(rows, transform, alignments[scene])
            predictions["causal"][arm][scene] = boxes, scores
            counts["causal"][arm]["prefix"] += len(new_prefix)
            counts["causal"][arm]["added"] += len(added)
            counts["causal"][arm]["total"] += len(rows)
            sources = np.asarray(["prefix"] * len(new_prefix) + ["calr"] * len(added))
            for threshold in THRESHOLDS:
                result = quality_counts(boxes, scores, sources, gts[scene], threshold)
                for key in ("tp", "fp", "duplicate"):
                    quality["causal"][arm][str(threshold)]["calr"][key] += result["calr"][key]

        for fraction in FRACTIONS:
            label = str(fraction)
            added = select_fraction(final_calr, fraction)
            rows = new_prefix + added
            write_rows(args.output / "predictions" / "budget" / label / filename, rows)
            boxes, scores = prediction(rows, transform, alignments[scene])
            predictions["budget"][label][scene] = boxes, scores
            counts["budget"][label]["prefix"] += len(new_prefix)
            counts["budget"][label]["added"] += len(added)
            counts["budget"][label]["total"] += len(rows)
            sources = np.asarray(["prefix"] * len(new_prefix) + ["calr"] * len(added))
            for threshold in THRESHOLDS:
                result = quality_counts(boxes, scores, sources, gts[scene], threshold)
                for key in ("tp", "fp", "duplicate"):
                    quality["budget"][label][str(threshold)]["calr"][key] += result["calr"][key]

    metrics = {
        group: {
            arm: {
                str(threshold): class_agnostic_ap(values, gts, threshold)["ap"]
                for threshold in THRESHOLDS
            }
            for arm, values in arms.items()
        }
        for group, arms in predictions.items()
    }
    total_gt = sum(len(value) for value in gts.values())
    topk_recall = {arm: {} for arm in CAUSAL_ARMS}
    fixed_fp_recall = {arm: {} for arm in CAUSAL_ARMS}
    for arm in CAUSAL_ARMS:
        for threshold in THRESHOLDS:
            key = str(threshold)
            topk_recall[arm][key] = {}
            for k in TOPK:
                value = class_agnostic_ap(
                    truncate_predictions(predictions["causal"][arm], k),
                    gts,
                    threshold,
                )
                topk_recall[arm][key][str(k)] = {
                    "recall": 100.0 * value["tp"] / total_gt,
                    "tp": int(value["tp"]),
                    "fp": int(value["fp"]),
                }
            scores, ends, tp, fp = ranked_flags(
                predictions["causal"][arm], gts, threshold
            )
            fixed_fp_recall[arm][key] = {}
            for budget in FP_PER_100_GT:
                allowed = int(np.floor(budget * total_gt / 100))
                valid = ends[fp[ends] <= allowed]
                last = int(valid[-1]) if len(valid) else None
                fixed_fp_recall[arm][key][str(budget)] = {
                    "recall": 100.0 * int(tp[last]) / total_gt if last is not None else 0.0,
                    "tp": int(tp[last]) if last is not None else 0,
                    "fp": int(fp[last]) if last is not None else 0,
                    "fp_budget": allowed,
                    "confidence_cutoff": float(scores[last]) if last is not None else None,
                }
    result = {
        "schema": "boxfusion.scannet100.calr_controls_rebased_1787.v1",
        "native_box_count": 1787,
        "new_prefix_box_count": counts["budget"]["0.0"]["prefix"],
        "metrics": metrics,
        "counts": counts,
        "quality": quality,
        "topk_recall": topk_recall,
        "fixed_fp_recall": fixed_fp_recall,
        "sources": {
            "old_prefix": str(args.old_prefix.resolve()),
            "new_prefix": str(args.new_prefix.resolve()),
            "calr_component": str(args.calr_component.resolve()),
            "old_causal_predictions": str(args.old_causal_predictions.resolve()),
        },
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "results.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    lines = [
        "# ScanNet-100 CALR controls rebased to 1,787 native boxes",
        "",
        "## Confirmation controls",
        "",
        "| Arm | Added | AP15 | AP25 | AP50 | Added TP/FP/duplicate @0.15 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for arm in CAUSAL_ARMS:
        q = quality["causal"][arm]["0.15"]["calr"]
        lines.append(
            f"| {arm} | {counts['causal'][arm]['added']:,} | "
            f"{metrics['causal'][arm]['0.15']:.4f} | {metrics['causal'][arm]['0.25']:.4f} | "
            f"{metrics['causal'][arm]['0.5']:.4f} | {q['tp']:,}/{q['fp']:,}/{q['duplicate']:,} |"
        )
    lines += [
        "",
        "## Terminal CALR budget",
        "",
        "| Fraction | Added | AP15 | AP25 | AP50 | Added TP/FP/duplicate @0.15 |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for fraction in FRACTIONS:
        label = str(fraction)
        q = quality["budget"][label]["0.15"]["calr"]
        lines.append(
            f"| {100*fraction:.0f}% | {counts['budget'][label]['added']:,} | "
            f"{metrics['budget'][label]['0.15']:.4f} | {metrics['budget'][label]['0.25']:.4f} | "
            f"{metrics['budget'][label]['0.5']:.4f} | {q['tp']:,}/{q['fp']:,}/{q['duplicate']:,} |"
        )
    (args.output / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
