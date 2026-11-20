#!/usr/bin/env python3
"""Audit final MVSR support statistics under three TP/FP label definitions."""
from __future__ import annotations

import argparse
import csv
import json
import pickle
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.audit_final_ledger import scannet_inputs
from tools.true_fusion_audit_core import aabb_iou, class_agnostic_ap


def read(path: Path):
    with path.open("rb") as handle:
        rows = pickle.load(handle)[0]
    boxes = np.asarray([row[1] for row in rows], dtype=np.float64).reshape(-1, 8, 3)
    scores = np.asarray([row[2] for row in rows], dtype=np.float64)
    return boxes, scores


def logit(values):
    values = np.clip(values, 1e-4, 1.0 - 1e-4)
    return np.log(values / (1.0 - values))


def ranks(values):
    order = np.argsort(-values, kind="stable")
    result = np.empty(len(values), dtype=np.int64)
    result[order] = np.arange(1, len(values) + 1)
    return result


def greedy(overlap, scores, threshold):
    labels = np.zeros(len(scores), dtype=bool)
    used = np.zeros(overlap.shape[1], dtype=bool)
    for row in np.argsort(-scores, kind="stable"):
        if not overlap.shape[1]:
            continue
        available = np.flatnonzero(~used)
        if not len(available):
            continue
        local = int(overlap[row, available].argmax())
        target = int(available[local])
        if overlap[row, target] > threshold:
            used[target] = True
            labels[row] = True
    return labels


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--components", type=Path, default=ROOT / "development/final_controls/experiments/mvsr_design_20261005/components")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--iou", type=float, default=0.15)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise RuntimeError(f"refusing to overwrite non-empty {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)
    gts, alignments, transform = scannet_inputs()
    suffix = "_boxes.pkl"
    available = {path.name[:-len(suffix)] for path in
                 (args.components / "native_native").glob("*_boxes.pkl")}
    scenes = list(gts)
    if available != set(scenes):
        raise RuntimeError("component and frozen-protocol scene sets differ")
    schemes = {"original_order_greedy": [], "reranked_order_greedy": [], "ranking_independent_max_iou": []}
    scene_rows = []
    native_predictions, reranked_predictions = {}, {}
    for scene in scenes:
        boxes, original = read(args.components / "native_native" / f"{scene}_boxes.pkl")
        reranked_boxes, reranked = read(args.components / "native_factor_no_visibility" / f"{scene}_boxes.pkl")
        if not np.array_equal(boxes, reranked_boxes):
            raise RuntimeError(f"geometry mismatch: {scene}")
        aligned = transform(boxes, alignments[scene])
        overlap = aabb_iou(aligned, gts[scene])
        effective = np.maximum(0.0, (logit(reranked) - logit(original)) / 2.0)
        rank_gain = ranks(original) - ranks(reranked)
        labels = {
            "original_order_greedy": greedy(overlap, original, args.iou),
            "reranked_order_greedy": greedy(overlap, reranked, args.iou),
            "ranking_independent_max_iou": overlap.max(axis=1) > args.iou if overlap.shape[1] else np.zeros(len(boxes), bool),
        }
        for name, value in labels.items():
            schemes[name].append((value, effective, rank_gain))
            scene_rows.append({
                "scene": scene, "label_definition": name, "tp": int(value.sum()),
                "fp": int((~value).sum()),
                "tp_mean_effective_support": float(effective[value].mean()) if value.any() else None,
                "fp_mean_effective_support": float(effective[~value].mean()) if (~value).any() else None,
            })
        native_predictions[scene] = (aligned, original)
        reranked_predictions[scene] = (aligned, reranked)
    summary = {}
    for name, chunks in schemes.items():
        labels = np.concatenate([row[0] for row in chunks])
        effective = np.concatenate([row[1] for row in chunks])
        rank_gain = np.concatenate([row[2] for row in chunks])
        summary[name] = {}
        for group, mask in (("TP", labels), ("FP", ~labels)):
            summary[name][group] = {
                "count": int(mask.sum()),
                "mean_effective_support": float(effective[mask].mean()),
                "positive_support_fraction": float(np.mean(effective[mask] > 1e-10)),
                "mean_rank_change": float(rank_gain[mask].mean()),
            }
    ap = {
        "native": {str(t): class_agnostic_ap(native_predictions, gts, t)["ap"] for t in (0.15, 0.25, 0.5)},
        "mvsr": {str(t): class_agnostic_ap(reranked_predictions, gts, t)["ap"] for t in (0.15, 0.25, 0.5)},
    }
    result = {
        "schema": "boxfusion.scannet100.mvsr_label_audit.v1",
        "scene_count": len(scenes), "native_box_count": sum(summary["original_order_greedy"][g]["count"] for g in ("TP", "FP")),
        "iou_threshold": args.iou,
        "definitions": {
            "original_order_greedy": "one-to-one GT matching in original-native score order",
            "reranked_order_greedy": "one-to-one GT matching recomputed in MVSR score order",
            "ranking_independent_max_iou": "box label is positive iff its maximum GT IoU exceeds the threshold; duplicates may all be positive",
        },
        "summary": summary, "ap": ap,
    }
    (args.output / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    with (args.output / "per_scene.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(scene_rows[0]))
        writer.writeheader(); writer.writerows(scene_rows)
    lines = [
        "# Final MVSR label-definition audit", "",
        f"ScanNet-100, native boxes={result['native_box_count']:,}, 3D AABB IoU={args.iou:.2f}.", "",
        "| Label definition | TP | FP | TP mean effective support | FP mean effective support | TP mean rank change | FP mean rank change |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name in schemes:
        tp, fp = summary[name]["TP"], summary[name]["FP"]
        lines.append(f"| {name} | {tp['count']:,} | {fp['count']:,} | {tp['mean_effective_support']:.4f} | {fp['mean_effective_support']:.4f} | {tp['mean_rank_change']:+.3f} | {fp['mean_rank_change']:+.3f} |")
    lines += ["", "MVSR AP: " + ", ".join(f"AP{int(float(t)*100)} {ap['native'][t]:.4f} → {ap['mvsr'][t]:.4f}" for t in ("0.15", "0.25", "0.5")),
              "", "The ranking-independent row is the preferred robustness check because its correctness label cannot change when MVSR changes scores. The reranked-order row is reported only to show the effect of recomputing greedy matches after reranking."]
    (args.output / "REPORT.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
