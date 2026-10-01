#!/usr/bin/env python3
"""Analyze MVSR-v2 support, TP/FP separation, and within-scene rank changes."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import pickle
import sys

import numpy as np

MAIN = Path("/data/ZhaoX/BoxFusion")
sys.path.insert(0, str(MAIN))
from tools.audit_ca1m_nms_child_headroom import pairwise_iou
from tools.audit_final_ledger import scannet_inputs

ZERO_EPS = 1e-10


def read(path: Path):
    with path.open("rb") as handle:
        rows = pickle.load(handle)[0]
    return (
        np.asarray([r[1] for r in rows], dtype=np.float64).reshape(-1, 8, 3),
        np.asarray([r[2] for r in rows], dtype=np.float64),
    )


def logit(values: np.ndarray) -> np.ndarray:
    values = np.clip(values, 1e-4, 1 - 1e-4)
    return np.log(values / (1 - values))


def ranks(scores: np.ndarray) -> np.ndarray:
    order = np.argsort(-scores, kind="stable")
    result = np.empty(len(scores), dtype=np.int64)
    result[order] = np.arange(1, len(scores) + 1)
    return result


def greedy_labels(boxes: np.ndarray, scores: np.ndarray, gt: np.ndarray, threshold: float) -> np.ndarray:
    labels = np.zeros(len(boxes), dtype=bool)
    if not len(boxes) or not len(gt):
        return labels
    overlaps = pairwise_iou(boxes, gt)
    used = set()
    for row in np.argsort(-scores, kind="stable"):
        choices = [(float(overlaps[row, col]), col) for col in range(len(gt)) if col not in used]
        if not choices:
            continue
        value, col = max(choices)
        if value > threshold:
            labels[row] = True
            used.add(col)
    return labels


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-list", type=Path, required=True)
    parser.add_argument("--components", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--iou", type=float, default=0.25)
    args = parser.parse_args()
    scenes = [r.strip() for r in args.scene_list.read_text().splitlines()
              if r.strip() and not r.lstrip().startswith("#")]
    gts, aligns, transform = scannet_inputs()
    rows = []
    for scene in scenes:
        boxes, original = read(args.components / "native_native" / f"{scene}_boxes.pkl")
        reranked_boxes, reranked = read(args.components / "native_max" / f"{scene}_boxes.pkl")
        if not np.array_equal(boxes, reranked_boxes):
            raise RuntimeError(f"MVSR geometry mismatch: {scene}")
        aligned = transform(boxes, aligns[scene]) if len(boxes) else boxes
        tp = greedy_labels(aligned, original, gts[scene], args.iou)
        before, after = ranks(original), ranks(reranked)
        effective = np.maximum(0.0, (logit(reranked) - logit(original)) / 2.0)
        inferred = np.where(effective > ZERO_EPS, 0.5 + effective, np.nan)
        for index in range(len(boxes)):
            rows.append({
                "scene": scene, "row": index, "tp": bool(tp[index]),
                "original_score": float(original[index]),
                "reranked_score": float(reranked[index]),
                "effective_support": float(effective[index]),
                "inferred_support_if_updated": None if math.isnan(inferred[index]) else float(inferred[index]),
                "rank_before": int(before[index]), "rank_after": int(after[index]),
                "rank_gain": int(before[index] - after[index]),
            })

    bounds = ((0.0, 0.0), (0.0, 0.1), (0.1, 0.2), (0.2, 0.3), (0.3, math.inf))
    labels = ("0 (support <= 0.5)", "(0,0.1]", "(0.1,0.2]", "(0.2,0.3]", ">0.3")
    bins = []
    for (low, high), label in zip(bounds, labels):
        if low == high == 0:
            selected = [r for r in rows if r["effective_support"] <= ZERO_EPS]
        elif math.isinf(high):
            selected = [r for r in rows if r["effective_support"] > low]
        else:
            # Values in (0, ZERO_EPS] are numerical round-trip noise from
            # recovering Delta from serialized sigmoid/logit scores.  They
            # belong only to the zero-support bin above.
            lower = ZERO_EPS if low == 0.0 else low
            selected = [r for r in rows if lower < r["effective_support"] <= high]
        bins.append({
            "bin": label, "count": len(selected),
            "tp_rate": sum(r["tp"] for r in selected) / len(selected) if selected else None,
            "mean_rank_gain": float(np.mean([r["rank_gain"] for r in selected])) if selected else None,
        })
    groups = {}
    for name, value in (("tp", True), ("fp", False)):
        selected = [r for r in rows if r["tp"] is value]
        groups[name] = {
            "count": len(selected),
            "mean_effective_support": float(np.mean([r["effective_support"] for r in selected])),
            "mean_rank_gain": float(np.mean([r["rank_gain"] for r in selected])),
            "updated_fraction": sum(r["effective_support"] > ZERO_EPS for r in selected) / len(selected),
        }
    high_support_fp = sorted(
        (r for r in rows if not r["tp"] and r["effective_support"] > ZERO_EPS),
        key=lambda r: (-r["effective_support"], -r["rank_gain"], r["scene"], r["row"]),
    )[:50]
    if sum(row["count"] for row in bins) != len(rows):
        raise RuntimeError("effective-support bins do not partition native boxes")
    if sum(row["count"] for row in groups.values()) != len(rows):
        raise RuntimeError("TP/FP groups do not partition native boxes")
    result = {
        "schema": "boxfusion.recar3d.mvsr_ranking_analysis.v1",
        "iou": args.iou,
        "native_box_count": len(rows),
        "effective_support_definition": "Delta_i = max(0, S_i - tau_s), with tau_s=0.5; recovered from the logit update",
        "zero_tolerance": ZERO_EPS,
        "groups": groups,
        "bins": bins,
        "high_support_fp": high_support_fp,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    with (args.output / "per_native_box.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader(); writer.writerows(rows)
    lines = [
        "# MVSR-v2 ranking behavior on ScanNet-100", "",
        f"TP/FP labels use greedy one-to-one matching at IoU {args.iou:.2f} with original native scores.", "",
        "Effective support is defined as `Delta_i = max(0, S_i - tau_s)` with `tau_s=0.5`. Values no larger than `1e-10` are assigned to the zero bin to remove serialized sigmoid/logit round-trip noise.", "",
        "| Group | Boxes | Mean effective support | Updated fraction | Mean rank gain |",
        "|---|---:|---:|---:|---:|",
    ]
    for name in ("tp", "fp"):
        row = groups[name]
        lines.append(f"| {name.upper()} | {row['count']} | {row['mean_effective_support']:.4f} | {100*row['updated_fraction']:.2f}% | {row['mean_rank_gain']:.3f} |")
    lines += ["", "| Effective-support bin | Boxes | TP rate | Mean rank gain |", "|---|---:|---:|---:|"]
    for row in bins:
        rate = "n/a" if row["tp_rate"] is None else f"{100*row['tp_rate']:.2f}%"
        gain = "n/a" if row["mean_rank_gain"] is None else f"{row['mean_rank_gain']:.3f}"
        lines.append(f"| {row['bin']} | {row['count']} | {rate} | {gain} |")
    (args.output / "REPORT.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
