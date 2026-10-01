#!/usr/bin/env python3
"""Audit final no-child Native/Proposal/Anchor evidence on ScanNet-100."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import pickle
import sys

import numpy as np

MAIN = Path("/data/ZhaoX/BoxFusion")
sys.path.insert(0, str(MAIN))
from tools.audit_ca1m_nms_child_headroom import pairwise_iou
from tools.audit_final_ledger import scannet_inputs
from tools.true_fusion_audit_core import class_agnostic_ap

THRESHOLDS = (0.15, 0.25, 0.50)


def read(path: Path):
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    rows = payload[0]
    return (
        np.asarray([row[1] for row in rows], dtype=np.float64).reshape(-1, 8, 3),
        np.asarray([row[2] for row in rows], dtype=np.float64),
    )


def maximum(candidates: np.ndarray, gt: np.ndarray) -> np.ndarray:
    if not len(gt):
        return np.empty(0, dtype=np.float64)
    if not len(candidates):
        return np.zeros(len(gt), dtype=np.float64)
    return pairwise_iou(candidates, gt).max(axis=0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-list", type=Path, required=True)
    parser.add_argument("--components", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--factorial", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    scenes = [r.strip() for r in args.scene_list.read_text().splitlines()
              if r.strip() and not r.lstrip().startswith("#")]
    if len(scenes) != 100 or len(set(scenes)) != 100:
        raise RuntimeError("expected ScanNet official100")
    gts, aligns, transform = scannet_inputs()
    quality = []
    candidate_counts = Counter()
    cost_predictions = {name: {} for name in ("plr_prefix", "full", "calr_prefix")}

    for index, scene in enumerate(scenes, 1):
        source_paths = {
            "native": args.components / "native_native" / f"{scene}_boxes.pkl",
            "plr": args.components / "births_plr_v1" / f"{scene}_boxes.pkl",
            "calr": args.components / "births_calr_v1" / f"{scene}_boxes.pkl",
        }
        sources = {name: read(path)[0] for name, path in source_paths.items()}
        with np.load(args.evidence / f"{scene}.npz", allow_pickle=False) as values:
            sources["proposal"] = np.asarray(values["proposal_corners"], dtype=np.float64)
            sources["anchor"] = np.asarray(values["anchor_corners"], dtype=np.float64)
        aligned = {
            name: transform(boxes, aligns[scene]) if len(boxes) else boxes
            for name, boxes in sources.items()
        }
        gt = gts[scene]
        q = {name: maximum(boxes, gt) for name, boxes in aligned.items()}
        for gt_index in range(len(gt)):
            quality.append({name: float(values[gt_index]) for name, values in q.items()})
        candidate_counts.update({name: len(boxes) for name, boxes in sources.items()})
        for name, arm in (("plr_prefix", "a_m2"), ("calr_prefix", "p_m2"), ("full", "p_a_m2")):
            boxes, scores = read(args.factorial / arm / f"{scene}_boxes.pkl")
            cost_predictions[name][scene] = (transform(boxes, aligns[scene]), scores)
        if index % 10 == 0:
            print(f"[{index}/100] {scene}", flush=True)

    thresholds = {}
    for threshold in THRESHOLDS:
        availability = Counter()
        recovery = Counter()
        signatures = Counter()
        for row in quality:
            a, p, n = (row[name] > threshold for name in ("anchor", "proposal", "native"))
            signatures[f"{int(a)}{int(p)}{int(n)}"] += 1
            category = "native_covered" if n else "proposal_evidence" if p else "anchor_only" if a else "no_evidence"
            availability[category] += 1
            if not n:
                plr, calr = row["plr"] > threshold, row["calr"] > threshold
                recovery["both" if plr and calr else "plr_only" if plr else "calr_only" if calr else "neither"] += 1
        thresholds[f"{threshold:.2f}"] = {
            "availability": {k: availability[k] for k in ("native_covered", "proposal_evidence", "anchor_only", "no_evidence")},
            "recovery_native_missed": {"total": sum(recovery.values()), **{k: recovery[k] for k in ("plr_only", "calr_only", "both", "neither")}},
            "signature_anchor_proposal_native": dict(sorted(signatures.items())),
        }

    metrics = {
        name: {str(t): class_agnostic_ap(pred, gts, t) for t in THRESHOLDS}
        for name, pred in cost_predictions.items()
    }
    costs = {}
    for branch, before in (("plr", "plr_prefix"), ("calr", "calr_prefix")):
        costs[branch] = {}
        for threshold in THRESHOLDS:
            key = str(threshold)
            costs[branch][f"{threshold:.2f}"] = {
                field: metrics["full"][key][field] - metrics[before][key][field]
                for field in ("ap", "tp", "fp")
            }
    result = {
        "schema": "boxfusion.recar3d.candidate_evidence_audit.v2",
        "scene_count": 100,
        "gt_count": len(quality),
        "children_used": False,
        "candidate_counts": dict(candidate_counts),
        "thresholds": thresholds,
        "conditional_recovery_cost": costs,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    lines = [
        "# ScanNet-100 final no-child candidate evidence audit", "",
        "Coverage is per-GT maximum 3D AABB IoU and is not ranked detection recall.", "",
        "| IoU | Native covered | Proposal evidence | Anchor only | No evidence |",
        "|---:|---:|---:|---:|---:|",
    ]
    for threshold, block in thresholds.items():
        row = block["availability"]
        lines.append(f"| {threshold} | {row['native_covered']} | {row['proposal_evidence']} | {row['anchor_only']} | {row['no_evidence']} |")
    lines += ["", "| IoU | Native missed | PLR only | CALR only | Both | Neither |", "|---:|---:|---:|---:|---:|---:|"]
    for threshold, block in thresholds.items():
        row = block["recovery_native_missed"]
        lines.append(f"| {threshold} | {row['total']} | {row['plr_only']} | {row['calr_only']} | {row['both']} | {row['neither']} |")
    (args.output / "REPORT.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
