#!/usr/bin/env python3
"""Three-source candidate coverage and recovery audit for final online ScanNet.

Sources are the current native map, live post-NMS proposals, and live top-300
pre-NMS anchors. Child evidence is deliberately excluded. Ground truth is read
only by this retrospective audit.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import pickle
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.audit_ca1m_nms_child_headroom import pairwise_iou
from tools.audit_final_ledger import scannet_inputs
from tools.true_fusion_audit_core import class_agnostic_ap


THRESHOLDS = (0.15, 0.25, 0.50)


def read(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    if not isinstance(payload, (list, tuple)) or len(payload) != 1:
        raise RuntimeError(f"invalid prediction container: {path}")
    boxes, scores = [], []
    for row in payload[0]:
        if len(row) != 3 or int(row[0]) != 0:
            raise RuntimeError(f"invalid class-agnostic prediction: {path}")
        boxes.append(np.asarray(row[1], dtype=np.float64).reshape(8, 3))
        scores.append(float(row[2]))
    return (
        np.asarray(boxes, dtype=np.float64).reshape(-1, 8, 3),
        np.asarray(scores, dtype=np.float64),
    )


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            value.update(chunk)
    return value.hexdigest()


def max_iou(candidates: np.ndarray, gt: np.ndarray) -> np.ndarray:
    if not len(gt):
        return np.empty(0, dtype=np.float64)
    if not len(candidates):
        return np.zeros(len(gt), dtype=np.float64)
    return pairwise_iou(candidates, gt).max(axis=0)


def evidence_category(anchor: bool, proposal: bool, native: bool) -> str:
    if native:
        return "native_covered"
    if proposal:
        return "proposal_evidence"
    if anchor:
        return "anchor_only"
    return "no_evidence"


def markdown(result: dict) -> str:
    lines = [
        "# Final-online no-child candidate evidence audit",
        "",
        "This is per-GT candidate-set coverage, not confidence-ranked, "
        "one-to-one detection recall. Child evidence is excluded.",
        "",
        f"Scenes: {result['scene_count']}; GT: {result['gt_count']}.",
        "",
        "## Candidate evidence availability",
        "",
        "| IoU | Native covered | Proposal evidence | Anchor only | No evidence |",
        "|---:|---:|---:|---:|---:|",
    ]
    for threshold, block in result["thresholds"].items():
        row = block["evidence_categories"]
        lines.append(
            f"| {threshold} | {row['native_covered']} | "
            f"{row['proposal_evidence']} | {row['anchor_only']} | "
            f"{row['no_evidence']} |"
        )
    lines += [
        "",
        "## Recovery attribution among native-missed GT",
        "",
        "| IoU | Native missed | PLR only | CALR only | Both | Neither |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for threshold, block in result["thresholds"].items():
        row = block["recovery_native_missed"]
        lines.append(
            f"| {threshold} | {row['total']} | {row['plr_only']} | "
            f"{row['calr_only']} | {row['both']} | {row['neither']} |"
        )
    lines += [
        "",
        "## Recovery cost",
        "",
        "| Branch | IoU | Delta TP | Delta FP | Added TP:FP |",
        "|---|---:|---:|---:|---:|",
    ]
    for branch in ("plr", "calr"):
        for threshold, row in result["recovery_cost"][branch].items():
            delta_tp, delta_fp = int(row["delta_tp"]), int(row["delta_fp"])
            ratio = (
                f"1:{delta_fp / delta_tp:.1f}"
                if delta_tp > 0 and delta_fp >= 0
                else "n/a"
            )
            lines.append(
                f"| {branch.upper()} | {threshold} | {delta_tp:+d} | "
                f"{delta_fp:+d} | {ratio} |"
            )
    lines += [
        "",
        "Matching uses strict `IoU > threshold`. The machine-readable result "
        "retains every GT's maximum IoU to all five audited sets.",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-list", type=Path, required=True)
    parser.add_argument("--native", type=Path, required=True)
    parser.add_argument("--full", type=Path, required=True)
    parser.add_argument("--diagnostics", type=Path, required=True)
    parser.add_argument("--evidence-cache", type=Path, required=True)
    parser.add_argument("--factorial", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    scenes = [line.strip() for line in args.scene_list.read_text().splitlines()
              if line.strip() and not line.lstrip().startswith("#")]
    if len(scenes) != 100 or len(set(scenes)) != 100:
        raise RuntimeError("expected the official 100 unique ScanNet scenes")
    gts, aligns, transform = scannet_inputs()
    per_gt = []
    counts = Counter()
    hashes = {}
    predictions = {name: {} for name in ("native", "plr", "calr_prefix", "calr")}

    for ordinal, scene in enumerate(scenes, 1):
        native_path = args.native / f"{scene}_boxes.pkl"
        full_path = args.full / f"{scene}_boxes.pkl"
        diagnostic_path = args.diagnostics / f"{scene}.json"
        cache_path = args.evidence_cache / f"{scene}.npz"
        native, native_scores = read(native_path)
        full, full_scores = read(full_path)
        diagnostic = json.loads(diagnostic_path.read_text(encoding="utf-8"))
        n, p, a = (
            int(diagnostic["state"]["terminal_counts"][key])
            for key in ("native", "m1p", "m1a")
        )
        if n != len(native) or n + p + a != len(full):
            raise RuntimeError(f"component-count mismatch: {scene}")
        plr_births = full[n:n + p]
        plr_scores = full_scores[n:n + p]
        calr_births = full[n + p:]
        calr_scores = full_scores[n + p:]
        with np.load(cache_path, allow_pickle=False) as values:
            anchors = np.asarray(values["corners_raw"], dtype=np.float64)
            proposals = np.asarray(
                values["proposal_corners_raw"], dtype=np.float64
            )

        sources = {
            "anchor": anchors,
            "proposal": proposals,
            "native": native,
            "plr": plr_births,
            "calr": calr_births,
        }
        aligned = {
            name: transform(boxes, aligns[scene]) if len(boxes) else boxes
            for name, boxes in sources.items()
        }
        gt = gts[scene]
        quality = {name: max_iou(boxes, gt) for name, boxes in aligned.items()}
        counts.update({name: len(boxes) for name, boxes in sources.items()})
        for index in range(len(gt)):
            per_gt.append({
                "gt_id": f"{scene}:{index}",
                "scene": scene,
                **{
                    f"q_{name}": float(values[index])
                    for name, values in quality.items()
                },
            })

        predictions["native"][scene] = (aligned["native"], native_scores)
        predictions["plr"][scene] = (
            transform(np.concatenate([native, plr_births]), aligns[scene]),
            np.concatenate([native_scores, plr_scores]),
        )
        for name, arm in (("calr_prefix", "p_m2"), ("calr", "p_a_m2")):
            arm_path = args.factorial / arm / f"{scene}_boxes.pkl"
            boxes, scores = read(arm_path)
            predictions[name][scene] = (transform(boxes, aligns[scene]), scores)
            hashes[str(arm_path.resolve())] = digest(arm_path)
        for path in (native_path, full_path, diagnostic_path, cache_path):
            hashes[str(path.resolve())] = digest(path)
        if ordinal % 10 == 0:
            print(f"[{ordinal:03d}/100] {scene}", flush=True)

    thresholds = {}
    for threshold in THRESHOLDS:
        categories, signatures, recovery = Counter(), Counter(), Counter()
        for row in per_gt:
            anchor = row["q_anchor"] > threshold
            proposal = row["q_proposal"] > threshold
            native = row["q_native"] > threshold
            signatures[f"{int(anchor)}{int(proposal)}{int(native)}"] += 1
            categories[evidence_category(anchor, proposal, native)] += 1
            if not native:
                plr = row["q_plr"] > threshold
                calr = row["q_calr"] > threshold
                recovery[
                    "both" if plr and calr else
                    "plr_only" if plr else
                    "calr_only" if calr else "neither"
                ] += 1
        thresholds[f"{threshold:.2f}"] = {
            "signatures_APN": dict(sorted(signatures.items())),
            "evidence_categories": {
                key: categories[key]
                for key in (
                    "native_covered", "proposal_evidence",
                    "anchor_only", "no_evidence",
                )
            },
            "recovery_native_missed": {
                "total": sum(recovery.values()),
                **{
                    key: recovery[key]
                    for key in ("plr_only", "calr_only", "both", "neither")
                },
            },
        }

    metrics = {
        name: {
            str(threshold): class_agnostic_ap(rows, gts, threshold)
            for threshold in THRESHOLDS
        }
        for name, rows in predictions.items()
    }
    recovery_cost = {}
    for branch, before, after in (
        ("plr", "native", "plr"),
        ("calr", "calr_prefix", "calr"),
    ):
        recovery_cost[branch] = {}
        for threshold in THRESHOLDS:
            key = str(threshold)
            recovery_cost[branch][f"{threshold:.2f}"] = {
                "delta_ap": metrics[after][key]["ap"] - metrics[before][key]["ap"],
                "delta_tp": metrics[after][key]["tp"] - metrics[before][key]["tp"],
                "delta_fp": metrics[after][key]["fp"] - metrics[before][key]["fp"],
            }

    result = {
        "schema": "boxfusion.current_candidate_evidence.v1",
        "dataset": "ScanNet official100",
        "scene_count": len(scenes),
        "gt_count": len(per_gt),
        "protocol": {
            "sources": ["native", "post-NMS proposal", "pre-NMS top-300 anchor"],
            "child_evidence": False,
            "matching": "per-GT maximum aligned 3D AABB IoU; strict > threshold",
            "coverage_is_not_ranked_recall": True,
            "gt_use": "retrospective audit and metrics only",
        },
        "candidate_counts": dict(counts),
        "thresholds": thresholds,
        "recovery_cost": recovery_cost,
        "metrics_for_cost_only": metrics,
        "per_gt": per_gt,
        "input_sha256": hashes,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "results.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (args.output / "REPORT.md").write_text(markdown(result) + "\n", encoding="utf-8")
    print(args.output / "REPORT.md")


if __name__ == "__main__":
    main()
