#!/usr/bin/env python3
"""Audit multi-source candidate evidence on the final ScanNet paper split.

The audit is deliberately candidate-set based.  For every GT box it records
the best 3D AABB IoU supplied by the exact inputs exposed to the submitted
method: lifted top-300 anchors, lifted post-NMS WeDetect proposals, native
NMS-suppressed children, and the strong native map.  It separately records
coverage by the final M1-P and strict-online M1-A births.

GT is read only by this retrospective audit.  A candidate may cover multiple
GT boxes, so these counts are evidence availability, not AP or one-to-one
detection recall.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.audit_ca1m_nms_child_headroom import pairwise_iou, sha256, valid_boxes
from tools.audit_final_ledger import scannet_inputs
from tools.audit_m1m2_remaining_children import read_prediction
from tools.audit_seedless_ablation_matrix import RUNS

THRESHOLDS = (0.15, 0.25, 0.50)
ANCHOR_POOL = "score_m300"
NATIVE = ROOT / "results/scannet_t05_boxer_kfmap_score05"
PROPOSALS = ROOT / "results/wedetect_lifted_cache"
CHILDREN = ROOT / "results/child_recycle_cache"
CHILD_LEDGER = ROOT / "diagnostics/kfmap_score05"
M1P = ROOT / "results/causal_m1_only_v9"
M1P_M2 = ROOT / "results/scannet_m2nl_m5_dual_full100/persistent"
FINAL = ROOT / "reports/m1a_strict_online_scannet_20260916/predictions"


def max_iou(candidates: np.ndarray, gt: np.ndarray) -> np.ndarray:
    if not len(gt):
        return np.empty(0, dtype=np.float64)
    if not len(candidates):
        return np.zeros(len(gt), dtype=np.float64)
    return pairwise_iou(candidates, gt).max(0)


def load_anchor_candidates(run: Path, scene: str) -> np.ndarray:
    directory = run / "scenes" / scene
    selection = json.loads((directory / "selection.json").read_text())
    blocks = []
    for row in selection["frames"]:
        frame = int(row["frame"])
        ids = np.asarray(row["selected_anchor_ids"][ANCHOR_POOL], dtype=np.int64)
        with np.load(directory / f"lifted_{frame:06d}.npz", allow_pickle=False) as z:
            union = z["anchor_ids"].astype(np.int64)
            positions = np.searchsorted(union, ids)
            if not np.array_equal(union[positions], ids):
                raise RuntimeError(f"anchor mismatch: {scene}/{frame}")
            blocks.append(z["corners"][positions].astype(np.float64))
    return np.concatenate(blocks) if blocks else np.empty((0, 8, 3), dtype=np.float64)


def load_npz_boxes(path: Path) -> np.ndarray:
    with np.load(path, allow_pickle=False) as z:
        return valid_boxes(z["corners_raw"], str(path))


def load_children(scene: str) -> np.ndarray:
    path = CHILD_LEDGER / f"{scene}_pvq_nms.jsonl"
    rows = [json.loads(line)["child_corners_world"]
            for line in path.read_text().splitlines()] if path.exists() else []
    return valid_boxes(rows, str(path))


def append_births(full_path: Path, prefix_path: Path) -> np.ndarray:
    full, _ = read_prediction(full_path)
    prefix, _ = read_prediction(prefix_path)
    if len(full) < len(prefix) or not np.array_equal(full[:len(prefix)], prefix):
        raise RuntimeError(f"prediction prefix mismatch: {full_path}")
    return full[len(prefix):]


def category(bits: tuple[bool, bool, bool, bool]) -> str:
    a, p, c, n = bits
    if n:
        return "native_matched"
    if p or c:
        return "explicit_non_native"
    if a:
        return "anchor_only"
    return "no_evidence"


def markdown(result: dict) -> str:
    lines = [
        "# Multi-Source Candidate Evidence Audit",
        "",
        "This is a per-GT candidate-set coverage audit, not AP or one-to-one recall. "
        "All candidate sources are valid lifted 3D boxes in the final ScanNet configuration.",
        "",
        f"Scenes: {result['scene_count']}; GT: {result['gt_count']}; "
        f"anchor pool: `{result['protocol']['anchor_pool']}`.",
        "",
        "## Evidence availability",
        "",
        "| IoU | No evidence | Anchor only | Explicit non-native | Native matched |",
        "|---:|---:|---:|---:|---:|",
    ]
    for threshold, block in result["thresholds"].items():
        c = block["categories"]
        lines.append(
            f"| {threshold} | {c['no_evidence']} | {c['anchor_only']} | "
            f"{c['explicit_non_native']} | {c['native_matched']} |")
    lines += [
        "",
        "`Explicit non-native` means proposal and/or suppressed-child evidence exists "
        "while the native map does not cover that GT at the same threshold.",
        "",
        "## Recovery attribution among native-missed GT",
        "",
        "| IoU | Native missed | M1-P only | M1-A only | Both | Neither |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for threshold, block in result["thresholds"].items():
        r = block["recovery_native_missed"]
        lines.append(
            f"| {threshold} | {r['total']} | {r['m1p_only']} | {r['m1a_only']} | "
            f"{r['both']} | {r['neither']} |")
    lines += [
        "",
        "The machine-readable JSON retains all 16 evidence signatures and per-GT "
        "maximum IoUs. Matching uses the same strict `IoU > threshold` convention as "
        "the paper evaluator.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "reports/multisource_candidate_evidence_20260921")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    run = RUNS["scannet"]
    protocol = json.loads((run / "protocol.json").read_text())
    scenes = protocol["scenes"]
    gts, aligns, transform = scannet_inputs()
    per_gt = []
    candidate_counts = Counter()
    inputs = {}

    for ordinal, scene in enumerate(scenes, 1):
        native, _ = read_prediction(NATIVE / f"{scene}_boxes.pkl")
        anchors = load_anchor_candidates(run, scene)
        proposals = load_npz_boxes(PROPOSALS / f"{scene}.npz")
        children = load_children(scene)
        m1p_births = append_births(
            M1P / f"{scene}_boxes.pkl", NATIVE / f"{scene}_boxes.pkl")
        m1a_births = append_births(
            FINAL / f"{scene}_boxes.pkl", M1P_M2 / f"{scene}_boxes.pkl")
        sources = {
            "anchor": anchors, "proposal": proposals, "child": children,
            "native": native, "m1p": m1p_births, "m1a": m1a_births,
        }
        transformed = {
            name: transform(boxes, aligns[scene]) if len(boxes) else boxes
            for name, boxes in sources.items()
        }
        gt = gts[scene]
        quality = {name: max_iou(boxes, gt) for name, boxes in transformed.items()}
        for name, boxes in sources.items():
            candidate_counts[name] += len(boxes)
        for index in range(len(gt)):
            per_gt.append({
                "gt_id": f"{scene}:{index}", "scene": scene,
                **{f"q_{name}": float(values[index])
                   for name, values in quality.items()},
            })
        for path in (
            NATIVE / f"{scene}_boxes.pkl", PROPOSALS / f"{scene}.npz",
            CHILD_LEDGER / f"{scene}_pvq_nms.jsonl", M1P / f"{scene}_boxes.pkl",
            M1P_M2 / f"{scene}_boxes.pkl", FINAL / f"{scene}_boxes.pkl",
            run / "scenes" / scene / "selection.json"):
            inputs[str(path.resolve())] = sha256(path)
        print(f"[{ordinal:03d}/{len(scenes)}] {scene}", flush=True)

    thresholds = {}
    for threshold in THRESHOLDS:
        signatures, categories, recovery = Counter(), Counter(), Counter()
        for row in per_gt:
            bits = tuple(row[f"q_{name}"] > threshold
                         for name in ("anchor", "proposal", "child", "native"))
            signature = "".join("1" if value else "0" for value in bits)
            signatures[signature] += 1
            categories[category(bits)] += 1
            if not bits[3]:
                p = row["q_m1p"] > threshold
                a = row["q_m1a"] > threshold
                recovery["both" if p and a else
                         "m1p_only" if p else
                         "m1a_only" if a else "neither"] += 1
        thresholds[f"{threshold:.2f}"] = {
            "signatures_APCN": dict(sorted(signatures.items())),
            "categories": {key: categories[key] for key in
                           ("no_evidence", "anchor_only",
                            "explicit_non_native", "native_matched")},
            "recovery_native_missed": {
                "total": sum(recovery.values()),
                **{key: recovery[key] for key in
                   ("m1p_only", "m1a_only", "both", "neither")},
            },
        }

    result = {
        "schema": "boxfusion.multisource_candidate_evidence.v1",
        "dataset": "scannet", "scene_count": len(scenes),
        "gt_count": len(per_gt),
        "protocol": {
            "anchor_pool": ANCHOR_POOL,
            "candidate_space": "valid lifted 3D boxes in aligned world coordinates",
            "matching": "per-GT maximum evaluator-equivalent AABB IoU; strict > threshold",
            "coverage_is_not_ap": True,
            "gt_use": "retrospective audit only",
        },
        "candidate_counts": dict(candidate_counts),
        "thresholds": thresholds,
        "per_gt": per_gt,
        "input_sha256": inputs,
    }
    (args.output / "results.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    (args.output / "REPORT.md").write_text(markdown(result) + "\n")
    print(args.output / "REPORT.md")


if __name__ == "__main__":
    main()
