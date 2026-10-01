#!/usr/bin/env python3
"""Paired official100 evaluation of proposal-only M1-P + M2 + online M1-A.

The experiment changes one factor relative to the sealed final route: M1-P
receives post-NMS proposals only.  It reuses the same native maps, frozen
proposal cache, M1-P rules, native-logit M2 rule, and sealed causal M1-A
births.  M2 support is recomputed because exclusive proposal assignment
depends on the M1-P output context.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import pickle
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.audit_ca1m_nms_child_headroom import sha256
from tools.audit_final_ledger import scannet_inputs
from tools.audit_m1m2_remaining_children import read_prediction
from tools.audit_m1p_final_ablation import load_source, recover
from tools.audit_m2_support_reranking import support_statistics, logit_update
from tools.audit_seedless_ablation_matrix import RUNS, THRESHOLDS
from tools.true_fusion_audit_core import class_agnostic_ap


NATIVE = ROOT / "results/scannet_t05_boxer_kfmap_score05"
PROPOSALS = ROOT / "results/wedetect_lifted_cache"
OLD_PREFIX = ROOT / "results/scannet_m2nl_m5_dual_full100/persistent"
OLD_FULL = ROOT / "reports/m1a_strict_online_scannet_20260916/predictions"
SCANS = Path("/extra/ZhaoX/scannet_data/scans")
ARM_SPECS = {
    "base": (False, False, False),
    "p": (True, False, False),
    "a": (False, True, False),
    "p_a": (True, True, False),
    "m2": (False, False, True),
    "p_m2": (True, False, True),
    "a_m2": (False, True, True),
    "p_a_m2": (True, True, True),
}


def write_prediction(path: Path, boxes: np.ndarray, scores: np.ndarray) -> None:
    rows = [(0, box, float(score)) for box, score in zip(boxes, scores)]
    with path.open("wb") as handle:
        pickle.dump([rows], handle)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "reports/scannet_nochild_full100_20260922")
    args = parser.parse_args()
    if args.output.exists():
        raise RuntimeError(f"refusing to overwrite {args.output}")
    prediction_root = args.output / "predictions"
    for arm in ARM_SPECS:
        (prediction_root / arm).mkdir(parents=True, exist_ok=True)

    protocol_path = RUNS["scannet"] / "protocol.json"
    protocol = json.loads(protocol_path.read_text())
    scenes = protocol["scenes"]
    if len(scenes) != 100 or len(set(scenes)) != 100:
        raise RuntimeError("expected the sealed ScanNet official100 scene list")
    gts, aligns, transform = scannet_inputs()
    raw_predictions = {arm: {} for arm in ARM_SPECS}
    counts = {arm: 0 for arm in ARM_SPECS}
    component_counts = {"native": 0, "m1p_proposal": 0, "m1a": 0}
    hashes = {str(protocol_path.resolve()): sha256(protocol_path)}

    for ordinal, scene in enumerate(scenes, 1):
        native_path = NATIVE / f"{scene}_boxes.pkl"
        proposal_path = PROPOSALS / f"{scene}.npz"
        old_prefix_path = OLD_PREFIX / f"{scene}_boxes.pkl"
        old_full_path = OLD_FULL / f"{scene}_boxes.pkl"
        native_boxes, native_scores = read_prediction(native_path)

        proposal_rows = load_source(proposal_path, "proposal")
        births = recover(proposal_rows, native_boxes)
        p_boxes = np.asarray([row["box"] for row in births], dtype=np.float64).reshape(-1, 8, 3)
        p_scores = np.asarray([row["output_score"] for row in births], dtype=np.float64)

        old_prefix_boxes, old_prefix_scores = read_prediction(old_prefix_path)
        old_full_boxes, old_full_scores = read_prediction(old_full_path)
        prefix_rows = len(old_prefix_boxes)
        if (len(old_full_boxes) < prefix_rows
                or not np.array_equal(old_full_boxes[:prefix_rows], old_prefix_boxes)
                or not np.array_equal(old_full_scores[:prefix_rows], old_prefix_scores)):
            raise RuntimeError(f"sealed M1-A prefix mismatch: {scene}")
        a_boxes = old_full_boxes[prefix_rows:]
        a_scores = old_full_scores[prefix_rows:]

        with np.load(proposal_path, allow_pickle=False) as values:
            boxes2d = np.asarray(values["boxes2d"], dtype=np.float64).reshape(-1, 4)
            frame_ids = np.asarray(values["frame_ids"], dtype=np.int64)
        by_frame = {int(frame): boxes2d[frame_ids == frame]
                    for frame in np.unique(frame_ids)}
        intrinsic = np.loadtxt(
            SCANS / scene / "intrinsic/intrinsic_color.txt")[:3, :3]
        pose_cache: dict[int, np.ndarray | None] = {}

        def pose_for(frame: int) -> np.ndarray | None:
            if frame not in pose_cache:
                value = np.loadtxt(
                    SCANS / scene / f"pose/{frame}.txt").reshape(4, 4)
                pose_cache[frame] = value if np.isfinite(value).all() else None
            return pose_cache[frame]

        support_context = (
            np.concatenate([native_boxes, p_boxes]) if len(p_boxes)
            else native_boxes
        )
        support = support_statistics(
            support_context, by_frame, pose_for, intrinsic,
            width=1296, height=968, exclusive=True)["max"]
        reranked_native_scores = logit_update(
            native_scores, support[:len(native_boxes)])

        for arm, (with_p, with_a, with_m2) in ARM_SPECS.items():
            boxes = [native_boxes]
            scores = [reranked_native_scores if with_m2 else native_scores]
            if with_p:
                boxes.append(p_boxes)
                scores.append(p_scores)
            if with_a:
                boxes.append(a_boxes)
                scores.append(a_scores)
            arm_boxes = np.concatenate(boxes) if len(boxes) > 1 else boxes[0]
            arm_scores = np.concatenate(scores) if len(scores) > 1 else scores[0]
            raw_predictions[arm][scene] = (
                transform(arm_boxes, aligns[scene]), arm_scores)
            write_prediction(
                prediction_root / arm / f"{scene}_boxes.pkl",
                arm_boxes, arm_scores)
            counts[arm] += len(arm_boxes)

        component_counts["native"] += len(native_boxes)
        component_counts["m1p_proposal"] += len(p_boxes)
        component_counts["m1a"] += len(a_boxes)
        for path in (native_path, proposal_path, old_prefix_path, old_full_path):
            hashes[str(path.resolve())] = sha256(path)
        print(
            f"[{ordinal:03d}/100] {scene} "
            f"native/p/a={len(native_boxes)}/{len(p_boxes)}/{len(a_boxes)}",
            flush=True)

    metrics = {
        arm: {str(threshold): class_agnostic_ap(rows, gts, threshold)
              for threshold in THRESHOLDS}
        for arm, rows in raw_predictions.items()
    }
    expected_p = (40.05114624013554, 35.93046904776405, 17.349231454133047)
    actual_p = tuple(metrics["p"][str(value)]["ap"] for value in THRESHOLDS)
    if not np.allclose(actual_p, expected_p, atol=5e-10):
        raise RuntimeError(("proposal-only anchor changed", actual_p, expected_p))

    result = {
        "schema": "boxfusion.scannet.nochild_full100.v1",
        "dataset": "scannet",
        "scene_count": 100,
        "single_changed_factor": "remove NMS-suppressed children from M1-P",
        "m1p_sources": ["post-NMS WeDetect proposals"],
        "m2_recomputed": True,
        "m2_context": "native rows plus proposal-only M1-P births; exclusive per-view matching",
        "m1a_births": "unchanged sealed strict-causal online births",
        "component_counts": component_counts,
        "arm_counts": counts,
        "metrics": metrics,
        "input_sha256": hashes,
        "gt_use": "evaluation only",
    }
    (args.output / "results.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    lines = [
        "# ScanNet official100: no-child paired full combination",
        "",
        "Only the NMS-suppressed child source is removed from M1-P. M2 is "
        "recomputed under the proposal-only context; the sealed causal M1-A "
        "births are unchanged.",
        "",
        "| Arm | Boxes | AP15 | AP25 | AP50 |",
        "|---|---:|---:|---:|---:|",
    ]
    for arm in ARM_SPECS:
        values = [metrics[arm][str(t)]["ap"] for t in THRESHOLDS]
        lines.append(
            f"| {arm} | {counts[arm]} | {values[0]:.4f} | "
            f"{values[1]:.4f} | {values[2]:.4f} |")
    (args.output / "REPORT.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
