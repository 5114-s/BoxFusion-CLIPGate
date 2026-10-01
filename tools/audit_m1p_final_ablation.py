#!/usr/bin/env python3
"""Final-configuration ScanNet ablation for Proposal/Child-Level Recovery.

All arms reuse the same frozen native map, WeDetect+Boxer proposal cache and
native NMS-child cache.  The script performs no model inference.  GT is read
only after each GT-free arm has been materialized in memory for evaluation.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.audit_ca1m_nms_child_headroom import sha256, valid_boxes
from tools.audit_final_ledger import scannet_inputs
from tools.audit_m1m2_remaining_children import read_prediction
from tools.audit_seedless_ablation_matrix import RUNS, THRESHOLDS
from tools.true_fusion_audit_core import aabb_iou, class_agnostic_ap

NATIVE = ROOT / "results/scannet_t05_boxer_kfmap_score05"
PROPOSALS = ROOT / "results/wedetect_lifted_cache"
CHILDREN = ROOT / "results/child_recycle_cache"
CHILD_LEDGER = ROOT / "diagnostics/kfmap_score05"
ARCHIVED_FULL = ROOT / "results/causal_m1_only_v9"
DEDUP, CAP, TTL, SELF_NMS = 0.25, 12, 10, 0.50
EDGES = (0.3, 0.5, 0.7, 1.0)
PRICES = (0.05, 0.10, 0.25, 0.40, 0.50)


def iou(left: np.ndarray, right: np.ndarray) -> float:
    return float(aabb_iou(left[None], right[None])[0, 0])


def price(box: np.ndarray, fixed: bool = False) -> float:
    if fixed:
        return 0.10
    edge = float(np.ptp(box, axis=0).max())
    for boundary, score in zip(EDGES, PRICES):
        if edge < boundary:
            return score
    return PRICES[-1]


def load_source(path: Path, source: str) -> list[dict]:
    with np.load(path, allow_pickle=False) as z:
        boxes = valid_boxes(z["corners_raw"], str(path))
        scores = np.asarray(z["scores"], dtype=np.float64)
        frames = np.asarray(z["frame_ids"], dtype=np.int64)
    return [dict(box=box, score=float(score), frame=int(frame), source=source)
            for box, score, frame in zip(boxes, scores, frames)]


def load_children(scene: str) -> list[dict]:
    path = CHILD_LEDGER / f"{scene}_pvq_nms.jsonl"
    rows = []
    if path.exists():
        for line in path.read_text().splitlines():
            record = json.loads(line)
            rows.append(dict(
                box=np.asarray(record["child_corners_world"], dtype=np.float64),
                score=float(record["child_score"]),
                frame=int(record["keyframe_id"]), source="child"))
    return rows


def recover(candidates: list[dict], native: np.ndarray, *,
            confirmation: str = "source_aware", representative: str = "medoid",
            native_dedup: bool = True, fixed_score: bool = False) -> list[dict]:
    if not candidates:
        return []
    frames = np.asarray([row["frame"] for row in candidates])
    order = np.argsort(frames, kind="stable")
    _, ordinals = np.unique(frames, return_inverse=True)
    receipts = []
    for index in order:
        row = candidates[int(index)]
        box, frame = row["box"], int(ordinals[index])
        if native_dedup and len(native) and np.any(aabb_iou(box[None], native) >= DEDUP):
            continue
        best, best_iou = None, 0.0
        for receipt in receipts:
            if frame - receipt["last"] > TTL:
                continue
            previous = receipt["obs"][-1]["box"]
            if np.linalg.norm(previous.mean(0) - box.mean(0)) > 0.50:
                continue
            overlap = iou(box, previous)
            if overlap >= 0.10 and overlap > best_iou:
                best, best_iou = receipt, overlap
        if best is None:
            receipts.append({"obs": [row], "frames": {frame}, "last": frame})
        else:
            best["obs"].append(row)
            best["frames"].add(frame)
            best["last"] = frame

    births = []
    for receipt in receipts:
        obs = receipt["obs"]
        proposal_count = sum(row["source"] == "proposal" for row in obs)
        required = (3 if 2 * proposal_count > len(obs) else 2) \
            if confirmation == "source_aware" else int(confirmation[-1])
        if len(receipt["frames"]) < required:
            continue
        if representative == "highest_score":
            selected = max(range(len(obs)), key=lambda j: (obs[j]["score"], -j))
        else:
            selected = max(
                range(len(obs)),
                key=lambda j: sum(iou(obs[j]["box"], other["box"])
                                  for k, other in enumerate(obs) if k != j))
        strength = float(np.mean([row["score"] for row in obs]))
        chosen = obs[selected]
        births.append({"strength": strength, "box": chosen["box"],
                       "source": chosen["source"]})
    births.sort(key=lambda row: -row["strength"])
    kept = []
    for row in births:
        if any(iou(row["box"], old["box"]) >= SELF_NMS for old in kept):
            continue
        row = dict(row)
        row["output_score"] = price(row["box"], fixed=fixed_score)
        kept.append(row)
        if len(kept) >= CAP:
            break
    return kept


def naive_matched_budget(candidates: list[dict], native: np.ndarray,
                         reference: list[dict]) -> list[dict]:
    """No temporal confirmation; match the full arm's per-source output budget."""
    quota = {source: sum(row["source"] == source for row in reference)
             for source in ("proposal", "child")}
    kept = []
    for source in ("proposal", "child"):
        if quota[source] == 0:
            continue
        rows = [row for row in candidates if row["source"] == source]
        rows.sort(key=lambda row: -row["score"])
        selected = []
        for row in rows:
            if len(native) and np.any(aabb_iou(row["box"][None], native) >= DEDUP):
                continue
            if any(iou(row["box"], old["box"]) >= SELF_NMS for old in selected):
                continue
            selected.append(row)
            if len(selected) >= quota[source]:
                break
        for row in selected:
            kept.append({**row, "strength": row["score"],
                         "output_score": price(row["box"])})
    return kept


def metrics_table(result: dict) -> str:
    lines = [
        "# Final-Configuration M1-P Ablation",
        "",
        "All variants use the same frozen ScanNet native map and candidate caches. "
        "The full arm is checked against the archived final M1-P prediction geometry and scores.",
        "",
        "| Variant | Boxes | Births | AP15 | AP25 | AP50 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, block in result["arms"].items():
        m = block["metrics"]
        lines.append(
            f"| {name} | {block['boxes']} | {block['births']} | "
            f"{m['0.15']['ap']:.4f} | {m['0.25']['ap']:.4f} | {m['0.5']['ap']:.4f} |")
    lines += [
        "",
        "`naive_matched_budget` preserves the full arm's per-scene, per-source "
        "birth counts but removes temporal confirmation. Proposal and child scores "
        "are ranked only within their own source.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "reports/m1p_final_ablation_20260921")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    protocol = json.loads((RUNS["scannet"] / "protocol.json").read_text())
    scenes = protocol["scenes"]
    gts, aligns, transform = scannet_inputs()
    names = (
        "base", "proposal_only", "child_only", "proposal_child",
        "fixed_2frame", "fixed_3frame", "highest_score_exemplar",
        "without_native_dedup", "fixed_birth_score", "naive_matched_budget",
    )
    predictions = {name: {} for name in names}
    counts = {name: 0 for name in names}
    source_counts = {name: {"proposal": 0, "child": 0} for name in names}
    hashes = {}

    for ordinal, scene in enumerate(scenes, 1):
        native, native_scores = read_prediction(NATIVE / f"{scene}_boxes.pkl")
        proposals = load_source(PROPOSALS / f"{scene}.npz", "proposal")
        children = load_children(scene)
        joint = proposals + children
        variants = {
            "base": [],
            "proposal_only": recover(proposals, native),
            "child_only": recover(children, native),
            "proposal_child": recover(joint, native),
            "fixed_2frame": recover(joint, native, confirmation="fixed2"),
            "fixed_3frame": recover(joint, native, confirmation="fixed3"),
            "highest_score_exemplar": recover(
                joint, native, representative="highest_score"),
            "without_native_dedup": recover(joint, native, native_dedup=False),
            "fixed_birth_score": recover(joint, native, fixed_score=True),
        }
        variants["naive_matched_budget"] = naive_matched_budget(
            joint, native, variants["proposal_child"])
        for name, births in variants.items():
            birth_boxes = np.asarray([row["box"] for row in births], dtype=np.float64).reshape(-1, 8, 3)
            birth_scores = np.asarray([row["output_score"] for row in births], dtype=np.float64)
            boxes = np.concatenate([native, birth_boxes]) if len(births) else native
            scores = np.concatenate([native_scores, birth_scores]) if len(births) else native_scores
            predictions[name][scene] = (transform(boxes, aligns[scene]), scores)
            counts[name] += len(births)
            for row in births:
                source_counts[name][row["source"]] += 1

        archived_boxes, archived_scores = read_prediction(
            ARCHIVED_FULL / f"{scene}_boxes.pkl")
        rebuilt_boxes, rebuilt_scores = predictions["proposal_child"][scene]
        rebuilt_raw = np.concatenate([
            native,
            np.asarray([row["box"] for row in variants["proposal_child"]]).reshape(-1, 8, 3)
        ])
        rebuilt_raw_scores = np.concatenate([
            native_scores,
            np.asarray([row["output_score"] for row in variants["proposal_child"]])
        ])
        if not np.array_equal(archived_boxes, rebuilt_raw) or not np.array_equal(
                archived_scores, rebuilt_raw_scores):
            raise RuntimeError(f"full-arm replay differs from archive: {scene}")
        for path in (NATIVE / f"{scene}_boxes.pkl", PROPOSALS / f"{scene}.npz",
                     CHILD_LEDGER / f"{scene}_pvq_nms.jsonl",
                     ARCHIVED_FULL / f"{scene}_boxes.pkl"):
            hashes[str(path.resolve())] = sha256(path)
        print(f"[{ordinal:03d}/{len(scenes)}] {scene}", flush=True)

    result = {
        "schema": "boxfusion.m1p.final_ablation.v1",
        "dataset": "scannet", "scene_count": len(scenes),
        "parameters": {
            "native_dedup_iou": DEDUP, "cap_per_scene": CAP,
            "receipt_ttl_keyframes": TTL, "association_iou": 0.10,
            "association_center_m": 0.50, "self_nms_iou": SELF_NMS,
            "source_aware_confirmation": "proposal-majority:3 frames; otherwise:2 frames",
        },
        "arms": {}, "input_sha256": hashes,
        "gt_use": "evaluation only",
    }
    for name in names:
        result["arms"][name] = {
            "births": counts[name],
            "birth_sources": source_counts[name],
            "boxes": int(sum(len(rows[0]) for rows in predictions[name].values())),
            "metrics": {str(t): class_agnostic_ap(predictions[name], gts, t)
                        for t in THRESHOLDS},
        }
    (args.output / "results.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    (args.output / "REPORT.md").write_text(metrics_table(result) + "\n")
    print(args.output / "REPORT.md")


if __name__ == "__main__":
    main()
