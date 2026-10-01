#!/usr/bin/env python3
"""Direct evidence and operator ablation for final native-map M2 reranking.

The same native geometry, labels and row count are used in every arm.  Cached
post-NMS 2D proposals are projected/matched without model inference.  The
script reports AP for score-update alternatives and reliability tables that
condition native correctness on both native-score and cross-view support.
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
from tools.audit_m1m2_remaining_children import GT_ROOT, read_prediction
from tools.audit_seedless_ablation_matrix import RUNS, THRESHOLDS
from tools.true_fusion_audit_core import aabb_iou, class_agnostic_ap

SCANNET_NATIVE = ROOT / "results/scannet_t05_boxer_kfmap_score05"
SCANNET_PROPOSALS = ROOT / "results/wedetect_lifted_cache"
SCANNET_SIDECAR = ROOT / "results/scannet_m2nl_m5_dual_full100/persistent"
SCANNET_M1P = ROOT / "results/causal_m1_only_v9"
SCANS = Path("/extra/ZhaoX/scannet_data/scans")

CA_NATIVE = ROOT / "results/ca1m_thr15"
CA_PROPOSALS = ROOT / "reports/ca1m_factorial8_20260916/proposal_cache"
CA_DATA = Path("/extra/ZhaoX/boxfusion_ca1m")
TAU, LOGIT_STRENGTH, ADDITIVE_ALPHA = 0.5, 2.0, 0.8


def project(corners: np.ndarray, pose: np.ndarray, intrinsic: np.ndarray,
            width: int, height: int) -> np.ndarray | None:
    inverse = np.linalg.inv(pose)
    camera = corners @ inverse[:3, :3].T + inverse[:3, 3]
    if np.any(camera[:, 2] < 0.1):
        return None
    pixels = camera @ intrinsic.T
    pixels = pixels[:, :2] / np.maximum(pixels[:, 2:3], 1e-9)
    lo, hi = pixels.min(0), pixels.max(0)
    if hi[0] <= 0 or hi[1] <= 0 or lo[0] >= width or lo[1] >= height:
        return None
    box = np.asarray([max(0.0, lo[0]), max(0.0, lo[1]),
                      min(float(width), hi[0]), min(float(height), hi[1])])
    if box[2] - box[0] < 2 or box[3] - box[1] < 2:
        return None
    return box


def iou2d(box: np.ndarray, proposals: np.ndarray) -> float:
    if not len(proposals):
        return 0.0
    lo = np.maximum(box[:2], proposals[:, :2])
    hi = np.minimum(box[2:], proposals[:, 2:])
    inter = np.maximum(0.0, hi - lo).prod(1)
    union = ((box[2] - box[0]) * (box[3] - box[1])
             + (proposals[:, 2] - proposals[:, 0])
             * (proposals[:, 3] - proposals[:, 1]) - inter)
    return float(np.max(inter / np.maximum(union, 1e-9)))


def support_statistics(boxes: np.ndarray, by_frame: dict[int, np.ndarray],
                       pose_for, intrinsic: np.ndarray, width: int,
                       height: int, exclusive: bool) -> dict[str, np.ndarray]:
    maximum = np.zeros(len(boxes), dtype=np.float64)
    mean = np.zeros(len(boxes), dtype=np.float64)
    first = np.zeros(len(boxes), dtype=np.float64)
    counts = np.zeros(len(boxes), dtype=np.int64)
    sums = np.zeros(len(boxes), dtype=np.float64)
    first_set = np.zeros(len(boxes), dtype=bool)
    for frame in sorted(by_frame):
        proposals = by_frame[frame]
        pose = pose_for(frame)
        if pose is None:
            continue
        visible, projections = [], {}
        for index, box in enumerate(boxes):
            projection = project(box, pose, intrinsic, width, height)
            if projection is None:
                continue
            visible.append(index)
            projections[index] = projection
        assigned = {}
        if exclusive:
            pairs = []
            for index in visible:
                box = projections[index]
                lo = np.maximum(box[:2], proposals[:, :2])
                hi = np.minimum(box[2:], proposals[:, 2:])
                inter = np.maximum(0.0, hi - lo).prod(1)
                union = ((box[2] - box[0]) * (box[3] - box[1])
                         + (proposals[:, 2] - proposals[:, 0])
                         * (proposals[:, 3] - proposals[:, 1]) - inter)
                values = inter / np.maximum(union, 1e-9)
                pairs.extend((float(values[j]), index, int(j))
                             for j in np.flatnonzero(values >= 0.10))
            pairs.sort(reverse=True)
            used_boxes, used_proposals = set(), set()
            for value, index, proposal in pairs:
                if index in used_boxes or proposal in used_proposals:
                    continue
                used_boxes.add(index)
                used_proposals.add(proposal)
                assigned[index] = value
        for index in visible:
            value = assigned.get(index, 0.0) if exclusive else iou2d(
                projections[index], proposals)
            maximum[index] = max(maximum[index], value)
            sums[index] += value
            counts[index] += 1
            if not first_set[index]:
                first[index] = value
                first_set[index] = True
    np.divide(sums, counts, out=mean, where=counts > 0)
    return {"max": maximum, "mean": mean, "first": first,
            "visible_views": counts}


def logit_update(scores: np.ndarray, support: np.ndarray) -> np.ndarray:
    clipped = np.clip(scores.astype(np.float64), 1e-4, 1 - 1e-4)
    logits = np.log(clipped / (1 - clipped))
    logits += LOGIT_STRENGTH * np.maximum(0.0, support - TAU)
    return 1.0 / (1.0 + np.exp(-logits))


def additive(scores: np.ndarray, support: np.ndarray) -> np.ndarray:
    return np.minimum(0.99, scores + ADDITIVE_ALPHA * np.maximum(0.0, support - TAU))


def original_tp_labels(predictions: dict, gts: dict, threshold: float) -> dict[str, np.ndarray]:
    entries = []
    overlaps = {}
    for scene, (boxes, scores) in predictions.items():
        overlaps[scene] = aabb_iou(boxes, gts[scene])
        entries.extend((float(score), scene, index)
                       for index, score in enumerate(scores))
    entries.sort(key=lambda row: -row[0])
    taken = {scene: np.zeros(len(gts[scene]), dtype=bool) for scene in gts}
    labels = {scene: np.zeros(len(rows[0]), dtype=bool)
              for scene, rows in predictions.items()}
    for _, scene, index in entries:
        values = overlaps[scene][index]
        if len(values) and values.max() > threshold:
            target = int(values.argmax())
            if not taken[scene][target]:
                labels[scene][index] = True
                taken[scene][target] = True
    return labels


def load_dataset(dataset: str):
    protocol = json.loads((RUNS[dataset] / "protocol.json").read_text())
    scenes = protocol["scenes"]
    if dataset == "scannet":
        gts, aligns, transform = scannet_inputs()
        native_root, proposal_root = SCANNET_NATIVE, SCANNET_PROPOSALS
    else:
        gts = {scene: valid_boxes(
            np.load(GT_ROOT / scene / "after_filter_boxes.npy", allow_pickle=False), scene)
            for scene in scenes}
        aligns = {scene: None for scene in scenes}
        transform = lambda boxes, _: boxes
        native_root, proposal_root = CA_NATIVE, CA_PROPOSALS
    return scenes, gts, aligns, transform, native_root, proposal_root


def reliability_rows(dataset: str, scene_data: dict, labels: dict,
                     threshold: float) -> list[dict]:
    score_edges = (0.0, 0.25, 0.50, 0.75, 1.000001)
    support_edges = (0.0, 0.25, 0.50, 0.75, 1.000001)
    rows = []
    for si in range(4):
        for ui in range(4):
            values_iou, values_tp = [], []
            for scene, data in scene_data.items():
                score, support = data["scores"], data["support"]["max"]
                mask = ((score >= score_edges[si]) & (score < score_edges[si + 1])
                        & (support >= support_edges[ui])
                        & (support < support_edges[ui + 1]))
                values_iou.extend(data["best_gt_iou"][mask].tolist())
                values_tp.extend(labels[scene][mask].tolist())
            if values_iou:
                rows.append({
                    "native_score_bin": [score_edges[si], min(1.0, score_edges[si + 1])],
                    "support_bin": [support_edges[ui], min(1.0, support_edges[ui + 1])],
                    "count": len(values_iou),
                    "original_ranking_tp_rate": float(np.mean(values_tp)),
                    "gt_overlap_rate": float(np.mean(np.asarray(values_iou) > threshold)),
                    "mean_best_gt_iou": float(np.mean(values_iou)),
                })
    return rows


def run(dataset: str) -> dict:
    scenes, gts, aligns, transform, native_root, proposal_root = load_dataset(dataset)
    scene_data, original_predictions, inputs = {}, {}, {}
    max_sidecar_error = 0.0
    for ordinal, scene in enumerate(scenes, 1):
        boxes, scores = read_prediction(native_root / f"{scene}_boxes.pkl")
        proposal_path = proposal_root / f"{scene}.npz"
        with np.load(proposal_path, allow_pickle=False) as z:
            key = "boxes2d" if dataset == "scannet" else "boxes"
            proposal_boxes = np.asarray(z[key], dtype=np.float64).reshape(-1, 4)
            frame_ids = np.asarray(z["frame_ids"], dtype=np.int64)
        by_frame = {int(frame): proposal_boxes[frame_ids == frame]
                    for frame in np.unique(frame_ids)}
        if dataset == "scannet":
            intrinsic = np.loadtxt(SCANS / scene / "intrinsic/intrinsic_color.txt")[:3, :3]
            pose_cache = {}
            def pose_for(frame):
                if frame not in pose_cache:
                    value = np.loadtxt(SCANS / scene / f"pose/{frame}.txt").reshape(4, 4)
                    pose_cache[frame] = value if np.isfinite(value).all() else None
                return pose_cache[frame]
            width, height = 1296, 968
        else:
            intrinsic = np.loadtxt(CA_DATA / scene / "K_rgb.txt").reshape(3, 3)
            poses = np.load(CA_DATA / scene / "all_poses.npy", allow_pickle=False)
            pose_for = lambda frame, values=poses: values[frame]
            width, height = 1024, 768
        exclusive = dataset == "scannet"
        support_boxes = boxes
        if dataset == "scannet":
            # The frozen final implementation performs exclusive per-view
            # proposal assignment over native rows plus M1-P births, then
            # updates native scores only.  Preserve that exact competition.
            support_boxes, _ = read_prediction(SCANNET_M1P / f"{scene}_boxes.pkl")
            if (len(support_boxes) < len(boxes)
                    or not np.array_equal(support_boxes[:len(boxes)], boxes)):
                raise RuntimeError(f"M1-P context prefix mismatch: {scene}")
        all_support = support_statistics(
            support_boxes, by_frame, pose_for, intrinsic, width, height, exclusive)
        support = {name: values[:len(boxes)] for name, values in all_support.items()}
        if dataset == "scannet":
            side_path = SCANNET_SIDECAR / f"{scene}_boxes.pkl.dual_state.json"
            side = json.loads(side_path.read_text())
            archived = np.asarray([
                row["support"] for row in side["rows"][:len(boxes)]], dtype=np.float64)
            max_sidecar_error = max(
                max_sidecar_error,
                float(np.max(np.abs(archived - support["max"]))) if len(boxes) else 0.0)
            inputs[str(side_path.resolve())] = sha256(side_path)
            inputs[str((SCANNET_M1P / f"{scene}_boxes.pkl").resolve())] = sha256(
                SCANNET_M1P / f"{scene}_boxes.pkl")
        eval_boxes = transform(boxes, aligns[scene])
        overlap = aabb_iou(eval_boxes, gts[scene])
        best_iou = overlap.max(1) if overlap.shape[1] else np.zeros(len(boxes))
        scene_data[scene] = {
            "raw_boxes": boxes, "eval_boxes": eval_boxes, "scores": scores,
            "support": support, "best_gt_iou": best_iou,
        }
        original_predictions[scene] = (eval_boxes, scores)
        for path in (native_root / f"{scene}_boxes.pkl", proposal_path):
            inputs[str(path.resolve())] = sha256(path)
        print(f"[{dataset} {ordinal:03d}/{len(scenes)}] {scene}", flush=True)

    score_arms = {
        "original": lambda d: d["scores"],
        "support_only": lambda d: d["support"]["max"],
        "additive_max": lambda d: additive(d["scores"], d["support"]["max"]),
        "native_logit_first": lambda d: logit_update(d["scores"], d["support"]["first"]),
        "native_logit_mean": lambda d: logit_update(d["scores"], d["support"]["mean"]),
        "native_logit_max": lambda d: logit_update(d["scores"], d["support"]["max"]),
    }
    metrics = {}
    for name, scorer in score_arms.items():
        predictions = {scene: (data["eval_boxes"], scorer(data))
                       for scene, data in scene_data.items()}
        metrics[name] = {str(t): class_agnostic_ap(predictions, gts, t)
                         for t in THRESHOLDS}
    labels = {str(t): original_tp_labels(original_predictions, gts, t)
              for t in THRESHOLDS}
    reliability = {str(t): reliability_rows(dataset, scene_data, labels[str(t)], t)
                   for t in THRESHOLDS}
    support_summary = {}
    for name in ("max", "mean", "first", "visible_views"):
        values = np.concatenate([data["support"][name] for data in scene_data.values()])
        support_summary[name] = {
            "mean": float(values.mean()), "median": float(np.median(values)),
            "p90": float(np.quantile(values, .9)), "max": float(values.max()),
        }
    return {
        "dataset": dataset, "scene_count": len(scenes),
        "native_rows": int(sum(len(d["scores"]) for d in scene_data.values())),
        "parameters": {"tau": TAU, "logit_strength": LOGIT_STRENGTH,
                       "additive_alpha": ADDITIVE_ALPHA,
                       "exclusive_per_view_matching": dataset == "scannet",
                       "mean_denominator": "frames where box projects in-view and proposal cache has a frame entry",
                       "single_view": "earliest in-view cached frame"},
        "metrics": metrics, "support_summary": support_summary,
        "reliability": reliability,
        "max_abs_error_vs_archived_support": max_sidecar_error if dataset == "scannet" else None,
        "invariants": {"same_geometry": True, "same_row_count": True,
                       "same_labels": True, "only_scores_change": True},
        "input_sha256": inputs, "gt_use": "evaluation and retrospective reliability analysis only",
    }


def markdown(result: dict) -> str:
    lines = [
        "# M2 Support and Reranking Audit",
        "",
    ]
    for dataset in ("scannet", "ca1m"):
        block = result[dataset]
        lines += [
            f"## {dataset}", "",
            f"Scenes: {block['scene_count']}; fixed native rows: {block['native_rows']}.",
            "",
            "| Score rule | AP15 | AP25 | AP50 |",
            "|---|---:|---:|---:|",
        ]
        for name, metrics in block["metrics"].items():
            lines.append(
                f"| {name} | {metrics['0.15']['ap']:.4f} | "
                f"{metrics['0.25']['ap']:.4f} | {metrics['0.5']['ap']:.4f} |")
        lines += [
            "",
            "All rows use identical native geometry, labels and cardinality. "
            "Detailed score/support-bin reliability tables are stored in `results.json`.",
            "",
        ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "reports/m2_support_reranking_20260921")
    parser.add_argument("--dataset", choices=("scannet", "ca1m", "both"),
                        default="both")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    selected = ("scannet", "ca1m") if args.dataset == "both" else (args.dataset,)
    result = {dataset: run(dataset) for dataset in selected}
    result["schema"] = "boxfusion.m2.support_reranking.v1"
    (args.output / "results.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    if args.dataset == "both":
        (args.output / "REPORT.md").write_text(markdown(result) + "\n")
    print(args.output / "results.json")


if __name__ == "__main__":
    main()
