#!/usr/bin/env python3
"""GT-assisted full107 lifting-factor counterfactual on a paired capture."""
from __future__ import annotations

import argparse
import json
import pickle
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

import integrated_online as online
from audit_ca1m_lifting_factors import (
    boxes_array, box2d_iou, decompose, depth_support, factor_boxes,
    project_gt_bbox, self_check, verify_anchor, write_new_json,
)
from true_fusion_audit_core import aabb_iou, class_agnostic_ap

ARMS = ("original", "center_only", "ray_depth_only", "lateral_only",
        "size_only", "orientation_only", "center_size")
POLICIES = ("projection_overlap", "projection_depth", "unique_projection_depth")


def read_predictions(path):
    with Path(path).open("rb") as f:
        payload = pickle.load(f)
    rows = payload[0]
    corners = boxes_array([r[1] for r in rows], str(path))
    scores = np.asarray([float(r[2]) for r in rows], dtype=np.float64)
    if not np.isfinite(scores).all():
        raise ValueError(f"{path}: invalid scores")
    return corners, scores


def funnel(corners, scores, frames, native):
    receipts = []
    unique_frames = np.unique(frames)
    ordinal = {int(frame): index for index, frame in enumerate(unique_frames)}
    order = np.argsort(frames, kind="stable")
    for index in order:
        box, score, frame = corners[index], float(scores[index]), ordinal[int(frames[index])]
        if len(native) and np.any(aabb_iou([box], native)[0] >= online.DEDUP):
            continue
        best, best_iou = None, 0.0
        for receipt in receipts:
            if frame - receipt["last"] > online.TTL:
                continue
            previous = receipt["obs"][-1]
            if np.linalg.norm(previous.mean(0) - box.mean(0)) > .50:
                continue
            value = float(aabb_iou([box], [previous])[0, 0])
            if value >= .10 and value > best_iou:
                best, best_iou = receipt, value
        if best is None:
            receipts.append({"obs": [box], "frames": {frame}, "last": frame,
                             "scores": [score]})
        else:
            best["obs"].append(box); best["frames"].add(frame)
            best["last"] = frame; best["scores"].append(score)
    candidates = []
    for receipt in receipts:
        if len(receipt["frames"]) < 3:
            continue
        obs = receipt["obs"]
        medoid = max(range(len(obs)), key=lambda i:
                     sum(float(aabb_iou([obs[i]], [obs[j]])[0, 0])
                         for j in range(len(obs)) if j != i))
        candidates.append((float(np.mean(receipt["scores"])), obs[medoid]))
    candidates.sort(key=lambda x: -x[0])
    kept = []
    for strength, box in candidates:
        if kept and np.any(aabb_iou([box], [b for _, b in kept])[0] >= online.SELF_NMS):
            continue
        kept.append((strength, box))
        if len(kept) >= online.CAP:
            break
    return kept


def nativelogit(native_rows, births, boxes2d, frame_ids, poses, K, width, height):
    rows = [(box, float(score), True) for box, score in zip(*native_rows)]
    rows += [(box, float(online.price(box)), False) for _, box in births]
    series = [[] for _ in rows]
    for frame in sorted(set(frame_ids.tolist())):
        proposals = boxes2d[frame_ids == frame]
        pairs, valid = [], []
        for i, (box, _, _) in enumerate(rows):
            projected = online.project_xyxy(box, poses[frame], K, W=width, H=height)
            if projected is None:
                continue
            valid.append(i)
            x1 = np.maximum(projected[0], proposals[:, 0]); y1 = np.maximum(projected[1], proposals[:, 1])
            x2 = np.minimum(projected[2], proposals[:, 2]); y2 = np.minimum(projected[3], proposals[:, 3])
            inter = np.maximum(0, x2-x1) * np.maximum(0, y2-y1)
            union = ((projected[2]-projected[0])*(projected[3]-projected[1]) +
                     (proposals[:, 2]-proposals[:, 0])*(proposals[:, 3]-proposals[:, 1]) - inter)
            for j in np.flatnonzero(inter / np.maximum(union, 1e-9) >= .10):
                pairs.append((float(inter[j]/max(union[j], 1e-9)), i, int(j)))
        used_rows, used_props, assigned = set(), set(), {}
        for value, i, j in sorted(pairs, reverse=True):
            if i in used_rows or j in used_props:
                continue
            used_rows.add(i); used_props.add(j); assigned[i] = value
        for i in valid:
            series[i].append(assigned.get(i, 0.0))
    output_boxes, output_scores = [], []
    for i, (box, score, native) in enumerate(rows):
        if native:
            clipped = min(max(score, 1e-4), 1-1e-4)
            support = max(series[i], default=0.0)
            score = 1 / (1 + np.exp(-(np.log(clipped/(1-clipped)) +
                                      2 * max(0., support-online.TAU))))
        output_boxes.append(box); output_scores.append(float(score))
    return np.asarray(output_boxes), np.asarray(output_scores)


def assign_targets(boxes2d, projected, supported, missing, policy):
    assignment = np.full(len(boxes2d), -1, dtype=np.int64)
    quality = np.zeros(len(boxes2d), dtype=np.float64)
    for gt_id in missing:
        box = projected[gt_id]
        if box is None:
            continue
        values = box2d_iou(boxes2d, box)
        if policy != "projection_overlap" and not supported[gt_id]:
            continue
        better = (values >= .5) & (values > quality)
        assignment[better] = gt_id; quality[better] = values[better]
    if policy == "unique_projection_depth":
        counts = np.zeros(len(boxes2d), dtype=np.int64)
        for gt_id in missing:
            if projected[gt_id] is not None:
                counts += box2d_iou(boxes2d, projected[gt_id]) >= .5
        assignment[counts != 1] = -1
    return assignment, quality


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--capture", type=Path, required=True)
    ap.add_argument("--scenes", type=Path, required=True)
    ap.add_argument("--native-root", type=Path, default=Path("results/ca1m_thr15"))
    ap.add_argument("--baseline-root", type=Path, default=Path("results/ca1m_thr15_m1_m2nl_full107"))
    ap.add_argument("--gt-root", type=Path, default=Path("/tmp/ca1m_clean_root"))
    ap.add_argument("--rgbd-root", type=Path, default=Path("/extra/ZhaoX/boxfusion_ca1m"))
    ap.add_argument("--only-scene", help=argparse.SUPPRESS)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    scenes = [x.strip() for x in args.scenes.read_text().splitlines()
              if x.strip() and not x.lstrip().startswith("#")]
    complete = json.loads((args.capture / "complete.json").read_text())
    if not complete.get("completed") or complete.get("scenes") != scenes or len(scenes) != 107:
        raise ValueError("Capture is not a complete ordered full107 run")
    if args.only_scene:
        if args.only_scene not in scenes:
            raise ValueError(f"Unknown scene: {args.only_scene}")
        scenes = [args.only_scene]
    self_check(); anchor = verify_anchor("/data/ZhaoX/OVM3D-Dett/boxfusion_boxer_dev/evaluation")
    gt_all, stored, replay = {}, {}, {p: {a: {} for a in ARMS} for p in POLICIES}
    totals = {p: {a: Counter() for a in ARMS} for p in POLICIES}
    scene_rows = []
    for number, scene in enumerate(scenes, 1):
        with np.load(args.capture / f"{scene}.npz", allow_pickle=False) as data:
            boxes2d, scores = data["boxes2d"].astype(float), data["scores"].astype(float)
            frames, original = data["frame_ids"].astype(int), boxes_array(data["corners"], "capture")
            K, Kd = data["K_rgb"].astype(float), data["K_depth"].astype(float)
            width, height = data["width_height"].astype(int)
        if not (len(boxes2d) == len(scores) == len(frames) == len(original)):
            raise ValueError(f"{scene}: capture cardinality mismatch")
        gt = boxes_array(np.load(args.gt_root / scene / "after_filter_boxes.npy"), "GT")
        gt_all[scene] = gt
        baseline = read_predictions(args.baseline_root / f"{scene}_boxes.pkl")
        stored[scene] = baseline
        native = read_predictions(args.native_root / f"{scene}_boxes.pkl")
        base_iou = aabb_iou(baseline[0], gt)
        missing = np.flatnonzero(base_iou.max(0) <= .15) if len(baseline[0]) else np.arange(len(gt))
        root = args.rgbd_root / scene
        poses = np.load(root / "all_poses.npy")
        gt_parts = [decompose(box, "GT") for box in gt]
        grouped = {int(frame): np.flatnonzero(frames == frame) for frame in np.unique(frames)}
        assigned_by_policy = {p: np.full(len(original), -1, np.int64) for p in POLICIES}
        q_by_policy = {p: np.zeros(len(original)) for p in POLICIES}
        for frame, indices in grouped.items():
            pose = poses[frame]
            projected = [project_gt_bbox(box, pose, K, width, height) for box in gt]
            depth = np.asarray(Image.open(root / "depth" / f"{frame}.png")).astype(float) / 1000
            scale = np.diag([depth.shape[1]/width, depth.shape[0]/height, 1.])
            if not np.allclose(scale @ K, Kd, atol=1e-4, rtol=0):
                raise ValueError(f"{scene}/{frame}: RGB/depth grids are not aligned")
            supported = np.zeros(len(gt), dtype=bool)
            for gt_id in missing:
                if projected[gt_id] is not None:
                    supported[gt_id] = depth_support(
                        gt_parts[gt_id], projected[gt_id], depth, Kd, pose, width, height)["passes"]
            for policy in POLICIES:
                assigned, quality = assign_targets(boxes2d[indices], projected, supported, missing, policy)
                assigned_by_policy[policy][indices] = assigned
                q_by_policy[policy][indices] = quality
        per_scene = {"scene": scene, "GT": len(gt), "baseline_missing_gt15": len(missing),
                     "proposals": len(original), "policies": {}}
        for policy in POLICIES:
            assigned = assigned_by_policy[policy]
            valid = assigned >= 0
            original_target_iou = np.zeros(len(original))
            for i in np.flatnonzero(valid):
                original_target_iou[i] = aabb_iou([original[i]], [gt[assigned[i]]])[0, 0]
            b5 = valid & (original_target_iou <= .15)
            targets = set(assigned[b5].tolist())
            per_scene["policies"][policy] = {"B5_targets": len(targets),
                                              "B5_observations": int(b5.sum())}
            for arm in ARMS:
                modified = original.copy()
                if arm != "original":
                    for i in np.flatnonzero(b5):
                        modified[i] = factor_boxes(original[i], gt[assigned[i]],
                                                   poses[frames[i]][:3, 3])[0][arm]
                births = funnel(modified, scores, frames, native[0])
                prediction = nativelogit(native, births, boxes2d, frames, poses, K, width, height)
                replay[policy][arm][scene] = prediction
                counter = totals[policy][arm]
                counter["modified_observations"] += int(b5.sum()) if arm != "original" else 0
                counter["births"] += len(births)
                counter["scenes"] += 1
        scene_rows.append(per_scene)
        print(f"[{number}/{len(scenes)}] {scene}: missing={len(missing)} proposals={len(original)}", flush=True)
    thresholds = (.15, .25, .50)
    stored_ap = {str(t): class_agnostic_ap(stored, gt_all, t) for t in thresholds}
    result = {"schema": "boxfusion.ca1m_lifting_factor_full107.v1", "completed": True,
              "scope": "GT-assisted counterfactual replay of current M1+M2 candidate generation",
              "stored_baseline_ap": stored_ap, "policies": {}, "scenes": scene_rows,
              "anchor_metric": anchor,
              "limitations": [
                  "Projected GT OBB rectangles are not visible-instance GT2D masks.",
                  "Depth-in-GT-volume is an identity proxy, not proof of proposal identity.",
                  "GT factor replacement is an oracle and not an executable module.",
                  "Fresh frozen-model replay can differ numerically from archived M1+M2 output.",
                  "This tests submitted WeDetect proposals; it does not classify B6 causes."]}
    for policy in POLICIES:
        policy_result = {"arms": {}}
        original_ap = {str(t): class_agnostic_ap(replay[policy]["original"], gt_all, t)
                       for t in thresholds}
        policy_result["replay_original_ap"] = original_ap
        for arm in ARMS:
            metrics = {str(t): class_agnostic_ap(replay[policy][arm], gt_all, t)
                       for t in thresholds}
            policy_result["arms"][arm] = {
                "metrics": metrics,
                "delta_vs_replay_original": {str(t): metrics[str(t)]["ap"]-original_ap[str(t)]["ap"]
                                             for t in thresholds},
                "counts": dict(totals[policy][arm])}
        result["policies"][policy] = policy_result
    write_new_json(args.output, result)
    print(json.dumps({"stored_baseline_ap": stored_ap,
                      "policies": {p: {a: result["policies"][p]["arms"][a]["delta_vs_replay_original"]
                                       for a in ARMS if a != "original"}
                                   for p in POLICIES}}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
