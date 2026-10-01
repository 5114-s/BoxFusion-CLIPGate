#!/usr/bin/env python3
"""CPU-only factor counterfactuals on an existing, paired lifting capture.

This is not AP, a model rerun, or a reproduction of the unlocated B5=91 table.
GT projected-box/depth-volume support are identity proxies, not 2D instance GT.
"""
from __future__ import annotations

import argparse
import itertools
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

from eval_ca1m_prenms_query import (
    boxes_array, box2d_iou, digest, load_baseline, pairwise_iou,
    project_gt_bbox, verify_anchor, write_new_json,
)

SIGNS = np.array(list(itertools.product((-1., 1.), repeat=3)))
SCENES = ("42446540", "42897501", "42897521")
ARMS = ("original", "center_only", "ray_depth_only", "lateral_only",
        "size_only", "orientation_only", "center_size", "all_gt")


def corners(center, size, rotation):
    return np.asarray(center)[None] + (SIGNS * np.asarray(size) / 2) @ rotation.T


def decompose(box, source):
    """Use verified capture/GT corner orders, never PCA on near-cubic boxes."""
    box = boxes_array([box], source)[0]
    indices = (4, 2, 1) if source == "prediction" else (1, 3, 4)
    edges = np.stack([box[i] - box[0] for i in indices], axis=1)
    size = np.linalg.norm(edges, axis=0)
    rotation = edges / size
    if np.linalg.det(rotation) < 0:
        rotation[:, 0] *= -1
    if np.max(np.abs(rotation.T @ rotation - np.eye(3))) > 2e-3:
        raise ValueError(f"Nonorthogonal {source} corners")
    center = box.mean(0)
    rebuilt = corners(center, size, rotation)
    error = np.linalg.norm(rebuilt[:, None] - box[None], axis=2).min(1).max()
    if error > 1e-4:
        raise ValueError(f"Unsupported {source} corner order: {error}")
    return center, size, rotation


def aligned_gt(pred_rotation, gt_size, gt_rotation):
    # Resolve size/axis exchange equivalence before individual factor replacement.
    perm = max(itertools.permutations(range(3)), key=lambda p:
               sum(abs(np.dot(pred_rotation[:, i], gt_rotation[:, p[i]]))
                   for i in range(3)))
    r = gt_rotation[:, perm].copy()
    for i in range(3):
        if np.dot(pred_rotation[:, i], r[:, i]) < 0:
            r[:, i] *= -1
    if np.linalg.det(r) < 0:
        r[:, 2] *= -1
    return gt_size[list(perm)], r


def factor_boxes(pred, gt, camera_center):
    pc, ps, pr = decompose(pred, "prediction")
    gc, gs, gr = decompose(gt, "GT")
    gs, gr = aligned_gt(pr, gs, gr)
    ray = pc - camera_center
    ray /= max(np.linalg.norm(ray), 1e-12)
    delta = gc - pc
    parallel = np.dot(delta, ray) * ray
    output = {
        "original": np.asarray(pred),
        "center_only": corners(gc, ps, pr),
        "ray_depth_only": corners(pc + parallel, ps, pr),
        "lateral_only": corners(pc + delta - parallel, ps, pr),
        "size_only": corners(pc, gs, pr),
        "orientation_only": corners(pc, ps, gr),
        "center_size": corners(gc, gs, pr),
        "all_gt": corners(gc, gs, gr),
    }
    errors = {
        "center_error_m": float(np.linalg.norm(delta)),
        "ray_depth_error_m": float(abs(np.dot(delta, ray))),
        "lateral_error_m": float(np.linalg.norm(delta - parallel)),
        "pred_size_m": ps.tolist(), "aligned_gt_size_m": gs.tolist(),
        "size_ratios": (ps / gs).tolist(),
        "volume_ratio": float(np.prod(ps) / np.prod(gs)),
        "pred_at_5cm_floor": bool(np.any(ps <= 0.05001)),
        "pred_near_4m_ceiling": bool(np.any(ps >= 3.99)),
        "gt_has_dimension_below_5cm": bool(np.any(gs < .05)),
    }
    return output, errors


def depth_support(gt_parts, projected_box, depth, kd, pose, width, height):
    """At most 16x16 depth pixels; volume membership is only a proxy."""
    hd, wd = depth.shape
    b = np.asarray(projected_box) * np.array([wd/width, hd/height] * 2)
    xs = np.unique(np.rint(np.linspace(b[0], b[2], 16)).astype(int))
    ys = np.unique(np.rint(np.linspace(b[1], b[3], 16)).astype(int))
    xs, ys = xs[(xs >= 0) & (xs < wd)], ys[(ys >= 0) & (ys < hd)]
    xx, yy = np.meshgrid(xs, ys)
    z = depth[yy, xx].reshape(-1)
    ok = np.isfinite(z) & (z > 0)
    if not ok.any():
        return {"valid_samples": 0, "inside_gt_volume": 0, "fraction": 0.0,
                "passes": False}
    pixels = np.stack([xx.reshape(-1), yy.reshape(-1), np.ones(z.size)], axis=1)
    pts = (pixels[ok] @ np.linalg.inv(kd).T) * z[ok, None]
    pts = pts @ pose[:3, :3].T + pose[:3, 3]
    gc, gs, gr = gt_parts
    local = (pts - gc) @ gr
    inside = (np.abs(local) <= gs / 2 + .01).all(1)
    count = int(inside.sum())
    fraction = float(inside.mean())
    return {"valid_samples": int(ok.sum()), "inside_gt_volume": count,
            "fraction": fraction, "passes": count >= 5 and fraction >= .1}


def self_check():
    angle = .43
    r = np.array([[np.cos(angle), -np.sin(angle), 0],
                  [np.sin(angle), np.cos(angle), 0], [0, 0, 1.]])
    center, size = np.array([.4, -.2, 2.]), np.array([.2, .5, .8])
    p = corners(center, size, r)
    # Dataset GT order: bottom 4 in cycle, then top 4.
    order = [0, 2, 6, 4, 1, 3, 7, 5]
    g = p[order]
    for source, box in (("prediction", p), ("GT", g)):
        c, s, rr = decompose(box, source)
        assert pairwise_iou([corners(c, s, rr)], [g])[0, 0] > 1 - 1e-10
    identical, _ = factor_boxes(p, g, np.zeros(3))
    assert all(pairwise_iou([b], [g])[0, 0] > 1 - 1e-10 for b in identical.values())
    shifted = p + np.array([.6, -.1, .3])
    variants, _ = factor_boxes(shifted, g, np.zeros(3))
    assert pairwise_iou([variants["center_only"]], [g])[0, 0] > 1 - 1e-10
    scaled = corners(center, size * 2, r)
    variants, _ = factor_boxes(scaled, g, np.zeros(3))
    assert pairwise_iou([variants["size_only"]], [g])[0, 0] > 1 - 1e-10
    assert np.allclose(variants["center_only"], scaled)
    return {"identity_axis_equivalence_translation_size": "passed"}


def summarize(records, targets):
    chosen = []
    for target in targets:
        rows = [r for r in records if (r["scene"], r["gt_id"]) == target]
        if rows:
            # Select one representative using 2D quality only, never output 3D IoU.
            chosen.append(min(rows, key=lambda r: (-r["iou2d"], -r["score"],
                                                   r["frame_id"], r["anchor_id"])))
    answer = {"targets": len(targets), "paired_targets": len(chosen),
              "paired_observations": len(records), "arms": {}}
    for name in ARMS:
        frame_support = defaultdict(set)
        for r in records:
            for threshold in (.15, .25, .5):
                if r["iou"][name] > threshold:
                    frame_support[(r["scene"], r["gt_id"], threshold)].add(r["frame_id"])
        answer["arms"][name] = {
            str(t): {
                "fixed_2d_selected_gt_hits": sum(r["iou"][name] > t for r in chosen),
                "any_paired_observation_gt_hits": sum(bool(frame_support[(*g, t)]) for g in targets),
                "at_least_3_distinct_frames": sum(len(frame_support[(*g, t)]) >= 3 for g in targets),
            } for t in (.15, .25, .5)}
    scalar_keys = ("center_error_m", "ray_depth_error_m", "lateral_error_m", "volume_ratio")
    answer["fixed_2d_selected_error_medians"] = {
        k: float(np.median([r["errors"][k] for r in chosen])) if chosen else None
        for k in scalar_keys}
    answer["fixed_2d_selected_flags"] = {
        k: sum(r["errors"][k] for r in chosen) for k in
        ("pred_at_5cm_floor", "pred_near_4m_ceiling", "gt_has_dimension_below_5cm")}
    answer["fixed_2d_selected_sources"] = [
        {k: r[k] for k in ("scene", "gt_id", "frame_id", "anchor_id")}
        for r in chosen]
    return answer


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--capture", type=Path, default=Path("reports/ca1m_prenms_query_pilot_20260908"))
    ap.add_argument("--baseline-root", type=Path, default=Path("results/ca1m_dual"))
    ap.add_argument("--data-root", type=Path, default=Path("/tmp/ca1m_clean_root"))
    ap.add_argument("--rgbd-root", type=Path, default=Path("/extra/ZhaoX/boxfusion_ca1m"))
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    hashes = {}

    def remember(path):
        path = Path(path).resolve()
        hashes[str(path)] = digest(path)
        return path

    checks = self_check()
    anchor = verify_anchor("/data/ZhaoX/OVM3D-Dett/boxfusion_boxer_dev/evaluation")
    for p in (Path(__file__), Path(__file__).with_name("eval_ca1m_prenms_query.py"),
              Path(__file__).with_name("audit_ca1m_nms_child_headroom.py")):
        remember(p)
    historical = json.loads(remember(args.capture / "diagnostic.json").read_text())
    # This archive binds each paired source file, baseline and GT to its prior audit.
    historical_hashes = historical.get("input_sha256", historical.get("sha256", {}))
    protocol = json.loads(remember(args.capture / "protocol.json").read_text())
    if tuple(protocol["scenes"]) != SCENES:
        raise ValueError("Fixed three-scene capture changed")
    records, all_targets, scene_results = [], [], []
    for scene in SCENES:
        sp = remember(args.capture / scene / "scene.json")
        payload = json.loads(sp.read_text())
        if payload.get("completed") is not True or payload["scene_id"] != scene:
            raise ValueError("Incomplete or mismatched capture")
        gt = boxes_array(np.load(remember(args.data_root / scene / "after_filter_boxes.npy")), "GT")
        base = load_baseline(remember(args.baseline_root / f"{scene}_boxes.pkl"))
        root = args.rgbd_root / scene
        kd = np.loadtxt(remember(root / "K_depth.txt"))
        kc = np.loadtxt(remember(root / "K_rgb.txt"))
        poses = np.load(remember(root / "all_poses.npy"))
        gt_parts = [decompose(g, "GT") for g in gt]
        base_best = pairwise_iou(base, gt).max(0) if len(base) else np.zeros(len(gt))
        natural_best = np.zeros(len(gt))
        frames = payload["frames"]
        normal_count = 0
        for f in frames:
            c = boxes_array(f["normal"]["corners"], "normal")
            normal_count += len(c)
            if len(c):
                natural_best = np.maximum(natural_best, pairwise_iou(c, gt).max(0))
        missing = np.flatnonzero(base_best <= .15)
        targets = set(int(i) for i in missing if natural_best[i] <= .15)
        all_targets.extend((scene, i) for i in sorted(targets))
        counts = Counter()
        for f in frames:
            fid, pose = int(f["frame_id"]), np.asarray(f["pose"])
            if not np.array_equal(pose, poses[fid]) or not np.array_equal(np.asarray(f["K"]), kc):
                raise ValueError("Pose/intrinsics changed since capture")
            normal = f["normal"]
            ids = np.asarray(normal["anchor_ids"], dtype=int)
            pred = boxes_array(normal["corners"], "prediction")
            scores = np.asarray(normal["scores"])
            if len(set(ids)) != len(ids) or len(pred) != len(ids) or len(scores) != len(ids):
                raise ValueError("Anchor/3D output cardinality mismatch")
            with np.load(remember(sp.parent / f["raw_file"])) as raw:
                pb = raw["boxes"][ids].astype(float)
                if not np.allclose(scores, raw["scores"][ids], atol=1e-6, rtol=0):
                    raise ValueError("Paired anchor score mismatch")
            projections = [project_gt_bbox(g, pose, kc, f["width"], f["height"]) for g in gt]
            overlap = np.zeros((len(pb), len(gt)))
            for gi, b in enumerate(projections):
                if b is not None:
                    overlap[:, gi] = box2d_iou(pb, b)
            candidates = [(pi, gi) for gi in sorted(targets)
                          for pi in np.flatnonzero(overlap[:, gi] >= .5)]
            if not candidates:
                continue
            dp = remember(root / "depth" / f"{fid}.png")
            depth_raw = np.asarray(Image.open(dp))
            if depth_raw.dtype != np.uint16:
                raise ValueError("Expected archived millimeter uint16 depth")
            depth = depth_raw.astype(float) / 1000
            # This fast support proxy assumes the archive's aligned optical frames.
            scale = np.diag([depth.shape[1]/f["width"], depth.shape[0]/f["height"], 1.])
            if not np.allclose(scale @ kc, kd, atol=1e-5, rtol=0):
                raise ValueError("Depth/RGB grids require an explicit registration transform")
            support_cache = {}
            for pi, gi in candidates:
                counts["projected_overlap_pairs"] += 1
                if gi not in support_cache:
                    support_cache[gi] = depth_support(gt_parts[gi], projections[gi], depth, kd,
                                                      pose, f["width"], f["height"])
                support = support_cache[gi]
                unique = int((overlap[pi] >= .5).sum()) == 1
                counts["unique_projection_pairs"] += int(unique)
                counts["depth_supported_pairs"] += int(support["passes"])
                variants, errors = factor_boxes(pred[pi], gt[gi], pose[:3, 3])
                iou = {k: float(pairwise_iou([b], [gt[gi]])[0, 0]) for k, b in variants.items()}
                if iou["all_gt"] < .999:
                    raise ValueError("GT reconstruction counterfactual did not recover target")
                records.append({"scene": scene, "gt_id": gi, "frame_id": fid,
                                "anchor_id": int(ids[pi]), "score": float(scores[pi]),
                                "iou2d": float(overlap[pi, gi]), "unique_projection": unique,
                                "depth_support": support, "iou": iou, "errors": errors})
        scene_results.append({"scene": scene, "gt": len(gt), "baseline_missing_gt": len(missing),
                              "no_normal_3d_support_gt": len(targets), "frames": len(frames),
                              "normal_observations": normal_count, "pair_counts": dict(counts)})
        print(json.dumps(scene_results[-1]), flush=True)
    subsets = {
        "projection_overlap_only": records,
        "projection_and_depth_support": [r for r in records if r["depth_support"]["passes"]],
        "unique_projection_and_depth_support": [r for r in records if r["unique_projection"]
                                                 and r["depth_support"]["passes"]],
    }
    # Inspect all historical fingerprint fields without inventing a missing key.
    historical_verified = 0
    if not historical_hashes:
        for key, value in historical.items():
            if isinstance(value, dict) and value and all(isinstance(k, str) and k.startswith("/")
                                                       and isinstance(v, str) and len(v) == 64
                                                       for k, v in value.items()):
                historical_hashes = value
                break
    if not historical_hashes:
        raise ValueError("No historical fingerprint ledger found")
    for path, expected in historical_hashes.items():
        if path in hashes:
            if hashes[path] != expected:
                raise ValueError(f"Historical input changed: {path}")
            historical_verified += 1
    if historical_verified < 6:
        raise ValueError("Insufficient historical source/GT/baseline hash overlap")
    for path, expected in hashes.items():
        if digest(path) != expected:
            raise ValueError(f"Input changed during diagnostic: {path}")
    result = {
        "schema": "boxfusion.lifting_factor_diagnostic.v1", "completed": True,
        "original_B5_91_table_reproduced": False,
        "scope": "fixed existing 3-scene capture; ordinary WeDetect-to-Boxer observations only",
        "rules": {"baseline_missing_iou": .15, "normal_3d_no_support_iou": .15,
                  "projected_2d_iou_min": .5, "depth_gt_volume_tolerance_m": .01,
                  "depth_min_samples_inside": 5, "depth_min_fraction_inside": .1,
                  "GT_for_offline_diagnosis_only": True, "model_forward_calls": 0,
                  "production_changes": False, "AP_or_FPS_evaluation": False},
        "limitations": [
            "Projected GT OBB rectangle is not visible-instance 2D ground truth.",
            "Depth volume support reduces occlusion confounding but does not prove proposal identity.",
            "Counterfactuals use GT and are not implementable gains; factor counts are nonadditive.",
            "Only supplied normal anchors have lifting records; B6/unsubmitted anchor outcomes are unknown.",
            "GT2D prompt relifting and depth-patch intervention are not run.",
            "These are observation-set GT coverage counts, not one-to-one recall, births or AP.",
            "Extents and orientation are decomposed with box-axis permutation equivalence.",
        ],
        "self_checks": checks, "anchor_metric": anchor, "scenes": scene_results,
        "summaries": {k: summarize(v, all_targets) for k, v in subsets.items()},
        "records": records, "historical_matching_hashes": historical_verified,
        "input_sha256": hashes,
    }
    write_new_json(args.output, result)
    print(json.dumps({"output": str(args.output), "summaries": {
        k: {kk: vv for kk, vv in v.items() if kk != "fixed_2d_selected_sources"}
        for k, v in result["summaries"].items()}}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
