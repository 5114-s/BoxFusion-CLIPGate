#!/usr/bin/env python3
"""CPU-only, GT-assisted diagnostics for the bounded pre-NMS query probe.

This is a candidate-observation diagnostic, not an AP evaluator.  Ground truth
is read only here, never by the GT-free query policy.  The separately emitted
oracle plan must only be consumed by an explicitly oracle-labelled run.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
THRESHOLDS = (0.15, 0.25, 0.50)


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            result.update(block)
    return result.hexdigest()


def boxes_array(value, name):
    result = np.asarray(value, dtype=np.float64)
    if result.size == 0:
        return np.empty((0, 8, 3), dtype=np.float64)
    if result.ndim != 3 or result.shape[1:] != (8, 3):
        raise ValueError(f"{name}: expected [N,8,3], got {result.shape}")
    if not np.isfinite(result).all() or (np.ptp(result, axis=1) <= 0).any():
        raise ValueError(f"{name}: nonfinite or degenerate world boxes")
    return result


def pairwise_iou(boxes, gt):
    """The CA-1M anchor metric: world axis-aligned corner bounds."""
    boxes = boxes_array(boxes, "prediction")
    gt = boxes_array(gt, "GT")
    if not len(boxes) or not len(gt):
        return np.zeros((len(boxes), len(gt)), dtype=np.float64)
    lo, hi = boxes.min(1), boxes.max(1)
    gl, gh = gt.min(1), gt.max(1)
    inter = np.maximum(0, np.minimum(hi[:, None], gh) - np.maximum(lo[:, None], gl)).prod(2)
    union = (hi - lo).prod(1)[:, None] + (gh - gl).prod(1) - inter
    return inter / union


def box2d_iou(boxes, target):
    boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    target = np.asarray(target, dtype=np.float64).reshape(4)
    area = np.maximum(0, boxes[:, 2:] - boxes[:, :2]).prod(1)
    target_area = np.maximum(0, target[2:] - target[:2]).prod()
    inter = np.maximum(0, np.minimum(boxes[:, 2:], target[2:]) - np.maximum(boxes[:, :2], target[:2])).prod(1)
    return inter / np.maximum(area + target_area - inter, 1e-12)


def project_gt_bbox(corners, camera_to_world, K, width, height):
    """Oracle projection, rejecting camera-plane intersections conservatively.

    A projected box is not proof of visibility or instance identity.  Projection
    is used only to choose the explicitly GT-assisted oracle candidates.
    """
    corners = boxes_array([corners], "GT projection")[0]
    pose = np.asarray(camera_to_world, dtype=np.float64)
    intrinsics = np.asarray(K, dtype=np.float64)
    if pose.shape != (4, 4) or intrinsics.shape != (3, 3):
        raise ValueError("Projection requires a 4x4 camera-to-world pose and 3x3 K")
    if not np.isfinite(pose).all() or not np.isfinite(intrinsics).all():
        return None
    inv = np.linalg.inv(pose)
    camera = corners @ inv[:3, :3].T + inv[:3, 3]
    if np.any(camera[:, 2] <= 0.05):
        return None
    pix = camera @ intrinsics.T
    uv = pix[:, :2] / pix[:, 2:3]
    bounds = np.r_[uv.min(0), uv.max(0)]
    bounds[[0, 2]] = np.clip(bounds[[0, 2]], 0, width)
    bounds[[1, 3]] = np.clip(bounds[[1, 3]], 0, height)
    if np.any(bounds[2:] <= bounds[:2]):
        return None
    return bounds


def choose_oracle_anchors(pool, target_bbox, top_k=3, nms_iou=0.70):
    """Choose geometrically distinct boxes by GT projected-box IoU.

    ``pool`` entries have ``anchor_id`` and ``bbox``.  No centre-containment
    shortcut or detector score is treated as proof of correct 3D recovery.
    """
    if not pool:
        return []
    boxes = np.asarray([item["bbox"] for item in pool], dtype=np.float64)
    if boxes.shape != (len(pool), 4) or not np.isfinite(boxes).all():
        raise ValueError("Malformed oracle 2D pool")
    if np.any(boxes[:, 2:] <= boxes[:, :2]):
        raise ValueError("Degenerate oracle 2D pool box")
    quality = box2d_iou(boxes, target_bbox)
    order = sorted(range(len(pool)), key=lambda i: (-float(quality[i]), int(pool[i]["anchor_id"])))
    chosen = []
    for index in order:
        if quality[index] <= 0:
            continue
        if chosen and np.any(box2d_iou(boxes[chosen], boxes[index]) >= nms_iou):
            continue
        chosen.append(index)
        if len(chosen) == top_k:
            break
    return [{"anchor_id": int(pool[index]["anchor_id"]),
             "projected_gt_iou_2d": float(quality[index]),
             "bbox": boxes[index].tolist()} for index in chosen]


def load_baseline(path):
    with Path(path).open("rb") as handle:
        payload = pickle.load(handle)
    if not isinstance(payload, (list, tuple)) or len(payload) != 1:
        raise ValueError(f"{path}: expected one outer scene list")
    rows = payload[0]
    for index, row in enumerate(rows):
        if len(row) < 3 or int(row[0]) != 0 or not np.isfinite(float(row[2])):
            raise ValueError(f"{path}: invalid class-agnostic row {index}")
    return boxes_array([row[1] for row in rows], str(path))


def verify_anchor(anchor_root):
    source = ROOT / "tools/audit_ca1m_nms_child_headroom.py"
    spec = importlib.util.spec_from_file_location("ca1m_query_anchor_audit", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.verify_anchor(Path(anchor_root))


def write_new_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")


def seed_assignment(seed_boxes, gt, threshold):
    """Require every seed's best GT to agree and clear a strict threshold."""
    matrix = pairwise_iou(seed_boxes, gt)
    if not len(matrix) or not len(gt):
        return None
    indices = matrix.argmax(1)
    best = matrix[np.arange(len(matrix)), indices]
    if np.any(best <= threshold) or np.any(indices != indices[0]):
        return None
    return int(indices[0])


def candidate_summary(candidate, gt, target, threshold):
    result = {"present": candidate is not None, "lifted": False,
              "accepted_geometry": False, "any_gt_hit": False,
              "best_gt": None, "best_gt_iou": 0.0,
              "target_iou": None, "correct_seed_instance": False}
    if candidate is None:
        return result
    result["accepted_geometry"] = bool(candidate["accepted_geometry"])
    if candidate.get("corners") is None:
        return result
    result["lifted"] = True
    matrix = pairwise_iou([candidate["corners"]], gt)
    if len(gt):
        index = int(matrix[0].argmax())
        value = float(matrix[0, index])
        result.update(best_gt=index, best_gt_iou=value, any_gt_hit=value > threshold)
        if target is not None:
            result["target_iou"] = float(matrix[0, target])
            # A huge box overlapping the intended GT but best matching another
            # instance is not treated as successful identity-conditioned query.
            result["correct_seed_instance"] = index == target and value > threshold
    return result


def empty_arm_counts():
    return {name: 0 for name in (
        "present", "lifted", "any_gt_hit", "correct_seed_instance",
        "baseline_missing_correct_seed", "baseline_missing_new_frame_support",
        "baseline_missing_third_support_opportunity")}


def audit_scene(path, args, hashes):
    def remember(source):
        source = Path(source).resolve()
        hashes[str(source)] = digest(source)
        return source

    path = remember(path)
    payload = json.loads(path.read_text())
    scene = str(payload["scene_id"])
    if not scene.isdecimal() or path.parent.name != scene or payload.get("completed") is not True:
        raise ValueError(f"{path}: scene ID mismatch or incomplete run")
    gt_path = remember(Path(args.data_root) / scene / "after_filter_boxes.npy")
    base_path = remember(Path(args.baseline_root) / f"{scene}_boxes.pkl")
    gt = boxes_array(np.load(gt_path, allow_pickle=False), str(gt_path))
    baseline = load_baseline(base_path)
    base_iou = pairwise_iou(baseline, gt)
    base_best = base_iou.max(0) if len(baseline) else np.zeros(len(gt))
    frames = payload["frames"]
    if not isinstance(frames, list) or not frames:
        raise ValueError(f"{path}: no frame ledger")
    frame_ids = [int(frame["frame_id"]) for frame in frames]
    ordinals = [int(frame["ordinal"]) for frame in frames]
    if frame_ids != sorted(set(frame_ids)) or ordinals != sorted(set(ordinals)):
        raise ValueError(f"{path}: repeated or noncausal frame order")
    # normal coverage is a set diagnostic; one box may cover multiple GT.
    normal_support = {threshold: defaultdict(set) for threshold in THRESHOLDS}
    normal_matrices = {}
    normal_count = 0
    for frame in frames:
        fid = int(frame["frame_id"])
        normal = frame["normal"]
        normal_boxes = boxes_array(normal["corners"], f"{scene}/{fid}/normal")
        normal_scores = np.asarray(normal["scores"], dtype=np.float64)
        normal_ids = np.asarray(normal["anchor_ids"], dtype=np.int64)
        if normal_scores.shape != (len(normal_boxes),) or normal_ids.shape != (len(normal_boxes),):
            raise ValueError(f"{scene}/{fid}: normal metadata length mismatch")
        if not np.isfinite(normal_scores).all():
            raise ValueError(f"{scene}/{fid}: nonfinite normal score")
        normal_count += len(normal_boxes)
        matrix = pairwise_iou(normal_boxes, gt)
        normal_matrices[fid] = matrix
        for threshold in THRESHOLDS:
            for target in np.flatnonzero((matrix > threshold).any(0)):
                normal_support[threshold][int(target)].add(fid)

    summaries = {}
    per_query = []
    oracle_frames = []
    for threshold in THRESHOLDS:
        key = f"{threshold:.2f}"
        summaries[key] = {
            "gt_count": len(gt), "baseline_box_count": len(baseline),
            "baseline_any_covered_gt": int((base_best > threshold).sum()),
            "baseline_missing_gt": int((base_best <= threshold).sum()),
            "query_events": 0, "seed_valid": 0, "seed_invalid": 0,
            "seed_valid_baseline_missing": 0,
            "seed_valid_natural_same_frame_already_supported": 0,
            "arms": {arm: {mode: empty_arm_counts() for mode in ("all", "accepted")}
                     for arm in ("query", "control")},
        }
    # Best-GT identity conditioned support, by selected candidate source.
    added_support = {threshold: {arm: defaultdict(set) for arm in ("query", "control")}
                     for threshold in THRESHOLDS}
    accepted_support = {threshold: {arm: defaultdict(set) for arm in ("query", "control")}
                        for threshold in THRESHOLDS}
    seen_query_keys = set()
    for frame in frames:
        fid = int(frame["frame_id"])
        raw_path = (path.parent / frame["raw_file"]).resolve()
        if not raw_path.is_relative_to(path.parent):
            raise ValueError(f"{scene}/{fid}: raw file escapes scene directory")
        remember(raw_path)
        with np.load(raw_path, allow_pickle=False) as raw:
            raw_boxes = np.asarray(raw["boxes"], dtype=np.float64)
            raw_scores = np.asarray(raw["scores"], dtype=np.float64)
        if raw_boxes.ndim != 2 or raw_boxes.shape[1] != 4 or raw_scores.shape != (len(raw_boxes),):
            raise ValueError(f"{scene}/{fid}: invalid raw array shapes")
        if not np.isfinite(raw_boxes).all() or not np.isfinite(raw_scores).all():
            raise ValueError(f"{scene}/{fid}: nonfinite raw proposals")
        items = []
        for event in frame["queries"]:
            track = int(event["track_id"])
            event_key = (fid, track)
            if event_key in seen_query_keys:
                raise ValueError(f"{scene}/{fid}/{track}: duplicate query event")
            seen_query_keys.add(event_key)
            seed_frames = [int(value) for value in event["seed_frames"]]
            seeds = boxes_array(event["seed_observations"], f"{scene}/{fid}/{track}/seed")
            if len(seeds) != len(seed_frames) or len(seed_frames) not in (1, 2):
                raise ValueError(f"{scene}/{fid}/{track}: expected one/two seed observations")
            if len(set(seed_frames)) != len(seed_frames) or any(value >= fid for value in seed_frames):
                raise ValueError(f"{scene}/{fid}/{track}: duplicate or future seed frames")
            pool_ids = [int(value) for value in event["pool_ids"]]
            if len(set(pool_ids)) != len(pool_ids) or any(value < 0 or value >= len(raw_boxes) for value in pool_ids):
                raise ValueError(f"{scene}/{fid}/{track}: invalid pool anchor IDs")
            for arm in ("query", "control"):
                selected = event.get(arm)
                if selected is not None and int(selected["anchor_id"]) not in pool_ids:
                    raise ValueError(f"{scene}/{fid}/{track}: {arm} selected outside shared pool")
            row = {"frame_id": fid, "track_id": track, "seed_frames": seed_frames,
                   "pool_size": len(pool_ids), "thresholds": {}}
            for threshold in THRESHOLDS:
                key = f"{threshold:.2f}"
                stats = summaries[key]
                stats["query_events"] += 1
                target = seed_assignment(seeds, gt, threshold)
                missing = target is not None and base_best[target] <= threshold
                natural_same_frame = target is not None and fid in normal_support[threshold][target]
                stats["seed_valid" if target is not None else "seed_invalid"] += 1
                stats["seed_valid_baseline_missing"] += int(missing)
                stats["seed_valid_natural_same_frame_already_supported"] += int(natural_same_frame)
                diagnostic = {"seed_gt": target, "seed_valid": target is not None,
                              "baseline_missing": bool(missing),
                              "natural_same_frame_already_supported": bool(natural_same_frame)}
                for arm in ("query", "control"):
                    result = candidate_summary(event.get(arm), gt, target, threshold)
                    diagnostic[arm] = result
                    correct = result["correct_seed_instance"]
                    for mode in ("all", "accepted"):
                        if mode == "accepted" and not result["accepted_geometry"]:
                            continue
                        counts = stats["arms"][arm][mode]
                        for name in ("present", "lifted", "any_gt_hit", "correct_seed_instance"):
                            counts[name] += int(result[name])
                        counts["baseline_missing_correct_seed"] += int(missing and correct)
                        new_support = missing and correct and not natural_same_frame
                        counts["baseline_missing_new_frame_support"] += int(new_support)
                        counts["baseline_missing_third_support_opportunity"] += int(new_support and len(seed_frames) == 2)
                    if correct:
                        added_support[threshold][arm][target].add(fid)
                        if result["accepted_geometry"]:
                            accepted_support[threshold][arm][target].add(fid)
                row["thresholds"][key] = diagnostic
                if threshold == 0.15 and missing and not natural_same_frame and pool_ids:
                    projection = project_gt_bbox(gt[target], frame["pose"], frame["K"],
                                                 int(frame["width"]), int(frame["height"]))
                    if projection is not None:
                        pool = [{"anchor_id": index, "bbox": raw_boxes[index].tolist()} for index in pool_ids]
                        selected = choose_oracle_anchors(pool, projection)
                        if selected:
                            items.append({"track_id": track, "target_gt": target,
                                          "anchor_ids": [item["anchor_id"] for item in selected],
                                          "box2d_iou": [item["projected_gt_iou_2d"] for item in selected]})
            per_query.append(row)
        if items:
            oracle_frames.append({"frame_id": fid, "items": items})

    for threshold in THRESHOLDS:
        stats = summaries[f"{threshold:.2f}"]
        missing_indices = list(map(int, np.flatnonzero(base_best <= threshold)))
        stats["natural_observation_count"] = normal_count
        stats["missing_gt_natural_support_counts"] = {
            str(count): sum(len(normal_support[threshold][target]) == count for target in missing_indices)
            for count in (0, 1, 2)}
        stats["missing_gt_natural_support_counts"]["3+"] = sum(
            len(normal_support[threshold][target]) >= 3 for target in missing_indices)
        stats["support_sets"] = {}
        for arm in ("query", "control"):
            stats["support_sets"][arm] = {}
            for mode, support in (("all", added_support[threshold][arm]),
                                  ("accepted", accepted_support[threshold][arm])):
                stats["support_sets"][arm][mode] = {
                    "baseline_missing_gt_with_correct_query": sum(bool(support[target]) for target in missing_indices),
                    "baseline_missing_gt_newly_covered_beyond_all_natural": sum(
                        bool(support[target]) and not normal_support[threshold][target] for target in missing_indices),
                    "baseline_missing_gt_offline_union_reaches_three_frames": sum(
                        len(normal_support[threshold][target]) < 3 and
                        len(normal_support[threshold][target] | support[target]) >= 3 for target in missing_indices),
                }
    return {"scene_id": scene, "frames": len(frames), "query_records": per_query,
            "thresholds": summaries, "reported_timings": payload.get("timings", {})}, {
                "scene_id": scene, "frames": oracle_frames}


def add_numeric_tree(destination, source):
    for key, value in source.items():
        if isinstance(value, dict):
            add_numeric_tree(destination.setdefault(key, {}), value)
        elif isinstance(value, (int, float)):
            destination[key] = destination.get(key, 0) + value
        else:
            raise ValueError(f"Non-numeric summary field: {key}")


def audit_oracle_results(args):
    """Evaluate a separate, fully labelled GT-assisted lifting pass.

    Existing phase-one report/plan are immutable inputs in this mode.  The
    best-of-three result is only for the GT-projected-2D-selected sample within
    the recorded query pools; it cannot establish the full raw-pool ceiling.
    """
    target_path = args.oracle_results_output
    if target_path.exists():
        raise FileExistsError(f"Refusing to overwrite {target_path}")
    if target_path.resolve() in (args.output.resolve(), args.oracle_plan.resolve()):
        raise ValueError("Oracle result report must be separate from first-phase artifacts")
    first = json.loads(args.output.read_text())
    plan = json.loads(args.oracle_plan.read_text())
    if first.get("schema") != "boxfusion.ca1m_prenms_query_diagnostic.v1":
        raise ValueError("Unexpected first-phase report schema")
    if plan.get("gt_assisted") is not True or plan.get("schema") != "boxfusion.ca1m_prenms_query_oracle_plan.v1":
        raise ValueError("Oracle plan must be explicitly GT-assisted")
    if Path(first["run_dir"]).resolve() != args.run_dir.resolve():
        raise ValueError("Oracle and first-phase run directories differ")
    if any(Path(value).resolve() != args.baseline_root.resolve()
           for value in (first["baseline_root"], plan["baseline_root"])):
        raise ValueError("Oracle and first-phase baseline paths differ")
    hashes = dict(first["input_sha256"])
    changed = [path for path, value in hashes.items() if digest(path) != value]
    if changed:
        raise RuntimeError(f"First-phase inputs changed before oracle analysis: {changed}")
    hashes[str(args.output.resolve())] = digest(args.output)
    hashes[str(args.oracle_plan.resolve())] = digest(args.oracle_plan)
    complete_path = args.run_dir / "oracle_complete.json"
    complete = json.loads(complete_path.read_text())
    if complete.get("completed") is not True or complete.get("gt_assisted") is not True:
        raise ValueError("Oracle lifting run is incomplete or not explicitly GT-assisted")
    if complete.get("plan_sha256") != digest(args.oracle_plan):
        raise ValueError("Oracle run used a different plan")
    hashes[str(complete_path.resolve())] = digest(complete_path)
    first_scenes = {scene["scene_id"]: scene for scene in first["scenes"]}
    plan_scenes = plan["scenes"]
    if sorted(first_scenes) != sorted(scene["scene_id"] for scene in plan_scenes):
        raise ValueError("First-phase/oracle scene coverage differs")
    if args.expected_scenes is not None and len(first_scenes) != args.expected_scenes:
        raise ValueError("Oracle does not cover the requested number of scenes")
    total = {f"{threshold:.2f}": {} for threshold in THRESHOLDS}
    scene_results = []
    for scene_plan in plan_scenes:
        scene = scene_plan["scene_id"]
        source = args.run_dir / scene / "scene.json"
        raw_scene = json.loads(source.read_text())
        first_records = {(row["frame_id"], row["track_id"]): row for row in first_scenes[scene]["query_records"]}
        raw_records = {(int(frame["frame_id"]), int(event["track_id"])): event
                       for frame in raw_scene["frames"] for event in frame["queries"]}
        path = args.run_dir / scene / "oracle.json"
        oracle = json.loads(path.read_text())
        hashes[str(path.resolve())] = digest(path)
        if str(oracle["scene_id"]) != scene or oracle.get("gt_assisted") is not True:
            raise ValueError(f"{path}: scene or GT-assisted marker mismatch")
        expected, groups = {}, {}
        for frame in scene_plan["frames"]:
            fid = int(frame["frame_id"])
            for item in frame["items"]:
                track, target = int(item["track_id"]), int(item["target_gt"])
                group_key = (fid, track, target)
                if group_key in groups:
                    raise ValueError(f"{scene}: duplicate plan query")
                first_event = first_records.get((fid, track))
                raw_event = raw_records.get((fid, track))
                if first_event is None or raw_event is None:
                    raise ValueError(f"{scene}: oracle query absent from first phase")
                d15 = first_event["thresholds"]["0.15"]
                if d15["seed_gt"] != target or not d15["baseline_missing"] or d15["natural_same_frame_already_supported"]:
                    raise ValueError(f"{scene}: oracle plan changed first-phase eligibility")
                ids = [int(value) for value in item["anchor_ids"]]
                if not 1 <= len(ids) <= 3 or len(ids) != len(set(ids)):
                    raise ValueError(f"{scene}: expected one to three distinct planned anchors")
                if any(value not in raw_event["pool_ids"] for value in ids):
                    raise ValueError(f"{scene}: oracle anchor outside recorded query pool")
                groups[group_key] = []
                for anchor in ids:
                    expected[(fid, track, target, anchor)] = group_key
        observed = set()
        for observation in oracle["observations"]:
            key = tuple(int(observation[name]) for name in ("frame_id", "track_id", "target_gt", "anchor_id"))
            if key not in expected or key in observed:
                raise ValueError(f"{path}: unplanned or duplicate lifted anchor {key}")
            observed.add(key)
            if observation.get("corners") is not None:
                boxes_array([observation["corners"]], f"{path}:{key}")
            groups[expected[key]].append(observation)
        if observed != set(expected):
            raise ValueError(f"{path}: missing {len(set(expected) - observed)} planned observations")
        gt = boxes_array(np.load(Path(args.data_root) / scene / "after_filter_boxes.npy", allow_pickle=False), "oracle GT")
        details = []
        stats = {}
        for threshold in THRESHOLDS:
            key = f"{threshold:.2f}"
            stats[key] = {name: 0 for name in (
                "planned_query_events", "planned_anchors", "lifted_anchors", "anchors_matching_target",
                "events_with_target_hit_in_sample", "events_with_valid_seed_at_threshold",
                "events_with_valid_seed_and_target_hit", "new_frame_support_events",
                "third_support_opportunities", "events_with_target_hit_passing_existing_geometry_gate")}
            stats[key]["planned_query_events"] = len(groups)
            stats[key]["planned_anchors"] = len(expected)
        distinct_support = {threshold: defaultdict(set) for threshold in THRESHOLDS}
        natural_support = {threshold: defaultdict(set) for threshold in THRESHOLDS}
        for frame in raw_scene["frames"]:
            matrix = pairwise_iou(frame["normal"]["corners"], gt)
            for threshold in THRESHOLDS:
                for target in np.flatnonzero((matrix > threshold).any(0)):
                    natural_support[threshold][int(target)].add(int(frame["frame_id"]))
        for (fid, track, target), observations in groups.items():
            first_event = first_records[(fid, track)]
            raw_event = raw_records[(fid, track)]
            seeds = boxes_array(raw_event["seed_observations"], "oracle seed")
            corners = [item["corners"] for item in observations if item.get("corners") is not None]
            matrix = pairwise_iou(corners, gt)
            hit_identity = matrix.argmax(1) == target if len(matrix) else np.zeros(0, dtype=bool)
            target_ious = matrix[:, target] if len(matrix) else np.empty(0)
            geometry_ok = np.array([
                np.linalg.norm(np.asarray(box).mean(0) - seeds[-1].mean(0)) <= 0.5
                and pairwise_iou([box], [seeds[-1]])[0, 0] >= 0.1 for box in corners], dtype=bool)
            detail = {"frame_id": fid, "track_id": track, "target_gt": target,
                      "planned_anchors": len(observations), "lifted_anchors": len(corners),
                      "maximum_target_iou_in_gt2d_selected_sample": float(target_ious.max()) if len(target_ious) else 0.0,
                      "thresholds": {}}
            for threshold in THRESHOLDS:
                key = f"{threshold:.2f}"
                counts = stats[key]
                valid_seed = first_event["thresholds"][key]["seed_gt"] == target
                hits = hit_identity & (target_ious > threshold)
                hit = bool(hits.any())
                natural_same = fid in natural_support[threshold][target]
                counts["lifted_anchors"] += len(corners)
                counts["anchors_matching_target"] += int(hits.sum())
                counts["events_with_target_hit_in_sample"] += int(hit)
                counts["events_with_valid_seed_at_threshold"] += int(valid_seed)
                counts["events_with_valid_seed_and_target_hit"] += int(valid_seed and hit)
                counts["new_frame_support_events"] += int(valid_seed and hit and not natural_same)
                counts["third_support_opportunities"] += int(valid_seed and hit and not natural_same and len(first_event["seed_frames"]) == 2)
                counts["events_with_target_hit_passing_existing_geometry_gate"] += int(valid_seed and bool((hits & geometry_ok).any()))
                if valid_seed and hit:
                    distinct_support[threshold][target].add(fid)
                detail["thresholds"][key] = {"seed_valid": valid_seed, "target_hit": hit,
                                              "target_hit_and_geometry_gate": bool((hits & geometry_ok).any())}
            details.append(detail)
        for threshold in THRESHOLDS:
            counts = stats[f"{threshold:.2f}"]
            support = distinct_support[threshold]
            counts["distinct_target_gt_with_new_support"] = sum(bool(value) for value in support.values())
            counts["target_gt_offline_union_reaches_three_frames"] = sum(
                len(natural_support[threshold][target]) < 3 and
                len(natural_support[threshold][target] | support[target]) >= 3 for target in support)
            add_numeric_tree(total[f"{threshold:.2f}"], counts)
        scene_results.append({"scene_id": scene, "thresholds": stats, "queries": details})
    changed = [path for path, value in hashes.items() if digest(path) != value]
    if changed:
        raise RuntimeError(f"Inputs changed during oracle analysis: {changed}")
    result = {"schema": "boxfusion.ca1m_prenms_query_sampled_oracle_diagnostic.v1",
              "gt_assisted": True, "scene_count": len(scene_results),
              "interpretation": "GT-projected-2D top3 sample in recorded per-query pools only; not exhaustive raw-pool oracle, not AP/realized recall or births. Geometry gate recomputed against final past seed using center<=0.5m and AABB IoU>=0.1.",
              "first_phase_report": str(args.output.resolve()), "oracle_plan": str(args.oracle_plan.resolve()),
              "totals": total, "scenes": scene_results, "input_sha256": hashes}
    write_new_json(target_path, result)
    print(json.dumps({"oracle_report": str(target_path), "scenes": len(scene_results),
                      "totals": total}, ensure_ascii=False), flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--data-root", type=Path, default=Path("/extra/ZhaoX/boxfusion_ca1m"))
    parser.add_argument("--baseline-root", type=Path, default=ROOT / "results/ca1m_dual")
    parser.add_argument("--anchor-root", type=Path,
                        default=Path("/data/ZhaoX/OVM3D-Dett/boxfusion_boxer_dev/evaluation"))
    parser.add_argument("--expected-scenes", type=int)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--oracle-plan", required=True, type=Path)
    parser.add_argument("--oracle-results-output", type=Path,
                        help="Evaluate completed oracle.json files; read existing --output/--oracle-plan, never overwrite them")
    args = parser.parse_args(argv)
    if args.oracle_results_output is not None:
        audit_oracle_results(args)
        return
    if args.output.resolve() == args.oracle_plan.resolve():
        raise ValueError("Report and oracle plan must be separate paths")
    if args.output.exists() or args.oracle_plan.exists():
        raise FileExistsError("Refusing to overwrite report or oracle plan")
    scenes = sorted(args.run_dir.glob("*/scene.json"))
    if not scenes or (args.expected_scenes is not None and len(scenes) != args.expected_scenes):
        raise ValueError(f"Expected {args.expected_scenes or 'at least one'} scenes, got {len(scenes)}")
    anchor = verify_anchor(args.anchor_root)
    hashes = {str(Path(__file__).resolve()): digest(__file__),
              str(ROOT / "tools/audit_ca1m_nms_child_headroom.py"):
                  digest(ROOT / "tools/audit_ca1m_nms_child_headroom.py")}
    results, plans = [], []
    totals = {f"{threshold:.2f}": {} for threshold in THRESHOLDS}
    for index, path in enumerate(scenes, 1):
        result, plan = audit_scene(path, args, hashes)
        results.append(result)
        plans.append(plan)
        for threshold, summary in result["thresholds"].items():
            add_numeric_tree(totals[threshold], summary)
        print(f"[{index}/{len(scenes)}] audited {result['scene_id']}: {len(result['query_records'])} query events", flush=True)
    changed = [path for path, value in hashes.items() if digest(path) != value]
    if changed:
        raise RuntimeError(f"Inputs changed during read-only analysis: {changed}")
    output = {
        "schema": "boxfusion.ca1m_prenms_query_diagnostic.v1", "gt_assisted": True,
        "scene_count": len(results), "run_dir": str(args.run_dir.resolve()),
        "baseline_root": str(args.baseline_root.resolve()), "anchor": anchor,
        "interpretation": {
            "not_ap": True, "not_measured_recall_gain": True,
            "coverage": "GT-set coverage; may count multiple GT for one normal box; no score ranking or one-to-one evaluation",
            "correct_seed_instance": "all past seed observations have the same best GT above the strict threshold; selected query must have that same best GT",
            "three_frames": "distinct original frame IDs, not anchor count; third-support opportunities are not actual confirmed births",
            "offline_union": "GT-assisted union over saved selections, not a replay of updated causal memory; query was never committed to the inference memory",
            "oracle": "top 3 distinct raw boxes inside the existing per-query pool, selected by projected GT-box IoU with NMS 0.7; not the full raw-pool 3D oracle upper bound",
            "projection": "projected box does not establish visibility; GT only appears in this offline analysis and explicit oracle plan",
        },
        "totals": totals, "scenes": results, "input_sha256": hashes,
    }
    plan = {"schema": "boxfusion.ca1m_prenms_query_oracle_plan.v1", "gt_assisted": True,
            "seed_iou_threshold": 0.15, "strict_comparison": ">",
            "baseline_root": str(args.baseline_root.resolve()),
            "selection": "missing-seed-GT and no same-frame natural support; top3 projected GT IoU in existing query pool, NMS0.7",
            "scenes": plans}
    write_new_json(args.oracle_plan, plan)
    write_new_json(args.output, output)
    print(json.dumps({"scenes": len(results), "oracle_items": sum(len(frame["items"]) for scene in plans for frame in scene["frames"]),
                      "report": str(args.output), "oracle_plan": str(args.oracle_plan)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
