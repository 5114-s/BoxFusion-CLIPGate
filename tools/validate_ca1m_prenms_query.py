#!/usr/bin/env python3
"""Isolated diagnostic of causal queries into WeDetect's pre-NMS boxes.

probe mode never opens GT or final predictions. State is updated ONLY from
ordinary detector observations; neither query nor control feeds itself back.
oracle mode is a separately marked, GT-assisted sampling diagnostic, NOT an
online method or an exhaustive 3D oracle. No evaluator prediction is written.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "third_party/WeDetect"))

from tools.ca1m_prenms_query_core import (
    PendingMemory, box_iou3d, normalize_features, project_box, select_local,
)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    with path.open("x") as f:
        json.dump(value, f, ensure_ascii=False, allow_nan=False)


def forbid_gt(event, args):
    if event == "open" and args and isinstance(args[0], (str, bytes)):
        name = str(args[0])
        if "after_filter_boxes.npy" in name or "full_annotations.json" in name:
            raise RuntimeError("GT access forbidden in model execution; use offline evaluator")


def build_lifter(run_dir):
    from boxfusion.boxer_lifter import build_lifting_adapter
    cfg = {"lifting": {"backend": "boxer", "boxer": {
        "mode": "observer", "apply_stage": "post_filter",
        "official_root": "/data/ZhaoX/OVM3D-Dett/boxfusion_boxer_dev/third_party/boxer",
        "checkpoint": "/data/ZhaoX/OVM3D-Dett/boxfusion_boxer_dev/third_party/boxer/ckpts/boxernet_hw960in2x6d768-c88128f8.ckpt",
        "expected_commit": "1f86542dc342a4b1d474c87c97c5d1d6566d9148",
        "checkpoint_sha256": "d5a30b348a8f5b0e5990ff3aa0e8f473ce77d860da22586322e7f47abc83ca6f",
        "dinov3_sha256": "4057cbaaad8c16657adb09d6815f28d4164eeba30532fde23f0d17313124caea",
        "precision": "bfloat16", "use_sdp": True, "sdp_samples": 10000,
        "seed": 0, "cache_image_features": True,
        "diagnostics_dir": str(run_dir / "boxer_diagnostics"),
    }}}
    return build_lifting_adapter(cfg, device="cuda", code_root=str(REPO))


class DenseCapture:
    """Observe the unmodified head's three level outputs; do not alter NMS."""

    def __init__(self):
        import torch
        from wedetect_uni_infer import SimpleYOLOWorldDetector
        self.torch = torch
        self.model = SimpleYOLOWorldDetector(
            backbone_size="base", prompt_dim=768, num_prompts=256, num_proposals=300,
        )
        ck = torch.load(REPO / "third_party/WeDetect/wedetect_base_uni.pth",
                        map_location="cpu", weights_only=False)
        for key in list(ck):
            if "backbone" in key:
                ck[key.replace("backbone.image_model.model.", "backbone.")] = ck.pop(key)
        for key in list(ck):
            if "bbox_head" in key:
                nk = key.replace("bbox_head.head_module.", "bbox_head.")
                nk = nk.replace("0.2.", "0.6.").replace("1.2.", "1.6.").replace("2.2.", "2.6.")
                nk = nk.replace("1.bn", "4").replace("1.conv", "3").replace("0.bn", "1").replace("0.conv", "0")
                ck[nk] = ck.pop(key)
        incompatible = self.model.load_state_dict(ck, strict=False)
        # The feature-only backbone omits the pretrained classification head
        # and final norm. The production loader ignores these same four keys;
        # reject every other mismatch and all missing inference parameters.
        permitted_unused = {"backbone.norm.weight", "backbone.norm.bias",
                            "backbone.head.weight", "backbone.head.bias"}
        if incompatible.missing_keys or set(incompatible.unexpected_keys) - permitted_unused:
            raise RuntimeError(f"Unexpected checkpoint mismatch: {incompatible}")
        self.model = self.model.cuda().eval()
        self.parts = []
        original = self.model.head_module_forward_single

        def capture(*args, **kwargs):
            value = original(*args, **kwargs)
            self.parts.append(value)
            return value

        self.model.head_module_forward_single = capture

    def forward(self, pil):
        from scipy.spatial import cKDTree
        from wedetect_uni_infer import distance2bbox, letterbox
        torch = self.torch
        self.parts.clear()
        torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.inference_mode():
            ordinary = self.model([pil])[0]
        torch.cuda.synchronize()
        forward_ms = (time.perf_counter() - start) * 1000
        start = time.perf_counter()
        if len(self.parts) != 3:
            raise RuntimeError("Expected exactly three captured feature levels")
        embeds = [x[0].permute(0, 2, 3, 1).reshape(1, -1, x[0].shape[1]) for x in self.parts]
        dists = [x[1].permute(0, 2, 3, 1).reshape(1, -1, 4) for x in self.parts]
        logits = [x[2].permute(0, 2, 3, 1).reshape(1, -1, x[2].shape[1]) for x in self.parts]
        sizes = [x[1].shape[2:] for x in self.parts]
        priors = self.model.prior_generator.grid_priors(
            sizes, dtype=embeds[0].dtype, device=embeds[0].device,
        )
        strides = torch.cat([p.new_full((len(p),), s)
                             for p, s in zip(priors, self.model.bbox_head.featmap_strides)])
        raw_boxes = distance2bbox(torch.cat(priors)[None], torch.cat(dists, 1) * strides[None, :, None])[0]
        _, ratio, offset = letterbox(pil, self.model.img_size)
        raw_boxes = (raw_boxes - raw_boxes.new_tensor([offset[0], offset[1], offset[0], offset[1]])) / ratio
        raw_boxes[:, 0::2].clamp_(0, pil.width)
        raw_boxes[:, 1::2].clamp_(0, pil.height)
        raw_boxes = raw_boxes.float().cpu().numpy()
        raw_embed = torch.cat(embeds, 1)[0].float().cpu().numpy()
        raw_scores = torch.cat(logits, 1).sigmoid().amax(-1)[0].float().cpu().numpy()
        post_boxes = ordinary["bboxes"].float().cpu().numpy()
        post_embed = ordinary["embeddings"].float().cpu().numpy()
        post_scores = ordinary["scores"].float().cpu().numpy()
        # Recover grid IDs without changing the detector's own filter/NMS.
        tree = cKDTree(raw_boxes)
        ids = []
        for b, e in zip(post_boxes, post_embed):
            matches = tree.query_ball_point(b, r=0.003)
            if not matches:
                raise RuntimeError("Could not recover original proposal's dense anchor")
            errors = np.max(np.abs(raw_embed[matches] - e), axis=1)
            chosen = matches[int(np.argmin(errors))]
            if np.max(np.abs(raw_boxes[chosen] - b)) > 0.002 or errors.min() > 1e-5:
                raise RuntimeError("Dense/post-NMS tensor identity check failed")
            ids.append(chosen)
        recovery_ms = (time.perf_counter() - start) * 1000
        self.parts.clear()
        return {"boxes": raw_boxes, "scores": raw_scores,
                "embeddings": normalize_features(raw_embed),
                "post_boxes": post_boxes, "post_scores": post_scores,
                "post_embeddings": normalize_features(post_embed),
                "post_ids": np.array(ids, dtype=np.int64),
                "forward_ms": forward_ms, "recovery_ms": recovery_ms}


def lift(adapter, scene, frame, rgb, depth, K, Kd, pose, boxes):
    import torch
    if not len(boxes):
        return np.empty((0, 8, 3)), 0.0, False
    start = time.perf_counter()
    datum, meta = adapter._make_datum(
        image=rgb, depth=depth, boxes_xyxy=torch.from_numpy(np.asarray(boxes, np.float32)),
        image_K=K, depth_K=Kd, camera_to_world=pose, scene_id=scene, frame_id=int(frame),
    )
    output, _, hit = adapter.forward_raw_with_feature_cache(
        datum, scene_id=scene, frame_id=int(frame), encoder_input_sha256=meta["encoder_input_sha256"],
    )
    obbs = output["obbs_pr_w"][0]
    c = obbs.bb3_center_world.float().cpu().numpy()
    d = np.abs(obbs.bb3_diagonal.float().cpu().numpy()) + 1e-6
    r = obbs.T_world_object.R.float().cpu().numpy()
    signs = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)])
    corners = c[:, None] + np.einsum("nki,nji->nkj", signs[None] * d[:, None] / 2, r)
    return corners, (time.perf_counter() - start) * 1000, hit


def valid3d(corners):
    return np.isfinite(corners).all(axis=(1, 2)) & (np.ptp(corners, axis=1) > 0).all(1)


def supports(corners, reference):
    if not len(corners):
        return np.zeros(0, dtype=bool)
    return ((np.linalg.norm(corners.mean(1) - reference.mean(0), axis=1) <= 0.5)
            & (box_iou3d(corners, reference[None])[:, 0] >= 0.1))


def load_frame(root, frame):
    from PIL import Image
    with Image.open(root / "rgb" / f"{frame}.png") as image:
        pil = image.convert("RGB")
    with Image.open(root / "depth" / f"{frame}.png") as image:
        depth = np.asarray(image).astype(np.float32) / 1000
    return pil, np.asarray(pil), depth


def probe(args):
    import torch
    out = Path(args.run_dir).resolve()
    out.mkdir(parents=True, exist_ok=False)
    scenes = [s.strip() for s in Path(args.scene_list).read_text().splitlines() if s.strip()][:args.scenes]
    if len(scenes) != args.scenes or len(set(scenes)) != len(scenes) or not all(s.isdecimal() for s in scenes):
        raise ValueError("Invalid pilot scene selection")
    source_paths = [Path(__file__), REPO / "tools/ca1m_prenms_query_core.py",
                    REPO / "third_party/WeDetect/wedetect_uni_infer.py", REPO / "boxfusion/boxer_lifter.py"]
    frozen = {str(p): digest(p) for p in source_paths}
    write_json(out / "protocol.json", {
        "schema": "boxfusion.prenms_query_probe.v1", "scenes": scenes,
        "selection": "first N listed scenes, development pilot, not held-out",
        "gap": args.gap, "max_keyframes": args.max_keyframes, "normal_score_min": 0.05,
        "normal_topk": 150, "pending_cap": 64, "query_targets_per_frame": 8,
        "query_and_control_each_topk": 1, "memory_commit": "normal observations only",
        "memory_ttl_keyframes": 10, "normal_confirmation_frames": 3,
        "production_predictions_modified": False, "ground_truth_in_probe": False,
        "online_fps_claim": False, "frozen_sha256": frozen,
    })
    sys.addaudithook(forbid_gt)
    torch.manual_seed(0)
    np.random.seed(0)
    detector = DenseCapture()
    adapter = build_lifter(out)
    for scene in scenes:
        root = Path(args.data_root) / scene
        directory = out / scene
        directory.mkdir()
        poses = np.load(root / "all_poses.npy", allow_pickle=False)
        K = np.loadtxt(root / "K_rgb.txt").reshape(3, 3)
        Kd = np.loadtxt(root / "K_depth.txt").reshape(3, 3)
        memory = PendingMemory(max_tracks=64, ttl=10, max_obs=3)
        records = []
        scene_started = time.perf_counter()
        frame_ids = list(range(0, len(poses), args.gap))
        if args.max_keyframes:
            frame_ids = frame_ids[:args.max_keyframes]
        for ordinal, frame in enumerate(frame_ids):
            if not ((root / "rgb" / f"{frame}.png").is_file() and (root / "depth" / f"{frame}.png").is_file()):
                raise ValueError(f"Missing sampled frame: {scene}/{frame}")
            pose = poses[frame]
            if not np.isfinite(pose).all():
                continue
            started = time.perf_counter()
            pil, rgb, depth = load_frame(root, frame)
            raw = detector.forward(pil)
            keep = np.flatnonzero(raw["post_scores"] >= 0.05)
            if len(keep) > 150:
                keep = keep[np.argsort(-raw["post_scores"][keep], kind="stable")[:150]]
            normal_ids = raw["post_ids"][keep]
            normal_scores = raw["post_scores"][keep]
            normal_features = raw["post_embeddings"][keep]
            normal_boxes = raw["post_boxes"][keep]
            normal_corners, normal_ms, _ = lift(adapter, scene, frame, rgb, depth, K, Kd, pose, normal_boxes)
            finite = valid3d(normal_corners)
            if not finite.all():
                raise ValueError("Normal lifting produced invalid geometry")
            targets = []
            search_started = time.perf_counter()
            # Visibility and current ordinary support are checked before the budget cap.
            for track in memory.eligible(ordinal, limit=64):
                reference = np.asarray(track["obs"][-1])
                if supports(normal_corners, reference).any():
                    continue
                projection = project_box(reference, pose, K, pil.width, pil.height)
                if projection is None:
                    continue
                selected = select_local(raw["boxes"], raw["scores"], raw["embeddings"], projection,
                                        np.asarray(track["features"]), normal_ids, limit=1)
                if not len(selected["pool_ids"]):
                    continue
                record = {"track_id": int(track["id"]), "seed_frames": [int(x) for x in track["frames"]],
                          "seed_observations": [np.asarray(c).tolist() for c in track["obs"]],
                          "seed_score": float(track["score"]), "projection": projection.tolist(),
                          "pool_ids": selected["pool_ids"].tolist(), "query": None, "control": None}
                for arm, key in (("query", "query_ids"), ("control", "control_ids")):
                    if len(selected[key]):
                        anchor = int(selected[key][0])
                        cosine = float(np.max(np.asarray(track["features"]) @ raw["embeddings"][anchor]))
                        record[arm] = {"anchor_id": anchor, "box2d": raw["boxes"][anchor].tolist(),
                                       "score": float(raw["scores"][anchor]), "cosine": cosine,
                                       "corners": None, "accepted_geometry": False}
                targets.append(record)
                if len(targets) >= 8:
                    break
            search_ms = (time.perf_counter() - search_started) * 1000
            unique_ids = sorted({t[arm]["anchor_id"] for t in targets for arm in ("query", "control") if t[arm]})
            extra_corners, extra_ms, hit = lift(adapter, scene, frame, rgb, depth, K, Kd, pose, raw["boxes"][unique_ids])
            if unique_ids and len(normal_boxes) and not hit:
                raise RuntimeError("Extra Boxer queries did not reuse current frame's encoder")
            by_anchor = dict(zip(unique_ids, extra_corners))
            for t in targets:
                reference = np.asarray(t["seed_observations"][-1])
                for arm in ("query", "control"):
                    item = t[arm]
                    if item is not None:
                        c = by_anchor[item["anchor_id"]]
                        if valid3d(c[None])[0]:
                            item["corners"] = c.tolist()
                            item["accepted_geometry"] = bool(supports(c[None], reference)[0])
            # No GT, query/control, or final-map information reaches state commit.
            memory.commit(ordinal, frame, normal_corners, normal_features, normal_scores)
            raw_name = f"raw_{frame:06d}.npz"
            np.savez_compressed(directory / raw_name, boxes=raw["boxes"], scores=raw["scores"])
            rec = {"frame_id": frame, "ordinal": ordinal, "pose": pose.tolist(), "K": K.tolist(),
                   "width": pil.width, "height": pil.height, "raw_file": raw_name,
                   "normal": {"corners": normal_corners.tolist(), "scores": normal_scores.tolist(),
                              "anchor_ids": normal_ids.tolist()}, "queries": targets,
                   "timings_ms": {"wedetect_forward": raw["forward_ms"], "dense_recovery": raw["recovery_ms"],
                                  "normal_lift": normal_ms, "query_search_cpu": search_ms,
                                  "both_arms_extra_lift": extra_ms, "extra_encoder_cache_hit": bool(hit),
                                  "frame_wall_with_diagnostic_io": (time.perf_counter()-started)*1000}}
            records.append(rec)
            print(f"{scene} kf={ordinal+1}/{len(frame_ids)} frame={frame} normal={len(keep)} targets={len(targets)} extra={len(unique_ids)} cache={hit}", flush=True)
        write_json(directory / "scene.json", {"scene_id": scene, "completed": True, "frames": records,
                   "timings": {"wall_seconds": time.perf_counter() - scene_started}})
    for path, expected in frozen.items():
        if digest(path) != expected:
            raise RuntimeError(f"Source changed during probe: {path}")
    write_json(out / "probe_complete.json", {"completed": True, "scenes": scenes, "source_hashes_unchanged": True})
    print("PROBE_COMPLETE", flush=True)


def oracle(args):
    out = Path(args.run_dir).resolve()
    plan = json.loads(Path(args.oracle_plan).read_text())
    if plan.get("gt_assisted") is not True:
        raise ValueError("Oracle plan must be explicitly GT-assisted")
    sys.addaudithook(forbid_gt)
    adapter = None
    for scene_plan in plan["scenes"]:
        scene = scene_plan["scene_id"]
        directory = out / scene
        target = directory / "oracle.json"
        if target.exists():
            raise FileExistsError(target)
        root = Path(args.data_root) / scene
        poses = np.load(root / "all_poses.npy", allow_pickle=False)
        K = np.loadtxt(root / "K_rgb.txt").reshape(3, 3)
        Kd = np.loadtxt(root / "K_depth.txt").reshape(3, 3)
        observations = []
        for row in scene_plan["frames"]:
            frame = row["frame_id"]
            selected = sorted({int(a) for item in row["items"] for a in item["anchor_ids"]})
            if not selected:
                continue
            if adapter is None:
                adapter = build_lifter(out / "oracle_run")
            with np.load(directory / f"raw_{frame:06d}.npz", allow_pickle=False) as raw:
                boxes = raw["boxes"][selected]
            _, rgb, depth = load_frame(root, frame)
            corners, elapsed, _ = lift(adapter, scene, frame, rgb, depth, K, Kd, poses[frame], boxes)
            by_anchor = {a: c for a, c in zip(selected, corners)}
            for item in row["items"]:
                for a in item["anchor_ids"]:
                    c = by_anchor[int(a)]
                    observations.append({"frame_id": frame, "track_id": item["track_id"],
                                         "target_gt": item["target_gt"], "anchor_id": int(a),
                                         "corners": c.tolist() if valid3d(c[None])[0] else None})
            print(f"ORACLE {scene} frame={frame} anchors={len(selected)} lift_ms={elapsed:.1f}", flush=True)
        write_json(target, {"scene_id": scene, "gt_assisted": True, "observations": observations})
    write_json(out / "oracle_complete.json", {"completed": True, "gt_assisted": True,
                                               "plan_sha256": digest(args.oracle_plan)})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("probe", "oracle"), default="probe")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--data-root", default="/extra/ZhaoX/boxfusion_ca1m")
    parser.add_argument("--scene-list", default=str(REPO / "tools/boxfusion_tr3d_pipeline/evaluation/data_util/meta_data/ca1m_val_full107.txt"))
    parser.add_argument("--scenes", type=int, default=3)
    parser.add_argument("--gap", type=int, default=20)
    parser.add_argument("--max-keyframes", type=int, default=0)
    parser.add_argument("--oracle-plan")
    args = parser.parse_args()
    if args.scenes < 1 or args.gap < 1 or args.max_keyframes < 0:
        parser.error("invalid positive pilot parameters")
    if args.mode == "oracle":
        if not args.oracle_plan:
            parser.error("--oracle-plan is required in oracle mode")
        oracle(args)
    else:
        probe(args)


if __name__ == "__main__":
    main()
