#!/usr/bin/env python3
"""Capture paired WeDetect proposals and Boxer outputs for CA-1M full107."""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

import integrated_online as online


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_json_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as f:
        json.dump(value, f, indent=2, sort_keys=True, allow_nan=False)
        f.write("\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    scenes = [x.strip() for x in args.scenes.read_text().splitlines()
              if x.strip() and not x.lstrip().startswith("#")]
    if len(scenes) != 107 or len(set(scenes)) != 107:
        raise ValueError("Expected exactly 107 unique scenes")
    args.output.mkdir(parents=True)
    source_paths = [Path(__file__), Path(online.__file__),
                    Path("boxfusion/boxer_lifter.py"),
                    Path("third_party/WeDetect/wedetect_uni_infer.py"), args.scenes]
    initial = {str(p.resolve()): sha256(p) for p in source_paths}
    write_json_new(args.output / "protocol.json", {
        "schema": "boxfusion.ca1m_wedetect_boxer_capture.v1",
        "scenes": scenes, "score_min": online.SCORE_LIFT,
        "topk_per_frame": online.TOPK_PER_FRAME, "gap": 20,
        "model_forward": "frozen WeDetect then shared cached Boxer, one pass per keyframe",
        "production_outputs_modified": False, "GT_read": False,
        "source_sha256": initial})
    model, adapter = online.load_models()
    completed = []
    start_all = time.time()
    for number, scene in enumerate(scenes, 1):
        path = args.output / f"{scene}.npz"
        if path.exists():
            raise FileExistsError(path)
        K, Kd, kfs, width, height, _ = online.load_ca1m_scene(scene)
        boxes, scores, frames, corners, valid_depth = [], [], [], [], []
        start = time.time()
        for frame, pose, rgb_path, depth_path in kfs:
            with torch.inference_mode():
                result = model([rgb_path])[0]
            pb = result["bboxes"].float().cpu().numpy()
            ps = result["scores"].float().cpu().numpy()
            keep = np.flatnonzero(ps >= online.SCORE_LIFT)
            if len(keep) > online.TOPK_PER_FRAME:
                keep = keep[np.argsort(-ps[keep], kind="stable")[:online.TOPK_PER_FRAME]]
            pb, ps = pb[keep], ps[keep]
            if not len(pb):
                continue
            rgb = np.asarray(Image.open(rgb_path).convert("RGB"))
            depth_raw = np.asarray(Image.open(depth_path))
            depth = depth_raw.astype(np.float32) / 1000.0
            datum, meta = adapter._make_datum(
                image=rgb, depth=depth, boxes_xyxy=torch.from_numpy(pb).float(),
                image_K=K, depth_K=Kd, camera_to_world=pose,
                scene_id=scene, frame_id=int(frame))
            output, _, _ = adapter.forward_raw_with_feature_cache(
                datum, scene_id=scene, frame_id=int(frame),
                encoder_input_sha256=meta["encoder_input_sha256"])
            obb = output["obbs_pr_w"][0]
            center = obb.bb3_center_world.float().cpu().numpy()
            extent = np.abs(obb.bb3_diagonal.float().cpu().numpy()) + 1e-6
            rotation = obb.T_world_object.R.float().cpu().numpy()
            if not (len(pb) == len(center) == len(extent) == len(rotation)):
                raise ValueError(f"{scene}/{frame}: Boxer cardinality mismatch")
            sign = np.array([[x, y, z] for x in (-1, 1)
                             for y in (-1, 1) for z in (-1, 1)])
            cc = center[:, None] + np.einsum(
                "nki,nji->nkj", sign[None] * extent[:, None] / 2, rotation)
            if not np.isfinite(cc).all():
                raise ValueError(f"{scene}/{frame}: invalid Boxer geometry")
            boxes.append(pb.astype(np.float32)); scores.append(ps.astype(np.float32))
            frames.append(np.full(len(pb), frame, dtype=np.int64))
            corners.append(cc.astype(np.float32))
            valid_depth.append(np.full(len(pb), np.count_nonzero(depth > 0), dtype=np.int64))
        arrays = {
            "boxes2d": np.concatenate(boxes) if boxes else np.empty((0, 4), np.float32),
            "scores": np.concatenate(scores) if scores else np.empty(0, np.float32),
            "frame_ids": np.concatenate(frames) if frames else np.empty(0, np.int64),
            "corners": np.concatenate(corners) if corners else np.empty((0, 8, 3), np.float32),
            "frame_valid_depth_pixels": np.concatenate(valid_depth) if valid_depth else np.empty(0, np.int64),
            "K_rgb": K.astype(np.float32), "K_depth": Kd.astype(np.float32),
            "width_height": np.array([width, height], np.int64),
        }
        np.savez_compressed(path, **arrays)
        completed.append(scene)
        print(f"[{number}/107] {scene}: kfs={len(kfs)} proposals={len(arrays['scores'])} "
              f"seconds={time.time()-start:.1f}", flush=True)
    final = {str(p.resolve()): sha256(p) for p in source_paths}
    if final != initial:
        raise RuntimeError("Source changed during capture")
    write_json_new(args.output / "complete.json", {
        "completed": True, "scenes": completed, "scene_count": len(completed),
        "wall_seconds": time.time() - start_all, "source_sha256_unchanged": True})


if __name__ == "__main__":
    main()
