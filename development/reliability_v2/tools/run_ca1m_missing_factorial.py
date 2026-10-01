#!/usr/bin/env python3
"""Complete the two missing CA-1M factorial arms: base+M2 and base+A+M2.

The archived CA-1M M1 run rewrote native map rows, so its M2 scores cannot be
copied back onto the native baseline.  This script reruns only the frozen
WeDetect-Uni 2D proposal forward, caches the exact post-NMS proposals, and
then applies the frozen CA-1M nativelogit M2 rule to (1) native base
and (2) native base plus the already-frozen strict-online M1-A births.
No GT is read until the final evaluation pass.
"""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "third_party/WeDetect"))

from tools.audit_ca1m_nms_child_headroom import sha256, valid_boxes
from tools.audit_m1m2_remaining_children import GT_ROOT, read_prediction
from tools.audit_seedless_ablation_matrix import RUNS, THRESHOLDS
from tools.true_fusion_audit_core import class_agnostic_ap

PROTOCOL = RUNS["ca1m"] / "protocol.json"
DATA = Path("/extra/ZhaoX/boxfusion_ca1m")
BASE = ROOT / "results/ca1m_thr15"
P = ROOT / "results/ca1m_dual_nom2"
P_M2 = ROOT / "results/ca1m_dual"
ONLINE = ROOT / "reports/m1a_strict_online_ca1m_20260916/predictions"
TAU = 0.5
TOPK = 150
SCORE_FLOOR = 0.05


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def load_model(device_index: int):
    import torch
    from wedetect_uni_infer import SimpleYOLOWorldDetector

    torch.cuda.set_device(device_index)
    model = SimpleYOLOWorldDetector(
        backbone_size="base", prompt_dim=768, num_prompts=256,
        num_proposals=300)
    checkpoint = torch.load(
        ROOT / "third_party/WeDetect/wedetect_base_uni.pth",
        map_location="cpu", weights_only=False)
    for key in list(checkpoint):
        if "backbone" in key:
            checkpoint[key.replace(
                "backbone.image_model.model.", "backbone.")] = checkpoint.pop(key)
    for key in list(checkpoint):
        if "bbox_head" in key:
            new = key.replace("bbox_head.head_module.", "bbox_head.")
            new = new.replace("0.2.", "0.6.").replace(
                "1.2.", "1.6.").replace("2.2.", "2.6.")
            new = new.replace("1.bn", "4").replace("1.conv", "3")
            new = new.replace("0.bn", "1").replace("0.conv", "0")
            checkpoint[new] = checkpoint.pop(key)
    incompatible = model.load_state_dict(checkpoint, strict=False)
    permitted = {"backbone.norm.weight", "backbone.norm.bias",
                 "backbone.head.weight", "backbone.head.bias"}
    if incompatible.missing_keys or set(incompatible.unexpected_keys) - permitted:
        raise RuntimeError(f"checkpoint mismatch: {incompatible}")
    return model.cuda().eval(), torch


def capture(output: Path, device_index: int, limit: int) -> None:
    protocol = json.loads(PROTOCOL.read_text())
    scenes = protocol["scenes"][:limit or None]
    cache = output / "proposal_cache"
    cache.mkdir(parents=True, exist_ok=True)
    model, torch = load_model(device_index)
    started = time.perf_counter()
    for ordinal, scene in enumerate(scenes, 1):
        target = cache / f"{scene}.npz"
        if target.is_file():
            print(f"[{ordinal}/{len(scenes)}] {scene} cached", flush=True)
            continue
        boxes, frame_ids = [], []
        for frame in protocol["frames"][scene]:
            image_path = DATA / scene / "rgb" / f"{frame}.png"
            with Image.open(image_path) as image:
                pil = image.convert("RGB")
            with torch.inference_mode():
                prediction = model([pil])[0]
            scores = prediction["scores"].float().cpu().numpy()
            frame_boxes = prediction["bboxes"].float().cpu().numpy()
            keep = np.flatnonzero(scores >= SCORE_FLOOR)
            if len(keep) > TOPK:
                keep = keep[np.argsort(-scores[keep], kind="stable")[:TOPK]]
            boxes.extend(frame_boxes[keep].tolist())
            frame_ids.extend([int(frame)] * len(keep))
        np.savez_compressed(
            target,
            boxes=np.asarray(boxes, dtype=np.float32).reshape(-1, 4),
            frame_ids=np.asarray(frame_ids, dtype=np.int32))
        print(f"[{ordinal}/{len(scenes)}] {scene} proposals={len(boxes)}", flush=True)
    write_json(output / "capture.json", {
        "completed": len(scenes) == len(protocol["scenes"]),
        "scenes": scenes,
        "frames": sum(len(protocol["frames"][s]) for s in scenes),
        "parameters": {"score_floor": SCORE_FLOOR, "topk_per_frame": TOPK,
                       "device_index": device_index},
        "protocol_sha256": sha256(PROTOCOL),
        "checkpoint_sha256": sha256(
            ROOT / "third_party/WeDetect/wedetect_base_uni.pth"),
        "wall_seconds": time.perf_counter() - started,
    })


def project(corners: np.ndarray, pose: np.ndarray, K: np.ndarray,
            width: int = 1024, height: int = 768):
    inverse = np.linalg.inv(pose)
    camera = (inverse[:3, :3] @ corners.T).T + inverse[:3, 3]
    if (camera[:, 2] < .1).any():
        return None
    pixels = (K @ camera.T).T
    pixels = pixels[:, :2] / pixels[:, 2:3]
    x1, y1 = pixels.min(0)
    x2, y2 = pixels.max(0)
    if x2 <= 0 or y2 <= 0 or x1 >= width or y1 >= height:
        return None
    if x2 - x1 < 2 or y2 - y1 < 2:
        return None
    return np.asarray([max(0., x1), max(0., y1),
                       min(float(width), x2), min(float(height), y2)])


def max_support(row_boxes: np.ndarray, proposals: dict[int, np.ndarray],
                poses: np.ndarray, K: np.ndarray) -> np.ndarray:
    """Maximum per-row reprojection support used by the archived CA-1M M2.

    CA-1M's frozen run predates the optional exclusive assignment switch.  A
    three-scene audit against its saved per-row support reproduces it to
    float32 precision only with independent row-wise maxima (exclusive=False).
    """
    support = np.zeros(len(row_boxes), dtype=np.float64)
    for frame in sorted(proposals):
        props = proposals[frame]
        if not len(props):
            continue
        for row_id, corners in enumerate(row_boxes):
            box = project(corners, poses[frame], K)
            if box is None:
                continue
            x1 = np.maximum(box[0], props[:, 0])
            y1 = np.maximum(box[1], props[:, 1])
            x2 = np.minimum(box[2], props[:, 2])
            y2 = np.minimum(box[3], props[:, 3])
            intersection = np.maximum(0., x2-x1) * np.maximum(0., y2-y1)
            union = ((box[2]-box[0])*(box[3]-box[1])
                     + (props[:, 2]-props[:, 0])*(props[:, 3]-props[:, 1])
                     - intersection)
            values = intersection / np.maximum(union, 1e-9)
            support[row_id] = max(support[row_id], float(values.max()))
    return support


def rescore(scores: np.ndarray, support: np.ndarray) -> np.ndarray:
    clipped = np.clip(scores.astype(np.float64), 1e-4, 1-1e-4)
    logits = np.log(clipped / (1-clipped))
    logits += 2.0 * np.maximum(0., support - TAU)
    return 1.0 / (1.0 + np.exp(-logits))


def load_payload(path: Path):
    with path.open("rb") as handle:
        return pickle.load(handle)


def save_payload(template, boxes, scores, path: Path):
    payload = list(template)
    payload[0] = [(0, box, float(score)) for box, score in zip(boxes, scores)]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(payload, handle)


def evaluate(output: Path) -> None:
    protocol = json.loads(PROTOCOL.read_text())
    scenes = protocol["scenes"]
    capture_manifest = json.loads((output / "capture.json").read_text())
    if not capture_manifest["completed"] or capture_manifest["scenes"] != scenes:
        raise RuntimeError("full107 proposal capture is incomplete")
    roots = {name: output / "predictions" / name
             for name in ("base_m2", "a_m2", "p_m2", "p_a_m2")}
    predictions = {name: {} for name in
                   ("base", "p", "a", "p_a", "base_m2", "a_m2",
                    "p_m2", "p_a_m2")}
    gts, score_audit = {}, {"rows": 0, "changed": 0,
                             "p_m2_max_abs_score_error": 0.0}
    hashes = {str(PROTOCOL.resolve()): sha256(PROTOCOL)}
    for ordinal, scene in enumerate(scenes, 1):
        cache_path = output / "proposal_cache" / f"{scene}.npz"
        with np.load(cache_path,
                     allow_pickle=False) as values:
            proposal_boxes = values["boxes"].astype(np.float64)
            proposal_frames = values["frame_ids"].astype(np.int64)
        by_frame = {int(frame): proposal_boxes[proposal_frames == frame]
                    for frame in np.unique(proposal_frames)}
        poses = np.load(DATA / scene / "all_poses.npy", allow_pickle=False)
        K = np.loadtxt(DATA / scene / "K_rgb.txt").reshape(3, 3)

        base_payload = load_payload(BASE / f"{scene}_boxes.pkl")
        base_boxes, base_scores = read_prediction(BASE / f"{scene}_boxes.pkl")
        p_payload = load_payload(P / f"{scene}_boxes.pkl")
        p_boxes, p_scores = read_prediction(P / f"{scene}_boxes.pkl")
        stored_p_boxes, stored_p_scores = read_prediction(
            P_M2 / f"{scene}_boxes.pkl")
        if not np.array_equal(p_boxes, stored_p_boxes):
            raise RuntimeError(f"paired P/P+M2 geometry changed: {scene}")

        online_boxes, online_scores = read_prediction(
            ONLINE / f"{scene}_boxes.pkl")
        prefix_boxes, _ = read_prediction(P_M2 / f"{scene}_boxes.pkl")
        if not np.array_equal(online_boxes[:len(prefix_boxes)], prefix_boxes):
            raise RuntimeError(f"online M1-A prefix changed: {scene}")
        birth_boxes = online_boxes[len(prefix_boxes):]
        birth_scores = online_scores[len(prefix_boxes):]

        base_support = max_support(base_boxes, by_frame, poses, K)
        base_m2_scores = rescore(base_scores, base_support)
        a_boxes = np.concatenate([base_boxes, birth_boxes])
        a_scores = np.concatenate([base_scores, birth_scores])
        a_m2_scores = np.concatenate([
            base_m2_scores, birth_scores])

        p_support = max_support(p_boxes, by_frame, poses, K)
        # The paired archived P arm contains native rows and M1-P births; M2
        # deliberately changes native scores only.  Its lossless sidecar is
        # the authoritative source mask for this anchor check.
        p_state = json.loads(
            (P_M2 / f"{scene}_boxes.pkl.dual_state.json").read_text())
        native_mask = np.asarray(
            [row["source"] == "native" for row in p_state["rows"]],
            dtype=bool)
        if len(native_mask) != len(p_scores):
            raise RuntimeError(f"P+M2 sidecar row mismatch: {scene}")
        replay_p_scores = p_scores.copy()
        replay_p_scores[native_mask] = rescore(
            p_scores[native_mask], p_support[native_mask])
        error = (float(np.max(np.abs(replay_p_scores - stored_p_scores)))
                 if len(p_scores) else 0.0)
        score_audit["p_m2_max_abs_score_error"] = max(
            score_audit["p_m2_max_abs_score_error"], error)
        score_audit["rows"] += len(base_scores)
        score_audit["changed"] += int(np.sum(base_m2_scores != base_scores))

        p_a_boxes = np.concatenate([p_boxes, birth_boxes])
        p_a_scores = np.concatenate([p_scores, birth_scores])
        p_a_m2_scores = np.concatenate([replay_p_scores, birth_scores])
        for name, template, boxes, scores in (
            ("base_m2", base_payload, base_boxes, base_m2_scores),
            ("a_m2", base_payload, a_boxes, a_m2_scores),
            ("p_m2", p_payload, p_boxes, replay_p_scores),
            ("p_a_m2", p_payload, p_a_boxes, p_a_m2_scores)):
            path = roots[name] / f"{scene}_boxes.pkl"
            save_payload(template, boxes, scores, path)
            predictions[name][scene] = (boxes, scores)
            hashes[str(path.resolve())] = sha256(path)
        predictions["base"][scene] = (base_boxes, base_scores)
        predictions["p"][scene] = (p_boxes, p_scores)
        predictions["a"][scene] = (a_boxes, a_scores)
        predictions["p_a"][scene] = (p_a_boxes, p_a_scores)
        gts[scene] = valid_boxes(
            np.load(GT_ROOT / scene / "after_filter_boxes.npy"), scene)
        for path in (cache_path, BASE / f"{scene}_boxes.pkl",
                     P / f"{scene}_boxes.pkl", P_M2 / f"{scene}_boxes.pkl",
                     P_M2 / f"{scene}_boxes.pkl.dual_state.json",
                     ONLINE / f"{scene}_boxes.pkl",
                     GT_ROOT / scene / "after_filter_boxes.npy"):
            hashes[str(path.resolve())] = sha256(path)
        print(f"[{ordinal}/{len(scenes)}] {scene} base={len(base_boxes)} "
              f"A={len(birth_boxes)} anchor_err={error:.3g}", flush=True)

    metrics = {
        name: {str(t): class_agnostic_ap(rows, gts, t) for t in THRESHOLDS}
        for name, rows in predictions.items()}
    expected = np.asarray([46.1315, 38.3400, 16.8523])
    replay = np.asarray([metrics["p_m2"][str(t)]["ap"]
                         for t in THRESHOLDS])
    summary = {
        "schema": "boxfusion.ca1m.missing_factorial.v1",
        "completed": True,
        "scene_count": len(scenes),
        "parameters": {"m2_mode": "nativelogit", "exclusive": False,
                       "tau": TAU,
                       "score_floor": SCORE_FLOOR, "topk_per_frame": TOPK},
        "metrics": metrics,
        "anchor_expected_p_m2": expected.tolist(),
        "anchor_replay_p_m2": replay.tolist(),
        "anchor_delta_replay_minus_archived": (replay - expected).tolist(),
        "factorial_policy": ("all four M2 arms use the same freshly captured "
                             "frozen proposal stream; archived P+M2 is retained "
                             "only as a reported numerical anchor"),
        "score_audit": score_audit,
        "input_output_sha256": hashes,
        "gt_use": "final evaluation only",
    }
    write_json(output / "results.json", summary)
    print(json.dumps({name: [metrics[name][str(t)]["ap"]
                            for t in THRESHOLDS]
                      for name in metrics}, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT /
                        "reports/ca1m_factorial8_20260916")
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--capture-only", action="store_true")
    parser.add_argument("--evaluate-only", action="store_true")
    args = parser.parse_args()
    if args.capture_only and args.evaluate_only:
        parser.error("choose at most one of --capture-only/--evaluate-only")
    args.output.mkdir(parents=True, exist_ok=True)
    if not args.evaluate_only:
        capture(args.output, args.device_index, args.limit)
    if not args.capture_only and not args.limit:
        evaluate(args.output)


if __name__ == "__main__":
    main()
