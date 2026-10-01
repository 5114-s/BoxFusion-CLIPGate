#!/usr/bin/env python3
"""Render real branch-colored 2D inputs and Boxer-lifted 3D candidates.

Proposal and anchor records come from the same frozen evidence cache used by
the final ReCaR-3D controls.  The images differ only in candidate source and
branch color; Boxer itself is unchanged.
"""
from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[1]
SCENE = "scene0432_01"
FRAME_ID = 0
CACHE = ROOT / (
    "development/final_controls/evidence/scannet_recar3d_key_controls/"
    f"{SCENE}.npz"
)
RAW_ANCHORS = ROOT / (
    "reports/scannet_seedless_full100_20260911/raw/"
    f"{SCENE}/raw_{FRAME_ID:06d}.npz"
)
FRAME_ROOT = ROOT / "upstream_clean/scannet_readme_frames" / SCENE / "frames"
RGB = FRAME_ROOT / "color" / f"{FRAME_ID}.jpg"
POSE = FRAME_ROOT / "pose" / f"{FRAME_ID}.txt"
INTRINSIC = FRAME_ROOT / "intrinsic" / "intrinsic_color.txt"
OUTPUT = ROOT / "figure_assets/pipeline_candidate_visuals_scene0432_01"

PROPOSAL_ORANGE = (255, 101, 24)
ANCHOR_AMBER = (242, 169, 0)
WEDETECT_GRAY = (112, 123, 134)
DISPLAY_SIZE = (720, 538)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def frame_slice(lengths: np.ndarray, frame_index: int) -> slice:
    start = int(np.sum(lengths[:frame_index]))
    return slice(start, start + int(lengths[frame_index]))


def iou2d(a: np.ndarray, b: np.ndarray) -> float:
    low = np.maximum(a[:2], b[:2])
    high = np.minimum(a[2:], b[2:])
    wh = np.maximum(0.0, high - low)
    inter = float(np.prod(wh))
    area_a = float(np.prod(np.maximum(0.0, a[2:] - a[:2])))
    area_b = float(np.prod(np.maximum(0.0, b[2:] - b[:2])))
    return inter / max(area_a + area_b - inter, 1e-8)


def diverse_top_indices(boxes: np.ndarray, scores: np.ndarray, count: int,
                        max_iou: float) -> list[int]:
    selected: list[int] = []
    for index in np.argsort(scores)[::-1]:
        if all(iou2d(boxes[index], boxes[other]) <= max_iou for other in selected):
            selected.append(int(index))
        if len(selected) == count:
            break
    return selected


def cuboid_edges(box: np.ndarray) -> np.ndarray:
    points = np.asarray(box, np.float64).reshape(8, 3)
    centered = points - points.mean(axis=0)
    opposite_cost = np.linalg.norm(centered + centered[0], axis=1)
    opposite_cost[0] = np.inf
    opposite = int(np.argmin(opposite_cost))
    others = [idx for idx in range(8) if idx not in (0, opposite)]
    diagonal = points[opposite] - points[0]
    best = None
    best_cost = np.inf
    for neighbours in itertools.combinations(others, 3):
        basis = np.stack([points[idx] - points[0] for idx in neighbours])
        if abs(float(np.linalg.det(basis))) < 1e-12:
            continue
        generated = np.asarray([
            points[0] + np.asarray(code, np.float64) @ basis
            for code in itertools.product((0, 1), repeat=3)
        ])
        nearest = np.argmin(
            np.linalg.norm(generated[:, None] - points[None], axis=2), axis=1
        )
        if len(set(nearest.tolist())) != 8:
            continue
        cost = float(np.max(np.linalg.norm(generated - points[nearest], axis=1)))
        cost += float(np.linalg.norm(basis.sum(axis=0) - diagonal))
        if cost < best_cost:
            best_cost = cost
            best = dict(zip(itertools.product((0, 1), repeat=3), nearest.tolist()))
    if best is None or best_cost > 1e-4:
        raise ValueError(f"cannot recover cuboid topology; error={best_cost}")
    return np.asarray(sorted({
        tuple(sorted((best[a], best[b])))
        for a in best for b in best
        if sum(x != y for x, y in zip(a, b)) == 1
    }), dtype=np.int64)


def project(corners_world: np.ndarray, c2w: np.ndarray,
            intrinsic: np.ndarray) -> np.ndarray:
    w2c = np.linalg.inv(c2w)
    camera = corners_world @ w2c[:3, :3].T + w2c[:3, 3]
    if np.any(camera[:, 2] < 0.1):
        raise ValueError("cuboid crosses the camera plane")
    uvw = camera @ intrinsic[:3, :3].T
    return uvw[:, :2] / uvw[:, 2:3]


def draw_2d(rgb: Image.Image, boxes: np.ndarray, color: tuple[int, int, int],
            path: Path, width: int = 8) -> None:
    canvas = rgb.copy()
    draw = ImageDraw.Draw(canvas)
    for box in boxes:
        draw.rectangle(tuple(float(v) for v in box), outline=color, width=width)
    canvas.resize(DISPLAY_SIZE, Image.Resampling.LANCZOS).save(path)


def draw_3d(rgb: Image.Image, corners: np.ndarray, color: tuple[int, int, int],
            c2w: np.ndarray, intrinsic: np.ndarray, path: Path) -> None:
    canvas = rgb.copy()
    draw = ImageDraw.Draw(canvas)
    for box in corners:
        uv = project(box, c2w, intrinsic)
        for first, second in cuboid_edges(box):
            p = tuple(float(v) for v in uv[first])
            q = tuple(float(v) for v in uv[second])
            draw.line((p, q), fill=(255, 255, 255), width=12)
            draw.line((p, q), fill=color, width=7)
    canvas.resize(DISPLAY_SIZE, Image.Resampling.LANCZOS).save(path)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    cache = np.load(CACHE, allow_pickle=False)
    raw = np.load(RAW_ANCHORS, allow_pickle=False)
    frame_index = int(np.flatnonzero(cache["frame_ids"] == FRAME_ID)[0])
    proposal_slice = frame_slice(cache["proposal_lengths"], frame_index)
    anchor_slice = frame_slice(cache["anchor_lengths"], frame_index)

    # The exported ScanNet JPEGs in this asset tree retain OpenCV BGR order.
    # Swap red and blue once so the paper visualization matches the true scene colors.
    rgb_bgr = np.asarray(Image.open(RGB).convert("RGB"))
    rgb = Image.fromarray(rgb_bgr[:, :, ::-1].copy(), mode="RGB")
    proposal_boxes = cache["proposal_boxes_2d"][proposal_slice].astype(np.float64)
    # Frozen evidence records use the detector's 640x480 image coordinates.
    proposal_boxes[:, [0, 2]] *= rgb.width / 640.0
    proposal_boxes[:, [1, 3]] *= rgb.height / 480.0
    proposal_scores = cache["proposal_scores"][proposal_slice].astype(np.float64)
    proposal_corners = cache["proposal_corners"][proposal_slice].astype(np.float64)
    proposal_selected = diverse_top_indices(proposal_boxes, proposal_scores, 3, 0.25)

    anchor_ids = cache["anchor_ids"][anchor_slice].astype(np.int64)
    raw_boxes = raw["boxes"].astype(np.float64)
    raw_scores = raw["scores"].astype(np.float64)
    anchor_boxes = raw_boxes[anchor_ids].astype(np.float64)
    anchor_scores = cache["anchor_scores"][anchor_slice].astype(np.float64)
    anchor_corners = cache["anchor_corners"][anchor_slice].astype(np.float64)
    anchor_wh = np.maximum(0.0, anchor_boxes[:, 2:] - anchor_boxes[:, :2])
    anchor_area_ratio = np.prod(anchor_wh, axis=1) / float(rgb.width * rgb.height)
    eligible = np.flatnonzero(
        (anchor_area_ratio >= 0.004) & (anchor_area_ratio <= 0.22)
    )
    local_selected = diverse_top_indices(
        anchor_boxes[eligible], anchor_scores[eligible], 4, 0.18
    )
    anchor_selected = [int(eligible[index]) for index in local_selected]

    # Representative high-score records from the frozen 8,400 pre-NMS anchors.
    raw_wh = np.maximum(0.0, raw_boxes[:, 2:] - raw_boxes[:, :2])
    raw_area_ratio = np.prod(raw_wh, axis=1) / float(rgb.width * rgb.height)
    raw_eligible = np.flatnonzero(
        (raw_area_ratio >= 0.003) & (raw_area_ratio <= 0.18)
    )
    raw_local_selected = diverse_top_indices(
        raw_boxes[raw_eligible], raw_scores[raw_eligible], 8, 0.58
    )
    raw_selected = [int(raw_eligible[index]) for index in raw_local_selected]

    c2w = np.loadtxt(POSE).astype(np.float64)
    intrinsic = np.loadtxt(INTRINSIC).astype(np.float64)
    outputs = {
        "proposal_2d": OUTPUT / "09a_post_nms_proposals_orange.png",
        "proposal_3d": OUTPUT / "09b_boxer_lifted_proposals_orange.png",
        "anchor_2d": OUTPUT / "09c_topm_anchors_amber.png",
        "anchor_3d": OUTPUT / "09d_boxer_lifted_anchors_amber.png",
        "wedetect_pre_nms_2d": OUTPUT / "09e_wedetect_pre_nms_candidates_gray.png",
    }
    draw_2d(rgb, proposal_boxes[proposal_selected], PROPOSAL_ORANGE,
            outputs["proposal_2d"])
    draw_3d(rgb, proposal_corners[proposal_selected], PROPOSAL_ORANGE,
            c2w, intrinsic, outputs["proposal_3d"])
    draw_2d(rgb, anchor_boxes[anchor_selected], ANCHOR_AMBER,
            outputs["anchor_2d"])
    draw_3d(rgb, anchor_corners[anchor_selected], ANCHOR_AMBER,
            c2w, intrinsic, outputs["anchor_3d"])
    draw_2d(rgb, raw_boxes[raw_selected], WEDETECT_GRAY,
            outputs["wedetect_pre_nms_2d"], width=4)

    manifest = {
        "schema": "boxfusion.figure.boxer_lifting_lanes.v1",
        "scene_id": SCENE,
        "frame_id": FRAME_ID,
        "same_fixed_boxer_parameters": True,
        "source_color_decode": "BGR-to-RGB channel swap",
        "proposal_color_rgb": PROPOSAL_ORANGE,
        "anchor_color_rgb": ANCHOR_AMBER,
        "wedetect_color_rgb": WEDETECT_GRAY,
        "proposal_indices": proposal_selected,
        "proposal_ids": cache["proposal_ids"][proposal_slice][proposal_selected].tolist(),
        "proposal_scores": proposal_scores[proposal_selected].tolist(),
        "anchor_indices": anchor_selected,
        "anchor_ids": anchor_ids[anchor_selected].tolist(),
        "anchor_scores": anchor_scores[anchor_selected].tolist(),
        "wedetect_pre_nms_indices": raw_selected,
        "wedetect_pre_nms_scores": raw_scores[raw_selected].tolist(),
        "cache": str(CACHE),
        "cache_sha256": sha256(CACHE),
        "raw_anchor_boxes": str(RAW_ANCHORS),
        "raw_anchor_boxes_sha256": sha256(RAW_ANCHORS),
        "rgb": str(RGB),
        "rgb_sha256": sha256(RGB),
        "outputs": {key: {"path": str(path), "sha256": sha256(path)}
                    for key, path in outputs.items()},
    }
    manifest_path = OUTPUT / "09_boxer_lifting_lanes_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    for path in outputs.values():
        print(path)
    print(manifest_path)


if __name__ == "__main__":
    main()
