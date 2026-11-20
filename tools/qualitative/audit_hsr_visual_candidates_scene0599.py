#!/usr/bin/env python3
"""Create a contact sheet of exact HSR record/proposal matches for visual audit."""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

ROOT = Path(__file__).resolve().parents[2]
SCENE = "scene0599_01"
RGB = ROOT / "data/scannet_val_rgbfix" / SCENE / "frames/color"
NATIVE = ROOT / "development/final_controls/experiments/mvsr_multiview_audit_20261012/native_prefix" / f"{SCENE}.npz"
EVIDENCE = ROOT / "development/final_controls/experiments/mvsr_unified_samepass_20261010/evidence" / f"{SCENE}.npz"
SOURCE = ROOT / "development/mvsr_multiview_audit_20261012_source"
OUT = ROOT / "figure_assets/hsr_visual_candidate_audit_scene0599_20261107.png"

sys.path.insert(0, str(SOURCE))
from boxfusion.native_reliability_reranker import _project_with_visibility  # noqa: E402
from boxfusion.online_candidate_map import _iou2d_matrix  # noqa: E402


def slices(lengths: np.ndarray):
    offsets = np.r_[0, np.cumsum(lengths)]
    return [slice(int(offsets[i]), int(offsets[i + 1])) for i in range(len(lengths))]


def crop_box(boxes: list[np.ndarray], width: int, height: int, ratio: float = 1.55):
    values = np.asarray(boxes)
    lo, hi = values[:, :2].min(0), values[:, 2:].max(0)
    center = (lo + hi) / 2
    span = np.maximum(hi - lo, [90, 90]) * 1.32
    if span[0] / span[1] < ratio:
        span[0] = span[1] * ratio
    else:
        span[1] = span[0] / ratio
    low = np.maximum(center - span / 2, 0)
    high = np.minimum(center + span / 2, [width, height])
    return tuple(np.r_[low, high].round().astype(int))


def main():
    n = np.load(NATIVE, allow_pickle=False)
    e = np.load(EVIDENCE, allow_pickle=False)
    ns, ps = slices(n["native_lengths"]), slices(e["proposal_lengths"])
    candidates, counts = [], Counter()
    for fi, frame in enumerate(n["frame_ids"]):
        rows, boxes = [], []
        native_corners = n["native_corners"][ns[fi]]
        native_ids = n["native_ids"][ns[fi]]
        proposal_boxes = e["proposal_boxes_2d"][ps[fi]]
        for row, corners in enumerate(native_corners):
            value = _project_with_visibility(corners, n["camera_to_world"][fi], n["intrinsic"][fi], int(n["width"][fi]), int(n["height"][fi]))
            if value is not None:
                rows.append(row); boxes.append(value[0])
        if not boxes or len(proposal_boxes) == 0:
            continue
        overlap = _iou2d_matrix(np.asarray(boxes), proposal_boxes)
        pairs = [(float(overlap[r, c]), r, c) for r, c in zip(*np.where(overlap >= .1))]
        pairs.sort(key=lambda x: (-x[0], x[1], x[2]))
        used_r, used_p = set(), set()
        for iou, rr, pr in pairs:
            if rr in used_r or pr in used_p:
                continue
            used_r.add(rr); used_p.add(pr)
            nr = rows[rr]; identity = int(native_ids[nr])
            counts[identity] += 1
            native_box, proposal_box = np.asarray(boxes[rr]), np.asarray(proposal_boxes[pr])
            union = np.r_[np.minimum(native_box[:2], proposal_box[:2]), np.maximum(native_box[2:], proposal_box[2:])]
            area = max(0, union[2]-union[0]) * max(0, union[3]-union[1]) / (640*480)
            candidates.append((iou, area, int(frame), identity, int(pr), native_box, proposal_box))

    eligible = [x for x in candidates if counts[x[3]] >= 3 and x[0] >= .68 and .018 <= x[1] <= .45]
    eligible.sort(key=lambda x: (-(x[0] - .14*abs(x[1]-.15)), -counts[x[3]], x[2]))
    selected = eligible[:24]
    tw, th, lh, cols = 430, 285, 42, 4
    rows = (len(selected)+cols-1)//cols
    sheet = Image.new("RGB", (tw*cols, (th+lh)*rows), "white")
    draw = ImageDraw.Draw(sheet)
    try: font = ImageFont.truetype("DejaVuSans.ttf", 20)
    except OSError: font = ImageFont.load_default()
    for idx, (iou, area, frame, identity, pr, nb, pb) in enumerate(selected):
        image = Image.open(RGB / f"{frame}.jpg").convert("RGB")
        sx, sy = image.width/640, image.height/480
        boxes = []
        for box in [nb, pb]:
            value = box.copy(); value[[0,2]] *= sx; value[[1,3]] *= sy; boxes.append(value)
        idraw = ImageDraw.Draw(image)
        for box, color in zip(boxes, ["#236FB6", "#F06418"]):
            idraw.rectangle(tuple(box), outline="white", width=8)
            idraw.rectangle(tuple(box), outline=color, width=4)
        left, top, right, bottom = crop_box(boxes, image.width, image.height)
        tile = ImageOps.contain(image.crop((left, top, right, bottom)), (tw-16, th-16), Image.Resampling.LANCZOS)
        col, row = idx % cols, idx // cols
        x = col*tw + (tw-tile.width)//2; y = row*(th+lh)+(th-tile.height)//2
        sheet.paste(tile,(x,y))
        label=f"f{frame} · id{identity} · n={counts[identity]} · IoU {iou:.3f}"
        draw.text((col*tw+10,row*(th+lh)+th+6),label,fill="#142033",font=font)
    sheet.save(OUT)
    print(OUT)


if __name__ == "__main__":
    main()
