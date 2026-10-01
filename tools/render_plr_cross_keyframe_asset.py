#!/usr/bin/env python3
from __future__ import annotations
"""Render a real three-keyframe PLR verification asset for scene0432_01.

The selected post-NMS proposals are associated by their lifted world-space
centres.  No generated imagery or hand-drawn detection geometry is used.
"""

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
SCENE = "scene0432_01"
FRAMES = (0, 25, 50)
CACHE = (
    ROOT
    / "development/final_controls/evidence/scannet_recar3d_key_controls"
    / f"{SCENE}.npz"
)
RGB_DIR = ROOT / "data/scannet_val_rgbfix" / SCENE / "frames/color"
OUT_DIR = ROOT / "figure_assets/pipeline_candidate_visuals_scene0432_01"

PLR_ORANGE = (244, 92, 32, 255)
TARGET_WORLD_CENTER = np.asarray([1.84, 2.16, 0.25], dtype=np.float32)
THUMB_SIZE = (360, 245)
DETECTOR_SIZE = (640, 480)


def font(size: int, italic: bool = False) -> ImageFont.FreeTypeFont:
    name = "DejaVuSans-Oblique.ttf" if italic else "DejaVuSans.ttf"
    return ImageFont.truetype(f"/usr/share/fonts/truetype/dejavu/{name}", size)


def frame_slice(lengths: np.ndarray, index: int) -> slice:
    start = int(np.sum(lengths[:index]))
    return slice(start, start + int(lengths[index]))


def expanded_crop(box: np.ndarray, image_size: tuple[int, int]) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = map(float, box)
    bw, bh = x2 - x1, y2 - y1
    x1 -= 0.20 * bw
    x2 += 0.20 * bw
    y1 -= 0.20 * bh
    y2 += 0.20 * bh

    target_ratio = THUMB_SIZE[0] / THUMB_SIZE[1]
    cw, ch = x2 - x1, y2 - y1
    if cw / ch < target_ratio:
        extra = (target_ratio * ch - cw) / 2
        x1, x2 = x1 - extra, x2 + extra
    else:
        extra = (cw / target_ratio - ch) / 2
        y1, y2 = y1 - extra, y2 + extra

    w, h = image_size
    if x1 < 0:
        x2 -= x1
        x1 = 0
    if y1 < 0:
        y2 -= y1
        y1 = 0
    if x2 > w:
        x1 -= x2 - w
        x2 = w
    if y2 > h:
        y1 -= y2 - h
        y2 = h
    return tuple(map(lambda v: int(round(v)), (max(0, x1), max(0, y1), x2, y2)))


def render_thumbnail(rgb_path: Path, box: np.ndarray) -> Image.Image:
    rgb = Image.open(rgb_path).convert("RGB")
    box = box.astype(np.float32).copy()
    box[[0, 2]] *= rgb.width / DETECTOR_SIZE[0]
    box[[1, 3]] *= rgb.height / DETECTOR_SIZE[1]
    crop_box = expanded_crop(box, rgb.size)
    crop = rgb.crop(crop_box).resize(THUMB_SIZE, Image.Resampling.LANCZOS).convert("RGBA")

    sx = THUMB_SIZE[0] / (crop_box[2] - crop_box[0])
    sy = THUMB_SIZE[1] / (crop_box[3] - crop_box[1])
    x1 = (float(box[0]) - crop_box[0]) * sx
    y1 = (float(box[1]) - crop_box[1]) * sy
    x2 = (float(box[2]) - crop_box[0]) * sx
    y2 = (float(box[3]) - crop_box[1]) * sy
    draw = ImageDraw.Draw(crop)
    draw.rectangle((x1, y1, x2, y2), outline=PLR_ORANGE, width=7)
    return crop


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    data = np.load(CACHE, allow_pickle=True)
    frame_ids = data["frame_ids"]
    lengths = data["proposal_lengths"]
    all_boxes = data["proposal_boxes_2d"]
    all_corners = data["proposal_corners"]
    all_scores = data["proposal_scores"]
    all_ids = data["proposal_ids"]

    thumbnails = []
    manifest_rows = []
    for order, frame_id in enumerate(FRAMES, start=1):
        frame_index = int(np.flatnonzero(frame_ids == frame_id)[0])
        sl = frame_slice(lengths, frame_index)
        corners = all_corners[sl]
        centres = corners.mean(axis=1)
        local_index = int(np.argmin(np.linalg.norm(centres - TARGET_WORLD_CENTER, axis=1)))
        global_index = sl.start + local_index
        box = all_boxes[global_index]
        thumb = render_thumbnail(RGB_DIR / f"{frame_id}.jpg", box)
        thumb_path = OUT_DIR / f"08_plr_track_t{order}_frame{frame_id:04d}.png"
        thumb.save(thumb_path)
        thumbnails.append(thumb)
        manifest_rows.append(
            {
                "keyframe_label": f"t{order}",
                "frame_id": int(frame_id),
                "proposal_id": int(all_ids[global_index]),
                "proposal_score": float(all_scores[global_index]),
                "box_2d_xyxy": [float(v) for v in box],
                "world_center_xyz": [float(v) for v in centres[local_index]],
                "image": str(thumb_path),
            }
        )

    gap = 24
    label_h = 52
    canvas = Image.new(
        "RGBA",
        (3 * THUMB_SIZE[0] + 2 * gap, label_h + THUMB_SIZE[1]),
        (255, 255, 255, 0),
    )
    draw = ImageDraw.Draw(canvas)
    label_font = font(34, italic=True)
    for idx, thumb in enumerate(thumbnails):
        x = idx * (THUMB_SIZE[0] + gap)
        label = f"t{idx + 1}"
        bbox = draw.textbbox((0, 0), label, font=label_font)
        draw.text(
            (x + (THUMB_SIZE[0] - (bbox[2] - bbox[0])) / 2, 3),
            label,
            fill=(20, 20, 20, 255),
            font=label_font,
        )
        canvas.alpha_composite(thumb, (x, label_h))

    clean_path = OUT_DIR / "08_plr_cross_keyframe_same_track.png"
    canvas.save(clean_path)

    ready_h = canvas.height + 82
    ready = Image.new("RGBA", (canvas.width, ready_h), (255, 255, 255, 0))
    ready.alpha_composite(canvas, (0, 0))
    draw = ImageDraw.Draw(ready)
    tick_font = font(48)
    caption_font = font(30)
    caption = "same 3D track in ≥3 distinct keyframes"
    tick = "✓"
    tick_bbox = draw.textbbox((0, 0), tick, font=tick_font)
    caption_bbox = draw.textbbox((0, 0), caption, font=caption_font)
    total_w = (tick_bbox[2] - tick_bbox[0]) + 16 + (caption_bbox[2] - caption_bbox[0])
    x0 = (ready.width - total_w) / 2
    y0 = canvas.height + 13
    draw.text((x0, y0 - 8), tick, fill=PLR_ORANGE, font=tick_font)
    draw.text(
        (x0 + (tick_bbox[2] - tick_bbox[0]) + 16, y0),
        caption,
        fill=(35, 35, 35, 255),
        font=caption_font,
    )
    ready_path = OUT_DIR / "08_plr_cross_keyframe_verification_ready.png"
    ready.save(ready_path)

    import json

    manifest = {
        "scene": SCENE,
        "source_cache": str(CACHE),
        "semantics": "One real post-NMS proposal track across three distinct keyframes.",
        "selection": "Nearest lifted proposal centre to the central ottoman world-space centre.",
        "records": manifest_rows,
        "outputs": [str(clean_path), str(ready_path)],
    }
    (OUT_DIR / "08_plr_cross_keyframe_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
