#!/usr/bin/env python3
"""Render a clear camera-view visualization of three real CaRR-3D records.

The RGB image is an original ScanNet frame.  The three cuboids are unchanged
cached 3D records projected with the recorded pose and intrinsics.  This is a
visualization view of the terminal scene-level records, not a synchronous
per-frame output claim.
"""

from pathlib import Path
import json
import pickle
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools/qualitative"))
import build_framework_scene0030_real_assets as source  # noqa: E402


PACK = ROOT / "figure_assets/carr3d_flowchart_assets_scene0030_01_vcr_20261115"
BASE = ROOT / "figure_assets/carr3d_framework_real_scene0030_01_20261114_bold"
FALLBACK = ROOT / "figure_assets/carr3d_framework_real_scene0030_01_20261114"
FRAME = 1250
BLUE = "#2499D0"
ORANGE = "#F28C12"
TEAL = "#00A88E"


def font(size: int, bold: bool = False):
    return ImageFont.truetype(
        "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf", size
    )


def draw_projected_box(image, box, color, native, width=8, halo=12):
    uv, depth = source.helper.project_world(
        np.asarray(box)[None], native.camera_to_world, native.intrinsic
    )
    if np.any(depth <= 0.001):
        raise RuntimeError("Selected box falls behind the visualization camera")
    scale = np.array([image.width / native.width, image.height / native.height])
    uv = uv[0] * scale
    draw = ImageDraw.Draw(image)
    for a, b in source.helper.cuboid_edges(box):
        line = [tuple(uv[a]), tuple(uv[b])]
        draw.line(line, fill="white", width=halo)
        draw.line(line, fill=color, width=width)
    return uv


def main():
    out = PACK / "08_clear_camera_view"
    out.mkdir(parents=True, exist_ok=True)
    payload_path = BASE / "08_audit/replay_payload.pkl"
    if not payload_path.exists():
        payload_path = FALLBACK / "08_audit/replay_payload.pkl"
    with payload_path.open("rb") as handle:
        data = pickle.load(handle)

    native_rows = dict(data["native_rows"])
    native = native_rows[FRAME]
    snapshot = data["snapshot"]
    ids = list(map(int, data["native_display_ids"]))
    hsr_id = 354
    hsr_index = ids.index(hsr_id)
    hsr_box = snapshot.boxes[data["native_display_indices"][hsr_index]]
    ptr_box = np.asarray(data["ptr_row"]["box"])
    vcr_box = np.asarray(data["var_row"]["box"])
    boxes = [hsr_box, ptr_box, vcr_box]
    colors = [BLUE, ORANGE, TEAL]

    original = Image.open(source.RGB / f"{FRAME}.jpg").convert("RGB")
    overlay = original.copy()
    projected = [
        draw_projected_box(overlay, box, color, native)
        for box, color in zip(boxes, colors)
    ]
    full_path = out / "online_3d_detections_camera_view_full.png"
    overlay.save(full_path, dpi=(400, 400))

    points = np.concatenate(projected)
    low = np.maximum(points.min(0) - np.array([90, 80]), 0)
    high = np.minimum(points.max(0) + np.array([90, 80]), overlay.size)
    crop_box = tuple(np.r_[low, high].round().astype(int))
    crop = overlay.crop(crop_box)
    crop_path = out / "online_3d_detections_camera_view_crop.png"
    crop.save(crop_path, dpi=(400, 400))

    # Reference-style card for direct insertion in the flowchart.
    card_w = 1500
    scale = (card_w - 90) / crop.width
    display = crop.resize(
        (card_w - 90, round(crop.height * scale)), Image.Resampling.LANCZOS
    )
    title_h, legend_h = 150, 150
    card = Image.new("RGB", (card_w, title_h + display.height + legend_h), "white")
    draw = ImageDraw.Draw(card)
    draw.rounded_rectangle(
        (5, 5, card.width - 6, card.height - 6), radius=28,
        outline="#A8B0B8", width=3,
    )
    draw.text(
        (card.width // 2, 45), "Online 3D Detections",
        anchor="mt", font=font(52, True), fill="#172A40",
    )
    card.paste(display, (45, title_h))
    labels = ["Native / HSR", "Proposal / PTR", "Candidate / VCR"]
    positions = [75, 535, 1000]
    y = title_h + display.height + 53
    for x, label, color in zip(positions, labels, colors):
        draw.rounded_rectangle((x, y, x + 62, y + 32), radius=7,
                               fill=color, outline=color)
        draw.text((x + 80, y - 2), label, font=font(31, True), fill="#28384C")
    panel_path = PACK / "00_recommended_clear_output_panel.png"
    card.save(panel_path, dpi=(400, 400))

    manifest = {
        "scene": "scene0030_01",
        "visualization_frame": FRAME,
        "rgb_source": str(source.RGB / f"{FRAME}.jpg"),
        "records": [
            {"module": "HSR", "native_id": hsr_id,
             "color": BLUE, "record_type": "native"},
            {"module": "PTR", "track_id": int(data["ptr_row"]["track_id"]),
             "color": ORANGE, "record_type": "post-NMS proposal recovery"},
            {"module": "VCR", "voxel_key": list(map(int, data["var_row"]["voxel_key"])),
             "color": TEAL, "record_type": "pre-NMS candidate recovery"},
        ],
        "prediction_geometry_modified": False,
        "image_generated_or_inpainted": False,
        "protocol_note": (
            "Terminal scene-level records projected into recorded frame 1250 "
            "for visualization; not a claim that all three records were emitted "
            "synchronously at frame 1250."
        ),
        "crop_xyxy": [int(value) for value in crop_box],
        "files": [str(full_path), str(crop_path), str(panel_path)],
    }
    (out / "source_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
    )
    print(panel_path)


if __name__ == "__main__":
    main()
