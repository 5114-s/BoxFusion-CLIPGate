#!/usr/bin/env python3
"""Render a clear, internal-camera mesh view for a real three-module example."""

from pathlib import Path
import json
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from plyfile import PlyData


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools/qualitative"))
import build_framework_scene0030_real_assets as source  # noqa: E402


SCENE = "scene0518_00"
FRAME = 75
OUT = ROOT / "figure_assets/carr3d_clear_flowchart_scene0518_00_20261115"
SELECTION = ROOT / "figure_assets/three_module_scene_shortlist_20261115/scene0518_00_selection.json"
GEOMETRY = ROOT / "reports/three_module_scene_search_20261115/scene0518_00/verified_geometry.npz"
MESH = source.common.RAW / SCENE / f"{SCENE}_vh_clean.ply"
BLUE, ORANGE, TEAL = "#2499D0", "#F28C12", "#00A88E"


def font(size, bold=False):
    return ImageFont.truetype(
        "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf", size
    )


def load_mesh():
    ply = PlyData.read(MESH, known_list_len={"face": {"vertex_indices": 3}})
    vertex = ply["vertex"].data
    xyz = np.column_stack([vertex[k] for k in ("x", "y", "z")]).astype(np.float64)
    rgb = np.column_stack([vertex[k] for k in ("red", "green", "blue")]).astype(np.float64) / 255.0
    # Mild, deterministic display correction only; geometry is unchanged.
    rgb = np.clip(rgb ** 0.90 * 1.04, 0, 1)
    faces = np.ascontiguousarray(ply["face"].data["vertex_indices"], dtype=np.int32)
    return xyz, rgb, faces


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    selection = json.loads(SELECTION.read_text())
    data = np.load(GEOMETRY, allow_pickle=True)
    indexes = selection["selected_prediction_indices"]
    boxes = data["boxes_world"][indexes]
    native = dict(source.replay.load_native(source.NATIVE.with_name(SCENE + ".npz")))[FRAME]

    xyz, rgb, faces = load_mesh()
    width, height = 1500, 1125
    intrinsic = np.asarray(native.intrinsic, dtype=float).copy()
    intrinsic[0] *= width / native.width
    intrinsic[1] *= height / native.height
    uv, depth = source.helper.project_world(xyz, native.camera_to_world, intrinsic)
    projected = np.column_stack([uv, depth])
    keep = (depth[faces] > 0.03).all(axis=1)
    renderer = source.common.load_renderer()
    rgba = renderer.rasterize(projected, rgb, faces[keep], width, height)
    base = Image.fromarray(rgba)

    # Keep exactly the verified HSR, PTR and VCR records in that order.
    colors = [BLUE, ORANGE, TEAL]
    draw = ImageDraw.Draw(base)
    projected_boxes = []
    for box, color in zip(boxes, colors):
        box_uv, box_depth = source.helper.project_world(
            box[None], native.camera_to_world, intrinsic
        )
        if np.any(box_depth <= 0.03):
            raise RuntimeError("selected box behind camera")
        box_uv = box_uv[0]
        projected_boxes.append(box_uv)
        for a, b in source.helper.cuboid_edges(box):
            line = [tuple(box_uv[a]), tuple(box_uv[b])]
            draw.line(line, fill="white", width=12)
            draw.line(line, fill=color, width=8)

    transparent = OUT / "online_3d_detections_internal_mesh.png"
    base.save(transparent, dpi=(400, 400))
    white = Image.new("RGB", base.size, "white")
    white.paste(base, mask=base.getchannel("A"))
    white_path = OUT / "online_3d_detections_internal_mesh_white.png"
    white.save(white_path, dpi=(400, 400))

    # Crop empty borders while preserving the complete selected cuboids.
    bbox = base.getbbox()
    pts = np.concatenate(projected_boxes)
    low = np.minimum(np.array(bbox[:2]), pts.min(0)) - 35
    high = np.maximum(np.array(bbox[2:]), pts.max(0)) + 35
    crop_box = tuple(np.r_[np.maximum(low, 0), np.minimum(high, [width, height])].astype(int))
    cropped = white.crop(crop_box)
    cropped_path = OUT / "online_3d_detections_internal_mesh_crop.png"
    cropped.save(cropped_path, dpi=(400, 400))

    # Flowchart-ready card, matching the visual hierarchy of the reference.
    card_w = 1500
    shown = cropped.resize((1410, round(cropped.height * 1410 / cropped.width)), Image.Resampling.LANCZOS)
    card = Image.new("RGB", (card_w, 150 + shown.height + 150), "white")
    d = ImageDraw.Draw(card)
    d.rounded_rectangle((5, 5, card.width - 6, card.height - 6), radius=28,
                        outline="#A8B0B8", width=3)
    d.text((card.width // 2, 45), "Online 3D Detections", anchor="mt",
           font=font(52, True), fill="#172A40")
    card.paste(shown, (45, 150))
    labels = ["Native / HSR", "Proposal / PTR", "Candidate / VCR"]
    y = 150 + shown.height + 52
    for x, label, color in zip((75, 535, 1000), labels, colors):
        d.rounded_rectangle((x, y, x + 62, y + 32), radius=7, fill=color)
        d.text((x + 80, y - 2), label, font=font(31, True), fill="#28384C")
    panel = OUT / "online_3d_detections_reference_style_panel.png"
    card.save(panel, dpi=(400, 400))

    manifest = {
        "scene": SCENE,
        "frame": FRAME,
        "mesh": str(MESH),
        "geometry": str(GEOMETRY),
        "selected_prediction_indices": indexes,
        "modules": ["HSR", "PTR", "VCR"],
        "objects": ["desk", "chair", "cabinet"],
        "prediction_geometry_modified": False,
        "mesh_geometry_modified_or_filled": False,
        "generated_imagery_used": False,
        "view": "recorded internal RGB-D camera pose",
        "terminology": "VCR uses decoded pre-NMS candidates, not predefined anchors",
        "files": [str(transparent), str(white_path), str(cropped_path), str(panel)],
    }
    (OUT / "source_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
    )
    print(panel)


if __name__ == "__main__":
    main()
