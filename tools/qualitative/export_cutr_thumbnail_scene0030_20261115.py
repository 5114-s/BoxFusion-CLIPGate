#!/usr/bin/env python3
"""Render a real logged CuTR candidate; no model inference or box adjustment."""
from __future__ import annotations

import hashlib
import json
import math
import zipfile
from pathlib import Path

import yaml
from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[2]
SCENE, FRAME = "scene0030_01", 1600
OUT = ROOT / "figure_assets/carr3d_cutr_thumbnail_scene0030_20261115"
RECEIPT = ROOT / (
    "reports/official_github_boxfusion_vs_final_recar_public_cli_coldcache_nowarmup100_20261009"
    "/receipts/recar"
) / SCENE
LOG = RECEIPT / "native_boxer_diagnostics" / f"{SCENE}_boxer_lifting.jsonl"
CONFIG = RECEIPT / "effective_config.yaml"
RGB = ROOT / "data/scannet_val_rgbfix" / SCENE / "frames/color" / f"{FRAME}.jpg"


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    config = yaml.safe_load(CONFIG.read_text())
    rows = [json.loads(line) for line in LOG.read_text().splitlines() if line.strip()]
    records = [row for row in rows if row["frame_id"] == FRAME]
    assert len(records) == 1, "The selected frame must identify one diagnostic record."
    record = records[0]
    boxes = record["input_boxes_xyxy"]
    assert len(boxes) == record["count"] == len(record["detector_scores"])
    rgb = Image.open(RGB).convert("RGB")
    width, height = config["cam"]["W"], config["cam"]["H"]
    sx, sy = rgb.width / width, rgb.height / height
    display_boxes = [[x1 * sx, y1 * sy, x2 * sx, y2 * sy] for x1, y1, x2, y2 in boxes]
    for x1, y1, x2, y2 in display_boxes:
        assert 0 <= x1 < x2 <= rgb.width and 0 <= y1 < y2 <= rgb.height

    # Draw every candidate in this frame; preserve their measured coordinates.
    marked = rgb.copy()
    draw = ImageDraw.Draw(marked)
    for box in display_boxes:
        draw.rectangle(box, outline="white", width=14)
        draw.rectangle(box, outline="#2779B5", width=10)

    # Optional context crop changes the canvas only, not box coordinates/geometry.
    x1 = min(box[0] for box in display_boxes)
    y1 = min(box[1] for box in display_boxes)
    x2 = max(box[2] for box in display_boxes)
    y2 = max(box[3] for box in display_boxes)
    margin = 0.12 * max(x2 - x1, y2 - y1)
    crop = [max(0, math.floor(x1 - margin)), max(0, math.floor(y1 - margin)),
            min(rgb.width, math.ceil(x2 + margin)), min(rgb.height, math.ceil(y2 + margin))]
    OUT.mkdir(parents=True, exist_ok=True)
    full = OUT / f"cutr_proposals_f{FRAME}.png"
    close = OUT / f"cutr_proposals_f{FRAME}_closeup.png"
    clean = OUT / f"rgb_f{FRAME}.png"
    marked.save(full, dpi=(400, 400))
    marked.crop(crop).save(close, dpi=(400, 400))
    rgb.save(clean, dpi=(400, 400))

    # Verify export did not modify the scene beyond the box-stroke raster.
    exported = Image.open(full).convert("RGB")
    mask = Image.new("L", rgb.size, 0)
    mask_draw = ImageDraw.Draw(mask)
    for box in display_boxes:
        mask_draw.rectangle(box, outline=255, width=14)
    unchanged = all(a == b for a, b, m in zip(rgb.getdata(), exported.getdata(), mask.getdata()) if not m)
    assert unchanged, "Unexpected pixel modification outside measured box strokes."
    for path in (full, close, clean):
        with Image.open(path) as image:
            image.verify()

    manifest = {
        "scene_id": SCENE, "frame_id": FRAME,
        "purpose": "Real CuTR 2D candidate illustration for the native mapping path",
        "candidate_selection": "All logged candidates in this visually selected frame; no GT used",
        "diagnostic_path": str(LOG), "diagnostic_sha256": sha256(LOG),
        "config_path": str(CONFIG), "config_sha256": sha256(CONFIG),
        "logged_input_datadir": config["data"]["datadir"],
        "display_rgb_path": str(RGB), "display_rgb_sha256": sha256(RGB),
        "display_rgb_note": "Color-corrected display image; not the exact inference input pixels",
        "prediction_coordinate_size_wh": [width, height],
        "display_size_wh": list(rgb.size), "coordinate_scale_xy": [sx, sy],
        "input_boxes_xyxy": boxes, "display_boxes_xyxy": display_boxes,
        "detector_scores": record["detector_scores"],
        "displayed_count": len(boxes), "logged_count": record["count"],
        "outline": {"color": "#2779B5", "width_px": 10, "white_halo_width_px": 14},
        "closeup_crop_xyxy": crop, "pixels_outside_strokes_unchanged": unchanged,
        "notes": [
            "2D detector candidates, not GT boxes or 3D wireframe projections.",
            "No manual change of predicted box position or extent; no inference rerun.",
            "Frame 1600 replaces the old frame-1100 shelf thumbnail only; old assets are preserved.",
            "The framework asset pack illustrates different times/runs; this is not a same-step trace."
        ],
        "outputs": [{"file": p.name, "sha256": sha256(p), "size_wh": list(Image.open(p).size)}
                    for p in (full, close, clean)],
    }
    metadata = OUT / "source_manifest.json"
    metadata.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    archive = OUT.with_suffix(".zip")
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        for path in (full, close, clean, metadata):
            bundle.write(path, arcname=path.name)
    print(json.dumps({"image": str(full), "closeup": str(close), "archive": str(archive),
                      "count": len(boxes), "verified": unchanged}, ensure_ascii=False))


if __name__ == "__main__":
    main()
