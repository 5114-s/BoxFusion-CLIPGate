#!/usr/bin/env python3
"""Export measured HSR matches and selected 3D predictions for the framework."""
from __future__ import annotations

import json
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

import build_framework_scene0030_real_assets as source


ROOT = source.ROOT
BASE = ROOT / "figure_assets/carr3d_framework_real_scene0030_01_20261114_bold"
OUT = ROOT / "figure_assets/carr3d_hsr_final_scene0030_20261115"
BLUE, PURPLE, ORANGE, GOLD = source.BLUE, source.PURPLE, source.ORANGE, source.GOLD


def dashed_segments(rect, dash=25, gap=17):
    x1, y1, x2, y2 = rect
    points = [(x1, y1), (x2, y1), (x2, y2), (x1, y2), (x1, y1)]
    segments = []
    for a, b in zip(points[:-1], points[1:]):
        a, b = np.asarray(a), np.asarray(b)
        length = np.linalg.norm(b - a)
        direction = (b - a) / length
        for start in np.arange(0, length, dash + gap):
            end = min(start + dash, length)
            segments.append([tuple(a + direction * start), tuple(a + direction * end)])
    return segments


def save(image, name):
    path = OUT / name
    image.save(path, dpi=(400, 400))
    with Image.open(path) as saved:
        saved.verify()
    return path


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    metadata = json.loads((BASE / "source_manifest.json").read_text())
    for path, digest in metadata["source_hashes"].items():
        assert source.common.sha(path) == digest, f"Changed source: {path}"
    hsr = metadata["hsr"]
    frame, identity = hsr["display_frame"], hsr["native_id"]
    native = dict(source.replay.load_native(source.NATIVE))[frame]
    evidence = dict(source.replay.load_evidence(source.EVIDENCE))[frame]
    row = int(np.flatnonzero(native.ids == identity)[0])
    matches = source.helper.exact_assignment(
        native.corners, evidence.proposal_boxes_2d, native.camera_to_world,
        native.intrinsic, native.width, native.height)
    proposal_row, iou, native_rect = matches[row]
    assert int(evidence.proposal_ids[proposal_row]) == hsr["matched_proposal_id"]
    assert abs(iou - hsr["iou2d"]) < 1e-10
    proposal_rect = evidence.proposal_boxes_2d[proposal_row]
    rgb_path = source.RGB / f"{frame}.jpg"
    image = Image.open(rgb_path).convert("RGB")
    scale = np.array([image.width / native.width, image.height / native.height] * 2)
    native_pixels, proposal_pixels = np.asarray(native_rect) * scale, proposal_rect * scale
    draw = ImageDraw.Draw(image)
    draw.rectangle(tuple(native_pixels), outline="white", width=14)
    draw.rectangle(tuple(native_pixels), outline=BLUE, width=10)
    segments = dashed_segments(proposal_pixels)
    for line in segments:
        draw.line(line, fill="white", width=14)
    for line in segments:
        draw.line(line, fill=PURPLE, width=10)
    full = save(image, f"hsr_match_f{frame}_full.png")
    points = np.stack([native_pixels, proposal_pixels]).reshape(-1, 2)
    crop_bounds = source.helper.crop_bounds(points, image.width, image.height)
    crop = image.crop(crop_bounds)
    match = save(crop, f"hsr_match_f{frame}.png")
    legend = Image.new("RGB", (max(crop.width, 680), crop.height + 54), "white")
    legend.paste(crop, ((legend.width - crop.width) // 2, 0))
    d = ImageDraw.Draw(legend)
    font = ImageFont.truetype("DejaVuSans.ttf", 19)
    y = crop.height + 27
    d.line((16, y, 58, y), fill=BLUE, width=5)
    d.text((68, y - 12), "Native-box projection", fill=source.INK, font=font)
    for x in (356, 376, 396):
        d.line((x, y, x + 11, y), fill=PURPLE, width=5)
    d.text((417, y - 12), "Matched proposal", fill=source.INK, font=font)
    match_legend = save(legend, f"hsr_match_f{frame}_legend.png")

    # Draw real 3D boxes on the original, common-camera visualization mesh.
    # Coordinates and selection are kept exactly as in the audited asset pack.
    geometry_path = BASE / "08_audit/geometry_and_scores.npz"
    geometry = np.load(geometry_path)
    boxes = np.concatenate([geometry["native_display"], geometry["ptr_box"][None],
                            geometry["var_box"][None]])
    for box in boxes:
        assert any(np.array_equal(box, record) for record in geometry["boxes_world"])
    assert len(boxes) == 6 and len(geometry["boxes_world"]) == metadata["readout"]["total"]
    alignment = geometry["axis_alignment"]
    camera = metadata["camera"]
    renderer = source.common.load_renderer()
    project_args = (np.asarray(camera["position"]), np.asarray(camera["look_at"]),
                    camera["focal_pixels"], np.asarray(camera["principal_point_pixels"]))
    crop_offset = np.asarray(metadata["crop_bounds"][:2])
    background_path = BASE / "07_output/scene_mesh.png"
    background = Image.open(background_path).convert("RGBA")
    result = background.copy()
    d = ImageDraw.Draw(result)
    colors = [BLUE] * len(geometry["native_display"]) + [ORANGE, GOLD]
    wireframe_mask = Image.new("1", result.size, 0)
    md = ImageDraw.Draw(wireframe_mask)
    for box, color in zip(boxes, colors):
        transformed = box @ alignment[:3, :3].T + alignment[:3, 3]
        uv = renderer.project(transformed, *project_args)[:, :2] - crop_offset
        line_width = 16 if color == BLUE else 22
        for a, b in source.helper.cuboid_edges(box):
            line = [tuple(uv[a]), tuple(uv[b])]
            d.line(line, fill=color, width=line_width)
            md.line(line, fill=1, width=line_width)
    # Ensure no scene pixel outside predicted wireframes was altered.
    outside = ~np.asarray(wireframe_mask, dtype=bool)
    assert np.array_equal(np.asarray(result)[outside], np.asarray(background)[outside])
    final = save(result, "carr3d_final_selected_transparent.png")
    white = Image.new("RGB", result.size, "white")
    white.paste(result, mask=result.getchannel("A"))
    final_white = save(white, "carr3d_final_selected_white.png")

    # Same HSR native ID is included in the final scene's selected native records.
    assert identity in metadata["readout"]["selected_native_ids"]
    history_path = BASE / "08_audit/hsr_history.csv"
    audit = {
        "scene": source.SCENE, "model": "CaRR-3D", "render_only_no_inference": True,
        "prediction_geometry_modified": False,
        "hsr": dict(hsr, native_projection_xyxy=np.asarray(native_rect).tolist(),
                    matched_proposal_xyxy=proposal_rect.tolist(), crop_bounds=list(crop_bounds),
                    rgb_path=str(rgb_path), rgb_sha256=source.common.sha(rgb_path),
                    native_style="blue solid", matched_proposal_style="purple dashed",
                    color_corrected_display=True),
        "final_readout": dict(metadata["readout"], shown_native=4, shown_ptr=1, shown_var=1),
        "sources": metadata["sources"], "source_hashes": metadata["source_hashes"],
        "geometry_source": str(geometry_path), "geometry_sha256": source.common.sha(geometry_path),
        "mesh_background": str(background_path), "mesh_background_sha256": source.common.sha(background_path),
        "mesh": metadata["mesh"], "camera": camera, "crop_bounds": metadata["crop_bounds"],
        "state_history_csv": str(history_path), "state_history_sha256": source.common.sha(history_path),
        "notes": [
            "HSR matching example is from frame 1400; final selected readout is from frame 1600.",
            "Blue boxes retain native geometry; HSR updates scores/ranking only.",
            "Selected targets/predictions are GT-assisted illustrations, not all 304 outputs.",
            "Mesh and cutaways are visualization-only, not model inputs or online reconstruction.",
            "Only visualization styling changed; no prediction moved, resized, added or fabricated."
        ]
    }
    assets = [match, full, match_legend, final, final_white]
    audit["outputs"] = [{"file": p.name, "sha256": source.common.sha(p),
                         "size_wh": list(Image.open(p).size)} for p in assets]
    manifest = OUT / "source_manifest.json"
    source.save_json(manifest, audit)
    with zipfile.ZipFile(OUT.with_suffix(".zip"), "w", zipfile.ZIP_DEFLATED) as bundle:
        for path in assets + [manifest]:
            bundle.write(path, path.name)
    print(json.dumps({"hsr_match": str(match), "hsr_iou2d": iou,
                      "native_score": hsr["native_score"], "hsr_score": hsr["hsr_score"],
                      "final": str(final), "shown": len(boxes),
                      "total": metadata["readout"]["total"], "verified": True}))


if __name__ == "__main__":
    main()
