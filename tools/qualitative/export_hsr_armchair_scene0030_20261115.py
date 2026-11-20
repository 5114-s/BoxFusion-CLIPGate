#!/usr/bin/env python3
"""Align the HSR illustration with the real frame-1600 CuTR armchair image."""
from __future__ import annotations

import csv
import json
import zipfile

import numpy as np
from PIL import Image, ImageDraw, ImageFont

import export_hsr_and_output_scene0030_20261115 as previous


source = previous.source
ROOT = source.ROOT
OUT = ROOT / "figure_assets/carr3d_hsr_armchair_scene0030_20261115"
CUTR = ROOT / "figure_assets/carr3d_cutr_thumbnail_scene0030_20261115"
FRAME, IDENTITY = 1600, 0


def save(image, name):
    path = OUT / name
    image.save(path, dpi=(400, 400))
    with Image.open(path) as saved:
        saved.verify()
    return path


def history_for_identity():
    history = []
    with np.load(source.READOUT) as z:
        off = np.r_[0, np.cumsum(z["lengths"])]
        for i, frame in enumerate(z["frame_ids"]):
            sl = slice(off[i], off[i + 1])
            rows = np.flatnonzero(z["ids"][sl] == IDENTITY)
            if not len(rows):
                continue
            assert len(rows) == 1
            row = int(rows[0])
            j = off[i] + row
            native_scores = z["scores_native"][sl]
            hsr_scores = z["scores_historical_max"][sl]
            history.append({
                "frame": int(frame), "matched": bool(z["matched"][j]),
                "iou2d": float(z["iou2d"][j]),
                "current_support": float(z["current_support"][j]),
                "historical_max_support": float(z["max_support"][j]),
                "native_score": float(native_scores[row]), "hsr_score": float(hsr_scores[row]),
                "native_rank": int(np.flatnonzero(np.argsort(-native_scores, kind="stable") == row)[0]) + 1,
                "hsr_rank": int(np.flatnonzero(np.argsort(-hsr_scores, kind="stable") == row)[0]) + 1,
                "native_readout_count": len(native_scores),
            })
    return history


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    prior = json.loads((previous.BASE / "source_manifest.json").read_text())
    for path, digest in prior["source_hashes"].items():
        assert source.common.sha(path) == digest, f"Source changed: {path}"
    cutr_meta = json.loads((CUTR / "source_manifest.json").read_text())
    assert cutr_meta["scene_id"] == source.SCENE and cutr_meta["frame_id"] == FRAME
    rgb_path = source.RGB / f"{FRAME}.jpg"
    assert source.common.sha(rgb_path) == cutr_meta["display_rgb_sha256"]

    native = dict(source.replay.load_native(source.NATIVE))[FRAME]
    evidence = dict(source.replay.load_evidence(source.EVIDENCE))[FRAME]
    row = int(np.flatnonzero(native.ids == IDENTITY)[0])
    assignment = source.helper.exact_assignment(
        native.corners, evidence.proposal_boxes_2d, native.camera_to_world,
        native.intrinsic, native.width, native.height)
    proposal_row, iou2d, native_rect = assignment[row]
    proposal_rect = evidence.proposal_boxes_2d[proposal_row]
    history = history_for_identity()
    event = next(item for item in history if item["frame"] == FRAME)
    assert event["matched"] and abs(event["iou2d"] - iou2d) < 1e-10
    assert abs(event["native_score"] - float(native.scores[row])) < 1e-10
    cutr_overlap = float(source.helper._iou2d_matrix(
        np.asarray([native_rect]), np.asarray(cutr_meta["input_boxes_xyxy"]))[0, 0])
    assert cutr_overlap > 0.90, "The selected native record must project onto the CuTR armchair."

    # Confirm this is the same native geometry already displayed in the final output.
    selected_ids = prior["readout"]["selected_native_ids"]
    selected_row = selected_ids.index(IDENTITY)
    geometry_path = previous.BASE / "08_audit/geometry_and_scores.npz"
    with np.load(geometry_path) as geometry:
        assert np.array_equal(geometry["native_display"][selected_row], native.corners[row])

    rgb = Image.open(rgb_path).convert("RGB")
    scale = np.array([rgb.width / native.width, rgb.height / native.height] * 2)
    native_pixels = np.asarray(native_rect) * scale
    proposal_pixels = np.asarray(proposal_rect) * scale
    marked = rgb.copy()
    draw = ImageDraw.Draw(marked)
    mask = Image.new("1", rgb.size, 0)
    md = ImageDraw.Draw(mask)
    draw.rectangle(tuple(native_pixels), outline="white", width=14)
    draw.rectangle(tuple(native_pixels), outline=source.BLUE, width=10)
    md.rectangle(tuple(native_pixels), outline=1, width=14)
    segments = previous.dashed_segments(proposal_pixels, dash=32, gap=20)
    for segment in segments:
        draw.line(segment, fill="white", width=14)
        md.line(segment, fill=1, width=14)
    for segment in segments:
        draw.line(segment, fill=source.PURPLE, width=10)
    outside = ~np.asarray(mask, dtype=bool)
    assert np.array_equal(np.asarray(marked)[outside], np.asarray(rgb)[outside])
    full = save(marked, "hsr_armchair_f1600.png")

    # Use exactly the CuTR crop to keep scale and framing consistent across modules.
    crop_bounds = cutr_meta["closeup_crop_xyxy"]
    crop = save(marked.crop(crop_bounds), "hsr_armchair_f1600_closeup.png")
    legend_image = Image.new("RGB", (rgb.width, rgb.height + 78), "white")
    legend_image.paste(marked, (0, 0))
    d = ImageDraw.Draw(legend_image)
    font = ImageFont.truetype("DejaVuSans.ttf", 29)
    y = rgb.height + 39
    d.line((45, y, 115, y), fill=source.BLUE, width=7)
    d.text((132, y - 19), "Native-box projection", fill=source.INK, font=font)
    for x in (710, 736, 762):
        d.line((x, y, x + 15, y), fill=source.PURPLE, width=7)
    d.text((800, y - 19), "Matched proposal", fill=source.INK, font=font)
    legend = save(legend_image, "hsr_armchair_f1600_legend.png")

    # Plain side-by-side preview; no implied cross-run identity inheritance.
    cu = Image.open(CUTR / "cutr_proposals_f1600.png").convert("RGB")
    assert cu.size == marked.size
    sheet = Image.new("RGB", (rgb.width * 2 + 28, rgb.height + 90), "white")
    sheet.paste(cu, (0, 90))
    sheet.paste(marked, (rgb.width + 28, 90))
    d = ImageDraw.Draw(sheet)
    label_font = ImageFont.truetype("DejaVuSans-Bold.ttf", 38)
    d.text((35, 24), "CuTR candidate | frame 1600", fill=source.INK, font=label_font)
    d.text((rgb.width + 63, 24), "HSR matching | frame 1600", fill=source.INK, font=label_font)
    comparison = save(sheet, "cutr_hsr_same_target_comparison.png")

    history_path = OUT / "hsr_native_id0_history.csv"
    with history_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    audit = {
        "scene": source.SCENE, "frame": FRAME, "native_id": IDENTITY,
        "matched_proposal_id": int(evidence.proposal_ids[proposal_row]),
        "proposal_score": float(evidence.proposal_scores[proposal_row]),
        "native_projection_xyxy": np.asarray(native_rect).tolist(),
        "matched_proposal_xyxy": proposal_rect.tolist(),
        "native_corners_world": native.corners[row].tolist(),
        "current_event": event,
        "ranking_scope": "size-filtered native records within the same scene/frame; stable score sorting",
        "overlap_with_cutr_display_candidate": cutr_overlap,
        "same_native_geometry_in_final_output": True,
        "source_paths": prior["sources"], "source_hashes": prior["source_hashes"],
        "cutr_manifest_path": str(CUTR / "source_manifest.json"),
        "rgb_path": str(rgb_path), "rgb_sha256": source.common.sha(rgb_path),
        "prediction_coordinate_size_wh": [native.width, native.height],
        "display_size_wh": list(rgb.size), "display_scale_xy": scale[:2].tolist(),
        "shared_cutr_crop_xyxy": crop_bounds,
        "styles": {"native": "blue solid 10 px, white halo 14 px",
                   "proposal": "purple dashed 10 px, dash 32 px, gap 20 px, white halo 14 px"},
        "outside_stroke_pixels_unchanged": True,
        "notes": [
            "Same observed armchair and same RGB frame as the CuTR illustration, not the former wooden chair.",
            "Native projected geometry and raw CuTR detector geometry are different types of records.",
            "CuTR and HSR sources are archived in separate runs; this is not claimed to be a single-run lineage trace.",
            "Uses the color-corrected display RGB, without spatial or generative scene editing.",
            "The recorded one-to-one match and readout IoU agree; no manual box edits.",
            "HSR preserves native 3D geometry and changes only scores and ranking."
        ],
    }
    assets = [full, crop, legend, comparison, history_path]
    audit["outputs"] = [{"file": path.name, "sha256": source.common.sha(path)} for path in assets]
    manifest = OUT / "source_manifest.json"
    source.save_json(manifest, audit)
    with zipfile.ZipFile(OUT.with_suffix(".zip"), "w", zipfile.ZIP_DEFLATED) as archive:
        for path in assets + [manifest]:
            archive.write(path, path.name)
    print(json.dumps({"image": str(full), "frame": FRAME, "native_id": IDENTITY,
                      "match_iou2d": iou2d, "event": event, "verified": True}))


if __name__ == "__main__":
    main()
