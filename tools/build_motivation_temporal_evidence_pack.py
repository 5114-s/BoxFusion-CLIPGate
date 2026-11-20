#!/usr/bin/env python3
"""Build a paper-figure asset pack from frozen ReCaR-3D evidence.

This script only composes and copies existing measured assets. It does not run
models, alter detections, or create synthetic evidence.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps


ROOT = Path("/data/ZhaoX/BoxFusion")
OUT = ROOT / "figure_assets/motivation_temporal_evidence_20261101"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    return ImageFont.truetype(f"/usr/share/fonts/truetype/dejavu/{name}", size)


def fit_rgba(image: Image.Image, size: tuple[int, int], bg=(255, 255, 255, 255)) -> Image.Image:
    image = image.convert("RGBA")
    fitted = ImageOps.contain(image, size, Image.Resampling.LANCZOS)
    canvas = Image.new("RGBA", size, bg)
    canvas.alpha_composite(fitted, ((size[0] - fitted.width) // 2, (size[1] - fitted.height) // 2))
    return canvas


def depth_preview(path: Path, size: tuple[int, int]) -> Image.Image:
    arr = np.asarray(Image.open(path), dtype=np.float32)
    valid = arr > 0
    out = np.zeros_like(arr, dtype=np.uint8)
    if valid.any():
        lo, hi = np.percentile(arr[valid], [2, 98])
        norm = np.clip((arr - lo) / max(hi - lo, 1.0), 0.0, 1.0)
        # Near surfaces are brighter, matching the visual convention used elsewhere.
        out[valid] = np.round((1.0 - norm[valid]) * 215 + 30).astype(np.uint8)
    return Image.fromarray(out, mode="L").convert("RGB").resize(size, Image.Resampling.BILINEAR)


def build_rgbd_triptych() -> list[Path]:
    scene = "scene0432_01"
    frames = [0, 25, 50]
    panel_w, panel_h = 720, 538
    gap, border = 24, 18
    label_h = 58
    canvas = Image.new(
        "RGB",
        (border * 2 + panel_w * 3 + gap * 2, border * 2 + label_h + panel_h),
        "white",
    )
    d = ImageDraw.Draw(canvas)
    for i, frame_id in enumerate(frames):
        x = border + i * (panel_w + gap)
        rgb_path = Path(f"/extra/ZhaoX/scannet_data/scans/{scene}/color/{frame_id}.jpg")
        dep_path = Path(f"/extra/ZhaoX/scannet_data/scans/{scene}/depth/{frame_id}.png")
        rgb = Image.open(rgb_path).convert("RGB").resize((panel_w, panel_h), Image.Resampling.LANCZOS)
        dep = depth_preview(dep_path, (190, 143))
        rgb.paste(dep, (panel_w - dep.width - 16, panel_h - dep.height - 16))
        ImageDraw.Draw(rgb).rectangle(
            (panel_w - dep.width - 17, panel_h - dep.height - 17, panel_w - 15, panel_h - 15),
            outline=(255, 255, 255), width=3,
        )
        canvas.paste(rgb, (x, border + label_h))
        d.text((x + 10, border + 6), f"t{i + 1}  |  frame {frame_id}", fill=(25, 25, 25), font=font(34, True))
    d.text((canvas.width - 310, canvas.height - 42), "depth inset", fill=(255, 255, 255), font=font(24, True),
           stroke_width=2, stroke_fill=(0, 0, 0))
    out = OUT / "01_top_posed_rgbd_keyframes_scene0432_01_f0000_f0025_f0050.png"
    canvas.save(out, dpi=(300, 300))

    clean = Image.new("RGB", canvas.size, "white")
    dc = ImageDraw.Draw(clean)
    for i, frame_id in enumerate(frames):
        x = border + i * (panel_w + gap)
        rgb_path = Path(f"/extra/ZhaoX/scannet_data/scans/{scene}/color/{frame_id}.jpg")
        rgb = Image.open(rgb_path).convert("RGB").resize((panel_w, panel_h), Image.Resampling.LANCZOS)
        clean.paste(rgb, (x, border + label_h))
        dc.text((x + 10, border + 6), f"t{i + 1}  |  frame {frame_id}", fill=(25, 25, 25), font=font(34, True))
    clean_out = OUT / "01b_top_rgb_keyframes_scene0432_01_f0000_f0025_f0050.png"
    clean.save(clean_out, dpi=(300, 300))
    return [out, clean_out]


def copy_asset(source: Path, name: str) -> Path:
    target = OUT / name
    shutil.copy2(source, target)
    return target


def crop_hcsr_case(source: Path) -> Path:
    im = Image.open(source).convert("RGBA")
    # The final paired figure is symmetric. Keep the complete left case and a
    # narrow safety margin before the right-case title.
    crop = im.crop((0, 0, int(im.width * 0.505), im.height))
    out = OUT / "10_hcsr_reranking_real_case_scene0644_00.png"
    crop.save(out, dpi=(300, 300))
    return out


def build_index(items: list[tuple[str, Path]]) -> Path:
    card_w, card_h = 720, 460
    cols = 2
    rows = (len(items) + cols - 1) // cols
    margin, gap = 36, 28
    header_h = 96
    canvas = Image.new("RGBA", (margin * 2 + cols * card_w + gap, header_h + margin + rows * (card_h + gap)), "white")
    d = ImageDraw.Draw(canvas)
    d.text((margin, 22), "Motivation figure: verified real-evidence asset index", fill=(20, 28, 45), font=font(40, True))
    for idx, (title, path) in enumerate(items):
        row, col = divmod(idx, cols)
        x = margin + col * (card_w + gap)
        y = header_h + row * (card_h + gap)
        d.rounded_rectangle((x, y, x + card_w, y + card_h), radius=18, fill=(248, 250, 253), outline=(195, 205, 219), width=3)
        d.text((x + 18, y + 14), title, fill=(20, 28, 45), font=font(27, True))
        im = Image.open(path)
        thumb = fit_rgba(im, (card_w - 32, card_h - 70), bg=(248, 250, 253, 255))
        canvas.alpha_composite(thumb, (x + 16, y + 58))
    out = OUT / "00_asset_index.png"
    canvas.convert("RGB").save(out, dpi=(220, 220))
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    generated = build_rgbd_triptych()

    sources = {
        "02_native_map_scene0432_01.png": ROOT / "figure_assets/pipeline_candidate_visuals_scene0432_01/04b_native_map_NT_native_only.png",
        "03_post_nms_proposal_track_scene0432_01_f0000_f0025_f0050.png": ROOT / "figure_assets/pipeline_candidate_visuals_scene0432_01/08_plr_cross_keyframe_verification_ready.png",
        "04_post_nms_proposals_multiview_scene0432_01.png": ROOT / "figure_assets/pipeline_candidate_visuals_scene0432_01/05a_wedetect_multiview_2d_proposals.png",
        "05_pre_nms_anchors_multiview_scene0432_01.png": ROOT / "figure_assets/pipeline_candidate_visuals_scene0432_01/05a_pre_nms_anchors_multiview.png",
        "10_hcsr_reranking_real_case_scene0644_00.png": ROOT / "figure_assets/real_qualitative_replacement_20261026/hcsr_real_case.png",
        "20_plr_three_keyframes_real_case_scene0426_03.png": ROOT / "figure_assets/real_qualitative_replacement_20261026/plr_three_keyframes.png",
        "21_plr_native_vs_recovered_scene0426_03.png": ROOT / "figure_assets/real_qualitative_replacement_20261026/plr_native_vs_recovered.png",
        "30_calr_anchor_consensus_real_case_scene0606_01.png": ROOT / "figure_assets/qualitative_fig5_threshold_label_20261029/cases/calr_recovery_2x2_source.png",
        "40_hcsr_pair_vector_source.svg": ROOT / "figure_assets/hcsr_qualitative_pair_20261031/hcsr_qualitative_main_pair.svg",
        "41_plr_calr_pair_vector_source.svg": ROOT / "figure_assets/hcsr_qualitative_pair_20261031/recovery_qualitative_supplement_pair.svg",
    }
    copied = [copy_asset(src, name) for name, src in sources.items()]
    hcsr = OUT / "10_hcsr_reranking_real_case_scene0644_00.png"

    index_items = [
        ("Top input: posed RGB-D, t1/t2/t3", generated[0]),
        ("Native object map", OUT / "02_native_map_scene0432_01.png"),
        ("Persistent post-NMS proposal track", OUT / "03_post_nms_proposal_track_scene0432_01_f0000_f0025_f0050.png"),
        ("Pre-NMS anchor evidence", OUT / "05_pre_nms_anchors_multiview_scene0432_01.png"),
        ("HCSR: native-record reranking", hcsr),
        ("PLR: cross-keyframe proposal evidence", OUT / "20_plr_three_keyframes_real_case_scene0426_03.png"),
        ("PLR: native map vs recovered target", OUT / "21_plr_native_vs_recovered_scene0426_03.png"),
        ("CALR: anchor consensus and recovery", OUT / "30_calr_anchor_consensus_real_case_scene0606_01.png"),
    ]
    index = build_index(index_items)

    source_meta = {
        "schema": "recar3d.motivation_temporal_evidence.v1",
        "created": "2026-11-01",
        "generative_images_used": False,
        "new_model_inference": False,
        "top_timeline": {
            "scene": "scene0432_01",
            "frames": [0, 25, 50],
            "source_cache": str(ROOT / "development/final_controls/evidence/scannet_recar3d_key_controls/scene0432_01.npz"),
            "note": "RGB and depth are raw ScanNet observations. Proposal/anchor overlays are measured cached outputs from the same scene and time sequence. The native map is a native-only map visualization and should be labelled N^(t) only when the selected export time is stated; otherwise use Native object map.",
        },
        "hcsr": {
            "scene": "scene0644_00",
            "native_index": 42,
            "score": [0.506031334400177, 0.6269968088560498],
            "rank": [51, 32],
            "warning": "The exported qualitative evidence shows a representative matched view with historical-maximum aggregation. Do not redraw it as three verified matched views unless a separate multi-view trace is exported.",
        },
        "plr": {
            "scene": "scene0426_03",
            "shown_frames": [425, 450, 550],
            "confirmation_frame": 550,
            "recovered_3d_aabb_iou": 0.5755334938979785,
        },
        "calr": {
            "scene": "scene0606_01",
            "shown_frames": [1400, 1575, 1600],
            "support_frames_terminal": [1400, 1575, 1600, 1625, 1875],
            "voxel_key": [8, 16, 2],
            "future_frames_used": False,
        },
        "sources": {str(p.relative_to(ROOT)): sha256(p) for p in sources.values()},
        "outputs": {},
    }

    readme = """# ReCaR-3D temporal-evidence motivation figure assets

This directory contains only real observations and frozen experiment outputs. No generative image and no new model inference was used.

## Recommended placement

### Upper timeline

- `01_top_posed_rgbd_keyframes...png`: common input sequence (`scene0432_01`, frames 0/25/50). A clean RGB-only version is also provided.
- `02_native_map_scene0432_01.png`: native object-map visual. Repeat it or reveal it progressively to show that the baseline already maintains object history.
- `03_post_nms_proposal_track...png`: one measured post-NMS proposal track across the same three keyframes. In the baseline row, show the three observations without connecting lines; in the ReCaR-3D row, add the temporal association line in the vector editor.
- `04_post_nms_proposals...png` and `05_pre_nms_anchors...png`: measured multi-view proposal/anchor evidence. Use a small crop to illustrate unretained auxiliary evidence or clutter; do not imply that every displayed anchor is kept indefinitely.

The upper row should communicate state retention, not rerun the full pipeline. Add arrows and broken links as vector elements in Draw.io/Illustrator so the raster evidence remains unchanged.

### Lower cases

- `10_hcsr_reranking_real_case_scene0644_00.png`: same native geometry, score 0.506→0.627 and rank 51→32. The available exported trace contains one representative matched view under historical-maximum aggregation; do not fabricate three matched views.
- `20_plr_three_keyframes...png` + `21_plr_native_vs_recovered...png`: real PLR case (`scene0426_03`, frames 425/450/550), confirmed at frame 550.
- `30_calr_anchor_consensus...png`: real CALR case (`scene0606_01`, frames 1400/1575/1600); the terminal voxel has five supporting frames, while three representative frames are shown.
- `40_...svg` and `41_...svg`: vector-source figures if text or line placement must be edited.

The HCSR, PLR, and CALR examples come from different scenes. Keep their scene/frame labels or state in the caption that representative cases are selected independently.

## Safe visual claims

- The native object map already retains accepted-object history.
- ReCaR-3D additionally maintains bounded causal state for auxiliary proposal/anchor evidence.
- HCSR changes native scores/ranks while leaving native geometry and identity unchanged.
- PLR associates post-NMS proposal records across distinct keyframes before recovery.
- CALR accumulates sparse world-space voxel support from lifted top-M pre-NMS anchors; it is not dense scene reconstruction.

Point clouds/meshes in the local map views are visualization backgrounds, not method inputs.
"""
    (OUT / "README.md").write_text(readme, encoding="utf-8")

    all_outputs = sorted(p for p in OUT.iterdir() if p.is_file() and p.name not in {"source_manifest.json", "source_hashes.sha256"})
    source_meta["outputs"] = {p.name: sha256(p) for p in all_outputs}
    (OUT / "source_manifest.json").write_text(json.dumps(source_meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    hashes = [f"{sha256(p)}  {p.name}" for p in sorted(OUT.iterdir()) if p.is_file() and p.name != "source_hashes.sha256"]
    (OUT / "source_hashes.sha256").write_text("\n".join(hashes) + "\n", encoding="utf-8")
    print(index)
    print(f"Built {len(list(OUT.iterdir()))} files in {OUT}")


if __name__ == "__main__":
    main()
