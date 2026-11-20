#!/usr/bin/env python3
"""Export verified RGB-D measurements for the scene0030_01 flowchart.

No generated scene content, RGB retouching, depth completion, or inference.
The color-corrected display images already exist; exact archived input files
are copied separately. Depth visualization uses one shared physical scale.
"""
from pathlib import Path
import hashlib
import json
import shutil
import zipfile

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

ROOT = Path(__file__).resolve().parents[2]
SCENE = "scene0030_01"
FRAMES = (100, 1100, 1200)
ARCHIVED = ROOT / "upstream_clean/scannet_readme_frames" / SCENE / "frames"
DISPLAY = ROOT / "figure_assets/carr3d_framework_real_scene0030_01_20261114_bold/01_input"
CONFIG = ROOT / "reports/official_github_boxfusion_vs_final_recar_public_cli_coldcache_nowarmup100_20261009/receipts/recar" / SCENE / "effective_config.yaml"
OUT = ROOT / "figure_assets/scene0030_01_rgbd_three_frames_20261116"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def font(size, bold=False):
    return ImageFont.truetype("DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf", size)


def pair(rgb, depth, frame):
    # Layout only. Each image keeps its aspect ratio and all its original field
    # of view; raw depth is never resized in the saved measurement files.
    width, height, margin, gap = 1296, 972, 40, 32
    panel = Image.new("RGB", (2 * width + 2 * margin + gap, height + 180), "white")
    draw = ImageDraw.Draw(panel)
    for i, (im, label, filter_) in enumerate([
        (rgb, "RGB (existing color-corrected display)", Image.Resampling.LANCZOS),
        (depth.convert("RGB"), "Depth (0.5-5.0 m; near = bright)", Image.Resampling.NEAREST),
    ]):
        x = margin + i * (width + gap)
        tile = ImageOps.contain(im, (width, height), filter_)
        panel.paste(tile, (x + (width-tile.width)//2, 100 + (height-tile.height)//2))
        draw.text((x, 60), label, font=font(28), fill="#20334A")
    draw.text((margin, 12), f"{SCENE} | frame {frame}", font=font(30, True), fill="#20334A")
    draw.text((margin, height+120), "Same frame ID. Black depth pixels are invalid measurements; no hole filling.",
              font=font(26), fill="#526274")
    return panel


def main():
    OUT.mkdir(parents=True, exist_ok=False)
    for name in ("rgb_input_original", "rgb_display", "depth_visual", "depth_raw_mm", "paired"):
        (OUT/name).mkdir()
    assert str(ARCHIVED) in CONFIG.read_text()
    manifest = {
        "scene": SCENE, "frames": list(FRAMES), "selection": "Previously selected frames from this scene's flowchart asset pack, not consecutive keyframes",
        "reference_3d_image": str(ROOT / "figure_assets/carr3d_framework_real_scene0030_01_20261114_bold/07_output/carr3d_output_selected_white.png"),
        "archived_run_config": str(CONFIG), "archived_run_config_sha256": sha(CONFIG),
        "rgb_input_original": "Byte-identical copies of archived input JPEGs; not resized, cropped or recolored",
        "rgb_display": "Copies of the previous flowchart's color-corrected display PNGs; not the exact archived inference-input pixels",
        "depth": {"raw_unit": "millimetres", "raw_type": "uint16", "invalid": 0,
            "display_scale_mm": [500, 5000], "display": "Shared linear inverse grayscale; valid values clipped to 0.5-5.0m map from245 to35; invalid remains0",
            "depth_completion": False, "metric_depth_modified": False},
        "pairing": "Matched frame IDs. RGB 1296x968 and depth640x480 retain their native sizes; no new registration or crop is applied",
        "generated_content": False, "gpu_used": False, "records": [],
    }
    panels = []
    for frame in FRAMES:
        input_rgb = ARCHIVED / "color" / f"{frame}.jpg"
        input_depth = ARCHIVED / "depth" / f"{frame}.png"
        display_rgb = DISPLAY / f"rgb_f{frame:04d}.png"
        depth = np.asarray(Image.open(input_depth))
        assert depth.dtype == np.uint16 and depth.shape == (480, 640)
        # Check the previous pack and this archived run carry the same depth.
        assert np.array_equal(depth, np.asarray(Image.open(DISPLAY / f"depth_raw_mm_f{frame:04d}.png")))
        rgb = Image.open(display_rgb).convert("RGB")
        assert rgb.size == (1296, 968)
        rgb_target = OUT / "rgb_input_original" / f"rgb_f{frame:04d}.jpg"
        depth_target = OUT / "depth_raw_mm" / f"depth_raw_mm_f{frame:04d}.png"
        display_target = OUT / "rgb_display" / f"rgb_f{frame:04d}.png"
        for source, target in ((input_rgb, rgb_target), (input_depth, depth_target), (display_rgb, display_target)):
            shutil.copy2(source, target)
            assert sha(source) == sha(target)
        valid = depth > 0
        gray = np.zeros(depth.shape, dtype=np.uint8)
        gray[valid] = np.rint(245 - 210*np.clip((depth[valid].astype(float)-500)/4500, 0, 1)).astype(np.uint8)
        vis = Image.fromarray(gray)
        vis_target = OUT / "depth_visual" / f"depth_f{frame:04d}.png"
        vis.save(vis_target, dpi=(400, 400))
        assert np.array_equal(np.asarray(Image.open(vis_target)) == 0, ~valid)
        panel = pair(rgb, vis, frame)
        panel_target = OUT / "paired" / f"rgbd_f{frame:04d}.png"
        panel.save(panel_target, dpi=(400, 400))
        panels.append(panel)
        manifest['records'].append({"frame": frame, "rgb_input_source": str(input_rgb),
            "rgb_input_sha256": sha(input_rgb), "rgb_display_source": str(display_rgb),
            "rgb_display_sha256": sha(display_rgb), "depth_source": str(input_depth),
            "depth_source_sha256": sha(input_depth), "depth_min_valid_mm": int(depth[valid].min()),
            "depth_max_mm": int(depth.max()), "depth_invalid_percent": 100*float((~valid).mean()),
            "rgb_wh": list(rgb.size), "depth_wh": list(vis.size)})
    thumbs = [ImageOps.contain(im, (1400, 600), Image.Resampling.LANCZOS) for im in panels]
    overview = Image.new("RGB", (1400, sum(im.height for im in thumbs)), "white")
    y = 0
    for im in thumbs:
        overview.paste(im, (0, y)); y += im.height
    overview.save(OUT / "preview_rgbd_three_frames.png", dpi=(300, 300))
    (OUT / "source_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+"\n")
    (OUT / "使用说明.txt").write_text(
        "scene0030_01：三组真实 RGB-D 帧（100、1100、1200）\n\n"
        "rgb_input_original：归档推理数据目录中的原始JPEG，字节完全一致。\n"
        "rgb_display：此前流程图使用的色彩修正版PNG，仅供展示，并非原始推理输入像素。\n"
        "depth_visual：同帧深度灰度图，三帧统一采用0.5–5.0米显示范围，近亮远暗，黑色为原始无效深度。\n"
        "depth_raw_mm：16位原始深度PNG，单位毫米，0表示无效；不要以8位显示灰度代替真实深度值。\n"
        "paired：三张RGB（展示版）与深度的并列图；preview为三组总览。\n\n"
        "图像来自同一真实场景。三个帧号按时间递增，但并非连续关键帧，不要标注为连续t-2/t-1/t。\n"
        "未生成场景、改变物体或补洞。RGB和深度保留各自原始尺寸，本次未做新的配准。\n"
        "场景三维网格仅为先前可视化背景，不是上述RGB-D输入的生成来源。\n", encoding="utf-8")
    for path in OUT.rglob("*.png"):
        with Image.open(path) as im:
            im.verify()
    archive = OUT.with_suffix(".zip")
    assert not archive.exists()
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for path in sorted(OUT.rglob("*")):
            if path.is_file():
                z.write(path, path.relative_to(OUT))
    print(json.dumps({"out": str(OUT), "archive": str(archive), "frames": FRAMES,
        "original_RGB_and_depth_byte_identical": True, "validation": "PASS"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
