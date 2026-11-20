#!/usr/bin/env python3
"""Package audited scene0030_01 assets for the CaRR-3D method flowchart.

This script only copies and composes existing, verified outputs.  It does not
modify prediction geometry, fill mesh holes, or run a detector.
"""

from pathlib import Path
import json
import shutil

from PIL import Image, ImageDraw, ImageFont, ImageOps


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "figure_assets/carr3d_framework_real_scene0030_01_20261114_bold"
OUT = ROOT / "figure_assets/carr3d_flowchart_assets_scene0030_01_vcr_20261115"


ASSETS = [
    ("01_input/rgb_keyframe_f1100.png", "01_input/rgb_f1100.png", "Posed RGB keyframe"),
    ("01_input/depth_keyframe_f1100.png", "01_input/depth_f1100.png", "Aligned depth"),
    ("02_native/cutr_candidates_f1100.png", "02_native/cutr_proposals_f1100.png", "Native CuTR candidates"),
    ("02_native/native_map_previous_f1575.png", "02_native/native_map_previous_f1575_white.png", "Previous native map"),
    ("02_native/native_map_current_f1600.png", "02_native/native_map_current_f1600_white.png", "Updated native map"),
    ("03_auxiliary/lifted_postnms_proposal_f1250.png", "03_auxiliary/lifted_proposal_f1250.png", "Lifted post-NMS proposal"),
    ("03_auxiliary/lifted_prenms_candidate_f1050.png", "03_auxiliary/lifted_anchor_f1050.png", "Lifted pre-NMS candidate"),
    ("04_ptr/ptr_three_supporting_keyframes.png", "04_ptr/ptr_three_keyframes.png", "PTR: three supporting keyframes"),
    ("04_ptr/ptr_recovery_3d.png", "04_ptr/ptr_recovery_context_white.png", "PTR recovery"),
    ("05_vcr/vcr_three_supporting_keyframes.png", "05_var/var_three_keyframes.png", "VCR: voxel-consistent candidates"),
    ("05_vcr/vcr_recovery_3d.png", "05_var/var_recovery_context_white.png", "VCR recovery"),
    ("06_hsr/native_proposal_match_f1400.png", "06_hsr/native_proposal_match_f1400.png", "HSR: native-proposal match"),
    ("06_hsr/historical_support.png", "06_hsr/actual_support_history.png", "HSR: historical maximum"),
    ("06_hsr/native_geometry_unchanged.png", "06_hsr/native_geometry_unchanged_white.png", "HSR: fixed native geometry"),
    ("07_output/carr3d_selected_output.png", "07_output/carr3d_output_selected_white.png", "CaRR-3D selected output"),
    ("07_output/scene_mesh_context.png", "07_output/scene_mesh_white.png", "Scene context"),
]


def font(size: int, bold: bool = False):
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    return ImageFont.truetype(name, size)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    copied = []
    for relative_out, relative_source, label in ASSETS:
        source = SOURCE / relative_source
        target = OUT / relative_out
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        copied.append((target, label, relative_source))

    # A compact index for visual checking; individual source-resolution images
    # remain untouched in their respective subdirectories.
    tile_w, tile_h = 520, 390
    margin, gap, title_h = 34, 24, 92
    cols, rows = 4, 4
    canvas = Image.new(
        "RGB",
        (margin * 2 + cols * tile_w + (cols - 1) * gap,
         title_h + margin + rows * tile_h + (rows - 1) * gap + margin),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    draw.text(
        (canvas.width // 2, 28),
        "CaRR-3D flowchart assets — real scene0030_01",
        anchor="mt", font=font(34, True), fill="#172A40",
    )
    for index, (path, label, _) in enumerate(copied):
        row, col = divmod(index, cols)
        x = margin + col * (tile_w + gap)
        y = title_h + row * (tile_h + gap)
        image = Image.open(path).convert("RGB")
        thumb = ImageOps.contain(image, (tile_w - 20, tile_h - 70), Image.Resampling.LANCZOS)
        px = x + (tile_w - thumb.width) // 2
        py = y + 8 + (tile_h - 70 - thumb.height) // 2
        canvas.paste(thumb, (px, py))
        draw.rounded_rectangle(
            (x, y, x + tile_w, y + tile_h), radius=12,
            outline="#C9D2DC", width=2,
        )
        draw.text(
            (x + tile_w // 2, y + tile_h - 45), label,
            anchor="mt", font=font(21, True), fill="#27384D",
        )
    overview = OUT / "00_all_flowchart_assets.png"
    canvas.save(overview, dpi=(300, 300))

    source_manifest = json.loads((SOURCE / "source_manifest.json").read_text())
    manifest = {
        "scene": "scene0030_01",
        "source_pack": str(SOURCE),
        "terminology": {
            "PTR": "Proposal Track Recovery",
            "VCR": "Voxelized Candidate Recovery",
            "HSR": "Historical Support Reranking",
            "pre_nms_stream": "decoded pre-NMS candidates, not predefined anchor boxes",
        },
        "prediction_geometry_modified": False,
        "mesh_holes_filled": False,
        "generated_furniture_or_detections": False,
        "selection_protocol": source_manifest.get("notes", []),
        "ptr": source_manifest["ptr"],
        "vcr": source_manifest["var"],
        "hsr": source_manifest["hsr"],
        "files": [
            {"path": str(path.relative_to(OUT)), "label": label,
             "source": source}
            for path, label, source in copied
        ],
    }
    (OUT / "source_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n"
    )

    readme = """CaRR-3D 流程图真实素材包 — scene0030_01

本包使用同一真实 ScanNet 场景，覆盖流程图需要的 RGB-D 输入、原生建图、辅助候选、PTR、VCR、HSR 和最终输出素材。

推荐使用：
1. 输入：01_input/rgb_keyframe_f1100.png、depth_keyframe_f1100.png
2. 原生路径：02_native/cutr_candidates_f1100.png、native_map_previous_f1575.png、native_map_current_f1600.png
3. 辅助路径：03_auxiliary/lifted_postnms_proposal_f1250.png、lifted_prenms_candidate_f1050.png
4. PTR：04_ptr/ptr_three_supporting_keyframes.png、ptr_recovery_3d.png
5. VCR：05_vcr/vcr_three_supporting_keyframes.png、vcr_recovery_3d.png
6. HSR：06_hsr/native_proposal_match_f1400.png、historical_support.png、native_geometry_unchanged.png
7. 输出：07_output/carr3d_selected_output.png

真实性与使用边界：
- 所有检测框均来自已冻结的真实缓存；未修改坐标、尺寸、分数或分支来源。
- 未填补场景网格空洞，也未生成家具或检测框。
- PTR 和 VCR 的三帧图是各自真实状态中的三个连续支持关键帧，并非同一个同步时刻。
- HSR 调整原生分数与排名，不改变原生框几何。
- 终态图只显示选定记录，不代表全部输出。
- VCR 使用解码后的 pre-NMS candidates，而非预定义 anchor boxes。
"""
    (OUT / "使用说明.txt").write_text(readme)
    print(overview)


if __name__ == "__main__":
    main()
