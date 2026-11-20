#!/usr/bin/env python3
"""Build a clear, audited CaRR-3D flowchart asset pack from scene0599_01.

All photographic pixels, candidate boxes, lifted 3D boxes, and terminal
predictions come from archived experiment artifacts.  The script only selects,
projects, crops, labels, and packages them; it does not infer or alter boxes.
"""
from __future__ import annotations

import json
import shutil
import sys
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps


ROOT = Path(__file__).resolve().parents[2]
SCENE = "scene0599_01"
OUT = ROOT / "figure_assets/carr3d_flowchart_clear_real_scene0599_01_vcr_20261115"
ZIP = OUT.with_suffix(".zip")
OLD = ROOT / "figure_assets/carr3d_framework_verified_real_pack_scene0599_20261107"
AUDIT = ROOT / "reports/three_module_scene_search_20261115" / SCENE
SELECTION = ROOT / "figure_assets/three_module_scene_shortlist_20261115/scene0599_01_cleansearch_selection.json"
RGB = ROOT / "data/scannet_val_rgbfix" / SCENE / "frames/color"
DEPTH = ROOT / "data/scannet_val_rgbfix" / SCENE / "frames/depth"
EVIDENCE = ROOT / "development/final_controls/experiments/mvsr_unified_samepass_20261010/evidence" / f"{SCENE}.npz"
RAW = ROOT / "reports/scannet_seedless_full100_20260911/raw" / SCENE
GEOMETRY = AUDIT / "verified_geometry.npz"

BLUE = "#2793C7"
ORANGE = "#F28B16"
TEAL = "#00A889"
PURPLE = "#8054AF"
INK = "#1A2A40"
GREY = "#AAB4C0"

sys.path.insert(0, str(ROOT / "tools/qualitative"))
import build_framework_scene0030_real_assets as source  # noqa: E402


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype("DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf", size)


def ensure_dirs() -> None:
    for name in (
        "01_input_stream", "02_native_mapping", "03_auxiliary_candidates",
        "04_hsr", "05_ptr", "06_vcr", "07_online_output", "08_audit",
    ):
        (OUT / name).mkdir(parents=True, exist_ok=True)


def copy(src: Path, dst: Path) -> Path:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return dst


def scale_box(box: np.ndarray, width: int, height: int) -> np.ndarray:
    out = np.asarray(box, dtype=float).copy()
    out[[0, 2]] *= width / 640.0
    out[[1, 3]] *= height / 480.0
    return out


def crop_around(image: Image.Image, boxes: list[np.ndarray], ratio: float = 1.35) -> Image.Image:
    values = np.asarray(boxes, dtype=float)
    x1, y1 = values[:, :2].min(0)
    x2, y2 = values[:, 2:].max(0)
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    bw, bh = max(x2 - x1, 100.0) * ratio, max(y2 - y1, 100.0) * ratio
    target = 1.40
    if bw / bh < target:
        bw = bh * target
    else:
        bh = bw / target
    left, top = max(0, cx - bw / 2), max(0, cy - bh / 2)
    right, bottom = min(image.width, cx + bw / 2), min(image.height, cy + bh / 2)
    return image.crop((round(left), round(top), round(right), round(bottom)))


def overlay_rectangles(
    frame: int,
    boxes: list[np.ndarray],
    colors: list[str],
    destination: Path,
    dashed: set[int] | None = None,
) -> tuple[Path, Path]:
    image = Image.open(RGB / f"{frame}.jpg").convert("RGB")
    scaled = [scale_box(box, image.width, image.height) for box in boxes]
    draw = ImageDraw.Draw(image)
    width = max(6, round(image.width / 190))
    dashed = dashed or set()
    for index, (box, color) in enumerate(zip(scaled, colors)):
        xy = tuple(map(float, box))
        if index not in dashed:
            draw.rectangle(xy, outline="white", width=width + 4)
            draw.rectangle(xy, outline=color, width=width)
            continue
        x1, y1, x2, y2 = xy
        length, gap = 22, 12
        for start in np.arange(x1, x2, length + gap):
            draw.line((start, y1, min(start + length, x2), y1), fill=color, width=width)
            draw.line((start, y2, min(start + length, x2), y2), fill=color, width=width)
        for start in np.arange(y1, y2, length + gap):
            draw.line((x1, start, x1, min(start + length, y2)), fill=color, width=width)
            draw.line((x2, start, x2, min(start + length, y2)), fill=color, width=width)
    full = destination.with_name(destination.name + "_full.png")
    crop = destination.with_name(destination.name + "_crop.png")
    image.save(full, dpi=(400, 400))
    crop_around(image, scaled).save(crop, dpi=(400, 400))
    return full, crop


def depth_render(frame: int, dst: Path) -> Path:
    depth = np.asarray(Image.open(DEPTH / f"{frame}.png"), dtype=float)
    valid = depth > 0
    lo, hi = np.percentile(depth[valid], [4, 96])
    norm = np.clip((depth - lo) / max(hi - lo, 1.0), 0.0, 1.0)
    out = (255.0 * (1.0 - norm)).astype(np.uint8)
    out[~valid] = 245
    Image.fromarray(out, mode="L").save(dst, dpi=(400, 400))
    return dst


def raw_candidate_box(frame: int, candidate_id: int) -> np.ndarray:
    with np.load(RAW / f"raw_{frame:06d}.npz") as raw:
        return np.asarray(raw["boxes"][int(candidate_id)], dtype=float)


def render_mesh_panels(selection: dict, audit: dict) -> dict[str, Path]:
    renderer = source.common.load_renderer()
    geometry = np.load(GEOMETRY)
    matrix = geometry["axis_alignment"]
    all_boxes = geometry["boxes_world"] @ matrix[:3, :3].T + matrix[:3, 3]
    selected_ids = selection["selected_prediction_indices"]
    selected = all_boxes[selected_ids]
    objects = json.loads((source.common.RAW / SCENE / f"{SCENE}.aggregation.json").read_text())["segGroups"]
    walls = [int(obj["objectId"]) for obj in objects if obj["label"] == "wall"]
    xyz, _, display_rgb, faces, _ = renderer.load_mesh(SCENE, matrix, walls)
    view = next(item for item in selection["camera_views"] if float(item["azimuth"]) == -45.0)
    width, height = map(int, view["image_size"])
    projection = (
        np.asarray(view["position"], dtype=float),
        np.asarray(view["look_at"], dtype=float),
        float(view["focal_pixels"]),
        np.asarray(view["principal_point_pixels"], dtype=float),
    )
    projected = renderer.project(xyz, *projection)
    rgba = renderer.rasterize(projected, display_rgb, faces, width, height)

    roles = ["hsr", "ptr", "vcr"]
    colors = {"hsr": BLUE, "ptr": ORANGE, "vcr": TEAL}
    result: dict[str, Path] = {}
    for name, keep in (("hsr", {0}), ("ptr", {1}), ("vcr", {2}), ("all", {0, 1, 2})):
        image = Image.fromarray(rgba.copy())
        draw = ImageDraw.Draw(image)
        for index in sorted(keep):
            uv = renderer.project(selected[index], *projection)[:, :2]
            for a, b in renderer.box_edges(selected[index]):
                segment = [tuple(uv[a]), tuple(uv[b])]
                draw.line(segment, fill="white", width=12)
                draw.line(segment, fill=colors[roles[index]], width=7)
        white = Image.new("RGB", image.size, "white")
        white.paste(image, mask=image.getchannel("A"))
        path = OUT / "07_online_output" / f"selected_{name}_3d_white.png"
        white.save(path, dpi=(400, 400))
        result[name] = path

    # Reference-style output card, retaining only the three audited targets.
    raw = Image.open(result["all"]).convert("RGB")
    content = ImageOps.contain(raw, (1480, 960), Image.Resampling.LANCZOS)
    panel = Image.new("RGB", (1600, 1290), "white")
    draw = ImageDraw.Draw(panel)
    draw.rounded_rectangle((5, 5, 1595, 1285), radius=30, outline=GREY, width=4)
    title = "Online 3D Detections"
    title_box = draw.textbbox((0, 0), title, font=font(58, True))
    draw.text(((1600 - (title_box[2] - title_box[0])) / 2, 25), title, fill=INK, font=font(58, True))
    panel.paste(content, ((1600 - content.width) // 2, 120))
    legend_y = 1155
    entries = ((BLUE, "Native / HSR"), (ORANGE, "Proposal / PTR"), (TEAL, "Candidate / VCR"))
    xs = (85, 565, 1080)
    for x, (color, label) in zip(xs, entries):
        draw.rounded_rectangle((x, legend_y, x + 64, legend_y + 38), radius=8, fill=color)
        draw.text((x + 82, legend_y - 3), label, fill=INK, font=font(32, True))
    panel_path = OUT / "07_online_output/online_3d_detections_reference_style.png"
    panel.save(panel_path, dpi=(400, 400))
    result["panel"] = panel_path

    source.save_json(OUT / "08_audit/selected_3d_targets.json", {
        "scene": SCENE,
        "selection": "GT-assisted qualitative selection; three predictions, not complete output",
        "prediction_indices": selected_ids,
        "hsr": selection["hsr"],
        "ptr": {
            "prediction_index": selected_ids[1],
            "gt_index": selection["pair"]["ptr_gt"],
            "iou": selection["pair"]["ptr_iou"],
        },
        "vcr": {
            "prediction_index": selected_ids[2],
            "gt_index": selection["pair"]["var_gt"],
            "iou": selection["pair"]["var_iou"],
            "legacy_internal_branch": "m1a / anchor_* fields",
        },
        "camera": view,
        "geometry_modified": False,
        "mesh_used_only_for_visualization": True,
    })
    return result


def contact_sheet(items: list[tuple[str, Path]], dst: Path, columns: int = 4) -> Path:
    tw, th, lh = 430, 290, 46
    rows = (len(items) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * tw, rows * (th + lh)), "white")
    draw = ImageDraw.Draw(sheet)
    for index, (label, path) in enumerate(items):
        image = Image.open(path).convert("RGBA")
        image = ImageOps.contain(image, (tw - 24, th - 18), Image.Resampling.LANCZOS)
        col, row = index % columns, index // columns
        x = col * tw + (tw - image.width) // 2
        y = row * (th + lh) + (th - image.height) // 2
        sheet.paste(image, (x, y), image)
        draw.text((col * tw + 12, row * (th + lh) + th + 6), label, fill=INK, font=font(19))
    sheet.save(dst, dpi=(400, 400))
    return dst


def main() -> None:
    ensure_dirs()
    audit = json.loads((AUDIT / "audit_summary.json").read_text())
    selection = json.loads(SELECTION.read_text())
    hsr_rows = json.loads((AUDIT / "hsr_verified_examples.json").read_text())
    lineage = json.loads((AUDIT / "branch_lineage.json").read_text())
    evidence = dict(source.replay.load_evidence(EVIDENCE))

    preview: list[tuple[str, Path]] = []

    # Posed RGB-D observations from the same stream.
    for frame in (0, 450, 900):
        rgb_dst = copy(RGB / f"{frame}.jpg", OUT / f"01_input_stream/rgb_frame_{frame:04d}.png")
        depth_dst = depth_render(frame, OUT / f"01_input_stream/depth_frame_{frame:04d}.png")
        preview.extend([(f"RGB frame {frame}", rgb_dst), (f"Depth frame {frame}", depth_dst)])

    # Native mapper assets from the already audited real pack.
    native_assets = (
        ("native_2d_inputs_frame_0025_crop.png", "native_2d_candidates.png", "Native frame candidates"),
        ("native_boxer_3d_projection_frame_0025_crop.png", "native_lifted_candidates.png", "Native lifted candidates"),
        ("previous_map.png", "previous_native_map.png", "Previous native map"),
        ("native_map.png", "current_native_map.png", "Current native map"),
    )
    for src_name, dst_name, label in native_assets:
        src_dir = "02_native_mapping"
        dst = copy(OLD / src_dir / src_name, OUT / "02_native_mapping" / dst_name)
        preview.append((label, dst))

    # Shared auxiliary front end: post-NMS proposals and dense pre-NMS candidates.
    proposal = copy(
        OLD / "03_auxiliary_candidates/post_nms_proposals_frame_0050_crop.png",
        OUT / "03_auxiliary_candidates/post_nms_proposals_frame_0050.png",
    )
    dense = copy(
        OLD / "03_auxiliary_candidates/top300_anchors_frame_0850_crop.png",
        OUT / "03_auxiliary_candidates/top300_prenms_candidates_frame_0850.png",
    )
    preview.extend([("Post-NMS proposals", proposal), ("Dense pre-NMS candidates", dense)])

    # HSR identity 58: exact projected native record and matched proposal.
    hsr_entries = [row for row in hsr_rows if row["native_id"] == 58 and row["frame"] in (350, 375, 400)]
    hsr_entries.sort(key=lambda row: row["frame"])
    for row in hsr_entries:
        _, crop = overlay_rectangles(
            row["frame"],
            [np.asarray(row["native_projection_xyxy"]), np.asarray(row["proposal_xyxy"])],
            [BLUE, PURPLE],
            OUT / "04_hsr" / f"native_proposal_match_frame_{row['frame']:04d}",
            dashed={1},
        )
        preview.append((f"HSR match, frame {row['frame']}", crop))
    source.save_json(OUT / "08_audit/hsr_identity58.json", hsr_entries)

    # PTR target: exact archived proposal observations for track 0.
    ptr_row = next(row for row in lineage if row["prediction_index"] == selection["selected_prediction_indices"][1])
    for frame in (0, 25, 50):
        src = OLD / "05_ptr_exact_target" / f"ptr_proposal_frame_{frame:04d}_crop.png"
        dst = copy(src, OUT / "05_ptr" / f"ptr_proposal_frame_{frame:04d}.png")
        preview.append((f"PTR proposal, frame {frame}", dst))
    source.save_json(OUT / "08_audit/ptr_selected_lineage.json", ptr_row)

    # VCR target: highest-scoring dense candidate in the selected voxel.
    vcr_row = next(row for row in lineage if row["prediction_index"] == selection["selected_prediction_indices"][2])
    voxel_key = np.asarray(vcr_row["state_row"]["voxel_key"], dtype=int)
    vcr_observations = []
    for frame in (0, 25, 50):
        ev = evidence[frame]
        centers = ev.anchor_corners.mean(axis=1)
        rows = np.flatnonzero(np.all(np.floor(centers / 0.3).astype(int) == voxel_key, axis=1))
        index = int(rows[np.argmax(ev.anchor_scores[rows])])
        candidate_id = int(ev.anchor_ids[index])
        box2d = raw_candidate_box(frame, candidate_id)
        _, crop = overlay_rectangles(
            frame, [box2d], [TEAL], OUT / "06_vcr" / f"vcr_prenms_candidate_frame_{frame:04d}"
        )
        preview.append((f"VCR candidate, frame {frame}", crop))
        vcr_observations.append({
            "frame": frame,
            "candidate_id": candidate_id,
            "score": float(ev.anchor_scores[index]),
            "voxel_key": voxel_key.tolist(),
            "legacy_internal_fields": "anchor_ids / anchor_corners / anchor_scores",
        })
    source.save_json(OUT / "08_audit/vcr_selected_lineage.json", {
        "terminal": vcr_row,
        "displayed_supporting_candidates": vcr_observations,
        "paper_term": "dense pre-NMS candidate",
    })

    rendered = render_mesh_panels(selection, audit)
    ptr_recovery = copy(rendered["ptr"], OUT / "05_ptr/ptr_recovery_3d.png")
    vcr_recovery = copy(rendered["vcr"], OUT / "06_vcr/vcr_recovery_3d.png")
    hsr_3d = copy(rendered["hsr"], OUT / "04_hsr/hsr_native_record_3d.png")
    preview.extend([
        ("HSR native record", hsr_3d),
        ("PTR recovery", ptr_recovery),
        ("VCR recovery", vcr_recovery),
        ("Selected online output", rendered["all"]),
    ])

    sheet = contact_sheet(preview, OUT / "00_all_flowchart_assets.png")
    contact_sheet(
        [("HSR: chair", hsr_3d), ("PTR: waste bin", ptr_recovery), ("VCR: door", vcr_recovery),
         ("Three selected outputs", rendered["all"])],
        OUT / "07_online_output/three_distinct_targets_overview.png",
        columns=4,
    )

    source.save_json(OUT / "manifest.json", {
        "schema": "carr3d.flowchart.clear_real_pack.v1",
        "scene": SCENE,
        "generative_images_used": False,
        "prediction_geometry_modified": False,
        "single_scene": True,
        "terminology": {
            "HSR": "Historical Support Reranking",
            "PTR": "Proposal Track Recovery",
            "VCR": "Voxelized Candidate Recovery",
            "pre_nms": "dense pre-NMS candidates; legacy code fields retain anchor_* names",
        },
        "selected_targets": {
            "HSR": "chair, native rank 4 to 3, terminal IoU 0.742",
            "PTR": "waste-bin-like other-furniture instance, IoU 0.734",
            "VCR": "door, IoU 0.743",
        },
        "selection_scope": "GT-assisted illustrative selection; not complete output",
        "module_examples_use_different_real_keyframes": True,
        "overview": str(sheet),
        "recommended_final_panel": str(rendered["panel"]),
        "sources": {
            "verified_old_pack": str(OLD),
            "audit": str(AUDIT / "audit_summary.json"),
            "selection": str(SELECTION),
            "geometry": str(GEOMETRY),
            "evidence_cache": str(EVIDENCE),
        },
    })

    readme = f"""# CaRR-3D 流程图真实素材包（{SCENE}）

本目录全部来自同一个 ScanNet 序列 `{SCENE}`。未使用生成式图像，未修改预测框坐标。

推荐用法：

- 输入与原生路径：`01_input_stream/`、`02_native_mapping/`
- 共享辅助前端：`03_auxiliary_candidates/`
- HSR、PTR、VCR：`04_hsr/`、`05_ptr/`、`06_vcr/`
- 最终面板：`07_online_output/online_3d_detections_reference_style.png`
- 全素材索引：`00_all_flowchart_assets.png`

三个终态示例对应不同目标：HSR 为椅子，PTR 为垃圾桶，VCR 为门。图中只展示经真实标注辅助选定的代表框，并非完整检测输出。模块示例来自同一真实序列的不同关键帧；场景网格仅用于可视化。

术语说明：论文图中统一使用 **dense pre-NMS candidate** 与 **VCR**。代码和旧缓存中的 `anchor_*`、`m1a` 仅是历史内部字段名。
"""
    (OUT / "README_CN.md").write_text(readme, encoding="utf-8")

    if ZIP.exists():
        ZIP.unlink()
    with zipfile.ZipFile(ZIP, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(OUT.rglob("*")):
            if path.is_file():
                archive.write(path, Path(OUT.name) / path.relative_to(OUT))
    print(OUT / "00_all_flowchart_assets.png")
    print(rendered["panel"])
    print(ZIP)


if __name__ == "__main__":
    main()
