#!/usr/bin/env python3
"""Make a cleaner motivation pair from the verified fixed-camera assets.

This is a selected qualitative illustration, not an evaluation modification.
Native records 5, 3 and 16 are retained in BOTH panels.  Coordinates, camera,
mesh, and the selected PLR record are copied exactly from the measured assets.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import zipfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
import matplotlib.patheffects as pe
import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "figure_assets/real_3d_motivation_20261026"
NATIVE_INDICES = [5, 3, 16]
OMITTED_FROM_PREVIOUS = [15, 0, 17]
BLUE = "#337CB6"
ORANGE = "#EE781C"
GREEN = "#179266"


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_renderer():
    path = ROOT / "tools/qualitative/render_real_3d_motivation.py"
    spec = importlib.util.spec_from_file_location("verified_real_3d_renderer", path)
    renderer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(renderer)
    return renderer


def draw_panel(ax, mesh, native_lines, recovery_lines, gt_lines,
               *, recovery, show_gt=False, annotate=False, bed_lines=None):
    height, width = mesh.shape[:2]
    ax.imshow(mesh, interpolation="antialiased")
    for lines in native_lines:
        ax.add_collection(LineCollection(
            lines, colors=BLUE, linewidths=1.65, alpha=.95, zorder=5,
            capstyle="round", joinstyle="round",
        ))
    if bed_lines is not None:
        ax.add_collection(LineCollection(
            bed_lines, colors=BLUE, linewidths=1.25, alpha=.84,
            linestyles=(0, (5, 3)), zorder=4,
            capstyle="round", joinstyle="round",
        ))
    if show_gt:
        ax.add_collection(LineCollection(
            gt_lines, colors=GREEN, linewidths=1.65,
            linestyles=(0, (4, 3)), zorder=7,
        ))
    if recovery:
        lines = LineCollection(
            recovery_lines, colors=ORANGE, linewidths=2.4,
            zorder=8, capstyle="round", joinstyle="round",
        )
        lines.set_path_effects([
            pe.Stroke(linewidth=3.6, foreground="white", alpha=.72), pe.Normal(),
        ])
        ax.add_collection(lines)
    if annotate:
        center = recovery_lines.mean((0, 1))
        text = "Recovered window" if recovery else "Window absent from native map"
        color = ORANGE if recovery else "#34445B"
        # Keep labels above the scene rather than across furniture.
        ax.annotate(
            text, xy=center, xytext=(width*.69, height*.055),
            fontsize=11.2, ha="center", va="center", color=color,
            weight="semibold", zorder=12,
            arrowprops={"arrowstyle": "-", "color": color, "lw": 1.3,
                        "shrinkA": 5, "shrinkB": 3},
            bbox={"facecolor": "white", "edgecolor": "none", "pad": 2.2,
                  "alpha": .94},
        )
        ax.scatter([center[0]], [center[1]], s=17, c=color,
                   edgecolors="white", linewidths=1.0, zorder=13)
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    ax.set_aspect("equal")
    ax.axis("off")


def save(fig, stem, *, transparent=False, dpi=320):
    for extension in ("png", "pdf", "svg"):
        fig.savefig(stem.with_suffix("."+extension), dpi=dpi,
                    transparent=transparent)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--include-bed", action="store_true",
                        help="Also show measured native bed record 15 as a fine dashed blue box.")
    args = parser.parse_args()
    output = args.output or ROOT / "figure_assets" / (
        "real_3d_motivation_with_bed_20261028" if args.include_bed
        else "real_3d_motivation_optimized_20261028"
    )
    output.mkdir(parents=True, exist_ok=True)
    renderer = load_renderer()
    with np.load(SOURCE / "measured_boxes_zup.npz", allow_pickle=False) as archive:
        values = {key: archive[key].copy() for key in archive.files}
    camera = json.loads((SOURCE / "fixed_camera.json").read_text())
    manifest = json.loads((SOURCE / "source_manifest.json").read_text())
    original_mesh = np.asarray(Image.open(SOURCE / "colored_mesh_perspective.png").convert("RGBA")).copy()
    assert tuple(camera["image_size"]) == (original_mesh.shape[1], original_mesh.shape[0])
    # Neutralize vertex colors only; preserve mesh silhouettes, holes and alpha.
    mesh = original_mesh.copy()
    luminance = original_mesh[..., :3] @ np.array([.2126, .7152, .0722]) / 255.
    gray = np.clip(.75*luminance+.19, 0., 1.)
    mesh[..., :3] = np.rint(gray[..., None]*255).astype(np.uint8)

    def segments(box):
        projection = renderer.project(
            box, np.asarray(camera["position"]), np.asarray(camera["look_at"]),
            camera["focal_pixels"], np.asarray(camera["principal_point_pixels"]),
        )[:, :2]
        return projection[renderer.box_edges(box)]

    native_lines = [segments(values["native_boxes"][index]) for index in NATIVE_INDICES]
    bed_lines = segments(values["native_boxes"][15]) if args.include_bed else None
    display_indices = NATIVE_INDICES + ([15] if args.include_bed else [])
    omitted_indices = [index for index in OMITTED_FROM_PREVIOUS if index not in display_indices]
    recovery_lines = segments(values["plr_recovered_box"])
    gt_lines = segments(values["ground_truth_box"])
    gt_array = np.load(ROOT / "evaluation/data_util/scannet_train_detection_data/scene0426_03_bbox.npy")
    gt_low = gt_array[:, :3]-gt_array[:, 3:6]/2
    gt_high = gt_array[:, :3]+gt_array[:, 3:6]/2
    selection_audit = []
    for index in display_indices + omitted_indices:
        box = values["native_boxes"][index]
        lo, hi = box.min(0), box.max(0)
        intersection = np.maximum(np.minimum(hi, gt_high)-np.maximum(lo, gt_low), 0).prod(1)
        overlap = intersection / np.maximum(np.prod(hi-lo)+np.prod(gt_high-gt_low, axis=1)-intersection, 1e-12)
        target_index = int(np.argmax(overlap))
        selection_audit.append({
            "native_index": index, "displayed": index in display_indices,
            "best_gt_index": target_index if overlap[target_index] > 0 else None,
            "max_aabb_iou": float(overlap[target_index]),
            "score": float(values["native_scores"][index]),
        })
    assert all(row["max_aabb_iou"] > .5 for row in selection_audit
               if row["native_index"] in NATIVE_INDICES)

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "pdf.fonttype": 42, "svg.fonttype": "none"})
    for neutral, background, color_suffix in (
        (True, mesh, ""), (False, original_mesh, "_color"),
    ):
        for recovery, name in ((False, "native_scene_3d"), (True, "recovered_scene_3d")):
            fig, ax = plt.subplots(figsize=(8.8, 7.2), dpi=300)
            fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
            draw_panel(ax, background, native_lines, recovery_lines, gt_lines,
                       recovery=recovery, bed_lines=bed_lines)
            save(fig, output / f"{name}{color_suffix}", transparent=True, dpi=300)
            if neutral:
                fig, ax = plt.subplots(figsize=(8.8, 7.2), dpi=300)
                fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
                draw_panel(ax, background, native_lines, recovery_lines, gt_lines,
                           recovery=recovery, show_gt=True, bed_lines=bed_lines)
                save(fig, output / f"{name}_with_gt", transparent=True, dpi=300)

    fig, axes = plt.subplots(1, 2, figsize=(13.6, 6.4))
    fig.subplots_adjust(left=.01, right=.99, bottom=.14, top=.84, wspace=.025)
    for ax, recovery in zip(axes, (False, True)):
        draw_panel(ax, mesh, native_lines, recovery_lines, gt_lines,
                   recovery=recovery, annotate=True, bed_lines=bed_lines)
    fig.text(.04, .965, "Object-map-only state", fontsize=19, weight="bold",
             color="#16283F", va="top")
    fig.text(.04, .915, "BoxFusion-Ctrl · selected native records", fontsize=10.5,
             color="#6B788B", va="top")
    fig.text(.535, .965, "ReCaR-3D", fontsize=19, weight="bold", color="#16283F", va="top")
    fig.text(.535, .915, "Same native records + selected PLR recovery", fontsize=10.5,
             color="#6B788B", va="top")
    legend_handles = [
        Line2D([0], [0], color=BLUE, lw=2, label="Selected native boxes"),
        Line2D([0], [0], color=ORANGE, lw=2.5, label="PLR recovery (3D AABB IoU = 0.58)"),
    ]
    if args.include_bed:
        legend_handles.insert(1, Line2D(
            [0], [0], color=BLUE, lw=1.4, ls=(0, (5, 3)),
            label="Native bed prediction (IoU = 0.32)",
        ))
    fig.legend(handles=legend_handles, loc="lower center",
               bbox_to_anchor=(.5, .057), frameon=False,
               ncol=len(legend_handles), fontsize=9.6 if args.include_bed else 10.5,
               columnspacing=1.6)
    fig.text(.5, .022,
             "Same scene, camera and native geometry. Selected predictions shown; mesh used only for visualization.",
             ha="center", fontsize=9, color="#6B788B")
    save(fig, output / "optimized_3d_comparison", dpi=320)

    np.savez_compressed(output / "displayed_boxes_and_camera_check.npz",
                        native_indices=np.asarray(display_indices),
                        native_boxes=values["native_boxes"][display_indices],
                        native_scores=values["native_scores"][display_indices],
                        plr_recovered_box=values["plr_recovered_box"],
                        ground_truth_box=values["ground_truth_box"])
    audit = {
        "schema": "recar3d.optimized_selected_3d_pair.v1",
        "source": str(SOURCE), "source_manifest_sha256": digest(SOURCE / "source_manifest.json"),
        "scene": manifest["scene"], "camera": camera,
        "same_camera_mesh_and_native_geometry": True,
        "displayed_native_indices": display_indices,
        "omitted_from_previous_illustration": omitted_indices,
        "coarse_bed_prediction_included": args.include_bed,
        "bed_line_style": "fine dashed blue; measured native record 15" if args.include_bed else None,
        "selection_is_for_qualitative_illustration_only": True,
        "selection_uses_gt_audit": True,
        "selection_reason": (
            "desk and two chairs with max GT AABB IoU > 0.5; bed retained as a separately labelled coarse prediction"
            if args.include_bed else "desk and two chairs with max GT AABB IoU > 0.5; reduce clutter"
        ),
        "selection_audit": selection_audit,
        "predicted_coordinates_modified": False,
        "detections_or_evaluation_modified": False,
        "plr_aabb_iou": manifest["plr_3d_aabb_iou"],
        "rendering": "neutral mesh default; original-color variant also exported; same silhouettes and alpha",
        "original_camera_file": str(SOURCE / "fixed_camera.json"),
        "original_mesh_file": str(SOURCE / "scene0426_03_aligned_cutaway.ply"),
        "source_hashes": {name: digest(SOURCE / name) for name in (
            "colored_mesh_perspective.png", "fixed_camera.json", "measured_boxes_zup.npz",
        )},
    }
    (output / "selection_audit.json").write_text(json.dumps(audit, indent=2)+"\n")
    (output / "图注与素材说明.txt").write_text(
        "优化版立体场景对照\n\n"
        "native_scene_3d.*：左侧透明底场景，展示选定原生预测。\n"
        "recovered_scene_3d.*：同视角、同原生框，增加真实PLR窗户恢复框。\n"
        "*_color.*：保留原扫描颜色的版本。默认版本仅将网格转为中性灰，以突出蓝框和橙框。\n"
        "*_with_gt.*：另外叠加所选窗户的绿色虚线GT，便于检查。\n"
        "optimized_3d_comparison.*：含标题和图例的并排成图。\n\n"
        "基于此前GT核对，展示索引5（书桌，IoU0.7423）、3（椅子，0.8555）、16（椅子，0.6309）。"
        + ("另将索引15的真实床预测加回两个面板；其IoU为0.3171，采用细蓝虚线并在图例中注明。"
           "索引0、17仍隐藏。" if args.include_bed else
           "此前索引0无GT匹配，索引15/17定位较粗，现仅从本图隐藏。")
        + "此筛选只用于定性示意，并未修改预测文件、模型或评测。两个面板始终展示相同原生记录。"
        "所有框坐标、网格几何、相机和剖切范围保持原值，未手动移动或缩紧预测框。\n\n"
        "建议图注：同一真实场景中的对象地图与PLR恢复示例。"
        "蓝色展示选定的原生预测，橙色表示PLR恢复的窗户；左右原生框的几何相同。"
        "为提高可读性，仅展示部分预测，不代表完整输出集合。"
        "场景网格仅用于可视化，不作为模型输入。\n"
        + ("蓝色细虚线表示原生床预测，定位较粗，IoU约0.32，不是真实标注框。\n"
           if args.include_bed else ""),
        encoding="utf-8")
    with zipfile.ZipFile(output.with_suffix(".zip"), "w", zipfile.ZIP_DEFLATED) as handle:
        for path in sorted(output.rglob("*")):
            if path.is_file():
                handle.write(path, path.relative_to(output.parent))
    print(json.dumps({"output": str(output), "native_indices": display_indices,
                      "selection_audit": selection_audit}, indent=2))


if __name__ == "__main__":
    main()
