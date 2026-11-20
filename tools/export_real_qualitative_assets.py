#!/usr/bin/env python3
"""Export measured RGB/point-cloud assets for replacing the concept drawing."""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import zipfile

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "development/final_pipeline/reports/qualitative_recar3d_final"
COMPONENTS = ROOT / "development/final_controls/results/scannet_recar3d_key_control_components"
FINAL_COMPONENTS = ROOT / "development/final_controls/experiments/mvsr_design_20261005/components"
RAW = Path("/extra/ZhaoX/scannet_data/scans")
GT = ROOT / "evaluation/data_util/scannet_train_detection_data"


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for part in iter(lambda: handle.read(1 << 20), b""):
            value.update(part)
    return value.hexdigest()


def load_renderer():
    path = ROOT / "development/final_pipeline/render_final_qualitative.py"
    spec = importlib.util.spec_from_file_location("qualitative_asset_renderer", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def save_figure(fig, path, dpi=360):
    fig.savefig(path.with_suffix(".png"), dpi=dpi, facecolor="white")
    fig.savefig(path.with_suffix(".pdf"), facecolor="white")
    fig.savefig(path.with_suffix(".svg"), facecolor="white")
    plt.close(fig)


def draw_rgb(ax, renderer, case, frame, *, native=False, full_frame=False):
    image = Image.open(frame["rgb_path"]).convert("RGB")
    if full_frame:
        sx, sy = image.width / 640, image.height / 480
        box = np.asarray(frame["candidate_xyxy"], dtype=float) * [sx, sy, sx, sy]
        ax.imshow(image)
        renderer.draw_rect(ax, box, renderer.COLORS["plr"], width=2.4)
        ax.set_xlim(0, image.width)
        ax.set_ylim(image.height, 0)
        ax.axis("off")
    else:
        renderer.rgb_strip(
            ax, case, {"frames": [frame]}, RAW, max_frames=1
        )
    ax.set_title(f"Keyframe {frame['frame_id']}", fontsize=11, pad=5)


def draw_map(ax, renderer, case, *, recovery):
    renderer.topdown(
        ax, case, COMPONENTS, GT, RAW,
        show_selected=recovery, crop_scale=1.25, min_margin=.65,
        point_darken=.65, point_alpha=.68, point_size=.38,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "figure_assets/real_qualitative_replacement_20261026",
    )
    args = parser.parse_args()
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    original_dir = output / "original_rgb"
    original_dir.mkdir(exist_ok=True)
    renderer = load_renderer()
    selection = json.loads((SOURCE / "selected_cases.json").read_text())
    parity = json.loads((SOURCE / "terminal_parity.json").read_text())
    if not parity["all_selected_cases_passed"]:
        raise RuntimeError("source selected-case parity audit failed")
    cases, evidence = {}, {}
    for panel in ("plr_recovery", "mvsr_reranking"):
        case = copy.deepcopy(selection["cases"][panel])
        path = Path(case["evidence_export"])
        if not case["evidence_verified"] or digest(path) != case["evidence_sha256"]:
            raise RuntimeError(f"unverified source evidence: {panel}")
        cases[panel] = case
        evidence[panel] = json.loads(path.read_text())

    plr = cases["plr_recovery"]
    frames = evidence["plr_recovery"]["frames"]
    if [row["frame_id"] for row in frames] != [425, 450, 550]:
        raise RuntimeError("unexpected PLR supporting frames")
    hcsr = cases["mvsr_reranking"]
    metrics = renderer.final_mvsr_metrics(
        hcsr, FINAL_COMPONENTS / "native_native",
        FINAL_COMPONENTS / "native_factor_no_visibility",
    )
    sources = {}
    for panel, rows in evidence.items():
        for row in rows["frames"]:
            original = Path(row["rgb_path"])
            filename = f"{cases[panel]['scene']}_frame_{row['frame_id']:06d}.jpg"
            destination = original_dir / filename
            shutil.copy2(original, destination)
            sources[str(original)] = {
                "sha256": digest(original), "exported_original": str(destination),
                "dimensions": list(Image.open(original).size),
            }

    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10,
        "pdf.fonttype": 42, "svg.fonttype": "none",
    })
    # Full original views with measured post-NMS 2D candidate rectangles.
    for row in frames:
        fig, ax = plt.subplots(figsize=(6.48, 5.12))
        fig.subplots_adjust(left=0, right=1, bottom=0, top=.945)
        draw_rgb(ax, renderer, plr, row, full_frame=True)
        save_figure(fig, output / f"plr_frame_{row['frame_id']:06d}_proposal", dpi=240)

    # A consistent crop scale makes the same target recognizable across views.
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.3))
    fig.subplots_adjust(left=.01, right=.99, bottom=.03, top=.90, wspace=.035)
    for ax, row in zip(axes, frames):
        draw_rgb(ax, renderer, plr, row)
    save_figure(fig, output / "plr_three_keyframes")

    for recovery, name, title in (
        (False, "plr_native_map", "BoxFusion-Ctrl: selected target missed"),
        (True, "plr_recovered_map", "ReCaR-3D: PLR recovery"),
    ):
        fig, ax = plt.subplots(figsize=(5, 5))
        fig.subplots_adjust(left=.025, right=.975, bottom=.025, top=.92)
        draw_map(ax, renderer, plr, recovery=recovery)
        ax.set_title(title, fontsize=12, pad=8)
        save_figure(fig, output / name)

    handles = [
        Line2D([0], [0], color=renderer.COLORS["native"], lw=2, label="Native box"),
        Line2D([0], [0], color=renderer.COLORS["plr"], lw=2, label="PLR evidence / recovery"),
        Line2D([0], [0], color=renderer.COLORS["gt"], lw=2, ls="--", label="Ground truth"),
    ]
    fig = plt.figure(figsize=(11.8, 8.1))
    fig.text(.04, .97, "Repeated proposal evidence and PLR recovery", fontsize=19,
             weight="bold", va="top", color="#12213A")
    fig.text(.04, .928, f"ScanNet {plr['scene']}  |  same target throughout", fontsize=11,
             color="#475569")
    grid = fig.add_gridspec(2, 6, left=.04, right=.96, bottom=.105, top=.875,
                           height_ratios=(.82, 1.42), hspace=.20, wspace=.22)
    for index, row in enumerate(frames):
        ax = fig.add_subplot(grid[0, index * 2:index * 2 + 2])
        draw_rgb(ax, renderer, plr, row)
    left, right = fig.add_subplot(grid[1, :3]), fig.add_subplot(grid[1, 3:])
    draw_map(left, renderer, plr, recovery=False)
    draw_map(right, renderer, plr, recovery=True)
    left.set_title("BoxFusion-Ctrl", weight="bold", pad=8, fontsize=14)
    right.set_title("ReCaR-3D", weight="bold", pad=8, fontsize=14)
    right.text(.5, .035, f"3D AABB IoU = {plr['iou']:.2f}", transform=right.transAxes,
               ha="center", va="bottom", fontsize=11,
               bbox={"facecolor": "white", "edgecolor": "none", "alpha": .9})
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.5, .052),
               ncol=3, frameon=False, fontsize=10)
    fig.text(.5, .023, "Point cloud: visualization only. Three of five final-track evidence keyframes shown.",
             ha="center", fontsize=9, color="#64748B")
    save_figure(fig, output / "plr_real_case_overview")

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.1),
                             gridspec_kw={"width_ratios": [1.15, 1]})
    fig.subplots_adjust(left=.03, right=.97, bottom=.09, top=.81, wspace=.12)
    renderer.rgb_strip(axes[0], hcsr, evidence["mvsr_reranking"], RAW, max_frames=1)
    axes[0].set_title(f"{hcsr['scene']}  |  keyframe 1675", fontsize=10, pad=6)
    axes[1].axis("off")
    text = (
        f"Native score: {metrics['original_score']:.3f}  →  {metrics['reranked_score']:.3f}\n\n"
        f"Native-record rank: {metrics['original_rank']}  →  {metrics['reranked_rank']}\n\n"
        f"Historical max support: {metrics['max_three_factor_support']:.3f}\n\n"
        "Native geometry unchanged"
    )
    axes[1].text(.04, .55, text, transform=axes[1].transAxes, fontsize=12,
                 va="center", linespacing=1.35, color="#12213A")
    fig.suptitle("HCSR: measured score and rank change", fontsize=17, weight="bold", y=.98)
    fig.text(.5, .035, "Separate scene from the PLR example. Blue: native projection; purple dashed: matched proposal.",
             ha="center", fontsize=9, color="#64748B")
    save_figure(fig, output / "hcsr_real_case")

    # Reusable map pair independent of the three-frame strip.
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.9))
    fig.subplots_adjust(left=.025, right=.975, bottom=.075, top=.92, wspace=.08)
    for ax, recovery, title in zip(axes, (False, True), ("BoxFusion-Ctrl", "ReCaR-3D")):
        draw_map(ax, renderer, plr, recovery=recovery)
        ax.set_title(title, fontsize=13, weight="bold", pad=7)
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False,
               bbox_to_anchor=(.5, 0), fontsize=9)
    save_figure(fig, output / "plr_native_vs_recovered")

    # Preserve trace provenance and the distinction between final HCSR metrics
    # and older qualitative matching evidence, without synthesizing new scenes.
    manifest = {
        "schema": "recar3d.real_qualitative_replacement.v1",
        "asset_origin": "original RGB, measured predicted boxes, ScanNet visualization vertices",
        "generative_images_used": False,
        "new_model_inference": False,
        "plr": {
            "scene": plr["scene"], "shown_frames": [425, 450, 550],
            "final_track_support_frames": evidence["plr_recovery"]["terminal_trace"]["evidence_frame_ids"],
            "confirmation_frame": 550,
            "recovered_3d_aabb_iou": plr["iou"],
            "native_max_gt_iou": plr["native_gt_iou"],
            "maps": "local top-down projections of actual 3D scene points; selected target only emphasized",
            "native_predictions": str(COMPONENTS / "native_native" / f"{plr['scene']}_boxes.pkl"),
            "evidence": str(SOURCE / "plr_recovery_evidence.json"),
        },
        "hcsr": {
            **metrics,
            "factors": ["proposal score", "2D overlap", "3D consistency"],
            "visibility": False, "aggregation": "historical maximum",
            "rank_scope": "within-scene fixed native records",
            "separate_scene_from_plr": True,
            "rgb_matching_evidence": str(SOURCE / "mvsr_reranking_evidence.json"),
            "historical_support_source": "inverted from verified final score transform",
        },
        "original_rgb_sources": sources,
        "selected_case_audit": str(SOURCE / "terminal_parity.json"),
        "selected_case_audit_sha256": digest(SOURCE / "terminal_parity.json"),
        "selection_sha256": digest(SOURCE / "selected_cases.json"),
        "point_cloud_is_model_input": False,
    }
    (output / "source_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    note = (
        "真实素材说明\n\n"
        "plr_real_case_overview.png / .pdf / .svg：同一PLR真实案例的三关键帧及原生遗漏/恢复对照。\n"
        "plr_three_keyframes.*：425、450、550帧局部视图；橙框来自缓存中的二维proposal。\n"
        "plr_frame_000425_proposal.*（另含450、550）：完整原始视图及对应辅助框。\n"
        "plr_native_map.* / plr_recovered_map.* / plr_native_vs_recovered.*：相同坐标、相同点云的终态局部俯视图。\n"
        "hcsr_real_case.*：独立场景的真实HCSR案例，分数0.506→0.627，排名51→32，历史最大支持0.748。\n"
        "original_rgb/：原始JPG原样复制，未重绘、未去噪、未作超分辨率。\n\n"
        "PLR场景为scene0426_03。最终轨迹共5个证据关键帧，图中展示425、450、550三个；"
        "恢复框与GT的3D AABB IoU为0.575533。图中恢复框为终态代表记录，不表示550帧时的几何。\n"
        "点云只用于可视化；俯视矩形是三维框在水平面的投影，不是三维立方体效果图。"
        "对照仅突出所选目标，不展示完整系统所有CALR/PLR输出。\n"
        "HCSR场景为scene0644_00，应作为单独插图，不应与PLR场景拼成同一房间。"
        "HCSR图中的分数和排名由无visibility最终组件重新读取；匹配RGB来自已核验的定性证据。\n"
        "本素材包采用科学绘图方式叠加测量框，未使用生成式图像，也未重跑模型。\n\n"
        "建议图注：同一真实场景中的跨关键帧候选证据与PLR恢复结果。"
        "上方展示425、450和550三个关键帧中的辅助proposal，下方展示BoxFusion-Ctrl与ReCaR-3D的终态局部点云结果。"
        "橙色表示PLR候选或恢复框，绿色虚线表示GT，蓝色表示原生框。所选恢复框的3D AABB IoU为0.58。"
        "最终轨迹具有5个证据关键帧，图中展示其中3个。场景点云仅用于可视化。\n"
    )
    (output / "素材说明.txt").write_text(note, encoding="utf-8")
    archive = output.with_suffix(".zip")
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as handle:
        for path in sorted(output.rglob("*")):
            if path.is_file():
                handle.write(path, path.relative_to(output.parent))
    print(json.dumps({"output": str(output), "zip": str(archive),
                      "plr_scene": plr["scene"], "hcsr": metrics}, indent=2))


if __name__ == "__main__":
    main()
