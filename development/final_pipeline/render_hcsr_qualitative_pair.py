#!/usr/bin/env python3
"""Render verified HCSR cases as a compact main-paper pair, without inference."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import numpy as np
from PIL import Image

import render_final_qualitative as source


ROOT = Path(__file__).resolve().parents[2]
LOCKED = ROOT / "development/final_pipeline/reports/qualitative_recar3d_final"
OUTPUT = ROOT / "figure_assets/hcsr_qualitative_pair_20261031"
INK = "#172033"
MUTED = "#64748B"


def export(fig, output: Path, name: str) -> list[Path]:
    paths = []
    for extension in ("png", "pdf", "svg", "tiff"):
        path = output / f"{name}.{extension}"
        options = {"dpi": 600} if extension in {"png", "tiff"} else {}
        if extension == "tiff":
            options["pil_kwargs"] = {"compression": "tiff_lzw"}
        fig.savefig(path, facecolor="white", **options)
        paths.append(path)
    preview = output / f"{name}_preview.png"
    fig.savefig(preview, facecolor="white", dpi=180)
    paths.append(preview)
    plt.close(fig)
    return paths


def draw_rgb(ax, case: dict, evidence: dict, args, *, failure: bool) -> int:
    row = evidence["frames"][0]
    frame = int(row["frame_id"])
    image = Image.open(row["rgb_path"]).convert("RGB").resize(
        (640, 480), Image.Resampling.LANCZOS
    )
    candidate = row.get("candidate_xyxy")
    if candidate is None:
        candidate = source.project_box(
            row["candidate_box"], args.raw_root, case["scene"], frame
        )
    native = source.project_box(
        np.asarray(case["box"]), args.raw_root, case["scene"], frame
    )
    focus = candidate if candidate is not None else native
    if candidate is not None and native is not None:
        focus = [
            min(candidate[0], native[0]), min(candidate[1], native[1]),
            max(candidate[2], native[2]), max(candidate[3], native[3]),
        ]
    image, boxes = source.crop_evidence_frame(
        image, focus, {"candidate": candidate, "native": native}, scale=1.85
    )
    ax.imshow(image)
    source.draw_rect(
        ax, boxes["native"],
        source.COLORS["failure"] if failure else source.COLORS["native"],
        width=2.2,
    )
    source.draw_rect(
        ax, boxes["candidate"], source.COLORS["support"], dashed=True, width=1.5
    )
    ax.set_xlim(0, 640)
    ax.set_ylim(480, 0)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color("#CBD5E1")
        spine.set_linewidth(0.55)
    ax.set_title(
        f"{case['scene']} · t={frame}", loc="left", fontsize=6.4, pad=3
    )
    return frame


def metric_card(fig, left: float, width: float, case: dict) -> None:
    ax = fig.add_axes([left, 0.075, width, 0.185])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.add_patch(Rectangle(
        (0, 0), 1, 1, facecolor="#F6F8FB", edgecolor="#DCE3EC", linewidth=0.6
    ))
    ax.text(0.045, 0.80, "Native → HCSR", fontsize=6.4, color=MUTED, va="center")
    ax.text(0.045, 0.51, "Score", fontsize=6.6, color=MUTED, va="center")
    ax.text(0.58, 0.51, "Native-record rank", fontsize=6.6, color=MUTED, va="center")
    ax.text(
        0.045, 0.19,
        f"{case['original_score']:.3f} → {case['reranked_score']:.3f}",
        fontsize=8.6, color=INK, va="center", fontweight="semibold",
    )
    ax.text(
        0.58, 0.19, f"{case['original_rank']} → {case['reranked_rank']}",
        fontsize=8.6, color=INK, va="center", fontweight="semibold",
    )


def render_main(payload: dict, evidence: dict, args) -> list[Path]:
    width_inches, height_inches = 183 / 25.4, 2.70
    fig = plt.figure(figsize=(width_inches, height_inches))
    legend = [
        Line2D([], [], color=source.COLORS["native"], lw=1.9, label="Native box"),
        Line2D([], [], color=source.COLORS["gt"], lw=1.8, ls="--", label="Ground truth"),
        Line2D([], [], color=source.COLORS["support"], lw=1.6, ls="--", label="Matched proposal"),
        Line2D([], [], color=source.COLORS["failure"], lw=1.9, label="False positive"),
    ]
    fig.legend(
        handles=legend, loc="upper center", bbox_to_anchor=(0.5, 1.0),
        ncol=4, frameon=False, fontsize=6.8, handlelength=1.9,
        columnspacing=1.6,
    )
    for index, panel in enumerate(("mvsr_reranking", "failure")):
        case = payload["cases"][panel]
        left = 0.035 if index == 0 else 0.535
        failure = panel == "failure"
        title = "a  Native-record reranking" if index == 0 else "b  High-support false positive"
        fig.text(left, 0.86, title, fontsize=8.4, fontweight="bold", color=INK)
        image_y, image_height = 0.325, 1.20 / height_inches
        rgb = fig.add_axes([left, image_y, 1.60 / width_inches, image_height])
        draw_rgb(rgb, case, evidence[panel], args, failure=failure)
        geometry = fig.add_axes([
            left + 1.71 / width_inches, image_y, 1.20 / width_inches, image_height
        ])
        source.topdown(
            geometry, case, args.components, args.gt_root, args.raw_root,
            failure=failure, crop_scale=1.72 if failure else 1.55,
            min_margin=0.84 if failure else 0.72,
            point_darken=0.52 if failure else 0.72,
            point_alpha=0.82 if failure else 0.52,
            point_size=0.62 if failure else 0.28,
            monochrome_points=failure,
            background_color="#EEF2F6" if failure else "#F8FAFC",
        )
        geometry.set_title("Shared geometry", loc="left", fontsize=6.7, pad=3)
        note = (
            "No GT match at IoU threshold 0.15"
            if failure else "Same native box before and after HCSR"
        )
        fig.text(
            left, 0.282, note, fontsize=6.6,
            color=source.COLORS["failure"] if failure else MUTED,
        )
        metric_card(fig, left, 2.91 / width_inches, case)
    fig.text(
        0.5, 0.022,
        "Native geometry unchanged · Rank computed within each scene's native records",
        fontsize=6.4, color=MUTED, ha="center",
    )
    return export(fig, args.output, "hcsr_qualitative_main_pair")


def render_supplement(payload: dict, evidence: dict, args) -> list[Path]:
    fig = plt.figure(figsize=(183 / 25.4, 3.05))
    grid = fig.add_gridspec(
        1, 2, left=0.035, right=0.98, bottom=0.04, top=0.88, wspace=0.10
    )
    for index, panel in enumerate(("plr_recovery", "calr_recovery")):
        source.comparison_panel(
            fig, grid[0, index], panel, payload["cases"][panel], evidence[panel], args
        )
    legend = [
        Line2D([], [], color=source.COLORS["native"], lw=1.8, label="Native box"),
        Line2D([], [], color=source.COLORS["plr"], lw=1.8, label="PLR evidence/recovery"),
        Line2D([], [], color=source.COLORS["calr"], lw=1.8, label="CALR evidence/recovery"),
        Line2D([], [], color=source.COLORS["gt"], lw=1.8, ls="--", label="Ground truth"),
    ]
    fig.legend(
        handles=legend, loc="upper center", bbox_to_anchor=(0.5, 0.99),
        ncol=4, frameon=False, fontsize=6.6, handlelength=1.9, columnspacing=1.1,
    )
    return export(fig, args.output, "recovery_qualitative_supplement_pair")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, default=LOCKED / "selected_cases.json")
    parser.add_argument(
        "--components", type=Path,
        default=ROOT / "development/final_controls/results/scannet_recar3d_key_control_components",
    )
    parser.add_argument(
        "--gt-root", type=Path, default=ROOT / "evaluation/data_util/scannet_train_detection_data"
    )
    parser.add_argument("--raw-root", type=Path, default=Path("/extra/ZhaoX/scannet_data/scans"))
    parser.add_argument("--mvsr-no-visibility", type=Path, default=source.DEFAULT_NO_VISIBILITY_COMPONENTS)
    parser.add_argument(
        "--reference-metrics", type=Path,
        default=ROOT / "figure_assets/qualitative_fig5_threshold_label_20261029/mvsr_metrics_no_visibility.json",
    )
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    payload = copy.deepcopy(json.loads(args.selection.read_text()))
    evidence, hashes = {}, {}
    for panel, case in payload["cases"].items():
        path = Path(case["evidence_export"])
        if not case.get("evidence_verified") or source.digest(path) != case["evidence_sha256"]:
            raise RuntimeError(f"Unverified evidence: {panel}")
        evidence[panel] = json.loads(path.read_text())
        hashes[str(path)] = source.digest(path)
        for row in evidence[panel]["frames"]:
            rgb_path = Path(row["rgb_path"])
            hashes[str(rgb_path)] = source.digest(rgb_path)
        for geometry_path in (
            args.gt_root / f"{case['scene']}_vert.npy",
            args.raw_root / case["scene"] / f"{case['scene']}.txt",
            args.components / "native_native" / f"{case['scene']}_boxes.pkl",
        ):
            hashes[str(geometry_path)] = source.digest(geometry_path)
    reference = json.loads(args.reference_metrics.read_text())["cases"]
    if float(payload["cases"]["failure"]["iou"]) >= 0.15:
        raise RuntimeError("Failure case does not satisfy the stated IoU threshold")
    metrics = {}
    for panel in ("mvsr_reranking", "failure"):
        row = source.final_mvsr_metrics(
            payload["cases"][panel], args.mvsr_no_visibility.parent / "native_native",
            args.mvsr_no_visibility,
        )
        keys = ("original_score", "reranked_score", "original_rank", "reranked_rank")
        for key in (*keys, "native_sha256", "reranked_sha256"):
            if row[key] != reference[panel][key]:
                raise RuntimeError(f"Changed reference metric: {panel} / {key}")
        payload["cases"][panel].update({key: row[key] for key in keys})
        metrics[panel] = {key: value for key, value in row.items() if key != "max_three_factor_support"}
    args.output.mkdir(parents=True, exist_ok=True)
    with plt.rc_context({
        "font.family": "DejaVu Sans", "font.size": 7, "text.color": INK,
        "axes.linewidth": 0.55, "pdf.fonttype": 42, "ps.fonttype": 42,
        "svg.fonttype": "none", "savefig.facecolor": "white",
    }):
        files = render_main(payload, evidence, args)
        files += render_supplement(payload, evidence, args)
    zh = (
        "图5｜HCSR的重排序示例与失败案例。a，原生框几何保持不变，分数由0.506提高至0.627，"
        "场景内原生记录排名由51提升至32。b，一个在三维AABB IoU阈值0.15下未匹配真实框的预测，"
        "其分数由0.586提高至0.693，排名由45提升至35。两例均展示同一原生框在分数更新前后的排序变化，"
        "每例仅绘制一张共享几何视图。蓝色表示原生框，绿色虚线表示真实标注，紫色虚线表示所示RGB帧中匹配的"
        "辅助proposal，红色表示失败案例中的误检框。所示RGB帧不用于指示历史最大支持的发生时刻。"
        "场景几何仅用于可视化。"
    )
    en = (
        "Figure 5. HCSR reranking and a failure case. (a) With native geometry unchanged, the score "
        "increases from 0.506 to 0.627 and the within-scene native-record rank improves from 51 to 32. "
        "(b) A prediction without a ground-truth match at the 3D AABB IoU threshold of 0.15 is also "
        "promoted: its score increases from 0.586 to 0.693 and its native-record rank changes from 45 "
        "to 35. Each example uses one shared geometry view of the same native box before and after "
        "score updating. Blue denotes the native box, dashed green ground truth, dashed purple the "
        "matched auxiliary proposal in the displayed RGB frame, and red the false positive. The "
        "displayed RGB frame does not identify when the historical maximum was attained. Scene "
        "geometry is used only for visualization."
    )
    supp_zh = (
        "补充图｜PLR与CALR的候选恢复示例。a，PLR案例（scene0426_03）展示关键帧425、450、550的"
        "辅助proposal证据，以及BoxFusion-Ctrl与ReCaR-3D的同视角几何对照；选中恢复框与真实框的"
        "三维AABB IoU为0.58。b，CALR案例（scene0606_01）展示关键帧1400、1575、1600的anchor证据"
        "及相同形式的几何对照，三维AABB IoU为0.86。两面板来自不同场景，不构成同一目标的两种恢复过程。"
        "蓝色表示原生框，橙色表示PLR证据或恢复框，深金色表示CALR证据或恢复框，绿色虚线表示真实标注。"
        "点云仅用于可视化。"
    )
    for name, caption in (
        ("caption_main_zh.txt", zh), ("caption_main_en.txt", en),
        ("caption_supplement_zh.txt", supp_zh),
    ):
        (args.output / name).write_text(caption + "\n", encoding="utf-8")
    manifest = {
        "schema": "recar3d.hcsr_qualitative_pair.v1",
        "selection": str(args.selection), "selection_sha256": source.digest(args.selection),
        "renderer_sha256": source.digest(Path(__file__)),
        "reference_metrics": str(args.reference_metrics),
        "reference_metrics_sha256": source.digest(args.reference_metrics),
        "cases": metrics, "input_sha256": hashes,
        "changes": [
            "Move PLR/CALR to a separate supplementary figure",
            "Relabel the HCSR cases a/b; one RGB and one shared geometry view per case",
            "Omit historical support values and approximation footnotes",
            "Retain reference scores/ranks and specify the IoU evaluation threshold",
        ],
        "learning_models_executed": False,
        "native_labels_and_geometry_equal_before_after": True,
        "display_metrics_match_reference_exactly": True,
        "artifacts_sha256": {path.name: source.digest(path) for path in files},
    }
    (args.output / "source_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"output": str(args.output), "artifacts": [p.name for p in files]}))


if __name__ == "__main__":
    main()
