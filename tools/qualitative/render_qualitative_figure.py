#!/usr/bin/env python3
"""Render the four-panel qualitative figure after evidence parity is verified."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from select_qualitative_cases import (  # noqa: E402
    DEFAULT_OUTPUT, FACTORIAL, GT_ROOT, aligned_boxes, axis_alignment, read_rows,
)


COLORS = {
    "native": "#1677FF",
    "plr": "#FF7A1A",
    "calr": "#F2C300",
    "gt": "#14B866",
    "failure": "#D62728",
}
TITLES = {
    "plr_recovery": "(a) PLR recovery",
    "calr_recovery": "(b) CALR recovery",
    "mvsr_reranking": "(c) MVSR reranking",
    "failure": "(d) Failure cases",
}


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            value.update(chunk)
    return value.hexdigest()


def _validate_evidence(case: dict[str, Any], locked_run: str) -> dict[str, Any]:
    if case.get("evidence_verified") is not True:
        raise RuntimeError(f"{case['panel']}: evidence_verified is not true")
    path = Path(case.get("evidence_export") or "")
    if not path.is_file():
        raise RuntimeError(f"{case['panel']}: missing evidence export: {path}")
    expected = case.get("evidence_sha256")
    if not expected or sha256(path) != expected:
        raise RuntimeError(f"{case['panel']}: evidence hash mismatch")
    data = json.loads(path.read_text())
    if data.get("locked_run") != locked_run or data.get("scene") != case["scene"]:
        raise RuntimeError(f"{case['panel']}: evidence provenance mismatch")
    parity = data.get("terminal_parity", {})
    if parity.get("base") is not True or parity.get("full") is not True:
        raise RuntimeError(f"{case['panel']}: terminal prediction parity was not proven")
    if data.get("future_frames_used") is not False:
        raise RuntimeError(f"{case['panel']}: non-causal evidence export")
    frames = data.get("frames", [])
    if case["panel"] in ("plr_recovery", "calr_recovery"):
        distinct = {int(row["frame_id"]) for row in frames}
        if len(distinct) < 3:
            raise RuntimeError(f"{case['panel']}: fewer than three supporting keyframes")
    if case["panel"] == "mvsr_reranking" and data.get("max_geometric_support") is None:
        raise RuntimeError("MVSR evidence lacks max geometric support")
    return data


def _draw_box(ax, xyxy, color, dashed=False, width=2.0):
    x0, y0, x1, y1 = map(float, xyxy)
    ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False,
                           edgecolor=color, linewidth=width,
                           linestyle="--" if dashed else "-"))


def _rgb_strip(ax, evidence: dict[str, Any], panel: str) -> None:
    frames = evidence.get("frames", [])[:3]
    if not frames:
        ax.text(0.5, 0.5, "No RGB evidence", ha="center", va="center")
        ax.axis("off")
        return
    images = []
    for row in frames:
        path = Path(row["rgb_path"])
        if not path.is_file():
            raise RuntimeError(f"missing RGB frame: {path}")
        image = np.asarray(Image.open(path).convert("RGB"))
        images.append((image, row))
    target_h = min(image.shape[0] for image, _ in images)
    rendered = []
    offsets = []
    offset = 0
    for image, row in images:
        if image.shape[0] != target_h:
            scale = target_h / image.shape[0]
            image = np.asarray(Image.fromarray(image).resize(
                (round(image.shape[1] * scale), target_h), Image.Resampling.LANCZOS))
        rendered.append(image)
        offsets.append((offset, image.shape[1], row))
        offset += image.shape[1]
    canvas = np.concatenate(rendered, axis=1)
    ax.imshow(canvas)
    for xoff, _, row in offsets:
        source = row.get("source", "native")
        if row.get("box_xyxy") is not None:
            box = list(map(float, row["box_xyxy"]))
            box[0] += xoff; box[2] += xoff
            _draw_box(ax, box, COLORS.get(source, COLORS["failure"]),
                      dashed=False, width=1.7)
        if row.get("gt_xyxy") is not None:
            box = list(map(float, row["gt_xyxy"]))
            box[0] += xoff; box[2] += xoff
            _draw_box(ax, box, COLORS["gt"], dashed=True, width=1.5)
        ax.text(xoff + 5, 14, f"t={row['frame_id']}", color="white", fontsize=7,
                bbox={"facecolor": "black", "alpha": 0.55, "pad": 1, "edgecolor": "none"})
    ax.set_xlim(0, canvas.shape[1]); ax.set_ylim(canvas.shape[0], 0); ax.axis("off")


def _rectangle_from_corners(ax, corners, color, dashed=False, width=1.5, alpha=1.0):
    corners = np.asarray(corners, dtype=float)
    lo, hi = corners[:, :2].min(0), corners[:, :2].max(0)
    ax.add_patch(Rectangle(lo, *(hi - lo), fill=False, edgecolor=color,
                           linewidth=width, linestyle="--" if dashed else "-",
                           alpha=alpha))


def _topdown(ax, case: dict[str, Any], panel: str) -> None:
    scene = case["scene"]
    vertices = np.load(GT_ROOT / f"{scene}_vert.npy")
    if len(vertices) > 30000:
        step = max(1, len(vertices) // 30000)
        vertices = vertices[::step]
    color = vertices[:, 3:6] / 255.0 if vertices.shape[1] >= 6 else np.full((len(vertices), 3), .75)
    ax.scatter(vertices[:, 0], vertices[:, 1], c=color, s=0.18, alpha=0.32,
               linewidths=0, rasterized=True)
    align = axis_alignment(scene)
    native = aligned_boxes(read_rows(FACTORIAL / "base" / f"{scene}_boxes.pkl"), align)
    for box in native:
        _rectangle_from_corners(ax, box, COLORS["native"], width=.75, alpha=.42)
    selected = np.asarray(case["box_aligned"], dtype=float)
    if panel == "plr_recovery":
        _rectangle_from_corners(ax, selected, COLORS["plr"], width=2.3)
    elif panel == "calr_recovery":
        _rectangle_from_corners(ax, selected, COLORS["calr"], width=2.3)
    elif panel == "failure":
        _rectangle_from_corners(ax, selected, COLORS["failure"], width=2.5)
    else:
        _rectangle_from_corners(ax, selected, COLORS["native"], width=2.5)
    if case.get("gt_box_aligned") is not None:
        _rectangle_from_corners(ax, case["gt_box_aligned"], COLORS["gt"],
                                dashed=True, width=2.0)
    lo, hi = selected[:, :2].min(0), selected[:, :2].max(0)
    margin = max(float((hi - lo).max()) * 2.5, 1.2)
    center = (lo + hi) / 2
    ax.set_xlim(center[0] - margin, center[0] + margin)
    ax.set_ylim(center[1] - margin, center[1] + margin)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    ax.set_facecolor("#F8FAFC")


def _annotation(case: dict[str, Any], evidence: dict[str, Any], panel: str) -> str:
    if panel == "mvsr_reranking":
        support = float(evidence["max_geometric_support"])
        return (f"score {case['original_score']:.3f} → {case['reranked_score']:.3f}\n"
                f"rank {case['original_rank']} → {case['reranked_rank']}\n"
                f"max support {support:.3f}")
    if panel == "plr_recovery":
        return f"PLR IoU={case['iou']:.2f} · {len(evidence['frames'])} keyframes"
    if panel == "calr_recovery":
        voxel = evidence.get("voxel_key", "recorded")
        return f"CALR IoU={case['iou']:.2f} · voxel {voxel}"
    return case.get("failure_reason", case.get("failure_type", "Observed failure"))


def render(cases_path: Path, output: Path) -> None:
    payload = json.loads(cases_path.read_text())
    if payload.get("locked_run") != "strict_causal_nochild_official100_v2_20260922":
        raise RuntimeError("selected cases do not belong to the locked original run")
    panels = ("plr_recovery", "calr_recovery", "mvsr_reranking", "failure")
    evidence = {panel: _validate_evidence(payload["cases"][panel], payload["locked_run"])
                for panel in panels}
    output.mkdir(parents=True, exist_ok=True)

    individual = output / "cases"
    individual.mkdir(exist_ok=True)
    for panel in panels:
        fig = plt.figure(figsize=(7.2, 3.4), constrained_layout=True)
        grid = fig.add_gridspec(1, 2, width_ratios=(1.45, 1.0))
        left, right = fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1])
        _rgb_strip(left, evidence[panel], panel)
        _topdown(right, payload["cases"][panel], panel)
        fig.suptitle(TITLES[panel] + " — " + payload["cases"][panel]["scene"],
                     fontsize=11, fontweight="bold")
        right.set_title(_annotation(payload["cases"][panel], evidence[panel], panel), fontsize=8)
        fig.savefig(individual / f"{panel}.png", dpi=300, bbox_inches="tight")
        plt.close(fig)

    fig = plt.figure(figsize=(16.0, 5.0), constrained_layout=True)
    outer = fig.add_gridspec(1, 4, wspace=.04)
    for column, panel in enumerate(panels):
        inner = outer[column].subgridspec(2, 1, height_ratios=(1.0, 1.05), hspace=.05)
        rgb_ax, map_ax = fig.add_subplot(inner[0]), fig.add_subplot(inner[1])
        _rgb_strip(rgb_ax, evidence[panel], panel)
        _topdown(map_ax, payload["cases"][panel], panel)
        rgb_ax.set_title(TITLES[panel], fontsize=11, fontweight="bold", pad=5)
        map_ax.set_title(_annotation(payload["cases"][panel], evidence[panel], panel), fontsize=7.5)
    basename = output / "qualitative_comparison"
    fig.savefig(basename.with_suffix(".png"), dpi=600, bbox_inches="tight")
    fig.savefig(basename.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(basename.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(basename.with_suffix(".tiff"), dpi=600, bbox_inches="tight",
                pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)

    zh = ("图X　Recover-and-Rerank的定性比较与失败案例。(a) PLR利用跨关键帧一致的"
          "post-NMS proposals恢复原生地图遗漏的目标；(b) CALR通过pre-NMS anchors的跨帧"
          "体素共识恢复弱目标；(c) MVSR在保持原生框几何和数量不变的情况下调整置信度与"
          "排序；(d) 从原始运行中筛选的真实失败案例。蓝色、橙色和黄色分别表示原生框、"
          "PLR框和CALR框，绿色虚线表示真实标注，红色表示失败。场景点云仅用于结果可视化，"
          "不作为模型输入。")
    en = ("Figure X. Qualitative comparisons and failure cases of Recover-and-Rerank. "
          "(a) PLR recovers a native-map miss from cross-keyframe-consistent post-NMS "
          "proposals. (b) CALR recovers weak evidence through cross-frame anchor-voxel "
          "consensus. (c) MVSR changes native scores and ranks while fixing geometry and "
          "box count. (d) A genuine failure selected from the locked run. Blue, orange, "
          "and yellow denote native, PLR, and CALR boxes; dashed green denotes ground "
          "truth; red marks failure. The point cloud is used only for visualization and "
          "is not an input to the method.")
    (output / "caption_zh.txt").write_text(zh + "\n", encoding="utf-8")
    (output / "caption_en.txt").write_text(en + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, default=DEFAULT_OUTPUT / "selected_cases.json")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    render(args.cases, args.output)


if __name__ == "__main__":
    main()
