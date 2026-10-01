#!/usr/bin/env python3
"""Render final-config four-panel qualitative evidence figure."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import pickle

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
from PIL import Image

COLORS = {"native": "#1677FF", "plr": "#FF7A1A", "calr": "#F2C300",
          "gt": "#14B866", "failure": "#D62728", "support": "#A855F7"}
TITLES = {"plr_recovery": "(a) PLR recovery", "calr_recovery": "(b) CALR recovery",
          "mvsr_reranking": "(c) MVSR reranking", "failure": "(d) Failure case"}


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            value.update(chunk)
    return value.hexdigest()


def read_boxes(path: Path) -> np.ndarray:
    with path.open("rb") as handle:
        rows = pickle.load(handle)[0]
    return np.asarray([r[1] for r in rows], dtype=np.float64).reshape(-1, 8, 3)


def alignment(raw_root: Path, scene: str) -> np.ndarray:
    path = raw_root / scene / f"{scene}.txt"
    for line in path.read_text().splitlines():
        if line.startswith("axisAlignment ="):
            return np.asarray(line.split("=", 1)[1].split(), dtype=float).reshape(4, 4)
    raise RuntimeError(f"missing axisAlignment: {path}")


def align_boxes(boxes: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    return np.einsum("ij,nkj->nki", matrix[:3, :3], boxes) + matrix[:3, 3]


def project_box(box: np.ndarray, raw_root: Path, scene: str, frame: int,
                output_size=(640, 480)) -> list[float] | None:
    pose = np.loadtxt(raw_root / scene / "pose" / f"{frame}.txt")
    intrinsic = np.loadtxt(raw_root / scene / "intrinsic" / "intrinsic_color.txt")[:3, :3]
    world = np.c_[np.asarray(box, dtype=float).reshape(8, 3), np.ones(8)]
    camera = (np.linalg.inv(pose) @ world.T).T[:, :3]
    valid = camera[:, 2] > 1e-4
    if valid.sum() < 2:
        return None
    uvw = (intrinsic @ camera[valid].T).T
    uv = uvw[:, :2] / uvw[:, 2:3]
    image = Image.open(raw_root / scene / "color" / f"{frame}.jpg")
    sx, sy = output_size[0] / image.width, output_size[1] / image.height
    uv[:, 0] *= sx; uv[:, 1] *= sy
    lo, hi = uv.min(0), uv.max(0)
    lo = np.maximum(lo, [0, 0]); hi = np.minimum(hi, [output_size[0]-1, output_size[1]-1])
    return [float(lo[0]), float(lo[1]), float(hi[0]), float(hi[1])]


def draw_rect(ax, box, color, dashed=False, width=2.0):
    if box is None:
        return
    x0, y0, x1, y1 = box
    ax.add_patch(Rectangle((x0, y0), x1-x0, y1-y0, fill=False,
                           edgecolor=color, linewidth=width,
                           linestyle="--" if dashed else "-"))


def rgb_strip(ax, case, evidence, raw_root: Path, failure=False):
    images, spans = [], []
    offset = 0
    for row in evidence["frames"][:3]:
        image = Image.open(row["rgb_path"]).convert("RGB").resize((640, 480), Image.Resampling.LANCZOS)
        images.append(np.asarray(image)); spans.append((offset, row)); offset += 640
    if not images:
        raise RuntimeError(f"no RGB evidence for {case['scene']}")
    ax.imshow(np.concatenate(images, axis=1))
    target = np.asarray(case["box"], dtype=float)
    for xoff, row in spans:
        frame = int(row["frame_id"])
        candidate = row.get("candidate_xyxy")
        if candidate is None:
            candidate = project_box(row["candidate_box"], raw_root, case["scene"], frame)
        candidate = None if candidate is None else [candidate[0]+xoff, candidate[1], candidate[2]+xoff, candidate[3]]
        if case["source"] == "native":
            native = project_box(target, raw_root, case["scene"], frame)
            native = None if native is None else [native[0]+xoff, native[1], native[2]+xoff, native[3]]
            draw_rect(ax, native, COLORS["failure"] if failure else COLORS["native"], width=2.2)
            draw_rect(ax, candidate, COLORS["support"], dashed=True, width=1.5)
        else:
            color = COLORS["failure"] if failure else COLORS[case["source"]]
            draw_rect(ax, candidate, color, width=2.1)
        ax.text(xoff+8, 22, f"t={frame}", color="white", fontsize=7,
                bbox={"facecolor": "black", "alpha": .55, "edgecolor": "none", "pad": 1})
    ax.set_xlim(0, 640*len(images)); ax.set_ylim(480, 0); ax.axis("off")


def box2d(ax, corners, color, dashed=False, width=1.5, alpha=1.0):
    corners = np.asarray(corners, dtype=float)
    lo, hi = corners[:, :2].min(0), corners[:, :2].max(0)
    ax.add_patch(Rectangle(lo, *(hi-lo), fill=False, edgecolor=color,
                           linewidth=width, linestyle="--" if dashed else "-", alpha=alpha))


def topdown(ax, case, components: Path, gt_root: Path, raw_root: Path, failure=False):
    scene = case["scene"]
    vertices = np.load(gt_root / f"{scene}_vert.npy")
    if len(vertices) > 35000:
        vertices = vertices[::max(1, len(vertices)//35000)]
    colors = vertices[:, 3:6]/255 if vertices.shape[1] >= 6 else "#AAAAAA"
    ax.scatter(vertices[:, 0], vertices[:, 1], c=colors, s=.15, alpha=.28, linewidth=0, rasterized=True)
    matrix = alignment(raw_root, scene)
    native = align_boxes(read_boxes(components / "native_native" / f"{scene}_boxes.pkl"), matrix)
    for box in native:
        box2d(ax, box, COLORS["native"], width=.65, alpha=.3)
    selected = np.asarray(case["box_aligned"], dtype=float)
    color = COLORS["failure"] if failure else COLORS[case["source"]]
    box2d(ax, selected, color, width=2.4)
    if case.get("gt_box_aligned") is not None:
        box2d(ax, case["gt_box_aligned"], COLORS["gt"], dashed=True, width=2.0)
    lo, hi = selected[:, :2].min(0), selected[:, :2].max(0)
    margin = max(float((hi-lo).max())*2.6, 1.2); center = (lo+hi)/2
    ax.set_xlim(center[0]-margin, center[0]+margin); ax.set_ylim(center[1]-margin, center[1]+margin)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([]); ax.set_facecolor("#F8FAFC")


def annotation(panel, case, evidence):
    if panel == "mvsr_reranking":
        return (f"score {case['original_score']:.3f} → {case['reranked_score']:.3f}\n"
                f"rank {case['original_rank']} → {case['reranked_rank']} · max support {evidence['support']:.3f}")
    if panel == "failure":
        return case["failure_reason"]
    return f"IoU={case['iou']:.2f} · {len(set(r['frame_id'] for r in evidence['frames']))} shown keyframes"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--evidence-dir", type=Path, required=True)
    parser.add_argument("--components", type=Path, required=True)
    parser.add_argument("--gt-root", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = json.loads(args.selection.read_text())
    panels = ("plr_recovery", "calr_recovery", "mvsr_reranking", "failure")
    evidence = {}
    for panel in panels:
        case = payload["cases"][panel]
        path = Path(case["evidence_export"])
        if not case.get("evidence_verified") or digest(path) != case["evidence_sha256"]:
            raise RuntimeError(f"unverified qualitative evidence: {panel}")
        evidence[panel] = json.loads(path.read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "cases").mkdir(exist_ok=True)
    for panel in panels:
        fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2), constrained_layout=True,
                                 gridspec_kw={"width_ratios": [1.45, 1]})
        rgb_strip(axes[0], payload["cases"][panel], evidence[panel], args.raw_root, panel=="failure")
        topdown(axes[1], payload["cases"][panel], args.components, args.gt_root,
                args.raw_root, panel=="failure")
        fig.suptitle(f"{TITLES[panel]} — {payload['cases'][panel]['scene']}", fontsize=11, fontweight="bold")
        axes[1].set_title(annotation(panel, payload["cases"][panel], evidence[panel]), fontsize=8)
        fig.savefig(args.output / "cases" / f"{panel}.png", dpi=300, bbox_inches="tight")
        plt.close(fig)
    fig = plt.figure(figsize=(16, 5), constrained_layout=True)
    outer = fig.add_gridspec(1, 4, wspace=.04)
    for column, panel in enumerate(panels):
        grid = outer[column].subgridspec(2, 1, height_ratios=(1, 1.05), hspace=.05)
        top, bottom = fig.add_subplot(grid[0]), fig.add_subplot(grid[1])
        rgb_strip(top, payload["cases"][panel], evidence[panel], args.raw_root, panel=="failure")
        topdown(bottom, payload["cases"][panel], args.components, args.gt_root,
                args.raw_root, panel=="failure")
        top.set_title(TITLES[panel], fontsize=11, fontweight="bold")
        bottom.set_title(annotation(panel, payload["cases"][panel], evidence[panel]), fontsize=7.5)
    base = args.output / "qualitative_comparison"
    fig.savefig(base.with_suffix(".png"), dpi=600, bbox_inches="tight")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".tiff"), dpi=600, bbox_inches="tight",
                pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)
    zh = ("图X　ReCaR-3D的定性结果与失败案例。(a) PLR利用跨关键帧一致的post-NMS proposals恢复原生地图漏检；"
          "(b) CALR通过pre-NMS anchors的跨关键帧体素共识恢复弱目标；(c) MVSR保持原生框几何和数量不变，"
          "仅更新其分数与排序；(d) 从最终配置真实输出中筛选的失败案例。蓝、橙、黄分别表示原生、PLR和CALR框，"
          "绿色虚线表示真实标注，紫色虚线表示二维支持，红色表示失败。点云仅用于可视化，不作为模型输入。")
    en = ("Figure X. Qualitative results and a failure case of ReCaR-3D. (a) PLR recovers a native-map miss from "
          "cross-keyframe-consistent post-NMS proposals. (b) CALR recovers weak targets through cross-keyframe anchor-voxel "
          "consensus. (c) MVSR changes native scores and ranks while fixing geometry and box count. (d) A genuine failure "
          "selected from the final configuration. Blue, orange, and yellow denote native, PLR, and CALR boxes; dashed green "
          "denotes ground truth; dashed purple denotes 2D support; red marks failure. Point clouds are used only for visualization.")
    (args.output / "caption_zh.txt").write_text(zh+"\n")
    (args.output / "caption_en.txt").write_text(en+"\n")


if __name__ == "__main__":
    main()
