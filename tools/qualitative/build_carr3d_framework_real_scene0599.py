#!/usr/bin/env python3
"""Build the CaRR-3D method figure from archived scene0599_01 evidence.

All photographic and 3D panels are derived from real ScanNet frames, archived
detector outputs, or archived native/PTR/VAR prediction boxes.  Only arrows,
module cards, the history plot, and the voxel glyph are schematic.
"""
from __future__ import annotations

import base64
import importlib.util
import itertools
import json
from io import BytesIO
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageOps


ROOT = Path(__file__).resolve().parents[2]
SCENE = "scene0599_01"
OUT = ROOT / "figure_assets/carr3d_framework_scene0599_real_20261107"
RGB_ROOT = ROOT / "data/scannet_val_rgbfix" / SCENE / "frames/color"
DEPTH_ROOT = ROOT / "data/scannet_val_rgbfix" / SCENE / "frames/depth"
RAW_ROOT = ROOT / "reports/scannet_seedless_full100_20260911/raw" / SCENE
RUN = ROOT / "reports/official_github_boxfusion_vs_final_recar_public_cli_coldcache_nowarmup100_20261009"
RECEIPT = RUN / "receipts/recar" / SCENE
BOXER_LOG = (
    ROOT
    / "reports/official_github_boxfusion_vs_final_recar_paper_native_sequential_coldcache_nowarmup100_20261009"
    / "receipts/recar"
    / SCENE
    / "native_boxer_diagnostics"
    / f"{SCENE}_boxer_lifting.jsonl"
)
HSR_READOUT = ROOT / "reports/scannet_hcsr_current_zero_vs_max_20261029/frame_readouts" / f"{SCENE}.npz"
FINAL_READOUT = (
    ROOT
    / "figure_assets/online_detection_readout_real_dual_branch_20261107"
    / "online_detection_readout_dual_white.png"
)

BLUE = "#1769E0"
PURPLE = "#7C22E8"
ORANGE = "#F06418"
GOLD = "#DBA000"
GREEN = "#19A65A"
INK = "#142033"
GRAY = "#667085"
LIGHT = "#F8FAFC"
CORNER_CODES = np.asarray(list(itertools.product((0, 1), repeat=3)), dtype=float)


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _iou_one(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])
    inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    area = (box[2] - box[0]) * (box[3] - box[1])
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    return inter / np.maximum(area + areas - inter, 1e-9)


def nms(boxes: np.ndarray, scores: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    order = np.argsort(-scores, kind="stable")
    keep: list[int] = []
    while len(order):
        current = int(order[0])
        keep.append(current)
        if len(order) == 1:
            break
        rest = order[1:]
        order = rest[_iou_one(boxes[current], boxes[rest]) <= threshold]
    return np.asarray(keep, dtype=int)


def detector_candidates(frame: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = np.load(RAW_ROOT / f"raw_{frame:06d}.npz")
    boxes = np.asarray(data["boxes"], dtype=float)
    scores = np.asarray(data["scores"], dtype=float)
    anchor_ids = np.argsort(-scores, kind="stable")[:300]
    proposal_pool = anchor_ids[scores[anchor_ids] >= 0.05]
    proposal_keep = nms(boxes[proposal_pool], scores[proposal_pool], 0.5)[:150]
    proposal_ids = proposal_pool[proposal_keep]
    return boxes, scores, proposal_ids


def boxer_rows() -> dict[int, dict]:
    return {int(row["frame_id"]): row for row in map(json.loads, BOXER_LOG.open())}


def native_boxes_2d(row: dict) -> np.ndarray:
    boxes = np.asarray(row["input_boxes_xyxy"], dtype=float)
    boxes[:, [0, 2]] *= 1296 / 640
    boxes[:, [1, 3]] *= 968 / 480
    return boxes


def best_match(frame: int, rows: dict[int, dict], max_area: float = 0.48) -> tuple[np.ndarray, np.ndarray]:
    native = native_boxes_2d(rows[frame])
    boxes, scores, proposals = detector_candidates(frame)
    proposal_boxes = boxes[proposals]
    best = (-1.0, 0, 0)
    for i, box in enumerate(native):
        overlaps = _iou_one(box, proposal_boxes)
        for j, overlap in enumerate(overlaps):
            p = proposal_boxes[j]
            area = (p[2] - p[0]) * (p[3] - p[1]) / (1296 * 968)
            if 0.02 < area < max_area and float(overlap) > best[0]:
                best = (float(overlap), i, j)
    return native[best[1]], proposal_boxes[best[2]]


def draw_boxes(
    frame: int,
    boxes: list[np.ndarray],
    colors: list[str],
    *,
    crop: bool = False,
    crop_ratio: float = 1.45,
) -> Image.Image:
    image = Image.open(RGB_ROOT / f"{frame}.jpg").convert("RGB")
    draw = ImageDraw.Draw(image)
    width = max(4, round(image.width / 250))
    for box, color in zip(boxes, colors):
        draw.rectangle(tuple(map(float, box)), outline=color, width=width)
    if crop and boxes:
        all_boxes = np.asarray(boxes, dtype=float)
        x1, y1 = all_boxes[:, :2].min(0)
        x2, y2 = all_boxes[:, 2:].max(0)
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        bw, bh = x2 - x1, y2 - y1
        bw = max(bw * crop_ratio, bh * 1.25)
        bh = max(bh * crop_ratio, bw / 1.75)
        left = max(0, cx - bw / 2)
        top = max(0, cy - bh / 2)
        right = min(image.width, cx + bw / 2)
        bottom = min(image.height, cy + bh / 2)
        image = image.crop((round(left), round(top), round(right), round(bottom)))
    return image


def depth_image(frame: int) -> Image.Image:
    path = DEPTH_ROOT / f"{frame}.png"
    depth = np.asarray(Image.open(path), dtype=float)
    valid = depth > 0
    lo, hi = np.percentile(depth[valid], [4, 96]) if np.any(valid) else (0, 1)
    norm = np.clip((depth - lo) / max(hi - lo, 1), 0, 1)
    out = (255 * (1 - norm)).astype(np.uint8)
    out[~valid] = 245
    return Image.fromarray(out, mode="L").convert("RGB")


def select_ptr_box(frame: int) -> np.ndarray:
    boxes, scores, proposals = detector_candidates(frame)
    proposal_boxes = boxes[proposals]
    # In the first three keyframes, the PTR target is the visible refuse bin.
    # Its post-NMS proposal occupies the lower-left region and is the highest
    # scoring proposal under the archived detector output.
    target = np.asarray([0.0, 420.0, 540.0, 968.0])
    overlaps = _iou_one(target, proposal_boxes)
    return proposal_boxes[int(np.argmax(overlaps))]


# Highest-IoU archived Top-300 anchors against the final VAR recovery projected
# into each image.  The projection is used only for audit/selection; the boxes
# drawn below are the unchanged raw detector anchors.
VAR_ANCHOR_INDEX = {825: 8089, 850: 8106, 875: 8143}


def select_var_anchor(frame: int) -> np.ndarray:
    data = np.load(RAW_ROOT / f"raw_{frame:06d}.npz")
    return np.asarray(data["boxes"][VAR_ANCHOR_INDEX[frame]], dtype=float)


def crop_white(image: Image.Image, pad: int = 8) -> Image.Image:
    rgb = image.convert("RGB")
    arr = np.asarray(rgb)
    mask = np.any(arr < 248, axis=2)
    if not np.any(mask):
        return rgb
    yy, xx = np.where(mask)
    x1, x2 = max(0, xx.min() - pad), min(rgb.width, xx.max() + pad + 1)
    y1, y2 = max(0, yy.min() - pad), min(rgb.height, yy.max() + pad + 1)
    return rgb.crop((x1, y1, x2, y2))


def render_mesh_assets() -> dict[str, Path]:
    common = load_module("box_render_common_scene0599", ROOT / "tools/qualitative/render_boxfusion_ctrl_recar_3d.py")
    renderer = common.load_renderer()
    matrix = common.alignment(SCENE)
    gt_array = np.load(ROOT / "evaluation/data_util/scannet_train_detection_data" / f"{SCENE}_bbox.npy")
    gt = gt_array[:, None, :3] + (CORNER_CODES[None] - 0.5) * gt_array[:, None, 3:6]
    native = common.read_predictions(RECEIPT / "native" / f"{SCENE}_boxes.pkl", matrix)
    full = common.read_predictions(RECEIPT / "recar" / f"{SCENE}_boxes.pkl", matrix)
    xyz, _, display_rgb, faces, _ = renderer.load_mesh(SCENE, matrix, [14, 15, 16, 18, 20])
    camera = common.make_camera(renderer, SCENE, xyz, faces, gt)
    args = (
        np.asarray(camera["position"]),
        np.asarray(camera["look_at"]),
        camera["focal_pixels"],
        np.asarray(camera["principal_point_pixels"]),
    )
    projected = renderer.project(xyz, *args)
    background = renderer.rasterize(projected, display_rgb, faces, *camera["image_size"])

    def draw_box(ax, box, color, linewidth=1.25):
        uv = renderer.project(box, *args)
        ax.add_collection(
            LineCollection(
                uv[:, :2][renderer.box_edges(box)],
                colors=color,
                linewidths=linewidth,
                alpha=0.98,
                zorder=10,
            )
        )

    specs = {
        "native_map": [(native["boxes"][i], BLUE) for i in [10, 3, 2, 5]],
        "previous_map": [(native["boxes"][i], BLUE) for i in [10, 3]],
        "ptr_recovery": [(full["boxes"][11], ORANGE)],
        "var_recovery": [(full["boxes"][61], GOLD)],
    }
    outputs: dict[str, Path] = {}
    for name, boxes in specs.items():
        fig = plt.figure(figsize=(5.1, 4.0), dpi=220)
        ax = fig.add_axes([0, 0, 1, 1])
        ax.imshow(background, interpolation="antialiased")
        for box, color in boxes:
            draw_box(ax, box, color)
        projected_boxes = renderer.project(np.concatenate([b for b, _ in boxes]), *args)[:, :2]
        if name in {"ptr_recovery", "var_recovery"}:
            low = projected_boxes.min(0) - np.asarray([260, 220])
            high = projected_boxes.max(0) + np.asarray([260, 220])
        else:
            mesh_uv = projected[np.unique(faces), :2]
            low = np.minimum(mesh_uv.min(0), projected_boxes.min(0)) - 20
            high = np.maximum(mesh_uv.max(0), projected_boxes.max(0)) + 20
        ax.set_xlim(low[0], high[0])
        ax.set_ylim(high[1], low[1])
        ax.set_aspect("equal")
        ax.axis("off")
        path = OUT / f"{name}.png"
        fig.savefig(path, dpi=320, facecolor="white", bbox_inches="tight", pad_inches=0.01)
        plt.close(fig)
        outputs[name] = path
    return outputs


def save_real_panels(rows: dict[int, dict]) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for frame in [0, 450, 900]:
        rgb = Image.open(RGB_ROOT / f"{frame}.jpg").convert("RGB")
        rgb.save(OUT / f"stream_rgb_{frame}.png")
        depth_image(frame).save(OUT / f"stream_depth_{frame}.png")
    native_box, proposal_box = best_match(900, rows)
    draw_boxes(900, [native_box], [BLUE], crop=True, crop_ratio=1.62).save(OUT / "native_frame_candidate.png")
    for frame in [1600, 2075]:
        native_box, proposal_box = best_match(frame, rows)
        draw_boxes(frame, [native_box, proposal_box], [BLUE, ORANGE], crop=True, crop_ratio=1.62).save(OUT / f"hsr_match_{frame}.png")
    for frame in [0, 25, 50]:
        box = select_ptr_box(frame)
        draw_boxes(frame, [box], [ORANGE], crop=True, crop_ratio=1.25).save(OUT / f"ptr_track_{frame}.png")
    for frame in [825, 850, 875]:
        box = select_var_anchor(frame)
        draw_boxes(frame, [box], [GOLD], crop=True, crop_ratio=1.30).save(OUT / f"var_anchor_{frame}.png")

    # Auxiliary overview: actual post-NMS proposals and pre-NMS anchors.
    frame = 0
    boxes, scores, proposals = detector_candidates(frame)
    proposal_boxes = boxes[proposals[:4]]
    anchor_ids = np.argsort(-scores, kind="stable")[:300]
    anchor_boxes = boxes[anchor_ids[:5]]
    draw_boxes(frame, list(proposal_boxes), [ORANGE] * len(proposal_boxes)).save(OUT / "aux_proposals.png")
    draw_boxes(frame, list(anchor_boxes), [GOLD] * len(anchor_boxes)).save(OUT / "aux_anchors.png")
    for p in OUT.glob("*.png"):
        paths[p.stem] = p
    return paths


def get_hsr_series(record_id: int = 20) -> tuple[list[int], list[float], list[float]]:
    data = np.load(HSR_READOUT)
    offset = 0
    frames, current, maximum = [], [], []
    for frame, length in zip(data["frame_ids"], data["lengths"]):
        length = int(length)
        for j in range(length):
            i = offset + j
            if int(data["ids"][i]) == record_id:
                frames.append(int(frame))
                current.append(float(data["current_support"][i]))
                maximum.append(float(data["max_support"][i]))
        offset += length
    return frames[:6], current[:6], maximum[:6]


def rounded(ax, x, y, w, h, edge, face="white", lw=1.15, radius=0.012, z=1):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad=0.004,rounding_size={radius}",
        linewidth=lw,
        edgecolor=edge,
        facecolor=face,
        zorder=z,
    )
    ax.add_patch(patch)
    return patch


def arrow(ax, x1, y1, x2, y2, color=GRAY, lw=1.5, z=7):
    ax.add_patch(
        FancyArrowPatch(
            (x1, y1),
            (x2, y2),
            arrowstyle="-|>",
            mutation_scale=12,
            linewidth=lw,
            color=color,
            shrinkA=1,
            shrinkB=1,
            zorder=z,
        )
    )


def add_image(ax, image_or_path, x, y, w, h, *, crop=True, z=5):
    image = Image.open(image_or_path).convert("RGB") if isinstance(image_or_path, (str, Path)) else image_or_path.convert("RGB")
    if crop:
        src_ratio = image.width / image.height
        dst_ratio = w / h
        if src_ratio > dst_ratio:
            new_w = round(image.height * dst_ratio)
            left = (image.width - new_w) // 2
            image = image.crop((left, 0, left + new_w, image.height))
        else:
            new_h = round(image.width / dst_ratio)
            top = (image.height - new_h) // 2
            image = image.crop((0, top, image.width, top + new_h))
    ax.imshow(image, extent=[x, x + w, y, y + h], aspect="auto", zorder=z, interpolation="lanczos")


def pill(ax, x, y, w, h, text, edge, face, fontsize=9.5, weight="semibold"):
    rounded(ax, x, y, w, h, edge, face, lw=0.95, radius=h / 2, z=8)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize, color=INK, weight=weight, zorder=9)


def compose(paths: dict[str, Path], mesh: dict[str, Path]):
    plt.rcParams.update({"font.family": "DejaVu Sans", "mathtext.fontset": "dejavusans"})
    fig = plt.figure(figsize=(24, 7.3), dpi=200, facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Main regions.
    rounded(ax, 0.008, 0.055, 0.105, 0.87, "#35A968", "#FBFFFC", 1.25, 0.012)
    rounded(ax, 0.122, 0.055, 0.245, 0.87, BLUE, "#FAFCFF", 1.25, 0.012)
    rounded(ax, 0.378, 0.035, 0.420, 0.91, "#B8C1CF", "#FCFCFD", 1.0, 0.015)
    rounded(ax, 0.812, 0.055, 0.180, 0.87, "#35A968", "#FBFFFC", 1.25, 0.012)

    ax.text(0.0605, 0.895, "Posed RGB-D\nStream", ha="center", va="center", fontsize=15, weight="bold", color=INK)
    ax.text(0.135, 0.895, "(a)  Per-frame perception and online mapping", fontsize=15, weight="bold", color=INK, va="center")
    ax.text(0.588, 0.907, "Cross-keyframe candidate evidence", fontsize=16, weight="bold", color=INK, ha="center")
    ax.text(0.823, 0.895, "(e)  Online detection readout", fontsize=15, weight="bold", color=INK, va="center")

    # Input stream: three exact RGB/depth pairs.
    for k, frame in enumerate([0, 450, 900]):
        yy = 0.68 - k * 0.235
        add_image(ax, paths[f"stream_rgb_{frame}"], 0.018, yy, 0.068, 0.17)
        add_image(ax, paths[f"stream_depth_{frame}"], 0.066, yy - 0.018, 0.038, 0.09)
    ax.text(0.060, 0.19, "⋮", fontsize=25, ha="center", color=INK)
    # Compact camera path glyph.
    ax.plot([0.024, 0.055, 0.091], [0.105, 0.085, 0.112], color="#7B8798", lw=1.2, linestyle=(0, (2, 2)))
    for xx, yy in [(0.024, 0.105), (0.055, 0.085), (0.091, 0.112)]:
        ax.add_patch(plt.Circle((xx, yy), 0.006, facecolor="white", edgecolor=GREEN, lw=1.4, zorder=6))

    # Per-frame perception: native path.
    rounded(ax, 0.132, 0.515, 0.225, 0.330, "#AFCBFA", "white", 0.85, 0.010)
    ax.text(0.143, 0.814, "Native mapping path", fontsize=12.5, weight="bold", color=INK)
    add_image(ax, paths["native_frame_candidate"], 0.142, 0.575, 0.075, 0.17)
    pill(ax, 0.226, 0.620, 0.064, 0.086, "Association\nTop-K · PFO", "#AFCBFA", "#F3F7FF", 9.0)
    add_image(ax, mesh["native_map"], 0.299, 0.565, 0.050, 0.185, crop=False)
    arrow(ax, 0.217, 0.66, 0.225, 0.66, BLUE)
    arrow(ax, 0.290, 0.66, 0.298, 0.66, BLUE)
    ax.text(0.180, 0.548, r"frame candidates", ha="center", fontsize=8.8, color=GRAY)
    ax.text(0.323, 0.548, r"native records $\mathcal{D}^{N}_{t}$", ha="center", fontsize=8.8, color=GRAY)

    # Auxiliary path with real candidate overlays.
    rounded(ax, 0.132, 0.105, 0.225, 0.375, "#C9D0DA", "white", 0.85, 0.010)
    ax.text(0.143, 0.447, "Auxiliary candidate path", fontsize=12.5, weight="bold", color=INK)
    pill(ax, 0.143, 0.260, 0.064, 0.105, "WeDetect-Uni\n+ Shared Boxer", "#B7C1CF", "#F7F8FA", 9.0)
    add_image(ax, paths["aux_proposals"], 0.219, 0.292, 0.082, 0.105)
    add_image(ax, paths["aux_anchors"], 0.219, 0.145, 0.082, 0.105)
    arrow(ax, 0.207, 0.312, 0.218, 0.342, ORANGE)
    arrow(ax, 0.207, 0.312, 0.218, 0.195, GOLD)
    ax.text(0.311, 0.342, r"proposals $\mathcal{P}_{t}$", fontsize=9.8, color=ORANGE, weight="bold", va="center")
    ax.text(0.311, 0.195, r"anchors $\mathcal{A}_{t}$", fontsize=9.8, color=GOLD, weight="bold", va="center")
    ax.text(0.244, 0.117, "one frozen pass · shared 3D lifting", fontsize=8.3, color=GRAY, ha="center", style="italic")

    # State progression header.
    ax.text(0.397, 0.858, r"state at $t-1$", fontsize=10.5, color=GRAY)
    ax.text(0.779, 0.858, r"state at $t$", fontsize=10.5, color=GRAY, ha="right")
    arrow(ax, 0.463, 0.861, 0.748, 0.861, "#98A2B3", 1.1)
    ax.text(0.606, 0.875, r"keyframe $t$", fontsize=10.5, style="italic", ha="center", color=INK)

    # HSR row.
    rounded(ax, 0.390, 0.590, 0.395, 0.245, PURPLE, "#FEFBFF", 1.05, 0.010)
    pill(ax, 0.400, 0.787, 0.043, 0.038, "HSR", PURPLE, "#EAD7FF", 10.5, "bold")
    ax.text(0.452, 0.806, "Historical support reranking", fontsize=13, weight="bold", color=INK, va="center")
    add_image(ax, paths["hsr_match_1600"], 0.412, 0.636, 0.068, 0.122)
    add_image(ax, paths["hsr_match_2075"], 0.486, 0.636, 0.068, 0.122)
    ax.text(0.483, 0.610, "record–proposal matches", fontsize=8.6, color=GRAY, ha="center")

    # Exact archived running-maximum support for record 20.
    frames, current, maximum = get_hsr_series(20)
    chart_x, chart_y, chart_w, chart_h = 0.579, 0.635, 0.092, 0.116
    rounded(ax, chart_x, chart_y, chart_w, chart_h, "#D9B4FF", "#F8F0FF", 0.8, 0.008)
    cx = np.linspace(chart_x + 0.010, chart_x + chart_w - 0.010, len(current))
    base = chart_y + 0.012
    scale = (chart_h - 0.032) / 0.82
    for xx, value in zip(cx, current):
        ax.add_patch(Rectangle((xx - 0.0042, base), 0.0084, value * scale, facecolor="#D9B4F2", edgecolor="none", zorder=7))
    ax.step(cx, base + np.asarray(maximum) * scale, where="post", color=PURPLE, lw=1.5, zorder=9)
    ax.scatter(cx, base + np.asarray(maximum) * scale, s=14, facecolor="white", edgecolor=PURPLE, lw=1.0, zorder=10)
    ax.text(chart_x + chart_w / 2, chart_y - 0.018, "history max", fontsize=8.6, color=PURPLE, ha="center", weight="bold")
    arrow(ax, 0.555, 0.694, 0.576, 0.694, PURPLE)
    arrow(ax, 0.674, 0.694, 0.693, 0.694, PURPLE)
    pill(ax, 0.700, 0.716, 0.069, 0.038, "A  >  B  >  C", "#D7B5FA", "#F7EFFF", 9.0)
    pill(ax, 0.700, 0.642, 0.069, 0.038, "C  >  A  >  B", "#D7B5FA", "#F7EFFF", 9.0, "bold")
    arrow(ax, 0.7345, 0.714, 0.7345, 0.684, PURPLE)

    # PTR row.
    rounded(ax, 0.390, 0.330, 0.395, 0.235, ORANGE, "#FFFCFA", 1.05, 0.010)
    pill(ax, 0.400, 0.517, 0.043, 0.038, "PTR", ORANGE, "#FFE0CF", 10.5, "bold")
    ax.text(0.452, 0.536, "Proposal track recovery", fontsize=13, weight="bold", color=INK, va="center")
    ptr_frames = [0, 25, 50]
    for k, frame in enumerate(ptr_frames):
        xx = 0.412 + k * 0.076
        add_image(ax, paths[f"ptr_track_{frame}"], xx, 0.385, 0.067, 0.103)
        ax.text(xx + 0.0335, 0.366, [r"$t-2$", r"$t-1$", r"$t$"][k], fontsize=9.2, ha="center", color=INK)
        if k < 2:
            arrow(ax, xx + 0.067, 0.436, xx + 0.075, 0.436, ORANGE, 1.2)
    pill(ax, 0.648, 0.401, 0.060, 0.074, "Track\nverification", ORANGE, "#FFF1E8", 9.2)
    add_image(ax, mesh["ptr_recovery"], 0.721, 0.372, 0.054, 0.124, crop=False)
    arrow(ax, 0.634, 0.436, 0.647, 0.436, ORANGE)
    arrow(ax, 0.708, 0.436, 0.719, 0.436, ORANGE)
    ax.text(0.748, 0.350, "PTR recovery", fontsize=8.8, color=ORANGE, weight="bold", ha="center")

    # VAR row.
    rounded(ax, 0.390, 0.070, 0.395, 0.235, GOLD, "#FFFDF8", 1.05, 0.010)
    pill(ax, 0.400, 0.257, 0.043, 0.038, "VAR", GOLD, "#FFE79C", 10.5, "bold")
    ax.text(0.452, 0.276, "Voxelized anchor recovery", fontsize=13, weight="bold", color=INK, va="center")
    for k, frame in enumerate([825, 850, 875]):
        xx = 0.412 + k * 0.076
        add_image(ax, paths[f"var_anchor_{frame}"], xx, 0.125, 0.067, 0.103)
        ax.text(xx + 0.0335, 0.106, [r"$t-2$", r"$t-1$", r"$t$"][k], fontsize=9.2, ha="center", color=INK)
    # Minimal voxel consensus glyph.
    vx, vy, vw, vh = 0.654, 0.140, 0.050, 0.076
    for i in range(4):
        ax.plot([vx + i * vw / 3, vx + i * vw / 3], [vy, vy + vh], color="#98A2B3", lw=0.8, zorder=8)
    for j in range(4):
        ax.plot([vx, vx + vw], [vy + j * vh / 3, vy + j * vh / 3], color="#98A2B3", lw=0.8, zorder=8)
    ax.add_patch(Rectangle((vx + vw / 3, vy + vh / 3), vw / 3, vh / 3, facecolor="#F2C600", alpha=0.9, zorder=7))
    ax.text(vx + vw / 2, vy - 0.020, "voxel consensus", fontsize=8.5, ha="center", color=GOLD, weight="bold")
    arrow(ax, 0.634, 0.177, 0.650, 0.177, GOLD)
    add_image(ax, mesh["var_recovery"], 0.721, 0.112, 0.054, 0.124, crop=False)
    arrow(ax, 0.706, 0.177, 0.719, 0.177, GOLD)
    ax.text(0.748, 0.090, "VAR recovery", fontsize=8.8, color=GOLD, weight="bold", ha="center")

    # Connections from per-frame paths to the state modules.
    arrow(ax, 0.348, 0.660, 0.389, 0.710, BLUE, 1.5)
    arrow(ax, 0.348, 0.342, 0.389, 0.440, ORANGE, 1.5)
    arrow(ax, 0.348, 0.195, 0.389, 0.180, GOLD, 1.5)

    # Final readout uses the user-specified real scene render without editing.
    final_img = crop_white(Image.open(FINAL_READOUT))
    add_image(ax, final_img, 0.830, 0.205, 0.145, 0.520, crop=False)
    ax.text(0.9025, 0.160, r"current detections at keyframe $t$", fontsize=11.5, weight="bold", color=INK, ha="center")
    arrow(ax, 0.785, 0.710, 0.830, 0.610, PURPLE, 1.7)
    arrow(ax, 0.785, 0.440, 0.830, 0.455, ORANGE, 1.7)
    arrow(ax, 0.785, 0.180, 0.830, 0.300, GOLD, 1.7)

    png = OUT / "carr3d_framework_scene0599_real.png"
    svg = OUT / "carr3d_framework_scene0599_real.svg"
    pdf = OUT / "carr3d_framework_scene0599_real.pdf"
    fig.savefig(png, dpi=260, facecolor="white", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(svg, facecolor="white", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(pdf, facecolor="white", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    return png, svg, pdf


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = boxer_rows()
    paths = save_real_panels(rows)
    mesh = render_mesh_assets()
    outputs = compose(paths, mesh)
    manifest = {
        "schema": "carr3d.framework.real_scene0599.v1",
        "scene": SCENE,
        "outputs": [str(p) for p in outputs],
        "final_readout_source": str(FINAL_READOUT),
        "input_rgb_root": str(RGB_ROOT),
        "input_depth_root": str(DEPTH_ROOT),
        "detector_raw_root": str(RAW_ROOT),
        "native_boxer_log": str(BOXER_LOG),
        "hsr_frame_readout": str(HSR_READOUT),
        "input_stream_frames": [0, 450, 900],
        "hsr_match_frames": [1600, 2075],
        "hsr_history_record_id": 20,
        "ptr_track_frames": [0, 25, 50],
        "var_anchor_frames": [825, 850, 875],
        "var_anchor_raw_indices": VAR_ANCHOR_INDEX,
        "native_prediction_indices": [10, 3, 2, 5],
        "ptr_prediction_index": 11,
        "var_prediction_index": 61,
        "data_scope": "All photographic and 3D result panels use scene0599_01.",
        "schematic_elements": ["arrows", "module cards", "voxel grid", "ranking tokens"],
        "geometry_edited": False,
        "new_model_inference": False,
    }
    (OUT / "source_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"output_dir": str(OUT), "figure": str(outputs[0])}, ensure_ascii=False))


if __name__ == "__main__":
    main()
