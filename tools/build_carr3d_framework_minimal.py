#!/usr/bin/env python3
"""Build a compact, paper-style CaRR-3D framework figure.

The figure deliberately keeps the posed RGB-D stream outside panel (a), uses
real project assets for every visual stage, and summarizes HSR/PTR/VAR as
online evidence states instead of expanding their low-level bookkeeping.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figure_assets/carr3d_framework_minimal_20261107"

RGBD = ROOT / "figure_assets/rgbd_input_scannet_scene0432_01"
PIPE = ROOT / "figure_assets/pipeline_candidate_visuals_scene0432_01"
NATIVE = ROOT / "figure_assets/native_online_mapper_corrected_scene0432_01_20261103"
INSERTS = ROOT / "figure_assets/native_online_mapper_real_inserts_scene0432_01_20261103"
TEMP3D = ROOT / "figure_assets/motivation_temporal_evidence_3d_20261101"
FLOW = ROOT / "figure_assets/recar3d_full_flowchart_optimized_20261103"

ASSETS = {
    "rgbd0": RGBD / "frame_0000_rgbd_inset.png",
    "rgbd1": RGBD / "frame_0144_rgbd_inset.png",
    "rgbd2": RGBD / "frame_0288_rgbd_inset.png",
    "candidate": PIPE / "02c_reliable_view_top3_frame0050.png",
    "views": INSERTS / "07_associated_observation_history_strip.png",
    "native": NATIVE / "02_updated_native_map_kf0050.png",
    "proposals": PIPE / "09b_boxer_lifted_proposals_orange.png",
    "anchors": PIPE / "09d_boxer_lifted_anchors_amber.png",
    "hsr": PIPE / "06b_m2_match_cached_proposals_triptych_ottoman.png",
    "ptr": PIPE / "08_plr_cross_keyframe_same_track.png",
    "var1": TEMP3D / "20a_calr_3d_anchor_t1_frame1400.png",
    "var2": TEMP3D / "20b_calr_3d_anchor_t2_frame1575.png",
    "var3": TEMP3D / "20c_calr_3d_anchor_t3_frame1600.png",
    "output": FLOW / "final_detection_ptr_orange.png",
}


INK = "#172235"
MUTED = "#667085"
LINE = "#8793A3"
LIGHT_LINE = "#D7DCE3"
BLUE = "#276EF1"
BLUE_BG = "#F5F8FD"
ORANGE = "#E96820"
ORANGE_BG = "#FFF8F3"
SOFT = "#FAFBFC"


def _font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    return ImageFont.truetype(f"/usr/share/fonts/truetype/dejavu/{name}", size)


def build_var_strip() -> Path:
    """Crop the real lifted-anchor views into a clean three-keyframe strip."""
    crops = []
    for key in ("var1", "var2", "var3"):
        im = Image.open(ASSETS[key]).convert("RGB")
        # Remove the large explanatory header while preserving the real 3D
        # observation, measured anchor box and coloured voxel-centre marker.
        crop = im.crop((300, 470, 1640, 1450))
        crops.append(crop)

    tile_w, tile_h = 420, 315
    gap = 12
    canvas = Image.new("RGB", (tile_w * 3 + gap * 2, tile_h), "white")
    draw = ImageDraw.Draw(canvas)
    for i, crop in enumerate(crops):
        crop.thumbnail((tile_w, tile_h), Image.Resampling.LANCZOS)
        x = i * (tile_w + gap) + (tile_w - crop.width) // 2
        y = (tile_h - crop.height) // 2
        canvas.paste(crop, (x, y))
        draw.rounded_rectangle(
            (i * (tile_w + gap), 0, i * (tile_w + gap) + tile_w - 1, tile_h - 1),
            radius=13,
            outline="#D4D8DE",
            width=3,
        )
        draw.rounded_rectangle(
            (i * (tile_w + gap) + 14, 14, i * (tile_w + gap) + 68, 56),
            radius=10,
            fill="white",
            outline="#D4D8DE",
            width=2,
        )
        draw.text(
            (i * (tile_w + gap) + 41, 35),
            f"t{i + 1}",
            font=_font(22, True),
            fill=INK,
            anchor="mm",
        )
    out = OUT / "var_real_anchor_history.png"
    canvas.save(out, quality=96)
    return out


def add_round(ax, x, y, w, h, edge, face="white", lw=1.8, radius=0.8, z=1):
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0.012,rounding_size={radius}",
        linewidth=lw, edgecolor=edge, facecolor=face, zorder=z,
    )
    ax.add_patch(patch)
    return patch


def add_image(ax, path: Path, x, y, w, h, *, contain=False, border=LIGHT_LINE, lw=1.0, z=3):
    raw = Image.open(path).convert("RGBA")
    # Several map renders have a transparent canvas.  Composite on white so
    # Matplotlib/PDF export does not turn the transparent area black.
    white = Image.new("RGBA", raw.size, "white")
    white.alpha_composite(raw)
    im = white.convert("RGB")
    src_ratio = im.width / im.height
    dst_ratio = w / h
    if contain:
        extent = (x, x + w, y, y + h)
    else:
        if src_ratio > dst_ratio:
            new_w = int(im.height * dst_ratio)
            left = max(0, (im.width - new_w) // 2)
            im = im.crop((left, 0, left + new_w, im.height))
        else:
            new_h = int(im.width / dst_ratio)
            top = max(0, (im.height - new_h) // 2)
            im = im.crop((0, top, im.width, top + new_h))
        extent = (x, x + w, y, y + h)
    ax.imshow(im, extent=extent, interpolation="lanczos", zorder=z, aspect="auto")
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.002,rounding_size=0.35",
        linewidth=lw, edgecolor=border, facecolor="none", zorder=z + 1,
    ))


def arrow(ax, x1, y1, x2, y2, color=LINE, lw=1.7, head=13, z=5):
    ax.add_patch(FancyArrowPatch(
        (x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=head,
        linewidth=lw, color=color, shrinkA=0, shrinkB=0, zorder=z,
    ))


def label(ax, x, y, s, size=11, color=INK, weight="normal", ha="center", va="center", z=8):
    ax.text(x, y, s, fontsize=size, color=color, fontweight=weight,
            ha=ha, va=va, zorder=z, family="DejaVu Sans")


def section_title(ax, x, y, letter, title, color):
    label(ax, x, y, f"({letter})", 15, color, "bold", "left")
    label(ax, x + 2.2, y, title, 15, INK, "bold", "left")


def build():
    OUT.mkdir(parents=True, exist_ok=True)
    missing = [str(p) for p in ASSETS.values() if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing assets:\n" + "\n".join(missing))
    var_strip = build_var_strip()

    mpl.rcParams.update({
        "font.family": "DejaVu Sans",
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    fig = plt.figure(figsize=(20, 7.2), dpi=250, facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 36)
    ax.axis("off")

    # Unnumbered input: visually separate from panel (a).
    add_round(ax, 1.0, 3.0, 9.8, 29.5, LIGHT_LINE, "white", 1.5, 0.8)
    label(ax, 5.9, 31.0, "Posed RGB-D stream", 12.5, INK, "bold")
    for i, key in enumerate(("rgbd0", "rgbd1", "rgbd2")):
        add_image(ax, ASSETS[key], 2.1, 22.4 - i * 7.7, 7.6, 6.0, border="#D7DCE3")
    label(ax, 5.9, 5.0, "RGB · depth · pose", 10.5, MUTED)
    arrow(ax, 10.9, 17.7, 13.1, 17.7, LINE, 1.8, 14)

    # (a) Two inherited/frozen perception paths.
    add_round(ax, 13.4, 2.0, 28.4, 32.0, BLUE, BLUE_BG, 1.8, 0.9)
    section_title(ax, 14.7, 32.5, "a", "Frozen perception paths", BLUE)
    add_round(ax, 14.8, 18.3, 25.6, 12.1, "#AFC8F7", "white", 1.2, 0.65)
    label(ax, 16.0, 29.2, "Native mapper", 12, BLUE, "bold", "left")
    label(ax, 38.7, 29.2, "frozen", 9.5, MUTED, "normal", "right")
    add_image(ax, ASSETS["candidate"], 15.8, 21.4, 6.0, 5.5, border="#C8D8F5")
    label(ax, 18.8, 20.2, "candidate", 9.5, MUTED)
    arrow(ax, 22.1, 24.2, 23.7, 24.2, LINE, 1.5, 11)
    label(ax, 24.5, 27.2, "association", 9.3, MUTED)
    add_image(ax, ASSETS["views"], 23.9, 21.4, 8.2, 5.5, border="#C8D8F5")
    label(ax, 28.0, 20.2, "reliable views", 9.5, MUTED)
    arrow(ax, 32.4, 24.2, 33.7, 24.2, LINE, 1.5, 11)
    label(ax, 33.0, 27.2, "PFO", 9.3, MUTED)
    add_image(ax, ASSETS["native"], 34.0, 20.7, 5.3, 7.0, contain=True, border="#C8D8F5")
    label(ax, 36.65, 19.6, "native map  Nₜ", 9.8, BLUE, "bold")

    add_round(ax, 14.8, 4.0, 25.6, 12.2, "#C9CED6", "white", 1.2, 0.65)
    label(ax, 16.0, 15.0, "Auxiliary candidates", 12, INK, "bold", "left")
    add_round(ax, 16.1, 8.1, 6.0, 4.2, "#AAB2BD", SOFT, 1.1, 0.45)
    label(ax, 19.1, 10.7, "WeDetect-Uni", 10.8, INK, "bold")
    label(ax, 19.1, 9.1, "+ Boxer lifting", 9.3, MUTED)
    arrow(ax, 22.4, 11.4, 24.2, 12.1, ORANGE, 1.6, 11)
    arrow(ax, 22.4, 9.0, 24.2, 7.5, ORANGE, 1.6, 11)
    add_image(ax, ASSETS["proposals"], 24.5, 10.1, 7.0, 4.1, border="#E9B596")
    add_image(ax, ASSETS["anchors"], 32.3, 10.1, 7.0, 4.1, border="#E9B596")
    label(ax, 28.0, 8.25, "post-NMS\nproposals  Pₜ", 8.8, ORANGE, "bold")
    label(ax, 35.8, 8.25, "pre-NMS\nanchors  Aₜ", 8.8, ORANGE, "bold")
    label(ax, 27.7, 5.6, "frame-level evidence", 9.6, MUTED)

    arrow(ax, 41.9, 17.7, 44.0, 17.7, LINE, 1.8, 14)

    # (b) Method contribution: three persistent evidence states.
    add_round(ax, 44.3, 2.0, 38.8, 32.0, ORANGE, ORANGE_BG, 1.9, 0.9)
    section_title(ax, 45.6, 32.5, "b", "Cross-keyframe candidate evidence", ORANGE)
    label(ax, 80.8, 32.5, "CaRR-3D", 11.5, ORANGE, "bold", "right")
    add_round(ax, 48.1, 28.5, 8.0, 2.2, LIGHT_LINE, "white", 1.0, 0.35)
    add_round(ax, 71.1, 28.5, 8.0, 2.2, LIGHT_LINE, "white", 1.0, 0.35)
    label(ax, 52.1, 29.6, "state at  t−1", 9.5, MUTED, "bold")
    label(ax, 75.1, 29.6, "state at  t", 9.5, MUTED, "bold")
    arrow(ax, 56.5, 29.6, 70.6, 29.6, LINE, 1.6, 12)

    cards = [
        (45.6, "HSR", "Nₜ + Pₜ", ASSETS["hsr"], "support history  →  rerank", "reranked native"),
        (57.8, "PTR", "Pₜ + Nₜ", ASSETS["ptr"], "proposal tracks  →  recover", "PTR boxes"),
        (70.0, "VAR", "Aₜ", var_strip, "anchor voxels  →  recover", "VAR boxes"),
    ]
    for x, name, inputs, image_path, action, output in cards:
        add_round(ax, x, 7.1, 11.7, 19.3, "#D6C0B2", "white", 1.2, 0.65)
        label(ax, x + 0.7, 24.9, name, 13.5, ORANGE, "bold", "left")
        add_round(ax, x + 7.2, 23.9, 3.8, 1.7, "#E8CFC0", "#FFF9F5", 0.9, 0.3)
        label(ax, x + 9.1, 24.75, inputs, 8.7, MUTED, "bold")
        # All three measured inserts are temporal strips; preserving their
        # full horizontal extent is essential to communicate cross-keyframe
        # evidence rather than a single-frame crop.
        add_image(ax, image_path, x + 0.7, 16.6, 10.3, 5.9,
                  contain=True, border="#D7DCE3")
        label(ax, x + 5.85, 14.75, action, 9.6, INK, "bold")
        arrow(ax, x + 2.1, 12.8, x + 9.6, 12.8, ORANGE, 1.5, 11)
        add_round(ax, x + 2.25, 9.4, 7.2, 2.1, "#E6B89C", "#FFF7F2", 0.9, 0.3)
        label(ax, x + 5.85, 10.45, output, 9.3, ORANGE, "bold")

    add_round(ax, 58.2, 3.25, 11.0, 2.1, "#E5C9B8", "white", 1.0, 0.35)
    label(ax, 63.7, 4.3, "causal · bounded · online", 9.8, MUTED, "bold")

    arrow(ax, 83.3, 17.7, 85.2, 17.7, LINE, 1.8, 14)

    # (c) A single online readout, not another algorithmic module.
    add_round(ax, 85.5, 2.0, 13.5, 32.0, "#8E9AAA", "white", 1.7, 0.9)
    section_title(ax, 86.5, 32.5, "c", "Online readout", INK)
    add_round(ax, 89.0, 28.4, 6.5, 2.1, "#BDC4CD", SOFT, 1.0, 0.35)
    label(ax, 92.25, 29.45, "direct assembly", 9.5, MUTED, "bold")
    add_image(ax, ASSETS["output"], 86.5, 10.0, 11.5, 16.7, contain=True, border="#D7DCE3")
    label(ax, 92.25, 7.7, "3D detections at keyframe  t", 11.2, INK, "bold")
    label(ax, 92.25, 5.6, "reranked native + PTR + VAR", 9.6, MUTED)

    png = OUT / "carr3d_framework_minimal.png"
    pdf = OUT / "carr3d_framework_minimal.pdf"
    svg = OUT / "carr3d_framework_minimal.svg"
    fig.savefig(png, dpi=300, bbox_inches="tight", pad_inches=0.05, facecolor="white")
    fig.savefig(pdf, dpi=300, bbox_inches="tight", pad_inches=0.05, facecolor="white")
    fig.savefig(svg, dpi=300, bbox_inches="tight", pad_inches=0.05, facecolor="white")
    plt.close(fig)

    (OUT / "README.md").write_text(
        "# CaRR-3D minimal framework\n\n"
        "- The posed RGB-D stream is intentionally outside panel (a).\n"
        "- Panel (a) contains only the two frozen perception paths.\n"
        "- Panel (b) summarizes the three causal, bounded cross-keyframe states.\n"
        "- Panel (c) shows direct online assembly of the current output.\n"
        "- All raster inserts are measured project assets; the flowchart does not invent detections.\n",
        encoding="utf-8",
    )
    print(png)
    print(pdf)
    print(svg)


if __name__ == "__main__":
    build()
