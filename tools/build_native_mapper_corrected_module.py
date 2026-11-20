#!/usr/bin/env python3
"""Build a scientifically conservative native-online-mapper module panel.

The panel uses real ScanNet RGB observations and real prefix native-map logs.
PFO is drawn as an operation block because the audited run did not persist a
per-update receipt containing selected view indices/weights and pre/post-PFO
boxes.  Consequently, this exporter never labels a terminal box as a PFO
intermediate result.
"""

from __future__ import annotations

import base64
import html
import importlib.util
import io
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection
import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
SCENE = "scene0432_01"
OUT = ROOT / "figure_assets/native_online_mapper_corrected_scene0432_01_20261103"
PACK = ROOT / "figure_assets/native_online_mapper_real_pack_scene0432_01_20261102"
SOURCE = ROOT / "figure_assets/pipeline_candidate_visuals_scene0432_01"
KFMAP = ROOT / (
    "reports/paper_matched_runtime_full100_20260916/final/scene0432_01/"
    "native_diagnostics/scene0432_01_pvq_kfmap.jsonl"
)


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


RENDER = load_module("render_native_map", ROOT / "tools/render_final_map_asset.py")


KFMAP_EDGES = np.asarray([
    (0, 1), (1, 2), (2, 3), (3, 0),
    (4, 5), (5, 6), (6, 7), (7, 4),
    (0, 4), (1, 5), (2, 6), (3, 7),
], dtype=np.int64)


def read_prefix(keyframe_id: int) -> dict:
    for line in KFMAP.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if row.get("type") == "kf_map" and row.get("keyframe_id") == keyframe_id:
            return row
    raise KeyError(f"keyframe {keyframe_id} missing from {KFMAP}")


def crop_alpha(path: Path, padding: int = 18) -> None:
    with Image.open(path) as image:
        rgba = image.convert("RGBA")
        bounds = rgba.getchannel("A").getbbox()
        if bounds is None:
            return
        crop = (
            max(0, bounds[0] - padding),
            max(0, bounds[1] - padding),
            min(rgba.width, bounds[2] + padding),
            min(rgba.height, bounds[3] + padding),
        )
        rgba.crop(crop).save(path, dpi=(600, 600))


def render_prefix_map(keyframe_id: int, target: Path) -> dict:
    row = read_prefix(keyframe_id)
    matrix = RENDER.axis_alignment()
    xyz, rgb, faces = RENDER.load_mesh(matrix)
    boxes = [
        RENDER.transform(np.asarray(box, np.float64), matrix)
        for box in row["boxes"]
    ]

    mpl.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42})
    fig = plt.figure(figsize=(7.6, 5.3), dpi=360)
    ax = fig.add_subplot(111, projection="3d")
    face_rgb = rgb[faces].mean(axis=1)
    face_rgb = 0.43 * face_rgb + 0.57
    ax.add_collection3d(Poly3DCollection(
        xyz[faces], facecolors=face_rgb, edgecolors="none", linewidths=0,
        alpha=0.52, rasterized=True,
    ))

    # The JSON diagnostic rounds corners to millimetres and preserves the
    # canonical bottom-then-top corner order.  Use that stored topology rather
    # than the exact-geometry topology recovery used for unrounded predictions.
    segments = np.concatenate([box[KFMAP_EDGES] for box in boxes], axis=0)
    halo = Line3DCollection(segments, colors="#FFFFFF", linewidths=3.9, alpha=0.90)
    halo.set_sort_zpos(1e6)
    ax.add_collection3d(halo)
    lines = Line3DCollection(segments, colors="#1769E8", linewidths=2.0, alpha=1.0)
    lines.set_sort_zpos(1e6 + 1)
    ax.add_collection3d(lines)

    lo, hi = xyz.min(axis=0), xyz.max(axis=0)
    center = 0.5 * (lo + hi)
    span = max(hi[0] - lo[0], hi[1] - lo[1]) * 0.54
    ax.set_xlim(center[0] - span, center[0] + span)
    ax.set_ylim(center[1] - span, center[1] + span)
    ax.set_zlim(lo[2] - 0.02, hi[2] + 0.10)
    ax.set_box_aspect((1, 1, 0.44))
    ax.view_init(elev=75.0, azim=-88.0)
    ax.set_axis_off()
    ax.set_facecolor((1, 1, 1, 0))
    fig.patch.set_alpha(0)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    fig.savefig(target, transparent=True, bbox_inches="tight", pad_inches=0.01, dpi=360)
    plt.close(fig)
    crop_alpha(target)
    return {
        "keyframe_id": keyframe_id,
        "record_count": int(row["n"]),
        "init_ids": row["init_ids"],
    }


def image_uri(path: Path) -> str:
    mime = "image/png"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def esc(text: str) -> str:
    return html.escape(text, quote=True)


def image_box(path: Path, x: int, y: int, w: int, h: int, clip_id: str) -> str:
    with Image.open(path) as im:
        iw, ih = im.size
    scale = max(w / iw, h / ih)
    dw, dh = iw * scale, ih * scale
    dx, dy = x + (w - dw) / 2, y + (h - dh) / 2
    return f'''\
    <defs><clipPath id="{clip_id}"><rect x="{x}" y="{y}" width="{w}" height="{h}" rx="24"/></clipPath></defs>
    <rect x="{x}" y="{y}" width="{w}" height="{h}" rx="24" fill="#ffffff" stroke="#b8c7d9" stroke-width="5"/>
    <image href="{image_uri(path)}" x="{dx:.2f}" y="{dy:.2f}" width="{dw:.2f}" height="{dh:.2f}" preserveAspectRatio="none" clip-path="url(#{clip_id})"/>
    '''


def arrow(x1: int, y1: int, x2: int, y2: int, marker: str = "arrow") -> str:
    return (
        f'<path d="M {x1} {y1} L {x2} {y2}" fill="none" '
        f'stroke="#6f8094" stroke-width="14" marker-end="url(#{marker})"/>'
    )


def build_svg(target: Path, assets: dict[str, Path]) -> None:
    width, height = 5000, 1250
    svg = [f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
    <defs>
      <linearGradient id="panel" x1="0" y1="0" x2="0" y2="1">
        <stop offset="0" stop-color="#f7fbff"/>
        <stop offset="1" stop-color="#edf5fd"/>
      </linearGradient>
      <linearGradient id="op" x1="0" y1="0" x2="1" y2="1">
        <stop offset="0" stop-color="#f6f1ff"/>
        <stop offset="1" stop-color="#eee6ff"/>
      </linearGradient>
      <marker id="arrow" markerWidth="20" markerHeight="20" refX="16" refY="7" orient="auto" markerUnits="strokeWidth">
        <path d="M 0 0 L 18 7 L 0 14 z" fill="#6f8094"/>
      </marker>
      <style>
        .title {{ font-family: Arial, Helvetica, sans-serif; font-size: 96px; font-weight: 700; fill: #101820; }}
        .stage {{ font-family: Arial, Helvetica, sans-serif; font-size: 50px; font-weight: 700; fill: #17263a; }}
        .label {{ font-family: Arial, Helvetica, sans-serif; font-size: 43px; font-weight: 650; fill: #17263a; }}
        .small {{ font-family: Arial, Helvetica, sans-serif; font-size: 31px; font-weight: 500; fill: #52657a; }}
        .op-title {{ font-family: Arial, Helvetica, sans-serif; font-size: 45px; font-weight: 700; fill: #4c178c; }}
        .op-small {{ font-family: Arial, Helvetica, sans-serif; font-size: 31px; font-weight: 500; fill: #5f4c78; }}
      </style>
    </defs>
    <rect x="24" y="24" width="4952" height="1202" rx="86" fill="url(#panel)" stroke="#1769e8" stroke-width="12"/>
    <text x="2500" y="145" text-anchor="middle" class="title">Inherited Native Online Mapper</text>
    <rect x="4290" y="58" width="550" height="110" rx="34" fill="#e7f1fc" stroke="#8db5df" stroke-width="5"/>
    <text x="4565" y="132" text-anchor="middle" class="label">Parameter-Frozen</text>
    ''']

    svg.append(image_box(assets["candidate"], 90, 340, 650, 500, "candidate"))
    svg.append('<text x="415" y="915" text-anchor="middle" class="label">Current Candidates C<tspan baseline-shift="sub" font-size="30">t</tspan></text>')
    svg.append('<text x="415" y="968" text-anchor="middle" class="small">current keyframe · frame 50</text>')
    svg.append('<text x="1110" y="235" text-anchor="middle" class="label">Previous Map N<tspan baseline-shift="super" font-size="29">(t−1)</tspan></text>')
    svg.append(image_box(assets["prev_map"], 850, 260, 520, 355, "prevmap"))
    svg.append('<text x="1110" y="660" text-anchor="middle" class="small">real prefix snapshot · keyframe 25</text>')

    # Two online inputs converge at association.
    svg.append(arrow(740, 750, 825, 750))
    svg.append('<path d="M 1110 615 L 1110 680" fill="none" stroke="#6f8094" stroke-width="12" marker-end="url(#arrow)"/>')
    svg.append('<rect x="850" y="700" width="520" height="235" rx="48" fill="#eaf3fd" stroke="#83add8" stroke-width="7"/>')
    svg.append('<text x="1110" y="800" text-anchor="middle" class="label">Native Association</text>')
    svg.append('<text x="1110" y="875" text-anchor="middle" class="small">3D NMS · 2D Matching</text>')
    svg.append(arrow(1370, 815, 1500, 815))

    # Real RGB history; no claim that the saved run logged these exact Top-K indices.
    svg.append('<text x="2115" y="260" text-anchor="middle" class="stage">Selected Reliable Views (K=3)</text>')
    frame_paths = [assets["view1"], assets["view2"], assets["view3"]]
    frame_ids = ["frame 0", "frame 25", "frame 50"]
    for index, (path, frame_id) in enumerate(zip(frame_paths, frame_ids), start=1):
        x = 1510 + (index - 1) * 410
        svg.append(image_box(path, x, 355, 380, 360, f"view{index}"))
        svg.append(f'<circle cx="{x + 45}" cy="400" r="42" fill="#50677e" stroke="#ffffff" stroke-width="6"/>')
        svg.append(f'<text x="{x + 45}" y="416" text-anchor="middle" font-family="Arial" font-size="42" font-weight="700" fill="#ffffff">{index}</text>')
        svg.append(f'<text x="{x + 190}" y="775" text-anchor="middle" class="small">{esc(frame_id)}</text>')
    svg.append('<text x="2115" y="855" text-anchor="middle" class="small">same native identity across observed keyframes</text>')
    svg.append(arrow(2740, 610, 2820, 610))

    svg.append('<rect x="2850" y="445" width="650" height="310" rx="56" fill="url(#op)" stroke="#8b28df" stroke-width="8"/>')
    svg.append('<text x="3175" y="570" text-anchor="middle" class="op-title">PFO Fusion</text>')
    svg.append('<text x="3175" y="650" text-anchor="middle" class="op-small">optimize native geometry</text>')
    svg.append(arrow(3500, 610, 3670, 610))

    svg.append(image_box(assets["next_map"], 3710, 270, 1120, 650, "nextmap"))
    svg.append('<text x="4270" y="985" text-anchor="middle" class="label">Updated Native Map N<tspan baseline-shift="super" font-size="29">(t)</tspan></text>')
    svg.append('<text x="4270" y="1040" text-anchor="middle" class="small">real prefix snapshot · keyframe 50</text>')
    svg.append('<text x="2500" y="1165" text-anchor="middle" class="small">Blue boxes denote the inherited native branch; scene geometry is used only for visualization.</text>')
    svg.append('</svg>')
    target.write_text("\n".join(svg), encoding="utf-8")


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    return ImageFont.truetype(f"/usr/share/fonts/truetype/dejavu/{name}", size)


def centered(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str,
             face: ImageFont.FreeTypeFont, fill: str) -> None:
    box = draw.textbbox((0, 0), text, font=face)
    draw.text((xy[0] - (box[2] - box[0]) / 2, xy[1] - (box[3] - box[1]) / 2),
              text, font=face, fill=fill)


def paste_image(canvas: Image.Image, path: Path, box: tuple[int, int, int, int],
                cover: bool = True, radius: int = 24) -> None:
    x, y, w, h = box
    source = Image.open(path).convert("RGBA")
    if cover:
        scale = max(w / source.width, h / source.height)
    else:
        scale = min(w / source.width, h / source.height)
    size = (max(1, round(source.width * scale)), max(1, round(source.height * scale)))
    source = source.resize(size, Image.Resampling.LANCZOS)
    panel = Image.new("RGBA", (w, h), "white")
    panel.alpha_composite(source, ((w - source.width) // 2, (h - source.height) // 2))
    mask = Image.new("L", (w, h), 0)
    ImageDraw.Draw(mask).rounded_rectangle((0, 0, w - 1, h - 1), radius=radius, fill=255)
    canvas.paste(panel, (x, y), mask)
    ImageDraw.Draw(canvas).rounded_rectangle((x, y, x + w, y + h), radius=radius,
                                             outline="#b8c7d9", width=5)


def draw_arrow(draw: ImageDraw.ImageDraw, start: tuple[int, int], end: tuple[int, int],
               color: str = "#6f8094", width: int = 14) -> None:
    x1, y1 = start
    x2, y2 = end
    draw.line((x1, y1, x2 - 24, y2), fill=color, width=width)
    draw.polygon(((x2 - 30, y2 - 22), (x2, y2), (x2 - 30, y2 + 22)), fill=color)


def build_png(target: Path, assets: dict[str, Path]) -> None:
    canvas = Image.new("RGBA", (5000, 1250), "white")
    draw = ImageDraw.Draw(canvas)
    draw.rounded_rectangle((24, 24, 4976, 1226), radius=86, fill="#f3f8fe",
                           outline="#1769e8", width=12)
    centered(draw, (2500, 108), "Inherited Native Online Mapper", font(96, True), "#101820")
    draw.rounded_rectangle((4290, 58, 4840, 168), radius=34, fill="#e7f1fc",
                           outline="#8db5df", width=5)
    centered(draw, (4565, 112), "Parameter-Frozen", font(43, True), "#17263a")
    paste_image(canvas, assets["candidate"], (90, 340, 650, 500), cover=True)
    centered(draw, (415, 895), "Current Candidates Cₜ", font(43, True), "#17263a")
    centered(draw, (415, 955), "current keyframe · frame 50", font(31), "#52657a")
    centered(draw, (1110, 210), "Previous Map N⁽ᵗ⁻¹⁾", font(43, True), "#17263a")
    paste_image(canvas, assets["prev_map"], (850, 260, 520, 355), cover=False)
    centered(draw, (1110, 650), "real prefix snapshot · keyframe 25", font(31), "#52657a")

    draw_arrow(draw, (740, 750), (825, 750))
    draw.line((1110, 615, 1110, 670), fill="#6f8094", width=12)
    draw.polygon(((1088, 652), (1110, 684), (1132, 652)), fill="#6f8094")
    draw.rounded_rectangle((850, 700, 1370, 935), radius=48, fill="#eaf3fd",
                           outline="#83add8", width=7)
    centered(draw, (1110, 790), "Native Association", font(43, True), "#17263a")
    centered(draw, (1110, 870), "3D NMS · 2D Matching", font(31), "#52657a")
    draw_arrow(draw, (1370, 815), (1500, 815))

    centered(draw, (2115, 235), "Selected Reliable Views (K=3)", font(50, True), "#17263a")
    for index, key in enumerate(("view1", "view2", "view3"), start=1):
        x = 1510 + (index - 1) * 410
        paste_image(canvas, assets[key], (x, 355, 380, 360), cover=True)
        draw.ellipse((x + 3, 358, x + 87, 442), fill="#50677e", outline="white", width=6)
        centered(draw, (x + 45, 399), str(index), font(42, True), "white")
        centered(draw, (x + 190, 765), f"frame {(index - 1) * 25}", font(31), "#52657a")
    centered(draw, (2115, 845), "same native identity across observed keyframes",
             font(31), "#52657a")
    draw_arrow(draw, (2740, 610), (2820, 610))

    draw.rounded_rectangle((2850, 445, 3500, 755), radius=56, fill="#f1eaff",
                           outline="#8b28df", width=8)
    centered(draw, (3175, 545), "PFO Fusion", font(45, True), "#4c178c")
    centered(draw, (3175, 640), "optimize native geometry", font(31), "#5f4c78")
    draw_arrow(draw, (3500, 610), (3670, 610))

    paste_image(canvas, assets["next_map"], (3710, 270, 1120, 650), cover=False)
    centered(draw, (4270, 965), "Updated Native Map N⁽ᵗ⁾", font(43, True), "#17263a")
    centered(draw, (4270, 1030), "real prefix snapshot · keyframe 50", font(31), "#52657a")
    centered(draw, (2500, 1145),
             "Blue boxes denote the inherited native branch; scene geometry is used only for visualization.",
             font(31), "#52657a")
    canvas.convert("RGB").save(target, dpi=(600, 600), quality=95)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    prev_meta = render_prefix_map(25, OUT / "01_previous_native_map_kf0025.png")
    next_meta = render_prefix_map(50, OUT / "02_updated_native_map_kf0050.png")
    assets = {
        "candidate": SOURCE / "02c_reliable_view_top3_frame0050.png",
        "view1": SOURCE / "02a_reliable_view_top1_frame0000.png",
        "view2": SOURCE / "02b_reliable_view_top2_frame0025.png",
        "view3": SOURCE / "02c_reliable_view_top3_frame0050.png",
        "prev_map": OUT / "01_previous_native_map_kf0025.png",
        "next_map": OUT / "02_updated_native_map_kf0050.png",
    }
    svg = OUT / "native_online_mapper_pfo_only.svg"
    png = OUT / "native_online_mapper_pfo_only.png"
    pdf = OUT / "native_online_mapper_pfo_only.pdf"
    build_svg(svg, assets)
    build_png(png, assets)
    Image.open(png).convert("RGB").save(pdf, "PDF", resolution=600.0)
    manifest = {
        "scene_id": SCENE,
        "current_keyframe": 50,
        "previous_map": prev_meta,
        "updated_map": next_meta,
        "pfo_rendering_policy": (
            "PFO is represented as an operation block. No terminal prediction "
            "is presented as a per-update PFO intermediate."
        ),
        "observation_images": [str(assets[f"view{i}"]) for i in (1, 2, 3)],
        "observation_note": (
            "Real views of the same object are used to illustrate observation "
            "history. The audited run did not persist per-update Top-K indices "
            "or weights, so the images are not claimed as an exact PFO receipt."
        ),
        "outputs": [str(svg), str(png), str(pdf)],
    }
    (OUT / "source_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (OUT / "README.md").write_text(
        "# Corrected native online mapper panel\n\n"
        "This panel uses real ScanNet `scene0432_01` RGB observations and real "
        "native prefix-map snapshots at keyframes 25 and 50. The PFO stage is "
        "drawn as an operation block because the existing diagnostics do not "
        "contain a per-update PFO receipt with exact selected view indices, "
        "weights, and pre/post optimization boxes.\n\n"
        "The former `07_pfo_fused_native_box_real_mesh.png` is deliberately not "
        "used. It was a terminal native box with local mesh, not an audited PFO "
        "intermediate.\n",
        encoding="utf-8",
    )
    print(svg)
    print(png)
    print(pdf)


if __name__ == "__main__":
    main()
