#!/usr/bin/env python3
"""Build a real-data image pack for the Frozen Native Online Mapper panel.

Every photographic or 3D insert comes from ScanNet ``scene0432_01`` and the
frozen controlled-Base prediction.  The script does not run a model and does
not invent geometry.  It only copies existing RGB-D/projection assets and
renders one native fused box together with the real scan surface inside it.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import pickle
import shutil
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection
import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "figure_assets/pipeline_candidate_visuals_scene0432_01"
RGBD = ROOT / "figure_assets/rgbd_input_scannet_scene0432_01"
MAPS = ROOT / "figure_assets/flowchart_clear_3d_maps_20261102"
OUT = ROOT / "figure_assets/native_online_mapper_real_pack_scene0432_01_20261102"
PREDICTION = ROOT / (
    "results/recar3d_final_scannet_factorial100_20260926/base/"
    "scene0432_01_boxes.pkl"
)


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


RENDER = load_module("render_final_map_asset", ROOT / "tools/render_final_map_asset.py")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def copy(source: Path, name: str) -> Path:
    target = OUT / name
    shutil.copy2(source, target)
    return target


def crop_alpha(path: Path, padding: int = 24) -> None:
    with Image.open(path) as image:
        rgba = image.convert("RGBA")
        bounds = rgba.getchannel("A").getbbox()
        if bounds is None:
            return
        bounds = (
            max(0, bounds[0] - padding),
            max(0, bounds[1] - padding),
            min(rgba.width, bounds[2] + padding),
            min(rgba.height, bounds[3] + padding),
        )
        rgba.crop(bounds).save(path, dpi=(600, 600))


def render_pfo_object(path: Path) -> dict[str, object]:
    """Render the real ottoman surface and its frozen fused native box."""
    matrix = RENDER.axis_alignment()
    xyz, rgb, faces = RENDER.load_mesh(matrix)
    rows = pickle.load(PREDICTION.open("rb"))[0]

    # Native row 0 is the central ottoman shown in the three reliable views.
    box = RENDER.transform(np.asarray(rows[0][1], np.float64), matrix)
    p0 = box[0]
    basis = np.column_stack((box[1] - p0, box[3] - p0, box[4] - p0))
    local = np.linalg.solve(basis, (xyz - p0).T).T
    vertex_inside = np.all((local >= -0.04) & (local <= 1.04), axis=1)
    face_center = local[faces].mean(axis=1)
    face_inside = np.all((face_center >= -0.05) & (face_center <= 1.05), axis=1)
    selected_faces = faces[np.any(vertex_inside[faces], axis=1) & face_inside]

    mpl.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42})
    fig = plt.figure(figsize=(6.4, 5.0), dpi=450)
    ax = fig.add_subplot(111, projection="3d")
    face_rgb = np.clip(rgb[selected_faces].mean(axis=1) * 1.03, 0.0, 1.0)
    ax.add_collection3d(Poly3DCollection(
        xyz[selected_faces],
        facecolors=face_rgb,
        edgecolors="none",
        linewidths=0,
        alpha=0.98,
        rasterized=True,
    ))

    segments = RENDER.box_segments([box])
    halo = Line3DCollection(segments, colors="#FFFFFF", linewidths=5.0, alpha=0.92)
    halo.set_sort_zpos(1e6)
    ax.add_collection3d(halo)
    lines = Line3DCollection(segments, colors="#1769E8", linewidths=2.8, alpha=1.0)
    lines.set_sort_zpos(1e6 + 1)
    ax.add_collection3d(lines)

    low, high = box.min(axis=0), box.max(axis=0)
    center = 0.5 * (low + high)
    span = float(max(high - low)) * 0.59
    ax.set_xlim(center[0] - span, center[0] + span)
    ax.set_ylim(center[1] - span, center[1] + span)
    ax.set_zlim(low[2] - 0.10, high[2] + 0.10)
    ax.set_box_aspect((1, 1, 0.66))
    ax.view_init(elev=22.0, azim=50.0)
    ax.set_axis_off()
    ax.set_facecolor((1, 1, 1, 0))
    fig.patch.set_alpha(0)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    fig.savefig(path, transparent=True, bbox_inches="tight", pad_inches=0.01, dpi=450)
    plt.close(fig)
    crop_alpha(path)
    return {
        "prediction_row": 0,
        "score": float(rows[0][2]),
        "selected_mesh_faces": int(len(selected_faces)),
        "camera": {"elev": 22.0, "azim": 50.0},
    }


def make_top3_strip(paths: list[Path], target: Path) -> None:
    images = [Image.open(path).convert("RGB") for path in paths]
    gap = 24
    height = max(image.height for image in images)
    width = sum(image.width for image in images) + gap * (len(images) - 1)
    canvas = Image.new("RGB", (width, height), "white")
    x = 0
    for image in images:
        canvas.paste(image, (x, (height - image.height) // 2))
        x += image.width + gap
    canvas.save(target, dpi=(600, 600))


def fit(image: Image.Image, size: tuple[int, int], background=(255, 255, 255, 0)) -> Image.Image:
    image = image.convert("RGBA")
    image.thumbnail(size, Image.Resampling.LANCZOS)
    canvas = Image.new("RGBA", size, background)
    canvas.alpha_composite(image, ((size[0] - image.width) // 2, (size[1] - image.height) // 2))
    return canvas


def make_preview(target: Path, assets: dict[str, Path]) -> None:
    width, height = 4000, 1080
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 42)
    bold = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 47)

    columns = [
        (40, 720, "RGB-D input", [assets["rgbd"]]),
        (830, 650, "Per-view candidate", [assets["candidate"]]),
        (1540, 980, "Reliable views (Top-3)", [assets["strip"]]),
        (2610, 610, "PFO fused native box", [assets["pfo"]]),
        (3290, 660, "Native map N^(T)", [assets["map"]]),
    ]
    for x, w, title, image_paths in columns:
        draw.text((x, 24), title, fill="#16263A", font=bold)
        image = Image.open(image_paths[0])
        panel = fit(image, (w, 820))
        canvas.paste(panel, (x, 110), panel)
    for x in (775, 1490, 2550, 3230):
        draw.line((x, 475, x + 25, 475), fill="#68809B", width=8)
        draw.polygon([(x + 25, 458), (x + 25, 492), (x + 50, 475)], fill="#68809B")
    draw.text(
        (40, 1008),
        "All inserts: ScanNet scene0432_01; blue = frozen native branch.",
        fill="#43556D",
        font=font,
    )
    canvas.save(target, dpi=(300, 300))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    sources = {
        "rgbd": RGBD / "frame_0000_rgbd_inset.png",
        "rgb": RGBD / "frame_0000_rgb.png",
        "depth": RGBD / "frame_0000_depth_gray.png",
        "candidate": SOURCE / "02a_reliable_view_top1_frame0000.png",
        "view1": SOURCE / "02a_reliable_view_top1_frame0000.png",
        "view2": SOURCE / "02b_reliable_view_top2_frame0025.png",
        "view3": SOURCE / "02c_reliable_view_top3_frame0050.png",
        "map": MAPS / "native_map_colored_topdown.png",
    }
    outputs = {
        "rgbd": copy(sources["rgbd"], "01_rgbd_keyframe.png"),
        "rgb": copy(sources["rgb"], "01a_rgb_keyframe.png"),
        "depth": copy(sources["depth"], "01b_depth_keyframe.png"),
        "candidate": copy(sources["candidate"], "02_per_view_candidate_native_blue.png"),
        "view1": copy(sources["view1"], "03_reliable_view_1_frame0000.png"),
        "view2": copy(sources["view2"], "04_reliable_view_2_frame0025.png"),
        "view3": copy(sources["view3"], "05_reliable_view_3_frame0050.png"),
        "map": copy(sources["map"], "08_native_map_native_only.png"),
    }
    pfo_meta = render_pfo_object(OUT / "07_pfo_fused_native_box_real_mesh.png")
    outputs["pfo"] = OUT / "07_pfo_fused_native_box_real_mesh.png"
    make_top3_strip(
        [outputs["view1"], outputs["view2"], outputs["view3"]],
        OUT / "06_reliable_view_top3_strip.png",
    )
    outputs["strip"] = OUT / "06_reliable_view_top3_strip.png"
    make_preview(OUT / "00_pack_preview.png", outputs)

    manifest = {
        "schema": "recar3d.native_online_mapper_real_pack.v1",
        "scene_id": "scene0432_01",
        "model_rerun": False,
        "all_content_from_real_experiment_data": True,
        "box_color_semantics": {"blue": "frozen native branch"},
        "time_note": "native map is the terminal online state N^(T)",
        "pfo_example": pfo_meta,
        "sources": [
            {"path": str(path), "sha256": sha256(path)} for path in sources.values()
        ] + [
            {"path": str(RENDER.MESH), "sha256": sha256(RENDER.MESH)},
            {"path": str(RENDER.META), "sha256": sha256(RENDER.META)},
            {"path": str(PREDICTION), "sha256": sha256(PREDICTION)},
        ],
        "outputs": [
            {"path": str(path), "sha256": sha256(path)}
            for path in sorted(OUT.glob("*.png"))
        ],
    }
    (OUT / "source_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (OUT / "README.md").write_text(
        "# Real image pack for Frozen Native Online Mapper\n\n"
        "All inserts use ScanNet `scene0432_01` and the frozen controlled-Base native branch.\n\n"
        "- `01_rgbd_keyframe.png`: input RGB-D keyframe.\n"
        "- `02_per_view_candidate_native_blue.png`: one native per-view 3D candidate.\n"
        "- `03`--`05`: three reliable observations of the same ottoman.\n"
        "- `06_reliable_view_top3_strip.png`: ready-to-place non-overlapping strip.\n"
        "- `07_pfo_fused_native_box_real_mesh.png`: real scan surface inside the frozen fused native box.\n"
        "- `08_native_map_native_only.png`: native-only terminal map; five blue boxes.\n"
        "- `00_pack_preview.png`: placement preview.\n\n"
        "The map is `N^(T)`. If the paper labels it `N^(t)`, export a matching prefix snapshot instead.\n",
        encoding="utf-8",
    )
    print(OUT)


if __name__ == "__main__":
    main()
