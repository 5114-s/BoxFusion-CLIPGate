#!/usr/bin/env python3
"""Render compact, high-clarity 3D map assets for the overview flowchart.

The exporter reuses the real ScanNet mesh and frozen prediction files behind
the existing scene0432_01 panels.  It changes visualization only: a tighter
camera crop, neutral geometry, stronger box/halo contrast, and vector exports.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import pickle
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection
import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figure_assets/flowchart_clear_3d_maps_20261102"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


FINAL = load_module("render_final_map_asset", ROOT / "tools/render_final_map_asset.py")
NATIVE = load_module("render_native_map_asset", ROOT / "tools/render_native_map_asset.py")

COLORS = {
    "native": "#1769E8",
    "plr": "#F26B21",
    "calr": "#D18A00",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def crop_alpha(path: Path, padding: int = 20) -> None:
    if path.suffix.lower() != ".png":
        return
    with Image.open(path) as image:
        rgba = image.convert("RGBA")
        alpha_box = rgba.getchannel("A").getbbox()
        if alpha_box is None:
            return
        box = (
            max(0, alpha_box[0] - padding),
            max(0, alpha_box[1] - padding),
            min(rgba.width, alpha_box[2] + padding),
            min(rgba.height, alpha_box[3] + padding),
        )
        rgba.crop(box).save(path, dpi=(600, 600))


def shared_limits(groups: dict[str, list[np.ndarray]]) -> tuple[np.ndarray, float, tuple[float, float]]:
    points = np.concatenate([np.concatenate(boxes, axis=0) for boxes in groups.values() if boxes], axis=0)
    lo = points.min(axis=0)
    hi = points.max(axis=0)
    center = 0.5 * (lo + hi)
    # Square XY framing remains stable when the assets are exchanged in Draw.io.
    span = float(max(hi[0] - lo[0], hi[1] - lo[1])) * 0.67
    zlim = (float(lo[2] - 0.28), float(hi[2] + 0.32))
    return center, span, zlim


def render(
    stem: Path,
    xyz: np.ndarray,
    faces: np.ndarray,
    groups: dict[str, list[np.ndarray]],
    limits: tuple[np.ndarray, float, tuple[float, float]],
) -> None:
    center, span, zlim = limits
    mpl.rcParams.update({
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    })
    fig = plt.figure(figsize=(6.2, 6.0), dpi=600)
    ax = fig.add_subplot(111, projection="3d")

    # A cool, low-contrast mesh keeps the scene recognizable without competing
    # with the output records at icon scale.
    mesh = Poly3DCollection(
        xyz[faces],
        facecolors="#C8D0DA",
        edgecolors="none",
        linewidths=0,
        alpha=0.34,
        rasterized=True,
    )
    ax.add_collection3d(mesh)

    for source in ("native", "plr", "calr"):
        boxes = groups.get(source, [])
        if not boxes:
            continue
        segments = FINAL.box_segments(boxes)
        halo = Line3DCollection(
            segments,
            colors="#FFFFFF",
            linewidths=7.2,
            alpha=0.98,
        )
        halo.set_sort_zpos(1e6)
        ax.add_collection3d(halo)
        lines = Line3DCollection(
            segments,
            colors=COLORS[source],
            linewidths=4.1,
            alpha=1.0,
        )
        lines.set_sort_zpos(1e6 + 1)
        ax.add_collection3d(lines)

    ax.set_xlim(center[0] - span, center[0] + span)
    ax.set_ylim(center[1] - span, center[1] + span)
    ax.set_zlim(*zlim)
    ax.set_box_aspect((1, 1, 0.52))
    ax.view_init(elev=68.0, azim=-88.0)
    ax.set_axis_off()
    ax.set_facecolor((1, 1, 1, 0))
    fig.patch.set_alpha(0)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

    for suffix in (".png", ".pdf", ".svg"):
        path = stem.with_suffix(suffix)
        kwargs = dict(transparent=True, bbox_inches="tight", pad_inches=0.015)
        if suffix == ".png":
            kwargs["dpi"] = 600
        fig.savefig(path, **kwargs)
        crop_alpha(path)
    plt.close(fig)


def render_colored(
    stem: Path,
    xyz: np.ndarray,
    rgb: np.ndarray,
    faces: np.ndarray,
    groups: dict[str, list[np.ndarray]],
    limits: tuple[np.ndarray, float, tuple[float, float]],
    elev: float,
    azim: float,
) -> None:
    """Render a real-color cutaway map for small flowchart panels."""
    center, span, zlim = limits
    face_center = xyz[faces].mean(axis=1)
    keep = (
        (np.abs(face_center[:, 0] - center[0]) <= 1.05 * span)
        & (np.abs(face_center[:, 1] - center[1]) <= 1.05 * span)
        & (face_center[:, 2] >= zlim[0] - 0.05)
        & (face_center[:, 2] <= zlim[1] + 0.70)
    )
    visible_faces = faces[keep]
    face_rgb = np.clip(0.92 * rgb[visible_faces].mean(axis=1) + 0.08, 0.0, 1.0)

    fig = plt.figure(figsize=(6.5, 6.0), dpi=600)
    ax = fig.add_subplot(111, projection="3d")
    ax.add_collection3d(Poly3DCollection(
        xyz[visible_faces],
        facecolors=face_rgb,
        edgecolors="none",
        linewidths=0,
        alpha=0.90,
        rasterized=True,
    ))

    for source in ("native", "plr", "calr"):
        boxes = groups.get(source, [])
        if not boxes:
            continue
        segments = FINAL.box_segments(boxes)
        halo = Line3DCollection(segments, colors="#FFFFFF", linewidths=6.8, alpha=0.98)
        halo.set_sort_zpos(1e6)
        ax.add_collection3d(halo)
        lines = Line3DCollection(segments, colors=COLORS[source], linewidths=3.8, alpha=1.0)
        lines.set_sort_zpos(1e6 + 1)
        ax.add_collection3d(lines)

    ax.set_xlim(center[0] - span, center[0] + span)
    ax.set_ylim(center[1] - span, center[1] + span)
    ax.set_zlim(*zlim)
    ax.set_box_aspect((1, 1, 0.55))
    ax.view_init(elev=elev, azim=azim)
    ax.set_axis_off()
    ax.set_facecolor((1, 1, 1, 0))
    fig.patch.set_alpha(0)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

    for suffix in (".png", ".pdf", ".svg"):
        path = stem.with_suffix(suffix)
        kwargs = dict(transparent=True, bbox_inches="tight", pad_inches=0.015)
        if suffix == ".png":
            kwargs["dpi"] = 600
        fig.savefig(path, **kwargs)
        crop_alpha(path)
    plt.close(fig)


def comparison_preview(path: Path, names: list[tuple[str, Path]]) -> None:
    canvas = Image.new("RGB", (2800, 2500), "white")
    from PIL import ImageDraw, ImageFont
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 54)
    for index, (label, image_path) in enumerate(names):
        image = Image.open(image_path).convert("RGBA")
        image.thumbnail((1300, 1040), Image.Resampling.LANCZOS)
        x = 50 + (index % 2) * 1375
        y = 80 + (index // 2) * 1200
        draw.text((x, y), label, fill="#17263A", font=font)
        canvas.paste(image, (x + (1300 - image.width) // 2, y + 100), image)
    canvas.save(path, dpi=(300, 300))


def load_native_boxes(matrix: np.ndarray) -> list[np.ndarray]:
    payload = pickle.load(NATIVE.PREDICTION.open("rb"))[0]
    return [NATIVE.transform(np.asarray(row[1], np.float64), matrix) for row in payload]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    matrix = FINAL.axis_alignment()
    xyz, rgb, faces = FINAL.load_mesh(matrix)

    # Preserve exactly the readable subset used by the existing final-output
    # panel: all native rows, one proposal recovery, and three anchor recoveries.
    rows = FINAL.load_rows(FINAL.PREDICTION)
    final_boxes = [FINAL.transform(box, matrix) for box, _ in rows]
    native_count = len(FINAL.load_rows(FINAL.NATIVE_PREFIX))
    plr_count = len(FINAL.load_rows(FINAL.M1P_PREFIX)) - native_count
    final_groups = {
        "native": final_boxes[:native_count],
        "plr": [final_boxes[9]],
        "calr": [final_boxes[index] for index in (21, 23, 26)],
    }

    native_boxes = load_native_boxes(matrix)
    native_groups = {"native": native_boxes, "plr": [], "calr": []}

    # Use a common crop based on the union so both panels can be swapped or
    # compared without an apparent camera change.
    union = {
        "native": final_groups["native"] + native_groups["native"],
        "plr": final_groups["plr"],
        "calr": final_groups["calr"],
    }
    limits = shared_limits(union)
    render(OUT / "native_map_clear", xyz, faces, native_groups, limits)
    render(OUT / "final_detections_clear", xyz, faces, final_groups, limits)
    colored_outputs = []
    for view_name, elev, azim in (
        ("colored_topdown", 68.0, -88.0),
        ("colored_oblique", 42.0, -65.0),
    ):
        native_stem = OUT / f"native_map_{view_name}"
        final_stem = OUT / f"final_detections_{view_name}"
        render_colored(native_stem, xyz, rgb, faces, native_groups, limits, elev, azim)
        render_colored(final_stem, xyz, rgb, faces, final_groups, limits, elev, azim)
        colored_outputs.extend([
            (f"Native map — {view_name.replace('colored_', '', 1)}", native_stem.with_suffix(".png")),
            (f"Final detections — {view_name.replace('colored_', '', 1)}", final_stem.with_suffix(".png")),
        ])
    comparison_preview(OUT / "colored_map_variants_preview.png", colored_outputs)

    sources = [
        FINAL.MESH,
        FINAL.META,
        FINAL.PREDICTION,
        FINAL.M1P_PREFIX,
        FINAL.NATIVE_PREFIX,
        NATIVE.PREDICTION,
    ]
    manifest = {
        "schema": "recar3d.flowchart_clear_maps.v1",
        "scene": FINAL.SCENE,
        "model_rerun": False,
        "visualization_only": True,
        "camera": {"elev": 68.0, "azim": -88.0, "shared_between_panels": True},
        "colored_camera_variants": {
            "topdown": {"elev": 68.0, "azim": -88.0},
            "oblique": {"elev": 42.0, "azim": -65.0},
        },
        "native_map": {
            "source": "controlled Base native-only terminal map",
            "box_count": len(native_boxes),
            "colors": {"native": COLORS["native"]},
        },
        "final_detections": {
            "source": "existing illustrative terminal-output subset",
            "native_count": len(final_groups["native"]),
            "plr_count": len(final_groups["plr"]),
            "calr_count": len(final_groups["calr"]),
            "full_output": False,
            "subset_reason": "small-panel legibility; identical rows to the previous panel",
            "colors": COLORS,
        },
        "sources": [{"path": str(p), "sha256": sha256(p)} for p in sources],
    }
    (OUT / "source_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (OUT / "README.md").write_text(
        "# Clear 3D map assets for the flowchart\n\n"
        "- `native_map_clear`: frozen controlled-Base native-only map; five blue boxes.\n"
        "- `final_detections_clear`: the same illustrative output rows as the previous final-map panel; blue=native, orange=PLR, amber=CALR.\n"
        "- Both panels use the same real ScanNet scene, camera, crop, and neutral visualization mesh.\n"
        "- `*_colored_topdown` and `*_colored_oblique` retain the real ScanNet RGB surface; the oblique version is recommended for the method overview.\n"
        "- PNG is 600 dpi. PDF/SVG keep the box edges as vectors. Use SVG in Draw.io when possible.\n"
        "- The final panel is an illustrative subset for legibility, not the complete output list.\n",
        encoding="utf-8",
    )
    print(OUT)


if __name__ == "__main__":
    main()
