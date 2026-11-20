#!/usr/bin/env python3
"""Render the CaRR-3D before/after motivation pair from measured outputs.

The scene background, camera, cuboids, confidence values, and ranks all come
from saved ScanNet/BoxFusion experiment artifacts.  Matplotlib only projects
the measured 3D cuboids and draws the colored lines and text; no generative
image model or manual box editing is used.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = ROOT / "figure_assets/real_3d_motivation_20261026"
DEFAULT_PREDICTIONS = (
    ROOT / "reports/scannet_hcsr_current_zero_vs_max_20261029/predictions"
)
DEFAULT_OUTPUT = ROOT / "figure_assets/motivation_candidate_evidence_code_20261111"

SCENE = "scene0426_03"
RERANKED_NATIVE_INDEX = 12
CONTEXT_NATIVE_INDEX = 17

# Crop in the pixel coordinate system of colored_mesh_perspective.png.
# It contains the complete context record, reranked record, and recovery box.
CROP = (340, 120, 1820, 1220)  # left, top, right, bottom

BLUE = "#0879E6"
PURPLE = "#6D24E8"
ORANGE = "#FF6A00"
MISSING = "#FFFFFF"
MISSING_SHADOW = "#8A8F98"

CUBOID_EDGES = (
    (0, 1), (1, 2), (2, 3), (3, 0),
    (4, 5), (5, 6), (6, 7), (7, 4),
    (0, 4), (1, 5), (2, 6), (3, 7),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--prediction-dir", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dpi", type=int, default=200)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def make_projector(camera: dict):
    position = np.asarray(camera["position"], dtype=float)
    target = np.asarray(camera["look_at"], dtype=float)
    world_up = np.asarray(camera["up"], dtype=float)

    forward = target - position
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, world_up)
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    up /= np.linalg.norm(up)

    focal = float(camera["focal_pixels"])
    cx, cy = camera["principal_point_pixels"]

    def project(points: np.ndarray) -> np.ndarray:
        delta = np.asarray(points, dtype=float) - position
        depth = delta @ forward
        if np.any(depth <= 0):
            raise ValueError("A cuboid corner lies behind the fixed camera")
        return np.column_stack(
            (
                cx + focal * (delta @ right) / depth,
                cy - focal * (delta @ up) / depth,
            )
        )

    return project


def load_arm(prediction_dir: Path, arm: str) -> list:
    path = prediction_dir / arm / f"{SCENE}_boxes.pkl"
    with path.open("rb") as handle:
        return pickle.load(handle)[0]


def load_reranking_measurement(prediction_dir: Path) -> dict:
    native = load_arm(prediction_dir, "native_p100")
    historical = load_arm(prediction_dir, "historical_max_p100")
    if len(native) != len(historical):
        raise ValueError("Native and HSR outputs contain different record counts")

    native_scores = np.asarray([record[2] for record in native], dtype=float)
    historical_scores = np.asarray([record[2] for record in historical], dtype=float)
    index = RERANKED_NATIVE_INDEX

    native_ranks = np.argsort(np.argsort(-native_scores)) + 1
    historical_ranks = np.argsort(np.argsort(-historical_scores)) + 1
    return {
        "record_index": index,
        "record_count": int(len(native_scores)),
        "before_score": float(native_scores[index]),
        "after_score": float(historical_scores[index]),
        "before_rank": int(native_ranks[index]),
        "after_rank": int(historical_ranks[index]),
    }


def crop_coordinates(xy: np.ndarray) -> np.ndarray:
    left, top, _, _ = CROP
    return np.asarray(xy, dtype=float) - np.asarray([left, top], dtype=float)


def draw_cuboid(
    ax: plt.Axes,
    xy: np.ndarray,
    color: str,
    *,
    linewidth: float = 1.8,
    dashed: bool = False,
    white_halo: bool = True,
) -> None:
    linestyle = (0, (3.0, 2.3)) if dashed else "-"
    dash_capstyle = "round" if dashed else "butt"
    for start, end in CUBOID_EDGES:
        xs = [xy[start, 0], xy[end, 0]]
        ys = [xy[start, 1], xy[end, 1]]
        if dashed:
            # A narrow gray underlay keeps white dashed edges visible on walls.
            ax.plot(
                xs,
                ys,
                color=MISSING_SHADOW,
                linewidth=linewidth + 1.15,
                linestyle=linestyle,
                dash_capstyle=dash_capstyle,
                alpha=0.72,
                zorder=4,
            )
        elif white_halo:
            ax.plot(
                xs,
                ys,
                color="white",
                linewidth=linewidth + 1.15,
                solid_capstyle="round",
                alpha=0.94,
                zorder=4,
            )
        ax.plot(
            xs,
            ys,
            color=color,
            linewidth=linewidth,
            linestyle=linestyle,
            solid_capstyle="round",
            dash_capstyle=dash_capstyle,
            zorder=5,
        )


def add_label(
    ax: plt.Axes,
    xy: np.ndarray,
    text: str,
    color: str,
    *,
    y_offset: float = 32.0,
) -> None:
    center_x = float((xy[:, 0].min() + xy[:, 0].max()) / 2.0)
    top_y = float(xy[:, 1].min() - y_offset)
    ax.text(
        center_x,
        top_y,
        text,
        color=color,
        fontsize=10.5,
        fontweight="bold",
        ha="center",
        va="bottom",
        zorder=8,
        bbox={
            "boxstyle": "round,pad=0.36",
            "facecolor": "white",
            "edgecolor": color,
            "linewidth": 1.15,
            "alpha": 0.97,
        },
    )


def render_panel(
    output_stem: Path,
    background: Image.Image,
    context_xy: np.ndarray,
    reranked_xy: np.ndarray,
    recovery_xy: np.ndarray,
    measurement: dict,
    *,
    after: bool,
    dpi: int,
) -> None:
    width, height = background.size
    fig = plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi, facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(background, interpolation="none")
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    ax.axis("off")

    draw_cuboid(ax, context_xy, BLUE, linewidth=1.75)
    draw_cuboid(ax, reranked_xy, PURPLE, linewidth=2.05)

    if after:
        draw_cuboid(ax, recovery_xy, ORANGE, linewidth=2.15)
        add_label(ax, recovery_xy, "Recovered", ORANGE, y_offset=26.0)
        score = measurement["after_score"]
        rank = measurement["after_rank"]
    else:
        draw_cuboid(
            ax,
            recovery_xy,
            MISSING,
            linewidth=1.75,
            dashed=True,
            white_halo=False,
        )
        score = measurement["before_score"]
        rank = measurement["before_rank"]

    add_label(ax, reranked_xy, f"score {score:.3f} | rank {rank}", PURPLE)

    for suffix in (".png", ".svg", ".pdf"):
        fig.savefig(
            output_stem.with_suffix(suffix),
            dpi=dpi,
            facecolor="white",
            transparent=False,
        )
    plt.close(fig)


def save_pair(before_path: Path, after_path: Path, output_path: Path) -> None:
    before = Image.open(before_path).convert("RGB")
    after = Image.open(after_path).convert("RGB")
    gap = 24
    canvas = Image.new("RGB", (before.width + after.width + gap, before.height), "white")
    canvas.paste(before, (0, 0))
    canvas.paste(after, (before.width + gap, 0))
    canvas.save(output_path)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica", "Nimbus Sans", "Arial", "Liberation Sans"],
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )

    camera_path = args.source_dir / "fixed_camera.json"
    boxes_path = args.source_dir / "measured_boxes_zup.npz"
    background_path = args.source_dir / "colored_mesh_perspective.png"
    camera = json.loads(camera_path.read_text(encoding="utf-8"))
    measured = np.load(boxes_path)
    project = make_projector(camera)

    base = Image.open(background_path).convert("RGBA")
    white = Image.new("RGBA", base.size, (255, 255, 255, 255))
    white.alpha_composite(base)
    background = white.convert("RGB").crop(CROP)

    context_box = measured["native_boxes"][CONTEXT_NATIVE_INDEX]
    reranked_box = measured["native_boxes"][RERANKED_NATIVE_INDEX]
    recovery_box = measured["plr_recovered_box"]
    context_xy = crop_coordinates(project(context_box))
    reranked_xy = crop_coordinates(project(reranked_box))
    recovery_xy = crop_coordinates(project(recovery_box))
    measurement = load_reranking_measurement(args.prediction_dir)

    before_stem = args.output_dir / "before_code_rendered"
    after_stem = args.output_dir / "after_code_rendered"
    render_panel(
        before_stem,
        background,
        context_xy,
        reranked_xy,
        recovery_xy,
        measurement,
        after=False,
        dpi=args.dpi,
    )
    render_panel(
        after_stem,
        background,
        context_xy,
        reranked_xy,
        recovery_xy,
        measurement,
        after=True,
        dpi=args.dpi,
    )
    save_pair(
        before_stem.with_suffix(".png"),
        after_stem.with_suffix(".png"),
        args.output_dir / "before_after_code_rendered.png",
    )

    source_files = [
        camera_path,
        boxes_path,
        background_path,
        args.prediction_dir / "native_p100" / f"{SCENE}_boxes.pkl",
        args.prediction_dir / "historical_max_p100" / f"{SCENE}_boxes.pkl",
    ]
    manifest = {
        "schema": "carr3d.motivation_before_after.code_rendered.v1",
        "scene": SCENE,
        "generative_images_used": False,
        "manual_box_editing_used": False,
        "renderer": "fixed-camera 3D-to-2D projection plus Matplotlib vector overlays",
        "crop_pixels_ltrb": list(CROP),
        "context_native_index": CONTEXT_NATIVE_INDEX,
        "reranked_native_index": RERANKED_NATIVE_INDEX,
        "reranking_measurement": measurement,
        "displayed_coordinates": {
            "context_native_box_z_up_m": context_box.tolist(),
            "reranked_native_box_z_up_m": reranked_box.tolist(),
            "recovery_box_z_up_m": recovery_box.tolist(),
        },
        "colors": {"context": BLUE, "reranked": PURPLE, "recovered": ORANGE},
        "sources": [
            {"path": str(path.resolve()), "sha256": sha256(path)} for path in source_files
        ],
    }
    (args.output_dir / "source_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
