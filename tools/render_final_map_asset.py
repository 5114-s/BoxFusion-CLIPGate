#!/usr/bin/env python3
"""Render real ScanNet mesh and final BoxFusion-route predictions for Fig. 1.

This is a visualization-only exporter.  It does not run or alter the model.
"""
from __future__ import annotations

import itertools
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection
import numpy as np
from PIL import Image
from plyfile import PlyData


ROOT = Path(__file__).resolve().parents[1]
SCENE = "scene0432_01"
MESH = Path(
    "/extra/ZhaoX/scannet_data/scans/scene0432_01/"
    "scene0432_01_vh_clean_2.ply"
)
META = MESH.with_name(f"{SCENE}.txt")
PREDICTION = ROOT / (
    "reports/m1a_strict_online_scannet_20260916/predictions/"
    "scene0432_01_boxes.pkl"
)
M1P_PREFIX = ROOT / (
    "results/scannet_m2nl_m5_dual_full100/persistent/"
    "scene0432_01_boxes.pkl"
)
NATIVE_PREFIX = ROOT / (
    "results/scannet_t05_boxer_kfmap_score05/scene0432_01_boxes.pkl"
)
OUTPUT = ROOT / "figure_assets/pipeline_candidate_visuals_scene0432_01"


SOURCE_COLORS = {
    "native": "#0057FF",
    "m1p": "#FF3B20",
    "m1a": "#F2A900",
}


def axis_alignment() -> np.ndarray:
    for line in META.read_text().splitlines():
        if line.startswith("axisAlignment"):
            return np.asarray(line.split("=", 1)[1].split(), np.float64).reshape(4, 4)
    raise RuntimeError(f"axisAlignment missing from {META}")


def transform(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    return points @ matrix[:3, :3].T + matrix[:3, 3]


def load_mesh(matrix: np.ndarray):
    ply = PlyData.read(MESH)
    vertex = ply["vertex"].data
    xyz = np.column_stack((vertex["x"], vertex["y"], vertex["z"]))
    xyz = transform(xyz.astype(np.float64), matrix)
    rgb = np.column_stack((vertex["red"], vertex["green"], vertex["blue"]))
    rgb = rgb.astype(np.float64) / 255.0
    faces = np.stack(ply["face"].data["vertex_indices"]).astype(np.int64)
    return xyz, rgb, faces


def load_rows(path: Path):
    payload = pickle.load(path.open("rb"))[0]
    return [(np.asarray(row[1], np.float64), float(row[2])) for row in payload]


def cuboid_edges(box: np.ndarray) -> np.ndarray:
    """Recover cuboid topology without assuming one stored corner ordering."""
    points = np.asarray(box, np.float64).reshape(8, 3)
    centered = points - points.mean(axis=0)
    opposite_cost = np.linalg.norm(centered + centered[0], axis=1)
    opposite_cost[0] = np.inf
    opposite = int(np.argmin(opposite_cost))
    others = [index for index in range(8) if index not in (0, opposite)]
    diagonal = points[opposite] - points[0]

    best = None
    best_cost = np.inf
    for neighbours in itertools.combinations(others, 3):
        basis = np.stack([points[index] - points[0] for index in neighbours])
        if abs(float(np.linalg.det(basis))) < 1e-12:
            continue
        generated = np.asarray([
            points[0] + np.asarray(code, np.float64) @ basis
            for code in itertools.product((0, 1), repeat=3)
        ])
        nearest = np.argmin(
            np.linalg.norm(generated[:, None] - points[None], axis=2), axis=1)
        if len(set(nearest.tolist())) != 8:
            continue
        reconstruction = float(np.max(np.linalg.norm(generated - points[nearest], axis=1)))
        diagonal_error = float(np.linalg.norm(basis.sum(axis=0) - diagonal))
        cost = reconstruction + diagonal_error
        if cost < best_cost:
            best_cost = cost
            best = dict(zip(itertools.product((0, 1), repeat=3), nearest.tolist()))
    if best is None or best_cost > 1e-4:
        raise ValueError(f"cannot recover cuboid corner topology; error={best_cost}")
    edges = {
        tuple(sorted((best[first], best[second])))
        for first in best
        for second in best
        if sum(a != b for a, b in zip(first, second)) == 1
    }
    if len(edges) != 12:
        raise AssertionError(f"expected 12 cuboid edges, got {len(edges)}")
    return np.asarray(sorted(edges), np.int64)


def box_segments(boxes: list[np.ndarray]) -> np.ndarray:
    if not boxes:
        return np.empty((0, 2, 3), np.float64)
    return np.concatenate([box[cuboid_edges(box)] for box in boxes], axis=0)


def crop_alpha(path: Path, padding: int = 12) -> None:
    with Image.open(path) as image:
        rgba = image.convert("RGBA")
        alpha_box = rgba.getchannel("A").getbbox()
        if alpha_box is None:
            return
        crop = (
            max(0, alpha_box[0] - padding), max(0, alpha_box[1] - padding),
            min(rgba.width, alpha_box[2] + padding),
            min(rgba.height, alpha_box[3] + padding),
        )
        rgba.crop(crop).save(path)


def footprint(box: np.ndarray) -> np.ndarray:
    """Return the counter-clockwise convex hull of a 3D box in the XY plane."""
    points = np.unique(np.round(np.asarray(box)[:, :2], 9), axis=0)
    points = points[np.lexsort((points[:, 1], points[:, 0]))]

    def cross(origin, first, second):
        return np.cross(first - origin, second - origin)

    lower = []
    for point in points:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0:
            lower.pop()
        lower.append(point)
    upper = []
    for point in points[::-1]:
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0:
            upper.pop()
        upper.append(point)
    hull = np.asarray(lower[:-1] + upper[:-1], np.float64)
    return np.vstack((hull, hull[0]))


def render_bev(path: Path, xyz: np.ndarray, rgb: np.ndarray, faces: np.ndarray,
               groups: dict[str, list[np.ndarray]]) -> None:
    """Crisp top-down map for the small final-output panel."""
    fig, ax = plt.subplots(figsize=(7.2, 7.2), dpi=240)
    order = np.argsort(xyz[faces, 2].mean(axis=1))
    polygons = xyz[faces[order], :2]
    face_rgb = rgb[faces[order]].mean(axis=1)
    face_rgb = 0.72 * face_rgb + 0.28
    ax.add_collection(PolyCollection(
        polygons, facecolors=face_rgb, edgecolors="none", alpha=0.92,
        rasterized=True,
    ))
    for source, boxes in groups.items():
        for box in boxes:
            hull = footprint(box)
            ax.plot(hull[:, 0], hull[:, 1], color="white", linewidth=6.0,
                    solid_joinstyle="round", zorder=20)
            ax.plot(hull[:, 0], hull[:, 1], color=SOURCE_COLORS[source],
                    linewidth=3.1, solid_joinstyle="round", zorder=21)
    lo = xyz[:, :2].min(axis=0)
    hi = xyz[:, :2].max(axis=0)
    margin = 0.04 * float(max(hi - lo))
    ax.set_xlim(lo[0] - margin, hi[0] + margin)
    ax.set_ylim(lo[1] - margin, hi[1] + margin)
    ax.set_aspect("equal")
    ax.set_axis_off()
    ax.set_facecolor((1, 1, 1, 0))
    fig.patch.set_alpha(0)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    fig.savefig(path, transparent=True, bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)
    crop_alpha(path)


def render(path: Path, xyz: np.ndarray, rgb: np.ndarray, faces: np.ndarray,
           groups: dict[str, list[np.ndarray]], linewidth: float,
           elev: float = 76.0) -> None:
    fig = plt.figure(figsize=(8.2, 5.8), dpi=220)
    ax = fig.add_subplot(111, projection="3d")
    face_rgb = rgb[faces].mean(axis=1)
    # Slight whitening keeps the real scan visible behind colored detections.
    face_rgb = 0.66 * face_rgb + 0.34
    mesh = Poly3DCollection(
        xyz[faces], facecolors=face_rgb, edgecolors="none",
        linewidths=0, alpha=0.78, rasterized=True,
    )
    ax.add_collection3d(mesh)

    for source, boxes in groups.items():
        segments = box_segments(boxes)
        if len(segments):
            halo = Line3DCollection(
                segments, colors="#FFFFFF", linewidths=linewidth + 1.9,
                alpha=0.82,
            )
            halo.set_sort_zpos(1e6)
            ax.add_collection3d(halo)
            lines = Line3DCollection(
                segments, colors=SOURCE_COLORS[source], linewidths=linewidth,
                alpha=1.0,
            )
            lines.set_sort_zpos(1e6 + 1)
            ax.add_collection3d(lines)

    lo = xyz.min(axis=0)
    hi = xyz.max(axis=0)
    center = 0.5 * (lo + hi)
    span = max(hi[0] - lo[0], hi[1] - lo[1]) * 0.54
    ax.set_xlim(center[0] - span, center[0] + span)
    ax.set_ylim(center[1] - span, center[1] + span)
    ax.set_zlim(lo[2] - 0.02, hi[2] + 0.10)
    ax.set_box_aspect((1, 1, 0.42))
    ax.view_init(elev=elev, azim=-88)
    ax.set_axis_off()
    ax.set_facecolor((1, 1, 1, 0))
    fig.patch.set_alpha(0)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    fig.savefig(path, transparent=True, bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)
    crop_alpha(path)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    matrix = axis_alignment()
    xyz, rgb, faces = load_mesh(matrix)
    rows = load_rows(PREDICTION)
    native_count = len(load_rows(NATIVE_PREFIX))
    m1p_count = len(load_rows(M1P_PREFIX)) - native_count
    boxes = [transform(box, matrix) for box, _ in rows]

    full = {
        "native": boxes[:native_count],
        "m1p": boxes[native_count:native_count + m1p_count],
        "m1a": boxes[native_count + m1p_count:],
    }
    # The small method panel uses real output rows but keeps only a readable
    # representative subset.  Exact row indices are recorded below; the
    # complete, unfiltered map is exported separately.
    clean = {
        "native": full["native"],
        "m1p": [boxes[9]],
        "m1a": [boxes[index] for index in (21, 23, 26)],
    }
    render(OUTPUT / "07_final_3d_map_all_predictions.png",
           xyz, rgb, faces, full, linewidth=0.86, elev=82.0)
    render(OUTPUT / "07_final_3d_map_figure_oblique.png",
           xyz, rgb, faces, clean, linewidth=2.15, elev=82.0)
    render_bev(OUTPUT / "07_final_3d_map_figure.png", xyz, rgb, faces, clean)

    readme = OUTPUT / "07_final_3d_map_README.txt"
    readme.write_text(
        "Real final-map assets for ScanNet scene0432_01.\n"
        f"Mesh: {MESH}\n"
        f"Predictions: {PREDICTION}\n"
        f"Axis alignment: {META}\n"
        f"Counts: native={len(full['native'])}, M1-P={len(full['m1p'])}, "
        f"M1-A={len(full['m1a'])}, total={len(rows)}.\n"
        "07_final_3d_map_all_predictions.png renders all final prediction rows.\n"
        "07_final_3d_map_figure.png is a crisp BEV method-figure asset: all "
        "five native rows, M1-P row 9, and M1-A rows 21/23/26 from the same "
        "prediction file. It must be described as an illustrative subset, not "
        "the complete output. The subset is chosen only for visual legibility.\n"
        "07_final_3d_map_figure_oblique.png contains the same subset in an "
        "oblique 3D view.\n"
        "Colors: blue=native, red-orange=M1-P, amber=M1-A.\n"
        "The ScanNet mesh is used for visualization only.\n",
        encoding="utf-8",
    )
    print(OUTPUT / "07_final_3d_map_all_predictions.png")
    print(OUTPUT / "07_final_3d_map_figure.png")
    print(readme)


if __name__ == "__main__":
    main()
