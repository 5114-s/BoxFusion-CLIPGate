#!/usr/bin/env python3
"""Render the real native-only BoxFusion map used by the method figure.

The exporter is visualization-only.  It reads the frozen controlled-Base
prediction for one ScanNet scene, draws every native box in blue, and never
loads PLR, CALR, or MVSR outputs.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
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
    "results/recar3d_final_scannet_factorial100_20260926/base/"
    "scene0432_01_boxes.pkl"
)
OUTPUT = ROOT / "figure_assets/pipeline_candidate_visuals_scene0432_01"
BLUE = "#1565FF"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def axis_alignment() -> np.ndarray:
    for line in META.read_text(encoding="utf-8").splitlines():
        if line.startswith("axisAlignment"):
            return np.asarray(
                line.split("=", 1)[1].split(), np.float64
            ).reshape(4, 4)
    raise RuntimeError(f"axisAlignment missing from {META}")


def transform(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    return points @ matrix[:3, :3].T + matrix[:3, 3]


def load_mesh(matrix: np.ndarray):
    ply = PlyData.read(MESH)
    vertex = ply["vertex"].data
    xyz = np.column_stack((vertex["x"], vertex["y"], vertex["z"]))
    xyz = transform(xyz.astype(np.float64), matrix)
    faces = np.stack(ply["face"].data["vertex_indices"]).astype(np.int64)
    return xyz, faces


def load_native_boxes(matrix: np.ndarray) -> tuple[list[np.ndarray], list[float]]:
    payload = pickle.load(PREDICTION.open("rb"))[0]
    boxes = [transform(np.asarray(row[1], np.float64), matrix) for row in payload]
    scores = [float(row[2]) for row in payload]
    return boxes, scores


def cuboid_edges(box: np.ndarray) -> np.ndarray:
    points = np.asarray(box, np.float64).reshape(8, 3)
    centered = points - points.mean(axis=0)
    opposite_cost = np.linalg.norm(centered + centered[0], axis=1)
    opposite_cost[0] = np.inf
    opposite = int(np.argmin(opposite_cost))
    others = [idx for idx in range(8) if idx not in (0, opposite)]
    diagonal = points[opposite] - points[0]
    best = None
    best_cost = np.inf
    for neighbours in itertools.combinations(others, 3):
        basis = np.stack([points[idx] - points[0] for idx in neighbours])
        if abs(float(np.linalg.det(basis))) < 1e-12:
            continue
        generated = np.asarray([
            points[0] + np.asarray(code, np.float64) @ basis
            for code in itertools.product((0, 1), repeat=3)
        ])
        nearest = np.argmin(
            np.linalg.norm(generated[:, None] - points[None], axis=2), axis=1
        )
        if len(set(nearest.tolist())) != 8:
            continue
        cost = float(np.max(np.linalg.norm(generated - points[nearest], axis=1)))
        cost += float(np.linalg.norm(basis.sum(axis=0) - diagonal))
        if cost < best_cost:
            best_cost = cost
            best = dict(zip(itertools.product((0, 1), repeat=3), nearest.tolist()))
    if best is None or best_cost > 1e-4:
        raise ValueError(f"cannot recover cuboid topology; error={best_cost}")
    edges = {
        tuple(sorted((best[a], best[b])))
        for a in best
        for b in best
        if sum(x != y for x, y in zip(a, b)) == 1
    }
    return np.asarray(sorted(edges), np.int64)


def box_segments(boxes: list[np.ndarray]) -> np.ndarray:
    return np.concatenate([box[cuboid_edges(box)] for box in boxes], axis=0)


def crop_alpha(path: Path, padding: int = 14) -> None:
    with Image.open(path) as image:
        rgba = image.convert("RGBA")
        box = rgba.getchannel("A").getbbox()
        if box is None:
            return
        crop = (
            max(0, box[0] - padding), max(0, box[1] - padding),
            min(rgba.width, box[2] + padding), min(rgba.height, box[3] + padding),
        )
        rgba.crop(crop).save(path)


def render(path: Path, xyz: np.ndarray, faces: np.ndarray,
           boxes: list[np.ndarray]) -> None:
    fig = plt.figure(figsize=(8.4, 5.8), dpi=300)
    ax = fig.add_subplot(111, projection="3d")

    # Neutral scene geometry keeps the native boxes visually unambiguous.
    mesh = Poly3DCollection(
        xyz[faces], facecolors="#D7DCE2", edgecolors="none",
        linewidths=0, alpha=0.48, rasterized=True,
    )
    ax.add_collection3d(mesh)

    segments = box_segments(boxes)
    halo = Line3DCollection(
        segments, colors="#FFFFFF", linewidths=4.2, alpha=0.95,
    )
    halo.set_sort_zpos(1e6)
    ax.add_collection3d(halo)
    lines = Line3DCollection(
        segments, colors=BLUE, linewidths=2.4, alpha=1.0,
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
    ax.view_init(elev=77.0, azim=-88.0)
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
    xyz, faces = load_mesh(matrix)
    boxes, scores = load_native_boxes(matrix)
    output = OUTPUT / "04b_native_map_NT_native_only.png"
    render(output, xyz, faces, boxes)

    manifest = {
        "schema": "boxfusion.figure.native_map.v1",
        "scene_id": SCENE,
        "time_state": "terminal online state N^(T)",
        "source": "controlled Base; native-only",
        "native_box_count": len(boxes),
        "native_scores": scores,
        "contains_plr": False,
        "contains_calr": False,
        "contains_mvsr_scores": False,
        "scene_geometry_role": "visualization only; not a model input",
        "prediction": str(PREDICTION),
        "prediction_sha256": sha256(PREDICTION),
        "mesh": str(MESH),
        "mesh_sha256": sha256(MESH),
        "axis_alignment": str(META),
        "axis_alignment_sha256": sha256(META),
        "output": str(output),
        "output_sha256": sha256(output),
    }
    manifest_path = OUTPUT / "04b_native_map_NT_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(output)
    print(manifest_path)


if __name__ == "__main__":
    main()
