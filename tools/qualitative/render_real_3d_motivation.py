#!/usr/bin/env python3
"""Render actual mesh and measured boxes from a fixed perspective camera."""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import importlib.util
import itertools
import json
from pathlib import Path
import pickle
import subprocess
import zipfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
import matplotlib.patheffects as pe
import numpy as np
from plyfile import PlyData, PlyElement

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "development/final_pipeline/reports/qualitative_recar3d_final"
RAW = Path("/extra/ZhaoX/scannet_data/scans")
COMPONENTS = ROOT / "development/final_controls/results/scannet_recar3d_key_control_components"
BLUE = "#116CC7"
ORANGE = "#F17B19"
GREEN = "#138953"


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def transform(points, matrix):
    return np.asarray(points) @ matrix[:3, :3].T + matrix[:3, 3]


def eval_to_zup(points):
    """Invert the evaluator's (x,y,z)->(x,-z,y) axis change."""
    points = np.asarray(points, dtype=float)
    return np.stack((points[..., 0], points[..., 2], -points[..., 1]), axis=-1)


def rotation_basis(camera, look_at):
    forward = look_at-camera
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, [0., 0., 1.])
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    return np.stack((right, up, forward))


def project(points, camera, look_at, focal, center):
    local = (np.asarray(points)-camera) @ rotation_basis(camera, look_at).T
    return np.column_stack((
        focal*local[:, 0]/local[:, 2]+center[0],
        -focal*local[:, 1]/local[:, 2]+center[1],
        local[:, 2],
    ))


def box_edges(box):
    """Use existing topology recovery without assuming stored corner order."""
    path = ROOT / "tools/render_final_map_asset.py"
    spec = importlib.util.spec_from_file_location("real_box_topology", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.cuboid_edges(box)


def load_mesh(scene, alignment, removed_wall_ids):
    folder = RAW/scene
    mesh_path = folder/f"{scene}_vh_clean_2.ply"
    mesh = PlyData.read(mesh_path)
    original = mesh["vertex"].data
    xyz = transform(np.column_stack([original[k] for k in ("x", "y", "z")]), alignment)
    colors = np.column_stack([original[k] for k in ("red", "green", "blue")])/255.
    faces = np.stack(mesh["face"].data["vertex_indices"]).astype(np.int32)
    segments = np.asarray(json.loads((folder/f"{scene}_vh_clean_2.0.010000.segs.json").read_text())["segIndices"])
    objects = json.loads((folder/f"{scene}.aggregation.json").read_text())["segGroups"]
    removed_segments = []
    for obj in objects:
        if obj["objectId"] in removed_wall_ids:
            if obj["label"] != "wall":
                raise RuntimeError("cutaway may only remove specified wall instances")
            removed_segments.extend(obj["segments"])
    remove_vertices = np.isin(segments, removed_segments)
    wall_mask = np.any(remove_vertices[faces], axis=1)
    triangles = xyz[faces]
    norms = np.cross(triangles[:, 1]-triangles[:, 0], triangles[:, 2]-triangles[:, 0])
    lens = np.linalg.norm(norms, axis=1)
    norms /= np.maximum(lens[:, None], 1e-12)
    # Remove only high nearly-horizontal surfaces (ceiling remnants), in both views.
    ceiling_mask = (triangles[:, :, 2].mean(1) > 2.18) & (np.abs(norms[:, 2]) > .65)
    keep = ~(wall_mask | ceiling_mask)
    kept_faces = faces[keep]
    # Subtle normal shading changes appearance, not measured geometry or vertex RGB.
    light = np.array([-.35, -.55, .76])
    light /= np.linalg.norm(light)
    illumination = .79+.21*np.abs(norms @ light)
    sums = np.zeros(len(xyz))
    count = np.zeros(len(xyz))
    for i in range(3):
        np.add.at(sums, faces[:, i], illumination)
        np.add.at(count, faces[:, i], 1)
    per_vertex = sums/np.maximum(count, 1)
    display_rgb = np.clip(colors**.80*per_vertex[:, None]*.92+.06, 0, 1)
    return xyz, colors, display_rgb, kept_faces, {
        "path": str(mesh_path), "sha256": sha(mesh_path),
        "vertices": len(xyz), "original_faces": len(faces),
        "cutaway_faces": len(kept_faces), "removed_wall_instance_ids": removed_wall_ids,
        "removed_wall_faces": int(wall_mask.sum()),
        "removed_ceiling_faces": int(ceiling_mask.sum()),
        "ceiling_rule": "mean Z > 2.18 m and absolute face-normal Z > 0.65",
        "color_display": "vertex RGB gamma 0.80; subtle normal illumination; 0.06 ambient lift",
        "new_surfaces_or_furniture_added": False,
    }


def rasterize(projected, rgb, faces, width, height):
    cpp = ROOT/"tools/qualitative/cpu_mesh_raster.cpp"
    library = Path("/tmp")/f"recar_real_mesh_{sha(cpp)[:14]}.so"
    if not library.is_file():
        subprocess.run(["g++", "-O3", "-std=c++17", "-shared", "-fPIC",
                        str(cpp), "-o", str(library)], check=True)
    lib = ctypes.CDLL(str(library))
    fn = lib.rasterize_mesh
    fn.argtypes = [ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
                   ctypes.POINTER(ctypes.c_int32), ctypes.c_int64, ctypes.c_int,
                   ctypes.c_int, ctypes.POINTER(ctypes.c_uint8)]
    fn.restype = None
    projected = np.ascontiguousarray(projected, dtype=np.float64)
    rgb = np.ascontiguousarray(rgb, dtype=np.float64)
    faces = np.ascontiguousarray(faces, dtype=np.int32)
    rgba = np.empty((height, width, 4), dtype=np.uint8)
    fn(projected.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
       rgb.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
       faces.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)), len(faces), width, height,
       rgba.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)))
    return rgba


def render_axis(ax, rgba, native_2d, target_2d, gt_2d, *, recovery, show_gt, labels):
    ax.imshow(rgba, interpolation="antialiased")
    for box in native_2d:
        ax.add_collection(LineCollection(box, colors=BLUE, linewidths=1.5,
                                         alpha=.88, zorder=6))
    if show_gt:
        collection = LineCollection(gt_2d, colors=GREEN, linewidths=1.5,
                                    linestyles="dashed", zorder=8)
        ax.add_collection(collection)
    if recovery:
        collection = LineCollection(target_2d, colors=ORANGE, linewidths=2.5, zorder=10)
        collection.set_path_effects([pe.Stroke(linewidth=3.6, foreground="white", alpha=.7), pe.Normal()])
        ax.add_collection(collection)
    if labels:
        focus = target_2d.mean((0, 1))
        ax.annotate("PLR recovery" if recovery else "No native record for this target",
                    xy=focus, xytext=(focus[0]-rgba.shape[1]*.30, focus[1]-rgba.shape[0]*.15),
                    fontsize=11, weight="semibold", color=ORANGE if recovery else "#18283E",
                    arrowprops={"arrowstyle": "-", "color": ORANGE if recovery else "#18283E", "lw": 1.2},
                    bbox={"facecolor": "white", "alpha": .88, "edgecolor": "none", "pad": 3}, zorder=20)
    ax.set_xlim(0, rgba.shape[1]); ax.set_ylim(rgba.shape[0], 0)
    ax.set_aspect("equal"); ax.axis("off")


def export_ply(path, xyz, rgb, faces):
    verts = np.empty(len(xyz), dtype=[("x", "f4"), ("y", "f4"), ("z", "f4"),
                                     ("red", "u1"), ("green", "u1"), ("blue", "u1")])
    for i, key in enumerate(("x", "y", "z")):
        verts[key] = xyz[:, i]
    for i, key in enumerate(("red", "green", "blue")):
        verts[key] = np.rint(rgb[:, i]*255).astype(np.uint8)
    polys = np.empty(len(faces), dtype=[("vertex_indices", "i4", (3,))])
    polys["vertex_indices"] = faces
    PlyData([PlyElement.describe(verts, "vertex"), PlyElement.describe(polys, "face")], text=False).write(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=ROOT/"figure_assets/real_3d_motivation_20261026")
    parser.add_argument("--azimuth", type=float, default=-58.)
    parser.add_argument("--elevation", type=float, default=29.)
    args = parser.parse_args()
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    payload = json.loads((SOURCE/"selected_cases.json").read_text())
    case = payload["cases"]["plr_recovery"]
    evidence = Path(case["evidence_export"])
    assert case["evidence_verified"] and sha(evidence) == case["evidence_sha256"]
    scene = case["scene"]
    matrix = None
    for line in (RAW/scene/f"{scene}.txt").read_text().splitlines():
        if line.startswith("axisAlignment"):
            matrix = np.array(line.split("=", 1)[1].split(), dtype=float).reshape(4, 4)
    assert matrix is not None
    target = transform(np.asarray(case["box"]), matrix)
    gt = eval_to_zup(case["gt_box_aligned"])
    eval_target_scene_aabb = eval_to_zup(case["box_aligned"])
    # Confirm that the raw OBB, evaluated AABB, mesh and GT agree after axis conversion.
    assert np.max(np.abs(target.min(0)-eval_target_scene_aabb.min(0))) < 2e-5
    assert np.max(np.abs(target.max(0)-eval_target_scene_aabb.max(0))) < 2e-5
    native_path = COMPONENTS/"native_native"/f"{scene}_boxes.pkl"
    with native_path.open("rb") as handle:
        rows = pickle.load(handle)[0]
    native = np.stack([transform(row[1], matrix) for row in rows])
    scores = np.array([row[2] for row in rows])
    # Show a readable subset, chosen without altering any box coordinates.
    volumes = np.prod(np.ptp(native, axis=1), axis=1)
    eligible = [i for i in range(len(native)) if volumes[i] > .075]
    display_indices = sorted(eligible, key=lambda i: (-volumes[i], -scores[i], i))[:6]

    xyz, rgb, display_rgb, faces, mesh_info = load_mesh(scene, matrix, [16, 18, 27])
    look_at = np.array([0., -.02, 1.02])
    az, el = np.radians([args.azimuth, args.elevation])
    distance = 9.
    camera = look_at+distance*np.array([np.cos(el)*np.cos(az), np.cos(el)*np.sin(az), np.sin(el)])
    width, height = 2200, 1800
    focal = width*1.35
    center = np.array([width/2, height/2])
    projected = project(xyz, camera, look_at, focal, center)
    visible_points = np.unique(faces)
    lo, hi = projected[visible_points, :2].min(0), projected[visible_points, :2].max(0)
    # Uniform 2D framing retains a physical perspective camera.
    scale = min(width*.89/(hi[0]-lo[0]), height*.89/(hi[1]-lo[1]))
    focal *= scale
    projected = project(xyz, camera, look_at, focal, center)
    lo, hi = projected[visible_points, :2].min(0), projected[visible_points, :2].max(0)
    center += np.array([width/2, height/2])-(lo+hi)/2
    projected = project(xyz, camera, look_at, focal, center)
    rgba = rasterize(projected, display_rgb, faces, width, height)

    def segments(box):
        return project(box, camera, look_at, focal, center)[:, :2][box_edges(box)]
    target_2d, gt_2d = segments(target), segments(gt)
    native_2d = [segments(native[index]) for index in display_indices]
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "pdf.fonttype": 42, "svg.fonttype": "none"})
    plt.imsave(output/"colored_mesh_perspective.png", rgba)
    for recovery, name in ((False, "native_scene_3d"), (True, "recovered_scene_3d")):
        for gt_flag in (False, True):
            fig, ax = plt.subplots(figsize=(8.8, 7.2), dpi=250)
            fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
            render_axis(ax, rgba, native_2d, target_2d, gt_2d,
                        recovery=recovery, show_gt=gt_flag, labels=False)
            suffix = "_with_gt" if gt_flag else ""
            fig.savefig(output/f"{name}{suffix}.png", transparent=True, dpi=250)
            fig.savefig(output/f"{name}{suffix}.svg", transparent=True)
            plt.close(fig)

    # The three upper miniature scenes can also use the identical 3D camera.
    # Their boxes are measured at the listed frames, while the static scan is
    # only a visualization background and is not a per-prefix reconstruction.
    frame_records = json.loads(evidence.read_text())["frames"]
    fig, mini_axes = plt.subplots(1, 3, figsize=(12.9, 3.9), dpi=220)
    fig.subplots_adjust(left=.005, right=.995, bottom=.07, top=.90, wspace=.025)
    for ax, row in zip(mini_axes, frame_records):
        candidate_box = transform(np.asarray(row["candidate_box"]), matrix)
        candidate_2d = segments(candidate_box)
        render_axis(ax, rgba, [], candidate_2d, gt_2d,
                    recovery=True, show_gt=False, labels=False)
        ax.set_title(f"Candidate evidence: t={row['frame_id']}", fontsize=11, pad=4)
        individual, single_ax = plt.subplots(figsize=(6.6, 5.4), dpi=250)
        individual.subplots_adjust(left=0, right=1, bottom=0, top=1)
        render_axis(single_ax, rgba, [], candidate_2d, gt_2d,
                    recovery=True, show_gt=False, labels=False)
        individual.savefig(output/f"evidence_3d_frame_{row['frame_id']:06d}.png", transparent=True)
        plt.close(individual)
    fig.text(.5, .025, "Measured per-frame proposal boxes; fixed scan mesh used only as a visualization background.",
             ha="center", fontsize=8, color="#64748B")
    for ext in ("png", "pdf", "svg"):
        fig.savefig(output/f"three_evidence_scenes_3d.{ext}", dpi=250, facecolor="white")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(14.2, 6.4), dpi=250)
    fig.subplots_adjust(left=.015, right=.985, bottom=.12, top=.84, wspace=.025)
    for ax, recovery, title in zip(axes, (False, True), ("BoxFusion-Ctrl", "ReCaR-3D")):
        render_axis(ax, rgba, native_2d, target_2d, gt_2d,
                    recovery=recovery, show_gt=False, labels=True)
        ax.set_title(title, fontsize=17, weight="bold", pad=6)
    fig.suptitle("Same real scene, same perspective", y=.98, fontsize=21, weight="bold", color="#12213A")
    fig.text(.5, .90, f"ScanNet {scene}  |  selected PLR target  |  3D AABB IoU = {case['iou']:.2f}",
             ha="center", color="#475569", fontsize=11)
    handles = [Line2D([0], [0], color=BLUE, lw=2, label="Native boxes (same geometry)"),
               Line2D([0], [0], color=ORANGE, lw=2, label="Selected PLR recovery")]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.5, .05),
               ncol=2, frameon=False, fontsize=11)
    fig.text(.5, .018, "Colored scan mesh is used only for visualization; identical cutaway and camera in both views.",
             ha="center", fontsize=9, color="#64748B")
    for ext in ("png", "pdf", "svg"):
        fig.savefig(output/f"real_3d_comparison.{ext}", dpi=300, facecolor="white")
    plt.close(fig)

    export_ply(output/f"{scene}_aligned_cutaway.ply", xyz, rgb, faces)
    np.savez_compressed(output/"measured_boxes_zup.npz", native_boxes=native,
                        native_scores=scores, displayed_native_indices=display_indices,
                        plr_recovered_box=target, ground_truth_box=gt)
    coordinates = {
        "scene": scene, "coordinate_system": "axis-aligned ScanNet; Z up; metres",
        "native_boxes": [
            {"index": int(i), "score": float(scores[i]), "corners": box.tolist(),
             "displayed": i in display_indices}
            for i, box in enumerate(native)
        ],
        "plr_recovered_box": target.tolist(), "ground_truth_box": gt.tolist(),
        "shown_evidence_candidates": [
            {"frame_id": int(row["frame_id"]), "score": row["candidate_score"],
             "corners": transform(np.asarray(row["candidate_box"]), matrix).tolist()}
            for row in frame_records
        ],
    }
    (output/"measured_boxes_zup.json").write_text(json.dumps(coordinates, indent=2)+"\n")
    camera_info = {"coordinate_system": "axis-aligned ScanNet; Z up, metres",
                   "position": camera.tolist(), "look_at": look_at.tolist(),
                   "up": [0, 0, 1], "focal_pixels": focal,
                   "principal_point_pixels": center.tolist(), "image_size": [width, height],
                   "azimuth_degrees": args.azimuth, "elevation_degrees": args.elevation}
    (output/"fixed_camera.json").write_text(json.dumps(camera_info, indent=2)+"\n")
    manifest = {"schema": "recar3d.real_mesh_perspective_comparison.v1", "scene": scene,
                "mesh": mesh_info, "camera": camera_info,
                "same_mesh_camera_and_native_geometry_in_both_views": True,
                "generative_images": False, "model_rerun": False,
                "visualization_only": True, "native_prediction_source": str(native_path),
                "native_prediction_sha256": sha(native_path),
                "displayed_native_indices": display_indices,
                "native_total_count": len(native), "selected_plr_index": case["source_index"],
                "plr_3d_aabb_iou": case["iou"], "plr_evidence_source": str(evidence),
                "box_coordinate_conversion": "native/PLR raw world -> axisAlignment; GT eval camera (x,y,z) -> Z-up (x,z,-y)",
                "axis_validation_max_error_m": float(np.max(np.abs(target.min(0)-eval_target_scene_aabb.min(0)))),
                "lines": "measured cuboid edges drawn as diagram overlays, including occluded edges",
                "evidence_miniatures": "per-frame candidate boxes from 425/450/550 on the same static visualization mesh; not a per-prefix reconstruction",
                "display_scope": "6 largest non-tiny native boxes for context, plus selected PLR; other recovery records not displayed"}
    (output/"source_manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    (output/"素材说明.txt").write_text(
        "同一真实场景的固定视角三维对照\n\n"
        "native_scene_3d.png / recovered_scene_3d.png：可直接放入动机图左右主场景的透明底立体图。\n"
        "real_3d_comparison.png / .pdf / .svg：带标题和图例的并排对照。\n"
        "colored_mesh_perspective.png：同视角、无检测框的彩色网格。\n"
        "three_evidence_scenes_3d.*：顶部三个立体证据小场景；另有evidence_3d_frame_*.png透明单图。"
        "小场景采用静态扫描网格作为统一背景，不表示当时已经建立了该网格或使用了未来观测。\n"
        "*_with_gt.png / .svg：另叠加绿色虚线GT。\n"
        "scene0426_03_aligned_cutaway.ply：实际彩色网格，Z向上，单位米；统一隐藏靠近相机的墙面及顶面残片。\n"
        "measured_boxes_zup.npz：全部21个原生框及分数、6个展示框索引、所选PLR框、GT，均与网格同坐标。\n"
        "measured_boxes_zup.json：相同框坐标的可读JSON，另含三个关键帧的候选框。\n"
        "fixed_camera.json：透视相机及构图参数，可复现两张相同视角。\n\n"
        "本图使用ScanNet真实网格、真实原生预测和PLR恢复框。没有补画家具或生成场景。"
        "仅显示6个代表性原生框及1个所选PLR恢复框以保持清晰；不表示完整输出集合。"
        "三维框的遮挡边也作为示意线绘出。前后使用相同网格、剖切、照明及相机。\n"
        "注意：网格仅用于论文可视化，不作为ReCaR-3D模型输入，也不改变无重建检测设定。"
        "原始扫描存在缺口和扫描噪声，因此与人工概念场景的光滑家具外观不同。\n",
        encoding="utf-8")
    with zipfile.ZipFile(output.with_suffix(".zip"), "w", zipfile.ZIP_DEFLATED) as bundle:
        for path in sorted(output.rglob("*")):
            if path.is_file():
                bundle.write(path, path.relative_to(output.parent))
    print(json.dumps({"output": str(output), "camera": camera_info,
                      "mesh": mesh_info, "display_indices": display_indices}, indent=2))


if __name__ == "__main__":
    main()
