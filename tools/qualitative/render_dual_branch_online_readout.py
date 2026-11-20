#!/usr/bin/env python3
"""Export the audited dual-branch online readout used in the method figure."""
from __future__ import annotations

import importlib.util
import itertools
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
RAW = Path("/extra/ZhaoX/scannet_data/scans")
RUN = ROOT / "reports/official_github_boxfusion_vs_final_recar_public_cli_coldcache_nowarmup100_20261009"
OUTPUT = ROOT / "figure_assets/online_detection_readout_real_dual_branch_20261107"
SCENE = "scene0599_01"
WALLS = [14, 15, 16, 18, 20]
COLORS = {"native": "#236FB6", "ptr": "#F07820", "var": "#E0A000"}
PREDICTION_LINEWIDTH = 1.25
AUDIT_LINEWIDTH = 0.90
CORNER_CODES = np.asarray(list(itertools.product((0, 1), repeat=3)), dtype=float)


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def camera_for(renderer, xyz, faces, gt, azimuth=-35., elevation=30.):
    width, height = 1800, 1440
    low, high = gt.min((0, 1)), gt.max((0, 1))
    look = (low + high) / 2
    look[2] = .95
    az, el = np.deg2rad([azimuth, elevation])
    direction = np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
    position = look + 12 * direction
    principal = np.array([width / 2, height / 2], dtype=float)
    focal = 2400.
    valid = np.unique(faces)
    uv = renderer.project(xyz[valid], position, look, focal, principal)[:, :2]
    lower, upper = uv.min(0), uv.max(0)
    focal *= min(width * .90 / (upper[0] - lower[0]), height * .88 / (upper[1] - lower[1]))
    uv = renderer.project(xyz[valid], position, look, focal, principal)[:, :2]
    principal += np.array([width / 2, height / 2]) - (uv.min(0) + uv.max(0)) / 2
    return {
        "position": position.tolist(), "look_at": look.tolist(), "focal_pixels": float(focal),
        "principal_point_pixels": principal.tolist(), "image_size": [width, height],
        "azimuth_degrees": azimuth, "elevation_degrees": elevation,
    }


def draw_box(ax, renderer, box, projection_args, color, linewidth, dashed=False):
    projected = renderer.project(box, *projection_args)
    edges = renderer.box_edges(box)
    ax.add_collection(LineCollection(
        projected[:, :2][edges], colors=color, linewidths=linewidth,
        linestyles="dashed" if dashed else "solid", alpha=.97, zorder=10,
    ))


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    common = load_module("box_render_common", ROOT / "tools/qualitative/render_boxfusion_ctrl_recar_3d.py")
    renderer = common.load_renderer()
    matrix = common.alignment(SCENE)
    gt_array = np.load(ROOT / "evaluation/data_util/scannet_train_detection_data" / f"{SCENE}_bbox.npy")
    gt = gt_array[:, None, :3] + (CORNER_CODES[None] - .5) * gt_array[:, None, 3:6]
    receipt = RUN / "receipts/recar" / SCENE
    native = common.read_predictions(receipt / "native" / f"{SCENE}_boxes.pkl", matrix)
    full = common.read_predictions(receipt / "recar" / f"{SCENE}_boxes.pkl", matrix)
    assert np.array_equal(native["raw"], full["raw"][:len(native["raw"])])

    # Audited same-scene recoveries: two distinct native-missed objects.
    ptr_index, ptr_gt = 11, 11
    var_index, var_gt = 61, 14
    overlaps = common.iou_matrix(full["boxes"], gt)
    ptr_iou = float(overlaps[ptr_index, ptr_gt])
    var_iou = float(overlaps[var_index, var_gt])
    assert full["scores"][ptr_index] >= .05 and full["scores"][var_index] < .05
    assert ptr_iou > .73 and var_iou > .77
    # Four distinct, readily visible native records.  Every displayed native
    # prediction has AABB IoU above 0.74 with its corresponding GT instance.
    native_indices = [10, 3, 2, 5]
    native_overlaps = common.iou_matrix(native["boxes"], gt)
    native_gt_indices = [int(np.argmax(native_overlaps[i])) for i in native_indices]

    xyz, _, display_rgb, faces, mesh_info = renderer.load_mesh(SCENE, matrix, WALLS)
    camera = common.make_camera(renderer, SCENE, xyz, faces, gt)
    args = (np.asarray(camera["position"]), np.asarray(camera["look_at"]),
            camera["focal_pixels"], np.asarray(camera["principal_point_pixels"]))
    projected = renderer.project(xyz, *args)
    width, height = camera["image_size"]
    background = renderer.rasterize(projected, display_rgb, faces, width, height)
    valid = np.unique(faces)
    uv = projected[valid, :2]
    boxes = [full["boxes"][ptr_index], full["boxes"][var_index]] + [native["boxes"][i] for i in native_indices]
    box_uv = renderer.project(np.concatenate(boxes), *args)[:, :2]
    low = np.minimum(uv.min(0), box_uv.min(0)) - 18
    high = np.maximum(uv.max(0), box_uv.max(0)) + 18
    camera["display_limits"] = [float(low[0]), float(high[0]), float(high[1]), float(low[1])]

    def render(stem, *, audit=False, transparent=False):
        fig = plt.figure(figsize=(7.5, 6), dpi=240)
        ax = fig.add_axes([0, 0, 1, 1])
        ax.imshow(background, interpolation="antialiased")
        for index in native_indices:
            draw_box(ax, renderer, native["boxes"][index], args, COLORS["native"], PREDICTION_LINEWIDTH)
        draw_box(ax, renderer, full["boxes"][ptr_index], args, COLORS["ptr"], PREDICTION_LINEWIDTH)
        draw_box(ax, renderer, full["boxes"][var_index], args, COLORS["var"], PREDICTION_LINEWIDTH)
        if audit:
            for gt_index in native_gt_indices:
                draw_box(ax, renderer, gt[gt_index], args, COLORS["native"], AUDIT_LINEWIDTH, dashed=True)
            draw_box(ax, renderer, gt[ptr_gt], args, COLORS["ptr"], AUDIT_LINEWIDTH, dashed=True)
            draw_box(ax, renderer, gt[var_gt], args, COLORS["var"], AUDIT_LINEWIDTH, dashed=True)
            native_text = "/".join(
                f"{native_overlaps[i, g]:.3f}" for i, g in zip(native_indices, native_gt_indices)
            )
            ax.text(.018, .982,
                    f"Native IoUs {native_text}    PTR {ptr_iou:.3f}    VAR {var_iou:.3f}",
                    transform=ax.transAxes, va="top", fontsize=12, weight="bold",
                    bbox={"facecolor": "white", "alpha": .9, "edgecolor": "none", "pad": 3})
        ax.set_xlim(camera["display_limits"][:2]); ax.set_ylim(camera["display_limits"][2:])
        ax.set_aspect("equal"); ax.axis("off")
        fig.savefig(OUTPUT / f"{stem}.png", dpi=400, transparent=transparent,
                    facecolor="none" if transparent else "white", bbox_inches="tight", pad_inches=.01)
        if not audit:
            fig.savefig(OUTPUT / f"{stem}.svg", transparent=transparent,
                        facecolor="none" if transparent else "white", bbox_inches="tight", pad_inches=.01)
        plt.close(fig)

    render("online_detection_readout_dual_white", transparent=False)
    render("online_detection_readout_dual_transparent", transparent=True)
    render("online_detection_readout_dual_audit", audit=True, transparent=False)

    manifest = {
        "schema": "carr3d.real_dual_branch_online_readout.v1", "scene": SCENE,
        "colors": COLORS,
        "line_width_points": {"prediction": PREDICTION_LINEWIDTH, "audit_gt": AUDIT_LINEWIDTH},
        "camera": camera, "mesh": mesh_info,
        "native_source": native["path"], "native_sha256": native["sha256"],
        "full_source": full["path"], "full_sha256": full["sha256"],
        "native_display": [{"prediction_index": i, "gt_index": g,
                            "best_gt_aabb_iou": float(native_overlaps[i, g])}
                           for i, g in zip(native_indices, native_gt_indices)],
        "ptr": {"prediction_index": ptr_index, "gt_index": ptr_gt, "aabb_iou": ptr_iou,
                "score": float(full["scores"][ptr_index])},
        "var": {"prediction_index": var_index, "gt_index": var_gt, "aabb_iou": var_iou,
                "score": float(full["scores"][var_index])},
        "same_scene": True, "distinct_recovery_targets": True,
        "prediction_geometry_edited": False, "new_model_inference": False,
        "display_selection": "GT-assisted qualitative selection; coordinates and scores unchanged",
        "audit_overlay": "dashed boxes are GT and appear only in the audit image",
    }
    (OUTPUT / "source_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    (OUTPUT / "使用说明.txt").write_text(
        f"同一真实ScanNet场景{SCENE}的在线检测读出。蓝色为原生记录，橙色为PTR恢复，金黄色为VAR恢复。\n"
        f"橙色PTR框与对应GT的三维AABB IoU为{ptr_iou:.3f}；金黄色VAR框为{var_iou:.3f}。"
        "两个恢复框对应不同目标。预测坐标、尺寸、朝向和分数均未修改。\n"
        "white/transparent文件用于论文流程图；audit文件额外叠加同色虚线GT，仅用于核对框的准确性。\n",
        encoding="utf-8",
    )
    print(json.dumps({"output": str(OUTPUT), "ptr_iou": ptr_iou, "var_iou": var_iou}))


if __name__ == "__main__":
    main()
